"""
gemini.py — Google Generative AI provider.

Owns everything google-genai specific: client construction, the proto message
dialect, retry/backoff on transient errors, and the `thinking`/`temperature`
generation settings. Behaviour is carried over unchanged from the original
`loop.py` implementation.

Gemini's message dialect differs from the neutral format in two ways worth
noting:
  - the assistant role is called "model", not "assistant";
  - tool results are sent back as a *user* turn carrying FunctionResponse
    parts, and consecutive results are batched into a single turn.

On the free tier (`GEMINI_TIER=free`, the default) quotas are per model, so a
daily-quota 429 switches to the next model of the fallback chain
(`GEMINI_FALLBACK_MODELS`, default from the catalog: 3.5 Flash-Lite → 3.1
Flash-Lite) and says so; a per-minute 429 backs off; a per-minute *token* cap
smaller than one request (free-tier Gemma) stops at once instead of retrying.

One setting is easy to misread. `include_thoughts=False` — set on every call
since this provider was written — suppresses *returning* the thought trace; it
does not stop the model thinking. Reasoning has therefore always been on here,
at whatever dynamic budget the model chooses. `GEMINI_THINKING_LEVEL` pins it
instead, and defaults to empty so that behaviour is unchanged unless asked for.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import time
from typing import Any, Callable

import httpx
from google import genai
from google.genai import types
from google.genai.errors import ClientError, ServerError

from ..config import (
    GEMINI_MODEL,
    GEMINI_THINKING_LEVEL,
    MODEL_OVERLOADED_RETRY_DELAY_S,
    MODEL_REQUEST_TIMEOUT_MS,
    MODEL_RETRY_ATTEMPTS,
)
from .. import ratelimit as rl
from ..errors import classify_error
from .base import ModelResponse, ToolCall

logger = logging.getLogger(__name__)

# The smallest prompt CONDUCTOR can send (system prompt and tool schemas). A
# per-minute input-token cap below it can never be met.
_MIN_PROMPT_TOKENS = 25_000


class GeminiProvider:
    """LLMProvider backed by the Google Generative AI API."""

    name = "google"

    def __init__(self, model: str | None = None,
                 thinking_level: str | None = None) -> None:
        # Live environment, not the import-time constant: a model saved in the
        # settings applies to the next message, not after a restart.
        self.model = model or os.environ.get("GEMINI_MODEL") or GEMINI_MODEL
        self.tier = (os.environ.get("GEMINI_TIER") or "free").strip().lower()
        # Read from the live environment so the settings panel can change it
        # mid-session (see providers.reset_providers). Empty is a real choice —
        # it means "leave the model on its own budget" — so `or` would be wrong.
        if thinking_level is None:
            raw = os.environ.get("GEMINI_THINKING_LEVEL")
            thinking_level = GEMINI_THINKING_LEVEL if raw is None else raw
        self.thinking_level = thinking_level.strip().lower()
        self._client: genai.Client | None = None

    def describe(self) -> dict[str, str]:
        """Settings worth recording in the session log.

        "model default" rather than "" or "off": Gemini reasons either way,
        and a blank field here would read as though it did not.
        """
        return {"reasoning": self.thinking_level or "model default"}

    # -- lifecycle ---------------------------------------------------------

    def preflight(self) -> None:
        """Verify an API key is available before the first request."""
        if not os.environ.get("GEMINI_API_KEY", "").strip():
            raise RuntimeError(
                "GEMINI_API_KEY is not set. "
                "Please restart the app and enter your key on the setup screen."
            )

    def _get_client(self) -> genai.Client:
        """Return the shared client, creating it on first use."""
        if self._client is None:
            self.preflight()
            self._client = genai.Client(
                api_key=os.environ.get("GEMINI_API_KEY", ""),
                http_options=types.HttpOptions(timeout=MODEL_REQUEST_TIMEOUT_MS),
            )
        return self._client

    # -- main entry point --------------------------------------------------

    def chat(
        self,
        messages: list[dict],
        tools: list,
        system: str,
        on_event: Callable[[str, dict], None] | None = None,
    ) -> ModelResponse:
        contents = self._to_contents(messages)
        try:
            response = self._generate_with_retry(
                contents, self._config(tools, system, self.thinking_level),
                on_event=on_event,
            )
        except ClientError as exc:
            # A model that predates thinking levels rejects the field outright.
            # That is a fact about the model, not about this request, so fall
            # back to its own budget rather than failing the turn over a knob.
            if not (self.thinking_level and _rejected_thinking_level(exc)):
                raise
            logger.info("Model %s rejected thinking_level=%s; retrying without it.",
                        self.model, self.thinking_level)
            response = self._generate_with_retry(
                contents, self._config(tools, system, ""), on_event=on_event,
            )
        return self._to_model_response(response)

    def fallback_chain(self) -> list[str]:
        """Models to try, in order, when the current one's daily free quota is
        spent: GEMINI_FALLBACK_MODELS, else the catalog's chain. Free tier only."""
        if self.tier != "free":
            return []
        raw = (os.environ.get("GEMINI_FALLBACK_MODELS") or "").strip()
        if raw:
            return [m.strip() for m in raw.split(",") if m.strip()]
        try:
            from .. import catalog
            return list(catalog.option("gemini_free").get("fallback_chain") or [])
        except Exception:  # noqa: BLE001 — a missing catalog must not break Gemini
            return []

    def _next_in_chain(self, tried: set[str]) -> str | None:
        chain = self.fallback_chain()
        if self.model in chain:
            chain = chain[chain.index(self.model) + 1:]
        for model in chain:
            if model not in tried:
                return model
        return None

    def _config(self, tools: list, system: str, thinking_level: str):
        """One request config, with thinking pinned only when asked for.

        `include_thoughts=False` suppresses *returning* the thought trace; it
        does not stop the model thinking. Whether it thinks, and how hard, is
        `thinking_level` — left unset, the model uses its own dynamic budget.
        """
        thinking = types.ThinkingConfig(include_thoughts=False)
        if thinking_level:
            thinking.thinking_level = thinking_level.upper()
        return types.GenerateContentConfig(
            # None, not [], for a tool-less call (the suggestion agent).
            tools=tools or None,
            system_instruction=system,
            temperature=0,
            thinking_config=thinking,
        )

    # -- retry -------------------------------------------------------------

    def _generate_with_retry(
        self,
        contents: list,
        config: Any,
        max_retries: int | None = None,
        on_event: Callable[[str, dict], None] | None = None,
    ):
        """
        Call generate_content with automatic retry on:
          - 429 ClientError (per-minute rate limit)
          - 503 ServerError (model overloaded / high demand)
        Raises a user-friendly RuntimeError on daily quota exhaustion.
        """
        retries = MODEL_RETRY_ATTEMPTS if max_retries is None else max_retries
        tried: set[str] = set()

        attempt = -1
        while attempt < retries:
            attempt += 1
            try:
                return self._get_client().models.generate_content(
                    model=self.model,
                    contents=contents,
                    config=config,
                )
            except ServerError as exc:
                classification = classify_error(str(exc), "ServerError")
                if attempt < retries and classification.is_retryable:
                    wait = classification.suggested_wait_s or MODEL_OVERLOADED_RETRY_DELAY_S
                    self._announce_retry(classification, attempt, retries, wait,
                                         "Model overloaded (503)", on_event)
                    time.sleep(wait)
                else:
                    raise RuntimeError(classification.user_message) from exc

            except ClientError as exc:
                # google-genai raises ClientError for all 4xx. 429 is a rate
                # limit or quota; 404 a retired model; the rest are permanent.
                if exc.code not in (404, 429):
                    raise
                failure = rl.classify_gemini(str(exc), exc.code)
                if failure.kind in (rl.PER_DAY, rl.MODEL_GONE):
                    replacement = self._next_in_chain(tried)
                    if replacement:
                        tried.add(self.model)
                        old, self.model = self.model, replacement
                        os.environ["GEMINI_MODEL"] = replacement
                        reason = ("its daily free quota is used up" if failure.kind == rl.PER_DAY
                                  else "Google no longer serves it")
                        logger.warning("Gemini %s: %s; switching to %s.", old, reason, replacement)
                        if on_event:
                            on_event("model_switch", {"from": old, "to": replacement,
                                                      "reason": f"{old}: {reason}"})
                        attempt -= 1          # a switch is not a retry
                        continue
                    if failure.kind == rl.MODEL_GONE:
                        raise
                    raise RuntimeError(self._daily_quota_message(failure)) from exc
                if failure.kind != rl.PER_MINUTE:
                    raise
                if failure.metric == "input_tokens" and failure.limit_value \
                        and failure.limit_value < _MIN_PROMPT_TOKENS:
                    # One request is bigger than the whole per-minute token
                    # budget (free-tier Gemma): waiting can never help.
                    raise RuntimeError(
                        f"⚠️ **`{self.model}` accepts at most {failure.limit_value:,} input tokens a minute "
                        f"on this tier** — less than one CONDUCTOR request (system prompt and tool schemas, "
                        f"~{_MIN_PROMPT_TOKENS:,} tokens). Choose a Flash or Flash-Lite model.") from exc
                if attempt < retries:
                    wait = int(rl.backoff(attempt, failure.retry_after_s, base=10))
                    classification = classify_error(str(exc), "ResourceExhausted")
                    self._announce_retry(classification, attempt, retries, wait,
                                         "Waiting for the per-minute rate limit", on_event)
                    time.sleep(wait)
                else:
                    raise RuntimeError(classify_error(str(exc), "ResourceExhausted").user_message) from exc

            except httpx.TimeoutException as exc:
                classification = classify_error(str(exc), "TimeoutException")
                if attempt < retries and classification.is_retryable:
                    wait = classification.suggested_wait_s or 30
                    self._announce_retry(classification, attempt, retries, wait,
                                         "Request timed out", on_event)
                    time.sleep(wait)
                else:
                    raise RuntimeError(classification.user_message) from exc

        raise RuntimeError("Unexpected exit from retry loop")  # pragma: no cover

    def _daily_quota_message(self, failure) -> str:
        reset = rl.describe_reset(failure.resets_at)
        chain = self.fallback_chain()
        tried = f" (the fallback chain {' → '.join(chain)} is spent too)" if len(chain) > 1 else ""
        return (f"⚠️ **The Gemini free tier's daily quota for `{self.model}` is used up{tried}.**\n\n"
                f"Free-tier quotas are per model and reset at midnight Pacific time"
                f"{' — ' + reset if reset else ''}.\n\n{rl.alternatives_text('gemini_free')}")

    @staticmethod
    def _announce_retry(classification, attempt, retries, wait, reason, on_event) -> None:
        logger.warning(
            "%s (attempt %d/%d). Retrying in %ds…",
            classification.status, attempt + 1, retries, wait,
        )
        if on_event:
            on_event("retry", {
                "attempt": attempt + 1, "total": retries,
                "wait_s": wait, "reason": reason,
            })

    # -- message conversion ------------------------------------------------

    def _to_contents(self, messages: list[dict]) -> list:
        """Neutral messages → genai Content list."""
        contents: list = []
        pending_tool_parts: list = []

        def flush_tools() -> None:
            if pending_tool_parts:
                contents.append(types.Content(role="user", parts=list(pending_tool_parts)))
                pending_tool_parts.clear()

        for msg in messages:
            role = msg.get("role")

            if role == "tool":
                # Consecutive tool results batch into one user turn.
                pending_tool_parts.append(
                    types.Part(
                        function_response=types.FunctionResponse(
                            name=msg["name"],
                            id=msg.get("id"),
                            response={"result": _sanitize_for_proto(msg.get("content"))},
                        )
                    )
                )
                continue

            flush_tools()

            if role == "user":
                contents.append(
                    types.Content(role="user", parts=[types.Part(text=msg.get("content") or "")])
                )
            elif role == "assistant":
                parts: list = []
                if msg.get("content"):
                    parts.append(
                        types.Part(
                            text=msg["content"],
                            thought_signature=_decode_signature(msg.get("signature")),
                        )
                    )
                for call in msg.get("tool_calls") or []:
                    # thought_signature is mandatory here: Gemini 3.x rejects
                    # replayed functionCall parts that arrive without it.
                    parts.append(
                        types.Part(
                            function_call=types.FunctionCall(
                                name=call["name"],
                                id=call.get("id"),
                                args=call.get("args") or {},
                            ),
                            thought_signature=_decode_signature(call.get("signature")),
                        )
                    )
                if parts:
                    contents.append(types.Content(role="model", parts=parts))

        flush_tools()
        return contents

    @staticmethod
    def _to_model_response(response) -> ModelResponse:
        """genai response → neutral ModelResponse."""
        parts = (response.candidates[0].content.parts or []) if response.candidates else []

        text_parts = [
            part for part in parts
            if part.text and not getattr(part, "thought", False)
        ]
        text = "\n".join(part.text for part in text_parts).strip()

        # Signatures must survive the round-trip or Gemini 3.x rejects the
        # replayed history with 400 INVALID_ARGUMENT.
        text_signature = next(
            (_encode_signature(p.thought_signature) for p in text_parts
             if getattr(p, "thought_signature", None)),
            None,
        )

        tool_calls = [
            ToolCall(
                name=part.function_call.name,
                # args is a proto MapComposite; flatten to plain Python.
                args=_sanitize_for_proto(dict(part.function_call.args))
                if part.function_call.args else {},
                signature=_encode_signature(getattr(part, "thought_signature", None)),
                id=getattr(part.function_call, "id", None),
            )
            for part in parts
            if part.function_call and part.function_call.name
        ]

        return ModelResponse(
            text=text, tool_calls=tool_calls, raw=response, signature=text_signature
        )


# ---------------------------------------------------------------------------
# Proto helpers
# ---------------------------------------------------------------------------


def _rejected_thinking_level(exc: ClientError) -> bool:
    """Whether this 400 is the model refusing `thinking_level`.

    Both `message` and the rendered exception are searched: `str()` on a genai
    APIError renders the raw response body, while `message` is the extracted
    text, and which of the two carries the wording varies by error shape.
    """
    if getattr(exc, "code", None) != 400:
        return False
    haystack = f"{getattr(exc, 'message', '') or ''} {exc}".lower()
    return "thinking" in haystack


def _encode_signature(raw: bytes | None) -> str | None:
    """Gemini thought signature (bytes) → base64 str, so history stays JSON-safe."""
    if not raw:
        return None
    if isinstance(raw, str):  # already encoded by a previous round-trip
        return raw
    return base64.b64encode(raw).decode("ascii")


def _decode_signature(encoded: str | None) -> bytes | None:
    """base64 str → the bytes the API expects back. Never fatal: a corrupt
    signature should degrade model quality, not crash the turn."""
    if not encoded:
        return None
    if isinstance(encoded, bytes):
        return encoded
    try:
        # validate=True so garbage raises instead of being silently stripped to b''.
        decoded = base64.b64decode(encoded, validate=True)
    except (ValueError, TypeError):
        logger.warning("Discarding unreadable thought_signature.")
        return None
    return decoded or None


def _sanitize_for_proto(obj):  # noqa: ANN001
    """
    Recursively convert an object to a JSON-safe structure that the
    Gemini proto FunctionResponse can accept.

    - None → empty string (proto Struct doesn't support null)
    - Non-JSON-serializable types → str(obj)
    - Floats: kept as-is (proto Struct supports float)
    """
    if obj is None:
        return ""
    if isinstance(obj, dict):
        return {str(k): _sanitize_for_proto(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_proto(v) for v in obj]
    if isinstance(obj, (bool, int, float, str)):
        return obj
    # Handle proto RepeatedComposite and other list-like iterables
    # (they are not list/tuple but are iterable and have __iter__)
    if hasattr(obj, "__iter__"):
        return [_sanitize_for_proto(v) for v in obj]
    # Fallback for anything else (e.g. pandas objects)
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        return str(obj)
