"""
anthropic.py — LLMProvider backed by the Anthropic Messages API (Claude).

Uses the official `anthropic` SDK rather than raw HTTP, unlike the OpenAI and
Ollama providers here. Those two exist partly to serve *compatible* endpoints —
vLLM, OpenRouter, a local llama.cpp — where owning the wire format is the point.
Nothing else speaks the Anthropic dialect, so the SDK's retry policy, typed
exceptions and content-block types are worth more than the symmetry.

Two things the other providers do not have to deal with:

**Thinking blocks must survive the round trip.** When a turn continues after a
tool call, the assistant content is replayed to the model, and a thinking block
that produced a tool call has to come back unchanged or the model loses the
reasoning behind its own call. The neutral history in `base.py` carries only
text and tool calls, so the block is stashed on the ToolCall's `signature`
field — the same slot Gemini uses for `thought_signature`, for the same reason.

**Two incompatible thinking APIs.** The current models take adaptive thinking
with `output_config.effort`; Haiku 4.5 and the other 4.5-generation models
predate both and take `{"type": "enabled", "budget_tokens": N}`, rejecting
`output_config` outright. Sending the wrong shape is a 400 on every request, so
the model picks the shape and `ANTHROPIC_EFFORT` stays the single knob in both
worlds. See `_thinking_payload`.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Callable

from ..config import (
    ANTHROPIC_BASE_URL,
    ANTHROPIC_EFFORT,
    ANTHROPIC_MAX_TOKENS,
    ANTHROPIC_MODEL,
    ANTHROPIC_THINKING,
    ANTHROPIC_TIMEOUT_S,
)
from .base import ModelResponse, ToolCall
from .schema import to_json_schema_tools

logger = logging.getLogger(__name__)

# Effort levels the API accepts. A value outside this set is dropped rather
# than sent: an unknown level is a 400, and losing the setting is better than
# losing the turn.
_EFFORTS = ("low", "medium", "high", "xhigh", "max")

# Thinking cannot be disabled above this level — the API rejects
# `{"type": "disabled"}` at `xhigh` and `max`.
_DISABLE_ALLOWED_UPTO = ("low", "medium", "high", "")

# Models that predate adaptive thinking. They take the older
# `{"type": "enabled", "budget_tokens": N}` shape and reject `output_config`
# outright, so sending the modern pair to one is a 400 on every request.
#
# Matched by prefix rather than listed exactly, because a dated snapshot
# (`claude-haiku-4-5-20251001`) is the same model with a longer id.
_LEGACY_THINKING_PREFIXES = ("claude-haiku-4-5", "claude-sonnet-4-5", "claude-opus-4-5")

# Budget for the legacy shape, standing in for the effort ladder those models
# do not have. Must be below `max_tokens`, and at least 1024.
_LEGACY_BUDGETS = {
    "low": 2048,
    "medium": 4096,
    "high": 8192,
    "xhigh": 12288,
    "max": 16384,
}


def _anthropic_tools(tools: list) -> list[dict]:
    """Convert the shared genai tool definitions into Anthropic's shape.

    Anthropic wants `{name, description, input_schema}` flat, where the
    OpenAI-style converter produces `{type: "function", function: {...}}`.
    Reusing that converter keeps `tool_schemas.TOOLS` the single source of
    truth — only the envelope differs.
    """
    converted = []
    for entry in to_json_schema_tools(tools):
        fn = entry["function"]
        converted.append({
            "name": fn["name"],
            "description": fn.get("description", ""),
            "input_schema": fn.get("parameters") or {"type": "object", "properties": {}},
        })
    return converted


def _stringify(result: Any) -> str:
    """A tool result as the API wants it: text.

    Tool results here are arbitrary Python — violation tables, dispatch
    records, numpy-derived floats. `default=str` rather than a failure, because
    an unserialisable corner of a payload should cost that corner, not the turn.
    """
    if isinstance(result, str):
        return result
    try:
        return json.dumps(result, ensure_ascii=False, default=str)
    except (TypeError, ValueError):  # pragma: no cover — default=str is total
        return str(result)


class AnthropicProvider:
    """LLMProvider backed by the Anthropic Messages API."""

    name = "anthropic"

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        # Read from the live environment so the settings panel can change any
        # of these mid-session (see providers.reset_providers).
        self.model = model or os.environ.get("ANTHROPIC_MODEL") or ANTHROPIC_MODEL
        self.base_url = (
            base_url or os.environ.get("ANTHROPIC_BASE_URL") or ANTHROPIC_BASE_URL
        ).rstrip("/")
        self._api_key = api_key
        self.timeout_s = float(
            os.environ.get("ANTHROPIC_TIMEOUT_S") or ANTHROPIC_TIMEOUT_S
        )
        self.max_tokens = int(
            os.environ.get("ANTHROPIC_MAX_TOKENS") or ANTHROPIC_MAX_TOKENS
        )
        # Empty string is a real choice — it means "send nothing", leaving the
        # API on its own default. `or` would turn that back into the default.
        effort = os.environ.get("ANTHROPIC_EFFORT")
        self.effort = (ANTHROPIC_EFFORT if effort is None else effort).strip().lower()
        thinking = os.environ.get("ANTHROPIC_THINKING")
        self.thinking = (
            ANTHROPIC_THINKING if thinking is None
            else thinking.strip().lower() not in ("0", "false", "no")
        )
        self._client = None

    def describe(self) -> dict[str, str]:
        """Settings worth recording in the session log.

        Effort and thinking together decide how much the model deliberates, and
        two turns from the same model at different efforts are not comparable —
        which is most of what the graders in `evaluation/` exist to do.
        """
        if not self.thinking:
            reasoning = "thinking off"
        elif self.legacy_thinking:
            # The budget, not the effort name: on a legacy model `medium` is a
            # token count this provider chose, and a log that recorded only the
            # label could not be compared against anything.
            budget = _LEGACY_BUDGETS.get(self.effort, _LEGACY_BUDGETS["medium"])
            budget = min(budget, max(1024, self.max_tokens // 2))
            reasoning = f"budget {budget}"
        else:
            reasoning = self.effort or "default"
        return {"reasoning": reasoning, "endpoint": "/v1/messages"}

    # -- lifecycle ---------------------------------------------------------

    @property
    def api_key(self) -> str:
        if self._api_key is not None:
            return self._api_key
        return os.environ.get("ANTHROPIC_API_KEY", "")

    def preflight(self) -> None:
        """Verify the SDK is installed and a key is available.

        As cheap as the other providers' — a missing dependency or key is the
        only failure worth catching before a round trip, and everything else
        produces a better message from the response that actually failed.
        """
        try:
            import anthropic  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "⚠️ **The `anthropic` package is not installed.**\n\n"
                "Run `pip install anthropic`, then restart the app."
            ) from exc

        if not self.api_key.strip():
            raise RuntimeError(
                "⚠️ **ANTHROPIC_API_KEY is not set.**\n\n"
                "Create a key at https://console.anthropic.com/settings/keys, "
                "then restart the app and enter it on the setup screen."
            )

    @property
    def client(self):
        """The SDK client, built once per provider instance."""
        if self._client is None:
            import anthropic

            kwargs: dict[str, Any] = {
                "api_key": self.api_key,
                "timeout": self.timeout_s,
            }
            # Only when overridden. Passing the default explicitly would
            # override an `ANTHROPIC_BASE_URL` the SDK would otherwise read.
            if self.base_url and self.base_url != ANTHROPIC_BASE_URL.rstrip("/"):
                kwargs["base_url"] = self.base_url
            self._client = anthropic.Anthropic(**kwargs)
        return self._client

    # -- main entry point --------------------------------------------------

    def chat(
        self,
        messages: list[dict],
        tools: list,
        system: str,
        on_event: Callable[[str, dict], None] | None = None,
    ) -> ModelResponse:
        """Send one request and return the normalised reply."""
        import anthropic

        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system,
            "messages": self._to_anthropic_messages(messages),
            "tools": _anthropic_tools(tools),
        }

        payload.update(self._thinking_payload())

        try:
            response = self.client.messages.create(**payload)
        except anthropic.APIStatusError as exc:
            raise RuntimeError(self._describe_error(exc)) from exc
        except anthropic.APIConnectionError as exc:
            raise RuntimeError(
                "⚠️ **Could not reach the Anthropic API.**\n\n"
                f"{exc}\n\nCheck the network connection and try again."
            ) from exc

        return self._to_model_response(response)

    @property
    def legacy_thinking(self) -> bool:
        """Whether this model predates adaptive thinking and `effort`."""
        return self.model.startswith(_LEGACY_THINKING_PREFIXES)

    def _thinking_payload(self) -> dict[str, Any]:
        """How much to think, in the shape this model accepts.

        Two incompatible APIs, and sending the wrong one is a 400 on every
        request rather than a degraded answer — so the model decides the
        shape, not the caller. `ANTHROPIC_EFFORT` stays the single knob in
        both worlds; on a legacy model it maps onto a token budget.
        """
        if self.legacy_thinking:
            if not self.thinking:
                return {}
            budget = _LEGACY_BUDGETS.get(self.effort, _LEGACY_BUDGETS["medium"])
            # The budget must leave room for an answer. Halving rather than
            # clamping to `max_tokens - 1`: a model that spends its entire
            # output allowance thinking has nothing left to say.
            budget = min(budget, max(1024, self.max_tokens // 2))
            return {"thinking": {"type": "enabled", "budget_tokens": budget}}

        out: dict[str, Any] = {}
        if self.thinking:
            # Adaptive: the model decides when and how much to think, and
            # `effort` sets the depth. `budget_tokens` is removed on these
            # models and returns a 400.
            out["thinking"] = {"type": "adaptive"}
        elif self.effort in _DISABLE_ALLOWED_UPTO:
            out["thinking"] = {"type": "disabled"}
        # Above `high`, disabling is rejected — so the request simply omits
        # `thinking` and takes the model's default rather than failing.

        if self.effort in _EFFORTS:
            out["output_config"] = {"effort": self.effort}
        elif self.effort:
            logger.warning(
                "Ignoring unknown ANTHROPIC_EFFORT %r; expected one of %s.",
                self.effort, ", ".join(_EFFORTS),
            )
        return out

    # -- request shaping ---------------------------------------------------

    def _to_anthropic_messages(self, messages: list[dict]) -> list[dict]:
        """Neutral history → Anthropic `messages`.

        Two shape rules the other providers do not share. Tool results are
        `user` messages carrying `tool_result` blocks, not a role of their own;
        and consecutive results must be merged into one message, because the
        API rejects two user messages in a row.
        """
        out: list[dict] = []

        for message in messages:
            role = message.get("role")

            if role == "user":
                out.append({"role": "user", "content": message.get("content", "")})

            elif role == "assistant":
                content = self._assistant_content(message)
                # An assistant turn with neither text nor tool calls is not a
                # turn the API will accept; dropping it is safe because it
                # carried nothing the model needs to see.
                if content:
                    out.append({"role": "assistant", "content": content})

            elif role == "tool":
                block = {
                    "type": "tool_result",
                    "tool_use_id": message.get("id") or message.get("name", ""),
                    "content": _stringify(message.get("content")),
                }
                # Parallel tool calls come back as several `tool` messages in a
                # row and must arrive as one user message — splitting them is
                # rejected, and elsewhere trains models to stop calling in
                # parallel at all.
                if out and out[-1]["role"] == "user" and isinstance(out[-1]["content"], list):
                    out[-1]["content"].append(block)
                else:
                    out.append({"role": "user", "content": [block]})

        return out

    @staticmethod
    def _assistant_content(message: dict) -> list[dict]:
        """Rebuild an assistant turn's content blocks from neutral history.

        Thinking blocks come first when present. A tool call made after
        thinking has to be replayed with the thinking that produced it, or the
        model sees its own call with no reasoning behind it; the block was
        stashed on the ToolCall's `signature` slot on the way out.
        """
        content: list[dict] = []
        seen_thinking: set[str] = set()

        for raw in message.get("tool_calls") or []:
            stashed = raw.get("signature")
            if not stashed or stashed in seen_thinking:
                continue
            seen_thinking.add(stashed)
            try:
                content.append(json.loads(stashed))
            except (TypeError, ValueError):
                # A signature that will not parse is not worth failing a turn
                # over: the call itself still replays, only the reasoning is
                # lost, which is what happens on every other backend anyway.
                logger.debug("Could not restore a thinking block; continuing without it.")

        text = message.get("content") or ""
        if text:
            content.append({"type": "text", "text": text})

        for raw in message.get("tool_calls") or []:
            content.append({
                "type": "tool_use",
                "id": raw.get("id") or raw["name"],
                "name": raw["name"],
                "input": raw.get("args") or {},
            })

        return content

    # -- response shaping --------------------------------------------------

    @staticmethod
    def _to_model_response(response) -> ModelResponse:
        """Anthropic response → `ModelResponse`.

        A refusal is surfaced as text rather than raised: it is an answer the
        operator should see, and the turn ends cleanly with no tool calls.
        """
        if getattr(response, "stop_reason", None) == "refusal":
            details = getattr(response, "stop_details", None)
            category = getattr(details, "category", None) or "unspecified"
            return ModelResponse(
                text=(
                    "⚠️ The model declined to answer this request "
                    f"(category: {category}). Rephrasing, or asking for the "
                    "underlying analysis directly, usually gets past it."
                ),
                raw=response,
            )

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        # Serialised so it can ride in the neutral history, which is plain
        # JSON. Attached to every tool call in the turn; the replay path
        # de-duplicates.
        thinking_blob: str | None = None

        for block in response.content or []:
            kind = getattr(block, "type", None)
            if kind == "text":
                text_parts.append(block.text)
            elif kind == "thinking":
                try:
                    thinking_blob = json.dumps(block.model_dump(exclude_none=True))
                except Exception:  # noqa: BLE001
                    logger.debug("Could not serialise a thinking block.", exc_info=True)
            elif kind == "tool_use":
                tool_calls.append(ToolCall(
                    name=block.name,
                    args=dict(block.input or {}),
                    id=block.id,
                ))

        if thinking_blob:
            tool_calls = [
                ToolCall(name=tc.name, args=tc.args, id=tc.id, signature=thinking_blob)
                for tc in tool_calls
            ]

        return ModelResponse(
            text="\n".join(p for p in text_parts if p).strip(),
            tool_calls=tool_calls,
            raw=response,
        )

    # -- errors ------------------------------------------------------------

    def _describe_error(self, exc) -> str:
        """A user-facing message for an API status error.

        Most-specific first, matching the SDK's typed exception hierarchy. The
        loop only ever sees a RuntimeError carrying this text, so the message
        has to say what to do rather than what went wrong.
        """
        import anthropic

        if isinstance(exc, anthropic.AuthenticationError):
            return (
                "⚠️ **Anthropic rejected the API key.**\n\n"
                "Check `ANTHROPIC_API_KEY` — the key may be revoked, or belong "
                "to a different organisation."
            )
        if isinstance(exc, anthropic.NotFoundError):
            return (
                f"⚠️ **Model `{self.model}` was not found.**\n\n"
                "Check `ANTHROPIC_MODEL`. The account may not have access to "
                "that model."
            )
        if isinstance(exc, anthropic.RateLimitError):
            return (
                "⚠️ **Anthropic rate limit reached.**\n\n"
                "The SDK already retried with backoff. Wait a moment and ask "
                "again, or lower `ANTHROPIC_EFFORT` to spend fewer tokens."
            )
        if isinstance(exc, anthropic.BadRequestError):
            return (
                "⚠️ **Anthropic rejected the request.**\n\n"
                f"{getattr(exc, 'message', str(exc))}\n\n"
                "If this mentions `thinking` or `effort`, check "
                "`ANTHROPIC_EFFORT` and `ANTHROPIC_THINKING` — thinking cannot "
                "be disabled above effort `high`."
            )
        status = getattr(exc, "status_code", "?")
        return (
            f"⚠️ **Anthropic API error ({status}).**\n\n"
            f"{getattr(exc, 'message', str(exc))}"
        )
