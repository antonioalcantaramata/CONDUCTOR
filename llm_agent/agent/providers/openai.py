"""
openai.py — OpenAI provider (and anything else speaking its dialect).

Talks to `/v1/chat/completions` over plain httpx rather than through the
`openai` SDK. httpx is already a dependency and `ollama.py` already works this
way, so the alternative was a new package for one endpoint. The side benefit is
that `OPENAI_BASE_URL` makes this the provider for every OpenAI-compatible
server too — Azure OpenAI, OpenRouter, Groq, a local vLLM — since only the base
changes.

`schema.to_json_schema_tools` already emits exactly the tool shape this API
wants, so tool definitions need no conversion here at all.

Two things this has to get right:

*Tool calls are paired by id, not by order.* OpenAI rejects an assistant
message whose `tool_calls` lack ids, and a `tool` message whose `tool_call_id`
matches nothing before it. The neutral history only carries an id when the
provider that produced it happened to set one — Ollama never does — so
switching backends mid-conversation would otherwise 400 on the replayed
history. Ids are therefore synthesised deterministically at conversion time.

*A quota failure is not a rate limit.* Both arrive as HTTP 429. Retrying
`rate_limit_exceeded` is right; retrying `insufficient_quota` burns the
configured ten attempts at ninety seconds each — a quarter of an hour of
silence — to re-learn that the account has no credit. They are separated below.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Callable

import httpx

from ..config import (
    MODEL_OVERLOADED_RETRY_DELAY_S,
    MODEL_RETRY_ATTEMPTS,
    OPENAI_API,
    OPENAI_BASE_URL,
    OPENAI_MODEL,
    OPENAI_REASONING_EFFORT,
    OPENAI_TIMEOUT_S,
)
from .base import ModelResponse, ToolCall
from .schema import to_json_schema_tools

logger = logging.getLogger(__name__)

# Request parameters that only some models accept. When one is rejected the
# request is retried without it, rather than failing a turn over a knob.
_OPTIONAL_PARAMS = ("reasoning_effort", "temperature")

# Error codes that no amount of retrying will clear. Everything else on a 429
# or 5xx is treated as transient.
_PERMANENT_CODES = {
    "insufficient_quota",
    "invalid_api_key",
    "model_not_found",
    "context_length_exceeded",
    "billing_hard_limit_reached",
}


def _responses_tools(tools: list) -> list[dict]:
    """Chat-completions tool shape → the flat shape /v1/responses wants.

    Chat/completions nests the declaration under a "function" key; Responses
    puts name, description and parameters at the top level of the item. Same
    content, one level of wrapping apart.
    """
    flat: list[dict] = []
    for tool in to_json_schema_tools(tools):
        fn = tool.get("function") or {}
        flat.append({
            "type": "function",
            "name": fn.get("name", ""),
            "description": fn.get("description", ""),
            "parameters": fn.get("parameters") or {"type": "object", "properties": {}},
        })
    return flat


def _encode_reasoning(items: list[dict]) -> str | None:
    """Reasoning items → a JSON string the neutral history can carry.

    Stored in the message's opaque `signature` slot, which `base.py` documents
    as a provider round-trip token. Kept as text so history stays
    JSON-serialisable for `st.session_state` and the session log.
    """
    if not items:
        return None
    try:
        return json.dumps(items, default=str)
    except (TypeError, ValueError):  # pragma: no cover — default=str covers it
        logger.warning("Discarding unserialisable reasoning items.")
        return None


def _decode_reasoning(encoded: str | None) -> list[dict]:
    """The inverse. Never fatal: a corrupt token costs reasoning context, not
    the turn, so it degrades to replaying nothing."""
    if not encoded:
        return []
    try:
        items = json.loads(encoded)
    except (TypeError, ValueError, json.JSONDecodeError):
        logger.warning("Discarding unreadable reasoning items.")
        return []
    return [i for i in items if isinstance(i, dict)] if isinstance(items, list) else []


def probe(base_url: str | None = None, api_key: str | None = None) -> dict:
    """
    Inspect an OpenAI-compatible endpoint without committing to a model.

    Used by the setup screen to show what the key can actually reach. Returns::

        {"reachable": bool, "error": str, "models": [str, ...]}

    `models` is best-effort: some compatible servers do not implement
    `/v1/models`, which is not a reason to refuse to run.
    """
    base = (base_url or os.environ.get("OPENAI_BASE_URL") or OPENAI_BASE_URL).rstrip("/")
    key = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY", "")
    out: dict[str, Any] = {"reachable": False, "error": "", "models": []}

    if not key.strip():
        out["error"] = "No API key set."
        return out

    try:
        resp = httpx.get(
            f"{base}/models",
            headers={"Authorization": f"Bearer {key}"},
            timeout=15,
        )
    except Exception as exc:  # noqa: BLE001
        out["error"] = f"Cannot reach {base}: {exc}"
        return out

    if resp.status_code == 401:
        out["error"] = "The API key was rejected (HTTP 401)."
        return out
    if resp.status_code != 200:
        out["error"] = f"HTTP {resp.status_code} from {base}/models."
        return out

    out["reachable"] = True
    try:
        out["models"] = sorted(
            str(m.get("id", "")) for m in resp.json().get("data", []) if m.get("id")
        )
    except Exception:  # noqa: BLE001
        logger.debug("Could not parse the model list from %s.", base)
    return out


class OpenAIProvider:
    """LLMProvider backed by the OpenAI chat-completions API."""

    name = "openai"

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        # Read from the live environment so the settings panel can change any of
        # these mid-session (see providers.reset_providers).
        self.model = model or os.environ.get("OPENAI_MODEL") or OPENAI_MODEL
        self.base_url = (
            base_url or os.environ.get("OPENAI_BASE_URL") or OPENAI_BASE_URL
        ).rstrip("/")
        self._api_key = api_key
        self.timeout_s = float(os.environ.get("OPENAI_TIMEOUT_S") or OPENAI_TIMEOUT_S)
        # Empty string is a real choice — it means "send nothing", which is what
        # a non-reasoning model or a compatible server without the parameter
        # needs. `or` would silently turn that back into the default.
        effort = os.environ.get("OPENAI_REASONING_EFFORT")
        self.reasoning_effort = (
            OPENAI_REASONING_EFFORT if effort is None else effort
        ).strip().lower()
        self.api = (os.environ.get("OPENAI_API") or OPENAI_API).strip().lower()

    @property
    def uses_responses(self) -> bool:
        """Whether this turn goes to /v1/responses rather than chat/completions.

        Under `auto`, reasoning is what decides. A model that reasons cannot
        also call functions on chat/completions, and CONDUCTOR sends tools on
        every turn — so wanting reasoning at all means wanting /v1/responses.
        A custom base URL stays on chat/completions, which is the endpoint
        OpenAI-compatible servers actually implement.
        """
        if self.api == "responses":
            return True
        if self.api == "chat":
            return False
        official = self.base_url == OPENAI_BASE_URL.rstrip("/")
        wants_reasoning = self.reasoning_effort not in ("", "none")
        return official and wants_reasoning

    def describe(self) -> dict[str, str]:
        """Settings worth recording in the session log.

        The endpoint is included because reasoning alone is ambiguous without
        it: on /v1/chat/completions a model that cannot combine tools with
        reasoning is silently retried at `none`, so two turns logged as
        `medium` could have reasoned very differently.
        """
        return {
            "reasoning": self.reasoning_effort or "unset",
            "endpoint": "/v1/responses" if self.uses_responses
            else "/v1/chat/completions",
        }

    # -- lifecycle ---------------------------------------------------------

    @property
    def api_key(self) -> str:
        if self._api_key is not None:
            return self._api_key
        return os.environ.get("OPENAI_API_KEY", "")

    def preflight(self) -> None:
        """Verify a key is available before the first request.

        Deliberately as cheap as the Gemini path's: a missing key is the only
        failure worth a round-trip to catch early, and everything else produces
        a far better message from the response that actually failed.
        """
        if not self.api_key.strip():
            raise RuntimeError(
                "⚠️ **OPENAI_API_KEY is not set.**\n\n"
                "Create a key at https://platform.openai.com/api-keys, then "
                "restart the app and enter it on the setup screen."
            )

    # -- main entry point --------------------------------------------------

    def chat(
        self,
        messages: list[dict],
        tools: list,
        system: str,
        on_event: Callable[[str, dict], None] | None = None,
    ) -> ModelResponse:
        self.preflight()
        if self.uses_responses:
            return self._chat_via_responses(messages, tools, system, on_event)
        return self._chat_via_completions(messages, tools, system, on_event)

    # -- /v1/responses -----------------------------------------------------

    def _chat_via_responses(
        self,
        messages: list[dict],
        tools: list,
        system: str,
        on_event: Callable[[str, dict], None] | None = None,
    ) -> ModelResponse:
        """The endpoint that can reason *and* call functions in the same turn."""
        payload: dict[str, Any] = {
            "model": self.model,
            "instructions": system,
            "input": self._to_responses_input(messages),
            "tools": _responses_tools(tools),
            # Grid data must not be retained server-side; CONDUCTOR's whole
            # premise is that the operator keeps their network to themselves.
            # The cost is that reasoning state is not held for us either, which
            # is why reasoning items are replayed from history below.
            "store": False,
        }
        if self.reasoning_effort:
            payload["reasoning"] = {"effort": self.reasoning_effort}

        data = self._post_with_retry(payload, "/responses", on_event=on_event)
        return self._responses_to_model_response(data)

    @staticmethod
    def _to_responses_input(messages: list[dict]) -> list[dict]:
        """Neutral messages → Responses input items.

        Flat items rather than nested messages: a tool call is a
        `function_call` item and its result a sibling `function_call_output`,
        paired by `call_id` exactly as chat/completions pairs by
        `tool_call_id`, so the same id bookkeeping applies.

        Reasoning items are replayed ahead of the call they belong to. With
        `store: false` OpenAI holds nothing between requests, so dropping them
        would make the model re-derive its reasoning after every tool result.
        """
        items: list[dict] = []
        pending: list[tuple[str, str]] = []
        counter = 0

        for msg in messages:
            role = msg.get("role")

            if role == "user":
                pending.clear()
                items.append({"role": "user", "content": msg.get("content") or ""})

            elif role == "assistant":
                pending = []
                for item in _decode_reasoning(msg.get("signature")):
                    items.append(item)
                if msg.get("content"):
                    items.append({
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": msg["content"]}],
                    })
                for call in msg.get("tool_calls") or []:
                    counter += 1
                    call_id = str(call.get("id") or f"call_{counter}")
                    pending.append((call_id, call.get("name", "")))
                    items.append({
                        "type": "function_call",
                        "call_id": call_id,
                        "name": call.get("name", ""),
                        "arguments": json.dumps(call.get("args") or {}, default=str),
                    })

            elif role == "tool":
                name = msg.get("name", "")
                call_id = OpenAIProvider._claim_call_id(pending, msg.get("id"), name)
                if call_id is None:
                    logger.warning(
                        "Dropping result for %r — no matching tool call in history.",
                        name or "<unnamed tool>",
                    )
                    continue
                content = msg.get("content")
                items.append({
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": content if isinstance(content, str)
                    else json.dumps(content, default=str),
                })

        return items

    @staticmethod
    def _responses_to_model_response(data: dict) -> ModelResponse:
        """Responses payload → neutral ModelResponse."""
        output = data.get("output") or []

        text = "\n".join(
            part.get("text", "")
            for item in output if item.get("type") == "message"
            for part in (item.get("content") or [])
            if part.get("type") == "output_text"
        ).strip()

        tool_calls: list[ToolCall] = []
        for item in output:
            if item.get("type") != "function_call" or not item.get("name"):
                continue
            raw_args = item.get("arguments")
            args: Any = raw_args
            if isinstance(raw_args, str):
                try:
                    args = json.loads(raw_args) if raw_args.strip() else {}
                except json.JSONDecodeError:
                    logger.warning("Unparseable tool arguments from %s: %r",
                                   item.get("name"), raw_args)
                    args = {}
            tool_calls.append(ToolCall(
                name=item["name"], args=args or {}, id=item.get("call_id"),
            ))

        # Reasoning items ride back on the neutral message's opaque signature
        # slot — the same mechanism Gemini uses for thought signatures — so the
        # next request can replay them.
        reasoning = [item for item in output if item.get("type") == "reasoning"]

        if data.get("status") == "incomplete" and not tool_calls:
            reason = (data.get("incomplete_details") or {}).get("reason", "")
            logger.warning("Reply was incomplete (%s).", reason or "unknown")
            text = (text + "\n\n*(This reply was cut off at the model's output "
                           "length limit.)*").strip()

        return ModelResponse(
            text=text, tool_calls=tool_calls, raw=data,
            signature=_encode_reasoning(reasoning),
        )

    # -- /v1/chat/completions ----------------------------------------------

    def _chat_via_completions(
        self,
        messages: list[dict],
        tools: list,
        system: str,
        on_event: Callable[[str, dict], None] | None = None,
    ) -> ModelResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": self._to_openai_messages(messages, system),
            "tools": to_json_schema_tools(tools),
            # Deterministic, matching the Gemini and Ollama paths. Reasoning
            # models reject an explicit temperature, so it is dropped on the
            # retry that says so rather than guessed at from the model name —
            # names are not a reliable signal, and compatible servers vary.
            "temperature": 0,
        }
        if self.reasoning_effort:
            payload["reasoning_effort"] = self.reasoning_effort
        data = self._post_with_retry(payload, "/chat/completions", on_event=on_event)
        return self._to_model_response(data)

    # -- transport and retry -----------------------------------------------

    def _post_with_retry(
        self,
        payload: dict,
        path: str = "/chat/completions",
        on_event: Callable[[str, dict], None] | None = None,
        max_retries: int | None = None,
    ) -> dict:
        retries = MODEL_RETRY_ATTEMPTS if max_retries is None else max_retries
        url = f"{self.base_url}{path}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        for attempt in range(retries + 1):
            try:
                resp = httpx.post(
                    url, json=payload, headers=headers, timeout=self.timeout_s
                )
            except httpx.TimeoutException as exc:
                if attempt < retries:
                    self._announce_retry(
                        attempt, retries, 30,
                        f"Request timed out after {self.timeout_s:.0f}s", on_event,
                    )
                    time.sleep(30)
                    continue
                raise RuntimeError(
                    f"⚠️ **OpenAI timed out after {self.timeout_s:.0f}s.**\n\n"
                    "Raise `OPENAI_TIMEOUT_S` in your `.env`, or start a new "
                    "conversation to shorten the prompt."
                ) from exc
            except httpx.HTTPError as exc:
                if attempt < retries:
                    self._announce_retry(attempt, retries, 10,
                                         "Connection error", on_event)
                    time.sleep(10)
                    continue
                raise RuntimeError(
                    f"⚠️ **Lost connection to {self.base_url}.**\n\n{exc}"
                ) from exc

            if resp.status_code == 200:
                return resp.json()

            message, retryable, wait = self._describe_http_error(resp)

            # `temperature` and `reasoning_effort` are each accepted by some
            # models and refused by others — a fact about the model, not about
            # this request. Adjust whichever was named and try again rather
            # than failing the turn over it.
            adjusted = self._adjust_rejected_param(payload, resp)
            if adjusted:
                logger.warning(
                    "Model %s refused the request; retrying with %s. "
                    "Note that a model which cannot combine tools with "
                    "reasoning on /v1/chat/completions will answer this turn "
                    "without reasoning.",
                    self.model, adjusted,
                )
                if on_event:
                    on_event("retry", {
                        "attempt": attempt + 1, "total": retries, "wait_s": 0,
                        "reason": f"model requires {adjusted}",
                    })
                continue

            if not retryable or attempt >= retries:
                raise RuntimeError(message)

            self._announce_retry(attempt, retries, wait,
                                 f"HTTP {resp.status_code}", on_event)
            time.sleep(wait)

        raise RuntimeError("Unexpected exit from retry loop")  # pragma: no cover

    @staticmethod
    def _announce_retry(attempt, retries, wait, reason, on_event) -> None:
        logger.warning("%s (attempt %d/%d). Retrying in %ds…",
                       reason, attempt + 1, retries, wait)
        if on_event:
            on_event("retry", {
                "attempt": attempt + 1, "total": retries,
                "wait_s": wait, "reason": reason,
            })

    @staticmethod
    def _adjust_rejected_param(payload: dict, resp: httpx.Response) -> str | None:
        """Fix up whichever optional parameter this 400 complained about.

        Returns a description of what changed, or None if nothing did.

        Most rejections are cured by dropping the parameter. One is not:
        `gpt-5.6-luna` refuses function tools combined with *any* reasoning
        effort on `/v1/chat/completions` and asks for the literal value
        `'none'`. Dropping it there just lets the model apply its own default
        effort and fail again identically — which is exactly what happened the
        first time this was written.
        """
        if resp.status_code != 400:
            return None
        try:
            error = (resp.json() or {}).get("error") or {}
        except Exception:  # noqa: BLE001
            error = {}

        message = str(error.get("message") or "")
        haystack = f"{message} {resp.text}"

        # Rule one: reasoning must be off rather than absent.
        if (
            "reasoning_effort" in haystack
            and "none" in haystack
            and payload.get("reasoning_effort") not in (None, "none")
        ):
            payload["reasoning_effort"] = "none"
            return "reasoning_effort=none"

        named = str(error.get("param") or "").strip()
        if named in _OPTIONAL_PARAMS and named in payload:
            payload.pop(named)
            return f"dropped {named}"

        for param in _OPTIONAL_PARAMS:
            if param in payload and param in haystack:
                payload.pop(param)
                return f"dropped {param}"
        return None

    def _describe_http_error(self, resp: httpx.Response) -> tuple[str, bool, int]:
        """(user-facing message, is_retryable, seconds to wait)."""
        try:
            error = (resp.json() or {}).get("error") or {}
        except Exception:  # noqa: BLE001
            error = {}
        code = str(error.get("code") or error.get("type") or "").strip()
        detail = str(error.get("message") or resp.text or "").strip()

        # The server's own backoff beats any guess we could make.
        try:
            wait = int(float(resp.headers.get("retry-after", "")))
        except (TypeError, ValueError):
            wait = MODEL_OVERLOADED_RETRY_DELAY_S

        if resp.status_code == 401:
            return (
                "⚠️ **OpenAI rejected the API key (HTTP 401).**\n\n"
                f"{detail}\n\nCheck `OPENAI_API_KEY` in `llm_agent/.env`, or "
                "enter a new key in the model settings.",
                False, wait,
            )

        if resp.status_code == 404 or code == "model_not_found":
            available = probe(self.base_url, self.api_key).get("models") or []
            shortlist = ", ".join(f"`{m}`" for m in available[:15]) or "none reported"
            return (
                f"⚠️ **Model `{self.model}` is not available to this key.**\n\n"
                f"{detail}\n\nModels this key can reach: {shortlist}.\n\n"
                "Set a different one in the model settings, or via `OPENAI_MODEL`.",
                False, wait,
            )

        if code == "insufficient_quota" or resp.status_code == 402:
            return (
                "⚠️ **This OpenAI account has no remaining credit.**\n\n"
                f"{detail}\n\nThis is a billing state, not a rate limit — "
                "retrying will not clear it. Add credit at "
                "https://platform.openai.com/settings/organization/billing, "
                "or switch to Ollama to run locally at no cost.",
                False, wait,
            )

        if code == "context_length_exceeded":
            return (
                "⚠️ **The conversation is longer than the model's context window.**\n\n"
                f"{detail}\n\nStart a new conversation to clear the accumulated "
                "history, or pick a model with a larger window.",
                False, wait,
            )

        if code in _PERMANENT_CODES:
            return (f"⚠️ **OpenAI error ({code}).**\n\n{detail}", False, wait)

        if resp.status_code == 429:
            return (
                f"⚠️ **OpenAI rate limit hit (429).**\n\n{detail}",
                True, wait,
            )

        if resp.status_code >= 500:
            return (
                f"⚠️ **OpenAI is unavailable (HTTP {resp.status_code}).**\n\n{detail}",
                True, wait,
            )

        return (
            f"⚠️ **OpenAI error (HTTP {resp.status_code}).**\n\n{detail}",
            False, wait,
        )

    # -- message conversion ------------------------------------------------

    @staticmethod
    def _to_openai_messages(messages: list[dict], system: str) -> list[dict]:
        """Neutral messages → OpenAI chat messages.

        Ids are the whole difficulty. OpenAI pairs a tool result to its call by
        `tool_call_id`; the neutral history pairs them by order, and only
        carries an id when the producing provider set one. So each assistant
        turn's calls are given stable ids here, and the tool messages that
        follow consume them — by id when the history has one, by name when it
        does not, and by position as a last resort.
        """
        out: list[dict] = [{"role": "system", "content": system}]
        pending: list[tuple[str, str]] = []  # (tool_call_id, tool name)
        counter = 0

        for msg in messages:
            role = msg.get("role")

            if role == "user":
                pending.clear()
                out.append({"role": "user", "content": msg.get("content") or ""})

            elif role == "assistant":
                pending = []
                entry: dict[str, Any] = {"role": "assistant"}
                # Null rather than "" — an empty string alongside tool calls is
                # accepted by OpenAI but rejected by some compatible servers.
                entry["content"] = msg.get("content") or None

                calls = msg.get("tool_calls") or []
                if calls:
                    rendered = []
                    for call in calls:
                        counter += 1
                        call_id = str(call.get("id") or f"call_{counter}")
                        pending.append((call_id, call.get("name", "")))
                        rendered.append({
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": call.get("name", ""),
                                # Arguments go as a JSON *string*, not an object.
                                "arguments": json.dumps(call.get("args") or {},
                                                        default=str),
                            },
                        })
                    entry["tool_calls"] = rendered

                # An assistant turn with neither text nor calls is not a legal
                # message and carries nothing anyway.
                if entry["content"] is not None or entry.get("tool_calls"):
                    out.append(entry)

            elif role == "tool":
                name = msg.get("name", "")
                call_id = OpenAIProvider._claim_call_id(pending, msg.get("id"), name)
                if call_id is None:
                    # No call to attach this to: OpenAI would reject the message
                    # outright, so drop it rather than fail the whole turn.
                    logger.warning(
                        "Dropping result for %r — no matching tool call in history.",
                        name or "<unnamed tool>",
                    )
                    continue
                content = msg.get("content")
                out.append({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": content if isinstance(content, str)
                    else json.dumps(content, default=str),
                })

        return out

    @staticmethod
    def _claim_call_id(
        pending: list[tuple[str, str]], wanted: str | None, name: str
    ) -> str | None:
        """Take the id this tool result belongs to out of the pending list."""
        if wanted:
            for index, (call_id, _) in enumerate(pending):
                if call_id == wanted:
                    return pending.pop(index)[0]
        for index, (call_id, call_name) in enumerate(pending):
            if call_name == name:
                return pending.pop(index)[0]
        return pending.pop(0)[0] if pending else None

    @staticmethod
    def _to_model_response(data: dict) -> ModelResponse:
        """OpenAI chat-completions response → neutral ModelResponse."""
        choices = data.get("choices") or []
        message = (choices[0].get("message") or {}) if choices else {}

        tool_calls: list[ToolCall] = []
        for call in message.get("tool_calls") or []:
            fn = call.get("function") or {}
            name = fn.get("name")
            if not name:
                continue
            raw_args = fn.get("arguments")
            args: Any = raw_args
            if isinstance(raw_args, str):
                try:
                    args = json.loads(raw_args) if raw_args.strip() else {}
                except json.JSONDecodeError:
                    logger.warning("Unparseable tool arguments from %s: %r",
                                   name, raw_args)
                    args = {}
            tool_calls.append(
                ToolCall(name=name, args=args or {}, id=call.get("id"))
            )

        text = (message.get("content") or "").strip()

        # A reply cut off at the token limit reads as a finished answer to the
        # loop, which would then present a half-sentence as the result.
        if choices and choices[0].get("finish_reason") == "length" and not tool_calls:
            logger.warning("Reply was truncated at the model's output limit.")
            text = (text + "\n\n*(This reply was cut off at the model's output "
                           "length limit.)*").strip()

        return ModelResponse(text=text, tool_calls=tool_calls, raw=data)
