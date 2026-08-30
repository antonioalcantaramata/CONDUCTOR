"""
ollama.py — local Ollama provider.

Talks to Ollama's native `/api/chat` endpoint rather than its OpenAI-compatible
`/v1` shim, because `options.num_ctx` is not reliably honoured through the shim
and that parameter is load-bearing here (see below).

The context trap
----------------
Ollama defaults `num_ctx` to roughly 4096 no matter how large a window the model
advertises, and when the prompt exceeds it the server **silently truncates from
the front** — no error, no warning. The system prompt sits at the front, so the
agent quietly loses its instructions and starts emitting plausible nonsense.

This provider therefore (a) always sends an explicit `num_ctx`, and (b) refuses
to send a request it estimates will not fit, rather than letting it be truncated.
A loud failure is worth far more than a silent lobotomy.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Callable

import httpx

from ..config import (
    OLLAMA_HOST,
    OLLAMA_KEEP_ALIVE,
    OLLAMA_MODEL,
    OLLAMA_NUM_CTX,
    OLLAMA_OUTPUT_RESERVE,
    OLLAMA_THINK,
    OLLAMA_TIMEOUT_S,
)
from .base import ModelResponse, ToolCall
from .schema import to_json_schema_tools

logger = logging.getLogger(__name__)

# Rough bytes-per-token for JSON-ish English payloads. Only used for the
# fit-check, which is deliberately conservative rather than exact.
_BYTES_PER_TOKEN = 4


def probe(host: str | None = None) -> dict:
    """
    Inspect a local Ollama server without committing to a model.

    Used by the setup screen to show what's available before the user chooses.
    Returns::

        {"reachable": bool, "version": str, "models": [
            {"name": str, "size_gb": float, "tools": bool | None}, ...
        ]}

    `tools` is None when the server doesn't report capabilities.
    """
    base = (host or os.environ.get("OLLAMA_HOST") or OLLAMA_HOST).rstrip("/")
    out: dict[str, Any] = {"reachable": False, "version": "", "models": []}

    try:
        version = httpx.get(f"{base}/api/version", timeout=5)
        version.raise_for_status()
        out["reachable"] = True
        out["version"] = version.json().get("version", "")
    except Exception:  # noqa: BLE001
        return out

    try:
        tags = httpx.get(f"{base}/api/tags", timeout=10)
        tags.raise_for_status()
        entries = tags.json().get("models", [])
    except Exception:  # noqa: BLE001
        return out

    for entry in entries:
        name = entry.get("model") or entry.get("name") or ""
        if not name:
            continue
        caps = OllamaProvider(model=name, host=base)._capabilities()
        out["models"].append({
            "name": name,
            "size_gb": round((entry.get("size") or 0) / 1e9, 1),
            "tools": None if caps is None else ("tools" in caps),
        })

    out["models"].sort(key=lambda m: m["name"])
    return out


class OllamaProvider:
    """LLMProvider backed by a local Ollama server."""

    name = "ollama"

    def __init__(self, model: str | None = None, host: str | None = None) -> None:
        # Every setting is read from the live environment so the settings panel
        # can change any of them mid-session (see providers.reset_providers).
        self.model = model or os.environ.get("OLLAMA_MODEL") or OLLAMA_MODEL
        self.host = (host or os.environ.get("OLLAMA_HOST") or OLLAMA_HOST).rstrip("/")
        self.num_ctx = int(os.environ.get("OLLAMA_NUM_CTX") or OLLAMA_NUM_CTX)
        self.output_reserve = int(
            os.environ.get("OLLAMA_OUTPUT_RESERVE") or OLLAMA_OUTPUT_RESERVE
        )
        self.keep_alive = os.environ.get("OLLAMA_KEEP_ALIVE") or OLLAMA_KEEP_ALIVE
        self.timeout_s = float(os.environ.get("OLLAMA_TIMEOUT_S") or OLLAMA_TIMEOUT_S)
        _think = os.environ.get("OLLAMA_THINK")
        self.think = OLLAMA_THINK if _think is None else _think.strip().lower() == "true"
        self._preflighted = False

    def describe(self) -> dict[str, str]:
        """Settings worth recording in the session log.

        `num_ctx` rides along because it is the setting most likely to explain
        a bad local answer: too small a window and Ollama truncates the system
        prompt away.
        """
        return {
            "reasoning": "on" if self.think else "off",
            "endpoint": f"{self.host}/api/chat",
            "num_ctx": str(self.num_ctx),
        }

    # -- preflight ---------------------------------------------------------

    def preflight(self) -> None:
        """
        Verify the server is reachable, the model is present, and it can call
        tools. Cached after the first success — these facts don't change
        mid-session, and each check costs a round-trip.
        """
        if self._preflighted:
            return

        # 1. Is the server up?
        try:
            httpx.get(f"{self.host}/api/version", timeout=5).raise_for_status()
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"⚠️ **Cannot reach Ollama at {self.host}.**\n\n"
                "Start it with `ollama serve`, or set `OLLAMA_HOST` in your `.env` "
                "if it runs elsewhere."
            ) from exc

        # 2. Is the model pulled? `None` means the query itself failed, which we
        #    let pass; an empty set is a real answer meaning nothing is installed.
        installed = self._installed_models()
        if installed is not None and not self._model_installed(installed):
            available = ", ".join(f"`{m}`" for m in sorted(installed)) or "none"
            raise RuntimeError(
                f"⚠️ **Model `{self.model}` is not installed in Ollama.**\n\n"
                f"Pull it with:\n\n    ollama pull {self.model}\n\n"
                f"Currently installed: {available}."
            )

        # 3. Can it call tools? Without this the 20 grid tools are unreachable
        #    and the agent can only chat.
        capabilities = self._capabilities()
        if capabilities is not None and "tools" not in capabilities:
            raise RuntimeError(
                f"⚠️ **Model `{self.model}` does not support tool calling.**\n\n"
                f"Reported capabilities: {', '.join(capabilities)}.\n\n"
                "CONDUCTOR drives the grid entirely through tools, so a model "
                "without this capability cannot run any analysis. Pick a model "
                "whose Ollama page lists the `tools` capability."
            )

        self._preflighted = True

    def _installed_models(self) -> set[str] | None:
        """
        Model names known to the server.

        Returns an empty set when the server genuinely has no models, and None
        when the query failed — the caller must not treat those alike.
        """
        try:
            resp = httpx.get(f"{self.host}/api/tags", timeout=10)
            resp.raise_for_status()
            return {m.get("model", "") for m in resp.json().get("models", [])}
        except Exception:  # noqa: BLE001
            logger.debug("Could not list Ollama models; skipping presence check.")
            return None

    def _model_installed(self, installed: set[str]) -> bool:
        """Ollama normalises a bare name to `name:latest`."""
        candidates = {self.model, f"{self.model}:latest"}
        return bool(candidates & installed)

    def _capabilities(self) -> list[str] | None:
        """
        Model capability tags (e.g. tools, vision).

        None means the server did not tell us — older Ollama builds omit the
        field — in which case we let the request proceed rather than blocking on
        a check we cannot perform.
        """
        try:
            resp = httpx.post(
                f"{self.host}/api/show", json={"model": self.model}, timeout=15
            )
            resp.raise_for_status()
            caps = resp.json().get("capabilities")
            return list(caps) if caps is not None else None
        except Exception:  # noqa: BLE001
            logger.debug("Could not read capabilities for %s.", self.model)
            return None

    # -- context fit -------------------------------------------------------

    def _check_fits(self, payload: dict) -> None:
        """
        Refuse rather than let the server truncate the system prompt away.

        Estimated from serialised bytes: imprecise, but the failure it guards
        against is silent, so erring toward refusal is the right bias.
        """
        estimated = len(json.dumps(payload, default=str)) // _BYTES_PER_TOKEN
        budget = self.num_ctx - self.output_reserve
        if estimated <= budget:
            return

        raise RuntimeError(
            f"⚠️ **Prompt too large for the configured context window.**\n\n"
            f"Estimated input: ~{estimated:,} tokens. "
            f"Usable budget: ~{budget:,} "
            f"({self.num_ctx:,} `OLLAMA_NUM_CTX` minus "
            f"{self.output_reserve:,} reserved for the reply).\n\n"
            "Ollama would silently discard the beginning of this prompt — including "
            "the system instructions — so the request was stopped instead.\n\n"
            "Fixes: raise `OLLAMA_NUM_CTX` in your `.env` (the model must support it, "
            "and larger windows need more memory), or start a new conversation to "
            "clear the accumulated history."
        )

    # -- warm-up -----------------------------------------------------------

    def is_registered(self) -> bool:
        """
        Whether Ollama still lists this model as loaded.

        This is a *hint*, not a fact. `/api/ps` reports Ollama's bookkeeping —
        the keep_alive registration — and keeps reporting a model with its full
        `size_vram` after the weights have actually been evicted (killing the
        runner process, memory pressure, sleep). Never use it to tell the user
        something is warm; only to decide whether a warning is worth showing.
        """
        try:
            resp = httpx.get(f"{self.host}/api/ps", timeout=5)
            resp.raise_for_status()
            running = {m.get("model", "") for m in resp.json().get("models", [])}
            return bool({self.model, f"{self.model}:latest"} & running)
        except Exception:  # noqa: BLE001
            return False

    def warm_up(self, tools: list, system: str) -> tuple[float, str | None]:
        """
        Prime the prompt cache with the system prompt and tool schemas.

        Processing that ~25k-token prefix is the dominant cost of a cold local
        model — measured at roughly 3 minutes on a 12B model — but Ollama reuses
        the cached prefix on later calls, dropping it to near zero. Paying it
        once at startup, where it can be explained and shown, is far better than
        ambushing the user's first question with it.

        Returns (elapsed_seconds, error_message). Never raises — a failed warm-up
        costs speed, not correctness — but the error is returned rather than only
        logged, so the caller can say so instead of appearing to do nothing.
        """
        import time as _time

        started = _time.perf_counter()
        payload = {
            "model": self.model,
            "messages": self._to_ollama_messages(
                [{"role": "user", "content": "Reply with OK."}], system
            ),
            "tools": to_json_schema_tools(tools),
            "stream": False,
            "think": self.think,
            "keep_alive": self.keep_alive,
            "options": {
                "num_ctx": self.num_ctx,
                "temperature": 0,
                "num_predict": 1,  # cache the prompt; don't pay for a real answer
            },
        }
        try:
            resp = httpx.post(f"{self.host}/api/chat", json=payload, timeout=self.timeout_s)
            elapsed = _time.perf_counter() - started
            if resp.status_code != 200:
                return elapsed, self._describe_http_error(resp)
            return elapsed, None
        except Exception as exc:  # noqa: BLE001
            logger.warning("Model warm-up failed; first question will be slow.",
                           exc_info=True)
            return _time.perf_counter() - started, str(exc)

    # -- main entry point --------------------------------------------------

    def chat(
        self,
        messages: list[dict],
        tools: list,
        system: str,
        on_event: Callable[[str, dict], None] | None = None,
    ) -> ModelResponse:
        self.preflight()

        # A cold model means the full prefix is about to be reprocessed (minutes,
        # not seconds). Warm-up covers the startup case, but the model can be
        # evicted mid-session — so say what's happening rather than appearing to
        # hang. `is_registered` under-reports (it can claim a model is resident
        # when it isn't), so this warning may be missed; it is never wrong when
        # shown, which is the safer direction for a hint.
        if on_event and not self.is_registered():
            on_event("model_loading", {"model": self.model})

        payload = {
            "model": self.model,
            "messages": self._to_ollama_messages(messages, system),
            "tools": to_json_schema_tools(tools),
            "stream": False,
            # Reasoning traces cost ~12s per call here and the Gemini path
            # disables them too; keep the two backends comparable.
            "think": self.think,
            # Keep the model — and its cached prompt prefix — resident between
            # calls. Without this the ~25k-token prefix is reprocessed from cold
            # on every question (~180s vs ~0.2s cached).
            "keep_alive": self.keep_alive,
            "options": {
                # Explicit every time — the server default would truncate us.
                "num_ctx": self.num_ctx,
                "temperature": 0,
            },
        }
        self._check_fits(payload)

        try:
            resp = httpx.post(
                f"{self.host}/api/chat", json=payload, timeout=self.timeout_s
            )
        except httpx.TimeoutException as exc:
            raise RuntimeError(
                f"⚠️ **Ollama timed out after {self.timeout_s:.0f}s.**\n\n"
                "Local generation over a large context can be slow, especially on "
                "the first call while the model loads. Raise `OLLAMA_TIMEOUT_S`, or "
                "use a smaller model."
            ) from exc
        except httpx.HTTPError as exc:
            raise RuntimeError(
                f"⚠️ **Lost connection to Ollama at {self.host}.**\n\n{exc}"
            ) from exc

        if resp.status_code != 200:
            raise RuntimeError(self._describe_http_error(resp))

        return self._to_model_response(resp.json())

    def _describe_http_error(self, resp: httpx.Response) -> str:
        try:
            detail = resp.json().get("error", resp.text)
        except Exception:  # noqa: BLE001
            detail = resp.text
        if resp.status_code == 404:
            return (
                f"⚠️ **Ollama could not find model `{self.model}`.**\n\n"
                f"Pull it with:\n\n    ollama pull {self.model}\n\n"
                f"Server said: {detail}"
            )
        return f"⚠️ **Ollama error (HTTP {resp.status_code}).**\n\n{detail}"

    # -- message conversion ------------------------------------------------

    @staticmethod
    def _to_ollama_messages(messages: list[dict], system: str) -> list[dict]:
        """Neutral messages → Ollama chat messages."""
        out: list[dict] = [{"role": "system", "content": system}]

        for msg in messages:
            role = msg.get("role")

            if role == "user":
                out.append({"role": "user", "content": msg.get("content") or ""})

            elif role == "assistant":
                entry: dict[str, Any] = {
                    "role": "assistant",
                    "content": msg.get("content") or "",
                }
                calls = msg.get("tool_calls") or []
                if calls:
                    entry["tool_calls"] = [
                        {"function": {"name": c["name"], "arguments": c.get("args") or {}}}
                        for c in calls
                    ]
                out.append(entry)

            elif role == "tool":
                # Ollama expects the result as text on a `tool` message.
                content = msg.get("content")
                out.append({
                    "role": "tool",
                    "tool_name": msg.get("name", ""),
                    "content": content if isinstance(content, str)
                    else json.dumps(content, default=str),
                })

        return out

    @staticmethod
    def _to_model_response(data: dict) -> ModelResponse:
        """Ollama /api/chat response → neutral ModelResponse."""
        message = data.get("message") or {}

        tool_calls: list[ToolCall] = []
        for call in message.get("tool_calls") or []:
            fn = call.get("function") or {}
            name = fn.get("name")
            if not name:
                continue
            args = fn.get("arguments")
            # Ollama sends a dict; some models/builds emit a JSON string.
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    logger.warning("Unparseable tool arguments from %s: %r", name, args)
                    args = {}
            tool_calls.append(ToolCall(name=name, args=args or {}))

        text = (message.get("content") or "").strip()

        # Thinking models sometimes put everything in `thinking` and leave
        # `content` empty. With no tool calls either, the loop would read that as
        # a finished turn and show the user a blank reply — so fall back to the
        # reasoning text rather than losing the answer entirely.
        if not text and not tool_calls:
            thinking = (message.get("thinking") or "").strip()
            if thinking:
                logger.warning(
                    "Model returned only a reasoning trace; using it as the reply."
                )
                text = thinking

        return ModelResponse(text=text, tool_calls=tool_calls, raw=data)
