"""
smoke_test.py — the opt-in "Test tool calling" check.

Sends one minimal request with a single dummy tool and checks that the model
calls it. It costs one request from the user's quota, so it only runs when
asked. The result (passed or not, latency, error) is cached per provider and
model for the discovery cache window, and the latency is recorded.

A model can list tool support and still not call tools reliably; this is the
cheapest way to find out before a real question spends several requests.
"""

from __future__ import annotations

import time

from google.genai import types

from . import usage
from .discovery import cache

_TOOL = types.Tool(function_declarations=[types.FunctionDeclaration(
    name="get_time",
    description="Return the current time. Always call this tool to answer.",
    parameters=types.Schema(type=types.Type.OBJECT, properties={}),
)])
_SYSTEM = "You are testing tool calling. Answer only by calling the get_time tool."
_PROMPT = "What time is it? Use the get_time tool."


def _provider(name: str, model: str):
    if name in ("openrouter", "nvidia", "deepseek"):
        from .providers.openai_compat import CompatProvider
        provider = CompatProvider(name, model=model)
        provider.timeout_retries = 0
        return provider
    if name == "google":
        from .providers.gemini import GeminiProvider
        return GeminiProvider(model=model)
    if name == "openai":
        from .providers.openai import OpenAIProvider
        return OpenAIProvider(model=model)
    if name == "anthropic":
        from .providers.claude import AnthropicProvider
        return AnthropicProvider(model=model)
    if name == "ollama":
        from .providers.ollama import OllamaProvider
        return OllamaProvider(model=model)
    raise ValueError(f"Unknown provider {name}")


def run(name: str, model: str, *, provider=None) -> dict:
    """{"ok", "latency_s", "error", "called"}; never raises."""
    from .providers.base import user_message
    started = time.monotonic()
    try:
        provider = provider or _provider(name, model)
        response = provider.chat([user_message(_PROMPT)], [_TOOL], _SYSTEM)
        called = [c.name for c in response.tool_calls]
        error = None
        if not called:
            error = "The model answered without calling the tool."
        elif "get_time" not in called:
            error = f"The model called {', '.join(called)} instead of get_time (a malformed tool call)."
        result = {"ok": "get_time" in called, "called": called, "error": error}
    except Exception as exc:  # noqa: BLE001 — a failed test is a result, not a crash
        result = {"ok": False, "called": [], "error": str(exc)[:500]}
    result["latency_s"] = round(time.monotonic() - started, 1)
    cache.put_smoke(name, model, result)
    usage.record_calls(name, model, 1)
    if result["ok"]:
        usage.record_latency(name, model, result["latency_s"])
    return result
