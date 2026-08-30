"""
providers — pluggable LLM backends for the agentic loop.

`loop.py` asks for `get_provider()` and talks to whatever comes back through
the `LLMProvider` interface in `base.py`. Selection is driven by the
`LLM_PROVIDER` environment variable (see `config.LLM_PROVIDER`):

    google  — Google Generative AI (default)
    openai  — the OpenAI API, or any endpoint speaking its chat-completions
              dialect (Azure OpenAI, OpenRouter, vLLM, …) via OPENAI_BASE_URL
    ollama  — a local Ollama server

Providers are imported lazily so that selecting one backend never requires the
other's dependencies to be installed.
"""

from __future__ import annotations

import os

from ..config import LLM_PROVIDER
from .base import (
    LLMProvider,
    ModelResponse,
    ToolCall,
    assistant_message,
    describe_provider,
    iter_tool_calls,
    tool_message,
    user_message,
)

__all__ = [
    "LLMProvider",
    "ModelResponse",
    "ToolCall",
    "assistant_message",
    "describe_provider",
    "iter_tool_calls",
    "tool_message",
    "user_message",
    "get_provider",
    "available_providers",
    "active_provider_name",
    "reset_providers",
]

_PROVIDERS = ("google", "openai", "ollama")

_cache: dict[str, LLMProvider] = {}


def available_providers() -> tuple[str, ...]:
    return _PROVIDERS


def active_provider_name() -> str:
    """
    The provider currently selected.

    Read from the live environment rather than the import-time constant, so the
    setup screen can switch backends and have it take effect on the next
    Streamlit rerun instead of requiring a restart.
    """
    return (os.environ.get("LLM_PROVIDER") or LLM_PROVIDER or "google").strip().lower()


def get_provider(name: str | None = None) -> LLMProvider:
    """
    Return the configured provider instance (memoised per name).

    Args:
        name: override for the configured provider; mainly for tests and for the
              Streamlit setup screen, which may want to probe a provider before
              committing to it.
    """
    key = (name or active_provider_name()).strip().lower()

    if key not in _cache:
        if key == "google":
            from .gemini import GeminiProvider

            _cache[key] = GeminiProvider()
        elif key == "openai":
            from .openai import OpenAIProvider

            _cache[key] = OpenAIProvider()
        elif key == "ollama":
            from .ollama import OllamaProvider

            _cache[key] = OllamaProvider()
        else:
            raise RuntimeError(
                f"Unknown LLM_PROVIDER {key!r}. "
                f"Supported values: {', '.join(_PROVIDERS)}."
            )

    return _cache[key]


def reset_providers() -> None:
    """
    Drop memoised provider instances.

    Call after changing provider settings at runtime (the setup screen does)
    so the next `get_provider()` rebuilds against the new environment.
    """
    _cache.clear()
