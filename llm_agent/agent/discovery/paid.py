"""
paid.py — model lists for the paid options, read from the providers with your key.

So paid options work like the free ones: pick from a list rather than type a
model ID. Each list is the provider's own (`GET /models` on OpenAI or a
compatible endpoint, Anthropic's models API, Google's models list), filtered to
chat models CONDUCTOR can drive, newest first, and cached like the other
discovery lists. Listing models generates nothing and costs nothing.

* OpenAI and compatible endpoints: the list has no capability fields, so
  non-chat models (embeddings, speech, images, moderation, …) are left out by
  name, as the launch check already does.
* Anthropic: every listed Claude model calls tools.
* Gemini: text models of the Flash-Lite / Flash / Pro families (no speech,
  image or alias models), with their input token limit.
"""

from __future__ import annotations

import hashlib
import re

from . import cache
from . import gemini as _gemini

_NOT_CHAT = None


def _not_chat():
    global _NOT_CHAT
    if _NOT_CHAT is None:
        from ..model_check import _NOT_CHAT_RE
        _NOT_CHAT = _NOT_CHAT_RE
    return _NOT_CHAT


_LIVE = re.compile(r"(?:^|-)live(?:-|$)", re.IGNORECASE)     # live / voice sessions, not chat


def openai_chat_models(ids: list[str], created: dict | None = None) -> list[dict]:
    """Chat models from an OpenAI-style /models list, newest first."""
    created = created or {}
    keep = [i for i in ids if not _not_chat().search(i) and not _LIVE.search(i)]
    keep.sort(key=lambda i: (-(created.get(i) or 0), i))
    return [{"id": i, "label": i, "context_window": None, "created": created.get(i)} for i in keep]


def gemini_text_models(listed: list[dict]) -> list[dict]:
    """Gemini text models (Flash-Lite, Flash, Pro), newest version first."""
    out = []
    for m in listed:
        parsed = _gemini.family(m.get("id", ""))
        if not parsed or "generateContent" not in (m.get("methods") or []):
            continue
        kind, version, preview = parsed
        out.append({"id": m["id"], "label": m["id"], "context_window": m.get("input_token_limit"),
                    "_order": (version, {"pro": 2, "flash": 1, "flash-lite": 0}[kind], not preview)})
    out.sort(key=lambda m: m["_order"], reverse=True)
    for m in out:
        m.pop("_order")
    return out


def _list(provider: str, api_key: str, base_url: str | None) -> list[dict]:
    if provider == "openai":
        from ..providers.openai import probe
        info = probe(base_url, api_key)
        if not info.get("reachable"):
            raise RuntimeError(info.get("error") or "the endpoint did not answer")
        return openai_chat_models(info.get("models") or [], info.get("created") or {})
    if provider == "anthropic":
        from ..providers.claude import AnthropicProvider
        client = AnthropicProvider(api_key=api_key).client
        models = sorted(client.models.list(limit=100), key=lambda m: getattr(m, "created_at", None) or 0,
                        reverse=True)
        return [{"id": m.id, "label": getattr(m, "display_name", None) or m.id, "context_window": None}
                for m in models]
    if provider == "google":
        return gemini_text_models(_gemini.fetch(api_key))
    raise ValueError(provider)


def discover(provider: str, api_key: str, base_url: str | None = None, *, force: bool = False) -> dict:
    """{"models", "fetched_at", "error"}; never raises. Empty without a key."""
    if not api_key:
        return {"models": [], "fetched_at": None, "error": None}
    tag = hashlib.sha1(f"{provider}|{base_url or ''}|{api_key[-6:]}".encode()).hexdigest()[:10]
    section = f"paid_{provider}_{tag}"
    hit = None if force else cache.get(section)
    if hit is not None:
        return {**hit, "error": None}
    try:
        entry = cache.put(section, {"models": _list(provider, api_key, base_url)})
        return {**entry, "error": None}
    except Exception as exc:  # noqa: BLE001 — a list we cannot read is not a crash
        reason = _reason(exc)
        stale = cache.get_any(section)
        if stale is not None:
            return {**stale, "error": f"Could not refresh the model list: {reason}"}
        return {"models": [], "fetched_at": None, "error": f"Could not list models: {reason}"}


def _reason(exc: Exception) -> str:
    """A short reason, not a raw SDK dump."""
    text = str(exc)
    low = text.lower()
    if "401" in text or "authentication" in low or "invalid x-api-key" in low or "api key" in low and "invalid" in low:
        return "the API key was rejected (HTTP 401) — update it in the settings below."
    if "403" in text or "permission" in low:
        return "this key is not allowed to list models (HTTP 403)."
    return text[:160]
