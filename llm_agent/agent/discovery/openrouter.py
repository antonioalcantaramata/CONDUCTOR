"""
openrouter.py — which OpenRouter models CONDUCTOR can use, found live.

Free models on OpenRouter are renamed and retired often, so none is
hard-coded. `GET https://openrouter.ai/api/v1/models` is public and served
from a CDN cache (it needs no key, so it cannot count against a key's
free-model requests). A model qualifies when it is:

* **free** (for the free tier): its ID ends in `:free`, or prompt and
  completion are both priced "0";
* **tool-calling**: `supported_parameters` includes "tools";
* **text in, text out**: "text" in both `architecture.input_modalities` and
  `output_modalities`;
* **big enough**: `context_length` ≥ `OPENROUTER_MIN_CONTEXT` (default 131,072);
* **not retiring soon**: `expiration_date` more than
  `FREE_MODEL_EXPIRY_GRACE_DAYS` (default 14) away. The rest carry a
  "retires on <date>" badge.

OpenRouter's routers and aliases (`openrouter/free` picks a free model at
random; `~…-latest` redirects) are left out: CONDUCTOR must know which model
answered.

The data-use warning is a **keyword heuristic** on the description — the API
exposes no per-endpoint training or retention policy — and is labelled so: no
warning never means a model is private.

Pure functions except `fetch` and `discover`.
"""

from __future__ import annotations

import datetime as _dt
import os
import re
from typing import Any

import httpx

from . import cache

MODELS_URL = "https://openrouter.ai/api/v1/models"
DEFAULT_MIN_CONTEXT = 131_072
DEFAULT_GRACE_DAYS = 14
_EXCLUDED_PREFIXES = ("openrouter/", "~")
_DATA_WARNING = re.compile(
    r"\b(?:train(?:s|ed|ing)?\s+on|used\s+(?:for|to)\s+train|retain(?:s|ed)?|retention|"
    r"improve\s+(?:our|their)\s+models?|may\s+be\s+logged|logs?\s+(?:your\s+)?prompts|"
    r"prompts?\s+(?:and\s+(?:outputs?|completions?)\s+)?(?:may\s+be|are)\s+(?:used|stored|logged|retained))\b",
    re.IGNORECASE,
)


def min_context() -> int:
    try:
        return int(os.environ.get("OPENROUTER_MIN_CONTEXT") or DEFAULT_MIN_CONTEXT)
    except ValueError:
        return DEFAULT_MIN_CONTEXT


def grace_days() -> int:
    try:
        return int(os.environ.get("FREE_MODEL_EXPIRY_GRACE_DAYS") or DEFAULT_GRACE_DAYS)
    except ValueError:
        return DEFAULT_GRACE_DAYS


def _price(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def is_free(model: dict) -> bool:
    if str(model.get("id", "")).endswith(":free"):
        return True
    pricing = model.get("pricing") or {}
    return _price(pricing.get("prompt")) == 0 and _price(pricing.get("completion")) == 0


def supports_tools(model: dict) -> bool:
    return "tools" in (model.get("supported_parameters") or [])


def text_in_out(model: dict) -> bool:
    arch = model.get("architecture") or {}
    return "text" in (arch.get("input_modalities") or []) and "text" in (arch.get("output_modalities") or [])


def context_length(model: dict) -> int:
    value = model.get("context_length") or (model.get("top_provider") or {}).get("context_length")
    return int(value) if isinstance(value, (int, float)) else 0


def expiry(model: dict) -> _dt.date | None:
    raw = model.get("expiration_date")
    if not raw:
        return None
    try:
        return _dt.date.fromisoformat(str(raw)[:10])
    except ValueError:
        return None


def data_warning(model: dict) -> str | None:
    """The description's phrase about data use, if any (a heuristic)."""
    match = _DATA_WARNING.search(model.get("description") or "")
    return match.group(0) if match else None


def check(model: dict, *, free_only: bool = True, min_ctx: int | None = None,
          grace: int | None = None, today: _dt.date | None = None) -> str | None:
    """Why a model does not qualify, or None when it does."""
    min_ctx = min_context() if min_ctx is None else min_ctx
    grace = grace_days() if grace is None else grace
    today = today or _dt.date.today()
    model_id = str(model.get("id") or "")
    if not model_id or model_id.startswith(_EXCLUDED_PREFIXES):
        return "a router or alias, not a model"
    if free_only and not is_free(model):
        return "not free"
    if not supports_tools(model):
        return "no tool calling"
    if not text_in_out(model):
        return "not text in and text out"
    if context_length(model) < min_ctx:
        return f"context below {min_ctx:,} tokens"
    ends = expiry(model)
    if ends is not None and (ends - today).days <= grace:
        return f"retires on {ends.isoformat()}"
    return None


def summarise(model: dict, *, today: _dt.date | None = None) -> dict:
    """The fields CONDUCTOR keeps and shows for a qualifying model."""
    today = today or _dt.date.today()
    ends = expiry(model)
    pricing = model.get("pricing") or {}
    return {
        "id": model["id"],
        "name": model.get("name") or model["id"],
        "context_length": context_length(model),
        "created": int(model.get("created") or 0),
        "expiration_date": ends.isoformat() if ends else None,
        "days_left": (ends - today).days if ends else None,
        "data_warning": data_warning(model),
        "reasoning": "reasoning" in (model.get("supported_parameters") or []),
        "free": is_free(model),
        "price_prompt": _price(pricing.get("prompt")),
        "price_completion": _price(pricing.get("completion")),
    }


def rank(models: list[dict]) -> list[dict]:
    """Largest context first, then newest, then by ID (stable and testable).
    The API reports neither usage nor popularity, so recency is the tiebreak."""
    return sorted(models, key=lambda m: (-int(m.get("context_length") or 0), -int(m.get("created") or 0),
                                         str(m.get("id"))))


def qualifying(payload: dict, *, free_only: bool = True, min_ctx: int | None = None,
               grace: int | None = None, today: _dt.date | None = None) -> dict:
    """{"models": ranked qualifying models, "excluded": {id: reason},
    "n_free": free models listed} from a /models response."""
    models, excluded, n_free = [], {}, 0
    for m in payload.get("data") or []:
        if not isinstance(m, dict) or not m.get("id"):
            continue
        n_free += is_free(m)
        why = check(m, free_only=free_only, min_ctx=min_ctx, grace=grace, today=today)
        if why is None:
            models.append(summarise(m, today=today))
        elif not free_only or is_free(m):
            excluded[m["id"]] = why
    return {"models": rank(models), "excluded": excluded, "n_free": n_free}


def fallbacks(models: list[dict], selected: str, n: int = 2) -> list[str]:
    """The next `n` qualifying models after `selected`, for OpenRouter's
    server-side `models` fallback list."""
    ids = [m["id"] for m in models]
    rest = [i for i in ids if i != selected]
    return rest[:n]


def next_model(models: list[dict], failed: set[str]) -> str | None:
    """The best qualifying model not already known to fail."""
    for m in models:
        if m["id"] not in failed:
            return m["id"]
    return None


def fetch(timeout: float = 20.0) -> dict:
    resp = httpx.get(MODELS_URL, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def discover(*, free_only: bool = True, force: bool = False) -> dict:
    """Qualifying models, from the cache when fresh, else fetched.

    Returns {"models", "excluded", "n_free", "fetched_at", "from_cache",
    "error"}. A failed fetch falls back to the last cached list (with the
    error), and to an empty list when there is none — never raises.
    """
    section = "openrouter_free" if free_only else "openrouter_paid"
    if not force:
        hit = cache.get(section)
        if hit is not None:
            return {**hit, "from_cache": True, "error": None}
    try:
        result = qualifying(fetch(), free_only=free_only)
    except Exception as exc:  # noqa: BLE001 — discovery must not take the page down
        stale = cache.get_any(section)
        if stale is not None:
            return {**stale, "from_cache": True, "error": f"Could not refresh the list: {exc}"}
        return {"models": [], "excluded": {}, "n_free": 0, "fetched_at": None, "from_cache": False,
                "error": f"Could not reach OpenRouter: {exc}"}
    entry = cache.put(section, result)
    return {**entry, "from_cache": False, "error": None}


def key_status(api_key: str, timeout: float = 10.0) -> dict | None:
    """`GET /api/v1/key`: the key's free-model requests today
    (`free_model_daily_requests` used / limit / remaining). None on failure."""
    if not api_key:
        return None
    try:
        resp = httpx.get("https://openrouter.ai/api/v1/key", headers={"Authorization": f"Bearer {api_key}"},
                         timeout=timeout)
        if resp.status_code != 200:
            return None
        return (resp.json() or {}).get("data") or None
    except Exception:  # noqa: BLE001
        return None
