"""
model_check.py — does the configured model still exist?

Hosted providers retire models on their own schedule. A retired model used to
surface as an API error on the operator's first question; this asks the
provider before the app starts, using the listing endpoints every hosted
backend offers for free:

    OpenAI     GET /v1/models          (via `openai.probe`)
    Anthropic  Models API              (`client.models.retrieve` / `.list`)
    Gemini     models.get / models.list

It never changes the model. When a newer model of the same family exists it
says so, and the operator decides: silently moving a research deployment to a
different model would make two sessions' logs incomparable without anyone
having chosen that.

Ollama is not handled here — its launch screen already checks the installed
models, which are the only ones it can run.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Iterable, NamedTuple

logger = logging.getLogger(__name__)

OK = "ok"
MISSING = "missing"             # the key works, the model is not offered to it
KEY_REJECTED = "key_rejected"   # the provider refused the key
UNVERIFIED = "unverified"       # could not tell — network, or no listing endpoint

# How many alternatives to show when a model is missing.
_MAX_ALTERNATIVES = 12

# Words that name the vendor or a release stage rather than the model family.
_NOT_FAMILY = frozenset({"gpt", "claude", "gemini", "models", "latest", "preview", "exp"})
_DATE_SUFFIX_RE = re.compile(r"-(\d{4}-\d{2}-\d{2}|\d{8})$")

# OpenAI's model list includes embedding, speech, image and legacy completion
# models, none of which can drive a tool-calling chat loop. Kept out of the
# alternatives offered for a missing model.
_NOT_CHAT_RE = re.compile(
    r"embedding|tts|whisper|transcribe|dall-e|image|audio|realtime|moderation|"
    r"search|babbage|davinci|-instruct|^chat-latest$", re.IGNORECASE)


class ModelCheck(NamedTuple):
    status: str
    model: str
    message: str = ""
    # Models the key can use, same family first. Filled when the model is missing.
    alternatives: tuple[str, ...] = ()
    # A newer model of the same family, when one is offered. Never applied.
    newer: str | None = None

    @property
    def blocks_start(self) -> bool:
        """Whether starting would only fail on the first question."""
        return self.status in (MISSING, KEY_REJECTED)


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def base_id(model: str) -> str:
    """The model without a dated-snapshot suffix (`…-2026-09-22`, `…-20251001`)."""
    model = model.strip()
    if model.startswith("models/"):
        model = model[len("models/"):]
    return _DATE_SUFFIX_RE.sub("", model)


def family(model: str) -> str:
    """The family a model belongs to: its name words, without version numbers.

    `gpt-5.6-luna` and `gpt-6-luna` → `luna`; `claude-haiku-4-5` → `haiku`;
    `gemini-3.5-flash-lite` → `flash-lite`.
    """
    words = [w for w in re.split(r"[-.]", base_id(model).lower())
             if w.isalpha() and w not in _NOT_FAMILY]
    return "-".join(words)


def newer_in_family(model: str, catalog: dict[str, Any]) -> str | None:
    """The newest model of `model`'s family, if it is newer than `model`.

    `catalog` maps model id → creation time (anything orderable; None when
    unknown). Dated snapshots count as their undated name, so a fresh snapshot
    of the configured model is not reported as a different, newer one.
    """
    fam = family(model)
    if not fam:
        return None
    mine = base_id(model)
    created: dict[str, Any] = {}
    for mid, when in catalog.items():
        if when is None or family(mid) != fam:
            continue
        name = base_id(mid)
        if name not in created or when > created[name]:
            created[name] = when
    if not created:
        return None
    newest = max(created, key=lambda name: created[name])
    if newest == mine:
        return None
    # Only "newer" when the configured model's own date is known and older.
    if mine in created and created[newest] <= created[mine]:
        return None
    return newest


def alternatives(model: str, ids: Iterable[str],
                 created: dict[str, Any] | None = None) -> tuple[str, ...]:
    """Models to offer instead of a missing one: undated, same family first,
    newest first within the family when release dates are known."""
    fam = family(model)
    names = sorted({base_id(i) for i in ids if i})
    when: dict[str, Any] = {}
    for mid, t in (created or {}).items():
        name = base_id(mid)
        if t is not None and (name not in when or t > when[name]):
            when[name] = t
    same = [n for n in names if family(n) == fam]
    if fam and all(n in when for n in same):
        same.sort(key=lambda n: when[n], reverse=True)
    other = [n for n in names if family(n) != fam]
    return tuple((same + other)[:_MAX_ALTERNATIVES])


def _contains(model: str, ids: Iterable[str]) -> bool:
    wanted = model.strip()
    return any(i == wanted or base_id(i) == wanted for i in ids)


# ---------------------------------------------------------------------------
# Per provider
# ---------------------------------------------------------------------------


def check_openai(model: str | None = None, api_key: str | None = None,
                 base_url: str | None = None) -> ModelCheck:
    from .config import OPENAI_BASE_URL, OPENAI_MODEL
    from .providers.openai import probe

    model = (model or os.environ.get("OPENAI_MODEL") or OPENAI_MODEL).strip()
    info = probe(base_url=base_url, api_key=api_key)
    if not info["reachable"]:
        status = KEY_REJECTED if "401" in info["error"] else UNVERIFIED
        return ModelCheck(status, model, info["error"])

    ids = info["models"]
    if not ids:
        # Many OpenAI-compatible servers do not list their models.
        base = (base_url or os.environ.get("OPENAI_BASE_URL") or OPENAI_BASE_URL).rstrip("/")
        return ModelCheck(UNVERIFIED, model, f"{base} does not list its models.")
    created = info.get("created") or {}
    if not _contains(model, ids):
        chat = [i for i in ids if not _NOT_CHAT_RE.search(i)]
        return ModelCheck(MISSING, model,
                          f"`{model}` is not offered to this key.",
                          alternatives=alternatives(model, chat, created))
    newer = newer_in_family(model, created)
    return ModelCheck(OK, model, newer=newer)


def check_anthropic(model: str | None = None, api_key: str | None = None) -> ModelCheck:
    import anthropic

    from .config import ANTHROPIC_MODEL
    from .providers.claude import AnthropicProvider

    model = (model or os.environ.get("ANTHROPIC_MODEL") or ANTHROPIC_MODEL).strip()
    provider = AnthropicProvider(model=model, api_key=api_key)
    if not provider.api_key.strip():
        return ModelCheck(KEY_REJECTED, model, "No API key set.")
    client = provider.client

    try:
        # `retrieve` resolves aliases (`claude-haiku-4-5` → its dated snapshot),
        # which a membership test on the list would not.
        client.models.retrieve(model)
        found = True
    except anthropic.NotFoundError:
        found = False
    except (anthropic.AuthenticationError, anthropic.PermissionDeniedError) as exc:
        return ModelCheck(KEY_REJECTED, model, f"The API key was rejected: {exc}")
    except Exception as exc:  # noqa: BLE001 — connection, timeout, 5xx
        return ModelCheck(UNVERIFIED, model, f"Could not reach the Anthropic API: {exc}")

    try:
        catalog = {m.id: m.created_at for m in client.models.list(limit=100)}
    except Exception:  # noqa: BLE001
        logger.debug("Could not list Anthropic models.", exc_info=True)
        catalog = {}

    if not found:
        return ModelCheck(MISSING, model, f"`{model}` is not offered to this key.",
                          alternatives=alternatives(model, catalog, catalog))
    return ModelCheck(OK, model, newer=newer_in_family(model, catalog))


def check_gemini(model: str | None = None, api_key: str | None = None) -> ModelCheck:
    from google import genai
    from google.genai import errors

    from .config import GEMINI_MODEL

    model = (model or os.environ.get("GEMINI_MODEL") or GEMINI_MODEL).strip()
    key = api_key if api_key is not None else os.environ.get("GEMINI_API_KEY", "")
    if not key.strip():
        return ModelCheck(KEY_REJECTED, model, "No API key set.")
    client = genai.Client(api_key=key)

    try:
        client.models.get(model=model)
        found = True
    except errors.ClientError as exc:
        if exc.code == 404:
            found = False
        elif exc.code in (400, 401, 403) and "key" in str(exc).lower():
            return ModelCheck(KEY_REJECTED, model, f"The API key was rejected: {exc}")
        else:
            return ModelCheck(UNVERIFIED, model, f"Could not check the model: {exc}")
    except Exception as exc:  # noqa: BLE001
        return ModelCheck(UNVERIFIED, model, f"Could not reach the Gemini API: {exc}")

    if found:
        # Gemini's listing carries no release dates, so no "newer" hint.
        return ModelCheck(OK, model)
    try:
        ids = [m.name for m in client.models.list()
               if "generateContent" in (m.supported_actions or [])]
    except Exception:  # noqa: BLE001
        logger.debug("Could not list Gemini models.", exc_info=True)
        ids = []
    return ModelCheck(MISSING, model, f"`{model}` is not offered to this key.",
                      alternatives=alternatives(model, ids))


_CHECKS = {
    "openai": check_openai,
    "anthropic": check_anthropic,
    "google": check_gemini,
}


def check(provider: str, model: str | None = None, **kwargs) -> ModelCheck | None:
    """Check `provider`'s configured (or the given) model. None for Ollama.

    Never raises: anything unexpected is reported as UNVERIFIED, so a broken
    check can cost a warning but never the ability to start.
    """
    fn = _CHECKS.get(provider)
    if fn is None:
        return None
    try:
        return fn(model, **kwargs)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Model check for %s failed.", provider, exc_info=True)
        return ModelCheck(UNVERIFIED, model or "", f"The model check itself failed: {exc}")
