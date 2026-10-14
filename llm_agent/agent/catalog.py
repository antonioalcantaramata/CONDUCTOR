"""
catalog.py — load and validate `provider_catalog.yaml`.

The catalog is the single source of truth for what the launch screen offers:
tiers, options, limits, links and data-use notes. This module only loads it,
checks it, and answers lookups; it holds no provider data of its own, so a
limit that changes is edited in the YAML, never here.

Validation is strict on purpose. A limit without a source, or a tier the UI
does not know, is a mistake that would otherwise reach an operator as a
confident number on a card.

Pure: no Streamlit, no HTTP.
"""

from __future__ import annotations

import functools
import os
import pathlib
from dataclasses import dataclass
from typing import Any

import yaml

CATALOG_PATH = pathlib.Path(__file__).resolve().parent.parent / "provider_catalog.yaml"

TIERS = ("free", "paid", "local")
API_STYLES = ("gemini", "anthropic", "openai_chat", "openai_responses", "ollama")
DISCOVERY = ("static", "openrouter", "nvidia_allowlist")
PROVIDERS = ("google", "openai", "anthropic", "ollama", "openrouter", "nvidia", "deepseek")
VERIFIED = ("official", "third_party", "unverified")
_LIMIT_KEYS = ("rpm", "tpm", "rpd", "rpd_with_credit", "per_request_input_cap", "credit_threshold_usd")
_REQUIRED = ("id", "display_name", "tier", "provider", "api_style", "discovery",
             "data_policy_summary", "last_verified")


class CatalogError(ValueError):
    """The catalog file is malformed: the message names the entry and field."""


@dataclass(frozen=True)
class Option:
    """One launch-screen option (a provider at a tier). `raw` keeps every
    field of the YAML entry; the properties are the ones code relies on."""

    raw: dict

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)

    @property
    def id(self) -> str:
        return self.raw["id"]

    @property
    def tier(self) -> str:
        return self.raw["tier"]

    @property
    def provider(self) -> str:
        return self.raw["provider"]

    @property
    def models(self) -> list[dict]:
        return list(self.raw.get("models") or [])

    @property
    def is_cloud_free(self) -> bool:
        """Free and hosted: the tier the data notice is for."""
        return self.tier == "free"

    def model(self, model_id: str | None) -> dict | None:
        """The catalog entry for one of this option's models, if listed."""
        for m in self.models + list(self.raw.get("incompatible_models") or []):
            if m.get("id") == model_id:
                return m
        return None

    def limits_for(self, model_id: str | None) -> dict:
        """Limits that apply to a model on this option: the model's own when it
        has them (Gemini's are per model), else the account's."""
        entry = self.model(model_id) or {}
        return dict(entry.get("limits") or self.raw.get("account_limits") or {})


def _check_limits(where: str, limits: Any) -> None:
    if not isinstance(limits, dict):
        raise CatalogError(f"{where}: limits must be a mapping.")
    for key in _LIMIT_KEYS:
        value = limits.get(key)
        if value is not None and not (isinstance(value, (int, float)) and value >= 0):
            raise CatalogError(f"{where}: {key} must be a non-negative number or null, not {value!r}.")
    if limits.get("verified") not in VERIFIED:
        raise CatalogError(f"{where}: verified must be one of {VERIFIED}.")
    for key in ("source_url", "last_verified"):
        if not limits.get(key):
            raise CatalogError(f"{where}: every limit needs {key}.")


def validate(data: Any) -> dict:
    """Check the parsed YAML; return it unchanged or raise CatalogError."""
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise CatalogError("The catalog must be a mapping with schema_version: 1.")
    options = data.get("options")
    if not isinstance(options, list) or not options:
        raise CatalogError("The catalog has no options.")
    tiers = data.get("tiers") or {}
    if set(tiers) != set(TIERS):
        raise CatalogError(f"tiers must define exactly {TIERS}.")
    seen: set[str] = set()
    for i, opt in enumerate(options):
        if not isinstance(opt, dict):
            raise CatalogError(f"options[{i}] is not a mapping.")
        where = f"option {opt.get('id') or i}"
        for key in _REQUIRED:
            if not opt.get(key):
                raise CatalogError(f"{where}: missing {key}.")
        if opt["id"] in seen:
            raise CatalogError(f"{where}: duplicate id.")
        seen.add(opt["id"])
        if opt["tier"] not in TIERS:
            raise CatalogError(f"{where}: tier {opt['tier']!r} is not one of {TIERS}.")
        if opt["api_style"] not in API_STYLES:
            raise CatalogError(f"{where}: api_style {opt['api_style']!r} is not one of {API_STYLES}.")
        if opt["discovery"] not in DISCOVERY:
            raise CatalogError(f"{where}: discovery {opt['discovery']!r} is not one of {DISCOVERY}.")
        if opt["provider"] not in PROVIDERS:
            raise CatalogError(f"{where}: provider {opt['provider']!r} is not one of {PROVIDERS}.")
        if opt["provider"] not in ("ollama",) and not opt.get("key_env"):
            raise CatalogError(f"{where}: a hosted option needs key_env.")
        if opt["tier"] == "free" and not opt.get("data_policy_url"):
            raise CatalogError(f"{where}: free options need data_policy_url.")
        if opt.get("account_limits") is not None:
            _check_limits(f"{where} account_limits", opt["account_limits"])
        if opt["discovery"] == "nvidia_allowlist" and not opt.get("models"):
            raise CatalogError(f"{where}: an allowlist option needs models.")
        for m in list(opt.get("models") or []) + list(opt.get("incompatible_models") or []):
            mwhere = f"{where} model {m.get('id')}"
            if not m.get("id"):
                raise CatalogError(f"{where}: a model without id.")
            if not isinstance(m.get("context_window"), int) or m["context_window"] <= 0:
                raise CatalogError(f"{mwhere}: context_window must be a positive integer.")
            if m.get("tools") is not True:
                raise CatalogError(f"{mwhere}: CONDUCTOR needs tool calling (tools: true).")
            if m.get("limits") is not None:
                _check_limits(mwhere, m["limits"])
            thinking = m.get("thinking")
            if thinking is not None and not (isinstance(thinking, dict) and all(
                    isinstance(k, str) and isinstance(v, dict) for k, v in thinking.items())):
                raise CatalogError(f"{mwhere}: thinking must map level names to request fields.")
        chain = opt.get("fallback_chain") or []
        listed = {m.get("id") for m in opt.get("models") or []}
        for model_id in chain:
            if model_id not in listed:
                raise CatalogError(f"{where}: fallback model {model_id} is not among its models.")
        if opt.get("trial") is not None:
            _check_limits(f"{where} trial", {k: v for k, v in opt["trial"].items()
                                              if k not in ("grant_tokens", "validity_days")})
        steps = opt.get("setup_steps")
        if steps is not None and not (isinstance(steps, list) and steps and all(isinstance(x, str) for x in steps)):
            raise CatalogError(f"{where}: setup_steps must be a list of sentences.")
        if opt.get("tier_env") is not None and not isinstance(opt["tier_env"], dict):
            raise CatalogError(f"{where}: tier_env must be a mapping.")
    return data


def load(path: pathlib.Path | str | None = None) -> dict:
    """Parse and validate the catalog file."""
    path = pathlib.Path(path or os.environ.get("CONDUCTOR_PROVIDER_CATALOG") or CATALOG_PATH)
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise CatalogError(f"Provider catalog not found at {path}.") from exc
    except yaml.YAMLError as exc:
        raise CatalogError(f"Provider catalog {path} is not valid YAML: {exc}") from exc
    return validate(data)


@functools.lru_cache(maxsize=4)
def _cached(path: str) -> dict:
    return load(path)


def catalog() -> dict:
    """The validated catalog, read once per process."""
    return _cached(str(os.environ.get("CONDUCTOR_PROVIDER_CATALOG") or CATALOG_PATH))


def options(data: dict | None = None, tier: str | None = None) -> list[Option]:
    data = data or catalog()
    out = [Option(o) for o in data["options"]]
    return [o for o in out if tier is None or o.tier == tier]


def option(option_id: str, data: dict | None = None) -> Option:
    for o in options(data):
        if o.id == option_id:
            return o
    raise KeyError(option_id)


DEFAULT_STALE_DAYS = 60


def days_since_verified(option: "Option", today=None) -> int | None:
    """Days since the option's figures were last checked (None if unknown)."""
    import datetime as _dt
    value = option.get("last_verified")
    if isinstance(value, str):
        try:
            value = _dt.date.fromisoformat(value)
        except ValueError:
            return None
    if not isinstance(value, _dt.date):
        return None
    return ((today or _dt.date.today()) - value).days


def stale_days(option: "Option", today=None) -> int | None:
    """Days since verification when older than CATALOG_STALE_DAYS (default
    60), else None — the cue to re-check the sources."""
    try:
        limit = int(os.environ.get("CATALOG_STALE_DAYS") or DEFAULT_STALE_DAYS)
    except ValueError:
        limit = DEFAULT_STALE_DAYS
    days = days_since_verified(option, today)
    return days if days is not None and days > limit else None


def active_option_id(env: dict | None = None, data: dict | None = None) -> str:
    """Which option the current environment selects.

    `LLM_PROVIDER` picks the provider; a tier variable (GEMINI_TIER,
    OPENROUTER_TIER) picks among its options; for `openai`, a base
    URL other than OpenAI's means the custom-endpoint option. Existing `.env`
    files without tier variables land on the option they always meant: Gemini
    free, OpenRouter free, OpenAI.
    """
    env = os.environ if env is None else env
    provider = (env.get("LLM_PROVIDER") or "google").strip().lower()
    candidates = [o for o in options(data) if o.provider == provider]
    if not candidates:
        return options(data)[0].id
    if provider == "openai":
        base = (env.get("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        custom = base != "https://api.openai.com/v1"
        for o in candidates:
            if bool(o.get("custom_endpoint")) == custom:
                return o.id
    for o in candidates:
        tier_env = o.get("tier_env") or {}
        if tier_env and all((env.get(k) or "").strip().lower() == v for k, v in tier_env.items()):
            return o.id
    return candidates[0].id
