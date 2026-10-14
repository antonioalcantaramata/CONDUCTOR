"""
openai_compat.py — OpenRouter, NVIDIA Build and DeepSeek, through the OpenAI
provider's chat-completions path.

All three speak OpenAI's chat-completions dialect, so `OpenAIProvider` does
the work; this class only points it at the right place and adds what each
preset needs:

* settings come from the catalog option (base URL, key/model variables,
  timeout) and the preset's own variables — `OPENROUTER_API_KEY`,
  `NVIDIA_API_KEY`, `DEEPSEEK_API_KEY`, … — so several keys can be stored at
  once;
* always `/chat/completions`, never `/v1/responses`;
* failures are classified by `ratelimit.classify_http`: a per-minute limit
  backs off, a daily limit or spent credit stops with when it resets and what
  else to use, and a model that is gone is replaced by the next qualifying one
  (OpenRouter's live list, NVIDIA's validated allowlist) and the request
  retried once;
* OpenRouter (free tier): the next two qualifying models are sent as
  server-side fallbacks (`models`); providers that collect data can be
  excluded (`provider.data_collection = "deny"`, OPENROUTER_DENY_DATA_COLLECTION=1),
  off by default because it currently leaves no free model working;
* a timeout is retried once, then reported — it is not a missing model.
"""

from __future__ import annotations

import logging
import os
from typing import Callable

from .. import catalog as _catalog
from .. import ratelimit as rl
from ..config import OPENAI_TIMEOUT_S
from .base import ModelResponse
from .openai import OpenAIProvider

logger = logging.getLogger(__name__)

COMPAT_PROVIDERS = ("openrouter", "nvidia", "deepseek")
_LABELS = {"openrouter": "OpenRouter", "nvidia": "NVIDIA Build", "deepseek": "DeepSeek"}


def _flag(name: str, default: bool = True) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off")


def option_for(provider: str) -> _catalog.Option:
    """The catalog option this provider runs as now (its tier from the
    environment), or its first option."""
    active = _catalog.active_option_id()
    candidates = [o for o in _catalog.options() if o.provider == provider]
    for o in candidates:
        if o.id == active:
            return o
    return candidates[0]


class CompatProvider(OpenAIProvider):
    """An OpenAI-compatible preset (OpenRouter, NVIDIA Build, DeepSeek)."""

    timeout_retries = 1

    def __init__(self, provider: str, model: str | None = None) -> None:
        if provider not in COMPAT_PROVIDERS:
            raise ValueError(f"Not an OpenAI-compatible preset: {provider}")
        self.option = option_for(provider)
        opt = self.option
        prefix = provider.upper()
        super().__init__(
            model=model or os.environ.get(opt["model_env"]) or opt.get("default_model") or "",
            base_url=os.environ.get(f"{prefix}_BASE_URL") or opt["base_url"],
        )
        self.name = provider
        self.label = _LABELS[provider]
        self.key_env = opt["key_env"]
        self.model_env = opt["model_env"]
        timeout_env = opt.get("timeout_env") or f"{prefix}_TIMEOUT_S"
        self.timeout_s = float(os.environ.get(timeout_env) or opt.get("timeout_default_s") or OPENAI_TIMEOUT_S)
        self.timeout_env = timeout_env
        # Thinking is set per model from the catalog (NVIDIA) or OpenRouter's
        # `reasoning` parameter — see _thinking_fields. Never a raw passthrough.
        self.reasoning_effort = ""
        self.thinking = (os.environ.get(f"{prefix}_THINKING") or "").strip().lower()
        self.api = "chat"
        self.last_failure: rl.Failure | None = None

    # -- settings ------------------------------------------------------------

    @property
    def api_key(self) -> str:
        return os.environ.get(self.key_env, "")

    @property
    def uses_responses(self) -> bool:
        return False

    @property
    def free_tier(self) -> bool:
        return self.option.tier == "free"

    def describe(self) -> dict[str, str]:
        return {"preset": self.name, "option": self.option.id, "endpoint": "/v1/chat/completions",
                "reasoning": self.thinking or "model default"}

    def preflight(self) -> None:
        if not self.api_key.strip():
            raise RuntimeError(
                f"⚠️ **{self.key_env} is not set.**\n\n"
                f"Create a key at {self.option.get('key_url')}, then enter it in the model settings.")
        if not self.model:
            raise RuntimeError(f"⚠️ **No {self.label} model is selected.**\n\n"
                               "Choose one in the model settings.")

    # -- request extras --------------------------------------------------------

    def _discovered(self) -> list[dict]:
        from ..discovery import nvidia, openrouter
        if self.name == "openrouter":
            return openrouter.discover(free_only=self.free_tier).get("models") or []
        if self.name == "nvidia":
            return nvidia.discover(self.option.models).get("available") or []
        return self.option.models

    def _thinking_fields(self) -> dict:
        """Request fields for the chosen thinking level, or {} for the model's
        default. NVIDIA: the per-model fields the catalog takes from each model
        card (enable_thinking, medium_effort, reasoning_effort). OpenRouter: its
        documented `reasoning.effort`, only for models that list support."""
        import copy
        if not self.thinking:
            return {}
        if self.name == "nvidia":
            options = (self.option.model(self.model) or {}).get("thinking") or {}
            return copy.deepcopy(options.get(self.thinking) or {})
        if self.name == "openrouter" and self.thinking in (self.option.get("thinking_values") or []):
            entry = next((m for m in self._discovered() if m.get("id") == self.model), None)
            if entry is not None and entry.get("reasoning"):
                return {"reasoning": {"effort": self.thinking}}
        return {}

    def _extra_payload(self) -> dict:
        extra: dict = dict(self._thinking_fields())
        if self.name == "openrouter":
            if self.free_tier and _flag("OPENROUTER_FALLBACKS"):
                from ..discovery import openrouter
                backups = openrouter.fallbacks(self._discovered(), self.model, 2)
                if backups:
                    extra["models"] = [self.model, *backups]
            # Off by default: tested Oct 14, 2026, "deny" left no free model
            # with an allowed provider. The card warns instead.
            if self.free_tier and _flag("OPENROUTER_DENY_DATA_COLLECTION", default=False):
                extra["provider"] = {"data_collection": "deny"}
        return extra

    def _extra_headers(self) -> dict:
        if self.name == "openrouter":
            return {"X-Title": "CONDUCTOR"}
        return {}

    def _timeout_message(self) -> str:
        tries = getattr(self, "_timeouts", 1)
        tried = "once" if tries == 1 else "twice" if tries == 2 else f"{tries} times"
        return (f"⚠️ **{self.label} did not answer within {self.timeout_s:.0f} s** (tried {tried}).\n\n"
                f"This is a timeout, not a missing model. Free endpoints can be slow — raise "
                f"`{self.timeout_env}` in `llm_agent/.env`, or try again in a moment.")

    # -- errors ----------------------------------------------------------------

    def _describe_http_error(self, resp) -> tuple[str, bool, int]:
        failure = rl.classify_http(resp.status_code, resp.text, resp.headers, self.name)
        self.last_failure = failure
        detail = failure.detail.strip()[:600]
        reset = rl.describe_reset(failure.resets_at)
        others = rl.alternatives_text(self.option.id)
        attempt = getattr(self, "_attempt", 0)
        if failure.kind == rl.PER_MINUTE:
            wait = int(rl.backoff(attempt, failure.retry_after_s))
            return (f"⚠️ **{self.label} rate limit (per minute).** Still limited after several waits.\n\n"
                    f"{detail}", True, wait)
        if failure.kind == rl.SERVER:
            return (f"⚠️ **{self.label} is unavailable (HTTP {resp.status_code}).**\n\n{detail}", True,
                    int(rl.backoff(attempt, failure.retry_after_s, base=10)))
        if failure.kind == rl.PER_DAY:
            extra = ""
            if self.name == "openrouter":
                extra = (" Buying $10 of OpenRouter credit once raises the free-model limit from 50 to "
                         "1,000 requests a day.")
            when = f" It resets {reset}." if reset else ""
            return (f"⚠️ **{self.label}'s daily request limit is used up.**{when}{extra}\n\n{others}", False, 0)
        if failure.kind == rl.QUOTA:
            if self.name == "deepseek":
                text = ("⚠️ **Your DeepSeek credit is used up.** Top up at https://platform.deepseek.com, "
                        "or use NVIDIA Build, which serves the same model (DeepSeek V4.1 Flash) on a free "
                        "endpoint.")
            else:
                text = (f"⚠️ **{self.label} reports no remaining credit (HTTP {resp.status_code}).** "
                        "A negative balance blocks even free models.")
            return (f"{text}\n\n{detail}\n\n{others}", False, 0)
        if failure.kind == rl.DATA_POLICY:
            return ("⚠️ **No provider of this model meets your data policy.**\n\n"
                    "CONDUCTOR asks OpenRouter for providers that do not collect data. Turn that off in the "
                    "model settings (\"Exclude providers that collect data\"), allow more providers at "
                    f"{self.option.get('privacy_settings_url')}, or pick another model.\n\n{detail}",
                    False, 0)
        if failure.kind == rl.MODEL_GONE:
            what = "does not offer" if resp.status_code == 403 else "no longer serves"
            return (f"⚠️ **{self.label} {what} `{self.model}` to this key.**\n\n{detail}", False, 0)
        if failure.kind == rl.AUTH:
            return (f"⚠️ **{self.label} rejected the API key (HTTP {resp.status_code}).**\n\n"
                    f"Check `{self.key_env}` in the model settings.\n\n{detail}", False, 0)
        return (f"⚠️ **{self.label} error (HTTP {resp.status_code}).**\n\n{detail}", False, 0)

    # -- main entry point ------------------------------------------------------

    def _next_model(self, failed: set[str]) -> str | None:
        from ..discovery import nvidia, openrouter
        if self.name == "openrouter":
            models = openrouter.discover(free_only=self.free_tier, force=True).get("models") or []
            return openrouter.next_model(models, failed)
        if self.name == "nvidia":
            available = nvidia.discover(self.option.models, force=True).get("available") or []
            return nvidia.next_model(available, failed)
        return None

    def chat(self, messages: list[dict], tools: list, system: str,
             on_event: Callable[[str, dict], None] | None = None) -> ModelResponse:
        failed: set[str] = set()
        for attempt in range(2):
            self.last_failure = None
            try:
                response = super().chat(messages, tools, system, on_event=on_event)
            except RuntimeError:
                failure = self.last_failure
                if attempt == 0 and failure is not None and failure.kind == rl.MODEL_GONE:
                    failed.add(self.model)
                    replacement = self._next_model(failed)
                    if replacement:
                        old, self.model = self.model, replacement
                        os.environ[self.model_env] = replacement
                        logger.warning("%s: %s is gone; switching to %s.", self.label, old, replacement)
                        if on_event:
                            on_event("model_switch", {"from": old, "to": replacement,
                                                      "reason": f"{self.label} no longer serves {old}"})
                        continue
                raise
            served = str((response.raw or {}).get("model") or "") if isinstance(response.raw, dict) else ""
            if served and on_event and self.name == "openrouter" \
                    and not served.split(":")[0].startswith(self.model.split(":")[0]):
                on_event("model_switch", {"from": self.model, "to": served, "temporary": True,
                                          "reason": "OpenRouter answered with a fallback model"})
            return response
        raise RuntimeError("Unexpected exit from model recovery")  # pragma: no cover
