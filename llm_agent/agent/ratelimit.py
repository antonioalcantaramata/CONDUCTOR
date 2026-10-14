"""
ratelimit.py — what kind of failure a provider returned, and what to do next.

The providers' retry loops ask this module one question: is this a limit that
waiting clears (per minute), one that it does not (per day, an exhausted
quota or credit), a model that no longer exists, or something else? The
answer decides between backing off (with a visible "waiting for the rate
limit" status), switching to the next model (Gemini's fallback chain,
OpenRouter's and NVIDIA's next qualifying model), and stopping with a message
that says which limit was hit, when it resets if known, and what else to use.

What each provider tells apart (checked against their docs and live errors,
October 2026):

* Gemini — the 429 body lists quota violations; a quota ID containing
  "PerDay" is a daily quota, "PerMinute" a per-minute one. Retired models
  return 404 NOT_FOUND.
* OpenRouter — a platform 429 carries `X-RateLimit-Limit`/`-Reset`; free
  models allow 20 a minute and 50 (or 1,000) a day, so a limit above 20 is the
  daily one. 402 is a negative balance. "No endpoints found" means the model
  (or every endpoint allowed by the data policy) is gone.
* NVIDIA — 404 (often plain text "404 page not found") for a model it no
  longer serves; 429 for its per-minute limit.
* DeepSeek — 402 "run out of balance"; 429 when over the concurrency limit.

Pure functions.
"""

from __future__ import annotations

import datetime as _dt
import re
from dataclasses import dataclass
from zoneinfo import ZoneInfo

PER_MINUTE = "per_minute"
PER_DAY = "per_day"
QUOTA = "quota"                 # credit or billing quota exhausted
MODEL_GONE = "model_gone"
DATA_POLICY = "data_policy"     # no endpoint allowed by the privacy settings
TIMEOUT = "timeout"
AUTH = "auth"
SERVER = "server"
OTHER = "other"

PACIFIC = ZoneInfo("America/Los_Angeles")
_GONE_TEXT = re.compile(r"not found|does not exist|no endpoints found|not a valid model|"
                        r"deprecated|end of life|no longer (?:available|supported)|has been (?:removed|retired)|"
                        r"model_not_found|unknown model", re.IGNORECASE)


@dataclass(frozen=True)
class Failure:
    kind: str
    detail: str = ""
    retry_after_s: float | None = None
    resets_at: _dt.datetime | None = None
    limit_value: int | None = None
    metric: str | None = None       # e.g. "input_tokens" for a per-minute token cap

    @property
    def waits(self) -> bool:
        """Whether waiting and retrying can clear it."""
        return self.kind in (PER_MINUTE, SERVER, TIMEOUT)


def backoff(attempt: int, retry_after: float | None = None, base: float = 5.0, cap: float = 120.0) -> float:
    """Exponential backoff (5, 10, 20, … s, capped), never shorter than the
    server's own hint."""
    wait = min(cap, base * (2 ** max(0, attempt)))
    return max(wait, float(retry_after or 0))


def next_pacific_midnight(now: _dt.datetime | None = None) -> _dt.datetime:
    now = (now or _dt.datetime.now(tz=PACIFIC)).astimezone(PACIFIC)
    tomorrow = (now + _dt.timedelta(days=1)).date()
    return _dt.datetime.combine(tomorrow, _dt.time(0, 0), tzinfo=PACIFIC)


def describe_reset(when: _dt.datetime | None) -> str:
    """'at 09:00 your time (in 6 h 12 min)' — or '' when unknown."""
    if when is None:
        return ""
    local = when.astimezone()
    delta = when - _dt.datetime.now(tz=when.tzinfo or PACIFIC)
    minutes = max(0, int(delta.total_seconds() // 60))
    span = f"{minutes // 60} h {minutes % 60} min" if minutes >= 60 else f"{minutes} min"
    return f"at {local:%H:%M} your time (in {span})"


def _retry_delay(text: str) -> float | None:
    m = re.search(r"retry_?delay\W+(?:seconds:\s*)?(\d+(?:\.\d+)?)s?", text, re.IGNORECASE)
    return float(m.group(1)) if m else None


def classify_gemini(message: str, code: int | None = None, *, now: _dt.datetime | None = None) -> Failure:
    """A google-genai ClientError/ServerError message and HTTP code."""
    text = message or ""
    low = text.lower()
    if code == 404 or "not_found" in low or "is not found for api version" in low:
        return Failure(MODEL_GONE, detail=text)
    if code in (401, 403) or "api key not valid" in low or "permission_denied" in low:
        return Failure(AUTH, detail=text)
    if code == 429 or "resource_exhausted" in low or "quota" in low:
        quota_ids = re.findall(r"quotaId['\"]?\s*[:=]\s*['\"]([A-Za-z0-9_-]+)", text)
        daily = any("perday" in q.lower() for q in quota_ids) or "perday" in low or "per_day" in low \
            or "daily quota" in low
        value = None
        m = re.search(r"quotaValue['\"]?\s*[:=]\s*['\"]?(\d+)", text)
        if m:
            value = int(m.group(1))
        metric = "input_tokens" if "input_token" in low else "requests"
        if daily:
            return Failure(PER_DAY, detail=text, resets_at=next_pacific_midnight(now), limit_value=value,
                           metric=metric)
        return Failure(PER_MINUTE, detail=text, retry_after_s=_retry_delay(text), limit_value=value,
                       metric=metric)
    if code is not None and code >= 500:
        return Failure(SERVER, detail=text)
    return Failure(OTHER, detail=text)


def _header(headers, name: str) -> str | None:
    if headers is None:
        return None
    for key in (name, name.lower()):
        try:
            value = headers.get(key)
        except AttributeError:
            value = None
        if value is not None:
            return str(value)
    return None


def _reset_from_header(value: str | None) -> _dt.datetime | None:
    """X-RateLimit-Reset: epoch milliseconds (OpenRouter) or seconds."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number > 1e11:
        number /= 1000.0
    return _dt.datetime.fromtimestamp(number, tz=_dt.timezone.utc)


def classify_http(status: int, text: str, headers=None, provider: str = "openai") -> Failure:
    """An OpenAI-compatible HTTP error (OpenRouter, NVIDIA, DeepSeek, OpenAI)."""
    body = text or ""
    low = body.lower()
    retry_after = None
    try:
        retry_after = float(_header(headers, "retry-after"))
    except (TypeError, ValueError):
        pass
    if status == 401 or (status == 403 and re.search(r"api.?key|auth|credential|unauthori[sz]ed|token", low)):
        return Failure(AUTH, detail=body)
    if status == 403:
        # Forbidden for this model, not for the key: OpenRouter's "only
        # available on agentic harnesses", a region or plan restriction. For
        # CONDUCTOR the model is unavailable — move on to the next one.
        return Failure(MODEL_GONE, detail=body)
    if "matching your data policy" in low or "data policy" in low and "no endpoints" in low:
        return Failure(DATA_POLICY, detail=body)
    if status == 402 or "insufficient_quota" in low or "run out of balance" in low \
            or "insufficient balance" in low or "negative credit" in low:
        return Failure(QUOTA, detail=body)
    if status == 429:
        limit = _header(headers, "X-RateLimit-Limit")
        resets = _reset_from_header(_header(headers, "X-RateLimit-Reset"))
        try:
            limit_value = int(float(limit)) if limit is not None else None
        except ValueError:
            limit_value = None
        daily_text = re.search(r"per[- _]?day|daily|requests per day|free-models-per-day", low)
        if provider == "openrouter":
            # Free models allow 20 a minute and 50 / 1,000 a day: a limit above
            # 20 can only be the daily one (inference from the documented values).
            if daily_text or (limit_value is not None and limit_value > 20):
                return Failure(PER_DAY, detail=body, resets_at=resets, limit_value=limit_value)
            wait = retry_after
            if wait is None and resets is not None:
                wait = max(1.0, (resets - _dt.datetime.now(tz=_dt.timezone.utc)).total_seconds())
            return Failure(PER_MINUTE, detail=body, retry_after_s=wait, limit_value=limit_value)
        if daily_text:
            return Failure(PER_DAY, detail=body, resets_at=resets, limit_value=limit_value)
        return Failure(PER_MINUTE, detail=body, retry_after_s=retry_after, limit_value=limit_value)
    if status in (404, 410) or (status in (400, 422) and _GONE_TEXT.search(body)) or "model_not_found" in low:
        if "context" in low and "length" in low:
            return Failure(OTHER, detail=body)
        return Failure(MODEL_GONE, detail=body)
    if status >= 500:
        return Failure(SERVER, detail=body, retry_after_s=retry_after)
    return Failure(OTHER, detail=body)


def alternatives_text(current_option: str | None = None) -> str:
    """Where else to go when a free limit is spent."""
    options = [("gemini_free", "Gemini API (free tier)"), ("nvidia_free", "NVIDIA Build (free endpoints)"),
               ("openrouter_free", "OpenRouter (free models)"), ("ollama_local", "a local model (Ollama)")]
    names = [name for oid, name in options if oid != current_option]
    return "Other options: " + ", ".join(names) + " — switch in the sidebar's Model settings."
