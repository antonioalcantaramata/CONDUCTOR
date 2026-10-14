"""
usage.py — what one CONDUCTOR question costs, and what a tier allows.

Three facts drive every estimate on the launch screen's cards:

* **Prompt overhead.** Every model call sends the system prompt and all tool
  schemas. `overhead_tokens` estimates it at characters ÷ 4 (labelled
  approximate; against Gemini's own token counter it came within 2 %), and
  the providers' reported prompt tokens refine it as calls are made.
* **Calls per question.** One question is several calls (reasoning, then a
  call per round of tools). The loop records each question's count here, and
  the running average is persisted locally; before any data it is the
  catalog default (4), and `CONDUCTOR_CALLS_PER_QUESTION` overrides it.
* **The tier's binding limit.** Usually requests per day or tokens per minute
  on free tiers, not the advertised context window.

The store (`usage.json` in the cache folder) also keeps calls made today per
provider and model — a best-effort local count for the sidebar — and the
latency of the first and latest call per model.

Pure functions plus one small JSON file; no Streamlit, no HTTP.
"""

from __future__ import annotations

import datetime as _dt
import json
import math
import os
import pathlib
import tempfile
from typing import Any

DEFAULT_CALLS_PER_QUESTION = 4.0
_HISTORY_GROWTH = 1.15        # later calls carry tool results: a modest allowance


def cache_dir() -> pathlib.Path:
    """Where CONDUCTOR keeps local, machine-specific state (gitignored)."""
    path = os.environ.get("CONDUCTOR_CACHE_DIR")
    return pathlib.Path(path) if path else pathlib.Path(__file__).resolve().parent.parent / ".cache"


def _store_path() -> pathlib.Path:
    return cache_dir() / "usage.json"


def load_store() -> dict:
    try:
        data = json.loads(_store_path().read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def save_store(store: dict) -> None:
    path = _store_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".usage", suffix=".json")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(store, fh, indent=1, sort_keys=True)
        os.replace(tmp, path)
    except OSError:
        pass                      # a local convenience: never worth failing a turn over


def _key(provider: str, model: str) -> str:
    return f"{provider}|{model}"


def _today() -> str:
    return _dt.date.today().isoformat()


# ── overhead ──────────────────────────────────────────────────────────────────

def overhead_tokens(system_prompt: str, tools_json: str) -> int:
    """Approximate tokens of the fixed part of every call: characters ÷ 4."""
    return math.ceil((len(system_prompt or "") + len(tools_json or "")) / 4)


def measured_prompt_tokens(store: dict | None = None) -> int | None:
    """The largest prompt a provider has reported for a first call of a
    question — the overhead plus a short question, measured rather than
    estimated. None until a provider has reported one."""
    store = load_store() if store is None else store
    values = [v.get("first") for v in (store.get("prompt_tokens") or {}).values() if v.get("first")]
    return max(values) if values else None


def prompt_tokens_from(raw: Any) -> int | None:
    """Prompt tokens a provider reported for one call, whatever its shape:
    OpenAI-style `usage.prompt_tokens` / `input_tokens`, Gemini
    `usage_metadata.prompt_token_count`, Anthropic `usage.input_tokens`, Ollama
    `prompt_eval_count`. None when it reported nothing."""
    if raw is None:
        return None
    if isinstance(raw, dict):
        usage = raw.get("usage") or {}
        for key in ("prompt_tokens", "input_tokens"):
            if isinstance(usage.get(key), int):
                return usage[key]
        if isinstance(raw.get("prompt_eval_count"), int):
            return raw["prompt_eval_count"]
        return None
    meta = getattr(raw, "usage_metadata", None)
    if meta is not None and isinstance(getattr(meta, "prompt_token_count", None), int):
        return meta.prompt_token_count
    usage = getattr(raw, "usage", None)
    if usage is not None and isinstance(getattr(usage, "input_tokens", None), int):
        return usage.input_tokens
    return None


# ── calls per question ───────────────────────────────────────────────────────

def calls_per_question(store: dict | None = None, default: float = DEFAULT_CALLS_PER_QUESTION) -> float:
    """The running average of model calls per question on this machine.
    `CONDUCTOR_CALLS_PER_QUESTION` overrides it; with no data, `default`."""
    override = os.environ.get("CONDUCTOR_CALLS_PER_QUESTION")
    if override:
        try:
            return max(1.0, float(override))
        except ValueError:
            pass
    store = load_store() if store is None else store
    turns = store.get("turns") or {}
    n, calls = turns.get("n") or 0, turns.get("calls") or 0
    return round(calls / n, 2) if n else float(default)


def record_turn(provider: str, model: str, n_calls: int, prompt_tokens: list[int] | None = None,
                store: dict | None = None) -> dict:
    """Add one answered question: its number of model calls, the prompt tokens
    the provider reported for each, and today's count per provider/model."""
    own = store is None
    store = load_store() if own else store
    if n_calls > 0:
        turns = store.setdefault("turns", {"n": 0, "calls": 0})
        turns["n"] = int(turns.get("n") or 0) + 1
        turns["calls"] = int(turns.get("calls") or 0) + int(n_calls)
        record_calls(provider, model, n_calls, store=store)
    tokens = [t for t in (prompt_tokens or []) if isinstance(t, int) and t > 0]
    if tokens:
        entry = store.setdefault("prompt_tokens", {}).setdefault(_key(provider, model), {})
        entry["first"] = tokens[0]
        entry["max"] = max(tokens + [int(entry.get("max") or 0)])
        entry["mean"] = round(sum(tokens) / len(tokens))
    if own:
        save_store(store)
    return store


def record_calls(provider: str, model: str, n: int = 1, store: dict | None = None) -> dict:
    """Count calls made today (any kind: questions, advisor, smoke tests)."""
    own = store is None
    store = load_store() if own else store
    daily = store.setdefault("daily", {})
    for day in [d for d in daily if d != _today()]:
        daily.pop(day)                       # only today matters
    today = daily.setdefault(_today(), {})
    today[_key(provider, model)] = int(today.get(_key(provider, model)) or 0) + int(n)
    if own:
        save_store(store)
    return store


def count_call(provider) -> None:
    """Count one model call made outside a question (the advisor's
    suggestions, discussion and decision notes) against today's total.
    Never raises: a convenience must not cost the call it counts."""
    try:
        record_calls(getattr(provider, "name", "unknown"), getattr(provider, "model", ""), 1)
    except Exception:  # noqa: BLE001
        pass


def calls_today(provider: str, model: str, store: dict | None = None) -> int:
    store = load_store() if store is None else store
    return int(((store.get("daily") or {}).get(_today()) or {}).get(_key(provider, model)) or 0)


def record_latency(provider: str, model: str, seconds: float, store: dict | None = None) -> dict:
    """Latency of a call: the first one per model is kept apart, since slow
    free endpoints are slowest on the first."""
    own = store is None
    store = load_store() if own else store
    entry = store.setdefault("latency", {}).setdefault(_key(provider, model), {})
    stamp = _dt.datetime.now().isoformat(timespec="seconds")
    if "first_s" not in entry:
        entry["first_s"], entry["first_at"] = round(float(seconds), 1), stamp
    entry["last_s"], entry["last_at"] = round(float(seconds), 1), stamp
    if own:
        save_store(store)
    return store


def latency(provider: str, model: str, store: dict | None = None) -> dict:
    store = load_store() if store is None else store
    return dict((store.get("latency") or {}).get(_key(provider, model)) or {})


# ── what a tier allows ───────────────────────────────────────────────────────

def effective_max_per_request(context_window: int | None, limits: dict) -> tuple[int | None, str | None]:
    """The largest prompt one request can carry on this tier, and why when it
    is below the advertised context window."""
    candidates = [(context_window, None)]
    if limits.get("per_request_input_cap"):
        candidates.append((int(limits["per_request_input_cap"]), "the tier caps input per request"))
    if limits.get("tpm"):
        candidates.append((int(limits["tpm"]), "one request cannot exceed the tokens-per-minute limit"))
    known = [(v, why) for v, why in candidates if v]
    if not known:
        return None, None
    value, why = min(known, key=lambda c: c[0])
    return value, (why if context_window and value < context_window else None)


def compatibility(context_window: int | None, limits: dict, overhead: int) -> tuple[bool, str | None]:
    """Whether CONDUCTOR can run here at all: every request carries `overhead`
    tokens, so a smaller window, per-request cap or tokens-per-minute limit
    fails on the first call."""
    checks = (
        (context_window, "context window"),
        (limits.get("per_request_input_cap"), "per-request input cap"),
        (limits.get("tpm"), "tokens-per-minute limit"),
    )
    for value, what in checks:
        if value and int(value) < overhead:
            return False, (f"Its {what} of {int(value):,} tokens is below the ~{overhead:,} tokens every "
                           "CONDUCTOR request carries (system prompt and tool schemas), so the first "
                           "request would fail.")
    return True, None


def tokens_per_question(overhead: int, calls: float, store: dict | None = None) -> int:
    """Approximate input tokens one question consumes."""
    store = load_store() if store is None else store
    means = [v.get("mean") for v in (store.get("prompt_tokens") or {}).values() if v.get("mean")]
    per_call = max(means) if means else overhead * _HISTORY_GROWTH
    return int(per_call * calls)


def estimate(limits: dict, calls: float, *, overhead: int, option: dict | None = None,
             balance_usd: float | None = None, price_per_mtok: float | None = None,
             credit: bool = False, store: dict | None = None) -> dict:
    """What the tier allows, in questions. Always approximate.

    Returns {"kind", "value", "text"} with kind one of: per_day (requests per
    day ÷ calls per question), time_per_question (only a per-minute limit),
    trial_total (credit ÷ tokens per question), plan (paid: your plan),
    hardware (local), unknown.
    """
    option = option or {}
    calls = max(1.0, float(calls))
    if option.get("tier") == "local" or option.get("hardware_limited"):
        return {"kind": "hardware", "value": None, "text": "Limited by your hardware, not by a quota."}
    trial = option.get("trial")
    if trial:
        per_q = tokens_per_question(overhead, calls, store)
        if balance_usd is not None and price_per_mtok:
            tokens = balance_usd / price_per_mtok * 1_000_000
            basis = f"your remaining credit (${balance_usd:,.2f})"
        else:
            tokens = float(trial.get("grant_tokens") or 0)
            basis = f"the full signup grant (~{tokens / 1e6:,.0f}M tokens, unconfirmed)"
        n = int(tokens // per_q) if per_q else 0
        return {"kind": "trial_total", "value": n,
                "text": f"About {n:,} questions in total from {basis}, at ~{per_q:,} input tokens per question."}
    if option.get("plan_limited"):
        return {"kind": "plan", "value": None, "text": "Limited by your plan."}
    rpd = limits.get("rpd_with_credit") if credit and limits.get("rpd_with_credit") else limits.get("rpd")
    rpm = limits.get("rpm")
    if rpd:
        n = int(rpd // calls)
        text = f"About {n:,} questions a day ({int(rpd):,} requests ÷ ~{calls:g} calls per question)."
        if rpm and calls / rpm * 60 >= 20:
            text += f" At most one every ~{math.ceil(calls / rpm * 60)} s ({int(rpm)} requests a minute)."
        tpm = limits.get("tpm")
        if tpm and overhead:
            per_call = overhead * _HISTORY_GROWTH
            by_tokens = tpm / per_call / calls
            by_requests = (rpm / calls) if rpm else by_tokens
            if by_tokens < by_requests:
                text += (f" Tokens per minute are the tighter limit: about {by_tokens:.1f} questions a minute "
                         f"({int(tpm):,} tokens ÷ ~{int(per_call * calls):,} per question).")
        return {"kind": "per_day", "value": n, "text": text}
    if rpm:
        seconds = math.ceil(calls / rpm * 60)
        return {"kind": "time_per_question", "value": seconds,
                "text": f"No daily cap published; at {int(rpm)} requests a minute a question takes at least ~{seconds} s."}
    return {"kind": "unknown", "value": None, "text": "Limits not published."}
