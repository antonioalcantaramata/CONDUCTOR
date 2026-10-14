"""
launch_ui.py — the launch screen's model choice: tiers, options, info cards.

Everything shown comes from the provider catalog (`provider_catalog.yaml`,
via agent/catalog.py) or from live discovery (agent/discovery/); nothing here
hard-codes a limit, a link or a policy sentence. The per-provider key and
settings flows that existed before stay in app.py; this module adds:

* the tier picker (Free / Paid / Local) and the option list;
* the model picker for options whose models are discovered or listed in the
  catalog (Gemini, NVIDIA Build, OpenRouter, DeepSeek);
* the info card shown for every option before Start: context window,
  effective size per request, estimated usage, rate limits, measured latency,
  sign-up needs, the compatibility check, the data notice and its sources;
* the settings form for the OpenAI-compatible presets;
* the sidebar's usage line.
"""

from __future__ import annotations

import os
import time

import httpx
import streamlit as st

from agent import catalog as cat
from agent import smoke_test, usage
from agent.discovery import cache as dcache
from agent.discovery import gemini as gemini_discovery
from agent.discovery import nvidia, openrouter
from agent.discovery import paid as paid_discovery

COMPAT = ("openrouter", "nvidia", "deepseek")


# ── shared facts ──────────────────────────────────────────────────────────────

def overhead_tokens() -> int:
    """Tokens every request carries (system prompt and tool schemas): the
    larger of the characters ÷ 4 estimate and what a provider has reported.
    Computed once per session."""
    if "_overhead_tokens" not in st.session_state:
        try:
            import json

            from agent.providers.schema import to_json_schema_tools
            from agent.system_prompt import get_system_prompt
            from agent.tool_schemas import TOOLS
            estimate = usage.overhead_tokens(get_system_prompt(), json.dumps(to_json_schema_tools(TOOLS)))
        except Exception:  # noqa: BLE001
            estimate = int(cat.catalog()["defaults"]["overhead_tokens_fallback"])
        st.session_state._overhead_tokens = max(estimate, usage.measured_prompt_tokens() or 0)
    return int(st.session_state._overhead_tokens)


def calls_per_question() -> float:
    return usage.calls_per_question(default=float(cat.catalog()["defaults"]["calls_per_question"]))


def _fmt(n) -> str:
    return "not published" if n in (None, "") else f"{int(n):,}"


def _md(text) -> str:
    """Text safe for st.markdown: a `$` would start LaTeX math."""
    return str(text).replace("$", "\\$")


def _first_sentence(text: str, limit: int = 120) -> str:
    text = (text or "").strip()
    end = text.find(". ")
    first = text if end < 0 else text[:end + 1]
    return first if len(first) <= limit else first[:limit].rsplit(" ", 1)[0] + " …"


def _key(option: cat.Option) -> str:
    return os.environ.get(option.get("key_env") or "", "").strip()


# ── tier and option ──────────────────────────────────────────────────────────

def tier_picker(key: str) -> cat.Option:
    """Tier first, then the option within it. Defaults to what the
    environment selects now."""
    data = cat.catalog()
    current = cat.option(cat.active_option_id())
    tiers = list(cat.TIERS)
    tier = st.segmented_control(
        "Tier", tiers, default=current.tier, key=f"{key}_tier", label_visibility="collapsed",
        format_func=lambda t: data["tiers"][t]["label"]) or current.tier
    st.caption(data["tiers"][tier]["caption"])
    opts = cat.options(tier=tier)
    ids = [o.id for o in opts]
    index = ids.index(current.id) if current.id in ids else 0

    def _label(oid: str) -> str:
        o = cat.option(oid)
        return o["display_name"] + ("  · recommended" if o.get("recommended") else "")

    chosen = st.radio("Option", ids, index=index, key=f"{key}_option_{tier}", format_func=_label,
                      label_visibility="collapsed",
                      captions=[_first_sentence(o.get("signup_note") or "") for o in opts])
    return cat.option(chosen)


# ── models ────────────────────────────────────────────────────────────────────

def model_choices(option: cat.Option, *, force: bool = False) -> dict:
    """{"models": [{id, label, context_window, badges, blocked}], "message",
    "error", "fetched_at"} for options with a model list; models empty for
    options whose model is set in their own settings (OpenAI, Anthropic,
    Ollama, a custom endpoint)."""
    disc = option["discovery"]
    out = {"models": [], "message": None, "error": None, "fetched_at": None}
    if disc == "openrouter":
        free = option.tier == "free"
        found = openrouter.discover(free_only=free, force=force)
        out["error"], out["fetched_at"] = found.get("error"), found.get("fetched_at")
        for m in found.get("models") or []:
            badges = []
            if m.get("expiration_date"):
                badges.append(f"retires {m['expiration_date']}")
            if m.get("data_warning"):
                badges.append("data-use wording (heuristic)")
            out["models"].append({"id": m["id"], "label": m["name"], "context_window": m["context_length"],
                                  "badges": badges, "blocked": None, "data_warning": m.get("data_warning"),
                                  "reasoning": bool(m.get("reasoning"))})
        if not out["models"]:
            out["message"] = (
                f"No {'free ' if free else ''}OpenRouter model qualifies right now (tool calling, text in and out, "
                f"at least {openrouter.min_context():,} tokens of context, not retiring within "
                f"{openrouter.grace_days()} days). Use Gemini API (free tier), NVIDIA Build or a local model.")
        return out
    if disc == "nvidia_allowlist":
        found = nvidia.discover(option.models, force=force)
        out["error"], out["fetched_at"] = found.get("error"), found.get("fetched_at")
        for m in found["available"]:
            blocked = ("Offered after a passing tool-calling test (use Test tool calling)."
                       if nvidia.needs_test(m) else None)
            out["models"].append({"id": m["id"], "label": m.get("label") or m["id"],
                                  "context_window": m["context_window"], "badges": [], "blocked": blocked,
                                  "note": m.get("note")})
        for m in found.get("candidates") or []:
            blocked = ("A newer model than the allowlist: offered after a passing tool-calling test."
                       if nvidia.needs_test(m) else None)
            out["models"].append({"id": m["id"], "label": m["label"], "context_window": None,
                                  "badges": ["newer · unconfirmed"], "blocked": blocked, "note": m["note"],
                                  "candidate": True})
        if found["missing"]:
            out["message"] = ("Not listed by NVIDIA right now, so not offered: "
                              + ", ".join(f"`{i}`" for i in found["missing"]) + ".")
        if not out["models"]:
            out["message"] = ("None of the allowlisted NVIDIA models is listed right now. Use Gemini API "
                              "(free tier), OpenRouter or a local model.")
        return out
    if not option.models and option.provider in ("openai", "anthropic", "google"):
        # A paid option: the provider's own list, read with the key.
        base = None
        if option.provider == "openai":
            base = (os.environ.get("OPENAI_BASE_URL") if option.get("custom_endpoint") else option["base_url"])
            if option.get("custom_endpoint") and not base:
                out["message"] = "Set the endpoint's base URL in the settings below to list its models."
                return out
        found = paid_discovery.discover(option.provider, _key(option), base, force=force)
        out["error"], out["fetched_at"] = found.get("error"), found.get("fetched_at")
        for m in found.get("models") or []:
            out["models"].append({"id": m["id"], "label": m["label"], "context_window": m.get("context_window"),
                                  "badges": [], "blocked": None})
        if not out["models"] and not _key(option):
            out["message"] = "Enter your API key below; the models it can use are then listed here."
        return out
    for m in option.models:
        badges = ["few questions a day"] if m.get("advanced") else []
        out["models"].append({"id": m["id"], "label": m.get("label") or m["id"],
                              "context_window": m["context_window"], "badges": badges, "blocked": None,
                              "advanced": bool(m.get("advanced"))})
    if option.provider == "google" and option.models and _key(option):
        # Newer Gemini models Google lists than the catalog knows: offered,
        # never switched to, and only after a passing tool-calling test.
        found = gemini_discovery.discover(_key(option), option.models, force=force)
        out["error"] = out["error"] or found.get("error")
        for m in found.get("candidates") or []:
            passed = (dcache.smoke_result("google", m["id"]) or {}).get("ok")
            out["models"].append({"id": m["id"], "label": m["label"], "context_window": m["context_window"],
                                  "badges": ["newer · unconfirmed"], "note": m["note"], "candidate": True,
                                  "blocked": None if passed else
                                  "A newer model than the catalog lists: offered after a passing tool-calling test."})
    return out


def model_picker(option: cat.Option, key: str) -> tuple[str | None, dict]:
    """The model choice for an option with a list; (None, choices) for the
    others. Includes Refresh for discovered lists."""
    if option.provider == "ollama":
        return None, {"models": []}             # its own picker, from the models you have pulled
    refreshable = option["discovery"] != "static" or bool(_key(option))
    force = st.session_state.pop(f"{key}_refresh", False)
    choices = model_choices(option, force=force)
    models = choices["models"]
    if choices.get("error"):
        st.caption(f"⚠️ {choices['error']}")
    if not models:
        if option["discovery"] == "static" and not option.models:
            if choices.get("message"):
                st.caption(choices["message"])
            return None, choices                # a paid option without a key yet: its setup form asks
        st.warning(choices.get("message") or "No model is available.")
        return None, choices
    ids = [m["id"] for m in models]
    current = os.environ.get(option.get("model_env") or "") or option.get("default_model")
    if option.get("default_model") and option.get("tier_env") and \
            cat.active_option_id() != option.id:
        current = option.get("default_model")
    index = ids.index(current) if current in ids else 0
    by_id = {m["id"]: m for m in models}

    def _label(mid: str) -> str:
        m = by_id[mid]
        extra = f" · {m['context_window'] // 1000:,}K" if m.get("context_window") else ""
        badges = f"  [{', '.join(m['badges'])}]" if m.get("badges") else ""
        return f"{m['label']}{extra}{badges}"

    cols = st.columns([5, 1], vertical_alignment="bottom")
    chosen = cols[0].selectbox("Model", ids, index=index, key=f"{key}_model_{option.id}", format_func=_label)
    if refreshable:
        if cols[1].button("Refresh", key=f"{key}_refresh_btn_{option.id}", help="Fetch the model list again."):
            st.session_state[f"{key}_refresh"] = True
            st.rerun()
    if choices.get("message"):
        st.caption(choices["message"])
    return chosen, choices


# ── the info card ────────────────────────────────────────────────────────────

def _deepseek_balance(option: cat.Option) -> float | None:
    """Remaining DeepSeek balance in USD (granted + topped up), cached 5 min."""
    api_key = _key(option)
    if not api_key:
        return None
    cached = st.session_state.get("_deepseek_balance")
    if cached and time.time() - cached[0] < 300:
        return cached[1]
    value = None
    try:
        resp = httpx.get(option["balance_url"], headers={"Authorization": f"Bearer {api_key}"}, timeout=10)
        if resp.status_code == 200:
            infos = (resp.json() or {}).get("balance_infos") or []
            usd = [i for i in infos if str(i.get("currency")).upper() == "USD"]
            if usd:
                value = float(usd[0].get("total_balance") or 0)
    except Exception:  # noqa: BLE001
        value = None
    st.session_state._deepseek_balance = (time.time(), value)
    return value


_CARD_CSS = """
<style>
.cd-tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(112px, 1fr)); gap: 0.5rem; margin: 0.4rem 0 0.75rem; }
[data-testid="stMarkdownContainer"]:has(.cd-tiles) { margin-bottom: 0 !important; }
.cd-tile { border: 1px solid rgba(46,49,64,0.10); border-radius: 10px; padding: 0.45rem 0.65rem; background: #fff; }
.cd-tile .v { font-size: 1.05rem; font-weight: 650; color: #22252f; line-height: 1.25; }
.cd-tile .k { font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.06em; color: #8a90a0; }
.cd-tile.bad .v { color: #b42318; }
.cd-head { display: flex; flex-wrap: wrap; align-items: baseline; gap: 0.45rem; }
.cd-head .name { font-weight: 650; font-size: 1.02rem; color: #22252f; }
.cd-badge { font-size: 0.7rem; padding: 0.08rem 0.5rem; border-radius: 999px; background: rgba(26,115,232,0.10); color: #1a73e8; }
.cd-badge.free { background: rgba(30,142,62,0.10); color: #1e8e3e; }
.cd-badge.warn { background: rgba(240,170,30,0.16); color: #8a5a00; }
.cd-model { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 0.8rem; color: #5c6272; }
.cd-est { font-size: 0.82rem; color: #6b7080; line-height: 1.45; margin: -0.2rem 0 0.2rem; }
.cd-sub { font-size: 0.82rem; color: #6b7080; margin: 0.1rem 0 0; }
.cd-notes { display: flex; flex-direction: column; gap: 0.4rem; margin: 0.7rem 0 0.2rem; }
.cd-note { display: flex; gap: 0.6rem; align-items: flex-start; font-size: 0.86rem; line-height: 1.45; color: #3c4152;
           padding: 0.5rem 0.7rem; border-radius: 8px; background: #f6f7fa; border-left: 3px solid #c9ced8; }
.cd-note .ic { font-size: 0.95rem; line-height: 1.3; }
.cd-note b { color: #22252f; font-weight: 650; }
.cd-note.info { background: rgba(26,115,232,0.06); border-left-color: #1a73e8; }
.cd-note.warn { background: rgba(240,170,30,0.10); border-left-color: #e0a106; }
.cd-note.bad { background: rgba(180,35,24,0.07); border-left-color: #b42318; }
.cd-foot { font-size: 0.74rem; color: #8a90a0; margin-top: 0.15rem; }
.cd-steps { border: 1px solid rgba(26,115,232,0.22); background: rgba(26,115,232,0.04); border-radius: 12px;
            padding: 0.75rem 0.95rem 0.6rem; margin: 0.2rem 0 0.6rem; }
.cd-steps .t { font-weight: 650; font-size: 0.95rem; color: #22252f; margin-bottom: 0.45rem; }
.cd-steps ol { margin: 0; padding-left: 0; list-style: none; counter-reset: s; }
.cd-steps li { counter-increment: s; position: relative; padding: 0 0 0.45rem 2rem; font-size: 0.87rem;
               line-height: 1.45; color: #3c4152; }
.cd-steps li::before { content: counter(s); position: absolute; left: 0; top: 0.05rem; width: 1.35rem; height: 1.35rem;
                       border-radius: 50%; background: #1a73e8; color: #fff; font-size: 0.75rem; font-weight: 650;
                       display: flex; align-items: center; justify-content: center; }
.cd-steps a { color: #1a73e8; text-decoration: none; font-weight: 600; }
.cd-steps .note { font-size: 0.76rem; color: #8a90a0; margin-top: 0.15rem; }
.cd-ready { display: flex; align-items: center; gap: 0.55rem; font-size: 0.88rem; color: #22252f; line-height: 1.45;
            padding: 0.55rem 0.8rem; border-radius: 10px; border: 1px solid rgba(30,142,62,0.28); background: rgba(30,142,62,0.06); }
.cd-ready .dot { width: 8px; height: 8px; border-radius: 50%; background: #1e8e3e; flex: none; box-shadow: 0 0 0 3px rgba(30,142,62,0.15); }
.cd-ready code { font-size: 0.8rem; background: rgba(46,49,64,0.06); padding: 0.05rem 0.3rem; border-radius: 4px; }
</style>
"""


def _short(n) -> str:
    """1,048,576 → 1M; 250,000 → 250K."""
    n = int(n)
    if n >= 1_000_000:
        return f"{n / 1_048_576:.0f}M" if n % 1_048_576 == 0 else f"{n / 1_000_000:.0f}M"
    return f"{n // 1000:,}K" if n >= 1000 else str(n)


def _tile(label: str, value: str, bad: bool = False) -> str:
    import html as _html
    return (f"<div class='cd-tile{' bad' if bad else ''}'><div class='v'>{_html.escape(value)}</div>"
            f"<div class='k'>{_html.escape(label)}</div></div>")


def info_card(option: cat.Option, model_id: str | None, choices: dict | None = None) -> bool:
    """Everything worth knowing before Start: the key figures as tiles, the
    data notice, and the details folded below. Returns whether the option can
    run CONDUCTOR at all (False disables Start)."""
    import html as _html
    overhead = overhead_tokens()
    calls = calls_per_question()
    entry = option.model(model_id) or next((m for m in (choices or {}).get("models") or []
                                           if m["id"] == model_id), None) or {}
    context = entry.get("context_window") or entry.get("context_length")
    limits = option.limits_for(model_id)
    ok, why = usage.compatibility(context, limits, overhead)

    balance = price = None
    if option.get("trial") and model_id:
        balance = _deepseek_balance(option)
        price = (option.model(model_id) or {}).get("price_input_usd_per_mtok")
    est = usage.estimate(limits, calls, overhead=overhead, option=option.raw, balance_usd=balance,
                         price_per_mtok=price)
    eff, eff_why = usage.effective_max_per_request(context, limits)
    lat = usage.latency(option.provider, model_id) if model_id else {}
    smoke = dcache.smoke_result(option.provider, model_id) if model_id else None

    # The four figures that decide a choice, at a glance.
    questions = {"per_day": lambda v: f"≈{v:,} / day", "trial_total": lambda v: f"≈{v:,} total",
                 "time_per_question": lambda v: f"≥{v} s each"}.get(est["kind"])
    tiles = [_tile("Questions", questions(est["value"]) if questions else
                   {"plan": "Your plan", "hardware": "Your hardware"}.get(est["kind"], "—"))]
    if context:
        tiles.append(_tile("Context window", _short(context)))
    if eff and eff_why:
        tiles.append(_tile("Max per request", _short(eff), bad=not ok))
    elif limits.get("rpm"):
        tiles.append(_tile("Requests / min", f"{int(limits['rpm']):,}"))
    if lat.get("last_s") is not None:
        tiles.append(_tile("Latency (measured)", f"{lat['last_s']:g} s"))
    elif smoke and smoke.get("latency_s") is not None:
        tiles.append(_tile("Tool test", ("✓ " if smoke.get("ok") else "✗ ") + f"{smoke['latency_s']:g} s",
                           bad=not smoke.get("ok")))

    if len(tiles) < 3:
        tiles.append(_tile("Prompt per request", f"~{_short(overhead)} tokens"))
    if len(tiles) < 3:
        tiles.append(_tile("Calls per question", f"~{calls:g}"))
    tier_badge = {"free": "<span class='cd-badge free'>Free</span>",
                  "paid": "<span class='cd-badge'>Paid</span>",
                  "local": "<span class='cd-badge free'>Local</span>"}[option.tier]
    extra_badges = "".join(f"<span class='cd-badge warn'>{_html.escape(b)}</span>" for b in entry.get("badges") or [])
    model_html = f"<span class='cd-model'>{_html.escape(model_id)}</span>" if model_id else ""

    est_text = est["text"] + (" Approximate." if est["kind"] not in ("plan", "hardware") else "")

    def note(kind: str, icon: str, text: str, label: str | None = None) -> str:
        body = (f"<b>{_html.escape(label)}</b> · " if label else "") + _html.escape(text).replace("$", "&#36;")
        return f"<div class='cd-note {kind}'><div>{body}</div></div>"

    notes = []
    if not ok:
        notes.append(note("bad", "⛔", why, "Not usable with CONDUCTOR"))
    if entry.get("blocked"):
        notes.append(note("warn", "🔒", entry["blocked"]))
    stale = cat.stale_days(option)
    if stale:
        notes.append(note("warn", "⏳", f"These figures were last verified {stale} days ago and may be out of date."))
    if entry.get("candidate"):
        notes.append(note("info", "🆕", "A newer model than the catalog lists: its free-tier status and limits are "
                                        "not confirmed."))
    notes.append(note("", "🛡️", option["data_policy_summary"], "Data"))
    if entry.get("data_warning"):
        notes.append(note("warn", "⚠️", f"This model's description mentions data use (\"{entry['data_warning']}\") "
                                        "— a keyword heuristic, not a verified policy."))
    if option.id == "openrouter_free" and os.environ.get("OPENROUTER_DENY_DATA_COLLECTION", "0") not in ("1", "true"):
        notes.append(note("warn", "⚠️", "Providers of free models may store your prompts and train on them. Use "
                                        "sample networks only, or turn on \"Exclude providers that collect data\" "
                                        "in the settings (free models may then stop working)."))
    if option.is_cloud_free:
        notes.append(note("info", "🧪", cat.catalog()["free_tier_notice"]))

    sub = f"<div class='cd-sub'>{_html.escape(entry['note'])}</div>" if entry.get("note") else ""
    with st.container(border=True, key=f"card_{option.id}"):
        st.markdown(_CARD_CSS + f"<div class='cd-head'><span class='name'>{_html.escape(option['display_name'])}"
                    f"</span>{tier_badge}{extra_badges}{model_html}</div>{sub}"
                    f"<div class='cd-tiles'>{''.join(tiles)}</div>"
                    f"<div class='cd-est'>{_html.escape(est_text).replace('$', '&#36;')}</div>"
                    f"<div class='cd-notes'>{''.join(notes)}</div>"
                    "<div style='height:16px'></div>", unsafe_allow_html=True)

        rows = [("Every request carries", f"~{overhead:,} tokens (system prompt and tool schemas; approximate)"),
                ("Calls per question", f"~{calls:g}" + (" (measured on this machine)" if usage.load_store().get("turns")
                                                       else " (default until measured)"))]
        if context:
            rows.append(("Context window", f"{int(context):,} tokens (advertised)"))
        if eff:
            rows.append(("Effective max per request",
                         f"{eff:,} tokens" + (f" — lower than advertised: {eff_why}" if eff_why else "")))
        if option.get("plan_limited"):
            rows.append(("Rate limits", "Limited by your plan."))
        elif option.get("hardware_limited"):
            rows.append(("Rate limits", "Limited by your hardware."))
        elif limits:
            parts = [f"{_fmt(limits.get('rpm'))} requests/min", f"{_fmt(limits.get('tpm'))} tokens/min",
                     f"{_fmt(limits.get('rpd'))} requests/day"]
            note = f" — {limits['note']}" if limits.get("note") else ""
            badge = {"official": "", "third_party": " (third-party figures)",
                     "unverified": " (not in the provider's docs; check your console)"}.get(limits.get("verified"), "")
            rows.append(("Rate limits", " · ".join(parts) + badge + note))
        if option.get("trial"):
            t = option["trial"]
            rows.append(("Trial credit", f"~{int(t['grant_tokens']) / 1e6:,.0f}M tokens, ~{t['validity_days']} days "
                                         f"({t.get('note') or 'unconfirmed'})"))
        if option.get("reset_note"):
            rows.append(("Daily reset", option["reset_note"]))
        if lat:
            rows.append(("Measured latency", f"first call {lat.get('first_s')} s, latest {lat.get('last_s')} s "
                                             "(on this machine)"))
        elif option.get("slow_note"):
            rows.append(("Latency", option["slow_note"]))
        if smoke:
            rows.append(("Tool-calling test", ("passed" if smoke.get("ok") else "failed")
                         + f" in {smoke.get('latency_s')} s" + (f" — {smoke['error']}" if smoke.get("error") else "")))
        if entry.get("note"):
            rows.append(("Model", entry["note"]))
        if option.get("signup_note"):
            rows.append(("Sign-up", option["signup_note"]))
        if option.get("hosting_note"):
            rows.append(("Hosting", option["hosting_note"]))
        if option.get("incompatible_models"):
            names = ", ".join(m.get("label") or m["id"] for m in option["incompatible_models"])
            tpm = option["incompatible_models"][0].get("limits", {}).get("tpm")
            rows.append(("Not offered", f"{names} — {_fmt(tpm)} tokens a minute on this tier, below one request."))
        links = [(label, option.get(k)) for label, k in (
            ("Data policy", "data_policy_url"), ("Privacy settings", "privacy_settings_url"),
            ("Guardrails", "guardrails_url"), ("Limits source", "source_url"))] + \
            [("Model card", entry.get("card_url"))] + [("Verification help", option.get("signup_source_url"))]
        with st.expander("Limits, data terms and sources"):
            st.markdown("\n".join(f"- **{label}:** {_md(value)}" for label, value in rows))
            if option.get("data_policy_details"):
                st.markdown(_md(option["data_policy_details"]))
            st.markdown(" · ".join(f"[{label}]({url})" for label, url in links if url))
        fetched = (choices or {}).get("fetched_at")
        age = ""
        if fetched:
            minutes = max(0, int((time.time() - float(fetched)) // 60))
            age = f" · model list fetched {minutes} min ago" if minutes < 120 else f" · model list fetched {minutes // 60} h ago"
        st.markdown(f"<div class='cd-foot'>Last verified {option['last_verified']}{age} · limits change often; "
                    "check your provider console.</div>", unsafe_allow_html=True)
    return ok


def ready_line(text: str) -> None:
    """A slim status line ("Ready — …") in place of a large green box.
    `text` is Markdown-ish: `code` spans and **bold** are kept."""
    import html as _html
    import re as _re
    body = _html.escape(text).replace("$", "&#36;")
    body = _re.sub(r"`([^`]+)`", r"<code>\1</code>", body)
    body = _re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", body)
    st.markdown(_CARD_CSS + f"<div class='cd-ready'><span class='dot'></span><span>{body}</span></div>",
                unsafe_allow_html=True)


def smoke_test_button(option: cat.Option, model_id: str | None, key: str) -> None:
    """Opt-in: one request with a dummy tool."""
    if not model_id or option.provider in ("ollama",) or not _key(option):
        return
    if st.button("Test tool calling", key=f"{key}_smoke_{option.id}", type="tertiary", icon=":material/science:",
                 help="Sends one small request with a dummy tool and checks the model calls it. "
                      "Uses one request from your quota."):
        with st.spinner(f"Testing {model_id}…"):
            result = smoke_test.run(option.provider, model_id)
        (st.success if result["ok"] else st.error)(
            f"Tool calling {'works' if result['ok'] else 'did not work'} ({result['latency_s']} s)"
            + (f": {result['error']}" if result.get("error") else "."))


# ── settings for the OpenAI-compatible presets ───────────────────────────────

def option_env(option: cat.Option, model_id: str | None) -> dict:
    """The environment that selects this option (and model)."""
    env = {"LLM_PROVIDER": option.provider, **(option.get("tier_env") or {})}
    if model_id and option.get("model_env"):
        env[option["model_env"]] = model_id
    return env


def thinking_levels(option: cat.Option, model_id: str | None, choices: dict | None = None) -> list[str]:
    """Thinking levels this model accepts ([] when none is documented)."""
    if option.provider == "nvidia":
        return list(((option.model(model_id) or {}).get("thinking") or {}).keys())
    if option.provider == "openrouter":
        entry = next((m for m in (choices or {}).get("models") or [] if m["id"] == model_id), None)
        if entry is None or entry.get("reasoning"):
            return list(option.get("thinking_values") or [])
    return []


def setup_steps(option: cat.Option) -> None:
    """"Get started" for someone without a key: numbered steps from the catalog."""
    import html as _html
    import re as _re
    steps = option.get("setup_steps") or []
    if not steps:
        return

    def fmt(text: str) -> str:
        body = _html.escape(text)
        return _re.sub(r"\[([^\]]+)\]\((https?://[^)\s]+)\)", r"<a href='\2' target='_blank'>\1</a>", body)
    items = "".join(f"<li>{fmt(x)}</li>" for x in steps)
    name = option["display_name"].split(" ·")[0]
    st.markdown(_CARD_CSS + f"<div class='cd-steps'><div class='t'>Get started with {_html.escape(name)}</div>"
                f"<ol>{items}</ol><div class='note'>The key is saved to <code>llm_agent/.env</code> on this "
                "machine only, and never sent anywhere but the provider.</div></div>", unsafe_allow_html=True)


def compat_form(option: cat.Option, model_id: str | None, key: str, write_env, *, submit: str,
                choices: dict | None = None) -> bool:
    """Key, privacy and timeout settings for OpenRouter / NVIDIA / DeepSeek.
    Without a key yet: the "Get started" steps, then just the key; the rest
    waits under Advanced. Returns True once saved."""
    stored = _key(option)
    if not stored:
        setup_steps(option)
    with st.form(f"{key}_compat_{option.id}"):
        api_key = st.text_input(
            f"{option['display_name'].split(' ·')[0]} API key", type="password",
            placeholder=f"stored …{stored[-4:]} — leave empty to keep" if stored else (option.get("key_prefix") or "paste it here"),
            help=f"Create one at {option.get('key_url')}")
        if stored:
            st.caption(f"Get a key: [{option.get('key_url')}]({option.get('key_url')})")
        extra: dict[str, str] = {}
        with st.expander("Advanced", expanded=bool(stored)):
            levels = thinking_levels(option, model_id, choices)
            thinking_env = f"{option.provider.upper()}_THINKING"
            if levels:
                current = (os.environ.get(thinking_env) or "").strip().lower()
                options_ = [""] + levels
                level = st.selectbox(
                    "Thinking", options_, index=options_.index(current) if current in options_ else 0,
                    key=f"{key}_thinking_{option.id}",
                    format_func=lambda v: "Model default" if not v else v,
                    help="How much the model reasons before answering. Less thinking is faster on free endpoints; "
                         "more can help with multi-step studies. Model default sends nothing.")
                extra[thinking_env] = level
            elif option.provider in ("nvidia", "openrouter"):
                st.caption("This model documents no thinking control; it uses its own default.")
            if option.provider == "openrouter" and option.tier == "free":
                deny = st.toggle("Exclude providers that collect data", key=f"{key}_deny_{option.id}",
                                 value=os.environ.get("OPENROUTER_DENY_DATA_COLLECTION", "0") in ("1", "true"),
                                 help="Sends provider.data_collection = \"deny\" with each request. When last "
                                      "tested (Oct 2026) this left no free model with an allowed provider, so it "
                                      "is off by default.")
                fallbacks = st.toggle("Server-side fallbacks (next two qualifying models)", key=f"{key}_fb_{option.id}",
                                      value=os.environ.get("OPENROUTER_FALLBACKS", "1") not in ("0", "false"))
                extra["OPENROUTER_DENY_DATA_COLLECTION"] = "1" if deny else "0"
                extra["OPENROUTER_FALLBACKS"] = "1" if fallbacks else "0"
                st.caption(f"Also consider your [OpenRouter privacy settings]({option.get('privacy_settings_url')}).")
            if option.get("timeout_env"):
                timeout = st.number_input(
                    "Request timeout (s)", min_value=30, max_value=1800, step=30,
                    value=int(float(os.environ.get(option["timeout_env"]) or option.get("timeout_default_s") or 180)),
                    key=f"{key}_timeout_{option.id}")
                extra[option["timeout_env"]] = str(int(timeout))
        saved = st.form_submit_button(submit, type="primary", width="stretch")
    if not saved:
        return False
    if not (api_key.strip() or stored):
        st.error(f"Enter an API key ({option['key_env']}).")
        return False
    if not model_id:
        st.error("Choose a model first.")
        return False
    updates = {**option_env(option, model_id), **extra}
    if api_key.strip():
        updates[option["key_env"]] = api_key.strip()
    write_env(updates)
    return True


def compat_settings(provider: str, key: str, write_env) -> bool:
    """The sidebar's settings for a preset: tier, model, key, extras."""
    opts = [o for o in cat.options() if o.provider == provider]
    active = cat.active_option_id()
    ids = [o.id for o in opts]
    index = ids.index(active) if active in ids else 0
    chosen = cat.option(st.radio("Tier", ids, index=index, key=f"{key}_ctier_{provider}", horizontal=True,
                                 format_func=lambda i: cat.catalog()["tiers"][cat.option(i).tier]["label"]))
    model_id, choices = model_picker(chosen, f"{key}_c")
    return compat_form(chosen, model_id, f"{key}_c", write_env, submit="Save settings", choices=choices)


# ── sidebar usage line ───────────────────────────────────────────────────────

def usage_line() -> str | None:
    """Calls made today against the daily limit (best-effort local count), or
    the remaining credit for a trial. None for paid and local options."""
    try:
        option = cat.option(cat.active_option_id())
    except Exception:  # noqa: BLE001
        return None
    if option.tier != "free" and not option.get("trial"):
        return None
    model = os.environ.get(option.get("model_env") or "") or option.get("default_model") or ""
    if option.provider == "openrouter" and _key(option):
        cached = st.session_state.get("_or_key_status")
        if not cached or time.time() - cached[0] > 60:
            cached = (time.time(), openrouter.key_status(_key(option)))
            st.session_state._or_key_status = cached
        daily = (cached[1] or {}).get("free_model_daily_requests") or {}
        if daily.get("limit"):
            return (f"OpenRouter free requests today: {daily.get('used', 0)} of {daily['limit']} "
                    f"({daily.get('remaining', '?')} left).")
    if option.get("trial"):
        balance = _deepseek_balance(option)
        price = (option.model(model) or {}).get("price_input_usd_per_mtok")
        est = usage.estimate(option.limits_for(model), calls_per_question(), overhead=overhead_tokens(),
                             option=option.raw, balance_usd=balance, price_per_mtok=price)
        return est["text"]
    limits = option.limits_for(model)
    rpd = limits.get("rpd")
    used = usage.calls_today(option.provider, model)
    if rpd:
        return f"Calls today on `{model}`: {used} of ~{int(rpd):,} (counted on this machine)."
    return f"Calls today on `{model}`: {used} (no published daily cap)." if model else None
