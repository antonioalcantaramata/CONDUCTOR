"""
investigation_report.py — the detailed report of an investigation, to download.

The answer in the chat is short on purpose. This report is the whole of it,
built from the turn record only — the studies' own results, the agent's record
of its hypotheses, the checks — never from the model's prose beyond quoting
the answer itself:

- the question and the answer;
- each investigation: what was compared with what, the factors with their exact
  shares and verdicts, what is inside a group, the causes that interact,
  whether the main cause holds against other references, how much the agent's
  own hypotheses explain, the classical views when they were asked for, the
  levers to act on now;
- recurring causes and prevention checks, when they ran, and what the forecast
  says is coming (the look-ahead);
- the hypotheses the agent recorded (supported, ruled out, untested) and the
  tools that untested ones would need;
- what backs the answer: the evidence levels, the reading check, the answer
  checks; the studies that ran, with their arguments; the method and its limits.

Markdown, so it reads as plain text and renders anywhere.
"""

from __future__ import annotations

import datetime

from . import tool_registry as reg

_VERDICT = {"cause": "cause", "contributes": "contributes", "ruled_out": "ruled out", "counteracts": "counteracts"}
_RECORDED = {"supported": "Supported", "refuted": "Ruled out", "untested": "Not tested"}
_INVESTIGATION_TOOLS = {"investigate", "test_prevention", "look_ahead", "record_investigation"}


def has_investigation(record: dict | None) -> bool:
    """Whether a turn investigated anything (and so has a report)."""
    return isinstance(record, dict) and any(
        isinstance(e, dict) and e.get("name") in _INVESTIGATION_TOOLS and isinstance(e.get("result"), dict)
        and not e["result"].get("error") for e in record.get("tool_results") or [])


def _cell(value) -> str:
    return str(value if value is not None else "–").replace("|", "\\|").replace("\n", " ")


def _table(header: list[str], rows: list[list]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(_cell(c) for c in row) + " |" for row in rows]
    return out + [""]


def _num(value, unit: str = "", nd: int = 3) -> str:
    if value is None:
        return "–"
    try:
        return f"{float(value):.{nd}f}{(' ' + unit) if unit else ''}"
    except (TypeError, ValueError):
        return str(value)


def _investigation(result: dict, n: int) -> list[str]:
    out = [f"### {n}. {result.get('element', 'The grid')}", ""]
    if result.get("status") not in ("explained",):
        return out + [f"Status: **{result.get('status')}**. {result.get('message', '')}", ""]
    if result.get("mode") == "recurring":
        return out + _recurring(result)
    sym, ref = result.get("symptom") or {}, result.get("reference") or {}
    unit = result.get("unit", "")
    nd = 4 if unit == "p.u." else 2
    out += [f"- **Investigated:** {sym.get('label', '–')} — {_num(sym.get('value'), '', nd)} ({unit})"
            + (f", outside its limit {_num(sym.get('limit'), '', nd)}" if sym.get("violated") else ""),
            f"- **Compared with:** {ref.get('label', '–')} ({ref.get('why', '–')}) — {_num(ref.get('value'), '', nd)}",
            f"- **Change:** {_num(result.get('change'), unit, nd)}"]
    for key, label in (("outside_limits_since", "Outside its limits since"), ("outage", "Outage in both states"),
                       ("narrowed", "Narrowed"), ("reference_note", "Note on the reference"),
                       ("thinned", "Moments"), ("coverage_note", "Coverage of the agent's hypotheses")):
        if result.get(key):
            out.append(f"- **{label}:** {result[key]}")
    if (result.get("data_check") or {}).get("message"):
        out.append(f"- **Data check:** {result['data_check']['message']}")
    out += ["", f"> {result.get('message', '')}", "", "**Factors** (shares are exact Shapley values over AC power "
            "flows; they add up to the change)", ""]
    rows = []
    for f in result.get("factors") or []:
        mw = f.get("mw") or {}
        rows.append([f["factor"], f"{f['share_pct']:.1f} %", _VERDICT.get(f["verdict"], f["verdict"]),
                     _num(f.get("effect"), "", nd + 1),
                     (f"{mw['reference']} → {mw['now']} MW" if mw else f.get("detail") or "–"),
                     ("–" if f.get("undo_alone_value") is None else
                      f"{f['undo_alone_value']}" + (" (within limits)" if f.get("undo_alone_within_limits") else ""))])
    out += _table(["Factor", "Share", "Verdict", "Effect", "What changed", "Undoing only this"], rows)
    for d in result.get("drill_down") or []:
        out += [f"**Inside “{d['factor']}”** — {d.get('note', '')}", ""]
        out += _table(["Member", "Share of the factor", "What changed"],
                      [[m["member"], f"{m['share_of_factor_pct']} %" if m.get("share_of_factor_pct") is not None
                        else "–", m.get("detail") or "–"] for m in d.get("members") or []])
    pairs = result.get("interactions") or []
    if pairs:
        out += ["**Causes that interact**", ""]
        out += [f"- {p['factors'][0]} and {p['factors'][1]} {p['kind']} each other ({p['share_pct']:+.1f} % of the "
                "change comes from the two together)." for p in pairs] + [""]
    rr = result.get("reference_robustness")
    if rr:
        out += ["**Against other references**", "", rr.get("message", ""), ""]
        out += _table(["Reference", "Moment", "Main cause", "Share"],
                      [[c["mode"], c.get("timestamp") or ("this one" if c.get("used") else "not available"),
                        c.get("main_cause") or "–", f"{c['share_pct']} %" if c.get("share_pct") is not None else "–"]
                       for c in rr.get("checked") or []])
    if result.get("hypotheses_coverage_pct") is not None:
        out += [f"**The agent's own hypotheses explain {result['hypotheses_coverage_pct']} % of the change.**", ""]
    out += _classical(result.get("classical") or {})
    levers = result.get("levers") or {}
    action = (levers.get("recommended_action") or {}).get("text")
    if action:
        out += ["**What to move now** (sensitivity attribution; what changed is not necessarily what to move)", "",
                f"> {action}", ""]
    return out


def _classical(views: dict) -> list[str]:
    out = []
    linear = views.get("linear") or {}
    if linear.get("factors"):
        out += ["**Classical view: linear estimate from sensitivities** (an approximation beside the exact shares)",
                ""]
        out += _table(["Factor", "Exact", "Linear", "Error (points)", "Method"],
                      [[r["factor"], f"{r.get('exact_share_pct')} %",
                        f"{r['share_pct']} %" if r.get("share_pct") is not None else "n/a",
                        r.get("error_points", "–"), r.get("method") or r.get("why", "–")] for r in linear["factors"]])
        out += [f"Covers {linear.get('covers_pct')} % of the change; the linear main cause "
                f"{'agrees' if linear.get('agrees_with_exact') else 'does not agree'} with the exact one.", ""]
    elif linear.get("not_applicable"):
        out += [f"**Linear estimate:** not applicable — {linear['not_applicable']}.", ""]
    tracing = views.get("tracing") or {}
    now = tracing.get("current")
    if isinstance(now, dict):
        out += [f"**Classical view: power-flow tracing** — {now.get('direction')}, {now.get('flow_mw')} MW now "
                "(proportional sharing, lossless)", ""]
        out += _table(["Comes from", "Share", "Goes to", "Share"],
                      [[s.get("name") if s else "", f"{s['share_pct']} %" if s else "",
                        k.get("name") if k else "", f"{k['share_pct']} %" if k else ""]
                       for s, k in _zip_longest(now.get("sources") or [], now.get("sinks") or [])])
    elif tracing.get("not_applicable"):
        out += [f"**Tracing:** not applicable — {tracing['not_applicable']}.", ""]
    return out


def _zip_longest(a: list, b: list) -> list[tuple]:
    n = max(len(a), len(b))
    return [(a[i] if i < len(a) else None, b[i] if i < len(b) else None) for i in range(n)]


def _recurring(result: dict) -> list[str]:
    out = [f"> {result.get('message', '')}", ""]
    if result.get("picked"):
        out += [f"- {result['picked']}", ""]
    out += _table(["Episode", "Peak", "Main cause", "Share", "Hours outside"],
                  [[f"{str(e['start'])[:16]} → {str(e['end'])[11:16]}", e.get("peak_value"), e.get("main_cause") or "–",
                    f"{e['share_pct']} %" if e.get("share_pct") is not None else "–", e.get("hours_outside")]
                   for e in result.get("events") or []])
    out += _table(["Cause", "Main in", "Share range"],
                  [[c["cause"], f"{c['episodes']} episodes ({c['main_in_pct']} %)",
                    f"{c['share_range_pct'][0]}–{c['share_range_pct'][1]} %"] for c in result.get("causes") or []])
    return out


def _prevention(result: dict, n: int) -> list[str]:
    ep, warn = result.get("episode") or {}, result.get("warning") or {}
    out = [f"### Prevention {n}: {result.get('element')}", "", f"> {result.get('message', '')}", "",
           f"- **Episode:** {str(ep.get('start'))[:16]} to {str(ep.get('end'))[:16]} ({ep.get('hours')} h), worst "
           f"{ep.get('worst_value')}",
           "- **Warning:** " + (f"{str(warn.get('start'))[:16]}, {warn.get('lead_minutes')} minutes before "
                                 f"(within {warn.get('margin')} of the limit)" if warn.get("seen") else "none"),
           f"- **Actions from:** {result.get('actions_from')}", ""]
    rows = []
    for a in result.get("actions") or []:
        given = a.get("as_given") or {}
        smallest = (a.get("smallest") or {}).get("action")
        obstacle = given.get("reason") or (", ".join(given.get("new_violations") or []) and
                                           "new violations: " + ", ".join(given["new_violations"]))
        rows.append([a["action"], "yes" if given.get("works") else "no", smallest or "–",
                     obstacle or ("falls short" if not given.get("works") else "–")])
    return out + _table(["Action", "Enough as given", "Smallest sufficient", "What stops it"], rows)


def _look_ahead(result: dict, n: int) -> list[str]:
    horizon = result.get("horizon") or {}
    out = [f"### {n}. Looking ahead: {str(horizon.get('start'))[:16]} to {str(horizon.get('end'))[:16]}", "",
           f"*{result.get('note', '')}*", "", f"> {result.get('message', '')}", ""]
    if result.get("status") != "expected":
        return out
    rows = []
    for e in result.get("episodes") or []:
        action = e.get("action") or {}
        under = action.get("holds_under") or []
        cause = e.get("cause") or {}
        rows.append([
            e["element"], f"{e['start'][:16]} to {e['end'][11:16]}" + (" (and beyond)" if e.get("beyond_horizon") else ""),
            e.get("peak_value"), "now" if e.get("decide_now") else
            (f"{e['decide_by'][:16]} ({e['lead_minutes']} min)" if e.get("lead_minutes") else "no warning"),
            f"{cause['main_cause']} ({cause['share_pct']:.0f} %)" if cause.get("main_cause") else "–",
            action.get("text") or e.get("no_action") or "–",
            "; ".join(f"{h['label']}: " + {True: "yes", False: f"no — {h.get('why_not', '')}"
                                            + (f"; {h['enough_then']} would" if h.get("enough_then") else "")}.get(
                h["holds"], "–") for h in under) or "–"])
    return out + _table(["Element", "When", "Worst", "Decide by", "Main cause", "Smallest sufficient action",
                         "Holds if the forecast errs"], rows)


def build(record: dict, answer: str | None = None) -> str:
    """The report of one turn's investigation, as Markdown."""
    answer = answer if answer is not None else record.get("assistant") or ""
    calls = list(zip(record.get("tool_calls") or [], record.get("tool_results") or []))
    when = record.get("timestamp_utc") or datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    out = ["# Investigation report", "",
           f"*CONDUCTOR · {when} · model {record.get('model', '–')} ({record.get('provider', '–')})*", "",
           "## The question", "", record.get("user") or "–", "", "## The answer", "", answer.strip(), "",
           "## The investigation", ""]
    n = 0
    for call, entry in calls:
        result = entry.get("result") if isinstance(entry, dict) else None
        if not isinstance(result, dict) or result.get("error"):
            continue
        if call.get("name") == "investigate":
            n += 1
            out += _investigation(result, n)
        elif call.get("name") == "test_prevention":
            n += 1
            out += _prevention(result, n)
        elif call.get("name") == "look_ahead":
            n += 1
            out += _look_ahead(result, n)
    trail = next((e["result"] for c, e in reversed(calls) if c.get("name") == "record_investigation"
                  and isinstance(e.get("result"), dict) and e["result"].get("status") == "recorded"), None)
    if trail:
        out += ["## The hypotheses the agent tested", "", f"*{trail.get('symptom', '')}*", ""]
        out += _table(["Verdict", "Hypothesis", "What showed it", "Tested with / would need"],
                      [[_RECORDED.get(h["verdict"], h["verdict"]), h["hypothesis"], h.get("finding") or "–",
                        (", ".join(reg.label(t) for t in h.get("evidence") or []) if h["verdict"] != "untested"
                         else {"analysis": "would need a tool", "data": "would need data"}.get(
                             h.get("missing_kind"), "would need") + f": {h.get('missing_tool', '–')}")]
                        for h in trail.get("hypotheses") or []])
    out += ["## What backs it", ""]
    try:
        from .evidence import build as evidence_of
        from .investigation_record import check_answer
        ledger = evidence_of(record)
        counts = ledger.counts()
        out.append("- **Evidence:** " + ", ".join(f"{v} {k.replace('_', ' ')}" for k, v in counts.items() if v))
        issues = check_answer(answer, record)
        out.append("- **Answer against the record:** " + ("; ".join(f"{i.kind.replace('_', ' ')}: {i.note}"
                                                                    for i in issues) or "consistent"))
    except Exception:  # noqa: BLE001 — a report without this line beats no report
        pass
    reading = record.get("reading_check") or {}
    if reading.get("headline"):
        out.append(f"- **Reading check:** {reading['headline']}")
    out += ["", "## The studies that ran", ""]
    out += _table(["#", "Study", "Arguments", "Result"],
                  [[i + 1, reg.label(c.get("name", "")),
                    ", ".join(f"{k}={v}" for k, v in (c.get("args") or {}).items()
                              if v not in (None, "", [], {}) and k not in ("hypotheses", "actions"))[:220],
                    "error" if isinstance(e.get("result"), dict) and e["result"].get("error") else
                    (e.get("result") or {}).get("status", "ok") if isinstance(e.get("result"), dict) else "ok"]
                   for i, (c, e) in enumerate(calls)])
    out += ["## Method and limits", "",
            "- A factor's share is its Shapley value: its average effect over every order in which the changes "
            "could have happened, from AC power flows of every combination of the two states. The shares add up "
            "to the change, interactions included, and the two states are reproduced exactly (checked).",
            "- Shares depend on the reference compared with; the report says which, why, and whether the main "
            "cause holds against others.",
            "- The verdicts are the studies'; the agent chose what to test, and its record was checked against "
            "them before it was accepted. A hypothesis no tool could test is not a finding.",
            "- Classical views (linear estimate, tracing) are approximations beside the exact shares.",
            "- The model is static: each moment is a power flow; a prevention check holds an action through the "
            "episode, it does not simulate how the grid would have evolved.",
            "- A look-ahead is as good as the forecast: its episodes are expectations, and each action is checked "
            "again with demand above the forecast to see whether it still holds.", ""]
    return "\n".join(out)
