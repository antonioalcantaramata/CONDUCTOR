"""
Investigation evaluation — does CONDUCTOR find the cause it was given, and
does the agent investigate when it should, and only then?

Two parts.

The bench (`bench`) needs no model and spends nothing. Each case injects
known causes into the IEEE 30-bus demo as a proposal — a load scaled up, a
data centre added, a unit's output or voltage set-point lowered, a line
opened, the external grid's voltage lowered — and asks `/api/investigate` why
the element they affect changed, against the base. The ground truth is the
injection itself. Three kinds of case:

- one cause, compared with the base at the same moment: the injected change is
  the only difference, so it must be the main cause with the whole change;
- two causes at once: both must be causes, and nothing else;
- a cause with background: compared with the base an hour earlier, so the
  demand and the units' schedule moved too; the injected change must still be
  found as the main cause (it is made large on purpose).

Each case is scored for the decomposition and two baselines: the sensitivity
attribution alone (its top driver for the element: what the agent could say
before), and undoing each factor alone (the usual one-at-a-time
counterfactual, read from the same result).

    python -m evaluation.investigation bench [--json]

The agent suite drives the real agent and SPENDS MODEL CREDIT. Its tasks check
that a "why" about a problem in the grid is investigated, that a "why" about a
method, a recommendation or physics, or a what-if, is not, that a symptom that
is not there is said, that an injected cause is reported, and that what is
coming in the forecast is looked ahead.

    python -m evaluation.investigation list
    python -m evaluation.investigation run --repeats 3 [--tasks why_bus29,why_method]
    python -m evaluation.investigation grade evaluation_runs/investigation-20261025-2130

Grading is deterministic, from the records: no LLM judge.
"""

from __future__ import annotations

import json
import sys
import time
from typing import Callable, NamedTuple

from .whatif import Suite, Task, main

PEAK = "20:00 on 6 January"
T = "2000-01-06 18:00"            # the base is secure here; the injections make it insecure
NIGHT, NIGHT_BEFORE = "2000-01-06 03:00", "2000-01-06 02:00"
EVENING_BEFORE = "2000-01-06 17:00"


class Case(NamedTuple):
    id: str
    kind: str                      # one | two | background
    edits: tuple[dict, ...]
    element: str
    expect: tuple[str, ...]        # text identifying each injected factor (its label or a member's)
    timestamp: str = T
    reference_timestamp: str | None = None
    attribution_source: tuple[str, ...] = ()   # the attribution's name for an injected source, if it is one
    # The caller's own hypotheses (the agent's), and those of them that must be
    # ruled out: a wrong hypothesis has to come back with no share.
    hypotheses: tuple[dict, ...] = ()
    reject: tuple[str, ...] = ()
    outage: str | None = None          # a line out in both states (N-1)


def _load_at(bus: int) -> dict:
    return {"element_type": "load", "bus": bus}


CASES: tuple[Case, ...] = (
    Case("load_bus29", "one", ({"op": "scale", **_load_at(29), "factor": 1.5},), "Bus_29",
         ("Demand at Bus_29",), attribution_source=("Load_20",)),
    Case("data_centre_bus25", "one", ({"op": "add_load", "name": "DC", "bus": 25, "p_mw": 10},), "Bus_25",
         ("DC at Bus_25",), attribution_source=("DC",)),
    Case("grid_voltage", "one", ({"op": "set_parameter", "element_type": "ext_grid", "values": {"vm_pu": 0.97}},),
         "Bus_0 -> Bus_1 [L0]", ("External grid voltage",)),
    Case("line_out_L38", "one", ({"op": "set_in_service", "element_type": "line", "element": 38,
                                  "in_service": False},), "Bus_29", ("[L38] out of service",)),
    Case("unit_output_gen1", "one", ({"op": "set_parameter", "element_type": "gen", "element": "Gen_bus1",
                                      "values": {"p_mw": 5.0}},), "Bus_0 -> Bus_1 [L0]",
         ("Output of Gen_bus1",), attribution_source=("Gen_bus1",)),
    Case("unit_setpoint_gen12", "one", ({"op": "set_parameter", "element_type": "gen", "element": "Gen_bus12",
                                         "values": {"vm_pu": 0.95}},), "Bus_29",
         ("Voltage set-point of Gen_bus12",)),
    Case("load_and_setpoint", "two", ({"op": "scale", **_load_at(29), "factor": 1.25},
                                      {"op": "set_parameter", "element_type": "gen", "element": "Gen_bus12",
                                       "values": {"vm_pu": 0.97}}), "Bus_29",
         ("Demand at Bus_29", "Voltage set-point of Gen_bus12")),
    Case("unit_and_data_centre", "two", ({"op": "set_parameter", "element_type": "gen", "element": "Gen_bus4",
                                          "values": {"p_mw": 15.0}},
                                         {"op": "add_load", "name": "DC", "bus": 3, "p_mw": 12}),
         "Bus_0 -> Bus_1 [L0]", ("Output of Gen_bus4", "DC at Bus_3")),
    # Two causes that interact: the load at Bus_29 drops its voltage more once
    # the line feeding it is open, so undoing either alone undoes more than its share.
    Case("line_out_and_load", "two", ({"op": "set_in_service", "element_type": "line", "element": 38,
                                       "in_service": False},
                                      {"op": "scale", **_load_at(29), "factor": 1.4}), "Bus_29",
         ("[L38] out of service", "Demand at Bus_29")),
    Case("grid_voltage_and_unit", "two", ({"op": "set_parameter", "element_type": "ext_grid",
                                           "values": {"vm_pu": 0.97}},
                                          {"op": "set_parameter", "element_type": "gen", "element": "Gen_bus1",
                                           "values": {"p_mw": 20.0}}), "Bus_0 -> Bus_1 [L0]",
         ("External grid voltage", "Output of Gen_bus1")),
    # The agent's own hypotheses. Two loads in different places, both outside
    # Bus_29's neighbourhood: the standard grouping puts them in one factor,
    # the hypotheses separate them, and a hypothesis about a unit that did not
    # change is ruled out.
    Case("hypotheses_two_loads", "hypotheses", ({"op": "scale", **_load_at(23), "factor": 2.0},
                                                {"op": "scale", **_load_at(25), "factor": 1.6}), "Bus_29",
         ("Demand at Bus_23", "Demand at Bus_25"),
         hypotheses=({"name": "Demand at Bus_23", "elements": ["Bus_23"]},
                     {"name": "Demand at Bus_25", "elements": ["Bus_25"]},
                     {"name": "Gen_bus12's set-point", "elements": ["Gen_bus12"]}),
         reject=("Gen_bus12's set-point",)),
    # A wrong hypothesis: the line was opened, the agent suspects the demand.
    # The demand did not change, so it is ruled out and the line comes back
    # under "other changes" with the whole change.
    Case("hypothesis_wrong", "hypotheses", ({"op": "set_in_service", "element_type": "line", "element": 38,
                                             "in_service": False},), "Bus_29", ("[L38]",),
         hypotheses=({"name": "Demand at Bus_29", "elements": ["Bus_29"]},), reject=("Demand at Bus_29",)),
    # Other scopes: the whole grid (the number of elements outside limits), and
    # after an outage (the same line out in both states).
    Case("whole_grid", "scope", ({"op": "scale", **_load_at(29), "factor": 1.5},
                                 {"op": "set_parameter", "element_type": "ext_grid", "values": {"vm_pu": 0.97}}),
         "all", ("Demand at Bus_29", "External grid voltage")),
    Case("after_outage", "scope", ({"op": "scale", **_load_at(29), "factor": 1.2},), "Bus_29",
         ("Demand at Bus_29",), outage="Bus_26 -> Bus_29 [L37]"),
    Case("load_at_night", "background", ({"op": "scale", **_load_at(29), "factor": 1.5},), "Bus_29",
         ("Demand at Bus_29",), timestamp=NIGHT, reference_timestamp=NIGHT_BEFORE),
    Case("line_out_evening", "background", ({"op": "set_in_service", "element_type": "line", "element": 38,
                                             "in_service": False},), "Bus_29", ("[L38] out of service",),
         reference_timestamp=EVENING_BEFORE),
    Case("unit_output_evening", "background", ({"op": "set_parameter", "element_type": "gen",
                                                "element": "Gen_bus1", "values": {"p_mw": 5.0}},),
         "Bus_0 -> Bus_1 [L0]", ("Output of Gen_bus1",), reference_timestamp=EVENING_BEFORE,
         attribution_source=("Gen_bus1",)),
)

PREFIX = "bench_"

# call(method, path, body) -> (status code, JSON body)
Call = Callable[[str, str, dict | None], tuple[int, dict]]


def _text(factor: dict) -> str:
    return " | ".join([factor["factor"], *factor.get("members", [])])


def _matches(factor: dict, expected: str) -> bool:
    return expected in _text(factor)


def run_case(call: Call, case: Case) -> dict:
    """One case: build the proposal, investigate, score, remove the proposal."""
    pid = PREFIX + case.id
    status, body = call("post", "/api/proposals/edit",
                        {"proposal": pid, "edits": list(case.edits), "replace": True})
    if status != 200:
        return {"case": case.id, "kind": case.kind, "error": f"proposal refused: {body}"}
    try:
        request = {"timestamp": case.timestamp, "network": pid, "element": case.element,
                   "reference_network": "base", "classical": ["linear"]}
        if case.reference_timestamp:
            request["reference_timestamp"] = case.reference_timestamp
        if case.hypotheses:
            request["hypotheses"] = list(case.hypotheses)
        if case.outage:
            request["outage"] = case.outage
        started = time.time()
        status, result = call("post", "/api/investigate", request)
        elapsed = time.time() - started
        if status != 200 or result.get("status") != "explained":
            return {"case": case.id, "kind": case.kind, "error": result.get("detail") or result.get("message")}
        score = score_case(case, result)
        score["elapsed_s"] = round(elapsed, 2)
        score["attribution_top1"] = _attribution_top1(call, case, pid, result)
        return score
    finally:
        call("delete", f"/api/proposals/{pid}", None)


def score_case(case: Case, result: dict) -> dict:
    """The decomposition and the one-at-a-time baseline against the injection."""
    factors = result["factors"]
    injected = [f for f in factors if any(_matches(f, e) for e in case.expect)]
    found = {e: next((f for f in factors if _matches(f, e)), None) for e in case.expect}
    top = factors[0] if factors else None
    # A group named as the main cause holds the injection only if its own
    # drill-down puts the injected member first.
    drill = {d["factor"]: d["members"] for d in result.get("drill_down") or []}

    def names_injection(factor: dict | None) -> bool:
        if factor is None or not any(_matches(factor, e) for e in case.expect):
            return False
        if factor["n_members"] > 1 and factor["factor"] in drill:
            return any(e in drill[factor["factor"]][0]["member"] for e in case.expect)
        return factor["n_members"] == 1 or factor["factor"] in drill
    top1 = names_injection(top) if case.kind != "two" else all(
        f is not None and f["verdict"] == "cause" for f in found.values()) and names_injection(top)
    # One at a time: rank by how much undoing each factor alone moves the value.
    value = result["symptom"]["value"]
    oat = sorted(factors, key=lambda f: -abs(value - f["undo_alone_value"]))
    oat_top1 = bool(oat) and any(_matches(oat[0], e) for e in case.expect)
    # What one-at-a-time effects add up to, as a share of the change: 100 % only
    # without interactions. The Shapley shares add up to 100 % by construction.
    oat_sum = sum(value - f["undo_alone_value"] for f in factors)
    false_causes = [f["factor"] for f in factors if f["verdict"] == "cause" and f not in injected]
    rejected = {r: next((f for f in factors if f["factor"] == r), None) for r in case.reject}
    # The classical baseline on the same factors: the linear (sensitivity) estimate.
    linear = (result.get("classical") or {}).get("linear") or {}
    lin_rows = {r["factor"]: r for r in linear.get("factors") or []}
    lin_top = linear.get("main_cause")
    lin_top_row = next((f for f in factors if f["factor"] == lin_top), None)
    blind = [f["factor"] for f in injected if lin_rows.get(f["factor"], {}).get("effect") is None] \
        if "factors" in linear else []
    return {
        "case": case.id, "kind": case.kind, "element": case.element,
        "main_cause": result.get("main_cause"),
        "wrong_hypotheses_ruled_out": (all(f is not None and f["verdict"] == "ruled_out" for f in rejected.values())
                                       if case.reject else None),
        "top1": bool(top1),
        "recall": sum(1 for f in found.values() if f is not None and f["verdict"] == "cause") / len(case.expect),
        "false_causes": false_causes if case.kind != "background" else [],
        "injected_share_pct": round(sum(f["share_pct"] for f in injected), 1),
        "one_at_a_time_top1": oat_top1,
        "linear_top1": (names_injection(lin_top_row) if lin_top_row is not None else False)
        if "factors" in linear else None,
        "linear_mae_points": linear.get("mean_abs_error_points"),
        "linear_covers_pct": linear.get("covers_pct"),
        "linear_blind_to": blind,
        "one_at_a_time_sum_pct": round(100 * oat_sum / result["change"], 1) if result.get("change") else None,
        "closure": result.get("closure", {}).get("reproduces_both_states"),
        "interaction_pct": result.get("interaction_pct"),
        "power_flows": result.get("power_flows"),
    }


def _attribution_top1(call: Call, case: Case, pid: str, result: dict) -> bool | None:
    """Whether the sensitivity attribution's top driver for the element is the
    injected source. None when the element is within its limits (the
    attribution explains violations only)."""
    if not result["symptom"].get("violated") or result.get("element_type") == "grid":
        return None
    status, attr = call("post", "/api/rsa/attribution", {"timestamp": case.timestamp, "network": pid})
    if status != 200:
        return None
    entry = next((v for v in attr.get("violations") or []
                  if v.get("kind") == result["element_type"] and _same_element(v, result)), None)
    if entry is None or not entry.get("drivers"):
        return False
    return entry["drivers"][0]["source"] in case.attribution_source


def _same_element(violation: dict, result: dict) -> bool:
    name = result["element"]
    return violation.get("element") == name or name.endswith(f"[L{violation.get('index')}]") \
        or name.endswith(f"[T{violation.get('index')}]") or name == f"Bus_{violation.get('index')}"


def bench(call: Call, cases: tuple[Case, ...] = CASES) -> dict:
    rows = [run_case(call, case) for case in cases]
    ok = [r for r in rows if "error" not in r]

    def rate(key: str, subset=None) -> float | None:
        values = [r[key] for r in (subset if subset is not None else ok) if r.get(key) is not None]
        return round(sum(bool(v) for v in values) / len(values), 3) if values else None
    summary = {
        "cases": len(rows), "errors": len(rows) - len(ok),
        "top1": rate("top1"),
        "recall": round(sum(r["recall"] for r in ok) / len(ok), 3) if ok else None,
        "false_causes": sum(len(r["false_causes"]) for r in ok),
        "closure": rate("closure"),
        "one_at_a_time_top1": rate("one_at_a_time_top1"),
        "attribution_top1": rate("attribution_top1"),
        "linear_top1": rate("linear_top1"),
        "linear_mae_points": (round(sum(r["linear_mae_points"] for r in ok if r.get("linear_mae_points") is not None)
                                    / max(1, sum(r.get("linear_mae_points") is not None for r in ok)), 1)),
        "linear_blind_cases": sum(bool(r.get("linear_blind_to")) for r in ok),
        "by_kind": {k: rate("top1", [r for r in ok if r["kind"] == k])
                    for k in ("one", "two", "background", "hypotheses", "scope")},
        "wrong_hypotheses_ruled_out": rate("wrong_hypotheses_ruled_out"),
        "one_at_a_time_sum_pct_range": ([min(r["one_at_a_time_sum_pct"] for r in ok),
                                          max(r["one_at_a_time_sum_pct"] for r in ok)] if ok else None),
        "mean_elapsed_s": round(sum(r["elapsed_s"] for r in ok) / len(ok), 2) if ok else None,
    }
    return {"summary": summary, "cases": rows}


def render_bench(out: dict) -> str:
    s = out["summary"]
    lines = ["| Case | Kind | Main cause | Top-1 | Recall | False causes | Injected share | 1-at-a-time "
             "(top-1, sum) | Linear (top-1, error) | Attribution |", "|---|---|---|:-:|--:|---|--:|:-:|:-:|:-:|"]
    mark = {True: "✓", False: "✗", None: "–"}
    for r in out["cases"]:
        if "error" in r:
            lines.append(f"| {r['case']} | {r['kind']} | error: {r['error']} | | | | | | |")
            continue
        lines.append(f"| {r['case']} | {r['kind']} | {r['main_cause']} | {mark[r['top1']]} | {r['recall']:.0%} | "
                     f"{', '.join(r['false_causes']) or '–'} | {r['injected_share_pct']:.0f} % | "
                     f"{mark[r['one_at_a_time_top1']]} {r['one_at_a_time_sum_pct']:.0f} % | "
                     f"{mark[r['linear_top1']]} "
                     + (f"{r['linear_mae_points']:.1f} pts" if r.get("linear_mae_points") is not None else "")
                     + (" (blind to a cause)" if r.get("linear_blind_to") else "")
                     + f" | {mark[r['attribution_top1']]} |")
    lines += ["", f"Top-1: {s['top1']:.0%} (one cause {s['by_kind']['one']:.0%}, two {s['by_kind']['two']:.0%}, "
                  f"background {s['by_kind']['background']:.0%}, own hypotheses {s['by_kind']['hypotheses']:.0%}, "
                  f"other scopes {s['by_kind']['scope']:.0%}) · "
                  f"wrong hypotheses ruled out {s['wrong_hypotheses_ruled_out']:.0%} · recall {s['recall']:.0%} · "
                  "false causes "
                  f"{s['false_causes']} · both states reproduced {s['closure']:.0%} · one-at-a-time top-1 "
                  f"{s['one_at_a_time_top1']:.0%} (its effects add up to "
                  f"{s['one_at_a_time_sum_pct_range'][0]:.0f}–{s['one_at_a_time_sum_pct_range'][1]:.0f} % of the "
                  "change) · attribution top-1 "
                  + (f"{s['attribution_top1']:.0%}" if s["attribution_top1"] is not None else "–")
                  + f" · linear estimate top-1 {s['linear_top1']:.0%}, mean error {s['linear_mae_points']} points, "
                  f"blind to an injected cause in {s['linear_blind_cases']} case(s)"
                  + f" · {s['mean_elapsed_s']} s per case"] if s["top1"] is not None else []
    return "\n".join(lines)


def _http_call(method: str, path: str, body: dict | None):
    import httpx
    from llm_agent.agent.config import BASE_URL, HTTP_TIMEOUT
    with httpx.Client(timeout=HTTP_TIMEOUT) as client:
        response = client.request(method.upper(), f"{BASE_URL}{path}", json=body)
        try:
            return response.status_code, response.json()
        except ValueError:
            return response.status_code, {"detail": response.text}


# ---------------------------------------------------------------------------
# The agent suite (spends model credit)
# ---------------------------------------------------------------------------

def _setup(pid: str, *edits: dict) -> dict:
    return {"proposal": pid, "edits": list(edits)}


TASKS: tuple[Task, ...] = (
    # Should investigate: a problem in the grid, and why.
    Task("why_bus29",
         (f"Why is Bus_29 below its voltage limit at {PEAK}?",),
         ({"kind": "ran", "label": "investigated", "tool": "investigate"},
          {"kind": "mentions", "label": "named the demand near Bus_29 as the main cause", "any": ["Bus_29", "Bus_28"]},
          {"kind": "trail", "label": "recorded the investigation"},
          {"kind": "answer_consistent", "label": "gave nothing untested as a finding",
           "absent": ["untested_as_finding"]})),
    Task("what_caused_overload",
         (f"What caused the overload of the line from the external grid at {PEAK}?",),
         ({"kind": "ran", "label": "investigated", "tool": "investigate"},)),
    # Several violations and no element named: ask which, or investigate each.
    Task("why_insecure",
         (f"Why is the grid insecure at {PEAK} when it was fine in the afternoon?",),
         ({"kind": "any", "label": "asked which element, or investigated", "of": [
             {"kind": "asked_first", "label": "asked"},
             {"kind": "ran", "label": "investigated", "tool": "investigate"}]},)),
    # The premise is wrong: said, not explained.
    Task("no_symptom",
         ("Why is Bus_29 below its voltage limit at 04:00 on 6 January?",),
         ({"kind": "ran", "label": "checked with the investigation", "tool": "investigate"},
          {"kind": "mentions", "label": "said it is within its limits",
           "any": ["within its limits", "within limits", "is not below", "isn't below", "not below", "above its"]})),
    # Should not investigate.
    Task("why_physics",
         ("Why does reactive power affect voltage?",),
         ({"kind": "not_tool", "label": "did not investigate", "tool": "investigate"},)),
    Task("what_if",
         (f"What happens if Gen_bus4 produces 10 MW less at {PEAK}?",),
         ({"kind": "not_tool", "label": "did not investigate", "tool": "investigate"},)),
    Task("why_method",
         (f"Is the grid secure at {PEAK}?", "Why did you use that power factor for the loads?"),
         ({"kind": "not_tool", "label": "did not investigate", "tool": "investigate"},)),
    # Its own hypotheses: the operator asks about two loads the standard
    # grouping would put together.
    Task("own_hypotheses",
         ("On proposal two_loads, Bus_29 is below its limit at 18:00 on 6 January but fine on the base. Is it "
          "the extra demand at Bus_23 or at Bus_25, or both?",),
         ({"kind": "hypotheses", "label": "tested its own hypotheses (Bus_23, Bus_25)",
           "elements": ["Bus_23", "Bus_25"]},
          {"kind": "trail", "label": "recorded Bus_23 as supported", "verdict": "supported", "mentions": ["Bus_23"]},
          {"kind": "trail", "label": "recorded Bus_25 as supported", "verdict": "supported", "mentions": ["Bus_25"]}),
         (_setup("two_loads", {"op": "scale", "element_type": "load", "bus": 23, "factor": 2.0},
                 {"op": "scale", "element_type": "load", "bus": 25, "factor": 1.6}),)),
    # A hypothesis no tool can test: said, with the tool it would need.
    Task("untestable",
         (f"Could the hot weather that evening explain the overload of the line from the external grid at {PEAK}?",),
         ({"kind": "trail", "label": "recorded the weather as untested, with the tool needed",
           "verdict": "untested", "needs_tool": True},
          {"kind": "said_untested", "label": "said it could not be tested"})),
    # Other scopes and references.
    Task("why_outage",
         ("Why would losing the line Bus_28 -> Bus_29 be critical at 18:00 on 6 January when it was not at 15:00?",),
         ({"kind": "ran_with", "label": "investigated after the outage", "tool": "investigate", "args": ["outage"]},
          {"kind": "trail", "label": "recorded the investigation"})),
    Task("why_window",
         ("Why was the line from the external grid outside its limit for longer on 6 January than on 5 January?",),
         ({"kind": "ran_with", "label": "investigated the window", "tool": "investigate",
           "args": ["start_timestamp"]},)),
    Task("why_so_close",
         ("At 16:30 on 6 January Bus_29 is within its limits but close to them. Why is it so close, compared with "
          "a moment when it had a comfortable margin?",),
         ({"kind": "ran_with", "label": "compared with a comfortable moment", "tool": "investigate",
           "args": ["reference_mode", "reference_timestamp"]},)),
    # Classical views, asked for in the operator's words.
    Task("classical_views",
         (f"Why is the line from the external grid overloaded at {PEAK}? Who is using that line, and how far "
          "would a sensitivity-based estimate get?",),
         ({"kind": "ran_with", "label": "asked for the classical views", "tool": "investigate",
           "args": ["classical"]},
          {"kind": "mentions", "label": "said who uses the line", "any": ["Bus_4", "external grid"]})),
    # Across the week, and before the episode.
    Task("recurring",
         ("Is the undervoltage at Bus_29 a recurring problem this week, and is it always the same cause?",),
         ({"kind": "ran_with", "label": "investigated every episode", "tool": "investigate", "args": ["recurring"]},
          {"kind": "mentions", "label": "named the recurring cause", "any": ["demand"]})),
    Task("prevention",
         # Two episodes that evening: asking which is fair, and the second turn says.
         ("Could the undervoltage at Bus_29 on the evening of 5 January have been avoided, and how much warning "
          "did we have?", "The one around 18:30."),
         ({"kind": "tool", "label": "checked prevention", "name": "test_prevention"},
          {"kind": "mentions", "label": "said how much warning there was", "any": ["minute", "warning"]})),
    # Ahead: what the forecast says is coming, and what would prevent it.
    Task("look_ahead",
         ("What should I watch for in the next day of the forecast, and what would prevent it?",),
         ({"kind": "tool", "label": "looked ahead", "name": "look_ahead"},
          {"kind": "mentions", "label": "said when to decide", "any": ["decide", "warning", "before"]},
          {"kind": "mentions", "label": "said it is a forecast", "any": ["forecast", "expected", "expect"]})),
    # Deep Research: in depth when it is a why, as usual when it is not.
    Task("deep_why",
         (f"Why is Bus_29 below its voltage limit at {PEAK}?",),
         ({"kind": "ran", "label": "investigated", "tool": "investigate"},
          {"kind": "tool", "label": "checked prevention", "name": "test_prevention"},
          {"kind": "ran_with", "label": "looked for recurrence", "tool": "investigate", "args": ["recurring"]},
          {"kind": "trail", "label": "recorded the investigation"},
          # Unprompted: nothing in the question points beyond CONDUCTOR's tools.
          {"kind": "trail", "label": "proposed an analysis of the grid CONDUCTOR lacks",
           "verdict": "untested", "needs_tool": True, "missing_kind": "analysis"},
          {"kind": "trail", "label": "proposed data for what lies behind the cause",
           "verdict": "untested", "needs_tool": True, "missing_kind": "data"},
          {"kind": "answer_consistent", "label": "said what was not tested, and nothing untested as a finding",
           "absent": ["untested_as_finding", "untested_unsaid"]}),
         deep=True),
    Task("deep_analysis",
         (f"At {PEAK}, give me options to clear the violations.",),
         ({"kind": "not_tool", "label": "did not investigate", "tool": "investigate"},
          {"kind": "options", "label": "asked the solver for options", "args": {"options": True}},
          {"kind": "deep_judged", "label": "recorded as answered without Deep Research", "investigated": False}),
         deep=True),
    # An injected cause, through a proposal.
    Task("injected_setpoint",
         ("On proposal inj_setpoint, Bus_29 is below its limit at 18:00 on 6 January but on the base it is "
          "fine. Why?",),
         ({"kind": "ran", "label": "investigated against the base", "tool": "investigate",
           "args": {"network": "inj_setpoint", "reference_network": "base"}},
          {"kind": "mentions", "label": "named Gen_bus12's set-point", "any": ["Gen_bus12"]}),
         (_setup("inj_setpoint", {"op": "set_parameter", "element_type": "gen", "element": "Gen_bus12",
                                  "values": {"vm_pu": 0.95}}),)),
)

INVESTIGATION = Suite("investigation", TASKS, (), "30", "Demo: IEEE 30 · active only")


def _main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["bench"]:
        out = bench(_http_call)
        print(json.dumps(out, indent=1) if "--json" in argv else render_bench(out))
        return 0 if out["summary"]["errors"] == 0 else 1
    return main(argv, suite=INVESTIGATION, doc=__doc__)


if __name__ == "__main__":
    sys.exit(_main())
