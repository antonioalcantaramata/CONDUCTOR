"""
Key performance indicators of an operating point, before and after a
corrective action.

The study compares two states of one moment, each checked by an AC power flow:
the schedule as it is, and the schedule with a corrective action applied (by
default the cheapest at that moment; or an option the operator names, held
from its own moment when it is applied later). Every figure comes from a
solver or a power flow; this module only puts them side by side.

The three indicators of the project notebook keep their names and intent,
with the flaws that made them uninformative fixed:

  KPI-1  Target demand flexibility — the units' spare upward capacity as a
         share of demand (load and losses): Σ max(Pmax − P, 0) / Σ supply.
         Computed on the dispatch itself after a power flow (it used to be read
         off a second solve, which moved it even under a cap that limited
         nothing). Spare capacity is not deliverable capacity: the network may
         not carry it — `deliverable_up_mw` says how much it can.
  KPI-2  Flexibility utilisation — the fewest MW of redispatch that clear the
         violations, as a share of the MW the action moves. It used to compare
         the MW moved with the net change in total generation, which redispatch
         keeps near zero (units up and down), so it read 0 % at every hour.
  KPI-3  Prevented violation ratio — the share of the schedule's violations
         the action clears, counted by power flow. It used to be 100 % whenever
         the solver converged, even with nothing to prevent; now it has no value
         then, and is never 0 % for "no solution".

Beside them: N-0 and N-1 security margins, the flexibility the units can
deliver within every limit (up and down, each unit alone and all together),
and the cost of the action (and of N-1 security, when asked).

Pure functions: no app state, no HTTP.
"""

from __future__ import annotations

from typing import NamedTuple

# Below this a figure is solver tolerance, not a movement anyone would order.
NOISE_MW = 0.01


class Indicator(NamedTuple):
    key: str
    name: str
    unit: str
    better: str           # "higher" / "lower": which way is an improvement
    group: str            # security / flexibility / cost
    what: str             # one line: what it measures


INDICATORS = (
    Indicator("violations", "Violations (N-0)", "", "lower", "security",
              "limits broken with every branch in service, by AC power flow"),
    Indicator("vm_margin_pu", "Voltage margin", "p.u.", "higher", "security",
              "the nearest bus voltage's distance to its limit (negative: outside it)"),
    Indicator("loading_headroom_pct", "Loading headroom", "pp", "higher", "security",
              "percentage points between the most loaded branch and its limit (negative: above it)"),
    Indicator("n1_outages_with_violations", "N-1 outages with violations", "", "lower", "security",
              "single outages after which a limit breaks, of all those tested"),
    Indicator("kpi_3_prevented_violation_ratio_pct", "KPI-3 Prevented violation ratio", "%", "higher", "security",
              "share of the schedule's violations the action clears"),
    Indicator("kpi_1_target_demand_flex_pct", "KPI-1 Target demand flexibility", "%", "higher", "flexibility",
              "spare upward capacity of the units, % of demand (load and losses)"),
    Indicator("spare_up_mw", "Spare upward capacity", "MW", "higher", "flexibility",
              "the units' headroom to their maximum (KPI-1's numerator), whether the network can carry it or not"),
    Indicator("deliverable_up_mw", "Deliverable upward flexibility", "MW", "higher", "flexibility",
              "how far the units together can raise their output within every limit"),
    Indicator("deliverable_down_mw", "Deliverable downward flexibility", "MW", "higher", "flexibility",
              "how far the units together can lower their output within every limit"),
    Indicator("required_mw", "Flexibility required", "MW", "lower", "flexibility",
              "the fewest MW of redispatch that clear the violations"),
    Indicator("moved_mw", "Flexibility activated", "MW", "lower", "flexibility",
              "the MW the action moves (units, and loads it reduces)"),
    Indicator("kpi_2_flex_utilization_pct", "KPI-2 Flexibility utilisation", "%", "higher", "flexibility",
              "the flexibility required as a share of the flexibility activated"),
    Indicator("cost_eur_per_h", "Cost of the action", "€/h", "lower", "cost",
              "the action at the cost table's prices (the schedule: none)"),
    Indicator("n1_security_eur_per_h", "Cost of N-1 security", "€/h", "lower", "cost",
              "the N-1 secure option above the cheapest action (asked with n1)"),
)
BY_KEY = {i.key: i for i in INDICATORS}
OFFICIAL = ("kpi_1_target_demand_flex_pct", "kpi_2_flex_utilization_pct", "kpi_3_prevented_violation_ratio_pct")


# ---------------------------------------------------------------------------
# The three indicators of the project notebook
# ---------------------------------------------------------------------------

def target_demand_flexibility(dispatch: dict, pmax: dict, supply_mw: float | None) -> dict:
    """KPI-1: Σ max(Pmax − P, 0) over the units with a known maximum, as a
    share of the supply (load and losses). None without a positive supply."""
    spare = sum(max(float(pmax[g]) - float(p), 0.0) for g, p in dispatch.items()
                if pmax.get(g) is not None)
    pct = 100.0 * spare / supply_mw if supply_mw and supply_mw > 0 else None
    return {"spare_mw": round(spare, 3), "supply_mw": round(supply_mw, 3) if supply_mw is not None else None,
            "pct": round(pct, 2) if pct is not None else None}


def flexibility_utilisation(required_mw: float | None, moved_mw: float | None, violations: int = 0) -> dict:
    """KPI-2: required / activated. 100 % when nothing is needed and nothing
    moves; 0 % when something moves though nothing is needed (`violations`:
    the schedule's — then reactive power or voltage set-points clear them);
    None when the requirement is unknown (redispatch alone cannot clear them)."""
    if required_mw is None or moved_mw is None:
        return {"required_mw": required_mw, "moved_mw": moved_mw, "pct": None,
                "note": "no action to measure" if moved_mw is None else
                "no requirement: redispatching the units alone cannot clear the violations"}
    required, moved = max(required_mw, 0.0), max(moved_mw, 0.0)
    if moved < NOISE_MW:
        pct, note = (100.0, "nothing needed, nothing moved") if required < NOISE_MW else \
            (0.0, "the action moves nothing, yet redispatch is needed")
    elif required < NOISE_MW:
        pct, note = 0.0, ("no MW needs to move: reactive power or voltage set-points clear the violations"
                          if violations else "the action moves units though nothing needs to move")
    else:
        pct, note = min(100.0, 100.0 * required / moved), None
    return {"required_mw": round(required, 3), "moved_mw": round(moved, 3), "pct": round(pct, 2),
            **({"note": note} if note else {})}


def prevented_violation_ratio(before: int | None, after: int | None) -> dict:
    """KPI-3: (violations before − after) / before. None when the schedule
    breaks no limit (nothing to prevent) or a state is unknown."""
    if before is None or after is None:
        return {"before": before, "after": after, "pct": None, "note": "a state could not be assessed"}
    if before == 0:
        return {"before": 0, "after": after, "pct": None,
                "note": "nothing to prevent: the schedule breaks no limit"
                + (f" (the action breaks {after})" if after else "")}
    pct = max(0.0, 100.0 * (before - after) / before)
    return {"before": before, "after": after, "pct": round(pct, 2),
            **({"note": f"the action leaves {after} of the {before}"} if after else {})}


# ---------------------------------------------------------------------------
# One moment
# ---------------------------------------------------------------------------

def indicator_rows(values: dict) -> list[dict]:
    """{key: (before, after)} as rows in the indicators' order, each with what
    it is and which way is better; keys without values left out."""
    rows = []
    for ind in INDICATORS:
        if ind.key not in values:
            continue
        before, after = values[ind.key]
        rows.append({"key": ind.key, "name": ind.name, "unit": ind.unit, "better": ind.better,
                     "group": ind.group, "what": ind.what, "before": before, "after": after})
    return rows


def flat_metrics(rows: list[dict]) -> dict:
    """{key: value with the action} — the numbers another result is compared
    with (`compare_results`)."""
    return {r["key"]: r["after"] for r in rows if isinstance(r.get("after"), (int, float))}


def _fmt(value, unit: str) -> str:
    if value is None:
        return "—"
    digits = {"p.u.": 3, "pp": 1, "€/h": 2, "MW": 1, "%": 1}.get(unit)
    value = round(value, digits) + 0.0 if digits is not None else value      # never "-0.000"
    if unit == "p.u.":
        return f"{value:+.3f} p.u."
    if unit == "pp":
        return f"{value:+.1f} pp"
    if unit == "€/h":
        return f"{value:,.2f} €/h"
    if unit == "MW":
        return f"{value:.1f} MW"
    if unit == "%":
        return f"{value:.1f} %"
    return f"{value:g}"


def moment_message(row: dict) -> str:
    """The moment in a few sentences, every number from the row."""
    if row.get("status") == "error":
        return f"{row['timestamp']}: no indicators — {row.get('why')}"
    vals = {r["key"]: r for r in row.get("indicators") or []}

    def pair(key):
        r = vals.get(key)
        if r is None:
            return None
        b, a = _fmt(r["before"], r["unit"]), _fmt(r["after"], r["unit"])
        return f"{r['name']} {a}" if r["before"] is None or b == a else f"{r['name']} {b} → {a}"

    state = row.get("schedule") or {}
    head = (f"At {row['timestamp']} the schedule breaks {state.get('violations')} limit(s)"
            + (f" ({'; '.join(state.get('worst') or [])})" if state.get("worst") else "")
            if state.get("violations") else f"At {row['timestamp']} the schedule breaks no limit")
    after_is = row.get("after_is")
    if after_is == "schedule":
        head += "; no corrective action is needed, so the indicators are the schedule's"
    elif after_is == "none":
        head += "; no corrective action was found — " + (row.get("why") or "see options_status")
    else:
        head += f"; compared with {row.get('after_label')} ({row.get('after_ref')})" + (
            f", held from {row.get('held_since')}" if row.get("held") else "")
    parts = [p for p in (pair(k) for k in (
        "violations", "vm_margin_pu", "loading_headroom_pct", "n1_outages_with_violations",
        "kpi_1_target_demand_flex_pct", "spare_up_mw", "deliverable_up_mw", "deliverable_down_mw",
        "required_mw", "moved_mw", "kpi_2_flex_utilization_pct", "kpi_3_prevented_violation_ratio_pct",
        "cost_eur_per_h", "n1_security_eur_per_h")) if p]
    notes = [n for n in ((row.get("kpi_details") or {}).get(k, {}).get("note") for k in ("kpi_2", "kpi_3")) if n]
    return head + ". " + "; ".join(parts) + "." + (" " + " ".join(f"({n})" for n in notes) if notes else "")


# ---------------------------------------------------------------------------
# A window
# ---------------------------------------------------------------------------

def compact_moments(rows: list[dict]) -> dict:
    """A window's rows for its result: each indicator described once
    (`indicators`), each moment's values as {key: [schedule, with the action]}
    and only the notes of its KPI details — a twelfth of the size."""
    keys = []
    for row in rows:
        found = row.pop("indicators", None)
        if found is None:
            continue
        keys += [r["key"] for r in found if r["key"] not in keys]
        row["values"] = {r["key"]: [r["before"], r["after"]] for r in found}
        if row.get("after_is") == "schedule":            # the same state twice: said once
            row["after"] = "the schedule"
            if isinstance(row.get("deliverable"), dict):
                row["deliverable"]["schedule"] = "as with_action"
        details = row.pop("kpi_details", None) or {}
        notes = [d["note"] for d in (details.get("kpi_2") or {}, details.get("kpi_3") or {}) if d.get("note")]
        if notes:
            row["notes"] = notes
    meta = [{"key": i.key, "name": i.name, "unit": i.unit, "better": i.better, "group": i.group, "what": i.what}
            for i in INDICATORS if i.key in keys]
    return {"indicators": meta, "values_are": "[as scheduled, with the action]", "moments": rows}


def window_summary(rows: list[dict], weights: list[float]) -> dict:
    """Per indicator, over the moments that have it: the range with the action
    (and as scheduled), where each extreme falls; and the window's totals —
    the cost of the actions (€) and the MWh moved and required, each moment
    weighted by its share of the window (`weights`, hours)."""
    ok = [(r, w) for r, w in zip(rows, weights) if r.get("status") != "error"]
    out: dict = {"indicators": {}, "totals": {}}
    for ind in INDICATORS:
        series = [(r["timestamp"], v) for r, _ in ok
                  for v in [next((x for x in r.get("indicators") or [] if x["key"] == ind.key), None)]
                  if v is not None and isinstance(v.get("after"), (int, float))]
        if not series:
            continue
        values = [v["after"] for _, v in series]
        worst = (min if ind.better == "higher" else max)(series, key=lambda s: s[1]["after"])
        best = (max if ind.better == "higher" else min)(series, key=lambda s: s[1]["after"])
        before = [v["before"] for _, v in series if isinstance(v.get("before"), (int, float))]
        out["indicators"][ind.key] = {
            "name": ind.name, "unit": ind.unit, "better": ind.better,
            "min": min(values), "max": max(values), "mean": round(sum(values) / len(values), 3),
            "worst": {"timestamp": worst[0], "value": worst[1]["after"]},
            "best": {"timestamp": best[0], "value": best[1]["after"]},
            **({"before_min": min(before), "before_max": max(before)} if before else {})}

    def total(key, digits=2):
        pairs = [(v, w) for r, w in ok for x in r.get("indicators") or []
                 if x["key"] == key and isinstance(v := x.get("after"), (int, float))]
        return round(sum(v * w for v, w in pairs), digits) if pairs else None

    out["totals"] = {k: v for k, v in {
        "cost_eur": total("cost_eur_per_h"), "n1_security_eur": total("n1_security_eur_per_h"),
        "mwh_moved": total("moved_mw", 3), "mwh_required": total("required_mw", 3),
        "hours": round(sum(weights), 3)}.items() if v is not None}
    out["moments_failed"] = [r["timestamp"] for r in rows if r.get("status") == "error"]
    out["moments_with_violations"] = [r["timestamp"] for r, _ in ok if (r.get("schedule") or {}).get("violations")]
    return out


def window_message(ticks: list[str], summary: dict) -> str:
    """The window in a few sentences, every number from the summary."""
    ind, tot = summary["indicators"], summary["totals"]
    lines = [f"{len(ticks)} moments from {ticks[0]} to {ticks[-1]}, each comparing the schedule with the "
             "corrective action of that moment"]
    hit = summary.get("moments_with_violations") or []
    lines.append(f"the schedule breaks a limit at {len(hit)} of them" if hit else "the schedule breaks no limit")
    for key in ("kpi_1_target_demand_flex_pct", "deliverable_up_mw", "deliverable_down_mw", "vm_margin_pu",
                "loading_headroom_pct", "n1_outages_with_violations", "kpi_2_flex_utilization_pct"):
        if key in ind:
            s = ind[key]
            lines.append(f"{s['name']} {_fmt(s['min'], s['unit'])} to {_fmt(s['max'], s['unit'])} "
                         f"(worst {_fmt(s['worst']['value'], s['unit'])} at {s['worst']['timestamp']})")
    if "cost_eur" in tot:
        net = tot["cost_eur"]
        lines.append((f"the corrective actions cost {net:,.2f} €" if net >= 0 else
                      f"the corrective actions save {-net:,.2f} € net (a negative cost: the change is cheaper "
                      "than the schedule)")
                     + f" over the {tot['hours']:g} h window at the cost table's prices"
                     + (f", moving {tot['mwh_moved']:.1f} MWh where {tot['mwh_required']:.1f} MWh were needed"
                        if "mwh_moved" in tot and "mwh_required" in tot else ""))
    if "n1_security_eur" in tot:
        lines.append(f"N-1 security costs {tot['n1_security_eur']:,.2f} € more")
    failed = summary.get("moments_failed") or []
    if failed:
        lines.append(f"{len(failed)} moment(s) have no indicators ({', '.join(failed)}): not zero, unknown")
    return "; ".join(lines) + ". The window totals weight each moment by half the time to its neighbours."
