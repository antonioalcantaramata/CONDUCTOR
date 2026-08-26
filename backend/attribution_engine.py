#!/usr/bin/env python
# coding: utf-8
"""
attribution_engine.py — why a violation is happening, not just that it is.

The security engines report *which* elements violate their limits. An operator's
next question is always *what is driving it*, and answering that from a violation
table alone requires the reader to supply the causal story themselves. Where an
LLM narrates such a story it is plausible but ungrounded; this engine measures it.

Method
------
Numerical sensitivity on the validated power flow: perturb one injection, re-run
the power flow, and measure how far each violated quantity moved. That ratio is
the definition of the sensitivity, so the result is checkable against the same
solver that produced the violation — no separate analytical model to validate.
It reuses the deterministic kernel exactly as the probabilistic and scenario
tools do.

    S = d(quantity) / d(injection)      from a finite difference

What is reported
----------------
Sensitivities and the movement each source would need to clear the violation.
Both are unambiguous. Percentage "contribution shares" are deliberately NOT
reported: decomposing a bus voltage into per-source contributions requires
choosing a baseline (zero injection? a typical operating point?), different
baselines give different shares, and the number would be quoted far more
confidently than it deserves.

Limits
------
The linearisation is local. It explains the operating point in front of it and
supports small corrective movements; it does not extrapolate to large
counterfactuals. The external grid is the slack and absorbs whatever imbalance
the others create, so it is reported as context rather than perturbed.
"""

from __future__ import annotations

import pandapower as pp
from pandapower.powerflow import LoadflowNotConverged

# Perturbation sizes. Small enough that the linearisation holds, large enough to
# stay well clear of the power flow's own convergence tolerance.
DEFAULT_DELTA_MW = 0.1
DEFAULT_DELTA_MVAR = 0.1

# Sources smaller than this cannot meaningfully move anything; skipping them
# keeps the power-flow count down on networks with many tiny injections.
_MIN_SOURCE_MW = 0.01


def _safe_runpp(net) -> bool:
    try:
        pp.runpp(net, numba=False)
        return True
    except (LoadflowNotConverged, Exception):  # noqa: BLE001
        return False


def _element_name(net, table: str, idx: int, prefix: str) -> str:
    frame = getattr(net, table)
    if "name" in frame.columns:
        raw = frame.at[idx, "name"]
        if raw is not None and str(raw).strip():
            return str(raw).strip()
    return f"{prefix}_{idx}"


def _voltage_controlled_buses(net) -> dict[int, str]:
    """
    Buses whose voltage is held at a setpoint (PV generators and the slack).

    No injection elsewhere can move them, so a violation here has a sensitivity
    of zero for every source. That is correct physics but useless advice unless
    stated: the remedy is the controlling element's setpoint, not redispatch.
    """
    controlled: dict[int, str] = {}
    for table, prefix in (("gen", "Gen"), ("ext_grid", "ExtGrid")):
        frame = getattr(net, table, None)
        if frame is None or frame.empty or "bus" not in frame.columns:
            continue
        for idx in frame.index:
            if "in_service" in frame.columns and not bool(frame.at[idx, "in_service"]):
                continue
            controlled[int(frame.at[idx, "bus"])] = _element_name(net, table, idx, prefix)
    return controlled


def _collect_violations(net, vm_lower, vm_upper, max_line_loading_pct,
                        max_trafo_loading_pct) -> list[dict]:
    """Violated quantities in the base case, with their limits."""
    found: list[dict] = []

    for idx, vm in net.res_bus["vm_pu"].items():
        if vm is None or vm != vm:  # NaN
            continue
        if vm > vm_upper:
            found.append({"element": _element_name(net, "bus", idx, "Bus"),
                          "kind": "bus", "index": int(idx), "quantity": "vm_pu",
                          "type": "overvoltage", "value": float(vm),
                          "limit": float(vm_upper)})
        elif vm < vm_lower:
            found.append({"element": _element_name(net, "bus", idx, "Bus"),
                          "kind": "bus", "index": int(idx), "quantity": "vm_pu",
                          "type": "undervoltage", "value": float(vm),
                          "limit": float(vm_lower)})

    for table, prefix, limit in (("line", "Line", max_line_loading_pct),
                                 ("trafo", "Trafo", max_trafo_loading_pct)):
        res = getattr(net, f"res_{table}", None)
        if res is None or res.empty or "loading_percent" not in res.columns:
            continue
        for idx, loading in res["loading_percent"].items():
            if loading is None or loading != loading:
                continue
            if loading > limit:
                found.append({"element": _element_name(net, table, idx, prefix),
                              "kind": table, "index": int(idx),
                              "quantity": "loading_percent", "type": "overload",
                              "value": float(loading), "limit": float(limit)})

    return found


def _collect_sources(net, top_n: int | None) -> list[dict]:
    """
    Injections whose movement could plausibly relieve a violation.

    Loads are included as diagnostic sources even though they are rarely
    controllable: knowing that a violation is driven by low demand rather than
    by a generator is the explanation an operator actually wants.
    """
    sources: list[dict] = []
    for table, prefix, controllable in (("sgen", "Sgen", True),
                                        ("gen", "Gen", True),
                                        ("load", "Load", False)):
        frame = getattr(net, table, None)
        if frame is None or frame.empty:
            continue
        for idx in frame.index:
            if "in_service" in frame.columns and not bool(frame.at[idx, "in_service"]):
                continue
            p = float(frame.at[idx, "p_mw"]) if "p_mw" in frame.columns else 0.0
            q = float(frame.at[idx, "q_mvar"]) if "q_mvar" in frame.columns else 0.0
            if abs(p) < _MIN_SOURCE_MW and abs(q) < _MIN_SOURCE_MW:
                continue
            sources.append({
                "source": _element_name(net, table, idx, prefix),
                "kind": table, "table": table, "index": int(idx),
                "controllable": controllable,
                "current_p_mw": round(p, 4), "current_q_mvar": round(q, 4),
                "bus": int(frame.at[idx, "bus"]) if "bus" in frame.columns else None,
            })

    sources.sort(key=lambda s: abs(s["current_p_mw"]), reverse=True)
    return sources[:top_n] if top_n else sources


def _measure(net, violations: list[dict]) -> list[float]:
    """Current value of each violated quantity, in violation order."""
    out = []
    for v in violations:
        if v["kind"] == "bus":
            out.append(float(net.res_bus.at[v["index"], "vm_pu"]))
        else:
            out.append(float(getattr(net, f"res_{v['kind']}").at[v["index"], "loading_percent"]))
    return out


def _relief_feasible(relief: float | None, current: float) -> bool | None:
    """
    Whether the source can actually make this movement.

    A finite-difference sensitivity extrapolates linearly and will happily
    report that a 3.5 MW generator should reduce by 6 MW. That is arithmetically
    consistent and operationally meaningless, so movements that would drive a
    source below zero are marked infeasible rather than left for the reader to
    catch.

    Returns False when provably impossible, True for a reduction the source can
    certainly make, and None for an increase whose headroom is not known here.
    """
    if relief is None:
        return None
    resulting = current + relief
    if resulting < 0:
        return False
    return True if relief <= 0 else None


def _relief(value: float, limit: float, sensitivity: float) -> float | None:
    """
    Signed movement of this source that would bring the quantity to its limit,
    assuming linearity and that it acts alone. None when the source has no
    meaningful influence.
    """
    if sensitivity is None or abs(sensitivity) < 1e-9:
        return None
    return round((limit - value) / sensitivity, 4)



def _fmt(value: float, unit: str) -> str:
    return f"{abs(value):.2f} {unit}"


def _withdraw_infeasible(
    driver: dict, relief_key: str, flag_key: str, current_key: str, deliverable_key: str
) -> None:
    """Remove a movement the source provably cannot make, and say what it can.

    Observed live, and the reason this exists. The engine had already decided:
    `relief_mw_feasible` was ``False``, and `recommended_action` said no single
    source could clear the bus. The model read that sentence, took the raw
    ``relief_mw`` of -6.2415 from the driver beside it, and wrote "Reduce
    05 ÅKI Sgen by 6.24 MW" — about a 3.54 MW unit. Moving the decision into
    the engine was not enough while the ingredients stayed on the table.

    So the impossible figure is withdrawn rather than flagged, and replaced by
    the largest movement the source could actually contribute. Reporting only
    what cannot be done is what made an earlier version of the prompt rule go
    silent, so there is always a correct number left to quote.

    Only provably impossible movements are withdrawn. An unknown headroom
    (``None``) is left in place — the same severity split the parameter guards
    use: refuse the impossible, allow the merely unusual.
    """
    if driver.get(flag_key) is not False:
        return
    current = driver.get(current_key) or 0.0
    driver[relief_key] = None
    driver[deliverable_key] = round(-current, 4) if current > 0 else 0.0


def _recommended_action(violation: dict, drivers: list[dict]) -> dict:
    """
    Resolve the corrective recommendation here, not in the language model.

    Deciding whether a movement is deliverable means reading a feasibility flag
    across every driver and branching correctly on all of them. A model asked to
    do that gets it mostly right, and "mostly" is the wrong reliability for a
    claim an operator may act on — in testing one infeasible movement was
    reported as feasible. The rule is deterministic, so it belongs in code, and
    the model's job reduces to quoting the sentence produced here.
    """
    if violation.get("voltage_controlled_by"):
        return {
            "kind": "setpoint",
            "text": (
                f"{violation['element']} is voltage-controlled by "
                f"{violation['voltage_controlled_by']}, which holds it at a "
                "setpoint. No redispatch elsewhere can move it — adjust that "
                "setpoint instead."
            ),
        }

    controllable = [d for d in drivers if d["controllable"]]
    if not controllable:
        return {
            "kind": "no_controllable_source",
            "text": (
                f"No controllable source influences {violation['element']} "
                "measurably at this operating point."
            ),
        }

    # Candidate movements that the source can actually deliver, smallest first.
    options = []
    for d in controllable:
        for relief, feasible, unit, current in (
            (d["relief_mw"], d["relief_mw_feasible"], "MW", d["current_p_mw"]),
            (d["relief_mvar"], d["relief_mvar_feasible"], "MVAr", d["current_q_mvar"]),
        ):
            if relief is None or feasible is not True:
                continue
            options.append((abs(relief), d["source"], relief, unit, current))
    options.sort()

    if options:
        _, source, relief, unit, _ = options[0]
        verb = "Reduce" if relief < 0 else "Increase"
        return {
            "kind": "single_source",
            "source": source,
            "movement": round(relief, 4),
            "unit": unit,
            "text": (
                f"{verb} {source} by {_fmt(relief, unit)} to bring "
                f"{violation['element']} back to {violation['limit']}."
            ),
        }

    # Nothing is deliverable: report the smallest shortfall so the operator can
    # see how far short a single source falls.
    shortfalls = []
    for d in controllable:
        for relief, unit, current in ((d["relief_mw"], "MW", d["current_p_mw"]),
                                      (d["relief_mvar"], "MVAr", d["current_q_mvar"])):
            if relief is not None:
                shortfalls.append((abs(relief), d["source"], relief, unit, current))
    shortfalls.sort()
    _, source, relief, unit, current = shortfalls[0]
    return {
        "kind": "none_sufficient",
        "text": (
            f"No single source can clear {violation['element']}: the smallest "
            f"sufficient movement ({source}, {_fmt(relief, unit)}) exceeds its "
            f"available {_fmt(current, unit)}. This needs several sources acting "
            "together, or a different remedy."
        ),
    }


def attribute_violations(
    net,
    timestamp: str,
    vm_lower: float = 0.95,
    vm_upper: float = 1.05,
    max_line_loading_pct: float = 90.0,
    max_trafo_loading_pct: float = 90.0,
    delta_mw: float = DEFAULT_DELTA_MW,
    delta_mvar: float = DEFAULT_DELTA_MVAR,
    top_n_sources: int | None = 20,
    top_n_drivers: int = 5,
    min_controllable: int = 2,
) -> dict:
    """
    Rank the injections driving each violated element at one operating point.

    Returns a structured result; never raises on non-convergence, which is
    reported in the payload instead so the agent can explain it.
    """
    if not _safe_runpp(net):
        return {"timestamp": timestamp, "converged": False, "violations": [],
                "error": "Base power flow did not converge; attribution is undefined."}

    violations = _collect_violations(net, vm_lower, vm_upper,
                                     max_line_loading_pct, max_trafo_loading_pct)
    if not violations:
        return {"timestamp": timestamp, "converged": True, "violations": [],
                "n_sources_tested": 0, "n_power_flows": 1,
                "message": "No violations at this operating point; nothing to attribute."}

    base_values = _measure(net, violations)
    sources = _collect_sources(net, top_n_sources)
    controlled = _voltage_controlled_buses(net)

    # sensitivities[violation_index][source_index] = (d_per_mw, d_per_mvar)
    sensitivities: dict[int, dict[int, dict]] = {i: {} for i in range(len(violations))}
    non_converged: list[str] = []
    power_flows = 1

    for s_i, source in enumerate(sources):
        frame = getattr(net, source["table"])
        idx = source["index"]

        for column, delta, key in (("p_mw", delta_mw, "d_per_mw"),
                                   ("q_mvar", delta_mvar, "d_per_mvar")):
            if column not in frame.columns:
                continue
            original = frame.at[idx, column]
            frame.at[idx, column] = original + delta
            ok = _safe_runpp(net)
            power_flows += 1

            if ok:
                moved = _measure(net, violations)
                for v_i, (before, after) in enumerate(zip(base_values, moved)):
                    sensitivities[v_i].setdefault(s_i, {})[key] = (after - before) / delta
            elif source["source"] not in non_converged:
                non_converged.append(source["source"])

            frame.at[idx, column] = original

    # Restore the base solution so the caller sees an unperturbed network.
    _safe_runpp(net)

    reported = []
    for v_i, violation in enumerate(violations):
        drivers = []
        for s_i, source in enumerate(sources):
            sens = sensitivities[v_i].get(s_i)
            if not sens:
                continue
            d_mw = sens.get("d_per_mw")
            d_mvar = sens.get("d_per_mvar")
            influence = max(abs(d_mw or 0.0), abs(d_mvar or 0.0))
            if influence < 1e-9:
                continue
            drivers.append({
                "source": source["source"],
                "kind": source["kind"],
                "controllable": source["controllable"],
                "current_p_mw": source["current_p_mw"],
                "current_q_mvar": source["current_q_mvar"],
                "d_per_mw": round(d_mw, 6) if d_mw is not None else None,
                "d_per_mvar": round(d_mvar, 6) if d_mvar is not None else None,
                "relief_mw": _relief(violation["value"], violation["limit"], d_mw),
                "relief_mvar": _relief(violation["value"], violation["limit"], d_mvar),
                "_influence": influence,
            })
            drivers[-1]["relief_mw_feasible"] = _relief_feasible(
                drivers[-1]["relief_mw"], source["current_p_mw"])
            drivers[-1]["relief_mvar_feasible"] = _relief_feasible(
                drivers[-1]["relief_mvar"], source["current_q_mvar"])

        drivers.sort(key=lambda d: d["_influence"], reverse=True)

        # Rank by influence, but never return a list an operator cannot act on.
        # The largest driver is often a load, which explains the regime without
        # offering a lever; the controllable sources that do offer one can sit
        # well down the ranking. Both belong in the answer.
        selected = drivers[:top_n_drivers]
        if not any(d["controllable"] for d in selected):
            extra = [d for d in drivers[top_n_drivers:] if d["controllable"]]
            selected = selected + extra[:min_controllable]

        for d in drivers:
            d.pop("_influence", None)
        drivers = selected

        entry = {
            **violation,
            "margin": round(violation["value"] - violation["limit"], 6),
            "drivers": drivers,
        }
        controller = controlled.get(violation["index"]) if violation["kind"] == "bus" else None
        if controller:
            entry["voltage_controlled_by"] = controller
            entry["explanation"] = (
                f"This bus is voltage-controlled by {controller}, which holds it "
                "at a setpoint. No injection elsewhere can move it, so the "
                "sensitivities are zero by construction — the remedy is the "
                f"setpoint of {controller}, not redispatch of other units."
            )
        # Resolved first: the recommendation needs the raw reliefs to report
        # how far short the best single source falls.
        entry["recommended_action"] = _recommended_action(entry, drivers)

        for d in drivers:
            _withdraw_infeasible(d, "relief_mw", "relief_mw_feasible",
                                 "current_p_mw", "max_deliverable_mw")
            _withdraw_infeasible(d, "relief_mvar", "relief_mvar_feasible",
                                 "current_q_mvar", "max_deliverable_mvar")

        reported.append(entry)

    notes = [
        "Sensitivities are local finite differences; they support small "
        "corrective movements, not large counterfactuals.",
        "relief_mw / relief_mvar is the movement that would bring the element "
        "to its limit assuming the source acts alone. A movement the source "
        "provably cannot make is withdrawn (null) rather than reported; "
        "max_deliverable_mw / max_deliverable_mvar then gives the most it "
        "could contribute.",
        "Each violation carries a `recommended_action` resolved here from the "
        "feasibility of every candidate movement. Report that rather than "
        "assembling an action from the driver fields.",
        "The external grid is the slack and absorbs the resulting imbalance, so "
        "it is not perturbed.",
    ]
    if non_converged:
        notes.append(
            "Power flow did not converge while perturbing: "
            + ", ".join(non_converged) + ". Those sources are omitted."
        )

    return {
        "timestamp": timestamp,
        "converged": True,
        "method": "numerical perturbation of the validated power flow",
        "delta_mw": delta_mw,
        "delta_mvar": delta_mvar,
        "n_sources_tested": len(sources),
        "n_power_flows": power_flows,
        "violations": reported,
        "notes": notes,
    }
