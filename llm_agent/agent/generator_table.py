"""
generator_table.py — the System tab's generator panel, without the page.

Units are listed in up to three small read-only tables — plain text, so they
stay crisp at any zoom (an editable grid is drawn on a canvas and blurs) — each
showing only what matters for that kind of unit:

- generators on voltage control — set-point and reactive range;
- generators at a fixed power factor — power factor and direction;
- synchronous condensers — set-point and reactive range.

A unit is edited in a small form of its own (`form_fields`, `form_changes`).
Pure functions over `/api/generators` payloads, so they can be tested without
Streamlit (importing app.py runs the whole page).
"""

from __future__ import annotations

import math

CONTROL_LABELS = {"voltage": "Voltage control", "power_factor": "Power factor", "fixed_q": "Fixed Q"}

GROUPS = {
    "voltage": {
        "title": "Generators on voltage control",
        "caption": "Each holds its bus at the set-point V; its reactive power is whatever that takes, "
                   "kept within its reactive range by the OPF.",
        "columns": ["Unit", "Bus", "Pmax", "V set-point", "Reactive range", "Dispatchable"],
    },
    "power_factor": {
        "title": "Generators at a fixed power factor",
        "caption": "Each injects its active power with reactive power in a fixed ratio "
                   "(Q = P · tan(acos pf)); pf 1.00 is none. Its bus voltage is free to move.",
        "columns": ["Unit", "Bus", "Pmax", "Power factor", "Dispatchable"],
    },
    "fixed_q": {
        "title": "Generators at a fixed reactive output",
        "caption": "Each injects or absorbs a set reactive power, whatever its active power. Its bus "
                   "voltage is free to move.",
        "columns": ["Unit", "Bus", "Pmax", "Reactive output", "Dispatchable"],
    },
    "condenser": {
        "title": "Synchronous condensers",
        "caption": "Machines with no active power that only produce or absorb reactive power to hold "
                   "their bus voltage at V, within their reactive range.",
        "columns": ["Unit", "Bus", "V set-point", "Reactive range"],
    },
}


def group_of(unit: dict) -> str:
    if unit.get("condenser"):
        return "condenser"
    return unit["control"] if unit["control"] in GROUPS else "power_factor"


def _num(x, fmt: str) -> str:
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else format(x, fmt)


def _qrange(u: dict) -> str:
    lo, hi = u.get("q_min_mvar"), u.get("q_max_mvar")
    if lo is None or hi is None:
        return "—"
    return f"{_num(lo, 'g')} to {_num(hi, 'g')} Mvar"


def _pf(u: dict) -> str:
    pf = u.get("pf")
    if pf is None:
        return "—"
    if pf >= 1.0:
        return "1.00 (no reactive power)"
    return f"{pf:.2f} {'absorbing' if u.get('absorbing') else 'injecting'}"


def _qset(u: dict) -> str:
    q = u.get("q_mvar")
    if q is None:
        return "—"
    return f"{abs(q):g} Mvar {'absorbed' if q < 0 else 'injected'}" if q else "0 Mvar"


def rows(units: list[dict], changed: dict) -> dict[str, list[dict]]:
    """Display rows per table (only non-empty tables), every cell a string."""
    out: dict[str, list[dict]] = {}
    for u in units:
        g = group_of(u)
        cells = {"Unit": u["unit"], "Bus": u.get("bus_name") or str(u["bus"]),
                 "Pmax": f"{_num(u['p_max_mw'], 'g')} MW", "V set-point": f"{_num(u['vm_pu'], '.3f')} p.u.",
                 "Reactive range": _qrange(u), "Power factor": _pf(u), "Reactive output": _qset(u),
                 "Dispatchable": "yes" if u.get("dispatchable") else "no"}
        out.setdefault(g, []).append({"unit": u["unit"], "changed": u["unit"] in changed,
                                      "cells": [cells[c] for c in GROUPS[g]["columns"]]})
    return {g: out[g] for g in GROUPS if g in out}


def form_fields(unit: dict, control: str) -> list[str]:
    """The fields a unit's form shows for a control (a condenser: no control
    choice and nothing to dispatch)."""
    if unit.get("condenser"):
        return ["vm_pu", "q_min_mvar", "q_max_mvar"]
    if control == "voltage":
        return ["control", "vm_pu", "q_min_mvar", "q_max_mvar", "dispatchable"]
    if control == "fixed_q":
        return ["control", "q_mvar", "dispatchable"]
    return ["control", "pf", "absorbing", "dispatchable"]


def _same(a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        return abs(float(a) - float(b)) < 1e-9
    return a == b


def form_changes(unit: dict, values: dict) -> dict | None:
    """The unit change a submitted form makes: only the fields shown and
    changed. None when nothing changed."""
    control = values.get("control", unit["control"])
    diff = {}
    for field in form_fields(unit, control):
        if field in values and not _same(values[field], unit.get(field)):
            v = values[field]
            diff[field] = bool(v) if field in ("absorbing", "dispatchable") else v if field == "control" \
                else float(v)
    return {"unit": unit["unit"], **diff} if diff else None
