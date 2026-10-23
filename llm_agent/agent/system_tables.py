"""
system_tables.py — the System tab's cost and name panels, without the page.

Like the generator panel (generator_table.py): plain-text tables, crisp at any
zoom, each row with an Edit button that opens a small form. Pure functions over
`/api/costs` and `/api/names` payloads, so they can be tested without
Streamlit (importing app.py runs the whole page).
"""

from __future__ import annotations

import math

# Field → (form label, table column, number format). Order is the table's.
PRODUCTION = {
    "cost_eur_per_mwh": ("Production (€/MWh)", "€/MWh", ".2f"),
    "cost_eur_per_mw2h": ("Quadratic (€/MW²h)", "€/MW²h", ".4g"),
    "cost_eur_per_h": ("Fixed (€/h)", "€/h", ".2f"),
}
REDISPATCH = {
    "up_eur_per_mwh": ("Upward redispatch (€/MWh)", "Up €/MWh", ".2f"),
    "down_eur_per_mwh": ("Downward redispatch (€/MWh; curtailment for renewables)", "Down €/MWh", ".2f"),
}
REACTIVE = {"q_eur_per_mvarh": ("Reactive power (€/Mvarh, injecting or absorbing)", "Q €/Mvarh", ".2f")}
INTERFACE = {
    "q_band_tan_phi": ("Free reactive band (tan φ; 0.4 ≈ pf 0.93)", "Free band tan φ", ".2f"),
    "q_penalty_eur_per_mvarh": ("Beyond the band (€/Mvarh)", "Beyond €/Mvarh", ".2f"),
}
LOAD = {
    "dr_share_pct": ("Demand response: share of the load reducible on request (%)", "DR share %", ".0f"),
    "dr_eur_per_mwh": ("Demand response price (€/MWh)", "DR €/MWh", ".2f"),
    "relief_eur_per_mwh": ("Load relief, the last resort (€/MWh)", "Relief €/MWh", ".0f"),
}

GROUPS = {
    "units": {
        "title": "Units",
        "caption": "Production cost = €/h + €/MWh × P + €/MW²h × P². Up / Down price redispatch; Q prices "
                   "the reactive power a unit is asked to inject or absorb.",
        "columns": ["Unit", "Bus", *(c for _, c, _ in PRODUCTION.values()),
                    *(c for _, c, _ in REDISPATCH.values()), *(c for _, c, _ in REACTIVE.values()), "Source"],
        "widths": [1.1, 0.8, 0.8, 0.8, 0.7, 0.9, 0.9, 0.9, 1.5],
    },
    "grid": {
        "title": "External grid",
        "caption": "The import price is its production cost. Reactive exchange within the band "
                   "(|Q| ≤ tan φ · |P|) is free; each Mvarh beyond it is charged.",
        "columns": ["Connection", "Bus", "Import €/MWh", *(c for _, c, _ in INTERFACE.values()), "Source"],
        "widths": [1.1, 0.8, 1.0, 1.0, 1.0, 1.5],
    },
    "loads": {
        "title": "Demand response and load relief",
        "caption": "The remedies after redispatching the units, in order: a share of each load reduced on request "
                   "(demand response, at its price), then involuntary load relief of the rest (the value of lost "
                   "load). Used only when asked for, or when redispatch alone cannot clear a violation.",
        "columns": ["Load", "Bus", *(c for _, c, _ in LOAD.values()), "Source"],
        "widths": [1.0, 0.9, 0.9, 0.9, 0.9, 1.3],
    },
}

SOURCE_LABELS = {"case": "case file", "synthetic": "synthetic", "operator": "changed"}
# Short forms for the table's Source column (the form's help says it in full).
_SHORT = {"case file": "case", "synthetic": "synth.", "changed": "changed"}


def _num(x, fmt: str) -> str:
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else format(x, fmt)


def is_grid(unit: dict) -> bool:
    return unit.get("kind") == "external_grid"


def is_load(unit: dict) -> bool:
    return unit.get("kind") == "load"


def fields_of(unit: dict) -> dict:
    """The cost fields a row's form shows, in sections: {section: {field: spec}}."""
    if is_load(unit):
        return {"Demand response": {k: LOAD[k] for k in ("dr_share_pct", "dr_eur_per_mwh")},
                "Load relief": {"relief_eur_per_mwh": LOAD["relief_eur_per_mwh"]}}
    if is_grid(unit):
        return {"Import price": {"cost_eur_per_mwh": ("Import price (€/MWh)", "", ".2f")},
                "Reactive exchange": INTERFACE}
    return {"Production": PRODUCTION, "Redispatch": REDISPATCH, "Reactive": REACTIVE}


def _source(unit: dict, keys) -> str:
    found = sorted({SOURCE_LABELS.get((unit.get("source") or {}).get(k), "?") for k in keys})
    return " + ".join(_SHORT.get(f, f) for f in found)


def cost_rows(units: list[dict]) -> dict[str, list[dict]]:
    """{group: [{"unit", "changed", "cells"}]} for the cost tables."""
    out: dict[str, list[dict]] = {"units": [], "grid": [], "loads": []}
    for u in units:
        bus = str(u.get("bus_name") or u.get("bus"))
        if is_load(u):
            cells = [u["unit"], bus, *(_num(u.get(k), f) for k, (_, _, f) in LOAD.items()), _source(u, LOAD)]
            out["loads"].append({"unit": u["unit"], "changed": bool(u.get("changed")), "cells": cells})
        elif is_grid(u):
            cells = [u["unit"], bus, _num(u.get("cost_eur_per_mwh"), ".2f"),
                     *(_num(u.get(k), f) for k, (_, _, f) in INTERFACE.items()),
                     _source(u, ("cost_eur_per_mwh", *INTERFACE))]
            out["grid"].append({"unit": u["unit"], "changed": bool(u.get("changed")), "cells": cells})
        else:
            specs = {**PRODUCTION, **REDISPATCH, **REACTIVE}
            cells = [u["unit"], bus, *(_num(u.get(k), f) for k, (_, _, f) in specs.items()),
                     _source(u, specs)]
            out["units"].append({"unit": u["unit"], "changed": bool(u.get("changed")), "cells": cells})
    return {k: v for k, v in out.items() if v}


def cost_changes(unit: dict, values: dict) -> dict | None:
    """The change to send for one unit's form, or None when nothing changed.
    An emptied field is sent as null (back to its default)."""
    change = {}
    for section in fields_of(unit).values():
        for key in section:
            if key not in values:
                continue
            new, old = values[key], unit.get(key)
            if new is None and old is None:
                continue
            if new is not None and old is not None and abs(float(new) - float(old)) < 1e-9:
                continue
            change[key] = None if new is None else float(new)
    return {"unit": unit["unit"], **change} if change else None


def defaults_change(unit: dict) -> dict | None:
    """The change that takes every operator figure of a unit back to its default."""
    changed = [k for k, src in (unit.get("source") or {}).items() if src == "operator"]
    return {"unit": unit["unit"], **{k: None for k in changed}} if changed else None


NAME_KINDS = {"bus": "Buses", "line": "Lines", "trafo": "Transformers", "load": "Loads",
              "gen": "Generators (voltage control)", "sgen": "Generators (power factor)"}
NAME_SINGULAR = {"bus": "Bus", "line": "Line", "trafo": "Transformer", "load": "Load",
                 "gen": "Generator", "sgen": "Generator"}
NAME_COLUMNS = {
    "bus": ["Name", "Index"],
    "line": ["Name", "Between", "Index"],
    "trafo": ["Name", "Between", "Index"],
    "load": ["Name", "Bus", "Index"],
    "gen": ["Name", "Bus", "Index"],
    "sgen": ["Name", "Bus", "Index"],
}


def name_rows(kind: str, elements: list[dict]) -> list[dict]:
    """[{"index", "name", "renamed", "cells"}] for one kind's names table. A
    line or transformer without a name of its own shows "(no name)"."""
    rows = []
    for e in elements:
        own = e.get("label") or ""
        cells = [own or "(no name)"]
        if kind in ("line", "trafo"):
            cells.append(e.get("ends") or "")
        elif kind != "bus":
            cells.append(e.get("bus") or "")
        cells.append(str(e["index"]))
        rows.append({"index": e["index"], "name": own, "shown": e.get("name") or own,
                     "renamed": bool(e.get("renamed")), "cells": cells})
    return rows


_KEPT_WORDS = {"renames": ("rename", "renames"), "costs": ("cost change", "cost changes"),
               "generators": ("generator model", "generator models"),
               "load_power_factor": ("load power factor", "load power factor"),
               "proposals": ("proposal", "proposals")}


def kept_line(kept: dict | None) -> str:
    """"Your changes are kept: 1 rename · 2 proposals" — or "" when none."""
    parts = [f"{n} {_KEPT_WORDS[k][n != 1]}" if k != "load_power_factor" else _KEPT_WORDS[k][0]
             for k, n in (kept or {}).items() if n and k in _KEPT_WORDS]
    return ("Your changes are kept: " + " · ".join(parts)) if parts else ""
