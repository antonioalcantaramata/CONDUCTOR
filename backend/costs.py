"""
What each unit costs: to produce, and to be redispatched up or down — the
operator's to see and change at any time.

Per unit (and per external-grid connection):

- **production** — the case's cost curve, cost(P) = c0 + c1·P + c2·P²:
  ``cost_eur_per_h`` (c0), ``cost_eur_per_mwh`` (c1), ``cost_eur_per_mw2h``
  (c2). For the external grid, c1 is the import price.
- **redispatch** — what moving the unit away from its schedule costs the
  operator: ``up_eur_per_mwh`` (each MWh more) and ``down_eur_per_mwh`` (each
  MWh less; for a renewable unit, curtailment).
- **reactive power** — ``q_eur_per_mvarh``: one price on each Mvarh the unit
  is asked to change, injecting or absorbing alike (its cost is losses, and
  far below energy: reactive services pay a few €/Mvarh). For the external
  grid instead, the TSO–DSO interface charge: reactive exchange within
  ``q_band_tan_phi`` (|Q| ≤ tan φ · |P|) is free, each Mvarh beyond it costs
  ``q_penalty_eur_per_mvarh``.

They price the flexibility options (flex_options.py) and define the cheapest
one; the default corrective dispatch still minimises the change in dispatch
itself.

Defaults, in order: the case's own cost table (pandapower ``poly_cost``,
from a MATPOWER/PGLib ``gencost``), else the **synthetic** rule below —
deterministic, the same network always gets the same costs, and labelled
synthetic wherever shown. Redispatch costs have no source in a case file and
are always synthetic until changed:

- production: a unit 50 €/MWh, the import 60 €/MWh, a condenser 0;
  a pv or wind unit a proposal added 0;
- up: 1.2 × the unit's marginal cost at full output (c1 + 2·c2·Pmax) — a
  premium over its own energy price;
- down: 0.5 × that marginal cost — the operator compensates the lost margin,
  the fuel is saved; a pv or wind unit 40 €/MWh (curtailment compensation);
- condensers and the external grid have no up/down (they are not dispatched
  for active power), until the operator gives one;
- reactive: 3 €/Mvarh for every unit, condensers included; the external
  grid's free band tan φ 0.4 (pf ≈ 0.93) and 10 €/Mvarh beyond it.

Per load, what it costs to reduce it — the remedies after unit redispatch,
in that order (flex_options.py):

- **demand response** — ``dr_share_pct`` of the load can be reduced on request
  (contracted flexibility, an aggregator), at ``dr_eur_per_mwh``;
- **load relief** — involuntary reduction of the rest, the last resort, at
  ``relief_eur_per_mwh`` (the value of lost load).

No case file has these; synthetic until changed: 10 % of every load at
150 €/MWh, load relief at 3,000 €/MWh.

The operator's changes are kept as a difference from the defaults, per unit
(or load), like the generator model (generators.py). Pure functions: no app
state.
"""

from __future__ import annotations

import copy
import math

import pandas as pd

import generators as gm

PRODUCTION = ("cost_eur_per_h", "cost_eur_per_mwh", "cost_eur_per_mw2h")
REDISPATCH = ("up_eur_per_mwh", "down_eur_per_mwh")
REACTIVE = ("q_eur_per_mvarh",)
INTERFACE = ("q_band_tan_phi", "q_penalty_eur_per_mvarh")      # the external grid only
LOAD = ("dr_share_pct", "dr_eur_per_mwh", "relief_eur_per_mwh")  # loads only
FIELDS = PRODUCTION + REDISPATCH + REACTIVE + INTERFACE + LOAD
MAX_TAN_PHI = 2.0
_POLY = {"cost_eur_per_h": "cp0_eur", "cost_eur_per_mwh": "cp1_eur_per_mw", "cost_eur_per_mw2h": "cp2_eur_per_mw2"}
# Bounds well beyond any real price, to catch a typo (a € figure in cents).
MAX_PRICE = 10_000.0            # €/MWh and €/h
MAX_QUADRATIC = 100.0           # €/MW²h

SYNTHETIC_UNIT, SYNTHETIC_IMPORT = 50.0, 60.0
UP_PREMIUM, DOWN_SHARE, CURTAILMENT = 1.2, 0.5, 40.0
SYNTHETIC_Q, SYNTHETIC_TAN_PHI, SYNTHETIC_Q_PENALTY = 3.0, 0.4, 10.0
SYNTHETIC_DR_SHARE, SYNTHETIC_DR, SYNTHETIC_RELIEF = 10.0, 150.0, 3000.0
RENEWABLE_KINDS = ("pv", "wind")
RULE = ("Synthetic where the case gives no cost: a unit 50 €/MWh, the import 60 €/MWh, condensers 0; "
        "up = 1.2 × marginal cost at full output, down = 0.5 × it; added pv/wind units 0 to produce and "
        "40 €/MWh to curtail; condensers and the external grid are not redispatched; reactive power 3 €/Mvarh "
        "per unit; at the external grid, reactive exchange free within tan φ 0.4 (pf ≈ 0.93), 10 €/Mvarh beyond. "
        "Loads: 10 % of each can be reduced as demand response at 150 €/MWh; load relief beyond it at 3,000 €/MWh.")
LOAD_RULE = ("Synthetic until changed: 10 % of each load available as demand response at 150 €/MWh; "
             "involuntary load relief of the rest at 3,000 €/MWh (the value of lost load).")


class CostError(ValueError):
    """A cost change that cannot be applied; the message says why."""


def _finite(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def ext_grid_name(net, idx) -> str:
    """The external grid as every table names it: "External grid", numbered
    when there are several."""
    if len(net.ext_grid) <= 1:
        return "External grid"
    return f"External grid {list(net.ext_grid.index).index(idx) + 1}"


def _case_cost(net, et: str, idx) -> dict | None:
    """The case's cost curve for one element, or None when it has none."""
    pc = net.get("poly_cost") if hasattr(net, "get") else None
    if pc is None or not len(pc):
        return None
    rows = pc[(pc["et"] == et) & (pc["element"] == idx)]
    if not len(rows):
        return None
    row = rows.iloc[0]
    return {k: float(row[col]) if col in row.index and pd.notna(row[col]) else 0.0 for k, col in _POLY.items()}


def _redispatch(production: dict, p_max: float, *, renewable: bool, condenser: bool) -> dict:
    if condenser:
        return {"up_eur_per_mwh": None, "down_eur_per_mwh": None}
    if renewable:
        return {"up_eur_per_mwh": None, "down_eur_per_mwh": CURTAILMENT}
    marginal = production["cost_eur_per_mwh"] + 2.0 * production["cost_eur_per_mw2h"] * max(p_max, 0.0)
    return {"up_eur_per_mwh": round(UP_PREMIUM * marginal, 4), "down_eur_per_mwh": round(DOWN_SHARE * marginal, 4)}


def defaults(net) -> list[dict]:
    """Every unit and external grid with its default costs and where each
    figure comes from ("case" or "synthetic")."""
    out = []
    for idx in net.ext_grid.index:
        case = _case_cost(net, "ext_grid", idx)
        prod = case or {"cost_eur_per_h": 0.0, "cost_eur_per_mwh": SYNTHETIC_IMPORT, "cost_eur_per_mw2h": 0.0}
        out.append({"unit": ext_grid_name(net, idx), "kind": "external_grid", "table": "ext_grid",
                    "index": int(idx), "bus": int(net.ext_grid.at[idx, "bus"]), "p_max_mw": None, **prod,
                    "up_eur_per_mwh": None, "down_eur_per_mwh": None, "q_eur_per_mvarh": None,
                    "q_band_tan_phi": SYNTHETIC_TAN_PHI, "q_penalty_eur_per_mvarh": SYNTHETIC_Q_PENALTY,
                    "source": {**{k: "case" if case else "synthetic" for k in PRODUCTION},
                               **{k: "synthetic" for k in REDISPATCH + REACTIVE + INTERFACE}}})
    for table in ("gen", "sgen"):
        tbl = net[table]
        for idx in tbl.index:
            row = tbl.loc[idx]
            proposed = gm._proposed(tbl, idx)
            p_max = gm._capacity(row)
            condenser = p_max < gm.CONDENSER_BELOW_MW
            kind = str(row.get("kind") or "") if proposed and "kind" in tbl.columns else ""
            renewable = kind in RENEWABLE_KINDS
            case = None if proposed else _case_cost(net, table, idx)
            synth = 0.0 if (condenser or renewable) else SYNTHETIC_UNIT
            prod = case or {"cost_eur_per_h": 0.0, "cost_eur_per_mwh": synth, "cost_eur_per_mw2h": 0.0}
            out.append({"unit": gm.unit_name(net, table, idx), "kind": kind or ("condenser" if condenser else "unit"),
                        "table": table, "index": int(idx), "bus": int(row["bus"]), "p_max_mw": round(p_max, 4),
                        "proposed": proposed, **prod,
                        **_redispatch(prod, p_max, renewable=renewable, condenser=condenser),
                        "q_eur_per_mvarh": SYNTHETIC_Q,
                        "source": {**{k: "case" if case else "synthetic" for k in PRODUCTION},
                                   **{k: "synthetic" for k in REDISPATCH + REACTIVE}}})
    return out


def _bus_names(net, bus) -> set[str]:
    """What a bus answers to, casefolded: "bus_29", its name as loaded and now."""
    names = {f"bus_{int(bus)}", f"bus {int(bus)}"}
    for col in ("name", "loaded_name"):
        if col in net.bus.columns:
            v = net.bus.at[bus, col]
            if isinstance(v, str) and v.strip():
                names.add(v.strip().casefold())
    return names


def load_name(net, idx) -> str:
    """A load as the cost table keys it: its name as loaded ("Load 20")."""
    tbl = net.load
    kept = tbl.at[idx, "loaded_name"] if "loaded_name" in tbl.columns else None
    raw = kept if isinstance(kept, str) else tbl.at[idx, "name"] if "name" in tbl.columns else None
    raw = str(raw).strip() if raw is not None and raw == raw else ""
    return raw or f"Load {int(idx)}"


def load_defaults(net) -> list[dict]:
    """Every load with its default demand-response and load-relief prices."""
    out = []
    tbl = net.load
    for idx in tbl.index:
        proposed = gm._proposed(tbl, idx)
        out.append({"unit": load_name(net, idx), "kind": "load", "table": "load", "index": int(idx),
                    "bus": int(tbl.at[idx, "bus"]), "proposed": proposed,
                    "dr_share_pct": SYNTHETIC_DR_SHARE, "dr_eur_per_mwh": SYNTHETIC_DR,
                    "relief_eur_per_mwh": SYNTHETIC_RELIEF, "source": {k: "synthetic" for k in LOAD}})
    return out


def load_table(net, changes: dict | None = None) -> list[dict]:
    """The loads' demand-response and relief prices in force."""
    changes = changes or {}
    out = []
    for row in load_defaults(net):
        diff = {k: v for k, v in (changes.get(row["unit"]) or {}).items() if k in LOAD}
        out.append({**row, **diff, "source": {**row["source"], **{k: "operator" for k in diff}},
                    "changed": bool(diff)})
    return out


def table(net, changes: dict | None = None) -> list[dict]:
    """The costs in force: the defaults with the operator's changes."""
    changes = changes or {}
    out = []
    for row in defaults(net):
        diff = changes.get(row["unit"]) or {}
        row = {**row, **diff, "source": {**row["source"], **{k: "operator" for k in diff}},
               "changed": bool(diff)}
        out.append(row)
    return out


def normalise(net, raw_changes: list[dict], current: dict | None = None) -> dict:
    """Check the operator's changes and return, per unit named, its full
    difference from the defaults ({} when it is back to them). A field set to
    null goes back to its default."""
    if not isinstance(raw_changes, list) or not raw_changes:
        raise CostError("Give a list of cost changes, each with a 'unit'.")
    current = current or {}
    rows = {r["unit"]: r for r in defaults(net) + load_defaults(net) if not r.get("proposed")}
    out: dict[str, dict] = {}
    for n, raw in enumerate(raw_changes, start=1):
        if not isinstance(raw, dict):
            raise CostError(f"Change {n}: each change is an object with a 'unit'.")
        name = str(raw.get("unit") or "").strip()
        row = rows.get(name) or next((r for k, r in rows.items() if k.casefold() == name.casefold()), None)
        if row is None and any(w in name.casefold() for w in ("grid", "slack", "import")) \
                and len(net.ext_grid) == 1:
            row = rows[ext_grid_name(net, net.ext_grid.index[0])]
        if row is None:
            # A load named by its bus ("Bus_29", "the load at Bus_29"), when the bus has one.
            at_bus = [r for r in rows.values() if r["kind"] == "load" and name.casefold().removeprefix(
                "the load at ").removeprefix("load at ").strip() in _bus_names(net, r["bus"])]
            row = at_bus[0] if len(at_bus) == 1 else None
        if row is None:
            units = sorted(k for k, r in rows.items() if r["kind"] != "load")
            loads = [k for k, r in rows.items() if r["kind"] == "load"]
            raise CostError(f"Change {n}: there is no unit or load {name!r}. Units: {units[:20]}"
                            + (f"; loads: {loads[0]} … {loads[-1]} (or a load's bus)." if loads else "."))
        name = row["unit"]
        if name in out:
            raise CostError(f"Change {n}: {name} is changed twice; give one change per unit.")
        unknown = sorted(set(raw) - set(FIELDS) - {"unit"})
        if unknown:
            raise CostError(f"{name}: unknown field(s) {unknown}. Fields: {list(FIELDS)}.")
        given = {k: raw[k] for k in FIELDS if k in raw}
        if not given:
            raise CostError(f"{name}: give at least one of {list(FIELDS)}.")
        grid = row["kind"] == "external_grid"
        load = row["kind"] == "load"
        if load != all(k in LOAD for k in given):
            wrong = [k for k in given if (k in LOAD) != load]
            raise CostError(f"{name}: leave out {', '.join(wrong)} " + (
                f"(units only); a load's fields are {', '.join(LOAD)}." if load else
                f"(loads only); for a unit give {', '.join(PRODUCTION + REDISPATCH + REACTIVE)}."))
        misplaced = [k for k in given if (k in INTERFACE) != grid and (k in INTERFACE or k in REACTIVE)]
        if misplaced:
            raise CostError(f"{name}: leave out {', '.join(misplaced)} " + (
                "(units only); the external grid's reactive charge is q_band_tan_phi and "
                "q_penalty_eur_per_mvarh." if grid else
                "(external grid only); a unit's reactive price is q_eur_per_mvarh."))
        diff = dict(current.get(name) or {})
        for key, value in given.items():
            if value is None or value == "":
                diff.pop(key, None)               # back to the default
                continue
            v = _finite(value)
            high = (MAX_QUADRATIC if key == "cost_eur_per_mw2h" else MAX_TAN_PHI if key == "q_band_tan_phi"
                    else 100.0 if key == "dr_share_pct" else MAX_PRICE)
            low = -MAX_PRICE if key == "cost_eur_per_mwh" else 0.0   # a negative energy price is real
            if v is None or not (low <= v <= high):
                raise CostError(f"{name}: {key} must be a number between {low:g} and {high:g}.")
            diff[key] = v
        out[name] = {k: v for k, v in diff.items() if v != row[k]}
    return out


def merge(model: dict, changes: dict) -> dict:
    """`model` with each named unit's difference replaced; an empty one leaves."""
    merged = copy.deepcopy(model)
    for name, diff in changes.items():
        if diff:
            merged[name] = dict(diff)
        else:
            merged.pop(name, None)
    return merged


def rename(model: dict, old: str, new: str) -> dict:
    """The same changes after a unit is renamed."""
    if old not in model:
        return model
    out = {k: v for k, v in model.items() if k != old}
    out[new] = model[old]
    return out
