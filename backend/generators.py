"""
How each generating unit is modelled: the operator's choice, changeable at any
time.

A unit has two independent properties, as a real machine does:

- **control** — how it behaves in a power flow:
  - ``"voltage"``: holds a voltage set-point; its reactive power is whatever
    that takes (a PV bus; pandapower's ``gen`` table). The OPF keeps its Q
    within the unit's own limits.
  - ``"power_factor"``: injects P at a fixed power factor, injecting or
    absorbing reactive power (a PQ unit; the ``sgen`` table). pf 1.0 is Q = 0.
    The OPF limits its Q by the study's power-factor cone.
  - ``"fixed_q"``: injects (> 0) or absorbs (< 0) a set reactive power
    whatever its active power (a PQ unit; ``sgen``). In the OPF it is treated
    as a power-factor unit (its Q may be redispatched within the cone).
- **dispatchable** — whether the OPF may move its active power. A unit that is
  not dispatchable keeps its output in every optimisation.

The loaded network is the reference: a model is a set of changes to it, and
applying it always starts from the network as loaded, so it can be applied,
changed and reset any number of times with the same result. Converting a
generator on load (``network_loader.gen_to_sgen``) keeps its voltage set-point
and reactive limits on the row, so it can be switched back to voltage control.

Units are named as the OPF names them (``unit_name``), the key everywhere a
unit appears. Units a proposal added are not part of the model.

Pure functions of a pandapower network: no app state, no HTTP.
"""

from __future__ import annotations

import copy
import math

import pandas as pd
import pandapower as pp

VOLTAGE, POWER_FACTOR, FIXED_Q = "voltage", "power_factor", "fixed_q"
CONTROLS = (VOLTAGE, POWER_FACTOR, FIXED_Q)
CONDENSER_BELOW_MW = 1.0          # the engine's rule (flex_engine._is_condenser)
VM_RANGE = (0.8, 1.2)
PF_MIN = 0.5
# Kept on a converted row so the unit can go back to voltage control.
ORIG_VM, ORIG_QMIN, ORIG_QMAX = "orig_vm_pu", "orig_min_q_mvar", "orig_max_q_mvar"
# The fixed power factor of a power-factor unit, read at every tick.
PF_COL, ABSORB_COL = "q_pf", "q_absorbing"
# The set reactive output of a fixed-Q unit (Mvar; > 0 injects), read at every tick.
QSET_COL = "q_set_mvar"


class ModelError(ValueError):
    """A unit model that cannot be applied; the message says why."""


def _finite(value, default=None):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def unit_name(net, table: str, idx) -> str:
    """The unit's name as the OPF and every result use it."""
    row = net[table].loc[idx]
    if table == "sgen":
        col = "substation_name" if "substation_name" in net.sgen.columns else "name"
        raw = row.get(col)
        raw = raw if isinstance(raw, str) and raw.strip() else row.get("name")
        return raw.strip() if isinstance(raw, str) and raw.strip() else f"sgen_{idx}"
    raw = row.get("name")
    return raw.strip() if isinstance(raw, str) and raw.strip() else f"Gen_bus{int(row['bus'])}"


def unit_key(net, table: str, idx) -> str:
    """The unit's name as loaded — the key of its model, costs and nominal
    values, which stay its own after the operator renames it (naming.py)."""
    if "loaded_unit" in net[table].columns:
        kept = net[table].at[idx, "loaded_unit"]
        if isinstance(kept, str) and kept:
            return kept
    return unit_name(net, table, idx)


def _proposed(tbl: pd.DataFrame, idx) -> bool:
    return "proposed" in tbl.columns and tbl.at[idx, "proposed"] is True


def editable(net) -> tuple[bool, str]:
    """Whether per-unit models apply to this network. A network loaded with no
    `gen` table is a measured-substation network: its units are rebuilt from
    the measurements at every tick, so a model would not hold."""
    if len(net.gen) or net.get("_gen_to_sgen_applied", False):
        return True, ""
    return False, ("This network's units are rebuilt from the measurements at every tick "
                   "(a measured-substation network), so their model cannot be changed here.")


def _capacity(row) -> float:
    for col in ("max_p_mw", "p_mw"):
        v = _finite(row.get(col))
        if v is not None and v > 0:
            return v
    return 0.0


def is_condenser(table: str, row) -> bool:
    """The engine's rule (flex_engine._is_condenser), in one place: a
    voltage-controlled unit (`gen`) whose declared `max_p_mw` — or, with none
    declared, its output — is under 1 MW. A small static generator is an
    ordinary unit to the optimisation: the cost table used to call it a
    condenser too, with no redispatch price, so it moved for free."""
    if table != "gen":
        return False
    cap = _finite(row.get("max_p_mw"))
    if cap is None:
        cap = _finite(row.get("p_mw"), 0.0)
    return cap < CONDENSER_BELOW_MW


def describe(net) -> list[dict]:
    """Every unit of the network (not those a proposal added), as modelled now."""
    out = []
    for table in ("gen", "sgen"):
        tbl = net[table]
        for idx in tbl.index:
            if _proposed(tbl, idx):
                continue
            row = tbl.loc[idx]
            pmax = _capacity(row)
            if table == "gen":
                vm = _finite(row.get("vm_pu"), 1.0)
                qmin, qmax = _finite(row.get("min_q_mvar")), _finite(row.get("max_q_mvar"))
                pf, absorbing = None, False
            else:
                vm = _finite(row.get(ORIG_VM)) if ORIG_VM in tbl.columns else None
                qmin = _finite(row.get(ORIG_QMIN)) if ORIG_QMIN in tbl.columns else None
                qmax = _finite(row.get(ORIG_QMAX)) if ORIG_QMAX in tbl.columns else None
                pf = _finite(row.get(PF_COL)) if PF_COL in tbl.columns else None
                absorbing = bool(row.get(ABSORB_COL)) if ABSORB_COL in tbl.columns and \
                    pd.notna(row.get(ABSORB_COL)) else False
                qset = _finite(row.get(QSET_COL)) if QSET_COL in tbl.columns else None
                if pf is None:                     # as loaded: the row's own power factor
                    p, q = _finite(row.get("p_mw"), 0.0), _finite(row.get("q_mvar"), 0.0)
                    s = math.hypot(p, q)
                    pf = round(abs(p) / s, 4) if s > 1e-9 else 1.0
                    absorbing = q < 0 < p
            ctrl = row.get("controllable")
            if table == "gen":
                control, q_set = VOLTAGE, None
            else:
                control = FIXED_Q if qset is not None else POWER_FACTOR
                q_set = qset
            out.append({
                "unit": unit_name(net, table, idx), "bus": int(row["bus"]),
                "control": control, "q_mvar": q_set,
                "vm_pu": vm, "pf": pf, "absorbing": absorbing,
                "q_min_mvar": qmin, "q_max_mvar": qmax,
                "dispatchable": False if is_condenser(table, row) else (
                    True if ctrl is None or pd.isna(ctrl) else bool(ctrl)),
                "p_max_mw": round(pmax, 4),
                "condenser": is_condenser(table, row),
                "in_service": bool(row.get("in_service", True)),
            })
    return out


FIELDS = ("control", "vm_pu", "pf", "absorbing", "q_mvar", "q_min_mvar", "q_max_mvar", "dispatchable")


def normalise(net, changes: list[dict], model: dict | None = None) -> dict:
    """Check the operator's changes against the network as loaded (`net`) and
    the model already applied (`model`), and return, per changed unit, its full
    difference from the network as loaded — {} when the change takes it back to
    how it was loaded. Every unit at most once."""
    if not isinstance(changes, list):
        raise ModelError("Give a list of unit changes.")
    ok, why = editable(net)
    if not ok:
        raise ModelError(why)
    model = model or {}
    loaded = {u["unit"]: u for u in describe(net)}
    out: dict[str, dict] = {}
    for n, raw in enumerate(changes, start=1):
        if not isinstance(raw, dict):
            raise ModelError(f"Change {n}: each change is an object with a 'unit'.")
        name = str(raw.get("unit") or "").strip()
        unit = loaded.get(name) or next((u for k, u in loaded.items() if k.casefold() == name.casefold()), None)
        if unit is None:
            hint = ""
            if any(w in str(name).casefold() for w in ("grid", "slack", "ext")):
                hint = (" The external grid is not a generator unit: change its voltage set-point in a proposal "
                        "with edit_proposal op 'set_parameter' (element_type 'ext_grid', field 'vm_pu').")
            raise ModelError(f"Change {n}: there is no unit {name!r}. Units: {sorted(loaded)[:20]}.{hint}")
        name = unit["unit"]
        if name in out:
            raise ModelError(f"Change {n}: {name} is changed twice; give one change per unit.")
        c: dict = {}
        if raw.get("control") not in (None, ""):
            c["control"] = str(raw["control"]).strip().lower()
            if c["control"] not in CONTROLS:
                raise ModelError(f"{name}: control must be 'voltage', 'power_factor' or 'fixed_q'.")
        for key, low, high in (("vm_pu", *VM_RANGE), ("pf", PF_MIN, 1.0)):
            if raw.get(key) not in (None, ""):
                v = _finite(raw[key])
                if v is None or not (low <= v <= high):
                    raise ModelError(f"{name}: {key} must be between {low} and {high}.")
                c[key] = v
        for key in ("q_mvar", "q_min_mvar", "q_max_mvar"):
            if raw.get(key) not in (None, ""):
                v = _finite(raw[key])
                if v is None:
                    raise ModelError(f"{name}: {key} must be a number.")
                c[key] = v
        for key in ("absorbing", "dispatchable"):
            if raw.get(key) is not None:
                if not isinstance(raw[key], bool):
                    raise ModelError(f"{name}: {key} must be true or false.")
                c[key] = raw[key]
        effective = {**unit, **model.get(name, {}), **c}
        if unit["condenser"] and effective["control"] != VOLTAGE:
            raise ModelError(f"{name} is a synchronous condenser (no active power): it stays on voltage "
                             "control here — at a fixed power factor it would supply nothing. For fixed "
                             "reactive compensation, add a capacitor bank or reactor in a proposal (add_shunt).")
        if unit["condenser"] and effective["dispatchable"]:
            raise ModelError(f"{name} has no active power to dispatch.")
        qmin, qmax = effective["q_min_mvar"], effective["q_max_mvar"]
        if qmin is not None and qmax is not None and qmin > qmax:
            raise ModelError(f"{name}: q_min_mvar ({qmin:g}) is above q_max_mvar ({qmax:g}).")
        if effective["control"] == VOLTAGE and (qmin is None or qmax is None):
            raise ModelError(f"{name}: voltage control needs reactive limits (q_min_mvar and "
                             "q_max_mvar) — the unit has none on record, so give them.")
        if effective["control"] == FIXED_Q and effective.get("q_mvar") is None:
            raise ModelError(f"{name}: fixed-Q control needs the reactive output to hold (q_mvar, Mvar; "
                             "positive injects, negative absorbs).")
        if effective["control"] == VOLTAGE and effective["vm_pu"] is None:
            raise ModelError(f"{name}: voltage control needs a set-point (vm_pu) — the unit has "
                             "none on record, so give one.")
        out[name] = {k: effective[k] for k in FIELDS if effective[k] != unit[k]}
    return out


def merge(model: dict, changes: dict) -> dict:
    """`model` with each changed unit's difference replaced; a unit whose
    difference is empty is back to how it was loaded and leaves the model."""
    merged = copy.deepcopy(model)
    for name, diff in changes.items():
        if diff:
            merged[name] = dict(diff)
        else:
            merged.pop(name, None)
    return merged


def _row_of(net, name: str) -> tuple[str, object]:
    for table in ("gen", "sgen"):
        for idx in net[table].index:
            if not _proposed(net[table], idx) and unit_name(net, table, idx) == name:
                return table, idx
    raise ModelError(f"There is no unit {name!r}.")


def _move_cost(net, table: str, idx, new_table: str, new_idx) -> None:
    """The case's cost row follows its unit to the table its control needs
    (costs.py reads it there): left behind, the unit fell back to synthetic
    costs, and a unit later created at the freed index took them over."""
    pc = net.get("poly_cost")
    if pc is None or not len(pc):
        return
    rows = (pc["et"] == table) & (pc["element"] == idx)
    pc.loc[rows, "et"] = new_table
    pc.loc[rows, "element"] = new_idx


def apply(loaded_net, model: dict):
    """A copy of the network as loaded with `model` applied: units moved to the
    table their control needs, set-points and limits set. Raises ModelError."""
    net = copy.deepcopy(loaded_net)
    if not model:
        return net
    ok, why = editable(net)
    if not ok:
        raise ModelError(why)
    current = {u["unit"]: u for u in describe(net)}
    for name, change in model.items():
        if name not in current:
            raise ModelError(f"There is no unit {name!r} in the loaded network.")
        want = {**current[name], **change}
        table, idx = _row_of(net, name)
        row = net[table].loc[idx].to_dict()
        common = dict(bus=int(row["bus"]), p_mw=_finite(row.get("p_mw"), 0.0),
                      max_p_mw=_finite(row.get("max_p_mw"), want["p_max_mw"]),
                      min_p_mw=_finite(row.get("min_p_mw"), 0.0),
                      in_service=bool(row.get("in_service", True)))
        if want["control"] == VOLTAGE:
            if table == "sgen":
                net.sgen.drop(index=idx, inplace=True)
                new = pp.create_gen(net, vm_pu=want["vm_pu"] or 1.0, name=name, **common)
                _move_cost(net, "sgen", idx, "gen", new)
                table, idx = "gen", new
            net.gen.at[idx, "vm_pu"] = float(want["vm_pu"] or 1.0)
            net.gen.at[idx, "min_q_mvar"] = float(want["q_min_mvar"])
            net.gen.at[idx, "max_q_mvar"] = float(want["q_max_mvar"])
        else:
            if table == "gen":
                net.gen.drop(index=idx, inplace=True)
                new = pp.create_sgen(net, q_mvar=0.0, name=name, **common)
                _move_cost(net, "gen", idx, "sgen", new)
                idx = new
                if "substation_name" in net.sgen.columns:
                    net.sgen.at[idx, "substation_name"] = name
                table = "sgen"
            fixed = want["control"] == FIXED_Q
            for col, value in ((ORIG_VM, want["vm_pu"]), (ORIG_QMIN, want["q_min_mvar"]),
                               (ORIG_QMAX, want["q_max_mvar"]),
                               (PF_COL, None if fixed else float(want["pf"] or 1.0)),
                               (ABSORB_COL, None if fixed else bool(want["absorbing"])),
                               (QSET_COL, float(want["q_mvar"]) if fixed else None)):
                if col not in net.sgen.columns:
                    net.sgen[col] = None
                net.sgen[col] = net.sgen[col].astype(object)
                net.sgen.at[idx, col] = value
            set_reactive(net)
        if "controllable" not in net[table].columns:
            net[table]["controllable"] = None          # unset: dispatchable, as before
        net[table]["controllable"] = net[table]["controllable"].astype(object)
        net[table].at[idx, "controllable"] = bool(want["dispatchable"])
    # Ticks go on taking the generic path, whatever table the units are in.
    net["_gen_to_sgen_applied"] = True
    return net


def set_reactive(net) -> None:
    """Give every power-factor unit with a set power factor its reactive power
    for its present active power, and every fixed-Q unit its set reactive
    power (called after each tick's assignment)."""
    if not len(net.sgen):
        return
    # Measured reactive generation, when the data carry it, is not overwritten.
    measured = (net.sgen["_q_from_meas"].fillna(False).astype(bool) if "_q_from_meas" in net.sgen.columns
                else pd.Series(False, index=net.sgen.index))
    if QSET_COL in net.sgen.columns:
        qset = pd.to_numeric(net.sgen[QSET_COL], errors="coerce")
        fixed = qset.notna() & ~measured
        if fixed.any():
            net.sgen.loc[fixed, "q_mvar"] = qset[fixed].astype(float)
    if PF_COL not in net.sgen.columns:
        return
    pf = pd.to_numeric(net.sgen[PF_COL], errors="coerce")
    rows = pf.notna() & ~measured
    if not rows.any():
        return
    absorbing = net.sgen.get(ABSORB_COL, pd.Series(False, index=net.sgen.index)).fillna(False).astype(bool)
    tan = pf[rows].clip(lower=PF_MIN, upper=1.0).map(lambda v: math.tan(math.acos(v)))
    sign = absorbing[rows].map(lambda a: -1.0 if a else 1.0)
    net.sgen.loc[rows, "q_mvar"] = (net.sgen.loc[rows, "p_mw"].astype(float) * tan * sign).astype(float)
