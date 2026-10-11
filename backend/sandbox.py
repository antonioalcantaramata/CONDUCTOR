"""
Network proposals: the base network plus a list of edits (sandbox, phase 3).

A proposal is never a second stored network. It is the base the operator
loaded and an ordered list of small, typed edits; the backend materialises it
(deep copy of the base, edits applied in order) when a study asks for it. The
base is therefore never modified, the edit list is what the operator and the
session log see, and replaying it rebuilds exactly the network a study ran on.

Edits only append rows (`pp.create_*`) or change existing ones; they never
delete. Element indices are then the same in the base and in every proposal.

Every element an edit adds carries `proposed = True` and its own profile.
Measurement assignment leaves those rows alone (`load_gen_assignment.
_proposed_mask`); `apply_profiles` sets them for the tick being studied, and
`check_profiles` confirms afterwards that each one injects what its edit says
— a 50 MW data centre silently overwritten by the bus's measurements is the
failure this module exists to rule out.

Pure functions of a pandapower network: no app state, no HTTP. Errors are
`EditError` with a message meant for the operator.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from datetime import datetime

import numpy as np
import pandas as pd
import pandapower as pp

import load_gen_assignment as lg
from element_names import bus_display_name, clean_label, line_display_name, trafo_display_name


class EditError(ValueError):
    """An edit that cannot be applied; the message says why."""


LOAD_PROFILES = ("flat", "follow_bus", "shift")
GENERATOR_KINDS = ("dispatchable", "pv", "wind")
SWITCHABLE = ("line", "trafo", "load", "sgen", "gen")
OPS = ("add_load", "add_generator", "reinforce_line", "upgrade_trafo",
       "add_bus_with_connection", "set_in_service")

# Sanity bounds, not engineering limits: they catch a mistyped 5000 MW, not
# an ambitious proposal.
MAX_REINFORCE_FACTOR = 10.0
MAX_CONNECTION_KM = 200.0
MAX_SIZE_VS_NETWORK = 10.0   # an added MW figure may be up to 10× the base demand
MAX_EDITS = 50


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------

# Structural columns: what makes two loaded networks the same network.
# Measured values (p, q) change every tick and are left out.
_FINGERPRINT_COLUMNS = {
    "bus": ("name", "vn_kv", "in_service"),
    "line": ("from_bus", "to_bus", "length_km", "r_ohm_per_km", "x_ohm_per_km", "c_nf_per_km",
             "max_i_ka", "parallel", "in_service"),
    "trafo": ("hv_bus", "lv_bus", "sn_mva", "vn_hv_kv", "vn_lv_kv", "vk_percent", "vkr_percent",
              "tap_pos", "in_service"),
    "load": ("name", "bus"),
    "sgen": ("name", "bus"),
    "gen": ("name", "bus"),
    "ext_grid": ("bus",),
    "shunt": ("bus", "q_mvar"),
}


def network_fingerprint(net) -> str:
    """A short hash of a network's structure, stable across restarts — saved
    proposals are restored only onto the network they were built on."""
    parts = []
    for table, cols in _FINGERPRINT_COLUMNS.items():
        if table not in net or not len(net[table]):
            continue
        tbl = net[table]
        mask = ~lg._proposed_mask(tbl)
        present = [c for c in cols if c in tbl.columns]
        rows = tbl.loc[mask, present].copy()
        for c in present:
            if pd.api.types.is_float_dtype(rows[c]):
                rows[c] = rows[c].round(9)
        parts.append(f"{table}:{rows.to_json(orient='split', default_handler=str)}")
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]


def edits_hash(edits: list[dict]) -> str:
    """A short, stable hash of an edit list — echoed with every result."""
    blob = json.dumps(edits, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Element lookup
# ---------------------------------------------------------------------------

def _bus_label(net, idx: int) -> str:
    return bus_display_name(net, idx)


def resolve_bus(net, bus) -> int:
    """A bus by index, by its name, or by the `Bus_<i>` / `Bus <i>` label."""
    if isinstance(bus, (int, np.integer)) and not isinstance(bus, bool):
        if int(bus) in net.bus.index:
            return int(bus)
        raise EditError(f"There is no bus {bus}.")
    wanted = str(bus).strip()
    if not wanted:
        raise EditError("A bus is required.")
    key = wanted.casefold()
    hits = [int(i) for i in net.bus.index if _bus_label(net, int(i)).casefold() == key]
    if not hits:
        for prefix in ("bus_", "bus "):
            if key.startswith(prefix) and key[len(prefix):].strip().isdigit():
                idx = int(key[len(prefix):])
                if idx in net.bus.index:
                    hits = [idx]
        if not hits and key.isdigit() and int(key) in net.bus.index:
            hits = [int(key)]
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        raise EditError(f"{len(hits)} buses are named {wanted!r}; give the bus index.")
    raise EditError(f"There is no bus {wanted!r}.")


def _element_labels(net, etype: str, idx: int) -> list[str]:
    """Every label results use for this element; the display name is last."""
    tbl = net[etype]
    labels = [str(idx)]
    for col in ("name", "substation_name"):
        if col in tbl.columns and clean_label(tbl.at[idx, col]):
            labels.append(clean_label(tbl.at[idx, col]))
    if etype in ("line", "trafo"):
        shown = (line_display_name if etype == "line" else trafo_display_name)(net, idx)
        labels += [shown.split(" | ")[-1], f"{etype[0].upper()}{idx}", shown]
    elif etype in ("sgen", "gen"):
        labels.append(f"Gen_bus{int(tbl.at[idx, 'bus'])}")
    return labels


def resolve_element(net, etype: str, element) -> int:
    """An element of `net[etype]` by index or by any of the labels results show."""
    if etype not in SWITCHABLE:
        raise EditError(f"element_type must be one of {list(SWITCHABLE)}, not {etype!r}.")
    tbl = net[etype]
    if isinstance(element, (int, np.integer)) and not isinstance(element, bool):
        if int(element) in tbl.index:
            return int(element)
        raise EditError(f"There is no {etype} {element}.")
    key = str(element).strip().casefold()
    hits = sorted({int(i) for i in tbl.index
                   if any(lab.casefold() == key for lab in _element_labels(net, etype, int(i)))})
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        raise EditError(f"{len(hits)} {etype} elements match {element!r} "
                        f"(indices {hits[:10]}); give the index.")
    raise EditError(f"There is no {etype} {element!r}.")


def _taken_names(net) -> set[str]:
    names = set()
    for etype in ("load", "sgen", "gen"):
        tbl = net[etype]
        for col in ("name", "substation_name"):
            if col in tbl.columns:
                names |= {str(v).strip().casefold() for v in tbl[col] if isinstance(v, str) and v.strip()}
    return names


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _number(edit: dict, key: str, *, default=None, low=None, high=None,
            low_open: bool = False) -> float:
    raw = edit.get(key, default)
    if raw is None:
        raise EditError(f"{edit.get('op')}: {key} is required.")
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise EditError(f"{edit.get('op')}: {key} must be a number, not {raw!r}.") from None
    if not math.isfinite(value):
        raise EditError(f"{edit.get('op')}: {key} must be finite.")
    if low is not None and (value <= low if low_open else value < low):
        raise EditError(f"{edit.get('op')}: {key} must be {'above' if low_open else 'at least'} {low}.")
    if high is not None and value > high:
        raise EditError(f"{edit.get('op')}: {key} must be at most {high}.")
    return value


def _name(edit: dict, net) -> str:
    name = str(edit.get("name") or "").strip()
    if not name:
        raise EditError(f"{edit['op']}: a name is required (it labels the element in every result).")
    if name.casefold() in _taken_names(net):
        raise EditError(f"{edit['op']}: the name {name!r} is already used in this network.")
    return name


def _size_cap(net) -> float:
    demand = float(net.load.loc[net.load.in_service, "p_mw"].clip(lower=0).sum())
    return max(MAX_SIZE_VS_NETWORK * demand, 1.0)


def normalise_edit(net, raw: dict) -> dict:
    """Check one edit against `net` (the network it will be applied to) and
    return it in canonical form — every default filled, the bus or element
    stored as its index. Raises EditError with the reason."""
    if not isinstance(raw, dict):
        raise EditError("Each edit must be an object with an 'op'.")
    op = str(raw.get("op") or "").strip()
    if op not in OPS:
        raise EditError(f"Unknown edit {op!r}. Use one of {list(OPS)}.")
    edit = dict(raw, op=op)

    if op == "add_load":
        cap = _size_cap(net)
        out = {"op": op, "name": _name(edit, net), "bus": resolve_bus(net, edit.get("bus")),
               "p_mw": _number(edit, "p_mw", low=0, high=cap, low_open=True),
               "pf": _number(edit, "pf", default=0.95, low=0.5, high=1.0),
               "profile": str(edit.get("profile") or "flat").strip()}
        if out["profile"] not in LOAD_PROFILES:
            raise EditError(f"add_load: profile must be one of {list(LOAD_PROFILES)}.")
        if out["profile"] == "follow_bus":
            # The demand shape of the load's own bus, or of another one — a
            # new site has no measured demand of its own to follow.
            follow = edit.get("follow")
            out["follow"] = out["bus"] if follow in (None, "") else resolve_bus(net, follow)
        if out["profile"] == "shift":
            out["start_hour"] = int(_number(edit, "start_hour", low=0, high=23))
            out["end_hour"] = int(_number(edit, "end_hour", low=0, high=24))
            if out["start_hour"] == out["end_hour"] % 24:
                raise EditError("add_load: a shift needs different start_hour and end_hour.")
        return out

    if op == "add_generator":
        cap = _size_cap(net)
        out = {"op": op, "name": _name(edit, net), "bus": resolve_bus(net, edit.get("bus")),
               "p_max_mw": _number(edit, "p_max_mw", low=0, high=cap, low_open=True),
               "kind": str(edit.get("kind") or "").strip()}
        if out["kind"] not in GENERATOR_KINDS:
            raise EditError(f"add_generator: kind must be one of {list(GENERATOR_KINDS)}.")
        if out["kind"] == "dispatchable":
            out["p_mw"] = _number(edit, "p_mw", default=0.0, low=0, high=out["p_max_mw"])
        return out

    if op == "reinforce_line":
        idx = resolve_element(net, "line", edit.get("line"))
        return {"op": op, "line": idx,
                "factor": _number(edit, "factor", low=1.0, high=MAX_REINFORCE_FACTOR, low_open=True)}

    if op == "upgrade_trafo":
        idx = resolve_element(net, "trafo", edit.get("trafo"))
        now = float(net.trafo.at[idx, "sn_mva"])
        return {"op": op, "trafo": idx,
                "new_sn_mva": _number(edit, "new_sn_mva", low=now, high=MAX_REINFORCE_FACTOR * now,
                                      low_open=True)}

    if op == "add_bus_with_connection":
        name = str(edit.get("name") or "").strip()
        if not name:
            raise EditError("add_bus_with_connection: a name is required for the new bus.")
        if any(_bus_label(net, int(i)).casefold() == name.casefold() for i in net.bus.index):
            raise EditError(f"add_bus_with_connection: a bus is already named {name!r}.")
        at = resolve_bus(net, edit.get("connect_to"))
        vn = float(net.bus.at[at, "vn_kv"])
        if edit.get("vn_kv") is not None and abs(_number(edit, "vn_kv", low=0, low_open=True) - vn) > 1e-6:
            raise EditError(
                f"add_bus_with_connection: {_bus_label(net, at)} is at {vn:g} kV; a line joins buses "
                "of one voltage (a new transformer is not an edit yet).")
        out = {"op": op, "name": name, "connect_to": at, "vn_kv": vn,
               "length_km": _number(edit, "length_km", low=0, high=MAX_CONNECTION_KM, low_open=True)}
        like, std = edit.get("line_like"), str(edit.get("std_type") or "").strip()
        if (like is None or like == "") == (not std):
            raise EditError("add_bus_with_connection: give exactly one of line_like (an existing "
                            "line whose conductor to copy) or std_type (a pandapower line type).")
        if std:
            if not pp.std_type_exists(net, std, element="line"):
                raise EditError(f"add_bus_with_connection: {std!r} is not a known line type.")
            out["std_type"] = std
        else:
            out["line_like"] = resolve_element(net, "line", like)
        return out

    # set_in_service
    etype = str(edit.get("element_type") or "").strip()
    idx = resolve_element(net, etype, edit.get("element"))
    state = edit.get("in_service")
    if not isinstance(state, bool):
        raise EditError("set_in_service: in_service must be true or false.")
    return {"op": op, "element_type": etype, "element": idx, "in_service": state}


# ---------------------------------------------------------------------------
# Materialisation
# ---------------------------------------------------------------------------

def _apply(net, edit: dict) -> None:
    op = edit["op"]
    if op == "add_load":
        q = edit["p_mw"] * math.tan(math.acos(edit["pf"]))
        idx = pp.create_load(net, bus=edit["bus"], p_mw=edit["p_mw"], q_mvar=q, name=edit["name"])
        _tag(net.load, idx, edit)
        if "substation_name" in net.load.columns:
            net.load.at[idx, "substation_name"] = edit["name"]
    elif op == "add_generator":
        p0 = edit.get("p_mw", 0.0)
        idx = pp.create_sgen(net, bus=edit["bus"], p_mw=p0, q_mvar=0.0, name=edit["name"],
                             max_p_mw=edit["p_max_mw"], min_p_mw=0.0, type=edit["kind"])
        _tag(net.sgen, idx, edit)
        # The OPF and every result name an sgen by `substation_name` when the
        # column exists; an added unit gets its own name, not NaN.
        if "substation_name" in net.sgen.columns:
            net.sgen.at[idx, "substation_name"] = edit["name"]
    elif op == "reinforce_line":
        f, i = edit["factor"], edit["line"]
        # Equivalent to `factor` identical circuits in parallel (exact in the
        # pi model for any factor, not only integers).
        for col in ("r_ohm_per_km", "x_ohm_per_km"):
            net.line.at[i, col] = float(net.line.at[i, col]) / f
        for col in ("c_nf_per_km", "g_us_per_km", "max_i_ka"):
            if col in net.line.columns and pd.notna(net.line.at[i, col]):
                net.line.at[i, col] = float(net.line.at[i, col]) * f
    elif op == "upgrade_trafo":
        i = edit["trafo"]
        f = edit["new_sn_mva"] / float(net.trafo.at[i, "sn_mva"])
        # A larger unit of the same design: short-circuit voltage and copper
        # losses unchanged on its own rating, so its impedance in ohms falls
        # by the same factor — exactly `f` identical units in parallel.
        # No-load losses grow with size.
        net.trafo.at[i, "sn_mva"] = edit["new_sn_mva"]
        if "pfe_kw" in net.trafo.columns and pd.notna(net.trafo.at[i, "pfe_kw"]):
            net.trafo.at[i, "pfe_kw"] = float(net.trafo.at[i, "pfe_kw"]) * f
    elif op == "add_bus_with_connection":
        at = edit["connect_to"]
        bus = pp.create_bus(net, vn_kv=edit["vn_kv"], name=edit["name"])
        # The site takes its voltage limits and zone from the bus it joins.
        for col in ("min_vm_pu", "max_vm_pu", "zone", "type"):
            if col in net.bus.columns and pd.notna(net.bus.at[at, col]):
                net.bus.at[bus, col] = net.bus.at[at, col]
        line_name = f"{edit['name']} connection"
        if "std_type" in edit:
            line = pp.create_line(net, from_bus=at, to_bus=bus, length_km=edit["length_km"],
                                  std_type=edit["std_type"], name=line_name)
        else:
            src = net.line.loc[edit["line_like"]]
            extra = {c: float(src[c]) for c in ("g_us_per_km", "max_loading_percent")
                     if c in net.line.columns and pd.notna(src.get(c))}
            line = pp.create_line_from_parameters(
                net, from_bus=at, to_bus=bus, length_km=edit["length_km"],
                r_ohm_per_km=float(src["r_ohm_per_km"]), x_ohm_per_km=float(src["x_ohm_per_km"]),
                c_nf_per_km=float(src["c_nf_per_km"]), max_i_ka=float(src["max_i_ka"]),
                df=float(src["df"]) if pd.notna(src.get("df")) else 1.0, name=line_name, **extra)
        _mark(net.bus, bus)
        _mark(net.line, line)
    else:
        net[edit["element_type"]].at[edit["element"], "in_service"] = edit["in_service"]


def _mark(tbl: pd.DataFrame, idx: int) -> None:
    """Mark a bus or branch a proposal added (it has no profile)."""
    if "proposed" not in tbl.columns:
        tbl["proposed"] = False
    tbl["proposed"] = tbl["proposed"].astype(object)
    tbl.at[idx, "proposed"] = True


def _tag(tbl: pd.DataFrame, idx: int, edit: dict) -> None:
    """Mark an added row and keep, on the row, what its profile needs."""
    for col, default in (("proposed", False), ("profile", None), ("profile_spec", None)):
        if col not in tbl.columns:
            tbl[col] = default
        tbl[col] = tbl[col].astype(object)
    tbl.at[idx, "proposed"] = True
    tbl.at[idx, "profile"] = edit.get("profile") or edit.get("kind")
    tbl.at[idx, "profile_spec"] = json.dumps(edit, sort_keys=True)


def materialise(base_net, edits: list[dict]) -> tuple[object, list[dict]]:
    """A deep copy of `base_net` with `edits` applied in order.

    Each edit is validated against the network as the previous edits left it
    (an added load's name is then taken for the next one). Returns the network
    and the edits in canonical form.
    """
    if len(edits) > MAX_EDITS:
        raise EditError(f"A proposal holds at most {MAX_EDITS} edits.")
    net = copy.deepcopy(base_net)
    canonical = []
    for n, raw in enumerate(edits, start=1):
        try:
            edit = normalise_edit(net, raw)
        except EditError as exc:
            raise EditError(f"Edit {n}: {exc}") from None
        _apply(net, edit)
        canonical.append(edit)
    return net, canonical


def describe(net, edit: dict) -> str:
    """One line an operator can read."""
    op = edit["op"]
    if op == "add_load":
        prof = edit["profile"]
        if prof == "shift":
            prof = f"shift {edit['start_hour']:02d}:00–{edit['end_hour']:02d}:00"
        elif prof == "follow_bus" and edit.get("follow", edit["bus"]) != edit["bus"]:
            prof = f"follow_bus (demand shape of {_bus_label(net, edit['follow'])})"
        return (f"Add load {edit['name']!r}: {edit['p_mw']:g} MW at {_bus_label(net, edit['bus'])}, "
                f"pf {edit['pf']:g}, profile {prof}")
    if op == "add_generator":
        extra = f", set-point {edit['p_mw']:g} MW" if edit["kind"] == "dispatchable" else ""
        return (f"Add {edit['kind']} generator {edit['name']!r}: up to {edit['p_max_mw']:g} MW at "
                f"{_bus_label(net, edit['bus'])}{extra}")
    if op == "reinforce_line":
        return f"Reinforce line {line_display_name(net, edit['line'])} ×{edit['factor']:g}"
    if op == "upgrade_trafo":
        return (f"Upgrade transformer {trafo_display_name(net, edit['trafo'])} to "
                f"{edit['new_sn_mva']:g} MVA")
    if op == "add_bus_with_connection":
        conductor = (edit["std_type"] if "std_type" in edit
                     else f"as {line_display_name(net, edit['line_like'])}")
        return (f"New bus {edit['name']!r} ({edit['vn_kv']:g} kV), {edit['length_km']:g} km line "
                f"from {_bus_label(net, edit['connect_to'])} ({conductor})")
    # set_in_service
    etype = edit["element_type"]
    labels = _element_labels(net, etype, edit["element"])
    # Branches by their display name; units and loads by their name, else index.
    label = labels[-1] if etype in ("line", "trafo") else (labels[1] if len(labels) > 1 else labels[0])
    verb = ("Close" if edit["in_service"] else "Open") if etype in ("line", "trafo") else \
        ("Switch on" if edit["in_service"] else "Switch off")
    return f"{verb} {etype} {label}"


def summary(net, edits: list[dict]) -> dict:
    """Counts and totals of what a proposal adds or changes."""
    def total(op, key):
        return round(sum(e[key] for e in edits if e["op"] == op), 4)
    return {
        "n_edits": len(edits),
        "added_load_mw": total("add_load", "p_mw"),
        "added_generation_mw": total("add_generator", "p_max_mw"),
        "reinforced_lines": sum(e["op"] == "reinforce_line" for e in edits),
        "upgraded_trafos": sum(e["op"] == "upgrade_trafo" for e in edits),
        "new_buses": sum(e["op"] == "add_bus_with_connection" for e in edits),
        "switched_elements": sum(e["op"] == "set_in_service" for e in edits),
        "edits": [describe(net, e) for e in edits],
        "notes": notes(net, edits),
    }


def notes(net, edits: list[dict]) -> list[str]:
    """What the operator should know about how an edit was modelled."""
    out = []
    lengths = net.line.loc[~lg._proposed_mask(net.line), "length_km"] if len(net.line) else []
    placeholder = len(lengths) > 1 and bool((pd.Series(lengths).round(6) == 1.0).all())
    for e in edits:
        if e["op"] == "add_bus_with_connection" and "line_like" in e and placeholder:
            # MATPOWER/PGLib conversions give every line 1 km and its whole
            # impedance per km: the copied line's length is not geography.
            out.append(
                f"Every line of this network has length 1 km (a converted per-unit case), so "
                f"{e['length_km']:g} km of {line_display_name(net, e['line_like'])}'s conductor "
                f"means {e['length_km']:g}× that line's impedance, not a {e['length_km']:g} km route.")
    return out


# ---------------------------------------------------------------------------
# Profiles at a tick
# ---------------------------------------------------------------------------

def measurement_key_for_bus(net, bus: int, names: list[str]) -> str | None:
    """The measurement row a bus's demand comes from — the same matching
    `assign_load_values_from_measurements` applies to the loads on it."""
    label = _bus_label(net, bus)
    key = lg.match_substation(label, names) if names else None
    if key is None and label in names:
        key = label
    return key


def dataset_peaks(dataset: dict) -> dict:
    """The largest consumption and production of every measurement row, and
    of total production, over a whole dataset — what `follow_bus`, `pv` and
    `wind` scale against, so that an element's peak equals its size."""
    frames = [df[["substation_name", "consumption", "production"]] for df in dataset.values()
              if isinstance(df, pd.DataFrame) and len(df)]
    if not frames:
        return {"consumption": {}, "production": {}, "total_production": 0.0}
    rows = pd.concat(frames, ignore_index=True)
    rows["production"] = rows["production"].abs()
    by_key = rows.groupby("substation_name")[["consumption", "production"]].max()
    totals = [float(df["production"].abs().sum()) for df in dataset.values()
              if isinstance(df, pd.DataFrame) and len(df)]
    return {
        "consumption": {k: float(v) for k, v in by_key["consumption"].items()},
        "production": {k: float(v) for k, v in by_key["production"].items()},
        "total_production": max(totals) if totals else 0.0,
    }


def _hour(ts: str) -> int:
    try:
        return pd.Timestamp(ts).hour
    except (ValueError, TypeError):
        return datetime.fromisoformat(str(ts)).hour


def _value(meas: pd.DataFrame, key: str, column: str) -> float:
    rows = meas.loc[meas["substation_name"] == key, column]
    return float(rows.abs().sum()) if column == "production" else float(rows.sum())


def profile_factor(net, spec: dict, ts: str, meas: pd.DataFrame, peaks: dict) -> tuple[float, str]:
    """(share of the element's size it injects at `ts`, where that comes from)."""
    names = meas["substation_name"].dropna().astype(str).tolist()
    if spec["op"] == "add_load":
        prof = spec["profile"]
        if prof == "flat":
            return 1.0, "flat"
        if prof == "shift":
            h, a, b = _hour(ts), spec["start_hour"], spec["end_hour"] % 24
            on = (a <= h < b) if a < b else (h >= a or h < b)
            return (1.0 if on else 0.0), f"shift {spec['start_hour']:02d}–{spec['end_hour']:02d}"
        followed = spec.get("follow", spec["bus"])
        key = measurement_key_for_bus(net, followed, names)
        peak = peaks["consumption"].get(key, 0.0) if key else 0.0
        if peak <= 0:
            raise EditError(f"{spec['name']!r} follows {_bus_label(net, followed)}, "
                            "which has no measured demand in this dataset"
                            + ("; give `follow` a bus that has." if followed == spec["bus"] else "."))
        return max(_value(meas, key, "consumption"), 0.0) / peak, f"demand of {key}"
    if spec["kind"] == "dispatchable":
        return spec["p_mw"] / spec["p_max_mw"], "set-point"
    key = measurement_key_for_bus(net, spec["bus"], names)
    peak = peaks["production"].get(key, 0.0) if key else 0.0
    if peak > 0:
        return _value(meas, key, "production") / peak, f"measured production of {key}"
    total = peaks.get("total_production", 0.0)
    if total <= 0:
        raise EditError(f"{spec['name']!r} ({spec['kind']}) follows measured production, "
                        "and this dataset has none.")
    return float(meas["production"].abs().sum()) / total, "total measured production"


def _proposed_rows(tbl: pd.DataFrame):
    mask = lg._proposed_mask(tbl)
    for idx in tbl.index[mask]:
        spec = tbl.at[idx, "profile_spec"] if "profile_spec" in tbl.columns else None
        if isinstance(spec, str):
            yield idx, json.loads(spec)


def apply_profiles(net, ts: str, meas: pd.DataFrame, peaks: dict) -> list[dict]:
    """Set every added load and generator to its profile at `ts`.

    Loads: p = size × factor, q from the edit's power factor. Variable
    generators (pv, wind) get their available power as both set-point and
    `max_p_mw`, so an optimiser may curtail them but not exceed the resource;
    a dispatchable unit keeps its capacity and its set-point.
    Returns, per element, what it injects and why.
    """
    report = []
    for idx, spec in _proposed_rows(net.load):
        f, source = profile_factor(net, spec, ts, meas, peaks)
        p = spec["p_mw"] * f
        net.load.at[idx, "p_mw"] = p
        net.load.at[idx, "q_mvar"] = p * math.tan(math.acos(spec["pf"]))
        report.append({"element": spec["name"], "type": "load", "p_mw": round(p, 4),
                       "size_mw": spec["p_mw"], "profile": spec["profile"], "source": source})
    for idx, spec in _proposed_rows(net.sgen):
        f, source = profile_factor(net, spec, ts, meas, peaks)
        p = spec["p_max_mw"] * min(max(f, 0.0), 1.0)
        net.sgen.at[idx, "p_mw"] = p
        net.sgen.at[idx, "q_mvar"] = 0.0
        net.sgen.at[idx, "max_p_mw"] = spec["p_max_mw"] if spec["kind"] == "dispatchable" else p
        report.append({"element": spec["name"], "type": "generator", "p_mw": round(p, 4),
                       "size_mw": spec["p_max_mw"], "profile": spec["kind"], "source": source})
    return report


def check_profiles(net, report: list[dict], load_scaling: float = 1.0) -> list[str]:
    """Each added element still injects what its profile gave it — after the
    measured values were assigned and any load stress applied. The failure
    this catches is silent by nature (sandbox hazard A.3 (1)); returns the
    discrepancies, empty when there are none."""
    wrong = []
    by_name = {r["element"]: r for r in report}
    for tbl, kind, scale in ((net.load, "load", load_scaling), (net.sgen, "generator", 1.0)):
        for idx, spec in _proposed_rows(tbl):
            want = by_name.get(spec["name"], {}).get("p_mw")
            got = float(tbl.at[idx, "p_mw"])
            if want is None or abs(got - want * scale) > 1e-3 * max(1.0, abs(want)):
                wrong.append(f"{kind} {spec['name']!r}: {got:.4f} MW instead of "
                             f"{(want or 0) * scale:.4f} MW")
    return wrong
