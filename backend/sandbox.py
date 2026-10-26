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
import resolve as rv
from nominal import NOMINAL
from element_names import (bus_display_name, clean_label, line_display_name, measurement_name,
                           trafo_display_name)


class EditError(ValueError):
    """An edit that cannot be applied; the message says why."""


class EditQuestion(EditError):
    """An edit that needs the operator's decision: something essential is
    missing or contradicts the network, and no default would be defensible.
    Carries the question and the options to put to them."""

    def __init__(self, question: str, options: list[str], edit: int | None = None):
        super().__init__(question)
        self.question, self.options, self.edit = question, list(options), edit


LOAD_PROFILES = ("flat", "follow_bus", "shift")
GENERATOR_KINDS = ("dispatchable", "pv", "wind")
GENERATOR_CONTROLS = ("voltage", "power_factor", "fixed_q")
SHUNT_KINDS = ("capacitor", "reactor")
# Assumed reactive limits of a new voltage-controlling unit when none are
# given: a synchronous machine's typical rated power factor (reported).
ASSUMED_RATED_PF = 0.85
SWITCHABLE = ("line", "trafo", "load", "sgen", "gen", "shunt", "storage", "trafo3w")
OPS = ("add_load", "add_generator", "add_shunt", "reinforce_line", "upgrade_trafo",
       "add_bus_with_connection", "add_bus_with_transformer", "set_in_service", "set_parameter",
       "add_line", "add_trafo", "move_element", "scale")
MOVABLE = ("load", "sgen", "gen", "shunt", "storage", "ext_grid", "line")
SCALABLE = ("load", "sgen", "gen")
MAX_SCALE = 10.0

# set_parameter: any setting of any existing element — the external grid's
# voltage, a tap, a rating, a set-point, a load's demand. Connection fields are
# not settings (rewiring is switching one element out and adding another).
EDITABLE_TABLES = ("ext_grid", "bus", "line", "trafo", "trafo3w", "load", "sgen", "gen", "shunt", "storage",
                   "impedance")
_STRUCTURAL = {"bus", "from_bus", "to_bus", "hv_bus", "lv_bus", "mv_bus", "element", "et", "proposed",
               "profile", "profile_spec", "override", "substation_name", "orig_vm_pu", "orig_min_q_mvar",
               "orig_max_q_mvar", "q_pf", "q_absorbing", "q_set_mvar", "_q_from_meas"}
# OPF limits a table may not carry yet but may be given.
_OPTIONAL_COLUMNS = {
    "ext_grid": ("max_p_mw", "min_p_mw", "max_q_mvar", "min_q_mvar"),
    "gen": ("max_p_mw", "min_p_mw", "max_q_mvar", "min_q_mvar"),
    "sgen": ("max_p_mw", "min_p_mw", "max_q_mvar", "min_q_mvar", "controllable"),
    "load": ("max_p_mw", "min_p_mw", "max_q_mvar", "min_q_mvar", "controllable"),
    "line": ("max_loading_percent",),
    "trafo": ("max_loading_percent",),
    "bus": ("max_vm_pu", "min_vm_pu"),
}
_BOOL_FIELDS = {"in_service", "controllable", "slack", "oltc"}
# Values the measurements set at every tick: kept as overrides and applied
# again after each tick's assignment (apply_overrides), or they would be lost.
_MEASURED = {"load": ("p_mw", "q_mvar"), "sgen": ("p_mw", "q_mvar"), "gen": ("p_mw",)}
_BOUNDS = {"vm_pu": (0.8, 1.2), "va_degree": (-180.0, 180.0), "max_vm_pu": (0.8, 1.3), "min_vm_pu": (0.6, 1.2),
           "vk_percent": (0.1, 50.0), "vkr_percent": (0.0, 50.0), "scaling": (0.0, 10.0),
           "max_loading_percent": (1.0, 1000.0), "parallel": (1, 10)}
_POSITIVE = {"max_i_ka", "sn_mva", "length_km", "vn_kv", "vn_hv_kv", "vn_lv_kv", "vn_mv_kv", "max_e_mwh"}
_NON_NEGATIVE = {"r_ohm_per_km", "x_ohm_per_km", "c_nf_per_km", "g_us_per_km", "rft_pu", "xft_pu"}
TRAFO_SIDES = ("hv", "lv")

# Sanity bounds, not engineering limits: they catch a mistyped 5000 MW, not
# an ambitious proposal.
MAX_REINFORCE_FACTOR = 10.0
MAX_CONNECTION_KM = 200.0
MAX_SIZE_VS_NETWORK = 10.0   # an added MW figure may be up to 10× the base demand
MAX_EDITS = 50
MAX_KV = 1000.0
MAX_PARALLEL_UNITS = 4
RATED_KV_TOLERANCE = 0.2     # a rated voltage may sit up to 20 % off its bus's nominal voltage
MAX_VK_PERCENT = 50.0        # real units: 4–25 %
MAX_TAP_STEPS = 40


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
        if "name" in present and "loaded_name" in tbl.columns:
            # A renamed element is the same network: its name as loaded counts.
            kept = tbl.loc[mask, "loaded_name"]
            renamed = kept.apply(lambda v: isinstance(v, str))
            rows["name"] = rows["name"].astype(object)
            rows.loc[renamed, "name"] = kept[renamed].map(lambda v: v or None)
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
    """A bus by index or by any name it answers to (`resolve.find`)."""
    try:
        return rv.find(net, bus, ("bus",)).idx
    except rv.ResolveError as exc:
        raise EditError(str(exc)) from None


_foreign_tail = rv.foreign_tail


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
    if etype in ("ext_grid", "shunt", "storage") and "bus" in tbl.columns:
        bus = int(tbl.at[idx, "bus"])
        labels += [f"Bus_{bus}", f"Bus {bus}", _bus_label(net, bus)]
    if etype == "ext_grid":
        labels += ["external grid", "external_grid", "ext_grid", "slack", "grid", "external connection"]
    if etype == "load" and "bus" in tbl.columns:
        # A load's substation_name is the measurement row it reads (the bus's
        # name as loaded), not a name: shown last, it kept a renamed bus's old
        # name in every description. Its own name, else its bus as named now.
        own = clean_label(tbl.at[idx, "name"]) if "name" in tbl.columns else ""
        bus = int(tbl.at[idx, "bus"])
        renamed = "loaded_name" in net.bus.columns and isinstance(net.bus.at[bus, "loaded_name"], str)
        if own:
            labels.append(own)
        elif renamed:
            labels.append(_bus_label(net, bus))
    return labels


def resolve_element(net, etype: str, element, allowed: tuple = SWITCHABLE) -> int:
    """An element of `net[etype]` by index or by any of the labels results show."""
    if etype not in allowed:
        raise EditError(f"element_type must be one of {list(allowed)}, not {etype!r}.")
    if etype == "bus":
        return resolve_bus(net, element)
    if etype in ("line", "trafo"):
        # By its label, tag, ends or index (`resolve.find`), like every other study.
        try:
            return rv.find(net, element, (etype,)).idx
        except rv.ResolveError as exc:
            raise EditError(str(exc)) from None
    tbl = net[etype]
    if isinstance(element, str) and element.strip().isdigit() and int(element) in tbl.index:
        return int(element)
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


def _new_bus_name(net, edit: dict) -> str:
    name = str(edit.get("name") or "").strip()
    if not name:
        raise EditError(f"{edit['op']}: a name is required for the new bus.")
    _bus_name_free(net, name, edit["op"])
    return name


def _bus_name_free(net, name: str, op: str) -> None:
    if any(_bus_label(net, int(i)).casefold() == name.casefold() for i in net.bus.index):
        raise EditError(f"{op}: a bus is already named {name!r}.")


def _given(edit: dict, key: str) -> bool:
    return edit.get(key) not in (None, "")


def _line_choice(net, edit: dict, like_key: str, std_key: str) -> dict:
    """The conductor of a new line: copied from an existing line, or a
    pandapower standard type — exactly one."""
    op = edit["op"]
    std = str(edit.get(std_key) or "").strip()
    if _given(edit, like_key) == bool(std):
        raise EditError(f"{op}: give exactly one of {like_key} (an existing line whose conductor "
                        f"to copy) or {std_key} (a pandapower line type).")
    if std:
        if not pp.std_type_exists(net, std, element="line"):
            raise EditError(f"{op}: {std!r} is not a known line type.")
        return {std_key: std}
    return {like_key: resolve_element(net, "line", edit[like_key])}


def _voltage_band(edit: dict) -> dict:
    """Optional voltage limits for a new bus (otherwise those of the bus it joins)."""
    out = {k: _number(edit, k, low=0.5, high=1.5) for k in ("min_vm_pu", "max_vm_pu") if _given(edit, k)}
    if "min_vm_pu" in out and "max_vm_pu" in out and out["min_vm_pu"] >= out["max_vm_pu"]:
        raise EditError(f"{edit['op']}: min_vm_pu must be below max_vm_pu.")
    return out


def _integer(edit: dict, key: str, **bounds) -> int:
    value = _number(edit, key, **bounds)
    if value != int(value):
        raise EditError(f"{edit.get('op')}: {key} must be a whole number, not {edit[key]!r}.")
    return int(value)


def _finite(value) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def substation_bus_name(name: str, kv: float) -> str:
    """The bus at the far end of the line, where the new transformer stands."""
    return f"{name} {kv:g} kV"


_DESIGN = ("vk_percent", "vkr_percent", "pfe_kw", "i0_percent", "shift_degree")
_TAP_RANGE = ("tap_side", "tap_step_percent", "tap_min", "tap_max")
_RATED = ("vn_hv_kv", "vn_lv_kv")


def _normalise_trafo_unit(net, edit: dict, out: dict, std: str, like, pair, hv_kv: float, lv_kv: float) -> None:
    """The unit of a new transformer — a type, a copy or explicit values —
    checked and written into `out` (shared by add_bus_with_transformer and
    add_trafo). `pair` is the (hv, lv) rating of the type or copied unit."""
    op = edit["op"]
    out["parallel"] = _integer(edit, "parallel", default=1, low=1, high=MAX_PARALLEL_UNITS)

    # The unit: a type, a copy, or explicit values.
    if std:
        fixed = [k for k in ("sn_mva", *_DESIGN, *_TAP_RANGE, *_RATED) if _given(edit, k)]
        if fixed:
            raise EditError(f"{op}: a standard type fixes {', '.join(fixed)}; use trafo_like or "
                            "explicit values (sn_mva, vk_percent, vkr_percent, ...) to change them.")
        for k, rated, bus_kv in (("hv", pair[0], hv_kv), ("lv", pair[1], lv_kv)):
            if abs(rated - bus_kv) > RATED_KV_TOLERANCE * bus_kv:
                raise EditError(f"{op}: type {std!r} is a {pair[0]:g}/{pair[1]:g} kV unit; it cannot "
                                f"join {hv_kv:g} kV and {lv_kv:g} kV.")
        out["trafo_std_type"] = std
    else:
        if like is not None:
            out["trafo_like"] = like
        elif not all(_given(edit, k) for k in ("sn_mva", "vk_percent", "vkr_percent")):
            raise EditError(f"{op}: give trafo_like (an existing transformer to copy), trafo_std_type "
                            "(a pandapower type) or the unit's sn_mva, vk_percent and vkr_percent.")
        cap = _size_cap(net)
        if like is not None:
            cap = max(cap, MAX_REINFORCE_FACTOR * float(net.trafo.at[like, "sn_mva"]))
        if _given(edit, "sn_mva"):
            out["sn_mva"] = _number(edit, "sn_mva", low=0, high=cap, low_open=True)
        for k, low, high, low_open in (("vk_percent", 0, MAX_VK_PERCENT, True),
                                       ("vkr_percent", 0, MAX_VK_PERCENT, False),
                                       ("pfe_kw", 0, None, False),
                                       ("i0_percent", 0, 10, False),
                                       ("shift_degree", -360, 360, False)):
            if _given(edit, k):
                out[k] = _number(edit, k, low=low, high=high, low_open=low_open)
        for k, bus_kv in (("vn_hv_kv", hv_kv), ("vn_lv_kv", lv_kv)):
            # In a per-unit case the rating is the buses' own: a kV the operator
            # gives is the real one (observed live: 20 kV as the low side under a
            # 1 kV placeholder bus, refused as out of range).
            if _given(edit, k) and not placeholder_voltages(net):
                out[k] = _number(edit, k, low=(1 - RATED_KV_TOLERANCE) * bus_kv,
                                 high=(1 + RATED_KV_TOLERANCE) * bus_kv)
        if _given(edit, "tap_side"):
            out["tap_side"] = str(edit["tap_side"]).strip().lower()
            if out["tap_side"] not in TRAFO_SIDES:
                raise EditError(f"{op}: tap_side must be 'hv' or 'lv'.")
        if _given(edit, "tap_step_percent"):
            out["tap_step_percent"] = _number(edit, "tap_step_percent", low=0, high=10, low_open=True)
        if _given(edit, "tap_min"):
            out["tap_min"] = _integer(edit, "tap_min", low=-MAX_TAP_STEPS, high=0)
        if _given(edit, "tap_max"):
            out["tap_max"] = _integer(edit, "tap_max", low=0, high=MAX_TAP_STEPS)
        if like is None:
            out.setdefault("pfe_kw", 0.0)
            out.setdefault("shift_degree", 0.0)
            # With no magnetising current given, the no-load losses alone flow in it.
            out.setdefault("i0_percent", round(out["pfe_kw"] / (10.0 * out["sn_mva"]), 6))
            if any(k in out for k in _TAP_RANGE) and not all(
                    k in out for k in ("tap_step_percent", "tap_min", "tap_max")):
                raise EditError(f"{op}: a tap changer needs tap_step_percent, tap_min and tap_max "
                                "(tap_side defaults to 'hv').")
    if _given(edit, "max_loading_percent"):
        out["max_loading_percent"] = _number(edit, "max_loading_percent", low=0, high=200, low_open=True)

    # The resulting unit must be physical, and the tap position inside its range.
    design = trafo_design(net, out)
    if design["vkr_percent"] >= design["vk_percent"]:
        raise EditError(f"{op}: vkr_percent ({design['vkr_percent']:g}) must be below vk_percent "
                        f"({design['vk_percent']:g}); the copper losses are part of the short-circuit voltage.")
    if design["pfe_kw"] > 50.0 * design["sn_mva"]:
        raise EditError(f"{op}: pfe_kw {design['pfe_kw']:g} is over 5 % of the rating; no-load losses "
                        "are typically 0.05–0.3 %.")
    if _given(edit, "tap_pos"):
        if design["tap_step_percent"] is None:
            raise EditError(f"{op}: tap_pos needs a tap changer, and this unit has none.")
        pos = _integer(edit, "tap_pos", low=-MAX_TAP_STEPS, high=MAX_TAP_STEPS)
        lo, hi = design["tap_min"], design["tap_max"]
        if (lo is not None and pos < lo) or (hi is not None and pos > hi):
            raise EditError(f"{op}: tap_pos {pos} is outside the tap range {lo:g}…{hi:g}.")
        out["tap_pos"] = pos


def _normalise_transformer_site(net, edit: dict) -> dict:
    """A new bus at another voltage behind a new transformer, optionally at
    the end of a new line (a remote substation). See `trafo_design` for how
    the unit's data are built from the three ways of giving them."""
    op = edit["op"]
    name = _new_bus_name(net, edit)
    at = resolve_bus(net, edit.get("connect_to"))
    v_at = float(net.bus.at[at, "vn_kv"])
    out = {"op": op, "name": name, "connect_to": at}

    std = str(edit.get("trafo_std_type") or "").strip()
    like = edit.get("trafo_like") if _given(edit, "trafo_like") else None
    if std and like is not None:
        raise EditError(f"{op}: give trafo_like or trafo_std_type, not both.")

    # The new bus's voltage: given, or the other side of the copied unit or type.
    vn = _number(edit, "vn_kv", low=0, high=MAX_KV, low_open=True) if _given(edit, "vn_kv") else None
    if std:
        if not pp.std_type_exists(net, std, element="trafo"):
            raise EditError(f"{op}: {std!r} is not a known transformer type.")
        data = pp.load_std_type(net, std, "trafo")
        pair, source = (float(data["vn_hv_kv"]), float(data["vn_lv_kv"])), f"type {std!r}"
    elif like is not None:
        like = resolve_element(net, "trafo", like)
        row = net.trafo.loc[like]
        pair = (float(net.bus.at[int(row["hv_bus"]), "vn_kv"]), float(net.bus.at[int(row["lv_bus"]), "vn_kv"]))
        source = trafo_display_name(net, like)
    else:
        pair, source = None, None
    if vn is None:
        if pair is None:
            raise EditError(f"{op}: vn_kv (the new bus's voltage) is required.")
        near = [k for k in (0, 1) if abs(pair[k] - v_at) <= RATED_KV_TOLERANCE * pair[k]]
        if abs(pair[0] - pair[1]) <= 1e-6 * pair[0]:
            raise EditError(f"{op}: {source} joins two {pair[0]:g} kV buses, so the new bus's voltage "
                            "cannot be taken from it; give vn_kv.")
        if not near:
            raise EditError(f"{op}: {_bus_label(net, at)} is at {v_at:g} kV, which is neither side of "
                            f"{source} ({pair[0]:g}/{pair[1]:g} kV); give vn_kv.")
        side = min(near, key=lambda k: abs(pair[k] - v_at))
        vn = pair[1 - side]
    if abs(vn - v_at) <= 1e-6 * v_at:
        raise EditError(f"{op}: the new bus would be at {v_at:g} kV like {_bus_label(net, at)}; "
                        "use add_bus_with_connection (a line) for a bus at the same voltage.")
    out["vn_kv"] = vn
    hv_kv, lv_kv = max(vn, v_at), min(vn, v_at)

    # Optional line first: from connect_to to the substation where the unit stands.
    if any(_given(edit, k) for k in ("length_km", "line_like", "line_std_type")):
        if not _given(edit, "length_km"):
            raise EditError(f"{op}: a line to the substation needs length_km. For the transformer at "
                            f"{_bus_label(net, at)} itself, leave length_km, line_like and line_std_type out.")
        out["length_km"] = _number(edit, "length_km", low=0, high=MAX_CONNECTION_KM, low_open=True)
        out.update(_line_choice(net, edit, "line_like", "line_std_type"))
        sub = substation_bus_name(name, v_at)
        if sub.casefold() == name.casefold():
            raise EditError(f"{op}: the new bus cannot be named like its substation bus {sub!r}.")
        _bus_name_free(net, sub, op)

    _normalise_trafo_unit(net, edit, out, std, like, pair, hv_kv, lv_kv)
    out.update(_voltage_band(edit))
    return out


def _site_kv(net, e: dict) -> tuple[float, float]:
    """(hv, lv) kV of the buses a new transformer joins."""
    if "hv_bus" in e:
        return float(net.bus.at[e["hv_bus"], "vn_kv"]), float(net.bus.at[e["lv_bus"], "vn_kv"])
    at = float(net.bus.at[e["connect_to"], "vn_kv"])
    return max(e["vn_kv"], at), min(e["vn_kv"], at)


def trafo_design(net, e: dict) -> dict:
    """The data of the unit an `add_bus_with_transformer` edit builds, for one
    unit (`parallel` multiplies it).

    - `trafo_std_type`: the pandapower catalogue entry, as it is.
    - `trafo_like`: an existing unit's design copied in per unit — short-circuit
      voltage, copper and no-load losses, magnetising current, vector-group
      shift and tap changer. With a new `sn_mva` it is a unit of the same
      design and another size: percentages kept, no-load losses scaled with
      size. Any design or tap value given overrides the copy.
    - explicit values: as given.
    Rated voltages are the buses' nominal voltages unless given (a type keeps
    its own). The tap starts at neutral unless `tap_pos` says otherwise.
    """
    hv_kv, lv_kv = _site_kv(net, e)
    d = {"sn_mva": None, "vn_hv_kv": hv_kv, "vn_lv_kv": lv_kv, "vk_percent": None,
         "vkr_percent": None, "pfe_kw": 0.0, "i0_percent": 0.0, "shift_degree": 0.0,
         "tap_side": None, "tap_neutral": None, "tap_min": None, "tap_max": None,
         "tap_step_percent": None, "tap_step_degree": None, "tap_changer_type": None,
         "max_loading_percent": None}
    if "trafo_std_type" in e:
        data = pp.load_std_type(net, e["trafo_std_type"], "trafo")
        for k in d:
            if k in data:
                d[k] = data[k] if isinstance(data[k], str) else _finite(data[k])
    elif "trafo_like" in e:
        src = net.trafo.loc[e["trafo_like"]]
        for k in d:
            if k in _RATED or k not in src.index:
                continue
            d[k] = src[k] if isinstance(src[k], str) else _finite(src[k])
        if "sn_mva" in e:
            f = e["sn_mva"] / float(src["sn_mva"])
            d["pfe_kw"] = (d["pfe_kw"] or 0.0) * f
    for k in ("sn_mva", *_DESIGN, *_TAP_RANGE, *_RATED, "max_loading_percent"):
        if k in e:
            d[k] = e[k]
    for k in ("pfe_kw", "i0_percent", "shift_degree"):
        d[k] = d[k] or 0.0
    if d["tap_step_percent"] is not None:
        d["tap_side"] = d["tap_side"] if d["tap_side"] in TRAFO_SIDES else "hv"
        d["tap_neutral"] = d["tap_neutral"] if d["tap_neutral"] is not None else 0.0
        if not isinstance(d["tap_changer_type"], str):
            d["tap_changer_type"] = "Ratio"
        d["tap_pos"] = e.get("tap_pos", d["tap_neutral"])
    else:
        for k in ("tap_side", "tap_neutral", "tap_min", "tap_max", "tap_step_degree", "tap_changer_type"):
            d[k] = None
        d["tap_pos"] = None
    return d


def _measured_substation(net) -> bool:
    """A network whose units are rebuilt from the measurements at every tick
    (loaded with no `gen` table): a voltage-controlled unit would change how
    its ticks are assigned."""
    own = net.gen[~lg._proposed_mask(net.gen)] if len(net.gen) else net.gen
    return not len(own) and not net.get("_gen_to_sgen_applied", False)


def _bus_setpoint(net, bus: int) -> tuple[float, str] | None:
    """(the voltage a bus is already held at, by what) — a unit or the grid."""
    for idx in net.ext_grid.index[net.ext_grid.bus == bus]:
        return float(net.ext_grid.at[idx, "vm_pu"]), "the external grid"
    for idx in net.gen.index[net.gen.bus == bus]:
        raw = net.gen.at[idx, "name"]
        name = raw if isinstance(raw, str) and raw.strip() else f"Gen_bus{bus}"
        return float(net.gen.at[idx, "vm_pu"]), name
    return None


def _normalise_generator(net, edit: dict) -> dict:
    """A new unit: what it is, how it behaves in a power flow (control), and
    whether the optimiser may move it. Essentials the operator did not give
    are asked (EditQuestion); standard defaults are filled and reported by
    the endpoint as assumed."""
    op = edit["op"]
    cap = _size_cap(net)
    kind = str(edit.get("kind") or "").strip().lower()
    if kind not in GENERATOR_KINDS:
        raise EditQuestion("What kind of unit is it?", [
            "dispatchable — a conventional unit whose output is set (gas, diesel, hydro, a battery…)",
            "pv — solar, following the measured production shape (curtailable)",
            "wind — wind, following the measured production shape (curtailable)"])
    out = {"op": op, "name": _name(edit, net), "bus": resolve_bus(net, edit.get("bus")),
           "p_max_mw": _number(edit, "p_max_mw", low=0, high=cap, low_open=True), "kind": kind}
    if kind == "dispatchable":
        out["p_min_mw"] = _number(edit, "p_min_mw", default=0.0, low=0, high=out["p_max_mw"])
        # At its rating unless told: "a 30 MW generator" is studied producing 30 MW
        # (observed live: studied at 0 MW, a "secure" that said nothing about it).
        out["p_mw"] = _number(edit, "p_mw", default=out["p_max_mw"], low=out["p_min_mw"], high=out["p_max_mw"])

    control = str(edit.get("control") or "").strip().lower()
    if control and control not in GENERATOR_CONTROLS:
        raise EditError(f"{op}: control must be 'voltage', 'power_factor' or 'fixed_q'.")
    if not control:
        if kind != "dispatchable":
            control = "power_factor"            # inverter-based: the usual grid-code behaviour
        else:
            raise EditQuestion(
                f"How should {out['name']!r} behave in a power flow?",
                ["voltage — a synchronous (classical) generator: holds its bus voltage at a set-point, "
                 "reactive power within its limits",
                 "power_factor — an inverter-based unit: fixed power factor (1.0 = no reactive power)"])
    out["control"] = control
    if control == "voltage":
        if _measured_substation(net):
            raise EditQuestion(
                "This network's units are rebuilt from the measurements at every tick, so a new unit "
                "cannot control voltage here. How should it run instead?",
                ["power_factor at 1.0 — no reactive power", "power_factor at another value (give pf)"])
        held = _bus_setpoint(net, out["bus"])
        if edit.get("vm_pu") not in (None, ""):
            vm = _number(edit, "vm_pu", low=0.8, high=1.2)
            if held and abs(held[0] - vm) > 1e-6:
                raise EditQuestion(
                    f"{_bus_label(net, out['bus'])} is already held at {held[0]:g} p.u. by {held[1]}, and two "
                    "units on one bus must hold the same voltage. Which do you want?",
                    [f"hold {held[0]:g} p.u., like {held[1]}",
                     f"run {out['name']!r} at a fixed power factor instead",
                     "put it on another bus"])
        else:
            vm = held[0] if held else 1.0
        out["vm_pu"] = vm
        default_q = round(out["p_max_mw"] * math.tan(math.acos(ASSUMED_RATED_PF)), 4)
        out["q_min_mvar"] = _number(edit, "q_min_mvar", default=-default_q)
        out["q_max_mvar"] = _number(edit, "q_max_mvar", default=default_q)
        if out["q_min_mvar"] > out["q_max_mvar"]:
            raise EditError(f"{op}: q_min_mvar is above q_max_mvar.")
    elif control == "fixed_q":
        if edit.get("q_mvar") in (None, ""):
            raise EditQuestion(
                f"What reactive power should {out['name']!r} hold?",
                ["0 Mvar — none", "a value in Mvar: positive injects (raises nearby voltage), negative "
                 "absorbs (lowers it)"])
        out["q_mvar"] = _number(edit, "q_mvar")
    else:
        out["pf"] = _number(edit, "pf", default=1.0, low=0.5, high=1.0)
        if edit.get("absorbing") is not None and not isinstance(edit["absorbing"], bool):
            raise EditError(f"{op}: absorbing must be true or false.")
        out["absorbing"] = bool(edit.get("absorbing", False))
    if edit.get("dispatchable") is not None and not isinstance(edit["dispatchable"], bool):
        raise EditError(f"{op}: dispatchable must be true or false.")
    out["dispatchable"] = bool(edit.get("dispatchable", True))
    return out


def upgrade_stored(edits: list[dict]) -> list[dict]:
    """Edits stored before a field existed, completed with what they meant
    then — a generator was a fixed-power-factor (1.0) unit the optimiser could
    move — so a saved proposal replays as it was built instead of asking."""
    out = []
    for e in edits:
        if isinstance(e, dict) and e.get("op") == "add_generator" and "control" not in e:
            e = {**e, "control": "power_factor", "pf": 1.0, "absorbing": False, "dispatchable": True}
        out.append(e)
    return out


# Optional fields for which 0 is never a value (each is refused at 0): a model
# that fills every field of the edit form sends 0 for those it means to leave
# out. Observed live: vn_kv 0, min_vm_pu 0 and length_km 0 refused an edit
# that needed none of them.
_ZERO_IS_UNSET = {"length_km", "vn_kv", "min_vm_pu", "max_vm_pu", "sn_mva", "vk_percent", "parallel",
                  "max_i_ka", "vn_hv_kv", "vn_lv_kv", "max_loading_percent", "new_sn_mva", "tap_step_percent"}


def _without_blanks(raw: dict) -> dict:
    """The edit without the fields it leaves blank: empty, or 0 where 0 is
    never a value — so a field is "given" only when it says something."""
    out = {k: v for k, v in raw.items()
           if v not in (None, "") and not (k in _ZERO_IS_UNSET and not isinstance(v, bool) and v == 0)}
    if raw.get("op") == "add_generator" and raw.get("p_mw") == 0 and not isinstance(raw.get("p_mw"), bool):
        out.pop("p_mw", None)              # an output left blank: the unit runs at its rating
    if raw.get("op") == "add_bus_with_transformer" and raw.get("length_km") == 0:
        # A length of 0 is no line to the substation, so no conductor for one
        # (observed live: line_like 'Bus_9' beside length_km 0).
        out.pop("line_like", None)
        out.pop("line_std_type", None)
    if not any(k in out for k in ("tap_step_percent", "trafo_like", "trafo_std_type")):
        # No step, and no unit or type to copy one from: no tap changer. A side,
        # a range of 0 to 0 and a position 0 are the form's defaults, not half of
        # one (observed live: tap_side 'hv' with zeros refused a plain unit).
        out.pop("tap_side", None)
        for k in ("tap_min", "tap_max", "tap_pos"):
            if out.get(k) == 0 and not isinstance(out.get(k), bool):
                out.pop(k)
    return out


def placeholder_voltages(net) -> bool:
    """Whether every bus of the network is at 1 kV: a converted per-unit case
    (MATPOWER/PGLib) whose voltages are placeholders, not physical levels."""
    vn = net.bus.loc[~lg._proposed_mask(net.bus), "vn_kv"] if len(net.bus) else []
    return len(vn) > 1 and bool((pd.Series(vn).astype(float).round(6) == 1.0).all())


PLACEHOLDER_KV_NOTE = (
    "Every bus of this network is at 1 kV: a converted per-unit case, so 1 kV is a placeholder for each "
    "bus's real voltage, not a physical level. A new transformer at such a bus is modelled in per unit on "
    "its own rating: its 1 kV side stands for the bus's real voltage, and its impedance (vk %) holds — "
    "build it with the voltage asked for. Its rated voltages are the buses' (a rated kV given for the real "
    "network is not used).")


def normalise_edit(net, raw: dict) -> dict:
    """Check one edit against `net` (the network it will be applied to) and
    return it in canonical form — every default filled, the bus or element
    stored as its index. Raises EditError with the reason."""
    if not isinstance(raw, dict):
        raise EditError("Each edit must be an object with an 'op'.")
    op = str(raw.get("op") or "").strip()
    if op not in OPS:
        raise EditError(f"Unknown edit {op!r}. Use one of {list(OPS)}.")
    edit = dict(_without_blanks(raw), op=op)

    if op == "add_load":
        cap = _size_cap(net)
        out = {"op": op, "name": _name(edit, net), "bus": resolve_bus(net, edit.get("bus")),
               "p_mw": _number(edit, "p_mw", low=0, high=cap, low_open=True),
               "pf": _number(edit, "pf", default=0.95, low=0.5, high=1.0),
               "profile": str(edit.get("profile") or "flat").strip()}
        if edit.get("leading") is not None:
            if not isinstance(edit["leading"], bool):
                raise EditError("add_load: leading must be true or false.")
            if edit["leading"]:
                out["leading"] = True       # injects reactive power (capacitive); only stored when so
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
        return _normalise_generator(net, edit)

    if op == "add_shunt":
        kind = str(edit.get("kind") or "").strip().lower()
        if kind not in SHUNT_KINDS:
            raise EditQuestion(
                "Is it a capacitor bank or a reactor?",
                ["capacitor — injects reactive power (raises nearby voltage)",
                 "reactor — absorbs reactive power (lowers nearby voltage)"])
        return {"op": op, "name": _name(edit, net), "bus": resolve_bus(net, edit.get("bus")),
                "kind": kind, "q_mvar": _number(edit, "q_mvar", low=0, high=_size_cap(net), low_open=True)}

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
        name = _new_bus_name(net, edit)
        at = resolve_bus(net, edit.get("connect_to"))
        vn = float(net.bus.at[at, "vn_kv"])
        if edit.get("vn_kv") is not None and abs(_number(edit, "vn_kv", low=0, low_open=True) - vn) > 1e-6:
            raise EditError(
                f"add_bus_with_connection: {_bus_label(net, at)} is at {vn:g} kV; a line joins buses "
                "of one voltage. Use add_bus_with_transformer for a bus at another voltage.")
        out = {"op": op, "name": name, "connect_to": at, "vn_kv": vn,
               "length_km": _number(edit, "length_km", low=0, high=MAX_CONNECTION_KM, low_open=True)}
        out.update(_line_choice(net, edit, "line_like", "std_type"))
        out.update(_voltage_band(edit))
        return out

    if op == "add_bus_with_transformer":
        return _normalise_transformer_site(net, edit)

    if op == "set_parameter":
        return _normalise_set_parameter(net, edit)
    if op == "add_line":
        return _normalise_add_line(net, edit)
    if op == "add_trafo":
        return _normalise_add_trafo(net, edit)
    if op == "move_element":
        return _normalise_move(net, edit)
    if op == "scale":
        return _normalise_scale(net, edit)

    # set_in_service
    etype = str(edit.get("element_type") or "").strip()
    idx = resolve_element(net, etype, edit.get("element"))
    state = edit.get("in_service")
    if not isinstance(state, bool):
        raise EditError("set_in_service: in_service must be true or false.")
    return {"op": op, "element_type": etype, "element": idx, "in_service": state}


def _normalise_add_line(net, edit: dict) -> dict:
    """A new line between two existing buses of one voltage (meshing, a
    second circuit, closing a loop)."""
    op = edit["op"]
    a = resolve_bus(net, edit.get("from_bus"))
    b = resolve_bus(net, edit.get("to_bus"))
    if a == b:
        raise EditError(f"{op}: a line needs two different buses.")
    va, vb = float(net.bus.at[a, "vn_kv"]), float(net.bus.at[b, "vn_kv"])
    if abs(va - vb) > 1e-6 * va:
        raise EditError(f"{op}: {_bus_label(net, a)} is at {va:g} kV and {_bus_label(net, b)} at {vb:g} kV; a line "
                        "joins buses of one voltage. Use add_trafo between buses of different voltages.")
    name = str(edit.get("name") or "").strip() or f"Line {_bus_label(net, a)}–{_bus_label(net, b)}"
    taken = set(net.line["name"].astype(str)) if "name" in net.line.columns else set()
    if name in taken:
        raise EditError(f"{op}: a line is already named {name!r}; give another name.")
    out = {"op": op, "from_bus": a, "to_bus": b, "name": name,
           "length_km": _number(edit, "length_km", low=0, high=MAX_CONNECTION_KM, low_open=True),
           "parallel": _integer(edit, "parallel", default=1, low=1, high=MAX_PARALLEL_UNITS)}
    out.update(_line_choice(net, edit, "line_like", "std_type"))
    return out


def _normalise_add_trafo(net, edit: dict) -> dict:
    """A new transformer between two existing buses of different voltages."""
    op = edit["op"]
    a = resolve_bus(net, edit.get("hv_bus"))
    b = resolve_bus(net, edit.get("lv_bus"))
    if a == b:
        raise EditError(f"{op}: a transformer needs two different buses.")
    va, vb = float(net.bus.at[a, "vn_kv"]), float(net.bus.at[b, "vn_kv"])
    if abs(va - vb) <= 1e-6 * va:
        raise EditError(f"{op}: both buses are at {va:g} kV; use add_line between buses of one voltage.")
    hv, lv = (a, b) if va > vb else (b, a)
    std = str(edit.get("trafo_std_type") or "").strip()
    like = edit.get("trafo_like") if _given(edit, "trafo_like") else None
    if std and like is not None:
        raise EditError(f"{op}: give trafo_like or trafo_std_type, not both.")
    if std:
        if not pp.std_type_exists(net, std, element="trafo"):
            raise EditError(f"{op}: {std!r} is not a known transformer type.")
        data = pp.load_std_type(net, std, "trafo")
        pair = (float(data["vn_hv_kv"]), float(data["vn_lv_kv"]))
    elif like is not None:
        like = resolve_element(net, "trafo", like)
        row = net.trafo.loc[like]
        pair = (float(net.bus.at[int(row["hv_bus"]), "vn_kv"]), float(net.bus.at[int(row["lv_bus"]), "vn_kv"]))
    else:
        pair = None
    name = str(edit.get("name") or "").strip() or f"Transformer {_bus_label(net, hv)}–{_bus_label(net, lv)}"
    out = {"op": op, "hv_bus": hv, "lv_bus": lv, "name": name}
    _normalise_trafo_unit(net, edit, out, std, like, pair, max(va, vb), min(va, vb))
    return out


def _normalise_move(net, edit: dict) -> dict:
    """Reconnect an element to another bus, or one end of a line. A measured
    load or unit keeps the measurements of the bus it was measured at (its
    own series): only its connection changes."""
    op = edit["op"]
    etype = str(edit.get("element_type") or "").strip()
    if etype not in MOVABLE:
        raise EditError(f"{op}: element_type must be one of {list(MOVABLE)}, not {etype!r}.")
    element = edit.get("element")
    if element in (None, "") and etype == "ext_grid" and len(net.ext_grid) == 1:
        element = int(net.ext_grid.index[0])
    idx = resolve_element(net, etype, element, allowed=MOVABLE)
    to = resolve_bus(net, edit.get("to_bus"))
    tbl = net[etype]
    if etype == "line":
        end = str(edit.get("end") or "").strip().lower()
        if end not in ("from", "to"):
            raise EditError(f"{op}: say which end of the line moves: end 'from' or 'to'.")
        other = int(tbl.at[idx, "to_bus" if end == "from" else "from_bus"])
        if to == other:
            raise EditError(f"{op}: the line would start and end at {_bus_label(net, to)}.")
        if int(tbl.at[idx, f"{end}_bus"]) == to:
            raise EditError(f"{op}: that end is already at {_bus_label(net, to)}.")
        v_to, v_other = float(net.bus.at[to, "vn_kv"]), float(net.bus.at[other, "vn_kv"])
        if abs(v_to - v_other) > 1e-6 * v_other:
            raise EditError(f"{op}: {_bus_label(net, to)} is at {v_to:g} kV and the line's other end at "
                            f"{v_other:g} kV; a line joins buses of one voltage.")
        return {"op": op, "element_type": etype, "element": idx, "end": end, "to_bus": to}
    if int(tbl.at[idx, "bus"]) == to:
        raise EditError(f"{op}: it is already at {_bus_label(net, to)}.")
    if etype == "sgen" and _measured_substation(net) and not bool(lg._proposed_mask(tbl).get(idx, False)):
        raise EditError(f"{op}: on this network the generation units are rebuilt from the measured substations at "
                        "every tick, so a measured unit cannot be moved. Switch it out (set_in_service) and add "
                        "a generator at the new bus instead.")
    if etype in ("gen", "ext_grid"):
        held = _bus_setpoint(net, to)
        vm = float(tbl.at[idx, "vm_pu"])
        if held is not None and abs(held[0] - vm) > 1e-6:
            raise EditError(f"{op}: {_bus_label(net, to)} is already held at {held[0]:g} p.u. by {held[1]}, and this "
                            f"unit holds {vm:g} p.u. Two units on one bus must hold the same voltage: set its vm_pu "
                            f"to {held[0]:g} first (set_parameter), or choose another bus.")
    return {"op": op, "element_type": etype, "element": idx, "to_bus": to}


def _normalise_scale(net, edit: dict) -> dict:
    """Scale existing loads or units by a factor, keeping their measured
    profile (load growth, a larger plant of the same shape)."""
    op = edit["op"]
    etype = str(edit.get("element_type") or "").strip()
    if etype not in SCALABLE:
        raise EditError(f"{op}: element_type must be one of {list(SCALABLE)}, not {etype!r}.")
    tbl = net[etype]
    if isinstance(edit.get("elements"), list) and edit["elements"]:     # a stored edit, or a list
        targets = [resolve_element(net, etype, e, allowed=SCALABLE) for e in edit["elements"]]
    elif _given(edit, "element"):
        targets = [resolve_element(net, etype, edit["element"], allowed=SCALABLE)]
    elif _given(edit, "bus"):
        bus = resolve_bus(net, edit["bus"])
        targets = [int(i) for i in tbl.index[tbl["bus"] == bus]]
        if not targets:
            raise EditError(f"{op}: there is no {etype} at {_bus_label(net, bus)}.")
    elif edit.get("all") is True:
        targets = [int(i) for i in tbl.index]
        if not targets:
            raise EditError(f"{op}: this network has no {etype}.")
    else:
        raise EditError(f"{op}: name the element (or elements), a bus (every {etype} at it), or all: true "
                        f"(every {etype}).")
    proposed = lg._proposed_mask(tbl)
    added = [i for i in targets if bool(proposed.get(i, False))]
    if added and not (_given(edit, "bus") or edit.get("all") is True):
        raise EditError(f"{op}: {_element_label(net, etype, added[0])} was added by this proposal; change its "
                        "size in the edit that added it.")
    targets = [i for i in targets if i not in added]
    if not targets:
        raise EditError(f"{op}: every {etype} there was added by this proposal; change their sizes in the "
                        "edits that added them.")
    if etype == "sgen" and _measured_substation(net):
        raise EditError(f"{op}: on this network the generation units are rebuilt from the measured substations at "
                        "every tick, so they cannot be scaled one by one; use the study's sgen scaling instead.")
    factor = _number(edit, "factor", low=0, high=MAX_SCALE)
    return {"op": op, "element_type": etype, "elements": targets, "factor": factor}


def _editable_fields(net, etype: str) -> list[str]:
    tbl = net[etype]
    cols = [c for c in tbl.columns if c not in _STRUCTURAL and not str(c).startswith("_")
            and (pd.api.types.is_numeric_dtype(tbl[c]) or pd.api.types.is_bool_dtype(tbl[c]) or c == "name")]
    return sorted(set(cols) | set(_OPTIONAL_COLUMNS.get(etype, ())))


def _normalise_set_parameter(net, edit: dict) -> dict:
    """Check a set_parameter edit: the element, the fields, and each value."""
    etype = str(edit.get("element_type") or "").strip()
    present = [t for t in EDITABLE_TABLES if t in net and len(net[t])]
    if etype not in present:
        raise EditError(f"set_parameter: element_type must be one of {present}, not {etype!r}.")
    element = edit.get("element")
    if element in (None, "") and etype == "ext_grid" and len(net.ext_grid) == 1:
        element = int(net.ext_grid.index[0])            # the only external grid
    idx = resolve_element(net, etype, element, allowed=EDITABLE_TABLES)
    values = edit.get("values")
    if not values:
        field = edit.get("field")
        if not field:
            raise EditError("set_parameter: give `field` and `value` (or `enabled` for a true/false field).")
        value = edit["value"] if _given(edit, "value") else edit.get("enabled")
        values = {field: value}
    if not isinstance(values, dict):
        raise EditError("set_parameter: values must map field names to values.")
    tbl = net[etype]
    editable = _editable_fields(net, etype)
    out: dict = {}
    for field, value in values.items():
        field = str(field).strip()
        if field in _STRUCTURAL:
            raise EditError(f"set_parameter: {field!r} is how the {etype} is connected, not a setting. To rewire, "
                            f"switch it out (set_in_service) and add a new element.")
        if field not in editable:
            raise EditError(f"set_parameter: a {etype} has no editable field {field!r}. Editable: {editable}.")
        is_bool = field in _BOOL_FIELDS or (field in tbl.columns and pd.api.types.is_bool_dtype(tbl[field]))
        if is_bool:
            if not isinstance(value, bool):
                raise EditError(f"set_parameter: {field} must be true or false.")
            out[field] = value
            continue
        if field == "name":
            if not str(value or "").strip():
                raise EditError("set_parameter: a name cannot be empty.")
            out[field] = str(value).strip()
            continue
        number = _finite(value)
        if number is None:
            raise EditError(f"set_parameter: {field} must be a number, not {value!r}.")
        if field in _BOUNDS:
            low, high = _BOUNDS[field]
            if not low <= number <= high:
                raise EditError(f"set_parameter: {field} = {number:g} is outside {low:g}–{high:g}.")
        if field in _POSITIVE and number <= 0:
            raise EditError(f"set_parameter: {field} must be positive.")
        if field in _NON_NEGATIVE and number < 0:
            raise EditError(f"set_parameter: {field} cannot be negative.")
        if field == "tap_pos":
            lo = _finite(tbl.at[idx, "tap_min"]) if "tap_min" in tbl.columns else None
            hi = _finite(tbl.at[idx, "tap_max"]) if "tap_max" in tbl.columns else None
            if lo is None or hi is None:
                raise EditError(f"set_parameter: this {etype} has no tap changer.")
            if number != int(number) or not lo <= number <= hi:
                raise EditError(f"set_parameter: tap_pos must be a whole step between {lo:g} and {hi:g}.")
            number = int(number)
        if field in ("parallel",):
            number = int(number)
        if field == "p_mw" and abs(number) > _size_cap(net):
            raise EditError(f"set_parameter: {number:g} MW is more than {MAX_SIZE_VS_NETWORK:g}× the network's "
                            "demand — check the value.")
        out[field] = number
    if not out:
        raise EditError("set_parameter: no field to set.")
    return {"op": "set_parameter", "element_type": etype, "element": idx, "values": out}


def before_tick(net) -> None:
    """Before a tick's assignment: moved loads and units go back to the bus
    they were measured at (so they receive their own measurements), and
    scaled ones back to their unscaled value (so a value the tick does not
    rewrite is not scaled twice)."""
    for etype in _MEASURED:
        tbl = net[etype] if etype in net else None
        if tbl is None or not len(tbl):
            continue
        if "bus_measured" in tbl.columns:
            moved = tbl["bus_measured"].notna()
            if moved.any():
                tbl.loc[moved, "bus"] = tbl.loc[moved, "bus_measured"].astype(int)
        for col in _MEASURED[etype]:
            base = f"_unscaled_{col}"
            if base in tbl.columns:
                rows = tbl[base].notna()
                tbl.loc[rows, col] = tbl.loc[rows, base]


def _scale(tbl, idx, etype: str, factor: float, base_from_column: bool) -> None:
    """P (and Q) of one row × factor, keeping the unscaled value. A field
    fixed by set_parameter stays as fixed."""
    fixed = tbl.at[idx, "override"] if "override" in tbl.columns else None
    fixed = json.loads(fixed) if isinstance(fixed, str) and fixed else {}
    for col in _MEASURED[etype]:
        if col not in tbl.columns or col in fixed:
            continue
        if f"_unscaled_{col}" not in tbl.columns:
            tbl[f"_unscaled_{col}"] = np.nan
        kept = tbl.at[idx, f"_unscaled_{col}"]
        base = float(kept) if base_from_column and pd.notna(kept) else float(tbl.at[idx, col])
        tbl.at[idx, f"_unscaled_{col}"] = base
        tbl.at[idx, col] = base * factor


def apply_overrides(net) -> None:
    """After a tick's assignment: the values a proposal fixed (set_parameter),
    the factors it scaled by (scale) and the buses it moved elements to
    (move_element) — all of which the measurements would otherwise undo.
    `before_tick` must have run before the assignment."""
    for etype in _MEASURED:
        tbl = net[etype] if etype in net else None
        if tbl is None or not len(tbl):
            continue
        if "override" in tbl.columns:
            for idx in tbl.index:
                spec = tbl.at[idx, "override"]
                if isinstance(spec, str) and spec:
                    for field, value in json.loads(spec).items():
                        tbl.at[idx, field] = value
        if "scale_factor" in tbl.columns:
            for idx in tbl.index[tbl["scale_factor"].notna()]:
                _scale(tbl, idx, etype, float(tbl.at[idx, "scale_factor"]), base_from_column=False)
        if "bus_moved" in tbl.columns:
            moved = tbl["bus_moved"].notna()
            if moved.any():
                tbl.loc[moved, "bus"] = tbl.loc[moved, "bus_moved"].astype(int)


def _element_label(net, etype: str, idx: int) -> str:
    if etype == "bus":
        return _bus_label(net, idx)
    if etype == "ext_grid":
        return f"external grid at {_bus_label(net, int(net.ext_grid.at[idx, 'bus']))}"
    return f"{etype} {_element_labels(net, etype, idx)[-1]}"


# ---------------------------------------------------------------------------
# Materialisation
# ---------------------------------------------------------------------------

def _create_trafo(net, hv: int, lv: int, edit: dict, name: str) -> int:
    """A proposed transformer between `hv` and `lv` from an edit's type,
    copy or values."""
    d = trafo_design(net, edit)
    if "trafo_std_type" in edit:
        extra = {k: edit[k] for k in ("tap_pos", "max_loading_percent") if k in edit}
        idx = pp.create_transformer(net, hv, lv, std_type=edit["trafo_std_type"], name=name,
                                    parallel=edit["parallel"], **extra)
    else:
        optional = {k: d[k] for k in ("tap_side", "tap_neutral", "tap_min", "tap_max",
                                      "tap_step_percent", "tap_step_degree", "tap_pos",
                                      "tap_changer_type", "max_loading_percent")
                    if d[k] is not None}
        idx = pp.create_transformer_from_parameters(
            net, hv, lv, sn_mva=d["sn_mva"], vn_hv_kv=d["vn_hv_kv"], vn_lv_kv=d["vn_lv_kv"],
            vk_percent=d["vk_percent"], vkr_percent=d["vkr_percent"], pfe_kw=d["pfe_kw"],
            i0_percent=d["i0_percent"], shift_degree=d["shift_degree"], name=name,
            parallel=edit["parallel"], **optional)
    _mark(net.trafo, idx)
    return idx


def _set_column(tbl, idx: int, col: str, value) -> None:
    if col not in tbl.columns:
        tbl[col] = np.nan
    tbl.at[idx, col] = value


def _apply(net, edit: dict) -> None:
    op = edit["op"]
    if op == "add_line":
        idx = _new_line(net, edit["from_bus"], edit["to_bus"], edit["length_km"], edit["name"],
                        like=edit.get("line_like"), std=edit.get("std_type"))
        net.line.at[idx, "parallel"] = edit["parallel"]
        return
    if op == "add_trafo":
        _create_trafo(net, edit["hv_bus"], edit["lv_bus"], edit, edit["name"])
        return
    if op == "move_element":
        tbl, idx = net[edit["element_type"]], edit["element"]
        if edit["element_type"] == "line":
            tbl.at[idx, f"{edit['end']}_bus"] = edit["to_bus"]
            return
        if edit["element_type"] in SCALABLE and not bool(lg._proposed_mask(tbl).get(idx, False)):
            if "bus_measured" not in tbl.columns or pd.isna(tbl.at[idx, "bus_measured"]):
                _set_column(tbl, idx, "bus_measured", int(tbl.at[idx, "bus"]))
            _set_column(tbl, idx, "bus_moved", int(edit["to_bus"]))
        tbl.at[idx, "bus"] = edit["to_bus"]
        return
    if op == "scale":
        tbl = net[edit["element_type"]]
        for idx in edit["elements"]:
            old = tbl.at[idx, "scale_factor"] if "scale_factor" in tbl.columns else np.nan
            total = (1.0 if pd.isna(old) else float(old)) * edit["factor"]
            _set_column(tbl, idx, "scale_factor", total)
            _scale(tbl, idx, edit["element_type"], total, base_from_column=True)   # the base case too
        return
    if op == "set_parameter":
        tbl = net[edit["element_type"]]
        idx = edit["element"]
        for field, value in edit["values"].items():
            if field not in tbl.columns:
                tbl[field] = np.nan if not isinstance(value, (bool, str)) else None
            if isinstance(value, (bool, str)) and tbl[field].dtype != object and not \
                    (isinstance(value, bool) and pd.api.types.is_bool_dtype(tbl[field])):
                tbl[field] = tbl[field].astype(object)
            tbl.at[idx, field] = value
        measured = {f: v for f, v in edit["values"].items() if f in _MEASURED.get(edit["element_type"], ())}
        if measured:
            if "override" not in tbl.columns:
                tbl["override"] = None
            tbl["override"] = tbl["override"].astype(object)
            old = tbl.at[idx, "override"]
            merged = {**(json.loads(old) if isinstance(old, str) and old else {}), **measured}
            tbl.at[idx, "override"] = json.dumps(merged, sort_keys=True)
        return
    if op == "add_load":
        q = edit["p_mw"] * math.tan(math.acos(edit["pf"])) * (-1 if edit.get("leading") else 1)
        idx = pp.create_load(net, bus=edit["bus"], p_mw=edit["p_mw"], q_mvar=q, name=edit["name"])
        _tag(net.load, idx, edit)
        if "substation_name" in net.load.columns:
            net.load.at[idx, "substation_name"] = edit["name"]
    elif op == "add_generator":
        p0 = edit.get("p_mw", 0.0)
        common = dict(bus=edit["bus"], p_mw=p0, name=edit["name"], max_p_mw=edit["p_max_mw"],
                      min_p_mw=edit.get("p_min_mw", 0.0))
        if edit.get("control") == "voltage":
            # A synchronous machine: holds its bus voltage, Q within its limits.
            idx = pp.create_gen(net, vm_pu=edit["vm_pu"], min_q_mvar=edit["q_min_mvar"],
                                max_q_mvar=edit["q_max_mvar"], **common)
            table = "gen"
        else:
            idx = pp.create_sgen(net, q_mvar=float(edit.get("q_mvar", 0.0) if edit.get("control") == "fixed_q"
                                                   else 0.0), type=edit["kind"], **common)
            table = "sgen"
            # The OPF and every result name an sgen by `substation_name` when the
            # column exists; an added unit gets its own name, not NaN.
            if "substation_name" in net.sgen.columns:
                net.sgen.at[idx, "substation_name"] = edit["name"]
        _tag(net[table], idx, edit)
        if "controllable" not in net[table].columns:
            net[table]["controllable"] = None
        net[table]["controllable"] = net[table]["controllable"].astype(object)
        net[table].at[idx, "controllable"] = bool(edit.get("dispatchable", True))
    elif op == "add_shunt":
        # pandapower's sign: q_mvar > 0 absorbs (a reactor), < 0 injects (a capacitor).
        q = edit["q_mvar"] if edit["kind"] == "reactor" else -edit["q_mvar"]
        idx = pp.create_shunt(net, bus=edit["bus"], q_mvar=q, p_mw=0.0, name=edit["name"])
        _mark(net.shunt, idx)
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
        bus = _new_bus(net, edit["vn_kv"], edit["name"], at, edit)
        _new_line(net, at, bus, edit["length_km"], f"{edit['name']} connection",
                  like=edit.get("line_like"), std=edit.get("std_type"))
    elif op == "add_bus_with_transformer":
        at = edit["connect_to"]
        v_at = float(net.bus.at[at, "vn_kv"])
        if "length_km" in edit:
            sub = _new_bus(net, v_at, substation_bus_name(edit["name"], v_at), at, {})
            _new_line(net, at, sub, edit["length_km"], f"{edit['name']} connection",
                      like=edit.get("line_like"), std=edit.get("line_std_type"))
            at = sub
        bus = _new_bus(net, edit["vn_kv"], edit["name"], edit["connect_to"], edit)
        hv, lv = (at, bus) if v_at > edit["vn_kv"] else (bus, at)
        _create_trafo(net, hv, lv, edit, f"{edit['name']} transformer")
    else:
        net[edit["element_type"]].at[edit["element"], "in_service"] = edit["in_service"]


def _new_bus(net, vn_kv: float, name: str, like_bus: int, edit: dict) -> int:
    """A proposed bus; voltage limits, zone and type from `like_bus` unless
    the edit gives its own limits."""
    bus = pp.create_bus(net, vn_kv=vn_kv, name=name)
    for col in ("min_vm_pu", "max_vm_pu", "zone", "type"):
        if col in edit:
            net.bus.at[bus, col] = edit[col]
        elif col in net.bus.columns and pd.notna(net.bus.at[like_bus, col]):
            net.bus.at[bus, col] = net.bus.at[like_bus, col]
    _mark(net.bus, bus)
    return bus


def _new_line(net, from_bus: int, to_bus: int, length_km: float, name: str, *, like=None, std=None) -> int:
    """A proposed line: a pandapower type, or the conductor of line `like`."""
    if std:
        line = pp.create_line(net, from_bus=from_bus, to_bus=to_bus, length_km=length_km,
                              std_type=std, name=name)
    else:
        src = net.line.loc[like]
        extra = {c: float(src[c]) for c in ("g_us_per_km", "max_loading_percent")
                 if c in net.line.columns and pd.notna(src.get(c))}
        line = pp.create_line_from_parameters(
            net, from_bus=from_bus, to_bus=to_bus, length_km=length_km,
            r_ohm_per_km=float(src["r_ohm_per_km"]), x_ohm_per_km=float(src["x_ohm_per_km"]),
            c_nf_per_km=float(src["c_nf_per_km"]), max_i_ka=float(src["max_i_ka"]),
            df=float(src["df"]) if pd.notna(src.get("df")) else 1.0, name=name, **extra)
    _mark(net.line, line)
    return line


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
        except EditQuestion as exc:
            raise EditQuestion(f"Edit {n}: {exc.question}", exc.options, edit=n) from None
        except EditError as exc:
            raise EditError(f"Edit {n}: {exc}") from None
        _apply(net, edit)
        canonical.append(edit)
    return net, canonical


def describe(net, edit: dict) -> str:
    """One line an operator can read."""
    op = edit["op"]
    if op == "add_line":
        how = f"like {line_display_name(net, edit['line_like'])}" if "line_like" in edit else f"type {edit['std_type']}"
        par = f", {edit['parallel']} in parallel" if edit["parallel"] > 1 else ""
        return (f"Add line {edit['name']!r}: {_bus_label(net, edit['from_bus'])} – {_bus_label(net, edit['to_bus'])}, "
                f"{edit['length_km']:g} km, {how}{par}")
    if op == "add_trafo":
        d = trafo_design(net, edit)
        par = f" ×{edit['parallel']}" if edit["parallel"] > 1 else ""
        return (f"Add transformer {edit['name']!r}: {_bus_label(net, edit['hv_bus'])} – "
                f"{_bus_label(net, edit['lv_bus'])}, {d['sn_mva']:g} MVA{par}")
    if op == "move_element":
        if edit["element_type"] == "line":
            return (f"Move the {edit['end']} end of line {line_display_name(net, edit['element'])} to "
                    f"{_bus_label(net, edit['to_bus'])}")
        kept = " (keeping its own measurements)" if edit["element_type"] in SCALABLE else ""
        return f"Move {_element_label(net, edit['element_type'], edit['element'])} to {_bus_label(net, edit['to_bus'])}{kept}"
    if op == "scale":
        names = ", ".join(_element_label(net, edit["element_type"], i) for i in edit["elements"][:4])
        more = f" and {len(edit['elements']) - 4} more" if len(edit["elements"]) > 4 else ""
        return f"Scale {names}{more} by ×{edit['factor']:g}, keeping the measured profile"
    if op == "set_parameter":
        what = ", ".join(f"{k} = {v:g}" if isinstance(v, (int, float)) and not isinstance(v, bool) else f"{k} = {v}"
                         for k, v in edit["values"].items())
        kept = [k for k in edit["values"] if k in _MEASURED.get(edit["element_type"], ())]
        tail = f" (held at every tick instead of the measured {', '.join(kept)})" if kept else ""
        return f"Set {_element_label(net, edit['element_type'], edit['element'])}: {what}{tail}"
    if op == "add_load":
        prof = edit["profile"]
        if prof == "shift":
            prof = f"shift {edit['start_hour']:02d}:00–{edit['end_hour']:02d}:00"
        elif prof == "follow_bus" and edit.get("follow", edit["bus"]) != edit["bus"]:
            prof = f"follow_bus (demand shape of {_bus_label(net, edit['follow'])})"
        return (f"Add load {edit['name']!r}: {edit['p_mw']:g} MW at {_bus_label(net, edit['bus'])}, "
                f"pf {edit['pf']:g}{' leading' if edit.get('leading') else ''}, profile {prof}")
    if op == "add_generator":
        extra = f", set-point {edit['p_mw']:g} MW" if edit["kind"] == "dispatchable" else ""
        if edit.get("p_min_mw"):
            extra += f" (minimum {edit['p_min_mw']:g} MW)"
        if edit.get("control") == "voltage":
            how = (f"holds {edit['vm_pu']:g} p.u., Q {edit['q_min_mvar']:g} to {edit['q_max_mvar']:g} Mvar")
        elif edit.get("control") == "fixed_q":
            q = edit["q_mvar"]
            how = f"fixed Q {abs(q):g} Mvar {'absorbed' if q < 0 else 'injected'}"
        else:
            pf = edit.get("pf", 1.0)
            how = "pf 1.00 (no reactive power)" if pf >= 1.0 else \
                f"pf {pf:g} {'absorbing' if edit.get('absorbing') else 'injecting'}"
        if edit.get("dispatchable") is False:
            how += ", not dispatchable"
        return (f"Add {edit['kind']} generator {edit['name']!r}: up to {edit['p_max_mw']:g} MW at "
                f"{_bus_label(net, edit['bus'])}{extra}; {how}")
    if op == "add_shunt":
        return (f"Add {edit['kind']} {edit['name']!r}: {edit['q_mvar']:g} Mvar at "
                f"{_bus_label(net, edit['bus'])} ({'injects' if edit['kind'] == 'capacitor' else 'absorbs'} "
                "reactive power)")
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
    if op == "add_bus_with_transformer":
        d = trafo_design(net, edit)
        units = f"{edit['parallel']} × " if edit["parallel"] > 1 else ""
        if "trafo_std_type" in edit:
            spec = f"type {edit['trafo_std_type']!r}"
        elif "trafo_like" in edit:
            spec = f"design of {trafo_display_name(net, edit['trafo_like'])}"
        else:
            spec = f"vk {d['vk_percent']:g} %, vkr {d['vkr_percent']:g} %"
        if d["tap_pos"] is not None and d["tap_pos"] != d["tap_neutral"]:
            spec += f", tap {int(d['tap_pos']):+d}"
        where = _bus_label(net, edit["connect_to"])
        if "length_km" in edit:
            conductor = (edit["line_std_type"] if "line_std_type" in edit
                         else f"as {line_display_name(net, edit['line_like'])}")
            where = f"the end of a {edit['length_km']:g} km line from {where} ({conductor})"
        return (f"New bus {edit['name']!r} ({edit['vn_kv']:g} kV) behind a new {units}{d['sn_mva']:g} MVA "
                f"{d['vn_hv_kv']:g}/{d['vn_lv_kv']:g} kV transformer ({spec}) at {where}")
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
        "new_buses": sum((e["op"] == "add_bus_with_connection")
                         + (e["op"] == "add_bus_with_transformer") * (1 + ("length_km" in e))
                         for e in edits),
        "new_trafos": sum(e["op"] in ("add_bus_with_transformer", "add_trafo") for e in edits),
        "new_transformer_mva": round(sum(trafo_design(net, e)["sn_mva"] * e["parallel"] for e in edits
                                         if e["op"] in ("add_bus_with_transformer", "add_trafo")), 4),
        "new_shunts_mvar": {k: round(sum(e["q_mvar"] for e in edits if e["op"] == "add_shunt"
                                         and e["kind"] == k), 4) for k in SHUNT_KINDS},
        "switched_elements": sum(e["op"] == "set_in_service" for e in edits),
        "changed_settings": sum(e["op"] == "set_parameter" for e in edits),
        "new_lines": sum(e["op"] == "add_line" for e in edits),
        "moved_elements": sum(e["op"] == "move_element" for e in edits),
        "scaled_elements": sum(len(e["elements"]) for e in edits if e["op"] == "scale"),
        "edits": [describe(net, e) for e in edits],
        "notes": notes(net, edits),
    }


def notes(net, edits: list[dict]) -> list[str]:
    """What the operator should know about how an edit was modelled."""
    out = []
    lengths = net.line.loc[~lg._proposed_mask(net.line), "length_km"] if len(net.line) else []
    placeholder = len(lengths) > 1 and bool((pd.Series(lengths).round(6) == 1.0).all())
    for e in edits:
        if e["op"] in ("add_bus_with_connection", "add_bus_with_transformer", "add_line") \
                and "line_like" in e and placeholder:
            # MATPOWER/PGLib conversions give every line 1 km and its whole
            # impedance per km: the copied line's length is not geography.
            out.append(
                f"Every line of this network has length 1 km (a converted per-unit case), so "
                f"{e['length_km']:g} km of {line_display_name(net, e['line_like'])}'s conductor "
                f"means {e['length_km']:g}× that line's impedance, not a {e['length_km']:g} km route.")
        if e["op"] in ("add_bus_with_transformer", "add_trafo"):
            if placeholder_voltages(net) and PLACEHOLDER_KV_NOTE not in out:
                out.append(PLACEHOLDER_KV_NOTE)
            out.extend(_transformer_notes(net, e))
    return out


def _transformer_notes(net, e: dict) -> list[str]:
    out = []
    d = trafo_design(net, e)
    label = (f"The new transformer {e['name']!r}" if e["op"] == "add_trafo"
             else f"The new transformer for {e['name']}")
    if "trafo_like" in e:
        src = net.trafo.loc[e["trafo_like"]]
        name = trafo_display_name(net, e["trafo_like"])
        vk = _finite(src.get("vk_percent"))
        if vk is not None and vk > 30 and "vk_percent" not in e:
            out.append(
                f"{name} has a short-circuit voltage of {vk:g} % on {float(src['sn_mva']):g} MVA — not a "
                "physical unit: a converted per-unit case stores the branch impedance that way. "
                f"{label} copies it{' on its new rating' if 'sn_mva' in e else ''}; give vk_percent and "
                "vkr_percent (or a trafo_std_type) for a realistic unit.")
        a = float(net.bus.at[int(src["hv_bus"]), "vn_kv"])
        b = float(net.bus.at[int(src["lv_bus"]), "vn_kv"])
        if (round(a, 6), round(b, 6)) != (round(d["vn_hv_kv"], 6), round(d["vn_lv_kv"], 6)):
            out.append(f"{name} joins {a:g}/{b:g} kV; its design is copied in per unit to a "
                       f"{d['vn_hv_kv']:g}/{d['vn_lv_kv']:g} kV unit.")
        pos, neutral = _finite(src.get("tap_pos")), _finite(src.get("tap_neutral")) or 0.0
        if pos is not None and pos != neutral and "tap_pos" not in e and d["tap_pos"] is not None:
            out.append(f"{name} runs at tap {int(pos):+d}; {label} starts at its neutral tap "
                       "(give tap_pos to match).")
    if "trafo_std_type" not in e and "trafo_like" not in e and d["pfe_kw"] == 0 and d["i0_percent"] == 0:
        out.append(f"{label} has no no-load losses or magnetising current (pfe_kw and i0_percent 0): "
                   "it draws nothing when unloaded.")
    hv_kv, lv_kv = _site_kv(net, e)
    off = (d["vn_hv_kv"] / d["vn_lv_kv"]) / (hv_kv / lv_kv) - 1
    if abs(off) > 1e-4:
        out.append(f"{label} is rated {d['vn_hv_kv']:g}/{d['vn_lv_kv']:g} kV between buses of "
                   f"{hv_kv:g}/{lv_kv:g} kV: at neutral tap its ratio is {off:+.1%} off nominal, which "
                   f"shifts the {'LV' if e['op'] == 'add_trafo' else 'new'} bus's voltage by about that much.")
    base_kv = net.bus.loc[~lg._proposed_mask(net.bus), "vn_kv"]
    if e["op"] == "add_bus_with_transformer" and len(net.trafo) > 1 and base_kv.nunique() == 1:
        out.append(f"Every bus of this network is at {float(base_kv.iloc[0]):g} kV (a converted per-unit "
                   f"case): the new bus's {e['vn_kv']:g} kV is a label. In per unit the transformer "
                   "adds its impedance and rating; its voltage ratio has no effect.")
    if e["parallel"] > 1:
        out.append(f"{label} is {e['parallel']} units in parallel, modelled as one branch: the N-1 sweep "
                   "takes all of them out together, which is harsher than losing one unit.")
    return out


# ---------------------------------------------------------------------------
# Profiles at a tick
# ---------------------------------------------------------------------------

def measurement_key_for_bus(net, bus: int, names: list[str]) -> str | None:
    """The measurement row a bus's demand comes from — the same matching
    `assign_load_values_from_measurements` applies to the loads on it."""
    label = measurement_name(net, bus)      # its name as loaded, if renamed
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
    if ts == NOMINAL:
        # The nominal point has no time: every element at its design value —
        # loads at their size, pv and wind at their rating, a dispatchable
        # unit at its set-point.
        if spec["op"] == "add_load":
            return 1.0, "nominal: its full size"
        if spec["kind"] == "dispatchable":
            return spec["p_mw"] / spec["p_max_mw"], "set-point"
        return 1.0, "nominal: its rated output"
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


def fixed_load_mask(tbl: pd.DataFrame) -> pd.Series:
    """Loads a proposal added with a flat profile: a stated, fixed demand that
    load-stress factors and demand-uncertainty draws leave unchanged."""
    if not len(tbl) or "profile" not in tbl.columns:
        return pd.Series(False, index=tbl.index)
    return lg._proposed_mask(tbl) & (tbl["profile"].astype(object) == "flat")


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
        net.load.at[idx, "q_mvar"] = p * math.tan(math.acos(spec["pf"])) * (-1 if spec.get("leading") else 1)
        report.append({"element": spec["name"], "type": "load", "p_mw": round(p, 4),
                       "size_mw": spec["p_mw"], "profile": spec["profile"], "source": source})
    for table in ("sgen", "gen"):
        for idx, spec in _proposed_rows(net[table]):
            f, source = profile_factor(net, spec, ts, meas, peaks)
            p = spec["p_max_mw"] * min(max(f, 0.0), 1.0)
            net[table].at[idx, "p_mw"] = p
            net[table].at[idx, "max_p_mw"] = spec["p_max_mw"] if spec["kind"] == "dispatchable" else p
            if table == "sgen" and spec.get("control") == "fixed_q":
                net.sgen.at[idx, "q_mvar"] = float(spec["q_mvar"])     # whatever its output
            elif table == "sgen":
                # A fixed-power-factor unit's reactive power follows its output.
                pf = spec.get("pf", 1.0)
                q = p * math.tan(math.acos(pf)) if pf < 1.0 else 0.0
                net.sgen.at[idx, "q_mvar"] = -q if spec.get("absorbing") else q
            report.append({"element": spec["name"], "type": "generator", "p_mw": round(p, 4),
                           "size_mw": spec["p_max_mw"], "profile": spec["kind"], "source": source,
                           "control": spec.get("control", "power_factor")})
    return report


def check_profiles(net, report: list[dict], load_scaling: float = 1.0) -> list[str]:
    """Each added element still injects what its profile gave it — after the
    measured values were assigned and any load stress applied. The failure
    this catches is silent by nature (sandbox hazard A.3 (1)); returns the
    discrepancies, empty when there are none."""
    wrong = []
    by_name = {r["element"]: r for r in report}
    for tbl, kind, scale in ((net.load, "load", load_scaling), (net.sgen, "generator", 1.0),
                             (net.gen, "generator", 1.0)):
        for idx, spec in _proposed_rows(tbl):
            want = by_name.get(spec["name"], {}).get("p_mw")
            got = float(tbl.at[idx, "p_mw"])
            if want is None or abs(got - want * scale) > 1e-3 * max(1.0, abs(want)):
                wrong.append(f"{kind} {spec['name']!r}: {got:.4f} MW instead of "
                             f"{(want or 0) * scale:.4f} MW")
    return wrong
