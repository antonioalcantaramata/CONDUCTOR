"""
The operator's names for buses, lines, transformers, loads and units.

A rename is a layer over the network as loaded, like the generator model
(generators.py): kept as {table: {index: name}}, applied after the model,
changeable and resettable, saved for this network across restarts. Every
result reads the new name from then on.

What must NOT follow a rename is kept on the row:

- ``loaded_name`` — the row's name as loaded ("" when it had none). A bus's
  measurements are filed under it (``element_names.measurement_name``), and
  the network fingerprint reads it, so saved proposals and assumptions stay
  this network's.
- ``loaded_unit`` — a unit's name as loaded: the key of its model, costs and
  nominal values (``generators.unit_key``).

Line and transformer labels embed their end buses' names, so renaming a bus
relabels its branches too — by design, they are the same elements.

Pure functions of a pandapower network: no app state, no HTTP.
"""

from __future__ import annotations

import copy

import pandas as pd

import generators as gm
import sandbox as sb
from element_names import bus_display_name, clean_label, line_display_name, trafo_display_name

NAMEABLE = ("bus", "line", "trafo", "load", "gen", "sgen")
UNITS = ("gen", "sgen")
LOADED_NAME, LOADED_UNIT = "loaded_name", "loaded_unit"
MAX_LENGTH = 64


class RenameError(ValueError):
    """A rename that cannot be applied; the message says why."""


def _raw(tbl: pd.DataFrame, idx) -> str:
    return clean_label(tbl.at[idx, "name"]) if "name" in tbl.columns else ""


def display(net, table: str, idx) -> str:
    """How results name the element now."""
    if table == "bus":
        return bus_display_name(net, idx)
    if table == "line":
        return line_display_name(net, idx)
    if table == "trafo":
        return trafo_display_name(net, idx)
    if table in UNITS:
        return gm.unit_name(net, table, idx)
    return _raw(net[table], idx) or f"{table.capitalize()} {int(idx)}"


def _taken(net, table: str, idx) -> set[str]:
    """Names another element of the same kind already answers to."""
    if table in UNITS:
        return {gm.unit_name(net, t, i).casefold() for t in UNITS for i in net[t].index
                if not (t == table and i == idx)}
    tbl = net[table]
    names = {_raw(tbl, i).casefold() for i in tbl.index if i != idx} - {""}
    if table == "bus":
        names |= {bus_display_name(net, i).casefold() for i in tbl.index if i != idx}
    return names


def normalise(net, changes: list[dict], model: dict | None = None) -> dict:
    """Check renames against the base as it is now (`net`, with the model and
    any earlier renames applied) and return the new model. Each change:
    element_type, element (any label results show, or its index) and name
    (empty or null: back to the name as loaded)."""
    if not isinstance(changes, list) or not changes:
        raise RenameError("Give a list of renames, each with element_type, element and name.")
    out = copy.deepcopy(model or {})
    work = copy.deepcopy(net)
    for n, raw in enumerate(changes, start=1):
        if not isinstance(raw, dict):
            raise RenameError(f"Rename {n}: each rename is an object with element_type, element and name.")
        table = str(raw.get("element_type") or "").strip()
        if table not in NAMEABLE:
            raise RenameError(f"Rename {n}: element_type must be one of {list(NAMEABLE)}, not {table!r}.")
        try:
            idx = sb.resolve_element(work, table, raw.get("element"), allowed=NAMEABLE)
        except sb.EditError as exc:
            raise RenameError(f"Rename {n}: {exc}") from None
        tbl = work[table]
        if sb.lg._proposed_mask(tbl).get(idx, False):
            raise RenameError(f"Rename {n}: that element was added by a proposal; name it in the proposal.")
        if table == "sgen" and sb._measured_substation(work):
            raise RenameError(f"Rename {n}: on this network the generation units are rebuilt from the measured "
                             "substations at every tick, so they keep the substations' names.")
        name = str(raw.get("name") or "").strip()
        key = str(int(idx))
        if not name:                                   # back to the name as loaded
            out.get(table, {}).pop(key, None)
            if not out.get(table):
                out.pop(table, None)
            work = apply(work, {table: {key: None}})
            continue
        if len(name) > MAX_LENGTH:
            raise RenameError(f"Rename {n}: a name has at most {MAX_LENGTH} characters.")
        if any(c in name for c in "|[]"):
            raise RenameError(f"Rename {n}: '|', '[' and ']' are reserved in element labels.")
        if name.casefold() in _taken(work, table, idx):
            raise RenameError(f"Rename {n}: another {table} is already called {name!r}; names must be unique.")
        out.setdefault(table, {})[key] = name
        work = apply(work, {table: {key: name}})
    return out


def apply(net, model: dict):
    """`net` (a copy) with the operator's names. A None name restores the
    name as loaded. The names as loaded are kept on the rows (see above)."""
    if not model:
        return net
    net = copy.deepcopy(net)
    for table, names in model.items():
        if table not in NAMEABLE or table not in net:
            continue
        tbl = net[table]
        for key, name in (names or {}).items():
            idx = int(key)
            if idx not in tbl.index:
                continue
            for col in (LOADED_NAME, LOADED_UNIT) if table in UNITS else (LOADED_NAME,):
                if col not in tbl.columns:
                    tbl[col] = None
                    tbl[col] = tbl[col].astype(object)
            if not isinstance(tbl.at[idx, LOADED_NAME], str):
                if table in UNITS:
                    tbl.at[idx, LOADED_UNIT] = gm.unit_name(net, table, idx)
                tbl.at[idx, LOADED_NAME] = _raw(tbl, idx)
            new = name if name else (tbl.at[idx, LOADED_NAME] or None)
            if "name" not in tbl.columns:
                tbl["name"] = None
            tbl["name"] = tbl["name"].astype(object)
            tbl.at[idx, "name"] = new
            if table == "sgen" and "substation_name" in tbl.columns:
                # A unit's name is its substation_name where it has one (gm.unit_name).
                tbl["substation_name"] = tbl["substation_name"].astype(object)
                tbl.at[idx, "substation_name"] = new if name else tbl.at[idx, LOADED_UNIT]
            if not name:
                tbl.at[idx, LOADED_NAME] = None
                if table in UNITS:
                    tbl.at[idx, LOADED_UNIT] = None
    return net


def describe(net, model: dict) -> list[dict]:
    """Every renamed element: kind, index, the name as loaded and now."""
    out = []
    for table, names in (model or {}).items():
        for key, name in names.items():
            idx = int(key)
            if table not in net or idx not in net[table].index:
                continue
            kept = net[table].at[idx, LOADED_NAME] if LOADED_NAME in net[table].columns else None
            if table == "bus":
                was = clean_label(kept) or f"Bus_{idx}"
            elif table in UNITS:
                was = net[table].at[idx, LOADED_UNIT]
            else:
                was = clean_label(kept) or f"(unnamed {table} {idx})"
            out.append({"element_type": table, "index": idx, "loaded_name": was, "name": name})
    return out


def unit_renames(net) -> dict[str, str]:
    """{name as loaded: name now} for every renamed unit."""
    out = {}
    for table in UNITS:
        tbl = net[table]
        if LOADED_UNIT not in tbl.columns:
            continue
        for idx in tbl.index:
            kept = tbl.at[idx, LOADED_UNIT]
            if isinstance(kept, str) and kept:
                out[kept] = gm.unit_name(net, table, idx)
    return out


def load_renames(net) -> dict[str, str]:
    """{name as loaded: name now} for every renamed load (the key of its
    demand-response prices, costs.load_name)."""
    tbl = net.load
    if LOADED_NAME not in tbl.columns:
        return {}
    out = {}
    for idx in tbl.index:
        kept = tbl.at[idx, LOADED_NAME]
        if isinstance(kept, str):
            out[kept.strip() or f"Load {int(idx)}"] = display(net, "load", idx)
    return out


def elements(net) -> dict[str, list[dict]]:
    """Every nameable element (not added by a proposal) with its name now and
    as loaded — what the System tab's names table lists."""
    out: dict[str, list[dict]] = {}
    for table in NAMEABLE:
        tbl = net[table] if table in net else None
        if tbl is None or not len(tbl):
            continue
        proposed = sb.lg._proposed_mask(tbl)
        rows = []
        for idx in tbl.index:
            if proposed.get(idx, False):
                continue
            kept = tbl.at[idx, LOADED_NAME] if LOADED_NAME in tbl.columns else None
            renamed = isinstance(kept, str)
            row = {"index": int(idx), "name": display(net, table, idx), "renamed": renamed}
            if table == "bus":
                row["label"] = bus_display_name(net, idx)
            elif table in ("line", "trafo"):
                row["label"] = _raw(tbl, idx)
                row["ends"] = (f"{bus_display_name(net, int(tbl.at[idx, 'from_bus' if table == 'line' else 'hv_bus']))}"
                               f" – {bus_display_name(net, int(tbl.at[idx, 'to_bus' if table == 'line' else 'lv_bus']))}")
            else:
                row["label"] = display(net, table, idx)
                row["bus"] = bus_display_name(net, int(tbl.at[idx, "bus"]))
            rows.append(row)
        out[table] = rows
    return out
