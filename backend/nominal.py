"""
The network's nominal operating point: the values it was loaded with.

A loaded case (PGLib, IEEE, an uploaded file) carries its own operating point —
each load's P and Q, each generator's set-point. The synthetic measurements are
built from it, and the first synthetic tick is then written over it, so without
a copy the case's own values were gone after loading. `snapshot` keeps them;
`apply` puts them back on a copy of the network for a study, per element and
exactly (a tick is spread per substation and assumes a 0.95 power factor; the
nominal point keeps the case's own reactive power).

"Nominal" is the case as the twin models it: generators converted to
controllable units keep their active set-point (their reactive power is 0 in
the model, as at every tick). It is a single moment with no time.

Pure functions of a pandapower network: no app state, no HTTP.
"""

from __future__ import annotations

import math

import pandas as pd

from generators import unit_key, unit_name

NOMINAL = "nominal"

# What a tick overwrites, per element table. Voltages and limits are not
# touched by a tick, so they need no copy. Generating units are kept by name,
# not by row: their model can move them between `gen` and `sgen` later
# (generators.py).
_COLUMNS = {
    "load": ("p_mw", "q_mvar"),
    "sgen": ("p_mw", "q_mvar"),
    "gen": ("p_mw",),
    "storage": ("p_mw", "q_mvar"),
}
_UNITS = ("gen", "sgen")
# Measured networks mark a missing value with a large negative number (-9999
# in the Bornholm data); a nominal point built on it would be nonsense.
_MISSING_BELOW = -9000.0


class NominalError(ValueError):
    """The nominal point cannot be used; the message says why."""


def _label(net, etype: str, idx) -> str:
    tbl = net[etype]
    for col in ("name", "substation_name"):
        if col in tbl.columns:
            raw = tbl.at[idx, col]
            if isinstance(raw, str) and raw.strip():
                return f"{etype} {raw.strip()!r}"
    return f"{etype} {idx}"


def snapshot(net) -> dict:
    """The values a tick will overwrite, as loaded. Call before the first
    tick is assigned."""
    tables, problems = {}, []
    for etype, cols in _COLUMNS.items():
        if etype not in net or not len(net[etype]):
            continue
        tbl = net[etype]
        present = [c for c in cols if c in tbl.columns]
        frame = tbl[present].apply(pd.to_numeric, errors="coerce").astype(float).copy()
        # Checked by row first: the labels read the row, before units are re-keyed by name.
        for idx, row in frame.iterrows():
            for col, value in row.items():
                if not math.isfinite(value):
                    problems.append(f"{_label(net, etype, idx)} has no {col}")
                elif value < _MISSING_BELOW:
                    problems.append(f"{_label(net, etype, idx)} {col} is {value:g} (a missing-value marker)")
        if etype in _UNITS:
            frame.index = [unit_name(net, etype, i) for i in frame.index]
        tables[etype] = frame
    return {"tables": tables, "problems": problems}


def usable(snap: dict | None) -> tuple[bool, str]:
    """(whether the nominal point can be studied, why not)."""
    if not snap:
        return False, "No nominal point is stored for the loaded network."
    if snap["problems"]:
        shown = "; ".join(snap["problems"][:5])
        more = f" (and {len(snap['problems']) - 5} more)" if len(snap["problems"]) > 5 else ""
        return False, f"The network file's own values are incomplete: {shown}{more}."
    return True, ""


def apply(net, snap: dict):
    """Put the nominal values back on `net` (a copy), row by row. Rows a
    proposal added are not in the snapshot and keep what their profile gives
    them. Raises NominalError when the snapshot is not usable."""
    ok, why = usable(snap)
    if not ok:
        raise NominalError(why)
    for etype, frame in snap["tables"].items():
        if etype in _UNITS:
            continue
        if etype not in net or not len(net[etype]):
            continue
        rows = frame.index.intersection(net[etype].index)
        for col in frame.columns:
            net[etype].loc[rows, col] = frame.loc[rows, col].to_numpy()
    # Units by name, in whichever table their model puts them now. Reactive
    # power only where the case fixed it (a static generator as loaded); a
    # voltage-controlled unit's is solved, a power-factor unit's set after.
    by_name = {name: row for etype in _UNITS for name, row in snap["tables"].get(etype, pd.DataFrame()).iterrows()}
    for etype in _UNITS:
        if etype not in net or not len(net[etype]):
            continue
        for idx in net[etype].index:
            row = by_name.get(unit_key(net, etype, idx))   # its name as loaded
            if row is None:
                continue
            net[etype].at[idx, "p_mw"] = float(row["p_mw"])
            if etype == "sgen" and "q_mvar" in row.index and math.isfinite(row["q_mvar"]):
                net.sgen.at[idx, "q_mvar"] = float(row["q_mvar"])
    return net


def totals(snap: dict) -> dict:
    """Total nominal demand and generation, MW and Mvar."""
    t = snap["tables"]

    def total(etype, col):
        frame = t.get(etype)
        return round(float(frame[col].sum()), 4) if frame is not None and col in frame else 0.0

    return {"load_mw": total("load", "p_mw"), "load_mvar": total("load", "q_mvar"),
            "generation_mw": round(total("sgen", "p_mw") + total("gen", "p_mw"), 4)}


def as_measurements(net, snap: dict, key_for_bus) -> pd.DataFrame:
    """The nominal point in the measurement schema (substation_name,
    production, consumption), so code that reads a tick's table works on it.
    `key_for_bus(bus) -> str` names a bus the way the measurements do."""
    rows: dict[str, dict] = {}
    for etype, column in (("load", "consumption"), ("sgen", "production"), ("gen", "production")):
        frame = snap["tables"].get(etype)
        if frame is None or "p_mw" not in frame:
            continue
        buses = _unit_buses(net) if etype in _UNITS else None
        for idx, p in frame["p_mw"].items():
            if etype in _UNITS:
                if idx not in buses:
                    continue
                bus = buses[idx]
            elif idx not in net[etype].index:
                continue
            else:
                bus = int(net[etype].at[idx, "bus"])
            key = key_for_bus(bus)
            entry = rows.setdefault(key, {"substation_name": key, "production": 0.0, "consumption": 0.0})
            entry[column] += float(p)
    return pd.DataFrame(list(rows.values()), columns=["substation_name", "production", "consumption"])


def _unit_buses(net) -> dict:
    """{unit name: bus} over both unit tables."""
    return {unit_key(net, etype, idx): int(net[etype].at[idx, "bus"])
            for etype in _UNITS if etype in net for idx in net[etype].index}
