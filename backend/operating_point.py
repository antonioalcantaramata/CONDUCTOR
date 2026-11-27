"""
operating_point.py — a moment of the clock as a steady-state hypothesis of the CIM model, by mRID.

At each tick the measurements are allocated to the network's loads and units (load_gen_assignment.py). What comes out
is a moment: the operating point of every measured element — its active and reactive power, whether its reactive
power was measured, the measurement it was matched to — keyed by mRID, as a CGMES steady-state hypothesis (SSH)
states it. A moment is computed once for the networks that share their measured elements and what the allocation
reads of them (`signature`: the base and the proposals that leave its measured elements alone) and applied to each by
mRID (`apply`); a network whose measured elements differ (a unit's capacity changed, a measured-substation network
whose units are rebuilt at every tick) is allocated on its own. `ssh_profile` writes a network's moment as an SSH
profile against its model, for another tool.
"""

from __future__ import annotations

import copy
import hashlib

import pandas as pd

import load_gen_assignment as lg

TABLES = ("load", "gen", "sgen")
# What the allocation reads of each measured element (besides its identity), and of the buses.
_READS = {"load": ("bus", "name", "_meas_weight", "p_mw"), "gen": ("bus", "name", "max_p_mw"),
          "sgen": ("bus", "name", "max_p_mw")}
_IDENTITY = ("origin_id", "loaded_table", "loaded_index")


def _own(df: pd.DataFrame) -> pd.Series:
    return lg._proposed_mask(df) if len(df) else pd.Series(False, index=df.index)


def signature(net) -> str | None:
    """What the allocation reads of a network's measured elements and buses, as a digest: two networks with the same
    signature get the same moment for those elements. None for a network whose units the allocation rebuilds at
    every tick (it is allocated on its own)."""
    if not (len(net.gen) or net.get("_gen_to_sgen_applied", False)):
        return None
    h = hashlib.sha1()
    for table in TABLES:
        df = net[table]
        rows = df[~_own(df)]
        cols = [c for c in (*_READS[table], *_IDENTITY) if c in rows.columns]
        if table == "load" and "_meas_weight" in rows.columns:
            cols = [c for c in cols if c != "p_mw"]          # the weight, once set, is what the split reads
        h.update(f"{table}|{list(rows.index)}|{cols}".encode())
        if len(rows) and cols:
            h.update(pd.util.hash_pandas_object(rows[cols].astype(str), index=True).values.tobytes())
    names = [lg.measurement_name(net, i) for i in net.bus.index]
    h.update(repr((list(net.bus.index), names)).encode())
    return h.hexdigest()


def allocate(net, meas, substations, power_factor: float) -> tuple[object, list, list]:
    """The measurements allocated on `net` as before (in place): (net, unmatched load buses, unmatched unit buses)."""
    net, unmatched_loads = lg.assign_load_values_from_measurements(net, meas, substations,
                                                                   power_factor=power_factor)
    net, unmatched_units = lg.assign_generators_values_from_measurements(net, meas, substations)
    return net, unmatched_loads, unmatched_units


def compute(net, meas, substations, power_factor: float, network_id: str) -> dict:
    """The moment of `net` at a tick: its measured elements' values after the allocation, by mRID —
    {"values": {table: {mRID: {column: value}}}, "columns": {table: [...]}, "share": the measured share of
    reactive demand}. The network itself is not changed."""
    import semantics as sm
    work = copy.deepcopy(net)
    before = {t: work[t].copy() for t in TABLES}
    work, _l, _u = allocate(work, meas, substations, power_factor)
    out = {"values": {}, "columns": {}, "share": float(work.get("_q_measured_share", 0.0) or 0.0)}
    for table in TABLES:
        old, new = before[table], work[table]
        own = _own(new)
        cols = []
        for c in new.columns:
            if c not in old.columns:
                cols.append(c)
                continue
            a, b = old[c].astype(object), new[c].astype(object)
            if ((a != b) & ~(a.isna() & b.isna())).any():
                cols.append(c)
        rows = new.index[~own]
        ids = sm.row_mrids(work, table, network_id)
        out["columns"][table] = cols
        out["values"][table] = {ids[int(i)]: {c: new.at[i, c] for c in cols} for i in rows}
    return out


def apply(net, moment: dict, meas, substations, network_id: str) -> None:
    """The moment on `net` (in place), by mRID: each measured element takes its values; an element the moment does
    not hold (one a proposal added — not measured) gets only what the allocation gives such an element (its bus's
    measurement name; no measured reactive power)."""
    import semantics as sm
    for table in TABLES:
        df = net[table]
        cols = moment["columns"].get(table) or []
        values = moment["values"].get(table) or {}
        created = [c for c in cols if c not in df.columns]
        for c in created:
            df[c] = None
            df[c] = df[c].astype(object)
        ids = sm.row_mrids(net, table, network_id)
        covered = [i for i, m in ids.items() if m in values]
        if covered and cols:
            frame = pd.DataFrame([values[ids[i]] for i in covered], index=covered, columns=cols)
            for c in cols:
                if pd.api.types.is_bool_dtype(df[c].dtype) or pd.api.types.is_numeric_dtype(df[c].dtype):
                    try:
                        df.loc[covered, c] = frame[c].astype(df[c].dtype).values
                        continue
                    except (TypeError, ValueError):
                        df[c] = df[c].astype(object)
                df.loc[covered, c] = frame[c].values
        rest = [i for i in df.index if int(i) not in set(covered)]
        if rest and table == "load":
            lg.label_loads(net, meas, substations, rows=rest)
            if "_meas_weight" in created:          # the allocation weighs every load by its demand when it starts
                df.loc[rest, "_meas_weight"] = df.loc[rest, "p_mw"].clip(lower=0).fillna(0.0).values
        if rest and table == "sgen" and "_q_from_meas" in cols:
            df.loc[rest, "_q_from_meas"] = False
    net["_q_measured_share"] = moment["share"]


def ssh_profile(net, network_id: str, version: str | None = None, description: str = "", created: str = "") -> bytes:
    """A network's operating point now as a CGMES SSH profile: its injections (loads, units, inverters, the external
    grid), the regulating controls' targets, terminals connected or not, switch positions — what the writer states in
    SSH for them, against the network's model (cgmes_writer.py: the same mRIDs)."""
    import cgmes_writer as cw
    only = {t: list(net[t].index) for t in ("load", "gen", "sgen", "storage", "ext_grid", "shunt", "switch", "line",
                                            "trafo", "trafo3w", "motor", "ward", "xward", "svc")
            if t in net and isinstance(net[t], pd.DataFrame)}
    import cim_changes as cc
    out = cw.write(net, network_id, version=version or cw.DEFAULT_VERSION, only=only)
    p = out["objects"]
    # A network that came as CIM: its model's own identifiers (terminals, controls…), as its differences use.
    rename = cc.own_ids(net, network_id)
    if rename:
        for profile, items in p.objects.items():
            p.objects[profile] = [(cls, rename.get(oid, oid),
                                   [(k, f"#{rename[v[1:]]}" if isinstance(v, str) and v.startswith("#")
                                     and v[1:] in rename else v) for k, v in props], about)
                                  for cls, oid, props, about in items]
    header = cw._id(network_id, "moment", description or created)
    return p.xml("SSH", header, [cw._id(network_id, "model", "EQ")], description or "Operating point", created)


# A result's sign as CGMES states a terminal's flow: into the equipment from the node (a load positive, a unit
# negative). pandapower reports a unit's and the external grid's output as positive.
_SV_SIGN = {"load": 1.0, "shunt": 1.0, "storage": 1.0, "ward": 1.0, "xward": 1.0, "motor": 1.0,
            "gen": -1.0, "sgen": -1.0, "ext_grid": -1.0}
_SV_BRANCH = {"line": (("p_from_mw", "q_from_mvar"), ("p_to_mw", "q_to_mvar")),
              "trafo": (("p_hv_mw", "q_hv_mvar"), ("p_lv_mw", "q_lv_mvar")),
              "trafo3w": (("p_hv_mw", "q_hv_mvar"), ("p_mv_mw", "q_mv_mvar"), ("p_lv_mw", "q_lv_mvar")),
              "impedance": (("p_from_mw", "q_from_mvar"), ("p_to_mw", "q_to_mvar"))}


def sv_profile(net, network_id: str, version: str | None = None, description: str = "", created: str = "") -> bytes:
    """A solved network as a CGMES state-variables (SV) profile against its model: each node's voltage and angle
    (SvVoltage), the power at every terminal of the equipment (SvPowerFlow), each tap changer's position
    (SvTapStep) — under the model's own identifiers for a network loaded as CIM, the writer's for any other (the
    same as GET /api/network/cim/package). `net` must have been solved."""
    import cgmes_writer as cw
    import cim_changes as cc
    import semantics as sm
    version = version or cw.DEFAULT_VERSION
    p = cw._Profiles(version)
    rename = cc.own_ids(net, network_id)

    def part(*parts) -> str:
        derived = cw._id(*parts)
        return rename.get(derived, derived)
    ids = sm.row_mrids(net, "bus", network_id)
    for i in net.res_bus.index:
        vm, va = net.res_bus.at[i, "vm_pu"], net.res_bus.at[i, "va_degree"]
        if pd.isna(vm) or int(i) not in ids:
            continue
        node = ids[int(i)]
        p.add("SV", "SvVoltage", cw._id(node, "sv"), [("cim:SvVoltage.v", float(vm) * float(net.bus.at[i, "vn_kv"])),
                                                      ("cim:SvVoltage.angle", float(va)),
                                                      ("cim:SvVoltage.TopologicalNode", f"#{node}")])
    for table, sign in _SV_SIGN.items():
        res = net[f"res_{table}"] if f"res_{table}" in net else None
        if res is None or not len(res):
            continue
        eq = sm.row_mrids(net, table, network_id)
        for i in res.index:
            pv, qv = res.at[i, "p_mw"], res.at[i, "q_mvar"] if "q_mvar" in res.columns else 0.0
            if pd.isna(pv) or int(i) not in eq:
                continue
            term = part(eq[int(i)], "terminal", 1)
            p.add("SV", "SvPowerFlow", cw._id(term, "sv"), [("cim:SvPowerFlow.p", sign * float(pv)),
                                                            ("cim:SvPowerFlow.q", sign * float(qv or 0.0)),
                                                            ("cim:SvPowerFlow.Terminal", f"#{term}")])
    for table, ends in _SV_BRANCH.items():
        res = net[f"res_{table}"] if f"res_{table}" in net else None
        if res is None or not len(res):
            continue
        eq = sm.row_mrids(net, table, network_id)
        for i in res.index:
            if int(i) not in eq:
                continue
            for seq, (pc, qc) in enumerate(ends, start=1):
                pv, qv = res.at[i, pc], res.at[i, qc]
                if pd.isna(pv):
                    continue
                term = part(eq[int(i)], "terminal", seq)
                p.add("SV", "SvPowerFlow", cw._id(term, "sv"), [("cim:SvPowerFlow.p", float(pv)),
                                                                ("cim:SvPowerFlow.q", float(qv)),
                                                                ("cim:SvPowerFlow.Terminal", f"#{term}")])
    for table in ("trafo", "trafo3w"):
        df = net[table] if table in net else None
        if df is None or not len(df) or "tap_pos" not in df.columns:
            continue
        eq = sm.row_mrids(net, table, network_id)
        for i in df.index:
            pos = df.at[i, "tap_pos"]
            if pd.isna(pos) or int(i) not in eq:
                continue
            kind = str(df.at[i, "tap_changer_type"]) if "tap_changer_type" in df.columns else ""
            tap = part(eq[int(i)], "phase_tap" if kind in ("Ideal", "Symmetrical", "Tabular") and table == "trafo"
                       and _missing_ratio(df, i) else "tap")
            p.add("SV", "SvTapStep", cw._id(tap, "sv"), [("cim:SvTapStep.position", float(pos)),
                                                         ("cim:SvTapStep.TapChanger", f"#{tap}")])
    header = cw._id(network_id, "solved", description or created)
    return p.xml("SV", header, [cw._id(network_id, "model", k) for k in ("TP", "SSH")],
                 description or "Solved state", created)


def _missing_ratio(df, i) -> bool:
    step = df.at[i, "tap_step_percent"] if "tap_step_percent" in df.columns else None
    return step is None or pd.isna(step) or float(step) == 0.0
