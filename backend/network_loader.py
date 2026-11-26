#!/usr/bin/env python
"""network_loader.py — Generic pandapower network loader and on-the-fly Ybus builder.

Public API
----------
load_profile(profile_name)     -> dict
    Read backend/network_profiles/{profile_name}.yaml and return as dict.

load_network(profile)          -> pp.pandapowerNet
    Load a pandapower network from a profile dict.

build_ybus_full(net)           -> db_full (an outage leaves its branch out)
    Compute the admittance database on the fly from pandapower's internal
    pypower representation (one database: an outage is studied by leaving its
    branch out). Format is identical to the legacy pre-computed pickle files.

Internal helpers
----------------
_build_ybus_entry(net)         -> dict
    Build one admittance dict from a *solved* pandapower network.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pandapower as pp
import yaml

# ---------------------------------------------------------------------------
# Profile loading
# ---------------------------------------------------------------------------

def load_profile(profile_name: str, backend_dir: Optional[str] = None) -> dict:
    """Read ``network_profiles/{profile_name}.yaml`` and return as dict.

    Parameters
    ----------
    profile_name : str
        Name of the profile (without .yaml extension), e.g. "pglib_case30", "cigre_mv".
    backend_dir : str, optional
        Absolute path to the backend/ directory.  Defaults to the directory
        containing this file.
    """
    base = Path(backend_dir or Path(__file__).parent)
    path = base / "network_profiles" / f"{profile_name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Profile not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Network loading
# ---------------------------------------------------------------------------

def load_network(profile: dict, backend_dir: Optional[str] = None) -> pp.pandapowerNet:
    """Load a pandapower network from a profile dict.

    Supported ``profile["source"]`` values:
    - ``"excel"``     → pp.from_excel(excel_path)
    - ``"matpower"``  → pandapower MATPOWER converter
    - ``"ieee"``      → pp.networks.case{case}()  (e.g. case14, case30, case118)
    - ``"pandapower"`` → pp.networks.<case>(**case_args) — any of pandapower's
                         bundled networks, e.g. ``create_cigre_network_mv``
                         with ``case_args: {with_der: all}``
    - ``"cgmes"``     → read_cgmes(cgmes_path): a CIM model (the model is kept too)
    - ``"json"``      → pp.from_json(json_path)
    - ``"pickle"``    → pp.from_pickle(pickle_path)
    """
    base = Path(backend_dir or Path(__file__).parent)
    source = profile.get("source", "").lower()

    if source == "excel":
        path = base.parent / profile["excel_path"]
        return pp.from_excel(str(path))

    if source == "matpower":
        return read_matpower(str(base.parent / profile["matpower_path"]))

    if source in ("ieee", "pandapower"):
        case = profile.get("case", "case14")
        loader = getattr(pp.networks, case, None)
        if loader is None:
            raise ValueError(f"Unknown pp.networks case: {case}")
        return loader(**(profile.get("case_args") or {}))

    if source == "cgmes":
        # A CIM model (a CGMES zip, or a zip of a model's and its boundary's zips), read as an upload is.
        return read_cgmes(str(base.parent / profile["cgmes_path"]))

    if source == "json":
        path = base.parent / profile["json_path"]
        return pp.from_json(str(path))

    if source == "pickle":
        path = base.parent / profile["pickle_path"]
        return pp.from_pickle(str(path))

    raise ValueError(f"Unsupported network source: '{source}'")


# ---------------------------------------------------------------------------
# MATPOWER gen → sgen conversion
# ---------------------------------------------------------------------------

def apply_voltage_setpoints(net: pp.pandapowerNet, profile: dict) -> dict:
    """Set the external-grid and generator voltage setpoints a profile declares.

    ``ext_grid_vm_pu`` sets every external grid; ``gen_vm_pu`` maps a bus index
    to the setpoint of the voltage-controlled generators on it. A MATPOWER OPF
    benchmark such as PGLib ships placeholder setpoints (1.0 everywhere); held
    in a power flow they demand more reactive power than its condensers have,
    so a profile states setpoints its case can actually hold. Returns what was
    applied, for the startup log.
    """
    applied = {}
    if profile.get("ext_grid_vm_pu") is not None and len(net.ext_grid):
        net.ext_grid["vm_pu"] = float(profile["ext_grid_vm_pu"])
        applied["ext_grid"] = float(profile["ext_grid_vm_pu"])
    for bus, vm in (profile.get("gen_vm_pu") or {}).items():
        rows = net.gen.index[net.gen["bus"] == int(bus)]
        if len(rows):
            net.gen.loc[rows, "vm_pu"] = float(vm)
            applied[f"gen@bus{int(bus)}"] = float(vm)
    return applied


def _move_cost(net, et: str, idx, new_et: str, new_idx) -> None:
    """Re-point the case's cost row of element (et, idx) to (new_et, new_idx)."""
    if "poly_cost" not in net or not len(net.poly_cost):
        return
    rows = (net.poly_cost["et"] == et) & (net.poly_cost["element"] == idx)
    net.poly_cost.loc[rows, "et"] = new_et
    net.poly_cost.loc[rows, "element"] = new_idx


class CgmesError(ValueError):
    """A CGMES upload that cannot be used; the message says why and what to add."""


def _name_switch_nodes(net) -> None:
    """pandapower's CIM converter adds a node at the far side of each switch
    without a name of its own ("be-sw nan"): every such node then shares one
    name, and a substation named after one of them merges with others. Each is
    named after the bus its switch joins it to, and takes that bus's substation
    when it has none."""
    names = net.bus["name"]
    unnamed = [int(i) for i, v in names.items()
               if pd.isna(v) or str(v).strip().casefold() in ("", "nan", "none", "be-sw nan")]
    if not unnamed:
        return
    taken = {str(v) for v in names if pd.notna(v)}
    sw = net.switch[net.switch.et == "b"] if len(getattr(net, "switch", [])) else net.switch
    for b in unnamed:
        joined = sw[(sw.bus == b) | (sw.element == b)]
        other = None
        if len(joined):
            row = joined.iloc[0]
            other = int(row.element) if int(row.bus) == b else int(row.bus)
        base = f"{names.at[other]} node" if other is not None and pd.notna(names.at[other]) else f"Node {b}"
        name, k = base, 2
        while name in taken:
            name, k = f"{base} {k}", k + 1
        taken.add(name)
        net.bus.at[b, "name"] = name
        if (other is not None and "Substation_id" in net.bus.columns
                and pd.isna(net.bus.at[b, "Substation_id"]) and pd.notna(net.bus.at[other, "Substation_id"])):
            net.bus.at[b, "Substation_id"] = net.bus.at[other, "Substation_id"]


def read_cgmes(path: str, partial: bool = False) -> pp.pandapowerNet:
    """A CGMES model (CIM 2.4.15 or 3.0) from a zip of its profiles — EQ, SSH,
    TP, SV, and the boundary set when the model has one — or a zip of such
    zips (a model and its boundary, as ENTSO-E publishes them). Converted by
    pandapower's CIM converter, which keeps each element's mRID (`origin_id`)
    and class (`origin_class`), and each bus's Substation (`Substation_id`).

    Refused, with what to add, when the profiles leave lines without a node at
    one end — a model whose boundary set is missing — or carry no equipment."""
    import logging
    import tempfile
    import warnings
    import zipfile

    from pandapower.converter.cim.cim2pp.from_cim import from_cim

    with tempfile.TemporaryDirectory() as scratch:
        files, xml_heads = [], []
        with zipfile.ZipFile(path) as outer:
            inner = [n for n in outer.namelist() if n.lower().endswith(".zip")]
            if inner:
                for n in inner:
                    target = os.path.join(scratch, os.path.basename(n))
                    with open(target, "wb") as fh:
                        fh.write(outer.read(n))
                    files.append(target)
                    with zipfile.ZipFile(target) as z:
                        xml_heads += [z.read(m)[:4000] for m in z.namelist() if m.lower().endswith(".xml")]
            else:
                files.append(path)
                xml_heads += [outer.read(m)[:4000] for m in outer.namelist() if m.lower().endswith(".xml")]
        if not xml_heads or not any(b"TC57/CIM" in h or b"iec.ch" in h for h in xml_heads):
            raise CgmesError("This zip holds no CIM/CGMES profiles (XML files in the IEC CIM namespace).")
        version = "3.0" if any(b"CIM100" in h for h in xml_heads) else "2.4.15"
        logging.disable(logging.CRITICAL)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                net = (_from_cim_v3(files, path) if version == "3.0"
                       else from_cim(file_list=files, cgmes_version=version))
        finally:
            logging.disable(logging.NOTSET)
    if not len(net.bus) and not partial:
        raise CgmesError("The CGMES profiles describe no topology: include the TP (or EQ with connectivity) profile.")
    equipment = sum(len(net[t]) for t in ("line", "trafo", "trafo3w", "load", "gen", "sgen", "ext_grid") if t in net)
    if (not equipment or net.bus["vn_kv"].isna().all()) and not partial:
        raise CgmesError("The CGMES profiles describe no equipment (no lines, transformers, loads, units or "
                         "voltages): include the EQ profile.")
    # The operating point — what each load draws and each unit produces — is the SSH profile's. Without
    # it the converter leaves those values empty, and the synthetic data and every study built on them
    # would be empty too.
    missing = [what for tbl, what in (("load", "loads' demand"), ("gen", "units' output"), ("sgen", "units' output"))
               if tbl in net and len(net[tbl]) and pd.to_numeric(net[tbl]["p_mw"], errors="coerce").isna().all()]
    if missing:
        raise CgmesError(f"The CGMES profiles give no operating point (the {' and the '.join(dict.fromkeys(missing))} "
                         "are missing): include the SSH profile.")
    _name_switch_nodes(net)
    dangling = []
    for tbl, a, b in (("line", "from_bus", "to_bus"), ("trafo", "hv_bus", "lv_bus")):
        bad = net[tbl][net[tbl][a].isna() | net[tbl][b].isna()]
        dangling += [str(n) for n in bad.get("name", bad.index)]
    if dangling:
        raise CgmesError(f"{len(dangling)} branch(es) end in nodes the model does not contain ("
                         + ", ".join(dangling[:5]) + ("…" if len(dangling) > 5 else "")
                         + "): the boundary set (EQ_BD, TP_BD) is probably missing. Add it to the zip.")
    # CONDUCTOR studies the network from a supply point. A model whose reference is a generator (CGMES
    # `referencePriority`, pandapower's `slack`) gets it as the external grid, as pandapower's MATPOWER
    # converter makes the reference bus's unit one; it keeps the unit's mRID.
    net["_import_notes"] = list(net.get("_v3_notes") or [])
    if not len(net.ext_grid) and len(net.gen) and not partial:
        slack = net.gen[net.gen["slack"].fillna(False).astype(bool)] if "slack" in net.gen.columns else net.gen.iloc[0:0]
        biggest = (net.gen["max_p_mw"].fillna(0).idxmax() if "max_p_mw" in net.gen.columns
                   and net.gen["max_p_mw"].notna().any() else net.gen.index[0])
        ref = slack.index[0] if len(slack) else biggest
        row = net.gen.loc[ref]
        vm = row.get("vm_pu")                              # NaN is truthy: `or 1.0` would keep it
        idx = pp.create_ext_grid(net, bus=int(row["bus"]), vm_pu=float(vm) if pd.notna(vm) and vm else 1.0,
                                 name=f"{row.get('name') or 'Reference unit'} (reference)")
        if "origin_id" in net.gen.columns:
            net.ext_grid.at[idx, "origin_id"] = row.get("origin_id")
        net.gen = net.gen.drop(index=ref)
        net["_import_notes"].append(f"The model has no external grid: its reference unit {row.get('name')!r} "
                                    "is the supply point (the external grid) in CONDUCTOR.")
    # An in-service flag pandapower's converter could not decide (CGMES 3.0 combines a motor's with its generating
    # unit's, which a motor does not have) is in service: the model states nothing else, and the power flow needs a
    # true or false.
    for table in ("bus", "line", "trafo", "trafo3w", "gen", "sgen", "load", "motor", "shunt", "ward", "xward",
                  "impedance", "ext_grid", "storage", "svc"):
        df = net[table] if table in net else None
        if isinstance(df, pd.DataFrame) and len(df) and "in_service" in df.columns and df["in_service"].dtype != bool:
            net[table]["in_service"] = [True if pd.isna(v) else bool(v) for v in df["in_service"]]
    # A boundary node can come with no nominal voltage (pandapower's converter reads none for some CGMES 3.0 boundary
    # sets): a line or a closed bus–bus switch joins two buses at one voltage, so it takes its neighbour's.
    if net.bus["vn_kv"].isna().any():
        filled = 0
        for _round in range(len(net.bus)):
            known = net.bus["vn_kv"]
            pairs = [(int(a), int(b)) for a, b in zip(net.line.from_bus, net.line.to_bus)]
            if len(getattr(net, "switch", [])):
                bb = net.switch[(net.switch.et == "b")]
                pairs += [(int(a), int(b)) for a, b in zip(bb.bus, bb.element)]
            step = 0
            for a, b in pairs:
                for x, y in ((a, b), (b, a)):
                    if pd.isna(known.get(x)) and pd.notna(known.get(y)):
                        net.bus.at[x, "vn_kv"] = known[y]
                        step += 1
            filled += step
            if not step:
                break
        if filled:
            net["_import_notes"].append(f"{filled} node(s) with no nominal voltage in the model (boundary nodes) take "
                                        "the voltage of the node a line joins them to.")
    # A static var compensator in reactive-power control has no voltage set point, which pandapower's converter takes
    # as the shunt's rated voltage: left empty, the power flow cannot build the network. It takes its bus's voltage.
    if len(net.shunt) and net.shunt["vn_kv"].isna().any():
        missing = net.shunt["vn_kv"].isna()
        net.shunt.loc[missing, "vn_kv"] = net.bus.loc[net.shunt.loc[missing, "bus"], "vn_kv"].to_numpy()
        net["_import_notes"].append(f"{int(missing.sum())} shunt(s) with no rated voltage in the model (a static var "
                                    "compensator in reactive-power control) take their bus's nominal voltage.")
    # A transformer with a ratio tap changer on one end and a linear phase tap changer on the other: pandapower's
    # converter gives each tap the other's step as well (the ratio tap an angle step, the phase tap a voltage step),
    # so every ratio step also turns the angle. Each keeps its own.
    cols = set(net.trafo.columns) if len(net.trafo) else set()
    if {"tapchanger_class", "tapchanger2_class"} <= cols:
        first = net.trafo["tapchanger_class"].astype(str)
        second = net.trafo["tapchanger2_class"].astype(str)
        fixed = 0
        for ratio, phase, phase_tap, ratio_tap in ((first, second, "tap2", "tap"), (second, first, "tap", "tap2")):
            both = ratio.str.startswith("RatioTapChanger") & phase.str.startswith("PhaseTapChangerLinear")
            for col in (f"{ratio_tap}_step_degree", f"{phase_tap}_step_percent"):
                if col in cols and (both & net.trafo[col].notna()).any():
                    fixed += int((both & net.trafo[col].notna()).sum())
                    net.trafo.loc[both, col] = float("nan")
        if fixed:
            net["_import_notes"].append("Transformer(s) with a ratio and a phase tap changer: each tap keeps only its "
                                        "own step (pandapower's converter copies the other's to it).")
    _plain_dtypes(net)
    added = add_unconverted_switches(net, path)
    if added:
        classes = sorted({c for _, c in added})
        net["_import_notes"].append(
            f"{len(added)} switch(es) of a class pandapower's converter does not read ({', '.join(classes)}) "
            "were added as switches between their nodes, so the topology is the model's.")
    net.name = net.name or os.path.splitext(os.path.basename(path))[0]
    return net


# Equipment pandapower's CGMES 3.0 converter reads `inService` for from its EQ table, where its parser keeps no such
# column (CGMES 3.0 states Equipment.inService in SSH, and may leave it out: then the equipment is in service).
_V3_IN_SERVICE_FROM_EQ = ("ACLineSegment", "DCLineSegment", "EquivalentBranch", "SeriesCompensator")


def _from_cim_v3(files: list, source: str):
    """A CGMES 3.0 model through pandapower's converter, with what its 3.0 parser drops put back. pandapower 3.5's
    3.0 data structure has no inService for a line (CGMES 3.0 states it in SSH, and may leave it out — then the
    equipment is in service), which its converter then reads: without it pandapower cannot read a 3.0 model with
    lines at all (ENTSO-E's MicroGrid included). The structure is extended for this parse only, and the state read
    from the whole model. (A transformer's phaseAngleClock is the ShortCircuit profile's in 3.0: a model sent
    without it reads every vector group as 0°, said in the import notes.)"""
    import cim_model
    from pandapower.converter.cim import cim_classes
    from pandapower.converter.cim.cim2pp.from_cim import from_cim_dict

    parser = cim_classes.CimParser(cgmes_version="3.0")
    structure = parser.get_cim_data_structure

    def extended():
        s = {profile: dict(classes) for profile, classes in structure().items()}
        for cls in _V3_IN_SERVICE_FROM_EQ:
            if cls in s["eq"] and "inService" not in s["eq"][cls].columns:
                s["eq"][cls] = pd.DataFrame(columns=list(s["eq"][cls].columns) + ["inService"])
        return s
    parser.get_cim_data_structure = extended
    parser.parse_files(file_list=files, prepare_cim_net=True, set_data_types=True)
    model = cim_model.CimModel.read(source)
    for cls in _V3_IN_SERVICE_FROM_EQ:
        df = parser.cim["eq"].get(cls)
        if isinstance(df, pd.DataFrame) and len(df):
            df["inService"] = [str((model.get(rid) or {}).get("Equipment.inService", "true")).strip().lower() != "false"
                               for rid in df["rdfId"]]
    net = from_cim_dict(parser, cim_version="3.0",
                        custom_converter_classes={"energySourcesCim16": _energy_sources_and_inverters()})
    _drop_twice_converted(net)
    if len(net.trafo) + len(net.trafo3w) and not model.profiles.get("SC"):
        net["_v3_notes"] = ["The model has no ShortCircuit profile, where CGMES 3.0 states each transformer's vector "
                            "group (phaseAngleClock): every transformer is read with no phase shift."]
    return net


def _plain_dtypes(net) -> None:
    """pandas' nullable types (boolean, Int64, Float64, string) that pandapower's CIM converter leaves in the
    tables, as plain ones: pandapower's own `create_*` cannot append a row to a nullable boolean column (adding a
    transformer to a network loaded from CGMES failed). A flag with no value is False; a number with none is NaN."""
    for table in ("bus", "line", "trafo", "trafo3w", "load", "gen", "sgen", "shunt", "ext_grid", "switch", "ward",
                  "xward", "impedance", "motor", "storage", "svc", "dcline"):
        df = net[table] if table in net else None
        if not isinstance(df, pd.DataFrame) or not len(df.columns):
            continue
        for c in df.columns:
            kind = str(df[c].dtype)
            if c in ("reactive_capability_curve", "slack", "oltc", "power_station_unit", "tap_dependency_table") \
                    and kind in ("object", "boolean"):
                df[c] = df[c].where(df[c].notna(), False).astype(bool)     # a flag the model leaves unset: off
                continue
            if kind == "boolean":
                df[c] = df[c].fillna(False).astype(bool)
            elif kind in ("Int64", "Int32", "UInt32", "Float64", "Float32"):
                df[c] = df[c].astype("float64")
            elif kind == "string":
                df[c] = df[c].astype(object).where(df[c].notna(), None)


def _drop_twice_converted(net) -> None:
    """The guard on `_energy_sources_and_inverters`: should a later pandapower convert power-electronics connections
    for CGMES 3.0 itself, each would come out twice — the rows with one mRID are kept once (the first)."""
    if not len(net.sgen) or "origin_id" not in net.sgen.columns:
        return
    twice = net.sgen["origin_id"].notna() & net.sgen["origin_id"].duplicated(keep="first")
    if twice.any():
        net.sgen = net.sgen[~twice]
        net["_v3_notes"] = list(net.get("_v3_notes") or []) + [
            f"{int(twice.sum())} unit(s) converted twice by pandapower kept once."]


def _energy_sources_and_inverters():
    """pandapower 3.5 converts PowerElectronicsConnections (PV, wind, batteries behind an inverter) for the LTDS
    profile only: a CGMES 3.0 model would lose every one. Its converter already reads 3.0; this runs it after the
    energy sources, where the LTDS profile runs it."""
    from pandapower.converter.cim.cim2pp.converter_classes.generators.energySourcesCim16 import EnergySourceCim16
    from pandapower.converter.cim.cim2pp.converter_classes.generators.powerElectronicsConnection import (
        PowerElectronicsConnection)

    class EnergySourcesAndInverters(EnergySourceCim16):
        def convert_energy_sources_cim16(self):
            super().convert_energy_sources_cim16()
            PowerElectronicsConnection(cimConverter=self.cimConverter).convert_power_electronics_connection()
    return EnergySourcesAndInverters


# CIM switch classes pandapower's converter does not read (it reads Breaker, Disconnector, LoadBreakSwitch and
# Switch). Left out, a fuse in series leaves its two nodes unjoined: everything beyond it looks cut off.
_UNREAD_SWITCHES = ("Fuse", "Jumper", "GroundDisconnector", "Recloser", "Sectionaliser", "ProtectedSwitch",
                    "DisconnectingCircuitBreaker", "Cut")
_SWITCH_TYPE = {"Recloser": "CB", "ProtectedSwitch": "CB", "DisconnectingCircuitBreaker": "CB",
                "Sectionaliser": "LBS"}


def add_unconverted_switches(net, path: str) -> list[tuple[str, str]]:
    """Add to `net` each switch of the CGMES model at `path` whose class the converter does not read, as a
    bus–bus switch between the buses of its two terminals, open or closed as the model has it now (SSH), with its
    mRID and class (`origin_id`, `origin_class`). Returns [(name, class)] of those added. A switch whose terminals
    fall on one bus (a bus-branch model merges them) changes nothing and is not added."""
    import cim_model
    try:
        model = cim_model.CimModel.read(path)
    except Exception:  # noqa: BLE001 — the converter read it; without our own read, nothing is added
        return []
    unread = model.of(*_UNREAD_SWITCHES)
    if not unread or "origin_id" not in net.bus.columns:
        return []
    node_bus = {cim_model.norm(v): int(i) for i, v in net.bus["origin_id"].items() if isinstance(v, str)}
    added = []
    for sw in unread:
        buses = []
        for t in model.terminals(sw):
            node = model.ref(t, "Terminal.ConnectivityNode") or model.ref(t, "Terminal.TopologicalNode")
            if node in node_bus and node_bus[node] not in buses:
                buses.append(node_bus[node])
        if len(buses) != 2:
            continue
        o = model.get(sw) or {}
        state = o.get("Switch.open", o.get("Switch.normalOpen"))
        closed = str(state).strip().lower() != "true"
        cls = o.get("class")
        idx = pp.create_switch(net, bus=buses[0], element=buses[1], et="b", closed=closed,
                               type=_SWITCH_TYPE.get(cls, "LS"), name=model.name(sw) or sw)
        if "origin_id" in net.switch.columns:
            net.switch.at[idx, "origin_id"] = sw
        if "origin_class" in net.switch.columns:
            net.switch.at[idx, "origin_class"] = cls
        added.append((model.name(sw) or sw, cls))
    return added


def read_matpower(path: str) -> pp.pandapowerNet:
    """A MATPOWER case as a pandapower network, every generator a generator.

    pandapower makes a generator at a load (PQ) bus a static generator, with
    no name and the case's dispatch reactive power as if it were a power
    factor; the case's other generators stay generators. MATPOWER has no static
    generators, so each one pandapower created here is one of the case's
    generators: it is put back, and `gen_to_sgen` then treats every unit the
    same way (name, power factor 1.0, reactive limits kept). PGLib's
    Alsac–Stott 30-bus case has three units at load buses.
    """
    from pandapower.converter.matpower import from_mpc

    net = from_mpc(path, f_hz=50, validate_conversion=False)
    for idx in list(net.sgen.index):
        row = net.sgen.loc[idx]

        def value(col, default):
            v = row.get(col)
            return float(v) if v is not None and pd.notna(v) else default

        new = pp.create_gen(net, bus=int(row["bus"]), p_mw=value("p_mw", 0.0), vm_pu=1.0,
                            min_p_mw=value("min_p_mw", 0.0), max_p_mw=value("max_p_mw", value("p_mw", 0.0)),
                            min_q_mvar=value("min_q_mvar", np.nan), max_q_mvar=value("max_q_mvar", np.nan),
                            controllable=True, in_service=bool(row.get("in_service", True)))
        _move_cost(net, "sgen", idx, "gen", new)
        net.sgen.drop(index=idx, inplace=True)
    net.sgen.reset_index(drop=True, inplace=True)
    return net


def gen_to_sgen(net: pp.pandapowerNet) -> tuple:
    """Convert ``net.gen`` PV-bus generators to ``net.sgen`` for OPF/flexibility
    tool compatibility.  The slack bus generator (already represented by
    ``net.ext_grid``) is skipped so it is never duplicated.

    Returns
    -------
    net : pandapowerNet
        The same network object, modified in-place.
    created_names : list[str]
        Names of the sgen entries that were created (e.g. ``["Gen_bus1"]``).
    kept_condenser_names : list[str]
        Names of zero-real-power generators kept as net.gen PV buses for voltage
        regulation (e.g. ``["Gen_bus2", "Gen_bus5", "Gen_bus7"]``).
    """
    slack_buses = set(int(b) for b in net.ext_grid["bus"].tolist())
    created_names = []
    kept_condenser_names = []
    rows_to_drop = []

    for idx, row in net.gen.iterrows():
        bus_id = int(row["bus"])
        if bus_id in slack_buses:
            continue
        max_p = float(row["max_p_mw"]) if "max_p_mw" in row.index and not np.isnan(float(row["max_p_mw"])) else float(row["p_mw"])
        # Skip synchronous condensers (zero real-power capacity): keep them as
        # net.gen PV buses so they continue to regulate voltage.
        if max_p < 1.0:
            kept_condenser_names.append(f"Gen_bus{bus_id}")
            continue
        # Unique per unit: two generators on one bus both used to become
        # `Gen_bus{b}`, and every table keyed by name (capacities, OPF sets)
        # silently kept one of them.
        # A unit the network names (a CIM model's machine) keeps its name; one it does not is named by its bus.
        own = row.get("name") if "name" in row.index else None
        own = own.strip() if isinstance(own, str) and own.strip() else None
        stem = own or f"Gen_bus{bus_id}"
        name = stem
        taken = set(created_names) | set(net.sgen["name"].astype(str)) if len(net.sgen) else set(created_names)
        suffix = 2
        while name in taken:
            name = f"{stem} {suffix}" if own else f"Gen_bus{bus_id}_{suffix}"
            suffix += 1
        min_p = float(row["min_p_mw"]) if "min_p_mw" in row.index and not np.isnan(float(row["min_p_mw"])) else 0.0
        new = pp.create_sgen(
            net,
            bus=bus_id,
            p_mw=float(row["p_mw"]),
            q_mvar=0.0,
            name=name,
            max_p_mw=max_p,
            min_p_mw=min_p,
            controllable=True,
            in_service=bool(row.get("in_service", True)),
        )
        # Kept so the unit can be switched back to voltage control later
        # (generators.py): its set-point and reactive limits as loaded.
        for col, src in (("orig_vm_pu", "vm_pu"), ("orig_min_q_mvar", "min_q_mvar"),
                         ("orig_max_q_mvar", "max_q_mvar")):
            value = row.get(src)
            if col not in net.sgen.columns:
                net.sgen[col] = np.nan
            net.sgen.at[new, col] = float(value) if value is not None and pd.notna(value) else np.nan
        # A CIM model's machine keeps its identity (its mRID, class and terminal).
        for col in ("origin_id", "origin_class", "terminal"):
            value = row.get(col) if col in row.index else None
            if isinstance(value, str) and value:
                if col not in net.sgen.columns:
                    net.sgen[col] = None
                    net.sgen[col] = net.sgen[col].astype(object)
                net.sgen.at[new, col] = value
        rows_to_drop.append(idx)
        created_names.append(name)
        _move_cost(net, "gen", idx, "sgen", new)

    net.gen.drop(index=rows_to_drop, inplace=True)
    # The case's cost rows follow their unit: re-pointed above for converted
    # units, re-numbered here for the generators kept (costs.py reads them).
    renumber = {int(old): new for new, old in enumerate(net.gen.index)}
    net.gen.reset_index(drop=True, inplace=True)
    if "poly_cost" in net and len(net.poly_cost):
        kept = net.poly_cost["et"] == "gen"
        net.poly_cost.loc[kept, "element"] = net.poly_cost.loc[kept, "element"].map(
            lambda e: renumber.get(int(e), -1))
        net.poly_cost.drop(index=net.poly_cost.index[kept & (net.poly_cost["element"] < 0)], inplace=True)
    # Mark the network so assignment functions know to use the Generic path.
    net["_gen_to_sgen_applied"] = True
    return net, created_names, kept_condenser_names


def keep_own_units(net: pp.pandapowerNet) -> None:
    """Mark an uploaded network that brings its own units (`gen` or `sgen`
    rows) so every tick keeps them: the generic path, as after `gen_to_sgen`.

    Only a network with no unit at all is a measured-substation network, its
    units rebuilt from the loads at each tick. An upload of static generators
    only (common for distribution grids) took that path for lack of a `gen`
    table: its units were replaced by one per load, each injecting minus the
    whole bus's production.
    """
    if len(net.gen) or len(net.sgen):
        net["_gen_to_sgen_applied"] = True


# ---------------------------------------------------------------------------
# On-the-fly Ybus builder
# ---------------------------------------------------------------------------

def _build_ybus_entry(net: pp.pandapowerNet) -> dict:
    """Compute branch admittances from a **solved** pandapower network.

    Requires ``net._ppc`` to be populated (i.e. pp.runpp must have been called).
    Produces the dict format expected by ``flex_engine.optimization_model_base``:

        {
            "Yff_r": {((from_bus, to_bus), tap): float, ...},
            "Yff_i": {((from_bus, to_bus), tap): float, ...},
            "Yft_r": {((from_bus, to_bus), tap): float, ...},
            "Yft_i": {((from_bus, to_bus), tap): float, ...},
            "TAPS":            list,
            "trafo_ranges":    {trafo_idx: [min_tap, max_tap]},
            "trafo_defaults":  {trafo_idx: default_tap},
            "branch_to_trafo": {(from_bus, to_bus): trafo_idx},
        }

    For lines  : tap_key = 1  (matching flex_engine's ``base_t = 1`` convention)
    For trafos : tap_key = tap_neutral  (matching flex_engine's ``t_default`` lookup)
    """
    from pandapower.pypower.idx_brch import (F_BUS, T_BUS, BR_R, BR_X, BR_B, BR_G, BR_R_ASYM, BR_X_ASYM,
                                             BR_G_ASYM, BR_B_ASYM, TAP, SHIFT, BR_STATUS)

    ppc = net._ppc

    # Build pandapower bus index ↔ ppc bus index mapping
    bus_lookup = net._pd2ppc_lookups["bus"]  # array: pp_idx → ppc_idx
    pp2ppc = {int(idx): int(bus_lookup[idx]) for idx in net.bus.index}
    ppc2pp = {v: k for k, v in pp2ppc.items()}

    Yff_r: dict = {}
    Yff_i: dict = {}
    Yft_r: dict = {}
    Yft_i: dict = {}
    branch_to_trafo: dict = {}
    trafo_defaults:  dict = {}
    trafo_ranges:    dict = {}

    # Pre-build trafo lookup: (hv_bus_pp, lv_bus_pp) → trafo_idx
    trafo_by_buses: dict = {}
    for tidx, t in net.trafo.iterrows():
        hv, lv = int(t["hv_bus"]), int(t["lv_bus"])
        trafo_by_buses[(hv, lv)] = tidx
        trafo_by_buses[(lv, hv)] = tidx

    for k in range(len(ppc["branch"])):
        fb_ppc = int(ppc["branch"][k, F_BUS])
        tb_ppc = int(ppc["branch"][k, T_BUS])

        if fb_ppc not in ppc2pp or tb_ppc not in ppc2pp:
            continue  # disconnected / out-of-service bus
        if float(ppc["branch"][k, BR_STATUS]) == 0.0:
            continue  # out-of-service branch: no admittance (code review 2.9)

        row = ppc["branch"][k]
        r, x = float(row[BR_R]), float(row[BR_X])
        tap      = float(row[TAP])
        shift_deg = float(row[SHIFT])

        if tap == 0.0:
            tap = 1.0

        i = ppc2pp[fb_ppc]   # pandapower from_bus
        j = ppc2pp[tb_ppc]   # pandapower to_bus

        # The π-model as pandapower's own `makeYbus.branch_vectors` builds it:
        # the shunt admittance is conductance + susceptance (a transformer's
        # iron losses, a line's conductance), and the to-end may carry the
        # asymmetric terms. Leaving the conductance out cost ~0.7 kW per end on a
        # typical distribution transformer.
        def _series(rr, xx):
            return (1.0 / complex(rr, xx)) if (rr != 0.0 or xx != 0.0) else 0j  # zero-impedance branch
        ys_f = _series(r, x)
        ys_t = _series(r + float(row[BR_R_ASYM]), x + float(row[BR_X_ASYM]))
        bc_f = complex(float(row[BR_G]), float(row[BR_B])) / 2
        bc_t = complex(float(row[BR_G]) + float(row[BR_G_ASYM]), float(row[BR_B]) + float(row[BR_B_ASYM])) / 2

        # Tap as complex phasor (real tap × phase shift)
        tap_complex = tap * np.exp(1j * shift_deg * np.pi / 180.0)

        # Pi-model branch admittance elements
        Yff = (ys_f + bc_f) / abs(tap_complex) ** 2   # from-end self
        Yft = -ys_f / np.conj(tap_complex)             # from→to mutual
        Ytf = -ys_t / tap_complex                      # to→from mutual
        Ytt = ys_t + bc_t                              # to-end self

        # Determine tap key and register trafo metadata
        is_trafo = (i, j) in trafo_by_buses or (j, i) in trafo_by_buses
        if is_trafo:
            tidx = trafo_by_buses.get((i, j), trafo_by_buses.get((j, i)))
            _raw = (
                net.trafo.at[tidx, "tap_neutral"]
                if "tap_neutral" in net.trafo.columns
                else None
            )
            try:
                tap_neutral = int(_raw) if _raw is not None and _raw == _raw else 0
            except (TypeError, ValueError):
                tap_neutral = 0
            branch_to_trafo[(i, j)] = tidx
            branch_to_trafo[(j, i)] = tidx
            if tidx not in trafo_defaults:
                trafo_defaults[tidx] = tap_neutral
                trafo_ranges[tidx]   = [tap_neutral, tap_neutral]
            tap_key = tap_neutral
        else:
            tap_key = 1  # lines always use tap=1 key

        # Store both directions. Added, not assigned: parallel branches between
        # the same buses are one bus pair here, and in a pi-model their
        # admittances sum — assigning kept only the last circuit read
        # (sandbox hazard A.3 (2)).
        for table, key, val in (
            (Yff_r, (i, j), Yff.real), (Yff_i, (i, j), Yff.imag),
            (Yft_r, (i, j), Yft.real), (Yft_i, (i, j), Yft.imag),
            (Yff_r, (j, i), Ytt.real), (Yff_i, (j, i), Ytt.imag),
            (Yft_r, (j, i), Ytf.real), (Yft_i, (j, i), Ytf.imag),
        ):
            table[key, tap_key] = table.get((key, tap_key), 0.0) + float(val)

    return {
        "Yff_r":         Yff_r,
        "Yff_i":         Yff_i,
        "Yft_r":         Yft_r,
        "Yft_i":         Yft_i,
        "TAPS":          list(set(trafo_defaults.values())) if trafo_defaults else [1],
        "trafo_ranges":  trafo_ranges,
        "trafo_defaults": trafo_defaults,
        "branch_to_trafo": branch_to_trafo,
    }


def _try_runpp(net: pp.pandapowerNet) -> bool:
    """Run power flow; return True on success, False on any failure."""
    try:
        pp.runpp(net, numba=False, calculate_voltage_angles=True)
        return True
    except pp.powerflow.LoadflowNotConverged:
        return False
    except Exception:
        return False


def build_ybus_full(net: pp.pandapowerNet) -> dict:
    """The admittance database of the network as loaded (one power flow).

    It used to come with one database per line and transformer outage, built
    by a power flow per branch at startup. The OPF models the network branch
    by branch and leaves out-of-service branches out, so an outage is exactly
    this database without the outaged branch; the per-outage copies were
    identical to it and are no longer built.

    Raises RuntimeError if the base-case power flow does not converge.
    """
    net_base = copy.deepcopy(net)
    if not _try_runpp(net_base):
        raise RuntimeError(
            "build_ybus_full: base-case power flow did not converge. "
            "Check network data (connectivity, voltage setpoints)."
        )
    return _build_ybus_entry(net_base)
