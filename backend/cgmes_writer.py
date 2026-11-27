"""
cgmes_writer.py — any network as a CGMES model, so every network has one.

A network loaded as CGMES has its CIM model already (cim_model.py keeps it whole). Any other — a MATPOWER case,
a pandapower file, the bundled demos — gets one here, written from its pandapower tables as CGMES 2.4.15 profiles:
EQ (equipment), SSH (the operating point) and TP (the topology, bus-branch). The model is then the network's in the
format system operators exchange, readable by pandapower's own CIM converter and by other tools (PowSyBl,
PowerFactory).

Each element keeps the mRID the CIM layer gives it (semantics._row_mrid), so the CIM layer, a study's results and
the model name the same element alike. What CGMES has no place for but CONDUCTOR's computation needs (generation
costs, OPF limits, how loads follow the voltage, load and unit scaling, storage, the network's frequency) goes beside
the profiles, by mRID (`extras`), for the adapter (cim_adapter.py) to put back; what neither can carry is listed in
`losses`, never dropped silently.

The parameters are written as pandapower's CIM converter reads them back (lines in ohm and siemens over their length;
a transformer's impedance on its high-voltage end; a shunt's susceptance per section), so that pandapower → CGMES →
pandapower gives the same power flow (tests/test_cim_adapter.py checks it on every bundled network).
"""

from __future__ import annotations

import copy
import io
import json
import math
import uuid
import xml.etree.ElementTree as ET
import zipfile

import pandas as pd

CIM16 = "http://iec.ch/TC57/2013/CIM-schema-cim16#"           # CGMES 2.4.15
CIM100 = "http://iec.ch/TC57/CIM100#"                           # CGMES 3.0
CIM = CIM16                                                     # kept for callers that name the 2.4.15 namespace
ENTSOE = "http://entsoe.eu/CIM/SchemaExtension/3/1#"            # 2.4.15's European extension
EU = "http://iec.ch/TC57/CIM100-European#"                      # 3.0's
MD = "http://iec.ch/TC57/61970-552/ModelDescription/1#"
RDF = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
VERSIONS = {
    "3.0": {"cim": CIM100, "ext": EU, "ext_prefix": "eu",
            "profiles": {"EQ": "http://iec.ch/TC57/ns/CIM/CoreEquipment-EU/3.0",
                         "SSH": "http://iec.ch/TC57/ns/CIM/SteadyStateHypothesis-EU/3.0",
                         "TP": "http://iec.ch/TC57/ns/CIM/Topology-EU/3.0",
                         "SC": "http://iec.ch/TC57/ns/CIM/ShortCircuit-EU/3.0",
                         "SV": "http://iec.ch/TC57/ns/CIM/StateVariables-EU/3.0"}},
    "2.4.15": {"cim": CIM16, "ext": ENTSOE, "ext_prefix": "entsoe",
               "profiles": {"EQ": "http://entsoe.eu/CIM/EquipmentCore/3/1",
                            "SSH": "http://entsoe.eu/CIM/SteadyStateHypothesis/1/1",
                            "TP": "http://entsoe.eu/CIM/Topology/4/1",
                            "SV": "http://entsoe.eu/CIM/StateVariables/4/1"}}}
DEFAULT_VERSION = "3.0"
# Bumped whenever what the writer states for the same network changes: a difference saved against an older writer's
# statements is made again from its source (the edits, the operator's changes) rather than reused.
WRITER_VERSION = "2026-11-26"
# CGMES 3.0 states these in its ShortCircuit profile, not in Equipment: a transformer end's vector group (its
# phaseAngleClock) and grounding, and the zero-sequence values.
_SC_FIELDS = {"PowerTransformerEnd": ("phaseAngleClock", "grounded", "r0", "x0", "b0", "g0"),
              "ACLineSegment": ("r0", "x0", "b0ch", "g0ch")}
PROFILES = VERSIONS["2.4.15"]["profiles"]
_NS = uuid.UUID("2c5b6f0e-3a7d-5e21-9b44-7d1f0c2a9e31")    # derived identifiers of the parts the tables do not have

# pandapower's switch types as CIM's Switch subclasses.
_SWITCH_CLASS = {"CB": "Breaker", "DS": "Disconnector", "LBS": "LoadBreakSwitch", "LS": "Switch"}
# A static generator's type as the EnergySchedulingType name pandapower's converter maps back.
_SGEN_TYPE = {"WP": "WP", "PV": "PV", "BIOGAS": "BioGas", "CHP": "CHP", "OTHERRES": "OtherRES"}


# CONDUCTOR's own columns on pandapower's tables (sandbox.py's marks and profiles, overrides, names as loaded).
CONDUCTOR_COLUMNS = ("proposed", "profile", "profile_spec", "override", "substation_name", "orig_vm_pu",
                     "orig_min_q_mvar", "orig_max_q_mvar", "q_pf", "q_absorbing", "q_set_mvar", "loaded_name",
                     # a proposal's scaling and moves: how the time series treat the element on later ticks
                     "scale_factor", "bus_measured", "bus_moved", "_unscaled_p_mw", "_unscaled_q_mvar",
                     "_q_from_meas",
                     # a unit's identity as loaded, when a generator model moved it to the other table
                     "loaded_table", "loaded_index", "loaded_unit")
_ELEMENT_TABLES = ("bus", "line", "trafo", "trafo3w", "gen", "sgen", "storage", "load", "shunt", "ext_grid", "switch",
                   "ward", "xward", "impedance", "motor", "svc", "dcline")


def _missing(v) -> bool:
    try:
        return v is None or (isinstance(v, float) and math.isnan(v))
    except TypeError:
        return False


def _id(*parts) -> str:
    return str(uuid.uuid5(_NS, "|".join(str(p) for p in parts)))


def _num(value, default=0.0) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    return default if math.isnan(v) else v


def _text(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        return repr(float(f"{v:.15g}"))          # 15 significant digits: a small susceptance keeps its precision
    return str(v)


class _Profiles:
    """The three profiles' objects, written in the order added, for one CGMES version."""

    def __init__(self, version: str = DEFAULT_VERSION):
        self.version = version
        self.spec = VERSIONS[version]
        self.objects = {p: [] for p in self.spec["profiles"]}
        self.in_service: dict[str, bool] = {}     # equipment -> in service (3.0 states it, Equipment.inService)

    def add(self, profile: str, cls: str, oid: str, props: list, about: bool = False) -> str:
        """`props`: (property, value) — a value "#id" is a reference; "cim:Enum.value" an enumeration."""
        props = [(k, v) for k, v in props if v is not None]
        if self.version == "3.0" and profile == "EQ" and cls in _SC_FIELDS:
            short = [(k, v) for k, v in props if k.rsplit(".", 1)[-1] in _SC_FIELDS[cls]]
            props = [(k, v) for k, v in props if k.rsplit(".", 1)[-1] not in _SC_FIELDS[cls]]
            if short:
                self.objects["SC"].append((cls, oid, short, True))
        if self.version == "3.0" and not about and profile in ("EQ", "TP"):
            props.insert(1 if props and props[0][0] == "cim:IdentifiedObject.name" else 0,
                         ("cim:IdentifiedObject.mRID", oid))          # CGMES 3.0 states each object's mRID
        self.objects[profile].append((cls, oid, props, about))
        return oid

    def service(self, equipment: str, on: bool) -> None:
        """Whether an equipment is in service: CGMES 3.0 says so in SSH (pandapower's converter reads it with each
        terminal's `connected`); 2.4.15 only through the terminals."""
        self.in_service[equipment] = bool(on)

    def _ns(self, prefixed: str) -> str:
        return self.spec["ext"] if prefixed.split(":", 1)[0] in ("entsoe", "eu") else self.spec["cim"]

    def xml(self, profile: str, header_id: str, depends_on: list[str], description: str, created: str) -> bytes:
        ET.register_namespace("cim", self.spec["cim"])
        ET.register_namespace(self.spec["ext_prefix"], self.spec["ext"])
        ET.register_namespace("md", MD)
        ET.register_namespace("rdf", RDF)
        objects = list(self.objects[profile])
        if profile == "SSH" and self.version == "3.0":
            # Each equipment's inService, beside what SSH already says of it.
            classes = {oid: cls for cls, oid, _p, about in self.objects["EQ"] if not about}
            stated = set()
            for k, (cls, oid, props, about) in enumerate(objects):
                if oid in self.in_service and oid not in stated:
                    objects[k] = (cls, oid, props + [("cim:Equipment.inService", self.in_service[oid])], about)
                    stated.add(oid)
            objects += [(classes[oid], oid, [("cim:Equipment.inService", on)], True)
                        for oid, on in self.in_service.items() if oid not in stated and oid in classes]
        root = ET.Element(f"{{{RDF}}}RDF")
        head = ET.SubElement(root, f"{{{MD}}}FullModel", {f"{{{RDF}}}about": f"urn:uuid:{header_id}"})
        ET.SubElement(head, f"{{{MD}}}Model.created").text = created
        ET.SubElement(head, f"{{{MD}}}Model.scenarioTime").text = created
        ET.SubElement(head, f"{{{MD}}}Model.description").text = description
        ET.SubElement(head, f"{{{MD}}}Model.version").text = "1"
        ET.SubElement(head, f"{{{MD}}}Model.profile").text = self.spec["profiles"][profile]
        ET.SubElement(head, f"{{{MD}}}Model.modelingAuthoritySet").text = "urn:conductor"
        for dep in depends_on:
            ET.SubElement(head, f"{{{MD}}}Model.DependentOn", {f"{{{RDF}}}resource": f"urn:uuid:{dep}"})
        for cls, oid, props, about in objects:
            el = ET.SubElement(root, f"{{{self._ns(cls)}}}{cls.split(':')[-1]}",
                               {f"{{{RDF}}}about" if about else f"{{{RDF}}}ID": f"#_{oid}" if about else f"_{oid}"})
            for prop, value in props:
                child = ET.SubElement(el, f"{{{self._ns(prop)}}}{prop.split(':')[-1]}")
                if isinstance(value, str) and value.startswith("#"):
                    child.set(f"{{{RDF}}}resource", f"#_{value[1:]}")
                elif isinstance(value, str) and value.startswith(("cim:", "entsoe:", "eu:")):
                    child.set(f"{{{RDF}}}resource", self._ns(value) + value.split(":", 1)[1])
                else:
                    child.text = _text(value)
        ET.indent(root)
        return b'<?xml version="1.0" encoding="utf-8"?>\n' + ET.tostring(root, encoding="utf-8")


def _only(net, only: dict) -> tuple[object, dict]:
    """A shallow copy of `net` with only the rows `only` names ({table: indices}) and what writing them needs: the
    buses they connect, and the switches at a kept line's or transformer's end (their nodes are the element's ends).
    Returns (the copy, {bus: substation name} of the whole network — the containers stay those of a full write)."""
    import semantics as sm
    subs, _basis = sm.substations(net)
    part = copy.copy(net)
    keep = {t: set(int(i) for i in only.get(t, ())) for t in _ELEMENT_TABLES}
    sw = net.switch if len(getattr(net, "switch", [])) else pd.DataFrame()
    for i in sw.index:
        et, element = str(sw.at[i, "et"]), int(sw.at[i, "element"])
        if (et == "l" and element in keep["line"]) or (et == "t" and element in keep["trafo"]):
            keep["switch"].add(int(i))
    buses = set(keep["bus"])
    for table in _ELEMENT_TABLES:
        df = getattr(net, table, None) if table in net else None
        if not isinstance(df, pd.DataFrame) or table == "bus":
            continue
        rows = df.loc[[i for i in df.index if int(i) in keep[table]]]
        part[table] = rows
        for c in ("bus", "from_bus", "to_bus", "hv_bus", "mv_bus", "lv_bus"):
            if c in rows.columns:
                buses |= {int(b) for b in rows[c]}
        if table == "switch":
            buses |= {int(e) for e, et in zip(rows["element"], rows["et"]) if str(et) == "b"}
    part["bus"] = net.bus.loc[[i for i in net.bus.index if int(i) in buses]]
    return part, subs


def write(net, network_id: str = "local", name: str | None = None, created: str = "2000-01-01T00:00:00Z",
          version: str = DEFAULT_VERSION, only: dict | None = None) -> dict:
    """The network as CGMES 2.4.15: {"EQ": bytes, "SSH": bytes, "TP": bytes, "extras": dict, "losses": [str]}.
    Elements carry the CIM layer's mRIDs (`network_id`: the network's identity, as for semantics.build).
    `only` ({table: indices}): write just those elements (with the nodes and containers they need), as a full write
    states them — what an edit re-emits, at a cost in proportion to what it touches."""
    import semantics as sm

    given_subs = None
    if only is not None:
        net, given_subs = _only(net, only)
    p = _Profiles(version)
    v30 = version == "3.0"
    extras: dict = {"f_hz": _num(getattr(net, "f_hz", 50.0), 50.0), "sn_mva": _num(getattr(net, "sn_mva", 1.0), 1.0),
                    "name": name or getattr(net, "name", "") or "network", "elements": {}}
    losses: list[str] = []

    mrids: dict[str, dict] = {}

    def mrid(table, idx) -> str:
        if table not in mrids:
            mrids[table] = sm.row_mrids(net, table, network_id)
        return mrids[table].get(int(idx)) or sm._row_mrid(net, table, idx, network_id)

    def keep(table, idx, **fields):
        """What the adapter puts back on this element after reading the model (by mRID). `unset`: columns the
        network leaves empty where the model must state something (read back empty)."""
        clean = {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in fields.items()
                 if v is not None and not (isinstance(v, float) and math.isnan(v))}
        if clean:
            kept = extras["elements"].setdefault(mrid(table, idx), {"table": table, "fields": {}})["fields"]
            if "unset" in clean and "unset" in kept:
                clean["unset"] = list(dict.fromkeys(kept["unset"] + list(clean["unset"])))
            kept.update(clean)

    def label(table, idx) -> str:
        v = net[table].at[idx, "name"] if "name" in net[table].columns else None
        if not isinstance(v, str) and not _missing(v):
            v = str(int(v)) if isinstance(v, float) and v == int(v) else str(v)     # a case that numbers its buses
        return str(v).strip() if isinstance(v, str) and v.strip() else f"{table} {int(idx)}"

    # -- containers and base voltages ------------------------------------------------------------------------------
    region, sub_region = _id(network_id, "region"), _id(network_id, "sub_region")
    p.add("EQ", "GeographicalRegion", region, [("cim:IdentifiedObject.name", "CONDUCTOR")])
    p.add("EQ", "SubGeographicalRegion", sub_region, [("cim:IdentifiedObject.name", extras["name"]),
                                                      ("cim:SubGeographicalRegion.Region", f"#{region}")])
    base_voltage = {}
    for kv in sorted({_num(v) for v in net.bus.vn_kv}):
        base_voltage[kv] = p.add("EQ", "BaseVoltage", _id(network_id, "base_voltage", kv),
                                 [("cim:IdentifiedObject.name", f"{kv:g} kV"), ("cim:BaseVoltage.nominalVoltage", kv)])
    subs = given_subs if given_subs is not None else sm.substations(net)[0]
    sub_id, level_of_bus = {}, {}
    for b in net.bus.index:
        sname = subs.get(int(b)) or f"Substation {int(b)}"
        if sname not in sub_id:
            sub_id[sname] = p.add("EQ", "Substation", _id(network_id, "substation", sname),
                                  [("cim:IdentifiedObject.name", sname), ("cim:Substation.Region", f"#{sub_region}")])
        kv = _num(net.bus.at[b, "vn_kv"])
        vl = _id(network_id, "voltage_level", sname, kv)
        if vl not in level_of_bus.values():
            p.add("EQ", "VoltageLevel", vl, [("cim:IdentifiedObject.name", f"{sname} {kv:g} kV"),
                                              ("cim:VoltageLevel.Substation", f"#{sub_id[sname]}"),
                                              ("cim:VoltageLevel.BaseVoltage", f"#{base_voltage[kv]}")])
        level_of_bus[int(b)] = vl

    # -- nodes (bus-branch: one TopologicalNode per bus) -------------------------------------------------------------
    node = {}
    for b in net.bus.index:
        node[int(b)] = p.add("TP", "TopologicalNode", mrid("bus", b),
                             [("cim:IdentifiedObject.name", label("bus", b)),
                              ("cim:TopologicalNode.ConnectivityNodeContainer", f"#{level_of_bus[int(b)]}"),
                              ("cim:TopologicalNode.BaseVoltage", f"#{base_voltage[_num(net.bus.at[b, 'vn_kv'])]}")])
        keep("bus", b, **{c: _num(net.bus.at[b, c], None) for c in ("min_vm_pu", "max_vm_pu")
                          if c in net.bus.columns})
        if "zone" in net.bus.columns:
            if _missing(net.bus.at[b, "zone"]):
                keep("bus", b, unset=["zone"])             # the converter would name it after the substation
            else:
                keep("bus", b, zone=net.bus.at[b, "zone"])
        if not bool(net.bus.at[b, "in_service"]):
            keep("bus", b, in_service=False)

    def terminal(equipment: str, seq: int, bus: int, connected: bool, name: str) -> str:
        t = _id(equipment, "terminal", seq)
        p.add("EQ", "Terminal", t, [("cim:IdentifiedObject.name", f"{name}_{seq}"),
                                     ("cim:ACDCTerminal.sequenceNumber", seq),
                                     ("cim:Terminal.phases", "cim:PhaseCode.ABC"),
                                     ("cim:Terminal.ConductingEquipment", f"#{equipment}")])
        p.add("SSH", "Terminal", t, [("cim:ACDCTerminal.connected", bool(connected))], about=True)
        if equipment not in p.in_service:
            p.service(equipment, connected)
        p.add("TP", "Terminal", t, [("cim:Terminal.TopologicalNode", f"#{node[bus]}")], about=True)
        return t

    patl = p.add("EQ", "OperationalLimitType", _id(network_id, "patl"),
                 [("cim:IdentifiedObject.name", "PATL"),
                  *([("eu:OperationalLimitType.kind", "eu:LimitKind.patl"),
                     ("cim:OperationalLimitType.isInfiniteDuration", True)] if v30 else
                    [("cim:OperationalLimitType.acceptableDuration", 45000),
                     ("entsoe:OperationalLimitType.limitType", "entsoe:LimitTypeKind.patl")]),
                  ("cim:OperationalLimitType.direction", "cim:OperationalLimitDirectionKind.absoluteValue")])

    # -- switches at a line's or transformer's end get a node of their own --------------------------------------------
    end_node: dict[tuple[str, int, int], int] = {}            # (et, element, bus) -> the node the element ends at
    sw = net.switch if len(getattr(net, "switch", [])) else pd.DataFrame()
    for i in sw.index:
        et, bus, element = str(sw.at[i, "et"]), int(sw.at[i, "bus"]), int(sw.at[i, "element"])
        if et in ("l", "t"):
            aux = _id(mrid("switch", i), "node")
            node[aux] = p.add("TP", "TopologicalNode", aux,
                              [("cim:IdentifiedObject.name", f"{label('bus', bus)} · {label('switch', i)}"),
                               ("cim:TopologicalNode.ConnectivityNodeContainer", f"#{level_of_bus[bus]}"),
                               ("cim:TopologicalNode.BaseVoltage", f"#{base_voltage[_num(net.bus.at[bus, 'vn_kv'])]}")])
            end_node[(et, element, bus)] = aux

    def at(et: str, element: int, bus: int):
        return end_node.get((et, element, bus), bus)

    # -- lines ---------------------------------------------------------------------------------------------------------
    f_hz = extras["f_hz"]
    negative_reads: list[str] = []
    for i in net.line.index:
        r = net.line.loc[i]
        eq = mrid("line", i)
        length, par = max(_num(r.length_km, 1.0), 1e-6), max(_num(r.get("parallel"), 1.0), 1.0)
        container = level_of_bus[int(r.from_bus)]
        p.add("EQ", "ACLineSegment", eq, [
            ("cim:IdentifiedObject.name", label("line", i)), ("cim:Equipment.aggregate", False),
            ("cim:Equipment.EquipmentContainer", f"#{container}"),
            ("cim:ConductingEquipment.BaseVoltage", f"#{base_voltage[_num(net.bus.at[r.from_bus, 'vn_kv'])]}"),
            ("cim:Conductor.length", length),
            ("cim:ACLineSegment.r", _num(r.r_ohm_per_km) * length / par),
            ("cim:ACLineSegment.x", _num(r.x_ohm_per_km) * length / par),
            ("cim:ACLineSegment.bch", 2 * math.pi * f_hz * _num(r.c_nf_per_km) * 1e-9 * length * par),
            ("cim:ACLineSegment.gch", _num(r.get("g_us_per_km")) * 1e-6 * length * par),
            ("cim:ACLineSegment.r0", 0.0), ("cim:ACLineSegment.x0", 0.0), ("cim:ACLineSegment.b0ch", 0.0),
            ("cim:ACLineSegment.g0ch", 0.0)])
        on = bool(r.in_service)
        t1 = terminal(eq, 1, at("l", int(i), int(r.from_bus)), on, label("line", i))
        terminal(eq, 2, at("l", int(i), int(r.to_bus)), on, label("line", i))
        ka = _num(r.max_i_ka, None)
        if ka:
            ols = p.add("EQ", "OperationalLimitSet", _id(eq, "limits"),
                        [("cim:IdentifiedObject.name", "Ratings"), ("cim:OperationalLimitSet.Terminal", f"#{t1}")])
            amps = ka * 1000.0 * _num(r.get("df"), 1.0) * par
            limit = p.add("EQ", "CurrentLimit", _id(eq, "patl"),
                          [("cim:IdentifiedObject.name", "PATL"), ("cim:OperationalLimit.OperationalLimitSet", f"#{ols}"),
                           ("cim:OperationalLimit.OperationalLimitType", f"#{patl}"),
                           ("cim:CurrentLimit.normalValue" if v30 else "cim:CurrentLimit.value", amps)])
            if v30:                                   # 3.0: the value in force is the operating point's
                p.add("SSH", "CurrentLimit", limit, [("cim:CurrentLimit.value", amps)], about=True)
        if par != 1 or _num(r.get("df"), 1.0) != 1:
            # The model folds parallel circuits and the derating into one segment; pandapower's own parameters are
            # put back whole, so nothing is counted twice.
            keep("line", i, **{c: _num(r.get(c), None) for c in ("r_ohm_per_km", "x_ohm_per_km", "c_nf_per_km",
                                                                  "g_us_per_km", "length_km", "max_i_ka", "df")},
                 parallel=int(par))
        signed = [c for c in ("r_ohm_per_km", "x_ohm_per_km", "c_nf_per_km", "g_us_per_km") if _num(r.get(c)) < 0]
        if signed:
            # The profiles carry the sign; pandapower's converter reads the magnitude: the adapter puts it back.
            keep("line", i, **{c: _num(r.get(c)) for c in signed})
            negative_reads.append(label("line", i))
        keep("line", i, max_loading_percent=_num(r.get("max_loading_percent"), None))

    # -- two-winding transformers --------------------------------------------------------------------------------------
    for i in net.trafo.index:
        r = net.trafo.loc[i]
        eq = mrid("trafo", i)
        par = max(_num(r.get("parallel"), 1.0), 1.0)
        sn, hv, lv = _num(r.sn_mva) * par, _num(r.vn_hv_kv), _num(r.vn_lv_kv)
        z = _num(r.vk_percent) / 100.0 * hv ** 2 / sn
        rr = _num(r.vkr_percent) / 100.0 * hv ** 2 / sn
        x = math.copysign(math.sqrt(max(z * z - rr * rr, 0.0)), _num(r.vk_percent, 1.0) or 1.0)
        g = _num(r.pfe_kw) * par / 1000.0 / hv ** 2
        y = _num(r.i0_percent) / 100.0 * sn / hv ** 2
        b = -math.sqrt(max(y * y - g * g, 0.0))
        shift = _num(r.get("shift_degree"))
        # The clock carries the multiple of 30°, a phase tap changer the rest (below).
        residual = shift - 30.0 * round(shift / 30.0)
        p.add("EQ", "PowerTransformer", eq, [("cim:IdentifiedObject.name", label("trafo", i)),
                                             ("cim:Equipment.aggregate", False),
                                             ("cim:Equipment.EquipmentContainer", f"#{level_of_bus[int(r.hv_bus)]}")])
        on = bool(r.in_service)
        ends = []
        for k, (bus, kv, rr_, x_, g_, b_, clock) in enumerate(
                ((int(r.hv_bus), hv, rr, x, g, b, 0), (int(r.lv_bus), lv, 0.0, 0.0, 0.0, 0.0,
                                                        int(round(shift / 30.0)) % 12)), start=1):
            t = terminal(eq, k, at("t", int(i), bus), on, label("trafo", i))
            end = p.add("EQ", "PowerTransformerEnd", _id(eq, "end", k), [
                ("cim:IdentifiedObject.name", label("trafo", i)), ("cim:TransformerEnd.endNumber", k),
                ("cim:TransformerEnd.grounded", False), ("cim:TransformerEnd.Terminal", f"#{t}"),
                ("cim:TransformerEnd.BaseVoltage", f"#{base_voltage[_num(net.bus.at[bus, 'vn_kv'])]}"),
                ("cim:PowerTransformerEnd.ratedS", sn), ("cim:PowerTransformerEnd.ratedU", kv),
                ("cim:PowerTransformerEnd.r", rr_), ("cim:PowerTransformerEnd.x", x_),
                ("cim:PowerTransformerEnd.g", g_), ("cim:PowerTransformerEnd.b", b_),
                ("cim:PowerTransformerEnd.r0", 0.0), ("cim:PowerTransformerEnd.x0", 0.0),
                ("cim:PowerTransformerEnd.g0", 0.0), ("cim:PowerTransformerEnd.b0", 0.0),
                ("cim:PowerTransformerEnd.phaseAngleClock", clock),
                ("cim:PowerTransformerEnd.connectionKind", "cim:WindingConnection.Y"),
                ("cim:PowerTransformerEnd.PowerTransformer", f"#{eq}")])
            ends.append((end, kv))
        tap_pos = _num(r.get("tap_pos"), None)
        if tap_pos is not None and _num(r.get("tap_step_percent"), 0.0):
            side = 0 if str(r.get("tap_side") or "hv") == "hv" else 1
            if _num(r.get("tap_step_degree"), 0.0):
                losses.append(f"transformer {label('trafo', i)!r}: the tap's phase step is not written")
            tc = p.add("EQ", "RatioTapChanger", _id(eq, "tap"), [
                ("cim:IdentifiedObject.name", label("trafo", i)),
                ("cim:TapChanger.highStep", int(_num(r.get("tap_max"), tap_pos))),
                ("cim:TapChanger.lowStep", int(_num(r.get("tap_min"), tap_pos))),
                ("cim:TapChanger.ltcFlag", False),
                ("cim:TapChanger.neutralStep", int(_num(r.get("tap_neutral"), 0))),
                ("cim:TapChanger.neutralU", ends[side][1]),
                ("cim:TapChanger.normalStep", int(tap_pos)),
                ("cim:RatioTapChanger.tculControlMode", "cim:TransformerControlMode.volt"),
                ("cim:RatioTapChanger.stepVoltageIncrement", _num(r.get("tap_step_percent"))),
                ("cim:RatioTapChanger.TransformerEnd", f"#{ends[side][0]}")])
            p.add("SSH", "RatioTapChanger", tc, [("cim:TapChanger.controlEnabled", False),
                                                  ("cim:TapChanger.step", int(tap_pos))], about=True)
            empty = [c for c in ("tap_min", "tap_max") if _num(r.get(c), None) is None]
            if empty:
                keep("trafo", i, unset=empty)          # a step stands in for them in the model; left empty here
        has_ratio_tap = tap_pos is not None and bool(_num(r.get("tap_step_percent"), 0.0))
        if abs(residual) > 1e-9 and not has_ratio_tap:
            # A linear phase tap changer at its first step on the high-voltage end carries the rest of the shift
            # (read back exactly, profiles alone included).
            ptc = p.add("EQ", "PhaseTapChangerLinear", _id(eq, "phase_tap"), [
                ("cim:IdentifiedObject.name", f"{label('trafo', i)} phase shift"),
                ("cim:TapChanger.highStep", 1), ("cim:TapChanger.lowStep", 0), ("cim:TapChanger.ltcFlag", False),
                ("cim:TapChanger.neutralStep", 0), ("cim:TapChanger.normalStep", 1),
                ("cim:TapChanger.neutralU", ends[0][1]),
                ("cim:PhaseTapChangerLinear.stepPhaseShiftIncrement", residual),
                ("cim:PhaseTapChangerLinear.xMin", 0.0), ("cim:PhaseTapChangerLinear.xMax", 0.0),
                ("cim:PhaseTapChanger.TransformerEnd", f"#{ends[0][0]}")])
            p.add("SSH", "PhaseTapChangerLinear", ptc, [("cim:TapChanger.controlEnabled", False),
                                                        ("cim:TapChanger.step", 1)], about=True)
            # CONDUCTOR reads the shift back as the network has it (an angle, no tap); another tool reads the tap.
            keep("trafo", i, shift_degree=shift,
                 unset=[c for c in ("tap_changer_type", "tap_pos", "tap_min", "tap_max", "tap_neutral",
                                    "tap_step_degree", "tap_step_percent", "tap_side") if _missing(r.get(c))])
        elif abs(residual) > 1e-9:
            # Beside a ratio tap, a linear phase tap changer on the other end at its first step carries the rest of
            # the shift (seen from the low-voltage end, the angle turns the other way).
            ptc = p.add("EQ", "PhaseTapChangerLinear", _id(eq, "phase_tap"), [
                ("cim:IdentifiedObject.name", f"{label('trafo', i)} phase shift"),
                ("cim:TapChanger.highStep", 1), ("cim:TapChanger.lowStep", 0), ("cim:TapChanger.ltcFlag", False),
                ("cim:TapChanger.neutralStep", 0), ("cim:TapChanger.normalStep", 1),
                ("cim:TapChanger.neutralU", ends[1 - side][1]),
                ("cim:PhaseTapChangerLinear.stepPhaseShiftIncrement", -residual if side == 0 else residual),
                ("cim:PhaseTapChangerLinear.xMin", 0.0), ("cim:PhaseTapChangerLinear.xMax", 0.0),
                ("cim:PhaseTapChanger.TransformerEnd", f"#{ends[1 - side][0]}")])
            p.add("SSH", "PhaseTapChangerLinear", ptc, [("cim:TapChanger.controlEnabled", False),
                                                        ("cim:TapChanger.step", 1)], about=True)
            # CONDUCTOR reads the transformer back as the network has it: its own tap, the angle, no second tap
            # (the converter puts the two tap changers in either slot, by end).
            taps = ("tap_side", "tap_pos", "tap_neutral", "tap_min", "tap_max", "tap_step_percent", "tap_step_degree",
                    "tap_changer_type")
            keep("trafo", i, shift_degree=shift, **{c: r.get(c) for c in taps if not _missing(r.get(c))},
                 unset=[c for c in (*taps, "tap2_changer_type", "tap2_pos", "tap2_min", "tap2_max", "tap2_neutral",
                                    "tap2_step_degree", "tap2_step_percent", "tap2_side") if _missing(r.get(c))])
        if par != 1:
            keep("trafo", i, **{c: _num(r.get(c), None) for c in ("sn_mva", "vk_percent", "vkr_percent", "pfe_kw",
                                                                   "i0_percent")}, parallel=int(par))
        if _num(r.vkr_percent) < 0:
            keep("trafo", i, vkr_percent=_num(r.vkr_percent))    # as for a line: the converter reads |r|
            negative_reads.append(label("trafo", i))
        keep("trafo", i, max_loading_percent=_num(r.get("max_loading_percent"), None))
        if y < g:
            # A magnetising current below the iron losses: no susceptance left to write, the converter reads i0 back
            # from the losses alone.
            keep("trafo", i, i0_percent=_num(r.i0_percent))
        keep("trafo", i, **{c: bool(r.get(c)) for c in ("oltc", "tap_dependency_table")
                            if c in net.trafo.columns and pd.notna(r.get(c))})

    # -- three-winding transformers: the star the pairwise short-circuit voltages make ---------------------------------
    for i in getattr(net, "trafo3w", pd.DataFrame()).index:
        r = net.trafo3w.loc[i]
        eq = mrid("trafo3w", i)
        u = {"hv": _num(r.vn_hv_kv), "mv": _num(r.vn_mv_kv), "lv": _num(r.vn_lv_kv)}
        s = {"hv": _num(r.sn_hv_mva), "mv": _num(r.sn_mv_mva), "lv": _num(r.sn_lv_mva)}
        pairs = {"hv": ("hv", "mv"), "mv": ("mv", "lv"), "lv": ("lv", "hv")}   # vk_<key>: between these two ends
        rp, xp = {}, {}
        for key, (a, b2) in pairs.items():
            base = u[a] ** 2 / min(s[a], s[b2])          # in ohm at end a's voltage
            zz, rr2 = _num(r[f"vk_{key}_percent"]) / 100 * base, _num(r[f"vkr_{key}_percent"]) / 100 * base
            ref = (u["hv"] / u[a]) ** 2                   # referred to the high-voltage end
            rp[key], xp[key] = rr2 * ref, math.sqrt(max(zz * zz - rr2 * rr2, 0.0)) * ref
        star = {}
        for end, (plus1, plus2, minus) in {"hv": ("hv", "lv", "mv"), "mv": ("hv", "mv", "lv"),
                                           "lv": ("mv", "lv", "hv")}.items():
            back = (u[end] / u["hv"]) ** 2                # from the high-voltage end to this end's voltage
            star[end] = ((rp[plus1] + rp[plus2] - rp[minus]) / 2 * back, (xp[plus1] + xp[plus2] - xp[minus]) / 2 * back)
        g = _num(r.pfe_kw) / 1000.0 / u["hv"] ** 2
        y = _num(r.i0_percent) / 100.0 * s["hv"] / u["hv"] ** 2
        b = -math.sqrt(max(y * y - g * g, 0.0))
        p.add("EQ", "PowerTransformer", eq, [("cim:IdentifiedObject.name", label("trafo3w", i)),
                                             ("cim:Equipment.aggregate", False),
                                             ("cim:Equipment.EquipmentContainer", f"#{level_of_bus[int(r.hv_bus)]}")])
        ends = {}
        for k, end in enumerate(("hv", "mv", "lv"), start=1):
            bus = int(r[f"{end}_bus"])
            t = terminal(eq, k, bus, bool(r.in_service), label("trafo3w", i))
            shift = _num(r.get(f"shift_{end}_degree")) if end != "hv" else 0.0
            ends[end] = p.add("EQ", "PowerTransformerEnd", _id(eq, "end", k), [
                ("cim:IdentifiedObject.name", label("trafo3w", i)), ("cim:TransformerEnd.endNumber", k),
                ("cim:TransformerEnd.grounded", False), ("cim:TransformerEnd.Terminal", f"#{t}"),
                ("cim:TransformerEnd.BaseVoltage", f"#{base_voltage[_num(net.bus.at[bus, 'vn_kv'])]}"),
                ("cim:PowerTransformerEnd.ratedS", s[end]), ("cim:PowerTransformerEnd.ratedU", u[end]),
                ("cim:PowerTransformerEnd.r", star[end][0]), ("cim:PowerTransformerEnd.x", star[end][1]),
                ("cim:PowerTransformerEnd.g", g if end == "hv" else 0.0),
                ("cim:PowerTransformerEnd.b", b if end == "hv" else 0.0),
                ("cim:PowerTransformerEnd.r0", 0.0), ("cim:PowerTransformerEnd.x0", 0.0),
                ("cim:PowerTransformerEnd.g0", 0.0), ("cim:PowerTransformerEnd.b0", 0.0),
                ("cim:PowerTransformerEnd.phaseAngleClock", int(round(shift / 30.0)) % 12),
                ("cim:PowerTransformerEnd.connectionKind", "cim:WindingConnection.Y"),
                ("cim:PowerTransformerEnd.PowerTransformer", f"#{eq}")])
        tap_pos = _num(r.get("tap_pos"), None)
        if tap_pos is not None and _num(r.get("tap_step_percent"), 0.0):
            side = str(r.get("tap_side") or "hv")
            if bool(r.get("tap_at_star_point", False)):
                losses.append(f"three-winding transformer {label('trafo3w', i)!r}: its tap sits at the star point, "
                              "written at its end")
            tc = p.add("EQ", "RatioTapChanger", _id(eq, "tap"), [
                ("cim:IdentifiedObject.name", label("trafo3w", i)),
                ("cim:TapChanger.highStep", int(_num(r.get("tap_max"), tap_pos))),
                ("cim:TapChanger.lowStep", int(_num(r.get("tap_min"), tap_pos))), ("cim:TapChanger.ltcFlag", False),
                ("cim:TapChanger.neutralStep", int(_num(r.get("tap_neutral"), 0))),
                ("cim:TapChanger.neutralU", u.get(side, u["hv"])), ("cim:TapChanger.normalStep", int(tap_pos)),
                ("cim:RatioTapChanger.tculControlMode", "cim:TransformerControlMode.volt"),
                ("cim:RatioTapChanger.stepVoltageIncrement", _num(r.get("tap_step_percent"))),
                ("cim:RatioTapChanger.TransformerEnd", f"#{ends.get(side, ends['hv'])}")])
            p.add("SSH", "RatioTapChanger", tc, [("cim:TapChanger.controlEnabled", False),
                                                  ("cim:TapChanger.step", int(tap_pos))], about=True)
            empty = [c for c in ("tap_min", "tap_max") if _num(r.get(c), None) is None]
            if empty:
                keep("trafo3w", i, unset=empty)
        for end in ("mv", "lv"):
            sh = _num(r.get(f"shift_{end}_degree"))
            if abs(sh / 30.0 - round(sh / 30.0)) > 1e-9:
                keep("trafo3w", i, **{f"shift_{end}_degree": sh})
        keep("trafo3w", i, max_loading_percent=_num(r.get("max_loading_percent"), None))
        keep("trafo3w", i, **{c: bool(r.get(c)) for c in ("oltc", "tap_dependency_table")
                              if c in net.trafo3w.columns and pd.notna(r.get(c))})

    # -- loads, units, shunts, the external grid ----------------------------------------------------------------------
    for i in net.load.index:
        r = net.load.loc[i]
        eq = mrid("load", i)
        scaling = _num(r.get("scaling"), 1.0)
        p.add("EQ", "EnergyConsumer", eq, [("cim:IdentifiedObject.name", label("load", i)),
                                           ("cim:Equipment.aggregate", False),
                                           ("cim:Equipment.EquipmentContainer", f"#{level_of_bus[int(r.bus)]}")])
        p.add("SSH", "EnergyConsumer", eq, [("cim:EnergyConsumer.p", _num(r.p_mw) * scaling),
                                            ("cim:EnergyConsumer.q", _num(r.q_mvar) * scaling)], about=True)
        terminal(eq, 1, int(r.bus), bool(r.in_service), label("load", i))
        keep("load", i, p_mw=_num(r.p_mw), q_mvar=_num(r.q_mvar), scaling=scaling if scaling != 1 else None,
             **{c: _num(r.get(c)) for c in ("const_z_p_percent", "const_i_p_percent", "const_z_q_percent",
                                            "const_i_q_percent") if _num(r.get(c))},
             **{c: (bool(r.get(c)) if c == "controllable" else _num(r.get(c), None))
                for c in ("controllable", "min_p_mw", "max_p_mw", "min_q_mvar", "max_q_mvar")
                if c in net.load.columns and pd.notna(r.get(c))})

    def regulating(eq: str, t: str, target_kv: float, name: str) -> str:
        rc = p.add("EQ", "RegulatingControl", _id(eq, "control"),
                   [("cim:IdentifiedObject.name", name), ("cim:RegulatingControl.mode",
                                                          "cim:RegulatingControlModeKind.voltage"),
                    ("cim:RegulatingControl.Terminal", f"#{t}")])
        p.add("SSH", "RegulatingControl", rc, [("cim:RegulatingControl.discrete", False),
                                               ("cim:RegulatingControl.enabled", True),
                                               ("cim:RegulatingControl.targetDeadband", 0.0),
                                               ("cim:RegulatingControl.targetValue", target_kv),
                                               ("cim:RegulatingControl.targetValueUnitMultiplier",
                                                "cim:UnitMultiplier.k")], about=True)
        return rc

    for i in net.gen.index:
        r = net.gen.loc[i]
        eq, bus = mrid("gen", i), int(r.bus)
        kv = _num(net.bus.at[bus, "vn_kv"])
        gu = p.add("EQ", "GeneratingUnit", _id(eq, "unit"), [
            ("cim:IdentifiedObject.name", label("gen", i)), ("cim:Equipment.aggregate", False),
            ("cim:Equipment.EquipmentContainer", f"#{level_of_bus[bus]}"),
            ("cim:GeneratingUnit.initialP", _num(r.p_mw)),
            ("cim:GeneratingUnit.maxOperatingP", _num(r.get("max_p_mw"), max(_num(r.p_mw), 0.0))),
            ("cim:GeneratingUnit.minOperatingP", _num(r.get("min_p_mw"), 0.0))])
        sn = _num(r.get("sn_mva"), None) or max(abs(_num(r.p_mw)), 1.0)
        t_name = label("gen", i)
        t_id = _id(eq, "terminal", 1)
        rc = regulating(eq, t_id, _num(r.vm_pu, 1.0) * kv, t_name)
        p.add("EQ", "SynchronousMachine", eq, [
            ("cim:IdentifiedObject.name", t_name), ("cim:Equipment.EquipmentContainer", f"#{level_of_bus[bus]}"),
            ("cim:RegulatingCondEq.RegulatingControl", f"#{rc}"), ("cim:RotatingMachine.ratedS", sn),
            ("cim:RotatingMachine.ratedPowerFactor", 0.85), ("cim:RotatingMachine.ratedU", kv),
            ("cim:RotatingMachine.GeneratingUnit", f"#{gu}"),
            ("cim:SynchronousMachine.maxQ", _num(r.get("max_q_mvar"), 9999.0)),
            ("cim:SynchronousMachine.minQ", _num(r.get("min_q_mvar"), -9999.0)),
            ("cim:SynchronousMachine.qPercent", 100.0), ("cim:SynchronousMachine.earthing", False),
            ("cim:SynchronousMachine.type", "cim:SynchronousMachineKind.generator")])
        p.add("SSH", "SynchronousMachine", eq, [
            ("cim:RegulatingCondEq.controlEnabled", True), ("cim:RotatingMachine.p", -_num(r.p_mw)),
            ("cim:RotatingMachine.q", -_num(r.get("q_mvar"))),
            ("cim:SynchronousMachine.referencePriority", 1 if bool(r.get("slack", False)) else 0),
            ("cim:SynchronousMachine.operatingMode", "cim:SynchronousMachineOperatingMode.generator")], about=True)
        terminal(eq, 1, bus, bool(r.in_service), t_name)
        keep("gen", i, **{c: r.get(c) for c in ("controllable", "min_p_mw", "max_p_mw", "min_q_mvar", "max_q_mvar",
                                                "slack_weight", "type") if c in net.gen.columns and pd.notna(r.get(c))})
        empty = [c for c in ("sn_mva", "type", "min_q_mvar", "max_q_mvar", "min_p_mw", "max_p_mw")
                 if _missing(r.get(c))]
        if empty:
            keep("gen", i, unset=empty)                    # the model states a rating and a kind in their place

    storage_rows = [("storage", i) for i in getattr(net, "storage", pd.DataFrame()).index]
    schedules: set[str] = set()
    for table, i in [("sgen", i) for i in net.sgen.index] + storage_rows:
        r = net[table].loc[i]
        eq, bus = mrid(table, i), int(r.bus)
        scaling = _num(r.get("scaling"), 1.0)
        p_mw = _num(r.p_mw) * scaling * (-1.0 if table == "storage" else 1.0)     # storage: charging is positive
        kind = _SGEN_TYPE.get(str(r.get("type") or "").replace(" ", "").upper())
        unit_class = ("BatteryUnit" if table == "storage" else
                      {"PV": "PhotoVoltaicUnit", "WP": "PowerElectronicsWindUnit"}.get(kind)) if v30 else None
        if unit_class:
            # CGMES 3.0: an inverter (PowerElectronicsConnection) and the unit behind it.
            q_mvar = _num(r.get("q_mvar")) * scaling * (-1.0 if table == "storage" else 1.0)
            kv_bus = _num(net.bus.at[bus, "vn_kv"])
            p.add("EQ", "PowerElectronicsConnection", eq, [
                ("cim:IdentifiedObject.name", label(table, i)), ("cim:Equipment.aggregate", False),
                ("cim:Equipment.EquipmentContainer", f"#{level_of_bus[bus]}"),
                ("cim:PowerElectronicsConnection.ratedU", kv_bus),
                ("cim:PowerElectronicsConnection.ratedS", _num(r.get("sn_mva"), None) or max(abs(p_mw), 1e-3)),
                ("cim:PowerElectronicsConnection.maxQ", _num(r.get("max_q_mvar"), None)),
                ("cim:PowerElectronicsConnection.minQ", _num(r.get("min_q_mvar"), None))])
            p.add("SSH", "PowerElectronicsConnection", eq, [("cim:RegulatingCondEq.controlEnabled", False),
                                                            ("cim:PowerElectronicsConnection.p", -p_mw),
                                                            ("cim:PowerElectronicsConnection.q", -q_mvar)], about=True)
            unit = p.add("EQ", unit_class, _id(eq, "unit"), [
                ("cim:IdentifiedObject.name", label(table, i)), ("cim:Equipment.aggregate", False),
                ("cim:Equipment.EquipmentContainer", f"#{level_of_bus[bus]}"),
                ("cim:PowerElectronicsUnit.PowerElectronicsConnection", f"#{eq}"),
                ("cim:PowerElectronicsUnit.maxP", _num(r.get("max_p_mw"), max(abs(p_mw), 0.0))),
                ("cim:PowerElectronicsUnit.minP", _num(r.get("min_p_mw"), -abs(p_mw) if table == "storage" else 0.0)),
                *([("cim:BatteryUnit.ratedE", _num(r.get("max_e_mwh"), 0.0))] if table == "storage" else [])])
            p.service(unit, bool(r.in_service))
            if table == "storage":
                p.add("SSH", "BatteryUnit", unit, [("cim:BatteryUnit.storedE", _num(r.get("max_e_mwh"), 0.0)
                                                    * _num(r.get("soc_percent"), 0.0) / 100.0)], about=True)
            terminal(eq, 1, bus, bool(r.in_service), label(table, i))
            keep(table, i, p_mw=_num(r.p_mw), q_mvar=_num(r.get("q_mvar")), scaling=scaling if scaling != 1 else None,
                 **({"storage": {c: r.get(c) for c in ("max_e_mwh", "min_e_mwh", "soc_percent", "sn_mva")
                                 if c in net.storage.columns and pd.notna(r.get(c))}} if table == "storage" else {}),
                 **{c: r.get(c) for c in ("type", "controllable", "min_p_mw", "max_p_mw", "min_q_mvar", "max_q_mvar",
                                          "sn_mva") if c in net[table].columns and pd.notna(r.get(c))
                    and table == "sgen"})
            continue
        est = _id(network_id, "schedule", kind) if kind else None
        if est and est not in schedules:          # one per type, shared by the units of that type
            schedules.add(p.add("EQ", "EnergySchedulingType", est, [("cim:IdentifiedObject.name", kind)]))
        p.add("EQ", "EnergySource", eq, [("cim:IdentifiedObject.name", label(table, i)),
                                         ("cim:Equipment.aggregate", False),
                                         ("cim:Equipment.EquipmentContainer", f"#{level_of_bus[bus]}"),
                                         ("cim:EnergySource.nominalVoltage", _num(net.bus.at[bus, "vn_kv"])),
                                         *([("cim:EnergySource.EnergySchedulingType", f"#{est}")] if est else [])])
        p.add("SSH", "EnergySource", eq, [("cim:EnergySource.activePower", -p_mw),
                                          ("cim:EnergySource.reactivePower",
                                           -_num(r.get("q_mvar")) * scaling * (-1.0 if table == "storage" else 1.0))],
              about=True)
        terminal(eq, 1, bus, bool(r.in_service), label(table, i))
        keep(table, i, p_mw=_num(r.p_mw), q_mvar=_num(r.get("q_mvar")), scaling=scaling if scaling != 1 else None,
             **({"storage": {c: r.get(c) for c in ("max_e_mwh", "min_e_mwh", "soc_percent", "sn_mva")
                             if c in net.storage.columns and pd.notna(r.get(c))}} if table == "storage" else {}),
             **{c: r.get(c) for c in ("type", "controllable", "min_p_mw", "max_p_mw", "min_q_mvar", "max_q_mvar",
                                      "sn_mva") if c in net[table].columns and pd.notna(r.get(c))
                and table == "sgen"})

    for i in getattr(net, "shunt", pd.DataFrame()).index:
        r = net.shunt.loc[i]
        eq, kv = mrid("shunt", i), _num(r.vn_kv) or _num(net.bus.at[r.bus, "vn_kv"])
        p.add("EQ", "LinearShuntCompensator", eq, [
            ("cim:IdentifiedObject.name", label("shunt", i)), ("cim:Equipment.aggregate", False),
            ("cim:Equipment.EquipmentContainer", f"#{level_of_bus[int(r.bus)]}"),
            ("cim:ShuntCompensator.maximumSections", int(_num(r.get("max_step"), 1))),
            ("cim:ShuntCompensator.nomU", kv), ("cim:ShuntCompensator.normalSections", int(_num(r.get("step"), 1))),
            ("cim:LinearShuntCompensator.bPerSection", -_num(r.q_mvar) / kv ** 2),
            ("cim:LinearShuntCompensator.gPerSection", _num(r.p_mw) / kv ** 2),
            ("cim:LinearShuntCompensator.b0PerSection", 0.0), ("cim:LinearShuntCompensator.g0PerSection", 0.0)])
        p.add("SSH", "LinearShuntCompensator", eq, [("cim:RegulatingCondEq.controlEnabled", False),
                                                    ("cim:ShuntCompensator.sections", int(_num(r.get("step"), 1)))],
              about=True)
        terminal(eq, 1, int(r.bus), bool(r.in_service), label("shunt", i))

    for i in net.ext_grid.index:
        r = net.ext_grid.loc[i]
        eq, bus = mrid("ext_grid", i), int(r.bus)
        t_name = label("ext_grid", i)
        rc = regulating(eq, _id(eq, "terminal", 1), _num(r.vm_pu, 1.0) * _num(net.bus.at[bus, "vn_kv"]), t_name)
        p.add("EQ", "ExternalNetworkInjection", eq, [
            ("cim:IdentifiedObject.name", t_name), ("cim:Equipment.aggregate", False),
            ("cim:Equipment.EquipmentContainer", f"#{level_of_bus[bus]}"),
            ("cim:RegulatingCondEq.RegulatingControl", f"#{rc}"),
            ("cim:ExternalNetworkInjection.governorSCD", 0.0),
            ("cim:ExternalNetworkInjection.maxP", _num(r.get("max_p_mw"), 99999.0)),
            ("cim:ExternalNetworkInjection.minP", _num(r.get("min_p_mw"), -99999.0)),
            ("cim:ExternalNetworkInjection.maxQ", _num(r.get("max_q_mvar"), 99999.0)),
            ("cim:ExternalNetworkInjection.minQ", _num(r.get("min_q_mvar"), -99999.0))])
        p.add("SSH", "ExternalNetworkInjection", eq, [("cim:RegulatingCondEq.controlEnabled", True),
                                                      ("cim:ExternalNetworkInjection.referencePriority", 1),
                                                      ("cim:ExternalNetworkInjection.p", 0.0),
                                                      ("cim:ExternalNetworkInjection.q", 0.0)], about=True)
        terminal(eq, 1, bus, bool(r.in_service), t_name)
        keep("ext_grid", i, va_degree=_num(r.get("va_degree"), 0.0) or None,
             **{c: r.get(c) for c in ("min_p_mw", "max_p_mw", "min_q_mvar", "max_q_mvar", "slack_weight")
                if c in net.ext_grid.columns and pd.notna(r.get(c))})

    # -- switches -------------------------------------------------------------------------------------------------------
    for i in sw.index:
        et, bus, element = str(sw.at[i, "et"]), int(sw.at[i, "bus"]), int(sw.at[i, "element"])
        other = element if et == "b" else end_node.get((et, element, bus))
        if other is None or (et == "b" and element not in node):
            losses.append(f"switch {label('switch', i)!r}: what it switches is not in the network")
            continue
        eq = mrid("switch", i)
        cls = _SWITCH_CLASS.get(str(sw.at[i, "type"] if "type" in sw.columns else "").upper(), "Breaker")
        closed = bool(sw.at[i, "closed"])
        if _num(sw.at[i, "z_ohm"] if "z_ohm" in sw.columns else 0.0):
            losses.append(f"switch {label('switch', i)!r}: its impedance ({_num(sw.at[i, 'z_ohm']):g} ohm) is not "
                          "written (a CIM switch has none)")
        p.add("EQ", cls, eq, [("cim:IdentifiedObject.name", label("switch", i)), ("cim:Equipment.aggregate", False),
                              ("cim:Equipment.EquipmentContainer", f"#{level_of_bus[bus]}"),
                              ("cim:Switch.normalOpen", not closed), ("cim:Switch.retained", True)])
        p.add("SSH", cls, eq, [("cim:Switch.open", not closed)], about=True)
        terminal(eq, 1, bus, True, label("switch", i))
        terminal(eq, 2, other, True, label("switch", i))

    # -- what CGMES has no place for --------------------------------------------------------------------------------------
    costs = []
    for table in ("poly_cost", "pwl_cost"):
        df = getattr(net, table, None) if table in net else None
        if isinstance(df, pd.DataFrame) and len(df):
            for _k, row in df.iterrows():
                et, element = str(row.et), int(row.element)
                if et in net and element in net[et].index:
                    costs.append({"table": table, "element": mrid(et, element),
                                  **{c: (row[c] if not isinstance(row[c], float) or not math.isnan(row[c]) else None)
                                     for c in df.columns if c not in ("et", "element")}})
    if costs:
        extras["costs"] = costs
    # -- equivalents of networks outside the model -----------------------------------------------------------------------
    for table in ("ward", "xward"):
        df = getattr(net, table, None) if table in net else None
        for i in (df.index if isinstance(df, pd.DataFrame) else []):
            r = df.loc[i]
            eq, bus = mrid(table, i), int(r.bus)
            kv = _num(net.bus.at[bus, "vn_kv"])
            p.add("EQ", "EquivalentInjection", eq, [
                ("cim:IdentifiedObject.name", label(table, i)), ("cim:Equipment.aggregate", False),
                ("cim:Equipment.EquipmentContainer", f"#{level_of_bus[bus]}"),
                ("cim:ConductingEquipment.BaseVoltage", f"#{base_voltage[kv]}"),
                ("cim:EquivalentInjection.regulationCapability", table == "xward"),
                ("cim:EquivalentInjection.r", _num(r.get("r_ohm"))), ("cim:EquivalentInjection.x", _num(r.get("x_ohm")))])
            p.add("SSH", "EquivalentInjection", eq, [
                ("cim:EquivalentInjection.p", _num(r.ps_mw)), ("cim:EquivalentInjection.q", _num(r.qs_mvar)),
                ("cim:EquivalentInjection.regulationStatus", table == "xward"),
                ("cim:EquivalentInjection.regulationTarget", _num(r.get("vm_pu"), 1.0) * kv)], about=True)
            terminal(eq, 1, bus, bool(r.in_service), label(table, i))
            # pandapower's converter reads no constant-impedance part and sets its own internal impedance.
            keep(table, i, pz_mw=_num(r.get("pz_mw")), qz_mvar=_num(r.get("qz_mvar")),
                 **({"r_ohm": _num(r.get("r_ohm")), "x_ohm": _num(r.get("x_ohm")),
                     "vm_pu": _num(r.get("vm_pu"), 1.0)} if table == "xward" else {}))
    df = getattr(net, "impedance", None) if "impedance" in net else None
    for i in (df.index if isinstance(df, pd.DataFrame) else []):
        r = df.loc[i]
        eq = mrid("impedance", i)
        kv = _num(net.bus.at[int(r.from_bus), "vn_kv"])
        zb = kv ** 2 / (_num(r.get("sn_mva"), None) or extras["sn_mva"])
        p.add("EQ", "EquivalentBranch", eq, [
            ("cim:IdentifiedObject.name", label("impedance", i)), ("cim:Equipment.aggregate", False),
            ("cim:Equipment.EquipmentContainer", f"#{level_of_bus[int(r.from_bus)]}"),
            ("cim:ConductingEquipment.BaseVoltage", f"#{base_voltage[kv]}"),
            ("cim:EquivalentBranch.r", _num(r.rft_pu) * zb), ("cim:EquivalentBranch.x", _num(r.xft_pu) * zb),
            ("cim:EquivalentBranch.r21", _num(r.get("rtf_pu"), _num(r.rft_pu)) * zb),
            ("cim:EquivalentBranch.x21", _num(r.get("xtf_pu"), _num(r.xft_pu)) * zb)])
        terminal(eq, 1, int(r.from_bus), bool(r.in_service), label("impedance", i))
        terminal(eq, 2, int(r.to_bus), bool(r.in_service), label("impedance", i))
        if any(_num(r.get(c)) for c in ("gf_pu", "bf_pu", "gt_pu", "bt_pu")):
            keep("impedance", i, **{c: _num(r.get(c)) for c in ("gf_pu", "bf_pu", "gt_pu", "bt_pu")})
    # -- motors (asynchronous machines) and static var compensators ------------------------------------------------------
    df = getattr(net, "motor", None) if "motor" in net else None
    for i in (df.index if isinstance(df, pd.DataFrame) else []):
        r = df.loc[i]
        eq, bus = mrid("motor", i), int(r.bus)
        p_el = (_num(r.pn_mech_mw) / max(_num(r.get("efficiency_percent"), 100.0), 1e-9) * 100.0
                * _num(r.get("loading_percent"), 100.0) / 100.0 * _num(r.get("scaling"), 1.0))
        cos_phi = min(max(_num(r.get("cos_phi"), 1.0), 1e-6), 1.0)
        p.add("EQ", "AsynchronousMachine", eq, [
            ("cim:IdentifiedObject.name", label("motor", i)), ("cim:Equipment.aggregate", False),
            ("cim:Equipment.EquipmentContainer", f"#{level_of_bus[bus]}"),
            ("cim:RotatingMachine.ratedU", _num(r.get("vn_kv"), _num(net.bus.at[bus, "vn_kv"]))),
            ("cim:RotatingMachine.ratedPowerFactor", _num(r.get("cos_phi_n"), cos_phi)),
            ("cim:RotatingMachine.ratedS", _num(r.get("sn_mva"), None)),
            ("cim:AsynchronousMachine.ratedMechanicalPower", _num(r.pn_mech_mw)),
            ("cim:AsynchronousMachine.efficiency", _num(r.get("efficiency_n_percent"),
                                                        _num(r.get("efficiency_percent"), 100.0))),
            ("cim:AsynchronousMachine.iaIrRatio", _num(r.get("lrc_pu"), None)),
            ("cim:AsynchronousMachine.rxLockedRotorRatio", _num(r.get("rx"), None)),
            ("cim:AsynchronousMachine.asynchronousMachineType", "cim:AsynchronousMachineKind.motor")])
        p.add("SSH", "AsynchronousMachine", eq, [
            ("cim:RotatingMachine.p", p_el), ("cim:RotatingMachine.q", p_el * math.tan(math.acos(cos_phi))),
            ("cim:AsynchronousMachine.asynchronousMachineType", "cim:AsynchronousMachineKind.motor")], about=True)
        terminal(eq, 1, bus, bool(r.in_service), label("motor", i))
        losses.append(f"motor {label('motor', i)!r}: written as an asynchronous machine drawing {p_el:.3g} MW; "
                      "pandapower's CIM converter reads its consumption with the generator's sign (the extras keep "
                      "the motor's own values for CONDUCTOR)")
        keep("motor", i, **{c: r.get(c) for c in ("pn_mech_mw", "cos_phi", "efficiency_percent", "loading_percent",
                                                   "scaling", "vn_kv", "cos_phi_n", "efficiency_n_percent", "lrc_pu",
                                                   "rx") if c in df.columns and pd.notna(r.get(c))})
    df = getattr(net, "svc", None) if "svc" in net else None
    for i in (df.index if isinstance(df, pd.DataFrame) else []):
        r = df.loc[i]
        eq, bus = mrid("svc", i), int(r.bus)
        kv = _num(net.bus.at[bus, "vn_kv"])
        q_now = _num(net.res_svc.at[i, "q_mvar"]) if "res_svc" in net and i in net.res_svc.index else 0.0
        p.add("EQ", "StaticVarCompensator", eq, [
            ("cim:IdentifiedObject.name", label("svc", i)), ("cim:Equipment.aggregate", False),
            ("cim:Equipment.EquipmentContainer", f"#{level_of_bus[bus]}"),
            ("cim:StaticVarCompensator.inductiveRating", _num(r.get("x_l_ohm"))),
            ("cim:StaticVarCompensator.capacitiveRating", -abs(_num(r.get("x_cvar_ohm")))),
            ("cim:StaticVarCompensator.voltageSetPoint", _num(r.get("set_vm_pu"), 1.0) * kv),
            ("cim:StaticVarCompensator.sVCControlMode", "cim:SVCControlMode.voltage")])
        p.add("SSH", "StaticVarCompensator", eq, [("cim:StaticVarCompensator.q", q_now),
                                                  ("cim:RegulatingCondEq.controlEnabled", bool(r.get("controllable", True)))],
              about=True)
        terminal(eq, 1, bus, bool(r.in_service), label("svc", i))
        keep("svc", i, svc={c: r.get(c) for c in df.columns if c not in ("bus", "name", "in_service")
                            and pd.notna(r.get(c))})
        losses.append(f"static var compensator {label('svc', i)!r}: written as one; pandapower's CIM converter reads "
                      "it as a fixed shunt (the extras keep it an SVC for CONDUCTOR)")
    # -- DC lines: a converter at each end, the DC line between their DC nodes ----------------------------------------
    # The DC line segment carries the element's mRID; a DC line container holds it, its converters and nodes.
    df = getattr(net, "dcline", None) if "dcline" in net else None
    for i in (df.index if isinstance(df, pd.DataFrame) else []):
        r = df.loc[i]
        eq, on = mrid("dcline", i), bool(r.in_service)
        name = label("dcline", i)
        box = p.add("EQ", "DCLine", _id(eq, "dc_line"), [("cim:IdentifiedObject.name", name),
                                                         ("cim:DCLine.Region", f"#{sub_region}")])
        p_from = _num(r.p_mw)
        p_to = p_from * (1.0 - _num(r.get("loss_percent")) / 100.0) - _num(r.get("loss_mw"))
        nodes = []
        for k, (bus, vm, p_ac) in enumerate(((int(r.from_bus), _num(r.get("vm_from_pu"), 1.0), p_from),
                                             (int(r.to_bus), _num(r.get("vm_to_pu"), 1.0), -p_to)), start=1):
            kv = _num(net.bus.at[bus, "vn_kv"])
            conv = p.add("EQ", "VsConverter", _id(eq, "converter", k), [
                ("cim:IdentifiedObject.name", f"{name} converter {k}"), ("cim:Equipment.aggregate", False),
                ("cim:Equipment.EquipmentContainer", f"#{box}"), ("cim:ACDCConverter.ratedUdc", kv),
                ("cim:ACDCConverter.idleLoss", _num(r.get("loss_mw")) if k == 1 else 0.0),
                ("cim:ACDCConverter.resistiveLoss", 0.0), ("cim:ACDCConverter.switchingLoss", 0.0),
                ("cim:ACDCConverter.minUdc", 0.0), ("cim:ACDCConverter.maxUdc", 2 * kv),
                ("cim:VsConverter.maxModulationIndex", 1.0), ("cim:VsConverter.maxValveCurrent", 0.0)])
            p.add("SSH", "VsConverter", conv, [
                ("cim:ACDCConverter.p", p_ac), ("cim:ACDCConverter.q", 0.0), ("cim:ACDCConverter.targetPpcc", p_ac),
                ("cim:ACDCConverter.targetUdc", kv), ("cim:VsConverter.droop", 0.0),
                ("cim:VsConverter.pPccControl", "cim:VsPpccControlKind.pPcc"),
                ("cim:VsConverter.qPccControl", "cim:VsQpccControlKind.voltagePcc"),
                ("cim:VsConverter.targetUpcc", vm * kv)], about=True)
            terminal(conv, 1, bus, on, f"{name} converter {k}")
            dc_node = p.add("EQ", "DCNode", _id(eq, "dc_node", k), [("cim:IdentifiedObject.name", f"{name} DC {k}"),
                                                                 ("cim:DCNode.DCEquipmentContainer", f"#{box}")])
            tn = p.add("TP", "DCTopologicalNode", _id(eq, "dc_topological_node", k), [
                ("cim:IdentifiedObject.name", f"{name} DC {k}"), ("cim:DCTopologicalNode.DCEquipmentContainer",
                                                                  f"#{box}")])
            p.add("TP", "DCNode", dc_node, [("cim:DCNode.DCTopologicalNode", f"#{tn}")], about=True)
            term = p.add("EQ", "ACDCConverterDCTerminal", _id(conv, "dc_terminal"), [
                ("cim:IdentifiedObject.name", f"{name} converter {k} DC"), ("cim:ACDCTerminal.sequenceNumber", 1),
                ("cim:DCBaseTerminal.DCNode", f"#{dc_node}"),
                ("cim:ACDCConverterDCTerminal.DCConductingEquipment", f"#{conv}"),
                ("cim:ACDCConverterDCTerminal.polarity", "cim:DCPolarityKind.positive")])
            p.add("SSH", "ACDCConverterDCTerminal", term, [("cim:ACDCTerminal.connected", on)], about=True)
            p.add("TP", "ACDCConverterDCTerminal", term, [("cim:DCBaseTerminal.DCTopologicalNode", f"#{tn}")],
                  about=True)
            nodes.append((dc_node, tn))
        seg = p.add("EQ", "DCLineSegment", eq, [
            ("cim:IdentifiedObject.name", name), ("cim:Equipment.aggregate", False),
            ("cim:Equipment.EquipmentContainer", f"#{box}"), ("cim:DCLineSegment.resistance", 0.0),
            ("cim:DCLineSegment.length", 1.0)])
        p.service(seg, on)
        for k, (dc_node, tn) in enumerate(nodes, start=1):
            term = p.add("EQ", "DCTerminal", _id(seg, "dc_terminal", k), [
                ("cim:IdentifiedObject.name", f"{name} DC {k}"), ("cim:ACDCTerminal.sequenceNumber", k),
                ("cim:DCBaseTerminal.DCNode", f"#{dc_node}"), ("cim:DCTerminal.DCConductingEquipment", f"#{seg}")])
            p.add("SSH", "DCTerminal", term, [("cim:ACDCTerminal.connected", on)], about=True)
            p.add("TP", "DCTerminal", term, [("cim:DCBaseTerminal.DCTopologicalNode", f"#{tn}")], about=True)
        # pandapower's converter reads the line's losses and set-points its own way: the extras keep its values.
        keep("dcline", i, dcline={c: r.get(c) for c in df.columns if c not in ("from_bus", "to_bus")
                                  and not _missing(r.get(c))},
             from_node=mrid("bus", int(r.from_bus)), to_node=mrid("bus", int(r.to_bus)))
        if _num(r.get("loss_percent")):
            losses.append(f"DC line {name!r}: its losses written as the converters' powers; pandapower's CIM "
                          "converter reads them as a fixed loss (the extras keep the percentage for CONDUCTOR)")
    for table in ("ssc", "tcsc"):
        df = getattr(net, table, None) if table in net else None
        if isinstance(df, pd.DataFrame) and len(df):
            losses.append(f"{len(df)} {table} element(s) are not written yet")
    # CONDUCTOR's own columns (a proposal's marks and profiles, operator overrides, names as loaded): no CIM place,
    # kept by mRID like the rest of what the computation needs.
    for table in _ELEMENT_TABLES:
        df = getattr(net, table, None) if table in net else None
        if not isinstance(df, pd.DataFrame) or not len(df):
            continue
        present = [c for c in CONDUCTOR_COLUMNS if c in df.columns]
        for i in df.index if present or table in ("load", "bus") else []:
            fields = {c: df.at[i, c] for c in present if not _missing(df.at[i, c])
                      and not (c == "proposed" and df.at[i, c] is not True and str(df.at[i, c]) != "True")}
            if fields.get("proposed") is True or isinstance(fields.get("loaded_table"), str):
                fields["row"] = int(i)          # its row when its identity is not its row (added, or moved)
            if table in ("load", "bus") and "type" in df.columns and isinstance(df.at[i, "type"], str):
                fields["type"] = df.at[i, "type"]          # wye or delta; busbar or node: pandapower's converter reads none
            if fields:
                keep(table, i, **fields)
    if negative_reads:
        losses.append(f"{len(negative_reads)} branch(es) with a negative resistance or reactance (e.g. "
                      f"{', '.join(negative_reads[:3])}): written with their sign, which pandapower's CIM converter "
                      "reads as a magnitude; the extras keep the sign for CONDUCTOR")
    extras["losses"] = losses
    return {"objects": p, "extras": extras, "losses": losses}


def package(net, network_id: str = "local", name: str | None = None, created: str = "2000-01-01T00:00:00Z",
            version: str = DEFAULT_VERSION) -> bytes:
    """The network as a CGMES zip: its EQ, SSH and TP profiles and `conductor_extras.json` (what the adapter puts
    back by mRID)."""
    out = write(net, network_id, name, created, version)
    p, extras = out["objects"], out["extras"]
    ids = {k: _id(network_id, "model", k) for k in p.spec["profiles"]}
    description = f"{extras['name']}, written as CGMES by CONDUCTOR from its pandapower tables."
    buf = io.BytesIO()
    stem = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in extras["name"])[:40] or "network"
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(f"{stem}_EQ_.xml", p.xml("EQ", ids["EQ"], [], description, created))
        z.writestr(f"{stem}_SSH_.xml", p.xml("SSH", ids["SSH"], [ids["EQ"]], description, created))
        z.writestr(f"{stem}_TP_.xml", p.xml("TP", ids["TP"], [ids["EQ"]], description, created))
        if "SC" in p.objects:
            z.writestr(f"{stem}_SC_.xml", p.xml("SC", ids["SC"], [ids["EQ"]], description, created))
        z.writestr("conductor_extras.json", json.dumps(extras, indent=1, default=_json_default))
    return buf.getvalue()


def _json_default(v):
    if hasattr(v, "item"):
        return v.item()
    if isinstance(v, float) and math.isnan(v):
        return None
    return str(v)
