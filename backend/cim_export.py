"""
cim_export.py — the semantic layer as CIM RDF, for other tools to read.

The layer (semantics.py) names its classes after the Common Information Model;
this writes it out *as* CIM, in the distribution namespace (CIM100, the one
GridAPPS-D uses), so a SPARQL query or a CIM tool can read what CONDUCTOR
knows about the network:

- containers: BaseVoltage, Substation, VoltageLevel, Feeder;
- topology: TopologicalNode, and a Terminal joining each piece of conducting
  equipment to its node;
- equipment: ACLineSegment, PowerTransformer (with its two ends),
  EnergyConsumer, PowerElectronicsConnection with its PhotovoltaicUnit /
  PowerElectronicsWindUnit / BatteryUnit / PowerElectronicsUnit,
  SynchronousMachine with its GeneratingUnit, LinearShuntCompensator,
  ExternalNetworkInjection, Breaker / LoadBreakSwitch / Disconnector / Switch
  with `normalOpen` and `open`;
- control: RegulatingControl and its mode.

What standard CIM has no place for — the basis of each fact (stated, named,
inferred, unknown, declared), a unit's energy source, a load's category, where
each value of the operating point comes from — goes in a namespace of its own
(`cdr:`), beside the standard properties, so a reader that knows only CIM
still reads a valid model.

Units follow the CGMES conventions: voltages in kV, powers in MW / Mvar,
lengths in km, impedances in ohm. The model is bus-branch (pandapower's): a
switch is placed at the end of the branch it opens (`cdr:atEquipment`), not
between two connectivity nodes.

Pure: from the layer's dict to text. Two formats: RDF/XML and JSON-LD.
"""

from __future__ import annotations

import json
import uuid
import xml.etree.ElementTree as ET

CIM_NS = "http://iec.ch/TC57/CIM100#"
CDR_NS = "urn:conductor:semantics#"
RDF_NS = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
_ID_NS = uuid.UUID("6f1c2a3e-1d7b-5c4e-9a8f-0c0d0c7a0b12")


def _urn(mrid: str) -> str:
    return f"urn:uuid:{mrid}"


def _child_id(parent: str, part) -> str:
    """A derived identifier for a part (a terminal, a transformer end, a unit)."""
    return str(uuid.uuid5(_ID_NS, f"{parent}|{part}"))


class _Graph:
    """Nodes as (id, class, [(property, value, is_reference)])."""

    def __init__(self):
        self.nodes: list[tuple[str, str, list]] = []

    def add(self, node_id: str, cls: str, props: list) -> str:
        self.nodes.append((node_id, cls, [p for p in props if p[1] is not None]))
        return node_id


def _lit(prop: str, value):
    return (prop, value, False)


def _ref(prop: str, target_mrid: str | None):
    return (prop, _urn(target_mrid) if target_mrid else None, True)


def _fact(prefix: str, fact: dict | None) -> list:
    """A fact's value, basis and origin as cdr: properties."""
    if not fact:
        return []
    return [_lit(f"cdr:{prefix}", fact.get("value")), _lit(f"cdr:{prefix}Basis", fact.get("basis")),
            _lit(f"cdr:{prefix}From", fact.get("from"))]


def build_graph(semantics: dict, network_id: str = "local") -> _Graph:
    g = _Graph()
    buses = {b["index"]: b for b in semantics["buses"]}
    containers = semantics.get("containers") or {}

    # Base voltages, substations and their voltage levels.
    base_voltage = {}
    for kv in sorted({b["kv"]["value"] for b in buses.values() if b["kv"]["value"] is not None}):
        base_voltage[kv] = _child_id(network_id, f"base_voltage:{kv}")
        g.add(base_voltage[kv], "cim:BaseVoltage",
              [_lit("cim:IdentifiedObject.mRID", base_voltage[kv]), _lit("cim:IdentifiedObject.name", f"{kv:g} kV"),
               _lit("cim:BaseVoltage.nominalVoltage", kv)])
    level_of_bus, sub_of_name = {}, {}
    for sub in containers.get("substations") or []:
        sub_of_name[sub["name"]] = sub["mrid"]
        g.add(sub["mrid"], "cim:Substation",
              [_lit("cim:IdentifiedObject.mRID", sub["mrid"]), _lit("cim:IdentifiedObject.name", sub["name"]),
               _lit("cdr:basis", sub["basis"]), _lit("cdr:from", sub["from"])])
        for vl in sub["voltage_levels"]:
            g.add(vl["mrid"], "cim:VoltageLevel",
                  [_lit("cim:IdentifiedObject.mRID", vl["mrid"]),
                   _lit("cim:IdentifiedObject.name", f"{sub['name']} {vl['kv']:g} kV" if vl["kv"] else sub["name"]),
                   _ref("cim:VoltageLevel.Substation", sub["mrid"]),
                   _ref("cim:VoltageLevel.BaseVoltage", base_voltage.get(vl["kv"]))])
            for name in vl["buses"]:
                level_of_bus[name] = vl["mrid"]

    # Nodes.
    for b in buses.values():
        g.add(b["mrid"], "cim:TopologicalNode",
              [_lit("cim:IdentifiedObject.mRID", b["mrid"]), _lit("cim:IdentifiedObject.name", b["name"]),
               _ref("cim:TopologicalNode.BaseVoltage", base_voltage.get(b["kv"]["value"])),
               _ref("cim:TopologicalNode.ConnectivityNodeContainer", level_of_bus.get(b["name"])),
               *_fact("nominalVoltage", b["kv"]), *_fact("feeder", b.get("feeder"))])

    def terminals(eq_mrid: str, bus_indices: list[int]) -> list[str]:
        out = []
        for seq, bus in enumerate(bus_indices, start=1):
            t = _child_id(eq_mrid, f"terminal:{seq}")
            g.add(t, "cim:Terminal",
                  [_lit("cim:IdentifiedObject.mRID", t), _lit("cim:ACDCTerminal.sequenceNumber", seq),
                   _ref("cim:Terminal.ConductingEquipment", eq_mrid),
                   _ref("cim:Terminal.TopologicalNode", buses[bus]["mrid"])])
            out.append(t)
        return out

    feeder_of_bus = {b["index"]: (b.get("feeder") or {}).get("value") for b in buses.values()}
    feeder_mrid = {f["name"]: f["mrid"] for f in containers.get("feeders") or []}

    def container_for(bus: int) -> str | None:
        f = feeder_of_bus.get(bus)
        return feeder_mrid.get(f) if f else level_of_bus.get(buses[bus]["name"])

    # Branches.
    line_terminal = {}
    for br in semantics["branches"]:
        if br["element_type"] == "line":
            g.add(br["mrid"], "cim:ACLineSegment",
                  [_lit("cim:IdentifiedObject.mRID", br["mrid"]), _lit("cim:IdentifiedObject.name", br["name"]),
                   _ref("cim:ConductingEquipment.BaseVoltage", base_voltage.get(br["kv"])),
                   _lit("cim:Equipment.inService", br["in_service"]),
                   *_fact("ratingMVA", br.get("rating_mva"))])
            line_terminal[br["name"]] = terminals(br["mrid"], br["ends_index"])
        else:
            sub = buses[br["ends_index"][0]]["substation"]["value"]
            g.add(br["mrid"], "cim:PowerTransformer",
                  [_lit("cim:IdentifiedObject.mRID", br["mrid"]), _lit("cim:IdentifiedObject.name", br["name"]),
                   _ref("cim:Equipment.EquipmentContainer", sub_of_name.get(sub)),
                   _lit("cim:Equipment.inService", br["in_service"])])
            ts = terminals(br["mrid"], br["ends_index"])
            for n, (bus, term) in enumerate(zip(br["ends_index"], ts), start=1):
                end = _child_id(br["mrid"], f"end:{n}")
                g.add(end, "cim:PowerTransformerEnd",
                      [_lit("cim:IdentifiedObject.mRID", end), _lit("cim:TransformerEnd.endNumber", n),
                       _ref("cim:PowerTransformerEnd.PowerTransformer", br["mrid"]),
                       _ref("cim:TransformerEnd.Terminal", term),
                       _lit("cim:PowerTransformerEnd.ratedU", (br.get("kv") or [None, None])[n - 1]),
                       _lit("cim:PowerTransformerEnd.ratedS", (br.get("rating_mva") or {}).get("value"))])

    def provenance(e) -> list:
        v = e.get("values") or {}
        return [_lit("cdr:pProvenance", (v.get("p") or {}).get("kind")), _lit("cdr:pFrom", (v.get("p") or {}).get("from")),
                _lit("cdr:qProvenance", (v.get("q") or {}).get("kind")), _lit("cdr:qFrom", (v.get("q") or {}).get("from"))]

    # Loads.
    for d in semantics.get("loads") or []:
        g.add(d["mrid"], "cim:EnergyConsumer",
              [_lit("cim:IdentifiedObject.mRID", d["mrid"]), _lit("cim:IdentifiedObject.name", d["name"]),
               _lit("cim:EnergyConsumer.p", d["p_mw"]), _lit("cim:Equipment.inService", d["in_service"]),
               _ref("cim:Equipment.EquipmentContainer", container_for(d["bus_index"])),
               *_fact("loadCategory", d["category"]), *provenance(d)])
        terminals(d["mrid"], [d["bus_index"]])

    # Units: the connecting equipment, the unit behind it, its control.
    for u in semantics["units"]:
        eq = u["cim_class"] if u["cim_class"] in ("PowerElectronicsConnection", "SynchronousMachine") \
            else "SynchronousMachine" if u["element_type"] == "gen" else "PowerElectronicsConnection"
        props = [_lit("cim:IdentifiedObject.mRID", u["mrid"]), _lit("cim:IdentifiedObject.name", u["name"]),
                 _ref("cim:Equipment.EquipmentContainer", container_for(u["bus_index"])),
                 *_fact("energySource", u["source"]), *_fact("control", u["control"]), *provenance(u)]
        if u["cim_class"] == "GeneratingUnit":
            props.append(_lit("cdr:equipmentClassBasis", "unknown"))
        unit_class = u.get("cim_unit")
        if unit_class:
            unit = _child_id(u["mrid"], "unit")
            link = ("cim:PowerElectronicsUnit.PowerElectronicsConnection" if eq == "PowerElectronicsConnection"
                    else None)
            g.add(unit, f"cim:{unit_class}",
                  [_lit("cim:IdentifiedObject.mRID", unit), _lit("cim:IdentifiedObject.name", u["name"]),
                   *([_ref(link, u["mrid"])] if link else []),
                   _lit("cim:GeneratingUnit.maxOperatingP" if eq == "SynchronousMachine" else "cim:PowerElectronicsUnit.maxP",
                        u.get("p_max_mw"))])
            if eq == "SynchronousMachine":
                props.append(_ref("cim:RotatingMachine.GeneratingUnit", unit))
        ctl = u.get("cim_control")
        terms = terminals(u["mrid"], [u["bus_index"]])
        if ctl:
            rc = _child_id(u["mrid"], "regulating_control")
            g.add(rc, "cim:RegulatingControl",
                  [_lit("cim:IdentifiedObject.mRID", rc),
                   ("cim:RegulatingControl.mode", f"{CIM_NS}RegulatingControlModeKind.{ctl['mode']}", True),
                   ("cim:RegulatingControl.Terminal", _urn(terms[0]), True)])
            props.append(_ref("cim:RegulatingCondEq.RegulatingControl", rc))
        g.add(u["mrid"], f"cim:{eq}", props)

    # Shunts and external grids.
    for x in semantics.get("shunts") or []:
        g.add(x["mrid"], "cim:LinearShuntCompensator",
              [_lit("cim:IdentifiedObject.mRID", x["mrid"]), _lit("cim:IdentifiedObject.name", x["name"]),
               _lit("cim:Equipment.inService", x["in_service"]), _lit("cdr:qAtNominalVoltage", x["q_mvar_at_1pu"]),
               *_fact("shuntKind", x["kind"])])
        terminals(x["mrid"], [x["bus_index"]])
    for e in semantics.get("external_grids") or []:
        g.add(e["mrid"], "cim:ExternalNetworkInjection",
              [_lit("cim:IdentifiedObject.mRID", e["mrid"]), _lit("cim:IdentifiedObject.name", e["name"])])
        terminals(e["mrid"], [e["bus_index"]])

    # Switches: placed at the end of what they switch (bus-branch).
    names = {b["name"]: b["mrid"] for b in semantics["branches"]}
    for sw in semantics.get("switches") or []:
        cls = sw["cim_class"]["value"] or "Switch"
        where = sw["where"]
        at_bus = next((i for i, b in buses.items() if b["name"] == where.get("at")), None)
        g.add(sw["mrid"], f"cim:{cls}",
              [_lit("cim:IdentifiedObject.mRID", sw["mrid"]), _lit("cim:IdentifiedObject.name", sw["name"]),
               _lit("cim:Switch.normalOpen", sw["normal_open"]["value"]), _lit("cim:Switch.open", sw["open"]["value"]),
               _ref("cdr:atEquipment", names.get(where.get("element"))),
               _lit("cdr:classBasis", sw["cim_class"]["basis"]), *_fact("role", sw["role"])])
        if at_bus is not None:
            terminals(sw["mrid"], [at_bus])

    # Feeders: containers, energised from a substation through their head line.
    for f in containers.get("feeders") or []:
        head = line_terminal.get(f["name"], [None])[0]
        first = next((buses[i]["substation"]["value"] for i, b in buses.items()
                      if b["name"] in (f["buses"] or [])[:1]), None)
        g.add(f["mrid"], "cim:Feeder",
              [_lit("cim:IdentifiedObject.mRID", f["mrid"]), _lit("cim:IdentifiedObject.name", f["name"]),
               _ref("cim:Feeder.NormalHeadTerminal", head),
               _ref("cim:Feeder.NormalEnergizingSubstation", sub_of_name.get(first)),
               _lit("cdr:basis", f["basis"]), _lit("cdr:from", f["from"])])
    return g


def _value_text(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def to_rdf_xml(semantics: dict, network_id: str = "local") -> str:
    """The layer as CIM RDF/XML."""
    ET.register_namespace("rdf", RDF_NS)
    ET.register_namespace("cim", CIM_NS)
    ET.register_namespace("cdr", CDR_NS)
    ns = {"cim": CIM_NS, "cdr": CDR_NS}
    root = ET.Element(f"{{{RDF_NS}}}RDF")
    for node_id, cls, props in build_graph(semantics, network_id).nodes:
        prefix, local = cls.split(":")
        el = ET.SubElement(root, f"{{{ns[prefix]}}}{local}", {f"{{{RDF_NS}}}about": _urn(node_id)})
        for prop, value, is_ref in props:
            p_prefix, p_local = prop.split(":")
            child = ET.SubElement(el, f"{{{ns[p_prefix]}}}{p_local}")
            if is_ref:
                child.set(f"{{{RDF_NS}}}resource", value)
            else:
                child.text = _value_text(value)
    ET.indent(root)
    return '<?xml version="1.0" encoding="utf-8"?>\n' + ET.tostring(root, encoding="unicode")


def to_json_ld(semantics: dict, network_id: str = "local") -> str:
    """The layer as JSON-LD (the same graph as the RDF/XML)."""
    graph = []
    for node_id, cls, props in build_graph(semantics, network_id).nodes:
        node: dict = {"@id": _urn(node_id), "@type": cls}
        for prop, value, is_ref in props:
            # Plain literals, as in CIM RDF/XML (CGMES writes no datatypes): both formats are one graph.
            node[prop] = {"@id": value} if is_ref else _value_text(value)
        graph.append(node)
    return json.dumps({"@context": {"cim": CIM_NS, "cdr": CDR_NS, "rdf": RDF_NS}, "@graph": graph}, indent=1,
                      default=str)
