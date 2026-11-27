"""
cim_export.py — the semantic layer as CIM RDF, for other tools to read.

The layer (semantics.py) names its classes after the Common Information Model;
this writes it out *as* CIM, in the distribution namespace (CIM100, the one
GridAPPS-D uses), so a SPARQL query or a CIM tool can read what CONDUCTOR
knows about the network:

- containers: BaseVoltage, Substation, VoltageLevel, Feeder;
- topology: TopologicalNode (or, for a node-breaker CIM model, its
  ConnectivityNodes), and a Terminal joining each piece of conducting
  equipment to its node — a bus coupler to the nodes on both sides;
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
import re
import uuid
import xml.etree.ElementTree as ET

CIM_NS = "http://iec.ch/TC57/CIM100#"
CDR_NS = "urn:conductor:semantics#"
RDF_NS = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
_ID_NS = uuid.UUID("6f1c2a3e-1d7b-5c4e-9a8f-0c0d0c7a0b12")


_UUID = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")


def _urn(mrid: str) -> str:
    """The resource IRI of an mRID: a UUID as urn:uuid; any other identifier (some CIM models use their
    own) percent-encoded under CONDUCTOR's namespace, so the IRI stays valid."""
    from urllib.parse import quote
    text = str(mrid)
    return f"urn:uuid:{text}" if _UUID.fullmatch(text) else f"urn:conductor:mrid:{quote(text, safe='')}"


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


_INTERNAL_KEY = re.compile(r" \| [^|\[\]]*? -> [^|\[\]]*? \[[LT]\d+\]")


def _label(text):
    """A branch as people name it ("Line 1-2"): the layer's display label adds its ends and its internal
    key ("Line 1-2 | Bus 1 -> Bus 2 [L0]"), which CIM states otherwise (terminals, the mRID)."""
    return _INTERNAL_KEY.sub("", text) if isinstance(text, str) else text


def _fact(prefix: str, fact: dict | None) -> list:
    """A fact's value, basis and origin as cdr: properties."""
    if not fact:
        return []
    return [_lit(f"cdr:{prefix}", _label(fact.get("value"))), _lit(f"cdr:{prefix}Basis", fact.get("basis")),
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
                   _lit("cim:IdentifiedObject.name", vl.get("name") or (f"{sub['name']} {vl['kv']:g} kV" if vl["kv"]
                                                                       else sub["name"])),
                   _ref("cim:VoltageLevel.Substation", sub["mrid"]),
                   _ref("cim:VoltageLevel.BaseVoltage", base_voltage.get(vl["kv"]))])
            for name in vl["buses"]:
                level_of_bus[name] = vl["mrid"]

    # Nodes: TopologicalNodes (a bus-branch network), or a node-breaker model's ConnectivityNodes as it has them.
    def node_class(bus: int) -> str:
        return "ConnectivityNode" if buses[bus].get("cim_class") == "ConnectivityNode" else "TopologicalNode"

    for i, b in buses.items():
        cls = node_class(i)
        g.add(b["mrid"], f"cim:{cls}",
              [_lit("cim:IdentifiedObject.mRID", b["mrid"]), _lit("cim:IdentifiedObject.name", b["name"]),
               *([_ref("cim:TopologicalNode.BaseVoltage", base_voltage.get(b["kv"]["value"]))]
                 if cls == "TopologicalNode" else []),
               _ref(f"cim:{cls}.ConnectivityNodeContainer", level_of_bus.get(b["name"])),
               *_fact("nominalVoltage", b["kv"]), *_fact("feeder", b.get("feeder"))])

    def terminals(eq_mrid: str, bus_indices: list[int]) -> list[str]:
        out = []
        for seq, bus in enumerate(bus_indices, start=1):
            t = _child_id(eq_mrid, f"terminal:{seq}")
            g.add(t, "cim:Terminal",
                  [_lit("cim:IdentifiedObject.mRID", t), _lit("cim:ACDCTerminal.sequenceNumber", seq),
                   _ref("cim:Terminal.ConductingEquipment", eq_mrid),
                   _ref(f"cim:Terminal.{node_class(bus)}", buses[bus]["mrid"])])
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
                  [_lit("cim:IdentifiedObject.mRID", br["mrid"]), _lit("cim:IdentifiedObject.name", _label(br["name"])),
                   _ref("cim:ConductingEquipment.BaseVoltage", base_voltage.get(br["kv"])),
                   _lit("cim:Equipment.inService", br["in_service"]),
                   *_fact("ratingMVA", br.get("rating_mva"))])
            line_terminal[br["name"]] = terminals(br["mrid"], br["ends_index"])
        else:
            sub = buses[br["ends_index"][0]]["substation"]["value"]
            g.add(br["mrid"], "cim:PowerTransformer",
                  [_lit("cim:IdentifiedObject.mRID", br["mrid"]), _lit("cim:IdentifiedObject.name", _label(br["name"])),
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
        between = [i for i in where.get("between_index") or [] if i in buses]
        if len(between) == 2:
            # A bus coupler, and every switch of a node-breaker model: joined to the nodes on both sides.
            terminals(sw["mrid"], between)
        elif at_bus is not None:
            terminals(sw["mrid"], [at_bus])

    # Feeders: containers, energised from a substation through their head line.
    for f in containers.get("feeders") or []:
        head = line_terminal.get(f["name"], [None])[0]
        head_line = next((b for b in semantics["branches"] if b["name"] == f["name"]), None)
        fed = set(f["buses"] or [])
        busbar = next((i for i in (head_line or {}).get("ends_index") or []
                       if i in buses and buses[i]["name"] not in fed), None)
        first = buses[busbar]["substation"]["value"] if busbar is not None else None
        g.add(f["mrid"], "cim:Feeder",
              [_lit("cim:IdentifiedObject.mRID", f["mrid"]), _lit("cim:IdentifiedObject.name", _label(f["name"])),
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


# ---------------------------------------------------------------------------
# A proposal as a difference: what it changes in the CIM layer
# ---------------------------------------------------------------------------

DM_NS = "http://iec.ch/2002/schema/CIM_difference_model#"
MD_NS = "http://iec.ch/TC57/61970-552/ModelDescription/1#"


def _statements(semantics: dict, network_id: str) -> dict:
    """{subject: (class, {(property, value as text, is reference)})} of the layer's graph."""
    out = {}
    for node_id, cls, props in build_graph(semantics, network_id).nodes:
        stmts = {(p, v if r else _value_text(v), r) for p, v, r in props}
        if node_id in out:                      # one subject written twice: its statements together
            cls0, before = out[node_id]
            out[node_id] = (cls0, before | stmts)
        else:
            out[node_id] = (cls, stmts)
    return out


def difference(base: dict, proposal: dict, network_id: str = "local") -> dict:
    """What a proposal changes in the CIM layer, element by element: {"added": [(id, class, statements)],
    "removed": [...], "changed": [(id, class, added statements, removed statements)]}. Elements keep their
    mRIDs across the two (the layer derives them from the network as loaded), so an element the proposal
    leaves alone is not in it."""
    before, after = _statements(base, network_id), _statements(proposal, network_id)
    added = [(s, after[s][0], after[s][1]) for s in after if s not in before]
    removed = [(s, before[s][0], before[s][1]) for s in before if s not in after]
    changed = []
    for s in after:
        if s in before:
            plus, minus = after[s][1] - before[s][1], before[s][1] - after[s][1]
            if plus or minus:
                changed.append((s, after[s][0], plus, minus))
    # Each element's name as the proposal has it (or the base, for one it removes): the changed statements
    # alone may hold both an old and a new name, or none.
    names = {s: _name(st) for s, (_c, st) in before.items()}
    names.update({s: _name(st) for s, (_c, st) in after.items()})
    return {"added": added, "removed": removed, "changed": changed, "names": names}


def _name(statements) -> str:
    return next((v for p, v, _r in statements if p == "cim:IdentifiedObject.name"), "")


def difference_summary(diff: dict) -> dict:
    """The difference in words, for the agent and the operator: each element added, removed or changed,
    with the properties that changed (old → new)."""
    def short(prop: str) -> str:
        return prop.split(":", 1)[1]

    names = diff.get("names") or {}

    def entry(sid, cls, stmts):
        # A part without a name of its own (a terminal, a control): its class, and its mRID to find it.
        name = names.get(sid) or _name(stmts)
        return {"element": name or f"a {cls.split(':', 1)[1]}", "class": cls.split(":", 1)[1],
                **({} if name else {"mrid": sid})}

    changed = []
    for sid, cls, plus, minus in diff["changed"]:
        old = {short(p): v for p, v, _r in minus}
        new = {short(p): v for p, v, _r in plus}
        what = [f"{p}: {old.get(p, '—')} → {new.get(p, '—')}" for p in sorted(set(old) | set(new))]
        changed.append({**entry(sid, cls, plus | minus), "changes": what[:8]})
    return {"added": [entry(*a) for a in diff["added"]], "removed": [entry(*r) for r in diff["removed"]],
            "changed": changed,
            "counts": {k: len(diff[k]) for k in ("added", "removed", "changed")}}


def to_difference_rdf_xml(base: dict, proposal: dict, network_id: str = "local", description: str = "",
                          created: str = "") -> str:
    """The proposal as a CIM difference model (IEC 61970-552, as CGMES exchanges changes): forward differences
    (what it adds — whole elements, or the properties it sets) and reverse differences (what it removes).
    Applied to the base's export (`to_rdf_xml`), it gives the proposal's."""
    diff = difference(base, proposal, network_id)
    ET.register_namespace("rdf", RDF_NS)
    ET.register_namespace("cim", CIM_NS)
    ET.register_namespace("cdr", CDR_NS)
    ET.register_namespace("dm", DM_NS)
    ET.register_namespace("md", MD_NS)
    ns = {"cim": CIM_NS, "cdr": CDR_NS}
    root = ET.Element(f"{{{RDF_NS}}}RDF")
    model = ET.SubElement(root, f"{{{DM_NS}}}DifferenceModel",
                          {f"{{{RDF_NS}}}about": _urn(_child_id(network_id, f"difference:{description}"))})
    if created:
        ET.SubElement(model, f"{{{MD_NS}}}Model.created").text = created
    if description:
        ET.SubElement(model, f"{{{MD_NS}}}Model.description").text = description

    def write(parent, sid: str, cls: str | None, stmts):
        tag = f"{{{ns[cls.split(':')[0]]}}}{cls.split(':')[1]}" if cls else f"{{{RDF_NS}}}Description"
        el = ET.SubElement(parent, tag, {f"{{{RDF_NS}}}about": _urn(sid)})
        for prop, value, is_ref in sorted(stmts):
            p_prefix, p_local = prop.split(":")
            child = ET.SubElement(el, f"{{{ns[p_prefix]}}}{p_local}")
            if is_ref:
                child.set(f"{{{RDF_NS}}}resource", value)
            else:
                child.text = value

    forward = ET.SubElement(model, f"{{{DM_NS}}}forwardDifferences", {f"{{{RDF_NS}}}parseType": "Statements"})
    reverse = ET.SubElement(model, f"{{{DM_NS}}}reverseDifferences", {f"{{{RDF_NS}}}parseType": "Statements"})
    for sid, cls, stmts in diff["added"]:
        write(forward, sid, cls, stmts)
    for sid, cls, stmts in diff["removed"]:
        write(reverse, sid, cls, stmts)
    for sid, _cls, plus, minus in diff["changed"]:
        if plus:
            write(forward, sid, None, plus)
        if minus:
            write(reverse, sid, None, minus)
    ET.indent(root)
    return '<?xml version="1.0" encoding="utf-8"?>\n' + ET.tostring(root, encoding="unicode")


ADDITIONS_PROFILE = "urn:conductor:profile:additions"


def additions_rdf_xml(semantics: dict, cim, created: str = "") -> str:
    """What CONDUCTOR adds to a CIM model it was given, as one more profile beside the model's own: each fact the
    layer adds about the model's objects — a unit's energy source, a load's category, a node's feeder, a switch's
    role, each with how it is known (stated, named, inferred, declared by the operator) — referring to them as the
    model's own profiles do (`cim.about`), and the feeders the layer infers, which the model does not hold. With the
    original profiles, nothing of the model is regenerated: the export is lossless."""
    ET.register_namespace("rdf", RDF_NS)
    ET.register_namespace("cdr", CDR_NS)
    ET.register_namespace("md", MD_NS)
    root = ET.Element(f"{{{RDF_NS}}}RDF")
    header = ET.SubElement(root, f"{{{MD_NS}}}FullModel",
                           {f"{{{RDF_NS}}}about": _urn(_child_id("additions", "|".join(
                               h.get("id") or "" for h in cim.headers)))})
    if created:
        ET.SubElement(header, f"{{{MD_NS}}}Model.created").text = created
    ET.SubElement(header, f"{{{MD_NS}}}Model.description").text = (
        "CONDUCTOR's additions to this model: the facts its CIM layer adds, with how each is known.")
    ET.SubElement(header, f"{{{MD_NS}}}Model.profile").text = ADDITIONS_PROFILE
    for h in cim.headers:
        if h.get("id") and h.get("profile") in ("EQ", "SSH", "TP"):
            ET.SubElement(header, f"{{{MD_NS}}}Model.DependentOn", {f"{{{RDF_NS}}}resource": h["id"]})

    def describe(mrid: str, facts: list) -> None:
        props = [f for f in facts if f[1] is not None]
        if not props or cim.get(mrid) is None:
            return
        el = ET.SubElement(root, f"{{{RDF_NS}}}Description", {f"{{{RDF_NS}}}about": cim.about(mrid)})
        for prop, value, is_ref in props:
            child = ET.SubElement(el, f"{{{CDR_NS}}}{prop.split(':')[1]}")
            if is_ref:
                child.set(f"{{{RDF_NS}}}resource", value)
            else:
                child.text = _value_text(value)

    for u in semantics.get("units") or []:
        describe(u["mrid"], _fact("energySource", u["source"]))
    for d in semantics.get("loads") or []:
        describe(d["mrid"], _fact("category", d["category"]))
    for b in semantics.get("buses") or []:
        describe(b["mrid"], _fact("feeder", b.get("feeder")) + _fact("substationName", b.get("substation")))
    for s in semantics.get("switches") or []:
        describe(s["mrid"], _fact("role", s.get("role")))
    # The feeders the layer infers: not in the model, so whole objects of CONDUCTOR's namespace, members by reference.
    node_of = {b["name"]: b["mrid"] for b in semantics.get("buses") or []}
    head_of = {b["name"]: b["mrid"] for b in semantics.get("branches") or []}
    for f in (semantics.get("containers") or {}).get("feeders") or []:
        el = ET.SubElement(root, f"{{{CDR_NS}}}Feeder", {f"{{{RDF_NS}}}about": _urn(f["mrid"])})
        ET.SubElement(el, f"{{{CDR_NS}}}name").text = _label(f["name"])
        ET.SubElement(el, f"{{{CDR_NS}}}basis").text = f["basis"]
        ET.SubElement(el, f"{{{CDR_NS}}}from").text = f["from"]
        if f["name"] in head_of and cim.get(head_of[f["name"]]) is not None:
            ET.SubElement(el, f"{{{CDR_NS}}}headBranch", {f"{{{RDF_NS}}}resource": cim.about(head_of[f["name"]])})
        for name in f["buses"]:
            if name in node_of and cim.get(node_of[name]) is not None:
                ET.SubElement(el, f"{{{CDR_NS}}}member", {f"{{{RDF_NS}}}resource": cim.about(node_of[name])})
    ET.indent(root)
    return '<?xml version="1.0" encoding="utf-8"?>\n' + ET.tostring(root, encoding="unicode")

