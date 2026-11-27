"""
A CIM/CGMES model as the operator gave it: every object of every profile, kept.

pandapower's converter turns a CGMES model into the tables a power flow needs, and keeps only what fits them.
This module keeps the rest: each object — whatever its class — by its mRID, with its class, its attributes and its
references to other objects, merged across the profiles that describe it (EQ, SSH, TP, SV, DL, GL and the boundary
set). The CIM layer (semantics.py) reads its facts here first, so a fact the model states is stated even when the
computation model has no place for it: a voltage level's name, a busbar section, a temporary limit, a shunt's voltage
control, the source tool's solved state.

A typed store, not an RDF graph: lookups are dictionary accesses (microseconds, against milliseconds for a SPARQL
query), since the layer and the answer checks make many on every turn. The model is read once, when it is loaded.
"""

from __future__ import annotations

import collections
import io
import re
import zipfile

from lxml import etree

RDF = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
MD = "http://iec.ch/TC57/61970-552/ModelDescription/1#"

# A profile as the model's header names it (md:Model.profile), or as its file name does.
_PROFILE_URI = (("EquipmentBoundary", "EQ_BD"), ("TopologyBoundary", "TP_BD"), ("SteadyStateHypothesis", "SSH"),
                ("StateVariables", "SV"), ("DiagramLayout", "DL"), ("GeographicalLocation", "GL"),
                ("ShortCircuit", "SC"), ("Topology", "TP"), ("Equipment", "EQ"))
_PROFILE_FILE = ("EQ_BD", "TP_BD", "SSH", "SV", "DL", "GL", "SC", "TP", "EQ")

# CIM's switch classes. pandapower's converter reads the first four; the others it leaves out.
SWITCH_CLASSES = ("Breaker", "Disconnector", "LoadBreakSwitch", "Switch", "Fuse", "Jumper", "GroundDisconnector",
                  "Recloser", "Sectionaliser", "ProtectedSwitch", "DisconnectingCircuitBreaker", "Cut")
CONVERTED_SWITCHES = frozenset({"Breaker", "Disconnector", "LoadBreakSwitch", "Switch"})


_ID_LIKE = re.compile(r"_?[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F-]*")


class CimModelError(ValueError):
    """A file that holds no CIM objects."""


class Ref(str):
    """An attribute's value that refers to another object of the model: its mRID."""


def norm(identifier) -> str:
    """An mRID as the layer writes it: no leading '#', '_' or 'urn:uuid:'."""
    s = str(identifier or "").strip()
    if s.startswith("#"):
        s = s[1:]
    if s.lower().startswith("urn:uuid:"):
        s = s[9:]
    return s.lstrip("_")


def enum(value) -> str | None:
    """An enumeration's value without its type: 'RegulatingControlModeKind.voltage' -> 'voltage'."""
    if value is None:
        return None
    return str(value).rsplit(".", 1)[-1]


def num(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _profile(name: str, header_profiles: list[str]) -> str:
    for uri in header_profiles:
        for key, short in _PROFILE_URI:
            if key in uri:
                return short
    upper = name.upper()
    for short in _PROFILE_FILE:
        if f"_{short}_" in upper or upper.endswith(f"_{short}.XML"):
            return short
    return "?"


def _xml_members(source) -> list[tuple[str, bytes]]:
    """(file name, bytes) of every XML in a zip of profiles, or in a zip of such zips."""
    out = []
    with zipfile.ZipFile(source) as z:
        for name in z.namelist():
            low = name.lower()
            if low.endswith(".zip"):
                out += _xml_members(io.BytesIO(z.read(name)))
            elif low.endswith(".xml"):
                out.append((name.rsplit("/", 1)[-1], z.read(name)))
    return out


class CimModel:
    """Every object of a CIM model by its mRID."""

    def __init__(self):
        self.objects: dict[str, dict] = {}            # mRID -> {"class": ..., attribute: value}
        self.by_class: dict[str, list[str]] = collections.defaultdict(list)
        self.profiles: dict[str, list[str]] = collections.defaultdict(list)   # profile -> its file names
        self.object_profiles: dict[str, set[str]] = collections.defaultdict(set)
        self.headers: list[dict] = []
        self.raw_ids: dict[str, str] = {}             # mRID -> the identifier as the model writes it ("_x", "urn:uuid:x")
        self.namespace: str | None = None             # the model's CIM namespace
        self.version: str | None = None
        self._reverse: dict[str, dict[str, list[str]]] | None = None

    # -- reading -------------------------------------------------------------------------------------------------

    @classmethod
    def read(cls, source) -> "CimModel":
        """The model in `source`: a path to (or file object of) a zip of profiles, or a zip of zips."""
        model = cls()
        members = _xml_members(source)
        if not members:
            raise CimModelError("This zip holds no XML files.")
        for name, data in members:
            model._add_file(name, data)
        if not model.objects:
            raise CimModelError("This zip holds no CIM objects.")
        return model

    def _add_file(self, name: str, data: bytes) -> None:
        parser = etree.XMLParser(resolve_entities=False, no_network=True, huge_tree=True, remove_comments=True)
        root = etree.fromstring(data, parser)
        nsmap = {v: k for k, v in (root.nsmap or {}).items()}
        cim_ns = next((uri for uri in nsmap if "CIM-schema-cim" in uri or "/CIM100" in uri), None)
        if cim_ns and self.version is None:
            self.version = "3.0" if "CIM100" in cim_ns else "2.4.15"
            self.namespace = cim_ns
        header_profiles = [el.text or "" for el in root.iter(f"{{{MD}}}Model.profile")]
        profile = _profile(name, header_profiles)
        self.profiles[profile].append(name)
        for el in root:
            if not isinstance(el.tag, str):
                continue
            qn = etree.QName(el)
            if qn.localname == "FullModel":
                fields = {etree.QName(c).localname.removeprefix("Model."): (c.text or c.get(f"{{{RDF}}}resource"))
                          for c in el if isinstance(c.tag, str)}
                fields.pop("profile", None)          # the header's profile URIs: kept below, under their own key
                self.headers.append({**fields, "file": name, "profile": profile,
                                     "id": el.get(f"{{{RDF}}}about") or el.get(f"{{{RDF}}}ID"),
                                     "profile_uris": header_profiles})
                continue
            rid, about = el.get(f"{{{RDF}}}ID"), el.get(f"{{{RDF}}}about")
            if not (rid or about):
                continue
            key = norm(rid or about)
            if rid or key not in self.raw_ids:
                raw = rid or about
                self.raw_ids[key] = raw[1:] if raw.startswith("#") else raw
            obj = self.objects.setdefault(key, {})
            # The profile that defines an object (rdf:ID) names its class; one that only adds attributes
            # (rdf:about) names it until the definition is read.
            if (rid or "class" not in obj) and obj.get("class") != qn.localname:
                if obj.get("class"):
                    self.by_class[obj["class"]].remove(key)
                obj["class"] = qn.localname
                self.by_class[qn.localname].append(key)
            self.object_profiles[key].add(profile)
            for child in el:
                if not isinstance(child.tag, str):
                    continue
                attr = etree.QName(child).localname
                ref = child.get(f"{{{RDF}}}resource")
                if ref is not None:
                    value = Ref(norm(ref)) if (ref.startswith("#") or ref.lower().startswith("urn:uuid:")
                                               or "://" not in ref) else ref.rsplit("#", 1)[-1]
                else:
                    value = child.text
                if attr in obj and attr != "class":
                    prev = obj[attr]
                    obj[attr] = (prev if isinstance(prev, list) else [prev]) + [value]
                else:
                    obj[attr] = value
        self._reverse = None

    # -- looking up ----------------------------------------------------------------------------------------------

    def get(self, mrid) -> dict | None:
        return self.objects.get(norm(mrid))

    def cls(self, mrid) -> str | None:
        o = self.get(mrid)
        return o.get("class") if o else None

    def name(self, mrid) -> str | None:
        o = self.get(mrid)
        if not o:
            return None
        n = o.get("IdentifiedObject.name")
        return str(n).strip() if n and str(n).strip() else None

    def label(self, mrid) -> str | None:
        """The object's name when it is one people use: None for a name that is only an identifier (some models
        name containers by their mRID)."""
        n = self.name(mrid)
        if not n or _ID_LIKE.fullmatch(n):
            return None
        return n

    def of(self, *classes: str) -> list[str]:
        out = []
        for c in classes:
            out += self.by_class.get(c, [])
        return out

    def ref(self, mrid, attr: str) -> str | None:
        """The mRID an object's attribute refers to."""
        o = self.get(mrid)
        v = o.get(attr) if o else None
        return str(v) if isinstance(v, Ref) else None

    def referring(self, mrid, attr: str) -> list[str]:
        """The objects whose `attr` refers to `mrid` (e.g. a line's terminals: Terminal.ConductingEquipment)."""
        if self._reverse is None:
            rev: dict[str, dict[str, list[str]]] = collections.defaultdict(lambda: collections.defaultdict(list))
            for key, obj in self.objects.items():
                for a, v in obj.items():
                    for x in (v if isinstance(v, list) else [v]):
                        if isinstance(x, Ref):
                            rev[a][str(x)].append(key)
            self._reverse = rev
        return list(self._reverse.get(attr, {}).get(norm(mrid), []))

    def counts(self) -> dict[str, int]:
        return {c: len(ids) for c, ids in sorted(self.by_class.items()) if ids}

    # -- what the layer reads ------------------------------------------------------------------------------------

    def container_chain(self, mrid) -> dict:
        """The containers above a node or a piece of equipment: {"bay", "voltage_level", "substation", "line"}
        (mRIDs), following ConnectivityNode.ConnectivityNodeContainer / Equipment.EquipmentContainer, then
        Bay.VoltageLevel and VoltageLevel.Substation."""
        out: dict = {}
        start = self.ref(mrid, "ConnectivityNode.ConnectivityNodeContainer") or self.ref(mrid, "Equipment.EquipmentContainer")
        seen = set()
        cur = start
        while cur and cur not in seen:
            seen.add(cur)
            c = self.cls(cur)
            if c == "Bay":
                out.setdefault("bay", cur)
                cur = self.ref(cur, "Bay.VoltageLevel")
            elif c == "VoltageLevel":
                out.setdefault("voltage_level", cur)
                cur = self.ref(cur, "VoltageLevel.Substation")
            elif c == "Substation":
                out.setdefault("substation", cur)
                break
            elif c == "Line":
                out.setdefault("line", cur)
                break
            else:
                break
        return out

    def base_kv(self, voltage_level) -> float | None:
        bv = self.ref(voltage_level, "VoltageLevel.BaseVoltage")
        return num((self.get(bv) or {}).get("BaseVoltage.nominalVoltage")) if bv else None

    def terminals(self, equipment) -> list[str]:
        """An equipment's terminals, in their sequence."""
        ts = self.referring(equipment, "Terminal.ConductingEquipment")
        return sorted(ts, key=lambda t: num((self.get(t) or {}).get("ACDCTerminal.sequenceNumber")) or 0)

    def limits(self, equipment) -> list[dict]:
        """The operational limits on an equipment's terminals (or on the equipment itself): each with its type
        (PATL, TATL…), how long it may be held (seconds; None: indefinitely), its kind and value."""
        out = []
        sets = []
        for t in self.terminals(equipment):
            sets += [(s, t) for s in self.referring(t, "OperationalLimitSet.Terminal")]
        sets += [(s, None) for s in self.referring(equipment, "OperationalLimitSet.Equipment")]
        for s, term in sets:
            for lim in self.referring(s, "OperationalLimit.OperationalLimitSet"):
                o = self.get(lim) or {}
                kind = o.get("class")
                value = num(o.get("CurrentLimit.value") or o.get("ActivePowerLimit.value")
                            or o.get("ApparentPowerLimit.value") or o.get("VoltageLimit.value")
                            or o.get("CurrentLimit.normalValue") or o.get("ApparentPowerLimit.normalValue")
                            or o.get("ActivePowerLimit.normalValue") or o.get("VoltageLimit.normalValue"))
                ltype = self.ref(lim, "OperationalLimit.OperationalLimitType")
                t = self.get(ltype) or {}
                duration = num(t.get("OperationalLimitType.acceptableDuration"))
                # ENTSO-E's limit types (CGMES 2.4.15): patl permanent, tatl temporary (held `acceptableDuration`
                # seconds), patlt the permanent threshold; CGMES 3.0 says isInfiniteDuration instead.
                ltk = (enum(t.get("OperationalLimitType.limitType")) or "").lower() or None
                permanent = (str(t.get("OperationalLimitType.isInfiniteDuration", "")).lower() == "true"
                             or ltk == "patl" or (ltk is None and (self.name(ltype) or "").upper() == "PATL"))
                out.append({"mrid": lim, "kind": kind, "type": self.name(ltype) or ltk, "limit_type": ltk,
                            "permanent": permanent, "duration_s": None if permanent else duration,
                            "direction": enum(t.get("OperationalLimitType.direction")), "value": value,
                            "terminal": (num((self.get(term) or {}).get("ACDCTerminal.sequenceNumber")) if term else None),
                            "set": self.name(s)})
        return out

    def regulating_control(self, equipment) -> dict | None:
        """The RegulatingControl an equipment follows: mode, target, whether it is enabled (SSH), and the node it
        regulates."""
        rc = self.ref(equipment, "RegulatingCondEq.RegulatingControl")
        if not rc:
            return None
        o = self.get(rc) or {}
        eq = self.get(equipment) or {}
        enabled = eq.get("RegulatingCondEq.controlEnabled")
        return {"mrid": rc, "class": o.get("class"), "mode": enum(o.get("RegulatingControl.mode")),
                "target": num(o.get("RegulatingControl.targetValue")),
                "deadband": num(o.get("RegulatingControl.targetDeadband")),
                "enabled": (str(enabled).lower() == "true") if enabled is not None else None,
                "control_enabled": (str(o.get("RegulatingControl.enabled")).lower() == "true"
                                    if o.get("RegulatingControl.enabled") is not None else None),
                "terminal": self.ref(rc, "RegulatingControl.Terminal")}

    def diagram_positions(self, nodes) -> dict[str, tuple[float, float]]:
        """{node mRID: (x, y)} in the model's own diagram (its DL profile): each node's DiagramObject points,
        averaged (a busbar drawn as a line sits at its middle). Of several diagrams, the one that places the most
        of `nodes`. y grows downwards in a CIM diagram; returned as drawn, flipped by the caller."""
        wanted = {norm(n) for n in nodes}
        by_diagram: dict[str, dict[str, list[tuple[float, float]]]] = collections.defaultdict(dict)
        for d in self.of("DiagramObject"):
            target = self.ref(d, "DiagramObject.IdentifiedObject")
            if target not in wanted:
                continue
            pts = []
            for p in self.referring(d, "DiagramObjectPoint.DiagramObject"):
                o = self.get(p) or {}
                x, y = num(o.get("DiagramObjectPoint.xPosition")), num(o.get("DiagramObjectPoint.yPosition"))
                if x is not None and y is not None:
                    pts.append((x, y))
            if pts:
                diagram = self.ref(d, "DiagramObject.Diagram") or ""
                by_diagram[diagram].setdefault(target, []).extend(pts)
        if not by_diagram:
            return {}
        best = max(by_diagram.values(), key=len)
        return {n: (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts)) for n, pts in best.items()}

    def describe(self, mrid, depth_names: bool = True) -> dict | None:
        """One object as the model states it: its class, its attributes, and each reference with the class and
        name of what it refers to — any class, so nothing the model holds is out of reach."""
        o = self.get(mrid)
        if o is None:
            return None
        attrs, refs = {}, {}
        for a, v in o.items():
            if a == "class":
                continue
            for x in (v if isinstance(v, list) else [v]):
                if isinstance(x, Ref):
                    refs.setdefault(a, []).append({"mrid": str(x), "class": self.cls(x), "name": self.name(x)}
                                                  if depth_names else str(x))
                else:
                    attrs[a] = x
        return {"mrid": norm(mrid), "class": o.get("class"), "name": self.name(mrid),
                "profiles": sorted(self.object_profiles.get(norm(mrid), ())), "attributes": attrs,
                "references": {a: (r[0] if len(r) == 1 else r) for a, r in refs.items()}}

    def about(self, mrid) -> str:
        """How another file refers to this object, as the model's own profiles do: '#_x' for an rdf:ID '_x'
        (CGMES 2.4.15), the URN itself for 'urn:uuid:x' (CGMES 3.0)."""
        raw = self.raw_ids.get(norm(mrid), norm(mrid))
        return raw if ":" in raw else f"#{raw}"

    def summary(self) -> dict:
        return {"version": self.version, "objects": len(self.objects),
                "profiles": {p: len(f) for p, f in sorted(self.profiles.items())},
                "classes": len([c for c, ids in self.by_class.items() if ids])}
