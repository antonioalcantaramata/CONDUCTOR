"""
cim_changes.py — a proposal as a change to the CIM model.

A proposal is a list of edits (sandbox.py: add a load, reinforce a line, switch…). Here it becomes what CIM makes of
a change: a difference between the base network's CIM model and the proposal's — the objects it adds, those it removes,
and for the others the statements it changes — keyed by mRID, as IEC 61970-552 difference models exchange changes
between tools. Applied to the base model, the difference gives the proposal's model; projected by the adapter
(cim_adapter.py), the network a study runs on.

The difference is computed from the two networks written as CGMES (cgmes_writer.py), with what CGMES has no place for
(`conductor_extras.json`) as statements of its own (profile "XT"), so it carries everything the computation needs. The
edits keep their language (the agent's tools and the interface are unchanged); only what a proposal *is* becomes a CIM
change.

A proposal's network is the base network with its difference applied by mRID (`project_onto`): the base's layout,
every other row untouched, the changed rows as the adapter reads them from the part of the model the difference
touches. `equivalent` and `compare` check two networks are the same element by element (evaluation/cim_central.py runs
them for every kind of edit on several networks).
"""

from __future__ import annotations

import copy
import io
import json
import zipfile

import pandas as pd

EXTRAS = "XT"                 # the pseudo-profile of what CGMES has no place for (per element, by mRID)


# -- a model as statements ---------------------------------------------------------------------------------------------

# The CIM model a network came from, by network identity (registered when it loads): its own identifiers for the
# parts pandapower does not keep (regulating controls, generating units, limits).
_MODELS: dict[str, object] = {}


def use_model(network_id: str, cim) -> None:
    """Register (or with None, forget) the CIM model a network came from."""
    if cim is None:
        _MODELS.pop(str(network_id), None)
    else:
        _MODELS[str(network_id)] = cim


# A part's column in pandapower's tables, by table and terminal number (the importer keeps the model's own ids).
_TERMINALS = {"line": {1: "terminal_from", 2: "terminal_to"}, "impedance": {1: "terminal_from", 2: "terminal_to"},
              "trafo": {1: "terminal_hv", 2: "terminal_lv"}, "trafo3w": {1: "terminal_hv", 2: "terminal_mv",
                                                                           3: "terminal_lv"},
              "switch": {1: "terminal_bus", 2: "terminal_element"}}
_ENDS = {"trafo": {1: "PowerTransformerEnd_id_hv", 2: "PowerTransformerEnd_id_lv"},
         "trafo3w": {1: "PowerTransformerEnd_id_hv", 2: "PowerTransformerEnd_id_mv", 3: "PowerTransformerEnd_id_lv"}}


def _own(v) -> str | None:
    return v.strip().lstrip("_") if isinstance(v, str) and v.strip() else None


def own_ids(net, network_id: str, only: dict | None = None) -> dict[str, str]:
    """{the id the writer derives for a part: the model's own id} for a network that came as CIM — terminals,
    transformer ends and tap changers from what pandapower's importer keeps; regulating controls, generating units
    and the permanent current limit from the model itself, when it is registered (`use_model`)."""
    import cgmes_writer as cw
    import pandas as pd

    import semantics as sm
    cim = _MODELS.get(str(network_id))
    out: dict[str, str] = {}
    for table in _TABLES:
        df = net[table] if table in net and isinstance(net[table], pd.DataFrame) else None
        if df is None or not len(df) or "origin_id" not in df.columns:
            continue
        rows = df.index if only is None else [i for i in df.index if int(i) in set(only.get(table, ()))]
        ids = sm.row_mrids(net, table, network_id)
        for i in rows:
            eq = ids[int(i)]
            if _own(df.at[i, "origin_id"]) != eq:
                continue                                  # not one of the model's own elements
            for seq, col in (_TERMINALS.get(table) or {1: "terminal"}).items():
                if col in df.columns and _own(df.at[i, col]):
                    out[cw._id(eq, "terminal", seq)] = _own(df.at[i, col])
            for k, col in (_ENDS.get(table) or {}).items():
                if col in df.columns and _own(df.at[i, col]):
                    out[cw._id(eq, "end", k)] = _own(df.at[i, col])
            for cls_col, id_col in (("tapchanger_class", "tapchanger_id"), ("tapchanger2_class", "tapchanger2_id")):
                if id_col in df.columns and _own(df.at[i, id_col]):
                    kind = str(df.at[i, cls_col]) if cls_col in df.columns else ""
                    out[cw._id(eq, "phase_tap" if kind.startswith("PhaseTapChanger") else "tap")] = \
                        _own(df.at[i, id_col])
            if cim is not None:
                control = cim.ref(eq, "RegulatingCondEq.RegulatingControl")
                if control:
                    out[cw._id(eq, "control")] = _own(control)
                unit = cim.ref(eq, "RotatingMachine.GeneratingUnit") or next(
                    iter(cim.referring(eq, "PowerElectronicsUnit.PowerElectronicsConnection")), None)
                if unit:
                    out[cw._id(eq, "unit")] = _own(unit)
                if table == "line":
                    for lim in cim.limits(eq):
                        if lim["permanent"] and lim["kind"] == "CurrentLimit":
                            out[cw._id(eq, "patl")] = _own(lim["mrid"])
                            sets = cim.ref(lim["mrid"], "OperationalLimit.OperationalLimitSet")
                            if sets:
                                out[cw._id(eq, "limits")] = _own(sets)
                            break
    return out


def _renamed(objects: dict, in_service: dict, rename: dict) -> tuple[dict, dict]:
    if not rename:
        return objects, in_service
    out = {}
    for (profile, mrid), entry in objects.items():
        props = {k: (f"#{rename[v[1:]]}" if isinstance(v, str) and v.startswith("#") and v[1:] in rename else v)
                 for k, v in entry["props"].items()}
        out[(profile, rename.get(mrid, mrid))] = {**entry, "props": props}
    return out, {rename.get(m, m): v for m, v in in_service.items()}

def _value(v):
    if isinstance(v, float) and v != v:
        return None
    if hasattr(v, "item"):
        v = v.item()
    return v


def statements(net, network_id: str, version: str | None = None, only: dict | None = None) -> dict:
    """{"objects": {(profile, mRID): {"class", "about", "props": {property: value}}}, "extras": {...}} — the
    network's CGMES model (and what it has no place for) as statements keyed by mRID. `only` ({table: indices}):
    just those elements' (cgmes_writer.write)."""
    import cgmes_writer as cw
    out = cw.write(net if only is not None else copy.deepcopy(net), network_id,
                   version=version or cw.DEFAULT_VERSION, only=only)
    p, extras = out["objects"], out["extras"]
    objects: dict = {}
    for profile, items in p.objects.items():
        for cls, oid, props, about in items:
            key = (profile, oid)
            entry = objects.setdefault(key, {"class": cls, "about": about, "props": {}})
            for k, v in props:
                entry["props"][k] = _value(v)
    for mrid, entry in (extras.get("elements") or {}).items():
        objects[(EXTRAS, mrid)] = {"class": entry["table"], "about": True,
                                   "props": {k: json.dumps(v, sort_keys=True, default=str)
                                             for k, v in (entry.get("fields") or {}).items()}}
    general = {k: v for k, v in extras.items() if k not in ("elements",)}
    # A network that came as CIM keeps its model's own identifiers for the parts the writer would name itself.
    rename = own_ids(net, network_id, only)
    objects, in_service = _renamed(objects, dict(p.in_service), rename)
    return {"objects": objects, "general": general, "version": p.version, "in_service": in_service,
            "parts": rename}


class _Overlay:
    """A model's objects changed copy-on-write: what is read or written is copied from the base on first use, so a
    change to a large model costs what it touches (`touched`), not the model's size."""

    def __init__(self, base: dict):
        self.base, self.own, self.gone = base, {}, set()

    def __getitem__(self, key):
        if key in self.own:
            return self.own[key]
        if key in self.gone or key not in self.base:
            raise KeyError(key)
        e = self.base[key]
        copied = self.own[key] = {"class": e["class"], "about": e["about"], "props": dict(e["props"])}
        return copied

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def __contains__(self, key) -> bool:
        return key in self.own or (key in self.base and key not in self.gone)

    def __setitem__(self, key, value) -> None:
        self.own[key] = value
        self.gone.discard(key)

    def setdefault(self, key, default):
        if key in self:
            return self[key]
        self[key] = default
        return default

    def pop(self, key, default=None):
        value = self.get(key, default)
        self.own.pop(key, None)
        if key in self.base:
            self.gone.add(key)
        return value

    def touched(self) -> list:
        return list(self.own) + sorted(k for k in self.gone if k not in self.own)

    def items(self):
        for key in self.base:
            if key in self:
                yield key, self[key] if key in self.own else self.base[key]
        for key, entry in self.own.items():
            if key not in self.base:
                yield key, entry


def difference(base: dict, proposal: dict) -> dict:
    """What turns `base` into `proposal` (both from `statements`): objects added and removed, and the statements
    changed on the others (`set`: new or changed values; `unset`: removed ones). JSON-serialisable."""
    added, removed, changed = [], [], []
    b, p = base["objects"], proposal["objects"]
    if isinstance(p, _Overlay) and p.base is b:
        # Only what the change touched can differ.
        for key in p.touched():
            if key not in p:
                removed.append({"profile": key[0], "mrid": key[1], "class": b[key]["class"]})
                continue
            entry = p.own[key]
            if key not in b:
                added.append({"profile": key[0], "mrid": key[1], **entry})
                continue
            old = b[key]["props"]
            set_ = {k: v for k, v in entry["props"].items() if old.get(k, object()) != v}
            unset = [k for k in old if k not in entry["props"]]
            if set_ or unset or b[key]["class"] != entry["class"]:
                changed.append({"profile": key[0], "mrid": key[1], "class": entry["class"], "about": entry["about"],
                                "set": set_, "unset": unset})
        general = {k: v for k, v in proposal["general"].items() if base["general"].get(k) != v}
        in_service = {k: v for k, v in proposal["in_service"].items() if base["in_service"].get(k) != v}
        return {"added": added, "removed": removed, "changed": changed, "general": general,
                "in_service": in_service, "version": proposal["version"]}
    for key, entry in p.items():
        if key not in b:
            added.append({"profile": key[0], "mrid": key[1], **entry})
            continue
        old = b[key]["props"]
        set_ = {k: v for k, v in entry["props"].items() if old.get(k, object()) != v}
        unset = [k for k in old if k not in entry["props"]]
        if set_ or unset or b[key]["class"] != entry["class"]:
            changed.append({"profile": key[0], "mrid": key[1], "class": entry["class"], "about": entry["about"],
                            "set": set_, "unset": unset})
    for key, entry in b.items():
        if key not in p:
            removed.append({"profile": key[0], "mrid": key[1], "class": entry["class"]})
    general = {k: v for k, v in proposal["general"].items() if base["general"].get(k) != v}
    in_service = {k: v for k, v in proposal["in_service"].items() if base["in_service"].get(k) != v}
    return {"added": added, "removed": removed, "changed": changed, "general": general, "in_service": in_service,
            "version": proposal["version"]}


def apply(base: dict, diff: dict) -> dict:
    """`base` (statements) with `diff` applied: the proposal's model."""
    objects = {k: {"class": v["class"], "about": v["about"], "props": dict(v["props"])}
               for k, v in base["objects"].items()}
    for r in diff.get("removed") or []:
        objects.pop((r["profile"], r["mrid"]), None)
    for c in diff.get("changed") or []:
        entry = objects.setdefault((c["profile"], c["mrid"]), {"class": c["class"], "about": c["about"], "props": {}})
        entry["class"] = c["class"]
        for k in c.get("unset") or []:
            entry["props"].pop(k, None)
        entry["props"].update(c.get("set") or {})
    for a in diff.get("added") or []:
        objects[(a["profile"], a["mrid"])] = {"class": a["class"], "about": a["about"], "props": dict(a["props"])}
    general = {**base["general"], **(diff.get("general") or {})}
    in_service = {k: v for k, v in base["in_service"].items()
                  if ("EQ", k) in objects}
    in_service.update(diff.get("in_service") or {})
    return {"objects": objects, "general": general, "version": diff.get("version") or base["version"],
            "in_service": in_service}


def summary(diff: dict) -> dict:
    """Counts by class: what a difference adds, removes and changes (the equipment, not its parts)."""
    def count(items):
        out: dict = {}
        for x in items:
            if x["profile"] == "EQ":
                out[x["class"]] = out.get(x["class"], 0) + 1
        return out
    return {"added": count(diff.get("added") or []), "removed": count(diff.get("removed") or []),
            "changed": count(diff.get("changed") or []),
            "pandapower_only": sum(1 for x in (diff.get("added") or []) + (diff.get("changed") or [])
                                   if x["profile"] == EXTRAS)}


# -- back to a model, and to a network -----------------------------------------------------------------------------------

def package(model: dict, network_id: str, name: str = "proposal", created: str = "2000-01-01T00:00:00Z") -> bytes:
    """A model (statements) as a CGMES zip the adapter reads."""
    import cgmes_writer as cw
    p = cw._Profiles(model["version"])
    for (profile, oid), entry in model["objects"].items():
        if profile == EXTRAS:
            continue
        p.objects[profile].append((entry["class"], oid, list(entry["props"].items()), entry["about"]))
    p.in_service = dict(model.get("in_service") or {})
    extras = {**model["general"], "elements": {}}
    for (profile, mrid), entry in model["objects"].items():
        if profile == EXTRAS:
            extras["elements"][mrid] = {"table": entry["class"],
                                        "fields": {k: json.loads(v) for k, v in entry["props"].items()}}
    ids = {k: cw._id(network_id, "model", k, name) for k in p.spec["profiles"]}
    description = f"{name}: written by CONDUCTOR from the base model and the proposal's difference."
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for profile in p.spec["profiles"]:
            if p.objects.get(profile) or profile not in ("SC", "SV"):
                deps = [] if profile == "EQ" else [ids["EQ"]]
                z.writestr(f"model_{profile}_.xml", p.xml(profile, ids[profile], deps, description, created))
        z.writestr("conductor_extras.json", json.dumps(extras, indent=1, default=str))
    return buf.getvalue()


def project(base: dict, diff: dict, network_id: str, name: str = "proposal"):
    """The network a study runs on: the base model with the difference applied, projected by the adapter.
    Returns (network, element map, notes)."""
    import cim_adapter
    return cim_adapter.PandapowerAdapter().project(package(apply(base, diff), network_id, name), network_id)


def compare(old_net, new_net, network_id: str, new_map=None) -> dict:
    """Both networks solved and their results compared element by element through the mRIDs (the dual-run check):
    {"matches", "compared", "max_diff", "worst", "n_missing"}."""
    import pandapower as pp

    import cim_adapter as ca
    a, b = copy.deepcopy(old_net), copy.deepcopy(new_net)
    failed = []
    for label, net in (("first", a), ("second", b)):
        try:
            pp.runpp(net, numba=False)
        except Exception as exc:  # noqa: BLE001 — reported
            failed.append(f"the {label} ({type(exc).__name__})")
    if failed:
        both = len(failed) == 2                 # the same network fails the same way: no difference to see
        return {"matches": both, "compared": 0, "max_diff": {}, "worst": [], "n_missing": 0,
                "note": ("Neither power flow solves." if both else f"The power flow of {failed[0]} does not solve.")}
    adapter = ca.PandapowerAdapter()
    before = adapter.results(a, ca.ElementMap.of(a, network_id))
    after = adapter.results(b, new_map or ca.ElementMap.of(b, network_id))
    diff, worst, missing = {}, [], []
    for mrid, row in before.items():
        other = after.get(mrid)
        if other is None:
            missing.append(mrid)
            continue
        for key, value in row.items():
            if key in other:
                d = abs(value - other[key])
                kind = key if key in ("vm_pu", "va_degree", "loading_percent") else "mw"
                diff[kind] = max(diff.get(kind, 0.0), d)
                worst.append((d / ca.ROUND_TRIP_TOLERANCE[kind], mrid, key, value, other[key]))
    worst.sort(reverse=True)
    matches = not missing and all(diff.get(k, 0.0) <= tol for k, tol in ca.ROUND_TRIP_TOLERANCE.items())
    return {"matches": matches, "compared": len(before) - len(missing), "max_diff": diff,
            "worst": [{"mrid": m, "result": k, "edits": round(x, 8), "model": round(y, 8)} for _r, m, k, x, y in worst[:5]],
            "n_missing": len(missing)}


def equivalent(a, b, network_id: str, layout: bool = True, ignore: tuple = ()) -> list[str]:
    """Where two networks differ, element by element through the mRIDs: a row on one side only, any value of any
    column (numbers to 1e-8; None, NaN and a missing column alike; False and 0 alike), a cost row. `layout`: the same
    element must also have the same row index. Empty: the same network."""
    import pandas as pd

    import semantics as sm
    out: list[str] = []
    for t in _TABLES:
        A = a[t] if t in a and isinstance(a[t], pd.DataFrame) else pd.DataFrame()
        B = b[t] if t in b and isinstance(b[t], pd.DataFrame) else pd.DataFrame()
        if not len(A) and not len(B):
            continue
        ma = {sm._row_mrid(a, t, i, network_id): i for i in A.index}
        mb = {sm._row_mrid(b, t, i, network_id): i for i in B.index}
        for m in set(ma) ^ set(mb):
            out.append(f"{t} {ma.get(m, mb.get(m))}: only in {'the first' if m in ma else 'the second'}")
        for m in set(ma) & set(mb):
            i, j = ma[m], mb[m]
            if layout and i != j:
                out.append(f"{t} {i}: at row {j} in the second")
            for c in A.columns.union(B.columns):
                if c in ignore:
                    continue
                x = A.at[i, c] if c in A.columns else None
                y = B.at[j, c] if c in B.columns else None
                empty = {str(x), str(y)} <= {"None", "nan", "NaN", "<NA>"}
                if not empty and not _same(x, y, c, t, i):
                    out.append(f"{t} {i}.{c}: {x!r} vs {y!r}")
    for table in ("poly_cost", "pwl_cost"):
        def costs(net):
            df = net[table] if table in net and isinstance(net[table], pd.DataFrame) else pd.DataFrame()
            rows = {}
            for k in df.index:
                et, el = str(df.at[k, "et"]), int(df.at[k, "element"])
                key = sm._row_mrid(net, et, el, network_id) if et in net and el in net[et].index else f"{et}:{el}"
                rows[key] = {c: df.at[k, c] for c in df.columns if c not in ("et", "element")}
            return rows
        ca_, cb_ = costs(a), costs(b)
        for m in set(ca_) ^ set(cb_):
            out.append(f"{table} of {m}: only in {'the first' if m in ca_ else 'the second'}")
        for m in set(ca_) & set(cb_):
            for c in set(ca_[m]) | set(cb_[m]):
                x, y = ca_[m].get(c), cb_[m].get(c)
                if not ({str(x), str(y)} <= {"None", "nan"}) and str(x) != str(y) and not _same(x, y, c, table, 0):
                    out.append(f"{table} of {m}.{c}: {x!r} vs {y!r}")
    return out


CX = "urn:conductor:cim-extras#"          # CONDUCTOR's annotations in a difference file (a namespace of its own)
_TEXT = ("cim:IdentifiedObject.name", "cim:IdentifiedObject.description", "cim:IdentifiedObject.shortName")


def to_xml(diff: dict, description: str = "", created: str = "", annotated: bool = True) -> str:
    """The difference as an IEC 61970-552 difference model: forward differences (objects added, statements set,
    equipment put in or out of service) and reverse differences (objects removed, statements unset). `annotated`:
    each statement also says which profile it belongs to, and what CGMES has no place for (CONDUCTOR's computation's:
    costs, pandapower-only values) comes along — in CONDUCTOR's own namespace, which another tool ignores — so the file
    reads back as the same difference (`from_xml`)."""
    import xml.etree.ElementTree as ET

    import cgmes_writer as cw
    spec = cw.VERSIONS[diff.get("version") or cw.DEFAULT_VERSION]
    dm, md, rdf = "http://iec.ch/2002/schema/CIM_difference_model#", cw.MD, cw.RDF
    ET.register_namespace("cim", spec["cim"])
    ET.register_namespace(spec["ext_prefix"], spec["ext"])
    ET.register_namespace("dm", dm)
    ET.register_namespace("md", md)
    ET.register_namespace("rdf", rdf)
    ET.register_namespace("cx", CX)
    root = ET.Element(f"{{{rdf}}}RDF")
    model = ET.SubElement(root, f"{{{dm}}}DifferenceModel", {f"{{{rdf}}}about": f"urn:uuid:{cw._id('difference', description)}"})
    if created:
        ET.SubElement(model, f"{{{md}}}Model.created").text = created
    if description:
        ET.SubElement(model, f"{{{md}}}Model.description").text = description
    if annotated:
        ET.SubElement(model, f"{{{CX}}}version").text = str(diff.get("version") or "")
        if diff.get("general"):
            ET.SubElement(model, f"{{{CX}}}general").text = json.dumps(diff["general"], sort_keys=True, default=str)
    forward = ET.SubElement(model, f"{{{dm}}}forwardDifferences", {f"{{{rdf}}}parseType": "Statements"})
    reverse = ET.SubElement(model, f"{{{dm}}}reverseDifferences", {f"{{{rdf}}}parseType": "Statements"})

    def ns(prefixed: str) -> str:
        return spec["ext"] if prefixed.split(":", 1)[0] in ("eu", "entsoe") else spec["cim"]

    def write(parent, cls: str | None, mrid: str, props: dict, profile: str = "", about=None, typed_as=None):
        tag = f"{{{ns(cls)}}}{cls.split(':')[-1]}" if cls else f"{{{rdf}}}Description"
        el = ET.SubElement(parent, tag, {f"{{{rdf}}}about": f"#_{mrid}"})
        if annotated and profile:
            ET.SubElement(el, f"{{{CX}}}profile").text = profile
            if typed_as:
                ET.SubElement(el, f"{{{CX}}}class").text = typed_as
            if about is not None:
                ET.SubElement(el, f"{{{CX}}}about").text = "true" if about else "false"
        for k, v in props.items():
            child = ET.SubElement(el, f"{{{ns(k)}}}{k.split(':')[-1]}")
            if isinstance(v, str) and v.startswith("#"):
                child.set(f"{{{rdf}}}resource", f"#_{v[1:]}")
            elif isinstance(v, str) and v.startswith(("cim:", "eu:", "entsoe:")):
                child.set(f"{{{rdf}}}resource", ns(v) + v.split(":", 1)[1])
            elif v is not None:
                child.text = cw._text(v)

    def write_extras(parent, mrid: str, table: str, props: dict, added: bool = False):
        el = ET.SubElement(parent, f"{{{rdf}}}Description", {f"{{{rdf}}}about": f"#_{mrid}"})
        ET.SubElement(el, f"{{{CX}}}profile").text = EXTRAS
        ET.SubElement(el, f"{{{CX}}}class").text = table
        if added:
            ET.SubElement(el, f"{{{CX}}}added").text = "true"
        for k, v in props.items():
            f = ET.SubElement(el, f"{{{CX}}}field")
            f.set(f"{{{CX}}}name", k)
            if v is not None:
                f.text = v
    for a in diff.get("added") or []:
        if a["profile"] != EXTRAS:
            write(forward, a["class"], a["mrid"], a["props"], a["profile"], a.get("about"))
        elif annotated:
            write_extras(forward, a["mrid"], a["class"], a["props"], added=True)
    for c in diff.get("changed") or []:
        if c["profile"] != EXTRAS:
            if c.get("set"):
                write(forward, None, c["mrid"], c["set"], c["profile"], c.get("about"), c["class"])
            if c.get("unset"):
                write(reverse, None, c["mrid"], {k: None for k in c["unset"]}, c["profile"], c.get("about"),
                      c["class"])
        elif annotated:
            if c.get("set"):
                write_extras(forward, c["mrid"], c["class"], c["set"])
            if c.get("unset"):
                write_extras(reverse, c["mrid"], c["class"], {k: None for k in c["unset"]})
    for mrid, on in (diff.get("in_service") or {}).items():
        write(forward, None, mrid, {"cim:Equipment.inService": bool(on)}, "IS" if annotated else "")
    for r in diff.get("removed") or []:
        if r["profile"] != EXTRAS:
            write(reverse, r["class"], r["mrid"], {}, r["profile"])
        elif annotated:
            write_extras(reverse, r["mrid"], r["class"], {})
    ET.indent(root)
    return '<?xml version="1.0" encoding="utf-8"?>\n' + ET.tostring(root, encoding="unicode")


def _read_value(key: str, el, cim: str, ext: str, ext_prefix: str):
    import cgmes_writer as cw
    res = el.get(f"{{{cw.RDF}}}resource")
    if res is not None:
        if res.startswith("#_"):
            return "#" + res[2:]
        for uri, prefix in ((cim, "cim"), (ext, ext_prefix)):
            if res.startswith(uri):
                return f"{prefix}:{res[len(uri):]}"
        return res
    text = el.text
    if text is None:
        return None
    if key in _TEXT:
        return text
    if text in ("true", "false"):
        return text == "true"
    try:
        return int(text) if text.lstrip("-").isdigit() else float(text)
    except ValueError:
        return text


def from_xml(xml: str | bytes) -> dict:
    """A difference written by `to_xml` (annotated) read back: the same difference."""
    import xml.etree.ElementTree as ET

    import cgmes_writer as cw
    root = ET.fromstring(xml if isinstance(xml, bytes) else xml.encode("utf-8"))
    dm, rdf = "http://iec.ch/2002/schema/CIM_difference_model#", cw.RDF
    model = root.find(f"{{{dm}}}DifferenceModel")
    version = (model.findtext(f"{{{CX}}}version") or "") or cw.DEFAULT_VERSION
    spec = cw.VERSIONS[version]
    cim, ext, ext_prefix = spec["cim"], spec["ext"], spec["ext_prefix"]
    general = json.loads(model.findtext(f"{{{CX}}}general") or "{}")

    def name_of(tag: str) -> str:
        uri, local = tag[1:].split("}", 1)
        return f"{ext_prefix}:{local}" if uri == ext else f"cim:{local}"

    def read(el) -> tuple[str, str, str | None, dict, bool | None, str | None]:
        mrid = el.get(f"{{{rdf}}}about", "")[2:]
        profile = el.findtext(f"{{{CX}}}profile") or ""
        cls = el.findtext(f"{{{CX}}}class")
        about = el.findtext(f"{{{CX}}}about")
        props: dict = {}
        if profile == EXTRAS:
            for f in el.findall(f"{{{CX}}}field"):
                props[f.get(f"{{{CX}}}name")] = f.text
        else:
            for child in el:
                if child.tag.startswith(f"{{{CX}}}"):
                    continue
                key = name_of(child.tag)
                props[key] = _read_value(key, child, cim, ext, ext_prefix)
        typed = None if el.tag == f"{{{rdf}}}Description" else name_of(el.tag).split(":", 1)[1]
        if el.findtext(f"{{{CX}}}added") == "true":
            typed = cls                                   # what CGMES has no place for, for an object added
        return mrid, profile, typed or cls, props, (None if about is None else about == "true"), typed
    added, removed, changed, in_service = [], [], [], {}
    pending: dict[tuple, dict] = {}
    for el in model.find(f"{{{dm}}}forwardDifferences"):
        mrid, profile, cls, props, about, typed = read(el)
        if profile == "IS":
            in_service[mrid] = bool(props.get("cim:Equipment.inService"))
        elif typed:
            added.append({"profile": profile, "mrid": mrid, "class": cls,
                          "about": True if profile == EXTRAS else bool(about), "props": props})
        else:
            entry = pending.setdefault((profile, mrid), {"profile": profile, "mrid": mrid, "class": cls,
                                                         "about": bool(about) if about is not None else True,
                                                         "set": {}, "unset": []})
            entry["set"].update(props)
    for el in model.find(f"{{{dm}}}reverseDifferences"):
        mrid, profile, cls, props, about, typed = read(el)
        if typed and not props:
            removed.append({"profile": profile, "mrid": mrid, "class": cls})
        elif profile == EXTRAS and not props:
            removed.append({"profile": profile, "mrid": mrid, "class": cls})
        else:
            entry = pending.setdefault((profile, mrid), {"profile": profile, "mrid": mrid, "class": cls,
                                                         "about": bool(about) if about is not None else True,
                                                         "set": {}, "unset": []})
            entry["unset"].extend(props)
    return {"added": added, "removed": removed, "changed": list(pending.values()), "general": general,
            "in_service": in_service, "version": version}


def by_element(base: dict, diff: dict, limit: int = 40) -> list[dict]:
    """What a difference changes, element by element: each equipment (or node) it touches, by mRID, with its class,
    whether it is added, removed or changed, and the statements that change on it and its parts (property: before →
    after) — the model's own account of a change, for the agent and the interface."""
    owners, added_root = _touched(base, diff)
    idx = _index(base)
    out: dict[str, dict] = {}

    def entry(owner: str) -> dict:
        if owner not in out:
            cls = next((base["objects"][k]["class"] for k in idx["keys"].get(owner, ()) if k[0] == "EQ"
                        and not base["objects"][k]["about"]), None)
            out[owner] = {"mrid": owner, "class": cls, "change": "changed", "statements": []}
        return out[owner]
    for a in diff.get("added") or []:
        owner = added_root.get(a["mrid"], a["mrid"])
        e = entry(owner)
        if a["mrid"] == owner and a["profile"] == "EQ":
            e["change"], e["class"] = "added", a["class"]
            name = a["props"].get("cim:IdentifiedObject.name")
            if name:
                e["name"] = name
        elif a["profile"] != EXTRAS:
            e["statements"].append({"object": a["class"], "added": True})
    for c in diff.get("changed") or []:
        owner = idx["root"].get(c["mrid"], added_root.get(c["mrid"], c["mrid"]))
        e = entry(owner)
        old = base["objects"].get((c["profile"], c["mrid"]), {}).get("props", {})
        for k, v in (c.get("set") or {}).items():
            e["statements"].append({"object": c["class"], "property": k, "before": old.get(k), "after": v,
                                    **({"computation_only": True} if c["profile"] == EXTRAS else {})})
        for k in c.get("unset") or []:
            e["statements"].append({"object": c["class"], "property": k, "before": old.get(k), "after": None})
    for r in diff.get("removed") or []:
        owner = idx["root"].get(r["mrid"], r["mrid"])
        if r["mrid"] == owner and r["profile"] == "EQ":
            entry(owner)["change"] = "removed"
    for mrid, on in (diff.get("in_service") or {}).items():
        entry(idx["root"].get(mrid, mrid))["statements"].append({"property": "in service", "after": bool(on)})
    rows = sorted(out.values(), key=lambda e: ({"added": 0, "removed": 1, "changed": 2}[e["change"]], e["mrid"]))
    for e in rows:
        e["statements"] = e["statements"][:12]
    return [e for e in rows if e["class"] or e["statements"]][:limit]


# -- the proposal's network with the base's layout ---------------------------------------------------------------------

# A part of the model and the reference that leads to the equipment it belongs to.
_PART_OF = {"Terminal": ("Terminal.ConductingEquipment",), "PowerTransformerEnd": ("PowerTransformerEnd.PowerTransformer",),
            "RatioTapChanger": ("RatioTapChanger.TransformerEnd",),
            "PhaseTapChangerLinear": ("PhaseTapChanger.TransformerEnd",),
            "OperationalLimitSet": ("OperationalLimitSet.Terminal", "OperationalLimitSet.Equipment"),
            "CurrentLimit": ("OperationalLimit.OperationalLimitSet",),
            "PhotoVoltaicUnit": ("PowerElectronicsUnit.PowerElectronicsConnection",),
            "PowerElectronicsWindUnit": ("PowerElectronicsUnit.PowerElectronicsConnection",),
            "BatteryUnit": ("PowerElectronicsUnit.PowerElectronicsConnection",),
            # a DC line's parts, held by the DC line (the element)
            "VsConverter": ("Equipment.EquipmentContainer",), "DCLineSegment": ("Equipment.EquipmentContainer",),
            "DCNode": ("DCNode.DCEquipmentContainer",),
            "DCTopologicalNode": ("DCTopologicalNode.DCEquipmentContainer",),
            "ACDCConverterDCTerminal": ("ACDCConverterDCTerminal.DCConductingEquipment",),
            "DCTerminal": ("DCTerminal.DCConductingEquipment",)}
# Referred to by the equipment it belongs to (the reference is on the equipment's side).
_REFERRED_BY = {"RegulatingControl": "RegulatingCondEq.RegulatingControl", "GeneratingUnit": "RotatingMachine.GeneratingUnit"}
_BUS_COLUMNS = {"line": ("from_bus", "to_bus"), "trafo": ("hv_bus", "lv_bus"), "trafo3w": ("hv_bus", "mv_bus", "lv_bus"),
                "impedance": ("from_bus", "to_bus"), "switch": ("bus",)}


def _ref(value) -> str | None:
    return value[1:] if isinstance(value, str) and value.startswith("#") else None


def _merged(objects: dict) -> dict[str, dict]:
    """One object's statements across profiles (an end's EQ and SC parts share its mRID), merged; its EQ class."""
    out: dict[str, dict] = {}
    for (profile, mrid), entry in objects.items():
        merged = out.setdefault(mrid, {"class": entry["class"], "props": {}})
        if profile == "EQ" and not entry["about"]:
            merged["class"] = entry["class"]
        merged["props"].update(entry["props"])
    return out


def _roots(objects: dict) -> dict[str, str]:
    """{mRID: the mRID of what it belongs to} — a terminal, an end, a tap changer, a limit, a regulating control or a
    generating unit belongs to its equipment; anything else (equipment, nodes, containers) to itself."""
    merged = _merged(objects)
    referrer: dict[str, str] = {}
    for mrid, entry in merged.items():
        for attr in _REFERRED_BY.values():
            target = _ref(entry["props"].get(f"cim:{attr}"))
            if target:
                referrer[target] = mrid
    parent: dict[str, str] = {}
    for mrid, entry in merged.items():
        cls = entry["class"].split(":")[-1]
        if cls == "DCLineSegment":                      # a DC line's container belongs to its segment (the element)
            box = _ref(entry["props"].get("cim:Equipment.EquipmentContainer"))
            if box:
                parent[box] = mrid
            continue
        for attr in _PART_OF.get(cls, ()):
            target = _ref(entry["props"].get(f"cim:{attr}"))
            if target:
                parent[mrid] = target
                break
        else:
            if cls in _REFERRED_BY and mrid in referrer:
                parent[mrid] = referrer[mrid]
    root: dict[str, str] = {}

    def find(mrid: str, depth: int = 0) -> str:
        if mrid not in root:
            up = parent.get(mrid)
            root[mrid] = mrid if up is None or depth > 6 else find(up, depth + 1)
        return root[mrid]
    for mrid in merged:
        find(mrid)
    return root


def _index(model: dict) -> dict:
    """What each object of a model belongs to, and each owner's objects: {"root": {mRID: owner}, "keys": {owner:
    [(profile, mRID)…]}}. Kept on the model (a base's statements are reused for every proposal)."""
    kept = model.get("_index")
    if kept is None:
        root = _roots(model["objects"])
        keys: dict[str, list] = {}
        for key in model["objects"]:
            keys.setdefault(root.get(key[1], key[1]), []).append(key)
        kept = model["_index"] = {"root": root, "keys": keys}
    return kept


def _touched(base: dict, diff: dict) -> tuple[set[str], dict[str, str]]:
    """The owners (equipment, nodes, containers) a difference touches, and the owner of each object it adds."""
    idx = _index(base)
    added = {(a["profile"], a["mrid"]): a for a in diff.get("added") or []}
    for c in diff.get("changed") or []:
        if (c["profile"], c["mrid"]) not in base["objects"]:
            added[(c["profile"], c["mrid"])] = {**c, "props": dict(c.get("set") or {})}
    # An added part can belong to an existing owner (a new limit on an existing line): resolved with the base's.
    context = dict(added)
    for a in list(added.values()):
        for v in a["props"].values():
            target = _ref(v)
            if target and target not in {k[1] for k in added}:
                for key in idx["keys"].get(idx["root"].get(target, target), []):
                    if key[1] == target:
                        context[key] = base["objects"][key]
    added_root = _roots(context)
    owners = set()
    for (profile, mrid) in added:
        r = added_root.get(mrid, mrid)
        owners.add(idx["root"].get(r, r))
    for item in (diff.get("changed") or []) + (diff.get("removed") or []):
        if (item["profile"], item["mrid"]) in base["objects"]:
            owners.add(idx["root"].get(item["mrid"], item["mrid"]))
    owners |= set(diff.get("in_service") or {})
    return owners, {m: idx["root"].get(added_root.get(m, m), added_root.get(m, m)) for (_p, m) in added}


def _sub_model(base: dict, diff: dict, owners: set[str], added_root: dict[str, str]) -> dict:
    """The part of the base model with the difference applied that `owners` need: their objects, the nodes their
    terminals connect to, and the containers all of these refer to — what the adapter reads to give their rows."""
    idx = _index(base)
    keys = set()
    for owner in owners:
        keys |= set(idx["keys"].get(owner, ()))
    added = {(a["profile"], a["mrid"]): a for a in diff.get("added") or []}
    keys |= {k for k in added if added_root.get(k[1], k[1]) in owners}
    for c in diff.get("changed") or []:
        if added_root.get(c["mrid"], idx["root"].get(c["mrid"], c["mrid"])) in owners:
            keys.add((c["profile"], c["mrid"]))
    removed = {(r["profile"], r["mrid"]) for r in diff.get("removed") or []}
    changed = {(c["profile"], c["mrid"]): c for c in diff.get("changed") or []}

    def entry(key):
        if key in removed:
            return None
        if key in added:
            a = added[key]
            return {"class": a["class"], "about": a["about"], "props": dict(a["props"])}
        if key in base["objects"]:
            e = base["objects"][key]
            out = {"class": e["class"], "about": e["about"], "props": dict(e["props"])}
        elif key in changed:
            out = {"class": changed[key]["class"], "about": changed[key]["about"], "props": {}}
        else:
            return None
        c = changed.get(key)
        if c:
            out["class"] = c["class"]
            for k in c.get("unset") or []:
                out["props"].pop(k, None)
            out["props"].update(c.get("set") or {})
        return out
    objects = {k: e for k in keys if (e := entry(k)) is not None}
    # The nodes and containers they refer to (and the containers' own containers), from the base or the difference.
    pending = [v for e in objects.values() for v in e["props"].values()]
    seen = {k[1] for k in objects}
    while pending:
        target = _ref(pending.pop())
        if not target or target in seen:
            continue
        seen.add(target)
        candidates = [k for k in idx["keys"].get(target, ()) if k[1] == target] + [k for k in added if k[1] == target]
        for key in candidates:
            e = entry(key)
            if e is None:
                continue
            cls = e["class"].split(":")[-1]
            if cls in _CONTAINERS or cls in ("TopologicalNode", "ConnectivityNode"):
                objects[key] = e
                pending.extend(e["props"].values())
    in_service = {**{m: v for m, v in base["in_service"].items() if m in seen}, **(diff.get("in_service") or {})}
    return {"objects": objects, "general": {**base["general"], **(diff.get("general") or {})},
            "version": diff.get("version") or base["version"],
            "in_service": {m: v for m, v in in_service.items() if m in seen}}


def project_onto(base_net, base: dict, diff: dict, network_id: str, name: str = "proposal"):
    """The proposal's network with the base's pandapower layout: a copy of `base_net` with the difference applied
    element by element, by mRID — an element changed takes its new values, one added is appended, one removed is
    dropped; every other row keeps its index. The new values are what the adapter reads from the part of the model
    the difference touches (cim_adapter: the same as the edits', checked by the round trip), so the cost follows the
    size of the change, not of the network. Returns (network, notes)."""
    import pandas as pd

    import cgmes_writer as cw
    import cim_adapter as ca
    owners, added_root = _touched(base, diff)
    net = copy.deepcopy(base_net)
    if not owners:
        return net, []
    sub = _sub_model(base, diff, owners, added_root)
    adapter = ca.PandapowerAdapter()
    projected, pmap, notes = adapter.project(package(sub, network_id, name), network_id, partial=True)
    bmap = ca.ElementMap.of(net, network_id)
    touched = {m for m in owners if pmap.element(m) or bmap.element(m)}
    # The same part of the model without the difference, read the same way: an element the difference changes takes
    # only the values that differ between the two readings — the rest of its row stays the base's (what the reader
    # adds of its own, its defaults, its rounding never reach it).
    existing = {m for m in owners if bmap.element(m)}
    empty_diff = {"added": [], "changed": [], "removed": [], "in_service": {}, "general": {},
                  "version": base["version"]}
    before_sub = _sub_model(base, empty_diff, existing, {}) if existing else None
    if before_sub and before_sub["objects"]:
        projected_before, pbmap, _n = adapter.project(package(before_sub, network_id, name), network_id,
                                                      partial=True)
    else:
        projected_before, pbmap = None, ca.ElementMap({})
    # A line's or transformer's end a switch stands at is a node of the switch's own in the model: the base's bus.
    aux_bus: dict[str, int] = {}
    if len(getattr(net, "switch", [])):
        for i in net.switch.index[net.switch["et"].astype(str).isin(("l", "t"))]:
            aux_bus[cw._id(bmap.mrid("switch", i), "node")] = int(net.switch.at[i, "bus"])
    bus_of: dict[int, int] = {}                 # projected bus -> base bus
    for mrid, (table, idx) in pmap.by_mrid.items():
        if table == "bus":
            if bmap.element(mrid):
                bus_of[idx] = bmap.element(mrid)[1]
            elif mrid in aux_bus:
                bus_of[idx] = aux_bus[mrid]

    def base_bus(p_bus) -> int:
        """The base's bus for a projected one: a bus the proposal adds is appended."""
        p_bus = int(p_bus)
        if p_bus not in bus_of:
            new = int(net.bus.index.max()) + 1 if len(net.bus) else 0
            for c, v in values("bus", p_bus).items():
                if c not in net.bus.columns:
                    _new_column(net.bus, c)
                net.bus.at[new, c] = v
            bus_of[p_bus] = new
        return bus_of[p_bus]

    def read(frame, model: dict, emap, table: str, idx: int) -> dict:
        """A projected row's values as the computation takes them (a column its extras leave empty, empty)."""
        row = frame[table].loc[idx]
        xt = model["objects"].get((EXTRAS, emap.mrid(table, idx)), {}).get("props", {})
        empty = set(json.loads(xt["unset"])) if "unset" in xt else set()
        # The base's columns, CONDUCTOR's own, and any other the element's extras state (a flag pandapower has).
        stated = [c for c in xt if c in row.index and c != "unset"]
        wanted = list(dict.fromkeys(list(net[table].columns) + [c for c in (*_KEPT_COLUMNS, *stated)
                                                                  if c in row.index]))
        out = {}
        for c in wanted:
            if _provenance(c):
                continue                        # the reader's own bookkeeping: never the element's value
            if c in empty:
                out[c] = float("nan")
            elif c in row.index and row[c] is not None and not (isinstance(row[c], float) and row[c] != row[c]):
                out[c] = row[c]
        # Buses by mRID: each reading numbers its own.
        for c in _BUS_COLUMNS.get(table, ("bus",)) + (("element",) if table == "switch" and out.get("et") == "b"
                                                       else ()):
            if c in out:
                out[c] = ("bus", emap.mrid("bus", int(out[c])))
        return out

    def values(table: str, p_idx: int, base_row: bool = False) -> dict:
        now = read(projected, sub, pmap, table, p_idx)
        mrid = pmap.mrid(table, p_idx)
        hit = pbmap.element(mrid) if base_row else None
        if hit and hit[0] == table:
            # An element the base has: what the difference changes, nothing else.
            was = read(projected_before, before_sub, pbmap, table, hit[1])
            out = {c: v for c, v in now.items() if not (c in was and (v == was[c] if isinstance(v, tuple)
                                                                       else _same(was[c], v, c, table, p_idx)))}
            out.update({c: float("nan") for c in was if c not in now})
        else:
            # An element added (or moved to another table): pandapower's own columns for it and what its extras
            # state — not what the reader adds of its own.
            standard = _standard_columns(table)
            xt = sub["objects"].get((EXTRAS, mrid), {}).get("props", {})
            out = {c: v for c, v in now.items() if c in standard or c in _KEPT_COLUMNS or c in xt}
            # What pandapower's create_* fills in on its own when the table has the column.
            for c, v in _created_defaults(table, tuple(net[table].columns)).items():
                out.setdefault(c, v)
        for c, v in list(out.items()):
            if isinstance(v, tuple):
                out[c] = base_bus(pmap.element(v[1])[1])
        return out
        if table == "switch" and base_row:
            for c in ("bus", "element", "et"):         # what it switches is the base's (the model gives it a node)
                out.pop(c, None)
        return out
    for mrid in sorted(m for m in touched if (pmap.element(m) or bmap.element(m) or ("",))[0] == "bus"):
        p, b = pmap.element(mrid), bmap.element(mrid)
        if p and b:
            for c, v in values("bus", p[1], base_row=True).items():
                net.bus.at[b[1], c] = v
        elif p:
            base_bus(p[1])
        elif b:
            net.bus = net.bus.drop(index=b[1])
    moved: dict[tuple[str, int], tuple[str, int]] = {}        # an element's base row -> its row now
    placed: dict[str, tuple[str, int]] = {}
    for mrid in sorted(touched):
        p, b = pmap.element(mrid), bmap.element(mrid)
        table = (p or b)[0]
        if table == "bus":
            continue
        if b and (p is None or p[0] != b[0]):
            net[b[0]] = net[b[0]].drop(index=b[1])
            moved[b] = None
            b0, b = b, None
        else:
            b0 = b
        if p is None:
            continue
        row = values(p[0], p[1], base_row=b is not None)
        for c in row:
            if c not in net[p[0]].columns:
                _new_column(net[p[0]], c, row[c])
        df = net[p[0]]
        if b:
            for c, v in row.items():
                if not _same(df.at[b[1], c] if c in df.columns else None, v, c, p[0], b[1]):
                    df.at[b[1], c] = _as_column(df, c, v)
            placed[mrid] = b
        else:
            new = int(df.index.max()) + 1 if len(df) else 0
            xt = sub["objects"].get((EXTRAS, mrid), {}).get("props", {})
            kept = json.loads(xt["row"]) if "row" in xt else None
            if kept is not None and int(kept) not in df.index:
                new = int(kept)                         # the row it had where it was made (added, or moved)
            if b0:
                # A unit moved to the other table keeps its identity as loaded (as generators.py carries it).
                for c in ("origin_id", "origin_class"):
                    old = base_net[b0[0]].at[b0[1], c] if c in base_net[b0[0]].columns else None
                    if isinstance(old, str) and old:
                        row[c] = old
                        if c not in df.columns:
                            _new_column(df, c)
            for c, v in {**{c: False for c in df.columns if c in _FLAGS}, **row}.items():
                df.at[new, c] = _as_column(df, c, v)    # a flag the model does not state: False, as pandapower
            placed[mrid] = (p[0], new)
            if b0:
                moved[b0] = (p[0], new)
    _carry_costs(net, moved)
    # Types the power flow relies on: an appended row leaves object columns behind. Each column gets back the
    # base table's type (a flag False where the model states none, as pandapower creates it).
    for table in {(pmap.element(m) or bmap.element(m))[0] for m in touched} | {"bus"}:
        if table not in net:
            continue
        df, before = net[table], base_net[table] if table in base_net else None
        for c in df.columns:
            kind = before[c].dtype if before is not None and c in before.columns else None
            if kind is None:
                # A column the change brings: numbers stay numbers (pandapower creates it as floats).
                filled = [v for v in df[c] if not _empty(v)]
                if filled and all(isinstance(v.item() if hasattr(v, "item") else v, (int, float))
                                  and not isinstance(v.item() if hasattr(v, "item") else v, bool) for v in filled):
                    df[c] = pd.to_numeric(df[c], errors="coerce")
                continue
            if pd.api.types.is_bool_dtype(kind):
                if c in _FLAGS or c == "in_service" or not any(_empty(v) for v in df[c]):
                    df[c] = [bool(v) if not _empty(v) else (c == "in_service") for v in df[c]]
            elif pd.api.types.is_integer_dtype(kind):
                if df[c].notna().all():
                    df[c] = df[c].astype("int64")
            elif pd.api.types.is_float_dtype(kind):
                df[c] = pd.to_numeric(df[c], errors="coerce")
    return net, notes


def _carry_costs(net, moved: dict) -> None:
    """A unit the model moved to another table (its control changed) takes its cost rows along, as generators.py
    moves them; an element removed leaves none behind. Edits never change a cost: the others stay as they are."""
    import pandas as pd
    for table in ("poly_cost", "pwl_cost"):
        df = net[table] if table in net and isinstance(net[table], pd.DataFrame) else None
        if df is None or not len(df):
            continue
        for (old_table, old_idx), new in moved.items():
            rows = (df["et"].astype(str) == old_table) & (df["element"].astype(int) == int(old_idx))
            if not rows.any():
                continue
            if new is None:
                net[table] = df = df[~rows]
            else:
                df.loc[rows, "et"] = new[0]
                df.loc[rows, "element"] = int(new[1])


# What pandapower's CIM reader adds to a row of its own: where it came from in the model (identifiers, classes,
# containers, terminals) — the reader's bookkeeping, not the element's values.
_READER_OWN = frozenset({"origin_id", "origin_class", "origin_profile", "terminal", "cim_topnode", "description",
                         "tapchanger_class", "tapchanger2_class", "RegulatingControl.mode", "referencePriority",
                         "converters", "converter_dups"})


def _standard_columns(table: str) -> frozenset:
    """pandapower's own columns for a table (an empty network's), kept once."""
    kept = _STANDARD.get(table)
    if kept is None:
        import pandapower as pp
        empty = pp.create_empty_network()
        kept = _STANDARD[table] = frozenset(empty[table].columns) if table in empty else frozenset()
    return kept


_STANDARD: dict[str, frozenset] = {}
_DEFAULTS: dict[tuple, dict] = {}


def _created_defaults(table: str, columns: tuple) -> dict:
    """The values pandapower's create_* gives a new row on its own in a table with these columns (a static
    generator's `generator_type`, say), found by creating one in an empty network with them; kept per table and
    columns."""
    key = (table, columns)
    if key in _DEFAULTS:
        return _DEFAULTS[key]
    import pandapower as pp
    out: dict = {}
    makers = {"load": lambda n: pp.create_load(n, 0, p_mw=0.0), "sgen": lambda n: pp.create_sgen(n, 0, p_mw=0.0),
              "gen": lambda n: pp.create_gen(n, 0, p_mw=0.0), "shunt": lambda n: pp.create_shunt(n, 0, q_mvar=0.0),
              "storage": lambda n: pp.create_storage(n, 0, p_mw=0.0, max_e_mwh=1.0),
              "line": lambda n: pp.create_line_from_parameters(n, 0, 1, 1.0, 0.1, 0.1, 0.0, 1.0),
              "trafo": lambda n: pp.create_transformer_from_parameters(n, 0, 1, 1.0, 110.0, 20.0, 0.1, 10.0, 0.0,
                                                                        0.0)}
    if table in makers:
        try:
            n = pp.create_empty_network()
            pp.create_bus(n, 110.0)
            pp.create_bus(n, 20.0)
            extra = [c for c in columns if c not in n[table].columns]
            for c in extra:
                n[table][c] = pd.Series(dtype=object)
            idx = makers[table](n)
            out = {c: n[table].at[idx, c] for c in extra if not _empty(n[table].at[idx, c])}
        except Exception:  # noqa: BLE001 — no defaults then
            out = {}
    _DEFAULTS[key] = out
    return out


def _provenance(column: str) -> bool:
    return (column in _READER_OWN or column.startswith("terminal_")
            or ((column.endswith("_id") or column.endswith("_name")) and column[:1].isupper())
            or column.startswith("PowerTransformerEnd_id") or column.startswith("tapchanger"))


def _empty(v) -> bool:
    return v is None or (isinstance(v, float) and v != v) or str(v) in ("nan", "<NA>")


def _as_column(df, column: str, value):
    """A value in the type of the column it goes into (a flag read back as 0.0 into a boolean column: False)."""
    if column in df.columns and df[column].dtype == bool and isinstance(value, (int, float)):
        return bool(value)
    return value


# Columns pandapower fills with False on every row when an element that has them is created; any other column a
# created element brings is empty on the other rows.
_FLAGS = frozenset({"proposed", "oltc", "tap_dependency_table", "reactive_capability_curve", "slack"})


def _new_column(df, column: str, sample=None) -> None:
    """A column the base's table lacks, as pandapower and the sandbox add it."""
    df[column] = False if column in _FLAGS else None
    df[column] = df[column].astype(object)


def _same(old, new, column: str, table: str, idx) -> bool:
    """Whether a projected value says nothing new about a base row: equal (False and 0.0 alike, floats to 1e-9), or
    the writer's stand-in name for a row the network leaves unnamed."""
    import math
    if column == "name" and (old is None or (isinstance(old, float) and math.isnan(old))) and new == f"{table} {int(idx)}":
        return True
    old, new = (v.item() if hasattr(v, "item") else v for v in (old, new))
    if isinstance(old, float) and isinstance(new, float) and old != old and new != new:
        return True
    try:
        if isinstance(old, (bool, int, float)) and isinstance(new, (bool, int, float)):
            return math.isclose(float(old), float(new), rel_tol=1e-8, abs_tol=1e-12)
    except (TypeError, ValueError):
        return False
    return str(old) == str(new)


# Columns a row the proposal adds brings even where the base's table has none: CONDUCTOR's own (its marks and
# profile) and the OPF limits.
_KEPT_COLUMNS = ("proposed", "profile", "profile_spec", "override", "substation_name", "loaded_name", "loaded_unit",
                 "controllable",
                 "min_p_mw", "max_p_mw", "min_q_mvar", "max_q_mvar", "max_loading_percent", "scale_factor",
                 "bus_measured", "bus_moved", "_unscaled_p_mw", "_unscaled_q_mvar", "_q_from_meas", "loaded_table",
                 "loaded_index")



# -- edits translated into CIM directly ---------------------------------------------------------------------------------

class NotTranslated(Exception):
    """An edit with no direct CIM translation yet: the proposal's difference is then taken via its edits."""


def _part(work: dict, *parts) -> str:
    """A part's id in the model being changed: the model's own where it has one (`own_ids`), else the writer's."""
    import cgmes_writer as cw
    derived = cw._id(*parts)
    return (work.get("parts") or {}).get(derived, derived)


def _xt(work: dict, mrid: str, table: str) -> dict:
    return work["objects"].setdefault((EXTRAS, mrid), {"class": table, "about": True, "props": {}})["props"]


def _xt_get(props: dict, key: str):
    v = props.get(key)
    return json.loads(v) if v is not None else None


def _xt_set(props: dict, key: str, value) -> None:
    props[key] = json.dumps(value, sort_keys=True, default=str)


def _props(work: dict, profile: str, mrid: str, cls: str | None = None) -> dict:
    key = (profile, mrid)
    if key not in work["objects"]:
        if cls is None:
            raise NotTranslated(f"no {profile} statements for {mrid}")
        work["objects"][key] = {"class": cls, "about": True, "props": {}}
    return work["objects"][key]["props"]


def _class(work: dict, mrid: str) -> str:
    entry = work["objects"].get(("EQ", mrid))
    if not entry:
        raise NotTranslated(f"{mrid} is not in the model")
    return entry["class"]


def _element(emap, table: str, idx) -> str:
    mrid = emap.mrid(table, idx)
    if mrid is None:
        raise NotTranslated(f"{table} {idx} is not an element of the base (an element an earlier edit adds)")
    return mrid


def _set_power(work: dict, mrid: str, table: str, p_mw: float | None = None, q_mvar: float | None = None) -> None:
    """An injection's operating point, where the writer states it (SSH, load sign as CGMES has it) and in the
    extras (pandapower's own values)."""
    xt = _xt(work, mrid, table)
    scaling = _xt_get(xt, "scaling") or 1.0
    cls = _class(work, mrid)
    sign = 1.0 if table == "load" else -1.0               # generation is negative in CGMES' load sign
    if table == "storage":
        sign = 1.0                                         # written as a source of -p (charging positive)
    for value, field, cim in ((p_mw, "p_mw", {"EnergyConsumer": "cim:EnergyConsumer.p",
                                              "PowerElectronicsConnection": "cim:PowerElectronicsConnection.p",
                                              "EnergySource": "cim:EnergySource.activePower",
                                              "SynchronousMachine": "cim:RotatingMachine.p"}),
                              (q_mvar, "q_mvar", {"EnergyConsumer": "cim:EnergyConsumer.q",
                                                  "PowerElectronicsConnection": "cim:PowerElectronicsConnection.q",
                                                  "EnergySource": "cim:EnergySource.reactivePower",
                                                  "SynchronousMachine": "cim:RotatingMachine.q"})):
        if value is None:
            continue
        if cls not in cim:
            raise NotTranslated(f"the power of a {cls}")
        _props(work, "SSH", mrid, cls)[cim[cls]] = sign * float(value) * float(scaling)
        if field in ("p_mw", "q_mvar") and (table != "gen" or field in xt):
            _xt_set(xt, field, float(value))
        if table == "gen" and field == "p_mw":
            unit = _part(work, mrid, "unit")
            if ("EQ", unit) in work["objects"]:
                work["objects"][("EQ", unit)]["props"]["cim:GeneratingUnit.initialP"] = float(value)


def _current_power(work: dict, mrid: str, table: str, field: str) -> float:
    xt = _xt(work, mrid, table)
    if _xt_get(xt, field) is not None:
        return float(_xt_get(xt, field))
    cls = _class(work, mrid)
    if table == "gen" and field == "p_mw":
        return -float(work["objects"][("SSH", mrid)]["props"]["cim:RotatingMachine.p"])
    raise NotTranslated(f"the {field} of a {cls}")


def _reinforce_line(work, net, emap, edit, network_id):
    f = float(edit["factor"])
    eq = _element(emap, "line", edit["line"])
    p = _props(work, "EQ", eq)
    for k in ("cim:ACLineSegment.r", "cim:ACLineSegment.x"):
        p[k] = float(p[k]) / f
    for k in ("cim:ACLineSegment.bch", "cim:ACLineSegment.gch"):
        if k in p:
            p[k] = float(p[k]) * f
    limit = _part(work, eq, "patl")
    for profile in ("EQ", "SSH"):
        if (profile, limit) in work["objects"]:
            q = work["objects"][(profile, limit)]["props"]
            for k in ("cim:CurrentLimit.normalValue", "cim:CurrentLimit.value"):
                if k in q:
                    q[k] = float(q[k]) * f
    xt = _xt(work, eq, "line")
    for col, scale in (("r_ohm_per_km", 1 / f), ("x_ohm_per_km", 1 / f), ("c_nf_per_km", f), ("g_us_per_km", f),
                       ("max_i_ka", f)):
        if _xt_get(xt, col) is not None:
            _xt_set(xt, col, float(_xt_get(xt, col)) * scale)


def _upgrade_trafo(work, net, emap, edit, network_id):
    eq = _element(emap, "trafo", edit["trafo"])
    xt = _xt(work, eq, "trafo")
    par = float(_xt_get(xt, "parallel") or 1.0)
    end1 = _props(work, "EQ", _part(work, eq, "end", 1))
    old = float(end1["cim:PowerTransformerEnd.ratedS"]) / par
    new = float(edit["new_sn_mva"])
    f = new / old
    for k in (1, 2):
        props = _props(work, "EQ", _part(work, eq, "end", k))
        props["cim:PowerTransformerEnd.ratedS"] = new * par
        if k == 1:
            for key, scale in (("cim:PowerTransformerEnd.r", 1 / f), ("cim:PowerTransformerEnd.x", 1 / f),
                               ("cim:PowerTransformerEnd.g", f), ("cim:PowerTransformerEnd.b", f)):
                props[key] = float(props[key]) * scale
    if _xt_get(xt, "sn_mva") is not None:
        _xt_set(xt, "sn_mva", new)
    if _xt_get(xt, "pfe_kw") is not None:
        _xt_set(xt, "pfe_kw", float(_xt_get(xt, "pfe_kw")) * f)


_MEASURED = {"load": ("p_mw", "q_mvar"), "sgen": ("p_mw", "q_mvar"), "gen": ("p_mw",)}


def _set_parameter(work, net, emap, edit, network_id):
    table = edit["element_type"]
    eq = _element(emap, table, edit["element"])
    values = edit["values"]
    for field, value in values.items():
        if field in ("p_mw", "q_mvar") and table in ("load", "sgen", "gen", "storage"):
            _set_power(work, eq, table, **{field: value})
        elif field == "in_service" and table in ("line", "trafo", "trafo3w", "load", "sgen", "gen", "shunt",
                                                 "storage", "ext_grid"):
            for seq in (1, 2, 3):
                term = _part(work, eq, "terminal", seq)
                if ("SSH", term) in work["objects"]:
                    work["objects"][("SSH", term)]["props"]["cim:ACDCTerminal.connected"] = bool(value)
            work.setdefault("in_service", {})[eq] = bool(value)
        elif field == "closed" and table == "switch":
            _props(work, "SSH", eq, _class(work, eq))["cim:Switch.open"] = not bool(value)
        else:
            raise NotTranslated(f"set_parameter {table}.{field}")
    measured = {f: v for f, v in values.items() if f in _MEASURED.get(table, ())}
    if measured:
        xt = _xt(work, eq, table)
        old = _xt_get(xt, "override")
        merged = {**(json.loads(old) if isinstance(old, str) and old else {}), **measured}
        _xt_set(xt, "override", json.dumps(merged, sort_keys=True))


def _scale(work, net, emap, edit, network_id):
    table, factor = edit["element_type"], float(edit["factor"])
    for idx in edit["elements"]:
        eq = _element(emap, table, idx)
        xt = _xt(work, eq, table)
        total = float(_xt_get(xt, "scale_factor") or 1.0) * factor
        _xt_set(xt, "scale_factor", total)
        override = _xt_get(xt, "override")
        fixed = json.loads(override) if isinstance(override, str) and override else {}
        for field in _MEASURED[table]:
            if field in fixed:
                continue
            kept = _xt_get(xt, f"_unscaled_{field}")
            base = float(kept) if kept is not None else _current_power(work, eq, table, field)
            _xt_set(xt, f"_unscaled_{field}", base)
            _set_power(work, eq, table, **{field: base * total})


def _move_element(work, net, emap, edit, network_id):
    import cgmes_writer as cw
    import semantics as sm
    table, idx, to_bus = edit["element_type"], edit["element"], int(edit["to_bus"])
    eq = _element(emap, table, idx)
    new_node = _element(emap, "bus", to_bus)
    if table == "line":
        old_bus = int(net.line.at[idx, f"{edit['end']}_bus"])
        seq = 1 if edit["end"] == "from" else 2
    else:
        old_bus = int(net[table].at[idx, "bus"])
        seq = 1
    if float(net.bus.at[old_bus, "vn_kv"]) != float(net.bus.at[to_bus, "vn_kv"]):
        raise NotTranslated("a move to another voltage level")
    term = _part(work, eq, "terminal", seq)
    tp = work["objects"].get(("TP", term))
    if tp is None or _ref(tp["props"].get("cim:Terminal.TopologicalNode")) != _element(emap, "bus", old_bus):
        raise NotTranslated("a move of an end a switch stands at")
    tp["props"]["cim:Terminal.TopologicalNode"] = f"#{new_node}"
    if seq == 1:
        subs, _basis = sm.substations(net)
        level = cw._id(network_id, "voltage_level", subs.get(to_bus) or f"Substation {to_bus}",
                       float(net.bus.at[to_bus, "vn_kv"]))
        if ("EQ", level) not in work["objects"]:
            raise NotTranslated("the new bus's voltage level is not in the model")
        _props(work, "EQ", eq)["cim:Equipment.EquipmentContainer"] = f"#{level}"
    if table in ("load", "sgen", "gen") and net[table].get("proposed", {}).get(idx) is not True:
        xt = _xt(work, eq, table)
        if _xt_get(xt, "bus_measured") is None:
            _xt_set(xt, "bus_measured", old_bus)
        _xt_set(xt, "bus_moved", to_bus)


def _set_switch(work, net, emap, edit, network_id):
    """A switch opened or closed: its SSH position."""
    eq = _element(emap, "switch", edit["switch"])
    _props(work, "SSH", eq, _class(work, eq))["cim:Switch.open"] = not bool(edit["closed"])


TRANSLATORS = {"reinforce_line": _reinforce_line, "upgrade_trafo": _upgrade_trafo, "set_parameter": _set_parameter,
               "scale": _scale, "move_element": _move_element, "set_switch": _set_switch}
_CONTAINERS = frozenset({"Substation", "VoltageLevel", "BaseVoltage", "GeographicalRegion", "SubGeographicalRegion",
                         "EnergySchedulingType", "OperationalLimitType"})
_TABLES = ("bus", "line", "trafo", "trafo3w", "gen", "sgen", "storage", "load", "shunt", "ext_grid", "switch", "ward",
           "xward", "impedance", "motor", "svc", "dcline")


def _snapshot(net) -> dict:
    import pandas as pd
    return {t: net[t].copy() for t in _TABLES if t in net and isinstance(net[t], pd.DataFrame)}


def changed_rows(before: dict, net) -> dict[str, set[int]]:
    """{table: indices} of the rows `net` has new or changed against a snapshot (`_snapshot`) of what it was."""
    import pandas as pd
    out: dict[str, set[int]] = {}
    for t in _TABLES:
        new = net[t] if t in net and isinstance(net[t], pd.DataFrame) else None
        if new is None:
            continue
        old = before.get(t)
        if old is None:
            rows = {int(i) for i in new.index}
        else:
            rows = {int(i) for i in new.index.difference(old.index)}
            common = new.index.intersection(old.index)
            for c in new.columns.union(old.columns):
                if c not in old.columns or c not in new.columns:
                    # A column an edit adds (or drops) says something only where it is set: its default — empty, or
                    # False as pandapower and the sandbox fill a flag — on the other rows is no change.
                    s = (new if c in new.columns else old).loc[common, c].astype(object)
                    differ = ~(s.isna() | s.isin([False]))
                else:
                    a, b = old.loc[common, c].astype(object), new.loc[common, c].astype(object)
                    differ = (a != b) & ~(a.isna() & b.isna())
                rows |= {int(i) for i in common[differ.to_numpy()]}
        if rows:
            out[t] = rows
    return out


def reemit(work: dict, net, rows: dict[str, set[int]], network_id: str, removed: set[str] = frozenset()) -> None:
    """Replace in `work` (statements) the objects of the elements `rows` names with what the writer states for them
    in `net` (their equipment and its parts, their nodes, the containers they need, what CGMES has no place for), and
    drop the objects of the elements in `removed` (mRIDs). One definition of how a row is written as CIM: the
    writer's (cgmes_writer.py)."""
    import semantics as sm
    part = statements(net, network_id, work["version"], only=rows)
    mine = {sm._row_mrid(net, t, i, network_id) for t, idx in rows.items() for i in idx}
    root = _roots(part["objects"])
    idx = _index(work) if "_index" in work else None
    if idx is None:
        owned_root = _roots(work["objects"])
        by_owner: dict[str, list] = {}
        for key in work["objects"]:
            by_owner.setdefault(owned_root.get(key[1], key[1]), []).append(key)
    else:
        by_owner = idx["keys"]
    for owner in mine | set(removed):
        for key in list(by_owner.get(owner, ())):
            work["objects"].pop(key, None)
        work["in_service"].pop(owner, None)
    for key, entry in part["objects"].items():
        cls = entry["class"].split(":")[-1]
        r = root.get(key[1], key[1])
        if r in mine or ((cls in _CONTAINERS or cls in ("TopologicalNode", "ConnectivityNode"))
                         and key not in work["objects"]):
            work["objects"][key] = {"class": entry["class"], "about": entry["about"], "props": dict(entry["props"])}
    for mrid, on in part["in_service"].items():
        if root.get(mrid, mrid) in mine:
            work["in_service"][mrid] = on
    work.pop("_index", None)


def difference_of(base_net, base: dict, other, network_id: str) -> dict:
    """The CIM difference that turns `base_net`'s model (`base`, its statements) into `other`'s — a network made from
    it (the operator's changes applied, say) — written for the rows that differ only: new or changed rows are
    written again, a row gone is removed unless it reappears elsewhere under the same mRID (a unit moved to the
    other table)."""
    import semantics as sm
    import pandas as pd
    rows = changed_rows({t: base_net[t] for t in _TABLES if t in base_net and isinstance(base_net[t], pd.DataFrame)},
                        other)
    kept = {m for t in _TABLES if t in other and isinstance(other[t], pd.DataFrame)
            for m in sm.row_mrids(other, t, network_id).values()}
    gone = set()
    for t in _TABLES:
        if t in base_net and isinstance(base_net[t], pd.DataFrame):
            before = sm.row_mrids(base_net, t, network_id)
            now = other[t].index if t in other and isinstance(other[t], pd.DataFrame) else []
            gone |= {m for i, m in before.items() if i not in now and m not in kept}
    work = {"objects": _Overlay(base["objects"]), "general": dict(base["general"]), "version": base["version"],
            "in_service": dict(base["in_service"]), "_index": _index(base)}
    if rows or gone:
        reemit(work, other, rows, network_id, removed=gone)
    work.pop("_index", None)
    diff = difference(base, work)
    diff["translation"] = {"elements_rewritten": {t: len(i) for t, i in rows.items()}, "removed": len(gone)}
    return diff


def translate(base_net, base: dict, edits: list[dict], network_id: str) -> dict:
    """The proposal's CIM difference from its (canonical) edits. An edit with a translation of its own changes the
    model's statements directly (a line reinforced: its segment's impedance and limit…); any other — equipment added,
    a parameter with no translation yet, a move to another voltage level — is applied to a scratch copy of the network
    as the edits build it, and the elements it changed are written again from it (`reemit`). Never the whole
    network: the cost follows the size of the change."""
    import cim_adapter as ca
    import sandbox as sb
    emap = ca.ElementMap.of(base_net, network_id)
    work = {"objects": _Overlay(base["objects"]), "general": dict(base["general"]), "version": base["version"],
            "in_service": dict(base["in_service"]), "_index": _index(base), "parts": base.get("parts") or {}}
    scratch, rows, done, routes = None, {}, [], {"direct": 0, "rewritten": 0}
    for edit in edits:
        translator = TRANSLATORS.get(edit["op"])
        direct = translator is not None
        if direct:
            try:
                translator(work, base_net, emap, edit, network_id)
            except NotTranslated:
                direct = False
        routes["direct" if direct else "rewritten"] += 1
        if not direct and scratch is None:
            scratch = copy.deepcopy(base_net)
            for earlier in done:
                sb._apply(scratch, earlier)
        if scratch is not None:
            before = _snapshot(scratch) if not direct else None
            sb._apply(scratch, edit)
            if not direct:
                for t, idx in changed_rows(before, scratch).items():
                    rows.setdefault(t, set()).update(idx)
        done.append(edit)
    if rows:
        reemit(work, scratch, rows, network_id)
    work.pop("_index", None)
    diff = difference(base, work)
    # How: edits changed in the model's statements directly, and those whose elements were written again.
    diff["translation"] = {**routes, "elements_rewritten": {t: len(i) for t, i in rows.items()}}
    return diff
