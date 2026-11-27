"""
semantics.py — what each element of the loaded network is, and how we know.

A study reads tables: buses, lines, generators. What an operator means by "the
renewable units", "the 132 kV substation" or "the lines to the external grid"
is not a table, and reading it off one goes wrong: every static generator
taken for a renewable unit (in a converted MATPOWER case, a conventional unit
is a static generator), a per-unit case's 1 kV taken for a voltage, the
external grid's injection taken for the lines that carry it. This module
states, for whatever network is loaded, what each element is — aligned with
the classes of the Common Information Model (IEC 61970) — and, for every fact,
its basis:

- ``stated``   — the network says so (a type column, a name in a CIM import,
                 a proposal's edit, the profile), or the operator declared it;
- ``named``    — read from the element's name ("Wind Park"): a likely reading,
                 said as such;
- ``inferred`` — by a rule named with it (a unit with no active power is a
                 synchronous condenser; buses joined by transformers or bus
                 couplers form a substation);
- ``unknown``  — the network does not say. Never filled in with a guess: an
                 answer, or a study, that needs it says it is unknown.

Nothing here changes the network. The operator's declarations (a unit's
energy source the network does not state) are kept per network, like renames.

Pure functions of a pandapower network: no app state, no HTTP.
"""

from __future__ import annotations

import json
import math
import re
import uuid

import pandas as pd

import generators as gm
from element_names import bus_display_name, clean_label, line_display_name, trafo_display_name

STATED, NAMED, INFERRED, UNKNOWN = "stated", "named", "inferred", "unknown"
BASES = (STATED, NAMED, INFERRED, UNKNOWN)

# A unit's energy source. "conventional" is a unit stated to be dispatchable
# and synchronous-style without its fuel; "unknown" is no statement at all.
SOURCES = ("wind", "solar", "hydro", "bioenergy", "thermal", "nuclear", "storage", "condenser", "conventional",
           "other", "unknown")
# A load's category: what kind of demand it is. "network" is a downstream
# network seen as one load (an MV or LV feeder aggregated at its supply point).
LOAD_CATEGORIES = ("residential", "commercial", "industrial", "commercial_industrial", "rural", "agricultural",
                   "ev_charging", "data_centre", "heat_pump", "network", "other", "unknown")
VARIABLE = frozenset({"wind", "solar"})                       # output follows the weather
RENEWABLE = frozenset({"wind", "solar", "hydro", "bioenergy"})

# CIM classes (IEC 61970-301) for the element kinds a pandapower network holds.
CIM_CLASS = {"bus": "TopologicalNode", "line": "ACLineSegment", "trafo": "PowerTransformer",
             "load": "EnergyConsumer", "ext_grid": "ExternalNetworkInjection", "storage": "BatteryUnit"}
# A unit as distribution CIM (CIM100, as GridAPPS-D uses it) has it: the
# equipment that connects it — a PowerElectronicsConnection for an inverter
# (PV, a wind turbine modelled as a static generator, a battery, a fuel cell),
# a SynchronousMachine for a rotating one — and the unit behind it. A unit
# whose source is unknown keeps the abstract GeneratingUnit: true whatever it is.
_ROTATING_UNIT = {"thermal": "ThermalGeneratingUnit", "bioenergy": "ThermalGeneratingUnit",
                  "hydro": "HydroGeneratingUnit", "nuclear": "NuclearGeneratingUnit",
                  "wind": "WindGeneratingUnit", "solar": "SolarGeneratingUnit", "conventional": "GeneratingUnit"}
_PE_UNIT = {"solar": "PhotovoltaicUnit", "wind": "PowerElectronicsWindUnit", "storage": "BatteryUnit"}
# RegulatingControl modes (CIM RegulatingControlModeKind) for a unit's control.
_CIM_MODE = {"voltage": "voltage", "power_factor": "powerFactor", "fixed_q": "reactivePower"}


def cim_unit(table: str, source: str, inverter: bool = False) -> tuple[str, str | None]:
    """(equipment class, unit class) for a unit of `table` with `source`;
    `inverter`: the network states it is inverter-based (a fuel cell)."""
    if table == "storage" or source == "storage":
        return "PowerElectronicsConnection", "BatteryUnit"
    if table == "sgen" and (source in _PE_UNIT or inverter):
        return "PowerElectronicsConnection", _PE_UNIT.get(source, "PowerElectronicsUnit")
    if source == "condenser":
        return "SynchronousMachine", None
    if source in _ROTATING_UNIT:
        return "SynchronousMachine", _ROTATING_UNIT[source]
    return ("SynchronousMachine" if table == "gen" else "GeneratingUnit"), None

# A type written in a network or an import, as the energy source it states.
# Matched on the whole value: pandapower's own types ("PV", "WP", "CHP diesel"),
# CIM's generating unit classes, a proposal's kind.
_TYPE_PATTERNS = (
    ("solar", r"pv|photovolt\w*|solar\w*|solargeneratingunit|photovoltaicunit"),
    ("wind", r"wp|wind\w*|windgeneratingunit|wind ?park|wind ?farm|wind ?turbine"),
    ("hydro", r"hydro\w*|run[- ]of[- ]river|hydrogeneratingunit|pumped\w*"),
    ("bioenergy", r"bio\w*"),
    ("nuclear", r"nuclear\w*|nucleargeneratingunit"),
    ("storage", r"battery|bess|storage\w*|batteryunit"),
    ("thermal", r"chp\b.*|.*diesel.*|gas\w*|ccgt|ocgt|coal|lignite|oil|thermal\w*|thermalgeneratingunit|"
                r".*fuel ?cell.*|steam|combustion\w*|cogeneration"),
)
# The same words inside a name ("Wind Park 3", "PV roof"): a weaker basis.
_NAME_PATTERNS = (
    ("solar", r"\b(?:pv|photovolt\w*|solar\w*)\b"),
    ("wind", r"\b(?:wind\w*|wp)\b"),
    ("hydro", r"\b(?:hydro\w*|run[- ]of[- ]river)\b"),
    ("bioenergy", r"\b(?:biogas|biomass|bio\w*)\b"),
    ("nuclear", r"\bnuclear\b"),
    ("storage", r"\b(?:battery|bess|storage)\b"),
    ("thermal", r"\b(?:chp|diesel|gas|ccgt|ocgt|coal|lignite|steam|fuel ?cell)\b"),
)
# A load's category, written as its type or inside its name. Never a CIGRE-
# or vendor-style code ("Load R1"): an abbreviation is not a statement.
_LOAD_TYPE_PATTERNS = (
    ("commercial_industrial", r"commercial[ /_&-]*(?:and )?industrial|c ?& ?i"),
    ("residential", r"residential|household|domestic|housing"),
    ("commercial", r"commercial|office|retail|services?|tertiary"),
    ("industrial", r"industr\w*|factory|manufactur\w*"),
    ("rural", r"rural"),
    ("agricultural", r"agricultur\w*|farm|irrigation"),
    ("ev_charging", r"ev|ev[ _-]?charg\w*|electric vehicles?|charging station"),
    ("data_centre", r"data ?cent(?:re|er)s?|datacent(?:re|er)s?"),
    ("heat_pump", r"heat ?pumps?"),
)
_LOAD_NAME_PATTERNS = (
    ("commercial_industrial", r"\bcommercial\W+(?:and\W+)?industrial\b"),
    ("residential", r"\b(?:residential|household|domestic|housing)\b"),
    ("commercial", r"\b(?:commercial|office|retail)\b"),
    ("industrial", r"\b(?:industr\w*|factory|manufactur\w*)\b"),
    ("rural", r"\brural\b"),
    ("agricultural", r"\b(?:agricultur\w*|irrigation)\b"),
    ("ev_charging", r"\b(?:ev[ _-]?charg\w*|electric vehicles?|charging)\b"),
    ("data_centre", r"\b(?:data ?cent(?:re|er)s?|datacent(?:re|er)s?)\b"),
    ("heat_pump", r"\bheat ?pumps?\b"),
    ("network", r"\b(?:[ml]v[ _-]?(?:net|grid|feeders?|rings?)|feeders?)\b"),
)


def _source_of_text(text: str, patterns) -> str | None:
    value = str(text or "").strip().casefold()
    if not value or value in {"none", "nan", "sgen", "gen", "wye", "delta", "sync", "dispatchable", "generator"}:
        return None
    for source, pattern in patterns:
        if (re.fullmatch(pattern, value) if patterns in (_TYPE_PATTERNS, _LOAD_TYPE_PATTERNS)
                else re.search(pattern, value)):
            return source
    return None


def fact(value, basis: str, why: str) -> dict:
    """One fact: its value, its basis and where it comes from."""
    return {"value": value, "basis": basis, "from": why}


# ---------------------------------------------------------------------------
# Units
# ---------------------------------------------------------------------------

def _proposal_kind(tbl, idx) -> str | None:
    spec = tbl.at[idx, "profile_spec"] if "profile_spec" in tbl.columns else None
    if not isinstance(spec, str):
        return None
    try:
        return json.loads(spec).get("kind")
    except ValueError:
        return None


# A source a CIM model states by its generating unit's class.
_MODEL_UNIT_SOURCE = {"ThermalGeneratingUnit": "thermal", "HydroGeneratingUnit": "hydro",
                      "NuclearGeneratingUnit": "nuclear", "WindGeneratingUnit": "wind", "SolarGeneratingUnit": "solar",
                      "PhotoVoltaicUnit": "solar", "PowerElectronicsWindUnit": "wind", "BatteryUnit": "storage"}


def unit_source(net, table: str, idx, declared: dict | None = None, profile: dict | None = None,
                cim=None) -> dict:
    """The energy source of one unit, as a fact (see the module's docstring).
    `declared`: the operator's statements; `profile`: the network profile's
    (`unit_sources`), both keyed by the unit's name as loaded. `cim`: the CIM model the network came from — its
    generating unit's class states the source (a ThermalGeneratingUnit is thermal)."""
    tbl = net[table]
    row = tbl.loc[idx]
    key = gm.unit_key(net, table, idx)
    declared = declared or {}
    if key in declared:
        return fact(declared[key], STATED, "declared by the operator")
    if key in (profile or {}) and str(profile[key]).casefold() in SOURCES:
        return fact(str(profile[key]).casefold(), STATED, "stated in the network's profile")
    kind = _proposal_kind(tbl, idx)
    if kind:
        source = {"pv": "solar", "wind": "wind", "dispatchable": "conventional"}.get(str(kind).casefold())
        if source:
            return fact(source, STATED, "as the proposal that added it describes it")
    if cim is not None:
        gu = _model_generating_unit(cim, _origin(net, table, idx))
        classes = ([gu.get("class")] if gu and gu.get("class") else
                   [u.get("class") for u in (gu or {}).get("units") or []])
        sources = {_MODEL_UNIT_SOURCE.get(str(c).split(":")[-1]) for c in classes} - {None}
        if len(sources) == 1:
            cls = str(classes[0]).split(":")[-1]
            return fact(sources.pop(), STATED, f"the model's generating unit ({cls})")
    for col in ("type", "origin_class", "generating_unit_class"):
        if col in tbl.columns:
            source = _source_of_text(row.get(col), _TYPE_PATTERNS)
            if source:
                return fact(source, STATED, f"its {col!r} in the network ({clean_label(row.get(col))!r})")
    if gm.is_condenser(table, row):
        return fact("condenser", INFERRED, "it has no active power to give: a synchronous condenser")
    for col in ("name", "substation_name"):
        if col in tbl.columns:
            source = _source_of_text(row.get(col), _NAME_PATTERNS)
            if source:
                return fact(source, NAMED, f"its name ({clean_label(row.get(col))!r})")
    return fact("unknown", UNKNOWN, "the network does not say what drives this unit")


def load_key(net, idx) -> str:
    """A load's name as loaded: the key of what the operator declares about it
    (a rename does not change it)."""
    tbl = net.load
    kept = tbl.at[idx, "loaded_name"] if "loaded_name" in tbl.columns else None
    raw = kept if isinstance(kept, str) else tbl.at[idx, "name"] if "name" in tbl.columns else None
    raw = clean_label(raw) if isinstance(raw, str) else ""
    return raw or f"Load {int(idx)}"


def load_category(net, idx, declared: dict | None = None, profile: dict | None = None) -> dict:
    """What kind of demand one load is, as a fact: declared by the operator,
    stated by the profile or a type column, read from its name, or unknown."""
    tbl = net.load
    key = load_key(net, idx)
    if key in (declared or {}):
        return fact(declared[key], STATED, "declared by the operator")
    if key in (profile or {}) and str(profile[key]).casefold() in LOAD_CATEGORIES:
        return fact(str(profile[key]).casefold(), STATED, "stated in the network's profile")
    for col in ("category", "type", "load_type", "customer_class"):
        if col in tbl.columns:
            cat = _source_of_text(tbl.at[idx, col], _LOAD_TYPE_PATTERNS)
            if cat:
                return fact(cat, STATED, f"its {col!r} in the network ({clean_label(tbl.at[idx, col])!r})")
    for col in ("name", "substation_name"):
        if col in tbl.columns:
            cat = _source_of_text(tbl.at[idx, col], _LOAD_NAME_PATTERNS)
            if cat:
                return fact(cat, NAMED, f"its name ({clean_label(tbl.at[idx, col])!r})")
    return fact("unknown", UNKNOWN, "the network does not say what kind of demand this is")


def _unit_rows(net):
    for table in ("gen", "sgen"):
        for idx in net[table].index:
            yield table, idx


def _control(net, table: str, idx) -> dict:
    if table == "gen":
        return fact("voltage", STATED, "it holds a voltage set-point (a PV unit)")
    row = net.sgen.loc[idx]
    if gm.QSET_COL in net.sgen.columns and pd.notna(row.get(gm.QSET_COL)):
        return fact("fixed_q", STATED, "it holds a set reactive output")
    return fact("power_factor", STATED, "it injects at a set power factor (a PQ unit)")


# ---------------------------------------------------------------------------
# Buses, substations and voltage levels
# ---------------------------------------------------------------------------

def per_unit_case(net) -> bool:
    """Every bus at 1 kV: a converted per-unit case whose voltages are placeholders."""
    vn = net.bus["vn_kv"] if len(net.bus) else []
    return len(vn) > 1 and bool((pd.Series(vn).astype(float).round(6) == 1.0).all())


def _components(n: int, edges) -> list[int]:
    parent = list(range(n))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for a, b in edges:
        ra, rb = root(a), root(b)
        if ra != rb:
            parent[ra] = rb
    return [root(i) for i in range(n)]


def substations(net, cim=None) -> tuple[dict, str]:
    """{bus index: substation name}, and the basis. Stated when the network
    names them (a 'substation' column, as a CIM import writes); otherwise the
    buses joined by transformers or bus couplers (bus–bus switches), named after
    the highest-voltage bus among them — a single bus is a substation of its own.
    `cim`: the CIM model as loaded, whose Substations give their own names."""
    if "substation" in net.bus.columns and net.bus["substation"].notna().any():
        named = {int(i): clean_label(v) for i, v in net.bus["substation"].items() if clean_label(v)}
        if len(named) == len(net.bus):
            return named, STATED
    # A CIM import states each bus's Substation by identifier (pandapower's
    # `Substation_id`): the membership is stated; the name, the highest-voltage
    # bus's, is ours.
    if "Substation_id" in net.bus.columns and net.bus["Substation_id"].notna().mean() >= 0.5:
        # Boundary nodes (an imported model's ties to neighbours) carry none: each is a substation of its own.
        groups: dict = {}
        for i, sid in net.bus["Substation_id"].items():
            groups.setdefault(cm_norm(sid) if isinstance(sid, str) and sid.strip() else f"node:{int(i)}",
                              []).append(int(i))
        own = _model_substation_names(net, cim)
        out, used = {}, set(own.values())
        for sid, members in groups.items():
            head = max(members, key=lambda b: (float(net.bus.at[b, "vn_kv"]), -b))
            name = own.get(sid)
            if not name:
                # Named after its highest-voltage bus; two substations whose buses share a name stay two.
                name = base = f"Substation {bus_display_name(net, head)}"
                k = 2
                while name in used:
                    name, k = f"{base} ({k})", k + 1
                used.add(name)
            out.update({b: name for b in members})
        return out, STATED
    order = [int(i) for i in net.bus.index]
    pos = {b: k for k, b in enumerate(order)}
    edges = []
    if len(net.trafo):
        edges += [(pos[int(a)], pos[int(b)]) for a, b in zip(net.trafo.hv_bus, net.trafo.lv_bus)
                  if pd.notna(a) and pd.notna(b) and int(a) in pos and int(b) in pos]
    if len(getattr(net, "switch", [])):
        bb = net.switch[net.switch.et == "b"]
        edges += [(pos[int(a)], pos[int(b)]) for a, b in zip(bb.bus, bb.element)
                  if pd.notna(a) and pd.notna(b) and int(a) in pos and int(b) in pos]
    roots = _components(len(order), edges)
    groups: dict[int, list[int]] = {}
    for k, r in enumerate(roots):
        groups.setdefault(r, []).append(order[k])
    out = {}
    for members in groups.values():
        head = max(members, key=lambda b: (float(net.bus.at[b, "vn_kv"]), -b))
        name = bus_display_name(net, head) if len(members) == 1 else f"Substation {bus_display_name(net, head)}"
        out.update({b: name for b in members})
    return out, INFERRED


# ---------------------------------------------------------------------------
# The whole network
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Switches, feeders and substations as CIM has them
# ---------------------------------------------------------------------------

# pandapower's switch types as the CIM Switch subclasses they are.
_SWITCH_CLASS = {"CB": "Breaker", "LBS": "LoadBreakSwitch", "LS": "LoadBreakSwitch", "DS": "Disconnector"}
# CIM's Switch subclasses (IEC 61970-301), as a CIM import's `origin_class` names them.
_CIM_SWITCHES = frozenset({"Switch", "Breaker", "Disconnector", "LoadBreakSwitch", "Fuse", "Jumper",
                           "GroundDisconnector", "Recloser", "Sectionaliser", "ProtectedSwitch",
                           "DisconnectingCircuitBreaker", "Cut"})
_FEEDER_RULE = ("inferred from the topology as switched now: the first line after the last transformer on the "
                "route from the external grid; only where the voltage level is radial")


def _level_islands(graph):
    """{bus: (island id, radial)} of the network without its transformers: a
    voltage level's connected parts, and whether each has no loop."""
    import networkx as nx
    level = graph.copy()
    level.remove_edges_from([(a, b, k) for a, b, k in graph.edges(keys=True) if k[0] in ("trafo", "trafo3w")])
    out = {}
    for n, comp in enumerate(nx.connected_components(nx.Graph(level))):
        sub = nx.Graph(level.subgraph(comp))
        radial = sub.number_of_edges() - sub.number_of_nodes() + 1 <= 0
        out.update({b: (n, radial) for b in comp})
    return out


def feeders_of(net, names: dict) -> tuple[dict, list[int]]:
    """{bus: (feeder, why not)} — the head line's name, or None with the reason:
    at a substation's busbar, or in a meshed voltage level — and the external
    grids' buses. One shortest-path tree per external grid; the nearest wins."""
    import networkx as nx
    graph = _graph(net)
    sources = [int(b) for b, on in zip(net.ext_grid.bus, net.ext_grid.in_service) if bool(on) and int(b) in graph]
    islands = _level_islands(graph)
    best: dict = {}
    for s in sources:
        for bus, path in nx.single_source_shortest_path(graph, s).items():
            if bus not in best or len(path) < len(best[bus]):
                best[bus] = path
    out = {}
    for bus, path in best.items():
        keys = [next(iter(graph.get_edge_data(a, b))) for a, b in zip(path, path[1:])]
        last_trafo = max((k for k, key in enumerate(keys) if key[0] in ("trafo", "trafo3w")), default=-1)
        head = next((key for key in keys[last_trafo + 1:] if key[0] == "line"), None)
        radial = islands.get(bus, (None, False))[1]
        out[int(bus)] = (names.get(("line", int(head[1]))) if head is not None and radial else None,
                         None if head is not None and radial else
                         "its voltage level is meshed: power comes over several routes, so no single feeder"
                         if not radial else "at the substation's busbar, before any feeder")
    return out, sources


def _switch_entries(net, names: dict, buses: dict, feeder: dict, normal_open: dict | None) -> list[dict]:
    """Every switch as CIM has it: its class, its normal and its present state,
    what it switches and its role between feeders."""
    out = []
    sw = getattr(net, "switch", None)
    if sw is None or not len(sw):
        return out
    for i, row in sw.iterrows():
        i = int(i)
        raw_type = clean_label(row.get("type")) if "type" in sw.columns else ""
        cls = _SWITCH_CLASS.get(raw_type.upper())
        own_class = row.get("origin_class") if "origin_class" in sw.columns else None
        if isinstance(own_class, str) and own_class in _CIM_SWITCHES:
            # A CIM import keeps the class the model gives (a Fuse, a Recloser…), finer than pandapower's type.
            cim = fact(own_class, STATED, "its class in the CIM model")
        else:
            cim = (fact(cls, STATED, f"its type in the network ({raw_type!r})") if cls else
                   fact("Switch", UNKNOWN, "the network does not say what kind of switch it is"))
        is_open = not bool(row["closed"])
        if normal_open is not None and i in normal_open:
            normal = fact(bool(normal_open[i]), STATED, "its state as the network was loaded")
        else:
            normal = fact(is_open, STATED, "its state as switched now (no state as loaded is kept for it)")
        if pd.isna(row["bus"]) or int(row["bus"]) not in buses:
            continue                                   # at a node the network does not have
        bus = int(row["bus"])
        line_ends = (net.line.at[int(row["element"]), "from_bus"], net.line.at[int(row["element"]), "to_bus"]) \
            if row["et"] == "l" and int(row["element"]) in net.line.index else (None, None)
        if row["et"] == "l" and int(row["element"]) in net.line.index and not (
                pd.notna(line_ends[0]) and pd.notna(line_ends[1]) and int(line_ends[0]) in buses
                and int(line_ends[1]) in buses):
            where, role = {"kind": "line", "element": None, "at": buses[bus]["name"]}, \
                "line switch (the line's other end is not in the network)"
        elif row["et"] == "l" and int(row["element"]) in net.line.index:
            line = int(row["element"])
            a, b = int(net.line.at[line, "from_bus"]), int(net.line.at[line, "to_bus"])
            far = b if bus == a else a
            where = {"kind": "line", "element": names.get(("line", line)), "at": buses[bus]["name"]}
            fa, fb = feeder.get(bus, (None,))[0], feeder.get(far, (None,))[0]
            if not normal["value"]:
                role = f"sectionalising switch on feeder {fa}" if fa else "line switch"
            elif fa and fb and fa != fb:
                role = f"tie between feeders {fa} and {fb}"
            elif fa and fa == fb:
                role = f"normally open point closing a loop within feeder {fa}"
            else:
                role = "normally open line switch"
        elif row["et"] == "b":
            other = int(row["element"])
            where = {"kind": "coupler", "between": [buses[bus]["name"], buses.get(other, {}).get("name", other)],
                     "between_index": [bus, other]}
            role = "bus coupler"
        elif row["et"] == "t":
            where = {"kind": "transformer", "element": names.get(("trafo", int(row["element"]))),
                     "at": buses[bus]["name"]}
            role = "transformer switch"
        else:
            where, role = {"kind": str(row["et"]), "at": buses[bus]["name"]}, "switch"
        name = clean_label(row.get("name")) if "name" in sw.columns else ""
        out.append({"name": name or f"Switch {i}", "index": i, "cim_class": cim, "normal_open": normal,
                    "open": fact(is_open, STATED, "as switched now"), "where": where,
                    "role": fact(role, INFERRED, "from what it switches and the feeders on either side")})
    return out


# ---------------------------------------------------------------------------
# Where each value comes from
# ---------------------------------------------------------------------------

# measured / forecast: the data give it for this element alone; allocated: the
# data give its bus's total, split across the elements there; derived: computed
# from an assumption (Q from the load power factor, or a unit's fixed power
# factor); solved: the power flow sets it (a unit holding a voltage); set: the
# element's own set-point; nominal: the case's own value (the data do not cover
# it); proposal: as the proposal that added it says.
VALUE_KINDS = ("measured", "forecast", "allocated", "derived", "solved", "set", "nominal", "proposal")


def _v(kind: str, why: str) -> dict:
    return {"kind": kind, "from": why}


def value_provenance(net, rows, *, dataset: str = "measurements", moment: str | None = None,
                     synthetic: bool = True, load_pf: float = 0.95, load_pf_source: str = "default",
                     generated: str | None = None) -> dict:
    """Where each load's and unit's P and Q at the present operating point come
    from — the data, a share of them, an assumption, the power flow — worked
    out from the moment's data rows (`rows`: substation_name, consumption,
    production, and consumption_q / production_q when measured) the way the
    assignment uses them. `generated`: how synthetic data were made, in words.
    Pure: the values themselves are not touched."""
    from element_names import measurement_name
    got = "forecast" if dataset == "forecasts" else "measured"
    noun = "forecast" if dataset == "forecasts" else ("synthetic data" if synthetic else "measurements")
    def per_bus(column):
        if rows is None or column not in getattr(rows, "columns", []):
            return {}
        return {str(k): v for k, v in rows.groupby("substation_name")[column].sum(min_count=1).items() if v == v}
    cons, prod, cons_q = per_bus("consumption"), per_bus("production"), per_bus("consumption_q")

    def proposed(tbl, i):
        return "proposed" in tbl.columns and tbl.at[i, "proposed"] is True

    loads: dict = {}
    load_at: dict = {}
    for i in net.load.index:
        if not proposed(net.load, i):
            load_at.setdefault(int(net.load.at[i, "bus"]), []).append(int(i))
    for i in net.load.index:
        bus = int(net.load.at[i, "bus"])
        name = measurement_name(net, bus)
        if proposed(net.load, i):
            p = q = _v("proposal", "as the proposal that added it sets it (its profile)")
        elif name not in cons:
            p = q = _v("nominal", f"{name} is not in the {noun}: the case's own value")
        else:
            n = len(load_at.get(bus, []))
            p = (_v(got, f"{name}'s consumption in the {noun}") if n == 1 else
                 _v("allocated", f"{name}'s consumption in the {noun}, split across its {n} loads by their demand "
                                 "in the network as loaded"))
            q = (_v(p["kind"], f"{name}'s reactive consumption in the {noun}" + ("" if n == 1 else ", split likewise"))
                 if name in cons_q else
                 _v("derived", f"P·tan(acos {load_pf:g}) — the load power factor ({load_pf_source}); not in the data"))
        loads[str(int(i))] = {"p": p, "q": q}

    units: dict = {}
    unit_at: dict = {}
    for table in ("gen", "sgen"):
        for i in net[table].index:
            if not proposed(net[table], i):
                unit_at.setdefault(int(net[table].at[i, "bus"]), []).append(f"{table}:{int(i)}")
    for table in ("gen", "sgen"):
        tbl = net[table]
        for i in tbl.index:
            bus = int(tbl.at[i, "bus"])
            name = measurement_name(net, bus)
            if proposed(tbl, i):
                p = q = _v("proposal", "as the proposal that added it sets it (its profile)")
            else:
                n = len(unit_at.get(bus, []))
                if name not in prod:
                    p = _v("nominal", f"{name} is not in the {noun}: the case's own value")
                elif n == 1:
                    p = _v(got, f"{name}'s production in the {noun}")
                else:
                    p = _v("allocated", f"{name}'s production in the {noun}, split by capacity across its {n} units "
                                        "— each unit's share is an estimate, the bus total is not")
                if table == "gen":
                    q = _v("solved", "the power flow sets it to hold the unit's voltage set-point")
                elif "_q_from_meas" in tbl.columns and bool(tbl.at[i, "_q_from_meas"]):
                    q = _v(got if n == 1 else "allocated", f"{name}'s reactive production in the {noun}")
                elif gm.QSET_COL in tbl.columns and pd.notna(tbl.at[i, gm.QSET_COL]):
                    q = _v("set", "its set reactive output")
                else:
                    q = _v("derived", "from its fixed power factor: Q follows P")
            units[f"{table}:{int(i)}"] = {"p": p, "q": q}
    for i in getattr(net, "storage", pd.DataFrame()).index:
        units[f"storage:{int(i)}"] = {"p": _v("set", "held at its set-point; the data carry no storage"),
                                      "q": _v("set", "held at its set-point")}
    context = {"dataset": dataset, "moment": moment, "synthetic": synthetic,
               **({"generated": generated} if synthetic and generated else {}),
               "load_power_factor": {"value": load_pf, "source": load_pf_source},
               "reactive_in_data": bool(cons_q)}
    return {"context": context, "loads": loads, "units": units}


# ---------------------------------------------------------------------------
# Identifiers (CIM mRID)
# ---------------------------------------------------------------------------

_MRID_NS = uuid.UUID("6f1c2a3e-1d7b-5c4e-9a8f-0c0d0c7a0b11")      # CONDUCTOR's namespace for derived mRIDs


def mrid(network_id: str, kind: str, key) -> str:
    """A stable identifier for an element: derived from the network's identity,
    the element's kind and its key as loaded — the same across renames and
    restarts, different between networks."""
    return str(uuid.uuid5(_MRID_NS, f"{network_id}|{kind}|{key}"))


def row_mrids(net, table: str, network_id: str) -> dict[int, str]:
    """{index: mRID} of every row of a table — `_row_mrid` for all of them at once (the columns read once)."""
    tbl = net[table]
    n = len(tbl)
    if not n:
        return {}
    def col(c):
        return list(tbl[c]) if c in tbl.columns else [None] * n
    own, proposed, names = col("origin_id"), col("proposed"), col("name")
    loaded_t, loaded_i = col("loaded_table"), col("loaded_index")
    out = {}
    for k, idx in enumerate(tbl.index):
        o = own[k]
        if isinstance(o, str) and o.strip():
            out[int(idx)] = o.strip().lstrip("_")
        elif proposed[k] is True:
            out[int(idx)] = mrid(network_id, f"{table}:proposed", clean_label(names[k]) or int(idx))
        elif isinstance(loaded_t[k], str) and loaded_t[k]:
            out[int(idx)] = mrid(network_id, loaded_t[k], int(loaded_i[k]))
        else:
            out[int(idx)] = mrid(network_id, table, int(idx))
    return out


def _row_mrid(net, table: str, idx, network_id: str) -> str:
    """The element's own mRID when the network brings one (a CIM/CGMES import:
    pandapower's `origin_id`), else one derived from its table and index; an
    element a proposal added is keyed by its name, so it is the same in every
    copy of the proposal."""
    tbl = net[table]
    if "origin_id" in tbl.columns:
        own = tbl.at[idx, "origin_id"]
        if isinstance(own, str) and own.strip():
            return own.strip().lstrip("_")
    if "proposed" in tbl.columns and tbl.at[idx, "proposed"] is True:
        name = clean_label(tbl.at[idx, "name"]) if "name" in tbl.columns else ""
        return mrid(network_id, f"{table}:proposed", name or int(idx))
    # A unit a generator model moved to the other table keeps the identity it had as loaded (generators.py).
    if "loaded_table" in tbl.columns and isinstance(tbl.at[idx, "loaded_table"], str) and tbl.at[idx, "loaded_table"]:
        return mrid(network_id, tbl.at[idx, "loaded_table"], int(tbl.at[idx, "loaded_index"]))
    return mrid(network_id, table, int(idx))


def _mrid_basis(net, table: str) -> str:
    return STATED if "origin_id" in net[table].columns else INFERRED


def _kv(net, bus: int, per_unit: bool) -> dict:
    proposed = "proposed" in net.bus.columns and bool(net.bus.at[bus, "proposed"] is True)
    if per_unit and not proposed:
        return fact(None, UNKNOWN, "a converted per-unit case: 1 kV stands for the bus's real voltage")
    kv = net.bus.at[bus, "vn_kv"]
    if pd.isna(kv):
        return fact(None, UNKNOWN, "the network gives this bus no nominal voltage")
    return fact(float(kv), STATED, "the bus's nominal voltage in the network")


# ---------------------------------------------------------------------------
# The CIM model as loaded (cim_model.CimModel), when the network came as CIM
# ---------------------------------------------------------------------------

# Element tables of a pandapower network the layer describes as its own kinds; any other table with elements
# (three-winding transformers, equivalents, DC lines, motors…) is listed as other equipment.
_DESCRIBED_TABLES = frozenset({"bus", "line", "trafo", "gen", "sgen", "storage", "load", "ext_grid", "shunt", "switch"})
_OTHER_CLASS = {"trafo3w": "PowerTransformer", "ward": "EquivalentInjection", "xward": "EquivalentInjection",
                "impedance": "EquivalentBranch", "dcline": "DCLineSegment", "motor": "AsynchronousMachine",
                "svc": "StaticVarCompensator", "ssc": "StaticSynchronousCompensator", "tcsc": "SeriesCompensator"}
_OTHER_WHAT = {"trafo3w": "three-winding transformer", "ward": "equivalent injection (a network outside the model, "
               "seen as an injection)", "xward": "extended equivalent injection", "impedance": "equivalent branch or "
               "series compensator", "dcline": "DC line", "motor": "motor (asynchronous machine)",
               "svc": "static var compensator", "ssc": "static synchronous compensator", "tcsc": "series compensator"}
_BUS_COLUMNS = {"trafo3w": ("hv_bus", "mv_bus", "lv_bus"), "impedance": ("from_bus", "to_bus"),
                "dcline": ("from_bus", "to_bus"), "tcsc": ("from_bus", "to_bus")}
# Equipment a node stands for: described as the node's, not as equipment of its own.
_NODE_EQUIPMENT = frozenset({"BusbarSection", "Junction"})


def _origin(net, table: str, idx) -> str | None:
    if "origin_id" not in net[table].columns:
        return None
    v = net[table].at[idx, "origin_id"]
    return cm_norm(v) if isinstance(v, str) and v.strip() else None


def cm_norm(identifier) -> str:
    """An mRID as the layer and the CIM model compare it (cim_model.norm)."""
    s = str(identifier or "").strip()
    if s.startswith("#"):
        s = s[1:]
    if s.lower().startswith("urn:uuid:"):
        s = s[9:]
    return s.lstrip("_")


def _node_of_terminal(cim, terminal) -> str | None:
    return cim.ref(terminal, "Terminal.ConnectivityNode") or cim.ref(terminal, "Terminal.TopologicalNode")


def _model_containers(cim, node) -> dict:
    """{"voltage_level", "bay", "substation"} (mRIDs) of a node of the model: a ConnectivityNode through its
    container, a TopologicalNode through its own."""
    chain = cim.container_chain(node)
    if not chain:
        vl = cim.ref(node, "TopologicalNode.ConnectivityNodeContainer")
        if vl:
            chain = {"voltage_level": vl} if cim.cls(vl) == "VoltageLevel" else {}
            if chain:
                sub = cim.ref(vl, "VoltageLevel.Substation")
                if sub:
                    chain["substation"] = sub
    return chain


def _named(cim, mrid: str | None) -> dict | None:
    return {"name": cim.label(mrid) or mrid, "mrid": mrid} if mrid else None


def _busbars(net, cim) -> dict[int, dict]:
    """{bus index: {"name", "mrid"}} of the model's BusbarSections: from the model when it is kept, else from the
    columns pandapower's converter writes (Busbar_id, Busbar_name)."""
    out: dict[int, dict] = {}
    if cim is not None:
        node_bus = {cm_norm(v): int(i) for i, v in net.bus.get("origin_id", pd.Series(dtype=object)).items()
                    if isinstance(v, str)}
        for bb in cim.of("BusbarSection"):
            for t in cim.terminals(bb):
                bus = node_bus.get(_node_of_terminal(cim, t) or "")
                if bus is not None:
                    out.setdefault(bus, {"name": cim.label(bb) or bb, "mrid": bb})
        return out
    if "Busbar_id" in net.bus.columns:
        for i, v in net.bus["Busbar_id"].items():
            if isinstance(v, str) and v.strip():
                name = net.bus.at[i, "Busbar_name"] if "Busbar_name" in net.bus.columns else None
                out[int(i)] = {"name": clean_label(name) or cm_norm(v), "mrid": cm_norm(v)}
    return out


def _bus_name_markers(net, cim) -> dict[int, dict]:
    """{bus index: {"name", "mrid"}} of the model's BusNameMarkers: the names a node-breaker model gives its
    buses, on the terminals at each node."""
    if cim is None or not cim.of("BusNameMarker"):
        return {}
    node_bus = {cm_norm(v): int(i) for i, v in net.bus.get("origin_id", pd.Series(dtype=object)).items()
                if isinstance(v, str)}
    out: dict[int, dict] = {}
    for marker in cim.of("BusNameMarker"):
        for t in cim.referring(marker, "ACDCTerminal.BusNameMarker"):
            bus = node_bus.get(_node_of_terminal(cim, t) or "")
            if bus is not None and cim.label(marker):
                out.setdefault(bus, {"name": cim.label(marker), "mrid": marker})
    return out


def _model_substation_names(net, cim) -> dict[str, str]:
    """{Substation mRID: its name in the model}, each name unique (a name the model gives twice takes its
    region or its mRID's start)."""
    if cim is None or "Substation_id" not in net.bus.columns:
        return {}
    ids = sorted({cm_norm(v) for v in net.bus["Substation_id"] if isinstance(v, str) and v.strip()})
    names = {s: cim.label(s) for s in ids}
    count: dict[str, int] = {}
    for n in names.values():
        if n:
            count[n] = count.get(n, 0) + 1
    out = {}
    for s, n in names.items():
        if not n:
            continue
        out[s] = n if count[n] == 1 else f"{n} ({s[:8]})"
    return out


def _model_limits(cim, mrid: str | None) -> dict | None:
    """The model's operational limits on a branch, as a fact: each type with how long it may be held."""
    if cim is None or not mrid:
        return None
    lims = cim.limits(mrid)
    if not lims:
        return None
    rows = sorted(({k: v for k, v in x.items() if k not in ("set",)} for x in lims),
                  key=lambda r: (r["terminal"] or 0, not r["permanent"], -(r["duration_s"] or 0)))
    return {"value": rows, "basis": STATED,
            "from": "the model's OperationalLimitSets (current limits in A unless the kind says otherwise)",
            "in_studies": "the studies check the permanent limit only (pandapower keeps one rating per branch)"}


def _model_control(cim, mrid: str | None) -> dict | None:
    if cim is None or not mrid:
        return None
    rc = cim.regulating_control(mrid)
    if not rc:
        return None
    return {"mrid": rc["mrid"], "class": rc["class"] or "RegulatingControl", "mode": rc["mode"],
            "target": rc["target"], "deadband": rc["deadband"],
            "enabled": rc["enabled"] if rc["enabled"] is not None else rc["control_enabled"],
            "basis": STATED, "from": "the model's RegulatingControl"}


def _yes(value) -> bool | None:
    return None if value is None else str(value).strip().lower() == "true"


def _model_transformer(cim, mrid: str | None) -> dict:
    """A transformer's ends (rated voltage and power) and tap changers as the model states them."""
    if cim is None or not mrid:
        return {}
    ends, taps = [], []
    for end in sorted(cim.referring(mrid, "PowerTransformerEnd.PowerTransformer"),
                      key=lambda e: num_or((cim.get(e) or {}).get("TransformerEnd.endNumber"), 0)):
        o = cim.get(end) or {}
        ends.append({"mrid": end, "end": num_or(o.get("TransformerEnd.endNumber"), None),
                     "rated_kv": num_or(o.get("PowerTransformerEnd.ratedU"), None),
                     "rated_mva": num_or(o.get("PowerTransformerEnd.ratedS"), None),
                     "connection": str(o.get("PowerTransformerEnd.connectionKind") or "").rsplit(".", 1)[-1] or None})
        for attr in ("RatioTapChanger.TransformerEnd", "PhaseTapChanger.TransformerEnd"):
            for tc in cim.referring(end, attr):
                t = cim.get(tc) or {}
                taps.append({"mrid": tc, "class": t.get("class"), "end": ends[-1]["end"],
                             "steps": [num_or(t.get("TapChanger.lowStep"), None),
                                       num_or(t.get("TapChanger.highStep"), None)],
                             "neutral": num_or(t.get("TapChanger.neutralStep"), None),
                             "normal": num_or(t.get("TapChanger.normalStep"), None),
                             "now": num_or(t.get("TapChanger.step"), None),
                             "step_pct": num_or(t.get("RatioTapChanger.stepVoltageIncrement")
                                                or t.get("PhaseTapChangerLinear.voltageStepIncrement"), None),
                             "control_mode": str(t.get("RatioTapChanger.tculControlMode") or "").rsplit(".", 1)[-1]
                                             or None,
                             "control_enabled": _yes(t.get("TapChanger.controlEnabled")),
                             "load_tap_changing": _yes(t.get("TapChanger.ltcFlag")),
                             **({"control": c} if (c := _control_of(cim, cim.ref(tc, "TapChanger.TapChangerControl")))
                                else {})})
    out = {}
    if ends:
        # Not "ends": a branch's ends are the buses it joins (other code reads them by that key).
        out["transformer_ends"] = fact(ends, STATED, "the model's PowerTransformerEnds")
    if taps:
        out["tap_changers"] = {"value": taps, "basis": STATED, "from": "the model's tap changers",
                               "in_studies": "held at a fixed position (the tap policy chosen at upload): the "
                                             "studies do not move a tap to follow its control"}
    return out


def num_or(value, default):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _control_of(cim, rc: str | None) -> dict | None:
    """A RegulatingControl (or TapChangerControl) by its mRID: mode, target, whether it is enabled."""
    if cim is None or not rc:
        return None
    o = cim.get(rc) or {}
    return {"mrid": rc, "class": o.get("class"), "mode": str(o.get("RegulatingControl.mode") or "").rsplit(".", 1)[-1]
            or None, "target": num_or(o.get("RegulatingControl.targetValue"), None),
            "enabled": _yes(o.get("RegulatingControl.enabled"))}


def _model_generating_unit(cim, mrid: str | None) -> dict | None:
    """The GeneratingUnit behind a machine (or the units behind an inverter connection), as the model states it."""
    if cim is None or not mrid:
        return None
    gu = cim.ref(mrid, "RotatingMachine.GeneratingUnit") or cim.ref(mrid, "SynchronousMachine.GeneratingUnit")
    if gu:
        o = cim.get(gu) or {}
        return {"mrid": gu, "class": o.get("class"), "name": cim.name(gu),
                "p_max_mw": num_or(o.get("GeneratingUnit.maxOperatingP"), None),
                "p_min_mw": num_or(o.get("GeneratingUnit.minOperatingP"), None),
                "basis": STATED, "from": "the model's generating unit"}
    pe = cim.referring(mrid, "PowerElectronicsUnit.PowerElectronicsConnection")
    if pe:
        return {"units": [{"mrid": u, "class": cim.cls(u), "name": cim.name(u)} for u in pe],
                "basis": STATED, "from": "the model's power electronics units"}
    return None


def _model_load_response(cim, mrid: str | None) -> dict | None:
    """How a load's power follows the voltage, as the model states it (its LoadResponseCharacteristic)."""
    if cim is None or not mrid:
        return None
    lrc = cim.ref(mrid, "EnergyConsumer.LoadResponse")
    if not lrc:
        return None
    o = cim.get(lrc) or {}
    shares = {f"{q}_{k}": num_or(o.get(f"LoadResponseCharacteristic.{q}Constant{K}"), None)
              for q in ("p", "q") for k, K in (("power", "Power"), ("current", "Current"), ("impedance", "Impedance"))}
    exponent = _yes(o.get("LoadResponseCharacteristic.exponentModel"))
    constant_power = (not exponent and shares["p_power"] in (1.0, None) and shares["q_power"] in (1.0, None)
                      and not any(shares[k] for k in ("p_current", "p_impedance", "q_current", "q_impedance")))
    return {"mrid": lrc, "name": cim.name(lrc), "exponent_model": exponent, **shares,
            **({"p_voltage_exponent": num_or(o.get("LoadResponseCharacteristic.pVoltageExponent"), None),
                "q_voltage_exponent": num_or(o.get("LoadResponseCharacteristic.qVoltageExponent"), None)}
               if exponent else {}),
            "constant_power": constant_power, "basis": STATED, "from": "the model's LoadResponseCharacteristic",
            "in_studies": "constant power (pandapower's converter does not read the model's load response)"}


def _model_region(cim, substation: str | None) -> dict | None:
    if cim is None or not substation:
        return None
    sub_region = cim.ref(substation, "Substation.Region")
    if not sub_region:
        return None
    region = cim.ref(sub_region, "SubGeographicalRegion.Region")
    return {"sub_region": _named(cim, sub_region), **({"region": _named(cim, region)} if region else {})}


def _other_equipment(net, cim, names_of_bus: dict[int, str], network_id: str) -> list[dict]:
    """What the layer has no kind of its own for: each element of the other pandapower tables (in the
    computation), and each piece of the model's equipment pandapower's converter did not keep (not in it)."""
    out = []
    kept: set[str] = set()
    for t in net.keys():
        df = net[t]
        if not isinstance(df, pd.DataFrame) or t.startswith("res_") or not len(df) or "origin_id" not in df.columns:
            continue
        kept |= {cm_norm(v) for v in df["origin_id"] if isinstance(v, str) and v.strip()}
    for table in _OTHER_CLASS:
        df = getattr(net, table, None) if table in net else None
        if not isinstance(df, pd.DataFrame) or not len(df):
            continue
        for i in df.index:
            cols = _BUS_COLUMNS.get(table, ("bus",))
            buses = [int(df.at[i, c]) for c in cols if c in df.columns and pd.notna(df.at[i, c])]
            oc = df.at[i, "origin_class"] if "origin_class" in df.columns else None
            name = clean_label(df.at[i, "name"]) if "name" in df.columns else ""
            out.append({"name": name or f"{_OTHER_WHAT[table].split(' (')[0].capitalize()} {int(i)}",
                        "element_type": table, "index": int(i),
                        "cim_class": oc if isinstance(oc, str) and oc else _OTHER_CLASS[table],
                        "what": _OTHER_WHAT[table], "buses": [names_of_bus.get(b, str(b)) for b in buses],
                        "buses_index": buses, "in_computation": True,
                        "in_service": bool(df.at[i, "in_service"]) if "in_service" in df.columns else True,
                        "mrid": _row_mrid(net, table, i, network_id)})
            own = _origin(net, table, i) if cim is not None else None
            if table == "trafo3w" and own:
                out[-1].update(_model_transformer(cim, own))
                limits = _model_limits(cim, own)
                if limits:
                    out[-1]["limits"] = limits
    if cim is None:
        return out
    node_bus = {cm_norm(v): int(i) for i, v in net.bus.get("origin_id", pd.Series(dtype=object)).items()
                if isinstance(v, str)}
    # Equipment is what has terminals; what the converter kept has its mRID in a table.
    seen = set()
    for term in cim.of("Terminal"):
        eq = cim.ref(term, "Terminal.ConductingEquipment")
        if not eq or eq in seen or eq in kept:
            continue
        seen.add(eq)
        cls = cim.cls(eq)
        if not cls or cls in _NODE_EQUIPMENT:
            continue
        buses = []
        for t in cim.terminals(eq):
            b = node_bus.get(_node_of_terminal(cim, t) or "")
            if b is not None and b not in buses:
                buses.append(b)
        o = cim.get(eq) or {}
        normal_open = o.get("Switch.normalOpen")
        open_now = o.get("Switch.open")
        out.append({"name": cim.name(eq) or eq, "element_type": None, "index": None, "cim_class": cls,
                    "what": f"{cls} (not in the computation: pandapower's converter has no place for it)",
                    "buses": [names_of_bus.get(b, str(b)) for b in buses], "buses_index": buses,
                    "in_computation": False, "mrid": eq,
                    **({"normal_open": str(normal_open).lower() == "true"} if normal_open is not None else {}),
                    **({"open": str(open_now).lower() == "true"} if open_now is not None else {})})
    return out


def _layer_mrids(obj, out: set) -> set:
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "mrid" and isinstance(v, str):
                out.add(cm_norm(v))
            else:
                _layer_mrids(v, out)
    elif isinstance(obj, list):
        for v in obj:
            _layer_mrids(v, out)
    return out


def model_coverage(net, cim, layer: dict) -> dict:
    """What of the CIM model reaches the computation (pandapower's tables) and the layer, class by class: the
    loss report of an import. "in_computation": objects whose mRID a pandapower table keeps (as an element or as
    an id column the converter writes); "in_layer": objects the layer describes or cites by mRID."""
    in_pp: set[str] = set()
    for t in net.keys():
        df = net[t]
        if not isinstance(df, pd.DataFrame) or t.startswith("res_") or not len(df):
            continue
        for c in df.columns:
            cs = str(c)
            if cs == "origin_id" or cs.endswith("_id"):
                in_pp |= {cm_norm(v) for v in df[c].dropna() if isinstance(v, str) and v.strip()}
    in_layer = _layer_mrids(layer, set())
    rows = []
    for cls, ids in sorted(cim.by_class.items()):
        if not ids:
            continue
        n_pp = sum(1 for i in ids if i in in_pp)
        n_layer = sum(1 for i in ids if i in in_layer)
        if n_layer >= len(ids):
            status = "described"
        elif cls in _READ_INTO:
            status = "read_into"
        elif n_layer:
            status = "partly"
        elif cls in _SOLVED_STATE:
            status = "solved_state"
        else:
            status = "model_only"
        rows.append({"class": cls, "in_model": len(ids), "in_computation": n_pp, "in_layer": n_layer,
                     "status": status, **({"read_into": _READ_INTO[cls]} if status == "read_into" else {}),
                     "profiles": sorted({p for i in ids for p in cim.object_profiles.get(i, ())})})
    return {"classes": rows,
            "note": "in_computation: kept by pandapower (an element, or an id it records); in_layer: described or "
                    "cited by the CIM layer. status: described (every object), partly, read_into (its content is "
                    "read into other entries, e.g. limit sets into each branch's limits), solved_state (the source "
                    "tool's power-flow solution), model_only (kept in the model, not in the layer: the agent can "
                    "still look an object up by its mRID)."}


# Classes whose objects the layer does not list but whose content it reads into other entries.
_READ_INTO = {"Terminal": "every element's connections", "OperationalLimitSet": "each branch's limits",
              "OperationalLimitType": "each limit's type and duration",
              "TopologicalNode": "the buses (a node-breaker model's buses are its ConnectivityNodes; the "
                                 "topological nodes are the source tool's grouping of them)",
              "TopologicalIsland": "the buses' connectivity", "CoordinateSystem": "the model's locations",
              "BusNameMarker": "the buses' names in the model",
              "Diagram": "the network map, drawn as the model's own diagram",
              "DiagramObject": "the network map, drawn as the model's own diagram",
              "DiagramObjectPoint": "the network map, drawn as the model's own diagram"}
# The source tool's power-flow solution (the SV profile).
_SOLVED_STATE = frozenset({"SvVoltage", "SvPowerFlow", "SvInjection", "SvShuntCompensatorSections", "SvTapStep",
                           "SvStatus", "SvSwitch"})


def build(net, declared: dict | None = None, profile: dict | None = None,
          declared_loads: dict | None = None, profile_loads: dict | None = None,
          normal_open: dict | None = None, provenance: dict | None = None, network_id: str = "local",
          cim=None) -> dict:
    """What every element of `net` is, with the basis of each fact (compact:
    one entry per element, for the agent and for checking answers).
    `declared`/`profile`: units' sources; `declared_loads`/`profile_loads`:
    loads' categories — each keyed by the element's name as loaded.
    `normal_open`: {switch index: open as loaded} (the switches' normal
    state); `provenance`: `value_provenance`'s result, where each value of the
    present operating point comes from. `network_id`: the network's identity,
    for derived mRIDs (the backend passes the fingerprint as loaded).
    `cim`: the CIM model as loaded (cim_model.CimModel) when the network came
    as CIM: what it states is read from it first — names, voltage levels, bays,
    busbar sections, limits, controls, and the equipment pandapower did not keep."""
    per_unit = per_unit_case(net)
    subs, sub_basis = substations(net, cim)
    model_named = bool(_model_substation_names(net, cim))
    sub_why = (("the model's Substation for each bus"
                + (" (its name in the model" if model_named else " (named after its highest-voltage bus")
                + "; a node with none, such as a boundary node, is a substation of its own)"
                if "Substation_id" in net.bus.columns and "substation" not in net.bus.columns else "named in the network")
               if sub_basis == STATED else
               "buses joined by transformers or bus couplers; a single bus is a substation of its own")
    busbar_of = _busbars(net, cim)
    marker_of = _bus_name_markers(net, cim)
    has_class = "origin_class" in net.bus.columns
    buses = []
    for i in net.bus.index:
        b = int(i)
        oc = net.bus.at[b, "origin_class"] if has_class else None
        entry = {"name": bus_display_name(net, b), "index": b,
                 # A node-breaker model's buses are its ConnectivityNodes; a bus-branch model's, and any other
                 # network's, the TopologicalNodes a power flow solves.
                 "cim_class": oc if oc in ("ConnectivityNode", "TopologicalNode") else CIM_CLASS["bus"],
                 "kv": _kv(net, b, per_unit), "substation": fact(subs.get(b), sub_basis, sub_why),
                 "in_service": bool(net.bus.at[b, "in_service"])}
        if b in busbar_of:
            entry["busbar"] = fact(busbar_of[b], STATED, "the model's BusbarSection at this node")
        if b in marker_of:
            entry["name_in_model"] = fact(marker_of[b], STATED, "the model's BusNameMarker for this node (the bus "
                                                                "name the source tool shows)")
        node = _origin(net, "bus", b) if cim is not None else None
        if node:
            chain = _model_containers(cim, node)
            if chain.get("voltage_level"):
                vl = chain["voltage_level"]
                entry["voltage_level"] = fact({"name": cim.label(vl) or vl, "mrid": vl}, STATED,
                                              "the model's VoltageLevel for this node")
            if chain.get("bay"):
                entry["bay"] = fact(_named(cim, chain["bay"]), STATED, "the model's Bay for this node")
        buses.append(entry)
    kv_of = {e["index"]: e["kv"]["value"] for e in buses}
    branches, unresolved = [], []
    for etype, label, a, b in (("line", line_display_name, "from_bus", "to_bus"),
                               ("trafo", trafo_display_name, "hv_bus", "lv_bus")):
        tbl = net[etype]
        for i in tbl.index:
            # An end the network does not have (a CIM import without its boundary set leaves the tie-lines
            # to neighbours unconnected): reported, not described.
            if pd.isna(tbl.at[i, a]) or pd.isna(tbl.at[i, b]) or int(tbl.at[i, a]) not in kv_of \
                    or int(tbl.at[i, b]) not in kv_of:
                name = clean_label(tbl.at[i, "name"]) if "name" in tbl.columns else ""
                unresolved.append(f"{etype} {int(i)}" + (f" ({name})" if name else ""))
                continue
            ends = [int(tbl.at[i, a]), int(tbl.at[i, b])]
            entry = {"name": label(net, int(i)), "element_type": etype, "index": int(i),
                     "cim_class": CIM_CLASS[etype], "ends": [bus_display_name(net, e) for e in ends],
                     "ends_index": ends, "in_service": bool(tbl.at[i, "in_service"])}
            if etype == "line":
                entry["kv"] = kv_of.get(ends[0])
                ka = tbl.at[i, "max_i_ka"] if "max_i_ka" in tbl.columns else None
                entry["rating_mva"] = (fact(round(math.sqrt(3) * float(ka) * kv_of[ends[0]], 1), STATED,
                                            "its current rating at the line's voltage")
                                       if ka is not None and pd.notna(ka) and kv_of.get(ends[0]) else
                                       fact(None, UNKNOWN, "no voltage or no current rating to state it in MVA"))
            else:
                entry["kv"] = [kv_of.get(ends[0]), kv_of.get(ends[1])]
                entry["rating_mva"] = fact(float(tbl.at[i, "sn_mva"]), STATED, "its rated power")
            own = _origin(net, etype, i) if cim is not None else None
            limits = _model_limits(cim, own)
            if limits:
                entry["limits"] = limits
            if etype == "trafo":
                entry.update(_model_transformer(cim, own))
            line = cim.container_chain(own).get("line") if own else None
            if line:
                # CIM's Line: the circuit a segment belongs to (a line of several segments has one Line).
                entry["cim_line"] = _named(cim, line)
            branches.append(entry)
    units = []
    for table, i in _unit_rows(net):
        source = unit_source(net, table, i, declared, profile, cim=cim)
        bus = int(net[table].at[i, "bus"])
        inverter = any(re.search(r"fuel ?cell", str(net[table].at[i, c]), re.IGNORECASE)
                       for c in ("type", "name") if c in net[table].columns)
        equipment, unit_class = cim_unit(table, source["value"], inverter)
        control = _control(net, table, i)
        own_unit = _origin(net, table, i) if cim is not None else None
        stated_control = _model_control(cim, own_unit)
        generating_unit = _model_generating_unit(cim, own_unit)
        units.append({**({"generating_unit": generating_unit} if generating_unit else {}),"name": gm.unit_name(net, table, i), "element_type": table, "index": int(i),
                      "cim_class": equipment, "cim_unit": unit_class,
                      "cim_control": {"class": "RegulatingControl", "mode": _CIM_MODE[control["value"]],
                                      **({"in_model": stated_control} if stated_control else {})},
                      "source": source, "control": control,
                      "bus": bus_display_name(net, bus), "bus_index": bus,
                      "p_max_mw": (round(float(net[table].at[i, "max_p_mw"]), 3)
                                   if "max_p_mw" in net[table].columns and pd.notna(net[table].at[i, "max_p_mw"])
                                   else None)})
    for i in getattr(net, "storage", pd.DataFrame()).index:
        bus = int(net.storage.at[i, "bus"])
        name = clean_label(net.storage.at[i, "name"]) if "name" in net.storage.columns else ""
        units.append({"name": name or f"Storage_{int(i)}", "element_type": "storage", "index": int(i),
                      "cim_class": "PowerElectronicsConnection", "cim_unit": "BatteryUnit", "cim_control": None,
                      "source": fact("storage", STATED, "a storage unit"),
                      "control": fact("power_factor", STATED, "a storage unit"),
                      "bus": bus_display_name(net, bus), "bus_index": bus, "p_max_mw": None})
    shunts = []
    for i in getattr(net, "shunt", pd.DataFrame()).index:
        bus = int(net.shunt.at[i, "bus"])
        q = float(net.shunt.at[i, "q_mvar"]) if pd.notna(net.shunt.at[i, "q_mvar"]) else 0.0
        step = net.shunt.at[i, "step"] if "step" in net.shunt.columns and pd.notna(net.shunt.at[i, "step"]) else 1
        kind = ("capacitor" if q < 0 else "reactor") if q else "unknown"
        name = clean_label(net.shunt.at[i, "name"]) if "name" in net.shunt.columns else ""
        oc = net.shunt.at[i, "origin_class"] if "origin_class" in net.shunt.columns else None
        entry = {"name": name or f"Shunt {int(i)}", "index": int(i),
                 # pandapower's converter makes a static var compensator or a non-linear shunt a shunt too; the
                 # class it came as is kept.
                 "cim_class": oc if isinstance(oc, str) and oc else "LinearShuntCompensator",
                 "kind": (fact(kind, STATED, "the sign of its reactive power in the network (negative: it "
                                             "injects)") if q else
                          fact("unknown", UNKNOWN, "no reactive power given")),
                 "q_mvar_at_1pu": round(q * float(step), 4), "bus": bus_display_name(net, bus),
                 "bus_index": bus, "in_service": bool(net.shunt.at[i, "in_service"])}
        stated_control = _model_control(cim, _origin(net, "shunt", i) if cim is not None else None)
        if stated_control:
            entry["control"] = {**stated_control,
                                "in_studies": "held at its present sections: the studies do not switch it to "
                                              "follow its target"}
        shunts.append(entry)
    loads = []
    for i in net.load.index:
        bus = int(net.load.at[i, "bus"])
        loads.append({"name": (clean_label(net.load.at[i, "name"]) if "name" in net.load.columns else "")
                      or f"Load {int(i)}", "index": int(i), "cim_class": CIM_CLASS["load"],
                      "category": load_category(net, i, declared_loads, profile_loads),
                      "bus": bus_display_name(net, bus), "bus_index": bus,
                      "p_mw": round(float(net.load.at[i, "p_mw"]), 3),
                      "in_service": bool(net.load.at[i, "in_service"])})
        response = _model_load_response(cim, _origin(net, "load", i) if cim is not None else None)
        if response:
            loads[-1]["load_response"] = response
    external = []
    for i in net.ext_grid.index:
        bus = int(net.ext_grid.at[i, "bus"])
        # What carries its power: the branches leaving its substation (a bus
        # coupler can stand between the busbar it hangs from and the lines),
        # and the transformers inside it.
        home = {b for b, name in subs.items() if name == subs.get(bus)} or {bus}
        via = [br["name"] for br in branches
               if (br["element_type"] == "line" and len(home & set(br["ends_index"])) == 1)
               or (br["element_type"] == "trafo" and home & set(br["ends_index"]))]
        external.append({"name": "External grid", "cim_class": CIM_CLASS["ext_grid"], "bus": bus_display_name(net, bus),
                         "bus_index": bus, "substation": subs.get(bus), "branches": via})
        # A model whose reference is a machine gets it as the external grid (network_loader.read_cgmes), keeping
        # its mRID: the machine's control and unit stay stated.
        own_ext = _origin(net, "ext_grid", i) if cim is not None else None
        if own_ext:
            external[-1]["in_model"] = {"class": cim.cls(own_ext), "name": cim.name(own_ext),
                                        **({"control": c} if (c := _model_control(cim, own_ext)) else {}),
                                        **({"generating_unit": g} if (g := _model_generating_unit(cim, own_ext))
                                           else {})}
    # Feeders and switches (the topology as switched now).
    names = {(b["element_type"], b["index"]): b["name"] for b in branches}
    try:
        feeder, sources = feeders_of(net, names) if len(net.bus) and not unresolved else ({}, [])
    except Exception:  # noqa: BLE001 — a topology pandapower cannot graph: no feeders, said below
        feeder, sources = {}, []
    bus_by_index = {b["index"]: b for b in buses}
    for b in buses:
        if not sources:
            b["feeder"] = fact(None, UNKNOWN, "no external grid in service: no supply route")
        elif b["index"] not in feeder:
            b["feeder"] = fact(None, INFERRED, "not connected to any external grid as switched now")
        else:
            name, why_not = feeder[b["index"]]
            b["feeder"] = fact(name, INFERRED, _FEEDER_RULE if name else f"no feeder: {why_not}")
    switches = _switch_entries(net, names, bus_by_index, feeder, normal_open)
    if provenance:
        for u in units:
            u["values"] = provenance.get("units", {}).get(f"{u['element_type']}:{u['index']}")
        for d in loads:
            d["values"] = provenance.get("loads", {}).get(str(d["index"]))
    for b in buses:
        b["mrid"] = _row_mrid(net, "bus", b["index"], network_id)
    for b in branches:
        b["mrid"] = _row_mrid(net, b["element_type"], b["index"], network_id)
    for u in units:
        u["mrid"] = _row_mrid(net, u["element_type"], u["index"], network_id)
    for d in loads:
        d["mrid"] = _row_mrid(net, "load", d["index"], network_id)
    for x in shunts:
        x["mrid"] = _row_mrid(net, "shunt", x["index"], network_id)
    for x in switches:
        x["mrid"] = _row_mrid(net, "switch", x["index"], network_id)
    for k, e in enumerate(external):
        e["mrid"] = _row_mrid(net, "ext_grid", net.ext_grid.index[k], network_id)
    levels = sorted({e["kv"]["value"] for e in buses if e["kv"]["value"] is not None}, reverse=True)
    by_source: dict[str, list[str]] = {}
    for u in units:
        by_source.setdefault(u["source"]["value"], []).append(u["name"])
    by_category: dict[str, list[str]] = {}
    for d in loads:
        by_category.setdefault(d["category"]["value"], []).append(d["name"])
    other = _other_equipment(net, cim, {b["index"]: b["name"] for b in buses}, network_id)
    # Each substation's identity: the model's Substation (its mRID) when the network came as CIM.
    sub_mrid = {}
    if "Substation_id" in net.bus.columns:
        for b in buses:
            sid = net.bus.at[b["index"], "Substation_id"]
            if isinstance(sid, str) and sid.strip():
                sub_mrid.setdefault(b["substation"]["value"], cm_norm(sid))
    layer = {"per_unit_case": per_unit, "voltage_levels_kv": levels,
             "substations": {"basis": sub_basis, "from": sub_why, "count": len(set(subs.values()))},
             "units_by_source": by_source, "loads_by_category": by_category,
             "buses": buses, "branches": branches, "units": units, "loads": loads, "external_grids": external,
             "switches": switches, "shunts": shunts,
             **({"other_equipment": other} if other else {}),
             **({"unresolved_branches": unresolved} if unresolved else {}),
             "containers": _containers(buses, units, loads, switches, sub_basis, sub_why, branches, network_id,
                                       sub_mrid=sub_mrid, cim=cim),
             "mrid_basis": _mrid_basis(net, "bus"),
             **({"provenance": provenance.get("context")} if provenance else {})}
    if cim is not None:
        layer["cim_model"] = {**cim.summary(), "coverage": model_coverage(net, cim, layer)}
    return layer


def _containers(buses, units, loads, switches, sub_basis, sub_why, branches=(), network_id="local",
                sub_mrid: dict | None = None, cim=None) -> dict:
    """CIM's equipment containers: each Substation with its VoltageLevels, and
    each Feeder with what it supplies and the switches that tie it to others.
    A voltage level is the model's own (its name, mRID and bays) when the model
    states the buses' VoltageLevel; otherwise the substation's buses at one
    voltage (inferred)."""
    sub_mrid = sub_mrid or {}
    subs: dict[str, dict] = {}
    for b in buses:
        s = subs.setdefault(b["substation"]["value"], {"levels": {}, "buses": []})
        s["buses"].append(b["name"])
        vl = (b.get("voltage_level") or {}).get("value")
        key = ("model", vl["mrid"]) if vl else ("kv", b["kv"]["value"])
        s["levels"].setdefault(key, {"buses": [], "kv": b["kv"]["value"], "stated": vl, "busbars": [], "bays": []})
        lv = s["levels"][key]
        lv["buses"].append(b["name"])
        bb = (b.get("busbar") or {}).get("value")
        if bb and bb["name"] not in lv["busbars"]:
            lv["busbars"].append(bb["name"])
        bay = (b.get("bay") or {}).get("value")
        if bay and bay["name"] not in lv["bays"]:
            lv["bays"].append(bay["name"])
    first_bus = {}
    for b in buses:
        first_bus.setdefault(b["substation"]["value"], b["index"])

    def level(n, key, lv):
        if lv["stated"]:
            kv = (cim.base_kv(lv["stated"]["mrid"]) if cim is not None else None) or lv["kv"]
            base = cim.ref(lv["stated"]["mrid"], "VoltageLevel.BaseVoltage") if cim is not None else None
            return {"cim_class": "VoltageLevel", "name": lv["stated"]["name"], "kv": kv, "buses": lv["buses"],
                    "mrid": lv["stated"]["mrid"], "basis": STATED, "from": "the model's VoltageLevel",
                    **({"base_voltage": {"mrid": base, "kv": kv}} if base else {}),
                    **({"busbars": lv["busbars"]} if lv["busbars"] else {}),
                    **({"bays": lv["bays"]} if lv["bays"] else {})}
        kv = key[1]
        return {"cim_class": "VoltageLevel", "kv": kv, "buses": lv["buses"],
                "mrid": mrid(network_id, "voltage_level", f"{first_bus[n]}:{kv}"), "basis": INFERRED,
                "from": "the substation's buses at one nominal voltage",
                **({"busbars": lv["busbars"]} if lv["busbars"] else {})}
    substation_list = [{"name": n, "cim_class": "Substation", "basis": sub_basis, "from": sub_why,
                        "mrid": sub_mrid.get(n) or mrid(network_id, "substation", first_bus[n]),
                        **({"region": r} if (r := _model_region(cim, sub_mrid.get(n))) else {}),
                        "voltage_levels": [level(n, key, lv) for key, lv in
                                           sorted(s["levels"].items(), key=lambda x: (-(x[1]["kv"] or 0),
                                                                                      str(x[0][1])))]}
                       for n, s in subs.items()]
    feeders: dict[str, dict] = {}
    on_feeder = {b["index"]: b["feeder"]["value"] for b in buses if b.get("feeder", {}).get("value")}
    for b in buses:
        f = on_feeder.get(b["index"])
        if f:
            feeders.setdefault(f, {"buses": [], "units": [], "loads": [], "load_mw": 0.0})["buses"].append(b["name"])
    for u in units:
        f = on_feeder.get(u["bus_index"])
        if f:
            feeders[f]["units"].append(u["name"])
    for d in loads:
        f = on_feeder.get(d["bus_index"])
        if f:
            feeders[f]["loads"].append(d["name"])
            feeders[f]["load_mw"] = round(feeders[f]["load_mw"] + d["p_mw"], 3)
    head = {b["name"]: b["index"] for b in branches if b["element_type"] == "line"}
    feeder_list = [{"name": f, "cim_class": "Feeder", "basis": INFERRED, "from": _FEEDER_RULE,
                    "mrid": mrid(network_id, "feeder", head.get(f, f)), **v,
                    "ties": [s["name"] for s in switches if f in str(s["role"]["value"])
                             and s["role"]["value"].startswith("tie")]}
                   for f, v in feeders.items()]
    return {"substations": substation_list, "feeders": feeder_list}


def units_of(semantics: dict, sources, bases=(STATED, NAMED)) -> list[dict]:
    """The units whose source is one of `sources`, on one of `bases`."""
    wanted = set(sources)
    return [u for u in semantics["units"] if u["source"]["value"] in wanted and u["source"]["basis"] in bases]


# ---------------------------------------------------------------------------
# The operator's declarations
# ---------------------------------------------------------------------------

class DeclarationError(ValueError):
    """A declaration that cannot be kept; the message says why."""


def normalise_declarations(net, changes: list[dict], declared: dict | None = None) -> dict:
    """The operator's statements of units' energy sources, checked against
    `net`: {unit name as loaded: source}. 'unknown' withdraws one."""
    import resolve as rv
    out = dict(declared or {})
    if not isinstance(changes, list) or not changes:
        raise DeclarationError("Give a list of {unit, source}.")
    for n, raw in enumerate(changes, start=1):
        if not isinstance(raw, dict):
            raise DeclarationError(f"Declaration {n}: give a unit and its source.")
        source = str(raw.get("source") or "").strip().casefold()
        if source not in SOURCES:
            raise DeclarationError(f"Declaration {n}: source must be one of {list(SOURCES)}.")
        try:
            found = rv.find_unit(net, raw.get("unit"))
        except rv.ResolveError as exc:
            raise DeclarationError(f"Declaration {n}: {exc}") from None
        key = gm.unit_key(net, found.etype, found.idx)
        if source == "unknown":
            out.pop(key, None)
        else:
            out[key] = source
    return out


def normalise_load_declarations(net, changes: list[dict], declared: dict | None = None) -> dict:
    """The operator's statements of loads' categories, checked against `net`:
    {load name as loaded: category}. 'unknown' withdraws one."""
    import resolve as rv
    out = dict(declared or {})
    if not isinstance(changes, list) or not changes:
        raise DeclarationError("Give a list of {load, category}.")
    for n, raw in enumerate(changes, start=1):
        if not isinstance(raw, dict):
            raise DeclarationError(f"Load declaration {n}: give a load and its category.")
        category = str(raw.get("category") or "").strip().casefold().replace(" ", "_")
        if category not in LOAD_CATEGORIES:
            raise DeclarationError(f"Load declaration {n}: category must be one of {list(LOAD_CATEGORIES)}.")
        try:
            found = rv.find(net, raw.get("load"), ("load",))
        except rv.ResolveError as exc:
            raise DeclarationError(f"Load declaration {n}: {exc}") from None
        key = load_key(net, found.idx)
        if category == "unknown":
            out.pop(key, None)
        else:
            out[key] = category
    return out


# ---------------------------------------------------------------------------
# Views: what the agent is sent, and what an answer is checked against
# ---------------------------------------------------------------------------

ABOUT = ("summary", "units", "loads", "voltage_levels", "substations", "external_grid", "switches", "feeders",
         "data", "model")
_MAX_LISTED = 40


def _basis_counts(items) -> dict:
    out: dict[str, int] = {}
    for f in items:
        out[f["basis"]] = out.get(f["basis"], 0) + 1
    return out


def view(semantics: dict, about: str = "summary") -> dict:
    """A compact answer about one aspect of the network, for the agent: never
    the whole network (thousands of elements on a large one)."""
    units, buses = semantics["units"], semantics["buses"]
    unknown_units = [u["name"] for u in units if u["source"]["basis"] == UNKNOWN]
    declare_hint = ("What drives a unit the network does not state can be declared by the operator "
                    "(declare_network_facts); until then it is unknown — never guessed.")
    if about == "units":
        groups = {}
        for u in units:
            g = groups.setdefault(u["source"]["value"], {"units": [], "bases": {}})
            if len(g["units"]) < _MAX_LISTED:
                g["units"].append({"name": u["name"], "bus": u["bus"], "basis": u["source"]["basis"],
                                   "from": u["source"]["from"], "control": u["control"]["value"],
                                   **({"p_max_mw": u["p_max_mw"]} if u["p_max_mw"] is not None else {})})
            g["bases"][u["source"]["basis"]] = g["bases"].get(u["source"]["basis"], 0) + 1
        variable = [u["name"] for u in units if u["source"]["value"] in VARIABLE and u["source"]["basis"] != UNKNOWN]
        return {"about": "units", "by_source": groups, "wind_or_solar": variable,
                "renewable": [u["name"] for u in units if u["source"]["value"] in RENEWABLE
                              and u["source"]["basis"] != UNKNOWN],
                **({"unknown": unknown_units[:_MAX_LISTED], "note": declare_hint} if unknown_units else {})}
    if about == "loads":
        groups = {}
        for d in semantics.get("loads") or []:
            g = groups.setdefault(d["category"]["value"], {"loads": [], "bases": {}, "p_mw": 0.0})
            if len(g["loads"]) < _MAX_LISTED:
                g["loads"].append({"name": d["name"], "bus": d["bus"], "basis": d["category"]["basis"],
                                   "from": d["category"]["from"], "p_mw": d["p_mw"]})
            g["bases"][d["category"]["basis"]] = g["bases"].get(d["category"]["basis"], 0) + 1
            g["p_mw"] = round(g["p_mw"] + d["p_mw"], 3)
        return {"about": "loads", "by_category": groups,
                "note": "p_mw is each load's demand at the operating point the network is at now (the "
                        "simulation clock's moment, `operating_point`), not a rating. A category the "
                        "network does not state can be declared by the operator (declare_network_facts, loads); "
                        "until then it is unknown — never guessed."}
    if about == "switches":
        sws = semantics.get("switches") or []
        return {"about": "switches", "count": len(sws),
                "normally_open": [s["name"] for s in sws if s["normal_open"]["value"]],
                "switched_from_normal": [s["name"] for s in sws if s["open"]["value"] != s["normal_open"]["value"]],
                "switches": [{"name": s["name"], "class": s["cim_class"]["value"], "class_basis": s["cim_class"]["basis"],
                              "normal_open": s["normal_open"]["value"], "open": s["open"]["value"],
                              "where": s["where"], "role": s["role"]["value"]} for s in sws[:_MAX_LISTED]],
                **({"more": len(sws) - _MAX_LISTED} if len(sws) > _MAX_LISTED else {}),
                "note": "normal_open is the switch's state as the network was loaded (its normal state), open its "
                        "state now. A tie joins two feeders: closing it restores supply across them (not checked "
                        "against flows). Roles are inferred from the topology."}
    if about == "feeders":
        fs = (semantics.get("containers") or {}).get("feeders") or []
        out = {"about": "feeders", "basis": INFERRED, "from": _FEEDER_RULE,
               "feeders": [{"name": f["name"], "buses": f["buses"][:_MAX_LISTED], "units": f["units"][:_MAX_LISTED],
                            "loads": f["loads"][:_MAX_LISTED], "load_mw": f["load_mw"], "ties": f["ties"]}
                           for f in fs[:_MAX_LISTED]]}
        if not fs:
            why = next((b["feeder"]["from"] for b in buses if b.get("feeder")), "no external grid")
            out["note"] = "No feeders: " + why.removeprefix("no feeder: ") + "."
        return out
    if about == "data":
        ctx = semantics.get("provenance")
        if not ctx:
            return {"about": "data", "note": "Where the values come from cannot be said for this network now."}
        def tally(items):
            out: dict = {}
            for e in items:
                v = e.get("values") or {}
                for q in ("p", "q"):
                    k = (v.get(q) or {}).get("kind")
                    if k:
                        out.setdefault(q, {}).setdefault(k, 0)
                        out[q][k] += 1
            return out
        def examples(items):
            rows = [{"name": e["name"], "p": e["values"]["p"]["kind"], "q": e["values"]["q"]["kind"],
                     "q_from": e["values"]["q"]["from"]} for e in items if e.get("values")]
            not_measured = [r for r in rows if r["p"] not in ("measured", "forecast") or r["q"] not in ("measured", "forecast")]
            return (not_measured or rows)[:_MAX_LISTED]
        return {"about": "data", **ctx,
                "loads": {"count_by_kind": tally(semantics.get("loads") or []),
                          "not_fully_measured": examples(semantics.get("loads") or [])},
                "units": {"count_by_kind": tally(units), "not_fully_measured": examples(units)},
                "note": "Kinds: measured (the data give it for this element), allocated (the data give its bus's "
                        "total, split across the elements there — each share is an estimate), derived (from an "
                        "assumption: Q from the load power factor, or a unit's fixed power factor), solved (the "
                        "power flow sets it: a unit holding a voltage), set, nominal (not in the data: the case's "
                        "own value), proposal. Say which when it matters: an allocated or derived value is not a "
                        "measurement."
                        + (" These are synthetic data, not field measurements"
                           + (f", generated with: {ctx['generated']}" if ctx.get("generated") else "") + "."
                           if ctx.get("synthetic") else "")}
    if about == "voltage_levels":
        if semantics["per_unit_case"]:
            return {"about": "voltage_levels", "levels_kv": [], "basis": UNKNOWN,
                    "note": "A converted per-unit case: every bus is at 1 kV, a placeholder for its real voltage. "
                            "The network's voltage levels are unknown."}
        levels = {}
        for b in buses:
            levels.setdefault(b["kv"]["value"], []).append(b["name"])
        out = {"about": "voltage_levels", "basis": STATED,
               "levels": [{"kv": kv, "buses": len(names), "e.g.": names[:5]}
                          for kv, names in sorted(levels.items(), key=lambda kv: -(kv[0] or 0))]}
        # The model's own VoltageLevels (a CIM model states them, with names, busbars and bays).
        named = [{"name": vl["name"], "substation": s["name"], "kv": vl["kv"], "buses": len(vl["buses"]),
                  **({"busbars": vl["busbars"]} if vl.get("busbars") else {}),
                  **({"bays": len(vl["bays"])} if vl.get("bays") else {})}
                 for s in (semantics.get("containers") or {}).get("substations") or []
                 for vl in s["voltage_levels"] if vl.get("basis") == STATED and vl.get("name")]
        if named:
            out["in_model"] = named[:_MAX_LISTED]
            if len(named) > _MAX_LISTED:
                out["more_in_model"] = len(named) - _MAX_LISTED
        return out
    if about == "substations":
        subs: dict[str, dict] = {}
        for b in buses:
            s = subs.setdefault(b["substation"]["value"], {"buses": [], "kv": set()})
            s["buses"].append(b["name"])
            if b["kv"]["value"] is not None:
                s["kv"].add(b["kv"]["value"])
        listed = sorted(subs.items(), key=lambda kv: (-len(kv[1]["buses"]), str(kv[0])))
        return {"about": "substations", **semantics["substations"],
                "substations": [{"name": n, "buses": v["buses"][:10], "n_buses": len(v["buses"]),
                                 "voltage_levels_kv": sorted(v["kv"], reverse=True)}
                                for n, v in listed[:_MAX_LISTED]],
                **({"more": len(listed) - _MAX_LISTED} if len(listed) > _MAX_LISTED else {})}
    if about == "external_grid":
        return {"about": "external_grid", "external_grids": semantics["external_grids"],
                "note": "The external grid is a connection, not a bus or a branch: what carries its power is the "
                        "branches listed."}
    if about == "model":
        return model_view(semantics)
    by_source = {k: len(v) for k, v in semantics["units_by_source"].items()}
    other = semantics.get("other_equipment") or []
    cimm = semantics.get("cim_model")
    return {"about": "summary", "buses": len(buses), "lines": sum(b["element_type"] == "line"
                                                                  for b in semantics["branches"]),
            "transformers": sum(b["element_type"] == "trafo" for b in semantics["branches"]),
            "units_by_source": by_source, "unit_source_bases": _basis_counts(u["source"] for u in units),
            "loads_by_category": {k: len(v) for k, v in (semantics.get("loads_by_category") or {}).items()},
            "voltage_levels_kv": semantics["voltage_levels_kv"] or "unknown (a per-unit case)",
            "substations": semantics["substations"],
            "external_grid_at": [e["bus"] for e in semantics["external_grids"]],
            "shunts": [{"name": x["name"], "bus": x["bus"], "kind": x["kind"]["value"],
                        "q_mvar_at_1pu": x["q_mvar_at_1pu"]} for x in semantics.get("shunts") or []][:_MAX_LISTED],
            "switches": {"count": len(semantics.get("switches") or []),
                         "normally_open": sum(1 for x in semantics.get("switches") or [] if x["normal_open"]["value"])},
            "feeders": [f["name"] for f in (semantics.get("containers") or {}).get("feeders") or []],
            **({"other_equipment": {"in_computation": _class_counts(o for o in other if o["in_computation"]),
                                    "not_in_computation": _class_counts(o for o in other if not o["in_computation"])}}
               if other else {}),
            **({"cim_model": {"version": cimm.get("version"), "objects": cimm.get("objects"),
                              "note": "The network came as a CIM model, kept whole: describe_network(about='model') "
                                      "says what the studies use of it and what they ignore."}} if cimm else {}),
            **({"note": declare_hint} if unknown_units else {})}


def _class_counts(items) -> dict:
    out: dict = {}
    for o in items:
        out[o["cim_class"]] = out.get(o["cim_class"], 0) + 1
    return out


def model_caveats(semantics: dict) -> list[dict]:
    """What the CIM model states that the studies do not use: each {"kind", "element", "note"}. Only for a network
    that came as CIM (its model kept)."""
    if not semantics.get("cim_model"):
        return []
    out = []
    temporary = [b["name"].split(" | ")[0] for b in semantics.get("branches") or []
                 if any(not r["permanent"] and r["duration_s"] for r in (b.get("limits") or {}).get("value") or [])]
    if temporary:
        out.append({"kind": "temporary_limits_after_outages_only", "element": f"{len(temporary)} branch(es)",
                    "note": "the model gives temporary limits (held for a stated time) as well as the permanent one; "
                            "the N-1 studies say how long an overload after an outage may be held under them, the "
                            "other studies check the permanent limit, e.g. " + ", ".join(temporary[:3])})
    controlled = [x["name"] for x in semantics.get("shunts") or [] if (x.get("control") or {}).get("enabled")]
    if controlled:
        out.append({"kind": "shunt_control_not_followed", "element": ", ".join(controlled[:5])
                    + (" …" if len(controlled) > 5 else ""),
                    "note": f"{len(controlled)} shunt(s) regulate a voltage in the model; the studies hold them at "
                            "their present sections"})
    taps = [b["name"].split(" | ")[0] for b in semantics.get("branches") or []
            for t in (b.get("tap_changers") or {}).get("value") or [] if t.get("control_enabled")]
    if taps:
        out.append({"kind": "tap_control_not_followed", "element": ", ".join(taps[:5]) + (" …" if len(taps) > 5 else ""),
                    "note": f"{len(taps)} tap changer(s) regulate in the model; the studies hold each tap fixed"})
    responsive = [d["name"] for d in semantics.get("loads") or []
                  if (d.get("load_response") or {}).get("constant_power") is False]
    if responsive:
        out.append({"kind": "load_response_not_used", "element": ", ".join(responsive[:5])
                    + (" …" if len(responsive) > 5 else ""),
                    "note": f"{len(responsive)} load(s) follow the voltage in the model (constant current or "
                            "impedance shares); the studies treat every load as constant power"})
    missing = [o for o in semantics.get("other_equipment") or [] if not o["in_computation"]]
    if missing:
        counts = _class_counts(missing)
        out.append({"kind": "equipment_not_in_computation",
                    "element": ", ".join(f"{n} {c}" for c, n in sorted(counts.items())),
                    "note": "the model holds equipment pandapower's converter has no place for; the studies do not "
                            "see it, e.g. " + ", ".join(o["name"] for o in missing[:3])})
    return out


SOLVED_TOLERANCE = {"vm_pu": 0.005, "angle_deg": 0.5, "mw": 1.0}


def solved_state_check(net_solved, cim) -> dict | None:
    """CONDUCTOR's power flow against the source tool's own solution (the model's SV profile), on the network as
    loaded: each node's voltage (magnitude, and angle relative to the supply point's) and each injection's power
    (loads, units, shunts, the supply point). A close match says the conversion kept the model electrically
    intact; a difference says where it did not. `net_solved`: the network as loaded after a power flow."""
    if cim is None or not cim.of("SvVoltage") or "res_bus" not in net_solved or not len(net_solved.res_bus):
        return None
    sv_v = {}
    for x in cim.of("SvVoltage"):
        o = cim.get(x) or {}
        tn = cim.ref(x, "SvVoltage.TopologicalNode")
        if tn:
            sv_v[tn] = (num_or(o.get("SvVoltage.v"), None), num_or(o.get("SvVoltage.angle"), None))
    rows, unsolved = [], []
    for i, oid in net_solved.bus.get("origin_id", pd.Series(dtype=object)).items():
        if not isinstance(oid, str):
            continue
        node = cm_norm(oid)
        tn = node if cim.cls(node) == "TopologicalNode" else cim.ref(node, "ConnectivityNode.TopologicalNode")
        v, ang = sv_v.get(tn, (None, None))
        kv = net_solved.bus.at[i, "vn_kv"]
        if v is None or not kv:
            continue
        if pd.isna(net_solved.res_bus.at[i, "vm_pu"]):
            # The model solved this node; CONDUCTOR's power flow did not reach it (cut off as converted).
            if v > 0:
                unsolved.append(bus_display_name(net_solved, int(i)))
            continue
        rows.append((int(i), float(net_solved.res_bus.at[i, "vm_pu"]), v / float(kv),
                     float(net_solved.res_bus.at[i, "va_degree"]), ang, cim.label(tn) or tn))
    if not rows:
        return None
    # Angles are relative: aligned at the supply point's bus (else at the median difference).
    ref_bus = int(net_solved.ext_grid.bus.iloc[0]) if len(net_solved.ext_grid) else None
    ref = next(((r[3] - r[4]) for r in rows if r[0] == ref_bus and r[4] is not None), None)
    if ref is None:
        diffs = sorted(r[3] - r[4] for r in rows if r[4] is not None)
        ref = diffs[len(diffs) // 2] if diffs else 0.0
    names = {int(i): bus_display_name(net_solved, int(i)) for i, *_ in rows}
    def angle_gap(d: float) -> float:          # angles compare modulo a full turn
        return abs((d + 180.0) % 360.0 - 180.0)
    volt = sorted(((abs(r[1] - r[2]), angle_gap(r[3] - r[4] - ref) if r[4] is not None else 0.0, r) for r in rows),
                  key=lambda x: -x[0])
    worst_v = [{"bus": names[r[0]], "node_in_model": r[5], "vm_pu": round(r[1], 4), "model_vm_pu": round(r[2], 4)}
               for _, _, r in volt[:3]]
    # Injections: the model's flows at each injection's terminal (load sign: positive into the equipment).
    table_of = {}
    for table in ("load", "gen", "sgen", "ext_grid", "shunt", "ward", "xward", "motor", "storage"):
        df = net_solved[table] if table in net_solved else None
        if isinstance(df, pd.DataFrame) and len(df) and "origin_id" in df.columns:
            for i, o in df["origin_id"].items():
                if isinstance(o, str) and o.strip():
                    table_of[cm_norm(o)] = (table, i)
    inj = []
    for x in cim.of("SvPowerFlow"):
        term = cim.ref(x, "SvPowerFlow.Terminal")
        eq = cim.ref(term, "Terminal.ConductingEquipment") if term else None
        hit = table_of.get(eq or "")
        res = f"res_{hit[0]}" if hit else None
        if not hit or res not in net_solved or hit[1] not in net_solved[res].index:
            continue
        o = cim.get(x) or {}
        p_model, q_model = num_or(o.get("SvPowerFlow.p"), None), num_or(o.get("SvPowerFlow.q"), None)
        sign = 1.0 if hit[0] in ("load", "shunt", "ward", "xward", "motor", "storage") else -1.0  # pandapower's sign
        p_ours = net_solved[res].at[hit[1], "p_mw"]
        q_ours = net_solved[res].at[hit[1], "q_mvar"] if "q_mvar" in net_solved[res].columns else None
        if p_model is None or pd.isna(p_ours):
            continue
        d_p = abs(sign * float(p_ours) - p_model)
        d_q = abs(sign * float(q_ours) - q_model) if q_model is not None and q_ours is not None \
            and not pd.isna(q_ours) else 0.0
        inj.append((max(d_p, d_q), d_p, d_q, cim.label(eq) or eq, hit[0]))
    inj.sort(key=lambda x: -x[0])
    max_dv = volt[0][0]
    max_da = max(x[1] for x in volt)
    max_dpq = inj[0][0] if inj else 0.0
    matches = (max_dv <= SOLVED_TOLERANCE["vm_pu"] and max_da <= SOLVED_TOLERANCE["angle_deg"]
               and max_dpq <= SOLVED_TOLERANCE["mw"] and not unsolved)
    return {"matches": matches, "buses_compared": len(rows), "injections_compared": len(inj),
            **({"buses_not_solved": len(unsolved), "e.g. not solved": unsolved[:5]} if unsolved else {}),
            "max_vm_diff_pu": round(max_dv, 5), "max_angle_diff_deg": round(max_da, 4),
            "max_injection_diff_mw_mvar": round(max_dpq, 4), "tolerance": SOLVED_TOLERANCE,
            "largest_voltage_differences": worst_v,
            "largest_injection_differences": [{"element": e, "table": t, "dp_mw": round(dp, 4), "dq_mvar": round(dq, 4)}
                                              for _, dp, dq, e, t in inj[:3]],
            "note": ("CONDUCTOR's power flow on the network as loaded reproduces the source tool's solution "
                     "(the model's SV profile) within the tolerance: the conversion kept the model electrically "
                     "intact." if matches else
                     "CONDUCTOR's power flow on the network as loaded differs from the source tool's solution "
                     "(the model's SV profile) beyond the tolerance: something the studies use differs from the "
                     "model — see the largest differences.")}


def model_view(semantics: dict) -> dict:
    """What the CIM model the network came from holds, what reaches the computation and the layer, and what the
    studies ignore (for the agent: describe_network(about="model"))."""
    cimm = semantics.get("cim_model")
    if not cimm:
        return {"about": "model", "note": "The network did not come as a CIM/CGMES model, or its model is not kept: "
                                          "the CIM layer is built from the network's own data."}
    by_status: dict[str, list] = {}
    for r in cimm["coverage"]["classes"]:
        by_status.setdefault(r["status"], []).append(
            {"class": r["class"], "in_model": r["in_model"], "in_layer": r["in_layer"],
             **({"read_into": r["read_into"]} if r.get("read_into") else {})})
    missing = [o for o in semantics.get("other_equipment") or [] if not o["in_computation"]]
    return {"about": "model", "version": cimm.get("version"), "objects": cimm.get("objects"),
            "profiles": cimm.get("profiles"), "classes_by_status": by_status,
            **({"not_in_computation": [{"name": o["name"], "class": o["cim_class"], "buses": o["buses"],
                                        "mrid": o["mrid"]} for o in missing[:_MAX_LISTED]]} if missing else {}),
            "studies_ignore": model_caveats(semantics),
            "note": "described: the layer describes every object of the class; read_into: its content is in other "
                    "entries; solved_state: the source tool's power-flow solution; model_only: kept, not in the "
                    "layer. Any object can be looked up by its mRID: describe_network(about='around', element=<mRID>)."}


def _value_kinds(e: dict) -> dict:
    """An element's values as the checks need them: the kind of P and Q, and where Q comes from."""
    v = e.get("values")
    if not v:
        return {}
    return {"p_kind": v["p"]["kind"], "p_from": v["p"]["from"], "q_kind": v["q"]["kind"], "q_from": v["q"]["from"]}


def compact(semantics: dict) -> dict:
    """The facts an answer is checked against, small enough to keep in each
    turn's record: names, kinds, voltages, ends, sources — with their bases."""
    # The feeder each bus is on (its head line's short name; None off any feeder), inferred from the topology.
    feeder = {b["index"]: _short_name((b.get("feeder") or {}).get("value")) or None for b in semantics["buses"]}
    return {
        "per_unit_case": semantics["per_unit_case"],
        "voltage_levels_kv": semantics["voltage_levels_kv"],
        "buses": {b["name"]: {"kv": b["kv"]["value"], "kv_basis": b["kv"]["basis"],
                              "substation": b["substation"]["value"], "feeder": feeder.get(b["index"])}
                  for b in semantics["buses"]},
        "branches": {b["name"]: {"type": b["element_type"], "ends": b["ends"]} for b in semantics["branches"]},
        "units": {u["name"]: {"source": u["source"]["value"], "basis": u["source"]["basis"], "bus": u["bus"],
                              "type": u["element_type"], "feeder": feeder.get(u["bus_index"]), **_value_kinds(u)}
                  for u in semantics["units"]},
        "loads": {d["name"]: {"category": d["category"]["value"], "basis": d["category"]["basis"], "bus": d["bus"],
                              "feeder": feeder.get(d["bus_index"]), **_value_kinds(d)}
                  for d in semantics.get("loads") or []},
        "switches": {s["name"]: {"open": s["open"]["value"], "normal_open": s["normal_open"]["value"],
                                 "role": s["role"]["value"]} for s in semantics.get("switches") or []},
        "feeders": [f["name"] for f in (semantics.get("containers") or {}).get("feeders") or []],
        "external_grids": [{"bus": e["bus"], "branches": e["branches"]} for e in semantics["external_grids"]],
    }


# ---------------------------------------------------------------------------
# Around an element: what it is connected to, in one answer
# ---------------------------------------------------------------------------

def _graph(net):
    """The network as connected now: in-service elements, open switches open."""
    import pandapower.topology as top
    return top.create_nxgraph(net, respect_switches=True, include_out_of_service=False, multi=True)


def _edge_name(net, names: dict, key) -> str:
    etype, idx = key[0], int(key[1])
    if (etype, idx) in names:
        return names[(etype, idx)]
    if etype == "switch":
        return f"bus coupler (switch {idx})"
    if etype == "trafo3w":
        raw = clean_label(net.trafo3w.at[idx, "name"]) if "name" in net.trafo3w.columns else ""
        return f"3-winding transformer {raw or idx}"
    return f"{etype} {idx}"


def _route(net, graph, names: dict, sources: list[int], target: int) -> list[tuple] | None:
    """The branches from the nearest external grid to `target`, in order, as
    (element type, name). Bus couplers are left out: they join busbars inside
    a substation."""
    import networkx as nx
    best = None
    for s in sources:
        try:
            path = nx.shortest_path(graph, s, target)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue
        if best is None or len(path) < len(best):
            best = path
    if best is None:
        return None
    keys = [next(iter(graph.get_edge_data(a, b))) for a, b in zip(best, best[1:])]
    return [(k[0], _edge_name(net, names, k)) for k in keys if k[0] != "switch"]


def _cut_off(net, graph, sources: list[int], *, edges=(), nodes=()) -> set[int]:
    """The buses that lose every connection to the external grid when `edges`
    (keys) or `nodes` are taken out — topology only."""
    import networkx as nx
    before = set().union(*(nx.node_connected_component(graph, s) for s in sources if s in graph)) if sources else set()
    g = graph.copy()
    for a, b, k in list(g.edges(keys=True)):
        if (k[0], int(k[1])) in edges:
            g.remove_edge(a, b, k)
    g.remove_nodes_from([n for n in nodes if n in g])
    after = set().union(*(nx.node_connected_component(g, s) for s in sources if s in g)) if sources else set()
    return (before - after) | (set(nodes) & before)


def scope_buses(semantics: dict, scope: str) -> tuple[set[int], str, str]:
    """The buses a scope names, by the CIM layer: a voltage level ("20 kV"), a
    feeder (by its name) or a substation. Returns ({bus index}, the kind —
    "level", "feeder" or "substation" — and what it covers, in words). Raises
    ValueError with the choices when nothing matches."""
    said = str(scope or "").strip()
    containers = semantics.get("containers") or {}
    by_name = {b["name"]: b["index"] for b in semantics["buses"]}
    levels = [float(v) for v in semantics.get("voltage_levels_kv") or []]
    m = re.fullmatch(r"(?:the\s+)?(\d+(?:\.\d+)?)\s*kv(?:\s+level)?", said, re.IGNORECASE)
    if m:
        kv = float(m.group(1))
        if not any(abs(kv - level) < 1e-6 for level in levels):
            raise ValueError(f"This network has no {kv:g} kV level; its levels are "
                             + (", ".join(f"{v:g} kV" for v in sorted(levels, reverse=True)) or "not stated") + ".")
        buses = {int(b["index"]) for b in semantics["buses"]
                 if b["kv"]["value"] is not None and abs(float(b["kv"]["value"]) - kv) < 1e-6}
        return buses, "level", f"the {kv:g} kV level"
    folded = re.sub(r"^(?:the\s+)?(?:feeder|substation)\s+", "", said, flags=re.IGNORECASE).casefold()
    # The name first, as it is said: "the Line 12-13 feeder", "the Bus 5 substation".
    folded = re.sub(r"^(?:the\s+)?(.+?)\s+(?:feeder|substation)$", r"\1", folded, flags=re.IGNORECASE)
    for f in containers.get("feeders") or []:
        if f["name"].casefold() == folded or f["name"].split(" | ")[0].casefold() == folded:
            return ({by_name[n] for n in f["buses"] if n in by_name}, "feeder",
                    f"feeder {f['name'].split(' | ')[0]}")
    for sub in containers.get("substations") or []:
        if str(sub["name"]).casefold() == folded:
            return ({by_name[n] for level in sub["voltage_levels"] for n in level["buses"] if n in by_name},
                    "substation", f"substation {sub['name']}")
    feeders = [f["name"].split(" | ")[0] for f in containers.get("feeders") or []]
    raise ValueError(
        f"No feeder, substation or voltage level is called {said!r}. "
        + (f"Feeders: {', '.join(feeders[:8])}. " if feeders else "This network has no feeders (none radial). ")
        + (f"Voltage levels: {', '.join(f'{v:g} kV' for v in sorted(levels, reverse=True))}. " if levels else "")
        + "Substations: by name (describe_network about='substations').")


def scope_branches(semantics: dict, scope: str) -> tuple[set[tuple[str, int]], str]:
    """The lines and transformers a scope names (`scope_buses`): a voltage
    level's lines and the transformers that connect it; a feeder's branches
    among the buses it supplies, and its head line; a substation's branches
    with an end in it. Returns ({(type, index)}, what the scope covers, in
    words). Raises ValueError with the choices when nothing matches."""
    buses, kind, words = scope_buses(semantics, scope)
    branches = [b for b in semantics["branches"] if b["element_type"] in ("line", "trafo")]
    if kind == "level":
        out = {(b["element_type"], int(b["index"])) for b in branches if buses & set(b["ends_index"])}
        return out, f"{words}: its lines and the transformers that connect it"
    if kind == "feeder":
        head = words.removeprefix("feeder ").casefold()
        out = {(b["element_type"], int(b["index"])) for b in branches
               if set(b["ends_index"]) <= buses or b["name"].split(" | ")[0].casefold() == head}
        return out, f"{words}: the branches among the {len(buses)} buses it supplies"
    out = {(b["element_type"], int(b["index"])) for b in branches if buses & set(b["ends_index"])}
    return out, f"{words}: the branches with an end in it"


# What a selection asks for, by the words people use (`select`).
_KIND_WORDS = (("units", r"units?|generators?|plants?|farms?|parks?|machines?|ders?"),
               ("loads", r"loads?|customers?|consumers?|demands?"),
               ("branches", r"branch(?:es)?|lines?|transformers?|trafos?|cables?"))
_GROUPS = {"renewable": set(RENEWABLE), "renewables": set(RENEWABLE), "variable": set(VARIABLE),
           "weather-dependent": set(VARIABLE), "conventional": {"thermal", "nuclear", "conventional"},
           "dispatchable": {"thermal", "nuclear", "conventional", "hydro", "bioenergy", "storage"}}


def _qualifier(words: str, kind: str) -> tuple[set[str] | None, str]:
    """The sources (units) or categories (loads) a qualifier names, and how it reads. None: every one."""
    words = words.strip().casefold().replace("-", " ")
    if not words or words in ("all", "every", "the", "all the"):
        return None, ""
    words = re.sub(r"^(?:all|every|the|all the)\s+", "", words)
    if words in ("unknown", "undescribed", "unidentified"):
        return {"unknown"}, "whose kind the network does not say"
    if kind == "units":
        if words in _GROUPS:
            return _GROUPS[words], words
        for value, pattern in _TYPE_PATTERNS:
            if re.fullmatch(r"(?:" + pattern + r")", words):
                return {value}, value
    if kind == "loads":
        for value, pattern in _LOAD_TYPE_PATTERNS:
            if re.fullmatch(r"(?:" + pattern + r")", words):
                return {value}, value.replace("_", " ")
    raise ValueError(f"{words!r} is not a kind of {kind} the CIM layer knows.")


def select(semantics: dict, phrase: str) -> dict:
    """The elements a phrase names, by the CIM layer, not by their names:
    "solar units", "the wind units on feeder Line 1-2", "industrial loads at 20 kV",
    "units in substation Bus 5", "loads whose category is unknown" ("unknown
    loads"), "lines at the 20 kV level". A qualifier is a source or a group
    (renewable, variable, conventional, dispatchable) for units, a category for
    loads; a place is a feeder, a substation or a voltage level (`scope_buses`).
    Returns {"kind", "elements": [{"name", "type", "index", "bus", "is", "basis"}],
    "asked", "means"}. Raises ValueError, with the choices, for a phrase it
    cannot read or one that matches nothing."""
    said = re.sub(r"\s+", " ", str(phrase or "").strip())
    place = None
    m = re.search(r"\s+(?:on|at|in|of|within|along)\s+(?:the\s+)?(?P<place>(?:feeder|substation)\s+.+|"
                  r"\d+(?:\.\d+)?\s*kv(?:\s+level)?|.+)$", said, re.IGNORECASE)
    head = said
    if m:
        place, head = m.group("place"), said[:m.start()]
    head = re.sub(r"\s+whose (?:source|category|kind) is unknown$", "", head, flags=re.IGNORECASE) \
        if re.search(r"whose (?:source|category|kind) is unknown$", head, re.IGNORECASE) else head
    unknown_tail = bool(re.search(r"whose (?:source|category|kind) is unknown", said, re.IGNORECASE))
    kind, qualifier_words = None, ""
    for name, pattern in _KIND_WORDS:
        k = re.search(r"\b(?:" + pattern + r")$", head.strip(), re.IGNORECASE)
        if k:
            kind, qualifier_words = name, head.strip()[:k.start()]
            # The word itself may qualify: "transformers" are branches of one type, "lines" of another.
            noun = k.group(0).casefold()
            break
    if kind is None:
        raise ValueError(f"Could not read {phrase!r}: say units, loads or branches, with what kind and where — "
                         "e.g. 'solar units on feeder Line 1-2', 'industrial loads at 20 kV', 'units in "
                         "substation Bus 5'.")
    wanted, reads = ({"unknown"}, "whose kind the network does not say") if unknown_tail else \
        _qualifier(qualifier_words, kind)
    buses, where = None, ""
    if place:
        buses, _kind, where = scope_buses(semantics, place)
    out = []
    if kind == "units":
        for u in semantics.get("units") or []:
            value, basis = u["source"]["value"], u["source"]["basis"]
            if wanted is not None and not ((("unknown" in wanted) and basis == UNKNOWN)
                                           or (value in wanted and basis != UNKNOWN)):
                continue
            if buses is not None and u["bus_index"] not in buses:
                continue
            out.append({"name": u["name"], "type": u["element_type"], "index": u["index"],
                        "bus": u.get("bus"), "is": value, "basis": basis})
    elif kind == "loads":
        for d in semantics.get("loads") or []:
            value, basis = d["category"]["value"], d["category"]["basis"]
            if wanted is not None and not ((("unknown" in wanted) and basis == UNKNOWN)
                                           or ((value in wanted or (value == "commercial_industrial" and wanted
                                                                    & {"commercial", "industrial"}))
                                               and basis != UNKNOWN)):
                continue
            if buses is not None and d["bus_index"] not in buses:
                continue
            out.append({"name": d["name"], "type": "load", "index": d["index"], "bus": d.get("bus"),
                        "is": value, "basis": basis})
    else:
        only = "trafo" if noun.startswith(("transformer", "trafo")) else "line" if noun.startswith(("line", "cable")) \
            else None
        for b in semantics.get("branches") or []:
            if b["element_type"] not in ("line", "trafo") or (only and b["element_type"] != only):
                continue
            if buses is not None and not buses & set(b["ends_index"]):
                continue
            out.append({"name": b["name"].split(" | ")[0], "type": b["element_type"], "index": b["index"],
                        "bus": " – ".join(b["ends"]), "is": b["element_type"], "basis": STATED})
    noun_words = kind if kind != "branches" else (noun if only else "branches")
    parts = (noun_words, reads) if reads.startswith("whose") else (reads, noun_words)
    means = " ".join(w for w in (*parts, f"in {where}" if where else "") if w)
    if not out:
        raise ValueError(f"No {means} in this network, by the CIM layer.")
    return {"kind": kind, "asked": said, "means": means, "elements": out}


# ---------------------------------------------------------------------------
# The layer checked against itself: what in the network's own data looks wrong
# ---------------------------------------------------------------------------

def _name_says(name: str, patterns) -> str | None:
    for value, pattern in patterns:
        if re.search(pattern, str(name or ""), re.IGNORECASE):
            return value
    return None


def data_issues(net, semantics: dict) -> list[dict]:
    """What in the network's own data looks inconsistent, found from the facts
    (no study run): each {"kind", "element", "note", "level"} — "check" for a
    likely error, "note" for something worth knowing. For any network:

    - a unit or a load whose name says one thing and whose stated type another
      ("Solar Farm North", typed a CHP unit) — which the agent must not trust;
    - a switch operated away from its normal state;
    - a line between buses at two voltage levels; a transformer whose rated
      voltages are not its buses';
    - loads with no path to a source; a load drawing negative power;
    - some loads with a category and others without; names used twice."""
    out: list[dict] = []
    for u in semantics.get("units") or []:
        f = u["source"]
        if f["basis"] == STATED and "declared" not in str(f.get("from")):
            says = _name_says(u["name"], _NAME_PATTERNS)
            if says and says != f["value"]:
                typed = re.search(r"\('([^']*)'\)", str(f.get("from") or ""))
                where = f" ('{typed.group(1)}')" if typed else f" ({f.get('from')})"
                out.append({"kind": "name_contradicts_type", "element": u["name"], "level": "check",
                            "note": f"its name says {says}, but its type in the network says {f['value']}{where}: "
                                    "the name is not to be trusted"})
    for d in semantics.get("loads") or []:
        f = d["category"]
        if f["basis"] == STATED and "declared" not in str(f.get("from")):
            says = _name_says(d["name"], _LOAD_NAME_PATTERNS)
            if says and says != f["value"] and not (f["value"] == "commercial_industrial"
                                                     and says in ("commercial", "industrial")):
                out.append({"kind": "name_contradicts_type", "element": d["name"], "level": "check",
                            "note": f"its name says {says.replace('_', ' ')}, but the network states "
                                    f"{f['value'].replace('_', ' ')}"})
        if d.get("p_mw") is not None and float(d["p_mw"]) < -1e-6:
            out.append({"kind": "negative_load", "element": d["name"], "level": "check",
                        "note": f"draws {float(d['p_mw']):g} MW (negative): a generator entered as a load?"})
    for sw in semantics.get("switches") or []:
        normal, now = (sw.get("normal_open") or {}).get("value"), (sw.get("open") or {}).get("value")
        if normal is not None and now is not None and normal != now:
            out.append({"kind": "switch_off_normal", "element": sw["name"], "level": "note",
                        "note": f"normally {'open' if normal else 'closed'}, {'open' if now else 'closed'} now"})
    kv = {b["index"]: b["kv"]["value"] for b in semantics.get("buses") or []}
    for b in semantics.get("branches") or []:
        if b["element_type"] == "line":
            ends = [kv.get(e) for e in b["ends_index"]]
            if None not in ends and abs(float(ends[0]) - float(ends[1])) > 1e-6:
                out.append({"kind": "line_across_levels", "element": b["name"].split(" | ")[0], "level": "check",
                            "note": f"joins buses at {ends[0]:g} kV and {ends[1]:g} kV: a line cannot change the "
                                    "voltage (a transformer entered as a line, or a bus's level wrong?)"})
        elif b["element_type"] == "trafo" and "trafo" in net and int(b["index"]) in net.trafo.index:
            row = net.trafo.loc[int(b["index"])]
            hv, lv = kv.get(int(row["hv_bus"])), kv.get(int(row["lv_bus"]))
            for side, rated, bus_kv in (("high", row.get("vn_hv_kv"), hv), ("low", row.get("vn_lv_kv"), lv)):
                if rated is not None and bus_kv is not None and rated == rated and bus_kv \
                        and abs(float(rated) - float(bus_kv)) / float(bus_kv) > 0.15:
                    out.append({"kind": "trafo_rating_vs_bus", "element": b["name"].split(" | ")[0],
                                "level": "check",
                                "note": f"its {side}-voltage side is rated {float(rated):g} kV, its bus is at "
                                        f"{float(bus_kv):g} kV"})
    try:
        import pandapower.topology as top
        lost = {int(b) for b in top.unsupplied_buses(net)}
    except Exception:  # noqa: BLE001 — a network the search cannot read says nothing here
        lost = set()
    cut = [d["name"] for d in semantics.get("loads") or [] if d.get("bus_index") in lost and d.get("in_service", True)]
    if cut:
        out.append({"kind": "loads_without_supply", "element": ", ".join(cut[:5]) + (" …" if len(cut) > 5 else ""),
                    "level": "check", "note": f"{len(cut)} load(s) with no path to a source as switched now"})
    # What the CIM model states that the studies do not use (a network that came as CIM).
    out += [{**c, "level": "note"} for c in model_caveats(semantics)]
    loads = semantics.get("loads") or []
    known = [d for d in loads if d["category"]["basis"] != UNKNOWN]
    if known and len(known) < len(loads):
        out.append({"kind": "categories_partly_known", "element": f"{len(loads) - len(known)} loads", "level": "note",
                    "note": f"{len(known)} of {len(loads)} loads have a category; the others follow the defaults "
                            "(declare them to use theirs)"})
    for kind_name, entries in (("bus", semantics.get("buses") or []), ("unit", semantics.get("units") or []),
                               ("load", loads)):
        seen: dict[str, int] = {}
        for e in entries:
            seen[e["name"]] = seen.get(e["name"], 0) + 1
        twice = sorted(n for n, k in seen.items() if k > 1)
        if twice:
            out.append({"kind": "duplicate_names", "element": ", ".join(twice[:5]), "level": "note",
                        "note": f"{len(twice)} {kind_name} name(s) used more than once: results tell them apart by "
                                "index, an answer by name cannot"})
    return out


def tie_switches(net, names: dict, lost: set[int]) -> list[tuple[int, str]]:
    """(switch index, label) of each open switch whose line joins the cut-off
    part to a supplied one: what could restore supply, by the topology."""
    out = []
    sw = getattr(net, "switch", None)
    if sw is None or not len(sw):
        return out
    for i, s in sw[~sw.closed.astype(bool)].iterrows():
        if s.et == "l" and int(s.element) in net.line.index:
            a, b = int(net.line.at[int(s.element), "from_bus"]), int(net.line.at[int(s.element), "to_bus"])
        elif s.et == "b":
            a, b = int(s.bus), int(s.element)
        else:
            continue
        if (a in lost) != (b in lost):
            what = names.get(("line", int(s.element))) if s.et == "l" else f"bus coupler {int(s.bus)}–{int(s.element)}"
            label = clean_label(s.get("name")) if "name" in sw.columns else ""
            out.append((int(i), f"{what} (switch {label or int(i)}, open at {bus_display_name(net, int(s.bus))})"))
    return out


def _ties(net, names: dict, lost: set[int]) -> list[str]:
    """The labels of `tie_switches`."""
    return [label for _i, label in tie_switches(net, names, lost)]


# At most this many ties are checked with a power flow (each is one).
MAX_TIES_CHECKED = 5


def around(net, semantics: dict, etype: str, idx, check_tie=None) -> dict:
    """An element and everything it is connected to, in one answer: its
    substation (levels, buses, units and loads there, branches leaving and
    where they lead, the external grid), its supply route from the external
    grid and its feeder (inferred from the topology), and — for a bus, a
    branch or a substation — what would be cut off without it, with the open
    ties that could restore it. `check_tie(outage, switch, label, lost)`, when
    given, checks each tie with a power flow (the backend's: this module runs
    none) and its result is kept under `ties_checked`. `etype`: bus, line, trafo, gen, sgen, storage,
    load or substation (`idx`: its name)."""
    import networkx as nx
    buses = {b["index"]: b for b in semantics["buses"]}
    names = {(b["element_type"], b["index"]): b["name"] for b in semantics["branches"]}
    sub_of = {i: b["substation"]["value"] for i, b in buses.items()}
    # The element, and the buses it sits on.
    if etype == "substation":
        anchor = sorted(i for i, s in sub_of.items() if s == idx)
        entry = {"name": idx, "kind": "substation", "basis": semantics["substations"]["basis"],
                 "from": semantics["substations"]["from"]}
    elif etype == "bus":
        anchor, entry = [int(idx)], {**buses[int(idx)], "kind": "bus"}
    elif etype in ("line", "trafo"):
        br = next(b for b in semantics["branches"] if b["element_type"] == etype and b["index"] == int(idx))
        anchor, entry = list(br["ends_index"]), {**br, "kind": "line" if etype == "line" else "transformer"}
    elif etype == "load":
        d = next(x for x in semantics["loads"] if x["index"] == int(idx))
        anchor, entry = [d["bus_index"]], {**d, "kind": "load"}
    else:
        u = next(x for x in semantics["units"] if x["element_type"] == etype and x["index"] == int(idx))
        anchor, entry = [u["bus_index"]], {**u, "kind": "unit"}
    entry = {k: v for k, v in entry.items() if k not in ("index", "ends_index", "bus_index")}

    def listed(items):
        return items[:_MAX_LISTED] + ([f"… {len(items) - _MAX_LISTED} more"] if len(items) > _MAX_LISTED else [])

    def station(name) -> dict:
        """One substation: levels, buses, what is connected there, what leaves it."""
        home = {i for i, s in sub_of.items() if s == name}
        leaving = []
        for b in semantics["branches"]:
            inside = [e in home for e in b["ends_index"]]
            if any(inside) and not all(inside):
                far = b["ends_index"][1] if inside[0] else b["ends_index"][0]
                leaving.append({"branch": b["name"], "to": buses[far]["name"], "to_substation": sub_of.get(far),
                                "in_service": b["in_service"]})
        return {
            "name": name, "basis": semantics["substations"]["basis"],
            "voltage_levels_kv": sorted({buses[i]["kv"]["value"] for i in home
                                         if buses[i]["kv"]["value"] is not None}, reverse=True),
            "buses": listed(sorted(buses[i]["name"] for i in home)),
            "units": listed([{"name": u["name"], "bus": u["bus"], "source": u["source"]["value"],
                              "basis": u["source"]["basis"]} for u in semantics["units"] if u["bus_index"] in home]),
            "loads": listed([{"name": d["name"], "bus": d["bus"], "category": d["category"]["value"],
                              "basis": d["category"]["basis"], "p_mw": d["p_mw"]}
                             for d in semantics["loads"] if d["bus_index"] in home]),
            "branches_leaving": listed(leaving),
            "transformers_inside": listed([b["name"] for b in semantics["branches"] if b["element_type"] == "trafo"
                                           and set(b["ends_index"]) <= home]),
            "external_grid_here": any(e["bus_index"] in home for e in semantics["external_grids"]),
        }
    # A line joins two substations: each end's, in order. Anything else sits in one.
    stations = list(dict.fromkeys(sub_of.get(a) for a in anchor))

    graph = _graph(net)
    sources = [e["bus_index"] for e in semantics["external_grids"]
               if bool(net.ext_grid.loc[net.ext_grid.bus == e["bus_index"], "in_service"].any())]
    rule = ("inferred from the topology as switched now: the shortest route by number of branches from an "
            "external grid; the feeder is the first line after the last transformer on it")
    if not sources:
        supply = {"basis": UNKNOWN, "note": "The network has no external grid in service: no supply route."}
    else:
        # The route to the far end of a branch includes the branch itself.
        # Over the external grids that reach it: with several in separate islands, one that cannot reach
        # the bus must not hide the one that does.
        def hops(b):
            reach = []
            for s in sources:
                try:
                    reach.append(nx.shortest_path_length(graph, s, b))
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    continue
            return min(reach) if reach else -1
        target = max(anchor, key=hops) if anchor else None
        route = _route(net, graph, names, sources, target) if target is not None and target in graph else None
        if route is None:
            supply = {"basis": STATED, "route": None,
                      "note": "Not connected to any external grid as the network is switched now (islanded or out "
                              "of service)."}
        else:
            comp = graph.subgraph(nx.node_connected_component(graph, target))
            loops = nx.number_of_edges(nx.Graph(comp)) - comp.number_of_nodes() + 1
            # Feeders belong to the element's own voltage level: the part of the
            # network reached from it without crossing a transformer.
            level = graph.copy()
            level.remove_edges_from([(a, b, k) for a, b, k in graph.edges(keys=True) if k[0] in ("trafo", "trafo3w")])
            island = nx.Graph(level.subgraph(nx.node_connected_component(level, target)))
            level_loops = island.number_of_edges() - island.number_of_nodes() + 1
            last_trafo = max((k for k, (t, _n) in enumerate(route) if t in ("trafo", "trafo3w")), default=-1)
            feeder = next((n for t, n in route[last_trafo + 1:] if t == "line"), None)
            supply = {"basis": INFERRED, "from": rule,
                      "route_from_external_grid": listed([n for _t, n in route]),
                      # A feeder is a radial idea: in a meshed network power comes over several.
                      "feeder": (feeder or ("— (inside the substation where the supply steps down)" if route
                                            else "— (at the external grid's substation)"))
                      if level_loops <= 0 else None,
                      "radial_at_its_voltage_level": level_loops <= 0,
                      **({"note": f"The network above it is meshed ({loops} independent loop(s) in all): power may "
                                  "flow over other routes too; this is the shortest one."
                                  + ("" if level_loops <= 0 else " Its own voltage level is meshed too, so it has "
                                     "no single feeder.")} if loops > 0 else {})}
    out = {"about": "around", "element": entry,
           **({"substations_at_its_ends": [station(n) for n in stations]} if len(stations) > 1 else
              {"substation": station(stations[0]) if stations else None}),
           "supply": supply}
    if etype in ("bus", "line", "trafo", "substation") and sources:
        if etype in ("line", "trafo"):
            lost = _cut_off(net, graph, sources, edges={(etype, int(idx))})
        else:
            lost = _cut_off(net, graph, sources, nodes=set(anchor))
        lost_units = [u["name"] for u in semantics["units"] if u["bus_index"] in lost]
        lost_loads = [d for d in semantics["loads"] if d["bus_index"] in lost]
        out["without_it"] = {
            "basis": INFERRED,
            "from": "topology only: the buses that lose every connection to an external grid (no power flow "
                    "run); the load is their demand at the network's current operating point",
            "buses_cut_off": listed(sorted(buses[i]["name"] for i in lost if i in buses)),
            "units_cut_off": listed(lost_units),
            "loads_cut_off": listed([d["name"] for d in lost_loads]),
            "load_cut_off_mw": round(sum(d["p_mw"] for d in lost_loads), 3),
            **({"open_ties_that_could_restore": _ties(net, names, lost)} if lost else {}),
        }
        ties = tie_switches(net, names, lost) if lost and check_tie else []
        if ties:
            outage = ({"etype": etype, "index": int(idx)} if etype in ("line", "trafo")
                      else {"etype": "buses", "buses": sorted(int(b) for b in anchor)})
            out["without_it"]["ties_checked"] = [check_tie(outage, switch, label, lost)
                                                 for switch, label in ties[:MAX_TIES_CHECKED]]
            out["without_it"]["ties_checked_by"] = ("a power flow per tie at the current operating point, with the "
                                                    "element out and the tie closed; limits are the network's")
    return out


# ---------------------------------------------------------------------------
# Links from a study's result to the layer: what each element it names is
# ---------------------------------------------------------------------------

# How many elements of one result are linked (the first named, in the result's own order).
MAX_LINKED = 8
MAX_STUDIED = 3          # elements a study's arguments name, linked first
_LISTED_HERE = 4          # units or loads listed at one bus; more are counted
_TYPE_ORDER = ("bus", "branch", "unit", "load")


def layer_version(semantics: dict) -> str:
    """A short digest of what the layer says — each element's kind, voltage, substation, feeder, ends, source or
    category with its basis, and the switches' state — not of the operating point. A study's links carry it: when
    the layer changes (a declaration, a proposal edit), the version does, so a link can say it is out of date."""
    import hashlib
    facts = {
        "buses": sorted((b["name"], b["kv"]["value"], b["kv"]["basis"], b["substation"]["value"],
                         (b.get("feeder") or {}).get("value")) for b in semantics["buses"]),
        "branches": sorted((b["name"], b["element_type"], *b["ends"], b["in_service"]) for b in semantics["branches"]),
        "units": sorted((u["name"], u["source"]["value"], u["source"]["basis"], u["bus"]) for u in semantics["units"]),
        "loads": sorted((d["name"], d["category"]["value"], d["category"]["basis"], d["bus"])
                        for d in semantics.get("loads") or []),
        "switches": sorted((s["name"], bool(s["open"]["value"]), bool(s["normal_open"]["value"]))
                           for s in semantics.get("switches") or []),
    }
    return hashlib.sha1(json.dumps(facts, default=str).encode("utf-8")).hexdigest()[:12]


def _short_name(name) -> str:
    return str(name or "").split(" | ")[0]


def link_targets(semantics: dict, candidates, limit: int = MAX_LINKED) -> list[dict]:
    """The layer's elements that `candidates` name — [(text, kind hint or None)], in the result's order — as
    [{name, type, index, mrid}], each once, at most `limit`. A text names an element when it is its name (a
    branch's short name too) or begins with it ("Bus 9 at 0.93 p.u."). The hint ("bus", "branch", "unit",
    "load") decides between elements of different kinds sharing a name; without one, a bus is taken first."""
    index: dict[str, dict[str, dict]] = {k: {} for k in _TYPE_ORDER}
    for b in semantics["buses"]:
        index["bus"].setdefault(b["name"], {"name": b["name"], "type": "bus", "index": b["index"], "mrid": b["mrid"]})
    for b in semantics["branches"]:
        entry = {"name": b["name"], "type": b["element_type"], "index": b["index"], "mrid": b["mrid"]}
        index["branch"].setdefault(b["name"], entry)
        index["branch"].setdefault(_short_name(b["name"]), entry)
    for u in semantics["units"]:
        index["unit"].setdefault(u["name"], {"name": u["name"], "type": u["element_type"], "index": u["index"],
                                             "mrid": u["mrid"]})
    for d in semantics.get("loads") or []:
        index["load"].setdefault(d["name"], {"name": d["name"], "type": "load", "index": d["index"], "mrid": d["mrid"]})
    # Longest first, so "Bus 1" never claims "Bus 12 at 0.93 p.u.".
    prefixes = sorted({n for names in index.values() for n in names}, key=len, reverse=True)

    def lookup(name: str, hint) -> dict | None:
        order = ((hint,) if hint in index else ()) + _TYPE_ORDER
        return next((index[k][name] for k in order if name in index[k]), None)

    out, seen = [], set()
    for text, hint in candidates:
        t = str(text or "").strip()
        if not t or len(t) > 120:
            continue
        hit = lookup(t, hint)
        if hit is None:
            head = next((n for n in prefixes if t.startswith(n) and len(t) > len(n) and not t[len(n)].isalnum()
                         and t[len(n)] not in "_-."), None)
            hit = lookup(head, hint) if head else None
        if hit is None or hit["mrid"] in seen:
            continue
        seen.add(hit["mrid"])
        out.append(hit)
        if len(out) >= limit:
            break
    return out


def ref_targets(semantics: dict, refs) -> list[dict]:
    """`link_targets`' entries for elements given by type and index — what a study's arguments name ("element_type":
    "line", "element_index": 10) rather than its result. Unknown ones are left out."""
    def pool(kind: str) -> list[dict]:
        if kind == "bus":
            return semantics["buses"]
        if kind == "load":
            return semantics.get("loads") or []
        if kind in ("line", "trafo"):
            return [b for b in semantics["branches"] if b["element_type"] == kind]
        if kind in ("gen", "sgen", "storage"):
            return [u for u in semantics["units"] if u["element_type"] == kind]
        return []

    out = []
    for ref in refs or []:
        kind = str((ref or {}).get("type") or "").lower()
        try:
            idx = int(ref.get("index"))
        except (TypeError, ValueError):
            continue
        hit = next((e for e in pool(kind) if e["index"] == idx), None)
        if hit is not None:
            out.append({"name": hit["name"], "type": kind, "index": idx, "mrid": hit["mrid"]})
    return out


def feeder_ties(semantics: dict, feeders) -> dict:
    """{feeder (short name): [its normally open points]} — each switch, the branch it is on and the bus it is at, and
    the feeder on the other side ("tie") or none (a loop within the feeder). From the layer's switch roles."""
    full = {_short_name(f["name"]): f["name"] for f in (semantics.get("containers") or {}).get("feeders") or []}
    out: dict[str, list] = {}
    for short in dict.fromkeys(f for f in feeders if f):
        name = full.get(short)
        if not name:
            continue
        rows = []
        for sw in semantics.get("switches") or []:
            role = str((sw.get("role") or {}).get("value") or "")
            if not (sw.get("normal_open") or {}).get("value") or name not in role:
                continue
            others = [k for k, v in full.items() if v in role and k != short]
            where = sw.get("where") or {}
            rows.append({"switch": sw["name"], "on": _short_name(where.get("element")) or None, "at": where.get("at"),
                         "to_feeder": others[0] if others else None,
                         "open_now": bool((sw.get("open") or {}).get("value"))})
        if rows:
            out[short] = rows
    return out


def context(net, semantics: dict, targets: list[dict], demand: dict | None = None) -> dict:
    """{mrid: what the layer says the element is} for `link_targets`' elements, each fact with its basis:

    - a bus: voltage, substation, feeder and where on it (counted in buses from the substation), the units and
      loads there;
    - a line or transformer: its ends, voltage, feeder, and which buses and loads lose supply without it
      (topology only: no demand, which is the operating point's, not the layer's);
    - a unit or a load: its source or category, its bus and feeder.

    Topology only — no power flow. The feeder is the layer's inference (radial voltage levels only).

    `demand`: {load index: MW} at the moment a study ran (the backend reads it as the study did); with it, a bus
    says the demand there (`load_mw`) and a branch the demand it cuts off (`load_cut_off_mw`). Without it, no MW:
    demand is the operating point's, and a figure from another moment would pass for the study's."""
    import networkx as nx
    buses = {b["index"]: b for b in semantics["buses"]}
    graph = _graph(net)
    sources = [int(b) for b, on in zip(net.ext_grid.bus, net.ext_grid.in_service) if bool(on) and int(b) in graph]
    depth: dict[int, int] = {}
    for s in sources:
        for bus, d in nx.single_source_shortest_path_length(graph, s).items():
            depth[int(bus)] = min(d, depth.get(int(bus), d))
    on_feeder: dict[str, list[int]] = {}
    for b in semantics["buses"]:
        f = (b.get("feeder") or {}).get("value")
        if f:
            on_feeder.setdefault(f, []).append(b["index"])
    units_at: dict[int, list[dict]] = {}
    for u in semantics["units"]:
        units_at.setdefault(u["bus_index"], []).append(u)
    loads_at: dict[int, list[dict]] = {}
    for d in semantics.get("loads") or []:
        loads_at.setdefault(d["bus_index"], []).append(d)

    def basis(f: dict) -> dict:
        return {"value": f.get("value"), "basis": f.get("basis")}

    def feeder_of(bus: int | None) -> dict | None:
        f = (buses.get(bus) or {}).get("feeder") or {}
        if not f.get("value"):
            why = str(f.get("from") or "")
            return {"value": None, "basis": f.get("basis"), "why": why.removeprefix("no feeder: ")} if why else None
        out = {"value": _short_name(f["value"]), "basis": f["basis"]}
        members = [b for b in on_feeder.get(f["value"], []) if b in depth]
        if bus in depth and members:
            top = min(depth[b] for b in members)
            out["position"] = depth[bus] - top + 1
            out["of"] = max(depth[b] for b in members) - top + 1
        return out

    def mw(loads: list[dict]) -> float:
        return round(sum(demand.get(d["index"], 0.0) for d in loads if d.get("in_service", True)), 3)

    def values(e: dict) -> dict:
        """How the element's P and Q are obtained from the data (measured, allocated, derived…), when known."""
        v = e.get("values") or {}
        if not v.get("p"):
            return {}
        return {"p_kind": v["p"].get("kind"), "q_kind": (v.get("q") or {}).get("kind"), "p_from": v["p"].get("from")}

    def here(bus: int) -> dict:
        units, loads = units_at.get(bus, []), loads_at.get(bus, [])
        return {"units": [{"name": u["name"], **basis(u["source"]), **values(u)} for u in units[:_LISTED_HERE]],
                "unit_count": len(units),
                "loads": [{"name": d["name"], **basis(d["category"]), **values(d)} for d in loads[:_LISTED_HERE]],
                "load_count": len(loads), **({"load_mw": mw(loads)} if demand is not None and loads else {})}

    out = {}
    for t in targets:
        kind, idx = t["type"], int(t["index"])
        if kind == "bus":
            b = buses[idx]
            entry = {"name": b["name"], "kind": "bus", "kv": basis(b["kv"]), "substation": basis(b["substation"]),
                     "feeder": feeder_of(idx), **here(idx)}
            if not (entry["feeder"] or {}).get("value") and idx in graph:
                # Off any feeder (a meshed level, a busbar): the buses next to it say where it is.
                near = sorted({buses[int(n)]["name"] for n in graph.neighbors(idx) if int(n) in buses})
                entry["next_to"] = near[:_LISTED_HERE] + ([f"{len(near) - _LISTED_HERE} more"]
                                                          if len(near) > _LISTED_HERE else [])
            out[t["mrid"]] = entry
        elif kind in ("line", "trafo"):
            br = next(b for b in semantics["branches"] if b["element_type"] == kind and b["index"] == idx)
            a, z = br["ends_index"]
            # The end further from the external grid is the one the branch feeds.
            far = z if depth.get(z, -1) >= depth.get(a, -1) else a
            entry = {"name": br["name"], "kind": "line" if kind == "line" else "transformer", "ends": br["ends"],
                     "kv": br.get("kv"), "feeder": feeder_of(far)}
            if sources and br["in_service"]:
                lost = _cut_off(net, graph, sources, edges={(kind, idx)})
                lost_loads = [d for d in semantics.get("loads") or [] if d["bus_index"] in lost]
                # What loses supply is the topology's, the same at any hour; its MW only at the study's moment.
                entry["without_it"] = {"basis": INFERRED, "buses_cut_off": len(lost),
                                       "loads_cut_off": [d["name"] for d in lost_loads[:3]],
                                       "load_count_cut_off": len(lost_loads),
                                       **({"load_cut_off_mw": mw(lost_loads)} if demand is not None and lost_loads
                                          else {})}
            out[t["mrid"]] = entry
        elif kind == "load":
            d = next(x for x in semantics.get("loads") or [] if x["index"] == idx)
            out[t["mrid"]] = {"name": d["name"], "kind": "load", "category": basis(d["category"]), "bus": d["bus"],
                              "feeder": feeder_of(d["bus_index"]), **values(d)}
        else:
            u = next(x for x in semantics["units"] if x["element_type"] == kind and x["index"] == idx)
            out[t["mrid"]] = {"name": u["name"], "kind": "unit", "source": basis(u["source"]), "bus": u["bus"],
                              "feeder": feeder_of(u["bus_index"]), **values(u)}
    return out


# How many of the elements one result names are placed together (`locate`, `where`): cheap facts only.
MAX_GROUPED = 80


def locate(semantics: dict, targets: list[dict]) -> list[dict]:
    """For `where`: each element's kind, feeder, voltage and substation, read from the layer as it is — no
    topology search, so every element a result names can be placed, not only the few described in full. A
    branch's feeder is its ends' (the one on a feeder; a branch leaving the busbar takes its far end's)."""
    buses = {b["index"]: b for b in semantics["buses"]}
    feeder = {i: _short_name((b.get("feeder") or {}).get("value")) or None for i, b in buses.items()}
    units = {(u["element_type"], u["index"]): u for u in semantics["units"]}
    loads = {d["index"]: d for d in semantics.get("loads") or []}
    branches = {(b["element_type"], b["index"]): b for b in semantics["branches"]}
    out = []
    for t in targets:
        kind, idx = t["type"], int(t["index"])
        if kind == "bus" and idx in buses:
            b = buses[idx]
            out.append({"kind": "bus", "feeder": feeder.get(idx), "kv": b["kv"]["value"],
                        "substation": b["substation"]["value"]})
        elif kind in ("line", "trafo") and (kind, idx) in branches:
            br = branches[(kind, idx)]
            ends = [feeder.get(e) for e in br["ends_index"]]
            on = [f for f in ends if f]
            out.append({"kind": "line" if kind == "line" else "transformer", "kv": br.get("kv"),
                        "feeder": on[-1] if on else None})
        elif kind == "load" and idx in loads:
            out.append({"kind": "load", "feeder": feeder.get(loads[idx]["bus_index"])})
        elif (kind, idx) in units:
            out.append({"kind": "unit", "feeder": feeder.get(units[(kind, idx)]["bus_index"])})
    return out


def where(entries) -> list[str]:
    """Where the elements sit together, when two or more do — by feeder, voltage level and substation, with how
    many share each ("feeder: 22 of 25 on Line 1-2, 3 on Line 12-13", "voltage: all 25 at 20 kV"). `entries`:
    `locate`'s, or `context`'s values."""
    def val(x):
        return x.get("value") if isinstance(x, dict) else x
    entries = list(entries.values()) if isinstance(entries, dict) else list(entries)
    if len(entries) < 2:
        return []
    out = []
    # Each grouping over the elements it applies to: a unit or a load has a feeder but no voltage of its own here,
    # a transformer two.
    for label, kinds, key, fmt in (("feeder", None, "feeder", lambda v: v),
                                   ("voltage", ("bus", "line"), "kv", lambda v: f"{v:g} kV"),
                                   ("substation", ("bus",), "substation", lambda v: v)):
        members = [e for e in entries if kinds is None or e.get("kind") in kinds]
        n = len(members)
        counts: dict = {}
        for e in members:
            v = val(e.get(key))
            # A transformer has two voltages: it shares no one level with the rest.
            if v is not None and not isinstance(v, (list, tuple, dict)):
                counts[v] = counts.get(v, 0) + 1
        if n < 2 or not counts or max(counts.values()) < 2:
            continue
        # Only what is shared: a value one element has is lumped with the rest ("9 elsewhere").
        ranked = [(v, k) for v, k in sorted(counts.items(), key=lambda x: -x[1]) if k >= 2 or key == "feeder"]
        if key == "substation" and ranked[0][1] * 2 < n:
            continue                    # most of the buses are in substations of their own: nothing shared to say
        if len(ranked) == 1 and ranked[0][1] == n:
            out.append(f"{label}: all {n} {'on' if key == 'feeder' else 'at' if key == 'kv' else 'in'} "
                       f"{fmt(ranked[0][0])}")
            continue
        parts = [f"{k} of {n} {'on' if key == 'feeder' else 'at' if key == 'kv' else 'in'} {fmt(v)}" if i == 0
                 else f"{k} {'on' if key == 'feeder' else 'at' if key == 'kv' else 'in'} {fmt(v)}"
                 for i, (v, k) in enumerate(ranked[:3])]
        rest = n - sum(k for _v, k in ranked[:3])
        off = sum(1 for e in members if val(e.get(key)) is None)
        if rest:
            parts.append(f"{rest} on no feeder" if key == "feeder" and rest == off else f"{rest} elsewhere")
        out.append(f"{label}: " + ", ".join(parts))
    return out
