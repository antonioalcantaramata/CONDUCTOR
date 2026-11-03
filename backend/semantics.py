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


def unit_source(net, table: str, idx, declared: dict | None = None, profile: dict | None = None) -> dict:
    """The energy source of one unit, as a fact (see the module's docstring).
    `declared`: the operator's statements; `profile`: the network profile's
    (`unit_sources`), both keyed by the unit's name as loaded."""
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


def substations(net) -> tuple[dict, str]:
    """{bus index: substation name}, and the basis. Stated when the network
    names them (a 'substation' column, as a CIM import writes); otherwise the
    buses joined by transformers or bus couplers (bus–bus switches), named after
    the highest-voltage bus among them — a single bus is a substation of its own."""
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
            groups.setdefault(str(sid) if pd.notna(sid) else f"node:{int(i)}", []).append(int(i))
        out = {}
        for members in groups.values():
            head = max(members, key=lambda b: (float(net.bus.at[b, "vn_kv"]), -b))
            out.update({b: f"Substation {bus_display_name(net, head)}" for b in members})
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
            where = {"kind": "coupler", "between": [buses[bus]["name"], buses.get(other, {}).get("name", other)]}
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
                     synthetic: bool = True, load_pf: float = 0.95, load_pf_source: str = "default") -> dict:
    """Where each load's and unit's P and Q at the present operating point come
    from — the data, a share of them, an assumption, the power flow — worked
    out from the moment's data rows (`rows`: substation_name, consumption,
    production, and consumption_q / production_q when measured) the way the
    assignment uses them. Pure: the values themselves are not touched."""
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
    return mrid(network_id, table, int(idx))


def _mrid_basis(net, table: str) -> str:
    return STATED if "origin_id" in net[table].columns else INFERRED


def _kv(net, bus: int, per_unit: bool) -> dict:
    proposed = "proposed" in net.bus.columns and bool(net.bus.at[bus, "proposed"] is True)
    if per_unit and not proposed:
        return fact(None, UNKNOWN, "a converted per-unit case: 1 kV stands for the bus's real voltage")
    return fact(float(net.bus.at[bus, "vn_kv"]), STATED, "the bus's nominal voltage in the network")


def build(net, declared: dict | None = None, profile: dict | None = None,
          declared_loads: dict | None = None, profile_loads: dict | None = None,
          normal_open: dict | None = None, provenance: dict | None = None, network_id: str = "local") -> dict:
    """What every element of `net` is, with the basis of each fact (compact:
    one entry per element, for the agent and for checking answers).
    `declared`/`profile`: units' sources; `declared_loads`/`profile_loads`:
    loads' categories — each keyed by the element's name as loaded.
    `normal_open`: {switch index: open as loaded} (the switches' normal
    state); `provenance`: `value_provenance`'s result, where each value of the
    present operating point comes from. `network_id`: the network's identity,
    for derived mRIDs (the backend passes the fingerprint as loaded)."""
    per_unit = per_unit_case(net)
    subs, sub_basis = substations(net)
    sub_why = (("the model's Substation for each bus (named after its highest-voltage bus; a node with none, "
                 "such as a boundary node, is a substation of its own)"
                if "Substation_id" in net.bus.columns and "substation" not in net.bus.columns else "named in the network")
               if sub_basis == STATED else
               "buses joined by transformers or bus couplers; a single bus is a substation of its own")
    buses = []
    for i in net.bus.index:
        b = int(i)
        buses.append({"name": bus_display_name(net, b), "index": b, "cim_class": CIM_CLASS["bus"],
                      "kv": _kv(net, b, per_unit), "substation": fact(subs.get(b), sub_basis, sub_why),
                      "in_service": bool(net.bus.at[b, "in_service"])})
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
            branches.append(entry)
    units = []
    for table, i in _unit_rows(net):
        source = unit_source(net, table, i, declared, profile)
        bus = int(net[table].at[i, "bus"])
        inverter = any(re.search(r"fuel ?cell", str(net[table].at[i, c]), re.IGNORECASE)
                       for c in ("type", "name") if c in net[table].columns)
        equipment, unit_class = cim_unit(table, source["value"], inverter)
        control = _control(net, table, i)
        units.append({"name": gm.unit_name(net, table, i), "element_type": table, "index": int(i),
                      "cim_class": equipment, "cim_unit": unit_class,
                      "cim_control": {"class": "RegulatingControl", "mode": _CIM_MODE[control["value"]]},
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
        shunts.append({"name": name or f"Shunt {int(i)}", "index": int(i), "cim_class": "LinearShuntCompensator",
                       "kind": (fact(kind, STATED, "the sign of its reactive power in the network (negative: it "
                                                   "injects)") if q else
                                fact("unknown", UNKNOWN, "no reactive power given")),
                       "q_mvar_at_1pu": round(q * float(step), 4), "bus": bus_display_name(net, bus),
                       "bus_index": bus, "in_service": bool(net.shunt.at[i, "in_service"])})
    loads = []
    for i in net.load.index:
        bus = int(net.load.at[i, "bus"])
        loads.append({"name": (clean_label(net.load.at[i, "name"]) if "name" in net.load.columns else "")
                      or f"Load {int(i)}", "index": int(i), "cim_class": CIM_CLASS["load"],
                      "category": load_category(net, i, declared_loads, profile_loads),
                      "bus": bus_display_name(net, bus), "bus_index": bus,
                      "p_mw": round(float(net.load.at[i, "p_mw"]), 3),
                      "in_service": bool(net.load.at[i, "in_service"])})
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
    return {"per_unit_case": per_unit, "voltage_levels_kv": levels,
            "substations": {"basis": sub_basis, "from": sub_why, "count": len(set(subs.values()))},
            "units_by_source": by_source, "loads_by_category": by_category,
            "buses": buses, "branches": branches, "units": units, "loads": loads, "external_grids": external,
            "switches": switches, "shunts": shunts,
            **({"unresolved_branches": unresolved} if unresolved else {}),
            "containers": _containers(buses, units, loads, switches, sub_basis, sub_why, branches, network_id),
            "mrid_basis": _mrid_basis(net, "bus"),
            **({"provenance": provenance.get("context")} if provenance else {})}


def _containers(buses, units, loads, switches, sub_basis, sub_why, branches=(), network_id="local") -> dict:
    """CIM's equipment containers: each Substation with its VoltageLevels, and
    each Feeder with what it supplies and the switches that tie it to others."""
    subs: dict[str, dict] = {}
    for b in buses:
        s = subs.setdefault(b["substation"]["value"], {"levels": {}, "buses": []})
        s["buses"].append(b["name"])
        s["levels"].setdefault(b["kv"]["value"], []).append(b["name"])
    first_bus = {}
    for b in buses:
        first_bus.setdefault(b["substation"]["value"], b["index"])
    substation_list = [{"name": n, "cim_class": "Substation", "basis": sub_basis, "from": sub_why,
                        "mrid": mrid(network_id, "substation", first_bus[n]),
                        "voltage_levels": [{"cim_class": "VoltageLevel", "kv": kv, "buses": names,
                                            "mrid": mrid(network_id, "voltage_level", f"{first_bus[n]}:{kv}")}
                                           for kv, names in sorted(s["levels"].items(), key=lambda x: -(x[0] or 0))]}
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
         "data")
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
            out["note"] = ("No feeders: " + next((b["feeder"]["from"] for b in buses if b.get("feeder")),
                                                 "no external grid") + ".")
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
                        + (" These are synthetic data, not field measurements." if ctx.get("synthetic") else "")}
    if about == "voltage_levels":
        if semantics["per_unit_case"]:
            return {"about": "voltage_levels", "levels_kv": [], "basis": UNKNOWN,
                    "note": "A converted per-unit case: every bus is at 1 kV, a placeholder for its real voltage. "
                            "The network's voltage levels are unknown."}
        levels = {}
        for b in buses:
            levels.setdefault(b["kv"]["value"], []).append(b["name"])
        return {"about": "voltage_levels", "basis": STATED,
                "levels": [{"kv": kv, "buses": len(names), "e.g.": names[:5]}
                           for kv, names in sorted(levels.items(), key=lambda kv: -(kv[0] or 0))]}
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
    by_source = {k: len(v) for k, v in semantics["units_by_source"].items()}
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
            **({"note": declare_hint} if unknown_units else {})}


def _value_kinds(e: dict) -> dict:
    """An element's values as the checks need them: the kind of P and Q, and where Q comes from."""
    v = e.get("values")
    if not v:
        return {}
    return {"p_kind": v["p"]["kind"], "p_from": v["p"]["from"], "q_kind": v["q"]["kind"], "q_from": v["q"]["from"]}


def compact(semantics: dict) -> dict:
    """The facts an answer is checked against, small enough to keep in each
    turn's record: names, kinds, voltages, ends, sources — with their bases."""
    return {
        "per_unit_case": semantics["per_unit_case"],
        "voltage_levels_kv": semantics["voltage_levels_kv"],
        "buses": {b["name"]: {"kv": b["kv"]["value"], "kv_basis": b["kv"]["basis"],
                              "substation": b["substation"]["value"]} for b in semantics["buses"]},
        "branches": {b["name"]: {"type": b["element_type"], "ends": b["ends"]} for b in semantics["branches"]},
        "units": {u["name"]: {"source": u["source"]["value"], "basis": u["source"]["basis"], "bus": u["bus"],
                              "type": u["element_type"], **_value_kinds(u)} for u in semantics["units"]},
        "loads": {d["name"]: {"category": d["category"]["value"], "basis": d["category"]["basis"], "bus": d["bus"],
                              **_value_kinds(d)}
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


def _ties(net, names: dict, lost: set[int]) -> list[str]:
    """Normally open switches whose line joins the cut-off part to a supplied
    one: what could restore supply (not checked against flows)."""
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
            out.append(f"{what} (switch {label or int(i)}, open at {bus_display_name(net, int(s.bus))})")
    return out


def around(net, semantics: dict, etype: str, idx) -> dict:
    """An element and everything it is connected to, in one answer: its
    substation (levels, buses, units and loads there, branches leaving and
    where they lead, the external grid), its supply route from the external
    grid and its feeder (inferred from the topology), and — for a bus, a
    branch or a substation — what would be cut off without it, with the open
    ties that could restore it. `etype`: bus, line, trafo, gen, sgen, storage,
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
        def hops(b):
            try:
                return min(nx.shortest_path_length(graph, s, b) for s in sources if s in graph)
            except (nx.NetworkXNoPath, nx.NodeNotFound, ValueError):
                return -1
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
    return out
