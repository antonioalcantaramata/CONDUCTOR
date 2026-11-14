"""
grid_context.py — a study's result linked to the CIM layer.

A result names elements ("Bus 14 at 0.912 p.u.", "Line 12-13 overloaded") but says nothing about where they
are; the model then either asks the layer again or guesses ("near the solar farm"). Here the elements a result
names *as a problem* — a violation, an outage that cuts supply, a binding limit, a curtailed unit — are linked
to the layer by their mRIDs (`link`), and the model is told what the layer says each one is: voltage,
substation, feeder and where on it, the units and loads there, what loses supply without a branch — each fact
with its basis.

The facts live in one place, the layer. The result as logged keeps only the links (`cim_links`: the layer's
version and each element's mRID); the facts go in the model's copy (`grid_context`), and whatever reads the
links later resolves them (`resolve`): the turn's record, for the answer checks (`cim_context`), and the answer
card's "Where" column. A link made before a declaration says so: the layer's version no longer matches.

Off when the agent is not given the layer (`config.ontology_enabled`), so an ablation's arms stay comparable.
"""

from __future__ import annotations

import re
from urllib.parse import urlencode

LINKS_KEY = "cim_links"
CONTEXT_KEY = "grid_context"

# The studies whose results can name a problem element. Anything else (settings, the layer itself, time) is
# never linked.
STUDIES = frozenset({
    "run_rsa", "simulate_contingency", "simulate_all_contingencies", "optimize_contingency",
    "optimize_flexibility", "flexibility_over_time", "evaluate_kpis", "kpis_over_time", "scan_rsa_over_time",
    "find_worst_case_timestamp", "scan_scenarios", "run_probabilistic_rsa", "optimize_robust_flexibility",
    "compute_flexibility_envelope", "compute_hosting_capacity", "compute_historical_risk",
    "compute_violation_attribution", "investigate", "test_prevention", "look_ahead", "compare_networks",
})

# A field on the way to a name that says the element is a problem, or the study's answer about it.
_PROBLEM_RE = re.compile(
    r"violat|overload|worst|binding|limiting|congest|unsupplied|disconnect|cut_off|lost|lose|critical|risk|"
    r"attribut|culprit|cause|outage|curtail|activated|max_line|max_trafo|max_vm|min_vm|exceed|bottleneck|"
    r"hosting|weak", re.IGNORECASE)
# Fields that list every element (all voltages, every unit's dispatch): never a problem by being there.
_EVERY_RE = re.compile(r"^all_|percentile|dispatch|bus_voltages|thresholds|series|timestamps|histogram",
                       re.IGNORECASE)
_MAX_CANDIDATES = 200


def _hint(key: str, sibling_type: str | None = None) -> str | None:
    """What kind of element a field names: bus, branch, unit or load — or None."""
    if sibling_type:
        t = sibling_type.lower()
        return ("bus" if t == "bus" else "branch" if t in ("line", "trafo", "transformer", "branch") else
                "load" if t == "load" else "unit" if t in ("gen", "sgen", "storage", "unit") else None)
    k = key.lower()
    if "bus" in k:
        return "bus"
    if any(w in k for w in ("line", "trafo", "branch", "outage")):
        return "branch"
    if "load" in k and "loading" not in k:
        return "load"
    if any(w in k for w in ("unit", "gen", "resource")):
        return "unit"
    return None


def candidates(result) -> list[tuple[str, str | None]]:
    """[(text, kind hint)] — every text in `result` that may name an element, under a field that says it is a
    problem (`_PROBLEM_RE`) and not one that lists every element (`_EVERY_RE`). Dict keys count too ({"Bus_23":
    0.4} under `bus_violation_probability`).

    Taken in turns across fields, each in the result's order: the outages that cut supply and the buses they cut
    off alternate, so one long list does not fill every link before the elements of another field are reached."""
    found: list[tuple[str, str, str | None]] = []           # (field path, text, hint)

    def walk(node, problem: bool, key: str, path: str, sibling_type: str | None = None) -> None:
        if len(found) >= _MAX_CANDIDATES:
            return
        if isinstance(node, dict):
            own_type = node.get("element_type") if isinstance(node.get("element_type"), str) else None
            for k, v in node.items():
                k = str(k)
                if _EVERY_RE.search(k) or k in (LINKS_KEY, CONTEXT_KEY):
                    continue
                here = problem or bool(_PROBLEM_RE.search(k))
                # A key inside a problem field may be a name ({"Bus_23": 0.4} under bus_violation_probability);
                # the field's own name ("violations") never is.
                if problem and isinstance(v, (int, float, dict, list)) and not isinstance(v, bool) and len(k) < 120:
                    found.append((path + "{}", k, _hint(key)))
                walk(v, here, k, f"{path}.{k}", own_type)
        elif isinstance(node, list):
            for v in node:
                walk(v, problem, key, path + "[]", sibling_type)
        elif isinstance(node, str) and problem:
            found.append((path, node, _hint(key, sibling_type)))

    walk(result, False, "", "")
    by_field: dict[str, list[tuple[str, str | None]]] = {}
    for path, text, hint in found:
        by_field.setdefault(path, []).append((text, hint))
    out: list[tuple[str, str | None]] = []
    queues = list(by_field.values())
    while queues and len(out) < _MAX_CANDIDATES:
        for q in queues:
            out.append(q.pop(0))
        queues = [q for q in queues if q]
    return out[:_MAX_CANDIDATES]


# Studies of one moment on the network's own demand: their result's moment is the moment of every MW they cut.
_ONE_MOMENT = frozenset({"run_rsa", "simulate_contingency", "simulate_all_contingencies",
                         "compute_violation_attribution"})
# Arguments that leave the loads' demand as the data give it (limits, the outage, units' scaling or sources).
_DEMAND_AS_DATA = frozenset({"vm_upper_pu", "vm_lower_pu", "max_line_loading_pct", "max_trafo_loading_pct",
                             "element_type", "element_index", "timestamp", "data_source", "network",
                             "sgen_scaling_factor", "assume_unit_sources", "scope", "top_n_sources",
                             "top_n_drivers"})


def study_moment(tool_name: str, result: dict, kwargs: dict | None) -> dict | None:
    """{timestamp, data_source} when the study ran at one moment on the base network's own demand — so the
    layer's MW at that moment is the study's — else None: a load scaling, a proposal, an option applied, a scan
    over time, a sampled or optimised demand, or any argument not known to leave demand as the data give it."""
    kwargs = kwargs or {}
    if tool_name not in _ONE_MOMENT or not isinstance(result, dict):
        return None
    ts = result.get("timestamp")
    if not isinstance(ts, str) or not ts.strip():
        return None
    if kwargs.get("network") not in (None, "", "base") or kwargs.get("option"):
        return None
    scale = kwargs.get("load_scaling_factor", 1.0)
    try:
        if scale is not None and abs(float(scale) - 1.0) > 1e-9:
            return None
    except (TypeError, ValueError):
        return None
    # An empty option ("option": "", as models send it) applies none; a set one was refused above.
    if set(kwargs) - _DEMAND_AS_DATA - {"load_scaling_factor", "option"}:
        return None
    source = kwargs.get("data_source") or result.get("data_source") or "measurements"
    return {"timestamp": ts, "data_source": str(source)}


def _backend_post(endpoint: str, body: dict):
    from .tools import _post
    return _post(endpoint, body)


def _backend_get(endpoint: str):
    from .tools import _get
    return _get(endpoint)


# Arguments that never name an element (when, on what data, which study, how).
_NOT_ELEMENTS = frozenset({"timestamp", "data_source", "network", "option", "study", "metric", "start", "end",
                           "start_timestamp", "end_timestamp", "target_timestamp", "mode", "q_mode", "format",
                           "about", "from_option", "element_type", "proposal", "symptom", "question", "scope"})


def studied(kwargs: dict | None) -> tuple[list[tuple[str, str | None]], list[dict]]:
    """What a study's arguments name: ([(text, kind hint)], [{type, index}]) — the outage, the bus, the units left
    out or held, by name; an element given by type and index ("element_type": "line", "element_index": 10)."""
    texts: list[tuple[str, str | None]] = []
    refs: list[dict] = []
    kwargs = kwargs or {}
    etype, eidx = kwargs.get("element_type"), kwargs.get("element_index")
    if isinstance(etype, str) and isinstance(eidx, int) and not isinstance(eidx, bool):
        refs.append({"type": etype.lower(), "index": eidx})
    for key, value in kwargs.items():
        if key in _NOT_ELEMENTS or key.startswith("_"):
            continue
        items = value if isinstance(value, list) else [value]
        for item in items:
            if isinstance(item, dict):
                item = next((item[k] for k in ("unit", "element", "name", "load", "gen_name") if
                             isinstance(item.get(k), str)), None)
            if isinstance(item, str) and item.strip() and len(item) <= 120:
                texts.append((item.strip(), _hint(key)))
    return texts, refs


def link(tool_name: str, result, kwargs: dict | None = None) -> tuple[dict | None, dict | None]:
    """(the links for the logged result, the facts for the model's copy) — or (None, None): not a study, an
    error, the layer not given to the agent, no element named (in the arguments, or in the result as a problem),
    or the backend cannot say."""
    from .config import ontology_enabled
    if tool_name not in STUDIES or not isinstance(result, dict) or "error" in result or not ontology_enabled():
        return None, None
    found = candidates(result)
    texts, refs = studied(kwargs)
    if not found and not texts and not refs:
        return None, None
    network = (kwargs or {}).get("network") or None
    moment = study_moment(tool_name, result, kwargs)
    try:
        out = _backend_post("/api/network/cim/links", {
            "names": [list(p) for p in found], "network": network, "studied": [list(p) for p in texts],
            "refs": refs, **({"moment": moment} if moment else {})})
    except Exception:  # noqa: BLE001 — context is a help, never a reason for a study to fail
        return None, None
    if not isinstance(out, dict) or "error" in out or not out.get("elements"):
        return None, None
    links = {"version": out["version"], **({"network": network} if network else {}),
             **({"moment": out["moment"]} if out.get("moment") else {}),
             "elements": [{"name": e["name"], "mrid": e["mrid"], **({"role": e["role"]} if e.get("role") else {})}
                          for e in out["elements"]]}
    roles = {e["mrid"]: e.get("role") for e in out["elements"]}
    return links, view(out.get("context") or {}, out.get("where") or [], out["version"], out.get("named"),
                       out.get("moment"), roles=roles, ties=out.get("ties"))


def without_links(result):
    """The model's copy without the mRIDs: it reads the facts (`grid_context`), not identifiers."""
    if isinstance(result, dict) and LINKS_KEY in result:
        return {k: v for k, v in result.items() if k != LINKS_KEY}
    return result


def with_context(result, context: dict | None):
    if context and isinstance(result, dict):
        return {**result, CONTEXT_KEY: context}
    return result


def resolve(links: dict) -> dict | None:
    """What the layer says now about a result's linked elements: {"context", "where", "changed", "version",
    "missing"} — or None when the backend cannot say."""
    if not isinstance(links, dict) or not links.get("elements"):
        return None
    params = {"mrids": ",".join(e["mrid"] for e in links["elements"]), "version": links.get("version") or ""}
    if links.get("network"):
        params["network"] = links["network"]
    moment = links.get("moment") or {}
    if moment.get("timestamp"):
        params.update({k: moment[k] for k in ("timestamp", "data_source", "data_key") if moment.get(k)})
    try:
        out = _backend_get("/api/network/cim/resolve?" + urlencode(params))
    except Exception:  # noqa: BLE001
        return None
    return out if isinstance(out, dict) and "error" not in out else None


def turn_context(results: list[dict]) -> dict:
    """The turn's links resolved, for its record (what the answer checks read): {"elements": {name: facts},
    "changed": [studies whose layer has changed since they ran], "version"}. {} when nothing was linked."""
    elements: dict[str, dict] = {}
    changed: list[str] = []
    where: list[str] = []
    ties: dict = {}
    studied_names: list[str] = []
    version = None
    for entry in results or []:
        res = entry.get("result") if isinstance(entry, dict) else None
        links = res.get(LINKS_KEY) if isinstance(res, dict) else None
        if not links:
            continue
        resolved = resolve(links)
        if resolved is None:
            continue
        version = resolved.get("version")
        if resolved.get("changed") and entry.get("name") not in changed:
            changed.append(entry.get("name"))
        roles = {e["mrid"]: e.get("role") for e in links.get("elements") or []}
        for mrid, facts in (resolved.get("context") or {}).items():
            elements.setdefault(facts["name"], facts)
            if roles.get(mrid) == "studied" and facts["name"] not in studied_names:
                studied_names.append(facts["name"])
        where += [w for w in resolved.get("where") or [] if w not in where]
        for feeder, rows_ in (resolved.get("ties") or {}).items():
            ties.setdefault(feeder, rows_)
    if not elements and not changed:
        return {}
    return {"elements": elements, "changed": changed, "version": version, **({"where": where} if where else {}),
            **({"ties": ties} if ties else {}), **({"studied": studied_names} if studied_names else {})}


# --- What the model reads ---------------------------------------------------------------------------------------

_BASES = "stated: the network says so; inferred: from the topology; named: read from a name; unknown: not said"


def _short(name) -> str:
    return str(name or "").split(" | ")[0]


def _with_basis(fact: dict | None, unit: str = "") -> str:
    if not isinstance(fact, dict) or fact.get("value") in (None, ""):
        return "not stated"
    value = fact["value"]
    shown = f"{value:g}{unit}" if isinstance(value, (int, float)) and not isinstance(value, bool) else str(value)
    return f"{shown.replace('_', ' ')} ({fact.get('basis')})"


def _feeder(f: dict | None, branch: str | None = None) -> str:
    if not f:
        return ""
    if f.get("value") is None:
        return "no feeder" if f.get("why") else ""
    if branch and _short(branch) == f["value"]:
        return f"head of feeder {f['value']} ({f.get('basis')})"
    where = ""
    if f.get("position") and f.get("of"):
        end = (", its far end" if f["position"] == f["of"] and f["of"] > 1 else
               ", next to the substation" if f["position"] == 1 and f["of"] > 1 else "")
        where = f", bus {f['position']} of {f['of']} from the substation{end}"
    return f"feeder {f['value']} ({f.get('basis')}){where}"


def _kv(value) -> str:
    """A voltage as written: "20 kV", or a transformer's sides, "110/20 kV"; "" when not known."""
    if isinstance(value, dict):
        value = value.get("value")
    if isinstance(value, (list, tuple)):
        sides = [f"{v:g}" for v in value if isinstance(v, (int, float)) and not isinstance(v, bool)]
        return f"{'/'.join(sides)} kV" if sides else ""
    return f"{value:g} kV" if isinstance(value, (int, float)) and not isinstance(value, bool) else ""


def _obtained(e: dict) -> str:
    """How an element's P and Q come from the data, in a few words: "P allocated, Q derived"."""
    p, q = e.get("p_kind"), e.get("q_kind")
    return ", ".join(x for x in (f"P {p}" if p else "", f"Q {q}" if q else "") if x)


def _members(items: list[dict], count: int, what: str) -> str:
    if not count:
        return f"no {what}s"

    def one(i: dict) -> str:
        what_it_is = (f"{str(i.get('value')).replace('_', ' ')}, {i.get('basis')}" if i.get("basis") != "unknown"
                      else "not stated")
        obtained = _obtained(i)
        return f"{i['name']} ({what_it_is}" + (f"; {obtained}" if obtained else "") + ")"
    listed = ", ".join(one(i) for i in items)
    more = f" and {count - len(items)} more" if count > len(items) else ""
    return f"{what}s: {listed}{more}"


def describe(facts: dict) -> str:
    """One line for one element (`semantics.context`)."""
    kind = facts.get("kind")
    parts: list[str]
    if kind == "bus":
        sub = facts.get("substation") or {}
        parts = ["bus", _with_basis(facts.get("kv"), " kV"),
                 # A substation of its own (one bus) adds nothing to the bus's name.
                 "" if sub.get("value") in (None, facts.get("name")) else f"substation {_with_basis(sub)}",
                 _feeder(facts.get("feeder")),
                 f"next to {', '.join(facts['next_to'])}" if facts.get("next_to") else "",
                 _members(facts.get("units") or [], facts.get("unit_count", 0), "unit"),
                 _members(facts.get("loads") or [], facts.get("load_count", 0), "load")
                 + (f" ({facts['load_mw']:g} MW at the study's moment)" if facts.get("load_mw") is not None else "")]
    elif kind in ("line", "transformer"):
        ends = facts.get("ends") or []
        parts = [f"{kind} {' → '.join(ends)}" if ends else kind, _kv(facts.get("kv")),
                 _feeder(facts.get("feeder"), facts.get("name"))]
        cut = facts.get("without_it")
        if cut:
            if cut.get("buses_cut_off"):
                loads = ", ".join(cut.get("loads_cut_off") or [])
                more = cut.get("load_count_cut_off", 0) - len(cut.get("loads_cut_off") or [])
                lost_mw = cut.get("load_cut_off_mw")
                parts.append(f"without it {cut['buses_cut_off']} bus(es) and {cut.get('load_count_cut_off', 0)} "
                             f"load(s) lose supply" + (f" ({loads}{f' and {more} more' if more > 0 else ''}"
                                                       + (f"; {lost_mw:g} MW at the study's moment" if
                                                          lost_mw is not None else "") + ")" if loads else "")
                             + " — topology only" + ("" if lost_mw is not None else "; the MW lost is the study's"))
            else:
                parts.append("nothing loses supply without it (another route exists) — topology only")
    elif kind == "unit":
        parts = ["unit", _with_basis(facts.get("source")), f"at {facts.get('bus')}", _feeder(facts.get("feeder")),
                 _obtained(facts)]
    else:
        parts = ["load", f"category {_with_basis(facts.get('category'))}", f"at {facts.get('bus')}",
                 _feeder(facts.get("feeder")), _obtained(facts)]
    return " · ".join(p for p in parts if p)


def tie_lines(ties: dict | None) -> dict:
    """{feeder: its normally open points, in words} — where supply could come from if part of it is lost."""
    out = {}
    for feeder, rows in (ties or {}).items():
        out[feeder] = "; ".join(
            f"{r['switch']}" + (f" on {r['on']}" if r.get("on") else "") + (f" at {r['at']}" if r.get("at") else "")
            + (f": a tie to feeder {r['to_feeder']}" if r.get("to_feeder") else ": closes a loop within the feeder")
            + ("" if r.get("open_now", True) else " (closed now)") for r in rows)
    return out


def view(contexts: dict, where: list[str], version: str, named: int | None = None,
         moment: dict | None = None, roles: dict | None = None, ties: dict | None = None) -> dict | None:
    """What the model's copy carries: one line per linked element, and where all the elements the result names
    (`named`) sit together."""
    if not contexts:
        return None
    more = named is not None and named > len(contexts)
    # Why an element is on no feeder, once (on a meshed network it is the same for every element).
    why = sorted({(f.get("feeder") or {}).get("why") for f in contexts.values()} - {None, ""})
    return {"from": f"the CIM layer (version {version}); each fact with its basis — {_BASES}",
            "elements": {_short(f["name"]): describe(f) for f in contexts.values()},
            **({"where": where} if where else {}),
            **({"covers": f"the {len(contexts)} elements above in full, of the {named} the result names; `where` "
                          f"places all {named}"} if more else {}),
            **({"no_feeder": "; ".join(why)} if why else {}),
            **({"studied": [_short(f["name"]) for m, f in contexts.items() if (roles or {}).get(m) == "studied"]}
               if any(r == "studied" for r in (roles or {}).values()) else {}),
            **({"normally_open_points": tie_lines(ties)} if ties else {}),
            **({"demand": f"MW are the loads' demand at {moment['timestamp']} ({moment.get('data_source')}), the "
                          "moment the study ran, on the network's own data"} if moment else {}),
            "use": "Say where the problem is from these facts — they are the CIM layer's: call them that — with "
                   "their basis, rather than guessing. `studied`: the element the study is about (from its "
                   "arguments). A value is called measured only where P or Q says so (allocated: a share of a bus's "
                   "measurement; derived: from the power factor). A normally open point may restore supply; whether "
                   "it does is a study (simulate_contingency checks each with a power flow). An element not listed "
                   "is placed only by `where`; for its own facts use describe_network."}


# --- What the answer card shows ---------------------------------------------------------------------------------

def rows(resolved: dict, roles: dict | None = None) -> list[dict]:
    """The "Where" table of a result card: one row per linked element, from its links resolved now. `roles`:
    {mrid: "studied"} from the links — the element the study is about is marked."""
    out = []
    for mrid, f in (resolved or {}).get("context", {}).items():
        kind = f.get("kind")
        feeder = f.get("feeder") or {}
        if feeder.get("value"):
            where = f"feeder {feeder['value']}"
            if kind in ("line", "transformer") and _short(f.get("name")) == feeder["value"]:
                where += " (its head)"
            elif feeder.get("position") and feeder.get("of"):
                where += f" · bus {feeder['position']} of {feeder['of']}"
                if feeder["position"] == feeder["of"] and feeder["of"] > 1:
                    where += " (far end)"
        else:
            where = "no feeder" + (f" ({feeder['why']})" if feeder.get("why") else "")
        if kind == "bus":
            there = "; ".join(p for p in (
                _members(f.get("units") or [], f.get("unit_count", 0), "unit") if f.get("unit_count") else "",
                _members(f.get("loads") or [], f.get("load_count", 0), "load") if f.get("load_count") else "",
                f"{f['load_mw']:g} MW" if f.get("load_mw") is not None else "")
                if p) or "nothing connected"
        elif kind in ("line", "transformer"):
            cut = f.get("without_it") or {}
            there = (f"without it {cut['buses_cut_off']} bus(es), {cut.get('load_count_cut_off', 0)} load(s) "
                     "lose supply" + (f" ({cut['load_cut_off_mw']:g} MW)" if cut.get("load_cut_off_mw") is not None
                                      else "")
                     if cut.get("buses_cut_off") else "another route exists" if cut else "")
        elif kind == "unit":
            there = " · ".join(x for x in (_with_basis(f.get("source")), f"at {f.get('bus')}", _obtained(f)) if x)
        else:
            there = " · ".join(x for x in (f"category {_with_basis(f.get('category'))}", f"at {f.get('bus')}",
                                           _obtained(f)) if x)
        studied_here = (roles or {}).get(mrid) == "studied"
        out.append({"Element": _short(f.get("name")), "Kind": f"{kind} (studied)" if studied_here else kind,
                    "Voltage": _kv(f.get("kv")) or "—",
                    "Where": where, "There": there})
    return out
