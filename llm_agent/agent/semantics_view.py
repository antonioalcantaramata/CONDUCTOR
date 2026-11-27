"""
semantics_view.py — what the network's elements are, for the System view.

The semantic layer (backend/semantics.py) as the operator sees it: a few
headline facts, a picture of where the units or the loads sit (voltage level ›
substation › element, coloured by energy source or demand category) and a
table per kind of element with the basis of each fact — stated by the network,
read from its name, inferred by a rule, declared by the operator, or unknown.
Set source / Set category write the same declarations the agent's
`declare_network_facts` does.

Pure functions of the backend's payload, so they are tested without Streamlit.
"""

from __future__ import annotations

from .renderers import _C

# The app's validated categorical hues. Related values share one: the name and
# the legend carry the value, so a shared hue never stands alone.
SOURCE_COLOURS = {
    "wind": _C.TEAL, "solar": _C.OLIVE,
    "hydro": _C.PRIMARY, "bioenergy": _C.PRIMARY,
    "thermal": _C.SIENNA, "nuclear": _C.SIENNA, "conventional": _C.SIENNA,
    "storage": _C.PURPLE,
    "condenser": _C.MAGENTA, "other": _C.MAGENTA,
    "unknown": _C.NEUTRAL,
}
CATEGORY_COLOURS = {
    "residential": _C.PRIMARY,
    "commercial": _C.PURPLE, "commercial_industrial": _C.PURPLE,
    "industrial": _C.SIENNA,
    "rural": _C.OLIVE, "agricultural": _C.OLIVE,
    "data_centre": _C.TEAL, "ev_charging": _C.TEAL, "heat_pump": _C.TEAL,
    "network": _C.MAGENTA, "other": _C.MAGENTA,
    "unknown": _C.NEUTRAL,
}
SETTABLE = ("wind", "solar", "hydro", "bioenergy", "thermal", "nuclear", "storage", "conventional", "other")
SETTABLE_CATEGORIES = ("residential", "commercial", "industrial", "commercial_industrial", "rural", "agricultural",
                       "ev_charging", "data_centre", "heat_pump", "network", "other")
BASIS_LABELS = {"stated": "Stated", "named": "From its name", "inferred": "Inferred", "unknown": "Unknown"}
UNIT_COLUMNS = ["Unit", "Bus", "Source", "Basis", "From"]
LOAD_COLUMNS = ["Load", "Bus", "Category", "Basis", "From", "Demand (MW)"]
_FADE = 0.45            # a weaker basis than stated: the hue, lighter

# What a kind of element is described by: its fact, its colours.
KINDS = {
    "units": {"fact": "source", "colours": SOURCE_COLOURS, "legend": "Energy source"},
    "loads": {"fact": "category", "colours": CATEGORY_COLOURS, "legend": "Demand category"},
    "model": {"fact": None, "colours": {}, "legend": "The CIM model's classes"},
}


def label(value: str) -> str:
    """A fact's value as the operator reads it ("data_centre" → "data centre")."""
    return str(value).replace("_", " ")


def _faded(hex_colour: str, share: float) -> str:
    """The colour mixed with white (share 0 = white, 1 = the colour)."""
    r, g, b = (int(hex_colour[i:i + 2], 16) for i in (1, 3, 5))
    return "#" + "".join(f"{round(255 - (255 - c) * share):02X}" for c in (r, g, b))


def colour(value: str, basis: str, colours: dict | None = None) -> str:
    base = (colours or SOURCE_COLOURS).get(value, _C.NEUTRAL)
    return base if basis == "stated" or value == "unknown" else _faded(base, _FADE)


def _declared(f: dict) -> bool:
    return f["from"] == "declared by the operator"


def basis_of(unit: dict, fact: str = "source") -> str:
    """The fact's basis, with the operator's declaration told apart."""
    return "declared" if _declared(unit[fact]) else unit[fact]["basis"]


def _basis_cell(f: dict) -> str:
    return "Declared by you" if _declared(f) else BASIS_LABELS.get(f["basis"], f["basis"])


def _from_cell(f: dict) -> str:
    # The reason is the same for every unknown: the column says it once (the caption).
    return "" if _declared(f) or f["basis"] == "unknown" else f["from"]


def unit_rows(semantics: dict) -> list[dict]:
    """One row per unit: what drives it and how we know. Unknown first —
    those are what the operator can add."""
    rows = []
    for u in semantics.get("units") or []:
        src = u["source"]
        rows.append({"unit": u["name"], "declared": _declared(src), "value": src["value"], "basis": src["basis"],
                     # Storage is stated as such by its table, and a declaration names a generating unit.
                     "editable": u.get("element_type") != "storage",
                     "cells": [u["name"], u.get("bus") or "", label(src["value"]), _basis_cell(src), _from_cell(src)]})
    rows.sort(key=lambda r: (r["value"] != "unknown", r["value"], r["unit"].casefold()))
    return rows


def load_rows(semantics: dict) -> list[dict]:
    """One row per load: what kind of demand it is and how we know."""
    rows = []
    for d in semantics.get("loads") or []:
        cat = d["category"]
        rows.append({"unit": d["name"], "declared": _declared(cat), "value": cat["value"], "basis": cat["basis"],
                     "cells": [d["name"], d.get("bus") or "", label(cat["value"]), _basis_cell(cat), _from_cell(cat),
                               f"{d['p_mw']:g}"]})
    rows.sort(key=lambda r: (r["value"] != "unknown", r["value"], r["unit"].casefold()))
    return rows


def line_table(semantics: dict) -> list[dict]:
    """Lines, read-only: ends, voltage, rating (with its basis)."""
    out = []
    for b in semantics.get("branches") or []:
        if b["element_type"] != "line":
            continue
        r = b["rating_mva"]
        out.append({"Line": b["name"], "From": b["ends"][0], "To": b["ends"][1],
                    "kV": b["kv"], "Rating (MVA)": r["value"],
                    "Rating basis": BASIS_LABELS.get(r["basis"], r["basis"]),
                    **({"Limits (model)": limits_text(b["limits"])} if b.get("limits") else {}),
                    "In service": b["in_service"], "CIM class": b["cim_class"]})
    return out


def limits_text(limits: dict) -> str:
    """A branch's limits as the model gives them, on one line: '1000 A permanent · 1150 A 15 min · 1400 A 1 min'
    (the first terminal's; the studies check the permanent one)."""
    rows = limits.get("value") or []
    first = min((r.get("terminal") or 0 for r in rows), default=0)
    parts = []
    for r in rows:
        if (r.get("terminal") or 0) != first or r.get("value") is None or r.get("limit_type") == "patlt":
            continue
        unit = "A" if r.get("kind") in (None, "CurrentLimit") else ("MVA" if "Apparent" in str(r.get("kind")) else
                                                                    "MW" if "Active" in str(r.get("kind")) else "kV")
        held = "permanent" if r.get("permanent") else _duration(r.get("duration_s"))
        parts.append(f"{r['value']:g} {unit} {held}")
    return " · ".join(dict.fromkeys(parts))


def _duration(seconds) -> str:
    if not seconds:
        return "temporary"
    s = float(seconds)
    return f"{s / 3600:g} h" if s >= 3600 else f"{s / 60:g} min" if s >= 60 else f"{s:g} s"


def trafo_table(semantics: dict) -> list[dict]:
    """Transformers, read-only: HV and LV ends with their voltages, rating."""
    out = []
    for b in semantics.get("branches") or []:
        if b["element_type"] != "trafo":
            continue
        hv, lv = (b["kv"] or [None, None])[:2]
        taps = (b.get("tap_changers") or {}).get("value") or []
        out.append({"Transformer": b["name"], "HV bus": b["ends"][0], "LV bus": b["ends"][1],
                    "HV kV": hv, "LV kV": lv, "Rating (MVA)": b["rating_mva"]["value"],
                    **({"Tap (model)": "; ".join(
                        f"step {t['now']:g} of {t['steps'][0]:g}–{t['steps'][1]:g}"
                        + (" · control on (held fixed in studies)" if t.get("control_enabled") else "")
                        for t in taps if t.get("now") is not None and None not in (t.get("steps") or [None]))}
                       if taps else {}),
                    "In service": b["in_service"], "CIM class": b["cim_class"]})
    return out


def bus_table(semantics: dict) -> list[dict]:
    """Buses, read-only: voltage and substation, each with its basis; the model's voltage level, busbar and bus
    name when a CIM model states them."""
    buses = semantics.get("buses") or []
    model = {k: any(b.get(k) for b in buses) for k in ("voltage_level", "busbar", "name_in_model")}

    def named(b, key):
        return ((b.get(key) or {}).get("value") or {}).get("name") or "—"
    return [{"Bus": b["name"],
             **({"Name in model": named(b, "name_in_model")} if model["name_in_model"] else {}),
             "kV": b["kv"]["value"], "kV basis": BASIS_LABELS.get(b["kv"]["basis"]),
             "Substation": b["substation"]["value"],
             "Substation basis": BASIS_LABELS.get(b["substation"]["basis"]),
             **({"Voltage level": named(b, "voltage_level")} if model["voltage_level"] else {}),
             **({"Busbar": named(b, "busbar")} if model["busbar"] else {}),
             "Feeder": ((b.get("feeder") or {}).get("value") or "—").split(" | ")[0],
             "In service": b["in_service"], "CIM class": b["cim_class"]} for b in buses]


# The CIM model the network came from: how much of it the layer describes.
STATUS_LABELS = {"described": "Described by the layer", "partly": "Partly described",
                 "read_into": "Read into other entries", "solved_state": "Source tool's solution (checked)",
                 "model_only": "In the model only"}
STATUS_COLOURS = {"described": _C.OK, "partly": _C.TEAL, "read_into": _C.PRIMARY, "solved_state": _C.PURPLE,
                  "model_only": _C.NEUTRAL}


def model_headline(semantics: dict) -> tuple[str, str, str] | None:
    """(label, value, how we know) for the CIM model, above the picture; None for a network not loaded as CIM."""
    cim = semantics.get("cim_model")
    if not cim:
        return None
    rows = (cim.get("coverage") or {}).get("classes") or []
    described = sum(1 for r in rows if r["status"] in ("described", "read_into"))
    return ("CIM model", f"{cim.get('objects', 0):,} objects",
            f"CGMES {cim.get('version')}: {described} of {len(rows)} classes in the layer, all kept")


def model_hierarchy(semantics: dict) -> dict:
    """Sunburst arrays for the CIM model: status › class, one sector per class (its objects in the hover)."""
    rows = ((semantics.get("cim_model") or {}).get("coverage") or {}).get("classes") or []
    ids, labels, parents, colours, hovers = [], [], [], [], []
    for status in STATUS_LABELS:
        mine = [r for r in rows if r["status"] == status]
        if not mine:
            continue
        ids.append(status), labels.append(STATUS_LABELS[status]), parents.append(""), colours.append(
            STATUS_COLOURS[status])
        hovers.append(f"<b>{STATUS_LABELS[status]}</b><br>{len(mine)} classes, "
                      f"{sum(r['in_model'] for r in mine):,} objects")
        for r in mine:
            ids.append(f"{status}/{r['class']}"), labels.append(r["class"]), parents.append(status)
            colours.append(_faded(STATUS_COLOURS[status], 0.55))
            hovers.append(f"<b>{r['class']}</b><br>{r['in_model']:,} in the model · {r['in_computation']:,} in the "
                          f"studies · {r['in_layer']:,} in the layer"
                          + (f"<br>read into: {r['read_into']}" if r.get("read_into") else ""))
    return {"ids": ids, "labels": labels, "parents": parents, "colours": colours, "hovers": hovers}


def model_legend(semantics: dict) -> list[tuple[str, str]]:
    rows = ((semantics.get("cim_model") or {}).get("coverage") or {}).get("classes") or []
    present = {r["status"] for r in rows}
    return [(STATUS_LABELS[s], c) for s, c in STATUS_COLOURS.items() if s in present]


def model_table(semantics: dict) -> list[dict]:
    """The loss report: each class of the CIM model, how many objects it has, reach the studies, and the layer."""
    rows = ((semantics.get("cim_model") or {}).get("coverage") or {}).get("classes") or []
    order = list(STATUS_LABELS)
    return [{"Class": r["class"], "In the model": r["in_model"], "In the studies": r["in_computation"],
             "In the layer": r["in_layer"], "Status": STATUS_LABELS.get(r["status"], r["status"]),
             "Profiles": ", ".join(r.get("profiles") or [])}
            for r in sorted(rows, key=lambda r: (order.index(r["status"]) if r["status"] in order else 9,
                                                 r["class"]))]


def other_table(semantics: dict) -> list[dict]:
    """Equipment the layer has no kind of its own for, and the model's equipment the studies do not see."""
    return [{"Name": o["name"], "CIM class": o["cim_class"], "What": o["what"], "Buses": ", ".join(o["buses"]),
             "In the studies": "yes" if o["in_computation"] else "no — not converted"}
            for o in semantics.get("other_equipment") or []]


def headline(semantics: dict) -> list[tuple[str, str, str]]:
    """(label, value, how we know) for the facts above the picture."""
    levels = semantics.get("voltage_levels_kv") or []
    subs = semantics.get("substations") or {}
    units = semantics.get("units") or []
    loads = semantics.get("loads") or []
    known = [u for u in units if u["source"]["basis"] != "unknown"]
    known_loads = [d for d in loads if d["category"]["basis"] != "unknown"]
    ext = semantics.get("external_grids") or []
    return [
        ("Voltage levels",
         "not stated" if semantics.get("per_unit_case") or not levels else " · ".join(f"{v:g} kV" for v in levels),
         "the case gives per-unit placeholders" if semantics.get("per_unit_case") else "stated by the network"),
        ("Substations", str(subs.get("count", "—")),
         "inferred: buses joined by transformers or couplers" if subs.get("basis") == "inferred"
         else BASIS_LABELS.get(subs.get("basis"), "").lower() + " by the network"),
        ("Units with a known source", f"{len(known)} of {len(units)}",
         ", ".join(f"{len(v)} {label(k)}" for k, v in sorted((semantics.get("units_by_source") or {}).items()))
         or "the network does not say what drives any unit"),
        ("Loads with a known category", f"{len(known_loads)} of {len(loads)}",
         ", ".join(f"{len(v)} {label(k)}" for k, v in sorted((semantics.get("loads_by_category") or {}).items()))
         or "no loads"),
        ("External grid", ", ".join(e["bus"] for e in ext) or "none",
         ((f"its power comes in over {n} branch{'es' if n != 1 else ''}"
           if (n := sum(len(e["branches"]) for e in ext)) else "no line or transformer leaves its substation")
          if ext else "")),
    ]


def hierarchy(semantics: dict, kind: str = "units") -> dict:
    """Plotly sunburst arrays: voltage level › substation › unit (or load).
    An element's sector carries its value's colour, lighter when the value is
    only named or inferred, grey when unknown."""
    spec = KINDS[kind]
    buses = {b["index"]: b for b in semantics.get("buses") or []}
    ids, labels, parents, colours, hovers = [], [], [], [], []
    seen = set()

    def add(node_id, text, parent, col, hover):
        if node_id in seen:
            return
        seen.add(node_id)
        ids.append(node_id), labels.append(text), parents.append(parent), colours.append(col)
        hovers.append(hover)

    for e in semantics.get(kind) or []:
        bus = buses.get(e.get("bus_index")) or {}
        kv = (bus.get("kv") or {})
        level = f"{kv['value']:g} kV" if kv.get("basis") == "stated" and kv.get("value") is not None \
            else "voltage not stated"
        sub = (bus.get("substation") or {}).get("value") or e.get("bus") or "?"
        add(level, level, "", "#F4F6FA", f"{level}")
        add(f"{level}/{sub}", sub, level, "#E6EAF2",
            f"{sub}<br>substation: {(bus.get('substation') or {}).get('basis', 'unknown')}")
        f = e[spec["fact"]]
        add(f"{level}/{sub}/{e.get('element_type', kind)}{e['index']}", e["name"], f"{level}/{sub}",
            colour(f["value"], f["basis"], spec["colours"]),
            f"<b>{e['name']}</b> at {e.get('bus')}<br>{spec['fact']}: {label(f['value'])} "
            f"({'declared by you' if _declared(f) else f['basis']})<br>{f['from']}")
    return {"ids": ids, "labels": labels, "parents": parents, "colours": colours, "hovers": hovers}


def figure(semantics: dict, kind: str = "units"):
    import plotly.graph_objects as go
    h = model_hierarchy(semantics) if kind == "model" else hierarchy(semantics, kind)
    fig = go.Figure(go.Sunburst(
        ids=h["ids"], labels=h["labels"], parents=h["parents"],
        marker={"colors": h["colours"], "line": {"color": "white", "width": 2}},
        hovertext=h["hovers"], hoverinfo="text", insidetextorientation="radial", maxdepth=3,
        textfont={"color": _C.INK, "size": 12}))
    fig.update_layout(margin={"t": 16, "b": 16, "l": 16, "r": 16}, height=460, paper_bgcolor="white",
                      font={"family": "Inter, system-ui, sans-serif"},
                      hoverlabel={"bgcolor": "white", "bordercolor": "#D5DAE3"})
    return fig


def legend(semantics: dict, kind: str = "units") -> list[tuple[str, str]]:
    """(value, colour) for the values present, in the colour map's order."""
    if kind == "model":
        return model_legend(semantics)
    spec = KINDS[kind]
    present = {e[spec["fact"]]["value"] for e in semantics.get(kind) or []}
    return [(label(v), c) for v, c in spec["colours"].items() if v in present]


def around_choices(semantics: dict) -> list[tuple[str, str]]:
    """(label shown, element sent) for every element one can look around:
    substations first, then buses, lines, transformers, units, loads. The
    element is sent by its mRID, so a bus named "1" is never taken for the bus
    at index 1; the label is the short name, made unique where two share it."""
    import re
    uuid_ = re.compile(r"_?[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")

    def sent(e: dict) -> str:
        """The mRID when the backend can look it up (UUID-shaped), else the name."""
        m = str(e.get("mrid") or "")
        return m if uuid_.fullmatch(m) else e["name"]

    bus_names = {b["name"] for b in semantics.get("buses") or []}
    out = [(f"Substation · {s['name']}", sent(s))
           for s in sorted((semantics.get("containers") or {}).get("substations") or [], key=lambda s: s["name"])
           if s["name"] not in bus_names]                      # a one-bus substation is its bus
    out += [(f"Bus · {b['name']}", sent(b)) for b in semantics.get("buses") or []]
    for kind, word in (("line", "Line"), ("trafo", "Transformer")):
        out += [(f"{word} · {_short(b['name'])}", sent(b)) for b in semantics.get("branches") or []
                if b["element_type"] == kind]
    out += [(f"Unit · {u['name']}", sent(u)) for u in semantics.get("units") or []]
    out += [(f"Load · {d['name']}", sent(d)) for d in semantics.get("loads") or []]
    seen: dict[str, int] = {}
    unique = []
    for label_, value in out:
        seen[label_] = seen.get(label_, 0) + 1
        unique.append((label_ if seen[label_] == 1 else f"{label_} ({seen[label_]})", value))
    return unique


def around_cards(result: dict) -> list[tuple[str, list[str]]]:
    """The connected view as (title, lines) cards: where it sits, how it is
    supplied, and what depends on it."""
    stations = result.get("substations_at_its_ends") or ([result["substation"]] if result.get("substation") else [])
    where = []
    for s in stations:
        levels = " / ".join(f"{v:g} kV" for v in s["voltage_levels_kv"]) or "voltage not stated"
        where.append(f"**{s['name']}** · {levels}")
        if s["units"]:
            where.append("Units: " + ", ".join(
                u if isinstance(u, str) else f"{u['name']} ({label(u['source'])})" for u in s["units"]))
        if s["loads"]:
            where.append("Loads: " + ", ".join(
                d if isinstance(d, str) else f"{d['name']} ({d['p_mw']:g} MW)" for d in s["loads"]))
        if s["branches_leaving"]:
            where.append("Leaving: " + ", ".join(
                b if isinstance(b, str) else f"to {b['to']}" for b in s["branches_leaving"]))
        if s["external_grid_here"]:
            where.append("The external grid connects here.")
    sup = result.get("supply") or {}
    supply = []
    if sup.get("route_from_external_grid"):
        supply.append(" → ".join(_short(r) for r in sup["route_from_external_grid"]))
        if sup.get("feeder"):
            supply.append(f"Feeder: **{sup['feeder'].split(' | ')[0]}**")
        elif sup.get("feeder") is None and not sup.get("radial_at_its_voltage_level", True):
            supply.append("Meshed at its voltage level: no single feeder.")
        if sup.get("note"):
            supply.append(sup["note"])
        supply.append("Inferred from the topology as switched now.")
    else:
        supply.append(sup.get("note") or "No supply route.")
    cards = [("Where it sits", where), ("How it is supplied", supply)]
    out = result.get("without_it")
    if out:
        lines = [f"**{len(out['buses_cut_off'])} bus(es)** lose supply"
                 + (f", with {out['load_cut_off_mw']:g} MW of load" if out["load_cut_off_mw"] else "") + "."]
        if out["units_cut_off"]:
            lines.append("Units cut off: " + ", ".join(out["units_cut_off"]))
        if out["loads_cut_off"]:
            lines.append("Loads cut off: " + ", ".join(out["loads_cut_off"]))
        checked = out.get("ties_checked") or []
        if checked:
            lines += [tie_line(t) for t in checked]
        elif out.get("open_ties_that_could_restore"):
            lines.append("Could be restored by closing: "
                         + "; ".join(t.split(" | ")[0] + " (" + t.split("(")[-1] for t in out["open_ties_that_could_restore"]))
        lines.append("What is cut off: topology. " + (f"Each tie: a power flow at {str(result['operating_point'])[:16]}, with "
                                                       "the element out and the tie closed."
                                                       if checked and result.get("operating_point") else
                                                       "No power flow run."))
        cards.append(("Without it", lines))
    return cards


def tie_line(t: dict) -> str:
    """One checked tie in words: what closing it restores, and whether the network stays within its limits."""
    tie = _short(t.get("tie", "")) if " | " in str(t.get("tie", "")) else str(t.get("tie", ""))
    if not t.get("converged"):
        return f"Closing **{tie}**: the power flow does not converge — it may not restore supply."
    back, cut = t.get("buses_restored", 0), t.get("buses_cut_off", 0)
    restores = "restores all " + str(cut) + " bus(es)" if back >= cut else f"restores {back} of {cut} bus(es)"
    parts = []
    low = t.get("lowest_voltage")
    if low:
        parts.append(f"lowest voltage {low['pu']:.3f} p.u. at {low['at']}")
    high = t.get("highest_loading")
    if high:
        parts.append(f"highest loading {high['pct']:.0f} % on {_short(high['on'])}")
    verdict = ("within limits" if t.get("within_limits")
               else "outside limits: " + ", ".join(t.get("violations") or []))
    return f"Closing **{tie}** {restores}: " + ", ".join(parts) + f" — {verdict}."


def short_label(text) -> str:
    """A branch's label as people say it ("Line 1-2"), without the ends and the internal key."""
    return _short(text)


def _short(text) -> str:
    """Branch labels as the tables show them: "Line 1-2 | Bus 1 -> Bus 2 [L0]" → "Line 1-2"."""
    import re
    return re.sub(r" \| [^|\[\]]*? -> [^|\[\]]*? \[[LT]\d+\]", "", str(text))


def switch_table(semantics: dict) -> list[dict]:
    """Switches, read-only: class (and how it is known), normal and present state, role."""
    rows = []
    for s in semantics.get("switches") or []:
        w = s["where"]
        place = (f"{w.get('element') or ''} at {w.get('at')}" if w.get("kind") in ("line", "transformer")
                 else " – ".join(str(x) for x in w.get("between", [])))
        rows.append({"Switch": s["name"], "Class": s["cim_class"]["value"],
                     "Class basis": BASIS_LABELS.get(s["cim_class"]["basis"]),
                     "Normally": "open" if s["normal_open"]["value"] else "closed",
                     "Now": "open" if s["open"]["value"] else "closed",
                     "Role": _short(s["role"]["value"]), "Where": _short(place)})
    return rows


def feeder_table(semantics: dict) -> list[dict]:
    """Feeders, read-only: what each supplies and the switches that tie it to others."""
    return [{"Feeder": _short(f["name"]), "Ties": ", ".join(f["ties"]) or "—", "Buses": len(f["buses"]),
             "Loads": len(f["loads"]), "Load now (MW)": f["load_mw"], "Units": ", ".join(f["units"])}
            for f in (semantics.get("containers") or {}).get("feeders") or []]


_KIND_LABELS = {"measured": "Measured", "forecast": "Forecast", "allocated": "Share of the bus's measurement",
                "derived": "Derived (assumption)", "solved": "Solved by the power flow", "set": "Set-point",
                "nominal": "Case's own value", "proposal": "From the proposal"}


def data_table(semantics: dict) -> list[dict]:
    """Where each load's and unit's P and Q come from, at the present operating point."""
    rows = []
    for kind, items in (("Load", semantics.get("loads") or []), ("Unit", semantics.get("units") or [])):
        for e in items:
            v = e.get("values")
            if not v:
                continue
            rows.append({"Element": e["name"], "Kind": kind, "Bus": e.get("bus"),
                         "P": _KIND_LABELS.get(v["p"]["kind"], v["p"]["kind"]),
                         "Q": _KIND_LABELS.get(v["q"]["kind"], v["q"]["kind"]), "Q from": v["q"]["from"]})
    return rows


def data_caption(semantics: dict) -> str:
    """The moment and the data the values are from, in one line."""
    ctx = semantics.get("provenance") or {}
    if not ctx:
        return "Where the values come from cannot be said now."
    pf = ctx.get("load_power_factor") or {}
    return (f"At {ctx.get('moment')}, from the {ctx.get('dataset')}"
            + (" (synthetic data)" if ctx.get("synthetic") else "") + ". "
            + ("Reactive power is in the data. " if ctx.get("reactive_in_data") else
               f"The data carry no reactive power: loads take it from the power factor {pf.get('value')} "
               f"({pf.get('source')}). ")
            + "A share of a bus's measurement is an estimate: the bus total is measured, not each element's part.")


def shunt_table(semantics: dict) -> list[dict]:
    """Shunt compensators, read-only: kind (capacitor or reactor), size, bus. The size is shown unsigned:
    the kind says which way the reactive power flows (the layer keeps pandapower's sign, negative for a
    capacitor)."""
    return [{"Shunt": x["name"], "Bus": x["bus"], "Kind": x["kind"]["value"],
             "Kind basis": BASIS_LABELS.get(x["kind"]["basis"]), "Size at 1 p.u. (Mvar)": abs(x["q_mvar_at_1pu"]),
             "In service": x["in_service"], "CIM class": x["cim_class"]} for x in semantics.get("shunts") or []]
