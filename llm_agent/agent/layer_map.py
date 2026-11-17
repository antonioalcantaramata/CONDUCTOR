"""
layer_map.py — the network map coloured by the CIM layer instead of the
operating point: each bus by the energy source of its units, by its feeder, or
by how well the layer knows what is there (stated, read from a name, unknown).

Pure classification (`bus_groups`, `edge_groups`) over the layer's payload
(`/api/network/semantics`) and the map's own model (`network_map.NetworkMap`),
then one figure (`render_layer_map`) drawn like the operating-point map. Works
for any network: everything comes from the layer, nothing from names.
"""

from __future__ import annotations

from .network_map import MARKER_PX, NetworkMap
from .renderers import _C

MODES = {"source": "Energy source", "feeder": "Feeder", "known": "What is known"}

# One colour per source, fixed (the order is the legend's). Red stays reserved for violations.
SOURCE_COLOURS = {"solar": _C.WARN, "wind": _C.TEAL, "hydro": _C.PRIMARY, "bioenergy": _C.OLIVE,
                  "thermal": _C.MAGENTA, "nuclear": "#6B5B95", "storage": _C.PURPLE, "condenser": "#7A7F8C",
                  "other": "#9C8B6E", "mixed": "#3F4458", "unknown": _C.NEUTRAL}
KNOWN_COLOURS = {"stated": _C.OK, "named": _C.WARN, "unknown": _C.NEUTRAL}
KNOWN_LABELS = {"stated": "all stated (by the network, its profile or you)",
                "named": "something read only from a name",
                "unknown": "something the network does not say"}
# Feeders take the five categorical hues validated together (renderers._C), in a fixed order; more fold into
# "other feeders".
FEEDER_COLOURS = [_C.PRIMARY, _C.TEAL, _C.PURPLE, _C.MAGENTA, _C.OLIVE]
NOTHING = "#D9DCE3"          # a bus with nothing to show in this view
_EDGE = "#B8BCC6"


def _short(name: str) -> str:
    return str(name or "").split(" | ")[0]


def _by_bus(entries: list[dict]) -> dict[int, list[dict]]:
    out: dict[int, list[dict]] = {}
    for e in entries or []:
        if e.get("bus_index") is not None:
            out.setdefault(int(e["bus_index"]), []).append(e)
    return out


def bus_groups(layer: dict, by: str) -> tuple[dict[int, tuple[str, str]], list[str], dict[str, str]]:
    """({bus index: (group, hover detail)}, the groups in legend order, {group: colour}) for one view. A bus
    with nothing to show in the view is left out (drawn as `NOTHING`)."""
    units, loads = _by_bus(layer.get("units")), _by_bus(layer.get("loads"))
    out: dict[int, tuple[str, str]] = {}
    if by == "source":
        for bus, members in units.items():
            sources = {u["source"]["value"] if u["source"]["basis"] != "unknown" else "unknown" for u in members}
            group = next(iter(sources)) if len(sources) == 1 else "mixed"
            group = group if group in SOURCE_COLOURS else "other"
            detail = "<br>".join(f"{u['name']}: {u['source']['value']} ({u['source']['basis']})" for u in members)
            out[bus] = (group, detail)
        order = [g for g in SOURCE_COLOURS if any(v[0] == g for v in out.values())]
        return out, order, {g: SOURCE_COLOURS[g] for g in order}
    if by == "feeder":
        for b in layer.get("buses") or []:
            feeder = (b.get("feeder") or {}).get("value")
            if feeder:
                out[int(b["index"])] = (_short(feeder), f"feeder {_short(feeder)} (inferred from the topology)")
        names = sorted({v[0] for v in out.values()})
        colours = {n: FEEDER_COLOURS[i] for i, n in enumerate(names[:len(FEEDER_COLOURS)])}
        if len(names) > len(FEEDER_COLOURS):
            colours["other feeders"] = "#7A7F8C"
            out = {k: (g if g in colours else "other feeders", d) for k, (g, d) in out.items()}
        return out, list(colours), colours
    if by == "known":
        for bus in set(units) | set(loads):
            facts = [(u["name"], "source", u["source"]) for u in units.get(bus, [])] + \
                    [(d["name"], "category", d["category"]) for d in loads.get(bus, [])]
            bases = {f["basis"] for _n, _w, f in facts}
            group = "unknown" if "unknown" in bases else "named" if "named" in bases else "stated"
            detail = "<br>".join(f"{n}: {w} {f['value'].replace('_', ' ')} ({f['basis']})" for n, w, f in facts)
            out[bus] = (group, detail)
        order = [g for g in KNOWN_COLOURS if any(v[0] == g for v in out.values())]
        return out, order, {g: KNOWN_COLOURS[g] for g in order}
    raise ValueError(f"Unknown view {by!r}; one of {', '.join(MODES)}.")


def edge_groups(network: NetworkMap, buses: dict[int, tuple[str, str]], by: str) -> dict[int, str]:
    """{edge position: group} — in the feeder view, a branch whose ends share a feeder takes its colour."""
    if by != "feeder":
        return {}
    out = {}
    for k, edge in enumerate(network.edges):
        a, b = buses.get(edge.from_bus), buses.get(edge.to_bus)
        if a and b and a[0] == b[0]:
            out[k] = a[0]
    return out


def legend_label(by: str, group: str) -> str:
    if by == "known":
        return KNOWN_LABELS.get(group, group)
    if by == "source" and group == "unknown":
        return "source unknown"
    return group


def render_layer_map(network: NetworkMap, positions: dict, layer: dict, by: str, subtitle: str = ""):
    """The map coloured by the CIM layer (`MODES`). Branches are grey but in the feeder view; buses with
    nothing to show in the view are small and pale."""
    import plotly.graph_objects as go

    from .renderers import _FONT_PX, _PLOT_PX, _empty_figure, _place_labels

    drawn = [n for n in network.nodes.values() if n.index in positions]
    if not drawn:
        return _empty_figure("No network to draw")
    buses, order, colours = bus_groups(layer, by)
    edges = edge_groups(network, buses, by)
    xs = [positions[n.index][0] for n in drawn]
    ys = [positions[n.index][1] for n in drawn]
    span = max(max(ys) - min(ys), max(xs) - min(xs)) or 1.0
    per_px = span / _PLOT_PX

    traces = []
    for k, edge in enumerate(network.edges):
        if edge.from_bus not in positions or edge.to_bus not in positions:
            continue
        (x0, y0), (x1, y1) = positions[edge.from_bus], positions[edge.to_bus]
        group = edges.get(k)
        colour = colours.get(group, _EDGE) if group else ("#3f3f46" if edge.kind == "ext_grid" else _EDGE)
        traces.append(go.Scatter(x=[x0, x1], y=[y0, y1], mode="lines", hoverinfo="skip", showlegend=False,
                                 line=dict(color=colour, width=3.2 if group else 1.4,
                                           dash="dot" if edge.kind == "trafo" else "solid")))

    sizes, names, ranked, obstacles, hovers, slots_of = [], [], [], [], [], {}
    for slot, node in enumerate(drawn):
        group, detail = buses.get(node.index, (None, ""))
        # A bus that has something to show is never smaller than its colour can be read at.
        base = MARKER_PX.get(node.kind, 10) * network.marker_scale
        size = max(base, 13.0 * max(network.marker_scale, 0.6)) if group else (
            base if node.kind == "external" else max(base * 0.7, 5.0))
        sizes.append(size)
        names.append(node.name)
        ranked.append((0 if node.kind in ("slack", "external") else 1 if group else 3, slot, slot))
        half = (size / 2 + 3) * per_px
        obstacles.append((xs[slot] - half, ys[slot] - half, xs[slot] + half, ys[slot] + half))
        what = ("external grid" if node.kind == "external" else
                f"{node.vn_kv:g} kV · bus {node.index}")
        hovers.append(f"<b>{node.name}</b><br>{what}" + (f"<br>{detail}" if detail else ""))
        key = "external grid" if node.kind == "external" else group or "nothing here"
        slots_of.setdefault(key, []).append(slot)
    labels, placements, _dropped = _place_labels(xs, ys, names, sizes, ranked, obstacles, per_px)

    for key in [*order, "nothing here", "external grid"]:
        members = slots_of.get(key) or []
        if not members:
            continue
        colour = (colours.get(key) or ("#3f3f46" if key == "external grid" else NOTHING))
        hollow = key == "unknown"
        traces.append(go.Scatter(
            x=[xs[i] for i in members], y=[ys[i] for i in members], mode="markers+text",
            name=("external grid" if key == "external grid" else
                  "no unit or load" if key == "nothing here" and by != "feeder" else
                  "on no feeder" if key == "nothing here" else legend_label(by, key)),
            marker=dict(size=[sizes[i] for i in members], color=colour,
                        symbol="square" if key == "external grid" else ("circle-open" if hollow else "circle"),
                        line=dict(color=colour if hollow else "#5b5b5b", width=2.2 if hollow else 1.0)),
            text=[labels[i] for i in members], textposition=[placements[i] for i in members],
            textfont=dict(size=_FONT_PX, color="#333"),
            hoverinfo="text", hovertext=[hovers[i] for i in members]))

    title = f"<b>{network.name}</b> · coloured by {MODES[by].lower()}"
    if subtitle:
        title += f"<br><span style='font-size:12px;color:#666'>{subtitle}</span>"
    fig = go.Figure(data=traces)
    fig.update_layout(
        title=dict(text=title, x=0.01, xanchor="left"), hovermode="closest",
        plot_bgcolor="white", paper_bgcolor="white", margin=dict(l=20, r=20, t=74, b=44), height=720,
        legend=dict(orientation="v", x=1.0, y=1.0, xanchor="left", bgcolor="rgba(255,255,255,0.8)",
                    font=dict(size=10)),
        # Room for the names at the edges ("External grid" was cut).
        xaxis=dict(visible=False, showgrid=False, zeroline=False,
                   range=[min(xs) - 0.08 * span, max(xs) + 0.14 * span]),
        yaxis=dict(visible=False, showgrid=False, zeroline=False, scaleanchor="x", scaleratio=1,
                   range=[min(ys) - 0.06 * span, max(ys) + 0.06 * span]),
        annotations=[dict(text="from the CIM layer · schematic connectivity, not geography", xref="paper",
                          yref="paper", x=0.0, y=-0.045, showarrow=False, font=dict(size=10, color="#888"),
                          xanchor="left")])
    return fig
