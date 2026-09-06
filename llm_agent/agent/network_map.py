"""
network_map.py — the shape of the loaded network, and where to draw each bus.

Reads `/api/network/topology`, classifies what each bus is, computes positions
for networks that carry no coordinates, and joins live results onto them. The
figure itself is built in `renderers.py`; nothing here imports plotly, so the
whole of it runs under the CI test dependencies.

Three things this has to get right, all of them learned by getting them wrong
in a standalone prototype first:

*Positions must account for how large things are drawn.* A layout that knows
only the graph puts a substation and its own transformer bus in the same
place. Marker sizes therefore live here, and positions are relaxed until
nothing overlaps.

*Spreading the drawing out achieves nothing.* Markers are sized in pixels and
the figure fits itself to whatever range the data occupies, so doubling every
coordinate doubles the marker's share of the plot with it. What can be shown
is fixed by marker pixels against plot pixels; only the arrangement inside
that budget is ours to change, and where it will not fit the markers shrink.

*Results join by index, never by name.* Bus names are not unique — one real
network has four buses called `Mike A_aux` — and a generator is named for
its substation while a bus is named with its voltage level. Joining those by
name silently placed thirteen of sixteen units one voltage level above where
they actually sat.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

import networkx as nx

# How large each kind of bus is drawn, in pixels. Here rather than in the
# renderer because the layout has to leave room for it.
MARKER_PX = {"external": 18, "slack": 17, "substation": 15,
             "distribution": 10, "junction": 5, "isolated": 9}

LAYOUTS = ("kamada", "spring", "shell")

Position = tuple[float, float]


# ---------------------------------------------------------------------------
# The graph
# ---------------------------------------------------------------------------


@dataclass
class Node:
    index: int
    name: str
    vn_kv: float
    in_service: bool
    kind: str = "bus"          # slack | substation | distribution | junction | external | isolated
    vm_pu: float | None = None
    load_mw: float = 0.0
    gen_mw: float = 0.0


@dataclass
class Edge:
    index: int
    name: str
    from_bus: int
    to_bus: int
    kind: str                  # line | trafo | ext_grid
    in_service: bool
    loading_pct: float | None = None


@dataclass
class NetworkMap:
    name: str
    nodes: dict[int, Node]
    edges: list[Edge]
    counts: dict[str, int] = field(default_factory=dict)
    voltage_levels: list[float] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    # Shrinks the markers where a network has more buses than the drawing can
    # show at full size. Set by the layout, the only place that knows how
    # densely the buses ended up packed.
    marker_scale: float = 1.0

    @property
    def duplicate_names(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for node in self.nodes.values():
            counts[node.name] = counts.get(node.name, 0) + 1
        return {name: n for name, n in counts.items() if n > 1}


def from_payload(payload: dict) -> NetworkMap:
    """Build the map from an `/api/network/topology` response."""
    nodes: dict[int, Node] = {}
    for entry in payload.get("buses") or []:
        index = int(entry["index"])
        nodes[index] = Node(
            index=index,
            name=str(entry.get("name") or f"Bus_{index}"),
            vn_kv=float(entry.get("vn_kv") or 0.0),
            in_service=bool(entry.get("in_service", True)),
        )

    edges: list[Edge] = []
    for entry in payload.get("branches") or []:
        edges.append(Edge(
            index=int(entry["index"]),
            name=str(entry.get("name") or ""),
            from_bus=int(entry["from_bus"]),
            to_bus=int(entry["to_bus"]),
            kind=str(entry.get("kind") or "line"),
            in_service=bool(entry.get("in_service", True)),
        ))

    # The external grid is an element of its own, not a property of the bus it
    # attaches to. Drawn only as a larger bus, nothing said where the system
    # connects to its neighbour, which is the first thing anyone looks for.
    # Negative indices, so they can never collide with a real bus.
    for position, entry in enumerate(payload.get("external_grids") or []):
        bus = int(entry["bus"])
        if bus not in nodes:
            continue
        index = -(position + 1)
        nodes[index] = Node(
            index=index,
            name=str(entry.get("name") or f"External grid {position + 1}"),
            vn_kv=nodes[bus].vn_kv,
            in_service=bool(entry.get("in_service", True)),
            kind="external",
        )
        edges.append(Edge(index=index, name=nodes[index].name, from_bus=index,
                          to_bus=bus, kind="ext_grid",
                          in_service=nodes[index].in_service))
        nodes[bus].kind = "slack"

    network = NetworkMap(
        name=str(payload.get("name") or "network"),
        nodes=nodes, edges=edges,
        counts=dict(payload.get("counts") or {}),
        voltage_levels=[float(v) for v in payload.get("voltage_levels") or []],
    )
    _classify(network)

    shared = network.duplicate_names
    if shared:
        network.notes.append(
            f"{sum(shared.values())} buses share {len(shared)} name(s) "
            f"({', '.join(sorted(shared))}); results are matched by index, not name."
        )
    return network


def _classify(network: NetworkMap) -> None:
    """Give every bus a role, because they are drawn differently.

    A cable joint carries two lines, no transformer and no demand. Drawing it
    the same size as a substation is what made the first attempt unreadable.
    """
    lines: dict[int, int] = {}
    trafos: dict[int, int] = {}
    for edge in network.edges:
        if edge.kind == "ext_grid":
            continue
        counter = lines if edge.kind == "line" else trafos
        counter[edge.from_bus] = counter.get(edge.from_bus, 0) + 1
        counter[edge.to_bus] = counter.get(edge.to_bus, 0) + 1

    for node in network.nodes.values():
        if node.kind in ("slack", "external"):
            continue
        n_lines, n_trafos = lines.get(node.index, 0), trafos.get(node.index, 0)
        if n_lines == 0 and n_trafos > 0:
            node.kind = "distribution"
        elif n_lines == 2 and n_trafos == 0:
            node.kind = "junction"
        elif n_lines or n_trafos:
            node.kind = "substation"
        else:
            node.kind = "isolated"


# ---------------------------------------------------------------------------
# Live results
# ---------------------------------------------------------------------------


def apply_assessment(network: NetworkMap, rsa: dict) -> list[str]:
    """Colour the map with the operating point an assessment reported.

    Returns what could not be joined. Silence would be the dangerous outcome:
    an unjoined element renders grey, which reads as "no problem" rather than
    "not measured".
    """
    problems: list[str] = []

    by_name: dict[str, list[Node]] = {}
    for node in network.nodes.values():
        by_name.setdefault(node.name, []).append(node)

    measured = 0
    for entry in rsa.get("all_voltages") or []:
        targets = by_name.get(str(entry.get("bus_name") or ""), [])
        if len(targets) != 1:
            if not targets:
                problems.append(f"voltage reported for unknown bus {entry.get('bus_name')!r}")
            else:
                problems.append(
                    f"voltage for {entry.get('bus_name')!r} matches {len(targets)} buses"
                )
            continue
        value = entry.get("vm_pu")
        targets[0].vm_pu = float(value) if value is not None else None
        measured += 1

    by_edge = {edge.name: edge for edge in network.edges}
    for key in ("all_line_loading", "all_trafo_loading"):
        for entry in rsa.get(key) or []:
            name = entry.get("line_name") or entry.get("trafo_name")
            edge = by_edge.get(str(name or ""))
            if edge is None:
                problems.append(f"loading reported for unknown element {name!r}")
                continue
            edge.loading_pct = float(entry.get("loading_percent") or 0.0)

    network.notes.append(f"{measured}/{len(network.nodes)} buses carry a measured voltage.")
    return problems


def apply_conditions(network: NetworkMap, conditions: dict) -> list[str]:
    """Attach load and generation from a network snapshot.

    Joined on `bus_index`, which is exact. The name fallback exists for older
    payloads and says so when used: a unit is named after its substation while
    a bus is named with its voltage level, so matching by name found the
    63 kV substation when the unit actually sits on the 10.5 kV bus beneath
    it — thirteen of sixteen joined that way, and every one was wrong.
    """
    problems: list[str] = []
    by_name: dict[str, list[Node]] = {}
    for node in network.nodes.values():
        by_name.setdefault(node.name, []).append(node)

    def locate(entry: dict, label: str) -> Node | None:
        index = entry.get("bus_index")
        if index is not None:
            node = network.nodes.get(int(index))
            if node is None:
                problems.append(f"{label} reported at bus {index}, absent from this network")
            return node

        key = str(entry.get("bus") or entry.get("name") or "")
        targets = by_name.get(key, [])
        if len(targets) == 1:
            problems.append(
                f"{label} {key!r} was located by name, which the payload cannot "
                "guarantee is the right bus"
            )
            return targets[0]
        problems.append(
            f"{label} {key!r} could not be located: "
            + ("no bus of that name" if not targets else f"{len(targets)} buses share it")
        )
        return None

    for entry in conditions.get("loads") or []:
        node = locate(entry, "load")
        if node is not None:
            node.load_mw = float(entry.get("P_mw") or 0.0)
    for entry in conditions.get("generators") or []:
        node = locate(entry, "generator")
        if node is not None:
            node.gen_mw = float(entry.get("Pg_mw") or 0.0)

    return problems


def violated_elements(rsa: dict) -> set[str]:
    """Names the assessment flagged, for highlighting."""
    return {str(v.get("element")) for v in (rsa.get("violations") or []) if v.get("element")}


_TIMESTAMP_RE = re.compile(r"\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}(?::\d{2})?)?")


def operating_point_of(results: Iterable[tuple[str, Any]]) -> str | None:
    """The timestamp a turn's tools actually ran at, if they agree on one.

    Used to keep the system view on whatever moment the conversation is
    discussing. Disagreement means the turn spanned more than one operating
    point, and no single timestamp describes it.
    """
    seen: set[str] = set()
    for _, result in results or ():
        if not isinstance(result, dict):
            continue
        value = result.get("timestamp") or result.get("current_timestamp")
        if isinstance(value, str) and _TIMESTAMP_RE.match(value.strip()):
            seen.add(value.strip()[:19])
    return seen.pop() if len(seen) == 1 else None


# ---------------------------------------------------------------------------
# Layout
#
# Core first, then whatever hangs off it: peel degree-one buses in rounds until
# what remains has a shape, lay that out, and re-attach the peeled buses
# outward. The behaviour then follows from the network rather than from an
# assumption about it — a meshed grid barely peels and this is plain
# kamada-kawai, a radial feeder has no core and is laid out whole.
# ---------------------------------------------------------------------------

_MIN_CORE = 4
# kamada-kawai solves a dense all-pairs problem and does not finish on a large
# transmission case. Past this the layout degrades on purpose, and says so.
_KAMADA_LIMIT = 900
_SPRING_ITERATIONS = ((2000, 200), (8000, 60), (float("inf"), 25))

_EXTENT_PX = 660
_PAD_PX = 2.5
# Circles pack to about 0.9 of a plane at best, and a drawing that dense is
# unreadable anyway. Past this share of the box, markers shrink.
_TARGET_PACKING = 0.22
_MIN_MARKER_SCALE = 0.3


def _graph(network: NetworkMap) -> nx.Graph:
    graph = nx.Graph()
    for node in network.nodes.values():
        graph.add_node(node.index)
    for edge in network.edges:
        if edge.from_bus in graph and edge.to_bus in graph:
            graph.add_edge(edge.from_bus, edge.to_bus)
    return graph


def _peel(graph: nx.Graph) -> tuple[nx.Graph, list[list[tuple[int, int]]]]:
    core = graph.copy()
    rounds: list[list[tuple[int, int]]] = []
    while True:
        leaves = [n for n in core.nodes if core.degree(n) == 1]
        if not leaves or core.number_of_nodes() - len(leaves) < _MIN_CORE:
            break
        rounds.append([(leaf, next(iter(core.neighbors(leaf)))) for leaf in leaves])
        core.remove_nodes_from(leaves)
    return core, rounds


def _iterations(size: int) -> int:
    for limit, count in _SPRING_ITERATIONS:
        if size <= limit:
            return count
    return _SPRING_ITERATIONS[-1][1]


def _lay_out(graph: nx.Graph, kind: str, seed: int,
             notes: list[str] | None = None) -> dict[int, Position]:
    size = graph.number_of_nodes()
    if size == 0:
        return {}
    if size == 1:
        return {int(next(iter(graph.nodes))): (0.0, 0.0)}

    if kind == "kamada" and size > _KAMADA_LIMIT:
        if notes is not None:
            notes.append(
                f"{size} buses is past the {_KAMADA_LIMIT} at which kamada-kawai stops "
                "finishing; a force-directed layout was used instead."
            )
        kind = "spring"

    if kind == "spring":
        raw = nx.spring_layout(graph, seed=seed, iterations=_iterations(size))
    elif kind == "shell":
        raw = nx.shell_layout(graph)
    else:
        raw = nx.kamada_kawai_layout(graph)
    return {int(k): (float(v[0]), float(v[1])) for k, v in raw.items()}


def compute_layout(network: NetworkMap, kind: str = "kamada", seed: int = 7,
                   satellite: float = 0.17, include_isolated: bool = False,
                   separate: bool = True) -> dict[int, Position]:
    """Positions keyed by bus index, and by negative index for external grids."""
    graph = _graph(network)
    connected = graph.subgraph([n for n in graph.nodes if graph.degree(n) > 0])

    positions: dict[int, Position] = {}
    for offset, component in enumerate(
        sorted(nx.connected_components(connected), key=len, reverse=True)
    ):
        placed = _lay_out_component(network, connected.subgraph(component),
                                    kind, seed, satellite, network.notes)
        if offset:
            placed = {n: (x, y - 2.6 * offset) for n, (x, y) in placed.items()}
        positions.update(placed)

    if include_isolated:
        _park(network, positions)
    if separate and positions:
        remaining = _separate(network, positions)
        if remaining:
            network.notes.append(
                f"{remaining} pair(s) of buses still overlap — the network is denser "
                "than the drawing area can show at this marker size."
            )

    hidden = len(network.nodes) - len(positions)
    if hidden:
        network.notes.append(f"{hidden} bus(es) connected to nothing are not drawn.")
    return positions


def _lay_out_component(network: NetworkMap, graph: nx.Graph, kind: str, seed: int,
                       satellite: float, notes: list[str]) -> dict[int, Position]:
    core, rounds = _peel(graph)
    if core.number_of_nodes() < _MIN_CORE:
        return _lay_out(graph, kind, seed, notes)

    positions = _lay_out(core, kind, seed, notes)
    for peeled in reversed(rounds):      # innermost first, so anchors exist
        _attach(network, positions, peeled, satellite)
    return positions


def _attach(network: NetworkMap, positions: dict[int, Position],
            peeled: list[tuple[int, int]], distance: float) -> None:
    """Place peeled buses outward from the anchor they hang off."""
    if not positions:
        return
    cx = sum(p[0] for p in positions.values()) / len(positions)
    cy = sum(p[1] for p in positions.values()) / len(positions)

    grouped: dict[int, list[int]] = {}
    for leaf, anchor in peeled:
        grouped.setdefault(anchor, []).append(leaf)

    for anchor, leaves in grouped.items():
        origin = positions.get(anchor)
        if origin is None:
            continue
        dx, dy = origin[0] - cx, origin[1] - cy
        base = math.atan2(dy, dx) if math.hypot(dx, dy) > 1e-9 else 0.0
        spread = math.radians(32 if len(leaves) <= 2 else 26)
        ordered = sorted(leaves, key=lambda i: (network.nodes[i].kind != "external", i))
        for position, leaf in enumerate(ordered):
            angle = base + spread * (position - (len(ordered) - 1) / 2)
            # The external grid sits further out: it is the boundary of the
            # modelled system, not part of it.
            reach = distance * (2.1 if network.nodes[leaf].kind == "external" else 1.0)
            positions[leaf] = (origin[0] + reach * math.cos(angle),
                               origin[1] + reach * math.sin(angle))


def _park(network: NetworkMap, positions: dict[int, Position]) -> None:
    ys = [p[1] for p in positions.values()] or [0.0]
    xs = [p[0] for p in positions.values()] or [0.0]
    row, start = min(ys) - 0.4, min(xs)
    for offset, node in enumerate(
        sorted(n.index for n in network.nodes.values() if n.index not in positions)
    ):
        positions[node] = (start + 0.12 * offset, row)


# ---------------------------------------------------------------------------
# Keeping buses off each other
# ---------------------------------------------------------------------------


def _normalise(positions: dict[int, Position]) -> None:
    xs = [p[0] for p in positions.values()]
    ys = [p[1] for p in positions.values()]
    scale = 1.0 / (max(max(xs) - min(xs), max(ys) - min(ys)) or 1.0)
    ox, oy = min(xs), min(ys)
    for index, (x, y) in positions.items():
        positions[index] = ((x - ox) * scale, (y - oy) * scale)


def marker_radii(network: NetworkMap, positions: dict[int, Position],
                 marker_scale: float | None = None) -> dict[int, float]:
    """Marker radius in unit-box coordinates. Constant, by construction.

    The pad shrinks with the marker. Leaving it fixed meant shrinking markers
    barely reduced the space they needed, so a dense network stayed dense
    however small its dots were drawn.
    """
    scale = network.marker_scale if marker_scale is None else marker_scale
    return {
        index: (MARKER_PX.get(network.nodes[index].kind, 10) / 2 + _PAD_PX)
        * scale / _EXTENT_PX
        for index in positions if index in network.nodes
    }


def _marker_scale(network: NetworkMap, positions: dict[int, Position]) -> float:
    packing = sum(math.pi * r * r for r in marker_radii(network, positions, 1.0).values())
    if packing <= _TARGET_PACKING:
        return 1.0
    return max(_MIN_MARKER_SCALE, math.sqrt(_TARGET_PACKING / packing))


def _separate(network: NetworkMap, positions: dict[int, Position],
              rounds: int = 1500, damping: float = 0.9) -> int:
    """Rearrange buses inside the unit box until none overlap.

    Displacements accumulate over a whole sweep and are applied together.
    Moving each bus the moment a clash is found — the obvious way to write
    this — makes matters worse: a bus already shoved aside is shoved again by
    the next pair in the same sweep, overshoots, and lands on something else.
    One network went from 20 overlaps to 25 that way.

    Damping is high on purpose. A gentler 0.55 looked safer and never
    finished: a 181-bus network sat at 137 overlaps however many sweeps it was
    given. Sweeps are cheap and the loop exits as soon as nothing clashes.
    """
    if len(positions) < 2:
        return 0

    _normalise(positions)
    network.marker_scale = _marker_scale(network, positions)
    radii = marker_radii(network, positions)
    if not radii:
        return 0

    mobility = {
        index: 1.0 if network.nodes[index].kind in ("distribution", "external", "isolated")
        else 0.35
        for index in radii
    }
    cell = 2 * max(radii.values())
    margin = max(radii.values())

    for _ in range(rounds):
        grid: dict[tuple[int, int], list[int]] = {}
        for index, (x, y) in positions.items():
            if index in radii:
                grid.setdefault((int(x // cell), int(y // cell)), []).append(index)

        shift: dict[int, list[float]] = {}
        clashes = 0
        for (gx, gy), members in grid.items():
            nearby: list[int] = []
            for ox in (-1, 0, 1):
                for oy in (-1, 0, 1):
                    nearby.extend(grid.get((gx + ox, gy + oy), ()))
            for a in members:
                ax, ay = positions[a]
                for b in nearby:
                    if b <= a:
                        continue
                    bx, by = positions[b]
                    dx, dy = bx - ax, by - ay
                    distance = math.hypot(dx, dy)
                    wanted = radii[a] + radii[b]
                    if distance >= wanted:
                        continue
                    clashes += 1
                    if distance < 1e-12:
                        # Coincident: a direction fixed per bus, so the result
                        # does not depend on dictionary order.
                        angle = (a * 2.399963) % (2 * math.pi)
                        dx, dy = math.cos(angle), math.sin(angle)
                    else:
                        dx, dy = dx / distance, dy / distance

                    push = (wanted - distance) * damping
                    share = mobility[a] + mobility[b]
                    for node, sign in ((a, -1.0), (b, 1.0)):
                        step = push * mobility[node] / share
                        entry = shift.setdefault(node, [0.0, 0.0])
                        entry[0] += sign * dx * step
                        entry[1] += sign * dy * step

        if not clashes:
            return 0
        # Held inside the box: letting the layout grow would look like progress
        # and change nothing on the page, since the figure fits itself to the
        # data and the markers grow with it.
        span = 1.0 + 2 * margin
        for node, (dx, dy) in shift.items():
            x, y = positions[node]
            positions[node] = (min(span, max(-margin, x + dx)),
                               min(span, max(-margin, y + dy)))

    return count_overlaps(network, positions)


def count_overlaps(network: NetworkMap, positions: dict[int, Position]) -> int:
    radii = marker_radii(network, positions)
    indices = [i for i in positions if i in radii]
    total = 0
    for first, a in enumerate(indices):
        for b in indices[first + 1:]:
            gap = math.hypot(positions[a][0] - positions[b][0],
                             positions[a][1] - positions[b][1])
            if gap < radii[a] + radii[b] - 1e-12:
                total += 1
    return total
