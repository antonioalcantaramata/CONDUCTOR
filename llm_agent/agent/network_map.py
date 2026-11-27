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

LAYOUTS = ("kamada", "spring", "shell", "diagram")

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
    # A node-breaker model's nodes joined by closed switches are drawn as one electrical node: the buses it stands
    # for, and their names (results report a bus by its own name or index).
    members: list[int] = field(default_factory=list)
    member_names: list[str] = field(default_factory=list)


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
    # {bus index: the drawn node's index} for the buses merged into another node.
    alias: dict[int, int] = field(default_factory=dict)
    # {node index: (x, y)}: the model's own diagram (a CIM model's DL profile), when it places the network.
    diagram: dict[int, Position] = field(default_factory=dict)
    drawn_from_diagram: bool = False      # set by the layout when it used the diagram

    @property
    def what_is_drawn(self) -> str:
        """For the map's footer."""
        return ("the model's own diagram, not geography" if self.drawn_from_diagram
                else "schematic connectivity, not geography")

    def node(self, index) -> Node | None:
        """The drawn node a bus is, following merges."""
        i = int(index)
        return self.nodes.get(self.alias.get(i, i))

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

    alias = _merge_switched_nodes(nodes, edges, payload)
    edges = [e for e in edges if e.from_bus != e.to_bus]

    # The external grid is an element of its own, not a property of the bus it
    # attaches to. Drawn only as a larger bus, nothing said where the system
    # connects to its neighbour, which is the first thing anyone looks for.
    # Negative indices, so they can never collide with a real bus.
    for position, entry in enumerate(payload.get("external_grids") or []):
        bus = alias.get(int(entry["bus"]), int(entry["bus"]))
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
        alias=alias, diagram=_diagram(nodes, payload),
    )
    _classify(network)
    merged = sum(len(n.members) - 1 for n in nodes.values() if n.members)
    if merged:
        network.notes.append(f"{merged} node(s) joined to others by closed switches are drawn as one electrical "
                             "node each (the model is node-breaker).")

    shared = network.duplicate_names
    if shared:
        network.notes.append(
            f"{sum(shared.values())} buses share {len(shared)} name(s) "
            f"({', '.join(sorted(shared))}); results are matched by index, not name."
        )
    return network


def _merge_switched_nodes(nodes: dict[int, Node], edges: list[Edge], payload: dict) -> dict[int, int]:
    """Merge the nodes a node-breaker model joins by closed switches into one drawn node each (in place), named
    after its busbar or bus name in the model when it has one; re-point the edges. Returns {merged bus: node}.
    Only for a payload that says the model is node-breaker: any other network's map is drawn bus by bus."""
    labels = {int(k): str(v) for k, v in (payload.get("labels") or {}).items()}
    if not payload.get("node_breaker"):
        return {}
    parent = {i: i for i in nodes}

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for c in payload.get("couplers") or []:
        a, b = int(c["from_bus"]), int(c["to_bus"])
        if c.get("closed") and a in parent and b in parent:
            ra, rb = root(a), root(b)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
    groups: dict[int, list[int]] = {}
    for i in nodes:
        groups.setdefault(root(i), []).append(i)
    alias: dict[int, int] = {}
    for members in groups.values():
        # The same head in the base and in a proposal's map (they are compared node by node): never chosen by
        # what is connected, which a proposal can change.
        named = [m for m in members if m in labels]
        head = min(named) if named else min(members)
        node = nodes[head]
        node.members = sorted(members)
        node.member_names = [nodes[m].name for m in node.members]
        if named:
            node.name = labels[head]
        node.in_service = any(nodes[m].in_service for m in members)
        for m in members:
            if m != head:
                alias[m] = head
    for m in alias:
        nodes.pop(m, None)
    for e in edges:
        e.from_bus, e.to_bus = alias.get(e.from_bus, e.from_bus), alias.get(e.to_bus, e.to_bus)
    return alias


def _diagram(nodes: dict[int, Node], payload: dict) -> dict[int, Position]:
    """{node: (x, y)} from the model's diagram positions of its buses (averaged over a merged node's members);
    y flipped, since a CIM diagram's y grows downwards."""
    placed = {int(k): (float(v[0]), float(v[1])) for k, v in (payload.get("positions") or {}).items()}
    if not placed:
        return {}
    out = {}
    for index, node in nodes.items():
        pts = [placed[m] for m in (node.members or [index]) if m in placed]
        if pts:
            out[index] = (sum(p[0] for p in pts) / len(pts), -sum(p[1] for p in pts) / len(pts))
    return out


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


def _nodes_by_name(network: NetworkMap) -> dict[str, list[Node]]:
    """{name: [drawn nodes]}: each node by its own name and, for a merged node, by its buses' names."""
    out: dict[str, list[Node]] = {}
    for node in network.nodes.values():
        for name in dict.fromkeys([node.name, *node.member_names]):
            out.setdefault(name, []).append(node)
    return out


def apply_assessment(network: NetworkMap, rsa: dict) -> list[str]:
    """Colour the map with the operating point an assessment reported.

    Returns what could not be joined. Silence would be the dangerous outcome:
    an unjoined element renders grey, which reads as "no problem" rather than
    "not measured".
    """
    problems: list[str] = []

    by_name = _nodes_by_name(network)

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
        if targets[0].members and targets[0].vm_pu is not None:
            continue                      # a merged node: its buses share one voltage, counted once
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
            # A branch the studied network has open (a proposal can open one
            # the base drawing shows closed) is drawn out of service.
            if entry.get("in_service") is False:
                edge.in_service = False
            # None stays None: an unsolved branch is "not measured" (grey),
            # never 0 % loaded — which drew it as the healthiest green.
            value = entry.get("loading_percent")
            try:
                edge.loading_pct = None if value is None else float(value)
            except (TypeError, ValueError):
                edge.loading_pct = None

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
    by_name = _nodes_by_name(network)

    def locate(entry: dict, label: str) -> Node | None:
        index = entry.get("bus_index")
        if index is not None:
            node = network.node(index)
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
    """Positions keyed by bus index, and by negative index for external grids. A network whose model brings its
    own diagram (a CIM model's DL profile) is drawn as that diagram unless another layout is asked for."""
    graph = _graph(network)
    connected = graph.subgraph([n for n in graph.nodes if graph.degree(n) > 0])
    if kind in ("kamada", "diagram") and network.diagram:
        drawn = _from_diagram(network, graph, connected)
        if drawn is not None:
            if include_isolated:
                _park(network, drawn)
            if separate and drawn:
                remaining = _separate(network, drawn)
                if remaining:
                    network.notes.append(f"{remaining} pair(s) of buses still overlap — the network is denser "
                                         "than the drawing area can show at this marker size.")
            hidden = len(network.nodes) - len(drawn)
            if hidden:
                network.notes.append(f"{hidden} bus(es) connected to nothing are not drawn.")
            return drawn

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


# Below this share of the drawn buses placed by the model's diagram, the diagram is not used.
_DIAGRAM_COVERAGE = 0.8


def _from_diagram(network: NetworkMap, graph: nx.Graph, connected: nx.Graph) -> dict[int, Position] | None:
    """The model's diagram as positions, or None when it places too few of the buses drawn. A bus it does not place
    (an external grid, a node a proposal added) goes beside its placed neighbours."""
    drawn = list(connected.nodes)
    placed = {n: network.diagram[n] for n in drawn if n in network.diagram}
    buses = [n for n in drawn if n >= 0]          # an external grid (negative index) is never in a diagram
    if not buses or len(placed) < _DIAGRAM_COVERAGE * len(buses):
        return None
    xs, ys = [p[0] for p in placed.values()], [p[1] for p in placed.values()]
    step = 0.03 * (max(max(xs) - min(xs), max(ys) - min(ys)) or 1.0)
    missing = [n for n in drawn if n not in placed]
    for _ in range(4):
        for n in list(missing):
            near = [placed[m] for m in graph.neighbors(n) if m in placed]
            if near:
                placed[n] = (sum(p[0] for p in near) / len(near) + step, sum(p[1] for p in near) / len(near) + step)
                missing.remove(n)
    for k, n in enumerate(missing):          # placed neither by the diagram nor through a neighbour
        placed[n] = (min(xs) + k * step, min(ys) - 2 * step)
    network.notes.append("Drawn as the model's own diagram (its DL profile).")
    network.drawn_from_diagram = True
    return placed


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
