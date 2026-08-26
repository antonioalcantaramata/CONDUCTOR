"""
The network map: what the system is, and where to draw it.

These cover the parts that decide whether a drawing is *true*, not whether it
is pretty. Every one of them exists because the standalone prototype got it
wrong first, and a wrong drawing is the most confidently misleading artifact
the app can produce — it looks authoritative and nothing in it says which bus
a figure really belongs to.

No plotly here: the figure lives in `renderers.py`, which CI does not install
a plotting stack for. Keeping the geometry separate is what lets it be tested
at all.
"""

import math

import pytest

from llm_agent.agent import network_map as nm


def payload(buses=None, branches=None, externals=None, **extra):
    return {
        "name": "test grid",
        "buses": buses if buses is not None else [
            {"index": 0, "name": "A", "vn_kv": 63.0, "in_service": True},
            {"index": 1, "name": "B", "vn_kv": 63.0, "in_service": True},
            {"index": 2, "name": "C", "vn_kv": 63.0, "in_service": True},
            {"index": 3, "name": "A 10 kV", "vn_kv": 10.0, "in_service": True},
        ],
        "branches": branches if branches is not None else [
            {"index": 0, "name": "A-B", "from_bus": 0, "to_bus": 1, "kind": "line",
             "in_service": True},
            {"index": 1, "name": "B-C", "from_bus": 1, "to_bus": 2, "kind": "line",
             "in_service": True},
            {"index": 2, "name": "C-A", "from_bus": 2, "to_bus": 0, "kind": "line",
             "in_service": True},
            {"index": 0, "name": "A Trf", "from_bus": 0, "to_bus": 3, "kind": "trafo",
             "in_service": True},
        ],
        "external_grids": externals if externals is not None else [
            {"name": "Slack", "bus": 0, "in_service": True},
        ],
        **extra,
    }


class TestClassification:
    def test_roles_follow_structure_not_voltage(self):
        network = nm.from_payload(payload())
        kinds = {n.name: n.kind for n in network.nodes.values()}
        assert kinds["A"] == "slack"          # carries the external grid
        assert kinds["A 10 kV"] == "distribution"   # reached only by a transformer
        assert kinds["Slack"] == "external"
        assert kinds["B"] == "junction"       # two lines, nothing else
        assert kinds["C"] == "junction"

    def test_a_bus_with_a_transformer_is_not_a_junction(self):
        network = nm.from_payload(payload())
        assert network.nodes[0].kind != "junction"

    def test_a_bus_connected_to_nothing_is_isolated(self):
        data = payload()
        data["buses"].append({"index": 9, "name": "Spare", "vn_kv": 63.0, "in_service": False})
        network = nm.from_payload(data)
        assert network.nodes[9].kind == "isolated"

    def test_the_external_grid_is_an_element_of_its_own(self):
        # Drawn only as a larger bus, nothing said where the system connects
        # to its neighbour — the first thing anyone looks for.
        network = nm.from_payload(payload())
        externals = [n for n in network.nodes.values() if n.kind == "external"]
        assert len(externals) == 1
        # Negative index, so it can never collide with a real bus.
        assert externals[0].index < 0
        assert any(e.kind == "ext_grid" for e in network.edges)

    def test_duplicate_bus_names_are_reported(self):
        data = payload()
        data["buses"][1]["name"] = "A"
        network = nm.from_payload(data)
        assert network.duplicate_names == {"A": 2}
        assert any("share" in note for note in network.notes)


class TestJoiningResults:
    def test_generation_joins_by_index_not_name(self):
        # The failure this prevents: a unit is named for its substation while a
        # bus is named with its voltage level, so "A" matched the 63 kV bus
        # when the unit sits on the 10 kV bus beneath it. Thirteen of sixteen
        # units joined that way on a real network and every one was wrong.
        network = nm.from_payload(payload())
        problems = nm.apply_conditions(network, {
            "generators": [{"name": "A", "bus": "A 10 kV", "bus_index": 3, "Pg_mw": 2.5}],
            "loads": [{"bus": "A 10 kV", "bus_index": 3, "P_mw": 1.0}],
        })
        assert problems == []
        assert network.nodes[3].gen_mw == 2.5
        assert network.nodes[0].gen_mw == 0.0

    def test_a_name_join_is_reported_as_unreliable(self):
        network = nm.from_payload(payload())
        problems = nm.apply_conditions(network, {
            "generators": [{"name": "A 10 kV", "Pg_mw": 2.5}], "loads": [],
        })
        assert network.nodes[3].gen_mw == 2.5
        assert any("located by name" in p for p in problems)

    def test_an_ambiguous_name_is_refused(self):
        data = payload()
        data["buses"][1]["name"] = "A"
        network = nm.from_payload(data)
        problems = nm.apply_conditions(network, {
            "generators": [{"name": "A", "Pg_mw": 2.5}], "loads": [],
        })
        assert all(n.gen_mw == 0.0 for n in network.nodes.values())
        assert any("share it" in p for p in problems)

    def test_unjoinable_results_are_never_silent(self):
        # An element whose result did not join renders grey, and grey must not
        # be mistaken for healthy.
        network = nm.from_payload(payload())
        problems = nm.apply_assessment(network, {
            "all_voltages": [{"bus_name": "Nowhere", "vm_pu": 1.02}],
            "all_line_loading": [{"line_name": "Nowhere-Else", "loading_percent": 12.0}],
        })
        assert len(problems) == 2

    def test_voltages_and_loadings_land_on_the_right_elements(self):
        network = nm.from_payload(payload())
        assert nm.apply_assessment(network, {
            "all_voltages": [{"bus_name": "A 10 kV", "vm_pu": 1.061}],
            "all_line_loading": [{"line_name": "A-B", "loading_percent": 42.0}],
        }) == []
        assert network.nodes[3].vm_pu == pytest.approx(1.061)
        assert next(e for e in network.edges if e.name == "A-B").loading_pct == 42.0


class TestOperatingPoint:
    def test_one_agreed_timestamp_is_returned(self):
        assert nm.operating_point_of([
            ("run_rsa", {"timestamp": "2022-01-02 21:45:00"}),
            ("attr", {"timestamp": "2022-01-02 21:45:00"}),
        ]) == "2022-01-02 21:45:00"

    def test_a_turn_spanning_two_points_names_neither(self):
        # No single timestamp describes it, so the view is left where it was
        # rather than snapped to whichever tool ran last.
        assert nm.operating_point_of([
            ("run_rsa", {"timestamp": "2022-01-02 21:45:00"}),
            ("run_rsa", {"timestamp": "2022-01-03 21:45:00"}),
        ]) is None

    def test_results_without_a_timestamp_are_ignored(self):
        assert nm.operating_point_of([("get_grid", {"name": "x"})]) is None


class TestLayout:
    def test_every_connected_bus_is_placed(self):
        network = nm.from_payload(payload())
        positions = nm.compute_layout(network)
        assert set(positions) == set(network.nodes)

    def test_isolated_buses_are_left_out_unless_asked_for(self):
        data = payload()
        data["buses"].append({"index": 9, "name": "Spare", "vn_kv": 63.0, "in_service": False})
        network = nm.from_payload(data)
        assert 9 not in nm.compute_layout(network)
        assert 9 in nm.compute_layout(network, include_isolated=True)

    def test_nothing_overlaps(self):
        network = nm.from_payload(payload())
        positions = nm.compute_layout(network)
        assert nm.count_overlaps(network, positions) == 0

    def test_overlaps_are_resolved_rather_than_inherited(self):
        # A layout that knows only the graph puts a substation and its own
        # transformer bus in the same place.
        network = nm.from_payload(payload())
        loose = nm.compute_layout(network, separate=False)
        tight = nm.compute_layout(network, separate=True)
        assert nm.count_overlaps(network, tight) == 0
        assert nm.count_overlaps(network, tight) <= nm.count_overlaps(network, loose)

    def test_a_dense_network_shrinks_its_markers(self):
        # The only lever the geometry leaves: spreading the layout out changes
        # nothing, because the figure fits itself to the data and the markers
        # grow with it.
        # Meshed rather than a chain: a bus with exactly two lines is a cable
        # joint and drawn small, so six hundred of those genuinely do fit. It
        # takes six hundred *substations* to run out of room.
        buses = [{"index": i, "name": f"B{i}", "vn_kv": 110.0, "in_service": True}
                 for i in range(600)]
        branches = []
        for i in range(600):
            for step in (1, 2):
                branches.append({"index": len(branches), "name": f"L{i}-{step}",
                                 "from_bus": i, "to_bus": (i + step) % 600,
                                 "kind": "line", "in_service": True})
        network = nm.from_payload(payload(buses=buses, branches=branches, externals=[]))
        assert all(n.kind == "substation" for n in network.nodes.values())
        nm.compute_layout(network)
        assert network.marker_scale < 1.0

    def test_a_radial_network_still_draws(self):
        # Peeling finds no core in a tree, so the whole graph is laid out at
        # once rather than a two-bus remnant.
        buses = [{"index": i, "name": f"B{i}", "vn_kv": 20.0, "in_service": True}
                 for i in range(8)]
        branches = [{"index": i, "name": f"L{i}", "from_bus": 0, "to_bus": i,
                     "kind": "line", "in_service": True} for i in range(1, 8)]
        network = nm.from_payload(payload(buses=buses, branches=branches, externals=[]))
        positions = nm.compute_layout(network)
        assert len(positions) == 8
        assert nm.count_overlaps(network, positions) == 0

    def test_positions_are_deterministic(self):
        # Two identical runs must place buses identically, or the picture
        # changes under the reader between reruns.
        first = nm.from_payload(payload())
        second = nm.from_payload(payload())
        a, b = nm.compute_layout(first), nm.compute_layout(second)
        for index in a:
            assert a[index] == pytest.approx(b[index], abs=1e-9)

    def test_the_marker_pad_shrinks_with_the_marker(self):
        # It did not, and shrinking markers then barely reduced the space they
        # needed, so a dense network stayed dense however small its dots were.
        network = nm.from_payload(payload())
        positions = nm.compute_layout(network)
        full = nm.marker_radii(network, positions, 1.0)
        half = nm.marker_radii(network, positions, 0.5)
        for index in full:
            assert half[index] == pytest.approx(full[index] / 2)

    def test_an_empty_network_does_not_raise(self):
        network = nm.from_payload(payload(buses=[], branches=[], externals=[]))
        assert nm.compute_layout(network) == {}


class TestLargeNetworksDegradeLoudly:
    def test_a_layout_too_large_for_kamada_says_so(self):
        import networkx as nx

        graph = nx.random_regular_graph(3, nm._KAMADA_LIMIT + 50, seed=3)
        notes: list[str] = []
        positions = nm._lay_out(graph, "kamada", seed=7, notes=notes)
        assert len(positions) == graph.number_of_nodes()
        assert any("kamada" in note for note in notes)

    def test_coincident_buses_separate_the_same_way_every_time(self):
        network = nm.from_payload(payload())
        positions = {index: (0.0, 0.0) for index in network.nodes}
        again = dict(positions)
        nm._separate(network, positions)
        nm._separate(network, again)
        for index in positions:
            assert positions[index] == pytest.approx(again[index], abs=1e-12)
        assert all(math.isfinite(v) for point in positions.values() for v in point)
