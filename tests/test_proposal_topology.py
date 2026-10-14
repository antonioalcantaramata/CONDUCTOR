"""
Proposal edits that change the topology or the measured elements, and
proposals built from and compared with other proposals:

* `add_line` / `add_trafo` — a new branch between two existing buses;
* `move_element` — an element (or a line end) reconnected to another bus; a
  measured load or unit keeps its own measurements;
* `scale` — existing loads or units × a factor, keeping their profile;
* `include_proposal` — "A + B" from stored proposals;
* `compare_networks` with `against` (B vs A) and `proposals` (several at once).
"""

import pandapower.networks as pn
import pytest

pytest.importorskip("fastapi.testclient")

import main_backend as mb  # noqa: E402
import sandbox as sb  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

# case14: buses 0–4 at 135 kV, bus 5 and 8–13 at 0.208 kV, 6 at 14 kV, 7 at 12 kV.


@pytest.fixture
def net():
    return pn.case14()


def _tick(net, p_mw: dict):
    """What `_assign_tick` does around a tick's assignment, with `p_mw`
    standing in for the measured values (loads not in it keep theirs)."""
    sb.before_tick(net)
    for idx, value in p_mw.items():
        net.load.at[idx, "p_mw"] = value
    sb.apply_overrides(net)


class TestAddBranches:
    def test_a_line_between_two_buses(self, net):
        out, edits = sb.materialise(net, [{"op": "add_line", "from_bus": 0, "to_bus": 3, "length_km": 10,
                                            "line_like": 0, "parallel": 2}])
        new = out.line.index[-1]
        assert len(out.line) == len(net.line) + 1 and out.line.at[new, "parallel"] == 2
        assert (out.line.at[new, "from_bus"], out.line.at[new, "to_bus"]) == (0, 3)
        assert sb.describe(out, edits[0]).startswith("Add line")
        assert sb.summary(out, edits)["new_lines"] == 1

    def test_a_transformer_between_two_buses_orders_its_sides(self, net):
        out, edits = sb.materialise(net, [{"op": "add_trafo", "hv_bus": 6, "lv_bus": 3, "sn_mva": 50,
                                            "vk_percent": 12, "vkr_percent": 0.4}])
        new = out.trafo.index[-1]
        assert (out.trafo.at[new, "hv_bus"], out.trafo.at[new, "lv_bus"]) == (3, 6)   # 135 kV over 14 kV
        assert out.trafo.at[new, "vn_hv_kv"] == 135 and out.trafo.at[new, "vn_lv_kv"] == 14
        assert sb.summary(out, edits)["new_trafos"] == 1

    @pytest.mark.parametrize("edit, reason", [
        ({"op": "add_line", "from_bus": 0, "to_bus": 6, "length_km": 5, "line_like": 0}, "Use add_trafo"),
        ({"op": "add_line", "from_bus": 0, "to_bus": 0, "length_km": 5, "line_like": 0}, "two different"),
        ({"op": "add_line", "from_bus": 0, "to_bus": 3, "length_km": 5}, "exactly one of"),
        ({"op": "add_trafo", "hv_bus": 0, "lv_bus": 3, "trafo_like": 0}, "use add_line"),
        ({"op": "add_trafo", "hv_bus": 3, "lv_bus": 6}, "give trafo_like"),
        ({"op": "add_trafo", "hv_bus": 3, "lv_bus": 6, "sn_mva": 50, "vk_percent": 5, "vkr_percent": 6},
         "must be below vk_percent"),
    ])
    def test_refusals_say_why(self, net, edit, reason):
        with pytest.raises(sb.EditError, match=reason):
            sb.materialise(net, [edit])


class TestMove:
    def test_a_measured_load_keeps_its_own_measurements(self, net):
        out, edits = sb.materialise(net, [{"op": "move_element", "element_type": "load", "element": 2,
                                            "to_bus": 4}])
        was = int(net.load.at[2, "bus"])
        assert out.load.at[2, "bus"] == 4 and "keeping its own measurements" in sb.describe(out, edits[0])
        sb.before_tick(out)
        assert out.load.at[2, "bus"] == was           # measured where it was measured
        out.load.at[2, "p_mw"] = 33.0
        sb.apply_overrides(out)
        assert out.load.at[2, "bus"] == 4 and out.load.at[2, "p_mw"] == 33.0

    def test_a_line_end(self, net):
        out, _ = sb.materialise(net, [{"op": "move_element", "element_type": "line", "element": 0,
                                       "end": "to", "to_bus": 2}])
        assert (out.line.at[0, "from_bus"], out.line.at[0, "to_bus"]) == (0, 2)

    def test_the_only_external_grid_needs_no_name(self, net):
        out, _ = sb.materialise(net, [{"op": "move_element", "element_type": "ext_grid", "to_bus": 3}])
        assert out.ext_grid.bus.iloc[0] == 3

    @pytest.mark.parametrize("edit, reason", [
        ({"element_type": "line", "element": 0, "to_bus": 2}, "which end"),
        ({"element_type": "line", "element": 0, "end": "to", "to_bus": 6}, "one voltage"),
        ({"element_type": "line", "element": 0, "end": "to", "to_bus": 0}, "start and end"),
        ({"element_type": "load", "element": 2, "to_bus": 99}, "no bus"),
        ({"element_type": "trafo", "element": 0, "to_bus": 2}, "element_type must be"),
        ({"element_type": "gen", "element": 0, "to_bus": 2}, "held at"),
    ])
    def test_refusals_say_why(self, net, edit, reason):
        with pytest.raises(sb.EditError, match=reason):
            sb.materialise(net, [{"op": "move_element", **edit}])


class TestScale:
    def test_by_bus_and_all(self, net):
        out, edits = sb.materialise(net, [{"op": "scale", "element_type": "load", "bus": 5, "factor": 2},
                                          {"op": "scale", "element_type": "load", "all": True, "factor": 1.5}])
        at5 = net.load.index[net.load.bus == 5][0]
        assert out.load.at[at5, "p_mw"] == pytest.approx(net.load.at[at5, "p_mw"] * 3)
        assert out.load.at[0, "p_mw"] == pytest.approx(net.load.at[0, "p_mw"] * 1.5)
        assert sb.summary(out, edits)["scaled_elements"] == 1 + len(net.load)

    def test_a_tick_is_scaled_once_whether_or_not_it_rewrites_the_value(self, net):
        out, _ = sb.materialise(net, [{"op": "scale", "element_type": "load", "element": 0, "factor": 2},
                                      {"op": "scale", "element_type": "load", "element": 1, "factor": 2}])
        base1 = float(net.load.at[1, "p_mw"])
        for value in (10.0, 20.0, 20.0, 40.0):        # the scaled value measured again must not confuse it
            _tick(out, {0: value})
            assert out.load.at[0, "p_mw"] == pytest.approx(2 * value)
            assert out.load.at[1, "p_mw"] == pytest.approx(2 * base1)    # not rewritten: never compounds

    def test_a_held_value_is_not_scaled(self, net):
        out, _ = sb.materialise(net, [
            {"op": "set_parameter", "element_type": "load", "element": 0, "values": {"p_mw": 7}},
            {"op": "scale", "element_type": "load", "element": 0, "factor": 3}])
        _tick(out, {0: 50.0})
        assert out.load.at[0, "p_mw"] == 7

    def test_a_stored_proposal_replays(self, net):
        _, edits = sb.materialise(net, [{"op": "scale", "element_type": "load", "bus": 5, "factor": 2}])
        again, replayed = sb.materialise(net, edits)
        assert replayed == edits

    @pytest.mark.parametrize("edit, reason", [
        ({"element_type": "load", "element": 0, "factor": 11}, "at most"),
        ({"element_type": "load", "element": 0, "factor": -1}, "at least"),
        ({"element_type": "load", "factor": 2}, "name the element"),
        ({"element_type": "line", "element": 0, "factor": 2}, "element_type must be"),
        ({"element_type": "sgen", "bus": 0, "factor": 2}, "no sgen"),
    ])
    def test_refusals_say_why(self, net, edit, reason):
        with pytest.raises(sb.EditError, match=reason):
            sb.materialise(net, [{"op": "scale", **edit}])

    def test_an_added_load_is_sized_where_it_was_added(self, net):
        with pytest.raises(sb.EditError, match="added by this proposal"):
            sb.materialise(net, [{"op": "add_load", "name": "dc", "bus": 4, "p_mw": 5, "profile": "flat"},
                                 {"op": "scale", "element_type": "load", "element": len(net.load), "factor": 2}])


@pytest.fixture
def client(tmp_path, monkeypatch):
    systems, data = tmp_path / "systems", tmp_path / "data_files"
    systems.mkdir()
    data.mkdir()
    monkeypatch.setenv("GRID_PROFILE", "pglib_case14")
    monkeypatch.setattr(mb, "SYSTEMS_DIR", str(systems))
    monkeypatch.setattr(mb, "DATA_FILES_DIR", str(data))
    for name in ("_LAST_UPLOAD_SENTINEL", "_LAST_TIMESERIES_SENTINEL", "_LAST_FORECAST_SENTINEL",
                 "_PROPOSALS_FILE", "_ASSUMPTIONS_FILE", "_SCENARIO_FILE"):
        monkeypatch.setattr(mb, name, str(tmp_path / f"{name}.json"))
    saved = dict(mb.app_data)
    with TestClient(mb.app) as c:
        yield c
    mb.app_data.clear()
    mb.app_data.update(saved)


def _edit(client, name, *edits):
    r = client.post("/api/proposals/edit", json={"proposal": name, "edits": list(edits)})
    assert r.status_code == 200, r.text
    return r.json()


def _loads(client, ts, network=None):
    r = client.post("/api/network/snapshot", json={"timestamp": ts, **({"network": network} if network else {})})
    assert r.status_code == 200, r.text
    return r.json()["loads"]


TICKS = ("2000-01-01 00:00:00", "2000-01-03 12:00:00")


def test_scaled_loads_follow_their_measurements_at_every_tick(client):
    _edit(client, "growth", {"op": "scale", "element_type": "load", "all": True, "factor": 1.2})
    for ts in TICKS:
        base, grown = _loads(client, ts), _loads(client, ts, "growth")
        assert [g["P_mw"] for g in grown] == pytest.approx([1.2 * b["P_mw"] for b in base], abs=1e-3)


def test_a_moved_load_carries_its_own_measurements(client):
    _edit(client, "moved", {"op": "move_element", "element_type": "load", "element": 1, "to_bus": "Bus_4"})
    for ts in TICKS:
        base, moved = _loads(client, ts), _loads(client, ts, "moved")
        assert moved[1]["bus_index"] == 4 and moved[1]["bus_index"] != base[1]["bus_index"]
        assert [m["P_mw"] for m in moved] == pytest.approx([b["P_mw"] for b in base], abs=1e-3)


def test_a_new_line_and_transformer_run_in_a_study(client):
    _edit(client, "mesh", {"op": "add_line", "from_bus": "Bus_0", "to_bus": "Bus_3", "length_km": 1,
                           "line_like": 0})
    r = client.post("/api/proposals/compare", json={"proposal": "mesh", "study": "rsa",
                                                    "timestamp": TICKS[1]})
    assert r.status_code == 200, r.text


class TestCombineAndCompare:
    @pytest.fixture
    def two(self, client):
        _edit(client, "grid105", {"op": "set_parameter", "element_type": "ext_grid", "field": "vm_pu",
                                  "value": 1.05})
        _edit(client, "growth", {"op": "scale", "element_type": "load", "all": True, "factor": 1.3})
        return client

    def test_a_plus_b(self, two):
        both = _edit(two, "both", {"op": "include_proposal", "proposal": "grid105"},
                     {"op": "include_proposal", "proposal": "growth"})
        assert [e["op"] for e in both["edits"]] == ["set_parameter", "scale"]

    @pytest.mark.parametrize("target, reason", [("both", "itself"), ("nope", "no proposal")])
    def test_include_refusals(self, two, target, reason):
        r = two.post("/api/proposals/edit", json={"proposal": "both", "edits": [
            {"op": "include_proposal", "proposal": target}]})
        assert r.status_code == 400 and reason in r.text

    def test_one_proposal_against_another(self, two):
        r = two.post("/api/proposals/compare", json={"proposal": "growth", "against": "grid105",
                                                     "study": "rsa", "timestamp": TICKS[1]})
        assert r.status_code == 200, r.text
        out = r.json()
        assert out["network"]["compared_with"] == "grid105"
        assert out["base"]["total_load_mw"] < out["proposal"]["total_load_mw"]

    def test_several_proposals_at_one_tick(self, two):
        r = two.post("/api/proposals/compare", json={"proposals": ["grid105", "growth"], "study": "rsa"})
        assert r.status_code == 200, r.text
        out = r.json()
        assert [row["network"] for row in out["table"]] == ["base", "grid105", "growth"]
        assert {c["timestamp"] for c in out["comparisons"]} == {out["timestamp"]}
        assert out["table"][2]["total_load_mw"] > out["table"][0]["total_load_mw"]

    def test_several_n1(self, two):
        r = two.post("/api/proposals/compare", json={"proposals": ["grid105", "growth"], "study": "n1",
                                                     "timestamp": TICKS[1]})
        assert r.status_code == 200, r.text
        assert "system_n1_secure" in r.json()["table"][1]

    def test_against_must_exist(self, two):
        r = two.post("/api/proposals/compare", json={"proposal": "growth", "against": "nope"})
        assert r.status_code == 400 and "against" in r.text
