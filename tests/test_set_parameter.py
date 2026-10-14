"""
`set_parameter` proposal edits: any setting of any existing element — the
external grid's voltage, a generator set-point, a tap, a rating, a load's
demand (held at every tick instead of the measured value).

The case that prompted it: "run an RSA at 2000-01-05 15:30 on a proposal where
the external grid voltage is 1.05 p.u." had no edit to express it, and the
agent spent all its steps on workarounds.
"""

import pandapower.networks as pn
import pytest

pytest.importorskip("fastapi.testclient")

import main_backend as mb  # noqa: E402
import sandbox as sb  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402


class TestNormalise:
    @pytest.fixture
    def net(self):
        return pn.case14()

    def test_the_external_grid_voltage(self, net):
        out, edits = sb.materialise(net, [{"op": "set_parameter", "element_type": "ext_grid",
                                            "field": "vm_pu", "value": 1.05}])
        assert edits[0]["values"] == {"vm_pu": 1.05} and out.ext_grid.vm_pu.iloc[0] == 1.05
        assert net.ext_grid.vm_pu.iloc[0] != 1.05                        # the base is untouched
        assert sb.describe(out, edits[0]).startswith("Set external grid at")

    @pytest.mark.parametrize("label", ["external grid", "slack", "0", 0])
    def test_the_external_grid_by_any_label(self, net, label):
        _, edits = sb.materialise(net, [{"op": "set_parameter", "element_type": "ext_grid", "element": label,
                                         "values": {"vm_pu": 1.03}}])
        assert edits[0]["element"] == 0

    def test_several_fields_and_optional_limits(self, net):
        out, edits = sb.materialise(net, [{"op": "set_parameter", "element_type": "line", "element": 3,
                                            "values": {"max_i_ka": 2.0, "max_loading_percent": 80}}])
        assert out.line.at[3, "max_i_ka"] == 2.0 and out.line.at[3, "max_loading_percent"] == 80

    @pytest.mark.parametrize("edit, reason", [
        ({"element_type": "ext_grid", "field": "vm_pu", "value": 1.5}, "outside 0.8–1.2"),
        ({"element_type": "line", "element": 0, "field": "from_bus", "value": 3}, "how the line is connected"),
        ({"element_type": "line", "element": 0, "field": "colour", "value": 3}, "Editable:"),
        ({"element_type": "trafo", "element": 0, "field": "tap_pos", "value": 2}, "no tap changer"),
        ({"element_type": "line", "element": 0, "field": "max_i_ka", "value": -1}, "positive"),
        ({"element_type": "load", "element": 0, "field": "controllable", "value": 3}, "true or false"),
        ({"element_type": "bus", "element": "Bus_99", "field": "max_vm_pu", "value": 1.1}, "no bus"),
        ({"element_type": "load", "element": 0}, "give `field`"),
    ])
    def test_refusals_say_why(self, net, edit, reason):
        with pytest.raises(sb.EditError, match=reason):
            sb.materialise(net, [{"op": "set_parameter", **edit}])

    def test_a_measured_value_is_held_at_every_tick(self, net):
        out, edits = sb.materialise(net, [{"op": "set_parameter", "element_type": "load", "element": 2,
                                            "values": {"p_mw": 50}}])
        assert "held at every tick" in sb.describe(out, edits[0])
        out.load.at[2, "p_mw"] = 1.0                 # what a tick's measurements would assign
        sb.apply_overrides(out)
        assert out.load.at[2, "p_mw"] == 50

    def test_more_elements_can_be_switched(self, net):
        out, _ = sb.materialise(net, [{"op": "set_in_service", "element_type": "shunt", "element": 0,
                                       "in_service": False}])
        assert not out.shunt.in_service.iloc[0]


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


def test_the_session_request_works_end_to_end(client):
    r = client.post("/api/proposals/edit", json={"proposal": "grid_1050", "edits": [
        {"op": "set_parameter", "element_type": "ext_grid", "field": "vm_pu", "value": 1.05}]})
    assert r.status_code == 200, r.text
    base = client.post("/api/grid/rsa", json={"timestamp": "2000-01-05 15:30:00"}).json()
    prop = client.post("/api/grid/rsa", json={"timestamp": "2000-01-05 15:30:00", "network": "grid_1050"}).json()
    v_base = {v["bus_name"]: v["vm_pu"] for v in base["all_voltages"]}
    v_prop = {v["bus_name"]: v["vm_pu"] for v in prop["all_voltages"]}
    assert abs(v_prop["Bus_0"] - 1.05) < 1e-6 and abs(v_base["Bus_0"] - 1.02) < 1e-6
    assert v_prop["Bus_3"] > v_base["Bus_3"]                     # the raised grid voltage lifts the weak bus
    assert prop["network"]["id"] == "grid_1050" and "Set external grid" in prop["network"]["edits"][0]


def test_a_load_override_holds_across_ticks(client):
    r = client.post("/api/proposals/edit", json={"proposal": "big_load", "edits": [
        {"op": "set_parameter", "element_type": "load", "element": 1, "field": "p_mw", "value": 55.0}]})
    assert r.status_code == 200, r.text
    for ts in ("2000-01-01 00:00:00", "2000-01-03 12:00:00"):
        cond = client.post("/api/network/snapshot", json={"timestamp": ts, "network": "big_load"})
        assert cond.status_code == 200, cond.text
        loads = cond.json()["loads"]
        assert any(abs(entry["P_mw"] - 55.0) < 1e-6 for entry in loads), loads


def test_asking_the_generator_model_for_the_external_grid_points_to_set_parameter(client):
    r = client.post("/api/generators", json={"changes": [{"unit": "external grid", "vm_pu": 1.05}]})
    assert r.status_code == 400 and "set_parameter" in r.json()["detail"]
