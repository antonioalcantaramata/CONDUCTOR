"""
How each generating unit is modelled — the operator's choice, at any time.

On load, generators with active power became fixed-power-factor units (Q = 0)
so the optimiser could dispatch them, and nothing could change that afterwards.
A real synchronous machine is dispatched *and* holds its voltage; the engine
already dispatched voltage-controlled units, so the choice is now per unit and
changeable: control (voltage or a fixed power factor), set-point, reactive
limits, and whether the OPF may move its output.
"""

import copy
import json
import pathlib
import shutil

import pandapower as pp
import pytest

import generators as gm
import network_loader as nl

BACKEND = pathlib.Path(__file__).resolve().parents[1] / "backend"


@pytest.fixture
def loaded():
    """The bundled network as the backend loads it."""
    profile = nl.load_profile("pglib_case14", backend_dir=str(BACKEND))
    net = nl.load_network(profile, backend_dir=str(BACKEND))
    net, _, _ = nl.gen_to_sgen(net)
    return net


class TestTheModel:
    def test_as_loaded(self, loaded):
        units = {u["unit"]: u for u in gm.describe(loaded)}
        g1 = units["Gen_bus1"]
        assert (g1["control"], g1["pf"], g1["dispatchable"], g1["condenser"]) == ("power_factor", 1.0, True, False)
        assert (g1["vm_pu"], g1["q_min_mvar"], g1["q_max_mvar"]) == (1.0, -30.0, 30.0)   # kept on conversion
        cond = units["Gen_bus2"]
        assert (cond["control"], cond["dispatchable"], cond["condenser"]) == ("voltage", False, True)

    def test_voltage_control_and_back_from_the_network_as_loaded(self, loaded):
        model = gm.merge({}, gm.normalise(loaded, [{"unit": "Gen_bus1", "control": "voltage", "vm_pu": 1.02}]))
        net = gm.apply(loaded, model)
        row = net.gen[net.gen.name == "Gen_bus1"].iloc[0]
        assert (row.vm_pu, row.min_q_mvar, row.max_q_mvar, int(row.bus)) == (1.02, -30.0, 30.0, 1)
        assert "Gen_bus1" not in set(net.sgen.name)
        pp.runpp(net)
        assert net.res_bus.vm_pu[1] == pytest.approx(1.02)                  # it holds its bus
        back = gm.merge(model, gm.normalise(loaded, [{"unit": "Gen_bus1", "control": "power_factor"}], model))
        assert back == {"Gen_bus1": {"vm_pu": 1.02}}                          # its set-point is remembered
        again = gm.apply(loaded, back)
        assert "Gen_bus1" in set(again.sgen.name) and len(again.gen) == len(loaded.gen)
        assert gm.apply(loaded, {}).gen.equals(loaded.gen)                    # nothing changed: as loaded

    def test_the_case_cost_follows_the_unit_to_its_table(self, loaded):
        import costs as uc
        case_cost = {r["unit"]: r for r in uc.defaults(loaded)}["Gen_bus1"]
        assert case_cost["source"]["cost_eur_per_mwh"] == "case"
        model = gm.merge({}, gm.normalise(loaded, [{"unit": "Gen_bus1", "control": "voltage", "vm_pu": 1.02}]))
        there = gm.apply(loaded, model)
        back = gm.apply(there, gm.normalise(there, [{"unit": "Gen_bus1", "control": "power_factor"}]))
        for net in (there, back):
            row = {r["unit"]: r for r in uc.defaults(net)}["Gen_bus1"]
            assert (row["cost_eur_per_mwh"], row["source"]["cost_eur_per_mwh"]) == \
                   (case_cost["cost_eur_per_mwh"], "case")

    def test_a_fixed_power_factor_injecting_or_absorbing(self, loaded):
        for absorbing, sign in ((False, 1), (True, -1)):
            model = gm.merge({}, gm.normalise(loaded, [{"unit": "Gen_bus1", "pf": 0.9, "absorbing": absorbing}]))
            net = gm.apply(loaded, model)
            i = net.sgen.index[net.sgen.name == "Gen_bus1"][0]
            net.sgen.at[i, "p_mw"] = 20.0
            gm.set_reactive(net)
            assert net.sgen.at[i, "q_mvar"] == pytest.approx(sign * 20.0 * 0.484322, abs=1e-4)

    def test_a_unit_back_to_as_loaded_leaves_the_model(self, loaded):
        model = gm.merge({}, gm.normalise(loaded, [{"unit": "Gen_bus1", "dispatchable": False}]))
        assert model == {"Gen_bus1": {"dispatchable": False}}
        assert gm.merge(model, gm.normalise(loaded, [{"unit": "Gen_bus1", "dispatchable": True}], model)) == {}

    @pytest.mark.parametrize("change, fragment", [
        ({"unit": "Gen_bus2", "control": "power_factor"}, "synchronous condenser"),
        ({"unit": "Gen_bus2", "dispatchable": True}, "no active power to dispatch"),
        ({"unit": "Nope"}, "there is no unit"),
        ({"unit": "Gen_bus1", "pf": 0.3}, "pf must be between"),
        ({"unit": "Gen_bus1", "vm_pu": 1.5}, "vm_pu must be between"),
        ({"unit": "Gen_bus1", "q_min_mvar": 10, "q_max_mvar": 5}, "is above"),
        ({"unit": "Gen_bus1", "control": "droop"}, "control must be"),
        ({"unit": "Gen_bus1", "dispatchable": "no"}, "true or false"),
    ])
    def test_impossible_models_are_refused(self, loaded, change, fragment):
        with pytest.raises(gm.ModelError, match=fragment):
            gm.normalise(loaded, [change])

    def test_one_change_per_unit_and_limits_needed_for_voltage_control(self, loaded):
        with pytest.raises(gm.ModelError, match="changed twice"):
            gm.normalise(loaded, [{"unit": "Gen_bus1"}, {"unit": "gen_bus1"}])
        bare = copy.deepcopy(loaded)
        bare.sgen.loc[bare.sgen.name == "Gen_bus1", ["orig_min_q_mvar", "orig_max_q_mvar"]] = float("nan")
        with pytest.raises(gm.ModelError, match="needs reactive limits"):
            gm.normalise(bare, [{"unit": "Gen_bus1", "control": "voltage"}])
        assert gm.normalise(bare, [{"unit": "Gen_bus1", "control": "voltage", "q_min_mvar": -5,
                                    "q_max_mvar": 5}])["Gen_bus1"]["q_max_mvar"] == 5.0

    def test_a_measured_substation_network_cannot_be_modelled_here(self):
        import pandapower.networks as pn
        net = pn.case14()
        net.gen = net.gen.iloc[0:0]
        assert gm.editable(net)[0] is False
        with pytest.raises(gm.ModelError, match="rebuilt from the measurements"):
            gm.normalise(net, [{"unit": "x"}])


@pytest.fixture
def client(backend_client):
    yield backend_client
    backend_client.post("/api/generators", json={"reset": True})
    import main_backend as mb
    mb._discard_proposals()


class TestThroughTheBackend:
    def test_the_model_changes_every_later_study(self, client):
        units = {u["unit"]: u for u in client.get("/api/generators").json()["units"]}
        assert units["Gen_bus1"]["control"] == "power_factor"
        out = client.post("/api/generators", json={"changes": [
            {"unit": "Gen_bus1", "control": "voltage", "vm_pu": 1.02}]}).json()
        assert out["changed"] == {"Gen_bus1": {"control": "voltage", "vm_pu": 1.02}}
        rsa = client.post("/api/grid/rsa", json={}).json()
        held = {v["unit"]: v for v in rsa["voltage_controlled_buses"]}
        assert held["Gen_bus1"]["setpoint_pu"] == pytest.approx(1.02)
        out = client.post("/api/generators", json={"changes": [
            {"unit": "Gen_bus1", "control": "power_factor", "pf": 0.9}]}).json()
        snap = client.post("/api/network/snapshot", json={}).json()
        g1 = next(g for g in snap["generators"] if g["name"] == "Gen_bus1")
        assert g1["Qg_mvar"] == pytest.approx(g1["Pg_mw"] * 0.484322, abs=1e-3)
        nominal = client.post("/api/network/snapshot", json={"data_source": "nominal"}).json()
        assert nominal["totals"]["total_generation_mw"] == pytest.approx(29.5, abs=1e-3)   # by name, any table

    def test_voltage_control_at_1_02_through_the_tool(self, client, monkeypatch):
        """Through the tool to the backend, with every slot filled in: only the
        mode and the set-point change, not the limits or dispatchability."""
        from llm_agent.agent import tools
        monkeypatch.setattr(tools, "_post", lambda path, body: client.post(path, json=body).json())
        before = {u["unit"]: u for u in client.get("/api/generators").json()["units"]}["Gen_bus1"]
        filled = {"control": "voltage", "value": 0, "on": False}
        out = tools.set_generator_model(changes=[
            {"unit": "Gen_bus1", "setting": "control", **filled},
            {"unit": "Gen_bus1", "setting": "vm_pu", **filled, "value": 1.02}])
        assert out["changed"] == {"Gen_bus1": {"control": "voltage", "vm_pu": 1.02}}
        after = {u["unit"]: u for u in client.get("/api/generators").json()["units"]}["Gen_bus1"]
        assert {k: after[k] for k in ("q_min_mvar", "q_max_mvar", "dispatchable")} == \
               {k: before[k] for k in ("q_min_mvar", "q_max_mvar", "dispatchable")}

    def test_refusals_say_why(self, client):
        r = client.post("/api/generators", json={"changes": [{"unit": "Gen_bus2", "control": "power_factor"}]})
        assert r.status_code == 400 and "synchronous condenser" in r.json()["detail"]
        assert client.post("/api/generators", json={}).status_code == 400

    def test_proposals_are_rechecked(self, client):
        client.post("/api/proposals/edit", json={"proposal": "dc", "edits": [
            {"op": "add_load", "name": "DC", "bus": "Bus 9", "p_mw": 10}]})
        client.post("/api/proposals/edit", json={"proposal": "off", "edits": [
            {"op": "set_in_service", "element_type": "sgen", "element": "Gen_bus1", "in_service": False}]})
        out = client.post("/api/generators", json={"changes": [{"unit": "Gen_bus1", "control": "voltage"}]}).json()
        assert [d.split(":")[0] for d in out["proposals_dropped"]] == ["off"]      # its sgen is now a gen
        ids = [p["id"] for p in client.get("/api/proposals").json()["proposals"]]
        assert ids == ["dc"]
        assert client.post("/api/proposals/compare", json={"proposal": "dc"}).status_code == 200

    @pytest.mark.skipif(shutil.which("ipopt") is None, reason="IPOPT not installed")
    def test_the_opf_keeps_a_voltage_unit_in_its_own_limits_and_a_fixed_unit_still(self, client):
        def gen1():
            out = client.post("/api/flexibility/optimize", json={"timestamp": "2000-01-03 14:15"}).json()
            assert out["feasible"], out
            return next(a for a in out["activated_resources"] if a.get("element") == "Gen_bus1")
        assert gen1()["Qg_new"] < 5.0 - 0.1          # as loaded: its power-factor cone binds below 5 Mvar
        client.post("/api/generators", json={"changes": [
            {"unit": "Gen_bus1", "control": "voltage", "q_min_mvar": -5, "q_max_mvar": 5}]})
        g1 = gen1()
        assert g1["Qg_new"] == pytest.approx(5.0, abs=1e-3)              # its own limit now
        assert g1["Pg_new"] != pytest.approx(g1["Pg_base"], abs=1e-3)    # and still dispatched
        client.post("/api/generators", json={"changes": [{"unit": "Gen_bus1", "dispatchable": False}]})
        g1 = gen1()
        assert g1["Pg_new"] == pytest.approx(g1["Pg_base"], abs=1e-3)    # kept where it was


def test_the_engine_builds_the_limits_and_pins_from_the_model(loaded):
    import flex_engine as fe
    model = gm.merge({}, gm.normalise(loaded, [{"unit": "Gen_bus1", "control": "voltage", "dispatchable": False}]))
    net = gm.apply(loaded, model)
    assert fe._fixed_output_units(net) == {"Gen_bus1"}
    assert fe._fixed_output_units(loaded) == set()


def test_the_model_survives_a_restart_and_the_proposals_after_it(tmp_path, monkeypatch):
    pytest.importorskip("fastapi.testclient")
    import main_backend as mb
    from fastapi.testclient import TestClient
    monkeypatch.setenv("GRID_PROFILE", "pglib_case14")
    monkeypatch.setattr(mb, "SYSTEMS_DIR", str(tmp_path / "systems"))
    monkeypatch.setattr(mb, "DATA_FILES_DIR", str(tmp_path / "data_files"))
    for name in ("_LAST_UPLOAD_SENTINEL", "_LAST_TIMESERIES_SENTINEL", "_LAST_FORECAST_SENTINEL"):
        monkeypatch.setattr(mb, name, str(tmp_path / f"{name}.json"))
    monkeypatch.setattr(mb, "_PROPOSALS_FILE", str(tmp_path / "systems" / "proposals.json"))
    store = tmp_path / "systems" / "assumptions.json"
    monkeypatch.setattr(mb, "_ASSUMPTIONS_FILE", str(store))
    monkeypatch.setattr(mb, "_SCENARIO_FILE", str(tmp_path / "scenario.json"))
    saved = dict(mb.app_data)
    try:
        with TestClient(mb.app) as c:
            c.post("/api/generators", json={"changes": [{"unit": "Gen_bus1", "control": "voltage"}]})
            c.post("/api/proposals/edit", json={"proposal": "dc", "edits": [
                {"op": "add_load", "name": "DC", "bus": "Bus 9", "p_mw": 10}]})
        (entry,) = json.loads(store.read_text())["by_fingerprint"].values()
        assert entry["generator_model"] == {"Gen_bus1": {"control": "voltage"}}
        mb.app_data["proposals"] = {}
        with TestClient(mb.app) as c:
            assert c.get("/api/generators").json()["changed"] == {"Gen_bus1": {"control": "voltage"}}
            assert [p["id"] for p in c.get("/api/proposals").json()["proposals"]] == ["dc"]
            c.post("/api/generators", json={"reset": True})
        assert not store.exists()
    finally:
        mb.app_data.clear()
        mb.app_data.update(saved)


def test_the_tables_show_each_kind_of_unit_in_plain_words_and_the_form_its_own_fields():
    from llm_agent.agent import generator_table as gt
    units = [{"unit": "Gen_bus1", "bus": 1, "bus_name": "Bus_1", "p_max_mw": 59.0, "control": "power_factor",
              "vm_pu": 1.0, "pf": 0.95, "absorbing": False, "q_min_mvar": -30.0, "q_max_mvar": 30.0,
              "dispatchable": True, "condenser": False},
             {"unit": "Gen_bus2", "bus": 2, "bus_name": "Bus_2", "p_max_mw": 0.0, "control": "voltage",
              "vm_pu": 0.942, "pf": None, "absorbing": False, "q_min_mvar": 0.0, "q_max_mvar": 40.0,
              "dispatchable": False, "condenser": True},
             {"unit": "G9", "bus": 9, "p_max_mw": 20.0, "control": "voltage", "vm_pu": 1.01, "pf": None,
              "absorbing": False, "q_min_mvar": -5.0, "q_max_mvar": 5.0, "dispatchable": True,
              "condenser": False}]
    tables = gt.rows(units, {"Gen_bus1": {"pf": 0.95}})
    assert list(tables) == ["voltage", "power_factor", "condenser"]
    assert tables["power_factor"][0] == {"unit": "Gen_bus1", "changed": True,
                                         "cells": ["Gen_bus1", "Bus_1", "59 MW", "0.95 injecting", "yes"]}
    assert tables["condenser"][0]["cells"] == ["Gen_bus2", "Bus_2", "0.942 p.u.", "0 to 40 Mvar"]
    assert tables["voltage"][0]["cells"] == ["G9", "9", "20 MW", "1.010 p.u.", "-5 to 5 Mvar", "yes"]

    assert gt.form_fields(units[1], "voltage") == ["vm_pu", "q_min_mvar", "q_max_mvar"]     # a condenser
    assert gt.form_fields(units[0], "voltage")[0] == "control"
    assert gt.form_changes(units[0], {"control": "voltage", "vm_pu": 1.0, "q_min_mvar": -30.0,
                                      "q_max_mvar": 30.0, "dispatchable": True}) == \
        {"unit": "Gen_bus1", "control": "voltage"}
    assert gt.form_changes(units[1], {"q_max_mvar": 35}) == {"unit": "Gen_bus2", "q_max_mvar": 35.0}
    assert gt.form_changes(units[2], {"control": "voltage", "vm_pu": 1.01}) is None


def test_the_tool_reads_and_changes_the_model(monkeypatch):
    from llm_agent.agent import tool_schemas as ts
    from llm_agent.agent import tools
    calls = []
    monkeypatch.setattr(tools, "_get", lambda path: calls.append(("get", path)) or {"units": []})
    monkeypatch.setattr(tools, "_post", lambda path, body: calls.append(("post", path, body)) or {"ok": True})
    tools.set_generator_model()
    tools.set_generator_model(changes=[{"unit": "Gen_bus1", "setting": "control", "control": "voltage"}])
    tools.set_generator_model(reset=True)
    assert calls == [("get", "/api/generators"),
                     ("post", "/api/generators", {"changes": [{"unit": "Gen_bus1", "control": "voltage"}],
                                                  "reset": False}),
                     ("post", "/api/generators", {"changes": [], "reset": True})]
    assert "error" in tools.set_generator_model(changes=[{"unit": "x"}], reset=True)
    decl = {fd.name: fd for t in ts.TOOLS for fd in t.function_declarations}
    props = decl["set_generator_model"].parameters.properties
    assert set(props) == {"changes", "reset"} and "set_generator_model" in ts.TOOL_DISPATCH
    item = props["changes"].items
    assert set(item.properties) == {"unit", "setting", "control", "value", "on"}
    assert set(item.properties["setting"].enum) == set(gm.FIELDS)          # every setting the backend takes


class TestTheTool:
    """The model changes one setting per item, its value in the slot that
    setting uses; only that slot is read."""

    @pytest.fixture
    def posted(self, monkeypatch):
        from llm_agent.agent import tools
        sent = []
        monkeypatch.setattr(tools, "_post", lambda path, body: sent.append(body) or {"ok": True})
        return tools, sent

    def test_a_filled_in_item_changes_only_its_setting(self, posted):
        # Seen live with the cost tool: a model that fills in every property.
        tools, sent = posted
        filled = {"control": "voltage", "value": 0, "on": False}
        tools.set_generator_model(changes=[{"unit": "Gen_bus1", "setting": "control", **filled,
                                            "control": "voltage"},
                                           {"unit": "Gen_bus1", "setting": "vm_pu", **filled, "value": 1.02},
                                           {"unit": "Gen_bus3", "setting": "dispatchable", **filled}])
        assert sent[0]["changes"] == [{"unit": "Gen_bus1", "control": "voltage", "vm_pu": 1.02},
                                      {"unit": "Gen_bus3", "dispatchable": False}]

    @pytest.mark.parametrize("item, says", [
        ({"unit": "Gen_bus1", "control": "voltage"}, "give unit, setting"),     # the old one-object-per-unit shape
        ({"unit": "Gen_bus1", "setting": "speed", "value": 1}, "setting is one of"),
        ({"unit": "Gen_bus1", "setting": "control", "control": "droop"}, "give control as one of"),
        ({"unit": "Gen_bus1", "setting": "vm_pu", "on": True}, "give the number in value"),
        ({"unit": "Gen_bus1", "setting": "dispatchable", "value": 0}, "give true or false in on"),
    ])
    def test_an_incomplete_item_is_refused_before_the_backend(self, posted, item, says):
        tools, sent = posted
        assert says in tools.set_generator_model(changes=[item])["error"] and sent == []

    def test_the_same_setting_twice_is_refused(self, posted):
        tools, sent = posted
        twice = [{"unit": "Gen_bus1", "setting": "vm_pu", "value": v} for v in (1.0, 1.02)]
        assert "given twice" in tools.set_generator_model(changes=twice)["error"] and sent == []


def test_fixed_q_holds_its_reactive_output_whatever_its_power(loaded):
    model = gm.merge({}, gm.normalise(loaded, [{"unit": "Gen_bus1", "control": "fixed_q", "q_mvar": -8}]))
    net = gm.apply(loaded, model)
    i = net.sgen.index[net.sgen.name == "Gen_bus1"][0]
    for p in (5.0, 40.0):
        net.sgen.at[i, "p_mw"] = p
        gm.set_reactive(net)
        assert net.sgen.at[i, "q_mvar"] == pytest.approx(-8.0)
    unit = next(u for u in gm.describe(net) if u["unit"] == "Gen_bus1")
    assert (unit["control"], unit["q_mvar"]) == ("fixed_q", -8.0)
    with pytest.raises(gm.ModelError, match="needs the reactive output"):
        gm.normalise(loaded, [{"unit": "Gen_bus1", "control": "fixed_q"}])
    from llm_agent.agent import generator_table as gt
    tables = gt.rows([{**unit, "bus_name": "Bus_1"}], model)
    assert tables["fixed_q"][0]["cells"] == ["Gen_bus1", "Bus_1", "59 MW", "8 Mvar absorbed", "yes"]
    assert gt.form_fields(unit, "fixed_q") == ["control", "q_mvar", "dispatchable"]
