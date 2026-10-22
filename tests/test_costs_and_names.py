"""
Unit costs (production and redispatch) and the operator's names for
elements — both kept per network like the generator model.

The rename's hazard: names are keys. Measurements are matched to buses by
name, unit models and costs are keyed by unit name, and saved proposals by a
fingerprint that read the names. A rename must change what results say and
nothing else.
"""

import json
import shutil

import pandapower.networks as pn
import pytest

pytest.importorskip("fastapi.testclient")

import costs as uc  # noqa: E402
import main_backend as mb  # noqa: E402
import network_loader as nl  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

TS = "2000-01-03 12:00:00"


class TestCaseCosts:
    def test_cost_rows_follow_converted_units(self):
        net = pn.case14()
        nl.gen_to_sgen(net)
        rows = {r["unit"]: r for r in uc.defaults(net)}
        # The generator at bus 1 (cp1 20 in case14's gencost) became an sgen and keeps its own cost.
        assert rows["Gen_bus1"]["cost_eur_per_mwh"] == 20.0 and rows["Gen_bus1"]["source"]["cost_eur_per_mwh"] == "case"
        assert all(r["cost_eur_per_mwh"] == 40.0 for k, r in rows.items() if k.startswith("Gen_bus") and k != "Gen_bus1")
        pc = net.poly_cost
        assert set(pc.loc[pc.et == "gen", "element"]) <= set(net.gen.index)

    def test_synthetic_rule_without_a_case_cost(self):
        net = pn.case14()
        net.poly_cost.drop(net.poly_cost.index, inplace=True)
        net.gen.loc[net.gen.bus == 2, "max_p_mw"] = 0.0            # a condenser, as in the PGLib case
        nl.gen_to_sgen(net)
        rows = {r["unit"]: r for r in uc.defaults(net)}
        unit = rows["Gen_bus1"]
        assert unit["cost_eur_per_mwh"] == uc.SYNTHETIC_UNIT and unit["source"]["cost_eur_per_mwh"] == "synthetic"
        assert unit["up_eur_per_mwh"] == pytest.approx(1.2 * 50) and unit["down_eur_per_mwh"] == pytest.approx(25)
        assert rows["External grid"]["cost_eur_per_mwh"] == uc.SYNTHETIC_IMPORT
        assert rows["Gen_bus2"]["up_eur_per_mwh"] is None                  # a condenser

    @pytest.mark.parametrize("change, reason", [
        ({"unit": "Nope", "up_eur_per_mwh": 1}, "no unit"),
        ({"unit": "Gen_bus1", "up_eur_per_mwh": -5}, "between"),
        ({"unit": "Gen_bus1", "colour": 3}, "unknown field"),
        ({"unit": "Gen_bus1"}, "at least one"),
    ])
    def test_refusals_say_why(self, change, reason):
        net = pn.case14()
        nl.gen_to_sgen(net)
        with pytest.raises(uc.CostError, match=reason):
            uc.normalise(net, [change])

    def test_reactive_prices_and_the_grid_band(self):
        net = pn.case14()
        nl.gen_to_sgen(net)
        rows = {r["unit"]: r for r in uc.defaults(net)}
        assert rows["Gen_bus1"]["q_eur_per_mvarh"] == uc.SYNTHETIC_Q
        grid = rows["External grid"]
        assert grid["q_eur_per_mvarh"] is None
        assert (grid["q_band_tan_phi"], grid["q_penalty_eur_per_mvarh"]) == (0.4, 10.0)
        model = uc.normalise(net, [{"unit": "Gen_bus1", "q_eur_per_mvarh": 5},
                                   {"unit": "External grid", "q_band_tan_phi": 0.33, "q_penalty_eur_per_mvarh": 25}])
        table = {r["unit"]: r for r in uc.table(net, model)}
        assert table["Gen_bus1"]["q_eur_per_mvarh"] == 5 and table["Gen_bus1"]["source"]["q_eur_per_mvarh"] == "operator"
        assert table["External grid"]["q_band_tan_phi"] == 0.33

    @pytest.mark.parametrize("change, reason", [
        ({"unit": "Gen_bus1", "q_band_tan_phi": 0.3}, "external grid only"),
        ({"unit": "External grid", "q_eur_per_mvarh": 3}, "units only"),
        ({"unit": "External grid", "q_band_tan_phi": 5}, "between"),
    ])
    def test_reactive_fields_belong_where_they_apply(self, change, reason):
        net = pn.case14()
        nl.gen_to_sgen(net)
        with pytest.raises(uc.CostError, match=reason):
            uc.normalise(net, [change])

    def test_null_restores_the_default(self):
        net = pn.case14()
        nl.gen_to_sgen(net)
        model = uc.normalise(net, [{"unit": "Gen_bus1", "up_eur_per_mwh": 99, "down_eur_per_mwh": 9}])
        model = uc.merge(model, uc.normalise(net, [{"unit": "Gen_bus1", "up_eur_per_mwh": None}], model))
        assert model == {"Gen_bus1": {"down_eur_per_mwh": 9.0}}


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


def _ok(r):
    assert r.status_code == 200, r.text
    return r.json()


def test_costs_endpoint_and_persistence(client):
    out = _ok(client.post("/api/costs", json={"changes": [
        {"unit": "Gen_bus1", "up_eur_per_mwh": 90}, {"unit": "external grid", "cost_eur_per_mwh": 65}]}))
    rows = {r["unit"]: r for r in out["units"]}
    assert rows["Gen_bus1"]["up_eur_per_mwh"] == 90 and rows["Gen_bus1"]["source"]["up_eur_per_mwh"] == "operator"
    assert rows["External grid"]["cost_eur_per_mwh"] == 65
    stored = json.load(open(mb._ASSUMPTIONS_FILE))["by_fingerprint"]
    assert next(iter(stored.values()))["unit_costs"]["External grid"] == {"cost_eur_per_mwh": 65.0}
    assert _ok(client.get("/api/assumptions"))["unit_costs"]
    assert _ok(client.post("/api/costs", json={"reset": True}))["changed"] == {}
    assert client.post("/api/costs", json={"changes": [{"unit": "x", "up_eur_per_mwh": 1}]}).status_code == 400


class TestRenames:
    def test_a_renamed_bus_keeps_its_measurements(self, client):
        before = _ok(client.post("/api/network/snapshot", json={"timestamp": TS}))["loads"]
        fp = mb.sb.network_fingerprint(mb.app_data["net"])
        _ok(client.post("/api/names", json={"changes": [{"element_type": "bus", "element": "Bus_3",
                                                         "name": "North"}]}))
        after = _ok(client.post("/api/network/snapshot", json={"timestamp": TS}))["loads"]
        assert [x["P_mw"] for x in after] == [x["P_mw"] for x in before]
        assert "North" in [x["bus"] for x in after]
        assert mb.sb.network_fingerprint(mb.app_data["net"]) == fp          # proposals stay this network's
        rsa = _ok(client.post("/api/grid/rsa", json={"timestamp": TS}))
        assert "North" in [v["bus_name"] for v in rsa["all_voltages"]]
        assert any("North" in x["line_name"] for x in rsa["all_line_loading"])  # its lines are relabelled

    def test_a_renamed_unit_keeps_its_model_costs_and_studies(self, client):
        _ok(client.post("/api/generators", json={"changes": [{"unit": "Gen_bus1", "dispatchable": False}]}))
        _ok(client.post("/api/costs", json={"changes": [{"unit": "Gen_bus1", "up_eur_per_mwh": 77}]}))
        _ok(client.post("/api/names", json={"changes": [{"element_type": "sgen", "element": "Gen_bus1",
                                                         "name": "Plant A"}]}))
        gens = _ok(client.get("/api/generators"))
        assert "Plant A" in [u["unit"] for u in gens["units"]] and gens["changed"] == {"Plant A": {"dispatchable": False}}
        rows = {r["unit"]: r for r in _ok(client.get("/api/costs"))["units"]}
        assert rows["Plant A"]["up_eur_per_mwh"] == 77
        # Changes under the new name reach the same unit.
        _ok(client.post("/api/generators", json={"changes": [{"unit": "Plant A", "dispatchable": True}]}))
        assert _ok(client.get("/api/generators"))["changed"] == {}
        assert "Plant A" in mb.PG_MAX_DATA

    @pytest.mark.skipif(shutil.which("ipopt") is None, reason="IPOPT not installed")
    def test_the_opf_runs_on_renamed_units(self, client):
        _ok(client.post("/api/names", json={"changes": [{"element_type": "sgen", "element": "Gen_bus1",
                                                         "name": "Plant A"}]}))
        r = client.post("/api/flexibility/optimize", json={"timestamp": TS})
        assert r.status_code == 200, r.text

    def test_restore_and_reset(self, client):
        _ok(client.post("/api/names", json={"changes": [{"element_type": "bus", "element": 3, "name": "North"}]}))
        out = _ok(client.post("/api/names", json={"changes": [{"element_type": "bus", "element": "North",
                                                               "name": ""}]}))
        assert out["renamed"] == []
        assert "Bus_3" in [b["name"] for b in out["elements"]["bus"]]
        _ok(client.post("/api/names", json={"changes": [{"element_type": "line", "element": 0, "name": "Tie"}]}))
        assert _ok(client.post("/api/names", json={"reset": True}))["renamed"] == []

    @pytest.mark.parametrize("change, reason", [
        ({"element_type": "bus", "element": "Bus_4", "name": "bus_3"}, "already called"),
        ({"element_type": "bus", "element": "Bus_99", "name": "X"}, "no bus"),
        ({"element_type": "shunt", "element": 0, "name": "X"}, "element_type must be"),
        ({"element_type": "bus", "element": 3, "name": "A | B"}, "reserved"),
    ])
    def test_refusals_say_why(self, client, change, reason):
        r = client.post("/api/names", json={"changes": [change]})
        assert r.status_code == 400 and reason in r.text

    def test_names_survive_a_restart(self, client):
        _ok(client.post("/api/names", json={"changes": [{"element_type": "bus", "element": 3, "name": "North"}]}))
        mb.app_data["element_names"] = {}
        mb._apply_generator_model({}, {})
        mb._restore_assumptions()
        assert "North" in [b["name"] for b in _ok(client.get("/api/names"))["elements"]["bus"]]

    def test_descriptions_and_echoes_use_the_names_now(self, client):
        _ok(client.post("/api/generators", json={"changes": [{"unit": "Gen_bus1", "dispatchable": False}]}))
        _ok(client.post("/api/names", json={"changes": [
            {"element_type": "bus", "element": "Bus_3", "name": "North"},
            {"element_type": "sgen", "element": "Gen_bus1", "name": "Plant A"}]}))
        edit = _ok(client.post("/api/proposals/edit", json={"proposal": "p1", "edits": [
            {"op": "scale", "element_type": "load", "bus": "North", "factor": 1.5}]}))
        assert "North" in edit["summary"]["edits"][0] and "Bus_3" not in edit["summary"]["edits"][0]
        block = _ok(client.post("/api/grid/rsa", json={"timestamp": TS}))["generator_model"]
        assert block["changed_units"] == {"Plant A": {"dispatchable": False}}
        assert block["changed_units_as_loaded"] == {"Gen_bus1": {"dispatchable": False}}

    def test_the_data_tab_names_renamed_rows(self, client):
        k0 = _ok(client.get("/api/assumptions"))["names_key"]
        _ok(client.post("/api/names", json={"changes": [{"element_type": "bus", "element": "Bus_3",
                                                         "name": "North"}]}))
        assert _ok(client.get("/api/assumptions"))["names_key"] != k0
        series = _ok(client.post("/api/data/series", json={"substations": ["Bus_3"]}))
        assert series["renamed"] == {"Bus_3": "North"} and "Bus_3" in series["available_substations"]
        assert series["substations"]["Bus_3"]["consumption"]          # its data, under the name as loaded

    def test_the_agent_identity_ignores_renames(self, client):
        gc0 = _ok(client.get("/api/grid_constants"))
        _ok(client.post("/api/names", json={"changes": [{"element_type": "bus", "element": "Bus_1",
                                                         "name": "North"}]}))
        gc1 = _ok(client.get("/api/grid_constants"))
        from llm_agent.agent.config import network_fingerprint
        assert network_fingerprint(gc0) == network_fingerprint(gc1)
        assert gc0["names_key"] != gc1["names_key"] and "North" in gc1["substation_names"]


def test_the_launcher_says_what_is_kept_and_start_fresh_clears_it(client):
    _ok(client.post("/api/scenario", json={"name": "ieee30"}))
    _ok(client.post("/api/names", json={"changes": [{"element_type": "bus", "element": "Bus_3", "name": "North"}]}))
    _ok(client.post("/api/costs", json={"changes": [{"unit": "Gen_bus1", "up_eur_per_mwh": 90}]}))
    _ok(client.post("/api/proposals/edit", json={"proposal": "p1", "edits": [
        {"op": "scale", "element_type": "load", "all": True, "factor": 1.1}]}))
    _ok(client.post("/api/scenario", json={"name": "ieee30_reactive"}))
    kept = {o["name"]: o.get("kept") for o in _ok(client.get("/api/scenario"))["options"]}
    assert kept["ieee30"] == kept["ieee30_reactive"] == {"renames": 1, "costs": 1, "proposals": 1}  # one network
    assert kept["ieee14"] is None                                                         # another network
    assert kept["resume"] is None
    # Switching back restores them; Start fresh clears them, live and saved.
    _ok(client.post("/api/scenario", json={"name": "ieee30"}))
    assert _ok(client.get("/api/names"))["renamed"]
    out = _ok(client.post("/api/scenario/forget", json={"name": "ieee30"}))
    assert all(o.get("kept") is None for o in out["options"])
    assert _ok(client.get("/api/names"))["renamed"] == [] and _ok(client.get("/api/costs"))["changed"] == {}
    assert _ok(client.get("/api/proposals"))["proposals"] == []
    assert client.post("/api/scenario/forget", json={"name": "resume"}).status_code == 400


class TestTheCostTool:
    """The model changes one price per item: a model that fills in every
    property it is shown turned "reactive price to 0" into all eight prices
    at 0 (seen live, and refused again and again)."""

    @pytest.fixture
    def posted(self, monkeypatch):
        from llm_agent.agent import tools
        sent = []
        monkeypatch.setattr(tools, "_post", lambda path, body: sent.append(body) or {"ok": True})
        return tools, sent

    def test_items_are_grouped_by_unit(self, posted):
        tools, sent = posted
        tools.set_unit_costs(changes=[{"unit": u, "field": "q_eur_per_mvarh", "value": 0}
                                      for u in ("Gen_bus1", "Gen_bus2")]
                             + [{"unit": "Gen_bus1", "field": "up_eur_per_mwh", "value": 80, "to_default": False},
                                {"unit": "External grid", "field": "q_band_tan_phi", "to_default": True}])
        assert sent == [{"changes": [{"unit": "Gen_bus1", "q_eur_per_mvarh": 0, "up_eur_per_mwh": 80},
                                     {"unit": "Gen_bus2", "q_eur_per_mvarh": 0},
                                     {"unit": "External grid", "q_band_tan_phi": None}], "reset": False}]

    @pytest.mark.parametrize("item, says", [
        ({"unit": "Gen_bus1", "q_eur_per_mvarh": 0}, "give unit, field"),        # the old one-object-per-unit shape
        ({"unit": "Gen_bus1", "field": "q_eur_per_mvarh"}, "give a value"),
        ({"unit": "Gen_bus1", "field": "q_eur_per_mvarh", "value": True}, "give a value"),
    ])
    def test_an_incomplete_item_is_refused_before_the_backend(self, posted, item, says):
        tools, sent = posted
        assert says in tools.set_unit_costs(changes=[item])["error"] and sent == []

    def test_to_default_beside_a_value(self, posted):
        tools, sent = posted
        item = {"unit": "Gen_bus1", "field": "up_eur_per_mwh", "to_default": True}
        tools.set_unit_costs(changes=[{**item, "value": 0}])           # a filled-in blank: back to default
        assert sent[-1]["changes"] == [{"unit": "Gen_bus1", "up_eur_per_mwh": None}]
        assert "not both" in tools.set_unit_costs(changes=[{**item, "value": 80}])["error"]

    def test_the_same_price_twice_is_refused(self, posted):
        tools, sent = posted
        twice = [{"unit": "Gen_bus1", "field": "q_eur_per_mvarh", "value": v} for v in (0, 1)]
        assert "given twice" in tools.set_unit_costs(changes=twice)["error"] and sent == []

    def test_the_schema_offers_one_price_per_item(self):
        from llm_agent.agent import tool_schemas as ts
        decl = {fd.name: fd for t in ts.TOOLS for fd in t.function_declarations}["set_unit_costs"]
        item = decl.parameters.properties["changes"].items
        assert set(item.properties) == {"unit", "field", "value", "to_default"}
        assert set(item.properties["field"].enum) == set(uc.FIELDS)          # the backend's prices, all of them


def test_reactive_price_of_every_unit_to_zero_end_to_end(client, monkeypatch):
    """The request that failed live, through the tool to the backend: only the
    reactive prices change."""
    from llm_agent.agent import tools
    monkeypatch.setattr(tools, "_post", lambda path, body: _ok(client.post(path, json=body)))
    before = {u["unit"]: u for u in _ok(client.get("/api/costs"))["units"]}
    units = [name for name, u in before.items() if u["kind"] != "external_grid"]
    tools.set_unit_costs(changes=[{"unit": u, "field": "q_eur_per_mvarh", "value": 0} for u in units])
    after = {u["unit"]: u for u in _ok(client.get("/api/costs"))["units"]}
    for name in units:
        assert after[name]["q_eur_per_mvarh"] == 0
        assert {k: after[name][k] for k in uc.PRODUCTION + uc.REDISPATCH} == \
               {k: before[name][k] for k in uc.PRODUCTION + uc.REDISPATCH}
