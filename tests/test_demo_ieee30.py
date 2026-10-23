"""
The default demo: PGLib's IEEE 30-bus case (Alsac–Stott), chosen because its
units give the flexibility studies real choices — five dispatchable units with
their own costs, on voltage control as the case describes them, and a week in
which the line from the external grid congests at the evening peaks.

The backend runs in-process with its storage in a temporary folder. The OPF
checks need IPOPT.
"""

import shutil

import pytest

pytest.importorskip("fastapi.testclient")

import main_backend as mb  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

needs_ipopt = pytest.mark.skipif(shutil.which("ipopt") is None, reason="IPOPT not installed")
PEAK = "2000-01-06 20:00:00"            # the worst hour of the week: the feeder line over its rating
UNITS = {"Gen_bus1", "Gen_bus4", "Gen_bus7", "Gen_bus10", "Gen_bus12"}


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    store = tmp_path_factory.mktemp("ieee30")
    patch = pytest.MonkeyPatch()
    patch.delenv("GRID_PROFILE", raising=False)                      # the default demo
    patch.setattr(mb, "SYSTEMS_DIR", str(store / "systems"))
    patch.setattr(mb, "DATA_FILES_DIR", str(store / "data_files"))
    for name in ("_LAST_UPLOAD_SENTINEL", "_LAST_TIMESERIES_SENTINEL", "_LAST_FORECAST_SENTINEL",
                 "_PROPOSALS_FILE", "_ASSUMPTIONS_FILE", "_SCENARIO_FILE"):
        patch.setattr(mb, name, str(store / f"{name}.json"))
    saved = dict(mb.app_data)
    with TestClient(mb.app) as c:
        yield c
    mb.app_data.clear()
    mb.app_data.update(saved)
    patch.undo()


def _ok(r):
    assert r.status_code == 200, r.text
    return r.json()


def test_it_is_the_default(client):
    payload = _ok(client.get("/api/scenario"))
    assert payload["current"] == "ieee30" and len(mb.app_data["net"].bus) == 30


def test_the_units_load_on_voltage_control_with_the_cases_own_costs(client):
    models = _ok(client.get("/api/generators"))
    units = {u["unit"]: u for u in models["units"]}
    assert set(units) == UNITS and models["changed"] == {}           # as loaded: nothing changed
    assert all(u["control"] == "voltage" and u["dispatchable"] for u in units.values())
    assert units["Gen_bus1"]["vm_pu"] == 1.025 and units["Gen_bus4"]["vm_pu"] == 1.0
    costs = {u["unit"]: u for u in _ok(client.get("/api/costs"))["units"]}
    assert all(costs[u]["source"]["cost_eur_per_mwh"] == "case" for u in UNITS)
    assert len({(costs[u]["cost_eur_per_mwh"], costs[u]["cost_eur_per_mw2h"]) for u in UNITS}) >= 4


def test_the_evening_peak_congests_the_line_from_the_external_grid(client):
    peak = _ok(client.post("/api/grid/rsa", json={"timestamp": PEAK}))
    assert any(v["violation_type"] == "line_loading" and v["element"].startswith("Bus_0 -> Bus_1")
               for v in peak["violations"])
    night = _ok(client.post("/api/grid/rsa", json={"timestamp": "2000-01-06 03:00:00"}))
    assert night["total_violations"] == 0


@needs_ipopt
def test_the_options_differ_and_nothing_is_left_out_unsaid(client):
    out = _ok(client.post("/api/flexibility/optimize", json={"timestamp": PEAK, "options": True}))
    assert out["status"] == "options" and len(out["options"]) >= 3
    moves = [{r["unit"]: r["dp_mw"] for r in o["dispatch"]} for o in out["options"]]
    for i, a in enumerate(moves):                                    # no two options dispatch alike
        for b in moves[i + 1:]:
            assert max(abs(a.get(u, 0.0) - b.get(u, 0.0)) for u in set(a) | set(b)) > 1.0
    assert len({frozenset(o["units_changing_p"]) for o in out["options"]}) >= 3
    for o in out["options"]:
        assert o["units_called"] == sorted(o["instructions"]) and o["n_units_called"] == len(o["instructions"])
        for row in o["dispatch"]:
            if row["unit"] in UNITS:                                     # every unit here holds a voltage
                assert "setpoint_vm_pu_before" in row and "setpoint_vm_pu_after" in row
    by_id = {o["id"]: o for o in out["options"]}
    assert by_id["fewest_units"]["n_units_called"] < by_id["least_disturbance"]["n_units_called"]
    for o in out["options"]:                                         # re-solved by an ordinary power flow
        check = o["power_flow_check"]
        assert check["converged"] and check["confirmed"] and check["violations"] == []
        assert check["min_vm_pu"] == pytest.approx(o["min_vm_pu"], abs=0.002)
        assert check["max_loading_pct"] == pytest.approx(o["max_loading_pct"], abs=0.5)
        assert check["import_mw"] == pytest.approx(o["import_mw"], abs=0.05)
    shown = {o["id"] for o in out["options"]} | {n["id"] for n in out["not_offered"]}
    assert shown == {"least_disturbance", "cheapest", "fewest_units", "without_main_unit", "least_import",
                     "most_headroom", "demand_response"}
    dr = next(n for n in out["not_offered"] if n["id"] == "demand_response")
    assert "redispatching the units is cheaper" in dr["reason"]       # at the synthetic prices
    assert all(n["reason"] for n in out["not_offered"])
    assert "case" in out["costs"]["prices_from"]


@needs_ipopt
def test_a_unit_the_operator_rules_out_stays_at_its_schedule(client):
    out = _ok(client.post("/api/flexibility/optimize",
                          json={"timestamp": PEAK, "options": True, "hold_units": ["gen_bus1"]}))
    assert list(out["held_units"]) == ["Gen_bus1"] and out["held_units"]["Gen_bus1"] > 0
    for option in out["options"]:
        assert "Gen_bus1" not in option["units_changing_p"]
        assert "Gen_bus1" not in option["units_called"]                  # no set-point change either
        row = next((r for r in option["dispatch"] if r["unit"] == "Gen_bus1"), None)
        assert row is None or abs(row["dp_mw"]) < 0.01
    for entry in out["not_offered"]:                                # names its unit, never "the main unit"
        if entry["id"] == "without_main_unit":
            assert entry["label"].startswith("Without Gen_bus") and entry["label"] != "Without Gen_bus1"
    one = _ok(client.post("/api/flexibility/optimize", json={"timestamp": PEAK, "hold_units": ["Gen_bus1"]}))
    assert one["held_units"] == out["held_units"]
    moved = {r.get("element") or r.get("name"): r for r in one["activated_resources"]}
    assert abs(float((moved.get("Gen_bus1") or {}).get("Pg_up", 0)) - float((moved.get("Gen_bus1") or {}).get("Pg_down", 0))) < 0.01


@needs_ipopt
def test_near_optimal_alternatives_stay_within_the_margin_and_change_other_units(client):
    out = _ok(client.post("/api/flexibility/optimize", json={"timestamp": PEAK, "options": True,
                                                            "near_optimal": True, "near_optimal_pct": 10}))
    assert out["set"] == "near_optimal" and out["near_optimal_pct"] == 10
    first, *alternatives = out["options"]
    assert first["id"] == "cheapest" and alternatives
    assert out["budget_eur_per_h"] == pytest.approx(first["cost_eur_per_h"] * 1.10, abs=0.02)
    moves = [{r["unit"]: r["dp_mw"] for r in o["dispatch"]} for o in out["options"]]
    for o, move in zip(alternatives, moves[1:]):
        assert o["id"].startswith("near_optimal_") and o["cost_eur_per_h"] <= out["budget_eur_per_h"] + 0.05
        assert -0.05 <= o["above_cheapest_eur_per_h"]
        assert o["avoids"] or o["adds"] or o["how"].endswith("moved differently")
        assert "above the cheapest" in o["how"]
    for i, a in enumerate(moves):                                    # each a different dispatch
        for b in moves[i + 1:]:
            assert max(abs(a.get(u, 0.0) - b.get(u, 0.0)) for u in set(a) | set(b)) > 0.1
    held = _ok(client.post("/api/flexibility/optimize", json={"timestamp": PEAK, "options": True, "near_optimal": True,
                                                             "hold_units": ["Gen_bus1"]}))
    assert all("Gen_bus1" not in o["units_called"] for o in held["options"])


@needs_ipopt
def test_options_for_the_corrective_action_after_an_outage(client):
    noon = "2000-01-03 12:00:00"
    out = _ok(client.post("/api/contingency/optimize", json={"timestamp": noon, "element_type": "line",
                                                             "element_index": 36, "options": True}))
    assert out["status"] == "options" and out["outage"]["element"] == "Bus_26 -> Bus_28 [L36]"
    assert len(out["options"]) >= 3
    assert all(o["power_flow_check"]["confirmed"] for o in out["options"])     # without the line
    by_id = {o["id"]: o for o in out["options"]}
    assert by_id["fewest_units"]["n_units_called"] < by_id["least_disturbance"]["n_units_called"]
    # After L29 no redispatch of the units clears it: the last resort is the answer
    # — options that also reduce load — and it says why the units alone cannot.
    stuck = _ok(client.post("/api/contingency/optimize", json={"timestamp": noon, "element_type": "line",
                                                               "element_index": 29, "options": True}))
    assert stuck["outage"]["element"] == "Bus_14 -> Bus_22 [L29]" and stuck["set"] == "last_resort"
    assert stuck["redispatch_alone"]["clears_violations"] is False and stuck["redispatch_alone"]["why"]
    assert all(o["load_reduction_mw"] > 0 and o["power_flow_check"]["confirmed"] for o in stuck["options"])


@needs_ipopt
def test_the_n1_screen_of_a_chosen_option(client):
    """Each option keeps a reference; the N-1 screen applies that option's
    dispatch at its own moment — and the options differ under N-1."""
    out = _ok(client.post("/api/flexibility/optimize", json={"timestamp": PEAK, "options": True}))
    refs = {o["id"]: o["ref"] for o in out["options"]}
    assert all(ref == f"{out['set_id']}:{oid}" for oid, ref in refs.items())
    schedule = _ok(client.post("/api/contingency/simulate_all", json={"timestamp": PEAK}))
    cheapest = _ok(client.post("/api/contingency/simulate_all", json={"option": refs["cheapest"]}))
    assert "option" not in schedule
    assert cheapest["option"]["ref"] == refs["cheapest"] and cheapest["option"]["screen"] == "N-1"
    assert cheapest["timestamp"] == PEAK and cheapest["total_outages_tested"] == schedule["total_outages_tested"]
    assert cheapest["total_violations"] < schedule["total_violations"]        # its dispatch, not the schedule's
    by_id = _ok(client.post("/api/contingency/simulate_all", json={"option": "cheapest"}))
    assert by_id["option"]["ref"] == refs["cheapest"]                           # the newest set's, by id


@needs_ipopt
def test_an_option_after_an_outage_is_screened_n1_1(client):
    out = _ok(client.post("/api/contingency/optimize", json={"timestamp": "2000-01-03 12:00:00", "element_type": "line",
                                                             "element_index": 36, "options": True}))
    screen = _ok(client.post("/api/contingency/simulate_all", json={"option": out["options"][0]["ref"]}))
    assert screen["option"]["screen"] == "N-1-1" and screen["option"]["after_outage"] == "Bus_26 -> Bus_28 [L36]"
    plain = _ok(client.post("/api/contingency/simulate_all", json={"timestamp": "2000-01-03 12:00:00"}))
    assert screen["total_outages_tested"] == plain["total_outages_tested"] - 1      # L36 is already out


@needs_ipopt
def test_an_option_is_only_applied_where_it_was_computed(client):
    out = _ok(client.post("/api/flexibility/optimize", json={"timestamp": PEAK, "options": True}))
    ref = out["options"][0]["ref"]
    for body, says in (({"option": "S999:cheapest"}, "there is no option"),
                       ({"option": ref, "timestamp": "2000-01-06 19:00:00"}, "another moment"),
                       ({"option": ref, "load_scaling_factor": 1.2}, "another load scaling")):
        r = client.post("/api/contingency/simulate_all", json=body)
        assert r.status_code == 400 and says in r.json()["detail"]
    _ok(client.post("/api/generators", json={"changes": [{"unit": "Gen_bus4", "control": "power_factor"}]}))
    try:
        r = client.post("/api/contingency/simulate_all", json={"option": ref})
        assert r.status_code == 400 and "compute the options again" in r.json()["detail"]
    finally:
        _ok(client.post("/api/generators", json={"reset": True}))
    assert _ok(client.post("/api/contingency/simulate_all", json={"option": ref}))["option"]["ref"] == ref


def test_a_margin_out_of_range_is_refused(client):
    r = client.post("/api/flexibility/optimize", json={"timestamp": PEAK, "options": True, "near_optimal": True,
                                                       "near_optimal_pct": 0})
    assert r.status_code == 422


def test_the_power_flow_check_catches_what_does_not_hold(client):
    """A dispatch that leaves the feeder overloaded is not confirmed, and says where."""
    import copy

    import flex_options as fo
    net = copy.deepcopy(mb._net_for(None))
    units = {u["unit"] for u in _ok(client.get("/api/generators"))["units"]}
    p = {u: 0.0 for u in units}                                       # every unit off: all from the grid
    nothing = fo.Option(id="least_disturbance", label="", how="", p=p, q={u: 0.0 for u in units}, p0=p, q0=p,
                        pext={}, qext={}, v={}, max_loading_pct=None, max_loading_branch=None)
    check = mb._power_flow_check(net, nothing, 0.95, 1.05)
    assert check["converged"] and not check["confirmed"]
    assert any(v.startswith("Bus_0 -> Bus_1 [L0]") for v in check["violations"])


def test_an_unknown_unit_to_hold_is_refused_by_name(client):
    r = client.post("/api/flexibility/optimize", json={"timestamp": PEAK, "options": True, "hold_units": ["Gen_bus99"]})
    assert r.status_code == 400 and "Gen_bus99" in r.json()["detail"] and "Gen_bus12" in r.json()["detail"]


def test_the_envelope_studies_a_voltage_controlled_unit_as_an_injection(client):
    out = _ok(client.post("/api/flexibility/envelope", json={"timestamp": "2000-01-03 12:00:00",
                                                            "gen_name": "Gen_bus7"}))
    assert "holds a voltage set-point" in out["unit_studied_as_injection"]
    assert out["n_feasible"] > 0
