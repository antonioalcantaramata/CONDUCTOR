"""
The facts the ontology ablation's tasks are graded on (evaluation/semantics.py,
evaluation/reasoning.py), checked against the networks themselves.

The tasks write their answers down ("Bus 11 is lowest at 18:00 on 3 January",
"L33 cuts 1.84 MW"). If a network, its data or a model change moves one, this
fails and names the task to update — the ablation must never grade against a
stale truth.
"""

import pandapower.networks as pn
import pytest

import network_loader as nl
import semantics as sm
from evaluation import reasoning as rs
from evaluation import semantics as se


@pytest.fixture(scope="module")
def cigre():
    net = pn.create_cigre_network_mv(with_der="all")
    return net, sm.build(net)


@pytest.fixture(scope="module")
def ieee30():
    profile = nl.load_profile("pglib_case30")
    net = nl.load_network(profile)
    net, _, _ = nl.gen_to_sgen(net)
    return net, sm.build(net)


def _around(net, s, name):
    import resolve as rv
    found = rv.find_unit(net, name, ("gen", "sgen", "storage")) if not any(k in name for k in ("->", "Bus")) \
        else rv.find(net, name, ("bus", "line", "trafo"))
    return sm.around(net, s, found.etype, found.idx)


def test_cigre_lookups(cigre):
    net, s = cigre
    units = {u["name"]: u for u in s["units"]}
    assert units["Battery 1"]["bus"] == "Bus 5" and units["Battery 2"]["bus"] == "Bus 10"     # cigre_storage
    assert s["voltage_levels_kv"] == [110.0, 20.0] and s["external_grids"][0]["bus"] == "Bus 0"  # cigre_levels
    assert {units[f"PV {k}"]["bus_index"] for k in (3, 11)} and all(                           # cigre_levels: PV at 20 kV
        b["kv"]["value"] == 20.0 for b in s["buses"] if b["index"] in {u["bus_index"] for u in s["units"]
                                                                     if u["source"]["value"] == "solar"})
    assert sm.view(s, "units")["wind_or_solar"] == ["PV 3", "PV 4", "PV 5", "PV 6", "PV 8", "PV 9", "PV 10",
                                                    "PV 11", "WKA 7"]                          # cigre_renewables, para_*
    r11 = _around(net, s, "Bus 11")["supply"]                                                    # cigre_feeder
    assert r11["feeder"].startswith("Line 1-2") and any("Line 3-8" in x for x in r11["route_from_external_grid"])
    l23 = _around(net, s, "Bus 2 -> Bus 3 [L1]")["without_it"]                                  # cigre_outage
    assert {"Load R11", "Load CI10"} <= set(l23["loads_cut_off"])
    assert any("Line 14-8" in t for t in l23["open_ties_that_could_restore"])


def test_reasoning_cigre_facts(cigre):
    net, s = cigre
    assert net.sgen.at[10, "name"] == "CHP diesel 1" and net.sgen.at[11, "name"] == "Fuel cell 1"  # names_cigre
    t0 = _around(net, s, "Bus 0 -> Bus 1 [T0]")["without_it"]                                     # compose_t0
    assert {"WKA 7", *[f"PV {k}" for k in (3, 4, 5, 6, 8, 9, 10, 11)]} <= set(t0["units_cut_off"])
    assert any("Line 14-8" in t for t in t0["open_ties_that_could_restore"])
    assert _around(net, s, "Battery 1")["supply"]["feeder"].startswith("Line 1-2")              # compose_battery
    assert sorted(rs._CIGRE_RENEWABLE) == sorted(sm.view(s, "units")["wind_or_solar"])           # drop_cigre_*
    units = {u["name"]: u for u in s["units"]}
    assert net.sgen.at[1, "name"] == "PV 4" and units["PV 4"]["source"]["value"] == "solar"       # names_cigre_wind
    assert units["WKA 7"]["source"]["value"] == "wind" and [u for u, f in units.items()
                                                            if f["source"]["value"] == "wind"] == ["WKA 7"]
    assert net.sgen.at[9, "name"] == "Residential fuel cell 1"                                   # names_cigre_storage
    assert units["Residential fuel cell 1"]["source"]["value"] == "thermal"
    assert sorted(u for u, f in units.items() if f["source"]["value"] == "storage") == ["Battery 1", "Battery 2"]


def test_ieee30_facts(ieee30):
    net, s = ieee30
    assert s["voltage_levels_kv"] == [135.0]                                                     # ieee30_level
    assert sm.view(s, "units")["wind_or_solar"] == []                                            # ieee30_renewables, drop_ieee30_none
    assert {d["category"]["value"] for d in s["loads"]} == {"unknown"}                           # ieee30_customers
    assert sorted(b.split(" [")[1] for b in s["external_grids"][0]["branches"]) == ["L0]", "L1]"]  # ieee30_external
    l33 = _around(net, s, "Bus_24 -> Bus_25 [L33]")["without_it"]                               # ieee30_radial, compose_l33
    assert l33["loads_cut_off"] == ["Load 18"] and l33["open_ties_that_could_restore"] == []
    assert "Gen_bus12" in {u["name"] for u in s["units"]}                                        # drop_ieee30_declared, persist
    as_loaded = nl.load_network(nl.load_profile("pglib_case30"))                                # names_ieee30_wind: the
    assert as_loaded.gen.at[3, "bus"] == 7                                                       # backend renames gen 3,
    assert {u["source"]["basis"] for u in s["units"]} == {"unknown"}                             # Gen_bus7, of unknown source


def test_the_lowest_bus_at_the_evening_peak(tmp_path_factory):
    """compose_battery: Bus 11 is the lowest bus at 18:00 on 3 January (the CIGRE demo's week)."""
    pytest.importorskip("fastapi.testclient")
    import main_backend as mb
    from fastapi.testclient import TestClient

    scratch = tmp_path_factory.mktemp("truth")
    patch = pytest.MonkeyPatch()
    patch.setenv("GRID_PROFILE", "cigre_mv")
    for name in ("_LAST_UPLOAD_SENTINEL", "_LAST_TIMESERIES_SENTINEL", "_LAST_FORECAST_SENTINEL",
                 "_PROPOSALS_FILE", "_ASSUMPTIONS_FILE", "_SCENARIO_FILE"):
        patch.setattr(mb, name, str(scratch / f"{name}.json"))
    saved = dict(mb.app_data)
    try:
        with TestClient(mb.app) as client:
            r = client.post("/api/grid/rsa", json={"timestamp": "2000-01-03 18:00:00"}).json()
            vs = r["all_voltages"]
            pairs = [(v if isinstance(v, (int, float)) else v["vm_pu"], k)
                     for k, v in (vs.items() if isinstance(vs, dict) else enumerate(vs))]
            assert min(pairs)[1] in (11, "11", "Bus 11")
            # IEEE 30 at noon on 3 January: a renewable drop changes nothing (drop_ieee30_none).
            client.post("/api/scenario", json={"name": "ieee30"})
            r = client.post("/api/grid/rsa", json={"timestamp": "2000-01-03 12:00:00", "sgen_scaling_factor": 0.5})
            assert r.json()["renewables_scaled"]["units"] == []
            # compose_l33: the demand L33 cuts off at the clock's moment (the first of the week) — the
            # layer reports the current operating point, not the case's nominal 3.5 MW.
            l33 = client.get("/api/network/semantics", params={"about": "around",
                                                                "element": "Bus_24 -> Bus_25 [L33]"}).json()
            assert l33["operating_point"] == "2000-01-01 00:00:00"
            assert round(l33["without_it"]["load_cut_off_mw"], 2) == 1.84
    finally:
        mb.app_data.clear()
        mb.app_data.update(saved)
        patch.undo()


def test_switch_and_data_facts(cigre, tmp_path_factory):
    net, s = cigre
    sw = {x["name"]: x for x in s["switches"]}                                         # switches_open, compose_tie
    assert {n for n, x in sw.items() if x["normal_open"]["value"]} == {"S1", "S2", "S3"}
    assert sw["S1"]["role"]["value"].startswith("tie between feeders")
    assert sw["S1"]["where"]["element"].startswith("Line 14-8")
    l23 = _around(net, s, "Bus 2 -> Bus 3 [L1]")["without_it"]["open_ties_that_could_restore"]
    assert any("S1" in t for t in l23)
    for name in ("S2", "S3"):                                                          # switches_loops
        assert sw[name]["role"]["value"].startswith("normally open point closing a loop within feeder Line 1-2")
    breakers = {n: x for n, x in sw.items() if x["cim_class"]["value"] == "Breaker"}   # switches_breakers
    assert sorted(breakers) == ["Switch 6", "Switch 7"]
    assert sorted(x["where"]["element"].split(" | ")[0] for x in breakers.values()) == ["Trafo 0-1", "Trafo 0-12"]
    assert sw["S1"]["normal_open"]["value"] and sw["S1"]["where"]["element"].startswith("Line 14-8")  # switches_line148


def test_what_is_measured_at_the_clock(tmp_path_factory):
    """data_cigre, data_ieee30: at the clock's first moment."""
    pytest.importorskip("fastapi.testclient")
    import main_backend as mb
    from fastapi.testclient import TestClient

    scratch = tmp_path_factory.mktemp("truth_data")
    patch = pytest.MonkeyPatch()
    patch.setenv("GRID_PROFILE", "cigre_mv")
    for name in ("_LAST_UPLOAD_SENTINEL", "_LAST_TIMESERIES_SENTINEL", "_LAST_FORECAST_SENTINEL",
                 "_PROPOSALS_FILE", "_ASSUMPTIONS_FILE", "_SCENARIO_FILE"):
        patch.setattr(mb, name, str(scratch / f"{name}.json"))
    saved = dict(mb.app_data)
    try:
        with TestClient(mb.app) as client:
            full = client.get("/api/network/semantics").json()
            loads = {d["name"]: d for d in full["loads"]}
            units = {u["name"]: u for u in full["units"]}
            assert loads["Load R5"]["values"]["q"]["kind"] == "derived" and "0.97" in loads["Load R5"]["values"]["q"]["from"]
            assert units["PV 9"]["values"]["p"]["kind"] == "allocated"
            assert units["WKA 7"]["values"]["p"]["kind"] == "measured"                        # data_cigre_wind
            assert units["Battery 1"]["values"]["p"]["kind"] == "set"                         # data_cigre_battery
            assert loads["Load R1"]["values"]["p"]["kind"] == "allocated"                     # data_cigre_load_split
            assert loads["Load CI1"]["bus"] == loads["Load R1"]["bus"] == "Bus 1"
            client.post("/api/scenario", json={"name": "ieee30"})
            full = client.get("/api/network/semantics").json()
            load5 = next(d for d in full["loads"] if d["name"] == "Load 5")
            assert load5["values"]["q"]["kind"] == "derived"
            gen12 = next(u for u in full["units"] if u["name"] == "Gen_bus12")                # data_ieee30_solved
            assert gen12["values"]["q"]["kind"] == "solved"
    finally:
        mb.app_data.clear()
        mb.app_data.update(saved)
        patch.undo()


def test_every_task_is_guarded():
    guarded = {"cigre_storage", "cigre_levels", "cigre_renewables", "cigre_feeder", "cigre_outage", "ieee30_level",
               "ieee30_renewables", "ieee30_customers", "ieee30_external", "ieee30_radial", "names_cigre",
               "names_ieee30", "drop_ieee30_none", "drop_ieee30_declared", "drop_cigre_stated", "drop_cigre_declared",
               "compose_t0", "compose_battery", "compose_l33", "para_weather", "para_vary", "para_variable",
               "persist_ieee30", "compose_tie", "switches_open", "data_cigre", "data_ieee30",
               "names_cigre_wind", "names_cigre_storage", "names_ieee30_wind", "switches_loops", "switches_breakers",
               "switches_line148", "data_cigre_wind", "data_cigre_battery", "data_cigre_load_split",
               "data_ieee30_solved"}
    tasks = {t.id for s in se.SUITES.values() for t in s.tasks} | set(rs.TASKS)
    assert tasks == guarded, f"a task without a guarded truth: {tasks ^ guarded}"
