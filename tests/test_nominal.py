"""
The nominal operating point: the network's own values as loaded, kept beside
the synthetic data built from them and studied like a moment with no time.

Before it existed, the first synthetic tick was written over the case's loads
and generator set-points at load time — the case's own operating point
(including its reactive power) was gone until the file was reloaded.
"""

import copy
import io
import pathlib

import numpy as np
import pandapower as pp
import pandapower.networks as pn
import pytest

import network_loader as nl
import nominal as nom
import sandbox as sb

BACKEND = pathlib.Path(__file__).resolve().parents[1] / "backend"


class TestSnapshot:
    def test_it_keeps_what_a_tick_overwrites_and_puts_it_back_exactly(self):
        net = pn.case14()
        snap = nom.snapshot(net)
        assert nom.usable(snap) == (True, "")
        moved = copy.deepcopy(net)
        moved.load["p_mw"] *= 1.7
        moved.load["q_mvar"] = moved.load["p_mw"] * np.tan(np.arccos(0.95))   # what a tick does
        moved.gen["p_mw"] = 0.0
        nom.apply(moved, snap)
        for etype in ("load", "gen"):
            for col in snap["tables"][etype].columns:
                assert moved[etype][col].tolist() == pytest.approx(net[etype][col].tolist())

    def test_rows_a_proposal_added_keep_their_own_values(self):
        net = pn.case14()
        snap = nom.snapshot(net)
        prop, _ = sb.materialise(net, [{"op": "add_load", "name": "DC", "bus": 9, "p_mw": 30}])
        prop.load.loc[prop.load.index[-1], "p_mw"] = 12.5
        nom.apply(prop, snap)
        assert prop.load["p_mw"].iloc[-1] == 12.5

    def test_missing_values_make_it_unusable_with_the_reason(self):
        net = pn.case14()
        net.load.loc[3, "p_mw"] = -9999.0                      # Bornholm's missing-value marker
        net.sgen = net.sgen.reindex([0])                         # a row of NaN
        net.sgen.loc[0, ["bus", "p_mw", "q_mvar"]] = [1, np.nan, 0.0]
        snap = nom.snapshot(net)
        ok, why = nom.usable(snap)
        assert not ok and "-9999 (a missing-value marker)" in why and "has no p_mw" in why
        with pytest.raises(nom.NominalError):
            nom.apply(copy.deepcopy(net), snap)
        assert nom.usable(None)[0] is False

    def test_added_elements_sit_at_their_design_value_at_nominal(self):
        net = pn.case14()
        prop, _ = sb.materialise(net, [
            {"op": "add_load", "name": "DC", "bus": 9, "p_mw": 30, "profile": "follow_bus"},
            {"op": "add_load", "name": "F", "bus": 4, "p_mw": 10, "profile": "shift",
             "start_hour": 8, "end_hour": 18},
            {"op": "add_generator", "name": "W", "bus": 9, "p_max_mw": 12, "kind": "wind"},
            {"op": "add_generator", "name": "G", "bus": 4, "p_max_mw": 20, "p_mw": 5,
             "kind": "dispatchable", "control": "power_factor"}])
        meas = nom.as_measurements(prop, nom.snapshot(net), lambda b: f"Bus_{b}")
        report = sb.apply_profiles(prop, nom.NOMINAL, meas, sb.dataset_peaks({nom.NOMINAL: meas}))
        got = {r["element"]: (r["p_mw"], r["source"]) for r in report}
        assert got == {"DC": (30.0, "nominal: its full size"), "F": (10.0, "nominal: its full size"),
                       "W": (12.0, "nominal: its rated output"), "G": (5.0, "set-point")}


def _pristine():
    """The app's network exactly as `_lifespan_generic` builds it, before any tick."""
    profile = nl.load_profile("pglib_case14", backend_dir=str(BACKEND))
    net = nl.load_network(profile, backend_dir=str(BACKEND))
    if len(net.gen):
        net, _, _ = nl.gen_to_sgen(net)
    nl.apply_voltage_setpoints(net, profile)
    return net


class TestThroughTheBackend:
    def test_the_snapshot_at_nominal_is_the_case_itself(self, backend_client):
        assert backend_client.get("/api/time/timeline").json()["nominal"] == {"available": True, "reason": ""}
        out = backend_client.post("/api/network/snapshot", json={"data_source": "nominal"}).json()
        assert out["timestamp"] == "nominal" and out["data_source"] == "nominal"
        net = _pristine()
        assert out["totals"]["total_load_mw"] == pytest.approx(float(net.load.p_mw.sum()), abs=1e-3)
        assert out["nominal"]["totals"]["load_mvar"] == pytest.approx(float(net.load.q_mvar.sum()), abs=1e-3)
        pp.runpp(net)
        # The same power flow on the untouched case: the twin's nominal point is that case.
        assert out["totals"]["total_losses_mw"] == pytest.approx(
            float(net.res_line.pl_mw.sum() + net.res_trafo.pl_mw.sum()), abs=1e-3)
        assert out["ext_grid"]["P_import_mw"] == pytest.approx(float(net.res_ext_grid.p_mw.sum()), abs=1e-3)

    def test_it_keeps_the_case_reactive_power_a_tick_replaces(self, backend_client):
        nominal = backend_client.post("/api/network/snapshot", json={"data_source": "nominal"}).json()
        tick = backend_client.post("/api/network/snapshot", json={}).json()
        ratios = {round(l["Q_mvar"] / l["P_mw"], 3) for l in tick["loads"] if l.get("P_mw")}
        assert ratios == {round(float(np.tan(np.arccos(0.95))), 3)}             # a tick: pf 0.95
        q_nominal = sum(l["Q_mvar"] for l in nominal["loads"])
        assert q_nominal == pytest.approx(float(_pristine().load.q_mvar.sum()), abs=1e-3)

    def test_how_the_datasets_relate_to_it(self, backend_client):
        rel = backend_client.post("/api/network/snapshot", json={"data_source": "nominal"}).json()["nominal"]
        m = rel["measurements"]
        assert m["total_load_mw"]["nominal"] == rel["totals"]["load_mw"]
        assert m["total_load_mw"]["min"] <= m["total_load_mw"]["mean"] <= m["total_load_mw"]["max"]
        assert 0 <= m["share_of_ticks_above_nominal_load"] <= 1 and m["n_ticks"] > 0
        assert m["by_substation"] and {"substation", "nominal_load_mw", "max_mw"} <= set(m["by_substation"][0])
        assert "no time" in rel["meaning"]

    def test_point_studies_run_at_nominal(self, backend_client):
        rsa = backend_client.post("/api/grid/rsa", json={"data_source": "nominal"})
        assert rsa.status_code == 200 and rsa.json()["timestamp"] == "nominal"
        n1 = backend_client.post("/api/contingency/simulate_all", json={"data_source": "nominal"})
        assert n1.status_code == 200
        hc = backend_client.post("/api/flexibility/hosting_capacity", json={"data_source": "nominal", "bus": "Bus 3"})
        assert hc.status_code == 200

    @pytest.mark.parametrize("path, body", [
        ("/api/rsa/worst_case", {}), ("/api/rsa/historical_risk", {}), ("/api/kpi/forecast", {}),
        ("/api/data/series", {})])
    def test_window_studies_refuse_it_with_the_reason(self, backend_client, path, body):
        r = backend_client.post(path, json={**body, "data_source": "nominal"})
        assert r.status_code == 400 and "single moment with no time" in r.json()["detail"]

    def test_it_has_no_timestamp(self, backend_client):
        r = backend_client.post("/api/grid/rsa", json={"data_source": "nominal", "timestamp": "2000-01-01"})
        assert r.status_code == 400 and "has no time" in r.json()["detail"]

    def test_a_proposal_at_nominal(self, backend_client):
        import main_backend as mb
        backend_client.post("/api/proposals/edit", json={"proposal": "dc", "edits": [
            {"op": "add_load", "name": "DC", "bus": "Bus 9", "p_mw": 20, "profile": "follow_bus"}]})
        out = backend_client.post("/api/proposals/compare", json={"proposal": "dc", "study": "rsa",
                                                                 "data_source": "nominal"}).json()
        added = out["network"]["added_elements_at_tick"]
        assert added["timestamp"] == "nominal"
        assert added["elements"][0]["p_mw"] == 20.0 and added["elements"][0]["source"] == "nominal: its full size"
        assert out["proposal"]["total_load_mw"] - out["base"]["total_load_mw"] == pytest.approx(20.0, abs=1e-3)
        scan = backend_client.post("/api/proposals/compare", json={"proposal": "dc", "study": "scan",
                                                                  "data_source": "nominal"})
        assert scan.status_code == 400
        mb._discard_proposals()

    def test_the_data_viewer_gets_the_nominal_values(self, backend_client):
        out = backend_client.post("/api/data/series", json={"substations": ["Bus_9"]}).json()
        assert out["nominal"]["totals"]["consumption"] == pytest.approx(float(_pristine().load.p_mw.sum()), abs=1e-3)
        assert set(out["nominal"]["substations"]) == {"Bus_9"}


def test_an_uploaded_network_keeps_its_nominal_point(tmp_path, monkeypatch):
    pytest.importorskip("fastapi.testclient")
    import main_backend as mb
    from fastapi.testclient import TestClient
    monkeypatch.setenv("GRID_PROFILE", "pglib_case14")
    monkeypatch.setattr(mb, "SYSTEMS_DIR", str(tmp_path / "systems"))
    monkeypatch.setattr(mb, "DATA_FILES_DIR", str(tmp_path / "data_files"))
    for name in ("_LAST_UPLOAD_SENTINEL", "_LAST_TIMESERIES_SENTINEL", "_LAST_FORECAST_SENTINEL",
                 "_PROPOSALS_FILE", "_ASSUMPTIONS_FILE", "_SCENARIO_FILE"):
        monkeypatch.setattr(mb, name, str(tmp_path / f"{name}.json"))
    saved = dict(mb.app_data)
    case9 = pn.case9()
    try:
        with TestClient(mb.app) as c:
            r = c.post("/api/network/upload",
                       files={"file": ("grid.json", io.BytesIO(pp.to_json(case9).encode()), "application/json")})
            assert r.status_code == 200, r.text
            out = c.post("/api/network/snapshot", json={"data_source": "nominal"}).json()
            assert out["totals"]["total_load_mw"] == pytest.approx(float(case9.load.p_mw.sum()), abs=1e-3)
    finally:
        mb.app_data.clear()
        mb.app_data.update(saved)


def test_an_unusable_nominal_point_is_refused_and_shown(backend_client):
    import main_backend as mb
    kept = mb.app_data["nominal"]
    try:
        bad = copy.deepcopy(kept)
        bad["problems"] = ["load 'L3' p_mw is -9999 (a missing-value marker)"]
        mb.app_data["nominal"] = bad
        tl = backend_client.get("/api/time/timeline").json()["nominal"]
        assert tl["available"] is False and "-9999" in tl["reason"]
        r = backend_client.post("/api/grid/rsa", json={"data_source": "nominal"})
        assert r.status_code == 422 and "not available" in r.json()["detail"]
        assert backend_client.post("/api/data/series", json={}).json()["nominal"] is None
    finally:
        mb.app_data["nominal"] = kept
