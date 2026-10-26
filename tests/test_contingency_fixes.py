"""
Contingency analysis — what a code review found wrong, fixed:

- an outage that cuts load off was read as harmless (an unsupplied bus has no
  voltage to violate): the screen now lists it and is not N-1 secure, and the
  single outage is not secure;
- the outage kernel stored element positions as indices (wrong for a network
  whose indices are not 0..n-1);
- critical outages were ranked with voltages measured from 1.0 p.u. instead of
  beyond the band;
- after one of two parallel circuits trips, the optimisation kept both
  circuits' admittance.
"""

import shutil

import pandapower as pp
import pytest

import ca_engine as ca
import main_backend as mb

PEAK, NIGHT = "2000-01-06 20:00:00", "2000-01-06 04:00:00"


@pytest.fixture(scope="module")
def demo(tmp_path_factory):
    pytest.importorskip("fastapi.testclient")
    from fastapi.testclient import TestClient
    store = tmp_path_factory.mktemp("contingency")
    patch = pytest.MonkeyPatch()
    patch.delenv("GRID_PROFILE", raising=False)
    patch.setattr(mb, "SYSTEMS_DIR", str(store / "systems"))
    patch.setattr(mb, "DATA_FILES_DIR", str(store / "data_files"))
    for name in ("_LAST_UPLOAD_SENTINEL", "_LAST_TIMESERIES_SENTINEL", "_LAST_FORECAST_SENTINEL",
                 "_PROPOSALS_FILE", "_ASSUMPTIONS_FILE", "_SCENARIO_FILE"):
        patch.setattr(mb, name, str(store / f"{name}.json"))
    saved = dict(mb.app_data)
    with TestClient(mb.app) as client:
        yield client
    mb.app_data.clear()
    mb.app_data.update(saved)
    patch.undo()


def test_the_screen_lists_an_outage_that_cuts_load_off(demo):
    out = demo.post("/api/contingency/simulate_all", json={"timestamp": NIGHT}).json()
    cut = {o["outage"]: o for o in out["outages_disconnecting_load"]}
    assert "Bus_24 -> Bus_25 [L33]" in cut and cut["Bus_24 -> Bus_25 [L33]"]["buses"] == ["Bus_25"]
    assert cut["Bus_24 -> Bus_25 [L33]"]["load_mw"] > 0
    assert out["total_outages_disconnecting_load"] == len(cut) and out["system_n1_secure"] is False


def test_a_single_outage_that_cuts_load_off_is_not_secure(demo):
    out = demo.post("/api/contingency/simulate", json={"timestamp": NIGHT, "element_type": "line",
                                                       "element_index": 33}).json()
    assert out["total_violations"] == 0 and out["unsupplied_buses"] == ["Bus_25"]
    assert out["system_secure"] is False and "cuts off" in out["message"]


def test_violations_are_named_by_index_label_not_position():
    net = pp.create_empty_network()
    b = [pp.create_bus(net, vn_kv=20.0, index=i) for i in (10, 20, 30)]
    pp.create_ext_grid(net, b[0], vm_pu=1.0)
    pp.create_line(net, b[0], b[1], length_km=5.0, std_type="NA2XS2Y 1x95 RM/25 12/20 kV", index=7)
    pp.create_line(net, b[1], b[2], length_km=5.0, std_type="NA2XS2Y 1x95 RM/25 12/20 kV", index=9)
    pp.create_load(net, b[2], p_mw=9.0, q_mvar=3.0)
    rows, _ = ca.contingency_assessment_single(net, "line", 99, "t", vm_lower=0.99, vm_upper=1.1,
                                               max_line_loading_pct=10.0, max_trafo_loading_pct=10.0)
    assert set(rows.loc[rows.violation_type == "bus_vm_pu", "violation_element_index"]) <= {10, 20, 30}
    assert set(rows.loc[rows.violation_type == "line_loading", "violation_element_index"]) == {7, 9}


def test_critical_outages_are_ranked_by_the_excess_beyond_the_band():
    sweep = {"violations": [
        {"outage_cause": "A", "outage_type": "line", "outage_element_index": 1, "violation_type": "bus_vm_pu",
         "value": 0.94},
        {"outage_cause": "B", "outage_type": "line", "outage_element_index": 2, "violation_type": "line_loading",
         "value": 103.0}]}
    ranked = mb._critical_outages(sweep, 100.0, (0.95, 1.05))
    assert [r[0] for r in ranked] == ["B", "A"]                  # 3 % over, against 0.01 p.u. under
    assert ranked[1][3] == pytest.approx(1.0)


def test_one_of_two_parallel_circuits_gets_its_own_admittances():
    net = pp.create_empty_network()
    b0, b1 = pp.create_bus(net, vn_kv=20.0), pp.create_bus(net, vn_kv=20.0)
    pp.create_ext_grid(net, b0)
    first = pp.create_line(net, b0, b1, length_km=2.0, std_type="NA2XS2Y 1x95 RM/25 12/20 kV")
    pp.create_line(net, b0, b1, length_km=2.0, std_type="NA2XS2Y 1x95 RM/25 12/20 kV")
    pp.create_load(net, b1, p_mw=2.0, q_mvar=0.5)
    db = mb.nl.build_ybus_full(net)
    net_k = pp.from_json_string(pp.to_json(net))
    net_k.line.at[first, "in_service"] = False
    after = mb._outage_admittances(db, net, net_k, "line", first)
    key = next(k for k in db["Yft_r"] if k[0] == (b0, b1))
    assert after is not db and after["Yft_r"][key] == pytest.approx(db["Yft_r"][key] / 2)
    single = pp.from_json_string(pp.to_json(net))
    single.line.drop(index=first, inplace=True)
    assert mb._outage_admittances(db, single, single, "line", single.line.index[0]) is db   # no parallel: as is


@pytest.mark.skipif(shutil.which("ipopt") is None, reason="IPOPT not installed")
def test_secure_options_skip_only_outages_that_cut_load_off(demo):
    out = demo.post("/api/flexibility/optimize", json={"timestamp": PEAK, "options": True,
                                                       "n1": "preventive"}).json()
    skipped = {s["outage"]: s["reason"] for s in out["n1_study"]["skipped"]}
    assert "without supply" in skipped["Bus_24 -> Bus_25 [L33]"]
    # L12 and L15 only disconnect a unit: not skipped for supply, whatever else may skip them.
    assert all("without supply" not in skipped.get(n, "") for n in ("Bus_8 -> Bus_10 [L12]", "Bus_11 -> Bus_12 [L15]"))
