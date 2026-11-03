"""
The CIGRE MV demo (backend/network_profiles/cigre_mv.yaml): a distribution
benchmark whose units are all static generators, typed by the network.

What it needed that the transmission demos did not: a profile network of
static generators only keeps them at every tick (they were rebuilt from the
loads, injecting minus the production), each unit's synthetic output follows
the source the network states for it, and the profile states its loads' own
power factor.
"""

import pandapower.networks as pn
import pytest

import synthetic_timeseries as st


@pytest.fixture(scope="module")
def cigre_client(tmp_path_factory):
    pytest.importorskip("fastapi.testclient")
    import main_backend as mb
    from fastapi.testclient import TestClient

    scratch = tmp_path_factory.mktemp("cigre")
    patch = pytest.MonkeyPatch()
    patch.setenv("GRID_PROFILE", "cigre_mv")
    for name in ("_LAST_UPLOAD_SENTINEL", "_LAST_TIMESERIES_SENTINEL", "_LAST_FORECAST_SENTINEL",
                 "_PROPOSALS_FILE", "_ASSUMPTIONS_FILE", "_SCENARIO_FILE"):
        patch.setattr(mb, name, str(scratch / f"{name}.json"))
    saved = dict(mb.app_data)
    with TestClient(mb.app) as client:
        yield client, mb
    mb.app_data.clear()
    mb.app_data.update(saved)
    patch.undo()


def test_each_unit_follows_the_source_the_network_states():
    net = pn.create_cigre_network_mv(with_der="all")
    shapes = {("sgen", int(i)): ("solar" if t == "PV" else "wind" if t == "WP" else "flat")
              for i, t in net.sgen["type"].items()}
    ts, meas = st.generate(net, n_days=1, generation_profile="flat", unit_shapes=shapes, stress_events=False)
    pv_only = "Bus 3"                                     # PV 3 is the only unit there
    night = meas["2000-01-01 02:00:00"].set_index("substation_name").loc[pv_only, "production"]
    noon = meas["2000-01-01 12:00:00"].set_index("substation_name").loc[pv_only, "production"]
    assert night == 0.0 and noon > 0.0


def test_without_unit_shapes_the_week_is_unchanged():
    net = pn.case30()
    a = st.generate(net, n_days=1, seed=3)
    b = st.generate(net, n_days=1, seed=3, unit_shapes=None)
    assert all(a[1][t].equals(b[1][t]) for t in a[0])


def test_the_demo_keeps_its_own_units_and_its_loads_power_factor(cigre_client):
    client, mb = cigre_client
    net = mb.app_data["net"]
    assert {"PV 3", "WKA 7", "CHP diesel 1"} <= set(net.sgen["name"])            # not rebuilt from loads
    assert (net.sgen["p_mw"] >= 0).all()
    pf = client.get("/api/assumptions").json()["load_power_factor"]
    assert pf["value"] == 0.97 and pf["source"] == "the network's profile"
    summary = client.get("/api/network/semantics", params={"about": "summary"}).json()
    assert summary["units_by_source"] == {"solar": 8, "wind": 1, "thermal": 4, "storage": 2}


def test_the_week_sags_at_the_evening_peaks_only(cigre_client):
    client, mb = cigre_client
    hours = []
    for t in mb.app_data["timestamps"][::8]:                                       # every 2 h
        r = client.post("/api/grid/rsa", json={"timestamp": t, "vm_lower_pu": 0.95, "vm_upper_pu": 1.05}).json()
        if r.get("total_violations"):
            hours.append(int(t[11:13]))
    assert 0 < len(hours) <= 10 and all(16 <= h <= 21 for h in hours)


def test_the_agent_can_ask_what_a_line_supplies(cigre_client):
    client, _ = cigre_client
    out = client.get("/api/network/semantics", params={"about": "around", "element": "Line 2-3"}).json()
    assert out["supply"]["feeder"].startswith("Line 1-2")
    assert "Bus 3" in out["without_it"]["buses_cut_off"] and out["without_it"]["open_ties_that_could_restore"]


def test_a_profile_no_demo_uses_is_not_called_a_demo():
    import scenarios as scn
    assert scn.demo_for_profile("cigre_mv") == "cigre_mv" and scn.demo_for_profile("pglib_case30") == "ieee30"
    assert scn.demo_for_profile("my_own_grid") is None


def test_storage_is_in_the_optimisation_as_the_power_flow_has_it():
    import flex_engine as fe
    net = pn.create_cigre_network_mv(with_der="all")
    demand = fe._fixed_demand(net)
    assert sum(p for p, _ in demand.values()) == pytest.approx(0.8)                # two batteries charging


def test_open_ties_are_not_branches_of_the_optimisation():
    import flex_engine as fe
    net = pn.create_cigre_network_mv(with_der="all")
    branches = fe._branch_current_data(net)
    assert (6, 7) not in branches and (7, 6) not in branches                       # Line 6-7, open at Bus 7
    assert set(fe._lines_switched_open(net)) == {12, 13, 14}


@pytest.mark.skipif(__import__("shutil").which("ipopt") is None, reason="IPOPT not installed")
def test_the_evening_sag_is_not_called_restored(cigre_client):
    # Observed: with the batteries left out of the optimisation, 18:00 on
    # 3 January was "Grid restored" while the power flow stayed below 0.95 p.u.
    client, _ = cigre_client
    out = client.post("/api/flexibility/optimize", json={"timestamp": "2000-01-03 18:00:00"}).json()
    assert "restored" not in (out.get("message") or "").lower()
    assert "tap-changer" in out["message"]
