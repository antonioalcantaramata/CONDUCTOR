"""The key performance indicators on another network: the CIGRE MV
distribution demo (static generators only, radial feeders). The IEEE 30
demo's own tests are in test_kpis.py."""

import shutil

import pytest


@pytest.fixture(scope="module")
def cigre_client(tmp_path_factory):
    """The real backend, in-process, on the CIGRE MV profile — storage in tmp."""
    pytest.importorskip("fastapi.testclient")
    import main_backend as mb
    from fastapi.testclient import TestClient

    scratch = tmp_path_factory.mktemp("cigre")
    patch = pytest.MonkeyPatch()
    patch.setenv("GRID_PROFILE", "cigre_mv")
    for name, sub in (("SYSTEMS_DIR", "systems"), ("DATA_FILES_DIR", "data_files")):
        patch.setattr(mb, name, str(scratch / sub))
    for name in ("_LAST_UPLOAD_SENTINEL", "_LAST_TIMESERIES_SENTINEL", "_LAST_FORECAST_SENTINEL",
                 "_PROPOSALS_FILE", "_ASSUMPTIONS_FILE", "_SCENARIO_FILE"):
        patch.setattr(mb, name, str(scratch / f"{name}.json"))
    saved = dict(mb.app_data)
    with TestClient(mb.app) as client:
        yield client
    mb.app_data.clear()
    mb.app_data.update(saved)
    patch.undo()


@pytest.mark.skipif(shutil.which("ipopt") is None, reason="IPOPT not installed")
def test_the_indicators_on_cigre_mv(cigre_client):
    import main_backend as mb
    ts = mb.app_data["timestamps"][0]

    def evaluate():
        r = cigre_client.post("/api/kpi/evaluate", json={"timestamp": ts})
        assert r.status_code == 200, r.text
        return r.json()

    out = evaluate()
    values = {row["key"]: (row["before"], row["after"]) for row in out["indicators"]}
    assert values["violations"][0] is not None and values["kpi_1_target_demand_flex_pct"][0] is not None
    assert out["schedule"]["n1_outages_tested"] > 0
    # The benchmark marks every resource not controllable: nothing to deliver.
    assert out["deliverable"]["with_action"] is None
    held = ["CHP diesel 1", "Fuel cell 1", "Residential fuel cell 1", "Residential fuel cell 2"]
    assert cigre_client.post("/api/generators", json={"changes": [
        {"unit": u, "dispatchable": True} for u in held]}).status_code == 200
    try:
        out = evaluate()
    finally:
        cigre_client.post("/api/generators", json={"reset": True})
    units = out["deliverable"]["with_action"]["units"]
    assert units and set(units) <= set(held)
    assert all(u["spare_mw"] is None or u["up_mw"] is None or u["up_mw"] <= u["spare_mw"] + 1e-3
               for u in units.values())


def test_the_cigre_demo_starts_with_its_own_units(cigre_client):
    """Its static generators are kept at every tick (a profile without `gen`
    units once had them rebuilt from the loads, each injecting minus the bus's
    production) and follow the source the network states for them."""
    import main_backend as mb
    net = mb.app_data["net"]
    assert len(net.sgen) == 13 and "Sgen R1" not in set(net.sgen.name)
    assert (net.sgen.p_mw >= 0).all()
    night = next(t for t in mb.app_data["timestamps"] if t.endswith("02:00:00"))
    snap = cigre_client.post("/api/network/snapshot", json={"timestamp": night}).json()
    # Measurements are per bus: where PV is a bus's only unit, it gives nothing at night.
    by_bus: dict = {}
    for g in snap["generators"]:
        by_bus.setdefault(g["bus_index"], []).append(g)
    pv_only = [gs[0] for gs in by_bus.values() if len(gs) == 1 and str(gs[0]["name"]).startswith("PV")]
    assert len(pv_only) == 5 and all(g["Pg_mw"] == pytest.approx(0.0, abs=1e-9) for g in pv_only)
