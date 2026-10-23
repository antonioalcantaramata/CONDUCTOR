"""The key performance indicators on another network: the bundled IEEE
14-bus profile (`backend_client`). The demo's own tests are in test_kpis.py."""

import shutil

import pytest


@pytest.mark.skipif(shutil.which("ipopt") is None, reason="IPOPT not installed")
def test_the_indicators_on_ieee14(backend_client):
    import main_backend as mb
    r = backend_client.post("/api/kpi/evaluate", json={"timestamp": mb.app_data["timestamps"][0]})
    assert r.status_code == 200, r.text
    out = r.json()
    values = {row["key"]: (row["before"], row["after"]) for row in out["indicators"]}
    assert values["violations"][0] is not None and values["kpi_1_target_demand_flex_pct"][0] is not None
    assert out["schedule"]["n1_outages_tested"] > 0
    if out["after_is"] != "none":
        units = out["deliverable"]["with_action"]["units"]
        assert units and all(u["spare_mw"] is None or u["up_mw"] is None or u["up_mw"] <= u["spare_mw"] + 1e-3
                             for u in units.values())
