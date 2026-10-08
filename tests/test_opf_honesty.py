"""A capped optimisation that cannot meet its cap is reported as failed
(code review 2.1).

It used to return status "optimal" and "Grid restored at slack cap ±10 MW"
with the uncapped dispatch — 190 MW of import — as the answer, cached for the
envelope tool. Needs the IPOPT solver; skipped where it is not installed (CI).
"""

import shutil

import pytest

import main_backend as mb

pytestmark = pytest.mark.skipif(shutil.which("ipopt") is None, reason="IPOPT not installed")


def test_an_unmeetable_cap_is_a_failed_result(backend_client):
    ts = mb.app_data["timestamps"][40]
    r = backend_client.post("/api/flexibility/optimize",
                            json={"timestamp": ts, "slack_max_mw": 10}).json()
    assert r["status"] == "cap_infeasible" and r["feasible"] is False
    assert r["slack_cap_respected"] is False and r["activated_resources"] == []
    assert r["uncapped_reference_dispatch"]
    assert "Grid restored" not in r["message"]


def test_a_met_cap_says_so(backend_client):
    ts = mb.app_data["timestamps"][40]
    r = backend_client.post("/api/flexibility/optimize", json={"timestamp": ts}).json()
    assert r["status"] == "optimal" and r["feasible"] is True and r["slack_cap_respected"] is True


def test_the_optimiser_uses_the_networks_band_by_default(backend_client):
    ts = mb.app_data["timestamps"][40]
    r = backend_client.post("/api/flexibility/optimize", json={"timestamp": ts}).json()
    assert (r["opf_vm_lower_used"], r["opf_vm_upper_used"]) == (
        mb.app_data["default_vm_lower"], mb.app_data["default_vm_upper"])
    asked = backend_client.post("/api/flexibility/optimize",
                                json={"timestamp": ts, "opf_vm_upper": 1.045}).json()
    assert asked["opf_vm_upper_used"] == 1.045


class TestInfeasibilityIsExplained:
    """A failed OPF says why (diagnosis re-solve with the grid voltage free).

    Found at 2000-01-03 14:15 on the old default data, where the OPF answered
    only "Infeasible.". The calibrated default network solves that hour, so
    the shortage is made here with 15 % more load: local reactive power runs
    out at the grid connection's 1.02 p.u., and a raise inside the band fixes it.
    """

    def test_short_of_local_reactive_power(self, backend_client):
        r = backend_client.post("/api/flexibility/optimize", json={
            "timestamp": "2000-01-03 14:15:00", "load_scaling_factor": 1.15}).json()
        assert r["feasible"] is False and r["activated_resources"] == []
        d = r["infeasibility_diagnosis"]
        assert d["solvable_with_grid_voltage_free"] is True
        assert d["grid_voltage_fixed_pu"] < d["grid_voltage_needed_pu"] <= d["voltage_band_pu"][1]
        assert d["reactive_sources_at_limit"]
        assert "tap-changer" in r["message"] and r["message"] != "Infeasible."

    def test_beyond_either_lever(self, backend_client):
        """Losing line 0 (the main import line): neither the grid voltage nor
        lifting branch limits — even both — gives a dispatch."""
        r = backend_client.post("/api/contingency/optimize", json={
            "element_type": "line", "element_index": 0,
            "timestamp": "2000-01-04 03:15:00"}).json()
        assert r["feasible"] is False
        d = r["infeasibility_diagnosis"]
        assert d["solvable_with_grid_voltage_free"] is False and d["grid_voltage_needed_pu"] is None
        assert d["binding_limit"] == "beyond_voltage_and_branch_levers"
        assert "even both together" in r["message"]

    def test_line_loading_is_named(self, backend_client):
        """With branches allowed only 30 % of their rating (a 0.09 margin on
        I²), loading is the binding limit and the branches carrying more than
        allowed are named. Judged against the OPF's own bound, not the rating:
        checking against 100 % of the rating found nothing and fell through to
        "beyond either lever"."""
        r = backend_client.post("/api/flexibility/optimize", json={
            "timestamp": "2000-01-04 03:15:00", "opf_current_safety_margin": 0.09}).json()
        assert r["feasible"] is False
        d = r["infeasibility_diagnosis"]
        assert d["binding_limit"] in ("branch_loading", "voltage_support_and_branch_loading")
        assert d["branch_overloads"] and all(
            o["loading_pct_of_rating"] > o["allowed_pct_of_rating"] for o in d["branch_overloads"])
        assert d["branch_overloads"][0]["allowed_pct_of_rating"] == pytest.approx(30.0, abs=0.1)
        assert "[L" in d["branch_overloads"][0]["branch"]  # the network's own branch name
        assert "line loading" in r["message"] or "overloaded" in r["message"]

    def test_a_feasible_solve_carries_no_diagnosis(self, backend_client):
        ts = mb.app_data["timestamps"][0]
        r = backend_client.post("/api/flexibility/optimize", json={"timestamp": ts}).json()
        assert r["feasible"] is True and "infeasibility_diagnosis" not in r
