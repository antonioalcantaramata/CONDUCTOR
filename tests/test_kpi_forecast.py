"""KPI forecast (`/api/kpi/forecast`) — code review 2.11.

A tick whose optimisation failed was reported as KPI-1 = 0 % ("no flexibility
available"); a tick whose model raised was dropped, so the 24 hours silently
shrank; and `fixed_setpoints` never reached either solve.
"""

import types

import pytest

pytest.importorskip("fastapi.testclient")

import main_backend as mb  # noqa: E402
from pyomo.opt import TerminationCondition  # noqa: E402


def _failing_solve(calls):
    def fake(*args, **kwargs):
        calls.append(kwargs)
        res = types.SimpleNamespace(solver=types.SimpleNamespace(
            termination_condition=TerminationCondition.infeasible))
        return res, None
    return fake


def test_failed_ticks_are_none_not_zero_and_setpoints_are_passed(backend_client, monkeypatch):
    calls = []
    monkeypatch.setattr(mb.fe, "optimization_model_base", _failing_solve(calls))
    r = backend_client.post("/api/kpi/forecast", json={
        "timestamp": mb.app_data["timestamps"][0], "fixed_setpoints": {"Gen_bus1": 10.0}}).json()
    assert r["n_ticks"] == len(r["forecast"]) == 96
    assert r["n_failed"] == 96 and "not 0 %" in r["note"]
    assert all(f["kpi_1_target_demand_flex_pct"] is None for f in r["forecast"])
    assert {f["status"] for f in r["forecast"]} == {"infeasible"}
    assert calls and all(c.get("fixed_setpoints") == {"Gen_bus1": 10.0} for c in calls)


def test_a_model_error_keeps_its_tick(backend_client, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("model build failed")
    monkeypatch.setattr(mb.fe, "optimization_model_base", boom)
    r = backend_client.post("/api/kpi/forecast", json={"timestamp": mb.app_data["timestamps"][0]}).json()
    assert len(r["forecast"]) == 96 and {f["status"] for f in r["forecast"]} == {"error"}


def test_the_chart_shows_gaps_and_says_so():
    pytest.importorskip("plotly")
    from llm_agent.agent.renderers import render_kpi_forecast
    fig = render_kpi_forecast({"forecast": [
        {"timestamp": "t1", "kpi_1_target_demand_flex_pct": 40.0, "status": "optimal"},
        {"timestamp": "t2", "kpi_1_target_demand_flex_pct": None, "status": "infeasible"},
        {"timestamp": "t3", "kpi_1_target_demand_flex_pct": 42.0, "status": "optimal"}]})
    assert list(fig.data[0].y) == [40.0, None, 42.0]
    assert "1 of 3 tick(s) without a solution" in fig.layout.title.text
