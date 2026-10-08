"""The Data tab: the loaded measurements and forecasts as they are
(`llm_agent/agent/data_view.py`, `/api/data/series`)."""

import pytest

from llm_agent.agent import data_view as dv

MEASURED = {"timestamps": ["t1", "t2", "t3"],
            "totals": {"consumption": [100.0, 110.0, 120.0], "production": [10.0, 20.0, 0.0]},
            "substations": {"A": {"consumption": [40.0, 41.0, 42.0], "production": [0.0, 0.0, 0.0]}}}
FORECAST = {"timestamps": ["t2", "t3", "t4"],
            "totals": {"consumption": [121.0, 108.0, 130.0], "production": [15.0, 0.0, 5.0]},
            "substations": {"A": {"consumption": [None, 40.0, 44.0], "production": [0.0, 0.0, 0.0]}}}


def test_values_per_quantity():
    assert dv.values(MEASURED, dv.TOTAL, "net") == [90.0, 90.0, 120.0]
    assert dv.values(MEASURED, "A", "consumption") == [40.0, 41.0, 42.0]
    assert dv.values(MEASURED, "missing", "consumption") == []
    assert dv.values(None, dv.TOTAL, "consumption") == []


def test_forecast_error_on_shared_ticks_only():
    err = dv.forecast_error(MEASURED, FORECAST, dv.TOTAL, "consumption")
    assert (err["n"], err["first"], err["last"]) == (2, "t2", "t3")
    assert err["bias"] == pytest.approx(-0.5) and err["mae"] == pytest.approx(11.5)
    assert err["mape_pct"] == pytest.approx(100 * (11 / 110 + 12 / 120) / 2, abs=0.01)


def test_missing_values_are_skipped_not_zero():
    err = dv.forecast_error(MEASURED, FORECAST, "A", "consumption")
    assert err["n"] == 1 and err["mae"] == pytest.approx(2.0)


def test_no_overlap_no_error():
    assert dv.forecast_error(MEASURED, {"timestamps": ["t9"], "totals": {"consumption": [1.0],
                                                                         "production": [0.0]}},
                             dv.TOTAL, "consumption") is None
    assert dv.forecast_error(MEASURED, None, dv.TOTAL, "consumption") is None


def test_figure_shades_the_overlap_and_marks_the_operating_point():
    fig = dv.figure({"measured": MEASURED, "forecast": FORECAST}, [dv.TOTAL], "consumption", marker="t2")
    assert [t.line.dash for t in fig.data] == ["solid", "dash"]
    kinds = [s.type for s in fig.layout.shapes]
    assert "rect" in kinds and "line" in kinds


def test_the_endpoint_serves_both_datasets(backend_client):
    m = backend_client.post("/api/data/series", json={"data_source": "measurements",
                                                      "substations": ["Bus_3"]}).json()
    f = backend_client.post("/api/data/series", json={"data_source": "forecasts"}).json()
    assert m["data_source"] == "measurements" and len(m["timestamps"]) == m["n_total"]
    assert len(m["substations"]["Bus_3"]["consumption"]) == len(m["timestamps"])
    assert "Bus_3" in m["available_substations"]
    assert set(m["timestamps"]) & set(f["timestamps"])  # the default forecast overlaps


def test_long_series_are_thinned(backend_client):
    r = backend_client.post("/api/data/series", json={"max_points": 100}).json()
    assert len(r["timestamps"]) <= 100 and r["stride"] > 1 and r["n_total"] > 100
