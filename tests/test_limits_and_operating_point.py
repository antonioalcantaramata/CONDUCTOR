"""
Limits, datasets and time windows travel intact (code review batch A).

Each test reproduces a defect found in the October 2026 review — a valid
solver given the wrong limits, dataset or moment, returning a confident answer
— and pins the fix. The endpoint tests run the real backend in-process on the
bundled IEEE 14-bus profile (about 1.5 s to start, no IPOPT needed), isolated
from any network or data uploaded on this machine.
"""

import pytest

pytest.importorskip("fastapi.testclient")

import main_backend as mb  # noqa: E402

from llm_agent.agent import loop, tools  # noqa: E402
from llm_agent.agent.validators import (  # noqa: E402
    TurnConsistency,
    TurnOperatingPoint,
    compares_datasets,
    requested_limits,
    validate_call,
)

NETWORK_LIMITS = {"vm_upper_pu": 1.06, "vm_lower_pu": 0.94,
                  "max_line_loading_pct": 100.0, "max_trafo_loading_pct": 100.0}


@pytest.fixture(scope="module")
def client(backend_client):
    return backend_client


# ---------------------------------------------------------------------------
# 1.1 / 1.2 — limits
# ---------------------------------------------------------------------------


class TestLimits:
    def test_unset_limits_are_the_networks_own(self, client):
        assert client.post("/api/grid/rsa", json={}).json()["thresholds_used"] == NETWORK_LIMITS

    def test_explicit_generic_values_are_honoured(self, client):
        """0.95 / 1.05 / 90 used to mean "not specified" and were replaced."""
        used = client.post("/api/grid/rsa", json={
            "vm_lower_pu": 0.95, "vm_upper_pu": 1.05, "max_line_loading_pct": 90,
        }).json()["thresholds_used"]
        assert (used["vm_lower_pu"], used["vm_upper_pu"], used["max_line_loading_pct"]) == (0.95, 1.05, 90)
        # The one not sent still comes from the network.
        assert used["max_trafo_loading_pct"] == 100.0

    def test_n1_judges_against_the_same_limits_as_rsa(self, client):
        """N-1 used the generic 0.95/1.05/90 while RSA used the network's."""
        ts = mb.app_data["timestamps"][40]
        rsa = client.post("/api/grid/rsa", json={"timestamp": ts}).json()
        n1 = client.post("/api/contingency/simulate_all", json={"timestamp": ts}).json()
        one = client.post("/api/contingency/simulate", json={
            "timestamp": ts, "element_type": "line", "element_index": 0}).json()
        assert rsa["thresholds_used"] == n1["thresholds_used"] == one["thresholds_used"]

    def test_internal_requests_fill_limits_too(self, monkeypatch):
        monkeypatch.setitem(mb.app_data, "default_vm_lower", 0.94)
        monkeypatch.setitem(mb.app_data, "default_vm_upper", 1.06)
        assert mb._thresholds_used(mb.RobustFlexibilityRequest())["vm_lower_pu"] == 0.94
        assert mb._effective_thresholds(mb.WorstCaseRequest(vm_upper_pu=1.05))[1] == 1.05


class TestWrappersSendOnlyChosenLimits:
    @pytest.fixture
    def sent(self, monkeypatch):
        bodies = []

        class _Response:
            def raise_for_status(self):
                pass

            def json(self):
                return {}

        class _Client:
            def __init__(self, *a, **k):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def post(self, url, json=None):
                bodies.append(json)
                return _Response()

        monkeypatch.setattr(tools.httpx, "Client", _Client)
        return bodies

    def test_unset_limits_are_not_sent(self, sent):
        tools.run_rsa()
        assert not set(sent[-1]) & tools._LIMIT_KEYS

    def test_explicit_limits_are_sent(self, sent):
        tools.run_rsa(vm_upper_pu=1.05)
        assert sent[-1]["vm_upper_pu"] == 1.05 and "vm_lower_pu" not in sent[-1]

    def test_opf_bounds_are_not_sent_when_unset(self, sent):
        tools.optimize_flexibility()
        assert "opf_vm_upper" not in sent[-1]


# ---------------------------------------------------------------------------
# 1.3 — limits read from the question
# ---------------------------------------------------------------------------


class TestRequestedLimits:
    @pytest.mark.parametrize("text", [
        "What is the minimum voltage if load is scaled by 1.15?",
        "maximum line loading at load scaling 1.2",
        "minimum power factor 0.95 at bus 3",
        "what if load is 1.1 times higher at Bus 3?",
        "Is Bus 3 voltage at 0.937 p.u. a problem?",
    ])
    def test_not_limits(self, text):
        assert requested_limits(text) == {}

    @pytest.mark.parametrize("text,expected", [
        # Asking which buses are below 0.96 sets a floor, not a ceiling.
        ("Are any buses below 0.96 p.u.?", {"vm_lower_pu": 0.96}),
        ("Which buses are above 1.04?", {"vm_upper_pu": 1.04}),
        ("check against a band between 0.95 and 1.05 p.u.",
         {"vm_lower_pu": 0.95, "vm_upper_pu": 1.05}),
        ("run with load_scaling_factor 1.15 and max voltage 1.04", {"vm_upper_pu": 1.04}),
    ])
    def test_limits(self, text, expected):
        assert requested_limits(text) == expected

    def test_an_inverted_band_is_kept_for_the_validator_to_refuse(self):
        limits = requested_limits("a lower voltage limit of 1.05 and an upper limit of 0.95")
        assert limits == {"vm_lower_pu": 1.05, "vm_upper_pu": 0.95}
        assert validate_call("run_rsa", limits).rejected


# ---------------------------------------------------------------------------
# 1.4 — datasets
# ---------------------------------------------------------------------------


class TestDataSource:
    def test_forecast_is_read_even_where_it_overlaps_measurements(self, client):
        t0 = mb.app_data["timestamps"][10]
        saved = mb.app_data["forecasts"], mb.app_data["forecast_timestamps"]
        forecast = mb.app_data["measurements"][t0].copy()
        forecast["consumption"] = forecast["consumption"] * 1.6
        mb.app_data["forecasts"], mb.app_data["forecast_timestamps"] = {t0: forecast}, [t0]
        try:
            measured = client.post("/api/grid/rsa", json={"timestamp": t0}).json()
            predicted = client.post("/api/grid/rsa", json={
                "timestamp": t0, "data_source": "forecasts"}).json()
        finally:
            mb.app_data["forecasts"], mb.app_data["forecast_timestamps"] = saved
        assert predicted["total_load_mw"] > measured["total_load_mw"] * 1.4
        assert (measured["data_source"], predicted["data_source"]) == ("measurements", "forecasts")

    def test_snapshot_reports_power_flow_results_not_zeros(self, client):
        """Condensers holding their bus voltage were reported at 0.0 Mvar, the
        external grid at 0.0 Mvar and the import without losses."""
        snap = client.post("/api/network/snapshot", json={}).json()
        assert snap["power_flow_converged"] is True
        condensers = [g for g in snap["generators"] if g.get("vm_setpoint_pu") is not None]
        assert condensers and any(abs(g["Qg_mvar"]) > 0.5 for g in condensers)
        assert snap["ext_grid"]["Q_mvar"] is not None
        totals = snap["totals"]
        assert totals["total_losses_mw"] > 0
        assert totals["net_import_mw"] == pytest.approx(
            totals["total_load_mw"] - totals["total_generation_mw"] + totals["total_losses_mw"], abs=0.05)

    def test_results_name_their_dataset(self, client):
        ts = mb.app_data["timestamps"]
        hist = client.post("/api/rsa/historical_risk", json={
            "window_start": ts[0], "window_end": ts[3], "target": "bus:Bus_3"}).json()
        snap = client.post("/api/network/snapshot", json={}).json()
        assert hist["data_source"] == snap["data_source"] == "measurements"

    def test_asking_to_compare_datasets_is_not_drift(self):
        assert compares_datasets("Compare the forecast with the measured load at noon")
        assert not compares_datasets("Is the grid secure in the forecast?")
        check = TurnConsistency(datasets_compared=True)
        check.observe("run_rsa", {"data_source": "measurements"})
        assert check.observe("run_rsa", {"data_source": "forecasts"}) == []
        drift = TurnConsistency()
        drift.observe("run_rsa", {"data_source": "measurements"})
        assert drift.observe("run_rsa", {"data_source": "forecasts"})


# ---------------------------------------------------------------------------
# 1.5 / 4.7 — the over-time scan
# ---------------------------------------------------------------------------


class TestScanOverTime:
    def test_schema_accepts_limits_so_the_turn_can_pass_them(self):
        assert {"vm_upper_pu", "vm_lower_pu", "max_line_loading_pct",
                "max_trafo_loading_pct"} <= loop._TOOL_ARGS["scan_rsa_over_time"]

    def test_stops_at_the_end_of_the_data(self, monkeypatch):
        replies = iter([
            {"new_timestamp": "2000-01-07 23:45:00"},
            {"status": "finished", "message": "No more timestamps"},
        ])

        def fake_post(endpoint, body):
            if endpoint == "/api/time/advance":
                return next(replies)
            return {"total_violations": 0, "all_voltages": [{"vm_pu": 1.0}],
                    "all_line_loading": [{"loading_percent": None}],
                    "thresholds_used": NETWORK_LIMITS}

        monkeypatch.setattr(tools, "_post", fake_post)
        result = tools.scan_rsa_over_time(n_steps=5)
        assert result["timestamps"] == ["2000-01-07 23:45:00"]
        assert result["reached_end_of_data"] is True
        assert result["thresholds_used"] == NETWORK_LIMITS
        summary = loop._prepare_tool_result_for_model("scan_rsa_over_time", result)
        assert summary["thresholds_used"] == NETWORK_LIMITS and summary["reached_end_of_data"]

    def test_advance_past_the_end_keeps_the_current_time(self, monkeypatch):
        monkeypatch.setattr(tools, "_post", lambda e, b: {"status": "finished"})
        monkeypatch.setattr(tools, "_get", lambda e: {"current_timestamp": "2000-01-07 23:45:00"})
        result = tools.advance_timestamp(steps=3)
        assert result["current_timestamp"] == "2000-01-07 23:45:00"
        assert result["reached_end_of_data"] is True


# ---------------------------------------------------------------------------
# 1.6 / 1.7 — time windows and element names
# ---------------------------------------------------------------------------


class TestWindowsAndNames:
    @pytest.mark.parametrize("name,expected", [
        ("Bus_3", "Bus_3"),               # the name RSA reports
        ("3", "Bus_3"),                   # an index
        ("Bus_0 -> Bus_1", "Bus_0 -> Bus_1 [L0]"),  # unique prefix of a line name
    ])
    def test_names_other_tools_report_resolve(self, client, name, expected):
        etype = "line" if "->" in name else "bus"
        r = client.post("/api/grid/element_timeseries", json={
            "element_type": etype, "element_name": name, "n_steps": 2})
        assert r.status_code == 200 and r.json()["element_name"] == expected

    def test_an_ambiguous_name_is_refused_with_candidates(self, client):
        r = client.post("/api/grid/element_timeseries", json={
            "element_type": "bus", "element_name": "Bus", "n_steps": 2})
        assert r.status_code == 400 and "Bus_3" in r.json()["detail"]

    def test_an_unmatched_window_is_an_error_not_the_clock(self, client):
        # By index, which resolved before the fix too, so only the window is tested.
        r = client.post("/api/grid/element_timeseries", json={
            "element_type": "bus", "element_name": "3", "start_timestamp": "2022-01-02"})
        assert r.status_code == 404 and "start_timestamp" in r.json()["detail"]

    def test_end_before_start_is_refused(self, client):
        r = client.post("/api/grid/element_timeseries", json={
            "element_type": "bus", "element_name": "Bus_3",
            "start_timestamp": "2000-01-03", "end_timestamp": "2000-01-02"})
        assert r.status_code == 400

    def test_scenario_scan_window_is_strict_too(self, client):
        assert client.post("/api/rsa/scan_scenarios", json={"start_timestamp": "2022-05"}).status_code == 404


# ---------------------------------------------------------------------------
# 1.8 — a rejected value is not inherited
# ---------------------------------------------------------------------------


class TestRollback:
    def test_rejected_value_is_withdrawn(self):
        turn = TurnOperatingPoint()
        kwargs, _ = turn.resolve("run_rsa", {"vm_upper_pu": 1.6}, {"vm_upper_pu"})
        assert validate_call("run_rsa", kwargs).rejected
        turn.rollback()
        retry, inherited = turn.resolve("run_rsa", {}, {"vm_upper_pu"})
        assert "vm_upper_pu" not in retry and not inherited

    def test_successful_value_is_still_inherited(self):
        turn = TurnOperatingPoint()
        turn.resolve("run_rsa", {"vm_upper_pu": 1.045}, {"vm_upper_pu"})
        later, _ = turn.resolve("simulate_all_contingencies", {}, {"vm_upper_pu"})
        assert later["vm_upper_pu"] == 1.045

    def test_a_limit_from_the_question_survives(self):
        turn = TurnOperatingPoint(requested={"vm_upper_pu": 1.04})
        turn.resolve("run_rsa", {"vm_upper_pu": 1.6}, {"vm_upper_pu"})
        turn.rollback()
        assert turn.resolve("run_rsa", {}, {"vm_upper_pu"})[0]["vm_upper_pu"] == 1.04


class TestOptimiserBounds:
    """The OPF's voltage band defaults to the network's own, like every
    assessment; it used to be a generic 0.95–1.05 whatever the network."""

    def test_unset_bounds_are_the_networks(self, monkeypatch):
        monkeypatch.setitem(mb.app_data, "default_vm_lower", 0.94)
        monkeypatch.setitem(mb.app_data, "default_vm_upper", 1.06)
        for request in (mb.FlexibilityRequest(), mb.KPIRequest(),
                        mb.ContingencyOptimizeRequest(element_type="line", element_index=0)):
            assert (request.opf_vm_lower, request.opf_vm_upper) == (0.94, 1.06)

    def test_an_explicit_bound_still_wins(self, monkeypatch):
        monkeypatch.setitem(mb.app_data, "default_vm_upper", 1.06)
        assert mb.FlexibilityRequest(opf_vm_upper=1.05).opf_vm_upper == 1.05
