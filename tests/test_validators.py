"""Parameter and result integrity checks.

The premise these defend: a validated solver given invalid parameters returns a
confidently wrong answer, so solver validity is necessary but not sufficient.
Schema validation catches type errors; these catch inverted voltage bands,
requests the loaded network cannot satisfy, and results that contradict
themselves.

The severity split is deliberate and load-bearing, so it is tested directly:
rejecting anything merely unusual would destroy the flexibility that makes the
agent useful, while letting an impossible request through wastes a solve and
produces a meaningless answer.
"""

import pytest

from llm_agent.agent.validators import (
    REJECT,
    WARN,
    TurnConsistency,
    requested_timestamps,
    TurnOperatingPoint,
    annotate,
    operating_context,
    validate_call,
    validate_result,
)

GRID = {
    "vm_lower": 0.95,
    "vm_upper": 1.05,
    "max_loading_pct": 90.0,
    "load_scaling_min": 0.0,
    "load_scaling_max": 4.0,
    "slack_max_mw_min": 0.0,
    "slack_max_mw_max": 70.0,
}


def codes(issues):
    return {i.code for i in issues}


def fields(issues):
    return {i.field for i in issues}


class TestAcceptsValidCalls:
    def test_empty_arguments_are_fine(self):
        assert not validate_call("run_rsa", {}, GRID).issues

    def test_a_normal_call_passes_cleanly(self):
        verdict = validate_call("run_rsa", {
            "vm_lower_pu": 0.95, "vm_upper_pu": 1.05,
            "load_scaling_factor": 1.2, "max_line_loading_pct": 90,
        }, GRID)
        assert not verdict.issues

    def test_probabilistic_defaults_pass(self):
        verdict = validate_call("run_probabilistic_rsa", {
            "n_samples": 200, "load_sigma": 0.05, "sgen_sigma": 0.0,
        }, GRID)
        assert not verdict.issues

    def test_robust_defaults_pass(self):
        verdict = validate_call("optimize_robust_flexibility", {
            "risk_target": 0.05, "n_samples": 200, "load_sigma": 0.05,
            "robust_method": "heuristic", "beta": 1e-3,
        }, GRID)
        assert not verdict.issues


class TestRejectsTheImpossible:
    def test_inverted_voltage_band(self):
        verdict = validate_call("run_rsa", {"vm_lower_pu": 1.05, "vm_upper_pu": 0.95}, GRID)
        assert verdict.rejected
        assert "inverted_range" in codes(verdict.rejections)

    def test_inverted_opf_envelope(self):
        verdict = validate_call("optimize_flexibility",
                                {"opf_vm_lower": 1.06, "opf_vm_upper": 1.02}, GRID)
        assert verdict.rejected

    def test_inverted_active_power_range(self):
        verdict = validate_call("compute_hosting_capacity",
                                {"p_min_mw": 10.0, "p_max_mw": 2.0}, GRID)
        assert verdict.rejected

    @pytest.mark.parametrize("value", [-0.1, 1.5])
    def test_sigma_outside_zero_to_one(self, value):
        assert validate_call("run_probabilistic_rsa", {"load_sigma": value}, GRID).rejected

    @pytest.mark.parametrize("value", [0.0, 1.0, -0.2, 3.0])
    def test_probability_must_be_strictly_between_zero_and_one(self, value):
        # A risk target of 0 or 1 is not a risk target.
        assert validate_call("optimize_robust_flexibility",
                             {"risk_target": value}, GRID).rejected

    @pytest.mark.parametrize("value", [0, -5])
    def test_sample_counts_must_be_positive(self, value):
        assert validate_call("run_probabilistic_rsa", {"n_samples": value}, GRID).rejected

    def test_negative_interconnection_capacity(self):
        assert validate_call("run_rsa", {"slack_max_mw": -10.0}, GRID).rejected

    @pytest.mark.parametrize("value", [0.0, 1.4])
    def test_power_factor_out_of_range(self, value):
        assert validate_call("optimize_flexibility",
                             {"opf_min_power_factor": value}, GRID).rejected

    def test_physically_implausible_voltage(self):
        verdict = validate_call("run_rsa", {"vm_upper_pu": 3.0}, GRID)
        assert verdict.rejected
        assert "implausible_voltage" in codes(verdict.rejections)

    def test_negative_generator_capacity(self):
        verdict = validate_call("optimize_flexibility",
                                {"pg_max_overrides": {"S2": -5.0}}, GRID)
        assert verdict.rejected
        assert "negative_capacity" in codes(verdict.rejections)

    def test_must_run_minimum_above_maximum(self):
        # No dispatch can satisfy both bounds.
        verdict = validate_call("optimize_flexibility", {
            "pg_min_overrides": {"S2": 8.0}, "pg_max_overrides": {"S2": 3.0},
        }, GRID)
        assert verdict.rejected


class TestChecksAgainstTheLoadedNetwork:
    def test_load_scaling_beyond_network_support(self):
        verdict = validate_call("run_rsa", {"load_scaling_factor": 9.0}, GRID)
        assert verdict.rejected

    def test_load_scaling_within_support_passes(self):
        assert not validate_call("run_rsa", {"load_scaling_factor": 3.5}, GRID).issues

    def test_slack_above_network_capacity(self):
        assert validate_call("run_rsa", {"slack_max_mw": 500.0}, GRID).rejected

    def test_limits_far_from_the_network_warn_but_run(self):
        verdict = validate_call("run_rsa", {"vm_upper_pu": 1.19}, GRID)
        assert not verdict.rejected
        assert "far_from_network_limits" in codes(verdict.warnings)

    def test_bounds_follow_the_network_not_a_constant(self):
        """The same value is valid on one network and refused on another."""
        permissive = {**GRID, "slack_max_mw_max": 999.0}
        assert validate_call("run_rsa", {"slack_max_mw": 500.0}, permissive).issues == ()
        assert validate_call("run_rsa", {"slack_max_mw": 500.0}, GRID).rejected


class TestWarnsWithoutBlocking:
    def test_unusual_voltage_warns(self):
        verdict = validate_call("run_rsa", {"vm_upper_pu": 1.25}, GRID)
        assert not verdict.rejected
        assert verdict.warnings

    def test_unknown_enum_warns_rather_than_rejects(self):
        # The backend is the authority on accepted values; wrongly refusing a
        # valid one would break a working capability.
        verdict = validate_call("run_rsa", {"data_source": "forcasts"}, GRID)
        assert not verdict.rejected
        assert "unknown_value" in codes(verdict.warnings)

    def test_known_enum_values_are_silent(self):
        for value in ("measurements", "forecasts"):
            assert not validate_call("run_rsa", {"data_source": value}, GRID).issues

    def test_unrecognised_parameters_are_ignored(self):
        # Unknown names are the schema's business, not ours.
        assert not validate_call("run_rsa", {"some_future_param": 42}, GRID).issues

    def test_non_numeric_values_do_not_crash(self):
        verdict = validate_call("run_rsa", {"load_sigma": None, "n_samples": "many"}, GRID)
        assert isinstance(verdict.issues, tuple)

    def test_booleans_are_not_treated_as_numbers(self):
        assert not validate_call("compute_historical_risk", {"parallel": True}, GRID).issues


class TestToolError:
    def test_error_names_the_offending_parameters(self):
        verdict = validate_call("run_rsa", {"vm_lower_pu": 1.05, "vm_upper_pu": 0.95}, GRID)
        err = verdict.as_tool_error("run_rsa")
        assert "run_rsa" in err["error"]
        assert "vm_lower_pu" in err["invalid_parameters"]
        assert err["problems"]

    def test_error_explains_the_problem_for_self_correction(self):
        verdict = validate_call("run_probabilistic_rsa", {"load_sigma": 5.0}, GRID)
        assert "between 0 and 1" in verdict.as_tool_error("run_probabilistic_rsa")["error"]


class TestResultSelfConsistency:
    def test_clean_result_produces_no_findings(self):
        assert validate_result("run_rsa", {
            "total_violations": 0,
            "violations": [],
            "all_voltages": [{"bus_name": "S1", "vm_pu": 1.01}],
            "thresholds_used": {"vm_lower_pu": 0.95, "vm_upper_pu": 1.05},
        }) == []

    def test_optimisation_reporting_success_outside_its_own_bounds(self):
        """The strongest available signal that a solve went wrong."""
        issues = validate_result("optimize_flexibility", {
            "status": "optimal",
            "opf_vm_lower_used": 0.95,
            "opf_vm_upper_used": 1.05,
            "bus_voltages_post_opf": [{"bus_name": "S2 10kV", "vm_pu": 1.09}],
        })
        assert "opf_bound_violation" in codes(issues)

    def test_excluded_buses_are_not_counted_against_the_solve(self):
        issues = validate_result("optimize_flexibility", {
            "status": "optimal",
            "opf_vm_lower_used": 0.95,
            "opf_vm_upper_used": 1.05,
            "bus_voltages_post_opf": [{"bus_name": "aux", "vm_pu": 1.30}],
            "excluded_buses_post_opf": [{"bus_name": "aux", "reason": "out_of_service"}],
        })
        assert issues == []

    def test_failed_solve_is_not_held_to_its_bounds(self):
        assert validate_result("optimize_flexibility", {
            "status": "infeasible",
            "opf_vm_lower_used": 0.95, "opf_vm_upper_used": 1.05,
            "bus_voltages_post_opf": [{"bus_name": "S2", "vm_pu": 1.20}],
        }) == []

    def test_violation_count_disagreeing_with_the_list(self):
        issues = validate_result("run_rsa", {
            "total_violations": 3,
            "violations": [{"bus": "S1"}],
        })
        assert "inconsistent_count" in codes(issues)

    def test_all_clear_contradicted_by_the_voltage_table(self):
        issues = validate_result("run_rsa", {
            "total_violations": 0,
            "violations": [],
            "all_voltages": [{"bus_name": "S3", "vm_pu": 1.08}],
            "thresholds_used": {"vm_lower_pu": 0.95, "vm_upper_pu": 1.05},
        })
        assert "unreported_violation" in codes(issues)

    def test_physically_impossible_voltage_is_flagged(self):
        issues = validate_result("run_rsa", {
            "all_voltages": [{"bus_name": "S1", "vm_pu": 3.4}],
        })
        assert "implausible_result" in codes(issues)

    def test_tool_errors_are_left_alone(self):
        assert validate_result("run_rsa", {"error": "backend unreachable"}) == []

    def test_non_dict_results_are_safe(self):
        assert validate_result("run_rsa", None) == []
        assert validate_result("run_rsa", [1, 2, 3]) == []

    def test_malformed_voltage_entries_do_not_crash(self):
        validate_result("run_rsa", {"all_voltages": ["not a dict", {"vm_pu": None}]})


class TestAnnotate:
    def test_findings_are_attached_not_substituted(self):
        result = {"total_violations": 0, "data": [1, 2, 3]}
        issues = validate_result("run_rsa", {
            "total_violations": 0, "violations": [],
            "all_voltages": [{"bus_name": "S3", "vm_pu": 1.08}],
            "thresholds_used": {"vm_lower_pu": 0.95, "vm_upper_pu": 1.05},
        })
        annotated = annotate(result, issues)
        # Original payload survives — the model still needs it.
        assert annotated["data"] == [1, 2, 3]
        assert annotated["_integrity_warnings"]

    def test_clean_results_are_returned_unchanged(self):
        result = {"ok": True}
        assert annotate(result, []) is result

    def test_severities_are_the_documented_two(self):
        verdict = validate_call("run_rsa", {"vm_upper_pu": 1.25, "load_sigma": 9.0}, GRID)
        assert {i.severity for i in verdict.issues} <= {REJECT, WARN}



class TestComparingTwoMomentsOnPurpose:
    """Asking about two operating points is a request, not a slip.

    The check exists because a follow-up tool once fell back to the simulation
    clock and the answer described two days as one. But a user who asks to
    compare Tuesday with Wednesday gets tools at two timestamps by design, and
    warning about that told the model its own correct behaviour was
    inconsistent — and put a `fail` on a clean turn in the grader.
    """

    def _results(self, *timestamps):
        return [(f"tool{i}", {"timestamp": t}) for i, t in enumerate(timestamps)]

    def test_both_moments_asked_for_is_not_drift(self):
        asked = requested_timestamps(
            "Compare 2022-01-02 21:45 with 2022-01-03 21:45.")
        consistency = TurnConsistency(expected=asked)
        issues = [i for name, result in self._results("2022-01-02 21:45:00",
                                                      "2022-01-03 21:45:00")
                  for i in consistency.observe(name, result)]
        assert issues == []

    def test_a_third_moment_nobody_asked_for_is_still_drift(self):
        asked = requested_timestamps(
            "Compare 2022-01-02 21:45 with 2022-01-03 21:45.")
        consistency = TurnConsistency(expected=asked)
        issues = [i for name, result in self._results("2022-01-02 21:45:00",
                                                      "2022-01-03 21:45:00",
                                                      "2022-01-01 00:00:00")
                  for i in consistency.observe(name, result)]
        assert [i.code for i in issues] == ["inconsistent_context"]

    def test_the_original_failure_is_unaffected(self):
        # One moment asked for, a later tool fell back to the clock.
        asked = requested_timestamps("Run an RSA at 2022-01-02 21:45 and attribute it.")
        consistency = TurnConsistency(expected=asked)
        issues = [i for name, result in self._results("2022-01-02 21:45:00",
                                                      "2022-01-01 00:00:00")
                  for i in consistency.observe(name, result)]
        assert [i.code for i in issues] == ["inconsistent_context"]

    def test_with_no_expectation_the_guard_is_as_strict_as_before(self):
        consistency = TurnConsistency()
        issues = [i for name, result in self._results("2022-01-02 21:45:00",
                                                      "2022-01-03 21:45:00")
                  for i in consistency.observe(name, result)]
        assert [i.code for i in issues] == ["inconsistent_context"]


class TestRequestedTimestamps:
    def test_reads_every_moment_the_question_names(self):
        assert requested_timestamps("Compare 2022-01-02 21:45 with 2022-01-03 21:45.") == {
            "2022-01-02 21:45", "2022-01-03 21:45"}

    def test_a_question_naming_none_yields_none(self):
        assert requested_timestamps("What is the state of the grid?") == frozenset()


class TestTurnConsistency:
    """An answer assembled from tools that ran at different operating points.

    The failure this exists to catch, observed live: the model passed an
    explicit timestamp to the security assessment and omitted it from the
    follow-up attribution, which fell back to the simulation clock. Both
    results were individually correct; the answer combined 11 violations from
    one day with 9 buses from another and read as a single coherent analysis.
    """

    def test_a_single_tool_is_always_consistent(self):
        tracker = TurnConsistency()
        assert tracker.observe("run_rsa", {"timestamp": "2022-01-02 21:45:00"}) == []

    def test_matching_operating_points_are_silent(self):
        tracker = TurnConsistency()
        tracker.observe("run_rsa", {"timestamp": "2022-01-02 21:45:00"})
        assert tracker.observe(
            "compute_violation_attribution", {"timestamp": "2022-01-02 21:45:00"}
        ) == []

    def test_timestamps_are_compared_to_the_minute(self):
        # Tools echo timestamps with and without seconds; that is not a mismatch.
        tracker = TurnConsistency()
        tracker.observe("run_rsa", {"timestamp": "2022-01-02 21:45"})
        assert tracker.observe("attr", {"timestamp": "2022-01-02 21:45:00"}) == []

    def test_a_dropped_timestamp_is_caught(self):
        tracker = TurnConsistency()
        tracker.observe("run_rsa", {"timestamp": "2022-01-02 21:45:00"})
        issues = tracker.observe(
            "compute_violation_attribution", {"timestamp": "2022-01-01 00:00:00"}
        )
        assert "inconsistent_context" in codes(issues)
        assert "2022-01-02 21:45" in issues[0].message
        assert "2022-01-01 00:00" in issues[0].message

    def test_the_message_names_both_tools(self):
        tracker = TurnConsistency()
        tracker.observe("run_rsa", {"timestamp": "2022-01-02 21:45:00"})
        issues = tracker.observe("compute_violation_attribution",
                                 {"timestamp": "2022-01-01 00:00:00"})
        assert "run_rsa" in issues[0].message
        assert "compute_violation_attribution" in issues[0].message

    def test_mixing_measurements_and_forecasts_is_caught(self):
        tracker = TurnConsistency()
        tracker.observe("run_rsa", {"data_source": "measurements"})
        issues = tracker.observe("forecast_kpis", {"data_source": "forecasts"})
        assert "inconsistent_context" in codes(issues)
        assert "dataset" in issues[0].message

    def test_differing_thresholds_are_caught(self):
        tracker = TurnConsistency()
        tracker.observe("run_rsa", {"thresholds_used": {"vm_upper_pu": 1.045}})
        issues = tracker.observe("simulate_all_contingencies",
                                 {"thresholds_used": {"vm_upper_pu": 1.05}})
        assert "inconsistent_thresholds" in codes(issues)
        assert "1.045" in issues[0].message and "1.05" in issues[0].message

    def test_identical_thresholds_are_silent(self):
        tracker = TurnConsistency()
        limits = {"vm_lower_pu": 0.95, "vm_upper_pu": 1.045}
        tracker.observe("run_rsa", {"thresholds_used": limits})
        assert tracker.observe("attr", {"thresholds_used": dict(limits)}) == []

    def test_partial_threshold_overlap_only_compares_shared_keys(self):
        tracker = TurnConsistency()
        tracker.observe("run_rsa", {"thresholds_used": {"vm_upper_pu": 1.045}})
        assert tracker.observe("attr", {"thresholds_used": {"max_line_loading_pct": 100}}) == []

    def test_the_first_observation_sets_the_reference(self):
        """A later stray call is reported against the established context,
        not allowed to redefine it."""
        tracker = TurnConsistency()
        tracker.observe("run_rsa", {"timestamp": "2022-01-02 21:45:00"})
        tracker.observe("attr", {"timestamp": "2022-01-01 00:00:00"})
        issues = tracker.observe("third", {"timestamp": "2022-01-01 00:00:00"})
        assert issues, "the established reference should still be 2022-01-02"
        assert "run_rsa" in issues[0].message

    def test_results_without_context_are_ignored(self):
        tracker = TurnConsistency()
        tracker.observe("run_rsa", {"timestamp": "2022-01-02 21:45:00"})
        assert tracker.observe("get_current_conditions", {"load_mw": 6.0}) == []

    def test_errors_and_non_dicts_are_ignored(self):
        tracker = TurnConsistency()
        assert tracker.observe("x", {"error": "backend down"}) == []
        assert tracker.observe("x", None) == []
        assert tracker.observe("x", [1, 2]) == []


class TestOperatingContext:
    def test_extracts_the_fields_that_identify_an_operating_point(self):
        context = operating_context({
            "timestamp": "2022-01-02 21:45:00",
            "data_source": "measurements",
            "thresholds_used": {"vm_upper_pu": 1.045},
            "violations": [],
        })
        assert context["timestamp"] == "2022-01-02 21:45"
        assert context["data_source"] == "measurements"
        assert context["thresholds"] == {"vm_upper_pu": 1.045}

    def test_ignores_payload_data(self):
        assert "violations" not in operating_context({"violations": [1, 2, 3]})

    def test_empty_for_results_that_declare_nothing(self):
        assert operating_context({"some": "payload"}) == {}


class TestTurnOperatingPoint:
    """Carrying the operating point across a turn, without asking the model to.

    Detection tells you an answer mixed two operating points; this prevents it.
    Requiring the argument instead would force the model to supply *a*
    timestamp, not the right one, and would break the common case where
    omitting it correctly means "the current operating point".
    """

    ACCEPTS = {"timestamp", "data_source"}

    def test_an_explicit_value_passes_through_untouched(self):
        point = TurnOperatingPoint()
        kwargs, inherited = point.resolve(
            "run_rsa", {"timestamp": "2022-01-02 21:45"}, self.ACCEPTS)
        assert kwargs["timestamp"] == "2022-01-02 21:45"
        assert inherited == {}

    def test_a_later_omission_inherits_it(self):
        point = TurnOperatingPoint()
        point.resolve("run_rsa", {"timestamp": "2022-01-02 21:45"}, self.ACCEPTS)
        kwargs, inherited = point.resolve("attribution", {}, self.ACCEPTS)
        assert kwargs["timestamp"] == "2022-01-02 21:45"
        assert inherited["timestamp"]["from"] == "run_rsa"

    def test_an_explicit_later_value_wins_over_inheritance(self):
        """A deliberate two-timestamp comparison must still work."""
        point = TurnOperatingPoint()
        point.resolve("run_rsa", {"timestamp": "2022-01-02 08:00"}, self.ACCEPTS)
        kwargs, inherited = point.resolve(
            "run_rsa", {"timestamp": "2022-01-02 21:00"}, self.ACCEPTS)
        assert kwargs["timestamp"] == "2022-01-02 21:00"
        assert inherited == {}

    def test_nothing_is_inherited_when_the_turn_never_stated_one(self):
        # Omitting the timestamp still correctly means "the current operating
        # point" when the user never named one.
        point = TurnOperatingPoint()
        kwargs, inherited = point.resolve("run_rsa", {}, self.ACCEPTS)
        assert "timestamp" not in kwargs
        assert inherited == {}

    def test_never_injected_into_a_tool_that_would_reject_it(self):
        point = TurnOperatingPoint()
        point.resolve("run_rsa", {"timestamp": "2022-01-02 21:45"}, self.ACCEPTS)
        kwargs, inherited = point.resolve("compare_results", {}, {"label_a", "label_b"})
        assert "timestamp" not in kwargs
        assert inherited == {}

    def test_unknown_signature_injects_nothing(self):
        point = TurnOperatingPoint()
        point.resolve("run_rsa", {"timestamp": "2022-01-02 21:45"}, self.ACCEPTS)
        kwargs, _ = point.resolve("mystery", {}, None)
        assert "timestamp" not in kwargs

    def test_the_dataset_is_carried_too(self):
        point = TurnOperatingPoint()
        point.resolve("forecast_kpis", {"data_source": "forecasts"}, self.ACCEPTS)
        kwargs, _ = point.resolve("attribution", {}, self.ACCEPTS)
        assert kwargs["data_source"] == "forecasts"

    def test_the_first_explicit_value_defines_the_turn(self):
        point = TurnOperatingPoint()
        point.resolve("a", {"timestamp": "2022-01-02 21:45"}, self.ACCEPTS)
        point.resolve("b", {"timestamp": "2022-05-05 05:05"}, self.ACCEPTS)
        kwargs, inherited = point.resolve("c", {}, self.ACCEPTS)
        assert kwargs["timestamp"] == "2022-01-02 21:45"
        assert inherited["timestamp"]["from"] == "a"

    def test_empty_strings_count_as_omitted(self):
        point = TurnOperatingPoint()
        point.resolve("run_rsa", {"timestamp": "2022-01-02 21:45"}, self.ACCEPTS)
        kwargs, inherited = point.resolve("attribution", {"timestamp": ""}, self.ACCEPTS)
        assert kwargs["timestamp"] == "2022-01-02 21:45"
        assert inherited

    def test_the_callers_arguments_are_not_mutated(self):
        point = TurnOperatingPoint()
        point.resolve("run_rsa", {"timestamp": "2022-01-02 21:45"}, self.ACCEPTS)
        original = {}
        point.resolve("attribution", original, self.ACCEPTS)
        assert original == {}
