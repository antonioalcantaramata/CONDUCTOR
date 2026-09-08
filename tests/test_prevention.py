"""Prevention rather than detection.

The guards in `provenance.py` and `characterisation.py` observe what the model
wrote. These stop the wrong figure reaching it in the first place — engines
stating their own verdicts, impossible values withdrawn, user limits carried
across a turn, unsupported requests refused in code rather than in prose.

Where an engine knows the answer, prevention is the stronger form: the model
cannot misquote what it never received.
"""

import pytest

from backend import withdrawal
from evaluation.graders.refusal import grade_refusal
from llm_agent.agent.validators import TurnOperatingPoint


class TestWithdrawal:
    def test_impossible_value_is_withdrawn_and_replaced(self):
        """The live case: a 3.54 MW unit asked to move 6.24 MW."""
        driver = {"relief_mw": -6.2415, "relief_mw_feasible": False, "current_p_mw": 3.54}
        assert withdrawal.withdraw(driver, "relief_mw", "relief_mw_feasible", "current_p_mw")
        assert driver["relief_mw"] is None
        assert driver["max_relief_mw"] == pytest.approx(3.54)

    def test_something_true_is_always_left_to_quote(self):
        """Reporting only what cannot be done made an earlier version of the
        prompt rule answer with no numbers at all."""
        driver = {"relief_mw": -6.2415, "relief_mw_feasible": False, "current_p_mw": 3.54}
        withdrawal.withdraw(driver, "relief_mw", "relief_mw_feasible", "current_p_mw")
        assert any(v is not None for k, v in driver.items() if k != "relief_mw")

    def test_a_feasible_value_survives(self):
        driver = {"relief_mw": -0.42, "relief_mw_feasible": True, "current_p_mw": 3.54}
        assert not withdrawal.withdraw(driver, "relief_mw", "relief_mw_feasible", "current_p_mw")
        assert driver["relief_mw"] == pytest.approx(-0.42)

    def test_unknown_feasibility_survives(self):
        """Refuse the impossible, allow the merely unusual. Deleting unknowns
        would strip the diagnostic content that explains the regime."""
        driver = {"relief_mw": -6.24, "relief_mw_feasible": None, "current_p_mw": 3.54}
        assert not withdrawal.withdraw(driver, "relief_mw", "relief_mw_feasible", "current_p_mw")
        assert driver["relief_mw"] == pytest.approx(-6.24)

    def test_withdraw_all_counts_what_it_took(self):
        records = [
            {"relief_mw": -6.24, "relief_mw_feasible": False, "current_p_mw": 3.54},
            {"relief_mw": -0.42, "relief_mw_feasible": True, "current_p_mw": 3.54},
            {"relief_mw": -9.10, "relief_mw_feasible": False, "current_p_mw": 1.00},
        ]
        assert withdrawal.withdraw_all(records, "relief_mw", "relief_mw_feasible",
                                       "current_p_mw") == 2
        assert records[1]["relief_mw"] == pytest.approx(-0.42)

    def test_mark_unusable_keeps_the_evidence(self):
        """A non-converged study's iterates are diagnostic. State the verdict,
        keep the numbers."""
        result = {"hosting_capacity_mw": 4.15, "iterations": 12}
        withdrawal.mark_unusable(result, "bisection did not converge")
        assert result["feasible"] is False
        assert result["usable"] is False
        assert result["iterations"] == 12
        assert "converge" in result["unusable_reason"]


class TestConstraintPropagation:
    """A user-set limit belongs to every study in the turn it was set for."""

    def test_a_voltage_ceiling_carries_to_the_next_call(self):
        turn = TurnOperatingPoint()
        accepts = {"timestamp", "vm_upper_pu"}
        turn.resolve("run_rsa", {"vm_upper_pu": 1.045}, accepts)
        resolved, inherited = turn.resolve("simulate_all_contingencies", {}, accepts)
        assert resolved["vm_upper_pu"] == pytest.approx(1.045)
        assert inherited["vm_upper_pu"]["from"] == "run_rsa"

    def test_an_explicit_value_is_never_overwritten(self):
        turn = TurnOperatingPoint()
        accepts = {"vm_upper_pu"}
        turn.resolve("run_rsa", {"vm_upper_pu": 1.045}, accepts)
        resolved, inherited = turn.resolve("other", {"vm_upper_pu": 1.06}, accepts)
        assert resolved["vm_upper_pu"] == pytest.approx(1.06)
        assert "vm_upper_pu" not in inherited

    def test_nothing_is_injected_into_a_tool_that_would_reject_it(self):
        turn = TurnOperatingPoint()
        turn.resolve("run_rsa", {"vm_upper_pu": 1.045}, {"vm_upper_pu"})
        resolved, inherited = turn.resolve("get_current_timestamp", {}, set())
        assert resolved == {}
        assert inherited == {}

    def test_the_operating_point_still_carries(self):
        """The original behaviour, unchanged by adding limits alongside it."""
        turn = TurnOperatingPoint()
        accepts = {"timestamp"}
        turn.resolve("run_rsa", {"timestamp": "2022-01-02 21:45"}, accepts)
        resolved, _ = turn.resolve("simulate_contingency", {}, accepts)
        assert resolved["timestamp"] == "2022-01-02 21:45"

    @pytest.mark.parametrize("field", [
        "vm_lower_pu", "max_line_loading_pct", "max_trafo_loading_pct",
    ])
    def test_every_limit_propagates(self, field):
        turn = TurnOperatingPoint()
        turn.resolve("run_rsa", {field: 42.0}, {field})
        resolved, _ = turn.resolve("next", {}, {field})
        assert resolved[field] == pytest.approx(42.0)


class TestRefusalIntegrity:
    """The ablation's failure, graded rather than read off a transcript."""

    def _unsupported(self, assistant, extra_results=()):
        return {
            "turn": 1,
            "user": "run N-1 with generator 3 unavailable",
            "assistant": assistant,
            "tool_results": [
                {"name": "simulate_all_contingencies",
                 "result": {"error": "Unsupported arguments", "refusal_required": True}},
                *extra_results,
            ],
        }

    def test_declining_passes(self):
        record = self._unsupported(
            "That combination is not supported — generator unavailability cannot be "
            "combined with a full N-1 screen. I can run either separately."
        )
        assert grade_refusal(record) == []

    def test_answering_anyway_fails(self):
        record = self._unsupported("The N-1 screen found 3 violations at Bus 4.")
        findings = grade_refusal(record)
        assert len(findings) == 1
        assert findings[0].code == "unsupported_not_declined"

    def test_substituting_another_study_fails(self):
        """The ablation's exact failure: a different study ran correctly, and
        its real numbers were reported as the answer."""
        record = self._unsupported(
            "The assessment found 3 violations at Bus 4.",
            extra_results=[{"name": "run_rsa", "result": {"total_violations": 3}}],
        )
        findings = grade_refusal(record)
        assert len(findings) == 1
        assert "run_rsa" in findings[0].detail

    def test_a_turn_with_no_unsupported_request_is_not_graded(self):
        record = {
            "turn": 1, "user": "status?", "assistant": "All secure.",
            "tool_results": [{"name": "run_rsa", "result": {"total_violations": 0}}],
        }
        assert grade_refusal(record) == []

    def test_a_negative_finding_is_not_a_refusal(self):
        """"No violations were found" is an answer, not a decline. Counting it
        as one would let a substituted study pass by describing its results."""
        record = self._unsupported("No violations were found in the screen.")
        assert grade_refusal(record), "reporting a result is not declining"


class TestUncontrollableDrivers:
    """A movement nobody can command is a sensitivity, not an action.

    Live case: every driver of an undervoltage was a load, each carrying
    `relief_mw: -12.411` and `relief_mw_feasible: True` — arithmetically
    feasible, since the demand is there to remove, and indistinguishable in
    the payload from a generator redispatch.
    """

    def _load_driver(self):
        return {"source": "Load_2", "controllable": False,
                "relief_mw": -12.411, "relief_mw_feasible": True,
                "relief_mvar": -3.2, "relief_mvar_feasible": True,
                "current_p_mw": 60.804, "current_q_mvar": 12.0}

    def test_a_load_movement_is_renamed_not_deleted(self):
        from backend.attribution_engine import _reframe_uncontrollable

        driver = self._load_driver()
        _reframe_uncontrollable(driver)
        assert driver["relief_mw"] is None, "no field an answer would quote as an action"
        assert driver["would_require_mw"] == pytest.approx(-12.411), "the diagnosis keeps it"
        assert driver["actionable"] is False

    def test_the_feasibility_flag_goes_with_it(self):
        """Leaving `relief_mw_feasible: True` beside a withdrawn movement tells
        a guard the action is available — the reverse of the intent."""
        from backend.attribution_engine import _reframe_uncontrollable

        driver = self._load_driver()
        _reframe_uncontrollable(driver)
        assert driver["relief_mw_feasible"] is False
        assert driver["relief_mvar_feasible"] is False

    def test_a_controllable_driver_is_untouched(self):
        from backend.attribution_engine import _reframe_uncontrollable

        driver = {"source": "Gen_1", "controllable": True,
                  "relief_mw": -0.42, "relief_mw_feasible": True,
                  "relief_mvar": None, "relief_mvar_feasible": None,
                  "current_p_mw": 3.54, "current_q_mvar": 0.0}
        _reframe_uncontrollable(driver)
        assert driver["relief_mw"] == pytest.approx(-0.42)
        assert "would_require_mw" not in driver
        assert "actionable" not in driver

    def test_quoting_it_as_an_action_is_caught(self):
        from llm_agent.agent.characterisation import check_answer
        from backend.attribution_engine import _reframe_uncontrollable

        driver = self._load_driver()
        _reframe_uncontrollable(driver)
        record = {"user": "why?", "tool_results": [{
            "name": "compute_violation_attribution",
            "result": {"violations": [{"element": "Bus_3", "violated": True,
                                       "drivers": [driver]}]},
        }]}
        conflicts = check_answer("Reduce Load_2 by 12.411 MW to clear Bus_3.", record)
        assert any(c.field == "actionable" for c in conflicts)

    def test_explaining_the_regime_with_it_is_not_caught(self):
        """The figure exists to be quoted diagnostically. A check that flagged
        the explanation would remove the reason loads are collected at all."""
        from llm_agent.agent.characterisation import check_answer
        from backend.attribution_engine import _reframe_uncontrollable

        driver = self._load_driver()
        _reframe_uncontrollable(driver)
        record = {"user": "why?", "tool_results": [{
            "name": "compute_violation_attribution",
            "result": {"violations": [{"element": "Bus_3", "violated": True,
                                       "drivers": [driver]}]},
        }]}
        assert check_answer(
            "Bus_3 is driven by demand: clearing it by load alone would require "
            "12.411 MW, which is not an available action.",
            record,
        ) == []


class TestLimitsFromThePrompt:
    """The first call of a turn has nothing earlier to inherit from.

    Observed live: a question setting a 1.045 ceiling ran a week-long scan
    first — at the tool default of 1.06, nothing established yet — then three
    studies at 1.045. The consistency guard correctly reported they were not
    one assessment, which warns about the gap rather than closing it.
    """

    def test_a_stated_ceiling_reaches_the_first_call(self):
        from llm_agent.agent.validators import requested_limits

        limits = requested_limits(
            "Would the system still be secure with a maximum voltage of 1.045?"
        )
        turn = TurnOperatingPoint(requested=limits)
        resolved, inherited = turn.resolve("find_worst_case_timestamp", {}, {"vm_upper_pu"})
        assert resolved["vm_upper_pu"] == pytest.approx(1.045)
        assert inherited["vm_upper_pu"]["from"] == "the request"

    @pytest.mark.parametrize("text,expected", [
        ("with a maximum voltage of 1.045", 1.045),
        ("keep voltages below 1.05", 1.05),
        ("cap it at 1.04", 1.04),
        ("no higher than 1.055", 1.055),
    ])
    def test_upper_limit_phrasings(self, text, expected):
        from llm_agent.agent.validators import requested_limits

        assert requested_limits(text)["vm_upper_pu"] == pytest.approx(expected)

    def test_lower_limits_are_read_too(self):
        from llm_agent.agent.validators import requested_limits

        assert requested_limits("keep voltages above 0.96")["vm_lower_pu"] == pytest.approx(0.96)

    @pytest.mark.parametrize("text", [
        "what is the worst timestamp this week",
        "the bus is at 1.049 p.u.",
        "run an assessment at 2022-01-02 21:45",
        "keep loading under 100 percent",
    ])
    def test_a_bare_number_is_not_a_limit(self, text):
        """A limit parser that read any nearby decimal would silently re-run
        every study against a number the operator never meant as a ceiling."""
        from llm_agent.agent.validators import requested_limits

        assert requested_limits(text) == {}

    def test_an_explicit_argument_still_wins(self):
        from llm_agent.agent.validators import requested_limits

        turn = TurnOperatingPoint(requested=requested_limits("maximum voltage of 1.045"))
        resolved, _ = turn.resolve("run_rsa", {"vm_upper_pu": 1.06}, {"vm_upper_pu"})
        assert resolved["vm_upper_pu"] == pytest.approx(1.06)


class TestAliasedLimits:
    """`opf_vm_upper` and `vm_upper_pu` are the same ceiling.

    Without alias handling the OPF is an island: it neither inherits the limit
    the operator set nor contributes its own to the calls after it.
    """

    def test_a_canonical_limit_reaches_an_aliased_tool(self):
        turn = TurnOperatingPoint()
        turn.resolve("run_rsa", {"vm_upper_pu": 1.045}, {"vm_upper_pu"})
        resolved, inherited = turn.resolve(
            "optimize_flexibility", {}, {"opf_vm_upper", "opf_vm_lower"}
        )
        assert resolved["opf_vm_upper"] == pytest.approx(1.045)
        assert inherited["opf_vm_upper"]["from"] == "run_rsa"

    def test_an_aliased_limit_establishes_the_canonical_one(self):
        turn = TurnOperatingPoint()
        turn.resolve("optimize_flexibility", {"opf_vm_upper": 1.04}, {"opf_vm_upper"})
        resolved, _ = turn.resolve("run_rsa", {}, {"vm_upper_pu"})
        assert resolved["vm_upper_pu"] == pytest.approx(1.04)

    def test_an_explicit_alias_is_not_overwritten(self):
        turn = TurnOperatingPoint()
        turn.resolve("run_rsa", {"vm_upper_pu": 1.045}, {"vm_upper_pu"})
        resolved, _ = turn.resolve(
            "optimize_flexibility", {"opf_vm_upper": 1.06}, {"opf_vm_upper"}
        )
        assert resolved["opf_vm_upper"] == pytest.approx(1.06)

    def test_an_alias_is_not_injected_into_a_tool_that_rejects_it(self):
        turn = TurnOperatingPoint()
        turn.resolve("run_rsa", {"vm_upper_pu": 1.045}, {"vm_upper_pu"})
        resolved, _ = turn.resolve("get_current_timestamp", {}, set())
        assert "opf_vm_upper" not in resolved

    def test_the_full_live_sequence(self):
        """The turn that produced the warning, replayed."""
        from llm_agent.agent.validators import requested_limits

        turn = TurnOperatingPoint(requested=requested_limits(
            "Would the system be secure with a maximum voltage of 1.045, "
            "and what if contingencies happen?"
        ))
        scan, _ = turn.resolve("find_worst_case_timestamp", {}, {"vm_upper_pu"})
        rsa, _ = turn.resolve("run_rsa", {}, {"vm_upper_pu"})
        opf, _ = turn.resolve("optimize_flexibility", {}, {"opf_vm_upper"})
        assert scan["vm_upper_pu"] == rsa["vm_upper_pu"] == opf["opf_vm_upper"] == pytest.approx(1.045)
