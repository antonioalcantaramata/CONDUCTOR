"""
Numeric provenance: does every figure in the answer come from the evidence?

The submitted paper asserted that it does. These tests exercise the instrument
that measures it, and in particular its precision: a grounding check that
flags the model's own correct arithmetic, the user's own numbers quoted back,
or the digits inside a bus name would produce a stream of false positives and
be abandoned within a day.
"""

import pytest

from llm_agent.agent.provenance import (
    GROUNDED,
    IMPRECISE,
    MISATTRIBUTED,
    UNGROUNDED,
    check_answer,
    check_claim,
    check_turn,
    collect_sources,
    extract_claims,
    subject_names,
)


def _values(answer):
    return [c.value for c in extract_claims(answer)]


class TestExtraction:
    def test_reads_value_and_unit(self):
        (claim,) = extract_claims("Curtail by 3.35 MW.")
        assert (claim.value, claim.unit, claim.decimals) == (3.35, "MW", 2)

    def test_reads_through_bold_emphasis(self):
        # The agent writes "**3.35 MW**"; the markers sit between the two.
        (claim,) = extract_claims("Reduce by **3.35 MW** today.")
        assert (claim.value, claim.unit) == (3.35, "MW")

    @pytest.mark.parametrize("text,unit", [
        ("1.045 p.u. limit", "p.u."),
        ("98.5% of tasks", "%"),
        ("12.0 MVAr injected", "MVAr"),
        ("1,234.5 MWh over the year", "MWh"),
    ])
    def test_units(self, text, unit):
        (claim,) = extract_claims(text)
        assert claim.unit == unit

    def test_thousands_separator(self):
        assert _values("1,234.5 MWh") == [1234.5]

    def test_trailing_zeros_set_the_precision(self):
        # 1.0470 and 1.047 are the same value but not the same claim: the
        # first asserts four decimal places and is graded at that precision.
        (claim,) = extract_claims("Voltage 1.0470")
        assert (claim.value, claim.decimals) == (1.047, 4)

    def test_ignores_numbers_glued_to_words(self):
        # "10kV" is part of a bus name and "L0" a line label — neither is a
        # claim about a quantity.
        assert _values("Viadukten 10kV on segment L0") == []

    def test_ignores_element_names_in_code_spans(self):
        assert _values("Reduce `05 ÅKI Sgen` output") == []

    def test_ignores_fenced_code(self):
        assert _values("Result:\n```\nvm_pu = 1.0456\n```\n") == []

    def test_ignores_timestamps(self):
        assert _values("At 2022-01-02 21:45 the network is secure.") == []

    def test_ignores_ordered_list_markers(self):
        assert _values("1. First point\n2. Second point\n") == []

    def test_ignores_table_alignment_rows(self):
        table = "| Bus | V |\n| :--- | ---: |\n| Hasle | 1.05 |\n"
        assert _values(table) == [1.05]

    def test_empty_answer(self):
        assert extract_claims("") == []


class TestSources:
    def test_collects_from_results_arguments_and_prompt(self):
        record = {
            "user": "Run an RSA with a maximum voltage of 1.045.",
            "tool_calls": [{"name": "run_rsa", "args": {"vm_upper_pu": 1.045}}],
            "tool_results": [{"name": "run_rsa", "result": {"total_violations": 11}}],
        }
        paths = {source.path for source in collect_sources(record)}
        assert "run_rsa.total_violations" in paths
        assert "args:run_rsa.vm_upper_pu" in paths
        assert "prompt" in paths

    def test_mines_numerals_out_of_strings(self):
        # Element names carry digits and the answer quotes them back. Without
        # this, "all on 10 kV buses" reads as an invention.
        record = {"tool_results": [{"name": "run_rsa", "result": {"element": "Viadukten 10kV"}}]}
        assert 10.0 in {source.value for source in collect_sources(record)}

    def test_booleans_are_not_numbers(self):
        record = {"tool_results": [{"name": "t", "result": {"converged": True}}]}
        assert collect_sources(record) == []

    def test_nested_paths_are_reported(self):
        record = {"tool_results": [
            {"name": "rsa", "result": {"violations": [{"value": 1.0456}]}},
        ]}
        (source,) = collect_sources(record)
        assert (source.value, source.path) == (1.0456, "rsa.violations[0].value")


class TestMatching:
    def test_exact_value_is_grounded(self):
        (claim,) = extract_claims("11 violations")
        assert check_claim(claim, [(11.0, "rsa.total_violations")]).status == GROUNDED

    def test_correctly_rounded_value_is_grounded(self):
        # The prose rounds what the solver returned in full.
        (claim,) = extract_claims("Voltage 1.0456")
        verdict = check_claim(claim, [(1.0456116504256603, "rsa.violations[0].value")])
        assert verdict.status == GROUNDED
        assert verdict.source == "rsa.violations[0].value"

    def test_truncated_value_is_imprecise_not_ungrounded(self):
        # Observed live: the answer wrote 1.0466 for a solver value of
        # 1.046666, where rounding gives 1.0467. A transcription slip, and not
        # the same thing as a number that appears nowhere.
        (claim,) = extract_claims("Voltage 1.0466")
        assert check_claim(claim, [(1.046666277724394, "rsa.value")]).status == IMPRECISE

    def test_a_whole_number_gets_no_truncation_band(self):
        # Observed live: "All 41 tested outages (24 lines and 17 transformers)"
        # — the split is the model counting the list itself, and it matched an
        # unrelated bus index of 23 under the truncation band. A whole number
        # is either in the evidence or the model produced it.
        (claim,) = extract_claims("24 lines")
        assert check_claim(claim, [(23.0, "rsa.violations[0].element_index")]).status == UNGROUNDED

    def test_a_decimal_keeps_its_truncation_band(self):
        (claim,) = extract_claims("1.0466 p.u.")
        assert check_claim(claim, [(1.046666, "rsa.value")]).status == IMPRECISE

    def test_absent_value_is_ungrounded(self):
        (claim,) = extract_claims("Curtail by 6.24 MW")
        verdict = check_claim(claim, [(3.54, "attr.source_p_mw")])
        assert verdict.status == UNGROUNDED
        assert verdict.nearest == (3.54, "attr.source_p_mw")

    def test_no_sources_at_all(self):
        (claim,) = extract_claims("42 MW")
        assert check_claim(claim, []).status == UNGROUNDED

    def test_typographic_minus_is_read_as_a_sign(self):
        # The model writes U+2212, not a hyphen. Read as +1.35, a correctly
        # reported export shows up as a fabricated number — which it did, on
        # the first real log this ran against.
        (claim,) = extract_claims("Slack import −1.35 MW (exporting)")
        assert claim.value == -1.35
        assert check_claim(claim, [(-1.3477, "conditions.ext_grid.P_import_mw")]).status == GROUNDED

    def test_en_dash_stays_a_range_separator(self):
        # "0.95–1.045" is two numbers, not one negative one.
        assert _values("limits 0.95–1.045 p.u.") == [0.95, 1.045]

    def test_a_direction_word_carries_the_sign(self):
        # The payload reports `slack_import_mw = -1.3146`; the answer says the
        # grid is "exporting 1.31 MW". Same flow, described from the other end,
        # and reading only the signed value called it fabricated.
        (claim,) = extract_claims("the external grid exporting 1.31 MW")
        verdict = check_claim(claim, [(-1.3146, "run_rsa.slack_import_mw")])
        assert verdict.status == GROUNDED
        assert verdict.inverted is True

    def test_without_a_direction_word_the_sign_still_counts(self):
        # Otherwise any fabricated figure that happened to equal some value's
        # opposite would ground.
        (claim,) = extract_claims("a shortfall of 1.31 MW remains")
        assert check_claim(claim, [(-1.3146, "run_rsa.slack_import_mw")]).status == UNGROUNDED

    def test_a_direction_word_does_not_ground_the_wrong_magnitude(self):
        (claim,) = extract_claims("the external grid exporting 99 MW")
        assert check_claim(claim, [(-1.3146, "run_rsa.slack_import_mw")]).status == UNGROUNDED

    def test_the_direction_itself_is_not_verified(self):
        """A stated limit, not an oversight.

        "importing 1.31 MW" grounds against an export just as "exporting"
        does: the rule accepts the magnitude once the prose claims a
        direction, and cannot tell a correct direction from a reversed one.
        Doing so needs the sign convention of each field — `slack_import_mw`
        positive meaning import — which is a different check from asking where
        a number came from.
        """
        (claim,) = extract_claims("the external grid importing 1.31 MW")
        assert check_claim(claim, [(-1.3146, "run_rsa.slack_import_mw")]).status == GROUNDED

    def test_percentage_matches_the_probability_it_came_from(self):
        # The risk tools return p_any_violation_after = 0.49; the answer says
        # "49%". Same figure, and the tolerance scales with it.
        (claim,) = extract_claims("risk falls to 49%")
        assert check_claim(claim, [(0.49, "robust.p_any_violation_after")]).status == GROUNDED

    def test_the_percentage_rule_does_not_launder_model_arithmetic(self):
        # Observed live: the model computed 0.27/6.31 and wrote "+4.3%". No
        # solver produced it, and the percentage rule must not ground it
        # against some unrelated 0.043.
        (claim,) = extract_claims("+4.3%")
        sources = [(0.27, "a.delta"), (6.31, "a.total"), (6.58, "b.total")]
        assert check_claim(claim, sources).status == UNGROUNDED


class TestReport:
    def test_rate_counts_imprecise_as_traceable(self):
        report = check_answer(
            "11 violations, worst 1.0466 p.u.",
            [(11.0, "rsa.total_violations"), (1.046666, "rsa.value")],
        )
        assert (len(report.grounded), len(report.imprecise), len(report.ungrounded)) == (1, 1, 0)
        assert report.rate == 1.0

    def test_fabricated_figure_lowers_the_rate(self):
        report = check_answer("11 violations and 99 MW of curtailment",
                              [(11.0, "rsa.total_violations")])
        assert len(report.ungrounded) == 1
        assert report.rate == 0.5

    def test_answer_with_no_figures_is_vacuously_grounded(self):
        assert check_answer("The network is secure.", []).rate == 1.0

    def test_user_supplied_limit_quoted_back_is_grounded(self):
        # The 1.045 in the answer came from the question, not from a solver.
        # Counting it as fabricated would be the instrument's most common
        # false positive.
        record = {
            "user": "Run an RSA with a maximum voltage of 1.045.",
            "tool_calls": [],
            "tool_results": [],
            "assistant": "Using a maximum voltage of 1.045 p.u., the network is secure.",
        }
        assert check_turn(record).ungrounded == []

    def test_figure_from_an_earlier_turn_is_not_grounded_in_this_one(self):
        # Sources are turn-scoped on purpose: a value carried over from a
        # previous answer is stale evidence, and reads as ungrounded here.
        record = {
            "user": "And the worst bus?",
            "tool_calls": [],
            "tool_results": [{"name": "rsa", "result": {"total_violations": 3}}],
            "assistant": "The worst is 1.0456 p.u.",
        }
        assert len(check_turn(record).ungrounded) == 1


def _attribution_record(assistant):
    """Two violations sharing one source — the shape that produced the failure.

    05 ÅKI Sgen can clear Viadukten with 3.354 MW. It cannot clear Åkirkeby at
    all: that movement was withdrawn as undeliverable, which is exactly why
    the model reached for the neighbouring number.
    """
    return {
        "user": "Attribute the worst violation.",
        "tool_calls": [{"name": "compute_violation_attribution", "args": {}}],
        "tool_results": [{"name": "compute_violation_attribution", "result": {
            "violations": [
                {"element": "Viadukten 10kV", "value": 1.0456, "limit": 1.045,
                 "drivers": [{"source": "05 ÅKI Sgen", "current_p_mw": 3.537,
                              "relief_mw": -3.354, "relief_mw_feasible": True}],
                 "recommended_action": {
                     "kind": "single_source",
                     "text": "Reduce 05 ÅKI Sgen by 3.35 MW to bring Viadukten 10kV back to 1.045.",
                 }},
                {"element": "Åkirkeby 10.5 kV", "value": 1.0506, "limit": 1.045,
                 "drivers": [{"source": "05 ÅKI Sgen", "current_p_mw": 3.537,
                              "relief_mw": None, "relief_mw_feasible": False,
                              "max_deliverable_mw": -3.537}],
                 "recommended_action": {
                     "kind": "none_sufficient",
                     "text": "No single source can clear Åkirkeby 10.5 kV.",
                 }},
            ],
        }}],
        "assistant": assistant,
    }


class TestSubjectVocabulary:
    def test_only_violated_elements_can_be_subjects(self):
        record = _attribution_record("")
        assert subject_names(record) == {"Viadukten 10kV", "Åkirkeby 10.5 kV"}

    def test_a_name_contained_in_a_longer_one_is_dropped(self):
        # The dispatch table names its rows `element` too, so the generator
        # "Åkirkeby" entered the vocabulary and shadowed the violated bus
        # "Åkirkeby 10.5 kV" the sentence was actually about. Prose that says
        # "Åkirkeby" has not said which one.
        record = _attribution_record("")
        record["tool_results"][0]["result"]["violations"].append(
            {"element": "Åkirkeby", "value": 1.05, "limit": 1.045, "drivers": []}
        )
        assert "Åkirkeby" not in subject_names(record)


class TestMisattribution:
    def test_a_figure_from_another_violation_is_caught(self):
        # Observed live. 3.35 MW clears Viadukten; on Åkirkeby the movement
        # was withdrawn as impossible, and the model filled the gap with the
        # neighbouring violation's number.
        record = _attribution_record(
            "### Attribution\n"
            "The worst violation is at Åkirkeby 10.5 kV (1.0506 p.u.).\n"
            "05 ÅKI Sgen is the main contributor, requiring a reduction of 3.35 MW."
        )
        (verdict,) = check_turn(record).misattributed
        assert verdict.claim.value == 3.35
        assert verdict.subject == "Åkirkeby 10.5 kV"
        assert verdict.belongs_to == "Viadukten 10kV"

    def test_the_same_figure_under_its_own_element_is_grounded(self):
        record = _attribution_record(
            "### Attribution\n"
            "Viadukten 10kV is violating. Reduce 05 ÅKI Sgen by 3.35 MW."
        )
        report = check_turn(record)
        assert report.misattributed == []
        assert len(report.grounded) == len(report.verdicts)

    def test_a_heading_ends_the_subject(self):
        # Without this the last element named in one section would still be
        # the subject halfway through the next.
        record = _attribution_record(
            "### Attribution\nThe worst violation is at Åkirkeby 10.5 kV.\n"
            "### Dispatch\nThe unit moved by 3.35 MW."
        )
        assert check_turn(record).misattributed == []

    def test_turn_level_figures_are_not_attributed_to_the_subject(self):
        # Totals, thresholds, dispatch rows and call arguments belong to no
        # violated element. A sentence about a unit's setpoint is not a claim
        # about whichever bus the paragraph opened with — leaving these scoped
        # produced a false positive on every dispatch figure.
        record = _attribution_record(
            "### Attribution\nThe worst violation is at Åkirkeby 10.5 kV.\n"
            "Åkirkeby was reduced from 3.54 MW to 3.46 MW."
        )
        record["tool_results"].append({"name": "optimize_flexibility", "result": {
            "activated_resources": [
                {"element": "Åkirkeby", "Pg_base": 3.537, "Pg_new": 3.4612},
            ],
        }})
        assert check_turn(record).misattributed == []

    def test_without_a_subject_the_check_stays_flat(self):
        # No element named, so nothing to attribute against — the figure is
        # graded against the turn as a whole, as before scoping existed.
        record = _attribution_record("A reduction of 3.35 MW is required.")
        report = check_turn(record)
        assert report.misattributed == []
        assert len(report.grounded) == 1

    def test_misattributed_does_not_count_as_traceable(self):
        record = _attribution_record(
            "### Attribution\nThe worst violation is at Åkirkeby 10.5 kV, "
            "requiring 3.35 MW."
        )
        report = check_turn(record)
        assert report.rate < 1.0
        assert MISATTRIBUTED in {v.status for v in report.verdicts}
