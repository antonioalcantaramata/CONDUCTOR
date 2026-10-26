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
        assert _values("Oscar 10kV on segment L0") == []

    def test_ignores_element_names_in_code_spans(self):
        assert _values("Reduce `05 ALP Sgen` output") == []

    def test_ignores_fenced_code(self):
        assert _values("Result:\n```\nvm_pu = 1.0456\n```\n") == []

    def test_ignores_timestamps(self):
        assert _values("At 2022-01-02 21:45 the network is secure.") == []

    @pytest.mark.parametrize("text", [
        "At 18:15 the bus is at 1.02 p.u.",
        "From 16:00 to 23:45:00 the bus is at 1.02 p.u.",
        "07:45–09:45: the bus is at 1.02 p.u.",
        "On 6 January the bus is at 1.02 p.u.",
        "On January 6 the bus is at 1.02 p.u.",
        "On 6 Jan the bus is at 1.02 p.u.",
        "On Jan 6 the bus is at 1.02 p.u.",
        "On 06 Jan 2000 the bus is at 1.02 p.u.",
        "On January 6, 2000 at 18:15 the bus is at 1.02 p.u.",
        "On the 6th of January the bus is at 1.02 p.u.",
        "At 2000\u201101\u201101\u202f00:00:00 the bus is at 1.02 p.u.",
    ])
    def test_ignores_dates_and_times_in_prose(self, text):
        """A clock time or a day and month is not a figure; their digits
        used to be grounded by a timestamp's, which no longer counts."""
        assert _values(text) == [1.02]

    def test_a_lowercase_month_word_is_not_a_date(self):
        assert _values("The 6 may change; 3 units in March 2000.") == [6, 3, 2000]

    def test_ignores_conductors_own_step_limit_note(self):
        answer = ("The grid had 2 violations.\n\n*CONDUCTOR used all 16 steps allowed for one question "
                  "(`DT_MAX_AGENT_TURNS`) before finishing; this answer is built from the results so far.*")
        assert _values(answer) == [2]
        stuck = "Partial.\n\n⚠️ CONDUCTOR stopped this question: the model kept repeating (Single N-1 contingency)"
        assert _values(stuck) == []

    def test_ignores_ordered_list_markers(self):
        assert _values("1. First point\n2. Second point\n") == []

    def test_ignores_table_alignment_rows(self):
        table = "| Bus | V |\n| :--- | ---: |\n| Echo | 1.05 |\n"
        assert _values(table) == [1.05]

    def test_empty_answer(self):
        assert extract_claims("") == []


class TestSignedCurrency:
    @pytest.mark.parametrize("text", ["-€24.27/h", "a saving of -€24.27/h", "- € 24.27 per hour", "€-24.27/h"])
    def test_a_sign_before_the_currency_is_kept(self, text):
        assert _values(text) == [-24.27]

    def test_without_a_sign_it_stays_positive(self):
        assert _values("costs €24.27, range 10–20 €") == [24.27, 10, 20]


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
        record = {"tool_results": [{"name": "run_rsa", "result": {"element": "Oscar 10kV"}}]}
        assert 10.0 in {source.value for source in collect_sources(record)}

    def test_booleans_are_not_numbers(self):
        record = {"tool_results": [{"name": "t", "result": {"converged": True}}]}
        assert collect_sources(record) == []

    def test_a_timestamp_is_not_a_source_of_numbers(self):
        """Its day, hour and minute grounded small fabricated counts: "9
        violations" traced to the day of 2000-01-09, "18 buses" to its hour."""
        record = {
            "user": "Is the grid secure at 2000-01-09 18:15?",
            "tool_calls": [{"name": "run_rsa", "args": {"timestamp": "2000-01-09 18"}}],
            "tool_results": [{"name": "run_rsa", "result": {
                "timestamp": "2000-01-09 18:15:00", "first_violation_timestamps": ["2000-01-09T18:30:00Z"],
                "text": "Peak at 19:45 on 9 January", "total_violations": 3}}],
        }
        assert {source.value for source in collect_sources(record)} == {3.0}

    def test_a_fabricated_count_is_not_grounded_by_a_timestamp(self):
        record = {
            "user": "Is the grid secure?",
            "tool_calls": [{"name": "run_rsa", "args": {}}],
            "tool_results": [{"name": "run_rsa", "result": {
                "timestamp": "2000-01-09 18:15:00", "total_violations": 3}}],
            "assistant": "At 18:15 on 9 January there are 9 violations across 18 buses; 3 in total.",
        }
        verdicts = {v.claim.value: v.status for v in check_turn(record).verdicts}
        assert verdicts == {9.0: UNGROUNDED, 18.0: UNGROUNDED, 3.0: GROUNDED}

    def test_digits_in_names_still_count(self):
        record = {"tool_results": [{"name": "t", "result": {"element": "Foxtrot Trf 1", "bus_name": "Bus_18"}}]}
        assert {source.value for source in collect_sources(record)} == {1.0, 18.0}

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

    05 ALP Sgen can clear Oscar with 3.354 MW. It cannot clear Alpha at
    all: that movement was withdrawn as undeliverable, which is exactly why
    the model reached for the neighbouring number.
    """
    return {
        "user": "Attribute the worst violation.",
        "tool_calls": [{"name": "compute_violation_attribution", "args": {}}],
        "tool_results": [{"name": "compute_violation_attribution", "result": {
            "violations": [
                {"element": "Oscar 10kV", "value": 1.0456, "limit": 1.045,
                 "drivers": [{"source": "05 ALP Sgen", "current_p_mw": 3.537,
                              "relief_mw": -3.354, "relief_mw_feasible": True}],
                 "recommended_action": {
                     "kind": "single_source",
                     "text": "Reduce 05 ALP Sgen by 3.35 MW to bring Oscar 10kV back to 1.045.",
                 }},
                {"element": "Alpha 10.5 kV", "value": 1.0506, "limit": 1.045,
                 "drivers": [{"source": "05 ALP Sgen", "current_p_mw": 3.537,
                              "relief_mw": None, "relief_mw_feasible": False,
                              "max_deliverable_mw": -3.537}],
                 "recommended_action": {
                     "kind": "none_sufficient",
                     "text": "No single source can clear Alpha 10.5 kV.",
                 }},
            ],
        }}],
        "assistant": assistant,
    }


class TestSubjectVocabulary:
    def test_only_violated_elements_can_be_subjects(self):
        record = _attribution_record("")
        assert subject_names(record) == {"Oscar 10kV", "Alpha 10.5 kV"}

    def test_a_name_contained_in_a_longer_one_is_dropped(self):
        # The dispatch table names its rows `element` too, so the generator
        # "Alpha" entered the vocabulary and shadowed the violated bus
        # "Alpha 10.5 kV" the sentence was actually about. Prose that says
        # "Alpha" has not said which one.
        record = _attribution_record("")
        record["tool_results"][0]["result"]["violations"].append(
            {"element": "Alpha", "value": 1.05, "limit": 1.045, "drivers": []}
        )
        assert "Alpha" not in subject_names(record)


class TestMisattribution:
    def test_a_figure_from_another_violation_is_caught(self):
        # Observed live. 3.35 MW clears Oscar; on Alpha the movement
        # was withdrawn as impossible, and the model filled the gap with the
        # neighbouring violation's number.
        record = _attribution_record(
            "### Attribution\n"
            "The worst violation is at Alpha 10.5 kV (1.0506 p.u.).\n"
            "05 ALP Sgen is the main contributor, requiring a reduction of 3.35 MW."
        )
        (verdict,) = check_turn(record).misattributed
        assert verdict.claim.value == 3.35
        assert verdict.subject == "Alpha 10.5 kV"
        assert verdict.belongs_to == "Oscar 10kV"

    def test_the_same_figure_under_its_own_element_is_grounded(self):
        record = _attribution_record(
            "### Attribution\n"
            "Oscar 10kV is violating. Reduce 05 ALP Sgen by 3.35 MW."
        )
        report = check_turn(record)
        assert report.misattributed == []
        assert len(report.grounded) == len(report.verdicts)

    def test_a_heading_ends_the_subject(self):
        # Without this the last element named in one section would still be
        # the subject halfway through the next.
        record = _attribution_record(
            "### Attribution\nThe worst violation is at Alpha 10.5 kV.\n"
            "### Dispatch\nThe unit moved by 3.35 MW."
        )
        assert check_turn(record).misattributed == []

    def test_turn_level_figures_are_not_attributed_to_the_subject(self):
        # Totals, thresholds, dispatch rows and call arguments belong to no
        # violated element. A sentence about a unit's setpoint is not a claim
        # about whichever bus the paragraph opened with — leaving these scoped
        # produced a false positive on every dispatch figure.
        record = _attribution_record(
            "### Attribution\nThe worst violation is at Alpha 10.5 kV.\n"
            "Alpha was reduced from 3.54 MW to 3.46 MW."
        )
        record["tool_results"].append({"name": "optimize_flexibility", "result": {
            "activated_resources": [
                {"element": "Alpha", "Pg_base": 3.537, "Pg_new": 3.4612},
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

    @pytest.mark.parametrize("written", ["Bus_0 -> Bus_1 [L0]", "Bus_0–Bus_1 [L0]", "Bus_0 → Bus_1 [L0]",
                                         "Bus_0 - Bus_1 [L0]", "Bus_0 to Bus_1 [L0]",
                                         # Its short forms (seen live: "Line L0 (Bus_0–Bus_1)").
                                         "Line L0 (Bus_0–Bus_1)", "L0", "Bus_0–Bus_1"])
    def test_a_line_is_recognised_however_its_arrow_is_written(self, written):
        # Observed live: "line Bus_0–Bus_1 [L0] at 114.7%" was read as written
        # about Bus_29, the violated bus named just before it.
        record = {"user": "q", "assistant": (
            "Two violations: **Bus_29 at 0.943 p.u.** (below 0.950) and line "
            f"**{written} at 114.7%** loading."),
            "tool_calls": [{"name": "run_rsa", "args": {}}],
            "tool_results": [{"name": "run_rsa", "result": {"violations": [
                {"element": "Bus_29", "violation_type": "bus_vm_pu", "value": 0.94327, "violated": True},
                {"element": "Bus_0 -> Bus_1 [L0]", "violation_type": "line_loading", "value": 114.6644,
                 "violated": True}]}}]}
        report = check_turn(record)
        assert report.misattributed == []
        line = next(v for v in report.verdicts if v.claim.value == 114.7)
        assert line.status == GROUNDED and line.subject == "Bus_0 -> Bus_1 [L0]"

    def test_a_short_form_two_branches_share_names_neither(self):
        # Parallel lines: "Bus_0–Bus_1" alone does not say which one.
        record = {"user": "q", "assistant": "Bus_29 at 0.943 p.u.; the line Bus_0–Bus_1 at 114.7%.",
                  "tool_calls": [{"name": "run_rsa", "args": {}}],
                  "tool_results": [{"name": "run_rsa", "result": {"violations": [
                      {"element": "Bus_29", "value": 0.94327, "violated": True},
                      {"element": "Bus_0 -> Bus_1 [L0]", "value": 114.6644, "violated": True},
                      {"element": "Bus_0 -> Bus_1 [L1]", "value": 60.0, "violated": True}]}}]}
        line = next(v for v in check_turn(record).verdicts if v.claim.value == 114.7)
        assert line.subject == "Bus_29"           # unchanged: the nearest name it can be sure of

    def test_misattributed_does_not_count_as_traceable(self):
        record = _attribution_record(
            "### Attribution\nThe worst violation is at Alpha 10.5 kV, "
            "requiring 3.35 MW."
        )
        report = check_turn(record)
        assert report.rate < 1.0
        assert MISATTRIBUTED in {v.status for v in report.verdicts}


class TestClarifyingQuestions:
    from llm_agent.agent import provenance as _p

    @pytest.mark.parametrize("answer, calls, asking", [
        ("Which line?\n1. L30\n2. L31", [], True),
        ("Which line?\n\n- **1.** L30\n- **2.** L31", [{"name": "locate_network_element"}], True),
        ("Which line?\n1. L30\n2. L31", [{"name": "run_rsa"}], False),            # a study ran: an answer
        ("1. Cheapest 26.48 €/h\n2. Least import\nWhich do you prefer?", [], False),  # results, then a question
        ("Which line do you mean?", [], False),                                      # no choices to exempt
        ("I can't run that.", [], False),
    ])
    def test_what_counts_as_asking(self, answer, calls, asking):
        assert self._p.is_clarifying_question(answer, calls) is asking

    def test_only_the_choices_are_left_out(self):
        text = "L30 is at 77.9 %. How far below?\n1. 70 %\n2. 60 %"
        assert self._p.answer_to_check(text) == "L30 is at 77.9 %. How far below?"
        assert self._p.answer_to_check(text, [{"name": "run_rsa"}]) == text



@pytest.mark.parametrize("text, values", [
    ("Bus 7 is at 0.98 p.u.", [0.98]),
    ("line 12 is at 95 %", [95.0]),
    ("transformer 1 carries 40 MW", [40.0]),
    ("an N-0 assessment, not an N-1 check, found 3 violations", [3.0]),
    ("an N-1-1 screen", []),
    ("line 95 % loaded", [95.0]),                  # a unit after it: a figure again
    ("3 lines are out", [3.0]),
])
def test_a_name_is_not_a_figure(text, values):
    # Observed live: the "0" of "N-0" and the "1" of "N-1" graded as figures.
    assert [c.value for c in extract_claims(text)] == values
