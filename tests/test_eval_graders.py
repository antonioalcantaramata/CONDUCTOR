"""
Offline grading of session logs.

The failure these exist for is the one a pass rate cannot see: every figure in
an answer correct, the prose coherent, and the turn as a whole describing an
operating point nobody asked about. It was found by comparing tool results
against each other, and it was found four times in ten runs — often enough to
matter and rarely enough to be mistaken for model variance.

The reference case below is that turn, reproduced from the real log.
"""

from evaluation.graders.integrity import grade_composition
from evaluation.graders.provenance import grade_provenance
from evaluation.report import grade_record, render_report
from evaluation.session_log import FAIL, PARAMETERISATION, WARN, load_records


def _record(**overrides):
    """A clean two-tool turn at a single requested operating point."""
    record = {
        "turn": 1,
        "status": "completed",
        "user": "Run an RSA at 2022-01-02 21:45 and tell me which unit is responsible.",
        "tool_calls": [
            {"name": "run_rsa", "args": {"timestamp": "2022-01-02 21:45"}},
            {"name": "compute_violation_attribution", "args": {"timestamp": "2022-01-02 21:45"}},
        ],
        "tool_results": [
            {"name": "run_rsa",
             "result": {"timestamp": "2022-01-02 21:45:00", "total_violations": 11}},
            {"name": "compute_violation_attribution",
             "result": {"timestamp": "2022-01-02 21:45:00", "n_sources_tested": 20}},
        ],
        "assistant": "At that operating point there are 11 violations across 20 tested sources.",
    }
    record.update(overrides)
    return record


def _codes(findings):
    return [f.code for f in findings]


class TestCompositionIntegrity:
    def test_clean_turn_has_no_findings(self):
        assert grade_composition(_record()) == []

    def test_two_tools_at_different_operating_points(self):
        # The observed failure: the attribution call fell back to the
        # simulation clock while the assessment ran at the requested time.
        record = _record(tool_results=[
            {"name": "run_rsa",
             "result": {"timestamp": "2022-01-02 21:45:00", "total_violations": 11}},
            {"name": "compute_violation_attribution",
             "result": {"timestamp": "2022-01-01 00:00:00", "n_sources_tested": 20}},
        ])
        findings = grade_composition(record)
        assert "inconsistent_context" in _codes(findings)
        assert all(f.channel == PARAMETERISATION for f in findings)

    def test_whole_turn_ran_at_the_wrong_point(self):
        # Internally consistent and still the wrong answer: both tools agree
        # with each other and neither agrees with the question. Cross-tool
        # comparison alone would pass this.
        record = _record(tool_results=[
            {"name": "run_rsa", "result": {"timestamp": "2022-01-01 00:00:00"}},
            {"name": "compute_violation_attribution", "result": {"timestamp": "2022-01-01 00:00:00"}},
        ])
        findings = grade_composition(record)
        assert _codes(findings).count("requested_point_not_used") == 2
        assert all(f.severity == FAIL for f in findings)

    def test_date_only_request_is_satisfied_by_any_time_that_day(self):
        record = _record(
            user="What happened on 2022-01-02?",
            tool_results=[{"name": "run_rsa", "result": {"timestamp": "2022-01-02 21:45:00"}}],
        )
        assert "requested_point_not_used" not in _codes(grade_composition(record))

    def test_two_requested_points_are_not_graded(self):
        # A deliberate comparison of two timestamps is legitimate, and without
        # a catalog there is no way to say which tool belonged to which.
        record = _record(
            user="Compare 2022-01-02 21:45 with 2022-06-01 12:00.",
            tool_results=[
                {"name": "run_rsa", "result": {"timestamp": "2022-01-02 21:45:00"}},
                {"name": "run_rsa", "result": {"timestamp": "2022-06-01 12:00:00"}},
            ],
        )
        assert "requested_point_not_used" not in _codes(grade_composition(record))

    def test_inherited_constraint_is_recorded_as_a_warning(self):
        # The loop restored what the model dropped, so the answer is right.
        # Counting these is how often it would have been wrong unaided.
        record = _record()
        record["tool_calls"][1]["inherited"] = {"timestamp": "2022-01-02 21:45"}
        findings = grade_composition(record)
        assert _codes(findings) == ["constraint_inherited"]
        assert findings[0].severity == WARN

    def test_live_integrity_warnings_are_surfaced(self):
        record = _record()
        record["tool_results"][0]["result"]["_integrity_warnings"] = [
            {"code": "opf_self_inconsistent", "message": "reported success outside its own bounds"},
        ]
        assert "opf_self_inconsistent" in _codes(grade_composition(record))

    def test_replayed_findings_are_not_counted_twice(self):
        # A turn recorded while the live guard was active already carries the
        # cross-tool findings in its payload. Reporting the replay and the
        # embedded copy shows one problem as two.
        record = _record(tool_results=[
            {"name": "run_rsa", "result": {"timestamp": "2022-01-02 21:45:00"}},
            {"name": "compute_violation_attribution", "result": {
                "timestamp": "2022-01-01 00:00:00",
                "_integrity_warnings": [
                    {"code": "inconsistent_context", "field": "timestamp",
                     "message": "combines analyses of different operating points"},
                ],
            }},
        ])
        assert _codes(grade_composition(record)).count("inconsistent_context") == 1

    def test_tool_error_is_reported(self):
        record = _record()
        record["tool_results"][1]["result"] = {"error": "backend unreachable"}
        assert "tool_error" in _codes(grade_composition(record))

    def test_failed_turn_is_reported(self):
        record = _record(status="execution_error", error_classification="execution_error",
                         runner_error="boom")
        findings = grade_composition(record)
        assert findings[0].severity == FAIL
        assert "boom" in findings[0].detail

    def test_completed_with_warning_is_not_a_failure(self):
        findings = grade_composition(_record(status="completed_with_warning"))
        assert [f.severity for f in findings] == [WARN]

    def test_moving_the_simulation_cursor_is_reported(self):
        # Observed live on a read-only comparison: the agent advanced the clock
        # twice instead of passing timestamps, so a question that only asked to
        # compare changed what the next question would see.
        record = _record(
            user="Compare the network at 2022-01-02 21:45 with 2022-01-03 21:45.",
            tool_calls=[
                {"name": "advance_timestamp", "args": {"target_timestamp": "2022-01-02 21:45"}},
                {"name": "run_rsa", "args": {"data_source": "measurements"}},
            ],
            tool_results=[
                # The wrapper's shape, not the endpoint's: it renames
                # `new_timestamp` to `current_timestamp`, and the grader read
                # only the endpoint key when first written.
                {"name": "advance_timestamp",
                 "result": {"current_timestamp": "2022-01-02 21:45:00",
                            "timestamps_traversed": ["2022-01-02 21:45:00"]}},
                {"name": "run_rsa", "result": {"timestamp": "2022-01-02 21:45:00"}},
            ],
        )
        findings = [f for f in grade_composition(record) if f.code == "simulation_cursor_moved"]
        assert len(findings) == 1
        assert findings[0].severity == WARN
        assert "2022-01-02 21:45:00" in findings[0].detail
        # The read-only path existed, and the finding has to say so — otherwise
        # it reads as a complaint about advancing at all.
        assert "without changing state" in findings[0].detail

    def test_advancing_on_request_is_not_dressed_up_as_avoidable(self):
        # The turn advanced and then named the operating point explicitly, so
        # nothing silently depended on the cursor it had moved.
        record = _record(
            user="Move to 2022-01-02 21:45 and run an assessment there.",
            tool_calls=[
                {"name": "advance_timestamp", "args": {"target_timestamp": "2022-01-02 21:45"}},
                {"name": "run_rsa", "args": {"timestamp": "2022-01-02 21:45"}},
            ],
            tool_results=[
                {"name": "advance_timestamp",
                 "result": {"status": "success", "new_timestamp": "2022-01-02 21:45:00"}},
                {"name": "run_rsa", "result": {"timestamp": "2022-01-02 21:45:00"}},
            ],
        )
        (finding,) = [f for f in grade_composition(record) if f.code == "simulation_cursor_moved"]
        assert "without changing state" not in finding.detail

    def test_a_turn_that_reads_only_is_silent_about_state(self):
        assert "simulation_cursor_moved" not in _codes(grade_composition(_record()))

    def test_answer_with_no_tool_behind_it_is_flagged(self):
        record = _record(tool_calls=[], tool_results=[],
                         assistant="I cannot run that analysis.")
        assert "no_tool_evidence" in _codes(grade_composition(record))


class TestProvenanceGrading:
    def test_grounded_answer_produces_no_findings(self):
        findings, report = grade_provenance(_record())
        assert findings == []
        assert report.rate == 1.0

    def test_misattributed_figure_fails(self):
        record = _record(
            tool_results=[{"name": "compute_violation_attribution", "result": {
                "timestamp": "2022-01-02 21:45:00",
                "violations": [
                    # The engine reports the movement as a magnitude in its
                    # resolved sentence, which is where the model read it.
                    {"element": "Oscar 10kV", "drivers": [],
                     "recommended_action": {
                         "text": "Reduce 05 ALP Sgen by 3.35 MW to bring Oscar 10kV back."}},
                    {"element": "Alpha 10.5 kV", "drivers": [],
                     "recommended_action": {
                         "text": "No single source can clear Alpha 10.5 kV."}},
                ],
            }}],
            assistant="### Attribution\nAlpha 10.5 kV needs a reduction of 3.35 MW.",
        )
        findings, report = grade_provenance(record)
        assert [f.code for f in findings] == ["misattributed_figure"]
        assert findings[0].severity == FAIL
        assert "Alpha 10.5 kV" in findings[0].detail
        assert len(report.misattributed) == 1

    def test_fabricated_figure_fails(self):
        record = _record(assistant="Curtail the Alpha unit by 6.24 MW.")
        findings, _ = grade_provenance(record)
        assert [f.code for f in findings] == ["ungrounded_figure"]
        assert findings[0].severity == FAIL


class TestReport:
    def test_grade_record_runs_every_grader(self):
        grade = grade_record(_record())
        assert grade.findings == ()
        assert grade.n_claims == 2 and grade.n_grounded == 2
        assert grade.n_misattributed == 0
        assert not grade.failed

    def test_render_reports_totals(self):
        text = render_report([grade_record(_record())], source="log.jsonl")
        assert "1 turn(s)" in text
        assert "traceable 100.0%" in text

    def test_render_handles_an_empty_log(self):
        assert "no turns" in render_report([], source="log.jsonl")


class TestSessionLog:
    def test_malformed_lines_are_skipped(self, tmp_path):
        # A log truncated by a crash is still worth grading — the crash is
        # often exactly what is being diagnosed.
        log = tmp_path / "s.jsonl"
        log.write_text('{"turn": 1}\nnot json\n{"turn": 2}\n', encoding="utf-8")
        assert [r["turn"] for r in load_records(log)] == [1, 2]
