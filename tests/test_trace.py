"""The operator-facing audit trace.

This is the one guard-layer component the operator actually sees, which changes
what the tests have to protect. Elsewhere a false negative costs a missed
finding; here it costs a person acting on a figure the system quietly doubted.

So the properties pinned below are mostly about *honesty of presentation*: the
collapsed label must carry any concern, the panel must never certify what it
cannot check, and the limits block must never be empty.
"""

import pytest

from llm_agent.agent.trace import BLIND_SPOTS, build


def _record(assistant, calls=None, results=None, user="what is the status?"):
    return {
        "user": user,
        "grid": {},
        "assistant": assistant,
        "tool_calls": calls if calls is not None else [
            {"name": "run_rsa", "args": {"timestamp": "2000-01-03 02:15"}},
        ],
        "tool_results": results if results is not None else [
            {"name": "run_rsa", "result": {
                "secure": False, "total_violations": 1,
                "violations": [{"element": "Bus_3", "value": 0.9366,
                                "limit": 0.94, "violated": True}],
            }},
        ],
    }


class TestWhatRan:
    def test_calls_appear_in_execution_order(self):
        trace = build(_record(
            "Two studies ran.",
            calls=[{"name": "run_rsa", "args": {}}, {"name": "optimize_flexibility", "args": {}}],
            results=[{"name": "run_rsa", "result": {}},
                     {"name": "optimize_flexibility", "result": {}}],
        ))
        assert [s.name for s in trace.steps] == ["run_rsa", "optimize_flexibility"]
        assert [s.order for s in trace.steps] == [1, 2]

    def test_the_same_tool_twice_is_not_merged(self):
        """A turn that studies two operating points with one tool is exactly
        the case an operator most needs to see separated."""
        trace = build(_record(
            "Compared two moments.",
            calls=[{"name": "run_rsa", "args": {"timestamp": "A"}},
                   {"name": "run_rsa", "args": {"timestamp": "B"}}],
            results=[{"name": "run_rsa", "result": {}}, {"name": "run_rsa", "result": {}}],
        ))
        assert len(trace.steps) == 2
        assert [s.args["timestamp"] for s in trace.steps] == ["A", "B"]

    def test_an_inherited_argument_is_surfaced(self):
        """The only place the repair guard becomes visible to a human."""
        trace = build(_record(
            "Done.",
            calls=[{"name": "run_rsa", "args": {"vm_upper_pu": 1.045},
                    "inherited": {"vm_upper_pu": {"value": 1.045, "from": "the request"}}}],
            results=[{"name": "run_rsa", "result": {}}],
        ))
        assert trace.steps[0].inherited
        assert "from the request" in trace.steps[0].inherited_summary

    def test_a_failed_tool_is_marked_and_explained(self):
        trace = build(_record(
            "It failed.",
            calls=[{"name": "run_rsa", "args": {}}],
            results=[{"name": "run_rsa", "result": {"error": "backend unreachable"}}],
        ))
        assert trace.steps[0].ok is False
        assert "unreachable" in trace.steps[0].error
        assert any("unreachable" in c for c in trace.concerns)

    def test_integrity_warnings_reach_the_operator(self):
        trace = build(_record(
            "All fine.",
            calls=[{"name": "run_rsa", "args": {}}],
            results=[{"name": "run_rsa", "result": {"_integrity_warnings": [
                {"code": "opf_bound_violation", "field": "x", "message": "left a bus outside its bounds"},
            ]}}],
        ))
        assert any("outside its bounds" in c for c in trace.concerns)


class TestWhereTheNumbersCameFrom:
    def test_a_grounded_figure_is_traced_to_its_source(self):
        trace = build(_record("Bus_3 is at 0.9366 p.u."))
        figure = next(f for f in trace.figures if f.value == pytest.approx(0.9366))
        assert figure.traced
        assert figure.source

    def test_an_invented_figure_is_not_traced(self):
        trace = build(_record("Bus_3 is at 0.8123 p.u."))
        figure = next(f for f in trace.figures if f.value == pytest.approx(0.8123))
        assert not figure.traced
        assert "not found" in figure.detail

    def test_every_status_has_operator_readable_wording(self):
        """The panel is read by an engineer, not by whoever wrote the check.
        A raw status string like `misattributed` is not an explanation."""
        trace = build(_record("Bus_3 is at 0.9366 p.u., against a 0.94 limit."))
        for figure in trace.figures:
            assert figure.detail != figure.status
            assert " " in figure.detail


class TestHeadlineHonesty:
    """The collapsed label is the only part every operator sees."""

    def test_a_concern_is_always_in_the_label(self):
        """A panel nobody opens cannot warn anybody."""
        trace = build(_record("Reduce Load_2 by 99.9 MW."))
        assert trace.has_concerns
        assert "⚠" in trace.headline()

    def test_a_clean_turn_does_not_cry_wolf(self):
        trace = build(_record("Bus_3 is at 0.9366 p.u., below its 0.94 limit."))
        assert trace.concerns == ()
        assert "⚠" not in trace.headline()

    def test_the_label_describes_and_does_not_certify(self):
        """A traced figure can still be wrong. A badge implying otherwise is
        worse than no badge at all."""
        trace = build(_record("Bus_3 is at 0.9366 p.u."))
        label = trace.headline().lower()
        assert "traced" in label
        for word in ("verified", "correct", "validated", "confirmed", "safe"):
            assert word not in label

    def test_counts_are_accurate(self):
        trace = build(_record("Bus_3 is at 0.9366 p.u., not 0.8123 p.u."))
        assert len(trace.figures) >= 2
        assert trace.n_traced < len(trace.figures)
        assert f"{trace.n_traced}/{len(trace.figures)}" in trace.headline()


class TestLimits:
    def test_limits_are_never_empty(self):
        """A count without its blind spot reads as a guarantee."""
        assert build(_record("Bus_3 is at 0.9366 p.u.")).limits

    def test_the_provenance_blind_spot_is_stated_whenever_figures_are_traced(self):
        trace = build(_record("Bus_3 is at 0.9366 p.u."))
        assert BLIND_SPOTS["provenance"] in trace.limits

    def test_the_characterisation_blind_spot_is_always_stated(self):
        """It applies whether or not the check fired: a payload with no
        feasibility field is exactly the case it cannot see."""
        assert BLIND_SPOTS["characterisation"] in build(_record("All secure.")).limits


class TestRobustness:
    def test_an_empty_turn_produces_an_empty_trace(self):
        trace = build({"user": "hi", "assistant": "Hello.", "tool_calls": [], "tool_results": []})
        assert trace.steps == ()
        assert "no tools" in trace.headline()

    def test_a_result_missing_its_call_does_not_raise(self):
        """Logs truncated by a crash are still worth showing."""
        trace = build(_record("Done.", calls=[{"name": "run_rsa", "args": {}}], results=[]))
        assert len(trace.steps) == 1
        assert trace.steps[0].ok

    def test_empty_arguments_are_not_shown(self):
        """`timestamp=''` in the panel reads as a setting, not an omission."""
        trace = build(_record(
            "Done.",
            calls=[{"name": "run_rsa", "args": {"timestamp": "", "data_source": None,
                                                "vm_upper_pu": 1.05}}],
            results=[{"name": "run_rsa", "result": {}}],
        ))
        assert trace.steps[0].args == {"vm_upper_pu": 1.05}


class TestAuditFindings:
    """Regressions for defects found in an independent audit of this module.

    Each of these shipped in the first version. They are grouped because they
    share a cause: the panel is operator-facing, so a check that merely *fails
    to report* is a safety defect, not a cosmetic one.
    """

    def test_a_concern_alone_is_enough_to_render(self):
        """The panel was suppressed when a turn had no tools and no figures —
        which is exactly the shape of an answer that skipped the question."""
        trace = build({
            "user": "How much MW of headroom is there?",
            "assistant": "The network is operating within its secure envelope.",
            "grid": {}, "tool_calls": [], "tool_results": [],
        })
        assert trace.concerns, "a completeness gap should be raised"
        # The renderer's guard, mirrored: it must not drop a trace with concerns.
        assert trace.steps or trace.figures or trace.concerns

    def test_the_completeness_caveat_accompanies_the_gap_it_qualifies(self):
        """It softens a reported omission, so it belongs on turns that have
        one. The first version had the condition inverted."""
        gap = build(_record("It is secure.", user="how much MW of headroom?"))
        clean = build(_record("Bus_3 is at 0.9366 p.u."))
        assert BLIND_SPOTS["completeness"] in gap.limits
        assert BLIND_SPOTS["completeness"] not in clean.limits

    def test_one_problem_is_counted_once(self):
        """A figure repeated in the prose is still one thing to check, and a
        count the operator can see is padded is a count they stop believing."""
        trace = build(_record("Reduce Load_2 by 99.9 MW. Yes, 99.9 MW is required."))
        assert len(trace.concerns) == len(set(trace.concerns))
        assert len(trace.concerns) == 1

    def test_a_repeated_integrity_warning_is_counted_once(self):
        trace = build(_record(
            "Done.",
            calls=[{"name": "run_rsa", "args": {}}, {"name": "run_rsa", "args": {}}],
            results=[{"name": "run_rsa", "result": {"_integrity_warnings": [
                        {"code": "c", "field": "f", "message": "left a bus outside its bounds"}]}},
                     {"name": "run_rsa", "result": {"_integrity_warnings": [
                        {"code": "c", "field": "f", "message": "left a bus outside its bounds"}]}}],
        ))
        assert len(trace.concerns) == 1

    def test_failed_tools_are_not_reported_as_no_tools(self):
        """`0 tools` after a failure is indistinguishable from a turn that ran
        nothing — the opposite reading."""
        trace = build(_record(
            "It failed.",
            calls=[{"name": "run_rsa", "args": {}}],
            results=[{"name": "run_rsa", "result": {"error": "backend unreachable"}}],
        ))
        assert "0/1 tools succeeded" in trace.headline()
        assert trace.headline() != "no tools used"

    @pytest.mark.parametrize("record", [
        {"tool_calls": [{"name": "t", "args": "not-a-dict"}], "tool_results": [{"name": "t", "result": {}}]},
        {"tool_calls": [{"name": "t", "args": {}, "inherited": "not-a-dict"}], "tool_results": [{"name": "t", "result": {}}]},
        {"tool_calls": [{"name": "t", "args": {}, "inherited": {"vm": "not-a-dict"}}], "tool_results": [{"name": "t", "result": {}}]},
        {"tool_calls": [{"name": None, "args": {}}], "tool_results": [{"name": "t", "result": {}}]},
        {"tool_calls": [{"name": 7, "args": {}}], "tool_results": [{"name": "t", "result": {}}]},
        {"tool_calls": ["not-a-dict"], "tool_results": [{"name": "t", "result": {}}]},
        {"tool_calls": [{"name": "t", "args": {}}], "tool_results": ["not-a-dict"]},
        {"tool_calls": [], "tool_results": [{"name": "t", "result": {"v": 1.5}}]},
    ])
    def test_a_malformed_record_never_raises_through_the_render_path(self, record):
        """`build()` is wrapped in a try/except; the render is not. Anything
        the renderer dereferences must therefore survive a broken record."""
        trace = build({"user": "x", "assistant": "The value is 1.5 MW.",
                       "grid": {}, **record})
        for step in trace.steps:                 # what _render_trace touches
            step.name.replace("_", " ")
            dict(step.args)
            assert isinstance(step.inherited_summary, str)
        for figure in trace.figures:
            assert f"{figure.value:g}"
        assert isinstance(trace.headline(), str)


class TestDerivedAndMisattributed:
    """Statuses the first test suite never constructed with a real payload.

    Mutation testing showed `traced` could be broadened to include either of
    these without a single test failing — which would make the headline
    over-claim, the one thing it must never do.
    """

    DIVISION = {
        "user": "what percentage?",
        "grid": {},
        "tool_calls": [{"name": "get_current_conditions", "args": {}}],
        "tool_results": [{"name": "get_current_conditions", "result": {
            "ext_grid": {"P_import_mw": 176.7793},
            "totals": {"total_load_mw": 220.1206}}}],
    }

    def test_a_shown_division_is_not_counted_as_traced(self):
        """The model computed it. Counting it as traced to tool output would
        let an answer raise its own score by showing more arithmetic."""
        from llm_agent.agent.provenance import DERIVED

        trace = build({**self.DIVISION,
                       "assistant": "176.7793 / 220.1206 = 0.803101, so 80.31%."})
        derived = [f for f in trace.figures if f.status == DERIVED]
        assert derived, "the division should be recognised"
        assert all(not f.traced for f in derived)

    def test_a_shown_division_is_not_reported_as_a_concern(self):
        """Flagging shown arithmetic penalises exactly the transparency the
        panel exists to encourage."""
        trace = build({**self.DIVISION,
                       "assistant": "176.7793 / 220.1206 = 0.803101, so 80.31%."})
        assert trace.concerns == ()

    def test_a_misattributed_figure_is_not_traced_and_is_a_concern(self):
        record = {
            "user": "why is Alpha violating?",
            "grid": {},
            "assistant": "Alpha 10.5 kV needs 3.35 MW of relief.",
            "tool_calls": [{"name": "attr", "args": {}}],
            "tool_results": [{"name": "attr", "result": {"violations": [
                {"element": "Alpha 10.5 kV", "value": 1.051, "violated": True,
                 "drivers": [{"source": "g1", "relief_mw": 1.20}]},
                {"element": "Oscar 10.5 kV", "value": 1.049, "violated": True,
                 "drivers": [{"source": "g2", "relief_mw": 3.35}]},
            ]}}],
        }
        from llm_agent.agent.provenance import MISATTRIBUTED

        trace = build(record)
        wrong = [f for f in trace.figures if f.status == MISATTRIBUTED]
        assert wrong, "3.35 belongs to Oscar, not Alpha"
        assert all(not f.traced for f in wrong)
        assert any("different element" in c for c in trace.concerns)

    def test_characterisation_conflicts_reach_the_operator(self):
        """Mutation testing showed these could be dropped silently."""
        trace = build({
            "user": "what should I do?",
            "grid": {},
            "assistant": "Reduce Load_2 by 12.411 MW to clear Bus_3.",
            "tool_calls": [{"name": "attr", "args": {}}],
            "tool_results": [{"name": "attr", "result": {"violations": [
                {"element": "Bus_3", "violated": True, "drivers": [
                    {"source": "Load_2", "relief_mw": None, "would_require_mw": -12.411,
                     "actionable": False, "relief_mw_feasible": False}]}]}}],
        })
        assert any("actionable" in c or "not actionable" in c for c in trace.concerns)


class TestStudyVerdictContradiction:
    """An answer that contradicts a study's own verdict without quoting a
    number. The figure-anchored checks have nothing to attach to, so before
    this the most obviously wrong answer imaginable was invisible.
    """

    INSECURE = {
        "user": "is the system secure?",
        "grid": {},
        "tool_calls": [{"name": "run_rsa", "args": {}}],
        "tool_results": [{"name": "run_rsa", "result": {
            "secure": False, "total_violations": 2,
            "violations": [{"element": "Bus_3", "value": 0.93,
                            "limit": 0.94, "violated": True}]}}],
    }

    def test_calling_an_insecure_grid_fine_is_a_concern(self):
        trace = build({**self.INSECURE, "assistant": "Everything looks fine."})
        assert trace.concerns, "a reassuring answer over an insecure study"
        assert "⚠" in trace.headline()

    def test_reporting_it_correctly_is_not(self):
        trace = build({**self.INSECURE,
                       "assistant": "The system is insecure: 2 violations."})
        assert trace.concerns == ()

    def test_a_negated_claim_is_not_flagged(self):
        """"It is not secure" contains the word `secure` and is correct."""
        trace = build({**self.INSECURE,
                       "assistant": "It is not secure — two buses are outside limits."})
        assert trace.concerns == ()

    def test_a_secure_study_described_as_fine_is_not_flagged(self):
        trace = build({
            "user": "is it secure?", "grid": {}, "assistant": "Everything looks fine.",
            "tool_calls": [{"name": "run_rsa", "args": {}}],
            "tool_results": [{"name": "run_rsa", "result": {
                "secure": True, "total_violations": 0}}],
        })
        assert trace.concerns == ()

    def test_a_non_converged_study_called_solved(self):
        trace = build({
            "user": "optimise it", "grid": {}, "assistant": "The optimisation solved successfully.",
            "tool_calls": [{"name": "optimize_flexibility", "args": {}}],
            "tool_results": [{"name": "optimize_flexibility", "result": {
                "converged": False, "feasible": False}}],
        })
        assert trace.concerns


class TestRevisionIsDisclosed:
    """Reflection can rewrite an answer. An operator reading the final text
    cannot otherwise tell a check fired and the first draft was withdrawn."""

    def test_a_revised_answer_says_so(self):
        trace = build({
            "user": "what percentage?", "grid": {}, "assistant": "It is 80.31%.",
            "tool_calls": [{"name": "c", "args": {}}],
            "tool_results": [{"name": "c", "result": {"a": 176.7793, "b": 220.1206}}],
            "reflection": {"mode": "inform", "changed": True,
                           "draft": "It is 80.3%.", "final": "It is 80.31%."},
        })
        assert any("revised" in c for c in trace.concerns)

    def test_an_unrevised_answer_does_not(self):
        trace = build({
            "user": "status?", "grid": {}, "assistant": "Bus_3 is at 0.9366 p.u.",
            "tool_calls": [{"name": "run_rsa", "args": {}}],
            "tool_results": [{"name": "run_rsa", "result": {"violations": [
                {"element": "Bus_3", "value": 0.9366, "limit": 0.94, "violated": True}]}}],
            "reflection": {"mode": "warn", "changed": False},
        })
        assert not any("revised" in c for c in trace.concerns)


class TestShownArithmeticInProse:
    """`DERIVED` recognised only `/` and `\\frac`, so a model that explained its
    arithmetic in words was scored as having invented the result — the same
    transparency penalty the category exists to remove."""

    SOURCES = {
        "user": "what percentage?", "grid": {},
        "tool_calls": [{"name": "c", "args": {}}],
        "tool_results": [{"name": "c", "result": {
            "import_mw": 176.7793, "load_mw": 220.1206}}],
    }

    @pytest.mark.parametrize("answer", [
        "176.7793 / 220.1206 = 0.803101, so 80.31%.",
        r"$\frac{176.7793}{220.1206} \approx 0.803101$, so 80.31%.",
        "176.7793 ÷ 220.1206 = 0.803101, so 80.31%.",
        "Dividing 176.7793 by 220.1206 gives 0.803101, so 80.31%.",
        "176.7793 divided by 220.1206 is 0.803101, so 80.31%.",
    ])
    def test_shown_working_is_never_reported_as_invented(self, answer):
        trace = build({**self.SOURCES, "assistant": answer})
        assert trace.concerns == (), f"penalised for showing its working: {answer}"

    def test_an_unshown_result_is_still_reported(self):
        """Only *shown* arithmetic is excused. A bare figure is the case the
        check exists for."""
        trace = build({**self.SOURCES, "assistant": "The external grid covers 80.31%."})
        assert trace.concerns

    def test_prose_division_does_not_pair_unrelated_numbers(self):
        """"12.4 MW was divided between Alpha and Bravo, leaving 3.1 MW" must
        not read as 12.4 ÷ 3.1."""
        from llm_agent.agent.provenance import DERIVED, check_answer, collect_sources

        record = {"user": "x", "grid": {}, "tool_calls": [],
                  "tool_results": [{"name": "c", "result": {"a": 12.4, "b": 3.1}}]}
        report = check_answer(
            "12.4 MW was divided between Alpha and Bravo, leaving 3.1 MW.",
            collect_sources(record))
        assert all(v.status != DERIVED for v in report.verdicts)


class TestArgumentLegibility:
    """Observed in use: `…loading_pct=100· data_source=…` — the separator sat
    flush against the preceding value and read as a decimal point. An operator
    scanning a limit should never have to decide whether they are looking at
    `100` or `100.`, so every value is rendered inside its own code span.
    """

    def test_no_argument_value_ends_adjacent_to_a_separator(self):
        """Mirrors the renderer's join. A separator touching a digit is the
        defect; a bounded span is the fix."""
        trace = build(_record(
            "Done.",
            calls=[{"name": "run_rsa", "args": {
                "max_line_loading_pct": 100, "data_source": "measurements",
                "vm_upper_pu": 1.045}}],
            results=[{"name": "run_rsa", "result": {}}],
        ))
        rendered = "&nbsp; ".join(
            f"`{k}={v}`" for k, v in trace.steps[0].args.items())
        assert "100·" not in rendered and "100 ·" not in rendered
        # Every value is closed before the next key begins.
        assert rendered.count("`") == 2 * len(trace.steps[0].args)

    def test_the_inherited_line_bounds_its_values_too(self):
        """It states a limit, so it carries the same risk as the args line."""
        trace = build(_record(
            "Done.",
            calls=[{"name": "run_rsa", "args": {"vm_upper_pu": 1.045},
                    "inherited": {"vm_upper_pu": {"value": 1.045,
                                                  "from": "the request"}}}],
            results=[{"name": "run_rsa", "result": {}}],
        ))
        summary = trace.steps[0].inherited_summary
        assert "1.045" in summary and "the request" in summary
