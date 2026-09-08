"""Reflection — grading a draft answer and handing the findings back.

Two halves. The first covers `reflection.py` in isolation: what it surfaces,
what it deliberately does not, and how it words what it shows. The second
drives the real loop with a stub provider to pin the control flow — that OFF
changes nothing, that WARN observes without intervening, and that INFORM asks
once and only once.
"""

import pytest

from llm_agent.agent import loop as loop_module
from llm_agent.agent.providers.base import ModelResponse, ToolCall
from llm_agent.agent.reflection import (
    INFORM,
    OFF,
    WARN,
    check_draft,
    format_findings,
    normalise_mode,
)


def _record(results, user="what is the voltage?"):
    """A turn record shaped like the session log, carrying one tool result."""
    return {
        "user": user,
        "grid": {},
        "tool_calls": [{"name": "run_rsa", "args": {}}],
        "tool_results": [{"name": "run_rsa", "result": results}],
    }


class TestModes:
    def test_known_modes_survive(self):
        assert normalise_mode("inform") == INFORM
        assert normalise_mode("WARN") == WARN
        assert normalise_mode("off") == OFF

    @pytest.mark.parametrize("value", [None, "", "nonsense", "true"])
    def test_unrecognised_falls_back_to_off(self, value):
        """Reflection is the non-default path: an unreadable setting must not
        silently switch on an intervention nobody asked for."""
        assert normalise_mode(value) == OFF


class TestWhatGetsSurfaced:
    def test_grounded_figure_produces_nothing(self):
        record = _record({"bus": "S2 10kV", "vm_pu": 1.049})
        findings, report = check_draft("S2 10kV is at 1.049 p.u.", record)
        assert findings == ()
        assert report.rate == 1.0

    def test_invented_figure_is_surfaced(self):
        record = _record({"bus": "S2 10kV", "vm_pu": 1.049})
        findings, _ = check_draft("S2 10kV is at 1.072 p.u.", record)
        assert len(findings) == 1
        assert findings[0].claim.value == pytest.approx(1.072)

    def test_imprecise_figure_is_not_surfaced(self):
        """Right figure, wrong precision is a transcription slip. Showing it
        invites fiddling with numbers that were essentially right, so the
        offline grader keeps it as a warning and reflection stays quiet."""
        record = _record({"bus": "S2 10kV", "vm_pu": 1.04936})
        findings, report = check_draft("S2 10kV is at 1.0493 p.u.", record)
        assert findings == ()
        # Still visible to the instrument, just not put to the agent.
        assert report.imprecise or report.grounded

    def test_no_numbers_no_findings(self):
        record = _record({"bus": "S2 10kV", "vm_pu": 1.049})
        findings, _ = check_draft("The system is secure.", record)
        assert findings == ()


class TestWording:
    """The register is the design. These pin it against drift."""

    def test_findings_carry_the_nearest_source(self):
        record = _record({"bus": "S2 10kV", "vm_pu": 1.049})
        findings, _ = check_draft("S2 10kV is at 1.072 p.u.", record)
        message = format_findings(findings)
        assert "1.049" in message, "the agent must be able to see what it likely meant"

    def test_message_does_not_command_a_correction(self):
        record = _record({"bus": "S2 10kV", "vm_pu": 1.049})
        findings, _ = check_draft("S2 10kV is at 1.072 p.u.", record)
        message = format_findings(findings).lower()
        assert "you may keep the answer" in message
        for directive in ("you must", "fix this", "remove the"):
            assert directive not in message

    def test_message_admits_the_check_can_be_wrong(self):
        record = _record({"bus": "S2 10kV", "vm_pu": 1.049})
        findings, _ = check_draft("S2 10kV is at 1.072 p.u.", record)
        assert "imperfect" in format_findings(findings).lower()


# ---------------------------------------------------------------------------
# Control flow, against the real loop
# ---------------------------------------------------------------------------


class StubProvider:
    name = "stub"
    model = "stub-1"

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def preflight(self):
        pass

    def chat(self, messages, tools, system, on_event=None):
        self.calls.append({"messages": [dict(m) for m in messages], "system": system})
        if not self._responses:
            return ModelResponse(text="(exhausted)")
        return self._responses.pop(0)


@pytest.fixture
def loop_stub(monkeypatch):
    """Install a stub provider, a tool returning a fixed result, and a log sink."""
    written = []

    def install(responses, tool_result=None):
        provider = StubProvider(responses)
        result = tool_result if tool_result is not None else {"bus": "S2 10kV", "vm_pu": 1.049}
        monkeypatch.setattr(loop_module, "get_provider", lambda *a, **k: provider)
        monkeypatch.setattr(loop_module, "get_system_prompt", lambda: "SYSTEM")
        monkeypatch.setattr(loop_module, "TOOLS", [])
        monkeypatch.setattr(loop_module, "TOOL_DISPATCH", {"run_rsa": lambda **kw: result})
        monkeypatch.setattr(loop_module, "_append_to_log", written.append)
        monkeypatch.setattr(loop_module, "_grid_facts", lambda: {})
        return provider

    install.written = written
    return install


def _with_tool_then(text):
    """A scripted run: one tool call, then a prose answer."""
    return [
        ModelResponse(tool_calls=[ToolCall(id="1", name="run_rsa", args={})]),
        ModelResponse(text=text),
    ]


class TestOffMode:
    def test_nothing_is_recorded_and_nothing_is_asked(self, loop_stub):
        """OFF is the baseline every earlier result was produced under. It has
        to stay byte-identical, including the absence of the log key."""
        provider = loop_stub(_with_tool_then("S2 10kV is at 1.072 p.u."))
        text, _ = loop_module.run_agent_turn("status?", [], reflection="off")

        assert text == "S2 10kV is at 1.072 p.u."
        assert len(provider.calls) == 2, "no extra model call"
        assert "reflection" not in loop_stub.written[0]


class TestWarnMode:
    def test_records_the_finding_without_changing_the_answer(self, loop_stub):
        provider = loop_stub(_with_tool_then("S2 10kV is at 1.072 p.u."))
        text, _ = loop_module.run_agent_turn("status?", [], reflection="warn")

        assert text == "S2 10kV is at 1.072 p.u.", "the draft is delivered untouched"
        assert len(provider.calls) == 2, "the agent is never re-asked"

        record = loop_stub.written[0]["reflection"]
        assert record["mode"] == "warn"
        assert record["triggered"] is True
        assert record["shown"] is False
        assert record["changed"] is False

    def test_clean_answer_records_a_quiet_check(self, loop_stub):
        loop_stub(_with_tool_then("S2 10kV is at 1.049 p.u."))
        loop_module.run_agent_turn("status?", [], reflection="warn")

        record = loop_stub.written[0]["reflection"]
        assert record["triggered"] is False
        assert record["n_findings"] == 0


class TestInformMode:
    def test_findings_go_back_and_the_agent_answers_again(self, loop_stub):
        provider = loop_stub([
            ModelResponse(tool_calls=[ToolCall(id="1", name="run_rsa", args={})]),
            ModelResponse(text="S2 10kV is at 1.072 p.u."),
            ModelResponse(text="Correction: S2 10kV is at 1.049 p.u."),
        ])
        text, _ = loop_module.run_agent_turn("status?", [], reflection="inform")

        assert text == "Correction: S2 10kV is at 1.049 p.u."
        assert len(provider.calls) == 3, "one extra call, and only one"

        # The findings reached the model as a message it could read.
        last_sent = provider.calls[-1]["messages"]
        assert any("grounding check" in str(m).lower() for m in last_sent)

    def test_the_agent_may_keep_its_answer(self, loop_stub):
        """Informing is not commanding. An unchanged answer is a legitimate
        outcome, and the record has to be able to say so."""
        loop_stub([
            ModelResponse(tool_calls=[ToolCall(id="1", name="run_rsa", args={})]),
            ModelResponse(text="S2 10kV is at 1.072 p.u."),
            ModelResponse(text="S2 10kV is at 1.072 p.u."),
        ])
        text, _ = loop_module.run_agent_turn("status?", [], reflection="inform")

        assert text == "S2 10kV is at 1.072 p.u."
        record = loop_stub.written[0]["reflection"]
        assert record["shown"] is True
        assert record["changed"] is False, "kept, not revised"

    def test_clean_draft_is_never_interrupted(self, loop_stub):
        provider = loop_stub(_with_tool_then("S2 10kV is at 1.049 p.u."))
        text, _ = loop_module.run_agent_turn("status?", [], reflection="inform")

        assert text == "S2 10kV is at 1.049 p.u."
        assert len(provider.calls) == 2, "a grounded answer costs nothing extra"

    def test_only_one_pass(self, loop_stub):
        """A second ungrounded answer is accepted. Re-checking the revision
        in-loop is how this oscillates, and the cost is unbounded."""
        provider = loop_stub([
            ModelResponse(tool_calls=[ToolCall(id="1", name="run_rsa", args={})]),
            ModelResponse(text="S2 10kV is at 1.072 p.u."),
            ModelResponse(text="Actually 1.081 p.u."),
        ])
        text, _ = loop_module.run_agent_turn("status?", [], reflection="inform")

        assert text == "Actually 1.081 p.u."
        assert len(provider.calls) == 3, "not re-checked, not re-asked"

    def test_the_agent_may_call_a_tool_instead(self, loop_stub):
        """The best outcome available: an ungrounded figure often means a tool
        was never called. Nothing in the design forces a prose edit."""
        provider = loop_stub([
            ModelResponse(tool_calls=[ToolCall(id="1", name="run_rsa", args={})]),
            ModelResponse(text="S2 10kV is at 1.072 p.u."),
            ModelResponse(tool_calls=[ToolCall(id="2", name="run_rsa", args={})]),
            ModelResponse(text="Confirmed: 1.049 p.u."),
        ])
        text, _ = loop_module.run_agent_turn("status?", [], reflection="inform")

        assert text == "Confirmed: 1.049 p.u."
        assert len(provider.calls) == 4


class TestRecordShape:
    def test_both_answers_are_kept(self, loop_stub):
        """Draft and final together. Without the pair there is no way to tell
        a revision from a rationalisation after the fact, which is the whole
        reason the mode is worth running."""
        loop_stub([
            ModelResponse(tool_calls=[ToolCall(id="1", name="run_rsa", args={})]),
            ModelResponse(text="S2 10kV is at 1.072 p.u."),
            ModelResponse(text="Corrected: 1.049 p.u."),
        ])
        loop_module.run_agent_turn("status?", [], reflection="inform")

        record = loop_stub.written[0]["reflection"]
        assert record["draft"] == "S2 10kV is at 1.072 p.u."
        assert record["final"] == "Corrected: 1.049 p.u."
        assert record["changed"] is True

    def test_findings_carry_their_context(self, loop_stub):
        loop_stub(_with_tool_then("S2 10kV is at 1.072 p.u."))
        loop_module.run_agent_turn("status?", [], reflection="warn")

        finding = loop_stub.written[0]["reflection"]["findings"][0]
        assert finding["value"] == pytest.approx(1.072)
        assert finding["status"] == "ungrounded"
        assert finding["nearest"] is not None


class TestDerivedFigures:
    """Arithmetic the model shows its working for.

    Split out after a live run flagged both a ratio and the division that
    produced it, so showing the work made the score worse.
    """

    def test_shown_division_is_derived_not_ungrounded(self):
        from llm_agent.agent.provenance import DERIVED, check_answer, collect_sources

        record = _record({"ext_grid": {"P_import_mw": 176.7793},
                          "totals": {"total_load_mw": 220.1206}})
        answer = ("The ratio of external grid import to total system load is "
                  "$\\frac{176.7793}{220.1206} \\approx 0.803101$. Expressed as "
                  "a percentage, this equals 80.31%.")
        report = check_answer(answer, collect_sources(record))
        statuses = {v.status for v in report.verdicts if v.claim.value in (0.803101, 80.31)}
        assert statuses == {DERIVED}

    def test_derived_is_not_surfaced_to_the_agent(self):
        record = _record({"ext_grid": {"P_import_mw": 176.7793},
                          "totals": {"total_load_mw": 220.1206}})
        answer = "$\\frac{176.7793}{220.1206} \\approx 0.803101$, so 80.31%."
        findings, _ = check_draft(answer, record)
        assert findings == (), "showing the arithmetic must not raise a finding"

    def test_derived_is_still_counted_offline(self):
        """The model did compute it. The grader says so even though reflection
        stays quiet — that is the whole point of splitting the two."""
        from evaluation.graders.provenance import grade_provenance

        record = _record({"ext_grid": {"P_import_mw": 176.7793},
                          "totals": {"total_load_mw": 220.1206}})
        record["assistant"] = "$\\frac{176.7793}{220.1206} \\approx 0.803101$."
        findings, report = grade_provenance(record)
        assert any(f.code == "derived_figure" for f in findings)
        assert report.derived_rate > 0

    def test_unshown_arithmetic_stays_ungrounded(self):
        """Only a division written in the prose counts. A bare percentage with
        no working is exactly the case the check exists for."""
        record = _record({"ext_grid": {"P_import_mw": 176.7793},
                          "totals": {"total_load_mw": 220.1206}})
        findings, _ = check_draft("The external grid covers 80.31% of load.", record)
        assert len(findings) == 1
        assert findings[0].status == "ungrounded"

    def test_division_of_numbers_not_in_evidence_is_not_excused(self):
        record = _record({"ext_grid": {"P_import_mw": 176.7793},
                          "totals": {"total_load_mw": 220.1206}})
        findings, _ = check_draft("$\\frac{500.0}{1000.0} = 0.5$, so 50%.", record)
        assert findings, "invented operands cannot launder an invented result"

    def test_derived_does_not_inflate_the_grounding_rate(self):
        from llm_agent.agent.provenance import check_answer, collect_sources

        record = _record({"ext_grid": {"P_import_mw": 176.7793},
                          "totals": {"total_load_mw": 220.1206}})
        report = check_answer("$\\frac{176.7793}{220.1206} \\approx 0.803101$.",
                              collect_sources(record))
        assert report.derived_rate > 0
        assert report.rate < 1.0, "computed figures must not count as traceable"


class TestNearestPlausibility:
    def test_unrelated_nearest_is_not_shown_to_the_agent(self):
        """0.94 was offered as the nearest source for 0.803101 — the network's
        lower voltage limit, and not a suggestion anyone should act on.

        The verdict keeps it either way: the distance is diagnostic, and the
        offline grader wants to see it. Only the message withholds it."""
        record = _record({"limits": {"vm_lower": 0.94}})
        findings, _ = check_draft("The share is 0.803101.", record)
        assert len(findings) == 1
        assert findings[0].nearest is not None, "the grader still sees it"
        assert "closest value" not in format_findings(findings)

    def test_close_nearest_is_shown(self):
        record = _record({"bus": "S2 10kV", "vm_pu": 1.049})
        findings, _ = check_draft("S2 10kV is at 1.072 p.u.", record)
        assert findings[0].nearest[0] == pytest.approx(1.049)
        assert "1.049" in format_findings(findings)
