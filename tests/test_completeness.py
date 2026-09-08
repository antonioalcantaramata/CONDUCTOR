"""Completeness — did the answer contain the kind of thing that was asked for.

Built around the live turn that motivated it. Asked what percentage of load the
external grid covers, the agent drafted "approximately 80.3%", was shown a
grounding finding, and delivered:

    External grid import: 176.7793 MW
    Total system load:    220.1206 MW

Two true, grounded, correctly attributed figures and no percentage. Every other
check passed.

The negative cases carry more weight here than in the other guards. A missing
figure is often the *right* answer, and a check that argued an agent out of
declining would recreate the failure this layer exists to prevent.
"""

import pytest

from evaluation.graders.completeness import grade_completeness
from llm_agent.agent.completeness import check_answer, format_gaps


class TestTheLiveCase:
    QUESTION = "then what percentage is covered by the external grid?"

    def test_the_withdrawn_answer_is_caught(self):
        answer = (
            "At the current timestamp, the external grid import is 176.7793 MW "
            "and the total system load is 220.1206 MW.\n\n"
            "To answer your question using strictly the raw values provided:\n"
            "- External grid import: 176.7793 MW\n"
            "- Total system load: 220.1206 MW"
        )
        gaps = check_answer(self.QUESTION, answer)
        assert len(gaps) == 1
        assert gaps[0].kind == "percentage"

    def test_the_draft_that_answered_is_not_caught(self):
        answer = (
            "Dividing the external grid import by the total system load "
            "(176.7793 / 220.1206), the external grid is covering approximately "
            "80.3% of the total system load."
        )
        assert check_answer(self.QUESTION, answer) == []


class TestKinds:
    @pytest.mark.parametrize("question,answer", [
        ("what percentage of load is generator 1 covering?", "Generator 1 covers 19.7% of load."),
        ("how much MW can S2 host?", "S2 can host 1.71 MW at unity power factor."),
        ("what is the voltage at Bus_3?", "Bus_3 is at 0.961 p.u."),
        ("when is the worst timestamp?", "The worst point is 2022-01-02 21:45."),
        ("how many violations are there?", "There are 2 violations."),
    ])
    def test_an_answered_question_is_complete(self, question, answer):
        assert check_answer(question, answer) == []

    @pytest.mark.parametrize("question,answer,kind", [
        ("what percentage of load is generator 1 covering?",
         "Generator 1 is producing 43.34 MW.", "percentage"),
        ("what is the voltage at Bus_3?",
         "Bus_3 is the most heavily loaded bus.", "voltage"),
        ("when is the worst timestamp?",
         "The worst point is during a low-load, high-injection period.", "timestamp"),
        ("how many violations are there?",
         "Several buses are outside their limits.", "count"),
    ])
    def test_an_unanswered_question_is_flagged(self, question, answer, kind):
        gaps = check_answer(question, answer)
        assert [g.kind for g in gaps] == [kind]


class TestDecliningIsComplete:
    """The cases where a missing figure is the best available answer.

    These matter more than the positives. Pushing an agent to produce a number
    the study cannot support is exactly the failure everything else here is
    built to prevent.
    """

    @pytest.mark.parametrize("answer", [
        "The hosting-capacity bisection did not converge, so a capacity cannot be determined.",
        "That analysis is not supported in this combination.",
        "No measurement data is available for that window.",
        "The OPF was infeasible, so no dispatch figure can be provided.",
        "I am unable to compute that without a forecast dataset.",
    ])
    def test_an_explicit_decline_is_not_a_gap(self, answer):
        assert check_answer("how much MW of hosting capacity is there?", answer) == []

    def test_a_question_asking_for_nothing_quantitative(self):
        assert check_answer("why is the voltage rising?",
                            "Local generation exceeds demand, pushing voltages up.") == []

    def test_an_unrelated_question_is_never_graded(self):
        assert check_answer("give me a summary of the current conditions",
                            "The grid is secure.") == []


class TestWording:
    """The register, pinned. This check must never assert the answer is wrong."""

    def test_the_message_says_the_absence_may_be_correct(self):
        gaps = tuple(check_answer("what percentage?", "It is 43.34 MW."))
        message = format_gaps(gaps).lower()
        assert "may well be correct" in message
        assert "keep it" in message

    def test_the_message_does_not_demand_a_number(self):
        gaps = tuple(check_answer("what percentage?", "It is 43.34 MW."))
        message = format_gaps(gaps).lower()
        for directive in ("you must", "always provide", "never omit"):
            assert directive not in message

    def test_it_asks_for_the_reason_when_unavailable(self):
        """Silently leaving a figure out and explaining why it is unavailable
        are different answers, and only one is useful to an operator."""
        gaps = tuple(check_answer("what percentage?", "It is 43.34 MW."))
        assert "say why it is unavailable" in format_gaps(gaps).lower()


class TestGrader:
    def test_a_gap_is_a_warning_not_a_failure(self):
        record = {
            "turn": 3,
            "user": "then what percentage is covered by the external grid?",
            "assistant": "External grid import: 176.7793 MW. Total system load: 220.1206 MW.",
        }
        findings, gaps = grade_completeness(record)
        assert len(findings) == 1
        assert findings[0].severity == "warn", "a missing figure is often the right answer"
        assert findings[0].code == "answer_incomplete"
        assert findings[0].turn == 3
        assert len(gaps) == 1

    def test_a_complete_answer_produces_nothing(self):
        record = {
            "turn": 1,
            "user": "what percentage is covered by the external grid?",
            "assistant": "The external grid covers 80.31% of total load.",
        }
        findings, gaps = grade_completeness(record)
        assert findings == []
        assert gaps == []
