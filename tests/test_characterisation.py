"""Characterisation — is a figure described the way its own record describes it.

The case these are built around is real. The engine resolved a movement,
marked it undeliverable, and the answer recommended it anyway:

    relief_mw: -6.2415, relief_mw_feasible: false
    "Reduce 05 ÅKI Sgen by 6.24 MW if acting alone."

Provenance had nothing to say — 6.24 was in the payload, correctly attributed.
It was caught by the sign, by luck. These pin the check that does not need the
luck, and — more importantly — the cases where saying nothing is correct.
"""

import pytest

from evaluation.graders.characterisation import grade_characterisation
from llm_agent.agent.characterisation import check_answer, check_turn


def _attribution(feasible=False, relief=-6.2415):
    """A driver record shaped like the attribution engine's output."""
    return {
        "user": "why is Akirkeby violating?",
        "grid": {},
        "tool_calls": [{"name": "compute_violation_attribution", "args": {}}],
        "tool_results": [{
            "name": "compute_violation_attribution",
            "result": {
                "converged": True,
                "violations": [{
                    "element": "Akirkeby 10.5 kV",
                    "value": 1.0512,
                    "limit": 1.05,
                    "violated": True,
                    "drivers": [{
                        "source": "05 AKI Sgen",
                        "controllable": True,
                        "current_p_mw": 3.54,
                        "relief_mw": relief,
                        "relief_mw_feasible": feasible,
                    }],
                }],
            },
        }],
    }


class TestTheLiveCase:
    def test_undeliverable_movement_quoted_as_an_action(self):
        record = _attribution()
        conflicts = check_answer("Reduce 05 AKI Sgen by 6.24 MW if acting alone.", record)
        assert len(conflicts) == 1
        assert conflicts[0].field == "relief_mw_feasible"

    def test_sign_does_not_matter(self):
        """The original was caught because the claim was positive and the source
        negative. A check that depended on that would miss the same sentence
        written with the minus."""
        record = _attribution()
        conflicts = check_answer("Reduce 05 AKI Sgen by -6.2415 MW.", record)
        assert conflicts, "the correctly-signed version is the same error"

    def test_a_deliverable_movement_is_not_flagged(self):
        record = _attribution(feasible=True, relief=-0.42)
        assert check_answer("Reduce 05 AKI Sgen by 0.42 MW.", record) == []


class TestNegation:
    """The false-positive cases. Every grader needs these, and this one needs
    them most: it reads words, and the sentences that get the physics right are
    the ones most likely to quote an infeasible figure."""

    def test_stating_the_movement_is_impossible_is_correct(self):
        record = _attribution()
        conflicts = check_answer(
            "No single source can clear it: the smallest sufficient movement "
            "(05 AKI Sgen, 6.24 MW) exceeds its available 3.54 MW.",
            record,
        )
        assert conflicts == []

    @pytest.mark.parametrize("sentence", [
        "05 AKI Sgen cannot be reduced by 6.24 MW.",
        "It is not possible to reduce 05 AKI Sgen by 6.24 MW.",
        "Reducing by 6.24 MW would be insufficient given its 3.54 MW output.",
        "The unit is unable to absorb the 6.24 MW required.",
    ])
    def test_negated_forms_pass(self, sentence):
        assert check_answer(sentence, _attribution()) == []

    def test_a_turn_with_no_qualifiers_is_never_flagged(self):
        record = {
            "user": "status?",
            "tool_results": [{"name": "run_rsa", "result": {"vm_pu": 1.049}}],
        }
        assert check_answer("The bus is at 1.049 p.u.", record) == []

    def test_the_qualifier_field_itself_is_not_a_claim(self):
        """`relief_mw_feasible: false` is the qualifier, not a figure the
        answer can mischaracterise."""
        record = _attribution()
        assert check_answer("The feasibility flag is false.", record) == []


class TestOtherQualifiers:
    def test_a_violating_element_called_secure(self):
        record = _attribution()
        conflicts = check_answer("Akirkeby 10.5 kV is secure at 1.0512 p.u.", record)
        assert conflicts
        assert any(c.field == "violated" for c in conflicts)

    def test_a_violating_element_correctly_reported(self):
        record = _attribution()
        assert check_answer(
            "Akirkeby 10.5 kV is at 1.0512 p.u., above its 1.05 limit.", record
        ) == []

    def test_a_failed_solve_called_solved(self):
        record = {
            "user": "run it",
            "tool_results": [{"name": "run_rsa", "result": {
                "converged": False, "max_vm_pu": 1.21,
            }}],
        }
        conflicts = check_answer("The system solved with a maximum of 1.21 p.u.", record)
        assert any(c.field == "converged" for c in conflicts)


class TestScope:
    def test_only_the_object_a_number_sits_in_is_consulted(self):
        """A qualifier must not reach across into a sibling record. Walking up
        the tree to find one is how a check starts flagging correct sentences."""
        record = {
            "user": "compare",
            "tool_results": [{"name": "t", "result": {"items": [
                {"source": "A", "relief_mw": -6.24, "relief_mw_feasible": False},
                {"source": "B", "relief_mw": -0.42, "relief_mw_feasible": True},
            ]}}],
        }
        assert check_answer("Reduce B by 0.42 MW.", record) == []
        assert check_answer("Reduce A by 6.24 MW.", record)


class TestGrader:
    def test_a_conflict_is_a_failure(self):
        record = _attribution()
        record["assistant"] = "Reduce 05 AKI Sgen by 6.24 MW if acting alone."
        record["turn"] = 3
        findings, conflicts = grade_characterisation(record)
        assert len(findings) == 1
        assert findings[0].severity == "fail"
        assert findings[0].code == "mischaracterised_figure"
        assert findings[0].turn == 3
        assert len(conflicts) == 1

    def test_a_clean_turn_produces_nothing(self):
        record = _attribution(feasible=True, relief=-0.42)
        record["assistant"] = "Reduce 05 AKI Sgen by 0.42 MW."
        findings, conflicts = grade_characterisation(record)
        assert findings == []
        assert conflicts == []

    def test_check_turn_reads_the_assistant_field(self):
        record = _attribution()
        record["assistant"] = "Reduce 05 AKI Sgen by 6.24 MW."
        assert check_turn(record)
