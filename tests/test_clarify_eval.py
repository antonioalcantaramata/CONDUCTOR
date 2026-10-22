"""
The clarification evaluation (`evaluation/clarify.py`) and the rule it holds
the agent to: ask before running only when the request leaves open something
that changes the answer — tested both ways, and graded from the records.
"""

import pytest

from evaluation import clarify, whatif


def _rec(calls, answer="Done.", user="q"):
    return {"user": user, "assistant": answer,
            "tool_calls": [{"name": n, "args": a} for n, a, _ in calls],
            "tool_results": [{"name": n, "result": r} for n, _, r in calls]}


QUESTION = ("Three lines run into Bus_23. Which one?\n\n1. Bus_22 -> Bus_23 [L31] (suggested)\n"
            "2. Bus_21 -> Bus_23 [L30]\n3. Bus_23 -> Bus_24 [L32]")
LOOKUP = ("locate_network_element", {"element": "Bus_23"}, {"name": "Bus_23", "neighbours": []})


def test_the_tasks_test_both_ways():
    ids = [t.id for t in clarify.TASKS]
    assert len(ids) == len(set(ids)) >= 5
    asking = [t for t in clarify.TASKS if any(c["kind"] == "asked_first" for c in t.checks)]
    running = [t for t in clarify.TASKS if t not in asking]
    assert len(asking) >= 2 and len(running) >= 3                     # over-asking fails too
    for t in asking:                                                     # the answer to the question follows
        assert len(t.turns) == 2
    for t in clarify.TASKS:
        assert len({c["label"] for c in t.checks}) == len(t.checks)
        for c in t.checks:
            whatif.check(c, [])                                          # every kind is known
    assert clarify.CLARIFY.needs == "30" and clarify.CLARIFY.always == ()


def test_the_prompts_examples_are_not_the_tasks():
    """The rule's examples stay generic: the evaluation is not taught its own questions."""
    from llm_agent.agent.system_prompt import build_system_prompt
    from llm_agent.agent.config import DEFAULT_GRID_CONSTANTS
    prompt = build_system_prompt(DEFAULT_GRID_CONSTANTS)
    assert "Ask first only when it changes the answer" in prompt and 'n1="both"' in prompt
    assert "Bus_23" not in prompt.split("Ask first only when it changes the answer")[1]
    assert "comfortably" not in prompt


class TestChecks:
    def test_a_question_after_a_lookup_is_asking_first(self):
        records = [_rec([LOOKUP], QUESTION)]
        assert whatif.check({"kind": "asked_first", "label": "x"}, records)
        assert whatif.check({"kind": "choices", "label": "x"}, records)

    def test_running_a_study_is_not_asking_even_with_a_question_mark(self):
        ran = _rec([LOOKUP, ("simulate_contingency", {"element_type": "line", "element_index": 30}, {"ok": True})],
                   "Losing L30 overloads L31. Want me to check another line?")
        assert not whatif.check({"kind": "asked_first", "label": "x"}, [ran])

    def test_a_change_is_acting_too(self):
        changed = _rec([("set_load_power_factor", {"value": 0.9}, {"ok": True})], "Done. Anything else?")
        assert not whatif.check({"kind": "asked_first", "label": "x"}, [changed])

    @pytest.mark.parametrize("answer, numbered", [
        (QUESTION, True),
        ("Which one?\n- **1.** L31\n- **2.** L30", True),
        ("Which one? (1) L31 or (2) L30", False),                         # inline, not a list
        ("Which line do you mean, the one from Bus_22?", False),
    ])
    def test_numbered_choices(self, answer, numbered):
        assert whatif.check({"kind": "choices", "label": "x"}, [_rec([], answer)]) is numbered

    def test_the_turn_after_the_answer_runs_the_chosen_study(self):
        spec = {"kind": "ran", "label": "x", "turn": 1, "tool": "simulate_contingency",
                "args": {"element_type": "line", "element_index": 31}}
        asked = _rec([LOOKUP], QUESTION)
        right = _rec([("simulate_contingency", {"element_type": "line", "element_index": 31}, {"ok": True})])
        wrong = _rec([("simulate_contingency", {"element_type": "line", "element_index": 30}, {"ok": True})])
        failed = _rec([("simulate_contingency", {"element_type": "line", "element_index": 31}, {"error": "x"})])
        assert whatif.check(spec, [asked, right])
        assert not whatif.check(spec, [asked, wrong]) and not whatif.check(spec, [asked, failed])
        assert not whatif.check(spec, [asked])                           # no second turn: not run

    def test_running_without_asking(self):
        spec = {"kind": "ran", "label": "x", "tool": "run_rsa"}
        assert whatif.check(spec, [_rec([("get_current_timestamp", {}, {"timestamp": "t"}),
                                         ("run_rsa", {}, {"secure": True})], "Secure.")])
        assert not whatif.check(spec, [_rec([("get_current_timestamp", {}, {"timestamp": "t"})],
                                            "Which moment do you mean?")])
        assert not whatif.check({"kind": "ran", "label": "x"}, [_rec([LOOKUP], QUESTION)])   # a lookup is not a study

    def test_a_task_graded_end_to_end(self, tmp_path):
        import json
        task = next(t for t in clarify.TASKS if t.id == "which_line")
        records = [_rec([LOOKUP], QUESTION),
                   _rec([("simulate_contingency", {"element_type": "line", "element_index": 31}, {"ok": True})])]
        with open(tmp_path / "which_line-01.jsonl", "w") as fh:
            fh.write(json.dumps({"kind": "clarify", "task": "which_line", "repeat": 1}) + "\n")
            for r in records:
                fh.write(json.dumps(r) + "\n")
        summary = whatif.grade(tmp_path, clarify.CLARIFY)
        assert summary["which_line"]["runs"] == 1
        assert all(passes == 1 for passes in summary["which_line"]["checks"].values()), summary
        assert set(summary["which_line"]["checks"]) == {c["label"] for c in task.checks}
