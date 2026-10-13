"""
The what-if evaluation: its checks, its grader, and its runner — all without
spending model credit (a scripted model stands in for the real one).
"""

import json

import pytest

from evaluation import whatif
from evaluation.graders.sandbox import grade_sandbox
from llm_agent.agent.validators import PROPOSAL_NOTE_PREFIX

NET = {"id": "bus9_datacenter", "edits": ["New bus ...", "Add load ..."]}

# The first live session (Oct 11), reduced to what the checks read.
LIVE = {
    "turn": 1, "status": "completed",
    "user": "What if we build a new substation 5 km from Bus 9 ... and put a 15 MW data centre there? "
            "Compare with today.",
    "tool_calls": [{"name": "locate_network_element", "args": {"element": "Bus 8"}},
                   {"name": "edit_proposal", "args": {"proposal": "bus9_datacenter"}},
                   {"name": "compare_networks", "args": {"proposal": "bus9_datacenter", "study": "rsa",
                                                         "timestamp": "2000-01-01 00:00:00"}}],
    "tool_results": [
        {"name": "locate_network_element", "result": {"error": "No bus named 'Bus 8'."}},
        {"name": "edit_proposal", "result": {"status": "created", "edits": [
            {"op": "add_bus_with_connection", "name": "New Substation near Bus 9", "connect_to": 9,
             "vn_kv": 1.0, "length_km": 5.0, "line_like": 12},
            {"op": "add_load", "name": "Data centre", "bus": 14, "p_mw": 15.0, "pf": 0.95, "profile": "flat"}]}},
        {"name": "compare_networks", "result": {"study": "rsa", "timestamp": "2000-01-01 00:00:00",
                                                "network": {**NET, "compared_with": "base"},
                                                "new_violations": [{"type": "bus_vm_pu", "element": "x"}]}}],
    "assistant": "At 00:00 the proposal is not secure ...\n\n" + PROPOSAL_NOTE_PREFIX + "**bus9_datacenter** ...",
}


def _task(task_id):
    return next(t for t in whatif.TASKS if t.id == task_id)


def test_the_live_session_graded_against_the_new_site_task():
    got = whatif.evaluate(_task("new_site_n1"), [LIVE])
    assert got == {
        "edit: new bus 5 km from Bus_9 like L12": True,
        "edit: 15 MW load at the new bus": True,
        "N-1 on the proposal": False,                  # it was not asked in that session
        "reported the line-length note": False,        # the note did not exist yet
        "named the proposal": False,                   # the guard had to add it
        "no refused edit": True,
    }


def test_each_check_kind():
    over_time = {"name": "compare_networks", "args": {"study": "scan"}}
    rec = {"tool_calls": [over_time, {"name": "simulate_all_contingencies", "args": {"network": "p"}},
                          {"name": "edit_proposal", "args": {}}],
           "tool_results": [{"result": {"study": "scan", "network": {"id": "p"}}},
                            {"result": {"network": {"id": "p"}, "violations": []}},
                            {"result": {"error": "bad", "detail": "Edit 1: There is no bus 99."}}],
           "assistant": "On the proposal, lines are placeholders of 1 km."}
    assert whatif.check({"kind": "over_time"}, [rec])
    assert whatif.check({"kind": "n1_on_proposal"}, [rec])
    assert whatif.check({"kind": "named"}, [rec])
    assert whatif.check({"kind": "mentions", "any": ["placeholder"]}, [rec])
    assert not whatif.check({"kind": "valid_edits"}, [rec])
    assert whatif.check({"kind": "tool", "name": "compare_networks", "args": {"study": "sc"}}, [rec])
    with pytest.raises(ValueError):
        whatif.check({"kind": "nope"}, [rec])


def test_numbers_match_within_one_percent_and_text_without_case():
    edits = [{"op": "add_load", "bus": 9, "p_mw": 50.2, "profile": "Flat"}]
    rec = {"tool_calls": [{"name": "edit_proposal"}], "tool_results": [{"result": {"edits": edits}}]}
    assert whatif.check({"kind": "edit", "match": {"op": "add_load", "p_mw": 50, "profile": "flat"}}, [rec])
    assert not whatif.check({"kind": "edit", "match": {"op": "add_load", "p_mw": 55}}, [rec])
    assert not whatif.check({"kind": "edit", "match": {"op": "add_load", "bus": 4}}, [rec])


def test_the_sandbox_grader_on_the_live_session():
    codes = [f.code for f in grade_sandbox(LIVE)]
    assert codes == ["proposal_single_moment", "proposal_without_n1"]


def test_tasks_are_well_formed():
    ids = [t.id for t in whatif.TASKS]
    assert len(ids) == len(set(ids)) >= 8
    for t in whatif.TASKS:
        assert t.turns and all(isinstance(q, str) and q for q in t.turns)
        labels = [c["label"] for c in (*t.checks, *whatif._ALWAYS)]
        assert len(labels) == len(set(labels))
        for c in t.checks:
            whatif.check(c, [])                      # every kind is known


def test_the_runner_end_to_end_with_a_scripted_model(backend_client, monkeypatch, tmp_path):
    """No model credit: the agent's replies are scripted, the backend is real."""
    import main_backend as mb
    from llm_agent.agent import loop, tools
    from llm_agent.agent.providers.base import ModelResponse, ToolCall

    def post(endpoint, body):
        r = backend_client.post(endpoint, json={k: v for k, v in body.items() if v is not None})
        return r.json() if r.status_code < 400 else {"error": r.text}

    def backend(method, path):
        return getattr(backend_client, method)(path).json()

    def scripted():
        return iter([
            [ToolCall("edit_proposal", {"proposal": "dc", "edits": [
                {"op": "add_load", "bus": "Bus 9", "p_mw": 50, "name": "Data centre"}]})],
            [ToolCall("compare_networks", {"proposal": "dc", "study": "scan",
                                           "start_timestamp": "2000-01-01", "end_timestamp": "2000-01-01",
                                           "step_size": 8})],
            "On proposal dc the data centre lowers Bus_9 at its worst moment.",
        ])

    script = scripted()

    class Model:
        def chat(self, messages, tools, system, on_event=None):
            step = next(script)
            return ModelResponse(text=step) if isinstance(step, str) else ModelResponse(text="", tool_calls=step)

    monkeypatch.setattr(tools, "_post", post)
    monkeypatch.setattr(whatif, "_backend", backend)
    monkeypatch.setattr(loop, "get_provider", lambda: Model())
    monkeypatch.setattr(loop, "describe_provider", lambda p: {"provider": "scripted"})
    monkeypatch.setattr(loop, "get_system_prompt", lambda: "test")
    written = []
    monkeypatch.setattr(loop, "_write_record", written.append)   # nothing reaches the session logs

    out = whatif.run([_task("dc_flat")], repeats=1, out_dir=tmp_path)
    lines = [json.loads(x) for x in open(out / "dc_flat-01.jsonl")]
    assert lines[0] == {"kind": "whatif", "task": "dc_flat", "repeat": 1, "follow_suggestion": False}
    summary = whatif.grade(out)["dc_flat"]
    assert summary["runs"] == 1
    assert summary["checks"] == {"edit: 50 MW load at Bus_9": 1, "looked beyond one moment": 1,
                                 "named the proposal": 1, "no refused edit": 1}
    assert written == [] and loop._append_to_log is not None
    assert "| dc_flat | 1 | edit: 50 MW load at Bus_9 | 1/1 |" in whatif.render(whatif.grade(out))
    mb._discard_proposals()


def test_a_question_to_the_operator_is_asking_not_a_refusal():
    question = {"error": "400", "detail": {"needs_operator": True, "question": "How should 'G' behave?",
                                           "options": ["voltage", "power_factor"]}}
    asked = {"tool_calls": [{"name": "edit_proposal", "args": {}}], "tool_results": [{"result": question}],
             "assistant": "How should the new 30 MW unit behave: (1) voltage control or (2) a fixed power factor?"}
    built = {"tool_calls": [{"name": "edit_proposal", "args": {}}],
             "tool_results": [{"result": {"edits": [{"op": "add_generator", "bus": 4, "p_max_mw": 30,
                                                     "control": "voltage", "vm_pu": 1.0}]}}],
             "assistant": "On proposal g the unit holds 1.0 p.u."}
    assert whatif.check({"kind": "valid_edits"}, [asked, built])
    assert whatif.check({"kind": "asked"}, [asked, built])
    assert not whatif.check({"kind": "asked"}, [built])
    got = whatif.evaluate(_task("classical_generator"), [asked, built])
    assert got["asked how the unit is modelled before building it"]
    assert got["edit: 30 MW unit on voltage control at Bus_4"]
