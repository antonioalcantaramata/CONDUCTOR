"""
The flexibility-options evaluation (`evaluation/options.py`): its tasks, its
check kinds, and the runner end to end with a scripted model — no credit spent.
"""

import json
import shutil

import pytest

from evaluation import options, whatif


def _rec(calls, answer="Four options.", user="q"):
    return {"user": user, "assistant": answer,
            "tool_calls": [{"name": n, "args": a} for n, a, _ in calls],
            "tool_results": [{"name": n, "result": r} for n, _, r in calls]}


OPTIONS_RESULT = {"mode": "options", "status": "options", "options": [
    {"id": "cheapest", "units_called": ["Gen_bus1"]},
    {"id": "near_optimal_1", "avoids": ["Gen_bus10"], "adds": ["Gen_bus4"]}]}


def test_the_tasks_are_well_formed():
    ids = [t.id for t in options.TASKS]
    assert len(ids) == len(set(ids)) >= 5
    for t in options.TASKS:
        labels = [c["label"] for c in (*t.checks, *options.OPTIONS.always)]
        assert len(labels) == len(set(labels))
        for c in t.checks:
            whatif.check(c, [])                      # every kind is known
    assert options.OPTIONS.needs == "30" and options.OPTIONS.name == "options"


class TestChecks:
    def test_options_with_a_unit_held_in_a_list(self):
        rec = _rec([("optimize_flexibility", {"options": True, "hold_units": ["gen_bus1"]}, OPTIONS_RESULT)])
        spec = {"kind": "options", "label": "x", "args": {"options": True, "hold_units": "Gen_bus1"}}
        assert whatif.check(spec, [rec])
        assert not whatif.check({**spec, "args": {"options": True, "hold_units": "Gen_bus4"}}, [rec])
        failed = _rec([("optimize_flexibility", {"options": True}, {"error": "refused"})])
        assert not whatif.check({"kind": "options", "label": "x", "args": {}}, [failed])

    def test_a_margin_is_a_number(self):
        rec = _rec([("optimize_flexibility", {"options": True, "near_optimal": True, "near_optimal_pct": 25.0},
                     OPTIONS_RESULT)])
        assert whatif.check({"kind": "options", "label": "x", "args": {"near_optimal_pct": 25}}, [rec])

    def test_taken_out_of_service_is_caught(self):
        spec = {"kind": "not_tool_arg", "label": "x", "tool": "optimize_flexibility",
                "arg": "disabled_generators", "value": "Gen_bus1"}
        out = _rec([("optimize_flexibility", {"disabled_generators": ["Gen_bus1"]}, OPTIONS_RESULT)])
        assert not whatif.check(spec, [out]) and whatif.check(spec, [_rec([])])

    @pytest.mark.parametrize("answer, picks", [
        ("I recommend the cheapest option.", True),
        ("I would go with T1·cheapest.", True),
        ("The best option is least import.", True),
        ("The cheapest option costs 26.48 €/h; least import imports least. The choice is yours.", False),
        ("Which matters more to you: cost or the number of instructions?", False),
    ])
    def test_picking_an_option_is_caught_and_describing_is_not(self, answer, picks):
        assert whatif.check({"kind": "no_winner", "label": "x"}, [_rec([], answer)]) is not picks

    def test_the_answer_names_what_an_alternative_avoids_or_adds(self):
        spec = {"kind": "quotes_alternatives", "label": "x"}
        calls = [("optimize_flexibility", {"options": True, "near_optimal": True}, OPTIONS_RESULT)]
        assert whatif.check(spec, [_rec(calls, "Near-optimal 1 avoids Gen_bus10.")])
        assert not whatif.check(spec, [_rec(calls, "There are alternatives.")])

    def test_any_of_two_ways(self):
        spec = {"kind": "any", "label": "x", "of": [
            {"kind": "result", "label": "a", "tool": "optimize_flexibility", "fields": {"status": "no_action_needed"}},
            {"kind": "result", "label": "b", "tool": "run_rsa", "fields": {"secure": True}}]}
        assert whatif.check(spec, [_rec([("run_rsa", {}, {"secure": True})])])
        assert not whatif.check(spec, [_rec([("run_rsa", {}, {"secure": False})])])

    def test_a_result_field(self):
        rec = _rec([("optimize_flexibility", {"options": True}, {"status": "no_action_needed", "mode": "options"})])
        assert whatif.check({"kind": "result", "label": "x", "tool": "optimize_flexibility",
                             "fields": {"status": "no_action_needed"}}, [rec])


@pytest.mark.skipif(shutil.which("ipopt") is None, reason="IPOPT not installed")
def test_the_runner_end_to_end_with_a_scripted_model(tmp_path, monkeypatch):
    """A real 30-bus backend; the model's replies are scripted."""
    pytest.importorskip("fastapi.testclient")
    import main_backend as mb
    from fastapi.testclient import TestClient
    from llm_agent.agent import loop, tools
    from llm_agent.agent.providers.base import ModelResponse, ToolCall

    monkeypatch.delenv("GRID_PROFILE", raising=False)
    monkeypatch.setattr(mb, "SYSTEMS_DIR", str(tmp_path / "systems"))
    monkeypatch.setattr(mb, "DATA_FILES_DIR", str(tmp_path / "data_files"))
    for name in ("_LAST_UPLOAD_SENTINEL", "_LAST_TIMESERIES_SENTINEL", "_LAST_FORECAST_SENTINEL",
                 "_PROPOSALS_FILE", "_ASSUMPTIONS_FILE", "_SCENARIO_FILE"):
        monkeypatch.setattr(mb, name, str(tmp_path / f"{name}.json"))
    saved = dict(mb.app_data)
    with TestClient(mb.app) as client:
        monkeypatch.setattr(tools, "_post", lambda e, b: client.post(e, json={k: v for k, v in b.items()
                                                                             if v is not None}).json())
        monkeypatch.setattr(whatif, "_backend", lambda method, path: getattr(client, method)(path).json())
        script = iter([
            [ToolCall("optimize_flexibility", {"options": True, "timestamp": "2000-01-06 20:00"})],
            "Four options; the choice is yours.",
            [ToolCall("optimize_flexibility", {"options": True, "timestamp": "2000-01-06 20:00",
                                               "hold_units": ["Gen_bus1"]})],
            "With Gen_bus1 held, these remain. Which matters more to you?",
        ])

        class Model:
            def chat(self, messages, tools, system, on_event=None):
                step = next(script)
                return ModelResponse(text=step) if isinstance(step, str) else ModelResponse(text="", tool_calls=step)

        monkeypatch.setattr(loop, "get_provider", lambda: Model())
        monkeypatch.setattr(loop, "describe_provider", lambda p: {"provider": "scripted"})
        monkeypatch.setattr(loop, "get_system_prompt", lambda: "test")
        monkeypatch.setattr(loop, "_write_record", lambda r: None)
        task = next(t for t in options.TASKS if t.id == "rule_out_unit")
        out = whatif.run([task], repeats=1, out_dir=tmp_path / "runs", suite_kind="options")
    mb.app_data.clear()
    mb.app_data.update(saved)
    first = json.loads(open(out / "rule_out_unit-01.jsonl").readline())
    assert first["kind"] == "options"
    summary = whatif.grade(out, options.OPTIONS)["rule_out_unit"]
    assert summary["checks"] == {"re-solved with Gen_bus1 held": 1, "did not take Gen_bus1 out of service": 1,
                                 "left the choice to the operator": 1}
