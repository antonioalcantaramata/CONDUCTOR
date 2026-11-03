"""The ablation runner (evaluation/ablation.py, arms.py, suites.py, ablation_stats.py) and the agent's
evaluation switches (hidden tools, dropped prompt parts, output tokens) — without model credit."""

import json
import re

import pytest

from evaluation import ablation, ablation_stats as st, arms, suites, whatif
from llm_agent.agent import loop as loop_module, system_prompt as sp, tool_schemas, usage
from llm_agent.agent.config import DEFAULT_GRID_CONSTANTS, GridConstants
from llm_agent.agent.providers.base import ModelResponse, ToolCall
from llm_agent.agent.tool_registry import ABLATION_GROUPS, REGISTRY, expand_hidden

PROMPT = sp.build_system_prompt(DEFAULT_GRID_CONSTANTS)


# ---------------------------------------------------------------------------
# The switches in the agent
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("part", sorted(sp.PROMPT_PARTS))
def test_every_prompt_part_is_found_and_removes_only_itself(part):
    lean, removed = sp.drop_parts(PROMPT, [part])
    assert removed[part] > 0 and len(lean) < len(PROMPT)
    # What is left still has its headings and every rule not removed.
    assert "## Behavioral rules" in lean
    kept = re.findall(r"(?m)^(\d+)\. \*\*", lean)
    assert kept == sorted(kept, key=int)


def test_a_dropped_rule_takes_its_paragraphs_and_leaves_the_next_rule():
    lean, _ = sp.drop_parts(PROMPT, ["ask_first"])
    assert "Ask first only when" not in lean and "33. **Word each claim by what backs it" in lean
    assert "31. **Confirm the data source matches" in lean


def test_routing_drops_the_question_to_tool_rules_only():
    lean, _ = sp.drop_parts(PROMPT, ["routing"])
    kept = {int(n) for n in re.findall(r"(?m)^(\d+)\. \*\*", lean)}
    assert kept == {1, 2, 3, 4, 5, 14, 15, 27, 28, 29, 30, 31, 32, 33}


def test_an_unknown_or_missing_part_raises():
    with pytest.raises(ValueError, match="Unknown prompt part"):
        sp.drop_parts(PROMPT, ["nope"])
    with pytest.raises(ValueError, match="found 0 times"):
        sp.drop_parts("no rules here", ["ask_first"])


def test_the_live_prompt_follows_the_environment(monkeypatch):
    monkeypatch.setattr(sp, "fetch_grid_constants", lambda: GridConstants(DEFAULT_GRID_CONSTANTS, True))
    monkeypatch.delenv("CONDUCTOR_DROP_RULES", raising=False)
    monkeypatch.setenv("CONDUCTOR_ONTOLOGY", "on")
    full = sp.get_system_prompt()
    assert "Ask first only when" in full and "What an element is" in full
    monkeypatch.setenv("CONDUCTOR_DROP_RULES", "ask_first")
    assert "Ask first only when" not in sp.get_system_prompt()
    monkeypatch.setenv("CONDUCTOR_ONTOLOGY", "off")
    lean = sp.get_system_prompt()
    assert "What an element is" not in lean and "Ask first only when" not in lean


def test_tool_groups_name_real_tools():
    assert all(t in REGISTRY for group in ABLATION_GROUPS.values() for t in group)
    assert expand_hidden(["group:proposals", "run_rsa"]) == {"edit_proposal", "compare_networks", "run_rsa"}
    with pytest.raises(ValueError):
        expand_hidden(["group:nope"])
    with pytest.raises(ValueError):
        expand_hidden(["run_rsaa"])


def _declared():
    return {fd.name for t in tool_schemas.tools_for_model() for fd in t.function_declarations}


def test_hidden_tools_are_not_declared(monkeypatch):
    monkeypatch.setenv("CONDUCTOR_ONTOLOGY", "on")
    monkeypatch.delenv("CONDUCTOR_HIDE_TOOLS", raising=False)
    assert tool_schemas.tools_for_model() is tool_schemas.TOOLS
    monkeypatch.setenv("CONDUCTOR_HIDE_TOOLS", "group:investigation,edit_proposal")
    declared = _declared()
    assert not declared & (set(ABLATION_GROUPS["investigation"]) | {"edit_proposal"})
    assert "run_rsa" in declared and "describe_network" in declared
    monkeypatch.setenv("CONDUCTOR_ONTOLOGY", "off")
    assert "describe_network" not in _declared()


def test_a_hidden_tool_called_by_name_is_refused(monkeypatch):
    class Provider:
        name, model = "stub", "stub-1"

        def __init__(self):
            self.responses = [ModelResponse(tool_calls=[ToolCall("investigate")]), ModelResponse(text="done")]

        def preflight(self):
            pass

        def chat(self, messages, tools, system, on_event=None):
            return self.responses.pop(0)

    ran = []
    provider = Provider()
    monkeypatch.setattr(loop_module, "get_provider", lambda *a, **k: provider)
    monkeypatch.setattr(loop_module, "get_system_prompt", lambda: "SYSTEM")
    monkeypatch.setattr(loop_module, "TOOL_DISPATCH", {"investigate": lambda **k: ran.append(1) or {"ok": 1}})
    monkeypatch.setattr(loop_module, "_append_to_log", lambda record: None)
    monkeypatch.setenv("CONDUCTOR_HIDE_TOOLS", "group:investigation")
    text, history = loop_module.run_agent_turn("why?", [], reading_check=False)
    assert not ran and "Unknown tool" in json.dumps(history)


def test_output_tokens_are_read_from_every_provider_shape():
    class Meta:
        candidates_token_count, thoughts_token_count = 40, 10

    class Gemini:
        usage_metadata = Meta()

    class Usage:
        output_tokens = 70

    class Claude:
        usage_metadata, usage = None, Usage()

    assert usage.output_tokens_from({"usage": {"output_tokens": 12}}) == 12
    assert usage.output_tokens_from({"usage": {"completion_tokens": 9}}) == 9
    assert usage.output_tokens_from({"eval_count": 5}) == 5
    assert usage.output_tokens_from(Gemini()) == 50
    assert usage.output_tokens_from(Claude()) == 70
    assert usage.output_tokens_from(None) is None and usage.output_tokens_from({}) is None


def test_run_task_passes_the_arm_settings_to_the_agent(monkeypatch):
    seen = []

    def fake_turn(question, history, **kwargs):
        seen.append(kwargs)
        return "ok", history

    monkeypatch.setattr(loop_module, "run_agent_turn", fake_turn)
    monkeypatch.setattr(whatif, "_clear_proposals", lambda: None)
    task = whatif.Task("t", ("q1", "q2"), (), deep=True)
    whatif.run_task(task, 1)
    assert [k["deep"] for k in seen] == [True, False] and seen[0]["reflection"] == "off"
    assert seen[0]["reading_check"] is None
    seen.clear()
    whatif.run_task(task, 1, reflection="inform", reading_check=False, deep=False)
    assert [k["deep"] for k in seen] == [False, False]
    assert seen[0]["reflection"] == "inform" and seen[0]["reading_check"] is False


# ---------------------------------------------------------------------------
# Arms and studies
# ---------------------------------------------------------------------------

def test_an_arm_is_parsed_and_checked():
    a = arms.parse("lean: drop_rules=routing,ask_first hide_tools=group:kpis review=inform max_steps=8")
    assert a.settings["drop_rules"] == ["routing", "ask_first"] and a.get("review") == "inform"
    env = a.env()
    assert env["CONDUCTOR_DROP_RULES"] == "routing,ask_first" and env["CONDUCTOR_HIDE_TOOLS"] == "group:kpis"
    assert env["DT_MAX_AGENT_TURNS"] == "8" and env["CONDUCTOR_ONTOLOGY"] == "on"
    assert a.run_kwargs() == {"reflection": "inform", "reading_check": None, "deep": None, "follow_suggestion": False}
    assert arms.parse("x: reading_check=off deep=on").run_kwargs()["reading_check"] is False
    assert arms.parse("x: deep=on").run_kwargs()["deep"] is True
    assert arms.parse("full:").describe() == "as it is" and arms.parse("full:").env() == {"CONDUCTOR_ONTOLOGY": "on"}
    for bad, msg in (("x: review=maybe", "review must be"), ("x: colour=red", "unknown knob"),
                     ("x y: review=off", "Arm name"), ("x: review", "not knob=value"),
                     ("x: drop_rules=nonsense", "Unknown prompt part"), ("x: hide_tools=group:no", "Unknown tool group"),
                     ("x: max_steps=many", "integer")):
        with pytest.raises(ValueError, match=msg):
            arms.parse(bad)


def test_the_model_and_effort_go_to_the_provider_setting(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    env = arms.parse("m: model=gpt-x effort=low env.OPENAI_TIMEOUT_S=30").env()
    assert env["OPENAI_MODEL"] == "gpt-x" and env["OPENAI_REASONING_EFFORT"] == "low" and env["OPENAI_TIMEOUT_S"] == "30"
    env = arms.parse("m: provider=anthropic model=claude-x effort=high").env()
    assert env["LLM_PROVIDER"] == "anthropic" and env["ANTHROPIC_MODEL"] == "claude-x" and env["ANTHROPIC_EFFORT"] == "high"


def test_the_hash_names_the_settings():
    assert arms.parse("a: review=off").hash == arms.parse("b: review=off").hash
    assert arms.parse("a: review=off").hash != arms.parse("a: review=inform").hash


@pytest.mark.parametrize("name", sorted(arms.PRESETS))
def test_every_study_parses_and_names_real_suites(name):
    preset = arms.PRESETS[name]
    parsed = arms.preset_arms(name)
    assert len(parsed) >= 2 and len({a.name for a in parsed}) == len(parsed)
    assert set(preset.suites) <= set(suites.names())
    for a in parsed:
        arms.footprint(a)


def test_the_ontology_study_needs_both_backends():
    modes = {a.name: a.backend_mode() for a in arms.preset_arms("ontology")}
    assert modes == {"on": "on", "off": "on", "none": "none"}
    fp = arms.footprint(arms.parse("off: ontology=off"))
    assert fp["prompt_chars_removed"]["element_identity"] > 0 and "describe_network" in fp["tools_hidden"]


def test_every_suite_lists_its_tasks_once():
    items = suites.items(suites.names())
    keys = [i.key for i in items]
    assert len(keys) == len(set(keys)) and all(i.network in suites.SCENARIOS for i in items)
    some = suites.items(["whatif"], ["dc_flat"])
    assert [i.key for i in some] == ["whatif/dc_flat"]
    with pytest.raises(ValueError):
        suites.items(["nope"])
    with pytest.raises(ValueError):
        suites.items(["whatif"], ["not_a_task"])


def test_a_pilot_draws_the_same_tasks_from_every_suite():
    items = suites.items(["whatif", "clarify", "options"])
    a, b = ablation.sample_items(items, 2, seed=1), ablation.sample_items(items, 2, seed=1)
    assert [i.key for i in a] == [i.key for i in b] and len(a) == 6
    assert {i.suite for i in a} == {"whatif", "clarify", "options"}


def test_the_plan_counts_the_turns_and_the_schedule_interleaves():
    study = ablation.resolve(None, ["a:", "b: follow_suggestion=on"], "whatif", "dc_flat,reinforce_after_load")
    text, total = ablation.render_plan(study, 2, "interleaved")
    # 1 + 2 turns per repeat; b adds a turn per task.
    assert total == (3 + 5) * 2 and "| b | follow_suggestion=on |" in text
    order = [(a.name, r) for a, r in ablation._schedule(study["arms"], 2, "interleaved")]
    assert order == [("a", [1]), ("b", [1]), ("a", [2]), ("b", [2])]
    assert [(a.name, r) for a, r in ablation._schedule(study["arms"], 2, "blocked")] == [("a", [1, 2]), ("b", [1, 2])]


def test_a_plan_over_the_budget_runs_nothing(tmp_path, monkeypatch):
    study = ablation.resolve(None, ["a:", "b: review=inform"], "clarify", "")
    monkeypatch.setattr(ablation, "_start_backend", lambda *a, **k: pytest.fail("started a backend"))
    assert ablation.run_study(study, 3, tmp_path, max_turns=5, yes=True) == 2
    assert not (tmp_path / "manifest.json").exists()


def test_an_existing_run_folder_is_not_overwritten(tmp_path):
    (tmp_path / "manifest.json").write_text("{}")
    assert ablation.main(["run", "--arm", "a:", "--suites", "clarify", "--out", str(tmp_path), "--yes"]) == 2


# ---------------------------------------------------------------------------
# One arm in its process: the records it writes, and resuming
# ---------------------------------------------------------------------------

def _fake_backend(monkeypatch, answer="Bus 9 at 2 MW.", fail_on=()):
    calls = []
    monkeypatch.setattr("evaluation.semantics.check_backend_mode", lambda arm: None)
    monkeypatch.setattr(whatif, "_post_backend", lambda path, body: calls.append((path, body)) or {})

    def fake_run_task(task, repeat, follow=False, **kwargs):
        calls.append(("run", task.id, repeat, kwargs))
        if task.id in fail_on:
            raise RuntimeError("backend went away")
        return [{"turn": 1, "user": task.turns[0], "assistant": answer, "tool_calls": [], "tool_results": [],
                 "model_calls": 2, "prompt_tokens": [100, 120], "output_tokens": [30], "timing": {"total_s": 4.0}}]

    monkeypatch.setattr(whatif, "run_task", fake_run_task)
    return calls


def test_the_child_writes_one_record_per_run_and_resumes(tmp_path, monkeypatch):
    calls = _fake_backend(monkeypatch, fail_on=("dc_flat",))
    arm = arms.parse("inf: review=inform")
    spec = json.dumps({"name": arm.name, "settings": arm.settings})
    keys = ["whatif/dc_flat", "whatif/wind_week"]
    ablation._child(tmp_path, spec, keys, [1], "0/2/0")
    runs = sorted((tmp_path / "runs" / "inf" / "whatif").glob("*.jsonl"))
    assert [p.name for p in runs] == ["dc_flat-01.jsonl", "wind_week-01.jsonl"]
    meta, records = st.read_run(runs[1])
    assert meta["arm"] == "inf" and meta["arm_hash"] == arm.hash and meta["settings"] == {"review": "inform"}
    assert ("/api/scenario", {"name": "cigre_mv"}) in calls
    assert [c[3] for c in calls if c[0] == "run"][0] == {"reflection": "inform", "reading_check": None, "deep": None}
    # Resuming: the good run is kept, the runner error is run again.
    calls.clear()
    ablation._child(tmp_path, spec, keys, [1], "0/2/0")
    assert [c[1] for c in calls if c[0] == "run"] == ["dc_flat"]
    # Other settings under the same name do not count as done.
    other = arms.parse("inf: review=warn")
    assert not st.completed(runs[1], other.hash) and st.completed(runs[1], arm.hash)


def test_a_reasoning_item_is_prepared_and_undone(tmp_path, monkeypatch):
    calls = _fake_backend(monkeypatch)
    import evaluation.reasoning as reasoning
    monkeypatch.setattr(reasoning, "_post_backend", lambda path, body: calls.append((path, body)) or {})
    item = next(i for i in suites.items(["reasoning"]) if i.task.id == "names_cigre")
    arm = arms.parse("full:")
    ablation._child(tmp_path, json.dumps({"name": "full", "settings": {}}), [item.key], [1], "0/1/0")
    paths = [c[0] for c in calls if isinstance(c[0], str) and c[0].startswith("/api/")]
    assert paths[0] == "/api/scenario" and "/api/names" in paths and paths[-1] == "/api/network/declare"
    assert arm.hash


# ---------------------------------------------------------------------------
# Statistics and tables
# ---------------------------------------------------------------------------

def test_wilson_and_the_sign_test():
    lo, hi = st.wilson(8, 10)
    assert round(lo, 3) == 0.490 and round(hi, 3) == 0.943
    assert st.wilson(0, 0) == (0.0, 1.0)
    assert st.sign_test(0, 0) == 1.0
    assert round(st.sign_test(9, 1), 4) == 0.0215
    assert st.sign_test(3, 3) == 1.0


def _row(arm, task, passed, n=2, repeat=1, **kw):
    return {"arm": arm, "suite": "s", "task": task, "family": "", "repeat": repeat, "runner_error": False,
            "checks": {f"c{k}": k < passed for k in range(n)}, "n_checks": n, "n_passed": passed,
            "passed_all": passed == n, "tool_calls": 2, "model_calls": 3, "input_tokens": 1000, "output_tokens": 100,
            "seconds": 5.0, "figures": 1, "ungrounded": 0, "mischaracterised": 0, "entity_errors": 0,
            "failing_turns": 0, "provider": "p", "model": "m", **kw}


def test_the_difference_is_over_tasks_with_an_interval():
    rows = [_row("a", f"t{i}", 2) for i in range(6)] + [_row("b", f"t{i}", 1 if i < 4 else 2) for i in range(6)]
    d = st.paired_difference(rows, "b", "a")
    assert d["tasks"] == 6 and d["better"] == 0 and d["worse"] == 4 and d["same"] == 2
    assert round(d["diff"], 4) == round(-0.5 * 4 / 6, 4) and d["lo"] <= d["diff"] <= d["hi"] <= 0
    assert d == st.paired_difference(rows, "b", "a")            # the seed makes it repeatable


def test_the_outputs_are_written(tmp_path):
    rows = [_row("full", "t1", 2), _row("lean", "t1", 1), _row("lean", "t2", 0, runner_error=True)]
    md = st.write_outputs(tmp_path, rows, ["full", "lean"], manifest={"study": "x", "arms": [{"name": "full"}]})
    assert "| full | 1 | 100.0" in md and "| lean | 2 (1 err)" in md and "## Checks where the arms differ" in md
    for name in ("analysis.md", "summary.csv", "runs.csv", "checks.csv", "summary.tex", "results.json"):
        assert (tmp_path / name).exists()
    assert r"\toprule" in (tmp_path / "summary.tex").read_text()
    assert len((tmp_path / "checks.csv").read_text().strip().splitlines()) == 1 + 2 + 2


def test_analyse_grades_a_run_folder_end_to_end(tmp_path, monkeypatch):
    _fake_backend(monkeypatch, answer="WKA 7, PV 3 and PV 11 are renewable.")
    for name, settings in (("on", {}), ("off", {"ontology": "off"})):
        ablation._child(tmp_path, json.dumps({"name": name, "settings": settings}),
                        ["semantics-cigre/cigre_renewables", "semantics-cigre/cigre_storage"], [1, 2], "0/0/0")
    (tmp_path / "manifest.json").write_text(json.dumps({"study": "custom", "baseline": "on",
                                                        "arms": [{"name": "on"}, {"name": "off"}]}))
    md = ablation.analyse([tmp_path])
    assert "| on | 4 | 50.0" in md and "| off | 4 | 50.0" in md
    assert "Δ" in md and "+0.0" in md
    assert "| Arm | Done |" in ablation.status(tmp_path)
    rows = st.grade_runs([tmp_path])
    assert {r["input_tokens"] for r in rows} == {220} and {r["output_tokens"] for r in rows} == {30}


def test_resuming_counts_only_what_is_left(tmp_path, monkeypatch):
    _fake_backend(monkeypatch, fail_on=("dc_flat",))
    study = ablation.resolve(None, ["a:"], "whatif", "dc_flat,wind_week")
    assert ablation.remaining_turns(study, 1, tmp_path) == 2
    arm = study["arms"][0]
    ablation._child(tmp_path, json.dumps({"name": arm.name, "settings": arm.settings}),
                    [i.key for i in study["items"]], [1], "0/2/0")
    assert ablation.remaining_turns(study, 1, tmp_path) == 1          # the runner error is left to run
    monkeypatch.setattr(ablation, "_start_backend", lambda *a, **k: pytest.fail("started a backend"))
    assert ablation.run_study(study, 1, tmp_path, max_turns=0, yes=True) == 2


def test_cached_tokens_are_read_from_every_provider_shape():
    class Meta:
        prompt_token_count, cached_content_token_count = 1000, 800

    class Gemini:
        usage_metadata = Meta()

    class Usage:
        input_tokens, cache_read_input_tokens, cache_creation_input_tokens = 50, 900, 100

    class Claude:
        usage_metadata, usage = None, Usage()

    assert usage.cached_tokens_from({"usage": {"input_tokens": 100, "input_tokens_details": {"cached_tokens": 90}}}) == 90
    assert usage.cached_tokens_from({"usage": {"prompt_tokens": 100, "prompt_tokens_details": {"cached_tokens": 7}}}) == 7
    assert usage.cached_tokens_from({"usage": {"prompt_tokens": 100}}) is None
    assert usage.cached_tokens_from({"prompt_eval_count": 10}) is None
    assert usage.cached_tokens_from(Gemini()) == 800
    assert usage.cached_tokens_from(Claude()) == 900
    # Anthropic reports the cache apart from input_tokens: the whole prompt is their sum.
    assert usage.prompt_tokens_from(Claude()) == 1050


def test_the_cost_is_only_from_the_prices_given_and_says_its_assumptions():
    est = {"source": "x", "input_tokens": 100_000, "cached_tokens": 97_000, "output_tokens": 400, "seconds": 10}
    lines = "\n".join(ablation.render_cost(est, 1000, None, None, None))
    assert "97 %" in lines and "$" not in lines and "CONDUCTOR knows no prices" in lines
    # 3 M uncached at 1, 97 M cached at 0.1, 0.4 M output at 4 = 3 + 9.7 + 1.6
    assert "~$14.30" in "\n".join(ablation.render_cost(est, 1000, 1.0, 0.1, 4.0))
    upper = "\n".join(ablation.render_cost(est, 1000, 1.0, None, None))
    assert "~$100.00" in upper and "upper bound" in upper and "output not included" in upper
    unknown = dict(est, cached_tokens=None)
    assert "cache use unknown" in "\n".join(ablation.render_cost(unknown, 1000, 1.0, 0.1, 4.0))


def test_the_reading_check_is_counted_per_turn_by_its_gravest_outcome():
    assert st._reading({"reading_check": {"outcomes": [{"status": "holds"}, {"status": "changes"}]}}) == "changes"
    assert st._reading({"reading_check": {"outcomes": [{"status": "holds"}, {"status": "shifts"}]}}) == "shifts"
    assert st._reading({"reading_check": {"outcomes": []}}) is None and st._reading({}) is None
    rows = [_row("a", "t1", 2, reading_checked=2, reading_shifts=1, reading_changes=1), _row("a", "t2", 2)]
    md = st.render_markdown(rows, ["a"])
    assert "## The reading check" in md and "| a | 2 | 1 (50 %) | 1 (50 %" in md


def test_the_components_study_has_a_solver_only_arm():
    arm = next(a for a in arms.preset_arms("components") if a.name == "solver-only")
    assert arm.backend_mode() == "none" and arm.run_kwargs()["reading_check"] is False


# ---------------------------------------------------------------------------
# Modularity: failures, fail fast, re-runs, narrowing and extending a run
# ---------------------------------------------------------------------------

def test_infrastructure_failures_are_failed_runs_and_the_agents_own_outcomes_are_graded():
    assert st.failure([{"status": "completed"}]) is None
    assert st.failure([{"status": "max_turns_exceeded"}]) is None          # the agent's own: graded
    for bad in ("completed_with_warning", "rate_limit", "timeout", "infra_error", "quota_exhausted", "runner_error"):
        assert st.failure([{"status": "completed"}, {"status": bad}]) == bad
    assert st.failure([]) == "no records"


def test_a_child_stops_after_three_failures_in_a_row_and_at_once_on_quota(tmp_path, monkeypatch):
    calls = _fake_backend(monkeypatch, fail_on=("dc_flat", "wind_week", "new_site_n1", "trafo_upgrade"))
    spec = json.dumps({"name": "a", "settings": {}})
    keys = ["whatif/dc_flat", "whatif/wind_week", "whatif/new_site_n1", "whatif/trafo_upgrade"]
    assert ablation._child(tmp_path, spec, keys, [1], "0/4/0") == ablation.EXIT_FAILING
    assert len([c for c in calls if c[0] == "run"]) == 3

    def quota(task, repeat, follow=False, **kw):
        return [{"turn": 1, "status": "quota_exhausted", "assistant": "quota"}]
    monkeypatch.setattr(whatif, "run_task", quota)
    assert ablation._child(tmp_path / "q", spec, keys, [1], "0/4/0") == ablation.EXIT_FATAL
    assert len(list((tmp_path / "q").rglob("*.jsonl"))) == 1


def _planned_run(tmp_path, monkeypatch):
    """A finished two-arm run of two tasks, two repeats; one run failed."""
    _fake_backend(monkeypatch, fail_on=("wind_week",))
    study = ablation.resolve(None, ["a:", "b: review=inform"], "whatif", "dc_flat,wind_week")
    for arm in study["arms"]:
        ablation._child(tmp_path, json.dumps({"name": arm.name, "settings": arm.settings}),
                        [i.key for i in study["items"]], [1, 2], "0/8/0")
    ablation.write_manifest(tmp_path, study, 2, "interleaved", "finished", "2026-11-03T10:00:00")
    return study


def test_status_says_what_a_resume_would_run(tmp_path, monkeypatch):
    _planned_run(tmp_path, monkeypatch)
    text = ablation.status(tmp_path, detail=True)
    assert "| a | 2 | 2 | 0 | 0 | 4 |" in text and "| b | 2 | 2 | 0 | 0 | 4 |" in text
    assert "a · whatif/wind_week · repeat 1: failed (runner_error)" in text


def test_a_selector_names_runs_and_a_rerun_sets_them_aside_with_its_reason(tmp_path, monkeypatch):
    _planned_run(tmp_path, monkeypatch)
    with pytest.raises(ValueError):
        ablation.parse_selector("colour=red")
    with pytest.raises(ValueError):
        ablation.parse_selector("")
    sel = ablation.parse_selector("arm=b task=whatif/dc_flat repeat=2")
    assert ablation.replace_runs(tmp_path, [sel], "grader bug in dc_flat") == 1
    assert not (tmp_path / "runs" / "b" / "whatif" / "dc_flat-02.jsonl").exists()
    assert list((tmp_path / "runs" / "_replaced").rglob("dc_flat-02.jsonl"))
    m = json.loads((tmp_path / "manifest.json").read_text())
    assert m["changes"][-1]["reason"] == "grader bug in dc_flat" and "b/whatif/dc_flat#2" in m["changes"][-1]["what"]
    # Set-aside records are not graded, and the manifest keeps its change log when rewritten.
    assert all("_replaced" not in str(p) for p in st.run_files(tmp_path))
    study = ablation.resolve(None, ["a:", "b: review=inform"], "whatif", "dc_flat,wind_week")
    ablation.write_manifest(tmp_path, study, 2, "interleaved", "finished", "x")
    assert json.loads((tmp_path / "manifest.json").read_text())["changes"]


def test_a_resume_can_run_part_of_the_plan_extend_it_and_rerun(tmp_path, monkeypatch):
    study = _planned_run(tmp_path, monkeypatch)
    seen = {}

    def fake_run_study(sub, repeats, out, **kw):
        seen.update(arms=[a.name for a in sub["arms"]], items=[i.key for i in sub["items"]], repeats=repeats,
                    plan_arms=[a.name for a in kw["plan_study"]["arms"]], plan_repeats=kw["plan_repeats"])
        return 0
    monkeypatch.setattr(ablation, "run_study", fake_run_study)
    assert ablation.main(["run", "--resume", str(tmp_path), "--only", "b", "--tasks", "dc_flat", "--yes"]) == 0
    assert seen == {"arms": ["b"], "items": ["whatif/dc_flat"], "repeats": 2, "plan_arms": ["a", "b"],
                    "plan_repeats": 2}
    assert ablation.main(["run", "--resume", str(tmp_path), "--repeats", "4", "--reason", "more power", "--yes"]) == 0
    assert seen["repeats"] == 4 and "extended from 2 to 4" in json.loads(
        (tmp_path / "manifest.json").read_text())["changes"][-1]["what"]
    assert ablation.main(["run", "--resume", str(tmp_path), "--rerun", "arm=a", "--yes"]) == 2      # no reason
    assert ablation.main(["run", "--resume", str(tmp_path), "--only", "zzz", "--yes"]) == 2
    assert ablation.main(["run", "--rerun", "arm=a", "--arm", "a:", "--suites", "whatif", "--yes"]) == 2
    assert study


def test_a_dead_backend_is_restarted_once_then_the_run_stops_cleanly(tmp_path, monkeypatch):
    class Proc:
        def kill(self):
            pass

        def terminate(self):
            pass

        def wait(self, timeout=None):
            return 0
    started = []
    monkeypatch.setattr(ablation, "_start_backend", lambda out, port, mode: started.append(mode) or Proc())
    monkeypatch.setattr(ablation, "_healthy", lambda url: False)
    real_run = ablation.subprocess.run

    def fake_run(cmd, *a, **k):
        if cmd[0] == "git":
            return real_run(cmd, *a, **k)
        return type("R", (), {"returncode": ablation.EXIT_FAILING})()
    monkeypatch.setattr(ablation.subprocess, "run", fake_run)
    study = ablation.resolve(None, ["a:"], "clarify", "")
    rc = ablation.run_study(study, 1, tmp_path, yes=True)
    assert rc == ablation.EXIT_FAILING and started == ["on", "on"]                 # restarted once
    m = json.loads((tmp_path / "manifest.json").read_text())
    assert m["status"] == "stopped" and any("restarted" in c["what"] for c in m["changes"])
    assert any(c["what"].startswith("stopped") for c in m["changes"])
