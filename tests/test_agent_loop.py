"""The agentic turn loop, driven by a stub provider.

The loop is where turn control, tool dispatch, and the MAX_AGENT_TURNS bound
live. It is provider-agnostic, so it can be exercised without any LLM at all —
which also means these tests pin the contract every provider must satisfy.
"""

import json

import pytest

from llm_agent.agent import loop as loop_module
from llm_agent.agent.providers.base import ModelResponse, ToolCall


class StubProvider:
    """Returns a scripted sequence of responses and records what it was sent."""

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
def stub(monkeypatch):
    """Install a stub provider and a no-op tool table."""
    def install(responses, tools=None):
        provider = StubProvider(responses)
        monkeypatch.setattr(loop_module, "get_provider", lambda *a, **k: provider)
        monkeypatch.setattr(loop_module, "get_system_prompt", lambda: "SYSTEM")
        monkeypatch.setattr(loop_module, "TOOLS", [])
        monkeypatch.setattr(
            loop_module, "TOOL_DISPATCH", tools if tools is not None else {}
        )
        monkeypatch.setattr(loop_module, "_append_to_log", lambda record: None)
        return provider
    return install


class TestBasicTurn:
    def test_prose_reply_ends_the_turn(self, stub):
        provider = stub([ModelResponse(text="All clear.")])
        text, history = loop_module.run_agent_turn("status?", [])
        assert text == "All clear."
        assert len(provider.calls) == 1

    def test_history_records_user_and_assistant(self, stub):
        stub([ModelResponse(text="ok")])
        _, history = loop_module.run_agent_turn("hi", [])
        assert [m["role"] for m in history] == ["user", "assistant"]

    def test_caller_history_is_not_mutated(self, stub):
        stub([ModelResponse(text="ok")])
        original = []
        loop_module.run_agent_turn("hi", original)
        assert original == []

    def test_history_is_json_serialisable(self, stub):
        """Streamlit stores it in session_state; vendor objects would break it."""
        stub([
            ModelResponse(tool_calls=[ToolCall("t", {"a": 1})]),
            ModelResponse(text="done"),
        ], tools={"t": lambda **kw: {"ok": True}})
        _, history = loop_module.run_agent_turn("go", [])
        assert json.loads(json.dumps(history)) == history

    def test_prior_history_is_carried_forward(self, stub):
        provider = stub([ModelResponse(text="ok")])
        prior = [{"role": "user", "content": "earlier"},
                 {"role": "assistant", "content": "reply"}]
        loop_module.run_agent_turn("now", prior)
        sent = provider.calls[0]["messages"]
        assert sent[0]["content"] == "earlier"
        assert sent[-1]["content"] == "now"


class TestToolDispatch:
    def test_tool_is_called_and_its_result_appended(self, stub):
        seen = {}

        def fake_tool(**kwargs):
            seen.update(kwargs)
            return {"violations": 0}

        stub([
            ModelResponse(tool_calls=[ToolCall("run_rsa", {"scale": 1.5})]),
            ModelResponse(text="No violations."),
        ], tools={"run_rsa": fake_tool})

        text, history = loop_module.run_agent_turn("run rsa", [])
        assert seen == {"scale": 1.5}
        assert text == "No violations."
        assert [m["role"] for m in history] == [
            "user", "assistant", "tool", "assistant"
        ]
        assert history[2]["content"] == {"violations": 0}

    def test_parallel_tool_calls_each_produce_a_result(self, stub):
        stub([
            ModelResponse(tool_calls=[ToolCall("a"), ToolCall("b")]),
            ModelResponse(text="both done"),
        ], tools={"a": lambda **k: {"r": 1}, "b": lambda **k: {"r": 2}})
        _, history = loop_module.run_agent_turn("go", [])
        assert [m["role"] for m in history] == [
            "user", "assistant", "tool", "tool", "assistant"
        ]

    def test_unknown_tool_returns_an_error_to_the_model(self, stub):
        stub([
            ModelResponse(tool_calls=[ToolCall("nope")]),
            ModelResponse(text="recovered"),
        ], tools={})
        text, history = loop_module.run_agent_turn("go", [])
        assert "Unknown tool" in history[2]["content"]["error"]
        assert text == "recovered"

    def test_tool_exception_is_fed_back_rather_than_aborting(self, stub):
        def broken(**kwargs):
            raise ValueError("backend exploded")

        stub([
            ModelResponse(tool_calls=[ToolCall("boom")]),
            ModelResponse(text="handled"),
        ], tools={"boom": broken})
        text, history = loop_module.run_agent_turn("go", [])
        assert "backend exploded" in history[2]["content"]["error"]
        assert text == "handled"

    def test_call_id_is_carried_onto_the_result(self, stub):
        stub([
            ModelResponse(tool_calls=[ToolCall("t", id="call-1")]),
            ModelResponse(text="done"),
        ], tools={"t": lambda **k: {}})
        _, history = loop_module.run_agent_turn("go", [])
        assert history[2]["id"] == "call-1"


class TestTurnBound:
    def test_stops_at_max_agent_turns(self, stub, monkeypatch):
        monkeypatch.setattr(loop_module, "MAX_AGENT_TURNS", 3)
        # A model that never stops asking for tools.
        provider = stub(
            [ModelResponse(tool_calls=[ToolCall("t")]) for _ in range(10)],
            tools={"t": lambda **k: {}},
        )
        text, _ = loop_module.run_agent_turn("loop forever", [])
        assert len(provider.calls) == 4                  # 3 steps, then one call asking for an answer
        assert "all 3 steps" in text                    # it kept asking for tools: the warning stands

    def test_at_the_limit_it_answers_from_what_it_has(self, stub, monkeypatch):
        monkeypatch.setattr(loop_module, "MAX_AGENT_TURNS", 2)
        provider = stub(
            [ModelResponse(tool_calls=[ToolCall("t")]), ModelResponse(tool_calls=[ToolCall("t")]),
             ModelResponse(text="Partial answer: the grid is secure; the proposal could not be built.")],
            tools={"t": lambda **k: {}},
        )
        text, _ = loop_module.run_agent_turn("loop", [])
        assert len(provider.calls) == 3
        assert text.startswith("Partial answer") and "all 2 steps" in text

    def test_one_llm_call_per_turn(self, stub):
        provider = stub([
            ModelResponse(tool_calls=[ToolCall("t")]),
            ModelResponse(tool_calls=[ToolCall("t")]),
            ModelResponse(text="done"),
        ], tools={"t": lambda **k: {}})
        loop_module.run_agent_turn("go", [])
        assert len(provider.calls) == 3


class TestEvents:
    def test_emits_the_expected_event_sequence(self, stub):
        stub([
            ModelResponse(tool_calls=[ToolCall("t")]),
            ModelResponse(text="done"),
        ], tools={"t": lambda **k: {}})

        events = []
        loop_module.run_agent_turn("go", [], on_event=lambda e, d: events.append(e))
        assert events == ["llm_call", "tool_start", "tool_done", "llm_call"]

    def test_tool_failure_emits_tool_error(self, stub):
        def broken(**kwargs):
            raise RuntimeError("nope")

        stub([
            ModelResponse(tool_calls=[ToolCall("t")]),
            ModelResponse(text="done"),
        ], tools={"t": broken})

        events = []
        loop_module.run_agent_turn("go", [], on_event=lambda e, d: events.append(e))
        assert "tool_error" in events

    def test_loop_works_without_an_event_callback(self, stub):
        stub([ModelResponse(text="ok")])
        assert loop_module.run_agent_turn("go", [])[0] == "ok"


class TestErrorSurfacing:
    def test_provider_runtime_error_becomes_the_reply(self, stub, monkeypatch):
        provider = stub([ModelResponse(text="unused")])

        def boom(**kwargs):
            raise RuntimeError("⚠️ Rate limit hit.")

        monkeypatch.setattr(provider, "chat", boom)
        text, _ = loop_module.run_agent_turn("go", [])
        assert "Rate limit" in text

    def test_unexpected_error_is_reported_not_raised(self, stub, monkeypatch):
        provider = stub([ModelResponse(text="unused")])

        def boom(**kwargs):
            raise ValueError("something odd")

        monkeypatch.setattr(provider, "chat", boom)
        text, _ = loop_module.run_agent_turn("go", [])
        assert "Unexpected error" in text and "something odd" in text


class TestSystemPromptIsFreshPerTurn:
    def test_prompt_is_read_each_turn(self, stub, monkeypatch):
        """Grid constants change when a network is uploaded mid-session."""
        provider = stub([ModelResponse(text="a")])
        loop_module.run_agent_turn("one", [])
        assert provider.calls[0]["system"] == "SYSTEM"

        monkeypatch.setattr(loop_module, "get_system_prompt", lambda: "CHANGED")
        provider._responses = [ModelResponse(text="b")]
        loop_module.run_agent_turn("two", [])
        assert provider.calls[-1]["system"] == "CHANGED"


class TestParameterIntegrity:
    """The guard must sit before dispatch, not after."""

    def test_impossible_parameters_never_reach_the_tool(self, stub):
        calls = []

        def spy(**kwargs):
            calls.append(kwargs)
            return {"ok": True}

        stub([
            # An inverted voltage band: no parameterisation satisfies it.
            ModelResponse(tool_calls=[ToolCall("run_rsa", {"vm_lower_pu": 1.05,
                                                          "vm_upper_pu": 0.95})]),
            ModelResponse(text="corrected"),
        ], tools={"run_rsa": spy})

        text, history = loop_module.run_agent_turn("check security", [])
        assert calls == [], "the solver ran on parameters that cannot be satisfied"
        assert "invalid_parameters" in history[2]["content"]
        assert text == "corrected"

    def test_the_model_is_told_what_to_fix(self, stub):
        stub([
            ModelResponse(tool_calls=[ToolCall("run_rsa", {"load_sigma": 5.0})]),
            ModelResponse(text="done"),
        ], tools={"run_rsa": lambda **k: {"ok": True}})

        _, history = loop_module.run_agent_turn("go", [])
        assert "load_sigma" in history[2]["content"]["invalid_parameters"]

    def test_rejection_emits_an_event(self, stub):
        stub([
            ModelResponse(tool_calls=[ToolCall("run_rsa", {"n_samples": -1})]),
            ModelResponse(text="done"),
        ], tools={"run_rsa": lambda **k: {"ok": True}})

        events = []
        loop_module.run_agent_turn("go", [], on_event=lambda e, d: events.append(e))
        assert "tool_rejected" in events
        assert "tool_start" not in events

    def test_valid_parameters_still_execute(self, stub):
        calls = []
        stub([
            ModelResponse(tool_calls=[ToolCall("run_rsa", {"vm_lower_pu": 0.95,
                                                           "vm_upper_pu": 1.05})]),
            ModelResponse(text="secure"),
        ], tools={"run_rsa": lambda **k: calls.append(k) or {"ok": True}})

        text, _ = loop_module.run_agent_turn("go", [])
        assert len(calls) == 1
        assert text == "secure"

    def test_self_contradictory_results_are_annotated_not_dropped(self, stub):
        payload = {
            "total_violations": 0,
            "violations": [],
            "all_voltages": [{"bus_name": "S3", "vm_pu": 1.09}],
            "thresholds_used": {"vm_lower_pu": 0.95, "vm_upper_pu": 1.05},
            "gen_dispatch": {"S3": 3.5},
        }
        stub([
            ModelResponse(tool_calls=[ToolCall("run_rsa", {})]),
            ModelResponse(text="reported"),
        ], tools={"run_rsa": lambda **k: payload})

        _, history = loop_module.run_agent_turn("go", [])
        tool_msg = history[2]["content"]
        assert tool_msg["_integrity_warnings"], "contradiction was not surfaced"
        # The model still needs the underlying data.
        assert tool_msg["gen_dispatch"] == {"S3": 3.5}

    def test_consistent_results_are_not_annotated(self, stub):
        stub([
            ModelResponse(tool_calls=[ToolCall("run_rsa", {})]),
            ModelResponse(text="ok"),
        ], tools={"run_rsa": lambda **k: {"total_violations": 0, "violations": []}})

        _, history = loop_module.run_agent_turn("go", [])
        assert "_integrity_warnings" not in history[2]["content"]


class TestCrossToolConsistency:
    """Reproduces a live failure: a chain that silently mixed two operating points."""

    def test_a_chain_on_two_different_timestamps_is_flagged(self, stub):
        # The model passed an explicit timestamp to the assessment and dropped it
        # from the follow-up, which fell back to the simulation clock.
        stub([
            ModelResponse(tool_calls=[ToolCall("run_rsa", {"timestamp": "2022-01-02 21:45"})]),
            ModelResponse(tool_calls=[ToolCall("attribution", {})]),
            ModelResponse(text="combined answer"),
        ], tools={
            "run_rsa": lambda **k: {"timestamp": "2022-01-02 21:45:00", "total_violations": 11},
            "attribution": lambda **k: {"timestamp": "2022-01-01 00:00:00", "violations": []},
        })

        _, history = loop_module.run_agent_turn("assess then explain", [])
        second_result = history[4]["content"]
        assert second_result["_integrity_warnings"], "the mismatch went unreported"
        assert "different operating points" in second_result["_integrity_warnings"][0]["message"]

    def test_a_consistent_chain_is_not_flagged(self, stub):
        stub([
            ModelResponse(tool_calls=[ToolCall("run_rsa", {})]),
            ModelResponse(tool_calls=[ToolCall("attribution", {})]),
            ModelResponse(text="done"),
        ], tools={
            "run_rsa": lambda **k: {"timestamp": "2022-01-02 21:45:00"},
            "attribution": lambda **k: {"timestamp": "2022-01-02 21:45:00"},
        })
        _, history = loop_module.run_agent_turn("go", [])
        assert "_integrity_warnings" not in history[4]["content"]

    def test_mixing_measurements_and_forecasts_is_flagged(self, stub):
        stub([
            ModelResponse(tool_calls=[ToolCall("a", {})]),
            ModelResponse(tool_calls=[ToolCall("b", {})]),
            ModelResponse(text="done"),
        ], tools={
            "a": lambda **k: {"data_source": "measurements"},
            "b": lambda **k: {"data_source": "forecasts"},
        })
        _, history = loop_module.run_agent_turn("go", [])
        assert history[4]["content"]["_integrity_warnings"]

    def test_the_mismatch_reaches_the_user_interface(self, stub):
        stub([
            ModelResponse(tool_calls=[ToolCall("a", {})]),
            ModelResponse(tool_calls=[ToolCall("b", {})]),
            ModelResponse(text="done"),
        ], tools={
            "a": lambda **k: {"timestamp": "2022-01-02 21:45:00"},
            "b": lambda **k: {"timestamp": "2022-01-01 00:00:00"},
        })
        events = []
        loop_module.run_agent_turn("go", [], on_event=lambda e, d: events.append(e))
        assert "integrity_warning" in events

    def test_tracking_is_per_turn_not_per_session(self, stub):
        """A new question at a new timestamp is not a mismatch."""
        stub([ModelResponse(tool_calls=[ToolCall("a", {})]), ModelResponse(text="one")],
             tools={"a": lambda **k: {"timestamp": "2022-01-02 21:45:00"}})
        _, history = loop_module.run_agent_turn("first", [])

        stub([ModelResponse(tool_calls=[ToolCall("a", {})]), ModelResponse(text="two")],
             tools={"a": lambda **k: {"timestamp": "2022-03-01 08:00:00"}})
        _, history2 = loop_module.run_agent_turn("second", history)
        assert "_integrity_warnings" not in history2[-2]["content"]


class TestRepeatedCalls:
    """An identical call later in the same question is answered from the
    earlier one instead of running again — unless something changed since, or
    the earlier one failed for a reason a retry might fix."""

    @staticmethod
    def counting(result=None):
        calls = []

        def tool(**kwargs):
            calls.append(kwargs)
            return dict(result if result is not None else {"value": len(calls)})
        return tool, calls

    def test_an_identical_call_is_reused(self, stub):
        tool, calls = self.counting()
        stub([ModelResponse(tool_calls=[ToolCall("read", {"x": 1})]),
              ModelResponse(tool_calls=[ToolCall("read", {"x": 1})]),
              ModelResponse(text="done")], tools={"read": tool})
        _, history = loop_module.run_agent_turn("go", [])
        assert len(calls) == 1
        reused = [m for m in history if m["role"] == "tool"][1]["content"]
        assert reused["reused_from"] == 1 and "not run again" in reused["note"]
        assert "value" not in reused                     # a pointer, not the payload again

    def test_other_arguments_run_again(self, stub):
        tool, calls = self.counting()
        stub([ModelResponse(tool_calls=[ToolCall("read", {"x": 1})]),
              ModelResponse(tool_calls=[ToolCall("read", {"x": 2})]),
              ModelResponse(text="done")], tools={"read": tool})
        loop_module.run_agent_turn("go", [])
        assert len(calls) == 2

    def test_two_identical_calls_in_one_reply(self, stub):
        tool, calls = self.counting()
        stub([ModelResponse(tool_calls=[ToolCall("read", {"x": 1}), ToolCall("read", {"x": 1})]),
              ModelResponse(text="done")], tools={"read": tool})
        _, history = loop_module.run_agent_turn("go", [])
        assert len(calls) == 1 and [m for m in history if m["role"] == "tool"][1]["content"]["reused_from"] == 1

    def test_a_state_change_in_between_runs_it_again(self, stub):
        read, reads = self.counting()
        move, moves = self.counting({"current_timestamp": "t"})
        stub([ModelResponse(tool_calls=[ToolCall("read", {"x": 1})]),
              ModelResponse(tool_calls=[ToolCall("advance_timestamp", {"steps": 1})]),
              ModelResponse(tool_calls=[ToolCall("read", {"x": 1})]),
              ModelResponse(tool_calls=[ToolCall("advance_timestamp", {"steps": 1})]),
              ModelResponse(text="done")], tools={"read": read, "advance_timestamp": move})
        loop_module.run_agent_turn("go", [])
        assert len(reads) == 2                           # the clock moved: the read is not reused
        assert len(moves) == 2                           # and moving the clock twice moves it twice

    def test_reading_settings_is_reused_changing_them_is_not(self, stub):
        gm, gm_calls = self.counting({"units": []})
        change = {"changes": [{"unit": "Gen_bus1", "dispatchable": False}]}
        stub([ModelResponse(tool_calls=[ToolCall("set_generator_model", {})]),
              ModelResponse(tool_calls=[ToolCall("set_generator_model", {})]),
              ModelResponse(tool_calls=[ToolCall("set_generator_model", change)]),
              ModelResponse(tool_calls=[ToolCall("set_generator_model", {})]),
              ModelResponse(text="done")], tools={"set_generator_model": gm})
        loop_module.run_agent_turn("go", [])
        assert gm_calls == [{}, change, {}]              # read, (reused read), change, read again

    def test_a_repeated_failure_is_not_retried(self, stub):
        tool, calls = self.counting({"error": "refused", "status_code": 400, "endpoint": "/api/x"})
        stub([ModelResponse(tool_calls=[ToolCall("edit", {"x": 1})]),
              ModelResponse(tool_calls=[ToolCall("edit", {"x": 1})]),
              ModelResponse(text="done")], tools={"edit": tool})
        _, history = loop_module.run_agent_turn("go", [])
        assert len(calls) == 1
        second = [m for m in history if m["role"] == "tool"][1]["content"]
        assert second["error"] == "refused" and "already failed" in second["note"]

    def test_repeating_a_failure_after_being_told_stops_the_question(self, stub, monkeypatch):
        # Seen live: a model resent the same refused call twelve times.
        records = []
        tool, calls = self.counting({"error": "refused", "status_code": 400, "endpoint": "/api/x"})
        provider = stub([ModelResponse(tool_calls=[ToolCall("edit", {"x": 1})]) for _ in range(3)]
                        + [ModelResponse(text="The change was refused; nothing was changed.")],
                        tools={"edit": tool})
        monkeypatch.setattr(loop_module, "_append_to_log", records.append)
        text, _ = loop_module.run_agent_turn("go", [])
        assert len(calls) == 1
        assert len(provider.calls) == 4         # ran, told it would fail again, repeated: then an answer
        assert "Do not call any more tools" in provider.calls[-1]["messages"][-1]["content"]
        assert text.startswith("The change was refused") and "kept repeating a call that had already failed" in text
        assert "\n\n*CONDUCTOR stopped" in text            # italic, not a bullet: no space after the asterisk
        assert records[-1]["status"] == "repeated_failure_summarised"

    def test_being_told_once_is_not_stopping(self, stub):
        tool, _ = self.counting({"error": "refused", "status_code": 400, "endpoint": "/api/x"})
        stub([ModelResponse(tool_calls=[ToolCall("edit", {"x": 1})]),
              ModelResponse(tool_calls=[ToolCall("edit", {"x": 1})]),
              ModelResponse(tool_calls=[ToolCall("edit", {"x": 2})]),
              ModelResponse(text="done")], tools={"edit": tool})
        text, _ = loop_module.run_agent_turn("go", [])
        assert text == "done"

    def test_a_failure_with_no_answer_from_the_backend_is_retried(self, stub):
        tool, calls = self.counting({"error": "timed out", "status_code": None, "endpoint": "/api/x"})
        stub([ModelResponse(tool_calls=[ToolCall("read", {"x": 1})]),
              ModelResponse(tool_calls=[ToolCall("read", {"x": 1})]),
              ModelResponse(text="done")], tools={"read": tool})
        loop_module.run_agent_turn("go", [])
        assert len(calls) == 2

    def test_reuse_is_per_question(self, stub):
        tool, calls = self.counting()
        stub([ModelResponse(tool_calls=[ToolCall("read", {"x": 1})]), ModelResponse(text="one"),
              ModelResponse(tool_calls=[ToolCall("read", {"x": 1})]), ModelResponse(text="two")],
             tools={"read": tool})
        _, history = loop_module.run_agent_turn("first", [])
        loop_module.run_agent_turn("second", history)
        assert len(calls) == 2

    def test_the_operator_sees_it(self, stub):
        tool, _ = self.counting()
        events = []
        stub([ModelResponse(tool_calls=[ToolCall("read", {"x": 1})]),
              ModelResponse(tool_calls=[ToolCall("read", {"x": 1})]),
              ModelResponse(text="done")], tools={"read": tool})
        loop_module.run_agent_turn("go", [], on_event=lambda e, d: events.append((e, d)))
        assert ("tool_reused", {"name": "read", "same_as": 1, "failed": False}) in events

    def test_the_record_and_the_audit_panel_mark_it(self, stub, monkeypatch):
        records = []
        tool, _ = self.counting()
        stub([ModelResponse(tool_calls=[ToolCall("read", {"x": 1})]),
              ModelResponse(tool_calls=[ToolCall("read", {"x": 1})]),
              ModelResponse(text="done")], tools={"read": tool})
        monkeypatch.setattr(loop_module, "_append_to_log", records.append)
        loop_module.run_agent_turn("go", [])
        record = records[-1]
        assert [c.get("reused_from") for c in record["tool_calls"]] == [None, 1]
        from llm_agent.agent import trace
        steps = trace._steps(record)
        assert steps[1].reused_from == 1 and steps[1].ok
        headline = trace.Trace(steps=steps, figures=(), concerns=(), limits=()).headline()
        assert headline.startswith("1 tool")                 # a reused step ran nothing


_ROUNDING = 0.0005          # durations are rounded to the millisecond


class TestProgress:
    """A long study reports how far it has got while it runs: the loop routes
    it to the live status, and the hook is gone once the tool returns."""

    def test_progress_reaches_the_events_and_the_hook_is_cleared(self, stub):
        from llm_agent.agent import tools as tools_module

        def long_study(**kwargs):
            tools_module._say_progress("3 of 9 moments done")
            return {"value": 1}

        stub([ModelResponse(tool_calls=[ToolCall("long", {"x": 1})]), ModelResponse(text="done")],
             tools={"long": long_study})
        events = []
        loop_module.run_agent_turn("go", [], on_event=lambda e, d: events.append((e, d)))
        assert ("tool_progress", {"name": "long", "text": "3 of 9 moments done"}) in events
        assert [e for e, _ in events].index("tool_progress") < [e for e, _ in events].index("tool_done")
        assert tools_module.progress_hook is None

    def test_without_a_listener_progress_is_silent(self):
        from llm_agent.agent import tools as tools_module
        tools_module._say_progress("nobody listens")                 # no hook: nothing, no error


class TestTiming:
    """Each answer records where its time went: the model, the studies, the checks.
    Timed with perf_counter: on Windows before Python 3.13, monotonic ticks every
    15.6 ms and this test failed there (a 50 ms study read as 46.9 ms)."""

    def test_the_record_breaks_the_time_down(self, stub, monkeypatch):
        import time as _time
        records = []

        def slow(**kwargs):
            _time.sleep(0.05)
            return {"value": 1}

        stub([ModelResponse(tool_calls=[ToolCall("read", {"x": 1})]),
              ModelResponse(text="done")], tools={"read": slow})
        monkeypatch.setattr(loop_module, "_append_to_log", records.append)
        events = []
        loop_module.run_agent_turn("go", [], on_event=lambda e, d: events.append((e, d)))
        timing = records[-1]["timing"]
        assert set(timing) == {"total_s", "model_s", "studies_s", "checks_s", "model_calls",
                               "waited_for_rate_limit"}
        # Each figure is rounded to the millisecond on its own: the parts may
        # outgrow the total by that rounding, never by more.
        assert timing["model_calls"] == 2 and timing["studies_s"] >= 0.05 - _ROUNDING
        assert timing["total_s"] >= timing["studies_s"] + timing["model_s"] - 2 * _ROUNDING
        assert records[-1]["tool_calls"][0]["duration_s"] >= 0.05 - _ROUNDING
        done = [d for e, d in events if e == "tool_done"][0]
        assert done["duration_s"] >= 0.05 - _ROUNDING

    def test_a_reused_or_refused_call_has_no_duration(self, stub, monkeypatch):
        records = []
        stub([ModelResponse(tool_calls=[ToolCall("read", {"x": 1})]),
              ModelResponse(tool_calls=[ToolCall("read", {"x": 1}), ToolCall("nope")]),
              ModelResponse(text="done")], tools={"read": lambda **k: {"v": 1}})
        monkeypatch.setattr(loop_module, "_append_to_log", records.append)
        loop_module.run_agent_turn("go", [])
        calls = records[-1]["tool_calls"]
        assert "duration_s" in calls[0] and "duration_s" not in calls[1] and "duration_s" not in calls[2]


def test_a_scan_moves_the_clock_so_a_later_read_runs_again(stub):
    reads = []
    stub([ModelResponse(tool_calls=[ToolCall("get_current_conditions", {"x": 1})]),
          ModelResponse(tool_calls=[ToolCall("scan_rsa_over_time", {"n_steps": 2})]),
          ModelResponse(tool_calls=[ToolCall("get_current_conditions", {"x": 1})]),
          ModelResponse(text="done")],
         tools={"get_current_conditions": lambda **k: reads.append(k) or {"v": len(reads)},
                "scan_rsa_over_time": lambda **k: {"timestamps": ["a", "b"]}})
    loop_module.run_agent_turn("go", [])
    assert len(reads) == 2


class TestComparisonsAreNotInherited:
    """The loop carries the network and the dataset across a question's calls,
    except the one the question compares: the side the model leaves empty is
    then the base (or the measurements), not the proposal used before it."""

    def _run(self, stub, prompt, first, second):
        seen = []
        stub([ModelResponse(tool_calls=[ToolCall("run_rsa", first)]),
              ModelResponse(tool_calls=[ToolCall("simulate_all_contingencies", second)]),
              ModelResponse(text="done")],
             tools={"run_rsa": lambda **k: seen.append(k) or {"ok": 1},
                    "simulate_all_contingencies": lambda **k: seen.append(k) or {"ok": 2}})
        loop_module.run_agent_turn(prompt, [])
        return seen

    def test_a_proposal_is_carried_within_a_question_about_it(self, stub):
        seen = self._run(stub, "Run an N-1 screen on proposal dc_bus9.", {"network": "dc_bus9"}, {})
        assert seen[1]["network"] == "dc_bus9"

    def test_the_base_is_the_base_when_the_question_compares(self, stub):
        seen = self._run(stub, "Would the data centre proposal overload anything compared with today?",
                         {"network": "dc_bus9"}, {})
        assert "network" not in seen[1]

    def test_the_measurements_when_the_question_compares_with_the_forecast(self, stub):
        seen = self._run(stub, "How does the forecast compare with what was measured?",
                         {"data_source": "forecasts"}, {})
        assert "data_source" not in seen[1]

    def test_a_power_factor_is_carried_within_a_question_that_sets_it(self, stub):
        seen = self._run(stub, "With loads at pf 0.9, is it secure, and under N-1?",
                         {"load_power_factor": 0.9}, {})
        assert seen[1]["load_power_factor"] == 0.9


class TestParameterWarningsReachTheResult:
    """An unusual value runs, and the model and the audit panel are told."""

    def test_the_warning_is_attached_to_the_result_and_logged(self, stub, monkeypatch):
        records = []
        stub([ModelResponse(tool_calls=[ToolCall("run_rsa", {"vm_upper_pu": 1.25})]),
              ModelResponse(text="done")], tools={"run_rsa": lambda **k: {"ok": True}})
        monkeypatch.setattr(loop_module, "_append_to_log", records.append)
        _, history = loop_module.run_agent_turn("go", [])

        sent = history[2]["content"]
        warnings = [w for w in sent["_integrity_warnings"] if w["code"] == "parameter_warning"]
        assert warnings and warnings[0]["field"] == "vm_upper_pu" and sent["ok"] is True
        logged = records[-1]["tool_results"][0]["result"]
        assert logged["_integrity_warnings"] == sent["_integrity_warnings"]

        from llm_agent.agent import trace
        assert any("vm_upper_pu=1.25" in c for c in trace.build(records[-1]).concerns)

    def test_the_operator_is_told_while_it_runs(self, stub):
        """Its own event: the app says "ran with unusual parameters", not that
        the result contradicted itself."""
        stub([ModelResponse(tool_calls=[ToolCall("run_rsa", {"vm_upper_pu": 1.25})]),
              ModelResponse(text="done")], tools={"run_rsa": lambda **k: {"ok": True}})
        events = []
        loop_module.run_agent_turn("go", [], on_event=lambda e, d: events.append((e, d)))
        names = [e for e, _ in events]
        assert "parameter_warning" in names and "integrity_warning" not in names
        problems = next(d["problems"] for e, d in events if e == "parameter_warning")
        assert any("vm_upper_pu=1.25" in p for p in problems)

    def test_a_clean_call_carries_none(self, stub):
        stub([ModelResponse(tool_calls=[ToolCall("run_rsa", {"vm_upper_pu": 1.05})]),
              ModelResponse(text="done")], tools={"run_rsa": lambda **k: {"ok": True}})
        _, history = loop_module.run_agent_turn("go", [])
        assert "_integrity_warnings" not in history[2]["content"]


class TestFiguresTheLoopComputedAreEvidence:
    """The model is sent a scan's summary (steps, violating steps) that the raw
    result does not hold; quoting it was reported as a figure from nowhere."""

    SCAN = {"timestamps": ["2000-01-09 18:00:00", "2000-01-09 18:15:00", "2000-01-09 18:30:00",
                           "2000-01-09 18:45:00", "2000-01-09 19:00:00", "2000-01-09 19:15:00"],
            "violation_counts": [0, 2, 5, 0, 0, 1], "min_voltage": [0.97] * 6, "max_voltage": [1.04] * 6,
            "max_line_loading": [55.0] * 6, "max_trafo_loading": [40.0] * 6, "any_violations": True}

    def test_the_scan_summary_is_logged_and_traces(self, stub, monkeypatch):
        from llm_agent.agent.provenance import check_turn
        records = []
        stub([ModelResponse(tool_calls=[ToolCall("scan_rsa_over_time", {"n_steps": 6})]),
              ModelResponse(text="Violations at 3 of the 6 steps scanned.")],
             tools={"scan_rsa_over_time": lambda **k: dict(self.SCAN)})
        monkeypatch.setattr(loop_module, "_append_to_log", records.append)
        loop_module.run_agent_turn("scan", [])

        logged = records[-1]["tool_results"][0]["result"]
        assert logged["summary_for_model"]["violating_step_count"] == 3
        assert logged["summary_for_model"]["n_steps"] == 6
        assert logged["violation_counts"] == self.SCAN["violation_counts"]      # the raw result is kept
        assert [v.status for v in check_turn(records[-1]).verdicts] == ["grounded", "grounded"]

    def test_a_series_length_is_logged(self, stub, monkeypatch):
        records = []
        series = {"timestamps": ["2000-01-09 18:00:00"] * 7, "series": {"vm_pu": [1.0] * 7}}
        stub([ModelResponse(tool_calls=[ToolCall("get_element_timeseries", {})]),
              ModelResponse(text="done")], tools={"get_element_timeseries": lambda **k: series})
        monkeypatch.setattr(loop_module, "_append_to_log", records.append)
        loop_module.run_agent_turn("series", [])
        assert records[-1]["tool_results"][0]["result"]["summary_for_model"]["n_points"] == 7

    def test_a_result_sent_as_is_is_logged_as_is(self, stub, monkeypatch):
        records = []
        stub([ModelResponse(tool_calls=[ToolCall("run_rsa", {})]), ModelResponse(text="done")],
             tools={"run_rsa": lambda **k: {"ok": True}})
        monkeypatch.setattr(loop_module, "_append_to_log", records.append)
        loop_module.run_agent_turn("go", [])
        assert records[-1]["tool_results"][0]["result"] == {"ok": True}

    def test_how_to_report_a_study_reaches_the_model_and_is_not_logged(self, stub, monkeypatch):
        """Moved out of the system prompt: paid for only when the study ran."""
        records = []
        provider = stub([ModelResponse(tool_calls=[ToolCall("compute_hosting_capacity", {})]),
                         ModelResponse(text="done")],
                        tools={"compute_hosting_capacity": lambda **k: {"hosting_capacity_mw": 12.5}})
        monkeypatch.setattr(loop_module, "_append_to_log", records.append)
        loop_module.run_agent_turn("hosting at bus 5?", [])
        sent = str(provider.calls[-1]["messages"][-1]["content"])
        assert "how_to_report" in sent and "q-mode glossary" in sent
        assert records[-1]["tool_results"][0]["result"] == {"hosting_capacity_mw": 12.5}
        # A failed study gets no note.
        assert "how_to_report" not in loop_module._prepare_tool_result_for_model(
            "compute_hosting_capacity", {"error": "no bus"})

    def test_a_zero_count_is_a_field_left_blank(self):
        from llm_agent.agent.validators import without_form_blanks
        assert without_form_blanks({"step_size": 0, "n_steps": 0, "study": "rsa", "load_scaling_factor": 0}) == {
            "study": "rsa", "load_scaling_factor": 0}

    def test_a_failed_scan_reaches_the_model_as_its_error(self):
        """A summary of the empty scan read as "no violations" and hid it."""
        failed = {"error": "backend unreachable", "partial_timestamps": []}
        assert loop_module._prepare_tool_result_for_model("scan_rsa_over_time", failed) is failed


def test_the_step_limit_note_is_not_a_figure_of_the_answer(stub, monkeypatch):
    """"CONDUCTOR used all 16 steps…" is the system speaking; its number is a
    setting, and was reported as an untraced figure."""
    from llm_agent.agent.provenance import check_turn
    records = []
    monkeypatch.setattr(loop_module, "MAX_AGENT_TURNS", 7)
    stub([ModelResponse(tool_calls=[ToolCall("t", {"i": i})]) for i in range(7)]
         + [ModelResponse(text="The grid had 2 violations.")],
         tools={"t": lambda **k: {"total_violations": 2}})
    monkeypatch.setattr(loop_module, "_append_to_log", records.append)
    text, _ = loop_module.run_agent_turn("loop", [])
    assert "all 7 steps" in text
    assert [(v.claim.value, v.status) for v in check_turn(records[-1]).verdicts] == [(2.0, "grounded")]


class TestTheAuditPanelFindsItsRecordByKey:
    """The record is matched by the `turn_key` the loop writes, not by the
    prompt: the same question asked twice must not get the first answer's
    evidence when the second turn aborted before logging."""

    def _logged(self, stub, monkeypatch):
        log = []
        monkeypatch.setattr(loop_module, "session_log", log)
        stub([ModelResponse(text="ok")])
        monkeypatch.setattr(loop_module, "_append_to_log", log.append)
        loop_module.run_agent_turn("Is it secure?", [], conversation_id="conv-key")
        return log

    def test_the_record_of_this_turn_is_found(self, stub, monkeypatch):
        log = self._logged(stub, monkeypatch)
        assert loop_module.latest_record_for("Is it secure?", "conv-key") is log[-1]

    def test_a_stale_record_for_the_same_prompt_is_not(self, stub, monkeypatch):
        self._logged(stub, monkeypatch)
        # The same question again, aborted before its record was written.
        monkeypatch.setitem(loop_module._attempt_counter, "conv-key",
                            loop_module._attempt_counter["conv-key"] + 1)
        assert loop_module.latest_record_for("Is it secure?", "conv-key") is None

    def test_another_conversation_is_not(self, stub, monkeypatch):
        self._logged(stub, monkeypatch)
        assert loop_module.latest_record_for("Is it secure?", "another") is None

    def test_a_record_without_a_key_is_matched_by_its_prompt(self, monkeypatch):
        monkeypatch.setattr(loop_module, "session_log", [{"user": "Is it secure?"}])
        assert loop_module.latest_record_for("Is it secure?", "c") == {"user": "Is it secure?"}
        assert loop_module.latest_record_for("Something else", "c") is None
        monkeypatch.setattr(loop_module, "session_log", [])
        assert loop_module.latest_record_for("Is it secure?", "c") is None


class TestDeepResearch:
    """Deep Research asks for depth for one question; the model still decides
    whether the question is an investigation, and the record says which."""

    def _run(self, stub, monkeypatch, responses, tools, deep):
        provider = stub(responses, tools=tools)
        records = []
        monkeypatch.setattr(loop_module, "_append_to_log", records.append)
        monkeypatch.setattr(loop_module, "MAX_AGENT_TURNS", 2)
        loop_module.run_agent_turn("why?", [], deep=deep, reading_check=False, reflection="off")
        return provider, records[-1]

    def test_the_note_and_the_steps_are_for_this_question_only(self, stub, monkeypatch):
        calls = [ModelResponse(tool_calls=[ToolCall("investigate", {})])] * 3 + [ModelResponse(text="done")]
        violated = {"status": "explained", "symptom": {"violated": True}}
        provider, record = self._run(stub, monkeypatch, calls, {"investigate": lambda **kw: violated}, deep=True)
        assert "Deep Research" in provider.calls[0]["system"]
        # Beyond the usual 2 steps, and one more: the reminder of what depth still lacks.
        assert len(provider.calls) == 5
        reminder = provider.calls[-1]["messages"][-1]["content"]
        assert "test_prevention" in reminder and "recurring" in reminder and "record_investigation" in reminder
        assert record["deep_research"] == {"requested": True, "investigated": True}

    def test_the_reminder_comes_once_and_leaves_no_trace(self, stub, monkeypatch):
        calls = [ModelResponse(tool_calls=[ToolCall("investigate", {})]), ModelResponse(text="draft"),
                 ModelResponse(text="final, and why prevention does not apply")]
        provider = stub(calls, tools={"investigate": lambda **kw: {"status": "explained"}})
        monkeypatch.setattr(loop_module, "_append_to_log", lambda r: None)
        text, history = loop_module.run_agent_turn("why?", [], deep=True, reading_check=False, reflection="off")
        assert text.startswith("final") and len(provider.calls) == 3
        assert not any("Deep Research, before you answer" in str(m.get("content")) for m in history)
        assert "draft" not in [m.get("content") for m in history]

    def test_the_reminder_asks_what_lies_behind_the_cause_until_one_is_recorded(self):
        point = ({"name": "investigate", "args": {}}, {"result": {"status": "explained"}})
        recurring = ({"name": "investigate", "args": {"recurring": True}}, {"result": {"mode": "recurring"}})

        def record(*verdicts):
            return ({"name": "record_investigation", "args": {}},
                    {"result": {"status": "recorded", "hypotheses": [{"verdict": v} for v in verdicts]}})

        def missing(*pairs):
            return loop_module._deep_missing([c for c, _ in pairs], [e for _, e in pairs])
        only_tested = missing(point, recurring, record("supported", "refuted"))
        assert len(only_tested) == 1 and "analysis of the grid" in only_tested[0] and "record again" in only_tested[0]
        # Data alone (why demand rose) is not the analysis that would decide the cause.
        assert len(missing(point, recurring, record("supported", "untested"))) == 1
        analysis = ({"name": "record_investigation", "args": {}}, {"result": {"status": "recorded", "hypotheses": [
            {"verdict": "supported"}, {"verdict": "untested", "missing_kind": "analysis"}]}})
        assert missing(point, recurring, analysis) == []
        assert any("analysis of the grid" in m and "record again" not in m for m in missing(point, recurring))

    def test_a_question_that_is_not_a_why_is_answered_as_an_analysis(self, stub, monkeypatch):
        calls = [ModelResponse(tool_calls=[ToolCall("run_rsa", {})]), ModelResponse(text="Answered as an analysis.")]
        _, record = self._run(stub, monkeypatch, calls, {"run_rsa": lambda **kw: {"secure": True}}, deep=True)
        assert record["deep_research"] == {"requested": True, "investigated": False}

    def test_without_it_nothing_changes(self, stub, monkeypatch):
        provider, record = self._run(stub, monkeypatch, [ModelResponse(text="ok")], {}, deep=False)
        assert "Deep Research" not in provider.calls[0]["system"] and "deep_research" not in record
