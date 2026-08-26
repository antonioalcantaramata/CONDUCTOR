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
        assert len(provider.calls) == 3
        assert "maximum of 3 turns" in text

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
