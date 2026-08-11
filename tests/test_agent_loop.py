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
