"""Anthropic provider: preflight, request shaping, and response normalisation.

Two shapes are unique to this backend and are where the tests concentrate.

Tool results are `user` messages carrying `tool_result` blocks rather than a
role of their own, and consecutive results must be *merged* into one message —
the API rejects two user messages in a row, and splitting parallel results also
trains models to stop calling in parallel.

Thinking blocks have to survive the round trip. A tool call made after thinking
must be replayed with the thinking that produced it, and the neutral history in
`base.py` carries only text and tool calls — so the block rides in the
ToolCall's `signature` slot, the same one Gemini uses for its thought
signature.

No test contacts a real API: the SDK client is replaced with a stub.
"""

import json
import pathlib
import types

import pytest

from llm_agent.agent.providers.base import ToolCall
from llm_agent.agent.providers.claude import (
    AnthropicProvider,
    _anthropic_tools,
    _stringify,
)


@pytest.fixture
def provider(monkeypatch):
    """A provider with fixed settings, insulated from the developer's .env."""
    for var in ("ANTHROPIC_MODEL", "ANTHROPIC_EFFORT", "ANTHROPIC_THINKING",
                "ANTHROPIC_TIMEOUT_S", "ANTHROPIC_MAX_TOKENS"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    return AnthropicProvider(model="claude-test")


def _block(**fields):
    """A stand-in for an SDK content block: attribute access plus model_dump."""
    obj = types.SimpleNamespace(**fields)
    obj.model_dump = lambda exclude_none=True: dict(fields)
    return obj


def _reply(content, stop_reason="end_turn", stop_details=None):
    return types.SimpleNamespace(
        content=content, stop_reason=stop_reason, stop_details=stop_details
    )


class TestPreflight:
    def test_a_missing_key_is_caught_before_the_request(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
            AnthropicProvider(model="claude-test").preflight()

    def test_a_key_passes(self, provider):
        provider.preflight()

    def test_the_message_says_what_to_do(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.raises(RuntimeError) as exc:
            AnthropicProvider(model="claude-test").preflight()
        assert "console.anthropic.com" in str(exc.value)


class TestToolConversion:
    def test_schemas_are_flattened_into_anthropic_shape(self):
        from llm_agent.agent.tool_schemas import TOOLS

        converted = _anthropic_tools(TOOLS)
        assert converted, "the shared tool definitions produced nothing"
        for tool in converted:
            assert set(tool) == {"name", "description", "input_schema"}
            assert tool["input_schema"]["type"] == "object"

    def test_every_shared_tool_survives(self):
        """One source of truth. A tool that converts for OpenAI and vanishes
        here is a backend that silently cannot do part of the job."""
        from llm_agent.agent.providers.schema import to_json_schema_tools
        from llm_agent.agent.tool_schemas import TOOLS

        assert (
            {t["name"] for t in _anthropic_tools(TOOLS)}
            == {t["function"]["name"] for t in to_json_schema_tools(TOOLS)}
        )


class TestStringify:
    def test_a_string_passes_through(self):
        assert _stringify("already text") == "already text"

    def test_a_dict_becomes_json(self):
        assert json.loads(_stringify({"vm_pu": 1.049})) == {"vm_pu": 1.049}

    def test_an_unserialisable_value_costs_that_value_not_the_turn(self):
        """Tool results carry numpy scalars and datetimes. Failing the whole
        turn over one unserialisable corner would be the wrong trade."""
        out = _stringify({"when": object()})
        assert "when" in out


class TestMessageShaping:
    def test_a_tool_result_becomes_a_user_message(self, provider):
        out = provider._to_anthropic_messages([
            {"role": "user", "content": "status?"},
            {"role": "assistant", "content": "",
             "tool_calls": [{"name": "run_rsa", "args": {}, "id": "t1"}]},
            {"role": "tool", "name": "run_rsa", "content": {"ok": True}, "id": "t1"},
        ])
        assert out[-1]["role"] == "user"
        assert out[-1]["content"][0]["type"] == "tool_result"
        assert out[-1]["content"][0]["tool_use_id"] == "t1"

    def test_parallel_results_merge_into_one_message(self, provider):
        """Two user messages in a row are rejected, and splitting parallel
        results also trains models out of calling in parallel."""
        out = provider._to_anthropic_messages([
            {"role": "user", "content": "status?"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"name": "a", "args": {}, "id": "t1"},
                {"name": "b", "args": {}, "id": "t2"},
            ]},
            {"role": "tool", "name": "a", "content": {"x": 1}, "id": "t1"},
            {"role": "tool", "name": "b", "content": {"y": 2}, "id": "t2"},
        ])
        roles = [m["role"] for m in out]
        assert roles == ["user", "assistant", "user"], roles
        assert len(out[-1]["content"]) == 2

    def test_an_empty_assistant_turn_is_dropped(self, provider):
        """Neither text nor tool calls is not a turn the API accepts, and it
        carried nothing the model needs to see."""
        out = provider._to_anthropic_messages([
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": ""},
        ])
        assert [m["role"] for m in out] == ["user"]

    def test_a_tool_call_replays_with_its_id(self, provider):
        out = provider._to_anthropic_messages([
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": "running",
             "tool_calls": [{"name": "run_rsa", "args": {"a": 1}, "id": "t9"}]},
        ])
        blocks = out[-1]["content"]
        assert blocks[0]["type"] == "text"
        assert blocks[1] == {"type": "tool_use", "id": "t9",
                             "name": "run_rsa", "input": {"a": 1}}


class TestThinkingRoundTrip:
    THINK = json.dumps({"type": "thinking", "thinking": "because…",
                        "signature": "abc"})

    def test_a_stashed_thinking_block_is_replayed_first(self, provider):
        """A tool call made after thinking must carry that thinking back, or
        the model sees its own call with no reasoning behind it."""
        out = provider._to_anthropic_messages([
            {"role": "user", "content": "why?"},
            {"role": "assistant", "content": "here", "tool_calls": [
                {"name": "attr", "args": {}, "id": "t1", "signature": self.THINK},
            ]},
        ])
        blocks = out[-1]["content"]
        assert blocks[0]["type"] == "thinking"
        assert [b["type"] for b in blocks] == ["thinking", "text", "tool_use"]

    def test_one_thinking_block_for_parallel_calls(self, provider):
        """The block is stamped on every call in the turn; replaying it once
        per call would send the same reasoning several times."""
        out = provider._to_anthropic_messages([
            {"role": "user", "content": "why?"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"name": "a", "args": {}, "id": "t1", "signature": self.THINK},
                {"name": "b", "args": {}, "id": "t2", "signature": self.THINK},
            ]},
        ])
        kinds = [b["type"] for b in out[-1]["content"]]
        assert kinds.count("thinking") == 1
        assert kinds.count("tool_use") == 2

    def test_an_unparseable_signature_does_not_cost_the_turn(self, provider):
        """Losing the reasoning is what happens on every other backend anyway;
        losing the call is not."""
        out = provider._to_anthropic_messages([
            {"role": "user", "content": "why?"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"name": "a", "args": {}, "id": "t1", "signature": "not json"},
            ]},
        ])
        assert [b["type"] for b in out[-1]["content"]] == ["tool_use"]


class TestResponseShaping:
    def test_text_only(self, provider):
        out = provider._to_model_response(_reply([_block(type="text", text="All secure.")]))
        assert out.text == "All secure."
        assert not out.wants_tools

    def test_tool_calls_are_extracted(self, provider):
        out = provider._to_model_response(_reply([
            _block(type="text", text="checking"),
            _block(type="tool_use", id="t1", name="run_rsa", input={"a": 1}),
        ]))
        assert out.text == "checking"
        assert out.tool_calls == [ToolCall(name="run_rsa", args={"a": 1}, id="t1")]

    def test_a_thinking_block_rides_on_the_tool_call(self, provider):
        out = provider._to_model_response(_reply([
            _block(type="thinking", thinking="because…", signature="sig"),
            _block(type="tool_use", id="t1", name="run_rsa", input={}),
        ]))
        assert out.tool_calls[0].signature
        assert json.loads(out.tool_calls[0].signature)["type"] == "thinking"

    def test_a_refusal_becomes_an_answer_not_an_exception(self, provider):
        """It is a reply the operator should see, and the turn ends cleanly."""
        out = provider._to_model_response(_reply(
            [], stop_reason="refusal",
            stop_details=types.SimpleNamespace(category="cyber"),
        ))
        assert "declined" in out.text
        assert "cyber" in out.text
        assert not out.wants_tools


class TestRequestPayload:
    def _capture(self, provider, monkeypatch):
        sent = {}

        def fake_create(**kwargs):
            sent.update(kwargs)
            return _reply([_block(type="text", text="ok")])

        monkeypatch.setattr(
            type(provider), "client",
            property(lambda self: types.SimpleNamespace(
                messages=types.SimpleNamespace(create=fake_create))),
        )
        provider.chat([{"role": "user", "content": "hi"}], [], "SYSTEM")
        return sent

    def test_adaptive_thinking_and_effort_are_sent(self, provider, monkeypatch):
        sent = self._capture(provider, monkeypatch)
        assert sent["thinking"] == {"type": "adaptive"}
        assert sent["output_config"] == {"effort": "medium"}
        assert "budget_tokens" not in json.dumps(sent), "removed on current models"

    def test_thinking_off_is_only_sent_where_it_is_accepted(self, provider, monkeypatch):
        """The API rejects `disabled` above effort `high`, so the request omits
        `thinking` there rather than failing the turn."""
        provider.thinking = False
        provider.effort = "max"
        assert "thinking" not in self._capture(provider, monkeypatch)

        provider.effort = "low"
        assert self._capture(provider, monkeypatch)["thinking"] == {"type": "disabled"}

    def test_an_unknown_effort_is_dropped_not_sent(self, provider, monkeypatch):
        """An unknown level is a 400. Losing the setting beats losing the turn."""
        provider.effort = "turbo"
        assert "output_config" not in self._capture(provider, monkeypatch)


class TestDescribe:
    def test_effort_and_thinking_reach_the_session_log(self, provider):
        assert provider.describe()["reasoning"] == "medium"
        provider.thinking = False
        assert "thinking off" in provider.describe()["reasoning"]


class TestRegistry:
    def test_the_provider_is_selectable(self):
        from llm_agent.agent.providers import available_providers, get_provider

        assert "anthropic" in available_providers()
        assert get_provider("anthropic").name == "anthropic"

    def test_it_satisfies_the_loop_contract(self, provider):
        from llm_agent.agent.providers.base import LLMProvider

        assert isinstance(provider, LLMProvider)


class TestLegacyThinkingModels:
    """Haiku 4.5 and the other 4.5-generation models take a different API.

    They predate adaptive thinking: `{"type": "enabled", "budget_tokens": N}`
    rather than `{"type": "adaptive"}`, and `output_config` is rejected
    outright. Sending the wrong shape is a 400 on every request, not a degraded
    answer — so this is the one place the provider must branch on the model.
    """

    @pytest.fixture
    def haiku(self, monkeypatch):
        for var in ("ANTHROPIC_EFFORT", "ANTHROPIC_THINKING", "ANTHROPIC_MAX_TOKENS"):
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        return AnthropicProvider(model="claude-haiku-4-5")

    def test_the_shipped_default_is_recognised_as_legacy(self):
        """If this fails, the shipped default sends a shape its own model
        rejects — a 400 on the very first question.

        Reads the literal out of the source rather than importing the constant:
        `config` resolves the developer's own `.env` first, so importing it
        would test whichever model this machine happens to be pinned to."""
        import re

        src = pathlib.Path("llm_agent/agent/config.py").read_text()
        default = re.search(
            r'ANTHROPIC_MODEL[^=]*= os\.environ\.get\("ANTHROPIC_MODEL", "([^"]+)"\)', src
        ).group(1)
        assert AnthropicProvider(model=default).legacy_thinking, default

    @pytest.mark.parametrize("model,legacy", [
        ("claude-haiku-4-5", True),
        ("claude-haiku-4-5-20251001", True),   # dated snapshot, same model
        ("claude-sonnet-4-5", True),
        ("claude-sonnet-5", False),
        ("claude-opus-5", False),
    ])
    def test_model_families_are_told_apart(self, model, legacy):
        assert AnthropicProvider(model=model).legacy_thinking is legacy

    def test_a_budget_is_sent_and_output_config_is_not(self, haiku):
        payload = haiku._thinking_payload()
        assert payload["thinking"]["type"] == "enabled"
        assert payload["thinking"]["budget_tokens"] >= 1024
        assert "output_config" not in payload, "rejected by these models"

    def test_effort_maps_onto_the_budget(self, haiku):
        """`ANTHROPIC_EFFORT` stays the single knob in both worlds."""
        budgets = {}
        for effort in ("low", "medium", "high"):
            haiku.effort = effort
            budgets[effort] = haiku._thinking_payload()["thinking"]["budget_tokens"]
        assert budgets["low"] < budgets["medium"] < budgets["high"]

    def test_the_budget_leaves_room_for_an_answer(self, haiku):
        """A model that spends its whole output allowance thinking has nothing
        left to say — and the API rejects a budget at or above max_tokens."""
        haiku.max_tokens = 2000
        haiku.effort = "max"
        budget = haiku._thinking_payload()["thinking"]["budget_tokens"]
        assert 1024 <= budget < haiku.max_tokens

    def test_thinking_off_sends_nothing_at_all(self, haiku):
        """These models have no `disabled` type — omitting the field is off."""
        haiku.thinking = False
        assert haiku._thinking_payload() == {}

    def test_a_current_model_still_gets_the_modern_shape(self, provider):
        payload = provider._thinking_payload()
        assert payload["thinking"] == {"type": "adaptive"}
        assert payload["output_config"] == {"effort": "medium"}

    def test_the_log_records_the_budget_not_the_label(self, haiku):
        """On a legacy model `medium` is a token count this provider chose. A
        log that recorded only the label could not be compared with anything."""
        assert haiku.describe()["reasoning"].startswith("budget ")
