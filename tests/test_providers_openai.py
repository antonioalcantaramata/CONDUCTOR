"""OpenAI provider: preflight, error classification, and message conversion.

The id pairing is the important part. OpenAI matches a tool result to its call
by `tool_call_id`, while the neutral history pairs them by order and only
carries an id when the producing provider set one — Ollama never does. A
conversation started on one backend and continued on this one would otherwise
be rejected wholesale, so every test here that asserts on ids is protecting
against that.

No test contacts a real API: httpx is monkeypatched, or the base URL points at
a closed port.
"""

import json

import httpx
import pytest

from llm_agent.agent.providers.base import ModelResponse
from llm_agent.agent.providers.openai import (
    OpenAIProvider,
    _decode_reasoning,
    _encode_reasoning,
    _responses_tools,
    probe,
)

CLOSED = "http://localhost:59998/v1"


@pytest.fixture
def provider(monkeypatch):
    """A provider with fixed settings, insulated from the developer's .env."""
    for var in ("OPENAI_MODEL", "OPENAI_BASE_URL", "OPENAI_TIMEOUT_S"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    return OpenAIProvider(model="test-model", base_url=CLOSED)


def _response(status, payload=None, headers=None, text=""):
    return httpx.Response(
        status_code=status,
        json=payload if payload is not None else None,
        headers=headers or {},
        text=None if payload is not None else text,
        request=httpx.Request("POST", "http://x/v1/chat/completions"),
    )


class TestSettingsAreLive:
    def test_reads_settings_from_the_environment(self, monkeypatch):
        monkeypatch.setenv("OPENAI_MODEL", "m2")
        monkeypatch.setenv("OPENAI_BASE_URL", "https://example.test/v1/")
        monkeypatch.setenv("OPENAI_TIMEOUT_S", "42")
        p = OpenAIProvider()
        # Trailing slash is stripped so URLs never end up doubled.
        assert (p.model, p.base_url, p.timeout_s) == (
            "m2", "https://example.test/v1", 42.0
        )

    def test_explicit_arguments_beat_the_environment(self, monkeypatch):
        monkeypatch.setenv("OPENAI_MODEL", "from-env")
        assert OpenAIProvider(model="explicit").model == "explicit"

    def test_api_key_is_read_late(self, monkeypatch):
        """The settings panel can set a key mid-session; a snapshot would miss it."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        p = OpenAIProvider()
        monkeypatch.setenv("OPENAI_API_KEY", "sk-later")
        assert p.api_key == "sk-later"


class TestPreflight:
    def test_missing_key_is_refused_with_a_usable_message(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
            OpenAIProvider(model="m", base_url=CLOSED).preflight()

    def test_blank_key_counts_as_missing(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "   ")
        with pytest.raises(RuntimeError):
            OpenAIProvider(model="m", base_url=CLOSED).preflight()

    def test_passes_when_a_key_is_present(self, provider):
        provider.preflight()  # must not raise


class TestReasoningEffort:
    """`reasoning_effort` is accepted by reasoning models and refused by others,
    so it has to be both configurable and droppable."""

    def test_defaults_to_medium(self, monkeypatch):
        monkeypatch.delenv("OPENAI_REASONING_EFFORT", raising=False)
        assert OpenAIProvider().reasoning_effort == "medium"

    @pytest.mark.parametrize("level", ["none", "low", "medium", "high", "xhigh"])
    def test_every_documented_level_is_accepted(self, monkeypatch, level):
        monkeypatch.setenv("OPENAI_REASONING_EFFORT", level)
        assert OpenAIProvider().reasoning_effort == level

    def test_value_is_normalised(self, monkeypatch):
        monkeypatch.setenv("OPENAI_REASONING_EFFORT", "  XHigh  ")
        assert OpenAIProvider().reasoning_effort == "xhigh"

    def test_empty_means_send_nothing_not_fall_back_to_default(self, monkeypatch):
        """A blank setting is how a non-reasoning model is configured; treating
        it as unset would put `medium` back and break that model."""
        monkeypatch.setenv("OPENAI_REASONING_EFFORT", "")
        assert OpenAIProvider().reasoning_effort == ""

    def test_effort_is_sent_in_the_payload(self, provider, monkeypatch):
        monkeypatch.setenv("OPENAI_REASONING_EFFORT", "high")
        p = OpenAIProvider(model="m", base_url=CLOSED)
        sent = {}

        def fake_post(url, json=None, headers=None, timeout=None):
            sent.update(json)
            return _response(200, {"choices": [{"message": {"content": "ok"}}]})

        monkeypatch.setattr(httpx, "post", fake_post)
        p.chat([], [], "S")
        assert sent["reasoning_effort"] == "high"

    def test_blank_effort_omits_the_parameter(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("OPENAI_REASONING_EFFORT", "")
        p = OpenAIProvider(model="m", base_url=CLOSED)
        sent = {}

        def fake_post(url, json=None, headers=None, timeout=None):
            sent.update(json)
            return _response(200, {"choices": [{"message": {"content": "ok"}}]})

        monkeypatch.setattr(httpx, "post", fake_post)
        p.chat([], [], "S")
        assert "reasoning_effort" not in sent


class TestDroppingRejectedParameters:
    """A model that refuses a knob should cost a retry, not the whole turn."""

    def test_named_param_is_dropped(self, provider):
        payload = {"temperature": 0, "reasoning_effort": "high"}
        dropped = provider._adjust_rejected_param(payload, _response(400, {"error": {
            "param": "reasoning_effort",
            "message": "Unsupported parameter: 'reasoning_effort'.",
        }}))
        assert dropped == "dropped reasoning_effort"
        assert "reasoning_effort" not in payload
        assert payload["temperature"] == 0, "only the named param may go"

    def test_falls_back_to_the_message_when_param_is_absent(self, provider):
        """Compatible servers word their errors however they like."""
        payload = {"temperature": 0}
        dropped = provider._adjust_rejected_param(payload, _response(400, {"error": {
            "message": "temperature is not supported with this model",
        }}))
        assert dropped == "dropped temperature"
        assert payload == {}

    def test_unrelated_400_drops_nothing(self, provider):
        payload = {"temperature": 0, "reasoning_effort": "low"}
        assert provider._adjust_rejected_param(payload, _response(400, {"error": {
            "code": "context_length_exceeded", "message": "Too long."}})) is None
        assert payload == {"temperature": 0, "reasoning_effort": "low"}

    def test_non_400_drops_nothing(self, provider):
        payload = {"temperature": 0}
        assert provider._adjust_rejected_param(
            payload, _response(429, {"error": {"param": "temperature"}})
        ) is None

    def test_tools_plus_reasoning_becomes_none_not_absent(self, provider):
        """The regression this class exists for.

        gpt-5.6-luna refuses function tools with any reasoning effort on
        /v1/chat/completions and asks for the literal 'none'. Dropping the
        parameter instead lets the model apply its own default effort and fail
        again identically, with nothing left to adjust.
        """
        payload = {"tools": [{}], "reasoning_effort": "medium", "temperature": 0}
        action = provider._adjust_rejected_param(payload, _response(400, {"error": {
            "message": "Function tools with reasoning_effort are not supported for "
                       "gpt-5.6-luna in /v1/chat/completions. To use function "
                       "tools, use /v1/responses or set reasoning_effort to 'none'.",
            "type": "invalid_request_error", "param": None, "code": None,
        }}))
        assert action == "reasoning_effort=none"
        assert payload["reasoning_effort"] == "none", "must be set, not removed"

    def test_already_none_falls_through_so_the_loop_terminates(self, provider):
        payload = {"reasoning_effort": "none"}
        action = provider._adjust_rejected_param(payload, _response(400, {"error": {
            "message": "reasoning_effort must be 'none'."}}))
        # Second time round the rule must not fire again, or this never ends.
        assert action != "reasoning_effort=none"

    def test_turn_completes_against_a_server_enforcing_that_rule(self, monkeypatch):
        """End to end: the user gets an answer instead of the 400 they saw."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("OPENAI_REASONING_EFFORT", "medium")
        p = OpenAIProvider(model="gpt-5.6-luna", base_url=CLOSED)
        seen = []

        def fake_post(url, json=None, headers=None, timeout=None):
            seen.append(json.get("reasoning_effort", "<absent>"))
            if json.get("tools") and json.get("reasoning_effort") != "none":
                return _response(400, {"error": {"message":
                    "Function tools with reasoning_effort are not supported for "
                    "gpt-5.6-luna in /v1/chat/completions. To use function tools, "
                    "use /v1/responses or set reasoning_effort to 'none'."}})
            return _response(200, {"choices": [{"message": {"content": "done"}}]})

        from llm_agent.agent.tool_schemas import TOOLS

        monkeypatch.setattr(httpx, "post", fake_post)
        assert p.chat([], TOOLS, "S").text == "done"
        assert seen == ["medium", "none"], "should settle in exactly one retry"

    def test_request_succeeds_after_the_model_refuses_the_knob(self, monkeypatch):
        """The behaviour that actually matters: the turn still completes."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("OPENAI_REASONING_EFFORT", "xhigh")
        p = OpenAIProvider(model="m", base_url=CLOSED)
        seen = []

        def fake_post(url, json=None, headers=None, timeout=None):
            seen.append(dict(json))
            if "reasoning_effort" in json:
                return _response(400, {"error": {
                    "param": "reasoning_effort",
                    "message": "Unsupported parameter: 'reasoning_effort'.",
                }})
            return _response(200, {"choices": [{"message": {"content": "done"}}]})

        monkeypatch.setattr(httpx, "post", fake_post)
        assert p.chat([], [], "S").text == "done"
        assert len(seen) == 2
        assert "reasoning_effort" in seen[0] and "reasoning_effort" not in seen[1]


class TestErrorClassification:
    """A quota failure and a rate limit both arrive as 429 and must not be
    treated alike: retrying the former burns ten attempts to learn nothing."""

    def test_quota_exhaustion_is_permanent(self, provider):
        msg, retryable, _ = provider._describe_http_error(
            _response(429, {"error": {"code": "insufficient_quota",
                                      "message": "You exceeded your quota."}})
        )
        assert retryable is False
        assert "no remaining credit" in msg
        assert "retrying will not clear it" in msg

    def test_rate_limit_is_retryable(self, provider):
        _, retryable, _ = provider._describe_http_error(
            _response(429, {"error": {"code": "rate_limit_exceeded",
                                      "message": "Slow down."}})
        )
        assert retryable is True

    def test_retry_after_header_beats_the_default(self, provider):
        _, _, wait = provider._describe_http_error(
            _response(429, {"error": {"code": "rate_limit_exceeded"}},
                      headers={"retry-after": "7"})
        )
        assert wait == 7

    def test_bad_key_is_permanent(self, provider):
        msg, retryable, _ = provider._describe_http_error(
            _response(401, {"error": {"message": "Incorrect API key."}})
        )
        assert retryable is False
        assert "rejected the API key" in msg

    def test_server_errors_are_retryable(self, provider):
        for status in (500, 502, 503):
            _, retryable, _ = provider._describe_http_error(_response(status, {}))
            assert retryable is True, status

    def test_context_overflow_names_the_fix(self, provider):
        msg, retryable, _ = provider._describe_http_error(
            _response(400, {"error": {"code": "context_length_exceeded",
                                      "message": "Too long."}})
        )
        assert retryable is False
        assert "new conversation" in msg

    def test_unparseable_body_does_not_crash(self, provider):
        msg, retryable, _ = provider._describe_http_error(
            _response(400, text="<html>gateway</html>")
        )
        assert retryable is False
        assert "HTTP 400" in msg


class TestMessageConversion:
    def test_system_prompt_leads(self, provider):
        out = provider._to_openai_messages([], "SYSTEM")
        assert out == [{"role": "system", "content": "SYSTEM"}]

    def test_user_and_assistant_text(self, provider):
        out = provider._to_openai_messages(
            [{"role": "user", "content": "hi"},
             {"role": "assistant", "content": "hello"}],
            "S",
        )
        assert out[1] == {"role": "user", "content": "hi"}
        assert out[2] == {"role": "assistant", "content": "hello"}

    def test_tool_arguments_are_serialised_as_a_json_string(self, provider):
        out = provider._to_openai_messages(
            [{"role": "assistant", "content": "",
              "tool_calls": [{"name": "run_rsa", "args": {"scale": 1.5}}]},
             {"role": "tool", "name": "run_rsa", "content": {"ok": True}}],
            "S",
        )
        call = out[1]["tool_calls"][0]
        assert call["type"] == "function"
        # A dict here is rejected by the API; it must be a string.
        assert isinstance(call["function"]["arguments"], str)
        assert json.loads(call["function"]["arguments"]) == {"scale": 1.5}

    def test_empty_assistant_text_becomes_null_not_empty_string(self, provider):
        out = provider._to_openai_messages(
            [{"role": "assistant", "content": "",
              "tool_calls": [{"name": "t", "args": {}}]},
             {"role": "tool", "name": "t", "content": "r"}],
            "S",
        )
        assert out[1]["content"] is None

    def test_tool_result_is_stringified(self, provider):
        out = provider._to_openai_messages(
            [{"role": "assistant", "content": "",
              "tool_calls": [{"name": "t", "args": {}}]},
             {"role": "tool", "name": "t", "content": {"vm_pu": 1.02}}],
            "S",
        )
        assert json.loads(out[2]["content"]) == {"vm_pu": 1.02}


class TestToolCallIdPairing:
    """History from another backend carries no ids; OpenAI requires them."""

    def test_ids_are_synthesised_when_history_has_none(self, provider):
        out = provider._to_openai_messages(
            [{"role": "assistant", "content": "",
              "tool_calls": [{"name": "a", "args": {}}, {"name": "b", "args": {}}]},
             {"role": "tool", "name": "a", "content": "ra"},
             {"role": "tool", "name": "b", "content": "rb"}],
            "S",
        )
        ids = [c["id"] for c in out[1]["tool_calls"]]
        assert len(set(ids)) == 2, "parallel calls must not share an id"
        assert [out[2]["tool_call_id"], out[3]["tool_call_id"]] == ids

    def test_existing_ids_are_preserved(self, provider):
        out = provider._to_openai_messages(
            [{"role": "assistant", "content": "",
              "tool_calls": [{"name": "a", "args": {}, "id": "call_abc"}]},
             {"role": "tool", "name": "a", "content": "r", "id": "call_abc"}],
            "S",
        )
        assert out[1]["tool_calls"][0]["id"] == "call_abc"
        assert out[2]["tool_call_id"] == "call_abc"

    def test_results_pair_by_name_when_they_arrive_out_of_order(self, provider):
        out = provider._to_openai_messages(
            [{"role": "assistant", "content": "",
              "tool_calls": [{"name": "slow", "args": {}},
                             {"name": "fast", "args": {}}]},
             {"role": "tool", "name": "fast", "content": "rf"},
             {"role": "tool", "name": "slow", "content": "rs"}],
            "S",
        )
        by_id = {c["id"]: c["function"]["name"] for c in out[1]["tool_calls"]}
        assert by_id[out[2]["tool_call_id"]] == "fast"
        assert by_id[out[3]["tool_call_id"]] == "slow"

    def test_orphan_tool_result_is_dropped_rather_than_sent(self, provider):
        """OpenAI rejects the whole request over one unmatched result."""
        out = provider._to_openai_messages(
            [{"role": "user", "content": "hi"},
             {"role": "tool", "name": "stray", "content": "r"}],
            "S",
        )
        assert all(m["role"] != "tool" for m in out)

    def test_every_tool_message_has_a_matching_call(self, provider):
        """The invariant OpenAI actually enforces, over a multi-turn history."""
        out = provider._to_openai_messages(
            [{"role": "user", "content": "q1"},
             {"role": "assistant", "content": "",
              "tool_calls": [{"name": "a", "args": {}}]},
             {"role": "tool", "name": "a", "content": "r"},
             {"role": "assistant", "content": "answer"},
             {"role": "user", "content": "q2"},
             {"role": "assistant", "content": "",
              "tool_calls": [{"name": "a", "args": {}}]},
             {"role": "tool", "name": "a", "content": "r2"}],
            "S",
        )
        offered = {c["id"] for m in out if m.get("tool_calls")
                   for c in m["tool_calls"]}
        claimed = [m["tool_call_id"] for m in out if m["role"] == "tool"]
        assert set(claimed) <= offered
        assert len(claimed) == len(set(claimed)), "an id was reused"


class TestResponseParsing:
    def test_plain_text_reply(self):
        out = OpenAIProvider._to_model_response(
            {"choices": [{"message": {"content": " hello "},
                          "finish_reason": "stop"}]}
        )
        assert isinstance(out, ModelResponse)
        assert out.text == "hello"
        assert out.wants_tools is False

    def test_tool_calls_are_parsed_from_the_json_string(self):
        out = OpenAIProvider._to_model_response({
            "choices": [{"message": {"content": None, "tool_calls": [
                {"id": "call_1", "type": "function",
                 "function": {"name": "run_rsa",
                              "arguments": '{"load_scaling_factor": 1.2}'}}
            ]}, "finish_reason": "tool_calls"}]
        })
        assert out.wants_tools
        assert out.tool_calls[0].name == "run_rsa"
        assert out.tool_calls[0].args == {"load_scaling_factor": 1.2}
        assert out.tool_calls[0].id == "call_1"

    def test_malformed_arguments_degrade_to_empty(self):
        out = OpenAIProvider._to_model_response({
            "choices": [{"message": {"tool_calls": [
                {"id": "c", "function": {"name": "t", "arguments": "{not json"}}
            ]}}]
        })
        assert out.tool_calls[0].args == {}

    def test_empty_arguments_string_is_an_empty_dict(self):
        out = OpenAIProvider._to_model_response({
            "choices": [{"message": {"tool_calls": [
                {"id": "c", "function": {"name": "t", "arguments": ""}}
            ]}}]
        })
        assert out.tool_calls[0].args == {}

    def test_truncated_reply_is_flagged_to_the_reader(self):
        out = OpenAIProvider._to_model_response(
            {"choices": [{"message": {"content": "half a sen"},
                          "finish_reason": "length"}]}
        )
        assert "cut off" in out.text

    def test_empty_response_does_not_crash(self):
        assert OpenAIProvider._to_model_response({}).text == ""


class TestProbe:
    def test_no_key_reports_unreachable(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        info = probe(base_url=CLOSED, api_key="")
        assert info["reachable"] is False
        assert "No API key" in info["error"]

    def test_unreachable_host_is_reported_not_raised(self, monkeypatch):
        info = probe(base_url=CLOSED, api_key="sk-test")
        assert info["reachable"] is False
        assert info["models"] == []

    def test_lists_models_on_success(self, monkeypatch):
        def fake_get(url, **kwargs):
            return _response(200, {"data": [{"id": "b"}, {"id": "a"}]})

        monkeypatch.setattr(httpx, "get", fake_get)
        info = probe(base_url=CLOSED, api_key="sk-test")
        assert info["reachable"] is True
        assert info["models"] == ["a", "b"]


class TestEndpointSelection:
    """Which endpoint a turn goes to, and why.

    A reasoning model cannot also call functions on /v1/chat/completions, and
    CONDUCTOR sends tools every turn — so wanting reasoning means wanting
    /v1/responses. Compatible servers implement only chat/completions, so a
    custom base URL stays there unless told otherwise.
    """

    def _p(self, monkeypatch, *, effort="medium", api="auto", base=None):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("OPENAI_REASONING_EFFORT", effort)
        monkeypatch.setenv("OPENAI_API", api)
        if base:
            monkeypatch.setenv("OPENAI_BASE_URL", base)
        else:
            monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        return OpenAIProvider(model="m")

    def test_auto_uses_responses_when_reasoning_is_wanted(self, monkeypatch):
        assert self._p(monkeypatch, effort="medium").uses_responses is True

    @pytest.mark.parametrize("effort", ["", "none"])
    def test_auto_uses_chat_without_reasoning(self, monkeypatch, effort):
        assert self._p(monkeypatch, effort=effort).uses_responses is False

    def test_auto_keeps_compatible_endpoints_on_chat(self, monkeypatch):
        """vLLM, OpenRouter and friends implement chat/completions only."""
        p = self._p(monkeypatch, effort="high", base="https://openrouter.test/api/v1")
        assert p.uses_responses is False

    def test_explicit_override_wins_both_ways(self, monkeypatch):
        assert self._p(monkeypatch, effort="none", api="responses").uses_responses
        assert not self._p(monkeypatch, effort="high", api="chat").uses_responses

    def test_forced_responses_works_on_a_custom_base(self, monkeypatch):
        p = self._p(monkeypatch, effort="high", api="responses",
                    base="https://selfhosted.test/v1")
        assert p.uses_responses is True

    def test_the_chosen_endpoint_is_the_one_called(self, monkeypatch):
        p = self._p(monkeypatch, effort="medium")
        seen = []

        def fake_post(url, json=None, headers=None, timeout=None):
            seen.append(url)
            return _response(200, {"status": "completed", "output": []})

        monkeypatch.setattr(httpx, "post", fake_post)
        p.chat([], [], "S")
        assert seen == ["https://api.openai.com/v1/responses"]


class TestResponsesToolShape:
    def test_declaration_is_flattened(self):
        from llm_agent.agent.tool_schemas import TOOLS

        flat = _responses_tools(TOOLS)
        assert flat, "no tools converted"
        for tool in flat:
            # Responses puts these at the top level; chat/completions nests
            # them under "function".
            assert tool["type"] == "function"
            assert "function" not in tool
            assert tool["name"] and isinstance(tool["parameters"], dict)


class TestResponsesInput:
    def test_calls_and_results_become_sibling_items(self, provider):
        items = provider._to_responses_input([
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "",
             "tool_calls": [{"name": "t", "args": {"a": 1}, "id": "call_x"}]},
            {"role": "tool", "name": "t", "content": {"ok": True}, "id": "call_x"},
        ])
        kinds = [i.get("type", "message") for i in items]
        assert kinds == ["message", "function_call", "function_call_output"]
        assert items[1]["call_id"] == items[2]["call_id"] == "call_x"
        # Arguments are a JSON string here too, not an object.
        assert json.loads(items[1]["arguments"]) == {"a": 1}

    def test_ids_are_synthesised_when_absent(self, provider):
        items = provider._to_responses_input([
            {"role": "assistant", "content": "",
             "tool_calls": [{"name": "a", "args": {}}, {"name": "b", "args": {}}]},
            {"role": "tool", "name": "a", "content": "ra"},
            {"role": "tool", "name": "b", "content": "rb"},
        ])
        calls = [i for i in items if i["type"] == "function_call"]
        outs = [i for i in items if i["type"] == "function_call_output"]
        assert len({c["call_id"] for c in calls}) == 2
        assert [o["call_id"] for o in outs] == [c["call_id"] for c in calls]

    def test_reasoning_is_replayed_before_the_call_it_belongs_to(self, provider):
        """With store:false OpenAI keeps nothing, so dropping these would make
        the model re-derive its reasoning after every tool result."""
        items = provider._to_responses_input([
            {"role": "assistant", "content": "",
             "signature": json.dumps([{"type": "reasoning", "id": "rs_1"}]),
             "tool_calls": [{"name": "t", "args": {}, "id": "c1"}]},
            {"role": "tool", "name": "t", "content": "r", "id": "c1"},
        ])
        assert [i.get("type") for i in items] == [
            "reasoning", "function_call", "function_call_output"
        ]

    def test_assistant_text_uses_output_text_parts(self, provider):
        items = provider._to_responses_input(
            [{"role": "assistant", "content": "hello"}]
        )
        assert items[0]["content"] == [{"type": "output_text", "text": "hello"}]

    def test_orphan_result_is_dropped(self, provider):
        items = provider._to_responses_input(
            [{"role": "user", "content": "q"},
             {"role": "tool", "name": "stray", "content": "r"}]
        )
        assert all(i.get("type") != "function_call_output" for i in items)


class TestResponsesParsing:
    def test_text_is_read_from_output_text_parts(self):
        out = OpenAIProvider._responses_to_model_response({"status": "completed",
            "output": [{"type": "message", "role": "assistant",
                        "content": [{"type": "output_text", "text": " hi "}]}]})
        assert out.text == "hi"

    def test_function_calls_are_read(self):
        out = OpenAIProvider._responses_to_model_response({"status": "completed",
            "output": [{"type": "function_call", "call_id": "call_z",
                        "name": "run_rsa", "arguments": '{"x": 2}'}]})
        assert out.tool_calls[0].name == "run_rsa"
        assert out.tool_calls[0].args == {"x": 2}
        assert out.tool_calls[0].id == "call_z"

    def test_reasoning_items_survive_onto_the_signature(self):
        out = OpenAIProvider._responses_to_model_response({"status": "completed",
            "output": [{"type": "reasoning", "id": "rs_1", "summary": []}]})
        assert json.loads(out.signature)[0]["id"] == "rs_1"

    def test_no_reasoning_leaves_the_signature_empty(self):
        out = OpenAIProvider._responses_to_model_response(
            {"status": "completed", "output": []})
        assert out.signature is None

    def test_malformed_arguments_degrade_to_empty(self):
        out = OpenAIProvider._responses_to_model_response({"status": "completed",
            "output": [{"type": "function_call", "call_id": "c", "name": "t",
                        "arguments": "{not json"}]})
        assert out.tool_calls[0].args == {}

    def test_incomplete_reply_is_flagged(self):
        out = OpenAIProvider._responses_to_model_response({
            "status": "incomplete",
            "incomplete_details": {"reason": "max_output_tokens"},
            "output": [{"type": "message", "content": [
                {"type": "output_text", "text": "half"}]}]})
        assert "cut off" in out.text

    def test_empty_payload_does_not_crash(self):
        assert OpenAIProvider._responses_to_model_response({}).text == ""


class TestReasoningRoundTrip:
    def test_round_trips(self):
        items = [{"type": "reasoning", "id": "rs_1", "encrypted_content": "AA=="}]
        assert _decode_reasoning(_encode_reasoning(items)) == items

    def test_empty_encodes_to_none(self):
        assert _encode_reasoning([]) is None

    @pytest.mark.parametrize("bad", [None, "", "{not json", "[1, 2]", '"text"'])
    def test_unusable_tokens_degrade_to_nothing(self, bad):
        """Losing reasoning context costs quality; it must never cost the turn."""
        assert _decode_reasoning(bad) == []


class TestDescribe:
    """The session log must be able to attribute a turn to a backend."""

    def test_reports_identity_reasoning_and_endpoint(self, monkeypatch):
        from llm_agent.agent.providers.base import describe_provider

        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("OPENAI_REASONING_EFFORT", "high")
        monkeypatch.setenv("OPENAI_API", "auto")
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        detail = describe_provider(OpenAIProvider(model="gpt-5.6-luna"))
        assert detail == {
            "provider": "openai", "model": "gpt-5.6-luna",
            "reasoning": "high", "endpoint": "/v1/responses",
        }

    def test_endpoint_disambiguates_the_same_reasoning_value(self, monkeypatch):
        """`medium` on chat/completions is silently retried at none, so the
        level alone does not say how hard the model actually thought."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("OPENAI_REASONING_EFFORT", "medium")
        monkeypatch.setenv("OPENAI_API", "chat")
        assert OpenAIProvider(model="m").describe()["endpoint"] == "/v1/chat/completions"

    def test_unset_reasoning_is_named_not_blank(self, monkeypatch):
        monkeypatch.setenv("OPENAI_REASONING_EFFORT", "")
        assert OpenAIProvider(model="m").describe()["reasoning"] == "unset"


class TestRegistry:
    def test_openai_is_selectable(self):
        from llm_agent.agent import providers

        assert "openai" in providers.available_providers()

    def test_get_provider_returns_the_openai_backend(self, monkeypatch):
        from llm_agent.agent import providers

        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        providers.reset_providers()
        try:
            assert providers.get_provider("openai").name == "openai"
        finally:
            providers.reset_providers()
