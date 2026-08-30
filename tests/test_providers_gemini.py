"""Gemini provider: message conversion in both directions.

The loop stores provider-neutral dicts and the Gemini adapter translates them to
and from genai protos. Anything the proto carries that the neutral format drops
is lost on the next request — which is how the `thought_signature` regression
happened — so these tests pin every field that has to survive a round-trip.
"""

from types import SimpleNamespace

import pytest

from llm_agent.agent.providers.base import ModelResponse, ToolCall, assistant_message
from google.genai.errors import ClientError

from llm_agent.agent.providers.gemini import (
    GeminiProvider,
    _rejected_thinking_level,
    _decode_signature,
    _encode_signature,
    _sanitize_for_proto,
)


def _part(text=None, thought=False, fn=None, sig=None, fn_id=None):
    """Build a stand-in for a genai response Part."""
    call = SimpleNamespace(name=fn, args={}, id=fn_id) if fn else None
    return SimpleNamespace(
        text=text, thought=thought, thought_signature=sig, function_call=call
    )


def _response(parts):
    return SimpleNamespace(
        candidates=[SimpleNamespace(content=SimpleNamespace(parts=parts))]
    )


@pytest.fixture
def provider():
    return GeminiProvider()


class TestDescribe:
    def test_default_reads_as_model_default_not_off(self, monkeypatch):
        """Gemini reasons either way; a blank field would imply it did not."""
        from llm_agent.agent.providers.base import describe_provider

        monkeypatch.delenv("GEMINI_THINKING_LEVEL", raising=False)
        detail = describe_provider(GeminiProvider(model="gemini-x"))
        assert detail["provider"] == "google"
        assert detail["model"] == "gemini-x"
        assert detail["reasoning"] == "model default"

    def test_pinned_level_is_reported(self, monkeypatch):
        monkeypatch.setenv("GEMINI_THINKING_LEVEL", "high")
        assert GeminiProvider().describe()["reasoning"] == "high"


class TestThinkingLevel:
    """`include_thoughts=False` hides the trace; it never stopped the model
    thinking. These pin that distinction, and that the default is unchanged."""

    def test_default_leaves_the_model_on_its_own_budget(self, monkeypatch):
        monkeypatch.delenv("GEMINI_THINKING_LEVEL", raising=False)
        assert GeminiProvider().thinking_level == ""

    def test_default_config_pins_no_level(self, monkeypatch):
        monkeypatch.delenv("GEMINI_THINKING_LEVEL", raising=False)
        cfg = GeminiProvider()._config([], "S", "")
        # The long-standing behaviour: trace suppressed, thinking untouched.
        assert cfg.thinking_config.include_thoughts is False
        assert cfg.thinking_config.thinking_level is None
        assert cfg.thinking_config.thinking_budget is None

    @pytest.mark.parametrize("level", ["minimal", "low", "medium", "high"])
    def test_every_documented_level_reaches_the_config(self, level):
        cfg = GeminiProvider(thinking_level=level)._config([], "S", level)
        # genai wants the enum spelling, which is upper case.
        assert cfg.thinking_config.thinking_level == level.upper()

    def test_level_is_normalised(self, monkeypatch):
        monkeypatch.setenv("GEMINI_THINKING_LEVEL", "  High  ")
        assert GeminiProvider().thinking_level == "high"

    def test_empty_means_unset_not_the_import_time_default(self, monkeypatch):
        monkeypatch.setenv("GEMINI_THINKING_LEVEL", "")
        assert GeminiProvider().thinking_level == ""

    def test_trace_is_still_suppressed_when_a_level_is_pinned(self):
        cfg = GeminiProvider(thinking_level="high")._config([], "S", "high")
        assert cfg.thinking_config.include_thoughts is False

    def test_other_generation_settings_are_untouched(self):
        cfg = GeminiProvider(thinking_level="low")._config(["T"], "SYS", "low")
        assert cfg.temperature == 0
        assert cfg.system_instruction == "SYS"


class TestRejectedThinkingLevel:
    """An older model refusing the field should cost a retry, not the turn."""

    def _error(self, code, message):
        """A ClientError built the way google-genai builds one."""
        return ClientError(code, {"error": {"code": code, "message": message,
                                            "status": "INVALID_ARGUMENT"}})

    def test_400_naming_thinking_is_recognised(self):
        assert _rejected_thinking_level(
            self._error(400, "Unknown name \"thinking_level\" in ThinkingConfig")
        ) is True

    def test_unrelated_400_is_not(self):
        assert _rejected_thinking_level(
            self._error(400, "Invalid function declaration")
        ) is False

    def test_non_400_is_not(self):
        assert _rejected_thinking_level(self._error(429, "thinking")) is False


class TestSignatureEncoding:
    def test_round_trips_arbitrary_bytes(self):
        for raw in [b"\x00", b"\xde\xad\xbe\xef", b"a" * 100, b"\xff" * 7]:
            assert _decode_signature(_encode_signature(raw)) == raw

    def test_encoding_none_yields_none(self):
        assert _encode_signature(None) is None
        assert _encode_signature(b"") is None

    def test_already_encoded_value_passes_through(self):
        assert _encode_signature("YWJj") == "YWJj"

    @pytest.mark.parametrize("bad", ["!!nope!!", "@@@", "", "   "])
    def test_unusable_signatures_degrade_to_none(self, bad):
        # A corrupt signature should cost model quality, never raise mid-turn.
        assert _decode_signature(bad) is None

    def test_alphabet_stripping_does_not_yield_empty_bytes(self):
        # b64decode without validate=True strips non-alphabet characters and
        # returns b'' rather than raising, which would then be attached to the
        # request as a real (empty) signature instead of being treated as absent.
        result = _decode_signature("@@@")
        assert result is None
        assert result != b""


class TestResponseToNeutral:
    def test_extracts_plain_text(self, provider):
        r = provider._to_model_response(_response([_part(text="Hello.")]))
        assert r.text == "Hello."
        assert not r.wants_tools

    def test_excludes_thought_parts_from_text(self, provider):
        r = provider._to_model_response(_response([
            _part(text="internal reasoning", thought=True),
            _part(text="The answer."),
        ]))
        assert r.text == "The answer."

    def test_collects_tool_calls(self, provider):
        r = provider._to_model_response(_response([_part(fn="run_rsa")]))
        assert [c.name for c in r.tool_calls] == ["run_rsa"]
        assert r.wants_tools

    def test_captures_signature_on_tool_calls(self, provider):
        r = provider._to_model_response(_response([_part(fn="run_rsa", sig=b"sig")]))
        assert _decode_signature(r.tool_calls[0].signature) == b"sig"

    def test_captures_function_call_id(self, provider):
        r = provider._to_model_response(_response([_part(fn="run_rsa", fn_id="c1")]))
        assert r.tool_calls[0].id == "c1"

    def test_empty_candidates_is_not_an_error(self, provider):
        r = provider._to_model_response(SimpleNamespace(candidates=[]))
        assert r.text == "" and r.tool_calls == []


class TestNeutralToContents:
    def test_user_message_becomes_user_content(self, provider):
        c = provider._to_contents([{"role": "user", "content": "hi"}])
        assert c[0].role == "user"
        assert c[0].parts[0].text == "hi"

    def test_assistant_role_is_renamed_to_model(self, provider):
        c = provider._to_contents([{"role": "assistant", "content": "hi"}])
        assert c[0].role == "model"

    def test_consecutive_tool_results_batch_into_one_turn(self, provider):
        # Gemini expects all function responses for a turn in a single user
        # Content; emitting one Content each is rejected.
        c = provider._to_contents([
            {"role": "tool", "name": "a", "content": {}},
            {"role": "tool", "name": "b", "content": {}},
        ])
        assert len(c) == 1
        assert c[0].role == "user"
        assert len(c[0].parts) == 2

    def test_tool_results_are_flushed_before_a_later_message(self, provider):
        c = provider._to_contents([
            {"role": "tool", "name": "a", "content": {}},
            {"role": "user", "content": "next"},
        ])
        assert [x.role for x in c] == ["user", "user"]
        assert c[0].parts[0].function_response is not None
        assert c[1].parts[0].text == "next"

    def test_signature_is_restored_onto_function_call_parts(self, provider):
        # Gemini 3.x rejects replayed functionCall parts that arrive without it.
        msg = {
            "role": "assistant", "content": "",
            "tool_calls": [{"name": "a", "args": {}, "signature": _encode_signature(b"s")}],
        }
        assert provider._to_contents([msg])[0].parts[0].thought_signature == b"s"

    def test_call_and_response_ids_pair_up(self, provider):
        c = provider._to_contents([
            {"role": "assistant", "content": "",
             "tool_calls": [{"name": "a", "args": {}, "id": "x1"}]},
            {"role": "tool", "name": "a", "content": {}, "id": "x1"},
        ])
        assert c[0].parts[0].function_call.id == "x1"
        assert c[1].parts[0].function_response.id == "x1"

    def test_history_without_signatures_still_converts(self, provider):
        # Conversations saved before signatures were tracked must not crash.
        c = provider._to_contents([
            {"role": "assistant", "content": "", "tool_calls": [{"name": "a", "args": {}}]}
        ])
        assert c[0].parts[0].function_call.name == "a"
        assert c[0].parts[0].thought_signature is None

    def test_assistant_message_with_no_content_or_calls_is_skipped(self, provider):
        assert provider._to_contents([{"role": "assistant", "content": ""}]) == []


class TestFullRoundTrip:
    def test_signature_and_id_survive_response_to_history_to_request(self, provider):
        response = _response([
            _part(text="Checking.", sig=b"text-sig"),
            _part(fn="run_rsa", sig=b"call-sig", fn_id="c9"),
        ])
        stored = assistant_message(provider._to_model_response(response))
        rebuilt = provider._to_contents([stored])[0]

        assert rebuilt.parts[0].thought_signature == b"text-sig"
        assert rebuilt.parts[1].thought_signature == b"call-sig"
        assert rebuilt.parts[1].function_call.id == "c9"

    def test_stored_history_is_json_serialisable(self, provider):
        import json

        response = _response([_part(fn="run_rsa", sig=b"\xff\xfe", fn_id="c1")])
        stored = assistant_message(provider._to_model_response(response))
        # Streamlit keeps history in session_state; bytes would not survive.
        assert json.loads(json.dumps(stored)) == stored


class TestSanitizeForProto:
    def test_none_becomes_empty_string(self):
        # proto Struct has no null.
        assert _sanitize_for_proto(None) == ""
        assert _sanitize_for_proto({"a": None}) == {"a": ""}

    def test_scalars_pass_through(self):
        assert _sanitize_for_proto({"i": 1, "f": 1.5, "b": True, "s": "x"}) == {
            "i": 1, "f": 1.5, "b": True, "s": "x"
        }

    def test_nested_structures_are_walked(self):
        assert _sanitize_for_proto({"a": [{"b": None}]}) == {"a": [{"b": ""}]}

    def test_non_serialisable_objects_become_strings(self):
        class Weird:
            def __str__(self):
                return "weird"

        assert _sanitize_for_proto(Weird()) == "weird"

    def test_dict_keys_are_coerced_to_strings(self):
        assert _sanitize_for_proto({1: "a"}) == {"1": "a"}


class TestModelResponse:
    def test_wants_tools_reflects_tool_calls(self):
        assert ModelResponse(tool_calls=[ToolCall("a")]).wants_tools
        assert not ModelResponse(text="done").wants_tools

    def test_tool_call_dict_round_trip(self):
        call = ToolCall("a", {"x": 1}, signature="sig", id="i1")
        assert ToolCall.from_dict(call.to_dict()) == call

    def test_optional_fields_omitted_when_absent(self):
        assert ToolCall("a").to_dict() == {"name": "a", "args": {}}
