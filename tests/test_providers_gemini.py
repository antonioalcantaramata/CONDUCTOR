"""Gemini provider: message conversion in both directions.

The loop stores provider-neutral dicts and the Gemini adapter translates them to
and from genai protos. Anything the proto carries that the neutral format drops
is lost on the next request — which is how the `thought_signature` regression
happened — so these tests pin every field that has to survive a round-trip.
"""

from types import SimpleNamespace

import pytest

from llm_agent.agent.providers.base import ModelResponse, ToolCall, assistant_message
from llm_agent.agent.providers.gemini import (
    GeminiProvider,
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
