"""
The AI Agent Suggestion (`llm_agent/agent/recommender.py`).

A second agent, run on demand, that proposes how the operator might continue.
These tests pin the properties that keep it honest:

  - it is called with no tools, on the orchestrator's provider, with its own
    system prompt — and every provider adapter accepts a tool-less call;
  - what it is told CONDUCTOR can do is generated from the real tool list;
  - its reply is parsed strictly: unknown angles, empty fields and repeats are
    dropped, never repaired;
  - a figure in a suggestion's `why` that no result contains is flagged;
  - an answer with no record is never described with an earlier turn's results;
  - the "this result has issues" highlight is decided in code, from verdicts;
  - its log records never reach the graders as turns.
"""

import json
import types

import httpx
import pytest

from llm_agent.agent import loop, recommender
from llm_agent.agent.providers.base import ModelResponse


def _record(results, user="Is the grid secure?", calls=None):
    return {
        "user": user,
        "grid": {"vm_lower": 0.95, "vm_upper": 1.05},
        "tool_calls": calls or [{"name": name, "args": {}} for name, _ in results],
        "tool_results": [{"name": name, "result": result} for name, result in results],
        "turn_key": "abc",
    }


RSA_VIOLATED = _record([("run_rsa", {
    "timestamp": "2000-01-03 02:15:00",
    "secure": False,
    "converged": True,
    "total_violations": 1,
    "violations": [{"element": "Bus_3", "violation_type": "bus_vm_pu",
                    "value": 0.9366, "violated": True}],
})])

RSA_SECURE = _record([("run_rsa", {"secure": True, "converged": True,
                                   "total_violations": 0, "violations": []})])


class FakeProvider:
    name = "fake"
    model = "fake-model"

    def __init__(self, text="", raises=None):
        self.text = text
        self.raises = raises
        self.calls = []

    def chat(self, messages, tools, system, on_event=None):
        self.calls.append({"messages": messages, "tools": tools, "system": system})
        if self.raises:
            raise self.raises
        return ModelResponse(text=self.text)


def _reply(*suggestions, note=""):
    return json.dumps({"suggestions": list(suggestions), "note": note})


def _s(angle="deeper", title="Look closer", prompt="Run X at 2000-01-03 02:15.", why=""):
    return {"angle": angle, "title": title, "prompt": prompt, "why": why}


# ---------------------------------------------------------------------------


class TestCapabilities:
    def test_generated_from_every_tool_except_the_clock(self):
        from llm_agent.agent.tool_schemas import TOOL_DISPATCH

        names = [name for name, _ in recommender.capability_list()]
        assert set(names) == set(TOOL_DISPATCH) - {"get_current_timestamp", "advance_timestamp"}
        assert all(summary for _, summary in recommender.capability_list())

    def test_summary_is_one_sentence(self):
        caps = dict(recommender.capability_list())
        assert caps["simulate_contingency"].endswith("outage).")

    def test_prompt_names_every_angle_and_capability(self):
        prompt = recommender.build_system_prompt()
        for angle in recommender.ANGLES:
            assert angle in prompt
        for name, _ in recommender.capability_list():
            assert f"- {name}:" in prompt
        # The format braces survive `.format`, the JSON example does not break it.
        assert '{"suggestions": [{"angle"' in prompt

    def test_prompt_is_small_compared_with_the_orchestrator(self):
        # One click is meant to be cheap; the orchestrator prompt is ~25k tokens.
        assert len(recommender.build_system_prompt()) // 4 < 3000


class TestParse:
    def test_fenced_json_with_prose_around_it(self):
        text = "Here you go:\n```json\n" + _reply(_s()) + "\n```\nThanks"
        result = recommender.parse_response(text)
        assert result.ok and len(result.suggestions) == 1

    def test_angle_spelling_is_normalised(self):
        result = recommender.parse_response(_reply(_s(angle="Step back")))
        assert result.suggestions[0].angle == "step_back"

    def test_unknown_angle_and_empty_fields_are_dropped_not_repaired(self):
        result = recommender.parse_response(_reply(
            _s(angle="bogus"), _s(title=""), _s(prompt="  "), _s(title="Kept")))
        assert [s.title for s in result.suggestions] == ["Kept"]

    def test_at_most_four(self):
        result = recommender.parse_response(_reply(
            *[_s(title=f"T{i}", prompt=f"P{i}") for i in range(6)]))
        assert len(result.suggestions) == recommender.MAX_SUGGESTIONS

    def test_repeats_within_a_reply_and_of_earlier_rounds_are_dropped(self):
        shown = [{"angle": "deeper", "title": "Old idea", "prompt": "Old prompt"}]
        result = recommender.parse_response(_reply(
            _s(title="old  IDEA", prompt="new"), _s(title="A", prompt="Old prompt"),
            _s(title="B", prompt="b"), _s(title="b", prompt="c")), shown)
        assert [s.title for s in result.suggestions] == ["B"]

    def test_not_json_is_an_error_without_raw_output_in_the_message(self):
        result = recommender.parse_response("I think you should check the voltages.")
        assert not result.ok and "JSON" in result.error
        assert "voltages" not in result.error

    def test_malformed_json_is_an_error(self):
        assert not recommender.parse_response('{"suggestions": [').ok

    def test_empty_list_with_note(self):
        result = recommender.parse_response(_reply(note="Nothing new to add."))
        assert result.ok and result.suggestions == () and result.note == "Nothing new to add."


class TestGrounding:
    def test_a_figure_from_the_results_is_not_flagged(self):
        (s,) = recommender.ground(
            (recommender.Suggestion("stress", "t", "p", "Bus_3 is at 0.9366 p.u."),),
            [RSA_VIOLATED])
        assert s.unverified == ()

    def test_an_invented_figure_is_flagged(self):
        (s,) = recommender.ground(
            (recommender.Suggestion("stress", "t", "p", "Bus_3 may reach 0.9012 p.u."),),
            [RSA_VIOLATED])
        assert s.unverified == ("0.9012 p.u.",)

    def test_the_prompt_is_not_checked(self):
        # A prompt proposes parameters ("at 120 % load"); those are not claims.
        (s,) = recommender.ground(
            (recommender.Suggestion("change", "t", "Scale load to 120 %.", ""),), [RSA_VIOLATED])
        assert s.unverified == ()

    def test_earlier_turns_count_as_evidence(self):
        later = _record([("get_current_timestamp", {"current_timestamp": "x"})])
        (s,) = recommender.ground(
            (recommender.Suggestion("deeper", "t", "p", "Bus_3 was at 0.9366 p.u."),),
            [RSA_VIOLATED, later])
        assert s.unverified == ()


class TestAttention:
    def test_secure_result_needs_none(self):
        assert recommender.needs_attention(RSA_SECURE) == []

    def test_no_record_needs_none(self):
        assert recommender.needs_attention(None) == []

    def test_insecure_result_with_violations(self):
        reasons = recommender.needs_attention(RSA_VIOLATED)
        assert "run_rsa: secure is false" in reasons
        assert "run_rsa: 1 violation(s)" in reasons

    def test_n1_failure(self):
        record = _record([("simulate_all_contingencies",
                           {"system_n1_secure": False, "total_violations": 4})])
        assert "simulate_all_contingencies: system_n1_secure is false" in \
            recommender.needs_attention(record)

    def test_direct_sub_object_counts(self):
        record = _record([("optimize_flexibility",
                           {"feasible": True, "constrained": {"feasible": False}})])
        assert recommender.needs_attention(record) == ["optimize_flexibility: feasible is false"]

    def test_per_sample_verdicts_inside_lists_do_not(self):
        record = _record([("run_probabilistic_rsa",
                           {"samples": [{"converged": False}, {"converged": True}]})])
        assert recommender.needs_attention(record) == []

    def test_tool_error(self):
        record = _record([("run_rsa", {"error": "backend down"})])
        assert recommender.needs_attention(record) == ["run_rsa: returned an error"]

    def test_bool_is_not_a_violation_count(self):
        record = _record([("x", {"total_violations": True})])
        assert recommender.needs_attention(record) == []


class TestCompactTurn:
    def test_long_lists_are_cut_with_their_length_kept(self):
        record = _record([("run_rsa", {"violations": [{"v": i} for i in range(20)]})])
        view = recommender.compact_turn(record)["tools"][0]["result"]["violations"]
        assert len(view) == 6 and view[-1] == "<15 more>"

    def test_private_fields_dropped_but_integrity_warnings_kept(self):
        record = _record([("run_rsa", {"_raw": 1, "_integrity_warnings": ["x"], "ok": 1})])
        view = recommender.compact_turn(record)["tools"][0]["result"]
        assert view == {"_integrity_warnings": ["x"], "ok": 1}

    def test_oversized_result_falls_back_to_its_verdicts(self):
        big = {"secure": False, "total_violations": 3,
               "table": {f"bus{i}": {"vm": i, "name": "x" * 150} for i in range(200)}}
        view = recommender.compact_turn(_record([("run_rsa", big)]))["tools"][0]["result"]
        assert view == {"secure": False, "total_violations": 3}

    def test_scan_over_time_uses_the_model_summary(self):
        result = {"timestamps": ["a", "b"], "violation_counts": [0, 2], "any_violations": True}
        view = recommender.compact_turn(_record([("scan_rsa_over_time", result)]))
        assert view["tools"][0]["result"]["worst_timestamp"] == "b"


class TestSuggest:
    def test_called_with_no_tools_and_its_own_prompt(self):
        provider = FakeProvider(_reply(_s()))
        result = recommender.suggest([{"role": "user", "text": "hi"}], RSA_VIOLATED,
                                     [RSA_VIOLATED], {"network": "n"}, provider=provider)
        (call,) = provider.calls
        assert call["tools"] == []
        assert call["system"].startswith("You are the suggestion agent of CONDUCTOR")
        assert len(call["messages"]) == 1 and call["messages"][0]["role"] == "user"
        assert result.ok and len(result.suggestions) == 1

    def test_request_carries_concerns_context_and_what_was_shown(self):
        provider = FakeProvider(_reply())
        recommender.suggest([], RSA_VIOLATED, [RSA_VIOLATED], {"network": "Bornholm"},
                            shown=[{"angle": "deeper", "title": "Old idea"}],
                            provider=provider)
        request = provider.calls[0]["messages"][0]["content"]
        assert "run_rsa: secure is false" in request
        assert "Bornholm" in request
        assert "[deeper] Old idea" in request

    def test_an_answer_without_a_record_is_not_given_an_earlier_one(self):
        provider = FakeProvider(_reply())
        recommender.suggest([], None, [RSA_VIOLATED], {}, provider=provider)
        request = provider.calls[0]["messages"][0]["content"]
        assert "no tools were run in the latest turn" in request
        assert "secure is false" not in request

    def test_provider_failure_becomes_an_error_result(self):
        result = recommender.suggest([], None, [], {},
                                     provider=FakeProvider(raises=RuntimeError("quota")))
        assert not result.ok and "quota" in result.error

    def test_figures_in_why_are_grounded_against_every_record(self):
        provider = FakeProvider(_reply(_s(why="Bus_3 sits at 0.9366 p.u."),
                                       _s(title="B", prompt="b", why="It may reach 0.85 p.u.")))
        result = recommender.suggest([], RSA_SECURE, [RSA_VIOLATED, RSA_SECURE], {},
                                     provider=provider)
        assert [s.unverified for s in result.suggestions] == [(), ("0.85 p.u.",)]


class TestConversationText:
    def test_payload_free_and_bounded(self):
        messages = [{"role": "user", "text": f"q{i}"} for i in range(30)]
        text = recommender._conversation_text(messages)
        assert "q29" in text and "q0" not in text

    def test_long_answers_are_truncated(self):
        text = recommender._conversation_text([{"role": "assistant", "text": "x" * 9000}])
        assert len(text) < 3000 and text.endswith("[…]")


# ---------------------------------------------------------------------------
# Every adapter accepts a tool-less call. The OpenAI API rejects an empty
# `tools` array outright, so the key must be absent rather than empty.
# ---------------------------------------------------------------------------


class TestToolLessCalls:
    def test_anthropic(self, monkeypatch):
        from llm_agent.agent.providers.claude import AnthropicProvider

        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        provider = AnthropicProvider(model="claude-test")
        sent = {}

        def fake_create(**kwargs):
            sent.update(kwargs)
            block = types.SimpleNamespace(type="text", text="ok")
            block.model_dump = lambda exclude_none=True: {"type": "text", "text": "ok"}
            return types.SimpleNamespace(content=[block], stop_reason="end_turn",
                                         stop_details=None)

        monkeypatch.setattr(type(provider), "client", property(
            lambda self: types.SimpleNamespace(messages=types.SimpleNamespace(create=fake_create))))
        provider.chat([{"role": "user", "content": "hi"}], [], "S")
        assert "tools" not in sent

    @pytest.mark.parametrize("endpoint", ["completions", "responses"])
    def test_openai(self, monkeypatch, endpoint):
        from llm_agent.agent.providers.openai import OpenAIProvider

        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        provider = OpenAIProvider(model="m", base_url="http://localhost:59998/v1")
        sent = {}

        def fake_post(url, json=None, headers=None, timeout=None):
            sent.update(json)
            payload = ({"choices": [{"message": {"content": "ok"}}]}
                       if endpoint == "completions" else {"status": "completed", "output": []})
            return httpx.Response(200, json=payload, request=httpx.Request("POST", url))

        monkeypatch.setattr(httpx, "post", fake_post)
        getattr(provider, f"_chat_via_{endpoint}")([], [], "S")
        assert "tools" not in sent

    def test_ollama(self, monkeypatch):
        from llm_agent.agent.providers.ollama import OllamaProvider

        provider = OllamaProvider(model="m", host="http://localhost:59999")
        provider._preflighted = True
        sent = {}

        def fake_post(url, json=None, timeout=None):
            sent.update(json)
            return httpx.Response(200, json={"message": {"content": "ok"}},
                                  request=httpx.Request("POST", url))

        monkeypatch.setattr(httpx, "post", fake_post)
        provider.chat([], [], "S")
        assert "tools" not in sent

    def test_gemini(self):
        from llm_agent.agent.providers.gemini import GeminiProvider

        assert GeminiProvider()._config([], "S", "").tools is None

    def test_tools_still_sent_when_given(self, monkeypatch):
        from llm_agent.agent.providers.ollama import OllamaProvider
        from llm_agent.agent.tool_schemas import TOOLS

        # Large enough that the full tool list fits the context guard.
        monkeypatch.setenv("OLLAMA_NUM_CTX", "200000")
        provider = OllamaProvider(model="m", host="http://localhost:59999")
        provider._preflighted = True
        sent = {}

        def fake_post(url, json=None, timeout=None):
            sent.update(json)
            return httpx.Response(200, json={"message": {"content": "ok"}},
                                  request=httpx.Request("POST", url))

        monkeypatch.setattr(httpx, "post", fake_post)
        provider.chat([], TOOLS, "S")
        assert len(sent["tools"]) == len(recommender.capability_list()) + 2


# ---------------------------------------------------------------------------
# Logging: suggestion records share the session file but are not turns.
# ---------------------------------------------------------------------------


class TestLogging:
    @pytest.fixture
    def log_dir(self, tmp_path, monkeypatch):
        monkeypatch.setattr(loop, "_LOG_DIR", tmp_path)
        monkeypatch.setattr(loop, "_LATEST", tmp_path / "last_session.jsonl")
        monkeypatch.setattr(loop, "SESSION_LOG_PATH", "")
        monkeypatch.setattr(loop, "_log_path", None)
        monkeypatch.setattr(loop, "_log_initialized", False)
        monkeypatch.setattr(loop, "session_log", [])
        return tmp_path

    def test_event_goes_to_the_file_not_the_turn_list(self, log_dir):
        loop._append_to_log({"turn": 1, "user": "q"})
        loop.log_event({"kind": "suggestion", "suggestions": []})
        assert loop.session_log == [{"turn": 1, "user": "q"}]
        lines = loop._session_log_path().read_text(encoding="utf-8").splitlines()
        assert [json.loads(line).get("kind") for line in lines] == [None, "suggestion"]

    def test_graders_read_turns_only(self, log_dir):
        from evaluation.report import grade_log
        from evaluation.session_log import load_records

        loop._append_to_log({"turn": 1, "user": "q", "assistant": "a"})
        loop.log_event({"kind": "suggestion", "suggestions": []})
        loop.log_event({"kind": "suggestion_outcome", "outcome": "used"})
        path = loop._session_log_path()
        assert [r["turn"] for r in load_records(path)] == [1]
        assert len(load_records(path, kind=None)) == 3
        assert len(load_records(path, kind="suggestion")) == 1
        assert len(grade_log(path)) == 1


class TestOrchestratorPrompt:
    def test_scripted_offers_are_gone(self):
        # The suggestion agent replaces the "Always offer…" follow-ups.
        from llm_agent.agent.system_prompt import build_system_prompt
        from llm_agent.agent.config import DEFAULT_GRID_CONSTANTS
        from llm_agent.agent.tool_schemas import TOOLS
        from llm_agent.agent.providers.schema import function_declarations

        prompt = build_system_prompt(DEFAULT_GRID_CONSTANTS)
        assert "Always offer" not in prompt
        assert "offer to jump" not in prompt
        assert not any("offer to advance_timestamp" in (fd.description or "")
                       for fd in function_declarations(TOOLS))


class TestRobustness:
    def test_grounding_failure_keeps_the_suggestions(self, monkeypatch):
        def broken(*a, **k):
            raise ValueError("boom")

        monkeypatch.setattr(recommender, "ground", broken)
        result = recommender.suggest([], None, [], {}, provider=FakeProvider(_reply(_s())))
        assert result.ok and len(result.suggestions) == 1
