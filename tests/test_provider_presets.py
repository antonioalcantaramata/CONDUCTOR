"""
OpenRouter / NVIDIA Build / DeepSeek presets, the Gemini fallback chain, and
the runtime handling around them: rate limits per minute vs per day, models
that are gone, timeouts, history compaction and the tool-calling test.
No test contacts a real API.
"""

import types

import httpx
import pytest
from google.genai.errors import ClientError

from llm_agent.agent import loop as loop_mod
from llm_agent.agent import smoke_test
from llm_agent.agent.discovery import cache, nvidia, openrouter
from llm_agent.agent.providers import gemini as gemini_mod
from llm_agent.agent.providers.base import ModelResponse, ToolCall, user_message
from llm_agent.agent.providers.gemini import GeminiProvider
from llm_agent.agent.providers.openai_compat import CompatProvider

OK = {"choices": [{"message": {"content": "Done."}, "finish_reason": "stop"}],
      "usage": {"prompt_tokens": 28100}, "model": "z-ai/glm-5.3"}


def _response(status, payload=None, text="", headers=None):
    return httpx.Response(status_code=status, json=payload, text=None if payload is not None else text,
                          headers=headers or {}, request=httpx.Request("POST", "http://x/chat/completions"))


@pytest.fixture(autouse=True)
def clean_env(tmp_path, monkeypatch):
    monkeypatch.setenv("CONDUCTOR_CACHE_DIR", str(tmp_path))
    for var in ("OPENROUTER_TIER", "OPENROUTER_MODEL", "NVIDIA_MODEL", "NVIDIA_TIMEOUT_S", "DEEPSEEK_TIER",
                "DEEPSEEK_MODEL", "OPENROUTER_FALLBACKS", "OPENROUTER_DENY_DATA_COLLECTION", "GEMINI_TIER",
                "GEMINI_FALLBACK_MODELS", "GEMINI_MODEL", "OPENAI_REASONING_EFFORT", "OPENAI_API"):
        monkeypatch.delenv(var, raising=False)
    for var in ("OPENROUTER_API_KEY", "NVIDIA_API_KEY", "DEEPSEEK_API_KEY"):
        monkeypatch.setenv(var, "test-key")
    monkeypatch.setattr("time.sleep", lambda s: None)


@pytest.fixture
def posts(monkeypatch):
    """Record every POST; reply from a queue of responses (or exceptions)."""
    sent, replies = [], []

    def fake_post(url, json=None, headers=None, timeout=None):
        sent.append({"url": url, "json": json, "headers": headers, "timeout": timeout})
        reply = replies.pop(0) if replies else _response(200, OK)
        if isinstance(reply, Exception):
            raise reply
        return reply
    monkeypatch.setattr(httpx, "post", fake_post)
    return types.SimpleNamespace(sent=sent, replies=replies)


def _chat(provider, events=None):
    return provider.chat([user_message("Is the grid secure?")], [], "SYS",
                         on_event=(lambda e, d: events.append((e, d))) if events is not None else None)


class TestPresetSettings:
    def test_nvidia_from_the_catalog_and_its_own_variables(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "nvidia")
        p = CompatProvider("nvidia")
        assert p.base_url == "https://integrate.api.nvidia.com/v1"
        assert p.model == "nvidia/nemotron-3-super-120b-a12b" and p.timeout_s == 180 and p.api_key == "test-key"
        monkeypatch.setenv("NVIDIA_MODEL", "deepseek-ai/deepseek-v4.1-flash")
        monkeypatch.setenv("NVIDIA_TIMEOUT_S", "300")
        p = CompatProvider("nvidia")
        assert p.model == "deepseek-ai/deepseek-v4.1-flash" and p.timeout_s == 300

    def test_always_chat_completions_never_responses(self, monkeypatch, posts):
        monkeypatch.setenv("OPENAI_API", "responses")
        monkeypatch.setenv("OPENAI_REASONING_EFFORT", "high")
        p = CompatProvider("deepseek")
        assert not p.uses_responses and p.reasoning_effort == ""
        _chat(p)
        assert posts.sent[0]["url"] == "https://api.deepseek.com/chat/completions"
        assert "reasoning_effort" not in posts.sent[0]["json"]
        assert p.model == "deepseek-flash"

    def test_a_missing_key_says_which_variable(self, monkeypatch):
        monkeypatch.delenv("NVIDIA_API_KEY")
        with pytest.raises(RuntimeError, match="NVIDIA_API_KEY"):
            CompatProvider("nvidia").preflight()

    def test_the_old_openai_setup_is_untouched(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        monkeypatch.setenv("OPENAI_BASE_URL", "https://my-vllm/v1")
        from llm_agent.agent.providers import get_provider, reset_providers
        reset_providers()
        p = get_provider()
        assert type(p).__name__ == "OpenAIProvider" and p.base_url == "https://my-vllm/v1"
        reset_providers()


class TestOpenRouterExtras:
    MODELS = [{"id": "a:free"}, {"id": "b:free"}, {"id": "c:free"}, {"id": "d:free"}]

    def test_fallbacks_on_the_free_tier_and_data_collection_allowed_by_default(self, monkeypatch, posts):
        monkeypatch.setenv("OPENROUTER_MODEL", "b:free")
        monkeypatch.setattr(openrouter, "discover", lambda **kw: {"models": self.MODELS})
        _chat(CompatProvider("openrouter"))
        body = posts.sent[0]["json"]
        assert body["model"] == "b:free" and body["models"] == ["b:free", "a:free", "c:free"]
        assert "provider" not in body

    def test_excluding_data_collection_is_opt_in(self, monkeypatch, posts):
        monkeypatch.setenv("OPENROUTER_MODEL", "b:free")
        monkeypatch.setenv("OPENROUTER_DENY_DATA_COLLECTION", "1")
        monkeypatch.setattr(openrouter, "discover", lambda **kw: {"models": self.MODELS})
        _chat(CompatProvider("openrouter"))
        assert posts.sent[0]["json"]["provider"] == {"data_collection": "deny"}
        assert posts.sent[0]["headers"]["X-Title"] == "CONDUCTOR"

    def test_both_can_be_turned_off(self, monkeypatch, posts):
        monkeypatch.setenv("OPENROUTER_MODEL", "b:free")
        monkeypatch.setenv("OPENROUTER_FALLBACKS", "0")
        monkeypatch.setenv("OPENROUTER_DENY_DATA_COLLECTION", "0")
        monkeypatch.setattr(openrouter, "discover", lambda **kw: {"models": self.MODELS})
        _chat(CompatProvider("openrouter"))
        assert "models" not in posts.sent[0]["json"] and "provider" not in posts.sent[0]["json"]

    def test_not_on_the_paid_tier(self, monkeypatch, posts):
        monkeypatch.setenv("LLM_PROVIDER", "openrouter")
        monkeypatch.setenv("OPENROUTER_TIER", "paid")
        monkeypatch.setenv("OPENROUTER_MODEL", "x/paid")
        _chat(CompatProvider("openrouter"))
        assert "models" not in posts.sent[0]["json"] and "provider" not in posts.sent[0]["json"]

    def test_a_server_side_fallback_is_announced(self, monkeypatch, posts):
        monkeypatch.setenv("OPENROUTER_MODEL", "b:free")
        monkeypatch.setattr(openrouter, "discover", lambda **kw: {"models": self.MODELS})
        posts.replies.append(_response(200, {**OK, "model": "c-20261001:free"}))
        events = []
        _chat(CompatProvider("openrouter"), events)
        assert events[-1][0] == "model_switch" and events[-1][1]["temporary"]

    def test_the_dated_name_of_the_same_model_is_not_a_fallback(self, monkeypatch, posts):
        monkeypatch.setenv("OPENROUTER_MODEL", "acme/inkling-small:free")
        monkeypatch.setattr(openrouter, "discover", lambda **kw: {"models": self.MODELS})
        posts.replies.append(_response(200, {**OK, "model": "acme/inkling-small-20260730:free"}))
        events = []
        _chat(CompatProvider("openrouter"), events)
        assert not [e for e in events if e[0] == "model_switch"]


class TestRuntimeHandling:
    def test_a_model_that_is_gone_is_replaced_and_retried_once(self, monkeypatch, posts):
        monkeypatch.setenv("LLM_PROVIDER", "nvidia")
        monkeypatch.setenv("NVIDIA_MODEL", "z-ai/glm-5.3")
        monkeypatch.setattr(nvidia, "discover", lambda allowlist, force=False: {"available": [
            {"id": "z-ai/glm-5.3"}, {"id": "deepseek-ai/deepseek-v4.1-flash"}]})
        posts.replies.append(_response(404, text="404 page not found"))
        events = []
        p = CompatProvider("nvidia")
        reply = _chat(p, events)
        assert reply.text == "Done." and p.model == "deepseek-ai/deepseek-v4.1-flash"
        assert [s["json"]["model"] for s in posts.sent] == ["z-ai/glm-5.3", "deepseek-ai/deepseek-v4.1-flash"]
        assert events[-1][0] == "model_switch" and events[-1][1]["to"] == "deepseek-ai/deepseek-v4.1-flash"
        import os
        assert os.environ["NVIDIA_MODEL"] == "deepseek-ai/deepseek-v4.1-flash"
        monkeypatch.delenv("NVIDIA_MODEL")

    def test_no_replacement_reports_the_missing_model(self, monkeypatch, posts):
        monkeypatch.setenv("NVIDIA_MODEL", "z-ai/glm-5.3")
        monkeypatch.setattr(nvidia, "discover", lambda allowlist, force=False: {"available": [{"id": "z-ai/glm-5.3"}]})
        posts.replies.extend([_response(404, text="Model does not exist")])
        with pytest.raises(RuntimeError, match="no longer serves"):
            _chat(CompatProvider("nvidia"))

    def test_openrouter_recovery_refreshes_the_list(self, monkeypatch, posts):
        monkeypatch.setenv("OPENROUTER_MODEL", "gone:free")
        forced = []

        def discover(free_only=True, force=False):
            forced.append(force)
            return {"models": [{"id": "gone:free"}, {"id": "next:free"}]}
        monkeypatch.setattr(openrouter, "discover", discover)
        posts.replies.append(_response(404, {"error": {"code": 404, "message": "No endpoints found for gone:free."}}))
        p = CompatProvider("openrouter")
        _chat(p)
        assert p.model == "next:free" and True in forced
        monkeypatch.delenv("OPENROUTER_MODEL")

    def test_a_daily_limit_stops_with_when_it_resets_and_alternatives(self, monkeypatch, posts):
        monkeypatch.setattr(openrouter, "discover", lambda **kw: {"models": []})
        posts.replies.append(_response(429, {"error": {"code": 429, "message": "Rate limit exceeded"}},
                                       headers={"X-RateLimit-Limit": "50", "X-RateLimit-Reset": "1792000000000"}))
        with pytest.raises(RuntimeError) as exc:
            _chat(CompatProvider("openrouter"))
        text = str(exc.value)
        assert "daily request limit" in text and "resets at" in text and "Gemini" in text and "$10" in text
        assert len(posts.sent) == 1                                   # no retries

    def test_a_per_minute_limit_backs_off_and_retries(self, monkeypatch, posts):
        monkeypatch.setattr(openrouter, "discover", lambda **kw: {"models": []})
        posts.replies.append(_response(429, {"error": {"message": "Rate limit exceeded"}},
                                       headers={"X-RateLimit-Limit": "20"}))
        events = []
        assert _chat(CompatProvider("openrouter"), events).text == "Done."
        assert [e for e, _ in events].count("retry") == 1 and len(posts.sent) == 2

    def test_spent_deepseek_credit_points_to_nvidia(self, posts):
        posts.replies.append(_response(402, {"error": {"message": "You have run out of balance"}}))
        with pytest.raises(RuntimeError, match="NVIDIA Build"):
            _chat(CompatProvider("deepseek"))

    def test_a_timeout_is_retried_once_then_reported_as_a_timeout(self, monkeypatch, posts):
        posts.replies.extend([httpx.ReadTimeout("slow"), httpx.ReadTimeout("slow")])
        monkeypatch.setattr(nvidia, "discover", lambda *a, **k: pytest.fail("a timeout is not a missing model"))
        with pytest.raises(RuntimeError) as exc:
            _chat(CompatProvider("nvidia"))
        assert "timeout, not a missing model" in str(exc.value) and "NVIDIA_TIMEOUT_S" in str(exc.value)
        assert "(tried twice)" in str(exc.value)
        assert len(posts.sent) == 2

    def test_a_timeout_then_success(self, posts):
        posts.replies.append(httpx.ReadTimeout("slow"))
        assert _chat(CompatProvider("nvidia")).text == "Done."


def _gemini_error(code, quota_id=None, message="quota"):
    details = [{"violations": [{"quotaId": quota_id}]}] if quota_id else []
    return ClientError(code, {"error": {"code": code, "message": message, "status": "RESOURCE_EXHAUSTED",
                                        "details": details}})


DAILY = "GenerateRequestsPerDayPerProjectPerModel-FreeTier"


class TestGeminiFallbackChain:
    def _provider(self, monkeypatch, behaviour, model="gemini-3.5-flash-lite"):
        monkeypatch.setenv("GEMINI_API_KEY", "test")
        calls = []

        def generate(model, contents, config):
            calls.append(model)
            outcome = behaviour(model, len(calls))
            if isinstance(outcome, Exception):
                raise outcome
            return outcome
        p = GeminiProvider(model=model)
        p._client = types.SimpleNamespace(models=types.SimpleNamespace(generate_content=generate))
        return p, calls

    def test_a_daily_quota_switches_to_the_next_model(self, monkeypatch):
        p, calls = self._provider(monkeypatch, lambda m, n: _gemini_error(429, DAILY)
                                  if m == "gemini-3.5-flash-lite" else "RESPONSE")
        events = []
        out = p._generate_with_retry([], None, on_event=lambda e, d: events.append((e, d)))
        assert out == "RESPONSE" and calls == ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite"]
        assert p.model == "gemini-3.1-flash-lite"
        assert events == [("model_switch", {"from": "gemini-3.5-flash-lite", "to": "gemini-3.1-flash-lite",
                                            "reason": "gemini-3.5-flash-lite: its daily free quota is used up"})]
        monkeypatch.delenv("GEMINI_MODEL", raising=False)

    def test_the_whole_chain_spent_stops_with_the_reset_time(self, monkeypatch):
        p, calls = self._provider(monkeypatch, lambda m, n: _gemini_error(429, DAILY))
        with pytest.raises(RuntimeError) as exc:
            p._generate_with_retry([], None, max_retries=0)
        assert calls == ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite"]
        assert "midnight Pacific" in str(exc.value) and "NVIDIA" in str(exc.value)
        monkeypatch.delenv("GEMINI_MODEL", raising=False)

    def test_the_chain_is_configurable_and_free_tier_only(self, monkeypatch):
        monkeypatch.setenv("GEMINI_FALLBACK_MODELS", "gemini-3.5-flash-lite, gemini-3.8-flash")
        assert GeminiProvider(model="gemini-3.5-flash-lite").fallback_chain() == [
            "gemini-3.5-flash-lite", "gemini-3.8-flash"]
        monkeypatch.setenv("GEMINI_TIER", "paid")
        p, calls = self._provider(monkeypatch, lambda m, n: _gemini_error(429, DAILY))
        with pytest.raises(RuntimeError):
            p._generate_with_retry([], None)
        assert calls == ["gemini-3.5-flash-lite"]

    def test_a_per_minute_limit_waits_then_succeeds(self, monkeypatch):
        sleeps = []
        monkeypatch.setattr(gemini_mod.time, "sleep", lambda s: sleeps.append(s))
        p, calls = self._provider(monkeypatch, lambda m, n: _gemini_error(
            429, "GenerateRequestsPerMinutePerProjectPerModel-FreeTier") if n == 1 else "RESPONSE")
        events = []
        assert p._generate_with_retry([], None, on_event=lambda e, d: events.append((e, d))) == "RESPONSE"
        assert sleeps == [10] and events[0][1]["reason"] == "Waiting for the per-minute rate limit"

    def test_a_token_cap_smaller_than_one_request_fails_at_once(self, monkeypatch):
        error = ClientError(429, {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "message": (
            "quotaMetric: generate_content_free_tier_input_token_count quotaId: "
            "'GenerateContentInputTokensPerModelPerMinute-FreeTier' quotaValue: '16000'")}})
        p, calls = self._provider(monkeypatch, lambda m, n: error, model="gemma-4-31b-it")
        with pytest.raises(RuntimeError, match="16,000 input tokens a minute"):
            p._generate_with_retry([], None)
        assert calls == ["gemma-4-31b-it"]

    def test_a_retired_model_moves_along_the_chain(self, monkeypatch):
        p, calls = self._provider(monkeypatch, lambda m, n: _gemini_error(404, message="NOT_FOUND")
                                  if n == 1 else "RESPONSE")
        assert p._generate_with_retry([], None) == "RESPONSE" and calls[-1] == "gemini-3.1-flash-lite"
        monkeypatch.delenv("GEMINI_MODEL", raising=False)

    def test_the_model_setting_applies_without_a_restart(self, monkeypatch):
        monkeypatch.setenv("GEMINI_MODEL", "gemini-3.8-flash")
        assert GeminiProvider().model == "gemini-3.8-flash"


class TestHistoryCompaction:
    def _history(self):
        big = {"all_voltages": list(range(2000))}
        return [user_message("q1"), {"role": "assistant", "content": "", "tool_calls": []},
                {"role": "tool", "name": "run_rsa", "content": big},
                {"role": "assistant", "content": "a1"},
                user_message("q2"), {"role": "tool", "name": "run_rsa", "content": big}]

    def test_only_earlier_questions_are_shortened(self):
        history = self._history()
        out = loop_mod.compact_history(history, turn_start=4)
        assert "earlier_result_shortened" in out[2]["content"]
        assert out[5]["content"] == history[5]["content"]                 # the current question: untouched
        assert history[2]["content"] == {"all_voltages": list(range(2000))}   # the stored history: untouched

    def test_small_results_are_kept(self):
        history = [user_message("q1"), {"role": "tool", "name": "x", "content": {"ok": True}}, user_message("q2")]
        assert loop_mod.compact_history(history, 2) == history

    def test_on_off_and_auto(self, monkeypatch):
        monkeypatch.setenv("CONDUCTOR_COMPACT_HISTORY", "on")
        assert loop_mod._compaction_on()
        monkeypatch.setenv("CONDUCTOR_COMPACT_HISTORY", "off")
        assert not loop_mod._compaction_on()
        monkeypatch.setenv("CONDUCTOR_COMPACT_HISTORY", "auto")
        monkeypatch.setenv("LLM_PROVIDER", "nvidia")
        assert loop_mod._compaction_on()
        monkeypatch.setenv("LLM_PROVIDER", "anthropic")
        assert not loop_mod._compaction_on()


class TestToolCallingTest:
    class _Fake:
        def __init__(self, calls):
            self.calls = calls

        def chat(self, messages, tools, system, on_event=None):
            return ModelResponse(text="", tool_calls=[ToolCall(name=n) for n in self.calls])

    def test_a_tool_call_passes_and_is_cached(self):
        result = smoke_test.run("nvidia", "z-ai/glm-5.3-flash", provider=self._Fake(["get_time"]))
        assert result["ok"] and cache.smoke_result("nvidia", "z-ai/glm-5.3-flash")["ok"]

    def test_no_tool_call_fails(self):
        result = smoke_test.run("openrouter", "x:free", provider=self._Fake([]))
        assert not result["ok"] and "without calling" in result["error"]

    def test_a_wrong_tool_name_is_a_malformed_call(self):
        result = smoke_test.run("openrouter", "y:free", provider=self._Fake(["get"]))
        assert not result["ok"] and "malformed" in result["error"]

    def test_an_error_is_a_failed_result(self):
        class Broken:
            def chat(self, *a, **k):
                raise RuntimeError("404")
        result = smoke_test.run("nvidia", "gone", provider=Broken())
        assert not result["ok"] and "404" in result["error"]


def test_the_loop_counts_calls_and_reported_tokens(monkeypatch, tmp_path):
    from llm_agent.agent import usage

    class Provider:
        name, model = "nvidia", "z-ai/glm-5.3"

        def chat(self, messages, tools, system, on_event=None):
            return ModelResponse(text="Answer.", raw={"usage": {"prompt_tokens": 28084}})
    monkeypatch.setattr(loop_mod, "get_provider", lambda: Provider())
    monkeypatch.setattr(loop_mod, "get_system_prompt", lambda: "SYS")
    monkeypatch.setattr(loop_mod, "_append_to_log", lambda record: None)
    loop_mod.run_agent_turn("Hello?", [], reflection="off")
    store = usage.load_store()
    assert store["turns"] == {"n": 1, "calls": 1}
    assert store["prompt_tokens"]["nvidia|z-ai/glm-5.3"]["first"] == 28084
    assert usage.latency("nvidia", "z-ai/glm-5.3")["first_s"] >= 0


def test_an_empty_fallback_variable_means_the_catalog_chain(monkeypatch):
    monkeypatch.setenv("GEMINI_FALLBACK_MODELS", "")
    assert GeminiProvider(model="gemini-3.5-flash-lite").fallback_chain() == [
        "gemini-3.5-flash-lite", "gemini-3.1-flash-lite"]


class TestThinking:
    def test_nothing_is_sent_by_default(self, monkeypatch, posts):
        monkeypatch.delenv("NVIDIA_THINKING", raising=False)
        monkeypatch.setenv("NVIDIA_MODEL", "z-ai/glm-5.3")
        _chat(CompatProvider("nvidia"))
        body = posts.sent[0]["json"]
        assert "reasoning_effort" not in body and "chat_template_kwargs" not in body

    def test_glm_takes_reasoning_effort(self, monkeypatch, posts):
        monkeypatch.setenv("NVIDIA_MODEL", "z-ai/glm-5.3")
        monkeypatch.setenv("NVIDIA_THINKING", "low")
        _chat(CompatProvider("nvidia"))
        assert posts.sent[0]["json"]["reasoning_effort"] == "low"

    def test_nemotron_takes_enable_thinking(self, monkeypatch, posts):
        monkeypatch.setenv("NVIDIA_MODEL", "nvidia/nemotron-3-ultra-550b-a55b")
        monkeypatch.setenv("NVIDIA_THINKING", "medium")
        _chat(CompatProvider("nvidia"))
        assert posts.sent[0]["json"]["chat_template_kwargs"] == {"enable_thinking": True, "medium_effort": True}

    def test_a_level_the_model_does_not_take_sends_nothing(self, monkeypatch, posts):
        monkeypatch.setenv("NVIDIA_MODEL", "deepseek-ai/deepseek-v4.1-flash")
        monkeypatch.setenv("NVIDIA_THINKING", "low")
        _chat(CompatProvider("nvidia"))
        body = posts.sent[0]["json"]
        assert "reasoning_effort" not in body and "chat_template_kwargs" not in body

    def test_openrouter_reasoning_only_for_models_that_support_it(self, monkeypatch, posts):
        monkeypatch.setenv("OPENROUTER_THINKING", "low")
        monkeypatch.setenv("OPENROUTER_FALLBACKS", "0")
        monkeypatch.setattr(openrouter, "discover", lambda **kw: {"models": [
            {"id": "thinks:free", "reasoning": True}, {"id": "plain:free", "reasoning": False}]})
        monkeypatch.setenv("OPENROUTER_MODEL", "thinks:free")
        _chat(CompatProvider("openrouter"))
        assert posts.sent[0]["json"]["reasoning"] == {"effort": "low"}
        monkeypatch.setenv("OPENROUTER_MODEL", "plain:free")
        _chat(CompatProvider("openrouter"))
        assert "reasoning" not in posts.sent[1]["json"]

    def test_the_level_is_in_the_session_log(self, monkeypatch):
        monkeypatch.setenv("NVIDIA_THINKING", "off")
        assert CompatProvider("nvidia").describe()["reasoning"] == "off"


def test_the_loop_gives_charts_the_arguments_and_leaves_the_record_alone(monkeypatch):
    from llm_agent.agent import tools as tools_mod
    from llm_agent.agent.providers.base import ToolCall

    def fake_rsa(**kwargs):
        return tools_mod._record("run_rsa", {"secure": True})
    calls = iter([ModelResponse(tool_calls=[ToolCall(name="run_rsa", args={"load_scaling_factor": 1.2})]),
                  ModelResponse(text="Done.")])

    class Provider:
        name, model = "nvidia", "m"

        def chat(self, messages, tools, system, on_event=None):
            return next(calls)
    records = []
    monkeypatch.setattr(loop_mod, "get_provider", lambda: Provider())
    monkeypatch.setattr(loop_mod, "get_system_prompt", lambda: "SYS")
    monkeypatch.setattr(loop_mod, "_append_to_log", lambda record: records.append(record))
    monkeypatch.setitem(loop_mod.TOOL_DISPATCH, "run_rsa", fake_rsa)
    loop_mod.run_agent_turn("Run it at 1.2", [], reflection="off")
    name, chart = tools_mod._last_tool_results[0]
    assert chart["_call_args"] == {"load_scaling_factor": 1.2}
    assert "_call_args" not in records[-1]["tool_results"][0]["result"]
