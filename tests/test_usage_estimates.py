"""
What one question costs and what a tier allows (llm_agent/agent/usage.py):
overhead, the compatibility check, usage estimates, and the local store of
calls per question, daily calls and latency.
"""

import types

import pytest

from llm_agent.agent import usage


@pytest.fixture(autouse=True)
def own_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("CONDUCTOR_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("CONDUCTOR_CALLS_PER_QUESTION", raising=False)


class TestOverhead:
    def test_characters_over_four(self):
        assert usage.overhead_tokens("a" * 400, "b" * 401) == 201

    def test_reported_prompt_tokens_any_shape(self):
        assert usage.prompt_tokens_from({"usage": {"prompt_tokens": 28100}}) == 28100
        assert usage.prompt_tokens_from({"usage": {"input_tokens": 9}}) == 9
        assert usage.prompt_tokens_from({"prompt_eval_count": 7}) == 7
        gem = types.SimpleNamespace(usage_metadata=types.SimpleNamespace(prompt_token_count=28084))
        assert usage.prompt_tokens_from(gem) == 28084
        ant = types.SimpleNamespace(usage_metadata=None, usage=types.SimpleNamespace(input_tokens=5))
        assert usage.prompt_tokens_from(ant) == 5
        assert usage.prompt_tokens_from(None) is None and usage.prompt_tokens_from({"x": 1}) is None


class TestCompatibility:
    def test_a_tpm_below_the_overhead_is_incompatible(self):
        ok, why = usage.compatibility(262144, {"tpm": 16000}, 28000)
        assert not ok and "16,000" in why and "tokens-per-minute" in why

    def test_a_per_request_cap_below_the_overhead_is_incompatible(self):
        ok, why = usage.compatibility(131072, {"per_request_input_cap": 8000}, 28000)
        assert not ok and "per-request" in why

    def test_a_small_context_window_is_incompatible(self):
        assert not usage.compatibility(16384, {}, 28000)[0]

    def test_enough_everywhere_is_compatible(self):
        assert usage.compatibility(1048576, {"tpm": 250000}, 28000) == (True, None)

    def test_effective_max_per_request(self):
        assert usage.effective_max_per_request(1048576, {"tpm": 250000}) == (
            250000, "one request cannot exceed the tokens-per-minute limit")
        assert usage.effective_max_per_request(1048576, {}) == (1048576, None)
        assert usage.effective_max_per_request(None, {}) == (None, None)


class TestEstimates:
    def test_questions_per_day(self):
        est = usage.estimate({"rpm": 15, "tpm": 250000, "rpd": 500}, 4, overhead=28000)
        assert est["kind"] == "per_day" and est["value"] == 125
        assert "Tokens per minute are the tighter limit" in est["text"]

    def test_few_questions_a_day_on_flash(self):
        assert usage.estimate({"rpm": 5, "rpd": 20}, 4, overhead=28000)["value"] == 5

    def test_openrouter_with_and_without_credit(self):
        limits = {"rpm": 20, "rpd": 50, "rpd_with_credit": 1000}
        assert usage.estimate(limits, 4, overhead=28000)["value"] == 12
        assert usage.estimate(limits, 4, overhead=28000, credit=True)["value"] == 250

    def test_time_per_question_when_only_rpm_is_known(self):
        est = usage.estimate({"rpm": 2}, 4, overhead=28000)
        assert est["kind"] == "time_per_question" and est["value"] == 120

    def test_trial_from_the_full_grant(self):
        option = {"tier": "free_trial", "trial": {"grant_tokens": 5_000_000}}
        est = usage.estimate({}, 4, overhead=25000, option=option)
        assert est["kind"] == "trial_total"
        assert est["value"] == 5_000_000 // int(25000 * 1.15 * 4)
        assert "unconfirmed" in est["text"]

    def test_trial_from_the_remaining_balance(self):
        option = {"tier": "free_trial", "trial": {"grant_tokens": 5_000_000}}
        est = usage.estimate({}, 4, overhead=25000, option=option, balance_usd=0.3, price_per_mtok=0.30)
        assert est["value"] == 1_000_000 // int(25000 * 1.15 * 4) and "$0.30" in est["text"]

    def test_paid_and_local(self):
        assert usage.estimate({}, 4, overhead=1, option={"plan_limited": True})["kind"] == "plan"
        assert usage.estimate({}, 4, overhead=1, option={"tier": "local"})["kind"] == "hardware"
        assert usage.estimate({}, 4, overhead=1)["kind"] == "unknown"


class TestTheStore:
    def test_calls_per_question_default_average_and_override(self, monkeypatch):
        assert usage.calls_per_question() == 4.0
        usage.record_turn("google", "m", 3)
        usage.record_turn("google", "m", 5)
        assert usage.calls_per_question() == 4.0 and usage.load_store()["turns"] == {"n": 2, "calls": 8}
        usage.record_turn("google", "m", 8)
        assert usage.calls_per_question() == 5.33
        monkeypatch.setenv("CONDUCTOR_CALLS_PER_QUESTION", "6")
        assert usage.calls_per_question() == 6.0

    def test_a_turn_with_no_call_is_not_counted(self):
        usage.record_turn("google", "m", 0)
        assert usage.calls_per_question() == 4.0

    def test_calls_today_and_prompt_tokens(self):
        usage.record_turn("nvidia", "z-ai/glm-5.3", 4, [28100, 29000, 30500, 31000])
        usage.record_calls("nvidia", "z-ai/glm-5.3", 1)
        assert usage.calls_today("nvidia", "z-ai/glm-5.3") == 5
        assert usage.calls_today("nvidia", "other") == 0
        assert usage.measured_prompt_tokens() == 28100
        assert usage.tokens_per_question(25000, 4) == round((28100 + 29000 + 30500 + 31000) / 4) * 4

    def test_latency_keeps_the_first_call_apart(self):
        usage.record_latency("nvidia", "z-ai/glm-5.3", 42.04)
        usage.record_latency("nvidia", "z-ai/glm-5.3", 9.96)
        lat = usage.latency("nvidia", "z-ai/glm-5.3")
        assert lat["first_s"] == 42.0 and lat["last_s"] == 10.0

    def test_a_corrupt_store_is_ignored(self, tmp_path):
        (tmp_path / "usage.json").write_text("{not json", encoding="utf-8")
        assert usage.load_store() == {} and usage.calls_per_question() == 4.0


def test_advisor_calls_count_against_today():
    from llm_agent.agent import recommender
    from llm_agent.agent.providers.base import ModelResponse

    class Provider:
        name, model = "openrouter", "x:free"

        def chat(self, messages, tools, system, on_event=None):
            return ModelResponse(text="{}")
    recommender._call("SYS", "request", Provider())
    recommender.suggest([], {}, [], {}, provider=Provider())
    assert usage.calls_today("openrouter", "x:free") == 2
