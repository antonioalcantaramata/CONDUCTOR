"""
Failure classification (llm_agent/agent/ratelimit.py): per-minute vs per-day
limits, exhausted credit, models that are gone, and backoff.
"""

import datetime as dt

from google.genai.errors import ClientError

from llm_agent.agent import ratelimit as rl


def _gemini(code, quota_id=None, extra=""):
    details = [{"violations": [{"quotaId": quota_id}]}] if quota_id else []
    return str(ClientError(code, {"error": {"code": code, "status": "RESOURCE_EXHAUSTED",
                                            "message": f"quota {extra}", "details": details}}))


class TestGemini:
    def test_a_daily_quota(self):
        f = rl.classify_gemini(_gemini(429, "GenerateRequestsPerDayPerProjectPerModel-FreeTier"), 429)
        assert f.kind == rl.PER_DAY and f.resets_at is not None and not f.waits

    def test_a_per_minute_limit_with_retry_delay(self):
        text = _gemini(429, "GenerateRequestsPerMinutePerProjectPerModel-FreeTier", "'retryDelay': '17s'")
        f = rl.classify_gemini(text, 429)
        assert f.kind == rl.PER_MINUTE and f.retry_after_s == 17 and f.waits

    def test_an_input_token_cap_and_its_value(self):
        text = ("429 RESOURCE_EXHAUSTED quotaMetric: generate_content_free_tier_input_token_count "
                "quotaId: 'GenerateContentInputTokensPerModelPerMinute-FreeTier' quotaValue: '16000'")
        f = rl.classify_gemini(text, 429)
        assert f.kind == rl.PER_MINUTE and f.metric == "input_tokens" and f.limit_value == 16000

    def test_a_retired_model(self):
        f = rl.classify_gemini("404 NOT_FOUND models/gemini-1.5-flash is not found for API version v1beta", 404)
        assert f.kind == rl.MODEL_GONE

    def test_the_daily_reset_is_midnight_pacific(self):
        now = dt.datetime(2026, 10, 14, 15, 30, tzinfo=rl.PACIFIC)
        assert rl.next_pacific_midnight(now) == dt.datetime(2026, 10, 15, 0, 0, tzinfo=rl.PACIFIC)


class TestOpenAICompatible:
    def test_openrouter_per_minute(self):
        f = rl.classify_http(429, '{"error":{"code":429,"message":"Rate limit exceeded"}}',
                             {"X-RateLimit-Limit": "20", "X-RateLimit-Reset": "1792000000000"}, "openrouter")
        assert f.kind == rl.PER_MINUTE and f.waits

    def test_openrouter_daily_from_the_limit_header(self):
        f = rl.classify_http(429, '{"error":{"message":"Rate limit exceeded"}}',
                             {"X-RateLimit-Limit": "50", "X-RateLimit-Reset": "1792000000000"}, "openrouter")
        assert f.kind == rl.PER_DAY and f.limit_value == 50
        assert f.resets_at == dt.datetime.fromtimestamp(1792000000, tz=dt.timezone.utc)

    def test_openrouter_daily_from_the_message(self):
        f = rl.classify_http(429, "Rate limit exceeded: free-models-per-day", {}, "openrouter")
        assert f.kind == rl.PER_DAY

    def test_credit_and_balance(self):
        assert rl.classify_http(402, "Insufficient credits", {}, "openrouter").kind == rl.QUOTA
        assert rl.classify_http(402, '{"error":{"message":"You have run out of balance"}}', {}, "deepseek").kind \
            == rl.QUOTA

    def test_models_that_are_gone(self):
        assert rl.classify_http(404, "404 page not found", {}, "nvidia").kind == rl.MODEL_GONE
        assert rl.classify_http(404, '{"error":{"message":"No endpoints found for x/y:free."}}', {},
                                "openrouter").kind == rl.MODEL_GONE
        assert rl.classify_http(410, "This model has been deprecated", {}, "nvidia").kind == rl.MODEL_GONE
        assert rl.classify_http(400, "x is not a valid model ID", {}, "openrouter").kind == rl.MODEL_GONE

    def test_a_data_policy_mismatch_is_not_a_missing_model(self):
        f = rl.classify_http(404, "No endpoints found matching your data policy", {}, "openrouter")
        assert f.kind == rl.DATA_POLICY

    def test_a_context_overflow_is_not_a_missing_model(self):
        assert rl.classify_http(400, "context length exceeded: not found room", {}, "nvidia").kind == rl.OTHER

    def test_a_model_forbidden_to_this_key_is_unavailable_not_a_bad_key(self):
        text = '{"error":{"message":"thinkingmachines/inkling:free is only available on agentic harnesses."}}'
        assert rl.classify_http(403, text, {}, "openrouter").kind == rl.MODEL_GONE
        assert rl.classify_http(403, '{"title":"Forbidden","detail":"Authorization failed"}', {}, "nvidia").kind \
            == rl.AUTH

    def test_auth_server_and_retry_after(self):
        assert rl.classify_http(401, "bad key", {}, "nvidia").kind == rl.AUTH
        f = rl.classify_http(503, "busy", {"retry-after": "12"}, "nvidia")
        assert f.kind == rl.SERVER and f.retry_after_s == 12

    def test_nvidia_per_minute(self):
        assert rl.classify_http(429, "Too Many Requests", {}, "nvidia").kind == rl.PER_MINUTE


class TestBackoff:
    def test_exponential_and_capped(self):
        assert [rl.backoff(a) for a in range(6)] == [5, 10, 20, 40, 80, 120]

    def test_never_shorter_than_the_servers_hint(self):
        assert rl.backoff(0, retry_after=30) == 30

    def test_alternatives_leave_out_the_current_option(self):
        text = rl.alternatives_text("nvidia_free")
        assert "NVIDIA" not in text and "Gemini" in text and "Ollama" in text
