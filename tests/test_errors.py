import pytest

from llm_agent.agent import errors


class TestExtractRetryDelay:
    def test_extracts_seconds_from_retry_delay_pattern(self):
        msg = "429 error. retry_delay { seconds: 42 }"
        assert errors.extract_retry_delay(msg) == 42

    def test_returns_none_when_pattern_absent(self):
        assert errors.extract_retry_delay("some unrelated error") is None


class TestClassifyError:
    def test_daily_quota_exhausted_is_not_retryable(self):
        result = errors.classify_error("You exceeded your current daily quota, PerDay limit.")
        assert result.status == "quota_exhausted"
        assert result.is_retryable is False

    def test_429_rate_limit_uses_retry_delay_plus_buffer(self):
        result = errors.classify_error("429 ResourceExhausted. retry_delay { seconds: 10 }")
        assert result.status == "rate_limit"
        assert result.is_retryable is True
        assert result.suggested_wait_s == 12

    def test_429_without_retry_delay_uses_default_wait(self):
        result = errors.classify_error("429 rate limit hit")
        assert result.suggested_wait_s == 65

    def test_503_overload_is_retryable(self):
        result = errors.classify_error("Error 503: model temporarily overloaded")
        assert result.status == "infra_error"
        assert result.category == "model_overload"
        assert result.is_retryable is True

    def test_500_internal_error(self):
        result = errors.classify_error("500 Internal error encountered")
        assert result.category == "model_internal_error"

    def test_timeout_is_retryable(self):
        result = errors.classify_error("Request timed out after 30s")
        assert result.status == "timeout"
        assert result.is_retryable is True

    def test_connection_error(self):
        result = errors.classify_error("Connection refused by host")
        assert result.category == "connection_error"

    def test_backend_unavailable(self):
        result = errors.classify_error("Please make sure the backend is running")
        assert result.category == "backend_unavailable"

    def test_unrecognized_error_is_execution_error_and_not_retryable(self):
        result = errors.classify_error("KeyError: 'foo'")
        assert result.status == "execution_error"
        assert result.is_retryable is False
        assert "KeyError: 'foo'" in result.user_message

    @pytest.mark.parametrize("marker", list(errors.INFRA_ERROR_MARKERS))
    def test_every_infra_marker_is_classified_as_retryable_or_more_specific(self, marker):
        result = errors.classify_error(f"boom: {marker}")
        assert result.status in {
            "quota_exhausted", "rate_limit", "infra_error", "timeout",
        }
