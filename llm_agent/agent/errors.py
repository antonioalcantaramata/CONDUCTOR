"""
errors.py — Error classification, handling, and user-friendly messaging for the LLM agent.

Provides utilities to:
- Classify errors into actionable categories
- Generate user-friendly error messages
- Extract retry delays from API error responses
"""

from __future__ import annotations

import re
from typing import NamedTuple


class ErrorClassification(NamedTuple):
    """Classification of an error with actionable metadata."""
    status: str  # "completed", "infra_error", "execution_error", "timeout", "quota_exhausted", "rate_limit"
    category: str  # For grouping similar errors
    is_retryable: bool
    suggested_wait_s: int | None
    user_message: str


# Error markers for infrastructure vs execution issues
INFRA_ERROR_MARKERS = (
    "rate limit",
    "resourceexhausted",
    "quota",
    "internal error encountered",
    "status': 'internal'",
    'status": "internal"',
    "temporarily overloaded",
    "500 internal",
    "deadline_exceeded",
    "deadline exceeded",
    "deadline expired before operation could complete",
    "504",
    "503",
    "429",
    "timed out",
    "timeout",
    "connection refused",
    "connection error",
    "backend is running",
    "failed to set start_timestamp",
    "all connection attempts failed",
    "name or service not known",
)


def extract_retry_delay(error_message: str) -> int | None:
    """Extract retry_delay in seconds from Google API error message.
    
    Looks for patterns like:
      retry_delay { seconds: 60 }
    
    Returns seconds to wait, or None if not found.
    """
    match = re.search(r"retry_delay\s*\{\s*seconds:\s*(\d+)", error_message)
    if match:
        return int(match.group(1))
    return None


def classify_error(error_message: str, error_type: str | None = None) -> ErrorClassification:
    """
    Classify an error into an actionable category.
    
    Args:
        error_message: The error text/exception message
        error_type: Optional error type name (e.g., "ResourceExhausted", "ServerError", "TimeoutException")
    
    Returns:
        ErrorClassification with status, category, retryable flag, and user message
    """
    error_lower = error_message.lower()
    
    # Daily quota exhaustion — non-retryable
    if "perday" in error_lower or "per_day" in error_lower or "daily quota" in error_lower:
        return ErrorClassification(
            status="quota_exhausted",
            category="quota_exhausted",
            is_retryable=False,
            suggested_wait_s=None,
            user_message=(
                "⚠️ **Gemini free-tier daily quota exhausted.**\n\n"
                "The free tier allows a limited number of requests per day. "
                "Options:\n"
                "- Wait until midnight Pacific time for the quota to reset.\n"
                "- Add a payment method at [Google AI Studio](https://aistudio.google.com) "
                "to get higher limits (pay-as-you-go is very cheap for this use case).\n"
                "- Switch to `gemini-2.5-flash-lite` by setting "
                "`GEMINI_MODEL=gemini-2.5-flash-lite` in your `.env` file."
            ),
        )
    
    # Per-minute rate limit (429)
    if "429" in error_message or "resourceexhausted" in error_lower or "rate limit" in error_lower:
        retry_delay = extract_retry_delay(error_message)
        suggested_wait = (retry_delay + 2) if retry_delay else 65
        return ErrorClassification(
            status="rate_limit",
            category="infra_error",
            is_retryable=True,
            suggested_wait_s=suggested_wait,
            user_message=(
                f"⚠️ **Gemini rate limit hit (429).**\n\n"
                f"Please wait approximately {suggested_wait}s and try again. "
                "This usually indicates high API demand or a sudden traffic spike."
            ),
        )
    
    # Model overload (503)
    if "503" in error_message or "temporarily overloaded" in error_lower or "overloaded" in error_lower:
        return ErrorClassification(
            status="infra_error",
            category="model_overload",
            is_retryable=True,
            suggested_wait_s=90,
            user_message=(
                "⚠️ **Gemini model temporarily overloaded (503).**\n\n"
                "The free tier shares capacity with many users — spikes are common. "
                "Please wait a moment and try again. "
                "Alternatively, set `GEMINI_MODEL=gemini-2.5-flash-lite` in `.env` "
                "for a lower-traffic model."
            ),
        )

    # Upstream Gemini internal errors (500)
    if "500 internal" in error_lower or "internal error encountered" in error_lower:
        return ErrorClassification(
            status="infra_error",
            category="model_internal_error",
            is_retryable=True,
            suggested_wait_s=20,
            user_message=(
                "⚠️ **Gemini internal server error (500).**\n\n"
                "The request reached Gemini, but the model backend failed before producing a response. "
                "This is usually transient and not caused by your grid data. "
                "Please retry in a moment."
            ),
        )
    
    # Timeout
    if (
        "timeout" in error_lower
        or "timed out" in error_lower
        or "deadline_exceeded" in error_lower
        or "deadline exceeded" in error_lower
        or "deadline expired before operation could complete" in error_lower
    ):
        return ErrorClassification(
            status="timeout",
            category="infra_error",
            is_retryable=True,
            suggested_wait_s=30,
            user_message=(
                "⚠️ **Gemini request timed out.**\n\n"
                "No response was received within the timeout window. "
                "This usually indicates a stuck upstream request rather than a backend-tool issue. "
                "Please retry the request, reduce concurrency, or switch to a lower-traffic model."
            ),
        )
    
    # Connection errors
    if any(marker in error_lower for marker in ("connection refused", "connection error", "name or service not known")):
        return ErrorClassification(
            status="infra_error",
            category="connection_error",
            is_retryable=True,
            suggested_wait_s=10,
            user_message=(
                "⚠️ **Connection error.**\n\n"
                "Could not connect to the Gemini API or backend service. "
                "Please check your internet connection and try again."
            ),
        )
    
    # Backend service issues
    if "backend is running" in error_lower or "all connection attempts failed" in error_lower:
        return ErrorClassification(
            status="infra_error",
            category="backend_unavailable",
            is_retryable=True,
            suggested_wait_s=15,
            user_message=(
                "⚠️ **Backend service unavailable.**\n\n"
                "The required backend service is not currently running or reachable. "
                "Please ensure all services are started and try again."
            ),
        )
    
    # Any other infra marker
    if any(marker in error_lower for marker in INFRA_ERROR_MARKERS):
        return ErrorClassification(
            status="infra_error",
            category="generic_infra",
            is_retryable=True,
            suggested_wait_s=10,
            user_message=(
                "⚠️ **Infrastructure error.**\n\n"
                f"Error: {error_message}\n\n"
                "This is a transient infrastructure issue. Please retry."
            ),
        )
    
    # Execution errors (tool errors, etc.)
    return ErrorClassification(
        status="execution_error",
        category="execution_error",
        is_retryable=False,
        suggested_wait_s=None,
        user_message=(
            "⚠️ **Execution error.**\n\n"
            f"Error: {error_message}"
        ),
    )
