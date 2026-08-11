"""
loop.py — provider-agnostic agentic loop for the digital twin chat interface.

The loop owns turn control, tool dispatch, and session logging. Which LLM
answers is decided by `providers.get_provider()` (see `config.LLM_PROVIDER`);
this module never imports a vendor SDK directly.

Public API:
    run_agent_turn(user_message: str, history: list) -> tuple[str, list]
"""

from __future__ import annotations

import datetime
import json
import logging
import pathlib
import time
from typing import Callable

from . import tools as _tools_module
from .config import MAX_AGENT_TURNS
from .errors import classify_error
from .providers import (
    assistant_message,
    get_provider,
    tool_message,
    user_message as make_user_message,
)
from .system_prompt import get_system_prompt
from .tool_schemas import TOOL_DISPATCH, TOOLS

logger = logging.getLogger(__name__)

# Attempt tracking — per-conversation attempt counter
_attempt_counter: dict[str, int] = {}

# ---------------------------------------------------------------------------
# Session logging — single file, overwritten each new process start
# ---------------------------------------------------------------------------
session_log: list[dict] = []
_LOG_PATH = pathlib.Path(__file__).parent.parent / "session_logs" / "last_session.jsonl"
_log_initialized: bool = False


def _append_to_log(record: dict) -> None:
    """Write one turn record to the fixed session log file.

    The file is truncated (mode='w') on the very first write of the process
    so each new session starts clean. Subsequent turns are appended.
    """
    global _log_initialized
    session_log.append(record)
    try:
        _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        mode = "w" if not _log_initialized else "a"
        with _LOG_PATH.open(mode, encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        _log_initialized = True
    except Exception:  # noqa: BLE001
        logger.warning("Session log write failed — logging silently disabled.", exc_info=True)


def run_agent_turn(
    user_message: str,
    history: list,
    conversation_id: str | None = None,
    on_event: Callable[[str, dict], None] | None = None,
) -> tuple[str, list]:
    """
    Execute one user turn in the digital twin agentic loop.

    Args:
        user_message:     The user's raw message string.
        history:          Provider-neutral message list from previous turns
                          (see `providers.base` for the dict shapes).
        conversation_id:  Optional unique ID for grouping related turns (for attempt tracking).

    Returns:
        (final_text, updated_history)
        final_text is the assistant's last text response.
        updated_history is the full conversation including this turn.
    """
    # Track execution time for performance metrics
    turn_start_time = time.perf_counter()
    turn_status = "completed"  # default, may be overridden by error
    error_classification = "none"
    runner_error = ""

    # Generate conversation ID if not provided (for attempt counting)
    if conversation_id is None:
        conversation_id = f"session_{id(history)}"
    if conversation_id not in _attempt_counter:
        _attempt_counter[conversation_id] = 0
    _attempt_counter[conversation_id] += 1
    attempt_number = _attempt_counter[conversation_id]

    # 1. Clear per-turn tool results so Streamlit renders only this turn's charts.
    _tools_module._last_tool_results.clear()

    # 2. Append user message to history.
    history = list(history)  # shallow copy to avoid mutating caller's list
    history.append(make_user_message(user_message))

    # Per-turn log accumulators.
    turn_tool_calls: list[dict] = []
    turn_tool_results: list[dict] = []
    tool_error_count = 0

    # 3. Resolve the backend. The system prompt is read per turn so edits to it
    #    take effect without restarting the app.
    provider = get_provider()
    system_prompt = get_system_prompt()

    # 4. Agentic loop.
    final_text = ""
    turns_used = 0

    try:
        while turns_used < MAX_AGENT_TURNS:
            turns_used += 1
            logger.debug("Agent turn %d / %d", turns_used, MAX_AGENT_TURNS)

            # 4a. Call the model (the provider handles retry on transient errors).
            if on_event:
                on_event("llm_call", {"turn": turns_used})
            response = provider.chat(
                messages=history,
                tools=TOOLS,
                system=system_prompt,
                on_event=on_event,
            )

            # 4b. Append model response to history.
            history.append(assistant_message(response))

            # 4c. No tool calls → we have the final answer.
            if not response.wants_tools:
                final_text = response.text
                turn_status = "completed"
                break

            # 4d. Execute each tool call and append its result.
            for call in response.tool_calls:
                tool_name = call.name
                kwargs = call.args

                tool_fn = TOOL_DISPATCH.get(tool_name)
                if tool_fn is None:
                    result = {"error": f"Unknown tool: {tool_name}"}
                    tool_error_count += 1
                else:
                    try:
                        if on_event:
                            on_event("tool_start", {"name": tool_name, "args": kwargs})
                        result = tool_fn(**kwargs)
                        if on_event:
                            on_event("tool_done", {"name": tool_name, "result": result})
                    except Exception as exc:  # noqa: BLE001
                        logger.exception("Tool %s raised an exception", tool_name)
                        result = {"error": str(exc), "tool": tool_name}
                        tool_error_count += 1
                        if on_event:
                            on_event("tool_error", {"name": tool_name, "error": str(exc)})

                logger.debug("Tool %s result keys: %s", tool_name, list(result.keys()) if isinstance(result, dict) else type(result))

                turn_tool_calls.append({"name": tool_name, "args": kwargs})
                turn_tool_results.append({"name": tool_name, "result": result})
                model_result = _prepare_tool_result_for_model(tool_name, result)

                history.append(tool_message(tool_name, model_result, call_id=call.id))

        else:
            # MAX_AGENT_TURNS exceeded.
            warning = (
                f"⚠️ Agent reached the maximum of {MAX_AGENT_TURNS} turns "
                "without a final answer. The last partial result is shown above."
            )
            final_text = (final_text + "\n\n" + warning).strip()
            turn_status = "max_turns_exceeded"
            logger.warning("MAX_AGENT_TURNS (%d) exceeded.", MAX_AGENT_TURNS)

    except RuntimeError as exc:
        # User-friendly error from retry logic
        runner_error = str(exc)
        # Classify the error
        classification = classify_error(runner_error)
        error_classification = classification.category

        if turn_tool_results and classification.is_retryable:
            tool_names = ", ".join(call["name"] for call in turn_tool_calls) or "the requested tool"
            final_text = (
                "⚠️ The analysis data was generated successfully and the charts below are valid, "
                "but the model timed out while composing the written summary. "
                f"Retry if you want a narrative explanation of {tool_names}."
            )
            turn_status = "completed_with_warning"
            logger.warning("Agent turn completed with partial results after model timeout: %s", runner_error)
        else:
            final_text = runner_error
            turn_status = classification.status
            logger.error("Agent turn failed with %s: %s", turn_status, runner_error)
        
    except Exception as exc:  # noqa: BLE001
        # Unexpected error
        runner_error = str(exc)
        final_text = f"⚠️ **Unexpected error:** {runner_error}"
        turn_status = "execution_error"
        error_classification = "execution_error"
        
        logger.exception("Agent turn raised unexpected exception")

    # Calculate execution duration
    duration_s = round(time.perf_counter() - turn_start_time, 3)

    _append_to_log({
        "turn": len(session_log) + 1,
        "attempt_number": attempt_number,
        "conversation_id": conversation_id,
        "timestamp_utc": datetime.datetime.utcnow().isoformat() + "Z",
        "duration_s": duration_s,
        "status": turn_status,
        "error_classification": error_classification,
        "runner_error": runner_error,
        "user": user_message,
        "tool_calls": turn_tool_calls,
        "tool_call_count": len(turn_tool_calls),
        "tool_results": turn_tool_results,
        "tool_error_count": tool_error_count,
        "assistant": final_text,
    })

    return final_text, history


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _prepare_tool_result_for_model(tool_name: str, result):  # noqa: ANN001
    """Keep full tool payloads for charting, but return compact summaries to the model when needed."""
    if tool_name != "scan_rsa_over_time" or not isinstance(result, dict):
        return result

    timestamps = result.get("timestamps") or []
    violation_counts = result.get("violation_counts") or []
    min_voltage = result.get("min_voltage") or []
    max_voltage = result.get("max_voltage") or []
    max_line_loading = result.get("max_line_loading") or []
    max_trafo_loading = result.get("max_trafo_loading") or []

    violating_steps = [
        i for i, count in enumerate(violation_counts)
        if isinstance(count, (int, float)) and float(count) > 0
    ]
    first_violations = [timestamps[i] for i in violating_steps[:12] if i < len(timestamps)]
    worst_idx = max(range(len(violation_counts)), key=lambda i: float(violation_counts[i]), default=None)

    summary = {
        "window_start": timestamps[0] if timestamps else "",
        "window_end": timestamps[-1] if timestamps else "",
        "n_steps": len(timestamps),
        "any_violations": bool(result.get("any_violations", False)),
        "violating_step_count": len(violating_steps),
        "max_violation_count": max(violation_counts) if violation_counts else 0,
        "first_violation_timestamps": first_violations,
        "min_observed_voltage": min(min_voltage) if min_voltage else None,
        "max_observed_voltage": max(max_voltage) if max_voltage else None,
        "peak_line_loading_pct": max(max_line_loading) if max_line_loading else None,
        "peak_trafo_loading_pct": max(max_trafo_loading) if max_trafo_loading else None,
    }
    if worst_idx is not None and worst_idx < len(timestamps):
        summary["worst_timestamp"] = timestamps[worst_idx]
        summary["worst_timestamp_violation_count"] = violation_counts[worst_idx]
        if worst_idx < len(min_voltage):
            summary["worst_timestamp_min_voltage"] = min_voltage[worst_idx]
        if worst_idx < len(max_line_loading):
            summary["worst_timestamp_max_line_loading_pct"] = max_line_loading[worst_idx]
        if worst_idx < len(max_trafo_loading):
            summary["worst_timestamp_max_trafo_loading_pct"] = max_trafo_loading[worst_idx]

    return summary
