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
import os
import pathlib
import time
from typing import Callable

from . import tools as _tools_module
from .config import (
    MAX_AGENT_TURNS,
    SESSION_LOG_DIR,
    SESSION_LOG_PATH,
    last_grid_constants_status,
)
from .errors import classify_error
from .validators import (
    TurnConsistency,
    TurnOperatingPoint,
    annotate,
    requested_timestamps,
    validate_call,
    validate_result,
)
from .providers import (
    assistant_message,
    get_provider,
    tool_message,
    user_message as make_user_message,
)
from .system_prompt import get_system_prompt
from .tool_schemas import TOOL_DISPATCH, TOOLS

logger = logging.getLogger(__name__)


def _tool_argument_names() -> dict[str, set[str]]:
    """Arguments each tool accepts, so nothing is injected into a tool that
    would reject it."""
    from .providers.schema import to_json_schema_tools

    names: dict[str, set[str]] = {}
    for tool in to_json_schema_tools(TOOLS):
        fn = tool["function"]
        names[fn["name"]] = set((fn.get("parameters") or {}).get("properties") or {})
    return names


_TOOL_ARGS = _tool_argument_names()

# Attempt tracking — per-conversation attempt counter
_attempt_counter: dict[str, int] = {}

# ---------------------------------------------------------------------------
# Session logging — one append-only file per run
#
# The turn record is the only durable evidence of what the agent did: the
# prompt, the arguments each tool actually ran with, the full results, and the
# answer built from them. The offline graders in `evaluation/` read nothing
# else, and post-hoc diagnosis of a bad answer depends on it entirely.
#
# It was previously one fixed file truncated at process start, which meant
# every run destroyed the evidence of the one before it. Runs now write their
# own file; `last_session.jsonl` becomes a symlink to the newest so the
# familiar path still resolves.
# ---------------------------------------------------------------------------
session_log: list[dict] = []
_LOG_DIR = pathlib.Path(SESSION_LOG_DIR)
_LATEST = _LOG_DIR / "last_session.jsonl"
_log_path: pathlib.Path | None = None
_log_initialized: bool = False


def _session_log_path() -> pathlib.Path:
    """The file this process writes to, chosen once and then reused.

    The pid is part of the name because two runs started in the same second
    would otherwise interleave into one file.
    """
    global _log_path
    if _log_path is None:
        if SESSION_LOG_PATH:
            _log_path = pathlib.Path(SESSION_LOG_PATH)
        else:
            stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            _log_path = _LOG_DIR / f"session-{stamp}-{os.getpid()}.jsonl"
    return _log_path


def set_session_log(path: str | pathlib.Path | None) -> None:
    """Direct this process's turn records at a specific file.

    For batch runs, which want one log per catalog entry rather than one per
    process. Passing None restores the per-run default. Resets the in-memory
    record too, so a caller reading `session_log` sees only the current entry.
    """
    global _log_path, _log_initialized
    _log_path = pathlib.Path(path) if path else None
    # A caller naming its own file does not want the `last_session` pointer
    # moved to it; that path is for whoever is using the app.
    _log_initialized = path is not None
    session_log.clear()


def _point_latest_at(target: pathlib.Path) -> None:
    """Keep `last_session.jsonl` resolving to the current run.

    The new link is built under a temporary name first. Windows refuses
    symlinks without elevation, and doing it in this order means the refusal
    happens before anything on disk has been touched — the pointer is a
    convenience, and losing it must not cost a log. Where it does work, the
    replace is atomic.

    A real file at that path predates this scheme and holds somebody's
    session, so it is archived rather than overwritten.
    """
    if target == _LATEST:
        return

    staging = _LATEST.with_name(_LATEST.name + ".new")
    try:
        staging.unlink(missing_ok=True)
        staging.symlink_to(target.name)
    except OSError:
        logger.debug("Symlinks unavailable; `last_session.jsonl` not updated.", exc_info=True)
        return

    try:
        if _LATEST.exists() and not _LATEST.is_symlink():
            stamp = datetime.datetime.fromtimestamp(
                _LATEST.stat().st_mtime
            ).strftime("%Y%m%d-%H%M%S")
            _LATEST.rename(_LOG_DIR / f"session-{stamp}-archived.jsonl")
        os.replace(staging, _LATEST)
    except OSError:
        staging.unlink(missing_ok=True)
        logger.debug("Could not update last_session.jsonl.", exc_info=True)


def _grid_facts() -> dict:
    """Constants the prompt stated this turn, or empty if none were known."""
    status = last_grid_constants_status()
    return dict(status.values) if status is not None else {}


def _append_to_log(record: dict) -> None:
    """Append one turn record to this run's session log."""
    global _log_initialized
    session_log.append(record)
    try:
        path = _session_log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        if not _log_initialized:
            _point_latest_at(path)
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
    # Tracks which operating point each tool actually ran at, so a chain
    # that silently mixes two of them is caught rather than presented as one.
    # Seeded with the moments the question names: comparing two of them is a
    # normal request, and warning about it told the model its own correct
    # behaviour was inconsistent.
    turn_consistency = TurnConsistency(expected=requested_timestamps(user_message))
    # Carries an explicitly stated operating point across the turn's calls,
    # so a later tool cannot silently fall back to the simulation clock.
    turn_operating_point = TurnOperatingPoint()

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
                kwargs, inherited = turn_operating_point.resolve(
                    call.name, call.args, _TOOL_ARGS.get(call.name)
                )
                if inherited:
                    logger.info("Inherited %s for %s from earlier in the turn.",
                                sorted(inherited), tool_name)

                tool_fn = TOOL_DISPATCH.get(tool_name)
                verdict = validate_call(tool_name, kwargs)
                if tool_fn is None:
                    result = {"error": f"Unknown tool: {tool_name}"}
                    tool_error_count += 1
                elif verdict.rejected:
                    # A validated solver given invalid parameters returns a
                    # confidently wrong answer, so impossible parameterisations
                    # are refused before execution. The model reads the
                    # structured error and corrects itself.
                    result = verdict.as_tool_error(tool_name)
                    tool_error_count += 1
                    logger.warning("Rejected %s call: %s", tool_name,
                                   [i.render() for i in verdict.rejections])
                    if on_event:
                        on_event("tool_rejected", {
                            "name": tool_name,
                            "problems": [i.message for i in verdict.rejections],
                        })
                else:
                    try:
                        if on_event:
                            on_event("tool_start", {"name": tool_name, "args": kwargs})
                        result = tool_fn(**kwargs)
                        # Check the result against itself before the model sees
                        # it; findings are attached, never substituted.
                        integrity = validate_result(tool_name, result)
                        integrity += turn_consistency.observe(tool_name, result)
                        if integrity:
                            result = annotate(result, integrity)
                            logger.warning("Integrity warnings from %s: %s", tool_name,
                                           [i.render() for i in integrity])
                            if on_event:
                                on_event("integrity_warning", {
                                    "name": tool_name,
                                    "problems": [i.message for i in integrity],
                                })
                        if on_event:
                            on_event("tool_done", {"name": tool_name, "result": result})
                    except Exception as exc:  # noqa: BLE001
                        logger.exception("Tool %s raised an exception", tool_name)
                        result = {"error": str(exc), "tool": tool_name}
                        tool_error_count += 1
                        if on_event:
                            on_event("tool_error", {"name": tool_name, "error": str(exc)})

                logger.debug("Tool %s result keys: %s", tool_name, list(result.keys()) if isinstance(result, dict) else type(result))

                if verdict.warnings:
                    logger.info("Parameter warnings for %s: %s", tool_name,
                                [i.render() for i in verdict.warnings])

                call_record = {"name": tool_name, "args": kwargs}
                if inherited:
                    call_record["inherited"] = inherited
                turn_tool_calls.append(call_record)
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
        # utcnow() is deprecated and naive; this keeps the identical wire format.
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "duration_s": duration_s,
        "status": turn_status,
        "error_classification": error_classification,
        "runner_error": runner_error,
        "user": user_message,
        # The grid facts the system prompt carried this turn. Recorded because
        # they are a legitimate source for a figure in the answer: asked to
        # describe the network, the model correctly quoted its bus and line
        # counts from here, and a grounding check that only looked at tool
        # results called them fabricated.
        "grid": _grid_facts(),
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
