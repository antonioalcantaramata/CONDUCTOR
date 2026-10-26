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
import hashlib
import json
import logging
import os
import pathlib
import time
from typing import Callable

from . import tools as _tools_module
from .reporting_notes import with_note as with_reporting_note
from .config import (
    DEEP_EXTRA_TURNS,
    MAX_AGENT_TURNS,
    READING_CHECK,
    REFLECTION_MODE,
    SESSION_LOG_DIR,
    SESSION_LOG_PATH,
    last_grid_constants_status,
)
from . import robustness as _robustness
from . import usage as _usage
from .errors import classify_error
from .reflection import (
    INFORM,
    OFF,
    Reflection,
    check_all,
    format_message,
    normalise_mode,
)
from .validators import (
    WARN,
    without_form_blanks,
    Issue,
    TurnConsistency,
    TurnOperatingPoint,
    annotate,
    compares_datasets,
    compares_networks,
    compares_power_factors,
    proposal_note,
    requested_limits,
    requested_timestamps,
    validate_call,
    validate_result,
)
from .provenance import is_clarifying_question
from .providers import (
    assistant_message,
    describe_provider,
    get_provider,
    tool_message,
    user_message as make_user_message,
)
from .system_prompt import DEEP_RESEARCH_NOTE, get_system_prompt
from .tool_registry import LONG_SERIES, changes_state, label
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
    session_log.append(record)
    _write_record(record)


def log_event(record: dict) -> None:
    """Append a non-turn record (e.g. an AI Agent Suggestion) to the session log.

    Written to the same file, so the sequence of what the operator saw stays in
    one place, but carrying a `kind` and kept out of the in-memory
    `session_log`: that list is read as "the turn records", and the audit panel
    takes its last entry to be the latest turn.
    """
    _write_record(record)


def _write_record(record: dict) -> None:
    global _log_initialized
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


def _turn_key(user_message: str, conversation_id: str, attempt: int) -> str:
    """Identity of one turn, for consumers that read the log after the fact.

    Not a hash of the whole record: the point is to answer "is this record the
    one for the question I just asked", which the prompt, the conversation and
    the attempt number settle.
    """
    digest = hashlib.sha1(
        f"{conversation_id}|{attempt}|{user_message}".encode("utf-8")
    ).hexdigest()
    return digest[:16]


def latest_record_for(user_message: str, conversation_id: str) -> dict | None:
    """The newest turn record, if it is the one for this question — else None.

    Matched by `turn_key`: the prompt, the conversation and the attempt the
    loop just ran. The prompt alone cannot tell a fresh record from a stale
    one when the same question is asked twice and the second turn aborted
    before logging. A record without a key (written before keys existed) is
    matched by its prompt.
    """
    record = session_log[-1] if session_log else None
    if not isinstance(record, dict):
        return None
    key = record.get("turn_key")
    if key:
        attempt = _attempt_counter.get(conversation_id, 0)
        return record if key == _turn_key(user_message, conversation_id, attempt) else None
    return record if record.get("user") == user_message else None


def run_agent_turn(
    user_message: str,
    history: list,
    conversation_id: str | None = None,
    on_event: Callable[[str, dict], None] | None = None,
    reflection: str | None = None,
    earlier_context: str | None = None,
    reading_check: bool | None = None,
    deep: bool = False,
) -> tuple[str, list]:
    """
    Execute one user turn in the digital twin agentic loop.

    Args:
        user_message:     The user's raw message string.
        history:          Provider-neutral message list from previous turns
                          (see `providers.base` for the dict shapes).
        conversation_id:  Optional unique ID for grouping related turns (for attempt tracking).
        reflection:       "off" | "warn" | "inform". Whether the agent is shown
                          its own provenance findings before the answer is
                          delivered. Defaults to the configured mode; see
                          `reflection.py`.
        reading_check:    Whether the turn's decisive studies are re-run under
                          the readings its question left open (robustness.py).
                          Defaults to the configured setting.
        earlier_context:  An earlier session the operator continues from
                          (`session_context.prompt_block`), added after the
                          system prompt as background.
        deep:             The operator switched on Deep Research for this
                          question: the model is asked to investigate in
                          depth if — it decides — the question is a why, and
                          has more steps to do it in.

    Returns:
        (final_text, updated_history)
        final_text is the assistant's last text response.
        updated_history is the full conversation including this turn.
    """
    # Track execution time for performance metrics
    # One clock for the whole answer and its parts: perf_counter is fine-grained
    # everywhere, where monotonic ticks every 15.6 ms on Windows before Python
    # 3.13 (a 50 ms study read as 46.9 ms, and the parts outgrew the total).
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
    turn_start = len(history) - 1
    model_calls = 0
    prompt_tokens: list[int] = []

    # Per-turn log accumulators.
    turn_tool_calls: list[dict] = []
    turn_tool_results: list[dict] = []
    tool_error_count = 0
    # Calls already made in this question, by tool and arguments: an identical
    # call is answered from here instead of running again (`_reuse`).
    turn_seen_calls: dict[str, tuple[int, object]] = {}
    # A failed call repeated after the model was told it would fail the same
    # way: the question stops there and is answered from what is known.
    told_it_failed = False
    stuck_on: str | None = None
    # Where the time went: the model, the studies, CONDUCTOR's own checks.
    timing = {"model_s": 0.0, "studies_s": 0.0, "checks_s": 0.0, "waited_for_rate_limit": False}
    # Tracks which operating point each tool actually ran at, so a chain
    # that silently mixes two of them is caught rather than presented as one.
    # Seeded with the moments the question names: comparing two of them is a
    # normal request, and warning about it told the model its own correct
    # behaviour was inconsistent.
    turn_consistency = TurnConsistency(
        expected=requested_timestamps(user_message),
        datasets_compared=compares_datasets(user_message),
        networks_compared=compares_networks(user_message),
    )
    # Carries an explicitly stated operating point, and the limits the user
    # set, across the turn's calls — so a later tool cannot silently fall back
    # to the simulation clock or to a default ceiling. Seeded from the prompt
    # because the first call of a turn has nothing earlier to inherit from,
    # and that is exactly where a week-long scan ran at the network's own
    # ceiling while every study after it ran at the 1.045 the operator asked
    # for. A field the question compares (proposal and base, forecast and
    # measurements, two power factors) is not carried: its omitted side is
    # the default one.
    turn_operating_point = TurnOperatingPoint(
        requested=requested_limits(user_message),
        networks_compared=compares_networks(user_message),
        datasets_compared=compares_datasets(user_message),
        power_factors_compared=compares_power_factors(user_message),
    )

    # 3. Resolve the backend. The system prompt is read per turn so edits to it
    #    take effect without restarting the app.
    provider = get_provider()
    system_prompt = get_system_prompt()
    if earlier_context:
        system_prompt = f"{system_prompt}\n\n{earlier_context}"
    if deep:
        system_prompt = f"{system_prompt}\n\n{DEEP_RESEARCH_NOTE}"
    # An investigation in depth runs several studies before it answers.
    max_turns = MAX_AGENT_TURNS + (DEEP_EXTRA_TURNS if deep else 0)

    # 4. Agentic loop.
    final_text = ""
    turns_used = 0
    # Reflection state. `reflected` guards the one-pass rule: the draft is
    # graded once, and whatever comes back is the answer.
    reflection_mode = normalise_mode(reflection if reflection is not None else REFLECTION_MODE)
    reflected = False
    reflection_record: Reflection | None = None
    # Where the draft the findings were about sits in `history`, when they
    # were shown: the draft and the findings are taken out of the history
    # returned to the caller (see the end of the turn).
    reflection_at: int | None = None
    # Deep Research: the one reminder of what an investigation in depth still
    # lacks, and where its draft sits (taken out of the returned history too).
    deep_reminded = False
    deep_draft_at: int | None = None

    try:
        while turns_used < max_turns and stuck_on is None:
            turns_used += 1
            logger.debug("Agent turn %d / %d", turns_used, max_turns)

            # 4a. Call the model (the provider handles retry on transient errors).
            if on_event:
                on_event("llm_call", {"turn": turns_used})
            waited = []

            def _watch(event: str, data: dict, _waited=waited) -> None:
                if event == "retry":
                    _waited.append(True)
                if on_event:
                    on_event(event, data)

            started = time.perf_counter()
            response = provider.chat(
                messages=_messages_for_model(history, turn_start),
                tools=TOOLS,
                system=system_prompt,
                on_event=_watch,
            )
            model_calls += 1
            timing["model_s"] += time.perf_counter() - started
            timing["waited_for_rate_limit"] = timing["waited_for_rate_limit"] or bool(waited)
            reported = _usage.prompt_tokens_from(response.raw)
            if reported:
                prompt_tokens.append(reported)
            if not waited:      # a latency measured across retries means nothing
                _usage.record_latency(getattr(provider, "name", "unknown"), getattr(provider, "model", ""),
                                      time.perf_counter() - started)

            # 4b. Append model response to history.
            history.append(assistant_message(response))

            # 4c. No tool calls → we have a draft answer.
            if not response.wants_tools:
                final_text = response.text
                turn_status = "completed"

                # 4c-0. Deep Research: an investigation about to answer
                # without its depth is reminded once of what it has not done.
                # It may do it, or say why it does not apply.
                missing = _deep_missing(turn_tool_calls, turn_tool_results) if deep and not deep_reminded else []
                if missing:
                    deep_reminded = True
                    deep_draft_at = len(history) - 1
                    if on_event:
                        on_event("deep_reminder", {"missing": missing})
                    history.append(make_user_message(
                        "Deep Research, before you answer: you have not yet " + "; ".join(missing) + ". Do them "
                        "now, or say in the answer why they do not apply here."))
                    continue

                # 4c-i. Optionally grade the draft and hand the findings back.
                # Only once per turn: a second pass could oscillate, and the
                # cost of an unbounded loop is not worth the marginal catch.
                if reflection_mode != OFF and not reflected:
                    reflected = True
                    draft_record = {
                        "user": user_message,
                        "grid": _grid_facts(),
                        "tool_calls": turn_tool_calls,
                        "tool_results": turn_tool_results,
                    }
                    checked = time.perf_counter()
                    findings, prov_report, conflicts, gaps = check_all(final_text, draft_record)
                    timing["checks_s"] += time.perf_counter() - checked
                    show = reflection_mode == INFORM and bool(findings or conflicts or gaps)
                    message = format_message(findings, conflicts, gaps) if show else ""
                    reflection_record = Reflection(
                        mode=reflection_mode,
                        draft=final_text,
                        findings=findings,
                        report=prov_report,
                        shown=show,
                        conflicts=conflicts,
                        gaps=gaps,
                        message=message,
                    )
                    if on_event:
                        on_event("reflection", {
                            "mode": reflection_mode,
                            "triggered": bool(findings or conflicts or gaps),
                            "shown": show,
                            "n_findings": len(findings),
                            "n_conflicts": len(conflicts),
                            "n_gaps": len(gaps),
                        })
                    if show:
                        # The findings enter as a user-role message: the agent
                        # is being told something about its draft, and is free
                        # to answer again, revise, or call a tool. Nothing here
                        # forces a change.
                        logger.info(
                            "Reflection surfaced %d grounding finding(s), %d "
                            "characterisation conflict(s) and %d gap(s) to the agent.",
                            len(findings), len(conflicts), len(gaps))
                        reflection_at = len(history) - 1
                        history.append(make_user_message(message))
                        continue

                break

            # 4d. Execute each tool call and append its result.
            repeated_a_failure = False
            for call in response.tool_calls:
                tool_name = call.name
                kwargs, inherited = turn_operating_point.resolve(
                    call.name, without_form_blanks(call.args or {}), _TOOL_ARGS.get(call.name)
                )
                if inherited:
                    logger.info("Inherited %s for %s from earlier in the turn.",
                                sorted(inherited), tool_name)

                key = _call_key(tool_name, kwargs)
                if key in turn_seen_calls:
                    # The same tool with the same arguments, nothing changed
                    # since: answer from the earlier call instead of running it
                    # again (a model repeating a study, or retrying one that
                    # failed for a reason a retry cannot fix).
                    same_as, earlier = turn_seen_calls[key]
                    result = _reuse(same_as, earlier)
                    if "error" in result:
                        tool_error_count += 1
                        turn_operating_point.rollback()
                        repeated_a_failure = True
                        if told_it_failed:
                            stuck_on = tool_name
                    logger.info("Reused %s: identical to call %d of this question.", tool_name, same_as)
                    if on_event:
                        on_event("tool_reused", {"name": tool_name, "same_as": same_as,
                                                 "failed": "error" in result})
                    call_record = {"name": tool_name, "args": kwargs, "reused_from": same_as}
                    if inherited:
                        call_record["inherited"] = inherited
                    turn_tool_calls.append(call_record)
                    turn_tool_results.append({"name": tool_name, "result": result})
                    history.append(tool_message(tool_name, result, call_id=call.id))
                    continue

                tool_fn = TOOL_DISPATCH.get(tool_name)
                checked = time.perf_counter()
                verdict = validate_call(tool_name, kwargs)
                timing["checks_s"] += time.perf_counter() - checked
                ran_s = None
                if tool_fn is None:
                    result = {"error": f"Unknown tool: {tool_name}"}
                    tool_error_count += 1
                    turn_operating_point.rollback()
                elif verdict.rejected:
                    # A validated solver given invalid parameters returns a
                    # confidently wrong answer, so impossible parameterisations
                    # are refused before execution. The model reads the
                    # structured error and corrects itself.
                    result = verdict.as_tool_error(tool_name)
                    tool_error_count += 1
                    turn_operating_point.rollback()
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
                            # A long study's progress, as it runs (tools._post_following).
                            _tools_module.progress_hook = (
                                lambda text, name=tool_name: on_event("tool_progress", {"name": name, "text": text}))
                        ran = time.perf_counter()
                        try:
                            result = tool_fn(**kwargs)
                        finally:
                            ran_s = time.perf_counter() - ran
                            timing["studies_s"] += ran_s
                            _tools_module.progress_hook = None
                        _attach_call_args(tool_name, result, kwargs)
                        if isinstance(result, dict) and "error" in result:
                            # Values that produced no result are not the turn's.
                            turn_operating_point.rollback()
                        # Check the result against itself before the model sees
                        # it; findings are attached, never substituted. The
                        # call's own parameter warnings come first: an unusual
                        # value the study ran with qualifies everything after.
                        checked = time.perf_counter()
                        parameters = [Issue(WARN, "parameter_warning", i.field, i.message)
                                      for i in verdict.warnings]
                        findings = validate_result(tool_name, result)
                        findings += turn_consistency.observe(tool_name, result)
                        timing["checks_s"] += time.perf_counter() - checked
                        if parameters or findings:
                            result = annotate(result, parameters + findings)
                            logger.warning("Integrity warnings from %s: %s", tool_name,
                                           [i.render() for i in parameters + findings])
                        if on_event and parameters:
                            on_event("parameter_warning", {
                                "name": tool_name,
                                "problems": [i.message for i in parameters],
                            })
                        if on_event and findings:
                            on_event("integrity_warning", {
                                "name": tool_name,
                                "problems": [i.message for i in findings],
                            })
                        if on_event:
                            on_event("tool_done", {"name": tool_name, "result": result,
                                                   "duration_s": round(ran_s, 3)})
                    except Exception as exc:  # noqa: BLE001
                        logger.exception("Tool %s raised an exception", tool_name)
                        result = {"error": str(exc), "tool": tool_name}
                        tool_error_count += 1
                        turn_operating_point.rollback()
                        if on_event:
                            on_event("tool_error", {"name": tool_name, "error": str(exc)})

                logger.debug("Tool %s result keys: %s", tool_name, list(result.keys()) if isinstance(result, dict) else type(result))

                call_record = {"name": tool_name, "args": kwargs}
                if inherited:
                    call_record["inherited"] = inherited
                if ran_s is not None:
                    call_record["duration_s"] = round(ran_s, 3)
                compact = _compact_for_model(tool_name, result)
                # How to report the study goes to the model only: guidance, not a figure to log.
                model_result = with_reporting_note(tool_name, compact)
                turn_tool_calls.append(call_record)
                turn_tool_results.append({"name": tool_name,
                                          "result": _with_model_summary(tool_name, result, compact)})
                _remember_call(turn_seen_calls, key, len(turn_tool_calls), tool_name, kwargs, result)

                history.append(tool_message(tool_name, model_result, call_id=call.id))

            told_it_failed = told_it_failed or repeated_a_failure

        else:
            # MAX_AGENT_TURNS reached, or the model repeated a failed call it
            # had been told would fail again: one last call asking for an
            # answer from the results so far, rather than ending on a bare
            # warning. Tools stay declared (some providers refuse tool history
            # without them); any tool call in the reply is ignored.
            if stuck_on:
                logger.warning("%s repeated after failing; asking for an answer from what is known.", stuck_on)
                why = (f"You repeated a `{stuck_on}` call that had already failed, after being told it "
                       "would fail the same way.")
                note = (f"⚠️ CONDUCTOR stopped this question: the model kept repeating a call that had "
                        f"already failed ({label(stuck_on)})")
            else:
                logger.warning("MAX_AGENT_TURNS (%d) reached; asking for an answer from what is known.",
                               max_turns)
                why = f"You have used all {max_turns} steps allowed for this question."
                note = (f"⚠️ CONDUCTOR used all {max_turns} steps allowed for one question "
                        f"(`DT_MAX_AGENT_TURNS`) before finishing")
            summary_text = ""
            try:
                started = time.perf_counter()
                response = provider.chat(
                    messages=_messages_for_model(history, turn_start) + [make_user_message(
                        f"{why} Do not call any more tools. Answer now from the tool results above: what "
                        "was found, and plainly what could not be done and why.")],
                    tools=TOOLS, system=system_prompt, on_event=on_event)
                model_calls += 1
                timing["model_s"] += time.perf_counter() - started
                summary_text = (response.text or "").strip()
            except Exception:  # noqa: BLE001 — the warning below still reaches the operator
                logger.warning("The final summary call failed.", exc_info=True)
            stopped = "repeated_failure" if stuck_on else "max_turns"
            if summary_text:
                final_text = (f"{summary_text}\n\n*{note.removeprefix('⚠️ ')}; this answer is built from the "
                              "results so far.*")
                turn_status = f"{stopped}_summarised"
            else:
                final_text = (final_text + "\n\n" + note + ". The last partial result is shown above.").strip()
                turn_status = "repeated_failure" if stuck_on else "max_turns_exceeded"

    except RuntimeError as exc:
        # User-friendly error from retry logic
        runner_error = str(exc)
        # Classify the error
        classification = classify_error(runner_error)
        error_classification = classification.category

        if turn_tool_results and classification.is_retryable:
            tool_names = ", ".join(call["name"] for call in turn_tool_calls) or "the requested tool"
            # Named from the classification: a quota or overload error used to
            # be reported as the model having "timed out".
            reason = classification.category.replace("_", " ")
            final_text = (
                "⚠️ The analysis data was generated successfully and the charts below are valid, "
                f"but the model could not compose the written summary ({reason}). "
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

    # An answer built on a network proposal names it (sandbox).
    note = proposal_note(final_text, turn_tool_results)
    if note:
        final_text = f"{final_text.rstrip()}\n\n{note}"

    # The reading check: the turn's decisive studies re-run under the readings
    # its question left open. Their verdicts, never the model's prose, are
    # compared; when one changes the conclusion, CONDUCTOR's note about it is
    # kept with the answer (the app shows it in the answer's card). Never fatal.
    reading_report = None
    if (READING_CHECK if reading_check is None else reading_check) and turn_status == "completed" \
            and turn_tool_results and not is_clarifying_question(final_text, turn_tool_calls):
        checked = time.perf_counter()
        try:
            reading_report = _robustness.check(
                {"user": user_message, "tool_calls": turn_tool_calls, "tool_results": turn_tool_results},
                grid=_grid_facts(), dispatch=TOOL_DISPATCH, on_event=on_event)
        except Exception:  # noqa: BLE001
            logger.warning("The reading check failed.", exc_info=True)
        timing["reading_s"] = time.perf_counter() - checked
        reading_note = _robustness.note(reading_report)
        if reading_note:
            final_text = f"{final_text.rstrip()}\n\n{reading_note}"
        if on_event and reading_report is not None:
            on_event("reading_check_done", {"headline": reading_report.headline(),
                                            "changes": len(reading_report.changes)})

    # Close the reflection record now the answer is settled. Both the draft and
    # the delivered answer are kept: the difference between them is the only
    # way to tell a revision from a rationalisation after the fact.
    if reflection_record is not None:
        reflection_record = reflection_record._replace(
            revised=reflection_record.shown,
            final=final_text,
        )

    # The findings were for this answer only. Left in the history they were
    # resent with every later call of the conversation, as if the operator had
    # written them; the draft they were about goes with them, so the history
    # holds the answer the operator received. Both stay in the turn record
    # (`reflection`: draft, findings, message).
    # Drafts the agent revised after a check or a reminder, with the message
    # that prompted the revision: the conversation keeps the final answer only.
    for at in sorted((i for i in (reflection_at, deep_draft_at) if i is not None), reverse=True):
        del history[at:at + 2]

    # How many calls this question took and what the provider reported, for
    # the launch screen's estimates (agent/usage.py). Never fatal.
    try:
        _usage.record_turn(getattr(provider, "name", "unknown"), getattr(provider, "model", ""), model_calls, prompt_tokens)
    except Exception:  # noqa: BLE001
        logger.debug("Could not record usage.", exc_info=True)

    _append_to_log({
        "turn": len(session_log) + 1,
        # The prompt this record belongs to. A consumer reading `session_log[-1]`
        # — the audit panel does — cannot otherwise tell a fresh record from a
        # stale one left behind when a turn aborted before logging, and would
        # then attribute one answer's evidence to a different question.
        "turn_key": _turn_key(user_message, conversation_id, attempt_number),
        "attempt_number": attempt_number,
        "conversation_id": conversation_id,
        # utcnow() is deprecated and naive; this keeps the identical wire format.
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "duration_s": duration_s,
        # Where those seconds went. The rest of the total (preparing messages,
        # writing the log) is small; the panel shows it as "other" when not.
        "timing": {"total_s": duration_s, "model_s": round(timing["model_s"], 3),
                   "studies_s": round(timing["studies_s"], 3),
                   "checks_s": round(timing["checks_s"] + timing.get("reading_s", 0.0), 3),
                   "model_calls": model_calls, "waited_for_rate_limit": timing["waited_for_rate_limit"]},
        "status": turn_status,
        # The earlier session this turn was given as background, if any.
        **({"earlier_context": earlier_context.split("\n", 1)[0].lstrip("# ")} if earlier_context else {}),
        "error_classification": error_classification,
        "runner_error": runner_error,
        # Which backend answered. Without this a log cannot say whether a turn
        # came from Gemini, a hosted OpenAI model at some reasoning level, or a
        # local one — so answers from different backends cannot be compared,
        # which is most of what the graders in `evaluation/` exist to do.
        **describe_provider(provider),
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
        "model_calls": model_calls,
        "prompt_tokens": prompt_tokens,
        "assistant": final_text,
        # Absent when reflection is off, so existing logs and the graders that
        # read them are unaffected.
        **({"reflection": reflection_record.as_record()}
           if reflection_record is not None else {}),
        # What re-running the studies under the question's other readings found.
        **({"reading_check": reading_report.as_record()} if reading_report is not None else {}),
        # Deep Research asked for, and whether the model judged the question
        # one (it investigated) or answered it as an analysis.
        **({"deep_research": {"requested": True, "investigated": any(
            c.get("name") in ("investigate", "test_prevention") for c in turn_tool_calls)}} if deep else {}),
    })

    return final_text, history


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _deep_missing(calls: list[dict], results: list[dict]) -> list[str]:
    """What an investigation in depth has not done yet; [] when the turn did
    not investigate (the model judged the question not a why) or did it all."""
    done = [(c.get("name"), c.get("args") or {}, e.get("result") if isinstance(e, dict) else None)
            for c, e in zip(calls, results)]
    ok = [(n, a, r) for n, a, r in done if isinstance(r, dict) and not r.get("error")]
    point = [r for n, a, r in ok if n == "investigate" and not a.get("recurring") and r.get("mode") != "recurring"
             and r.get("status") == "explained"]
    if not point:
        return []
    missing = []
    violated = any((r.get("symptom") or {}).get("violated") for r in point)
    if violated and not any(n == "test_prevention" for n, _a, _r in ok):
        missing.append("checked whether it could have been avoided and how much warning there was "
                       "(`test_prevention`)")
    if not any(n == "investigate" and (a.get("recurring") is True or r.get("mode") == "recurring")
               for n, a, r in ok):
        missing.append("looked for whether it recurs (`investigate` with `recurring`)")
    if any(r.get("coverage_note") for r in point) and not any(
            r.get("hypotheses_coverage_pct", 0) >= 75 for r in point):
        missing.append("refined your hypotheses until they explain most of the change")
    records = [r for n, _a, r in ok if n == "record_investigation" and r.get("status") == "recorded"]
    if not records:
        missing.append("recorded every hypothesis tested (`record_investigation`)")
    untested = [h for r in records for h in r.get("hypotheses") or [] if h.get("verdict") == "untested"]
    if not any(h.get("missing_kind") == "analysis" for h in untested):
        # Where the tools stop: the analysis that would decide the cause, and
        # what lies behind it — what CONDUCTOR should build or integrate next.
        missing.append("asked which analysis of the grid would confirm the cause or how serious it is and "
                       "CONDUCTOR cannot do (a stability margin, ramps over time, dynamics, its cost), and what "
                       "lies behind the cause outside the grid (why demand rose…); recorded each as untested, "
                       "with what would test it"
                       + (" (record again, with every hypothesis)" if records else ""))
    return missing


def _without_long_series(tool_name: str, result: dict) -> dict:
    """The answer-bearing part of a scan, without its per-tick series.

    A week-long worst-case scan carried ~9.5k tokens of series the model never
    needs to answer — and it is replayed with every later call of the
    conversation. On a year of data it would not fit the context at all. The
    charts and the guard layer still read the full result; only what the model
    is sent is reduced. Whatever stays is copied verbatim; the few figures
    computed here (`n_points`, `violations_per_tick_truncated`) are kept in
    the logged result too (`_with_model_summary`), so provenance can trace
    every figure the answer quotes.
    """
    out = {k: v for k, v in result.items() if k not in ("series", MODEL_SUMMARY_KEY)}
    series = result.get("series")
    if tool_name == "get_element_timeseries" and isinstance(series, dict):
        stamps = result.get("timestamps") or []
        summary = {}
        for name, values in series.items():
            pairs = [(v, t) for v, t in zip(values or [], stamps)
                     if isinstance(v, (int, float)) and not isinstance(v, bool)]
            if pairs:
                lo, hi = min(pairs), max(pairs)
                summary[name] = {"min": lo[0], "min_at": lo[1], "max": hi[0], "max_at": hi[1]}
        out["series_summary"] = summary
        out.pop("timestamps", None)
        out["n_points"] = len(stamps)
    if tool_name == "compare_networks" and isinstance(out.get("comparisons"), list):
        # Several proposals: each comparison of a scan carries its own series.
        out["comparisons"] = [{k: v for k, v in c.items() if k != "series"} if isinstance(c, dict) else c
                              for c in out["comparisons"]]
    if tool_name == "scan_scenarios":
        for scenario in out.get("scenarios") or []:
            if isinstance(scenario, dict):
                scenario.pop("series", None)
                scenario.pop("timestamps", None)
                ticks = scenario.get("violations_per_tick")
                if isinstance(ticks, list) and len(ticks) > 20:
                    scenario["violations_per_tick"] = ticks[:20]
                    scenario["violations_per_tick_truncated"] = len(ticks)
    return out


# ---------------------------------------------------------------------------
# Repeated calls within one question
# ---------------------------------------------------------------------------

def _call_key(tool_name: str, kwargs: dict) -> str:
    """One call, as tool and arguments (after the operating point is filled in)."""
    return tool_name + "|" + json.dumps(kwargs, sort_keys=True, default=str)


def _transient(result) -> bool:
    """A failure a retry might fix: no answer from the backend at all (a
    timeout, the backend restarting). A failure the backend answered (a
    refused argument, a power flow that did not converge) repeats exactly
    for identical arguments."""
    return isinstance(result, dict) and "endpoint" in result and result.get("status_code") is None


def _remember_call(seen: dict, key: str, order: int, tool_name: str, kwargs: dict, result) -> None:
    """Record a call that ran, for reuse later in the question — or forget
    everything when it changed the state the earlier results describe."""
    failed = isinstance(result, dict) and "error" in result
    if not failed and changes_state(tool_name, kwargs):      # its effect, in tool_registry
        seen.clear()
    elif not (failed and _transient(result)):
        seen[key] = (order, result)


def _reuse(same_as: int, earlier) -> dict:
    """What the model receives for a repeated call: a pointer to the earlier
    result (not the payload again), or the earlier failure with a note that
    repeating it will not help."""
    if isinstance(earlier, dict) and "error" in earlier:
        return {**earlier, "reused_from": same_as,
                "note": (f"This exact call already failed in this question (call {same_as}), and nothing has "
                         "changed since, so it would fail the same way again. Change the arguments or the "
                         "approach, or answer with what is known.")}
    return {"reused_from": same_as,
            "note": (f"Identical to call {same_as} of this question (same tool and arguments, nothing changed "
                     "since). It was not run again: its result above still applies.")}


# Tools whose result carries per-tick series the model is not sent (tool_registry).
_SERIES_TOOLS = LONG_SERIES


def _attach_call_args(tool_name: str, result, kwargs: dict) -> None:
    """Give the chart copy of this result the arguments it ran with, so the
    app can say on each chart what differs between runs of the same study in
    one answer (load scaling, limits, the outage, …). Only the entry kept for
    the charts changes; the model and the session record see the result as is."""
    if not isinstance(result, dict):
        return
    entries = _tools_module._last_tool_results
    for i in range(len(entries) - 1, -1, -1):
        name, recorded = entries[i]
        if recorded is result:
            entries[i] = (name, {**result, "_call_args": dict(kwargs)})
            return


_COMPACT_CHARS = 1500


def _compaction_on() -> bool:
    """Whether earlier questions' tool results are shortened in what is sent:
    CONDUCTOR_COMPACT_HISTORY=on|off|auto (default auto: on for free
    tiers, whose per-minute token budgets shrink as a conversation grows)."""
    mode = (os.environ.get("CONDUCTOR_COMPACT_HISTORY") or "auto").strip().lower()
    if mode in ("on", "1", "true"):
        return True
    if mode in ("off", "0", "false"):
        return False
    try:
        from . import catalog
        return catalog.option(catalog.active_option_id()).is_cloud_free
    except Exception:  # noqa: BLE001
        return False


def compact_history(history: list[dict], turn_start: int, max_chars: int = _COMPACT_CHARS) -> list[dict]:
    """A copy of `history` in which tool results from earlier questions (before
    `turn_start`) longer than `max_chars` are cut to their start, with a note.
    The current question's results are never touched, and the full results
    stay in the session record and the charts."""
    out = []
    for index, message in enumerate(history):
        if index < turn_start and message.get("role") == "tool":
            try:
                text = json.dumps(message.get("content"), default=str)
            except (TypeError, ValueError):
                text = str(message.get("content"))
            if len(text) > max_chars:
                message = {**message, "content": {
                    "earlier_result_shortened": text[:max_chars] + " …",
                    "note": "Shortened to save tokens (an earlier question). Re-run the tool for its full result.",
                }}
        out.append(message)
    return out


def _messages_for_model(history: list[dict], turn_start: int) -> list[dict]:
    return compact_history(history, turn_start) if _compaction_on() else history


# Where the logged result keeps what the loop computed for the model.
MODEL_SUMMARY_KEY = "summary_for_model"


def _computed_for_model(tool_name: str, view) -> dict:
    """The figures in what the model is sent that the loop computed rather
    than copied from the result: the whole summary of a security scan, a
    series' length and its extremes, the full length of a capped list."""
    if not isinstance(view, dict) or "error" in view:
        return {}
    if tool_name == "scan_rsa_over_time":
        return dict(view)
    out = {k: view[k] for k in ("n_points", "series_summary") if k in view}
    capped = [{"label": s.get("label"), "violations_per_tick_truncated": s["violations_per_tick_truncated"]}
              for s in view.get("scenarios") or []
              if isinstance(s, dict) and "violations_per_tick_truncated" in s]
    if capped:
        out["scenarios"] = capped
    return out


def _with_model_summary(tool_name: str, result, view):
    """The result as logged: unchanged, plus — when the model was sent a
    reduced view — the figures the loop computed for it, under
    `summary_for_model`.

    The answer checks read the logged result. A count the model was given
    but the result does not hold ("12 of 96 steps violated") was otherwise
    reported as a figure from nowhere."""
    if view is result or not isinstance(result, dict):
        return result
    computed = _computed_for_model(tool_name, view)
    return {**result, MODEL_SUMMARY_KEY: computed} if computed else result


def _numbers(values) -> list[float]:
    return [v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]


def _prepare_tool_result_for_model(tool_name: str, result):  # noqa: ANN001
    """What the model is sent: a compact summary when needed, and how to
    report the study (`reporting_notes`) — never in the logged result."""
    return with_reporting_note(tool_name, _compact_for_model(tool_name, result))


def _compact_for_model(tool_name: str, result):  # noqa: ANN001
    """Keep full tool payloads for charting, but return compact summaries to the model when needed."""
    if tool_name in _SERIES_TOOLS and isinstance(result, dict) and "error" not in result:
        import copy
        return _without_long_series(tool_name, copy.deepcopy(result))
    # A failed or refused scan goes to the model as it is: a summary of the
    # empty scan read as "no violations" and hid the error.
    if tool_name != "scan_rsa_over_time" or not isinstance(result, dict) or "error" in result:
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
        "reached_end_of_data": bool(result.get("reached_end_of_data", False)),
        "thresholds_used": result.get("thresholds_used"),
        "violating_step_count": len(violating_steps),
        "max_violation_count": max(violation_counts) if violation_counts else 0,
        "first_violation_timestamps": first_violations,
        # A step whose assessment returned no voltage has None (tools.scan_rsa_over_time).
        "min_observed_voltage": min(_numbers(min_voltage), default=None),
        "max_observed_voltage": max(_numbers(max_voltage), default=None),
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
