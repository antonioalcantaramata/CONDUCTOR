"""
Composition integrity — did every tool in this turn run at the point the
answer claims to describe?

This is the grader the pass-rate catalog has no equivalent of. Every figure in
a turn can be individually correct while the turn as a whole describes
something that never happened, because two tools ran against two different
operating points and the answer presented them as one assessment. Nothing in
the text reveals it; only comparing the results against each other does.

No catalog is needed. The evidence is already in the record: what the user
asked for, what each tool was called with, and what operating point each result
came back stamped with.
"""

from __future__ import annotations

from llm_agent.agent.tool_schemas import _POINT_IN_TIME_TOOLS
from llm_agent.agent.validators import (
    TurnConsistency,
    operating_context,
    requested_timestamps,
)

from ..session_log import (
    EXECUTION,
    FAIL,
    INTERPRETATION,
    PARAMETERISATION,
    STATE,
    WARN,
    Finding,
)

# Tools that move the simulation cursor, which every later turn then reads.
# `scan_rsa_over_time` is a composite of advance + assess, so it moves it too.
# Taken from the dispatch table rather than restated here would be better, but
# the property is not recorded there; the set is small and its members say so
# in their own docstrings.
_CURSOR_TOOLS = {"advance_timestamp", "scan_rsa_over_time"}

# Codes the replay below produces itself. A turn recorded while the live guard
# was active carries the same findings inside the result payload, and counting
# both reports one problem twice — which reads as two.
_REPLAYED_CODES = {"inconsistent_context", "inconsistent_thresholds"}


def _agrees(actual: str, requested: str) -> bool:
    """A date-only request is satisfied by any time on that date."""
    return actual.startswith(requested) if len(requested) == 10 else actual == requested


def grade_composition(record: dict) -> list[Finding]:
    """Findings about how one turn was assembled."""
    turn = record.get("turn", 0)
    findings: list[Finding] = []
    results = record.get("tool_results") or []

    # 1. Tools in the same turn disagreeing with each other. This replays the
    #    live guard, so a log written before the guard existed is graded by the
    #    same rule as one written after it.
    requested = requested_timestamps(record.get("user", ""))
    consistency = TurnConsistency(expected=requested)
    for entry in results:
        for issue in consistency.observe(entry.get("name", "tool"), entry.get("result")):
            findings.append(Finding(
                turn, PARAMETERISATION, FAIL, issue.code,
                f"{issue.field}: {issue.message}",
            ))

    # 2. Tools agreeing with each other but not with the request. A turn where
    #    every call fell back to the simulation clock is internally consistent
    #    and still answers a different question than the one asked.
    if len(requested) == 1:
        wanted = next(iter(requested))
        for entry in results:
            actual = operating_context(entry.get("result")).get("timestamp")
            if actual and not _agrees(actual, wanted):
                findings.append(Finding(
                    turn, PARAMETERISATION, FAIL, "requested_point_not_used",
                    f"{entry.get('name')} ran at {actual}; the question asked for {wanted}.",
                ))

    # 3. Constraints the model dropped and the loop had to restore. Not a wrong
    #    answer — the mechanism worked — but it is the measurement of how often
    #    the model would have got it wrong unaided, which is the number the
    #    paper needs.
    for call in record.get("tool_calls") or []:
        inherited = call.get("inherited")
        if inherited:
            fields = ", ".join(sorted(inherited)) if isinstance(inherited, dict) else str(inherited)
            findings.append(Finding(
                turn, PARAMETERISATION, WARN, "constraint_inherited",
                f"{call.get('name')} omitted {fields}; the turn's operating point was applied for it.",
            ))

    # 4. Whatever the live guards already flagged, surfaced rather than left
    #    buried in the result payload.
    for entry in results:
        result = entry.get("result")
        if not isinstance(result, dict):
            continue
        for warning in result.get("_integrity_warnings") or []:
            code = warning.get("code", "integrity_warning")
            if code in _REPLAYED_CODES:
                continue
            findings.append(Finding(
                turn, PARAMETERISATION, WARN, code,
                f"{entry.get('name')}: {warning.get('message', '')}",
            ))

    # 5. Calls that did not produce a result at all.
    for entry in results:
        result = entry.get("result")
        if isinstance(result, dict) and "error" in result:
            findings.append(Finding(
                turn, EXECUTION, WARN, "tool_error",
                f"{entry.get('name')}: {result['error']}",
            ))

    status = record.get("status")
    if status == "completed_with_warning":
        findings.append(Finding(
            turn, EXECUTION, WARN, str(record.get("error_classification") or "completed_with_warning"),
            str(record.get("runner_error") or status),
        ))
    elif status not in (None, "completed"):
        findings.append(Finding(
            turn, EXECUTION, FAIL, str(record.get("error_classification") or "turn_failed"),
            str(record.get("runner_error") or status),
        ))

    # 6. Turns that moved the simulation cursor. Reported as a fact rather
    #    than a verdict: advancing is correct when the user asked for it, and
    #    wrong when a read-only question quietly changed what the next
    #    question will see. Without a catalog the intent is unknown, so the
    #    finding records what happened and whether it was avoidable.
    findings += _cursor_mutations(turn, record)

    # 7. An answer with no tool behind it. The agent is allowed to decline and
    #    to answer questions about itself, so this is a flag to look at, not a
    #    verdict — an unsupported request correctly refused lands here too.
    if not results and (record.get("assistant") or "").strip():
        findings.append(Finding(
            turn, INTERPRETATION, WARN, "no_tool_evidence",
            "The turn answered without calling any tool.",
        ))

    return findings


def _cursor_mutations(turn: int, record: dict) -> list[Finding]:
    """Did this turn move the simulation clock, and did it need to?"""
    calls = record.get("tool_calls") or []
    movers = [c for c in calls if c.get("name") in _CURSOR_TOOLS]
    if not movers:
        return []

    # Where it landed. The tool wrapper renames the endpoint's `new_timestamp`
    # to `current_timestamp`, and a grader reading only one of them reports a
    # cursor move with no destination — which is what it did on first use.
    landed = []
    for entry in record.get("tool_results") or []:
        if entry.get("name") not in _CURSOR_TOOLS or not isinstance(entry.get("result"), dict):
            continue
        result = entry["result"]
        moved_to = result.get("current_timestamp") or result.get("new_timestamp")
        if moved_to and str(moved_to) not in landed:
            landed.append(str(moved_to))

    detail = (
        f"{', '.join(sorted({c['name'] for c in movers}))} moved the simulation cursor"
        + (f" to {', '.join(landed)}" if landed else "")
        + "."
    )

    # A read-only path existed if the analysis that followed was done by tools
    # that accept an operating point, and none of them named one — the turn
    # relied on the cursor it had just moved instead of asking directly.
    reliant = sorted({
        c["name"] for c in calls
        if c.get("name") in _POINT_IN_TIME_TOOLS and not (c.get("args") or {}).get("timestamp")
    })
    if reliant:
        detail += (
            f" {', '.join(reliant)} then read that cursor rather than naming an operating "
            "point, so the same answer was available without changing state."
        )

    return [Finding(turn, STATE, WARN, "simulation_cursor_moved", detail)]
