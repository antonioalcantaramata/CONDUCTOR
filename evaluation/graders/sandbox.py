"""
Network proposals — how a turn that studied a proposal was assembled.

The integrity grader already reports a turn that mixed the base and a proposal
without being asked to, and an answer that did not name its proposal. This one
reports what a planner would object to in a what-if study, from the record
alone: an edit the agent built and the backend refused, a proposal with
violations the base does not have, and a proposal judged at a single moment
or without an N-1 screen. All are warnings, not failures — a quick question
can legitimately be answered at one moment, and a new violation is a finding
about the proposal rather than about the answer — but their rate is the
measurement the AI Agent Suggestion exists to move.

The rules are the ones the live suggestion button lights up on
(`recommender.proposal_gaps`), so the live system and the grade agree.
"""

from __future__ import annotations

from llm_agent.agent.recommender import proposal_gaps

from ..session_log import EXECUTION, INTERPRETATION, WARN, Finding

_GAP_CODES = (
    ("violation(s) the base does not have", "proposal_new_violations"),
    ("judged at a single moment", "proposal_single_moment"),
    ("no N-1 screen", "proposal_without_n1"),
)


def grade_sandbox(record: dict) -> list[Finding]:
    turn = record.get("turn", 0)
    findings: list[Finding] = []
    for call, entry in zip(record.get("tool_calls") or [], record.get("tool_results") or []):
        result = entry.get("result") if isinstance(entry, dict) else None
        if call.get("reused_from"):
            continue            # a repeat of an earlier call, answered from it: counted there
        if call.get("name") == "edit_proposal" and isinstance(result, dict) and result.get("error"):
            findings.append(Finding(turn, EXECUTION, WARN, "invalid_edit",
                                    f"edit_proposal was refused: {result.get('detail') or result['error']}"))
    for gap in proposal_gaps(record):
        for phrase, code in _GAP_CODES:
            if phrase in gap:
                findings.append(Finding(turn, INTERPRETATION, WARN, code, gap))
    return findings
