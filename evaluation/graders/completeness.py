"""
Completeness — did the answer contain the kind of thing the question asked for.

Thin wrapper, like the others: the matcher lives in
`llm_agent.agent.completeness` so it can also run live.

Every finding is a **warning**, and this one is not a severity judgement to be
revisited later. A missing quantity is frequently the correct answer: the tool
errored, the study does not support the question, the bisection did not
converge. Reporting an absence as a failure would push the agent toward
inventing a figure, which is the failure this whole layer exists to prevent.

What it is for is the other case — a figure that was available and went
missing. Observed live: asked for a percentage, the agent drafted 80.3%, was
shown a grounding finding, and delivered two raw operands with no percentage at
all. Every other check passed on that answer.

For the reflection experiment this is also the field that separates *defended*
from *withdrew*. Both change the answer; only one removes what was asked for.
"""

from __future__ import annotations

from llm_agent.agent.completeness import Gap, check_turn

from ..session_log import INTERPRETATION, WARN, Finding


def grade_completeness(record: dict) -> tuple[list[Finding], list[Gap]]:
    """Findings about what one turn's answer left out."""
    turn = record.get("turn", 0)
    gaps = check_turn(record)
    findings = [
        Finding(turn, INTERPRETATION, WARN, "answer_incomplete", gap.render())
        for gap in gaps
    ]
    return findings, gaps
