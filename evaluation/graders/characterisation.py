"""
Characterisation — is a figure described the way its own record describes it.

Thin wrapper, like the provenance grader: the matcher lives in
`llm_agent.agent.characterisation` so it can also run live, and this turns its
conflicts into findings.

Every conflict is a `fail`, and this is the most serious class the graders
report. A misattributed figure is real but attached to the wrong element; an
ungrounded one is visibly absent from the evidence. A mischaracterised one is
real, correctly attributed, correctly transcribed — and describes an action the
payload says cannot be taken. It is the case the additions log records as
caught by luck: `relief_mw: -6.2415` with `relief_mw_feasible: false`, reported
as "reduce by 6.24 MW", flagged only because the sign happened not to match.

The blind spot inherited from the matcher is worth restating wherever this is
reported: a wrong characterisation with no qualifier field behind it is
invisible here too.
"""

from __future__ import annotations

from llm_agent.agent.characterisation import Conflict, check_turn

from ..session_log import FAIL, INTERPRETATION, Finding


def grade_characterisation(record: dict) -> tuple[list[Finding], list[Conflict]]:
    """Findings about how one turn's answer describes its figures."""
    turn = record.get("turn", 0)
    conflicts = check_turn(record)
    findings = [
        Finding(turn, INTERPRETATION, FAIL, "mischaracterised_figure", conflict.render())
        for conflict in conflicts
    ]
    return findings, conflicts
