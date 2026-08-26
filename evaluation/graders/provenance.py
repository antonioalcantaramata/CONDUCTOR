"""
Numeric provenance — every figure in the answer traced back to the evidence.

Thin wrapper: the matcher itself lives in `llm_agent.agent.provenance` so it
can also run live, and this turns its verdicts into findings.

Severity split. An ungrounded figure is a `fail`: it appears in no tool result,
no call argument and nothing the user wrote, which is the definition of a
number the model produced itself. A misattributed one is also a `fail`, and
the more dangerous of the two: the figure is real and correctly transcribed,
and it describes a different element than the sentence containing it. An
imprecise one is a `warn` — right figure, wrong precision.
"""

from __future__ import annotations

from llm_agent.agent.provenance import ProvenanceReport, check_turn

from ..session_log import FAIL, INTERPRETATION, WARN, Finding


def grade_provenance(record: dict) -> tuple[list[Finding], ProvenanceReport]:
    """Findings about the numbers in one turn's answer, plus the full report.

    The report is returned alongside because the interesting quantity is the
    rate over many turns, not the individual findings.
    """
    turn = record.get("turn", 0)
    report = check_turn(record)
    findings: list[Finding] = []

    for verdict in report.misattributed:
        findings.append(Finding(
            turn, INTERPRETATION, FAIL, "misattributed_figure",
            f"stated about {verdict.subject}: {verdict.render()}",
        ))

    for verdict in report.ungrounded:
        findings.append(Finding(
            turn, INTERPRETATION, FAIL, "ungrounded_figure",
            verdict.render(),
        ))

    for verdict in report.imprecise:
        findings.append(Finding(
            turn, INTERPRETATION, WARN, "imprecise_figure",
            verdict.render(),
        ))

    return findings, report
