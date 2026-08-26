"""Aggregating findings into something readable."""

from __future__ import annotations

import pathlib
from typing import NamedTuple

from .graders.integrity import grade_composition
from .graders.provenance import grade_provenance
from .session_log import FAIL, Finding, load_records


class TurnGrade(NamedTuple):
    turn: int
    prompt: str
    n_tool_calls: int
    findings: tuple[Finding, ...]
    n_claims: int
    n_grounded: int
    n_imprecise: int
    n_misattributed: int
    n_ungrounded: int

    @property
    def failed(self) -> bool:
        return any(f.severity == FAIL for f in self.findings)


def grade_record(record: dict) -> TurnGrade:
    """Every grader, over one turn."""
    provenance_findings, report = grade_provenance(record)
    findings = grade_composition(record) + provenance_findings

    return TurnGrade(
        turn=record.get("turn", 0),
        prompt=(record.get("user") or "").strip(),
        n_tool_calls=len(record.get("tool_calls") or []),
        findings=tuple(findings),
        n_claims=len(report.verdicts),
        n_grounded=len(report.grounded),
        n_imprecise=len(report.imprecise),
        n_misattributed=len(report.misattributed),
        n_ungrounded=len(report.ungrounded),
    )


def grade_log(path: str | pathlib.Path) -> list[TurnGrade]:
    return [grade_record(r) for r in load_records(path)]


def _truncate(text: str, width: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= width else text[: width - 1] + "…"


def render_report(grades: list[TurnGrade], source: str = "") -> str:
    """A short human-readable summary. Machine-readable output is the grades."""
    if not grades:
        return f"{source}: no turns to grade."

    lines: list[str] = []
    if source:
        lines.append(f"{source} — {len(grades)} turn(s)")
        lines.append("")

    claims = sum(g.n_claims for g in grades)
    grounded = sum(g.n_grounded for g in grades)
    imprecise = sum(g.n_imprecise for g in grades)
    misattributed = sum(g.n_misattributed for g in grades)
    ungrounded = sum(g.n_ungrounded for g in grades)
    failed = [g for g in grades if g.failed]

    for grade in grades:
        marker = "FAIL" if grade.failed else ("warn" if grade.findings else "  ok")
        lines.append(
            f"  [{marker}] turn {grade.turn:>2}  "
            f"{grade.n_tool_calls} call(s), {grade.n_claims} figure(s)  "
            f"{_truncate(grade.prompt, 58)}"
        )
        for finding in grade.findings:
            lines.append(f"          [{finding.severity}] {finding.code}: {_truncate(finding.detail, 100)}")

    traceable = grounded + imprecise
    share = f"{traceable / claims:.1%}" if claims else "n/a"
    lines.append("")
    lines.append(
        f"  figures {claims} — traceable {share} "
        f"({grounded} exact, {imprecise} imprecise, "
        f"{misattributed} misattributed, {ungrounded} ungrounded)"
    )
    lines.append(f"  turns   {len(grades)} — {len(failed)} with a failing finding")
    return "\n".join(lines)
