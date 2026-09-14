"""
trace.py — what an operator needs to audit one answer.

The guards in `validators.py`, `provenance.py`, `characterisation.py` and
`completeness.py` all run on every turn, and until now every one of them
reported only to a log file. The operator reading the answer — the person who
might act on a dispatch figure — saw prose and charts and nothing else.

This assembles, from the turn record the loop already writes, the evidence that
lets that person check the answer themselves: which tools ran and with what
arguments, where each figure in the prose came from, and what the system did
*not* verify.

Three rules shape it, and all three exist to stop the panel becoming a
trust badge it has not earned:

  1. **Describe, never certify.** The summary says "12 figures traced to tool
     output", not "verified". A traced figure can still be wrong — provenance
     is blind to a correct number under a wrong label — and a badge that
     implies otherwise is worse than no badge.

  2. **State the limits in the same breath as the counts.** Every check has a
     blind spot recorded in its own docstring. Those belong next to the number
     they qualify, not in documentation the operator will never read.

  3. **Never hide a finding.** A turn with warnings must say so in its
     *collapsed* label, because a panel nobody opens cannot warn anybody.

Pure data in, pure data out: no Streamlit here, so the assembly is testable
without a browser and could feed a report, an API response or a log line.
"""

from __future__ import annotations

from typing import Any, NamedTuple

from .characterisation import check_answer as _check_characterisation
from .characterisation import check_study_claims as _check_study_claims
from .completeness import check_answer as _check_completeness
from .provenance import (
    DERIVED,
    GROUNDED,
    IMPRECISE,
    MISATTRIBUTED,
    UNGROUNDED,
    check_turn as _check_provenance,
)

# What each check cannot see. Quoted from the modules' own stated scope, and
# shown to the operator rather than left in the source, because a count without
# its blind spot reads as a guarantee.
BLIND_SPOTS = {
    "provenance": (
        "A figure can be traced and still be wrong: this checks where a number "
        "came from, not whether the sentence around it is true."
    ),
    "characterisation": (
        "Only claims with a feasibility or status field behind them are "
        "checked. Where the payload records no verdict, nothing here can tell."
    ),
    "completeness": (
        "Cannot tell an answer that omitted something from one that correctly "
        "declined to answer."
    ),
}


class ToolStep(NamedTuple):
    """One tool call, as it actually ran."""

    order: int
    name: str
    args: dict
    # Arguments the operator did not supply and the model did not repeat, which
    # the turn carried forward. Surfacing these is the only place the repair
    # guard becomes visible to a human.
    inherited: dict
    ok: bool
    error: str | None
    # Integrity findings the backend attached to this result.
    warnings: list[str]

    @property
    def inherited_summary(self) -> str:
        """Human-readable inheritance, tolerant of a malformed record.

        A turn record can be truncated by a crash or hand-edited, and this
        string is built during rendering — where an exception costs the
        operator the whole conversation view, not just this panel.
        """
        if not isinstance(self.inherited, dict):
            return ""
        parts = []
        for field, detail in sorted(self.inherited.items(), key=lambda kv: str(kv[0])):
            if isinstance(detail, dict):
                parts.append(f"{field}={detail.get('value')} (from {detail.get('from')})")
            else:
                parts.append(f"{field}={detail}")
        return ", ".join(parts)


class FigureTrace(NamedTuple):
    """One number in the answer, and where it came from."""

    value: float
    unit: str | None
    status: str
    source: str | None
    context: str
    detail: str

    @property
    def traced(self) -> bool:
        return self.status in (GROUNDED, IMPRECISE)


class Trace(NamedTuple):
    """Everything needed to audit one answer."""

    steps: tuple[ToolStep, ...]
    figures: tuple[FigureTrace, ...]
    concerns: tuple[str, ...]
    limits: tuple[str, ...]

    @property
    def n_traced(self) -> int:
        return sum(1 for f in self.figures if f.traced)

    @property
    def has_concerns(self) -> bool:
        return bool(self.concerns)

    def headline(self) -> str:
        """The collapsed label. Descriptive, and never hides a concern.

        A panel nobody opens cannot warn anybody, so anything the checks
        flagged is counted here — in the one line the operator always sees.
        """
        parts = []
        if self.steps:
            ok = sum(1 for s in self.steps if s.ok)
            total = len(self.steps)
            # "0 tools" after a failed call is indistinguishable from a turn
            # that ran nothing, which is the opposite reading. Show the
            # denominator whenever anything failed.
            parts.append(f"{ok} tool{'s' if ok != 1 else ''}" if ok == total
                         else f"{ok}/{total} tools succeeded")
        if self.figures:
            parts.append(f"{self.n_traced}/{len(self.figures)} figures traced")
        if self.concerns:
            n = len(self.concerns)
            parts.append(f"⚠ {n} to check")
        return " · ".join(parts) if parts else "no tools used"


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def _warnings_of(result: Any) -> list[str]:
    if not isinstance(result, dict):
        return []
    return [
        str(w.get("message", ""))
        for w in result.get("_integrity_warnings") or []
        if isinstance(w, dict)
    ]


def _steps(record: dict) -> tuple[ToolStep, ...]:
    """Tool calls paired with their results, in execution order.

    Paired by position rather than by name: a turn that calls the same tool
    twice at two operating points is exactly the case an operator most needs to
    see separated, and matching on name would merge them.
    """
    calls = record.get("tool_calls") or []
    results = record.get("tool_results") or []
    steps: list[ToolStep] = []

    for index, call in enumerate(calls):
        if not isinstance(call, dict):
            continue
        entry = results[index] if index < len(results) else None
        result = entry.get("result") if isinstance(entry, dict) else None
        failed = isinstance(result, dict) and "error" in result
        raw_args = call.get("args")
        raw_inherited = call.get("inherited")
        steps.append(ToolStep(
            order=index + 1,
            # Coerced here rather than trusted, because every one of these is
            # dereferenced during rendering: `name.replace(...)`, `args.items()`,
            # `inherited.items()`. `build()` is wrapped in a try/except; the
            # render is not, and a raise there takes the chat down with it.
            name=str(call.get("name") or "?"),
            args=({k: v for k, v in raw_args.items() if v not in (None, "")}
                  if isinstance(raw_args, dict) else {}),
            inherited=raw_inherited if isinstance(raw_inherited, dict) else {},
            ok=not failed,
            error=str(result.get("error")) if failed else None,
            warnings=_warnings_of(result),
        ))
    return tuple(steps)


_FIGURE_DETAIL = {
    GROUNDED: "matches the tool output",
    IMPRECISE: "matches, rounded differently in the text",
    MISATTRIBUTED: "real, but belongs to a different element",
    DERIVED: "computed in the answer from figures above",
    UNGROUNDED: "not found in this turn's tool output",
}


def _figures(record: dict) -> tuple[FigureTrace, ...]:
    report = _check_provenance(record)
    return tuple(
        FigureTrace(
            value=v.claim.value,
            unit=v.claim.unit,
            status=v.status,
            source=v.source,
            context=v.claim.context,
            detail=_FIGURE_DETAIL.get(v.status, v.status),
        )
        for v in report.verdicts
    )


def _concerns(record: dict, steps: tuple[ToolStep, ...],
              figures: tuple[FigureTrace, ...]) -> tuple[str, ...]:
    """Everything the operator should look at before acting.

    Deliberately one flat list rather than one section per check. The operator
    is asking "is there anything wrong with this answer", not "what did the
    characterisation module conclude".
    """
    out: list[str] = []
    answer = record.get("assistant") or ""

    for figure in figures:
        if figure.status == UNGROUNDED:
            shown = f"{figure.value:g}" + (f" {figure.unit}" if figure.unit else "")
            out.append(f"{shown} does not appear in any tool output from this turn.")
        elif figure.status == MISATTRIBUTED:
            shown = f"{figure.value:g}" + (f" {figure.unit}" if figure.unit else "")
            out.append(f"{shown} is real but belongs to a different element than the "
                       "sentence using it.")

    for conflict in _check_characterisation(answer, record):
        out.append(conflict.render())

    # A verdict contradiction needs no figure to anchor it — "everything looks
    # fine" over a result carrying `secure: False` is the most obviously wrong
    # answer possible, and nothing else in the layer can see it.
    for conflict in _check_study_claims(answer, record):
        out.append(f"The answer reads as reassuring, but {conflict.detail} "
                   f"({conflict.path}).")

    for gap in _check_completeness(record.get("user") or "", answer):
        out.append(gap.render() + ". This may be correct if the figure was unavailable.")

    reflection = record.get("reflection")
    if isinstance(reflection, dict) and reflection.get("changed"):
        out.append(
            "This answer was revised after an automatic check flagged the "
            "first draft. Both versions are in the session log."
        )

    for step in steps:
        for warning in step.warnings:
            out.append(f"{step.name}: {warning}")
        if step.error:
            out.append(f"{step.name} did not complete: {step.error}")

    # One entry per distinct problem. A figure repeated in the prose, or the
    # same tool called twice with the same warning, is still one thing to
    # check — and a count the operator can see is padded is a count they stop
    # believing. Order is preserved: the first mention is the one shown.
    return tuple(dict.fromkeys(out))


# Marks a concern that came from the completeness check, so its caveat can be
# shown beside it. Matching on the sentence the check itself appends is
# deliberate: it keeps the two definitions in one place.
_GAP_MARKER = "This may be correct if the figure was unavailable."


def _limits(figures: tuple[FigureTrace, ...], concerns: tuple[str, ...]) -> tuple[str, ...]:
    """What was not checked. Always non-empty when anything was checked at all.

    The completeness caveat is shown when a gap was actually raised, because it
    qualifies *that finding* — it tells the operator the reported omission may
    be a correct refusal. An earlier version had this condition inverted, so
    the caveat appeared on clean turns (qualifying nothing) and was missing
    from the turns it exists to soften.
    """
    out = []
    if figures:
        out.append(BLIND_SPOTS["provenance"])
    out.append(BLIND_SPOTS["characterisation"])
    if any(_GAP_MARKER in c for c in concerns):
        out.append(BLIND_SPOTS["completeness"])
    return tuple(out)


def build(record: dict) -> Trace:
    """Assemble the audit trace for one turn record.

    `record` is the session-log shape the loop writes: `user`, `assistant`,
    `tool_calls`, `tool_results`, `grid`. The same input the offline graders
    read, so the panel and the evaluation cannot disagree about a turn.
    """
    steps = _steps(record)
    figures = _figures(record)
    concerns = _concerns(record, steps, figures)
    return Trace(
        steps=steps,
        figures=figures,
        concerns=concerns,
        limits=_limits(figures, concerns),
    )
