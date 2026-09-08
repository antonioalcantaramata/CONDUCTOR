"""
reflection.py — showing the agent what the checks found before it answers.

Two instruments run over a draft answer, and both are deliberately observers:
they report and never intervene. This module is the optional intervention built
on top of them, kept separate so they stay clean — with reflection off, nothing
here runs and the measurements are exactly what they were.

  `provenance.py`        does every figure come from the evidence, and does it
                         belong to the element the sentence is about
  `characterisation.py`  is the figure described the way its own record
                         describes it — the deliverable-sounding sentence over
                         an infeasible movement
  `completeness.py`      does the answer contain the kind of thing the question
                         asked for — the percentage that went missing between
                         draft and reply

They are one setting rather than two. An operator choosing how much the agent
second-guesses itself is making a single decision; which checks implement it is
ours to make, and a matrix of independent switches would turn a preference into
a configuration exercise. The record keeps them separable so the evaluation can
still attribute a finding to the check that made it.

The three modes, and why there are three:

  OFF     The baseline. Provenance still grades the turn offline; the agent
          never sees it. This is the configuration every earlier result was
          produced under, so it has to remain reachable unchanged.

  WARN    Provenance runs on the draft and the findings are recorded, but the
          agent is not shown them and the draft is returned as the answer.
          This measures what a reflection loop *would* have fired on, at zero
          behavioural cost — how often it would trigger, and on what, without
          changing a single answer.

  INFORM  The findings are handed back to the agent as evidence, and the agent
          decides: keep the answer, revise it, or call another tool. It is not
          told to correct anything.

The distinction between INFORM and a correction loop is the whole design. A
directive ("this figure is ungrounded, fix it") has one cheap form of
compliance — delete the number — and an operator tool that answers without
figures is worse than one that occasionally misattributes them. The findings
are therefore phrased as observations, with the nearest source value included
so the agent can tell a bad transcription from an invented quantity.

What we expect to have to distinguish afterwards, and why both answers are
kept in the record:

  revised   the figure changed or went away
  defended  the figure stayed and an explanation appeared around it
  ignored   the answer came back unchanged

`defended` is the outcome that would make informing worse than not informing:
the same wrong figure, now with a rationalisation attached. Nothing here can
prevent that — it can only make it visible, which is why the draft, the
findings and the final answer are all logged.

Only `ungrounded` and `misattributed` findings are surfaced. Imprecise ones are
transcription slips at warning severity; showing them invites fiddling with
figures that were essentially right.

Nor are `derived` ones. In a live run the check flagged a ratio *and* the
division that produced it, so an answer that showed its working scored worse
than one that did not — the cheapest way to satisfy the check became hiding
the arithmetic, which is the opposite of what any of this is for. A figure
whose operands and operation are both on the page is checkable by the operator
reading it. The offline grader still counts these, since the model did compute
them; reflection simply does not ask about them.
"""

from __future__ import annotations

from typing import NamedTuple

from .characterisation import Conflict
from .characterisation import check_answer as check_characterisation
from .completeness import Gap, format_gaps
from .completeness import check_answer as check_completeness
from .provenance import (
    MISATTRIBUTED,
    UNGROUNDED,
    ClaimVerdict,
    ProvenanceReport,
    check_answer,
    collect_sources,
    is_plausible_nearest,
    subject_names,
)

# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

OFF = "off"
WARN = "warn"
INFORM = "inform"

MODES = (OFF, WARN, INFORM)

# Surfaced to the agent. Imprecise claims are deliberately excluded: right
# figure, wrong precision is a transcription slip, and asking the model to
# revisit it trades a real risk of over-correction for nothing.
_SURFACED = (UNGROUNDED, MISATTRIBUTED)


def normalise_mode(value: str | None) -> str:
    """A mode string, falling back to OFF for anything unrecognised.

    Reflection is the non-default path, so an unreadable setting resolves to
    the behaviour that changes nothing.
    """
    if value is None:
        return OFF
    candidate = str(value).strip().lower()
    return candidate if candidate in MODES else OFF


class Reflection(NamedTuple):
    """What the check found on a draft, and what was done about it.

    Recorded on the turn whether or not the agent was shown anything, so a
    WARN run and an INFORM run produce comparable records.
    """

    mode: str
    draft: str
    findings: tuple[ClaimVerdict, ...]
    report: ProvenanceReport
    shown: bool          # were the findings actually put to the agent
    # Kept in its own field rather than merged into `findings`: the two checks
    # answer different questions, and an evaluation that cannot say which one
    # fired cannot say which one is worth its cost.
    conflicts: tuple[Conflict, ...] = ()
    # Kept separate again, and for a sharper reason than the others: a gap is
    # the only finding here that is routinely *correct*, so an evaluation that
    # merged it into the rest could not tell a withdrawn answer from a
    # properly declined one.
    gaps: tuple[Gap, ...] = ()
    revised: bool = False  # did a second answer come back
    final: str | None = None

    @property
    def triggered(self) -> bool:
        """Did either check find anything worth surfacing."""
        return bool(self.findings) or bool(self.conflicts) or bool(self.gaps)

    @property
    def changed(self) -> bool:
        """Did the answer actually change after the agent saw the findings."""
        return self.revised and (self.final or "").strip() != (self.draft or "").strip()

    def as_record(self) -> dict:
        """The turn-log shape. Both answers are kept deliberately.

        Without the draft alongside the final answer there is no way to tell
        revision from rationalisation after the fact, and that difference is
        the reason the mode exists.
        """
        return {
            "mode": self.mode,
            "triggered": self.triggered,
            "shown": self.shown,
            "revised": self.revised,
            "changed": self.changed,
            "n_findings": len(self.findings),
            "findings": [
                {
                    "status": v.status,
                    "value": v.claim.value,
                    "unit": v.claim.unit,
                    "context": v.claim.context,
                    "subject": v.subject,
                    "belongs_to": v.belongs_to,
                    "nearest": list(v.nearest) if v.nearest else None,
                }
                for v in self.findings
            ],
            "n_conflicts": len(self.conflicts),
            "conflicts": [
                {
                    "value": c.value,
                    "unit": c.unit,
                    "context": c.context,
                    "path": c.path,
                    "field": c.field,
                    "detail": c.detail,
                }
                for c in self.conflicts
            ],
            "n_gaps": len(self.gaps),
            "gaps": [{"kind": g.kind, "describe": g.describe} for g in self.gaps],
            "provenance_rate": round(self.report.rate, 4),
            "derived_rate": round(self.report.derived_rate, 4),
            "draft": self.draft,
            "final": self.final,
        }


# ---------------------------------------------------------------------------
# Checking a draft
# ---------------------------------------------------------------------------


def check_draft(draft: str, record: dict) -> tuple[tuple[ClaimVerdict, ...], ProvenanceReport]:
    """Grade a draft answer against the evidence the turn has produced so far.

    `record` carries the same fields the session log holds — `tool_results`,
    `tool_calls`, `grid`, `user` — so the source set is assembled exactly as
    the offline grader assembles it. Same matcher, same sources, same verdicts;
    the only difference is that this runs before the answer is delivered.
    """
    report = check_answer(
        draft or "",
        collect_sources(record),
        subjects=subject_names(record),
    )
    findings = tuple(v for v in report.verdicts if v.status in _SURFACED)
    return findings, report


def check_all(draft: str, record: dict) -> tuple[tuple[ClaimVerdict, ...],
                                                 ProvenanceReport,
                                                 tuple[Conflict, ...],
                                                 tuple[Gap, ...]]:
    """Both checks over one draft.

    Separate from `check_draft` so the provenance instrument keeps a callable
    of its own: the evaluation compares the two checks against each other, and
    that is easier when neither can only be run through the other.
    """
    findings, report = check_draft(draft, record)
    conflicts = tuple(check_characterisation(draft or "", record))
    gaps = tuple(check_completeness(record.get("user") or "", draft or ""))
    return findings, report, conflicts, gaps


# ---------------------------------------------------------------------------
# Wording
#
# The register matters more than anything else in this module. Each finding is
# stated as an observation with its nearest source attached, and the closing
# line hands the decision back rather than asking for a correction.
# ---------------------------------------------------------------------------


def _describe(verdict: ClaimVerdict) -> str:
    shown = f"{verdict.claim.value:g}" + (f" {verdict.claim.unit}" if verdict.claim.unit else "")

    if verdict.status == MISATTRIBUTED:
        subject = verdict.subject or "the element under discussion"
        owner = verdict.belongs_to or "a different element"
        return (
            f'- {shown} — written about {subject}, but this value comes from '
            f'{verdict.source}, which belongs to {owner}.'
        )

    # Only when it could plausibly be what was meant. The grader keeps every
    # nearest value as diagnostic context; putting a distant one to the agent
    # turns it into a suggestion, and a wrong suggestion is worse than none.
    near = ""
    if is_plausible_nearest(verdict.nearest, verdict.claim.value):
        near = f" The closest value in the evidence is {verdict.nearest[0]:g} at {verdict.nearest[1]}."
    return (
        f'- {shown} — in "{verdict.claim.context}". This figure does not appear '
        f'in any tool result, call argument, or anything the user provided '
        f'this turn.{near}'
    )


def format_findings(findings: tuple[ClaimVerdict, ...]) -> str:
    """The message the agent is shown. Evidence, not instruction.

    Deliberately does not say "fix this". The agent is better placed than the
    check to know whether a figure is wrong, whether it came from a tool it
    has not called yet, or whether the check simply cannot see its source —
    and the last of those is a real case, since a grounding check has no view
    of anything outside this turn.
    """
    lines = [
        "An automatic grounding check ran over your draft answer and could not "
        "trace the following figures to this turn's evidence:",
        "",
    ]
    lines.extend(_describe(v) for v in findings)
    lines.extend([
        "",
        "This check is imperfect and does not know your reasoning. Consider whether "
        "each figure is one you can point to in the evidence. You may keep the answer "
        "as it stands, correct it, or call a tool to obtain the value — whichever is "
        "actually right. If you keep a figure, say where it came from.",
    ])
    return "\n".join(lines)


def format_conflicts(conflicts: tuple[Conflict, ...]) -> str:
    """The characterisation message. Same register, sharper subject.

    Where a grounding finding says "I cannot see where this came from", this
    one says "the record you took it from says otherwise" — a claim the check
    can actually stand behind, since the contradicting field is quoted. The
    closing line still hands the decision back, because the check reads words
    and can misread a sentence that is in fact correct.
    """
    lines = [
        "An automatic consistency check compared your draft against the fields "
        "beside each figure in the tool output, and found these disagreements:",
        "",
    ]
    for conflict in conflicts:
        shown = f"{conflict.value:g}" + (f" {conflict.unit}" if conflict.unit else "")
        lines.append(
            f'- {shown} — in "{conflict.context}". Here {conflict.detail} '
            f"(`{conflict.path}`, `{conflict.field}`)."
        )
    lines.extend([
        "",
        "A figure can be correct and still be described wrongly. Consider whether "
        "each sentence above says what its own record says. You may keep the answer, "
        "restate it, or use a different figure the record supports.",
    ])
    return "\n".join(lines)


def format_message(findings: tuple[ClaimVerdict, ...],
                   conflicts: tuple[Conflict, ...],
                   gaps: tuple[Gap, ...] = ()) -> str:
    """Everything the agent is shown this turn, as one message.

    One message rather than two so the agent sees the whole picture before it
    decides. Sections appear only when they have content.
    """
    parts = []
    if findings:
        parts.append(format_findings(findings))
    if conflicts:
        parts.append(format_conflicts(conflicts))
    if gaps:
        parts.append(format_gaps(gaps))
    return "\n\n".join(parts)
