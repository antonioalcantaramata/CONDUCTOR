"""
characterisation.py — is the figure described as the payload describes it?

The blind spot the other checks leave. Provenance answers two questions about
a number in the answer: does it come from the evidence, and does it belong to
the element the sentence is about. Both can pass while the sentence is still
wrong, because neither reads what the payload says *about* the figure.

The live case, from the additions log. The engine had resolved a movement and
marked it undeliverable:

    relief_mw:          -6.2415
    relief_mw_feasible: false

and the answer said:

    "Reduce 05 ÅKI Sgen by 6.24 MW (or absorb 1.06 MVAr) if acting alone."

6.24 was in the payload and correctly attributed to its own driver, so
provenance had nothing to say. It was caught by the *sign* — the source was
negative, the claim positive — and had the model written the minus, nothing
would have flagged it. That was luck, and this module is the part that does
not depend on the luck repeating.

Method
------
Every numeric field in a tool result is read together with the qualifier fields
beside it: booleans in the same object that say whether the quantity is
deliverable, whether the element is violating, whether the study converged.
When a figure in the answer traces to a path whose qualifier is unfavourable,
and the surrounding prose asserts the favourable reading, the two disagree and
that is reported.

The comparison is between the payload and the *claim the prose makes*, not
between the payload and a keyword. "Cannot be reduced by 6.24 MW" and "reduce
by 6.24 MW" quote the same figure from the same unfavourable record; only the
second is wrong. Negation is therefore read before a mismatch is raised.

Scope, stated plainly
---------------------
  Catches   a deliverable-sounding sentence over an infeasible movement; a
            secure-sounding sentence over a violating record; a converged
            claim over a failed study.

  Misses    a wrong characterisation with no qualifier field behind it. If the
            payload does not record whether something is feasible, nothing
            here can tell. This narrows the gap the additions log describes; it
            does not close it, and the paper should say so.

Like `provenance.py` this is an instrument first: it reports, and the caller
decides whether the agent ever sees it.
"""

from __future__ import annotations

import re
from typing import Any, NamedTuple

# ---------------------------------------------------------------------------
# Qualifiers
#
# A qualifier is a boolean sitting beside a number that says how the number may
# be spoken about. They are named rather than inferred: a heuristic over every
# boolean in a payload would read `controllable` — a property of a unit, not of
# a movement — as a licence to talk about relief figures.
# ---------------------------------------------------------------------------


class Qualifier(NamedTuple):
    """A boolean field, and what an answer may claim when it is false."""

    field: str
    # Words that assert the favourable reading. Present in the sentence with an
    # unfavourable qualifier, they conflict.
    asserts: tuple[str, ...]
    # How to say what went wrong.
    describe: str


_QUALIFIERS = (
    Qualifier(
        "relief_mw_feasible",
        ("reduce", "increase", "curtail", "raise", "lower", "adjust", "move",
         "dispatch", "absorb", "inject", "set"),
        "the payload marks this movement as one the source cannot deliver",
    ),
    Qualifier(
        "relief_mvar_feasible",
        ("reduce", "increase", "curtail", "raise", "lower", "adjust", "move",
         "dispatch", "absorb", "inject", "set"),
        "the payload marks this movement as one the source cannot deliver",
    ),
    Qualifier(
        "feasible",
        ("feasible", "achievable", "deliverable", "possible", "can be", "able to"),
        "the payload marks this as infeasible",
    ),
    Qualifier(
        # A load's movement is arithmetically possible and operationally
        # unavailable. Observed live: every driver of an undervoltage was a
        # load carrying a feasible-looking 12.4 MW relief, which read back as
        # a recommendation says "shed 12.4 MW of customer demand".
        "actionable",
        ("reduce", "increase", "curtail", "raise", "lower", "adjust", "shed",
         "dispatch", "redispatch", "recommend", "should", "action"),
        "the payload marks this driver as not actionable — it explains the "
        "regime rather than offering a lever",
    ),
    Qualifier(
        "converged",
        ("secure", "within limits", "no violation", "solved", "converged"),
        "the payload records that the solve did not converge",
    ),
    Qualifier(
        "secure",
        ("secure", "within limits", "no violation", "compliant"),
        "the payload records this operating point as insecure",
    ),
)

# Fields whose *true* value is the unfavourable one — a violated element is
# violated when the flag is set, which is the reverse of the others.
_INVERTED = (
    Qualifier(
        "violated",
        ("secure", "within limits", "no violation", "compliant", "acceptable"),
        "the payload records this element as violating its limit",
    ),
)

_QUALIFIER_FIELDS = {q.field for q in _QUALIFIERS} | {q.field for q in _INVERTED}

# Fields a qualifier does NOT govern, even when they sit in the same object.
#
# A driver record carries the movement being judged (`relief_mw`,
# `would_require_mw`) alongside descriptive quantities that are not actions at
# all: the sensitivity `d_per_mw` (p.u. per MW), and the unit's present output
# `current_p_mw`. `actionable: false` says nothing about those.
#
# Observed live, and the reason this exists: a sentence reading "violating the
# lower limit by 0.001 p.u." matched `d_per_mw = -0.000509` — a sensitivity in
# p.u./MW that rounds into the same band at three decimals — and the panel
# reported a voltage margin as an undeliverable dispatch action. Twenty times,
# once per driver field in the payload.
_NOT_A_MOVEMENT = (
    "d_per_",          # sensitivities: p.u. per MW / per MVAr
    "current_",        # the source's present output, not a proposed change
    "max_deliverable", # a capability statement, already the safe figure
    "_index", "index",
)

# Negation immediately before an assertion flips it. Kept deliberately short:
# these are the forms the model actually writes, and a longer list starts
# matching across clause boundaries.
_NEGATION_RE = re.compile(
    r"\b(?:cannot|can't|could not|couldn't|not|no|never|unable|insufficient|"
    r"exceeds|beyond|short of|fails? to|without)\b",
    re.IGNORECASE,
)

# How far back from an assertion a negation still governs it.
_NEGATION_WINDOW = 60

# Which qualifier to report when several fire on one claim. Lower is more
# specific. `actionable` outranks a feasibility flag because "nobody can
# command this" explains the problem where "the source cannot deliver it"
# only restates it — for a load, both are true and only one is the reason.
_SPECIFICITY = {
    "actionable": 10,
    "violated": 20,
    "secure": 20,
    "converged": 20,
    "relief_mw_feasible": 30,
    "relief_mvar_feasible": 30,
    "feasible": 40,
}


class Conflict(NamedTuple):
    """A figure spoken about in a way its own record contradicts."""

    value: float
    unit: str | None
    context: str
    path: str
    field: str
    detail: str

    def render(self) -> str:
        shown = f"{self.value:g}" + (f" {self.unit}" if self.unit else "")
        return f'{shown} — "{self.context}" — {self.detail} ({self.path}.{self.field})'


# ---------------------------------------------------------------------------
# Reading qualifiers out of a payload
# ---------------------------------------------------------------------------


def _qualified_numbers(node: Any, path: str, out: dict[float, list[tuple[str, Qualifier]]]) -> None:
    """Numbers whose own object carries an unfavourable qualifier.

    Only the object a number sits in is consulted, never an ancestor. A
    violation record's `violated: true` says nothing about a threshold quoted
    from the enclosing settings block, and walking upward to find a qualifier
    is how a check starts flagging correct sentences.
    """
    if isinstance(node, dict):
        unfavourable: list[Qualifier] = []
        for qualifier in _QUALIFIERS:
            if node.get(qualifier.field) is False:
                unfavourable.append(qualifier)
        for qualifier in _INVERTED:
            if node.get(qualifier.field) is True:
                unfavourable.append(qualifier)

        for key, value in node.items():
            child = f"{path}.{key}"
            if isinstance(value, bool):
                continue
            if (isinstance(value, (int, float)) and unfavourable
                    and key not in _QUALIFIER_FIELDS
                    and not key.startswith(_NOT_A_MOVEMENT)
                    and not key.endswith(_NOT_A_MOVEMENT)):
                for qualifier in unfavourable:
                    out.setdefault(float(value), []).append((child, qualifier))
            else:
                _qualified_numbers(value, child, out)
        return

    if isinstance(node, list):
        for index, item in enumerate(node):
            _qualified_numbers(item, f"{path}[{index}]", out)


def qualified_numbers(record: dict) -> dict[float, list[tuple[str, Qualifier]]]:
    """Every number in the turn's tool results that carries an unfavourable flag."""
    out: dict[float, list[tuple[str, Qualifier]]] = {}
    for entry in record.get("tool_results") or []:
        # A malformed entry costs its own qualifiers, not the whole check. This
        # also runs live behind the operator-facing audit panel, where a raise
        # is far more costly than a missing finding.
        if not isinstance(entry, dict):
            continue
        _qualified_numbers(entry.get("result"), entry.get("name", "tool"), out)
    return out


# ---------------------------------------------------------------------------
# Reading the claim the prose makes
# ---------------------------------------------------------------------------


def _asserts_favourable(context: str, qualifier: Qualifier) -> bool:
    """Does this sentence assert the reading the qualifier denies.

    A negated assertion is not an assertion: "cannot be reduced by 6.24 MW"
    quotes the same infeasible figure as "reduce by 6.24 MW" and is correct.
    Without this the check would flag precisely the answers that got it right,
    which is the failure mode that matters most in a guard an agent reads.
    """
    lowered = context.lower()
    for word in qualifier.asserts:
        # Whole words only. Without the boundary, "move" matched inside
        # "movement" and the engine's own correct sentence — "the smallest
        # sufficient movement (…) exceeds its available 3.54 MW" — was read as
        # recommending that movement.
        pattern = re.escape(word) if " " in word else rf"\b{re.escape(word)}\b"
        for match in re.finditer(pattern, lowered):
            window = lowered[max(0, match.start() - _NEGATION_WINDOW):match.start()]
            if not _NEGATION_RE.search(window):
                return True
    return False


# Sentence boundaries. A newline ends one too: the model writes its
# recommendations as bullet lists, where nothing carries a full stop.
_BOUNDARY_RE = re.compile(r"[.!?;\n]")


def _sentence_around(text: str, position: int) -> str:
    """The sentence containing a claim, as the unit a negation governs.

    A decimal point is not a boundary, so a figure is never split from the
    clause that qualifies it.
    """
    start = 0
    for match in _BOUNDARY_RE.finditer(text, 0, position):
        # A full stop between two digits is a decimal point.
        end = match.end()
        if (match.group() == "." and end < len(text)
                and text[end].isdigit() and match.start() > 0
                and text[match.start() - 1].isdigit()):
            continue
        start = end
    end_match = _BOUNDARY_RE.search(text, position)
    end = end_match.start() if end_match else len(text)
    return text[start:max(end, position)].strip()


# Top-level verdicts an engine states about the whole study, and the words an
# answer uses to claim the favourable reading. Separate from `_QUALIFIERS`
# above, which qualify a *number*: these qualify the study itself, and an
# answer can contradict them without quoting any figure at all.
_STUDY_VERDICTS = (
    Qualifier(
        "secure",
        ("secure", "within limits", "within its limits", "no violation",
         "no violations", "compliant", "looks fine", "looks good", "all good",
         "all clear", "nothing to report", "operating normally"),
        "the assessment recorded this operating point as insecure",
    ),
    Qualifier(
        "converged",
        ("secure", "within limits", "no violation", "solved", "converged",
         "successful", "succeeded"),
        "the study did not converge, so its result is not usable",
    ),
    Qualifier(
        "usable",
        ("secure", "within limits", "no violation", "solved", "converged",
         "successful", "succeeded", "result", "answer"),
        "the study ran but did not produce a usable result",
    ),
    Qualifier(
        "feasible",
        ("feasible", "achievable", "deliverable", "successful", "succeeded",
         "solved", "optimal"),
        "the optimisation recorded this as infeasible",
    ),
)


def _study_verdicts(record: dict) -> list[tuple[str, Qualifier]]:
    """Unfavourable verdicts each tool result states about itself."""
    out: list[tuple[str, Qualifier]] = []
    for entry in record.get("tool_results") or []:
        if not isinstance(entry, dict):
            continue
        result = entry.get("result")
        if not isinstance(result, dict):
            continue
        name = entry.get("name", "tool")
        for verdict in _STUDY_VERDICTS:
            if result.get(verdict.field) is False:
                out.append((name, verdict))
        # A violation count above zero is the same statement as `secure: False`
        # and some endpoints carry only one of the two.
        total = result.get("total_violations")
        if isinstance(total, int) and total > 0 and not any(
            v.field == "secure" for _, v in out
        ):
            out.append((name, _STUDY_VERDICTS[0]))
    return out


def check_study_claims(answer: str, record: dict) -> list[Conflict]:
    """Sentences that call a study fine when the study says otherwise.

    The gap the figure-anchored check above cannot reach. Asked whether the
    system was secure, an answer of "everything looks fine" over a result
    carrying `secure: False` quotes no number, so nothing in the layer had
    anything to attach a finding to — the most obviously wrong answer a reader
    could imagine was the one it could not see.

    Reported with value 0 and no unit: the subject is the study, not a figure.
    """
    text = answer or ""
    if not text.strip():
        return []

    conflicts: list[Conflict] = []
    seen: set[str] = set()
    for name, verdict in _study_verdicts(record):
        if verdict.field in seen:
            continue
        # Judged over the whole answer rather than a sentence: the claim may be
        # a standalone "All clear." with the detail in a neighbouring sentence.
        if _asserts_favourable(text, verdict):
            seen.add(verdict.field)
            conflicts.append(Conflict(
                value=0.0, unit=None,
                context=" ".join(text.split())[:120],
                path=name, field=verdict.field, detail=verdict.describe,
            ))
    return conflicts


def check_answer(answer: str, record: dict) -> list[Conflict]:
    """Figures the answer characterises against their own records."""
    from .provenance import extract_claims  # local: avoids an import cycle

    qualified = qualified_numbers(record)
    if not qualified:
        return []

    text = answer or ""
    conflicts: list[Conflict] = []
    for claim in extract_claims(text):
        # The claim's own 55-char window is too narrow to judge by: the
        # negation that makes a sentence correct routinely sits before it —
        # "No single source can clear it: the smallest sufficient movement
        # (…, 6.24 MW)" — and reading only the window flags the one answer
        # that got it right. The sentence containing the figure is the unit
        # of judgement.
        sentence = _sentence_around(text, claim.position)

        # Every payload path whose value matches this claim and whose qualifier
        # the sentence contradicts. Collected across all matching values, not
        # one conflict per value: a claim of "0.001 p.u." matched twenty
        # sensitivity fields in one attribution payload, and the panel reported
        # the same sentence twenty times. Twenty identical warnings for one
        # sentence is how an operator learns to stop reading them.
        matched: list[tuple[str, Qualifier]] = []
        for value, entries in qualified.items():
            # Magnitude only. The 6.24 case was written positive against a
            # negative source, and a check that required the sign to match
            # would have missed exactly the sentence it exists for.
            if abs(abs(value) - abs(claim.value)) > 10.0 ** -claim.decimals:
                continue
            matched.extend(
                (path, q) for path, q in entries if _asserts_favourable(sentence, q)
            )

        if matched:
            # One conflict per claim, and where several qualifiers fire the
            # most specific one is reported. A load's movement trips both
            # `relief_mw_feasible` and `actionable`; the first says the source
            # cannot deliver it, the second says nobody can command it, and
            # only the second tells the reader why.
            path, qualifier = min(
                matched, key=lambda pair: _SPECIFICITY.get(pair[1].field, 50)
            )
            conflicts.append(Conflict(
                value=claim.value,
                unit=claim.unit,
                context=claim.context,
                path=path if len(matched) == 1 else f"{path} (+{len(matched) - 1} more)",
                field=qualifier.field,
                detail=qualifier.describe,
            ))
    return conflicts


def check_turn(record: dict) -> list[Conflict]:
    """Grade one session-log turn record."""
    return check_answer(record.get("assistant") or "", record)
