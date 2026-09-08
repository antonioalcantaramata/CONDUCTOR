"""
completeness.py — did the answer contain the kind of thing that was asked for?

The failure none of the other checks can see, because every one of them passed
while it happened. Asked what percentage of load the external grid covers, the
agent drafted "approximately 80.3%", was shown a grounding finding, and
delivered this instead:

    External grid import: 176.7793 MW
    Total system load:    220.1206 MW

Two true, grounded, correctly attributed figures, and no percentage. The
operator asked a question, the system had the answer, and the answer was not
in the reply. It took another turn — "compute the percentage" — to get it back.

That matters more here than it would elsewhere, because we built the thing that
caused it. Reflection can make an agent withdraw a figure rather than defend
it, and shipping a mechanism that removes answers without a way to see it
happening is not a defensible position.

Severity, and why this one is different
---------------------------------------
Every finding here is a **warning**, never a failure, and the wording put to
the agent never asserts the answer is wrong.

A missing quantity is often correct. The tool may have errored; the study may
not support the question; the honest reply to "what is the hosting capacity"
may be that the bisection did not converge. An answer that declines for a good
reason is a *better* answer than one that invents a number, and this module
must never push against that — it is the same trap the first grounding loop
fell into, one level up.

So the check reports an absence and says explicitly that the absence may be
right. What it is for is the case where the quantity was available and simply
went missing.

Scope
-----
  Catches   a question naming a kind of quantity, answered without one.

  Misses    unusual phrasings; a question whose answer is a quantity of a kind
            nobody named; and any case where the right answer genuinely has no
            number, which it cannot distinguish from a withdrawal. That last
            one is why this is a warning.
"""

from __future__ import annotations

import re
from typing import NamedTuple

# ---------------------------------------------------------------------------
# What a question asks for
#
# Each kind pairs a way of asking with a way of answering. Both sides are
# deliberately narrow: a false "you did not answer" on a reply that did is
# worse than silence, because it invites the agent to pad an answer that was
# already complete.
# ---------------------------------------------------------------------------


class Kind(NamedTuple):
    name: str
    asks: re.Pattern
    answers: re.Pattern
    describe: str


_KINDS = (
    Kind(
        "percentage",
        re.compile(r"\b(?:what|which|how much)\b[^?]{0,80}?\b(?:percent(?:age)?|share|fraction|proportion)\b"
                   r"|\bpercent(?:age)?\s+(?:of|is|does|covered)\b", re.IGNORECASE),
        re.compile(r"\d+(?:\.\d+)?\s*%|\bpercent\b|\d+(?:\.\d+)?\s*(?:per ?cent)\b", re.IGNORECASE),
        "a percentage",
    ),
    Kind(
        "power",
        re.compile(r"\bhow (?:much|many)\b[^?]{0,60}?\b(?:mw|mvar|mva|power|capacity|headroom|import|export)\b"
                   r"|\b(?:exact|how many)\s+mw\b", re.IGNORECASE),
        re.compile(r"\d+(?:\.\d+)?\s*(?:MW|MVAr|MVA|kW)\b", re.IGNORECASE),
        "a power figure",
    ),
    Kind(
        "voltage",
        re.compile(r"\bwhat\b[^?]{0,60}?\b(?:voltage|p\.?u\.?)\b", re.IGNORECASE),
        re.compile(r"\d+(?:\.\d+)?\s*(?:p\.?u\.?|kV)\b", re.IGNORECASE),
        "a voltage",
    ),
    Kind(
        "timestamp",
        re.compile(r"\b(?:when|what time|which (?:timestamp|hour|moment)|worst timestamp)\b", re.IGNORECASE),
        re.compile(r"\d{4}-\d{2}-\d{2}|\d{1,2}:\d{2}", re.IGNORECASE),
        "a timestamp",
    ),
    Kind(
        "count",
        re.compile(r"\bhow many\b[^?]{0,60}?\b(?:violation|bus|buses|line|lines|transformer|"
                   r"generator|element|contingenc)\w*", re.IGNORECASE),
        re.compile(r"\b\d+\b|\bno\b|\bnone\b|\bzero\b", re.IGNORECASE),
        "a count",
    ),
)

# A reply that declines, and says why. Not an incomplete answer — the most
# useful thing the system can say when the study cannot produce the figure,
# and flagging it would push the agent back toward inventing one.
_DECLINED_RE = re.compile(
    r"\b(?:not (?:available|supported|possible|converge\w*)"
    r"|cannot (?:be )?(?:determin|comput|provid|calculat|report)\w*"
    r"|can(?:not| ?'?t) be (?:determin|comput|provid|calculat|report)\w*"
    r"|unable to|infeasible|did not converge|unsupported|insufficient data"
    # "no dispatch figure can be provided", "no capacity is available" — the
    # noun between "no" and the verb varies too much to enumerate, so the
    # bound is a short window rather than a word list.
    r"|no\b[^.\n]{0,40}?\b(?:can be|is|are|was|were)\b[^.\n]{0,20}?"
    r"(?:provid|determin|comput|calculat|availab|report)\w*"
    r"|no (?:tool|data|measurement|result|value|figure)s?\b)",
    re.IGNORECASE,
)


class Gap(NamedTuple):
    """A kind of quantity the question named and the answer did not contain."""

    kind: str
    describe: str

    def render(self) -> str:
        return f"the question asked for {self.describe}; the answer contains none"


def asked_for(question: str) -> list[Kind]:
    """Kinds of quantity the question named."""
    return [k for k in _KINDS if k.asks.search(question or "")]


def check_answer(question: str, answer: str) -> list[Gap]:
    """Kinds the question asked for that the answer does not contain.

    An answer that explicitly declines is complete: saying the figure cannot be
    produced, and why, is a real answer to the question.
    """
    text = answer or ""
    if _DECLINED_RE.search(text):
        return []
    return [
        Gap(k.name, k.describe)
        for k in asked_for(question)
        if not k.answers.search(text)
    ]


def check_turn(record: dict) -> list[Gap]:
    """Grade one session-log turn record."""
    return check_answer(record.get("user") or "", record.get("assistant") or "")


def format_gaps(gaps: tuple[Gap, ...]) -> str:
    """The message the agent is shown.

    Says the absence may be correct, and says it first. The agent knows things
    this check does not — whether a tool failed, whether the study supports the
    question at all — and an answer that declines for a good reason must not be
    argued out of declining.
    """
    lines = [
        "A completeness check compared your draft against the question. It could "
        "not find the following in your answer:",
        "",
    ]
    lines.extend(f"- {g.describe}" for g in gaps)
    lines.extend([
        "",
        "This may well be correct — if the figure is unavailable, the study does "
        "not support it, or a tool failed, then saying so plainly is the right "
        "answer and you should keep it. Say why it is unavailable rather than "
        "leaving it out silently. If the figure is available and was simply "
        "omitted, include it.",
    ])
    return "\n".join(lines)
