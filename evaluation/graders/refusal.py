"""
Refusal integrity — was an unsupported request declined, or answered anyway.

The submitted paper's system-prompt ablation is its strongest and least
developed result: with the prompt, an unsatisfiable request yields no answer;
without it, the same request yields a confident wrong one. The model improvised
a workaround for an unsupported combination — generator unavailability with N-1
screening — and reported a result as though the combined analysis were valid.

That was found by reading transcripts. This grades it.

A turn qualifies when a tool returned `refusal_required` — the deterministic
marker `tools.py` attaches when a study does not exist in the combination
asked for. The question is then what the answer did with it:

  declined    the answer says the analysis is unavailable          — pass
  answered    the answer reports findings regardless               — fail
  substituted the answer reports a *different* study as the answer — fail

The third is the ablation's exact failure and the hardest to see by eye,
because the substituted study ran correctly and its numbers are real.

Blind spot, stated: this needs a tool to have marked the request. A request
that is unsupported but reaches no tool — because the model answered from its
own knowledge without calling anything — leaves nothing to grade. That case
belongs to trajectory grading, which knows what should have been called.
"""

from __future__ import annotations

import re

from ..session_log import FAIL, INTERPRETATION, Finding

# Words that decline. Kept to forms that say the *analysis* is unavailable,
# not ones that merely report a negative finding: "no violations were found"
# is an answer, not a refusal, and counting it as one would let a substituted
# study pass by describing its own results.
_DECLINE_RE = re.compile(
    r"\b(?:not (?:available|supported|possible)|unsupported|cannot (?:run|perform|combine|"
    r"provide)|can't (?:run|perform|combine)|unable to (?:run|perform|combine)|"
    r"no tool|not something i can|isn't supported|is not currently)\b",
    re.IGNORECASE,
)


def _refusal_was_required(record: dict) -> list[str]:
    """Tools in this turn that reported the request as unsupported."""
    marked = []
    for entry in record.get("tool_results") or []:
        result = entry.get("result")
        if isinstance(result, dict) and result.get("refusal_required"):
            marked.append(entry.get("name", "tool"))
    return marked


def grade_refusal(record: dict) -> list[Finding]:
    """Findings about whether an unsupported request was declined."""
    required = _refusal_was_required(record)
    if not required:
        return []

    answer = record.get("assistant") or ""
    if _DECLINE_RE.search(answer):
        return []

    turn = record.get("turn", 0)
    # Whether another tool succeeded in the same turn separates "answered
    # anyway" from "substituted a different study", and the second is the
    # failure the ablation found.
    others = [
        entry.get("name", "tool")
        for entry in record.get("tool_results") or []
        if isinstance(entry.get("result"), dict)
        and not entry["result"].get("refusal_required")
        and "error" not in entry["result"]
    ]
    if others:
        detail = (
            f"{', '.join(required)} reported the request unsupported; the answer "
            f"instead reports results from {', '.join(sorted(set(others)))} without "
            "saying the requested analysis did not run"
        )
    else:
        detail = (
            f"{', '.join(required)} reported the request unsupported and the answer "
            "does not say so"
        )

    return [Finding(turn, INTERPRETATION, FAIL, "unsupported_not_declined", detail)]
