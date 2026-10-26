"""
Evidence — is each claim worded within what backs it.

Thin wrapper over `llm_agent.agent.evidence`, which also runs live (the answer
review and the evidence card under every answer). The graded finding is the
overstated claim: "proved", "impossible", "the best possible" with no
certificate in the turn's results, or "secure" with no study behind it. It is
a `fail`, as a mischaracterised study is — the answer claims more certainty
than any result gives, which is the one way an answer becomes more dangerous
than no answer.

Two figures for the paper come with it: how the answer's claims spread over
the evidence levels (proved, power flow, optimised, estimated, input, …), and
what the reading check found when the turn has one — whether the conclusion
held under the readings the question left open.
"""

from __future__ import annotations

from llm_agent.agent.evidence import Ledger, build

from ..session_log import FAIL, INTERPRETATION, WARN, Finding


def grade_evidence(record: dict) -> tuple[list[Finding], Ledger]:
    """Findings about how certain one turn's answer sounds, and its ledger."""
    turn = record.get("turn", 0)
    ledger = build(record)
    findings = [Finding(turn, INTERPRETATION, FAIL, "overstated_claim", f"“{c.text[:200]}” — {c.note}")
                for c in ledger.overstated]
    reading = record.get("reading_check") or {}
    changed = [o for o in reading.get("outcomes") or [] if isinstance(o, dict) and o.get("status") == "changes"]
    if changed and "Reading check." not in (record.get("assistant") or ""):
        # The check found a reading that changes the conclusion and the answer
        # does not carry CONDUCTOR's note about it (an older or edited record).
        findings.append(Finding(turn, INTERPRETATION, WARN, "reading_dependent_unstated",
                                "; ".join(f"{o.get('label')}: {o.get('after')}" for o in changed)))
    return findings, ledger
