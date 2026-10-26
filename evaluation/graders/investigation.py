"""
Investigation — does the answer keep to what the investigation established.

Thin wrapper over `llm_agent.agent.investigation_record.check_answer`, which
also runs live (the answer review and the audit panel). An untested hypothesis
stated as a finding is a `fail`: it is a cause no study tested, presented as
one that was. Untested hypotheses the answer never says were untested, and a
supported one it never names, are warnings.
"""

from __future__ import annotations

from llm_agent.agent.investigation_record import check_answer

from ..session_log import FAIL, INTERPRETATION, WARN, Finding

_SEVERITY = {"untested_as_finding": FAIL, "untested_unsaid": WARN, "hypothesis_unsaid": WARN,
             "coverage_unsaid": WARN, "data_doubt_unsaid": FAIL}


def grade_investigation(record: dict) -> list[Finding]:
    turn = record.get("turn", 0)
    return [Finding(turn, INTERPRETATION, _SEVERITY[i.kind], i.kind, f"“{i.text[:200]}” — {i.note}")
            for i in check_answer(record.get("assistant") or "", record)]
