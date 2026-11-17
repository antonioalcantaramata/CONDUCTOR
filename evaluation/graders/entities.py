"""
Entities — does the answer say each element is what the network says it is.

Thin wrapper over `llm_agent.agent.entities.check_answer`, which also runs live
(the answer review and the audit panel), on the facts the turn's record keeps
(`network_facts`). An element called something the network states otherwise,
or one it does not have, is a `fail`; a source or a level the network does not
state is a warning — the network may know more than it says.
"""

from __future__ import annotations

from llm_agent.agent.entities import CONTRADICTED, UNSUPPORTED, check_answer

from ..session_log import FAIL, INTERPRETATION, WARN, Finding

_SEVERITY = {CONTRADICTED: FAIL, UNSUPPORTED: WARN}


def grade_entities(record: dict) -> list[Finding]:
    turn = record.get("turn", 0)
    return [Finding(turn, INTERPRETATION, _SEVERITY[i.kind], i.kind, f"“{i.text[:200]}” — {i.note}")
            for i in check_answer(record.get("assistant") or "", record)]
