"""
evaluation — offline grading of what the agent actually did.

The reviewers' objection to the submitted paper was that a 98.5% pass rate on a
prompt catalog says little: the tests looked easy, and a pass rate says nothing
about *how* the agent fails when it does. The failures found since are all of a
kind that a pass/fail score cannot see — a correct figure attached to the wrong
operating point, a recommendation that inverted the flag it was reading, an
answer composed from two different days.

So this package grades turns rather than scoring them, and its output is a list
of findings by failure channel, from which a pass rate can be derived if one is
wanted.

Two properties are deliberate:

*The oracle is the system's own verification layer.* `graders.integrity` runs
`TurnConsistency` and `operating_context` from `llm_agent.agent.validators` —
the same code that runs live. There is no LLM judge and nothing hand-labelled,
so a grade is exactly reproducible and can be recomputed on old logs.

*The input is the session log, not a bespoke run format.* Anything the agent
has ever done can be graded after the fact, including a real operator session.
The batch runner and prompt catalog build on top of this; they are not
prerequisites for it.
"""

from .graders.integrity import grade_composition
from .graders.provenance import grade_provenance
from .session_log import Finding, load_records, newest_log
from .report import grade_record, render_report

__all__ = [
    "Finding",
    "grade_composition",
    "grade_provenance",
    "grade_record",
    "load_records",
    "newest_log",
    "render_report",
]
