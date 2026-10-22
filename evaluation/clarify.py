"""
Clarification evaluation — does the agent ask when it should, and only then?

The rule it is held to (system prompt, behavioural rule 32): ask one short
question, with numbered choices, before running anything when the request
leaves open something that changes the result, one study cannot show every
reading, and no standard default fits. Otherwise run, with the default stated,
or cover the readings in one study. The tasks test both ways — asking when it
should, and not asking when it should not: an agent that asks about everything
fails as surely as one that guesses.

    python -m evaluation.clarify list
    python -m evaluation.clarify run --repeats 10 [--tasks which_line,secure_now]
    python -m evaluation.clarify grade evaluation_runs/clarify-20261022-2130

`run` drives the real agent against the running backend and SPENDS MODEL
CREDIT (it asks for confirmation unless --yes). Its records go to their own
folder, never to the session logs. `grade` reads a folder of runs and costs
nothing. The tasks are written for the IEEE 30-bus demo (the default).

Grading is deterministic, from the records. "Asked first" means the first
answer holds a question and the turn called nothing but lookups (finding an
element, reading the clock); the second message of an asking task answers it,
and the check is that the agent then ran the study the answer chose.
"""

from __future__ import annotations

import sys

from .whatif import Suite, Task, main

PEAK = "20:00 on 6 January"

TASKS: tuple[Task, ...] = (
    # Should ask: a study of one element, and three lines fit (L30, L31, L32).
    Task("which_line",
         (f"At {PEAK}, simulate losing the line into Bus_23.", "The one from Bus_22."),
         ({"kind": "asked_first", "label": "asked which line before running"},
          {"kind": "choices", "label": "offered numbered choices"},
          {"kind": "ran", "label": "then simulated the outage of L31", "turn": 1, "tool": "simulate_contingency",
           "args": {"element_type": "line", "element_index": 31}})),
    # Should ask: a margin implied, not given.
    Task("unstated_margin",
         (f"At {PEAK}, give me options that keep L30 comfortably below its rating.", "75 %."),
         ({"kind": "asked_first", "label": "asked how far below before running"},
          {"kind": "result", "label": "then held L30 at 75 % in every option", "tool": "optimize_flexibility",
           "fields": {"branch_limits": {"Bus_21 -> Bus_23 [L30]": 75.0}}})),
    # Should not ask: "now" is the simulation clock.
    Task("secure_now",
         ("Is the grid secure right now?",),
         ({"kind": "ran", "label": "assessed without asking", "tool": "run_rsa"},)),
    # Should not ask: preventive or corrective unsaid — one study covers both.
    Task("n1_unspecified",
         (f"At {PEAK}, give me options that stay secure if a line trips.",),
         ({"kind": "ran", "label": "ran the N-1 study without asking", "tool": "optimize_flexibility"},
          {"kind": "options", "label": "covered preventive and corrective (n1=both)", "args": {"n1": "both"}})),
    # Should not ask: one sweep of every outage answers it.
    Task("worst_outage",
         (f"Which single outage is the most critical at {PEAK}?",),
         ({"kind": "ran", "label": "screened every outage without asking", "tool": "simulate_all_contingencies"},)),
)

CLARIFY = Suite("clarify", TASKS, (), "30", "Demo: IEEE 30 · active only")


if __name__ == "__main__":
    sys.exit(main(suite=CLARIFY, doc=__doc__))
