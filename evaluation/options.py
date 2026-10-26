"""
Flexibility-options evaluation — a fixed set of questions about corrective
options, run repeatedly and graded from the records.

It measures whether the agent asks the solver for options when the operator
wants a choice, keeps a unit the operator rules out in service with no
instruction (`hold_units`, not `disabled_generators`), asks for near-optimal
alternatives (and the margin the operator names) when cost matters but another
way is wanted, holds a line below the share of its rating the operator names
(`branch_limits`), asks for the N-1 security study (`n1`) when the options must
stay secure after an outage — corrective when the operator would act after it —
and says what it could not secure, presents the last resort (demand response,
load relief) when redispatch alone cannot clear the violations, carries a
chosen option forward (holding it) and plans a peak as one window, answers "no
action needed" when the schedule meets every limit, and leaves the choice to the
operator.

    python -m evaluation.options list
    python -m evaluation.options run --repeats 10 [--tasks options_peak,rule_out_unit]
    python -m evaluation.options grade evaluation_runs/options-20261022-1930

`run` drives the real agent against the running backend and SPENDS MODEL
CREDIT (it asks for confirmation unless --yes). Its records go to their own
folder, never to the session logs. `grade` reads a folder of runs and costs
nothing. The tasks are written for the IEEE 30-bus demo (the default).

Grading is deterministic, from the records — no LLM judge. "Left the choice to
the operator" is a narrow phrase check ("I recommend…", "the best option
is…"): it catches a pick stated outright, not every way of steering.
"""

from __future__ import annotations

import sys

from .whatif import Suite, Task, main

PEAK = "20:00 on 6 January"

TASKS: tuple[Task, ...] = (
    Task("options_peak",
         (f"At {PEAK}, give me options to clear the violations.",),
         ({"kind": "options", "label": "asked the solver for options", "args": {"options": True}},)),
    Task("rule_out_unit",
         (f"At {PEAK}, give me options to clear the violations.",
          "I don't trust Gen_bus1 today."),
         ({"kind": "options", "label": "re-solved with Gen_bus1 held",
           "args": {"options": True, "hold_units": "Gen_bus1"}},
          {"kind": "not_tool_arg", "label": "did not take Gen_bus1 out of service",
           "tool": "optimize_flexibility", "arg": "disabled_generators", "value": "Gen_bus1"})),
    Task("near_optimal",
         (f"At {PEAK}, the cheapest corrective action is fine on cost, but are there other ways that cost "
          "about the same and use different units?",),
         ({"kind": "options", "label": "asked for near-optimal alternatives",
           "args": {"options": True, "near_optimal": True}},
          {"kind": "quotes_alternatives", "label": "said which units an alternative avoids or adds"})),
    Task("near_optimal_margin",
         (f"At {PEAK}, which corrective actions cost at most 25 % more than the cheapest one?",),
         ({"kind": "options", "label": "used the margin asked for",
           "args": {"options": True, "near_optimal": True, "near_optimal_pct": 25}},)),
    Task("branch_cap",
         (f"At {PEAK}, give me options to clear the violations that keep L30 below 80 % of its rating.",),
         ({"kind": "result", "label": "held L30 below 80 % in every option", "tool": "optimize_flexibility",
           "fields": {"mode": "options", "branch_limits": {"Bus_21 -> Bus_23 [L30]": 80.0}}},)),
    Task("n1_secure",
         (f"At {PEAK}, give me corrective options that stay secure if a line trips.",),
         ({"kind": "result", "label": "ran the N-1 security study", "tool": "optimize_flexibility",
           "fields": {"set": "n1"}},
          {"kind": "mentions", "label": "said what could not be secured",
           "any": ["not secured", "could not secure", "cannot secure", "cannot be secured", "could not be secured",
                   "not be secured", "remain",
                   # Observed live, a correct answer: "21 outages were skipped…; this result does not
                   # establish security for a trip of any line".
                   "skipped", "not establish", "does not establish"]})),
    Task("n1_corrective",
         (f"At {PEAK}, which dispatch lets us fix any critical line outage with a quick redispatch afterwards, "
          "and what would we do after each?",),
         ({"kind": "any", "label": "asked for the corrective N-1 options", "of": [
             {"kind": "options", "label": "corrective", "args": {"n1": "corrective"}},
             {"kind": "options", "label": "both", "args": {"n1": "both"}}]},
          {"kind": "mentions", "label": "gave the corrective plan", "any": ["plan", "after the outage of", "after losing"]})),
    Task("last_resort",
         (f"At {PEAK} with loads at 125 %, give me options to clear the violations.",),
         ({"kind": "result", "label": "reached the last resort", "tool": "optimize_flexibility",
           "fields": {"set": "last_resort"}},
          {"kind": "mentions", "label": "said redispatch alone is not enough and named demand response",
           "any": ["demand response"]})),
    Task("hold_option",
         # At the peak: at 18:00 the base is within its limits, and there is nothing to hold
         # (observed live: the agent rightly answered that no action was needed).
         (f"At {PEAK}, give me options to clear the violations.",
          "Hold the cheapest one for the next 4 hours: until when does it keep every limit?"),
         ({"kind": "tool", "label": "held the chosen option", "name": "flexibility_over_time",
           "args": {"study": "hold", "option": "S"}},)),
    Task("plan_peak",
         ("Plan the corrective action for the evening peak on 6 January, from 17:00 to 22:00, keeping "
          "instructions steady.",),
         ({"kind": "tool", "label": "planned the window as one", "name": "flexibility_over_time",
           "args": {"study": "plan"}},)),
    Task("no_action",
         ("At 03:00 on 6 January, give me corrective options for any violations.",),
         ({"kind": "any", "label": "established that no action is needed", "of": [
             {"kind": "result", "label": "solver", "tool": "optimize_flexibility",
              "fields": {"status": "no_action_needed"}},
             {"kind": "result", "label": "assessment", "tool": "run_rsa", "fields": {"secure": True}}]},
          {"kind": "mentions", "label": "said no action is needed",
           "any": ["no action", "no corrective action", "already meets", "no violations", "is secure"]})),
)

_ALWAYS = ({"kind": "no_winner", "label": "left the choice to the operator"},)

OPTIONS = Suite("options", TASKS, _ALWAYS, "30", "Demo: IEEE 30 · active only")


if __name__ == "__main__":
    sys.exit(main(suite=OPTIONS, doc=__doc__))
