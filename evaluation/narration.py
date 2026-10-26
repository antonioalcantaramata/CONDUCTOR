"""
Narration evaluation — is a study reported as its result says it must be?

How to report six studies (scenario scans, the probabilistic assessment, the
robust dispatch, the flexibility envelope, hosting capacity, historical risk)
is not in the system prompt: it comes with each study's result, under
`how_to_report` (llm_agent/agent/reporting_notes.py), so a question pays for it
only when the study ran. These tasks check that the answer follows it — one
task per study, each check a line of its note.

    python -m evaluation.narration list
    python -m evaluation.narration run --repeats 2 [--tasks hosting,robust]
    python -m evaluation.narration grade evaluation_runs/narration-20261026-2130

`run` drives the real agent against the running backend and SPENDS MODEL
CREDIT (it asks for confirmation unless --yes). The tasks are written for the
IEEE 30-bus demo (the default). Grading is deterministic, from the records.
"""

from __future__ import annotations

import sys

from .whatif import Suite, Task, main

PEAK = "20:00 on 6 January"

TASKS: tuple[Task, ...] = (
    Task("hosting",
         ("How much more generation can Bus_13 host?",),
         ({"kind": "ran", "label": "ran the hosting capacity", "tool": "compute_hosting_capacity"},
          {"kind": "mentions", "label": "glossary: unity", "any": ["unity"]},
          {"kind": "mentions", "label": "glossary: fixed power factor", "any": ["fixed_pf", "fixed power factor",
                                                                                 "fixed-power-factor"]},
          {"kind": "mentions", "label": "glossary: the reactive proxy is a proxy", "any": ["proxy"]})),
    Task("probabilistic",
         (f"What is the probability of a violation at {PEAK}?",),
         ({"kind": "ran", "label": "ran the probabilistic assessment", "tool": "run_probabilistic_rsa"},
          {"kind": "mentions", "label": "led with the probability of any violation",
           "any": ["probability", "chance", "%"]},
          {"kind": "mentions", "label": "explained the uncertainty modelled",
           "any": ["demand uncertainty", "load uncertainty", "uncertainty in demand", "uncertainty in load",
                   "load_sigma", "sampled", "samples"]})),
    Task("robust",
         (f"Give me a dispatch at {PEAK} that stays within limits with 95 % confidence.",),
         ({"kind": "ran", "label": "ran the robust dispatch", "tool": "optimize_robust_flexibility"},
          {"kind": "mentions", "label": "reported the risk out of sample",
           "any": ["out of sample", "out-of-sample", "validation", "validated", "after validation"]},
          # What the method did, or why it fell short: the back-off tightens voltage limits only, so a
          # missed target with a thermal risk is explained by that (observed live: "the residual risk
          # was mainly thermal") rather than by the list of tiny back-offs.
          {"kind": "mentions", "label": "explained the back-off, or why it fell short",
           "any": ["back-off", "backoff", "back off", "tighten", "voltage limits", "thermal"]})),
    Task("envelope",
         ("How much reactive power can Gen_bus7 inject or absorb right now?",),
         ({"kind": "ran", "label": "ran the flexibility envelope", "tool": "compute_flexibility_envelope"},
          {"kind": "mentions", "label": "said it holds at this operating point only",
           "any": ["operating point", "this moment", "current conditions", "current state", "right now",
                   "present output", "present active", "at this time"]},
          {"kind": "mentions", "label": "named the binding constraint", "any": ["binding", "limit", "limited by"]})),
    Task("historical_risk",
         ("How often does Bus_29 violate its limits over the week, and how often does it come close?",),
         ({"kind": "ran", "label": "ran the historical risk", "tool": "compute_historical_risk"},
          {"kind": "mentions", "label": "gave the near misses", "any": ["near miss", "near-miss", "close to", "came within", "within 0."]},
          {"kind": "mentions", "label": "told the worst episodes", "any": ["episode"]})),
    Task("renewables",
         ("What if renewable output drops 30 %? Compare with the baseline over the week.",),
         ({"kind": "ran", "label": "ran the scenario scan", "tool": "scan_scenarios"},
          {"kind": "mentions", "label": "compared with the baseline", "any": ["baseline", "×1.0", "x1.0", "1.0"]},
          {"kind": "mentions", "label": "stated the uniform-scaling limitation",
           "any": ["uniform", "all together", "every unit", "all units", "same factor", "no renewable",
                   "no wind", "no solar", "has no"]})),
)

NARRATION = Suite("narration", TASKS, (), "30", "Demo: IEEE 30 · active only")


if __name__ == "__main__":
    sys.exit(main(suite=NARRATION, doc=__doc__))
