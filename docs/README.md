# CONDUCTOR documentation

<div align="center" markdown>
![CONDUCTOR logo](assets/logo-on-light.svg#only-light){ width="220" }
![CONDUCTOR logo](assets/logo-on-dark.svg#only-dark){ width="220" }
</div>

CONDUCTOR is an assistant for studying distribution grids in plain language.
You ask a question; it runs the power-system studies that answer it (security
assessment, N-1 contingencies, risk, optimal and robust dispatch, flexibility,
hosting capacity, KPIs), shows the results as charts, and tells you where every
figure in its answer came from. Corrective actions come as *options* to choose
from — cheapest, least disturbance, near-optimal alternatives, dispatches that
stay secure after an outage — each re-checked by a power flow. Changes to the
grid can be tried as *proposals* without touching the loaded network, and an
*advisor* helps you think results and options through. Ask *why* something is
wrong and it **investigates**: what changed since it was fine, how much of the
change each cause explains, whether it recurs and whether it could have been
avoided. Ask what is coming and it reads the forecast: when to decide, and what
would prevent each episode.

These pages are for the people who use it: **grid operators and planners**
asking questions about a network, and **researchers** using CONDUCTOR in
studies or evaluating it. To install and launch it, start with the
[main README](https://github.com/antonioalcantaramata/CONDUCTOR#readme).

## Guide

| Page | What it covers |
| --- | --- |
| [Getting started](guide/getting-started.md) | The launch screen, your first questions, the layout of the app |
| [Choosing a model](guide/choosing-a-model.md) | Free, paid and local options, their limits and data terms |
| [Data and the operating point](guide/data-and-operating-point.md) | Measurements, forecasts and the nominal point; uploading a network and data; data quality |
| [Asking questions](guide/asking-questions.md) | What CONDUCTOR can study, with example questions for each kind of study — corrective options, N-1 secure dispatches, *why* questions and what is coming among them — Deep Research, and when it asks you first |
| [Reading an answer](guide/reading-an-answer.md) | "What backs this answer", the investigation card and report, charts and their captions, answer checks, answers computed under other settings |
| [What-if proposals](guide/what-if-proposals.md) | Changing the grid without touching it: every kind of change, combining and comparing proposals |
| [The System tab](guide/system-tab.md) | The network map, generator models, costs, the load power factor, renaming elements |
| [The advisor](guide/advisor.md) | Discussing results and corrective options, and getting suggested next steps |
| [Sessions and scenarios](guide/sessions-and-scenarios.md) | Resume and the demos, what is kept between sessions, Start fresh, session context, the session reports (raw and written by the model) |
| [Troubleshooting](guide/troubleshooting.md) | Limits reached, slow or retired models, common problems |

## How it works

What is behind an answer: the models, the mathematics and the algorithms of
every study, the conventions they share and what they assume — for engineers and
researchers who want to know exactly what a figure means.

| Page | What it covers |
| --- | --- |
| [Overview](methods/index.md) | The division of labour, the principles every study follows, conventions and notation |
| [Network, data and the operating point](methods/network-and-data.md) | From a network file and a moment of data to a solvable operating point |
| [Power flow and security assessment](methods/power-flow-and-security.md) | The AC power flow and the limits |
| [Contingency analysis (N-1)](methods/contingency-analysis.md) | Single outages, the N-1 screen, N-1-1, attribution |
| [Investigating a cause](methods/investigation.md) | Why an element is outside its limits: what changed, and how much of the change is each cause; the agent's hypotheses and record, recurrence, prevention, the data check, looking ahead, Deep Research |
| [The optimal power flow](methods/optimal-power-flow.md) | The AC OPF behind every corrective action |
| [Corrective options](methods/corrective-options.md) | The strategies, the cost of an action, the power-flow check, the last resort |
| [Near-optimal alternatives](methods/near-optimal-alternatives.md) | Modelling to generate alternatives |
| [N-1 security](methods/n1-security.md) | Preventive and corrective security-constrained OPF |
| [Studies over time](methods/over-time.md) | Holding an option, one plan for a window, security over a window |
| [Key performance indicators](methods/key-performance-indicators.md) | The schedule against a corrective action |
| [Risk, uncertainty and capacity](methods/risk-and-capacity.md) | Probabilistic assessment, robust dispatch, envelopes, hosting capacity, historical risk |
| [The agent and its checks](methods/the-agent.md) | The orchestrator, the advisor and the checks on every answer |
| [Assumptions and limits](methods/assumptions-and-limits.md) | What the models leave out, and how they have been tested |

## For researchers

| Page | What it covers |
| --- | --- |
| [Using CONDUCTOR in research](research/using-conductor-in-research.md) | Reproducible demos, session logs, grading sessions, the what-if, options, clarification, investigation and narration evaluations, what the checks can and cannot see |

## Reference

| Page | What it covers |
| --- | --- |
| [Tools](reference/tools.md) | Every study and setting CONDUCTOR can use, with its arguments (generated) |
| [Model options](reference/providers.md) | Every model option with its limits, sources and data terms (generated) |
| Backend API | Interactive documentation served by the running backend at <http://localhost:8000/docs> |
