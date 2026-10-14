# CONDUCTOR documentation

<div align="center" markdown>
![CONDUCTOR logo](assets/logo-on-light.svg#only-light){ width="220" }
![CONDUCTOR logo](assets/logo-on-dark.svg#only-dark){ width="220" }
</div>

CONDUCTOR is an assistant for studying distribution grids in plain language.
You ask a question; it runs the power-system studies that answer it (security
assessment, N-1 contingencies, risk, optimal and robust dispatch, flexibility,
hosting capacity, KPIs), shows the results as charts, and tells you where every
figure in its answer came from. Changes to the grid can be tried as
*proposals* without touching the loaded network, and an *advisor* helps you
think results through.

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
| [Asking questions](guide/asking-questions.md) | What CONDUCTOR can study, with example questions for each kind of study |
| [Reading an answer](guide/reading-an-answer.md) | Charts and their captions, "How this answer was produced", answer checks, answers computed under other settings |
| [What-if proposals](guide/what-if-proposals.md) | Changing the grid without touching it: every kind of change, combining and comparing proposals |
| [The System tab](guide/system-tab.md) | The network map, generator models, costs, the load power factor, renaming elements |
| [The advisor](guide/advisor.md) | Discussing results and getting suggested next steps |
| [Sessions and scenarios](guide/sessions-and-scenarios.md) | Resume and the demos, what is kept between sessions, Start fresh, the session report |
| [Troubleshooting](guide/troubleshooting.md) | Limits reached, slow or retired models, common problems |

## For researchers

| Page | What it covers |
| --- | --- |
| [Using CONDUCTOR in research](research/using-conductor-in-research.md) | Reproducible demos, session logs, grading sessions, the what-if evaluation, what the checks can and cannot see |

## Reference

| Page | What it covers |
| --- | --- |
| [Tools](reference/tools.md) | Every study and setting CONDUCTOR can use, with its arguments (generated) |
| [Model options](reference/providers.md) | Every model option with its limits, sources and data terms (generated) |
| Backend API | Interactive documentation served by the running backend at <http://localhost:8000/docs> |
