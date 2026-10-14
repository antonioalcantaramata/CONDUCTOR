# Asking questions

Ask in plain language. CONDUCTOR chooses the studies, runs them on the solver,
and answers from their results. It never computes grid figures itself: every
number in an answer comes from a study, and the panel under the answer shows
which (see [Reading an answer](reading-an-answer.md)).

Below, what it can study, with example questions. The exact tools and their
arguments are in the [tool reference](../reference/tools.md).

## The state of the grid

| Study | Ask, for example |
| --- | --- |
| **Security assessment** — a power flow at one moment: voltages, line and transformer loading, violations | *Is the grid secure at the current timestamp?* · *Is it secure at 18:00 with loads at 120 %?* |
| **Current conditions** — loads, generation and exchange with the grid at one moment | *What is each substation consuming right now?* |
| **An element over time** — one bus, line or unit across the data | *Show the loading of the Bus_1 – Bus_4 line over the week.* |
| **Where an element is** — what a bus, line or transformer connects to | *What is connected to Bus 5?* |
| **What drives a violation** — which loads and units contribute most | *Why is Bus 13 under-voltage at 18:00, and what drives it?* |

## Contingencies (N-1)

| Study | Ask, for example |
| --- | --- |
| **One outage** | *What happens if the line Bus_0 – Bus_1 trips?* |
| **Every single outage** — each line and transformer out in turn | *Which single N-1 outage is the most critical right now?* · *Is the grid N-1 secure at the worst moment of the week?* |
| **Corrective dispatch after an outage** | *If that line trips, can redispatch restore security, and how?* |

## Over time, scenarios and risk

| Study | Ask, for example |
| --- | --- |
| **Scan a window** — the security assessment at every step | *How many hours this week had a violation?* |
| **Worst moment** — the step with the lowest voltage, highest loading, … | *Find the worst-case timestamp in this week's measurements.* |
| **Renewable scenarios** — the same window with generation scaled up or down | *What if solar and wind production doubled — where does it break?* |
| **Probabilistic assessment** — many samples of load and generation around one moment | *Run a probabilistic risk assessment under load and generation uncertainty.* |
| **Historical risk** — how often and how badly an element violated in the data | *How often does Bus 13 go under 0.94 p.u., and when?* |

## Optimisation and flexibility

| Study | Ask, for example |
| --- | --- |
| **Flexibility OPF** — the smallest change in dispatch that removes violations | *Find a corrective dispatch to clear the violations at 18:00.* |
| **Robust OPF** — a dispatch that stays secure with a stated confidence under uncertainty | *Find a robust corrective dispatch that is secure 95 % of the time.* |
| **Flexibility envelope** — how far one unit can move while the grid stays secure | *How much can Gen_bus1 move up or down at 18:00?* |
| **Hosting capacity** — how much new generation a bus can take | *What is the hosting capacity at the most heavily loaded bus?* · *Compare hosting capacity at Bus 9 for unity and 0.95 power factor.* |

## Key performance indicators

| Study | Ask, for example |
| --- | --- |
| **KPIs at one moment** — flexibility needed, margins and similar indicators | *Evaluate the flexibility KPIs for the current operating point.* |
| **KPIs over the forecast** | *How do the KPIs evolve over the forecast horizon?* |

## Changes to the grid

What-if questions build a **proposal** — the network plus your changes —
without touching the loaded network, and compare it with today's grid:

- *What if we connect a 20 MW data centre at Bus 9?*
- *Add a line between Bus 0 and Bus 3 and compare the N-1 screen.*
- *Grow every load by 20 % and compare the week with today.*

See [What-if proposals](what-if-proposals.md).

## Settings you can change by asking

- *Use a load power factor of 0.92 from now on.*
- *Put Gen_bus1 on voltage control at 1.02 p.u.*
- *Set the up-regulation cost of Gen_bus1 to 80 €/MWh.*
- *Rename Bus 3 to North.*

These change every later study; CONDUCTOR says so. The same settings are in the
[System tab](system-tab.md).

## Tips

- **Name the moment and the data.** "at 18:00", "over the forecast", "at the
  nominal point". Without one, CONDUCTOR uses the operating point in the
  sidebar and says which it used.
- **Name what differs.** "with loads at 130 %", "with a 90 % line limit", "with
  voltage limits 0.95–1.05": these become study arguments, and the chart
  captions say so.
- **Ask for comparisons in one question.** "Compare power factor 0.90, 0.95 and
  1.0" runs the three studies at the same moment and labels each chart.
- **A result at one moment is about that moment.** For a verdict about a
  period, ask for a scan or the worst moment; for security, ask for N-1 too.
- **Follow-ups keep context.** "And at 19:00?" or "Now on the proposal" reuse
  what the previous question set up.
