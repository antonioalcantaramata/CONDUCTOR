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
| **Corrective options** — several ways to remove the violations, side by side | *Give me options to clear the violations at 18:00.* |
| **Options with a branch held lower** — every option keeps chosen lines below a share of their rating | *Give me options at 18:00 that keep L30 below 80 %.* |
| **N-1 secure options** — dispatches that stay within limits after the critical outages | *Give me options at 20:00 that are secure under N-1.* · *What would a preventive N-1 secure dispatch cost?* |
| **Robust OPF** — a dispatch that stays secure with a stated confidence under uncertainty | *Find a robust corrective dispatch that is secure 95 % of the time.* |
| **Flexibility envelope** — how far one unit can move while the grid stays secure | *How much can Gen_bus1 move up or down at 18:00?* |
| **Hosting capacity** — how much new generation a bus can take | *What is the hosting capacity at the most heavily loaded bus?* · *Compare hosting capacity at Bus 9 for unity and 0.95 power factor.* |

### Corrective options

Ask for *options* or *alternatives* and CONDUCTOR returns up to five corrective
dispatches instead of one, each the best answer to a different question:

| Option | What it does |
| --- | --- |
| **Least disturbance** | The smallest change to the schedule (the default dispatch) |
| **Cheapest** | The lowest cost of the action at the [cost table's](system-tab.md#costs) prices |
| **Fewest units** | As few units as possible get an instruction; every other unit keeps its active power and its set-point |
| **Without *unit*** | The unit whose active power the least-disturbance option moves most gets no instruction |
| **Least import** | The smallest import from the external grid, then the smallest change |
| **Most headroom** | Every line as far below its rating as possible: the lowest highest loading in the network — spare capacity for what the forecast missed, at a cost |

Every option meets every limit and is scored the same way: units called, MW and
Mvar moved, import, the cost of the action (what it costs against the schedule;
doing nothing costs 0) and how far it is above the cheapest, the lowest voltage
and the highest loading.

**Checked by a power flow** — every option is a solution of the optimisation;
CONDUCTOR then applies its dispatch to the network and solves an ordinary AC
power flow, independently of the optimisation's own model, and the table's last
column says what it found: *✓ confirmed* (within the voltage band, at most 100 %
of every rating, every unit within its reactive limits), or *✗* and the first
limit it breaks. An option the power flow does not confirm is reported as such.

**Units called** are the units that get an instruction: a new active power, a
new voltage set-point for a unit that holds a voltage, or a new reactive output
for one that does not. A unit that keeps its set-point still changes its
reactive output as the grid around it changes; that is not an instruction, so
it is not counted (it does show in the Mvar chart). Each option's dispatch lists
every instruction, set-points included. A table lists them, and bar charts show what each asks
of each unit.

Options that come out the same as another, or no better than another on any
count, are left out, and the answer says which and why; so is one beyond the
five shown (the one closest to another option; one that is best of all at something,
such as the lowest import or the most headroom, goes last). A cost below zero is a saving
against the schedule (less import, or less reactive exchange beyond the free
band). CONDUCTOR does not pick one: the choice is the operator's — the
[advisor](advisor.md#discussing-corrective-options) helps think it through. When
the schedule already meets every limit, the answer is that no action is needed.

### Screening an option under N-1

A plain N-1 screen runs on the schedule. To screen an option instead, ask
*Is the cheapest option N-1 secure?* — or use **N-1 screen of an option** under
the answer, which puts the request in CONDUCTOR's box. Each option carries a
reference (*S2:cheapest*: its set and strategy); the screen applies that
option's dispatch at its own moment, network and load level, and screens every
single outage. The table's title says which option was applied. Options can
differ a lot here: at the 30-bus demo's worst hour the cheapest option leaves 31
of 41 outages with violations, least import all 41.

An option computed after an outage is screened with that outage kept out — the
**N-1-1** screen: is the corrected state secure against a further outage?

The screen applies the option as a fixed dispatch; it does not redispatch after
each outage. Options are kept until the backend restarts or another network is
loaded, and are refused when the unit models, names or load power factor have
changed since — compute them again then.

### After an outage

The same works for the corrective action after an outage: *If line L36 trips at
noon on 3 January, what are my options?* Each option is priced against the
schedule after the outage and checked by a power flow of the network without
the line; the table's title names the outage. When no local redispatch can
clear what the outage causes, the answer says so and why.

### Near-optimal alternatives

The cheapest option is cheapest by the cost table — which knows nothing of how
far you trust a unit or how many instructions you would rather send. Ask *Are
there other ways that cost about the same?* (or name a margin: *within 20 %*)
and CONDUCTOR returns the cheapest option and up to three **near-optimal
alternatives**: each costs at most that margin above the cheapest (10 % unless
you say otherwise) and changes other units. Each says, from the solver, how far
above the cheapest it is and which units it **avoids** or **adds** — for example
*10.0 % above the cheapest; avoids Gen_bus10; adds Gen_bus4* — and the chart
title states the margin.

They are found by *modelling to generate alternatives* (the Hop-Skip-Jump
method): each alternative keeps within the cost margin while moving as little
power as possible on the units earlier options moved, so it finds another way
when there is one. When none exists within the margin, the answer says so; a
wider margin may find some.

Rule a unit out and ask again: *Give me options without Gen_bus1.* The unit
stays in service and gets no instruction in any option — its active power and
its set-point stay as scheduled (unlike a unit taken out of service) — and the
table's subtitle says so.

### Keeping a line below a share of its rating

Name a line or transformer and a share of its rating: *Give me options at 18:00
that keep L30 below 80 %.* Every option then holds that branch at most at 80 %
of its rating, instead of the usual safety margin (90 %), and the power-flow
check holds each option to the same cap — an option the power flow finds above
it is marked *✗ … above its 80 % cap*. Use it to keep headroom on a line that
overloads after outages, or that you would rather not load today. A branch can
be named by its tag (*L30*), its buses (*Bus_21 -> Bus_23*) or its full name;
a name CONDUCTOR does not know is refused with examples. Caps run from 10 % to
100 % of the rating. The table's subtitle lists them. The same works after an
outage.

The **Most headroom** option asks a related question without naming a line:
the dispatch whose most loaded branch is as low as it can be.

### Options secure after an outage (N-1)

Every corrective option meets every limit **with everything in service**. Ask
for options that are *secure under N-1* and CONDUCTOR also looks at what happens
when a line or transformer is lost — a security-constrained OPF (SCOPF). There
are two ways to be secure, and you can ask for either or both:

- **N-1 preventive** — the dispatch alone stays within limits after each studied
  outage: when one happens, nothing has to be done. Every unit keeps its active
  power; a unit that holds a voltage keeps its set-point.
- **N-1 corrective** — after each studied outage a quick action of the units
  brings the grid back within limits: each unit may move its active power by at
  most a share of its capacity (20 % unless you say otherwise — what it can
  deliver soon after an outage) and change its voltage set-point. The dispatch is
  chosen so that such an action exists for every outage studied, and the answer
  includes the **corrective plan**: what to do after each outage.

Corrective is cheaper than preventive — it relies on acting after the outage —
so ask for both when you want to see the price of each.

**Which outages.** CONDUCTOR screens every single outage with the
least-disturbance option and studies the most critical ones — at most eight at
once, the worst first; then it screens the new dispatch and adds outages that
now fail. You can name your own: *… secure against the loss of L31 and L32*.
An outage that leaves a bus without supply is skipped — no dispatch can supply
an island — and the answer says so, as it does for the outages beyond the eight.

**After an outage** a line may carry up to the network's loading limit — 100 %
of its rating on the bundled networks, the same limit every N-1 screen uses, not
the 90 % margin the optimisation keeps in normal operation (you can set another) —
and voltages must stay in the usual band. So the study's screen and a plain N-1
screen of the same option agree.

**What you get.** The options table holds the least-disturbance and cheapest
options for reference and the secure ones. Then:

- **N-1 screen of each option** — every option is screened under N-1 the same way:
  how many of the outages leave violations, how many violations, how many of
  the studied outages it secured, and which it could not. A corrective option is
  screened *before* its action — the plan is what removes the rest.
- **Corrective plan** — one row per studied outage: the change each unit makes
  (MW, and a new voltage set-point), or *no action needed* when the dispatch
  already survives that outage. Each plan is checked by an ordinary AC power flow
  of the network without the outaged branch: *✓ confirmed*, or *✗* and the first
  limit still broken.

At the 30-bus demo's evening peak (6 January, 20:00), the cheapest option
leaves 29 of the 41 outages with violations. The preventive dispatch brings that
to 16 at 783.83 €/h instead of 26.48 €/h; the corrective one secures more of the
studied outages, at 281.25 €/h, by acting after them. Some outages cannot be
secured by any dispatch of the five units — the answer lists them with what
remains (for example *Bus_23 -> Bus_24 [L32] at 133.2 %*): those need another
remedy (a topology change, load relief, a new line).

**What secure means here.** Secure for the outages studied, at that moment, at
that load — not for every hour. The screen shows every outage; the study covers
the most critical. A dispatch is never called N-1 secure beyond what its screen
shows.

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
- **CONDUCTOR asks when it has to.** When a question leaves open something that
  changes the answer, and one study cannot cover every reading — which of three
  lines into a bus you want taken out, how far below its rating "comfortably"
  means — it asks one short question before running anything, with numbered
  choices and its suggestion first. Answer with the number or in your own words.
  Otherwise it does not ask: it uses the default and says which (the current
  moment for "now", the network's own limits), or covers the readings at once —
  preventive and corrective N-1 options together, one assessment showing every
  line.
- **Name what differs.** "with loads at 130 %", "with a 90 % line limit", "with
  voltage limits 0.95–1.05": these become study arguments, and the chart
  captions say so.
- **Ask for comparisons in one question.** "Compare power factor 0.90, 0.95 and
  1.0" runs the three studies at the same moment and labels each chart.
- **A result at one moment is about that moment.** For a verdict about a
  period, ask for a scan or the worst moment; for security, ask for N-1 too.
- **Follow-ups keep context.** "And at 19:00?" or "Now on the proposal" reuse
  what the previous question set up.
