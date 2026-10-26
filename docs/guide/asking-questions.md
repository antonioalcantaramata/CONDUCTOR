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
| **Why something is wrong** — what changed since the element was fine, and how much of the change each cause explains | *Why is Bus_29 below its limit at 20:00?* · *What caused the overload of L0 this evening?* · *Why does the proposal overload L7 when the base does not?* · *Why is the grid insecure at 20:00?* · *Why is losing L38 critical at 18:00 and not at 16:00?* · *Why was L0 overloaded for longer today than yesterday?* · *Compared with a quiet moment, why is Bus_29 so close to its limit?* · *Is the undervoltage at Bus_29 a recurring problem this week?* · *Could it have been avoided, and how much warning did we have?* |
| **What is coming** — the episodes the forecast expects, when to decide, what would prevent each, and whether that holds if the forecast is wrong | *What should I watch for in the next day of the forecast?* · *Anything to act on tomorrow at Bus_29?* |
| **What to move** — which units and loads move a violated element, and by how much they would have to | *What should I move to clear the undervoltage at Bus 13?* |

A *why* about a problem in the grid is investigated: CONDUCTOR compares the
element with the nearest earlier moment it was within its limits (or the
moment, dataset or proposal you name) and says which changes caused the
difference, with the share of each. If the element is not actually outside
its limits, the answer says so. If several are and you did not say which, it
asks. A *why* about how CONDUCTOR worked, about a recommendation or about
physics is answered directly. See [Investigating a cause](../methods/investigation.md).

When a cause is a measured load or unit that jumps, sits far outside its usual
range or stays frozen, the answer first says to **check the data**: the cause
may be a faulty meter rather than something that happened.

Ask what is coming and CONDUCTOR reads the forecast for the next day: each
episode outside the limits, its cause, the time to decide by, the smallest
action that would prevent it, and whether that action still holds if the
forecast errs against it: demand 5 % above it, or renewable output 5 % below it.
These are expectations, and the answer says so. See
[What is coming](../methods/investigation.md#what-is-coming-looking-ahead).

### Deep Research

For a *why* that deserves more than a first answer, click the **Deep Research**
icon (a microscope, beside **Run**) before sending the question. It turns blue
and shows its name while it is on. CONDUCTOR then investigates in depth:

- it tests its own hypotheses until they explain most of the change;
- it checks the cause against other moments;
- it asks whether the episode could have been avoided, and how much warning
  there was;
- it looks for whether the problem recurs over the week;
- it asks where CONDUCTOR's tools stop. One question is which analysis of the
  grid would settle the cause and does not exist yet (how far the bus is from
  voltage collapse, whether the units could have ramped in time). The other is
  what lies behind the cause outside the grid (why demand rose: the weather,
  the calendar). It cannot test these. The answer says so and names what
  would test them. The sidebar collects them under **Tools investigations
  needed**, as tools to build and data to integrate.

It takes one to three minutes. While it works, a card shows where it is and
what it has found so far; the list of studies stays closed under it.

The switch asks for depth; it does not force an investigation. If the question
is not a *why* about a problem in the grid (options, a what-if, an
assessment), CONDUCTOR answers it as usual. The card and the answer say *Deep
Research not needed — answered as an analysis*. If the investigation is about
to answer without part of its depth, CONDUCTOR reminds it once. It then either
does that part or says why it does not apply. The icon turns itself off after
each question.

## Contingencies (N-1)

| Study | Ask, for example |
| --- | --- |
| **One outage** | *What happens if the line Bus_0 – Bus_1 trips?* |
| **Every single outage** — each line and transformer out in turn; an outage that breaks a limit, or cuts load off, makes the grid not N-1 secure | *Which single N-1 outage is the most critical right now?* · *Is the grid N-1 secure at the worst moment of the week?* |
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
| **When redispatch is not enough** — demand response, load relief and switching, as a last resort | *At 20:00 with loads at 125 %, what are my options?* |
| **Carrying an option forward** — what follows from the option you chose | *Hold option S3:cheapest for the next 4 hours.* · *Starting from S3:cheapest, what are my options at 21:00?* |
| **Over a window** — one plan for a peak, the cost of N-1 security hour by hour | *Plan the corrective action from 17:00 to 22:00.* · *What does N-1 security cost this evening?* |
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
| **With demand response** | The cheapest action when flexible loads may also be reduced, at their [demand-response price](system-tab.md#costs) — shown only when it does use demand response (otherwise the answer says redispatch was cheaper) |

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

**How close the cheapest is to the best possible** — the optimisation finds a
good dispatch, not necessarily the best one. With the options (in the table's
title, and in the answer), CONDUCTOR states a cost no dispatch of the units can
beat under the same limits, from a convex relaxation of the same problem: at
the 30-bus demo's evening peak, *no dispatch
costs less than 22.48 €/h, none saves more than 4.00 €/h on the cheapest
(26.48 €/h)*. When the two meet, the cheapest option is called the best
possible; a larger gap means a cheaper dispatch may exist, not that one does
([how it is computed](../methods/corrective-options.md#how-close-the-cheapest-is-to-the-best-possible)).

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

### Carrying an option forward

Choosing an option is rarely the end: you want to know whether it holds after an
outage, for how long, and what comes next. Each option carries a reference
(*S2:cheapest*: its set and strategy), and **Use an option** under the answer
puts any of these in CONDUCTOR's box — or ask in your own words:

- **Its N-1 screen** — *Is the cheapest option N-1 secure?* A plain N-1 screen
  runs on the schedule; this one applies the option's dispatch and screens every
  single outage. Options can differ a lot here: at 18:00 on the 30-bus demo the
  cheapest option leaves 29 of 41 outages with violations, the fewest-units
  option 17. An option computed after an outage is screened with that outage kept out —
  the **N-1-1** screen.
- **The assessment with it applied** — *Run the security assessment with option
  S2:cheapest applied*, or one outage with it (*… if L31 trips*).
- **How long it holds** — *Hold option S2:cheapest for the next 4 hours.* At each
  moment after it, its **instructions** stay (the set-points it changed); every
  other unit follows its own schedule, as it would in operation. The charts show
  the highest loading and lowest voltage held against the schedule untouched,
  with the limits, and mark the first moment it no longer keeps every limit;
  ask for the N-1 screen at each moment too if you want it. The answer says when
  a new action is needed.
- **Its KPIs** — *Evaluate the KPIs with S2:cheapest.* The schedule against the
  option: what it needs, what it moves, what flexibility it leaves and what it
  costs (see [Key performance indicators](#key-performance-indicators)).
- **The next action from it** — *Starting from S2:cheapest, what are my options
  at 21:00?* The corrective options then start from the option's instructions
  held, not from the schedule — what you would actually face an hour later.

The option's own moment, network and load level always apply; a later moment
holds its instructions, an earlier one is refused. Options are kept on disk with
the network (beside its proposals), so a reference still works after a restart;
one computed before the unit models, names or load power factor changed is
refused with the reason — compute the options again then.

### After an outage

The same works for the corrective action after an outage: *If line L36 trips at
noon on 3 January, what are my options?* Each option is priced against the
schedule after the outage and checked by a power flow of the network without
the line; the table's title names the outage. When no local redispatch can
clear what the outage causes, the answer says so and why, and goes on to
[the last resort](#when-redispatching-the-units-is-not-enough) — after the loss
of L29 at that noon, demand response and load relief do.

### Near-optimal alternatives

The cheapest option is cheapest by the cost table — which knows nothing of how
far you trust a unit or how many instructions you would rather send. Ask *Are
there other ways that cost about the same?* (or name a margin: *within 20 %*)
and CONDUCTOR returns the cheapest option and up to five **near-optimal
alternatives**: each costs at most that margin above the cheapest (10 % unless
you say otherwise), and they are as different from each other as the margin
allows. Each says, from the solver, how far above the cheapest it is, how it was
found, and which units it **avoids** or **adds** — for example *10.0 % above the
cheapest; Gen_bus1 as high as it can be within the margin; avoids Gen_bus10*.

They are found by *modelling to generate alternatives* (MGA), several ways at
once, every one inside the same cost margin:

| Method | What it looks for |
| --- | --- |
| **Hop-Skip-Jump** | Moving as little as possible on the units earlier alternatives moved — another way, when there is one |
| **Unit ranges** | Each unit as low, and as high, as it can be |
| **Other objectives** | The fewest units, the least import, the most headroom, the least disturbance — each within the margin |
| **Random directions** | Seeded random directions in the space of unit outputs (the same question gives the same alternatives) |

From everything found, the ones shown are picked farthest-first — each the most
different from those already shown. You can ask for fewer (*give me two
alternatives*) or one method (*… by Hop-Skip-Jump only*).

The unit ranges give a chart of their own: **what each unit can do within the
margin** — a bar from its lowest to its highest output, with its schedule and
where the cheapest puts it. A short bar is a unit every cheap way needs at that
level; a long one is a unit you can trade. At the demo's evening peak, Gen_bus7
must rise by at least 13 MW in every alternative within 10 %, while Gen_bus10
can stay anywhere from its schedule to its maximum. When no alternative exists
within the margin, the answer says so; a wider margin may find some.

Rule a unit out and ask again: *Give me options without Gen_bus1.* The unit
stays in service and gets no instruction in any option — its active power and
its set-point stay as scheduled (unlike a unit taken out of service) — and the
table's subtitle says so.

### When redispatching the units is not enough

Sometimes no redispatch of the units clears the violations — the feeder is too
loaded, or the units too far away. CONDUCTOR then says so, with the reason, and
goes on to the remedies after redispatch, in a fixed order:

1. **Demand response** — a share of a load reduced on request (contracted
   flexibility, an aggregator), at its price;
2. **Load relief** — involuntary reduction of the rest, the last resort, at the
   value of lost load.

The **last resort** options are the **least load reduction** that, with
redispatch, clears every violation (demand response before any relief) and the
**cheapest** with loads reduced at their prices — each with the loads it reduces
(load, bus, MW of demand response and of relief) and checked by a power flow.
At the 30-bus demo's evening peak with loads at 125 %, 3.9 MW of demand response
at six loads (Bus_20 to Bus_29) does it, with no relief. CONDUCTOR also
screens **switching**: single openings an AC power flow found to clear or relieve
the violations — a screen, not an optimisation; protection and supply are yours
to check.

The prices — 10 % of every load as demand response at 150 €/MWh and relief at
3,000 €/MWh, synthetic until you change them — are in the
[System tab](system-tab.md#costs), per load, or by asking: *Bus_29's load has
2 MW of demand response at 90 €/MWh.* The default corrective dispatch never
touches a load; with options, the **With demand response** option shows whether
demand response would be worth calling at its price.

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
  deliver soon after an outage) and change its voltage set-point (a unit that
  does not hold a voltage, its reactive output). The dispatch is chosen so that
  such an action exists for every outage studied, and the answer includes the
  **corrective plan**: what to do after each outage.

Corrective is cheaper than preventive — it relies on acting after the outage —
so ask for both when you want to see the price of each.

**Which outages.** CONDUCTOR screens every single outage with the
least-disturbance option and studies the most critical ones — at most eight at
once, the worst first; then it screens the new dispatch and adds outages that
now fail. You can name your own: *… secure against the loss of L31 and L32*.
An outage that cuts load off is skipped — no dispatch can keep an island's load
supplied — and the answer says so, as it does for the outages beyond the eight.
An outage that only disconnects a unit is studied: the other units must cover
what it produced.

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
  already survives that outage. When moving the units is not enough, the plan
  also **reduces load after the outage** — demand response first, load relief
  only beyond it (ask without it: *… with no load reduction*). Each plan is
  checked by an ordinary AC power flow of the network without the outaged branch:
  *✓ confirmed*, or *✗* and the first limit still broken.
- **Switching that may help** — for an outage a dispatch cannot secure, single
  openings of another line after it that an AC power flow found to clear or
  relieve what remains.

At the 30-bus demo's evening peak (6 January, 20:00), the cheapest option
leaves 29 of the 41 outages with violations. The preventive dispatch brings that
to 18 at 784.00 €/h instead of 26.48 €/h; it cannot secure five of the eight
studied outages (for example after L35, *Bus_23 -> Bus_24 [L32] at 132.9 %*);
after the loss of L29, opening L13 relieves most of what remains, leaving L20
at 107.3 %. The corrective dispatch, at 276.54 €/h, secures all eight: four by
moving the units, four by also reducing load after the outage — after L35,
3.7 MW of relief at Bus_29 with demand response at five loads.

**What secure means here.** Secure for the outages studied, at that moment, at
that load — not for every hour. The screen shows every outage; the study covers
the most critical. A dispatch is never called N-1 secure beyond what its screen
shows.

### Over a window: a plan for the peak, and the cost of security

The studies above are for one moment. Two look across a window (*from 17:00 to
22:00*, or *the next 6 hours*; up to twelve moments, evenly spaced):

**A corrective plan** — *Plan the corrective action from 17:00 to 22:00.* One
optimisation over every moment, each meeting every limit, linked so that an
instruction is kept from one moment to the next unless changing it is worth it —
instead of a separate answer each hour that would call units in and out. Ask for
it steadier (*steadiness 4*), cheapest rather than least disturbance, or within a
ramp limit (*units ramp at most 30 % of capacity per hour*). The chart shows each
unit's change as a step (an instruction holds until it changes); the table, each
moment checked by a power flow; and the title, how many instructions the plan
issues against solving each moment on its own — every change of a unit's
active power counts. At the demo's evening peak: 17 instructions instead of 25
by default, 14 at steadiness 4.

**The cost of N-1 security** — *What does N-1 security cost this evening, hour by
hour?* At each moment, the same study a single question gets — the cheapest
corrective action and the N-1 secure option, preventive, corrective or both —
then what security costs above the cheapest, and in total over the window — each
moment weighted by half the time to its neighbours, so 16:00 to midnight counts
eight hours (an estimate between the moments studied). Compare
the secure option with the cheapest: where the schedule itself breaks a
normal-operation limit (the 90 % margin included), it is only a reference — the
answer says at which moments. Each option's N-1 figure is the full screen (as in
the chart: outages still with violations, of all those tested); the few outages
its optimisation studied, and how many it secured, are given apart — secure only
for those. Each moment is a full study, so the moments run in
parallel (on an 8-core laptop, nine moments take about 20 seconds), and the
status line shows how many are done and roughly how long is left.
From 16:00 to midnight on 6 January, hour by hour (nine moments), preventive
security costs about 6,400 € above the cheapest, corrective about 3,500 €. The
window's first and last moments are always studied; with the default of at most
six, a 15-minute series is sampled every two hours.

## Key performance indicators

| Study | Ask, for example |
| --- | --- |
| **KPIs at one moment** — the schedule against a corrective action | *Evaluate the KPIs at 20:00.* |
| **With a chosen option** — before and after it, held from its moment | *What are the KPIs at 21:00 with S3:cheapest?* |
| **Over a window** — measurements or the forecast | *How do the KPIs evolve this evening, hour by hour?* |

Every answer compares two states of the same moment, each checked by an AC
power flow: the schedule as it is, and the schedule with a corrective action —
the cheapest at that moment unless you name an option (at a later moment its
instructions are held, as in operation). When the schedule needs nothing, the
answer says so and shows the schedule's own indicators.

- **KPI-1 Target demand flexibility** — the units' spare upward capacity as a
  share of demand (load and losses).
- **KPI-2 Flexibility utilisation** — the fewest MW of redispatch that clear the
  violations (*flexibility required*), as a share of the MW the action moves
  (*flexibility activated*). A low value means the action moves more than it
  needs to, usually because that is cheaper.
- **KPI-3 Prevented violation ratio** — the share of the schedule's violations
  the action clears. When the schedule breaks no limit there is nothing to
  prevent, and the answer says that rather than 0 % or 100 %.
- **Security margins** — violations, the voltage margin to the nearest limit
  and the loading headroom (negative: outside the limit), and the single
  outages that leave a violation, of all those tested.
- **Deliverable flexibility** — how far the units can move up and down within
  every limit: each unit alone (the others held) and all together. Spare
  capacity is not the same thing: a unit may have spare MW the network cannot
  carry, and the chart shows both side by side.
- **Costs** — the action at the cost table's prices (a negative cost is a
  saving) and, if you ask for N-1, what N-1 security costs above the cheapest.

A value the study could not compute is shown as unknown, never as zero.

At the demo's evening peak (6 January, 20:00) the schedule breaks two limits.
The cheapest action clears them for 26.48 €/h and moves 34.4 MW, of which
9.0 MW are needed, so KPI-2 is 26 %. KPI-1 falls from 31.2 % to 20.7 %; the
units can still deliver 65.5 MW up and 31.3 MW down within every limit; the
N-1 screen improves from 41 to 29 outages with violations of 41. Preventive
N-1 security would cost 757 €/h more.

Over a window, every moment is a full study, run in parallel with its progress
shown (on an 8-core laptop, nine moments in about 4 seconds). The answer gives
each indicator's range and worst moment, the actions' cost over the window and
the MWh activated against the MWh required. From 16:00 to midnight on
6 January, the actions activate 38.2 MWh where 9.0 MWh are needed; upward
flexibility is lowest at 20:00 (65.5 MW).

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
