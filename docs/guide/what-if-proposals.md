# What-if proposals

A **proposal** is the loaded network plus a list of changes. The loaded network
(the *base*) is never modified: you can build several proposals, study each one
with any study, compare them with the base or with each other, and discard them.

## Making one

Describe the change: *"What if we connect a 20 MW data centre at Bus 9?"*.
CONDUCTOR builds the proposal, names it (e.g. `dc_bus9`), and usually studies it
straight away against the base.

- **Missing decisions are asked, not guessed.** When the request leaves a real
  choice open (what kind of unit, whether a generator holds voltage or runs at a
  fixed power factor, a capacitor or a reactor, which of two buses), CONDUCTOR
  asks you with numbered options and builds nothing until you answer.
- **Values it fills in are stated.** "Assumed: set-point 1.0 p.u., reactive
  range ±18.6 Mvar" — any of them can be changed.
- **Notes on modelling.** When a change copies a component that is not physical
  (common in converted per-unit cases) or a value looks unusual, the answer
  passes the note on. In a converted per-unit case every bus is at 1 kV, which
  stands for its real voltage. A new transformer to *a 20 kV bus* is built all
  the same, with its rated voltages taken from the buses it joins and its
  impedance unchanged, and the answer says so.

Your proposals are listed in the sidebar under **Proposals**: each change with
a ✕ to remove it, and **Discard proposal**. Proposals are kept for the network
across restarts.

## Every kind of change

| Change | Ask, for example |
| --- | --- |
| **New load** — flat, following a bus's measured shape, or on working hours | *Add a 20 MW flat data centre at Bus 9.* · *A 5 MW factory at Bus 4, 08:00–18:00.* |
| **New generator** — dispatchable, solar or wind (following the measured production shape); on voltage control, at a fixed power factor or at a fixed reactive output | *Add 30 MW of solar at Bus 13.* · *A 50 MW gas unit at Bus 2 on voltage control at 1.02 p.u.* |
| **Capacitor bank or reactor** | *Add a 10 Mvar capacitor at Bus 13.* |
| **Reinforce a line** — more capacity, e.g. a second identical circuit | *Double the capacity of the Bus_1 – Bus_4 line.* |
| **Larger transformer** | *Upgrade transformer T0 to 100 MVA.* |
| **New bus joined by a line** — a new site | *Connect a new site "Harbour" to Bus 9 with a 12 km line like the Bus_8 – Bus_9 line.* |
| **New bus at another voltage behind a new transformer** — optionally at the end of a new line | *A new 20 kV bus behind a 25 MVA transformer at Bus 5.* |
| **New line between existing buses** — meshing, a second circuit | *Add a line between Bus 0 and Bus 3, 10 km, like line 0.* |
| **New transformer between existing buses** | *Add a 60 MVA transformer between Bus 3 and Bus 6.* |
| **Switch an element in or out** — lines, transformers, loads, units, shunts, storage | *Open the Bus_2 – Bus_3 line.* · *Switch off Gen_bus1.* |
| **Change any setting** of an existing element | *Set the external grid voltage to 1.05 p.u.* · *Set transformer T1 to tap +2.* · *Lower the line limit of line 3 to 80 %.* · *Hold the load at Bus 3 at 50 MW.* |
| **Move an element** — a load, unit, shunt, storage, the external grid, or one end of a line | *Move the load at Bus 2 to Bus 4.* · *Reconnect the far end of line 5 to Bus 2.* |
| **Scale existing loads or units** — keeping their measured profile | *Grow every load by 20 %.* · *Halve the loads at Bus 5.* |
| **Combine proposals** — "A + B" | *Make a proposal with both dc_bus9 and grid105.* |

A few rules worth knowing:

- An **added load or unit** follows its own profile, not the measurements of
  the bus it joins. A **flat** load is a fixed demand: load scaling and the
  uncertainty of probabilistic studies leave it unchanged.
- A **moved** load or unit keeps its own measurements; only its connection
  changes.
- A **setting held** on a measured load or unit (e.g. "hold the load at 50 MW")
  replaces the measured value at every time step.
- **Scaling** multiplies the measured values at every time step; two scalings
  multiply together.
- A **combined** proposal copies the other proposals' changes. Changing those
  proposals later does not change it.

## Studying a proposal

Any study can run on a proposal: *"Run the N-1 screen on dc_bus9"*. Answers about
a proposal always name it, and results list what each added element injects at
that moment.

To compare, ask for a comparison: CONDUCTOR runs the same study **at the same
operating point** on both networks and reports the differences.

| Comparison | Ask, for example |
| --- | --- |
| **One moment** — new and resolved violations, largest voltage and loading changes | *Compare dc_bus9 with today's grid at 18:00.* |
| **N-1** — outages that cause violations on one network and not the other | *Does dc_bus9 make the N-1 screen worse?* |
| **Corrective dispatch** — whether redispatch can restore security on each, and what it takes | *Compare the corrective dispatch needed with and without dc_bus9.* |
| **Over a window** — how often and when each violates | *Compare dc_bus9 with the base over the whole week.* |
| **Against another proposal** | *Is option B better than option A at the worst moment?* |
| **Several at once** — up to six against the same reference, ranked in a table | *Rank options A, B and C by violations and lowest voltage at 18:00.* |

What a change does depends on the hour, so unless you name one moment,
CONDUCTOR compares over the period you name (*compare with today*: the day; *at
the worst hour of the week*: the week) or over the current day. It finds each
network's worst moment, which the change may move. A comparison at one moment
is a verdict about that moment. For a decision, ask for an N-1 screen at the
proposal's worst moment too; the advisor will remind you when it is missing.

## Proposals in the System tab

The [System tab](system-tab.md) can draw a proposal instead of the base, or
the **difference from the base**: what the proposal adds and which violations
it creates or resolves.
