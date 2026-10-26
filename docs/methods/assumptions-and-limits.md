# Assumptions and limits

Every page of this section ends with the limits of its own study. This page
gathers the ones that cut across studies, says where results can mislead, and
how the models have been tested.

## What the models are

| Modelled | Not modelled |
| --- | --- |
| Balanced, single-phase-equivalent AC steady state at one moment | Dynamics, protection, unbalance, harmonics |
| Lines and two-winding transformers (π-models, taps at their position) | Three-winding transformers, impedance elements, wards, DC lines, storage in the optimisation; tap, phase-shifter and switched-shunt control |
| Constant-power loads; constant-impedance shunts | Voltage-dependent loads |
| Units: static generators held to a power factor, voltage-controlled machines and condensers with a reactive range | Capability curves, minimum active power (every unit may go to zero), ramping except in a plan |
| One voltage band for every bus, one loading limit for every branch | Per-element limits, angle limits, apparent-power ratings in the optimisation |
| Single outages of lines and transformers | Generator, bus and common-mode outages |
| Two scalar uncertainties (all loads, all static generators) | Spatial and temporal correlation, per-unit forecast errors |
| Linear prices for corrective actions | Production cost curves beyond the derived up and down prices, market rules |

## Where results can mislead

**Local optima.** Every optimisation is a non-convex AC OPF solved by an
interior-point method from one starting point, the moment's power flow. A
solution is a local optimum. Two checks from a convex relaxation of the same
problem limit the consequences: the cheapest option's cost has a lower bound,
and when no dispatch is found the relaxation either proves that none exists,
gives IPOPT a second start, or says that nothing is proven
([The optimal power flow](optimal-power-flow.md#when-there-is-no-solution)).
They do not cover the N-1 secure dispatch, the plan over a window or the other
options, which remain local.

**Two sets of limits.** Assessments judge against the network's limits (100 % of
each rating by default); optimisations keep a current margin (90 %). An option at
90 % is "within limits" for both, and an assessment can be clean while the
optimisation still asks for action: that is the margin at work, not an
inconsistency. N-1 security uses emergency ratings after an outage (100 % by
default); the post-contingency optimisation of a single outage keeps the normal
margin; the N-1 screen judges at the network's limits unless given others.

**One moment.** A secure moment is not a secure day. Studies over time
interpolate between the moments they study; nothing between them is simulated.

**Synthetic data.** The bundled week is generated, with every load following one
shape and every unit one factor, and stress events placed to produce violations.
It exercises the studies; it is not evidence about a real grid. The synthetic
prices are a rule (production cost × 1.2 up, × 0.5 down, import at 60 €/MWh…),
not market data.

**What a figure is.**

- An N-1 screen counts limit violations and, apart, the outages that cut load
  off; either makes the grid not N-1 secure. The KPIs' "N-1 outages with
  violations" counts the first only.
- The cost of N-1 security is that of the least-disturbance secure dispatch, an
  upper estimate of the least cost of security.
- Probabilistic figures are frequencies of limit breaches over seeded samples,
  without confidence intervals, a sample whose power flow fails counting as a
  breach; no energy-not-supplied or loss-of-load index is computed.
- KPI-2 compares the fewest megawatts needed with what the action moves; a low
  value often reflects cheaper units moving more, not a poor action.

**The language model.** It chooses studies and writes about results; the checks
compare what it wrote with the results, and describe rather than certify (see
[The agent and its checks](the-agent.md#what-the-checks-cannot-see)). A traced
figure can still answer the wrong question.

## How it has been tested

**The test suite** — about 2,100 tests, run on Linux, macOS and Windows (the
solver-backed ones where IPOPT is installed). Among them:

- **Physics.** The optimisation's admittances, evaluated at a converged power
  flow, give back every bus's injection to within $10^{-6}$ MVA on the three
  bundled networks. The optimum of the OPF, replayed in an ordinary pandapower
  power flow, reproduces its voltages to $10^{-4}$ p.u. and the grid exchange to
  0.05 MW and Mvar. The current margin is a fraction of rated current; the
  power-factor cone and reactive limits hold.
- **The demo.** On the IEEE 30 demo: the evening peak congests the line from the
  external grid; options differ and nothing dropped is left unexplained; a unit
  the operator rules out stays at its schedule; near-optimal alternatives stay
  within their margin and change other units; options after an outage, their N-1
  and N-1-1 screens; the power-flow check catches a dispatch that does not hold.
- **Studies.** Corrective options and their selection, the cost model, the N-1
  secure options and their plans, holding and planning over time, the cost of
  security over a window (in parallel and one moment at a time, the same answer),
  the KPIs (each formula's edge cases, the flexibility required against the
  least-disturbance option's movement, the cost against the options study's).
- **Refusals and honesty.** A study that finds no solution says so; values the
  study could not compute are reported as unknown; impossible parameters are
  refused with the reason.

**Every answer**, at run time: each option re-solved by an AC power flow; each
written figure traced to the results (see [The agent and its checks](the-agent.md)).

**Evaluation** — fixed tasks run with real models and graded by rules from the
session logs (see [Using CONDUCTOR in research](../research/using-conductor-in-research.md)).

## Reproducibility

- The synthetic data are seeded (42 for measurements, 137 for the forecast), as
  are the random samples (42 for Monte Carlo and robust calibration, 4242 for
  validation, 7 for near-optimal random directions): the same network and question
  give the same figures.
- Every turn is logged with its resolved arguments and full results, so an answer
  can be re-graded or re-run.
- Solver settings are the libraries' defaults; the pandapower version is not
  pinned beyond 3.0, and IPOPT comes with the environment. Record both with
  published results.
