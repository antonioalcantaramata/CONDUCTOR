# Risk, uncertainty and capacity

The studies on this page look beyond one deterministic power flow: what happens
when loads and renewables differ from the data, which dispatch stays secure under
that spread, how far one unit or one new connection can go, and how often limits
were broken in the past.

## Probabilistic security assessment

A Monte Carlo study of AC power flows at **one** moment.

**What is uncertain.** Two multipliers: one shared by every static generator
($s$), one by every load ($\ell$) — so all generators move together, and all
loads together. Voltage-controlled units, the external grid, set-points and
topology are not uncertain. Each draw:

$$
s_i = \operatorname{clip}\!\big(\Phi^{-1}_{\mathcal N(1, \sigma_s)}(u_{i,1}),\, 0.01,\, 3\big),
\qquad
\ell_i = \operatorname{clip}\!\big(\Phi^{-1}_{\mathcal N(1, \sigma_\ell)}(u_{i,2}),\, 0.01,\, 3\big)
$$

from a two-dimensional **Latin hypercube** $u$ with a fixed seed (42), so a
study is reproducible. By default $\sigma_\ell = 0.05$ and $\sigma_s = 0$: only
load is uncertain unless asked otherwise. 200 samples by default (10 to 1,000).

!!! note "On the default demo"
    Every unit of the IEEE 30 demo is voltage-controlled, so the generation
    multiplier changes nothing there: only load uncertainty has an effect.
    The same holds for the renewable scenarios of the scan below.

**Generation can only fall.** Each static generator's available output is
$\operatorname{clip}(P^0_g\, s_i, 0, \overline P_g)$, but its injection is capped at
its measured set-point: the study asks what happens when the resource falls
short, not when it exceeds the schedule. Loads scale active and reactive power
together.

**The indices.** Each sample is solved by Newton–Raphson and judged with the
assessment's tolerance ($10^{-6}$). A sample whose power flow fails counts as
violating: it is most often the voltage collapse the study is about. Of $n$
samples, $n_c$ converge (`n_converged`, and `n_non_converged` the rest):

| Index | Definition |
| --- | --- |
| `p_any_violation` | share of all $n$ samples with at least one violation, the non-converged ones included |
| `expected_violations` | mean number of violations per converged sample |
| per-element probability | share of converged samples in which that element breaks a limit (an element counted once per sample, by name) |
| voltage percentiles | 5th, 50th and 95th per bus over the converged samples, by linear interpolation (as the robust study computes them) |
| histogram | number of converged samples with 0, 1, 2… violations |

These are **frequencies of limit breaches**. No load is shed and no energy is
integrated: energy-not-supplied and loss-of-load indices (ENS, LOLP) are not
computed. Each figure is a point estimate without a confidence interval: at
$p = 0.05$ and 200 samples the sampling error is about ±3 percentage points.

## Robust corrective dispatch

A corrective dispatch meant to stay within limits under the same two-multiplier
spread, with a risk target $\alpha$ (5 % by default). Two methods.

**Back-off (the default).** An iteration around the deterministic
[optimal power flow](optimal-power-flow.md), at most three times:

1. evaluate the current dispatch on the samples (the schedule, the first time);
2. stop if the share of samples with a violation is at or below the target, or
   if it improved by less than 0.005 since the last iteration;
3. tighten each bus's voltage band by the spread the samples show around the
   central power flow — at confidence $c = 1 - \alpha$:

$$
\begin{aligned}
\overline V_b &= \overline V - \max\big(0,\; Q_c(V_b) - V^{c}_b\big) \\
\underline V_b &= \underline V + \max\big(0,\; V^{c}_b - Q_{1-c}(V_b)\big)
\end{aligned}
$$

4. re-solve the OPF from the schedule with those per-bus bounds, the request's
   minimum power factor and exchange cap.

Only voltages are tightened; branch currents keep the OPF's current margin of
90 %, or the request's loading limit when that is stricter:
$\min(0.9,\ \overline\ell/100)$, with $\overline\ell$ the lower of its line and
transformer limits (the network's own unless given). The new set-points are
applied to the static generators by their position in the network's table. The
final dispatch is evaluated on the calibration samples and on an independent
validation set (seed 4242), and the latter gives the headline risk. In each
evaluation, as in the probabilistic assessment, a sample whose power flow fails
counts as violating.

The result names the buses whose limits were tightened most
(`largest_back_offs`, and a sentence, `back_off_summary`). When the risk target
is missed, it says why (`why_not_met`): the back-off tightens voltage limits
only, so a thermal risk is beyond its reach, and a voltage risk that remains
needs more than a back-off. Both are in the result's `message`.

**Scenario.** One AC OPF over $K$ sampled scenarios, with a set-point shared by
all and an AC state per scenario: in scenario $k$, each static generator injects
at most its set-point and at most the resource drawn; a unit with no resource
series — every voltage-controlled unit — injects its set-point in every
scenario. Every scenario keeps voltage and current limits (the current margin
above) and the exchange cap. The number of scenarios follows the
scenario approach's bound for a confidence $1 - \beta$ ($\beta = 10^{-3}$):

$$
K = \Big\lceil \frac{2}{\alpha}\Big( \ln \frac{1}{\beta} + d \Big) \Big\rceil, \qquad d = 2\,|\mathcal G|
$$

capped at 120 by default, and the answer says whether the bound was met. The
bound is nominal: it holds for convex problems, and an AC OPF is not convex.

!!! warning "What the scenario method guarantees in practice"
    With the defaults, any network with three units or more needs at least 517
    scenarios, so the cap of 120 binds and the bound is reported as not met
    (`guarantee_met` false, `K_capped` true).
    A static generator may inject less than its set-point in a scenario —
    curtailment — at almost no cost (injection is rewarded at only $10^{-6}$
    per MW), so for those units the shared set-point acts as an upper bound with
    nearly free downward recourse; the answer reports how often, and how much,
    the plan relies on it (`scenario_curtailment`).
    The voltages and exchange shown are those of the power flow of the dispatch
    at the central forecast, with no forecast error (`post_opf_state` says so);
    only if that power flow fails are they those of the first training
    scenario.

## Flexibility envelope of a unit

A feasibility map of **one** unit at one moment (a voltage-controlled unit is
studied as an injection of what it produces now, within its own reactive limits,
and the result says so): a
grid of (P, Q) set-points for that unit, the others held at the reference
state, each point checked by an AC power flow against every limit in the
network — by default the network's own, with the assessment's tolerance
($10^{-6}$).

- The reference state: measured, one the operator gives, or the last dispatch
  (`post_opf`), applied only if it was computed for the same moment, network,
  data source and load scaling, and not after an outage; otherwise the measured
  state is used and the result says why (`cached_dispatch_context_mismatch`, for
  example).
- P from 0 to the unit's maximum; Q over the unit's own reactive limits for a
  voltage-controlled unit, otherwise from $-q_c$ to $+q_c$ with
  $q_c = \max(P_0 \tan(\arccos 0.9),\ 0.3)$ Mvar at the unit's present output
  $P_0$; a 20 × 20 grid by default (5–25), plus the present point itself — up to
  676 power flows, eight at a time;
- each point: feasible, or the binding constraint (upper voltage, then thermal,
  then lower voltage), or not converged;
- the safe reactive range at the present active power
  (`safe_q_range_at_base_p`): in the grid column nearest $P_0$, the unbroken run
  of feasible points around the present Q — a feasible point beyond an
  infeasible one is not reached by moving Q alone — clipped to the unit's
  capability (its own limits, or $\pm q_c$, `q_cap_at_base_p`); none when no
  point of that column is feasible;
- the elements outside their limits at **every** point of the sweep
  (`violated_everywhere`): no output of this unit clears them. When there are
  any, the result says so and gives the safe range apart from them
  (`safe_q_range_apart_from_them`): how far the unit can move without putting
  anything else outside its limits. At the IEEE 30 demo's evening peak, the
  line from the external grid is overloaded whatever Gen_bus7 does;
- a `message` stating the safe range in words, which the answer leads with.

It is not an interface (TSO–DSO) envelope or a regional one: it answers how far
this unit can move, with everything else as it is.

## Hosting capacity

The largest additional injection $P$ of a **new** generator at one bus such that
the whole network stays within limits at one moment.

**Deterministic.** Bisection on $[0, P_{\max}]$: test the midpoint by power
flow (judged with the assessment's tolerance, $10^{-6}$), keep the half where
the frontier lies, stop when the bracket is narrower than 0.1 MW (or after 40
steps). The search ceiling defaults to twice the largest existing generator.
The new unit's reactive power follows a rule: unity power factor, a fixed power
factor (absorbing by default), or a constant reactive output (`reactive_proxy`):
it absorbs half of what the in-service static generators already at the bus
inject at that moment (their set-points, scaled), within the capability of
power factor 0.9 at the search ceiling. The answer gives the capacity, the
binding constraint at the lowest infeasible point tested, and, when the
bisection did not converge, no figure (it says why).

**Probabilistic.** The same bisection (at most 40 steps), but each candidate is
accepted when the share of samples with a violation is at most a risk threshold
(5 %): the new unit's output varies (availability $a_i \in [0, 1]$,
$\sigma = 0.1$), and optionally the loads. Samples that do not converge count as
violations. A bisection that did not converge is withdrawn, as in the
deterministic search: no figure.

The test is system-wide: an existing violation anywhere makes every point
infeasible and the capacity zero. Bisection assumes feasibility is monotone in
$P$; a frontier that closes and reopens (low voltage relieved, then high voltage)
is not detected.

## Historical risk

Statistics over **recorded** moments — deterministic replays, not probabilities
of future states. For each moment of a window (by default the whole dataset),
the full security assessment, and a severity for the target:

| Target | Severity $s_t$ | Violation | Near miss |
| --- | --- | --- | --- |
| the whole grid | number of violations | $s_t > 0$ | — |
| a bus | $\max(V_t - \overline V,\; \underline V - V_t)$ [p.u.] | $s_t > 0$ | within `near_miss_band` of the band (0.005 p.u. by default) |
| a line or transformer | loading minus limit [pp] | $s_t > 0$ | within `near_miss_band_pct` of the limit (5 percentage points by default) |

From these: the exceedance frequency (over the moments that converged, and an
upper version counting non-converged moments as exceedances), the near-miss
frequency, the worst moment, **episodes** — runs of consecutive moments of the
window in violation, with their peak and their length in moments
(`duration_steps`); a moment with no converged power flow ends an episode — a
duration curve (the number of violations per moment for the whole grid,
`violation_count`; the signed margin to the limit for one element,
`signed_limit_margin`), and optionally frequencies by hour of day, load level or
import level (`condition`).

## Scanning time and scenarios

- **The worst moment** (`find_worst_case_timestamp`): a power flow at every
  moment forward from the clock (or the first forecast moment), and the moment
  that maximises a metric — violations, import (summed over every external
  grid), highest or lowest voltage, highest loading of any line or transformer
  (with the transformers' own series beside it). The clock does not move.
- **Renewable scenarios** (`scan_scenarios`): the same scan repeated with the
  static generators scaled by fixed factors (0.7, 1.0 and 1.3 by default: low,
  baseline and high renewables — fixed multipliers, not percentiles of any
  distribution). Active and reactive output scale together, so each unit keeps
  its power factor. Per element, the result counts the moments it violates (once
  per moment and name).
- **A short look-ahead** (`scan_rsa_over_time`): the clock is advanced one step
  at a time and assessed at each — so it moves the operator's clock, and the
  starting moment itself is not assessed.

## Comparing a proposal with the base

The same study on the base network and on a proposal, at the same moment, data,
limits and stress: a security assessment, the N-1 screen, the optimal power flow,
or a scan over a window (thinned to at most 700 moments). The answer gives each
side's figures, the differences, the violations the proposal adds or removes,
and, for a scan, the moments and elements that violate on one side only. Up to
six proposals can be ranked at one moment. See
[What-if proposals](../guide/what-if-proposals.md).

## Assumptions and limits

- **Uncertainty is two scalars**, normal and independent, at one moment: no
  spatial or temporal correlation, no per-unit forecast error, no outages, no
  uncertainty on voltage-controlled units.
- **Monte Carlo figures are point estimates**, reproducible because seeded, with
  an unreported sampling error.
- **Non-converged samples** count as violating in the share of samples with a
  violation (probabilistic assessment, robust dispatch, probabilistic hosting
  capacity), which may overstate risk where a power flow fails for numerical
  reasons. Per-element probabilities, expected violations and percentiles are
  over the converged samples only.
- **The robust back-off** tightens voltages only, and evaluates only the static
  generators' set-points.
- **Generation uncertainty and renewable scenarios** act on static generators
  only: on a network of voltage-controlled units they change nothing.
- **The envelope and hosting capacity** judge the whole network: they are zero
  wherever the grid already breaks a limit.
- **Historical figures** are frequencies over what happened, not forecasts.

## In the code

| What | Where (`backend/main_backend.py` unless noted) |
| --- | --- |
| Probabilistic assessment | `POST /api/rsa/probabilistic` (`probabilistic_rsa`, `_run_probabilistic_sample`) |
| Robust dispatch | `POST /api/flexibility/robust` (`robust_flexibility`); scenario model `scenario_optimization_model_base` in `backend/flex_engine.py` |
| Flexibility envelope | `POST /api/flexibility/envelope` (`compute_flexibility_envelope`, `_run_envelope_point`) |
| Hosting capacity | `POST /api/flexibility/hosting_capacity` (`_solve_hosting_capacity_mode`, `_solve_hosting_capacity_mode_probabilistic`) |
| Historical risk | `POST /api/rsa/historical_risk` |
| Worst moment, scenario scan | `POST /api/rsa/worst_case`, `POST /api/rsa/scan_scenarios` |
| Look-ahead | `scan_rsa_over_time` in `llm_agent/agent/tools.py` |
| Proposal comparison | `POST /api/proposals/compare` |
