# Key performance indicators

The indicators describe one moment of the grid **twice**: as scheduled, and
with a corrective action applied. Every figure comes from a power flow or an
optimisation on the network at that moment; nothing is estimated by the
language model.

## The two states

At a moment $t$ the study builds two operating points and solves an AC power
flow on each (see [Power flow and security assessment](power-flow-and-security.md)):

- **the schedule** $s^0$ — the measured (or forecast) injections as they are;
- **with the action** $s^a$ — the schedule with a corrective dispatch applied.

The action is, by default, the **cheapest** corrective option the options
study finds at $t$ (see [Corrective options](corrective-options.md)). The
operator may name another option instead, by its reference (`S3:cheapest`).
An option computed at an earlier moment is applied **held**: only the
instructions it gave (the set-points it changed) are kept, and every other
unit follows its own schedule at $t$, as it would in operation.

When the schedule already meets every limit of the optimisation, no action is
needed and both states are the same; the answer says so
(`after_is = "schedule"`). When no action is found at all, the indicators of
the action are unknown (`after_is = "none"`) and are reported as such, never as
zero.

Comparing two states solved by the same power flow is what makes the
indicators readable as *the effect of the action*: the schedule's figures are
the reference, the action's figures the result.

## The indicators

The notation follows [the overview](index.md#notation). $\mathcal G^{f}$ are
the units the optimisation may move (every unit except those held at their
schedule by the operator and those whose bounds fix their output — a
synchronous condenser, a unit marked not dispatchable), $P_g^0$ a unit's
scheduled output and $P_g^a$ its output with the action, both read from the
power flow of each state.

### Security

**Violations (N-0).** The number of limits broken with every branch in
service: bus voltages outside $[\underline V, \overline V]$, lines and
transformers above their loading limit. One violation per element and limit.

**Voltage margin.** The distance of the nearest bus voltage to its limit, over
the buses in service:

$$
m_V = \min\Big( \min_{b} V_b - \underline V,\;\; \overline V - \max_{b} V_b \Big) \quad [\text{p.u.}]
$$

Negative when a voltage is outside its band.

**Loading headroom.** Percentage points between the most loaded branch and its
limit, lines and transformers each against their own:

$$
h = \min\Big( \overline\ell^{\,\text{line}} - \max_{l} \ell_l,\;\; \overline\ell^{\,\text{trafo}} - \max_{k} \ell_k \Big) \quad [\text{pp}]
$$

where $\ell$ is the loading in percent of the rating. Negative when a branch
is overloaded.

**N-1 outages with violations.** The single outages — every line and
transformer in service, one at a time — after which some limit breaks, out of
all those tested (see [Contingency analysis](contingency-analysis.md)). For the
action, the screen runs with its dispatch applied. An outage that only cuts
load off breaks no limit and is not counted here; the N-1 screen itself
reports it.

**KPI-3 Prevented violation ratio.** The share of the schedule's violations the
action clears, counted by the power flows of the two states:

$$
\text{KPI}_3 = 100 \cdot \max\!\Big(0,\; \frac{n^0 - n^a}{n^0}\Big) \;\;[\%], \qquad n^0 > 0
$$

with $n^0, n^a$ the N-0 violations of the schedule and of the action. When the
schedule breaks no limit ($n^0 = 0$) there is nothing to prevent and KPI-3 has
no value.

### Flexibility

**KPI-1 Target demand flexibility.** The units' spare upward capacity as a
share of the demand they and the external grid supply (load and losses):

$$
\text{KPI}_1 = 100 \cdot \frac{\sum_{g} \max\big(\overline P_g - P_g,\, 0\big)}{\sum_{g} P_g + \sum_{e} P_e^{\text{ext}}} \;\;[\%]
$$

summed over the units with a known maximum $\overline P_g$ (a renewable's
maximum is what its resource gives at that moment). Computed for each state from
its power flow. The numerator is reported on its own as **spare upward
capacity** (MW).

Spare capacity is not the same as flexibility the network can use: a unit may
have spare megawatts that no line can carry. That is what the next indicator
measures.

**Deliverable flexibility.** How far the units can move from a dispatch
$\hat P$ and still meet **every limit of the optimisation** — the set
$\mathcal F$ of dispatches that satisfy the AC power-flow equations, voltage,
current and unit limits (see [The optimal power flow](optimal-power-flow.md)).
Reactive outputs and voltage set-points are free to follow. Together:

$$
\Delta^{\uparrow} = \max_{x \in \mathcal F} \sum_{g \in \mathcal G^{f}} P_g \;-\; \sum_{g \in \mathcal G^{f}} \hat P_g,
\qquad
\Delta^{\downarrow} = \sum_{g \in \mathcal G^{f}} \hat P_g \;-\; \min_{x \in \mathcal F} \sum_{g \in \mathcal G^{f}} P_g
$$

and each unit alone, the others held at $\hat P$:

$$
\Delta_g^{\uparrow} = \max\big\{ P_g \;:\; x \in \mathcal F,\; P_h = \hat P_h \;\forall h \in \mathcal G^{f}\setminus\{g\} \big\} - \hat P_g
$$

and $\Delta_g^{\downarrow}$ likewise with the minimum. $\hat P$ is the dispatch
of the state (with the action, or the schedule when no action is needed), read
from its power flow and clipped to each unit's bounds (a power flow may land a
hair beyond them). A range the solver finds no dispatch for is reported as
unknown — the state itself is then outside the limits.

These are $2\,|\mathcal G^{f}| + 2$ small nonlinear programmes on the
options study's own model; on the 30-bus demo they take about a second.

Per unit, $\Delta_g^{\uparrow} \le \overline P_g - \hat P_g$: a unit cannot
deliver more than it has. Where the two differ, the network is the limit.

**Flexibility required.** The fewest megawatts of active-power redispatch,
summed over the units, that bring the schedule within every limit — an L1
minimum, solved with split variables $u_g, d_g \ge 0$:

$$
\begin{aligned}
R = \min_{x \in \mathcal F}\; & \sum_{g \in \mathcal G^{f}} (u_g + d_g) + 10^{-4} \sum_{g \in \mathcal G^{f}} \big(P_g - P_g^0\big)^2 \\
\text{s.t.}\; & P_g - P_g^0 = u_g - d_g, \quad u_g, d_g \ge 0
\end{aligned}
$$

reported as $\sum_g |P_g - P_g^0|$ at the optimum. The small quadratic term
picks, among dispatches moving equally few megawatts, the one spread least. A
megawatt up and a megawatt down count alike, which is why $R$ is not the
least-disturbance option's movement: that option minimises squares (see
[Corrective options](corrective-options.md#least-disturbance)) and may move more
megawatts in total. $R = 0$ when no active power needs to move — when the
violations clear with reactive power or voltage set-points alone. When
redispatching the units cannot clear them (the options fall back to load
reduction), $R$ is unknown.

**Flexibility activated.** What the action moves, read from the two power flows:

$$
A = \sum_{g} \big|P_g^a - P_g^0\big|\,\mathbb 1\big[\,|P_g^a - P_g^0| \ge 0.01\ \text{MW}\big] \;+\; \max\big(L^0 - L^a,\, 0\big)
$$

where $L$ is the load served: a load the action reduces (demand response, load
relief) counts as activated flexibility too.

**KPI-2 Flexibility utilisation.** The flexibility required as a share of the
flexibility activated:

$$
\text{KPI}_2 = 100 \cdot \min\Big(1,\; \frac{R}{A}\Big) \;\;[\%]
$$

| Case | KPI-2 | Note given with it |
| --- | --- | --- |
| $R > 0,\ A > 0$ | $100\,R/A$ | — |
| $R = 0,\ A = 0$ | 100 % | nothing needed, nothing moved |
| $R = 0,\ A > 0$, schedule has violations | 0 % | no MW needs to move: reactive power or voltage set-points clear the violations |
| $R = 0,\ A > 0$, no violations | 0 % | the action moves units though nothing needs to move |
| $R > 0,\ A = 0$ | 0 % | the action moves nothing, yet redispatch is needed |
| $R$ unknown | — | redispatching the units alone cannot clear the violations |

A low KPI-2 is not an error: the cheapest action often moves more megawatts than
strictly needed because moving cheap units is cheaper than moving few. KPI-2
shows how much more.

### Cost

**Cost of the action** at the cost table's prices (€/h), relative to the
schedule — the same figure the options study gives the option (see
[Corrective options](corrective-options.md#the-cost-of-an-action)). A negative
cost is a saving: the changed dispatch is cheaper than the schedule. An option
held from an earlier moment is priced from the change its power flow shows
against this moment's schedule.

**Cost of N-1 security** (when asked, `n1`): the N-1 secure option's cost above
the cheapest action's at the same moment, the lower of the preventive and
corrective ones when both are asked (see [N-1 security](n1-security.md)).

## Over a window

The same study runs at each moment of a window — measurements or the forecast,
from a start to an end, at most twelve moments, evenly spaced with the first
and last always kept (see [Studies over time](over-time.md#choosing-the-moments)).
Each moment compares its own schedule with its own cheapest action, or with the
named option held. The moments run in parallel worker processes; a moment that
cannot be studied is reported with its reason, never as zeros.

Over the window, for each indicator: its range with the action (and as
scheduled), and the moment where it is worst and best — worst meaning lowest
for an indicator where higher is better. And the window's totals, each moment
weighted by half the time to its neighbours (the trapezoid rule):

$$
\begin{aligned}
w_k &= \tfrac12\,(t_k - t_{k-1}) + \tfrac12\,(t_{k+1} - t_k) \\
C &= \sum_k w_k\, c_k \;[\text{€}], \qquad
E^{a} = \sum_k w_k\, A_k, \qquad
E^{r} = \sum_k w_k\, R_k \;[\text{MWh}]
\end{aligned}
$$

with the first and last moments taking only their inner half, so the weights
sum to the window's length.

## An example

On the default demo (PGLib IEEE 30-bus), at 20:00 on 6 January the schedule
breaks two limits: Bus_29 at 0.943 p.u. and line L0 at 114.7 % of its rating.

| Indicator | Schedule | With the cheapest action |
| --- | --- | --- |
| Violations (N-0) | 2 | 0 |
| Voltage margin | −0.007 p.u. | 0.000 p.u. |
| Loading headroom | −14.7 pp | +10.0 pp |
| N-1 outages with violations (of 41) | 41 | 29 |
| KPI-1 Target demand flexibility | 31.2 % | 20.7 % |
| Spare upward capacity | 99.9 MW | 65.5 MW |
| Deliverable up / down | — | 65.5 / 31.3 MW |
| Flexibility required / activated | — | 9.0 / 34.4 MW |
| KPI-2 / KPI-3 | — | 26.3 % / 100 % |
| Cost of the action | 0 | 26.48 €/h |

The action holds the lowest voltage exactly at the optimisation's bound
(0.95 p.u.) and the most loaded line at 90 % (the optimisation's current margin),
which is why the margins land at 0 p.u. and 10 pp. It moves 34.4 MW where 9.0 MW
would do — moving cheap units is cheaper. Every unit can still deliver its
spare capacity upwards except as the network allows: Gen_bus10 has 1.2 MW
spare and can deliver all of it, Gen_bus7 is at its maximum.

Over the evening (16:00 to midnight, hourly), the actions activate 38.2 MWh
where 9.0 MWh are needed, upward flexibility is lowest at 20:00 (65.5 MW), and at
18:00 the cheapest action leaves more single outages with violations than the
schedule does (29 against 19): cheapest at N-0 is not N-1 friendly.

## Relation to the project notebook's definitions

KPI-1 to KPI-3 keep the names and intent of the project's original definitions;
three flaws made them uninformative and were fixed:

- **KPI-1** was read off a second optimisation solved under the external-grid
  cap, which moved it even when the cap limited nothing (a non-binding 999 MW cap
  changed it from 67 % to 54 % at a quiet hour). It is now computed on each
  state's own dispatch.
- **KPI-2** compared the megawatts moved with the *net* change in total
  generation, $|\sum_g P_g^a - \sum_g P_g^0|$. Redispatch moves units up and
  down, so the net change is about the change in losses and KPI-2 read 0 % at
  every hour. It now compares with the flexibility required.
- **KPI-3** was 100 % whenever the solver converged, even with nothing to
  prevent, and 0 % when it did not. It now counts the violations that existed
  and has no value when there were none.

## Assumptions and limits

- Two sets of limits are at play, on purpose. The **power flow** of each state
  is judged against the network's limits (by default its own voltage band and
  100 % loading). The **optimisation** — the action, the flexibility required and
  deliverable — keeps its own: the voltage band it is given and a current margin
  (90 % of the rating by default). An action therefore sits inside the power
  flow's limits with room to spare, and the deliverable ranges are what the
  units can do *within the optimisation's* limits.
- Deliverable flexibility per unit holds the others where they are; the units
  together may deliver more or less than the sum of the units alone.
- The optimisation is a non-convex AC OPF solved to a local optimum by IPOPT
  (see [The optimal power flow](optimal-power-flow.md#assumptions-and-limits)):
  the flexibility required is the fewest megawatts the solver finds, not a
  certified global minimum.
- Demand response is not part of deliverable flexibility; it enters only as
  activated flexibility when an action reduces load.
- Each indicator describes one moment. Over a window, totals between moments
  are an estimate (the trapezoid rule), not a simulation of what happens between
  them.

## In the code

| What | Where |
| --- | --- |
| The formulas, the indicators' order and meaning, the window summary | `backend/kpis.py` |
| One state by power flow (violations, margins, N-1 screen, dispatch) | `_kpi_state` in `backend/main_backend.py` |
| One moment: the options study, the two states, the indicators | `_kpi_moment` in `backend/main_backend.py` |
| Flexibility required (L1 minimum), deliverable ranges | `least_moved_mw`, `deliverable` in `backend/flex_options.py` |
| Endpoints | `POST /api/kpi/evaluate`, `POST /api/kpi/over_time` |
| Tools | `evaluate_kpis`, `kpis_over_time` |
