# N-1 security

An N-1 secure option is a dispatch that keeps the grid within limits after any
of the critical single outages — **preventive**: nothing to do when one happens;
or **corrective**: a quick action after it, planned in advance. Both come from a
security-constrained optimal power flow (SCOPF) that stacks the intact network
and each studied outage in one optimisation.

## The model

One optimisation holds:

- the **base state** $k = 0$: the intact network, with the
  [optimal power flow](optimal-power-flow.md)'s hard limits — the voltage band
  and the normal current margin $\mu$ (90 % of each rating by default);
- one **post-outage state** $k \in \mathcal K$ per studied outage: the same
  OPF built on the network without that branch.

There is no separate admittance matrix per outage: each state uses the
network's admittance database, and the outaged branch simply drops out of its
power balance because it is out of service. The exception is a branch with a
parallel circuit between the same buses: after it trips, that state's
admittances are rebuilt from the post-outage network. A bus the outage leaves
without supply (a unit's bus, with no load: see below) is taken out of service
in that state, and the unit's output is lost.

Every state keeps the full AC power balance, the unit limits and the reactive
limits **hard**. After an outage, the voltage and current limits become **soft**
— they may be exceeded, at a price — so the optimisation can say *how close to
secure* a dispatch gets when it cannot reach secure, and which outages remain.

The external grid is the slack of every state, within the base state's active
and reactive bounds (the exchange cap included). A unit the operator holds keeps
its active power in every state; after an outage its reactive output is linked
to the base state like any other unit's (below), not frozen at the post-outage
power flow.

### Limits after an outage

Branch currents are judged against an **emergency rating** $\epsilon$ (by
default the network's loading limit, 100 % on the bundled networks, one value
for lines and transformers; the operator may give 50–150 %) instead of the
normal margin, with one slack $s^k_{ij} \ge 0$ per branch end:

$$
\frac{|I^k_{ij}|^2}{\bar I_{ij}^{\,2}} \le \epsilon^2 + s^k_{ij}
$$

Bus voltages, except at buses whose voltage a unit holds (a set-point is a
control, so its band stays hard), with slacks $s^{k,\text{lo}}_b, s^{k,\text{hi}}_b \ge 0$:

$$
\underline V - s^{k,\text{lo}}_b \;\le\; V^k_b \;\le\; \overline V + s^{k,\text{hi}}_b,
\qquad 0.5 \le V^k_b \le 1.5
$$

The penalty of a state:

$$
\Phi_k = \sum_{(i,j)} s^k_{ij} + w_V \sum_b \big(s^{k,\text{lo}}_b + s^{k,\text{hi}}_b\big), \qquad w_V = 20
$$

(a 0.01 p.u. voltage excess weighs about as much as a 10 % loading excess).

### Preventive: one dispatch for every state

Each unit's active power is the same before and after every outage; a unit that
holds a voltage keeps its set-point (its reactive output follows the grid
within its limits); every other unit keeps its reactive output:

$$
\begin{aligned}
P^k_g &= P^0_g && \forall g \\
V^k_{b(g)} &= V^0_{b(g)} && \forall g \in \mathcal G^{V} \\
Q^k_g &= Q^0_g && \forall g \notin \mathcal G^{V}
\end{aligned}
$$

The external grid is free in each state: it is the slack.

### Corrective: a bounded action after each outage

After outage $k$ each unit may move, within a **reach** $r_g$ of its
capacity (20 % of $\overline P_g$ by default, the operator may change it);
set-points may move within the voltage band; and a unit that does not hold a
voltage may be told a new reactive output:

$$
\begin{aligned}
P^k_g - P^0_g &= u^k_g - d^k_g, & 0 \le u^k_g,\, d^k_g &\le r_g \\
V^k_b - V^0_b &= \nu^{k,+}_b - \nu^{k,-}_b, & \nu^{k,\pm}_b &\ge 0 \\
Q^k_g - Q^0_g &= q^{k,+}_g - q^{k,-}_g, & q^{k,\pm}_g &\ge 0, \quad g \notin \mathcal G^{V}
\end{aligned}
$$

When redispatch after the outage is not enough, **load may be reduced** after it
(on by default): demand response up to each load's share in the cost table, and
load relief beyond it. It enters the penalty with small uniform weights,
$w_{dr} = 2\cdot 10^{-3}$ and $w_{rel} = 2\cdot 10^{-2}$ per MW, so it is used
only when it removes a violation:

$$
\Lambda = \sum_{k} \sum_{d} \big( w_{dr}\, P^{dr,k}_d + w_{rel}\, P^{\text{shed},k}_d \big)
$$

The base state never reduces load: reduction only ever happens after an outage.

### The objective: as secure as possible, then the least change

Two stages. First, the least violation, with a tiny weight on preference:

$$
\min \;\; \mathcal V + 10^{-6}\, \mathcal P, \qquad \mathcal V = \sum_{k \in \mathcal K} \Phi_k + \Lambda
$$

Then, keeping that security ($\mathcal V \le \mathcal V^\star (1 + 10^{-3}) + 10^{-6}$),
the preference alone:

$$
\begin{aligned}
\mathcal P ={}& \lambda_p \sum_g \big(P^0_g - P^{0,\text{sch}}_g\big)^2 + \lambda_q \sum_g \big(Q^0_g - Q^{0,\text{sch}}_g\big)^2 \\
&+ \sum_{k} \sum_g \Big( c_1 (u^k_g + d^k_g) + c_2 (u^k_g - d^k_g)^2 \Big) \\
&+ \sum_k \sum_b \Big( c_3 (\nu^{k,+}_b + \nu^{k,-}_b) + c_4 (\nu^{k,+}_b - \nu^{k,-}_b)^2 \Big) \\
&+ \sum_k \sum_{g \notin \mathcal G^{V}} \Big( c_5 (q^{k,+}_g + q^{k,-}_g) + c_6 (q^{k,+}_g - q^{k,-}_g)^2 \Big)
\end{aligned}
$$

with $\lambda_p = 0.01$, $\lambda_q = 0.001$ (the OPF's own weights),
$c_1 = 10^{-2}$ per MW, $c_2 = 10^{-3}$ per MW², $c_3 = 2$ per p.u., $c_4 = 10$
per p.u.², $c_5 = 10^{-3}$ per Mvar, $c_6 = 10^{-4}$ per Mvar². The move terms
exist only in corrective mode.

!!! note "Least disturbance, not least cost"
    The secure dispatch is the one that changes the schedule **least** (and,
    in corrective mode, plans the smallest actions), not the cheapest secure
    one. Its cost is priced afterwards, at the cost table's prices, for the
    base dispatch only. "What N-1 security costs above the cheapest" is
    therefore the cost of the least-disturbance secure dispatch minus the
    cheapest unsecured action — an upper estimate of the least cost of
    security, not that minimum.

Both stages are solved by IPOPT. If the second fails, the first stands. The
problem is a non-convex AC OPF: the solution is a local optimum.

## Which outages are studied

Stacking every outage would be large and mostly useless, so the study picks the
**critical** ones:

1. **Candidates.** The N-1 screen of the least-disturbance option (see
   [Contingency analysis](contingency-analysis.md#the-n-1-screen)), at the
   emergency rating. Or exactly the outages the operator names.
2. **Severity.** Per outage, the sum over its violations of
   $\max(e, 0.01)$, with $e$ the excess beyond the limit: the loading above the
   emergency rating in percent, or for a voltage 100 times its distance beyond
   the voltage band (in p.u.). Most severe first.
3. **Outages that cut load off are skipped,** with the reason: no dispatch
   keeps an island's load supplied. An outage that only disconnects a unit, and
   no load, is studied: its buses are out of service in that state, and the
   other units must cover what it produced. (On the 30-bus demo, L12 and L15
   each disconnect one unit's bus; L33 cuts off Bus_25's load.)
4. **At most eight** outages are studied at once; the rest are reported as
   skipped — the screen still checks them.
5. **A second round.** The secure dispatch is screened again; critical outages
   it reveals that were not studied are added (up to eight in all; those that
   cut load off are listed as skipped) and the problem is solved once more. If
   that second solve fails, the first round's dispatch stands.

## What is reported

For each mode, an option like any other (with its reference, cost and
power-flow check), plus:

| Field | Meaning |
| --- | --- |
| `screen` | the full N-1 screen of the option's dispatch at the emergency rating (corrective: before any action) |
| `studied` | the outages in the optimisation; `rounds` how many solves |
| `secured` | studied outages with no residual violation in the optimisation |
| `not_secured` | per outage, what remains (up to four residuals, e.g. "L30 at 104.2 %"), and **switching that may help** |
| `corrective_plan` | corrective mode: per outage, the actions (units moving at least 0.1 MW, set-points at least 0.002 p.u., a new reactive output of at least 0.05 Mvar for a unit that does not hold a voltage, `dq_mvar`; loads reduced), each checked by an AC power flow after the outage |
| `skipped` | outages not studied, each with its reason (it cuts load off; beyond the eight; its model could not be built) |

**Switching that may help.** For an outage that is not secured, each further
in-service branch is opened in turn on the post-outage network, by power flow;
openings that leave a bus without supply are dropped. An opening is shown when it
lowers the total excess (loading above the emergency rating plus 100 times the
voltage beyond the band) by at least 10 %, or the number of violations; at most
three, those that clear everything first. These are candidates for an
engineer's judgement, not part of the optimisation.

**A corrective plan, checked.** For each studied outage, the base dispatch
alone is first checked by power flow after the outage: when it already holds,
the plan says no action is needed. Otherwise the plan lists the post-outage
dispatch's moves and whether an AC power flow confirms it within limits.

## Assumptions and limits

- **Two judges.** `secured` comes from the optimisation's residuals; the plan's
  check is a power flow with its own tolerances. An outage can count as secured
  while its plan's power flow is not confirmed, or the reverse. Both are shown.
- **Emergency versus normal limits.** After an outage the study allows the
  emergency rating (100 % by default), not the normal margin (90 %). The
  post-contingency optimisation of a single outage keeps the normal margin and
  has no reach limit, so the same outage can need action there and be secured
  here. The N-1 screen judges at the network's limit unless given another.
- **One emergency rating** for lines and transformers alike.
- **Load cut off** by an outage is not studied: no dispatch can supply it, and
  the outage is listed as skipped.
- Size: up to eight post-outage OPFs built and solved once each, two SCOPF
  stages, up to four N-1 screens and a switching search per unsecured outage — a
  few seconds per moment on the 30-bus demo.

## In the code

| What | Where |
| --- | --- |
| Softened limits, linking, load reduction, the two-stage objective, residuals | `backend/scopf.py` (`soften`, `build`, `secure_first`, `residuals`) |
| Outage selection, rounds, reporting, corrective plans | `_n1_study` in `backend/main_backend.py` |
| Severity ranking, switching remedies, power-flow check | `_critical_outages`, `_switching_remedies`, `_power_flow_check` |
| Request | `optimize_flexibility` with `n1`, `n1_outages`, `n1_emergency_loading_pct`, `n1_reach_pct`, `n1_load_reduction` |
