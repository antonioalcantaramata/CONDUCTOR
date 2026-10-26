# The optimal power flow

Every corrective action — the options, the N-1 secure dispatches, the plans, the
flexibility indicators — is the solution of one AC optimal power flow (OPF), or a
variant of it with another objective or extra constraints. This page gives the
model as implemented.

## The model

A balanced, single-phase-equivalent, steady-state model of one moment. Voltages
in **polar form**; power balances in MW and Mvar; currents in per unit of each
bus's current base.

### Sets

| Symbol | Meaning |
| --- | --- |
| $\mathcal B$ | buses in service; $s$ the slack bus, at the **first** external grid |
| $\mathcal A$ | directed bus pairs $(i,j)$ of the lines and two-winding transformers in service, both directions; parallel branches form one pair |
| $\mathcal G$ | units in service: static generators and voltage-controlled machines |
| $\mathcal G^{V} \subseteq \mathcal G$ | voltage-controlled units (machines, synchronous condensers) |
| $\mathcal G^{PF} = \mathcal G \setminus \mathcal G^{V}$ | the others, held to a power factor |
| $\mathcal L$, $\mathcal S$, $\mathcal E$ | loads, shunts, external grids |

### Variables

| Variable | Bounds |
| --- | --- |
| $V_b,\ \theta_b$ | $\underline V_b \le V_b \le \overline V_b$; $V_s,\ \theta_s$ fixed at their power-flow values; every other bus starts from the moment's power flow |
| $P_g$ | $0 \le P_g \le \overline P_g$ |
| $Q_g$ | limited by the constraints below |
| $P_e^{\text{ext}},\ Q_e^{\text{ext}}$ | $\lvert P_e^{\text{ext}} \rvert \le \overline P_e$, a symmetric capacity: the larger magnitude of the case's minimum and maximum (±200 MW when the case gives none); $\lvert Q_e^{\text{ext}} \rvert \le 200$ Mvar. The exchange cap of the capped solve can only tighten these |
| $P^{dr}_l,\ P^{\text{shed}}_l$ | demand response and load relief; **fixed at zero** unless a strategy frees them |

### Constraints

**Power balance** at every bus $b$ (active; the reactive balance is the
imaginary counterpart):

$$
\begin{aligned}
&\sum_{g \in \mathcal G_b} P_g + \sum_{e \in \mathcal E_b} P^{\text{ext}}_e - \sum_{l \in \mathcal L_b} \sigma_l P^L_l - \sum_{k \in \mathcal S_b} P^{sh}_k V_b^2 \\
&\qquad = S_{\text{base}} \sum_{j:(b,j) \in \mathcal A} \Big[ V_b^2 G^{ff}_{bj} + V_b V_j \big( G^{ft}_{bj} \cos\theta_{bj} + B^{ft}_{bj} \sin\theta_{bj} \big) \Big]
\end{aligned}
$$

where $Y^{ff}_{bj} = G^{ff}_{bj} + jB^{ff}_{bj}$ and $Y^{ft}_{bj}$ are the
branch's π-model admittances seen from bus $b$, and $\sigma_l = 1 - (P^{dr}_l + P^{\text{shed}}_l)/P^L_l$ is the share of load $l$ served (1 unless load is
reduced; a reduced load keeps its power factor). Loads are constant power;
shunts are constant impedance.

**Branch currents**, at both ends of every branch, with the current
$I_{ij} = Y^{ff}_{ij} \tilde V_i + Y^{ft}_{ij} \tilde V_j$ in per unit, where
$\tilde V_i = V_i e^{j\theta_i}$ is the complex voltage:

$$
|I_{ij}|^2 \le \mu_{ij}^2 \left( \frac{\overline I_{ij}}{I^{\text{base}}_{ij}} \right)^2
$$

$\overline I_{ij}$ is the rating (a line's maximum current; a transformer's rated
power over $\sqrt3$ times the bus voltage), $I^{\text{base}}_{ij} = S_{\text{base}} / (\sqrt 3\, V^{n}_i)$. The **current margin** $\mu_{ij}$ is 0.9 by default —
90 % of rated current — or the share the operator gives that branch
(`branch_limits`, 10–100 %). $\mathcal A$ holds both directions of every
branch, so each end's limit appears once: the limit at bus $i$ comes from the
pair $(i,j)$, the one at bus $j$ from $(j,i)$.

**Unit reactive limits.** A unit held to a power factor (minimum 0.95 by
default):

$$
Q_g^2 \le \tan^2\!\varphi\; P_g^2, \qquad \cos\varphi = 0.95 \;\Rightarrow\; \tan\varphi = 0.329
$$

a cone that forces $Q_g = 0$ at $P_g = 0$. A voltage-controlled unit:
$\underline Q_g \le Q_g \le \overline Q_g$ (its own range, ±50 Mvar when the case
gives none). Its bus voltage is a free variable within the band: the
optimisation returns a new **set-point**, which an option then states as an
instruction.

**Voltage band.** $\underline V \le V_b \le \overline V$, the network's own band
unless the question gives one (0.95–1.05 p.u. on the 30-bus demo), optionally
tightened bus by bus. The slack voltage is fixed: the grid-connection voltage is
not a lever (its bounds widen to admit its measured value when that lies outside
the band). The result reports the band asked for (`opf_vm_lower_used`,
`opf_vm_upper_used`), not any one bus's bounds.

**Held units** — the operator's conditions and pinned set-points — have their
active power fixed. A set-point outside its unit's range (or the exchange's
bounds) is a contradiction in the request: the flexibility and post-contingency
studies refuse it, naming each one (HTTP 422); a set-point within 1 kW of a
bound is placed on the bound. Capacity overrides (`pg_max_overrides`,
`pg_min_overrides`) must not be negative.

### Objective

The smallest weighted change to the schedule, in MW and Mvar:

$$
\min\; \lambda_p \sum_{g \in \mathcal G} \big(P_g - P_g^0\big)^2 + \lambda_q \sum_{g \in \mathcal G} \big(Q_g - Q_g^0\big)^2
$$

with $\lambda_p = 0.01$ and $\lambda_q = 0.001$.

$P_g^0, Q_g^0$ are the schedule: the power flow of the moment (a
voltage-controlled unit's reactive output clipped to its range). A Mvar of change
weighs ten times less than a MW. The external grid is not in the objective: it
takes any imbalance, within its bounds. Other studies replace this objective —
the cost of the action, the import, the headroom (see
[Corrective options](corrective-options.md)).

## The admittances

The π-model admittances of every branch come from pandapower's own internal
model of the network, built once when the network is loaded:

$$
\begin{aligned}
Y^{ff}_{ij} &= \frac{y_s + y_c/2}{|t|^2}, & Y^{ft}_{ij} &= -\frac{y_s}{t^*}, \\
Y^{ff}_{ji} &= y'_s + y'_c/2, & Y^{ft}_{ji} &= -\frac{y'_s}{t}
\end{aligned}
$$

with $y_s = 1/(r + jx)$ the series admittance, $y_c = g + jb$ the shunt
admittance (the conductance $g$ — a transformer's iron losses, a line's
conductance — and the charging susceptance $b$) and $t = \tau e^{j\varphi}$ the
transformer ratio and phase shift, all in per unit on the system base. The
to-end terms $y'_s$, $y'_c$ add the branch's asymmetric parts, where it has any
($r + r^{\text{asym}}$, $x + x^{\text{asym}}$, $g + g^{\text{asym}}$,
$b + b^{\text{asym}}$); otherwise they equal $y_s$, $y_c$. This is the π-model
pandapower builds its own bus admittance matrix from. Parallel branches between
the same buses are added into one equivalent branch.

Evaluated at a converged power flow, these admittances reproduce pandapower's
bus injections on the three bundled networks to within $10^{-6}$ MVA.

An uploaded network can instead bring its admittances as CSV files; the upload
is refused (HTTP 400) unless they cover every entry the optimisation reads —
each in-service branch in both directions — and only buses the network has.

**An outage needs no matrix of its own**: the optimisation reads only branches
in service, so an outaged branch drops out of the balance equations. The
exception is a branch with a parallel circuit between the same buses, whose
admittances were added into one: after it trips, the admittances are rebuilt
from the post-outage network.

**Transformer taps** are fixed at their position when the network was loaded;
there is no tap, phase-shifter or switched-shunt decision.

## Solving it

1. **Prepare the moment.** The network with the moment's measurements, any
   units the question takes offline removed (`disabled_generators`: static and
   voltage-controlled units alike; an unknown name is refused), the operator's
   held units turned into fixed set-points at their schedule, branch caps, load
   scaling, and a power flow of the network as studied (loads scaled) — the
   schedule $P^0, Q^0$, the external grid's exchange before redispatch and the
   slack voltage come from it.
2. **Fold 1: is the schedule already secure?** Every unit fixed at its schedule:
   with the slack's $V, \theta$ fixed this is an AC power flow, solved together
   with every inequality. If it is feasible, nothing needs to move.
3. **Fold 2: redispatch.** The units are freed and the objective minimised.
4. **The exchange cap.** The model is solved again with the external grid's
   exchange bounded by the cap in force, $\lvert P_e^{\text{ext}} \rvert \le$
   `slack_max_mw`: the network's own limit (the profile's, 999 MW on the bundled
   networks; for an upload, the external grid's maximum, or 999 MW when it has
   none) unless the question gives a lower one. The reactive bound is the same
   figure unless given separately (`slack_q_max_mvar`). A cap only tightens the
   connection's capacity (above), never loosens it. If the capped solve has no
   solution, the result says so (`cap_infeasible`) with a diagnosis — the import
   or export the moment would need against the limit, or a numerical failure
   when the uncapped dispatch already fits — and shows the uncapped dispatch
   only as a reference. The post-contingency optimisation solves again under
   the same cap.

The solver is **IPOPT** (interior point, MUMPS linear solver) with its default
options: tolerance $10^{-8}$, at most 3,000 iterations, exact Hessian. Every
solve is single-threaded. A solution counts only when IPOPT reports it optimal.

## When there is no solution

When Fold 2 finds no dispatch, the study first asks whether one exists, then
explains why and offers what is left.

**Proof, or a second start.** IPOPT failing from the power-flow point does not
show that no dispatch exists. The study therefore solves the convex relaxation
of the same model (the semidefinite relaxation of
[Corrective options](corrective-options.md#how-close-the-cheapest-is-to-the-best-possible),
with the MW moved from the schedule, $\sum_g |P_g - P^0_g|$, as its objective):

- **Infeasible** (the conic solver's infeasibility certificate): since the
  relaxation holds every AC solution, **no dispatch of the units meets these
  limits — proved**. The result's `no_dispatch_check` says
  `no_dispatch_exists` and the message leads with it.
- **Feasible**: IPOPT is restarted from the relaxation's solution — voltages
  $\sqrt{c_{ii}}$, angles from $(c_{ij}, s_{ij})$ along a spanning tree from the
  slack, every unit's and grid's output. A dispatch found this way is the
  study's answer, marked `found_by_relaxation`.
- Otherwise — the restart finds nothing either, or the relaxation could not be
  solved to an optimal status — `not_proven`: none was found, which does not
  show that none exists.

The same check runs when the re-solve under the exchange cap fails (the proof
is then for the exchange within the cap), and after an outage
(`optimize_contingency`). An IPOPT failure on the cone relaxation, used where
the semidefinite one is not available, is never taken as a proof. On the
30-bus demo at its evening peak, loads at 125 %, an import cap of 10 MW, and the
loss of L1 with loads at 110 % are each proved to have no dispatch, in about a
second.

**A diagnosis**, on a copy of the model, never applied; each re-solve restarts
from the power-flow point:

1. Free the slack voltage within the band. If a dispatch now exists, the binding
   limit is **voltage support**: the answer gives the grid voltage that would be
   needed and the reactive sources at their limits.
2. Otherwise lift the current limits. If a dispatch now exists, the binding limit
   is **branch loading**: the answer lists the branches over their rating and, for
   the worst, the lowest loading it can reach.
3. Otherwise both together.
4. Otherwise lift the external grid's exchange limits. If a dispatch now exists
   outside them, the binding limit is the **exchange** (`exchange_limit`): the
   answer gives the import or export the moment would need against the limit.
   Failing that, the violations are beyond what voltage and branch levers can
   reach.

**The last resort**: options that also reduce load — see
[Corrective options](corrective-options.md#when-redispatching-is-not-enough).

## After the optimisation

Every option is checked by an ordinary AC power flow (see
[Corrective options](corrective-options.md#every-option-checked-by-a-power-flow)):
the optimisation proposes, the power flow confirms or not. The single
least-disturbance dispatch a study returns when no options are asked for is not
re-checked this way; ask for options to have the check.

## Assumptions and limits

- **Non-convex, solved locally.** The polar balances, the current limits and the
  cone make the problem non-convex; IPOPT returns a local optimum. "No solution"
  is checked by the relaxation (above): proved, found on a second start, or not
  proven either way. The
  voltages and angles start from the moment's power flow (clipped to their
  bounds), and the diagnosis and the last resort restart from that point.
- **Not modelled:** voltage-dependent loads; tap and shunt control; angle
  difference limits; apparent-power ratings (only current is limited — at
  voltages below 1 p.u. slightly stricter than the equivalent MVA limit); a unit's
  capability curve (the cone or the reactive range instead); a unit's minimum
  active power (every unit may go to zero); the external grid's own reactive
  limits (±200 Mvar instead) and any minimum import; ramping (except in
  [a plan](over-time.md#one-plan-for-a-window)).
- **Elements outside the model:** three-winding transformers, impedance elements,
  wards, DC lines and storage; buses joined by bus–bus switches. The bundled
  networks have none.
- **Only the first external grid** is the reference; any other is a free injection.
- **Parallel branches** are one equivalent branch with the ratings added: exact
  for identical circuits.
- **A converged power flow is needed** at the moment studied: without one there is
  no schedule to start from, and the study fails.
- **The exchange bounds are a capacity, not the case's generator limits.** A
  MATPOWER case describes the external grid as a slack generator: its minimum
  output (50 MW on the 30-bus case) is not a minimum import, and its reactive
  limits are not the connection's. Only the larger magnitude of its active
  limits is kept, as a capacity in both directions (±200 MW on the 30-bus
  case).

## In the code

| What | Where |
| --- | --- |
| The model and Fold 1/2 | `optimization_model_base` in `backend/flex_engine.py` |
| The admittance database | `build_ybus_full`, `_build_ybus_entry` in `backend/network_loader.py` |
| Exchange bounds, set-points checked against their ranges | `_exchange_bounds`, `_checked_pins` in `backend/flex_engine.py` |
| The diagnosis, restarts | `diagnose_infeasible`, `_exchange_check`, `_restart` in `backend/flex_engine.py` |
| Proof that no dispatch exists, second start | `second_look`, `proof_block` in `backend/relaxation.py`; `_second_look`, `_with_proof` in `backend/main_backend.py` |
| The exchange cap and its diagnosis | `solve_constrained_slack_scenario`, `_cap_exchange`, `_physical_exchange_cap`, `_exchange_cap_diagnosis` in `backend/main_backend.py` |
| Held units, branch caps | `_held_at_schedule`, `_branch_caps` |
| The study | `POST /api/flexibility/optimize` (`optimise_flexibility`), tool `optimize_flexibility` |
