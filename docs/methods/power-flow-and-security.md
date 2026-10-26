# Power flow and security assessment

The state of the grid at a moment is an **AC power flow**: given what every
load consumes and every unit produces, the bus voltages and branch flows that
satisfy the network's physics. A **security assessment** compares that state with
the limits.

## The power flow

With complex bus voltages $\tilde V_i = V_i e^{j\theta_i}$ and the bus admittance
matrix $\mathbf Y = \mathbf G + j\mathbf B$ built from the branches' π-models and
the shunts, the injections at each bus satisfy

$$
\begin{aligned}
P_i &= \sum_k V_i V_k \big( G_{ik} \cos\theta_{ik} + B_{ik} \sin\theta_{ik} \big) \\
Q_i &= \sum_k V_i V_k \big( G_{ik} \sin\theta_{ik} - B_{ik} \cos\theta_{ik} \big)
\end{aligned}
$$

in per unit of the system base (100 MVA on the bundled networks), with the
specified injection $S_i = \sum_{g \in i} (P_g + jQ_g) - \sum_{l \in i} (P_l + jQ_l)$.
Buses are of three kinds:

| Bus | Known | Solved for |
| --- | --- | --- |
| slack (each external grid) | $V$, $\theta$ | $P$, $Q$: the grid takes the balance |
| PV (a voltage-controlled unit) | $P$, $V$ (its set-point) | $Q$, $\theta$ |
| PQ (everything else) | $P$, $Q$ | $V$, $\theta$ |

Loads are constant power and shunts constant admittance.

**Solver.** pandapower's Newton–Raphson: angles initialised from a DC power flow,
at most 10 iterations, converged when the largest mismatch is below
$10^{-8}$ per unit ($10^{-6}$ MVA on a 100 MVA base). Reactive limits of
voltage-controlled units and the external grid's limits are **not enforced**: a
PV unit holds its set-point whatever reactive power it needs. When the power flow
does not converge, the study says so and treats the moment as insecure, never as
violation-free.

## What counts as a violation

Each in-service element is compared with its limit, with a tolerance
$\epsilon = 10^{-6}$ that absorbs rounding (a dispatch placed exactly on a limit):

| Element | Violation |
| --- | --- |
| bus | $V < \underline V - \epsilon$ or $V > \overline V + \epsilon$ (p.u.) |
| line | loading $\ell > \overline\ell^{\,\text{line}} + \epsilon$ (%) |
| transformer | loading $\ell > \overline\ell^{\,\text{trafo}} + \epsilon$ (%) |

A line's loading is its current against its rating (with its derating factor
$df$ and number of parallel circuits); a transformer's is its current at rated
voltage against its rated power:

$$
\ell_{\text{line}} = 100\, \frac{\max\big(|I_{\text{from}}|,\, |I_{\text{to}}|\big)}{I^{\max}\, df\, n^{\text{par}}}
$$

$$
\ell_{\text{trafo}} = 100\, \frac{\max\big(\sqrt3\, |I_{hv}|\, V^n_{hv},\; \sqrt3\, |I_{lv}|\, V^n_{lv}\big)}{S_n\, n^{\text{par}}\, df}
$$

One record per element and limit: the element, the kind of violation and its
value. The limits used are
stated once with the result (`thresholds_used`).

**The limits** are the network's own unless the question gives others: one
voltage band for every bus and one loading limit for every branch — on the
bundled networks 100 % of each rating, and 0.95–1.05 p.u. (IEEE 30) or
0.94–1.06 p.u. (IEEE 14). A limit given explicitly always wins, even if it equals
a default.

**Not checked:** unit reactive limits and the external grid's limits in the
assessment itself (the power-flow check of a corrective option does flag unit
reactive limits), angle limits, per-element limits that differ from the
network-wide ones.

## What the assessment reports

The voltage at every bus and the loading of every branch (an out-of-service
branch is marked, so its 0 % is not read as lightly loaded), the violations, the
total load and import, each unit's active and reactive output (static and
voltage-controlled units alike, and the external grid) and the spare upward
capacity of the units together (`total_available_capacity_mw`), the
voltage-controlled buses with each unit's set-point and reactive headroom, and
the limits used. Corrective action, contingencies and every other study build on
this state.

## Assumptions and limits

- **Steady state, balanced**: one phase-equivalent snapshot; no dynamics, no
  protection, no unbalance.
- **Library defaults**: the power-flow settings are pandapower's, and the
  installed version is not pinned beyond 3.0.
- **Reactive limits**: not enforced, so a voltage-controlled unit may hold its
  set-point with more reactive power than it has. On the default week this
  happens at 59 of 672 moments.
- **Two tolerances**: the assessment uses $10^{-6}$, and so do the worst-moment
  and renewable-scenario scans, the probabilistic assessment, the points of the
  flexibility envelope and of hosting capacity, and the robust dispatch's
  evaluation; after an outage — the N-1 screen and the single-outage study —
  limits are widened by $10^{-4}$ p.u. and 0.01 % (see
  [Contingency analysis](contingency-analysis.md)).

## In the code

| What | Where |
| --- | --- |
| The power flow and the violation checks | `real_time_security_assessment` in `backend/rsa_engine.py` (`FLOAT_TOL`, shared as `_LIMIT_TOL` in `backend/main_backend.py`) |
| The moment's network, the assessment's result | `_prepared_net`, `_run_rsa_snapshot` in `backend/main_backend.py` |
| Limits defaulting to the network's own | `_NetworkLimitsMixin`, `_fill_network_limits` |
| Study | `POST /api/grid/rsa`, tool `run_rsa` |
