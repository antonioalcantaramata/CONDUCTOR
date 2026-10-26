# Contingency analysis (N-1)

A contingency study asks what happens to the grid **after** an element trips.
CONDUCTOR studies single outages of lines and two-winding transformers: one at
a time (a single outage), or all of them in turn (the N-1 screen). Corrective
dispatches that stay secure after an outage are the subject of
[N-1 security](n1-security.md).

## One outage

The study takes the network at the moment (see
[Network, data and the operating point](network-and-data.md)), switches the
element out of service, and solves an AC power flow on what remains:

- Newton–Raphson, with pandapower's defaults (mismatch tolerance
  $10^{-8}$ MVA, at most 10 iterations, initialised from a DC power flow, so
  each outage is solved independently of the previous one);
- the **external grid is the only slack**: it takes whatever imbalance the
  outage creates. Units keep their pre-outage active power and voltage
  set-points; there is no governor response, no distributed slack and no tap
  or shunt action;
- unit reactive limits are **not enforced** in this power flow: a unit holding
  a voltage holds it whatever reactive power that needs.

The result is then judged against the limits like any other power flow (see
[Power flow and security assessment](power-flow-and-security.md#what-counts-as-a-violation)):
bus voltages outside $[\underline V, \overline V]$, lines and transformers
above their loading limit, each widened by the screen's solver tolerance
($10^{-4}$ p.u., 0.01 %; see [the N-1 screen](#the-n-1-screen)). The limits are
the network's own unless the question gives others — on the bundled networks
100 % of each rating, and 0.95–1.05 p.u. (IEEE 30) or 0.94–1.06 p.u. (IEEE 14).

**Islanding.** An outage may leave buses without a path to a source. The
single-outage study finds them with pandapower's connectivity search
(`unsupplied_buses`, sources being the external grids and slack units) and
reports the buses, their number and the load on them (`unsupplied_load_mw`),
with a sentence on what was disconnected. An unsupplied bus has no voltage to
violate, so it adds no violation; but an outage that cuts load off is not
secure: `system_secure` is false, and when no limit breaks the message says
how many MW of load the outage cuts off.

**Non-convergence.** When the power flow finds no solution after the outage,
the result says so (`converged: false`) and asks for it to be treated as
insecure — never as zero violations.

## The N-1 screen

The screen repeats the single-outage study for **every in-service line and
transformer**, one at a time, restoring each before the next. A branch already
out of service is not a contingency, nor is one whose end bus is out of service
(an island an earlier outage left, in the N-1-1 screen below). For each outage
it collects the violations, tagged with the outage that caused them
(`outage_cause`), notes any load the outage cuts off, and reports:

| Field | Meaning |
| --- | --- |
| `total_outages_tested` | outages attempted (every in-service line and transformer) |
| `total_outages_causing_violations` | outages after which at least one limit breaks |
| `total_violations` | violations over all outages (one per element and limit, per outage) |
| `non_converged_outages` | outages whose power flow found no solution, or failed |
| `outages_disconnecting_load` | outages that leave buses carrying load without supply: the outage, the buses and the MW cut off (`total_outages_disconnecting_load` counts them) |
| `system_n1_secure` | true only with no violation, no outage that failed to converge **and** no outage that cuts load off |

An outage that cuts load off is counted apart from the violations: the buses it
leaves without supply have no voltage, so they break no limit, but the load on
them is lost. An outage that only disconnects a unit, and no load, is not
listed there; whatever its lost output does to the rest of the grid shows as
violations.

Each limit is widened by a solver tolerance — $10^{-4}$ p.u. on voltages,
0.01 % on loadings — so a unit holding exactly the edge of the band is not read
as "1.05 p.u. above 1.05". The single-outage study uses the same tolerance, so
one outage reads the same there and in the screen.

On the 30-bus demo the screen tests 41 outages. At the evening peak
(6 January, 20:00) all 41 leave a violation, because the schedule itself breaks
two limits before any outage; with the cheapest corrective action applied, 29
do. Line L33 cuts off Bus_25 (3.5 MW in the case), so the screen lists it among
the outages that cut load off at every moment, and never calls the demo N-1
secure.

## N-1-1: after an outage, the next one

A corrective dispatch computed **after** an outage (the post-contingency
optimisation below) keeps its outage with it. Screening it re-applies that
outage first — the element out, and any bus it left without supply out of
service — then its dispatch, and tests every further single outage: the
**N-1-1** screen. The answer labels it so.

## The post-contingency optimisation

*What should we do if line L31 trips?* solves the
[optimal power flow](optimal-power-flow.md) on the network without the element:

1. the outage as the screen applies it, plus every bus left without supply
   taken out of service (their load reported as `load_not_supplied_mw`);
2. a power flow as the starting point;
3. the OPF with its normal-operation limits — the voltage band and the current
   margin (90 % of each rating by default) — and the operator's held units and
   branch caps. Every unit may use its full range. The admittances are the
   network's own without the outaged branch, rebuilt from the post-outage
   network when a parallel circuit joins the same buses;
4. the exchange cap, as in the flexibility study: the OPF is solved again with
   the external grid's exchange within the network's own cap (`slack_max_mw`;
   see [The optimal power flow](optimal-power-flow.md#solving-it)). When only
   that capped solve fails, the answer gives the import or export the outage
   would need against the limit, or calls the failure numerical when the
   uncapped dispatch already fits;
5. the result: one dispatch, or, when options are asked, the usual
   [corrective options](corrective-options.md) on the post-outage model, each
   checked by an AC power flow on the post-outage network. When no dispatch
   exists, the last-resort options reduce load (see
   [Corrective options](corrective-options.md#when-redispatching-is-not-enough)).

This is a corrective action for an outage that has *happened*. Planning for
outages that *might* happen is [N-1 security](n1-security.md), which uses
emergency ratings after the outage instead of the normal margin.

## Violation attribution

*Which sources drive the violations?* is answered numerically, on the intact
network at one moment (not after an outage):

1. A power flow, and the violated quantities $x_v$ — a bus voltage or a branch
   loading.
2. The sources: every unit and load in service with more than 0.01 MW or Mvar,
   the largest 20 by active power. The external grid is the slack and is never
   perturbed.
3. For each source $s$ and each of its active and reactive outputs $u_s$, a
   forward difference with a step $\delta = 0.1$ MW (or Mvar), one power flow
   each:

$$
S_{v,s} = \frac{x_v(u_s + \delta) - x_v(u_s)}{\delta}
$$

4. For each violation, the drivers are the sources with
   $\max(|S^{P}|, |S^{Q}|) \ge 10^{-9}$, ranked by that influence (the five
   strongest, plus at least two the operator controls when the strongest are
   loads).
5. The move that would bring the quantity to its limit if that source acted
   alone, linearly:

$$
\Delta u_s = \frac{x_v^{\text{lim}} - x_v}{S_{v,s}}
$$

   A move that would need negative output is withdrawn as impossible (the most
   the source can deliver is given instead); a move by a load is shown as what
   it *would* take, not as an action.
6. For a voltage violation, the **voltage set-points** of the units that hold
   one (not the external grid, whose voltage this operator does not set). A
   unit on voltage control has no reactive injection to perturb, so the steps
   above never see its real lever. Each unit's set-point is stepped by
   0.005 p.u. to measure its sensitivity; for the three most sensitive, the
   set-point that brings the bus 0.001 p.u. inside its limit is found (linear,
   then corrected by re-solving up to three times, rounded to 0.001 p.u.) and
   **checked by a full power flow**: the violation cleared, the unit's own bus
   and set-point within the voltage band, its reactive output within its range,
   and no element outside its limits that was not before. A move is `verified`
   only if every check passes; otherwise its `reason` says which failed. This
   is done for the five worst voltage violations, at most 9 power flows each.
7. A recommended action, chosen in code: a voltage set-point where a unit holds
   the violated bus's voltage; otherwise the smallest verified set-point move;
   otherwise the smallest single-source move that is certainly deliverable;
   otherwise the smallest with unknown headroom. On the IEEE 30-bus demo's
   evening undervoltage at Bus_29, this replaced "a 192.81 MW increase of
   Gen_bus7" with "raise Gen_bus7's set-point from 1.000 to 1.014 p.u.",
   checked: Bus_29 at 0.951 p.u., Gen_bus7 at 39.5 Mvar of its −15 to 60 Mvar.

The study costs at most $2 + 2N$ power flows for $N$ sources (the base, one
per perturbed output, and one that restores the unperturbed state), plus, with
voltage violations, one per voltage-controlling unit and up to nine per
violation checked for a set-point move (`n_setpoint_power_flows`). A
perturbation whose power flow does not converge leaves that sensitivity
missing for the source, and the result names the sources. Sensitivities are local
and linear — they describe small moves from this operating point, one source at a
time — so a large or combined move should be checked with an optimisation.
Percentage shares are not computed here: a share needs a reference state. The
[investigation](investigation.md) sets one explicitly (a moment, dataset or
network where the element was fine) and divides the change since then among
its causes; the attribution answers what to move now.

## Assumptions and limits

- **Which outages.** Single lines and two-winding transformers. Generator,
  bus, busbar and common-mode outages are not studied.
- **Post-outage physics.** One slack (the external grid), no reactive-limit
  enforcement, no automatic tap or shunt action: a unit past its reactive range
  still holds its voltage in the screen. The power-flow check of an option does
  flag units outside their reactive limits.
- **Islanding.** Load an outage cuts off is reported, not modelled further: no
  load transfer, no restoration. The screen counts such outages apart from
  those with violations; either makes the grid not N-1 secure.
- **Parallel circuits.** The bundled networks have none. The power flows handle
  them, and the optimisation, which sums parallel branches per pair of buses,
  rebuilds its admittances from the post-outage network after one of two
  parallel circuits trips (which needs that network's power flow to converge).
- **Tolerance.** The screen and the single-outage study widen limits by a
  solver tolerance ($10^{-4}$ p.u., 0.01 %); the security assessment of the
  intact network uses $10^{-6}$. A loading of 100.005 % is a violation in the
  assessment and not after an outage.

## In the code

| What | Where |
| --- | --- |
| One outage: the power flow and its violations | `contingency_assessment_single` in `backend/ca_engine.py` |
| Unsupplied buses and load | `_supply_lost`, `_load_cut_off` in `backend/main_backend.py` |
| The N-1 screen | `_n1_sweep`, `POST /api/contingency/simulate_all` |
| One outage | `POST /api/contingency/simulate` |
| Post-contingency optimisation | `POST /api/contingency/optimize` (`optimize_contingency`, `_outage_admittances`) |
| Violation attribution | `attribute_violations` in `backend/attribution_engine.py`, `POST /api/rsa/attribution` |
| Tools | `simulate_contingency`, `simulate_all_contingencies`, `optimize_contingency`, `compute_violation_attribution` |
