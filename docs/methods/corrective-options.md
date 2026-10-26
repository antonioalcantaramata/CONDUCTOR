# Corrective options

When the schedule breaks a limit, there is rarely one right corrective action:
moving fewer units, moving cheaper ones, importing less, keeping more headroom
are different goals. CONDUCTOR therefore offers **options** — each the optimum
of a stated strategy on the same model, scored the same way, checked by the
same power flow — and leaves the choice to the operator.

Every option is a re-solve of the [optimal power flow](optimal-power-flow.md)
with a different objective or extra constraints. Nothing is estimated: every
figure is read from a solved model.

## Notation

In addition to [the overview's](index.md#notation): $\Delta P_g = P_g - P_g^0$;
$D(P) = \sum_{g} (P_g - P_g^0)^2$ the disturbance in active power; $\mathcal X$
the OPF's feasible set (the AC power flow equations, the voltage band, the
current margin, unit and reactive limits, the external-grid cap, the operator's
held units and branch limits — see [The optimal power flow](optimal-power-flow.md));
$D^{dr}_d, D^{rel}_d$ MW of demand response and of load relief at load $d$
(zero unless a strategy frees them).

**Keeping a unit untouched.** Several strategies require some units to receive
no instruction at all, written $\mathrm{KEEP}(S)$: each unit in $S$ keeps its
scheduled active power; a unit that holds a voltage keeps its set-point (its
reactive output follows the grid); any other unit keeps its reactive output.

**An instruction** is what an operator would have to issue: "P" when a unit's
active power changes by at least 0.01 MW; "V" when a voltage-holding unit's
set-point changes by at least 0.001 p.u.; "Q" when another unit's reactive
output changes by at least 0.05 Mvar. *Units called* is the number of units
given an instruction.

## The strategies

### Least disturbance

The OPF's own objective, the smallest weighted change to the schedule:

$$
\min_{x \in \mathcal X}\; \lambda_p \sum_g \big(P_g - P_g^0\big)^2 + \lambda_q \sum_g \big(Q_g - Q_g^0\big)^2
$$

with $\lambda_p = 0.01$ and $\lambda_q = 0.001$.

Squares spread a correction over several units rather than concentrating it.
This is the reference every other option starts from, and it is always shown.

### Cheapest

The cost of the action (below) with a small tie-break on disturbance:

$$
\min_{x \in \mathcal X}\; C(x) + 10^{-3}\, D(P)
$$

Reactive output is priced (per Mvarh) rather than penalised by $\lambda_q$.

### Fewest units

A greedy search, not a minimum-cardinality search. The units the
least-disturbance option instructs are ranked — largest active-power movers
first, then those changing only a set-point — and for $k = 1, 2, \dots$ (at most
four, and fewer than the least-disturbance option calls) the least-disturbance
problem is solved with every unit outside the top $k$ kept untouched. The first
$k$ that is feasible is the option. When there is none, the answer says why:
the least-disturbance option already calls one unit, or no dispatch instructing
only the four units it leans on most (or fewer) clears the violations.

### Without the main unit

The least-disturbance problem with $\mathrm{KEEP}(\{g^\star\})$, where $g^\star$ is
the unit the least-disturbance option moves most. The label names the unit
("Without Gen_bus2"): the unit left out, not one the operator ruled out. Not
offered when the least-disturbance option moves no unit's active power.

### Least import

Two stages. First the smallest import:

$$
I^\star = \min_{x \in \mathcal X,\; I \ge 0}\; I + 10^{-4} D(P) \quad \text{s.t.} \quad I \ge \sum_e P_e^{\text{ext}}
$$

then the least change to the schedule at (about) that import:

$$
\min_{x \in \mathcal X}\; D(P) \quad \text{s.t.} \quad \sum_e P_e^{\text{ext}} \le 1.001\, I^\star + 10^{-3}
$$

Not offered when the grid is already exporting.

### Most headroom

The lowest highest loading, over both ends of every rated branch:

$$
\min_{x \in \mathcal X,\; s \ge 0}\; s + 10^{-4} D(P) \quad \text{s.t.} \quad |I_{ij}|^2 \le s\, \bar I_{ij}^{\,2}
$$

$s$ is the squared highest loading as a fraction of the full rating; the
current margin still applies.

### With demand response

Demand response is freed at every load with available demand response (a share
of the load, from the cost table), then the cheapest problem is solved with its
price. Offered only when the solution actually uses demand response; otherwise
the answer says redispatching the units is cheaper, or that no load has
demand-response capacity in the cost table.

### Selecting what is shown

Up to seven candidates (least disturbance and the six strategies) become at most
five options:

1. **Duplicates.** An option within 0.1 MW of an earlier one on every unit and
   every load reduction, and instructing the same units, is dropped, and the
   kept one lists it ("least disturbance is also the cheapest"). Two options
   with the same active power are not duplicates when one calls a unit the
   other does not (a voltage or reactive instruction only one of them gives).
2. **Dominated options.** An option is dropped when another is at least as good
   on every count and better on one: units called, MW moved, cost, import,
   highest loading, MW of load reduced. Least disturbance always stays.
3. **At most five.** While more remain, the option closest to another (in MW,
   summed over units) is dropped — sparing least disturbance, cheapest, and any
   option that is the best at something.

The order is the order of generation; nothing ranks them. Every option not shown
is listed with the reason (*not offered*).

## The cost of an action

Every option is priced the same way, in €/h, as a change from the schedule — doing
nothing costs nothing:

$$
\begin{aligned}
C ={}& \sum_g \Big[ c^{\uparrow}_g \max(\Delta P_g, 0) + c^{\downarrow}_g \max(-\Delta P_g, 0) + c^{q}_g\, |\Delta Q_g| \Big] \\
&+ \sum_e c^{\text{imp}}_e \big(P_e^{\text{ext}} - P_e^{\text{ext},0}\big) \\
&+ \sum_e c^{\text{pen}}_e \Big( \max\big(|Q_e^{\text{ext}}| - \beta_e, 0\big) - \max\big(|Q_e^{\text{ext},0}| - \beta_e, 0\big) \Big) \\
&+ \sum_d \Big[ c^{dr}_d\, D^{dr}_d + c^{rel}_d\, D^{rel}_d \Big]
\end{aligned}
$$

- **Redispatch**: an upward and a downward price per unit.
- **Reactive power**: a price per Mvar changed.
- **Import**: signed — less import is a saving.
- **Reactive exchange** with the external grid: free within a band
  $\beta_e = \tan\varphi_e\, |P_e^{\text{ext},0}|$, penalised beyond it.
- **Load reduction**: demand response and load relief, each at its price.

Inside an optimisation the same cost is a linear expression with split
variables ($\Delta P_g = \delta^+_g - \delta^-_g$, and so on), so minimising it
stays smooth.

**Where the prices come from**, in order:

1. the **case file** for production costs (the PGLib cases carry them);
2. a **synthetic rule** for everything else: a unit's upward price is 1.2 times
   its marginal production cost at full output, its downward price 0.5 times (a
   negative marginal cost counts as zero); 3 €/Mvarh for reactive power; import
   60 €/MWh (or the case's), a free band of $\tan\varphi = 0.4$, 10 €/Mvarh
   beyond it; demand response on 10 % of each load at 150 €/MWh, load relief at
   3,000 €/MWh; a PV or wind unit a proposal adds can be curtailed at 40 €/MWh
   and has no upward price;
3. the **operator's edits** in the System tab, kept per network.

A **synchronous condenser** — a voltage-controlled unit with under 1 MW of
declared capacity, the same rule the optimisation and the unit table use — has
no upward or downward price until the operator gives one: it has no active power
to move. A small static generator is an ordinary unit, with both prices. The
**external grid** has no redispatch price at all, and refuses one: its exchange
follows from the units' redispatch and is priced at the import price. A price
left empty counts as 0 €/MWh.

With the synthetic prices (production 50 €/MWh, so up 60 and down 25; import
60), raising a unit 1 MW so that import falls 1 MW costs about 0 €/h; lowering
one so that import rises costs about 85 €/h. That is why the cheapest option
favours raising units that displace import, and why it may move more megawatts
than strictly needed (see KPI-2 in [Key performance indicators](key-performance-indicators.md)).

## Every option, checked by a power flow

Each option's dispatch is applied to a copy of the network — every unit's
active power, the set-points of voltage-holding units, the reactive output of
the others, the load reductions at constant power factor — and an ordinary AC
power flow (Newton–Raphson) is solved, with the external grid taking the
balance. The option is **confirmed** when the power flow converges with:

- every bus within the optimisation's voltage band, ±0.002 p.u.;
- every line and transformer within 100 % of its rating (or the operator's cap
  for a capped branch), +0.1 %;
- every voltage-holding unit within its reactive limits, ±0.5 Mvar (the power
  flow does not enforce them, so the check does).

The optimisation and the power flow are different models — the optimisation
works from its admittance database and its own bounds, the power flow from the
full network — and the check is there to expose any disagreement. It does not check the current
margin (the optimisation's 90 %), which is a margin, not a limit; nor per-bus
voltage limits the optimisation was given (it uses the one band), nor the
exchange cap.

## How close the cheapest is to the best possible

The optimisation is not convex and IPOPT finds a local optimum, so the cheapest
option alone does not say whether a cheaper dispatch exists. Every options
result (except the last resort, whose options reduce load) therefore carries a
**lower bound** on the cheapest option's cost, `optimality`: the minimum of a
convex relaxation of the cheapest option's own problem — the same units,
prices, limits, exchange cap and held units, loads not reduced — with only the
AC power-flow equations relaxed. Every AC dispatch is feasible in the
relaxation, so no dispatch meeting the limits costs less than the bound, and no
other dispatch saves more than the gap $C(x^{\text{cheapest}}) - \underline C$.

With $c_{ii} = V_i^2$, $c_{ij} = V_iV_j\cos(\theta_i-\theta_j)$ and
$s_{ij} = V_iV_j\sin(\theta_i-\theta_j)$, the power balances and the branch
current limits are linear:

$$
\begin{aligned}
P_{ij} &= G_{ff}\,c_{ii} + G_{ft}\,c_{ij} + B_{ft}\,s_{ij}, \qquad
Q_{ij} = -B_{ff}\,c_{ii} - B_{ft}\,c_{ij} + G_{ft}\,s_{ij}, \\
|I_{ij}|^2 &= |Y_{ff}|^2 c_{ii} + |Y_{ft}|^2 c_{jj}
  + 2\big[(G_{ff}G_{ft} + B_{ff}B_{ft})\,c_{ij} - (B_{ff}G_{ft} - G_{ff}B_{ft})\,s_{ij}\big],
\end{aligned}
$$

and the voltage bounds become $\underline V_i^2 \le c_{ii} \le \overline V_i^2$.
What is not convex is that these numbers come from one voltage vector: the
matrix $W$ with $W_{ii} = c_{ii}$ and $W_{ij} = c_{ij} + j\,s_{ij}$ is
$W = VV^{\mathsf H}$, of rank one. The relaxation keeps only
$W \succeq 0$ (Lavaei & Low 2012), imposed on the maximal cliques of a chordal
completion of the network — as strong as on the whole matrix, in small
blocks — and solved by Clarabel. The power-factor limit
$Q^2 \le \tan^2\!\varphi\,P^2$ is $|Q| \le \tan\varphi\,P$ exactly (units do not
run below 0 MW); every other constraint is linear.

The simpler second-order cone relaxation keeps only the 2×2 minors,
$c_{ij}^2 + s_{ij}^2 \le c_{ii}c_{jj}$, and drops that the angles around a loop
add up. On a congested meshed network that is loose: at the 30-bus demo's
evening peak it lets flow leave the overloaded line L0 for free, and bounds the
cheapest option (26.48 €/h) at about −150.5 €/h. The semidefinite relaxation bounds
it at 22.48 €/h: no other dispatch saves more than 4.00 €/h. With loads at
115 % the bound is 381.18 €/h against 381.21 €/h, and the cheapest option is
called the best possible (a gap of at most 0.05 €/h). Where the semidefinite
relaxation is unavailable (no cvxpy) or the network has more than 300 buses,
the cone bound is given, and `relaxation` says which.

A large gap says a cheaper dispatch *may* exist, not that one does. On the
demo, IPOPT started from the relaxation's solution returns to the same cheapest
option at 19:00, 20:00 and 21:00, where the gaps are 27, 4 and 27 €/h: the
relaxation, not the option, is loose there. The bound is rounded down to the
cent; a bound that came out above the cheapest option (which a relaxation
cannot do) is not reported, nor one whose solve did not reach an optimal
status.

## When redispatching is not enough

When no dispatch of the units meets the limits — proved by the relaxation
where it can be, and then said ("proved by a convex relaxation") — the study
falls back to the **last resort**: options that also reduce load. Both start from the power-flow
point and free all demand response and all load relief:

- **Least load reduction**, two stages: the fewest MW of reduction, a MW of
  relief counting as 100 MW of demand response,

$$
R^\star = \min\; \sum_d D^{dr}_d + 100 \sum_d D^{rel}_d + 10^{-4} D(P),
$$

  then the least change to the schedule keeping $R \le 1.001\, R^\star + 10^{-3}$;
- **Cheapest with load reduction**: the cost $C$ with every price, plus the
  small disturbance tie-break.

So demand response comes before relief by construction in the first, and by
price (150 against 3,000 €/MWh) in the second. Each load reduction keeps the
load's power factor. One that finds no dispatch is listed with the reason (not
even reducing load clears the violations, for example), and the second is
listed as the same dispatch when it is. The answer explains *why* redispatch
alone fails (a diagnosis of the infeasible problem — see
[The optimal power flow](optimal-power-flow.md#when-there-is-no-solution)), and
adds **switching that may help**: each in-service branch opened in turn on the
schedule, by power flow, shown when it relieves at least 10 % of the excess or
removes a violation (at most three).

## Carrying an option forward

Each set of options is kept, with its moment, network, data source, load level
and conditions, under a reference (`S3`; an option is `S3:cheapest`). Up to 20
sets are kept, on disk beside the network's proposals, so references survive a
restart. A later study can apply an option:

- **at its own moment**: its whole dispatch;
- **at a later moment, held**: only what it *instructed* stays (the active
  power, set-point or reactive output it changed); every other unit follows its
  own schedule there, as in operation. An earlier moment is refused;
- **as a starting point** (`from_option`): the held set-points become the new
  schedule, and the next options are priced from it.

An option computed under other unit models, names or load power factor is
refused, with the reason.

## Assumptions and limits

- Prices are linear and an action is priced as a rate held for an hour.
  Production cost curves enter only through the derived up and down prices.
- "Fewest units" and "without the main unit" are heuristics anchored on the
  least-disturbance solution, not exact combinatorial searches.
- Secondary objectives are small weighted tie-breaks, except where two stages
  are stated (least import, least load reduction). In most headroom the
  tie-break is not negligible: $s$ is a squared loading fraction, so near 90 %
  a 10 MW move ($10^{-4} \cdot 100 = 0.01$) weighs about as much as half a
  percentage point of loading.
- Duplicates compare active power, load reductions and which units are called:
  two options that call the same units with different set-points count as the
  same. Distances (the at-most-five rule) compare active power only.
- Every solve is local (IPOPT). The cheapest option's cost has a lower bound
  (above), and a study that finds no dispatch is checked by the relaxation:
  proved to have none, solved on a second start, or not proven
  ([The optimal power flow](optimal-power-flow.md#when-there-is-no-solution)).
  A strategy that finds no dispatch ("not offered") is not checked.
- The reactive band is computed from the schedule's exchange for the option too,
  and the reactive price applies to voltage-holding units' automatic reactive
  changes, which are not instructions.
- A PV or wind unit a proposal adds has no upward price, so raising it, where
  its maximum allows, costs 0 €/MWh in the cheapest option.

## In the code

| What | Where |
| --- | --- |
| Strategies, selection, near-optimal, cost expression | `backend/flex_options.py` |
| The cost table and its defaults | `backend/costs.py` |
| The lower bound on the cheapest option | `soc_relax`, `_sdp_problem`, `cost_lower_bound`, `optimality` in `backend/relaxation.py` |
| Building the options, prices, the power-flow check, switching, last resort | `_flexibility_options`, `_flex_prices`, `_power_flow_check`, `_switching_remedies`, `_with_last_resort` in `backend/main_backend.py` |
| Option sets and references | `_remember_options`, `_resolve_option`, `_option_context`, `_apply_option_dispatch` |
| Request | `optimize_flexibility` with `options`, `hold_units`, `branch_limits`, `from_option` |
