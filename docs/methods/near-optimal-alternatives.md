# Near-optimal alternatives

The cheapest corrective action is one point in a space of dispatches that cost
almost as much. An operator may prefer one of those for reasons the cost table
does not know — a unit they distrust, one that is hard to reach, a contract.
Near-optimal alternatives explore that space: **modelling to generate
alternatives** (MGA), in the sense of Brill and DeCarolis, applied to the AC
optimal power flow.

## The budget

The cheapest option $x^C$ is solved first (see
[Corrective options](corrective-options.md#cheapest)); its cost $C^\star$ sets the
budget, with a margin $\varepsilon$ (10 % by default) and a floor of 1 €/h:

$$
B = C^\star + \max\Big( |C^\star|\, \frac{\varepsilon}{100},\; 1\ \text{€/h} \Big)
$$

The floor matters: with the synthetic prices an action that only raises units
can cost about 0 €/h, and a margin of 10 % of nothing would leave no room. A
saving ($C^\star < 0$) keeps its margin on its magnitude: $B(-80, 10\,\%) = -72$.

When the cheapest problem has no feasible dispatch there is no budget: the
study returns the least-disturbance dispatch alone, with that reason.

Every alternative is solved on a copy of the model with the cost bounded:

$$
x \in \mathcal X, \qquad C(x) \le B
$$

## Four ways to fill the pool

| Method | What it solves on the bounded model |
| --- | --- |
| **Hop-skip-jump** | Three rounds of $\min \sum_g w_g\,(\delta^+_g + \delta^-_g) + 10^{-3} D(P)$: a weighted L1 movement where every unit's movement costs, and units already moved by the cheapest option or an earlier round cost ten times more ($w_g = 10$). Pushes the solution onto other units. Stops at the first infeasible round. |
| **Unit ranges** | For every free unit, $\min P_g$ and $\max P_g$ (each with a $10^{-4} D(P)$ tie-break): how far that unit can go within the budget. Both solutions join the pool, and the ranges are reported. |
| **Other objectives** | Fewest units, least import, most headroom and least disturbance, each within the budget. |
| **Random directions** | Four directions $d_g \sim \mathcal U(-1, 1)$ over the free units, $\min \sum_g d_g P_g + 10^{-4} D(P)$. Seeded (seed 7), so a study is reproducible. |

A **free unit** is one whose active power can move: not held by the operator,
and not fixed by its own bounds — a synchronous condenser ($0 \le P_g \le 0$) or
a unit marked not dispatchable ($P_g = P_g^0$) is not searched and gets no
range. Each solution remembers how it was found.

## Choosing diverse alternatives

From the pool, alternatives are picked **farthest-first**, by the L1 distance
between dispatches in active power:

$$
y^\star = \arg\max_{y \in \text{pool}} \; \min_{s \in S} \sum_g \big| P^y_g - P^s_g \big|
$$

starting from $S = \{x^C\}$, adding $y^\star$ to $S$ each time, until the requested
number (at most five) is reached or the farthest remaining candidate is less
than **1 MW** away from every chosen one: closer than that, it adds nothing an
operator would act on. A candidate that is the same dispatch as the cheapest
(see [duplicates](corrective-options.md#selecting-what-is-shown)) never enters.
When fewer alternatives are found than asked, the answer says why: every
remaining candidate is within 1 MW of one shown, or the methods run found no
further dispatch within the margin.

The answer shows the cheapest and the alternatives in the order picked, each
with its cost above the cheapest, how it was found, and which units it avoids or
adds compared with the cheapest; and, from the unit-range method, each unit's
range within the budget (its minimum, maximum, value in the cheapest option and
schedule).

## Assumptions and limits

- All solves are local (IPOPT): the pool samples the near-optimal region, it
  does not map it exhaustively.
- Distances and diversity use active power only; set-points are not compared.
- Loads are not reduced in this mode.

## In the code

| What | Where |
| --- | --- |
| Budget, bounded model, pool methods, farthest-first selection | `near_optimal_budget`, `_bounded`, `generate_near_optimal` in `backend/flex_options.py` |
| Request | `optimize_flexibility` with `options` and `near_optimal`, `near_optimal_pct`, `near_optimal_methods`, `near_optimal_count` |
