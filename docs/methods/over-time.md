# Studies over time

A corrective action is decided at one moment and lives through the next ones.
Three studies follow it over time: **holding** an option over the next hours,
**one plan** for a window instead of one action per moment, and the **cost of
N-1 security** over a window. The [key performance indicators](key-performance-indicators.md#over-a-window)
can be studied over a window too.

## Choosing the moments

A window runs from a start (by default the clock, or the first forecast moment)
to an end, or for a number of hours (at most 48), on measurements or the
forecast — never the nominal point, which has no time. Within it, at most a
given number of moments (2–12, six by default) are studied, evenly spaced, the
**first and last always kept**. A round step is used when one lands on the last
moment and keeps at least half the moments asked for; otherwise the moments are
placed as evenly as the series allows. With 15-minute data, 16:00 to midnight at
most six gives 16:00, 18:00, 20:00, 22:00 and 00:00; hour by hour is nine moments.

### Totals over a window

A total weights each moment by the time it stands for — half the gap to each
neighbour (the trapezoid rule):

$$
\begin{aligned}
w_k &= \tfrac12 \big(t_k - t_{k-1}\big) + \tfrac12 \big(t_{k+1} - t_k\big) \\
w_0 &= \tfrac12 (t_1 - t_0), \qquad w_{T-1} = \tfrac12 (t_{T-1} - t_{T-2})
\end{aligned}
$$

so the weights sum to the window's length, $t_{T-1} - t_0$: 16:00 to midnight
counts eight hours, not nine. A total is an estimate between the moments
studied; the moments in between are not simulated.

### Moments in parallel

The window studies that solve each moment fully (the cost of security and the
indicators) run their moments in **parallel worker processes** where the
platform allows: each worker is a fork of the backend, with the network, data
and settings as they are, and runs one moment's study in a single thread (the
linear algebra of the solver is limited to one thread per worker, which on these
networks is faster than sharing threads). At most four workers, and one fewer
than the processor cores; `CONDUCTOR_STUDY_WORKERS` caps it, and 1 runs the
moments in turn — as on Windows, where processes cannot be forked. Options a
worker computes are adopted under fresh references. While a study runs, it
reports how many moments are done (`GET /api/progress/{id}`), and the app shows
it in the status line.

## Holding an option

*Hold option S3:cheapest for the next four hours: until when does it keep every
limit?* From the option's own moment, over the next hours (at most 49 moments,
13 with the N-1 screen), each moment is assessed twice by power flow: the
schedule untouched, and the schedule with the option applied.

- At the option's own moment, its whole dispatch applies.
- At every later moment it is **held**: only what it *instructed* stays — the
  absolute active power, set-point or reactive output it gave a unit — and every
  other unit follows its own schedule at that moment, as in operation. Load
  reductions stay as the megawatts cut.

The answer gives the last moment it keeps every limit (`holds_until`), the first
it does not, and suggests the next corrective action from it there
(`from_option`). Optionally, the N-1 screen of the held dispatch at each moment.

## One plan for a window

*Plan the evening peak as one.* Solving each moment on its own may move a unit
up at 18:00, down at 19:00 and up again at 20:00. The plan solves the window as
**one** optimisation, every moment within its limits, and keeps instructions
from one moment to the next unless changing them is worth it.

**The model.** One AC OPF block per moment $t$ (see
[The optimal power flow](optimal-power-flow.md)), each with its own network,
loads, limits, held units, branch caps and the exchange cap. The current margin
is the OPF's 90 %, or the request's loading limit when that is stricter:
$\mu = \min(0.9,\ \overline\ell / 100)$, with $\overline\ell$ the lower of its
line and transformer limits (the network's own unless given). The blocks are
coupled only through the units present in every block, $\mathcal U$, by the
change of each unit's deviation from its own schedule:

$$
\delta_{g,t} = P_{g,t} - P^0_{g,t}, \qquad \delta_{g,-1} = 0,
\qquad c_{g,t} \ge \big|\delta_{g,t} - \delta_{g,t-1}\big|
$$

(written as two linear inequalities). The objective:

$$
\min\; \sum_{t} \omega_t\, J_t \;+\; w \,\sigma \sum_{t} \sum_{g \in \mathcal U} c_{g,t},
\qquad \omega_t = \frac{w_t}{\bar w}
$$

where $w_t$ is the time moment $t$ stands for (the weights of
[totals over a window](#totals-over-a-window)) and $\bar w$ their mean: each
moment counts in proportion to its share of the window, as in the reported
totals, and the steadiness weight keeps its scale. $\sigma$ is the
**steadiness** (0–10, default 1), and

- with `objective = least_disturbance` (default), $J_t$ is the OPF's own
  objective at moment $t$ and $w = 0.15$;
- with `objective = cheapest`, $J_t$ is the cost of the action at $t$ (see
  [Corrective options](corrective-options.md#the-cost-of-an-action)) plus
  $10^{-3} D(P_t)$, and $w = 6$.

The first change from the schedule counts too. An L1 term makes keeping an
instruction exactly free and changing it cost in proportion to the change, so
small changes are not issued. With steadiness 0 the moments decouple.

**Ramp.** Optionally, a ramp limit $\rho$ in percent of capacity per hour
between consecutive moments:

$$
\big|P_{g,t} - P_{g,t-1}\big| \le \frac{\rho}{100}\, \max_{t'} \overline P_{g,t'}\, \Delta_{t-1}
$$

with $\Delta_{t-1}$ the hours between the two moments and
$\max_{t'} \overline P_{g,t'}$ the unit's largest capacity over the window (a
capacity that follows a resource changes from moment to moment).

**What is reported.** The plan is solved twice — linked, and every moment on
its own — and both are described: per moment, each unit's change, the units
whose instruction changed (by at least 0.01 MW), the cost, the highest loading
and lowest voltage, and an AC power-flow check of the dispatch; over the window,
the **instructions issued** (every change of a unit's active power counts, the
first from the schedule included), the MWh moved and the cost (both weighted as
above). At the demo's evening peak (17:00–22:00, six moments): 17 instructions
instead of 25, and 14 at steadiness 4.

## The cost of N-1 security over a window

At each moment, the same study a single question gets: the N-1 screen of the
schedule, and the corrective options with the N-1 secure ones (`n1`:
preventive, corrective or both; see [N-1 security](n1-security.md)). The
security cost at a moment is the secure option's cost minus the cheapest
action's:

$$
c^{\text{sec}}_k = c^{\text{N-1}}_k - c^{\text{cheapest}}_k \;\;[\text{€/h}],
\qquad C^{\text{sec}} = \sum_k w_k\, c^{\text{sec}}_k \;\;[\text{€}]
$$

Each moment also reports the schedule's N-1 screen, each option's full screen
(outages still with violations, of all those tested) and, apart, the outages the
optimisation studied and how many it secured. Where the schedule itself breaks a
normal-operation limit (the current margin included), it is only a reference:
the comparison is with the cheapest action.

On the demo, from 16:00 to midnight on 6 January, hour by hour, preventive
security costs about 6,400 € above the cheapest and corrective about 3,500 €;
the nine moments take about 20 seconds with four workers.

## Assumptions and limits

- **Between moments** nothing is simulated: totals interpolate, a plan is
  checked at its moments only, and a held option may break a limit between two
  moments that both hold.
- **The plan's objective** weighs each moment's term by the time it stands for,
  as the reported totals do; the steadiness term counts each change of an
  instruction once, whatever the spacing.
- **"Keeping an instruction"** means a fixed deviation from each moment's
  schedule in the plan, and a fixed absolute set-point when holding an option.
  Both are reasonable readings; they are not the same.
- **The ramp** limits absolute output, using each unit's largest capacity over
  the window, and also applies to units held at a schedule that itself ramps.
- **The plan's power-flow check** judges voltages against the optimisation's
  band (`opf_vm_lower`, `opf_vm_upper`; the network's own unless given), not a
  separate assessment band.
- **The security cost** is that of the least-disturbance secure dispatch (see
  [N-1 security](n1-security.md#the-objective-as-secure-as-possible-then-the-least-change)),
  an upper estimate of the least cost of security. A moment that cannot be priced
  is left out of the total, and the total says so: the hours it covers
  (`hours_priced`) and a note that it is not the whole window's. It then
  understates.

## In the code

| What | Where |
| --- | --- |
| Moments of a window, weights, gaps | `_window`, `_ticks_within`, `_weights`, `_gaps` in `backend/main_backend.py` |
| Parallel moments and progress | `_run_moments`, `_study_workers`, `GET /api/progress/{id}` |
| Holding an option | `POST /api/flexibility/hold` |
| One plan | `POST /api/flexibility/plan` |
| The cost of security over a window | `POST /api/flexibility/security_over_time` (`_security_moment`) |
| Tool | `flexibility_over_time` (`study` = `hold`, `plan`, `security`) |
