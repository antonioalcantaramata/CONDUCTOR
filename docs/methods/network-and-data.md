# Network, data and the operating point

Every study runs on one network at one moment. This page describes how a
network file and a time series become that operating point, what the bundled
networks and data are, and how proposals change a network without touching it.

## From a file to a network

**Formats.** MATPOWER (`.m`, converted by pandapower; PGLib cases included),
pandapower JSON or Excel, and UCTE. A MATPOWER generator at a load bus is turned
back into a voltage-controlled unit at 1.0 p.u.

**Units.** On loading, each generator not at the external grid becomes a
controllable **static generator** at unity power factor, keeping what it needs to
return to voltage control (its set-point and reactive limits). A unit below 1 MW
is a **synchronous condenser** and stays voltage-controlled. A bundled profile may
then set each unit's model (see [The System tab](../guide/system-tab.md)):

| Control | Behaviour |
| --- | --- |
| voltage | holds a voltage set-point; its reactive power follows from the power flow |
| power factor | reactive power = ±P·tan(arccos pf), producing or absorbing |
| fixed Q | a constant reactive output |
| dispatchable or not | whether the optimisation may move its active power |

**Limits.** One voltage band for every bus and one loading limit for every
branch: the profile's, or for an upload the widest band over the case's per-bus
limits and 100 % loading. These are "the network's own limits", which every study
uses unless the question gives others.

**Admittances.** The branch admittances the optimisation uses are built once
from pandapower's internal model (see
[The optimal power flow](optimal-power-flow.md#the-admittances)).

## The bundled networks

| | IEEE 30 (default) | IEEE 14 |
| --- | --- | --- |
| Source | PGLib-OPF `pglib_opf_case30_as` (Alsac–Stott) | PGLib-OPF `pglib_opf_case14_ieee` |
| Buses / lines / transformers / loads | 30 / 41 / 0 / 21 | 14 / 17 / 3 / 11 |
| Units | five, all voltage-controlled (Gen_bus1 and Gen_bus12 at 1.025 p.u., the others at 1.0) | Gen_bus1 at unity power factor; three synchronous condensers |
| External grid | Bus_0 at 1.0 p.u. | Bus_0 at 1.02 p.u. |
| Case load | 283.4 MW, 126.2 Mvar | 259.0 MW, 73.5 Mvar |
| Limits | 0.95–1.05 p.u., 100 % | 0.94–1.06 p.u., 100 % |
| Synthetic week | residential load, flat generation | two-peak load, wind |

Ratings are the cases' own (PGLib `rateA`). Element names use 0-based bus
indices: `Gen_bus1` sits at MATPOWER bus 2.

### Naming an element

Every study finds a bus, line, transformer or unit with the same lookup
(`backend/resolve.py`), so a name that works in one works in all. It accepts:

- the index, the name, the name as loaded, and the label results show (`Bus_0 ->
  Bus_1 [L0]`), with or without its tag;
- the tag alone (`L12`, `T1`), and `line 12` or `transformer 1`;
- a branch by its two ends, in either order and joined by any arrow, dash, *to*
  or *and* (`Bus_8 - Bus_9`, *between Bus_8 and Bus_9*), with or without a
  kind word before it (*line …*);
- a unit by its name, `Gen_bus7`, or its bus when it is the only unit there.

Case, spaces, underscores and dashes do not count (`bus7`, `Bus 7`, `BUS_7`
are one bus).

It never guesses. When several elements match, it lists them. When none does,
it offers the closest names. Asked for the external grid, it says that this is
the connection at a bus, and which branches reach it.

A moment between two of the data's (*23:59* on 15-minute data) is the one
that holds it: a window's start takes the moment at or after it, anything else
the moment at or before it, and the result says so.

## The time series

Each network comes with a week of **measurements** and a **forecast**, either
uploaded or synthetic. Both are tables of production and consumption per
substation (a bus with load or generation), at a regular interval.

### Synthetic data

Seven days at 15 minutes (672 moments from Saturday 1 January 2000), generated
with fixed seeds (42 for measurements, 137 for the forecast), so every run
sees the same week.

**Load.** Every load follows one shape $s(h, d)$ of the hour $h$ and day $d$,
with noise $u_t \sim \mathcal U(1 - \nu, 1 + \nu)$:

$$
\alpha_L(t) = \operatorname{clip}\big(s(h_t, d_t)\, u_t,\; 0.05,\; 1.5\big) \cdot \frac{k_{\text{peak}}}{\max_h s(h, \text{weekday})}
$$

so the weekday peak is $k_{\text{peak}}$ times the case's load (1.05 on the bundled
networks). The **residential** shape, with
$\beta(h; c, w) = \exp\!\big(-\tfrac12 ((h - c)/w)^2\big)$ on the 24-hour circle:

$$
\begin{aligned}
s(h, d) ={}& \big(0.55 + 0.25\,\beta(h; 8, 1.6) + 0.15\,\beta(h; 13, 3) + 0.42\,\beta(h; 19, 2)\big) \\
&\times \big(1 - 0.15 \cdot \mathbb 1[\text{weekend}]\big)
\end{aligned}
$$

— a morning peak, a midday shoulder and an evening peak — with $\nu = 0.05$. A
two-peak, an industrial and a flat shape are also available.

**Generation.** One factor $\alpha_G(t)$ for every unit: **flat**,
$0.6\,\mathcal U(0.98, 1.02)$; **wind**, a random walk from
$0.4 \times \text{Weibull}(2)$ with steps $\mathcal N(0, 0.025^2)$, clipped to
[0, 1]; or **solar**, a bell around noon (σ = 3 h) between 05:00 and 20:00.

**Stress events.** On about one day in three (each day with probability 0.35),
a two-hour window starting between 04:00 and 21:15 where load rises by 15 % (to
at most 1.6 times the case) and generation falls 3 %: moments likely to stress
the grid, though a violation is not guaranteed.

**Forecast.** For the last two days of measurements, the measured values times
$(1 + e_t)$, with an AR(1) error common to all substations,
$e_t = 0.97\, e_{t-1} + \sigma \sqrt{1 - 0.97^2}\, \eta_t$ (σ = 4 % for load, 20 % for
generation); then seven days ahead, generated independently. Measurements and
forecast share the overlapping timestamps.

On the default demo the week breaks a limit at 31 of its 672 moments
(undervoltage at Bus_29, overload of line L0), the worst at 19:15 on 6 January
(L0 at 134 %, 221.7 MW imported).

**Uploaded data** (see [Data and the operating point](../guide/data-and-operating-point.md)):
production and consumption in MW per substation and moment, optionally reactive
power. Missing active values are interpolated in time; missing reactive values
stay missing, never zero. Quality findings — coverage, implausible power factors,
reactive values that look like kvar — are reported, not corrected.

## One moment: the operating point

For every study, a fresh copy of the network (the base, or a proposal) receives
the moment's data:

1. **Loads.** Each load is matched to a substation of the data by name (exact,
   then by prefix, then by a three-letter prefix shared with exactly one
   substation). A substation's consumption is split over its loads in proportion
   to their case values:

$$
P_l = P_s \frac{\max(P^{\text{case}}_l, 0)}{\sum_{l' \in s} \max(P^{\text{case}}_{l'}, 0)},
\qquad
Q_l = P_l \tan(\arccos \text{pf})
$$

   with the **load power factor** (0.95 lagging by default; settable for the session
   or for one study). Where the data carry measured reactive consumption, it
   replaces the power-factor value, split the same way.
2. **Units.** A substation's production is matched to its units by exact name and
   split by their capacity. A static generator keeps its power factor; a
   voltage-controlled unit gets its active power, and its reactive power follows
   from the power flow. Measured reactive production goes to static generators
   only. A network that brings its own units (any voltage-controlled or static
   generator) keeps them at every moment; only a network with no unit at all is
   treated as a set of measured substations, its units rebuilt from the loads
   and the data at each moment.
3. **Proposals.** A proposal's own elements follow their profiles (below).
4. **Scaling.** Loads are multiplied by the study's load scaling (active and
   reactive together, so the power factor holds); static generators by the
   generation scaling, where a study takes one.
5. **A corrective option**, when the study applies one (see
   [Corrective options](corrective-options.md#carrying-an-option-forward)).

The moment is the **clock** (measurements), the first forecast moment
(forecasts), an explicit timestamp (matched by a unique prefix), or the network's
**nominal point** — the case's own loads and units, with no time and no load power
factor (the case's own reactive power).

A day or an hour given as one moment (`2000-01-06`, `2000-01-06 20`) matches
several ticks; a study of one moment then takes its **first moment** (00:00,
20:00), since that is how an hour is named and a tick's timestamp starts its
interval. The result says so in `timestamp_resolved` (what was asked, what was
used, why). Any other prefix matching several ticks is refused as ambiguous.
A study over a window keeps a prefix's meaning: `2000-01-06 20` as a bound
covers the whole hour.

## Proposals

A proposal is the base network plus an ordered list of typed edits — never a
stored copy. Edits: add a load (flat, following a bus, or on a schedule), a
generator (dispatchable, PV or wind, with its control), a shunt, a bus with a
line or a transformer; reinforce a line; upgrade a transformer; switch an element
in or out of service; set a parameter; move or scale an element; include another
proposal.

Each edit is checked against the network as the previous edits left it (sizes
within bounds, a new voltage-controlled unit on a held bus holding the same
voltage…); a default filled in is reported as an assumption, and a missing
essential becomes a question for the operator. A proposal's added elements follow
their profiles at each moment: a load at its size times its schedule, a PV or wind
unit at its capacity times the measured production's shape. Every study takes a
`network` argument, and comparisons run the same study on the base and the
proposal (see [Risk, uncertainty and capacity](risk-and-capacity.md#comparing-a-proposal-with-the-base)).

## Assumptions and limits

- **Perfectly correlated synthetic data**: one load shape for every load and one
  generation factor for every unit, whatever its type. Fine for exercising the
  studies, not a model of real diversity.
- **Constant-power loads**; one voltage band and one loading limit for the whole
  network.
- **Unit reactive limits are not enforced** in power flows: on the default week a
  unit is outside its reactive range at 59 of 672 moments. The optimisation keeps
  them, and the power-flow check of an option flags them.
- **On the default IEEE 30 demo every unit is voltage-controlled**, so studies
  that scale or perturb static generators (generation scaling, renewable scenarios,
  generation uncertainty) change nothing there, and measured reactive production
  does not reach the power flow.
- **Generation is matched by exact name**; loads by name, prefix or a unique
  three-letter prefix.
- An uploaded network with no unit at all is read as measured substations: its
  units are rebuilt one per load at each moment, from the data's production, and
  cannot be given a model in the System tab. Check the data quality report and
  the assessment of the first moment after such an upload.

## In the code

| What | Where |
| --- | --- |
| Loading, conversion, admittances, profiles | `backend/network_loader.py`, `backend/network_profiles/*.yaml` |
| Unit models and names | `backend/generators.py` |
| Finding an element by its name | `find`, `find_unit` in `backend/resolve.py`; `GET /api/network/resolve` |
| Data to loads and units | `assign_load_values_from_measurements`, `assign_generators_values_from_measurements` in `backend/load_gen_assignment.py` |
| Synthetic measurements and forecast | `generate`, `forecast` in `backend/synthetic_timeseries.py` |
| Data quality, the nominal point | `backend/data_quality.py`, `backend/nominal.py` |
| One moment | `_assign_tick`, `_prepared_net`, `_resolve_tick` in `backend/main_backend.py` |
| Proposals | `backend/sandbox.py`, `POST /api/proposals/edit`, `POST /api/proposals/compare` |
