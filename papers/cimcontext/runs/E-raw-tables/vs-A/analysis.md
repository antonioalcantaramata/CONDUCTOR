# Ablation: raw_tables

What the layer adds over the data it is built from: the agent given the network's raw data tables instead of the layer (the studies read the layer in every arm).

Commit `5bf4d30474` (uncommitted changes) · 5 repeats · suites: semantics-cigre, semantics-ieee30, reasoning · order: interleaved

| Arm | Settings |
|---|---|
| tables | ontology=tables |

Baseline: **on**. Rates in %, 95 % Wilson interval; Δ is the arm minus the baseline over the tasks both ran, with a 95 % bootstrap interval over tasks, then tasks better ↑ / worse ↓ / same = and the sign test's p.

| Arm | Runs | Checks passed | Δ checks | Tasks fully passed | Δ tasks | Tool calls | Input tok | of which cached | Output tok | s | Ungrounded figures | Entity errors |
|---|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| on | 185 | 97.2 [95, 99] | – | 95.1 [91, 97] | – | 2.2 | 83,834 | 82,312 | 321 | 9 | 0.00 | 0.00 |
| tables | 185 | 83.7 [79, 87] | -16.6 [-27, -8]; 3↑ 15↓ 19= p=0.0075 | 72.4 [66, 78] | -22.7 [-36, -11]; 3↑ 15↓ 19= p=0.0075 | 2.2 | 89,087 | 85,415 | 598 | 23 | 0.01 | 0.02 |
| off | 185 | 64.9 [60, 70] | -32.3 [-46, -19]; 4↑ 21↓ 12= p=0.00091 | 54.1 [47, 61] | -41.1 [-56, -25]; 4↑ 21↓ 12= p=0.00091 | 2.9 | 95,268 | 92,795 | 732 | 15 | 0.02 | 0.04 |

## The reading check

Turns whose decisive studies were re-run under the readings the question left open, and how often another reading moved the figure by more than 10 % (shifts) or changed the conclusion (changes).

| Arm | Turns re-checked | Shifts | Changes |
|---|---:|---:|---:|
| on | 25 | 10 (40 %) | 5 (20 % [9, 39]) |
| tables | 28 | 9 (32 %) | 5 (18 % [8, 36]) |
| off | 38 | 13 (34 %) | 9 (24 % [13, 39]) |

## By suite

| Suite | on | tables | off |
|---|---:|---:|---:|
| reasoning | 259/265 | 222/265 | 165/265 |
| reasoning · composition | 48/50 | 39/50 | 37/50 |
| reasoning · consistency | 39/40 | 40/40 | 40/40 |
| reasoning · data | 37/40 | 20/40 | 22/40 |
| reasoning · intervention | 45/45 | 44/45 | 45/45 |
| reasoning · names | 50/50 | 47/50 | 12/50 |
| reasoning · switches | 40/40 | 32/40 | 9/40 |
| semantics-cigre | 27/30 | 30/30 | 23/30 |
| semantics-ieee30 | 30/30 | 20/30 | 23/30 |

## Checks where the arms differ

| Task | Check | on | tables | off |
|---|---|---:|---:|---:|
| reasoning/compose_battery | found Bus 11 lowest | 4/5 | 5/5 | 5/5 |
| reasoning/compose_battery | named the feeder Line 1-2 | 5/5 | 0/5 | 0/5 |
| reasoning/compose_battery | said yes, the same feeder | 5/5 | 5/5 | 4/5 |
| reasoning/compose_t0 | eight of them solar | 5/5 | 5/5 | 5/5 |
| reasoning/compose_t0 | the tie Line 14-8 restores it | 4/5 | 0/5 | 0/5 |
| reasoning/compose_tie | named the tie S1 (Line 14-8) | 5/5 | 4/5 | 3/5 |
| reasoning/data_cigre | Load R5's Q: derived from the power factor, not measured | 5/5 | 4/5 | 3/5 |
| reasoning/data_cigre | PV 9's output: a share of its bus's measurement | 5/5 | 0/5 | 0/5 |
| reasoning/data_cigre_battery | held at its set-point; the data carry no storage | 5/5 | 0/5 | 0/5 |
| reasoning/data_cigre_load_split | a share of Bus 1's measurement, split with Load CI1 | 5/5 | 1/5 | 0/5 |
| reasoning/data_cigre_wind | PV 9's output: a share of its bus's measurement | 5/5 | 4/5 | 4/5 |
| reasoning/data_cigre_wind | WKA 7's output: measured | 5/5 | 4/5 | 5/5 |
| reasoning/data_ieee30 | said it is derived from the power factor | 4/5 | 4/5 | 5/5 |
| reasoning/data_ieee30_solved | said the power flow solves it | 3/5 | 3/5 | 5/5 |
| reasoning/drop_cigre_declared | named CHP diesel 1 as affected (declared solar) | 5/5 | 4/5 | 5/5 |
| reasoning/drop_cigre_declared | the study scaled CHP diesel 1 too | 5/5 | 5/5 | 5/5 |
| reasoning/drop_cigre_stated | did not count CHP diesel 1 | 5/5 | 5/5 | 5/5 |
| reasoning/drop_cigre_stated | the study scaled the nine wind and PV units | 5/5 | 5/5 | 5/5 |
| reasoning/drop_ieee30_declared | named Gen_bus12 as affected | 5/5 | 5/5 | 5/5 |
| reasoning/drop_ieee30_declared | the study scaled Gen_bus12 only | 5/5 | 5/5 | 5/5 |
| reasoning/drop_ieee30_none | said no unit is stated wind or solar | 5/5 | 5/5 | 5/5 |
| reasoning/names_cigre | counted eight | 5/5 | 5/5 | 0/5 |
| reasoning/names_cigre | did not take 'PV Park East' (a fuel cell) for solar | 5/5 | 5/5 | 0/5 |
| reasoning/names_cigre | did not take 'Solar Farm North' (a CHP unit) for solar | 5/5 | 5/5 | 0/5 |
| reasoning/names_cigre_storage | did not take 'Battery Storage 3' (a fuel cell) for storage | 5/5 | 5/5 | 0/5 |
| reasoning/names_cigre_storage | named Battery 1 and Battery 2 | 5/5 | 5/5 | 0/5 |
| reasoning/names_cigre_wind | did not take 'Wind Park West' (a PV unit) for wind | 5/5 | 5/5 | 0/5 |
| reasoning/names_ieee30 | said 'Solar Plant A' is solar only by its name | 5/5 | 3/5 | 1/5 |
| reasoning/names_ieee30_wind | said 'Offshore Wind 1' is wind only by its name | 5/5 | 4/5 | 1/5 |
| reasoning/para_vary | named the wind and PV units | 4/5 | 5/5 | 5/5 |
| reasoning/switches_breakers | named Switch 6 and Switch 7 | 5/5 | 0/5 | 0/5 |
| reasoning/switches_breakers | said they switch the transformers | 5/5 | 5/5 | 4/5 |
| reasoning/switches_line148 | named its switch S1 | 5/5 | 5/5 | 0/5 |
| reasoning/switches_line148 | said it is normally open | 5/5 | 5/5 | 0/5 |
| reasoning/switches_loops | named S2 and S3 | 5/5 | 5/5 | 0/5 |
| reasoning/switches_open | named the three normally open switches | 5/5 | 5/5 | 0/5 |
| reasoning/switches_open | said S1 is the tie between feeders | 5/5 | 2/5 | 0/5 |
| semantics-cigre/cigre_levels | 20 kV, and Bus 0 at 110 kV | 4/5 | 5/5 | 5/5 |
| semantics-cigre/cigre_outage | named the tie Line 14-8 | 3/5 | 5/5 | 4/5 |
| semantics-cigre/cigre_outage | named what loses supply, as far as Bus 11 | 5/5 | 5/5 | 4/5 |
| semantics-cigre/cigre_storage | both batteries and their buses | 5/5 | 5/5 | 0/5 |
| semantics-ieee30/ieee30_customers | said the network does not say | 5/5 | 0/5 | 2/5 |
| semantics-ieee30/ieee30_level | 135 kV | 5/5 | 5/5 | 4/5 |
| semantics-ieee30/ieee30_radial | said it loses supply | 5/5 | 4/5 | 4/5 |
| semantics-ieee30/ieee30_renewables | said the network does not say | 5/5 | 1/5 | 3/5 |
