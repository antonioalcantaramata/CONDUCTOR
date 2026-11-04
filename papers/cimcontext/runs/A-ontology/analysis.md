# Ablation: ontology

What the semantic layer is worth — to the agent, and to the studies.

Commit `5bdfb2eb1b` · 5 repeats · suites: semantics-cigre, semantics-ieee30, reasoning · order: interleaved

| Arm | Settings |
|---|---|
| on | ontology=on |
| off | ontology=off |
| none | ontology=none |

Baseline: **on**. Rates in %, 95 % Wilson interval; Δ is the arm minus the baseline over the tasks both ran, with a 95 % bootstrap interval over tasks, then tasks better ↑ / worse ↓ / same = and the sign test's p.

| Arm | Runs | Checks passed | Δ checks | Tasks fully passed | Δ tasks | Tool calls | Input tok | of which cached | Output tok | s | Ungrounded figures | Entity errors |
|---|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| on | 185 | 97.2 [95, 99] | – | 95.1 [91, 97] | – | 2.2 | 83,834 | 82,312 | 321 | 9 | 0.02 | 0.03 |
| off | 185 | 64.9 [60, 70] | -32.3 [-46, -19]; 4↑ 21↓ 12= p=0.00091 | 54.1 [47, 61] | -41.1 [-56, -25]; 4↑ 21↓ 12= p=0.00091 | 2.9 | 95,268 | 92,795 | 732 | 15 | 0.03 | 0.04 |
| none | 185 | 54.8 [49, 60] | -41.2 [-55, -28]; 4↑ 24↓ 9= p=0.00018 | 42.2 [35, 49] | -53.0 [-69, -37]; 4↑ 24↓ 9= p=0.00018 | 2.9 | 92,034 | 90,307 | 772 | 15 | 0.04 | 0.05 |

## The reading check

Turns whose decisive studies were re-run under the readings the question left open, and how often another reading moved the figure by more than 10 % (shifts) or changed the conclusion (changes).

| Arm | Turns re-checked | Shifts | Changes |
|---|---:|---:|---:|
| on | 25 | 10 (40 %) | 5 (20 % [9, 39]) |
| off | 38 | 13 (34 %) | 9 (24 % [13, 39]) |
| none | 33 | 11 (33 %) | 7 (21 % [11, 38]) |

## By suite

| Suite | on | off | none |
|---|---:|---:|---:|
| reasoning | 259/265 | 165/265 | 134/265 |
| reasoning · composition | 48/50 | 37/50 | 34/50 |
| reasoning · consistency | 39/40 | 40/40 | 40/40 |
| reasoning · data | 37/40 | 22/40 | 22/40 |
| reasoning · intervention | 45/45 | 45/45 | 19/45 |
| reasoning · names | 50/50 | 12/50 | 10/50 |
| reasoning · switches | 40/40 | 9/40 | 9/40 |
| semantics-cigre | 27/30 | 23/30 | 22/30 |
| semantics-ieee30 | 30/30 | 23/30 | 22/30 |

## Checks where the arms differ

| Task | Check | on | off | none |
|---|---|---:|---:|---:|
| reasoning/compose_battery | found Bus 11 lowest | 4/5 | 5/5 | 4/5 |
| reasoning/compose_battery | named the feeder Line 1-2 | 5/5 | 0/5 | 0/5 |
| reasoning/compose_battery | said yes, the same feeder | 5/5 | 4/5 | 5/5 |
| reasoning/compose_t0 | eight of them solar | 5/5 | 5/5 | 4/5 |
| reasoning/compose_t0 | the tie Line 14-8 restores it | 4/5 | 0/5 | 0/5 |
| reasoning/compose_tie | named the tie S1 (Line 14-8) | 5/5 | 3/5 | 1/5 |
| reasoning/data_cigre | Load R5's Q: derived from the power factor, not measured | 5/5 | 3/5 | 1/5 |
| reasoning/data_cigre | PV 9's output: a share of its bus's measurement | 5/5 | 0/5 | 0/5 |
| reasoning/data_cigre_battery | held at its set-point; the data carry no storage | 5/5 | 0/5 | 1/5 |
| reasoning/data_cigre_load_split | a share of Bus 1's measurement, split with Load CI1 | 5/5 | 0/5 | 0/5 |
| reasoning/data_cigre_wind | PV 9's output: a share of its bus's measurement | 5/5 | 4/5 | 5/5 |
| reasoning/data_ieee30 | said it is derived from the power factor | 4/5 | 5/5 | 5/5 |
| reasoning/data_ieee30_solved | said the power flow solves it | 3/5 | 5/5 | 5/5 |
| reasoning/drop_cigre_declared | named CHP diesel 1 as affected (declared solar) | 5/5 | 5/5 | 1/5 |
| reasoning/drop_cigre_declared | the study scaled CHP diesel 1 too | 5/5 | 5/5 | 0/5 |
| reasoning/drop_cigre_stated | did not count CHP diesel 1 | 5/5 | 5/5 | 2/5 |
| reasoning/drop_cigre_stated | the study scaled the nine wind and PV units | 5/5 | 5/5 | 0/5 |
| reasoning/drop_ieee30_declared | named Gen_bus12 as affected | 5/5 | 5/5 | 4/5 |
| reasoning/drop_ieee30_declared | the study scaled Gen_bus12 only | 5/5 | 5/5 | 0/5 |
| reasoning/drop_ieee30_none | said no unit is stated wind or solar | 5/5 | 5/5 | 2/5 |
| reasoning/names_cigre | counted eight | 5/5 | 0/5 | 0/5 |
| reasoning/names_cigre | did not take 'PV Park East' (a fuel cell) for solar | 5/5 | 0/5 | 0/5 |
| reasoning/names_cigre | did not take 'Solar Farm North' (a CHP unit) for solar | 5/5 | 0/5 | 0/5 |
| reasoning/names_cigre_storage | did not take 'Battery Storage 3' (a fuel cell) for storage | 5/5 | 0/5 | 0/5 |
| reasoning/names_cigre_storage | named Battery 1 and Battery 2 | 5/5 | 0/5 | 0/5 |
| reasoning/names_cigre_wind | did not take 'Wind Park West' (a PV unit) for wind | 5/5 | 0/5 | 0/5 |
| reasoning/names_ieee30 | said 'Solar Plant A' is solar only by its name | 5/5 | 1/5 | 0/5 |
| reasoning/names_ieee30_wind | said 'Offshore Wind 1' is wind only by its name | 5/5 | 1/5 | 0/5 |
| reasoning/para_vary | named the wind and PV units | 4/5 | 5/5 | 5/5 |
| reasoning/switches_breakers | named Switch 6 and Switch 7 | 5/5 | 0/5 | 0/5 |
| reasoning/switches_breakers | said they switch the transformers | 5/5 | 4/5 | 4/5 |
| reasoning/switches_line148 | named its switch S1 | 5/5 | 0/5 | 0/5 |
| reasoning/switches_line148 | said it is normally open | 5/5 | 0/5 | 0/5 |
| reasoning/switches_loops | named S2 and S3 | 5/5 | 0/5 | 0/5 |
| reasoning/switches_open | named the three normally open switches | 5/5 | 0/5 | 0/5 |
| reasoning/switches_open | said S1 is the tie between feeders | 5/5 | 0/5 | 0/5 |
| semantics-cigre/cigre_levels | 20 kV, and Bus 0 at 110 kV | 4/5 | 5/5 | 5/5 |
| semantics-cigre/cigre_outage | named the tie Line 14-8 | 3/5 | 4/5 | 5/5 |
| semantics-cigre/cigre_outage | named what loses supply, as far as Bus 11 | 5/5 | 4/5 | 2/5 |
| semantics-cigre/cigre_storage | both batteries and their buses | 5/5 | 0/5 | 0/5 |
| semantics-ieee30/ieee30_customers | said the network does not say | 5/5 | 2/5 | 1/5 |
| semantics-ieee30/ieee30_level | 135 kV | 5/5 | 4/5 | 4/5 |
| semantics-ieee30/ieee30_radial | said it loses supply | 5/5 | 4/5 | 5/5 |
| semantics-ieee30/ieee30_renewables | said the network does not say | 5/5 | 3/5 | 2/5 |

## What the agent gains: the layer for the agent (both arms' studies use it) — on vs off

### Looking up what elements are — cigre

| Task | Check | Layer on | Agent without it |
|---|---|---:|---:|
| cigre_feeder | feeder Line 1-2 and the route through Line 3-8 to Line 10-11 | 5/5 | 5/5 |
| cigre_levels | 20 kV, and Bus 0 at 110 kV | 4/5 | 5/5 |
| cigre_outage | named what loses supply, as far as Bus 11 | 5/5 | 4/5 |
| cigre_outage | named the tie Line 14-8 | 3/5 | 4/5 |
| cigre_renewables | named the wind turbine and PV units | 5/5 | 5/5 |
| cigre_storage | both batteries and their buses | 5/5 | 0/5 |

| | Layer on | Agent without it |
|---|---:|---:|
| Checks passed | 27/30 | 23/30 |
| Answers contradicting the network (entity) | 0 | 1 |
| Claims the network does not support (entity) | 0 | 0 |
| Tool calls per run | 4.0 | 8.1 |
| Seconds per run | 11.5 | 26.4 |

### Looking up what elements are — ieee30

| Task | Check | Layer on | Agent without it |
|---|---|---:|---:|
| ieee30_customers | said the network does not say | 5/5 | 2/5 |
| ieee30_external | L0 and L1 | 5/5 | 5/5 |
| ieee30_level | 135 kV | 5/5 | 4/5 |
| ieee30_radial | named what is cut off | 5/5 | 5/5 |
| ieee30_radial | said it loses supply | 5/5 | 4/5 |
| ieee30_renewables | said the network does not say | 5/5 | 3/5 |

| | Layer on | Agent without it |
|---|---:|---:|
| Checks passed | 30/30 | 23/30 |
| Answers contradicting the network (entity) | 0 | 0 |
| Claims the network does not support (entity) | 0 | 0 |
| Tool calls per run | 1.4 | 1.8 |
| Seconds per run | 6.4 | 12.2 |

### Reasoning

## Per task (runs with every check passed; checks passed)

| Family | Task | Layer on | Agent without it |
|---|---|---:|---:|
| names | names_cigre | 5/5 (20/20) | 0/5 (5/20) |
| names | names_ieee30 | 5/5 (5/5) | 1/5 (1/5) |
| names | names_cigre_wind | 5/5 (10/10) | 0/5 (5/10) |
| names | names_cigre_storage | 5/5 (10/10) | 0/5 (0/10) |
| names | names_ieee30_wind | 5/5 (5/5) | 1/5 (1/5) |
| intervention | drop_ieee30_none | 5/5 (10/10) | 5/5 (10/10) |
| intervention | drop_ieee30_declared | 5/5 (10/10) | 5/5 (10/10) |
| intervention | drop_cigre_stated | 5/5 (15/15) | 5/5 (15/15) |
| intervention | drop_cigre_declared | 5/5 (10/10) | 5/5 (10/10) |
| composition | compose_t0 | 4/5 (14/15) | 0/5 (10/15) |
| composition | compose_battery | 4/5 (14/15) | 0/5 (9/15) |
| composition | compose_l33 | 5/5 (10/10) | 5/5 (10/10) |
| composition | compose_tie | 5/5 (10/10) | 3/5 (8/10) |
| switches | switches_open | 5/5 (10/10) | 0/5 (0/10) |
| switches | switches_loops | 5/5 (10/10) | 0/5 (5/10) |
| switches | switches_breakers | 5/5 (10/10) | 0/5 (4/10) |
| switches | switches_line148 | 5/5 (10/10) | 0/5 (0/10) |
| data | data_cigre | 5/5 (10/10) | 0/5 (3/10) |
| data | data_ieee30 | 4/5 (4/5) | 5/5 (5/5) |
| data | data_cigre_wind | 5/5 (10/10) | 4/5 (9/10) |
| data | data_cigre_battery | 5/5 (5/5) | 0/5 (0/5) |
| data | data_cigre_load_split | 5/5 (5/5) | 0/5 (0/5) |
| data | data_ieee30_solved | 3/5 (3/5) | 5/5 (5/5) |
| consistency | para_weather | 5/5 (10/10) | 5/5 (10/10) |
| consistency | para_vary | 4/5 (9/10) | 5/5 (10/10) |
| consistency | para_variable | 5/5 (10/10) | 5/5 (10/10) |
| consistency | persist_ieee30 | 5/5 (10/10) | 5/5 (10/10) |

## Per family

| Family | Layer on | Agent without it |
|---|---:|---:|
| names | runs 25/25, checks 50/50 | runs 2/25, checks 12/50 |
| intervention | runs 20/20, checks 45/45 | runs 20/20, checks 45/45 |
| composition | runs 18/20, checks 48/50 | runs 8/20, checks 37/50 |
| switches | runs 20/20, checks 40/40 | runs 0/20, checks 9/40 |
| data | runs 27/30, checks 37/40 | runs 14/30, checks 22/40 |
| consistency | runs 19/20, checks 39/40 | runs 20/20, checks 40/40 |

## Sensitivity to an intervention (both sides of the pair right, same repeat)

| Pair | Layer on | Agent without it |
|---|---:|---:|
| drop_ieee30_none → drop_ieee30_declared | 5/5 | 5/5 |
| drop_cigre_stated → drop_cigre_declared | 5/5 | 5/5 |

## Consistency across wordings (same repeat)

| Group | Measure | Layer on | Agent without it |
|---|---|---:|---:|
| renewables | all wordings right | 4/5 | 5/5 |
| renewables | wordings agree | 4/5 | 5/5 |

## Process

| | Layer on | Agent without it |
|---|---:|---:|
| Tool calls per run | 2.1 | 2.2 |
| Seconds per run | 9.5 | 13.6 |
| Entity-check findings (total) | 5 | 7 |
| A fact the operator stated, recorded for later studies | 5/5 | 0/5 |
| Runs that used the layer's tools | 125/135 | 0/135 |

## What grounding the studies gains (neither arm's agent sees the layer) — off vs none

### Looking up what elements are — cigre

| Task | Check | Agent without it | Layer off everywhere |
|---|---|---:|---:|
| cigre_feeder | feeder Line 1-2 and the route through Line 3-8 to Line 10-11 | 5/5 | 5/5 |
| cigre_levels | 20 kV, and Bus 0 at 110 kV | 5/5 | 5/5 |
| cigre_outage | named what loses supply, as far as Bus 11 | 4/5 | 2/5 |
| cigre_outage | named the tie Line 14-8 | 4/5 | 5/5 |
| cigre_renewables | named the wind turbine and PV units | 5/5 | 5/5 |
| cigre_storage | both batteries and their buses | 0/5 | 0/5 |

| | Agent without it | Layer off everywhere |
|---|---:|---:|
| Checks passed | 23/30 | 22/30 |
| Answers contradicting the network (entity) | 1 | 0 |
| Claims the network does not support (entity) | 0 | 0 |
| Tool calls per run | 8.1 | 8.8 |
| Seconds per run | 26.4 | 24.4 |

### Looking up what elements are — ieee30

| Task | Check | Agent without it | Layer off everywhere |
|---|---|---:|---:|
| ieee30_customers | said the network does not say | 2/5 | 1/5 |
| ieee30_external | L0 and L1 | 5/5 | 5/5 |
| ieee30_level | 135 kV | 4/5 | 4/5 |
| ieee30_radial | named what is cut off | 5/5 | 5/5 |
| ieee30_radial | said it loses supply | 4/5 | 5/5 |
| ieee30_renewables | said the network does not say | 3/5 | 2/5 |

| | Agent without it | Layer off everywhere |
|---|---:|---:|
| Checks passed | 23/30 | 22/30 |
| Answers contradicting the network (entity) | 0 | 0 |
| Claims the network does not support (entity) | 0 | 1 |
| Tool calls per run | 1.8 | 1.6 |
| Seconds per run | 12.2 | 11.7 |

### Reasoning

## Per task (runs with every check passed; checks passed)

| Family | Task | Agent without it | Layer off everywhere |
|---|---|---:|---:|
| names | names_cigre | 0/5 (5/20) | 0/5 (5/20) |
| names | names_ieee30 | 1/5 (1/5) | 0/5 (0/5) |
| names | names_cigre_wind | 0/5 (5/10) | 0/5 (5/10) |
| names | names_cigre_storage | 0/5 (0/10) | 0/5 (0/10) |
| names | names_ieee30_wind | 1/5 (1/5) | 0/5 (0/5) |
| intervention | drop_ieee30_none | 5/5 (10/10) | 2/5 (7/10) |
| intervention | drop_ieee30_declared | 5/5 (10/10) | 0/5 (4/10) |
| intervention | drop_cigre_stated | 5/5 (15/15) | 0/5 (7/15) |
| intervention | drop_cigre_declared | 5/5 (10/10) | 0/5 (1/10) |
| composition | compose_t0 | 0/5 (10/15) | 0/5 (9/15) |
| composition | compose_battery | 0/5 (9/15) | 0/5 (9/15) |
| composition | compose_l33 | 5/5 (10/10) | 5/5 (10/10) |
| composition | compose_tie | 3/5 (8/10) | 1/5 (6/10) |
| switches | switches_open | 0/5 (0/10) | 0/5 (0/10) |
| switches | switches_loops | 0/5 (5/10) | 0/5 (5/10) |
| switches | switches_breakers | 0/5 (4/10) | 0/5 (4/10) |
| switches | switches_line148 | 0/5 (0/10) | 0/5 (0/10) |
| data | data_cigre | 0/5 (3/10) | 0/5 (1/10) |
| data | data_ieee30 | 5/5 (5/5) | 5/5 (5/5) |
| data | data_cigre_wind | 4/5 (9/10) | 5/5 (10/10) |
| data | data_cigre_battery | 0/5 (0/5) | 1/5 (1/5) |
| data | data_cigre_load_split | 0/5 (0/5) | 0/5 (0/5) |
| data | data_ieee30_solved | 5/5 (5/5) | 5/5 (5/5) |
| consistency | para_weather | 5/5 (10/10) | 5/5 (10/10) |
| consistency | para_vary | 5/5 (10/10) | 5/5 (10/10) |
| consistency | para_variable | 5/5 (10/10) | 5/5 (10/10) |
| consistency | persist_ieee30 | 5/5 (10/10) | 5/5 (10/10) |

## Per family

| Family | Agent without it | Layer off everywhere |
|---|---:|---:|
| names | runs 2/25, checks 12/50 | runs 0/25, checks 10/50 |
| intervention | runs 20/20, checks 45/45 | runs 2/20, checks 19/45 |
| composition | runs 8/20, checks 37/50 | runs 6/20, checks 34/50 |
| switches | runs 0/20, checks 9/40 | runs 0/20, checks 9/40 |
| data | runs 14/30, checks 22/40 | runs 16/30, checks 22/40 |
| consistency | runs 20/20, checks 40/40 | runs 20/20, checks 40/40 |

## Sensitivity to an intervention (both sides of the pair right, same repeat)

| Pair | Agent without it | Layer off everywhere |
|---|---:|---:|
| drop_ieee30_none → drop_ieee30_declared | 5/5 | 0/5 |
| drop_cigre_stated → drop_cigre_declared | 5/5 | 0/5 |

## Consistency across wordings (same repeat)

| Group | Measure | Agent without it | Layer off everywhere |
|---|---|---:|---:|
| renewables | all wordings right | 5/5 | 5/5 |
| renewables | wordings agree | 5/5 | 5/5 |

## Process

| | Agent without it | Layer off everywhere |
|---|---:|---:|
| Tool calls per run | 2.2 | 2.1 |
| Seconds per run | 13.6 | 14.0 |
| Entity-check findings (total) | 7 | 8 |
| A fact the operator stated, recorded for later studies | 0/5 | 0/5 |
| Runs that used the layer's tools | 0/135 | 0/135 |

## The whole contribution — on vs none

### Looking up what elements are — cigre

| Task | Check | Layer on | Layer off everywhere |
|---|---|---:|---:|
| cigre_feeder | feeder Line 1-2 and the route through Line 3-8 to Line 10-11 | 5/5 | 5/5 |
| cigre_levels | 20 kV, and Bus 0 at 110 kV | 4/5 | 5/5 |
| cigre_outage | named what loses supply, as far as Bus 11 | 5/5 | 2/5 |
| cigre_outage | named the tie Line 14-8 | 3/5 | 5/5 |
| cigre_renewables | named the wind turbine and PV units | 5/5 | 5/5 |
| cigre_storage | both batteries and their buses | 5/5 | 0/5 |

| | Layer on | Layer off everywhere |
|---|---:|---:|
| Checks passed | 27/30 | 22/30 |
| Answers contradicting the network (entity) | 0 | 0 |
| Claims the network does not support (entity) | 0 | 0 |
| Tool calls per run | 4.0 | 8.8 |
| Seconds per run | 11.5 | 24.4 |

### Looking up what elements are — ieee30

| Task | Check | Layer on | Layer off everywhere |
|---|---|---:|---:|
| ieee30_customers | said the network does not say | 5/5 | 1/5 |
| ieee30_external | L0 and L1 | 5/5 | 5/5 |
| ieee30_level | 135 kV | 5/5 | 4/5 |
| ieee30_radial | named what is cut off | 5/5 | 5/5 |
| ieee30_radial | said it loses supply | 5/5 | 5/5 |
| ieee30_renewables | said the network does not say | 5/5 | 2/5 |

| | Layer on | Layer off everywhere |
|---|---:|---:|
| Checks passed | 30/30 | 22/30 |
| Answers contradicting the network (entity) | 0 | 0 |
| Claims the network does not support (entity) | 0 | 1 |
| Tool calls per run | 1.4 | 1.6 |
| Seconds per run | 6.4 | 11.7 |

### Reasoning

## Per task (runs with every check passed; checks passed)

| Family | Task | Layer on | Layer off everywhere |
|---|---|---:|---:|
| names | names_cigre | 5/5 (20/20) | 0/5 (5/20) |
| names | names_ieee30 | 5/5 (5/5) | 0/5 (0/5) |
| names | names_cigre_wind | 5/5 (10/10) | 0/5 (5/10) |
| names | names_cigre_storage | 5/5 (10/10) | 0/5 (0/10) |
| names | names_ieee30_wind | 5/5 (5/5) | 0/5 (0/5) |
| intervention | drop_ieee30_none | 5/5 (10/10) | 2/5 (7/10) |
| intervention | drop_ieee30_declared | 5/5 (10/10) | 0/5 (4/10) |
| intervention | drop_cigre_stated | 5/5 (15/15) | 0/5 (7/15) |
| intervention | drop_cigre_declared | 5/5 (10/10) | 0/5 (1/10) |
| composition | compose_t0 | 4/5 (14/15) | 0/5 (9/15) |
| composition | compose_battery | 4/5 (14/15) | 0/5 (9/15) |
| composition | compose_l33 | 5/5 (10/10) | 5/5 (10/10) |
| composition | compose_tie | 5/5 (10/10) | 1/5 (6/10) |
| switches | switches_open | 5/5 (10/10) | 0/5 (0/10) |
| switches | switches_loops | 5/5 (10/10) | 0/5 (5/10) |
| switches | switches_breakers | 5/5 (10/10) | 0/5 (4/10) |
| switches | switches_line148 | 5/5 (10/10) | 0/5 (0/10) |
| data | data_cigre | 5/5 (10/10) | 0/5 (1/10) |
| data | data_ieee30 | 4/5 (4/5) | 5/5 (5/5) |
| data | data_cigre_wind | 5/5 (10/10) | 5/5 (10/10) |
| data | data_cigre_battery | 5/5 (5/5) | 1/5 (1/5) |
| data | data_cigre_load_split | 5/5 (5/5) | 0/5 (0/5) |
| data | data_ieee30_solved | 3/5 (3/5) | 5/5 (5/5) |
| consistency | para_weather | 5/5 (10/10) | 5/5 (10/10) |
| consistency | para_vary | 4/5 (9/10) | 5/5 (10/10) |
| consistency | para_variable | 5/5 (10/10) | 5/5 (10/10) |
| consistency | persist_ieee30 | 5/5 (10/10) | 5/5 (10/10) |

## Per family

| Family | Layer on | Layer off everywhere |
|---|---:|---:|
| names | runs 25/25, checks 50/50 | runs 0/25, checks 10/50 |
| intervention | runs 20/20, checks 45/45 | runs 2/20, checks 19/45 |
| composition | runs 18/20, checks 48/50 | runs 6/20, checks 34/50 |
| switches | runs 20/20, checks 40/40 | runs 0/20, checks 9/40 |
| data | runs 27/30, checks 37/40 | runs 16/30, checks 22/40 |
| consistency | runs 19/20, checks 39/40 | runs 20/20, checks 40/40 |

## Sensitivity to an intervention (both sides of the pair right, same repeat)

| Pair | Layer on | Layer off everywhere |
|---|---:|---:|
| drop_ieee30_none → drop_ieee30_declared | 5/5 | 0/5 |
| drop_cigre_stated → drop_cigre_declared | 5/5 | 0/5 |

## Consistency across wordings (same repeat)

| Group | Measure | Layer on | Layer off everywhere |
|---|---|---:|---:|
| renewables | all wordings right | 4/5 | 5/5 |
| renewables | wordings agree | 4/5 | 5/5 |

## Process

| | Layer on | Layer off everywhere |
|---|---:|---:|
| Tool calls per run | 2.1 | 2.1 |
| Seconds per run | 9.5 | 14.0 |
| Entity-check findings (total) | 5 | 8 |
| A fact the operator stated, recorded for later studies | 5/5 | 0/5 |
| Runs that used the layer's tools | 125/135 | 0/135 |

