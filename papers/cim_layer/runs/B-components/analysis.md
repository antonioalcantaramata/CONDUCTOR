# Ablation: components

Leave one out: what each part of CONDUCTOR adds over the rest (the paper's main table).

Commit `3aa1afce42` · 3 repeats · suites: whatif, investigation, options, clarify, narration, semantics-cigre, semantics-ieee30, reasoning · order: interleaved

| Arm | Settings |
|---|---|
| full | as it is |
| no-semantic-layer | ontology=off |
| no-reading-check | reading_check=off |
| no-routing-rules | drop_rules=routing |
| no-evidence-wording | drop_rules=evidence_wording |
| no-ask-first | drop_rules=ask_first |
| no-investigation-tools | hide_tools=group:investigation |
| solver-only | ontology=none reading_check=off |

Baseline: **full**. Rates in %, 95 % Wilson interval; Δ is the arm minus the baseline over the tasks both ran, with a 95 % bootstrap interval over tasks, then tasks better ↑ / worse ↓ / same = and the sign test's p.

| Arm | Runs | Checks passed | Δ checks | Tasks fully passed | Δ tasks | Tool calls | Input tok | of which cached | Output tok | s | Ungrounded figures | Entity errors |
|---|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| full | 264 | 97.1 [95, 98] | – | 94.3 [91, 97] | – | 3.0 | 121,162 | 112,607 | 991 | 21 | 0.11 | 0.03 |
| no-semantic-layer | 264 | 87.8 [85, 90] | -13.2 [-20, -6]; 11↑ 24↓ 53= p=0.041 | 77.7 [72, 82] | -16.7 [-25, -8]; 11↑ 24↓ 53= p=0.041 | 3.2 | 122,830 | 113,743 | 1,138 | 23 | 0.09 | 0.03 |
| no-reading-check | 264 | 95.7 [94, 97] | -1.1 [-3, +1]; 7↑ 12↓ 69= p=0.36 | 91.7 [88, 94] | -2.7 [-6, +1]; 7↑ 12↓ 69= p=0.36 | 3.1 | 125,404 | 116,546 | 974 | 20 | 0.09 | 0.02 |
| no-routing-rules | 264 | 95.0 [93, 96] | -2.3 [-5, -0]; 5↑ 12↓ 71= p=0.14 | 90.5 [86, 94] | -3.8 [-8, -0]; 5↑ 12↓ 71= p=0.14 | 2.7 | 101,764 | 93,953 | 922 | 20 | 0.13 | 0.02 |
| no-evidence-wording | 264 | 97.6 [96, 99] | +0.1 [-2, +2]; 10↑ 5↓ 73= p=0.3 | 95.1 [92, 97] | +0.8 [-3, +4]; 9↑ 5↓ 74= p=0.42 | 3.0 | 121,826 | 113,175 | 977 | 22 | 0.07 | 0.03 |
| no-ask-first | 264 | 97.3 [96, 98] | +0.7 [-2, +3]; 9↑ 8↓ 71= p=1 | 94.3 [91, 97] | +0.0 [-4, +4]; 9↑ 8↓ 71= p=1 | 3.1 | 124,674 | 115,837 | 1,042 | 22 | 0.08 | 0.02 |
| no-investigation-tools | 264 | 84.8 [82, 87] | -12.3 [-18, -7]; 1↑ 19↓ 68= p=4e-05 | 78.4 [73, 83] | -15.9 [-23, -9]; 1↑ 19↓ 68= p=4e-05 | 3.0 | 104,754 | 96,174 | 974 | 18 | 0.12 | 0.02 |
| solver-only | 264 | 84.6 [82, 87] | -17.4 [-25, -10]; 10↑ 27↓ 51= p=0.0076 | 73.1 [67, 78] | -21.2 [-31, -12]; 9↑ 26↓ 53= p=0.006 | 3.5 | 130,976 | 122,443 | 1,201 | 24 | 0.11 | 0.03 |

## The reading check

Turns whose decisive studies were re-run under the readings the question left open, and how often another reading moved the figure by more than 10 % (shifts) or changed the conclusion (changes).

| Arm | Turns re-checked | Shifts | Changes |
|---|---:|---:|---:|
| full | 103 | 63 (61 %) | 21 (20 % [14, 29]) |
| no-semantic-layer | 108 | 66 (61 %) | 27 (25 % [18, 34]) |
| no-routing-rules | 96 | 61 (64 %) | 26 (27 % [19, 37]) |
| no-evidence-wording | 110 | 68 (62 %) | 23 (21 % [14, 29]) |
| no-ask-first | 108 | 66 (61 %) | 21 (19 % [13, 28]) |
| no-investigation-tools | 106 | 66 (62 %) | 26 (25 % [17, 34]) |

## By suite

| Suite | full | no-semantic-layer | no-reading-check | no-routing-rules | no-evidence-wording | no-ask-first | no-investigation-tools | solver-only |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| clarify | 26/27 | 27/27 | 26/27 | 24/27 | 23/27 | 20/27 | 26/27 | 25/27 |
| investigation | 117/123 | 123/123 | 115/123 | 112/123 | 120/123 | 123/123 | 45/123 | 122/123 |
| narration | 53/57 | 56/57 | 52/57 | 50/57 | 56/57 | 56/57 | 53/57 | 56/57 |
| options | 83/84 | 84/84 | 84/84 | 84/84 | 84/84 | 81/84 | 84/84 | 84/84 |
| reasoning | 157/159 | 97/159 | 153/159 | 156/159 | 156/159 | 156/159 | 154/159 | 79/159 |
| reasoning · composition | 29/30 | 20/30 | 27/30 | 28/30 | 29/30 | 27/30 | 28/30 | 21/30 |
| reasoning · consistency | 24/24 | 24/24 | 24/24 | 24/24 | 24/24 | 24/24 | 24/24 | 24/24 |
| reasoning · data | 23/24 | 13/24 | 21/24 | 23/24 | 23/24 | 24/24 | 22/24 | 12/24 |
| reasoning · intervention | 27/27 | 27/27 | 27/27 | 27/27 | 26/27 | 27/27 | 26/27 | 10/27 |
| reasoning · names | 30/30 | 7/30 | 30/30 | 30/30 | 30/30 | 30/30 | 30/30 | 6/30 |
| reasoning · switches | 24/24 | 6/24 | 24/24 | 24/24 | 24/24 | 24/24 | 24/24 | 6/24 |
| semantics-cigre | 18/18 | 14/18 | 17/18 | 16/18 | 15/18 | 17/18 | 16/18 | 13/18 |
| semantics-ieee30 | 18/18 | 14/18 | 17/18 | 18/18 | 18/18 | 18/18 | 17/18 | 13/18 |
| whatif | 134/138 | 133/138 | 133/138 | 133/138 | 137/138 | 136/138 | 134/138 | 136/138 |

## Checks where the arms differ

| Task | Check | full | no-semantic-layer | no-reading-check | no-routing-rules | no-evidence-wording | no-ask-first | no-investigation-tools | solver-only |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| clarify/unstated_margin | asked how far below before running | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 0/3 | 3/3 | 3/3 |
| clarify/which_line | asked which line before running | 2/3 | 3/3 | 2/3 | 1/3 | 1/3 | 2/3 | 3/3 | 2/3 |
| clarify/which_line | offered numbered choices | 3/3 | 3/3 | 3/3 | 2/3 | 1/3 | 1/3 | 2/3 | 2/3 |
| clarify/which_line | then simulated the outage of L31 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 2/3 | 3/3 | 3/3 |
| investigation/classical_views | asked for the classical views | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 0/3 | 3/3 |
| investigation/deep_why | checked prevention | 3/3 | 3/3 | 2/3 | 3/3 | 3/3 | 3/3 | 0/3 | 3/3 |
| investigation/deep_why | investigated | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 0/3 | 3/3 |
| investigation/deep_why | looked for recurrence | 3/3 | 3/3 | 2/3 | 3/3 | 3/3 | 3/3 | 0/3 | 3/3 |
| investigation/deep_why | proposed an analysis of the grid CONDUCTOR lacks | 3/3 | 3/3 | 2/3 | 3/3 | 3/3 | 3/3 | 0/3 | 3/3 |
| investigation/deep_why | proposed data for what lies behind the cause | 3/3 | 3/3 | 2/3 | 3/3 | 3/3 | 3/3 | 0/3 | 3/3 |
| investigation/deep_why | recorded the investigation | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 0/3 | 3/3 |
| investigation/injected_setpoint | investigated against the base | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 0/3 | 3/3 |
| investigation/look_ahead | looked ahead | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 0/3 | 3/3 |
| investigation/look_ahead | said when to decide | 2/3 | 3/3 | 1/3 | 3/3 | 3/3 | 3/3 | 1/3 | 2/3 |
| investigation/no_symptom | checked with the investigation | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 0/3 | 3/3 |
| investigation/no_symptom | said it is within its limits | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 1/3 | 3/3 |
| investigation/own_hypotheses | recorded Bus_23 as supported | 3/3 | 3/3 | 3/3 | 1/3 | 3/3 | 3/3 | 0/3 | 3/3 |
| investigation/own_hypotheses | recorded Bus_25 as supported | 3/3 | 3/3 | 3/3 | 1/3 | 3/3 | 3/3 | 0/3 | 3/3 |
| investigation/own_hypotheses | tested its own hypotheses (Bus_23, Bus_25) | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 0/3 | 3/3 |
| investigation/prevention | checked prevention | 3/3 | 3/3 | 3/3 | 3/3 | 2/3 | 3/3 | 0/3 | 3/3 |
| investigation/recurring | investigated every episode | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 0/3 | 3/3 |
| investigation/recurring | named the recurring cause | 2/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 0/3 | 3/3 |
| investigation/untestable | recorded the weather as untested, with the tool needed | 2/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 0/3 | 3/3 |
| investigation/untestable | said it could not be tested | 2/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 1/3 | 3/3 |
| investigation/what_caused_overload | investigated | 3/3 | 3/3 | 3/3 | 2/3 | 3/3 | 3/3 | 0/3 | 3/3 |
| investigation/why_bus29 | investigated | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 0/3 | 3/3 |
| investigation/why_bus29 | recorded the investigation | 3/3 | 3/3 | 3/3 | 2/3 | 3/3 | 3/3 | 0/3 | 3/3 |
| investigation/why_outage | investigated after the outage | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 0/3 | 3/3 |
| investigation/why_outage | recorded the investigation | 3/3 | 3/3 | 2/3 | 1/3 | 3/3 | 3/3 | 0/3 | 3/3 |
| investigation/why_so_close | compared with a comfortable moment | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 0/3 | 3/3 |
| investigation/why_window | investigated the window | 1/3 | 3/3 | 2/3 | 0/3 | 1/3 | 3/3 | 0/3 | 3/3 |
| narration/envelope | named the binding constraint | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 2/3 | 3/3 | 3/3 |
| narration/envelope | said it holds at this operating point only | 2/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 2/3 | 3/3 |
| narration/historical_risk | told the worst episodes | 2/3 | 3/3 | 2/3 | 2/3 | 3/3 | 3/3 | 2/3 | 3/3 |
| narration/renewables | compared with the baseline | 3/3 | 3/3 | 2/3 | 2/3 | 3/3 | 3/3 | 2/3 | 3/3 |
| narration/renewables | ran the scenario scan | 2/3 | 3/3 | 3/3 | 2/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| narration/renewables | stated the uniform-scaling limitation | 2/3 | 3/3 | 0/3 | 0/3 | 2/3 | 3/3 | 2/3 | 2/3 |
| narration/robust | explained the back-off, or why it fell short | 3/3 | 2/3 | 3/3 | 2/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| options/n1_secure | said what could not be secured | 2/3 | 3/3 | 3/3 | 3/3 | 3/3 | 1/3 | 3/3 | 3/3 |
| options/no_action | said no action is needed | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 2/3 | 3/3 | 3/3 |
| reasoning/compose_battery | found Bus 11 lowest | 3/3 | 3/3 | 2/3 | 3/3 | 3/3 | 2/3 | 3/3 | 2/3 |
| reasoning/compose_battery | named the feeder Line 1-2 | 3/3 | 0/3 | 2/3 | 3/3 | 3/3 | 3/3 | 3/3 | 0/3 |
| reasoning/compose_battery | said yes, the same feeder | 3/3 | 2/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| reasoning/compose_t0 | eight of them solar | 3/3 | 2/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| reasoning/compose_t0 | named the wind turbine among them | 3/3 | 3/3 | 3/3 | 2/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| reasoning/compose_t0 | the tie Line 14-8 restores it | 2/3 | 0/3 | 2/3 | 2/3 | 2/3 | 1/3 | 1/3 | 0/3 |
| reasoning/compose_tie | named the tie S1 (Line 14-8) | 3/3 | 1/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 1/3 |
| reasoning/data_cigre | Load R5's Q: derived from the power factor, not measured | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 1/3 |
| reasoning/data_cigre | PV 9's output: a share of its bus's measurement | 3/3 | 0/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 0/3 |
| reasoning/data_cigre_battery | held at its set-point; the data carry no storage | 3/3 | 0/3 | 2/3 | 3/3 | 2/3 | 3/3 | 3/3 | 0/3 |
| reasoning/data_cigre_load_split | a share of Bus 1's measurement, split with Load CI1 | 3/3 | 0/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 0/3 |
| reasoning/data_ieee30 | said it is derived from the power factor | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 2/3 |
| reasoning/data_ieee30_solved | said the power flow solves it | 2/3 | 1/3 | 1/3 | 2/3 | 3/3 | 3/3 | 1/3 | 3/3 |
| reasoning/drop_cigre_declared | named CHP diesel 1 as affected (declared solar) | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 1/3 |
| reasoning/drop_cigre_declared | the study scaled CHP diesel 1 too | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 0/3 |
| reasoning/drop_cigre_stated | did not count CHP diesel 1 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 0/3 |
| reasoning/drop_cigre_stated | the study scaled the nine wind and PV units | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 0/3 |
| reasoning/drop_ieee30_declared | named Gen_bus12 as affected | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 2/3 |
| reasoning/drop_ieee30_declared | the study scaled Gen_bus12 only | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 0/3 |
| reasoning/drop_ieee30_none | said no unit is stated wind or solar | 3/3 | 3/3 | 3/3 | 3/3 | 2/3 | 3/3 | 2/3 | 1/3 |
| reasoning/names_cigre | counted eight | 3/3 | 0/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 0/3 |
| reasoning/names_cigre | did not take 'PV Park East' (a fuel cell) for solar | 3/3 | 0/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 0/3 |
| reasoning/names_cigre | did not take 'Solar Farm North' (a CHP unit) for solar | 3/3 | 0/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 0/3 |
| reasoning/names_cigre_storage | did not take 'Battery Storage 3' (a fuel cell) for storage | 3/3 | 0/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 0/3 |
| reasoning/names_cigre_storage | named Battery 1 and Battery 2 | 3/3 | 0/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 0/3 |
| reasoning/names_cigre_wind | did not take 'Wind Park West' (a PV unit) for wind | 3/3 | 0/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 0/3 |
| reasoning/names_ieee30 | said 'Solar Plant A' is solar only by its name | 3/3 | 1/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 0/3 |
| reasoning/names_ieee30_wind | said 'Offshore Wind 1' is wind only by its name | 3/3 | 0/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 0/3 |
| reasoning/switches_breakers | named Switch 6 and Switch 7 | 3/3 | 0/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 0/3 |
| reasoning/switches_line148 | named its switch S1 | 3/3 | 0/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 0/3 |
| reasoning/switches_line148 | said it is normally open | 3/3 | 0/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 0/3 |
| reasoning/switches_loops | named S2 and S3 | 3/3 | 0/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 0/3 |
| reasoning/switches_open | named the three normally open switches | 3/3 | 0/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 0/3 |
| reasoning/switches_open | said S1 is the tie between feeders | 3/3 | 0/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 0/3 |
| semantics-cigre/cigre_feeder | feeder Line 1-2 and the route through Line 3-8 to Line 10-11 | 3/3 | 3/3 | 3/3 | 2/3 | 3/3 | 3/3 | 3/3 | 2/3 |
| semantics-cigre/cigre_outage | named the tie Line 14-8 | 3/3 | 2/3 | 2/3 | 3/3 | 0/3 | 2/3 | 1/3 | 3/3 |
| semantics-cigre/cigre_outage | named what loses supply, as far as Bus 11 | 3/3 | 3/3 | 3/3 | 2/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| semantics-cigre/cigre_renewables | named the wind turbine and PV units | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 2/3 |
| semantics-cigre/cigre_storage | both batteries and their buses | 3/3 | 0/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 0/3 |
| semantics-ieee30/ieee30_customers | said the network does not say | 3/3 | 0/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 1/3 |
| semantics-ieee30/ieee30_radial | said it loses supply | 3/3 | 3/3 | 2/3 | 3/3 | 3/3 | 3/3 | 2/3 | 3/3 |
| semantics-ieee30/ieee30_renewables | said the network does not say | 3/3 | 2/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 0/3 |
| whatif/classical_generator | asked how the unit is modelled before building it | 3/3 | 3/3 | 2/3 | 3/3 | 3/3 | 2/3 | 3/3 | 3/3 |
| whatif/classical_generator | no refused edit | 3/3 | 2/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| whatif/new_site_n1 | no refused edit | 2/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| whatif/new_site_n1 | said a single-line site loses supply | 3/3 | 0/3 | 0/3 | 2/3 | 2/3 | 2/3 | 2/3 | 1/3 |
| whatif/new_substation_trafo | edit: 2 × 0.63 MVA transformer to a 0.4 kV bus at Bus 9 | 2/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 2/3 | 3/3 |
| whatif/new_substation_trafo | reported how the units are modelled | 2/3 | 3/3 | 3/3 | 1/3 | 3/3 | 3/3 | 2/3 | 3/3 |
| whatif/open_line_n1 | said what loses supply | 3/3 | 2/3 | 2/3 | 2/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| whatif/reinforce_after_load | named the proposal | 2/3 | 3/3 | 3/3 | 2/3 | 3/3 | 3/3 | 2/3 | 3/3 |
