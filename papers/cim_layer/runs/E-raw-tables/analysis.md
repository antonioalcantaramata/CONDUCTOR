# Ablation: raw_tables

What the layer adds over the data it is built from: the agent given the network's raw data tables instead of the layer (the studies read the layer in every arm).

Commit `5bf4d30474` (uncommitted changes) · 5 repeats · suites: semantics-cigre, semantics-ieee30, reasoning · order: interleaved

| Arm | Settings |
|---|---|
| tables | ontology=tables |

Baseline: **tables**. Rates in %, 95 % Wilson interval; Δ is the arm minus the baseline over the tasks both ran, with a 95 % bootstrap interval over tasks, then tasks better ↑ / worse ↓ / same = and the sign test's p.

| Arm | Runs | Checks passed | Δ checks | Tasks fully passed | Δ tasks | Tool calls | Input tok | of which cached | Output tok | s | Ungrounded figures | Entity errors |
|---|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| tables | 185 | 83.7 [79, 87] | – | 72.4 [66, 78] | – | 2.2 | 89,087 | 85,415 | 598 | 23 | 0.01 | 0.02 |

## The reading check

Turns whose decisive studies were re-run under the readings the question left open, and how often another reading moved the figure by more than 10 % (shifts) or changed the conclusion (changes).

| Arm | Turns re-checked | Shifts | Changes |
|---|---:|---:|---:|
| tables | 28 | 9 (32 %) | 5 (18 % [8, 36]) |

## By suite

| Suite | tables |
|---|---:|
| reasoning | 222/265 |
| reasoning · composition | 39/50 |
| reasoning · consistency | 40/40 |
| reasoning · data | 20/40 |
| reasoning · intervention | 44/45 |
| reasoning · names | 47/50 |
| reasoning · switches | 32/40 |
| semantics-cigre | 30/30 |
| semantics-ieee30 | 20/30 |

## Checks where the arms differ

| Task | Check | tables |
|---|---|---:|
| (none) | |  |
