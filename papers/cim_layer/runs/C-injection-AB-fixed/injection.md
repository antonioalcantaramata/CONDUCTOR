# Error injection — 2717 answered turns, seed 0

Recall: injected errors the check flagged (excluding those no figure check could see). False alarms: true statements, worded the same way, that the check flagged. 95 % Wilson intervals.

| Level | Error put into the answer | Check | Injected | Undetectable | Flagged | Rate | 95 % CI |
|---|---|---|---:|---:|---:|---:|---|
| L1 | figure_changed | provenance | 2140 | 716 | 1331/1424 | 93 % | [92, 95] |
| L1 | figure_swapped | provenance | 60 | 1 | 59/59 | 100 % | [94, 100] |
| L1 | insecure_called_secure | characterisation | 198 | 0 | 198/198 | 100 % | [98, 100] |
| L1 | proof_without_certificate | evidence | 46 | 0 | 46/46 | 100 % | [92, 100] |
| L1 | secure_without_study | evidence | 216 | 0 | 211/216 | 98 % | [95, 99] |
| L2 | bus_voltage_wrong | entities | 1748 | 0 | 1748/1748 | 100 % | [100, 100] |
| L2 | invented_element | entities | 2457 | 0 | 2445/2457 | 100 % | [99, 100] |
| L2 | line_called_transformer_after | entities | 257 | 0 | 256/257 | 100 % | [98, 100] |
| L2 | line_called_transformer_before | entities | 257 | 0 | 256/257 | 100 % | [98, 100] |
| L2 | unit_source_contradicted | entities | 427 | 0 | 427/427 | 100 % | [99, 100] |
| L2 | unit_source_unsupported | entities | 642 | 0 | 642/642 | 100 % | [99, 100] |
| L2 | value_called_measured | entities | 1507 | 0 | 1507/1507 | 100 % | [100, 100] |
| L2 | bus_voltage_true (control: true) | entities | 1748 | 0 | 0/1748 | 0 % | [0, 0] |
| L2 | unit_source_true (control: true) | entities | 427 | 0 | 0/427 | 0 % | [0, 1] |
| L2 | value_measured_true (control: true) | entities | 1359 | 0 | 0/1359 | 0 % | [0, 0] |

How each was judged (verdict of the check on the injected item):

- figure_changed: grounded ×11, imprecise ×73, not extracted ×9, ungrounded ×1331
- figure_swapped: misattributed ×59
- insecure_called_secure: conflict ×198
- proof_without_certificate: overstated ×46
- secure_without_study: none ×5, overstated ×211
- bus_voltage_wrong: entity_contradicted ×1748
- invented_element: entity_contradicted ×2445, none ×12
- line_called_transformer_after: entity_contradicted ×256, none ×1
- line_called_transformer_before: entity_contradicted ×256, none ×1
- unit_source_contradicted: entity_contradicted ×427
- unit_source_unsupported: entity_unsupported ×642
- value_called_measured: entity_contradicted ×1507
- bus_voltage_true: none ×1748
- unit_source_true: none ×427
- value_measured_true: none ×1359

The same checks on the 2717 answers as written (no error injected) — label each before calling it a false alarm:

- entity_contradicted: 7 (0.3 per 100 answers)
- entity_unsupported: 4 (0.1 per 100 answers)
- ungrounded_figure: 61 (2.2 per 100 answers)
