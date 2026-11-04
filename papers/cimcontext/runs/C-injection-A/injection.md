# Error injection — 527 answered turns, seed 0

Recall: injected errors the check flagged (excluding those no figure check could see). False alarms: true statements, worded the same way, that the check flagged. 95 % Wilson intervals.

| Level | Error put into the answer | Check | Injected | Undetectable | Flagged | Rate | 95 % CI |
|---|---|---|---:|---:|---:|---:|---|
| L1 | figure_changed | provenance | 329 | 190 | 134/139 | 96 % | [92, 98] |
| L1 | figure_swapped | provenance | 15 | 0 | 15/15 | 100 % | [80, 100] |
| L1 | insecure_called_secure | characterisation | 15 | 0 | 15/15 | 100 % | [80, 100] |
| L1 | proof_without_certificate | evidence | 15 | 0 | 15/15 | 100 % | [80, 100] |
| L1 | secure_without_study | evidence | 104 | 0 | 104/104 | 100 % | [96, 100] |
| L2 | bus_voltage_wrong | entities | 228 | 0 | 228/228 | 100 % | [98, 100] |
| L2 | invented_element | entities | 445 | 0 | 441/445 | 99 % | [98, 100] |
| L2 | line_called_transformer_after | entities | 50 | 0 | 50/50 | 100 % | [93, 100] |
| L2 | line_called_transformer_before | entities | 50 | 0 | 50/50 | 100 % | [93, 100] |
| L2 | unit_source_contradicted | entities | 157 | 0 | 157/157 | 100 % | [98, 100] |
| L2 | unit_source_unsupported | entities | 51 | 0 | 51/51 | 100 % | [93, 100] |
| L2 | value_called_measured | entities | 356 | 0 | 356/356 | 100 % | [99, 100] |
| L2 | bus_voltage_true (control: true) | entities | 228 | 0 | 0/228 | 0 % | [0, 2] |
| L2 | unit_source_true (control: true) | entities | 157 | 0 | 0/157 | 0 % | [0, 2] |
| L2 | value_measured_true (control: true) | entities | 297 | 0 | 0/297 | 0 % | [0, 1] |

How each was judged (verdict of the check on the injected item):

- figure_changed: imprecise ×5, ungrounded ×134
- figure_swapped: misattributed ×15
- insecure_called_secure: conflict ×15
- proof_without_certificate: overstated ×15
- secure_without_study: overstated ×104
- bus_voltage_wrong: entity_contradicted ×228
- invented_element: entity_contradicted ×441, none ×4
- line_called_transformer_after: entity_contradicted ×50
- line_called_transformer_before: entity_contradicted ×50
- unit_source_contradicted: entity_contradicted ×157
- unit_source_unsupported: entity_unsupported ×51
- value_called_measured: entity_contradicted ×356
- bus_voltage_true: none ×228
- unit_source_true: none ×157
- value_measured_true: none ×297

The same checks on the 527 answers as written (no error injected) — label each before calling it a false alarm:

- entity_contradicted: 6 (1.1 per 100 answers)
- entity_unsupported: 8 (1.5 per 100 answers)
- ungrounded_figure: 17 (3.2 per 100 answers)
