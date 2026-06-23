# Backend — Power-System Digital Twin

FastAPI service that sits between the Dash frontend and the four
physics / optimization engines of the power-grid digital
twin. This document is a dissection of `main_backend.py`: what runs at
startup, what state lives in memory, and what every endpoint does
under the hood.

All other files in this directory (`rsa_engine.py`, `ca_engine.py`,
`flex_engine.py`, `modify_network.py`, `load_gen_assignment.py`,
`pipeline_functions.py`) are **engines** — pure computation, off-limits
for modification. `main_backend.py` is the only orchestration layer,
and this README describes only that orchestration.

---

## 1. File layout

```
Backend/
├── main_backend.py          ← FastAPI app, all endpoints (this doc)
├── rsa_engine.py            ← Real-Time Security Assessment (pp.runpp)
├── ca_engine.py             ← Contingency Assessment (single outage)
├── flex_engine.py           ← Pyomo + Ipopt optimizer, Ybus helpers
├── modify_network.py        ← Topological surgery (islanded buses, dead trafos)
├── load_gen_assignment.py   ← Maps measurement rows → pandapower loads/sgens
├── pipeline_functions.py    ← EDDK data-space HTTP client
└── main.py                  ← Pre-existing standalone script (not the API)
```

---

## 2. Startup lifecycle (`lifespan`)

FastAPI's `asynccontextmanager` runs exactly once when uvicorn boots.
All expensive disk loads happen here so per-request handlers stay in
the ~0.03 s range.

| Step | Action | Typical cost |
|-----:|---|---:|
| 1 | Resolve the active network profile from `GRID_PROFILE`. | fast |
| 2 | Load the network model from MATPOWER, pandapower, JSON, or pickle sources. | 1-5 s |
| 3 | Generate or load the operating time series for the active profile. | 1-10 s |
| 4 | Assign the first operating point to the network and run a seed power flow. | fast |
| 5 | Build the Ybus/admittance databases used by the optimization routines. | 2-20 s |

On `FileNotFoundError` the backend still starts but the flex / KPI
endpoints return **503** because `db_full` stays `None`.

---

## 3. The `app_data` singleton

The single source of truth. Populated once by `lifespan`; only
`/api/time/advance` mutates it (bumps `current_index`).

```python
app_data = {
    "timestamps":     [],      # sorted list[str] of all simulation ticks
    "measurements":   {},      # dict[ts -> DataFrame] of smart-meter rows
    "current_index":  0,       # cursor into timestamps
    "net":            None,    # pandapowerNet — base grid, surgery applied
    "db_full":        None,    # dict — intact-grid Ybus
    "db_n1_line":     None,    # dict[line_idx -> Ybus dict]
    "db_n1_trafo":    None,    # dict[trafo_idx -> Ybus dict]
}
```

Every request handler begins with a `copy.deepcopy(app_data["net"])`
so concurrent requests cannot interfere with each other.

---

## 4. Constants

### `PG_MAX_DATA`

Generator maximum active-power capacities (MW), keyed by substation.
The active values are the canonical Anosh-provided numbers from
`main_KPIs_Included.ipynb`. The previous in-backend values are
preserved as a commented block at the top of `main_backend.py` for
traceability.

Used by:
1. `/api/grid/rsa` → `total_available_capacity_mw` flexibility metric.
2. `/api/kpi/evaluate` → KPI-1 (Target Demand Flexibility).

### `substations`

Canonical list of the active profile's bus/substation names. Order is
used only for UI dropdowns; the assignment engines match by string equality.

---

## 5. Helper functions

### `prepare_measurement_df(df)`
Normalizes a raw measurement slice so `load_gen_assignment.py`
accepts it: renames `element` → `substation_name`; duplicates
`Pg_new` into `production` and adds a zero `consumption` column.

### `extract_flexibility_results(model, net_base, ts)`
Walks a solved Pyomo model and returns **only** the dispatch records
whose setpoint actually moved (|Pg_new − Pg_base| > 1 kW). Used by
the Flexibility and Contingency tables where the user cares about
*what changed*. **Not** used by the KPI endpoint — KPIs need *all*
generators, including the unchanged ones.

### KPI functions (3)
Ported verbatim from `main_KPIs_Included.ipynb`:

| Function | Formula |
|---|---|
| `calculate_kpi_target_demand_flexibility(df, pgmax)` | `Σ(Pg_max − Pg_base) / Σ Pg_new × 100` (generators only) |
| `calculate_kpi_flexibility_utilization(df)` | `(1 − \|F_act − F_req\|/F_req) × 100` |
| `calculate_kpi_prevented_violation_ratio(df)` | 100 if `df` non-empty else 0 |

Where `F_required = |ΣPg_new − ΣPg_base|` and `F_activated = Σ(Pg_up + Pg_down)`.

---

## 6. Endpoints

| Method | Path | Purpose | Engines called |
|:---|:---|:---|:---|
| `GET`  | `/` | Liveness probe. | — |
| `GET`  | `/api/time/current` | Read the current simulation timestamp. | — |
| `POST` | `/api/time/advance` | Bump the cursor by one tick. | — |
| `POST` | `/api/grid/rsa` | RSA at current timestamp (no optimizer). | `rsa_engine` |
| `POST` | `/api/contingency/simulate` | Single N-1 outage, no cure. | `ca_engine` |
| `POST` | `/api/contingency/optimize` | Outage + Pyomo cure. | `modify_network`, `flex_engine` |
| `POST` | `/api/flexibility/optimize` | Pyomo cure on intact grid. | `flex_engine` |
| `POST` | `/api/kpi/evaluate` | 3 official KPIs. | `rsa_engine`, `flex_engine` |
| `POST` | `/api/eddk/fetch` | Pull time-series from EDDK. | `pipeline_functions` |
| `POST` | `/api/eddk/push` | Placeholder push (no-op). | — |

### `/api/grid/rsa`
1. Deepcopy base net → apply measurements → apply load scaling.
2. `rsa_engine.real_time_security_assessment(net, ts)` → DataFrame of violations + computed bus voltages.
3. Compute `total_available_capacity_mw` from `PG_MAX_DATA`.

⚠ Taps are **not** neutralised here — this endpoint reflects actual
mechanical taps, which can produce over-voltage readings that look
bad but are physically accurate. Taps are only reset in the
optimization endpoints.

### `/api/contingency/simulate`
1. Deepcopy base net + apply measurements + load scaling.
2. Set `in_service = False` on the targeted line/trafo.
3. `ca_engine.contingency_assessment_single(...)` → DataFrame of violations after the outage.
4. Map each violation index back to a human-readable bus/line/trafo name.

### `/api/contingency/optimize`
1. Requires `db_n1_line` and `db_n1_trafo` loaded → else **503**.
2. Deepcopy + measurements + scaling + **tap reset to neutral** (Ybus precompute assumed neutral taps).
3. `modify_network.remove_*` to excise islanded buses created by the outage.
4. One `pp.runpp` (swallow `LoadflowNotConverged` — the optimizer can still solve from a partial state).
5. Pull the correct Ybus from `db_n1_{line,trafo}` via `flex_engine.get_admittance_parameters(db, idx)`.
6. `flex_engine.optimization_model_base(...)` — returns `(results_opt, model_opt)`.
7. On `TerminationCondition.optimal` → `extract_flexibility_results(...)` for the UI table.

### `/api/flexibility/optimize`
Same shape as `/api/contingency/optimize` but:
- Uses `db_full` (intact grid) — no outage modelled.
- Optionally drops one substation's sgens (`disabled_generators`) to simulate a generator kill.

The Two-Fold optimization logic (Fold 1: feasibility with fixed gens;
Fold 2: minimize deviation) lives inside `flex_engine`.

### `/api/kpi/evaluate`
This is the endpoint that changed most when the mock was replaced:

1. Same preparation as `/api/flexibility/optimize`.
2. Pre-optimization RSA pass → `violations_before` (informational).
3. Time the solve: `t = time.perf_counter()` around `flex_engine.optimization_model_base(...)`.
4. Build the **full** regulation DataFrame via `flex_engine.prepare_regulation_df_rsa(model, net, net_base, ts)`. Every generator + the external-grid row is present.
5. Feed the frame into the three KPI functions.

Response shape:
```json
{
  "timestamp": "2022-08-01 00:00:00",
  "status": "success",
  "metrics": {
    "kpi_1_target_demand_flex_pct": 42.17,
    "kpi_1_available_upward_flex_mw": 7.832,
    "kpi_1_system_demand_mw": 18.573,
    "kpi_2_flex_utilization_pct": 98.5,
    "kpi_2_required_flex_mw": 0.214,
    "kpi_2_activated_flex_mw": 0.217,
    "kpi_3_prevented_violation_ratio_pct": 100.0,
    "kpi_3_status": "All detected violations resolved",
    "violations_before": 3,
    "algorithm_runtime_sec": 0.04
  }
}
```

On infeasibility (`TerminationCondition != optimal`) the endpoint still
returns 200 with `status: "failed"`, KPI-1 and KPI-2 zeroed, and
KPI-3 = 0%.

### `/api/eddk/fetch`
Thin wrapper around `pipeline_functions`:
1. `define_timespan(start, end)` → ISO interval strings.
2. `get_ids_by_substation("*")` → all 16 substation datastream IDs.
3. `fetch_timespan_values(token, ids, start, end)` → list of `{timestamp, substation, value}` records.

Requires `EDDK_API_TOKEN` in the `.env` file.

### `/api/eddk/push`
Phase-4 placeholder. Returns success unconditionally.

---

## 7. Request-flow diagrams

### RSA
```
UI slider / timestamp tick
    │
    ▼
POST /api/grid/rsa {load_scaling_factor}
    │
    ▼  deepcopy(app_data["net"])
load_gen_assignment.assign_*
    │
    ▼  load *= load_scaling_factor
rsa_engine.real_time_security_assessment
    │
    ▼
JSON {violations, all_voltages, flexibility_metrics}
```

### Flexibility / KPI
```
POST /api/flexibility/optimize  (or /api/kpi/evaluate)
    │
    ▼  deepcopy → assign → tap neutral → optional disable sgen
flex_engine.optimization_model_base(net, net_base, Yff_*, Yft_*, TAPS, ...)
    │
    ▼ Ipopt solve (~0.03 s thanks to pre-computed Ybus)
    │
    ├──► extract_flexibility_results  (active rows) → /flexibility/optimize response
    │
    └──► prepare_regulation_df_rsa   (full rows)
             │
             ▼
         calculate_kpi_{1,2,3}  → /api/kpi/evaluate response
```

### Contingency + optimize
```
POST /api/contingency/optimize {element_type, element_index}
    │
    ▼  deepcopy → assign → scale → tap neutral
modify_network.remove_* (surgery)
    │
    ▼  pp.runpp (swallow LoadflowNotConverged)
flex_engine.get_admittance_parameters(db_n1_*, idx)
    │
    ▼
flex_engine.optimization_model_base(modified_net, ...)
    │
    ▼
extract_flexibility_results → JSON
```

---

## 8. Known quirks

- `app_data` is process-global. Running under gunicorn with `--workers > 1` would give each worker its own copy, including 20-30 s of startup per worker. Keep it at 1 worker or refactor to shared memory.
- `copy.deepcopy(net)` on every request is the biggest per-request cost. If that becomes a bottleneck, consider pandapower's `pp.toolbox.net_with_only_*` or lazy views.
- Several handlers swallow `pp.powerflow.LoadflowNotConverged` — this is intentional: a partial seed flow is still useful input to the optimizer. Do not turn these into hard errors without updating the solver path.
- `/api/time/advance` silently caps at the last timestamp; the frontend should show a "no more data" state but currently does not.
