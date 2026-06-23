# CONDUCTOR — backend (`backend/`)

FastAPI service that exposes the power-system analysis engines as HTTP tools.
The [LLM agent](../llm_agent/) is the primary client: it calls these endpoints,
and the chat app renders the results. The backend holds the network and its
time-series in memory and runs pandapower / Pyomo+IPOPT under the hood.

> See the [root README](../README.md) for the work-in-progress notice and setup.
> `./start.sh` (project root) launches this service on `http://localhost:8000`
> (interactive API docs at `/docs`).

## Layout

```
backend/
├── main_backend.py          ← FastAPI app + all endpoints (orchestration)
├── rsa_engine.py            ← Real-time security assessment (power flow + limits)
├── ca_engine.py             ← Contingency assessment (single outage)
├── flex_engine.py           ← Pyomo + IPOPT optimizer, Ybus helpers
├── modify_network.py        ← Topology surgery (islanded buses, dead trafos)
├── load_gen_assignment.py   ← Maps measurement rows → pandapower loads/sgens
├── network_loader.py        ← Load networks, gen→sgen conversion, build Ybus
├── synthetic_timeseries.py  ← Generate synthetic measurement + forecast series
├── pipeline_functions.py    ← Optional external time-series data client
└── network_profiles/        ← Bundled grid profiles (YAML): pglib_case14, ieee14
```

The files other than `main_backend.py` are computation **engines**;
`main_backend.py` is the orchestration layer the endpoints live in.

## Startup (`lifespan`)

Runs once when the server boots; all expensive loads happen here so requests
stay fast:

1. Resolve the active network profile from the `GRID_PROFILE` env var
   (defaults to `pglib_case14`), or restore the last uploaded network.
2. Load the network (MATPOWER `.m`, pandapower JSON/Excel, or UCTE).
3. Generate (or restore) the **measurement** and **forecast** time-series.
4. Apply the first operating point and run a seed power flow.
5. Build the Ybus / admittance databases used by the optimizer (intact + N-1).

## In-memory state (`app_data`)

A process-global singleton, populated at startup. Each request deep-copies the
network so concurrent requests don't interfere.

```python
app_data = {
    "timestamps": [],            # measurement ticks (the simulation clock runs here)
    "measurements": {},          # {timestamp -> DataFrame}
    "current_index": 0,          # clock cursor into `timestamps`
    "forecast_timestamps": [],   # forecast ticks (read-only look-ahead; no clock)
    "forecasts": {},             # {timestamp -> DataFrame}
    "measurements_source": "synthetic",  # or "uploaded"
    "forecasts_source": "synthetic",     # or "uploaded"
    "net": None,                 # pandapower network
    "db_full": None,             # intact-grid Ybus
    "db_n1_line": None,          # per-line N-1 Ybus database
    "db_n1_trafo": None,         # per-trafo N-1 Ybus database
    "grid_profile": {},          # metadata exposed via /api/grid_constants
}
```

## Measurements vs forecasts

Two parallel series. **Measurements** are the historical actuals the simulation
clock advances through. **Forecasts** are a read-only look-ahead horizon for
planning. Analysis endpoints take a `data_source` field (`"measurements"` |
`"forecasts"`) and point-in-time endpoints take an optional `timestamp`; both are
resolved against the chosen series. On network upload, both series are generated
synthetically and persisted to `data_files/`; users can replace either with their
own CSV via `/api/data/upload`.

## Endpoints

| Group | Endpoints |
|---|---|
| **State / clock** | `GET /api/grid_constants`, `GET /api/time/current`, `GET /api/time/timeline`, `POST /api/time/advance` |
| **Upload** | `POST /api/network/upload`, `POST /api/data/upload` (`kind=measurements\|forecasts`), `POST /api/network/snapshot` |
| **Security** | `POST /api/grid/rsa`, `POST /api/rsa/worst_case`, `POST /api/rsa/scan_scenarios`, `POST /api/rsa/probabilistic`, `POST /api/rsa/historical_risk`, `POST /api/grid/element_timeseries` |
| **Contingency (N-1)** | `POST /api/contingency/simulate`, `POST /api/contingency/simulate_all`, `POST /api/contingency/optimize` |
| **Flexibility / OPF** | `POST /api/flexibility/optimize`, `POST /api/flexibility/robust`, `POST /api/flexibility/envelope`, `POST /api/flexibility/hosting_capacity` |
| **KPIs** | `POST /api/kpi/evaluate`, `POST /api/kpi/forecast` |
| **External data** *(optional)* | `POST /api/eddk/fetch`, `POST /api/eddk/push` |

Full request/response schemas are browsable at `/docs` while the server runs.

## Notes

- **Single process.** `app_data` is process-global; run with **one** worker
  (running multiple would duplicate the startup cost and the state).
- **Per-request isolation** comes from deep-copying the network on each call.
- Optimizer-backed endpoints (OPF / N-1 / KPIs) return **503** until the Ybus
  databases finish building at startup.
- The simulation clock only advances over measurements; forecasts are queried by
  timestamp, never stepped into.
