# CONDUCTOR — backend (`backend/`)

FastAPI service that exposes the power-system analysis engines as HTTP tools.
The [LLM agent](../llm_agent/) is the primary client: it calls these endpoints,
and the chat app renders the results. The backend holds the network and its
time-series in memory and runs pandapower / Pyomo+IPOPT under the hood.

> See the [root README](../README.md) for setup.
> `./start.sh` (project root) launches this service on `http://localhost:8000`
> (interactive API docs at `/docs`).

## Layout

```
backend/
├── main_backend.py          ← FastAPI app + all endpoints (orchestration)
├── rsa_engine.py            ← Real-time security assessment (power flow + limits)
├── ca_engine.py             ← Contingency assessment (single outage)
├── flex_engine.py           ← Pyomo + IPOPT optimizer, Ybus helpers
├── element_names.py         ← One spelling of every bus/line/trafo name
├── load_gen_assignment.py   ← Maps measurement rows → pandapower loads/sgens
├── network_loader.py        ← Load networks, gen→sgen conversion, build Ybus
├── sandbox.py               ← Network proposals: edits, validation, profiles
├── synthetic_timeseries.py  ← Generate synthetic measurement + forecast series
├── pipeline_functions.py    ← Optional external time-series data client
└── network_profiles/        ← Bundled grid profiles (YAML): pglib_case14, ieee14
```

The files other than `main_backend.py` are computation **engines**;
`main_backend.py` is the orchestration layer the endpoints live in.
`element_names.py` is the exception: a leaf module both the endpoints and
`rsa_engine.py` import, so that a bus, line or transformer is named identically
wherever it is reported.

## Element names

Callers join results onto topology **by name** — the system map draws
`/api/network/topology` and colours it with `/api/grid/rsa` — so every endpoint
has to spell a name the same way. `element_names.py` is the single place that
decides:

```
Bus_3 -> Bus_6 [T0]                       no name of its own
T1 60/10 | Bus_3 -> Bus_6 [T0]            the same transformer, named
```

The endpoints and the index are appended even when the element is named,
because indices shift when a network is modified and two transformers between
the same pair of buses are ordinary. Nothing outside this module should build
an element name.

## The datastream map

`pipeline_functions.py` is an optional client for one specific utility's
time-series API. To fetch anything it needs to know that datastream `515613`
is transformer 1's loading at a particular substation — a table of several
hundred rows tying numeric ids to real substation names.

That table is **operational metadata about a real network**, and a literal
`dict` in the source is a literal dict in every clone, every fork and every
commit of this public repository. So it lives in a file instead:

| file | contents | in git? |
| --- | --- | --- |
| `datastream_map.local.json` | the real ids and substation names | **no** — gitignored |
| `datastream_map.example.json` | the same shape, invented ids and names | yes |

`_load_datastream_map()` takes the first of `$CONDUCTOR_DATASTREAM_MAP`, the
local file, then the example. A machine with the real file behaves exactly as
before; a fresh clone gets the synthetic one and every code path still runs.
Nothing else in the backend changes behaviour based on which is loaded.

Five sections, all optional:

```
datastreams       id   -> {substation, parameter}    used to label fetched rows
transformer_p_c   id   -> {substation, parameter}    the transformer P/C series
substations       NAME -> {parameter: id}            the same facts, keyed by name
public_ids        [id, ...]                          what /api/eddk/fetch requests
public_id_labels  id   -> display name               labels for the above
```

Two shapes carry the same facts because both are in use: some callers look up
by id, others by substation name.

**Adding a substation** means editing your local file, not the source. The
tests pin themselves to the example map, so they never depend on a real map
being present and pass identically on a machine that has none.

## Startup (`lifespan`)

Runs once when the server boots; all expensive loads happen here so requests
stay fast:

1. Resolve the active network profile from the `GRID_PROFILE` env var
   (defaults to `pglib_case14`), or restore the last uploaded network.
2. Load the network (MATPOWER `.m`, pandapower JSON/Excel, or UCTE).
3. Generate (or restore) the **measurement** and **forecast** time-series.
4. Apply the first operating point and run a seed power flow.
5. Build the Ybus / admittance database used by the optimizer (one power flow; an
   outage is that database without the outaged branch).

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
    "grid_profile": {},          # metadata exposed via /api/grid_constants
    "proposals": {},             # network proposals {id -> edits, hash, summary}
}
```

## Network proposals (sandbox)

A proposal is the loaded network plus an ordered list of edits — `add_load`,
`add_generator`, `reinforce_line`, `upgrade_trafo` (a larger unit of the same
design: impedance falls in proportion, like units in parallel),
`add_bus_with_connection` (a new site joined to an existing bus by a line
copied from an existing line or a pandapower type), `add_bus_with_transformer`
(a new bus at another voltage behind a new transformer, step-down or step-up,
optionally at the far end of a new line; the unit copied from an existing one —
in per unit, resized with `sn_mva` — or a pandapower catalogue type, or given
explicitly: rating, vk/vkr, no-load losses, magnetising current, vector-group
shift, rated voltages, tap changer and position; up to 4 units in `parallel`,
one branch in the model), `set_in_service`
(`sandbox.py`). Edits only
append rows or change existing ones, so base indices are the same in every
proposal, and the base itself is never modified. Every analysis endpoint takes
`network` (`"base"`, the default, or a proposal id); `_net_for` / `_db_for`
materialise the proposal on first use and cache it with its own admittance
database. Proposals are shared by every browser tab and saved to
`systems/proposals.json` with a fingerprint of the base network's structure;
after a restart they are restored if the same network is loaded (and their
edits still apply). Uploading or resetting a network discards them.

Added elements are tagged `proposed` and follow their own profile, never the
measurements: loads `flat`, `follow_bus` (the bus's measured demand shape,
peaking at the load's size) or `shift` (on between two hours); generators
`dispatchable` (a set-point, redispatchable up to capacity) or `pv` / `wind`
(the dataset's measured production shape; the optimiser may curtail them but
not exceed it). A flat added load is a stated, fixed demand: load-stress factors and demand
uncertainty draws (probabilistic RSA, robust OPF, hosting capacity) leave it
unchanged (`_scale_loads`); a `follow_bus` load scales with the demand it
follows. `_assign_tick` applies the profiles after the measurements and
checks each added element injects what its edit says. A result computed on a
proposal carries a `network` block — its edits, and what each added element
injects at the tick. `GET /api/network/topology?network=<id>` returns a
proposal's topology, with the buses and lines its edits added marked `added`.

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
| **Network proposals** | `GET /api/proposals`, `POST /api/proposals/edit`, `DELETE /api/proposals/{id}`, `DELETE /api/proposals/{id}/edits/{n}`, `POST /api/proposals/compare` (`study=rsa\|n1\|opf\|scan` — `opf`: corrective dispatch on both; `scan`: both networks at every tick of a window) |
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
