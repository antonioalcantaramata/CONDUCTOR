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

1. Load the saved **scenario** (`systems/scenario.json`, chosen in the app —
   see *Starting scenarios* below): a demo, or Resume — the last uploaded
   network, else the profile named by `GRID_PROFILE` (defaults to `pglib_case14`).
2. Load the network (MATPOWER `.m`, pandapower JSON/Excel, or UCTE).
3. Generate (or restore) the **measurement** and **forecast** time-series.
4. Apply the first operating point and run a seed power flow.
5. Build the Ybus / admittance database used by the optimizer (one power flow; an
   outage is that database without the outaged branch).

### Starting scenarios (`scenarios.py`)

| Scenario | Loads |
|---|---|
| `resume` | The last uploaded network and data (the "last uploaded" records). Only offered once something has been uploaded. |
| `ieee14` | PGLib IEEE 14-bus with its synthetic week, active power only (loads use the load power factor). The nominal operating point is included. |
| `ieee14_reactive` | The same network and week plus measured reactive demand and generation (`demo_reactive_csv`, fixed seed): per-substation pf 0.88–0.98, one leading substation, about 1 % of cells and three hours of one substation left empty so the data-quality report shows gaps. |

`GET /api/assumptions` also returns `data_key`, which changes whenever other
measurements or forecasts are loaded (a hash of a few sums, so it survives a
restart); the app stamps each answer with it to tell when an answer was
computed on other data.

`GET /api/scenario` returns the current one and the options (Resume with what it
would load, and whether it is available); `POST /api/scenario {name}` loads one
and remembers it for the next start. Demos never read or write the "last
uploaded" records, so Resume stays as it was; an upload makes Resume current;
`reset_active` forgets the choice. Saved proposals and assumptions are kept **per
network** (`{"version": 2, "by_fingerprint": {...}}` in `proposals.json` /
`assumptions.json`; the older single-network format is still read), so switching
between a demo and your own network keeps each one's work.

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
one branch in the model), `add_shunt` (a capacitor bank or reactor, Mvar),
`set_in_service` (lines, transformers, loads, units, shunts, storage), and `set_parameter` — any setting of any existing element: the external grid's voltage set-point (`element_type: ext_grid`, `field: vm_pu`), a generator's set-point or limits, a transformer tap, a line rating or loading limit, a bus's voltage limits, a load's or unit's power (held at every tick instead of the measured value, via `sandbox.apply_overrides` in `_assign_tick`). Connection fields are refused there (use `move_element`); an unknown field is refused with the list of editable ones. `add_line` / `add_trafo` add a branch between two **existing** buses (same voltage for a line, different for a transformer, whose unit is given like `add_bus_with_transformer`'s). `move_element` reconnects a load, unit, shunt, storage or the external grid to another bus, or one end of a line; a measured load or unit keeps its own measurements (`sandbox.before_tick` puts it back on its measured bus for the assignment, `apply_overrides` moves it again). `scale` multiplies existing loads or units (by element, by bus, or `all`) at every tick, keeping their profile; `before_tick` restores the unscaled value first so a value the tick does not rewrite is never scaled twice. `include_proposal` (expanded by the endpoint, not stored) copies another proposal's edits in to build "A + B". `add_generator` takes the unit's model like the System tab:
`control` (`voltage` — a `gen` row holding `vm_pu`, Q within `q_min/q_max_mvar`;
`power_factor` — an `sgen` at `pf`, injecting or `absorbing`), `dispatchable`,
`p_min_mw`; `add_load` takes `leading`. **Missing decisions are questions, not
guesses:** an edit that lacks something essential (the unit's kind; how a
conventional unit is controlled; capacitor or reactor; a set-point clashing with
the bus's) raises `sandbox.EditQuestion`, which the endpoint returns as
`{"needs_operator": true, "question", "options"}` — nothing is stored. Every
default the tool fills in is returned as `assumed`. Edits stored before a field
existed replay with what they meant then (`sandbox.upgrade_stored`).
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

**Nominal operating point** (`data_source="nominal"`, `nominal.py`): the network's
own values as loaded — each load's P and Q, each generator's set-point — kept in
`app_data["nominal"]` before the first synthetic tick is written over them (at
every network load, uploads included). Point studies apply them per element and
exactly (a tick is spread per substation at a 0.95 power factor; nominal keeps the
case's own reactive power); it has no time, so a `timestamp` is refused and studies
over a time window (`_window_dataset`: worst case, scans, historical risk, forecast
KPIs, data series, comparison over time) refuse it with the reason. Elements added
by a proposal sit at their design value there. The snapshot at nominal adds a
`nominal` block: totals, and how the measurement and forecast series relate to it
per substation. Missing values in the file (NaN, the −9999 marker) make it
unavailable — the timeline's `nominal` block says why.

**Measured reactive power** (`data_quality.py`): an uploaded CSV may add
`consumption_mvar` (reactive demand; positive draws) and `production_mvar`
(reactive generation, same sign as `production_mw`). When present they are always
used — loads take the measured Q, split over a substation's loads like its P;
fixed-power-factor / fixed-Q units at a measured bus take the measured Q, split
by capacity and protected from their own model (`_q_from_meas`); a bus with a
voltage-controlled unit is left to that unit — and the load power factor only
fills cells left empty (a missing value is never read as 0; a −9999 marker is
missing). Missing **active** power (an empty cell or a marker) is interpolated in
time per substation (nearest value at the ends; a substation with none is 0) and
reported. `POST /api/generators/follow_measured` switches the voltage-controlled
units at given substations to fixed Q (their median measured Q where data are
missing), so they follow the measured reactive generation. Every parse keeps notes, and `GET /api/data/quality` (also in the upload
response) reports coverage, empty columns, power factors below 0.7, leading
loads, reactive demand without load, values that look like kvar, all-zero
columns, interpolated or absent active power, missing-value markers, and measured
generation at voltage-controlled units. Results
state the share of load whose Q was measured (`load_power_factor.
measured_reactive_power`); the timeline, the assumptions and the grid constants
say whether each dataset carries reactive power; the data series include it.

**Load power factor** (reactive power of measured loads): where the series carry
active power only, each measured load gets Q = P·tan(acos pf), lagging.
`lg.DEFAULT_LOAD_POWER_FACTOR` = 0.95 is the default of a session setting
(`GET/POST /api/assumptions`, saved for that network in
`systems/assumptions.json` like the proposals, reset when the network changes);
any study can override it for itself with `load_power_factor` (every request
model takes it, via `_NetworkLimitsMixin`). The study running in a thread carries
its value (`_study`, set where `_off_event_loop` starts it), so every tick
assignment uses it, and every result that assigned a tick carries a
`load_power_factor` block (value, and source: default / session setting / this
study). The nominal point uses none. Measured generation on the
measured-substation path (a network loaded without a `gen` table) uses
`MEASURED_GENERATION_POWER_FACTOR` = 0.99 (the code comment said 0.95); the
generic path keeps each unit's own Q. The robust OPF takes
`opf_min_power_factor` like the other OPFs (it hardcoded 0.95).

**Generator models** (`generators.py`, `GET/POST /api/generators`): each unit's
*control* — `voltage` (a `gen` row: holds `vm_pu`, Q free within its limits) or
`power_factor` (an `sgen` row: fixed pf, injecting or absorbing; Q set for each
tick's P by `generators.set_reactive`) — and whether it is *dispatchable*
(`controllable`; the OPFs pin a unit that is not, `flex_engine._fixed_output_units`).
A model is a set of differences from the network as loaded (`app_data["net_loaded"]`),
always applied to it, so it can be changed and reset freely; changing it rebuilds
the base, the capacities and the admittances, re-checks the proposals (those that
no longer apply are dropped and reported) and is saved with the load power factor
in `systems/assumptions.json`, keyed to the network as loaded (restored before the
proposals at startup). Defaults reproduce the previous behaviour: generators with
active power at pf 1.0 (Q = 0), condensers on voltage control. `gen_to_sgen` keeps
a converted unit's set-point and reactive limits on its row (`orig_*`) so it can go
back to voltage control. In the OPF every voltage-controlled unit keeps its own Q
limits (they used to get the power-factor cone whenever they had active power).
Networks loaded without a `gen` table (measured substations, rebuilt every tick)
are not editable.

**Unit costs** (`costs.py`, `GET/POST /api/costs`): per unit and external grid,
the production cost c0 + c1·P + c2·P² (`cost_eur_per_h`, `cost_eur_per_mwh`,
`cost_eur_per_mw2h`; the grid's c1 is the import price), the redispatch costs
`up_eur_per_mwh` / `down_eur_per_mwh`, a reactive price per unit
(`q_eur_per_mvarh`, symmetric: injecting or absorbing) and, for the external grid
only, the TSO–DSO reactive charge — exchange within `q_band_tan_phi`
(|Q| ≤ tan φ·|P|) free, `q_penalty_eur_per_mvarh` beyond it (synthetic 3 €/Mvarh,
tan φ 0.4, 10 €/Mvarh). Defaults come from the case's `poly_cost`
(a MATPOWER/PGLib `gencost`), else a documented synthetic rule (`costs.RULE`);
redispatch costs are synthetic until changed. Every figure carries its source
(`case` / `synthetic` / `operator`). The operator's changes are a difference from
the defaults, saved with the other assumptions. Read from the network as loaded,
keyed by unit name. `gen_to_sgen` now moves a converted unit's `poly_cost` row with
it and renumbers the generators kept (before, case14's 23.27 €/MWh sat on the
bus-2 condenser). No OPF uses the costs yet: they are kept for the economic
objective of the flexibility work (B.6 step 3).

**Names** (`naming.py`, `GET/POST /api/names`): the operator renames buses, lines,
transformers, loads and units; every result uses the new name. A layer over the
network as loaded like the generator model (applied after it; saved with the
assumptions). What must not follow a rename stays on the row: `loaded_name` (a
bus's measurements are matched by it — `element_names.measurement_name` — and the
network fingerprint reads it, so saved proposals and assumptions stay this
network's) and `loaded_unit` (a unit's model, costs and nominal values are keyed by
it — `generators.unit_key`; the generator and cost endpoints translate names both
ways). Rebuilding the base refreshes the capacities (`PG_MAX_DATA`). Names are
unique per kind; `|`, `[`, `]` are reserved (they structure branch labels). Units of
a measured-substation network (rebuilt every tick) cannot be renamed.
`/api/grid_constants` gives the agent `network_id` (the network as loaded), so a
rename does not reset the conversation, and `names_key` (also in
`/api/assumptions`). Results echo the generator model both ways
(`changed_units` under the names now, `changed_units_as_loaded` as the settings
are kept). `/api/data/series` adds `renamed` ({data row name: bus name now}); the
measurement rows themselves keep the names they were filed under. A load's
label is its own name, else its bus as named now (its `substation_name` is the
measurement row it reads, not a name).

## Endpoints

| Group | Endpoints |
|---|---|
| **Scenario** | `GET /api/scenario`, `POST /api/scenario` (`resume\|ieee14\|ieee14_reactive`) |
| **State / clock** | `GET /api/grid_constants`, `GET /api/time/current`, `GET /api/time/timeline`, `POST /api/time/advance` |
| **Upload** | `POST /api/network/upload`, `POST /api/data/upload` (`kind=measurements\|forecasts`), `GET /api/data/template` (a CSV for the loaded network: substation names, the first hour as example values, the optional reactive columns), `GET /api/data/quality`, `POST /api/network/snapshot` |
| **Security** | `POST /api/grid/rsa`, `POST /api/rsa/worst_case`, `POST /api/rsa/scan_scenarios`, `POST /api/rsa/probabilistic`, `POST /api/rsa/historical_risk`, `POST /api/grid/element_timeseries` |
| **Contingency (N-1)** | `POST /api/contingency/simulate`, `POST /api/contingency/simulate_all`, `POST /api/contingency/optimize` |
| **Flexibility / OPF** | `POST /api/flexibility/optimize`, `POST /api/flexibility/robust`, `POST /api/flexibility/envelope`, `POST /api/flexibility/hosting_capacity` |
| **KPIs** | `POST /api/kpi/evaluate`, `POST /api/kpi/forecast` |
| **Network proposals** | `GET /api/proposals`, `POST /api/proposals/edit`, `DELETE /api/proposals/{id}`, `DELETE /api/proposals/{id}/edits/{n}`, `POST /api/proposals/compare` (`study=rsa\|n1\|opf\|scan` — `opf`: corrective dispatch on both; `scan`: both networks at every tick of a window; `against`: another proposal as the reference; `proposals`: up to 6 against one reference at one tick, with a ranking `table`) |
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
