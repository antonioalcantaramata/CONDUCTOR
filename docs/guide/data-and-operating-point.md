# Data and the operating point

Every study runs on the network **at one operating point**: how much each load
consumes and each unit produces. That comes from one of three places.

| Source | What it is | Time |
| --- | --- | --- |
| **Measurements** | Historical actuals, one row per substation and time step | Drives the simulation clock |
| **Forecasts** | Look-ahead values in the same format, for planning | Read-only; does not move the clock |
| **Nominal point** | The network file's own loads and generation | No time at all |

## Choosing the operating point

In the sidebar's **Operating point** card:

- **Measured / Forecast / Compare** chooses the dataset. *Compare* shows both
  and how far the forecast was off where they overlap.
- The **date, time and slider** choose the moment. Studies use it unless your
  question names another time ("at 18:00 on Tuesday").
- **Nominal point** switches to the network's own operating point. The clock
  gives way to "Nominal" and CONDUCTOR studies that point. It is only offered
  when the network file has complete values.

You can also simply ask: *"…using the forecast for tomorrow 08:00"*,
*"…at the nominal operating point"*. CONDUCTOR says in its answer which data
and moment it used.

## Uploading a network

**Upload Network** in the header accepts:

| Format | Notes |
| --- | --- |
| `.m` | MATPOWER case file: IEEE cases, pglib-opf, or any compatible model |
| `.json` | pandapower JSON, exported with `pp.to_json(net, "file.json")` |
| `.xlsx` | pandapower Excel, exported with `pp.to_excel(net, "file.xlsx")` |
| `.uct` | UCTE exchange format (ENTSO-E) |

Two choices in the upload dialog:

- **Start generators at a fixed power factor (pf 1.0)** — ticked, units with
  active power start as fixed-power-factor units; unticked, they keep voltage
  control. Either way you can change each unit afterwards in the
  [System tab](system-tab.md#generators).
- **Transformer tap policy** — keep the taps in the file, or force neutral taps.

If the file has generation costs (a MATPOWER `gencost`, or a pandapower
`poly_cost` table), they become the units' production costs.

An uploaded network becomes what **Resume** loads. Demos never replace it.

## Uploading data

**Upload Data** takes a CSV for **measurements** or **forecasts**, in long
format: one row per time step and substation.

| Column | Required | Meaning |
| --- | --- | --- |
| `timestamp` | yes | Any date-time pandas can read, e.g. `2024-01-01 00:15:00`. Any regular interval; 15 minutes recommended |
| `substation_name` | yes | A bus name of the loaded network. Partial or prefix names are matched |
| `production_mw` | yes | Total generation at that bus, MW (`0.0` for load-only buses) |
| `consumption_mw` | yes | Total load at that bus, MW (`0.0` for generation-only buses) |
| `consumption_mvar` | no | Reactive demand, Mvar (positive draws, negative injects) |
| `production_mvar` | no | Reactive generation, Mvar, same sign as `production_mw` |

```
timestamp,substation_name,production_mw,consumption_mw
2024-01-01 00:00:00,Bus 1,2.5,1.2
2024-01-01 00:00:00,Bus 2,0.0,3.4
2024-01-01 00:15:00,Bus 1,2.3,1.4
2024-01-01 00:15:00,Bus 2,0.0,3.6
```

**Download a template for this network** in the upload dialog gives a CSV with
every substation name the loaded network expects, example values, and the
optional reactive columns. Fill it in and upload it back.

Leave a cell empty when a value was not measured. Missing active power is
interpolated in time, and the data-quality report says where.

## Reactive power

If your data carry **active power only**, each load's reactive power comes from
the **load power factor** — 0.95 lagging by default, changeable in the
[System tab](system-tab.md#load-power-factor). Any single study can use another
value: *"rerun at power factor 0.90"*.

If your data include `consumption_mvar` / `production_mvar`, the measured values
are used wherever they are given, and the power factor only fills the gaps.

## Data quality

The **Data** tab has a **Data quality** panel for each dataset: coverage per
substation, gaps, implausible power factors, values that look like kvar instead
of Mvar, missing-value markers such as `-9999`. CONDUCTOR can read the same
report: *"What does the data-quality report say, and does any of it affect the
studies?"*

## The Data tab

Plots the loaded series: the system total or chosen substations, consumption,
production or net load (and reactive power when measured), measurements and
forecasts together with their overlap, and the nominal value as a dotted line.
A **Table** under the chart lists the values. A bus you have renamed appears as
"North (data: Bus_3)": its data stay filed under the name it was loaded with.
