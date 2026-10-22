"""
Starting scenarios: what CONDUCTOR loads — the operator's own work, or a demo.

- ``resume`` — the last uploaded network and data (whatever the "last
  uploaded" records restore). Demos never touch those records, so it stays
  available whatever is tried in between.
- ``ieee30`` (the default) — PGLib's IEEE 30-bus case (Alsac–Stott): five
  dispatchable units with their own costs, on voltage control, and line
  ratings; a synthetic week of active power in which the line from the
  external grid congests at the evening peaks. Loads take their reactive
  power from the load power factor. Its nominal operating point is included.
- ``ieee30_reactive`` — the same network and week, with measured reactive
  demand and generation (`demo_reactive_csv`), a few values left out on
  purpose so the data-quality report shows how gaps are handled.
- ``ieee14`` — the smaller PGLib IEEE 14-bus case and its week; one unit can
  move its active power. The what-if evaluation is written for it.

The demo data are generated deterministically (fixed seeds), so a demo is the
same starting point every time — evaluation runs can name one.

Pure functions: no app state, no HTTP.
"""

from __future__ import annotations

import math

import numpy as np

DEMO_PROFILE = "pglib_case30"
SMALL_DEMO_PROFILE = "pglib_case14"
DEFAULT_DEMO = "ieee30"
SCENARIOS = {
    "resume": {
        "label": "Resume",
        "description": "Your last uploaded network and data.",
    },
    "ieee30": {
        "label": "Demo: IEEE 30 · active only",
        "description": "PGLib IEEE 30-bus: five dispatchable units with their own costs, and a synthetic week of "
                       "active power in which the line from the external grid congests at the evening peaks. "
                       "No reactive measurements: loads use the power factor setting. Its nominal operating "
                       "point is included.",
        "profile": DEMO_PROFILE, "reactive": False,
    },
    "ieee30_reactive": {
        "label": "Demo: IEEE 30 · with reactive measurements",
        "description": "The same network and week, with measured reactive demand and generation — a few values "
                       "left out on purpose, so the data-quality report shows how gaps are handled.",
        "profile": DEMO_PROFILE, "reactive": True,
    },
    "ieee14": {
        "label": "Demo: IEEE 14 · smaller",
        "description": "PGLib IEEE 14-bus with a synthetic week of active power: one unit can move its active "
                       "power, three condensers hold voltages. The what-if evaluation is written for it.",
        "profile": SMALL_DEMO_PROFILE, "reactive": False,
    },
}
DEMOS = tuple(k for k, v in SCENARIOS.items() if "profile" in v)


def demo_for_profile(profile: str) -> str:
    """The demo a profile loads as (its active-power one); the default demo
    for a profile no demo uses."""
    return next((k for k, v in SCENARIOS.items() if v.get("profile") == profile and not v.get("reactive")),
                DEFAULT_DEMO)


def demo_reactive_csv(measurements: dict, seed: int = 7) -> bytes:
    """The demo week with reactive power, as an upload CSV.

    Each substation draws reactive demand at its own power factor (0.88–0.98,
    lagging), drifting a little over the day; the smallest one is leading (a
    cable-rich area); reactive generation follows production at pf 0.95. About
    1 % of reactive cells are left empty, plus one substation for three hours —
    gaps the power factor setting then fills.
    """
    rng = np.random.default_rng(seed)
    timestamps = sorted(measurements)
    first = measurements[timestamps[0]]
    names = sorted(first["substation_name"].astype(str).unique())
    pf_of = {n: float(rng.uniform(0.88, 0.98)) for n in names}
    mean_load = {n: float(np.mean([measurements[t].set_index("substation_name").at[n, "consumption"]
                                   for t in timestamps[:96]])) for n in names}
    loaded = [n for n in names if mean_load[n] > 1e-6]
    leading = min(loaded, key=lambda n: mean_load[n]) if loaded else None
    gap_sub = max(loaded, key=lambda n: mean_load[n]) if loaded else None
    gap_ticks = set(timestamps[200:212])                        # three hours on day 3
    rows = ["timestamp,substation_name,consumption_mw,production_mw,consumption_mvar,production_mvar"]
    for k, ts in enumerate(timestamps):
        hour = int(ts[11:13]) + int(ts[14:16]) / 60.0
        drift = 0.015 * math.sin(2 * math.pi * (hour - 14) / 24)        # worse pf in the afternoon
        frame = measurements[ts].set_index("substation_name")
        for n in names:
            p = float(frame.at[n, "consumption"])
            g = float(frame.at[n, "production"])
            pf = min(max(pf_of[n] - drift, 0.8), 0.995)
            q = p * math.tan(math.acos(pf))
            if n == leading:
                q = -0.15 * p
            missing = rng.random() < 0.01 or (n == gap_sub and ts in gap_ticks)
            qc = "" if missing or p <= 1e-9 else f"{q:.4f}"
            qg = f"{g * math.tan(math.acos(0.95)):.4f}" if g > 1e-9 else ""
            rows.append(f"{ts},{n},{p:.6f},{g:.6f},{qc},{qg}")
    return ("\n".join(rows) + "\n").encode()


def resume_detail(network_sentinel: dict | None, timeseries_sentinel: dict | None) -> dict:
    """What Resume would load, from the "last uploaded" records."""
    import os

    def name(rec, key):
        path = (rec or {}).get(key) or ""
        return os.path.basename(path) if path else None

    return {"network": name(network_sentinel, "network_path"),
            "data": name(timeseries_sentinel, "csv_path"),
            "available": bool(network_sentinel or timeseries_sentinel)}
