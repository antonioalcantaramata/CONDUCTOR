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
- ``smallgrid`` — ENTSO-E's CGMES SmallGrid test model (CC BY-SA 4.0, ENTSO-E;
  ``backend/network_data/entsoe_smallgrid/``), loaded as a CIM model and kept
  whole: 118 nodes at 220, 132 and 33 kV, 18 machines holding the model's own
  voltage set-points, three boundary equivalents. A synthetic week in which the
  units follow the demand and the 220 kV corridor Olive–X9–Breed congests at the
  evening peaks, with measured reactive demand and generation
  (`demo_reactive_csv`), a few values left out on purpose so the data-quality
  report shows how gaps are handled.
- ``cigre_mv`` — pandapower's CIGRE medium-voltage benchmark (110/20 kV, two
  radial feeders with normally open ties) with all its distributed resources —
  PV, wind, CHP, fuel cells, batteries — each typed by the network, so its
  units' sources are stated, not guessed; a synthetic week in which the
  feeder ends sag below 0.95 p.u. at the evening peaks. The what-if
  evaluation is written for it.

The demo data are generated deterministically (fixed seeds), so a demo is the
same starting point every time — evaluation runs can name one.

Pure functions: no app state, no HTTP.
"""

from __future__ import annotations

import math
import re

import numpy as np

DEMO_PROFILE = "pglib_case30"
DISTRIBUTION_DEMO_PROFILE = "cigre_mv"
CIM_DEMO_PROFILE = "entsoe_smallgrid"
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
    "smallgrid": {
        "label": "Demo: ENTSO-E SmallGrid · a CIM model",
        "description": "ENTSO-E's CGMES SmallGrid test model (CC BY-SA 4.0, © ENTSO-E), loaded as CIM and kept "
                       "whole: 118 nodes at 220/132/33 kV, 18 machines, boundary equivalents. A synthetic week in "
                       "which the 220 kV corridor Olive–X9–Breed congests at the evening peaks, with measured "
                       "reactive power — a few values left out on purpose, so the data-quality report shows how "
                       "gaps are handled.",
        "profile": CIM_DEMO_PROFILE, "reactive": True,
    },
    "cigre_mv": {
        "label": "Demo: CIGRE MV · distribution with DER",
        "description": "The CIGRE medium-voltage benchmark: a 110/20 kV substation and two radial feeders with "
                       "PV, wind, CHP, fuel cells and batteries, each typed by the network. A synthetic week in "
                       "which the feeder ends sag below 0.95 p.u. at the evening peaks.",
        "profile": DISTRIBUTION_DEMO_PROFILE, "reactive": False,
    },
}
DEMOS = tuple(k for k, v in SCENARIOS.items() if "profile" in v)


def demo_for_profile(profile: str) -> str | None:
    """The demo a profile loads as (its active-power one); None for a profile
    no demo uses (`GRID_PROFILE=<your own>`). Calling it the default demo filed
    its saved changes under that demo, and the next start loaded the demo
    instead of the profile."""
    same = [k for k, v in SCENARIOS.items() if v.get("profile") == profile]
    return next((k for k in same if not SCENARIOS[k].get("reactive")), same[0] if same else None)


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
        """The file as the operator named it: stored copies carry a prefix ("uploaded_system__grid.m"), and
        data generated for an upload are a synthetic week, not a file of theirs."""
        path = (rec or {}).get(key) or ""
        if not path:
            return None
        base = os.path.basename(path)
        if key == "csv_path" and re.fullmatch(r"(measurements|forecasts?)_synthetic[^/]*\.csv", base):
            return "a synthetic week"
        return base.split("__", 1)[1] if "__" in base else base

    return {"network": name(network_sentinel, "network_path"),
            "data": name(timeseries_sentinel, "csv_path"),
            "available": bool(network_sentinel or timeseries_sentinel)}
