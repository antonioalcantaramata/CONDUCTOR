#!/usr/bin/env python
"""synthetic_timeseries.py — Generate synthetic measurement timeseries for any pandapower network.

Public API
----------
generate(net, n_days, resolution_min, load_profile, generation_profile, seed, stress_events)
    -> (timestamps: list[str], measurements: dict[str, pd.DataFrame])

The output format is identical to the historical measurement_database.pkl schema so
all existing main_backend.py endpoints work unchanged:
    measurements[timestamp_str] = pd.DataFrame({
        "substation_name": [...],   # bus name (e.g. "Bus_3", "1", "Foxtrot")
        "production":      [...],   # total generator output at that bus in MW
        "consumption":     [...],   # total load at that bus in MW
    })

Profile options
---------------
load_profile  : "residential" | "industrial" | "flat"
generation_profile : "wind" | "solar" | "flat"

Stress events
-------------
If stress_events=True, one 2-hour high-load / low-generation window is injected
per day, ensuring the OPF agent finds at least one constraint violation window.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pandapower as pp

_ORIGIN = datetime(2000, 1, 1)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _bus_name(net: pp.pandapowerNet, bus_id: int) -> str:
    """Return bus name, falling back to 'Bus_{id}' for missing / empty names."""
    if "name" in net.bus.columns:
        raw = net.bus.at[bus_id, "name"]
        if raw is not None and pd.notna(raw) and str(raw).strip():
            return str(raw).strip()
    return f"Bus_{bus_id}"


def _get_max_ref(row: pd.Series) -> float:
    """Return the capacity reference (max_p_mw if valid, else p_mw, else 0)."""
    for col in ("max_p_mw", "p_mw"):
        if col in row.index:
            v = row[col]
            if v is not None and pd.notna(v):
                val = float(v)
                if val > 0:
                    return val
    return 0.0


def _load_shape(hours: np.ndarray, dow: np.ndarray, profile: str) -> np.ndarray:
    """The noise-free load shape of a profile."""
    n = len(hours)
    if profile == "residential":
        # Morning peak ~8 h, evening peak ~19 h
        alpha = (0.6
                 + 0.4 * np.sin(np.pi * (hours - 7.0) / 12.0) ** 2
                 + 0.3 * np.sin(np.pi * (hours - 18.0) / 6.0) ** 2)
        # Weekend (Sat=5, Sun=6): 15% lower
        return alpha * (1.0 - 0.15 * (dow >= 5).astype(float))
    if profile == "industrial":
        alpha = np.full(n, 0.85)
        alpha[(hours < 6) | (hours >= 22)] *= 0.70
        return alpha
    return np.ones(n)  # "flat"


_NOISE = {"residential": 0.05, "industrial": 0.03}


def _load_scale(hours: np.ndarray, dow: np.ndarray,
                profile: str, rng: np.random.Generator) -> np.ndarray:
    """Per-step load scale factor in [0.05, 1.5]: the shape with white noise."""
    noise = _NOISE.get(profile, 0.02)
    alpha = _load_shape(hours, dow, profile) * rng.uniform(1.0 - noise, 1.0 + noise, len(hours))
    return np.clip(alpha, 0.05, 1.5)


def _shape_peak(profile: str) -> float:
    """A profile's noise-free weekday maximum."""
    hours = np.linspace(0.0, 24.0, 2881)
    return float(_load_shape(hours, np.zeros_like(hours), profile).max())


def _gen_scale(profile: str, n_steps: int,
               hours: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Per-step generation capacity factor in [0, 1]."""
    if profile == "wind":
        # Weibull-seeded random walk
        alpha = np.empty(n_steps)
        alpha[0] = float(np.clip(rng.weibull(2) * 0.4, 0.0, 1.0))
        increments = rng.normal(0.0, 0.025, n_steps - 1)
        for t in range(1, n_steps):
            alpha[t] = alpha[t - 1] + increments[t - 1]
        alpha = np.clip(alpha, 0.0, 1.0)
    elif profile == "solar":
        # Truncated Gaussian centred at noon
        alpha = np.exp(-0.5 * ((hours - 12.0) / 3.0) ** 2)
        alpha[(hours < 5.0) | (hours > 20.0)] = 0.0
        alpha *= np.clip(rng.uniform(0.90, 1.10, n_steps), 0.0, 1.0)
        alpha = np.clip(alpha, 0.0, 1.0)
    else:  # "flat"
        alpha = np.full(n_steps, 0.60)
        alpha *= rng.uniform(0.98, 1.02, n_steps)
    return alpha


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate(
    net: pp.pandapowerNet,
    n_days: int = 7,
    resolution_min: int = 15,
    load_profile: str = "residential",
    generation_profile: str = "wind",
    seed: int = 42,
    stress_events: bool = True,
    start_dt: datetime = _ORIGIN,
    peak_load_factor: float | None = 1.0,
) -> tuple[list[str], dict[str, pd.DataFrame]]:
    """Generate a synthetic measurement timeseries from a static pandapower network.

    Parameters
    ----------
    net : pp.pandapowerNet
        The base network.  Not modified.
    n_days : int
        Number of simulated days.
    resolution_min : int
        Timestep in minutes (e.g. 15).
    load_profile : str
        Shape of load scaling: ``"residential"``, ``"industrial"``, or ``"flat"``.
    generation_profile : str
        Shape of generation scaling: ``"wind"``, ``"solar"``, or ``"flat"``.
    seed : int
        RNG seed for reproducibility.
    stress_events : bool
        If True, inject one 2-hour high-load / low-generation stress window per
        day so that constraint violations are guaranteed in the series.
    start_dt : datetime
        First timestamp of the series. Defaults to 2000-01-01. Used to place a
        forecast series on a time range immediately after the measurement series.
    peak_load_factor : float or None
        The usual daily peak as a multiple of the network's own load. A
        network's loads are its design point (a PGLib or IEEE case is built
        to be operable there), so the default 1.0 peaks at that load; stress
        windows go above it. The profile shapes alone peak higher (residential
        at ~1.22×), which pushed a test case past its reactive capability every
        afternoon. ``None`` keeps the raw shape.

    Returns
    -------
    timestamps : list[str]
        ISO-8601 timestamp strings, starting at "2000-01-01 00:00:00".
    measurements : dict[str, pd.DataFrame]
        Maps each timestamp string to a DataFrame with columns
        ``substation_name``, ``production`` (MW), ``consumption`` (MW).
        One row per bus that has load or generation.
    """
    rng = np.random.default_rng(seed)
    n_steps = int(n_days * 24 * 60 / resolution_min)

    # --- Timestamps and calendar arrays ---
    dts = [start_dt + timedelta(minutes=t * resolution_min) for t in range(n_steps)]
    timestamps = [dt.strftime("%Y-%m-%d %H:%M:%S") for dt in dts]
    hours = np.array([dt.hour + dt.minute / 60.0 for dt in dts])
    dow   = np.array([dt.weekday() for dt in dts])

    # --- Base values per bus (use max_p_mw as capacity reference for gen) ---
    bus_max_gen: dict[int, float] = {}
    for _, row in net.gen.iterrows():
        bus_id = int(row["bus"])
        bus_max_gen[bus_id] = bus_max_gen.get(bus_id, 0.0) + _get_max_ref(row)
    for _, row in net.sgen.iterrows():
        bus_id = int(row["bus"])
        bus_max_gen[bus_id] = bus_max_gen.get(bus_id, 0.0) + _get_max_ref(row)

    bus_base_load: dict[int, float] = {}
    for _, row in net.load.iterrows():
        bus_id = int(row["bus"])
        bus_base_load[bus_id] = bus_base_load.get(bus_id, 0.0) + max(float(row["p_mw"]), 0.0)

    all_buses = sorted(set(bus_max_gen.keys()) | set(bus_base_load.keys()))
    if not all_buses:
        # Degenerate case: no loads or generators — return single flat timestamp
        return [timestamps[0]], {timestamps[0]: pd.DataFrame(
            [{"substation_name": "EXT_GRID", "production": 0.0, "consumption": 0.0}]
        )}

    # --- Generate profile timeseries ---
    load_alpha = _load_scale(hours, dow, load_profile, rng)
    gen_alpha  = _gen_scale(generation_profile, n_steps, hours, rng)
    if peak_load_factor is not None:
        load_alpha = load_alpha * (float(peak_load_factor) / _shape_peak(load_profile))

    # --- Stress events: occasional, mild 2-hour high-load / low-gen windows ---
    # The twin should read as *mostly secure*, with the odd stressed window —
    # not a guaranteed daily violation. So events fire on a minority of days
    # (~1 in 3) with a gentle load bump, rather than a hard ×1.3 every day.
    if stress_events:
        steps_per_day = max(1, int(24 * 60 / resolution_min))
        width = max(1, int(2 * 60 / resolution_min))  # 2 hours worth of steps
        for d in range(n_days):
            if rng.random() > 0.35:          # only ~35% of days see a stress window
                continue
            lo = d * steps_per_day + steps_per_day // 6
            hi = (d + 1) * steps_per_day - width - 2
            if lo < hi:
                peak = int(rng.integers(lo, hi))
                load_alpha[peak:peak + width] = np.clip(
                    load_alpha[peak:peak + width] * 1.15, 0.0, 1.6
                )
                gen_alpha[peak:peak + width] = np.clip(
                    gen_alpha[peak:peak + width] * 0.97, 0.0, 1.0
                )

    # --- Assemble measurement DataFrames ---
    measurements: dict[str, pd.DataFrame] = {}
    for t, ts_str in enumerate(timestamps):
        rows = [
            {
                "substation_name": _bus_name(net, bus_id),
                "production":  float(bus_max_gen.get(bus_id, 0.0) * gen_alpha[t]),
                "consumption": float(bus_base_load.get(bus_id, 0.0) * load_alpha[t]),
            }
            for bus_id in all_buses
        ]
        measurements[ts_str] = pd.DataFrame(rows)

    return timestamps, measurements


def _ar1(n: int, sigma: float, rng: np.random.Generator, phi: float = 0.97) -> np.ndarray:
    """A slowly varying error: forecast errors persist for hours, they are not white noise."""
    e = np.empty(n)
    e[0] = rng.normal(0.0, sigma)
    step = sigma * np.sqrt(1.0 - phi ** 2)
    for t in range(1, n):
        e[t] = phi * e[t - 1] + rng.normal(0.0, step)
    return e


def forecast(
    net: pp.pandapowerNet,
    timestamps: list[str],
    measurements: dict[str, pd.DataFrame],
    overlap_days: float = 2.0,
    n_days: int = 7,
    resolution_min: int = 15,
    load_profile: str = "residential",
    generation_profile: str = "wind",
    seed: int = 137,
    peak_load_factor: float | None = 1.0,
    load_error: float = 0.04,
    generation_error: float = 0.20,
) -> tuple[list[str], dict[str, pd.DataFrame]]:
    """A synthetic forecast that overlaps the end of the measurements.

    Two parts:

    - the last ``overlap_days`` of measured ticks, forecast as the measured
      values times a slowly varying error (~``load_error`` on consumption,
      ~``generation_error`` on production — typical day-ahead sizes for load
      and wind). This is a forecast *of* those hours, so comparing it with what
      was measured means something; an independent series would not;
    - ``n_days`` after the measurements: an independent synthetic realisation
      (``generate`` with ``seed``), the look-ahead.

    ``overlap_days=0`` gives the look-ahead alone.
    """
    rng = np.random.default_rng(seed)
    steps_per_day = int(24 * 60 / resolution_min)
    n_overlap = min(len(timestamps), int(round(overlap_days * steps_per_day)))
    overlap = timestamps[len(timestamps) - n_overlap:] if n_overlap else []

    out: dict[str, pd.DataFrame] = {}
    if overlap:
        load_err = _ar1(len(overlap), load_error, rng)
        gen_err = _ar1(len(overlap), generation_error, rng)
        for t, ts in enumerate(overlap):
            df = measurements[ts].copy()
            df["consumption"] = (df["consumption"] * (1.0 + load_err[t])).clip(lower=0.0)
            df["production"] = (df["production"] * (1.0 + gen_err[t])).clip(lower=0.0)
            out[ts] = df

    ahead_ts: list[str] = []
    if n_days > 0 and timestamps:
        start = datetime.strptime(timestamps[-1], "%Y-%m-%d %H:%M:%S") + timedelta(minutes=resolution_min)
        ahead_ts, ahead = generate(
            net, n_days=n_days, resolution_min=resolution_min, load_profile=load_profile,
            generation_profile=generation_profile, seed=seed, start_dt=start,
            peak_load_factor=peak_load_factor,
        )
        out.update(ahead)
    return list(overlap) + ahead_ts, out
