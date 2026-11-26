#!/usr/bin/env python
"""synthetic_timeseries.py — Generate synthetic measurement timeseries for any pandapower network.

Public API
----------
generate(net, n_days, resolution_min, load_profile, generation_profile, seed, stress_events)
    -> (timestamps: list[str], measurements: dict[str, pd.DataFrame])
forecast(net, timestamps, measurements, ...) -> the same, for the forecast
basis(net) -> what the generator reads from a network, kept to generate again later
normalise_settings(changes, base) / describe(settings) / preview(basis, settings) — the
    settings an operator can change, checked, in words, and their system totals

The output format is identical to the historical measurement_database.pkl schema so
all existing main_backend.py endpoints work unchanged:
    measurements[timestamp_str] = pd.DataFrame({
        "substation_name": [...],   # bus name (e.g. "Bus_3", "1", "Foxtrot")
        "production":      [...],   # total generator output at that bus in MW
        "consumption":     [...],   # total load at that bus in MW
    })

Profile options
---------------
load_profile  : "residential" | "twin_peak" | "industrial" | "flat" (LOAD_PROFILES)
generation_profile : "wind" | "solar" | "flat" | "dispatched"

Stress events
-------------
If stress_events=True, a mild 2-hour high-load / low-generation window (load
×1.15, generation ×0.97) is injected on about one day in three. It makes a
stressed hour likely; it does not guarantee a violation.
"""

from __future__ import annotations

from dataclasses import dataclass
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
        # Night trough ~3–4 h, morning peak ~8 h, a higher evening peak ~19 h.
        # (Until October 2026 it drew what is now `twin_peak`.)
        def bump(centre, width):
            d = (hours - centre + 12.0) % 24.0 - 12.0          # wraps around midnight
            return np.exp(-0.5 * (d / width) ** 2)
        alpha = 0.55 + 0.25 * bump(8.0, 1.6) + 0.15 * bump(13.0, 3.0) + 0.42 * bump(19.0, 2.0)
        return alpha * (1.0 - 0.15 * (dow >= 5).astype(float))
    if profile == "twin_peak":
        # Peaks ~2–3 h and ~14–15 h — what "residential" drew until October
        # 2026. The PGLib 14-bus profile keeps it, so its demo week (and the
        # evaluation written for it) stays exactly as it was.
        alpha = (0.6
                 + 0.4 * np.sin(np.pi * (hours - 7.0) / 12.0) ** 2
                 + 0.3 * np.sin(np.pi * (hours - 18.0) / 6.0) ** 2)
        # Weekend (Sat=5, Sun=6): 15% lower
        return alpha * (1.0 - 0.15 * (dow >= 5).astype(float))
    if profile == "industrial":
        alpha = np.full(n, 0.85)
        alpha[(hours < 6) | (hours >= 22)] *= 0.70
        return alpha
    if profile == "commercial":
        # Offices and shops: on from ~8 h to ~18 h on weekdays, a little at weekends, low at night.
        on = 1.0 / (1.0 + np.exp(-2.0 * (hours - 7.5))) / (1.0 + np.exp(2.0 * (hours - 18.5)))
        weekend = (dow >= 5).astype(float)
        return 0.4 + on * (0.6 * (1.0 - weekend) + 0.15 * weekend)
    return np.ones(n)  # "flat"


_NOISE = {"residential": 0.05, "twin_peak": 0.05, "industrial": 0.03, "commercial": 0.04}


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
    elif profile == "dispatched":
        alpha = np.full(n_steps, 0.60)       # set from the demand once it is drawn (_draw); no random draw here
    else:  # "flat"
        alpha = np.full(n_steps, 0.60)
        alpha *= rng.uniform(0.98, 1.02, n_steps)
    return alpha


# The share of the demand the dispatched units cover at every moment; the external grid (or the reference unit)
# takes the rest.
DISPATCH_SHARE = 0.95


# ---------------------------------------------------------------------------
# Settings — what the operator can change (the Data tab's synthetic-data panel)
# ---------------------------------------------------------------------------

LOAD_PROFILES = {
    "residential": "Residential: a morning peak and a higher evening peak, 15 % lower at weekends",
    "twin_peak": "Twin peak: peaks around 2–3 h and 14–15 h, 15 % lower at weekends",
    "industrial": "Industrial: steady by day, 30 % lower from 22 h to 6 h, every day",
    "commercial": "Commercial: office hours on weekdays, low at night and at weekends",
    "flat": "Flat: the same all day, with small noise",
}
# The shape a load follows when the CIM layer says what kind of demand it is (`semantics.LOAD_CATEGORIES`).
# A category not here (a downstream network, other, unknown) follows the shape chosen for the rest.
CATEGORY_SHAPES = {"residential": "residential", "rural": "residential", "heat_pump": "residential",
                   "ev_charging": "residential", "commercial": "commercial", "commercial_industrial": "commercial",
                   "agricultural": "commercial", "industrial": "industrial", "data_centre": "flat"}
GENERATION_PROFILES = {
    "wind": "Wind: wanders slowly, day and night",
    "solar": "Solar: follows the sun, nothing at night",
    "flat": "Flat: steady at 60 % of capacity",
    "dispatched": "Dispatched: follows the demand, covering 95 % of it (a transmission system's units)",
}
RESOLUTIONS = (5, 10, 15, 30, 60)
MAX_DAYS = 31
MAX_TICKS = 2976            # 31 days at 15 minutes, 10 days at 5
PEAK_RANGE = (0.3, 2.0)
LOAD_ERROR_MAX = 0.5
GENERATION_ERROR_MAX = 1.0
# The forecast's seed is the measurements' plus this: 42 gives 137, the
# forecast's seed before the settings could be changed.
FORECAST_SEED_OFFSET = 95

DEFAULTS = {
    "n_days": 7,
    "resolution_min": 15,
    "load_profile": "residential",
    "generation_profile": "wind",
    "peak_load_factor": 1.0,
    "stress_events": True,
    "seed": 42,
    "load_error": 0.04,
    "generation_error": 0.20,
    "forecast_overlap_days": 2.0,
}


class SettingsError(ValueError):
    """Settings the generator cannot use; the message says which and why."""


def normalise_settings(changes: dict | None, base: dict | None = None) -> dict:
    """`base` (default `DEFAULTS`) with `changes` applied, checked: a full
    settings dict. Raises SettingsError naming the first setting that is wrong."""
    out = {**DEFAULTS, **(base or {})}
    for key, value in (changes or {}).items():
        if key not in DEFAULTS:
            raise SettingsError(f"Unknown setting {key!r}. Known: {', '.join(DEFAULTS)}.")
        out[key] = value

    def number(key, kind, lo, hi):
        value = out[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise SettingsError(f"{key} must be a number, not {value!r}.")
        if kind is int and float(value) != int(value):
            raise SettingsError(f"{key} must be a whole number, not {value!r}.")
        value = kind(value)
        if not lo <= value <= hi:
            raise SettingsError(f"{key} must be between {lo:g} and {hi:g}, not {value:g}.")
        out[key] = value

    number("n_days", int, 1, MAX_DAYS)
    number("resolution_min", int, 1, 60)
    if out["resolution_min"] not in RESOLUTIONS:
        raise SettingsError(f"resolution_min must be one of {', '.join(map(str, RESOLUTIONS))} minutes.")
    ticks = out["n_days"] * 24 * 60 // out["resolution_min"]
    if ticks > MAX_TICKS:
        raise SettingsError(f"{out['n_days']} days at {out['resolution_min']} minutes is {ticks:,} ticks; at most "
                            f"{MAX_TICKS:,} (31 days at 15 minutes, 10 days at 5).")
    for key, known in (("load_profile", LOAD_PROFILES), ("generation_profile", GENERATION_PROFILES)):
        if out[key] not in known:
            raise SettingsError(f"{key} must be one of {', '.join(known)}, not {out[key]!r}.")
    number("peak_load_factor", float, *PEAK_RANGE)
    if not isinstance(out["stress_events"], bool):
        raise SettingsError(f"stress_events must be true or false, not {out['stress_events']!r}.")
    number("seed", int, 0, 2**31 - 1)
    number("load_error", float, 0.0, LOAD_ERROR_MAX)
    number("generation_error", float, 0.0, GENERATION_ERROR_MAX)
    number("forecast_overlap_days", float, 0.0, float(MAX_DAYS))
    out["forecast_overlap_days"] = min(out["forecast_overlap_days"], float(out["n_days"]))
    return out


def describe(settings: dict) -> str:
    """The settings in one sentence, for the operator and the agent."""
    s = settings
    shape = s["load_profile"].replace("_", " ")
    return (f"{s['n_days']} day{'s' if s['n_days'] != 1 else ''} at {s['resolution_min']}-minute steps; "
            f"{shape} load, its usual daily peak at {s['peak_load_factor'] * 100:.0f} % of the network's own; "
            f"units the network does not describe follow a {s['generation_profile']} shape; "
            f"stress windows {'on' if s['stress_events'] else 'off'}; forecast errors about "
            f"{s['load_error'] * 100:.0f} % (load) and {s['generation_error'] * 100:.0f} % (generation); "
            f"seed {s['seed']}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Basis:
    """What the generator reads from a network: each unit's bus and capacity,
    each load's bus and demand, and the buses' names. Taken when the network is
    loaded, before a tick overwrites the loads, so data can be generated again
    later from the network's own values."""
    units: tuple          # ((table, index, bus, capacity MW), ...)
    loads: tuple          # ((bus, MW, index), ...)
    names: dict           # {bus: the name its data are filed under}


def basis(net: pp.pandapowerNet) -> Basis:
    """The network's `Basis` (see the class)."""
    units = tuple((table, int(idx), int(row["bus"]), _get_max_ref(row))
                  for table in ("gen", "sgen") for idx, row in net[table].iterrows())
    loads = tuple((int(row["bus"]), max(float(row["p_mw"]), 0.0), int(idx)) for idx, row in net.load.iterrows())
    buses = {u[2] for u in units} | {entry[0] for entry in loads}
    return Basis(units=units, loads=loads, names={b: _bus_name(net, b) for b in buses})


def _draw(b: Basis, n_days, resolution_min, load_profile, generation_profile, seed, stress_events,
          start_dt, peak_load_factor, unit_shapes, load_shapes=None):
    """The series behind `generate`: (timestamps, {load shape: scale per
    tick}, {unit shape: capacity factor per tick}, {unit shape: {bus: MW}},
    {load shape: {bus: MW of load}}, buses). None as the buses when the
    network has no load and no unit."""
    rng = np.random.default_rng(seed)
    n_steps = int(n_days * 24 * 60 / resolution_min)

    # --- Timestamps and calendar arrays ---
    dts = [start_dt + timedelta(minutes=t * resolution_min) for t in range(n_steps)]
    timestamps = [dt.strftime("%Y-%m-%d %H:%M:%S") for dt in dts]
    hours = np.array([dt.hour + dt.minute / 60.0 for dt in dts])
    dow   = np.array([dt.weekday() for dt in dts])

    # --- Base values per bus (use max_p_mw as capacity reference for gen) ---
    # Capacity per bus and per shape: {shape: {bus: MW}}.
    unit_shapes = unit_shapes or {}
    cap_by_shape: dict[str, dict[int, float]] = {}
    bus_max_gen: dict[int, float] = {}
    for table, idx, bus_id, ref in b.units:
        shape = unit_shapes.get((table, idx), generation_profile)
        cap_by_shape.setdefault(shape, {})
        cap_by_shape[shape][bus_id] = cap_by_shape[shape].get(bus_id, 0.0) + ref
        bus_max_gen[bus_id] = bus_max_gen.get(bus_id, 0.0) + ref

    bus_base_load: dict[int, float] = {}
    load_by_shape: dict[str, dict[int, float]] = {}
    load_shapes = load_shapes or {}
    for bus_id, p_mw, *rest in b.loads:
        bus_base_load[bus_id] = bus_base_load.get(bus_id, 0.0) + p_mw
        shape = load_shapes.get(rest[0], load_profile) if rest else load_profile
        load_by_shape.setdefault(shape, {})
        load_by_shape[shape][bus_id] = load_by_shape[shape].get(bus_id, 0.0) + p_mw

    all_buses = sorted(set(bus_max_gen.keys()) | set(bus_base_load.keys()))
    if not all_buses:
        return timestamps, None, None, cap_by_shape, load_by_shape, None

    # --- Generate profile timeseries ---
    load_alpha = _load_scale(hours, dow, load_profile, rng)
    gen_alpha  = _gen_scale(generation_profile, n_steps, hours, rng)
    # Other shapes only when a unit needs one (drawn after the default's, so a
    # network whose units all follow the default gets the same week as before).
    alphas = {generation_profile: gen_alpha}
    for shape in sorted(set(cap_by_shape) - {generation_profile}):
        alphas[shape] = _gen_scale(shape, n_steps, hours, rng)
    if peak_load_factor is not None:
        load_alpha = load_alpha * (float(peak_load_factor) / _shape_peak(load_profile))
    load_alphas = {load_profile: load_alpha}
    if "dispatched" in alphas:
        # Units dispatched to the demand: together they follow it, each in proportion to its capacity.
        capacity = sum(cap_by_shape.get("dispatched", {}).values())
        demand = sum(bus_base_load.values())
        alphas["dispatched"][:] = np.clip(load_alpha * demand * DISPATCH_SHARE / max(capacity, 1e-9), 0.0, 1.0)

    # --- Stress events: occasional, mild 2-hour high-load / low-gen windows ---
    # The twin should read as *mostly secure*, with the odd stressed window —
    # not a guaranteed daily violation. So events fire on a minority of days
    # (~1 in 3) with a gentle load bump, rather than a hard ×1.3 every day.
    windows = []
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
                windows.append((peak, width))
                load_alpha[peak:peak + width] = np.clip(
                    load_alpha[peak:peak + width] * 1.15, 0.0, 1.6
                )
                for alpha in alphas.values():
                    alpha[peak:peak + width] = np.clip(alpha[peak:peak + width] * 0.97, 0.0, 1.0)
    # Loads of another shape (their category says so), drawn last of all — after the stress windows, which then
    # fall on them too — so a network whose loads all follow the chosen shape gets the same week as before, and
    # one load's category leaves every other bus as it was.
    for shape in sorted(set(load_by_shape) - {load_profile}):
        alpha = _load_scale(hours, dow, shape, rng)
        if peak_load_factor is not None:
            alpha = alpha * (float(peak_load_factor) / _shape_peak(shape))
        for peak, width in windows:
            alpha[peak:peak + width] = np.clip(alpha[peak:peak + width] * 1.15, 0.0, 1.6)
        load_alphas[shape] = alpha
    return timestamps, load_alphas, alphas, cap_by_shape, load_by_shape, all_buses


def generate(
    net: pp.pandapowerNet | Basis,
    n_days: int = 7,
    resolution_min: int = 15,
    load_profile: str = "residential",
    generation_profile: str = "wind",
    seed: int = 42,
    stress_events: bool = True,
    start_dt: datetime = _ORIGIN,
    peak_load_factor: float | None = 1.0,
    unit_shapes: dict | None = None,
    load_shapes: dict | None = None,
) -> tuple[list[str], dict[str, pd.DataFrame]]:
    """Generate a synthetic measurement timeseries from a static pandapower network.

    Parameters
    ----------
    net : pp.pandapowerNet or Basis
        The base network (not modified), or its `basis` taken earlier.
    n_days : int
        Number of simulated days.
    resolution_min : int
        Timestep in minutes (e.g. 15).
    load_profile : str
        Shape of load scaling: one of `LOAD_PROFILES`.
    generation_profile : str
        Shape of generation scaling: ``"wind"``, ``"solar"``, or ``"flat"``.
    seed : int
        RNG seed for reproducibility.
    stress_events : bool
        If True, inject a mild 2-hour high-load / low-generation window on
        about one day in three (load ×1.15, generation ×0.97). Violations are
        likely in it, not guaranteed.
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
    unit_shapes : dict or None
        {(table, index): "wind" | "solar" | "flat"} — the shape a unit follows
        when the network says what drives it (a PV unit follows the sun, not
        the wind). Units not in it follow ``generation_profile``; ``None``
        (the default) gives every unit that shape, as before.
    load_shapes : dict or None
        {load index: one of `LOAD_PROFILES`} — the shape a load follows when
        the network says what kind of demand it is (`CATEGORY_SHAPES`). Loads
        not in it follow ``load_profile``.

    Returns
    -------
    timestamps : list[str]
        ISO-8601 timestamp strings, starting at "2000-01-01 00:00:00".
    measurements : dict[str, pd.DataFrame]
        Maps each timestamp string to a DataFrame with columns
        ``substation_name``, ``production`` (MW), ``consumption`` (MW).
        One row per bus that has load or generation.
    """
    b = net if isinstance(net, Basis) else basis(net)
    timestamps, load_alphas, alphas, cap_by_shape, load_by_shape, all_buses = _draw(
        b, n_days, resolution_min, load_profile, generation_profile, seed, stress_events, start_dt,
        peak_load_factor, unit_shapes, load_shapes)
    if all_buses is None:
        # Degenerate case: no loads or generators — return single flat timestamp
        return [timestamps[0]], {timestamps[0]: pd.DataFrame(
            [{"substation_name": "EXT_GRID", "production": 0.0, "consumption": 0.0}]
        )}

    # --- Assemble measurement DataFrames ---
    measurements: dict[str, pd.DataFrame] = {}
    for t, ts_str in enumerate(timestamps):
        rows = [
            {
                "substation_name": b.names[bus_id],
                "production":  float(sum(cap.get(bus_id, 0.0) * alphas[shape][t]
                                         for shape, cap in cap_by_shape.items())),
                "consumption": float(sum(base.get(bus_id, 0.0) * load_alphas[shape][t]
                                         for shape, base in load_by_shape.items())),
            }
            for bus_id in all_buses
        ]
        measurements[ts_str] = pd.DataFrame(rows)

    return timestamps, measurements


def preview(b: Basis, settings: dict, unit_shapes: dict | None = None, load_shapes: dict | None = None) -> dict:
    """The system totals `generate` would give with `settings` (normalised),
    without building the per-bus tables: fast enough to redraw on every change.
    {"timestamps", "consumption", "production": MW per tick, "production_by_shape":
    {shape: MW per tick}}."""
    s = settings
    timestamps, load_alphas, alphas, cap_by_shape, load_by_shape, all_buses = _draw(
        b, s["n_days"], s["resolution_min"], s["load_profile"], s["generation_profile"], s["seed"],
        s["stress_events"], _ORIGIN, s["peak_load_factor"], unit_shapes, load_shapes)
    if all_buses is None:
        return {"timestamps": timestamps[:1], "consumption": [0.0], "production": [0.0], "production_by_shape": {}}
    by_shape = {shape: alphas[shape] * sum(cap.values()) for shape, cap in cap_by_shape.items()}
    production = sum(by_shape.values()) if by_shape else np.zeros(len(timestamps))
    consumption = sum(load_alphas[shape] * sum(base.values()) for shape, base in load_by_shape.items())
    return {"timestamps": timestamps,
            "consumption": [round(float(v), 4) for v in consumption],
            "production": [round(float(v), 4) for v in production],
            "production_by_shape": {shape: [round(float(v), 4) for v in series] for shape, series in by_shape.items()}}


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
    unit_shapes: dict | None = None,
    stress_events: bool = True,
    load_shapes: dict | None = None,
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
            peak_load_factor=peak_load_factor, unit_shapes=unit_shapes, stress_events=stress_events,
            load_shapes=load_shapes,
        )
        out.update(ahead)
    return list(overlap) + ahead_ts, out
