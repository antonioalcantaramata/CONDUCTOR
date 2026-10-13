"""
Measured reactive power in uploaded data, and what the data look like.

The upload format has two required power columns, `consumption_mw` and
`production_mw`. Two optional columns carry reactive power:

- `consumption_mvar` — the substation's reactive demand (positive draws
  reactive power, as an inductive load; negative injects it, a leading load);
- `production_mvar` — the substation's reactive generation, with the same sign
  convention as `production_mw` (positive with positive production injects).

When a dataset carries them they are always used: a load's (or a fixed-power-
factor unit's) reactive power comes from the measurement, and the load power
factor setting applies only where a measurement is missing. A missing value
stays missing (it is never read as 0), so the fallback is visible.

`quality` checks a dataset and returns findings the operator (and CONDUCTOR)
can read — coverage, empty columns, implausible power factors, values that look
like kvar, filled-in zeros, missing-value markers. It never changes the data.

Pure functions of DataFrames: no app state, no HTTP.
"""

from __future__ import annotations

import io
import math

import numpy as np
import pandas as pd

REQUIRED = ("timestamp", "substation_name", "production_mw", "consumption_mw")
OPTIONAL_Q = {"consumption_mvar": "consumption_q", "production_mvar": "production_q"}
MISSING_BELOW = -9000.0          # e.g. -9999: a missing-value marker, not a value
LOW_PF = 0.7                     # below this, a load's implied power factor is implausible


def parse(content: bytes) -> tuple[dict, list, dict]:
    """(measurements by timestamp, sorted timestamps, parse notes).

    Each timestamp's frame has `substation_name`, `production`, `consumption`
    and, when the file carries them, `consumption_q` / `production_q` (NaN where
    missing). Raises ValueError with a readable message.
    """
    text = content.decode("utf-8-sig")  # strip BOM if Excel-exported
    df = pd.read_csv(io.StringIO(text))
    missing = set(REQUIRED) - set(df.columns)
    if missing:
        raise ValueError(
            f"CSV is missing required columns: {sorted(missing)}. Required: timestamp, substation_name, "
            "production_mw, consumption_mw (optional: consumption_mvar, production_mvar).")
    notes: dict = {"markers": {}, "empty_columns": [], "reactive_columns": [], "interpolated": {},
                   "no_values": {}}
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    # Active power that is missing (an empty cell, or a marker such as -9999) is
    # filled from the same substation's neighbouring measurements — interpolated
    # in time, nearest value at the ends. A substation with no value at all is 0.
    # All of it is reported; nothing is silently read as 0 any more.
    df = df.sort_values(["substation_name", "timestamp"], kind="stable").reset_index(drop=True)
    for col in ("production_mw", "consumption_mw"):
        values = pd.to_numeric(df[col], errors="coerce")
        marker = values < MISSING_BELOW
        if marker.any():
            notes["markers"][col] = int(marker.sum())
        values = values.where(values >= MISSING_BELOW)
        missing = values.isna()
        if missing.any():
            filled = values.groupby(df["substation_name"]).transform(
                lambda v: v.interpolate(limit_direction="both"))
            empty = sorted(df.loc[filled.isna(), "substation_name"].astype(str).unique())
            if empty:
                notes["no_values"][col] = empty
            done = missing & filled.notna()
            notes["interpolated"][col] = {"count": int(done.sum()), "markers": int((done & marker).sum())}
            values = filled.fillna(0.0)
        df[col] = values
    keep = ["substation_name", "production_mw", "consumption_mw"]
    for col, internal in OPTIONAL_Q.items():
        if col not in df.columns:
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        n = int((values < MISSING_BELOW).sum())
        if n:
            notes["markers"][col] = n
        values = values.where(values >= MISSING_BELOW)          # a marker is a missing value
        if values.notna().any():
            df[internal] = values
            keep.append(internal)
            notes["reactive_columns"].append(col)
        else:
            notes["empty_columns"].append(col)
    measurements = {
        ts: (group[keep].rename(columns={"production_mw": "production", "consumption_mw": "consumption"})
             .reset_index(drop=True))
        for ts, group in df.groupby("timestamp", sort=False)
    }
    timestamps = sorted(measurements)
    if not timestamps:
        raise ValueError("CSV contained no valid rows after parsing.")
    return measurements, timestamps, notes


def has_reactive(dataset: dict) -> dict:
    """{"consumption": bool, "production": bool} — which reactive columns a
    dataset carries."""
    first = next(iter(dataset.values()), None) if dataset else None
    cols = set(first.columns) if first is not None else set()
    return {"consumption": "consumption_q" in cols, "production": "production_q" in cols}


def _finding(level: str, code: str, message: str, **extra) -> dict:
    return {"level": level, "code": code, "message": message, **extra}


def quality(dataset: dict, notes: dict | None = None, voltage_substations: set | None = None) -> dict:
    """What a dataset looks like, as findings (level info / warning / error).

    `voltage_substations`: substations whose units control voltage (they decide
    their own reactive power, so a measured production Q is not imposed there).
    """
    notes = notes or {}
    findings: list[dict] = []
    if not dataset:
        return {"reactive": {"consumption": False, "production": False}, "findings": findings}
    frame = pd.concat([df.assign(_t=t) for t, df in dataset.items()], ignore_index=True)
    reactive = {"consumption": "consumption_q" in frame, "production": "production_q" in frame}

    for col, info in (notes.get("interpolated") or {}).items():
        n, markers = info["count"], info["markers"]
        if n:
            what = ("missing-value markers" if markers == n else "empty cells" if not markers
                    else f"{markers} missing-value marker(s) and {n - markers} empty cell(s)")
            findings.append(_finding(
                "warning", "active_interpolated",
                f"{n} missing value(s) of {col} ({what}) were filled by interpolating each substation's "
                "measurements in time — check them.", count=n))
    for col, names in (notes.get("no_values") or {}).items():
        findings.append(_finding(
            "warning", "active_no_values",
            f"{col} has no valid value at all for {', '.join(names[:8])}{' …' if len(names) > 8 else ''}: "
            "set to 0.", substations=names))
    for col, n in (notes.get("markers") or {}).items():
        if col.endswith("_mw"):
            continue                                   # reported with the interpolation above
        else:
            findings.append(_finding(
                "info", "missing_marker_reactive",
                f"{n} value(s) of {col} are a missing-value marker: treated as missing.", count=n))
    for col in notes.get("empty_columns") or []:
        findings.append(_finding("warning", "empty_reactive_column",
                                 f"The {col} column is present but empty: ignored (the load power factor "
                                 "setting applies)."))

    if not reactive["consumption"]:
        findings.append(_finding("info", "no_reactive_consumption",
                                 "No measured reactive demand: loads take their reactive power from the load "
                                 "power factor setting."))
    else:
        p, q = frame["consumption"].astype(float), frame["consumption_q"]
        loaded = p > 1e-6
        coverage = float(q[loaded].notna().mean()) if loaded.any() else 1.0
        findings.append(_finding(
            "info" if coverage >= 0.999 else "warning", "reactive_consumption_coverage",
            (f"Reactive demand is measured for {coverage:.0%} of the loaded substation-ticks"
             + ("." if coverage >= 0.999 else "; the load power factor setting fills the rest.")),
            coverage=round(coverage, 4)))
        both = loaded & q.notna()
        if both.any():
            pf = p[both] / np.hypot(p[both], q[both])
            low = int((pf < LOW_PF).sum())
            if low:
                findings.append(_finding(
                    "warning", "low_power_factor",
                    f"{low} substation-tick(s) imply a load power factor below {LOW_PF} (min {pf.min():.2f}) — "
                    "check the reactive values or their units.", count=low, min_pf=round(float(pf.min()), 3)))
            leading = int((q[both] < 0).sum())
            if leading:
                findings.append(_finding(
                    "info", "leading_load",
                    f"{leading} substation-tick(s) have negative reactive demand (a leading, capacitive load — "
                    "e.g. cables or compensation).", count=leading))
            ratio = float((q[both].abs() / p[both]).median())
            if ratio > 10:
                findings.append(_finding(
                    "error", "reactive_units",
                    f"Reactive demand is typically {ratio:.0f}× the active demand — the values look like kvar, "
                    "not Mvar. Check the units.", ratio=round(ratio, 1)))
            if float(q[both].abs().max()) == 0.0:
                findings.append(_finding(
                    "warning", "reactive_all_zero",
                    "Every measured reactive demand is exactly 0 — filled-in data rather than measurements? "
                    "Remove the column to use the power factor setting instead."))
        idle = (~loaded) & q.notna() & (q.abs() > 1e-6)
        if idle.any():
            findings.append(_finding(
                "info", "reactive_without_load",
                f"{int(idle.sum())} substation-tick(s) have reactive demand but no active demand (cable charging "
                "or compensation); used as measured.", count=int(idle.sum())))

    if reactive["production"]:
        qp = frame["production_q"]
        producing = frame["production"].abs() > 1e-6
        coverage = float(qp[producing].notna().mean()) if producing.any() else 1.0
        findings.append(_finding(
            "info" if coverage >= 0.999 else "warning", "reactive_production_coverage",
            (f"Reactive generation is measured for {coverage:.0%} of the producing substation-ticks"
             + ("." if coverage >= 0.999 else "; each unit's own model fills the rest.")),
            coverage=round(coverage, 4)))
        if voltage_substations:
            measured = set(frame.loc[qp.notna(), "substation_name"].astype(str))
            held = sorted(measured & set(voltage_substations))
            if held:
                findings.append(_finding(
                    "info", "reactive_at_voltage_units",
                    f"Measured reactive generation is not imposed at {', '.join(held[:8])}"
                    f"{' …' if len(held) > 8 else ''}: a unit there controls voltage and decides its own reactive "
                    "power (switch it to fixed Q in the System tab to follow the data instead).",
                    substations=held))
    return {"reactive": reactive, "findings": findings,
            "summary": _summary(reactive, findings)}


def _summary(reactive: dict, findings: list[dict]) -> str:
    worst = "error" if any(f["level"] == "error" for f in findings) else \
        "warning" if any(f["level"] == "warning" for f in findings) else "ok"
    parts = []
    if reactive["consumption"]:
        cov = next((f["coverage"] for f in findings if f["code"] == "reactive_consumption_coverage"), None)
        parts.append("measured reactive demand" + (f" ({cov:.0%} coverage)" if cov is not None and cov < 0.999
                                                   else ""))
    else:
        parts.append("no reactive measurements (power factor setting)")
    if reactive["production"]:
        parts.append("measured reactive generation")
    return f"{'; '.join(parts)} · {worst}"


def distribute(total, weights: pd.Series) -> pd.Series:
    """`total` split over rows by `weights` (equally when they sum to 0)."""
    w = weights.clip(lower=0).fillna(0.0).astype(float)
    s = float(w.sum())
    if not math.isfinite(s) or s <= 0:
        return pd.Series(total / max(len(w), 1), index=w.index)
    return w / s * total
