"""
timeline.py — the operating point: which data, at what time.

The sidebar's operating-point card is built on these functions; they hold no
Streamlit state, so they can be tested directly (importing app.py runs the
whole page).

Two datasets can be loaded, and either may be missing:

- **measurements** — what happened. The simulation clock is a position in
  this series; moving it replays the actuals.
- **forecasts** — what was expected. Query-only: a forecast does not happen,
  so there is nothing for a clock to replay.

They may cover different periods or share timestamps (a day-ahead forecast
uploaded for a period that was also measured). Where they share a timestamp
the operator can look at either, or at both side by side ("compare").

The dataset is chosen first and the time second, within that dataset's own
timestamps — so a time that has no data in the chosen dataset never arises,
and stepping moves at that dataset's own resolution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from bisect import bisect_left

MEASURED, FORECAST, COMPARE = "measured", "forecast", "compare"
LABELS = {MEASURED: "Measured", FORECAST: "Forecast", COMPARE: "Compare"}


def _parse(ts: str) -> datetime:
    return datetime.fromisoformat(ts)


@dataclass(frozen=True)
class Timeline:
    measured: tuple[str, ...] = ()
    forecast: tuple[str, ...] = ()
    measured_source: str = "synthetic"
    forecast_source: str = "synthetic"
    clock: str | None = None
    shared: tuple[str, ...] = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, "shared", tuple(sorted(set(self.measured) & set(self.forecast))))

    @classmethod
    def from_payload(cls, payload: dict) -> Timeline:
        """From `/api/time/timeline`; an unreachable backend gives an empty one."""
        payload = payload or {}
        return cls(
            measured=tuple(sorted(payload.get("timestamps") or [])),
            forecast=tuple(sorted(payload.get("forecast_timestamps") or [])),
            measured_source=payload.get("measurements_source") or "synthetic",
            forecast_source=(payload.get("forecast") or {}).get("source") or "synthetic",
            clock=payload.get("current_timestamp"),
        )

    def modes(self) -> list[str]:
        """The views that have data, in display order."""
        return [m for m in (MEASURED, FORECAST, COMPARE) if self.ticks(m)]

    def ticks(self, mode: str) -> tuple[str, ...]:
        return {MEASURED: self.measured, FORECAST: self.forecast, COMPARE: self.shared}.get(mode, ())


def nearest(ticks: tuple[str, ...] | list[str], ts: str | None) -> str | None:
    """`ts` itself if it is a tick, else the closest tick in time (earlier on a tie)."""
    if not ticks:
        return None
    if ts is None:
        return ticks[0]
    i = bisect_left(ticks, ts)
    if i < len(ticks) and ticks[i] == ts:
        return ts
    candidates = [ticks[j] for j in (i - 1, i) if 0 <= j < len(ticks)]
    try:
        target = _parse(ts)
        return min(candidates, key=lambda t: abs((_parse(t) - target).total_seconds()))
    except ValueError:
        return candidates[0]


def step(ticks: tuple[str, ...] | list[str], ts: str | None, n: int) -> str | None:
    """The tick `n` places from `ts`, clamped to the ends."""
    here = nearest(ticks, ts)
    if here is None:
        return None
    i = ticks.index(here)
    return ticks[max(0, min(len(ticks) - 1, i + n))]


def resolution(ticks: tuple[str, ...] | list[str]) -> str | None:
    """The usual spacing between ticks ("15 min", "1 h", "1 day")."""
    gaps = []
    for a, b in zip(ticks[:50], ticks[1:51]):
        try:
            gaps.append(int((_parse(b) - _parse(a)).total_seconds() // 60))
        except ValueError:
            continue
    gaps = sorted(g for g in gaps if g > 0)
    if not gaps:
        return None
    minutes = gaps[len(gaps) // 2]
    if minutes % 1440 == 0:
        return f"{minutes // 1440} day"
    if minutes % 60 == 0:
        return f"{minutes // 60} h"
    return f"{minutes} min"


def coverage(timeline: Timeline, cursor: str | None) -> dict:
    """Where each dataset has data across the whole span, as fractions 0–1.

    A dataset is drawn from its first to its last tick; the cursor marks the
    selected time. Used for the strip under the slider, so overlap and the
    forecast horizon are visible at a glance.
    """
    every = (timeline.measured[:1] + timeline.measured[-1:]
             + timeline.forecast[:1] + timeline.forecast[-1:])
    if not every:
        return {"rows": [], "cursor": None}
    start, end = _parse(min(every)), _parse(max(every))
    span = (end - start).total_seconds() or 1.0

    def frac(ts: str) -> float:
        return round((_parse(ts) - start).total_seconds() / span, 4)

    rows = []
    for mode, ticks, source in ((MEASURED, timeline.measured, timeline.measured_source),
                                (FORECAST, timeline.forecast, timeline.forecast_source)):
        if ticks:
            rows.append({"mode": mode, "start": frac(ticks[0]), "end": frac(ticks[-1]),
                         "first": ticks[0], "last": ticks[-1],
                         "resolution": resolution(ticks), "source": source})
    return {"rows": rows, "cursor": frac(cursor) if cursor else None}


def agent_note(mode: str, ts: str | None) -> str | None:
    """What the agent is told about the operating point, prefixed to the message.

    Nothing for measurements: the simulation clock is already there, and every
    tool defaults to it.
    """
    if not ts or mode == MEASURED:
        return None
    if mode == FORECAST:
        return (
            f"[The user is viewing the FORECAST timestamp {ts} on the timeline. "
            f"Treat this as the timestamp of interest: use data_source=\"forecasts\" and "
            f"timestamp=\"{ts}\" for time-specific tools unless they ask otherwise.]"
        )
    if mode == COMPARE:
        return (
            f"[The user is comparing FORECAST against MEASURED at {ts}, a timestamp present "
            f"in both datasets. Unless they ask otherwise, run each time-specific tool twice "
            f"at timestamp=\"{ts}\" — once with data_source=\"measurements\" and once with "
            f"data_source=\"forecasts\" — and contrast what was forecast with what happened.]"
        )
    return None
