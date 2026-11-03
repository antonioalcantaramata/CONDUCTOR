"""
Progress of an ablation run, shared by the processes that make it (`evaluation.ablation`).

Each arm runs in its own process, so the progress lives in a small file in the run folder (`progress.json`): the
turns of the plan already done (including those a resume found finished), the turns run in this session and the
time they took (what the estimate of the time left is based on), and the runs that failed. Every process updates it
after each run; `status` reads it from another terminal.

In a terminal the line is redrawn in place:

    [██████░░░░░░░░░░] 42%  120/290 turns · 18 min · ~23 min left · full · reasoning/names_cigre #2 · 1 failed

written to a file (a log), it is printed once per run instead.
"""

from __future__ import annotations

import json
import pathlib
import shutil
import sys
import time

FILE = "progress.json"
WIDTH = 16


def start(out: pathlib.Path, total: int, done: int) -> dict:
    """A new session of a run: `total` turns in the plan, `done` of them already finished."""
    state = {"total": total, "done": done, "ran": 0, "ran_seconds": 0.0, "failed": 0, "started": time.time(),
             "current": ""}
    save(out, state)
    return state


def load(out: pathlib.Path) -> dict | None:
    try:
        return json.loads((pathlib.Path(out) / FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def save(out: pathlib.Path, state: dict) -> None:
    path = pathlib.Path(out) / FILE
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state), encoding="utf-8")
    tmp.replace(path)


def record(out: pathlib.Path, turns: int, seconds: float | None = None, failed: bool = False,
           current: str = "") -> dict:
    """One run finished (`seconds` when it was run now; None when a resume found it already done)."""
    state = load(out) or start(out, turns, 0)
    state["done"] += turns
    if seconds is not None:
        state["ran"] += turns
        state["ran_seconds"] += seconds
    state["failed"] += int(failed)
    state["current"] = current
    save(out, state)
    return state


def _minutes(seconds: float) -> str:
    m = seconds / 60
    return f"{m:.0f} min" if m < 90 else f"{m / 60:.1f} h"


def line(state: dict, current: str | None = None) -> str:
    """The progress as one line: bar, share, turns, time so far, time left, what is running, failures."""
    total, done = max(int(state["total"]), 1), min(int(state["done"]), int(state["total"]))
    share = done / total
    full = round(share * WIDTH)
    bar = "█" * full + "░" * (WIDTH - full)
    parts = [f"[{bar}] {100 * share:3.0f}%", f"{done}/{state['total']} turns",
             _minutes(time.time() - state["started"])]
    if state.get("ran"):
        per_turn = state["ran_seconds"] / state["ran"]
        parts.append(f"~{_minutes(per_turn * (state['total'] - done))} left")
    now = state.get("current") if current is None else current
    if now:
        parts.append(now)
    if state.get("failed"):
        parts.append(f"{state['failed']} failed")
    return " · ".join(parts)


def show(state: dict, current: str | None = None, stream=None, final: bool = False) -> None:
    """Draw the line: in place in a terminal, one line per call otherwise."""
    stream = stream or sys.stdout
    text = line(state, current)
    if stream.isatty():
        width = shutil.get_terminal_size((100, 20)).columns
        stream.write("\r" + text[:width - 1].ljust(width - 1) + ("\n" if final else ""))
    else:
        stream.write(text + "\n")
    stream.flush()
