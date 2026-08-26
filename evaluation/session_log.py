"""Reading session logs, and the shape a grader reports in."""

from __future__ import annotations

import json
import pathlib
from typing import Iterator, NamedTuple

# Failure channels, following the framing the additions are organised around.
COMPUTATION = "computation"       # guarded by construction: the agent never computes
PARAMETERISATION = "parameterisation"  # which arguments the model chose
INTERPRETATION = "interpretation"      # what the model wrote about the results
EXECUTION = "execution"                # the call did not complete
STATE = "state"                        # the turn changed state shared with later turns

FAIL = "fail"
WARN = "warn"


class Finding(NamedTuple):
    """One thing wrong with one turn.

    Deliberately the same shape as `validators.Issue` plus the turn it belongs
    to: a grader reports in the vocabulary the live guards already use, so the
    two are directly comparable.
    """

    turn: int
    channel: str
    severity: str
    code: str
    detail: str

    def render(self) -> str:
        return f"turn {self.turn}  [{self.severity}] {self.channel}/{self.code}: {self.detail}"


def load_records(path: str | pathlib.Path) -> list[dict]:
    """Every turn record in one JSONL session log.

    Malformed lines are skipped rather than fatal: a log truncated by a crash
    is still worth grading, and the crash is often the thing being diagnosed.
    """
    records: list[dict] = []
    for line in pathlib.Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def iter_logs(directory: str | pathlib.Path) -> Iterator[pathlib.Path]:
    """Session logs in a directory, oldest first, excluding the `last` pointer."""
    root = pathlib.Path(directory)
    if not root.is_dir():
        return
    logs = [p for p in root.glob("*.jsonl") if not p.is_symlink()]
    yield from sorted(logs, key=lambda p: p.stat().st_mtime)


def newest_log(directory: str | pathlib.Path | None = None) -> pathlib.Path | None:
    """The most recent session log, or None if there are none."""
    if directory is None:
        from llm_agent.agent.config import SESSION_LOG_DIR

        directory = SESSION_LOG_DIR
    logs = list(iter_logs(directory))
    return logs[-1] if logs else None
