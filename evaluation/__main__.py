"""
Grade session logs from the command line.

    python -m evaluation                     # the newest log
    python -m evaluation path/to/log.jsonl   # specific logs
    python -m evaluation --all               # every log in the log directory
    python -m evaluation --json              # machine-readable

Exit status is 1 if any turn has a failing finding, so this can gate a run.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

from .report import grade_log, render_report
from .session_log import iter_logs, newest_log


def _resolve(args: argparse.Namespace) -> list[pathlib.Path]:
    if args.logs:
        return [pathlib.Path(p) for p in args.logs]
    if args.all:
        from llm_agent.agent.config import SESSION_LOG_DIR

        return list(iter_logs(args.log_dir or SESSION_LOG_DIR))
    latest = newest_log(args.log_dir)
    return [latest] if latest else []


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evaluation", description=__doc__)
    parser.add_argument("logs", nargs="*", help="session log files (default: the newest)")
    parser.add_argument("--all", action="store_true", help="grade every log in the log directory")
    parser.add_argument("--log-dir", default=None, help="override the session log directory")
    parser.add_argument("--json", action="store_true", help="emit findings as JSON")
    args = parser.parse_args(argv)

    paths = _resolve(args)
    if not paths:
        print("No session logs found.", file=sys.stderr)
        return 2

    failed = False
    payload = []

    for path in paths:
        grades = grade_log(path)
        failed = failed or any(g.failed for g in grades)
        if args.json:
            payload.append({
                "log": str(path),
                "turns": [
                    {
                        "turn": g.turn,
                        "prompt": g.prompt,
                        "tool_calls": g.n_tool_calls,
                        "figures": {
                            "total": g.n_claims,
                            "grounded": g.n_grounded,
                            "imprecise": g.n_imprecise,
                            "misattributed": g.n_misattributed,
                            "ungrounded": g.n_ungrounded,
                        },
                        "findings": [f._asdict() for f in g.findings],
                    }
                    for g in grades
                ],
            })
        else:
            print(render_report(grades, source=str(path)))
            print()

    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
