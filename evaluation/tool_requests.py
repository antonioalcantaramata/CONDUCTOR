"""
Tool requests — the tools investigations needed and CONDUCTOR did not have.

When the agent investigates a cause, it records every hypothesis it considered
(`record_investigation`). One no tool can test is recorded as untested, with a
description of the tool that would test it. Those descriptions are kept here,
across sessions, as a list of what to build next, ranked by how often it was
needed. Nothing in it is a finding: an untested hypothesis was never tested.

    python -m evaluation.tool_requests                      # the requests file (llm_agent/.cache)
    python -m evaluation.tool_requests --logs <folder>      # or the records of session logs / eval runs
    python -m evaluation.tool_requests --json

Each request is either a tool to build (an analysis of the grid CONDUCTOR
could compute and does not: a voltage stability margin, dynamics, ramps over
time, the cost of a cause) or data to integrate (an input it does not have:
weather, customer metering, operator logs); the two are listed apart.
Requests are grouped by the needs they ask for, so the many ways the agent
words one need count together: for data, weather, calendar and events,
customer data, operator logs, protection events, equipment history, line
ratings, prices; for analyses, voltage stability, dynamics, time-coupled
studies, costs, short circuit, multiple contingencies… A request counts under
each need it asks for, not under those it names only in saying why CONDUCTOR
cannot do it. One asking for none is grouped by its wording, loosely (case,
punctuation and filler words do not count).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

from llm_agent.agent.investigation_record import read_requests as from_file  # noqa: F401
from llm_agent.agent.investigation_record import summarise_requests as summarise  # noqa: F401
from llm_agent.agent.investigation_record import ANALYSIS, DATA


def from_logs(folder: pathlib.Path) -> list[dict]:
    """The untested hypotheses of every accepted record in a folder of turn
    records (session logs, or evaluation runs)."""
    rows = []
    for path in sorted(pathlib.Path(folder).rglob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if not isinstance(rec, dict):
                continue
            for entry in rec.get("tool_results") or []:
                result = entry.get("result") if isinstance(entry, dict) else None
                if (isinstance(entry, dict) and entry.get("name") == "record_investigation"
                        and isinstance(result, dict) and result.get("status") == "recorded"):
                    for h in result.get("hypotheses") or []:
                        if h.get("verdict") == "untested" and h.get("missing_tool"):
                            rows.append({"symptom": result.get("symptom", ""), "hypothesis": h["hypothesis"],
                                         "missing_tool": h["missing_tool"], "missing_kind": h.get("missing_kind"),
                                         "time": rec.get("timestamp", "")})
    return rows


def render(groups: list[dict]) -> str:
    if not groups:
        return "No tool requests recorded."
    lines = []
    for kind, title in ((ANALYSIS, "Tools to build — analyses of the grid CONDUCTOR could compute and does not"),
                        (DATA, "Data to integrate — inputs CONDUCTOR does not have")):
        rows = [g for g in groups if g["kind"] == kind]
        lines += [f"## {title}", ""]
        if not rows:
            lines += ["None.", ""]
            continue
        lines += ["| Needed | Need | As the agent described it | For hypotheses such as |", "|---:|---|---|---|"]
        for g in rows:
            more = f" (+{len(g['wordings']) - 1} other wordings)" if len(g["wordings"]) > 1 else ""
            lines.append(f"| {g['count']} | {g['need'] or '–'} | {g['tool']}{more} | "
                         f"{'; '.join(g['hypotheses'][:3])} |")
        lines.append("")
    return "\n".join(lines).rstrip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evaluation.tool_requests", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--logs", default="", help="read the records in this folder instead of the requests file")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.logs:
        rows = from_logs(pathlib.Path(args.logs))
    else:
        rows = from_file()
    groups = summarise(rows)
    print(json.dumps(groups, indent=1) if args.json else render(groups))
    return 0


if __name__ == "__main__":
    sys.exit(main())
