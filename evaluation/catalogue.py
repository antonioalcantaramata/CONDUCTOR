"""
The task catalogue: every evaluation task, written out from the same definitions the runner uses.

    python -m evaluation.catalogue [--out papers/gm2027/task-catalogue.md]

A paper can quote only a few tasks; the catalogue lists them all — the question (or the turns), the network, what is
set up before it, and every check it is graded on — so a reader can see exactly what was asked and how each answer
was judged. It is generated, never edited by hand: `tests/test_catalogue.py` fails when a copy in the repository
differs from what the code would write. It is written with the paper's material (`papers/`, kept out of git until
the paper is public).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_OUT = _ROOT / "papers" / "gm2027" / "task-catalogue.md"

NETWORKS = {"cigre": "CIGRE MV (pandapower; 110/20 kV, two radial feeders, typed DER)",
            "ieee30": "IEEE 30 (PGLib; unit types not stated)"}

INTRO = {
    "semantics-cigre": "Looking up what elements are, on a network that states nearly everything.",
    "semantics-ieee30": "Looking up what elements are, on a network that states almost nothing: often the right "
                        "answer is that it does not say.",
    "reasoning": "Reasoning from what the elements are: misleading names, interventions (a fact changed, the answer "
                 "and the study must follow), composition, switches, what is measured, consistency.",
    "whatif": "Building the proposal the operator describes and studying it.",
    "investigation": "Investigating a \"why\" about the grid — and only then.",
    "options": "Corrective options, not a winner.",
    "clarify": "Asking first when the question leaves the answer open.",
    "narration": "Reporting a study as its own result says.",
}


def _cell(text: str) -> str:
    return " ".join(str(text).split())


def _setup(setup: dict) -> list[str]:
    out = []
    for table, index, name in setup.get("rename") or []:
        out.append(f"element {table} {index} renamed “{name}”")
    for d in setup.get("declare") or []:
        out.append(f"declared by the operator: {d.get('unit')} is {d.get('source')}")
    for p in setup.get("proposals") or []:
        edits = "; ".join(json.dumps(e, sort_keys=True) for e in p.get("edits") or [])
        out.append(f"proposal `{p.get('proposal')}`: {edits}")
    return out


def _check(spec: dict) -> str:
    return _cell(spec.get("label") or spec.get("kind"))


def render() -> str:
    from . import suites
    lines = ["# Task catalogue", "",
             "Every task of the evaluation suites, generated from the code that runs them "
             "(`evaluation/catalogue.py`). Each check is a fact of the network or of the study's own record, computed "
             "rather than judged: no model grades another, and nothing is labelled by hand. The facts the reasoning "
             "and look-up tasks are graded against are themselves checked against the networks by "
             "`tests/test_ablation_ground_truth.py`.", "",
             "| Suite | Tasks | Turns per repeat |", "|---|---:|---:|"]
    by_suite = {name: suites.items([name]) for name in suites.names()}
    for name, its in by_suite.items():
        lines.append(f"| [{name}](#{name}) | {len(its)} | {sum(len(i.task.turns) for i in its)} |")
    total = sum(len(v) for v in by_suite.values())
    lines += [f"| **all** | **{total}** | **{sum(len(i.task.turns) for v in by_suite.values() for i in v)}** |", ""]
    for name, its in by_suite.items():
        lines += [f"## {name}", "", INTRO.get(name, ""), ""]
        always = its[0].always if its else ()
        if always:
            lines += ["Every task here is also graded on: " + "; ".join(_check(c) for c in always) + ".", ""]
        families = list(dict.fromkeys(i.family for i in its))
        for family in families:
            if family:
                lines += [f"### {name} · {family}", ""]
            for item in (i for i in its if i.family == family):
                t = item.task
                head = f"**`{t.id}`** — {NETWORKS.get(item.network, item.network)}"
                if t.deep:
                    head += " — Deep Research on the first question"
                lines += [head, ""]
                for k, q in enumerate(t.turns, 1):
                    lines.append(f"> {k}. {_cell(q)}" if len(t.turns) > 1 else f"> {_cell(q)}")
                lines.append("")
                for s in _setup(item.setup):
                    lines.append(f"- Set-up: {s}")
                lines += [f"- Check: {_check(c)}" for c in t.checks]
                lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evaluation.catalogue", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args(argv)
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(), encoding="utf-8")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
