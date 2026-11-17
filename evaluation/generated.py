"""
generated.py — a test set built from the loaded network's own CIM layer, for any network.

The suites in `semantics` and `reasoning` are written by hand for CIGRE MV and IEEE 30. This one is
generated: questions whose answers the layer already knows, with deterministic checks, for whatever network is
loaded (an upload, a CGMES model). It measures how well a model answers on *this* network, before relying on it:

  unit_source     "What kind of unit is PV 3?"                  — its stated source, in the answer
  unit_unknown    "What energy source drives Gen_bus2?"         — that the network does not say
  units_of        "Which units are solar?"                      — every one of them (up to four named)
  bus_voltage     "At what voltage is Bus 3?"                   — its stated level, in kV
  load_feeder     "Which feeder supplies Load R11?"             — the feeder the layer infers
  outage          "If Line 2-3 trips, which loads lose supply?" — the loads cut off (up to three named)

A fact the layer only reads from a name, or infers no answer for, is not asked about. The tasks are written
to a JSON file with the network they were built from (its name, fingerprint and the commit), so a run can be
repeated and compared — across models, or with and without the layer (`CONDUCTOR_ONTOLOGY`).

    python -m evaluation.generated generate --out tasks.json [--per-kind 3] [--seed 0]
    python -m evaluation.generated run --tasks tasks.json --out <folder> [--repeats 1] [--yes]
    python -m evaluation.generated grade <folder> --tasks tasks.json

`generate` and `grade` call no AI model; `run` drives the real agent against the running backend and SPENDS
MODEL CREDIT (it asks for confirmation unless --yes).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import subprocess
import sys

from .whatif import Task

KINDS = ("unit_source", "unit_unknown", "units_of", "bus_voltage", "load_feeder", "outage")
# How an answer may say a source (any of these).
SOURCE_WORDS = {"solar": ["solar", "PV", "photovoltaic"], "wind": ["wind"], "hydro": ["hydro"],
                "bioenergy": ["bio"], "nuclear": ["nuclear"], "storage": ["storage", "battery", "batteries"],
                "thermal": ["thermal", "CHP", "diesel", "gas", "fuel cell", "combustion", "fossil"],
                "conventional": ["conventional", "thermal", "dispatchable"], "condenser": ["condenser"]}
# That the network does not say (the semantics suite's list).
from .semantics import _NOT_STATED  # noqa: E402


def _short(name: str) -> str:
    return str(name or "").split(" | ")[0]


def generate(layer: dict, outages: dict[str, list[str]] | None = None, per_kind: int = 3,
             seed: int = 0) -> list[Task]:
    """Tasks from a CIM layer (`/api/network/semantics`). `outages`: {line: [loads it cuts off]}, from the
    layer's `around` (`fetch`); without it, no outage task. At most `per_kind` of each kind, drawn with
    `seed`, so a test set is reproducible."""
    rng = random.Random(seed)
    pick = lambda items: rng.sample(items, min(per_kind, len(items)))  # noqa: E731
    out: list[Task] = []
    units = layer.get("units") or []
    stated = [u for u in units if u["source"]["basis"] == "stated" and u["source"]["value"] in SOURCE_WORDS]
    for u in pick(stated):
        out.append(Task(f"unit_source-{u['name']}", (f"What kind of unit is {u['name']}?",),
                        ({"kind": "mentions", "label": f"said {u['name']} is {u['source']['value']}",
                          "any": SOURCE_WORDS[u["source"]["value"]]},)))
    unknown = [u for u in units if u["source"]["basis"] == "unknown"]
    for u in pick(unknown):
        out.append(Task(f"unit_unknown-{u['name']}", (f"What energy source drives {u['name']}?",),
                        ({"kind": "mentions", "label": f"said the network does not say what drives {u['name']}",
                          "any": _NOT_STATED},)))
    by_source: dict[str, list[str]] = {}
    for u in stated:
        by_source.setdefault(u["source"]["value"], []).append(u["name"])
    for source in pick(sorted(by_source)):
        names = sorted(by_source[source])
        out.append(Task(f"units_of-{source}", (f"Which units in this network are {source}?",),
                        ({"kind": "mentions_all", "label": f"named the {source} units", "all": names[:4]},)))
    buses = [b for b in layer.get("buses") or [] if b["kv"]["basis"] == "stated" and b["kv"]["value"]]
    for b in pick(buses):
        kv = float(b["kv"]["value"])
        out.append(Task(f"bus_voltage-{b['name']}", (f"At what voltage is {b['name']}?",),
                        ({"kind": "mentions", "label": f"{b['name']} at {kv:g} kV",
                          "any": [f"{kv:g} kV", f"{kv:g}kV", f"{kv:g} kv"]},)))
    feeder_of = {b["index"]: _short((b.get("feeder") or {}).get("value"))
                 for b in layer.get("buses") or [] if (b.get("feeder") or {}).get("value")}
    fed = [d for d in layer.get("loads") or [] if d.get("bus_index") in feeder_of]
    for d in pick(fed):
        feeder = feeder_of[d["bus_index"]]
        out.append(Task(f"load_feeder-{d['name']}", (f"Which feeder supplies {d['name']}?",),
                        ({"kind": "mentions_all", "label": f"feeder {feeder}", "all": [feeder]},)))
    cut = {line: loads for line, loads in (outages or {}).items() if loads}
    for line in pick(sorted(cut)):
        loads = sorted(cut[line])
        out.append(Task(f"outage-{line}", (f"If {line} trips, which loads lose supply?",),
                        ({"kind": "mentions_all", "label": f"named the loads {line} cuts off", "all": loads[:3]},)))
    return out


def as_json(tasks: list[Task], network: dict) -> dict:
    return {"network": network, "tasks": [{"id": t.id, "turns": list(t.turns), "checks": list(t.checks)}
                                          for t in tasks]}


def from_json(data: dict) -> list[Task]:
    return [Task(t["id"], tuple(t["turns"]), tuple(t["checks"])) for t in data.get("tasks") or []]


def fetch(max_outages: int = 12) -> tuple[dict, dict, dict]:
    """(the layer, {line: loads it cuts off}, what network it is) from the running backend."""
    from .whatif import _backend
    layer = _backend("get", "/api/network/semantics")
    outages = {}
    lines = [b for b in layer.get("branches") or [] if b["element_type"] == "line"][:max_outages]
    for b in lines:
        name = _short(b["name"])
        around = _backend("get", "/api/network/semantics?about=around&element=" + name.replace(" ", "%20"))
        outages[name] = list(((around or {}).get("without_it") or {}).get("loads_cut_off") or [])
    summary = _backend("get", "/api/network/semantics?about=summary")
    topo = _backend("get", "/api/network/topology")
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                                timeout=10).stdout.strip()
    except Exception:  # noqa: BLE001
        commit = ""
    from llm_agent.agent.config import fetch_grid_constants, network_fingerprint
    status = fetch_grid_constants()
    network = {"name": (topo or {}).get("name"),
               "fingerprint": network_fingerprint(status.values) if status.live else None,
               "buses": (summary or {}).get("buses"), "commit": commit}
    return layer, outages, network


def grade(folder: pathlib.Path, tasks: list[Task]) -> dict:
    """{task: {"runs": n, "passed": n}} and the share of checks passed, from a run's records."""
    from .whatif import evaluate
    by_id = {t.id: t for t in tasks}
    out, passed, total = {}, 0, 0
    for path in sorted(folder.glob("*.jsonl")):
        lines = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
        head, records = lines[0], lines[1:]
        task = by_id.get(head.get("task"))
        if task is None:
            continue
        checks = evaluate(task, records, ())
        row = out.setdefault(task.id, {"runs": 0, "passed": 0})
        row["runs"] += 1
        row["passed"] += int(all(checks.values()))
        passed += sum(bool(v) for v in checks.values())
        total += len(checks)
    return {"tasks": out, "checks_passed": passed, "checks": total,
            "share": round(passed / total, 3) if total else None}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evaluation.generated", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p_gen = sub.add_parser("generate")
    p_gen.add_argument("--out", required=True)
    p_gen.add_argument("--per-kind", type=int, default=3)
    p_gen.add_argument("--seed", type=int, default=0)
    p_run = sub.add_parser("run")
    p_run.add_argument("--tasks", required=True)
    p_run.add_argument("--out", required=True)
    p_run.add_argument("--repeats", type=int, default=1)
    p_run.add_argument("--yes", action="store_true")
    p_grade = sub.add_parser("grade")
    p_grade.add_argument("folder")
    p_grade.add_argument("--tasks", required=True)
    args = parser.parse_args(argv)

    if args.command == "generate":
        layer, outages, network = fetch()
        tasks = generate(layer, outages, args.per_kind, args.seed)
        data = as_json(tasks, {**network, "seed": args.seed, "per_kind": args.per_kind})
        pathlib.Path(args.out).write_text(json.dumps(data, indent=1), encoding="utf-8")
        print(f"{len(tasks)} tasks for {network.get('name')}: {args.out}")
        return 0
    tasks = from_json(json.loads(pathlib.Path(args.tasks).read_text(encoding="utf-8")))
    if args.command == "grade":
        print(json.dumps(grade(pathlib.Path(args.folder), tasks), indent=1))
        return 0
    print(f"{len(tasks)} task(s) × {args.repeats} repeats: {len(tasks) * args.repeats} agent turns. "
          "Every turn spends model credit.")
    if not args.yes and input("Proceed? [y/N] ").strip().lower() != "y":
        return 1
    from .whatif import run
    run(tasks, args.repeats, pathlib.Path(args.out), suite_kind="generated")
    print(json.dumps(grade(pathlib.Path(args.out), tasks), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
