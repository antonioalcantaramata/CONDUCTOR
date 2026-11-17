"""
Semantics evaluation — what the semantic layer is worth to the agent.

The same questions about what the network's elements are, asked with the layer
and without it (an ablation: `CONDUCTOR_ONTOLOGY=off` hides `describe_network`
and `declare_network_facts`, the facts `locate_network_element` adds, and the
prompt's rule on what an element is). Answers are graded the same way in both
arms: each task's checks are facts of the network (computed from it, not
written by judgement), and every answer goes through the entity checks
(`llm_agent/agent/entities.py`) — an element called what the network says it
is not, or one it does not have.

Two networks, because the layer's worth differs: CIGRE MV states nearly
everything (types, levels, open ties); PGLib IEEE 30 states almost nothing, so
the right answer is often that the network does not say.

    python -m evaluation.semantics list
    python -m evaluation.semantics run --network cigre --arm on --repeats 3
    python -m evaluation.semantics compare <folder on> <folder off> [<folder on> <folder off> …]

`run` loads the network's scenario, drives the real agent against the running
backend and SPENDS MODEL CREDIT (it asks for confirmation unless --yes).
Grading is deterministic, from the records.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import pathlib
import sys

from .report import grade_record
from .whatif import Suite, Task, _post_backend, evaluate, run

_NOT_STATED = ["not stated", "does not state", "doesn't state", "does not say", "doesn't say", "not specified",
               "unknown", "no information", "not known", "cannot tell", "can't tell", "not identified",
               "not labelled", "not labeled", "no indication", "not recorded", "does not record",
               "doesn't record", "not given", "does not include", "doesn't include", "no data on", "not available",
               # Said as "I have nothing that says" (observed in the arm without the layer — correct answers):
               "don't have data", "do not have data", "don't have a tool or customer-level data",
               "data that identifies", "identified as wind or solar", "identifies what"]
_LOSE = ["lose supply", "loses supply", "lost supply", "lose their supply", "loses its supply", "disconnected",
         "islanded", "cut off", "without supply", "de-energised", "de-energized", "no supply", "unsupplied",
         "not supplied", "loses its connection", "isolated"]

CIGRE_TASKS: tuple[Task, ...] = (
    Task("cigre_renewables", ("Which units in this network are renewable?",),
         ({"kind": "mentions_all", "label": "named the wind turbine and PV units",
           "all": ["WKA 7", "PV 3", "PV 11"]},)),
    Task("cigre_levels", ("At what voltage are the PV units connected, and where does the network connect to "
                          "the external grid?",),
         ({"kind": "mentions_all", "label": "20 kV, and Bus 0 at 110 kV", "all": ["20 kV", "Bus 0", "110 kV"]},)),
    Task("cigre_storage", ("Is there any storage in this network, and where?",),
         ({"kind": "mentions_all", "label": "both batteries and their buses",
           "all": ["Battery 1", "Bus 5", "Battery 2", "Bus 10"]},)),
    Task("cigre_outage", ("If Line 2-3 trips, which loads lose supply, and which normally open tie could restore "
                          "them?",),
         # The loads by name or by bus: "Buses 3–11" answers the question too (observed).
         ({"kind": "mentions", "label": "named what loses supply, as far as Bus 11",
           "any": ["Load R11", "Bus 11", "Buses 3–11", "Buses 3-11", "Bus 3–11", "Bus 3-11", "Bus 3 through Bus 11",
                   "Bus 3 to Bus 11", "Buses 3 to 11", "Buses 3 through 11"]},
          # Named, not named and rejected (observed without the layer: "Line 14-8 … is in service, so I
          # can't present it as a normally open tie" — its open switch was not seen).
          {"kind": "mentions_all", "label": "named the tie Line 14-8", "all": [["Line 14-8", "L14"]],
           "unless_any": ["can't present it", "cannot present it", "couldn't identify or verify a normally open tie",
                          "could not identify or verify a normally open tie"]})),
    Task("cigre_feeder", ("Which feeder supplies Load R11, and through which lines does its power come from the "
                          "external grid?",),
         ({"kind": "mentions_all", "label": "feeder Line 1-2 and the route through Line 3-8 to Line 10-11",
           "all": [["Line 1-2", "L0"], ["Line 3-8", "L9"], ["Line 10-11", "L8"]]},)),
)

IEEE30_TASKS: tuple[Task, ...] = (
    Task("ieee30_renewables", ("Which units in this network are wind or solar?",),
         ({"kind": "mentions", "label": "said the network does not say", "any": _NOT_STATED},)),
    Task("ieee30_customers", ("What kind of customers does the load at Bus_7 serve?",),
         ({"kind": "mentions", "label": "said the network does not say", "any": _NOT_STATED},)),
    Task("ieee30_level", ("What is the nominal voltage of this network?",),
         ({"kind": "mentions_all", "label": "135 kV", "all": ["135"]},)),
    Task("ieee30_radial", ("If the line Bus_24 -> Bus_25 [L33] trips, does any load lose supply?",),
         ({"kind": "mentions", "label": "named what is cut off", "any": ["Load 18", "Bus_25"]},
          {"kind": "mentions", "label": "said it loses supply", "any": _LOSE})),
    Task("ieee30_external", ("Which lines carry the power coming in from the external grid?",),
         ({"kind": "mentions_all", "label": "L0 and L1", "all": [["L0", "Bus_0 -> Bus_1"], ["L1", "Bus_0 -> Bus_2"]]},)),
)

SUITES = {
    "cigre": Suite("semantics", CIGRE_TASKS, (), "CIGRE", "Demo: CIGRE MV · distribution with DER"),
    "ieee30": Suite("semantics", IEEE30_TASKS, (), "IEEE 30", "Demo: IEEE 30 · active only"),
}
_SCENARIO = {"cigre": "cigre_mv", "ieee30": "ieee30"}


def _turn_stats(records: list[dict]) -> dict:
    calls = sum(len(r.get("tool_calls") or []) for r in records)
    seconds = sum(float((r.get("timing") or {}).get("total_s") or 0.0) for r in records)
    tools = [c.get("name") for r in records for c in r.get("tool_calls") or []]
    return {"calls": calls, "seconds": seconds, "used_layer": any(t == "describe_network" for t in tools)}


def grade_folder(folder: pathlib.Path) -> dict:
    """{task: {"runs", "checks": {label: passes}, "entity": {code: n}, "calls", "seconds", "used_layer"}}."""
    tasks = {t.id: t for s in SUITES.values() for t in s.tasks}
    out: dict = {}
    for path in sorted(folder.glob("*.jsonl")):
        lines = [json.loads(x) for x in open(path, encoding="utf-8") if x.strip()]
        meta, records = lines[0], [r for r in lines[1:] if r.get("status") != "runner_error"]
        task = tasks.get(meta.get("task"))
        if task is None or not records:
            continue
        e = out.setdefault(task.id, {"runs": 0, "checks": {}, "entity": {}, "calls": 0, "seconds": 0.0,
                                     "used_layer": 0})
        e["runs"] += 1
        for label, ok in evaluate(task, records, ()).items():
            e["checks"][label] = e["checks"].get(label, 0) + int(ok)
        for rec in records:
            for f in grade_record(rec).findings:
                if f.code.startswith("entity_"):
                    e["entity"][f.code] = e["entity"].get(f.code, 0) + 1
        st = _turn_stats(records)
        e["calls"] += st["calls"]
        e["seconds"] += st["seconds"]
        e["used_layer"] += int(st["used_layer"])
    return out


def compare(pairs: list[tuple[pathlib.Path, pathlib.Path]], labels: tuple[str, str] = ("With the layer", "Without")
            ) -> str:
    """A table: per task, each arm's checks passed, entity errors, tool calls and time."""
    rows = [f"| Task | Check | {labels[0]} | {labels[1]} |", "|---|---|---:|---:|"]
    totals = {"on": {"passed": 0, "checks": 0, "contradicted": 0, "unsupported": 0, "calls": 0, "seconds": 0.0,
                     "runs": 0},
              "off": {"passed": 0, "checks": 0, "contradicted": 0, "unsupported": 0, "calls": 0, "seconds": 0.0,
                      "runs": 0}}
    for on_dir, off_dir in pairs:
        on, off = grade_folder(on_dir), grade_folder(off_dir)
        for task in sorted(set(on) | set(off)):
            a, b = on.get(task, {}), off.get(task, {})
            for label in (a.get("checks") or b.get("checks") or {}):
                rows.append(f"| {task} | {label} | {a.get('checks', {}).get(label, 0)}/{a.get('runs', 0)} | "
                            f"{b.get('checks', {}).get(label, 0)}/{b.get('runs', 0)} |")
            for arm, e in (("on", a), ("off", b)):
                t = totals[arm]
                t["passed"] += sum((e.get("checks") or {}).values())
                t["checks"] += len(e.get("checks") or {}) * e.get("runs", 0)
                t["contradicted"] += (e.get("entity") or {}).get("entity_contradicted", 0)
                t["unsupported"] += (e.get("entity") or {}).get("entity_unsupported", 0)
                t["calls"] += e.get("calls", 0)
                t["seconds"] += e.get("seconds", 0.0)
                t["runs"] += e.get("runs", 0)
    rows += ["", f"| | {labels[0]} | {labels[1]} |", "|---|---:|---:|"]
    on, off = totals["on"], totals["off"]
    rows.append(f"| Checks passed | {on['passed']}/{on['checks']} | {off['passed']}/{off['checks']} |")
    rows.append(f"| Answers contradicting the network (entity) | {on['contradicted']} | {off['contradicted']} |")
    rows.append(f"| Claims the network does not support (entity) | {on['unsupported']} | {off['unsupported']} |")
    for key, label in (("calls", "Tool calls per run"), ("seconds", "Seconds per run")):
        rows.append(f"| {label} | {on[key] / max(on['runs'], 1):.1f} | {off[key] / max(off['runs'], 1):.1f} |")
    return "\n".join(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evaluation.semantics", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list")
    p_run = sub.add_parser("run")
    p_run.add_argument("--network", choices=sorted(SUITES), required=True)
    p_run.add_argument("--arm", choices=("on", "off", "none"), required=True)
    p_run.add_argument("--repeats", type=int, default=3)
    p_run.add_argument("--tasks", default="")
    p_run.add_argument("--out", default="")
    p_run.add_argument("--yes", action="store_true")
    p_cmp = sub.add_parser("compare")
    p_cmp.add_argument("folders", nargs="+", help="pairs: <folder with the layer> <folder without>")
    args = parser.parse_args(argv)

    if args.command == "list":
        for key, suite in SUITES.items():
            for t in suite.tasks:
                print(f"[{key}] {t.id}: {t.turns[0]}")
        return 0
    if args.command == "compare":
        if len(args.folders) % 2:
            print("Give the folders in pairs: with the layer, then without.", file=sys.stderr)
            return 2
        f = [pathlib.Path(x) for x in args.folders]
        print(compare(list(zip(f[::2], f[1::2]))))
        return 0

    suite = SUITES[args.network]
    wanted = [s for s in args.tasks.split(",") if s.strip()]
    tasks = [t for t in suite.tasks if not wanted or t.id in wanted]
    print(f"{len(tasks)} task(s) × {args.repeats} repeats on {args.network}, layer {args.arm}: "
          f"{len(tasks) * args.repeats} agent turns. Every turn spends model credit.")
    if not args.yes and input("Proceed? [y/N] ").strip().lower() != "y":
        return 1
    out = pathlib.Path(args.out or f"evaluation_runs/semantics-{args.network}-{args.arm}-"
                                    f"{datetime.datetime.now():%Y%m%d-%H%M}")
    run_arm(args.network, args.arm, args.repeats, out, tasks)
    print(f"Records: {out}")
    return 0


ARMS = {"on": "the layer for the agent and the studies",
        "off": "hidden from the agent; the studies still use it",
        "none": "hidden from the agent and switched off in the studies (as before the layer)"}


def check_backend_mode(arm: str) -> None:
    """The backend must be in the arm's mode: `none` needs one started with
    CONDUCTOR_ONTOLOGY=none (the studies read it), the others one without."""
    from .whatif import _backend
    uses = _backend("get", "/api/network/semantics?about=summary").get("studies_use_semantic_layer", True)
    if uses != (arm != "none"):
        raise RuntimeError(f"Arm '{arm}' needs a backend whose studies {'do not ' if arm == 'none' else ''}use the "
                           f"semantic layer; this one {'does' if uses else 'does not'}. Start it with "
                           f"CONDUCTOR_ONTOLOGY={'none' if arm == 'none' else 'on'}.")


def run_arm(network: str, arm: str, repeats: int, out: pathlib.Path, tasks: list | None = None) -> pathlib.Path:
    """One network, one arm: load its scenario, switch the layer, run the tasks."""
    check_backend_mode(arm)
    _post_backend("/api/scenario", {"name": _SCENARIO[network]})
    os.environ["CONDUCTOR_ONTOLOGY"] = arm
    return run(list(tasks or SUITES[network].tasks), repeats, out, suite_kind=f"semantics-{arm}")


if __name__ == "__main__":
    sys.exit(main())
