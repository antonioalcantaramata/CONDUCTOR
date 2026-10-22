"""
What-if evaluation — a fixed set of network-proposal questions, run
repeatedly and graded from the logs.

A single live session shows what the agent did once. This shows how often it
does it: whether it builds the edit the operator described, names the
proposal, looks beyond one moment, screens N-1 when asked, and passes on the
modelling notes it is given. Run it with and without following the AI Agent
Suggestion to measure what the suggestion changes.

    python -m evaluation.whatif list
    python -m evaluation.whatif run --repeats 10 [--tasks dc_flat,new_site] [--follow-suggestion]
    python -m evaluation.whatif grade evaluation_runs/whatif-20261011-1530

`run` drives the real agent against the running backend and SPENDS MODEL
CREDIT (it asks for confirmation unless --yes). Its records go to their own
folder, never to the session logs. `grade` reads a folder of runs and costs
nothing. The tasks are written for the bundled PGLib IEEE 14-bus profile.

Grading is deterministic, from the records — the same rule as the rest of
`evaluation`: no LLM judge, nothing hand-labelled.
"""

from __future__ import annotations

import argparse
import datetime
import json
import pathlib
import re
import sys
from typing import NamedTuple

from llm_agent.agent.tool_registry import OVER_TIME
from llm_agent.agent.validators import PROPOSAL_NOTE_PREFIX

from .report import grade_record

# Studies that look at more than one moment: the same list the advisor uses.
_OVER_TIME = OVER_TIME


class Task(NamedTuple):
    id: str
    turns: tuple[str, ...]
    checks: tuple[dict, ...]


# Every task also gets the generic checks in `_ALWAYS`.
TASKS: tuple[Task, ...] = (
    Task("dc_flat",
         ("What if we connect a 50 MW data centre at Bus 9? Compare with today.",),
         ({"kind": "edit", "label": "edit: 50 MW load at Bus_9",
           "match": {"op": "add_load", "bus": 9, "p_mw": 50}},
          {"kind": "over_time", "label": "looked beyond one moment"})),
    Task("dc_follow_worst",
         ("Add a 30 MW load at Bus 4 that follows the bus's own demand. When in the week is it "
          "worst compared with today?",),
         ({"kind": "edit", "label": "edit: 30 MW follow_bus load at Bus_4",
           "match": {"op": "add_load", "bus": 4, "p_mw": 30, "profile": "follow_bus"}},
          {"kind": "over_time", "label": "studied the week"})),
    Task("factory_shift",
         ("A 40 MW factory would run from 08:00 to 18:00 at Bus 13. Would it cause violations "
          "during the week?",),
         ({"kind": "edit", "label": "edit: 40 MW shift 08–18 at Bus_13",
           "match": {"op": "add_load", "bus": 13, "p_mw": 40, "profile": "shift",
                     "start_hour": 8, "end_hour": 18}},
          {"kind": "over_time", "label": "studied the week"})),
    Task("reinforce_after_load",
         ("Create a proposal with a 60 MW load at Bus 9 and compare it with today at "
          "2000-01-03 14:45.",
          "Would doubling the line Bus_8 -> Bus_9 help?"),
         ({"kind": "edit", "label": "edit: 60 MW load at Bus_9",
           "match": {"op": "add_load", "bus": 9, "p_mw": 60}},
          {"kind": "edit", "label": "edit: line L12 ×2", "match": {"op": "reinforce_line", "line": 12,
                                                                   "factor": 2}},
          {"kind": "tool", "label": "compared at the same moment",
           "name": "compare_networks", "args": {"timestamp": "2000-01-03 14:45"}})),
    Task("new_site_n1",
         ("Build a new substation 5 km from Bus 9 with the same conductor as the line Bus_8 -> Bus_9 "
          "and put a 15 MW data centre there. Is it N-1 secure?",),
         ({"kind": "edit", "label": "edit: new bus 5 km from Bus_9 like L12",
           "match": {"op": "add_bus_with_connection", "connect_to": 9, "length_km": 5, "line_like": 12}},
          {"kind": "edit", "label": "edit: 15 MW load at the new bus",
           "match": {"op": "add_load", "bus": 14, "p_mw": 15}},
          {"kind": "n1_on_proposal", "label": "N-1 on the proposal"},
          {"kind": "mentions", "label": "reported the line-length note",
           "any": ["1 km", "placeholder", "impedance"]})),
    Task("open_line_n1",
         ("What happens if we permanently open the line Bus_0 -> Bus_4 [L1]? Check N-1 too.",),
         ({"kind": "edit", "label": "edit: open L1",
           "match": {"op": "set_in_service", "element_type": "line", "element": 1, "in_service": False}},
          {"kind": "n1_on_proposal", "label": "N-1 on the proposal"})),
    Task("wind_week",
         ("Add a 20 MW wind farm at Bus 13 and compare the external grid import with today over "
          "the week.",),
         ({"kind": "edit", "label": "edit: 20 MW wind at Bus_13",
           "match": {"op": "add_generator", "bus": 13, "p_max_mw": 20, "kind": "wind"}},
          {"kind": "over_time", "label": "studied the week"})),
    Task("trafo_upgrade",
         ("Would upgrading the transformer Bus_3 -> Bus_8 [T1] to 80 MVA change anything at the "
          "worst hour of the week?",),
         ({"kind": "edit", "label": "edit: T1 to 80 MVA",
           "match": {"op": "upgrade_trafo", "trafo": 1, "new_sn_mva": 80}},
          {"kind": "over_time", "label": "found the worst hour"})),
    Task("classical_generator",
         ("Add a 30 MW generator at Bus 4 and tell me whether the grid stays secure.",
          "A classical one that holds its bus voltage — use 1.0 p.u."),
         ({"kind": "asked", "label": "asked how the unit is modelled before building it"},
          {"kind": "edit", "label": "edit: 30 MW unit on voltage control at Bus_4",
           "match": {"op": "add_generator", "bus": 4, "p_max_mw": 30, "control": "voltage", "vm_pu": 1.0}})),
    Task("new_substation_trafo",
         ("Build a new 20 kV substation at Bus 9 with two 40 MVA transformers (vk 12 %, vkr 0.4 %) "
          "and connect a 30 MW data centre to it. Is it N-1 secure?",),
         ({"kind": "edit", "label": "edit: 2 × 40 MVA transformer to a 20 kV bus at Bus_9",
           "match": {"op": "add_bus_with_transformer", "connect_to": 9, "vn_kv": 20, "sn_mva": 40,
                     "parallel": 2, "vk_percent": 12}},
          {"kind": "edit", "label": "edit: 30 MW load at the new bus",
           "match": {"op": "add_load", "bus": 14, "p_mw": 30}},
          {"kind": "n1_on_proposal", "label": "N-1 on the proposal"},
          {"kind": "mentions", "label": "reported how the units are modelled",
           "any": ["together", "one branch", "label", "per unit"]})),
)

_ALWAYS = ({"kind": "named", "label": "named the proposal"},
           {"kind": "valid_edits", "label": "no refused edit"})


# ---------------------------------------------------------------------------
# Checks — on one run's records (one task, possibly several turns)
# ---------------------------------------------------------------------------

def _successful(records):
    for rec in records:
        for call, entry in zip(rec.get("tool_calls") or [], rec.get("tool_results") or []):
            result = entry.get("result") if isinstance(entry, dict) else None
            if isinstance(result, dict) and not result.get("error"):
                yield call, result


def _same(want, got) -> bool:
    if isinstance(want, bool) or isinstance(got, bool):
        return want == got
    if isinstance(want, (int, float)) and isinstance(got, (int, float)):
        return abs(want - got) <= 1e-6 + 0.01 * abs(want)
    return str(want).strip().casefold() == str(got).strip().casefold()


def _edits(records) -> list[dict]:
    """Every edit a successful edit_proposal stored, in canonical form."""
    return [e for call, result in _successful(records) if call.get("name") == "edit_proposal"
            for e in result.get("edits") or []]


def _on_proposal(result) -> bool:
    net = result.get("network")
    return isinstance(net, dict) and bool(net.get("id"))


def _asks(result: dict) -> bool:
    detail = result.get("detail")
    return isinstance(detail, dict) and bool(detail.get("needs_operator"))


def check(spec: dict, records: list[dict]) -> bool:
    kind = spec["kind"]
    if kind == "edit":
        return any(all(k in e and _same(v, e[k]) for k, v in spec["match"].items()) for e in _edits(records))
    if kind == "tool":
        return any(call.get("name") == spec["name"] and all(
            str((call.get("args") or {}).get(k, "")).startswith(str(v)) for k, v in spec.get("args", {}).items())
            for call, _ in _successful(records))
    if kind == "over_time":
        return any(_on_proposal(r) and (call.get("name") in _OVER_TIME or (
            call.get("name") == "compare_networks" and r.get("study") == "scan"))
            for call, r in _successful(records))
    if kind == "n1_on_proposal":
        return any(_on_proposal(r) and (call.get("name") == "simulate_all_contingencies" or (
            call.get("name") == "compare_networks" and r.get("study") == "n1"))
            for call, r in _successful(records))
    if kind == "named":
        used = any(_on_proposal(r) for _, r in _successful(records))
        return used and not any(PROPOSAL_NOTE_PREFIX in (rec.get("assistant") or "") for rec in records)
    if kind == "mentions":
        text = " ".join(rec.get("assistant") or "" for rec in records).casefold()
        return any(word.casefold() in text for word in spec["any"])
    if kind == "valid_edits":
        # A question for the operator (needs_operator) is not a refused edit.
        return not any(call.get("name") == "edit_proposal" and isinstance(entry.get("result"), dict)
                       and entry["result"].get("error") and not _asks(entry["result"])
                       for rec in records
                       for call, entry in zip(rec.get("tool_calls") or [], rec.get("tool_results") or []))
    if kind == "asked":
        # A turn that ended with a question to the operator and stored no edit.
        return any("?" in (rec.get("assistant") or "") and not any(
            call.get("name") == "edit_proposal" and not (isinstance(entry.get("result"), dict)
                                                         and entry["result"].get("error"))
            for call, entry in zip(rec.get("tool_calls") or [], rec.get("tool_results") or []))
            for rec in records)
    if kind == "options":
        # optimize_flexibility in options mode, with each given argument.
        return any(call.get("name") == "optimize_flexibility" and result.get("mode") == "options"
                   and all(_arg_matches(want, (call.get("args") or {}).get(key))
                           for key, want in spec.get("args", {}).items())
                   for call, result in _successful(records))
    if kind == "result":
        # Some successful call of `tool` returned these fields.
        return any(call.get("name") == spec["tool"] and all(_same(v, result.get(k)) for k, v in spec["fields"].items())
                   for call, result in _successful(records))
    if kind == "not_tool_arg":
        # No call of `tool` passed `arg` naming this value (e.g. a unit ruled
        # out sent to disabled_generators, which takes it out of service).
        return not any(call.get("name") == spec["tool"] and _arg_matches(spec["value"], (call.get("args") or {}).get(spec["arg"]))
                       for rec in records for call in rec.get("tool_calls") or [])
    if kind == "any":
        # Passes when any of the given checks does — two valid ways to the same end.
        return any(check(sub, records) for sub in spec["of"])
    if kind == "no_winner":
        # The orchestrator presents options; it does not choose one.
        return not any(_PICKS.search(rec.get("assistant") or "") for rec in records)
    if kind == "quotes_alternatives":
        # The answer names a unit some near-optimal alternative avoids or adds.
        units = {u for _, r in _successful(records) for o in r.get("options") or []
                 for u in (o.get("avoids") or []) + (o.get("adds") or [])}
        text = " ".join(rec.get("assistant") or "" for rec in records)
        return bool(units) and any(u in text for u in units)
    if kind == "asked_first":
        # That turn put a question to the operator and ran nothing: no study,
        # no change — looking an element up or reading the clock is allowed.
        rec = _turn(records, spec.get("turn", 0))
        return rec is not None and "?" in (rec.get("assistant") or "") and not _acted(rec)
    if kind == "choices":
        # The question offered numbered choices ("1. … 2. …").
        rec = _turn(records, spec.get("turn", 0))
        return rec is not None and len(_CHOICE_RE.findall(rec.get("assistant") or "")) >= 2
    if kind == "ran":
        # That turn ran a study (or a given tool, with the given arguments)
        # without stopping to ask.
        rec = _turn(records, spec.get("turn", 0))
        if rec is None:
            return False
        calls = [(c, r) for c, r in _successful([rec]) if c.get("name") not in LOOKUPS]
        return any((not spec.get("tool") or call.get("name") == spec["tool"])
                   and all(_arg_matches(want, (call.get("args") or {}).get(key))
                           for key, want in spec.get("args", {}).items())
                   for call, _ in calls)
    raise ValueError(f"Unknown check kind {kind!r}")


# Tools that only look something up: calling one before asking is not acting
# (the same list the answer checks use to recognise a clarifying question).
from llm_agent.agent.provenance import LOOKUP_TOOLS as LOOKUPS  # noqa: E402
_CHOICE_RE = re.compile(r"(?m)^\s*(?:[-*]\s*)?(?:\*\*)?\(?[1-9][.)]")


def _turn(records: list[dict], turn: int) -> dict | None:
    """The record of one turn of the task (0 = the first question)."""
    return records[turn] if 0 <= turn < len(records) else None


def _acted(rec: dict) -> bool:
    """Whether a turn called anything beyond a lookup (a study or a change)."""
    return any(call.get("name") not in LOOKUPS for call in rec.get("tool_calls") or [])


# Phrases that choose an option for the operator. Deliberately narrow: "the
# cheapest option costs…" describes; "I recommend the cheapest" chooses.
_PICKS = re.compile(r"\b(I (?:would )?recommend|I(?:'d| would) (?:choose|pick|go with|take)|my recommendation|"
                    r"you should (?:choose|pick|go with|take)|the best option is|the best choice is)\b", re.IGNORECASE)


def _arg_matches(want, got) -> bool:
    """A requested argument against the call's: a value in a list is enough
    for list arguments ("Gen_bus1" in hold_units)."""
    if isinstance(got, list):
        return any(_same(want, g) for g in got) if not isinstance(want, list) else \
            all(any(_same(w, g) for g in got) for w in want)
    return got is not None and _same(want, got)


def evaluate(task: Task, records: list[dict], always: tuple = None) -> dict[str, bool]:
    return {spec["label"]: check(spec, records) for spec in (*task.checks, *(_ALWAYS if always is None else always))}


class Suite(NamedTuple):
    """A set of tasks, the checks every task gets, and the network it is written for."""
    name: str
    tasks: tuple[Task, ...]
    always: tuple
    needs: str               # a substring of the loaded network's name
    scenario: str            # the scenario that loads it


WHATIF = Suite("whatif", TASKS, _ALWAYS, "14", "Demo: IEEE 14 · smaller")


# ---------------------------------------------------------------------------
# Running (spends model credit) and grading (free)
# ---------------------------------------------------------------------------

def _backend(method: str, path: str):
    import httpx
    from llm_agent.agent.config import BASE_URL, HTTP_TIMEOUT
    with httpx.Client(timeout=HTTP_TIMEOUT) as client:
        response = getattr(client, method)(f"{BASE_URL}{path}")
        response.raise_for_status()
        return response.json()


def _clear_proposals() -> None:
    """Each run starts from the loaded network alone."""
    for prop in _backend("get", "/api/proposals").get("proposals") or []:
        _backend("delete", f"/api/proposals/{prop['id']}")


def run_task(task: Task, repeat: int, follow_suggestion: bool = False) -> list[dict]:
    """One run of one task through the real agent. Returns its turn records."""
    from llm_agent.agent import loop, recommender

    _clear_proposals()
    records: list[dict] = []
    original = loop._append_to_log
    loop._append_to_log = records.append          # keep runs out of the session logs
    try:
        history: list = []
        conversation = f"whatif-{task.id}-{repeat}"
        visible = []
        for question in task.turns:
            text, history = loop.run_agent_turn(question, history, conversation_id=conversation,
                                                reflection="off")
            visible += [{"role": "user", "text": question}, {"role": "assistant", "text": text}]
        if follow_suggestion and records:
            proposals = _backend("get", "/api/proposals").get("proposals") or []
            result = recommender.suggest(visible, records[-1], records, {
                "proposals": [{"id": p["id"], "edits": p["summary"]["edits"]} for p in proposals]})
            picks = [s for s in result.suggestions if s.angle == "stress"] or list(result.suggestions)
            if picks:
                records[-1] = {**records[-1], "followed_suggestion": picks[0].as_record()}
                loop.run_agent_turn(picks[0].prompt, history, conversation_id=conversation, reflection="off")
    finally:
        loop._append_to_log = original
    return records


def run(tasks: list[Task], repeats: int, out_dir: pathlib.Path, follow_suggestion: bool = False,
        suite_kind: str = "whatif") -> pathlib.Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    for task in tasks:
        for k in range(1, repeats + 1):
            print(f"[{suite_kind}] {task.id} run {k}/{repeats}", flush=True)
            try:
                records = run_task(task, k, follow_suggestion)
            except Exception as exc:  # noqa: BLE001 — one failed run must not end the batch
                records = [{"turn": 0, "status": "runner_error", "runner_error": str(exc)}]
            with open(out_dir / f"{task.id}-{k:02d}.jsonl", "w", encoding="utf-8") as fh:
                fh.write(json.dumps({"kind": suite_kind, "task": task.id, "repeat": k,
                                     "follow_suggestion": follow_suggestion}) + "\n")
                for rec in records:
                    fh.write(json.dumps(rec, default=str) + "\n")
    return out_dir


def grade(out_dir: pathlib.Path, suite: Suite = WHATIF) -> dict:
    """{task: {"runs": n, "checks": {label: passes}, "findings": {code: count}}}."""
    by_id = {t.id: t for t in suite.tasks}
    summary: dict = {}
    for path in sorted(pathlib.Path(out_dir).glob("*.jsonl")):
        lines = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
        meta, records = lines[0], [r for r in lines[1:] if r.get("status") != "runner_error"]
        task = by_id.get(meta.get("task"))
        if task is None:
            continue
        entry = summary.setdefault(task.id, {"runs": 0, "checks": {}, "findings": {}})
        entry["runs"] += 1
        for label, ok in evaluate(task, records, suite.always).items():
            entry["checks"][label] = entry["checks"].get(label, 0) + int(ok)
        for rec in records:
            for f in grade_record(rec).findings:
                entry["findings"][f.code] = entry["findings"].get(f.code, 0) + 1
    return summary


def render(summary: dict) -> str:
    lines = ["| Task | Runs | Check | Passed |", "|---|---:|---|---:|"]
    for task, entry in summary.items():
        for n, (label, passes) in enumerate(entry["checks"].items()):
            lines.append(f"| {task if n == 0 else ''} | {entry['runs'] if n == 0 else ''} | {label} | "
                         f"{passes}/{entry['runs']} |")
    codes: dict[str, int] = {}
    for entry in summary.values():
        for code, count in entry["findings"].items():
            codes[code] = codes.get(code, 0) + count
    if codes:
        lines += ["", "Findings across all runs: " + ", ".join(f"{c} ×{n}" for c, n in sorted(codes.items()))]
    return "\n".join(lines)


def main(argv: list[str] | None = None, suite: Suite = WHATIF, doc: str | None = None) -> int:
    parser = argparse.ArgumentParser(prog=f"python -m evaluation.{suite.name}", description=doc or __doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="show the tasks")
    p_run = sub.add_parser("run", help="run tasks through the real agent (spends model credit)")
    p_run.add_argument("--repeats", type=int, default=10)
    p_run.add_argument("--tasks", default="", help="comma-separated task ids (default: all)")
    p_run.add_argument("--out", default="", help="output folder (default: evaluation_runs/whatif-<time>)")
    p_run.add_argument("--follow-suggestion", action="store_true",
                       help="after the task, follow the AI Agent Suggestion's stress suggestion")
    p_run.add_argument("--yes", action="store_true", help="do not ask for confirmation")
    p_grade = sub.add_parser("grade", help="grade a folder of runs (free)")
    p_grade.add_argument("folder")
    p_grade.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    if args.command == "list":
        for t in suite.tasks:
            print(f"{t.id}: " + " → ".join(t.turns))
        return 0
    if args.command == "grade":
        summary = grade(pathlib.Path(args.folder), suite)
        print(json.dumps(summary, indent=1) if args.json else render(summary))
        return 0

    wanted = [s for s in args.tasks.split(",") if s.strip()]
    tasks = [t for t in suite.tasks if not wanted or t.id in wanted]
    unknown = set(wanted) - {t.id for t in suite.tasks}
    if unknown:
        print(f"Unknown task(s): {sorted(unknown)}", file=sys.stderr)
        return 2
    turns = sum(len(t.turns) + int(args.follow_suggestion) for t in tasks) * args.repeats
    name = _backend("get", "/api/grid_constants").get("name", "")
    if suite.needs not in str(name):
        print(f"Warning: the tasks are written for another network than {name!r}. "
              f"Load the '{suite.scenario}' scenario first.")
    print(f"{len(tasks)} task(s) × {args.repeats} repeats = {turns} agent turns"
          + (" plus one suggestion call per run" if args.follow_suggestion else "")
          + ". Every turn calls the configured model and spends credit.")
    if not args.yes and input("Proceed? [y/N] ").strip().lower() != "y":
        return 1
    out = pathlib.Path(args.out or f"evaluation_runs/{suite.name}-{datetime.datetime.now():%Y%m%d-%H%M}")
    run(tasks, args.repeats, out, args.follow_suggestion, suite_kind=suite.name)
    print(render(grade(out, suite)))
    print(f"\nRecords: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
