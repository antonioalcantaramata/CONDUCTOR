"""
Reasoning evaluation — does the semantic layer change how the agent reasons,
not only what it can look up?

Four families, each run with the layer and without it (`CONDUCTOR_ONTOLOGY`,
as in `evaluation.semantics`), on CIGRE MV and IEEE 30:

- **misleading names** — elements renamed so the name contradicts what the
  network states ("CHP diesel 1" called "Solar Farm North"): reasoning from
  the facts follows the type; reasoning from surface cues follows the name;
- **intervention** — the same question before and after one fact changes (a
  unit declared wind): an answer that reasons from the fact changes with it,
  and the study it runs applies the change (`renewables_scaled`);
- **composition** — several hops: what an outage cuts off, which of that is
  solar, what restores it; which feeder a battery and the lowest bus share;
- **consistency** — one question in three wordings; a fact stated in one turn
  used in the next.

Every check is a fact of the network or of the study's own record, never a
judgement. Tasks prepare the network (renames, declarations) through the
backend and undo it afterwards.

    python -m evaluation.reasoning list
    python -m evaluation.reasoning run --arm on --repeats 3 --out <folder>
    python -m evaluation.reasoning analyse <folder with the layer> <folder without>

`run` SPENDS MODEL CREDIT (asks for confirmation unless --yes).
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import sys
from typing import NamedTuple

from .report import grade_record
from .semantics import _NOT_STATED, _turn_stats
from .whatif import Task, _post_backend, check as _base_check, run_task

_NEG = r"\b(?:not|no|isn['’]t|aren['’]t|doesn['’]t|don['’]t|despite|although|though|but|excluding|except|" \
       r"rather than|instead|non-)\b"


class Item(NamedTuple):
    task: Task
    family: str
    network: str                 # cigre | ieee30
    prepare: dict = {}           # {"rename": [(table, index, name)], "declare": [{unit, source}]}
    pair: str = ""               # intervention: the id of the task it is the counterfactual of
    group: str = ""              # consistency: the paraphrases that must agree


_Q_DROP = "If wind and solar output dropped by 50 % at 2000-01-03 12:00, would the grid stay secure, and which " \
          "units would that affect?"
_CIGRE_RENEWABLE = ["PV 3", "PV 4", "PV 5", "PV 6", "PV 8", "PV 9", "PV 10", "PV 11", "WKA 7"]

ITEMS: tuple[Item, ...] = (
    # --- Misleading names ----------------------------------------------------
    Item(Task("names_cigre", ("Which units in this network are solar, and how many are there?",),
              ({"kind": "mentions_all", "label": "named the PV units", "all": ["PV 3", "PV 11"]},
               {"kind": "mentions", "label": "counted eight", "any": ["8 ", "eight"]},
               {"kind": "name_handled", "label": "did not take 'Solar Farm North' (a CHP unit) for solar",
                "name": "Solar Farm North",
                "ok_any": ["chp", "diesel", "thermal", "despite", "its name", "name suggests", "misleading",
                           "not solar", "isn't solar", "is not solar", "actually", "type"]},
               {"kind": "name_handled", "label": "did not take 'PV Park East' (a fuel cell) for solar",
                "name": "PV Park East",
                "ok_any": ["fuel cell", "thermal", "despite", "its name", "name suggests", "misleading",
                           "not solar", "isn't solar", "is not solar", "actually", "type"]})),
         "names", "cigre", {"rename": [("sgen", 10, "Solar Farm North"), ("sgen", 11, "PV Park East")]}),
    Item(Task("names_ieee30", ("Which units in this network are solar?",),
              ({"kind": "name_hedged", "label": "said 'Solar Plant A' is solar only by its name",
                "name": "Solar Plant A",
                "hedge_any": ["its name", "by name", "the name", "name suggests", "named", "not stated",
                              "does not state", "doesn't state", "not confirmed", "cannot confirm", "can't confirm",
                              "not specified", "suggests", "only from", "inferred from"]},)),
         "names", "ieee30", {"rename": [("gen", 0, "Solar Plant A")]}),
    # A PV unit named like a wind park: the stated type (solar) wins over the name.
    Item(Task("names_cigre_wind", ("Which units in this network are wind turbines?",),
              ({"kind": "mentions_all", "label": "named WKA 7", "all": ["WKA 7"]},
               {"kind": "name_handled", "label": "did not take 'Wind Park West' (a PV unit) for wind",
                "name": "Wind Park West",
                "ok_any": ["solar", "pv", "photovoltaic", "despite", "its name", "name suggests", "misleading",
                           "not wind", "isn't wind", "is not wind", "actually", "type"]})),
         "names", "cigre", {"rename": [("sgen", 1, "Wind Park West")]}),
    # A fuel cell named like storage: the network's storage are its two batteries.
    Item(Task("names_cigre_storage", ("Where is the storage in this network?",),
              ({"kind": "mentions_all", "label": "named Battery 1 and Battery 2", "all": ["Battery 1", "Battery 2"]},
               {"kind": "name_handled", "label": "did not take 'Battery Storage 3' (a fuel cell) for storage",
                "name": "Battery Storage 3",
                "ok_any": ["fuel cell", "thermal", "not storage", "isn't storage", "is not storage", "not a battery",
                           "despite", "its name", "name suggests", "misleading", "actually", "type", "generat"]})),
         "names", "cigre", {"rename": [("sgen", 9, "Battery Storage 3")]}),
    # IEEE 30 states no unit's source: a unit named "Offshore Wind 1" is wind by its name only.
    Item(Task("names_ieee30_wind", ("Which units in this network are wind?",),
              ({"kind": "name_hedged", "label": "said 'Offshore Wind 1' is wind only by its name",
                "name": "Offshore Wind 1",
                "hedge_any": ["its name", "by name", "the name", "name suggests", "named", "not stated",
                              "does not state", "doesn't state", "not confirmed", "cannot confirm", "can't confirm",
                              "not specified", "suggests", "only from", "inferred from"]},)),
         "names", "ieee30", {"rename": [("gen", 3, "Offshore Wind 1")]}),

    # --- Intervention ---------------------------------------------------------
    Item(Task("drop_ieee30_none", (_Q_DROP,),
              ({"kind": "mentions", "label": "said no unit is stated wind or solar",
                "any": [*_NOT_STATED, "no unit", "no units", "no wind", "no solar", "none of", "changes nothing",
                        "no effect", "no renewable"]},
               {"kind": "none_affected", "label": "named no unit as affected", "names": [
                   "Gen_bus1", "Gen_bus4", "Gen_bus7", "Gen_bus10", "Gen_bus12"]})),
         "intervention", "ieee30"),
    Item(Task("drop_ieee30_declared", (_Q_DROP,),
              ({"kind": "mentions_all", "label": "named Gen_bus12 as affected", "all": ["Gen_bus12"]},
               {"kind": "scaled_units", "label": "the study scaled Gen_bus12 only", "units": ["Gen_bus12"]})),
         "intervention", "ieee30", {"declare": [{"unit": "Gen_bus12", "source": "wind"}]}, pair="drop_ieee30_none"),
    Item(Task("drop_cigre_stated", (_Q_DROP,),
              ({"kind": "mentions_all", "label": "named the wind and PV units", "all": ["WKA 7", "PV 3"]},
               {"kind": "name_handled", "label": "did not count CHP diesel 1",
                "name": "CHP diesel 1", "ok_any": ["not", "unaffected", "no change", "dispatchable", "thermal",
                                                   "conventional", "except", "excluding"]},
               {"kind": "scaled_units", "label": "the study scaled the nine wind and PV units",
                "units": _CIGRE_RENEWABLE})),
         "intervention", "cigre"),
    Item(Task("drop_cigre_declared", (_Q_DROP,),
              ({"kind": "mentions_all", "label": "named CHP diesel 1 as affected (declared solar)",
                "all": ["CHP diesel 1"]},
               {"kind": "scaled_units", "label": "the study scaled CHP diesel 1 too",
                "units": [*_CIGRE_RENEWABLE, "CHP diesel 1"]})),
         "intervention", "cigre", {"declare": [{"unit": "CHP diesel 1", "source": "solar"}]},
         pair="drop_cigre_stated"),

    # --- Composition ----------------------------------------------------------
    Item(Task("compose_t0", ("If the transformer Bus 0 -> Bus 1 [T0] trips, which generating units lose their "
                             "connection, how many of them are solar, and could supply be restored?",),
              ({"kind": "mentions_all", "label": "named the wind turbine among them", "all": ["WKA 7"]},
               # "**Solar (8):** PV 3 …" counts too (observed, both arms).
               {"kind": "mentions", "label": "eight of them solar",
                "any": ["8 pv", "eight pv", "8 solar", "eight solar", "all 8", "all eight", "8 of", "eight of",
                        "solar (8)", "8 are solar", "eight are solar", "pv (8)"]},
               {"kind": "mentions_all", "label": "the tie Line 14-8 restores it", "all": [["Line 14-8", "L14"]],
                "unless_any": ["can't present it", "cannot present it", "couldn't identify", "could not identify",
                               "no normally open tie", "no tie"]})),
         "composition", "cigre"),
    Item(Task("compose_battery", ("At 18:00 on 3 January, is Battery 1 on the same feeder as the bus with the "
                                  "lowest voltage?",),
              ({"kind": "mentions_all", "label": "found Bus 11 lowest", "all": ["Bus 11"]},
               {"kind": "mentions", "label": "said yes, the same feeder",
                "any": ["yes", "same feeder", "both on", "both supplied", "both fed", "share the feeder",
                        "share a feeder", "on the feeder"]},
               {"kind": "mentions", "label": "named the feeder Line 1-2", "any": ["Line 1-2", "L0"]})),
         "composition", "cigre"),
    Item(Task("compose_l33", ("If the line Bus_24 -> Bus_25 [L33] trips, how much load loses supply, and is "
                              "there a tie that could restore it?",),
              ({"kind": "mentions", "label": "1.84 MW", "any": ["1.84", "1.8 mw"]},
               {"kind": "mentions", "label": "no tie restores it",
                # "no existing tie", "no modeled tie", "its only connection" (observed, both arms).
                "any": ["no tie", "no normally open", "no alternative", "no other path", "no open tie",
                        "no switch", "cannot be restored", "can't be restored", "no open", "no restoring",
                        "not be restored", "no backup", "no other route", "no alternate", "no existing",
                        "no modeled tie", "no modelled tie", "only connection", "only one network connection",
                        "only one connection", "only that one connection"]})),
         "composition", "ieee30"),

    Item(Task("compose_tie", ("If Line 2-3 trips, which switch could restore supply to the far end of the feeder, "
                              "and is it normally open?",),
              ({"kind": "mentions_all", "label": "named the tie S1 (Line 14-8)", "all": [["S1", "Line 14-8", "L14"]],
                "unless_any": ["can't present it", "cannot present it", "couldn't identify", "could not identify",
                               "no normally open tie", "no tie"]},
               {"kind": "mentions", "label": "said it is normally open",
                "any": ["normally open", "normally-open", "open in normal", "normally kept open"]})),
         "composition", "cigre"),

    # --- Switches -------------------------------------------------------------
    Item(Task("switches_open", ("Which switches in this network are normally open, and which of them ties two "
                                "feeders?",),
              ({"kind": "mentions_all", "label": "named the three normally open switches", "all": ["S1", "S2", "S3"]},
               {"kind": "name_hedged", "label": "said S1 is the tie between feeders", "name": "S1",
                "hedge_any": ["tie", "between feeders", "two feeders", "joins", "connects the feeders",
                              "between the feeders", "feeder line 12-13"]})),
         "switches", "cigre"),
    Item(Task("switches_loops", ("Which normally open switches close a loop within one feeder, rather than tying two "
                                 "feeders together?",),
              ({"kind": "mentions_all", "label": "named S2 and S3", "all": ["S2", "S3"]},
               {"kind": "name_handled", "label": "did not count S1 (it ties two feeders)", "name": "S1",
                "ok_any": ["tie", "between feeders", "two feeders", "other feeder", "line 12-13", "not within",
                           "different feeder", "rather than", "except", "excluding", "joins", "connects"]})),
         "switches", "cigre"),
    Item(Task("switches_breakers", ("Which switches in this network are circuit breakers, and what do they switch?",),
              ({"kind": "mentions_all", "label": "named Switch 6 and Switch 7", "all": ["Switch 6", "Switch 7"]},
               {"kind": "mentions", "label": "said they switch the transformers",
                "any": ["transformer", "Trafo 0-1", "Trafo 0-12", "T0", "T1"]})),
         "switches", "cigre"),
    # Line 14-8 is a line in service whose switch S1 is normally open: it carries no power in normal operation
    # (observed without the layer: "Line 14-8 is in service, so it is not a normally open tie"). Asked as "does it
    # carry power?" the pilot showed the question answerable from the flow alone (0.00 MW, charging current from
    # the energised end), so it asks for the switch and its normal state.
    Item(Task("switches_line148", ("Which switch decides whether Line 14-8 carries power, and what is its normal "
                                   "state?",),
              ({"kind": "mentions_all", "label": "named its switch S1", "all": ["S1"]},
               {"kind": "mentions", "label": "said it is normally open",
                "any": ["normally open", "normally-open", "is open", "kept open", "open switch", "no power",
                        "does not carry", "doesn't carry", "not carry", "carries no"]})),
         "switches", "cigre"),

    # --- Data: what is measured and what is not -------------------------------
    Item(Task("data_cigre", ("Is the reactive demand of Load R5 measured? And is PV 9's output measured?",),
              ({"kind": "name_hedged", "label": "Load R5's Q: derived from the power factor, not measured",
                "name": "Load R5", "hedge_any": ["not measured", "derived", "power factor", "estimated", "assum",
                                                 "calculated", "not in the data", "isn't measured"]},
               {"kind": "name_hedged", "label": "PV 9's output: a share of its bus's measurement",
                "name": "PV 9", "hedge_any": ["share", "split", "allocated", "estimate", "shared", "not its own",
                                              "bus total", "apportioned", "not measured individually",
                                              "not separately measured", "together with"]})),
         "data", "cigre"),
    Item(Task("data_ieee30", ("Is the reactive power of Load 5 measured?",),
              ({"kind": "mentions", "label": "said it is derived from the power factor",
                "any": ["not measured", "derived", "power factor", "estimated", "assum", "isn't measured",
                        "not in the data", "calculated"]},)),
         "data", "ieee30"),
    # WKA 7 is alone at Bus 7, so its output is that bus's measured production; PV 9 shares Bus 9 with two units.
    Item(Task("data_cigre_wind", ("Which output is measured directly, and which is estimated: WKA 7's or PV 9's?",),
              ({"kind": "name_hedged", "label": "WKA 7's output: measured",
                "name": "WKA 7", "hedge_any": ["is measured", "directly measured", "measured directly",
                                               "fully measured", "is metered", "measured on its own", "only unit",
                                               "alone at", "its own measurement", "is the measured"]},
               {"kind": "name_hedged", "label": "PV 9's output: a share of its bus's measurement",
                "name": "PV 9", "hedge_any": ["share", "split", "allocated", "estimate", "shared", "not its own",
                                              "bus total", "apportioned", "not measured individually",
                                              "not separately measured", "together with"]})),
         "data", "cigre"),
    Item(Task("data_cigre_battery", ("Is Battery 1's output in the measurement data?",),
              ({"kind": "name_hedged", "label": "held at its set-point; the data carry no storage",
                "name": "Battery 1", "hedge_any": ["set-point", "setpoint", "set point", "not in the data",
                                                   "no storage", "not measured", "held at", "fixed", "not included",
                                                   "isn't in the data", "is not in the data", "not part of"]},)),
         "data", "cigre"),
    Item(Task("data_cigre_load_split", ("Is Load R1's demand measured on its own?",),
              ({"kind": "name_hedged", "label": "a share of Bus 1's measurement, split with Load CI1",
                "name": "Load R1", "hedge_any": ["share", "split", "allocated", "estimate", "shared", "not its own",
                                                 "bus total", "apportioned", "not measured individually",
                                                 "not separately measured", "together with", "Load CI1"]},)),
         "data", "cigre"),
    Item(Task("data_ieee30_solved", ("Is Gen_bus12's reactive power measured?",),
              ({"kind": "mentions", "label": "said the power flow solves it",
                # "modeled rather than taken from a measurement" (pilot, correct) was not recognised.
                "any": ["power flow", "solved", "computed", "calculated", "voltage control", "not measured",
                        "isn't measured", "is not measured", "result of", "modeled", "modelled",
                        "not taken from a measurement", "rather than measured", "rather than taken from"]},)),
         "data", "ieee30"),

    # --- Consistency ----------------------------------------------------------
    *(Item(Task(f"para_{k}", (q,),
                ({"kind": "mentions_all", "label": "named the wind and PV units", "all": ["WKA 7", "PV 3", "PV 11"]},
                 {"kind": "name_handled", "label": "did not count CHP diesel 1", "name": "CHP diesel 1",
                  "ok_any": ["not", "independent", "dispatchable", "conventional", "thermal", "fuel", "excluding",
                             "except"]})),
           "consistency", "cigre", group="renewables")
      for k, q in (("weather", "Which units in this network depend on the weather?"),
                   ("vary", "Which generators' output varies with the wind or the sun?"),
                   ("variable", "List the variable renewable units in this network."))),
    Item(Task("persist_ieee30", ("Gen_bus12 is a wind farm.",
                                 "If wind output dropped by 50 % at 2000-01-03 12:00, which units would be "
                                 "affected, and would the grid stay secure?"),
              ({"kind": "mentions_all", "label": "named Gen_bus12 as affected", "all": ["Gen_bus12"]},
               # By a renewable scaling that hits Gen_bus12 only, or by a proposal that lowers it (gen 1) —
               # both answer the question (observed, both arms). Recording the fact for later
               # (declare_network_facts) is reported, not graded: only one arm has the tool.
               {"kind": "unit_studied", "label": "a study lowered Gen_bus12 only",
                "units": ["Gen_bus12"], "gen_index": 1})),
         "consistency", "ieee30"),
)

TASKS = {i.task.id: i for i in ITEMS}
_SCENARIO = {"cigre": "cigre_mv", "ieee30": "ieee30"}


# ---------------------------------------------------------------------------
# Checks beyond `whatif.check`
# ---------------------------------------------------------------------------

def _text(records) -> str:
    return re.sub(r"[*`]+", "", " ".join(r.get("assistant") or "" for r in records)).replace("’", "'")


def _sentences_with(text: str, name: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?;])\s+|\n+", text) if name.casefold() in s.casefold()]


def check(spec: dict, records: list[dict]) -> bool:
    kind = spec["kind"]
    text = _text(records)
    if kind == "name_handled":
        # Not mentioned, or every mention says what it really is.
        hits = _sentences_with(text, spec["name"])
        return all(any(w in s.casefold() for w in spec["ok_any"]) for s in hits)
    if kind == "name_hedged":
        # Mentioned, and said to be solar only by its name (or not stated) — in the sentence that names
        # it or the next ("Solar Plant A is the only unit identified as solar. Its source is inferred from
        # its name" — observed).
        parts = [p for p in re.split(r"(?<=[.!?;])\s+|\n+", text) if p.strip()]
        spots = [" ".join(parts[k:k + 2]) for k, p in enumerate(parts) if spec["name"].casefold() in p.casefold()]
        return bool(spots) and any(w in s.casefold() for s in spots for w in spec["hedge_any"])
    if kind == "none_affected":
        # No unit named as affected, unless the sentence denies it.
        for name in spec["names"]:
            for s in _sentences_with(text, name):
                if re.search(r"affect|reduc|drop|scal|lower|curtail", s, re.IGNORECASE) \
                        and not re.search(_NEG, s, re.IGNORECASE):
                    return False
        return True
    if kind == "unit_studied":
        if check({"kind": "scaled_units", "units": spec["units"]}, records):
            return True
        for rec in records:
            for call, entry in zip(rec.get("tool_calls") or [], rec.get("tool_results") or []):
                res = entry.get("result") if isinstance(entry, dict) else None
                if call.get("name") != "edit_proposal" or not isinstance(res, dict) or res.get("error"):
                    continue
                for e in res.get("edits") or []:
                    targets = e.get("elements") or ([e["element"]] if "element" in e else [])
                    if e.get("element_type") == "gen" and [int(t) for t in targets] == [spec["gen_index"]]:
                        return True
        return False
    if kind == "scaled_units":
        # A study in the turn applied the scaling to exactly these units.
        want = sorted(spec["units"])
        for rec in records:
            for entry in rec.get("tool_results") or []:
                res = entry.get("result") if isinstance(entry, dict) else None
                scaled = (res or {}).get("renewables_scaled") if isinstance(res, dict) else None
                if isinstance(scaled, dict) and sorted(scaled.get("units") or []) == want:
                    return True
        return False
    return _base_check(spec, records)


def evaluate(task: Task, records: list[dict]) -> dict[str, bool]:
    return {spec["label"]: check(spec, records) for spec in task.checks}


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------

def _reset() -> None:
    _post_backend("/api/names", {"reset": True})
    _post_backend("/api/network/declare", {"reset": True})


def _prepare(item: Item) -> None:
    _reset()
    changes = [{"element_type": t, "element": i, "name": n} for t, i, n in item.prepare.get("rename", [])]
    if changes:
        _post_backend("/api/names", {"changes": changes})
    if item.prepare.get("declare"):
        _post_backend("/api/network/declare", {"units": item.prepare["declare"]})


def run(items: list[Item], repeats: int, out: pathlib.Path, arm: str) -> pathlib.Path:
    from .semantics import check_backend_mode
    check_backend_mode(arm)
    out.mkdir(parents=True, exist_ok=True)
    os.environ["CONDUCTOR_ONTOLOGY"] = arm
    for network in ("cigre", "ieee30"):
        batch = [i for i in items if i.network == network]
        if not batch:
            continue
        _post_backend("/api/scenario", {"name": _SCENARIO[network]})
        for item in batch:
            for k in range(1, repeats + 1):
                print(f"[reasoning-{arm}] {item.task.id} run {k}/{repeats}", flush=True)
                try:
                    _prepare(item)
                    records = run_task(item.task, k)
                except Exception as exc:  # noqa: BLE001 — one failed run must not end the batch
                    records = [{"turn": 0, "status": "runner_error", "runner_error": str(exc)}]
                finally:
                    _reset()
                with open(out / f"{item.task.id}-{k:02d}.jsonl", "w", encoding="utf-8") as fh:
                    fh.write(json.dumps({"kind": f"reasoning-{arm}", "task": item.task.id, "repeat": k}) + "\n")
                    for rec in records:
                        fh.write(json.dumps(rec, default=str) + "\n")
    return out


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def grade_folder(folder: pathlib.Path) -> dict:
    """{task: [ {"checks": {label: bool}, "entity": n, "calls", "seconds", "layer_tools": [...]} per run ]}."""
    out: dict = {}
    for path in sorted(folder.glob("*.jsonl")):
        lines = [json.loads(x) for x in open(path, encoding="utf-8") if x.strip()]
        meta, records = lines[0], [r for r in lines[1:] if r.get("status") != "runner_error"]
        item = TASKS.get(meta.get("task"))
        if item is None:
            continue
        st = _turn_stats(records) if records else {"calls": 0, "seconds": 0.0}
        tools = [c.get("name") for r in records for c in r.get("tool_calls") or []]
        declared = any(c.get("name") == "declare_network_facts" for r in records for c in r.get("tool_calls") or [])
        out.setdefault(item.task.id, []).append({
            "declared": declared,
            "repeat": meta.get("repeat"),
            "ok_run": bool(records),
            "checks": evaluate(item.task, records) if records else {s["label"]: False for s in item.task.checks},
            "entity": sum(1 for r in records for f in grade_record(r).findings if f.code.startswith("entity_")),
            "calls": st["calls"], "seconds": st["seconds"],
            "layer_tools": sorted({t for t in tools if t in ("describe_network", "declare_network_facts")}),
        })
    return out


def _rate(runs, pred) -> tuple[int, int]:
    return sum(1 for r in runs if pred(r)), len(runs)


def analyse(on_dir: pathlib.Path, off_dir: pathlib.Path, labels: tuple[str, str] = ("With the layer", "Without")
            ) -> str:
    arms = {"on": grade_folder(on_dir), "off": grade_folder(off_dir)}
    a, b = labels
    lines = ["## Per task (runs with every check passed; checks passed)", "",
             f"| Family | Task | {a} | {b} |", "|---|---|---:|---:|"]
    fam_tot: dict = {}
    for item in ITEMS:
        tid = item.task.id
        cells = []
        for arm in ("on", "off"):
            runs = arms[arm].get(tid, [])
            full, n = _rate(runs, lambda r: all(r["checks"].values()))
            checks = sum(sum(r["checks"].values()) for r in runs)
            total = sum(len(r["checks"]) for r in runs)
            cells.append(f"{full}/{n} ({checks}/{total})")
            f = fam_tot.setdefault((item.family, arm), [0, 0, 0, 0])
            f[0] += full
            f[1] += n
            f[2] += checks
            f[3] += total
        lines.append(f"| {item.family} | {tid} | {cells[0]} | {cells[1]} |")
    lines += ["", "## Per family", "", f"| Family | {a} | {b} |", "|---|---:|---:|"]
    for fam in ("names", "intervention", "composition", "switches", "data", "consistency"):
        on, off = fam_tot.get((fam, "on"), [0] * 4), fam_tot.get((fam, "off"), [0] * 4)
        lines.append(f"| {fam} | runs {on[0]}/{on[1]}, checks {on[2]}/{on[3]} | "
                     f"runs {off[0]}/{off[1]}, checks {off[2]}/{off[3]} |")

    # Intervention: the answer changes when the fact does — both sides of a pair right in the same repeat.
    lines += ["", "## Sensitivity to an intervention (both sides of the pair right, same repeat)", "",
              f"| Pair | {a} | {b} |", "|---|---:|---:|"]
    for item in ITEMS:
        if not item.pair:
            continue
        cells = []
        for arm in ("on", "off"):
            base = {r["repeat"]: all(r["checks"].values()) for r in arms[arm].get(item.pair, [])}
            changed = {r["repeat"]: all(r["checks"].values()) for r in arms[arm].get(item.task.id, [])}
            both = sum(1 for k in changed if changed[k] and base.get(k))
            cells.append(f"{both}/{len(changed)}")
        lines.append(f"| {item.pair} → {item.task.id} | {cells[0]} | {cells[1]} |")

    # Consistency: the paraphrases agree (all right, or all wrong) within a repeat.
    lines += ["", "## Consistency across wordings (same repeat)", "", f"| Group | Measure | {a} | {b} |",
              "|---|---|---:|---:|"]
    groups = sorted({i.group for i in ITEMS if i.group})
    for g in groups:
        ids = [i.task.id for i in ITEMS if i.group == g]
        row_all, row_agree = [], []
        for arm in ("on", "off"):
            reps = sorted({r["repeat"] for t in ids for r in arms[arm].get(t, [])})
            verdicts = [[next((all(r["checks"].values()) for r in arms[arm].get(t, []) if r["repeat"] == k), False)
                         for t in ids] for k in reps]
            row_all.append(f"{sum(all(v) for v in verdicts)}/{len(verdicts)}")
            row_agree.append(f"{sum(len(set(v)) == 1 for v in verdicts)}/{len(verdicts)}")
        lines.append(f"| {g} | all wordings right | {row_all[0]} | {row_all[1]} |")
        lines.append(f"| {g} | wordings agree | {row_agree[0]} | {row_agree[1]} |")

    lines += ["", "## Process", "", f"| | {a} | {b} |", "|---|---:|---:|"]
    for label, key in (("Tool calls per run", "calls"), ("Seconds per run", "seconds"),
                       ("Entity-check findings (total)", "entity")):
        vals = []
        for arm in ("on", "off"):
            runs = [r for rs in arms[arm].values() for r in rs]
            total = sum(r[key] for r in runs)
            vals.append(f"{total:g}" if key == "entity" else f"{total / max(len(runs), 1):.1f}")
        lines.append(f"| {label} | {vals[0]} | {vals[1]} |")
    told = [i.task.id for i in ITEMS if len(i.task.turns) > 1]          # a fact stated in a turn
    cells_rec, cells_used = [], []
    for arm in ("on", "off"):
        rec = sum(1 for t in told for r in arms[arm].get(t, []) if r["declared"])
        n_told = sum(len(arms[arm].get(t, [])) for t in told)
        used = sum(1 for rs in arms[arm].values() for r in rs if r["layer_tools"])
        n = sum(len(rs) for rs in arms[arm].values())
        cells_rec.append(f"{rec}/{n_told}")
        cells_used.append(f"{used}/{n}")
    lines.append(f"| A fact the operator stated, recorded for later studies | {cells_rec[0]} | {cells_rec[1]} |")
    lines.append(f"| Runs that used the layer's tools | {cells_used[0]} | {cells_used[1]} |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evaluation.reasoning", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list")
    p_run = sub.add_parser("run")
    p_run.add_argument("--arm", choices=("on", "off", "none"), required=True)
    p_run.add_argument("--repeats", type=int, default=3)
    p_run.add_argument("--tasks", default="")
    p_run.add_argument("--out", required=True)
    p_run.add_argument("--yes", action="store_true")
    p_an = sub.add_parser("analyse")
    p_an.add_argument("on")
    p_an.add_argument("off")
    args = parser.parse_args(argv)
    if args.command == "list":
        for i in ITEMS:
            print(f"[{i.family}/{i.network}] {i.task.id}: " + " → ".join(i.task.turns))
        return 0
    if args.command == "analyse":
        print(analyse(pathlib.Path(args.on), pathlib.Path(args.off)))
        return 0
    wanted = [s for s in args.tasks.split(",") if s.strip()]
    items = [i for i in ITEMS if not wanted or i.task.id in wanted]
    turns = sum(len(i.task.turns) for i in items) * args.repeats
    print(f"{len(items)} task(s) × {args.repeats} repeats, layer {args.arm}: {turns} agent turns (model credit).")
    if not args.yes and input("Proceed? [y/N] ").strip().lower() != "y":
        return 1
    run(items, args.repeats, pathlib.Path(args.out), args.arm)
    return 0


if __name__ == "__main__":
    sys.exit(main())
