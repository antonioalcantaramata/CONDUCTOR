"""
Error injection — what the answer checks catch, measured on real answers with known errors put into them.

Hand-labelling the checks' findings on real runs measures their precision (how often a flag is right). It cannot
measure their recall: an error no check flagged is invisible in the records. This module measures recall the way a
fault-injection campaign does. It takes real turn records (from any run folder), copies each answer, puts one known
error into the copy — a figure changed, a figure swapped for another element's, an element called what it is not,
a study called secure when it is not — and runs the same checks the live system runs on the copy. An error counts as
caught only when the check flags *that* error (the sentence or the figure injected), never because the copy was
flagged for something else.

Two more things are measured with it:

- **True statements as controls.** The same sentences, worded the same way but true ("PV 3 is a solar PV unit"
  where it is), are injected too: a check that flags them raises false alarms, and the rate is reported beside the
  recall.
- **What no check could catch.** A figure changed to a value that some other result of the same turn happens to
  hold is, by construction, invisible to a check that matches figures to results. Such injections are counted
  apart ("undetectable") rather than as misses.

Each kind of error is tied to the check that claims to catch it, and its wording follows what the check reads, so
what is reported is the reach of each check on real answers — including the blind spots the checks state (a branch
called a transformer *after* its name is not read: the check looks before the name; both wordings are injected).

    python -m evaluation.injection <run folder> [<run folder> …] [--seed 0] [--out <folder>]

Free: no model is called. Deterministic for a seed. Writes `injection.md`, `injection.csv` and `injection.json`
into `--out` (default: the first folder).
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import pathlib
import random
import re
import sys
from collections import defaultdict
from typing import Callable, NamedTuple

from llm_agent.agent import characterisation as _char
from llm_agent.agent import entities as _ent
from llm_agent.agent import evidence as _evid
from llm_agent.agent import provenance as _prov

_ROOT = pathlib.Path(__file__).resolve().parents[1]


class Injection(NamedTuple):
    kind: str
    level: str              # L1 figures / L2 entities
    check: str              # the check that should catch it
    control: bool           # a true statement: it should NOT be flagged
    answer: str             # the answer with the error put in
    target: dict            # what was injected (to recognise the check's finding of it)
    record: dict            # the original record (its results are unchanged)


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------

def load_records(folders) -> list[dict]:
    """Every answered turn in the folders' run files, once (the same answer twice is one record)."""
    seen, out = set(), []
    for folder in folders:
        for path in sorted(pathlib.Path(folder).rglob("*.jsonl")):
            if "_replaced" in path.parts:                       # records set aside by a re-run
                continue
            for line in list(open(path, encoding="utf-8"))[1:]:
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                answer = rec.get("assistant") or ""
                if rec.get("status") == "runner_error" or not answer or "tool_results" not in rec:
                    continue
                if _prov.is_clarifying_question(answer, rec.get("tool_calls") or ()):
                    continue
                key = (rec.get("user"), answer)
                if key not in seen:
                    seen.add(key)
                    rec["_source"] = str(path)
                    out.append(rec)
    return out


# The full facts of the networks the demos load, built offline as the backend builds them, so an injected answer
# is checked against what the live system would fetch for it (the record keeps only what its own answer named).
_FULL: dict[str, dict | None] = {}


def _full_facts(record: dict) -> dict | None:
    name = str((record.get("grid") or {}).get("name") or "")
    if name in _FULL:
        return _FULL[name]
    backend = str(_ROOT / "backend")
    if backend not in sys.path:
        sys.path.insert(0, backend)
    facts = None
    try:
        import semantics as sm
        if name.startswith("CIGRE MV"):
            import pandapower.networks as pn
            facts = sm.compact(sm.build(pn.create_cigre_network_mv(with_der="all")))
        elif "IEEE 30" in name:
            import network_loader as nl
            net = nl.load_network(nl.load_profile("pglib_case30"))
            net, _, _ = nl.gen_to_sgen(net)
            facts = sm.compact(sm.build(net))
    except Exception:  # noqa: BLE001 — an unknown network falls back to the record's own facts
        facts = None
    _FULL[name] = facts
    return facts


def _all_names(facts: dict) -> list[str]:
    return [*facts.get("buses", {}), *facts.get("loads", {}), *facts.get("units", {}),
            *facts.get("branches", {}), *(k.split(" | ")[0] for k in facts.get("branches", {}))]


def facts_for(record: dict, answer: str) -> dict:
    """The record's `network_facts` as the live loop would have built them for `answer`: the record's own facts,
    plus the network's facts about elements only the new answer names, and the names it uses that the network
    does not have (`loop._network_facts`)."""
    out = copy.deepcopy(record.get("network_facts") or {})
    full = _full_facts(record)
    if not full:
        return out
    for f in out.values():
        for key in ("units", "buses", "branches", "loads"):
            have = f.setdefault(key, {})
            for name, fact in (full.get(key) or {}).items():
                if name not in have and re.search(r"(?<![\w])" + re.escape(name.split(" | ")[0]) + r"(?![\w])",
                                                  answer):
                    have[name] = fact
        names = set(_all_names(full)) | set(_all_names(f)) | {u for us in (f.get("units_by_source") or {}).values()
                                                               for u in us}
        f["names_not_in_network"] = _ent.names_not_in_network(answer, names)
    return out


def _with(record: dict, answer: str) -> dict:
    return {**record, "assistant": answer, "network_facts": facts_for(record, answer)}


# ---------------------------------------------------------------------------
# Injectors — each returns the injections one record admits (possibly none)
# ---------------------------------------------------------------------------

_OWN_NOTES = re.compile(r"\n+\**Reading check\.\**|\n+CONDUCTOR (?:used all|stopped this question)")


def _append(answer: str, sentence: str) -> str:
    """The answer with one more sentence where the model's own text ends — before the notes CONDUCTOR appends
    (the reading check's, the step limit's), as the model would have written it."""
    m = _OWN_NOTES.search(answer)
    if not m:
        return answer.rstrip() + "\n\n" + sentence
    return answer[:m.start()].rstrip() + "\n\n" + sentence + answer[m.start():]


def _fmt(value: float, decimals: int) -> str:
    return f"{abs(value):.{decimals}f}"


def _replace_number(text: str, old: str, new: str) -> str | None:
    pattern = re.compile(r"(?<![\w.])" + re.escape(old) + r"(?![\w]|\.\d)")
    m = pattern.search(text)
    return None if not m else text[:m.start()] + new + text[m.end():]


def _results(record: dict):
    for entry in record.get("tool_results") or []:
        res = entry.get("result") if isinstance(entry, dict) else None
        if isinstance(res, dict):
            yield entry.get("name"), res


def inject_perturbed_figure(record: dict, rng: random.Random) -> list[Injection]:
    """A grounded figure changed by a few percent: the commonest fabrication (a figure from no result)."""
    report = _prov.check_turn(record)
    grounded = [v for v in report.verdicts if v.status == "grounded" and abs(v.claim.value) >= 0.01]
    rng.shuffle(grounded)
    for v in grounded:
        d = v.claim.decimals
        factor = rng.choice((0.85, 0.92, 1.08, 1.17))
        new = round(v.claim.value * factor, d)
        old_s, new_s = _fmt(v.claim.value, d), _fmt(new, d)
        if new_s == old_s:
            continue
        answer = _replace_number(record["assistant"], old_s, new_s)
        if answer:
            return [Injection("figure_changed", "L1", "provenance", False, answer,
                              {"value": float(new_s) * (1 if v.claim.value >= 0 else -1), "decimals": d,
                               "was": v.claim.value}, record)]
    return []


def inject_swapped_figure(record: dict, rng: random.Random) -> list[Injection]:
    """A violated element's figure replaced by another violated element's: real, correctly transcribed, about
    the wrong element (misattribution)."""
    answer = record["assistant"]
    for _name, res in _results(record):
        viol = [v for v in res.get("violations") or [] if isinstance(v, dict) and v.get("element")
                and isinstance(v.get("value"), (int, float))]
        pairs = [(a, b) for a in viol for b in viol if a is not b and a.get("violation_type") == b.get("violation_type")
                 and a["element"] != b["element"]]
        rng.shuffle(pairs)
        for a, b in pairs:
            for d in (3, 2, 1, 4):
                old_s, new_s = _fmt(a["value"], d), _fmt(b["value"], d)
                if old_s == new_s:
                    continue
                for sentence in _ent._sentences(answer):
                    if a["element"] in sentence and re.search(r"(?<![\w.])" + re.escape(old_s) + r"(?![\w]|\.\d)",
                                                              sentence):
                        new_sentence = _replace_number(sentence, old_s, new_s)
                        return [Injection("figure_swapped", "L1", "provenance", False,
                                          answer.replace(sentence, new_sentence, 1),
                                          {"value": float(new_s), "decimals": d, "about": a["element"],
                                           "belongs_to": b["element"]}, record)]
    return []


_ASSESSING = {"run_rsa", "simulate_contingency", "simulate_all_contingencies", "scan_rsa_over_time",
              "find_worst_case_timestamp", "investigate"}
_CORRECTING = {"optimize_flexibility", "optimize_contingency", "optimize_robust_flexibility", "flexibility_over_time",
               "compare_networks", "evaluate_kpis", "kpis_over_time", "edit_proposal", "compare_results"}


def inject_secure_claim(record: dict, rng: random.Random) -> list[Injection]:
    """An insecure assessment called secure. Only where nothing else in the turn could make "secure" true (no
    corrective dispatch, no comparison with a proposal)."""
    names = {n for n, _ in _results(record)}
    if names & _CORRECTING or not names & _ASSESSING:
        return []
    insecure = any(res.get("secure") is False or (isinstance(res.get("total_violations"), int)
                                                  and res["total_violations"] > 0)
                   for n, res in _results(record) if n in _ASSESSING)
    if not insecure:
        return []
    sentence = rng.choice(("Overall, the grid is secure, with no violations.",
                           "All voltages and loadings are within limits."))
    return [Injection("insecure_called_secure", "L1", "characterisation", False,
                      _append(record["assistant"], sentence), {"sentence": sentence}, record)]


_CERTIFICATE = re.compile(r"lower_bound|best_possible|certificate|proved|no_dispatch_exists|relaxation", re.IGNORECASE)


def inject_overstated(record: dict, rng: random.Random) -> list[Injection]:
    """A claim worded beyond its evidence: a dispatch "proved" optimal with no certificate in the results, or a
    grid called secure with no study behind it."""
    names = {n for n, _ in _results(record)}
    if names & {"optimize_flexibility", "optimize_contingency", "optimize_robust_flexibility"} \
            and not _CERTIFICATE.search(json.dumps(record.get("tool_results"), default=str)):
        sentence = "This dispatch is proved to be the best possible."
        return [Injection("proof_without_certificate", "L1", "evidence", False,
                          _append(record["assistant"], sentence), {"sentence": sentence}, record)]
    if names and names <= set(_prov.LOOKUP_TOOLS):
        sentence = "The grid is secure."
        return [Injection("secure_without_study", "L1", "evidence", False,
                          _append(record["assistant"], sentence), {"sentence": sentence}, record)]
    return []


_SOURCE_PHRASE = {"wind": "a wind turbine", "solar": "a solar PV unit", "thermal": "a diesel CHP unit",
                  "hydro": "a hydro unit", "storage": "a battery"}


def _named_units(record: dict) -> dict:
    """Units the answer names and the question does not (what the operator says about a unit is theirs to say:
    the check exempts it by design, so an injection there would measure the exemption, not the check)."""
    question = record.get("user") or ""
    return {k: v for f in (record.get("network_facts") or {}).values() for k, v in (f.get("units") or {}).items()
            if re.search(r"(?<![\w])" + re.escape(k) + r"(?![\w])", record["assistant"])
            and not re.search(r"(?<![\w])" + re.escape(k) + r"(?![\w])", question)}


def inject_unit_source(record: dict, rng: random.Random) -> list[Injection]:
    """A unit called a source the network states otherwise (contradicted), one whose source the network does not
    state (unsupported), and — as a control — the true source."""
    out = []
    units = list(_named_units(record).items())
    rng.shuffle(units)
    stated = [(u, f) for u, f in units if f.get("basis") in ("stated", "declared") and f.get("source") in _SOURCE_PHRASE]
    if stated:
        u, f = stated[0]
        wrong = rng.choice([s for s in ("wind", "solar", "thermal") if s != f["source"]])
        sentence = f"{u} is {_SOURCE_PHRASE[wrong]}."
        out.append(Injection("unit_source_contradicted", "L2", "entities", False,
                             _append(record["assistant"], sentence), {"sentence": sentence}, record))
        sentence = f"{u} is {_SOURCE_PHRASE[f['source']]}."
        out.append(Injection("unit_source_true", "L2", "entities", True,
                             _append(record["assistant"], sentence), {"sentence": sentence}, record))
    unknown = [(u, f) for u, f in units if f.get("basis") == "unknown"]
    if unknown:
        u, _f = unknown[0]
        sentence = f"{u} is {_SOURCE_PHRASE[rng.choice(('wind', 'solar'))]}."
        out.append(Injection("unit_source_unsupported", "L2", "entities", False,
                             _append(record["assistant"], sentence), {"sentence": sentence}, record))
    return out


def inject_voltage(record: dict, rng: random.Random) -> list[Injection]:
    """A bus put at a voltage the network does not give it, and — as a control — its own."""
    facts = facts_for(record, record["assistant"])
    levels = {float(v) for f in facts.values() for v in f.get("voltage_levels_kv") or []}
    buses = [(b, f) for net in facts.values() for b, f in (net.get("buses") or {}).items()
             if f.get("kv_basis") == "stated" and f.get("kv") is not None
             and re.search(r"(?<![\w])" + re.escape(b) + r"(?![\w])", record["assistant"])]
    if not buses:
        return []
    b, f = rng.choice(buses)
    wrong = next(kv for kv in (132.0, 66.0, 33.0, 400.0) if kv not in levels)
    out = []
    for kv, control in ((wrong, False), (float(f["kv"]), True)):
        sentence = f"{b} is at {kv:g} kV."
        out.append(Injection("bus_voltage_true" if control else "bus_voltage_wrong", "L2", "entities", control,
                             _append(record["assistant"], sentence), {"sentence": sentence}, record))
    return out


def inject_branch_kind(record: dict, rng: random.Random) -> list[Injection]:
    """A line called a transformer, worded two ways: before its name (what the check reads) and after it (a blind
    spot the check states)."""
    facts = facts_for(record, record["assistant"])
    lines = [k.split(" | ")[0] for net in facts.values() for k, f in (net.get("branches") or {}).items()
             if f.get("type") == "line" and re.search(r"(?<![\w])" + re.escape(k.split(" | ")[0]) + r"(?![\w])",
                                                      record["assistant"])]
    if not lines:
        return []
    line = rng.choice(lines)
    out = []
    for kind, sentence in (("line_called_transformer_before", f"The transformer {line} is the most loaded branch."),
                           ("line_called_transformer_after", f"{line} is a transformer.")):
        out.append(Injection(kind, "L2", "entities", False, _append(record["assistant"], sentence),
                             {"sentence": sentence}, record))
    return out


def inject_measured(record: dict, rng: random.Random) -> list[Injection]:
    """A value called measured that was derived from an assumption (or solved, set, nominal), and — as a
    control — one that was measured."""
    els = {k: v for f in (record.get("network_facts") or {}).values()
           for part in ("units", "loads") for k, v in (f.get(part) or {}).items()
           if re.search(r"(?<![\w])" + re.escape(k) + r"(?![\w])", record["assistant"])}
    out = []
    derived = [(k, q) for k, v in els.items() for q in ("q", "p") if v.get(f"{q}_kind") in ("derived", "solved", "set",
                                                                                           "nominal")]
    if derived:
        k, q = rng.choice(derived)
        sentence = f"The {'reactive' if q == 'q' else 'active'} power of {k} is measured."
        out.append(Injection("value_called_measured", "L2", "entities", False,
                             _append(record["assistant"], sentence), {"sentence": sentence}, record))
    measured = [(k, q) for k, v in els.items() for q in ("p", "q") if v.get(f"{q}_kind") == "measured"]
    if measured:
        k, q = rng.choice(measured)
        sentence = f"The {'reactive' if q == 'q' else 'active'} power of {k} is measured."
        out.append(Injection("value_measured_true", "L2", "entities", True,
                             _append(record["assistant"], sentence), {"sentence": sentence}, record))
    return out


def inject_invented_name(record: dict, rng: random.Random) -> list[Injection]:
    """A real element's name in the answer replaced by one shaped like the network's names that the network does
    not have ("Load R9" where there is no R9 — observed live)."""
    full = _full_facts(record)
    if not full:
        return []
    names = [n for n in _all_names(full) if re.search(r"\d", n)]
    shapes: dict[str, list[str]] = defaultdict(list)
    for n in names:
        shapes[_ent._template(n)].append(n)
    candidates = [n for shape, ns in shapes.items() if len(ns) >= 3 and re.search(r"[A-Za-z]", shape) for n in ns
                  if " | " not in n and re.search(r"(?<![\w])" + re.escape(n) + r"(?![\w])", record["assistant"])]
    if not candidates:
        return []
    real = rng.choice(candidates)
    numbers = [int(x) for n in shapes[_ent._template(real)] for x in re.findall(r"\d+", n)]
    m = list(re.finditer(r"\d+", real))[-1]
    invented = real[:m.start()] + str(max(numbers) + 50) + real[m.end():]
    # Only in the model's own text, not in the notes CONDUCTOR appends.
    text = record["assistant"]
    m_notes = _OWN_NOTES.search(text)
    own, notes = (text[:m_notes.start()], text[m_notes.start():]) if m_notes else (text, "")
    if not re.search(r"(?<![\w])" + re.escape(real) + r"(?![\w])", own):
        return []
    answer = re.sub(r"(?<![\w])" + re.escape(real) + r"(?![\w])", invented, own, count=1) + notes
    return [Injection("invented_element", "L2", "entities", False, answer, {"name": invented, "was": real}, record)]


INJECTORS: tuple[Callable, ...] = (inject_perturbed_figure, inject_swapped_figure, inject_secure_claim,
                                   inject_overstated, inject_unit_source, inject_voltage, inject_branch_kind,
                                   inject_measured, inject_invented_name)


# ---------------------------------------------------------------------------
# Did the check catch it?
# ---------------------------------------------------------------------------

def _same_value(a: float, b: float, decimals: int) -> bool:
    return abs(a - b) <= 0.5 * 10 ** -decimals + 1e-12


def undetectable(inj: Injection) -> bool:
    """Errors no check that matches figures to results can see: a changed figure that some result of the turn
    holds anyway; a swapped figure that is also held by the element the sentence is about, or by nothing in
    particular (a value outside any element's record) — clustered voltages make this common."""
    if inj.kind not in ("figure_changed", "figure_swapped"):
        return False
    matches = [src for src in _prov.collect_sources(inj.record)
               if isinstance(getattr(src, "value", None), (int, float))
               and _same_value(float(src.value), inj.target["value"], inj.target["decimals"])]
    if inj.kind == "figure_changed":
        return bool(matches)
    about = inj.target["about"]
    return any(not src.scopes or about in src.scopes for src in matches)


def caught(inj: Injection) -> tuple[bool, str]:
    """Whether the check flagged this injection (its figure or its sentence), and how."""
    rec = _with(inj.record, inj.answer)
    if inj.check == "provenance":
        for v in _prov.check_turn(rec).verdicts:
            if _same_value(v.claim.value, inj.target["value"], inj.target["decimals"]):
                if v.status in ("ungrounded", "misattributed"):
                    return True, v.status
                return False, v.status
        return False, "not extracted"
    if inj.check == "characterisation":
        # A study-level conflict quotes the start of the answer, not the sentence: caught means the injected
        # answer has a conflict over a study that the original answer did not have.
        before = {(c.path, c.field) for c in _char.check_study_claims(inj.record.get("assistant") or "", inj.record)}
        after = {(c.path, c.field) for c in _char.check_study_claims(inj.answer, rec)}
        hit = bool(after - before)
        return hit, "conflict" if hit else ("already flagged" if before else "none")
    if inj.check == "evidence":
        hit = any(inj.target["sentence"][:25] in c.text for c in _evid.build(rec).overstated)
        return hit, "overstated" if hit else "none"
    issues = _ent.check_answer(inj.answer, rec)
    if inj.kind == "invented_element":
        hit = [i for i in issues if inj.target["name"] in i.note]
    else:
        hit = [i for i in issues if inj.target["sentence"] in i.text]
    return bool(hit), (hit[0].kind if hit else "none")


# ---------------------------------------------------------------------------
# Running and reporting
# ---------------------------------------------------------------------------

def run(records: list[dict], seed: int = 0) -> list[dict]:
    """One row per injection: its kind, level, check, whether it was caught (or could not be)."""
    rows = []
    for i, rec in enumerate(records):
        for inject in INJECTORS:
            rng = random.Random(f"{seed}-{i}-{inject.__name__}")
            try:
                injections = inject(rec, rng)
            except Exception:  # noqa: BLE001 — a record an injector cannot handle is skipped, not fatal
                continue
            for inj in injections:
                blind = undetectable(inj)
                hit, how = caught(inj)
                rows.append({"kind": inj.kind, "level": inj.level, "check": inj.check, "control": inj.control,
                             "caught": hit, "how": how, "undetectable": blind, "record": i,
                             "question": (rec.get("user") or "")[:120], "injected": json.dumps(inj.target)})
    return rows


_CODES = ("ungrounded_figure", "misattributed_figure", "mischaracterised_study", "mischaracterised_figure",
          "overstated_claim", "entity_contradicted", "entity_unsupported")


def baseline(records: list[dict]) -> dict:
    """What the same checks flag on the answers as they were written (facts rebuilt as the live loop builds them):
    the findings a reader would see with no error injected. Each needs labelling by hand before it is called a
    false alarm — the agent's own answers do contain errors."""
    from .report import grade_record
    counts: dict[str, int] = defaultdict(int)
    examples: dict[str, list[str]] = defaultdict(list)
    findings: list[dict] = []
    for rec in records:
        for f in grade_record(_with(rec, rec["assistant"])).findings:
            if f.code in _CODES:
                counts[f.code] += 1
                if len(examples[f.code]) < 5:
                    examples[f.code].append(f.detail[:220])
                findings.append({"source": rec.get("_source", ""), "turn": rec.get("turn", ""), "code": f.code,
                                 "finding": f.detail, "question": rec.get("user", ""), "answer": rec["assistant"],
                                 "label": "", "note": ""})
    return {"answers": len(records), "counts": dict(counts), "examples": dict(examples), "findings": findings}


def summarise(rows: list[dict]) -> list[dict]:
    from .ablation_stats import wilson
    by: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by[r["kind"]].append(r)
    out = []
    for kind, rs in by.items():
        first = rs[0]
        scored = [r for r in rs if not r["undetectable"]]
        k = sum(r["caught"] for r in scored)
        out.append({"kind": kind, "level": first["level"], "check": first["check"], "control": first["control"],
                    "injected": len(rs), "undetectable": len(rs) - len(scored), "scored": len(scored),
                    "flagged": k, "rate": k / len(scored) if scored else None,
                    "ci": wilson(k, len(scored)),
                    "how": dict(sorted({h: sum(r["how"] == h for r in scored) for h in {r["how"] for r in scored}}
                                       .items()))})
    order = {"L1": 0, "L2": 1}
    return sorted(out, key=lambda s: (order.get(s["level"], 9), s["control"], s["kind"]))


def render(summary: list[dict], n_records: int, seed: int, base: dict | None = None) -> str:
    lines = [f"# Error injection — {n_records} answered turns, seed {seed}", "",
             "Recall: injected errors the check flagged (excluding those no figure check could see). False alarms: "
             "true statements, worded the same way, that the check flagged. 95 % Wilson intervals.", "",
             "| Level | Error put into the answer | Check | Injected | Undetectable | Flagged | Rate | 95 % CI |",
             "|---|---|---|---:|---:|---:|---:|---|"]
    for s in summary:
        rate = "–" if s["rate"] is None else f"{100 * s['rate']:.0f} %"
        label = s["kind"] + (" (control: true)" if s["control"] else "")
        lines.append(f"| {s['level']} | {label} | {s['check']} | {s['injected']} | {s['undetectable']} | "
                     f"{s['flagged']}/{s['scored']} | {rate} | [{100 * s['ci'][0]:.0f}, {100 * s['ci'][1]:.0f}] |")
    lines += ["", "How each was judged (verdict of the check on the injected item):", ""]
    lines += [f"- {s['kind']}: " + ", ".join(f"{h} ×{n}" for h, n in s["how"].items()) for s in summary]
    if base:
        lines += ["", f"The same checks on the {base['answers']} answers as written (no error injected) — label each "
                  "before calling it a false alarm:", ""]
        lines += [f"- {code}: {n} ({100 * n / max(base['answers'], 1):.1f} per 100 answers)"
                  for code, n in sorted(base["counts"].items())] or ["- none"]
    return "\n".join(lines)


def write(out: pathlib.Path, rows: list[dict], summary: list[dict], n_records: int, seed: int,
          base: dict | None = None) -> str:
    out.mkdir(parents=True, exist_ok=True)
    md = render(summary, n_records, seed, base)
    (out / "injection.md").write_text(md + "\n", encoding="utf-8")
    (out / "injection.json").write_text(json.dumps({"seed": seed, "records": n_records, "summary": summary,
                                                    "baseline": base, "rows": rows}, indent=1), encoding="utf-8")
    with open(out / "injection.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]) if rows else ["kind"])
        w.writeheader()
        w.writerows(rows)
    # The findings on the answers as written, one per row, with an empty `label` column to fill by hand
    # (error / false alarm / ambiguous): the precision of the checks. Left alone when it exists, so labels survive.
    found = (base or {}).get("findings") or []
    sheet = out / "findings-to-label.csv"
    if found and not sheet.exists():
        with open(sheet, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(found[0]))
            w.writeheader()
            w.writerows(found)
    return md


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evaluation.injection", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folders", nargs="+", help="run folders whose records are used (any *.jsonl inside)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="", help="where to write the tables (default: the first folder)")
    args = parser.parse_args(argv)
    records = load_records(args.folders)
    if not records:
        print("No answered turns found in those folders.", file=sys.stderr)
        return 2
    rows = run(records, args.seed)
    print(write(pathlib.Path(args.out or args.folders[0]), rows, summarise(rows), len(records), args.seed,
                baseline(records)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
