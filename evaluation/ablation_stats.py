"""
Grading and statistics for an ablation's records (`evaluation.ablation`).

Free: reads the records, calls no model. Every arm is graded the same way — the
suite's own checks (facts of the network or of the tool calls, never a judge)
and every grader of `evaluation.report` — and summarised with what a paper
needs to say how sure it is:

- rates with a Wilson 95 % interval (checks passed; tasks with every check
  passed);
- the difference from the baseline arm with a 95 % interval from a bootstrap
  over tasks (runs of one task are not independent: the task is the unit), and
  a two-sided sign test over tasks (how many tasks the arm did better on,
  worse on, the same);
- the cost per run: tool calls, model calls, input and output tokens, seconds.

Written to the run folder: `analysis.md`, `summary.csv`, `runs.csv`,
`checks.csv` (one row per run and check — the long form for any plot),
`summary.tex` (a booktabs table) and `results.json`.
"""

from __future__ import annotations

import csv
import json
import math
import pathlib
import random
from collections import defaultdict

Z95 = 1.959964


# ---------------------------------------------------------------------------
# Reading the records
# ---------------------------------------------------------------------------

def run_files(folder: pathlib.Path):
    """Every run's record file — not those a re-run set aside (runs/_replaced/)."""
    yield from sorted(p for p in (pathlib.Path(folder) / "runs").glob("*/*/*.jsonl") if "_replaced" not in p.parts)


def read_run(path: pathlib.Path) -> tuple[dict, list[dict]]:
    lines = [json.loads(x) for x in open(path, encoding="utf-8") if x.strip()]
    return (lines[0], lines[1:]) if lines else ({}, [])


# Turn statuses that say the infrastructure failed, not the agent: the model or its API (quota, rate limit,
# timeout, server error), an unexpected exception, or an answer the model could not write. Such a run measures
# nothing about the arm: it is run again on resume and left out of the grading (counted as failed). The agent's own
# outcomes — stopping at the step limit, repeating a failed call — are graded like any other answer.
INFRA_STATUSES = frozenset({"runner_error", "quota_exhausted", "rate_limit", "infra_error", "timeout",
                            "execution_error", "completed_with_warning"})
# Failures no retry fixes: the run stops at once.
FATAL_STATUSES = frozenset({"quota_exhausted"})


def failure(records: list[dict]) -> str | None:
    """Why a run measures nothing (its first infrastructure status), or None."""
    if not records:
        return "no records"
    return next((r.get("status") for r in records if r.get("status") in INFRA_STATUSES), None)


def completed(path: pathlib.Path, arm_hash: str) -> bool:
    """A run to keep on resume: made under the same settings, and not failed by the infrastructure."""
    if not path.exists():
        return False
    meta, records = read_run(path)
    return meta.get("arm_hash") == arm_hash and failure(records) is None


_GRAVITY = {"holds": 0, "shifts": 1, "changes": 2}


def _reading(record: dict) -> str | None:
    """The reading check's verdict on a turn: `changes` if any other reading changed the conclusion, else `shifts`
    if one moved its figure by more than 10 %, else `holds`; None when no study was re-checked."""
    outcomes = [o.get("status") for o in (record.get("reading_check") or {}).get("outcomes") or []
                if isinstance(o, dict) and o.get("status") in _GRAVITY]
    return max(outcomes, key=_GRAVITY.get) if outcomes else None


def grade_runs(folders) -> list[dict]:
    """One row per run: its checks and its cost, graded now (the checks of today's code)."""
    from .report import grade_record
    from .suites import by_key
    items = by_key()
    rows = []
    for folder in folders:
        for path in run_files(folder):
            meta, records = read_run(path)
            item = items.get(f"{meta.get('suite')}/{meta.get('task')}")
            if item is None:
                continue
            why = failure(records)
            error = why is not None
            good = [] if error else records
            checks = item.evaluate(item.task, good) if good else {}
            grades = [grade_record(r) for r in good]
            first = good[0] if good else {}
            rows.append({
                "arm": meta.get("arm"), "arm_hash": meta.get("arm_hash"), "suite": item.suite,
                "task": item.task.id, "family": item.family, "repeat": meta.get("repeat"),
                "runner_error": error, "failure": why, "checks": checks,
                "n_checks": len(checks), "n_passed": sum(checks.values()),
                "passed_all": bool(checks) and all(checks.values()),
                "turns": len(good),
                "tool_calls": sum(len(r.get("tool_calls") or []) for r in good),
                "model_calls": sum(int(r.get("model_calls") or 0) for r in good),
                "input_tokens": sum(sum(r.get("prompt_tokens") or []) for r in good),
                "output_tokens": sum(sum(r.get("output_tokens") or []) for r in good),
                "cached_tokens": sum(sum(r.get("cached_tokens") or []) for r in good),
                "seconds": round(sum(float((r.get("timing") or {}).get("total_s") or 0.0) for r in good), 2),
                "figures": sum(g.n_claims for g in grades),
                "ungrounded": sum(g.n_ungrounded + g.n_misattributed for g in grades),
                "mischaracterised": sum(g.n_mischaracterised for g in grades),
                "entity_errors": sum(1 for g in grades for f in g.findings if f.code.startswith("entity_")),
                "failing_turns": sum(1 for g in grades if g.failed),
                # The reading check, per turn: the gravest outcome over the readings it re-ran.
                "reading_checked": sum(1 for r in good if (r.get("reading_check") or {}).get("outcomes")),
                "reading_shifts": sum(1 for r in good if _reading(r) == "shifts"),
                "reading_changes": sum(1 for r in good if _reading(r) == "changes"),
                "provider": first.get("provider"), "model": first.get("model"),
            })
    return rows


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def wilson(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    """The Wilson score interval of a proportion k/n (0, 1 for n = 0)."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def _per_task(rows, arm, metric) -> dict[str, float]:
    acc: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        if r["arm"] == arm and not r["runner_error"]:
            acc[f"{r['suite']}/{r['task']}"].append(metric(r))
    return {k: sum(v) / len(v) for k, v in acc.items() if v}


def check_rate(r) -> float:
    return r["n_passed"] / r["n_checks"] if r["n_checks"] else 0.0


def task_success(r) -> float:
    return float(r["passed_all"])


def paired_difference(rows, arm, baseline, metric=check_rate, n_boot: int = 5000, seed: int = 0) -> dict:
    """The arm minus the baseline, averaged over the tasks both ran, with a 95 % bootstrap interval over
    tasks and a two-sided sign test (ties dropped)."""
    a, b = _per_task(rows, arm, metric), _per_task(rows, baseline, metric)
    tasks = sorted(set(a) & set(b))
    if not tasks:
        return {"tasks": 0, "diff": None, "lo": None, "hi": None, "better": 0, "worse": 0, "same": 0, "p": None}
    diffs = [a[t] - b[t] for t in tasks]
    rng = random.Random(seed)
    boots = sorted(sum(rng.choice(diffs) for _ in diffs) / len(diffs) for _ in range(n_boot))
    better = sum(d > 1e-12 for d in diffs)
    worse = sum(d < -1e-12 for d in diffs)
    return {"tasks": len(tasks), "diff": sum(diffs) / len(diffs), "lo": boots[int(0.025 * n_boot)],
            "hi": boots[int(0.975 * n_boot) - 1], "better": better, "worse": worse,
            "same": len(tasks) - better - worse, "p": sign_test(better, worse)}


def sign_test(better: int, worse: int) -> float:
    """Two-sided exact binomial (p = 1/2) on the tasks that differ."""
    n = better + worse
    if n == 0:
        return 1.0
    k = min(better, worse)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def summarise(rows: list[dict], arms: list[str], baseline: str | None = None) -> list[dict]:
    baseline = baseline or (arms[0] if arms else None)
    out = []
    for arm in arms:
        mine = [r for r in rows if r["arm"] == arm]
        ok = [r for r in mine if not r["runner_error"]]
        k, n = sum(r["n_passed"] for r in ok), sum(r["n_checks"] for r in ok)
        s, m = sum(r["passed_all"] for r in ok), len(ok)

        def per(key: str, ok=ok) -> float:
            return sum(r.get(key) or 0 for r in ok) / len(ok) if ok else 0.0

        entry = {
            "arm": arm, "runs": len(mine), "runner_errors": len(mine) - len(ok),
            "checks_passed": k, "checks": n, "check_rate": k / n if n else None,
            "check_rate_ci": wilson(k, n), "tasks_passed": s, "task_runs": m,
            "task_success": s / m if m else None, "task_success_ci": wilson(s, m),
            **{f"{key}_per_run": per(key) for key in ("tool_calls", "model_calls", "input_tokens", "cached_tokens", "output_tokens",
                                                       "seconds", "figures", "ungrounded", "mischaracterised",
                                                       "entity_errors", "failing_turns")},
            "models": sorted({f"{r['provider']}/{r['model']}" for r in ok if r.get("model")}),
            "reading_checked": sum(r.get("reading_checked", 0) for r in ok),
            "reading_shifts": sum(r.get("reading_shifts", 0) for r in ok),
            "reading_changes": sum(r.get("reading_changes", 0) for r in ok),
        }
        if baseline and arm != baseline:
            entry["vs_baseline"] = paired_difference(rows, arm, baseline)
            entry["vs_baseline_success"] = paired_difference(rows, arm, baseline, task_success)
        out.append(entry)
    return out


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def _pct(x) -> str:
    return "–" if x is None else f"{100 * x:.1f}"


def _ci(ci) -> str:
    return f"[{100 * ci[0]:.0f}, {100 * ci[1]:.0f}]"


def _delta(d: dict) -> str:
    if not d or d.get("diff") is None:
        return "–"
    return (f"{100 * d['diff']:+.1f} [{100 * d['lo']:+.0f}, {100 * d['hi']:+.0f}]; "
            f"{d['better']}↑ {d['worse']}↓ {d['same']}= p={d['p']:.2g}")


def render_markdown(rows: list[dict], arms: list[str], baseline: str | None = None, manifest: dict | None = None
                    ) -> str:
    summary = summarise(rows, arms, baseline)
    baseline = baseline or (arms[0] if arms else "")
    lines = [f"# Ablation{': ' + manifest['study'] if manifest and manifest.get('study') else ''}", ""]
    if manifest:
        if manifest.get("question"):
            lines += [manifest["question"], ""]
        lines += [f"Commit `{(manifest.get('commit') or '?')[:10]}`"
                  f"{' (uncommitted changes)' if manifest.get('uncommitted_changes') else ''} · "
                  f"{manifest.get('repeats')} repeats · suites: {', '.join(manifest.get('suites') or [])} · "
                  f"order: {manifest.get('order', '?')}", ""]
        lines += ["| Arm | Settings |", "|---|---|"]
        lines += [f"| {a['name']} | {a.get('describe', '')} |" for a in manifest.get("arms") or []]
        lines.append("")
    lines += [f"Baseline: **{baseline}**. Rates in %, 95 % Wilson interval; Δ is the arm minus the baseline over "
              "the tasks both ran, with a 95 % bootstrap interval over tasks, then tasks better ↑ / worse ↓ / "
              "same = and the sign test's p.", "",
              "| Arm | Runs | Checks passed | Δ checks | Tasks fully passed | Δ tasks | Tool calls | Input tok | "
              "of which cached | Output tok | s | Ungrounded figures | Entity errors |",
              "|---|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for s in summary:
        errors = f" ({s['runner_errors']} err)" if s["runner_errors"] else ""
        lines.append(
            f"| {s['arm']} | {s['runs']}{errors} | "
            f"{_pct(s['check_rate'])} {_ci(s['check_rate_ci'])} | {_delta(s.get('vs_baseline'))} | "
            f"{_pct(s['task_success'])} {_ci(s['task_success_ci'])} | {_delta(s.get('vs_baseline_success'))} | "
            f"{s['tool_calls_per_run']:.1f} | {s['input_tokens_per_run']:,.0f} | {s['cached_tokens_per_run']:,.0f} | "
            f"{s['output_tokens_per_run']:,.0f} | "
            f"{s['seconds_per_run']:.0f} | {s['ungrounded_per_run']:.2f} | {s['entity_errors_per_run']:.2f} |")
    lines.append("")
    if any(s["reading_checked"] for s in summary):
        lines += ["## The reading check", "",
                  "Turns whose decisive studies were re-run under the readings the question left open, and how "
                  "often another reading moved the figure by more than 10 % (shifts) or changed the conclusion "
                  "(changes).", "", "| Arm | Turns re-checked | Shifts | Changes |", "|---|---:|---:|---:|"]
        for s in summary:
            n = s["reading_checked"]
            if n:
                lo, hi = wilson(s["reading_changes"], n)
                lines.append(f"| {s['arm']} | {n} | {s['reading_shifts']} ({100 * s['reading_shifts'] / n:.0f} %) | "
                             f"{s['reading_changes']} ({100 * s['reading_changes'] / n:.0f} % "
                             f"[{100 * lo:.0f}, {100 * hi:.0f}]) |")
        lines.append("")
    # Per suite (and per family where a suite has them).
    groups = sorted({(r["suite"], r["family"]) for r in rows})
    lines += ["## By suite", "", "| Suite | " + " | ".join(arms) + " |", "|---|" + "---:|" * len(arms)]
    for suite in sorted({g[0] for g in groups}):
        for fam in [""] + sorted({f for s_, f in groups if s_ == suite and f}):
            sel = [r for r in rows if r["suite"] == suite and (not fam or r["family"] == fam) and not r["runner_error"]]
            cells = []
            for arm in arms:
                k = sum(r["n_passed"] for r in sel if r["arm"] == arm)
                n = sum(r["n_checks"] for r in sel if r["arm"] == arm)
                cells.append(f"{k}/{n}" if n else "–")
            lines.append(f"| {suite}{' · ' + fam if fam else ''} | " + " | ".join(cells) + " |")
    lines.append("")
    # Per check, where the arms differ.
    lines += ["## Checks where the arms differ", "", "| Task | Check | " + " | ".join(arms) + " |",
              "|---|---|" + "---:|" * len(arms)]
    table: dict = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for r in rows:
        if r["runner_error"]:
            continue
        for label, ok in r["checks"].items():
            cell = table[(f"{r['suite']}/{r['task']}", label)][r["arm"]]
            cell[0] += int(ok)
            cell[1] += 1
    differ = 0
    for (task, label), cells in sorted(table.items()):
        rates = {a: c[0] / c[1] for a, c in cells.items() if c[1]}
        if len(set(round(v, 6) for v in rates.values())) > 1:
            differ += 1
            lines.append(f"| {task} | {label} | " + " | ".join(
                f"{cells[a][0]}/{cells[a][1]}" if cells[a][1] else "–" for a in arms) + " |")
    if not differ:
        lines.append("| (none) | | " + " | ".join("" for _ in arms) + " |")
    return "\n".join(lines)


def render_latex(rows: list[dict], arms: list[str], baseline: str | None = None) -> str:
    """A booktabs table of the headline numbers, ready for a paper."""
    summary = summarise(rows, arms, baseline)

    def esc(t: str) -> str:
        return t.replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")

    out = [r"\begin{tabular}{lrrrrr}", r"\toprule",
           r"Arm & Checks passed (\%) & $\Delta$ (pp) & Tasks passed (\%) & Tool calls & Tokens (k) \\", r"\midrule"]
    for s in summary:
        d = s.get("vs_baseline")
        delta = "--" if not d or d.get("diff") is None else \
            f"${100 * d['diff']:+.1f}$ $[{100 * d['lo']:+.0f}, {100 * d['hi']:+.0f}]$"
        tokens = (s["input_tokens_per_run"] + s["output_tokens_per_run"]) / 1000
        out.append(f"{esc(s['arm'])} & {_pct(s['check_rate'])} & {delta} & {_pct(s['task_success'])} & "
                   f"{s['tool_calls_per_run']:.1f} & {tokens:.0f} \\\\")
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out)


def write_outputs(folder: pathlib.Path, rows: list[dict], arms: list[str], baseline: str | None = None,
                  manifest: dict | None = None) -> str:
    folder = pathlib.Path(folder)
    md = render_markdown(rows, arms, baseline, manifest)
    (folder / "analysis.md").write_text(md + "\n", encoding="utf-8")
    (folder / "summary.tex").write_text(render_latex(rows, arms, baseline) + "\n", encoding="utf-8")
    summary = summarise(rows, arms, baseline)
    flat = []
    for s in summary:
        row = {k: v for k, v in s.items() if not isinstance(v, (dict, list, tuple))}
        row["check_rate_lo"], row["check_rate_hi"] = s["check_rate_ci"]
        row["task_success_lo"], row["task_success_hi"] = s["task_success_ci"]
        for key in ("vs_baseline", "vs_baseline_success"):
            for k, v in (s.get(key) or {}).items():
                row[f"{key}_{k}"] = v
        flat.append(row)
    _write_csv(folder / "summary.csv", flat)
    _write_csv(folder / "runs.csv", [{k: v for k, v in r.items() if k != "checks"} for r in rows])
    _write_csv(folder / "checks.csv", [{"arm": r["arm"], "suite": r["suite"], "task": r["task"],
                                        "family": r["family"], "repeat": r["repeat"], "check": label,
                                        "passed": int(ok)}
                                       for r in rows if not r["runner_error"] for label, ok in r["checks"].items()])
    (folder / "results.json").write_text(json.dumps({"arms": arms, "baseline": baseline or (arms[0] if arms else None),
                                                     "summary": summary, "runs": rows}, indent=1, default=str),
                                         encoding="utf-8")
    return md


def _write_csv(path: pathlib.Path, rows: list[dict]) -> None:
    keys: list[str] = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
