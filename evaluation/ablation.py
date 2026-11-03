"""
Ablations, end to end: what each part of CONDUCTOR is worth, one command per question a paper asks.

    python -m evaluation.ablation list                                  # studies, suites, knobs, prompt parts
    python -m evaluation.ablation plan components --repeats 5           # what it would run and cost (free)
    python -m evaluation.ablation run components --repeats 5            # run it (spends model credit)
    python -m evaluation.ablation run --arm "full:" --arm "lean: drop_rules=routing hide_tools=group:kpis" \\
                                      --suites options,narration --repeats 3        # a study of your own
    python -m evaluation.ablation run --resume evaluation_runs/ablation-components-20261103-1500
    python -m evaluation.ablation status <folder>                       # how far a run is
    python -m evaluation.ablation analyse <folder> [<folder> …] [--baseline full]    # again, from the records
    python -m evaluation.ablation prompt --drop ask_first,routing       # what an arm takes out of the prompt

A study is a set of arms (settings; `evaluation.arms`) run on a set of suites (`evaluation.suites`), each
task `--repeats` times. The first arm is the baseline. The studies that come ready (`list`): `ontology`,
`components` (leave one out — the paper's main table), `answer_review`, `reading_check`, `prompt_rules`,
`tools`, `deep_research`, `advisor`, `step_budget`, `effort`. `--arm` adds arms to a study, or makes one.

How a run is made, so its numbers can be trusted and repeated:

- each arm runs in its own process with its settings in the environment, against an isolated backend
  (`evaluation.isolated_backend`; one per kind of backend an arm needs), never the operator's; session
  logs, the cache and the saved state go to the run folder;
- `--order interleaved` (the default) runs repeat 1 of every arm, then repeat 2 …, so a model or an API
  that drifts during the run affects every arm alike; `blocked` runs one arm after the other;
- every scenario is loaded afresh before each task, and what a task set up is undone after it;
- each run's record names its arm and a hash of the arm's settings; `--resume` keeps the runs made under the
  same settings and runs the rest (a runner error is run again);
- `manifest.json` holds the date, the commit and whether the tree had uncommitted changes, the arms (with
  what each removes from the prompt and the tools it hides), the suites, the tasks, the repeats, the order,
  the command; the analysis (`evaluation.ablation_stats`) writes `analysis.md`, CSV, LaTeX and JSON.

`run` SPENDS MODEL CREDIT; it shows the plan and asks for confirmation unless --yes, and refuses a plan above
--max-turns. Everything else is free.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import pathlib
import platform
import subprocess
import sys
import time

from . import arms as _arms

_ROOT = pathlib.Path(__file__).resolve().parents[1]
ORDERS = ("interleaved", "blocked")
# How an arm's process ends when it stops early (`_child`), and when it stops.
EXIT_FAILING, EXIT_FATAL = 3, 4
FAILURES_IN_A_ROW = 3


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=_ROOT, capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------

def resolve(preset: str | None, arm_specs: list[str], suites: str, tasks: str) -> dict:
    """The study: its arms (the preset's, then --arm), suites and items."""
    from . import suites as _suites
    if preset and preset not in _arms.PRESETS:
        raise ValueError(f"Unknown study {preset!r}; known: {sorted(_arms.PRESETS)}")
    arms = _arms.preset_arms(preset) if preset else []
    arms += [_arms.parse(s) for s in arm_specs]
    if not arms:
        raise ValueError("No arms: name a study (`list`) or give --arm.")
    names = [a.name for a in arms]
    if len(set(names)) != len(names):
        raise ValueError(f"Two arms share a name: {names}")
    suite_names = [s for s in suites.split(",") if s.strip()] if suites else \
        list(_arms.PRESETS[preset].suites if preset else _suites.names())
    items = _suites.items(suite_names, [t for t in tasks.split(",") if t.strip()] if tasks else None)
    return {"study": preset or "custom", "question": _arms.PRESETS[preset].question if preset else "",
            "arms": arms, "suites": suite_names, "items": items}


def turns_per_repeat(arm, items) -> int:
    follow = int(arm.run_kwargs()["follow_suggestion"])
    return sum(len(i.task.turns) + follow for i in items)


def _configured_model(provider: str) -> str:
    from llm_agent.agent import config
    return {"openai": config.OPENAI_MODEL, "google": config.GEMINI_MODEL, "anthropic": config.ANTHROPIC_MODEL,
            "ollama": config.OLLAMA_MODEL}.get(provider, "")


def _estimate_per_turn(from_folders: list[pathlib.Path] | None = None) -> dict:
    """Tokens and seconds per question: measured on earlier runs' records when given (`--estimate-from`),
    else from CONDUCTOR's usage store for the configured model (input only, approximate). Cached and
    output tokens are None where the records do not report them (records before Nov 2026 do not)."""
    if from_folders:
        turns = [json.loads(x) for f in from_folders for p in pathlib.Path(f).rglob("*.jsonl")
                 if "_replaced" not in p.parts for x in list(open(p, encoding="utf-8"))[1:] if x.strip()]
        turns = [t for t in turns if t.get("status") != "runner_error" and "model_calls" in t]
        if turns:
            n = len(turns)

            def share(key: str) -> float | None:
                # Per turn, over the turns whose record reports it; scaled to every turn's input.
                known = [t for t in turns if t.get(key)]
                if not known:
                    return None
                return sum(sum(t[key]) for t in known) / sum(sum(t.get("prompt_tokens") or []) or 1 for t in known)

            inp = sum(sum(t.get("prompt_tokens") or []) for t in turns) / n
            cached = share("cached_tokens")
            outs = [t for t in turns if t.get("output_tokens")]
            return {"source": f"{n} turns of earlier runs", "input_tokens": inp,
                    "cached_tokens": inp * cached if cached is not None else None,
                    "output_tokens": sum(sum(t["output_tokens"]) for t in outs) / len(outs) if outs else None,
                    "seconds": sum(float((t.get("timing") or {}).get("total_s") or 0) for t in turns) / n}
    from llm_agent.agent import usage
    from llm_agent.agent.config import LLM_PROVIDER
    provider = (os.environ.get("LLM_PROVIDER") or LLM_PROVIDER).lower()
    key = usage._key(provider, _configured_model(provider))
    store = usage.load_store()
    calls = usage.calls_per_question(store)
    tokens = (store.get("prompt_tokens") or {}).get(key, {}).get("mean")
    lat = (store.get("latency") or {}).get(key, {}).get("last_s")
    return {"source": f"this machine's past questions with {key.replace('|', ' / ')}",
            "input_tokens": tokens * calls if tokens else None, "cached_tokens": None, "output_tokens": None,
            "seconds": lat * calls if lat else None}


def render_cost(est: dict, turns: int, price_in: float | None, price_cached: float | None,
                price_out: float | None) -> list[str]:
    """The estimate for `turns` agent turns, and a cost only from the prices given (CONDUCTOR knows none),
    each assumption said."""
    lines = [f"Estimate for {turns} turns, from {est['source']} (approximate):"]
    inp, cached, out = est["input_tokens"], est["cached_tokens"], est["output_tokens"]
    if not inp:
        return lines + ["- no token measurements yet: run a pilot, then plan with --estimate-from <its folder>."]
    if cached is not None:
        lines.append(f"- input: ~{inp * turns / 1e6:.1f} M tokens, of which ~{cached * turns / 1e6:.1f} M "
                     f"({100 * cached / inp:.0f} %) served from the provider's prompt cache")
    else:
        lines.append(f"- input: ~{inp * turns / 1e6:.1f} M tokens; how many are cached is not in these records")
    lines.append(f"- output: ~{out * turns / 1e6:.2f} M tokens" if out else "- output: not in these records")
    if est["seconds"]:
        lines.append(f"- time: ~{est['seconds'] * turns / 3600:.1f} h, one turn at a time")
    if price_in is None:
        lines.append("- cost: give your model's prices (--price-in, --price-cached, --price-out, USD per million "
                     "tokens) from the provider's price list or your bill; CONDUCTOR knows no prices.")
        return lines
    notes = []
    uncached = inp - (cached or 0)
    usd = uncached * turns / 1e6 * price_in
    if cached:
        if price_cached is None:
            usd += cached * turns / 1e6 * price_in
            notes.append("cached input priced as uncached (no --price-cached): an upper bound")
        else:
            usd += cached * turns / 1e6 * price_cached
    elif cached is None:
        notes.append("all input priced as uncached (cache use unknown): an upper bound")
    if out and price_out is not None:
        usd += out * turns / 1e6 * price_out
    else:
        notes.append("output not included")
    lines.append(f"- cost at the prices given: ~${usd:,.2f}" + (f" ({'; '.join(notes)})" if notes else ""))
    return lines


def render_plan(study: dict, repeats: int, order: str, price_in: float | None = None,
                price_out: float | None = None, estimate_from: list[pathlib.Path] | None = None,
                price_cached: float | None = None) -> tuple[str, int]:
    est = _estimate_per_turn(estimate_from)
    lines = [f"Study: {study['study']}" + (f" — {study['question']}" if study["question"] else ""),
             f"Suites: {', '.join(study['suites'])} — {len(study['items'])} tasks; {repeats} repeats; order {order}.",
             "", "| Arm | Settings | Backend | Agent turns |", "|---|---|---|---:|"]
    total = 0
    for arm in study["arms"]:
        n = turns_per_repeat(arm, study["items"]) * repeats
        total += n
        lines.append(f"| {arm.name} | {arm.describe()} | {'without the layer' if arm.backend_mode() == 'none' else 'standard'} "
                     f"| {n} |")
    lines.append(f"| **total** | | | **{total}** |")
    lines += ["", *render_cost(est, total, price_in, price_cached, price_out)]
    return "\n".join(lines), total


def sample_items(items: list, per_suite: int, seed: int) -> list:
    """`per_suite` tasks of each suite, drawn with the seed (a pilot that still covers every suite)."""
    import random
    rng = random.Random(seed)
    out = []
    for suite in dict.fromkeys(i.suite for i in items):
        mine = [i for i in items if i.suite == suite]
        keep = set(i.key for i in rng.sample(mine, min(per_suite, len(mine))))
        out += [i for i in mine if i.key in keep]
    return out


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------

def _wait_for(url: str, proc: subprocess.Popen, seconds: float = 240.0) -> None:
    import httpx
    deadline = time.time() + seconds
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"The isolated backend for {url} exited (code {proc.returncode}); see its log.")
        try:
            if httpx.get(f"{url}/api/grid_constants", timeout=5).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(1)
    raise RuntimeError(f"The isolated backend at {url} did not start.")


def _start_backend(out: pathlib.Path, port: int, mode: str) -> subprocess.Popen:
    """An isolated backend whose studies use the layer (`on`) or not (`none`)."""
    env = {**os.environ, "CONDUCTOR_ONTOLOGY": mode}
    proc = subprocess.Popen([sys.executable, "-m", "evaluation.isolated_backend", "--port", str(port),
                             "--store", str(out / "backends" / f"store-{mode}")], cwd=_ROOT, env=env,
                            stdout=open(out / "backends" / f"backend-{mode}.log", "a"), stderr=subprocess.STDOUT)
    _wait_for(f"http://127.0.0.1:{port}", proc)
    return proc


def _schedule(arms, repeats: int, order: str) -> list[tuple]:
    """(arm, repeats) blocks in the order they run."""
    if order == "blocked":
        return [(a, list(range(1, repeats + 1))) for a in arms]
    return [(a, [k]) for k in range(1, repeats + 1) for a in arms]


def remaining_turns(study: dict, repeats: int, out: pathlib.Path) -> int:
    """Agent turns not yet run under the same settings (all of them in a new folder)."""
    from .ablation_stats import completed
    left = 0
    for arm in study["arms"]:
        follow = int(arm.run_kwargs()["follow_suggestion"])
        for item in study["items"]:
            for k in range(1, repeats + 1):
                if not completed(out / "runs" / arm.name / item.suite / f"{item.task.id}-{k:02d}.jsonl", arm.hash):
                    left += len(item.task.turns) + follow
    return left


def log_change(out: pathlib.Path, what: str, reason: str = "") -> None:
    """Add an entry to the manifest's change log (re-runs, extensions, backend restarts, stops)."""
    path = out / "manifest.json"
    if not path.exists():
        return
    m = json.loads(path.read_text(encoding="utf-8"))
    m.setdefault("changes", []).append({"at": datetime.datetime.now().isoformat(timespec="seconds"),
                                        "what": what, **({"reason": reason} if reason else {})})
    path.write_text(json.dumps(m, indent=1), encoding="utf-8")


def write_manifest(out: pathlib.Path, study: dict, repeats: int, order: str, status: str, started: str,
                   extra: dict | None = None) -> dict:
    previous = _manifest(out)
    manifest = {
        "kind": "ablation", "study": study["study"], "question": study["question"], "status": status,
        "started": started, "updated": datetime.datetime.now().isoformat(timespec="seconds"),
        "commit": _git("rev-parse", "HEAD"), "uncommitted_changes": bool(_git("status", "--porcelain")),
        "arms": [{"name": a.name, "spec": f"{a.name}: {a.describe() if a.settings else ''}".rstrip(),
                  "settings": a.settings, "hash": a.hash, "describe": a.describe(), "backend": a.backend_mode(),
                  **_arms.footprint(a)} for a in study["arms"]],
        "baseline": study["arms"][0].name,
        "suites": study["suites"], "tasks": [i.key for i in study["items"]],
        "repeats": repeats, "order": order, **({"sample": study["sample"]} if study.get("sample") else {}),
        "command": " ".join(["python -m evaluation.ablation", *sys.argv[1:]]),
        "python": platform.python_version(),
        **(extra or {}),
        "changes": previous.get("changes", []),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return manifest


def _healthy(url: str) -> bool:
    import httpx
    try:
        return httpx.get(f"{url}/api/grid_constants", timeout=10).status_code == 200
    except httpx.HTTPError:
        return False


def run_study(study: dict, repeats: int, out: pathlib.Path, *, order: str = "interleaved", port: int = 8791,
              yes: bool = False, max_turns: int | None = None, price_in: float | None = None,
              price_out: float | None = None, price_cached: float | None = None, started: str | None = None,
              plan_study: dict | None = None, plan_repeats: int | None = None) -> int:
    """Run `study` (all of it, or the part of a resumed plan asked for now) into `out`. The manifest always describes
    the whole plan (`plan_study`, `plan_repeats`; by default the study itself)."""
    plan_study, plan_repeats = plan_study or study, plan_repeats or repeats
    plan, total = render_plan(study, repeats, order, price_in, price_out, price_cached=price_cached)
    print(plan)
    left = remaining_turns(study, repeats, out)
    if left < total:
        print(f"\nResuming: {total - left} turns are already done under the same settings; {left} remain.")
    if not left:
        print(analyse([out]))
        return 0
    print("\nEvery agent turn calls the configured model and spends credit.")
    if max_turns is not None and left > max_turns:
        print(f"The plan has {left} turns to run, above --max-turns {max_turns}. Nothing was run.", file=sys.stderr)
        return 2
    if not yes and input("Proceed? [y/N] ").strip().lower() != "y":
        return 1
    for sub in ("runs", "backends", "logs", "cache", "context"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    started = started or datetime.datetime.now().isoformat(timespec="seconds")
    write_manifest(out, plan_study, plan_repeats, order, "running", started)

    from . import progress as _progress
    _progress.start(out, total, total - left)
    modes = sorted({a.backend_mode() for a in study["arms"]})
    ports = {mode: port + k for k, mode in enumerate(modes)}
    backends = {}
    t0 = time.time()
    stopped = None
    try:
        for mode in modes:
            backends[mode] = _start_backend(out, ports[mode], mode)
        for arm, reps in _schedule(study["arms"], repeats, order):
            mode = arm.backend_mode()
            url = f"http://127.0.0.1:{ports[mode]}"
            env = {**os.environ, **arm.env(), "DT_BACKEND_URL": url,
                   "CONDUCTOR_SESSION_LOG_DIR": str(out / "logs"), "CONDUCTOR_CACHE_DIR": str(out / "cache"),
                   "CONDUCTOR_CONTEXT_DIR": str(out / "context")}
            for attempt in (1, 2):
                rc = subprocess.run([sys.executable, "-m", "evaluation.ablation", "_child", "--out", str(out),
                                     "--arm", json.dumps({"name": arm.name, "settings": arm.settings}),
                                     "--items", ",".join(i.key for i in study["items"]),
                                     "--repeats", ",".join(map(str, reps))], cwd=_ROOT, env=env,
                                    check=False).returncode
                if rc == 0:
                    break
                if rc == EXIT_FATAL:
                    stopped = "the model's API refused for good (quota exhausted)"
                    break
                if attempt == 1 and not _healthy(url):
                    # The backend died: start it again and retry this block once (finished runs are skipped).
                    log_change(out, f"backend '{mode}' restarted during arm {arm.name}, repeats {reps}")
                    backends[mode].kill()
                    backends[mode] = _start_backend(out, ports[mode], mode)
                    continue
                stopped = f"runs kept failing in arm {arm.name} (exit {rc})"
                break
            if stopped:
                break
    finally:
        for proc in backends.values():
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
    state = _progress.load(out)
    if state:
        _progress.show(state, "stopped" if stopped else "done", final=True)
    if stopped:
        write_manifest(out, plan_study, plan_repeats, order, "stopped", started, {"seconds": round(time.time() - t0)})
        log_change(out, f"stopped: {stopped}")
        print(f"\nStopped: {stopped}. Nothing finished is lost. When the cause is fixed:\n"
              f"  python -m evaluation.ablation status {out} --detail\n"
              f"  python -m evaluation.ablation run --resume {out}", file=sys.stderr)
        return EXIT_FATAL if "quota" in stopped else EXIT_FAILING
    left = remaining_turns(plan_study, plan_repeats, out)
    write_manifest(out, plan_study, plan_repeats, order, "finished" if not left else "partial", started,
                   {"seconds": round(time.time() - t0)})
    print(analyse([out]))
    print(f"\nRecords, analysis and manifest: {out}" + (f" ({left} turns of the plan still to run)" if left else ""))
    return 0


def _child(out: pathlib.Path, arm_json: str, item_keys: list[str], reps: list[int]) -> int:
    """One arm, some repeats, in this process (its settings are in the environment)."""
    from . import ablation_stats as st
    from . import progress as _progress
    from .semantics import check_backend_mode
    from .suites import SCENARIOS, by_key
    from .whatif import _post_backend, run_task
    raw = json.loads(arm_json)
    arm = _arms.Arm(raw["name"], raw["settings"])
    check_backend_mode("none" if arm.get("ontology") == "none" else "on")
    items = by_key()
    kwargs = arm.run_kwargs()
    follow = kwargs.pop("follow_suggestion")
    failed_in_a_row = 0
    for k in reps:
        for key in item_keys:
            item = items[key]
            path = out / "runs" / arm.name / item.suite / f"{item.task.id}-{k:02d}.jsonl"
            turns = len(item.task.turns) + int(follow)
            if st.completed(path, arm.hash):
                continue                                    # counted as done when the session started
            now = f"{arm.name} · {key} #{k}"
            state = _progress.load(out)
            if state:
                _progress.show(state, now)
            began = time.time()
            started = datetime.datetime.now().isoformat(timespec="seconds")
            try:
                _post_backend("/api/scenario", {"name": SCENARIOS[item.network]})
                if item.prepare:
                    item.prepare()
                records = run_task(item.task, k, follow, **kwargs)
            except Exception as exc:  # noqa: BLE001 — one failed run must not end the study
                records = [{"turn": 0, "status": "runner_error", "runner_error": str(exc)}]
            finally:
                if item.cleanup:
                    try:
                        item.cleanup()
                    except Exception:  # noqa: BLE001
                        pass
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(json.dumps({"kind": "ablation", "arm": arm.name, "arm_hash": arm.hash,
                                     "settings": arm.settings, "suite": item.suite, "task": item.task.id,
                                     "repeat": k, "started": started,
                                     "finished": datetime.datetime.now().isoformat(timespec="seconds")}) + "\n")
                for rec in records:
                    fh.write(json.dumps(rec, default=str) + "\n")
            why = st.failure(records)
            _progress.record(out, turns, time.time() - began, failed=bool(why), current=now)
            # Fail fast: an exhausted quota will not come back; three failed runs in a row mean the backend or the
            # model's API is down — continuing would only write failures. The parent decides what to do.
            if why in st.FATAL_STATUSES:
                print(f"Stopping: {why} ({arm.name} · {key} · repeat {k}).", flush=True)
                return EXIT_FATAL
            failed_in_a_row = failed_in_a_row + 1 if why else 0
            if failed_in_a_row >= FAILURES_IN_A_ROW:
                print(f"Stopping: {failed_in_a_row} runs in a row failed (last: {why}).", flush=True)
                return EXIT_FAILING
    return 0


# ---------------------------------------------------------------------------
# Status and analysis (free)
# ---------------------------------------------------------------------------

def _manifest(folder: pathlib.Path) -> dict:
    path = pathlib.Path(folder) / "manifest.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


_SELECTOR_KEYS = ("arm", "suite", "task", "repeat")


def parse_selector(text: str) -> dict[str, set[str]]:
    """`"arm=full task=names_cigre,names_ieee30 repeat=2"` → which runs: every key optional, values comma-separated,
    a task by id or by suite/task."""
    out: dict[str, set[str]] = {}
    for token in text.split():
        key, eq, value = token.partition("=")
        if not eq or key not in _SELECTOR_KEYS or not value:
            raise ValueError(f"Selector {text!r}: use key=value with keys {_SELECTOR_KEYS}")
        out[key] = {v for v in value.split(",") if v}
    if not out:
        raise ValueError("An empty selector would select every run; name at least one of " + ", ".join(_SELECTOR_KEYS))
    return out


def _plan_runs(m: dict):
    """(arm, suite, task, repeat, path) for every run the manifest plans."""
    folder_runs = pathlib.Path(m["_folder"]) / "runs"
    for a in m.get("arms") or []:
        for key in m.get("tasks") or []:
            suite, task = key.split("/", 1)
            for k in range(1, int(m.get("repeats") or 0) + 1):
                yield a, suite, task, k, folder_runs / a["name"] / suite / f"{task}-{k:02d}.jsonl"


def _selected(sel: dict, arm: str, suite: str, task: str, repeat: int) -> bool:
    return (("arm" not in sel or arm in sel["arm"]) and ("suite" not in sel or suite in sel["suite"])
            and ("task" not in sel or task in sel["task"] or f"{suite}/{task}" in sel["task"])
            and ("repeat" not in sel or str(repeat) in sel["repeat"]))


def replace_runs(folder: pathlib.Path, selectors: list[dict], reason: str) -> int:
    """Move the selected runs' records aside (runs/_replaced/<time>/…, never deleted) so a resume runs them again;
    the manifest's change log says which and why."""
    m = {**_manifest(folder), "_folder": str(folder)}
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    moved = []
    for a, suite, task, k, path in _plan_runs(m):
        if path.exists() and any(_selected(sel, a["name"], suite, task, k) for sel in selectors):
            dest = folder / "runs" / "_replaced" / stamp / a["name"] / suite / path.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            path.rename(dest)
            moved.append(f"{a['name']}/{suite}/{task}#{k}")
    if moved:
        log_change(folder, f"re-run {len(moved)} run(s): {', '.join(moved[:20])}"
                           + (" …" if len(moved) > 20 else "") + f" (old records in runs/_replaced/{stamp}/)", reason)
    return len(moved)


def status(folder: pathlib.Path, detail: bool = False) -> str:
    from . import ablation_stats as st
    m = _manifest(folder)
    if not m:
        return f"{folder}: no manifest.json — not an ablation run folder."
    m["_folder"] = str(folder)
    lines = [f"{folder}: study {m.get('study')}, {m.get('status')}, started {m.get('started')}, "
             f"updated {m.get('updated')}", "",
             "| Arm | Done | Failed | Made under other settings | Not run yet | Planned |", "|---|---:|---:|---:|---:|---:|"]
    counts: dict = {}
    todo = []
    for a, suite, task, k, path in _plan_runs(m):
        c = counts.setdefault(a["name"], {"done": 0, "failed": 0, "stale": 0, "missing": 0, "planned": 0})
        c["planned"] += 1
        if not path.exists():
            c["missing"] += 1
            continue
        meta, records = st.read_run(path)
        if meta.get("arm_hash") != a.get("hash"):
            c["stale"] += 1
            todo.append(f"- {a['name']} · {suite}/{task} · repeat {k}: made under other settings")
        elif st.failure(records):
            c["failed"] += 1
            todo.append(f"- {a['name']} · {suite}/{task} · repeat {k}: failed ({st.failure(records)})")
        else:
            c["done"] += 1
    for name, c in counts.items():
        lines.append(f"| {name} | {c['done']} | {c['failed']} | {c['stale']} | {c['missing']} | {c['planned']} |")
    from . import progress as _progress
    state = _progress.load(folder)
    if state and m.get("status") == "running":
        lines += ["", "Now: " + _progress.line(state)]
    if detail and todo:
        lines += ["", "Runs a resume will run again:", *todo]
    if m.get("changes"):
        lines += ["", "Changes:"] + [f"- {ch['at']}: {ch['what']}" + (f" — {ch['reason']}" if ch.get("reason") else "")
                                     for ch in m["changes"]]
    return "\n".join(lines)


def analyse(folders: list[pathlib.Path], baseline: str | None = None, arm_names: list[str] | None = None,
            out: pathlib.Path | None = None) -> str:
    """Grade every run in the folders and write the tables into the first (or `out`)."""
    from . import ablation_stats as st
    manifest = _manifest(folders[0])
    rows = st.grade_runs(folders)
    seen = list(dict.fromkeys(r["arm"] for r in rows))
    order = [a["name"] for m in map(_manifest, folders) for a in m.get("arms") or []]
    arms = arm_names or list(dict.fromkeys([a for a in order if a in seen] + seen))
    baseline = baseline or manifest.get("baseline") or (arms[0] if arms else None)
    if baseline in arms:
        arms = [baseline] + [a for a in arms if a != baseline]
    md = st.write_outputs(out or folders[0], rows, arms, baseline, manifest)
    if manifest.get("study") == "ontology":
        extra = _ontology_sections(pathlib.Path(folders[0]))
        if extra:
            md = f"{md}\n\n{extra}"
            (pathlib.Path(out or folders[0]) / "analysis.md").write_text(md + "\n", encoding="utf-8")
    return md


# The ontology study's own tables (per network and per reasoning family), next to the generic ones.
_ONTOLOGY_COMPARISONS = (("on", "off", "What the agent gains: the layer for the agent (both arms' studies use it)"),
                         ("off", "none", "What grounding the studies gains (neither arm's agent sees the layer)"),
                         ("on", "none", "The whole contribution"))
_ONTOLOGY_LABEL = {"on": "Layer on", "off": "Agent without it", "none": "Layer off everywhere"}


def _ontology_sections(out: pathlib.Path) -> str:
    from . import reasoning, semantics
    runs = out / "runs"
    parts = []
    for a, b, title in _ONTOLOGY_COMPARISONS:
        labels = (_ONTOLOGY_LABEL[a], _ONTOLOGY_LABEL[b])
        section = []
        for net in ("cigre", "ieee30"):
            da, db = runs / a / f"semantics-{net}", runs / b / f"semantics-{net}"
            if da.is_dir() and db.is_dir():
                section += [f"### Looking up what elements are — {net}", "", semantics.compare([(da, db)], labels), ""]
        if (runs / a / "reasoning").is_dir() and (runs / b / "reasoning").is_dir():
            section += ["### Reasoning", "", reasoning.analyse(runs / a / "reasoning", runs / b / "reasoning", labels), ""]
        if section:
            parts += [f"## {title} — {a} vs {b}", "", *section]
    return "\n".join(parts)


def prompt_parts(drop: list[str], show: bool) -> str:
    from llm_agent.agent.config import DEFAULT_GRID_CONSTANTS
    from llm_agent.agent.system_prompt import PROMPT_PARTS, _part_spans, build_system_prompt, drop_parts
    prompt = build_system_prompt(DEFAULT_GRID_CONSTANTS)
    if not drop:
        rows = ["| Part | Kind | Characters |", "|---|---|---:|"]
        for part, (kind, _) in PROMPT_PARTS.items():
            rows.append(f"| {part} | {kind} | {sum(b - a for a, b in _part_spans(prompt, part))} |")
        return f"The system prompt: {len(prompt)} characters (IEEE 30 fallback constants).\n\n" + "\n".join(rows)
    lean, removed = drop_parts(prompt, drop)
    lines = [f"{len(prompt)} → {len(lean)} characters; removed: "
             + ", ".join(f"{k} ({v})" for k, v in removed.items())]
    if show:
        for part in drop:
            for a, b in _part_spans(prompt, part):
                lines += ["", f"--- {part} ---", prompt[a:b].strip()]
    return "\n".join(lines)


def render_list() -> str:
    from llm_agent.agent.tool_registry import ABLATION_GROUPS
    from . import suites as _suites
    lines = ["Studies (`run <study>`; the first arm is the baseline):", ""]
    for p in _arms.PRESETS.values():
        lines.append(f"- {p.name}: {p.question}")
        lines.append(f"    arms: {' | '.join(p.arms)}")
        lines.append(f"    suites: {', '.join(p.suites)}")
    lines += ["", "Suites:", ""]
    for name in _suites.names():
        its = _suites.items([name])
        lines.append(f"- {name}: {len(its)} tasks, {sum(len(i.task.turns) for i in its)} turns per repeat")
    lines += ["", "Knobs (`--arm \"name: knob=value …\"`): " + ", ".join(_arms.KNOBS) + ", env.<NAME>",
              "Prompt parts (drop_rules=…): `python -m evaluation.ablation prompt`",
              "Tool groups (hide_tools=group:…): " + ", ".join(f"{k} ({len(v)})" for k, v in ABLATION_GROUPS.items())]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def _add_study_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("study", nargs="?", default=None, help="a study from `list` (optional with --arm)")
    p.add_argument("--arm", action="append", default=[], help='an arm: "name: knob=value …" (repeatable)')
    p.add_argument("--suites", default="", help="comma-separated suites (default: the study's)")
    p.add_argument("--tasks", default="", help="comma-separated task ids or suite/task keys (default: all)")
    p.add_argument("--repeats", type=int, default=None, help="repeats per task (default 3; on resume: extend to)")
    p.add_argument("--order", choices=ORDERS, default="interleaved")
    p.add_argument("--per-suite", type=int, default=0, help="a pilot: this many tasks of each suite, drawn with --seed")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--price-in", type=float, default=None, help="USD per million input tokens, for the estimate")
    p.add_argument("--price-cached", type=float, default=None,
                   help="USD per million cached input tokens, for the estimate")
    p.add_argument("--price-out", type=float, default=None, help="USD per million output tokens, for the estimate")
    p.add_argument("--estimate-from", action="append", default=[],
                   help="estimate the cost from these earlier run folders' records (repeatable)")


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    # Kept from before the general runner: `ontology [--repeats N] [--analyse <folder>] …`.
    if argv[:1] == ["ontology"]:
        legacy = argparse.ArgumentParser(prog="python -m evaluation.ablation ontology")
        legacy.add_argument("--analyse", default="")
        legacy.add_argument("--arms", default="")
        args, rest = legacy.parse_known_args(argv[1:])
        if args.analyse:
            print(analyse([pathlib.Path(args.analyse)]))
            return 0
        arms = [a for a in args.arms.split(",") if a]
        if arms:
            rest += ["--only", ",".join(arms)]
        argv = ["run", "ontology", *rest]

    parser = argparse.ArgumentParser(prog="python -m evaluation.ablation", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="studies, suites, knobs, tool groups")
    p_plan = sub.add_parser("plan", help="what a study would run, and roughly what it costs (free)")
    _add_study_args(p_plan)
    p_run = sub.add_parser("run", help="run a study (spends model credit)")
    _add_study_args(p_run)
    p_run.add_argument("--only", default="", help="run only these arms of the study (comma-separated)")
    p_run.add_argument("--out", default="")
    p_run.add_argument("--resume", default="", help="continue the run in this folder, with its own settings; "
                       "--only/--suites/--tasks then run part of its plan now, --repeats extends it")
    p_run.add_argument("--rerun", action="append", default=[],
                       help='with --resume: run these again, e.g. "arm=full task=names_cigre repeat=2" (repeatable)')
    p_run.add_argument("--reason", default="", help="why the --rerun (kept in the manifest's change log)")
    p_run.add_argument("--port", type=int, default=8791, help="the first isolated backend's port")
    p_run.add_argument("--max-turns", type=int, default=None, help="refuse a plan with more agent turns")
    p_run.add_argument("--yes", action="store_true")
    p_status = sub.add_parser("status", help="how far a run is (free)")
    p_status.add_argument("folder")
    p_status.add_argument("--detail", action="store_true", help="list each run a resume would run again")
    p_an = sub.add_parser("analyse", help="grade and tabulate run folders (free)")
    p_an.add_argument("folders", nargs="+")
    p_an.add_argument("--baseline", default=None)
    p_an.add_argument("--arms", default="", help="the arms to show, in this order")
    p_an.add_argument("--out", default="", help="where to write the tables (default: the first folder)")
    p_pr = sub.add_parser("prompt", help="the prompt's parts, or what dropping some removes (free)")
    p_pr.add_argument("--drop", default="")
    p_pr.add_argument("--show", action="store_true", help="print the removed text")
    p_child = sub.add_parser("_child")             # internal: one arm in its own process
    p_child.add_argument("--out", required=True)
    p_child.add_argument("--arm", required=True)
    p_child.add_argument("--items", required=True)
    p_child.add_argument("--repeats", required=True)
    args = parser.parse_args(argv)

    if args.command == "list":
        print(render_list())
        return 0
    if args.command == "prompt":
        print(prompt_parts([d for d in args.drop.split(",") if d], args.show))
        return 0
    if args.command == "status":
        print(status(pathlib.Path(args.folder), args.detail))
        return 0
    if args.command == "analyse":
        print(analyse([pathlib.Path(f) for f in args.folders], args.baseline,
                      [a for a in args.arms.split(",") if a] or None, pathlib.Path(args.out) if args.out else None))
        return 0
    if args.command == "_child":
        return _child(pathlib.Path(args.out), args.arm, args.items.split(","),
                      [int(k) for k in args.repeats.split(",")])

    try:
        if args.command == "run" and args.resume:
            folder = pathlib.Path(args.resume).resolve()
            m = _manifest(folder)
            if not m:
                print(f"{folder} has no manifest.json to resume from.", file=sys.stderr)
                return 2
            plan = resolve(None, [a["spec"] for a in m["arms"]], ",".join(m["suites"]), "")
            plan["items"] = [i for i in plan["items"] if i.key in set(m["tasks"])]
            plan["study"], plan["question"] = m.get("study", "custom"), m.get("question", "")
            if m.get("sample"):
                plan["sample"] = m["sample"]
            if [a.hash for a in plan["arms"]] != [a["hash"] for a in m["arms"]]:
                print("The arms' settings no longer match the manifest (a knob changed meaning?). "
                      "Start a new run.", file=sys.stderr)
                return 2
            repeats = int(m["repeats"])
            if args.repeats and args.repeats > repeats:
                log_change(folder, f"repeats extended from {repeats} to {args.repeats}", args.reason)
                repeats = args.repeats
            # Part of the plan now: some arms, suites or tasks (the manifest keeps the whole plan).
            study = dict(plan)
            if args.only:
                keep = {a for a in args.only.split(",") if a}
                study["arms"] = [a for a in plan["arms"] if a.name in keep]
                if keep - {a.name for a in plan["arms"]}:
                    raise ValueError(f"--only: not arms of this run: {sorted(keep - {a.name for a in plan['arms']})}")
            if args.suites:
                keep = {x for x in args.suites.split(",") if x}
                study["items"] = [i for i in study["items"] if i.suite in keep]
            if args.tasks:
                keep = {x for x in args.tasks.split(",") if x}
                study["items"] = [i for i in study["items"] if i.task.id in keep or i.key in keep]
            if not study["arms"] or not study["items"]:
                raise ValueError("Nothing of this run's plan matches --only/--suites/--tasks.")
            if args.rerun:
                if not args.reason:
                    raise ValueError("--rerun needs --reason: the manifest keeps why each run was run again.")
                n = replace_runs(folder, [parse_selector(x) for x in args.rerun], args.reason)
                print(f"{n} run(s) set aside to be run again.")
            return run_study(study, repeats, folder, order=m.get("order", "interleaved"), port=args.port,
                             yes=args.yes, max_turns=args.max_turns, price_in=args.price_in,
                             price_out=args.price_out, price_cached=args.price_cached, started=m.get("started"),
                             plan_study=plan, plan_repeats=repeats)
        if getattr(args, "rerun", None):
            raise ValueError("--rerun works on an existing run: use it with --resume <folder>.")
        args.repeats = args.repeats or 3
        study = resolve(args.study, args.arm, args.suites, args.tasks)
        if args.per_suite:
            study["items"] = sample_items(study["items"], args.per_suite, args.seed)
            study["sample"] = {"per_suite": args.per_suite, "seed": args.seed}
        if getattr(args, "only", ""):
            keep = [a for a in args.only.split(",") if a]
            study["arms"] = [a for a in study["arms"] if a.name in keep]
            if not study["arms"]:
                raise ValueError(f"--only {keep}: no such arm in the study")
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2
    if args.command == "plan":
        print(render_plan(study, args.repeats, args.order, args.price_in, args.price_out,
                          [pathlib.Path(f) for f in args.estimate_from], args.price_cached)[0])
        return 0
    out = pathlib.Path(args.out or f"evaluation_runs/ablation-{study['study']}-{datetime.datetime.now():%Y%m%d-%H%M}")
    out = out.resolve()
    if (out / "manifest.json").exists():
        print(f"{out} already holds a run: use --resume {out}, or another --out.", file=sys.stderr)
        return 2
    return run_study(study, args.repeats, out, order=args.order, port=args.port, yes=args.yes,
                     max_turns=args.max_turns, price_in=args.price_in, price_out=args.price_out,
                     price_cached=args.price_cached)


if __name__ == "__main__":
    sys.exit(main())
