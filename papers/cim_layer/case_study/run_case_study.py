"""
The case study of the CIM layer paper (Section IV): one session with the full system on the CIGRE MV demo.

The questions are put in order in one conversation, as an operator would, to the same configuration as the full
system of run B (default settings), on an isolated backend so nothing touches the operator's `systems/`,
`data_files/` or caches. Each turn's record is kept, and a transcript lists every answer with what the checks found
(figures traced, entity findings, the reading check), for the authors to read and quote. Spends model credit.

    python papers/cim_layer/case_study/run_case_study.py [--out papers/cim_layer/runs/D-case-study] [--port 8795]
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
DEFAULT_OUT = ROOT / "papers" / "cim_layer" / "runs" / "D-case-study"
SCENARIO = "cigre_mv"

# The session, in order. The operator's own messages only; the agent's questions back are answered by the next one.
TURNS = (
    "Is the grid secure at the evening peak on 3 January?",
    "If Line 2-3 trips, which loads lose supply, and which switch could restore them?",
    "If wind and solar output drop by 50 % at noon on 3 January, does the grid stay secure, and which units does "
    "that affect?",
    "Fuel cell 1 has been replaced by a small wind turbine. Record that, and check the same 50 % drop again.",
    "Give me options that stay secure if a line trips.",
)
# Sent only if the agent asks back after the options request (the coordination the paper describes).
ANSWER_TO_QUESTION_BACK = ("At the evening peak on 3 January, keep every bus between 0.95 and 1.05 p.u. after any "
                           "single line outage.")


def _start_backend(out: pathlib.Path, port: int):
    sys.path.insert(0, str(ROOT))
    from evaluation.ablation import _start_backend as start
    (out / "backends").mkdir(parents=True, exist_ok=True)
    return start(out, port, "on")


def _audit(record: dict) -> dict:
    from evaluation.report import grade_record
    g = grade_record(record)
    return {
        "figures": g.n_claims, "grounded": g.n_grounded, "derived": g.n_derived, "imprecise": g.n_imprecise,
        "misattributed": g.n_misattributed, "ungrounded": g.n_ungrounded, "mischaracterised": g.n_mischaracterised,
        "evidence": g.evidence,
        "findings": [{"code": f.code, "detail": f.detail} for f in g.findings],
        "reading_check": g.reading_check,
        "tools": [c.get("name") for c in record.get("tool_calls") or []],
    }


def _asks_back(text: str, record: dict) -> bool:
    from llm_agent.agent import provenance
    return provenance.is_clarifying_question(text, record.get("tool_calls") or ())


def run(out: pathlib.Path, port: int) -> int:
    out.mkdir(parents=True, exist_ok=True)
    os.environ.update({"DT_BACKEND_URL": f"http://127.0.0.1:{port}",
                       "CONDUCTOR_SESSION_LOG_DIR": str(out / "logs"), "CONDUCTOR_CACHE_DIR": str(out / "cache"),
                       "CONDUCTOR_CONTEXT_DIR": str(out / "context")})
    proc = _start_backend(out, port)
    try:
        from evaluation.whatif import _post_backend
        from llm_agent.agent import loop
        _post_backend("/api/scenario", {"name": SCENARIO})
        records: list[dict] = []
        original = loop._append_to_log
        loop._append_to_log = records.append
        history: list = []
        conversation = "case-study-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        try:
            queue = list(TURNS)
            answered_back = False
            while queue:
                question = queue.pop(0)
                print(f"\n>>> {question}", flush=True)
                text, history = loop.run_agent_turn(question, history, conversation_id=conversation)
                print(text, flush=True)
                if (not queue and not answered_back and records and question == TURNS[-1]
                        and _asks_back(text, records[-1])):
                    queue.append(ANSWER_TO_QUESTION_BACK)
                    answered_back = True
        finally:
            loop._append_to_log = original
    finally:
        proc.terminate()

    started = datetime.datetime.now().isoformat(timespec="seconds")
    with open(out / "session.jsonl", "w", encoding="utf-8") as fh:
        fh.write(json.dumps({"kind": "case_study", "scenario": SCENARIO, "conversation": conversation,
                             "finished": started}) + "\n")
        for rec in records:
            fh.write(json.dumps(rec, default=str) + "\n")
    lines = [f"# Case study session ({SCENARIO}), {started}", ""]
    audits = []
    for i, rec in enumerate(records, 1):
        a = _audit(rec)
        audits.append(a)
        lines += [f"## Turn {i}", "", f"**Operator:** {rec.get('user', '')}", "", "**CONDUCTOR:**", "",
                  rec.get("assistant", ""), "",
                  f"*Audit:* {a['figures']} figures ({a['grounded']} grounded, {a['derived']} derived, "
                  f"{a['imprecise']} imprecise, {a['misattributed']} misattributed, {a['ungrounded']} ungrounded); "
                  f"{a['mischaracterised']} mischaracterised; evidence {a['evidence']}; tools {a['tools']}", ""]
        if a["reading_check"]:
            lines += [f"*Reading check:* {a['reading_check']}", ""]
        for f in a["findings"]:
            lines.append(f"- `{f['code']}` {f['detail']}")
        lines.append("")
    (out / "transcript.md").write_text("\n".join(lines), encoding="utf-8")
    (out / "audit.json").write_text(json.dumps(audits, indent=1), encoding="utf-8")
    print(f"\nwrote {out / 'transcript.md'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--port", type=int, default=8795)
    args = parser.parse_args(argv)
    sys.path.insert(0, str(ROOT))
    os.chdir(ROOT)
    return run(pathlib.Path(args.out).resolve(), args.port)


if __name__ == "__main__":
    sys.exit(main())
