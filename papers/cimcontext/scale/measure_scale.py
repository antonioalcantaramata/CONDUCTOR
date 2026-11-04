"""
How the semantic layer and the answer checks scale with the network (GM 2027 paper, Section IV-E). No model is
called: the layer is built as the backend builds it, one outage is looked up in it, and one answer naming the
network's elements is checked as the live loop checks it (facts kept for the named elements only, every name
searched for elements the network does not have). Median of five timings each, on this machine.

    python papers/cimcontext/scale/measure_scale.py [--out papers/cimcontext/scale/scale.json]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import platform
import statistics
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT))

import pandapower.networks as pn  # noqa: E402

import semantics as sm  # noqa: E402
from llm_agent.agent import entities as en  # noqa: E402
from llm_agent.agent.provenance import check_turn  # noqa: E402

CASES = {"CIGRE MV": lambda: pn.create_cigre_network_mv(with_der="all"), "IEEE 30": pn.case30,
         "PEGASE 2869": pn.case2869pegase, "PEGASE 9241": pn.case9241pegase}
REPEATS = 5


def _timed(fn, repeats: int = REPEATS):
    times, out = [], None
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        times.append(time.perf_counter() - t0)
    return out, statistics.median(times)


def _answer(facts: dict) -> str:
    buses = list(facts["buses"])[:3]
    branch = next(iter(facts["branches"])).split(" | ")[0]
    unit = next(iter(facts["units"]), None)
    kv = facts["voltage_levels_kv"][0] if facts["voltage_levels_kv"] else 20
    text = (f"At 18:00, {buses[0]} is at 0.942 p.u. and {buses[1]} at 0.951 p.u. The line {branch} carries most of the "
            f"import. {buses[2]} is at {kv:g} kV.")
    return text + (f" {unit} is a wind turbine." if unit else "")


def measure(name: str, make) -> dict:
    net = make()
    sem, t_build = _timed(lambda: sm.build(net))
    facts, t_compact = _timed(lambda: sm.compact(sem))
    _, t_around = _timed(lambda: sm.around(net, sem, "line", int(net.line.index[0])))
    answer = _answer(facts)
    names = [*facts["buses"], *facts["loads"], *facts["units"], *facts["branches"],
             *(k.split(" | ")[0] for k in facts["branches"])]

    def check():
        named = lambda k: k.split(" | ")[0] in answer or k in answer  # noqa: E731
        kept = {"per_unit_case": facts["per_unit_case"], "voltage_levels_kv": facts["voltage_levels_kv"],
                "external_grids": facts["external_grids"], "units_by_source": {},
                **{part: {k: v for k, v in facts[part].items() if named(k)}
                   for part in ("buses", "branches", "units", "loads")},
                "names_not_in_network": en.names_not_in_network(answer, names)}
        rec = {"user": "", "assistant": answer, "network_facts": {"base": kept}, "tool_calls": [], "tool_results": []}
        return en.check_answer(answer, rec), check_turn(rec)
    _, t_check = _timed(check)
    return {"network": name, "buses": len(net.bus), "lines": len(net.line), "transformers": len(net.trafo),
            "units": len(net.gen) + len(net.sgen), "loads": len(net.load),
            "build_s": round(t_build, 3), "compact_s": round(t_compact, 3), "outage_lookup_s": round(t_around, 3),
            "check_answer_s": round(t_check, 4)}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=str(ROOT / "papers" / "cimcontext" / "scale" / "scale.json"))
    args = p.parse_args(argv)
    rows = []
    for name, make in CASES.items():
        rows.append(measure(name, make))
        print(json.dumps(rows[-1]), flush=True)
    pathlib.Path(args.out).write_text(json.dumps({"machine": platform.platform(), "processor": platform.processor(),
                                                  "python": platform.python_version(), "repeats": REPEATS,
                                                  "rows": rows}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
