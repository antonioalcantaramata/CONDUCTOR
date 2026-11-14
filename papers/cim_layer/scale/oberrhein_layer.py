"""
The layer on a realistic distribution network (CIM layer paper, Section V-D): pandapower's MV Oberrhein, 179 buses and
322 switches. No model is called: the layer is built as the backend builds it, and what it infers is counted —
the feeders, the role of every normally open switch, and, for each feeder's first line, what its outage cuts off
and which normally open ties could restore it. Median of five build timings, on this machine.

    python papers/cim_layer/scale/oberrhein_layer.py [--out papers/cim_layer/scale/oberrhein_layer.json]
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import platform
import re
import statistics
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "backend"))

import pandapower.networks as pn  # noqa: E402

import semantics as sm  # noqa: E402

REPEATS = 5


def _role(text: str) -> str:
    for prefix in ("tie between feeders", "normally open point closing a loop", "sectionalising switch",
                   "normally open line switch", "line switch", "transformer switch", "bus coupler"):
        if text.startswith(prefix):
            return prefix
    return text


def measure() -> dict:
    net = pn.mv_oberrhein()
    times = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        layer = sm.build(net)
        times.append(time.perf_counter() - t0)
    feeders = sorted({b["feeder"]["value"] for b in layer["buses"]
                      if isinstance(b.get("feeder"), dict) and b["feeder"].get("value")})
    switches = layer["switches"]
    outages = []
    for name in feeders:
        m = re.search(r"\[L(\d+)\]", name)
        if not m:
            continue
        cut = sm.around(net, layer, "line", int(m.group(1)))["without_it"]
        outages.append({"feeder_first_line": name, "buses_cut_off": len(cut["buses_cut_off"]),
                        "load_cut_off_mw": cut["load_cut_off_mw"],
                        "open_ties_that_could_restore": len(cut["open_ties_that_could_restore"])})
    return {
        "network": "MV Oberrhein (pandapower)", "buses": len(net.bus), "lines": len(net.line),
        "transformers": len(net.trafo), "switches": len(net.switch), "units": len(net.sgen) + len(net.gen),
        "loads": len(net.load), "voltage_levels_kv": layer["voltage_levels_kv"],
        "build_s_median": round(statistics.median(times), 3),
        "feeders": len(feeders),
        "normally_open_switches": sum(bool(s["normal_open"]["value"]) for s in switches),
        "switch_roles": dict(collections.Counter(_role(s["role"]["value"]) for s in switches)),
        "unit_sources": {f"{k[0]} ({k[1]})": v for k, v in
                         collections.Counter((u["source"]["value"], u["source"]["basis"]) for u in layer["units"]).items()},
        "load_categories": {f"{k[0]} ({k[1]})": v for k, v in
                            collections.Counter((d["category"]["value"], d["category"]["basis"])
                                                for d in layer["loads"]).items()},
        "feeder_outages": outages,
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=str(ROOT / "papers" / "cim_layer" / "scale" / "oberrhein_layer.json"))
    args = p.parse_args(argv)
    result = measure()
    result.update({"machine": platform.platform(), "python": platform.python_version(), "repeats": REPEATS})
    pathlib.Path(args.out).write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
