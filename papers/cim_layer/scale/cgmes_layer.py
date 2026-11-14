"""
The CIM layer on a CGMES model (CIM layer paper): what the layer takes as stated from the model itself, what it has to
infer, and what stays unknown, and how long it takes. The model is read with CONDUCTOR's own importer
(backend/network_loader.read_cgmes), exactly as an operator's upload is. No AI model is called.

    python papers/cim_layer/scale/cgmes_layer.py MODEL.zip [MODEL.zip ...] [--out papers/cim_layer/scale/cgmes_layer.json]

MODEL.zip is a zip of the CGMES profiles, or a zip of zips (model and boundary set, as ENTSO-E publishes them).

Models measured for the paper (4 Nov 2026): pandapower's CGMES example (tests/fixtures/cgmes), and ENTSO-E's
SmallGrid conformity test configuration, bus-branch and node-breaker (CGMES 2.4.15, v3.0.0), as distributed in
powsybl-core: cgmes/cgmes-conformity/src/main/resources/conformity/cas-1.1.3-data-4.0.3/SmallGrid/ — each the
BaseCase_Complete profiles zipped, with the Boundary profiles zipped beside them in one outer zip.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import platform
import statistics
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT))

import pandapower as pp  # noqa: E402

import network_loader as nl  # noqa: E402
import semantics as sm  # noqa: E402

REPEATS = 5


def _timed(fn):
    times, out = [], None
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        out = fn()
        times.append(time.perf_counter() - t0)
    return out, statistics.median(times)


def _basis(fact) -> str:
    return fact.get("basis", "?") if isinstance(fact, dict) else "?"


def measure(path: str) -> dict:
    t0 = time.perf_counter()
    net = nl.read_cgmes(path)
    t_read = time.perf_counter() - t0
    try:
        pp.runpp(net)
        converged = bool(net.converged)
    except Exception:  # noqa: BLE001 — reported, not fatal
        converged = False
    sem, t_build = _timed(lambda: sm.build(net, network_id=pathlib.Path(path).stem))
    out = {"model": pathlib.Path(path).name, "read_s": round(t_read, 2), "build_s": round(t_build, 3),
           "power_flow_converged": converged,
           "elements": {"buses": len(net.bus), "lines": len(net.line), "transformers": len(net.trafo),
                        "generators": len(net.gen), "static_generators": len(net.sgen), "loads": len(net.load),
                        "switches": len(net.switch), "external_grids": len(net.ext_grid)},
           "mrid_basis": sem.get("mrid_basis"),
           "substations": {"basis": _basis(sem.get("substations") or {}),
                           "count": len((sem.get("containers") or {}).get("substations") or [])
                           if isinstance(sem.get("containers"), dict) else None},
           "import_notes": list(net.get("_import_notes") or [])}
    tally = collections.defaultdict(collections.Counter)
    for b in sem["buses"]:
        tally["bus voltage"][_basis(b.get("kv"))] += 1
        tally["bus substation"][_basis(b.get("substation"))] += 1
        feeder = b.get("feeder") or {}
        # Only buses that get a feeder: in a meshed (transmission) network most buses have none.
        if feeder.get("value"):
            tally["bus feeder (where one applies)"][_basis(feeder)] += 1
    for u in sem["units"]:
        src = u.get("source") or {}
        tally["unit source"][_basis(src)] += 1
        tally["unit source value"][f"{src.get('value')} ({_basis(src)})"] += 1
        tally["unit CIM class"][f"{u.get('cim_class')}/{u.get('cim_unit')}"] += 1
    for ld in sem.get("loads") or []:
        tally["load category"][_basis(ld.get("category"))] += 1
    for s in sem.get("switches") or []:
        tally["switch class"][_basis(s.get("cim_class"))] += 1
        tally["switch normal state"][_basis(s.get("normal_open"))] += 1
        if (s.get("normal_open") or {}).get("value"):
            tally["switches normally open"]["count"] += 1
    for br in sem.get("branches") or []:
        tally["branch kind"][_basis(br.get("kind")) if isinstance(br.get("kind"), dict) else "stated"] += 1
    out["facts"] = {k: dict(v) for k, v in tally.items()}
    if len(net.line):
        _, t_around = _timed(lambda: sm.around(net, sem, "line", int(net.line.index[0])))
        out["outage_lookup_s"] = round(t_around, 4)
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("models", nargs="+")
    p.add_argument("--out", default=str(ROOT / "papers" / "cim_layer" / "scale" / "cgmes_layer.json"))
    args = p.parse_args(argv)
    rows = [measure(m) for m in args.models]
    for r in rows:
        print(json.dumps(r, indent=1))
    pathlib.Path(args.out).write_text(json.dumps({"machine": platform.platform(), "repeats": REPEATS, "rows": rows},
                                                 indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
