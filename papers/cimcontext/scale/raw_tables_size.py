"""
How large the raw-tables baseline grows with the network (GM 2027 paper, Section V-B). No model is called: the
network's standard pandapower tables, as `read_network_tables` returns them but with every row, are written out as
JSON and measured; tokens are estimated as characters / 4.

    python papers/cimcontext/scale/raw_tables_size.py [--out papers/cimcontext/scale/raw_tables_size.json]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "backend"))

import pandapower.networks as pn  # noqa: E402

import raw_tables as rt  # noqa: E402

CASES = {"CIGRE MV": lambda: pn.create_cigre_network_mv(with_der="all"), "IEEE 30": pn.case30,
         "IEEE 118": pn.case118, "MV Oberrhein": pn.mv_oberrhein, "IEEE 300": pn.case300,
         "PEGASE 1354": pn.case1354pegase, "PEGASE 2869": pn.case2869pegase, "RTE 6470": pn.case6470rte,
         "PEGASE 9241": pn.case9241pegase}
CHARS_PER_TOKEN = 4


def measure(name: str, make) -> dict:
    net = make()
    chars = len(json.dumps(rt.raw_tables(net)))
    return {"network": name, "buses": len(net.bus), "switches": len(net.switch), "characters": chars,
            "tokens_estimate": round(chars / CHARS_PER_TOKEN), "tokens_per_bus": round(chars / CHARS_PER_TOKEN / len(net.bus))}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=str(ROOT / "papers" / "cimcontext" / "scale" / "raw_tables_size.json"))
    args = p.parse_args(argv)
    rt.MAX_ROWS = 10**9                      # every row: the size the whole tables would take
    rows = []
    for name, make in CASES.items():
        rows.append(measure(name, make))
        print(json.dumps(rows[-1]), flush=True)
    pathlib.Path(args.out).write_text(json.dumps({"chars_per_token": CHARS_PER_TOKEN, "rows": rows}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
