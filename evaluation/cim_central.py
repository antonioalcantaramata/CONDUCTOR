"""
The CIM model as the network's state — checked end to end.

    python -m evaluation.cim_central run                     # the standard networks
    python -m evaluation.cim_central run --large             # also PEGASE 1354 and 9241 (timing)
    python -m evaluation.cim_central run --networks case14,cigre_mv --out evaluation_runs/cim-x

CONDUCTOR keeps a network as its CIM model plus differences (cim_changes.py): a proposal is a difference, the
operator's standing changes are a difference, and the pandapower network a study runs on is projected from them. Each
part is checked here against the way CONDUCTOR built the same network before, element by element through the mRIDs —
every column of every table and the costs (`cim_changes.equivalent`), and a power flow (`cim_changes.compare`):

- proposals: the edits applied to a copy of the base (sandbox.materialise) / the base with the proposal's CIM
  difference applied (cim_changes.translate + project_onto), for every kind of edit;
- operator changes: generator models and names applied to the network as loaded (generators.apply, naming.apply) /
  the operator's CIM difference projected onto it;
- moments: the clock's operating point allocated on each network / the base's moment applied by mRID
  (operating_point.py);
- saved state: the differences written as IEC 61970-552 and read back give the same networks.

Writes results.json, results.csv and manifest.json (date, commit, uncommitted changes, command, networks).
"""

from __future__ import annotations

import argparse
import copy
import csv
import datetime
import json
import pathlib
import platform
import subprocess
import sys
import time

_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "backend"))


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=_ROOT, capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


# -- the networks ---------------------------------------------------------------------------------------------------------

def _networks() -> dict:
    import pandapower.networks as pn
    return {
        "case14": pn.case14,                                            # transmission, no switches
        "case30": pn.case30,
        "cigre_mv": lambda: pn.create_cigre_network_mv(with_der="all"),  # distribution, switches at line ends, DER
        "multivoltage": pn.example_multivoltage,                        # 3-winding transformers, wards, impedances
        "oberrhein": pn.mv_oberrhein,
        "multivoltage_cgmes": None,                                     # the same, loaded from CGMES (its model)
    }


_FIXTURE = _ROOT / "tests" / "fixtures" / "cgmes" / "pandapower_example_multivoltage.zip"


LARGE = {"case1354pegase": "case1354pegase", "case9241pegase": "case9241pegase"}


def load(name: str):
    import pandapower.networks as pn
    if name == "multivoltage_cgmes":
        # A network that came as CIM: its model is kept and registered, so differences name its own parts.
        import cim_changes as cc
        import cim_model
        import network_loader as nl
        net = nl.read_cgmes(str(_FIXTURE))
        cc.use_model(name, cim_model.CimModel.read(str(_FIXTURE)))
        net.bus["name"] = net.bus["name"].astype(object)
        return net
    net = getattr(pn, LARGE[name])() if name in LARGE else _networks()[name]()
    net.bus["name"] = net.bus["name"].astype(object)
    return net


# -- proposals --------------------------------------------------------------------------------------------------------------

def edit_catalogue(net) -> dict[str, list[dict]]:
    """Every kind of edit, on elements this network has (a kind it cannot take is left out)."""
    out: dict[str, list[dict]] = {}
    buses = list(net.bus.index[net.bus.in_service.astype(bool)])
    kv = net.bus.vn_kv
    same = [(a, b) for a in buses[:40] for b in buses[:40] if a < b and kv[a] == kv[b]]
    load0 = int(net.load.index[0]) if len(net.load) else None
    line0 = int(net.line.index[min(1, len(net.line) - 1)]) if len(net.line) else None
    out["add a load"] = [{"op": "add_load", "name": "DC", "bus": int(buses[len(buses) // 2]), "p_mw": 2.0}]
    out["add a unit"] = [{"op": "add_generator", "name": "G", "bus": int(buses[len(buses) // 3]), "p_max_mw": 3.0,
                          "kind": "pv"}]
    out["add a shunt"] = [{"op": "add_shunt", "name": "C", "bus": int(buses[len(buses) // 3]), "q_mvar": 1.0,
                           "kind": "capacitor"}]
    if same and line0 is not None:
        out["add a line"] = [{"op": "add_line", "from_bus": int(same[-1][0]), "to_bus": int(same[-1][1]),
                              "length_km": 3.0, "line_like": line0}]
    if len(net.trafo):
        t = net.trafo.index[0]
        out["add a transformer"] = [{"op": "add_trafo", "hv_bus": int(net.trafo.at[t, "hv_bus"]),
                                     "lv_bus": int(net.trafo.at[t, "lv_bus"]), "trafo_like": int(t)}]
        out["upgrade a transformer"] = [{"op": "upgrade_trafo", "trafo": int(t),
                                         "new_sn_mva": float(net.trafo.at[t, "sn_mva"]) * 2}]
        out["a transformer's impedance"] = [{"op": "set_parameter", "element_type": "trafo", "element": int(t),
                                             "values": {"vk_percent": min(float(net.trafo.at[t, "vk_percent"]) * 1.1, 40.0)}}]
    if line0 is not None:
        out["reinforce a line"] = [{"op": "reinforce_line", "line": line0, "factor": 2.0}]
        out["a line out"] = [{"op": "set_parameter", "element_type": "line", "element": line0,
                              "values": {"in_service": False}}]
        out["a line's length"] = [{"op": "set_parameter", "element_type": "line", "element": line0,
                                   "values": {"length_km": float(net.line.at[line0, "length_km"]) * 0.7}}]
    if load0 is not None:
        out["a load's power"] = [{"op": "set_parameter", "element_type": "load", "element": load0,
                                  "values": {"p_mw": 1.5}}]
        out["scale the loads"] = [{"op": "scale", "element_type": "load", "all": True, "factor": 1.2}]
        bus = int(net.load.at[load0, "bus"])
        near = [b for b in buses if b != bus and kv[b] == kv[bus]]
        far = [b for b in buses if kv[b] != kv[bus]]
        if near:
            out["move a load"] = [{"op": "move_element", "element_type": "load", "element": load0, "to_bus": near[0]}]
        if far:
            out["move a load to another voltage"] = [{"op": "move_element", "element_type": "load", "element": load0,
                                                      "to_bus": far[0]}]
    if len(net.gen):
        g = int(net.gen.index[0])
        out["a unit's voltage set-point"] = [{"op": "set_parameter", "element_type": "gen", "element": g,
                                              "values": {"vm_pu": 1.03}}]
    if line0 is not None:
        out["a new bus with a line"] = [{"op": "add_bus_with_connection", "name": "Site", "connect_to": int(buses[-1]),
                                         "length_km": 4.0, "line_like": line0}]
    hv = max(buses, key=lambda b: float(kv[b]))
    lv = next((v for v in (0.4, 0.69, 6.6, 13.8, 33.0, 66.0) if not (kv == v).any() and v < float(kv[hv])), 0.4)
    out["a new bus with a transformer and a load"] = [
        {"op": "add_bus_with_transformer", "name": "Site LV", "connect_to": int(hv), "vn_kv": lv,
         "sn_mva": 2.0, "vk_percent": 6.0, "vkr_percent": 1.0},
        {"op": "add_load", "name": "Site load", "bus": "Site LV", "p_mw": 0.5}]
    if len(getattr(net, "switch", [])):
        closed = net.switch["closed"].astype(bool)
        if (~closed).any():
            out["close a normally open switch"] = [{"op": "set_switch", "switch": int(net.switch.index[~closed][0]),
                                                    "closed": True}]
        if closed.any():
            out["open a switch"] = [{"op": "set_switch", "switch": int(net.switch.index[closed][-1]),
                                     "closed": False}]
    if load0 is not None and line0 is not None:
        out["mixed"] = [{"op": "add_load", "name": "DC", "bus": int(buses[1]), "p_mw": 2.0},
                        {"op": "set_parameter", "element_type": "load", "element": "DC", "values": {"p_mw": 3.0}},
                        {"op": "reinforce_line", "line": line0, "factor": 1.5},
                        {"op": "scale", "element_type": "load", "all": True, "factor": 0.9},
                        {"op": "set_parameter", "element_type": "line", "element": line0,
                         "values": {"length_km": float(net.line.at[line0, "length_km"]) * 2}}]
    return out


def check_proposals(name: str, net, base: dict) -> list[dict]:
    import cim_changes as cc
    import sandbox as sb
    rows = []
    for kind, edits in edit_catalogue(net).items():
        row = {"part": "proposal", "network": name, "case": kind}
        try:
            t0 = time.perf_counter()
            edited, canonical = sb.materialise(net, edits)
            row["edits_s"] = round(time.perf_counter() - t0, 4)
        except Exception as exc:  # noqa: BLE001 — an edit this network cannot take
            rows.append({**row, "skipped": str(exc)[:120]})
            continue
        try:
            t0 = time.perf_counter()
            diff = cc.translate(net, base, canonical, name)
            onto, _notes = cc.project_onto(net, base, diff, name, kind)
            row["cim_s"] = round(time.perf_counter() - t0, 4)
            differences = cc.equivalent(edited, onto, name)
            flow = cc.compare(edited, onto, name)
            model = cc._MODELS.get(name)
            if model is not None:
                # Every object the difference changes or removes is one of the model's own: it applies to its files.
                foreign = [x["mrid"] for x in diff["changed"] + diff["removed"]
                           if x["profile"] != cc.EXTRAS and model.get(x["mrid"]) is None]
                if foreign:
                    differences.append(f"{len(foreign)} changed object(s) not in the model, e.g. {foreign[0]}")
                row["model_objects_only"] = not foreign
            row.update(same_tables=not differences, differences=differences[:5], same_power_flow=flow["matches"],
                       objects_changed=len(diff["added"]) + len(diff["changed"]) + len(diff["removed"]))
        except Exception as exc:  # noqa: BLE001 — a failure is a result
            row.update(same_tables=False, same_power_flow=False, error=f"{type(exc).__name__}: {exc}"[:300])
        rows.append(row)
    return rows


# -- the operator's standing changes ----------------------------------------------------------------------------------------

def operator_catalogue(net) -> dict[str, tuple[dict, dict]]:
    """Generator models and names, as the operator sets them: {case: (generator model, names)}."""
    import generators as gm
    if not gm.editable(net)[0]:
        return {}
    units = gm.describe(net)
    gens = [u for u in units if gm._row_of(net, u["unit"])[0] == "gen"]
    sgens = [u for u in units if gm._row_of(net, u["unit"])[0] == "sgen"]
    out: dict[str, tuple[dict, dict]] = {}
    if gens:
        out["a unit to a power factor"] = ({gens[0]["unit"]: {"control": gm.POWER_FACTOR, "pf": 0.95}}, {})
        out["a unit to a fixed Q"] = ({gens[-1]["unit"]: {"control": gm.FIXED_Q, "q_mvar": 0.0}}, {})
        out["a unit not dispatchable"] = ({gens[0]["unit"]: {"dispatchable": False}}, {})
    if sgens:
        out["a static unit to voltage control"] = ({sgens[0]["unit"]: {"control": gm.VOLTAGE, "vm_pu": 1.01}}, {})
    names = {"bus": {str(int(net.bus.index[1])): "North"}}
    if len(net.line):
        names["line"] = {str(int(net.line.index[0])): "Tie A"}
    if len(net.load):
        names["load"] = {str(int(net.load.index[0])): "Factory"}
    out["renames"] = ({}, names)
    if len(gens) > 1:
        out["several units and renames"] = ({gens[0]["unit"]: {"control": gm.POWER_FACTOR, "pf": 0.9},
                                             gens[1]["unit"]: {"control": gm.FIXED_Q, "q_mvar": 1.0},
                                             **({sgens[0]["unit"]: {"control": gm.VOLTAGE, "vm_pu": 1.0}}
                                                if sgens else {})}, names)
    return out


def check_operator(name: str, net, base: dict) -> list[dict]:
    import cim_changes as cc
    import generators as gm
    import naming
    rows = []
    for case, (model, names) in operator_catalogue(net).items():
        row = {"part": "operator", "network": name, "case": case}
        try:
            t0 = time.perf_counter()
            operated = naming.apply(gm.apply(net, model), names)
            row["edits_s"] = round(time.perf_counter() - t0, 4)
        except Exception as exc:  # noqa: BLE001 — a change this network cannot take
            rows.append({**row, "skipped": str(exc)[:120]})
            continue
        try:
            t0 = time.perf_counter()
            diff = cc.difference_of(net, base, operated, name)
            onto, _notes = cc.project_onto(net, base, diff, name, "operator")
            row["cim_s"] = round(time.perf_counter() - t0, 4)
            differences = cc.equivalent(operated, onto, name)
            flow = cc.compare(operated, onto, name)
            row.update(same_tables=not differences, differences=differences[:5], same_power_flow=flow["matches"],
                       objects_changed=len(diff["added"]) + len(diff["changed"]) + len(diff["removed"]))
        except Exception as exc:  # noqa: BLE001 — a failure is a result
            row.update(same_tables=False, same_power_flow=False, error=f"{type(exc).__name__}: {exc}"[:300])
        rows.append(row)
    return rows


# -- moments of the clock ---------------------------------------------------------------------------------------------------

def measurements(net, tick: int):
    """A tick's measurements for any network: each bus with loads or units measured by its name — demand, generation
    and, on every other bus, the reactive demand — varying with the tick."""
    import load_gen_assignment as lg
    import pandas as pd
    rows = []
    for k, b in enumerate(net.bus.index):
        load = float(net.load.loc[net.load.bus == b, "p_mw"].clip(lower=0).sum()) if len(net.load) else 0.0
        units = sum(float(net[t].loc[net[t].bus == b, "p_mw"].clip(lower=0).sum()) for t in ("gen", "sgen")
                    if len(net[t]))
        if load or units:
            f = 0.7 + 0.1 * ((tick * 7 + k) % 6)
            rows.append({"substation_name": lg.measurement_name(net, b), "consumption": load * f,
                         "production": units * (1.6 - f), "consumption_q": load * f * 0.3 if k % 2 else None})
    return pd.DataFrame(rows)


def check_moments(name: str, net, base: dict) -> list[dict]:
    """For every proposal and a few ticks: the tick's measurements allocated on the proposal itself (as before) /
    the base's moment applied to it by mRID (operating_point.py) — the same network."""
    import cim_changes as cc
    import operating_point as op
    import sandbox as sb
    rows = []
    cases = {"the base": [], **edit_catalogue(net)}
    for kind, edits in cases.items():
        row = {"part": "moment", "network": name, "case": kind}
        try:
            proposal, canonical = sb.materialise(net, edits)
            if edits:
                diff = cc.translate(net, base, canonical, name)
                proposal, _ = cc.project_onto(net, base, diff, name, kind)
        except Exception as exc:  # noqa: BLE001
            rows.append({**row, "skipped": str(exc)[:120]})
            continue
        try:
            shared, differences, t_old, t_new = 0, [], 0.0, 0.0
            for tick in range(3):
                meas = measurements(net, tick)
                subs = list(meas["substation_name"])
                t0 = time.perf_counter()
                old = op.allocate(copy.deepcopy(proposal), meas, subs, 0.95)[0]
                t_old += time.perf_counter() - t0
                t0 = time.perf_counter()
                new = copy.deepcopy(proposal)
                if op.signature(new) is not None and op.signature(new) == op.signature(net):
                    op.apply(new, op.compute(net, meas, subs, 0.95, name), meas, subs, name)
                    shared += 1
                else:
                    new = op.allocate(new, meas, subs, 0.95)[0]
                t_new += time.perf_counter() - t0
                differences += cc.equivalent(old, new, name)
                if old.get("_q_measured_share") != new.get("_q_measured_share"):
                    differences.append(f"measured reactive share {old.get('_q_measured_share')} vs "
                                       f"{new.get('_q_measured_share')}")
            row.update(same_tables=not differences, differences=differences[:5], shared_ticks=shared,
                       edits_s=round(t_old, 4), cim_s=round(t_new, 4))
        except Exception as exc:  # noqa: BLE001
            row.update(same_tables=False, error=f"{type(exc).__name__}: {exc}"[:300])
        rows.append(row)
    return rows


# -- the state saved as files ----------------------------------------------------------------------------------------------

def check_state(name: str, net, base: dict) -> list[dict]:
    """Every proposal's and every operator change's difference written as an IEC 61970-552 file and read back
    (cim_changes.to_xml / from_xml): the network projected from the file is the one its source builds."""
    import cim_changes as cc
    import generators as gm
    import naming
    import sandbox as sb
    rows = []
    sources = [(f"proposal: {k}", "edits", e) for k, e in edit_catalogue(net).items()]
    sources += [(f"operator: {k}", "operator", m) for k, m in operator_catalogue(net).items()]
    for case, how, source in sources:
        row = {"part": "state", "network": name, "case": case}
        try:
            if how == "edits":
                built, canonical = sb.materialise(net, source)
                diff = cc.translate(net, base, canonical, name)
            else:
                built = naming.apply(gm.apply(net, source[0]), source[1])
                diff = cc.difference_of(net, base, built, name)
        except Exception as exc:  # noqa: BLE001
            rows.append({**row, "skipped": str(exc)[:120]})
            continue
        try:
            xml = cc.to_xml(diff, description=case)
            onto, _notes = cc.project_onto(net, base, cc.from_xml(xml), name, "state")
            differences = cc.equivalent(built, onto, name)
            row.update(same_tables=not differences, differences=differences[:5], bytes=len(xml))
        except Exception as exc:  # noqa: BLE001
            row.update(same_tables=False, error=f"{type(exc).__name__}: {exc}"[:300])
        rows.append(row)
    return rows


# -- the run ----------------------------------------------------------------------------------------------------------------

PARTS = {"proposals": check_proposals, "operator": check_operator, "moments": check_moments, "state": check_state}


def run(networks: list[str], out: pathlib.Path, parts: list[str]) -> dict:
    import cim_changes as cc
    out.mkdir(parents=True, exist_ok=True)
    started = datetime.datetime.now().isoformat(timespec="seconds")
    results: list[dict] = []
    for name in networks:
        net = load(name)
        t0 = time.perf_counter()
        base = cc.statements(net, name)
        results.append({"part": "model", "network": name, "case": "written as statements",
                        "cim_s": round(time.perf_counter() - t0, 4), "buses": len(net.bus),
                        "objects": len(base["objects"])})
        for part in parts:
            for row in PARTS[part](name, copy.deepcopy(net), base):
                results.append(row)
                ok = row.get("skipped") and "skipped" or ("ok" if row.get("same_tables") and row.get(
                    "same_power_flow", True) else "DIFFERS")
                print(f"{name:16} {row['part']:9} {row['case'][:44]:44} {ok}", flush=True)
    checked = [r for r in results if "same_tables" in r]
    summary = {"checked": len(checked), "same": sum(1 for r in checked if r["same_tables"]
                                                       and r.get("same_power_flow", True)),
               "skipped": sum(1 for r in results if r.get("skipped"))}
    (out / "results.json").write_text(json.dumps(results, indent=1, default=str), encoding="utf-8")
    keys = ["part", "network", "case", "same_tables", "same_power_flow", "edits_s", "cim_s", "objects_changed",
            "shared_ticks", "skipped", "error"]
    with open(out / "results.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)
    manifest = {"kind": "cim_central", "started": started,
                "finished": datetime.datetime.now().isoformat(timespec="seconds"),
                "commit": _git("rev-parse", "HEAD"), "uncommitted_changes": bool(_git("status", "--porcelain")),
                "command": " ".join(["python -m evaluation.cim_central", *sys.argv[1:]]),
                "python": platform.python_version(), "networks": networks, "parts": parts, "summary": summary}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"\n{summary['same']} of {summary['checked']} the same ({summary['skipped']} skipped) — {out}")
    return manifest


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m evaluation.cim_central", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="check every part on every network")
    r.add_argument("--networks", default="", help="comma-separated (default: the standard ones)")
    r.add_argument("--large", action="store_true", help="also the PEGASE 1354 and 9241 networks")
    r.add_argument("--parts", default=",".join(PARTS), help=f"comma-separated, of {', '.join(PARTS)}")
    r.add_argument("--out", default="")
    args = p.parse_args(argv)
    networks = [n for n in args.networks.split(",") if n] or list(_networks()) + (list(LARGE) if args.large else [])
    out = pathlib.Path(args.out or f"evaluation_runs/cim-central-{datetime.datetime.now():%Y%m%d-%H%M}")
    manifest = run(networks, out, [x for x in args.parts.split(",") if x])
    return 0 if manifest["summary"]["same"] == manifest["summary"]["checked"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
