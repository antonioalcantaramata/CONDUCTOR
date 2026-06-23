#!/usr/bin/env python
"""network_loader.py — Generic pandapower network loader and on-the-fly Ybus builder.

Public API
----------
load_profile(profile_name)     -> dict
    Read backend/network_profiles/{profile_name}.yaml and return as dict.

load_network(profile)          -> pp.pandapowerNet
    Load a pandapower network from a profile dict.

build_ybus(net)                -> (db_full, db_n1_line, db_n1_trafo)
    Compute the three admittance databases on-the-fly using pandapower's
    internal pypower representation.  Format is identical to the legacy
    pre-computed pickle files so all existing flex_engine / main_backend code works
    without modification.

Internal helpers
----------------
_build_ybus_entry(net)         -> dict
    Build one admittance dict from a *solved* pandapower network.
"""

from __future__ import annotations

import copy
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandapower as pp
import yaml

# ---------------------------------------------------------------------------
# Profile loading
# ---------------------------------------------------------------------------

def load_profile(profile_name: str, backend_dir: Optional[str] = None) -> dict:
    """Read ``network_profiles/{profile_name}.yaml`` and return as dict.

    Parameters
    ----------
    profile_name : str
        Name of the profile (without .yaml extension), e.g. "pglib_case14", "ieee14".
    backend_dir : str, optional
        Absolute path to the backend/ directory.  Defaults to the directory
        containing this file.
    """
    base = Path(backend_dir or Path(__file__).parent)
    path = base / "network_profiles" / f"{profile_name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Profile not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Network loading
# ---------------------------------------------------------------------------

def load_network(profile: dict, backend_dir: Optional[str] = None) -> pp.pandapowerNet:
    """Load a pandapower network from a profile dict.

    Supported ``profile["source"]`` values:
    - ``"excel"``     → pp.from_excel(excel_path)
    - ``"matpower"``  → pandapower MATPOWER converter
    - ``"ieee"``      → pp.networks.case{case}()  (e.g. case14, case30, case118)
    - ``"json"``      → pp.from_json(json_path)
    - ``"pickle"``    → pp.from_pickle(pickle_path)
    """
    base = Path(backend_dir or Path(__file__).parent)
    source = profile.get("source", "").lower()

    if source == "excel":
        path = base.parent / profile["excel_path"]
        return pp.from_excel(str(path))

    if source == "matpower":
        from pandapower.converter.matpower import from_mpc
        path = base.parent / profile["matpower_path"]
        return from_mpc(str(path), f_hz=50)

    if source == "ieee":
        case = profile.get("case", "case14")
        loader = getattr(pp.networks, case, None)
        if loader is None:
            raise ValueError(f"Unknown pp.networks case: {case}")
        return loader()

    if source == "json":
        path = base.parent / profile["json_path"]
        return pp.from_json(str(path))

    if source == "pickle":
        path = base.parent / profile["pickle_path"]
        return pp.from_pickle(str(path))

    raise ValueError(f"Unsupported network source: '{source}'")


# ---------------------------------------------------------------------------
# MATPOWER gen → sgen conversion
# ---------------------------------------------------------------------------

def gen_to_sgen(net: pp.pandapowerNet) -> tuple:
    """Convert ``net.gen`` PV-bus generators to ``net.sgen`` for OPF/flexibility
    tool compatibility.  The slack bus generator (already represented by
    ``net.ext_grid``) is skipped so it is never duplicated.

    Returns
    -------
    net : pandapowerNet
        The same network object, modified in-place.
    created_names : list[str]
        Names of the sgen entries that were created (e.g. ``["Gen_bus1"]``).
    kept_condenser_names : list[str]
        Names of zero-real-power generators kept as net.gen PV buses for voltage
        regulation (e.g. ``["Gen_bus2", "Gen_bus5", "Gen_bus7"]``).
    """
    slack_buses = set(int(b) for b in net.ext_grid["bus"].tolist())
    created_names = []
    kept_condenser_names = []
    rows_to_drop = []

    for idx, row in net.gen.iterrows():
        bus_id = int(row["bus"])
        if bus_id in slack_buses:
            continue
        max_p = float(row["max_p_mw"]) if "max_p_mw" in row.index and not np.isnan(float(row["max_p_mw"])) else float(row["p_mw"])
        # Skip synchronous condensers (zero real-power capacity): keep them as
        # net.gen PV buses so they continue to regulate voltage.
        if max_p < 1.0:
            kept_condenser_names.append(f"Gen_bus{bus_id}")
            continue
        name = f"Gen_bus{bus_id}"
        min_p = float(row["min_p_mw"]) if "min_p_mw" in row.index and not np.isnan(float(row["min_p_mw"])) else 0.0
        pp.create_sgen(
            net,
            bus=bus_id,
            p_mw=float(row["p_mw"]),
            q_mvar=0.0,
            name=name,
            max_p_mw=max_p,
            min_p_mw=min_p,
            controllable=True,
            in_service=bool(row.get("in_service", True)),
        )
        rows_to_drop.append(idx)
        created_names.append(name)

    net.gen.drop(index=rows_to_drop, inplace=True)
    net.gen.reset_index(drop=True, inplace=True)
    # Mark the network so assignment functions know to use the Generic path.
    net["_gen_to_sgen_applied"] = True
    return net, created_names, kept_condenser_names


# ---------------------------------------------------------------------------
# On-the-fly Ybus builder
# ---------------------------------------------------------------------------

def _build_ybus_entry(net: pp.pandapowerNet) -> dict:
    """Compute branch admittances from a **solved** pandapower network.

    Requires ``net._ppc`` to be populated (i.e. pp.runpp must have been called).
    Produces the dict format expected by ``flex_engine.optimization_model_base``:

        {
            "Yff_r": {((from_bus, to_bus), tap): float, ...},
            "Yff_i": {((from_bus, to_bus), tap): float, ...},
            "Yft_r": {((from_bus, to_bus), tap): float, ...},
            "Yft_i": {((from_bus, to_bus), tap): float, ...},
            "TAPS":            list,
            "trafo_ranges":    {trafo_idx: [min_tap, max_tap]},
            "trafo_defaults":  {trafo_idx: default_tap},
            "branch_to_trafo": {(from_bus, to_bus): trafo_idx},
        }

    For lines  : tap_key = 1  (matching flex_engine's ``base_t = 1`` convention)
    For trafos : tap_key = tap_neutral  (matching flex_engine's ``t_default`` lookup)
    """
    from pandapower.pypower.idx_brch import F_BUS, T_BUS, BR_R, BR_X, BR_B, TAP, SHIFT

    ppc = net._ppc

    # Build pandapower bus index ↔ ppc bus index mapping
    bus_lookup = net._pd2ppc_lookups["bus"]  # array: pp_idx → ppc_idx
    pp2ppc = {int(idx): int(bus_lookup[idx]) for idx in net.bus.index}
    ppc2pp = {v: k for k, v in pp2ppc.items()}

    Yff_r: dict = {}
    Yff_i: dict = {}
    Yft_r: dict = {}
    Yft_i: dict = {}
    branch_to_trafo: dict = {}
    trafo_defaults:  dict = {}
    trafo_ranges:    dict = {}

    # Pre-build trafo lookup: (hv_bus_pp, lv_bus_pp) → trafo_idx
    trafo_by_buses: dict = {}
    for tidx, t in net.trafo.iterrows():
        hv, lv = int(t["hv_bus"]), int(t["lv_bus"])
        trafo_by_buses[(hv, lv)] = tidx
        trafo_by_buses[(lv, hv)] = tidx

    for k in range(len(ppc["branch"])):
        fb_ppc = int(ppc["branch"][k, F_BUS])
        tb_ppc = int(ppc["branch"][k, T_BUS])

        if fb_ppc not in ppc2pp or tb_ppc not in ppc2pp:
            continue  # disconnected / out-of-service bus

        r        = float(ppc["branch"][k, BR_R])
        x        = float(ppc["branch"][k, BR_X])
        b_ch     = float(ppc["branch"][k, BR_B])
        tap      = float(ppc["branch"][k, TAP])
        shift_deg = float(ppc["branch"][k, SHIFT])

        if tap == 0.0:
            tap = 1.0

        i = ppc2pp[fb_ppc]   # pandapower from_bus
        j = ppc2pp[tb_ppc]   # pandapower to_bus

        # Series admittance (protect against zero impedance branches)
        ys = (1.0 / complex(r, x)) if (r != 0.0 or x != 0.0) else 0j
        bc = 1j * b_ch / 2

        # Tap as complex phasor (real tap × phase shift)
        tap_complex = tap * np.exp(1j * shift_deg * np.pi / 180.0)

        # Pi-model branch admittance elements
        Yff = (ys + bc) / abs(tap_complex) ** 2   # from-end self
        Yft = -ys / np.conj(tap_complex)           # from→to mutual
        Ytf = -ys / tap_complex                    # to→from mutual
        Ytt = ys + bc                              # to-end self

        # Determine tap key and register trafo metadata
        is_trafo = (i, j) in trafo_by_buses or (j, i) in trafo_by_buses
        if is_trafo:
            tidx = trafo_by_buses.get((i, j), trafo_by_buses.get((j, i)))
            _raw = (
                net.trafo.at[tidx, "tap_neutral"]
                if "tap_neutral" in net.trafo.columns
                else None
            )
            try:
                tap_neutral = int(_raw) if _raw is not None and _raw == _raw else 0
            except (TypeError, ValueError):
                tap_neutral = 0
            branch_to_trafo[(i, j)] = tidx
            branch_to_trafo[(j, i)] = tidx
            if tidx not in trafo_defaults:
                trafo_defaults[tidx] = tap_neutral
                trafo_ranges[tidx]   = [tap_neutral, tap_neutral]
            tap_key = tap_neutral
        else:
            tap_key = 1  # lines always use tap=1 key

        # Store both directions
        Yff_r[(i, j), tap_key] = float(Yff.real)
        Yff_i[(i, j), tap_key] = float(Yff.imag)
        Yft_r[(i, j), tap_key] = float(Yft.real)
        Yft_i[(i, j), tap_key] = float(Yft.imag)

        Yff_r[(j, i), tap_key] = float(Ytt.real)
        Yff_i[(j, i), tap_key] = float(Ytt.imag)
        Yft_r[(j, i), tap_key] = float(Ytf.real)
        Yft_i[(j, i), tap_key] = float(Ytf.imag)

    return {
        "Yff_r":         Yff_r,
        "Yff_i":         Yff_i,
        "Yft_r":         Yft_r,
        "Yft_i":         Yft_i,
        "TAPS":          list(set(trafo_defaults.values())) if trafo_defaults else [1],
        "trafo_ranges":  trafo_ranges,
        "trafo_defaults": trafo_defaults,
        "branch_to_trafo": branch_to_trafo,
    }


def _try_runpp(net: pp.pandapowerNet) -> bool:
    """Run power flow; return True on success, False on any failure."""
    try:
        pp.runpp(net, numba=False, calculate_voltage_angles=True)
        return True
    except pp.powerflow.LoadflowNotConverged:
        return False
    except Exception:
        return False


def build_ybus(net: pp.pandapowerNet) -> tuple[dict, dict, dict]:
    """Compute ``(db_full, db_n1_line, db_n1_trafo)`` on-the-fly.

    Parameters
    ----------
    net : pp.pandapowerNet
        The base network (not modified — all solves use deep copies).

    Returns
    -------
    db_full : dict
        Admittance dict for the intact network.
    db_n1_line : dict
        ``{line_idx: admittance_dict}`` for each in-service line.
        Lines that cause islanding/divergence are omitted (→ KeyError → 404).
    db_n1_trafo : dict
        ``{trafo_idx: admittance_dict}`` for each in-service trafo.
        Non-converging contingencies are omitted.

    Raises
    ------
    RuntimeError
        If the full-network power flow fails (base case must converge).
    """
    # --- Full network (base case) ---
    net_base = copy.deepcopy(net)
    if not _try_runpp(net_base):
        raise RuntimeError(
            "build_ybus: base-case power flow did not converge. "
            "Check network data (connectivity, voltage setpoints)."
        )
    db_full = _build_ybus_entry(net_base)

    n_lines  = len(net.line[net.line["in_service"] == True])
    n_trafos = len(net.trafo[net.trafo["in_service"] == True])
    print(f"[build_ybus] Base case OK. Computing N-1 for {n_lines} lines + {n_trafos} trafos...")
    t0 = time.perf_counter()

    # --- N-1 lines ---
    db_n1_line: dict = {}
    skipped_lines = 0
    for line_idx in net.line.index:
        if not net.line.at[line_idx, "in_service"]:
            continue
        net_n1 = copy.deepcopy(net)
        net_n1.line.at[line_idx, "in_service"] = False
        if _try_runpp(net_n1):
            db_n1_line[line_idx] = _build_ybus_entry(net_n1)
        else:
            skipped_lines += 1  # islanded/non-converging — omit (→ 404 on request)

    # --- N-1 trafos ---
    db_n1_trafo: dict = {}
    skipped_trafos = 0
    for trafo_idx in net.trafo.index:
        if not net.trafo.at[trafo_idx, "in_service"]:
            continue
        net_n1 = copy.deepcopy(net)
        net_n1.trafo.at[trafo_idx, "in_service"] = False
        if _try_runpp(net_n1):
            db_n1_trafo[trafo_idx] = _build_ybus_entry(net_n1)
        else:
            skipped_trafos += 1

    elapsed = time.perf_counter() - t0
    print(
        f"[build_ybus] Done in {elapsed:.1f}s — "
        f"lines: {len(db_n1_line)} OK / {skipped_lines} islanded; "
        f"trafos: {len(db_n1_trafo)} OK / {skipped_trafos} islanded"
    )
    return db_full, db_n1_line, db_n1_trafo
