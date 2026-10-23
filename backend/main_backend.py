"""
Power-System Digital Twin Backend (FastAPI).

RSA, CA, flexibility, and supporting engines run through this FastAPI
backend so the user interfaces and external clients can drive
simulations against the active network profile.

Overview
---------------------------
1. **Startup (``lifespan``)**: load the measurement database,
   the pandapower Excel network, and three pre-computed admittance-matrix
   (``Ybus``) databases into the global app_data array. These loads
   are time consuming (10-20s) so they happen once at boot to keep per-request
   latency in the ~0.03s range.
2. **Time control** (``/api/time/*``): advances the simulation clock through
   the pickled timestamp list.
3. **Grid assessment** (``/api/grid/rsa``): runs a Real-Time Security
   Assessment against the currently selected timestamp.
4. **Contingency** (``/api/contingency/*``): simulates N-1 outages and can
   optionally dispatch the Pyomo optimizer to cure the resulting violations.
5. **Flexibility** (``/api/flexibility/optimize``): runs the Pyomo AI to
   find the cheapest generator dispatch that keeps the grid secure.
6. **KPIs** (``/api/kpi/evaluate``): computes the three official KPIs
   defined in the notebook (``main_KPIs_Included.ipynb``).
7. **EDDK integration** (``/api/eddk/*``): pulls 24-hour load profiles
   from the Energy Data Denmark data space and (placeholder) pushes
   results back(remains to be completed).
"""

import os

# One thread per solve, unless the operator set otherwise: IPOPT runs as a
# subprocess and inherits these. On distribution-sized networks its
# multithreaded linear algebra only adds overhead — measured on the IEEE 30
# demo, a 9-moment N-1 study took 78 s with it and 52 s without, and four
# parallel workers each using every core were barely faster than one.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import asyncio
import contextvars
import functools
import hashlib
import json
import math
import threading
import logging
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from fastapi import FastAPI, Form, HTTPException, UploadFile, File
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
import io
import pandapower as pp
from contextlib import asynccontextmanager
import copy
import pandas as pd
import numpy as np
import requests
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator
import anyio
from pyomo.environ import value
from pyomo.opt import TerminationCondition
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import load_gen_assignment as lg
import rsa_engine as rs
# Never ship a number that would be wrong to state. See backend/withdrawal.py.
import withdrawal
# Element naming lives in its own module so rsa_engine spells a name the same
# way this file does. They disagreed about transformers, and the system map —
# which joins topology to results by name — could not match any of them.
from element_names import (
    clean_label as _clean_label,
    bus_display_name as _bus_display_name,
    line_display_name as _line_display_name,
    trafo_display_name as _trafo_display_name,
    measurement_name,
)
import ca_engine as ca

import flex_engine as fe
import flex_options as fo
import kpis as kp
import scopf
import network_loader as nl
import data_quality as dq
import scenarios as scn
import costs as uc
import generators as gm
import naming
import nominal as nom
import sandbox as sb

from dotenv import load_dotenv

load_dotenv()  # This safely loads your EDDK_API_TOKEN from the .env file
EDDK_API_TOKEN = os.getenv("EDDK_API_TOKEN")

import pipeline_functions as eddk_pipe


BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(BACKEND_DIR, ".."))
DATA_FILES_DIR = os.path.join(REPO_ROOT, "data_files")
SYSTEMS_DIR = os.path.join(REPO_ROOT, "systems")
DEFAULT_GRID_PROFILE = "pglib_case30"
_LAST_UPLOAD_SENTINEL      = os.path.join(SYSTEMS_DIR,    "last_uploaded.json")
_LAST_TIMESERIES_SENTINEL  = os.path.join(DATA_FILES_DIR, "last_uploaded_timeseries.json")
_LAST_FORECAST_SENTINEL    = os.path.join(DATA_FILES_DIR, "last_uploaded_forecast.json")
# Network proposals, kept across restarts for the network they were built on.
_PROPOSALS_FILE            = os.path.join(SYSTEMS_DIR,    "proposals.json")
_ASSUMPTIONS_FILE          = os.path.join(SYSTEMS_DIR,    "assumptions.json")
_SCENARIO_FILE             = os.path.join(SYSTEMS_DIR,    "scenario.json")


def _ensure_storage_dirs() -> None:
    os.makedirs(DATA_FILES_DIR, exist_ok=True)
    os.makedirs(SYSTEMS_DIR, exist_ok=True)


def _sanitize_upload_name(filename: str | None, fallback_stem: str, suffix: str) -> str:
    raw_name = os.path.basename((filename or "").strip())
    stem, ext = os.path.splitext(raw_name)
    clean_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem or fallback_stem).strip("._") or fallback_stem
    clean_ext = ext.lower() if ext else suffix
    if clean_ext != suffix:
        clean_ext = suffix
    return f"{clean_stem}{clean_ext}"


def _persist_uploaded_bytes(
    target_dir: str,
    filename: str | None,
    content: bytes,
    fallback_stem: str,
    suffix: str,
) -> str:
    _ensure_storage_dirs()
    stored_name = _sanitize_upload_name(filename, fallback_stem, suffix)
    if filename:
        # Prefixed with its kind: a measurement file and a forecast file that
        # share a name (`data.csv`) used to land on the same path, so the
        # measurements on disk were silently replaced by the forecast.
        stored_name = f"{fallback_stem}__{stored_name}"
    stored_path = os.path.join(target_dir, stored_name)
    with open(stored_path, "wb") as fh:
        fh.write(content)
    return stored_path


_MAX_UPLOAD_BYTES = int(float(os.environ.get("CONDUCTOR_MAX_UPLOAD_MB", "200")) * 1024 * 1024)


def _check_upload_size(content: bytes, label: str) -> None:
    """Refuse an upload larger than the configured cap (HTTP 413).

    Uploads were read whole with no limit; a mistaken multi-gigabyte file
    would exhaust memory in the one process every tool runs in.
    """
    if len(content) > _MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"{label} is {len(content) / 1e6:.0f} MB; the limit is "
                   f"{_MAX_UPLOAD_BYTES / 1e6:.0f} MB (set CONDUCTOR_MAX_UPLOAD_MB to change it).",
        )


def _safe_float(v, ndigits: int = 4):
    """Return a JSON-safe rounded float, or None for NaN/inf values."""
    try:
        f = float(v)
        return None if not math.isfinite(f) else round(f, ndigits)
    except (TypeError, ValueError):
        return None


# -------------------------------------------------------------------
# from mqtt_publisher import publish_rsa_results, publish_ca_results, publish_flex_rsa_results, publish_flex_ca_results

# Generator maximum active-power capacities (MW), keyed by substation.

# Used by:
#   1. /api/grid/rsa  -> "total_available_capacity_mw" flexibility metric
#   2. /api/kpi/evaluate -> KPI-1 (Target Demand Flexibility)


# ---------------------------------------------------------------------------
# Network-derived lookups — populated at startup from the loaded network.
# All call-sites reference these module-level objects directly; the lifespan()
# function (and upload_network) mutates them in-place so no call-site changes
# are needed.  T4 will drive these from YAML network profiles instead.
# ---------------------------------------------------------------------------

# {substation_name -> Pg_max_mw} — built from net.sgen/net.gen["max_p_mw"].
PG_MAX_DATA: dict = {}

# Canonical substation/generator name list for lg.assign_* fuzzy matching.
substations: list = []


def get_substation_names(net) -> list:
    """Derive the canonical substation name list from a loaded pandapower network.

    Reads bus names for buses that carry at least one load, which gives the
    correct list for substation-style measured networks (one load per substation bus) and
    a sensible fallback for IEEE / MATPOWER networks (bus names like 'Bus 2').
    Falls back to all bus names if no loads are present.
    MATPOWER networks have no bus names — generates 'Bus {id}' labels.
    """
    def _nonempty(vals):
        return [v for v in vals if v and str(v).strip()]

    if net.load.empty or "name" not in net.bus.columns:
        raw = net.bus["name"].dropna().tolist()
    else:
        load_bus_ids = net.load["bus"].unique()
        raw = net.bus.loc[net.bus.index.isin(load_bus_ids), "name"].dropna().tolist()
        # Also include named generators not already covered (IEEE net.gen).
        if not net.gen.empty and "name" in net.gen.columns:
            for n in net.gen["name"].dropna().tolist():
                if n and n not in raw:
                    raw.append(n)

    # Always text: some networks name their buses with numbers (case33bw: 1, 2, …),
    # and every consumer — matching, the CSV, the agent's prompt — reads names as text.
    names = [str(v).strip() for v in _nonempty(raw)]
    if names:
        return list(dict.fromkeys(names))
    # Fallback for MATPOWER / nameless networks: use bus indices of load buses
    if not net.load.empty:
        return [f"Bus {i}" for i in sorted(net.load["bus"].unique())]
    return [f"Bus {i}" for i in net.bus.index]


def get_pg_max_data(net) -> dict:
    """Build a {name -> Pg_max_mw} dict from net.sgen and net.gen.

    Reads the ``max_p_mw`` column when present and > 0; falls back to
    1.5 × ``p_mw`` so the OPF always has a finite upper bound even for
    networks without explicit capacity limits.
    MATPOWER networks have no generator names — uses 'Gen_bus{bus_id}'.
    """
    def _gen_name(raw_name, fallback: str) -> str:
        s = str(raw_name).strip() if raw_name is not None else ""
        return s if s else fallback

    result = {}
    if not net.sgen.empty:
        key_col = "substation_name" if "substation_name" in net.sgen.columns else "name"
        for idx, row in net.sgen.iterrows():
            raw = row.get(key_col, None)
            name = _gen_name(raw, f"sgen_{idx}")
            max_p = row.get("max_p_mw", None) if "max_p_mw" in net.sgen.columns else None
            if max_p is not None and pd.notna(max_p) and float(max_p) > 0:
                result[name] = float(max_p)
            else:
                result[name] = max(float(row.get("p_mw", 0.0)) * 1.5, 0.001)
    if not net.gen.empty:
        for idx, row in net.gen.iterrows():
            bus_id = int(row.get("bus", idx))
            name = _gen_name(row.get("name", None), f"Gen_bus{bus_id}")
            max_p = row.get("max_p_mw", None) if "max_p_mw" in net.gen.columns else None
            if max_p is not None and pd.notna(max_p) and float(max_p) > 0:
                result[name] = float(max_p)
            else:
                result[name] = max(float(row.get("p_mw", 0.0)) * 1.5, 0.001)
    return result


def prepare_measurement_df(df):
    """Normalize a raw measurement row-set so the assignment engines accept it.

    This helper papers
    over naming differences so ``load_gen_assignment.py`` always sees
    the schema it expects.

    Transformations applied:
      * ``element`` column -> renamed to ``substation_name`` (required
        by ``lg.assign_*_from_measurements``).
      * ``Pg_new`` column -> duplicated into ``production`` and a
        zero-filled ``consumption`` column is added. This lets the
        same frame drive both load and generator assignment passes.

    Parameters
    df : pandas.DataFrame
        A single timestamp slice from ``app_data["measurements"]``.

    Returns
    pandas.DataFrame
        A shallow copy with schema suitable for the assignment engines.
        The original frame is not mutated.
    """
    df_copy = df.copy(deep=True)
    if 'element' in df_copy.columns:
        df_copy = df_copy.rename(columns={'element': 'substation_name'})
    if 'Pg_new' in df_copy.columns:
        df_copy['production'] = df_copy['Pg_new']
        df_copy['consumption'] = 0.0
    return df_copy


def extract_flexibility_results(model, net_base, ts):
    """Extract only *active* regulation rows from a solved Pyomo model.

    Walks the optimized model's generator set (``model.G``) and external
    grid set (``model.Ext``) and returns a list of dispatch records for
    elements whose active or reactive output moved (by more than 1 kW / 1 kvar).
    Reactive output is reported too: without it a unit that only changed Q
    was missing, and a dispatch could not be checked against a power flow.

    This is the lean view consumed by the Flexibility / Contingency
    activation tables in the UI. It is deliberately different from the
    *full* regulation DataFrame produced by
    ``flex_engine.prepare_regulation_df_rsa``: the KPI endpoint needs
    all generators (including unchanged ones) to compute sums correctly
    and therefore uses the engine helper directly — not this function.

    Parameters
    ----------
    model : pyomo.ConcreteModel
        Solved optimization model from ``flex_engine.optimization_model_base``.
    net_base : pandapowerNet
        Base (pre-optimization) network, used to read the external grid's
        baseline active-power injection.
    ts : str
        Current simulation timestamp (unused here but kept in signature
        for parity with the engine helpers).

    Returns
    -------
    list[dict]
        One record per generator / external-grid element whose setpoint
        changed. Each record has keys: ``element``, ``type``, ``Pg_base``,
        ``Pg_new``, ``Pg_up``, ``Pg_down`` and the same four for ``Qg``.
    """
    records = []
    # Local generators
    def q_fields(q_base: float, q_new: float) -> dict:
        return {"Qg_base": round(q_base, 4), "Qg_new": round(q_new, 4),
                "Qg_up": round(max(q_new - q_base, 0), 4),
                "Qg_down": round(max(q_base - q_new, 0), 4)}

    for g in model.G:
        Pg_new_val = value(model.Pg_new[g])
        Pg_base_val = value(model.Pg_base[g])
        Qg_new_val = value(model.Qg_new[g])
        Qg_base_val = value(model.Qg_base[g])

        if abs(Pg_new_val - Pg_base_val) > 0.001 or abs(Qg_new_val - Qg_base_val) > 0.001:
            records.append({
                "element": str(g),
                "type": "Generator",
                # The bus, stated: unit names ("Gen_bus7") and substation names
                # ("Bus 8") use different numberings, and an answer guessed one
                # from the other.
                "bus": _bus_display_name(net_base, int(value(model.gen_bus[g]))),
                "Pg_base": round(Pg_base_val, 4),
                "Pg_new": round(Pg_new_val, 4),
                "Pg_up": round(max(Pg_new_val - Pg_base_val, 0), 4),
                "Pg_down": round(max(Pg_base_val - Pg_new_val, 0), 4),
                **q_fields(Qg_base_val, Qg_new_val),
            })

    # Check for external grid in the model Pymo does not store its baseline: Slack bus check necessary
    for e in model.Ext:
        try:
            Pext_new_val = value(model.Pext[e])
            Qext_new_val = value(model.Qext[e])
            Pext_base_val, Qext_base_val = Pext_new_val, Qext_new_val
            if hasattr(net_base, 'res_ext_grid') and not net_base.res_ext_grid.empty:
                ext_idx = net_base.ext_grid[net_base.ext_grid['name'] == e].index
                if not ext_idx.empty:
                    Pext_base_val = net_base.res_ext_grid.at[ext_idx[0], 'p_mw']
                    Qext_base_val = net_base.res_ext_grid.at[ext_idx[0], 'q_mvar']
                    # No converged base power flow: no "before", so no change claimed.
                    if Pext_base_val != Pext_base_val:
                        Pext_base_val = Pext_new_val
                    if Qext_base_val != Qext_base_val:
                        Qext_base_val = Qext_new_val

            if abs(Pext_new_val - Pext_base_val) > 0.001 or abs(Qext_new_val - Qext_base_val) > 0.001:
                records.append({
                    "element": str(e),
                    "type": "External Grid",
                    "bus": _bus_display_name(net_base, int(value(model.ext_bus[e]))),
                    "Pg_base": round(Pext_base_val, 4),
                    "Pg_new": round(Pext_new_val, 4),
                    "Pg_up": round(max(Pext_new_val - Pext_base_val, 0), 4),
                    "Pg_down": round(max(Pext_base_val - Pext_new_val, 0), 4),
                    **q_fields(float(Qext_base_val), Qext_new_val),
                })
        except Exception:
            pass
    return records


# KPI calculations — ported verbatim from main_KPIs_Included.ipynb
# The function below is the source of truth for the /api/kpi/evaluate endpoint.

# ===================================================================
# Constrained-slack scenario re-solve helper
# ===================================================================
# Takes the model built (and already solved) by
# ``flex_engine.optimization_model_base`` and re-runs Ipopt with the
# external-grid interface constrained to a user-chosen bidirectional
# capacity in MW. Replaces the older binary islanded helper: the slider
# gives a richer "what-if" — at the physical import/export limit it
# matches the engine default, at 0 MW it's a fully-islanded scenario,
# and any intermediate value models a degraded / partially-available
# interconnection.
# ===================================================================
def solve_constrained_slack_scenario(model_opt, max_slack_mw: float, slack_q_max_mvar: float | None = None, fixed_setpoints: dict = None):
    """Re-solve an existing Pyomo model with the slack capped to ±max_slack_mw.

    Call AFTER ``flex_engine.optimization_model_base`` has returned.
    Mutates the passed-in model:

      * Unfixes every generator P/Q variable. Fold 1 in the engine may
        have left them fixed at base values; if the slack now has less
        headroom, local generators may need to pick up the slack
        (literally), so they must be free to move.
      * Tightens the bounds on every external grid variable to
        ``(-max_slack_mw, +max_slack_mw)`` for both active and
        reactive injection. The engine's default is ±200 MW (a
        modelling artefact); we additionally clamp ``max_slack_mw``
        to the active profile's physical import/export limit.
        ``max_slack_mw == 0`` is the fully-islanded case (the external
        interface carries no power but the bus still provides the angular
        reference for the local grid).

    Physical interpretation
    A real interconnection has a finite rating. This helper lets the
    operator probe *"what if external support could only export 50 MW today?"*
    or *"what if the interface were de-rated to 20 MW for maintenance?"*.

    Parameters
    model_opt : pyomo.ConcreteModel
        Solved model returned by
        ``flex_engine.optimization_model_base``.
    max_slack_mw : float
        Bidirectional active power limit on the external-grid interface,
        in MW. Must be >= 0. Values are clamped to the active profile's
        physical limit.
    slack_q_max_mvar : float or None
        Separate reactive power limit on the external-grid interface in Mvar. If None
        (default) the reactive bound equals ``max_slack_mw``, preserving
        the original symmetric behaviour. Clamped to the same physical limit.

    Returns
    tuple
        ``(results, model)`` — same shape as
        ``optimization_model_base``. ``results`` is ``None`` if Ipopt
        itself raised an exception. Check
        ``results.solver.termination_condition`` for optimality.
    """
    from pyomo.environ import SolverFactory

    _cap_exchange(model_opt, max_slack_mw, slack_q_max_mvar, fixed_setpoints)
    try:
        results = SolverFactory('ipopt').solve(model_opt, tee=False)
    except Exception:
        return None, model_opt
    return results, model_opt


def _cap_exchange(model_opt, max_slack_mw: float, slack_q_max_mvar: float | None = None,
                  fixed_setpoints: dict | None = None) -> None:
    """The external-grid cap on a model (in place): units free, the exchange
    within ±cap, pinned set-points kept. The capped re-solve's setup, also
    used by the last resort."""
    _phys_cap = float(app_data.get("slack_max_mw", 999.0))
    bound = max(0.0, min(float(max_slack_mw), _phys_cap))
    q_bound = max(0.0, min(float(slack_q_max_mvar), _phys_cap)) if slack_q_max_mvar is not None else bound

    for g in model_opt.G:
        model_opt.Pg_new[g].unfix()
        model_opt.Qg_new[g].unfix()

    for e in model_opt.Ext:
        # `setlb`/`setub` mutate the bounds on the existing Var; safer
        # than `.fix()` because it lets the optimizer find any value
        # within the new envelope rather than nailing it to a single
        # number.
        model_opt.Pext[e].unfix()
        model_opt.Qext[e].unfix()
        model_opt.Pext[e].setlb(-bound)
        model_opt.Pext[e].setub(bound)
        model_opt.Qext[e].setlb(-q_bound)
        model_opt.Qext[e].setub(q_bound)

    # Re-apply any fixed setpoints so pinned generators/slack stay at
    # their user-specified values during the constrained re-solve.
    if fixed_setpoints:
        for _name, _mw in fixed_setpoints.items():
            if _name in model_opt.G:
                model_opt.Pg_new[_name].fix(float(_mw))
            elif _name in model_opt.Ext:
                model_opt.Pext[_name].fix(float(_mw))


# ``/api/time/advance`` which bumps ``current_index``. is the only write API point

#   timestamps     : sorted list[str] is every available simulation tick from pkl file
#   measurements   : dict[timestamp_str -> DataFrame] of smart-meter rows
#   current_index  : int — points to ``timestamps`` shows how many 15min steps have been made
#   net            : pandapowerNet — base grid
#   db_full        : dict — pre-computed Ybus for the network as loaded; an
#                    outage is this database without the outaged branch
app_data = {
    "timestamps": [],
    "measurements": {},
    "current_index": 0,
    # Forecast dataset — a parallel, read-only look-ahead series used by planning
    # and rescheduling tools. Has no simulation clock of its own (window scans
    # start at index 0). Empty until a forecast is generated or uploaded.
    "forecast_timestamps": [],
    "forecasts": {},
    # Provenance of each dataset: "synthetic" (auto-generated on network upload)
    # or "uploaded" (user-supplied CSV). Drives the overwrite-confirmation on
    # data upload — replacing user data asks first; replacing synthetic does not.
    "measurements_source": "synthetic",
    "forecasts_source": "synthetic",
    "net": None,
    "db_full": None,
    "admittance_source": "generated",  # generated | uploaded_csv
    "ipopt_available": None,  # probed once at startup; None = not yet checked
    "grid_profile": {},   # metadata exposed via GET /api/grid_constants
    "last_dispatch_result": [],
    "last_dispatch_timestamp": None,
    "default_vm_lower": 0.95,  # network-specific defaults, overwritten at load time
    "default_vm_upper": 1.05,
    # Network proposals (sandbox): {id -> {"id", "edits", "hash", ...}} and
    # their materialisations; emptied whenever the base network is replaced.
    "proposals": {},
    "_proposal_cache": {},
    "_peaks_cache": {},
}


# ---------------------------------------------------------------------------
# Network access
#
# Every analysis endpoint reads the network it studies, and that network's
# admittance database, through `_net_for` / `_db_for`. The network is the base
# the operator loaded, or a proposal — the base plus a list of edits
# (`sandbox.py`) — named by the request's `network` field. Startup, uploads,
# resets and the topology view act on the base directly; they are not studies
# of a network but the loading of one.
#
# Proposals are kept in memory, shared by every browser tab like the
# simulation clock, until they are discarded or the base network is replaced.
# A proposal is materialised (deep copy of the base, edits applied) the first
# time a study asks for it and cached with its own admittance database.
# ---------------------------------------------------------------------------
BASE_NETWORK = "base"
_DB_KEYS = {"full": "db_full"}
_PROPOSAL_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_\-]{0,47}$")


def _proposals() -> dict:
    return app_data.setdefault("proposals", {})


def _discard_proposals() -> None:
    """The base changed: every proposal was built on the old one."""
    app_data["proposals"] = {}
    app_data["_proposal_cache"] = {}
    app_data["_peaks_cache"] = {}


def _read_store(path: str) -> dict:
    """A saved file as {network fingerprint: entry}. Each network keeps its
    own entry, so moving between scenarios or networks loses nothing; a file in
    the older one-network format is read as a single entry."""
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as exc:
        print(f"[store] Could not read {path}: {exc}")
        return {}
    if isinstance(data.get("by_fingerprint"), dict):
        return dict(data["by_fingerprint"])
    if data.get("base_fingerprint"):
        fp = data.pop("base_fingerprint")
        data.pop("version", None)
        return {fp: data}
    return {}


def _write_store(path: str, store: dict) -> None:
    """Write {fingerprint: entry}; no entries left removes the file."""
    try:
        if not store:
            if os.path.exists(path):
                os.remove(path)
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump({"version": 2, "by_fingerprint": store}, fh, indent=1, default=str)
        os.replace(tmp, path)
    except OSError as exc:
        print(f"[store] Could not write {path}: {exc}")


def _save_proposals() -> None:
    """Write this network's proposals to disk under its fingerprint (other
    networks' entries are kept). Called after every change to them. A failure to
    write is reported, not raised: the proposal itself was stored and works
    until the next restart.
    """
    base = app_data.get("net")
    if base is None:
        return
    store = _read_store(_PROPOSALS_FILE)
    fp, proposals = sb.network_fingerprint(base), app_data.get("proposals") or {}
    if proposals:
        store[fp] = {"proposals": proposals}
    else:
        store.pop(fp, None)
    _write_store(_PROPOSALS_FILE, store)
    _note_where_changes_are_kept()


def _option_sets_path() -> str:
    """Kept beside the proposals (per network, like them)."""
    return os.path.join(os.path.dirname(_PROPOSALS_FILE), "option_sets.json")


def _option_to_json(option) -> dict:
    out = option._asdict()
    out["max_loading_branch"] = list(option.max_loading_branch) if option.max_loading_branch else None
    return out


def _option_from_json(raw: dict):
    """An option as saved, its bus and load keys back to numbers."""
    def by_int(d):
        return {int(k): v for k, v in d.items()} if isinstance(d, dict) else d
    fields = {k: raw.get(k) for k in fo.Option._fields}
    for key in ("v", "v0", "dr", "relief"):
        fields[key] = by_int(fields[key])
    if fields.get("vc_bus"):
        fields["vc_bus"] = {g: int(b) for g, b in fields["vc_bus"].items()}
    if fields.get("max_loading_branch"):
        fields["max_loading_branch"] = tuple(int(b) for b in fields["max_loading_branch"])
    return fo.Option(**fields)


def _save_option_sets() -> None:
    """This network's option sets to disk, so a reference ("S3:cheapest")
    still applies after a restart. A failure to write is reported, not raised.
    A study worker saves nothing: its parent keeps what it returns."""
    fp = app_data.get("loaded_fingerprint")
    if not fp or _IN_WORKER:
        return
    store = _read_store(_option_sets_path())
    sets = app_data.get("option_sets") or {}
    if sets:
        store[fp] = {"counter": app_data.get("option_set_counter", 0),
                     "sets": {sid: {**{k: v for k, v in s.items() if k != "options"},
                                    "options": {oid: _option_to_json(o) for oid, o in s["options"].items()}}
                              for sid, s in sets.items()}}
    else:
        store.pop(fp, None)
    _write_store(_option_sets_path(), store)


def _restore_option_sets() -> None:
    """This network's saved option sets, if any. One computed under other unit
    models, names or load power factor is restored too — `_resolve_option`
    refuses it with the reason, rather than it silently vanishing."""
    entry = _read_store(_option_sets_path()).get(app_data.get("loaded_fingerprint")) or {}
    try:
        sets = {sid: {**s, "options": {oid: _option_from_json(o) for oid, o in s["options"].items()}}
                for sid, s in (entry.get("sets") or {}).items()}
    except Exception as exc:  # noqa: BLE001 — a file from another version is ignored, not fatal
        print(f"[options] Saved option sets not restored ({exc}).")
        return
    app_data["option_sets"] = sets
    app_data["option_set_counter"] = max(int(entry.get("counter") or 0), app_data.get("option_set_counter", 0))


def _remember_loaded(net) -> None:
    """The network as loaded: the reference every generator model is applied
    to (generators.py), and the identity the saved assumptions are keyed to."""
    app_data["net_loaded"] = copy.deepcopy(net)
    app_data["loaded_fingerprint"] = sb.network_fingerprint(net)
    app_data["generator_model"] = {}
    app_data["unit_costs"] = {}
    app_data["element_names"] = {}
    app_data["option_sets"] = {}        # options of another network mean nothing on this one
    _restore_option_sets()              # ... but this network's, saved earlier, still do


def _apply_generator_model(model: dict, names: dict | None = None) -> list[str]:
    """Make the base the network as loaded with `model` applied, then the
    operator's names (`names`, default the current ones); refresh what is
    derived from it; re-check the proposals. Returns the proposals that no
    longer apply (dropped, with the reason). Raises gm.ModelError."""
    loaded = app_data.get("net_loaded")
    if loaded is None:
        raise gm.ModelError("No network is loaded.")
    names = (app_data.get("element_names") or {}) if names is None else names
    new = naming.apply(gm.apply(loaded, model), names)
    app_data["net"] = new
    app_data["generator_model"] = model
    app_data["element_names"] = names
    PG_MAX_DATA.clear()
    PG_MAX_DATA.update(get_pg_max_data(new))
    # The admittances do not depend on which units control voltage; rebuilt
    # anyway when they were generated here (an uploaded set is the operator's).
    if app_data.get("admittance_source") != "uploaded_csv":
        _build_and_store_ybus(new)
    app_data["_proposal_cache"] = {}
    dropped = []
    for pid, prop in list(_proposals().items()):
        try:
            net, canonical = sb.materialise(new, sb.upgrade_stored(prop["edits"]))
        except sb.EditError as exc:
            dropped.append(f"{pid}: {exc}")
            _proposals().pop(pid)
            continue
        _proposals()[pid] = {**prop, "edits": canonical, "hash": sb.edits_hash(canonical),
                             "summary": sb.summary(net, canonical)}
    if _proposals() or dropped:
        _save_proposals()
    return dropped


def _load_pf_setting() -> tuple[float, str]:
    """(the session's load power factor, "default" or "session setting")."""
    value = app_data.get("load_power_factor")
    if value is None:
        return lg.DEFAULT_LOAD_POWER_FACTOR, "default"
    return float(value), "session setting"


def _save_assumptions() -> None:
    """Keep the session's assumptions for this network across a restart; the
    default is not stored (no file means the default)."""
    value = app_data.get("load_power_factor")
    model = app_data.get("generator_model") or {}
    unit_costs = app_data.get("unit_costs") or {}
    names = app_data.get("element_names") or {}
    # Keyed to the network as loaded: the generator model changes the base's own
    # fingerprint, the network it applies to does not change.
    store = _read_store(_ASSUMPTIONS_FILE)
    fp = app_data.get("loaded_fingerprint")
    if value is None and not model and not unit_costs and not names:
        store.pop(fp, None)
    else:
        store[fp] = {"load_power_factor": None if value is None else float(value), "generator_model": model,
                     "unit_costs": unit_costs, "element_names": names}
    _write_store(_ASSUMPTIONS_FILE, store)
    _note_where_changes_are_kept()


def _forget_saved_assumptions() -> None:
    try:
        os.remove(_ASSUMPTIONS_FILE)
    except FileNotFoundError:
        pass
    except OSError as exc:
        print(f"[assumptions] Could not remove {_ASSUMPTIONS_FILE}: {exc}")


def _restore_assumptions() -> None:
    """At startup, the setting saved for this same network, if any."""
    if app_data.get("net") is None:
        return
    saved = _read_store(_ASSUMPTIONS_FILE).get(app_data.get("loaded_fingerprint"))
    if saved is None:
        return
    value = saved.get("load_power_factor")
    if isinstance(value, (int, float)) and LOAD_PF_MIN <= value <= 1.0:
        app_data["load_power_factor"] = float(value)
        print(f"[assumptions] Load power factor {value:g} restored.")
    model = saved.get("generator_model") or {}
    if model:
        try:
            # Validated again against the network as loaded, as a proposal is.
            changes = [{"unit": name, **diff} for name, diff in model.items()]
            _apply_generator_model(gm.merge({}, gm.normalise(app_data["net_loaded"], changes)))
            print(f"[assumptions] Generator model restored for {len(model)} unit(s).")
        except gm.ModelError as exc:
            print(f"[assumptions] Generator model no longer applies ({exc}) — not restored.")
    saved_names = saved.get("element_names") or {}
    if saved_names:
        try:
            changes = [{"element_type": t, "element": int(k), "name": v}
                       for t, entries in saved_names.items() for k, v in entries.items()]
            names = naming.normalise(_net_for(None), changes, {})
            _apply_generator_model(app_data.get("generator_model") or {}, names)
            print(f"[assumptions] Names restored for {len(changes)} element(s).")
        except (naming.RenameError, gm.ModelError) as exc:
            print(f"[assumptions] Saved names no longer apply ({exc}) — not restored.")
    saved_costs = saved.get("unit_costs") or {}
    if saved_costs:
        try:
            changes = [{"unit": name, **diff} for name, diff in saved_costs.items()]
            app_data["unit_costs"] = uc.merge({}, uc.normalise(app_data["net_loaded"], changes))
            print(f"[assumptions] Costs restored for {len(saved_costs)} unit(s).")
        except uc.CostError as exc:
            print(f"[assumptions] Saved costs no longer apply ({exc}) — not restored.")


def _forget_saved_proposals() -> None:
    """The operator replaced the base network: its proposals are not kept."""
    try:
        os.remove(_PROPOSALS_FILE)
    except FileNotFoundError:
        pass
    except OSError as exc:
        print(f"[proposals] Could not remove {_PROPOSALS_FILE}: {exc}")


def _restore_proposals() -> list[str]:
    """At startup, bring back the proposals saved for this network.

    Only when the loaded network is structurally the one they were built on
    (same fingerprint), and only those whose edits still apply. Returns the
    ids restored.
    """
    base = app_data.get("net")
    if base is None:
        return []
    store = _read_store(_PROPOSALS_FILE)
    saved = store.get(sb.network_fingerprint(base))
    if saved is None:
        if store:
            print("[proposals] None saved for this network.")
        return []
    restored = []
    for pid, prop in (saved.get("proposals") or {}).items():
        try:
            net, canonical = sb.materialise(base, sb.upgrade_stored(prop["edits"]))
        except (sb.EditError, KeyError, TypeError) as exc:
            print(f"[proposals] {pid!r} no longer applies ({exc}) — dropped.")
            continue
        _proposals()[pid] = {**prop, "edits": canonical, "hash": sb.edits_hash(canonical),
                             "summary": sb.summary(net, canonical)}
        restored.append(pid)
    if restored:
        print(f"[proposals] Restored {len(restored)}: {', '.join(restored)}")
    return restored


def _network_name(request=None) -> str:
    """The network a request names: the base, or an existing proposal."""
    raw = getattr(request, "network", None) if request is not None else None
    name = (str(raw).strip() if raw else "") or BASE_NETWORK
    if name != BASE_NETWORK and name not in _proposals():
        known = sorted(_proposals())
        raise HTTPException(
            status_code=400,
            detail=f"Unknown network {name!r}. Use 'base'"
                   + (f" or one of the proposals {known}." if known else
                      " — there are no proposals; create one with edit_proposal."),
        )
    return name


def _materialised(name: str) -> dict:
    """The cached materialisation of proposal `name`, built on first use."""
    base = app_data.get("net")
    if base is None:
        raise HTTPException(status_code=503, detail="No network loaded yet.")
    proposal = _proposals().get(name)
    if proposal is None:
        # Discarding is not serialised with the studies (the sidebar must not
        # wait behind a long optimisation), so a proposal can go mid-study.
        raise HTTPException(status_code=409, detail=f"Proposal {name!r} was discarded during the study.")
    cache = app_data.setdefault("_proposal_cache", {})
    key = (id(base), proposal["hash"])
    entry = cache.get(name)
    if entry is None or entry["key"] != key:
        try:
            net, _ = sb.materialise(base, sb.upgrade_stored(proposal["edits"]))
        except sb.EditError as exc:
            raise HTTPException(status_code=409, detail=f"Proposal {name!r} no longer applies: {exc}")
        entry = {"key": key, "net": net, "db_full": None, "db_error": None}
        cache[name] = entry
    return entry


def _net_for(request=None):
    """The network a request studies.

    The shared object: deep-copy it before changing anything.
    """
    name = _network_name(request)
    if name != BASE_NETWORK:
        return _materialised(name)["net"]
    net = app_data.get("net")
    if net is None:
        raise HTTPException(status_code=503, detail="No network loaded yet.")
    return net


def _db_for(kind: str = "full", request=None):
    """That network's admittance database ("full"), or None while it has not
    been built. Outages need no database of their own (see build_ybus_full)."""
    key = _DB_KEYS[kind]
    name = _network_name(request)
    if name == BASE_NETWORK:
        return app_data.get(key)
    if app_data.get("admittance_source") == "uploaded_csv":
        # The base's admittances came from the operator's files; a proposal's
        # would come from pandapower — the comparison would mix two models.
        raise HTTPException(
            status_code=409,
            detail="The base network's admittances were uploaded as CSV files, so a "
                   "proposal cannot be optimised on a consistent model. Assessments "
                   "(power flows) of proposals still work.",
        )
    entry = _materialised(name)
    if entry["db_full"] is None and entry["db_error"] is None:
        try:
            entry["db_full"] = nl.build_ybus_full(entry["net"])
        except Exception as exc:
            entry["db_error"] = str(exc)
    if entry["db_error"]:
        raise HTTPException(status_code=422, detail=f"Proposal {name!r}: {entry['db_error']}")
    return entry["db_full"]


def _pg_max_for(request=None, net=None) -> dict:
    """{generator name -> Pmax} of the request's network.

    PG_MAX_DATA on the base. A proposal adds its own units, read from `net`
    (the network at the tick, where a pv or wind unit's maximum is what the
    resource gives) or, without one, from the proposal as built.
    """
    if _network_name(request) == BASE_NETWORK:
        return PG_MAX_DATA
    out = dict(PG_MAX_DATA)
    src = _net_for(request) if net is None else net
    named = "substation_name" if "substation_name" in src.sgen.columns else "name"
    for idx in src.sgen.index[lg._proposed_mask(src.sgen)]:
        name = src.sgen.at[idx, named]
        if not isinstance(name, str):
            name = src.sgen.at[idx, "name"]
        out[str(name)] = float(src.sgen.at[idx, "max_p_mw"])
    # A unit a proposal added on voltage control is a `gen` row.
    for idx in src.gen.index[lg._proposed_mask(src.gen)]:
        out[str(src.gen.at[idx, "name"])] = float(src.gen.at[idx, "max_p_mw"])
    return out


def _dataset_peaks(dataset: dict) -> dict:
    """`sandbox.dataset_peaks` of a dataset, computed once per dataset."""
    cache = app_data.setdefault("_peaks_cache", {})
    ts = list(dataset)
    key = (id(dataset), len(ts), ts[0] if ts else None, ts[-1] if ts else None)
    if key not in cache:
        cache.clear()  # one dataset at a time is enough; uploads replace it
        cache[key] = sb.dataset_peaks(dataset)
    return cache[key]


def _dataset_of(request) -> dict:
    """The dataset a request reads (measurements or forecasts)."""
    return _resolve_dataset(getattr(request, "data_source", None) or "measurements")[1]


def _scale_loads(net, factor) -> None:
    """Multiply demand by `factor`: a load-stress scenario, or one draw of
    demand uncertainty (probabilistic RSA, robust OPF, hosting capacity).

    A flat load a network proposal added (a data centre at a fixed size) is
    left as it is: it is a stated quantity, not a measured demand that could
    be higher or uncertain. An added load that follows its bus's demand
    (`follow_bus`) scales with it, as the demand it follows does.
    """
    f = float(factor)
    if f == 1.0 or not len(net.load):
        return
    rows = net.load.index[~sb.fixed_load_mask(net.load)]
    net.load.loc[rows, "p_mw"] *= f
    net.load.loc[rows, "q_mvar"] *= f


def _assign_tick(net, meas, ts: str, dataset: dict | None = None, subs: list | None = None):
    """Give `net` the values of one tick: measured loads and generation, then
    every element a proposal added at its own profile.

    `dataset` is the series the tick comes from (its peaks scale the
    `follow_bus`, `pv` and `wind` profiles); only needed when `net` has added
    elements. Returns (net, report) — the report lists each added element and
    what it injects at this tick, empty on the base.
    """
    subs = substations if subs is None else subs
    # Moved loads and units back on their measured bus, scaled ones unscaled:
    # the assignment below then gives them their own values.
    sb.before_tick(net)
    if ts == nom.NOMINAL:
        # The case's own values, per element and exactly — not spread per
        # substation at a 0.95 power factor as a tick is.
        try:
            net = nom.apply(net, _nominal_snapshot())
        except nom.NominalError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
    else:
        own = getattr(_study, "load_pf", None)
        pf = own if own is not None else _load_pf_setting()[0]
        net, _ = lg.assign_load_values_from_measurements(net, meas, subs, power_factor=pf)
        net, _ = lg.assign_generators_values_from_measurements(net, meas, subs)
        _study.pf_used = True
        share = net.get("_q_measured_share", 0.0) if hasattr(net, "get") else 0.0
        if share:
            seen = getattr(_study, "q_share", None) or (share, share)
            _study.q_share = (min(seen[0], share), max(seen[1], share))
    # What a proposal changed on existing loads and units (set_parameter,
    # scale, move_element) wins over the measurements, at every tick.
    sb.apply_overrides(net)
    # Units the operator set to a fixed power factor: Q for this tick's P.
    gm.set_reactive(net)
    if not (lg._proposed_mask(net.load).any() or lg._proposed_mask(net.sgen).any()
            or lg._proposed_mask(net.gen).any()):
        return net, []
    if dataset is None:
        raise HTTPException(status_code=500, detail="Internal: a proposal tick needs its dataset.")
    try:
        report = sb.apply_profiles(net, ts, meas, _dataset_peaks(dataset))
    except sb.EditError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    wrong = sb.check_profiles(net, report)
    if wrong:
        raise HTTPException(status_code=500, detail="Added elements were overwritten: " + "; ".join(wrong))
    if getattr(_tick_report, "first", "unset") is None:
        _tick_report.first = {"timestamp": ts, "elements": report}
    return net, report


def _proposal_card(name: str) -> dict:
    """What a result on proposal `name` carries about it."""
    proposal = _proposals().get(name)
    if proposal is None:
        raise HTTPException(status_code=409, detail=f"Proposal {name!r} was discarded during the study.")
    return {"id": name, "edits_hash": proposal["hash"],
            "base": "the loaded network, unchanged",
            "edits": list(proposal["summary"]["edits"])}


def _resolve_dataset(data_source: str = "measurements"):
    """Resolve a tool's requested data source to a (timestamps, table, label) tuple.

    ``data_source`` is the LLM-facing selector:
      - ``"measurements"`` (default) → historical actuals driving the sim clock.
      - ``"forecasts"`` → the look-ahead planning series.

    Returns ``(timestamps_list, measurements_dict, label)``.

    Raises ``HTTPException(400)`` if forecasts are requested but none are loaded,
    so the LLM gets a clear, actionable error rather than a silent fallback.
    """
    src = (data_source or "measurements").strip().lower()
    if src == nom.NOMINAL:
        return [nom.NOMINAL], {nom.NOMINAL: _nominal_measurements()}, nom.NOMINAL
    if src in ("forecast", "forecasts"):
        ts = app_data.get("forecast_timestamps") or []
        if not ts:
            raise HTTPException(
                status_code=400,
                detail="No forecast data is loaded. Upload a forecast dataset, or "
                       "call this tool with data_source='measurements'.",
            )
        return ts, app_data["forecasts"], "forecasts"
    if src not in ("measurement", "measurements"):
        raise HTTPException(
            status_code=400,
            detail=f"Unknown data_source {data_source!r}. Use 'measurements', 'forecasts' or 'nominal'.",
        )
    return app_data["timestamps"], app_data["measurements"], "measurements"


def _nominal_snapshot() -> dict:
    """The loaded network's nominal point, or a 422 saying why it cannot be used."""
    snap = app_data.get("nominal")
    ok, why = nom.usable(snap)
    if not ok:
        raise HTTPException(status_code=422, detail=f"The nominal operating point is not available. {why}")
    return snap


def _nominal_measurements() -> pd.DataFrame:
    """The nominal point in the measurement schema (per substation), so code
    reading a tick's table works on it. The study itself uses the exact
    per-element values (`_assign_tick`)."""
    snap = _nominal_snapshot()
    net = _net_for(None)          # the base: rows a proposal adds are not in the snapshot
    names = list(substations)

    def key(bus: int) -> str:
        return sb.measurement_key_for_bus(net, bus, names) or measurement_name(net, bus)

    return nom.as_measurements(net, snap, key)


def _window_dataset(data_source: str = "measurements"):
    """`_resolve_dataset` for studies over a time window: the nominal point is
    one moment with no time, so they refuse it with the reason."""
    if (data_source or "").strip().lower() == nom.NOMINAL:
        raise HTTPException(
            status_code=400,
            detail="The nominal operating point is a single moment with no time, so a study over a "
                   "time window cannot use it. Use data_source='measurements' or 'forecasts' here, "
                   "or a point study (security assessment, N-1, OPF, snapshot…) with "
                   "data_source='nominal'.")
    return _resolve_dataset(data_source)


def _dispatch_rows_from_overrides(dispatch_overrides):
    """Normalize custom dispatch overrides into activated_resources-like rows.

    Supported input forms:
    - {"GenName": 3.2}                              # P only
    - {"GenName": {"p_mw": 3.2, "q_mvar": 0.4}} # P/Q explicit
    - {"GenName": {"Pg_new": 3.2, "Qg_new": 0.4}}
    """
    rows = []
    if not isinstance(dispatch_overrides, dict):
        return rows

    for elem, spec in dispatch_overrides.items():
        row = {"element": str(elem)}
        p_new = None
        q_new = None
        if isinstance(spec, dict):
            p_new = spec.get("p_mw", spec.get("Pg_new", spec.get("p")))
            q_new = spec.get("q_mvar", spec.get("Qg_new", spec.get("q")))
        else:
            p_new = spec

        if p_new is not None:
            try:
                row["Pg_new"] = float(p_new)
            except Exception:
                pass
        if q_new is not None:
            try:
                row["Qg_new"] = float(q_new)
            except Exception:
                pass

        if "Pg_new" in row or "Qg_new" in row:
            rows.append(row)

    return rows


def _voltage_unit_as_injection(net, requested: str) -> str | None:
    """If `requested` names a voltage-controlled unit (net.gen) and no static
    generator, replace it in `net` (solved) by a static generator injecting
    what it injects now; returns a note saying so, else None."""
    if net.gen.empty or not str(requested or "").strip():
        return None
    want = str(requested).strip().casefold()
    sgen_names = set(net.sgen["name"].astype(str).str.casefold()) if len(net.sgen) else set()
    if want in sgen_names:
        return None
    for idx, row in net.gen.iterrows():
        name = str(row.get("name") or "").strip() or f"Gen_bus{int(row['bus'])}"
        if want not in (name.casefold(), f"gen_bus{int(row['bus'])}"):
            continue
        p, q = float(net.res_gen.at[idx, "p_mw"]), float(net.res_gen.at[idx, "q_mvar"])
        pp.create_sgen(net, bus=int(row["bus"]), p_mw=p, q_mvar=q, name=name,
                       max_p_mw=row.get("max_p_mw"), min_p_mw=row.get("min_p_mw"), controllable=True)
        net.gen.drop(index=idx, inplace=True)
        pp.runpp(net, algorithm="nr", numba=False)
        return (f"{name} holds a voltage set-point in operation; for this envelope it is an injection of "
                f"active and reactive power, from where it runs now ({p:.2f} MW, {q:.2f} Mvar).")
    return None


def _apply_dispatch_to_net(net, dispatch_rows):
    """Apply dispatch rows (element, Pg_new, Qg_new) to sgen/gen/ext_grid
    setpoints. A voltage-controlled unit (net.gen) takes the active power; its
    reactive power follows its set-point.

    Returns a summary dict with applied/skipped element names.
    """
    if not dispatch_rows:
        return {"applied": 0, "total": 0, "applied_elements": [], "skipped_elements": []}

    def _norm_name(v):
        return str(v).strip().casefold()

    sgen_lookup = {}
    if hasattr(net, "sgen") and not net.sgen.empty:
        for idx, row in net.sgen.iterrows():
            labels = []
            for col in ("substation_name", "name"):
                if col in net.sgen.columns:
                    raw = row.get(col, None)
                    if pd.notna(raw) and str(raw).strip():
                        labels.append(str(raw).strip())
            bus_val = row.get("bus", None)
            try:
                b = int(bus_val)
                labels.extend([f"Bus {b}", f"Gen_bus{b}"])
            except Exception:
                pass
            for label in labels:
                sgen_lookup[_norm_name(label)] = int(idx)

    gen_lookup = {}
    if hasattr(net, "gen") and not net.gen.empty:
        for idx, row in net.gen.iterrows():
            raw = row.get("name", None)
            labels = [str(raw).strip()] if pd.notna(raw) and str(raw).strip() else []
            labels.append(f"Gen_bus{int(row['bus'])}")
            for label in labels:
                gen_lookup.setdefault(_norm_name(label), int(idx))

    ext_lookup = {}
    if hasattr(net, "ext_grid") and not net.ext_grid.empty:
        for idx, row in net.ext_grid.iterrows():
            labels = ["ExtGrid_0", "Slack", "External Grid"]
            raw_name = row.get("name", None)
            if pd.notna(raw_name) and str(raw_name).strip():
                labels.append(str(raw_name).strip())
            for label in labels:
                ext_lookup[_norm_name(label)] = int(idx)

    applied = []
    skipped = []
    for row in dispatch_rows:
        elem = str(row.get("element", "")).strip()
        if not elem:
            continue
        key = _norm_name(elem)
        p_new = row.get("Pg_new", None)
        q_new = row.get("Qg_new", None)

        if key in sgen_lookup:
            sidx = sgen_lookup[key]
            try:
                if p_new is not None:
                    net.sgen.at[sidx, "p_mw"] = float(p_new)
                if q_new is not None and "q_mvar" in net.sgen.columns:
                    net.sgen.at[sidx, "q_mvar"] = float(q_new)
                applied.append(elem)
                continue
            except Exception:
                skipped.append(elem)
                continue

        if key in gen_lookup and p_new is not None:
            net.gen.at[gen_lookup[key], "p_mw"] = float(p_new)
            applied.append(elem)
            continue

        if key in ext_lookup:
            eidx = ext_lookup[key]
            try:
                if p_new is not None:
                    net.ext_grid.at[eidx, "p_mw"] = float(p_new)
                if q_new is not None and "q_mvar" in net.ext_grid.columns:
                    net.ext_grid.at[eidx, "q_mvar"] = float(q_new)
                applied.append(elem)
                continue
            except Exception:
                skipped.append(elem)
                continue

        skipped.append(elem)

    return {
        "applied": len(applied),
        "total": len(dispatch_rows),
        "applied_elements": applied,
        "skipped_elements": skipped,
    }


class AdvanceTimeRequest(BaseModel):
    """Request body for ``POST /api/time/advance``.

    Attributes
    target_timestamp : str | None
        If provided, jump the simulation clock directly to this timestamp
        instead of stepping forward by one tick. Accepts full timestamp
        strings (e.g. ``"2022-03-15 08:00"`` or ``"2022-03-15 08:00:00"``).
        Prefix matching is supported so partial strings resolve to the
        first matching tick. If ``None`` (default), advances by one tick.
    """
    target_timestamp: str | None = None


# Generic limits, used only when no network is loaded to supply its own.
_STATIC_VM_LOWER = 0.95
_STATIC_VM_UPPER = 1.05
_STATIC_MAX_LOADING = 90.0


def _network_limits() -> dict[str, float]:
    """The loaded network's own security limits.

    Also the default voltage band inside the OPF (`opf_vm_*`): a corrective
    dispatch is judged against the same limits the assessment flagged it by,
    unless the caller asks for different ones. The OPF used to default to a
    generic 0.95–1.05 whatever the network's band.
    """
    gp = app_data.get("grid_profile") or {}
    loading = float(gp.get("max_loading_pct", 100.0) or 100.0)
    vlo = float(app_data.get("default_vm_lower", _STATIC_VM_LOWER))
    vhi = float(app_data.get("default_vm_upper", _STATIC_VM_UPPER))
    return {
        "vm_lower_pu": vlo,
        "vm_upper_pu": vhi,
        "max_line_loading_pct": loading,
        "max_trafo_loading_pct": loading,
        "opf_vm_lower": vlo,
        "opf_vm_upper": vhi,
    }


def _fill_network_limits(request) -> None:
    """Give every limit field the caller did not send the network's own value.

    Called from `_NetworkLimitsMixin` on every request model. An explicit
    value always wins — including one equal to a generic default.
    """
    sent = request.model_fields_set
    for field, limit in _network_limits().items():
        # (`opf_vm_*` exist only on the OPF requests; the check below skips
        # every field a request does not declare.)
        if field in type(request).model_fields and field not in sent:
            setattr(request, field, limit)


LOAD_PF_MIN = 0.5


class _NetworkLimitsMixin(BaseModel):
    """Fills every security limit the caller did not send with the network's own.

    Request models declare `vm_upper_pu`, `vm_lower_pu`, `max_line_loading_pct`
    and `max_trafo_loading_pct` with generic defaults. Those defaults used to
    double as "not specified": `_effective_thresholds` replaced any value equal
    to 0.95 / 1.05 / 90 with the network's limits, so an operator who *asked*
    for exactly 0.95–1.05 p.u. or 90 % was silently given 0.94–1.06 / 100 on
    the bundled networks. And endpoints that never called it (N-1, envelope,
    hosting capacity, robust OPF) judged the same state against different
    limits from the RSA.

    Whether a field was sent is now what decides, read from pydantic's own
    record of the fields set — one rule, applied to every request that carries
    limits, wherever it is built.
    """

    # Power factor (lagging) giving each measured load its reactive power, for
    # this study only; None = the session setting (`/api/assumptions`).
    load_power_factor: float | None = None

    @field_validator("load_power_factor")
    @classmethod
    def _valid_power_factor(cls, value):
        if value is not None and not (LOAD_PF_MIN <= value <= 1.0):
            raise ValueError(f"load_power_factor must be between {LOAD_PF_MIN} and 1.0 (lagging).")
        return value

    @model_validator(mode="after")
    def _fill_unset_limits(self):
        _fill_network_limits(self)
        return self


class _TimeseriesRequest(_NetworkLimitsMixin):
    """Mixin for every tool that reads the timeseries.

    ``data_source`` lets the LLM choose which dataset the tool operates on:
      - ``"measurements"`` (default): historical actuals; point tools default to
        the live simulation clock.
      - ``"forecasts"``: read-only look-ahead series; point tools default to the
        first forecast tick.
    ``timestamp`` optionally targets one tick within the chosen source (ISO
    prefix matching, e.g. ``'2000-01-09 18'``). When omitted the default above
    applies. The two series may share timestamps (a forecast of measured days).
    ``network`` names the grid studied: ``"base"`` (default) or a proposal id.
    """
    data_source: str = "measurements"
    timestamp: str | None = None
    network: str | None = None


class GridRequest(_TimeseriesRequest):
    """Request body for ``POST /api/grid/rsa``.

    Attributes
    load_scaling_factor : float
        Multiplicative stress factor applied to every load's P and Q
        before the power flow runs. 1.0 = nominal, 2.0 = doubled
        demand. Driven by the Global Load Stress slider in the UI.
    vm_upper_pu : float
        Upper voltage limit in p.u. Default 1.05.
    vm_lower_pu : float
        Lower voltage limit in p.u. Default 0.95.
    max_line_loading_pct : float
        Thermal overload threshold for lines in %. Default 90.
    max_trafo_loading_pct : float
        Thermal overload threshold for transformers in %. Default 90.
    """
    load_scaling_factor: float = 1.0
    sgen_scaling_factor: float = 1.0
    vm_upper_pu: float = 1.05
    vm_lower_pu: float = 0.95
    max_line_loading_pct: float = 90.0
    max_trafo_loading_pct: float = 90.0
    # A corrective option's reference ("S3:cheapest"): the study runs with its
    # dispatch applied, at its moment or a later one (its set-points held).
    option: str | None = None
    # island_mode: bool = False was meant to be added for testing without the swedish slack bus as a contingency event


class WorstCaseRequest(_NetworkLimitsMixin):
    """Request body for ``POST /api/rsa/worst_case``.

    Attributes
    ----------
    metric : str
        The stress metric to find the worst timestamp for.
        One of: 'violations', 'slack_import', 'max_voltage', 'min_voltage', 'max_loading'.
    n_steps : int | None
        Number of ticks to scan from the current index. None = scan all remaining.
    step_size : int
        Scan every Nth tick for speed. Default 1 (every tick).
    vm_upper_pu / vm_lower_pu : float
        Voltage violation thresholds in p.u.
    max_line_loading_pct / max_trafo_loading_pct : float
        Thermal overload thresholds.
    """
    metric: str = "violations"
    n_steps: int | None = None
    step_size: int = 1
    data_source: str = "measurements"  # "measurements" (live actuals) or "forecasts"
    network: str | None = None         # "base" or a proposal id
    vm_upper_pu: float = 1.05
    vm_lower_pu: float = 0.95
    max_line_loading_pct: float = 90.0
    max_trafo_loading_pct: float = 90.0


class ContingencyRequest(_TimeseriesRequest):
    """Request body for ``POST /api/contingency/simulate`` (no optimizer).

    Attributes
    element_type : str
        Either ``"line"`` or ``"trafo"``. Anything else will return HTTP 400.
    element_index : int
        Row index in ``net.line`` or ``net.trafo`` to take out of service.
    load_scaling_factor : float
        Same semantics as :class:`GridRequest`.
    vm_upper_pu : float
        Upper voltage limit in p.u. Default 1.05.
    vm_lower_pu : float
        Lower voltage limit in p.u. Default 0.95.
    max_line_loading_pct : float
        Thermal overload threshold for lines in %. Default 90.
    max_trafo_loading_pct : float
        Thermal overload threshold for transformers in %. Default 90.
    """
    element_type: str
    element_index: int
    load_scaling_factor: float = 1.0
    vm_upper_pu: float = 1.05
    vm_lower_pu: float = 0.95
    max_line_loading_pct: float = 90.0
    max_trafo_loading_pct: float = 90.0
    # A corrective option's reference: the outage studied with its dispatch applied.
    option: str | None = None
    # island_mode: bool = False was meant to be added without the swedish slack bus as a contingency event


class ContingencySweepRequest(GridRequest):
    """Request body for ``POST /api/contingency/simulate_all``. ``option``: a
    corrective option's reference from an options result ("S3:cheapest") —
    the sweep then runs at that option's network and load level, with its
    dispatch applied (and, for an option after an outage, that outage), at
    its moment or a later one (its set-points held)."""


class ContingencyOptimizeRequest(_TimeseriesRequest):
    """Request body for ``POST /api/contingency/optimize``.

    Same fields as :class:`ContingencyRequest` but this endpoint additionally
    runs topological surgery + Pyomo to fix the resulting violations.
    """
    element_type: str
    element_index: int
    load_scaling_factor: float = 1.0
    opf_vm_upper: float = 1.05
    opf_vm_lower: float = 0.95
    opf_lambda_p: float = 0.01
    opf_lambda_q: float = 0.001
    pg_max_overrides: dict[str, float] = {}
    pg_min_overrides: dict[str, float] = {}
    fixed_setpoints: dict[str, float] = {}
    opf_min_power_factor: float = 0.95
    opf_current_safety_margin: float = 0.9
    # As for the flexibility OPF: the operator's conditions, and several
    # corrective options (or near-optimal alternatives) after the outage.
    hold_units: list[str] = []
    options: bool = False
    near_optimal: bool = False
    near_optimal_pct: float = Field(default=10.0, gt=0, le=100)
    near_optimal_methods: list[Literal["hsj", "unit_ranges", "objectives", "random"]] = []
    near_optimal_count: int = Field(default=5, ge=1, le=5)
    branch_limits: dict[str, float] = {}


class FlexibilityRequest(_TimeseriesRequest):
    """Request body for ``POST /api/flexibility/optimize``.

    Attributes
    ----------
    load_scaling_factor : float
        Load multiplier. Defaults to 1.0 (actual current load).
    disabled_generators : list[str]
        Substation names to remove from the sgen table before solving.
        The UI currently forwards at most one entry.
    slack_max_mw : float
        Maximum bidirectional capacity (MW) of the external-grid
        interface. Same semantics and clamp as :class:`KPIRequest`.
        Drives the same global slider as the KPI tab so the dispatch
        reported here is consistent with the KPI numbers.
    opf_vm_upper : float
        Voltage upper bound enforced inside the OPF (p.u.). Default 1.05.
    opf_vm_lower : float
        Voltage lower bound enforced inside the OPF (p.u.). Default 0.95.
    opf_lambda_p : float
        Weight on active power deviation squared in the OPF objective. Default 0.01.
    opf_lambda_q : float
        Weight on reactive power deviation squared in the OPF objective. Default 0.001.
    pg_max_overrides : dict[str, float]
        Per-substation active power upper bound overrides (MW). Keys are substation names.
        Merged on top of the hardcoded Pg_max_data defaults inside the OPF engine.
    pg_min_overrides : dict[str, float]
        Per-substation active power lower bound overrides (MW). Negative = allow curtailment.
        Use positive values for must-run constraints.
    """
    load_scaling_factor: float = 1.0
    disabled_generators: list[str] = []
    slack_max_mw: float | None = None
    opf_vm_upper: float = 1.05
    opf_vm_lower: float = 0.95
    opf_lambda_p: float = 0.01
    opf_lambda_q: float = 0.001
    pg_max_overrides: dict[str, float] = {}
    pg_min_overrides: dict[str, float] = {}
    fixed_setpoints: dict[str, float] = {}
    opf_min_power_factor: float = 0.95
    slack_q_max_mvar: float | None = None
    opf_current_safety_margin: float = 0.9
    vm_upper_per_bus: dict[str, float] = {}
    vm_lower_per_bus: dict[str, float] = {}
    # Units that keep their active power at the schedule (the operator's
    # conditions: "without Gen 2", "don't call the battery"); pinned at their
    # MW at this moment, in the one dispatch and in every option.
    hold_units: list[str] = []
    # Lines or transformers held below a share of their rating, % ("L30": 80):
    # to avoid violations on chosen branches, or to keep headroom on one that
    # overloads after outages. Replaces the safety margin on those branches.
    branch_limits: dict[str, float] = {}
    # With options: instead of the named strategies, the cheapest option and
    # near-optimal alternatives (Hop-Skip-Jump), each within `near_optimal_pct`
    # of its cost and changing other units.
    near_optimal: bool = False
    near_optimal_pct: float = Field(default=10.0, gt=0, le=100)
    # An N-1 security study (scopf.py): the dispatch that stays within limits
    # after each critical outage — "preventive" (no action needed), "corrective"
    # (an action within each unit's reach exists), or "both" — compared with the
    # least-disturbance and cheapest options by their N-1 screens.
    n1: Literal["preventive", "corrective", "both"] | None = None
    n1_outages: list[str] = []                       # outages to study; default: the critical ones
    # Loading allowed after an outage; None = the network's own limit, the one
    # every N-1 screen uses (100 % on the bundled networks), so they agree.
    n1_emergency_loading_pct: float | None = Field(default=None, ge=50, le=150)
    n1_reach_pct: float = Field(default=20.0, gt=0, le=100)
    # After an outage, corrective actions may also reduce load — demand
    # response, then load relief — when redispatch alone cannot secure it.
    n1_load_reduction: bool = True
    # Near-optimal: which MGA methods fill the pool (default: all), and how
    # many alternatives to show.
    near_optimal_methods: list[Literal["hsj", "unit_ranges", "objectives", "random"]] = []
    near_optimal_count: int = Field(default=5, ge=1, le=5)
    # True: several corrective dispatches (flex_options.py), each the solution
    # of a strategy, scored the same way — instead of the one least-disturbance
    # dispatch. The default dispatch is unchanged either way.
    options: bool = False
    # Start from a corrective option ("S3:cheapest") instead of the schedule:
    # its set-points held to this moment (its own or a later one) are what the
    # corrective action changes — "we took the cheapest at 20:00; what now at 21:00?"
    from_option: str | None = None


class _KPIStudy(BaseModel):
    """What a KPI study takes at one moment and over a window (`kpis.py`):
    the corrective options' conditions, and what to compare the schedule with."""
    slack_max_mw: float | None = None
    slack_q_max_mvar: float | None = None
    opf_lambda_p: float = 0.01
    opf_lambda_q: float = 0.001
    pg_max_overrides: dict[str, float] = {}
    pg_min_overrides: dict[str, float] = {}
    fixed_setpoints: dict[str, float] = {}
    opf_min_power_factor: float = 0.95
    opf_current_safety_margin: float = 0.9
    hold_units: list[str] = []
    branch_limits: dict[str, float] = {}
    # A corrective option to compare with the schedule, by its ref ("S3:cheapest"),
    # held from its own moment at a later one; default: the cheapest at each moment.
    option: str | None = None
    # Also the cost of N-1 security: a security-constrained OPF at each moment (slower).
    n1: Literal["preventive", "corrective", "both"] | None = None


class KPIRequest(_KPIStudy, _TimeseriesRequest):
    """Request body for ``POST /api/kpi/evaluate``: the indicators at one
    moment, the schedule against a corrective action. The `vm_*_pu` and
    `max_*_loading_pct` limits judge each state's power flow (the network's own
    unless sent); `opf_vm_*` bound the corrective options' OPF."""
    load_scaling_factor: float = 1.0
    vm_upper_pu: float = 1.05
    vm_lower_pu: float = 0.95
    max_line_loading_pct: float = 90.0
    max_trafo_loading_pct: float = 90.0
    opf_vm_upper: float | None = None
    opf_vm_lower: float | None = None


class ElementTimeseriesRequest(_TimeseriesRequest):
    """Request body for ``POST /api/grid/element_timeseries``.

    Attributes
    ----------
    element_type : str
        One of ``'bus'``, ``'line'``, or ``'trafo'``.
    element_name : str
        Partial or exact element name. Case-insensitive substring match
        against the ``name`` column of the corresponding pandapower table.
        For buses use substation name fragments (e.g. ``'Alpha'``).
        Integer string accepted as fallback index.
    start_timestamp : str | None
        Prefix of the first timestamp to include (e.g. ``'2022-01-03'``).
        If ``None``, starts at the current simulation index.
    end_timestamp : str | None
        Prefix of the last timestamp to include (inclusive).
        If ``None``, extends to the end of the dataset.
    n_steps : int | None
        Cap on the number of ticks to scan. Applied after resolving the
        start index.
    step_size : int
        Scan every Nth tick. Default 1 (every tick). Use 4 for hourly.
    load_scaling_factor : float
        Multiplicative load multiplier. Default 1.0.
    vm_upper_pu / vm_lower_pu : float
        Voltage violation thresholds stored in the response for the renderer.
    max_line_loading_pct / max_trafo_loading_pct : float
        Thermal thresholds stored in the response for the renderer.
    """
    element_type: str
    element_name: str
    start_timestamp: str | None = None
    end_timestamp: str | None = None
    n_steps: int | None = None
    step_size: int = 1
    load_scaling_factor: float = 1.0
    vm_upper_pu: float = 1.05
    vm_lower_pu: float = 0.95
    max_line_loading_pct: float = 90.0
    max_trafo_loading_pct: float = 90.0


class ScanScenariosRequest(_TimeseriesRequest):
    """Request body for ``POST /api/rsa/scan_scenarios``.

    Attributes
    ----------
    sgen_scales : list[float]
        Renewable output scaling factors to compare. Default is
        ``[0.7, 1.0, 1.3]`` (P10 / P50 / P90 proxy). Each value
        multiplies all sgen ``p_mw`` and ``q_mvar`` columns before
        ``pp.runpp`` is called. 1.0 = measured nominal output.
    start_timestamp / end_timestamp : str | None
        ISO prefix for time window endpoints. Prefix matching supported.
    n_steps : int | None
        Cap on the number of ticks to scan per scenario.
    step_size : int
        Sample every Nth tick. Default 1.
    load_scaling_factor : float
        Multiplicative load multiplier applied in every scenario.
    vm_upper_pu / vm_lower_pu : float
        Voltage violation thresholds. Default 1.05 / 0.95.
    max_line_loading_pct / max_trafo_loading_pct : float
        Thermal overload thresholds. Default 90.0.
    """
    sgen_scales: list[float] = [0.7, 1.0, 1.3]
    start_timestamp: str | None = None
    end_timestamp: str | None = None
    n_steps: int | None = None
    step_size: int = 1
    load_scaling_factor: float = 1.0
    vm_upper_pu: float = 1.05
    vm_lower_pu: float = 0.95
    max_line_loading_pct: float = 90.0
    max_trafo_loading_pct: float = 90.0


class ProbabilisticRSARequest(_TimeseriesRequest):
    """Request body for ``POST /api/rsa/probabilistic``.

    Attributes
    ----------
    n_samples : int
        Number of Latin Hypercube samples to draw. Default 200.
    sgen_sigma : float
        Relative std deviation for renewable forecast error around the current
        sgen estimate. Default 0.00 (disabled unless explicitly requested).
        Uncertainty is applied to available
        renewable resource (with capacity clipping and setpoint ceiling), not
        directly to commanded setpoints.
    load_sigma : float
        Relative std deviation for the load multiplier distribution. Default 0.05 (±5%).
    vm_upper_pu / vm_lower_pu : float
        Voltage violation thresholds. Default 1.05 / 0.95.
    max_line_loading_pct / max_trafo_loading_pct : float
        Thermal overload thresholds. Default 90.0.
    """
    n_samples: int = 200
    sgen_sigma: float = 0.0
    load_sigma: float = 0.05
    vm_upper_pu: float = 1.05
    vm_lower_pu: float = 0.95
    max_line_loading_pct: float = 90.0
    max_trafo_loading_pct: float = 90.0


class HistoricalRiskRequest(_TimeseriesRequest):
    """Request body for ``POST /api/rsa/historical_risk``.

    First implementation slice focuses on empirical exceedance / near-miss
    statistics over a historical window, with conditional bins deferred.
    """
    target: str = "all"
    window_start: str | None = None
    window_end: str | None = None
    condition: str | None = None
    n_bins: int = 4
    near_miss_band: float = 0.005
    worst_n: int = 5
    load_scaling_factor: float = 1.0
    vm_upper_pu: float = 1.05
    vm_lower_pu: float = 0.95
    max_line_loading_pct: float = 90.0
    max_trafo_loading_pct: float = 90.0
    parallel: bool = False
    max_workers: int | None = None


class FlexibilityEnvelopeRequest(_TimeseriesRequest):
    """Request body for ``POST /api/flexibility/envelope``.

    Sweeps a target generator over a rectangular (P, Q) grid while holding
    all other generators at their current setpoints. Returns a feasibility
    map: for each (P, Q) point, whether voltages and thermal loading stay
    within limits.

    Attributes
    ----------
    gen_name : str
        Substation name of the generator to sweep (e.g. ``"Echo"``).
    p_min_mw / p_max_mw : float | None
        Active power sweep range. Defaults to ``[0.0, PG_MAX_DATA[gen_name]]``.
    q_min_mvar / q_max_mvar : float | None
        Reactive power sweep range. Defaults to ``±Pmax × tan(acos(0.9))``.
    resolution : int
        Grid resolution: ``resolution × resolution`` power-flow runs. Default 10.
        Capped at 25 to keep response time under ~10 s with parallel execution.
    vm_upper_pu / vm_lower_pu : float
        Voltage violation thresholds. Default 1.05 / 0.95.
    max_line_loading_pct / max_trafo_loading_pct : float
        Thermal overload thresholds. Default 100.0.
    load_scaling_factor : float
        Load stress multiplier applied before the sweep. Default 1.0.
    """
    gen_name: str
    p_min_mw: float | None = None
    p_max_mw: float | None = None
    q_min_mvar: float | None = None
    q_max_mvar: float | None = None
    resolution: int = 20
    vm_upper_pu: float = 1.05
    vm_lower_pu: float = 0.95
    max_line_loading_pct: float = 100.0
    max_trafo_loading_pct: float = 100.0
    load_scaling_factor: float = 1.0
    reference_state: str = "scada"  # "scada" | "post_opf" | "custom"
    dispatch_overrides: dict | None = None


class HostingCapacityRequest(_TimeseriesRequest):
    """Request body for ``POST /api/flexibility/hosting_capacity``.

    First implementation slice is deterministic and single-bus only. The
    default behavior evaluates all three ``q_mode`` variants in one call so
    the spread is visible to the user.
    """
    bus: int | str
    q_mode: str = "all"
    power_factor: float = 0.95
    pf_sign: str = "absorbing"
    mode: str = "deterministic"
    risk_threshold: float = 0.05
    n_samples: int = 200
    added_gen_sigma: float = 0.1
    load_sigma: float = 0.0
    uncertainty_scope: str = "added_generation_only"
    timestamp: str | None = None
    p_max_search_mw: float | None = None
    tolerance_mw: float = 0.1
    vm_upper_pu: float = 1.05
    vm_lower_pu: float = 0.95
    max_line_loading_pct: float = 90.0
    max_trafo_loading_pct: float = 90.0


class FetchDataRequest(BaseModel):
    """Request body for ``POST /api/eddk/fetch``.

    Date strings are YYYY-MM-DD. They are parsed by
    ``pipeline_functions.define_timespan``.
    """
    start_date: str = "2022-08-01"
    end_date: str = "2022-08-02"


class PushDataRequest(BaseModel):
    """Placeholder for the Phase-4 EDDK push payload. Intentionally empty."""
    pass


class ResetActiveNetworkRequest(BaseModel):
    """Request body for ``POST /api/network/reset_active``; no profile: the default."""
    profile_name: str | None = DEFAULT_GRID_PROFILE


def _build_and_store_ybus(net) -> None:
    """Build Ybus databases on-the-fly from *net* and populate app_data."""
    print("[lifespan] Building Ybus databases on-the-fly...")
    t0 = time.time()
    try:
        app_data["db_full"] = nl.build_ybus_full(net)
        app_data["admittance_source"] = "generated"
        print(f"[lifespan] Ybus ready in {round(time.time() - t0, 2)}s.")
    except Exception as exc:
        print(f"[lifespan] Ybus build failed (flex/OPF will be unavailable): {exc}")


def _extract_vm_limits(net) -> tuple[float, float]:
    """Return (vm_lower, vm_upper) in p.u. from the bus table, falling back to 0.95/1.05."""
    vm_lower = 0.95
    vm_upper = 1.05
    if "min_vm_pu" in net.bus.columns:
        vals = net.bus["min_vm_pu"].dropna()
        if not vals.empty:
            vm_lower = round(float(vals.min()), 4)
    if "max_vm_pu" in net.bus.columns:
        vals = net.bus["max_vm_pu"].dropna()
        if not vals.empty:
            vm_upper = round(float(vals.max()), 4)
    return vm_lower, vm_upper


def _normalize_tap_policy(raw_policy: str | None) -> str:
    """Normalize user-provided transformer tap policy.

    Supported values:
      - ``current``: preserve current tap positions from the uploaded file
        and redefine ``tap_neutral`` to match them.
      - ``neutral``: force ``tap_pos = tap_neutral``.
    """
    val = str(raw_policy or "current").strip().lower()
    aliases_current = {"current", "preserve", "as_is", "as-is", "uploaded"}
    aliases_neutral = {"neutral", "tap_neutral", "tap-neutral"}
    if val in aliases_current:
        return "current"
    if val in aliases_neutral:
        return "neutral"
    raise ValueError(f"Unknown tap_policy {raw_policy!r}. Use 'current' or 'neutral'.")


def _apply_tap_policy(net, tap_policy: str, context: str = "upload") -> None:
    """Apply transformer tap policy in-place for any supported network format."""
    if net.trafo.empty:
        return
    if "tap_pos" not in net.trafo.columns or "tap_neutral" not in net.trafo.columns:
        return

    policy = _normalize_tap_policy(tap_policy)
    diff_mask = (net.trafo["tap_pos"] != net.trafo["tap_neutral"]).fillna(False)
    changed = int(diff_mask.sum())

    if policy == "neutral":
        net.trafo["tap_pos"] = net.trafo["tap_neutral"]
        if changed:
            print(f"[{context}] Tap policy=neutral applied to {changed} trafo(s): tap_pos <- tap_neutral")
        return

    # policy == "current": the taps as imported, untouched. It used to set
    # tap_neutral = tap_pos, which made every imported off-nominal ratio 1.0 —
    # the opposite of keeping the current taps (review 2.3).
    if changed:
        print(f"[{context}] Tap policy=current: {changed} trafo(s) keep their imported tap positions")


def _write_upload_sentinel(
    network_path: str,
    grid_profile: dict,
    convert_gen_to_sgen: bool,
    tap_policy: str,
) -> None:
    """Write a JSON sentinel so the backend can restore the uploaded network on restart."""
    record = {
        "network_path": network_path,
        "grid_profile": grid_profile,
        "convert_gen_to_sgen": convert_gen_to_sgen,
        "tap_policy": tap_policy,
        "uploaded_at": time.time(),
    }
    try:
        with open(_LAST_UPLOAD_SENTINEL, "w", encoding="utf-8") as fh:
            json.dump(record, fh, indent=2)
        print(f"[upload] Sentinel written → {_LAST_UPLOAD_SENTINEL}")
    except Exception as exc:
        print(f"[upload] WARNING: Could not write upload sentinel: {exc}")


def _parse_timeseries_csv(content: bytes, kind: str | None = None) -> tuple[dict, list]:
    """Parse a long-format timeseries CSV into (measurements_dict, timestamps_list).

    Shared by the measurement and forecast upload/restore paths. Optional
    reactive columns (`consumption_mvar`, `production_mvar`) are kept
    (data_quality.py). With `kind`, the parse notes are kept for that dataset's
    quality report. Raises ``ValueError`` with a human-readable message on any
    problem so callers can surface it as a 400 or log it on restart.
    """
    measurements, timestamps, notes = dq.parse(content)
    if kind:
        app_data.setdefault("_parse_notes", {})[kind] = notes
    return measurements, timestamps


def _voltage_substations() -> set:
    """Substations whose units control voltage (they decide their own Q)."""
    net = _net_for(None)
    names = list(substations)
    out = set()
    for idx in net.gen.index[~lg._proposed_mask(net.gen)] if len(net.gen) else []:
        bus = int(net.gen.at[idx, "bus"])
        out.add(sb.measurement_key_for_bus(net, bus, names) or measurement_name(net, bus))
    return out


def _data_quality(kind: str) -> dict | None:
    """The quality report of a loaded dataset ("measurements" / "forecasts"),
    cached until the data or the generator model change."""
    dataset = app_data.get("measurements" if kind == "measurements" else "forecasts") or {}
    if not dataset:
        return None
    key = (id(dataset), len(dataset), _assumptions_key())
    cache = app_data.setdefault("_quality_cache", {})
    if cache.get(kind, (None,))[0] != key:
        report = dq.quality(dataset, (app_data.get("_parse_notes") or {}).get(kind), _voltage_substations())
        report["source"] = app_data.get(f"{kind}_source")
        cache[kind] = (key, report)
    return cache[kind][1]


def _empty_admittance_entry() -> dict:
    return {
        "Yff_r": {},
        "Yff_i": {},
        "Yft_r": {},
        "Yft_i": {},
        "TAPS": [],
        "trafo_ranges": {},
        "trafo_defaults": {},
        "branch_to_trafo": {},
    }


def _parse_upload_admittance_csvs(core_content: bytes, meta_content: bytes, net) -> tuple[dict, dict, dict, dict]:
    """Parse two generic CSV files into (db_full, db_n1_line, db_n1_trafo).

    core CSV columns:
      outage_scope, outage_index, section, from_bus, to_bus, tap, value
      where outage_scope in {full,line,trafo} and section in {Yff_r,Yff_i,Yft_r,Yft_i}

    meta CSV columns:
      outage_scope, outage_index, meta_type, from_bus, to_bus, trafo_index, tap, value
      where meta_type in {TAPS,trafo_defaults,trafo_ranges,branch_to_trafo}
    """
    def _as_int(v, allow_empty: bool = False):
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return None if allow_empty else (_ for _ in ()).throw(ValueError("missing integer value"))
        s = str(v).strip()
        if not s:
            return None if allow_empty else (_ for _ in ()).throw(ValueError("empty integer value"))
        return int(float(s))

    def _scope_and_idx(raw_scope, raw_idx):
        scope = str(raw_scope or "").strip().lower()
        if scope not in {"full", "line", "trafo"}:
            raise ValueError(f"Unsupported outage_scope {raw_scope!r}. Use full|line|trafo.")
        idx = _as_int(raw_idx, allow_empty=True)
        if scope == "full":
            idx = None
        elif idx is None:
            raise ValueError(f"outage_index is required for outage_scope={scope!r}")
        return scope, idx

    def _entry_key(scope, idx):
        return (scope, idx)

    text_core = core_content.decode("utf-8-sig")
    text_meta = meta_content.decode("utf-8-sig")
    df_core = pd.read_csv(io.StringIO(text_core))
    df_meta = pd.read_csv(io.StringIO(text_meta))

    core_required = {"outage_scope", "outage_index", "section", "from_bus", "to_bus", "tap", "value"}
    meta_required = {"outage_scope", "outage_index", "meta_type", "from_bus", "to_bus", "trafo_index", "tap", "value"}
    miss_core = core_required - set(df_core.columns)
    miss_meta = meta_required - set(df_meta.columns)
    if miss_core:
        raise ValueError(f"Core CSV missing columns: {sorted(miss_core)}")
    if miss_meta:
        raise ValueError(f"Meta CSV missing columns: {sorted(miss_meta)}")

    bus_set = set(int(i) for i in net.bus.index)
    line_set = set(int(i) for i in net.line.index)
    trafo_set = set(int(i) for i in net.trafo.index)

    entries: dict[tuple[str, int | None], dict] = {}

    def _get_entry(scope, idx):
        k = _entry_key(scope, idx)
        if k not in entries:
            entries[k] = _empty_admittance_entry()
        return entries[k]

    allowed_sections = {"Yff_r", "Yff_i", "Yft_r", "Yft_i"}
    for _, row in df_core.iterrows():
        scope, idx = _scope_and_idx(row.get("outage_scope"), row.get("outage_index"))
        if scope == "line" and idx not in line_set:
            raise ValueError(f"Core CSV references unknown line outage_index={idx}")
        if scope == "trafo" and idx not in trafo_set:
            raise ValueError(f"Core CSV references unknown trafo outage_index={idx}")

        section = str(row.get("section") or "").strip()
        if section not in allowed_sections:
            raise ValueError(f"Unsupported section {section!r}. Use {sorted(allowed_sections)}")

        fb = _as_int(row.get("from_bus"))
        tb = _as_int(row.get("to_bus"))
        tap = _as_int(row.get("tap"))
        val = float(row.get("value"))
        if fb not in bus_set or tb not in bus_set:
            raise ValueError(f"Core CSV branch ({fb},{tb}) is not valid for current network buses.")

        e = _get_entry(scope, idx)
        e[section][((fb, tb), tap)] = val

    for _, row in df_meta.iterrows():
        scope, idx = _scope_and_idx(row.get("outage_scope"), row.get("outage_index"))
        if scope == "line" and idx not in line_set:
            raise ValueError(f"Meta CSV references unknown line outage_index={idx}")
        if scope == "trafo" and idx not in trafo_set:
            raise ValueError(f"Meta CSV references unknown trafo outage_index={idx}")

        mtype = str(row.get("meta_type") or "").strip()
        e = _get_entry(scope, idx)

        if mtype == "TAPS":
            t = _as_int(row.get("tap"), allow_empty=True)
            if t is None:
                t = _as_int(row.get("value"))
            e["TAPS"].append(int(t))
            continue

        if mtype == "trafo_defaults":
            tidx = _as_int(row.get("trafo_index"))
            if tidx not in trafo_set:
                raise ValueError(f"Meta CSV trafo_defaults references unknown trafo_index={tidx}")
            t = _as_int(row.get("tap"), allow_empty=True)
            if t is None:
                t = _as_int(row.get("value"))
            e["trafo_defaults"][tidx] = int(t)
            continue

        if mtype == "trafo_ranges":
            tidx = _as_int(row.get("trafo_index"))
            if tidx not in trafo_set:
                raise ValueError(f"Meta CSV trafo_ranges references unknown trafo_index={tidx}")
            t = _as_int(row.get("tap"), allow_empty=True)
            if t is None:
                t = _as_int(row.get("value"))
            e["trafo_ranges"].setdefault(tidx, []).append(int(t))
            continue

        if mtype == "branch_to_trafo":
            fb = _as_int(row.get("from_bus"))
            tb = _as_int(row.get("to_bus"))
            tidx = _as_int(row.get("trafo_index"), allow_empty=True)
            if tidx is None:
                tidx = _as_int(row.get("value"))
            if fb not in bus_set or tb not in bus_set:
                raise ValueError(f"Meta CSV branch_to_trafo references invalid branch ({fb},{tb})")
            if tidx not in trafo_set:
                raise ValueError(f"Meta CSV branch_to_trafo references unknown trafo_index={tidx}")
            e["branch_to_trafo"][(fb, tb)] = tidx
            continue

        raise ValueError(
            f"Unsupported meta_type {mtype!r}. Use TAPS|trafo_defaults|trafo_ranges|branch_to_trafo"
        )

    # Finalize dedupe/sort for list-like fields and provide defaults where absent.
    for e in entries.values():
        e["TAPS"] = sorted(set(int(t) for t in e.get("TAPS", []))) or [1]
        for tidx, taps in list(e.get("trafo_ranges", {}).items()):
            e["trafo_ranges"][tidx] = sorted(set(int(t) for t in taps))

    db_full = entries.get(("full", None))
    if not db_full:
        raise ValueError("Advanced admittance upload requires at least one full-scope entry (outage_scope=full).")

    db_n1_line = {idx: e for (scope, idx), e in entries.items() if scope == "line"}
    db_n1_trafo = {idx: e for (scope, idx), e in entries.items() if scope == "trafo"}

    summary = {
        "n_full_scalars": sum(len(db_full[k]) for k in ("Yff_r", "Yff_i", "Yft_r", "Yft_i")),
        "n_line_outages": len(db_n1_line),
        "n_trafo_outages": len(db_n1_trafo),
    }
    return db_full, db_n1_line, db_n1_trafo, summary


def _validate_uploaded_admittance(db_full: dict, net) -> None:
    """Reject admittance data whose topology does not match the active network.

    Two independent checks, both fatal (HTTP 400):
    - the CSV must not reference bus indices absent from ``net.bus`` (data was
      built for a different/larger network);
    - every in-service branch of the active network (lines + trafos) must have
      an admittance entry, otherwise the OPF crashes later with an opaque
      ``KeyError`` on the first request that touches the uncovered branch.
    """
    yff_keys = db_full.get("Yff_r", {}).keys()
    covered_pairs = {pair for (pair, _tap) in yff_keys}

    net_bus_ids = {int(b) for b in net.bus.index}
    csv_bus_ids = {int(b) for pair in covered_pairs for b in pair}
    unknown_buses = sorted(csv_bus_ids - net_bus_ids)

    required_pairs = set()
    for _, row in net.line[net.line["in_service"] == True].iterrows():  # noqa: E712
        required_pairs.add((int(row["from_bus"]), int(row["to_bus"])))
    for _, row in net.trafo[net.trafo["in_service"] == True].iterrows():  # noqa: E712
        required_pairs.add((int(row["hv_bus"]), int(row["lv_bus"])))
    missing_pairs = sorted(
        p for p in required_pairs
        if p not in covered_pairs and (p[1], p[0]) not in covered_pairs
    )

    problems = []
    if unknown_buses:
        shown = ", ".join(map(str, unknown_buses[:10]))
        problems.append(
            f"CSV references bus indices that do not exist in the active network: {shown}"
            + (" …" if len(unknown_buses) > 10 else "")
        )
    if missing_pairs:
        shown = ", ".join(f"{i}-{j}" for i, j in missing_pairs[:10])
        problems.append(
            f"CSV has no admittance entry for {len(missing_pairs)} in-service branch(es) "
            f"of the active network (from_bus-to_bus): {shown}"
            + (" …" if len(missing_pairs) > 10 else "")
        )
    if problems:
        raise HTTPException(
            status_code=400,
            detail=(
                "Advanced admittance data does not match the currently loaded network — "
                "it was likely generated from a different network version. "
                + " | ".join(problems)
                + " Re-export the CSVs from the active network, or re-upload the network first."
            ),
        )


def _measurements_to_csv_bytes(measurements: dict, timestamps: list) -> bytes:
    """Serialize a {timestamp: DataFrame[substation_name, production, consumption]}
    series to the canonical long-format CSV (the same shape uploads use), so a
    generated series can be persisted and later re-parsed by _parse_timeseries_csv.
    """
    frames = []
    for ts in timestamps:
        df = measurements[ts].copy()
        df.insert(0, "timestamp", ts)
        frames.append(df)
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["timestamp", "substation_name", "production", "consumption"]
    )
    out = out.rename(columns={"production": "production_mw", "consumption": "consumption_mw"})
    out = out[["timestamp", "substation_name", "production_mw", "consumption_mw"]]
    return out.to_csv(index=False).encode("utf-8")


def _persist_generated_series(measurements: dict, timestamps: list, kind: str) -> str:
    """Write a generated (synthetic) series to data_files as a CSV and return its
    path. ``kind`` is 'measurements' or 'forecast'. Uses a fixed filename per kind
    so each network upload overwrites the previous synthetic series rather than
    accumulating files.
    """
    content = _measurements_to_csv_bytes(measurements, timestamps)
    return _persist_uploaded_bytes(
        DATA_FILES_DIR, None, content, f"{kind}_synthetic", ".csv"
    )


def _write_timeseries_sentinel(csv_path: str, source: str = "uploaded") -> None:
    """Write a JSON sentinel so the backend can restore the measurement CSV on
    restart. ``source`` records whether the CSV is user-'uploaded' or 'synthetic'."""
    record = {"csv_path": csv_path, "source": source, "uploaded_at": time.time()}
    try:
        with open(_LAST_TIMESERIES_SENTINEL, "w", encoding="utf-8") as fh:
            json.dump(record, fh, indent=2)
        print(f"[upload] Timeseries sentinel written ({source}) → {_LAST_TIMESERIES_SENTINEL}")
    except Exception as exc:
        print(f"[upload] WARNING: Could not write timeseries sentinel: {exc}")


def _restore_timeseries_from_sentinel() -> bool:
    """After network startup, check for a saved CSV and overlay it on top of
    the synthetic timeseries that was just generated.

    Returns True if the CSV was successfully restored (app_data measurements
    are replaced), False if anything goes wrong (synthetic data stays active).
    """
    if not os.path.isfile(_LAST_TIMESERIES_SENTINEL):
        return False
    try:
        with open(_LAST_TIMESERIES_SENTINEL, encoding="utf-8") as fh:
            sentinel = json.load(fh)
    except Exception as exc:
        print(f"[lifespan] Could not read timeseries sentinel: {exc}")
        return False

    csv_path = sentinel.get("csv_path", "")
    source = sentinel.get("source", "uploaded")
    if not os.path.isfile(csv_path):
        print(f"[lifespan] Timeseries sentinel points to missing file {csv_path!r} — using synthetic.")
        return False

    print(f"[lifespan] Restoring {source} timeseries: {csv_path!r}")
    try:
        with open(csv_path, "rb") as fh:
            content = fh.read()
        new_measurements, new_timestamps = _parse_timeseries_csv(content, "measurements")
    except Exception as exc:
        print(f"[lifespan] Could not parse timeseries CSV: {exc} — using synthetic.")
        return False

    app_data["measurements"]  = new_measurements
    app_data["timestamps"]    = new_timestamps
    app_data["current_index"] = 0
    app_data["measurements_source"] = source

    try:
        net = app_data["net"]
        subs = substations
        initial_meas = prepare_measurement_df(new_measurements[new_timestamps[0]])
        net, _ = lg.assign_load_values_from_measurements(net, initial_meas, subs)
        net, _ = lg.assign_generators_values_from_measurements(net, initial_meas, subs)
        app_data["net"] = net
    except Exception as exc:
        print(f"[lifespan] Timeseries restore: network assignment failed (non-fatal): {exc}")

    print(f"[lifespan] Timeseries restored: {len(new_timestamps)} timestamps "
          f"({new_timestamps[0]} → {new_timestamps[-1]}).")
    return True


def _write_forecast_sentinel(csv_path: str, source: str = "uploaded") -> None:
    """Write a JSON sentinel so the backend can restore the forecast CSV on restart.
    ``source`` records whether the CSV is user-'uploaded' or 'synthetic'."""
    record = {"csv_path": csv_path, "source": source, "uploaded_at": time.time()}
    try:
        with open(_LAST_FORECAST_SENTINEL, "w", encoding="utf-8") as fh:
            json.dump(record, fh, indent=2)
        print(f"[upload] Forecast sentinel written ({source}) → {_LAST_FORECAST_SENTINEL}")
    except Exception as exc:
        print(f"[upload] WARNING: Could not write forecast sentinel: {exc}")


def _restore_forecast_from_sentinel() -> bool:
    """After startup, restore a previously uploaded forecast CSV (if any).

    Forecasts are read-only look-ahead data: this never touches the live network
    or the simulation clock — it only replaces ``app_data["forecasts"]`` and
    ``app_data["forecast_timestamps"]``. Returns True on success.
    """
    if not os.path.isfile(_LAST_FORECAST_SENTINEL):
        return False
    try:
        with open(_LAST_FORECAST_SENTINEL, encoding="utf-8") as fh:
            sentinel = json.load(fh)
    except Exception as exc:
        print(f"[lifespan] Could not read forecast sentinel: {exc}")
        return False

    csv_path = sentinel.get("csv_path", "")
    source = sentinel.get("source", "uploaded")

    # If measurements come from a user upload, do not auto-attach a synthetic
    # forecast from an older network upload. This keeps the timeline truthful:
    # forecast appears only when explicitly uploaded by the user.
    if source == "synthetic" and app_data.get("measurements_source") == "uploaded":
        app_data["forecasts"] = {}
        app_data["forecast_timestamps"] = []
        app_data["forecasts_source"] = "none"
        print("[lifespan] Skipping synthetic forecast restore because uploaded measurements are active.")
        return False

    if not os.path.isfile(csv_path):
        print(f"[lifespan] Forecast sentinel points to missing file {csv_path!r} — using synthetic forecast.")
        return False

    print(f"[lifespan] Restoring {source} forecast: {csv_path!r}")
    try:
        with open(csv_path, "rb") as fh:
            content = fh.read()
        measurements, timestamps = _parse_timeseries_csv(content, "forecasts")
    except Exception as exc:
        print(f"[lifespan] Could not parse forecast CSV: {exc} — using synthetic forecast.")
        return False

    app_data["forecasts"] = measurements
    app_data["forecast_timestamps"] = timestamps
    app_data["forecasts_source"] = source
    print(f"[lifespan] Forecast restored: {len(timestamps)} timestamps "
          f"({timestamps[0]} → {timestamps[-1]}).")
    return True


def _prepare_runtime_reload() -> None:
    """Reset mutable runtime containers before loading a profile/network again."""
    substations.clear()
    PG_MAX_DATA.clear()
    app_data["forecasts"] = {}
    app_data["forecast_timestamps"] = []
    app_data["forecasts_source"] = "none"
    app_data["measurements_source"] = "synthetic"
    app_data["last_dispatch_result"] = []
    app_data["last_dispatch_timestamp"] = None
    app_data["nominal"] = None
    app_data["load_power_factor"] = None
    app_data["generator_model"] = {}
    app_data["unit_costs"] = {}
    app_data["element_names"] = {}
    app_data["option_sets"] = {}
    app_data["_parse_notes"] = {}
    app_data["_quality_cache"] = {}
    _discard_proposals()


def _lifespan_from_uploaded(sentinel: dict) -> bool:
    """Startup path that restores a previously uploaded network.

    Supports .m (MATPOWER), .json (pandapower JSON), .xlsx (pandapower Excel),
    and .uct (UCTE) formats, detected from the stored file extension.

    Returns True when the network is loaded and global state is fully
    populated, False on any error so the caller can fall back to the
    default YAML profile.
    """
    import synthetic_timeseries as st

    _prepare_runtime_reload()

    network_path = sentinel.get("network_path", "")
    if not os.path.isfile(network_path):
        print(f"[lifespan] Sentinel points to missing file {network_path!r} — falling back.")
        return False

    ext = os.path.splitext(network_path)[1].lower()
    print(f"[lifespan] Restoring previously uploaded network: {network_path!r} (format: {ext})")
    try:
        if ext == ".m":
            net = nl.read_matpower(network_path)
        elif ext == ".json":
            net = pp.from_json(network_path)
        elif ext == ".xlsx":
            net = pp.from_excel(network_path)
        elif ext == ".uct":
            from pandapower.converter.ucte.from_ucte import from_ucte
            net = from_ucte(network_path)
        else:
            print(f"[lifespan] Unknown extension {ext!r} in sentinel — falling back.")
            return False
    except Exception as exc:
        print(f"[lifespan] Could not parse uploaded network: {exc} — falling back.")
        return False

    # Restore the same tap policy selected at upload time.
    tap_policy = sentinel.get("tap_policy", "current")
    try:
        tap_policy = _normalize_tap_policy(tap_policy)
    except ValueError:
        print(f"[lifespan] Invalid tap_policy in sentinel ({tap_policy!r}) — using 'current'.")
        tap_policy = "current"
    _apply_tap_policy(net, tap_policy, context="lifespan")

    # gen→sgen conversion.
    if sentinel.get("convert_gen_to_sgen", True) and len(net.gen) > 0:
        net, created_names, kept_condensers = nl.gen_to_sgen(net)
        if created_names or kept_condensers:
            print(f"[lifespan] gen→sgen: created={created_names}, kept={kept_condensers}")

    try:
        pp.runpp(net, algorithm="nr", calculate_voltage_angles=True)
        print("[lifespan] Uploaded network power flow converged.")
    except Exception as exc:
        print(f"[lifespan] Power flow failed (non-fatal): {exc}")

    print("[lifespan] Generating synthetic timeseries for uploaded network...")
    timestamps, measurements = st.generate(net, n_days=7, resolution_min=15, seed=42)
    print(f"[lifespan] Timeseries ready: {len(timestamps)} timestamps.")

    # The case's own operating point, before the first tick is written over it.
    app_data["nominal"] = nom.snapshot(net)
    initial_meas = measurements[timestamps[0]]
    subs = get_substation_names(net)
    substations.extend(subs)
    net, _ = lg.assign_load_values_from_measurements(net, initial_meas, subs)
    net, _ = lg.assign_generators_values_from_measurements(net, initial_meas, subs)

    grid_profile = sentinel.get("grid_profile", {})
    grid_profile["tap_policy"] = tap_policy
    # Re-extract vm limits from the live net in case the sentinel was written
    # before the vm-limit extraction fix was in place.
    _vm_lower, _vm_upper = _extract_vm_limits(net)
    grid_profile["vm_lower"] = _vm_lower
    grid_profile["vm_upper"] = _vm_upper

    app_data["net"]            = net
    app_data["measurements"]   = measurements
    app_data["timestamps"]     = timestamps
    app_data["current_index"]  = 0
    app_data["slack_max_mw"]   = float(grid_profile.get("slack_max_mw", 999.0))
    app_data["grid_profile"]   = grid_profile
    app_data["default_vm_lower"] = _vm_lower
    app_data["default_vm_upper"] = _vm_upper
    app_data["db_full"]        = None

    PG_MAX_DATA.update(get_pg_max_data(net))
    print(f"[lifespan] PG_MAX_DATA: {len(PG_MAX_DATA)} generators.")

    _build_and_store_ybus(net)
    _remember_loaded(net)
    _restore_timeseries_from_sentinel()
    _restore_forecast_from_sentinel()
    print("\n !API IS READY TO RECEIVE REQUESTS! \n")
    return True


def _lifespan_generic(profile_name: str, backend_dir: str, restore_uploads: bool = True) -> None:
    """Startup path for any YAML profile.

    Loads the network via network_loader, generates a synthetic timeseries,
    builds Ybus databases on-the-fly, and populates all global app_data fields.
    """
    import synthetic_timeseries as st

    _prepare_runtime_reload()
    print(f"[lifespan] Loading profile '{profile_name}'...")
    try:
        profile = nl.load_profile(profile_name, backend_dir=backend_dir)
    except FileNotFoundError as exc:
        print(f"[lifespan] Profile not found: {exc}")
        return

    print(f"[lifespan] Loading network (source='{profile.get('source')}')...")
    try:
        net = nl.load_network(profile, backend_dir=backend_dir)
    except Exception as exc:
        print(f"[lifespan] Failed to load network: {exc}")
        return

    if len(net.gen) > 0:
        net, created_names, kept_condensers = nl.gen_to_sgen(net)
        if created_names or kept_condensers:
            print(
                "[lifespan] gen->sgen conversion: "
                f"created={created_names}, kept_condensers={kept_condensers}"
            )

    # A profile can load its units under another control than power factor
    # 1.0 — synchronous machines holding a voltage, as the case describes
    # them. Applied by the generator model itself, before the network is
    # recorded as loaded, so it is how these units were loaded: "reset to as
    # loaded" comes back to it.
    if profile.get("unit_model"):
        changes = [{"unit": unit, **settings} for unit, settings in profile["unit_model"].items()]
        net = gm.apply(net, gm.normalise(net, changes))
        print(f"[lifespan] Unit model from the profile: {sorted(profile['unit_model'])}")

    applied = nl.apply_voltage_setpoints(net, profile)
    if applied:
        print(f"[lifespan] Voltage setpoints from the profile: {applied}")

    # Initial power flow to establish an operating point.
    try:
        pp.runpp(net, algorithm="nr", calculate_voltage_angles=True)
        print("[lifespan] Initial power flow converged.")
    except Exception as exc:
        print(f"[lifespan] Initial power flow failed (non-fatal): {exc}")

    # Synthetic timeseries.
    n_days      = int(profile.get("synthetic_n_days", 7))
    resolution  = int(profile.get("synthetic_resolution_min", 15))
    load_prof   = profile.get("synthetic_load_profile", "residential")
    gen_prof    = profile.get("synthetic_generation_profile", "wind")
    print(f"[lifespan] Generating synthetic timeseries ({n_days}d, {resolution}min)...")
    peak = profile.get("synthetic_peak_load_factor", 1.0)
    timestamps, measurements = st.generate(
        net, n_days=n_days, resolution_min=resolution,
        load_profile=load_prof, generation_profile=gen_prof, seed=42,
        peak_load_factor=peak,
    )
    print(f"[lifespan] Timeseries ready: {len(timestamps)} timestamps.")
    # A synthetic forecast, as a network upload gets — so the default network
    # shows the forecast path too. It covers the last days of the measurements
    # (measured values with a day-ahead-sized error, so forecast and measured
    # can be compared) and the days after (an independent realisation).
    forecast_timestamps, forecasts = st.forecast(
        net, timestamps, measurements,
        overlap_days=float(profile.get("synthetic_forecast_overlap_days", 2)),
        n_days=n_days, resolution_min=resolution,
        load_profile=load_prof, generation_profile=gen_prof, seed=137, peak_load_factor=peak,
    )

    # The case's own operating point, before the first tick is written over it.
    app_data["nominal"] = nom.snapshot(net)
    # Assign first-tick measurements to initialise operating point.
    initial_meas = measurements[timestamps[0]]
    subs = get_substation_names(net)
    substations.extend(subs)
    net, _ = lg.assign_load_values_from_measurements(net, initial_meas, subs)
    net, _ = lg.assign_generators_values_from_measurements(net, initial_meas, subs)

    # Populate global state.
    app_data["net"] = net
    app_data["measurements"] = measurements
    app_data["timestamps"] = timestamps
    app_data["current_index"] = 0
    app_data["slack_max_mw"] = float(profile.get("slack_max_mw", 999.0))
    app_data["grid_profile"] = {
        "name": profile.get("name", profile_name),
        "profile_name": profile_name,
        "source": profile.get("source", ""),
        "measurement_source": profile.get("measurement_source", "synthetic"),
        "vm_lower": float(profile.get("vm_lower", 0.9)),
        "vm_upper": float(profile.get("vm_upper", 1.1)),
        "max_loading_pct": float(profile.get("max_loading_pct", 100.0)),
        "slack_max_mw": float(profile.get("slack_max_mw", 999.0)),
        "slack_name": profile.get("slack_name", "External Grid"),
        "grid_type": profile.get("grid_type", "unknown"),
        "description": profile.get("description", ""),
        "voltage_level_kv": float(profile.get("voltage_level_kv", 0.0)),
        "load_scaling_max": 4.0,
    }
    app_data["default_vm_lower"] = float(profile.get("vm_lower", 0.95))
    app_data["default_vm_upper"] = float(profile.get("vm_upper", 1.05))
    app_data["forecasts"] = forecasts
    app_data["forecast_timestamps"] = forecast_timestamps
    app_data["forecasts_source"] = "synthetic"

    PG_MAX_DATA.update(get_pg_max_data(net))
    print(f"[lifespan] PG_MAX_DATA: {len(PG_MAX_DATA)} generators.")

    # Build Ybus on-the-fly (generic profiles never use pre-computed pickles).
    _build_and_store_ybus(net)
    _remember_loaded(net)
    if restore_uploads:
        # Not for a demo: it must load the same data every time, and leave the
        # operator's uploaded data for Resume.
        _restore_timeseries_from_sentinel()
        _restore_forecast_from_sentinel()
    # Uploaded measurements never come with an invented forecast (the same rule
    # as uploading them at runtime): a forecast appears only when uploaded.
    if app_data.get("measurements_source") == "uploaded" and app_data.get("forecasts_source") == "synthetic":
        app_data["forecasts"] = {}
        app_data["forecast_timestamps"] = []
        app_data["forecasts_source"] = "none"

    print("\n !API IS READY TO RECEIVE REQUESTS! \n")


def _restore_proposals_after_upload() -> list[str]:
    _restore_assumptions()
    return _restore_proposals()


def _read_json(path: str) -> dict | None:
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def _saved_scenario() -> str | None:
    name = (_read_json(_SCENARIO_FILE) or {}).get("scenario")
    return name if name in scn.SCENARIOS else None


def _save_scenario(name: str, record: dict | None = None) -> None:
    """The scenario chosen, and (`record`) where each scenario's network keeps
    its saved changes — kept from earlier writes."""
    data = _read_json(_SCENARIO_FILE) if os.path.isfile(_SCENARIO_FILE) else None
    data = data if isinstance(data, dict) else {}
    networks = dict(data.get("networks") or {})
    networks.update(record or {})
    try:
        os.makedirs(os.path.dirname(_SCENARIO_FILE), exist_ok=True)
        with open(_SCENARIO_FILE, "w", encoding="utf-8") as fh:
            json.dump({"scenario": name, "networks": networks}, fh)
    except OSError as exc:
        print(f"[scenario] Could not save the choice: {exc}")


def _note_where_changes_are_kept() -> None:
    """Remember, for the scenario loaded now, the keys its saved assumptions
    and proposals are filed under — so the launcher can say what is kept for a
    scenario that is not loaded, and Start fresh can clear it."""
    name, base = app_data.get("scenario"), app_data.get("net")
    if not name or base is None or not app_data.get("loaded_fingerprint"):
        return
    keys = {"fingerprint": app_data["loaded_fingerprint"], "proposals_fingerprint": sb.network_fingerprint(base)}
    # Demos on one network share what is kept: say so for each of them.
    profile = scn.SCENARIOS.get(name, {}).get("profile")
    same = [k for k, spec in scn.SCENARIOS.items() if profile and spec.get("profile") == profile] or [name]
    _save_scenario(_saved_scenario() or name, {k: keys for k in same})


def _kept_changes(keys: dict | None) -> dict:
    """What is saved for a network: renames, cost changes, units modelled
    differently, a load power factor, proposals (only the non-zero ones)."""
    if not keys:
        return {}
    saved = _read_store(_ASSUMPTIONS_FILE).get(keys.get("fingerprint")) or {}
    proposals = (_read_store(_PROPOSALS_FILE).get(keys.get("proposals_fingerprint")) or {}).get("proposals") or {}
    counts = {
        "renames": sum(len(v or {}) for v in (saved.get("element_names") or {}).values()),
        "costs": len(saved.get("unit_costs") or {}),
        "generators": len(saved.get("generator_model") or {}),
        "load_power_factor": int(saved.get("load_power_factor") is not None),
        "proposals": len(proposals),
    }
    return {k: v for k, v in counts.items() if v}


def _has_uploads() -> bool:
    return os.path.isfile(_LAST_UPLOAD_SENTINEL) or os.path.isfile(_LAST_TIMESERIES_SENTINEL)


def _load_demo(name: str) -> None:
    """A demo scenario: the bundled network and its synthetic week (with
    measured reactive power for the reactive demo). The "last uploaded" records
    are not read or touched, so Resume stays as it was."""
    spec = scn.SCENARIOS[name]
    _lifespan_generic(spec["profile"], BACKEND_DIR, restore_uploads=False)
    if spec.get("reactive"):
        content = scn.demo_reactive_csv(app_data["measurements"])
        measurements, timestamps = _parse_timeseries_csv(content, "measurements")
        app_data["measurements"], app_data["timestamps"] = measurements, timestamps
        app_data["current_index"] = 0
        app_data["measurements_source"] = "demo"
        _assign_tick(_net_for(None), prepare_measurement_df(measurements[timestamps[0]]), timestamps[0])
    app_data["scenario"] = name


def _load_resume() -> bool:
    """The operator's last uploaded network and data. False when the uploaded
    network cannot be restored (the default profile with any uploaded data is
    loaded instead)."""
    sentinel = _read_json(_LAST_UPLOAD_SENTINEL) if os.path.isfile(_LAST_UPLOAD_SENTINEL) else None
    ok = bool(sentinel) and _lifespan_from_uploaded(sentinel)
    profile_name = os.environ.get("GRID_PROFILE", DEFAULT_GRID_PROFILE).lower().strip()
    if not ok:
        _lifespan_generic(profile_name, BACKEND_DIR)
    app_data["scenario"] = "resume" if _has_uploads() else scn.demo_for_profile(profile_name)
    return ok


def _scenario_payload() -> dict:
    resume = scn.resume_detail(_read_json(_LAST_UPLOAD_SENTINEL), _read_json(_LAST_TIMESERIES_SENTINEL))
    current = app_data.get("scenario") or ("resume" if _has_uploads() else scn.DEFAULT_DEMO)
    options = []
    networks = ((_read_json(_SCENARIO_FILE) if os.path.isfile(_SCENARIO_FILE) else None) or {}).get("networks") or {}
    for key, spec in scn.SCENARIOS.items():
        option = {"name": key, "label": spec["label"], "description": spec["description"], "available": True}
        kept = _kept_changes(networks.get(key))
        if kept:
            option["kept"] = kept
        if key == "resume":
            option["available"] = resume["available"]
            parts = [p for p in (resume["network"] and f"network {resume['network']}",
                                 resume["data"] and f"data {resume['data']}") if p]
            option["description"] = ("Your last upload: " + ", ".join(parts) + "." if parts else
                                     "Nothing uploaded yet — upload a network or data to have something to resume.")
        options.append(option)
    return {"current": current, "label": scn.SCENARIOS[current]["label"], "options": options}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown function for FastAPI.

    Runs once when uvicorn boots. The active network is selected by
    ``GRID_PROFILE`` and initialized through the generic profile loader.
    """
    print("0) Starting Backend initialization...")

    _ensure_storage_dirs()
    print(f"1) Upload storage ready: data_files={DATA_FILES_DIR}, systems={SYSTEMS_DIR}")

    # Probe the IPOPT solver once so a missing/broken binary is an obvious
    # startup message instead of a mysterious 500 on the first OPF request.
    try:
        from pyomo.environ import SolverFactory
        app_data["ipopt_available"] = bool(SolverFactory("ipopt").available(exception_flag=False))
    except Exception:
        app_data["ipopt_available"] = False
    if app_data["ipopt_available"]:
        print("[lifespan] IPOPT solver available ✓")
    else:
        print(
            "[lifespan] WARNING: IPOPT solver NOT available — optimization endpoints "
            "(flexibility, contingency optimize, KPIs) will fail. "
            "Install it with: conda install -c conda-forge ipopt"
        )

    # The scenario chosen last time: a demo loads as itself; otherwise the
    # operator's uploads (Resume), as before.
    _chosen = _saved_scenario()
    if _chosen in scn.DEMOS:
        print(f"[lifespan] Scenario: {_chosen}")
        _load_demo(_chosen)
        _restore_assumptions()
        _restore_proposals()
        yield
        print("Shutting down backend")
        return

    # Prefer a previously uploaded network over the default YAML profile.
    if os.path.isfile(_LAST_UPLOAD_SENTINEL):
        print(f"[lifespan] Found upload sentinel: {_LAST_UPLOAD_SENTINEL}")
        try:
            with open(_LAST_UPLOAD_SENTINEL, encoding="utf-8") as fh:
                sentinel = json.load(fh)
            if _lifespan_from_uploaded(sentinel):
                app_data["scenario"] = "resume"
                _restore_assumptions()
                _restore_proposals()
                yield
                print("Shutting down backend")
                return
            print("[lifespan] Sentinel restore failed — falling back to YAML profile.")
        except Exception as exc:
            print(f"[lifespan] Could not read sentinel: {exc} — falling back to YAML profile.")

    profile_name = os.environ.get("GRID_PROFILE", DEFAULT_GRID_PROFILE).lower().strip()
    print(f"[lifespan] GRID_PROFILE={profile_name!r}")

    _lifespan_generic(profile_name, BACKEND_DIR)
    app_data["scenario"] = "resume" if _has_uploads() else scn.demo_for_profile(profile_name)
    _restore_assumptions()
    _restore_proposals()

    yield
    print("Shutting down backend")


app = FastAPI(lifespan=lifespan)


# Endpoints are `async def` but do blocking pandapower / IPOPT work, so on the
# event loop one long scan stalled every request, `/health` and the clock
# included. Each endpoint that reads or changes `app_data` now runs in a worker
# thread instead. They share the network, the clock and the admittance
# database, so one lock still runs them one at a time — as before — while the
# cheap reads below stay answerable.
_STATE_LOCK = threading.Lock()
_holds_state = threading.local()
# What the elements a proposal added inject at the first tick a request
# studied (set by `_assign_tick`, read by `_echo_network`).
_tick_report = threading.local()
# The load power factor of the study running in this thread, and whether a
# tick used it (studies run one at a time, under _STATE_LOCK).
_study = threading.local()


def _echo_power_factor(result):
    """A result that assigned a tick says which load power factor gave the
    loads their reactive power — and whether it was the default, the
    session's setting or this study's own."""
    if not isinstance(result, dict) or not getattr(_study, "pf_used", False) or "load_power_factor" in result:
        return result
    own = getattr(_study, "load_pf", None)
    value, source = (own, "this study") if own is not None else _load_pf_setting()
    result["load_power_factor"] = {"value": value, "source": source, "lagging": True,
                                   "applies_to": "measured loads (Q = P·tan(acos pf)); the nominal "
                                                 "point keeps the case's own Q"}
    q_share = getattr(_study, "q_share", None)
    if q_share:
        lo, hi = q_share
        result["load_power_factor"]["measured_reactive_power"] = {
            "share_of_load": round(hi, 4) if abs(hi - lo) < 1e-4 else [round(lo, 4), round(hi, 4)],
            # Judged on the least-covered tick: a window where any tick had a
            # gap used the power factor there.
            "note": ("The data carry measured reactive demand: loads with a measurement use it; the power "
                     "factor applies only to the rest." if lo < 0.9999 else
                     "The data carry measured reactive demand for every load: the power factor is not used."),
        }
    model = app_data.get("generator_model") or {}
    if model and "generator_model" not in result:
        # Units modelled differently from the network as loaded: said in every
        # result, so a figure under a changed model is never read as the default.
        result["generator_model"] = {"changed_units": _by_name_now(model),
                                     # Keyed as the settings are (names as loaded): a later
                                     # rename must not read as a change of model.
                                     "changed_units_as_loaded": model,
                                     "note": "These units are modelled differently from the network "
                                             "as loaded (System tab / set_generator_model)."}
    return result


def _echo_network(result, args, kwargs):
    """A result computed on a proposal says so, with the proposal's edits and
    what its added elements inject at the tick — so no figure from a proposal
    can be read as the base's. Results on the base are returned unchanged."""
    request = next((a for a in (*args, *kwargs.values()) if hasattr(a, "network")), None)
    if request is None or not isinstance(result, dict):
        return result
    name = _network_name(request)
    if name == BASE_NETWORK or "network" in result:
        return result
    result["network"] = _proposal_card(name)
    first = getattr(_tick_report, "first", None)
    if first:
        result["network"]["added_elements_at_tick"] = first
    return result


def _off_event_loop(endpoint):
    @functools.wraps(endpoint)
    async def run_in_worker(*args, **kwargs):
        if getattr(_holds_state, "active", False):
            # One locked endpoint calling another (robust → optimise): this
            # thread already holds the lock.
            return await endpoint(*args, **kwargs)

        def locked():
            with _STATE_LOCK:
                _holds_state.active = True
                _tick_report.first = None
                request = next((a for a in (*args, *kwargs.values()) if hasattr(a, "load_power_factor")), None)
                _study.load_pf = getattr(request, "load_power_factor", None)
                _study.pf_used = False
                _study.q_share = None
                try:
                    result = asyncio.run(endpoint(*args, **kwargs))
                    return _echo_power_factor(_echo_network(result, args, kwargs))
                finally:
                    _holds_state.active = False
                    _study.load_pf = None

        return await anyio.to_thread.run_sync(locked)

    return run_in_worker

app.add_middleware(
    CORSMiddleware,
    # Only pages served from this machine (the Streamlit app, on whatever port)
    # may call the backend from a browser. `*` let any website the operator
    # happened to open post uploads or resets to the local backend. A LAN
    # deployment adds its origins in CONDUCTOR_CORS_ORIGINS (comma-separated).
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1|\[::1\])(:\d+)?$",
    allow_origins=[o.strip() for o in os.environ.get("CONDUCTOR_CORS_ORIGINS", "").split(",") if o.strip()],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Tests if API is running. Returns a static hello so the frontend can confirm the backend is up."""
    return {"message": "Power-system digital twin backend is online"}


@app.get("/health")
async def health():
    """Lightweight liveness/readiness probe for launchers and Docker healthchecks."""
    return {
        "status": "ok",
        "network_loaded": app_data.get("net") is not None,
        "ipopt_available": app_data.get("ipopt_available"),
    }


@app.get("/api/network/topology")
@_off_event_loop
async def network_topology(network: str | None = None):
    """Buses and branches of the network currently loaded — or of a proposal
    (`?network=<id>`), whose buses and branches added by its edits carry
    `"added": true`.

    The shape of the system, so a caller can draw it or reason about how it is
    connected. `/api/grid_constants` reports counts and names; this reports
    what is joined to what.

    Read from the live network rather than from the file it came from: after
    an upload the two differ, and a picture drawn from the file would show a
    system nobody is analysing.
    """
    import types
    request = types.SimpleNamespace(network=network)
    net = _net_for(request)
    proposed = _network_name(request) != BASE_NETWORK

    def added(table, idx) -> dict:
        if not proposed or "proposed" not in net[table].columns:
            return {}
        return {"added": True} if net[table].at[idx, "proposed"] is True else {}

    buses = [
        {
            "index": int(idx),
            "name": _bus_display_name(net, int(idx)),
            "vn_kv": _safe_float(net.bus.at[idx, "vn_kv"]) or 0.0,
            "in_service": bool(net.bus.at[idx, "in_service"])
            if "in_service" in net.bus.columns else True,
            **added("bus", idx),
        }
        for idx in net.bus.index
    ]

    branches = []
    for idx in net.line.index:
        branches.append({
            "index": int(idx),
            "name": _line_display_name(net, int(idx)),
            "from_bus": int(net.line.at[idx, "from_bus"]),
            "to_bus": int(net.line.at[idx, "to_bus"]),
            "kind": "line",
            "in_service": bool(net.line.at[idx, "in_service"])
            if "in_service" in net.line.columns else True,
            **added("line", idx),
        })
    for idx in net.trafo.index:
        hv, lv = int(net.trafo.at[idx, "hv_bus"]), int(net.trafo.at[idx, "lv_bus"])
        branches.append({
            "index": int(idx),
            "name": _trafo_display_name(net, int(idx)),
            "from_bus": hv,
            "to_bus": lv,
            "kind": "trafo",
            "in_service": bool(net.trafo.at[idx, "in_service"])
            if "in_service" in net.trafo.columns else True,
            **added("trafo", idx),
        })

    external_grids = [
        {
            "name": _clean_label(net.ext_grid.at[idx, "name"]) or f"External grid {position + 1}",
            "bus": int(net.ext_grid.at[idx, "bus"]),
            "in_service": bool(net.ext_grid.at[idx, "in_service"])
            if "in_service" in net.ext_grid.columns else True,
        }
        for position, idx in enumerate(net.ext_grid.index)
    ]

    profile = app_data.get("grid_profile", {}) or {}
    return {
        "name": str(profile.get("name") or getattr(net, "name", "") or "network"),
        **({"network": _proposal_card(_network_name(request))} if proposed else {}),
        "buses": buses,
        "branches": branches,
        "external_grids": external_grids,
        "counts": {
            "buses": len(buses),
            "lines": int(len(net.line)),
            "trafos": int(len(net.trafo)),
            "loads": int(len(net.load)),
            "generators": int(len(net.sgen)) + int(len(net.gen)),
        },
        "voltage_levels": sorted({b["vn_kv"] for b in buses}, reverse=True),
    }


@app.get("/api/grid_constants")
async def get_grid_constants_endpoint():
    """Return grid metadata for the currently loaded network.

    The agent's config.py fetches this on every turn so the system prompt
    stays current after a runtime network upload.
    """
    net = app_data.get("net")
    if net is None:
        raise HTTPException(status_code=503, detail="No network loaded yet.")
    profile = app_data.get("grid_profile", {})
    subs = get_substation_names(net)

    meas_ts = app_data.get("timestamps") or []
    fc_ts = app_data.get("forecast_timestamps") or []
    # A forecast uploaded for a period that was also measured shares its
    # timestamps; the agent is told how many rather than assuming none.
    shared = sorted(set(meas_ts) & set(fc_ts))
    measurements_info = {
        "loaded": bool(meas_ts),
        "n_timestamps": len(meas_ts),
        "first_timestamp": meas_ts[0] if meas_ts else None,
        "last_timestamp": meas_ts[-1] if meas_ts else None,
        "source": app_data.get("measurements_source", "synthetic"),
        "reactive_measured": dq.has_reactive(app_data.get("measurements") or {}),
        "voltage_units_not_following_measured_q": next(
            (f.get("substations") for f in ((_data_quality("measurements") or {}).get("findings") or [])
             if f["code"] == "reactive_at_voltage_units"), []),
    }
    forecast_info = {
        "loaded": bool(fc_ts),
        "n_timestamps": len(fc_ts),
        "first_timestamp": fc_ts[0] if fc_ts else None,
        "last_timestamp": fc_ts[-1] if fc_ts else None,
        "source": app_data.get("forecasts_source", "synthetic"),
        "reactive_measured": dq.has_reactive(app_data.get("forecasts") or {}),
        "n_shared_with_measurements": len(shared),
        "first_shared": shared[0] if shared else None,
        "last_shared": shared[-1] if shared else None,
    }
    return {
        "name": profile.get("name", "Unknown Grid"),
        "scenario": scn.SCENARIOS.get(app_data.get("scenario") or "", {}).get("label"),
        # Buses are every node; substations are the buses with at least one
        # load, plus the bus of any named generator (get_substation_names) —
        # what the data and most tools are keyed by. Both are reported: with
        # only the second, "how many buses?" was answered with 11, not 14.
        "n_buses": len(net.bus),
        "n_substations": len(subs),
        "n_lines": len(net.line),
        "n_trafos": len(net.trafo),
        "substation_names": subs,
        # The network as loaded: the same after the operator renames elements
        # (the agent keys its conversation on it, and must not reset on a rename).
        "network_id": app_data.get("loaded_fingerprint"),
        "names_key": _names_key(),
        "measurements": measurements_info,
        "forecasts": forecast_info,
        "vm_lower": profile.get("vm_lower", 0.9),
        "vm_upper": profile.get("vm_upper", 1.1),
        "max_loading_pct": profile.get("max_loading_pct", 100.0),
        "slack_max_mw_default": profile.get("slack_max_mw", 999.0),
        "slack_max_mw_max": profile.get("slack_max_mw", 999.0),
        "slack_name": profile.get("slack_name", "External Grid"),
        "grid_type": profile.get("grid_type", "unknown"),
        "description": profile.get("description", ""),
        "voltage_level_kv": profile.get("voltage_level_kv", 0.0),
        "load_scaling_max": profile.get("load_scaling_max", 4.0),
        # Network proposals, so the agent knows what it can study.
        "proposals": [{"id": pid, "edits": list(p["summary"]["edits"])}
                      for pid, p in (app_data.get("proposals") or {}).items()],
    }


@app.get("/api/time/current")
async def get_current_time():
    """Return the timestamp the simulation is currently working on.

    Used by the frontend on page load to initialize the Simulation Time
    display before the user has clicked ``Advance 15 Mins``.
    """
    if not app_data["timestamps"]: raise HTTPException(status_code=500, detail="No timestamps loaded.")
    return {"current_timestamp": app_data["timestamps"][app_data["current_index"]]}


@app.get("/api/time/timeline")
async def get_timeline():
    """Return the full measurement timeline plus the forecast horizon.

    Powers the sidebar time scrubber: the measurement timestamps are the
    positions the simulation clock can occupy; the forecast block is the
    read-only look-ahead range that tools query via ``data_source="forecasts"``.
    """
    ts = app_data.get("timestamps") or []
    fc = app_data.get("forecast_timestamps") or []
    idx = app_data.get("current_index", 0)
    return {
        "timestamps": ts,
        "n": len(ts),
        "current_index": idx,
        "current_timestamp": ts[idx] if ts and idx < len(ts) else None,
        "measurements_source": app_data.get("measurements_source", "synthetic"),
        "forecast_timestamps": fc,
        "forecast": {
            "first": fc[0] if fc else None,
            "last": fc[-1] if fc else None,
            "n": len(fc),
            "source": app_data.get("forecasts_source", "synthetic"),
        },
        "nominal": dict(zip(("available", "reason"), nom.usable(app_data.get("nominal")))),
        "reactive_measured": {"measurements": dq.has_reactive(app_data.get("measurements") or {}),
                              "forecasts": dq.has_reactive(app_data.get("forecasts") or {})},
    }


class FollowMeasuredRequest(BaseModel):
    """Substations whose voltage-controlled units should follow the measured
    reactive generation instead (switched to fixed Q)."""
    substations: list[str] = []


@app.post("/api/generators/follow_measured")
async def follow_measured_reactive(request: FollowMeasuredRequest):
    """Switch the voltage-controlled units at these substations to fixed Q, so
    the measured reactive generation is used at every tick that has it; their
    fixed value (the substation's median measured Q) applies where it is
    missing. Condensers stay on voltage control (reported)."""
    dataset = app_data.get("measurements") or {}
    if not dq.has_reactive(dataset)["production"]:
        raise HTTPException(status_code=400, detail="The measured data carry no reactive generation to follow.")
    frame = pd.concat([df[["substation_name", "production_q"]] for df in dataset.values()], ignore_index=True)
    median = frame.groupby("substation_name")["production_q"].median()
    with _STATE_LOCK:
        base, loaded = _net_for(None), app_data.get("net_loaded")
        names = list(substations)
        units = {u["unit"]: u for u in gm.describe(base)}
        wanted = set(request.substations or [])
        changes, skipped = [], []
        for idx in base.gen.index[~lg._proposed_mask(base.gen)] if len(base.gen) else []:
            bus = int(base.gen.at[idx, "bus"])
            sub = sb.measurement_key_for_bus(base, bus, names) or measurement_name(base, bus)
            if sub not in wanted:
                continue
            name = gm.unit_name(base, "gen", idx)
            if units.get(name, {}).get("condenser"):
                skipped.append(f"{name} (a condenser: it stays on voltage control)")
                continue
            q = median.get(sub)
            # The model is keyed by the unit's name as loaded (a renamed unit keeps its model).
            changes.append({"unit": gm.unit_key(base, "gen", idx), "control": "fixed_q",
                            "q_mvar": float(q) if pd.notna(q) else 0.0})
        if not changes:
            raise HTTPException(status_code=400, detail="No voltage-controlled generator at those substations"
                                + (f" ({'; '.join(skipped)})" if skipped else "") + ".")
        try:
            current = app_data.get("generator_model") or {}
            model = gm.merge(current, gm.normalise(loaded, changes, current))
            dropped = _apply_generator_model(model)
        except gm.ModelError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        _save_assumptions()
        out = _generators_payload()
    shown = naming.unit_renames(_net_for(None))
    out.update({"switched": [shown.get(c["unit"], c["unit"]) for c in changes], "skipped": skipped,
                "proposals_dropped": dropped})
    return out


def _template_csv(hours: int = 1) -> str:
    """An upload CSV for the loaded network: its substation names, the loaded
    data's first hour as example values (zeros without data), and the optional
    reactive columns (filled when the loaded data carry them, else empty)."""
    dataset = app_data.get("measurements") or {}
    ts = sorted(dataset)[: max(1, 4 * hours)]
    names = list(dict.fromkeys(str(n) for n in substations))
    if ts:
        names = list(dict.fromkeys(dataset[ts[0]]["substation_name"].astype(str))) or names
    stamps = ts or [f"2026-01-01 {h:02d}:{m:02d}:00" for h in range(hours) for m in (0, 15, 30, 45)]
    rows = ["timestamp,substation_name,consumption_mw,production_mw,consumption_mvar,production_mvar"]

    def fmt(v):
        return "" if v is None or pd.isna(v) else f"{float(v):.4f}"
    for t in stamps:
        frame = dataset[t].set_index(dataset[t]["substation_name"].astype(str)) if ts else None
        for n in names:
            if frame is None or n not in frame.index:
                rows.append(f"{t},{n},0,0,,")
                continue
            r = frame.loc[n]
            rows.append(f"{t},{n},{fmt(r['consumption'])},{fmt(r['production'])},"
                        f"{fmt(r.get('consumption_q'))},{fmt(r.get('production_q'))}")
    return "\n".join(rows) + "\n"


@app.get("/api/data/template")
async def data_template(hours: int = 1):
    """A CSV to start an upload from, for the loaded network: every substation
    name it expects, the loaded data's first hour as example values, and the
    optional reactive columns."""
    hours = min(max(int(hours), 1), 24)
    name = re.sub(r"[^A-Za-z0-9_-]+", "_", str((app_data.get("grid_profile") or {}).get("name") or "network")).strip("_") or "network"
    return Response(_template_csv(hours), media_type="text/csv",
                    headers={"Content-Disposition": f'attachment; filename="conductor_template_{name}.csv"'})


@app.get("/api/data/quality")
async def data_quality():
    """What the loaded datasets look like: reactive columns, coverage, and
    anything implausible (data_quality.py). Never changes the data."""
    return {"measurements": _data_quality("measurements"), "forecasts": _data_quality("forecasts")}


class ScenarioRequest(BaseModel):
    name: str


@app.get("/api/scenario")
async def get_scenario():
    """What is loaded (Resume or a demo) and what can be loaded."""
    return _scenario_payload()


@app.post("/api/scenario")
async def set_scenario(request: ScenarioRequest):
    """Load a scenario. Demos never touch the uploaded network and data, so
    Resume stays available; each network keeps its own saved proposals and
    assumptions. The choice is remembered for the next start."""
    name = (request.name or "").strip()
    if name not in scn.SCENARIOS:
        raise HTTPException(status_code=400, detail=f"Unknown scenario {request.name!r}. Use one of "
                                                    f"{list(scn.SCENARIOS)}.")
    if name == "resume" and not _has_uploads():
        raise HTTPException(status_code=400, detail="Nothing to resume yet: upload a network or data first.")
    with _STATE_LOCK:
        try:
            if name in scn.DEMOS:
                _load_demo(name)
            else:
                _load_resume()
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"Could not load {name}: {exc}")
        _restore_assumptions()
        _restore_proposals()
        _save_scenario(app_data["scenario"])
        _note_where_changes_are_kept()
        return _scenario_payload()


class ForgetRequest(BaseModel):
    """Clear what is saved for a scenario's network: renames, costs, unit
    models, the load power factor and its proposals."""
    name: str


@app.post("/api/scenario/forget")
async def forget_scenario_changes(request: ForgetRequest):
    """Start fresh on a scenario's network. When it is the one loaded, the
    session goes back to the network as loaded too."""
    networks = ((_read_json(_SCENARIO_FILE) if os.path.isfile(_SCENARIO_FILE) else None) or {}).get("networks") or {}
    keys = networks.get((request.name or "").strip())
    if not keys:
        raise HTTPException(status_code=400, detail=f"Nothing is saved for {request.name!r}.")
    with _STATE_LOCK:
        if keys.get("fingerprint") == app_data.get("loaded_fingerprint"):
            app_data["load_power_factor"] = None
            app_data["unit_costs"] = {}
            _apply_generator_model({}, {})
            _discard_proposals()
        for path, key in ((_ASSUMPTIONS_FILE, "fingerprint"), (_PROPOSALS_FILE, "proposals_fingerprint")):
            store = _read_store(path)
            if store.pop(keys.get(key), None) is not None:
                _write_store(path, store)
        return _scenario_payload()


class AssumptionsRequest(BaseModel):
    """Change the session's study assumptions. `load_power_factor` None goes
    back to the default."""
    load_power_factor: float | None = None

    @field_validator("load_power_factor")
    @classmethod
    def _valid(cls, value):
        if value is not None and not (LOAD_PF_MIN <= value <= 1.0):
            raise ValueError(f"load_power_factor must be between {LOAD_PF_MIN} and 1.0 (lagging).")
        return value


def _assumptions_key() -> str:
    """Changes whenever an assumption does: the UI keys its caches on it, so a
    map drawn before a change is never shown after it."""
    payload = {"pf": app_data.get("load_power_factor"), "gm": app_data.get("generator_model") or {},
               "costs": app_data.get("unit_costs") or {}, "names": app_data.get("element_names") or {}}
    return hashlib.sha1(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:12]


def _dataset_key(dataset: dict) -> str:
    """Identifies a loaded series by its content (a few sums), so it survives a
    restart and changes whenever other data are loaded."""
    if not dataset:
        return "none"
    ts = sorted(dataset)
    parts = [len(ts), ts[0], ts[-1]]
    for t in (ts[0], ts[len(ts) // 2], ts[-1]):
        frame = dataset[t]
        parts += [round(float(pd.to_numeric(frame[c], errors="coerce").sum()), 6)
                  for c in ("consumption", "production", "consumption_q", "production_q") if c in frame]
    return hashlib.sha1(json.dumps(parts, default=str).encode()).hexdigest()[:12]


def _data_key() -> str:
    """Changes whenever other measurements or forecasts are loaded: an answer
    stamped with an older key was computed on other data."""
    m, f = app_data.get("measurements") or {}, app_data.get("forecasts") or {}
    marker = (id(m), len(m), id(f), len(f))
    cached = app_data.get("_data_key_cache")
    if not cached or cached[0] != marker:
        cached = (marker, f"{_dataset_key(m)}-{_dataset_key(f)}")
        app_data["_data_key_cache"] = cached
    return cached[1]


def _renamed_measurement_rows() -> dict:
    """{measurement row name: the bus's name now}, for every renamed bus."""
    if app_data.get("net") is None:
        return {}
    base = _net_for(None)
    if "loaded_name" not in base.bus.columns:
        return {}
    return {measurement_name(base, b): _bus_display_name(base, b) for b in base.bus.index
            if isinstance(base.bus.at[b, "loaded_name"], str)}


def _names_key() -> str:
    """Changes whenever the operator renames an element: an answer stamped
    with an older key shows the names it ran with."""
    return hashlib.sha1(json.dumps(app_data.get("element_names") or {}, sort_keys=True)
                        .encode()).hexdigest()[:8]


def _assumptions_payload() -> dict:
    value, source = _load_pf_setting()
    return {"key": _assumptions_key(), "data_key": _data_key(), "names_key": _names_key(),
            "scenario": app_data.get("scenario"),
            "generator_model": app_data.get("generator_model") or {},
            "unit_costs": app_data.get("unit_costs") or {},
            "element_names": (naming.describe(_net_for(None), app_data.get("element_names") or {})
                              if app_data.get("net") is not None else []),
            "reactive_measured": {"measurements": dq.has_reactive(app_data.get("measurements") or {}),
                                  "forecasts": dq.has_reactive(app_data.get("forecasts") or {})},
            "load_power_factor": {
        "value": value, "default": lg.DEFAULT_LOAD_POWER_FACTOR, "source": source, "lagging": True,
        "meaning": "The measurements carry active power only: each measured load's reactive power is "
                   "P·tan(acos pf). Every study uses this unless it is given its own load_power_factor; "
                   "the nominal point keeps the case's own reactive power."}}


@app.get("/api/assumptions")
async def get_assumptions():
    """The session's study assumptions (today: the load power factor)."""
    return _assumptions_payload()


@app.post("/api/assumptions")
async def set_assumptions(request: AssumptionsRequest):
    """Set (or, with null, reset) the session's load power factor. Kept for
    this network across a restart, like a proposal."""
    with _STATE_LOCK:
        before = _load_pf_setting()
        app_data["load_power_factor"] = request.load_power_factor
        _save_assumptions()
        out = _assumptions_payload()
    out["previous"] = {"value": before[0], "source": before[1]}
    return out


class GeneratorModelRequest(BaseModel):
    """Change how units are modelled (generators.py). `changes`: one object per
    unit — unit, and any of control, vm_pu, pf, absorbing, q_min_mvar,
    q_max_mvar, dispatchable. `reset` takes every unit back to how it was
    loaded."""
    changes: list[dict] = []
    reset: bool = False


def _renames_now() -> dict:
    """{name as loaded: name now} for every renamed unit and load."""
    if app_data.get("net") is None:
        return {}
    base = _net_for(None)
    return {**naming.unit_renames(base), **naming.load_renames(base)}


def _by_name_now(keyed: dict) -> dict:
    """A {unit or load name as loaded: …} table under their names now."""
    now = _renames_now()
    return {now.get(k, k): v for k, v in keyed.items()}


def _by_name_loaded(changes: list) -> list:
    """Unit changes named as the operator sees the units now, keyed back to
    their names as loaded (what the model and costs are kept under)."""
    now = _renames_now()
    back = {v.casefold(): k for k, v in now.items()}
    out = []
    for c in changes or []:
        if isinstance(c, dict) and isinstance(c.get("unit"), str) and c["unit"].strip().casefold() in back:
            c = {**c, "unit": back[c["unit"].strip().casefold()]}
        out.append(c)
    return out


def _generators_payload() -> dict:
    base = _net_for(None)
    ok, why = gm.editable(app_data.get("net_loaded") if app_data.get("net_loaded") is not None else base)
    return {
        "editable": ok, "reason": why,
        "units": [{**u, "bus_name": _bus_display_name(base, u["bus"])} for u in gm.describe(base)],
        "changed": _by_name_now(app_data.get("generator_model") or {}),
        "meaning": {
            "voltage": "holds its voltage set-point; its reactive power is what that takes (within "
                       "q_min/q_max in the OPF; power flows do not enforce the limits)",
            "power_factor": "injects its active power at a fixed power factor (injecting, or absorbing "
                            "reactive power); pf 1.0 is no reactive power",
            "dispatchable": "the OPF may move its active power; otherwise it keeps its output",
        },
    }


@app.get("/api/generators")
async def get_generators():
    """Every unit of the loaded network and how it is modelled now."""
    return _generators_payload()


@app.post("/api/generators")
async def set_generators(request: GeneratorModelRequest):
    """Change units' models (or reset them all). Applied to the network as
    loaded; proposals are re-checked; kept for this network across a restart."""
    with _STATE_LOCK:
        loaded = app_data.get("net_loaded")
        if loaded is None:
            raise HTTPException(status_code=503, detail="No network loaded yet.")
        try:
            if request.reset:
                model = {}
            else:
                if not request.changes:
                    raise gm.ModelError("Give at least one unit change, or reset=true.")
                current = app_data.get("generator_model") or {}
                model = gm.merge(current, gm.normalise(loaded, _by_name_loaded(request.changes), current))
            dropped = _apply_generator_model(model)
        except gm.ModelError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        _save_assumptions()
        out = _generators_payload()
    out["proposals_dropped"] = dropped
    return out


class CostsRequest(BaseModel):
    """Change what units cost (costs.py). `changes`: one object per unit —
    unit, and any of cost_eur_per_h, cost_eur_per_mwh, cost_eur_per_mw2h,
    up_eur_per_mwh, down_eur_per_mwh, q_eur_per_mvarh (units), q_band_tan_phi,
    q_penalty_eur_per_mvarh (the external grid) — null = back to the default.
    `reset` takes every unit back to its default."""
    changes: list[dict] = []
    reset: bool = False


def _costs_payload() -> dict:
    loaded = app_data.get("net_loaded")
    if loaded is None:
        raise HTTPException(status_code=503, detail="No network loaded yet.")
    rows = uc.table(loaded, app_data.get("unit_costs") or {})
    loads = uc.load_table(loaded, app_data.get("unit_costs") or {})
    base = _net_for(None)
    now = {**naming.unit_renames(base), **naming.load_renames(base)}
    for r in rows + loads:
        r["unit"] = now.get(r["unit"], r["unit"])
        r["bus_name"] = _bus_display_name(base, r["bus"])
        for k in ("table", "index", "proposed"):
            r.pop(k, None)
    return {
        "units": rows, "changed": _by_name_now(app_data.get("unit_costs") or {}), "currency": "EUR",
        # Per load: demand response and load relief — the remedies after redispatch.
        "loads": [r for r in loads], "loads_rule": uc.LOAD_RULE,
        "synthetic_rule": uc.RULE,
        "used_by": "They price the flexibility options and find the cheapest one; the default corrective "
                   "dispatch (least disturbance) does not use them.",
        "meaning": {
            "cost_eur_per_h / cost_eur_per_mwh / cost_eur_per_mw2h": "production cost c0 + c1·P + c2·P² "
                                                                    "(for the external grid, c1 is the import price)",
            "up_eur_per_mwh": "what each MWh of upward redispatch costs the operator",
            "down_eur_per_mwh": "what each MWh of downward redispatch (curtailment, for renewables) costs",
            "q_eur_per_mvarh": "a unit's price per Mvarh of reactive power it is asked to change, injecting or "
                               "absorbing",
            "q_band_tan_phi / q_penalty_eur_per_mvarh": "external grid: reactive exchange within |Q| ≤ tan φ·|P| "
                                                        "is free; each Mvarh beyond costs the penalty",
            "source": "per figure: 'case' (the network file's gencost), 'synthetic' (the rule) or 'operator'",
            "dr_share_pct / dr_eur_per_mwh": "a load: the share that can be reduced on request (demand response) "
                                             "and its price",
            "relief_eur_per_mwh": "a load: involuntary load relief beyond its demand response, the last resort",
        },
    }


@app.get("/api/costs")
async def get_costs():
    """Every unit's production and redispatch costs, and where each comes from."""
    return _costs_payload()


@app.post("/api/costs")
async def set_costs(request: CostsRequest):
    """Change units' costs (or reset them all). Kept for this network across a restart."""
    with _STATE_LOCK:
        loaded = app_data.get("net_loaded")
        if loaded is None:
            raise HTTPException(status_code=503, detail="No network loaded yet.")
        try:
            if request.reset:
                model = {}
            else:
                current = app_data.get("unit_costs") or {}
                model = uc.merge(current, uc.normalise(loaded, _by_name_loaded(request.changes), current))
        except uc.CostError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        app_data["unit_costs"] = model
        _save_assumptions()
        return _costs_payload()


class NamesRequest(BaseModel):
    """Rename elements (naming.py). `changes`: one object per element —
    element_type (bus, line, trafo, load, gen, sgen), element (any label
    results show, or its index) and name (empty: back to the name as loaded).
    `reset` restores every name as loaded."""
    changes: list[dict] = []
    reset: bool = False


def _names_payload() -> dict:
    base = _net_for(None)
    return {"elements": naming.elements(base),
            "renamed": naming.describe(base, app_data.get("element_names") or {}),
            "meaning": "A rename changes how every result names the element from the next study on. "
                       "Measurements stay matched to the name as loaded; a bus's lines and transformers are "
                       "relabelled with it. Kept for this network across restarts."}


@app.get("/api/names")
async def get_names():
    """Every nameable element: its name now and whether it was renamed."""
    return _names_payload()


@app.post("/api/names")
async def set_names(request: NamesRequest):
    """Rename elements (or restore every name as loaded)."""
    with _STATE_LOCK:
        if app_data.get("net_loaded") is None:
            raise HTTPException(status_code=503, detail="No network loaded yet.")
        before = {(r["element_type"], r["index"]): r["name"]
                  for r in naming.describe(_net_for(None), app_data.get("element_names") or {})}
        try:
            names = {} if request.reset else naming.normalise(
                _net_for(None), request.changes, app_data.get("element_names") or {})
            dropped = _apply_generator_model(app_data.get("generator_model") or {}, names)
        except (naming.RenameError, gm.ModelError) as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        _save_assumptions()
        out = _names_payload()
    out["proposals_dropped"] = dropped
    out["previous"] = [{"element_type": t, "index": i, "name": n} for (t, i), n in before.items()]
    return out


def _settings_snapshot() -> dict:
    """The session's settings as stored — what an answer could change and
    Undo puts back — plus how to name their subjects now."""
    base = _net_for(None) if app_data.get("net") is not None else None
    return {
        "load_power_factor": app_data.get("load_power_factor"),
        "generator_model": copy.deepcopy(app_data.get("generator_model") or {}),
        "unit_costs": copy.deepcopy(app_data.get("unit_costs") or {}),
        "element_names": copy.deepcopy(app_data.get("element_names") or {}),
        # For the note only: units by their names now, renamed elements' names as loaded.
        "unit_names_now": naming.unit_renames(base) if base is not None else {},
        "names_as_loaded": ({f"{r['element_type']}:{r['index']}": r["loaded_name"]
                             for r in naming.describe(base, app_data.get("element_names") or {})}
                            if base is not None else {}),
        "default_load_power_factor": lg.DEFAULT_LOAD_POWER_FACTOR,
    }


@app.get("/api/settings")
async def get_settings():
    """The session's settings (load power factor, generator models, unit
    costs, names) as stored: the app compares them before and after an answer."""
    return _settings_snapshot()


class SettingsRevertRequest(BaseModel):
    """Undo what one answer changed: the settings before it and after it.
    Only the items that differ between the two are put back."""
    before: dict
    after: dict


def _flat_names(model: dict | None) -> dict:
    return {f"{table}:{key}": name for table, names in (model or {}).items() for key, name in (names or {}).items()}


@app.post("/api/settings/revert")
async def revert_settings(request: SettingsRevertRequest):
    """Put back what an answer changed — only if each of those settings is
    still as the answer left it; otherwise nothing changes (409), so a later
    change made by hand is never overwritten."""
    before, after = request.before or {}, request.after or {}
    with _STATE_LOCK:
        if app_data.get("net_loaded") is None:
            raise HTTPException(status_code=503, detail="No network loaded yet.")
        now = _settings_snapshot()
        changed_since, reverted = [], []

        new_pf = now["load_power_factor"]
        if before.get("load_power_factor") != after.get("load_power_factor") \
                and now["load_power_factor"] != before.get("load_power_factor"):    # not already back
            if now["load_power_factor"] != after.get("load_power_factor"):
                changed_since.append("the load power factor")
            else:
                new_pf = before.get("load_power_factor")
                reverted.append("load power factor")

        units_now = now["unit_names_now"]

        def revert_map(key: str, what: str, flat=lambda m: dict(m or {}),
                       shown=lambda item: units_now.get(item, item)) -> dict:
            b, a, cur = flat(before.get(key)), flat(after.get(key)), flat(now[key])
            for item in sorted(set(b) | set(a)):
                if b.get(item) == a.get(item) or cur.get(item) == b.get(item):    # unchanged, or already back
                    continue
                if cur.get(item) != a.get(item):
                    changed_since.append(f"{what} {shown(item)}")
                    continue
                if b.get(item) is None:
                    cur.pop(item, None)
                else:
                    cur[item] = b[item]
                reverted.append(f"{what} {shown(item)}")
            return cur

        new_gm = revert_map("generator_model", "the generator model of")
        new_costs = revert_map("unit_costs", "the costs of")
        new_names_flat = revert_map("element_names", "the name of", flat=_flat_names,
                                    shown=lambda item: item.replace(":", " "))
        if changed_since:
            raise HTTPException(status_code=409, detail=(
                "Not undone: changed again since — " + ", ".join(changed_since)
                + ". Change it in the System tab instead."))
        if not reverted:
            return {"reverted": [], "settings": now, "proposals_dropped": []}

        loaded = app_data["net_loaded"]
        old = {"pf": app_data.get("load_power_factor"), "gm": app_data.get("generator_model"),
               "costs": app_data.get("unit_costs"), "names": app_data.get("element_names")}
        try:
            # Validated as if entered by hand, then applied together.
            model = gm.merge({}, gm.normalise(loaded, [{"unit": u, **d} for u, d in new_gm.items()])) \
                if new_gm else {}
            costs = uc.merge({}, uc.normalise(loaded, [{"unit": u, **d} for u, d in new_costs.items()])) \
                if new_costs else {}
            current_names = _flat_names(app_data.get("element_names"))
            renames = [{"element_type": k.split(":")[0], "element": int(k.split(":")[1]),
                        "name": new_names_flat.get(k) or ""}
                       for k in sorted(set(current_names) | set(new_names_flat))
                       if current_names.get(k) != new_names_flat.get(k)]
            names = (naming.normalise(_net_for(None), renames, app_data.get("element_names") or {})
                     if renames else (app_data.get("element_names") or {}))
            app_data["load_power_factor"] = new_pf
            app_data["unit_costs"] = costs
            dropped = _apply_generator_model(model, names)
        except (gm.ModelError, uc.CostError, naming.RenameError) as exc:
            app_data.update({"load_power_factor": old["pf"], "unit_costs": old["costs"]})
            _apply_generator_model(old["gm"] or {}, old["names"] or {})
            raise HTTPException(status_code=400, detail=f"Could not undo: {exc}")
        _save_assumptions()
        return {"reverted": reverted, "settings": _settings_snapshot(), "proposals_dropped": dropped}


class DataSeriesRequest(BaseModel):
    """What the data viewer asks for: one dataset, system totals, and the
    substations named in `substations` (none by default)."""
    data_source: str = "measurements"
    substations: list[str] = []
    max_points: int = 2000


@app.post("/api/data/series")
@_off_event_loop
async def data_series(request: DataSeriesRequest):
    """The loaded measurement or forecast series, for the UI's data viewer.

    Raw data as it was loaded — consumption and production per substation, no
    power flow. Long series are thinned to about `max_points` evenly spaced
    ticks (a year of 15-minute data is ~35k), and the response says so.
    """
    timestamps, dataset, label = _window_dataset(request.data_source)
    if not timestamps:
        return {"data_source": label, "timestamps": [], "n_total": 0, "available_substations": [],
                "totals": {"consumption": [], "production": []}, "substations": {}}
    stride = max(1, -(-len(timestamps) // max(1, int(request.max_points))))
    picked = timestamps[::stride]
    wanted = set(request.substations or [])
    totals_c, totals_p, totals_qc, totals_qp = [], [], [], []
    per: dict = {name: {"consumption": [], "production": [], "consumption_q": [], "production_q": []}
                 for name in wanted}
    available: set = set()
    for ts in picked:
        df = prepare_measurement_df(dataset[ts])
        totals_c.append(round(float(df["consumption"].sum()), 4))
        totals_p.append(round(float(df["production"].sum()), 4))
        for col, acc in (("consumption_q", totals_qc), ("production_q", totals_qp)):
            if col in df.columns:
                v = df[col].sum(min_count=1)
                acc.append(None if pd.isna(v) else round(float(v), 4))
        by_name = df.set_index("substation_name")
        available.update(str(n) for n in by_name.index)
        for name in wanted:
            row = by_name.loc[name] if name in by_name.index else None
            if isinstance(row, pd.DataFrame):  # a name listed twice: its total
                row = row.sum(numeric_only=True)
            per[name]["consumption"].append(None if row is None else round(float(row["consumption"]), 4))
            per[name]["production"].append(None if row is None else round(float(row["production"]), 4))
            for col in ("consumption_q", "production_q"):
                v = None if row is None or col not in row.index else row[col]
                per[name][col].append(None if v is None or pd.isna(v) else round(float(v), 4))
    source = app_data.get("measurements_source" if label == "measurements" else "forecasts_source")
    return {
        "data_source": label,
        "source": source,
        "timestamps": picked,
        "n_total": len(timestamps),
        "stride": stride,
        "available_substations": sorted(available),
        # {name the data are filed under: the bus's name now} for renamed buses.
        "renamed": _renamed_measurement_rows(),
        "totals": {"consumption": totals_c, "production": totals_p,
                   **({"consumption_q": totals_qc} if totals_qc else {}),
                   **({"production_q": totals_qp} if totals_qp else {})},
        "substations": per,
        "reactive_measured": dq.has_reactive(dataset),
        # The network's own values, drawn as a reference beside the series.
        "nominal": _nominal_for_series(wanted),
    }


def _nominal_for_series(wanted: set) -> dict | None:
    """The nominal point in the data viewer's shape: totals and the wanted
    substations. None when the loaded network has no usable nominal point."""
    if not nom.usable(app_data.get("nominal"))[0]:
        return None
    df = _nominal_measurements().groupby("substation_name")[["consumption", "production"]].sum()
    return {
        "totals": {"consumption": round(float(df["consumption"].sum()), 4),
                   "production": round(float(df["production"].sum()), 4)},
        "substations": {name: {"consumption": round(float(df.at[name, "consumption"]), 4),
                               "production": round(float(df.at[name, "production"]), 4)}
                        for name in wanted if name in df.index},
    }


@app.post("/api/time/advance")
@_off_event_loop
async def advance_time(request: AdvanceTimeRequest = AdvanceTimeRequest()):
    """Advance the simulation clock by one tick (typically 15 minutes).

    If ``request.target_timestamp`` is provided, jump directly to that
    timestamp instead of stepping forward. Prefix matching is supported.
    Otherwise bumps app_data["current_index"] by 1 unless we are already at
    the last available timestamp, in which case we return a ``finished``
    sentinel and leave the cursor in place.
    """
    timestamps = app_data["timestamps"]
    if request.target_timestamp is not None:
        ts = request.target_timestamp
        if ts in timestamps:
            idx = timestamps.index(ts)
        else:
            matches = [i for i, t in enumerate(timestamps) if t.startswith(ts)]
            if not matches:
                raise HTTPException(
                    status_code=404,
                    detail=f"Timestamp '{ts}' not found. Use get_current_timestamp to see the available range.",
                )
            idx = matches[0]
        app_data["current_index"] = idx
        return {"status": "success", "new_timestamp": timestamps[idx]}

    if app_data["current_index"] < len(timestamps) - 1:
        app_data["current_index"] += 1
        return {"status": "success", "new_timestamp": timestamps[app_data["current_index"]]}
    return {"status": "finished", "message": "No more timestamps"}


@app.post("/api/data/upload")
@_off_event_loop
async def upload_timeseries(
    file: UploadFile = File(...),
    kind: str = Form("measurements"),
    overwrite: bool = Form(False),
):
    """Replace either the measurement or the forecast timeseries with a CSV.

    ``kind`` selects the target dataset:
      - ``"measurements"`` (default) → historical actuals that drive the
        simulation clock. Replaces ``app_data["measurements"]`` /
        ``["timestamps"]``, resets ``current_index`` to 0, and applies the
        first timestamp to the live network so RSA reflects the new data.
      - ``"forecasts"`` → read-only look-ahead series used by planning and
        rescheduling tools. Replaces ``app_data["forecasts"]`` /
        ``["forecast_timestamps"]`` only — the live network and the simulation
        clock are left untouched.

    Expected CSV format (long, comma-separated, UTF-8):

        timestamp,substation_name,production_mw,consumption_mw
        2022-01-01 00:00:00,Bus 1,2.5,1.2
        2022-01-01 00:00:00,Bus 2,0.0,3.4
        ...

    All four columns are required.  ``timestamp`` must be parseable by
    ``pandas.to_datetime``; any regular interval is accepted (15 min
    recommended).  ``substation_name`` must match the bus names used by
    the loaded pandapower network (fuzzy-matched by
    ``load_gen_assignment.match_substation``).
    """
    if app_data.get("net") is None:
        raise HTTPException(status_code=503, detail="Network not loaded yet — backend still starting up.")

    kind_norm = (kind or "measurements").strip().lower()
    if kind_norm in ("forecast", "forecasts"):
        kind_norm = "forecasts"
    elif kind_norm in ("measurement", "measurements"):
        kind_norm = "measurements"
    else:
        raise HTTPException(status_code=400, detail=f"Unknown kind {kind!r}. Use 'measurements' or 'forecasts'.")

    # Overwrite guard: if a *user-uploaded* dataset of this kind is already loaded,
    # ask for confirmation before replacing it. Synthetic (auto-generated) data is
    # treated as a placeholder and replaced without prompting.
    if not overwrite:
        src_key = "measurements_source" if kind_norm == "measurements" else "forecasts_source"
        ts_key = "timestamps" if kind_norm == "measurements" else "forecast_timestamps"
        existing_ts = app_data.get(ts_key) or []
        if app_data.get(src_key) == "uploaded" and existing_ts:
            return {
                "status": "confirm_required",
                "kind": kind_norm,
                "message": (
                    f"{kind_norm.capitalize()} data is already loaded from a previous upload "
                    f"({len(existing_ts)} timestamps, {existing_ts[0]} → {existing_ts[-1]}). "
                    f"Uploading will replace it."
                ),
                "existing": {
                    "n_timestamps": len(existing_ts),
                    "first_timestamp": existing_ts[0],
                    "last_timestamp": existing_ts[-1],
                },
            }

    content = await file.read()
    _check_upload_size(content, "The uploaded file")
    # Parsed before anything is written: a file that fails to parse must not
    # replace the one the next restart would load.
    try:
        new_measurements, new_timestamps = _parse_timeseries_csv(content, kind_norm)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {exc}")
    stored_path = _persist_uploaded_bytes(
        DATA_FILES_DIR,
        file.filename,
        content,
        f"uploaded_{kind_norm}",
        ".csv",
    )

    # --- Forecast path: store as read-only look-ahead; never touch live net/clock.
    if kind_norm == "forecasts":
        app_data["forecasts"] = new_measurements
        app_data["forecast_timestamps"] = new_timestamps
        app_data["forecasts_source"] = "uploaded"
        _write_forecast_sentinel(stored_path, source="uploaded")
        return {
            "status": "success",
            "kind": "forecasts",
            "n_timestamps": len(new_timestamps),
            "first_timestamp": new_timestamps[0],
            "last_timestamp": new_timestamps[-1],
            "unmatched_buses": [],
            "stored_path": stored_path,
            "quality": _data_quality("forecasts"),
        }

    # --- Measurement path: drives the simulation clock.
    app_data["measurements"] = new_measurements
    app_data["timestamps"] = new_timestamps
    app_data["current_index"] = 0
    app_data["measurements_source"] = "uploaded"

    # Measurements upload should not imply forecast availability. If the current
    # forecast is only synthetic placeholder data, clear it.
    if app_data.get("forecasts_source") == "synthetic":
        app_data["forecasts"] = {}
        app_data["forecast_timestamps"] = []
        app_data["forecasts_source"] = "none"
        try:
            if os.path.isfile(_LAST_FORECAST_SENTINEL):
                with open(_LAST_FORECAST_SENTINEL, encoding="utf-8") as fh:
                    _fc = json.load(fh)
                if _fc.get("source") == "synthetic":
                    os.remove(_LAST_FORECAST_SENTINEL)
                    print(f"[upload] Cleared synthetic forecast sentinel: {_LAST_FORECAST_SENTINEL}")
        except Exception as exc:
            print(f"[upload] WARNING: Could not clear synthetic forecast sentinel: {exc}")

    # Apply first timestamp to the live network so RSA calls reflect new data.
    try:
        initial_meas = prepare_measurement_df(app_data["measurements"][new_timestamps[0]])
        net = app_data["net"]
        net, unmatched_loads = lg.assign_load_values_from_measurements(net, initial_meas, substations)
        net, unmatched_gens = lg.assign_generators_values_from_measurements(net, initial_meas, substations)
        app_data["net"] = net
    except Exception as exc:
        # Data loaded but network assignment failed — warn but don't block.
        return {
            "status": "partial",
            "kind": "measurements",
            "warning": f"Data loaded but network assignment failed: {exc}",
            "n_timestamps": len(new_timestamps),
            "first_timestamp": new_timestamps[0],
            "last_timestamp": new_timestamps[-1],
            "stored_path": stored_path,
        }

    unmatched = sorted(set(unmatched_loads) | set(unmatched_gens))
    _write_timeseries_sentinel(stored_path, source="uploaded")
    app_data["scenario"] = "resume"
    _save_scenario("resume")
    return {
        "status": "success",
        "kind": "measurements",
        "n_timestamps": len(new_timestamps),
        "first_timestamp": new_timestamps[0],
        "last_timestamp": new_timestamps[-1],
        "unmatched_buses": unmatched,
        "stored_path": stored_path,
        "quality": _data_quality("measurements"),
    }


@app.post("/api/network/upload")
@_off_event_loop
async def upload_network(
    file: UploadFile = File(...),
    convert_gen_to_sgen: bool = Form(True),
    tap_policy: str = Form("current"),
):
    """Replace the in-memory pandapower network with a user-supplied MATPOWER .m file.

    Converts the file with ``pandapower.converter.from_mpc``, runs a base-case
    power flow to initialise the operating point, and builds a minimal
    single-timestamp synthetic measurement series from the network's base-case
    load/generation so that subsequent calls to ``/api/grid/rsa`` (power-flow
    only) and the LLM network-info tools work immediately.

    The admittance database (``db_full``) is rebuilt from the uploaded network;
    contingency optimisation needs no per-outage database (the outaged branch
    is left out of it).

    Returns
    -------
    dict
        ``status``, ``n_buses``, ``n_lines``, ``n_trafos``,
        ``first_timestamp``, ``bus_names`` (first 30), ``note``.
    """
    content = await file.read()
    _check_upload_size(content, "The uploaded file")
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    ext = os.path.splitext(file.filename or "")[1].lower()
    _SUPPORTED_EXTS = {".m", ".json", ".xlsx", ".uct"}
    if ext not in _SUPPORTED_EXTS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format '{ext}'. Accepted: .m  .json  .xlsx  .uct",
        )

    # The converters read from a path, so the upload is parsed from a
    # temporary copy and kept only once it has parsed. Writing it to its final
    # name first meant a broken upload with the active network's filename
    # overwrote that network on disk even though the request returned 400.
    import tempfile
    with tempfile.TemporaryDirectory() as scratch:
        probe_path = os.path.join(scratch, f"upload{ext}")
        with open(probe_path, "wb") as fh:
            fh.write(content)
        try:
            if ext == ".m":
                net = nl.read_matpower(probe_path)
            elif ext == ".json":
                net = pp.from_json(probe_path)
            elif ext == ".xlsx":
                net = pp.from_excel(probe_path)
            elif ext == ".uct":
                from pandapower.converter.ucte.from_ucte import from_ucte
                net = from_ucte(probe_path)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Could not parse {ext} file: {exc}")
    stored_path = _persist_uploaded_bytes(
        SYSTEMS_DIR,
        file.filename,
        content,
        "uploaded_system",
        ext,
    )

    try:
        tap_policy_norm = _normalize_tap_policy(tap_policy)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    _apply_tap_policy(net, tap_policy_norm, context="upload")

    # Optionally convert net.gen (PV-bus) → net.sgen for OPF/flexibility tools.
    gen_conversion_note = ""
    if convert_gen_to_sgen and len(net.gen) > 0:
        net, created_names, kept_condensers = nl.gen_to_sgen(net)
        if created_names or kept_condensers:
            parts = []
            if created_names:
                parts.append(
                    f"{len(created_names)} generator(s) converted to controllable sgen "
                    f"(participate in OPF/flexibility dispatch): "
                    + ", ".join(created_names) + "."
                )
            if kept_condensers:
                parts.append(
                    f"{len(kept_condensers)} synchronous condenser(s) kept as PV buses "
                    f"(provide reactive power for voltage regulation, no real-power dispatch): "
                    + ", ".join(kept_condensers) + "."
                )
            gen_conversion_note = " " + " ".join(parts)
            print(f"[upload] gen→sgen conversion: {created_names}, kept condensers: {kept_condensers}")

    # Run NR power flow to initialise operating point (non-fatal if it fails).
    try:
        pp.runpp(net, algorithm="nr", calculate_voltage_angles=True)
    except Exception:
        pass

    # bus_name_map for the response; handles empty strings from MATPOWER networks.
    bus_name_map = {
        int(idx): (str(row["name"]).strip()
                   if "name" in net.bus.columns and pd.notna(row["name"]) and str(row["name"]).strip()
                   else f"Bus_{idx}")
        for idx, row in net.bus.iterrows()
    }

    # Build synthetic multi-timestamp measurement series (7 days, 15-min resolution).
    import synthetic_timeseries as st
    print(f"[upload] Generating synthetic timeseries ({len(net.load)} load buses, {len(net.sgen)} sgens, {len(net.gen)} gens remaining)...")
    timestamps, measurements = st.generate(net, n_days=7, resolution_min=15, seed=42)
    print(f"[upload] Timeseries ready: {len(timestamps)} timestamps from {timestamps[0]} to {timestamps[-1]}.")

    # A synthetic forecast: the last 2 measured days with a day-ahead-sized
    # error (so forecast and measured can be compared) and the 7 days after
    # (an independent realisation). Measurements are the historical actuals
    # that drive the simulation clock; forecasts are read-only, for planning.
    forecast_timestamps, forecasts = st.forecast(
        net, timestamps, measurements, overlap_days=2, n_days=7, resolution_min=15, seed=137
    )
    print(f"[upload] Forecast ready: {len(forecast_timestamps)} timestamps from "
          f"{forecast_timestamps[0]} to {forecast_timestamps[-1]}.")

    # Replace global app_data atomically.
    net_name = os.path.splitext(file.filename)[0] if file.filename else "Uploaded Network"
    app_data["nominal"] = nom.snapshot(net)
    app_data["net"] = net
    app_data["measurements"] = measurements
    app_data["timestamps"] = timestamps
    app_data["current_index"] = 0
    app_data["forecasts"] = forecasts
    app_data["forecast_timestamps"] = forecast_timestamps
    app_data["measurements_source"] = "synthetic"
    app_data["forecasts_source"] = "synthetic"
    app_data["db_full"] = None
    app_data["admittance_source"] = "generated"
    _discard_proposals()
    # A new network starts from its own saved assumptions and proposals, if it
    # has been loaded before (each network keeps its own entry), else defaults.
    app_data["load_power_factor"] = None
    app_data["scenario"] = "resume"
    _save_scenario("resume")
    # Derive the ext_grid active power cap from the .m file (Pmax of the slack-bus
    # generator).  Fall back to 999 (effectively uncapped) if the column is absent.
    _ext_pmax = None
    if 'max_p_mw' in net.ext_grid.columns:
        _vals = net.ext_grid['max_p_mw'].dropna()
        if len(_vals) > 0 and float(_vals.max()) > 0:
            _ext_pmax = float(_vals.max())
    _slack_cap = _ext_pmax if _ext_pmax is not None else 999.0

    _vm_lower, _vm_upper = _extract_vm_limits(net)
    print(f"[upload] Voltage limits from network: vm_lower={_vm_lower}, vm_upper={_vm_upper}")

    app_data["slack_max_mw"] = _slack_cap
    app_data["default_vm_lower"] = _vm_lower
    app_data["default_vm_upper"] = _vm_upper
    app_data["grid_profile"] = {
        "name": net_name,
        "profile_name": "uploaded_system",
        "source": ext.lstrip("."),
        "tap_policy": tap_policy_norm,
        "measurement_source": "synthetic",
        "vm_lower": _vm_lower,
        "vm_upper": _vm_upper,
        "max_loading_pct": 100.0,
        "slack_max_mw": _slack_cap,
        "slack_name": "External Grid",
        "grid_type": "transmission",
        "description": (
            f"Uploaded MATPOWER network: {file.filename}. "
            f"{len(net.bus)} buses, {len(net.line)} lines, {len(net.trafo)} trafos. "
            "Synthetic 7-day 15-min measurement + forecast series generated on upload."
        ),
        "voltage_level_kv": 0.0,
        "load_scaling_max": 4.0,
    }

    substations.clear()
    substations.extend(get_substation_names(net))
    print(f"[upload] Substation list ({len(substations)} entries): {substations}")
    PG_MAX_DATA.clear()
    PG_MAX_DATA.update(get_pg_max_data(net))
    print(f"[upload] PG_MAX_DATA populated ({len(PG_MAX_DATA)} generators): {PG_MAX_DATA}")
    _remember_loaded(net)

    # Build admittance databases on-the-fly (required for OPF / RSA / N-1)
    print(f"[upload] Building admittance databases ({len(net.line)} lines, {len(net.trafo)} trafos)...")
    t0 = time.perf_counter()
    ybus_note = ""
    try:
        app_data["db_full"] = nl.build_ybus_full(net)
        app_data["admittance_source"] = "generated"
        print(f"[upload] Admittance database built in {time.perf_counter() - t0:.1f}s.")
    except Exception as _ybus_exc:
        print(f"[upload] WARNING: Ybus build failed: {_ybus_exc} — OPF disabled.")
        app_data["admittance_source"] = "generated"
        ybus_note = f" Admittance build failed ({_ybus_exc}); OPF unavailable."

    bus_names = [bus_name_map[i] for i in sorted(bus_name_map.keys())]
    # Separate buses that have loads/gens from pure transit buses so the guide
    # can highlight the ones the user actually needs in their CSV.
    load_buses = set(int(r["bus"]) for _, r in net.load.iterrows())
    gen_buses  = set(int(r["bus"]) for _, r in net.sgen.iterrows()) | \
                 set(int(r["bus"]) for _, r in net.gen.iterrows())
    active_bus_ids = load_buses | gen_buses
    active_bus_names = [bus_name_map[i] for i in sorted(active_bus_ids) if i in bus_name_map]

    # Persist the generated synthetic series to data_files and point the timeseries
    # + forecast sentinels at them (source="synthetic"). This supersedes any
    # previously uploaded CSVs (which referenced the old network's buses) and makes
    # the generated data the dataset restored on the next restart — same model as a
    # user upload, just auto-generated.
    try:
        meas_path = _persist_generated_series(measurements, timestamps, "measurements")
        _write_timeseries_sentinel(meas_path, source="synthetic")
        fc_path = _persist_generated_series(forecasts, forecast_timestamps, "forecast")
        _write_forecast_sentinel(fc_path, source="synthetic")
    except Exception as exc:
        print(f"[upload] WARNING: Could not persist generated series: {exc}")

    # Persist sentinel so this network is restored automatically on next restart.
    _write_upload_sentinel(stored_path, app_data["grid_profile"], convert_gen_to_sgen, tap_policy_norm)

    return {
        "status": "ok",
        "n_buses": len(net.bus),
        "n_lines": len(net.line),
        "n_trafos": len(net.trafo),
        "n_timestamps": len(timestamps),
        "first_timestamp": timestamps[0],
        "n_forecast_timestamps": len(forecast_timestamps),
        "forecast_first_timestamp": forecast_timestamps[0],
        "forecast_last_timestamp": forecast_timestamps[-1],
        "bus_names": bus_names,
        "active_bus_names": active_bus_names,
        "stored_path": stored_path,
        "note": f"Network loaded.{gen_conversion_note}{ybus_note}",
        "restored": {"proposals": _restore_proposals_after_upload()},
        "n_sgens": len(net.sgen),
        "n_gens_remaining": len(net.gen),
        "gen_conversion_applied": convert_gen_to_sgen and bool(gen_conversion_note),
        "tap_policy_applied": tap_policy_norm,
    }


@app.post("/api/network/upload_advanced_admittance")
@_off_event_loop
async def upload_advanced_admittance(
    core_file: UploadFile = File(...),
    meta_file: UploadFile = File(...),
    overwrite: bool = Form(False),
):
    """Upload advanced OPF admittance data (2 CSV files).

    This endpoint is network-scoped and **does not upload timeseries**.
    It only replaces the optimizer admittance databases consumed by OPF-related
    endpoints (flexibility, contingency optimize, KPI evaluate/forecast).

    Required files:
      1) core_file: outage_scope/outage_index/section/from_bus/to_bus/tap/value
      2) meta_file: outage_scope/outage_index/meta_type/from_bus/to_bus/trafo_index/tap/value
    """
    if app_data.get("net") is None:
        raise HTTPException(status_code=503, detail="Network not loaded yet — upload a network first.")

    if not overwrite and app_data.get("admittance_source") == "uploaded_csv":
        return {
            "status": "confirm_required",
            "message": "Advanced admittance data is already uploaded. Uploading again will replace it.",
        }

    core_content = await core_file.read()
    meta_content = await meta_file.read()
    _check_upload_size(core_content, "The admittance core file")
    _check_upload_size(meta_content, "The admittance metadata file")
    if not core_content or not meta_content:
        raise HTTPException(status_code=400, detail="Both core_file and meta_file must be non-empty CSV files.")

    try:
        db_full, db_n1_line, db_n1_trafo, summary = _parse_upload_admittance_csvs(
            core_content,
            meta_content,
            app_data["net"],
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not parse advanced admittance CSVs: {exc}")

    # Reject data built for a different network version *before* installing it —
    # a mismatched upload would otherwise silently corrupt every OPF endpoint.
    _validate_uploaded_admittance(db_full, app_data["net"])

    # Persist uploaded CSVs for traceability/reuse.
    core_path = _persist_uploaded_bytes(DATA_FILES_DIR, core_file.filename, core_content, "uploaded_admittance_core", ".csv")
    meta_path = _persist_uploaded_bytes(DATA_FILES_DIR, meta_file.filename, meta_content, "uploaded_admittance_meta", ".csv")

    # Per-outage entries are accepted for compatibility but not used: an
    # outage is the full database without the outaged branch.
    app_data["db_full"] = db_full
    app_data["_proposal_cache"] = {}
    app_data["admittance_source"] = "uploaded_csv"

    return {
        "status": "success",
        "admittance_source": "uploaded_csv",
        "summary": summary,
        "stored_paths": {
            "core_file": core_path,
            "meta_file": meta_path,
        },
        "note": (
            "Advanced admittance uploaded. This only affects OPF-related endpoints; "
            "RSA and contingency simulation continue to use power-flow calculations from the loaded network. "
            + (f"The {len(db_n1_line) + len(db_n1_trafo)} per-outage entries are not needed and not "
               "used: contingency optimisation leaves the outaged branch out of the full database."
               if (db_n1_line or db_n1_trafo) else "")
        ),
    }


@app.post("/api/network/reset_active")
@_off_event_loop
async def reset_active_network(request: ResetActiveNetworkRequest = ResetActiveNetworkRequest()):
    """Reload active in-memory state from a YAML profile without restarting the backend."""
    profile_name = (request.profile_name or DEFAULT_GRID_PROFILE).strip().lower()
    if not profile_name:
        profile_name = DEFAULT_GRID_PROFILE

    # The reset used to reload the profile and then re-apply the previous
    # network's saved measurements and forecasts — none of whose names match,
    # so every tick silently fell back to the same static load — and left the
    # upload record in place, so the next restart loaded the old network again.
    # Only the records go; the uploaded files themselves stay on disk.
    for record in (_LAST_UPLOAD_SENTINEL, _LAST_TIMESERIES_SENTINEL, _LAST_FORECAST_SENTINEL, _SCENARIO_FILE):
        try:
            os.remove(record)
        except FileNotFoundError:
            pass

    try:
        _lifespan_generic(profile_name, BACKEND_DIR)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to reset active network: {exc}")
    _forget_saved_proposals()
    _forget_saved_assumptions()
    app_data["scenario"] = scn.demo_for_profile(profile_name)

    net = app_data.get("net")
    if net is None:
        raise HTTPException(
            status_code=500,
            detail="Reset endpoint ran but no active network is loaded.",
        )

    ts = app_data.get("timestamps") or []
    fc = app_data.get("forecast_timestamps") or []
    return {
        "status": "success",
        "profile_name": profile_name,
        "n_buses": len(net.bus),
        "n_lines": len(net.line),
        "n_trafos": len(net.trafo),
        "n_timestamps": len(ts),
        "first_timestamp": ts[0] if ts else None,
        "n_forecast_timestamps": len(fc),
        "forecast_first_timestamp": fc[0] if fc else None,
        "forecast_last_timestamp": fc[-1] if fc else None,
        "admittance_source": app_data.get("admittance_source", "generated"),
    }


@app.post("/api/network/snapshot")
@_off_event_loop
async def network_snapshot(request: _TimeseriesRequest = _TimeseriesRequest()):
    """Return the network state at a timestamp.

    Loads and generator set-points come from the measurement-assigned tables.
    Reactive output of voltage-controlled units, the external grid's exchange
    and the losses are results of a power flow, so one is run; they used to be
    reported as 0.0 without it — the agent then told an operator that
    condensers holding their bus voltages produced no reactive power. If the
    power flow does not converge, those fields are None and say so.
    Honors ``data_source`` / ``timestamp`` (defaults to the live measurement clock).
    """
    current_ts = _resolve_tick(request.data_source, request.timestamp)
    meas_df = prepare_measurement_df(_lookup_measurement_df(current_ts, request.data_source))

    net_copy = copy.deepcopy(_net_for(request))
    net_copy, _ = _assign_tick(net_copy, meas_df, current_ts, _dataset_of(request))
    pg_max = _pg_max_for(request, net_copy)
    try:
        pp.runpp(net_copy)
        solved = bool(net_copy.converged)
    except Exception:  # noqa: BLE001 — reported as unknown below
        solved = False

    if not net_copy.ext_grid.empty and "name" in net_copy.ext_grid.columns:
        _raw = net_copy.ext_grid["name"].iloc[0]
        slack_name = str(_raw).strip() if _raw and str(_raw).strip() else "Slack"
    else:
        slack_name = "Slack"

    # --- Generators: read from assigned p_mw / q_mvar columns directly ---
    #
    # Each entry carries the bus it sits on as well as its own name. A unit is
    # named after its substation while a bus is named with its voltage level
    # ("Oscar" against "Oscar A"), so a caller given only the name has
    # to guess which bus is meant — and on this network three units have no
    # bus of the same name at all. The bus is known here; reporting it removes
    # the guess.
    generators = []
    if not net_copy.sgen.empty:
        key_col = "substation_name" if "substation_name" in net_copy.sgen.columns else "name"
        for _, row in net_copy.sgen.iterrows():
            bus_id = int(row["bus"])
            name = str(row.get(key_col) or "").strip() or f"Bus_{bus_id}"
            on = bool(row.get("in_service", True))
            generators.append({
                "name": name,
                "bus": _bus_display_name(net_copy, bus_id),
                "bus_index": bus_id,
                # A unit switched off injects nothing, whatever its measured share.
                "Pg_mw": (_safe_float(row["p_mw"]) or 0.0) if on else 0.0,
                "Qg_mvar": (_safe_float(row.get("q_mvar") or 0.0) or 0.0) if on else 0.0,
                "Pg_max_mw": _safe_float(pg_max[name]) if name in pg_max else None,
                "Pg_min_mw": 0.0,
                **({} if on else {"in_service": False}),
            })
    if not net_copy.gen.empty:
        for gen_idx, row in net_copy.gen.iterrows():
            bus_id = int(row.get("bus", 0))
            raw_name = row.get("name", None)
            name = str(raw_name).strip() if (raw_name and str(raw_name).strip()) else f"Gen_bus{bus_id}"
            on = bool(row.get("in_service", True))
            generators.append({
                "name": name,
                "bus": _bus_display_name(net_copy, bus_id),
                "bus_index": bus_id,
                "Pg_mw": (_safe_float(row["p_mw"]) or 0.0) if on else 0.0,
                **({} if on else {"in_service": False}),
                # Voltage-controlled: its Q is whatever holds the bus voltage.
                "Qg_mvar": (_safe_float(net_copy.res_gen.at[gen_idx, "q_mvar"]) if solved else None) if on else 0.0,
                "vm_setpoint_pu": _safe_float(row.get("vm_pu")),
                "Pg_max_mw": _safe_float(row.get("max_p_mw") or 0.0) or 0.0,
                "Pg_min_mw": _safe_float(row.get("min_p_mw") or 0.0) or 0.0,
            })

    # --- Loads: read from assigned p_mw / q_mvar columns directly ---
    loads = []
    for _, row in net_copy.load.iterrows():
        bus_id = int(row["bus"])
        bus_name = net_copy.bus.at[bus_id, "name"] if "name" in net_copy.bus.columns else ""
        label = str(bus_name).strip() if (bus_name and str(bus_name).strip()) else f"Bus_{bus_id}"
        on = bool(row.get("in_service", True))
        loads.append({
            "bus": label,
            # Bus names are not unique on every network — this one has four
            # buses sharing a name — so the index is reported alongside it.
            "bus_index": bus_id,
            "P_mw": (_safe_float(row["p_mw"]) or 0.0) if on else 0.0,
            "Q_mvar": (_safe_float(row.get("q_mvar") or 0.0) or 0.0) if on else 0.0,
            **({} if on else {"in_service": False}),
        })

    # --- External grid: from the power flow; without one, only the balance ---
    total_gen_mw = round(sum(g["Pg_mw"] for g in generators), 4)
    total_load_mw = round(sum(l["P_mw"] for l in loads), 4)
    if solved and not net_copy.res_ext_grid.empty:
        total_import_mw = round(float(net_copy.res_ext_grid["p_mw"].sum()), 4)
        ext_q = round(float(net_copy.res_ext_grid["q_mvar"].sum()), 4)
        losses = round(float(net_copy.res_line["pl_mw"].sum()
                             + (net_copy.res_trafo["pl_mw"].sum() if len(net_copy.trafo) else 0.0)), 4)
    else:
        # Load minus generation: an estimate that leaves out the losses.
        total_import_mw = round(total_load_mw - total_gen_mw, 4)
        ext_q = None
        losses = None
    ext_grid = {
        "name": slack_name,
        "P_import_mw": total_import_mw,
        "Q_mvar": ext_q,
    }

    return {
        "timestamp": current_ts,
        "data_source": _dataset_label(request),
        "generators": generators,
        "loads": loads,
        "ext_grid": ext_grid,
        "totals": {
            "total_generation_mw": total_gen_mw,
            "total_load_mw": total_load_mw,
            "net_import_mw": total_import_mw,
            # From the power flow; None (never an invented 0.0, which the
            # provenance check would treat as grounded) when it did not converge.
            "total_losses_mw": losses,
        },
        "power_flow_converged": solved,
        **({} if solved else {"note": "The power flow did not converge: reactive output of "
                                      "voltage-controlled units, the external grid's reactive "
                                      "exchange and the losses are unknown; import is load minus "
                                      "generation."}),
        **({"nominal": _nominal_relation(meas_df)} if current_ts == nom.NOMINAL else {}),
    }


def _nominal_relation(nominal_df: pd.DataFrame) -> dict:
    """At the nominal point: what it is, and how the loaded datasets relate to
    it — per substation, the nominal demand against the series' range."""
    out = {"meaning": "The network's own operating point as loaded (the case file's values, as the "
                      "twin models it) — the design point the synthetic data are built from. It has "
                      "no time.",
           "totals": nom.totals(_nominal_snapshot())}
    nominal = nominal_df.set_index("substation_name")
    for label in ("measurements", "forecasts"):
        timestamps, dataset, _ = _resolve_dataset(label) if (
            label == "measurements" or app_data.get("forecast_timestamps")) else ([], {}, label)
        if not timestamps:
            continue
        frames = [prepare_measurement_df(dataset[t])[["substation_name", "consumption", "production"]]
                  .assign(_t=t) for t in timestamps]
        series = pd.concat(frames, ignore_index=True)
        total = series.groupby("_t")[["consumption", "production"]].sum()
        nom_load = float(nominal["consumption"].sum())
        rows = []
        for sub, grp in series.groupby("substation_name"):
            if sub not in nominal.index:
                continue
            n_load = float(nominal.at[sub, "consumption"])
            c = grp["consumption"]
            rows.append({"substation": sub, "nominal_load_mw": round(n_load, 4),
                         "min_mw": round(float(c.min()), 4), "mean_mw": round(float(c.mean()), 4),
                         "max_mw": round(float(c.max()), 4),
                         "share_of_ticks_above_nominal": round(float((c > n_load + 1e-9).mean()), 4)})
        rows.sort(key=lambda r: -abs(r["max_mw"] - r["nominal_load_mw"]))
        out[label] = {
            "source": app_data.get(f"{label}_source"),
            "n_ticks": len(timestamps),
            "total_load_mw": {"nominal": round(nom_load, 4),
                              "min": round(float(total["consumption"].min()), 4),
                              "mean": round(float(total["consumption"].mean()), 4),
                              "max": round(float(total["consumption"].max()), 4)},
            "share_of_ticks_above_nominal_load": round(float((total["consumption"] > nom_load + 1e-9).mean()), 4),
            "by_substation": rows[:15],
        }
    return out


def _sgen_names(net) -> pd.Series:
    """Each static generator's name, as the OPF engine names it.

    `substation_name` exists only on networks whose measurements were matched
    to substations; MATPOWER and pandapower cases (the default profile among
    them) have just `name`, often empty. Indexing `substation_name` directly
    made contingency optimisation, and any `disabled_generators` request, fail
    with a KeyError on those networks. Same fallback as
    `flex_engine._sgen_name`, so the names the agent sees in a dispatch result
    are the names these endpoints match against.
    """
    col = "substation_name" if "substation_name" in net.sgen.columns else "name"

    def _name(idx):
        value = net.sgen.at[idx, col] if col in net.sgen.columns else None
        text = str(value).strip() if value is not None and not pd.isna(value) else ""
        return text or f"sgen_{idx}"

    return pd.Series({idx: _name(idx) for idx in net.sgen.index}, dtype=object)


def _drop_disabled_sgens(net, disabled: list) -> None:
    """Remove the named static generators from `net` (in place)."""
    keep = ~_sgen_names(net).isin(disabled)
    net.sgen = net.sgen[keep.reindex(net.sgen.index).fillna(True).values].reset_index(drop=True)


def _branch_labeller(net):
    """(bus, bus) → the branch's display name, as tables and N-1 results show it."""
    names = {}
    for idx, row in net.line.iterrows():
        names.setdefault(tuple(sorted((int(row.from_bus), int(row.to_bus)))), _line_display_name(net, idx))
    for idx, row in net.trafo.iterrows():
        names.setdefault(tuple(sorted((int(row.hv_bus), int(row.lv_bus)))), _trafo_display_name(net, idx))

    def label(i: int, j: int) -> str:
        return names.get(tuple(sorted((int(i), int(j)))), f"Bus_{i}–Bus_{j}")
    return label


def _with_last_resort(failed: dict, request, ts: str, model, net, net_base, held: dict | None,
                      caps_shown: dict | None, fixed: dict | None, cap: float | None,
                      outage: dict | None = None) -> dict:
    """A failed dispatch — no redispatch of the units clears the violations —
    with the remedies beyond redispatch: the options that also reduce load
    (demand response, then load relief; flex_options.last_resort) and single
    branches whose opening helps (checked by power flow). Asked for options,
    the answer is those options; otherwise the failure stays the answer and
    carries them as `last_resort`. `model`: a clone of the failed model."""
    if model is None or not hasattr(model, "LFlex") or getattr(request, "n1", None):
        return failed
    try:
        if cap is not None:
            _cap_exchange(model, cap, getattr(request, "slack_q_max_mvar", None), fixed)
        rescue = _flexibility_options(request, ts, model, net, net_base, cap if cap is not None else float(
            app_data.get("slack_max_mw", 999.0)), held, outage=outage, caps_shown=caps_shown,
            last_resort={"why": failed.get("message"), "restart": fe._restart})
        switching = _switching_remedies(net, None, request.opf_vm_lower, request.opf_vm_upper,
                                        _network_limits()["max_line_loading_pct"])
    except Exception:  # noqa: BLE001 — the failure is still the answer
        logging.getLogger(__name__).warning("Last resort failed.", exc_info=True)
        return failed
    if switching:
        rescue["switching"] = switching
    if getattr(request, "options", False) and rescue.get("status") == "options":
        return {**rescue, "redispatch_alone": {**rescue.get("redispatch_alone", {}), "status": failed.get("status"),
                                               **({"diagnosis": failed["infeasibility_diagnosis"]}
                                                  if failed.get("infeasibility_diagnosis") else {})}}
    failed["last_resort"] = {k: v for k, v in rescue.items() if k not in ("timestamp", "mode")}
    return failed


def _with_infeasibility_diagnosis(failed: dict, model, net=None) -> dict:
    """A failed OPF result, with the reason when one can be found.

    "Infeasible." alone left the operator — and the agent — to guess. See
    `flex_engine.diagnose_infeasible`: it re-solves with the grid-connection
    voltage free, then with branch limits lifted, and reports what that would
    take. The result stays a failure either way.
    """
    try:
        label = _branch_labeller(net) if net is not None else None
        diagnosis = fe.diagnose_infeasible(model, label) if model is not None else None
    except Exception:  # noqa: BLE001
        diagnosis = None
    if diagnosis:
        failed["infeasibility_diagnosis"] = diagnosis
        failed["message"] = diagnosis["explanation"]
    return failed


def _check_outage_element(element_type: str, element_index: int, net=None) -> None:
    """Refuse an outage of something that is not a line or transformer of
    this network, before any copy of it is touched.

    An unknown index used to be written with `.at[...]`, which silently
    *appended* a garbage row and then failed the power flow with a 500; an
    unknown type reached an unbound variable in contingency optimisation.
    """
    if element_type not in ("line", "trafo"):
        raise HTTPException(
            status_code=400,
            detail=f"element_type must be 'line' or 'trafo' for an outage; got {element_type!r}.",
        )
    net = _net_for() if net is None else net
    table = net.line if element_type == "line" else net.trafo
    if element_index not in table.index:
        valid = [int(i) for i in table.index]
        raise HTTPException(
            status_code=404,
            detail=f"No {element_type} with index {element_index}. Valid indices: "
                   f"{valid[:40]}{' …' if len(valid) > 40 else ''}. "
                   "Use locate_network_element to find an index by name.",
        )


_WORST_CASE_METRICS = frozenset(
    {"violations", "slack_import", "max_voltage", "min_voltage", "max_loading"})


def _dataset_label(request) -> str:
    """The dataset a result was computed from, normalised ("measurements" or
    "forecasts") so the agent's consistency check compares like with like."""
    return _resolve_dataset(request.data_source)[2]


def _thresholds_used(request) -> dict[str, float]:
    """The limits a result was judged against, echoed so the agent's
    consistency check can compare them across the tools of one turn."""
    return {
        "vm_upper_pu": request.vm_upper_pu,
        "vm_lower_pu": request.vm_lower_pu,
        "max_line_loading_pct": request.max_line_loading_pct,
        "max_trafo_loading_pct": request.max_trafo_loading_pct,
    }


def _effective_thresholds(request) -> tuple[float, float, float, float]:
    """(vm_lower, vm_upper, max_line_loading, max_trafo_loading) for a request.

    The request already holds the effective values — anything the caller did
    not send was filled with the network's limits when it was validated.
    """
    return (request.vm_lower_pu, request.vm_upper_pu,
            request.max_line_loading_pct, request.max_trafo_loading_pct)


def _voltage_controlled_buses(net) -> list[dict]:
    """Buses whose voltage a unit holds at a setpoint, and that unit's margin.

    In a power flow such a bus sits exactly at its setpoint whatever the load,
    so its distance to a voltage limit is not a margin: 0.942 p.u. on a
    0.94 floor read as "close to the limit" at every hour of the week. The
    margin is the unit's reactive headroom — how much more it can supply
    before it stops holding the voltage. The external grid holds its bus too,
    with no reactive limit in the model.
    """
    out = []
    if len(net.gen) and len(net.res_gen):
        for idx, row in net.gen.iterrows():
            if not bool(row.get("in_service", True)) or idx not in net.res_gen.index:
                continue
            q = _safe_float(net.res_gen.at[idx, "q_mvar"])
            q_max = _safe_float(row.get("max_q_mvar"))
            q_min = _safe_float(row.get("min_q_mvar"))
            bus = int(row["bus"])
            raw = row.get("name")
            out.append({
                "bus": _bus_display_name(net, bus), "bus_index": bus,
                "unit": str(raw).strip() if raw and str(raw).strip() else f"Gen_bus{bus}",
                "setpoint_pu": _safe_float(row.get("vm_pu")),
                "q_mvar": q, "q_max_mvar": q_max, "q_min_mvar": q_min,
                "q_headroom_up_mvar": round(q_max - q, 3) if q is not None and q_max is not None else None,
                "q_headroom_down_mvar": round(q - q_min, 3) if q is not None and q_min is not None else None,
            })
    for idx, row in net.ext_grid.iterrows():
        if not bool(row.get("in_service", True)):
            continue
        bus = int(row["bus"])
        out.append({
            "bus": _bus_display_name(net, bus), "bus_index": bus, "unit": "external grid",
            "setpoint_pu": _safe_float(row.get("vm_pu")),
            "q_mvar": _safe_float(net.res_ext_grid.at[idx, "q_mvar"]) if idx in net.res_ext_grid.index else None,
            "q_max_mvar": None, "q_min_mvar": None,
            "q_headroom_up_mvar": None, "q_headroom_down_mvar": None,
        })
    return out


def _prepared_net(timestamp: str, request, applied: tuple | None = None):
    """The network of a study at one moment: its measurements, the load and
    renewable scaling asked for, then — `applied`, (option set, option[, held])
    — a corrective option's outage and dispatch. Returns (net, pg_max)."""
    net_copy = copy.deepcopy(_net_for(request))
    measurement_prod_cons = prepare_measurement_df(_lookup_measurement_df(timestamp, request.data_source))

    net_copy, _ = _assign_tick(net_copy, measurement_prod_cons, timestamp, _dataset_of(request))
    pg_max = _pg_max_for(request, net_copy)
    _scale_loads(net_copy, request.load_scaling_factor)
    if getattr(request, "sgen_scaling_factor", 1.0) != 1.0:
        net_copy.sgen['p_mw'] *= request.sgen_scaling_factor
        net_copy.sgen['q_mvar'] *= request.sgen_scaling_factor
    if applied:
        stored, option, *rest = applied
        _apply_stored_option(net_copy, stored, option, held=bool(rest and rest[0]))
    return net_copy, pg_max


def _run_rsa_snapshot(timestamp: str, request: GridRequest, applied: tuple | None = None) -> dict:
    """Run the existing RSA pipeline for one explicit timestamp. `applied`:
    (option set, option[, held]) — its dispatch (and outage) applied first."""
    net_copy, pg_max = _prepared_net(timestamp, request, applied)

    _vlo, _vhi, _mll, _mtl = _effective_thresholds(request)
    net_calculated, df_results_rsa = rs.real_time_security_assessment(
        net_copy, timestamp,
        vm_lower=_vlo,
        vm_upper=_vhi,
        max_line_loading_pct=_mll,
        max_trafo_loading_pct=_mtl,
    )

    # Guard against serialization crash when power flow did not converge.
    # pandapower leaves NaN values in res_bus after a failed run; json.dumps
    # rejects NaN floats with ValueError.
    if net_calculated.res_bus.empty or net_calculated.res_bus["vm_pu"].isna().all():
        raise HTTPException(
            status_code=503,
            detail=(
                f"Power flow did not converge at timestamp {timestamp}. "
                "The grid state may be infeasible at this operating point. "
                "Try advancing to the next timestamp."
            ),
        )

    if isinstance(df_results_rsa, list): df_results_rsa = pd.DataFrame(df_results_rsa)

    violations_json = []
    if not df_results_rsa.empty:
        df_results_rsa["timestamp"] = df_results_rsa["timestamp"].astype(str)
        violations_json = df_results_rsa.to_dict(orient="records")
        # Each record here is a violation by construction. Saying so in the
        # record itself costs a field and lets a reader — a guard, or the
        # model — judge a sentence about this element without re-deriving the
        # verdict from a value and a limit it may not have both of.
        for _entry in violations_json:
            _entry["violated"] = True
        # Phase 3 Placeholder: publish_rsa_results(df_results_rsa, current_ts, topic="topic6907"...)

    df_bus = net_calculated.res_bus.reset_index()
    df_bus["bus_name"] = [
        (str(net_calculated.bus.at[i, "name"]).strip()
         if "name" in net_calculated.bus.columns
         and pd.notna(net_calculated.bus.at[i, "name"])
         and str(net_calculated.bus.at[i, "name"]).strip()
         else f"Bus_{i}")
        for i in df_bus["index"]
    ]

    # Exclude buses not in service and rows with non-finite vm_pu from the
    # chart payload. This keeps RSA visuals aligned with the active network.
    bus_in_service_map = net_calculated.bus.get("in_service", pd.Series(True, index=net_calculated.bus.index)).to_dict()
    all_voltages = []
    for _, row in df_bus.iterrows():
        bus_idx = int(row["index"])
        if not bool(bus_in_service_map.get(bus_idx, True)):
            continue
        vm = _safe_float(row["vm_pu"])
        if vm is None:
            continue
        all_voltages.append({"bus_name": row["bus_name"], "vm_pu": vm})

    df_line = net_calculated.res_line.reset_index()
    df_line["line_name"] = [_line_display_name(net_calculated, int(i)) for i in df_line["index"]]
    # An open branch carries nothing; it is marked so a 0 % is never read as
    # a lightly loaded line (only out-of-service rows carry the field).
    def _off(table, idx) -> dict:
        return {} if bool(table.at[idx, "in_service"]) else {"in_service": False}

    all_line_loading = [
        {"line_name": row["line_name"], "loading_percent": _safe_float(row["loading_percent"]),
         **_off(net_calculated.line, int(row["index"]))}
        for _, row in df_line.iterrows()
    ]

    df_trafo = net_calculated.res_trafo.reset_index()
    df_trafo["trafo_name"] = [_trafo_display_name(net_calculated, int(i)) for i in df_trafo["index"]]
    all_trafo_loading = [
        {"trafo_name": row["trafo_name"], "loading_percent": _safe_float(row["loading_percent"]),
         **_off(net_calculated.trafo, int(row["index"]))}
        for _, row in df_trafo.iterrows()
    ]

    df_sgen = net_calculated.res_sgen.reset_index()
    # Named as the OPF and the snapshot name them. Without `substation_name`
    # (every PGLib case) units used to be called "0", "1" here — so no name
    # matched the Pmax table and the headroom below was always 0.
    df_sgen["gen_name"] = _sgen_names(net_calculated).reindex(df_sgen["index"]).values
    in_service = net_calculated.sgen["in_service"].astype(bool).reindex(df_sgen["index"]).values
    df_sgen = df_sgen[in_service]

    total_available_flex_mw = sum(
        max(0, pg_max.get(row["gen_name"], row["p_mw"]) - row["p_mw"]) for _, row in df_sgen.iterrows())

    gen_dispatch = [
        {"name": row["gen_name"], "Pg_mw": _safe_float(row["p_mw"]), "Qg_mvar": _safe_float(row["q_mvar"])}
        for _, row in df_sgen.iterrows()
    ]
    total_load_mw = _safe_float(net_calculated.res_load["p_mw"].sum()) if not net_calculated.res_load.empty else 0.0
    slack_import_mw = None
    if not net_calculated.res_ext_grid.empty:
        _ext_raw = (
            net_calculated.ext_grid["name"].iloc[0]
            if "name" in net_calculated.ext_grid.columns
            else None
        )
        ext_name = (
            str(_ext_raw).strip()
            if _ext_raw is not None and pd.notna(_ext_raw) and str(_ext_raw).strip()
            else "Slack"
        )
        gen_dispatch.append({
            "name": ext_name,
            "Pg_mw": _safe_float(net_calculated.res_ext_grid["p_mw"].iloc[0]),
            "Qg_mvar": _safe_float(net_calculated.res_ext_grid["q_mvar"].iloc[0]),
        })
        slack_import_mw = _safe_float(net_calculated.res_ext_grid["p_mw"].iloc[0])

    return {
        "timestamp": timestamp, "total_violations": len(violations_json),
        "voltage_controlled_buses": _voltage_controlled_buses(net_calculated),
        # The engine's own verdict, stated rather than implied. An answer that
        # calls this operating point secure can then be checked against it
        # instead of against a violation count the prose may never mention.
        "secure": len(violations_json) == 0,
        "converged": True,
        "violations": violations_json, "all_voltages": all_voltages,
        "all_line_loading": all_line_loading, "all_trafo_loading": all_trafo_loading,
        "gen_dispatch": gen_dispatch,
        "total_load_mw": total_load_mw,
        "slack_import_mw": slack_import_mw,
        "flexibility_metrics": {"total_available_capacity_mw": round(total_available_flex_mw, 3)},
        "data_source": _dataset_label(request),
        "thresholds_used": {
            "vm_upper_pu": _vhi,
            "vm_lower_pu": _vlo,
            "max_line_loading_pct": _mll,
            "max_trafo_loading_pct": _mtl,
        },
    }


def _lookup_measurement_df(ts: str, data_source: str | None = None):
    """Return the measurement slice for a timestamp.

    With `data_source`, only that dataset is read. Without it, measurements
    are tried before forecasts — which is only safe while the two time ranges
    are disjoint. Nothing enforces that: an uploaded forecast for days already
    measured is the obvious way to compare a forecast with what happened, and
    the old lookup then answered every `data_source="forecasts"` request with
    the measurements, while tools reading the dataset directly used the
    forecast — two answers in one turn, both labelled as the forecast.
    """
    if data_source is not None:
        _timestamps, dataset, label = _resolve_dataset(data_source)
        if ts in dataset:
            return dataset[ts]
        raise HTTPException(status_code=404, detail=f"Timestamp {ts!r} not found in {label}.")
    meas = app_data.get("measurements", {}) or {}
    if ts in meas:
        return meas[ts]
    fc = app_data.get("forecasts", {}) or {}
    if ts in fc:
        return fc[ts]
    raise HTTPException(status_code=404, detail=f"Timestamp {ts!r} not found in measurements or forecasts.")


def _resolve_tick(data_source: str = "measurements", requested_ts: str | None = None) -> str:
    """Resolve the single timestamp a point-in-time tool should operate on.

    Honors the LLM-chosen ``data_source``:
      - ``"measurements"`` → defaults to the live simulation clock
        (``current_index``); an explicit ``requested_ts`` is prefix-matched
        within the measurement series.
      - ``"forecasts"`` → has no clock, so defaults to the first forecast tick;
        an explicit ``requested_ts`` is prefix-matched within the forecast series.

    Raises HTTP 400 if forecasts are requested but none are loaded.
    """
    timestamps, _dataset, label = _resolve_dataset(data_source)
    if label == nom.NOMINAL:
        if requested_ts is not None and str(requested_ts).strip() not in ("", nom.NOMINAL):
            raise HTTPException(
                status_code=400,
                detail="The nominal operating point has no time: leave timestamp out with "
                       "data_source='nominal', or use 'measurements' / 'forecasts' for a moment.")
        return nom.NOMINAL
    if requested_ts is not None and str(requested_ts).strip():
        return _resolve_window_timestamp(requested_ts, "timestamp", timestamps=timestamps)
    idx = app_data["current_index"] if label == "measurements" else 0
    if not timestamps:
        raise HTTPException(status_code=500, detail=f"No {label} timestamps loaded.")
    return timestamps[min(idx, len(timestamps) - 1)]


def _resolve_window_timestamp(requested_ts: str | None, label: str, timestamps: list | None = None) -> str | None:
    if requested_ts is None:
        return None
    needle = str(requested_ts).strip()
    if not needle:
        return None
    if timestamps is None:
        timestamps = app_data.get("timestamps", []) or []
    exact_match = next((ts for ts in timestamps if ts == needle), None)
    if exact_match is not None:
        return exact_match
    prefix_matches = [ts for ts in timestamps if str(ts).startswith(needle)]
    if len(prefix_matches) == 1:
        return prefix_matches[0]
    if len(prefix_matches) > 1:
        raise HTTPException(
            status_code=400,
            detail=f"{label} '{needle}' is ambiguous; provide a longer timestamp prefix.",
        )
    raise HTTPException(
        status_code=404,
        detail=f"{label} '{needle}' was not found in the available timestamps.",
    )


def _parse_historical_target(target: str) -> tuple[str, str | None]:
    raw_target = str(target or "all").strip()
    if not raw_target or raw_target.lower() == "all":
        return "all", None
    if ":" in raw_target:
        kind, identifier = raw_target.split(":", 1)
        kind = kind.strip().lower()
        identifier = identifier.strip()
        if kind in {"bus", "line", "trafo"} and identifier:
            return kind, identifier
    raise HTTPException(
        status_code=400,
        detail="target must be 'all' or one of 'bus:<name>', 'line:<name>', 'trafo:<name>'.",
    )


def _extract_historical_metric(snapshot: dict, target_kind: str, target_name: str | None, request: HistoricalRiskRequest) -> dict:
    if target_kind == "all":
        total_violations = int(snapshot.get("total_violations", 0) or 0)
        return {
            "timestamp": snapshot.get("timestamp"),
            "target_label": "System-wide",
            "units": "violation_count",
            "value": float(total_violations),
            "limit": 0.0,
            "severity": float(total_violations),
            "is_violation": total_violations > 0,
            "near_miss": False,
        }

    if target_kind == "bus":
        for row in snapshot.get("all_voltages", []):
            if str(row.get("bus_name", "")).strip() != target_name:
                continue
            vm_pu = _safe_float(row.get("vm_pu"))
            upper_margin = vm_pu - request.vm_upper_pu
            lower_margin = request.vm_lower_pu - vm_pu
            severity = max(upper_margin, lower_margin)
            near_miss = (not severity > 0.0) and (severity >= -abs(request.near_miss_band))
            return {
                "timestamp": snapshot.get("timestamp"),
                "target_label": target_name,
                "units": "pu_margin",
                "value": vm_pu,
                "limit": 0.0,
                "severity": float(severity),
                "is_violation": severity > 0.0,
                "near_miss": near_miss,
            }
        raise HTTPException(status_code=404, detail=f"Bus '{target_name}' was not found in RSA output.")

    collection_key = "all_line_loading" if target_kind == "line" else "all_trafo_loading"
    name_key = "line_name" if target_kind == "line" else "trafo_name"
    loading_limit = request.max_line_loading_pct if target_kind == "line" else request.max_trafo_loading_pct
    for row in snapshot.get(collection_key, []):
        if str(row.get(name_key, "")).strip() != target_name:
            continue
        loading = _safe_float(row.get("loading_percent"))
        severity = loading - loading_limit
        near_miss = (severity <= 0.0) and (severity >= -5.0)
        return {
            "timestamp": snapshot.get("timestamp"),
            "target_label": target_name,
            "units": "pct_margin",
            "value": loading,
            "limit": float(loading_limit),
            "severity": float(severity),
            "is_violation": severity > 0.0,
            "near_miss": near_miss,
        }
    entity_name = "Line" if target_kind == "line" else "Transformer"
    raise HTTPException(status_code=404, detail=f"{entity_name} '{target_name}' was not found in RSA output.")


def _extract_condition_value(snapshot: dict, timestamp: str, condition: str | None) -> float | int | None:
    cond = str(condition or "").strip().lower()
    if not cond:
        return None

    if cond == "hour":
        try:
            return int(pd.to_datetime(timestamp).hour)
        except Exception:
            return None

    if cond in {"load", "load_mw", "total_load"}:
        value = _safe_float(snapshot.get("total_load_mw"))
        return value if value is not None else None

    if cond in {"slack", "slack_import", "slack_import_mw", "cable_flow"}:
        value = _safe_float(snapshot.get("slack_import_mw"))
        return value if value is not None else None

    return None


def _build_conditional_bins(
    metric_rows: list[dict],
    condition_rows: list[float | int | None],
    condition: str,
    n_bins: int,
) -> list[dict]:
    try:
        cond = str(condition).strip().lower()
        paired_rows = [
            (row, cond_val)
            for row, cond_val in zip(metric_rows, condition_rows)
            if cond_val is not None
        ]
        if not paired_rows:
            return []

        if cond == "hour":
            grouped: dict[int, list[dict]] = {}
            for row, cond_val in paired_rows:
                hour = int(cond_val)
                grouped.setdefault(hour, []).append(row)

            bins = []
            for hour in sorted(grouped.keys()):
                rows = grouped[hour]
                n_steps = len(rows)
                exceedance_count = sum(1 for r in rows if r.get("is_violation"))
                near_miss_count = sum(1 for r in rows if r.get("near_miss"))
                bins.append({
                    "label": f"{hour:02d}:00-{hour:02d}:59",
                    "range_low": float(hour),
                    "range_high": float(hour),
                    "n_timesteps": n_steps,
                    "exceedance_frequency": (exceedance_count / n_steps) if n_steps else 0.0,
                    "near_miss_frequency": (near_miss_count / n_steps) if n_steps else 0.0,
                })
            return bins

        numeric_values = np.array([float(cond_val) for _, cond_val in paired_rows], dtype=float)
        if numeric_values.size == 0:
            return []

        quantiles = np.linspace(0.0, 1.0, max(2, int(n_bins) + 1))
        raw_edges = np.quantile(numeric_values, quantiles)

        # Collapse duplicate edges if the conditioning signal has plateaus.
        edges: list[float] = []
        for edge in raw_edges:
            edge = float(edge)
            if not edges or abs(edge - edges[-1]) > 1e-12:
                edges.append(edge)

        if len(edges) < 2:
            rows = [row for row, _ in paired_rows]
            n_steps = len(rows)
            exceedance_count = sum(1 for r in rows if r.get("is_violation"))
            near_miss_count = sum(1 for r in rows if r.get("near_miss"))
            val = float(edges[0]) if edges else 0.0
            return [{
                "label": f"{cond}={val:.3f}",
                "range_low": val,
                "range_high": val,
                "n_timesteps": n_steps,
                "exceedance_frequency": (exceedance_count / n_steps) if n_steps else 0.0,
                "near_miss_frequency": (near_miss_count / n_steps) if n_steps else 0.0,
            }]

        row_bins: list[list[dict]] = [[] for _ in range(len(edges) - 1)]
        for row, cond_val in paired_rows:
            value = float(cond_val)
            idx = int(np.searchsorted(edges, value, side="right") - 1)
            idx = max(0, min(idx, len(edges) - 2))
            row_bins[idx].append(row)

        bins = []
        for idx, rows in enumerate(row_bins):
            if not rows:
                continue
            low = float(edges[idx])
            high = float(edges[idx + 1])
            n_steps = len(rows)
            exceedance_count = sum(1 for r in rows if r.get("is_violation"))
            near_miss_count = sum(1 for r in rows if r.get("near_miss"))
            bins.append({
                "label": f"[{low:.3f}, {high:.3f}]",
                "range_low": low,
                "range_high": high,
                "n_timesteps": n_steps,
                "exceedance_frequency": (exceedance_count / n_steps) if n_steps else 0.0,
                "near_miss_frequency": (near_miss_count / n_steps) if n_steps else 0.0,
            })

        return bins
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"conditional binning failed ({cond}): {type(exc).__name__}: {exc}",
        )


@app.post("/api/grid/rsa")
@_off_event_loop
async def run_security_assessment(request: GridRequest):
    """Real-Time Security Assessment on the current timestamp.

    Pipeline
    1. Deep-copy the base network so this request is isolated.
    2. Assign load + generator values from the current timestamp's
       measurement slice.
    3. Apply the UI's ``load_scaling_factor`` to
       every load's P and Q.
    4. Hand the scaled network to ``rsa_engine.real_time_security_assessment``
       which runs ``pp.runpp`` under the hood and flags any voltage /
       thermal violations.
    5. Package the result in a json dictionary so the frontend can render:
       * total violation count,
       * a table of violation rows,
       * a scatter of per-bus voltages, and
       * the total available upward flexibility (MW) capped by
         :data:`PG_MAX_DATA`.

    Note on transformer taps
    ------------------------
    Every endpoint — assessments and optimisations alike — uses the taps of
    the network as loaded. The OPF's equations always did (its admittance
    database is built from them), but its endpoints reset taps to neutral for
    the power flow that sets the starting point, so the base dispatch and base
    voltages they reported came from a grid the assessments never showed
    (review 2.2).
    """
    if request.option:
        # The assessment with a corrective option applied — at its moment, or
        # held to a later one.
        stored, option, request, held = _option_context(request, later=True)
        current_ts = _resolve_tick(request.data_source, request.timestamp)
        result = _run_rsa_snapshot(current_ts, request, applied=(stored, option, held))
        return {**result, "option": _option_echo(stored, option, current_ts, held)}
    current_ts = _resolve_tick(request.data_source, request.timestamp)
    return _run_rsa_snapshot(current_ts, request)


@app.post("/api/rsa/historical_risk")
@_off_event_loop
async def compute_historical_risk(request: HistoricalRiskRequest):
    """Empirical risk statistics over a historical window.

    First slice: core exceedance / near-miss stats plus a duration curve over
    a signed limit-margin metric. Conditional binning is deferred.
    """
    if not 0.0 <= request.near_miss_band <= 1.0:
        raise HTTPException(status_code=400, detail="near_miss_band must be within [0, 1].")
    if request.worst_n < 1:
        raise HTTPException(status_code=400, detail="worst_n must be at least 1.")
    if request.n_bins < 1:
        raise HTTPException(status_code=400, detail="n_bins must be at least 1.")
    if request.max_workers is not None and request.max_workers < 1:
        raise HTTPException(status_code=400, detail="max_workers must be at least 1 when provided.")

    condition_aliases = {
        "": None,
        "none": None,
        "hour": "hour",
        "load": "load",
        "load_mw": "load",
        "total_load": "load",
        "slack": "slack_import",
        "slack_import": "slack_import",
        "slack_import_mw": "slack_import",
        "cable_flow": "slack_import",
    }
    normalized_condition = condition_aliases.get(str(request.condition or "").strip().lower())
    if str(request.condition or "").strip().lower() not in condition_aliases:
        raise HTTPException(
            status_code=400,
            detail=(
                "condition must be one of: none, hour, load (aliases: load_mw, total_load), "
                "slack_import (aliases: slack, slack_import_mw, cable_flow)."
            ),
        )

    (request.vm_lower_pu, request.vm_upper_pu,
     request.max_line_loading_pct, request.max_trafo_loading_pct) = _effective_thresholds(request)
    timestamps, measurements, source_label = _window_dataset(request.data_source)
    grid_profile = app_data.get("grid_profile", {}) or {}
    system_name = str(grid_profile.get("name") or grid_profile.get("grid_type") or "active system")
    if not timestamps or not measurements:
        return {
            "error": (
                f"compute_historical_risk requires an attached {source_label} time series; "
                f"the active system '{system_name}' does not expose {source_label}."
            ),
            "system_name": system_name,
            "target": request.target,
        }

    start_ts = _resolve_window_timestamp(request.window_start, "window_start", timestamps=timestamps) or timestamps[0]
    end_ts = _resolve_window_timestamp(request.window_end, "window_end", timestamps=timestamps) or timestamps[-1]
    start_idx = timestamps.index(start_ts)
    end_idx = timestamps.index(end_ts)
    if end_idx < start_idx:
        raise HTTPException(status_code=400, detail="window_end must be at or after window_start.")

    selected_timestamps = timestamps[start_idx : end_idx + 1]
    if not selected_timestamps:
        raise HTTPException(status_code=400, detail="Requested window is empty.")

    target_kind, target_name = _parse_historical_target(request.target)
    rsa_request = GridRequest(
        data_source=request.data_source,
        network=request.network,
        load_scaling_factor=request.load_scaling_factor,
        vm_upper_pu=request.vm_upper_pu,
        vm_lower_pu=request.vm_lower_pu,
        max_line_loading_pct=request.max_line_loading_pct,
        max_trafo_loading_pct=request.max_trafo_loading_pct,
    )

    use_parallel = bool(request.parallel) and len(selected_timestamps) > 1
    max_workers_default = min(8, os.cpu_count() or 4)
    effective_max_workers = min(
        len(selected_timestamps),
        request.max_workers if request.max_workers is not None else max_workers_default,
    )

    metric_rows: list[dict] = []
    condition_rows: list[float | int | None] = []
    # One tick whose power flow does not converge used to fail the whole
    # window with a 503 — the opposite of the other scans, which dropped such
    # ticks silently. Both were wrong: the tick is recorded and reported, and
    # the window goes on.
    non_converged: list[str] = []

    def _snapshot(timestamp):
        try:
            return _run_rsa_snapshot(timestamp, rsa_request)
        except HTTPException as exc:
            if exc.status_code == 503:
                return None
            raise

    if use_parallel:
        indexed_rows: list[tuple[int, dict, float | int | None]] = []
        with ThreadPoolExecutor(max_workers=effective_max_workers) as pool:
            future_map = {
                pool.submit(_snapshot, timestamp): idx
                for idx, timestamp in enumerate(selected_timestamps)
            }
            for future in as_completed(future_map):
                idx = future_map[future]
                snapshot = future.result()
                timestamp = selected_timestamps[idx]
                if snapshot is None:
                    non_converged.append(timestamp)
                    continue
                metric_row = _extract_historical_metric(snapshot, target_kind, target_name, request)
                cond_val = _extract_condition_value(snapshot, timestamp, normalized_condition)
                indexed_rows.append((idx, metric_row, cond_val))
        indexed_rows.sort(key=lambda item: item[0])
        metric_rows = [row for _, row, _ in indexed_rows]
        condition_rows = [cond_val for _, _, cond_val in indexed_rows]
    else:
        for timestamp in selected_timestamps:
            snapshot = _snapshot(timestamp)
            if snapshot is None:
                non_converged.append(timestamp)
                continue
            metric_rows.append(_extract_historical_metric(snapshot, target_kind, target_name, request))
            condition_rows.append(_extract_condition_value(snapshot, timestamp, normalized_condition))

    non_converged.sort()
    if not metric_rows:
        raise HTTPException(
            status_code=503,
            detail=f"The power flow did not converge at any of the {len(non_converged)} "
                   "ticks in the window, so no statistics can be computed.")
    exceedance_rows = [row for row in metric_rows if row["is_violation"]]
    near_miss_rows = [row for row in metric_rows if row.get("near_miss")]
    exceedance_count = len(exceedance_rows)
    near_miss_count = len(near_miss_rows)
    total_steps = len(metric_rows)

    worst_timestamp = None
    if metric_rows:
        worst_timestamp = max(metric_rows, key=lambda row: float(row.get("severity", float("-inf"))))["timestamp"]

    episode_rows = []
    active_episode = None
    for row in metric_rows:
        if row["is_violation"]:
            if active_episode is None:
                active_episode = {
                    "start": row["timestamp"],
                    "end": row["timestamp"],
                    "peak_severity": float(row["severity"]),
                    "peak_value": float(row["value"]),
                    "limit": float(row["limit"]),
                }
            else:
                active_episode["end"] = row["timestamp"]
                if float(row["severity"]) > float(active_episode["peak_severity"]):
                    active_episode["peak_severity"] = float(row["severity"])
                    active_episode["peak_value"] = float(row["value"])
                    active_episode["limit"] = float(row["limit"])
        elif active_episode is not None:
            episode_rows.append(active_episode)
            active_episode = None
    if active_episode is not None:
        episode_rows.append(active_episode)

    for episode in episode_rows:
        duration_steps = timestamps.index(episode["end"]) - timestamps.index(episode["start"]) + 1
        episode["duration_steps"] = int(duration_steps)

    episode_rows.sort(key=lambda row: (float(row["peak_severity"]), int(row["duration_steps"])), reverse=True)
    worst_episodes = episode_rows[: request.worst_n]

    sorted_rows = sorted(metric_rows, key=lambda row: float(row.get("severity", float("-inf"))), reverse=True)
    n_full = len(sorted_rows)
    max_curve_points = 400
    if n_full <= max_curve_points:
        sampled_indices = list(range(n_full))
    else:
        # Quantile-like sampling across the full sorted range keeps the curve
        # shape and zero-crossing aligned with full-window frequencies.
        sampled_indices = []
        for i in range(max_curve_points):
            idx = int(round(i * (n_full - 1) / (max_curve_points - 1)))
            if not sampled_indices or idx != sampled_indices[-1]:
                sampled_indices.append(idx)

    duration_curve = [
        {
            "rank_fraction": (idx / (n_full - 1)) if n_full > 1 else 0.0,
            "severity": float(sorted_rows[idx]["severity"]),
            "value": float(sorted_rows[idx]["value"]),
            "timestamp": sorted_rows[idx]["timestamp"],
        }
        for idx in sampled_indices
    ]

    conditional_bins = None
    if normalized_condition is not None:
        conditional_bins = _build_conditional_bins(metric_rows, condition_rows, normalized_condition, request.n_bins)

    notes = [
        "First implementation slice uses a signed limit-margin metric for the duration curve.",
    ]
    if non_converged:
        notes.append(
            f"{len(non_converged)} tick(s) had no converged power flow and are excluded from "
            "the statistics; exceedance_frequency_upper_bound counts them as exceedances.")
    if normalized_condition is None:
        notes.append("Conditional binning disabled for this request (condition not set).")
    elif conditional_bins:
        notes.append(
            f"Conditional bins computed for '{normalized_condition}' using "
            f"{'hour groups' if normalized_condition == 'hour' else f'{len(conditional_bins)} quantile bins'}.")
    else:
        notes.append(
            f"Condition '{normalized_condition}' requested but unavailable for the selected window/system.")

    return {
        "target": request.target,
        "target_kind": target_kind,
        "target_label": metric_rows[0]["target_label"] if metric_rows else request.target,
        "system_name": system_name,
        "window_start": selected_timestamps[0],
        "window_end": selected_timestamps[-1],
        "n_timestamps": total_steps,
        "data_source": _dataset_label(request),
        "thresholds_used": {
            "vm_upper_pu": request.vm_upper_pu,
            "vm_lower_pu": request.vm_lower_pu,
            "max_line_loading_pct": request.max_line_loading_pct,
            "max_trafo_loading_pct": request.max_trafo_loading_pct,
        },
        "exceedance_count": exceedance_count,
        "exceedance_frequency": (exceedance_count / total_steps) if total_steps else 0.0,
        # Over converged ticks only. The bound counts every non-converged tick
        # as an exceedance — a tick with no solution is often the worst one.
        "n_non_converged": len(non_converged),
        "non_converged_timestamps": non_converged[:50],
        "exceedance_frequency_upper_bound": (
            (exceedance_count + len(non_converged)) / (total_steps + len(non_converged))
            if total_steps + len(non_converged) else 0.0),
        "near_miss_count": near_miss_count,
        "near_miss_frequency": (near_miss_count / total_steps) if total_steps else 0.0,
        "worst_timestamp": worst_timestamp,
        "worst_episodes": worst_episodes,
        "duration_curve": duration_curve,
        "duration_curve_limit": 0.0,
        "duration_curve_metric": "signed_limit_margin",
        "used_margin_fallback": exceedance_count == 0,
        "execution_mode": "parallel" if use_parallel else "sequential",
        "max_workers_used": effective_max_workers if use_parallel else 1,
        "condition_used": normalized_condition,
        "conditional_bins": conditional_bins,
        "notes": notes,
    }


@app.post("/api/rsa/worst_case")
@_off_event_loop
async def worst_case_scan(request: WorstCaseRequest):
    """Scan timestamps for the worst-case grid operating point.

    Iterates through timestamps **without** mutating ``app_data["current_index"]``.
    Runs pandapower (no OPF) at each tick, collecting key security metrics per
    tick. Returns the worst timestamp per metric plus the full scan series for
    charting.

    ``data_source`` selects which series to scan:
      - ``"measurements"`` (default): historical actuals, scanned forward from
        the current simulation index.
      - ``"forecasts"``: the look-ahead planning series, scanned from its start.

    Metrics available
    -----------------
    violations   : total count of voltage + thermal limit breaches
    slack_import : external-grid active-power import (MW)
    max_voltage  : highest bus voltage (p.u.)
    min_voltage  : lowest bus voltage (p.u.) — worst = most negative
    max_loading  : highest line loading (%)
    """
    # Refused rather than guessed: an unknown metric used to scan for
    # "violations" while echoing the requested name, and n_steps <= 0 became
    # a 500 after scanning nothing.
    if request.metric not in _WORST_CASE_METRICS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown metric {request.metric!r}; use one of {sorted(_WORST_CASE_METRICS)}.",
        )
    if request.n_steps is not None and request.n_steps <= 0:
        raise HTTPException(status_code=400, detail="n_steps must be a positive number of ticks.")
    timestamps, dataset, source_label = _window_dataset(request.data_source)
    vlo, vhi, mll, mtl = _effective_thresholds(request)
    # Forecasts have no simulation clock — always scan from their start.
    start = app_data["current_index"] if source_label == "measurements" else 0
    n = len(timestamps)

    end = min(start + request.n_steps, n) if request.n_steps is not None else n
    indices = list(range(start, end, max(1, request.step_size)))

    series: dict[str, list] = {
        "timestamps": [], "violation_counts": [], "slack_import_mw": [],
        "min_voltage": [], "max_voltage": [], "max_line_loading": [],
    }

    def _finite_array(values) -> np.ndarray:
        arr = np.asarray(values, dtype=float)
        return arr[np.isfinite(arr)]

    # Ticks whose power flow did not converge used to vanish from the scan
    # without trace — and a tick that does not converge is often the voltage
    # collapse the scan is looking for. They are now reported.
    non_converged: list[str] = []
    for idx in indices:
        ts = timestamps[idx]
        try:
            net_copy = copy.deepcopy(_net_for(request))
            meas = prepare_measurement_df(dataset[ts])
            net_copy, _ = _assign_tick(net_copy, meas, ts, dataset)
            pp.runpp(net_copy, algorithm="nr", numba=False)
        except HTTPException:
            raise  # a request error (unknown network, profile) is not a non-converged tick
        except Exception:
            non_converged.append(ts)
            continue

        vm_series = net_copy.res_bus["vm_pu"]
        bus_in_service = net_copy.bus.get("in_service", pd.Series(True, index=net_copy.bus.index))
        active_mask = bus_in_service.reindex(vm_series.index).fillna(True).astype(bool)
        vm = _finite_array(vm_series.loc[active_mask].values)
        line_load = _finite_array(net_copy.res_line.get("loading_percent", pd.Series(dtype=float)).values)
        trafo_load = _finite_array(net_copy.res_trafo.get("loading_percent", pd.Series(dtype=float)).values)
        slack_mw = _safe_float(net_copy.res_ext_grid.iloc[0].get("p_mw"))

        # Skip timestamps that produce unusable metrics (typically islanding/non-converged residue).
        if vm.size == 0:
            non_converged.append(ts)
            continue

        v_viol = int(((vm < vlo) | (vm > vhi)).sum())
        l_viol = int((line_load > mll).sum()) if line_load.size else 0
        t_viol = int((trafo_load > mtl).sum()) if trafo_load.size else 0

        series["timestamps"].append(ts)
        series["violation_counts"].append(v_viol + l_viol + t_viol)
        series["slack_import_mw"].append(slack_mw if slack_mw is not None else 0.0)
        series["min_voltage"].append(_safe_float(vm.min()) if vm.size else None)
        series["max_voltage"].append(_safe_float(vm.max()) if vm.size else None)
        series["max_line_loading"].append(_safe_float(line_load.max(), ndigits=2) if line_load.size else 0.0)

    if not series["timestamps"]:
        raise HTTPException(status_code=500, detail="No timestamps could be scanned.")

    def _worst(key: str, maximize: bool = True) -> dict:
        vals = [v for v in series[key] if v is not None]
        if not vals:
            return {"timestamp": None, "value": None}
        best_val = max(vals) if maximize else min(vals)
        best_idx = series[key].index(best_val)
        return {"timestamp": series["timestamps"][best_idx], "value": best_val}

    worst_per_metric = {
        "violations": _worst("violation_counts"),
        "slack_import": _worst("slack_import_mw"),
        "max_voltage": _worst("max_voltage"),
        "min_voltage": _worst("min_voltage", maximize=False),
        "max_loading": _worst("max_line_loading"),
    }

    primary = worst_per_metric.get(request.metric, worst_per_metric["violations"])

    return {
        "metric": request.metric,
        "data_source": source_label,
        "worst_timestamp": primary["timestamp"],
        "worst_value": primary["value"],
        "current_timestamp": timestamps[start],
        "scan_window": {"start": series["timestamps"][0], "end": series["timestamps"][-1]},
        "n_scanned": len(series["timestamps"]),
        "n_non_converged": len(non_converged),
        "non_converged_timestamps": non_converged[:50],
        "series": series,
        "worst_per_metric": worst_per_metric,
        "thresholds_used": {
            "vm_upper_pu": vhi,
            "vm_lower_pu": vlo,
            "max_line_loading_pct": mll,
            "max_trafo_loading_pct": mtl,
        },
    }


def _scan_one_scenario(
    sgen_scale: float,
    indices: list,
    timestamps: list,
    net,
    measurements: dict,
    substations_list: list,
    load_scaling_factor: float,
    vm_upper_pu: float,
    vm_lower_pu: float,
    max_line_loading_pct: float,
    max_trafo_loading_pct: float,
) -> dict:
    """Run a full timestamp scan for one sgen_scale. Thread-safe — all shared
    data is read-only; each call does its own deep-copy per tick."""
    out_timestamps: list[str] = []
    series: dict[str, list] = {
        "violations": [], "slack_import_mw": [],
        "max_vm_pu": [], "min_vm_pu": [], "max_line_loading_pct": [],
    }
    # Per-element violation counters: element_name -> number of ticks it violated
    bus_viol_counts: dict[str, int] = {}
    line_viol_counts: dict[str, int] = {}
    trafo_viol_counts: dict[str, int] = {}
    # Violated-ticks detail: only ticks with at least one violation
    violations_per_tick: list[dict] = []

    # Ticks whose power flow did not converge used to vanish from the scan
    # without trace — and a tick that does not converge is often the voltage
    # collapse the scan is looking for. They are now reported.
    non_converged: list[str] = []
    for idx in indices:
        ts = timestamps[idx]
        try:
            net_copy = copy.deepcopy(net)
            meas = prepare_measurement_df(measurements[ts])
            net_copy, _ = _assign_tick(net_copy, meas, ts, measurements, subs=substations_list)
            if load_scaling_factor != 1.0:
                _scale_loads(net_copy, load_scaling_factor)
            if sgen_scale != 1.0:
                net_copy.sgen["p_mw"] *= sgen_scale
                net_copy.sgen["q_mvar"] *= sgen_scale
            pp.runpp(net_copy, algorithm="nr", numba=False)
        except HTTPException:
            raise  # a request error (unknown network, profile) is not a non-converged tick
        except Exception:
            non_converged.append(ts)
            continue

        # ---- count violations and track which elements violated ------------
        viol = 0
        tick_buses: list[str] = []
        tick_lines: list[str] = []
        tick_trafos: list[str] = []
        for bus_idx, row in net_copy.res_bus.iterrows():
            if row["vm_pu"] > vm_upper_pu or row["vm_pu"] < vm_lower_pu:
                viol += 1
                name = _bus_display_name(net_copy, int(bus_idx))
                if name not in tick_buses:  # a tick count, once per name
                    bus_viol_counts[name] = bus_viol_counts.get(name, 0) + 1
                tick_buses.append(name)
        for line_idx, row in net_copy.res_line.iterrows():
            if row["loading_percent"] > max_line_loading_pct:
                viol += 1
                name = _line_display_name(net_copy, int(line_idx))
                line_viol_counts[name] = line_viol_counts.get(name, 0) + 1
                tick_lines.append(name)
        for trafo_idx, row in net_copy.res_trafo.iterrows():
            if row["loading_percent"] > max_trafo_loading_pct:
                viol += 1
                name = _trafo_display_name(net_copy, int(trafo_idx))
                trafo_viol_counts[name] = trafo_viol_counts.get(name, 0) + 1
                tick_trafos.append(name)

        if tick_buses or tick_lines or tick_trafos:
            violations_per_tick.append({
                "timestamp": ts,
                "buses": tick_buses,
                "lines": tick_lines,
                "trafos": tick_trafos,
            })

        slack_mw = float(net_copy.res_ext_grid["p_mw"].iloc[0]) if not net_copy.res_ext_grid.empty else 0.0
        out_timestamps.append(ts)
        series["violations"].append(viol)
        series["slack_import_mw"].append(round(slack_mw, 4))
        series["max_vm_pu"].append(round(float(net_copy.res_bus["vm_pu"].max()), 5))
        series["min_vm_pu"].append(round(float(net_copy.res_bus["vm_pu"].min()), 5))
        series["max_line_loading_pct"].append(
            round(float(net_copy.res_line["loading_percent"].max()), 2)
            if not net_copy.res_line.empty else 0.0
        )

    total_viol = sum(series["violations"])
    worst_ts = None
    if out_timestamps and series["violations"]:
        worst_ts = out_timestamps[series["violations"].index(max(series["violations"]))]

    if abs(sgen_scale - 1.0) < 1e-6:
        label = "Baseline (×1.0)"
    elif sgen_scale < 1.0:
        label = f"Low renewables (×{sgen_scale:.2f})"
    else:
        label = f"High renewables (×{sgen_scale:.2f})"

    # Sort by violation count descending for easy reading
    def _sorted_desc(d: dict) -> dict:
        return dict(sorted(d.items(), key=lambda x: x[1], reverse=True))

    return {
        "sgen_scale": sgen_scale,
        "label": label,
        "n_scanned": len(out_timestamps),
        "n_non_converged": len(non_converged),
        "non_converged_timestamps": non_converged[:50],
        "timestamps": out_timestamps,
        "series": series,
        "summary": {
            "total_violations": total_viol,
            "max_slack_import_mw": round(max(series["slack_import_mw"]), 4) if series["slack_import_mw"] else 0.0,
            "mean_violations_per_tick": round(total_viol / len(out_timestamps), 3) if out_timestamps else 0.0,
            "worst_timestamp": worst_ts,
        },
        "violation_summary": {
            "buses": _sorted_desc(bus_viol_counts),
            "lines": _sorted_desc(line_viol_counts),
            "trafos": _sorted_desc(trafo_viol_counts),
        },
        "violations_per_tick": violations_per_tick,
    }


@app.post("/api/rsa/scan_scenarios")
@_off_event_loop
async def scan_scenarios_endpoint(request: ScanScenariosRequest):
    """Run RSA across a time range for multiple sgen scaling factors in parallel.

    Each scenario (sgen_scale) is scanned in its own thread — the scans are
    fully independent and pandapower's C solver releases the GIL, so
    ``ThreadPoolExecutor`` gives near-linear speedup with scenario count.

    ``app_data["current_index"]`` is **never mutated** by this endpoint.
    """
    (request.vm_lower_pu, request.vm_upper_pu,
     request.max_line_loading_pct, request.max_trafo_loading_pct) = _effective_thresholds(request)
    all_timestamps, scenario_measurements, source_label = _window_dataset(request.data_source)
    n = len(all_timestamps)
    # Forecasts have no simulation clock — default the window start to index 0.
    default_start = app_data["current_index"] if source_label == "measurements" else 0

    # ---- resolve time window (same rule as element_timeseries_scan) --------
    # An unmatched bound is an error, not the whole dataset: a request for May
    # 2022 used to scan every tick of January 2000 and report it as May.
    start = _window_index(all_timestamps, request.start_timestamp, "start_timestamp",
                          source_label, last=False, default=default_start)
    end = _window_index(all_timestamps, request.end_timestamp, "end_timestamp",
                        source_label, last=True, default=n - 1) + 1
    if end <= start:
        raise HTTPException(
            status_code=400,
            detail=f"end_timestamp ({request.end_timestamp}) is before start "
                   f"({all_timestamps[start]}).",
        )

    if request.n_steps is not None:
        end = min(start + request.n_steps, end)

    indices = list(range(start, end, max(1, request.step_size)))
    scales = request.sgen_scales if request.sgen_scales else [0.7, 1.0, 1.3]

    # ---- run scenarios in parallel -----------------------------------------
    scenario_results: list[dict | None] = [None] * len(scales)
    max_workers = min(len(scales), 4)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(
                _scan_one_scenario,
                scale, indices, all_timestamps,
                _net_for(request), scenario_measurements, substations,
                request.load_scaling_factor,
                request.vm_upper_pu, request.vm_lower_pu,
                request.max_line_loading_pct, request.max_trafo_loading_pct,
            ): i
            for i, scale in enumerate(scales)
        }
        for future in as_completed(futures):
            i = futures[future]
            scenario_results[i] = future.result()

    n_scanned = max((s["n_scanned"] for s in scenario_results if s), default=0)
    return {
        "scenarios": scenario_results,
        "n_scanned": n_scanned,
        "thresholds": {
            "vm_upper_pu": request.vm_upper_pu,
            "vm_lower_pu": request.vm_lower_pu,
            "max_line_loading_pct": request.max_line_loading_pct,
            "max_trafo_loading_pct": request.max_trafo_loading_pct,
        },
    }


def _run_probabilistic_sample(
    sgen_mult: float,
    load_mult: float,
    net_base,
    sgen_p_base,
    sgen_p_ceiling,
    sgen_p_max,
    vm_upper_pu: float,
    vm_lower_pu: float,
    max_line_loading_pct: float,
    max_trafo_loading_pct: float,
) -> dict | None:
    """Run one Monte Carlo sample. net_base already has measurements assigned.
    Thread-safe — deep-copies net_base before any mutation."""
    try:

        net_copy = copy.deepcopy(net_base)

        # Generation uncertainty model (Step 2):
        #   xi = clip(P_base + e, 0, P_max), where e is forecast error.
        #   P_inj = min(P_ceiling, xi).
        # We encode e via a relative draw (sgen_mult), so:
        #   xi = clip(P_base * sgen_mult, 0, P_max).
        # For probabilistic RSA, P_ceiling is the current/base setpoint.
        if len(net_copy.sgen.index) > 0:
            xi = np.clip(sgen_p_base * float(sgen_mult), 0.0, sgen_p_max)
            p_inj = np.minimum(sgen_p_ceiling, xi)
            net_copy.sgen["p_mw"] = p_inj

        if abs(load_mult - 1.0) > 1e-6:
            _scale_loads(net_copy, load_mult)
        pp.runpp(net_copy, algorithm="nr", numba=False)
    except Exception:
        return None

    viol_buses: list[str] = []
    viol_lines: list[str] = []
    viol_trafos: list[str] = []
    bus_vm: dict[str, float] = {}
    excluded_buses: list[str] = []

    for bus_idx, row in net_copy.res_bus.iterrows():
        name = _bus_display_name(net_copy, int(bus_idx))
        vm = float(row["vm_pu"])
        # Out-of-service buses come back with vm_pu = NaN. Carried through,
        # they made the whole response unserialisable — json.dumps rejects NaN
        # and the endpoint returned 500 with the analysis already done — and
        # they corrupt the percentile sort, since NaN has no ordering. Skipped
        # and reported, the way the OPF endpoints already handle them.
        if not math.isfinite(vm):
            excluded_buses.append(name)
            continue
        bus_vm[name] = round(vm, 5)
        if vm > vm_upper_pu or vm < vm_lower_pu:
            viol_buses.append(name)

    for line_idx, row in net_copy.res_line.iterrows():
        if row["loading_percent"] > max_line_loading_pct:
            name = _line_display_name(net_copy, int(line_idx))
            viol_lines.append(name)

    for trafo_idx, row in net_copy.res_trafo.iterrows():
        if row["loading_percent"] > max_trafo_loading_pct:
            viol_trafos.append(_trafo_display_name(net_copy, int(trafo_idx)))

    return {
        "viol_buses": viol_buses,
        "viol_lines": viol_lines,
        "viol_trafos": viol_trafos,
        "bus_vm": bus_vm,
        "excluded_buses": excluded_buses,
        "total_violations": len(viol_buses) + len(viol_lines) + len(viol_trafos),
    }


def _run_envelope_point(
    p_mw: float,
    q_mvar: float,
    net_base,
    sgen_idx: int,
    vm_upper: float,
    vm_lower: float,
    max_line: float,
    max_trafo: float,
) -> dict:
    """Run one power flow for the (P, Q) grid sweep. Thread-safe — deep-copies net_base."""
    _net = copy.deepcopy(net_base)
    _net.sgen.at[sgen_idx, "p_mw"] = p_mw
    _net.sgen.at[sgen_idx, "q_mvar"] = q_mvar
    try:
        pp.runpp(_net, algorithm="nr", numba=False)
    except Exception:
        return {
            "p_mw": round(p_mw, 5), "q_mvar": round(q_mvar, 5),
            "feasible": False, "converged": False,
            "max_vm_pu": None, "min_vm_pu": None,
            "max_loading_pct": None, "violations": -1,
            "binding_constraint": "no_convergence",
        }

    max_vm = float(_net.res_bus["vm_pu"].max())
    min_vm = float(_net.res_bus["vm_pu"].min())
    max_line_load = float(_net.res_line["loading_percent"].max()) if len(_net.res_line) > 0 else 0.0
    max_trafo_load = float(_net.res_trafo["loading_percent"].max()) if len(_net.res_trafo) > 0 else 0.0
    max_loading = max(max_line_load, max_trafo_load)

    viol_upper   = max_vm > vm_upper
    viol_lower   = min_vm < vm_lower
    # Lines against the line limit and transformers against the transformer
    # limit. Both against the larger of the two let a line at 90 % pass an
    # 80 % line limit whenever transformers were allowed 100 % (review 2.12).
    line_over  = max_line_load > max_line
    trafo_over = max_trafo_load > max_trafo
    viol_thermal = line_over or trafo_over
    feasible = not (viol_upper or viol_lower or viol_thermal)
    thermal = None
    if viol_thermal:
        r_line = max_line_load / max_line if max_line else 0.0
        r_trafo = max_trafo_load / max_trafo if max_trafo else 0.0
        kind = "line" if (line_over and (not trafo_over or r_line >= r_trafo)) else "trafo"
        thermal = {"branch_type": kind,
                   "value": round(max_line_load if kind == "line" else max_trafo_load, 2),
                   "limit": float(max_line if kind == "line" else max_trafo)}

    if viol_upper:
        binding = "upper_voltage"
    elif viol_thermal:
        binding = "thermal"
    elif viol_lower:
        binding = "lower_voltage"
    else:
        binding = None

    return {
        "p_mw": round(p_mw, 5), "q_mvar": round(q_mvar, 5),
        "feasible": feasible, "converged": True,
        "max_vm_pu": round(max_vm, 5), "min_vm_pu": round(min_vm, 5),
        "max_loading_pct": round(max_loading, 2),
        "violations": int(viol_upper) + int(viol_lower) + int(viol_thermal),
        "binding_constraint": binding,
        "thermal": thermal,
    }


@app.post("/api/rsa/probabilistic")
@_off_event_loop
async def probabilistic_rsa(request: ProbabilisticRSARequest):
    """Monte Carlo probabilistic security assessment at the current timestamp.

    Draws ``n_samples`` Latin Hypercube samples of generation/load uncertainty.
    Generation is modeled as clipped forecast-error availability with setpoint
    ceiling (``P_inj = min(P_ceiling, xi)``), while load stays multiplicative.
    Runs ``pp.runpp`` for each
    sample in parallel, and returns:
    - Per-bus / per-line / per-trafo violation probability
    - Voltage P5/P50/P95 envelopes per bus
    - Distribution of total violation count across samples
    - Overall probability that any violation occurs

    ``app_data["current_index"]`` is **never mutated** by this endpoint.
    """
    from scipy.stats.qmc import LatinHypercube
    from scipy.stats import norm as sp_norm

    net = _net_for(request)
    (request.vm_lower_pu, request.vm_upper_pu,
     request.max_line_loading_pct, request.max_trafo_loading_pct) = _effective_thresholds(request)
    ts = _resolve_tick(request.data_source, request.timestamp)

    # Assign measurements once; net_base is read-only after this
    meas = prepare_measurement_df(_lookup_measurement_df(ts, request.data_source))
    net_base = copy.deepcopy(net)
    net_base, _ = _assign_tick(net_base, meas, ts, _dataset_of(request))

    out_of_service_names = {
        _bus_display_name(net_base, int(idx))
        for idx in net_base.bus.index
        if "in_service" in net_base.bus.columns
        and not bool(net_base.bus.at[idx, "in_service"])
    }

    # Generation uncertainty inputs (Step 2), with all sgens treated as
    # uncertain generators for now (Step 1 intentionally skipped).
    if len(net_base.sgen.index) > 0:

        sgen_p_base = net_base.sgen["p_mw"].to_numpy(dtype=float, copy=True)
        sgen_p_ceiling = sgen_p_base.copy()  # pre-OPF probabilistic RSA

        if "max_p_mw" in net_base.sgen.columns:
            raw_pmax = net_base.sgen["max_p_mw"].to_numpy(dtype=float, copy=True)
        else:
            raw_pmax = sgen_p_base.copy()
        sgen_p_max = np.where(np.isfinite(raw_pmax), raw_pmax, sgen_p_base)
        sgen_p_max = np.maximum(sgen_p_max, 0.0)
    else:
        sgen_p_base = []
        sgen_p_ceiling = []
        sgen_p_max = []

    # LHC sampling: draw n points in [0,1]^2, map to N(1, sigma) and clip.
    # When sigma is zero, use deterministic multipliers = 1.0.
    n = max(10, min(request.n_samples, 1000))
    sampler = LatinHypercube(d=2, seed=42)
    unit_samples = sampler.random(n=n)           # shape (n, 2)
    if request.sgen_sigma <= 0:
        sgen_mults = np.ones(n)
    else:
        sgen_mults = sp_norm.ppf(unit_samples[:, 0], loc=1.0, scale=request.sgen_sigma).clip(0.01, 3.0)
    if request.load_sigma <= 0:
        load_mults = np.ones(n)
    else:
        load_mults = sp_norm.ppf(unit_samples[:, 1], loc=1.0, scale=request.load_sigma).clip(0.01, 3.0)

    # Run all samples in parallel
    raw_results: list[dict | None] = [None] * n
    with ThreadPoolExecutor(max_workers=min(n, 8)) as pool:
        futures = {
            pool.submit(
                _run_probabilistic_sample,
                float(sgen_mults[i]), float(load_mults[i]),
                net_base,
                sgen_p_base, sgen_p_ceiling, sgen_p_max,
                request.vm_upper_pu, request.vm_lower_pu,
                request.max_line_loading_pct, request.max_trafo_loading_pct,
            ): i
            for i in range(n)
        }
        for future in as_completed(futures):
            raw_results[futures[future]] = future.result()

    converged = [r for r in raw_results if r is not None]
    n_converged = len(converged)
    if n_converged == 0:
        raise HTTPException(status_code=500, detail="All Monte Carlo samples failed to converge.")

    # --- Aggregate ---
    bus_viol_counts: dict[str, int] = {}
    line_viol_counts: dict[str, int] = {}
    trafo_viol_counts: dict[str, int] = {}
    bus_vm_samples: dict[str, list[float]] = {}
    total_viol_dist: list[int] = []
    any_violation_count = 0

    for r in converged:
        # Each name once per sample. Several buses can share a display name
        # (one real network has four called "Værket A_aux"), and counting each
        # of them made a "probability" of 5.0. Shared names are reported
        # together: the probability that any element of that name violates.
        for name in set(r["viol_buses"]):
            bus_viol_counts[name] = bus_viol_counts.get(name, 0) + 1
        for name in set(r["viol_lines"]):
            line_viol_counts[name] = line_viol_counts.get(name, 0) + 1
        for name in set(r["viol_trafos"]):
            trafo_viol_counts[name] = trafo_viol_counts.get(name, 0) + 1
        for bname, vm in r["bus_vm"].items():
            bus_vm_samples.setdefault(bname, []).append(vm)
        total_viol_dist.append(r["total_violations"])
        if r["total_violations"] > 0:
            any_violation_count += 1

    def _to_prob(d: dict) -> dict:
        return dict(sorted(
            {k: round(v / n_converged, 4) for k, v in d.items()}.items(),
            key=lambda x: x[1], reverse=True,
        ))

    # Voltage P5 / P50 / P95 per bus
    voltage_percentiles: dict[str, dict] = {}
    for bname, vm_list in bus_vm_samples.items():
        vm_s = sorted(vm_list)
        n_vm = len(vm_s)
        voltage_percentiles[bname] = {
            "p5":  round(vm_s[max(0, int(0.05 * n_vm) - 1)], 5),
            "p50": round(vm_s[max(0, int(0.50 * n_vm) - 1)], 5),
            "p95": round(vm_s[min(n_vm - 1, int(0.95 * n_vm))], 5),
        }

    max_viol = max(total_viol_dist) if total_viol_dist else 0
    viol_histogram = {k: total_viol_dist.count(k) for k in range(max_viol + 1)}

    # Buses with no finite voltage in any sample. Reported rather than
    # dropped silently, so a caller can tell "this bus is secure" apart from
    # "this bus was never assessed" — the same distinction the OPF endpoints
    # make with excluded_buses_post_opf.
    excluded_bus_names = sorted({
        name for r in converged for name in r.get("excluded_buses") or ()
    })
    excluded_buses = [
        {"bus_name": name,
         "reason": "out_of_service" if name in out_of_service_names else "non_finite_voltage"}
        for name in excluded_bus_names
    ]

    return {
        "timestamp": ts,
        "data_source": _dataset_label(request),
        "n_samples": n,
        "n_converged": n_converged,
        "excluded_buses": excluded_buses,
        "excluded_bus_count": len(excluded_buses),
        "p_any_violation": round(any_violation_count / n_converged, 4),
        "expected_violations": round(sum(total_viol_dist) / n_converged, 3),
        "bus_violation_probability": _to_prob(bus_viol_counts),
        "line_violation_probability": _to_prob(line_viol_counts),
        "trafo_violation_probability": _to_prob(trafo_viol_counts),
        "voltage_percentiles": voltage_percentiles,
        "violation_count_histogram": viol_histogram,
        "samples_summary": {
            "sgen_mean": 1.0,
            "sgen_sigma": request.sgen_sigma,
            "load_mean": 1.0,
            "load_sigma": request.load_sigma,
        },
        "thresholds": {
            "vm_upper_pu": request.vm_upper_pu,
            "vm_lower_pu": request.vm_lower_pu,
            "max_line_loading_pct": request.max_line_loading_pct,
            "max_trafo_loading_pct": request.max_trafo_loading_pct,
        },
    }


def _window_index(timestamps: list, prefix: str | None, label: str, source: str,
                  *, last: bool, default: int) -> int:
    """Index of the first (or last) timestamp starting with `prefix`.

    `default` when no bound was given; HTTP 404 when one was given and nothing
    matches — never a silent substitute.
    """
    if prefix is None or not str(prefix).strip():
        return default
    prefix = str(prefix).strip().replace("T", " ")
    matched = [i for i, t in enumerate(timestamps) if t.startswith(prefix)]
    if not matched:
        span = f"{timestamps[0]} to {timestamps[-1]}" if timestamps else "no data"
        raise HTTPException(
            status_code=404,
            detail=f"{label} {prefix!r} matches no timestamp in the {source} ({span}).",
        )
    return matched[-1] if last else matched[0]


_DISPLAY_NAMES = {
    "bus": _bus_display_name,
    "line": _line_display_name,
    "trafo": _trafo_display_name,
}


def _resolve_element(net, etype: str, tbl, name: str) -> tuple[int, str]:
    """(index, display name) for a bus, line or transformer named by the agent.

    In order: the display name other tools report ("Bus_3",
    "Bus_1 -> Bus_2 [L2]"), the raw `name` column, an integer index, and only
    then a substring — and a substring only when it picks out exactly one
    element. The previous lookup matched substrings of the raw column first, so
    the names in every other result (raw names are empty on MATPOWER cases)
    were never found, and "Oscar" silently resolved to "Oscar A".
    """
    wanted = str(name).strip()
    key = wanted.lower()
    plural = {"bus": "buses", "line": "lines", "trafo": "transformers"}[etype]
    display = {int(i): str(_DISPLAY_NAMES[etype](net, int(i))) for i in tbl.index}
    raw = ({int(i): str(v) for i, v in tbl["name"].items() if isinstance(v, str)}
           if "name" in tbl.columns else {})

    for names in (display, raw):
        exact = [i for i, n in names.items() if n.strip().lower() == key]
        if len(exact) == 1:
            return exact[0], display[exact[0]]
        if len(exact) > 1:
            raise HTTPException(
                status_code=400,
                detail=f"{len(exact)} {plural} are named {wanted!r} (indices "
                       f"{exact[:10]}); pass the index instead.",
            )
    try:
        idx = int(wanted)
        if idx in display:
            return idx, display[idx]
    except ValueError:
        pass
    partial = sorted({i for names in (display, raw) for i, n in names.items()
                      if key and key in n.lower()})
    if len(partial) == 1:
        return partial[0], display[partial[0]]
    if len(partial) > 1:
        raise HTTPException(
            status_code=400,
            detail=f"{wanted!r} matches {len(partial)} {plural}: "
                   f"{[display[i] for i in partial[:12]]}. Use the full name.",
        )
    raise HTTPException(
        status_code=404,
        detail=f"No {etype} named {wanted!r}. First names: {list(display.values())[:20]}",
    )


@app.post("/api/grid/element_timeseries")
@_off_event_loop
async def element_timeseries_scan(request: ElementTimeseriesRequest):
    """Scan a time range and return the evolution of one specific element.

    Iterates through timestamps in [start, end) **without** mutating
    ``app_data["current_index"]``.  Runs ``pp.runpp`` at each tick and
    extracts the requested element's key metrics.

    Supported element types and returned series keys
    -------------------------------------------------
    ``bus``   → vm_pu, p_mw, q_mvar
    ``line``  → loading_percent, p_from_mw, i_ka
    ``trafo`` → loading_percent, p_hv_mw, q_hv_mvar
    """
    (request.vm_lower_pu, request.vm_upper_pu,
     request.max_line_loading_pct, request.max_trafo_loading_pct) = _effective_thresholds(request)
    all_timestamps, ets_measurements, ets_source = _window_dataset(request.data_source)
    n = len(all_timestamps)
    default_start = app_data["current_index"] if ets_source == "measurements" else 0

    # ---- resolve the window --------------------------------------------------
    # A window bound that matches nothing is an error. It used to fall back to
    # the simulation clock (start) or the end of the data (end), so asking for
    # 2022-01-02 returned a series from 2000-01-01 described as the day asked.
    start = _window_index(all_timestamps, request.start_timestamp, "start_timestamp",
                          ets_source, last=False, default=default_start)
    end = _window_index(all_timestamps, request.end_timestamp, "end_timestamp",
                        ets_source, last=True, default=n - 1) + 1
    if end <= start:
        raise HTTPException(
            status_code=400,
            detail=f"end_timestamp ({request.end_timestamp}) is before start "
                   f"({all_timestamps[start]}).",
        )

    if request.n_steps is not None:
        end = min(start + request.n_steps, end)

    indices = list(range(start, end, max(1, request.step_size)))

    # ---- locate the element ------------------------------------------------
    etype = request.element_type.lower().strip()
    net_ref = _net_for(request)

    if etype == "bus":
        tbl = net_ref.bus
    elif etype == "line":
        tbl = net_ref.line
    elif etype == "trafo":
        tbl = net_ref.trafo
    else:
        raise HTTPException(
            status_code=400,
            detail=f"element_type must be 'bus', 'line', or 'trafo'; got '{request.element_type}'.",
        )

    elem_idx, elem_name = _resolve_element(net_ref, etype, tbl, request.element_name)

    # ---- prepare series containers -----------------------------------------
    out_timestamps: list[str] = []
    if etype == "bus":
        series: dict[str, list] = {"vm_pu": [], "p_mw": [], "q_mvar": []}
        primary_metric = "vm_pu"
    elif etype == "line":
        series = {"loading_percent": [], "p_from_mw": [], "i_ka": []}
        primary_metric = "loading_percent"
    else:
        series = {"loading_percent": [], "p_hv_mw": [], "q_hv_mvar": []}
        primary_metric = "loading_percent"

    # ---- scan --------------------------------------------------------------
    # Non-converged ticks are reported, not silently dropped (see worst case).
    non_converged: list[str] = []
    for idx in indices:
        ts = all_timestamps[idx]
        try:
            net_copy = copy.deepcopy(_net_for(request))
            meas = prepare_measurement_df(ets_measurements[ts])
            net_copy, _ = _assign_tick(net_copy, meas, ts, ets_measurements)
            if request.load_scaling_factor != 1.0:
                _scale_loads(net_copy, request.load_scaling_factor)
            pp.runpp(net_copy, algorithm="nr", numba=False)
        except HTTPException:
            raise  # a request error (unknown network, profile) is not a non-converged tick
        except Exception:
            non_converged.append(ts)
            continue

        out_timestamps.append(ts)
        if etype == "bus":
            series["vm_pu"].append(round(float(net_copy.res_bus.at[elem_idx, "vm_pu"]), 5))
            series["p_mw"].append(round(float(net_copy.res_bus.at[elem_idx, "p_mw"]), 4))
            series["q_mvar"].append(round(float(net_copy.res_bus.at[elem_idx, "q_mvar"]), 4))
        elif etype == "line":
            series["loading_percent"].append(
                round(float(net_copy.res_line.at[elem_idx, "loading_percent"]), 2))
            series["p_from_mw"].append(
                round(float(net_copy.res_line.at[elem_idx, "p_from_mw"]), 4))
            series["i_ka"].append(
                round(float(net_copy.res_line.at[elem_idx, "i_ka"]), 5))
        else:  # trafo
            series["loading_percent"].append(
                round(float(net_copy.res_trafo.at[elem_idx, "loading_percent"]), 2))
            series["p_hv_mw"].append(
                round(float(net_copy.res_trafo.at[elem_idx, "p_hv_mw"]), 4))
            series["q_hv_mvar"].append(
                round(float(net_copy.res_trafo.at[elem_idx, "q_hv_mvar"]), 4))

    if not out_timestamps:
        raise HTTPException(
            status_code=500,
            detail="No timestamps could be processed for the requested element.",
        )

    return {
        "element_type": etype,
        "element_name": elem_name,
        "element_index": elem_idx,
        "primary_metric": primary_metric,
        "timestamps": out_timestamps,
        "series": series,
        "n_scanned": len(out_timestamps),
        "n_non_converged": len(non_converged),
        "non_converged_timestamps": non_converged[:50],
        "start_timestamp": out_timestamps[0],
        "end_timestamp": out_timestamps[-1],
        "thresholds": {
            "vm_upper_pu": request.vm_upper_pu,
            "vm_lower_pu": request.vm_lower_pu,
            "max_line_loading_pct": request.max_line_loading_pct,
            "max_trafo_loading_pct": request.max_trafo_loading_pct,
        },
    }


def _supply_lost(net, element_type: str, element_index: int) -> dict:
    """Which buses the outage leaves without a path to a source.

    Answers the question an operator actually asks of a contingency — "what
    goes dark" — which the violation table cannot. Zero violations does not
    mean nothing was islanded: an unsupplied bus has no voltage to violate.

    Without this the agent had no way to answer it and tried to walk the
    network one lookup at a time, exhausting its turn budget before saying
    anything. Computing it here costs one graph traversal.
    """
    try:
        import pandapower.topology as top

        unsupplied = sorted(int(b) for b in top.unsupplied_buses(net))
    except Exception:  # noqa: BLE001
        return {"supply_analysis": "unavailable for this network"}

    named = [_bus_display_name(net, b) for b in unsupplied]
    # Loads are what an operator means by "loses supply"; a de-energised bus
    # with nothing on it is not an outage anyone notices.
    affected_load_mw = 0.0
    if not net.load.empty and "bus" in net.load.columns:
        affected_load_mw = float(
            net.load.loc[net.load["bus"].astype(int).isin(unsupplied), "p_mw"].sum()
        )

    return {
        "unsupplied_buses": named,
        "unsupplied_bus_count": len(named),
        "unsupplied_load_mw": round(affected_load_mw, 4),
        "supply_analysis": (
            "Every bus keeps a path to a source; nothing is islanded."
            if not named else
            f"{len(named)} bus(es) lose their path to a source with "
            f"{element_type} {element_index} out."
        ),
    }


@app.post("/api/contingency/simulate")
@_off_event_loop
async def simulate_contingency(request: ContingencyRequest):
    """Single-element N-1 outage simulation *without* corrective action.

    Takes one specific line or transformer out of service, runs the
    power flow, and reports whatever violations appear. Used by the
    *"Simulate Single Outage"* button in the UI.

    For the full sweep across every line + transformer, use
    ``/api/contingency/simulate_all`` instead — it is ~40× cheaper
    than calling this endpoint in a loop.

    No optimization is performed — see ``/api/contingency/optimize`` for
    the cure-the-blackout variant.
    """
    _check_outage_element(request.element_type, request.element_index, _net_for(request))

    applied = None
    if request.option:
        # The outage with a corrective option applied: at its moment, or held to a later one.
        stored, option, request, held = _option_context(request, later=True)
        applied = (stored, option, held)
    current_ts = _resolve_tick(request.data_source, request.timestamp)
    net_copy = copy.deepcopy(_net_for(request))
    measurement_prod_cons = prepare_measurement_df(_lookup_measurement_df(current_ts, request.data_source))

    net_copy, _ = _assign_tick(net_copy, measurement_prod_cons, current_ts, _dataset_of(request))
    _scale_loads(net_copy, request.load_scaling_factor)
    if applied:
        _apply_stored_option(net_copy, applied[0], applied[1], held=applied[2])

    def echoed(result: dict) -> dict:
        return {**result, "option": _option_echo(applied[0], applied[1], current_ts, applied[2])} if applied else result

    if request.element_type == "line":
        net_copy.line.at[request.element_index, "in_service"] = False
    elif request.element_type == "trafo":
        net_copy.trafo.at[request.element_index, "in_service"] = False

    try:
        df_results_ca, _ = ca.contingency_assessment_single(
            net_copy, request.element_type, request.element_index, current_ts,
            vm_lower=request.vm_lower_pu,
            vm_upper=request.vm_upper_pu,
            max_line_loading_pct=request.max_line_loading_pct,
            max_trafo_loading_pct=request.max_trafo_loading_pct,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed: {str(e)}")

    if isinstance(df_results_ca, list): df_results_ca = pd.DataFrame(df_results_ca)

    # The engine returns an empty violation table when the post-outage power
    # flow does not converge, which this endpoint used to report as secure —
    # an outage that collapses the system, presented as harmless.
    if not getattr(net_copy, "converged", True):
        return echoed({"timestamp": current_ts, "system_secure": False, "converged": False,
                "total_violations": None, "violations": [],
                "message": "The power flow did not converge after this outage — the "
                           "system has no stable operating point without this element. "
                           "Treat it as insecure.",
                "data_source": _dataset_label(request),
                "thresholds_used": _thresholds_used(request),
                **_supply_lost(net_copy, request.element_type, request.element_index)})

    if df_results_ca.empty:
        return echoed({"timestamp": current_ts, "system_secure": True, "total_violations": 0,
                "violations": [],
                "data_source": _dataset_label(request),
                "thresholds_used": _thresholds_used(request),
                **_supply_lost(net_copy, request.element_type, request.element_index)})

    element_names = []
    for _, row in df_results_ca.iterrows():
        v_idx = int(row["violation_element_index"])
        if row["violation_type"] == "bus_vm_pu":
            name = _bus_display_name(net_copy, v_idx)
        elif row["violation_type"] == "line_loading":
            name = _line_display_name(net_copy, v_idx)
        elif row["violation_type"] == "trafo_loading":
            name = _trafo_display_name(net_copy, v_idx)
        else:
            name = f"Element_{v_idx}"
        element_names.append(name)

    df_results_ca["element_name"] = element_names
    df_results_ca["timestamp"] = df_results_ca["timestamp"].astype(str)

    # Placeholder: publish_ca_results(df_results_ca, current_ts, topic="topic6913"...)

    return echoed({"timestamp": current_ts, "system_secure": False, "total_violations": len(df_results_ca),
            "violations": df_results_ca.to_dict(orient="records"),
            "data_source": _dataset_label(request),
            "thresholds_used": _thresholds_used(request),
            **_supply_lost(net_copy, request.element_type, request.element_index)})


_SCREEN_VM_TOL, _SCREEN_LOADING_TOL = 1e-4, 0.01      # p.u., % — solver precision, not a violation


def _n1_sweep(net_copy, current_ts: str, vm_lower: float, vm_upper: float,
              max_line_loading_pct: float, max_trafo_loading_pct: float) -> dict:
    """Every single outage of an in-service line or transformer, one power
    flow each, on `net_copy` as prepared (an operating point, maybe a dispatch
    applied, maybe a branch already out). Returns the counts, the violation
    rows (each with its `outage_cause`), and the outages that did not converge."""
    all_violations: list[dict] = []
    outages_with_violations = 0
    total_tested = 0
    # Outages after which the power flow did not converge, or the assessment
    # failed. They used to count as zero violations — i.e. as secure.
    non_converged_outages: list[str] = []

    for el_type, table in (("line", net_copy.line), ("trafo", net_copy.trafo)):
        # A branch already out of service is not a contingency. It used to be
        # "tested" and then put back *in* service for the rest of the sweep.
        for idx in table.index[table["in_service"].astype(bool)]:
            total_tested += 1
            table.at[idx, "in_service"] = False
            try:
                # Beyond solver precision only: a unit holding exactly the band's
                # edge read as "1.05 p.u. above 1.05" after every outage.
                df_results_ca, _ = ca.contingency_assessment_single(
                    net_copy, el_type, idx, current_ts,
                    vm_lower=vm_lower - _SCREEN_VM_TOL,
                    vm_upper=vm_upper + _SCREEN_VM_TOL,
                    max_line_loading_pct=max_line_loading_pct + _SCREEN_LOADING_TOL,
                    max_trafo_loading_pct=max_trafo_loading_pct + _SCREEN_LOADING_TOL,
                )
            except Exception:
                # One bad outage must not abort the sweep — but it is reported.
                table.at[idx, "in_service"] = True
                non_converged_outages.append(
                    _line_display_name(net_copy, int(idx)) if el_type == "line"
                    else _trafo_display_name(net_copy, int(idx)))
                continue
            converged = getattr(net_copy, "converged", True)
            table.at[idx, "in_service"] = True
            if not converged:
                non_converged_outages.append(
                    _line_display_name(net_copy, int(idx)) if el_type == "line"
                    else _trafo_display_name(net_copy, int(idx)))
                continue

            if isinstance(df_results_ca, list):
                df_results_ca = pd.DataFrame(df_results_ca)
            if df_results_ca.empty:
                continue

            outages_with_violations += 1
            # Map each violation's raw index to a readable bus/line/trafo name.
            element_names = []
            for _, row in df_results_ca.iterrows():
                v_idx = int(row["violation_element_index"])
                if row["violation_type"] == "bus_vm_pu":
                    name = _bus_display_name(net_copy, v_idx)
                elif row["violation_type"] == "line_loading":
                    name = _line_display_name(net_copy, v_idx)
                elif row["violation_type"] == "trafo_loading":
                    name = _trafo_display_name(net_copy, v_idx)
                else:
                    name = f"Element_{v_idx}"
                element_names.append(name)

            df_results_ca["element_name"] = element_names
            df_results_ca["timestamp"] = df_results_ca["timestamp"].astype(str)
            _oc = (_line_display_name(net_copy, int(idx)) if el_type == "line"
                   else _trafo_display_name(net_copy, int(idx)))
            df_results_ca["outage_cause"] = _oc
            if "value" in df_results_ca.columns:
                df_results_ca["value"] = df_results_ca["value"].round(3)
            all_violations.extend(df_results_ca.to_dict(orient="records"))

    return {"total_outages_tested": total_tested, "total_outages_causing_violations": outages_with_violations,
            "total_violations": len(all_violations), "violations": all_violations,
            "non_converged_outages": non_converged_outages}


@app.post("/api/contingency/simulate_all")
@_off_event_loop
async def simulate_all_contingencies(request: ContingencySweepRequest):
    """Full N-1 sweep in a single backend call.

    Equivalent to calling ``/api/contingency/simulate`` once per line and
    once per transformer, but condensed into one HTTP round-trip and
    one ``deepcopy`` of the base net. This replaces the frontend's
    40-iteration Python loop, which previously caused a ~6-second stall
    on the UI every time the user clicked *Simulate All Contingencies*.

    Pipeline
    --------
    1. Deep-copy the base net and apply measurements + load scaling **once**.
    2. For each ``idx`` in ``net.line.index``:
         * flip ``in_service = False``,
         * call ``ca.contingency_assessment_single(net, "line", idx, ts)``,
         * flip ``in_service = True`` (restore for next iteration),
         * tag every returned violation row with ``outage_cause`` so the
           UI table can show which outage caused which cascade violation.
    3. Same for ``net.trafo.index``.
    4. Collect everything into one flat list of violation rows.

    Response shape
    --------------
    ``{
        timestamp: str,
        total_outages_tested: int,
        total_outages_causing_violations: int,
        total_violations: int,
        system_n1_secure: bool,
        violations: [ {..., outage_cause: "LINE 3", element_name, value, ...} ]
     }``.

    Notes
    -----
    * Uses one shared ``net_copy`` and toggles ``in_service`` rather than
      re-copying per iteration. The CA engine is read-only w.r.t.
      anything else on the net, so this is safe.
    * A single CA iteration may itself crash (e.g. ``LoadflowNotConverged``
      on a truly islanded topology). We swallow per-iteration exceptions
      and keep going so one bad outage doesn't abort the whole sweep.
      The aggregated response will still be accurate for the other
      iterations; the failed indices simply contribute zero violations.
    """
    chosen = None
    # Network comparisons call this with a plain GridRequest: no option there.
    if getattr(request, "option", None):
        # At the option's own network and load level — at its moment, or a
        # later one with its set-points held.
        stored, option, request, held = _option_context(request, later=True)
        chosen = (stored, option, held)

    current_ts = _resolve_tick(request.data_source, request.timestamp)
    net_copy = copy.deepcopy(_net_for(request))
    measurement_prod_cons = prepare_measurement_df(_lookup_measurement_df(current_ts, request.data_source))

    net_copy, _ = _assign_tick(net_copy, measurement_prod_cons, current_ts, _dataset_of(request))
    _scale_loads(net_copy, request.load_scaling_factor)
    screened_with = None
    if chosen:
        stored, option, held = chosen
        # An option computed after an outage keeps it out, and the sweep
        # screens every further one (N-1-1).
        _apply_stored_option(net_copy, stored, option, held=held)
        screened_with = _option_echo(stored, option, current_ts, held,
                                     screen="N-1-1" if stored.get("outage") else "N-1")
    # Accepted by the request and the agent's wrapper, but never applied: the
    # sweep ran at measured renewable output whatever was asked.
    if request.sgen_scaling_factor != 1.0:
        net_copy.sgen['p_mw'] *= request.sgen_scaling_factor
        net_copy.sgen['q_mvar'] *= request.sgen_scaling_factor

    sweep = _n1_sweep(net_copy, current_ts, request.vm_lower_pu, request.vm_upper_pu,
                      request.max_line_loading_pct, request.max_trafo_loading_pct)
    all_violations, non_converged_outages = sweep["violations"], sweep["non_converged_outages"]
    total_tested, outages_with_violations = sweep["total_outages_tested"], sweep["total_outages_causing_violations"]

    return {
        "timestamp": current_ts,
        "total_outages_tested": total_tested,
        "total_outages_causing_violations": outages_with_violations,
        "total_violations": len(all_violations),
        "system_n1_secure": len(all_violations) == 0 and not non_converged_outages,
        "n_non_converged_outages": len(non_converged_outages),
        "non_converged_outages": non_converged_outages,
        "violations": all_violations,
        "data_source": _dataset_label(request),
        "thresholds_used": _thresholds_used(request),
        # The corrective option this screen ran with, applied — not the schedule.
        **({"option": screened_with} if screened_with else {}),
    }


@app.post("/api/contingency/optimize")
@_off_event_loop
async def optimize_contingency(request: ContingencyOptimizeRequest):
    """Sever an element, then run the Pyomo optimizer to cure the blackout.

    Pipeline
    1. Deep-copy the base network and apply the current measurements.
    2. Apply ``load_scaling_factor`` and reset transformer taps to
       neutral — neutral taps are a prerequisite for the optimizer
       because the pre-computed admittance matrices assumed so.
    3. **Topological surgery**: call the appropriate
       ``modify_network`` helper to delete buses / sgens / loads that
       become orphaned once the element is out. Without this step
       the matrices would divide-by-zero on islanded nodes.-->crash
    4. One ``pp.runpp`` to seed the solver with a feasible starting
       point
    5. Pull the correct Ybus block from ``db_n1_line`` or
       ``db_n1_trafo`` (keyed by element index) instead of recomputing.
    6. Solve with Ipopt via ``flex_engine.optimization_model_base``.
    7. If the termination condition is ``optimal``, return the
       extracted dispatch records. Otherwise return the status string
       and an empty activation list.

    Raises
    HTTPException 503
        If the N-1 admittance databases failed to load at startup.
    HTTPException 404
        If no pre-computed matrix exists for the requested element index.
    HTTPException 500
        If the Ipopt solver itself raises during the solve.
    """
    _check_outage_element(request.element_type, request.element_index, _net_for(request))

    current_ts = _resolve_tick(request.data_source, request.timestamp)
    net_copy = copy.deepcopy(_net_for(request))
    measurement_prod_cons = prepare_measurement_df(_lookup_measurement_df(current_ts, request.data_source))

    net_copy, _ = _assign_tick(net_copy, measurement_prod_cons, current_ts, _dataset_of(request))
    _scale_loads(net_copy, request.load_scaling_factor)
    # Transformers stay at the taps of the network as loaded — the ones the
    # assessments use and the admittance database was built at; the starting
    # point used to come from a neutral-tap power flow (review 2.2).
    net_base = copy.deepcopy(net_copy)

    # The outage, as the N-1 power flows apply it: the element switched out of
    # service, and any bus left without a path to a source switched out with
    # it (its loads unsupplied, its units unavailable). It used to be cut out
    # of the tables and the remaining buses renumbered: a still-meshed bus and
    # its loads could be deleted, lines re-attached to the wrong bus, and the
    # OPF crashed for line 10 and every transformer (code review 5.2).
    modified_net = copy.deepcopy(net_copy)
    outaged = modified_net.line if request.element_type == "line" else modified_net.trafo
    outaged.at[request.element_index, "in_service"] = False
    islanded = sorted(int(b) for b in pp.topology.unsupplied_buses(modified_net))
    if islanded:
        modified_net.bus.loc[islanded, "in_service"] = False
    lost_load = round(float(net_copy.load.loc[net_copy.load.bus.isin(islanded), "p_mw"].sum()), 4)

    try:
        pp.runpp(modified_net)
    except pp.powerflow.LoadflowNotConverged:
        pass

    # Admittances: the network's full database, as before (the N-1 databases
    # built at startup are identical to it). The OPF only reads the branches
    # still in service, so the outaged one drops out. Building them from the
    # outaged network instead changes results through the transformer taps the
    # database was built at — review 2.2, a separate decision.
    adm = _db_for("full", request)
    if adm is None:
        raise HTTPException(status_code=503, detail="Admittance database not loaded.")
    Yff_r, Yff_i, Yft_r, Yft_i = adm["Yff_r"], adm["Yff_i"], adm["Yft_r"], adm["Yft_i"]
    TAPS, trafo_ranges = adm["TAPS"], adm["trafo_ranges"]
    trafo_defaults, branch_to_trafo = adm["trafo_defaults"], adm["branch_to_trafo"]
    islanding = {"islanded_buses": [_bus_display_name(net_copy, b) for b in islanded],
                 "load_not_supplied_mw": lost_load}
    fixed_setpoints, held = _held_at_schedule(net_base, request)
    caps, caps_shown = _branch_caps(net_base, request.branch_limits)

    try:
        results_opt, model_opt = fe.optimization_model_base(
            modified_net, net_base, Yff_r, Yff_i, Yft_r, Yft_i, TAPS, trafo_ranges, trafo_defaults,
            branch_to_trafo,
            opf_vm_lower=request.opf_vm_lower, opf_vm_upper=request.opf_vm_upper,
            lambda_p=request.opf_lambda_p, lambda_q=request.opf_lambda_q,
            pg_max_overrides=request.pg_max_overrides or None,
            pg_min_overrides=request.pg_min_overrides or None,
            min_power_factor=request.opf_min_power_factor,
            current_safety_margin=request.opf_current_safety_margin,
            fixed_setpoints=fixed_setpoints or None,
            branch_limits=caps or None,
            load_flex=_load_flex(modified_net),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Solver crashed: {str(e)}")

    if results_opt is None: raise HTTPException(status_code=500)

    tc = getattr(results_opt.solver, "termination_condition", None)
    outage = {"element_type": request.element_type, "element_index": request.element_index,
              "element": (_line_display_name if request.element_type == "line" else _trafo_display_name)(
                  net_copy, request.element_index)}
    if tc == TerminationCondition.optimal and request.options:
        # Options for the corrective action after the outage: priced against
        # the post-outage schedule, each checked by a power flow of the network
        # without the outaged element.
        result = _flexibility_options(request, current_ts, model_opt, modified_net, net_base,
                                      float(app_data.get("slack_max_mw", 999.0)), held, outage=outage,
                                      caps_shown=caps_shown)
        return {**result, "outage": outage, **islanding}
    if tc == TerminationCondition.optimal:
        # Base values from the post-outage power flow (before redispatch), the
        # state the OPF starts from. `net_base` carries no results of its own
        # for this hour: its tables were the stored network's startup power
        # flow, so the external grid read 245 MW "before" at an hour importing
        # 186 MW, and a 0.13 MW change was reported as −59 MW.
        regulation_data = extract_flexibility_results(model_opt, modified_net, current_ts)
        # Issue 1: extract post-OPF bus voltages
        voltage_data = fe.extract_post_opf_voltages(model_opt, modified_net)
        # Phase 3 Placeholder: publish_flex_ca_results(df, current_ts, topic="topic6919"...)
        # `status` is a solver string; `feasible` and `converged` are the same
        # verdict as booleans. A reader checking whether a dispatch may be
        # described as deliverable should not have to know which of Pyomo's
        # termination strings count as success.
        result = {"timestamp": current_ts, "status": "optimal", "message": "Optimization successful.",
                  "feasible": True, "converged": True,
                  "activated_resources": regulation_data, "outage": outage, **islanding,
                  **({"held_units": held} if held else {}), **({"branch_limits": caps_shown} if caps_shown else {})}
        result.update(voltage_data)
        app_data["last_dispatch_result"] = regulation_data
        app_data["last_dispatch_timestamp"] = current_ts
        return result
    else:
        failed = {"timestamp": current_ts, "status": str(tc), "message": "Infeasible.",
                  "feasible": False, "converged": False, "activated_resources": [], "outage": outage, **islanding,
                  **({"held_units": held} if held else {}), **({"branch_limits": caps_shown} if caps_shown else {})}
        rescue = model_opt.clone() if model_opt is not None else None      # the diagnosis changes the model
        failed = _with_infeasibility_diagnosis(failed, model_opt, modified_net)
        out = _with_last_resort(failed, request, current_ts, rescue, modified_net, net_base, held, caps_shown,
                                fixed_setpoints, cap=None, outage=outage)
        return {**out, "outage": outage, **islanding}


# ---------------------------------------------------------------------------
# Robust OPF — constraint-tightening (back-off) endpoint
# ---------------------------------------------------------------------------

class RobustFlexibilityRequest(_TimeseriesRequest):
    robust_method: str = "heuristic"  # "heuristic" | "scenario"
    sgen_sigma: float = 0.0
    load_sigma: float = 0.05
    n_samples: int = 200
    risk_target: float = 0.05
    confidence: float | None = None
    vm_upper_pu: float = 1.05
    vm_lower_pu: float = 0.95
    max_line_loading_pct: float = 90.0
    max_trafo_loading_pct: float = 90.0
    slack_max_mw: float | None = None
    opf_lambda_p: float = 0.01
    opf_lambda_q: float = 0.001
    opf_min_power_factor: float = 0.95
    load_scaling_factor: float = 1.0
    target_p_any: float | None = None
    max_iter: int = 3
    min_improvement: float = 0.005
    validation_samples: int | None = None
    alpha: float | None = None
    beta: float = 1e-3
    n_scenarios: int | None = None
    scenario_k_cap: int = 120
    allowed_violation_fraction: float = 0.0


@app.post("/api/flexibility/robust")
@_off_event_loop
async def robust_flexibility(request: RobustFlexibilityRequest):
    """Robust OPF with selectable methods.

    robust_method="heuristic" (default): existing percentile back-off loop.
    robust_method="scenario": scenario-based chance-constrained AC-OPF
    with shared first-stage dispatch and scenario-indexed AC states.

    Scenario K sizing uses the convex Calafiore-Campi style bound:
      K >= (2/alpha) * (ln(1/beta) + d)
    where d is the number of first-stage decision variables.

    Warm start for scenario mode:
    1) solve deterministic flex OPF on forecast base case,
    2) initialize Pg/Qg and scenario V/theta from that solved state,
    3) solve scenario AC-OPF with IPOPT warm-start options.

    Renewable min() handling in scenario mode uses smooth inequalities:
      P_inj[g,k] <= P_new[g], P_inj[g,k] <= xi[g,k]
    plus a tiny injection reward in the objective so P_inj follows min().

    Heuristic steps:
    1. Run probabilistic RSA to get per-bus voltage percentile envelope.
    2. Compute per-bus back-offs from probabilistic voltage envelopes:
       - upper: Δu_b = max(0, V_b,pct - V_b,base)
       - lower: Δl_b = max(0, V_b,base - V_b,1-pct)
    3. Solve OPF with tightened per-bus bounds:
       - vm_upper_b = vm_upper_pu - Δu_b
       - vm_lower_b = vm_lower_pu + Δl_b
    4. Re-run probabilistic RSA with the new dispatch applied to quantify
       the risk reduction (p_any_violation_after).

    Returns everything from optimize_flexibility plus:
    - back_off_per_bus: {bus_name: Δu_b} (legacy alias for upper back-off)
    - tightened_bounds: {bus_name: effective_upper} (legacy alias for upper bounds)
    - back_off_upper_per_bus / back_off_lower_per_bus
    - tightened_upper_bounds / tightened_lower_bounds
    - p_any_violation_before / p_any_violation_after
    - n_samples, sgen_sigma, confidence used
    """
    from scipy.stats.qmc import LatinHypercube
    import scipy.stats as sp_norm_mod
    import numpy as np

    if _db_for("full", request) is None:
        raise HTTPException(status_code=503, detail="Data not loaded.")

    current_ts = _resolve_tick(request.data_source, request.timestamp)
    net_base = copy.deepcopy(_net_for(request))
    measurement_prod_cons = prepare_measurement_df(_lookup_measurement_df(current_ts, request.data_source))
    net_base, _ = _assign_tick(net_base, measurement_prod_cons, current_ts, _dataset_of(request))

    bus_idx_to_name = {int(idx): _bus_display_name(net_base, int(idx))
                       for idx in net_base.bus.index}

    line_idx_to_name = {int(idx): _line_display_name(net_base, int(idx))
                        for idx in net_base.line.index}
    trafo_idx_to_name = {int(idx): _trafo_display_name(net_base, int(idx))
                         for idx in net_base.trafo.index}

    try:
        pp.runpp(net_base, algorithm="nr", numba=False)
    except Exception:
        raise HTTPException(status_code=500, detail="Base power flow did not converge.")

    # -- Step 1: base voltages -------------------------------------------------
    base_vm = {bus_idx_to_name.get(int(i), str(i)): float(net_base.res_bus.at[i, "vm_pu"])
               for i in net_base.bus.index}

    # Generation uncertainty inputs (Step 2), with all sgens treated as
    # uncertain generators for now (Step 1 intentionally skipped).
    if len(net_base.sgen.index) > 0:
        base_sgen_p = net_base.sgen["p_mw"].to_numpy(dtype=float, copy=True)
        if "max_p_mw" in net_base.sgen.columns:
            raw_pmax = net_base.sgen["max_p_mw"].to_numpy(dtype=float, copy=True)
        else:
            raw_pmax = base_sgen_p.copy()
        sgen_p_max = np.where(np.isfinite(raw_pmax), raw_pmax, base_sgen_p)
        sgen_p_max = np.maximum(sgen_p_max, 0.0)
    else:
        base_sgen_p = np.array([], dtype=float)
        sgen_p_max = np.array([], dtype=float)

    robust_method = str(request.robust_method or "heuristic").strip().lower()
    if robust_method not in {"heuristic", "scenario"}:
        raise HTTPException(status_code=400, detail="robust_method must be 'heuristic' or 'scenario'.")

    # Canonical risk parameter with deterministic precedence:
    # risk_target wins. Aliases alpha/confidence are consulted only when
    # risk_target is absent. Inconsistent duplicates are rejected.
    fields_set = set(getattr(request, "model_fields_set", set()))
    risk_tolerance = 1e-6
    risk_default = 0.05

    risk_target_raw = request.risk_target if "risk_target" in fields_set else None
    alpha_alias = request.alpha if "alpha" in fields_set else None
    confidence_alias = request.confidence if "confidence" in fields_set else None

    if risk_target_raw is not None:
        if alpha_alias is not None and abs(float(alpha_alias) - float(risk_target_raw)) > risk_tolerance:
            raise HTTPException(status_code=400, detail="Inconsistent risk inputs: risk_target conflicts with alpha.")
        if confidence_alias is not None and abs((1.0 - float(confidence_alias)) - float(risk_target_raw)) > risk_tolerance:
            raise HTTPException(status_code=400, detail="Inconsistent risk inputs: risk_target conflicts with confidence.")
        risk_target = float(risk_target_raw)
        risk_source = "risk_target"
    else:
        if alpha_alias is not None and confidence_alias is not None:
            if abs(float(alpha_alias) - (1.0 - float(confidence_alias))) > risk_tolerance:
                raise HTTPException(status_code=400, detail="Inconsistent risk inputs: alpha conflicts with confidence.")
        if alpha_alias is not None:
            risk_target = float(alpha_alias)
            risk_source = "alpha_alias"
        elif confidence_alias is not None:
            risk_target = 1.0 - float(confidence_alias)
            risk_source = "confidence_alias"
        else:
            risk_target = risk_default
            risk_source = "default"

    risk_target = max(1e-4, min(risk_target, 0.5))
    confidence_effective = max(0.5, min(0.999, 1.0 - risk_target))
    alpha_effective = risk_target
    target_p_any_effective = float(request.target_p_any) if request.target_p_any is not None else risk_target

    deprecated_aliases_used: list[str] = []
    if "alpha" in fields_set:
        deprecated_aliases_used.append("alpha")
    if "confidence" in fields_set:
        deprecated_aliases_used.append("confidence")
    if "target_p_any" in fields_set:
        deprecated_aliases_used.append("target_p_any")

    ignored_parameters: list[dict] = []

    def _mark_ignored(name: str, active_mode: str, reason: str):
        ignored_parameters.append({"name": name, "active_mode": active_mode, "reason": reason})

    def _is_non_default(name: str, value, default):
        if name not in fields_set:
            return False
        if default is None:
            return value is not None
        if isinstance(default, float):
            return abs(float(value) - float(default)) > 1e-12
        return value != default

    if robust_method == "scenario":
        if _is_non_default("max_iter", request.max_iter, 3):
            _mark_ignored("max_iter", "scenario", "Only used by heuristic iterative tightening.")
        if _is_non_default("min_improvement", request.min_improvement, 0.005):
            _mark_ignored("min_improvement", "scenario", "Only used by heuristic iterative tightening.")
        if _is_non_default("target_p_any", request.target_p_any, None):
            _mark_ignored("target_p_any", "scenario", "Scenario mode uses risk_target/alpha and K-based constraints.")
    else:
        if _is_non_default("beta", request.beta, 1e-3):
            _mark_ignored("beta", "heuristic", "Only used by scenario method K sizing/guarantee metadata.")
        if _is_non_default("n_scenarios", request.n_scenarios, None):
            _mark_ignored("n_scenarios", "heuristic", "Only used by scenario method.")
        if _is_non_default("scenario_k_cap", request.scenario_k_cap, 120):
            _mark_ignored("scenario_k_cap", "heuristic", "Only used by scenario method.")
        if _is_non_default("allowed_violation_fraction", request.allowed_violation_fraction, 0.0):
            _mark_ignored("allowed_violation_fraction", "heuristic", "Only used by scenario discard variant.")

    response_param_meta = {
        "risk_target": round(float(risk_target), 6),
        "risk_target_source": risk_source,
        "confidence_effective": round(float(confidence_effective), 6),
        "alpha_effective": round(float(alpha_effective), 6),
        "target_p_any_effective": round(float(target_p_any_effective), 6),
        "deprecated_aliases_used": deprecated_aliases_used,
        "ignored_parameters": ignored_parameters,
    }

    # -- Step 2: sampling (calibration + validation) -------------------------
    n = max(10, min(request.n_samples, 1000))
    n_validation = max(10, min(request.validation_samples if request.validation_samples is not None else n, 1000))
    sigma_s = max(0.0, request.sgen_sigma)
    sigma_l = max(0.0, request.load_sigma)
    pctile = confidence_effective

    def _draw_multipliers(n_draws: int, seed: int):
        sampler = LatinHypercube(d=2, seed=seed)
        u = sampler.random(n=n_draws)
        if sigma_s <= 0:
            sm = np.ones(n_draws)
        else:
            sm = sp_norm_mod.norm.ppf(u[:, 0], loc=1.0, scale=sigma_s).clip(0.01, 3.0)
        if sigma_l <= 0:
            lm = np.ones(n_draws)
        else:
            lm = sp_norm_mod.norm.ppf(u[:, 1], loc=1.0, scale=sigma_l).clip(0.01, 3.0)
        return sm, lm

    sgen_mults_cal, load_mults_cal = _draw_multipliers(n, seed=42)
    sgen_mults_val, load_mults_val = _draw_multipliers(n_validation, seed=4242)

    def _extract_violation_sets(_net):
        bus_over = {
            bus_idx_to_name.get(int(i), str(i))
            for i in _net.bus.index
            if float(_net.res_bus.at[i, "vm_pu"]) > request.vm_upper_pu
        }
        bus_under = {
            bus_idx_to_name.get(int(i), str(i))
            for i in _net.bus.index
            if float(_net.res_bus.at[i, "vm_pu"]) < request.vm_lower_pu
        }
        bus_viol = bus_over | bus_under

        line_viol = set()
        if len(_net.line.index) > 0 and "loading_percent" in _net.res_line.columns:
            for i in _net.line.index:
                lp = _net.res_line.at[i, "loading_percent"]
                if pd.notna(lp) and float(lp) > request.max_line_loading_pct:
                    line_viol.add(line_idx_to_name.get(int(i), f"Line_{int(i)}"))

        trafo_viol = set()
        if len(_net.trafo.index) > 0 and "loading_percent" in _net.res_trafo.columns:
            for i in _net.trafo.index:
                tp = _net.res_trafo.at[i, "loading_percent"]
                if pd.notna(tp) and float(tp) > request.max_trafo_loading_pct:
                    trafo_viol.add(trafo_idx_to_name.get(int(i), f"Trafo_{int(i)}"))
        return bus_viol, bus_over, bus_under, line_viol, trafo_viol

    def _build_center_net(p_ceiling: np.ndarray, q_set: np.ndarray):
        _net = copy.deepcopy(net_base)
        if len(_net.sgen.index) > 0:
            _net.sgen["p_mw"] = p_ceiling
            _net.sgen["q_mvar"] = q_set
        _scale_loads(_net, request.load_scaling_factor)
        return _net

    def _to_prob_dict(counts: dict[str, int], denom: int) -> dict[str, float]:
        if denom <= 0:
            return {}
        return {
            k: round(v / denom, 4)
            for k, v in counts.items()
            if v > 0
        }

    def _evaluate_dispatch(p_ceiling: np.ndarray, q_set: np.ndarray, sm_arr, lm_arr, collect_vm: bool = False):
        center_net = _build_center_net(p_ceiling, q_set)
        try:
            pp.runpp(center_net, algorithm="nr", numba=False)
        except Exception:
            return None

        base_vm_current = {
            bus_idx_to_name.get(int(i), str(i)): float(center_net.res_bus.at[i, "vm_pu"])
            for i in center_net.bus.index
        }

        bus_viol_counts: dict[str, int] = {name: 0 for name in base_vm_current}
        bus_over_counts: dict[str, int] = {name: 0 for name in base_vm_current}
        bus_under_counts: dict[str, int] = {name: 0 for name in base_vm_current}
        line_viol_counts: dict[str, int] = {name: 0 for name in line_idx_to_name.values()}
        trafo_viol_counts: dict[str, int] = {name: 0 for name in trafo_idx_to_name.values()}
        vm_samples: dict[str, list[float]] = {name: [] for name in base_vm_current} if collect_vm else {}
        viol_count_list: list[int] = []
        any_bus_over = 0
        any_bus_under = 0
        curtailment_samples: list[float] = []

        def _run_one(sm, lm):
            _net = copy.deepcopy(center_net)
            curtailment_mw = 0.0
            if len(_net.sgen.index) > 0:
                xi = np.clip(base_sgen_p * float(sm), 0.0, sgen_p_max)
                p_inj = np.minimum(p_ceiling, xi)
                _net.sgen["p_mw"] = p_inj
                _net.sgen["q_mvar"] = q_set
                curtailment_mw = float(np.maximum(xi - p_ceiling, 0.0).sum())
            _scale_loads(_net, float(lm))
            try:
                pp.runpp(_net, algorithm="nr", numba=False)
            except Exception:
                return None
            vms = {
                bus_idx_to_name.get(int(i), str(i)): float(_net.res_bus.at[i, "vm_pu"])
                for i in _net.bus.index
            }
            bus_viol, bus_over, bus_under, line_viol, trafo_viol = _extract_violation_sets(_net)
            n_viol = len(bus_viol) + len(line_viol) + len(trafo_viol)
            return vms, n_viol, bus_viol, bus_over, bus_under, line_viol, trafo_viol, curtailment_mw

        with ThreadPoolExecutor(max_workers=min(len(sm_arr), 8)) as pool:
            futures = {
                pool.submit(_run_one, sm, lm): i
                for i, (sm, lm) in enumerate(zip(sm_arr, lm_arr))
            }
            for fut in as_completed(futures):
                r = fut.result()
                if r is None:
                    continue
                vms, n_viol, bus_viol, bus_over, bus_under, line_viol, trafo_viol, curtailment_mw = r
                viol_count_list.append(n_viol)
                curtailment_samples.append(curtailment_mw)
                if bus_over:
                    any_bus_over += 1
                if bus_under:
                    any_bus_under += 1
                for name in set(bus_viol):  # once per sample; see probabilistic_rsa
                    if name in bus_viol_counts:
                        bus_viol_counts[name] += 1
                for name in bus_over:
                    if name in bus_over_counts:
                        bus_over_counts[name] += 1
                for name in bus_under:
                    if name in bus_under_counts:
                        bus_under_counts[name] += 1
                for name in line_viol:
                    if name in line_viol_counts:
                        line_viol_counts[name] += 1
                for name in trafo_viol:
                    if name in trafo_viol_counts:
                        trafo_viol_counts[name] += 1
                if collect_vm:
                    for bname, vm in vms.items():
                        if bname in vm_samples:
                            vm_samples[bname].append(vm)

        n_conv = len(viol_count_list)
        if n_conv <= 0:
            return None

        out = {
            "n_converged": n_conv,
            "p_any": sum(1 for v in viol_count_list if v > 0) / n_conv,
            "p_any_bus_over": any_bus_over / n_conv,
            "p_any_bus_under": any_bus_under / n_conv,
            "bus_violation_probability": _to_prob_dict(bus_viol_counts, n_conv),
            "bus_overvoltage_probability": _to_prob_dict(bus_over_counts, n_conv),
            "bus_undervoltage_probability": _to_prob_dict(bus_under_counts, n_conv),
            "line_violation_probability": _to_prob_dict(line_viol_counts, n_conv),
            "trafo_violation_probability": _to_prob_dict(trafo_viol_counts, n_conv),
            "expected_curtailment_mw": float(np.mean(curtailment_samples)) if curtailment_samples else 0.0,
            "base_vm": base_vm_current,
        }

        if collect_vm:
            pctile_vm_upper: dict[str, float] = {}
            pctile_vm_lower: dict[str, float] = {}
            for bname, vals in vm_samples.items():
                if vals:
                    pctile_vm_upper[bname] = float(np.percentile(vals, pctile * 100))
                    pctile_vm_lower[bname] = float(np.percentile(vals, (1.0 - pctile) * 100))

            back_off_upper: dict[str, float] = {}
            back_off_lower: dict[str, float] = {}
            tightened_upper: dict[str, float] = {}
            tightened_lower: dict[str, float] = {}
            for bname, base_v in base_vm_current.items():
                p_v_up = pctile_vm_upper.get(bname, base_v)
                p_v_low = pctile_vm_lower.get(bname, base_v)
                delta_up = max(0.0, p_v_up - base_v)
                delta_low = max(0.0, base_v - p_v_low)
                back_off_upper[bname] = round(delta_up, 6)
                back_off_lower[bname] = round(delta_low, 6)
                tightened_upper[bname] = round(request.vm_upper_pu - delta_up, 6)
                tightened_lower[bname] = round(request.vm_lower_pu + delta_low, 6)

            out.update({
                "back_off_upper": back_off_upper,
                "back_off_lower": back_off_lower,
                "tightened_upper": tightened_upper,
                "tightened_lower": tightened_lower,
            })

        return out

    # ------------------------------------------------------------------
    # Scenario mode: sample-based chance-constrained AC-OPF
    # ------------------------------------------------------------------
    base_q_set = (
        net_base.sgen["q_mvar"].to_numpy(dtype=float, copy=True)
        if len(net_base.sgen.index) > 0 else np.array([], dtype=float)
    )

    if robust_method == "scenario":
        # Baseline risk before control using calibration draws.
        eval_before = _evaluate_dispatch(base_sgen_p.copy(), base_q_set.copy(), sgen_mults_cal, load_mults_cal, collect_vm=False)
        if eval_before is None:
            raise HTTPException(status_code=500, detail="Scenario mode baseline risk evaluation failed.")

        # Build deterministic warm-start dispatch on base forecast.
        db = _db_for("full", request)
        net_warm = copy.deepcopy(net_base)
        _scale_loads(net_warm, request.load_scaling_factor)
        try:
            pp.runpp(net_warm, algorithm="nr", numba=False)
        except Exception:
            pass

        det_results, det_model = fe.optimization_model_base(
            net_warm,
            net_base,
            db["Yff_r"], db["Yff_i"], db["Yft_r"], db["Yft_i"],
            db["TAPS"], db["trafo_ranges"], db["trafo_defaults"], db["branch_to_trafo"],
            missing_substation=None,
            opf_vm_lower=request.vm_lower_pu,
            opf_vm_upper=request.vm_upper_pu,
            lambda_p=request.opf_lambda_p,
            lambda_q=request.opf_lambda_q,
            pg_max_overrides=None,
            pg_min_overrides=None,
            min_power_factor=request.opf_min_power_factor,
            current_safety_margin=0.9,
            fixed_setpoints=None,
            vm_upper_per_bus=None,
            vm_lower_per_bus=None,
        )
        tc_det = getattr(det_results.solver, "termination_condition", None) if det_results is not None else None
        if tc_det != TerminationCondition.optimal:
            return {
                "timestamp": current_ts,
                "status": str(tc_det),
                "message": (
                    "Scenario warm-start deterministic OPF failed. Consider setting "
                    "allowed_violation_fraction to 0.01-0.02 to relax from strict-robust "
                    "semantics to chance/discard semantics."
                ),
                "feasible": False,
                "guarantee_met": False,
                "robust_method_requested": robust_method,
                "robust_method_effective": "scenario",
                "reason_code": "scenario_warmstart_infeasible",
                "fallback_policy_used": "none",
                "activated_resources": [],
                **response_param_meta,
            }

        ws_pg = {str(g): float(value(det_model.Pg_new[g])) for g in det_model.G}
        ws_qg = {str(g): float(value(det_model.Qg_new[g])) for g in det_model.G}
        ws_v = {int(b): float(value(det_model.V[b])) for b in det_model.B}
        ws_theta = {int(b): float(value(det_model.theta[b])) for b in det_model.B}

        # First-stage decision count d = |Pg_new| + |Qg_new|.
        d = 2 * len(list(det_model.G))
        alpha = max(1e-4, min(float(alpha_effective), 0.5))
        beta = max(1e-9, min(float(request.beta), 0.5))

        # Convex scenario bound (Calafiore-Campi style):
        # K >= (2/alpha) * (ln(1/beta) + d)
        k_computed = int(math.ceil((2.0 / alpha) * (math.log(1.0 / beta) + d)))
        k_requested = int(request.n_scenarios) if request.n_scenarios is not None else k_computed
        k_cap = max(1, int(request.scenario_k_cap))
        k_effective = min(k_requested, k_cap)
        k_capped = bool(k_effective < k_requested)
        effective_alpha_upper_bound = min(1.0, max(1e-6, 2.0 * (math.log(1.0 / beta) + d) / max(1, k_effective)))
        # The bound is scenario theory for *convex* programs (Calafiore-Campi);
        # the AC OPF is not convex, so it is a nominal figure, not a
        # certificate. The out-of-sample validation risk is the evidence
        # (code review 2.6).
        guarantee_interpretation = "nominal_bound_convex_scenario_theory_not_certified_for_ac_opf"
        guarantee_note = (
            "effective_alpha_upper_bound comes from scenario theory for convex problems; the AC OPF is "
            "not convex, so it is a nominal figure, not a certificate. guarantee_met means only that the "
            "scenario count reached that nominal bound. p_any_violation_after_validation (out of sample) "
            "is the evidence of the risk that remains."
        )
        guarantee_met = not k_capped

        # Training scenarios: K draws. Sliced from the calibration pool when it
        # is large enough (as before); drawn on their own when it is not — the
        # slice silently trained on n_samples scenarios while reporting K.
        if len(sgen_mults_cal) >= k_effective:
            sm_train = sgen_mults_cal[:k_effective]
            lm_train = load_mults_cal[:k_effective]
        else:
            sm_train, lm_train = _draw_multipliers(k_effective, seed=42)

        # Build scenario xi[g,k] for uncertain sgens; non-uncertain entries
        # default to Pg_max in the engine.
        sgen_name_by_idx = {}
        for idx, row in net_base.sgen.iterrows():
            raw_sub = row.get("substation_name", None)
            raw_name = row.get("name", None)
            name = None
            if pd.notna(raw_sub) and str(raw_sub).strip():
                name = str(raw_sub).strip()
            elif pd.notna(raw_name) and str(raw_name).strip():
                name = str(raw_name).strip()
            else:
                name = f"sgen_{int(idx)}"
            sgen_name_by_idx[int(idx)] = name

        xi_by_gen = {}
        xi_min_seen = float("inf")
        xi_max_seen = float("-inf")
        xi_oob_count = 0
        for idx in range(len(base_sgen_p)):
            gname = sgen_name_by_idx.get(int(net_base.sgen.index[idx]), f"sgen_{int(net_base.sgen.index[idx])}")
            vals = np.clip(base_sgen_p[idx] * sm_train, 0.0, sgen_p_max[idx]).tolist()
            xi_by_gen[gname] = [float(v) for v in vals]
            if vals:
                vmin = float(min(vals))
                vmax = float(max(vals))
                xi_min_seen = min(xi_min_seen, vmin)
                xi_max_seen = max(xi_max_seen, vmax)
                pmax = float(sgen_p_max[idx])
                xi_oob_count += int(sum((v < -1e-9) or (v > pmax + 1e-9) for v in vals))

        if not np.isfinite(xi_min_seen):
            xi_min_seen = 0.0
        if not np.isfinite(xi_max_seen):
            xi_max_seen = 0.0

        sc_results, sc_model, sc_meta = fe.scenario_optimization_model_base(
            net_warm,
            net_base,
            db["Yff_r"], db["Yff_i"], db["Yft_r"], db["Yft_i"],
            db["TAPS"], db["trafo_ranges"], db["trafo_defaults"], db["branch_to_trafo"],
            xi_by_gen=xi_by_gen,
            load_multipliers=[float(x) for x in lm_train],
            opf_vm_lower=request.vm_lower_pu,
            opf_vm_upper=request.vm_upper_pu,
            lambda_p=request.opf_lambda_p,
            lambda_q=request.opf_lambda_q,
            pg_max_overrides=None,
            pg_min_overrides=None,
            min_power_factor=request.opf_min_power_factor,
            current_safety_margin=0.9,
            fixed_setpoints=None,
            allowed_violation_fraction=max(0.0, min(float(request.allowed_violation_fraction), 0.5)),
            warm_start={"Pg_new": ws_pg, "Qg_new": ws_qg, "V": ws_v, "theta": ws_theta},
        )

        tc_sc = getattr(sc_results.solver, "termination_condition", None) if sc_results is not None else None
        if tc_sc != TerminationCondition.optimal:
            infeasible_reason = "k_required_exceeds_cap" if k_capped else "scenario_model_infeasible"
            infeasible_msg = (
                "Scenario robust OPF infeasible. Consider setting allowed_violation_fraction to 0.01-0.02 "
                "to relax from strict-robust semantics to chance/discard semantics."
            )
            return {
                "timestamp": current_ts,
                "status": str(tc_sc),
                "message": infeasible_msg,
                "feasible": False,
                "guarantee_met": False,
                "robust_method": "scenario",
                "robust_method_requested": robust_method,
                "robust_method_effective": "scenario",
                "reason_code": infeasible_reason,
                "fallback_policy_used": "none",
                "alpha": alpha,
                "beta": beta,
                "d": d,
                "K": k_effective,
                "K_requested": k_requested,
                "K_computed": k_computed,
                "K_capped": k_capped,
                "effective_alpha_upper_bound": round(float(effective_alpha_upper_bound), 6),
                "effective_beta_fixed": round(float(beta), 12),
                "guarantee_interpretation": guarantee_interpretation,
                "guarantee_note": guarantee_note,
                "K_trained": int(len(sm_train)),
                "implied_alpha": round(float(effective_alpha_upper_bound), 6),
                "allowed_violation_fraction": max(0.0, min(float(request.allowed_violation_fraction), 0.5)),
                "scenario_discard_indicator_relaxation": bool(sc_meta.get("discard_indicator_relaxation", False)),
                "scenario_warnings": ([f"Computed K={k_requested} exceeded cap {k_cap}; clamped to {k_effective}."] if k_capped else []),
                "scenario_xi_min": round(float(xi_min_seen), 6),
                "scenario_xi_max": round(float(xi_max_seen), 6),
                "scenario_xi_oob_count": int(xi_oob_count),
                "p_any_violation_before": round(float(eval_before["p_any"]), 4),
                "p_any_bus_overvoltage_before": round(float(eval_before["p_any_bus_over"]), 4),
                "p_any_bus_undervoltage_before": round(float(eval_before["p_any_bus_under"]), 4),
                "activated_resources": [],
                **response_param_meta,
            }

        # Extract scenario solution to arrays and UI-compatible outputs.
        p_ceiling_after = base_sgen_p.copy()
        q_set_after = base_q_set.copy()

        for g in sc_model.G:
            gname = str(g)
            pnew = float(value(sc_model.Pg_new[g]))
            qnew = float(value(sc_model.Qg_new[g]))
            if gname in sgen_name_by_idx.values():
                # Map back by name; fallback by first matching index.
                idx_match = None
                for idx, nm in sgen_name_by_idx.items():
                    if nm == gname:
                        idx_match = int(np.where(net_base.sgen.index.to_numpy() == idx)[0][0])
                        break
                if idx_match is not None:
                    p_ceiling_after[idx_match] = pnew
                    q_set_after[idx_match] = qnew

        eval_after_operational = _evaluate_dispatch(
            p_ceiling_after,
            q_set_after,
            sgen_mults_cal,
            load_mults_cal,
            collect_vm=False,
        )
        eval_after_validation = _evaluate_dispatch(
            p_ceiling_after,
            q_set_after,
            sgen_mults_val,
            load_mults_val,
            collect_vm=False,
        )

        p_any_before = float(eval_before["p_any"])
        p_any_after_fair_ab = float(eval_after_operational["p_any"]) if eval_after_operational is not None else None
        p_any_after_operational = p_any_after_fair_ab
        p_any_after_validation = float(eval_after_validation["p_any"]) if eval_after_validation is not None else None
        p_any_after = p_any_after_validation if p_any_after_validation is not None else p_any_after_fair_ab

        # Activated resources from scenario first-stage dispatch.
        # Keep all controllable elements (including unchanged ones) so the
        # dispatch chart shows the full generator/slack portfolio.
        activated_resources = []
        for g in sc_model.G:
            gname = str(g)
            pg_base = float(value(sc_model.Pg_base[g]))
            pg_new = float(value(sc_model.Pg_new[g]))
            qg_base = float(value(sc_model.Qg_base[g]))
            qg_new = float(value(sc_model.Qg_new[g]))
            activated_resources.append({
                "element": gname,
                "type": "Generator",
                "Pg_base": round(pg_base, 4),
                "Pg_new": round(pg_new, 4),
                "Pg_up": round(max(pg_new - pg_base, 0.0), 4),
                "Pg_down": round(max(pg_base - pg_new, 0.0), 4),
                "Qg_base": round(qg_base, 4),
                "Qg_new": round(qg_new, 4),
                "Qg_up": round(max(qg_new - qg_base, 0.0), 4),
                "Qg_down": round(max(qg_base - qg_new, 0.0), 4),
            })

        for e in sc_model.Ext:
            ename = str(e)
            pext_new = float(value(sc_model.Pext[e, 0]))
            qext_new = float(value(sc_model.Qext[e, 0]))
            if hasattr(net_warm, "res_ext_grid") and not net_warm.res_ext_grid.empty:
                pext_base = float(net_warm.res_ext_grid["p_mw"].iloc[0])
                qext_base = float(net_warm.res_ext_grid["q_mvar"].iloc[0])
            else:
                pext_base = pext_new
                qext_base = qext_new

            activated_resources.append({
                "element": ename,
                "type": "External Grid",
                "Pg_base": round(pext_base, 4),
                "Pg_new": round(pext_new, 4),
                "Pg_up": round(max(pext_new - pext_base, 0.0), 4),
                "Pg_down": round(max(pext_base - pext_new, 0.0), 4),
                "Qg_base": round(qext_base, 4),
                "Qg_new": round(qext_new, 4),
                "Qg_up": round(max(qext_new - qext_base, 0.0), 4),
                "Qg_down": round(max(qext_base - qext_new, 0.0), 4),
            })

        voltage_data = fe.extract_post_opf_voltages_scenario(
            sc_model,
            net_warm,
            scenario_idx=0,
            display_vm_lower=request.vm_lower_pu,
            display_vm_upper=request.vm_upper_pu,
        )

        # Curtailment the plan relies on in its scenarios: reported so the
        # operator sees it — the dispatch above is setpoints, not injections.
        try:
            curtailment = fe.scenario_curtailment(sc_model)
        except Exception:  # noqa: BLE001
            curtailment = None

        # Shared response structure with heuristic path.
        sc_msg = "Scenario robust OPF solved successfully."
        if curtailment and curtailment["scenarios_with_curtailment"]:
            top = curtailment["by_unit"][0]
            sc_msg += (
                f" In {curtailment['scenarios_with_curtailment']} of {curtailment['n_scenarios']} "
                f"scenarios the plan relies on curtailing renewable output (up to "
                f"{curtailment['max_total_mw']:.2f} MW in total; most at {top['unit']}, up to "
                f"{top['max_mw']:.2f} MW). The dispatch shown is the setpoint, not what is "
                "injected in those scenarios."
            )
        _excluded_sc = int(voltage_data.get("excluded_bus_count_post_opf") or 0)
        if _excluded_sc > 0:
            sc_msg += f" Excluded {_excluded_sc} bus(es) from post-OPF voltage output; see excluded_buses_post_opf for details."

        result = {
            "timestamp": current_ts,
            "status": "optimal",
            "message": sc_msg,
            "feasible": True,
            "guarantee_met": guarantee_met,
            "nominal_bound_met": guarantee_met,
            "activated_resources": activated_resources,
            "scenario_curtailment": curtailment,
            **voltage_data,
            "back_off_per_bus": {},
            "tightened_bounds": {},
            "back_off_upper_per_bus": {},
            "back_off_lower_per_bus": {},
            "tightened_upper_bounds": {},
            "tightened_lower_bounds": {},
            "p_any_violation_before": round(p_any_before, 4),
            "p_any_violation_after": round(p_any_after, 4) if p_any_after is not None else None,
            "p_any_violation_after_operational": round(p_any_after_operational, 4) if p_any_after_operational is not None else None,
            "p_any_violation_after_fair_ab": round(p_any_after_fair_ab, 4) if p_any_after_fair_ab is not None else None,
            "p_any_violation_after_validation": round(p_any_after_validation, 4) if p_any_after_validation is not None else None,
            "p_any_bus_overvoltage_before": round(float(eval_before["p_any_bus_over"]), 4),
            "p_any_bus_undervoltage_before": round(float(eval_before["p_any_bus_under"]), 4),
            "p_any_bus_overvoltage_after": round(float(eval_after_validation["p_any_bus_over"]), 4) if eval_after_validation is not None else None,
            "p_any_bus_undervoltage_after": round(float(eval_after_validation["p_any_bus_under"]), 4) if eval_after_validation is not None else None,
            "p_any_bus_overvoltage_after_operational": round(float(eval_after_operational["p_any_bus_over"]), 4) if eval_after_operational is not None else None,
            "p_any_bus_undervoltage_after_operational": round(float(eval_after_operational["p_any_bus_under"]), 4) if eval_after_operational is not None else None,
            "p_any_bus_overvoltage_after_fair_ab": round(float(eval_after_operational["p_any_bus_over"]), 4) if eval_after_operational is not None else None,
            "p_any_bus_undervoltage_after_fair_ab": round(float(eval_after_operational["p_any_bus_under"]), 4) if eval_after_operational is not None else None,
            "bus_violation_probability_before": eval_before["bus_violation_probability"],
            "bus_overvoltage_probability_before": eval_before["bus_overvoltage_probability"],
            "bus_undervoltage_probability_before": eval_before["bus_undervoltage_probability"],
            "line_violation_probability_before": eval_before["line_violation_probability"],
            "trafo_violation_probability_before": eval_before["trafo_violation_probability"],
            "bus_violation_probability_after": eval_after_validation["bus_violation_probability"] if eval_after_validation is not None else {},
            "bus_overvoltage_probability_after": eval_after_validation["bus_overvoltage_probability"] if eval_after_validation is not None else {},
            "bus_undervoltage_probability_after": eval_after_validation["bus_undervoltage_probability"] if eval_after_validation is not None else {},
            "line_violation_probability_after": eval_after_validation["line_violation_probability"] if eval_after_validation is not None else {},
            "trafo_violation_probability_after": eval_after_validation["trafo_violation_probability"] if eval_after_validation is not None else {},
            "bus_violation_probability_after_operational": eval_after_operational["bus_violation_probability"] if eval_after_operational is not None else {},
            "bus_overvoltage_probability_after_operational": eval_after_operational["bus_overvoltage_probability"] if eval_after_operational is not None else {},
            "bus_undervoltage_probability_after_operational": eval_after_operational["bus_undervoltage_probability"] if eval_after_operational is not None else {},
            "line_violation_probability_after_operational": eval_after_operational["line_violation_probability"] if eval_after_operational is not None else {},
            "trafo_violation_probability_after_operational": eval_after_operational["trafo_violation_probability"] if eval_after_operational is not None else {},
            "bus_violation_probability_after_fair_ab": eval_after_operational["bus_violation_probability"] if eval_after_operational is not None else {},
            "bus_overvoltage_probability_after_fair_ab": eval_after_operational["bus_overvoltage_probability"] if eval_after_operational is not None else {},
            "bus_undervoltage_probability_after_fair_ab": eval_after_operational["bus_undervoltage_probability"] if eval_after_operational is not None else {},
            "line_violation_probability_after_fair_ab": eval_after_operational["line_violation_probability"] if eval_after_operational is not None else {},
            "trafo_violation_probability_after_fair_ab": eval_after_operational["trafo_violation_probability"] if eval_after_operational is not None else {},
            "bus_violation_probability_after_validation": eval_after_validation["bus_violation_probability"] if eval_after_validation is not None else {},
            "bus_overvoltage_probability_after_validation": eval_after_validation["bus_overvoltage_probability"] if eval_after_validation is not None else {},
            "bus_undervoltage_probability_after_validation": eval_after_validation["bus_undervoltage_probability"] if eval_after_validation is not None else {},
            "line_violation_probability_after_validation": eval_after_validation["line_violation_probability"] if eval_after_validation is not None else {},
            "trafo_violation_probability_after_validation": eval_after_validation["trafo_violation_probability"] if eval_after_validation is not None else {},
            "expected_curtailment_mw_before": round(float(eval_before["expected_curtailment_mw"]), 4),
            "expected_curtailment_mw_after": round(float(eval_after_validation["expected_curtailment_mw"]), 4) if eval_after_validation is not None else None,
            "expected_curtailment_mw_after_operational": round(float(eval_after_operational["expected_curtailment_mw"]), 4) if eval_after_operational is not None else None,
            "expected_curtailment_mw_after_fair_ab": round(float(eval_after_operational["expected_curtailment_mw"]), 4) if eval_after_operational is not None else None,
            "expected_curtailment_mw_after_validation": round(float(eval_after_validation["expected_curtailment_mw"]), 4) if eval_after_validation is not None else None,
            "deterministic_setpoint_curtailment_mw_post_opf": float(np.maximum(base_sgen_p - p_ceiling_after, 0.0).sum()) if len(base_sgen_p) > 0 else 0.0,
            "risk_comparison_mode": "fair_ab_paired_common_random_numbers_with_validation_split",
            "robust_loop_stop_reason": "scenario_single_solve",
            "robust_loop_iterations": [],
            "n_samples": n,
            "n_samples_validation": n_validation,
            "n_converged": int(eval_before["n_converged"]),
            "sgen_sigma": request.sgen_sigma,
            "confidence": confidence_effective,
            "vm_upper_per_bus_applied": {},
            "vm_lower_per_bus_applied": {},
            "robust_method": "scenario",
            "robust_method_requested": robust_method,
            "robust_method_effective": "scenario",
            "reason_code": ("k_required_exceeds_cap" if k_capped else None),
            "fallback_policy_used": "none",
            "alpha": alpha,
            "beta": beta,
            "d": int(d),
            "K": int(k_effective),
            "K_requested": int(k_requested),
            "K_computed": int(k_computed),
            "K_capped": bool(k_capped),
            "scenario_k_cap": int(k_cap),
            "allowed_violation_fraction": max(0.0, min(float(request.allowed_violation_fraction), 0.5)),
            "effective_alpha_upper_bound": round(float(effective_alpha_upper_bound), 6),
            "effective_beta_fixed": round(float(beta), 12),
            "guarantee_interpretation": guarantee_interpretation,
            "guarantee_note": guarantee_note,
            "K_trained": int(len(sm_train)),
            "implied_alpha": round(float(effective_alpha_upper_bound), 6),
            "scenario_discard_indicator_relaxation": bool(sc_meta.get("discard_indicator_relaxation", False)),
            "scenario_warnings": ([f"Computed K={k_requested} exceeded cap {k_cap}; clamped to {k_effective}."] if k_capped else []),
            "scenario_xi_min": round(float(xi_min_seen), 6),
            "scenario_xi_max": round(float(xi_max_seen), 6),
            "scenario_xi_oob_count": int(xi_oob_count),
            **response_param_meta,
        }
        app_data["last_dispatch_result"] = activated_resources
        app_data["last_dispatch_timestamp"] = current_ts
        return result

    # Name lookup for applying OPF dispatch back into arrays.
    sgen_name_to_idx: dict[str, int] = {}
    for idx, row in net_base.sgen.iterrows():
        sidx = int(idx)
        raw_sub = row.get("substation_name", None)
        if pd.notna(raw_sub) and str(raw_sub).strip():
            sgen_name_to_idx[str(raw_sub).strip()] = sidx
        raw_name = row.get("name", None)
        if pd.notna(raw_name) and str(raw_name).strip():
            sgen_name_to_idx[str(raw_name).strip()] = sidx
        bus_val = row.get("bus", None)
        try:
            bus_int = int(bus_val)
            sgen_name_to_idx[f"Bus {bus_int}"] = sidx
            sgen_name_to_idx[f"Gen_bus{bus_int}"] = sidx
        except Exception:
            pass

    def _apply_opf_dispatch_to_arrays(opf_res: dict, p_prev: np.ndarray, q_prev: np.ndarray):
        p_new = p_prev.copy()
        q_new = q_prev.copy()
        for res_row in opf_res.get("activated_resources", []):
            elem = str(res_row.get("element", ""))
            idx = sgen_name_to_idx.get(elem)
            if idx is None:
                continue
            if res_row.get("Pg_new") is not None:
                p_new[idx] = float(res_row["Pg_new"])
            if res_row.get("Qg_new") is not None:
                q_new[idx] = float(res_row["Qg_new"])
        return p_new, q_new

    # -- Step 3-6: iterative robust loop ------------------------------------
    p_ceiling_current = base_sgen_p.copy()
    q_set_current = net_base.sgen["q_mvar"].to_numpy(dtype=float, copy=True) if len(net_base.sgen.index) > 0 else np.array([], dtype=float)
    max_iter = max(1, min(int(request.max_iter), 10))
    min_improvement = max(0.0, float(request.min_improvement))
    target_p_any = target_p_any_effective

    iteration_history: list[dict] = []
    p_any_prev: float | None = None
    first_eval = None
    last_eval = None
    opf_result: dict = {}
    vm_upper_per_bus: dict[str, float] = {}
    vm_lower_per_bus: dict[str, float] = {}
    stop_reason = "max_iter"

    for iter_idx in range(1, max_iter + 1):
        eval_cal = _evaluate_dispatch(p_ceiling_current, q_set_current, sgen_mults_cal, load_mults_cal, collect_vm=True)
        if eval_cal is None:
            raise HTTPException(status_code=500, detail="Robust calibration samples failed to converge.")

        if first_eval is None:
            first_eval = eval_cal
        last_eval = eval_cal

        back_off_upper = eval_cal["back_off_upper"]
        back_off_lower = eval_cal["back_off_lower"]
        tightened_upper = eval_cal["tightened_upper"]
        tightened_lower = eval_cal["tightened_lower"]
        vm_upper_per_bus = {b: v for b, v in tightened_upper.items() if back_off_upper[b] > 0.0001}
        vm_lower_per_bus = {b: v for b, v in tightened_lower.items() if back_off_lower[b] > 0.0001}

        iteration_history.append({
            "iteration": iter_idx,
            "p_any_calibration": round(float(eval_cal["p_any"]), 4),
            "p_any_bus_overvoltage_calibration": round(float(eval_cal["p_any_bus_over"]), 4),
            "p_any_bus_undervoltage_calibration": round(float(eval_cal["p_any_bus_under"]), 4),
            "expected_curtailment_mw_calibration": round(float(eval_cal["expected_curtailment_mw"]), 4),
            "n_converged_calibration": int(eval_cal["n_converged"]),
            "n_buses_tightened_upper": int(len(vm_upper_per_bus)),
            "n_buses_tightened_lower": int(len(vm_lower_per_bus)),
        })

        if target_p_any is not None and float(eval_cal["p_any"]) <= float(target_p_any):
            stop_reason = "target_reached"
            break

        if p_any_prev is not None and abs(float(p_any_prev) - float(eval_cal["p_any"])) < min_improvement:
            stop_reason = "min_improvement"
            break
        p_any_prev = float(eval_cal["p_any"])

        flex_req = FlexibilityRequest(
            # The operating point must follow the request into the sub-call.
            # Without it the OPF resolved its own tick from the simulation
            # clock and solved a different day than the back-off calibration
            # above, then stamped the whole robust result with that day.
            timestamp=request.timestamp,
            data_source=request.data_source,
            network=request.network,
            load_scaling_factor=request.load_scaling_factor,
            slack_max_mw=request.slack_max_mw,
            opf_vm_upper=request.vm_upper_pu,
            opf_vm_lower=request.vm_lower_pu,
            opf_lambda_p=request.opf_lambda_p,
            opf_lambda_q=request.opf_lambda_q,
            vm_upper_per_bus=vm_upper_per_bus,
            vm_lower_per_bus=vm_lower_per_bus,
        )
        # Called in-process. It used to go over HTTP to a hard-coded
        # localhost:8000, which ignores BACKEND_PORT — on any other port the
        # sub-call failed, or reached a different backend with a different
        # network loaded. A refused OPF ends the loop exactly as before.
        try:
            opf_result = await optimise_flexibility(flex_req)
        except HTTPException as e:
            opf_result = {"status": "error", "detail": e.detail}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"OPF sub-call failed: {e}")

        if str(opf_result.get("status", "")).lower() != "optimal":
            stop_reason = "opf_nonoptimal"
            break

        if len(p_ceiling_current) > 0:
            p_ceiling_current, q_set_current = _apply_opf_dispatch_to_arrays(opf_result, p_ceiling_current, q_set_current)

    # If no OPF run happened in-loop, build a minimal compatible result view.
    if not opf_result:
        center_net_final = _build_center_net(p_ceiling_current, q_set_current)
        try:
            pp.runpp(center_net_final, algorithm="nr", numba=False)
            bus_voltages_post = [
                {
                    "bus": int(i),
                    "bus_name": bus_idx_to_name.get(int(i), f"Bus_{int(i)}"),
                    "vm_pu": round(float(center_net_final.res_bus.at[i, "vm_pu"]), 6),
                }
                for i in center_net_final.bus.index
            ]
        except Exception:
            bus_voltages_post = []
        fallback_status = "already_secure" if stop_reason == "target_reached" else "skipped"
        fallback_message = (
            "Robust target already satisfied at the current operating point; no OPF redispatch was required."
            if fallback_status == "already_secure"
            else f"Robust loop stopped ({stop_reason}) before OPF dispatch update."
        )
        opf_result = {
            "timestamp": current_ts,
            "status": fallback_status,
            "message": fallback_message,
            "activated_resources": [],
            "bus_voltages_post_opf": bus_voltages_post,
            "opf_vm_lower_used": request.vm_lower_pu,
            "opf_vm_upper_used": request.vm_upper_pu,
            "bus_voltages_base": [
                {
                    "bus": int(i),
                    "bus_name": bus_idx_to_name.get(int(i), f"Bus_{int(i)}"),
                    "vm_pu": round(float(net_base.res_bus.at[i, "vm_pu"]), 6),
                }
                for i in net_base.bus.index
            ],
        }

    # -- Step 7: calibration vs certification split --------------------------
    eval_after_operational = _evaluate_dispatch(
        p_ceiling_current,
        q_set_current,
        sgen_mults_cal,
        load_mults_cal,
        collect_vm=False,
    )
    eval_after_validation = _evaluate_dispatch(
        p_ceiling_current,
        q_set_current,
        sgen_mults_val,
        load_mults_val,
        collect_vm=False,
    )

    if first_eval is None:
        raise HTTPException(status_code=500, detail="Robust calibration did not produce any converged samples.")

    # Before is the first calibration evaluation (pre-control baseline).
    p_any_before = float(first_eval["p_any"])
    p_any_bus_overvoltage_before = round(float(first_eval["p_any_bus_over"]), 4)
    p_any_bus_undervoltage_before = round(float(first_eval["p_any_bus_under"]), 4)
    bus_violation_probability_before = first_eval["bus_violation_probability"]
    bus_overvoltage_probability_before = first_eval["bus_overvoltage_probability"]
    bus_undervoltage_probability_before = first_eval["bus_undervoltage_probability"]
    line_violation_probability_before = first_eval["line_violation_probability"]
    trafo_violation_probability_before = first_eval["trafo_violation_probability"]
    expected_curtailment_mw_before = round(float(first_eval["expected_curtailment_mw"]), 4)
    n_converged = int(first_eval["n_converged"])

    # Fair A/B (paired) currently uses calibration sample set.
    p_any_after_fair_ab = float(eval_after_operational["p_any"]) if eval_after_operational is not None else None
    p_any_after_operational = p_any_after_fair_ab

    if eval_after_operational is not None:
        p_any_bus_overvoltage_after_operational = round(float(eval_after_operational["p_any_bus_over"]), 4)
        p_any_bus_undervoltage_after_operational = round(float(eval_after_operational["p_any_bus_under"]), 4)
        p_any_bus_overvoltage_after_fair_ab = p_any_bus_overvoltage_after_operational
        p_any_bus_undervoltage_after_fair_ab = p_any_bus_undervoltage_after_operational
        bus_violation_probability_after_operational = eval_after_operational["bus_violation_probability"]
        line_violation_probability_after_operational = eval_after_operational["line_violation_probability"]
        trafo_violation_probability_after_operational = eval_after_operational["trafo_violation_probability"]
        bus_overvoltage_probability_after_operational = eval_after_operational["bus_overvoltage_probability"]
        bus_undervoltage_probability_after_operational = eval_after_operational["bus_undervoltage_probability"]
        bus_violation_probability_after_fair_ab = bus_violation_probability_after_operational
        line_violation_probability_after_fair_ab = line_violation_probability_after_operational
        trafo_violation_probability_after_fair_ab = trafo_violation_probability_after_operational
        bus_overvoltage_probability_after_fair_ab = bus_overvoltage_probability_after_operational
        bus_undervoltage_probability_after_fair_ab = bus_undervoltage_probability_after_operational
        expected_curtailment_mw_after_operational = round(float(eval_after_operational["expected_curtailment_mw"]), 4)
        expected_curtailment_mw_after_fair_ab = expected_curtailment_mw_after_operational
    else:
        p_any_bus_overvoltage_after_operational = None
        p_any_bus_undervoltage_after_operational = None
        p_any_bus_overvoltage_after_fair_ab = None
        p_any_bus_undervoltage_after_fair_ab = None
        bus_violation_probability_after_operational = {}
        line_violation_probability_after_operational = {}
        trafo_violation_probability_after_operational = {}
        bus_overvoltage_probability_after_operational = {}
        bus_undervoltage_probability_after_operational = {}
        bus_violation_probability_after_fair_ab = {}
        line_violation_probability_after_fair_ab = {}
        trafo_violation_probability_after_fair_ab = {}
        bus_overvoltage_probability_after_fair_ab = {}
        bus_undervoltage_probability_after_fair_ab = {}
        expected_curtailment_mw_after_operational = None
        expected_curtailment_mw_after_fair_ab = None

    # Certification (headline after-risk) uses independent validation set.
    if eval_after_validation is not None:
        p_any_after_validation = float(eval_after_validation["p_any"])
        p_any_bus_overvoltage_after_validation = round(float(eval_after_validation["p_any_bus_over"]), 4)
        p_any_bus_undervoltage_after_validation = round(float(eval_after_validation["p_any_bus_under"]), 4)
        bus_violation_probability_after_validation = eval_after_validation["bus_violation_probability"]
        line_violation_probability_after_validation = eval_after_validation["line_violation_probability"]
        trafo_violation_probability_after_validation = eval_after_validation["trafo_violation_probability"]
        bus_overvoltage_probability_after_validation = eval_after_validation["bus_overvoltage_probability"]
        bus_undervoltage_probability_after_validation = eval_after_validation["bus_undervoltage_probability"]
        expected_curtailment_mw_after_validation = round(float(eval_after_validation["expected_curtailment_mw"]), 4)
    else:
        p_any_after_validation = None
        p_any_bus_overvoltage_after_validation = None
        p_any_bus_undervoltage_after_validation = None
        bus_violation_probability_after_validation = {}
        line_violation_probability_after_validation = {}
        trafo_violation_probability_after_validation = {}
        bus_overvoltage_probability_after_validation = {}
        bus_undervoltage_probability_after_validation = {}
        expected_curtailment_mw_after_validation = None

    # Headline after values prefer validation (honest certification).
    p_any_after = p_any_after_validation if p_any_after_validation is not None else p_any_after_fair_ab
    p_any_bus_overvoltage_after = (
        p_any_bus_overvoltage_after_validation
        if p_any_bus_overvoltage_after_validation is not None
        else p_any_bus_overvoltage_after_fair_ab
    )
    p_any_bus_undervoltage_after = (
        p_any_bus_undervoltage_after_validation
        if p_any_bus_undervoltage_after_validation is not None
        else p_any_bus_undervoltage_after_fair_ab
    )
    bus_violation_probability_after = (
        bus_violation_probability_after_validation
        if bus_violation_probability_after_validation
        else bus_violation_probability_after_fair_ab
    )
    line_violation_probability_after = (
        line_violation_probability_after_validation
        if line_violation_probability_after_validation
        else line_violation_probability_after_fair_ab
    )
    trafo_violation_probability_after = (
        trafo_violation_probability_after_validation
        if trafo_violation_probability_after_validation
        else trafo_violation_probability_after_fair_ab
    )
    bus_overvoltage_probability_after = (
        bus_overvoltage_probability_after_validation
        if bus_overvoltage_probability_after_validation
        else bus_overvoltage_probability_after_fair_ab
    )
    bus_undervoltage_probability_after = (
        bus_undervoltage_probability_after_validation
        if bus_undervoltage_probability_after_validation
        else bus_undervoltage_probability_after_fair_ab
    )
    expected_curtailment_mw_after = (
        expected_curtailment_mw_after_validation
        if expected_curtailment_mw_after_validation is not None
        else expected_curtailment_mw_after_fair_ab
    )

    deterministic_setpoint_curtailment_mw = (
        float(np.maximum(base_sgen_p - p_ceiling_current, 0.0).sum()) if len(base_sgen_p) > 0 else 0.0
    )

    # Use latest calibration back-offs if available.
    back_off_upper = last_eval["back_off_upper"] if last_eval is not None else {}
    back_off_lower = last_eval["back_off_lower"] if last_eval is not None else {}
    tightened_upper = last_eval["tightened_upper"] if last_eval is not None else {}
    tightened_lower = last_eval["tightened_lower"] if last_eval is not None else {}

    result_status = str(opf_result.get("status", "")).lower()
    guarantee_met = (
        p_any_after is not None
        and (target_p_any is None or float(p_any_after) <= float(target_p_any))
        and result_status in {"optimal", "already_secure"}
    )
    result_feasible = result_status in {"optimal", "already_secure"}

    result = {
        **opf_result,
        "feasible": result_feasible,
        "guarantee_met": guarantee_met,
        "robust_method": "heuristic",
        "robust_method_requested": robust_method,
        "robust_method_effective": "heuristic",
        # Legacy aliases preserved for renderers currently plotting upper-tail tightening.
        "back_off_per_bus": back_off_upper,
        "tightened_bounds": tightened_upper,
        # New explicit symmetric tightening outputs.
        "back_off_upper_per_bus": back_off_upper,
        "back_off_lower_per_bus": back_off_lower,
        "tightened_upper_bounds": tightened_upper,
        "tightened_lower_bounds": tightened_lower,
        "p_any_violation_before": round(p_any_before, 4),
        "p_any_violation_after": round(p_any_after, 4) if p_any_after is not None else None,
        "p_any_violation_after_operational": (
            round(p_any_after_operational, 4) if p_any_after_operational is not None else None
        ),
        "p_any_violation_after_fair_ab": (
            round(p_any_after_fair_ab, 4) if p_any_after_fair_ab is not None else None
        ),
        "p_any_violation_after_validation": (
            round(p_any_after_validation, 4) if p_any_after_validation is not None else None
        ),
        "p_any_bus_overvoltage_before": p_any_bus_overvoltage_before,
        "p_any_bus_undervoltage_before": p_any_bus_undervoltage_before,
        "p_any_bus_overvoltage_after": p_any_bus_overvoltage_after,
        "p_any_bus_undervoltage_after": p_any_bus_undervoltage_after,
        "p_any_bus_overvoltage_after_operational": p_any_bus_overvoltage_after_operational,
        "p_any_bus_undervoltage_after_operational": p_any_bus_undervoltage_after_operational,
        "p_any_bus_overvoltage_after_fair_ab": p_any_bus_overvoltage_after_fair_ab,
        "p_any_bus_undervoltage_after_fair_ab": p_any_bus_undervoltage_after_fair_ab,
        "p_any_bus_overvoltage_after_validation": p_any_bus_overvoltage_after_validation,
        "p_any_bus_undervoltage_after_validation": p_any_bus_undervoltage_after_validation,
        "bus_violation_probability_before": bus_violation_probability_before,
        "bus_overvoltage_probability_before": bus_overvoltage_probability_before,
        "bus_undervoltage_probability_before": bus_undervoltage_probability_before,
        "line_violation_probability_before": line_violation_probability_before,
        "trafo_violation_probability_before": trafo_violation_probability_before,
        "bus_violation_probability_after": bus_violation_probability_after,
        "bus_overvoltage_probability_after": bus_overvoltage_probability_after,
        "bus_undervoltage_probability_after": bus_undervoltage_probability_after,
        "line_violation_probability_after": line_violation_probability_after,
        "trafo_violation_probability_after": trafo_violation_probability_after,
        "bus_violation_probability_after_operational": bus_violation_probability_after_operational,
        "bus_overvoltage_probability_after_operational": bus_overvoltage_probability_after_operational,
        "bus_undervoltage_probability_after_operational": bus_undervoltage_probability_after_operational,
        "line_violation_probability_after_operational": line_violation_probability_after_operational,
        "trafo_violation_probability_after_operational": trafo_violation_probability_after_operational,
        "bus_violation_probability_after_fair_ab": bus_violation_probability_after_fair_ab,
        "bus_overvoltage_probability_after_fair_ab": bus_overvoltage_probability_after_fair_ab,
        "bus_undervoltage_probability_after_fair_ab": bus_undervoltage_probability_after_fair_ab,
        "line_violation_probability_after_fair_ab": line_violation_probability_after_fair_ab,
        "trafo_violation_probability_after_fair_ab": trafo_violation_probability_after_fair_ab,
        "bus_violation_probability_after_validation": bus_violation_probability_after_validation,
        "bus_overvoltage_probability_after_validation": bus_overvoltage_probability_after_validation,
        "bus_undervoltage_probability_after_validation": bus_undervoltage_probability_after_validation,
        "line_violation_probability_after_validation": line_violation_probability_after_validation,
        "trafo_violation_probability_after_validation": trafo_violation_probability_after_validation,
        "expected_curtailment_mw_before": expected_curtailment_mw_before,
        "expected_curtailment_mw_after": expected_curtailment_mw_after,
        "expected_curtailment_mw_after_operational": expected_curtailment_mw_after_operational,
        "expected_curtailment_mw_after_fair_ab": expected_curtailment_mw_after_fair_ab,
        "expected_curtailment_mw_after_validation": expected_curtailment_mw_after_validation,
        "deterministic_setpoint_curtailment_mw_post_opf": deterministic_setpoint_curtailment_mw,
        "risk_comparison_mode": "fair_ab_paired_common_random_numbers_with_validation_split",
        "robust_loop_stop_reason": stop_reason,
        "robust_loop_iterations": iteration_history,
        "n_samples": n,
        "n_samples_validation": n_validation,
        "n_converged": n_converged,
        "sgen_sigma": request.sgen_sigma,
        "confidence": confidence_effective,
        "alpha": alpha_effective,
        "vm_upper_per_bus_applied": vm_upper_per_bus,
        "vm_lower_per_bus_applied": vm_lower_per_bus,
        **response_param_meta,
    }
    app_data["last_dispatch_result"] = result.get("activated_resources", []) or []
    app_data["last_dispatch_timestamp"] = current_ts
    return result


@app.post("/api/flexibility/envelope")
@_off_event_loop
async def compute_flexibility_envelope(request: FlexibilityEnvelopeRequest):
    """PQ feasibility map for a single generator at the current operating point.

    Holds all other generators at their current SCADA setpoints, sweeps the
    target generator over a rectangular (P, Q) grid, and runs ``pp.runpp``
    at each point in parallel. Returns a feasibility map and the safe Q range
    at the generator's current active power output.

    The capability curve (PF=0.9 circle) is returned as metadata so the
    renderer can overlay it on the heatmap — it is NOT used to skip points.
    """
    import math
    import numpy as np

    net = _net_for(request)
    ts = _resolve_tick(request.data_source, request.timestamp)
    meas = prepare_measurement_df(_lookup_measurement_df(ts, request.data_source))

    net_base = copy.deepcopy(net)
    net_base, _ = _assign_tick(net_base, meas, ts, _dataset_of(request))
    _scale_loads(net_base, request.load_scaling_factor)

    reference_state_requested = str(request.reference_state or "scada").strip().lower()
    if reference_state_requested not in {"scada", "post_opf", "custom"}:
        raise HTTPException(status_code=400, detail="reference_state must be one of: scada, post_opf, custom")

    reference_state_used = "scada"
    reference_dispatch_applied = False
    reference_dispatch_summary = {"reason": "scada_baseline"}

    if reference_state_requested == "post_opf":
        cached_rows = app_data.get("last_dispatch_result") or []
        cached_ts = app_data.get("last_dispatch_timestamp")
        if not cached_rows:
            reference_dispatch_summary = {"reason": "no_cached_dispatch"}
        elif cached_ts != ts:
            reference_dispatch_summary = {
                "reason": "cached_dispatch_timestamp_mismatch",
                "cached_timestamp": cached_ts,
                "current_timestamp": ts,
            }
        else:
            apply_summary = _apply_dispatch_to_net(net_base, cached_rows)
            reference_dispatch_applied = apply_summary.get("applied", 0) > 0
            reference_dispatch_summary = apply_summary
            if reference_dispatch_applied:
                reference_state_used = "post_opf"

    elif reference_state_requested == "custom":
        custom_rows = _dispatch_rows_from_overrides(request.dispatch_overrides)
        if not custom_rows:
            raise HTTPException(
                status_code=400,
                detail="dispatch_overrides must include at least one valid setpoint when reference_state='custom'.",
            )
        apply_summary = _apply_dispatch_to_net(net_base, custom_rows)
        reference_dispatch_applied = apply_summary.get("applied", 0) > 0
        reference_dispatch_summary = apply_summary
        if reference_dispatch_applied:
            reference_state_used = "custom"

    # Run base power flow to get the current operating point
    try:
        pp.runpp(net_base, algorithm="nr", numba=False)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Base power flow failed: {e}")

    # A voltage-controlled unit (net.gen) is studied as what the envelope
    # sweeps: an injection of active and reactive power, from where it runs
    # now. Only this study's copy changes; the note says so.
    as_injection = _voltage_unit_as_injection(net_base, request.gen_name)

    # Find target sgen by canonical label with robust fallbacks for uploaded nets.
    # Some imported MATPOWER networks do not populate ``substation_name`` and
    # only have ``name`` (e.g., Gen_bus1), while users may ask for "Bus 1".
    if net_base.sgen.empty:
        raise HTTPException(status_code=404, detail="No generators available in sgen table.")

    label_col = "substation_name" if "substation_name" in net_base.sgen.columns else (
        "name" if "name" in net_base.sgen.columns else None
    )
    if label_col is None:
        raise HTTPException(
            status_code=500,
            detail=(
                "Generator name lookup failed: neither 'substation_name' nor 'name' "
                f"exists in sgen columns ({list(net_base.sgen.columns)})."
            ),
        )

    def _norm(v) -> str:
        return str(v).strip().casefold()

    def _parse_bus_alias(v) -> int | None:
        s = str(v).strip().lower().replace("-", "_")
        if s.startswith("gen_"):
            s = s[4:]
        if not s.startswith("bus"):
            return None
        tail = s[3:].strip(" _")
        return int(tail) if tail.isdigit() else None

    requested = _norm(request.gen_name)
    labels = net_base.sgen[label_col].fillna("").astype(str)
    sgen_mask = labels.map(_norm) == requested

    # Alias fallback: allow "Bus 1" or "Gen_bus1" to match by bus index.
    if not sgen_mask.any() and "bus" in net_base.sgen.columns:
        req_bus = _parse_bus_alias(request.gen_name)
        if req_bus is not None:
            sgen_mask = net_base.sgen["bus"].astype(int) == req_bus

    if not sgen_mask.any():
        known_names = sorted({str(v) for v in labels.tolist() if str(v).strip()})
        if "bus" in net_base.sgen.columns:
            known_names.extend(sorted({f"Gen_bus{int(b)}" for b in net_base.sgen["bus"].tolist()}))
        raise HTTPException(
            status_code=404,
            detail=f"Generator '{request.gen_name}' not found. Known names: {known_names}",
        )

    sgen_idx = int(net_base.sgen[sgen_mask].index[0])
    resolved_gen_name = str(net_base.sgen.at[sgen_idx, label_col]).strip() or (
        f"Gen_bus{int(net_base.sgen.at[sgen_idx, 'bus'])}" if "bus" in net_base.sgen.columns else request.gen_name
    )

    # Base operating point from the power flow result
    pg_base = round(float(net_base.res_sgen.at[sgen_idx, "p_mw"]), 5)
    qg_base = round(float(net_base.res_sgen.at[sgen_idx, "q_mvar"]), 5)

    # Build sweep ranges
    pg_max_data = _pg_max_for(request, net_base)
    pg_max = pg_max_data.get(
        resolved_gen_name,
        pg_max_data.get(request.gen_name, max(1.0, pg_base * 1.5)),
    )
    p_min = request.p_min_mw if request.p_min_mw is not None else 0.0
    p_max = request.p_max_mw if request.p_max_mw is not None else pg_max
    tan_phi = math.tan(math.acos(0.9))          # PF=0.9 capability boundary
    # Default Q range: exactly the PF=0.9 capability arc at the base active power.
    # Using the rated capability (p_max) would make the range too wide and cause
    # coarse grid spacing near the actual feasibility boundary — which is always
    # close to the base operating point.  A minimum of 0.3 MVAr ensures a sensible
    # range even for near-zero generators.
    q_cap_rated = p_max * tan_phi
    q_cap_base  = max(pg_base * tan_phi, 0.3)   # exact capability at base P
    q_min = request.q_min_mvar if request.q_min_mvar is not None else -q_cap_base
    q_max = request.q_max_mvar if request.q_max_mvar is not None else  q_cap_base

    resolution = max(5, min(request.resolution, 25))
    # Always include the base operating point in the grid so safe_q_range_at_base_p
    # is computed at the exact (pg_base, qg_base) and not a nearby grid cell.
    # Without this, a coarse grid can place qg_base between two grid Q values,
    # making the safe range appear lower than the actual feasibility boundary.
    p_values = sorted(set(list(np.linspace(p_min, p_max, resolution)) + [pg_base]))
    q_values = sorted(set(list(np.linspace(q_min, q_max, resolution)) + [qg_base]))
    grid_points = [(p, q) for p in p_values for q in q_values]

    # Parallel sweep
    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=min(len(grid_points), 8)) as pool:
        futs = {
            pool.submit(
                _run_envelope_point,
                p, q, net_base, sgen_idx,
                request.vm_upper_pu, request.vm_lower_pu,
                request.max_line_loading_pct, request.max_trafo_loading_pct,
            ): (p, q)
            for p, q in grid_points
        }
        for fut in as_completed(futs):
            results.append(fut.result())

    results.sort(key=lambda r: (round(r["p_mw"], 4), round(r["q_mvar"], 4)))

    # Safe Q range at the closest grid column to pg_base,
    # clipped to the PF=0.9 capability boundary at the base active power.
    q_cap_at_base_p = round(pg_base * tan_phi, 5)
    p_arr = sorted(set(round(r["p_mw"], 4) for r in results))
    closest_p = min(p_arr, key=lambda p: abs(p - pg_base))
    q_slice = [r for r in results if round(r["p_mw"], 4) == closest_p and r["feasible"]]
    safe_q_range = (
        [round(max(min(r["q_mvar"] for r in q_slice), -q_cap_at_base_p), 5),
         round(min(max(r["q_mvar"] for r in q_slice),  q_cap_at_base_p), 5)]
        if q_slice else None
    )

    n_feasible = sum(1 for r in results if r["feasible"])
    n_converged = sum(1 for r in results if r["converged"])

    return {
        "timestamp": ts,
        "gen_name": request.gen_name,
        "resolved_gen_name": resolved_gen_name,
        **({"unit_studied_as_injection": as_injection} if as_injection else {}),
        "reference_state_requested": reference_state_requested,
        "reference_state_used": reference_state_used,
        "reference_dispatch_applied": reference_dispatch_applied,
        "reference_dispatch_summary": reference_dispatch_summary,
        "p_range": [round(p_min, 5), round(p_max, 5)],
        "q_range": [round(q_min, 5), round(q_max, 5)],
        "resolution": resolution,
        "n_feasible": n_feasible,
        "n_converged": n_converged,
        "n_total": len(results),
        "base_point": {"p_mw": pg_base, "q_mvar": qg_base},
        "safe_q_range_at_base_p": safe_q_range,
        "q_cap_at_base_p": q_cap_at_base_p,
        "capability_curve_pf": 0.9,
        "vm_upper_pu": request.vm_upper_pu,
        "vm_lower_pu": request.vm_lower_pu,
        "envelope": results,
    }


def _resolve_hosting_bus(net, bus_input: int | str) -> int:
    """Resolve bus input (index or name) to a net.bus integer index."""
    if isinstance(bus_input, int):
        if bus_input in net.bus.index:
            return int(bus_input)
        raise HTTPException(status_code=404, detail=f"Bus index {bus_input} not found.")

    text = str(bus_input).strip()
    if text.isdigit():
        bus_idx = int(text)
        if bus_idx in net.bus.index:
            return bus_idx

    norm = text.casefold()

    # 1) Exact match on bus name
    if "name" in net.bus.columns:
        exact = [int(i) for i, raw in net.bus["name"].items() if str(raw).strip().casefold() == norm]
        if len(exact) == 1:
            return exact[0]
        if len(exact) > 1:
            raise HTTPException(status_code=400, detail=f"Bus '{text}' is ambiguous; use bus index.")

    # 2) Alias forms like Bus 1, bus_1, Gen_bus1
    alias = text.lower().replace("-", "_").replace(" ", "_")
    if alias.startswith("gen_"):
        alias = alias[4:]
    if alias.startswith("bus"):
        tail = alias[3:].strip("_")
        if tail.isdigit():
            bus_idx = int(tail)
            if bus_idx in net.bus.index:
                return bus_idx

    raise HTTPException(status_code=404, detail=f"Bus '{text}' not found.")


def _hosting_q_target(
    mode: str,
    p_mw: float,
    base_q_mvar: float,
    power_factor: float,
    pf_sign: str,
    q_cap_mvar: float,
) -> float:
    """Compute the candidate Q dispatch for a trial P under selected q mode."""
    mode_key = str(mode).strip().lower()
    if mode_key == "unity":
        return 0.0
    if mode_key == "fixed_pf":
        q_mag = abs(float(p_mw)) * math.tan(math.acos(float(power_factor)))
        # pandapower sgen sign convention: injection positive, absorption negative.
        return -q_mag if pf_sign == "absorbing" else q_mag
    if mode_key in {"reactive_proxy", "voltage_control"}:
        # First deterministic slice: hold baseline Q and saturate at capability.
        return float(max(-q_cap_mvar, min(q_cap_mvar, base_q_mvar)))
    raise HTTPException(status_code=400, detail="q_mode must be one of: all, unity, fixed_pf, reactive_proxy")


def _build_hosting_binding(
    point_result: dict,
    vm_upper: float,
    vm_lower: float,
    thermal_limit: float,
) -> dict:
    """Convert envelope point feasibility diagnostics into a binding-constraint object."""
    binding = point_result.get("binding_constraint")
    if binding == "upper_voltage":
        return {
            "type": "voltage_upper",
            "element": "max_bus_vm_pu",
            "value": float(point_result.get("max_vm_pu") or 0.0),
            "limit": float(vm_upper),
            "units": "p.u.",
        }
    if binding == "lower_voltage":
        return {
            "type": "voltage_lower",
            "element": "min_bus_vm_pu",
            "value": float(point_result.get("min_vm_pu") or 0.0),
            "limit": float(vm_lower),
            "units": "p.u.",
        }
    if binding == "thermal":
        # The branch type that binds, against its own limit (review 2.12);
        # `thermal_limit` remains the fallback for results without it.
        own = point_result.get("thermal") or {}
        out = {
            "type": "thermal",
            "element": "max_branch_loading",
            "value": float(own.get("value", point_result.get("max_loading_pct") or 0.0)),
            "limit": float(own.get("limit", thermal_limit)),
            "units": "%",
        }
        if own.get("branch_type"):
            out["branch_type"] = own["branch_type"]
        return out
    if binding == "no_convergence":
        return {
            "type": "no_convergence",
            "element": "power_flow",
            "value": 1.0,
            "limit": 0.0,
            "units": "flag",
        }
    if binding == "search_cap":
        return {
            "type": "search_cap",
            "element": "search_upper_bound",
            "value": float(point_result.get("p_mw") or 0.0),
            "limit": float(point_result.get("search_limit_mw") or 0.0),
            "units": "MW",
        }
    return {
        "type": "none",
        "element": "none",
        "value": 0.0,
        "limit": 0.0,
        "units": "n/a",
    }


def _solve_hosting_capacity_mode(
    net_base,
    bus_idx: int,
    mode: str,
    power_factor: float,
    pf_sign: str,
    base_q_mvar: float,
    p_max_search_mw: float,
    tolerance_mw: float,
    vm_upper_pu: float,
    vm_lower_pu: float,
    max_line_loading_pct: float,
    max_trafo_loading_pct: float,
) -> dict:
    """Deterministic single-mode hosting capacity via bisection."""
    net_probe = copy.deepcopy(net_base)
    test_idx = int(net_probe.sgen.index.max() + 1) if len(net_probe.sgen.index) else 0
    pp.create_sgen(
        net_probe,
        bus=int(bus_idx),
        p_mw=0.0,
        q_mvar=0.0,
        name=f"HC_TEST_{mode}_{bus_idx}",
        in_service=True,
    )

    if mode == "reactive_proxy":
        q_cap_mvar = max(0.1, float(p_max_search_mw) * math.tan(math.acos(0.9)))
    else:
        q_cap_mvar = max(0.1, float(p_max_search_mw) * math.tan(math.acos(float(power_factor))))

    low = 0.0
    high = float(p_max_search_mw)
    best_p = 0.0
    best_diag = None
    first_infeasible_p = None
    first_infeasible_diag = None
    last_diag = None
    converged = False
    max_iter = 40

    for _ in range(max_iter):
        mid = 0.5 * (low + high)
        q_mid = _hosting_q_target(mode, mid, base_q_mvar, power_factor, pf_sign, q_cap_mvar)
        diag = _run_envelope_point(
            mid,
            q_mid,
            net_probe,
            test_idx,
            vm_upper_pu,
            vm_lower_pu,
            max_line_loading_pct,
            max_trafo_loading_pct,
        )
        last_diag = diag
        if diag.get("feasible"):
            best_p = mid
            best_diag = diag
            low = mid
        else:
            # Track the nearest infeasible point from above the feasible frontier.
            if first_infeasible_p is None or mid < first_infeasible_p:
                first_infeasible_p = mid
                first_infeasible_diag = diag
            high = mid
        if (high - low) <= tolerance_mw:
            converged = True
            break

    had_feasible = best_diag is not None
    if best_diag is None:
        best_diag = last_diag or {
            "binding_constraint": "no_convergence",
            "max_vm_pu": None,
            "min_vm_pu": None,
            "max_loading_pct": None,
        }

    hit_search_cap = first_infeasible_diag is None and best_p >= (float(p_max_search_mw) - float(tolerance_mw))
    if (not had_feasible) and first_infeasible_diag is not None:
        binding_diag = first_infeasible_diag
        binding_source = "first_infeasible"
        termination_reason = "no_feasible_point"
    elif first_infeasible_diag is not None:
        binding_diag = first_infeasible_diag
        binding_source = "first_infeasible"
        termination_reason = "limit_found" if converged else "max_iter"
    elif hit_search_cap:
        binding_diag = {
            "binding_constraint": "search_cap",
            "p_mw": float(best_p),
            "search_limit_mw": float(p_max_search_mw),
        }
        binding_source = "search_cap"
        termination_reason = "search_cap_reached"
    else:
        binding_diag = best_diag
        binding_source = "best_feasible"
        termination_reason = "converged_feasible"

    payload = {
        "q_mode": mode,
        "hosting_capacity_mw": round(float(best_p), 4),
        "binding_constraint": _build_hosting_binding(
            binding_diag,
            vm_upper_pu,
            vm_lower_pu,
            max(float(max_line_loading_pct), float(max_trafo_loading_pct)),
        ),
        "converged": bool(converged),
        "termination_reason": termination_reason,
        "binding_source": binding_source,
        "p_first_infeasible_mw": round(float(first_infeasible_p), 4) if first_infeasible_p is not None else None,
        "feasibility_gap_mw": (
            round(float(first_infeasible_p - best_p), 4)
            if first_infeasible_p is not None else None
        ),
    }
    # A bisection that did not converge still carries a `hosting_capacity_mw`,
    # and nothing in the number itself says it is the last iterate rather than
    # an answer. Withdrawing it leaves `p_first_infeasible_mw` and the
    # termination reason, so the turn still has true figures to report.
    if not converged:
        withdrawal.mark_unusable(
            payload,
            "The hosting-capacity bisection did not converge; the value is the "
            "last iterate, not a determined capacity.",
        )
        payload["hosting_capacity_mw_unconverged"] = payload.pop("hosting_capacity_mw")
        payload["hosting_capacity_mw"] = None
    return payload


def _uses_measured_dataset(grid_profile: dict) -> bool:
    """Return whether the active profile is backed by measured historical data."""
    measurement_source = str(grid_profile.get("measurement_source") or "").strip().casefold()
    return measurement_source in {"measured", "pickle", "historical"}


def _draw_hosting_uncertainty_samples(
    n_samples: int,
    added_gen_sigma: float,
    load_sigma: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw bounded availability and load multipliers for hosting analysis."""
    from scipy.stats.qmc import LatinHypercube
    from scipy.stats import norm as sp_norm

    n = max(10, min(int(n_samples), 1000))
    sampler = LatinHypercube(d=2, seed=42)
    unit_samples = sampler.random(n=n)

    if float(added_gen_sigma) <= 0.0:
        availability = np.ones(n, dtype=float)
    else:
        availability = sp_norm.ppf(unit_samples[:, 0], loc=1.0, scale=float(added_gen_sigma)).clip(0.0, 1.0)

    if float(load_sigma) <= 0.0:
        load_mults = np.ones(n, dtype=float)
    else:
        load_mults = sp_norm.ppf(unit_samples[:, 1], loc=1.0, scale=float(load_sigma)).clip(0.01, 3.0)

    return availability, load_mults


def _run_hosting_probabilistic_sample(
    availability: float,
    load_mult: float,
    net_base,
    sgen_idx: int,
    p_candidate_mw: float,
    q_mode: str,
    power_factor: float,
    pf_sign: str,
    base_q_mvar: float,
    q_cap_mvar: float,
    vm_upper_pu: float,
    vm_lower_pu: float,
    max_line_loading_pct: float,
    max_trafo_loading_pct: float,
) -> dict:
    """Evaluate one uncertain realization for a hosting-capacity candidate."""
    net_copy = copy.deepcopy(net_base)
    if abs(float(load_mult) - 1.0) > 1e-6 and len(net_copy.load.index) > 0:
        _scale_loads(net_copy, float(load_mult))

    realized_p = max(0.0, float(availability) * float(p_candidate_mw))
    q_mvar = _hosting_q_target(q_mode, realized_p, base_q_mvar, power_factor, pf_sign, q_cap_mvar)
    return _run_envelope_point(
        realized_p,
        q_mvar,
        net_copy,
        sgen_idx,
        vm_upper_pu,
        vm_lower_pu,
        max_line_loading_pct,
        max_trafo_loading_pct,
    )


def _aggregate_hosting_probabilistic_binding(
    sample_diags: list[dict],
    risk_threshold: float,
    n_samples: int,
    vm_upper_pu: float,
    vm_lower_pu: float,
    thermal_limit: float,
) -> dict:
    """Summarize the dominant violation mechanism at a probabilistic boundary."""
    non_converged = sum(1 for diag in sample_diags if not diag.get("converged", False))
    if non_converged:
        return {
            "type": "no_convergence",
            "element": "monte_carlo_samples",
            "value": float(non_converged),
            "limit": 0.0,
            "units": "count",
            "sample_probability": round(non_converged / max(1, int(n_samples)), 4),
            "risk_threshold": float(risk_threshold),
        }

    feasible = [diag for diag in sample_diags if diag.get("converged", False)]
    if not feasible:
        return {
            "type": "no_convergence",
            "element": "monte_carlo_samples",
            "value": 1.0,
            "limit": 0.0,
            "units": "flag",
            "sample_probability": 1.0,
            "risk_threshold": float(risk_threshold),
        }

    counts = {
        "upper_voltage": sum(1 for diag in feasible if diag.get("binding_constraint") == "upper_voltage"),
        "lower_voltage": sum(1 for diag in feasible if diag.get("binding_constraint") == "lower_voltage"),
        "thermal": sum(1 for diag in feasible if diag.get("binding_constraint") == "thermal"),
    }
    dominant = max(counts, key=counts.get)
    dominant_count = counts[dominant]
    if dominant_count <= 0:
        return {
            "type": "none",
            "element": "none",
            "value": 0.0,
            "limit": float(risk_threshold),
            "units": "probability",
            "sample_probability": 0.0,
            "risk_threshold": float(risk_threshold),
        }

    if dominant == "upper_voltage":
        worst_diag = max(
            (diag for diag in feasible if diag.get("binding_constraint") == "upper_voltage"),
            key=lambda diag: float(diag.get("max_vm_pu") or -1.0),
        )
    elif dominant == "lower_voltage":
        worst_diag = min(
            (diag for diag in feasible if diag.get("binding_constraint") == "lower_voltage"),
            key=lambda diag: float(diag.get("min_vm_pu") or 99.0),
        )
    else:
        worst_diag = max(
            (diag for diag in feasible if diag.get("binding_constraint") == "thermal"),
            key=lambda diag: float(diag.get("max_loading_pct") or -1.0),
        )

    binding = _build_hosting_binding(worst_diag, vm_upper_pu, vm_lower_pu, thermal_limit)
    binding["sample_probability"] = round(dominant_count / max(1, int(n_samples)), 4)
    binding["risk_threshold"] = float(risk_threshold)
    return binding


def _solve_hosting_capacity_mode_probabilistic(
    net_base,
    bus_idx: int,
    q_mode: str,
    power_factor: float,
    pf_sign: str,
    base_q_mvar: float,
    p_max_search_mw: float,
    tolerance_mw: float,
    vm_upper_pu: float,
    vm_lower_pu: float,
    max_line_loading_pct: float,
    max_trafo_loading_pct: float,
    risk_threshold: float,
    n_samples: int,
    added_gen_sigma: float,
    load_sigma: float,
) -> dict:
    """Probabilistic single-mode hosting capacity via bisection on violation risk."""
    net_probe = copy.deepcopy(net_base)
    test_idx = int(net_probe.sgen.index.max() + 1) if len(net_probe.sgen.index) else 0
    pp.create_sgen(
        net_probe,
        bus=int(bus_idx),
        p_mw=0.0,
        q_mvar=0.0,
        name=f"HC_TEST_{q_mode}_{bus_idx}",
        in_service=True,
    )

    if q_mode == "reactive_proxy":
        q_cap_mvar = max(0.1, float(p_max_search_mw) * math.tan(math.acos(0.9)))
    else:
        q_cap_mvar = max(0.1, float(p_max_search_mw) * math.tan(math.acos(float(power_factor))))

    low = 0.0
    high = float(p_max_search_mw)
    best_p = 0.0
    best_eval = None
    first_infeasible_p = None
    first_infeasible_eval = None
    last_eval = None
    converged = False
    max_iter = 30

    for _ in range(max_iter):
        mid = 0.5 * (low + high)
        availability, load_mults = _draw_hosting_uncertainty_samples(n_samples, added_gen_sigma, load_sigma)

        sample_diags: list[dict | None] = [None] * len(availability)
        with ThreadPoolExecutor(max_workers=min(len(availability), 8)) as pool:
            futures = {
                pool.submit(
                    _run_hosting_probabilistic_sample,
                    float(availability[i]),
                    float(load_mults[i]),
                    net_probe,
                    test_idx,
                    mid,
                    q_mode,
                    power_factor,
                    pf_sign,
                    base_q_mvar,
                    q_cap_mvar,
                    vm_upper_pu,
                    vm_lower_pu,
                    max_line_loading_pct,
                    max_trafo_loading_pct,
                ): i
                for i in range(len(availability))
            }
            for future in as_completed(futures):
                sample_diags[futures[future]] = future.result()

        diags = [diag for diag in sample_diags if diag is not None]
        if not diags:
            eval_res = {
                "p_mw": float(mid),
                "n_samples": int(len(availability)),
                "n_converged": 0,
                "p_any_violation": 1.0,
                "binding_constraint": {
                    "type": "no_convergence",
                    "element": "monte_carlo_samples",
                    "value": 1.0,
                    "limit": 0.0,
                    "units": "flag",
                    "sample_probability": 1.0,
                    "risk_threshold": float(risk_threshold),
                },
                "feasible": False,
            }
        else:
            violation_count = sum(1 for diag in diags if not diag.get("feasible", False))
            p_any_violation = violation_count / len(diags)
            eval_res = {
                "p_mw": float(mid),
                "n_samples": int(len(availability)),
                "n_converged": int(len(diags)),
                "p_any_violation": round(p_any_violation, 4),
                "binding_constraint": _aggregate_hosting_probabilistic_binding(
                    diags,
                    risk_threshold,
                    len(availability),
                    vm_upper_pu,
                    vm_lower_pu,
                    max(float(max_line_loading_pct), float(max_trafo_loading_pct)),
                ),
                "feasible": p_any_violation <= float(risk_threshold),
            }

        last_eval = eval_res
        if eval_res["feasible"]:
            best_p = mid
            best_eval = eval_res
            low = mid
        else:
            if first_infeasible_p is None or mid < first_infeasible_p:
                first_infeasible_p = mid
                first_infeasible_eval = eval_res
            high = mid

        if (high - low) <= tolerance_mw:
            converged = True
            break

    had_feasible = best_eval is not None
    if best_eval is None:
        best_eval = last_eval or {
            "p_mw": 0.0,
            "n_samples": max(10, min(int(n_samples), 1000)),
            "n_converged": 0,
            "p_any_violation": 1.0,
            "binding_constraint": {
                "type": "no_convergence",
                "element": "monte_carlo_samples",
                "value": 1.0,
                "limit": 0.0,
                "units": "flag",
                "sample_probability": 1.0,
                "risk_threshold": float(risk_threshold),
            },
            "feasible": False,
        }

    hit_search_cap = first_infeasible_eval is None and best_p >= (float(p_max_search_mw) - float(tolerance_mw))
    if (not had_feasible) and first_infeasible_eval is not None:
        binding = first_infeasible_eval["binding_constraint"]
        binding_source = "first_infeasible"
        termination_reason = "no_feasible_point"
    elif first_infeasible_eval is not None:
        binding = first_infeasible_eval["binding_constraint"]
        binding_source = "first_infeasible"
        termination_reason = "risk_limit_found" if converged else "max_iter"
    elif hit_search_cap:
        binding = {
            "type": "search_cap",
            "element": "search_upper_bound",
            "value": float(best_p),
            "limit": float(p_max_search_mw),
            "units": "MW",
            "sample_probability": float(best_eval.get("p_any_violation") or 0.0),
            "risk_threshold": float(risk_threshold),
        }
        binding_source = "search_cap"
        termination_reason = "search_cap_reached"
    else:
        binding = best_eval["binding_constraint"]
        binding_source = "best_feasible"
        termination_reason = "converged_feasible"

    return {
        "q_mode": q_mode,
        "hosting_capacity_mw": round(float(best_p), 4),
        "binding_constraint": binding,
        "converged": bool(converged),
        "termination_reason": termination_reason,
        "binding_source": binding_source,
        "risk_threshold": float(risk_threshold),
        "n_samples": int(best_eval.get("n_samples") or max(10, min(int(n_samples), 1000))),
        "n_converged": int(best_eval.get("n_converged") or 0),
        "p_any_violation_at_best": round(float(best_eval.get("p_any_violation") or 0.0), 4),
        "p_first_infeasible_mw": round(float(first_infeasible_p), 4) if first_infeasible_p is not None else None,
        "feasibility_gap_mw": (
            round(float(first_infeasible_p - best_p), 4)
            if first_infeasible_p is not None else None
        ),
    }


@app.post("/api/flexibility/hosting_capacity")
@_off_event_loop
async def compute_hosting_capacity(request: HostingCapacityRequest):
    """Hosting-capacity scan for one bus.

    Supports deterministic mode and a first probabilistic slice with bounded
    uncertainty on the added injection.
    """
    if str(request.q_mode).strip().lower() in {"all", ""}:
        modes = ["unity", "fixed_pf", "reactive_proxy"]
    else:
        requested_mode = str(request.q_mode).strip().lower()
        modes = ["reactive_proxy" if requested_mode == "voltage_control" else requested_mode]

    invalid_modes = [m for m in modes if m not in {"unity", "fixed_pf", "reactive_proxy"}]
    if invalid_modes:
        raise HTTPException(status_code=400, detail="q_mode must be one of: all, unity, fixed_pf, reactive_proxy")
    requested_run_mode = str(request.mode).strip().lower()
    if requested_run_mode not in {"deterministic", "probabilistic"}:
        raise HTTPException(status_code=400, detail="mode must be 'deterministic' or 'probabilistic'.")
    if not 0.8 <= float(request.power_factor) <= 1.0:
        raise HTTPException(status_code=400, detail="power_factor must be within [0.8, 1.0].")
    if str(request.pf_sign).strip().lower() not in {"absorbing", "injecting"}:
        raise HTTPException(status_code=400, detail="pf_sign must be 'absorbing' or 'injecting'.")
    if float(request.tolerance_mw) <= 0.0:
        raise HTTPException(status_code=400, detail="tolerance_mw must be > 0.")
    if not 0.0 <= float(request.risk_threshold) <= 1.0:
        raise HTTPException(status_code=400, detail="risk_threshold must be within [0, 1].")
    if int(request.n_samples) <= 0:
        raise HTTPException(status_code=400, detail="n_samples must be > 0.")
    if float(request.added_gen_sigma) < 0.0:
        raise HTTPException(status_code=400, detail="added_gen_sigma must be >= 0.")
    if float(request.load_sigma) < 0.0:
        raise HTTPException(status_code=400, detail="load_sigma must be >= 0.")
    uncertainty_scope = str(request.uncertainty_scope).strip().lower()
    if uncertainty_scope not in {"added_generation_only", "added_generation_plus_load"}:
        raise HTTPException(
            status_code=400,
            detail="uncertainty_scope must be 'added_generation_only' or 'added_generation_plus_load'.",
        )

    timestamps = app_data.get("timestamps", []) or []
    measurements = app_data.get("measurements", {}) or {}
    grid_profile = app_data.get("grid_profile", {}) or {}
    system_name = str(grid_profile.get("name") or grid_profile.get("grid_type") or "active system")
    if not timestamps or not measurements:
        return {
            "error": (
                f"compute_hosting_capacity requires an attached time series or initialized operating point; "
                f"the active system '{system_name}' does not expose measurements in this run."
            ),
            "system_name": system_name,
            "bus": request.bus,
        }

    target_ts = _resolve_tick(request.data_source, request.timestamp)
    measurement_prod_cons = prepare_measurement_df(_lookup_measurement_df(target_ts, request.data_source))
    net_base = copy.deepcopy(_net_for(request))
    net_base, _ = _assign_tick(net_base, measurement_prod_cons, target_ts, _dataset_of(request))

    bus_input_text = str(request.bus).strip().lower()
    scan_all_buses = bool(requested_run_mode == "deterministic" and bus_input_text == "all")
    if requested_run_mode == "probabilistic" and bus_input_text == "all":
        raise HTTPException(
            status_code=400,
            detail="bus='all' is currently supported only for deterministic mode.",
        )

    if request.p_max_search_mw is not None:
        p_max_search_mw = float(request.p_max_search_mw)
    else:
        p_existing = float(net_base.sgen["p_mw"].abs().max()) if not net_base.sgen.empty else 1.0
        p_max_search_mw = max(1.0, 2.0 * p_existing)

    if p_max_search_mw <= 0.0:
        raise HTTPException(status_code=400, detail="p_max_search_mw must be > 0.")

    def _base_q_for_bus(local_bus_idx: int) -> float:
        """Reactive baseline of the units already at this bus.

        `res_sgen` holds only the units the power flow actually solved, so an
        out-of-service sgen is present in `sgen` and absent from `res_sgen`.
        Indexing the results by the setpoint table's index then raises
        KeyError and the endpoint 500s — which it did, on a bus with such a
        unit. The solved rows are used where they exist and the setpoint
        elsewhere, rather than choosing one table for all of them.
        """
        if "bus" not in net_base.sgen.columns or net_base.sgen.empty:
            return 0.0
        sgen_rows = net_base.sgen[net_base.sgen["bus"].astype(int) == int(local_bus_idx)]
        if sgen_rows.empty:
            return 0.0

        res = getattr(net_base, "res_sgen", None)
        solved = sgen_rows.index[:0]
        total = 0.0
        if res is not None and "q_mvar" in getattr(res, "columns", []):
            solved = sgen_rows.index.intersection(res.index)
            if len(solved):
                total += float(res.loc[solved, "q_mvar"].sum())

        unsolved = sgen_rows.index.difference(solved)
        if len(unsolved) and "q_mvar" in sgen_rows.columns:
            total += float(sgen_rows.loc[unsolved, "q_mvar"].sum())

        return -0.5 * total

    bus_targets = (
        [{"bus_index": int(idx), "bus": _bus_display_name(net_base, int(idx))} for idx in net_base.bus.index]
        if scan_all_buses
        else [{"bus_index": int(_resolve_hosting_bus(net_base, request.bus)), "bus": _bus_display_name(net_base, int(_resolve_hosting_bus(net_base, request.bus)))}]
    )

    if len(bus_targets) == 1 and not scan_all_buses:
        # avoid duplicate resolver calls in the single-bus branch
        bus_targets[0]["bus_index"] = int(_resolve_hosting_bus(net_base, request.bus))
        bus_targets[0]["bus"] = _bus_display_name(net_base, int(bus_targets[0]["bus_index"]))

    bus_results = []
    synthetic_uncertainty = bool(requested_run_mode == "probabilistic" and not _uses_measured_dataset(grid_profile))
    effective_load_sigma = float(request.load_sigma) if uncertainty_scope == "added_generation_plus_load" else 0.0
    for bus_target in bus_targets:
        bus_idx = int(bus_target["bus_index"])
        base_q_mvar = _base_q_for_bus(bus_idx)
        mode_results = []
        for mode in modes:
            if requested_run_mode == "probabilistic":
                mode_res = _solve_hosting_capacity_mode_probabilistic(
                    net_base=net_base,
                    bus_idx=bus_idx,
                    q_mode=mode,
                    power_factor=float(request.power_factor),
                    pf_sign=str(request.pf_sign).strip().lower(),
                    base_q_mvar=float(base_q_mvar),
                    p_max_search_mw=p_max_search_mw,
                    tolerance_mw=float(request.tolerance_mw),
                    vm_upper_pu=float(request.vm_upper_pu),
                    vm_lower_pu=float(request.vm_lower_pu),
                    max_line_loading_pct=float(request.max_line_loading_pct),
                    max_trafo_loading_pct=float(request.max_trafo_loading_pct),
                    risk_threshold=float(request.risk_threshold),
                    n_samples=int(request.n_samples),
                    added_gen_sigma=float(request.added_gen_sigma),
                    load_sigma=effective_load_sigma,
                )
            else:
                mode_res = _solve_hosting_capacity_mode(
                    net_base=net_base,
                    bus_idx=bus_idx,
                    mode=mode,
                    power_factor=float(request.power_factor),
                    pf_sign=str(request.pf_sign).strip().lower(),
                    base_q_mvar=float(base_q_mvar),
                    p_max_search_mw=p_max_search_mw,
                    tolerance_mw=float(request.tolerance_mw),
                    vm_upper_pu=float(request.vm_upper_pu),
                    vm_lower_pu=float(request.vm_lower_pu),
                    max_line_loading_pct=float(request.max_line_loading_pct),
                    max_trafo_loading_pct=float(request.max_trafo_loading_pct),
                )
            mode_results.append(mode_res)

        bus_results.append(
            {
                "bus": str(bus_target["bus"]),
                "bus_index": int(bus_idx),
                "results": mode_results,
            }
        )

    results = bus_results[0]["results"] if len(bus_results) == 1 else []

    notes = []
    if requested_run_mode == "probabilistic":
        notes.append("Probabilistic hosting uses bounded availability uncertainty on the added generator, so realized injection stays between zero and the candidate hosting value.")
        notes.append(f"Uncertainty scope: {uncertainty_scope}.")
        if uncertainty_scope == "added_generation_plus_load" and effective_load_sigma > 0.0:
            notes.append("Load uncertainty is applied multiplicatively on top of the current operating point during each sample.")
        if synthetic_uncertainty:
            notes.append("Synthetic uncertainty is active because the current system does not have an attached measured historical dataset.")
    else:
        notes.append("First hosting-capacity slice: deterministic only.")
        if scan_all_buses:
            notes.append("Deterministic all-bus scan evaluated every network bus independently at the same operating point.")
    notes.append("reactive_proxy uses a damped current-Q baseline clipped to capability as an initial approximation.")

    bus_scan_summary = None
    if scan_all_buses:
        ranking_rows = []
        ranking_rows_by_mode = {
            "unity": [],
            "fixed_pf": [],
            "reactive_proxy": [],
        }
        for entry in bus_results:
            for mode_row in entry["results"]:
                row = {
                    "bus": entry["bus"],
                    "bus_index": int(entry["bus_index"]),
                    "q_mode": mode_row.get("q_mode"),
                    "hosting_capacity_mw": float(mode_row.get("hosting_capacity_mw") or 0.0),
                    "binding_type": str((mode_row.get("binding_constraint") or {}).get("type", "none")),
                }
                ranking_rows.append(row)
                mode_key = str(mode_row.get("q_mode") or "").strip().lower()
                if mode_key in ranking_rows_by_mode:
                    ranking_rows_by_mode[mode_key].append(row)
        ranking_rows.sort(key=lambda row: row["hosting_capacity_mw"], reverse=True)
        top_n = ranking_rows[: min(15, len(ranking_rows))]
        top_rows_by_mode = {}
        for mode_key, rows in ranking_rows_by_mode.items():
            rows_sorted = sorted(rows, key=lambda row: row["hosting_capacity_mw"], reverse=True)
            top_rows_by_mode[mode_key] = rows_sorted[: min(5, len(rows_sorted))]
        bus_scan_summary = {
            "n_buses_evaluated": len(bus_results),
            "n_rows": len(ranking_rows),
            "top_rows": top_n,
            "top_rows_by_mode": top_rows_by_mode,
        }

    return {
        "bus": "all" if scan_all_buses else f"{bus_results[0]['bus']}",
        "bus_index": None if scan_all_buses else int(bus_results[0]["bus_index"]),
        "mode": requested_run_mode,
        "risk_threshold": float(request.risk_threshold) if requested_run_mode == "probabilistic" else None,
        "synthetic_uncertainty": synthetic_uncertainty,
        "uncertainty_scope": uncertainty_scope if requested_run_mode == "probabilistic" else None,
        "scan_scope": "all_buses" if scan_all_buses else "single_bus",
        "timestamp": target_ts,
        "q_mode_requested": request.q_mode,
        "power_factor": float(request.power_factor),
        "pf_sign": str(request.pf_sign).strip().lower(),
        "p_max_search_mw": float(p_max_search_mw),
        "tolerance_mw": float(request.tolerance_mw),
        "n_samples": int(request.n_samples) if requested_run_mode == "probabilistic" else None,
        "uncertainty_model": (
            {
                "added_generation": {
                    "distribution": "bounded_normal_availability",
                    "sigma": float(request.added_gen_sigma),
                    "bounds": [0.0, 1.0],
                },
                "load_multiplier": {
                    "distribution": "normal",
                    "sigma": float(effective_load_sigma),
                    "bounds": [0.01, 3.0],
                    "active": bool(uncertainty_scope == "added_generation_plus_load" and effective_load_sigma > 0.0),
                },
            }
            if requested_run_mode == "probabilistic" else None
        ),
        "data_source": _dataset_label(request),
        "thresholds_used": {
            "vm_upper_pu": float(request.vm_upper_pu),
            "vm_lower_pu": float(request.vm_lower_pu),
            "max_line_loading_pct": float(request.max_line_loading_pct),
            "max_trafo_loading_pct": float(request.max_trafo_loading_pct),
        },
        "bus_results": bus_results if scan_all_buses else None,
        "bus_scan_summary": bus_scan_summary,
        "results": results,
        "notes": notes,
    }


def _flex_prices(net, model) -> tuple[fo.Prices, dict]:
    """The cost table's prices for the units and external grid in `model`, and
    where each came from. Units a proposal added get the default for their kind."""
    loaded = app_data.get("net_loaded")
    rows = {r["unit"]: r for r in uc.table(loaded, app_data.get("unit_costs") or {})} if loaded is not None else {}
    added = {r["unit"]: r for r in uc.defaults(net) if r.get("proposed")}
    key_of = {gm.unit_name(net, t, i): gm.unit_key(net, t, i) for t in ("gen", "sgen") for i in net[t].index}
    up, down, q, sources = {}, {}, {}, {}
    for g in model.G:
        row = rows.get(key_of.get(g, g)) or added.get(g)
        if row is None:
            continue
        up[g], down[g], q[g] = row.get("up_eur_per_mwh"), row.get("down_eur_per_mwh"), row.get("q_eur_per_mvarh")
        sources[g] = sorted({(row.get("source") or {}).get(k, "synthetic")
                             for k in ("up_eur_per_mwh", "down_eur_per_mwh", "q_eur_per_mvarh")})
    imp, band, penalty = {}, {}, {}
    for position, e in enumerate(model.Ext):
        idx = net.ext_grid.index[position] if position < len(net.ext_grid) else None
        row = rows.get(uc.ext_grid_name(net, idx)) if idx is not None else None
        if row is None:
            continue
        imp[e], band[e], penalty[e] = (row.get("cost_eur_per_mwh"), row.get("q_band_tan_phi"),
                                       row.get("q_penalty_eur_per_mvarh"))
        sources[str(e)] = sorted({(row.get("source") or {}).get(k, "synthetic")
                                  for k in ("cost_eur_per_mwh", "q_band_tan_phi", "q_penalty_eur_per_mvarh")})
    _avail, dr, relief, _src = _load_prices(net)
    return fo.Prices(up, down, q, imp, band, penalty, dr=dr, relief=relief), sources


def _load_prices(net) -> tuple[dict, dict, dict, list]:
    """Per load of `net` (by index): the MW of demand response available now
    (its share of the load as it stands — scaled, at this moment), and the
    €/MWh of demand response and of load relief, from the cost table; and
    where those figures come from. A load a proposal added has the defaults."""
    loaded = app_data.get("net_loaded")
    rows = ({r["unit"]: r for r in uc.load_table(loaded, app_data.get("unit_costs") or {})}
            if loaded is not None else {})
    added = {r["unit"]: r for r in uc.load_defaults(net)}
    avail, dr, relief, sources = {}, {}, {}, set()
    for idx in net.load.index:
        name = uc.load_name(net, idx)
        row = rows.get(name) or added.get(name)
        if row is None:
            continue
        scale = net.load.at[idx, "scaling"] if "scaling" in net.load.columns else 1.0
        scale = 1.0 if scale is None or scale != scale else float(scale)
        p = max(float(net.load.at[idx, "p_mw"]) * scale, 0.0)
        avail[int(idx)] = p * float(row.get("dr_share_pct") or 0.0) / 100.0
        dr[int(idx)], relief[int(idx)] = row.get("dr_eur_per_mwh"), row.get("relief_eur_per_mwh")
        sources |= set((row.get("source") or {}).values())
    return avail, dr, relief, sorted(sources)


def _load_flex(net) -> dict:
    """What the OPF may reduce at each load (`flex_engine` load_flex): its
    demand response, MW. Load relief covers the rest. Used only when a
    strategy frees them; the default dispatch never touches a load."""
    return _load_prices(net)[0]


def _load_label(net, idx) -> dict:
    """A load as results name it: its name now and its bus."""
    idx = int(idx)
    if idx not in net.load.index:
        return {"load": f"Load {idx}", "bus": None}
    return {"load": naming.display(net, "load", idx), "bus": _bus_display_name(net, int(net.load.at[idx, "bus"]))}


def _load_actions(net, option) -> list[dict]:
    """The load reductions an option asks for, by load: demand response and
    load relief, MW."""
    out = []
    for idx in sorted(set(option.dr or {}) | set(option.relief or {})):
        dr_mw, relief_mw = (option.dr or {}).get(idx, 0.0), (option.relief or {}).get(idx, 0.0)
        out.append({**_load_label(net, idx), "demand_response_mw": round(dr_mw, 3),
                    "load_relief_mw": round(relief_mw, 3)})
    return out


def _finite_exchange(net) -> bool:
    res = getattr(net, "res_ext_grid", None)
    return (res is not None and len(res) == len(net.ext_grid) and len(res) > 0
            and bool(np.isfinite(res["p_mw"].to_numpy(dtype=float)).all()))


# An option the power flow confirms: within the band (this close), at most 100 %
# of every rating (this close: an option at its rating is within it), every
# unit within its reactive limits (this close).
_PF_VM_TOL, _PF_LOADING_MAX, _PF_LOADING_TOL, _PF_Q_TOL = 0.002, 100.0, 0.1, 0.5


def _apply_option_dispatch(net, option, units: bool = True, loads: bool = True, held: bool = False) -> None:
    """An option's dispatch onto `net` (in place): every unit's active power,
    the set-point of a unit that holds a voltage, the reactive output of one
    that does not; and its load reductions. The external grid takes the
    balance, as in the OPF. `held` (at a later moment than it was computed
    for): only what the option told a unit to change stays — every other unit
    follows its own schedule there, as it would in operation."""
    where = {gm.unit_name(net, table, idx): (table, idx) for table in ("gen", "sgen") for idx in net[table].index}
    told = option.instructions if held else None
    for unit, p in (option.p if units else {}).items():
        if unit not in where or (told is not None and unit not in told):
            continue
        what = told.get(unit, "") if told is not None else "P+V+Q"
        table, idx = where[unit]
        if "P" in what:
            net[table].at[idx, "p_mw"] = p
        bus = (option.vc_bus or {}).get(unit)
        if table == "gen" and bus is not None and option.v.get(bus) is not None:
            if "V" in what:
                net.gen.at[idx, "vm_pu"] = option.v[bus]      # the set-point the option gives it
        elif table == "sgen" and "Q" in what:
            net.sgen.at[idx, "q_mvar"] = option.q.get(unit, net.sgen.at[idx, "q_mvar"])
    # Load reductions (demand response, load relief), each load keeping its power factor.
    for idx in (set(option.dr or {}) | set(option.relief or {})) if loads else ():
        if idx not in net.load.index:
            continue
        scale = net.load.at[idx, "scaling"] if "scaling" in net.load.columns else 1.0
        scale = 1.0 if scale is None or scale != scale else float(scale)
        served = float(net.load.at[idx, "p_mw"]) * scale
        cut = (option.dr or {}).get(idx, 0.0) + (option.relief or {}).get(idx, 0.0)
        share = max(1.0 - cut / served, 0.0) if served > 1e-9 else 1.0
        net.load.at[idx, "p_mw"] = float(net.load.at[idx, "p_mw"]) * share
        net.load.at[idx, "q_mvar"] = float(net.load.at[idx, "q_mvar"]) * share


def _conditions_key() -> str:
    """What an option's dispatch depends on besides its moment: how the units
    are modelled and named, and the load power factor."""
    return json.dumps({"gm": app_data.get("generator_model") or {}, "names": app_data.get("element_names") or {},
                       "pf": app_data.get("load_power_factor")}, sort_keys=True, default=str)


_MAX_OPTION_SETS = 20


def _remember_options(request, ts: str, kept: list, outage: dict | None, held: dict | None) -> str:
    """Keep a set of options in memory so a later study can apply one of them
    (the N-1 screen): its moment, network, load level, outage, conditions and
    each option's dispatch. Returns the set's id ("S3")."""
    sets = app_data.setdefault("option_sets", {})
    app_data["option_set_counter"] = app_data.get("option_set_counter", 0) + 1
    set_id = f"S{app_data['option_set_counter']}"
    sets[set_id] = {"id": set_id, "timestamp": ts, "network": request.network or "base",
                    "data_source": request.data_source, "load_scaling_factor": request.load_scaling_factor,
                    "outage": outage, "held_units": held or {}, "conditions": _conditions_key(),
                    "options": {o.id: o for o, _ in kept}}
    for old in list(sets)[:-_MAX_OPTION_SETS]:
        sets.pop(old)
    _save_option_sets()
    return set_id


def _resolve_option(ref: str) -> tuple[dict, object]:
    """A stored option by its reference: "S3:cheapest", or "cheapest" (the
    newest set that has one). Refuses, with the reason, an unknown one and one
    computed under other unit models, names or load power factor."""
    sets = app_data.get("option_sets") or {}
    text = str(ref or "").strip().replace("·", ":")
    set_id, _, oid = text.rpartition(":")
    oid = oid.strip()
    candidates = [sets[set_id.strip()]] if set_id.strip() in sets else (
        [] if set_id.strip() else [s for s in reversed(sets.values()) if oid in s["options"]])
    found = next((s for s in candidates if oid in s["options"]), None)
    if found is None:
        known = [f"{s['id']}:{o}" for s in list(sets.values())[-3:] for o in s["options"]]
        raise HTTPException(status_code=400, detail=(
            f"option: there is no option {ref!r}" + (f". Options kept now: {', '.join(known)}." if known else
            ". No options are kept (none computed since the backend started, or another network was loaded): "
            "compute the options first.")))
    if found["conditions"] != _conditions_key():
        raise HTTPException(status_code=400, detail=(
            f"option: {found['id']}:{oid} was computed before the unit models, names or load power factor "
            "changed; compute the options again."))
    return found, found["options"][oid]


def _option_context(request, later: bool = True, ref: str | None = None) -> tuple[dict, object, object, bool]:
    """Resolve `request.option` for a study that applies it: (its set, the
    option, the request at the option's network, data and load level, held).
    The moment is the option's own unless the request names a later one — then
    (`later`) its set-points are held since it was computed, `held` True.
    Refused: an unknown option, another network or load level, an earlier
    moment (or any other moment, when not `later`)."""
    stored, option = _resolve_option(ref or request.option)
    own = stored["timestamp"]
    asked = _resolve_tick(stored["data_source"], request.timestamp) if request.timestamp else None
    clashes = [what for what, asked_, own_ in (
        ("network", request.network, stored["network"] if request.network else None),
        ("load scaling", request.load_scaling_factor if request.load_scaling_factor != 1.0 else None,
         stored["load_scaling_factor"] if request.load_scaling_factor != 1.0 else None))
        if asked_ is not None and asked_ != own_]
    if asked is not None and asked != own and (not later or own == nom.NOMINAL or str(asked) < str(own)):
        clashes.append("moment" + (" (before it was computed)" if later and str(asked) < str(own) else ""))
    if clashes:
        raise HTTPException(status_code=400, detail=(
            f"option: {stored['id']}:{option.id} was computed at {own} on {stored['network']} with loads "
            f"×{stored['load_scaling_factor']:g}; this study asked for another {' and '.join(clashes)}. "
            + ("Use it at its own moment or a later one (its set-points held), or compute options for this one."
               if later else "Screen it at its own moment, or compute options for this one.")))
    at = asked or own
    request = request.model_copy(update={
        "timestamp": at, "data_source": stored["data_source"],
        "network": None if stored["network"] == "base" else stored["network"],
        "load_scaling_factor": stored["load_scaling_factor"]})
    return stored, option, request, at != own


def _apply_stored_option(net, stored: dict, option, units: bool = True, loads: bool = True,
                         held: bool = False) -> None:
    """An option onto `net` (in place): the outage it was computed after, if
    any (buses it leaves without supply out of service too), then its dispatch
    — `held` at a later moment: its instructions only (`_apply_option_dispatch`)."""
    outage = stored.get("outage")
    if outage:
        table = net.line if outage["element_type"] == "line" else net.trafo
        table.at[outage["element_index"], "in_service"] = False
        islanded = sorted(int(b) for b in pp.topology.unsupplied_buses(net))
        if islanded:
            net.bus.loc[islanded, "in_service"] = False
    _apply_option_dispatch(net, option, units=units, loads=loads, held=held)


def _option_echo(stored: dict, option, at: str, held: bool, **extra) -> dict:
    """What a study ran with, said in its result: the option, when it was
    computed, when it is applied, and whether its set-points are held."""
    return {"ref": f"{stored['id']}:{option.id}", "label": option.label, "timestamp": stored["timestamp"],
            "applied_at": at, "held": held, "network": stored["network"],
            # Held at a later moment: what it told units to change stays; the rest follow their schedule.
            **({"held_instructions": dict(sorted(option.instructions.items()))} if held else {}),
            "load_scaling_factor": stored["load_scaling_factor"], "held_units": stored.get("held_units") or {},
            "after_outage": (stored.get("outage") or {}).get("element"),
            **({"load_reduction_mw": round(option.load_reduction, 3)} if option.load_reduction else {}), **extra}


def _power_flow_check(net, option, vm_lower: float, vm_upper: float, caps: dict | None = None,
                      max_loading_pct: float = _PF_LOADING_MAX) -> dict:
    """The option's dispatch applied to the network and solved by an ordinary
    AC power flow — its own voltages, loadings and reactive outputs, not the
    OPF's. Independent of the OPF model (its taps, its current limits), so an
    option it does not confirm is reported as such."""
    n = copy.deepcopy(net)
    _apply_option_dispatch(n, option)
    try:
        pp.runpp(n, algorithm="nr", calculate_voltage_angles=True, numba=False)
    except Exception:  # noqa: BLE001 — not converging is the finding
        return {"converged": False, "confirmed": False, "violations": ["the power flow did not converge"]}
    live = n.bus.index[n.bus["in_service"]] if "in_service" in n.bus.columns else n.bus.index
    vm = n.res_bus.vm_pu.reindex(live).dropna()
    violations = []
    for b, v in vm.items():
        if v < vm_lower - _PF_VM_TOL or v > vm_upper + _PF_VM_TOL:
            violations.append(f"{_bus_display_name(n, b)} at {v:.3f} p.u.")
    worst = (None, None)
    for table, label in (("line", _line_display_name), ("trafo", _trafo_display_name)):
        res = n.get(f"res_{table}")
        if res is None or not len(res):
            continue
        for idx, pct in res["loading_percent"].dropna().items():
            name = label(n, idx)
            if worst[0] is None or pct > worst[0]:
                worst = (float(pct), name)
            cap = (caps or {}).get(name)
            if pct > max_loading_pct + _PF_LOADING_TOL:
                violations.append(f"{name} at {pct:.1f} %")
            elif cap is not None and pct > cap + _PF_LOADING_TOL:
                violations.append(f"{name} at {pct:.1f} %, above its {cap:g} % cap")
    for idx in n.gen.index:
        q = float(n.res_gen.at[idx, "q_mvar"])
        lo, hi = n.gen.at[idx, "min_q_mvar"], n.gen.at[idx, "max_q_mvar"]
        if pd.notna(hi) and q > float(hi) + _PF_Q_TOL or pd.notna(lo) and q < float(lo) - _PF_Q_TOL:
            violations.append(f"{gm.unit_name(n, 'gen', idx)} needs {q:.1f} Mvar, beyond its "
                              f"{float(lo):.1f}…{float(hi):.1f} Mvar")
    return {"converged": True, "confirmed": not violations,
            "min_vm_pu": round(float(vm.min()), 4), "max_vm_pu": round(float(vm.max()), 4),
            "max_loading_pct": round(worst[0], 2) if worst[0] is not None else None,
            "max_loading_element": worst[1],
            "import_mw": round(float(n.res_ext_grid["p_mw"].sum()), 4),
            "violations": violations[:6]}


_SWITCH_SHOWN = 3              # switching candidates reported, best first
_SWITCH_GAIN_PCT = 10.0        # an opening must relieve at least this share of the excess, or clear a violation


def _pf_violations(n, lo: float, hi: float, limit_pct: float) -> list[tuple[str, float]]:
    """What a solved power flow breaks, worst first: (in words, excess) —
    loading above the limit in % of rating, voltage outside the band in
    hundredths of a p.u."""
    out = []
    live = n.bus.index[n.bus["in_service"]] if "in_service" in n.bus.columns else n.bus.index
    for b, v in n.res_bus.vm_pu.reindex(live).dropna().items():
        if v < lo - _PF_VM_TOL or v > hi + _PF_VM_TOL:
            out.append((f"{_bus_display_name(n, b)} at {v:.3f} p.u.", 100.0 * max(lo - v, v - hi)))
    for table, label in (("line", _line_display_name), ("trafo", _trafo_display_name)):
        res = n.get(f"res_{table}")
        if res is None or not len(res):
            continue
        for idx, pct in res["loading_percent"].dropna().items():
            if pct > limit_pct + _PF_LOADING_TOL:
                out.append((f"{label(n, idx)} at {pct:.1f} %", float(pct) - limit_pct))
    return sorted(out, key=lambda x: -x[1])


def _switching_remedies(net, option, lo: float, hi: float, limit_pct: float) -> list[dict]:
    """Single lines or transformers whose opening clears or relieves what
    `net` (an operating point; an outage already applied, maybe; `option`'s
    dispatch applied when given) breaks — each opening checked by an AC power
    flow, any that leaves a bus without supply skipped. Best first, at most
    _SWITCH_SHOWN. A screen, not an optimisation: protection, supply paths
    and switching sequence are the operator's to check."""
    n = copy.deepcopy(net)
    if option is not None:
        _apply_option_dispatch(n, option)
    try:
        pp.runpp(n, algorithm="nr", calculate_voltage_angles=True, numba=False)
    except Exception:  # noqa: BLE001 — nothing to compare against
        return []
    before = _pf_violations(n, lo, hi, limit_pct)
    if not before:
        return []
    severity = sum(e for _, e in before)
    found = []
    for table, label in (("line", _line_display_name), ("trafo", _trafo_display_name)):
        frame = n.get(table)
        if frame is None or not len(frame):
            continue
        for idx in frame.index[frame["in_service"].astype(bool)]:
            frame.at[idx, "in_service"] = False
            try:
                if len(pp.topology.unsupplied_buses(n)):
                    continue
                pp.runpp(n, algorithm="nr", calculate_voltage_angles=True, numba=False)
                after = _pf_violations(n, lo, hi, limit_pct)
                left = sum(e for _, e in after)
                if left < severity and (100.0 * (severity - left) / severity >= _SWITCH_GAIN_PCT
                                        or len(after) < len(before)):
                    found.append({"open": label(n, idx), "clears_all": not after,
                                  "relieves_pct": round(100.0 * (severity - left) / severity),
                                  "remaining": [text for text, _ in after[:3]], "checked_by_power_flow": True,
                                  "_left": left})
            except Exception:  # noqa: BLE001 — an opening the power flow cannot solve is not offered
                pass
            finally:
                frame.at[idx, "in_service"] = True
    found.sort(key=lambda f: (not f["clears_all"], f["_left"]))
    return [{k: v for k, v in f.items() if k != "_left"} for f in found[:_SWITCH_SHOWN]]


_N1_MAX_OUTAGES = 8            # outages a study takes on at once (the most severe first)
_PLAN_MW, _PLAN_VM = 0.1, 0.002   # the smallest corrective instruction worth issuing
_N1_ROUNDS = 2                 # solve, screen, add what newly fails, solve again


def _natural(name) -> list:
    """Gen_bus4 before Gen_bus10."""
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", str(name))]


def _outage_net(net, table: str, idx: int):
    """`net` without one branch, and the buses that leaves without supply."""
    n = copy.deepcopy(net)
    n[table].at[idx, "in_service"] = False
    islanded = sorted(int(b) for b in pp.topology.unsupplied_buses(n))
    return n, islanded


def _critical_outages(sweep: dict, limit_pct: float) -> list[tuple]:
    """The outages a screen found violations after, most severe first:
    (name, table, index, severity) — severity the summed excess, loading in %
    and voltage in hundredths of a p.u."""
    by: dict[tuple, float] = {}
    for v in sweep.get("violations") or []:
        key = (v.get("outage_cause"), v.get("outage_type"), v.get("outage_element_index"))
        value_ = float(v.get("value") or 0.0)
        excess = (value_ - limit_pct) if v.get("violation_type") in ("line_loading", "trafo_loading") else \
            100 * max(value_ - 1.0, 1.0 - value_, 0.0)
        by[key] = by.get(key, 0.0) + max(excess, 0.01)
    return [(k[0], k[1], int(k[2]), round(s, 3)) for k, s in sorted(by.items(), key=lambda kv: -kv[1])
            if k[0] and k[1] in ("line", "trafo") and k[2] is not None]


def _n1_study(request, ts: str, model, net, net_base, prices, pext0: dict, qext0: dict, held: dict | None) -> dict:
    """An N-1 security study (scopf.py): the least-disturbance and cheapest
    options for reference, and the dispatch secure after each critical outage
    — preventive, corrective, or both — every one screened under N-1 the same
    way. Returns {"kept": [(option, cost)], "info": per option id, "study": …}."""
    modes = [scopf.PREVENTIVE, scopf.CORRECTIVE] if request.n1 == "both" else [request.n1]
    lo, hi = request.opf_vm_lower, request.opf_vm_upper
    emergency = float(request.n1_emergency_loading_pct or _network_limits()["max_line_loading_pct"])
    first = fo.capture(model, "least_disturbance")
    cheapest = fo.cheapest(model, first, prices, pext0, qext0)

    def screen(option, base_net=None):
        n = copy.deepcopy(base_net if base_net is not None else net)
        _apply_option_dispatch(n, option)
        return _n1_sweep(n, ts, lo, hi, emergency, emergency)

    def summary(sweep):
        return {"outages_tested": sweep["total_outages_tested"],
                "outages_with_violations": sweep["total_outages_causing_violations"],
                "violations": sweep["total_violations"], "not_converged": len(sweep["non_converged_outages"])}

    reference = screen(first)
    # Which outages: the operator's, or the critical ones after the least-disturbance option.
    if request.n1_outages:
        index = _branch_index(net)
        wanted, unknown = [], []
        for raw in request.n1_outages:
            hit = index.get(str(raw).strip().replace("→", "->").casefold())
            (wanted.append((hit[0], hit[2], hit[3], None)) if hit else unknown.append(str(raw)))
        if unknown:
            raise HTTPException(status_code=400, detail=(
                f"n1_outages: there is no line or transformer {', '.join(map(repr, unknown))}."))
        candidates = list(dict.fromkeys(wanted))
    else:
        candidates = _critical_outages(reference, emergency)
    skipped, usable = [], []
    for name, table, idx, _sev in candidates:
        _, islanded = _outage_net(net, table, idx)
        if islanded:
            skipped.append({"outage": name, "reason": (
                f"it leaves {', '.join(_bus_display_name(net, b) for b in islanded[:4])} without supply: no "
                "dispatch keeps them supplied")})
        else:
            usable.append((name, table, idx))
    if len(usable) > _N1_MAX_OUTAGES:
        skipped += [{"outage": u[0], "reason": f"beyond the {_N1_MAX_OUTAGES} most severe outages studied at once; "
                                               "the screen still checks it"} for u in usable[_N1_MAX_OUTAGES:]]
        usable = usable[:_N1_MAX_OUTAGES]

    db = _db_for("full", request)
    fixed, held_now = _held_at_schedule(net_base, request)
    engine_kwargs = dict(opf_vm_lower=lo, opf_vm_upper=hi, lambda_p=request.opf_lambda_p,
                         lambda_q=request.opf_lambda_q, pg_max_overrides=request.pg_max_overrides or None,
                         pg_min_overrides=request.pg_min_overrides or None,
                         min_power_factor=request.opf_min_power_factor,
                         current_safety_margin=request.opf_current_safety_margin,
                         fixed_setpoints=fixed or None, load_flex=_load_flex(net))
    keep = set(held or {}) | set(held_now)
    load_reduction = bool(getattr(request, "n1_load_reduction", True))
    reach = {g: float(request.n1_reach_pct) / 100.0 * max(float(value(model.Pg_max[g])), 0.0) for g in model.G}

    def solve_mode(mode, outages):
        base = model.clone()
        scopf.release(base, keep)
        states, built = {}, []
        for name, table, idx in outages:
            net_k, _ = _outage_net(net, table, idx)
            try:
                pp.runpp(net_k)
            except Exception:  # noqa: BLE001 — the OPF starts from wherever it can
                pass
            try:
                _, m_k = fe.optimization_model_base(net_k, net_base, db["Yff_r"], db["Yff_i"], db["Yft_r"], db["Yft_i"],
                                                    db["TAPS"], db["trafo_ranges"], db["trafo_defaults"],
                                                    db["branch_to_trafo"], **engine_kwargs)
            except Exception:  # noqa: BLE001 — an outage the model cannot be built for is reported
                skipped.append({"outage": name, "reason": "the post-outage model could not be built"})
                continue
            scopf.release(m_k, keep)
            scopf.soften(m_k, emergency / 100.0, lo, hi)
            for e in m_k.Ext:                                   # the same grid-connection limits after the outage
                if e in base.Ext:
                    m_k.Pext[e].setlb(base.Pext[e].lb)
                    m_k.Pext[e].setub(base.Pext[e].ub)
            states[name] = m_k
            built.append((name, table, idx))
        top = scopf.build(base, states, mode, reach, request.opf_lambda_p, request.opf_lambda_q,
                          load_reduction=load_reduction)
        if not fo.solve(top):
            return None, built
        scopf.secure_first(top)
        fo.solve(top)              # if stage 2 fails, stage 1's solution stands (load_solutions=False)
        return top, built

    # A branch in a finding is named as everywhere else ("Bus_21 -> Bus_23 [L30]"),
    # whichever end the model measured it at.
    by_pair: dict[frozenset, list] = {}
    for name, pair, _t, _i in _branch_index(net).values():
        if name not in by_pair.setdefault(frozenset(pair), []):
            by_pair[frozenset(pair)].append(name)

    def branch_name(i, j) -> str:
        return " / ".join(by_pair.get(frozenset((i, j))) or
                          [f"{_bus_display_name(net, i)} -> {_bus_display_name(net, j)}"])

    info, kept, study_outages = {}, [(first, fo.action_cost(first, prices, pext0, qext0))], []
    info["least_disturbance"] = {"screen": summary(reference)}
    if cheapest is not None:
        kept.append((cheapest, fo.action_cost(cheapest, prices, pext0, qext0)))
        info["cheapest"] = {"screen": summary(screen(cheapest))}
    for mode in modes:
        outages, rounds = list(usable), 0
        top = built = None
        while outages and rounds < _N1_ROUNDS:
            rounds += 1
            top, built = solve_mode(mode, outages)
            if top is None:
                break
            if request.n1_outages:
                break                          # the operator's outages, and only those
            # Outages the new dispatch fails that were not studied: added.
            sweep = screen(fo.capture(top.base, f"n1_{mode}"))
            names = {o[0] for o in outages}
            new = [(n, t, i) for n, t, i, _ in _critical_outages(sweep, emergency)
                   if n not in names and not _outage_net(net, t, i)[1]]
            if not new or len(outages) >= _N1_MAX_OUTAGES:
                break
            outages += new[:_N1_MAX_OUTAGES - len(outages)]
        sid = f"n1_{mode}"
        if not usable:
            info[sid] = {"not_offered": "no outage to study: the screen found none with violations"}
            continue
        if top is None:
            info[sid] = {"not_offered": "the security-constrained OPF found no solution"}
            continue
        option = fo.capture(top.base, sid)
        kept.append((option, fo.action_cost(option, prices, pext0, qext0)))
        studied = [n for n, _, _ in built]
        study_outages = list(dict.fromkeys(study_outages + studied))
        detail = {"screen": summary(screen(option)), "studied": studied, "rounds": rounds}
        not_secured, plan = [], []
        for k, (name, table, idx) in enumerate(built):
            block = getattr(top, f"c{k}")
            remaining = scopf.residuals(block, branch_name, lambda b, n=net: _bus_display_name(n, b))
            if remaining:
                # What else might: opening one more branch after it (power flow),
                # with this option's dispatch — after its corrective action, if any.
                net_k, _ = _outage_net(net, table, idx)
                dispatch = fo.capture(block, sid) if mode == scopf.CORRECTIVE else option
                switching = _switching_remedies(net_k, dispatch, lo, hi, emergency)
                not_secured.append({"outage": name, "remaining": remaining[:4],
                                    **({"switching": switching} if switching else {})})
            if mode == scopf.CORRECTIVE:
                net_k, _ = _outage_net(net, table, idx)
                # First the dispatch alone: an outage it already survives needs no plan.
                alone = _power_flow_check(net_k, option, lo, hi, max_loading_pct=emergency)
                if alone.get("confirmed"):
                    plan.append({"outage": name, "actions": [], "no_action_needed": True,
                                 "checked_by_power_flow": True, "after_action": []})
                    continue
                after = fo.capture(block, sid)
                check = _power_flow_check(net_k, after, lo, hi, max_loading_pct=emergency)
                actions = []
                for g in sorted(after.p, key=_natural):
                    dp = round(after.p[g] - option.p[g], 2) + 0.0
                    bus = (option.vc_bus or {}).get(g)
                    dv = abs(after.v.get(bus, 0) - option.v.get(bus, 0)) if bus is not None else 0.0
                    if abs(dp) < _PLAN_MW and dv < _PLAN_VM:
                        continue                      # solver residue, not an instruction anyone would issue
                    actions.append({"unit": g, "dp_mw": dp,
                                    **({"setpoint_vm_pu_before": round(option.v.get(bus, 0), 3),
                                        "setpoint_vm_pu_after": round(after.v.get(bus, 0), 3)}
                                       if bus is not None and dv >= _PLAN_VM else {})})
                loads = _load_actions(net, after)
                plan.append({"outage": name, "actions": actions, "no_action_needed": False,
                             # When moving the units is not enough: load reduced after the outage.
                             **({"loads": loads} if loads else {}),
                             "checked_by_power_flow": check.get("confirmed"),
                             "after_action": check.get("violations") or []})
        detail["secured"] = len(built) - len(not_secured)
        if mode == scopf.CORRECTIVE:
            unsecured = {x["outage"] for x in not_secured}
            detail["secured_with_load_reduction"] = sum(1 for p in plan if p.get("loads")
                                                        and p["outage"] not in unsecured)
            detail["load_reduction_allowed"] = load_reduction
        detail["not_secured"] = not_secured
        if plan:
            detail["corrective_plan"] = plan
            detail["screen_is"] = "before any corrective action"
        info[sid] = detail
    study = {"modes": modes, "outages_studied": study_outages, "skipped": skipped,
             "emergency_loading_pct": emergency, "post_outage_band_pu": [lo, hi],
             **({"reach_pct": request.n1_reach_pct} if scopf.CORRECTIVE in modes else {}),
             "outages_tested_by_screen": reference["total_outages_tested"]}
    return {"kept": kept, "info": info, "study": study}


# A study that reads a moment's indicators (`_kpi_moment`) sets a dict here
# and gets, besides the options, the solved model they came from: the flexibility
# the moment requires and how far the units can go are solved on the same model.
_FLEX_MODEL_SINK: contextvars.ContextVar = contextvars.ContextVar("flex_model_sink", default=None)


def _flexibility_options(request, ts: str, model, net, net_base, slack_cap: float, held: dict | None = None,
                         outage: dict | None = None, caps_shown: dict | None = None,
                         last_resort: dict | None = None) -> dict:
    """Several corrective dispatches from the solved, capped least-disturbance
    model, scored the same way. The operator chooses; nothing here ranks them.
    `last_resort` ({"why", "restart", "fixed"}): unit redispatch alone found no
    dispatch on `model`; the options are those that also reduce load."""
    # The exchange at the studied point (its loads, generation at schedule) —
    # not the unscaled network's, or a load-scaling study would count the extra
    # load itself as a cost of the corrective action.
    ref = net if _finite_exchange(net) else net_base
    pext0 = {e: float(ref.res_ext_grid["p_mw"].values[i]) for i, e in enumerate(model.Ext)}
    qext0 = {e: float(ref.res_ext_grid["q_mvar"].values[i]) for i, e in enumerate(model.Ext)}
    common = {"timestamp": ts, "mode": "options", "slack_max_mw": slack_cap,
              "opf_vm_lower_used": request.opf_vm_lower, "opf_vm_upper_used": request.opf_vm_upper,
              "current_safety_margin": request.opf_current_safety_margin,
              # The operator's conditions: units kept at their schedule in every option,
              # branches held below a share of their rating.
              **({"held_units": held} if held else {}), **({"branch_limits": caps_shown} if caps_shown else {})}
    # A held unit gets no instruction at all: its set-point is kept too, not
    # only its active power (pinned already).
    model, kept_too = (fo.with_setpoints_kept(model, [g for g in (held or {}) if g in model.G])
                       if not last_resort else (model, True))
    if held and not kept_too:
        common["held_note"] = ("Their active power is held; keeping their voltage set-points as well has no "
                               "feasible dispatch, so those may change.")
    sink = _FLEX_MODEL_SINK.get()
    if sink is not None:
        sink.update(model=model, net=net, pext0=pext0, qext0=qext0, last_resort=bool(last_resort))
    first = fo.capture(model, "least_disturbance")
    # No action: the schedule itself (active and reactive output as scheduled)
    # meets every limit. Not "no instruction": a condenser past its reactive
    # limit cannot hold its set-point, and its bus voltage differs untouched.
    n1 = getattr(request, "n1", None)
    # An N-1 study goes on even then: secure now is not secure after an outage.
    if not last_resort and not n1 and not first.moved and all(
            abs(first.q[g] - first.q0[g]) < fo.MOVE_MVAR for g in first.q):
        return {**common, "status": "no_action_needed", "feasible": True, "converged": True, "options": [],
                "message": "The schedule already meets every limit: no corrective action is needed, so there "
                           "are no options to compare."}
    prices, sources = _flex_prices(net, model)
    budget = None
    n1_info = {}
    if last_resort:
        kept, not_offered = fo.last_resort(model, prices, pext0, qext0, restart=last_resort.get("restart"))
        cheapest_total = min((c["total"] for _, c in kept), default=None)
        common.update({"set": "last_resort",
                       # Why the units alone could not: the diagnosis, in words. A remedy
                       # it names is another way out, not a condition of these options
                       # (seen live: "raise the grid voltage" read as a prerequisite).
                       "redispatch_alone": {"clears_violations": False, "why": last_resort.get("why"),
                                            "note": "Why redispatching the units alone fails. Any other remedy "
                                                    "named here (e.g. raising the grid-connection voltage) is "
                                                    "an alternative to the load-reduction options, not part of "
                                                    "them: they need nothing else changed."}})
        if not kept:
            return {**common, "status": "infeasible", "feasible": False, "converged": False, "options": [],
                    "not_offered": not_offered,
                    "message": "Neither redispatching the units nor reducing load clears the violations: "
                               + "; ".join(n["reason"] for n in not_offered) + "."}
    elif n1:
        study = _n1_study(request, ts, model, net, net_base, prices, pext0, qext0, held)
        kept, n1_info = study["kept"], study["info"]
        not_offered = [{"id": sid, "label": fo.STRATEGIES[sid][0], "reason": entry["not_offered"]}
                       for sid, entry in n1_info.items() if entry.get("not_offered")]
        cheapest_total = next((c["total"] for o, c in kept if o.id == "cheapest"), None)
        common.update({"set": "n1", "n1_study": study["study"]})
    elif request.near_optimal:
        methods = tuple(request.near_optimal_methods or fo.NEAR_OPTIMAL_METHODS)
        kept, not_offered, cheapest_total, budget, ranges = fo.generate_near_optimal(
            model, prices, pext0, qext0, request.near_optimal_pct, methods=methods,
            count=request.near_optimal_count)
        common.update({"set": "near_optimal", "near_optimal_pct": request.near_optimal_pct,
                       "budget_eur_per_h": budget, "near_optimal_methods": list(methods),
                       # How far each unit can go within the margin — what the
                       # alternatives have in common, and where they differ.
                       **({"near_optimal_ranges": ranges} if ranges else {})})
    else:
        kept, not_offered, cheapest_total = fo.generate(model, prices, pext0, qext0)
    best = next((o for o, _ in kept if o.id == "cheapest"), None)
    set_id = _remember_options(request, ts, kept, outage, held) if kept else None
    if set_id:
        common["set_id"] = set_id

    def bus(b: int) -> str:
        return _bus_display_name(net, b) if b in net.bus.index else f"Bus_{b}"

    options = []
    for option, cost in kept:
        moved_q = {g: option.q[g] - option.q0[g] for g in option.q if abs(option.q[g] - option.q0[g]) >= fo.MOVE_MVAR}
        told = option.instructions

        def row(g):
            out = {"unit": g, "p_mw_before": round(option.p0[g], 4), "p_mw_after": round(option.p[g], 4),
                   "dp_mw": round(option.p[g] - option.p0[g], 4), "q_mvar_before": round(option.q0[g], 4),
                   "q_mvar_after": round(option.q[g], 4), "dq_mvar": round(option.q[g] - option.q0[g], 4),
                   "instruction": told.get(g)}
            b = (option.vc_bus or {}).get(g)
            if b is not None and option.v0 and b in option.v0:
                # A unit that holds a voltage: its instruction is the set-point;
                # its reactive output follows from it.
                out["setpoint_vm_pu_before"] = round(option.v0[b], 4)
                out["setpoint_vm_pu_after"] = round(option.v.get(b, option.v0[b]), 4)
            return out
        rows = [row(g) for g in option.p if g in told or g in moved_q]
        for position, e in enumerate(option.pext):
            grid = uc.ext_grid_name(net, net.ext_grid.index[position]) if position < len(net.ext_grid) else str(e)
            rows.append({"unit": grid, "p_mw_before": round(pext0[e], 4),
                         "p_mw_after": round(option.pext[e], 4), "dp_mw": round(option.pext[e] - pext0[e], 4),
                         "q_mvar_before": round(qext0[e], 4), "q_mvar_after": round(option.qext[e], 4),
                         "dq_mvar": round(option.qext[e] - qext0[e], 4)})
        volts = [v for v in option.v.values() if v is not None]
        entry = {
            "id": option.id, "label": option.label, "how": option.how,
            # Its reference: a later study applies this option by it (the N-1 screen).
            "ref": f"{set_id}:{option.id}",
            # Strategies that came out as this same dispatch: "least disturbance
            # is also the cheapest" is an answer in itself.
            "also": [d["label"] for d in not_offered if d.get("same_as") == option.id],
            # Units given an instruction: active power, and the voltage set-point
            # of a unit that holds one (or the reactive output of one that does
            # not). A held voltage moves reactive output with no instruction.
            "units_called": sorted(told),
            "n_units_called": len(told),
            "instructions": dict(sorted(told.items())),
            "units_changing_p": sorted(option.moved),
            "moved_mw": round(sum(abs(v) for v in option.moved.values()), 4),
            "moved_mvar": round(sum(abs(v) for v in moved_q.values()), 4),
            "import_mw": round(sum(option.pext.values()), 4),
            "cost_eur_per_h": cost["total"],
            "cost_breakdown_eur_per_h": {k: v for k, v in cost.items() if k != "total"},
            "min_vm_pu": round(min(volts), 4) if volts else None,
            "max_vm_pu": round(max(volts), 4) if volts else None,
            "max_loading_pct": round(option.max_loading_pct, 2) if option.max_loading_pct is not None else None,
            "max_loading_branch": (f"{bus(option.max_loading_branch[0])} – {bus(option.max_loading_branch[1])}"
                                   if option.max_loading_branch else None),
            "dispatch": rows,
            # The same dispatch, re-solved by an ordinary AC power flow.
            "power_flow_check": _power_flow_check(net, option, request.opf_vm_lower, request.opf_vm_upper,
                                                  caps_shown),
            # In an N-1 study: its N-1 screen, and for a secure option what it
            # studied, what it could not secure and (corrective) its plan.
            **({"n1": {k: v for k, v in n1_info[option.id].items() if k != "not_offered"}}
               if option.id in n1_info else {}),
            # Load reductions, where an option has them: demand response and
            # load relief, by load — instructions to customers, not units.
            **({"load_actions": _load_actions(net, option), "loads_called": len(_load_actions(net, option)),
                "load_reduction_mw": round(option.load_reduction, 3),
                "demand_response_mw": round(sum((option.dr or {}).values()), 3),
                "load_relief_mw": round(sum((option.relief or {}).values()), 3)}
               if option.load_reduction else {}),
        }
        if cheapest_total is not None:
            gap = cost["total"] - cheapest_total
            entry["above_cheapest_eur_per_h"] = round(gap, 2)
            if cheapest_total > 0.01:
                entry["above_cheapest_pct"] = round(100.0 * gap / cheapest_total, 1)
        if option.id.startswith("near_optimal") and best is not None:
            # What this alternative does without: the units the cheapest changes
            # and it does not — the reason to look at it.
            avoids = sorted(set(best.moved) - set(option.moved))
            adds = sorted(set(option.moved) - set(best.moved))
            entry["avoids"], entry["adds"] = avoids, adds
            gap_text = (f"{entry['above_cheapest_pct']:.1f} % above the cheapest" if "above_cheapest_pct" in entry
                        else f"{entry.get('above_cheapest_eur_per_h', 0):+.2f} €/h against the cheapest")
            changes = ([f"avoids {', '.join(avoids)}"] if avoids else []) + ([f"adds {', '.join(adds)}"] if adds else [])
            # How it was found (the MGA method), then what it does differently.
            entry["found_by"] = option.how
            entry["how"] = "; ".join([gap_text, option.how, "; ".join(changes) or "the same units, moved differently"])
        options.append(entry)
    counts: dict = {}
    for kinds in sources.values():
        for k in kinds:
            counts[k] = counts.get(k, 0) + 1
    used_loads = sorted({i for o, _ in kept for i in set(o.dr or {}) | set(o.relief or {})})
    return {**common, "status": "options", "feasible": True, "converged": True,
            "message": ((f"Redispatching the units alone cannot clear the violations (why: `redispatch_alone`). "
                         f"{len(options)} option(s) that also reduce load — demand response first, load relief only "
                         "beyond it — each meeting every limit with nothing else changed; the operator chooses."
                         if last_resort else
                         f"N-1 security study: {len(common['n1_study']['outages_studied'])} critical outage(s) "
                         "studied; the options are compared by their N-1 screens, each meets every limit with "
                         "everything in service; the operator chooses." if n1 else
                         f"The cheapest option and {len(options) - 1} near-optimal alternative(s), each costing at "
                         f"most {budget:.2f} €/h ({request.near_optimal_pct:g} % above the cheapest) and picked to "
                         "be as different from the others as the margin allows (how each was found is in `how`); "
                         "each meets every limit; the operator chooses." if request.near_optimal else
                         f"{len(options)} corrective option(s), each meeting every limit; the operator chooses.")
                        + " Costs are of the corrective action (doing nothing costs 0)."),
            "options": options, "not_offered": not_offered,
            "costs": {"currency": "EUR", "prices_from": counts,
                      "meaning": "Prices from the cost table (System tab): the case file, a synthetic rule, or "
                                 "the operator's changes. Say which when quoting a cost.",
                      "prices": {"up_eur_per_mwh": prices.up, "down_eur_per_mwh": prices.down,
                                 "reactive_eur_per_mvarh": prices.q, "import_eur_per_mwh": prices.import_,
                                 "reactive_band_tan_phi": prices.band,
                                 "beyond_band_eur_per_mvarh": prices.penalty},
                      # The loads the options reduce, with their prices (cost table, per load).
                      **({"load_prices": {_load_label(net, i)["load"]: {
                          "demand_response_eur_per_mwh": (prices.dr or {}).get(i),
                          "load_relief_eur_per_mwh": (prices.relief or {}).get(i)} for i in used_loads}}
                         if used_loads else {})}}


def _branch_index(net) -> dict:
    """Every line and transformer under each name an operator may use: the
    results' name ("Bus_21 -> Bus_23 [L30]"), its tag ("L30") and its buses
    ("Bus_21 -> Bus_23"). alias (casefolded) -> (name, pair of buses, table, index)."""
    names: dict[str, tuple] = {}
    for table, label in (("line", _line_display_name), ("trafo", _trafo_display_name)):
        frame = net.get(table)
        if frame is None or not len(frame):
            continue
        a, b = ("from_bus", "to_bus") if table == "line" else ("hv_bus", "lv_bus")
        for idx in frame.index:
            name = label(net, idx)
            pair = (int(frame.at[idx, a]), int(frame.at[idx, b]))
            for alias in {name, name.split(" [")[0], name.rsplit("[", 1)[-1].rstrip("]") if "[" in name else name}:
                names.setdefault(alias.casefold(), (name, pair, table, int(idx)))
    return names


def _branch_caps(net, raw: dict | None) -> tuple[dict, dict]:
    """`branch_limits` as the engine takes them — {(from bus, to bus): fraction
    of the rating} — and as results show them: {branch name: %}. A branch is
    named as the results name it ("Bus_21 -> Bus_23 [L30]"), by its tag ("L30",
    "T1") or by its buses ("Bus_21 -> Bus_23"). Refuses an unknown branch or a
    share outside 10–100 %, with the reason."""
    if not raw:
        return {}, {}
    names = _branch_index(net)
    engine, shown, unknown = {}, {}, []
    for key, pct in raw.items():
        found = names.get(str(key).strip().replace("→", "->").casefold())
        if found is None:
            unknown.append(str(key))
            continue
        try:
            pct = float(pct)
        except (TypeError, ValueError):
            pct = -1.0
        if not 10.0 <= pct <= 100.0:
            raise HTTPException(status_code=400, detail=(
                f"branch_limits: {found[0]} at {pct:g} % — a cap is a share of the rating, 10–100 %."))
        engine[found[1]] = pct / 100.0
        shown[found[0]] = pct
    if unknown:
        examples = sorted({v[0] for v in names.values()})[:6]
        raise HTTPException(status_code=400, detail=(
            f"branch_limits: there is no line or transformer {', '.join(map(repr, unknown))}. Name one as the results "
            f"do, by its tag or by its buses — e.g. {', '.join(examples)}."))
    return engine, shown


def _held_at_schedule(net, request) -> tuple[dict, dict]:
    """`hold_units` as set-points at their schedule (MW at this moment), merged
    into `fixed_setpoints` (an explicit set-point wins). Returns (set-points,
    the units held with their MW). An unknown unit is refused by name: the
    engine would ignore it, and the dispatch would move the unit after all."""
    fixed = dict(request.fixed_setpoints or {})
    if not request.hold_units:
        return fixed, {}
    schedule = {gm.unit_name(net, table, idx): float(net[table].at[idx, "p_mw"])
                for table in ("sgen", "gen") for idx in net[table].index}
    by_fold = {name.casefold(): name for name in schedule}
    held, unknown = {}, []
    for raw in request.hold_units:
        name = by_fold.get(str(raw).strip().casefold())
        if name is None:
            unknown.append(str(raw))
            continue
        held[name] = round(schedule[name], 4)
        fixed.setdefault(name, held[name])
    if unknown:
        raise HTTPException(status_code=400, detail=(
            f"hold_units: there is no unit {', '.join(map(repr, unknown))}. Units: {', '.join(sorted(schedule))}."))
    return fixed, held


@app.post("/api/flexibility/optimize")
@_off_event_loop
async def optimise_flexibility(request: FlexibilityRequest):
    """Run the Pyomo AI to find the cheapest dispatch that cures the grid.

    Uses the **intact-grid** admittance database (``db_full``) — this
    endpoint does not model any outage, only voltage / thermal issues
    induced by the load-stress slider.

    Pipeline
    --------
    1. Clone the base network, apply measurements, neutralize taps.
    2. Optionally drop a single substation's sgens (``disabled_generators``)
       to simulate a generator forced offline.
    3. Run a seed power flow (swallowed on failure).
    4. Build the scaled network (copy + load multiplication).
    5. Connected solve via ``flex_engine.optimization_model_base`` with
       the intact Ybus matrices and the engine's default ±200 MW slack.
    6. **Constrained-slack re-solve** via
       :func:`solve_constrained_slack_scenario` so the dispatch
       respects the global Sweden-cable slider (``slack_max_mw``).
       This makes the table on the *Flexibility Management* tab
       consistent with the KPI tab — both reflect the same
       slider-driven cable cap.
    7. Return the **full** regulation DataFrame from
       :func:`flex_engine.prepare_regulation_df_rsa` — every generator
    row plus the external-grid row — so the user
       can see exactly where the dispatch landed (typically on the
       slack when it has spare capacity).

    Two-Fold logic is handled inside ``flex_engine``:
      * Fold 1 checks if the grid is naturally secure with generators fixed.
      * Fold 2 unfixes them and minimizes ``|Pg_new - Pg_base|``.
    """
    if _db_for("full", request) is None: raise HTTPException(status_code=503)
    start = None
    if request.from_option:
        stored, held_option, request, held_since = _option_context(request, later=True, ref=request.from_option)
        if stored.get("outage"):
            raise HTTPException(status_code=400, detail=(
                f"from_option: {stored['id']}:{held_option.id} was computed after the outage of "
                f"{stored['outage']['element']}; start from it with optimize_contingency for that outage."))
        start = (stored, held_option, held_since)

    def echoed(result: dict) -> dict:
        if not start or not isinstance(result, dict):
            return result
        return {**result, "from_option": _option_echo(start[0], start[1], result.get("timestamp") or "", start[2])}
    if request.n1 and request.near_optimal:
        raise HTTPException(status_code=400, detail=(
            "n1 and near_optimal are two different option sets: ask for the N-1 secure options, or for "
            "near-optimal alternatives, one at a time."))

    current_ts = _resolve_tick(request.data_source, request.timestamp)
    net_base = copy.deepcopy(_net_for(request))
    measurement_prod_cons = prepare_measurement_df(_lookup_measurement_df(current_ts, request.data_source))

    net_base, _ = _assign_tick(net_base, measurement_prod_cons, current_ts, _dataset_of(request))
    if start:
        _apply_option_dispatch(net_base, start[1], loads=False, held=start[2])   # the held set-points: the schedule now

    disabled_gens = request.disabled_generators

    # Issue 3: capture the full-grid operating point *before* any generators are
    # removed so the dispatch chart can show all three stages:
    #   P_pre_disable → P_base (post-disable seed) → P_new (OPF result).
    dispatch_pre_disable = None
    if disabled_gens:
        _net_pre = copy.deepcopy(net_base)
        try:
            pp.runpp(_net_pre)
            names = _sgen_names(_net_pre)
            dispatch_pre_disable = {
                names[idx]: round(float(_net_pre.res_sgen.at[idx, 'p_mw']), 4)
                for idx in _net_pre.sgen.index
            }
            slack_label = str((app_data.get("grid_profile") or {}).get("slack_name") or "External Grid")
            dispatch_pre_disable[slack_label] = round(
                float(_net_pre.res_ext_grid['p_mw'].values[0]), 4
            )
        except Exception:
            dispatch_pre_disable = None
        _drop_disabled_sgens(net_base, disabled_gens)

    try:
        pp.runpp(net_base)
    except Exception:
        pass
    fixed_setpoints, held = _held_at_schedule(net_base, request)
    caps, caps_shown = _branch_caps(net_base, request.branch_limits)

    net = copy.deepcopy(net_base)
    _scale_loads(net, request.load_scaling_factor)
    if start:
        _apply_option_dispatch(net, start[1], units=False, held=start[2])        # and its load reductions, if any

    try:
        pp.runpp(net)
    except Exception:
        pass

    db = _db_for("full", request)

    try:
        results_opt, model_opt = fe.optimization_model_base(
            net, net_base, db["Yff_r"], db["Yff_i"], db["Yft_r"], db["Yft_i"],
            db["TAPS"], db["trafo_ranges"], db["trafo_defaults"], db["branch_to_trafo"],
            missing_substation=disabled_gens or None,
            opf_vm_lower=request.opf_vm_lower, opf_vm_upper=request.opf_vm_upper,
            lambda_p=request.opf_lambda_p, lambda_q=request.opf_lambda_q,
            pg_max_overrides=request.pg_max_overrides or None,
            pg_min_overrides=request.pg_min_overrides or None,
            min_power_factor=request.opf_min_power_factor,
            current_safety_margin=request.opf_current_safety_margin,
            fixed_setpoints=fixed_setpoints or None,
            vm_upper_per_bus=request.vm_upper_per_bus or None,
            vm_lower_per_bus=request.vm_lower_per_bus or None,
            branch_limits=caps or None,
            load_flex=_load_flex(net),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if results_opt is None: raise HTTPException(status_code=500)

    tc = getattr(results_opt.solver, "termination_condition", None)
    if tc != TerminationCondition.optimal:
        _phys_cap = app_data.get("slack_max_mw", 70.0)
        _reported_cap = float(_phys_cap if request.slack_max_mw is None else min(request.slack_max_mw, _phys_cap))
        failed = {"timestamp": current_ts, "status": str(tc), "message": "Infeasible.",
                  "feasible": False, "converged": False,
                  "activated_resources": [], "slack_max_mw": _reported_cap,
                  **({"held_units": held} if held else {}), **({"branch_limits": caps_shown} if caps_shown else {})}
        rescue = model_opt.clone() if model_opt is not None else None      # the diagnosis changes the model
        failed = _with_infeasibility_diagnosis(failed, model_opt, net)
        return echoed(_with_last_resort(failed, request, current_ts, rescue, net, net_base, held, caps_shown,
                                        fixed_setpoints, cap=_reported_cap))

    # Constrained-slack re-solve so the dispatch respects the global
    # Sweden-cable slider. If the constrained solve fails we fall back
    # to the connected dispatch so the user always sees *something*,
    # but flag it in the message so the inconsistency is visible.
    _phys_cap = app_data.get("slack_max_mw", 70.0)
    slack_cap = float(max(0.0, _phys_cap if request.slack_max_mw is None else min(request.slack_max_mw, _phys_cap)))
    # Snapshot regulation + voltages from the unconstrained (connected) solve BEFORE
    # solve_constrained_slack_scenario mutates model_opt's Pext bounds.  If the
    # constrained re-solve is infeasible the snapshot is the physically valid fallback.
    df_reg_flex_rsa = fe.prepare_regulation_df_rsa(model_opt, net, net_base, current_ts)
    voltage_data = fe.extract_post_opf_voltages(model_opt, net)
    # Whether the dispatch returned actually respects the cap. When it does
    # not, the result used to say status "optimal" and "Grid restored at slack
    # cap ±X MW" — with the uncapped dispatch (190 MW against a 10 MW cap, in
    # a reproduction) as the answer, cached for the envelope tool. A failed
    # capped solve is now a failed result; the uncapped dispatch is kept only
    # as a labelled reference (the withhold rule: never hand over a figure as
    # an answer to a question it does not answer).
    cap_respected = True
    cap_failure = ""
    try:
        cscen_results, cscen_model = solve_constrained_slack_scenario(model_opt, slack_cap, slack_q_max_mvar=request.slack_q_max_mvar, fixed_setpoints=fixed_setpoints or None)
        cscen_tc = (getattr(cscen_results.solver, "termination_condition", None)
                    if cscen_results is not None else None)
        if cscen_tc == TerminationCondition.optimal:
            df_reg_flex_rsa = fe.prepare_regulation_df_rsa(cscen_model, net, net_base, current_ts)
            voltage_data = fe.extract_post_opf_voltages(cscen_model, net)
        else:
            cap_respected = False
            cap_failure = f"the capped solve ended {cscen_tc or 'without a solution'}"
    except Exception as e:
        cap_respected = False
        cap_failure = f"the capped solve failed: {e}"

    # ----------------------------------------------------------------------
    # Numerical-noise filter for the dispatch table.
    # Ipopt converges to a tolerance of ~1e-8, so unchanged generators show
    # Pg_up / Pg_down values like 5.3e-9 — physically meaningless, just
    # solver residuals. We zero those out and round all P-columns to
    # 4 decimals (0.1 kW resolution) so the table is readable.
    # We deliberately DO NOT zero Pg_base / Pg_new below the same threshold
    # because some substations have legitimately small baselines (e.g.
    # Juliet South ≈ 0.00023 MW = 0.23 kW) that we want to keep visible.
    # ----------------------------------------------------------------------
    NOISE_THRESHOLD_MW = 1e-4
    DECIMALS = 4
    for _col in ('Pg_up', 'Pg_down', 'Qg_up', 'Qg_down'):
        if _col in df_reg_flex_rsa.columns:
            mask = df_reg_flex_rsa[_col].abs() < NOISE_THRESHOLD_MW
            df_reg_flex_rsa.loc[mask, _col] = 0.0
    for _col in ('Pg_base', 'Pg_new', 'Pg_up', 'Pg_down',
                 'Qg_base', 'Qg_new', 'Qg_up', 'Qg_down'):
        if _col in df_reg_flex_rsa.columns:
            df_reg_flex_rsa[_col] = df_reg_flex_rsa[_col].round(DECIMALS)

    # Ensure NaN/inf from solver artifacts cannot leak into JSON payloads.
    df_reg_flex_rsa = df_reg_flex_rsa.replace([np.inf, -np.inf], np.nan)
    df_reg_flex_rsa = df_reg_flex_rsa.where(pd.notna(df_reg_flex_rsa), None)
    regulation_data = df_reg_flex_rsa.to_dict(orient="records")
    if cap_respected:
        msg = f"Grid restored at slack cap ±{slack_cap:.0f} MW."
    else:
        msg = (f"No dispatch keeps the external-grid exchange within ±{slack_cap:.0f} MW "
               f"({cap_failure}). The dispatch without the cap is attached only as "
               "`uncapped_reference_dispatch`; it does not respect the cap.")
    excluded_count = int(voltage_data.get("excluded_bus_count_post_opf") or 0)
    if excluded_count > 0:
        msg += f" Excluded {excluded_count} bus(es) from post-OPF voltage output; see excluded_buses_post_opf for details."

    # Base bus voltages (pre-OPF power flow on the scaled network)
    # so the frontend can render ΔV = V_post_opf − V_base per bus.
    try:
        bus_voltages_base = []
        for i in net.bus.index:
            vm = _safe_float(net.res_bus.at[i, "vm_pu"], ndigits=6)
            if vm is None:
                continue
            bus_voltages_base.append(
                {
                    "bus": int(i),
                    "bus_name": (str(net.bus.at[i, "name"]).strip() or f"Bus_{int(i)}") if "name" in net.bus.columns else f"Bus_{int(i)}",
                    "vm_pu": vm,
                }
            )
    except Exception:
        bus_voltages_base = []

    # Phase 3 Placeholder: publish_flex_rsa_results(df, current_ts, topic="topic6916"...)
    if not cap_respected:
        failed = {
            "timestamp": current_ts, "status": "cap_infeasible", "message": msg,
            "feasible": False, "converged": True, "slack_cap_respected": False,
            "activated_resources": [], "slack_max_mw": slack_cap,
            "uncapped_reference_dispatch": regulation_data,
            **({"held_units": held} if held else {}), **({"branch_limits": caps_shown} if caps_shown else {}),
        }
        return echoed(_with_last_resort(failed, request, current_ts, model_opt.clone(), net, net_base, held,
                                        caps_shown, fixed_setpoints, cap=slack_cap))
    if request.options or request.n1:          # an N-1 study is a set of options
        app_data["last_dispatch_result"] = regulation_data
        app_data["last_dispatch_timestamp"] = current_ts
        return echoed(_flexibility_options(request, current_ts, cscen_model, net, net_base, slack_cap, held,
                                           caps_shown=caps_shown))
    response = {
        "timestamp": current_ts, "status": "optimal", "message": msg,
        "feasible": True, "converged": True, "slack_cap_respected": True,
        "activated_resources": regulation_data, "slack_max_mw": slack_cap,
        # The operator's conditions: units kept at their schedule, with the MW,
        # and branches held below a share of their rating.
        **({"held_units": held} if held else {}), **({"branch_limits": caps_shown} if caps_shown else {}),
    }
    response.update(voltage_data)
    if bus_voltages_base:
        response["bus_voltages_base"] = bus_voltages_base
    if dispatch_pre_disable is not None:
        response["dispatch_pre_disable"] = dispatch_pre_disable
    app_data["last_dispatch_result"] = regulation_data
    app_data["last_dispatch_timestamp"] = current_ts
    return echoed(response)


class HoldOptionRequest(GridRequest):
    """Request body for ``POST /api/flexibility/hold``: a corrective option
    (`option`, required) held from its moment over the next `hours` hours;
    `n1` adds the N-1 screen of the held dispatch at each moment studied."""
    hours: float = Field(default=6, gt=0, le=48)
    n1: bool = False


_HOLD_MAX_TICKS = 49             # moments studied at most (sampled evenly beyond)
_HOLD_N1_MAX_TICKS = 13          # with the N-1 screen at each


def _ticks_within(timestamps: list, start: str, hours: float, most: int) -> list:
    """The series' moments from `start` to `hours` after it, at most `most`
    of them. When there are more, they are evenly spaced with the first and
    the last always kept — on a round step where one lands on the last and
    keeps at least half of `most` (16:00–00:00 every 15 min, at most 6:
    16:00, 18:00, …, 00:00), else as evenly as the series allows."""
    first = timestamps.index(start)
    try:
        end = pd.Timestamp(start) + pd.Timedelta(hours=float(hours))
        ticks = [t for t in timestamps[first:] if pd.Timestamp(t) <= end]
    except (ValueError, TypeError):
        ticks = timestamps[first:first + int(hours) + 1]
    if len(ticks) > most:
        last = len(ticks) - 1
        fewest = max(2, (most + 1) // 2)
        step = next((s for s in range(math.ceil(last / (most - 1)), last + 1)
                     if last % s == 0 and last // s + 1 >= fewest), None)
        picks = range(0, last + 1, step) if step else sorted({round(i * last / (most - 1)) for i in range(most)})
        ticks = [ticks[i] for i in picks]
    return ticks


def _rsa_brief(result: dict) -> dict:
    """An assessment in a line: its violations and worst figures."""
    rows = result.get("violations") or []

    def words(v):
        name = v.get("element_name") or v.get("element") or "?"
        value_ = v.get("value")
        unit = " p.u." if "vm" in str(v.get("violation_type")) else " %"
        return f"{name} at {value_:.3f}{unit}" if unit == " p.u." and isinstance(value_, (int, float)) else \
            f"{name} at {value_:.1f}{unit}" if isinstance(value_, (int, float)) else str(name)
    loadings = [r.get("loading_percent") for r in result.get("all_line_loading") or []
                if isinstance(r.get("loading_percent"), (int, float))]
    volts = [r.get("vm_pu") for r in result.get("all_voltages") or [] if isinstance(r.get("vm_pu"), (int, float))]
    return {"violations": int(result.get("total_violations") or 0),
            "max_line_loading_pct": round(max(loadings), 2) if loadings else None,
            "min_vm_pu": round(min(volts), 4) if volts else None,
            "worst": [words(v) for v in rows[:3]]}


@app.post("/api/flexibility/hold")
@_off_event_loop
async def hold_option(request: HoldOptionRequest):
    """Hold a corrective option from its moment over the next hours: at each
    tick, the assessment with its instructions kept (every other unit follows
    its schedule; load reductions stay) beside the schedule untouched — how
    long the action keeps the grid within limits, and when a new one is
    needed. With `n1`, each tick's N-1 screen of the held dispatch too."""
    if not request.option:
        raise HTTPException(status_code=400, detail="option: give the reference of a corrective option "
                                                    "(e.g. S3:cheapest) to hold.")
    stored, option, req, _ = _option_context(request.model_copy(update={"timestamp": None}), later=True)
    timestamps, _dataset, _label = _resolve_dataset(stored["data_source"])
    if stored["timestamp"] not in timestamps:
        raise HTTPException(status_code=400, detail=(
            f"option: {stored['id']}:{option.id} was computed at {stored['timestamp']}, which has no hours "
            "after it in the data; hold an option computed at a moment of the series."))
    ticks = _ticks_within(timestamps, stored["timestamp"], request.hours,
                          _HOLD_N1_MAX_TICKS if request.n1 else _HOLD_MAX_TICKS)
    note = None
    rows, lo, hi = [], req.vm_lower_pu, req.vm_upper_pu
    for k, ts in enumerate(ticks):
        held = k > 0
        schedule = _rsa_brief(_run_rsa_snapshot(ts, req))
        applied = (stored, option, held)
        with_option = _rsa_brief(_run_rsa_snapshot(ts, req, applied=applied))
        row = {"timestamp": ts, "held": held, "schedule": schedule, "with_option": with_option}
        if request.n1:
            net_held, _ = _prepared_net(ts, req, applied)
            sweep = _n1_sweep(net_held, ts, lo, hi, req.max_line_loading_pct, req.max_trafo_loading_pct)
            row["with_option"]["n1_outages_with_violations"] = sweep["total_outages_causing_violations"]
            row["with_option"]["n1_outages_tested"] = sweep["total_outages_tested"]
        rows.append(row)
    failing = next((r for r in rows if r["with_option"]["violations"]), None)
    holds_until = None
    for r in rows:
        if r["with_option"]["violations"]:
            break
        holds_until = r["timestamp"]
    echo = _option_echo(stored, option, stored["timestamp"], len(rows) > 1)
    if failing is None:
        message = (f"Held from {stored['timestamp']}, {option.label} keeps every limit through {rows[-1]['timestamp']} "
                   f"({len(rows) - 1} hour(s) after it).")
    elif failing is rows[0]:
        message = f"{option.label} does not meet every limit even at its own moment ({failing['timestamp']})."
    else:
        message = (f"Held from {stored['timestamp']}, {option.label} keeps every limit until {holds_until}; at "
                   f"{failing['timestamp']} it no longer does ({'; '.join(failing['with_option']['worst'][:2])}). "
                   f"A new corrective action is needed then — options from it: optimize_flexibility with "
                   f"from_option={echo['ref']} at {failing['timestamp']}.")
    return {"timestamp": stored["timestamp"], "mode": "hold", "option": echo, "hours": rows,
            "holds_until": holds_until, "first_failure": failing["timestamp"] if failing else None,
            "message": message + (f" {note}" if note else ""),
            "data_source": _dataset_label(req), "thresholds_used": _thresholds_used(req)}


class _WindowRequest(_TimeseriesRequest):
    """A window of moments: from `start` (default the clock, or the first
    forecast tick) to `end`, or `hours` after it; `moments` of them at most,
    evenly spaced."""
    start: str | None = None
    end: str | None = None
    hours: float = Field(default=6, gt=0, le=48)
    moments: int = Field(default=6, ge=2, le=12)
    load_scaling_factor: float = 1.0
    vm_upper_pu: float = 1.05            # the network's own limits unless sent (_NetworkLimitsMixin)
    vm_lower_pu: float = 0.95
    max_line_loading_pct: float = 90.0
    max_trafo_loading_pct: float = 90.0
    opf_vm_lower: float | None = None
    opf_vm_upper: float | None = None
    hold_units: list[str] = []
    branch_limits: dict[str, float] = {}
    # A caller-chosen id to follow a long study's progress (GET /api/progress/{id}).
    progress_id: str | None = Field(default=None, max_length=64)


def _window(request: _WindowRequest) -> list[str]:
    """The moments a window study runs at (the nominal point refused: it has no time)."""
    timestamps, _dataset, _label = _window_dataset(request.data_source)
    start = _resolve_tick(request.data_source, request.start or request.timestamp)
    hours = request.hours
    if request.end:
        end = _resolve_tick(request.data_source, request.end)
        if str(end) <= str(start):
            raise HTTPException(status_code=400, detail=f"end ({end}) must come after start ({start}).")
        hours = (pd.Timestamp(end) - pd.Timestamp(start)).total_seconds() / 3600.0
    ticks = _ticks_within(timestamps, start, hours, request.moments)
    if len(ticks) < 2:
        raise HTTPException(status_code=400, detail=f"The window from {start} holds one moment only; make it longer.")
    return ticks


def _gaps(ticks: list[str]) -> list[float]:
    """Hours between consecutive moments (a ramp limit is per gap)."""
    times = [pd.Timestamp(t) for t in ticks]
    return [(b - a).total_seconds() / 3600.0 for a, b in zip(times, times[1:])]


def _weights(ticks: list[str]) -> list[float]:
    """Hours each moment stands for in a total over the window — half the gap
    to each neighbour (the trapezoid rule), so the total covers exactly the
    window from the first moment to the last: 16:00–00:00 is eight hours,
    not nine."""
    gaps = _gaps(ticks)
    if not gaps:
        return [0.0] * len(ticks)
    return [(gaps[k - 1] if k > 0 else 0.0) / 2 + (gaps[k] if k < len(gaps) else 0.0) / 2
            for k in range(len(ticks))]


class KPIOverTimeRequest(_KPIStudy, _WindowRequest):
    """Request body for ``POST /api/kpi/over_time``: the indicators at each
    moment of a window."""


class SecurityOverTimeRequest(_WindowRequest):
    """Request body for ``POST /api/flexibility/security_over_time``."""
    n1: Literal["preventive", "corrective", "both"] = "preventive"


# ---------------------------------------------------------------------------
# Long studies: progress, and moments in parallel worker processes
# ---------------------------------------------------------------------------

# Progress of a long study, by the id its caller chose (`progress_id`), read
# while the study runs (GET /api/progress/{id}) — that endpoint takes no lock.
_PROGRESS: dict[str, dict] = {}
_PROGRESS_LOCK = threading.Lock()
_PROGRESS_KEEP_S = 3600.0


def _report_progress(job: str | None, **fields) -> None:
    if not job:
        return
    now = time.time()
    with _PROGRESS_LOCK:
        for old in [k for k, v in _PROGRESS.items() if now - v.get("updated", now) > _PROGRESS_KEEP_S]:
            _PROGRESS.pop(old, None)
        _PROGRESS.setdefault(job, {"started": now}).update(fields, updated=now)


@app.get("/api/progress/{job}")
async def study_progress(job: str):
    """How far a long study has got: {"done", "total", "last", "elapsed_s"}."""
    with _PROGRESS_LOCK:
        entry = dict(_PROGRESS.get(job) or {})
    if not entry:
        return {"known": False}
    return {"known": True, **{k: v for k, v in entry.items() if k not in ("started", "updated")},
            "elapsed_s": round(time.time() - entry["started"], 1)}


# Inside a study worker process: it computes and returns; saving to disk is
# the parent's (two workers writing one file would race).
_IN_WORKER = False
_MAX_STUDY_WORKERS = 4


def _study_workers(n: int) -> int:
    """How many worker processes a study of `n` independent moments uses: one
    per moment, at most _MAX_STUDY_WORKERS and one less than the cores; 1
    (in turn, in this process) where processes cannot be forked (Windows) or
    CONDUCTOR_STUDY_WORKERS says so."""
    import multiprocessing
    wanted = os.environ.get("CONDUCTOR_STUDY_WORKERS", "").strip()
    if "fork" not in multiprocessing.get_all_start_methods() or n <= 1:
        return 1
    cap = int(wanted) if wanted.isdigit() else _MAX_STUDY_WORKERS
    return max(1, min(n, cap, (os.cpu_count() or 2) - 1))


def _worker_start() -> None:
    global _IN_WORKER
    _IN_WORKER = True
    # One thread per solve in a worker whatever the parent was told: several
    # workers each using every core only fight over the cores.
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = "1"


def _run_coroutine(coro):
    """A coroutine that never suspends — the study endpoints compute, they do
    not wait — run to its result without an event loop (a forked worker has
    none of its own)."""
    try:
        coro.send(None)
    except StopIteration as done:
        return done.value
    coro.close()
    raise RuntimeError("the study suspended")


def _security_moment(request, ts: str) -> tuple[dict, dict]:
    """One moment of security_over_time — the same study a question at that
    moment gets — as (its row, the option sets it created)."""
    before = set(app_data.get("option_sets") or {})
    modes = ["preventive", "corrective"] if request.n1 == "both" else [request.n1]
    limits = {k: getattr(request, k) for k in ("vm_upper_pu", "vm_lower_pu", "max_line_loading_pct",
                                               "max_trafo_loading_pct")}
    opf = {k: v for k, v in (("opf_vm_lower", request.opf_vm_lower), ("opf_vm_upper", request.opf_vm_upper))
           if v is not None}
    study = FlexibilityRequest(timestamp=ts, data_source=request.data_source, network=request.network,
                               load_scaling_factor=request.load_scaling_factor, n1=request.n1,
                               hold_units=request.hold_units, branch_limits=request.branch_limits,
                               load_power_factor=request.load_power_factor, **opf, **limits)
    sweep_req = GridRequest(timestamp=ts, data_source=request.data_source, network=request.network,
                            load_scaling_factor=request.load_scaling_factor,
                            load_power_factor=request.load_power_factor, **limits)
    net_now, _ = _prepared_net(ts, sweep_req)
    schedule = _n1_sweep(net_now, ts, sweep_req.vm_lower_pu, sweep_req.vm_upper_pu,
                         sweep_req.max_line_loading_pct, sweep_req.max_trafo_loading_pct)
    row = {"timestamp": ts, "schedule_n1_outages_with_violations": schedule["total_outages_causing_violations"],
           "outages_tested": schedule["total_outages_tested"]}
    try:
        # The study itself, not its locked wrapper: this thread (or worker) already runs one.
        result = _run_coroutine(optimise_flexibility.__wrapped__(study))
    except HTTPException as exc:
        return {**row, "status": "error", "why": str(exc.detail)[:300]}, {}
    created = {sid: entry for sid, entry in (app_data.get("option_sets") or {}).items() if sid not in before}
    if result.get("status") != "options":
        return {**row, "status": result.get("status"), "why": (result.get("message") or "")[:300]}, created
    by_id = {o["id"]: o for o in result.get("options") or []}

    def brief(o):
        n1 = o.get("n1") or {}
        # The full N-1 screen (of the row's outages_tested) and, apart, the few
        # outages the security-constrained OPF studied and how many it secured.
        return {"ref": o.get("ref"), "cost_eur_per_h": o.get("cost_eur_per_h"),
                "n1_outages_with_violations": (n1.get("screen") or {}).get("outages_with_violations"),
                **({"outages_studied_by_the_opf": len(n1.get("studied") or []),
                    "of_those_secured": n1["secured"]} if "secured" in n1 else {}),
                **({"secured_with_load_reduction": n1["secured_with_load_reduction"]}
                   if n1.get("secured_with_load_reduction") else {})}
    # Whether the schedule itself meets every normal-operation limit (the
    # OPF's current margin included): where it does not, it is a reference,
    # not an option — security is compared with the cheapest action.
    least = by_id.get("least_disturbance") or {}
    row.update({"status": "options", "cheapest": brief(by_id["cheapest"]) if "cheapest" in by_id else None,
                "schedule_meets_limits": not least.get("n_units_called") and not least.get("moved_mw")})
    for mode in modes:
        option = by_id.get(f"n1_{mode}")
        if option is None:
            why = next((n["reason"] for n in result.get("not_offered") or [] if n["id"] == f"n1_{mode}"), None)
            row[mode] = {"not_offered": why or "not found"}
            continue
        row[mode] = brief(option)
        if row["cheapest"] is not None:
            row[mode]["security_cost_eur_per_h"] = round(option["cost_eur_per_h"] - row["cheapest"]["cost_eur_per_h"], 2)
    return row, created


def _moment_in_worker(moment, request, ts: str) -> tuple[str, dict, dict]:
    return (ts, *moment(request, ts))


def _adopt_option_sets(created: dict) -> dict:
    """Option sets a worker created, kept here under fresh ids (the worker's
    own ids were from its copy). Returns {its id: id here}."""
    sets = app_data.setdefault("option_sets", {})
    renamed = {}
    for old in sorted(created, key=lambda sid: int(sid[1:]) if sid[1:].isdigit() else 0):
        app_data["option_set_counter"] = app_data.get("option_set_counter", 0) + 1
        new = f"S{app_data['option_set_counter']}"
        sets[new] = {**created[old], "id": new}
        renamed[old] = new
    for old in list(sets)[:-_MAX_OPTION_SETS]:
        sets.pop(old)
    return renamed


def _renamed_refs(row: dict, renamed: dict) -> dict:
    """A row's option references ("S5:cheapest") under the ids kept here:
    every `ref` or `*_ref` string, however deep."""
    def walk(node):
        if isinstance(node, dict):
            for key, val in node.items():
                if (key == "ref" or key.endswith("_ref")) and isinstance(val, str) and ":" in val:
                    sid, oid = val.split(":", 1)
                    node[key] = f"{renamed.get(sid, sid)}:{oid}"
                else:
                    walk(val)
        elif isinstance(node, list):
            for val in node:
                walk(val)
    walk(row)
    return row


def _run_moments(request, ticks: list[str], job: str | None, moment=None) -> tuple[list[dict], int]:
    """Every moment's row — `moment(request, ts)`, (row, option sets it
    created); the cost of security by default — in order: in parallel worker
    processes when the platform allows (each a fork of this one: the network,
    data and settings as they are now), else in turn. A worker pool that fails
    falls back to running what is left in turn. Returns (rows, workers used)."""
    moment = moment or _security_moment
    workers = _study_workers(len(ticks))
    rows: dict[str, dict] = {}

    def note():
        last = max(rows, default=None)
        _report_progress(job, done=len(rows), total=len(ticks), last=last, workers=workers)

    note()
    if workers > 1:
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor
        from concurrent.futures.process import BrokenProcessPool
        try:
            with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("fork"),
                                     initializer=_worker_start) as pool:
                futures = [pool.submit(_moment_in_worker, moment, request, ts) for ts in ticks]
                for future in as_completed(futures):
                    ts, row, created = future.result()
                    rows[ts] = _renamed_refs(row, _adopt_option_sets(created))
                    note()
        except (BrokenProcessPool, OSError, RuntimeError):
            logging.getLogger(__name__).warning("Study workers failed; finishing in turn.", exc_info=True)
            workers = 1
        _save_option_sets()
    for ts in ticks:
        if ts not in rows:
            rows[ts], _created = moment(request, ts)
            note()
    return [rows[ts] for ts in ticks], workers


@app.post("/api/flexibility/security_over_time")
@_off_event_loop
async def security_over_time(request: SecurityOverTimeRequest):
    """The cost of N-1 security over a window: at each moment the same study
    a single question gets (the cheapest corrective action, and the N-1
    secure option — preventive and/or corrective — with every option's N-1
    screen), then hour by hour what security costs above the cheapest, and in
    total over the window (each moment weighted by half the time to its
    neighbours, see `_weights`). The moments run
    in parallel worker processes where the platform allows; `progress_id`
    lets the caller follow them (GET /api/progress/{id})."""
    ticks = _window(request)
    modes = ["preventive", "corrective"] if request.n1 == "both" else [request.n1]
    started = time.time()
    rows, workers = _run_moments(request, ticks, request.progress_id)
    spans = _weights(ticks)
    totals = {}
    for mode in modes:
        priced = [(r[mode]["security_cost_eur_per_h"], h) for r, h in zip(rows, spans)
                  if isinstance(r.get(mode), dict) and "security_cost_eur_per_h" in r[mode]]
        if priced:
            totals[mode] = {"security_cost_eur": round(sum(c * h for c, h in priced), 2),
                            "moments_priced": len(priced),
                            "highest_eur_per_h": max(c for c, _ in priced)}
    worst = max(rows, key=lambda r: r.get("schedule_n1_outages_with_violations") or 0)
    lines = [f"{len(rows)} moments from {ticks[0]} to {ticks[-1]}"]
    short = [r["timestamp"] for r in rows if r.get("schedule_meets_limits") is False]
    def span(counts):
        counts = [c for c in counts if isinstance(c, int)]
        return "" if not counts else f"{min(counts)}" if min(counts) == max(counts) else f"{min(counts)}–{max(counts)}"
    tested = next((r["outages_tested"] for r in rows if r.get("outages_tested")), None)
    for mode, t in totals.items():
        screened = span([(r.get(mode) or {}).get("n1_outages_with_violations") for r in rows])
        lines.append(f"N-1 {mode}: {t['security_cost_eur']:.2f} € above the cheapest over the window "
                     f"(at most {t['highest_eur_per_h']:.2f} €/h)"
                     + (f", and in the full N-1 screen {screened} of {tested} outages still with violations"
                        if screened and tested else ""))
    if tested:
        lines.append(f"the cheapest: {span([(r.get('cheapest') or {}).get('n1_outages_with_violations') for r in rows])}"
                     f" of {tested}; the schedule: {span([r.get('schedule_n1_outages_with_violations') for r in rows])}"
                     f" of {tested}")
    return {"timestamp": ticks[0], "mode": "security_over_time", "modes": modes, "moments": rows,
            "hours_each": [round(h, 3) for h in spans], "totals": totals,
            "worst_moment": worst["timestamp"],
            "message": "; ".join(lines) + ". Each moment is the same study as asking at that moment — secure only "
                       "for the outages its optimisation studied (outages_studied_by_the_opf, of_those_secured), so "
                       "report the full screen (n1_outages_with_violations of outages_tested) as the chart does; the total "
                       f"covers the {sum(spans):g} h from the first moment to the last, each moment weighted by half "
                       "the time to its neighbours — an estimate between the moments studied."
                       + (f" At {len(short)} of the {len(rows)} moments the schedule itself breaks a normal-operation "
                          "limit (the 90 % current margin included), so it is a reference, not an option: compare the "
                          "secure option with the cheapest corrective action, not with the schedule." if short else ""),
            "schedule_breaks_limits_at": short,
            "computed_in": {"seconds": round(time.time() - started, 1), "workers": workers},
            "data_source": _dataset_label(request)}


class PlanRequest(_WindowRequest):
    """Request body for ``POST /api/flexibility/plan``."""
    objective: Literal["least_disturbance", "cheapest"] = "least_disturbance"
    # How strongly the plan keeps an instruction from one moment to the next
    # (0: each moment on its own; 1: the default).
    steadiness: float = Field(default=1.0, ge=0, le=10)
    # Ramp limit between consecutive moments, % of a unit's capacity per hour (None: none).
    ramp_pct_per_h: float | None = Field(default=None, gt=0, le=1000)


_PLAN_CHANGE_MW = fo.MOVE_MW      # an instruction changes when it moves at all (as options count units moved)


@app.post("/api/flexibility/plan")
@_off_event_loop
async def plan_over_time(request: PlanRequest):
    """A corrective plan across a window (an evening peak): one OPF over every
    moment, each meeting every limit, linked so that an instruction is kept
    from one moment to the next unless changing it is worth it (`steadiness`),
    and — when given — within each unit's ramp. Compared with solving each
    moment on its own: how many active-power instructions each issues, the MW
    moved and, for the cheapest, the cost."""
    if _db_for("full", request) is None:
        raise HTTPException(status_code=503)
    from pyomo.environ import ConcreteModel, Constraint, NonNegativeReals, Objective, Var, minimize, value as pv
    ticks = _window(request)
    spans, gaps = _weights(ticks), _gaps(ticks)
    lo = request.opf_vm_lower if request.opf_vm_lower is not None else _network_limits()["opf_vm_lower"]
    hi = request.opf_vm_upper if request.opf_vm_upper is not None else _network_limits()["opf_vm_upper"]
    cap = float(app_data.get("slack_max_mw", 999.0))
    db = _db_for("full", request)
    blocks, nets, prices_at, ext0 = [], [], [], []
    held_any: dict = {}
    for ts in ticks:
        req = FlexibilityRequest(timestamp=ts, data_source=request.data_source, network=request.network,
                                 load_scaling_factor=request.load_scaling_factor, hold_units=request.hold_units,
                                 load_power_factor=request.load_power_factor)
        net_base, _ = _prepared_net(ts, GridRequest(timestamp=ts, data_source=request.data_source,
                                                    network=request.network, load_power_factor=request.load_power_factor))
        try:
            pp.runpp(net_base)
        except Exception:  # noqa: BLE001
            pass
        fixed, held = _held_at_schedule(net_base, req)
        held_any.update(held)
        caps, _shown = _branch_caps(net_base, request.branch_limits)
        net = copy.deepcopy(net_base)
        _scale_loads(net, request.load_scaling_factor)
        try:
            pp.runpp(net)
        except Exception:  # noqa: BLE001
            pass
        _res, m = fe.optimization_model_base(
            net, net_base, db["Yff_r"], db["Yff_i"], db["Yft_r"], db["Yft_i"], db["TAPS"], db["trafo_ranges"],
            db["trafo_defaults"], db["branch_to_trafo"], opf_vm_lower=lo, opf_vm_upper=hi,
            fixed_setpoints=fixed or None, branch_limits=caps or None)
        _cap_exchange(m, cap, None, fixed)
        scopf.release(m, set(held))
        blocks.append(m)
        nets.append(net)
        prices_at.append(_flex_prices(net, m)[0])
        ref = net if _finite_exchange(net) else net_base
        ext0.append(({e: float(ref.res_ext_grid["p_mw"].values[i]) for i, e in enumerate(m.Ext)},
                     {e: float(ref.res_ext_grid["q_mvar"].values[i]) for i, e in enumerate(m.Ext)}))
    units = sorted((g for g in blocks[0].G if all(g in set(b.G) for b in blocks)), key=_natural)

    def solve_window(link: bool):
        top = ConcreteModel()
        objective = 0.0
        copies = [b.clone() for b in blocks]
        for k, m in enumerate(copies):
            setattr(top, f"t{k}", m)
            for obj in list(m.component_objects(Objective, active=True)):
                obj.deactivate()
            if request.objective == "cheapest":
                objective += fo.add_cost(m, prices_at[k], *ext0[k]) + 1e-3 * fo._disturbance(m)
            else:
                objective += (0.01 * sum((m.Pg_new[g] - m.Pg_base[g]) ** 2 for g in m.G)
                              + 0.001 * sum((m.Qg_new[g] - m.Qg_base[g]) ** 2 for g in m.G))
        if link and len(copies) > 1:
            # Per MW of change: a 5 MW move at least-disturbance prices ~0.1 at the
            # margin, so the default holds an instruction unless changing it moves
            # a good deal more (cheapest: € per MW).
            weight = (6.0 if request.objective == "cheapest" else 0.15) * request.steadiness
            # Every change of an instruction counts — the first one too (from
            # the schedule), or the plan would issue at the first moment what
            # only a later one needs, to save a "change".
            pairs = [(k, g) for k in range(len(copies)) for g in units]
            top.change = Var(pairs, within=NonNegativeReals, initialize=0.0)

            def delta(k, g):
                if k < 0:
                    return 0.0
                m = copies[k]
                return m.Pg_new[g] - m.Pg_base[g]
            top.change_up = Constraint(pairs, rule=lambda t, k, g: t.change[k, g] >= delta(k, g) - delta(k - 1, g))
            top.change_down = Constraint(pairs, rule=lambda t, k, g: t.change[k, g] >= delta(k - 1, g) - delta(k, g))
            objective += weight * sum(top.change[p] for p in pairs)
            pairs = [(k, g) for k in range(1, len(copies)) for g in units]
            if request.ramp_pct_per_h:
                pmax = {g: float(pv(copies[0].Pg_max[g])) for g in units}
                ramp = {(k, g): request.ramp_pct_per_h / 100.0 * pmax[g] * gaps[k - 1] for k, g in pairs}
                top.ramp_up = Constraint(pairs, rule=lambda t, k, g: (
                    copies[k].Pg_new[g] - copies[k - 1].Pg_new[g] <= ramp[k, g]))
                top.ramp_down = Constraint(pairs, rule=lambda t, k, g: (
                    copies[k - 1].Pg_new[g] - copies[k].Pg_new[g] <= ramp[k, g]))
        top.obj = Objective(expr=objective, sense=minimize)
        if not fo.solve(top):
            return None
        return [fo.capture(m, "least_disturbance" if request.objective == "least_disturbance" else "cheapest")
                for m in copies]

    plan = solve_window(link=True)
    if plan is None:
        raise HTTPException(status_code=422, detail=(
            "No plan meets every limit at every moment of the window"
            + (" within the ramp limit" if request.ramp_pct_per_h else "")
            + ". Ask for the options at the worst moment, or relax the ramp."))
    separate = solve_window(link=False)

    def describe(options):
        steps, issued, moved, cost = [], 0, 0.0, 0.0
        previous = {g: 0.0 for g in units}
        for k, (o, ts) in enumerate(zip(options, ticks)):
            dp = {g: o.p[g] - o.p0[g] for g in units}
            changed = [g for g in units if abs(dp[g] - previous[g]) >= _PLAN_CHANGE_MW]
            issued += len(changed)
            moved += sum(abs(v) for v in dp.values()) * spans[k]
            c = fo.action_cost(o, prices_at[k], *ext0[k])["total"]
            cost += c * spans[k]
            volts = [v for v in o.v.values() if v is not None]
            steps.append({"timestamp": ts, "dp_mw": {g: round(dp[g], 2) + 0.0 for g in units},
                          "changed": changed, "cost_eur_per_h": c,
                          "max_loading_pct": round(o.max_loading_pct, 2) if o.max_loading_pct is not None else None,
                          "min_vm_pu": round(min(volts), 4) if volts else None,
                          "power_flow_check": _power_flow_check(nets[k], o, lo, hi).get("confirmed")})
            previous = dp
        return {"steps": steps, "instructions_issued": issued, "mwh_moved": round(moved, 2),
                "cost_eur": round(cost, 2)}
    planned = describe(plan)
    alone = describe(separate) if separate else None
    msg = (f"A plan over {len(ticks)} moments from {ticks[0]} to {ticks[-1]}: {planned['instructions_issued']} "
           f"active-power instruction(s)" + (f" against {alone['instructions_issued']} solving each moment on its "
                                             f"own" if alone else "")
           + f"; {planned['mwh_moved']:.1f} MWh moved, {planned['cost_eur']:.2f} € of corrective action at the cost "
             f"table's prices over the {sum(spans):g} h window. Every moment meets every limit.")
    return {"timestamp": ticks[0], "mode": "plan", "objective": request.objective, "units": units,
            "moments": ticks, "hours_each": [round(h, 3) for h in spans], "steadiness": request.steadiness,
            **({"ramp_pct_per_h": request.ramp_pct_per_h} if request.ramp_pct_per_h else {}),
            **({"held_units": held_any} if held_any else {}),
            "plan": planned, "each_moment_alone": alone, "message": msg,
            "data_source": _dataset_label(request)}


# ---------------------------------------------------------------------------
# Key performance indicators (kpis.py): the schedule against a corrective action
# ---------------------------------------------------------------------------

def _kpi_state(ts: str, req: GridRequest, applied: tuple | None = None) -> dict:
    """One state of a moment, by AC power flow — the schedule, or `applied`
    ((option set, option, held)) on it: its violations and margins to the
    limits, its N-1 screen, each unit's output, the supply and the import."""
    net, pmax = _prepared_net(ts, req, applied)
    vlo, vhi, mll, mtl = _effective_thresholds(req)
    sweep = _n1_sweep(copy.deepcopy(net), ts, vlo, vhi, mll, mtl)
    net_c, found = rs.real_time_security_assessment(net, ts, vm_lower=vlo, vm_upper=vhi,
                                                    max_line_loading_pct=mll, max_trafo_loading_pct=mtl)
    if net_c.res_bus.empty or net_c.res_bus["vm_pu"].isna().all():
        raise HTTPException(status_code=503, detail=f"The power flow did not converge at {ts}"
                                                    + (" with the option applied." if applied else "."))
    found = pd.DataFrame(found) if isinstance(found, list) else found
    records = found.to_dict(orient="records") if not found.empty else []

    def in_service(table):
        return net_c[table].index[net_c[table]["in_service"].astype(bool)] if len(net_c[table]) else []

    vm = net_c.res_bus.loc[in_service("bus"), "vm_pu"].dropna()
    lines = net_c.res_line.loc[in_service("line"), "loading_percent"].dropna()
    trafos = net_c.res_trafo.loc[in_service("trafo"), "loading_percent"].dropna() if len(net_c.trafo) else pd.Series()
    margins = [m for m in ((vm.min() - vlo) if len(vm) else None, (vhi - vm.max()) if len(vm) else None)
               if m is not None]
    headroom = [h for h in ((mll - lines.max()) if len(lines) else None, (mtl - trafos.max()) if len(trafos) else None)
                if h is not None]
    units = {gm.unit_name(net_c, t, i): (float(net_c[f"res_{t}"].at[i, "p_mw"]), float(net_c[f"res_{t}"].at[i, "q_mvar"]))
             for t in ("gen", "sgen") for i in in_service(t)}
    ext_p = [float(v) for v in net_c.res_ext_grid["p_mw"].values]
    ext_q = [float(v) for v in net_c.res_ext_grid["q_mvar"].values]
    return {
        "violations": len(records), "worst": _rsa_brief({"violations": records})["worst"],
        "min_vm_pu": round(float(vm.min()), 4) if len(vm) else None,
        "min_vm_bus": _bus_display_name(net_c, int(vm.idxmin())) if len(vm) else None,
        "max_vm_pu": round(float(vm.max()), 4) if len(vm) else None,
        "max_vm_bus": _bus_display_name(net_c, int(vm.idxmax())) if len(vm) else None,
        "vm_margin_pu": round(float(min(margins)), 4) + 0.0 if margins else None,       # never "-0.0"
        "max_line_loading_pct": round(float(lines.max()), 2) if len(lines) else None,
        "max_line": _line_display_name(net_c, int(lines.idxmax())) if len(lines) else None,
        **({"max_trafo_loading_pct": round(float(trafos.max()), 2),
            "max_trafo": _trafo_display_name(net_c, int(trafos.idxmax()))} if len(trafos) else {}),
        "loading_headroom_pct": round(float(min(headroom)), 2) + 0.0 if headroom else None,
        "n1_outages_with_violations": sweep["total_outages_causing_violations"],
        "n1_outages_tested": sweep["total_outages_tested"],
        "supply_mw": round(sum(p for p, _ in units.values()) + sum(ext_p), 3),
        "load_mw": round(float(net_c.res_load["p_mw"].sum()), 3) if len(net_c.load) else 0.0,
        "import_mw": round(sum(ext_p), 3),
        # For the indicators; not shown.
        "_p": {u: p for u, (p, _) in units.items()}, "_q": {u: q for u, (_, q) in units.items()},
        "_ext": (ext_p, ext_q), "_pmax": {**pmax, **(getattr(req, "pg_max_overrides", None) or {})},
    }


def _kpi_public(state: dict | None) -> dict | None:
    return None if state is None else {k: v for k, v in state.items() if not k.startswith("_")}


def _kpi_option_cost(option, held: bool, before: dict, after: dict, sink: dict) -> float | None:
    """An option's cost at this moment, at the cost table's prices: as the
    options study prices it at its own moment; held at a later one, the change
    from this moment's schedule its power flow shows."""
    if sink.get("model") is None:
        return None
    prices, _ = _flex_prices(sink["net"], sink["model"])
    if not held:
        return round(fo.action_cost(option, prices, sink["pext0"], sink["qext0"])["total"], 2)
    exts = list(sink["model"].Ext)
    (p0, q0), (p1, q1) = before["_ext"], after["_ext"]
    change = option._replace(p=dict(after["_p"]), p0=dict(before["_p"]), q=dict(after["_q"]), q0=dict(before["_q"]),
                             pext={e: p1[i] for i, e in enumerate(exts) if i < len(p1)},
                             qext={e: q1[i] for i, e in enumerate(exts) if i < len(q1)})
    return round(fo.action_cost(change, prices, {e: p0[i] for i, e in enumerate(exts) if i < len(p0)},
                                {e: q0[i] for i, e in enumerate(exts) if i < len(q0)})["total"], 2)


def _kpi_deliverable(model, state: dict | None) -> dict | None:
    """`fo.deliverable` from a state's dispatch, rounded for the result, with
    each unit's spare capacity beside its range (what it has, against what the
    network lets it deliver). A range the solver found no dispatch for: None."""
    if model is None or state is None:
        return None
    found = fo.deliverable(model, {g: state["_p"][g] for g in model.G if g in state["_p"]})
    if found is None:
        return None
    r = lambda v: round(v, 2) if v is not None else None  # noqa: E731
    pmax, p = state["_pmax"], state["_p"]
    return {"up_mw": r(found["up_mw"]), "down_mw": r(found["down_mw"]),
            "units": {g: {"up_mw": r((u or {}).get("up_mw")), "down_mw": r((u or {}).get("down_mw")),
                          "spare_mw": r(max(float(pmax[g]) - p[g], 0.0)) if pmax.get(g) is not None else None}
                      for g, u in sorted(found["units"].items(), key=lambda kv: _natural(kv[0]))}}


_KPI_FLEX_FIELDS = ("load_scaling_factor", "load_power_factor", "slack_max_mw", "slack_q_max_mvar", "opf_vm_upper",
                    "opf_vm_lower", "opf_lambda_p", "opf_lambda_q", "pg_max_overrides", "pg_min_overrides",
                    "fixed_setpoints", "opf_min_power_factor", "opf_current_safety_margin", "hold_units",
                    "branch_limits", "n1")


def _kpi_option(request, ts: str) -> tuple:
    """`request.option` for the indicators at `ts`: (its set, the option,
    held). Refused: an option computed after an outage — the schedule it would
    be compared with has every branch in service."""
    stored, option, _req, held = _option_context(request.model_copy(update={"timestamp": ts}), later=True)
    if stored.get("outage"):
        raise HTTPException(status_code=400, detail=(
            f"option: {stored['id']}:{option.id} was computed after the outage of {stored['outage']['element']}; "
            "the indicators compare it with the schedule, every branch in service. Screen it with "
            "simulate_all_contingencies or assess it with run_rsa instead."))
    return stored, option, held


def _kpi_moment(request, ts: str) -> tuple[dict, dict]:
    """The indicators of one moment (`kpis.py`), as (row, option sets created):
    the schedule and a corrective action — `request.option` (held from its own
    moment at a later one), else the cheapest at this moment — each checked by
    AC power flow; the flexibility required and deliverable solved on the
    options' own model. Never raises: a moment that cannot be studied says why."""
    before_sets = set(app_data.get("option_sets") or {})
    try:
        return _kpi_moment_unguarded(request, ts, before_sets)
    except HTTPException as exc:
        why = str(exc.detail)
    except Exception as exc:  # noqa: BLE001 — one moment's failure is reported, not the window's
        logging.getLogger(__name__).warning("KPI moment %s failed", ts, exc_info=True)
        why = f"{type(exc).__name__}: {exc}"
    created = {sid: e for sid, e in (app_data.get("option_sets") or {}).items() if sid not in before_sets}
    return {"timestamp": ts, "status": "error", "why": why[:300]}, created


def _kpi_moment_unguarded(request, ts: str, before_sets: set) -> tuple[dict, dict]:
    limits = {k: getattr(request, k) for k in ("vm_upper_pu", "vm_lower_pu", "max_line_loading_pct",
                                               "max_trafo_loading_pct")}
    grid_req = GridRequest(timestamp=ts, data_source=request.data_source, network=request.network,
                           load_scaling_factor=request.load_scaling_factor,
                           load_power_factor=request.load_power_factor, **limits)
    chosen = _kpi_option(request, ts) if request.option else None
    # The corrective options at this moment, and the model they were solved on.
    sink: dict = {}
    token = _FLEX_MODEL_SINK.set(sink)
    try:
        study = FlexibilityRequest(timestamp=ts, data_source=request.data_source, network=request.network,
                                   options=True, **{k: getattr(request, k) for k in _KPI_FLEX_FIELDS
                                                    if getattr(request, k, None) is not None})
        flex = _run_coroutine(optimise_flexibility.__wrapped__(study))
    except HTTPException as exc:
        flex = {"status": "error", "message": str(exc.detail)[:300]}
    finally:
        _FLEX_MODEL_SINK.reset(token)
    created = {sid: e for sid, e in (app_data.get("option_sets") or {}).items() if sid not in before_sets}
    status = flex.get("status")
    offered = flex.get("options") or []
    by_id = {o["id"]: o for o in offered}
    last_resort = flex.get("set") == "last_resort"
    least = by_id.get("least_disturbance") or {}
    needs_action = last_resort or bool(least.get("n_units_called") or least.get("moved_mw"))

    row = {"timestamp": ts, "status": "ok", "options_status": status}
    pick = None
    if chosen:
        stored, option, held = chosen
        applied = (stored, option, held)
        row.update(after_is="option", after_ref=f"{stored['id']}:{option.id}", after_label=option.label, held=held,
                   option=_option_echo(stored, option, ts, held),
                   **({"held_since": stored["timestamp"]} if held else {}))
    elif status == "no_action_needed" or (status == "options" and not needs_action):
        applied = None
        row.update(after_is="schedule", after_label="the schedule (no action needed)")
    elif status == "options" and offered:
        pick = by_id.get("cheapest") or min(offered, key=lambda o: o.get("cost_eur_per_h") or 0.0)
        stored, option = _resolve_option(pick["ref"])
        applied = (stored, option, False)
        row.update(after_is="last_resort" if last_resort else "cheapest", after_ref=pick["ref"],
                   after_label=pick["label"], held=False)
    else:
        applied = None
        row.update(after_is="none", why=flex.get("message") or f"the options study ended {status}")

    before = _kpi_state(ts, grid_req)
    after = before if row["after_is"] == "schedule" else _kpi_state(ts, grid_req, applied) if applied else None
    model = None if sink.get("last_resort") else sink.get("model")

    # Flexibility: what the moment requires, what the action activates, how far the units can go.
    required = (0.0 if status == "no_action_needed" else
                fo.least_moved_mw(model) if model is not None and status == "options" else None)
    moved = None
    if after is not None:
        moved = sum(abs(after["_p"][u] - p) for u, p in before["_p"].items()
                    if u in after["_p"] and abs(after["_p"][u] - p) >= kp.NOISE_MW)
        moved += max(before["load_mw"] - after["load_mw"], 0.0)       # loads the action reduces
    deliver_after = _kpi_deliverable(model, after)
    deliver_before = deliver_after if row["after_is"] == "schedule" else (
        _kpi_deliverable(model, before) if status == "no_action_needed" else None)
    k1 = {"schedule": kp.target_demand_flexibility(before["_p"], before["_pmax"], before["supply_mw"]),
          "after": kp.target_demand_flexibility(after["_p"], after["_pmax"], after["supply_mw"]) if after else None}
    k2 = kp.flexibility_utilisation(required, moved, violations=before["violations"])
    k3 = kp.prevented_violation_ratio(before["violations"], after["violations"] if after else None)

    # Costs: the action's, the cheapest at this moment, and N-1 security above it.
    cheapest = (by_id["cheapest"]["cost_eur_per_h"] if "cheapest" in by_id else
                0.0 if status == "no_action_needed" else None)
    cost = (0.0 if row["after_is"] == "schedule" else pick["cost_eur_per_h"] if pick else
            _kpi_option_cost(chosen[1], chosen[2], before, after, sink) if chosen and after else None)
    n1_costs = {}
    for mode in (("preventive", "corrective") if request.n1 == "both" else (request.n1,) if request.n1 else ()):
        secure = by_id.get(f"n1_{mode}")
        if secure is not None and cheapest is not None:
            n1_costs[mode] = round(secure["cost_eur_per_h"] - cheapest, 2)
        elif secure is None:
            why = next((n["reason"] for n in flex.get("not_offered") or [] if n.get("id") == f"n1_{mode}"), None)
            n1_costs[mode] = None
            row.setdefault("n1_not_offered", {})[mode] = why or "not found"
    a = after or {}
    values = {
        "violations": (before["violations"], a.get("violations")),
        "vm_margin_pu": (before["vm_margin_pu"], a.get("vm_margin_pu")),
        "loading_headroom_pct": (before["loading_headroom_pct"], a.get("loading_headroom_pct")),
        "n1_outages_with_violations": (before["n1_outages_with_violations"], a.get("n1_outages_with_violations")),
        "kpi_3_prevented_violation_ratio_pct": (None, k3["pct"]),
        "kpi_1_target_demand_flex_pct": (k1["schedule"]["pct"], (k1["after"] or {}).get("pct")),
        "spare_up_mw": (k1["schedule"]["spare_mw"], (k1["after"] or {}).get("spare_mw")),
        "deliverable_up_mw": ((deliver_before or {}).get("up_mw"), (deliver_after or {}).get("up_mw")),
        "deliverable_down_mw": ((deliver_before or {}).get("down_mw"), (deliver_after or {}).get("down_mw")),
        "required_mw": (None, round(required, 3) if required is not None else None),
        "moved_mw": (0.0, round(moved, 3) if moved is not None else None),
        "kpi_2_flex_utilization_pct": (None, k2["pct"]),
        "cost_eur_per_h": (0.0, cost),
    }
    if request.n1:
        known = [c for c in n1_costs.values() if c is not None]
        values["n1_security_eur_per_h"] = (None, min(known) if known else None)
    row.update(
        schedule=_kpi_public(before), after=_kpi_public(after) if after is not before else _kpi_public(before),
        indicators=kp.indicator_rows(values),
        kpi_details={"kpi_1": k1, "kpi_2": k2, "kpi_3": k3},
        deliverable={"with_action": deliver_after, "schedule": deliver_before,
                     **({"note": "the units the operator holds are not among them"} if request.hold_units else {})},
        costs={"action_eur_per_h": cost, "cheapest_eur_per_h": cheapest,
               **({"n1_security_above_cheapest_eur_per_h": n1_costs} if request.n1 else {})},
    )
    return row, created


@app.post("/api/kpi/evaluate")
@_off_event_loop
async def evaluate_kpis(request: KPIRequest):
    """The key performance indicators at one moment (`kpis.py`): the schedule
    against a corrective action — the cheapest at that moment, or `option`
    (held from its own moment when applied later) — each checked by AC power
    flow. The three of the project notebook (target demand flexibility,
    flexibility utilisation, prevented violation ratio), N-0 and N-1 security
    margins, the flexibility the units can deliver within every limit, the
    flexibility required, and costs (`n1`: of N-1 security too)."""
    if _db_for("full", request) is None:
        raise HTTPException(status_code=503, detail="Admittance database not loaded.")
    ts = _resolve_tick(request.data_source, request.timestamp)
    if request.option:                  # refused with the reason before anything runs (another moment, network…)
        _kpi_option(request, ts)
    started = time.time()
    row, _created = _kpi_moment(request, ts)
    if row.get("status") == "error":
        raise HTTPException(status_code=503, detail=f"No indicators at {ts}: {row.get('why')}")
    return {**row, "mode": "kpis", "message": kp.moment_message(row), "metrics": kp.flat_metrics(row["indicators"]),
            "computed_in_s": round(time.time() - started, 1), "data_source": _dataset_label(request)}


@app.post("/api/kpi/over_time")
@_off_event_loop
async def kpis_over_time(request: KPIOverTimeRequest):
    """The indicators at each moment of a window, as `/api/kpi/evaluate` gives
    them, and over the window: each indicator's range and worst moment, the
    cost of the actions and the MWh moved and required (each moment weighted by
    half the time to its neighbours). `option` is held from its moment. The
    moments run in parallel where the platform allows; `progress_id` lets the
    caller follow them (GET /api/progress/{id})."""
    if _db_for("full", request) is None:
        raise HTTPException(status_code=503, detail="Admittance database not loaded.")
    ticks = _window(request)
    if request.option:                               # an option from after the window's start: refused now
        _kpi_option(request, ticks[0])
    started = time.time()
    rows, workers = _run_moments(request, ticks, request.progress_id, _kpi_moment)
    weights = _weights(ticks)
    summary = kp.window_summary(rows, weights)
    return {"timestamp": ticks[0], "mode": "kpis_over_time", **kp.compact_moments(rows),
            "hours_each": [round(h, 3) for h in weights], "summary": summary,
            "message": kp.window_message(ticks, summary),
            **({"option": request.option} if request.option else {}), **({"n1": request.n1} if request.n1 else {}),
            "computed_in": {"seconds": round(time.time() - started, 1), "workers": workers},
            "data_source": _dataset_label(request)}

EDDK_VALUES_URL = "https://admin.energydata.dk/api/v1/datastreams/values"


# -------------------------------------------------------------------
# Transformer datastream IDs this endpoint fetches.
#
# A token is licensed for a specific set of datastreams, and an id outside
# that set returns 403 rather than an empty result — an earlier, larger set
# stopped working for exactly that reason. Only a few are fetched, which keeps
# the resulting chart readable; extend the list once the map documents
# metadata for the additional datastreams.
#
# Read from the datastream map rather than written here: real ids identify
# real measurement points on a real utility's network. See backend/README.md,
# "The datastream map".
# -------------------------------------------------------------------
EDDK_PUBLIC_IDS = eddk_pipe.get_public_ids()


def _resolve_eddk_token() -> tuple[str | None, str]:
    """Resolve the EDDK token from any supported source.

    Lookup order (deliberately *config.json first*):
      1. ``config.json`` at the project root or next to the Backend
         folder, key ``EDDK_token`` .
      2. ``EDDK_API_TOKEN`` environment variable (``.env``) — fallback
         for users who manage their own token.

    Returns
    tuple[str | None, str]
        ``(token, source)``. ``source`` is ``"config.json"``,
        ``".env"``, or ``"none"`` for diagnostics. ``token`` is the
        first non-empty value encountered, or ``None``.

    Looked up at request time, not import time, so the user can swap
    files without restarting uvicorn.
    """
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    cfg_candidates = [
        os.path.abspath(os.path.join(backend_dir, "..", "config.json")),
        os.path.join(backend_dir, "config.json"),
    ]
    import json as _json
    for path in cfg_candidates:
        if os.path.isfile(path):
            try:
                with open(path, "r") as f:
                    cfg = _json.load(f)
                tok = cfg.get("EDDK_token") or cfg.get("EDDK_API_TOKEN")
                if tok:
                    return tok, "config.json"
            except Exception:
                continue

    env_token = os.getenv("EDDK_API_TOKEN")
    if env_token:
        return env_token, ".env"

    return None, "none"


def _eddk_preflight(token: str, probe_ids: list[int], start: str, end: str) -> dict | None:
    """Tiny diagnostic probe so we can surface real EDDK errors.

    ``pipeline_functions.fetch_timespan_values`` silently returns ``[]``
    on any HTTP failure, which leaves the UI showing a blank chart with
    no hint as to *why*. This helper sends the same request shape with
    a minimal ID batch and inspects the HTTP status ourselves.

    Returns
    -------
    None
        If the probe succeeded (HTTP 200). Caller should proceed with
        the real fetch.
    dict
        ``{"status": "error", "message": ..., "http_status": ...}`` if the
        probe failed. Caller should short-circuit and return this to the UI.
    """
    try:
        r = requests.get(
            EDDK_VALUES_URL,
            headers={"Accept": "application/json", "Authorization": f"Bearer {token}"},
            params={"ids": ",".join(map(str, probe_ids[:3])), "from": start, "to": end},
            timeout=15,
        )
    except requests.exceptions.Timeout:
        return {"status": "error", "message": "EDDK timed out after 15s.", "http_status": None}
    except Exception as e:
        return {"status": "error", "message": f"EDDK network error: {e}", "http_status": None}

    if r.status_code == 200:
        return None  # preflight OK, proceed with real fetch

    # Try to surface the server's error payload so the user sees
    # "Token not authorized to view listed datastream IDs" etc.
    detail = r.text.strip()[:300] if r.text else f"HTTP {r.status_code}"
    if r.status_code == 401:
        msg = ("EDDK rejected the token (401 Unauthorized). Check EDDK_API_TOKEN in "
               ".env or EDDK_token in config.json — the token may be expired or wrong.")
    elif r.status_code == 403:
        msg = (
            "EDDK 403 — token authenticated but lacks license for these "
            "the configured transformer datastreams."
            f"Server said: {detail}"
        )
    elif r.status_code == 404:
        msg = f"EDDK 404 — the datastream IDs do not exist. Server said: {detail}"
    else:
        msg = f"EDDK HTTP {r.status_code}. Server said: {detail}"

    return {"status": "error", "message": msg, "http_status": r.status_code}


@app.post("/api/eddk/fetch")
@_off_event_loop
async def fetch_eddk_data(request: FetchDataRequest):
    """Pull transformer time-series from the EDDK data space.

    Pipeline
    --------
    1. Resolve the API token via :func:`_resolve_eddk_token` —
       ``config.json`` first, ``.env`` as fallback.
    2. Build ISO timespan via ``pipeline_functions.define_timespan``.
    3. **Preflight probe**: 3-ID HTTP call so we can surface the real
       EDDK status (401 / 403 / etc.) instead of a silent empty chart
       — necessary because the engine helper swallows all HTTP failures
       into ``[]``.
    4. Real fetch via
    ``pipeline_functions.fetch_timespan_values_transformer_p_c``.
       Returns ``{datastream_id, timestamp, value, substation, parameter}``
       per row, decorated with metadata from the engine.
    5. **Post-process labelling**: when the engine's metadata lookup
       comes back ``"Unknown"`` (it does not yet carry entries for every
       fetched datastream), substitute ``f"ID {datastream_id}"`` so the
       frontend's ``groupby('substation')`` gives one line per ID
       rather than collapsing everything onto a single "Unknown" line.

    Response shape
    --------------
    Success: ``{"status": "success", "data": [...], "total_ids": N,
    "token_source": "config.json" | ".env"}``.
    Failure: ``{"status": "error", "message": "...", "http_status": 403}``
    (HTTP 200 with error body so the frontend can render a clean alert
    rather than a generic 500 blob).
    """
    token, token_source = _resolve_eddk_token()
    if not token:
        raise HTTPException(
            status_code=500,
            detail=("EDDK token missing. Drop a config.json with key "
                    "'EDDK_token' at the project root, or set EDDK_API_TOKEN "
                    "in .env."),
        )

    try:
        start, end = eddk_pipe.define_timespan(request.start_date, request.end_date)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Bad date format (expected YYYY-MM-DD): {e}")

    ids = list(EDDK_PUBLIC_IDS)

    preflight = _eddk_preflight(token, ids, start, end)
    if preflight is not None:
        preflight["token_source"] = token_source
        return preflight

    try:
        data = eddk_pipe.fetch_timespan_values_transformer_p_c(token, ids, start, end)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"EDDK fetch crashed: {e}")

    if not data:
        return {
            "status": "error",
            "message": ("Preflight passed but EDDK returned zero rows. "
                        "Verify the date range overlaps the dataset's coverage "
                        "(the new datastreams have data from 2025-02-01 onwards)."),
            "http_status": 200,
            "token_source": token_source,
        }

    # Replace engine's "Unknown" label with the datastream ID or a known substation name.
    id_to_substation = eddk_pipe.get_public_id_labels()
    for row in data:
        ds_id = row.get("datastream_id")
        if ds_id in id_to_substation:
            row["substation"] = id_to_substation[ds_id]
        elif row.get("substation") in (None, "Unknown", ""):
            row["substation"] = f"ID {ds_id}"

    return {
        "status": "success",
        "data": data,
        "total_ids": len(ids),
        "token_source": token_source,
    }


@app.post("/api/eddk/push")
@_off_event_loop
async def push_eddk_data():
    """Placeholder for publishing simulation results back to EDDK.

    The real MQTT / REST upload is not wired up. It used to answer "successfully
    pushed" unconditionally — a report of something that did not happen.
    """
    return {"status": "not_implemented",
            "message": "Publishing to the EDDK Data Space is not implemented yet; nothing was sent."}


class _NoOption(BaseModel):
    """For a study that does not apply a corrective option: refused, not ignored."""

    @field_validator("option", check_fields=False)
    @classmethod
    def _no_option(cls, value):
        if value:
            raise ValueError("option is not applied by this study; run it on the schedule, or use the N-1 "
                             "screen, the assessment or a single outage with the option.")
        return value


class AttributionRequest(_NoOption, GridRequest):
    """Request body for ``POST /api/rsa/attribution``.

    Attributes
    top_n_sources : int
        Cap on how many injections are perturbed, largest first. Each source
        costs two power flows, so this bounds runtime on large networks.
    top_n_drivers : int
        How many ranked drivers to return per violated element.
    delta_mw / delta_mvar : float
        Finite-difference step sizes. Small enough that the linearisation
        holds, large enough to stay clear of the power flow's tolerance.
    """
    top_n_sources: int = 20
    top_n_drivers: int = 5
    delta_mw: float = 0.1
    delta_mvar: float = 0.1


@app.post("/api/rsa/attribution")
@_off_event_loop
async def violation_attribution(request: AttributionRequest):
    """Rank the injections driving each violated element at one operating point.

    The security engines report *which* elements violate. This reports *what is
    driving them*, by measuring the sensitivity of each violated quantity to each
    injection on the same validated power flow — perturb, re-solve, measure.

    Returns per-violation driver rankings with the movement each source would
    need to clear the violation. Percentage contribution shares are deliberately
    not reported: they require choosing a baseline, and different baselines give
    different answers.

    ``app_data["current_index"]`` is **never mutated** by this endpoint.
    """
    import attribution_engine as attr

    (request.vm_lower_pu, request.vm_upper_pu,
     request.max_line_loading_pct, request.max_trafo_loading_pct) = _effective_thresholds(request)
    ts = _resolve_tick(request.data_source, request.timestamp)

    meas = prepare_measurement_df(_lookup_measurement_df(ts, request.data_source))
    net_copy = copy.deepcopy(_net_for(request))
    net_copy, _ = _assign_tick(net_copy, meas, ts, _dataset_of(request))
    _scale_loads(net_copy, request.load_scaling_factor)
    if request.sgen_scaling_factor != 1.0:
        net_copy.sgen["p_mw"] *= request.sgen_scaling_factor
        net_copy.sgen["q_mvar"] *= request.sgen_scaling_factor

    result = attr.attribute_violations(
        net_copy,
        timestamp=str(ts),
        vm_lower=request.vm_lower_pu,
        vm_upper=request.vm_upper_pu,
        max_line_loading_pct=request.max_line_loading_pct,
        max_trafo_loading_pct=request.max_trafo_loading_pct,
        delta_mw=request.delta_mw,
        delta_mvar=request.delta_mvar,
        top_n_sources=request.top_n_sources,
        top_n_drivers=request.top_n_drivers,
    )
    result["data_source"] = _dataset_label(request)
    result["thresholds_used"] = {
        "vm_lower_pu": request.vm_lower_pu,
        "vm_upper_pu": request.vm_upper_pu,
        "max_line_loading_pct": request.max_line_loading_pct,
        "max_trafo_loading_pct": request.max_trafo_loading_pct,
    }
    return result


# ---------------------------------------------------------------------------
# Network proposals (sandbox)
#
# A proposal is the base network plus an ordered list of edits (`sandbox.py`).
# These endpoints create, edit, list and discard proposals and compare one
# with the base. Every analysis endpoint studies a proposal when its request
# names it in `network`; the base itself is never modified.
# ---------------------------------------------------------------------------

class ProposalEditRequest(BaseModel):
    """Create a proposal or append edits to it.

    proposal : str
        Its id — letters, digits, `_` and `-`, e.g. "dc_bus9". Created when
        it does not exist yet.
    edits : list[dict]
        Edits in order, each with an `op` (see `sandbox.OPS`).
    replace : bool
        True replaces the proposal's edits instead of appending to them.
    """
    proposal: str
    edits: list[dict] = []
    replace: bool = False


class CompareNetworksRequest(_NoOption, GridRequest):
    """Run one study on the base and on a proposal at the same operating point.

    proposal : str
        The proposal compared with the base.
    study : str
        "rsa" (security assessment), "n1" (all single outages), "opf" (the
        deterministic flexibility OPF: can a corrective dispatch restore
        security, and what does it take) or "scan" (the security assessment
        at every tick of a window, on both networks).
    start_timestamp / end_timestamp : str | None
        scan: the window (prefix match); the whole dataset by default.
    step_size : int | None
        scan: every Nth tick. By default every tick, thinned only when the
        window holds more than `_SCAN_MAX_TICKS`, and the result says so.
    against : str | None
        Compare with this proposal instead of the base ("is B better than A?").
        In the result, "base" is then that proposal; `compared_with` says so.
    proposals : list[str] | None
        Several proposals, each compared with the base (or `against`) in the
        same study at the same operating point, plus a ranking table. Takes
        the place of `proposal`.
    """
    proposal: str = ""
    study: str = "rsa"
    against: str | None = None
    proposals: list[str] | None = None
    start_timestamp: str | None = None
    end_timestamp: str | None = None
    step_size: int | None = None


# A comparison over time runs the full assessment twice per tick (~20 ms each
# on the 14-bus case); above this many ticks the window is thinned and the
# result says so, rather than running for minutes on a year of data.
_SCAN_MAX_TICKS = 700


def _proposal_view(name: str) -> dict:
    p = _proposals()[name]
    return {"id": name, "edits": p["edits"], "edits_hash": p["hash"],
            "summary": p["summary"], "created": p["created"], "updated": p["updated"]}


def _check_profiles_now(net, name: str) -> list[str]:
    """Run every added element's profile at the current measurement tick and
    the first forecast tick, so a profile that cannot be followed (a load
    following a bus with no measured demand, a pv unit on a dataset without
    production) is refused when it is created, not at the first study."""
    checked = []
    for source in ("measurements", "forecasts"):
        try:
            ts = _resolve_tick(source, None)
        except HTTPException:
            continue  # that dataset is not loaded
        meas = prepare_measurement_df(_lookup_measurement_df(ts, source))
        _assign_tick(copy.deepcopy(net), meas, ts, _resolve_dataset(source)[1])
        checked.append(f"{source} at {ts}")
    return checked


@app.get("/api/proposals")
async def list_proposals():
    """Every proposal, oldest first."""
    return {"proposals": [_proposal_view(n) for n in list(_proposals())]}


@app.post("/api/proposals/edit")
@_off_event_loop
async def edit_proposal(request: ProposalEditRequest):
    """Create a proposal or add edits to it; nothing is stored unless the
    whole edit list applies to the base and every profile can be followed."""
    name = request.proposal.strip()
    if not _PROPOSAL_ID.match(name) or name == BASE_NETWORK:
        raise HTTPException(status_code=400, detail=(
            f"Invalid proposal id {request.proposal!r}: use up to 48 letters, digits, "
            "'_' or '-', starting with a letter or digit (and not 'base')."))
    base = app_data.get("net")
    if base is None:
        raise HTTPException(status_code=503, detail="No network loaded yet.")
    existing = _proposals().get(name)
    previous = [] if (existing is None or request.replace) else list(existing["edits"])
    incoming = _include_proposals(name, list(request.edits))
    if not incoming and not previous:
        raise HTTPException(status_code=400, detail="A proposal needs at least one edit.")
    try:
        net, canonical = sb.materialise(base, sb.upgrade_stored(previous) + incoming)
    except sb.EditQuestion as exc:
        # Not an error to report: a decision to put to the operator.
        raise HTTPException(status_code=400, detail={
            "needs_operator": True, "edit": (exc.edit - len(previous)) if exc.edit else None,
            "question": exc.question.split(": ", 1)[-1], "options": exc.options,
            "instruction": "Ask the operator this question with these options; do not choose for them. "
                           "Build the edit once they answer."})
    except sb.EditError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    checked = _check_profiles_now(net, name)
    assumed = _assumed_defaults(net, incoming, canonical[len(previous):])

    now = time.strftime("%Y-%m-%d %H:%M:%S")
    _proposals()[name] = {
        "id": name, "edits": canonical, "hash": sb.edits_hash(canonical),
        "summary": sb.summary(net, canonical),
        "created": existing["created"] if existing else now, "updated": now,
    }
    app_data.setdefault("_proposal_cache", {}).pop(name, None)
    _save_proposals()
    return {**_proposal_view(name), "status": "created" if existing is None else "updated",
            "profiles_checked": checked,
            **({"assumed": assumed} if assumed else {})}


def _include_proposals(name: str, edits: list) -> list:
    """Expand each `include_proposal` edit into that proposal's stored edits,
    in place — "A + B" is a proposal of A's edits followed by B's. The copies
    are independent: a later change to A does not change this proposal."""
    out = []
    for raw in edits:
        if not (isinstance(raw, dict) and raw.get("op") == "include_proposal"):
            out.append(raw)
            continue
        other = str(raw.get("proposal") or "").strip()
        if other == name:
            raise HTTPException(status_code=400, detail="include_proposal: a proposal cannot include itself.")
        if other not in _proposals():
            raise HTTPException(status_code=400, detail=(
                f"include_proposal: no proposal {other!r}. Proposals: {sorted(_proposals()) or 'none'}."))
        out.extend(sb.upgrade_stored(copy.deepcopy(_proposals()[other]["edits"])))
    return out


def _assumed_defaults(net, raw_edits: list, canonical: list) -> list[dict]:
    """Every value the edit tool filled in that the request did not give —
    to be stated to the operator, who can change any of them."""
    out = []
    for n, (raw, edit) in enumerate(zip(raw_edits, canonical), start=1):
        given = set(raw) if isinstance(raw, dict) else set()
        filled = {k: v for k, v in edit.items() if k not in given and k != "op"}
        if filled:
            out.append({"edit": n, "description": sb.describe(net, edit), "defaults": filled})
    return out


@app.delete("/api/proposals/{name}")
async def discard_proposal(name: str):
    """Discard a proposal. The base is unaffected."""
    if name not in _proposals():
        raise HTTPException(status_code=404, detail=f"No proposal {name!r}.")
    del _proposals()[name]
    app_data.setdefault("_proposal_cache", {}).pop(name, None)
    _save_proposals()
    return {"status": "discarded", "id": name, "remaining": list(_proposals())}


@app.delete("/api/proposals/{name}/edits/{position}")
@_off_event_loop
async def remove_proposal_edit(name: str, position: int):
    """Remove one edit (1-based). The rest must still apply; the last edit's
    removal discards the proposal."""
    if name not in _proposals():
        raise HTTPException(status_code=404, detail=f"No proposal {name!r}.")
    edits = list(_proposals()[name]["edits"])
    if not 1 <= position <= len(edits):
        raise HTTPException(status_code=400, detail=f"{name!r} has edits 1–{len(edits)}.")
    del edits[position - 1]
    if not edits:
        return await discard_proposal(name)
    return await edit_proposal(ProposalEditRequest(proposal=name, edits=edits, replace=True))


def _rsa_digest(r: dict) -> dict:
    volts = [v["vm_pu"] for v in r.get("all_voltages", []) if v.get("vm_pu") is not None]
    lines = [x["loading_percent"] for x in r.get("all_line_loading", []) if x.get("loading_percent") is not None]
    trafos = [x["loading_percent"] for x in r.get("all_trafo_loading", []) if x.get("loading_percent") is not None]
    return {
        "secure": r.get("secure"), "total_violations": r.get("total_violations"),
        "min_vm_pu": min(volts) if volts else None, "max_vm_pu": max(volts) if volts else None,
        "max_line_loading_pct": max(lines) if lines else None,
        "max_trafo_loading_pct": max(trafos) if trafos else None,
        "total_load_mw": r.get("total_load_mw"), "slack_import_mw": r.get("slack_import_mw"),
    }


def _violation_keys(rows: list[dict], with_cause: bool = False) -> set:
    def name(v):
        # The RSA names the element in `element`, the N-1 sweep in
        # `element_name`; a missing name used to come out as "None".
        for key in ("element", "element_name", "bus_name", "line_name", "trafo_name"):
            if v.get(key) not in (None, ""):
                return str(v[key])
        index = v.get("element_index", v.get("violation_element_index"))
        return f"{v.get('violation_type', 'element')} {index}"
    return {((v.get("outage_cause"),) if with_cause else ()) + (v.get("violation_type"), name(v))
            for v in rows}


def _largest_changes(base_rows, prop_rows, key, value, n=5):
    b = {x[key]: x[value] for x in base_rows if x.get(value) is not None}
    p = {x[key]: x[value] for x in prop_rows if x.get(value) is not None}
    deltas = [(k, b[k], p[k], p[k] - b[k]) for k in b.keys() & p.keys()]
    deltas.sort(key=lambda t: abs(t[3]), reverse=True)
    return [{"element": k, "base": round(vb, 4), "proposal": round(vp, 4), "change": round(d, 4)}
            for k, vb, vp, d in deltas[:n] if abs(d) > 1e-6]


@app.post("/api/proposals/compare")
@_off_event_loop
async def compare_networks(request: CompareNetworksRequest):
    """The same study on the base and on a proposal, at the same operating
    point (one timestamp, one dataset, the same limits and stress factors),
    and the difference. Two separate calls could land on two different ticks;
    one call cannot."""
    study = (request.study or "rsa").strip().lower()
    if study not in ("rsa", "n1", "opf", "scan"):
        raise HTTPException(status_code=400, detail="study must be 'rsa', 'n1', 'opf' or 'scan'.")
    ref = _reference(request)
    if ref != BASE_NETWORK and ref not in _proposals():
        raise HTTPException(status_code=400, detail=f"against: no proposal {ref!r}. Existing: {sorted(_proposals())}.")
    if request.proposals:
        return await _compare_several(request, study)
    name = (request.proposal or "").strip()
    if name in ("", BASE_NETWORK):
        raise HTTPException(status_code=400, detail="Name a proposal to compare with the base.")
    if name not in _proposals():
        raise HTTPException(status_code=400, detail=f"No proposal {name!r}. Existing: {sorted(_proposals())}.")
    if name == ref:
        raise HTTPException(status_code=400, detail=f"{name!r} cannot be compared with itself.")
    if study == "scan":
        return _compare_over_time(request, name)
    if study == "opf":
        return await _compare_opf(request, name)
    ts = _resolve_tick(request.data_source, request.timestamp)
    runner = run_security_assessment if study == "rsa" else simulate_all_contingencies

    results = {}
    for label, network in (("base", _reference(request)), ("proposal", name)):
        req = GridRequest(**request.model_dump(exclude={"proposal", "study", "network", "against",
                                                        "proposals", "timestamp", "data_source"}),
                          network=network, timestamp=ts, data_source=request.data_source)
        _tick_report.first = None
        results[label] = await runner(req)
        results[label + "_added"] = getattr(_tick_report, "first", None)
    base, prop = results["base"], results["proposal"]

    out = {"study": study, "timestamp": ts, "data_source": _dataset_label(request),
           "thresholds_used": _thresholds_used(request),
           "load_scaling_factor": request.load_scaling_factor,
           "network": {**_proposal_card(name), "compared_with": _reference(request)}}
    if results["proposal_added"]:
        out["network"]["added_elements_at_tick"] = results["proposal_added"]
    if study == "rsa":
        kb, kp = _violation_keys(base.get("violations", [])), _violation_keys(prop.get("violations", []))
        digest_b, digest_p = _rsa_digest(base), _rsa_digest(prop)
        out.update({
            "base": digest_b, "proposal": digest_p,
            # Stated, so the answer quotes them rather than subtracting two
            # rounded figures (16.96 written over a computed 16.97).
            "changes": {k: round(digest_p[k] - digest_b[k], 4)
                        for k in ("total_load_mw", "slack_import_mw", "min_vm_pu", "max_vm_pu",
                                  "max_line_loading_pct", "max_trafo_loading_pct")
                        if isinstance(digest_b.get(k), (int, float)) and isinstance(digest_p.get(k), (int, float))},
            "new_violations": [{"type": t, "element": e} for t, e in sorted(kp - kb)],
            "resolved_violations": [{"type": t, "element": e} for t, e in sorted(kb - kp)],
            "largest_voltage_changes": _largest_changes(
                base.get("all_voltages", []), prop.get("all_voltages", []), "bus_name", "vm_pu"),
            "largest_line_loading_changes": _largest_changes(
                base.get("all_line_loading", []), prop.get("all_line_loading", []),
                "line_name", "loading_percent"),
            "largest_trafo_loading_changes": _largest_changes(
                base.get("all_trafo_loading", []), prop.get("all_trafo_loading", []),
                "trafo_name", "loading_percent"),
            # Buses the proposal built have no base value to subtract; their
            # voltage is reported as it is.
            "added_buses": [
                {"bus": v["bus_name"], "vm_pu": v["vm_pu"]}
                for v in prop.get("all_voltages", [])
                if v["bus_name"] not in {b["bus_name"] for b in base.get("all_voltages", [])}],
        })
    else:
        keys = ("system_n1_secure", "total_outages_tested", "total_outages_causing_violations",
                "total_violations", "n_non_converged_outages", "non_converged_outages")
        kb = _violation_keys(base.get("violations", []), with_cause=True)
        kp = _violation_keys(prop.get("violations", []), with_cause=True)
        out.update({
            "base": {k: base.get(k) for k in keys}, "proposal": {k: prop.get(k) for k in keys},
            "new_violations": [{"outage": c, "type": t, "element": e} for c, t, e in sorted(kp - kb, key=str)],
            "resolved_violations": [{"outage": c, "type": t, "element": e} for c, t, e in sorted(kb - kp, key=str)],
        })
    return out


def _reference(request: CompareNetworksRequest) -> str:
    """The network a proposal is compared with: the base, or `against`."""
    return (request.against or "").strip() or BASE_NETWORK


_COMPARE_MAX_PROPOSALS = 6
# Per study, the figures a ranking of several proposals puts side by side.
_RANK_FIELDS = {
    "rsa": ("total_load_mw", "slack_import_mw", "min_vm_pu", "max_vm_pu", "max_line_loading_pct",
            "max_trafo_loading_pct", "secure", "total_violations"),
    "n1": ("system_n1_secure", "total_outages_causing_violations", "total_violations",
           "n_non_converged_outages"),
}


async def _compare_several(request: CompareNetworksRequest, study: str) -> dict:
    """Each proposal against the same reference, one study, one operating
    point; the full comparison of each plus a table to rank them by."""
    names = list(dict.fromkeys(n.strip() for n in request.proposals if n and n.strip()))
    unknown = [n for n in names if n not in _proposals()]
    if unknown:
        raise HTTPException(status_code=400, detail=f"No proposal {unknown}. Existing: {sorted(_proposals())}.")
    names = [n for n in names if n != _reference(request)]
    if not names:
        raise HTTPException(status_code=400, detail="Name at least one proposal other than the reference.")
    if len(names) > _COMPARE_MAX_PROPOSALS:
        raise HTTPException(status_code=400, detail=(
            f"Compare up to {_COMPARE_MAX_PROPOSALS} proposals at once ({len(names)} given)."))
    if study != "scan" and request.timestamp is None:
        # One tick for all: "latest" must not move between the comparisons.
        request = request.model_copy(update={"timestamp": _resolve_tick(request.data_source, None)})
    comparisons = []
    for n in names:
        comparisons.append(await compare_networks(request.model_copy(update={"proposal": n, "proposals": None})))
    first = comparisons[0]
    table = [{"network": _reference(request), "role": "reference",
              **{k: first["base"].get(k) for k in _RANK_FIELDS.get(study, first["base"].keys())}}]
    for n, c in zip(names, comparisons):
        row = {"network": n, "role": "proposal",
               **{k: c["proposal"].get(k) for k in _RANK_FIELDS.get(study, c["proposal"].keys())}}
        if "new_violations" in c:
            row["new_violations"] = len(c["new_violations"]) if isinstance(c["new_violations"], list) \
                else c["new_violations"]
            row["resolved_violations"] = len(c["resolved_violations"]) \
                if isinstance(c["resolved_violations"], list) else c["resolved_violations"]
        table.append(row)
    return {"study": study, "several": True, "compared_with": _reference(request),
            "timestamp": first.get("timestamp"), "data_source": first.get("data_source"),
            "proposals": names, "table": table, "comparisons": comparisons}


def _compare_over_time(request: CompareNetworksRequest, name: str) -> dict:
    """The security assessment on the base and on proposal `name` at every
    tick of one window, with the same dataset, limits and stress factors.

    What a single moment cannot show: at how many ticks the proposal violates
    where the base does not (and the reverse), which elements, for how long,
    and its worst moments — the hour to take an N-1 screen to next.
    """
    timestamps, _dataset, label = _window_dataset(request.data_source)
    if not timestamps:
        raise HTTPException(status_code=400, detail=f"No {label} loaded.")
    start = _window_index(timestamps, request.start_timestamp, "start_timestamp", label,
                          last=False, default=0)
    end = _window_index(timestamps, request.end_timestamp, "end_timestamp", label,
                        last=True, default=len(timestamps) - 1) + 1
    if end <= start:
        raise HTTPException(status_code=400, detail="end_timestamp is before start_timestamp.")
    if request.step_size is not None and request.step_size < 1:
        raise HTTPException(status_code=400, detail="step_size must be at least 1.")
    span = end - start
    step = request.step_size or max(1, math.ceil(span / _SCAN_MAX_TICKS))
    ticks = [timestamps[i] for i in range(start, end, step)]

    common = request.model_dump(exclude={"proposal", "study", "network", "timestamp", "data_source",
                                         "start_timestamp", "end_timestamp", "step_size", "against",
                                         "proposals"})
    requests = {label_: GridRequest(**common, network=network, timestamp=None,
                                    data_source=request.data_source)
                for label_, network in (("base", _reference(request)), ("proposal", name))}

    def keys(rows):
        return {(v.get("violation_type"), str(v.get("element"))): v.get("value") for v in rows}

    per = {"base": [], "proposal": []}
    for ts in ticks:
        for side, req in requests.items():
            try:
                r = _run_rsa_snapshot(ts, req)
            except HTTPException as exc:
                if exc.status_code != 503:      # 503 = power flow did not converge here
                    raise
                per[side].append(None)
                continue
            volts = [v["vm_pu"] for v in r.get("all_voltages", []) if v.get("vm_pu") is not None]
            lines = [x["loading_percent"] for x in r.get("all_line_loading", [])
                     if x.get("loading_percent") is not None]
            trafos = [x["loading_percent"] for x in r.get("all_trafo_loading", [])
                      if x.get("loading_percent") is not None]
            per[side].append({"violations": keys(r.get("violations", [])),
                              "min_vm": min(volts) if volts else None,
                              "max_vm": max(volts) if volts else None,
                              "max_line": max(lines) if lines else None,
                              "max_trafo": max(trafos) if trafos else None,
                              "slack": r.get("slack_import_mw")})

    def worst(side, field, highest=True):
        pairs = [(row[field], ts) for row, ts in zip(per[side], ticks)
                 if row is not None and row[field] is not None]
        if not pairs:
            return None
        best, at = (max if highest else min)(pairs)
        return {"timestamp": at, "value": round(best, 4)}

    def digest(side):
        rows = per[side]
        return {
            "ticks_with_violations": sum(1 for r in rows if r and r["violations"]),
            "non_converged_ticks": sum(1 for r in rows if r is None),
            "worst_min_vm": worst(side, "min_vm", highest=False),
            "worst_max_vm": worst(side, "max_vm"),
            "worst_line_loading": worst(side, "max_line"),
            "worst_trafo_loading": worst(side, "max_trafo"),
            "max_slack_import": worst(side, "slack"),
        }

    def differences(a, b):
        """Elements violating on `b` at ticks where they do not on `a`."""
        found: dict = {}
        for ts, ra, rb in zip(ticks, per[a], per[b]):
            if rb is None:
                continue
            before = ra["violations"] if ra else {}
            for key, val in rb["violations"].items():
                if key in before:
                    continue
                entry = found.setdefault(key, {"type": key[0], "element": key[1], "ticks": 0,
                                               "first": ts, "worst_value": val})
                entry["ticks"] += 1
                if val is not None and entry["worst_value"] is not None:
                    low = key[0] == "bus_vm_pu" and val < 1.0
                    entry["worst_value"] = min(entry["worst_value"], val) if low else \
                        max(entry["worst_value"], val)
        out = sorted(found.values(), key=lambda e: -e["ticks"])
        for e in out:
            if isinstance(e["worst_value"], float):
                e["worst_value"] = round(e["worst_value"], 4)
        return out[:15]

    def only(a, b):
        hits = [ts for ts, ra, rb in zip(ticks, per[a], per[b])
                if rb and rb["violations"] and not (ra and ra["violations"])]
        return {"count": len(hits), "first": hits[:10]}

    def column(side, field):
        return [None if r is None or r[field] is None else round(r[field], 4) for r in per[side]]

    proposal_digest = digest("proposal")
    next_hour = (proposal_digest["worst_min_vm"] or proposal_digest["worst_line_loading"] or {}).get("timestamp")
    out = {
        "study": "scan", "data_source": _dataset_label(request),
        "thresholds_used": _thresholds_used(request),
        "load_scaling_factor": request.load_scaling_factor,
        "window": {"start": ticks[0], "end": ticks[-1], "ticks_scanned": len(ticks),
                   "step": step, "ticks_in_window": span,
                   **({"note": f"every {step}th tick: the window holds {span} ticks; narrow it or "
                               "set step_size to scan every tick"} if step > 1 and not request.step_size
                      else {})},
        "network": {**_proposal_card(name), "compared_with": _reference(request)},
        "base": digest("base"), "proposal": proposal_digest,
        "ticks_violating_only_on_proposal": only("base", "proposal"),
        "ticks_violating_only_on_base": only("proposal", "base"),
        "new_violations": differences("base", "proposal"),
        "resolved_violations": differences("proposal", "base"),
        "suggested_next": (f"An N-1 screen of the proposal at {next_hour} (its worst moment in the window)."
                           if next_hour else None),
        # For the chart; the model is sent the summary above, not these.
        "series": {"timestamps": ticks,
                   "base_min_vm": column("base", "min_vm"), "proposal_min_vm": column("proposal", "min_vm"),
                   "base_max_line_loading": column("base", "max_line"),
                   "proposal_max_line_loading": column("proposal", "max_line"),
                   "base_violations": [None if r is None else len(r["violations"]) for r in per["base"]],
                   "proposal_violations": [None if r is None else len(r["violations"])
                                           for r in per["proposal"]]},
    }
    first = getattr(_tick_report, "first", None)
    if first:
        out["network"]["added_elements_at_tick"] = first
    return out


def _opf_digest(r: dict) -> dict:
    """What a flexibility OPF result says, in comparable figures."""
    acts = r.get("activated_resources") or []
    units = [a for a in acts if a.get("type") != "External Grid"]

    def moved(a, key):
        new, base = a.get(f"{key}_new"), a.get(f"{key}_base")
        return abs(float(new) - float(base)) if isinstance(new, (int, float)) and isinstance(base, (int, float)) else 0.0

    grid = next((a for a in acts if a.get("type") == "External Grid"), {})
    volts = [v["vm_pu"] for v in r.get("bus_voltages_post_opf") or []
             if isinstance(v, dict) and isinstance(v.get("vm_pu"), (int, float))]
    out = {
        "status": r.get("status"), "feasible": r.get("feasible"),
        "units_moved": sum(1 for a in units if moved(a, "Pg") > 1e-3 or moved(a, "Qg") > 1e-3),
        "redispatch_mw": round(sum(moved(a, "Pg") for a in units), 4),
        "reactive_redispatch_mvar": round(sum(moved(a, "Qg") for a in units), 4),
        "grid_import_before_mw": grid.get("Pg_base"), "grid_import_after_mw": grid.get("Pg_new"),
        "min_vm_after": min(volts) if volts else None, "max_vm_after": max(volts) if volts else None,
    }
    diagnosis = r.get("infeasibility_diagnosis")
    if not r.get("feasible"):
        out["why_not"] = (diagnosis or {}).get("explanation") or r.get("message")
    return out


async def _compare_opf(request: CompareNetworksRequest, name: str) -> dict:
    """The deterministic flexibility OPF on the base and on a proposal at one
    operating point: whether a corrective dispatch exists on each, what it
    takes, and how each unit's dispatch differs."""
    ts = _resolve_tick(request.data_source, request.timestamp)
    # The comparison must not leave the proposal's dispatch as "the last one".
    last = (app_data.get("last_dispatch_result"), app_data.get("last_dispatch_timestamp"))
    results = {}
    try:
        for label, network in (("base", _reference(request)), ("proposal", name)):
            req = FlexibilityRequest(
                timestamp=ts, data_source=request.data_source, network=network,
                load_scaling_factor=request.load_scaling_factor,
                opf_vm_upper=request.vm_upper_pu, opf_vm_lower=request.vm_lower_pu)
            _tick_report.first = None
            results[label] = await optimise_flexibility(req)
            results[label + "_added"] = getattr(_tick_report, "first", None)
    finally:
        app_data["last_dispatch_result"], app_data["last_dispatch_timestamp"] = last
    base, prop = _opf_digest(results["base"]), _opf_digest(results["proposal"])

    def by_unit(r):
        return {a["element"]: a for a in r.get("activated_resources") or []
                if a.get("type") != "External Grid" and a.get("element")}

    def delta(a, key):
        if not a:
            return 0.0
        new, old = a.get(f"{key}_new"), a.get(f"{key}_base")
        return round(float(new) - float(old), 4) if isinstance(new, (int, float)) and isinstance(old, (int, float)) else 0.0

    ub, up = by_unit(results["base"]), by_unit(results["proposal"])
    units = []
    for element in sorted(set(ub) | set(up)):
        row = {"element": element,
               "base_dP_mw": delta(ub.get(element), "Pg"), "proposal_dP_mw": delta(up.get(element), "Pg"),
               "base_dQ_mvar": delta(ub.get(element), "Qg"), "proposal_dQ_mvar": delta(up.get(element), "Qg")}
        if any(abs(v) > 1e-3 for k, v in row.items() if k != "element"):
            units.append(row)
    units.sort(key=lambda r: -max(abs(r["proposal_dP_mw"] - r["base_dP_mw"]),
                                  abs(r["proposal_dQ_mvar"] - r["base_dQ_mvar"])))
    numeric = ("units_moved", "redispatch_mw", "reactive_redispatch_mvar", "grid_import_after_mw")
    out = {
        "study": "opf", "timestamp": ts, "data_source": _dataset_label(request),
        "thresholds_used": _thresholds_used(request), "load_scaling_factor": request.load_scaling_factor,
        "network": {**_proposal_card(name), "compared_with": _reference(request)},
        "base": base, "proposal": prop,
        "changes": {k: round(prop[k] - base[k], 4) for k in numeric
                    if isinstance(prop.get(k), (int, float)) and isinstance(base.get(k), (int, float))},
        "units": units[:12],
    }
    if results["proposal_added"]:
        out["network"]["added_elements_at_tick"] = results["proposal_added"]
    return out
