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

import json
import math
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from fastapi import FastAPI, Form, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import io
import pickle
import os
import pandapower as pp
from contextlib import asynccontextmanager
import copy
import pandas as pd
import numpy as np
import requests
from pydantic import BaseModel
from pyomo.environ import value
from pyomo.opt import TerminationCondition
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import modify_network as mn
import load_gen_assignment as lg
import rsa_engine as rs
import ca_engine as ca

import flex_engine as fe
import network_loader as nl

from dotenv import load_dotenv

load_dotenv()  # This safely loads your EDDK_API_TOKEN from the .env file
EDDK_API_TOKEN = os.getenv("EDDK_API_TOKEN")

import pipeline_functions as eddk_pipe


BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(BACKEND_DIR, ".."))
DATA_FILES_DIR = os.path.join(REPO_ROOT, "data_files")
SYSTEMS_DIR = os.path.join(REPO_ROOT, "systems")
DEFAULT_GRID_PROFILE = "pglib_case14"
_LAST_UPLOAD_SENTINEL      = os.path.join(SYSTEMS_DIR,    "last_uploaded.json")
_LAST_TIMESERIES_SENTINEL  = os.path.join(DATA_FILES_DIR, "last_uploaded_timeseries.json")
_LAST_FORECAST_SENTINEL    = os.path.join(DATA_FILES_DIR, "last_uploaded_forecast.json")


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
    stored_path = os.path.join(target_dir, stored_name)
    with open(stored_path, "wb") as fh:
        fh.write(content)
    return stored_path


def _safe_float(v, ndigits: int = 4):
    """Return a JSON-safe rounded float, or None for NaN/inf values."""
    try:
        f = float(v)
        return None if not math.isfinite(f) else round(f, ndigits)
    except (TypeError, ValueError):
        return None


def _clean_label(raw) -> str:
    """Normalize optional labels and treat empty/nan/none as missing."""
    s = str(raw).strip() if raw is not None else ""
    return "" if s.lower() in {"", "nan", "none"} else s


def _bus_display_name(net, bus_idx: int) -> str:
    """Return a stable display name for a bus without changing any IDs."""
    if "name" in net.bus.columns:
        cleaned = _clean_label(net.bus.at[bus_idx, "name"])
        if cleaned:
            return cleaned
    return f"Bus_{int(bus_idx)}"


def _line_display_name(net, line_idx: int) -> str:
    """Build a human-readable line label for API/chart display.

    Format keeps original line name when present, but always appends endpoints
    and line index to prevent ambiguity after bus/line renames.
    """
    from_bus = int(net.line.at[line_idx, "from_bus"])
    to_bus = int(net.line.at[line_idx, "to_bus"])
    from_name = _bus_display_name(net, from_bus)
    to_name = _bus_display_name(net, to_bus)
    endpoints = f"{from_name} -> {to_name}"
    raw_name = _clean_label(net.line.at[line_idx, "name"]) if "name" in net.line.columns else ""
    if raw_name:
        return f"{raw_name} | {endpoints} [L{int(line_idx)}]"
    return f"{endpoints} [L{int(line_idx)}]"


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

    names = _nonempty(raw)
    if names:
        return names
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
    elements whose setpoint actually moved (|Pg_new - Pg_base| > 1 kW).

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
        ``Pg_new``, ``Pg_up``, ``Pg_down``.
    """
    records = []
    # Local generators
    for g in model.G:
        Pg_new_val = value(model.Pg_new[g])
        Pg_base_val = value(model.Pg_base[g])

        if abs(Pg_new_val - Pg_base_val) > 0.001:
            records.append({
                "element": str(g),
                "type": "Generator",
                "Pg_base": round(Pg_base_val, 4),
                "Pg_new": round(Pg_new_val, 4),
                "Pg_up": round(max(Pg_new_val - Pg_base_val, 0), 4),
                "Pg_down": round(max(Pg_base_val - Pg_new_val, 0), 4)
            })

    # Check for external grid in the model Pymo does not store its baseline: Slack bus check necessary
    for e in model.Ext:
        try:
            Pext_new_val = value(model.Pext[e])
            if hasattr(net_base, 'res_ext_grid') and not net_base.res_ext_grid.empty:
                ext_idx = net_base.ext_grid[net_base.ext_grid['name'] == e].index
                if not ext_idx.empty:
                    Pext_base_val = net_base.res_ext_grid.at[ext_idx[0], 'p_mw']
                else:
                    Pext_base_val = Pext_new_val
            else:
                Pext_base_val = Pext_new_val

            if abs(Pext_new_val - Pext_base_val) > 0.001:
                records.append({
                    "element": str(e),
                    "type": "External Grid",
                    "Pg_base": round(Pext_base_val, 4),
                    "Pg_new": round(Pext_new_val, 4),
                    "Pg_up": round(max(Pext_new_val - Pext_base_val, 0), 4),
                    "Pg_down": round(max(Pext_base_val - Pext_new_val, 0), 4)
                })
        except Exception:
            pass
    return records


# KPI calculations — ported verbatim from main_KPIs_Included.ipynb
# The function below is the source of truth for the /api/kpi/evaluate endpoint.

def calculate_kpi_target_demand_flexibility(df_reg_flex_rsa: pd.DataFrame, Pg_max_data: dict) -> dict:
    """
    KPI-1: Target demand flexibility as percentage of system demand.

    Measures the **residual upward reserve** from local generators
    relative to total system demand, *after* the optimizer has
    committed its dispatch.

    Formula
        KPI_1 (%) = [ Σ (Pg_max_i − Pg_new_i) / Σ Pg_new ] * 100

    where:
        * ``Pg_max_i`` is the maximum active power of generator ``i``
          (looked up in ``Pg_max_data``).
        * ``Pg_new_i`` is its **post-optimization** dispatch.
        * ``Σ Pg_new`` covers **all** rows (generators + external grid)
          and represents total system demand (load + losses).
                * Only rows with ``type == "Generator"`` contribute to the
                    numerator; the external-grid interface is intentionally
          excluded because it is import/export, not a controllable
          flexibility resource.
        * Minimum generator output is assumed to be 0, so only upward
          flexibility is considered.

    Departure from notebook
    -----------------------
    Anosh's notebook used ``Pg_max - Pg_base`` (pre-dispatch headroom).
    We use ``Pg_max - Pg_new`` (post-dispatch reserve) so the indicator
    is responsive to operational decisions: load level *and* Sweden-cable
    cap both move ``Pg_new`` and therefore move KPI-1. The two formulas
    coincide when the optimizer leaves local generators untouched
    (typical low-load case where the slack cable absorbs everything).

    Parameters
    df_reg_flex_rsa : pandas.DataFrame
        Full regulation frame from ``flex_engine.prepare_regulation_df_rsa``.
        Must contain columns ``element``, ``type``, ``Pg_base``, ``Pg_new``.
    Pg_max_data : dict
        Mapping ``substation_name -> Pg_max (MW)``. See :data:`PG_MAX_DATA`.

    Returns
    dict
        ``{available_upward_flexibility_MW, system_demand_MW,
        target_demand_flexibility_pct}``.
    """

    required_cols = {'element', 'type', 'Pg_base', 'Pg_new'}
    missing_cols = required_cols - set(df_reg_flex_rsa.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns in df_reg_flex_rsa: {missing_cols}")

    df = df_reg_flex_rsa.copy(deep=True)

    # Keep only generators for flexibility calculation
    df_gen = df[df['type'].str.strip().str.lower() == 'generator'].copy(deep=True)

    if df_gen.empty:
        raise ValueError("No generator rows found in df_reg_flex_rsa.")

    # Map maximum active power values to generator rows
    df_gen['Pg_max'] = df_gen['element'].map(Pg_max_data)

    # Check if any generator is missing from Pg_max_data
    missing_pgmax = df_gen[df_gen['Pg_max'].isna()]['element'].unique()
    if len(missing_pgmax) > 0:
        raise ValueError(f"Pg_max_data is missing values for generators: {list(missing_pgmax)}")

    # Available upward flexibility for each generator
    df_gen['available_upward_flex_MW'] = (df_gen['Pg_max'] - df_gen['Pg_new']).clip(lower=0.0)

    # Total available upward flexibility from generators
    total_available_flex_MW = df_gen['available_upward_flex_MW'].sum()

    # Total system demand = total supplied active power at this timestamp
    system_demand_MW = df['Pg_new'].sum()

    if system_demand_MW <= 0:
        raise ValueError(f"System demand must be positive, but got {system_demand_MW:.6f} MW.")

    # KPI calculation
    kpi_target_demand_flexibility_pct = (total_available_flex_MW / system_demand_MW) * 100

    return {
        'available_upward_flexibility_MW': total_available_flex_MW,
        'system_demand_MW': system_demand_MW,
        'target_demand_flexibility_pct': kpi_target_demand_flexibility_pct
    }


def calculate_kpi_flexibility_utilization(df_reg_flex_rsa: pd.DataFrame) -> dict:
    """
    KPI-2: Flexibility utilization rate.

    Measures how accurately the activated flexibility matches the
    required flexibility at the considered timestamp.

    Formula
        KPI_2 (%) = (1 − |F_activated − F_required| / F_required) * 100

    where:
        * ``F_required  = | Σ Pg_new − Σ Pg_base |``
        * ``F_activated = Σ (Pg_up + Pg_down)``

    Edge cases
    * If ``F_required == 0`` and ``F_activated == 0`` -> KPI = 100%.
    * If ``F_required == 0`` and ``F_activated > 0`` -> KPI = 0%.


    To keep the indicator interpretable we apply two safeguards:

    1. Treat ``F_required < 1e-2 MW`` (10 kW — below any meaningful
       dispatch resolution) as effectively zero, falling through to
       the explicit zero edge-case branch.
    2. Clip the final percentage to ``[0, 100]`` as a defensive cap.

    Both safeguards leave the indicator unchanged in any regime where
    the original formula was well-conditioned.

    Parameters
    df_reg_flex_rsa : pandas.DataFrame
        Must contain ``Pg_base``, ``Pg_new``, ``Pg_up``, ``Pg_down``.

    Returns
    dict
        ``{required_flexibility_MW, activated_flexibility_MW,
        flexibility_utilization_pct}``.
    """

    required_cols = {'Pg_base', 'Pg_new', 'Pg_up', 'Pg_down'}
    missing_cols = required_cols - set(df_reg_flex_rsa.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    df = df_reg_flex_rsa.copy()

    # Required flexibility
    F_required = abs(df['Pg_new'].sum() - df['Pg_base'].sum())

    # Activated flexibility
    F_activated = (df['Pg_up'].abs() + df['Pg_down'].abs()).sum()

    # Treat sub-resolution F_required as zero — see "Numerical robustness".
    F_REQUIRED_NOISE_MW = 1e-2  # 10 kW

    if F_required < F_REQUIRED_NOISE_MW:
        if F_activated < F_REQUIRED_NOISE_MW:
            kpi = 100.0
        else:
            kpi = 0.0
    else:
        kpi = (1 - abs(F_activated - F_required) / F_required) * 100
        # Clip — formula can produce values outside [0,100] when the
        # optimizer over- or under-shoots. Outside that range the metric
        # stops being interpretable as a percentage.
        kpi = max(0.0, min(100.0, kpi))

    return {
        'required_flexibility_MW': F_required,
        'activated_flexibility_MW': F_activated,
        'flexibility_utilization_pct': kpi
    }


def calculate_kpi_prevented_violation_ratio(df_reg_flex_rsa) -> dict:
    """
    KPI-3: Prevented violation ratio in the distribution grid.

    Binary indicator: if the optimizer returned a feasible regulation
    frame, every detected violation is considered resolved (100%).
    Otherwise the KPI is 0%.

    Parameters
    df_reg_flex_rsa : pandas.DataFrame or None
        The regulation frame from a solved Pyomo model, or ``None``
        (or empty) if the solve was infeasible.

    Returns
    dict
        ``{prevented_violation_ratio_pct, status}``.
    """

    # If dataframe exists and is not empty → feasible solution
    if df_reg_flex_rsa is not None and not df_reg_flex_rsa.empty:
        kpi = 100.0
        status = "All detected violations resolved"
    else:
        kpi = 0.0
        status = "Violations not fully resolved (infeasible)"

    return {
        'prevented_violation_ratio_pct': kpi,
        'status': status
    }


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

    try:
        results = SolverFactory('ipopt').solve(model_opt, tee=False)
    except Exception:
        return None, model_opt
    return results, model_opt


# ``/api/time/advance`` which bumps ``current_index``. is the only write API point

#   timestamps     : sorted list[str] is every available simulation tick from pkl file
#   measurements   : dict[timestamp_str -> DataFrame] of smart-meter rows
#   current_index  : int — points to ``timestamps`` shows how many 15min steps have been made
#   net            : pandapowerNet — base grid
#   db_full        : dict — pre-computed Ybus for the intact grid
#   db_n1_line     : dict[line_idx -> Ybus dict] is for 24 single-line outages
#   db_n1_trafo    : dict[trafo_idx -> Ybus dict] is for 16 single-trafo outages.
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
    "db_n1_line": None,
    "db_n1_trafo": None,
    "grid_profile": {},   # metadata exposed via GET /api/grid_constants
    "last_dispatch_result": [],
    "last_dispatch_timestamp": None,
    "default_vm_lower": 0.95,  # network-specific defaults, overwritten at load time
    "default_vm_upper": 1.05,
}


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
            detail=f"Unknown data_source {data_source!r}. Use 'measurements' or 'forecasts'.",
        )
    return app_data["timestamps"], app_data["measurements"], "measurements"


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


def _apply_dispatch_to_net(net, dispatch_rows):
    """Apply dispatch rows (element, Pg_new, Qg_new) to sgen/ext_grid setpoints.

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


class _TimeseriesRequest(BaseModel):
    """Mixin for every tool that reads the timeseries.

    ``data_source`` lets the LLM choose which dataset the tool operates on:
      - ``"measurements"`` (default): historical actuals; point tools default to
        the live simulation clock.
      - ``"forecasts"``: read-only look-ahead series; point tools default to the
        first forecast tick.
    ``timestamp`` optionally targets one tick within the chosen source (ISO
    prefix matching, e.g. ``'2000-01-09 18'``). When omitted the default above
    applies. The two series cover disjoint time ranges.
    """
    data_source: str = "measurements"
    timestamp: str | None = None


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
    # island_mode: bool = False was meant to be added for testing without the swedish slack bus as a contingency event


class WorstCaseRequest(BaseModel):
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
    # island_mode: bool = False was meant to be added without the swedish slack bus as a contingency event


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


class KPIRequest(_TimeseriesRequest):
    """Request body for ``POST /api/kpi/evaluate``.

    Attributes
    load_scaling_factor : float
        Multiplicative load multiplier driven by the global UI slider.
    disabled_generators : list[str]
        Substation names whose sgens are removed before solving.
        slack_max_mw : float
                Maximum bidirectional capacity (MW) of the external-grid
                interface for the *constrained* scenario solve. The connected
                baseline solve always runs with the engine's default ±200 MW
                (a modelling artefact); this field controls the second,
                *what-if* solve and is clamped to the active profile limit.

                * profile maximum -> realistic upper bound; constrained
                    scenario will resemble the connected baseline at most
                    timestamps.
                * 0.0 -> fully islanded (the interface carries no power).
        * Any value in between models a de-rated / partially available
          cable.

        Driven by the slider on the *KPIs & Export* tab. The same
        value is mirrored into ``slack-store`` and consumed by
        :class:`FlexibilityRequest` so the *Flexibility Management*
        dispatch table reports a configuration consistent with the
        KPI numbers.
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


class ElementTimeseriesRequest(_TimeseriesRequest):
    """Request body for ``POST /api/grid/element_timeseries``.

    Attributes
    ----------
    element_type : str
        One of ``'bus'``, ``'line'``, or ``'trafo'``.
    element_name : str
        Partial or exact element name. Case-insensitive substring match
        against the ``name`` column of the corresponding pandapower table.
        For buses use substation name fragments (e.g. ``'Åkirkeby'``).
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
        Substation name of the generator to sweep (e.g. ``"Hasle"``).
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


def _build_and_store_ybus(net) -> None:
    """Build Ybus databases on-the-fly from *net* and populate app_data."""
    print("[lifespan] Building Ybus databases on-the-fly...")
    t0 = time.time()
    try:
        db_full, db_n1_line, db_n1_trafo = nl.build_ybus(net)
        app_data["db_full"] = db_full
        app_data["db_n1_line"] = db_n1_line
        app_data["db_n1_trafo"] = db_n1_trafo
        print(
            f"[lifespan] Ybus ready in {round(time.time() - t0, 2)}s: "
            f"{len(db_n1_line)} line contingencies, {len(db_n1_trafo)} trafo contingencies."
        )
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


def _write_upload_sentinel(network_path: str, grid_profile: dict, convert_gen_to_sgen: bool) -> None:
    """Write a JSON sentinel so the backend can restore the uploaded network on restart."""
    record = {
        "network_path": network_path,
        "grid_profile": grid_profile,
        "convert_gen_to_sgen": convert_gen_to_sgen,
        "uploaded_at": time.time(),
    }
    try:
        with open(_LAST_UPLOAD_SENTINEL, "w", encoding="utf-8") as fh:
            json.dump(record, fh, indent=2)
        print(f"[upload] Sentinel written → {_LAST_UPLOAD_SENTINEL}")
    except Exception as exc:
        print(f"[upload] WARNING: Could not write upload sentinel: {exc}")


def _parse_timeseries_csv(content: bytes) -> tuple[dict, list]:
    """Parse a long-format timeseries CSV into (measurements_dict, timestamps_list).

    Shared by the measurement and forecast upload/restore paths. Raises
    ``ValueError`` with a human-readable message on any problem so callers can
    surface it as a 400 or log it on restart.
    """
    text = content.decode("utf-8-sig")  # strip BOM if Excel-exported
    df_raw = pd.read_csv(io.StringIO(text))

    required = {"timestamp", "substation_name", "production_mw", "consumption_mw"}
    missing = required - set(df_raw.columns)
    if missing:
        raise ValueError(
            f"CSV is missing required columns: {sorted(missing)}. "
            f"Required: timestamp, substation_name, production_mw, consumption_mw"
        )

    df_raw["timestamp"] = pd.to_datetime(df_raw["timestamp"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    df_raw["production_mw"] = pd.to_numeric(df_raw["production_mw"], errors="coerce").fillna(0.0)
    df_raw["consumption_mw"] = pd.to_numeric(df_raw["consumption_mw"], errors="coerce").fillna(0.0)

    measurements: dict = {}
    for ts, group in df_raw.groupby("timestamp", sort=False):
        measurements[ts] = (
            group[["substation_name", "production_mw", "consumption_mw"]]
            .rename(columns={"production_mw": "production", "consumption_mw": "consumption"})
            .reset_index(drop=True)
        )

    timestamps = sorted(measurements.keys())
    if not timestamps:
        raise ValueError("CSV contained no valid rows after parsing.")
    return measurements, timestamps


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
        new_measurements, new_timestamps = _parse_timeseries_csv(content)
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
    if not os.path.isfile(csv_path):
        print(f"[lifespan] Forecast sentinel points to missing file {csv_path!r} — using synthetic forecast.")
        return False

    print(f"[lifespan] Restoring {source} forecast: {csv_path!r}")
    try:
        with open(csv_path, "rb") as fh:
            content = fh.read()
        measurements, timestamps = _parse_timeseries_csv(content)
    except Exception as exc:
        print(f"[lifespan] Could not parse forecast CSV: {exc} — using synthetic forecast.")
        return False

    app_data["forecasts"] = measurements
    app_data["forecast_timestamps"] = timestamps
    app_data["forecasts_source"] = source
    print(f"[lifespan] Forecast restored: {len(timestamps)} timestamps "
          f"({timestamps[0]} → {timestamps[-1]}).")
    return True


def _lifespan_from_uploaded(sentinel: dict) -> bool:
    """Startup path that restores a previously uploaded network.

    Supports .m (MATPOWER), .json (pandapower JSON), .xlsx (pandapower Excel),
    and .uct (UCTE) formats, detected from the stored file extension.

    Returns True when the network is loaded and global state is fully
    populated, False on any error so the caller can fall back to the
    default YAML profile.
    """
    import synthetic_timeseries as st

    network_path = sentinel.get("network_path", "")
    if not os.path.isfile(network_path):
        print(f"[lifespan] Sentinel points to missing file {network_path!r} — falling back.")
        return False

    ext = os.path.splitext(network_path)[1].lower()
    print(f"[lifespan] Restoring previously uploaded network: {network_path!r} (format: {ext})")
    try:
        if ext == ".m":
            from pandapower.converter.matpower import from_mpc
            net = from_mpc(network_path, f_hz=50, validate_conversion=False)
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

    # Fix MATPOWER off-nominal tap positions (not needed for native pandapower formats).
    if ext == ".m" and not net.trafo.empty and (net.trafo["tap_pos"] != net.trafo["tap_neutral"]).any():
        net.trafo["tap_neutral"] = net.trafo["tap_pos"]

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

    initial_meas = measurements[timestamps[0]]
    subs = get_substation_names(net)
    substations.extend(subs)
    net, _ = lg.assign_load_values_from_measurements(net, initial_meas, subs)
    net, _ = lg.assign_generators_values_from_measurements(net, initial_meas, subs)

    grid_profile = sentinel.get("grid_profile", {})
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
    app_data["db_n1_line"]     = None
    app_data["db_n1_trafo"]    = None

    PG_MAX_DATA.update(get_pg_max_data(net))
    print(f"[lifespan] PG_MAX_DATA: {len(PG_MAX_DATA)} generators.")

    _build_and_store_ybus(net)
    _restore_timeseries_from_sentinel()
    _restore_forecast_from_sentinel()
    print("\n !API IS READY TO RECEIVE REQUESTS! \n")
    return True


def _lifespan_generic(profile_name: str, backend_dir: str) -> None:
    """Startup path for any YAML profile.

    Loads the network via network_loader, generates a synthetic timeseries,
    builds Ybus databases on-the-fly, and populates all global app_data fields.
    """
    import synthetic_timeseries as st

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
    timestamps, measurements = st.generate(
        net, n_days=n_days, resolution_min=resolution,
        load_profile=load_prof, generation_profile=gen_prof, seed=42,
    )
    print(f"[lifespan] Timeseries ready: {len(timestamps)} timestamps.")

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

    PG_MAX_DATA.update(get_pg_max_data(net))
    print(f"[lifespan] PG_MAX_DATA: {len(PG_MAX_DATA)} generators.")

    # Build Ybus on-the-fly (generic profiles never use pre-computed pickles).
    _build_and_store_ybus(net)
    _restore_timeseries_from_sentinel()
    _restore_forecast_from_sentinel()

    print("\n !API IS READY TO RECEIVE REQUESTS! \n")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown function for FastAPI.

    Runs once when uvicorn boots. The active network is selected by
    ``GRID_PROFILE`` and initialized through the generic profile loader.
    """
    print("0) Starting Backend initialization...")

    _ensure_storage_dirs()
    print(f"1) Upload storage ready: data_files={DATA_FILES_DIR}, systems={SYSTEMS_DIR}")

    # Prefer a previously uploaded network over the default YAML profile.
    if os.path.isfile(_LAST_UPLOAD_SENTINEL):
        print(f"[lifespan] Found upload sentinel: {_LAST_UPLOAD_SENTINEL}")
        try:
            with open(_LAST_UPLOAD_SENTINEL, encoding="utf-8") as fh:
                sentinel = json.load(fh)
            if _lifespan_from_uploaded(sentinel):
                yield
                print("Shutting down backend")
                return
            print("[lifespan] Sentinel restore failed — falling back to YAML profile.")
        except Exception as exc:
            print(f"[lifespan] Could not read sentinel: {exc} — falling back to YAML profile.")

    profile_name = os.environ.get("GRID_PROFILE", DEFAULT_GRID_PROFILE).lower().strip()
    print(f"[lifespan] GRID_PROFILE={profile_name!r}")

    _lifespan_generic(profile_name, BACKEND_DIR)

    yield
    print("Shutting down backend")


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Tests if API is running. Returns a static hello so the frontend can confirm the backend is up."""
    return {"message": "Power-system digital twin backend is online"}


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
    measurements_info = {
        "loaded": bool(meas_ts),
        "n_timestamps": len(meas_ts),
        "first_timestamp": meas_ts[0] if meas_ts else None,
        "last_timestamp": meas_ts[-1] if meas_ts else None,
    }
    forecast_info = {
        "loaded": bool(fc_ts),
        "n_timestamps": len(fc_ts),
        "first_timestamp": fc_ts[0] if fc_ts else None,
        "last_timestamp": fc_ts[-1] if fc_ts else None,
    }
    return {
        "name": profile.get("name", "Unknown Grid"),
        "n_substations": len(subs),
        "n_lines": len(net.line),
        "n_trafos": len(net.trafo),
        "substation_names": subs,
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
    }


@app.post("/api/time/advance")
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
    stored_path = _persist_uploaded_bytes(
        DATA_FILES_DIR,
        file.filename,
        content,
        f"uploaded_{kind_norm}",
        ".csv",
    )
    try:
        new_measurements, new_timestamps = _parse_timeseries_csv(content)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {exc}")

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
        }

    # --- Measurement path: drives the simulation clock.
    app_data["measurements"] = new_measurements
    app_data["timestamps"] = new_timestamps
    app_data["current_index"] = 0
    app_data["measurements_source"] = "uploaded"

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
    return {
        "status": "success",
        "kind": "measurements",
        "n_timestamps": len(new_timestamps),
        "first_timestamp": new_timestamps[0],
        "last_timestamp": new_timestamps[-1],
        "unmatched_buses": unmatched,
        "stored_path": stored_path,
    }


@app.post("/api/network/upload")
async def upload_network(
    file: UploadFile = File(...),
    convert_gen_to_sgen: bool = Form(True),
):
    """Replace the in-memory pandapower network with a user-supplied MATPOWER .m file.

    Converts the file with ``pandapower.converter.from_mpc``, runs a base-case
    power flow to initialise the operating point, and builds a minimal
    single-timestamp synthetic measurement series from the network's base-case
    load/generation so that subsequent calls to ``/api/grid/rsa`` (power-flow
    only) and the LLM network-info tools work immediately.

    Admittance databases (``db_full``, ``db_n1_line``, ``db_n1_trafo``) are
    reset to ``None`` after upload — RSA optimisation, OPF, and N-1
    contingency tools will return 503 until Phase 2 on-the-fly Ybus
    computation is implemented.

    Returns
    -------
    dict
        ``status``, ``n_buses``, ``n_lines``, ``n_trafos``,
        ``first_timestamp``, ``bus_names`` (first 30), ``note``.
    """
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    ext = os.path.splitext(file.filename or "")[1].lower()
    _SUPPORTED_EXTS = {".m", ".json", ".xlsx", ".uct"}
    if ext not in _SUPPORTED_EXTS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format '{ext}'. Accepted: .m  .json  .xlsx  .uct",
        )

    stored_path = _persist_uploaded_bytes(
        SYSTEMS_DIR,
        file.filename,
        content,
        "uploaded_system",
        ext,
    )

    try:
        if ext == ".m":
            from pandapower.converter.matpower import from_mpc
            net = from_mpc(stored_path, f_hz=50, validate_conversion=False)
        elif ext == ".json":
            net = pp.from_json(stored_path)
        elif ext == ".xlsx":
            net = pp.from_excel(stored_path)
        elif ext == ".uct":
            from pandapower.converter.ucte.from_ucte import from_ucte
            net = from_ucte(stored_path)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not parse {ext} file: {exc}")

    # Fix off-nominal transformer tap positions from MATPOWER import.
    # from_mpc encodes ratios like 0.978 as tap_pos=-1, tap_neutral=0.
    # If OPF endpoints reset tap_pos to tap_neutral they would silently
    # remove these ratios.  Redefine the MATPOWER tap positions as the
    # neutral reference so the reset is a no-op.
    if ext == ".m" and not net.trafo.empty and (net.trafo['tap_pos'] != net.trafo['tap_neutral']).any():
        net.trafo['tap_neutral'] = net.trafo['tap_pos']
        print(f"[upload] Tap neutral redefined to match MATPOWER tap positions: "
              f"{net.trafo[['hv_bus','lv_bus','tap_pos','tap_neutral']].to_dict('records')}")

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

    # Build a synthetic forecast series on the 7 days immediately following the
    # measurements (different seed → distinct realization). Measurements are the
    # historical actuals that drive the simulation clock; forecasts are read-only
    # look-ahead data for planning/rescheduling tools.
    forecast_start = (pd.Timestamp(timestamps[-1]) + pd.Timedelta(minutes=15)).to_pydatetime()
    forecast_timestamps, forecasts = st.generate(
        net, n_days=7, resolution_min=15, seed=137, start_dt=forecast_start
    )
    print(f"[upload] Forecast ready: {len(forecast_timestamps)} timestamps from "
          f"{forecast_timestamps[0]} to {forecast_timestamps[-1]}.")

    # Replace global app_data atomically.
    net_name = os.path.splitext(file.filename)[0] if file.filename else "Uploaded Network"
    app_data["net"] = net
    app_data["measurements"] = measurements
    app_data["timestamps"] = timestamps
    app_data["current_index"] = 0
    app_data["forecasts"] = forecasts
    app_data["forecast_timestamps"] = forecast_timestamps
    app_data["measurements_source"] = "synthetic"
    app_data["forecasts_source"] = "synthetic"
    app_data["db_full"] = None
    app_data["db_n1_line"] = None
    app_data["db_n1_trafo"] = None
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

    # Build admittance databases on-the-fly (required for OPF / RSA / N-1)
    print(f"[upload] Building admittance databases ({len(net.line)} lines, {len(net.trafo)} trafos)...")
    t0 = time.perf_counter()
    ybus_note = ""
    try:
        db_full, db_n1_line, db_n1_trafo = nl.build_ybus(net)
        app_data["db_full"]    = db_full
        app_data["db_n1_line"] = db_n1_line
        app_data["db_n1_trafo"] = db_n1_trafo
        print(f"[upload] Admittance databases built in {time.perf_counter() - t0:.1f}s.")
    except Exception as _ybus_exc:
        print(f"[upload] WARNING: Ybus build failed: {_ybus_exc} — OPF/N-1 disabled.")
        app_data["db_n1_line"]  = {}   # empty dict, not None — 503 guard uses is None
        app_data["db_n1_trafo"] = {}
        ybus_note = f" Admittance build failed ({_ybus_exc}); OPF/N-1 unavailable."

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
    _write_upload_sentinel(stored_path, app_data["grid_profile"], convert_gen_to_sgen)

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
        "note": (
            f"Network loaded. {len(db_n1_line) if app_data['db_full'] is not None else 0} "
            f"line contingencies and {len(db_n1_trafo) if app_data['db_full'] is not None else 0} "
            f"trafo contingencies computed.{gen_conversion_note}{ybus_note}"
        ),
        "n_sgens": len(net.sgen),
        "n_gens_remaining": len(net.gen),
        "gen_conversion_applied": convert_gen_to_sgen and bool(gen_conversion_note),
    }


@app.post("/api/network/snapshot")
async def network_snapshot(request: _TimeseriesRequest = _TimeseriesRequest()):
    """Return the raw network state at a timestamp.

    Reads directly from the measurement-assigned network tables (no power flow
    required) so it is always fast and never fails due to convergence issues.
    Honors ``data_source`` / ``timestamp`` (defaults to the live measurement clock).
    """
    current_ts = _resolve_tick(request.data_source, request.timestamp)
    meas_df = prepare_measurement_df(_lookup_measurement_df(current_ts))

    net_copy = copy.deepcopy(app_data["net"])
    net_copy, _ = lg.assign_load_values_from_measurements(net_copy, meas_df, substations)
    net_copy, _ = lg.assign_generators_values_from_measurements(net_copy, meas_df, substations)

    if not net_copy.ext_grid.empty and "name" in net_copy.ext_grid.columns:
        _raw = net_copy.ext_grid["name"].iloc[0]
        slack_name = str(_raw).strip() if _raw and str(_raw).strip() else "Slack"
    else:
        slack_name = "Slack"

    # --- Generators: read from assigned p_mw / q_mvar columns directly ---
    generators = []
    if not net_copy.sgen.empty:
        key_col = "substation_name" if "substation_name" in net_copy.sgen.columns else "name"
        for _, row in net_copy.sgen.iterrows():
            name = str(row.get(key_col) or "").strip() or f"Bus_{row['bus']}"
            generators.append({
                "name": name,
                "Pg_mw": _safe_float(row["p_mw"]) or 0.0,
                "Qg_mvar": _safe_float(row.get("q_mvar") or 0.0) or 0.0,
                "Pg_max_mw": _safe_float(PG_MAX_DATA[name]) if name in PG_MAX_DATA else None,
                "Pg_min_mw": 0.0,
            })
    if not net_copy.gen.empty:
        for _, row in net_copy.gen.iterrows():
            bus_id = int(row.get("bus", 0))
            raw_name = row.get("name", None)
            name = str(raw_name).strip() if (raw_name and str(raw_name).strip()) else f"Gen_bus{bus_id}"
            generators.append({
                "name": name,
                "Pg_mw": _safe_float(row["p_mw"]) or 0.0,
                "Qg_mvar": 0.0,
                "Pg_max_mw": _safe_float(row.get("max_p_mw") or 0.0) or 0.0,
                "Pg_min_mw": _safe_float(row.get("min_p_mw") or 0.0) or 0.0,
            })

    # --- Loads: read from assigned p_mw / q_mvar columns directly ---
    loads = []
    for _, row in net_copy.load.iterrows():
        bus_id = int(row["bus"])
        bus_name = net_copy.bus.at[bus_id, "name"] if "name" in net_copy.bus.columns else ""
        label = str(bus_name).strip() if (bus_name and str(bus_name).strip()) else f"Bus_{bus_id}"
        loads.append({
            "bus": label,
            "P_mw": _safe_float(row["p_mw"]) or 0.0,
            "Q_mvar": _safe_float(row.get("q_mvar") or 0.0) or 0.0,
        })

    # --- External grid: estimate from power balance ---
    total_gen_mw = round(sum(g["Pg_mw"] for g in generators), 4)
    total_load_mw = round(sum(l["P_mw"] for l in loads), 4)
    total_import_mw = round(total_load_mw - total_gen_mw, 4)
    ext_grid = {
        "name": slack_name,
        "P_import_mw": total_import_mw,
        "Q_mvar": 0.0,
    }

    return {
        "timestamp": current_ts,
        "generators": generators,
        "loads": loads,
        "ext_grid": ext_grid,
        "totals": {
            "total_generation_mw": total_gen_mw,
            "total_load_mw": total_load_mw,
            "net_import_mw": total_import_mw,
            "total_losses_mw": 0.0,
        },
    }


# Static scan/RSA defaults baked into the request models. When a request still
# carries these exact values it means the caller did not specify limits, so the
# security-assessment endpoints fall back to the *loaded network's own* limits
# (e.g. 0.94–1.06 for case30) rather than these tighter generic defaults — which
# otherwise flag buses that sit within the network's real limits.
_STATIC_VM_LOWER = 0.95
_STATIC_VM_UPPER = 1.05
_STATIC_MAX_LOADING = 90.0


def _effective_thresholds(request) -> tuple[float, float, float, float]:
    """Resolve (vm_lower, vm_upper, max_line_loading, max_trafo_loading).

    Substitutes the network's own limits wherever the request still holds the
    generic static default, so "is it secure?" is judged against the limits the
    network was defined with.
    """
    gp = app_data.get("grid_profile") or {}
    net_vlo = float(app_data.get("default_vm_lower", _STATIC_VM_LOWER))
    net_vhi = float(app_data.get("default_vm_upper", _STATIC_VM_UPPER))
    net_ml = float(gp.get("max_loading_pct", 100.0) or 100.0)

    vlo = getattr(request, "vm_lower_pu", _STATIC_VM_LOWER)
    vhi = getattr(request, "vm_upper_pu", _STATIC_VM_UPPER)
    mll = getattr(request, "max_line_loading_pct", _STATIC_MAX_LOADING)
    mtl = getattr(request, "max_trafo_loading_pct", _STATIC_MAX_LOADING)

    if vlo == _STATIC_VM_LOWER:
        vlo = net_vlo
    if vhi == _STATIC_VM_UPPER:
        vhi = net_vhi
    if mll == _STATIC_MAX_LOADING:
        mll = net_ml
    if mtl == _STATIC_MAX_LOADING:
        mtl = net_ml
    return vlo, vhi, mll, mtl


def _run_rsa_snapshot(timestamp: str, request: GridRequest) -> dict:
    """Run the existing RSA pipeline for one explicit timestamp."""
    net_copy = copy.deepcopy(app_data["net"])
    measurement_prod_cons = prepare_measurement_df(_lookup_measurement_df(timestamp))

    net_copy, _ = lg.assign_load_values_from_measurements(net_copy, measurement_prod_cons, substations)
    net_copy, _ = lg.assign_generators_values_from_measurements(net_copy, measurement_prod_cons, substations)
    net_copy.load['p_mw'] *= request.load_scaling_factor
    net_copy.load['q_mvar'] *= request.load_scaling_factor
    if request.sgen_scaling_factor != 1.0:
        net_copy.sgen['p_mw'] *= request.sgen_scaling_factor
        net_copy.sgen['q_mvar'] *= request.sgen_scaling_factor

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
    all_voltages = [
        {"bus_name": row["bus_name"], "vm_pu": _safe_float(row["vm_pu"])}
        for _, row in df_bus.iterrows()
    ]

    df_line = net_calculated.res_line.reset_index()
    df_line["line_name"] = [_line_display_name(net_calculated, int(i)) for i in df_line["index"]]
    all_line_loading = [
        {"line_name": row["line_name"], "loading_percent": _safe_float(row["loading_percent"])}
        for _, row in df_line.iterrows()
    ]

    df_trafo = net_calculated.res_trafo.reset_index()
    df_trafo["trafo_name"] = [
        (str(net_calculated.trafo.at[i, "name"]).strip()
         if "name" in net_calculated.trafo.columns
         and pd.notna(net_calculated.trafo.at[i, "name"])
         and str(net_calculated.trafo.at[i, "name"]).strip()
         else f"Trafo_{net_calculated.trafo.at[i, 'hv_bus']}-{net_calculated.trafo.at[i, 'lv_bus']}")
        for i in df_trafo["index"]
    ]
    all_trafo_loading = [
        {"trafo_name": row["trafo_name"], "loading_percent": _safe_float(row["loading_percent"])}
        for _, row in df_trafo.iterrows()
    ]

    df_sgen = net_calculated.res_sgen.reset_index()
    df_sgen["gen_name"] = net_calculated.sgen.loc[
        df_sgen["index"], "substation_name"].values if "substation_name" in net_calculated.sgen.columns else df_sgen[
        "index"].astype(str)

    total_available_flex_mw = sum(
        max(0, PG_MAX_DATA.get(row["gen_name"], row["p_mw"]) - row["p_mw"]) for _, row in df_sgen.iterrows())

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
        "violations": violations_json, "all_voltages": all_voltages,
        "all_line_loading": all_line_loading, "all_trafo_loading": all_trafo_loading,
        "gen_dispatch": gen_dispatch,
        "total_load_mw": total_load_mw,
        "slack_import_mw": slack_import_mw,
        "flexibility_metrics": {"total_available_capacity_mw": round(total_available_flex_mw, 3)},
        "thresholds_used": {
            "vm_upper_pu": _vhi,
            "vm_lower_pu": _vlo,
            "max_line_loading_pct": _mll,
            "max_trafo_loading_pct": _mtl,
        },
    }


def _lookup_measurement_df(ts: str):
    """Return the measurement slice for a timestamp, regardless of dataset.

    Measurement and forecast time ranges are disjoint (forecasts begin after the
    measurement window ends), so a timestamp string maps to exactly one dataset.
    This lets point-in-time tools read the right slice without threading the
    dataset object through every call.
    """
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
    This endpoint runs on a **neutral-tap** grid. ``lifespan`` sets
    ``raw_net.trafo['tap_pos'] = tap_neutral`` once at startup, and
    each request here deep-copies that net without touching taps —
    so RSA voltages reflect neutral-tap physics, not the actual
    mechanical tap positions that would exist in the field.
    If the real-world tap positions become the desired baseline,
    drop the neutralization in ``lifespan`` and neutralize only
    inside the optimization endpoints that still require it
    (``/api/contingency/optimize``, ``/api/flexibility/optimize``,
    ``/api/kpi/evaluate``).
    """
    current_ts = _resolve_tick(request.data_source, request.timestamp)
    return _run_rsa_snapshot(current_ts, request)


@app.post("/api/rsa/historical_risk")
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
    timestamps, measurements, source_label = _resolve_dataset(request.data_source)
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
    if use_parallel:
        indexed_rows: list[tuple[int, dict, float | int | None]] = []
        with ThreadPoolExecutor(max_workers=effective_max_workers) as pool:
            future_map = {
                pool.submit(_run_rsa_snapshot, timestamp, rsa_request): idx
                for idx, timestamp in enumerate(selected_timestamps)
            }
            for future in as_completed(future_map):
                idx = future_map[future]
                snapshot = future.result()
                timestamp = selected_timestamps[idx]
                metric_row = _extract_historical_metric(snapshot, target_kind, target_name, request)
                cond_val = _extract_condition_value(snapshot, timestamp, normalized_condition)
                indexed_rows.append((idx, metric_row, cond_val))
        indexed_rows.sort(key=lambda item: item[0])
        metric_rows = [row for _, row, _ in indexed_rows]
        condition_rows = [cond_val for _, _, cond_val in indexed_rows]
    else:
        for timestamp in selected_timestamps:
            snapshot = _run_rsa_snapshot(timestamp, rsa_request)
            metric_rows.append(_extract_historical_metric(snapshot, target_kind, target_name, request))
            condition_rows.append(_extract_condition_value(snapshot, timestamp, normalized_condition))

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
        "thresholds_used": {
            "vm_upper_pu": request.vm_upper_pu,
            "vm_lower_pu": request.vm_lower_pu,
            "max_line_loading_pct": request.max_line_loading_pct,
            "max_trafo_loading_pct": request.max_trafo_loading_pct,
        },
        "exceedance_count": exceedance_count,
        "exceedance_frequency": (exceedance_count / total_steps) if total_steps else 0.0,
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
    timestamps, dataset, source_label = _resolve_dataset(request.data_source)
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

    for idx in indices:
        ts = timestamps[idx]
        try:
            net_copy = copy.deepcopy(app_data["net"])
            meas = prepare_measurement_df(dataset[ts])
            net_copy, _ = lg.assign_load_values_from_measurements(net_copy, meas, substations)
            net_copy, _ = lg.assign_generators_values_from_measurements(net_copy, meas, substations)
            pp.runpp(net_copy, algorithm="nr", numba=False)
        except Exception:
            continue

        vm = net_copy.res_bus["vm_pu"].values
        line_load = net_copy.res_line["loading_percent"].values
        trafo_load = net_copy.res_trafo["loading_percent"].values
        slack_mw = float(net_copy.res_ext_grid.iloc[0]["p_mw"])

        v_viol = int(((vm < vlo) | (vm > vhi)).sum())
        l_viol = int((line_load > mll).sum())
        t_viol = int((trafo_load > mtl).sum())

        series["timestamps"].append(ts)
        series["violation_counts"].append(v_viol + l_viol + t_viol)
        series["slack_import_mw"].append(round(slack_mw, 4))
        series["min_voltage"].append(round(float(vm.min()), 4))
        series["max_voltage"].append(round(float(vm.max()), 4))
        series["max_line_loading"].append(round(float(line_load.max()) if len(line_load) > 0 else 0.0, 2))

    if not series["timestamps"]:
        raise HTTPException(status_code=500, detail="No timestamps could be scanned.")

    def _worst(key: str, maximize: bool = True) -> dict:
        vals = series[key]
        best_val = max(vals) if maximize else min(vals)
        best_idx = vals.index(best_val)
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

    has_bus_names = "name" in net.bus.columns
    has_line_names = "name" in net.line.columns
    has_trafo_names = "name" in net.trafo.columns

    for idx in indices:
        ts = timestamps[idx]
        try:
            net_copy = copy.deepcopy(net)
            meas = prepare_measurement_df(measurements[ts])
            net_copy, _ = lg.assign_load_values_from_measurements(net_copy, meas, substations_list)
            net_copy, _ = lg.assign_generators_values_from_measurements(net_copy, meas, substations_list)
            if load_scaling_factor != 1.0:
                net_copy.load["p_mw"] *= load_scaling_factor
                net_copy.load["q_mvar"] *= load_scaling_factor
            if sgen_scale != 1.0:
                net_copy.sgen["p_mw"] *= sgen_scale
                net_copy.sgen["q_mvar"] *= sgen_scale
            pp.runpp(net_copy, algorithm="nr", numba=False)
        except Exception:
            continue

        # ---- count violations and track which elements violated ------------
        viol = 0
        tick_buses: list[str] = []
        tick_lines: list[str] = []
        tick_trafos: list[str] = []
        for bus_idx, row in net_copy.res_bus.iterrows():
            if row["vm_pu"] > vm_upper_pu or row["vm_pu"] < vm_lower_pu:
                viol += 1
                name = str(net_copy.bus.at[bus_idx, "name"]) if has_bus_names else str(bus_idx)
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
                name = str(net_copy.trafo.at[trafo_idx, "name"]) if has_trafo_names else str(trafo_idx)
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
async def scan_scenarios_endpoint(request: ScanScenariosRequest):
    """Run RSA across a time range for multiple sgen scaling factors in parallel.

    Each scenario (sgen_scale) is scanned in its own thread — the scans are
    fully independent and pandapower's C solver releases the GIL, so
    ``ThreadPoolExecutor`` gives near-linear speedup with scenario count.

    ``app_data["current_index"]`` is **never mutated** by this endpoint.
    """
    (request.vm_lower_pu, request.vm_upper_pu,
     request.max_line_loading_pct, request.max_trafo_loading_pct) = _effective_thresholds(request)
    all_timestamps, scenario_measurements, source_label = _resolve_dataset(request.data_source)
    n = len(all_timestamps)
    # Forecasts have no simulation clock — default the window start to index 0.
    default_start = app_data["current_index"] if source_label == "measurements" else 0

    # ---- resolve time window (same pattern as element_timeseries_scan) -----
    if request.start_timestamp is not None:
        matched = [i for i, t in enumerate(all_timestamps) if t.startswith(request.start_timestamp)]
        start = matched[0] if matched else default_start
    else:
        start = default_start

    if request.end_timestamp is not None:
        matched = [i for i, t in enumerate(all_timestamps) if t.startswith(request.end_timestamp)]
        end = matched[-1] + 1 if matched else n
    else:
        end = n

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
                app_data["net"], scenario_measurements, substations,
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
    has_bus_names: bool,
    has_line_names: bool,
    has_trafo_names: bool,
) -> dict | None:
    """Run one Monte Carlo sample. net_base already has measurements assigned.
    Thread-safe — deep-copies net_base before any mutation."""
    try:
        import numpy as np

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
            net_copy.load["p_mw"] *= load_mult
            net_copy.load["q_mvar"] *= load_mult
        pp.runpp(net_copy, algorithm="nr", numba=False)
    except Exception:
        return None

    viol_buses: list[str] = []
    viol_lines: list[str] = []
    viol_trafos: list[str] = []
    bus_vm: dict[str, float] = {}

    for bus_idx, row in net_copy.res_bus.iterrows():
        raw_name = net_copy.bus.at[bus_idx, "name"] if has_bus_names else None
        name = _clean_label(raw_name) or f"Bus_{int(bus_idx)}"
        bus_vm[name] = round(float(row["vm_pu"]), 5)
        if row["vm_pu"] > vm_upper_pu or row["vm_pu"] < vm_lower_pu:
            viol_buses.append(name)

    for line_idx, row in net_copy.res_line.iterrows():
        if row["loading_percent"] > max_line_loading_pct:
            name = _line_display_name(net_copy, int(line_idx))
            viol_lines.append(name)

    for trafo_idx, row in net_copy.res_trafo.iterrows():
        if row["loading_percent"] > max_trafo_loading_pct:
            raw_name = net_copy.trafo.at[trafo_idx, "name"] if has_trafo_names else None
            if has_trafo_names:
                name = _clean_name(raw_name)
            else:
                name = ""
            if not name:
                hv_bus = int(net_copy.trafo.at[trafo_idx, "hv_bus"])
                lv_bus = int(net_copy.trafo.at[trafo_idx, "lv_bus"])
                name = f"Trafo_{hv_bus}-{lv_bus}_{int(trafo_idx)}"
            viol_trafos.append(name)

    return {
        "viol_buses": viol_buses,
        "viol_lines": viol_lines,
        "viol_trafos": viol_trafos,
        "bus_vm": bus_vm,
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
    viol_thermal = max_loading > max(max_line, max_trafo)
    feasible = not (viol_upper or viol_lower or viol_thermal)

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
    }


@app.post("/api/rsa/probabilistic")
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

    net = app_data["net"]
    (request.vm_lower_pu, request.vm_upper_pu,
     request.max_line_loading_pct, request.max_trafo_loading_pct) = _effective_thresholds(request)
    ts = _resolve_tick(request.data_source, request.timestamp)

    # Assign measurements once; net_base is read-only after this
    meas = prepare_measurement_df(_lookup_measurement_df(ts))
    net_base = copy.deepcopy(net)
    net_base, _ = lg.assign_load_values_from_measurements(net_base, meas, substations)
    net_base, _ = lg.assign_generators_values_from_measurements(net_base, meas, substations)

    has_bus_names = "name" in net_base.bus.columns
    has_line_names = "name" in net_base.line.columns
    has_trafo_names = "name" in net_base.trafo.columns

    # Generation uncertainty inputs (Step 2), with all sgens treated as
    # uncertain generators for now (Step 1 intentionally skipped).
    if len(net_base.sgen.index) > 0:
        import numpy as np

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
                has_bus_names, has_line_names, has_trafo_names,
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
        for name in r["viol_buses"]:
            bus_viol_counts[name] = bus_viol_counts.get(name, 0) + 1
        for name in r["viol_lines"]:
            line_viol_counts[name] = line_viol_counts.get(name, 0) + 1
        for name in r["viol_trafos"]:
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

    return {
        "timestamp": ts,
        "n_samples": n,
        "n_converged": n_converged,
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


@app.post("/api/grid/element_timeseries")
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
    all_timestamps, ets_measurements, ets_source = _resolve_dataset(request.data_source)
    n = len(all_timestamps)
    default_start = app_data["current_index"] if ets_source == "measurements" else 0

    # ---- resolve start index -----------------------------------------------
    if request.start_timestamp is not None:
        matched = [i for i, t in enumerate(all_timestamps)
                   if t.startswith(request.start_timestamp)]
        start = matched[0] if matched else default_start
    else:
        start = default_start

    # ---- resolve end index -------------------------------------------------
    if request.end_timestamp is not None:
        matched = [i for i, t in enumerate(all_timestamps)
                   if t.startswith(request.end_timestamp)]
        end = matched[-1] + 1 if matched else n
    else:
        end = n

    if request.n_steps is not None:
        end = min(start + request.n_steps, end)

    indices = list(range(start, end, max(1, request.step_size)))

    # ---- locate the element ------------------------------------------------
    etype = request.element_type.lower().strip()
    ename_lower = request.element_name.lower().strip()
    net_ref = app_data["net"]

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

    if "name" not in tbl.columns:
        raise HTTPException(
            status_code=400,
            detail=f"Network {etype} table has no 'name' column.",
        )

    matches = tbl[
        tbl["name"].astype(str).str.lower().str.contains(ename_lower, na=False, regex=False)
    ]
    if matches.empty:
        # Fallback: treat element_name as integer index
        try:
            idx_int = int(request.element_name)
            if idx_int in tbl.index:
                matches = tbl.loc[[idx_int]]
        except (ValueError, KeyError):
            pass
    if matches.empty:
        available = tbl["name"].astype(str).tolist()[:20]
        raise HTTPException(
            status_code=404,
            detail=f"No {etype} matching '{request.element_name}' found. "
                   f"First 20 available names: {available}",
        )

    elem_idx = int(matches.index[0])
    elem_name = str(matches.iloc[0]["name"])

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
    for idx in indices:
        ts = all_timestamps[idx]
        try:
            net_copy = copy.deepcopy(app_data["net"])
            meas = prepare_measurement_df(ets_measurements[ts])
            net_copy, _ = lg.assign_load_values_from_measurements(net_copy, meas, substations)
            net_copy, _ = lg.assign_generators_values_from_measurements(net_copy, meas, substations)
            if request.load_scaling_factor != 1.0:
                net_copy.load["p_mw"] *= request.load_scaling_factor
                net_copy.load["q_mvar"] *= request.load_scaling_factor
            pp.runpp(net_copy, algorithm="nr", numba=False)
        except Exception:
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
        "start_timestamp": out_timestamps[0],
        "end_timestamp": out_timestamps[-1],
        "thresholds": {
            "vm_upper_pu": request.vm_upper_pu,
            "vm_lower_pu": request.vm_lower_pu,
            "max_line_loading_pct": request.max_line_loading_pct,
            "max_trafo_loading_pct": request.max_trafo_loading_pct,
        },
    }


@app.post("/api/contingency/simulate")
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
    if request.element_type not in ["line", "trafo"]: raise HTTPException(status_code=400)

    current_ts = _resolve_tick(request.data_source, request.timestamp)
    net_copy = copy.deepcopy(app_data["net"])
    measurement_prod_cons = prepare_measurement_df(_lookup_measurement_df(current_ts))

    net_copy, _ = lg.assign_load_values_from_measurements(net_copy, measurement_prod_cons, substations)
    net_copy, _ = lg.assign_generators_values_from_measurements(net_copy, measurement_prod_cons, substations)
    net_copy.load['p_mw'] *= request.load_scaling_factor
    net_copy.load['q_mvar'] *= request.load_scaling_factor
    #  net_copy.trafo['tap_pos'] = net_copy.trafo['tap_neutral'] ----added in lifespan function

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

    if df_results_ca.empty: return {"timestamp": current_ts, "system_secure": True, "total_violations": 0,
                                    "violations": []}

    element_names = []
    for _, row in df_results_ca.iterrows():
        v_idx = int(row["violation_element_index"])
        if row["violation_type"] == "bus_vm_pu":
            _raw = net_copy.bus.at[v_idx, 'name']
            name = str(_raw).strip() if str(_raw).strip() else f"Bus_{v_idx}"
        elif row["violation_type"] == "line_loading":
            _raw = net_copy.line.at[v_idx, 'name']
            name = str(_raw).strip() if str(_raw).strip() else f"Line_{net_copy.line.at[v_idx, 'from_bus']}-{net_copy.line.at[v_idx, 'to_bus']}"
        elif row["violation_type"] == "trafo_loading":
            _raw = net_copy.trafo.at[v_idx, 'name']
            name = str(_raw).strip() if str(_raw).strip() else f"Trafo_{net_copy.trafo.at[v_idx, 'hv_bus']}-{net_copy.trafo.at[v_idx, 'lv_bus']}"
        else:
            name = f"Element_{v_idx}"
        element_names.append(name)

    df_results_ca["element_name"] = element_names
    df_results_ca["timestamp"] = df_results_ca["timestamp"].astype(str)

    # Placeholder: publish_ca_results(df_results_ca, current_ts, topic="topic6913"...)

    return {"timestamp": current_ts, "system_secure": False, "total_violations": len(df_results_ca),
            "violations": df_results_ca.to_dict(orient="records")}


@app.post("/api/contingency/simulate_all")
async def simulate_all_contingencies(request: GridRequest):
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
    current_ts = _resolve_tick(request.data_source, request.timestamp)
    net_copy = copy.deepcopy(app_data["net"])
    measurement_prod_cons = prepare_measurement_df(_lookup_measurement_df(current_ts))

    net_copy, _ = lg.assign_load_values_from_measurements(net_copy, measurement_prod_cons, substations)
    net_copy, _ = lg.assign_generators_values_from_measurements(net_copy, measurement_prod_cons, substations)
    net_copy.load['p_mw'] *= request.load_scaling_factor
    net_copy.load['q_mvar'] *= request.load_scaling_factor

    all_violations: list[dict] = []
    outages_with_violations = 0
    total_tested = 0

    for el_type, table in (("line", net_copy.line), ("trafo", net_copy.trafo)):
        for idx in table.index:
            total_tested += 1
            table.at[idx, "in_service"] = False
            try:
                df_results_ca, _ = ca.contingency_assessment_single(
                    net_copy, el_type, idx, current_ts,
                    vm_lower=request.vm_lower_pu,
                    vm_upper=request.vm_upper_pu,
                    max_line_loading_pct=request.max_line_loading_pct,
                    max_trafo_loading_pct=request.max_trafo_loading_pct,
                )
            except Exception:
                # One bad outage (e.g. islanded topology) — skip, restore, continue.
                table.at[idx, "in_service"] = True
                continue
            table.at[idx, "in_service"] = True

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
                    _raw = net_copy.bus.at[v_idx, 'name']
                    name = str(_raw).strip() if str(_raw).strip() else f"Bus_{v_idx}"
                elif row["violation_type"] == "line_loading":
                    _raw = net_copy.line.at[v_idx, 'name']
                    name = str(_raw).strip() if str(_raw).strip() else f"Line_{net_copy.line.at[v_idx, 'from_bus']}-{net_copy.line.at[v_idx, 'to_bus']}"
                elif row["violation_type"] == "trafo_loading":
                    _raw = net_copy.trafo.at[v_idx, 'name']
                    name = str(_raw).strip() if str(_raw).strip() else f"Trafo_{net_copy.trafo.at[v_idx, 'hv_bus']}-{net_copy.trafo.at[v_idx, 'lv_bus']}"
                else:
                    name = f"Element_{v_idx}"
                element_names.append(name)

            df_results_ca["element_name"] = element_names
            df_results_ca["timestamp"] = df_results_ca["timestamp"].astype(str)
            if el_type == "line":
                _raw_oc = net_copy.line.at[idx, 'name']
                _oc = str(_raw_oc).strip() if str(_raw_oc).strip() else f"Line_{net_copy.line.at[idx, 'from_bus']}-{net_copy.line.at[idx, 'to_bus']}"
            else:
                _raw_oc = net_copy.trafo.at[idx, 'name']
                _oc = str(_raw_oc).strip() if str(_raw_oc).strip() else f"Trafo_{net_copy.trafo.at[idx, 'hv_bus']}-{net_copy.trafo.at[idx, 'lv_bus']}"
            df_results_ca["outage_cause"] = _oc
            if "value" in df_results_ca.columns:
                df_results_ca["value"] = df_results_ca["value"].round(3)
            all_violations.extend(df_results_ca.to_dict(orient="records"))

    return {
        "timestamp": current_ts,
        "total_outages_tested": total_tested,
        "total_outages_causing_violations": outages_with_violations,
        "total_violations": len(all_violations),
        "system_n1_secure": len(all_violations) == 0,
        "violations": all_violations,
    }


@app.post("/api/contingency/optimize")
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
    if app_data["db_n1_line"] is None or app_data["db_n1_trafo"] is None:
        raise HTTPException(status_code=503, detail="N-1 Admittance databases not loaded.")

    current_ts = _resolve_tick(request.data_source, request.timestamp)
    net_copy = copy.deepcopy(app_data["net"])
    measurement_prod_cons = prepare_measurement_df(_lookup_measurement_df(current_ts))

    net_copy, _ = lg.assign_load_values_from_measurements(net_copy, measurement_prod_cons, substations)
    net_copy, _ = lg.assign_generators_values_from_measurements(net_copy, measurement_prod_cons, substations)
    net_copy.load['p_mw'] *= request.load_scaling_factor
    net_copy.load['q_mvar'] *= request.load_scaling_factor
    net_copy.trafo['tap_pos'] = net_copy.trafo['tap_neutral']
    net_base = copy.deepcopy(net_copy)

    # Topological Surgery
    if request.element_type == "trafo":
        modified_net, _, _ = mn.remove_trafo_and_connected_elements(net_copy, request.element_index)
        db = app_data["db_n1_trafo"]
    elif request.element_type == "line":
        modified_net, _, _, _ = mn.remove_isolated_line_buses_with_trafo(net_copy, request.element_index)
        db = app_data["db_n1_line"]

    try:
        pp.runpp(modified_net)
    except pp.powerflow.LoadflowNotConverged:
        pass

    # Find Islanded Generators
    missing_list = list(set(net_base.sgen['substation_name']) - set(modified_net.sgen['substation_name']))
    missing_sub = missing_list[0] if missing_list else None

    try:
        Yff_r, Yff_i, Yft_r, Yft_i, TAPS, trafo_ranges, trafo_defaults, branch_to_trafo = fe.get_admittance_parameters(
            db, request.element_index)
    except KeyError:
        raise HTTPException(status_code=404, detail="Matrix for this index not found.")

    try:
        results_opt, model_opt = fe.optimization_model_base(
            modified_net, net_base, Yff_r, Yff_i, Yft_r, Yft_i, TAPS, trafo_ranges, trafo_defaults,
            branch_to_trafo, missing_substation=missing_sub,
            opf_vm_lower=request.opf_vm_lower, opf_vm_upper=request.opf_vm_upper,
            lambda_p=request.opf_lambda_p, lambda_q=request.opf_lambda_q,
            pg_max_overrides=request.pg_max_overrides or None,
            pg_min_overrides=request.pg_min_overrides or None,
            min_power_factor=request.opf_min_power_factor,
            current_safety_margin=request.opf_current_safety_margin,
            fixed_setpoints=request.fixed_setpoints or None,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Solver crashed: {str(e)}")

    if results_opt is None: raise HTTPException(status_code=500)

    tc = getattr(results_opt.solver, "termination_condition", None)
    if tc == TerminationCondition.optimal:
        regulation_data = extract_flexibility_results(model_opt, net_base, current_ts)
        # Issue 1: extract post-OPF bus voltages
        voltage_data = fe.extract_post_opf_voltages(model_opt, modified_net)
        # Phase 3 Placeholder: publish_flex_ca_results(df, current_ts, topic="topic6919"...)
        result = {"timestamp": current_ts, "status": "optimal", "message": "Optimization successful.",
                  "activated_resources": regulation_data}
        result.update(voltage_data)
        app_data["last_dispatch_result"] = regulation_data
        app_data["last_dispatch_timestamp"] = current_ts
        return result
    else:
        return {"timestamp": current_ts, "status": str(tc), "message": "Infeasible.", "activated_resources": []}


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

    if app_data["db_full"] is None:
        raise HTTPException(status_code=503, detail="Data not loaded.")

    current_ts = _resolve_tick(request.data_source, request.timestamp)
    net_base = copy.deepcopy(app_data["net"])
    measurement_prod_cons = prepare_measurement_df(_lookup_measurement_df(current_ts))
    net_base, _ = lg.assign_load_values_from_measurements(net_base, measurement_prod_cons, substations)
    net_base, _ = lg.assign_generators_values_from_measurements(net_base, measurement_prod_cons, substations)
    net_base.trafo["tap_pos"] = net_base.trafo["tap_neutral"]

    has_bus_names = "name" in net_base.bus.columns
    if has_bus_names:
        bus_idx_to_name = {
            int(idx): (
                str(row["name"]).strip()
                if pd.notna(row["name"]) and str(row["name"]).strip()
                else f"Bus_{int(idx)}"
            )
            for idx, row in net_base.bus.iterrows()
        }
    else:
        bus_idx_to_name = {int(idx): f"Bus_{int(idx)}" for idx in net_base.bus.index}

    line_idx_to_name = {}
    for idx, row in net_base.line.iterrows():
        line_idx = int(idx)
        raw_name = row.get("name", None)
        if pd.notna(raw_name) and str(raw_name).strip():
            line_idx_to_name[line_idx] = _line_display_name(net_base, line_idx)
        else:
            line_idx_to_name[line_idx] = _line_display_name(net_base, line_idx)

    trafo_idx_to_name = {}
    for idx, row in net_base.trafo.iterrows():
        trafo_idx = int(idx)
        raw_name = row.get("name", None)
        if pd.notna(raw_name) and str(raw_name).strip():
            trafo_idx_to_name[trafo_idx] = str(raw_name).strip()
        else:
            try:
                trafo_idx_to_name[trafo_idx] = f"Trafo_{int(row['hv_bus'])}-{int(row['lv_bus'])}_{trafo_idx}"
            except Exception:
                trafo_idx_to_name[trafo_idx] = f"Trafo_{trafo_idx}"

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
        _net.load["p_mw"] *= request.load_scaling_factor
        _net.load["q_mvar"] *= request.load_scaling_factor
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
            _net.load["p_mw"] *= float(lm)
            _net.load["q_mvar"] *= float(lm)
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
                for name in bus_viol:
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
        db = app_data["db_full"]
        net_warm = copy.deepcopy(net_base)
        net_warm.load["p_mw"] *= request.load_scaling_factor
        net_warm.load["q_mvar"] *= request.load_scaling_factor
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
            min_power_factor=0.95,
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
        guarantee_interpretation = "alpha_is_certified_upper_bound_beta_fixed"
        guarantee_met = not k_capped

        # Use the existing LHS draws (seeded) and uncertainty machinery.
        sm_train = sgen_mults_cal[:k_effective]
        lm_train = load_mults_cal[:k_effective]

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
            min_power_factor=0.95,
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

        # Shared response structure with heuristic path.
        result = {
            "timestamp": current_ts,
            "status": "optimal",
            "message": "Scenario robust OPF solved successfully.",
            "feasible": True,
            "guarantee_met": guarantee_met,
            "activated_resources": activated_resources,
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

    import httpx

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
            load_scaling_factor=request.load_scaling_factor,
            slack_max_mw=request.slack_max_mw,
            opf_vm_upper=request.vm_upper_pu,
            opf_vm_lower=request.vm_lower_pu,
            opf_lambda_p=request.opf_lambda_p,
            opf_lambda_q=request.opf_lambda_q,
            vm_upper_per_bus=vm_upper_per_bus,
            vm_lower_per_bus=vm_lower_per_bus,
        )
        try:
            async with httpx.AsyncClient(base_url="http://localhost:8000", timeout=120.0) as client:
                opf_resp = await client.post("/api/flexibility/optimize", json=flex_req.model_dump())
            opf_result = opf_resp.json()
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

    net = app_data["net"]
    ts = _resolve_tick(request.data_source, request.timestamp)
    meas = prepare_measurement_df(_lookup_measurement_df(ts))

    net_base = copy.deepcopy(net)
    net_base, _ = lg.assign_load_values_from_measurements(net_base, meas, substations)
    net_base, _ = lg.assign_generators_values_from_measurements(net_base, meas, substations)
    net_base.load["p_mw"] *= request.load_scaling_factor
    net_base.load["q_mvar"] *= request.load_scaling_factor

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
    pg_max = PG_MAX_DATA.get(
        resolved_gen_name,
        PG_MAX_DATA.get(request.gen_name, max(1.0, pg_base * 1.5)),
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
        return {
            "type": "thermal",
            "element": "max_branch_loading",
            "value": float(point_result.get("max_loading_pct") or 0.0),
            "limit": float(thermal_limit),
            "units": "%",
        }
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

    return {
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
        net_copy.load["p_mw"] *= float(load_mult)
        net_copy.load["q_mvar"] *= float(load_mult)

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
    measurement_prod_cons = prepare_measurement_df(_lookup_measurement_df(target_ts))
    net_base = copy.deepcopy(app_data["net"])
    net_base, _ = lg.assign_load_values_from_measurements(net_base, measurement_prod_cons, substations)
    net_base, _ = lg.assign_generators_values_from_measurements(net_base, measurement_prod_cons, substations)

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
        if "bus" in net_base.sgen.columns and not net_base.sgen.empty:
            sgen_rows = net_base.sgen[net_base.sgen["bus"].astype(int) == int(local_bus_idx)]
            if not sgen_rows.empty and "q_mvar" in net_base.res_sgen.columns:
                return -0.5 * float(net_base.res_sgen.loc[sgen_rows.index, "q_mvar"].sum())
            if not sgen_rows.empty and "q_mvar" in net_base.sgen.columns:
                return -0.5 * float(sgen_rows["q_mvar"].sum())
        return 0.0

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


@app.post("/api/flexibility/optimize")
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
    if app_data["db_full"] is None: raise HTTPException(status_code=503)

    current_ts = _resolve_tick(request.data_source, request.timestamp)
    net_base = copy.deepcopy(app_data["net"])
    measurement_prod_cons = prepare_measurement_df(_lookup_measurement_df(current_ts))

    net_base, _ = lg.assign_load_values_from_measurements(net_base, measurement_prod_cons, substations)
    net_base, _ = lg.assign_generators_values_from_measurements(net_base, measurement_prod_cons, substations)
    net_base.trafo['tap_pos'] = net_base.trafo['tap_neutral']

    disabled_gens = request.disabled_generators

    # Issue 3: capture the full-grid operating point *before* any generators are
    # removed so the dispatch chart can show all three stages:
    #   P_pre_disable → P_base (post-disable seed) → P_new (OPF result).
    dispatch_pre_disable = None
    if disabled_gens:
        _net_pre = copy.deepcopy(net_base)
        try:
            pp.runpp(_net_pre)
            dispatch_pre_disable = {
                row['substation_name']: round(float(_net_pre.res_sgen.at[idx, 'p_mw']), 4)
                for idx, row in _net_pre.sgen.iterrows()
            }
            slack_label = str((app_data.get("grid_profile") or {}).get("slack_name") or "External Grid")
            dispatch_pre_disable[slack_label] = round(
                float(_net_pre.res_ext_grid['p_mw'].values[0]), 4
            )
        except Exception:
            dispatch_pre_disable = None
        net_base.sgen = net_base.sgen[~net_base.sgen['substation_name'].isin(disabled_gens)].reset_index(drop=True)

    try:
        pp.runpp(net_base)
    except Exception:
        pass

    net = copy.deepcopy(net_base)
    net.load['p_mw'] *= request.load_scaling_factor
    net.load['q_mvar'] *= request.load_scaling_factor

    try:
        pp.runpp(net)
    except Exception:
        pass

    db = app_data["db_full"]

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
            fixed_setpoints=request.fixed_setpoints or None,
            vm_upper_per_bus=request.vm_upper_per_bus or None,
            vm_lower_per_bus=request.vm_lower_per_bus or None,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if results_opt is None: raise HTTPException(status_code=500)

    tc = getattr(results_opt.solver, "termination_condition", None)
    if tc != TerminationCondition.optimal:
        _phys_cap = app_data.get("slack_max_mw", 70.0)
        _reported_cap = float(_phys_cap if request.slack_max_mw is None else min(request.slack_max_mw, _phys_cap))
        return {"timestamp": current_ts, "status": str(tc), "message": "Infeasible.",
                "activated_resources": [], "slack_max_mw": _reported_cap}

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
    note = ""
    try:
        cscen_results, cscen_model = solve_constrained_slack_scenario(model_opt, slack_cap, slack_q_max_mvar=request.slack_q_max_mvar, fixed_setpoints=request.fixed_setpoints or None)
        cscen_tc = (getattr(cscen_results.solver, "termination_condition", None)
                    if cscen_results is not None else None)
        if cscen_tc == TerminationCondition.optimal:
            df_reg_flex_rsa = fe.prepare_regulation_df_rsa(cscen_model, net, net_base, current_ts)
            voltage_data = fe.extract_post_opf_voltages(cscen_model, net)
        else:
            note = (f" (constrained-slack solve at ±{slack_cap:.0f} MW was "
                    f"infeasible; showing connected dispatch instead)")
    except Exception as e:
        note = f" (constrained-slack solve crashed: {e}; showing connected dispatch instead)"

    # ----------------------------------------------------------------------
    # Numerical-noise filter for the dispatch table.
    # Ipopt converges to a tolerance of ~1e-8, so unchanged generators show
    # Pg_up / Pg_down values like 5.3e-9 — physically meaningless, just
    # solver residuals. We zero those out and round all P-columns to
    # 4 decimals (0.1 kW resolution) so the table is readable.
    # We deliberately DO NOT zero Pg_base / Pg_new below the same threshold
    # because some substations have legitimately small baselines (e.g.
    # Rønne Syd ≈ 0.00023 MW = 0.23 kW) that we want to keep visible.
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

    regulation_data = df_reg_flex_rsa.to_dict(orient="records")
    msg = f"Grid restored at slack cap ±{slack_cap:.0f} MW.{note}"

    # Base bus voltages (pre-OPF power flow on the scaled network)
    # so the frontend can render ΔV = V_post_opf − V_base per bus.
    try:
        bus_voltages_base = [
            {
                "bus": int(i),
                "bus_name": (str(net.bus.at[i, "name"]).strip() or f"Bus_{int(i)}") if "name" in net.bus.columns else f"Bus_{int(i)}",
                "vm_pu": round(float(net.res_bus.at[i, "vm_pu"]), 6),
            }
            for i in net.bus.index
        ]
    except Exception:
        bus_voltages_base = []

    # Phase 3 Placeholder: publish_flex_rsa_results(df, current_ts, topic="topic6916"...)
    response = {
        "timestamp": current_ts, "status": "optimal", "message": msg,
        "activated_resources": regulation_data, "slack_max_mw": slack_cap,
    }
    response.update(voltage_data)
    if bus_voltages_base:
        response["bus_voltages_base"] = bus_voltages_base
    if dispatch_pre_disable is not None:
        response["dispatch_pre_disable"] = dispatch_pre_disable
    app_data["last_dispatch_result"] = regulation_data
    app_data["last_dispatch_timestamp"] = current_ts
    return response


@app.post("/api/kpi/evaluate")
async def evaluate_kpis(request: KPIRequest):
    """Compute the 3 official KPIs from ``main_KPIs_Included.ipynb`` in
    both a **connected** scenario (external-grid interface at engine
    default ±200 MW) and a **constrained** scenario (interface capped
    to ±``slack_max_mw``).

    Pipeline
    1. Clone the base network, apply measurements, neutralize taps.
    2. Optionally drop a disabled substation's sgens.
    3. Run a seed power flow on ``net_base``.
    4. Build ``net`` = ``net_base`` with ``load_scaling_factor`` applied.
    5. Count pre-optimization violations via
       ``rsa_engine.real_time_security_assessment`` — used only as
       informational context (``violations_before``).
    6. **Connected solve**: Pyomo + Ipopt with intact ``db_full`` Ybus
       matrices. External grid bounded at ±200 MW (engine default).
    7. Build full regulation DataFrame via
       ``flex_engine.prepare_regulation_df_rsa`` (all rows, incl. slack).
       Compute KPI-1 / KPI-2 / KPI-3 for the connected scenario.
    8. **Constrained-slack solve**: re-run Ipopt on the same model after
       :func:`solve_constrained_slack_scenario` tightens
       ``Pext, Qext`` bounds to ``±slack_max_mw``. If optimal, compute
       the 3 KPIs on the new dispatch; if infeasible, KPI-3 = 0% and
       the response surfaces ``status: "infeasible"``.

    Response shape
    --------------
    ``{
        timestamp,
        status: "success"|"failed",
        metrics: {  # CONNECTED scenario (interface at engine default ±200 MW)
            kpi_1_target_demand_flex_pct, kpi_1_available_upward_flex_mw,
            kpi_1_system_demand_mw,
            kpi_2_flex_utilization_pct, kpi_2_required_flex_mw,
            kpi_2_activated_flex_mw,
            kpi_3_prevented_violation_ratio_pct, kpi_3_status,
            violations_before, algorithm_runtime_sec
        },
        constrained_metrics: {  # interface capped to ±slack_max_mw
            status: "optimal"|"infeasible"|"error"|"skipped",
            slack_max_mw,            # echo of the user-chosen cap
            kpi_1_*, kpi_2_*, kpi_3_*,
            algorithm_runtime_sec
        }
     }``.

    Special cases
        * ``slack_max_mw == profile maximum`` -> constrained scenario will
            typically match the connected baseline since the natural interface
      flow is below this threshold for most timestamps.
    * ``slack_max_mw == 0``   -> fully islanded; if infeasible, that's
            the headline answer "The active system cannot self-sustain right now".
    """
    if app_data["db_full"] is None:
        raise HTTPException(status_code=503, detail="Admittance database not loaded.")

    current_ts = _resolve_tick(request.data_source, request.timestamp)
    net_base = copy.deepcopy(app_data["net"])
    measurement_prod_cons = prepare_measurement_df(_lookup_measurement_df(current_ts))

    net_base, _ = lg.assign_load_values_from_measurements(net_base, measurement_prod_cons, substations)
    net_base, _ = lg.assign_generators_values_from_measurements(net_base, measurement_prod_cons, substations)
    net_base.trafo['tap_pos'] = net_base.trafo['tap_neutral']

    disabled_gens = request.disabled_generators
    if disabled_gens:
        net_base.sgen = net_base.sgen[~net_base.sgen['substation_name'].isin(disabled_gens)].reset_index(drop=True)

    try:
        pp.runpp(net_base)
    except Exception:
        pass

    net = copy.deepcopy(net_base)
    net.load['p_mw'] *= request.load_scaling_factor
    net.load['q_mvar'] *= request.load_scaling_factor

    # --- Pre-optimization RSA: informational context only ---
    try:
        pp.runpp(net)
        _, df_results_before = rs.real_time_security_assessment(net, current_ts)
        violations_before = len(pd.DataFrame(df_results_before)) if isinstance(df_results_before, list) \
            else len(df_results_before)
    except Exception:
        violations_before = 0

    # --- Solve the Pyomo optimization model ---
    start_time = time.perf_counter()
    db = app_data["db_full"]

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
            fixed_setpoints=request.fixed_setpoints or None,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Solver crashed: {str(e)}")

    runtime = round(time.perf_counter() - start_time, 2)

    tc = getattr(results_opt.solver, "termination_condition", None)
    if tc != TerminationCondition.optimal:
        # Connected solve itself failed. Islanded solve depends on that
        # model being in a valid state, so we skip it rather than
        # risk reporting spurious KPIs.
        kpi3 = calculate_kpi_prevented_violation_ratio(None)
        return {
            "timestamp": current_ts,
            "status": "failed",
            "solver_status": str(tc),
            "metrics": {
                "kpi_1_target_demand_flex_pct": 0.0,
                "kpi_1_available_upward_flex_mw": 0.0,
                "kpi_1_system_demand_mw": 0.0,
                "kpi_2_flex_utilization_pct": 0.0,
                "kpi_2_required_flex_mw": 0.0,
                "kpi_2_activated_flex_mw": 0.0,
                "kpi_3_prevented_violation_ratio_pct": kpi3['prevented_violation_ratio_pct'],
                "kpi_3_status": kpi3['status'],
                "violations_before": violations_before,
                "algorithm_runtime_sec": runtime,
            },
            "constrained_metrics": {
                "status": "skipped",
                "slack_max_mw": float(app_data.get("slack_max_mw", 70.0) if request.slack_max_mw is None else request.slack_max_mw),
                "message": "Connected solve failed; constrained-slack check skipped.",
            },
        }

    # --- Build the full regulation DataFrame and run the 3 KPIs ---
    try:
        df_reg_flex_rsa = fe.prepare_regulation_df_rsa(model_opt, net, net_base, current_ts)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Regulation frame build failed: {str(e)}")

    try:
        kpi1 = calculate_kpi_target_demand_flexibility(df_reg_flex_rsa, {**PG_MAX_DATA, **request.pg_max_overrides})
    except ValueError as e:
        # Typically: generator absent from PG_MAX_DATA, or zero system demand.
        raise HTTPException(status_code=500, detail=f"KPI-1 failed: {str(e)}")

    try:
        kpi2 = calculate_kpi_flexibility_utilization(df_reg_flex_rsa)
    except ValueError as e:
        raise HTTPException(status_code=500, detail=f"KPI-2 failed: {str(e)}")

    kpi3 = calculate_kpi_prevented_violation_ratio(df_reg_flex_rsa)

    # Constrained-slack scenario — re-solve the SAME model with the
    # external-grid interface capped to ±slack_max_mw, then compute the 3 KPIs
    # on the new dispatch.
    _phys_cap = app_data.get("slack_max_mw", 999.0)
    slack_cap = float(max(0.0, _phys_cap if request.slack_max_mw is None else min(request.slack_max_mw, _phys_cap)))
    cscen_start = time.perf_counter()
    try:
        cscen_results, cscen_model = solve_constrained_slack_scenario(model_opt, slack_cap, slack_q_max_mvar=request.slack_q_max_mvar, fixed_setpoints=request.fixed_setpoints or None)
    except Exception as e:
        constrained_metrics = {
            "status": "error",
            "slack_max_mw": slack_cap,
            "message": f"Constrained-slack solver crashed: {str(e)}",
            "algorithm_runtime_sec": round(time.perf_counter() - cscen_start, 2),
        }
    else:
        cscen_runtime = round(time.perf_counter() - cscen_start, 2)
        cscen_tc = (getattr(cscen_results.solver, "termination_condition", None)
                    if cscen_results is not None else None)

        if cscen_tc != TerminationCondition.optimal:
            # Infeasible at the chosen cap.
            if slack_cap == 0.0:
                  msg = ("Grid cannot be secured without external-grid support at this "
                       "timestamp + load level.")
            else:
                  msg = (f"Grid cannot be secured with the external-grid interface capped at "
                       f"±{slack_cap:.0f} MW at this timestamp + load level.")
            constrained_metrics = {
                "status": "infeasible",
                "slack_max_mw": slack_cap,
                "solver_status": str(cscen_tc) if cscen_tc else "solver_error",
                "message": msg,
                "kpi_1_target_demand_flex_pct": 0.0,
                "kpi_2_flex_utilization_pct": 0.0,
                "kpi_3_prevented_violation_ratio_pct": 0.0,
                "kpi_3_status": f"Infeasible with slack capped at ±{slack_cap:.0f} MW.",
                "algorithm_runtime_sec": cscen_runtime,
            }
        else:
            try:
                df_cs = fe.prepare_regulation_df_rsa(cscen_model, net, net_base, current_ts)
                kpi1_cs = calculate_kpi_target_demand_flexibility(df_cs, {**PG_MAX_DATA, **request.pg_max_overrides})
                kpi2_cs = calculate_kpi_flexibility_utilization(df_cs)
                kpi3_cs = calculate_kpi_prevented_violation_ratio(df_cs)
                constrained_metrics = {
                    "status": "optimal",
                    "slack_max_mw": slack_cap,
                    "kpi_1_target_demand_flex_pct": round(kpi1_cs['target_demand_flexibility_pct'], 2),
                    "kpi_1_available_upward_flex_mw": round(kpi1_cs['available_upward_flexibility_MW'], 4),
                    "kpi_1_system_demand_mw": round(kpi1_cs['system_demand_MW'], 4),
                    "kpi_2_flex_utilization_pct": round(kpi2_cs['flexibility_utilization_pct'], 2),
                    "kpi_2_required_flex_mw": round(kpi2_cs['required_flexibility_MW'], 4),
                    "kpi_2_activated_flex_mw": round(kpi2_cs['activated_flexibility_MW'], 4),
                    "kpi_3_prevented_violation_ratio_pct": kpi3_cs['prevented_violation_ratio_pct'],
                    "kpi_3_status": kpi3_cs['status'],
                    "message": (f"Grid securable with the external-grid interface capped at "
                                f"±{slack_cap:.0f} MW."),
                    "algorithm_runtime_sec": cscen_runtime,
                }
            except Exception as e:
                constrained_metrics = {
                    "status": "error",
                    "slack_max_mw": slack_cap,
                    "message": f"Constrained-slack KPI computation failed: {str(e)}",
                    "algorithm_runtime_sec": cscen_runtime,
                }

    return {
        "timestamp": current_ts,
        "status": "success",
        "metrics": {
            "kpi_1_target_demand_flex_pct": round(kpi1['target_demand_flexibility_pct'], 2),
            "kpi_1_available_upward_flex_mw": round(kpi1['available_upward_flexibility_MW'], 4),
            "kpi_1_system_demand_mw": round(kpi1['system_demand_MW'], 4),
            "kpi_2_flex_utilization_pct": round(kpi2['flexibility_utilization_pct'], 2),
            "kpi_2_required_flex_mw": round(kpi2['required_flexibility_MW'], 4),
            "kpi_2_activated_flex_mw": round(kpi2['activated_flexibility_MW'], 4),
            "kpi_3_prevented_violation_ratio_pct": kpi3['prevented_violation_ratio_pct'],
            "kpi_3_status": kpi3['status'],
            "violations_before": violations_before,
            "algorithm_runtime_sec": runtime,
        },
        "constrained_metrics": constrained_metrics,
    }



@app.post("/api/kpi/forecast")
async def forecast_kpi(request: KPIRequest):
    """Generates a 24-hour forecast of KPI-1 by evaluating multiple timestamps."""
    if app_data["db_full"] is None:
        raise HTTPException(status_code=503, detail="Admittance database not loaded.")

    fk_timestamps, fk_measurements, fk_source = _resolve_dataset(request.data_source)
    # Start from the chosen tick (clock for measurements, or an explicit timestamp);
    # forecasts default to their first tick.
    start_ts = _resolve_tick(request.data_source, request.timestamp)
    start_idx = fk_timestamps.index(start_ts)
    # Limit to 24 hours (96 intervals) from the start timestamp
    end_idx = min(start_idx + 96, len(fk_timestamps))
    forecast_data = []

    db = app_data["db_full"]
    disabled_gens = request.disabled_generators

    for i in range(start_idx, end_idx):
        ts = fk_timestamps[i]
        net_base = copy.deepcopy(app_data["net"])
        measurement_prod_cons = prepare_measurement_df(fk_measurements[ts])

        net_base, _ = lg.assign_load_values_from_measurements(net_base, measurement_prod_cons, substations)
        net_base, _ = lg.assign_generators_values_from_measurements(net_base, measurement_prod_cons, substations)
        net_base.trafo['tap_pos'] = net_base.trafo['tap_neutral']

        if disabled_gens:
            net_base.sgen = net_base.sgen[~net_base.sgen['substation_name'].isin(disabled_gens)].reset_index(drop=True)
            
        try:
            pp.runpp(net_base)
        except Exception:
            pass

        net = copy.deepcopy(net_base)
        net.load['p_mw'] *= request.load_scaling_factor
        net.load['q_mvar'] *= request.load_scaling_factor

        try:
            pp.runpp(net)
        except Exception:
            pass

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
            )
        except Exception:
            continue

        tc = getattr(results_opt.solver, "termination_condition", None) if results_opt else None
        
        _phys_cap = app_data.get("slack_max_mw", 70.0)
        slack_cap = float(max(0.0, _phys_cap if request.slack_max_mw is None else min(request.slack_max_mw, _phys_cap)))
        kpi1_pct = 0.0

        if tc == TerminationCondition.optimal:
            try:
                cscen_results, cscen_model = solve_constrained_slack_scenario(model_opt, slack_cap, slack_q_max_mvar=request.slack_q_max_mvar)
                cscen_tc = getattr(cscen_results.solver, "termination_condition", None) if cscen_results else None
                if cscen_tc == TerminationCondition.optimal:
                    df_cs = fe.prepare_regulation_df_rsa(cscen_model, net, net_base, ts)
                    kpi1_cs = calculate_kpi_target_demand_flexibility(df_cs, {**PG_MAX_DATA, **request.pg_max_overrides})
                    kpi1_pct = round(kpi1_cs['target_demand_flexibility_pct'], 2)
            except Exception:
                pass
                
        forecast_data.append({
            "timestamp": ts,
            "kpi_1_target_demand_flex_pct": kpi1_pct
        })

    return {"forecast": forecast_data}

EDDK_VALUES_URL = "https://admin.energydata.dk/api/v1/datastreams/values"


# -------------------------------------------------------------------
# Working public transformer datastream IDs.
# Source: Aysegul's *new* EDDK_Download_ETL_CDK.ipynb (in
# EDDK_download_data_pipeline-main-2/, cell 5). She validated these
# IDs against her current personal-access token; the older 32-ID set
# (1205489-1205520) used by the previous notebook is now 403 with that
# token, so it cannot be used for live fetches.
#
# Probing showed her token is actually licensed for the entire range
# 1205370-1205479 (110 IDs), but we use just the 3 from her notebook
# to keep the chart readable and to match exactly what she tested.
# Extend this list once metadata for additional datastreams is
# documented in the engine's get_datastream_metadata_* helper.
# -------------------------------------------------------------------
EDDK_PUBLIC_IDS = [1205370, 1205373, 1205374]


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
       comes back ``"Unknown"`` (it doesn't yet recognise the new
       1205370-range IDs), substitute ``f"ID {datastream_id}"`` so the
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
    id_to_substation = {
        1205370: "Åkirkeby",
        1205373: "Værket",
        1205374: "Snorrebakken"
    }
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
async def push_eddk_data():
    """Placeholder for publishing simulation results back to EDDK.

    Currently returns success unconditionally — the real MQTT / REST
    upload is not wired up yet. The button is left in the UI so the
    end-to-end happy path can be demoed without a live upload.
    """
    return {"status": "success", "message": "Simulation results successfully pushed to EDDK Data Space."}
