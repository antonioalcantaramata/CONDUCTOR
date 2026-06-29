#!/usr/bin/env python
# coding: utf-8

# In[1]:


# ---------------------------------------------------------------------------------------
# Main Script 
# ---------------------------------------------------------------------------------------

# ===========================
# Import Required Libraries
# ===========================
#import re
import pandas as pd
import numpy as np
import pandapower as pp
import os
import copy
import math
import gc
from pyomo.environ import sqrt, value, sin, cos
from pyomo.environ import ConcreteModel, Set, Var, Param, Constraint,sin, cos, value, Any, Reals, Expression, Binary,  BuildAction
from pyomo.environ import Objective, minimize, value, SolverFactory, NonNegativeReals
from pyomo.environ import value
import pickle
from pyomo.util.infeasible import log_infeasible_constraints, log_infeasible_bounds
from pyomo.opt import TerminationCondition
import logging
logging.basicConfig(level=logging.INFO)


# ---------------------------------------------------------------------------
# Helper: extract Pg_max from the network object (T1)
# ---------------------------------------------------------------------------
def _extract_pg_max(net) -> dict:
    """Build {name -> Pg_max_mw} from net.sgen and net.gen.

    Naming convention (must match _gen_name_* helpers below):
    - sgen: use 'substation_name' col if present, else 'name', else 'sgen_{idx}'
    - gen:  use 'name' col if non-empty, else 'Gen_bus{bus_id}'
    max_p_mw = 0 is treated as missing (reactive-only machine) → fallback 1.5×p_mw.
    """
    def _clean(raw, fallback: str) -> str:
        s = str(raw).strip() if raw is not None else ""
        return s if s else fallback

    result = {}
    if not net.sgen.empty:
        key_col = "substation_name" if "substation_name" in net.sgen.columns else "name"
        for idx, row in net.sgen.iterrows():
            name = _clean(row.get(key_col, None), f"sgen_{idx}")
            max_p = row.get("max_p_mw", None) if "max_p_mw" in net.sgen.columns else None
            if max_p is not None and pd.notna(max_p) and float(max_p) > 0:
                result[name] = float(max_p)
            else:
                result[name] = max(float(row.get("p_mw", 0.0)) * 1.5, 0.001)
    if not net.gen.empty:
        for idx, row in net.gen.iterrows():
            bus_id = int(row.get("bus", idx))
            name = _clean(row.get("name", None), f"Gen_bus{bus_id}")
            max_p = row.get("max_p_mw", None) if "max_p_mw" in net.gen.columns else None
            if max_p is not None and pd.notna(max_p) and float(max_p) > 0:
                result[name] = float(max_p)
            else:
                result[name] = max(float(row.get("p_mw", 0.0)) * 1.5, 0.001)
    return result


def active_power_balance_rule(m, bus_idx):
    # --- Contributions from generators, external grids, and shunts ---
    P_inj = sum(m.Pg_new[g] for g in m.G if m.gen_bus[g] == bus_idx) \
          + sum(m.Pext[e] for e in m.Ext if m.ext_bus[e] == bus_idx) \
          - sum(m.Pl[l] for l in m.L if m.load_bus[l] == bus_idx) \
          - sum(m.Ps_nom[s] for s in m.S if m.shunt_bus[s] == bus_idx)

    # --- Branch outflows ---
    P_flow = sum(
        m.V[i]**2 * m.Yff_real_used[i, j] +
        m.V[i]*m.V[j]*(
            m.Yft_real_used[i, j]*cos(m.theta[i]-m.theta[j]) +
            m.Yft_imag_used[i, j]*sin(m.theta[i]-m.theta[j])
        )
        for (i, j) in m.Branches if i == bus_idx
    )

    # Convert per-unit flows to MW. For buses with no connected model terms,
    # Pyomo can simplify to a plain bool (e.g. 0 == 0), which is invalid.
    expr = P_inj == P_flow * 100
    if isinstance(expr, (bool, np.bool_)):
        return Constraint.Feasible if expr else Constraint.Infeasible
    return expr


def reactive_power_balance_rule(m, bus_idx):
    # --- Contributions from generators, external grids, loads, and shunts ---
    Q_inj = sum(m.Qg_new[g] for g in m.G if m.gen_bus[g] == bus_idx) \
          + sum(m.Qext[e] for e in m.Ext if m.ext_bus[e] == bus_idx) \
          - sum(m.Ql[l] for l in m.L if m.load_bus[l] == bus_idx) 
            #\- sum(m.Qs[s] for s in m.S if m.shunt_bus[s] == bus_idx)

    # --- Add shunt reactive injection (voltage-dependent) ---
    # Qs_nom is Mvar at V=1.0; actual Q = Qs_nom * V^2
    Q_shunts_at_bus = sum(m.Qs_nom[s] * (m.V[bus_idx]**2) for s in m.S if m.shunt_bus[s] == bus_idx)
    # In your original code you subtracted model.Qs; keep same sign convention:
    Q_inj = Q_inj - Q_shunts_at_bus


    # --- Branch outflows ---
    Q_flow = sum(
        - m.V[i]**2 * m.Yff_imag_used[i, j]
        - m.V[i]*m.V[j]*(
            m.Yft_imag_used[i, j]*cos(m.theta[i]-m.theta[j])
            - m.Yft_real_used[i, j]*sin(m.theta[i]-m.theta[j])
        )
        for (i, j) in m.Branches if i == bus_idx
    )

    # Convert per-unit flows to MVAr. Guard against trivial bool expressions.
    expr = Q_inj == Q_flow * 100
    if isinstance(expr, (bool, np.bool_)):
        return Constraint.Feasible if expr else Constraint.Infeasible
    return expr


# Current from i->j
def branch_current_from_rule(m, i, j):
    I_real = m.Yff_real_used[i,j]*m.V[i]*cos(m.theta[i]) - m.Yff_imag_used[i,j]*m.V[i]*sin(m.theta[i]) \
             + m.Yft_real_used[i,j]*m.V[j]*cos(m.theta[j]) - m.Yft_imag_used[i,j]*m.V[j]*sin(m.theta[j])
    I_imag = m.Yff_real_used[i,j]*m.V[i]*sin(m.theta[i]) + m.Yff_imag_used[i,j]*m.V[i]*cos(m.theta[i]) \
             + m.Yft_real_used[i,j]*m.V[j]*sin(m.theta[j]) + m.Yft_imag_used[i,j]*m.V[j]*cos(m.theta[j])
    
    current_pu_sq = I_real**2 + I_imag**2
    max_current_pu_sq = (m.I_from_max[i, j] / m.Ibase_from_kA[i, j])**2
    return current_pu_sq <= 0.9 * max_current_pu_sq
    #return current_pu_sq <= max_current_pu_sq


# Current from j->i
def branch_current_to_rule(m, i, j):
    I_real = m.Yff_real_used[j,i]*m.V[j]*cos(m.theta[j]) - m.Yff_imag_used[j,i]*m.V[j]*sin(m.theta[j]) \
             + m.Yft_real_used[j,i]*m.V[i]*cos(m.theta[i]) - m.Yft_imag_used[j,i]*m.V[i]*sin(m.theta[i])
    I_imag = m.Yff_real_used[j,i]*m.V[j]*sin(m.theta[j]) + m.Yff_imag_used[j,i]*m.V[j]*cos(m.theta[j]) \
             + m.Yft_real_used[j,i]*m.V[i]*sin(m.theta[i]) + m.Yft_imag_used[j,i]*m.V[i]*cos(m.theta[i])

    current_pu_sq = I_real**2 + I_imag**2
    max_current_pu_sq = (m.I_from_max[j, i] / m.Ibase_from_kA[j, i])**2
    return current_pu_sq <= 0.9 * max_current_pu_sq
    #return current_pu_sq <= max_current_pu_sq


# --- Objective function ---
def objective_rule(m):
    # 1️⃣ Generation deviation penalties
    p_dev_sq = sum((m.Pg_new[g] - m.Pg_base[g])**2 for g in m.G)
    q_dev_sq = sum((m.Qg_new[g] - m.Qg_base[g])**2 for g in m.G)
    
    
    # 3️⃣ Weights
    lambda_p = 0.01
    lambda_q = 0.001
    
    return lambda_p * p_dev_sq + lambda_q * q_dev_sq 



def optimization_model_base(net, net_base, Yff_r, Yff_i, Yft_r, Yft_i, TAPS, trafo_ranges, trafo_defaults, branch_to_trafo,
                       missing_substation=None, opf_vm_lower: float = 0.95, opf_vm_upper: float = 1.05,
                       lambda_p: float = 0.01, lambda_q: float = 0.001,
                       pg_max_overrides: dict = None, pg_min_overrides: dict = None,
                       min_power_factor: float = 0.95,
                       current_safety_margin: float = 0.9,
                       fixed_setpoints: dict = None,
                       vm_upper_per_bus: dict = None,
                       vm_lower_per_bus: dict = None):

    
    # -------------------------
    # Step 1: Create Pyomo model
    # -------------------------
    model = ConcreteModel()
    
    # -------------------------
    # Step 2: Define bus set
    # -------------------------
    # Model only electrically active buses. Uploaded/custom networks can carry
    # auxiliary offline buses (in_service=False) that have no branch equations
    # and would otherwise yield uninitialized variables.
    if "in_service" in net.bus.columns:
        bus_indices = [int(i) for i in net.bus.index if bool(net.bus.at[i, "in_service"])]
    else:
        bus_indices = [int(i) for i in net.bus.index]
    # Always keep slack bus in the model set.
    for _sb in net.ext_grid["bus"].tolist() if "bus" in net.ext_grid.columns else []:
        _sb_i = int(_sb)
        if _sb_i not in bus_indices:
            bus_indices.append(_sb_i)
    # Create a Pyomo set of buses
    model.B = Set(initialize=bus_indices)
    
    # -------------------------
    # Step 3: Store base-case results from power flow
    # -------------------------
    vm_pu_map = {}   # voltage magnitudes (p.u.)
    va_rad_map = {}  # voltage angles (radians)
    
    for idx in net.bus.index:
        b = int(idx)
        vm_pu_map[b] = float(net.res_bus.at[idx, 'vm_pu'])         # p.u. magnitude
        va_deg = float(net.res_bus.at[idx, 'va_degree'])           # degrees
        va_rad_map[b] = math.radians(va_deg)                       # convert to radians
    
    # -------------------------
    # Step 4: Voltage magnitude and angle variables
    # -------------------------
    # Bounds for voltage magnitudes (if too tight, problem may become infeasible)
    V_min = opf_vm_lower
    V_max = opf_vm_upper

    # Decision variables:
    # V[b]   = voltage magnitude at bus b
    # theta[b] = voltage angle at bus b
    model.V = Var(model.B, bounds=(V_min, V_max))
    model.theta = Var(model.B, bounds=(-math.pi, math.pi))

    # Per-bus bound overrides (back-off / robust OPF support).
    # Keys are bus names from API payloads; handle blank names via fallback labels.
    bus_name_to_idx = {}
    for idx, row in net.bus.iterrows():
        b_idx = int(idx)
        raw_name = row.get('name', None)
        if pd.notna(raw_name) and str(raw_name).strip():
            bus_name_to_idx[str(raw_name).strip()] = b_idx
        bus_name_to_idx[f"Bus_{b_idx}"] = b_idx

    # Collect target bounds first, then apply once to avoid ordering issues.
    per_bus_lb = {}
    per_bus_ub = {}
    if vm_upper_per_bus:
        for key, ub in vm_upper_per_bus.items():
            b_idx = bus_name_to_idx.get(str(key).strip(), None)
            if b_idx is not None and b_idx in model.B:
                per_bus_ub[b_idx] = max(V_min, min(V_max, float(ub)))
    if vm_lower_per_bus:
        for key, lb in vm_lower_per_bus.items():
            b_idx = bus_name_to_idx.get(str(key).strip(), None)
            if b_idx is not None and b_idx in model.B:
                per_bus_lb[b_idx] = min(V_max, max(V_min, float(lb)))

    for b_idx in model.B:
        lb = per_bus_lb.get(b_idx, V_min)
        ub = per_bus_ub.get(b_idx, V_max)
        if lb > ub:
            lb = ub
        model.V[b_idx].setlb(lb)
        model.V[b_idx].setub(ub)
    
    # -------------------------
    # Step 5: Slack bus (reference angle)
    # -------------------------
    # External grid is used as slack bus.
    # Both angle AND voltage magnitude are fixed at the measured base-case values:
    #   - theta is fixed to 0 (reference angle)
    #   - V is fixed to the measured p.u. so the OPF cannot treat the
    #     external grid voltage as a free lever to absorb local overvoltages.
    slack_bus = int(net.ext_grid['bus'].iloc[0])
    model.theta[slack_bus].fix(va_rad_map[slack_bus])
    model.V[slack_bus].fix(vm_pu_map[slack_bus])

        
    # -------------------------
    # Step 6: Prepare load data
    # -------------------------
    
    # Extract all load names from the network (substation names)
    load_name_list = net.load['substation_name'].tolist()
    # Active power demand (MW) per load
    load_data_p = dict(zip(net.load['substation_name'], net.load['p_mw']))
    
    # Reactive power demand (Mvar) per load
    load_data_q = dict(zip(net.load['substation_name'], net.load['q_mvar']))
    
    # Map each load to its bus number
    load_bus_map = dict(zip(net.load['substation_name'], net.load['bus']))
    
    # -------------------------
    # Step 7: Define load set and parameters in Pyomo
    # -------------------------
    
    # Set of loads (indexed by load names)
    model.L = Set(initialize=load_name_list)
    
    # Parameters for load data
    model.Pl = Param(model.L, initialize=load_data_p)       # Active power demand (MW)
    model.Ql = Param(model.L, initialize=load_data_q)       # Reactive power demand (Mvar)
    model.load_bus = Param(model.L, initialize=load_bus_map) # Bus index for each load

    
    # -------------------------
    # Step 8: Prepare generator data  (T1+T2: merged net.sgen + net.gen)
    # -------------------------

    def _sgen_name(row, idx) -> str:
        key_col = "substation_name" if "substation_name" in net.sgen.columns else "name"
        raw = row.get(key_col, None)
        s = str(raw).strip() if raw is not None else ""
        return s if s else f"sgen_{idx}"

    def _gen_name(row, idx) -> str:
        raw = row.get("name", None)
        s = str(raw).strip() if raw is not None else ""
        return s if s else f"Gen_bus{int(row.get('bus', idx))}"

    gen_name_list: list = []
    P_base_data:   dict = {}
    Q_base_data:   dict = {}
    gen_bus_map:   dict = {}

    # --- Static generators (measured renewables / IEEE sgen) ---
    if not net.sgen.empty:
        for idx, row in net.sgen.iterrows():
            name = _sgen_name(row, idx)
            gen_name_list.append(name)
            res_p = net.res_sgen.at[idx, "p_mw"]   if not net.res_sgen.empty and idx in net.res_sgen.index else float(row.get("p_mw",   0.0))
            res_q = net.res_sgen.at[idx, "q_mvar"] if not net.res_sgen.empty and idx in net.res_sgen.index else float(row.get("q_mvar", 0.0))
            P_base_data[name] = float(res_p)
            Q_base_data[name] = float(res_q)
            gen_bus_map[name]  = int(row["bus"])

    # --- Controllable generators (IEEE net.gen; absent in measured substation profiles) ---
    condenser_q_max: dict = {}
    condenser_q_min: dict = {}
    if not net.gen.empty:
        for idx, row in net.gen.iterrows():
            name = _gen_name(row, idx)
            if name in gen_name_list:
                continue  # avoid double-counting
            gen_name_list.append(name)
            res_p = net.res_gen.at[idx, "p_mw"]   if not net.res_gen.empty and idx in net.res_gen.index else float(row.get("p_mw",   0.0))
            res_q = net.res_gen.at[idx, "q_mvar"] if not net.res_gen.empty and idx in net.res_gen.index else float(row.get("q_mvar", 0.0))
            # Collect Q limits before storing the base Q so we can clip it.
            # pandapower runpp does NOT enforce Q limits for PV buses, so the raw
            # power-flow Q can exceed Qmax for synchronous condensers.  Clipping
            # here gives a physically meaningful baseline in the dispatch charts.
            try:
                mxq = float(row.get("max_q_mvar", 50.0))
                mxq = mxq if not math.isnan(mxq) else 50.0
            except (TypeError, ValueError):
                mxq = 50.0
            try:
                mnq = float(row.get("min_q_mvar", -50.0))
                mnq = mnq if not math.isnan(mnq) else -50.0
            except (TypeError, ValueError):
                mnq = -50.0
            condenser_q_max[name] = mxq
            condenser_q_min[name] = mnq
            # Clip base Q to declared limits.
            res_q = max(mnq, min(mxq, float(res_q)))
            P_base_data[name] = float(res_p)
            Q_base_data[name] = float(res_q)
            gen_bus_map[name]  = int(row["bus"])

    # Build Pg_max from the network object (replaces hardcoded profile-specific limits)
    _pg_max_all = _extract_pg_max(net)
    Pg_max_data = {g: _pg_max_all.get(g, 0.001) for g in gen_name_list}
    Pg_min_data = {g: 0.0 for g in gen_name_list}

    # Synchronous condensers produce zero active power — fix Pg at 0
    for g in condenser_q_max:
        Pg_max_data[g] = 0.0

    if missing_substation:
        _to_remove = [missing_substation] if isinstance(missing_substation, str) else list(missing_substation)
        for _sub in _to_remove:
            Pg_max_data.pop(_sub, None)
            Pg_min_data.pop(_sub, None)

    # Apply per-substation overrides from the API request (if provided)
    if pg_max_overrides:
        Pg_max_data.update(pg_max_overrides)
    if pg_min_overrides:
        Pg_min_data.update(pg_min_overrides)
    
    # -------------------------
    # Step 9: Define generator set and parameters in Pyomo
    # -------------------------
    
    # Set of generators
    model.G = Set(initialize=gen_name_list)
    
    # Parameters: base values and limits
    model.Pg_base = Param(model.G, initialize=P_base_data)   # Base active power
    model.Pg_min = Param(model.G, initialize=Pg_min_data)    # Minimum active power
    model.Pg_max = Param(model.G, initialize=Pg_max_data)    # Maximum active power
    model.Qg_base = Param(model.G, initialize=Q_base_data)   # Base reactive power
    model.gen_bus = Param(model.G, initialize=gen_bus_map)   # Generator → bus mapping
    
    # -------------------------
    # Step 10: Define decision variables
    # -------------------------
    
    # Active power bounds depend on generator-specific min/max
    def pg_bounds_rule(m, g):
        return (m.Pg_min[g], m.Pg_max[g])
    
    # Decision variables: optimized generator outputs
    model.Pg_new = Var(model.G, bounds=pg_bounds_rule)  # Active power (MW)

    ### New code fro Qg_new
    ###################################################
    # Reactive power variable (no bounds here)
    model.Qg_new = Var(model.G)

    # -------------------------
    # PF coupling parameters
    # -------------------------
    PF_min = min_power_factor
    tan_phi = math.tan(math.acos(PF_min))
    tan_phi_sq = tan_phi ** 2
    model.tan_phi_sq = Param(initialize=tan_phi_sq)

    # -------------------------
    # PF constraint: Q^2 <= tan^2(phi) * P^2  — applied only to real generators.
    # Synchronous condensers (Pg_max < 1 MW) are excluded: the cone would force Q=0
    # because their P is fixed at 0, removing all reactive voltage support.
    # Instead they get explicit Q-limit constraints from their net.gen data.
    # -------------------------
    pf_gens   = [g for g in gen_name_list if Pg_max_data.get(g, 0.0) >= 1.0]
    cond_gens = [g for g in gen_name_list if Pg_max_data.get(g, 0.0) <  1.0 and g in condenser_q_max]

    model.PF_set = Set(initialize=pf_gens)
    def pf_cone(m, g):
        return m.Qg_new[g] ** 2 <= m.tan_phi_sq * m.Pg_new[g] ** 2
    model.PF_cone = Constraint(model.PF_set, rule=pf_cone)

    # Q-limit constraints for synchronous condensers
    if cond_gens:
        model.CondSet = Set(initialize=cond_gens)
        model.Qg_cond_max = Param(model.CondSet, initialize={g: condenser_q_max[g] for g in cond_gens})
        model.Qg_cond_min = Param(model.CondSet, initialize={g: condenser_q_min[g] for g in cond_gens})
        model.QgCondUpper = Constraint(model.CondSet, rule=lambda m, g: m.Qg_new[g] <= m.Qg_cond_max[g])
        model.QgCondLower = Constraint(model.CondSet, rule=lambda m, g: m.Qg_new[g] >= m.Qg_cond_min[g])
    

    
    shunt_name_list = net.shunt['name'].tolist()
    shunt_bus_map = dict(zip(net.shunt['name'], net.shunt['bus']))
    
    # NOMINAL reactive power (Mvar) at nominal voltage (this is NOT the actual Q at solved V)
    shunt_Q_nom = dict(zip(net.shunt['name'], net.shunt['q_mvar'].fillna(0.0)))
    
    # If you also need P (rare), collect it similarly
    shunt_P_nom = dict(zip(net.shunt['name'], net.shunt['p_mw'].fillna(0.0)))
    
    model.S = Set(initialize=shunt_name_list)
    model.shunt_bus = Param(model.S, initialize=shunt_bus_map)
    model.Qs_nom = Param(model.S, initialize=shunt_Q_nom)   # nominal Q (Mvar at V=1.0)
    model.Ps_nom = Param(model.S, initialize=shunt_P_nom)   # nominal P (MW at V=1.0), if any

    
    # ---------------------------------------------------
    # Step 3: Extract external grid data from pandapower
    # ---------------------------------------------------

    # List of external grid names — sanitise NaN (MATPOWER has no names) to "ExtGrid_0" etc.
    _raw_names = net.ext_grid['name'].tolist()
    ext_grid_name_list = [
        (n if (isinstance(n, str) and n.strip()) else f"ExtGrid_{i}")
        for i, n in enumerate(_raw_names)
    ]
    # Keep the sanitised names in the DataFrame so all downstream dicts stay consistent.
    net.ext_grid = net.ext_grid.copy()
    net.ext_grid['name'] = ext_grid_name_list
    net_base.ext_grid = net_base.ext_grid.copy()
    net_base.ext_grid['name'] = ext_grid_name_list

    # Dictionaries: name -> active/reactive power from loadflow results
    ext_grid_data_p = dict(zip(ext_grid_name_list, net_base.res_ext_grid['p_mw']))
    ext_grid_data_q = dict(zip(ext_grid_name_list, net_base.res_ext_grid['q_mvar']))

    # Mapping: external grid name -> bus index
    ext_grid_bus_map = dict(zip(ext_grid_name_list, net_base.ext_grid['bus']))

    # Build Pext bounds from net.ext_grid limits (read from .m file Pmin/Pmax).
    # Fall back to ±200 MW when columns are absent, NaN, or degenerate (lo >= hi).
    # Q limits from MATPOWER describe the *generator* at that bus, not the reactive
    # capacity of the external network connection — always use ±200 Mvar for Qext.
    _pext_lo_map: dict = {}
    _pext_hi_map: dict = {}
    for name, (_, row) in zip(ext_grid_name_list, net.ext_grid.iterrows()):
        lo = float(row['min_p_mw']) if ('min_p_mw' in net.ext_grid.columns and pd.notna(row.get('min_p_mw'))) else -200.0
        hi = float(row['max_p_mw']) if ('max_p_mw' in net.ext_grid.columns and pd.notna(row.get('max_p_mw'))) else  200.0
        if hi <= lo or (lo == 0.0 and hi == 0.0):
            lo, hi = -200.0, 200.0
        _pext_lo_map[name] = lo
        _pext_hi_map[name] = hi

    def Pext_bounds_rule(m, e):
        return (_pext_lo_map.get(e, -200.0), _pext_hi_map.get(e, 200.0))

    def Qext_bounds_rule(m, e):
        return (-200.0, 200.0)
        
    # Set of all external grids
    model.Ext = Set(initialize=ext_grid_name_list)
    
    # Param: map ext_grid name -> bus
    model.ext_bus = Param(model.Ext, initialize=ext_grid_bus_map)
    
    model.Pext = Var(model.Ext, bounds=Pext_bounds_rule, initialize=ext_grid_data_p)
    model.Qext = Var(model.Ext, bounds=Qext_bounds_rule, initialize=ext_grid_data_q)
    # Fixed active and reactive injections (MW / Mvar)
    #model.Pext = Param(model.Ext, initialize=ext_grid_data_p)
    #model.Qext = Param(model.Ext, initialize=ext_grid_data_q)

        
    # ---------------------------
    # Prepare branch current info (unchanged)
    # ---------------------------
    Sbase_MVA = net.sn_mva  # system base power
    trafo_current_data = {}
    
    for trafo_idx, trafo_data in net.trafo.iterrows():
        i = trafo_data['hv_bus']
        j = trafo_data['lv_bus']
    
        sn_mva = trafo_data['sn_mva']
        Vbase_hv_kV = net.bus.at[i, 'vn_kv']
        Vbase_lv_kV = net.bus.at[j, 'vn_kv']
    
        # Rated currents
        Irated_hv_kA = sn_mva / (np.sqrt(3) * Vbase_hv_kV)
        Irated_lv_kA = sn_mva / (np.sqrt(3) * Vbase_lv_kV)
    
        # Base currents
        Ibase_hv_kA = Sbase_MVA / (np.sqrt(3) * Vbase_hv_kV)
        Ibase_lv_kA = Sbase_MVA / (np.sqrt(3) * Vbase_lv_kV)
    
        trafo_current_data[(i, j)] = {
            'Ibase_from_kA': Ibase_hv_kA,
            'I_from_max': Irated_hv_kA
        }
        trafo_current_data[(j, i)] = {
            'Ibase_from_kA': Ibase_lv_kA,
            'I_from_max': Irated_lv_kA
        }
    
    line_current_data = {}
    for line_idx, line_data in net.line.iterrows():
        i = line_data['from_bus']
        j = line_data['to_bus']
    
        Vbase_from_kV = net.bus.at[i, 'vn_kv']
        Vbase_to_kV = net.bus.at[j, 'vn_kv']
    
        Ibase_from_kA = Sbase_MVA / (np.sqrt(3) * Vbase_from_kV)
        Ibase_to_kA = Sbase_MVA / (np.sqrt(3) * Vbase_to_kV)
    
        I_from_max = line_data['max_i_ka']
        I_to_max = line_data['max_i_ka']
    
        line_current_data[(i, j)] = {
            'Ibase_from_kA': Ibase_from_kA,
            'I_from_max': I_from_max
        }
        line_current_data[(j, i)] = {
            'Ibase_from_kA': Ibase_to_kA,
            'I_from_max': I_to_max
        }
    
    # merge
    branch_data = {}
    branch_data.update(line_current_data)
    branch_data.update(trafo_current_data)
        
    
    # ============================================================
    # --- Fixed transformer taps: set admittances from defaults ---
    # ============================================================
    # Use the same branches_list you created earlier (branch_data.keys()).
    branches_list = list(branch_data.keys())
    model.Branches = Set(initialize=branches_list, dimen=2)

    # Variables to store used admittances (kept, but will be fixed)
    model.Yff_real_used = Var(model.Branches)
    model.Yff_imag_used = Var(model.Branches)
    model.Yft_real_used = Var(model.Branches)
    model.Yft_imag_used = Var(model.Branches)

    # Fix Y values: for transformer branches use trafo_defaults; for lines use tap = 1 (base)
    for (i, j) in branches_list:
     
        if (i, j) in branch_to_trafo:
            # Transformer branch — get its trafo index and neutral tap
            trafo_idx = branch_to_trafo[(i, j)]
    
            # Get neutral tap position from pandapower (default to 0 if not available)
            if "tap_neutral" in net.trafo.columns:
                t_default = net.trafo.at[trafo_idx, "tap_neutral"]
            else:
                t_default = 0
    
            # Make sure it's an integer key present in your Y dictionaries
            t_default = int(t_default)

            model.Yff_real_used[i, j].fix(Yff_r[(i, j), t_default])
            model.Yff_imag_used[i, j].fix(Yff_i[(i, j), t_default])
            model.Yft_real_used[i, j].fix(Yft_r[(i, j), t_default])
            model.Yft_imag_used[i, j].fix(Yft_i[(i, j), t_default])
        else:
            # Line branch — use the base (tap=1) admittance you've used elsewhere
            base_t = 1
            model.Yff_real_used[i, j].fix(Yff_r[(i, j), base_t])
            model.Yff_imag_used[i, j].fix(Yff_i[(i, j), base_t])
            model.Yft_real_used[i, j].fix(Yft_r[(i, j), base_t])
            model.Yft_imag_used[i, j].fix(Yft_i[(i, j), base_t])

    
    # ============================================================
    # --- 8. Branch current parameters (lines + trafos) ---
    # ============================================================
    model.Ibase_from_kA = Param(
        model.Branches,
        initialize={(i, j): branch_data[(i, j)]['Ibase_from_kA'] for (i, j) in branches_list},
        within=Reals
    )
    
    model.I_from_max = Param(
        model.Branches,
        initialize={(i, j): branch_data[(i, j)]['I_from_max'] for (i, j) in branches_list},
        within=Reals
    )
    

    model.active_balance = Constraint(model.B, rule=active_power_balance_rule)

    # Add constraint to model
    model.reactive_balance = Constraint(model.B, rule=reactive_power_balance_rule)

    _csm = float(current_safety_margin)

    def branch_current_from_rule(m, i, j):
        I_real = m.Yff_real_used[i,j]*m.V[i]*cos(m.theta[i]) - m.Yff_imag_used[i,j]*m.V[i]*sin(m.theta[i]) \
                 + m.Yft_real_used[i,j]*m.V[j]*cos(m.theta[j]) - m.Yft_imag_used[i,j]*m.V[j]*sin(m.theta[j])
        I_imag = m.Yff_real_used[i,j]*m.V[i]*sin(m.theta[i]) + m.Yff_imag_used[i,j]*m.V[i]*cos(m.theta[i]) \
                 + m.Yft_real_used[i,j]*m.V[j]*sin(m.theta[j]) + m.Yft_imag_used[i,j]*m.V[j]*cos(m.theta[j])
        current_pu_sq = I_real**2 + I_imag**2
        max_current_pu_sq = (m.I_from_max[i, j] / m.Ibase_from_kA[i, j])**2
        return current_pu_sq <= _csm * max_current_pu_sq

    def branch_current_to_rule(m, i, j):
        I_real = m.Yff_real_used[j,i]*m.V[j]*cos(m.theta[j]) - m.Yff_imag_used[j,i]*m.V[j]*sin(m.theta[j]) \
                 + m.Yft_real_used[j,i]*m.V[i]*cos(m.theta[i]) - m.Yft_imag_used[j,i]*m.V[i]*sin(m.theta[i])
        I_imag = m.Yff_real_used[j,i]*m.V[j]*sin(m.theta[j]) + m.Yff_imag_used[j,i]*m.V[j]*cos(m.theta[j]) \
                 + m.Yft_real_used[j,i]*m.V[i]*sin(m.theta[i]) + m.Yft_imag_used[j,i]*m.V[i]*cos(m.theta[i])
        current_pu_sq = I_real**2 + I_imag**2
        max_current_pu_sq = (m.I_from_max[j, i] / m.Ibase_from_kA[j, i])**2
        return current_pu_sq <= _csm * max_current_pu_sq

    model.branch_current_from_limit = Constraint(model.Branches, rule=branch_current_from_rule)

    model.branch_current_to_limit = Constraint(model.Branches, rule=branch_current_to_rule)

    # --- Objective function (captures lambda_p / lambda_q from outer scope) ---
    def objective_rule(m):
        p_dev_sq = sum((m.Pg_new[g] - m.Pg_base[g])**2 for g in m.G)
        q_dev_sq = sum((m.Qg_new[g] - m.Qg_base[g])**2 for g in m.G)
        return lambda_p * p_dev_sq + lambda_q * q_dev_sq

    model.obj = Objective(rule=objective_rule, sense=minimize)

    from pyomo.environ import SolverFactory    

    opt = SolverFactory('ipopt')

    
    # =====================================================
    # FOLD 1 — FIX GENERATION
    # =====================================================
    print("▶ Fold 1: Checking feasibility with base generation fixed")

    fold1_success = False

    try:
        for g in model.G:
            p_pin = float(fixed_setpoints[g]) if (fixed_setpoints and g in fixed_setpoints) else value(model.Pg_base[g])
            model.Pg_new[g].fix(p_pin)
            model.Qg_new[g].fix(value(model.Qg_base[g]))
        if fixed_setpoints:
            for _name, _mw in fixed_setpoints.items():
                if _name in model.Ext:
                    model.Pext[_name].fix(float(_mw))

        res1 = opt.solve(model, tee=True)
        tc1 = res1.solver.termination_condition

        if tc1 == TerminationCondition.optimal:
            print("✅ Base dispatch is feasible — no flexibility applied.")
            fold1_success = True
            return res1, model
    
    except Exception as e:
        print(f"⚠ Fold 1 failed with exception: {e}")

    
    # =====================================================
    # FOLD 2 — UNFIX GENERATION
    # =====================================================
    print("▶ Fold 2: Dispatch infeasible — enabling flexibility")

    for g in model.G:
        model.Pg_new[g].unfix()
        model.Qg_new[g].unfix()

    # Re-apply user-specified fixed setpoints — these elements remain pinned
    # throughout Fold 2 so only the other generators are free to redispatch.
    if fixed_setpoints:
        for _name, _mw in fixed_setpoints.items():
            if _name in model.G:
                model.Pg_new[_name].fix(float(_mw))
            elif _name in model.Ext:
                model.Pext[_name].fix(float(_mw))

    try:
        res2 = opt.solve(model, tee=True)
        tc2 = res2.solver.termination_condition
        
        if tc2 == TerminationCondition.optimal:
            print("✅ Flex optimization converged optimally.")
        else:
            print(f"❌ Flex optimization failed (termination = {tc2})")
    
    except Exception as e:
        print(f"❌ Fold 2 solver raised an exception: {e}")
        res2 = None

    return res2, model



def prepare_regulation_df_rsa(model, net, net_base, ts):
    """
    Prepare a DataFrame with generator and external grid regulation results.

    Parameters
    ----------
    model : Pyomo model
        Optimized model containing Pg_new, Pg_base, Qg_new, Qg_base, Pext, Qext
    net : pandapower network
        Network with base values for external grids
    timestamp : str or pd.Timestamp
        Current timestamp for the results

    Returns
    -------
    pd.DataFrame
        DataFrame containing regulation results
    """
    records = []

    # --- Generators regulation ---
    for g in model.G:
        Pg_new_val = value(model.Pg_new[g])
        Pg_base_val = value(model.Pg_base[g])
        Qg_new_val = value(model.Qg_new[g])
        Qg_base_val = value(model.Qg_base[g])

        records.append({
            "timestamp": str(ts),
            "element": g,
            "type": "Generator",
            "Pg_base": Pg_base_val,
            "Pg_new": Pg_new_val,
            "Pg_up": max(Pg_new_val - Pg_base_val, 0),
            "Pg_down": max(Pg_base_val - Pg_new_val, 0),
            "Qg_base": Qg_base_val,
            "Qg_new": Qg_new_val,
            "Qg_up": max(Qg_new_val - Qg_base_val, 0),
            "Qg_down": max(Qg_base_val - Qg_new_val, 0),
        })

    # --- External grid regulation (use actual ext_grid name from network) ---
    _ext_name = next(iter(model.Ext))  # use Pyomo's own key to avoid None/NaN mismatch
    Pext_new_val = value(model.Pext[_ext_name])
    Qext_new_val = value(model.Qext[_ext_name])
    Pext_base_val = net_base.res_ext_grid['p_mw'].values[0]
    Qext_base_val = net_base.res_ext_grid['q_mvar'].values[0]

    records.append({
        "timestamp": str(ts),
        "element": str(_ext_name) if _ext_name is not None else "Ext_Grid",
        "type": "External Grid",
        "Pg_base": Pext_base_val,
        "Pg_new": Pext_new_val,
        "Pg_up": max(Pext_new_val - Pext_base_val, 0),
        "Pg_down": max(Pext_base_val - Pext_new_val, 0),
        "Qg_base": Qext_base_val,
        "Qg_new": Qext_new_val,
        "Qg_up": max(Qext_new_val - Qext_base_val, 0),
        "Qg_down": max(Qext_base_val - Qext_new_val, 0),
    })

    # Convert to DataFrame
    df = pd.DataFrame(records)
    return df


def prepare_regulation_df_ca(model, net, ts, outage_type, outage_element_index):
    """
    Prepare a DataFrame with generator and external grid regulation results.

    Parameters
    ----------
    model : Pyomo model
        Optimized model containing Pg_new, Pg_base, Qg_new, Qg_base, Pext, Qext
    net : pandapower network
        Network with base values for external grids
    timestamp : str or pd.Timestamp
        Current timestamp for the results

    Returns
    -------
    pd.DataFrame
        DataFrame containing regulation results
    """
    records = []

    # --- Generators regulation ---
    for g in model.G:
        Pg_new_val = value(model.Pg_new[g])
        Pg_base_val = value(model.Pg_base[g])
        Qg_new_val = value(model.Qg_new[g])
        Qg_base_val = value(model.Qg_base[g])

        records.append({
            "timestamp": str(ts),
            "element": g,
            "type": "Generator",
            "Pg_base": Pg_base_val,
            "Pg_new": Pg_new_val,
            "Pg_up": max(Pg_new_val - Pg_base_val, 0),
            "Pg_down": max(Pg_base_val - Pg_new_val, 0),
            "Qg_base": Qg_base_val,
            "Qg_new": Qg_new_val,
            "Qg_up": max(Qg_new_val - Qg_base_val, 0),
            "Qg_down": max(Qg_base_val - Qg_new_val, 0),
            "outage_type": outage_type,
            "outage_element_index": outage_element_index,

        })

    # --- External grid regulation (use actual ext_grid name from network) ---
    _ext_name = next(iter(model.Ext))  # use Pyomo's own key to avoid None/NaN mismatch
    Pext_new_val = value(model.Pext[_ext_name])
    Qext_new_val = value(model.Qext[_ext_name])
    Pext_base_val = net.res_ext_grid['p_mw'].values[0]
    Qext_base_val = net.res_ext_grid['q_mvar'].values[0]

    records.append({
        "timestamp": str(ts),
        "element": str(_ext_name) if _ext_name is not None else "Ext_Grid",
        "type": "External Grid",
        "Pg_base": Pext_base_val,
        "Pg_new": Pext_new_val,
        "Pg_up": max(Pext_new_val - Pext_base_val, 0),
        "Pg_down": max(Pext_base_val - Pext_new_val, 0),
        "Qg_base": Qext_base_val,
        "Qg_new": Qext_new_val,
        "Qg_up": max(Qext_new_val - Qext_base_val, 0),
        "Qg_down": max(Qext_base_val - Qext_new_val, 0),
        "outage_type": outage_type,
        "outage_element_index": outage_element_index,
    })

    # Convert to DataFrame
    df = pd.DataFrame(records)
    return df


def get_admittance_parameters(database_admittance, ele_index):
    
    # Extract variables
    Yff_r           = database_admittance[ele_index]["Yff_r"]
    Yff_i           = database_admittance[ele_index]["Yff_i"]
    Yft_r           = database_admittance[ele_index]["Yft_r"]
    Yft_i           = database_admittance[ele_index]["Yft_i"]
    TAPS            = database_admittance[ele_index]["TAPS"]
    trafo_ranges    = database_admittance[ele_index]["trafo_ranges"]
    trafo_defaults  = database_admittance[ele_index]["trafo_defaults"]
    branch_to_trafo = database_admittance[ele_index]["branch_to_trafo"]

    return  Yff_r, Yff_i, Yft_r, Yft_i, TAPS, trafo_ranges, trafo_defaults, branch_to_trafo


def extract_post_opf_voltages(model, net) -> dict:
    """Extract post-OPF bus voltage magnitudes from a solved Pyomo model.

    Parameters
    ----------
    model : pyomo.ConcreteModel
        Solved model whose ``V`` variables hold the optimized voltage magnitudes.
    net : pandapowerNet
        Network used to resolve bus indices to human-readable names via
        ``net.bus['name']``.

    Returns
    -------
    dict
        Keys:
        - ``bus_voltages_post_opf``: list of ``{bus, bus_name, vm_pu}`` records.
        - ``excluded_buses_post_opf``: list of ``{bus, bus_name, reason}`` records.
        - ``excluded_bus_count_post_opf``: integer count of excluded bus records.
        - ``opf_vm_lower_used``: lower voltage bound applied inside the OPF (p.u.).
        - ``opf_vm_upper_used``: upper voltage bound applied inside the OPF (p.u.).
    """
    def _bus_name(bus_idx: int) -> str:
        _raw = str(net.bus.at[bus_idx, 'name']).strip() if 'name' in net.bus.columns else ""
        return _raw if _raw else f"Bus_{bus_idx}"

    bus_voltages = []
    excluded_buses = []
    model_buses = {int(b) for b in model.B}

    # Buses omitted from model.B are still reported so callers/LLMs can explain why.
    for idx in net.bus.index:
        b = int(idx)
        if b in model_buses:
            continue
        if "in_service" in net.bus.columns and not bool(net.bus.at[idx, "in_service"]):
            reason = "out_of_service"
        else:
            reason = "not_in_model_bus_set"
        excluded_buses.append({"bus": b, "bus_name": _bus_name(b), "reason": reason})

    for b in sorted(model.B):
        # Some buses can remain structurally disconnected in uploaded/custom
        # admittance cases, leaving V[b] uninitialized after solve.
        v = value(model.V[b], exception=False)
        if v is None:
            excluded_buses.append({"bus": int(b), "bus_name": _bus_name(int(b)), "reason": "uninitialized_voltage"})
            continue
        if not np.isfinite(v):
            excluded_buses.append({"bus": int(b), "bus_name": _bus_name(int(b)), "reason": "non_finite_voltage"})
            continue
        bus_voltages.append({"bus": int(b), "bus_name": _bus_name(int(b)), "vm_pu": round(float(v), 6)})

    first_b = next(iter(model.B))
    vm_lower = model.V[first_b].lb
    vm_upper = model.V[first_b].ub

    return {
        "bus_voltages_post_opf": bus_voltages,
        "excluded_buses_post_opf": excluded_buses,
        "excluded_bus_count_post_opf": len(excluded_buses),
        "opf_vm_lower_used": float(vm_lower) if vm_lower is not None else 0.95,
        "opf_vm_upper_used": float(vm_upper) if vm_upper is not None else 1.05,
    }


def extract_post_opf_voltages_scenario(
    model,
    net,
    scenario_idx: int = 0,
    display_vm_lower: float | None = None,
    display_vm_upper: float | None = None,
) -> dict:
    """Extract post-OPF bus voltages from a scenario-indexed model.

    Uses one representative scenario (default k=0) for UI-compatible output.

    When scenario relaxations are enabled, the model variable bounds can be wider
    than the requested OPF security limits. The optional display_vm_* arguments let
    the caller expose the enforced security limits in the UI instead of the relaxed
    internal box.
    """
    def _bus_name(bus_idx: int) -> str:
        _raw = str(net.bus.at[bus_idx, 'name']).strip() if 'name' in net.bus.columns else ""
        return _raw if _raw else f"Bus_{bus_idx}"

    bus_voltages = []
    excluded_buses = []
    model_buses = {int(b) for b in model.B}

    for idx in net.bus.index:
        b = int(idx)
        if b in model_buses:
            continue
        if "in_service" in net.bus.columns and not bool(net.bus.at[idx, "in_service"]):
            reason = "out_of_service"
        else:
            reason = "not_in_model_bus_set"
        excluded_buses.append({"bus": b, "bus_name": _bus_name(b), "reason": reason})

    for b in sorted(model.B):
        v = value(model.V[b, scenario_idx], exception=False)
        if v is None:
            excluded_buses.append({"bus": int(b), "bus_name": _bus_name(int(b)), "reason": "uninitialized_voltage"})
            continue
        if not np.isfinite(v):
            excluded_buses.append({"bus": int(b), "bus_name": _bus_name(int(b)), "reason": "non_finite_voltage"})
            continue
        bus_voltages.append({"bus": int(b), "bus_name": _bus_name(int(b)), "vm_pu": round(float(v), 6)})

    first_b = next(iter(model.B))
    vm_lower = display_vm_lower if display_vm_lower is not None else model.V[first_b, scenario_idx].lb
    vm_upper = display_vm_upper if display_vm_upper is not None else model.V[first_b, scenario_idx].ub

    return {
        "bus_voltages_post_opf": bus_voltages,
        "excluded_buses_post_opf": excluded_buses,
        "excluded_bus_count_post_opf": len(excluded_buses),
        "opf_vm_lower_used": float(vm_lower) if vm_lower is not None else 0.95,
        "opf_vm_upper_used": float(vm_upper) if vm_upper is not None else 1.05,
    }


def scenario_optimization_model_base(
    net,
    net_base,
    Yff_r,
    Yff_i,
    Yft_r,
    Yft_i,
    TAPS,
    trafo_ranges,
    trafo_defaults,
    branch_to_trafo,
    xi_by_gen: dict,
    load_multipliers: list,
    opf_vm_lower: float = 0.95,
    opf_vm_upper: float = 1.05,
    lambda_p: float = 0.01,
    lambda_q: float = 0.001,
    pg_max_overrides: dict = None,
    pg_min_overrides: dict = None,
    min_power_factor: float = 0.95,
    current_safety_margin: float = 0.9,
    fixed_setpoints: dict = None,
    allowed_violation_fraction: float = 0.0,
    warm_start: dict | None = None,
):
    """Scenario-based AC-OPF with shared dispatch and scenario-indexed PF states.

    Formulation notes:
    - Shared first-stage decisions: Pg_new[g], Qg_new[g].
    - Scenario variables: V[b,k], theta[b,k], Pinj[g,k], Pext[e,k], Qext[e,k].
    - Renewable min(P_new, xi) is represented with Pinj[g,k] <= Pg_new[g]
      and Pinj[g,k] <= xi[g,k]. A tiny objective reward on Pinj encourages
      the optimizer to hit the active bound, reproducing min() behavior.
    - Optional discard mode uses a continuous indicator relaxation z[k] with
      a scenario budget (sum z <= frac*K) to stay compatible with IPOPT.
    """
    model = ConcreteModel()

    # Core index sets.
    if "in_service" in net.bus.columns:
        bus_indices = [int(i) for i in net.bus.index if bool(net.bus.at[i, "in_service"])]
    else:
        bus_indices = [int(i) for i in net.bus.index]
    for _sb in net.ext_grid["bus"].tolist() if "bus" in net.ext_grid.columns else []:
        _sb_i = int(_sb)
        if _sb_i not in bus_indices:
            bus_indices.append(_sb_i)
    model.B = Set(initialize=bus_indices)
    K = max(1, len(load_multipliers))
    model.K = Set(initialize=list(range(K)))

    # Base solved state for initialization.
    vm_pu_map = {}
    va_rad_map = {}
    for idx in net.bus.index:
        b = int(idx)
        vm_pu_map[b] = float(net.res_bus.at[idx, 'vm_pu'])
        va_rad_map[b] = math.radians(float(net.res_bus.at[idx, 'va_degree']))

    # Loads.
    load_name_list = net.load['substation_name'].tolist()
    load_data_p = dict(zip(net.load['substation_name'], net.load['p_mw']))
    load_data_q = dict(zip(net.load['substation_name'], net.load['q_mvar']))
    load_bus_map = dict(zip(net.load['substation_name'], net.load['bus']))
    model.L = Set(initialize=load_name_list)
    model.Pl0 = Param(model.L, initialize=load_data_p)
    model.Ql0 = Param(model.L, initialize=load_data_q)
    model.load_bus = Param(model.L, initialize=load_bus_map)

    # Generator naming conventions (must match optimization_model_base).
    def _sgen_name(row, idx) -> str:
        key_col = "substation_name" if "substation_name" in net.sgen.columns else "name"
        raw = row.get(key_col, None)
        s = str(raw).strip() if raw is not None else ""
        return s if s else f"sgen_{idx}"

    def _gen_name(row, idx) -> str:
        raw = row.get("name", None)
        s = str(raw).strip() if raw is not None else ""
        return s if s else f"Gen_bus{int(row.get('bus', idx))}"

    gen_name_list = []
    P_base_data = {}
    Q_base_data = {}
    gen_bus_map = {}
    condenser_q_max = {}
    condenser_q_min = {}

    if not net.sgen.empty:
        for idx, row in net.sgen.iterrows():
            name = _sgen_name(row, idx)
            gen_name_list.append(name)
            res_p = net.res_sgen.at[idx, "p_mw"] if (not net.res_sgen.empty and idx in net.res_sgen.index) else float(row.get("p_mw", 0.0))
            res_q = net.res_sgen.at[idx, "q_mvar"] if (not net.res_sgen.empty and idx in net.res_sgen.index) else float(row.get("q_mvar", 0.0))
            P_base_data[name] = float(res_p)
            Q_base_data[name] = float(res_q)
            gen_bus_map[name] = int(row["bus"])

    if not net.gen.empty:
        for idx, row in net.gen.iterrows():
            name = _gen_name(row, idx)
            if name in gen_name_list:
                continue
            gen_name_list.append(name)
            res_p = net.res_gen.at[idx, "p_mw"] if (not net.res_gen.empty and idx in net.res_gen.index) else float(row.get("p_mw", 0.0))
            res_q = net.res_gen.at[idx, "q_mvar"] if (not net.res_gen.empty and idx in net.res_gen.index) else float(row.get("q_mvar", 0.0))
            try:
                mxq = float(row.get("max_q_mvar", 50.0))
                mxq = mxq if not math.isnan(mxq) else 50.0
            except (TypeError, ValueError):
                mxq = 50.0
            try:
                mnq = float(row.get("min_q_mvar", -50.0))
                mnq = mnq if not math.isnan(mnq) else -50.0
            except (TypeError, ValueError):
                mnq = -50.0
            condenser_q_max[name] = mxq
            condenser_q_min[name] = mnq
            res_q = max(mnq, min(mxq, float(res_q)))
            P_base_data[name] = float(res_p)
            Q_base_data[name] = float(res_q)
            gen_bus_map[name] = int(row["bus"])

    _pg_max_all = _extract_pg_max(net)
    Pg_max_data = {g: _pg_max_all.get(g, 0.001) for g in gen_name_list}
    Pg_min_data = {g: 0.0 for g in gen_name_list}

    for g in condenser_q_max:
        Pg_max_data[g] = 0.0

    if pg_max_overrides:
        Pg_max_data.update(pg_max_overrides)
    if pg_min_overrides:
        Pg_min_data.update(pg_min_overrides)

    model.G = Set(initialize=gen_name_list)
    model.Pg_base = Param(model.G, initialize=P_base_data)
    model.Pg_min = Param(model.G, initialize=Pg_min_data)
    model.Pg_max = Param(model.G, initialize=Pg_max_data)
    model.Qg_base = Param(model.G, initialize=Q_base_data)
    model.gen_bus = Param(model.G, initialize=gen_bus_map)

    # Shared first-stage decisions.
    def pg_bounds_rule(m, g):
        return (m.Pg_min[g], m.Pg_max[g])
    model.Pg_new = Var(model.G, bounds=pg_bounds_rule)
    model.Qg_new = Var(model.G)

    PF_min = min_power_factor
    tan_phi = math.tan(math.acos(PF_min))
    model.tan_phi_sq = Param(initialize=tan_phi ** 2)
    pf_gens = [g for g in gen_name_list if Pg_max_data.get(g, 0.0) >= 1.0]
    cond_gens = [g for g in gen_name_list if Pg_max_data.get(g, 0.0) < 1.0 and g in condenser_q_max]
    model.PF_set = Set(initialize=pf_gens)
    model.PF_cone = Constraint(model.PF_set, rule=lambda m, g: m.Qg_new[g] ** 2 <= m.tan_phi_sq * m.Pg_new[g] ** 2)
    if cond_gens:
        model.CondSet = Set(initialize=cond_gens)
        model.Qg_cond_max = Param(model.CondSet, initialize={g: condenser_q_max[g] for g in cond_gens})
        model.Qg_cond_min = Param(model.CondSet, initialize={g: condenser_q_min[g] for g in cond_gens})
        model.QgCondUpper = Constraint(model.CondSet, rule=lambda m, g: m.Qg_new[g] <= m.Qg_cond_max[g])
        model.QgCondLower = Constraint(model.CondSet, rule=lambda m, g: m.Qg_new[g] >= m.Qg_cond_min[g])

    # Shunts.
    shunt_name_list = net.shunt['name'].tolist()
    shunt_bus_map = dict(zip(net.shunt['name'], net.shunt['bus']))
    shunt_Q_nom = dict(zip(net.shunt['name'], net.shunt['q_mvar'].fillna(0.0)))
    shunt_P_nom = dict(zip(net.shunt['name'], net.shunt['p_mw'].fillna(0.0)))
    model.S = Set(initialize=shunt_name_list)
    model.shunt_bus = Param(model.S, initialize=shunt_bus_map)
    model.Qs_nom = Param(model.S, initialize=shunt_Q_nom)
    model.Ps_nom = Param(model.S, initialize=shunt_P_nom)

    # External grid.
    _raw_names = net.ext_grid['name'].tolist()
    ext_grid_name_list = [
        (n if (isinstance(n, str) and n.strip()) else f"ExtGrid_{i}")
        for i, n in enumerate(_raw_names)
    ]
    net.ext_grid = net.ext_grid.copy()
    net.ext_grid['name'] = ext_grid_name_list
    net_base.ext_grid = net_base.ext_grid.copy()
    net_base.ext_grid['name'] = ext_grid_name_list

    ext_grid_bus_map = dict(zip(ext_grid_name_list, net_base.ext_grid['bus']))
    _pext_lo_map = {}
    _pext_hi_map = {}
    for name, (_, row) in zip(ext_grid_name_list, net.ext_grid.iterrows()):
        lo = float(row['min_p_mw']) if ('min_p_mw' in net.ext_grid.columns and pd.notna(row.get('min_p_mw'))) else -200.0
        hi = float(row['max_p_mw']) if ('max_p_mw' in net.ext_grid.columns and pd.notna(row.get('max_p_mw'))) else 200.0
        if hi <= lo or (lo == 0.0 and hi == 0.0):
            lo, hi = -200.0, 200.0
        _pext_lo_map[name] = lo
        _pext_hi_map[name] = hi

    model.Ext = Set(initialize=ext_grid_name_list)
    model.ext_bus = Param(model.Ext, initialize=ext_grid_bus_map)
    model.Pext = Var(model.Ext, model.K, bounds=lambda m, e, k: (_pext_lo_map.get(e, -200.0), _pext_hi_map.get(e, 200.0)))
    model.Qext = Var(model.Ext, model.K, bounds=lambda m, e, k: (-200.0, 200.0))

    # Branch and admittance data.
    Sbase_MVA = net.sn_mva
    trafo_current_data = {}
    for trafo_idx, trafo_data in net.trafo.iterrows():
        i = int(trafo_data['hv_bus'])
        j = int(trafo_data['lv_bus'])
        sn_mva = float(trafo_data['sn_mva'])
        Vbase_hv_kV = float(net.bus.at[i, 'vn_kv'])
        Vbase_lv_kV = float(net.bus.at[j, 'vn_kv'])
        Irated_hv_kA = sn_mva / (np.sqrt(3) * Vbase_hv_kV)
        Irated_lv_kA = sn_mva / (np.sqrt(3) * Vbase_lv_kV)
        Ibase_hv_kA = Sbase_MVA / (np.sqrt(3) * Vbase_hv_kV)
        Ibase_lv_kA = Sbase_MVA / (np.sqrt(3) * Vbase_lv_kV)
        trafo_current_data[(i, j)] = {'Ibase_from_kA': Ibase_hv_kA, 'I_from_max': Irated_hv_kA}
        trafo_current_data[(j, i)] = {'Ibase_from_kA': Ibase_lv_kA, 'I_from_max': Irated_lv_kA}

    line_current_data = {}
    for line_idx, line_data in net.line.iterrows():
        i = int(line_data['from_bus'])
        j = int(line_data['to_bus'])
        Vbase_from_kV = float(net.bus.at[i, 'vn_kv'])
        Vbase_to_kV = float(net.bus.at[j, 'vn_kv'])
        Ibase_from_kA = Sbase_MVA / (np.sqrt(3) * Vbase_from_kV)
        Ibase_to_kA = Sbase_MVA / (np.sqrt(3) * Vbase_to_kV)
        I_from_max = float(line_data['max_i_ka'])
        I_to_max = float(line_data['max_i_ka'])
        line_current_data[(i, j)] = {'Ibase_from_kA': Ibase_from_kA, 'I_from_max': I_from_max}
        line_current_data[(j, i)] = {'Ibase_from_kA': Ibase_to_kA, 'I_from_max': I_to_max}

    branch_data = {}
    branch_data.update(line_current_data)
    branch_data.update(trafo_current_data)
    branches_list = list(branch_data.keys())
    model.Branches = Set(initialize=branches_list, dimen=2)

    # Fixed admittance values by branch.
    Yff_real_used = {}
    Yff_imag_used = {}
    Yft_real_used = {}
    Yft_imag_used = {}
    for (i, j) in branches_list:
        if (i, j) in branch_to_trafo:
            trafo_idx = branch_to_trafo[(i, j)]
            t_default = int(net.trafo.at[trafo_idx, "tap_neutral"]) if "tap_neutral" in net.trafo.columns else 0
        else:
            t_default = 1
        Yff_real_used[(i, j)] = float(Yff_r[(i, j), t_default])
        Yff_imag_used[(i, j)] = float(Yff_i[(i, j), t_default])
        Yft_real_used[(i, j)] = float(Yft_r[(i, j), t_default])
        Yft_imag_used[(i, j)] = float(Yft_i[(i, j), t_default])

    model.Yff_real_used = Param(model.Branches, initialize=Yff_real_used)
    model.Yff_imag_used = Param(model.Branches, initialize=Yff_imag_used)
    model.Yft_real_used = Param(model.Branches, initialize=Yft_real_used)
    model.Yft_imag_used = Param(model.Branches, initialize=Yft_imag_used)
    model.Ibase_from_kA = Param(model.Branches, initialize={(i, j): branch_data[(i, j)]['Ibase_from_kA'] for (i, j) in branches_list}, within=Reals)
    model.I_from_max = Param(model.Branches, initialize={(i, j): branch_data[(i, j)]['I_from_max'] for (i, j) in branches_list}, within=Reals)

    # Scenario uncertainty parameters.
    model.load_mult = Param(model.K, initialize={k: float(load_multipliers[k]) for k in range(K)})
    xi_init = {}
    for g in gen_name_list:
        vals = xi_by_gen.get(g, None)
        if vals is None:
            vals = [float(Pg_max_data.get(g, 0.0))] * K
        for k in range(K):
            xi_init[(g, k)] = float(vals[k])
    model.xi = Param(model.G, model.K, initialize=xi_init)

    # Voltage/angle and injection variables.
    V_min = opf_vm_lower
    V_max = opf_vm_upper
    allow_frac = max(0.0, float(allowed_violation_fraction))
    if allow_frac > 0.0:
        big_m_v = 0.20
        model.V = Var(model.B, model.K, bounds=(V_min - big_m_v, V_max + big_m_v))
        model.z = Var(model.K, bounds=(0.0, 1.0))
        model.v_up_relaxed = Constraint(model.B, model.K, rule=lambda m, b, k: m.V[b, k] <= V_max + big_m_v * m.z[k])
        model.v_lo_relaxed = Constraint(model.B, model.K, rule=lambda m, b, k: m.V[b, k] >= V_min - big_m_v * m.z[k])
        model.violation_budget = Constraint(expr=sum(model.z[k] for k in model.K) <= allow_frac * K)
    else:
        model.V = Var(model.B, model.K, bounds=(V_min, V_max))
    model.theta = Var(model.B, model.K, bounds=(-math.pi, math.pi))
    model.Pinj = Var(model.G, model.K, within=NonNegativeReals)

    # Renewable min(P_new, xi) envelope.
    model.pinj_le_dispatch = Constraint(model.G, model.K, rule=lambda m, g, k: m.Pinj[g, k] <= m.Pg_new[g])
    model.pinj_le_resource = Constraint(model.G, model.K, rule=lambda m, g, k: m.Pinj[g, k] <= m.xi[g, k])

    # Slack reference per scenario.
    slack_bus = int(net.ext_grid['bus'].iloc[0])
    for k in model.K:
        model.theta[slack_bus, k].fix(va_rad_map[slack_bus])
        model.V[slack_bus, k].fix(vm_pu_map[slack_bus])

    # Scenario AC balance constraints.
    def active_balance_rule(m, b, k):
        p_gen = sum(m.Pinj[g, k] for g in m.G if m.gen_bus[g] == b)
        p_ext = sum(m.Pext[e, k] for e in m.Ext if m.ext_bus[e] == b)
        p_load = sum(m.Pl0[l] * m.load_mult[k] for l in m.L if m.load_bus[l] == b)
        p_sh = sum(m.Ps_nom[s] for s in m.S if m.shunt_bus[s] == b)
        p_inj = p_gen + p_ext - p_load - p_sh
        p_flow = sum(
            m.V[i, k] ** 2 * m.Yff_real_used[i, j]
            + m.V[i, k] * m.V[j, k] * (
                m.Yft_real_used[i, j] * cos(m.theta[i, k] - m.theta[j, k])
                + m.Yft_imag_used[i, j] * sin(m.theta[i, k] - m.theta[j, k])
            )
            for (i, j) in m.Branches if i == b
        )
        expr = p_inj == p_flow * 100
        if isinstance(expr, (bool, np.bool_)):
            return Constraint.Feasible if expr else Constraint.Infeasible
        return expr

    def reactive_balance_rule(m, b, k):
        q_gen = sum(m.Qg_new[g] for g in m.G if m.gen_bus[g] == b)
        q_ext = sum(m.Qext[e, k] for e in m.Ext if m.ext_bus[e] == b)
        q_load = sum(m.Ql0[l] * m.load_mult[k] for l in m.L if m.load_bus[l] == b)
        q_sh = sum(m.Qs_nom[s] * (m.V[b, k] ** 2) for s in m.S if m.shunt_bus[s] == b)
        q_inj = q_gen + q_ext - q_load - q_sh
        q_flow = sum(
            - m.V[i, k] ** 2 * m.Yff_imag_used[i, j]
            - m.V[i, k] * m.V[j, k] * (
                m.Yft_imag_used[i, j] * cos(m.theta[i, k] - m.theta[j, k])
                - m.Yft_real_used[i, j] * sin(m.theta[i, k] - m.theta[j, k])
            )
            for (i, j) in m.Branches if i == b
        )
        expr = q_inj == q_flow * 100
        if isinstance(expr, (bool, np.bool_)):
            return Constraint.Feasible if expr else Constraint.Infeasible
        return expr

    model.active_balance = Constraint(model.B, model.K, rule=active_balance_rule)
    model.reactive_balance = Constraint(model.B, model.K, rule=reactive_balance_rule)

    _csm = float(current_safety_margin)

    def branch_current_from_rule(m, i, j, k):
        I_real = (
            m.Yff_real_used[i, j] * m.V[i, k] * cos(m.theta[i, k])
            - m.Yff_imag_used[i, j] * m.V[i, k] * sin(m.theta[i, k])
            + m.Yft_real_used[i, j] * m.V[j, k] * cos(m.theta[j, k])
            - m.Yft_imag_used[i, j] * m.V[j, k] * sin(m.theta[j, k])
        )
        I_imag = (
            m.Yff_real_used[i, j] * m.V[i, k] * sin(m.theta[i, k])
            + m.Yff_imag_used[i, j] * m.V[i, k] * cos(m.theta[i, k])
            + m.Yft_real_used[i, j] * m.V[j, k] * sin(m.theta[j, k])
            + m.Yft_imag_used[i, j] * m.V[j, k] * cos(m.theta[j, k])
        )
        current_pu_sq = I_real ** 2 + I_imag ** 2
        max_current_pu_sq = (m.I_from_max[i, j] / m.Ibase_from_kA[i, j]) ** 2
        return current_pu_sq <= _csm * max_current_pu_sq

    def branch_current_to_rule(m, i, j, k):
        I_real = (
            m.Yff_real_used[j, i] * m.V[j, k] * cos(m.theta[j, k])
            - m.Yff_imag_used[j, i] * m.V[j, k] * sin(m.theta[j, k])
            + m.Yft_real_used[j, i] * m.V[i, k] * cos(m.theta[i, k])
            - m.Yft_imag_used[j, i] * m.V[i, k] * sin(m.theta[i, k])
        )
        I_imag = (
            m.Yff_real_used[j, i] * m.V[j, k] * sin(m.theta[j, k])
            + m.Yff_imag_used[j, i] * m.V[j, k] * cos(m.theta[j, k])
            + m.Yft_real_used[j, i] * m.V[i, k] * sin(m.theta[i, k])
            + m.Yft_imag_used[j, i] * m.V[i, k] * cos(m.theta[i, k])
        )
        current_pu_sq = I_real ** 2 + I_imag ** 2
        max_current_pu_sq = (m.I_from_max[j, i] / m.Ibase_from_kA[j, i]) ** 2
        return current_pu_sq <= _csm * max_current_pu_sq

    model.branch_current_from_limit = Constraint(model.Branches, model.K, rule=branch_current_from_rule)
    model.branch_current_to_limit = Constraint(model.Branches, model.K, rule=branch_current_to_rule)

    # Optional fixed setpoints.
    if fixed_setpoints:
        for _name, _mw in fixed_setpoints.items():
            if _name in model.G:
                model.Pg_new[_name].fix(float(_mw))

    # Warm-start.
    if warm_start:
        ws_pg = warm_start.get("Pg_new", {})
        ws_qg = warm_start.get("Qg_new", {})
        ws_v = warm_start.get("V", {})
        ws_th = warm_start.get("theta", {})
        for g in model.G:
            if g in ws_pg:
                model.Pg_new[g].value = float(ws_pg[g])
            else:
                model.Pg_new[g].value = float(value(model.Pg_base[g]))
            if g in ws_qg:
                model.Qg_new[g].value = float(ws_qg[g])
            else:
                model.Qg_new[g].value = float(value(model.Qg_base[g]))
            for k in model.K:
                model.Pinj[g, k].value = min(float(model.Pg_new[g].value), float(value(model.xi[g, k])))
        for b in model.B:
            v0 = float(ws_v.get(b, vm_pu_map[b]))
            t0 = float(ws_th.get(b, va_rad_map[b]))
            for k in model.K:
                model.V[b, k].value = v0
                model.theta[b, k].value = t0
        for e in model.Ext:
            base_p = float(net_base.res_ext_grid['p_mw'].values[0]) if not net_base.res_ext_grid.empty else 0.0
            base_q = float(net_base.res_ext_grid['q_mvar'].values[0]) if not net_base.res_ext_grid.empty else 0.0
            for k in model.K:
                model.Pext[e, k].value = base_p
                model.Qext[e, k].value = base_q

    # Objective: same redispatch effort as deterministic OPF + tiny Pinj reward.
    pinj_reward = 1e-6
    z_pen = 1e-4

    def objective_rule(m):
        p_dev_sq = sum((m.Pg_new[g] - m.Pg_base[g]) ** 2 for g in m.G)
        q_dev_sq = sum((m.Qg_new[g] - m.Qg_base[g]) ** 2 for g in m.G)
        inj_term = sum(m.Pinj[g, k] for g in m.G for k in m.K)
        z_term = sum(m.z[k] for k in m.K) if hasattr(m, 'z') else 0.0
        return lambda_p * p_dev_sq + lambda_q * q_dev_sq - pinj_reward * inj_term + z_pen * z_term

    model.obj = Objective(rule=objective_rule, sense=minimize)

    opt = SolverFactory('ipopt')
    opt.options['warm_start_init_point'] = 'yes'
    opt.options['warm_start_bound_push'] = 1e-7
    opt.options['warm_start_mult_bound_push'] = 1e-7
    opt.options['mu_init'] = 1e-4
    opt.options['max_iter'] = 1500

    try:
        results = opt.solve(model, tee=True)
    except Exception:
        results = None

    meta = {
        "n_scenarios": K,
        "allowed_violation_fraction": allow_frac,
        "discard_mode": bool(allow_frac > 0.0),
        "discard_indicator_relaxation": bool(allow_frac > 0.0),
    }
    return results, model, meta
