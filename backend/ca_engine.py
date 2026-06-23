#!/usr/bin/env python
# coding: utf-8

import re
import pandas as pd
import numpy as np
import pandapower.networks
import pandapower as pp
import os
import copy
from pandapower.powerflow import LoadflowNotConverged


def contingency_assessment_single(
    net,
    element,
    element_index,
    timestamp,
    vm_lower: float = 0.95,
    vm_upper: float = 1.05,
    max_line_loading_pct: float = 90.0,
    max_trafo_loading_pct: float = 90.0,
):
    """
    Perform a single N-1 contingency assessment for a specified network element.

    Parameters
    ----------
    net : pandapowerNet
        The original pandapower network object.
    element : str
        Type of element to be taken out of service: "line" or "trafo".
    element_index : int
        Index of the element (line/trafo) in `net.line` or `net.trafo`.
    timestamp : str
        Time label for storing results (e.g., '2025-06-19 15:30:00').
    vm_lower : float
        Lower voltage limit in p.u. (default 0.95).
    vm_upper : float
        Upper voltage limit in p.u. (default 1.05).
    max_line_loading_pct : float
        Thermal overload threshold for lines in percent (default 90.0).
    max_trafo_loading_pct : float
        Thermal overload threshold for transformers in percent (default 90.0).

    Returns
    -------
    df_results : pd.DataFrame
        DataFrame listing all violations (voltage or loading).
    net_copy : pandapowerNet
        The modified network with the specified element switched out.
    """

    print("\n================ Contingency Assessment (Single Outage) =============================")
    print(f"=====================================================================================\n")
    print(f"→ Outage Element: {element.upper()} | Index: {element_index}\n")

    # ---------------------------------------------------------
    # Define violation thresholds (use passed-in values)
    # ---------------------------------------------------------
    max_line_loading = max_line_loading_pct
    max_trafo_loading = max_trafo_loading_pct

    # Attempt power flow
    try:
        pp.runpp(net)
        pf_converged = True
    except LoadflowNotConverged:
        pf_converged = False

    # If PF did not converge → return empty result
    if not pf_converged:
        print(f"⚠️ Power flow did NOT converge after outage of {element} {element_index}.")
        return pd.DataFrame(), net

    # ---------------------------------------------------------
    # Collect Violations
    # ---------------------------------------------------------
    results = []

    # Voltage violations
    for bus_idx, vm in enumerate(net.res_bus.vm_pu):
        if vm > vm_upper or vm < vm_lower:
            results.append({
                "timestamp": timestamp,
                "outage_type": element,
                "outage_element_index": element_index,  # store the element taken out
                "violation_type": "bus_vm_pu",
                "violation_element_index": bus_idx,
                "value": round(vm, 3)
            })

    # Line overloads
    for line_idx, loading in enumerate(net.res_line.loading_percent):
        if loading > max_line_loading:
            results.append({
                "timestamp": timestamp,
                "outage_type": element,
                "outage_element_index": element_index,
                "violation_type": "line_loading",
                "violation_element_index": line_idx,
                "value": round(loading, 2)
            })

    # Transformer overloads
    for trafo_idx, loading in enumerate(net.res_trafo.loading_percent):
        if loading > max_trafo_loading:
            results.append({
                "timestamp": timestamp,
                "outage_type": element,
                "outage_element_index": element_index,
                "violation_type": "trafo_loading",
                "violation_element_index": trafo_idx,
                "value": round(loading, 2)
            })

    df_results = pd.DataFrame(results)

    print("✅ Contingency assessment completed.\n")
    print("=======================================================================\n")

    return df_results, net






