#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import pandapower as pp

# Violations are joined back onto the topology by element name, so they have
# to be spelled exactly as /api/network/topology spells them.
from element_names import bus_display_name, line_display_name, trafo_display_name


# In[2]:



FLOAT_TOL = 1e-6           # p.u. and %: rounding, not a violation

def map_bus_names_to_lines(net):
    """
    Map bus names to line endpoints in a pandapower network.

    This function adds two new columns to net.line:
        - 'from_bus_name': The name of the bus at the 'from_bus' end of the line
        - 'to_bus_name': The name of the bus at the 'to_bus' end of the line

    It uses the 'from_bus' and 'to_bus' index values in net.line to look up
    corresponding 'name' entries in net.bus.

    Parameters
    ----------
    net : pandapowerNet
        The pandapower network object. Assumes net.bus has a 'name' column.

    Returns
    -------
    net : pandapowerNet
        The updated pandapower network with 'from_bus_name' and 'to_bus_name'
        columns added to net.line.
    """

    # -------------------- Step 1: Verify the 'name' column exists in net.bus --------------------
    if 'name' not in net.bus.columns:
        raise ValueError("The 'net.bus' DataFrame must contain a 'name' column.")

    # -------------------- Step 2: Create a mapping from bus index to bus name --------------------
    bus_name_map = net.bus['name'].to_dict()  # {bus_index: "bus_name"}

    # -------------------- Step 3: Map from_bus and to_bus to their names --------------------
    net.line['from_bus_name'] = net.line['from_bus'].map(bus_name_map)
    net.line['to_bus_name'] = net.line['to_bus'].map(bus_name_map)

    # -------------------- Step 4: Return the updated network --------------------
    return net


# In[3]:


def real_time_security_assessment(
    net,
    timestamp,
    vm_lower: float = 0.95,
    vm_upper: float = 1.05,
    max_line_loading_pct: float = 90.0,
    max_trafo_loading_pct: float = 90.0,
):
    """
    Perform Real-Time Security Assessment (RSA) on a given pandapower network.

    Parameters
    ----------
    net : pandapowerNet
        The pandapower network object with assigned load and generation (p_mw, q_mvar) values.
    timestamp : str
        Timestamp of the assessment in string format (e.g., "2025-06-19 15:30:00").
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
    net : pandapowerNet
        The updated pandapower network with power flow results after security analysis.

    The assessment includes the following components:
        1. Base Case Power Flow Analysis
        2. Voltage Violation Detection
        3. Thermal Overload Detection
        4. Reporting Dashboard

    
    results_df : pandas.DataFrame
            DataFrame containing violation records with columns:
                ['timestamp', 'violation_type', 'element_index', 'element', 'value']
            (the limits used are the arguments; the caller reports them once)
        """


    print("\n==================== Real-Time Security Assessment Engine (RSAE) ====================")
    print("=====================================================================================\n")

    results = []  # List to store structured violations

    # -------------------- Step 0: Map Bus Names to Lines --------------------
    print("[0] Mapping bus names to line endpoints...")
    map_bus_names_to_lines(net)

    # -------------------- Step 1: Run Base Case Power Flow Analysis --------------------
    print("[1] Running base case power flow...")
    try:
        pp.runpp(net)
        print("    ✅ Power flow successful.\n")
    except Exception as e:
        print("    ❌ Power flow did not converge.")
        print("    Error:", str(e))
        return net, results

    # -------------------- Step 2: Voltage Violation Check --------------------
    print("[2] Checking for voltage violations...")
    # A tolerance for floating-point noise only: a dispatch the OPF put exactly
    # on a limit comes back from the power flow at 0.9499999904 p.u., which is
    # the limit, not a violation of it.
    v_min, v_max = vm_lower - FLOAT_TOL, vm_upper + FLOAT_TOL
    voltage_violations = net.res_bus[(net.res_bus.vm_pu < v_min) | (net.res_bus.vm_pu > v_max)]
    print(f"    ⚡ Voltage Violations: {len(voltage_violations)}")

    for idx, row in voltage_violations.iterrows():
        bus_name = bus_display_name(net, int(idx))
        results.append({
            "timestamp": timestamp,
            "violation_type": "bus_vm_pu",
            "element_index": idx,
            "element": bus_name,
            "value": row.vm_pu
        })
    
    # -------------------- Step 3: Thermal Overload Check --------------------
    print("[3] Checking for thermal overloads on lines...")
    overloads = net.res_line[net.res_line.loading_percent > max_line_loading_pct + FLOAT_TOL]
    print(f"    🌡️ Line Overloads: {len(overloads)}")

    for idx, row in overloads.iterrows():
        line_name = line_display_name(net, int(idx))
        results.append({
            "timestamp": timestamp,
            "violation_type": "line_loading",
            "element_index": idx,
            "element": line_name,
            "value": row.loading_percent
        })

    # -------------------- Step 3b: Thermal Overload Check for Transformers --------------------
    print("[3b] Checking for thermal overloads on transformers...")
    trafo_overloads = net.res_trafo[net.res_trafo.loading_percent > max_trafo_loading_pct + FLOAT_TOL]
    print(f"    🌡️ Transformer Overloads: {len(trafo_overloads)}")

    for idx, row in trafo_overloads.iterrows():
        trafo_name = trafo_display_name(net, int(idx))
        results.append({
            "timestamp": timestamp,
            "violation_type": "trafo_loading",
            "element_index": idx,
            "element": trafo_name,
            "value": row.loading_percent
        })

    # -------------------- Step 4: Reporting Dashboard --------------------
    print("\n📊 ================= Security Assessment Dashboard ================ 📊\n")

    if not voltage_violations.empty:
        print("🔺 Voltage Violations Detected:")
        for idx, row in voltage_violations.iterrows():
            bus_name = net.bus.at[idx, 'name'] if 'name' in net.bus.columns else f"Bus {idx}"
            print(f"    - Bus {idx} ({bus_name}): Voltage = {row.vm_pu:.3f} p.u.")
    else:
        print("✅ No voltage violations.")

    print("\n" + "-" * 70 + "\n")

    if not overloads.empty:
        print("🔺 Thermal Overloads Detected:")
        for idx, row in overloads.iterrows():
            line_name = net.line.at[idx, 'name'] if 'name' in net.line.columns else f"Line {idx}"
            from_name = net.line.at[idx, 'from_bus_name'] if 'from_bus_name' in net.line.columns else row.from_bus
            to_name = net.line.at[idx, 'to_bus_name'] if 'to_bus_name' in net.line.columns else row.to_bus
            print(f"    - Line {idx} ({line_name}): {from_name} ➝ {to_name}, Loading = {row.loading_percent:.2f}%")
    else:
        print("✅ No line overloads.")

    print("\n" + "-" * 70 + "\n")
    print("\n===============================================================================\n")

    if not trafo_overloads.empty:
        print("\n🔺 Transformer Overloads Detected:")
        for idx, row in trafo_overloads.iterrows():
            trafo_name = net.trafo.at[idx, 'name'] if 'name' in net.trafo.columns else f"Trafo {idx}"
            hv_name = net.bus.at[net.trafo.at[idx, 'hv_bus'], 'name'] if 'name' in net.bus.columns else net.trafo.at[idx, 'hv_bus']
            lv_name = net.bus.at[net.trafo.at[idx, 'lv_bus'], 'name'] if 'name' in net.bus.columns else net.trafo.at[idx, 'lv_bus']
            print(f"    - Trafo {idx} ({trafo_name}): {hv_name} ➝ {lv_name}, Loading = {row.loading_percent:.2f}%")
    else:
        print("✅ No transformer overloads.")

    print("\n" + "-" * 70 + "\n")
    print("\n===============================================================================\n")


    # The timestamp stays as given: "nominal" (the nominal point) is not a
    # date, and parsing it crashed every assessment there with a violation.
    df_results = pd.DataFrame(results)

    return net, df_results
