#!/usr/bin/env python
# coding: utf-8

# In[1]:


import re
import pandas as pd
import numpy as np
import pandapower.networks
import pandapower as pp
import os
import copy
import unicodedata


# In[2]:


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
                ['timestamp', 'violation_type', 'element_index', 'element', 'value', 'limit_min', 'limit_max']
        """


    print("\n==================== Real-Time Security Assessment Engine (RSAE) ====================")
    print(f"=====================================================================================\n")

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
    v_min, v_max = vm_lower, vm_upper
    voltage_violations = net.res_bus[(net.res_bus.vm_pu < v_min) | (net.res_bus.vm_pu > v_max)]
    print(f"    ⚡ Voltage Violations: {len(voltage_violations)}")

    for idx, row in voltage_violations.iterrows():
        _raw = net.bus.at[idx, 'name'] if 'name' in net.bus.columns else ""
        bus_name = str(_raw).strip() if str(_raw).strip() else f"Bus_{idx}"
        results.append({
            "timestamp": timestamp,
            "violation_type": "bus_vm_pu",
            "element_index": idx,
            "element": bus_name,
            "value": row.vm_pu
        })
    
    # -------------------- Step 3: Thermal Overload Check --------------------
    print("[3] Checking for thermal overloads on lines...")
    overloads = net.res_line[net.res_line.loading_percent > max_line_loading_pct]
    print(f"    🌡️ Line Overloads: {len(overloads)}")

    for idx, row in overloads.iterrows():
        _raw = net.line.at[idx, 'name'] if 'name' in net.line.columns else ""
        line_name = str(_raw).strip() if str(_raw).strip() else f"Line_{net.line.at[idx, 'from_bus']}-{net.line.at[idx, 'to_bus']}"
        results.append({
            "timestamp": timestamp,
            "violation_type": "line_loading",
            "element_index": idx,
            "element": line_name,
            "value": row.loading_percent
        })

    # -------------------- Step 3b: Thermal Overload Check for Transformers --------------------
    print("[3b] Checking for thermal overloads on transformers...")
    trafo_overloads = net.res_trafo[net.res_trafo.loading_percent > max_trafo_loading_pct]
    print(f"    🌡️ Transformer Overloads: {len(trafo_overloads)}")

    for idx, row in trafo_overloads.iterrows():
        _raw = net.trafo.at[idx, 'name'] if 'name' in net.trafo.columns else ""
        trafo_name = str(_raw).strip() if str(_raw).strip() else f"Trafo_{net.trafo.at[idx, 'hv_bus']}-{net.trafo.at[idx, 'lv_bus']}"
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


    # Convert to DataFrame
    df_results = pd.DataFrame(results)
    
    # Convert timestamp to datetime
    #df_results['timestamp'] = pd.to_datetime(df_results['timestamp'])

    if not df_results.empty and 'timestamp' in df_results.columns:
        df_results['timestamp'] = pd.to_datetime(df_results['timestamp'])

    return net, df_results


# In[4]:


def plot_violation_counts(df_results):
    """
    Generate and save a bar chart showing the number of violations by type.

    Parameters
    ----------
    df_results : pandas.DataFrame
        A DataFrame containing real-time security assessment results.
        Must include a 'violation_type' column.

    Returns
    -------
    None
        Saves the plot as a PNG image in the 'rsa_engine' directory under the current working directory.
    """

    # Check for required column
    if 'violation_type' not in df_results.columns:
        print("Error: 'violation_type' column not found in the DataFrame.")
        return

    # Count violations by type
    violation_counts = df_results['violation_type'].value_counts()

    # Create the rsa_engine directory if it doesn't exist
    output_dir = os.path.join(os.getcwd(), "rsa_engine")
    os.makedirs(output_dir, exist_ok=True)

    # Create the plot
    plt.figure(figsize=(8, 5))
    violation_counts.plot(kind='bar', color='salmon', edgecolor='black')

    # Plot styling
    plt.title("Number of Violations by Type")
    plt.xlabel("Violation Type")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Save the figure
    plot_path = os.path.join(output_dir, "violation_counts.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()  # Close to free memory

    print(f"✅ Violation count plot saved to: {plot_path}")


# In[5]:


def plot_bus_voltage_violations(df_results):
    """
    Generate and save a scatter plot of bus voltage violations by bus index.

    Parameters
    ----------
    df_results : pandas.DataFrame
        A DataFrame containing the results of real-time security assessment.
        It must include the columns: 'violation_type', 'timestamp', 'element_index', and 'value'.

    Returns
    -------
    None
        The function saves the plot as a PNG image to the 'rsa_engine' directory
        in the current working directory.
    """

    # Filter for only bus voltage magnitude violations
    bus_violations = df_results[df_results['violation_type'] == 'bus_vm_pu'].copy()

    # Ensure the timestamp is in datetime format
    bus_violations['timestamp'] = pd.to_datetime(bus_violations['timestamp'])

    # Create the output directory if it does not exist
    output_dir = os.path.join(os.getcwd(), "rsa_engine")
    os.makedirs(output_dir, exist_ok=True)

    # Initialize the plot
    plt.figure(figsize=(12, 6))

    # Create scatter plot using seaborn
    sns.scatterplot(
        data=bus_violations,
        x="element_index",       # x-axis shows the numerical index of the bus
        y="value",               # y-axis shows the voltage value
        hue="timestamp",         # color-coded by timestamp (if multiple exist)
        palette="coolwarm",
        s=100,                   # marker size
        edgecolor="black"        # marker border
    )

    # Add horizontal reference lines for voltage limits
    plt.axhline(1.05, color='green', linestyle='--', label='Upper Limit (1.05 p.u.)')
    plt.axhline(0.95, color='orange', linestyle='--', label='Lower Limit (0.95 p.u.)')

    # Set plot labels and title
    plt.title("Scatter Plot: Bus Voltage Violations")
    plt.xlabel("Bus Index")
    plt.ylabel("Voltage (p.u.)")
    plt.xticks(rotation=45)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()

    # Save the figure to the rsa_engine directory
    plot_path = os.path.join(output_dir, "bus_voltage_violations.png")
    plt.savefig(plot_path, dpi=300)

    # Show the plot on the console
    plt.show() 
    
    plt.close()  # Free up memory

    print(f"✅ Bus voltage violation plot saved to: {plot_path}")


# In[6]:


def plot_sgen_bar_log(net, filename="gen_bar_log.html"):
    
    """
    Plot and save a grouped bar chart of static generator active and reactive power 
    outputs from a Pandapower network on a logarithmic scale.

    Parameters
    ----------
    net : pandapowerNet
        A Pandapower network object. The power flow must have been run (pp.runpp),
        and net.res_sgen must contain the columns 'p_mw' and 'q_mvar'.
        
    filename : str, optional
        The name of the file to save the plot as an HTML file. Default is 
        'gen_bar_log.html'. The file will be saved in a subfolder named 'rsa_engine'
        in the current working directory.

    Returns
    -------
    None
        The function saves an interactive HTML plot in the specified directory and 
        also displays the plot in an interactive window.

    Purpose
    -------
    This function visualizes the static generator outputs (active and reactive power)
    from a Pandapower network. Absolute values are plotted to ensure negative values 
    are visible. The logarithmic Y-axis helps in displaying generators with very low 
    outputs alongside larger ones without losing clarity. This plot is particularly 
    useful for operational and analysis purposes in power system studies.

    Notes
    -----
    - Uses Plotly for interactive visualization.
    - Creates a folder 'rsa_engine' in the current working directory if it does not exist.
    - The X-axis represents the generator indices.
    - Bars for |P| and |Q| are grouped for easy comparison.
    """
    
    df = net.res_sgen.copy()
    gen_idx = df.index.astype(str)  # generator indices as labels
    
    fig = go.Figure()

    # Active power bars
    fig.add_trace(go.Bar(
        x=gen_idx,
        y=df["p_mw"].abs(),  # take absolute so negatives also visible
        name="|P| (MW)",
        marker_color="royalblue"
    ))

    # Reactive power bars
    fig.add_trace(go.Bar(
        x=gen_idx,
        y=df["q_mvar"].abs(),
        name="|Q| (Mvar)",
        marker_color="darkorange"
    ))

    # Layout with log scale
    fig.update_layout(
        barmode="group",  
        title="Static Generator Outputs (P & Q) on Log Scale",
        xaxis_title="Generator Index",
        yaxis_title="Power (log scale)",
        yaxis_type="log",   # ✅ key change: logarithmic axis
        legend=dict(title=""),
        font=dict(size=14),
        bargap=0.25
    )


    # Create output directory
    output_dir = os.path.join(os.getcwd(), "rsa_engine")
    os.makedirs(output_dir, exist_ok=True)

    # Save figure
    filepath = os.path.join(output_dir, filename)
    fig.write_html(filepath)

    # Show interactive graph
    fig.show()

    print(f"Graph saved at: {filepath}")


# In[7]:


def plot_load_bar_log(net, filename="load_bar_log.html"):

    """
    Plot and save a grouped bar chart of load active and reactive power consumption
    from a Pandapower network on a logarithmic scale.

    Parameters
    ----------
    net : pandapowerNet
        A Pandapower network object. The power flow must have been run (pp.runpp),
        and net.res_load must contain the columns 'p_mw' and 'q_mvar'.

    filename : str, optional
        The name of the file to save the plot as an HTML file. Default is 
        'load_bar_log.html'. The file will be saved in a subfolder named 'rsa_engine'
        in the current working directory.

    Returns
    -------
    None
        The function saves an interactive HTML plot in the specified directory and 
        also displays the plot in an interactive window.

    Purpose
    -------
    This function visualizes the load power consumption (active and reactive) from a 
    Pandapower network. Absolute values are plotted to ensure negative values 
    (if any) are visible. The logarithmic Y-axis helps in displaying loads with 
    very low values alongside larger ones without losing clarity. This plot is 
    particularly useful for operational and analysis purposes in power system studies.

    Notes
    -----
    - Uses Plotly for interactive visualization.
    - Creates a folder 'rsa_engine' in the current working directory if it does not exist.
    - The X-axis represents the load indices.
    - Bars for |P| and |Q| are grouped for easy comparison.
    """
    
    df = net.res_load.copy()
    load_idx = df.index.astype(str)  # load indices as labels
    
    fig = go.Figure()

    # Active power bars
    fig.add_trace(go.Bar(
        x=load_idx,
        y=df["p_mw"].abs(),
        name="|P| (MW)",
        marker_color="royalblue"
    ))

    # Reactive power bars
    fig.add_trace(go.Bar(
        x=load_idx,
        y=df["q_mvar"].abs(),
        name="|Q| (Mvar)",
        marker_color="darkorange"
    ))

    # Layout (consistent with generators)
    fig.update_layout(
        barmode="group",
        title="Load Power Consumption (P & Q) on Log Scale",
        xaxis_title="Load Index",
        yaxis_title="Power (log scale)",
        yaxis_type="log",
        legend=dict(title=""),
        font=dict(size=14),
        bargap=0.25
    )

    # Create output directory
    output_dir = os.path.join(os.getcwd(), "rsa_engine")
    os.makedirs(output_dir, exist_ok=True)

    # Save figure
    filepath = os.path.join(output_dir, filename)
    fig.write_html(filepath)

    # Show interactive graph
    fig.show()

    print(f"Graph saved at: {filepath}")


# In[ ]:




