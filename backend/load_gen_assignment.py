#!/usr/bin/env python
# coding: utf-8

# In[1]:


import re
import pandas as pd
import numpy as np
import os
import unicodedata

from element_names import measurement_name


# In[2]:


# The measurements carry active power only; reactive power comes from these.
# Loads: the default of a setting the operator (or the agent) can change per
# study or for the session (main_backend). Measured generation: only on the
# measured-substation path (a network loaded without a `gen` table).
DEFAULT_LOAD_POWER_FACTOR = 0.95
MEASURED_GENERATION_POWER_FACTOR = 0.99


def normalize(text):
    """
    Normalize a string by removing diacritics and converting to lowercase.

    Parameters:
    -----------
    text : str
        The string to normalize.

    Returns:
    --------
    str
        Normalized version of the input string.
    """
    return unicodedata.normalize('NFKC', str(text)).strip().lower()


# In[3]:


def match_substation(name, substations):
    """
    Match a bus name to its corresponding substation name based on exact match, 
    prefix match, or fallback on partial component match.

    Parameters:
    -----------
    name : str
        The bus name to match.

    substations : list of str
        A list of known substation names.

    Returns:
    --------
    str or None
        The matched substation name, or None if no match is found.
    """
    normalized_subs = [normalize(s) for s in substations]
    name_norm = normalize(name)

    for i, sub in enumerate(normalized_subs):
        if name_norm == sub:
            return substations[i]
    for i, sub in enumerate(normalized_subs):
        if name_norm.startswith(sub + " ") or name_norm.startswith(sub + "-"):
            return substations[i]
    # Last resort: a three-letter prefix shared by a word of the name and a
    # substation. Accepted only when it points at exactly one substation —
    # "Bus_13" shares "bus" with every "Bus_*", and the first-match version
    # gave it Bus_1's load while reporting nothing unmatched.
    parts = re.split(r'[\s\-()]', name_norm)
    candidates = {
        i for part in parts if len(part) >= 3
        for i, sub in enumerate(normalized_subs) if part[:3] == sub[:3]
    }
    if len({substations[i] for i in candidates}) == 1:
        return substations[candidates.pop()]
    return None


# In[4]:


def aggregated_measurements_substation(timestamp_input):
    """
    Function to aggregate smart meter data for different substations in a sample system.
    The function reads the data from an Excel file, processes it by combining columns, 
    filtering data based on a given timestamp, and calculates the total production and 
    consumption for each substation.

    Parameters:
    -----------
    timestamp_input : str
        A string representing the timestamp (in a format convertible by pandas to datetime) 
        for which the measurements need to be aggregated.

    Returns:
    --------
    pd.DataFrame
        A DataFrame containing aggregated production and consumption values for each substation 
        at the specified timestamp. Columns include 'Station', 'Consumption', and 'Production'.
    """
    
    # Get the current working directory to locate the data file
    current_directory = os.getcwd()
    
    # Define the file path for the smart meter data
    smart_meter_data = os.path.join(current_directory, 'CDK_Data_Sample.xlsx')
    
    # Step 1: Load the data from the Excel file into a pandas DataFrame
    df = pd.read_excel(smart_meter_data, engine='calamine', header=[0])

    # Step 2: Remove the first two rows to clean up the data (skip irrelevant data)
    df = df.iloc[2:]

    # Step 3: Extract new column names from the second row onward and append them to the existing columns
    new_col_parts = df.iloc[0, 1:].astype(str).tolist()
    df.columns = [df.columns[0]] + [f"{df.columns[i]}+{new_col_parts[i-1]}" for i in range(1, len(df.columns))]
    
    # Step 4: Drop the first row now that its information has been incorporated into column names
    df = df.iloc[1:].reset_index(drop=True)

    # Step 5: Combine the 'Date' and 'Time' columns into a single 'Date&time' column
    df['Date&time'] = pd.to_datetime(df.iloc[:, 0].astype(str) + ' ' + df.iloc[:, 1].astype(str), errors='coerce')
    
    # Step 6: Drop the original 'Date' and 'Time' columns after combining them
    df.drop(df.columns[[0, 1]], axis=1, inplace=True)
    
    # Step 7: Reorder columns so 'Date&time' appears first
    cols = ['Date&time'] + [col for col in df.columns if col != 'Date&time']
    df = df[cols]
    
    # Step 8: Set 'Date&time' as the DataFrame index for easy filtering
    df.set_index('Date&time', inplace=True)

    # Step 9: Convert the input timestamp to a pandas datetime object
    timestamp = pd.to_datetime(timestamp_input)

    # Step 10: Filter the data to obtain the row corresponding to the input timestamp
    row = df.loc[[timestamp]]

    # Step 11: Define the names of all substations for which data will be aggregated
    station_names = ['Alpha', 'Bravo', 'Charlie', 'Delta', 'Echo', 'Foxtrot',
                     'Golf', 'Hotel', 'India', 'Juliet North', 'Juliet South',
                     'Kilo', 'Lima', 'Mike', 'November', 'Oscar']

    # Step 12: Prepare a list to store the results for each station
    results = []
    
    # Step 13: Loop over each station and calculate the production and consumption sums
    for station in station_names:
        # Step 13.1: Find all columns associated with the current station
        station_cols = [col for col in row.columns if str(col).startswith(station)]
    
        # Step 13.2: Calculate the sum of production values (columns ending with '-2')
        production_cols = [col for col in station_cols if str(col).endswith('-2')]
        sum_of_prod_values = np.nansum(row[production_cols].values)
    
        # Step 13.3: Calculate the sum of consumption values (columns ending with '-1')
        consumption_cols = [col for col in station_cols if str(col).endswith('-1')]
        sum_of_cons_values = np.nansum(row[consumption_cols].values)
    
        # Step 13.4: Convert the production and consumption sums to kWh (dividing by 1000)
        production_sum = sum_of_prod_values / 1000
        consumption_sum = sum_of_cons_values / 1000
    
        # Step 13.5: Store the results for the current station
        results.append({'substation_name': station, 'consumption': consumption_sum, 'production': production_sum})
    
    # Step 14: Convert the results list to a DataFrame
    measurement_prod_cons = pd.DataFrame(results)

    # Step 15: Rename 'India' to 'India'
    measurement_prod_cons['substation_name'] = measurement_prod_cons['substation_name'].replace('India', 'India')
    
    # Step 15: Return the final DataFrame containing the aggregated production and consumption
    return measurement_prod_cons


# In[5]:


def _proposed_mask(df: pd.DataFrame) -> pd.Series:
    """Rows a network proposal added (`proposed` is true).

    They follow their own profile, not the measurements: a 50 MW data centre
    added on a measured bus used to receive a share of that bus's measured
    consumption — or all of it — at every tick, and vanish (sandbox hazard
    A.3 (1)). Every assignment below leaves these rows alone.
    """
    if "proposed" not in df.columns or not len(df):
        return pd.Series(False, index=df.index)
    return df["proposed"].map(lambda v: bool(v) if pd.notna(v) else False).astype(bool)


def assign_load_values_from_measurements(net, measurement_prod_cons, substations,
                                         power_factor=DEFAULT_LOAD_POWER_FACTOR):
    """
    Assigns active (p_mw) and reactive (q_mvar) power values to loads in a pandapower network
    based on substation-level consumption measurements.

    Steps:
    1. Normalize names in bus and load tables.
    2. Drop unwanted load entries by name.
    3. Map bus indices to bus names.
    4. Match loads to substations.
    5. Assign active power (p_mw) using substation consumption.
    6. Calculate reactive power (q_mvar) from `power_factor` (lagging; default 0.95):
       the measurements carry active power only.
    7. Drop temporary columns and retain original indices.

    Parameters:
    -----------
    net : pandapowerNet
        The pandapower network object to be updated.

    measurement_prod_cons : pandas.DataFrame
        DataFrame containing substation-level 'substation_name' and 'consumption' columns.

    substations : list of str
        List of valid substation names used for matching.

    Returns:
    --------
    net : pandapowerNet
        The updated pandapower network with p_mw and q_mvar assigned to net.load.
    """

    # -------- Step 1: Standardize names in bus and load --------
    if 'Gl Dampmike C afg Load' in net.bus['name'].values:
        net.bus['name'] = net.bus['name'].replace({'Gl Dampmike C afg Load': 'Mike 10kV'})
        net.load['name'] = net.load['name'].replace({'00 Gl Dampværk Load': '00 Vær Load'})

    # -------- Step 2: Drop unwanted load entries --------
    # Check if any of the unwanted load names are present
    names_to_drop = ['Blok 5', 'Blok 6', 'Diesel generatorer']
    if any(name in net.load['name'].values for name in names_to_drop):
        net.load = net.load[~net.load['name'].isin(names_to_drop)].copy()

    # -------- Step 3: Map bus index to bus name (Bus_{idx} fallback for empty/missing names) --------
    # The name as loaded: a bus the operator renamed keeps its measurements.
    bus_name_map = {int(idx): measurement_name(net, idx) for idx in net.bus.index}
    net.load['bus_name'] = net.load['bus'].map(bus_name_map)

    # -------- Step 4: Assign substation names using helper function --------
    # Prefer matching against measurement substation names (the canonical short
    # names like 'Bravo') so the lookup in Step 5 works for measured substation-style networks
    # where bus names have kV suffixes ('Bravo 10 kV'). Falls back to the
    # network-derived substations list for IEEE / MATPOWER networks.
    meas_sub_names = (
        measurement_prod_cons['substation_name'].dropna().tolist()
        if 'substation_name' in measurement_prod_cons.columns
        else []
    )
    match_pool = meas_sub_names if meas_sub_names else substations
    net.load['substation_name'] = net.load['bus_name'].apply(
        lambda name: match_substation(name, match_pool)
    )

    # -------- Step 5: Assign active power (p_mw) using external measurement data --------
    # Preserve the original index
    original_index = net.load.index

    # Create a mapping from substation name to consumption value
    consumption_map = measurement_prod_cons.set_index('substation_name')['consumption']
    net.load['consumption'] = net.load['substation_name'].map(consumption_map)
    net.load['_meas_key'] = net.load['substation_name'].where(net.load['consumption'].notna())

    # T6 fallback: for generic/IEEE networks where fuzzy substation matching returns
    # None, try a direct match on the raw bus name (measurement rows use bus names).
    unmatched = net.load['consumption'].isna()
    if unmatched.any():
        net.load.loc[unmatched, 'consumption'] = (
            net.load.loc[unmatched, 'bus_name'].map(consumption_map)
        )
        net.load.loc[unmatched, '_meas_key'] = net.load.loc[unmatched, 'bus_name'].where(
            net.load.loc[unmatched, 'consumption'].notna())

    # Proposed loads are not measured: no share of a measured total, and their
    # own p_mw / q_mvar stay as the proposal set them.
    own = _proposed_mask(net.load)
    if own.any():
        net.load.loc[own, 'consumption'] = np.nan
        net.load.loc[own, '_meas_key'] = np.nan

    # A measurement is the total of its substation (or bus). Every load matched
    # to it used to receive that whole total, so two loads on one bus doubled
    # the measured demand (309 MW against 223 measured in a reproduction).
    # Split it across them by their share in the network as first loaded —
    # kept in `_meas_weight`, which is set once, before any measured value
    # has overwritten `p_mw`. A substation with one load is unaffected.
    if '_meas_weight' not in net.load.columns:
        net.load['_meas_weight'] = net.load['p_mw'].clip(lower=0).fillna(0.0)
    keyed = net.load['_meas_key'].notna()
    share_all = pd.Series(1.0, index=net.load.index)
    if keyed.any():
        weights = net.load.loc[keyed, '_meas_weight'].astype(float)
        keys = net.load.loc[keyed, '_meas_key']
        totals = weights.groupby(keys).transform('sum')
        counts = keys.groupby(keys).transform('count')
        share = (weights / totals).where(totals > 0, 1.0 / counts)
        share_all.loc[keyed] = share
        net.load.loc[keyed, 'consumption'] = net.load.loc[keyed, 'consumption'] * share

    # Identify buses that are still unmatched after both attempts.
    still_unmatched = net.load['consumption'].isna() & ~own
    unmatched_bus_names = (
        net.load.loc[still_unmatched, 'bus_name'].dropna().unique().tolist()
    )

    # Only overwrite p_mw / q_mvar for matched rows — unmatched rows keep their
    # original base-case values so the power flow doesn't receive NaN injections.
    matched = ~still_unmatched & ~own
    net.load.loc[matched, 'p_mw'] = net.load.loc[matched, 'consumption']
    net.load.loc[matched, 'q_mvar'] = (
        net.load.loc[matched, 'p_mw'] * np.tan(np.arccos(power_factor))
    )
    # Measured reactive demand, when the data carry it, always wins over the
    # power factor — split over the substation's loads like its active power.
    # Where it is missing (NaN) the power factor above stays.
    measured_share = 0.0
    if 'consumption_q' in measurement_prod_cons.columns:
        # One value per substation (a name listed twice: its total; all missing: NaN).
        q_map = measurement_prod_cons.groupby('substation_name')['consumption_q'].sum(min_count=1)
        q_meas = net.load['_meas_key'].map(q_map) * share_all
        use = matched & q_meas.notna()
        net.load.loc[use, 'q_mvar'] = q_meas[use].astype(float)
        total = float(net.load.loc[matched, 'p_mw'].clip(lower=0).sum())
        if total > 0:
            measured_share = float(net.load.loc[use, 'p_mw'].clip(lower=0).sum()) / total
        elif use.any():
            measured_share = 1.0
    net["_q_measured_share"] = round(measured_share, 4)

    # -------- Step 6 (old): reactive power already handled above --------

    # -------- Step 7: Clean up --------
    net.load.drop(columns=['consumption', '_meas_key'], inplace=True)
    net.load.index = original_index  # Restore original indices

    return net, unmatched_bus_names


# In[ ]:


def _update_gen_table_inplace(gen_df: pd.DataFrame, bus_id_to_prod: dict, scale_q: bool = False) -> None:
    """Update p_mw in gen_df (net.gen or net.sgen) from per-bus production values.

    Production is distributed across generators on the same bus proportionally
    to max_p_mw.  If max_p_mw is unavailable or zero, it is split equally.
    Modifies gen_df in place.

    scale_q : bool
        If True, scale q_mvar by the same ratio as p_mw so the power factor
        stays constant.  Should be True for PQ elements (sgen) and False for
        PV elements (gen) where the solver determines Q.
    """
    own = _proposed_mask(gen_df)
    for bus_id, total_prod in bus_id_to_prod.items():
        indices = gen_df.index[(gen_df['bus'] == bus_id) & ~own].tolist()
        if not indices:
            continue
        max_vals = []
        for idx in indices:
            v = gen_df.at[idx, 'max_p_mw'] if 'max_p_mw' in gen_df.columns else 0
            max_vals.append(max(float(v) if pd.notna(v) else 0.0, 0.0))
        total_max = sum(max_vals)
        for i, idx in enumerate(indices):
            old_p = float(gen_df.at[idx, 'p_mw']) if pd.notna(gen_df.at[idx, 'p_mw']) else 0.0
            if total_max > 0:
                new_p = total_prod * max_vals[i] / total_max
            else:
                new_p = total_prod / len(indices)
            gen_df.at[idx, 'p_mw'] = new_p
            if scale_q and 'q_mvar' in gen_df.columns and old_p != 0.0:
                ratio = new_p / old_p
                gen_df.at[idx, 'q_mvar'] = float(gen_df.at[idx, 'q_mvar']) * ratio


def _assign_measured_generation_q(net, measurement_prod_cons, bus_name_map) -> None:
    """Measured reactive generation, when the data carry it: given to the
    fixed-power-factor / fixed-Q units (`sgen`) of each measured bus, split by
    capacity, and marked so the units' own model does not overwrite it. A bus
    with a voltage-controlled unit is left alone — that unit decides its own
    reactive power (the data-quality report says so)."""
    if len(net.sgen):
        net.sgen['_q_from_meas'] = False
    if 'production_q' not in measurement_prod_cons.columns or not len(net.sgen):
        return
    q_map = measurement_prod_cons.groupby('substation_name')['production_q'].sum(min_count=1)
    own_s, own_g = _proposed_mask(net.sgen), _proposed_mask(net.gen) if len(net.gen) else None
    for bus_id, name in bus_name_map.items():
        q = q_map.get(name)
        if q is None or pd.isna(q):
            continue
        if len(net.gen) and ((net.gen['bus'] == bus_id) & ~own_g).any():
            continue
        rows = net.sgen.index[(net.sgen['bus'] == bus_id) & ~own_s]
        if not len(rows):
            continue
        weights = net.sgen.loc[rows, 'max_p_mw'] if 'max_p_mw' in net.sgen.columns else pd.Series(1.0, index=rows)
        w = weights.clip(lower=0).fillna(0.0).astype(float)
        w = w / w.sum() if w.sum() > 0 else pd.Series(1.0 / len(rows), index=rows)
        net.sgen.loc[rows, 'q_mvar'] = (w * float(q)).astype(float)
        net.sgen.loc[rows, '_q_from_meas'] = True


def _capacity_on_bus(gen_df: pd.DataFrame, bus_id) -> tuple[float, int]:
    """(total max_p_mw, number of units) of `gen_df` on one bus."""
    rows = gen_df[gen_df['bus'] == bus_id] if len(gen_df) else gen_df
    if 'max_p_mw' in rows.columns:
        cap = float(rows['max_p_mw'].clip(lower=0).fillna(0.0).sum())
    else:
        cap = 0.0
    return cap, len(rows)


def _capacities_by_bus(gen_df: pd.DataFrame) -> tuple[dict, dict]:
    """({bus: total max_p_mw}, {bus: number of units}) of `gen_df` — what
    `_capacity_on_bus` gives for one bus, for every bus at once."""
    if not len(gen_df):
        return {}, {}
    cap = (gen_df['max_p_mw'].clip(lower=0).fillna(0.0).astype(float) if 'max_p_mw' in gen_df.columns
           else pd.Series(0.0, index=gen_df.index))
    grouped = cap.groupby(gen_df['bus'])
    return {k: float(v) for k, v in grouped.sum().items()}, {k: int(v) for k, v in grouped.size().items()}


def _split_between_tables(gen_df: pd.DataFrame, sgen_df: pd.DataFrame,
                          bus_id_to_prod: dict) -> tuple[dict, dict]:
    """Divide each bus's production between net.gen and net.sgen.

    By installed capacity when known, otherwise by number of units; a bus with
    units in only one table gives that table everything, as before.
    """
    gen_share: dict = {}
    sgen_share: dict = {}
    # Capacities by bus, once per call: filtering both tables bus by bus cost
    # most of every moment's assignment (an hour of a week's scan).
    caps_g, counts_g = _capacities_by_bus(gen_df)
    caps_s, counts_s = _capacities_by_bus(sgen_df)
    for bus_id, total in bus_id_to_prod.items():
        cap_g, n_g = caps_g.get(bus_id, 0.0), counts_g.get(bus_id, 0)
        cap_s, n_s = caps_s.get(bus_id, 0.0), counts_s.get(bus_id, 0)
        if n_g == 0 or n_s == 0:
            gen_share[bus_id] = sgen_share[bus_id] = total
            continue
        frac_g = cap_g / (cap_g + cap_s) if cap_g + cap_s > 0 else n_g / (n_g + n_s)
        gen_share[bus_id] = total * frac_g
        sgen_share[bus_id] = total * (1.0 - frac_g)
    return gen_share, sgen_share


def assign_generators_values_from_measurements(net, measurement_prod_cons, substations):
    """
    Assign static generator (sgen) data in a pandapower network using measured production data.

    This function populates the net.sgen DataFrame by copying bus and name information from net.load,
    replaces load names with generator names, and assigns active (p_mw) and reactive (q_mvar) power values 
    based on measured production data from the measurement_prod_cons DataFrame. On the
    measured-substation path (a network with no `gen` table) q_mvar follows
    MEASURED_GENERATION_POWER_FACTOR (0.99); on the generic path each unit keeps its
    own power factor (Q scales with P — 0 for units converted from `gen`).

    Parameters:
    -----------
    net : pandapowerNet
        The pandapower network object to which sgens will be added.

    measurement_prod_cons : pandas.DataFrame
        A DataFrame containing at least the columns 'substation_name' and 'production' (in MW).
        This is used to assign active power (p_mw) values to the generators.

    substations : list of str
        List of known substation names used for validation or reference, not directly modified here.

    Returns:
    --------
    net : pandapowerNet
        The updated pandapower network with net.sgen populated with generator values.
    """

    # T6 — Route: measured-substation rebuild path vs generic/IEEE path
    # (preserve existing gen/sgen, update p_mw by bus-name match).
    # Detection: use Generic path if net.gen has entries, OR if gen_to_sgen conversion
    # was applied (net.gen was emptied but sgens were created from it).
    if len(net.gen) > 0 or net.get("_gen_to_sgen_applied", False):
        # --- Generic / IEEE / MATPOWER path ---
        # Build bus_id → production from measurement rows (keyed by bus name)
        prod_map = measurement_prod_cons.set_index('substation_name')['production']
        bus_name_map_local = {int(idx): measurement_name(net, idx) for idx in net.bus.index}
        bus_id_to_prod = {
            bus_id: float(prod_map[bname])
            for bus_id, bname in bus_name_map_local.items()
            if bname in prod_map.index
        }
        # A bus with both a `gen` and an `sgen` used to receive its full
        # production in each table. Split it once across both, by capacity.
        gen_share, sgen_share = _split_between_tables(
            net.gen[~_proposed_mask(net.gen)], net.sgen[~_proposed_mask(net.sgen)], bus_id_to_prod)
        _update_gen_table_inplace(net.gen, gen_share, scale_q=False)
        if len(net.sgen) > 0:
            _update_gen_table_inplace(net.sgen, sgen_share, scale_q=True)
        _assign_measured_generation_q(net, measurement_prod_cons, bus_name_map_local)

        # Identify generator buses that had no matching row in the CSV.
        gen_buses = set(net.gen['bus'].tolist()) | set(net.sgen['bus'].tolist())
        unmatched_gen_bus_names = [
            bus_name_map_local.get(bid, f"Bus_{bid}")
            for bid in gen_buses
            if bid not in bus_id_to_prod
        ]
        return net, unmatched_gen_bus_names

    # --- Measured-substation path (original code): rebuild net.sgen from net.load structure ---
    # Proposed units are set aside and appended again after the rebuild.
    kept = net.sgen[_proposed_mask(net.sgen)].copy() if len(net.sgen) else None
    # A unit a proposal switched off stays off: the rebuild used to create
    # every unit in service again at each tick.
    switched_off = (set(net.sgen.loc[~net.sgen["in_service"].astype(bool), "name"].astype(str))
                    if len(net.sgen) and "in_service" in net.sgen.columns else set())

    # --- Step 0: Initialize empty net.sgen DataFrame with appropriate structure
    net.sgen = pd.DataFrame(columns=[
        'name', 'bus', 'p_mw', 'q_mvar', 'sn_mva', 'scaling', 'in_service', 'type'
    ])

    # Enforce correct datatypes
    net.sgen = net.sgen.astype({
        'name': 'str',
        'bus': 'int',
        'p_mw': 'float',
        'q_mvar': 'float',
        'sn_mva': 'float',
        'scaling': 'float',
        'in_service': 'bool',
        'type': 'str'
    })

    # --- Step 1: Generate 'name' column for sgen from load names
    sgen_name = net.load['name'].str.replace('Load', 'Sgen', case=False)

    # --- Step 2: Copy 'bus' and optional 'substation_name' columns from net.load
    sgen_bus = net.load['bus']
    sgen_substation_name = net.load['substation_name'] if 'substation_name' in net.load.columns else None

    # --- Step 3: Set default values for new sgen fields
    sgen_scaling = 1.0
    sgen_in_service = True

    # --- Step 4: Create net.sgen with initial values
    net.sgen = pd.DataFrame({
        'name': sgen_name,
        'bus': sgen_bus,
        'p_mw': np.nan,       # Placeholder for active power
        'q_mvar': np.nan,     # Placeholder for reactive power
        'sn_mva': np.nan,     # Optional; can be filled if known
        'scaling': sgen_scaling,
        'in_service': sgen_in_service,
        'type': 'static'
    })

    # --- Step 5: Add substation_name if available
    if sgen_substation_name is not None:
        net.sgen['substation_name'] = sgen_substation_name

    # --- Step 6: Merge production data based on substation_name
    measured_cols = ['substation_name', 'production'] + (
        ['production_q'] if 'production_q' in measurement_prod_cons.columns else [])
    net.sgen = net.sgen.merge(
        measurement_prod_cons[measured_cols],
        on='substation_name',
        how='left'
    )

    # --- Step 7: Assign p_mw (note: sign convention; sgens inject power → positive)
    net.sgen['p_mw'] = -1 * net.sgen['production']

    # --- Step 8: reactive power at the measured-generation power factor (0.99;
    # the comment here used to say 0.95 while the code used 0.99)
    net.sgen['q_mvar'] = net.sgen['p_mw'] * np.tan(np.arccos(MEASURED_GENERATION_POWER_FACTOR))
    # Measured reactive generation wins where the data carry it (same sign
    # convention as the production column: it is negated like p_mw above).
    net.sgen['_q_from_meas'] = False
    if 'production_q' in net.sgen.columns:
        has = net.sgen['production_q'].notna()
        net.sgen.loc[has, 'q_mvar'] = -1 * net.sgen.loc[has, 'production_q'].astype(float)
        net.sgen.loc[has, '_q_from_meas'] = True
        net.sgen.drop(columns=['production_q'], inplace=True)

    # --- Step 9: Clean up
    net.sgen.drop(columns=['production'], inplace=True)
    if switched_off:
        net.sgen.loc[net.sgen['name'].astype(str).isin(switched_off), 'in_service'] = False
    if kept is not None and len(kept):
        kept.index = range(len(net.sgen), len(net.sgen) + len(kept))
        net.sgen = pd.concat([net.sgen, kept])

    return net, []  # measured-substation path rebuilds sgen fully — no frozen buses


# In[ ]:





# In[ ]:




