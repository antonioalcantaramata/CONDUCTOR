#!/usr/bin/env python
# coding: utf-8

# In[1]:


# ===========================
# Import Required Libraries
# ===========================
#import re
import pandas as pd
import numpy as np
#import pandapower.networks
import pandapower as pp
import os
import copy
import unicodedata
from pandapower.powerflow import LoadflowNotConverged


# In[2]:


def update_bus_mapping(net_copy, bus_num):

    """
    Updates bus indices in a pandapower network after a bus removal.

    This function adjusts all references to buses in the network elements 
    (lines, transformers, shunts, loads, and generators) so that bus numbering
    remains consistent after a bus is deleted. Any bus with an index greater than
    the deleted bus will be decremented by 1.

    Parameters
    ----------
    net_copy : pandapowerNet
        A copy of the pandapower network to be updated.
    bus_num : int
        The index of the bus that has been removed. All bus references 
        higher than this index will be decremented by 1.

    Returns
    -------
    pandapowerNet
        The updated pandapower network with corrected bus mappings.
    """

    # Update line connections
    net_copy.line['from_bus'] = net_copy.line['from_bus'].apply(
        lambda x: x - 1 if x > bus_num else x
    )
    net_copy.line['to_bus'] = net_copy.line['to_bus'].apply(
        lambda x: x - 1 if x > bus_num else x
    )

    # Update transformer connections
    net_copy.trafo['hv_bus'] = net_copy.trafo['hv_bus'].apply(
        lambda x: x - 1 if x > bus_num else x
    )
    net_copy.trafo['lv_bus'] = net_copy.trafo['lv_bus'].apply(
        lambda x: x - 1 if x > bus_num else x
    )

     # Update shunts, loads, and generators
    net_copy.shunt['bus'] = net_copy.shunt['bus'].apply(
        lambda x: x - 1 if x > bus_num else x
    )

    net_copy.load['bus'] = net_copy.load['bus'].apply(
        lambda x: x - 1 if x > bus_num else x
    )

    net_copy.sgen['bus'] = net_copy.sgen['bus'].apply(
        lambda x: x - 1 if x > bus_num else x
    )

    return net_copy


# In[3]:


def remove_trafo_and_connected_elements(net_copy, trafo_index):
    """
    Removes a transformer (and all elements connected to its LV bus)
    from a deep copy of the given pandapower network, without altering the original.

    Parameters
    ----------
    net : pandapowerNet
        The pandapower network object.
    trafo_index : int
        Index of the transformer to remove.

    Returns
    -------
    net_copy : pandapowerNet
        A deep-copied and modified network with the selected transformer and its
        connected elements (sgen, load, shunt, and bus) removed.
    """

    # Check transformer index validity
    if trafo_index not in net_copy.trafo.index:
        raise ValueError(f"Transformer index {trafo_index} not found in net.trafo.")

    # --- 1️ Get the LV bus of that transformer ---
    lv_bus_value = net_copy.trafo.loc[trafo_index, 'lv_bus']
    print(f"LV bus of transformer {trafo_index}: {lv_bus_value}")

    # --- 2 Get the HV bus of that transformer ---
    hv_bus_value = net_copy.trafo.loc[trafo_index, 'hv_bus']
    print(f"HV bus of transformer {trafo_index}: {hv_bus_value}")

    # --- 3 Remove the transformer row ---
    net_copy.trafo = net_copy.trafo.drop(index=trafo_index).reset_index(drop=True)

    # --- 4 Remove connected sgens ---
    mask_sgen = net_copy.sgen['bus'] == lv_bus_value
    net_copy.sgen = net_copy.sgen.drop(index=net_copy.sgen[mask_sgen].index).reset_index(drop=True)

    # --- 5 Remove connected loads ---
    mask_load = net_copy.load['bus'] == lv_bus_value
    net_copy.load = net_copy.load.drop(index=net_copy.load[mask_load].index).reset_index(drop=True)

    # --- 6 Remove connected shunts ---
    mask_shunt = net_copy.shunt['bus'] == lv_bus_value
    net_copy.shunt = net_copy.shunt.drop(index=net_copy.shunt[mask_shunt].index).reset_index(drop=True)

    # --- 7 Remove the LV bus itself ---
    if lv_bus_value in net_copy.bus.index:
        net_copy.bus = net_copy.bus.drop(index=lv_bus_value).reset_index(drop=True)
    else:
        print(f"Bus {lv_bus_value} not found in net.bus — skipping removal.")

    # --- 8 Reindex bus references in all elements after LV bus removal ---
    # When the LV bus is removed, all bus indices above it must be shifted down by 1.
    # This maintains internal consistency since bus indices represent row positions.

    net_copy = update_bus_mapping(net_copy, lv_bus_value)

    # --- 9 Remove corresponding controller entry if present ---
    net_copy.controller = net_copy.controller.drop(index=trafo_index).reset_index(drop=True)

    print(f"✅ Transformer {trafo_index} and connected elements removed successfully.")
    return net_copy, hv_bus_value, lv_bus_value


# In[4]:


def remove_inactive_buses(net_copy):

    """
    Removes all inactive buses from a pandapower network and updates 
    the network bus references to maintain consistency.

    This function iteratively identifies buses marked as inactive 
    (`in_service == False`), removes them from the network, and then 
    updates all bus indices in connected elements (lines, transformers, 
    loads, shunts, and generators) using the `update_bus_mapping` function.

    Parameters
    ----------
    net_copy : pandapowerNet
        A pandapower network copy in which inactive buses will be removed.

    Returns
    -------
    pandapowerNet
        The updated pandapower network with all inactive buses removed 
        and bus indices adjusted accordingly.
    """

    # Step 1: Identify inactive buses
    inactive_buses_len = net_copy.bus[net_copy.bus['in_service'] == False]
    
    for i in range(0, len(inactive_buses_len)):
    
        inactive_bus = net_copy.bus[net_copy.bus['in_service'] == False].index.to_list()[0]
        
        net_copy.bus = net_copy.bus.drop(index=inactive_bus).reset_index(drop=True)
        
        net_copy = update_bus_mapping(net_copy, inactive_bus)

    return net_copy


# In[5]:


def remove_isolated_line_buses_with_trafo(net_copy, line_index):
    """
    Always removes the given line (line_index) from net.line.
    Checks if from_bus or to_bus of that line are isolated.
    If isolated buses are found:
      • Removes associated transformer (based on hv_bus)
      • Removes sgen, load, shunt on corresponding lv_bus
      • Removes both lv_bus & isolated buses.
    
    Returns:
        net_copy, from_bus, to_bus, lv_bus_values
    """

    if line_index not in net_copy.line.index:
        raise ValueError(f"Line index {line_index} not found in net.line.")

    # Extract from_bus and to_bus before removing line
    from_bus = net_copy.line.loc[line_index, 'from_bus']
    to_bus = net_copy.line.loc[line_index, 'to_bus']

    # ✅ Always remove the line
    net_copy.line = net_copy.line.drop(index=line_index).reset_index(drop=True)

    # Determine isolation in the updated line set
    other_lines = net_copy.line

    isolated_buses = []
    if not (from_bus in other_lines['from_bus'].values or from_bus in other_lines['to_bus'].values):
        isolated_buses.append(from_bus)

    if not (to_bus in other_lines['from_bus'].values or to_bus in other_lines['to_bus'].values):
        isolated_buses.append(to_bus)

    # ✅ If no isolated bus -> return now (empty lv_bus_values)
    if not isolated_buses:
        return net_copy, from_bus, to_bus, []

    # ✅ NEW: store all LV buses encountered
    lv_bus_values = []

    # Process isolation
    for bus in isolated_buses:

        # Find trafo where this bus is the HV side
        mask_trafo = net_copy.trafo['hv_bus'] == bus
        if mask_trafo.any():
            trafo_idx = net_copy.trafo[mask_trafo].index[0]
            lv_bus_value = net_copy.trafo.loc[trafo_idx, 'lv_bus']

            # Store LV bus used
            lv_bus_values.append(lv_bus_value)

            # Remove transformer
            net_copy.trafo = net_copy.trafo.drop(index=trafo_idx).reset_index(drop=True)

            # Remove connected SGens
            mask_sgen = net_copy.sgen['bus'] == lv_bus_value
            net_copy.sgen = net_copy.sgen.drop(index=net_copy.sgen[mask_sgen].index).reset_index(drop=True)

            # Remove connected Loads
            mask_load = net_copy.load['bus'] == lv_bus_value
            net_copy.load = net_copy.load.drop(index=net_copy.load[mask_load].index).reset_index(drop=True)

            # Remove connected Shunts
            mask_shunt = net_copy.shunt['bus'] == lv_bus_value
            net_copy.shunt = net_copy.shunt.drop(index=net_copy.shunt[mask_shunt].index).reset_index(drop=True)

            # Remove LV bus itself
            if lv_bus_value in net_copy.bus.index:
                net_copy.bus = net_copy.bus.drop(index=lv_bus_value).reset_index(drop=True)

                net_copy = update_bus_mapping(net_copy, lv_bus_value)

        # Remove the isolated (HV) bus itself
        if bus in net_copy.bus.index:
            net_copy.bus = net_copy.bus.drop(index=bus).reset_index(drop=True)

            net_copy = update_bus_mapping(net_copy, bus)

    print(f"✅ Processed line {line_index}. Removed isolated buses: {isolated_buses}, LV buses removed: {lv_bus_values}")

    return net_copy, from_bus, to_bus, lv_bus_values


# In[ ]:





# In[ ]:




