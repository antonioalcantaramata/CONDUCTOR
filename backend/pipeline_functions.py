import json
import pathlib
from functools import lru_cache
import requests
import csv
from io import StringIO
from typing import List
from datetime import datetime
import os
#from pipeline_functions import *
import pandas as pd

def get_ids_by_substation(*substation: str) -> List[int]:
    """
    Return the datastream_id for a given substation and parameter using the metadata dictionary.
    If '*' is passed as a substation, it returns IDs for all substations.
    Returns an empty list if no IDs are found or if a specified substation is not in metadata.
    """
    ID_list = []
    
    # Check if '*' is among the requested substations
    if '*' in substation:
        # If '*' is present, collect IDs from all substations
        for sub_data in metadata.values():
            ID_list.extend(sub_data.values())
        # If '*' is the only argument, we are done.
        # If other specific substations are also provided with '*',
        # the behavior here is to get all, effectively ignoring others.
        # A more complex logic could be implemented if a mix of '*' and specific
        # substations should result in a union or difference.
        # For this request, '*' implies "all".
        return ID_list
    
    # If '*' is not present, process each specified substation
    for sub in substation:
        if sub in metadata:
            ID_list.extend(metadata[sub].values())
        else:
            print(f"Substation '{sub}' not found in metadata.")
            
    return ID_list


def _load_datastream_map() -> dict:
    """Read the datastream map from disk, preferring the real one.

    The map ties datastream ids to substation names for one specific utility.
    Those names and ids are operational metadata about a real network, so they
    live in `datastream_map.local.json`, which is gitignored, and only the
    synthetic `datastream_map.example.json` ships with the repository. Keeping
    them out of the source is also why this is loaded rather than hardcoded —
    a literal dict here is a literal dict in every clone of the repo.

    Set `CONDUCTOR_DATASTREAM_MAP` to point at a map elsewhere.
    """
    here = pathlib.Path(__file__).parent
    override = os.environ.get("CONDUCTOR_DATASTREAM_MAP")
    for candidate in ([pathlib.Path(override)] if override else []) + [
        here / "datastream_map.local.json",
        here / "datastream_map.example.json",
    ]:
        if candidate.exists():
            with open(candidate, "r", encoding="utf-8") as fh:
                return json.load(fh)
    return {"datastreams": {}, "transformer_p_c": {}, "substations": {},
            "public_ids": [], "public_id_labels": {}}


@lru_cache(maxsize=1)
def _datastream_map() -> dict:
    """Parsed once: the file does not change while the process runs."""
    raw = _load_datastream_map()
    parsed = {
        section: {int(k): v for k, v in (raw.get(section) or {}).items()}
        for section in ("datastreams", "transformer_p_c")
    }
    parsed["public_ids"] = [int(i) for i in (raw.get("public_ids") or [])]
    parsed["public_id_labels"] = {
        int(k): str(v) for k, v in (raw.get("public_id_labels") or {}).items()
    }
    # Keyed by name rather than by id, so its keys stay strings.
    parsed["substations"] = {
        str(sub): {str(param): int(sid) for param, sid in (entries or {}).items()}
        for sub, entries in (raw.get("substations") or {}).items()
    }
    return parsed


# Substation -> {parameter: datastream_id}. The same facts as the
# `datastreams` section, keyed the other way round; both shapes are in use.
metadata = _datastream_map()["substations"]


def get_public_ids() -> list:
    """The datastream ids the backend's own EDDK endpoint fetches.

    Real ids identify real measurement points on a real utility's network, so
    they live in the map file with everything else rather than as a literal in
    the source.
    """
    return list(_datastream_map()["public_ids"])


def get_public_id_labels() -> dict:
    """Display names for the public ids, where the metadata lookup has none."""
    return dict(_datastream_map()["public_id_labels"])


def get_datastream_metadata():
    """Return a dictionary mapping datastream_id to substation and parameter names."""
    return dict(_datastream_map()["datastreams"])


def group_ids_by_substation():
    """Group datastream IDs by substation from the metadata DataFrame.
    Returns a dictionary where keys are substation names and values are lists of datastream IDs.
    """
    metadata_df = pd.DataFrame.from_dict(get_datastream_metadata(), orient='index')
    return metadata_df.groupby("substation")["id"].apply(list).to_dict()

def get_id_by_substation_and_parameter(substation: str, parameter: str) -> str:
    """
    Return the datastream_id for a given substation and parameter using the metadata dictionary.
    Returns None if not found.
    """
    metadata = get_datastream_metadata()
    # Build a reverse mapping: (substation, parameter) -> id
    reverse_map = { (v['substation'], v['parameter']): k for k, v in metadata.items() }
    return reverse_map.get((substation, parameter))

def define_timespan(start: str, end: str):
    date_obj = datetime.strptime(start, '%Y-%m-%d')
    start = date_obj.strftime('%Y-%m-%dT00:00:00')
    date_obj = datetime.strptime(end, '%Y-%m-%d')
    end = date_obj.strftime('%Y-%m-%dT00:00:00')
    return start, end

def convert_datatypes(df, verbose=True):
    """
    Convert dataframe columns to appropriate data types for EDDK transformer data.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with columns: datastream_id, value, substation, parameter, timestamp
    verbose : bool, optional (default=True)
        If True, print conversion summary
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with converted data types
        
    Raises:
    -------
    ValueError
        If required columns are missing

    """
    # Check if required columns exist
    required_columns = ["datastream_id", "value", "substation", "parameter", "timestamp"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Convert data types
    try:
        df = df.astype({
            "datastream_id": "int",
            "value": "float",
            "substation": "string",
            "parameter": "string"
        })
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    except Exception as e:
        raise ValueError(f"Error converting data types: {e}")
    
    return df

def fetch_timespan_values(token: str, datastream_ids: List[int], start, end) -> List[dict]:
    """
    Fetch datastream values for a specific time period from Energy Data Service API.
    
    Args:
        token (str): API authentication token
        datastream_ids (List[int]): List of datastream IDs to fetch
        start (str): Start datetime in ISO format (e.g., "2022-08-01T00:00:00")
        end (str): End datetime in ISO format (e.g., "2022-08-02T00:00:00")
    
    Returns:
        List[dict]: List of records with keys: datastream_id, timestamp, value, 
                    substation, parameter. Returns empty list on error.
    
    Example:
        data = fetch_timespan_values(token, [900101, 900102], 
                                    "2022-08-01T00:00:00", "2022-08-02T00:00:00")
    """
    url = "https://admin.energydata.dk/api/v1/datastreams/values"
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {token}"
    }
    params = {
        "ids": ",".join(map(str, datastream_ids)),
        "from": start,
        "to": end
    }
    
    url = "https://admin.energydata.dk/api/v1/datastreams/values"
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {token}"
    }
    params = {
        "ids": ",".join(map(str, datastream_ids)),
        "from": start,
        "to": end
    }

    #print("Fetching data from URL:", url)                              # Debugging line, can be removed later
    #print("Headers:", headers)
    #print("Parameters:", params)

    try:
        response = requests.get(url, headers=headers, params=params, timeout=15)
        print("Status Code:", response.status_code)
        #print("Response Text (first 500 chars):", response.text[:500]) # Debugging line, can be removed later
        response.raise_for_status()

    except requests.exceptions.Timeout:
        print("Request timed out. The server took too long to respond.")
    except Exception as e:
        print("HTTP Request failed:", e)
        return []  # Ensures an empty list, not None

    metadata = get_datastream_metadata()
    result = []

    csv_file = StringIO(response.text)
    reader = csv.reader(csv_file)

    for row in reader:
        if len(row) == 3:
            datastream_id, timestamp, value = row
            try:
                datastream_id = int(datastream_id)
                value = float(value)
                timestamp = timestamp.strip()
            except ValueError as ve:
                print("Skipping malformed row:", row, "| Error:", ve)
                continue

            substation = metadata.get(datastream_id, {}).get("substation", "Unknown")
            parameter = metadata.get(datastream_id, {}).get("parameter", "Unknown")

            #print(f"Substation: {substation}, Parameter: {parameter}, Value: {value}, Timestamp: {timestamp}")
            result.append({
                "datastream_id": datastream_id,
                "timestamp": timestamp,
                "value": value,
                "substation": substation,
                "parameter": parameter
            })

    return result

def shift_timespan(data: List[dict], new_start: str) -> List[dict]:
    """
    Shift all timestamps in the data so they begin at new_start,
    preserving the original relative time intervals.

    Args:
        data (List[dict]): List of records with 'timestamp' field in ISO format.
        new_start (str): New start timestamp (e.g., '2023-01-01T00:00:00').

    Returns:
        List[dict]: Modified list with shifted timestamps.
    """
    if not data:
        return []

    # Parse original and new start times
    original_start = datetime.fromisoformat(data[0]['timestamp'])
    new_start_dt = datetime.fromisoformat(new_start)

    # Compute time delta
    delta = new_start_dt - original_start

    # Shift all timestamps
    for entry in data:
        original_ts = datetime.fromisoformat(entry['timestamp'])
        new_ts = original_ts + delta
        entry['timestamp'] = new_ts.isoformat()

    print(f"Shifted {len(data)} timestamps to start from {new_start}")
    return data


def store_data(df: pd.DataFrame, filename: str, path: str = "./Data/Stage") -> None:
    """
    Stores the DataFrame as a CSV file in the specified path.

    Args:
        df (pd.DataFrame): The data to store.
        filename (str): Name of the output CSV file (without extension).
        path (str): Directory to store the file. Defaults to './Data/Stage'.

    Returns:
        None
    """
    os.makedirs(path, exist_ok=True)
    full_path = os.path.join(path, f"{filename}.csv")

    try:
        df.to_csv(full_path, index=False)
        print(f"Data stored at: {full_path} ({len(df)} rows)")
    except Exception as e:
        print(f"Failed to store data to {full_path}:", e)


#############################################
### Transformer metadata functions
#############################################


def get_datastream_metadata_transformer_p_c():
    """Return the transformer P/C datastream_id -> substation/parameter map."""
    return dict(_datastream_map()["transformer_p_c"])


def fetch_timespan_values_transformer_p_c(token: str, datastream_ids: List[int], start, end) -> List[dict]:
    """
    Fetch datastream values for a specific time period from 
    Energy Data Service API transformer production and consumption dataset.
    
    Args:
        token (str): API authentication token
        datastream_ids (List[int]): List of datastream IDs to fetch
        start (str): Start datetime in ISO format (e.g., "2022-08-01T00:00:00")
        end (str): End datetime in ISO format (e.g., "2022-08-02T00:00:00")
    
    Returns:
        List[dict]: List of records with keys: datastream_id, timestamp, value, 
                    substation, parameter. Returns empty list on error.
    """

    url = "https://admin.energydata.dk/api/v1/datastreams/values"
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {token}"
    }
    params = {
        "ids": ",".join(map(str, datastream_ids)),
        "from": start,
        "to": end
    }

    #print("Fetching data from URL:", url)                              # Debugging line, can be removed later
    #print("Headers:", headers)
    #print("Parameters:", params)

    try:
        response = requests.get(url, headers=headers, params=params, timeout=15)
        print("Status Code:", response.status_code)
        #print("Response Text (first 500 chars):", response.text[:500]) # Debugging line, can be removed later
        response.raise_for_status()

    except requests.exceptions.Timeout:
        print("Request timed out. The server took too long to respond.")
    except Exception as e:
        print("HTTP Request failed:", e)
        return []  # Ensures an empty list, not None

    metadata = get_datastream_metadata_transformer_p_c()
    result = []

    csv_file = StringIO(response.text)
    reader = csv.reader(csv_file)

    for row in reader:
        if len(row) == 3:
            datastream_id, timestamp, value = row
            try:
                datastream_id = int(datastream_id)
                value = float(value)
                timestamp = timestamp.strip()
            except ValueError as ve:
                print("Skipping malformed row:", row, "| Error:", ve)
                continue

            substation = metadata.get(datastream_id, {}).get("substation", "Unknown")
            parameter = metadata.get(datastream_id, {}).get("parameter", "Unknown")

            #print(f"Substation: {substation}, Parameter: {parameter}, Value: {value}, Timestamp: {timestamp}")
            result.append({
                "datastream_id": datastream_id,
                "timestamp": timestamp,
                "value": value,
                "substation": substation,
                "parameter": parameter
            })

    return result

def fetch_latest_values_transformer_p_c(token: str, datastream_ids: List[int]) -> List[dict]:
    """
    Fetch the latest (most recent) values from transformer production 
    and consumption datastreams.
    
    Args:
        token (str): API authentication token
        datastream_ids (List[int]): List of datastream IDs to fetch latest values for
    
    Returns:
        List[dict]: List of records with keys: datastream_id, timestamp, value, 
                    substation, parameter. Returns empty list on error.
    """
    
    url = "https://admin.energydata.dk/api/v1/datastreams/values"
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {token}"
    }
    params = {
        "ids": ",".join(map(str, datastream_ids)),
        "latest": "true"
    }

    #print("Fetching data from URL:", url)                              # Debugging line, can be removed later
    #print("Headers:", headers)
    #print("Parameters:", params)

    try:
        response = requests.get(url, headers=headers, params=params, timeout=15)
        print("Status Code:", response.status_code)
        #print("Response Text (first 500 chars):", response.text[:500]) # Debugging line, can be removed later
        response.raise_for_status()

    except requests.exceptions.Timeout:
        print("Request timed out. The server took too long to respond.")
    except Exception as e:
        print("HTTP Request failed:", e)
        return []  # Ensures an empty list, not None

    metadata = get_datastream_metadata_transformer_p_c()
    result = []

    csv_file = StringIO(response.text)
    reader = csv.reader(csv_file)

    for row in reader:
        if len(row) == 3:
            datastream_id, timestamp, value = row
            try:
                datastream_id = int(datastream_id)
                value = float(value)
                timestamp = timestamp.strip()
            except ValueError as ve:
                print("Skipping malformed row:", row, "| Error:", ve)
                continue

            substation = metadata.get(datastream_id, {}).get("substation", "Unknown")
            parameter = metadata.get(datastream_id, {}).get("parameter", "Unknown")

            #print(f"Substation: {substation}, Parameter: {parameter}, Value: {value}, Timestamp: {timestamp}")
            result.append({
                "datastream_id": datastream_id,
                "timestamp": timestamp,
                "value": value,
                "substation": substation,
                "parameter": parameter
            })

    return result