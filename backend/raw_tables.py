"""
raw_tables.py — the network's own data tables, as the network stores them.

The baseline the semantic layer is measured against (`CONDUCTOR_ONTOLOGY=tables`,
evaluation only): instead of the layer, the agent is given pandapower's element
tables with no interpretation — a unit's `type` as the file writes it ("PV",
"CHP diesel"), a switch's `closed`, a bus's `vn_kv` — and must work out what
each element is itself.

Only pandapower's standard columns are returned (those of an empty network's
tables). The columns CONDUCTOR adds while loading data (substation names, which
measurement a value was split from, whether a reactive power came from a
measurement) are left out: they are part of what the layer reads, not of the
network file, and would hand the baseline some of the layer's facts.
"""

from __future__ import annotations

import functools

# The element tables an operator's model describes, in the order an engineer reads them.
TABLES = ("bus", "line", "trafo", "switch", "load", "sgen", "gen", "storage", "ext_grid")

# More rows than an answer can use: a network this large is not one the baseline is run on.
MAX_ROWS = 400


@functools.lru_cache(maxsize=None)
def _standard_columns(table: str) -> tuple[str, ...]:
    import pandapower as pp
    return tuple(c for c in pp.create_empty_network()[table].columns if c != "geo")


def _plain(value):
    """A JSON value: numpy scalars as Python ones, missing (NaN, None, pandas' NA) as None."""
    import pandas as pd
    if not isinstance(value, (list, tuple, dict)) and pd.isna(value):
        return None
    return value.item() if hasattr(value, "item") else value


def raw_tables(net, tables=None) -> dict:
    """{table: [row, …]} for the given tables (all of TABLES by default), each row its
    index and the table's standard columns. An unknown table raises; one longer than
    MAX_ROWS is cut, and says so."""
    wanted = list(tables or TABLES)
    unknown = [t for t in wanted if t not in TABLES]
    if unknown:
        raise ValueError(f"Unknown table(s) {unknown}; known: {list(TABLES)}")
    out: dict = {}
    for table in wanted:
        df = net[table]
        columns = [c for c in _standard_columns(table) if c in df.columns]
        rows = [{"index": int(idx), **{c: _plain(row[c]) for c in columns}}
                for idx, row in df.head(MAX_ROWS).iterrows()]
        out[table] = rows
        if len(df) > MAX_ROWS:
            out[f"{table}_truncated"] = f"first {MAX_ROWS} of {len(df)} rows"
    return out
