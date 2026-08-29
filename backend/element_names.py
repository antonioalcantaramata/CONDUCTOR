"""element_names.py — one way to name a bus, a line or a transformer.

Every endpoint that reports an element has to spell its name the same way,
because the callers join on that string. `/api/network/topology` describes the
shape of the system and `/api/grid/rsa` describes its operating point; the
system map joins one onto the other by element name, and a name that differs
by so much as a suffix renders as "not measured" rather than as an error.

That is exactly what happened to transformers. Lines already went through
`line_display_name`, so they matched. Transformers did not: seven call sites
had each grown their own fallback for an unnamed transformer — `Trafo_3-6`,
`Trafo_3-6_2`, `Trafo_2`, the bare pandapower index — so a network whose
transformers carry no name (every PGLib case) reported loading under one name
and topology under another, and the map drew all three IEEE 14-bus
transformers grey.

The fallbacks are built from bus names and the element index, never from the
index alone: indices shift when a network is modified, and two transformers
between the same pair of buses are ordinary. Both parts are needed for a name
that is readable and unique.
"""

from __future__ import annotations


def clean_label(raw) -> str:
    """Normalize an optional label, treating empty/nan/none as missing."""
    s = str(raw).strip() if raw is not None else ""
    return "" if s.lower() in {"", "nan", "none"} else s


def bus_display_name(net, bus_idx: int) -> str:
    """A stable display name for a bus, without changing any IDs."""
    if "name" in net.bus.columns:
        cleaned = clean_label(net.bus.at[bus_idx, "name"])
        if cleaned:
            return cleaned
    return f"Bus_{int(bus_idx)}"


def _branch_display_name(net, table, idx: int, from_col: str, to_col: str,
                         tag: str) -> str:
    """Shared shape: original name if present, then endpoints and index.

    The endpoints and index are appended even when the element has a name of
    its own, so the label stays unambiguous after a rename and so two elements
    that share a name stay distinguishable.
    """
    frame = getattr(net, table)
    endpoints = (f"{bus_display_name(net, int(frame.at[idx, from_col]))} -> "
                 f"{bus_display_name(net, int(frame.at[idx, to_col]))}")
    raw = clean_label(frame.at[idx, "name"]) if "name" in frame.columns else ""
    stamped = f"{endpoints} [{tag}{int(idx)}]"
    return f"{raw} | {stamped}" if raw else stamped


def line_display_name(net, line_idx: int) -> str:
    """Human-readable line label for API and chart display."""
    return _branch_display_name(net, "line", int(line_idx),
                                "from_bus", "to_bus", "L")


def trafo_display_name(net, trafo_idx: int) -> str:
    """Human-readable transformer label, in the same shape as a line's."""
    return _branch_display_name(net, "trafo", int(trafo_idx),
                                "hv_bus", "lv_bus", "T")
