"""
investigation.py — why a quantity is what it is now: its change since a
reference state, divided among the changes that caused it.

An operator's "why is Bus_29 low?" is a comparison: low now, and not low at
some moment they have in mind. This module answers it by measurement, not
narration.

1. The two states are one network at two operating points: two moments, the
   forecast and the measurements, or a proposal and the base. Every difference
   between them a power flow sees (each load, each unit's output and
   set-point, each element's state and parameters) is listed (`diff`).
2. The differences are grouped into at most six factors an operator would
   name: the demand near the element, the demand elsewhere, the units' output,
   a unit's set-point, a line taken out… (`group`). Beyond six, the factors
   with the smallest effect are put together as "other changes".
3. Every combination of factors is solved, each factor at its value now or at
   the reference: 2^k AC power flows, 64 at most. A factor's share is its
   Shapley value, its average effect over every order in which the changes
   could have happened. The shares add up exactly to the observed change,
   interactions included, and the two end combinations reproduce the two real
   states (checked: `closure`).
4. A factor explaining a large share is a cause; a small one contributes; one
   with no measurable effect is ruled out; one that pushed the other way
   counteracts. The same table gives two counterfactuals per factor at no
   cost: undoing only this change, and this change alone.

A factor that is a group (and a cause) is then divided among its members the
same way, with the other factors as they are now (`drill_down`).

What it does not say is the remedy: what changed is not necessarily what to
move now. The endpoint adds the levers from the sensitivity attribution.
"""

from __future__ import annotations

import copy
import math
import re
from math import factorial
from typing import Callable, NamedTuple

import networkx as nx
import numpy as np
import pandapower as pp
import pandapower.topology as top

import generators as gm
from element_names import bus_display_name, line_display_name, trafo_display_name

MAX_FACTORS = 6            # 2^6 = 64 power flows per decomposition
MAX_HYPOTHESES = MAX_FACTORS - 1   # the caller's own factors; the sixth is "other changes"
NEAR_HOPS = 2              # "near" an element: within two buses of it
CAUSE_SHARE = 0.15         # a share at or above this is a cause
RULED_OUT_SHARE = 0.03     # below this (either way) it is ruled out
DRILL_SHARE = 0.25         # a group explaining this much is divided among its members
MAX_DRILLS = 2
MAX_UNITS_APART = 2        # more units than this changing output: one factor for all of them
MEMBERS_LISTED = 12

INTERACTION_SHARE = 0.10   # a pair whose interaction is this much of the change is reported
LIMIT_TOL = 1e-6           # the security assessment's own tolerance (rsa_engine.FLOAT_TOL)


class Target(NamedTuple):
    """What is divided among the causes: a quantity of one solved network.

    `metric(net)` reads it from a solved network (None if it cannot);
    `within(value)` says whether a value is acceptable; below `min_change`
    the two states give the same value; the decomposition's end states must
    reproduce the real ones within `closure_tol`.
    """
    quantity: str
    unit: str
    metric: Callable
    within: Callable
    label: str
    buses: frozenset
    decimals: int
    min_change: float
    closure_tol: float
    # How far a value lies outside its limits (0 inside them): a window's exceedance.
    excess: Callable | None = None
    # The element measured, for the classical views (None for the grid).
    etype: str | None = None
    idx: int | None = None


def element_target(etype: str, idx: int, label: str, buses: set[int], limits: dict) -> Target:
    """A bus voltage (p.u.) or a branch's loading (% of its rating)."""
    if etype == "bus":
        lo, hi = limits["vm_lower_pu"], limits["vm_upper_pu"]
        return Target("vm_pu", "p.u.", lambda net: measure(net, etype, idx),
                      lambda v: within(v, etype, limits), label, frozenset(buses), 5, 1e-4, 1e-5,
                      lambda v: max(0.0, lo - v) + max(0.0, v - hi), etype, idx)
    cap = limits["max_line_loading_pct" if etype == "line" else "max_trafo_loading_pct"]
    return Target("loading_percent", "% of rating", lambda net: measure(net, etype, idx),
                  lambda v: within(v, etype, limits), label, frozenset(buses), 2, 0.05, 0.01,
                  lambda v: max(0.0, v - cap), etype, idx)


def count_violations(net, limits: dict) -> int:
    """Elements of a solved network outside their limits, as the assessment
    counts them (buses, lines, transformers in service)."""
    n = 0
    for idx in net.bus.index[net.bus["in_service"].astype(bool)]:
        v = measure(net, "bus", int(idx))
        n += v is not None and not within(v, "bus", limits)
    for etype in ("line", "trafo"):
        tbl = net[etype]
        for idx in tbl.index[tbl["in_service"].astype(bool)]:
            v = measure(net, etype, int(idx))
            n += v is not None and not within(v, etype, limits)
    return n


def violations_target(limits: dict) -> Target:
    """The number of elements outside their limits: "why is the grid insecure"."""
    return Target("violations", "elements outside their limits", lambda net: float(count_violations(net, limits)),
                  lambda v: v < 0.5, "the grid", frozenset(), 2, 0.5, 1e-9)

# What a power flow reads from each table. A column outside these (an OPF
# limit, a cost, a name) changes nothing the decomposition could measure.
PF_COLUMNS = {
    "bus": ("in_service",),
    "load": ("p_mw", "q_mvar", "scaling", "in_service", "bus"),
    "sgen": ("p_mw", "q_mvar", "scaling", "in_service", "bus"),
    "gen": ("p_mw", "vm_pu", "scaling", "in_service", "bus"),
    "ext_grid": ("vm_pu", "va_degree", "in_service"),
    "shunt": ("p_mw", "q_mvar", "step", "in_service", "bus"),
    "line": ("in_service", "r_ohm_per_km", "x_ohm_per_km", "c_nf_per_km", "g_us_per_km", "length_km",
             "parallel", "max_i_ka", "df", "from_bus", "to_bus"),
    "trafo": ("in_service", "sn_mva", "vn_hv_kv", "vn_lv_kv", "vk_percent", "vkr_percent", "pfe_kw",
              "i0_percent", "shift_degree", "tap_pos", "parallel", "df", "hv_bus", "lv_bus"),
}


class InvestigationError(ValueError):
    """The two states cannot be compared; the message says why."""


class Change(NamedTuple):
    table: str
    idx: int
    ref: dict      # column -> value in the reference state
    cur: dict      # column -> value now


class Member(NamedTuple):
    label: str
    changes: tuple[int, ...]       # indices into the change list


class Factor(NamedTuple):
    label: str
    kind: str      # demand | units | unit | renewables | setpoint | topology | added | other
    members: tuple[Member, ...]

    @property
    def changes(self) -> frozenset[int]:
        return frozenset(i for m in self.members for i in m.changes)


# ---------------------------------------------------------------------------
# The two states
# ---------------------------------------------------------------------------

def _plain(value):
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def _same(a, b) -> bool:
    if isinstance(a, (bool, np.bool_)) or isinstance(b, (bool, np.bool_)):
        return bool(a) == bool(b)
    try:
        fa, fb = float(a), float(b)
    except (TypeError, ValueError):
        return a == b
    if math.isnan(fa) or math.isnan(fb):
        return math.isnan(fa) and math.isnan(fb)
    return abs(fa - fb) <= 1e-9 * max(1.0, abs(fa), abs(fb))


def _in_service(tbl, idx) -> bool:
    return bool(tbl.at[idx, "in_service"]) if "in_service" in tbl.columns else True


def diff(ref_net, cur_net) -> tuple[str, list[Change]]:
    """Every difference a power flow sees between the two states.

    Returns (template, changes): `template` is the state whose copy is
    modified to build every combination ("current" or "reference") — the one
    holding every element, since an element only one state has (a proposal's
    addition) is out of service in the other.
    """
    changes: list[Change] = []
    only = {"current": [], "reference": []}
    for table, wanted in PF_COLUMNS.items():
        if table not in ref_net or table not in cur_net:
            continue
        r, c = ref_net[table], cur_net[table]
        cols = [k for k in wanted if k in r.columns and k in c.columns]
        for idx in r.index.intersection(c.index):
            ref, cur = {}, {}
            for k in cols:
                a, b = r.at[idx, k], c.at[idx, k]
                if not _same(a, b):
                    ref[k], cur[k] = _plain(a), _plain(b)
            if ref:
                changes.append(Change(table, int(idx), ref, cur))
        for idx in c.index.difference(r.index):
            if _in_service(c, idx):
                only["current"].append(Change(table, int(idx), {"in_service": False}, {"in_service": True}))
        for idx in r.index.difference(c.index):
            if _in_service(r, idx):
                only["reference"].append(Change(table, int(idx), {"in_service": True}, {"in_service": False}))
    if only["current"] and only["reference"]:
        raise InvestigationError(
            "Each state has elements the other lacks, so neither can stand for both. Compare each "
            "proposal with the base instead.")
    return ("reference" if only["reference"] else "current"), changes + only["current"] + only["reference"]


def _solve(net) -> bool:
    for init in ("auto", "dc", "flat"):
        try:
            pp.runpp(net, init=init)
            return True
        except Exception:  # noqa: BLE001 — a state that does not solve is reported, not raised
            continue
    return False


def measure(net, etype: str, idx: int) -> float | None:
    """The investigated quantity in a solved network: a bus voltage (p.u.) or
    a branch's loading (% of its rating)."""
    if etype == "bus":
        value = net.res_bus.at[idx, "vm_pu"]
    else:
        value = net[f"res_{etype}"].at[idx, "loading_percent"]
    value = float(value) if value is not None else float("nan")
    return value if math.isfinite(value) else None


class Combinations:
    """The network with each change at its value now or at the reference, solved."""

    def __init__(self, ref_net, cur_net, template: str, changes: list[Change], metric: Callable):
        self.net = copy.deepcopy(cur_net if template == "current" else ref_net)
        self.changes = changes
        self.metric = metric
        self.memo: dict[frozenset, float | None] = {}
        self.power_flows = 0

    def value(self, now: frozenset[int]) -> float | None:
        """The quantity with the changes in `now` as they are now and every
        other change as it was in the reference; None if it does not solve."""
        if now in self.memo:
            return self.memo[now]
        for i, ch in enumerate(self.changes):
            tbl = self.net[ch.table]
            for k, v in (ch.cur if i in now else ch.ref).items():
                tbl.at[ch.idx, k] = v
        self.power_flows += 1
        out = self.metric(self.net) if _solve(self.net) else None
        self.memo[now] = out
        return out


# ---------------------------------------------------------------------------
# Factors
# ---------------------------------------------------------------------------

def _bus_of(ref_net, cur_net, ch: Change) -> int | None:
    for net in (cur_net, ref_net):
        tbl = net[ch.table] if ch.table in net else None
        if tbl is not None and ch.idx in tbl.index and "bus" in tbl.columns:
            return int(tbl.at[ch.idx, "bus"])
    return None


def _branch_name(net, table: str, idx: int) -> str:
    return line_display_name(net, idx) if table == "line" else trafo_display_name(net, idx)


def _element_label(ref_net, cur_net, ch: Change) -> str:
    net = cur_net if ch.table in cur_net and ch.idx in cur_net[ch.table].index else ref_net
    if ch.table == "bus":
        return bus_display_name(net, ch.idx)
    if ch.table in ("line", "trafo"):
        return _branch_name(net, ch.table, ch.idx)
    if ch.table in ("gen", "sgen"):
        return gm.unit_name(net, ch.table, ch.idx)
    if ch.table == "ext_grid":
        return "the external grid"
    bus = _bus_of(ref_net, cur_net, ch)
    where = f" at {bus_display_name(net, bus)}" if bus is not None else ""
    raw = net[ch.table].at[ch.idx, "name"] if "name" in net[ch.table].columns else None
    name = str(raw).strip() if isinstance(raw, str) and raw.strip() else ch.table.capitalize()
    return f"{name}{where}"


def _near_buses(net, element_buses: set[int]) -> set[int]:
    try:
        graph = top.create_nxgraph(net, include_out_of_service=False)
        reach = nx.multi_source_dijkstra_path_length(graph, element_buses, cutoff=NEAR_HOPS,
                                                     weight=lambda u, v, d: 1)
        return set(int(b) for b in reach)
    except Exception:  # noqa: BLE001 — without a graph, "near" is the element's own buses
        return set(element_buses)


def _dispatchable(net, ch: Change) -> bool:
    if ch.table == "gen":
        return True
    tbl = net[ch.table]
    return ("controllable" in tbl.columns and ch.idx in tbl.index
            and bool(tbl.at[ch.idx, "controllable"]) is True)


def _joined(members: list[Member], many: str) -> str:
    """A group's name: its only member's when it has one."""
    return members[0].label if len(members) == 1 else many


def group(ref_net, cur_net, template: str, changes: list[Change], element_label: str,
          element_buses: set[int]) -> list[Factor]:
    """The changes as the factors an operator would name (see the module docstring)."""
    template_net = cur_net if template == "current" else ref_net
    near = _near_buses(cur_net, element_buses)
    added = {i for i, ch in enumerate(changes)
             if ch.idx not in (ref_net[ch.table].index if template == "current" else cur_net[ch.table].index)}

    by_bus: dict[str, dict[int, list[int]]] = {"near": {}, "far": {}}
    units: dict[str, list[int]] = {}
    renewables: dict[str, list[int]] = {}
    alone: list[Factor] = []
    added_members: list[Member] = []

    for i, ch in enumerate(changes):
        label = _element_label(ref_net, cur_net, ch)
        if i in added:
            added_members.append(Member(f"{label} (added)" if template == "current" else f"{label} (removed)",
                                        (i,)))
            continue
        if ch.table == "load":
            bus = _bus_of(ref_net, cur_net, ch)
            side = "near" if bus in near else "far"
            by_bus[side].setdefault(bus, []).append(i)
            continue
        if ch.table in ("gen", "sgen"):
            output = {k for k in ch.cur if k in ("p_mw", "q_mvar", "scaling")}
            other = set(ch.cur) - output
            if "in_service" in other:
                state = "back in service" if ch.cur["in_service"] else "out of service"
                alone.append(Factor(f"{label} {state}", "unit", (Member(label, (i,)),)))
            elif "vm_pu" in other:
                alone.append(Factor(f"Voltage set-point of {label} ({ch.ref['vm_pu']:.3f} → "
                                    f"{ch.cur['vm_pu']:.3f} p.u.)", "setpoint", (Member(label, (i,)),)))
            elif other:
                alone.append(Factor(f"{label} moved", "topology", (Member(label, (i,)),)))
            elif _dispatchable(template_net, ch):
                units.setdefault(label, []).append(i)
            else:
                renewables.setdefault(label, []).append(i)
            continue
        if ch.table == "ext_grid":
            if "in_service" in ch.cur:
                text = "The external grid " + ("connected" if ch.cur["in_service"] else "disconnected")
                alone.append(Factor(text, "topology", (Member("the external grid", (i,)),)))
            else:
                parts = [f"{ch.ref[k]:.3f} → {ch.cur[k]:.3f} {'p.u.' if k == 'vm_pu' else '°'}"
                         for k in ("vm_pu", "va_degree") if k in ch.cur]
                alone.append(Factor(f"External grid voltage set-point ({', '.join(parts)})", "setpoint",
                                    (Member("the external grid", (i,)),)))
            continue
        # Branches, buses and shunts: each its own factor.
        if set(ch.cur) == {"in_service"}:
            text = f"{label} {'back in service' if ch.cur['in_service'] else 'out of service'}"
        else:
            shown = ", ".join(f"{k} {_short(ch.ref[k])} → {_short(ch.cur[k])}" for k in list(ch.cur)[:3])
            text = f"{label} changed ({shown})"
        alone.append(Factor(text, "topology", (Member(label, (i,)),)))

    factors: list[Factor] = []
    for side in ("near", "far"):
        members = [Member(f"Demand at {bus_display_name(template_net, bus)}" if bus is not None else "Demand",
                          tuple(ix)) for bus, ix in sorted(by_bus[side].items(), key=lambda kv: kv[0] or -1)]
        if not members:
            continue
        if not by_bus["near"] or not by_bus["far"]:
            many = f"Demand ({len(members)} buses)"
        elif side == "near":
            many = f"Demand near {element_label} ({len(members)} buses)"
        else:
            many = f"Demand elsewhere ({len(members)} buses)"
        factors.append(Factor(_joined(members, many), "demand", tuple(members)))
    unit_members = [Member(f"Output of {name}", tuple(ix)) for name, ix in units.items()]
    if len(unit_members) > MAX_UNITS_APART:
        factors.append(Factor(f"Units' output ({len(unit_members)} units)", "units", tuple(unit_members)))
    else:
        factors += [Factor(m.label, "unit", (m,)) for m in unit_members]
    if renewables:
        members = [Member(f"Output of {name}", tuple(ix)) for name, ix in renewables.items()]
        factors.append(Factor(_joined(members, f"Non-dispatchable generation ({len(members)} units)"),
                              "renewables", tuple(members)))
    factors += alone
    if added_members:
        what = "added" if template == "current" else "removed"
        factors.append(Factor(_joined(added_members, f"Elements {what} ({len(added_members)})"),
                              "added", tuple(added_members)))
    return factors


# Words naming every change of a kind, in a caller's hypothesis.
_ALL_DEMAND = {"demand", "all demand", "load", "loads", "all loads"}
_ALL_UNITS = {"units", "all units", "generation", "generators", "units' output", "unit output"}
_GRID = {"external grid", "the external grid", "ext_grid", "slack", "grid"}


def _names_of(ref_net, cur_net, ch: Change) -> set[str]:
    """Every name a caller may use for the element a change belongs to,
    case-folded: a bus name covers the demand and shunts at it; a unit, its
    name; a branch, its display name and its tag (L38, T1)."""
    net = cur_net if ch.table in cur_net and ch.idx in cur_net[ch.table].index else ref_net
    names = {_element_label(ref_net, cur_net, ch)}
    if ch.table in ("load", "shunt", "bus"):
        bus = ch.idx if ch.table == "bus" else _bus_of(ref_net, cur_net, ch)
        if bus is not None:
            label = bus_display_name(net, bus)
            names |= {label, f"Demand at {label}", str(bus)}
    if ch.table in ("line", "trafo"):
        label = _branch_name(net, ch.table, ch.idx)
        names.add(label)
        tag = label.rsplit("[", 1)[-1].rstrip("]") if label.endswith("]") else None
        if tag:
            names.add(tag)
    if ch.table == "ext_grid":
        names |= _GRID
    return {n.strip().casefold() for n in names if n}


def _named(name: str, ch: Change, names: set[str]) -> bool:
    key = name.strip().casefold()
    if key in names:
        return True
    if key in _ALL_DEMAND:
        return ch.table == "load"
    if key in _ALL_UNITS:
        return ch.table in ("gen", "sgen")
    return False


def group_hypotheses(ref_net, cur_net, template: str, changes: list[Change],
                     hypotheses: list[dict]) -> tuple[list[Factor], list[dict]]:
    """The caller's hypotheses as factors, each holding the changes of the
    elements it names; every other change is one more factor, "other changes".

    Returns (factors, notes): a note per hypothesis that named something that
    did not change between the two states (or nothing that did), so the result
    can say so instead of silently dropping it. A change two hypotheses name
    goes to the first.
    """
    names = [_names_of(ref_net, cur_net, ch) for ch in changes]
    taken: set[int] = set()
    factors, notes = [], []
    for h in hypotheses:
        label = str(h.get("name") or "").strip() or ", ".join(h.get("elements") or [])
        mine, unmatched = [], []
        for element in h.get("elements") or []:
            hits = [i for i, ch in enumerate(changes) if _named(element, ch, names[i])]
            if not hits:
                unmatched.append(element)
            mine += [i for i in hits if i not in taken and i not in mine]
        if not mine:
            notes.append({"hypothesis": label, "unchanged": unmatched, "empty": True,
                          "note": ("nothing it names changed between the two states" if unmatched else
                                   "everything it names is already in an earlier hypothesis")})
            continue
        if unmatched:
            notes.append({"hypothesis": label, "unchanged": unmatched,
                          "note": "part of what it names did not change between the two states"})
        taken.update(mine)
        factors.append(Factor(label, "hypothesis", _members(ref_net, cur_net, changes, mine)))
    rest = [i for i in range(len(changes)) if i not in taken]
    if rest:
        members = _members(ref_net, cur_net, changes, rest)
        factors.append(Factor(f"Other changes ({len(members)})", "other", members))
    return factors, notes


def _members(ref_net, cur_net, changes: list[Change], indices: list[int]) -> tuple[Member, ...]:
    """One member per element: the demand at a bus, a unit, a branch."""
    by_label: dict[str, list[int]] = {}
    for i in indices:
        ch = changes[i]
        if ch.table == "load":
            bus = _bus_of(ref_net, cur_net, ch)
            net = cur_net if ch.idx in cur_net.load.index else ref_net
            label = f"Demand at {bus_display_name(net, bus)}" if bus is not None else "Demand"
        elif ch.table in ("gen", "sgen") and set(ch.cur) <= {"p_mw", "q_mvar", "scaling"}:
            label = f"Output of {_element_label(ref_net, cur_net, ch)}"
        else:
            label = _element_label(ref_net, cur_net, ch)
        by_label.setdefault(label, []).append(i)
    return tuple(Member(label, tuple(ix)) for label, ix in by_label.items())


def _short(value) -> str:
    if isinstance(value, bool):
        return "on" if value else "off"
    try:
        return f"{float(value):.4g}"
    except (TypeError, ValueError):
        return str(value)


# ---------------------------------------------------------------------------
# Shapley values
# ---------------------------------------------------------------------------

def shapley(k: int, v: Callable[[int], float]) -> list[float]:
    """Exact Shapley values of k players for a value function on bit masks.

    v(mask) is the quantity with the players in `mask` as they are now. The
    values add up to v(all) − v(none).
    """
    weight = [factorial(s) * factorial(k - s - 1) / factorial(k) for s in range(k)]
    phi = [0.0] * k
    for mask in range(1 << k):
        base = v(mask)
        size = bin(mask).count("1")
        for i in range(k):
            if not mask >> i & 1:
                phi[i] += weight[size] * (v(mask | 1 << i) - base)
    return phi


def interactions(labels: list[str], v: Callable[[int], float], change: float, nd: int) -> list[dict]:
    """Pairs of factors whose joint effect differs from the sum of their own
    (the Shapley interaction index), when it is at least INTERACTION_SHARE of
    the change: "amplify" when together they push further in the direction of
    the change, "dampen" when they hold each other back.

        I_ij = Σ_{S ⊆ N∖{i,j}} |S|! (k−|S|−2)! / (k−1)! · [v(S∪ij) − v(S∪i) − v(S∪j) + v(S)]
    """
    k = len(labels)
    if k < 2 or abs(change) < 1e-12:
        return []
    weight = [factorial(s) * factorial(k - s - 2) / factorial(k - 1) for s in range(k - 1)]
    out = []
    for i in range(k):
        for j in range(i + 1, k):
            total = 0.0
            for mask in range(1 << k):
                if mask >> i & 1 or mask >> j & 1:
                    continue
                size = bin(mask).count("1")
                total += weight[size] * (v(mask | 1 << i | 1 << j) - v(mask | 1 << i) - v(mask | 1 << j) + v(mask))
            share = total / change
            if abs(share) >= INTERACTION_SHARE:
                out.append({"factors": [labels[i], labels[j]], "effect": round(total, nd + 1),
                            "share_pct": round(100 * share, 1), "kind": "amplify" if share > 0 else "dampen"})
    return sorted(out, key=lambda r: -abs(r["share_pct"]))


def verdict(share: float) -> str:
    if share >= CAUSE_SHARE:
        return "cause"
    if share >= RULED_OUT_SHARE:
        return "contributes"
    if share > -RULED_OUT_SHARE:
        return "ruled_out"
    return "counteracts"


class Unsolved(Exception):
    """A combination of the two states has no power-flow solution."""


def _valuer(combos: Combinations, sets: list[frozenset[int]], fixed: frozenset[int] = frozenset()):
    def v(mask: int) -> float:
        now = fixed.union(*(s for i, s in enumerate(sets) if mask >> i & 1))
        value = combos.value(now)
        if value is None:
            raise Unsolved()
        return value
    return v


def _prune(items: list, sets: list[frozenset[int]], combos: Combinations, fixed: frozenset[int],
           merge: Callable[[list], object]) -> tuple[list, list[frozenset[int]]]:
    """At most MAX_FACTORS items: the ones with the largest effect, by the
    larger of adding each alone and removing it alone, and the rest merged."""
    if len(items) <= MAX_FACTORS:
        return items, sets
    full = fixed.union(*sets)
    v_none, v_all = combos.value(fixed), combos.value(full)
    if v_none is None or v_all is None:
        raise Unsolved()
    weight = []
    for s in sets:
        alone, without = combos.value(fixed | s), combos.value(full - s)
        if alone is None or without is None:
            raise Unsolved()
        weight.append(max(abs(alone - v_none), abs(v_all - without)))
    order = sorted(range(len(items)), key=lambda i: -weight[i])
    keep, rest = order[:MAX_FACTORS - 1], order[MAX_FACTORS - 1:]
    kept = [items[i] for i in keep] + [merge([items[i] for i in rest])]
    return kept, [sets[i] for i in keep] + [frozenset().union(*(sets[i] for i in rest))]


def within(value: float, etype: str, limits: dict) -> bool:
    if etype == "bus":
        return limits["vm_lower_pu"] - LIMIT_TOL <= value <= limits["vm_upper_pu"] + LIMIT_TOL
    key = "max_line_loading_pct" if etype == "line" else "max_trafo_loading_pct"
    return value <= limits[key] + LIMIT_TOL


def _member_mw(ref_net, cur_net, changes: list[Change], idx) -> dict | None:
    """The active power of demand or output that changed, in both states and
    the difference: {"reference", "now", "change"} in MW; None for anything else."""
    rows = [changes[i] for i in idx]
    if not rows or not all(ch.table in ("load", "sgen", "gen") and set(ch.cur) <= {"p_mw", "q_mvar", "scaling"}
                           for ch in rows):
        return None

    def total(net):
        out = 0.0
        for ch in rows:
            tbl = net[ch.table]
            if ch.idx in tbl.index and _in_service(tbl, ch.idx):
                scale = float(tbl.at[ch.idx, "scaling"]) if "scaling" in tbl.columns else 1.0
                out += float(tbl.at[ch.idx, "p_mw"]) * scale
        return out
    before, after = total(ref_net), total(cur_net)
    return {"reference": round(before, 2), "now": round(after, 2), "change": round(after - before, 2)}


def _member_detail(ref_net, cur_net, changes: list[Change], idx: tuple[int, ...]) -> str:
    """What changed, in an operator's units: MW for demand and output."""
    mw = _member_mw(ref_net, cur_net, changes, idx)
    return f"{mw['reference']:.1f} → {mw['now']:.1f} MW" if mw else ""


CLASSICAL = ("linear", "tracing")
# Below this share explained, the caller's hypotheses leave too much unnamed.
COVERAGE_TARGET = 75.0
# A cause by kind (demand, units' output, a set-point…), the level at which
# conclusions are compared across references and events.
_KINDS = {"demand": "demand", "units": "units' output", "unit": "units' output", "renewables": "renewables",
          "setpoint": "a set-point", "added": "elements added"}


def cause_kind(row: dict | None) -> str | None:
    """A factor's kind of cause, or its own name for one that has no kind."""
    if not row:
        return None
    kind = _KINDS.get(row.get("kind"))
    return kind or re.sub(r"\s*\([^()]*\)\s*$", "", str(row.get("factor") or "")).strip() or None


def decompose(ref_net, cur_net, target: Target, hypotheses: list[dict] | None = None,
              classical: list[str] | None = None, drill: bool = True,
              inspect: Callable[[str, int], dict | None] | None = None) -> dict:
    """Divide the change of one quantity between the two states among its causes.

    Both networks must be prepared (values assigned) but need not be solved.
    Returns the factors with their shares and verdicts, the drill-down into the
    groups that are causes, and the checks; never raises on a state that does
    not solve (status "not_decomposed").

    `hypotheses` ([{"name", "elements"}], at most MAX_HYPOTHESES) replaces the
    standard grouping with the caller's own (`group_hypotheses`): the shares and
    verdicts are measured the same way, whoever chose what to test.

    `inspect(table, index)` judges whether a measured change is plausible
    (plausibility.py); it is asked about the injections behind every factor
    that is a cause, and its doubts are reported with the factor (`data_check`).

    `classical` adds the classical views the caller asks for (classical.py):
    "linear" — each factor estimated from sensitivities (LODF for a branch
    out), beside its exact share; "tracing" — who feeds and who draws the
    investigated branch's flow, in both states. They inform; they decide nothing.
    """
    quantity, nd = target.quantity, target.decimals
    template, changes = diff(ref_net, cur_net)
    out = {"quantity": quantity, "unit": target.unit, "n_differences": len(changes)}
    combos = Combinations(ref_net, cur_net, template, changes, target.metric)

    # The real states, solved on their own: what the combinations must reproduce.
    real, solved = {}, {}
    for side, net in (("reference", ref_net), ("current", cur_net)):
        solo = copy.deepcopy(net)
        real[side] = target.metric(solo) if _solve(solo) else None
        solved[side] = solo
        combos.power_flows += 1
    if real["current"] is None or real["reference"] is None:
        side = "now" if real["current"] is None else "in the reference state"
        return {**out, "status": "not_decomposed", "power_flows": combos.power_flows,
                "reason": f"The power flow does not solve {side}."}
    change = real["current"] - real["reference"]
    out.update(value=real["current"], reference_value=real["reference"], change=change)
    if abs(change) < target.min_change or not changes:
        return {**out, "status": "no_change", "power_flows": combos.power_flows,
                "reason": "The quantity is the same in both states: there is no change to divide."}

    notes: list[dict] = []
    if hypotheses:
        factors, notes = group_hypotheses(ref_net, cur_net, template, changes, hypotheses)
    else:
        factors = group(ref_net, cur_net, template, changes, target.label, set(target.buses))
    sets = [f.changes for f in factors]
    try:
        factors, sets = _prune(factors, sets, combos, frozenset(),
                               lambda rest: Factor(f"Other changes ({len(rest)})", "other",
                                                   tuple(m for f in rest for m in f.members)))
        k = len(factors)
        v = _valuer(combos, sets)
        v_none, v_all = v(0), v((1 << k) - 1)
        phi = shapley(k, v)
    except Unsolved:
        return {**out, "status": "not_decomposed", "power_flows": combos.power_flows,
                "reason": "Some mixes of the two states have no power-flow solution, so the change "
                          "cannot be divided among its causes."}

    gap = max(abs(v_all - real["current"]), abs(v_none - real["reference"]))
    closure = {"reproduces_both_states": gap <= target.closure_tol, "largest_gap": round(gap, 7)}
    if not closure["reproduces_both_states"]:
        return {**out, "status": "not_decomposed", "closure": closure, "power_flows": combos.power_flows,
                "reason": "The two states differ in something the decomposition cannot separate "
                          "(it does not reproduce both of them)."}

    full = (1 << k) - 1
    rows = []
    for i, f in enumerate(factors):
        share = phi[i] / change
        without = v(full & ~(1 << i))
        rows.append({
            "factor": f.label, "kind": f.kind,
            "members": [m.label for m in f.members][:MEMBERS_LISTED], "n_members": len(f.members),
            "detail": _member_detail(ref_net, cur_net, changes, tuple(sorted(f.changes))),
            **({"mw": mw} if (mw := _member_mw(ref_net, cur_net, changes, tuple(sorted(f.changes)))) else {}),
            "effect": round(phi[i], nd + 1),
            "share_pct": round(100 * share, 1),
            "verdict": verdict(share),
            "alone_value": round(v(1 << i), nd),
            "undo_alone_value": round(without, nd),
            "undo_alone_within_limits": target.within(without),
            "_i": i,
        })
    solo_sum = sum(v(1 << i) - v_none for i in range(k))
    rows.sort(key=lambda r: -r["share_pct"])

    drills = []
    for row in rows:
        f = factors[row["_i"]]
        if not drill or len(drills) >= MAX_DRILLS or row["share_pct"] < 100 * DRILL_SHARE or len(f.members) < 2:
            continue
        try:
            drills.append(_drill(combos, f, frozenset().union(*sets) - f.changes, nd, ref_net, cur_net,
                                 changes))
        except Unsolved:
            continue
    if inspect is not None:
        for row in rows:
            if row["verdict"] == "cause":
                row["data_check"] = _data_check(factors[row["_i"]], changes, inspect)
    for row in rows:
        row.pop("_i")

    # A hypothesis naming nothing that changed explains none of the change.
    for note in notes:
        if note.get("empty"):
            rows.append({"factor": note["hypothesis"], "kind": "hypothesis", "members": [], "n_members": 0,
                         "detail": "", "effect": 0.0, "share_pct": 0.0, "verdict": "ruled_out",
                         "nothing_changed": True, "unchanged": note["unchanged"],
                         "alone_value": round(v_none, nd),
                         "undo_alone_value": round(v_all, nd),
                         "undo_alone_within_limits": target.within(v_all)})
    causes = [r for r in rows if r["verdict"] == "cause"]
    extra = {"hypotheses_notes": [{k: v for k, v in n.items() if k != "empty"} for n in notes]} if notes else {}
    if hypotheses:
        # How much of the change the caller's own hypotheses explain: the rest
        # is in "other changes", which nobody named.
        covered = sum(r["share_pct"] for r in rows if r["kind"] == "hypothesis")
        extra["hypotheses_coverage_pct"] = round(covered, 1)
        if covered < COVERAGE_TARGET:
            extra["coverage_note"] = (
                f"Your hypotheses explain {covered:.0f} % of the change; the rest is in changes you did not name. "
                "Split or add hypotheses until they explain most of it, or say in the answer what remains "
                "unexplained.")
    if classical:
        extra["classical"] = _classical(classical, ref_net, cur_net, template, changes, factors, sets, target,
                                        v_none, v_all, {r["factor"]: r["share_pct"] for r in rows}, change, solved)
    return {
        **out, **extra, "status": "explained", "template": template,
        "grouping": "hypotheses" if hypotheses else "standard",
        "factors": rows,
        "main_cause": causes[0]["factor"] if causes else None,
        "drill_down": drills,
        "interaction_pct": round(100 * (change - solo_sum) / change, 1),
        "interactions": interactions([f.label for f in factors], v, change, nd),
        "closure": closure,
        "power_flows": combos.power_flows,
    }


def _data_check(factor: Factor, changes: list[Change], inspect) -> dict:
    """Whether the measured changes behind a cause look physical: each member's
    injections (a load's or a unit's active power) judged against their history."""
    checked, doubts = 0, []
    for m in factor.members:
        flags = []
        for i in m.changes:
            ch = changes[i]
            if ch.table not in ("load", "sgen", "gen") or "p_mw" not in ch.cur:
                continue
            found = inspect(ch.table, ch.idx)
            if found is None:
                continue
            checked += 1
            flags += found.get("flags") or []
        if flags:
            doubts.append({"member": m.label, "flags": flags})
    return {"checked": checked, "suspicious": doubts}


def _classical(asked, ref_net, cur_net, template, changes, factors, sets, target, v_none, v_all, exact, change,
               solved) -> dict:
    """The classical views asked for (classical.py), with a reason for any
    that does not apply here."""
    import classical as cl

    out: dict = {}
    if "linear" in asked:
        if target.quantity == "violations":
            out["linear"] = {"not_applicable": "a count of violations has no sensitivity: it jumps at the limit"}
        else:
            out["linear"] = cl.linear_estimates(ref_net, cur_net, template, changes, factors, sets, target,
                                                v_none, v_all, exact, change)
    if "tracing" in asked:
        if target.etype not in ("line", "trafo"):
            out["tracing"] = {"not_applicable": "tracing follows active power along branches; it says nothing "
                                                "about a bus voltage or a count of violations"}
        else:
            out["tracing"] = {side: cl.trace_branch(solved[side], target.etype, target.idx)
                              for side in ("current", "reference")}
    out["note"] = ("Classical views, beside the exact shares: they inform, the verdicts are the exact ones.")
    return out


def _drill(combos: Combinations, factor: Factor, others: frozenset[int], nd: int,
           ref_net, cur_net, changes: list[Change]) -> dict:
    """One group's effect divided among its members, the other factors as now."""
    members = list(factor.members)
    sets = [frozenset(m.changes) for m in members]
    members, sets = _prune(members, sets, combos, others,
                           lambda rest: Member(f"The rest ({len(rest)})", tuple(i for m in rest for i in m.changes)))
    k = len(members)
    psi = shapley(k, _valuer(combos, sets, fixed=others))
    total = sum(psi)
    # Members with no measurable effect are counted, not listed at 0 %.
    idle = [j for j in range(k) if abs(total) > 1e-12 and abs(psi[j] / total) < 0.005
            and not members[j].label.startswith("The rest (")]
    keep = [j for j in range(k) if j not in idle]
    members, psi = [members[j] for j in keep], [psi[j] for j in keep]
    rows = [{"member": m.label, "detail": _member_detail(ref_net, cur_net, changes, m.changes),
             **({"mw": mw} if (mw := _member_mw(ref_net, cur_net, changes, m.changes)) else {}),
             "effect": round(psi[j], nd + 1),
             "share_of_factor_pct": round(100 * psi[j] / total, 1) if abs(total) > 1e-12 else None}
            for j, m in enumerate(members)]
    # Largest first; the merged rest, if any, last.
    rows.sort(key=lambda r: (r["member"].startswith("The rest ("), -(r["share_of_factor_pct"] or 0.0)))
    return {"factor": factor.label, "members": rows,
            **({"no_measurable_effect": len(idle)} if idle else {}),
            "note": "Measured with the other factors as they are now."
                    + (f" {len(idle)} member(s) had no measurable effect." if idle else "")}


# ---------------------------------------------------------------------------
# Over a window
# ---------------------------------------------------------------------------

# A window's moment pairs solved per combination: as many as this budget of
# power flows allows for 2^k combinations (all of a day with three factors),
# never fewer than WINDOW_MIN_PAIRS; and at most WINDOW_MAX_MOMENTS prepared.
WINDOW_POWER_FLOWS = 1600
WINDOW_MIN_PAIRS = 24
WINDOW_MAX_MOMENTS = 192
WINDOW_MAX_FACTORS = 5
# The order in which a window's categories are kept when there are too many.
_CATEGORY_ORDER = ("Demand near", "Demand elsewhere", "Demand", "Units' output", "Non-dispatchable generation",
                   "Voltage set-points", "Topology and unit status", "Elements added", "Other changes")


def coarse_groups(ref_net, cur_net, template: str, changes: list[Change], label: str,
                  buses: set[int]) -> dict[str, frozenset[int]]:
    """The changes of one moment pair in categories that keep their name from
    one moment to the next (no counts, no single-member names), so a window's
    factors are the same at every moment."""
    near = _near_buses(cur_net, set(buses)) if buses else set()
    out: dict[str, set[int]] = {}
    for f in group(ref_net, cur_net, template, changes, label, set(buses)):
        if f.kind == "demand":
            for m in f.members:
                bus = _bus_of(ref_net, cur_net, changes[m.changes[0]])
                key = ("Demand" if not buses else
                       f"Demand near {label}" if bus in near else "Demand elsewhere")
                out.setdefault(key, set()).update(m.changes)
            continue
        key = {"units": "Units' output", "renewables": "Non-dispatchable generation", "setpoint": "Voltage set-points",
               "added": "Elements added", "other": "Other changes"}.get(f.kind)
        if f.kind == "unit":
            key = "Units' output" if f.label.startswith("Output of") else "Topology and unit status"
        out.setdefault(key or "Topology and unit status", set()).update(f.changes)
    return {k: frozenset(v) for k, v in out.items()}


def _hypothesis_groups(ref_net, cur_net, template, changes, hypotheses) -> dict[str, frozenset[int]]:
    factors, _ = group_hypotheses(ref_net, cur_net, template, changes, hypotheses)
    return {("Other changes" if f.kind == "other" else f.label): f.changes for f in factors}


def _rank(category: str) -> int:
    return next((i for i, prefix in enumerate(_CATEGORY_ORDER) if category.startswith(prefix)), len(_CATEGORY_ORDER))


def decompose_window(pairs: list[tuple], tick: Target, aggregate: str, hours: float,
                     hypotheses: list[dict] | None = None) -> dict:
    """Divide the change of a quantity summed over a window among its causes.

    `pairs` are (reference net, current net) for each moment of the two
    windows, paired in order; `tick` reads each moment's value; `aggregate`
    turns it into the window's quantity: "exceedance" (how far outside its
    limits, times the hours it lasted), "violations" (element-hours outside
    limits) or "mean" (the average). Factors are categories that keep their
    name across moments (`coarse_groups`, or the caller's hypotheses), at most
    WINDOW_MAX_FACTORS; each combination solves every pair.
    """
    excess = tick.excess or (lambda value: 0.0 if tick.within(value) else 1.0)

    # The moment pairs solved: as many as the power-flow budget allows for the
    # number of factors, evenly spread, each standing for the ones around it.
    all_groups = []
    for ref_net, cur_net in pairs:
        template, changes = diff(ref_net, cur_net)
        all_groups.append((template, changes, _hypothesis_groups(ref_net, cur_net, template, changes, hypotheses)
                           if hypotheses else coarse_groups(ref_net, cur_net, template, changes, tick.label,
                                                            set(tick.buses))))
    k_est = min(WINDOW_MAX_FACTORS, len({c for *_, g in all_groups for c in g}) or 1)
    most = max(WINDOW_MIN_PAIRS, WINDOW_POWER_FLOWS // (1 << k_est))
    keep = list(range(len(pairs))) if len(pairs) <= most else \
        sorted({round(i * (len(pairs) - 1) / (most - 1)) for i in range(most)})
    weight = len(pairs) / len(keep)
    n_total = len(pairs)

    def amount(value: float) -> float:
        if aggregate == "mean":
            return value / len(keep)
        if aggregate == "violations":
            return value * hours * weight
        return excess(value) * hours * weight

    combos, groups, power_flows = [], [], 0
    real = {"reference": 0.0, "current": 0.0}
    for i in keep:
        ref_net, cur_net = pairs[i]
        template, changes, found = all_groups[i]
        combos.append(Combinations(ref_net, cur_net, template, changes, tick.metric))
        groups.append(found)
        for side, net in (("reference", ref_net), ("current", cur_net)):
            solo = copy.deepcopy(net)
            value = tick.metric(solo) if _solve(solo) else None
            power_flows += 1
            if value is None:
                return {"status": "not_decomposed", "power_flows": power_flows,
                        "reason": "The power flow does not solve at one of the moments compared."}
            real[side] += amount(value)
    change = real["current"] - real["reference"]
    names = sorted({c for g in groups for c in g}, key=lambda c: (_rank(c), c))
    if len(names) > WINDOW_MAX_FACTORS:
        kept, merged = names[:WINDOW_MAX_FACTORS - 1], names[WINDOW_MAX_FACTORS - 1:]
        groups = [{**{c: g[c] for c in kept if c in g},
                   **({"Other changes": frozenset().union(*(g[c] for c in merged if c in g))}
                      if any(c in g for c in merged) else {})} for g in groups]
        names = kept + ["Other changes"]
    base = {"aggregate": aggregate, "moments_compared": n_total, "moments_solved": len(keep),
            "value": real["current"], "reference_value": real["reference"], "change": change}
    if abs(change) < 1e-9 or not names:
        return {**base, "status": "no_change", "power_flows": power_flows,
                "reason": "The quantity is the same over both windows: there is no change to divide."}

    def v(mask: int) -> float:
        total = 0.0
        for c, g in zip(combos, groups):
            now = frozenset().union(*(g[n] for i, n in enumerate(names) if mask >> i & 1 and n in g))
            value = c.value(now)
            if value is None:
                raise Unsolved()
            total += amount(value)
        return total

    k = len(names)
    try:
        phi = shapley(k, v)
        v_none, v_all = v(0), v((1 << k) - 1)
    except Unsolved:
        return {**base, "status": "not_decomposed", "power_flows": power_flows + sum(c.power_flows for c in combos),
                "reason": "Some mixes of the two windows have no power-flow solution."}
    gap = max(abs(v_all - real["current"]), abs(v_none - real["reference"]))
    nd = tick.decimals
    rows = [{"factor": n, "kind": "category" if not hypotheses else "hypothesis", "members": [], "n_members": 0,
             "detail": "", "effect": round(phi[i], nd + 1), "share_pct": round(100 * phi[i] / change, 1),
             "verdict": verdict(phi[i] / change), "alone_value": round(v(1 << i), nd),
             "undo_alone_value": round(v(((1 << k) - 1) & ~(1 << i)), nd), "undo_alone_within_limits": None}
            for i, n in enumerate(names)]
    rows.sort(key=lambda r: -r["share_pct"])
    causes = [r for r in rows if r["verdict"] == "cause"]
    return {**base, "status": "explained" if gap <= max(tick.closure_tol * len(pairs), 1e-9) else "not_decomposed",
            "grouping": "hypotheses" if hypotheses else "standard", "factors": rows,
            "main_cause": causes[0]["factor"] if causes else None, "drill_down": [],
            "interactions": interactions(names, v, change, nd),
            "closure": {"reproduces_both_states": gap <= max(tick.closure_tol * len(pairs), 1e-9),
                        "largest_gap": round(gap, 7)},
            "power_flows": power_flows + sum(c.power_flows for c in combos),
            **({} if gap <= max(tick.closure_tol * len(pairs), 1e-9) else
               {"reason": "The two windows differ in something the decomposition cannot separate."})}


# ---------------------------------------------------------------------------
# The sentence
# ---------------------------------------------------------------------------

def _fmt(value: float, quantity: str) -> str:
    """A value in its quantity's own words."""
    return {
        "vm_pu": f"{value:.3f} p.u.", "loading_percent": f"{value:.1f} %",
        "violations": f"{value:g} element{'s' if abs(value - 1) > 1e-9 else ''} outside its limits",
        "exceedance_pu_h": f"{value:.4f} p.u.·h outside its limits",
        "exceedance_pct_h": f"{value:.1f} %·h outside its limits",
        "mean_vm_pu": f"{value:.4f} p.u. on average", "mean_loading_percent": f"{value:.1f} % on average",
        "violation_hours": f"{value:.2f} element-hours outside their limits",
    }.get(quantity, f"{value:g}")


def _fmt_change(value: float, quantity: str) -> str:
    return {
        "vm_pu": f"{abs(value):.3f} p.u.", "loading_percent": f"{abs(value):.1f} points",
        "violations": f"{abs(value):g} element{'s' if abs(abs(value) - 1) > 1e-9 else ''} outside limits",
        "exceedance_pu_h": f"{abs(value):.4f} p.u.·h", "exceedance_pct_h": f"{abs(value):.1f} %·h",
        "mean_vm_pu": f"{abs(value):.4f} p.u.", "mean_loading_percent": f"{abs(value):.1f} points",
        "violation_hours": f"{abs(value):.2f} element-hours",
    }.get(quantity, f"{abs(value):g}")


# Factor names that start with a common word, lower-cased inside a sentence
# (a unit's or element's own name never is).
_COMMON_FIRST = {"Demand", "Output", "Units'", "Voltage", "External", "Non-dispatchable", "Other", "Elements"}


def _lower_first(label: str) -> str:
    first = label.split(" ", 1)[0]
    return label[0].lower() + label[1:] if first in _COMMON_FIRST else label


def _head(result: dict) -> str:
    """The comparison, in the words of its quantity."""
    q, element, sym, ref = result["quantity"], result["element"], result["symptom"], result["reference"]
    if q == "violations":
        now = (f"{sym['value']:g} elements are outside their limits" if abs(sym['value'] - 1) > 1e-9
               else "1 element is outside its limits")
        return f"{now} at {sym['label']}; at {ref['label']} ({ref['why']}) there were {ref['value']:g}."
    if q not in ("vm_pu", "loading_percent"):
        return (f"Over {sym['label']}, {element} has {_fmt(sym['value'], q)}; over {ref['label']} "
                f"({ref['why']}) it had {_fmt(ref['value'], q)}.")
    since = result.get("outside_limits_since")
    head = (f"{element} is at {_fmt(sym['value'], q)} at {sym['label']}"
            + (f", outside its limit ({_fmt(sym['limit'], q)})" if sym.get("violated") else "")
            + (f" since {str(since)[11:16] if str(since)[:10] == str(sym['timestamp'])[:10] else str(since)[:16]}"
               if since else "")
            + f"; at {ref['label']} ({ref['why']}) it was {_fmt(ref['value'], q)}")
    return head + ("" if head.endswith(".") else ".")


def sentence(result: dict) -> str:
    """The decomposition in two to four sentences, from its own numbers."""
    q, element, sym = result["quantity"], result["element"], result["symptom"]
    rise = "rise" if result["change"] > 0 else "fall"
    parts = [f"{r['factor']} {r['share_pct']:.0f} %" for r in result["factors"] if r["verdict"] in ("cause", "contributes")]
    against = [r["factor"] for r in result["factors"] if r["verdict"] == "counteracts"]
    # "the 0.009 p.u. fall", but "the rise of 2 elements outside limits".
    what = (f"{rise} of {_fmt_change(result['change'], q)}" if q in ("violations", "violation_hours")
            else f"{_fmt_change(result['change'], q)} {rise}")
    body = f" Of the {what}: " + "; ".join(parts) + "." if parts else (
        f" The {what} is spread over many small changes, none decisive.")
    if against:
        body += f" Pushing the other way: {', '.join(against)}."
    unchanged = [r["factor"] for r in result["factors"] if r.get("nothing_changed")]
    if unchanged:
        body += f" Ruled out, as nothing it names changed between the two states: {', '.join(unchanged)}."
    for pair in (result.get("interactions") or [])[:2]:
        a, b = pair["factors"]
        body += (f" {a} and {_lower_first(b)} {'amplify' if pair['kind'] == 'amplify' else 'dampen'} each other "
                 f"({pair['share_pct']:+.0f} % of the change comes from the two together).")
    undo = [r for r in result["factors"] if r["verdict"] == "cause" and r.get("undo_alone_within_limits")
            and sym.get("violated")]
    if undo:
        r = undo[0]
        if q == "violations":
            body += f" Undoing only {_lower_first(r['factor'])} would leave no element outside its limits."
        else:
            body += (f" Undoing only {_lower_first(r['factor'])} would bring {element} back within limits "
                     f"({_fmt(r['undo_alone_value'], q)}).")
    return _head(result) + body
