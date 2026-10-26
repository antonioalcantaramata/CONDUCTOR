"""
classical.py — the classical views beside an investigation's exact shares.

The investigation divides a change exactly (Shapley values over AC power flows
of every combination, investigation.py). Power-system engineering has older,
cheaper ways to look at the same question, and they are worth seeing next to
it — as a complement, never in its place:

1. **A linear estimate from sensitivities.** Each factor's effect as its
   sensitivity times its change: the directional derivative of the quantity
   along the factor's change, at both end states, averaged (the trapezoid
   rule). That is a two-point approximation of the Aumann–Shapley value, the
   continuous counterpart used in loss and cost allocation. It needs two power
   flows per factor. Where it differs from the exact share, the quantity is not
   linear over the change: large changes, causes that interact.
2. **Line outage distribution factors (LODF, DC model).** A switching change
   has no sensitivity. For a branch taken out of service and a branch loading
   investigated, the classical estimate is the DC LODF: the monitored flow
   picks up LODF × the outaged flow.
3. **Power-flow tracing (proportional sharing, Bialek).** Not a cause but a
   picture: which sources feed the investigated branch's flow and which loads
   it serves, now and in the reference state — on the solved AC flows, lossless
   (each branch's flow taken as the mean of its two ends).

None of these is used to decide anything: the verdicts stay the exact ones.
"""

from __future__ import annotations

import copy
import math

import numpy as np

import generators as gm
from element_names import bus_display_name

EPSILON = 0.05             # the step along a factor's change for its directional derivative
# Columns a change can move continuously; any other (in service, bus, tap,
# parallel units) is a switch, taken at whichever end is nearer.
CONTINUOUS = {"p_mw", "q_mvar", "vm_pu", "va_degree", "scaling", "r_ohm_per_km", "x_ohm_per_km", "c_nf_per_km",
              "g_us_per_km", "length_km", "max_i_ka", "sn_mva", "vk_percent", "vkr_percent", "pfe_kw",
              "i0_percent", "shift_degree", "vn_hv_kv", "vn_lv_kv", "df"}
_INJECTIONS = ("load", "sgen", "gen")
TOP_TRACED = 5


def _number(value) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool)


def _appears(ch) -> bool | None:
    """True for an element only the current state has, False for one only the
    reference has, None for one in both."""
    if set(ch.cur) == {"in_service"} and ch.ref == {"in_service": False} and ch.table in _INJECTIONS:
        return True if ch.cur["in_service"] else None
    if set(ch.cur) == {"in_service"} and ch.cur == {"in_service": False} and ch.table in _INJECTIONS:
        return False if ch.ref["in_service"] else None
    return None


class _Path:
    """The network with each change a fraction t of the way from the reference
    (t = 0) to now (t = 1): continuous columns in proportion, switches at the
    nearer end, an injection that appears scaled by t."""

    def __init__(self, ref_net, cur_net, template: str, changes):
        self.net = copy.deepcopy(cur_net if template == "current" else ref_net)
        self.changes = changes
        # The full values of injections that only one state has, to scale them.
        self.full = {}
        for i, ch in enumerate(changes):
            if _appears(ch) is not None:
                tbl = self.net[ch.table]
                self.full[i] = {k: float(tbl.at[ch.idx, k]) for k in ("p_mw", "q_mvar") if k in tbl.columns}
        self.power_flows = 0

    def value(self, t: list[float], metric) -> float | None:
        from investigation import _solve

        for i, ch in enumerate(self.changes):
            tbl = self.net[ch.table]
            appears = _appears(ch)
            if appears is not None:
                share = t[i] if appears else 1.0 - t[i]
                tbl.at[ch.idx, "in_service"] = share > 0
                for k, v in self.full[i].items():
                    tbl.at[ch.idx, k] = v * share
                continue
            for k in ch.cur:
                a, b = ch.ref[k], ch.cur[k]
                if k in CONTINUOUS and _number(a) and _number(b):
                    tbl.at[ch.idx, k] = a + t[i] * (b - a)
                else:
                    tbl.at[ch.idx, k] = b if t[i] > 0.5 else a
        self.power_flows += 1
        return metric(self.net) if _solve(self.net) else None


def _differentiable(changes, members: frozenset) -> tuple[bool, bool]:
    """(has a continuous part, has a switching part) among a factor's changes."""
    cont = switch = False
    for i in members:
        ch = changes[i]
        if _appears(ch) is not None:
            cont = True
            continue
        for k in ch.cur:
            if k in CONTINUOUS and _number(ch.ref[k]) and _number(ch.cur[k]):
                cont = True
            else:
                switch = True
    return cont, switch


def linear_estimates(ref_net, cur_net, template: str, changes, factors, sets, target, v_ref: float,
                     v_now: float, exact: dict[str, float], change: float) -> dict:
    """Each factor's effect estimated from sensitivities (trapezoid of its
    directional derivative at both ends), beside its exact share."""
    path = _Path(ref_net, cur_net, template, changes)
    n = len(changes)
    rows = []
    for f, members in zip(factors, sets):
        cont, switch = _differentiable(changes, members)
        row = {"factor": f.label, "exact_share_pct": exact.get(f.label)}
        if not cont:
            lodf = _lodf_estimate(ref_net, cur_net, changes, members, target) if switch else None
            if lodf is not None:
                row.update(method="LODF (DC)", effect=lodf)
            else:
                row.update(method=None, effect=None,
                           why="a switching change has no sensitivity" if switch else "nothing continuous changed")
            rows.append(row)
            continue
        at_ref = [EPSILON if i in members else 0.0 for i in range(n)]
        at_now = [1.0 - EPSILON if i in members else 1.0 for i in range(n)]
        up, down = path.value(at_ref, target.metric), path.value(at_now, target.metric)
        if up is None or down is None:
            row.update(method=None, effect=None, why="a small step along it has no power-flow solution")
            rows.append(row)
            continue
        d_ref, d_now = (up - v_ref) / EPSILON, (v_now - down) / EPSILON
        row.update(method="sensitivities (trapezoid)", effect=0.5 * (d_ref + d_now),
                   at_reference=d_ref, at_now=d_now)
        if switch:
            row["partial"] = "its switching part (an element in or out of service) has no sensitivity"
        rows.append(row)
    nd = target.decimals + 1
    for row in rows:
        if row.get("effect") is not None:
            row["effect"] = round(row["effect"], nd)
            row["share_pct"] = round(100 * row["effect"] / change, 1) if abs(change) > 1e-12 else None
            if row["exact_share_pct"] is not None and row["share_pct"] is not None:
                row["error_points"] = round(row["share_pct"] - row["exact_share_pct"], 1)
            for k in ("at_reference", "at_now"):
                if k in row:
                    row[k] = round(row[k], nd)
    estimated = [r for r in rows if r.get("share_pct") is not None]
    top = max(estimated, key=lambda r: r["share_pct"], default=None)
    exact_top = max(exact.items(), key=lambda kv: kv[1], default=(None, None))[0]
    errors = [abs(r["error_points"]) for r in rows if r.get("error_points") is not None]
    return {
        "factors": rows,
        "main_cause": top["factor"] if top else None,
        "agrees_with_exact": (top["factor"] == exact_top) if top else None,
        "covers_pct": round(100 * sum(r["effect"] for r in estimated) / change, 1) if estimated and change else None,
        "mean_abs_error_points": round(sum(errors) / len(errors), 1) if errors else None,
        "not_estimated": [r["factor"] for r in rows if r.get("effect") is None],
        "power_flows": path.power_flows,
    }


def _lodf_estimate(ref_net, cur_net, changes, members, target) -> float | None:
    """The DC LODF estimate for a factor that takes one branch out of service,
    when a branch's loading is investigated: the monitored flow picks up
    LODF × the outaged flow, and its loading scales with its flow."""
    etype, idx = getattr(target, "etype", None), getattr(target, "idx", None)
    if etype not in ("line", "trafo") or len(members) != 1:
        return None
    ch = changes[next(iter(members))]
    if ch.table not in ("line", "trafo") or set(ch.cur) != {"in_service"}:
        return None
    if ch.ref["in_service"] and not ch.cur["in_service"]:
        before, sign = ref_net, 1.0                     # taken out now: from the reference
    elif ch.cur["in_service"] and not ch.ref["in_service"]:
        before, sign = cur_net, -1.0                    # it was out in the reference
    else:
        return None
    from investigation import _solve
    from pandapower.pypower.makeLODF import makeLODF
    from pandapower.pypower.makePTDF import makePTDF

    net = copy.deepcopy(before)
    if not _solve(net):
        return None
    try:
        ppc = net._ppc
        look = net._pd2ppc_lookups["branch"]
        row_of = {t: {int(i): look[t][0] + p for p, i in enumerate(net[t].index)} for t in ("line", "trafo") if t in look}
        k, m = row_of[ch.table][ch.idx], row_of[etype][idx]
        ptdf = makePTDF(ppc["baseMVA"], ppc["bus"], ppc["branch"])
        lodf = makeLODF(ppc["branch"], ptdf)
    except Exception:  # noqa: BLE001 — no estimate is better than a wrong one
        return None
    flow = net[f"res_{etype}"].at[idx, "p_from_mw" if etype == "line" else "p_hv_mw"]
    outaged = net[f"res_{ch.table}"].at[ch.idx, "p_from_mw" if ch.table == "line" else "p_hv_mw"]
    loading = net[f"res_{etype}"].at[idx, "loading_percent"]
    factor = float(lodf[m, k])
    if not all(math.isfinite(x) for x in (flow, outaged, loading, factor)) or abs(flow) < 1e-6:
        return None
    after = loading * abs(flow + factor * outaged) / abs(flow)
    return sign * (after - loading)


# ---------------------------------------------------------------------------
# Power-flow tracing
# ---------------------------------------------------------------------------

def trace_branch(net, etype: str, idx: int) -> dict | None:
    """Which sources feed a branch's active flow and which loads it serves, by
    proportional sharing (Bialek) on a solved network, lossless: each branch's
    flow is the mean of its two ends. None when the branch carries nothing."""
    buses = list(net.bus.index)
    pos = {b: i for i, b in enumerate(buses)}
    nb = len(buses)
    flows = []                                           # (from bus, to bus, MW, table, index)
    for table in ("line", "trafo"):
        ends = ("from_bus", "to_bus") if table == "line" else ("hv_bus", "lv_bus")
        cols = ("p_from_mw", "p_to_mw") if table == "line" else ("p_hv_mw", "p_lv_mw")
        tbl, r = net[table], net[f"res_{table}"]
        for i in tbl.index[tbl["in_service"].astype(bool)]:
            a, b = float(r.at[i, cols[0]]), float(r.at[i, cols[1]])
            if not (math.isfinite(a) and math.isfinite(b)):
                continue
            mw = 0.5 * (abs(a) + abs(b))
            u, w = (int(tbl.at[i, ends[0]]), int(tbl.at[i, ends[1]])) if a >= 0 else \
                (int(tbl.at[i, ends[1]]), int(tbl.at[i, ends[0]]))
            flows.append((u, w, mw, table, int(i)))
    gen = np.zeros(nb)
    load = np.zeros(nb)
    sources: dict[int, list[tuple[str, float]]] = {}
    sinks: dict[int, list[tuple[str, float]]] = {}
    for table in ("gen", "sgen", "ext_grid"):
        tbl, r = net[table], net[f"res_{table}"]
        for i in tbl.index[tbl["in_service"].astype(bool)]:
            p = float(r.at[i, "p_mw"]) if i in r.index else float("nan")
            if not math.isfinite(p):
                continue
            b = pos[int(tbl.at[i, "bus"])]
            name = "the external grid" if table == "ext_grid" else gm.unit_name(net, table, i)
            if p >= 0:
                gen[b] += p
                sources.setdefault(b, []).append((name, p))
            else:
                load[b] += -p
                sinks.setdefault(b, []).append((name, -p))
    for i in net.load.index[net.load["in_service"].astype(bool)]:
        p = float(net.res_load.at[i, "p_mw"])
        if math.isfinite(p) and p > 0:
            b = pos[int(net.load.at[i, "bus"])]
            load[b] += p
            sinks.setdefault(b, []).append((f"Demand at {bus_display_name(net, int(net.load.at[i, 'bus']))}", p))
    up = np.eye(nb)
    down = np.eye(nb)
    through_in = gen.copy()
    through_out = load.copy()
    for u, w, mw, *_ in flows:
        through_in[pos[w]] += mw
        through_out[pos[u]] += mw
    for u, w, mw, *_ in flows:
        if through_in[pos[u]] > 1e-9:
            up[pos[w], pos[u]] -= mw / through_in[pos[u]]
        if through_out[pos[w]] > 1e-9:
            down[pos[u], pos[w]] -= mw / through_out[pos[w]]
    try:
        up_inv, down_inv = np.linalg.inv(up), np.linalg.inv(down)
    except np.linalg.LinAlgError:
        return None
    mine = next((f for f in flows if f[3] == etype and f[4] == idx), None)
    if mine is None or mine[2] < 1e-6:
        return None
    u, w, mw = mine[0], mine[1], mine[2]
    from_src: dict[str, float] = {}
    if through_in[pos[u]] > 1e-9:
        for b, entries in sources.items():
            weight = up_inv[pos[u], b] * mw / through_in[pos[u]]
            for name, p in entries:
                from_src[name] = from_src.get(name, 0.0) + weight * p
    to_sink: dict[str, float] = {}
    if through_out[pos[w]] > 1e-9:
        for b, entries in sinks.items():
            weight = down_inv[pos[w], b] * mw / through_out[pos[w]]
            for name, p in entries:
                to_sink[name] = to_sink.get(name, 0.0) + weight * p

    def ranked(shares: dict[str, float]) -> list[dict]:
        total = sum(v for v in shares.values() if v > 0) or 1.0
        rows = sorted(((n, v) for n, v in shares.items() if v > 1e-9), key=lambda kv: -kv[1])
        out = [{"name": n, "mw": round(v, 2), "share_pct": round(100 * v / total, 1)} for n, v in rows[:TOP_TRACED]]
        rest = sum(v for _, v in rows[TOP_TRACED:])
        if rest > 1e-9:
            out.append({"name": f"The rest ({len(rows) - TOP_TRACED})", "mw": round(rest, 2),
                        "share_pct": round(100 * rest / total, 1)})
        return out
    return {"direction": f"{bus_display_name(net, u)} → {bus_display_name(net, w)}", "flow_mw": round(mw, 2),
            "sources": ranked(from_src), "sinks": ranked(to_sink)}
