"""
flex_options.py — several corrective dispatches instead of one.

The flexibility OPF returns the dispatch that changes the schedule least. An
operator choosing a corrective action weighs things that objective does not
see: what an action costs, how many units have to be called, whether one
unit is trusted today, how much the grid leans on the external connection.
So the solver offers **options**, each the solution of a strategy an operator
already understands, all scored the same way, and the operator chooses.

Options come from the engine, not the language model: every option is a
re-solve of the same (already capped) AC-OPF, so all of them respect exactly
the same limits; every figure is read from the solved model.

Strategies (stable ids — "fewest_units" means the same thing in every answer):

  least_disturbance    the OPF as it is: the smallest change to the schedule
  cheapest             the lowest cost of the corrective action at the cost
                       table's prices (redispatch, reactive power, import,
                       the external grid's reactive charge)
  fewest_units         as few units as possible get an instruction: all but the
                       top-k units of least_disturbance keep their set-points,
                       k increased until feasible
  without_main_unit    least_disturbance's biggest mover gets no instruction

An instruction is what an operator would have to issue: a change of a unit's
active power, and of its voltage set-point when it holds a voltage (its
reactive output then follows the grid, as it would untouched) — or of its
reactive output when it does not. "Units called" counts instructions.
  least_import         the smallest import from the external grid, then the
                       smallest change to the schedule at that import
  most_headroom        every branch as far below its rating as possible: the
                       lowest highest loading (more margin for what comes
                       next — usually, not always, a better N-1 result)
  demand_response      the cheapest action when flexible loads may also be
                       reduced at their demand-response price; offered only
                       when it does use demand response

The remedies after unit redispatch, in order: demand response (a share of a
load reduced on request, at its price), then load relief (involuntary, the
last resort, at the value of lost load). The model can reduce loads only when
built with `load_flex` (flex_engine), and only a strategy here frees them.
When redispatch alone cannot clear the violations, `last_resort` gives the
least load reduction that, with redispatch, does — demand response before any
relief — and the cheapest at the cost table's prices.

Near-optimal alternatives (`near_optimal`, modelling to generate alternatives,
MGA): the cheapest option, then up to MAX_ALTERNATIVES more, each costing at
most ε above it (`add_cost` bounded). They are drawn from a pool several
methods fill, every one inside the same budget:

  hsj          Hop-Skip-Jump: away from the units earlier alternatives change
               (weights on |ΔP|)
  unit_ranges  each unit as low, and as high, as it can be — which also gives
               every unit's near-optimal range ("Gen_bus1: 40–72 MW within 10 %")
  objectives   the other strategies inside the budget: the fewest units, the
               least import, the most headroom, the least disturbance
  random       seeded random directions in the space of unit outputs

and the shown ones are picked from the pool farthest-first (each the most
different from those already shown), so the set is as diverse as the margin
allows. An operator who distrusts a unit or prefers fewer instructions sees
what an alternative costs, as computed by the solver: "+4 % above the
cheapest, avoids Gen_bus1".

Pure functions over a solved Pyomo model: no app state, no HTTP.
"""

from __future__ import annotations

import math
from typing import Callable, NamedTuple

from pyomo.environ import (Constraint, NonNegativeReals, Objective, Param, SolverFactory, Var, maximize,
                           minimize, value)
from pyomo.opt import TerminationCondition

# A unit "moved" when its active power changed by at least this much; below it
# the difference is solver tolerance, not an instruction anyone would issue.
MOVE_MW = 0.01
MOVE_MVAR = 0.05
MOVE_VM = 0.001              # p.u.: a set-point change worth an instruction
MAX_OPTIONS = 5
# Near-optimal alternatives: how many, how far above the cheapest by default,
# and the smallest budget when the cheapest costs about nothing (or saves).
MAX_ALTERNATIVES = 5
NEAR_OPTIMAL_PCT = 10.0
NEAR_OPTIMAL_METHODS = ("hsj", "unit_ranges", "objectives", "random")
HSJ_ROUNDS, RANDOM_DIRECTIONS = 3, 4
DIVERSE_MW = 1.0             # an alternative closer than this to one shown adds nothing
NEAR_OPTIMAL_FLOOR_EUR_H = 1.0
USED_WEIGHT = 10.0           # HSJ: moving a unit an earlier option moved costs this much more
MAX_FEWEST_K = 4             # at most this many solves to find the fewest units
DUPLICATE_MW = 0.1           # two options closer than this on every unit are one
RELIEF_WEIGHT = 100.0        # last resort: a MW of load relief counts as this many of demand response

STRATEGIES = {
    "least_disturbance": ("Least disturbance", "the smallest change to the schedule"),
    "cheapest": ("Cheapest", "the lowest cost of the corrective action at the cost table's prices"),
    "fewest_units": ("Fewest units", "as few units as possible get an instruction; every other unit keeps "
                                     "its active power and set-point"),
    "without_main_unit": ("Without {unit}", "{unit}, which the least-disturbance option moves most, gets "
                                            "no instruction"),
    "least_import": ("Least import", "the smallest import from the external grid, then the smallest "
                                     "change to the schedule"),
    "most_headroom": ("Most headroom", "every line as far below its rating as possible: the lowest highest "
                                       "loading"),
    "demand_response": ("With demand response", "the cheapest action when flexible loads may also be reduced, at "
                                                "their demand-response price"),
    "last_resort_least": ("Least load reduction", "unit redispatch alone cannot clear the violations: the smallest "
                                                  "load reduction that, with redispatch, does — demand response "
                                                  "first, load relief only beyond it"),
    "last_resort_cheapest": ("Cheapest with load reduction", "unit redispatch alone cannot clear the violations: "
                                                             "the cheapest action with loads reduced at their "
                                                             "prices (demand response, then load relief)"),
    # Not "N-1 secure": an outage the study cannot secure is reported, not hidden.
    "n1_preventive": ("N-1 preventive", "a dispatch that stays within limits after each studied outage: "
                                        "nothing to do when one happens"),
    "n1_corrective": ("N-1 corrective", "a dispatch that is brought back within limits after each studied "
                                        "outage by a quick action of the units (its corrective plan)"),
    **{f"near_optimal_{k}": (f"Near-optimal {k}", "within the margin of the cheapest, as different as possible "
                                                   "from the options before it") for k in range(1, MAX_ALTERNATIVES + 1)},
}


class Prices(NamedTuple):
    """Prices of a corrective action, €. None = not priced (counted as 0)."""
    up: dict            # unit -> €/MWh of upward redispatch
    down: dict          # unit -> €/MWh of downward redispatch
    q: dict             # unit -> €/Mvarh of reactive power changed
    import_: dict       # ext grid -> €/MWh of import (change from the schedule)
    band: dict          # ext grid -> free reactive band, tan φ
    penalty: dict       # ext grid -> €/Mvarh beyond the band
    dr: dict | None = None        # load -> €/MWh of demand response
    relief: dict | None = None    # load -> €/MWh of load relief


class Option(NamedTuple):
    id: str
    label: str
    how: str
    p: dict             # unit -> MW after
    q: dict             # unit -> Mvar after
    p0: dict            # unit -> MW scheduled
    q0: dict
    pext: dict          # ext grid -> MW after (positive = import)
    qext: dict
    v: dict             # bus -> p.u.
    max_loading_pct: float | None
    max_loading_branch: tuple | None
    v0: dict | None = None        # bus -> p.u. as scheduled (a voltage-controlled unit's set-point)
    vc_bus: dict | None = None    # voltage-controlled unit -> its bus
    dr: dict | None = None        # load -> MW reduced by demand response
    relief: dict | None = None    # load -> MW of load relief

    @property
    def load_reduction(self) -> float:
        """MW of load reduced, demand response and relief together."""
        return sum((self.dr or {}).values()) + sum((self.relief or {}).values())

    @property
    def moved(self) -> dict:
        """unit -> ΔP for the units that moved."""
        return {g: self.p[g] - self.p0[g] for g in self.p if abs(self.p[g] - self.p0[g]) >= MOVE_MW}

    @property
    def instructions(self) -> dict:
        """unit -> what it is told to change: "P", "V" (the set-point of a
        unit that holds a voltage), "Q" (the reactive output of one that does
        not), or both ("P+V"). A held voltage moves a unit's reactive output
        without any instruction; that is not counted."""
        out = {}
        vc = self.vc_bus or {}
        for g in self.p:
            what = ["P"] if abs(self.p[g] - self.p0[g]) >= MOVE_MW else []
            if g in vc and self.v0 is not None:
                b = vc[g]
                if b in self.v and b in self.v0 and abs(self.v[b] - self.v0[b]) >= MOVE_VM:
                    what.append("V")
            elif abs(self.q[g] - self.q0[g]) >= MOVE_MVAR:
                what.append("Q")
            if what:
                out[g] = "+".join(what)
        return out


# ---------------------------------------------------------------------------
# Solving
# ---------------------------------------------------------------------------

def solve(model) -> bool:
    """Solve in place; True only for an optimal solution (else the model keeps
    the values it had)."""
    try:
        res = SolverFactory("ipopt").solve(model, tee=False, load_solutions=False)
    except Exception:  # noqa: BLE001 — an option that cannot be solved is not offered
        return False
    if res.solver.termination_condition != TerminationCondition.optimal:
        return False
    model.solutions.load_from(res)
    return True


def _disturbance(m):
    return sum((m.Pg_new[g] - m.Pg_base[g]) ** 2 for g in m.G)


def _replace_objective(m, expr) -> None:
    for obj in list(m.component_objects(Objective, active=True)):
        obj.deactivate()
    m.options_obj = Objective(expr=expr, sense=minimize)


def branch_loadings(m) -> tuple[float | None, tuple | None]:
    """The highest branch current as a percentage of its rating, and where —
    computed the way the OPF's own thermal check computes it."""
    worst, where = None, None
    for name in ("branch_current_from_limit", "branch_current_to_limit"):
        c = getattr(m, name, None)
        if c is None:
            continue
        for (i, j) in c:
            rating_sq = None
            for a, b in ((i, j), (j, i)):
                if (a, b) in m.I_from_max and float(value(m.Ibase_from_kA[a, b])):
                    rating_sq = (float(value(m.I_from_max[a, b])) / float(value(m.Ibase_from_kA[a, b]))) ** 2
                    break
            if not rating_sq:
                continue
            current_sq = max(float(value(c[i, j].body)), 0.0)
            pct = 100.0 * math.sqrt(current_sq / rating_sq)
            if worst is None or pct > worst:
                worst, where = pct, (int(i), int(j))
    return worst, where


def _reduced(m, var) -> dict:
    """load -> MW a load-reduction variable took (above solver tolerance)."""
    if not hasattr(m, "LFlex"):
        return {}
    out = {}
    for load in m.LFlex:
        mw = float(value(var[load]) or 0.0)
        if mw >= MOVE_MW:
            out[int(load)] = mw
    return out


def capture(m, sid: str, unit: str | None = None) -> Option:
    """The solved model as an option."""
    label, how = STRATEGIES[sid]
    loading, where = branch_loadings(m)
    return Option(
        id=sid, label=label.format(unit=unit), how=how.format(unit=unit),
        p={g: float(value(m.Pg_new[g])) for g in m.G}, q={g: float(value(m.Qg_new[g])) for g in m.G},
        p0={g: float(value(m.Pg_base[g])) for g in m.G}, q0={g: float(value(m.Qg_base[g])) for g in m.G},
        pext={e: float(value(m.Pext[e])) for e in m.Ext}, qext={e: float(value(m.Qext[e])) for e in m.Ext},
        v={int(b): float(value(m.V[b])) for b in m.B if m.V[b].value is not None},
        max_loading_pct=loading, max_loading_branch=where,
        v0=_scheduled_voltages(m),
        vc_bus={g: int(value(m.gen_bus[g])) for g in getattr(m, "CondSet", [])},
        dr=_reduced(m, m.Pdr) if hasattr(m, "Pdr") else {},
        relief=_reduced(m, m.Pshed) if hasattr(m, "Pshed") else {},
    )


def _scheduled_voltages(m) -> dict | None:
    """bus -> p.u. at the schedule (the power flow the OPF started from): a
    voltage-controlled unit's bus holds its set-point there."""
    start = (getattr(m, "_diagnosis_start", None) or {}).get("V")
    return {int(b): float(v) for b, v in start.items()} if start else None


# ---------------------------------------------------------------------------
# Cost of a corrective action — one definition, used to score every option and
# (as a Pyomo expression) to find the cheapest; MGA will bound it.
# ---------------------------------------------------------------------------

def _price(table: dict, key) -> float:
    v = table.get(key)
    return float(v) if v is not None else 0.0


def action_cost(option: Option, prices: Prices, pext0: dict, qext0: dict) -> dict:
    """€/h of moving from the schedule to `option`: redispatch, reactive power,
    import and the external grid's reactive charge, each as a change from the
    schedule (doing nothing costs 0)."""
    parts = {"redispatch": 0.0, "reactive": 0.0, "import": 0.0, "reactive_exchange": 0.0}
    for g in option.p:
        dp, dq = option.p[g] - option.p0[g], option.q[g] - option.q0[g]
        parts["redispatch"] += _price(prices.up, g) * max(dp, 0.0) + _price(prices.down, g) * max(-dp, 0.0)
        parts["reactive"] += _price(prices.q, g) * abs(dq)
    for e in option.pext:
        parts["import"] += _price(prices.import_, e) * (option.pext[e] - pext0.get(e, 0.0))
        band = _price(prices.band, e) * abs(pext0.get(e, 0.0))
        before = max(abs(qext0.get(e, 0.0)) - band, 0.0)
        after = max(abs(option.qext[e]) - band, 0.0)
        parts["reactive_exchange"] += _price(prices.penalty, e) * (after - before)
    # Load reductions, only where an option has them (a breakdown stays as it was otherwise).
    if option.dr:
        parts["demand_response"] = sum(_price(prices.dr or {}, k) * mw for k, mw in option.dr.items())
    if option.relief:
        parts["load_relief"] = sum(_price(prices.relief or {}, k) * mw for k, mw in option.relief.items())
    parts = {k: round(v, 2) for k, v in parts.items()}
    return {"total": round(sum(parts.values()), 2), **parts}


def add_cost(m, prices: Prices, pext0: dict, qext0: dict):
    """`action_cost` as a Pyomo expression on `m` (adds the split variables
    it needs); returns the expression. Its value at a solution equals
    `action_cost(...)["total"]` up to the constant reactive-exchange term."""
    m.dPp = Var(m.G, within=NonNegativeReals, initialize=0.0)
    m.dPm = Var(m.G, within=NonNegativeReals, initialize=0.0)
    m.dQp = Var(m.G, within=NonNegativeReals, initialize=0.0)
    m.dQm = Var(m.G, within=NonNegativeReals, initialize=0.0)
    m.split_p = Constraint(m.G, rule=lambda m, g: m.Pg_new[g] - m.Pg_base[g] == m.dPp[g] - m.dPm[g])
    m.split_q = Constraint(m.G, rule=lambda m, g: m.Qg_new[g] - m.Qg_base[g] == m.dQp[g] - m.dQm[g])
    m.q_beyond = Var(m.Ext, within=NonNegativeReals, initialize=0.0)
    band = {e: _price(prices.band, e) * abs(pext0.get(e, 0.0)) for e in m.Ext}
    m.q_beyond_up = Constraint(m.Ext, rule=lambda m, e: m.q_beyond[e] >= m.Qext[e] - band[e])
    m.q_beyond_down = Constraint(m.Ext, rule=lambda m, e: m.q_beyond[e] >= -m.Qext[e] - band[e])
    for g in m.G:                                   # warm start at the current solution
        dp, dq = value(m.Pg_new[g]) - value(m.Pg_base[g]), value(m.Qg_new[g]) - value(m.Qg_base[g])
        m.dPp[g].value, m.dPm[g].value = max(dp, 0.0), max(-dp, 0.0)
        m.dQp[g].value, m.dQm[g].value = max(dq, 0.0), max(-dq, 0.0)
    loads = 0.0
    if hasattr(m, "LFlex"):                         # zero while the loads are fixed (not reduced)
        loads = sum(_price(prices.dr or {}, int(k)) * m.Pdr[k] + _price(prices.relief or {}, int(k)) * m.Pshed[k]
                    for k in m.LFlex)
    return (sum(_price(prices.up, g) * m.dPp[g] + _price(prices.down, g) * m.dPm[g]
                + _price(prices.q, g) * (m.dQp[g] + m.dQm[g]) for g in m.G)
            + sum(_price(prices.import_, e) * (m.Pext[e] - pext0.get(e, 0.0))
                  + _price(prices.penalty, e) * m.q_beyond[e] for e in m.Ext)
            + loads)


# ---------------------------------------------------------------------------
# Strategies — each takes the solved least-disturbance model and returns an
# option, or None when it has no feasible dispatch (then it is not offered).
# ---------------------------------------------------------------------------

def _hold(m, units) -> None:
    """Keep these units' active power at their schedule."""
    for g in units:
        if not m.Pg_new[g].fixed:
            m.Pg_new[g].fix(value(m.Pg_base[g]))


def keep_setpoints(m, units) -> None:
    """These units get no instruction: active power at the schedule; a unit
    that holds a voltage keeps its set-point (its reactive output follows the
    grid, as it would untouched); any other unit keeps its reactive output."""
    _hold(m, units)
    v0 = _scheduled_voltages(m) or {}
    vc = set(getattr(m, "CondSet", []))
    for g in units:
        if g in vc:
            b = int(value(m.gen_bus[g]))
            if b in v0 and not m.V[b].fixed:
                m.V[b].fix(v0[b])
        elif not m.Qg_new[g].fixed:
            m.Qg_new[g].fix(value(m.Qg_base[g]))


def with_setpoints_kept(model, units) -> tuple:
    """The least-disturbance solution with `units` given no instruction at all
    (the operator's held units: their active power is already pinned; here
    their set-points too). Returns (model, True), or (the model as it was,
    False) when that has no feasible dispatch."""
    if not units:
        return model, True
    m = model.clone()
    keep_setpoints(m, units)
    return (m, True) if solve(m) else (model, False)


def cheapest(model, first: Option, prices: Prices, pext0: dict, qext0: dict) -> Option | None:
    m = model.clone()
    cost = add_cost(m, prices, pext0, qext0)
    # A small share of the least-disturbance objective: among equally cheap
    # dispatches, the one that changes the schedule least.
    _replace_objective(m, cost + 1e-3 * _disturbance(m))
    return capture(m, "cheapest") if solve(m) else None


def fewest_units(model, first: Option, *_) -> Option | None:
    called = first.instructions
    # The units least disturbance leans on most first: active power, then set-points.
    ranked = sorted(called, key=lambda g: (-abs(first.p[g] - first.p0[g]), "P" not in called[g]))
    for k in range(1, min(len(ranked), MAX_FEWEST_K + 1)):
        m = model.clone()
        keep_setpoints(m, [g for g in m.G if g not in ranked[:k]])
        if solve(m):
            return capture(m, "fewest_units")
    return None


def without_main_unit(model, first: Option, *_) -> Option | None:
    if not first.moved:
        return None
    main = max(first.moved, key=lambda g: abs(first.moved[g]))
    m = model.clone()
    keep_setpoints(m, [main])
    return capture(m, "without_main_unit", unit=main) if solve(m) else None


def _rating_sq(m, i, j):
    """A directed branch's rating, (I max / I base)², as the OPF's own limit reads it."""
    for a, b in ((i, j), (j, i)):
        if (a, b) in m.I_from_max and float(value(m.Ibase_from_kA[a, b])):
            return (float(value(m.I_from_max[a, b])) / float(value(m.Ibase_from_kA[a, b]))) ** 2
    return None


def most_headroom(model, first: Option, *_) -> Option | None:
    """The highest loading of any branch as low as it goes (a shared bound on
    every branch's current, minimised), then the smallest change."""
    m = model.clone()
    m.peak_loading_sq = Var(within=NonNegativeReals, initialize=1.0)
    bounds = []
    for name, direction in (("branch_current_from_limit", "from"), ("branch_current_to_limit", "to")):
        c = getattr(m, name, None)
        for (i, j) in c or []:
            rating = _rating_sq(m, i, j) if direction == "from" else _rating_sq(m, j, i)
            if rating:
                bounds.append((c[i, j].body, rating))
    if not bounds:
        return None
    m.headroom = Constraint(range(len(bounds)), rule=lambda m, k: bounds[k][0] <= m.peak_loading_sq * bounds[k][1])
    _replace_objective(m, m.peak_loading_sq + 1e-4 * _disturbance(m))
    return capture(m, "most_headroom") if solve(m) else None


def least_import(model, first: Option, *_) -> Option | None:
    if sum(first.pext.values()) <= MOVE_MW:          # already exporting: nothing to reduce
        return None
    m = model.clone()
    m.imported = Var(within=NonNegativeReals, initialize=max(sum(first.pext.values()), 0.0))
    m.imported_def = Constraint(expr=m.imported >= sum(m.Pext[e] for e in m.Ext))
    _replace_objective(m, m.imported + 1e-4 * _disturbance(m))
    if not solve(m):
        return None
    # Then the smallest change to the schedule at (about) that import.
    m.imported_cap = Constraint(expr=m.imported <= value(m.imported) * 1.001 + 1e-3)
    _replace_objective(m, _disturbance(m))
    return capture(m, "least_import") if solve(m) else None


def free_loads(m, demand_response: bool = True, relief: bool = False) -> bool:
    """Let the model reduce loads: demand response up to each load's
    contracted share, and (`relief`) load relief of the rest. False when the
    model cannot (built without `load_flex`, or nothing to reduce)."""
    if not hasattr(m, "LFlex"):
        return False
    freed = False
    for load in m.LFlex:
        if demand_response and (m.Pdr[load].ub or 0.0) > MOVE_MW:
            m.Pdr[load].unfix()
            freed = True
        if relief:
            m.Pshed[load].unfix()
            freed = True
    return freed


def _load_mw(m, var):
    return sum(var[k] for k in m.LFlex) if hasattr(m, "LFlex") else 0.0


def with_demand_response(model, first: Option, prices: Prices, pext0: dict, qext0: dict) -> Option | None:
    m = model.clone()
    if not free_loads(m, demand_response=True):
        return None
    cost = add_cost(m, prices, pext0, qext0)
    _replace_objective(m, cost + 1e-3 * _disturbance(m))
    return capture(m, "demand_response") if solve(m) else None


STRATEGY_FUNCTIONS: dict[str, Callable] = {
    "cheapest": cheapest,
    "fewest_units": fewest_units,
    "without_main_unit": without_main_unit,
    "least_import": least_import,
    "most_headroom": most_headroom,
    "demand_response": with_demand_response,       # last: the remedy after redispatch
}


def last_resort(model, prices: Prices, pext0: dict, qext0: dict, restart: Callable | None = None
                ) -> tuple[list[tuple[Option, dict]], list[dict]]:
    """When unit redispatch alone has no feasible dispatch: the least load
    reduction that, with redispatch, clears every violation (demand response
    first; a MW of relief counts RELIEF_WEIGHT MW of it), then — keeping that
    — the smallest change to the schedule; and the cheapest action with load
    reductions at their prices. `model` is left as it is; `restart` puts a
    clone back at a sensible starting point (the failed solve's is not one).
    Returns (options with their cost, what could not be found and why)."""
    found, missing = [], []
    m = model.clone()
    if restart:
        restart(m)
    if not free_loads(m, demand_response=True, relief=True):
        return [], [{"id": "last_resort", "label": "Load reduction",
                     "reason": "the model cannot reduce loads here (no load to reduce)"}]
    reduction = _load_mw(m, m.Pdr) + RELIEF_WEIGHT * _load_mw(m, m.Pshed)
    _replace_objective(m, reduction + 1e-4 * _disturbance(m))
    if not solve(m):
        return [], [{"id": "last_resort", "label": "Load reduction",
                     "reason": "not even reducing loads clears the violations (the limits it breaks are not "
                               "ones less load relieves, e.g. a high voltage)"}]
    m.least_reduction = Constraint(expr=reduction <= value(reduction) * 1.001 + 1e-3)
    _replace_objective(m, _disturbance(m))
    if solve(m):
        found.append((capture(m, "last_resort_least"), None))
    m2 = model.clone()
    if restart:
        restart(m2)
    free_loads(m2, demand_response=True, relief=True)
    cost = add_cost(m2, prices, pext0, qext0)
    _replace_objective(m2, cost + 1e-3 * _disturbance(m2))
    if solve(m2):
        option = capture(m2, "last_resort_cheapest")
        if not any(_same(option, other) for other, _ in found):
            found.append((option, None))
        elif found:
            missing.append({"id": "last_resort_cheapest", "label": STRATEGIES["last_resort_cheapest"][0],
                            "reason": "the same dispatch as Least load reduction", "same_as": "last_resort_least"})
    return [(o, action_cost(o, prices, pext0, qext0)) for o, _ in found], missing


# ---------------------------------------------------------------------------
# Choosing what to show
# ---------------------------------------------------------------------------

def _same(a: Option, b: Option) -> bool:
    units = set(a.p) | set(b.p)
    loads = [(a.dr or {}, b.dr or {}), (a.relief or {}, b.relief or {})]
    return (all(abs(a.p.get(g, 0.0) - b.p.get(g, 0.0)) < DUPLICATE_MW for g in units)
            and all(abs(x.get(k, 0.0) - y.get(k, 0.0)) < DUPLICATE_MW for x, y in loads for k in set(x) | set(y)))


def _key(option: Option, cost: dict) -> tuple:
    moved = option.moved
    return (len(option.instructions), sum(abs(v) for v in moved.values()), cost["total"],
            sum(option.pext.values()), option.max_loading_pct if option.max_loading_pct is not None else 1e9,
            option.load_reduction)


def _distance(a: Option, b: Option) -> float:
    """MW between two dispatches, unit by unit."""
    return sum(abs(a.p.get(g, 0.0) - b.p.get(g, 0.0)) for g in set(a.p) | set(b.p))


ANCHORS = ("least_disturbance", "cheapest")       # kept whenever they differ from the rest


def _best_at_something(option: Option, cost: dict, pool: list) -> bool:
    """Whether `option` beats every other in `pool` on one count of `_key`."""
    mine = _key(option, cost)
    others = [_key(o, c) for o, c in pool if o is not option]
    return any(all(mine[i] < k[i] - 1e-6 for k in others) for i in range(len(mine)))


def select(options: list[tuple[Option, dict]]) -> tuple[list[tuple[Option, dict]], list[dict]]:
    """Drop near-duplicates and options no better than another on any count
    (units called, MW moved, cost, import, highest loading); least_disturbance
    always stays. Beyond MAX_OPTIONS, the option closest to another one shown
    goes — never an anchor, and one that is best of all on some count (the
    fewest units called, MW moved, cost, import or highest loading) last. Returns (kept, dropped with the reason)
    — nothing is dropped without one."""
    kept, dropped = [], []
    for option, cost in options:
        twin = next((k for k, _ in kept if _same(k, option)), None)
        if twin is not None:
            dropped.append({"id": option.id, "label": option.label, "reason": f"the same dispatch as {twin.label}",
                            "same_as": twin.id})
            continue
        kept.append((option, cost))
    final = []
    for option, cost in kept:
        mine = _key(option, cost)
        better = next((o for o, c in kept if o is not option and all(x <= y + 1e-6 for x, y in zip(_key(o, c), mine))
                       and any(x < y - 1e-6 for x, y in zip(_key(o, c), mine))), None)
        if better is not None and option.id != "least_disturbance":
            dropped.append({"id": option.id, "label": option.label,
                            "reason": f"{better.label} is at least as good on every count"})
            continue
        final.append((option, cost))
    while len(final) > MAX_OPTIONS:
        free = [(o, c) for o, c in final if o.id not in ANCHORS]
        plain = [(o, c) for o, c in free if not _best_at_something(o, c, final)] or free
        closest = min(((o, other) for o, _ in plain for other, _ in final if other is not o),
                      key=lambda pair: _distance(*pair))
        option, other = closest
        final = [(o, c) for o, c in final if o is not option]
        dropped.append({"id": option.id, "label": option.label,
                        "reason": f"at most {MAX_OPTIONS} options are shown, and it is the closest to {other.label} "
                                  f"({_distance(option, other):.1f} MW apart)"})
    return final, dropped


def near_optimal_budget(cheapest_total: float, pct: float) -> float:
    """The most an alternative may cost: ε above the cheapest, at least the floor."""
    return cheapest_total + max(abs(cheapest_total) * pct / 100.0, NEAR_OPTIMAL_FLOOR_EUR_H)


def _exchange_before(prices: Prices, pext0: dict, qext0: dict) -> float:
    """The constant `add_cost` carries and `action_cost` subtracts: the
    reactive-exchange charge of the schedule itself."""
    total = 0.0
    for e in pext0:
        band = _price(prices.band, e) * abs(pext0.get(e, 0.0))
        total += _price(prices.penalty, e) * max(abs(qext0.get(e, 0.0)) - band, 0.0)
    return total


def _bounded(model, prices: Prices, pext0: dict, qext0: dict, budget: float):
    """A clone of `model` whose action may cost at most `budget`."""
    m = model.clone()
    cost = add_cost(m, prices, pext0, qext0)
    m.near_optimal_budget = Constraint(expr=cost <= budget + _exchange_before(prices, pext0, qext0))
    return m


def _free_units(m) -> list:
    return [g for g in m.G if not m.Pg_new[g].fixed]


def generate_near_optimal(model, prices: Prices, pext0: dict, qext0: dict, pct: float = NEAR_OPTIMAL_PCT,
                          methods: tuple = NEAR_OPTIMAL_METHODS, count: int = MAX_ALTERNATIVES):
    """The cheapest option and up to `count` near-optimal alternatives, each
    costing at most `near_optimal_budget`, picked farthest-first from a pool
    the `methods` fill (see the module's doc). Returns (options with their
    cost, what was not found and why, the cheapest cost, the budget, every
    unit's near-optimal range — [] unless unit_ranges ran)."""
    first = capture(model, "least_disturbance")
    best = cheapest(model, first, prices, pext0, qext0)
    if best is None:
        return [], [{"id": "near_optimal", "label": "Near-optimal alternatives",
                     "reason": "the cheapest option has no feasible dispatch to start from"}], None, None, []
    best_cost = action_cost(best, prices, pext0, qext0)
    budget = near_optimal_budget(best_cost["total"], pct)
    pool: list[tuple[Option, str]] = []          # (dispatch, how it was found)
    ranges: list[dict] = []

    def add(m, found_by: str) -> Option | None:
        if not solve(m):
            return None
        option = capture(m, "near_optimal_1")
        pool.append((option, found_by))
        return option

    if "hsj" in methods:
        used = set(best.moved)
        for k in range(HSJ_ROUNDS):
            m = _bounded(model, prices, pext0, qext0, budget)
            weight = {g: USED_WEIGHT if g in used else 1.0 for g in m.G}
            _replace_objective(m, sum(weight[g] * (m.dPp[g] + m.dPm[g]) for g in m.G) + 1e-3 * _disturbance(m))
            option = add(m, "away from the units earlier alternatives change (Hop-Skip-Jump)")
            if option is None:
                break
            used |= set(option.moved)
    if "unit_ranges" in methods:
        for g in _free_units(model):
            low = high = None
            for sense, word in ((1.0, "low"), (-1.0, "high")):
                m = _bounded(model, prices, pext0, qext0, budget)
                _replace_objective(m, sense * m.Pg_new[g] + 1e-4 * _disturbance(m))
                option = add(m, f"{g} as {word} as it can be within the margin")
                if option is not None:
                    if word == "low":
                        low = option.p[g]
                    else:
                        high = option.p[g]
            if low is not None and high is not None:
                ranges.append({"unit": g, "min_mw": round(low, 3), "max_mw": round(high, 3),
                               "cheapest_mw": round(best.p[g], 3), "schedule_mw": round(best.p0[g], 3)})
        import re
        ranges.sort(key=lambda r: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", r["unit"])])
    if "objectives" in methods:
        bounded = _bounded(model, prices, pext0, qext0, budget)
        for sid, word in (("fewest_units", "the fewest units"), ("least_import", "the least import"),
                          ("most_headroom", "the most headroom")):
            option = STRATEGY_FUNCTIONS[sid](bounded, best, prices, pext0, qext0)
            if option is not None:
                pool.append((option, f"{word} within the margin"))
        m = _bounded(model, prices, pext0, qext0, budget)
        _replace_objective(m, _disturbance(m))
        add(m, "the least disturbance within the margin")
    if "random" in methods:
        import random
        rng = random.Random(7)                   # seeded: the same question, the same alternatives
        units = _free_units(model)
        for k in range(RANDOM_DIRECTIONS if units else 0):
            direction = {g: rng.uniform(-1.0, 1.0) for g in units}
            m = _bounded(model, prices, pext0, qext0, budget)
            _replace_objective(m, sum(direction[g] * m.Pg_new[g] for g in units) + 1e-4 * _disturbance(m))
            add(m, f"a random direction ({k + 1})")

    # Farthest first: each pick the most different from everything shown.
    shown: list[tuple[Option, str]] = [(best, "")]
    candidates = [(o, why) for o, why in pool if not _same(o, best)]
    while candidates and len(shown) <= count:
        def gap(entry):
            return min(_distance(entry[0], other) for other, _ in shown)
        pick = max(candidates, key=gap)
        if gap(pick) < DIVERSE_MW:
            break
        shown.append(pick)
        candidates = [c for c in candidates if c is not pick]
    found = [(best, best_cost)]
    for k, (option, why) in enumerate(shown[1:], start=1):
        # `how` says how it was found ("Gen_bus1 as low as it can be within the margin").
        option = option._replace(id=f"near_optimal_{k}", label=f"Near-optimal {k}", how=why)
        found.append((option, action_cost(option, prices, pext0, qext0)))
    missing = []
    if len(found) == 1:
        missing.append({"id": "near_optimal", "label": "Near-optimal alternatives",
                        "reason": "no other dispatch within the margin changes the units differently"})
    elif len(found) - 1 < count:
        missing.append({"id": "near_optimal", "label": "Near-optimal alternatives",
                        "reason": f"{len(found) - 1} found: no further dispatch within the margin differs by "
                                  f"{DIVERSE_MW:g} MW or more from those shown"})
    return found, missing, best_cost["total"], round(budget, 2), ranges


def generate(model, prices: Prices, pext0: dict, qext0: dict
             ) -> tuple[list[tuple[Option, dict]], list[dict], float | None]:
    """Every strategy on the solved least-disturbance `model` (which stays as
    it is). Returns (options with their cost, strategies not offered and why,
    the cheapest strategy's cost — the reference every option's gap is measured
    from, kept even when that option is dropped as a duplicate)."""
    first = capture(model, "least_disturbance")
    found = [(first, action_cost(first, prices, pext0, qext0))]
    unavailable = []
    for sid, fn in STRATEGY_FUNCTIONS.items():
        option = fn(model, first, prices, pext0, qext0)
        if sid == "demand_response":
            if option is None and not hasattr(model, "LFlex"):
                continue                                  # a study that cannot reduce loads: nothing to say
            if option is not None and not option.dr:
                unavailable.append({"id": sid, "label": STRATEGIES[sid][0],
                                    "reason": "demand response would not be used at its price: redispatching the "
                                              "units is cheaper"})
                continue
        if option is None:
            reason = ("no feasible dispatch" if sid != "least_import" or sum(first.pext.values()) > MOVE_MW
                      else "the grid is not importing")
            if sid == "fewest_units" and len(first.instructions) <= 1:
                reason = "the least-disturbance option already calls one unit"
            # Named: "Without the main unit" was read as the unit the operator
            # had ruled out (seen live), when it was another one.
            main = max(first.moved, key=lambda g: abs(first.moved[g])) if first.moved else None
            unavailable.append({"id": sid, "label": STRATEGIES[sid][0].format(unit=main or "the main unit"),
                                "reason": reason})
            continue
        found.append((option, action_cost(option, prices, pext0, qext0)))
    cheapest_total = next((c["total"] for o, c in found if o.id == "cheapest"), None)
    kept, dropped = select(found)
    return kept, unavailable + dropped, cheapest_total


# ---------------------------------------------------------------------------
# Indicators: the flexibility a moment needs, and how far the units can go
# ---------------------------------------------------------------------------

def least_moved_mw(model) -> float | None:
    """The fewest MW of active power, summed over the units, that bring the
    schedule within every limit the model keeps: the flexibility the moment
    requires. A MW up and a MW down count alike (an L1 minimum, unlike the
    least-disturbance option's squares). None when redispatching the units
    cannot."""
    m = model.clone()
    free = _free_units(m)
    if not free:
        return 0.0
    for obj in list(m.component_objects(Objective, active=True)):
        obj.deactivate()
    moved = {g: float(value(m.Pg_new[g])) - float(value(m.Pg_base[g])) for g in free}
    m.kpi_up = Var(free, within=NonNegativeReals, initialize={g: max(moved[g], 0.0) for g in free})
    m.kpi_down = Var(free, within=NonNegativeReals, initialize={g: max(-moved[g], 0.0) for g in free})
    m.kpi_split = Constraint(free, rule=lambda m, g: m.Pg_new[g] - m.Pg_base[g] == m.kpi_up[g] - m.kpi_down[g])
    # A small share of the squares: among dispatches moving as few MW, the one spread least.
    m.kpi_obj = Objective(expr=sum(m.kpi_up[g] + m.kpi_down[g] for g in free) + 1e-4 * _disturbance(m),
                          sense=minimize)
    if not solve(m):
        return None
    return sum(abs(float(value(m.Pg_new[g])) - float(value(m.Pg_base[g]))) for g in free)


def deliverable(model, dispatch: dict) -> dict | None:
    """How far the units can move from `dispatch` (unit -> MW) within every
    limit the model keeps, reactive power and voltage set-points free to
    follow: each unit alone (the others held at `dispatch`) and all together.
    {"units": {unit: {"up_mw", "down_mw"} or None}, "up_mw", "down_mw"} —
    None in a range the solver found no dispatch for (the dispatch itself
    outside the limits); None overall when no range was found. Held units
    (fixed in the model) are not among them."""
    m = model.clone()
    free = [g for g in _free_units(m) if g in dispatch]
    if not free:
        return None
    # A power flow's output can sit a hair beyond a unit's bound (35.0000003 of
    # 35 MW): held there, every range with that unit held would be infeasible.
    dispatch = {g: min(max(float(dispatch[g]), m.Pg_new[g].lb if m.Pg_new[g].lb is not None else -math.inf),
                       m.Pg_new[g].ub if m.Pg_new[g].ub is not None else math.inf) for g in free}
    for obj in list(m.component_objects(Objective, active=True)):
        obj.deactivate()
    m.kpi_weight = Param(free, mutable=True, initialize=0.0)
    m.kpi_obj = Objective(expr=sum(m.kpi_weight[g] * m.Pg_new[g] for g in free), sense=maximize)

    def furthest(weights: dict, moving: list) -> float | None:
        """The most of Σ weight·P with only `moving` free (the rest at the dispatch)."""
        for g in free:
            m.kpi_weight[g] = weights.get(g, 0.0)
            m.Pg_new[g].value = dispatch[g]
            if g in moving:
                m.Pg_new[g].unfix()
            else:
                m.Pg_new[g].fix(dispatch[g])
        if not solve(m):
            return None
        return sum(weights.get(g, 0.0) * float(value(m.Pg_new[g])) for g in moving)

    units = {}
    for g in free:
        hi, lo = furthest({g: 1.0}, [g]), furthest({g: -1.0}, [g])
        units[g] = (None if hi is None or lo is None else
                    {"up_mw": max(hi - dispatch[g], 0.0), "down_mw": max(dispatch[g] - (-lo), 0.0)})
    total = sum(dispatch[g] for g in free)
    hi = furthest({g: 1.0 for g in free}, free)
    lo = furthest({g: -1.0 for g in free}, free)
    if hi is None and lo is None and all(u is None for u in units.values()):
        return None
    return {"units": units,
            "up_mw": max(hi - total, 0.0) if hi is not None else None,
            "down_mw": max(total + lo, 0.0) if lo is not None else None}

