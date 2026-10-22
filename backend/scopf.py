"""
scopf.py — dispatches that stay secure when a branch is lost.

The flexibility OPF meets every limit with everything in service. A
security-constrained OPF (SCOPF) also looks at a set of outages: one AC
power-flow state per outage, all tied to the same dispatch.

  preventive  the dispatch alone keeps every studied outage within limits:
              nothing has to be done when one happens. Every unit keeps its
              active power after the outage; a unit that holds a voltage keeps
              its set-point (its reactive output follows the grid), any other
              unit its reactive output.
  corrective  after each studied outage a corrective action restores the
              limits: each unit may move its active power by at most its reach
              (a share of its capacity — what it can deliver soon after) and
              change its voltage set-point (within the voltage band). The
              dispatch is chosen so that such an action exists for every outage
              studied; the action is part of the answer, as few units and as
              small moves as it can be.

Each state is the OPF model every other study builds
(`flex_engine.optimization_model_base`), one Pyomo block per state, linked —
so the constraints are the validated ones, not a second copy. Limits after an
outage are soft: a slack per limit, weighted far above everything else, so the
answer always exists and says what no dispatch can secure. The loading allowed
after an outage is its own setting — by default the network's limit, the one
every N-1 screen uses (100 % of the rating on the bundled networks), not the
OPF's 90 % margin of normal operation.

Pure model logic: no app state, no HTTP. The orchestration (which outages,
building the states, checking the result by power flow) is in main_backend.
"""

from __future__ import annotations

import math

from pyomo.environ import (ConcreteModel, Constraint, Expression, NonNegativeReals, Objective, Var, minimize, value)

PREVENTIVE, CORRECTIVE = "preventive", "corrective"
VOLTAGE_WEIGHT = 20.0       # 0.01 p.u. of voltage weighed like ~10 % of loading² excess
# A corrective action as short as it can be: an absolute-value cost (few units
# move, the rest exactly not) and a small quadratic one (a unique answer).
MOVE_L1, MOVE_WEIGHT = 1e-2, 1e-3            # per MW, per MW²
SETPOINT_L1, SETPOINT_WEIGHT = 2.0, 10.0     # per p.u., per p.u.²: 0.01 p.u. weighs like 2 MW
RESIDUAL_TOL = 1e-4         # slack below this is solver tolerance, not a violation
SECURE_TOL = 1e-6           # stage 2 keeps stage 1's security to within this (absolute) ...
SECURE_REL = 1e-3           # ... and this share


def _deactivate_objectives(m) -> None:
    for obj in list(m.component_objects(Objective, active=True)):
        obj.deactivate()


def release(m, keep_fixed: set) -> None:
    """Unfix what the engine's first stage fixed (every unit at its schedule,
    when that schedule was already feasible): the SCOPF chooses them. Units
    in `keep_fixed` (the operator's held units) stay as they are."""
    for g in m.G:
        if g in keep_fixed:
            continue
        if m.Pg_new[g].fixed and value(m.Pg_min[g]) < value(m.Pg_max[g]):
            m.Pg_new[g].unfix()
        if m.Qg_new[g].fixed:
            m.Qg_new[g].unfix()


def _rating_sq(m, i, j):
    for a, b in ((i, j), (j, i)):
        if (a, b) in m.I_from_max and float(value(m.Ibase_from_kA[a, b])):
            return (float(value(m.I_from_max[a, b])) / float(value(m.Ibase_from_kA[a, b]))) ** 2
    return None


def _setpoint_buses(m) -> set:
    """Buses whose voltage a unit holds: there the voltage is a set-point."""
    return {int(value(m.gen_bus[g])) for g in getattr(m, "CondSet", [])}


def soften(m, emergency: float, vm_lower: float, vm_upper: float) -> None:
    """A post-outage state: its limits become soft. A branch's current may
    exceed `emergency` × its rating by a slack (as a share of rating²); a
    bus's voltage may leave the band by a slack (p.u.). Power balance, unit
    limits, the slack bus and the voltage set-points (a control, kept in the
    band) stay as they are."""
    rows = []
    for name, direction in (("branch_current_from_limit", "from"), ("branch_current_to_limit", "to")):
        c = getattr(m, name, None)
        if c is None:
            continue
        for (i, j) in c:
            rating = _rating_sq(m, i, j) if direction == "from" else _rating_sq(m, j, i)
            if rating:
                rows.append((c[i, j].body, rating, (int(i), int(j)) if direction == "from" else (int(j), int(i))))
        c.deactivate()
    m.s_branch = Var(range(len(rows)), within=NonNegativeReals, initialize=0.0)
    m.soft_branch = Constraint(range(len(rows)), rule=lambda m, k: rows[k][0] / rows[k][1] <= emergency ** 2 + m.s_branch[k])
    m._soft_rows = [r[2] for r in rows]                    # (from, to) per slack, for the report
    m._soft_rating = [r[1] for r in rows]
    held = _setpoint_buses(m)
    free = [b for b in m.B if not m.V[b].fixed and b not in held]
    for b in free:
        m.V[b].setlb(0.5)
        m.V[b].setub(1.5)
    m.s_vlo = Var(free, within=NonNegativeReals, initialize=0.0)
    m.s_vhi = Var(free, within=NonNegativeReals, initialize=0.0)
    m.soft_vlo = Constraint(free, rule=lambda m, b: m.V[b] >= vm_lower - m.s_vlo[b])
    m.soft_vhi = Constraint(free, rule=lambda m, b: m.V[b] <= vm_upper + m.s_vhi[b])
    m._band = (vm_lower, vm_upper)
    m._emergency = emergency


def _slacks(m):
    return (sum(m.s_branch[k] for k in m.s_branch) + VOLTAGE_WEIGHT * (
        sum(m.s_vlo[b] for b in m.s_vlo) + sum(m.s_vhi[b] for b in m.s_vhi)))


def build(base, states: dict, mode: str, reach: dict, lambda_p: float = 0.01, lambda_q: float = 0.001):
    """The SCOPF: `base` (the model with everything in service, hard limits)
    and one softened block per outage in `states` ({outage name: model}),
    linked by `mode`. `reach`: unit -> MW it may move after an outage
    (corrective). Returns the parent model; its blocks are `base` and `c0`…"""
    top = ConcreteModel()
    top.base = base
    _deactivate_objectives(base)
    names = list(states)
    vc = set(getattr(base, "CondSet", []))
    for k, name in enumerate(names):
        m = states[name]
        setattr(top, f"c{k}", m)
        _deactivate_objectives(m)
        units = [g for g in base.G if g in set(m.G)]
        if mode == PREVENTIVE:
            setattr(top, f"link_p{k}", Constraint(units, rule=lambda t, g, m=m: m.Pg_new[g] == t.base.Pg_new[g]))
            held_v = [g for g in units if g in vc and int(value(base.gen_bus[g])) in set(m.B)
                      and not base.V[int(value(base.gen_bus[g]))].fixed]
            setattr(top, f"link_v{k}", Constraint(held_v, rule=lambda t, g, m=m: (
                m.V[int(value(t.base.gen_bus[g]))] == t.base.V[int(value(t.base.gen_bus[g]))])))
            fixed_q = [g for g in units if g not in vc]
            setattr(top, f"link_q{k}", Constraint(fixed_q, rule=lambda t, g, m=m: m.Qg_new[g] == t.base.Qg_new[g]))
        else:
            # The move after the outage, split into up and down (each within
            # the unit's reach) so that its size can be costed.
            setattr(top, f"up{k}", Var(units, bounds=lambda t, g: (0.0, reach.get(g, 0.0)), initialize=0.0))
            setattr(top, f"down{k}", Var(units, bounds=lambda t, g: (0.0, reach.get(g, 0.0)), initialize=0.0))
            setattr(top, f"move{k}", Constraint(units, rule=lambda t, g, m=m, k=k: (
                m.Pg_new[g] - t.base.Pg_new[g] == getattr(t, f"up{k}")[g] - getattr(t, f"down{k}")[g])))
            buses = sorted({int(value(base.gen_bus[g])) for g in units if g in vc} & set(m.B))
            buses = [b for b in buses if not base.V[b].fixed and not m.V[b].fixed]
            setattr(top, f"vup{k}", Var(buses, within=NonNegativeReals, initialize=0.0))
            setattr(top, f"vdown{k}", Var(buses, within=NonNegativeReals, initialize=0.0))
            setattr(top, f"vmove{k}", Constraint(buses, rule=lambda t, b, m=m, k=k: (
                m.V[b] - t.base.V[b] == getattr(t, f"vup{k}")[b] - getattr(t, f"vdown{k}")[b])))
    blocks = [getattr(top, f"c{k}") for k in range(len(names))]
    disturbance = (lambda_p * sum((base.Pg_new[g] - base.Pg_base[g]) ** 2 for g in base.G)
                   + lambda_q * sum((base.Qg_new[g] - base.Qg_base[g]) ** 2 for g in base.G))
    moves = 0.0
    if mode == CORRECTIVE:
        for k in range(len(names)):
            up, down, vup, vdown = (getattr(top, f"{n}{k}") for n in ("up", "down", "vup", "vdown"))
            moves += sum(MOVE_L1 * (up[g] + down[g]) + MOVE_WEIGHT * (up[g] - down[g]) ** 2 for g in up)
            moves += sum(SETPOINT_L1 * (vup[b] + vdown[b]) + SETPOINT_WEIGHT * (vup[b] - vdown[b]) ** 2 for b in vup)
    # Two stages (`secure_first`): as secure as possible, then — keeping that —
    # the smallest change and the smallest corrective actions.
    top.violation = Expression(expr=sum(_slacks(m) for m in blocks))
    top.preference = Expression(expr=disturbance + moves)
    top.obj = Objective(expr=top.violation + 1e-6 * top.preference, sense=minimize)
    top._outages = names
    return top


def secure_first(top) -> None:
    """After stage 1 (least violation): keep that security, and prefer the
    smallest change to the schedule and the smallest corrective actions."""
    reached = max(value(top.violation), 0.0)
    top.keep_secure = Constraint(expr=top.violation <= reached * (1 + SECURE_REL) + SECURE_TOL)
    top.obj.deactivate()
    top.obj2 = Objective(expr=top.preference, sense=minimize)


def residuals(m, branch_name, bus_name) -> list[str]:
    """What the soft limits of one post-outage state still break, in words,
    worst first: "Bus_21 -> Bus_23 [L30] at 104.2 %", "Bus_23 at 1.063 p.u.".
    `branch_name(from_bus, to_bus)`; a branch is named once (its worse end)."""
    loading: dict[str, float] = {}
    for k in m.s_branch:
        if value(m.s_branch[k]) > RESIDUAL_TOL:
            name = branch_name(*m._soft_rows[k])
            pct = math.sqrt(m._emergency ** 2 + value(m.s_branch[k])) * 100
            loading[name] = max(loading.get(name, 0.0), pct)
    out = [f"{name} at {pct:.1f} %" for name, pct in sorted(loading.items(), key=lambda kv: -kv[1])]
    lo, hi = m._band
    volts = []
    for b in m.s_vlo:
        if value(m.s_vlo[b]) > RESIDUAL_TOL:
            volts.append((value(m.s_vlo[b]), f"{bus_name(b)} at {lo - value(m.s_vlo[b]):.3f} p.u."))
        if value(m.s_vhi[b]) > RESIDUAL_TOL:
            volts.append((value(m.s_vhi[b]), f"{bus_name(b)} at {hi + value(m.s_vhi[b]):.3f} p.u."))
    return out + [text for _, text in sorted(volts, key=lambda x: -x[0])]
