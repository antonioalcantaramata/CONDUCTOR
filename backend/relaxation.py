"""
A convex relaxation of the OPF, for two things the local solver cannot say:
how far the cheapest option can be from the best possible (a lower bound on
its cost), and, when IPOPT finds no dispatch, whether none exists (a proof) or
one was missed (a second start from the relaxation's solution; `second_look`).

The AC optimal power flow is not convex, and IPOPT finds a local optimum:
nothing in the cheapest option itself says whether a cheaper dispatch exists.
Here the same model — the same units, prices, limits, exchange cap and units
held — has its AC power-flow equations replaced by their second-order cone
(SOC) relaxation (Jabr 2006). With

    c_ii = V_i²,   c_ij = V_i V_j cos(θ_i − θ_j),   s_ij = V_i V_j sin(θ_i − θ_j)

the power balances and the branch-current limits become linear in (c, s), and
the one relation between them that is not, c_ij² + s_ij² = c_ii c_jj, is
relaxed to c_ij² + s_ij² ≤ c_ii c_jj: a convex set that holds every AC
solution. Every other constraint of the model is linear (the power-factor
cone Q² ≤ tan²φ P² is |Q| ≤ tanφ·P exactly, P being non-negative), so the
relaxed problem is convex and its optimum — global — is a cost below which no
AC dispatch meeting the same limits exists.

That SOC relaxation keeps voltage magnitudes but not that the angles around
a loop of a meshed network add up, and on a congested line it is loose: on
the 30-bus demo at its evening peak it lets flow move off the overloaded line
for free, and bounds the cheapest option (26.48 €/h) at −150 €/h. The
semidefinite relaxation (Lavaei & Low 2012) keeps that: W = V Vᴴ ⪰ 0, here on
the maximal cliques of a chordal completion of the network (as strong as on
the whole matrix, in small blocks), solved by Clarabel through cvxpy. It
bounds the same moment at 22.48 €/h. Where cvxpy is not installed, or the
network has more than `SDP_MAX_BUSES` buses, the SOC bound is given instead,
and says so.

The bound is only as good as the relaxation is tight. A small gap certifies
that the cheapest option is (nearly) the best possible; a large gap says only
that it may not be — at most the gap could be saved.

Refused rather than guessed: a model with a constraint on voltages this module
does not know how to relax, or with anything else nonlinear left, gets no
bound and the reason.
"""

from __future__ import annotations

import math

from pyomo.core.expr.visitor import identify_variables
from pyomo.environ import Constraint, Objective, Set, SolverFactory, Var, value
from pyomo.opt import TerminationCondition

import flex_engine as fe
import flex_options as fo

# The model's own AC constraints, replaced here; anything else on V or θ is refused.
_AC_CONSTRAINTS = ("active_balance", "reactive_balance", "branch_current_from_limit", "branch_current_to_limit")
# Clarabel settings tried in turn until one reaches an optimal status.
_CLARABEL_TRIES = ({}, {"equilibrate_max_iter": 50, "max_iter": 400},
                   {"iterative_refinement_reltol": 1e-15, "iterative_refinement_max_iter": 50, "max_iter": 400})
# Beyond this many buses the semidefinite relaxation is built only as cones
# (slow to set up in cvxpy past it): the SOC relaxation, looser, is used.
SDP_MAX_BUSES = 300
# A bound above the cheapest option by more than this (€/h) is not a bound —
# a modelling mismatch — and is not reported.
INCONSISTENT_EUR_H = 0.05


def _fixed_or(var, default):
    return value(var) if var.fixed else default


def soc_relax(m) -> str | None:
    """Replace the AC physics of a `flex_engine` OPF model by its SOC
    relaxation, in place. Returns None, or why it cannot."""
    angles = {id(v) for v in m.V.values()} | {id(v) for v in m.theta.values()}
    for con in m.component_objects(Constraint, active=True):
        if con.local_name in _AC_CONSTRAINTS or con.local_name == "PF_cone":
            continue
        for c in con.values():
            if c.active and any(id(v) in angles for v in identify_variables(c.body, include_fixed=False)):
                return f"the model has a constraint on voltages ({con.local_name}) with no relaxation here"
    for obj in m.component_data_objects(ctype=Objective, active=True):
        if any(id(v) in angles for v in identify_variables(obj.expr, include_fixed=False)):
            return "the objective depends on voltages"

    # c_ii: V_i² within the bus's bounds (a held set-point: its value).
    def c_bounds(m, b):
        v = m.V[b]
        if v.fixed:
            return (value(v) ** 2, value(v) ** 2)
        return ((v.lb or 0.0) ** 2, (v.ub if v.ub is not None else 2.0) ** 2)
    m.rx_c = Var(m.B, bounds=c_bounds, initialize=lambda m, b: value(m.V[b]) ** 2)
    pairs = sorted({tuple(sorted((int(i), int(j)))) for i, j in m.Branches})
    m.rx_pairs = Set(initialize=pairs, dimen=2)

    def vmax(b):
        return math.sqrt(m.rx_c[b].ub)

    def pair_bounds(m, i, j):
        r = vmax(i) * vmax(j)
        return (-r, r)
    m.rx_cc = Var(m.rx_pairs, bounds=pair_bounds, initialize=lambda m, i, j: (
        value(m.V[i]) * value(m.V[j]) * math.cos(value(m.theta[i]) - value(m.theta[j]))))
    m.rx_ss = Var(m.rx_pairs, bounds=pair_bounds, initialize=lambda m, i, j: (
        value(m.V[i]) * value(m.V[j]) * math.sin(value(m.theta[i]) - value(m.theta[j]))))
    m.rx_soc = Constraint(m.rx_pairs, rule=lambda m, i, j: (
        m.rx_cc[i, j] ** 2 + m.rx_ss[i, j] ** 2 <= m.rx_c[i] * m.rx_c[j]))

    def cs(i, j):
        """(c_ij, s_ij) for the branch end (i, j): s changes sign with the direction."""
        a, b = (i, j) if i < j else (j, i)
        return m.rx_cc[a, b], (m.rx_ss[a, b] if i < j else -m.rx_ss[a, b])

    def y(name, i, j):
        return value(getattr(m, name)[i, j])

    def flows(b):
        p = q = 0.0
        for i, j in m.Branches:
            if i != b:
                continue
            c_ij, s_ij = cs(int(i), int(j))
            gff, bff = y("Yff_real_used", i, j), y("Yff_imag_used", i, j)
            gft, bft = y("Yft_real_used", i, j), y("Yft_imag_used", i, j)
            p = p + m.rx_c[i] * gff + gft * c_ij + bft * s_ij
            q = q - m.rx_c[i] * bff - (bft * c_ij - gft * s_ij)
        return p, q

    # The balances of `fe.active_power_balance_rule` and
    # `fe.reactive_power_balance_rule`, with V² → c_ii and the flows in (c, s).
    def active_rule(m, b):
        if b not in m.active_balance or not m.active_balance[b].active:
            return Constraint.Skip
        injected = (sum(m.Pg_new[g] for g in m.G if m.gen_bus[g] == b)
                    + sum(m.Pext[e] for e in m.Ext if m.ext_bus[e] == b)
                    - sum(m.Pl[l] * fe._served(m, l) for l in m.L if m.load_bus[l] == b)
                    - sum(m.Ps_nom[s] * m.rx_c[b] for s in m.S if m.shunt_bus[s] == b))
        return injected - flows(b)[0] * m.Sbase == 0

    def reactive_rule(m, b):
        if b not in m.reactive_balance or not m.reactive_balance[b].active:
            return Constraint.Skip
        injected = (sum(m.Qg_new[g] for g in m.G if m.gen_bus[g] == b)
                    + sum(m.Qext[e] for e in m.Ext if m.ext_bus[e] == b)
                    - sum(m.Ql[l] * fe._served(m, l) for l in m.L if m.load_bus[l] == b)
                    - sum(m.Qs_nom[s] * m.rx_c[b] for s in m.S if m.shunt_bus[s] == b))
        return injected - flows(b)[1] * m.Sbase == 0

    # |I_ij|² = |Y_ff|² c_ii + |Y_ft|² c_jj + 2 Re(Y_ff conj(Y_ft)(c_ij + j s_ij)):
    # linear, so each limit is kept exactly (its right-hand side as the model has it).
    def current_rule(m, i, j):
        con = m.branch_current_from_limit[i, j] if (i, j) in m.branch_current_from_limit else None
        if con is None or not con.active or con.upper is None:
            return Constraint.Skip
        gf, bf = y("Yff_real_used", i, j), y("Yff_imag_used", i, j)
        gt, bt = y("Yft_real_used", i, j), y("Yft_imag_used", i, j)
        c_ij, s_ij = cs(int(i), int(j))
        current_sq = ((gf ** 2 + bf ** 2) * m.rx_c[i] + (gt ** 2 + bt ** 2) * m.rx_c[j]
                      + 2 * ((gf * gt + bf * bt) * c_ij - (bf * gt - gf * bt) * s_ij))
        return current_sq <= value(con.upper)

    m.rx_active_balance = Constraint(m.B, rule=active_rule)
    m.rx_reactive_balance = Constraint(m.B, rule=reactive_rule)
    m.rx_current = Constraint(m.Branches, rule=current_rule)
    for name in _AC_CONSTRAINTS:
        if hasattr(m, name):
            getattr(m, name).deactivate()

    # Q² ≤ tan²φ P² is |Q| ≤ tanφ P for P ≥ 0. A unit allowed below 0 MW
    # keeps no reactive limit here: dropping a constraint only weakens the bound.
    if hasattr(m, "PF_cone"):
        tan_phi = math.sqrt(value(m.tan_phi_sq))
        units = [g for g in m.PF_set if _fixed_or(m.Pg_new[g], m.Pg_new[g].lb if m.Pg_new[g].lb is not None
                                                 else -1.0) >= 0.0]
        m.rx_pf_up = Constraint(units, rule=lambda m, g: m.Qg_new[g] <= tan_phi * m.Pg_new[g])
        m.rx_pf_down = Constraint(units, rule=lambda m, g: -m.Qg_new[g] <= tan_phi * m.Pg_new[g])
        m.PF_cone.deactivate()

    # What is left must be convex: the cones above, and nothing else nonlinear.
    for con in m.component_objects(Constraint, active=True):
        if con.local_name == "rx_soc":
            continue
        for c in con.values():
            if c.active and (c.body.polynomial_degree() is None or c.body.polynomial_degree() > 1):
                return f"the model has a nonlinear constraint ({con.local_name}) the relaxation does not cover"
    return None


def _sdp_problem(m, cone: str = "chordal"):
    """The conic problem of the relaxed model `m` (after `soc_relax`) with its cones
    replaced by W ⪰ 0, W the Hermitian matrix of the voltage products
    (W_ii = c_ii, W_ij = c_ij + j s_ij for i < j). Every other constraint, the bounds and the
    objective are linear, and are carried over from the model as they are.

    The semidefinite relaxation (Lavaei & Low 2012) keeps what the SOC
    relaxation drops on a meshed network — that the angles around a loop add
    up — and is exact on many IEEE test cases. (value or None, why not)."""
    import cvxpy as cp
    import networkx as nx
    import numpy as np
    from pyomo.repn import generate_standard_repn
    from scipy import sparse

    columns: dict = {}

    def col(var) -> int:
        return columns.setdefault(id(var), (len(columns), var))[0]

    rows_eq, rows_in = [], []           # (coefficients {column: a}, constant) for a·x + k = 0 / ≤ 0
    for con in m.component_objects(Constraint, active=True):
        if con.local_name == "rx_soc":
            continue
        for c in con.values():
            if not c.active:
                continue
            repn = generate_standard_repn(c.body, compute_values=True)
            if not repn.is_linear():
                return None, f"the model has a nonlinear constraint ({con.local_name}) the relaxation does not cover"
            coefs: dict = {}
            for var, a in zip(repn.linear_vars, repn.linear_coefs):
                coefs[col(var)] = coefs.get(col(var), 0.0) + float(a)
            k = float(repn.constant)
            lo = None if c.lower is None else float(value(c.lower))
            hi = None if c.upper is None else float(value(c.upper))
            if lo is not None and hi is not None and abs(hi - lo) < 1e-12:
                rows_eq.append((coefs, k - lo))
                continue
            if hi is not None:
                rows_in.append((coefs, k - hi))
            if lo is not None:
                rows_in.append(({j: -a for j, a in coefs.items()}, lo - k))
    objective = next(m.component_data_objects(ctype=Objective, active=True))
    repn = generate_standard_repn(objective.expr, compute_values=True)
    if not repn.is_linear():
        return None, "the objective is not linear"
    goal = {}
    for var, a in zip(repn.linear_vars, repn.linear_coefs):
        goal[col(var)] = goal.get(col(var), 0.0) + float(a)
    buses = list(m.B)
    for b in buses:
        col(m.rx_c[b])
    for i, j in m.rx_pairs:
        col(m.rx_cc[i, j]), col(m.rx_ss[i, j])

    n = len(columns)
    x = cp.Variable(n)
    order = sorted(columns.values(), key=lambda t: t[0])

    def matrix(rows):
        # Each row divided by its largest coefficient: the balances are in MW
        # (Sbase × admittances of 10⁴ on a short line) beside costs of a few
        # €/MWh, and conic solvers stall on that spread.
        data, ri, ci, rhs = [], [], [], []
        for r, (coefs, k) in enumerate(rows):
            scale = max((abs(a) for a in coefs.values()), default=1.0) or 1.0
            for j, a in coefs.items():
                data.append(a / scale), ri.append(r), ci.append(j)
            rhs.append(-k / scale)
        return sparse.csr_matrix((data, (ri, ci)), shape=(len(rows), n)), np.array(rhs)

    constraints = []
    if rows_eq:
        a_eq, b_eq = matrix(rows_eq)
        constraints.append(a_eq @ x == b_eq)
    if rows_in:
        a_in, b_in = matrix(rows_in)
        constraints.append(a_in @ x <= b_in)
    # A variable whose bounds coincide (a condenser's P in [0, 0], a unit not
    # dispatchable) is fixed: as two inequalities it left the problem no
    # interior, and interior-point solvers fail on that.
    def pinned(v):
        return v.fixed or (v.lb is not None and v.ub is not None and v.ub - v.lb < 1e-9)
    fixed = [(j, float(value(v)) if v.fixed else float(v.lb)) for j, v in order if pinned(v)]
    lower = [(j, float(v.lb)) for j, v in order if not pinned(v) and v.lb is not None]
    upper = [(j, float(v.ub)) for j, v in order if not pinned(v) and v.ub is not None]
    for pairs, sense in ((fixed, "=="), (lower, ">="), (upper, "<=")):
        if pairs:
            idx, val = [p[0] for p in pairs], np.array([p[1] for p in pairs])
            constraints.append(x[idx] == val if sense == "==" else x[idx] >= val if sense == ">=" else x[idx] <= val)
    def entry(var) -> object:
        return x[columns[id(var)][0]]

    diag = {b: entry(m.rx_c[b]) for b in buses}
    pair = {(i, j): (entry(m.rx_cc[i, j]), entry(m.rx_ss[i, j])) for i, j in m.rx_pairs}
    parts = {"problem": None, "x": x, "columns": columns, "fill": ([], None), "cliques": []}
    if cone == "soc":                   # the cones of `soc_relax`, for comparison
        for (i, j), (cij, sij) in pair.items():
            constraints.append(cp.SOC(diag[i] + diag[j], cp.hstack([2 * cij, 2 * sij, diag[i] - diag[j]])))
    else:
        # W ⪰ 0 on every maximal clique of a chordal completion of the network:
        # as strong as W ⪰ 0 on the whole matrix (a partial matrix PSD on the
        # cliques of a chordal graph has a PSD completion), in small blocks.
        # One block of every bus pair left its many free entries to the
        # solver, which then failed on all but the smallest networks.
        graph = nx.Graph()
        graph.add_nodes_from(buses)
        graph.add_edges_from(pair)
        filled, _ = nx.complete_to_chordal_graph(graph)
        fill = sorted({tuple(sorted(e)) for e in filled.edges} - set(pair))
        if fill:
            z = cp.Variable((len(fill), 2))
            reach = np.array([math.sqrt(m.rx_c[i].ub * m.rx_c[j].ub) for i, j in fill])
            constraints += [cp.abs(z[:, 0]) <= reach, cp.abs(z[:, 1]) <= reach]
            pair.update({(i, j): (z[k, 0], z[k, 1]) for k, (i, j) in enumerate(fill)})
            parts["fill"] = (fill, z)
        for clique in nx.chordal_graph_cliques(filled):
            nodes = sorted(clique)
            k = len(nodes)
            if k == 1:
                continue
            if k == 2:
                (i, j), = [tuple(nodes)]
                cij, sij = pair[(i, j)]
                constraints.append(cp.SOC(diag[i] + diag[j], cp.hstack([2 * cij, 2 * sij, diag[i] - diag[j]])))
                continue
            # The Hermitian block C + jS as the real symmetric [[C, −S], [S, C]].
            block = cp.Variable((2 * k, 2 * k), symmetric=True)
            links = []
            for p, a in enumerate(nodes):
                links += [block[p, p] == diag[a], block[k + p, k + p] == diag[a], block[p, k + p] == 0]
                for q in range(p + 1, k):
                    cab, sab = pair[(a, nodes[q])]
                    links += [block[p, q] == cab, block[k + p, k + q] == cab,
                              block[k + p, q] == sab, block[p, k + q] == -sab]
            constraints += [block >> 0, *links]
            parts["cliques"].append((nodes, block))
    cost = np.zeros(n)
    for j, a in goal.items():
        cost[j] = a
    parts["problem"] = cp.Problem(cp.Minimize(cost @ x + float(repn.constant)), constraints)
    return parts


def _sdp_minimum(m) -> tuple[float | None, str | None]:
    """The minimum of `_sdp_problem(m)` by Clarabel: (value or None, why not).
    Only an optimal status counts; an inaccurate one is no bound."""
    import cvxpy as cp
    parts = _sdp_problem(m)
    if isinstance(parts, tuple):
        return None, parts[1]
    problem = parts["problem"]
    status = "not solved"
    # Clarabel stops "almost solved" on some moments; a few more iterations of
    # equilibration or refinement then reach a full solution.
    for settings in _CLARABEL_TRIES:
        try:
            problem.solve(solver="CLARABEL", **settings)
        except Exception as exc:  # noqa: BLE001 — try the next settings
            status = f"failed ({str(exc)[:80]})"
            continue
        status = problem.status
        if status == cp.OPTIMAL:
            return float(problem.value), None
        if status == cp.INFEASIBLE:
            return None, "the relaxation is infeasible"
    return None, f"the semidefinite relaxation ended {status}"


def _sdp_available() -> bool:
    try:
        import cvxpy as cp
    except ImportError:
        return False
    return "CLARABEL" in cp.installed_solvers()


def cost_lower_bound(model, prices: fo.Prices, pext0: dict, qext0: dict) -> dict:
    """The least cost (€/h, as `action_cost` counts it) of any dispatch of
    the units meeting the model's limits, from a convex relaxation of the
    cheapest option's own problem: semidefinite where cvxpy and a conic
    solver are installed, else second-order cone (IPOPT; looser on meshed
    networks). {"lower_bound_eur_per_h": float | None, "relaxation": "sdp" |
    "soc", "why": None | str}. `model` stays as it is."""
    m = model.clone()
    cost = fo.add_cost(m, prices, pext0, qext0)
    fo._replace_objective(m, cost)
    why = soc_relax(m)
    kind = "sdp" if _sdp_available() and len(m.B) <= SDP_MAX_BUSES else "soc"
    if why:
        return {"lower_bound_eur_per_h": None, "relaxation": kind, "why": why}
    before = fo._exchange_before(prices, pext0, qext0)
    if kind == "sdp":
        found, why = _sdp_minimum(m)
        return {"lower_bound_eur_per_h": None if found is None else found - before, "relaxation": kind, "why": why}
    try:
        res = SolverFactory("ipopt").solve(m, tee=False, load_solutions=False)
    except Exception as exc:  # noqa: BLE001 — no bound is an answer too
        return {"lower_bound_eur_per_h": None, "relaxation": kind, "why": f"the relaxation could not be solved ({exc})"}
    tc = res.solver.termination_condition
    if tc != TerminationCondition.optimal:
        return {"lower_bound_eur_per_h": None, "relaxation": kind, "why": f"the relaxation ended {tc}"}
    m.solutions.load_from(res)
    return {"lower_bound_eur_per_h": float(value(cost)) - before, "relaxation": kind, "why": None}


_RELAXATIONS = {
    "sdp": ("semidefinite relaxation (W = V Vᴴ ⪰ 0 on the cliques of a chordal completion of the network) "
            "of the cheapest option's own optimisation"),
    "soc": ("second-order cone relaxation of the cheapest option's own optimisation (looser than the "
            "semidefinite one on a meshed network)"),
}
# A gap this small (€/h) is read as none: the cheapest option is the best possible.
NO_GAP_EUR_H = 0.05


def optimality(bound: dict, cheapest_eur_per_h: float | None) -> dict | None:
    """The `optimality` block of an options result: the cheapest option's cost
    against the bound, the gap, and what they mean. None without a cheapest."""
    if cheapest_eur_per_h is None:
        return None
    lb = bound.get("lower_bound_eur_per_h")
    if lb is not None and lb > cheapest_eur_per_h + INCONSISTENT_EUR_H:
        lb, bound = None, {"why": f"the relaxation's bound ({lb:.2f} €/h) came out above the cheapest option, "
                                  "which a relaxation cannot do: not reported"}
    if lb is None:
        return {"cheapest_eur_per_h": cheapest_eur_per_h, "lower_bound_eur_per_h": None, "why": bound.get("why")}
    lb = math.floor(lb * 100.0) / 100.0             # rounded down: a bound, never above the true one
    gap = round(max(cheapest_eur_per_h - lb, 0.0), 2)
    return {
        "cheapest_eur_per_h": cheapest_eur_per_h,
        "lower_bound_eur_per_h": lb,
        "gap_eur_per_h": gap,
        # A share of a cost only: of a saving (a negative cost) it means nothing.
        **({"gap_pct": round(100.0 * gap / cheapest_eur_per_h, 2)} if cheapest_eur_per_h >= 1.0 else {}),
        "best_possible": gap <= NO_GAP_EUR_H,
        "relaxation": bound.get("relaxation"),
        "method": (_RELAXATIONS.get(bound.get("relaxation"), "convex relaxation of the cheapest option's own "
                                                             "optimisation")
                   + ": the same units, prices, limits, exchange cap and held units, loads not reduced; its AC "
                     "power-flow equations relaxed to a convex set that holds every AC solution, solved to its "
                     "global optimum"),
        "meaning": ("No dispatch of the units that meets these limits costs less than the lower bound, so no "
                    "other dispatch saves more than the gap against the cheapest option. A large gap says a "
                    "cheaper dispatch may exist (the relaxation can be loose), not that one does."),
    }


def sentence(block: dict | None) -> str:
    """The optimality block in one sentence for the result's message ('' without a bound)."""
    if not block or block.get("lower_bound_eur_per_h") is None:
        return ""
    lb, gap = block["lower_bound_eur_per_h"], block["gap_eur_per_h"]
    if block.get("best_possible"):
        return (f" No dispatch of the units meeting these limits costs less than {lb:.2f} €/h: the cheapest option "
                f"is the best possible (within {gap:.2f} €/h).")
    return (f" No dispatch of the units meeting these limits costs less than {lb:.2f} €/h, so no other dispatch "
            f"saves more than {gap:.2f} €/h against the cheapest option.")


# ---------------------------------------------------------------------------
# When IPOPT finds no dispatch: proof that none exists, or a second start
# ---------------------------------------------------------------------------

NO_DISPATCH = "no_dispatch_exists"            # the relaxation is infeasible: proved
FOUND = "found_from_relaxation"               # IPOPT, restarted from the relaxation's solution, found one
NOT_PROVEN = "not_proven"                     # neither


def _near_schedule(m) -> None:
    """Replace the objective by the MW moved from the schedule, Σ|ΔP| (linear,
    so the relaxation can minimise it): a relaxed point near the schedule is
    the better restart for a least-disturbance solve."""
    m.rx_up = Var(m.G, bounds=(0.0, None), initialize=0.0)
    m.rx_down = Var(m.G, bounds=(0.0, None), initialize=0.0)
    m.rx_split = Constraint(m.G, rule=lambda m, g: m.Pg_new[g] - m.Pg_base[g] == m.rx_up[g] - m.rx_down[g])
    fo._replace_objective(m, sum(m.rx_up[g] + m.rx_down[g] for g in m.G))


def _angles(m) -> dict:
    """Bus angles from the relaxed (c, s): along a spanning tree from each
    fixed angle (the slack), θ_j = θ_i − atan2(s_ij, c_ij)."""
    import networkx as nx
    graph = nx.Graph()
    graph.add_nodes_from(int(b) for b in m.B)
    graph.add_edges_from(m.rx_pairs)
    theta = {int(b): value(m.theta[b]) for b in m.B if m.theta[b].fixed}
    for root in list(theta):
        for a, b in nx.bfs_edges(graph, root):
            if b in theta:
                continue
            i, j = (a, b) if a < b else (b, a)
            step = math.atan2(value(m.rx_ss[i, j]), value(m.rx_cc[i, j]))      # θ_i − θ_j
            theta[b] = theta[a] - step if a == i else theta[a] + step
    return theta


def _restart_from(model, relaxed):
    """A clone of `model` (its own objective) started at the relaxed point —
    voltages √c_ii, angles from (c, s), every unit's and grid's output — and
    solved. The solved clone, or None."""
    target = model.clone()
    theta = _angles(relaxed)
    for b in target.B:
        if not target.V[b].fixed:
            target.V[b].set_value(math.sqrt(max(value(relaxed.rx_c[b]), 0.0)), skip_validation=True)
        if not target.theta[b].fixed and int(b) in theta:
            target.theta[b].set_value(theta[int(b)], skip_validation=True)
    for name in ("Pg_new", "Qg_new", "Pext", "Qext"):
        own, found = getattr(target, name), getattr(relaxed, name)
        for k in own:
            if not own[k].fixed and found[k].value is not None:
                own[k].set_value(value(found[k]), skip_validation=True)
    return target if fo.solve(target) else None


def second_look(model) -> dict:
    """What the relaxation says about a model IPOPT found no solution for.

    Infeasible relaxation: no dispatch of the units meets the model's limits
    — proved, as the relaxation holds every AC solution (only a conic
    solver's infeasibility certificate counts; IPOPT's "locally infeasible"
    proves nothing). Feasible: IPOPT is restarted from the relaxation's
    solution, and a dispatch it then finds is returned. {"verdict", "relaxation",
    "why", "model": the solved model when found, else None}. `model` stays as it is."""
    m = model.clone()
    _near_schedule(m)
    why = soc_relax(m)
    kind = "sdp" if _sdp_available() and len(m.B) <= SDP_MAX_BUSES else "soc"
    out = {"verdict": NOT_PROVEN, "relaxation": kind, "why": why, "model": None}
    if why:
        return out
    if kind == "sdp":
        import cvxpy as cp
        parts = _sdp_problem(m)
        if isinstance(parts, tuple):
            return {**out, "why": parts[1]}
        status = "not solved"
        for settings in _CLARABEL_TRIES:
            try:
                parts["problem"].solve(solver="CLARABEL", **settings)
            except Exception as exc:  # noqa: BLE001 — try the next settings
                status = f"failed ({str(exc)[:80]})"
                continue
            status = parts["problem"].status
            if status in (cp.OPTIMAL, cp.INFEASIBLE):
                break
        if status == cp.INFEASIBLE:
            return {**out, "verdict": NO_DISPATCH}
        if status != cp.OPTIMAL:
            return {**out, "why": f"the semidefinite relaxation ended {status}"}
        for j, var in parts["columns"].values():
            if not var.fixed:
                var.set_value(float(parts["x"].value[j]), skip_validation=True)
    else:
        try:
            res = SolverFactory("ipopt").solve(m, tee=False, load_solutions=False)
        except Exception as exc:  # noqa: BLE001
            return {**out, "why": f"the relaxation could not be solved ({exc})"}
        if res.solver.termination_condition != TerminationCondition.optimal:
            return {**out, "why": (f"the cone relaxation ended {res.solver.termination_condition}; without the "
                                   "semidefinite solver that proves nothing")}
        m.solutions.load_from(res)
    found = _restart_from(model, m)
    if found is not None:
        return {**out, "verdict": FOUND, "model": found}
    return {**out, "why": "the relaxation has a solution, and IPOPT restarted from it found no dispatch either"}


def proof_block(look: dict, scope: str = "") -> dict:
    """`second_look` as a result reports it (without the model)."""
    meaning = {
        NO_DISPATCH: (f"Proved: no dispatch of the units meets these limits{scope}. The convex relaxation, which "
                      "holds every AC solution, has none (the conic solver's infeasibility certificate)."),
        FOUND: ("IPOPT found no dispatch from the power flow; restarted from the convex relaxation's solution, "
                "it found this one."),
        NOT_PROVEN: ("Not proven either way: the solver found no dispatch, which does not show that none "
                     "exists."),
    }[look["verdict"]]
    return {"verdict": look["verdict"], "relaxation": look.get("relaxation"), "meaning": meaning,
            **({"why": look["why"]} if look.get("why") and look["verdict"] == NOT_PROVEN else {})}
