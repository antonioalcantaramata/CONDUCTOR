"""
A lower bound on the cheapest option's cost (backend/relaxation.py): no
dispatch of the units meeting the same limits costs less, so the gap is the
most any other dispatch could save.

What makes it a bound is that every AC solution is feasible in the relaxation.
That is tested directly — the AC cheapest solution, mapped into the relaxed
model and into the semidefinite problem, meets every constraint — besides the
bound never exceeding the cheapest, the semidefinite bound being at least as
tight as the second-order cone one, and what the result says.
"""

import contextlib
import io
import math
import shutil

import numpy as np
import pytest
from pyomo.environ import Constraint, value

import flex_options as fo
import main_backend as mb
import relaxation as rx

needs_ipopt = pytest.mark.skipif(shutil.which("ipopt") is None, reason="IPOPT not installed")
PEAK = "2000-01-06 20:00:00"


@pytest.fixture(scope="module")
def demo(tmp_path_factory):
    pytest.importorskip("fastapi.testclient")
    from fastapi.testclient import TestClient
    store = tmp_path_factory.mktemp("bound")
    patch = pytest.MonkeyPatch()
    patch.delenv("GRID_PROFILE", raising=False)
    patch.setattr(mb, "SYSTEMS_DIR", str(store / "systems"))
    patch.setattr(mb, "DATA_FILES_DIR", str(store / "data_files"))
    for name in ("_LAST_UPLOAD_SENTINEL", "_LAST_TIMESERIES_SENTINEL", "_LAST_FORECAST_SENTINEL",
                 "_PROPOSALS_FILE", "_ASSUMPTIONS_FILE", "_SCENARIO_FILE"):
        patch.setattr(mb, name, str(store / f"{name}.json"))
    saved = dict(mb.app_data)
    with TestClient(mb.app) as client:
        yield client
    mb.app_data.clear()
    mb.app_data.update(saved)
    patch.undo()


@pytest.fixture(scope="module")
def peak(demo):
    """The options at the 30-bus demo's evening peak, and the model they were solved on."""
    sink: dict = {}
    token = mb._FLEX_MODEL_SINK.set(sink)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            result = mb._run_coroutine(mb.optimise_flexibility.__wrapped__(
                mb.FlexibilityRequest(timestamp=PEAK, options=True)))
    finally:
        mb._FLEX_MODEL_SINK.reset(token)
    prices = mb._flex_prices(sink["net"], sink["model"])[0]
    return result, sink, prices


def _ac_cheapest(sink, prices):
    """The cheapest option's own problem (cost alone), solved by IPOPT."""
    m = sink["model"].clone()
    cost = fo.add_cost(m, prices, sink["pext0"], sink["qext0"])
    fo._replace_objective(m, cost)
    assert fo.solve(m)
    return m, cost


def _at_the_ac_point(m):
    """The relaxation's (c, s) set to the products of the AC voltages."""
    for b in m.B:
        m.rx_c[b].set_value(value(m.V[b]) ** 2, skip_validation=True)
    for i, j in m.rx_pairs:
        angle = value(m.theta[i]) - value(m.theta[j])
        m.rx_cc[i, j].set_value(value(m.V[i]) * value(m.V[j]) * math.cos(angle))
        m.rx_ss[i, j].set_value(value(m.V[i]) * value(m.V[j]) * math.sin(angle))


@needs_ipopt
def test_the_relaxed_constraints_are_the_model_s_own_at_an_ac_point(peak):
    _, sink, prices = peak
    m, _ = _ac_cheapest(sink, prices)
    assert rx.soc_relax(m) is None
    _at_the_ac_point(m)
    for original, relaxed in ((m.active_balance, m.rx_active_balance), (m.reactive_balance, m.rx_reactive_balance)):
        for b in relaxed:
            assert value(relaxed[b].body) == pytest.approx(value(original[b].body) - value(original[b].upper), abs=1e-8)
    for k in m.rx_current:
        assert value(m.rx_current[k].body) == pytest.approx(value(m.branch_current_from_limit[k].body), abs=1e-8)
        assert value(m.rx_current[k].upper) == pytest.approx(value(m.branch_current_from_limit[k].upper))


@needs_ipopt
def test_the_ac_cheapest_is_feasible_in_the_semidefinite_relaxation(peak):
    pytest.importorskip("cvxpy")
    _, sink, prices = peak
    m, cost = _ac_cheapest(sink, prices)
    assert rx.soc_relax(m) is None
    _at_the_ac_point(m)
    parts = rx._sdp_problem(m)
    parts["x"].value = np.array([value(v) for _, v in sorted(parts["columns"].values(), key=lambda t: t[0])])
    voltage = {b: value(m.V[b]) * np.exp(1j * value(m.theta[b])) for b in m.B}
    fill, z = parts["fill"]
    if fill:
        products = [voltage[i] * np.conj(voltage[j]) for i, j in fill]
        z.value = np.array([[p.real, p.imag] for p in products])
    for nodes, block in parts["cliques"]:
        w = np.array([[voltage[a] * np.conj(voltage[b]) for b in nodes] for a in nodes])
        block.value = np.block([[w.real, -w.imag], [w.imag, w.real]])
    worst = max(float(np.max(np.abs(c.violation()))) for c in parts["problem"].constraints)
    assert worst < 1e-5
    assert parts["problem"].objective.value == pytest.approx(value(cost), abs=1e-6)


@needs_ipopt
def test_the_bound_is_below_the_cheapest_and_the_sdp_is_tighter(peak, monkeypatch):
    pytest.importorskip("cvxpy")
    result, sink, prices = peak
    cheapest = next(o["cost_eur_per_h"] for o in result["options"] if o["id"] == "cheapest")
    sdp = rx.cost_lower_bound(sink["model"], prices, sink["pext0"], sink["qext0"])
    assert sdp["relaxation"] == "sdp" and sdp["why"] is None
    monkeypatch.setattr(rx, "_sdp_available", lambda: False)
    soc = rx.cost_lower_bound(sink["model"], prices, sink["pext0"], sink["qext0"])
    assert soc["relaxation"] == "soc" and soc["why"] is None
    assert soc["lower_bound_eur_per_h"] <= sdp["lower_bound_eur_per_h"] + 1e-4
    assert sdp["lower_bound_eur_per_h"] <= cheapest + 0.01
    # The congested peak: the cone relaxation is far below, the semidefinite one close.
    assert cheapest - sdp["lower_bound_eur_per_h"] < 10.0 < cheapest - soc["lower_bound_eur_per_h"]


@needs_ipopt
def test_the_options_result_states_the_bound(peak):
    pytest.importorskip("cvxpy")
    result, _, _ = peak
    bound = result["optimality"]
    assert bound["relaxation"] == "sdp" and bound["lower_bound_eur_per_h"] <= bound["cheapest_eur_per_h"]
    assert bound["gap_eur_per_h"] == pytest.approx(bound["cheapest_eur_per_h"] - bound["lower_bound_eur_per_h"],
                                                   abs=0.011)
    assert f"costs less than {bound['lower_bound_eur_per_h']:.2f} €/h" in result["message"]


@needs_ipopt
def test_a_tight_moment_is_called_the_best_possible(demo):
    pytest.importorskip("cvxpy")
    with contextlib.redirect_stdout(io.StringIO()):
        out = demo.post("/api/flexibility/optimize", json={"timestamp": PEAK, "options": True,
                                                           "load_scaling_factor": 1.15}).json()
    assert out["optimality"]["best_possible"] is True and "the best possible" in out["message"]


@needs_ipopt
def test_the_last_resort_has_no_bound(demo):
    with contextlib.redirect_stdout(io.StringIO()):
        out = demo.post("/api/flexibility/optimize", json={"timestamp": PEAK, "options": True,
                                                           "load_scaling_factor": 1.25}).json()
    assert out["set"] == "last_resort" and "optimality" not in out


@needs_ipopt
def test_a_voltage_constraint_it_cannot_relax_is_refused(peak):
    _, sink, prices = peak
    m = sink["model"].clone()
    bus = next(b for b in m.B if not m.V[b].fixed and not m.theta[b].fixed)
    m.extra = Constraint(expr=m.V[bus] * m.theta[bus] <= 1.0)
    assert "extra" in rx.soc_relax(m.clone())
    bound = rx.cost_lower_bound(m, prices, sink["pext0"], sink["qext0"])
    assert bound["lower_bound_eur_per_h"] is None and "extra" in bound["why"]


def test_the_bound_is_rounded_down_and_a_saving_has_no_share():
    block = rx.optimality({"lower_bound_eur_per_h": 22.4861, "relaxation": "sdp"}, 26.48)
    assert block["lower_bound_eur_per_h"] == 22.48 and block["gap_eur_per_h"] == 4.0
    assert block["gap_pct"] == pytest.approx(15.11) and block["best_possible"] is False
    assert "saves more than 4.00 €/h" in rx.sentence(block)
    saving = rx.optimality({"lower_bound_eur_per_h": -110.34, "relaxation": "sdp"}, -83.55)
    assert "gap_pct" not in saving and saving["gap_eur_per_h"] == 26.79
    tight = rx.optimality({"lower_bound_eur_per_h": 381.19, "relaxation": "sdp"}, 381.21)
    assert tight["best_possible"] is True and "is the best possible (within 0.02 €/h)" in rx.sentence(tight)


def test_a_bound_above_the_cheapest_is_not_reported():
    block = rx.optimality({"lower_bound_eur_per_h": 30.0, "relaxation": "sdp"}, 26.48)
    assert block["lower_bound_eur_per_h"] is None and "cannot" in block["why"]
    assert rx.sentence(block) == ""
    assert rx.optimality({"lower_bound_eur_per_h": None, "why": "x"}, None) is None


# ---------------------------------------------------------------------------
# When IPOPT finds no dispatch: a proof that none exists, or a second start
# ---------------------------------------------------------------------------

@needs_ipopt
def test_a_feasible_problem_is_never_proved_infeasible(peak):
    """The model the options were solved on has a dispatch: the relaxation must
    find it again (restarting IPOPT from its solution), never 'prove' none."""
    pytest.importorskip("cvxpy")
    _, sink, _ = peak
    look = rx.second_look(sink["model"])
    assert look["verdict"] == rx.FOUND and look["model"] is not None
    assert rx.proof_block(look)["verdict"] == rx.FOUND and "model" not in rx.proof_block(look)


@needs_ipopt
def test_no_redispatch_at_125_percent_is_proved(demo):
    pytest.importorskip("cvxpy")
    with contextlib.redirect_stdout(io.StringIO()):
        out = demo.post("/api/flexibility/optimize", json={"timestamp": PEAK, "load_scaling_factor": 1.25}).json()
    assert out["feasible"] is False and out["no_dispatch_check"]["verdict"] == rx.NO_DISPATCH
    assert out["message"].startswith("Proved: no dispatch of the units meets these limits.")
    with contextlib.redirect_stdout(io.StringIO()):
        options = demo.post("/api/flexibility/optimize", json={"timestamp": PEAK, "load_scaling_factor": 1.25,
                                                               "options": True}).json()
    assert options["set"] == "last_resort" and "proved by a convex relaxation" in options["message"]
    assert options["redispatch_alone"]["no_dispatch_check"]["verdict"] == rx.NO_DISPATCH


@needs_ipopt
def test_an_impossible_exchange_cap_is_proved(demo):
    pytest.importorskip("cvxpy")
    with contextlib.redirect_stdout(io.StringIO()):
        out = demo.post("/api/flexibility/optimize", json={"timestamp": PEAK, "slack_max_mw": 10}).json()
    assert out["status"] == "cap_infeasible" and out["no_dispatch_check"]["verdict"] == rx.NO_DISPATCH
    assert "within ±10 MW" in out["no_dispatch_check"]["meaning"]


@needs_ipopt
def test_a_dispatch_ipopt_missed_is_found_from_the_relaxation(demo, monkeypatch):
    """IPOPT made to fail on a solvable moment: the second start finds the
    dispatch, and the result says how it was found."""
    pytest.importorskip("cvxpy")
    from pyomo.opt import TerminationCondition

    real = mb.fe.optimization_model_base

    class Failed:
        class solver:
            termination_condition = TerminationCondition.infeasible

    def failing(*args, **kwargs):
        _, model = real(*args, **kwargs)
        return Failed(), model
    monkeypatch.setattr(mb.fe, "optimization_model_base", failing)
    with contextlib.redirect_stdout(io.StringIO()):
        out = demo.post("/api/flexibility/optimize", json={"timestamp": PEAK}).json()
    assert out["status"] == "optimal" and out["found_by_relaxation"] is True
    assert "restarted from the solution of the convex relaxation" in out["message"]


def test_a_failed_check_proves_nothing():
    look = mb._second_look(object())                  # not a model: the check fails, never fatal
    assert look["verdict"] == rx.NOT_PROVEN and look["model"] is None
    failed = mb._with_proof({"message": "Infeasible."}, look)
    assert failed["message"] == "Infeasible." and failed["no_dispatch_check"]["verdict"] == rx.NOT_PROVEN
