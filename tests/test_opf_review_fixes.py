"""
The optimal power flow — what a code review found wrong, fixed:

- the exchange cap replaced the connection's own capacity, so a cap of 999 MW
  loosened the 30-bus case's 200 MW (and the reactive limit with it);
- a failure caused by the import limit was diagnosed as "beyond voltage and
  branch levers", and a capped solve that failed got no diagnosis at all;
- the external grid's "before" came from the network before load scaling;
- the reported OPF band was the first bus's bounds (often the slack's, widened
  to hold its measured voltage);
- a set-point outside its unit's range crashed the solver (a bare HTTP 500);
- `disabled_generators` ignored voltage-controlled units, and unknown names.
"""

import shutil

import pandapower as pp
import pandapower.networks as pn
import pytest
from pyomo.environ import ConcreteModel, Constraint, Objective, Set, Var

import flex_engine as fe
import main_backend as mb

needs_ipopt = pytest.mark.skipif(shutil.which("ipopt") is None, reason="IPOPT not installed")


def _exchange_model(p_cap=200.0, q_cap=200.0):
    m = ConcreteModel()
    m.G = Set(initialize=["u"])
    m.Pg_new = Var(m.G, bounds=(0.0, 10.0), initialize=0.0)
    m.Qg_new = Var(m.G, initialize=0.0)
    m.Ext = Set(initialize=["grid"])
    m.Pext = Var(m.Ext, bounds=(-p_cap, p_cap), initialize=0.0)
    m.Qext = Var(m.Ext, bounds=(-q_cap, q_cap), initialize=0.0)
    m._ext_capacity = {"grid": ((-p_cap, p_cap), (-q_cap, q_cap))}
    return m


def test_the_cap_tightens_the_connection_never_loosens_it():
    m = _exchange_model()
    mb._cap_exchange(m, 999.0)
    assert (m.Pext["grid"].lb, m.Pext["grid"].ub) == (-200.0, 200.0)
    assert (m.Qext["grid"].lb, m.Qext["grid"].ub) == (-200.0, 200.0)
    mb._cap_exchange(m, 50.0)
    assert (m.Pext["grid"].lb, m.Pext["grid"].ub) == (-50.0, 50.0)
    mb._cap_exchange(m, 120.0)                       # against the capacity, not the earlier cap
    assert m.Pext["grid"].ub == 120.0


def test_a_failed_capped_solve_says_what_exchange_it_needs():
    m = _exchange_model()
    mb._cap_exchange(m, 10.0)
    diagnosis = mb._exchange_cap_diagnosis({"grid": (155.8, 4.0)}, m, " after the outage")
    assert diagnosis["binding_limit"] == "exchange_limit"
    assert "would import 155.8 MW (limit 10 MW)" in diagnosis["explanation"]
    assert "Mvar" not in diagnosis["explanation"]                         # 4 Mvar is within its limit
    fits = mb._exchange_cap_diagnosis({"grid": (5.0, 1.0)}, m)
    assert fits["binding_limit"] == "solver_failure" and "numerical" in fits["explanation"]


@needs_ipopt
def test_the_diagnosis_names_the_exchange_limit():
    m = _exchange_model(p_cap=1.0, q_cap=1.0)
    m.need = Constraint(expr=m.Pext["grid"] + m.Pg_new["u"] == 13.0)     # 10 MW local at most
    m.obj = Objective(expr=m.Pg_new["u"] ** 2)
    needed = fe._exchange_check(m)
    assert [(x["name"], x["outside"]) for x in needed] == [("grid", ["active"])]
    assert needed[0]["p_mw"] >= 3.0 - 1e-3
    assert (m.Pext["grid"].lb, m.Pext["grid"].ub) == (-1.0, 1.0)      # restored
    assert "imports" in fe._exchange_sentence(needed[0])


def test_a_set_point_outside_its_range_is_refused_by_name():
    m = _exchange_model()
    with pytest.raises(fe.PinnedOutsideBounds, match="u at 12.00 MW"):
        fe._checked_pins(m, {"u": 12.0})
    assert fe._checked_pins(m, {"u": 10.0004}) == {"u": 10.0}            # the schedule's rounding


def test_the_reported_band_is_the_one_asked_for():
    net = pp.create_empty_network()
    for _ in range(2):
        pp.create_bus(net, vn_kv=20.0)
    m = ConcreteModel()
    m.B = Set(initialize=[0, 1])
    m.V = Var(m.B, initialize=1.0)
    m.V[0].setlb(0.94)
    m.V[0].setub(1.02)                                  # the slack's bound, widened to its 1.02 p.u.
    m._opf_band = (0.94, 1.01)
    out = fe.extract_post_opf_voltages(m, net)
    assert (out["opf_vm_lower_used"], out["opf_vm_upper_used"]) == (0.94, 1.01)


def test_disabled_units_include_voltage_controlled_ones():
    net = pn.case14()
    pp.create_sgen(net, 3, p_mw=1.0)
    pp.create_sgen(net, 4, p_mw=1.0)
    mb._drop_disabled_units(net, ["gen_bus2", "sgen_0"])
    assert "Gen_bus2" not in {mb.gm.unit_name(net, "gen", i) for i in net.gen.index}
    assert list(net.sgen.index) == [1]                                    # sgen_1 keeps its name
    with pytest.raises(mb.HTTPException) as refused:
        mb._drop_disabled_units(net, ["Nope"])
    assert refused.value.status_code == 400 and "Gen_bus1" in refused.value.detail


def test_the_grid_before_is_the_power_flow_of_the_moment_studied():
    net, base = pn.case14(), pn.case14()
    pp.runpp(base)
    net.load["p_mw"] *= 1.2
    pp.runpp(net)
    assert fe._ext_grid_before(net, base)[0] == pytest.approx(float(net.res_ext_grid.p_mw.iloc[0]))
    net.res_ext_grid["p_mw"] = float("nan")                               # no converged flow: the other one
    assert fe._ext_grid_before(net, base)[0] == pytest.approx(float(base.res_ext_grid.p_mw.iloc[0]))


@needs_ipopt
def test_the_dispatch_starts_from_the_scaled_moment(backend_client):
    ts = mb.app_data["timestamps"][40]
    flex = backend_client.post("/api/flexibility/optimize", json={"timestamp": ts, "load_scaling_factor": 1.2}).json()
    rsa = backend_client.post("/api/grid/rsa", json={"timestamp": ts, "load_scaling_factor": 1.2}).json()
    grid = next(a for a in flex["activated_resources"] if a["type"] == "External Grid")
    assert grid["Pg_base"] == pytest.approx(rsa["slack_import_mw"], abs=1e-3)


@needs_ipopt
def test_contradictions_and_caps_are_explained(backend_client):
    ts = mb.app_data["timestamps"][40]
    pinned = backend_client.post("/api/flexibility/optimize",
                                 json={"timestamp": ts, "fixed_setpoints": {"Gen_bus2": 500}})
    assert pinned.status_code == 422 and "Gen_bus2 at 500.00 MW" in pinned.json()["detail"]
    capped = backend_client.post("/api/flexibility/optimize", json={"timestamp": ts, "slack_max_mw": 10}).json()
    assert capped["status"] == "cap_infeasible"
    assert capped["infeasibility_diagnosis"]["binding_limit"] == "exchange_limit"
    assert "would import" in capped["message"]


@pytest.mark.parametrize("request_type", [mb.FlexibilityRequest, mb.KPIRequest])
def test_a_negative_unit_limit_is_refused(request_type):
    with pytest.raises(ValueError, match="cannot be negative"):
        request_type(pg_min_overrides={"Gen_bus1": -5.0})
    assert request_type(pg_min_overrides={"Gen_bus1": 5.0}).pg_min_overrides == {"Gen_bus1": 5.0}
