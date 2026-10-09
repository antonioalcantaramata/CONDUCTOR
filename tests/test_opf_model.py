"""
The OPF model itself, built and solved directly on the default network
(sandbox phase 0, code review Batch C). Needs IPOPT.

Each test states a property the physics fixes — an answer must not depend on
the per-unit base, a branch limit means what the parameter says — so it holds
whatever the operating point.
"""

import copy
import shutil

import pandapower as pp
import pytest
from pyomo.environ import value
from pyomo.opt import TerminationCondition

import flex_engine as fe
import network_loader as nl

pytestmark = pytest.mark.skipif(shutil.which("ipopt") is None, reason="IPOPT not installed")


@pytest.fixture(scope="module")
def default_net():
    profile = nl.load_profile("pglib_case14")
    net = nl.load_network(profile)
    net, _, _ = nl.gen_to_sgen(net)
    nl.apply_voltage_setpoints(net, profile)
    net.trafo["tap_pos"] = net.trafo["tap_neutral"]
    return net


def _solve(net, **kw):
    net = copy.deepcopy(net)
    pp.runpp(net)
    db = nl._build_ybus_entry(net)
    res, model = fe.optimization_model_base(
        net, net, db["Yff_r"], db["Yff_i"], db["Yft_r"], db["Yft_i"], db["TAPS"],
        db["trafo_ranges"], db["trafo_defaults"], db["branch_to_trafo"],
        opf_vm_lower=0.94, opf_vm_upper=1.06, **kw)
    assert res.solver.termination_condition == TerminationCondition.optimal
    return model


def test_the_answer_does_not_depend_on_the_per_unit_base(default_net):
    """The balance equations used ×100 whatever the network's base: on a
    10 MVA base every flow was off by a factor of ten (review 2.10)."""
    other = copy.deepcopy(default_net)
    other.sn_mva = 10.0
    a, b = _solve(default_net), _solve(other)
    for e in a.Ext:
        assert value(b.Pext[e]) == pytest.approx(value(a.Pext[e]), abs=1e-3)
        assert value(b.Qext[e]) == pytest.approx(value(a.Qext[e]), abs=1e-3)
    for bus in a.B:
        assert value(b.V[bus]) == pytest.approx(value(a.V[bus]), abs=1e-5)


def test_a_real_generator_kept_as_gen_can_produce():
    """Without the gen→sgen conversion every `net.gen` row was pinned to 0 MW
    as if it were a condenser (review 2.7). Only units without capacity are."""
    profile = nl.load_profile("pglib_case14")
    net = nl.load_network(profile)          # no gen_to_sgen: the 59 MW unit stays a gen
    nl.apply_voltage_setpoints(net, profile)
    net.trafo["tap_pos"] = net.trafo["tap_neutral"]
    model = _solve(net)
    caps = {g: value(model.Pg_max[g]) for g in model.G}
    assert caps["Gen_bus1"] == pytest.approx(59.0)
    assert caps["Gen_bus2"] == caps["Gen_bus5"] == caps["Gen_bus7"] == 0.0


def test_a_small_unit_has_a_reactive_limit(default_net):
    """A static unit under 1 MW that is not a condenser had no Q constraint at
    all — the OPF could draw unlimited reactive power from it (review 2.8)."""
    net = copy.deepcopy(default_net)
    idx = pp.create_sgen(net, bus=13, p_mw=0.5, max_p_mw=0.8, name="Small PV")
    net.sgen.at[idx, "substation_name"] = "Small PV"
    model = _solve(net, min_power_factor=0.95)
    assert "Small PV" in set(model.PF_set)
    q, p = abs(value(model.Qg_new["Small PV"])), value(model.Pg_new["Small PV"])
    assert q <= 0.329 * p + 1e-4


def _replay(net, model):
    """The OPF's setpoints in a pandapower power flow: the validation method of
    the phase 0 addendum. Returns (max |ΔV| p.u., |ΔP grid| MW, |ΔQ grid| Mvar)."""
    n = copy.deepcopy(net)
    names = {}
    for idx, row in n.sgen.iterrows():
        for key in (row.get("substation_name"), row.get("name"), f"sgen_{idx}"):
            if isinstance(key, str) and key.strip():
                names[key.strip()] = ("sgen", idx)
    for idx, row in n.gen.iterrows():
        names[str(row.get("name") or "").strip() or f"Gen_bus{int(row.bus)}"] = ("gen", idx)
    for g in model.G:
        table, idx = names[g]
        if table == "sgen":
            n.sgen.at[idx, "p_mw"] = value(model.Pg_new[g])
            n.sgen.at[idx, "q_mvar"] = value(model.Qg_new[g])
        else:
            n.gen.at[idx, "p_mw"] = value(model.Pg_new[g])
            n.gen.at[idx, "vm_pu"] = value(model.V[int(n.gen.at[idx, "bus"])])
    slack = int(n.ext_grid.bus.iloc[0])
    n.ext_grid["vm_pu"] = value(model.V[slack])
    pp.runpp(n)
    dv = max(abs(float(n.res_bus.vm_pu[b]) - value(model.V[b])) for b in model.B)
    ext = next(iter(model.Ext))
    dp = abs(float(n.res_ext_grid.p_mw.iloc[0]) - value(model.Pext[ext]))
    dq = abs(float(n.res_ext_grid.q_mvar.iloc[0]) - value(model.Qext[ext]))
    return dv, dp, dq


def test_the_optimum_replays_in_pandapower(default_net):
    dv, dp, dq = _replay(default_net, _solve(default_net))
    assert dv < 1e-4 and dp < 0.05 and dq < 0.05


def test_shunts_replay_with_step_active_power_and_no_names(default_net):
    """Shunts were keyed by name (two unnamed ones became one), ignored `step`,
    and took their active power as constant rather than ∝ V² (review 2.9)."""
    net = copy.deepcopy(default_net)
    pp.create_shunt(net, bus=4, q_mvar=-3.0, p_mw=2.0, step=2)
    pp.create_shunt(net, bus=12, q_mvar=-1.5, p_mw=1.0)
    assert net.shunt["name"].isna().sum() + (net.shunt["name"] == "").sum() >= 2
    dv, dp, dq = _replay(net, _solve(net))
    assert dv < 1e-4 and dp < 0.05 and dq < 0.05


def test_the_current_margin_is_a_fraction_of_rated_current(default_net):
    """`opf_current_safety_margin` 0.9 means 90 % of rated current. It was
    applied to I², so 0.9 allowed √0.9 ≈ 94.9 % (review 2.4). Checked on the
    constraint itself — every branch's bound on I² is (margin × rating)² —
    so it holds whether or not that margin leaves a feasible dispatch."""
    net = copy.deepcopy(default_net)
    pp.runpp(net)
    db = nl._build_ybus_entry(net)
    _, model = fe.optimization_model_base(
        net, net, db["Yff_r"], db["Yff_i"], db["Yft_r"], db["Yft_i"], db["TAPS"],
        db["trafo_ranges"], db["trafo_defaults"], db["branch_to_trafo"],
        opf_vm_lower=0.94, opf_vm_upper=1.06, current_safety_margin=0.8)
    checked = 0
    for c in (model.branch_current_from_limit, model.branch_current_to_limit):
        for (i, j) in c:
            for a, b in ((i, j), (j, i)):
                if (a, b) in model.I_from_max:
                    rating_sq = (value(model.I_from_max[a, b]) / value(model.Ibase_from_kA[a, b])) ** 2
                    break
            assert value(c[i, j].upper) == pytest.approx(0.8 ** 2 * rating_sq, rel=1e-9)
            checked += 1
    assert checked == 2 * len(model.Branches)
