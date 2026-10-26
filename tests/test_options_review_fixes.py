"""
Corrective options and their prices — what a code review found wrong, fixed:

- an option issuing fewer instructions was dropped as "the same dispatch" as
  one with the same active power;
- units held by equal bounds (condensers, non-dispatchable units) counted as
  free in the near-optimal search;
- the cost table called a small static generator a condenser (no redispatch
  price: it moved for free), unlike the optimisation;
- an external-grid redispatch price was accepted and never used;
- a negative case marginal cost gave negative redispatch prices.
"""

import pandapower as pp
import pandapower.networks as pn
import pytest
from pyomo.environ import ConcreteModel, Set, Var

import costs
import flex_options as fo


def _option(sid, p, q=None):
    p0 = {"A": 10.0, "B": 5.0}
    return fo.Option(id=sid, label=fo.STRATEGIES[sid][0], how="", p=p, q=q or {g: 0.0 for g in p}, p0=p0,
                     q0={g: 0.0 for g in p}, pext={"grid": 50.0}, qext={"grid": 0.0}, v={0: 1.0},
                     max_loading_pct=None, max_loading_branch=None)


def test_fewer_instructions_is_not_the_same_dispatch():
    many = _option("least_disturbance", {"A": 12.0, "B": 5.0}, q={"A": 0.0, "B": 3.0})   # P on A, Q on B
    few = _option("fewest_units", {"A": 12.0, "B": 5.0})                                 # P on A only
    assert not fo._same(many, few)
    kept, _ = fo.select([(many, {"total": 10.0}), (few, {"total": 10.0})])
    assert {o.id for o, _ in kept} == {"least_disturbance", "fewest_units"}


def test_units_held_by_their_bounds_are_not_free():
    m = ConcreteModel()
    m.G = Set(initialize=["free", "condenser", "fixed_output", "pinned"])
    m.Pg_new = Var(m.G)
    m.Pg_new["free"].setlb(0.0)
    m.Pg_new["free"].setub(10.0)
    m.Pg_new["condenser"].setlb(0.0)
    m.Pg_new["condenser"].setub(0.0)
    m.Pg_new["fixed_output"].setlb(4.2)
    m.Pg_new["fixed_output"].setub(4.2)
    m.Pg_new["pinned"].fix(3.0)
    assert fo._free_units(m) == ["free"]


def test_a_small_static_generator_is_priced_like_any_unit():
    net = pn.case14()
    pp.create_sgen(net, 4, p_mw=0.4, max_p_mw=0.5, name="Small PV")
    pp.create_gen(net, 9, p_mw=0.0, max_p_mw=0.0, vm_pu=1.0, name="Condenser")
    rows = {r["unit"]: r for r in costs.defaults(net)}
    assert rows["Small PV"]["kind"] == "unit" and rows["Small PV"]["up_eur_per_mwh"] > 0
    assert rows["Condenser"]["kind"] == "condenser" and rows["Condenser"]["up_eur_per_mwh"] is None


def test_the_external_grid_takes_no_redispatch_price():
    net = pn.case14()
    with pytest.raises(costs.CostError, match="not redispatched"):
        costs.normalise(net, [{"unit": "External grid", "up_eur_per_mwh": 80}])
    assert costs.normalise(net, [{"unit": "External grid", "cost_eur_per_mwh": 70}])


def test_a_negative_marginal_cost_gives_no_negative_price():
    prices = costs._redispatch({"cost_eur_per_mwh": -20.0, "cost_eur_per_mw2h": 0.0}, 50.0,
                               renewable=False, condenser=False)
    assert prices == {"up_eur_per_mwh": 0.0, "down_eur_per_mwh": 0.0}
