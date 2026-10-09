"""
Parallel and out-of-service branches in the OPF's branch data (sandbox hazard
A.3 (2), code review 2.9).

The admittance database and the branch-limit table are keyed by bus pair. A
second circuit between the same buses used to overwrite the first, so the OPF
saw one circuit at half the capacity; out-of-service branches stayed in both
tables. Two identical parallel lines must now look exactly like one line with
`parallel = 2`, and a branch switched out must disappear.
"""

import copy

import pandapower as pp
import pandapower.networks as pn
import pytest

import flex_engine as fe
import network_loader as nl


def _db(net):
    pp.runpp(net)
    return nl._build_ybus_entry(net)


def _pair(db, i, j, tap=1):
    return tuple(round(db[k][(i, j), tap], 9) for k in ("Yff_r", "Yff_i", "Yft_r", "Yft_i"))


@pytest.fixture
def base():
    return pn.case14()


def _twin(net):
    """Line 0 duplicated: a second identical circuit between the same buses."""
    twin = copy.deepcopy(net)
    row = twin.line.loc[0]
    pp.create_line_from_parameters(
        twin, from_bus=int(row.from_bus), to_bus=int(row.to_bus), length_km=float(row.length_km),
        r_ohm_per_km=float(row.r_ohm_per_km), x_ohm_per_km=float(row.x_ohm_per_km),
        c_nf_per_km=float(row.c_nf_per_km), max_i_ka=float(row.max_i_ka))
    return twin


def _doubled(net):
    doubled = copy.deepcopy(net)
    doubled.line.at[0, "parallel"] = 2
    return doubled


def test_two_parallel_lines_admit_like_one_doubled_line(base):
    i, j = int(base.line.from_bus[0]), int(base.line.to_bus[0])
    twin, doubled, single = _db(_twin(base)), _db(_doubled(base)), _db(copy.deepcopy(base))
    assert _pair(twin, i, j) == pytest.approx(_pair(doubled, i, j))
    assert _pair(twin, j, i) == pytest.approx(_pair(doubled, j, i))
    assert _pair(twin, i, j) != pytest.approx(_pair(single, i, j))


def test_their_current_limits_add(base):
    i, j = int(base.line.from_bus[0]), int(base.line.to_bus[0])
    single = fe._branch_current_data(base)[(i, j)]["I_from_max"]
    assert fe._branch_current_data(_twin(base))[(i, j)]["I_from_max"] == pytest.approx(2 * single)
    assert fe._branch_current_data(_doubled(base))[(i, j)]["I_from_max"] == pytest.approx(2 * single)


def test_derating_factor_applies(base):
    i, j = int(base.line.from_bus[0]), int(base.line.to_bus[0])
    single = fe._branch_current_data(base)[(i, j)]["I_from_max"]
    base.line.at[0, "df"] = 0.8
    assert fe._branch_current_data(base)[(i, j)]["I_from_max"] == pytest.approx(0.8 * single)


def test_an_out_of_service_branch_is_left_out(base):
    i, j = int(base.line.from_bus[0]), int(base.line.to_bus[0])
    base.line.at[0, "in_service"] = False
    assert (i, j) not in fe._branch_current_data(base)
    assert ((i, j), 1) not in _db(base)["Yff_r"]


def test_an_unchanged_network_keeps_its_branch_data(base):
    data = fe._branch_current_data(base)
    assert len(data) == 2 * (len(base.line) + len(base.trafo))


class TestLoads:
    """Two loads on one substation used to become one OPF entry holding only
    the last load's demand (sandbox hazard A.3 (3))."""

    def test_two_loads_with_one_name_both_count(self, base):
        base.load["substation_name"] = [f"Bus_{b}" for b in base.load.bus]
        idx = pp.create_load(base, bus=int(base.load.bus.iloc[0]), p_mw=50.0, q_mvar=5.0)
        base.load.at[idx, "substation_name"] = base.load.substation_name.iloc[0]
        keys, p, q, bus = fe._load_data(base)
        assert len(keys) == len(base.load)
        assert sum(p.values()) == pytest.approx(base.load.p_mw.sum())

    def test_out_of_service_and_scaling(self, base):
        base.load.at[0, "in_service"] = False
        base.load.at[1, "scaling"] = 0.5
        keys, p, _, _ = fe._load_data(base)
        assert 0 not in keys
        assert p[1] == pytest.approx(0.5 * base.load.p_mw[1])


def test_scenario_curtailment_is_what_the_plan_holds_back():
    """Code review 2.5: in a scenario a unit may inject below min(setpoint,
    available resource) at almost no cost; that difference is reported."""
    import types
    model = types.SimpleNamespace(
        K=[0, 1, 2], G=["Wind"],
        Pg_new={"Wind": 5.0},
        xi={("Wind", 0): 4.0, ("Wind", 1): 6.0, ("Wind", 2): 6.0},
        Pinj={("Wind", 0): 3.0, ("Wind", 1): 5.0, ("Wind", 2): 2.5},
    )
    c = fe.scenario_curtailment(model)
    assert (c["n_scenarios"], c["scenarios_with_curtailment"]) == (3, 2)
    assert c["max_total_mw"] == pytest.approx(2.5)          # scenario 2: 5 − 2.5
    (unit,) = c["by_unit"]
    assert unit["unit"] == "Wind" and unit["max_mw"] == pytest.approx(2.5)
    assert unit["mean_mw"] == pytest.approx((1.0 + 2.5) / 2)
