"""
Flexibility options (backend/flex_options.py): several corrective dispatches,
each the solution of a strategy, scored the same way; the operator chooses.

Pure parts (the cost of an action, which options are shown) run without a
solver; the end-to-end checks on the demo grid need IPOPT.
"""

import shutil

import pytest

pytest.importorskip("pyomo")

import flex_options as fo  # noqa: E402


def _option(sid="least_disturbance", p=None, p0=None, q=None, q0=None, pext=None):
    p = p or {"A": 10.0, "B": 5.0}
    p0 = p0 or {"A": 10.0, "B": 5.0}
    return fo.Option(id=sid, label=fo.STRATEGIES[sid][0], how="", p=p, q=q or {g: 0.0 for g in p},
                     p0=p0, q0=q0 or {g: 0.0 for g in p}, pext=pext or {"grid": 50.0}, qext={"grid": 0.0},
                     v={0: 1.0}, max_loading_pct=None, max_loading_branch=None)


PRICES = fo.Prices(up={"A": 60.0, "B": 30.0}, down={"A": 25.0, "B": None}, q={"A": 3.0, "B": 3.0},
                   import_={"grid": 60.0}, band={"grid": 0.4}, penalty={"grid": 10.0})


class TestActionCost:
    def test_doing_nothing_costs_nothing(self):
        assert fo.action_cost(_option(), PRICES, {"grid": 50.0}, {"grid": 0.0})["total"] == 0

    def test_each_part(self):
        moved = _option(p={"A": 12.0, "B": 3.0}, q={"A": 2.0, "B": 0.0}, pext={"grid": 50.0})
        cost = fo.action_cost(moved, PRICES, {"grid": 50.0}, {"grid": 0.0})
        # A +2 MW at 60, B −2 MW unpriced (None), 2 Mvar at 3
        assert cost["redispatch"] == 120.0 and cost["reactive"] == 6.0 and cost["total"] == 126.0

    def test_less_import_is_a_saving(self):
        moved = _option(p={"A": 12.0, "B": 5.0}, pext={"grid": 48.0})
        cost = fo.action_cost(moved, PRICES, {"grid": 50.0}, {"grid": 0.0})
        assert cost["import"] == -120.0 and cost["total"] == 0.0     # +2 MW at 60 − 2 MW import at 60


class TestSelect:
    def test_a_duplicate_is_dropped_and_said_why(self):
        a, b = _option("least_disturbance"), _option("cheapest")
        kept, dropped = fo.select([(a, {"total": 1.0}), (b, {"total": 1.0})])
        assert [o.id for o, _ in kept] == ["least_disturbance"]
        assert dropped[0]["id"] == "cheapest" and "same dispatch" in dropped[0]["reason"]
        assert dropped[0]["same_as"] == "least_disturbance"           # the kept one says "also Cheapest"

    def test_an_option_no_better_on_any_count_is_dropped(self):
        first = _option("least_disturbance", p={"A": 11.0, "B": 5.0})
        worse = _option("least_import", p={"A": 13.0, "B": 7.0}, pext={"grid": 50.0})
        better = _option("cheapest", p={"A": 10.5, "B": 5.0}, pext={"grid": 49.0})
        kept, dropped = fo.select([(first, {"total": 60.0}), (better, {"total": 30.0}), (worse, {"total": 300.0})])
        assert [o.id for o, _ in kept] == ["least_disturbance", "cheapest"]
        assert dropped[0]["id"] == "least_import" and "at least as good" in dropped[0]["reason"]

    def test_least_disturbance_always_stays(self):
        first = _option("least_disturbance", p={"A": 13.0, "B": 7.0})
        better = _option("cheapest", p={"A": 10.5, "B": 5.0})
        kept, _ = fo.select([(first, {"total": 300.0}), (better, {"total": 30.0})])
        assert "least_disturbance" in [o.id for o, _ in kept]

    @staticmethod
    def headroom(p0):
        """Far from the others, and the only one with a highest loading below the rest."""
        return (_option("most_headroom", p={"A": 11, "B": 8, "C": 8}, p0=p0, pext={"grid": 43.0})._replace(
            max_loading_pct=60.0), {"total": 80.0})

    def test_beyond_the_limit_the_closest_goes_and_says_so(self):
        p0 = {"A": 10.0, "B": 5.0, "C": 5.0}

        def option(sid, a, b, c, cost, imported):        # none dominated
            return (_option(sid, p={"A": a, "B": b, "C": c}, p0=p0, pext={"grid": imported}), {"total": cost})
        six = [option("least_disturbance", 12, 6, 6, 50.0, 46.0), option("cheapest", 10, 5, 9, 20.0, 46.0),
               option("fewest_units", 13, 5, 5, 40.0, 47.0), option("without_main_unit", 10, 6.1, 7.9, 45.0, 45.5),
               option("least_import", 10, 10, 10, 30.0, 40.0), self.headroom(p0)]
        kept, dropped = fo.select(six)
        assert len(kept) == fo.MAX_OPTIONS and {"least_disturbance", "cheapest"} <= {o.id for o, _ in kept}
        assert [d["id"] for d in dropped] == ["without_main_unit"]          # 2.2 MW from Cheapest
        assert "closest to Cheapest (2.2 MW apart)" in dropped[0]["reason"]

    def test_an_option_best_at_something_is_never_the_one_capped(self):
        p0 = {"A": 10.0, "B": 5.0, "C": 5.0}

        def option(sid, a, b, c, cost, imported):
            return (_option(sid, p={"A": a, "B": b, "C": c}, p0=p0, pext={"grid": imported}), {"total": cost})
        # fewest_units is nearest to least disturbance, but the only one calling a single unit.
        six = [option("least_disturbance", 12, 6, 6, 50.0, 46.0), option("cheapest", 10, 5, 9, 20.0, 46.0),
               option("fewest_units", 13.5, 5, 5, 40.0, 47.0), option("without_main_unit", 10, 7.9, 6.1, 45.0, 45.5),
               option("least_import", 10, 10, 10, 30.0, 40.0), self.headroom(p0)]
        kept, dropped = fo.select(six)
        assert {"fewest_units", "most_headroom"} <= {o.id for o, _ in kept}
        assert [d["id"] for d in dropped] == ["without_main_unit"]

    def test_load_reductions_are_priced_and_tell_options_apart(self):
        prices = PRICES._replace(dr={7: 150.0}, relief={7: 3000.0})
        base = _option("cheapest", p={"A": 12.0, "B": 5.0})
        with_dr = base._replace(id="demand_response", dr={7: 2.0}, relief={7: 0.5})
        cost = fo.action_cost(with_dr, prices, {"grid": 50.0}, {"grid": 0.0})
        assert cost["demand_response"] == 300.0 and cost["load_relief"] == 1500.0
        assert "demand_response" not in fo.action_cost(base, prices, {"grid": 50.0}, {"grid": 0.0})
        assert with_dr.load_reduction == 2.5 and base.load_reduction == 0
        assert not fo._same(base, with_dr) and fo._same(with_dr, with_dr._replace(id="x"))
        assert fo._key(with_dr, cost)[-1] == 2.5            # less load reduced is better, as fewer MW moved

    def test_the_last_strategy_is_demand_response(self):
        assert list(fo.STRATEGY_FUNCTIONS)[-1] == "demand_response"
        assert fo.NEAR_OPTIMAL_METHODS == ("hsj", "unit_ranges", "objectives", "random")

    def test_moved_ignores_solver_noise(self):
        assert _option(p={"A": 10.000001, "B": 5.5}).moved == {"B": 0.5}


class TestNearOptimalBudget:
    def test_a_margin_above_the_cheapest(self):
        assert fo.near_optimal_budget(30.0, 10) == pytest.approx(33.0)

    def test_a_floor_when_the_cheapest_costs_about_nothing_or_saves(self):
        assert fo.near_optimal_budget(0.0, 10) == fo.NEAR_OPTIMAL_FLOOR_EUR_H
        assert fo.near_optimal_budget(-80.0, 10) == pytest.approx(-72.0)        # 10 % of a saving, still a budget

    def test_the_schedules_own_exchange_charge_is_the_constant(self):
        assert fo._exchange_before(PRICES, {"grid": 50.0}, {"grid": 30.0}) == pytest.approx(10.0 * (30 - 0.4 * 50))


class TestInstructions:
    """What an operator would have to issue: active power; the set-point of a
    unit that holds a voltage; the reactive output of one that does not."""

    def _option(self, **kw):
        base = dict(p={"V1": 10.0, "F1": 5.0}, p0={"V1": 10.0, "F1": 5.0}, q={"V1": 8.0, "F1": 1.0},
                    q0={"V1": 2.0, "F1": 1.0}, v={1: 1.02, 2: 1.0}, v0={1: 1.02, 2: 1.0}, vc_bus={"V1": 1})
        base.update(kw)
        return fo.Option(id="least_disturbance", label="", how="", pext={"g": 0.0}, qext={"g": 0.0},
                         max_loading_pct=None, max_loading_branch=None, **base)

    def test_a_held_voltage_moving_reactive_output_is_no_instruction(self):
        assert self._option().instructions == {}                         # V1's Q moved by 6 Mvar on its own

    def test_a_set_point_change_is(self):
        assert self._option(v={1: 1.035, 2: 1.0}).instructions == {"V1": "V"}

    def test_active_power_and_reactive_output_of_other_units(self):
        moved = self._option(p={"V1": 12.0, "F1": 5.0}, q={"V1": 8.0, "F1": 3.0}, v={1: 1.035, 2: 1.0})
        assert moved.instructions == {"V1": "P+V", "F1": "Q"}

    def test_without_voltages_known_reactive_output_counts(self):
        assert self._option(v0=None).instructions == {"V1": "Q"}


# ---------------------------------------------------------------------------
# End to end on the demo grid
# ---------------------------------------------------------------------------

needs_ipopt = pytest.mark.skipif(shutil.which("ipopt") is None, reason="IPOPT not installed")
TS = "2000-01-03 18:00:00"


@pytest.fixture
def client(tmp_path, monkeypatch):
    pytest.importorskip("fastapi.testclient")
    import main_backend as mb
    from fastapi.testclient import TestClient
    systems, data = tmp_path / "systems", tmp_path / "data_files"
    systems.mkdir()
    data.mkdir()
    monkeypatch.setenv("GRID_PROFILE", "pglib_case14")
    monkeypatch.setattr(mb, "SYSTEMS_DIR", str(systems))
    monkeypatch.setattr(mb, "DATA_FILES_DIR", str(data))
    for name in ("_LAST_UPLOAD_SENTINEL", "_LAST_TIMESERIES_SENTINEL", "_LAST_FORECAST_SENTINEL",
                 "_PROPOSALS_FILE", "_ASSUMPTIONS_FILE", "_SCENARIO_FILE"):
        monkeypatch.setattr(mb, name, str(tmp_path / f"{name}.json"))
    saved = dict(mb.app_data)
    with TestClient(mb.app) as c:
        yield c
    mb.app_data.clear()
    mb.app_data.update(saved)


def _options(client, **extra):
    r = client.post("/api/flexibility/optimize", json={"timestamp": TS, "options": True, **extra})
    assert r.status_code == 200, r.text
    return r.json()


@needs_ipopt
def test_options_when_the_grid_needs_action(client):
    out = _options(client, load_scaling_factor=1.8)
    assert out["status"] == "options" and out["mode"] == "options"
    ids = [o["id"] for o in out["options"]]
    assert ids[0] == "least_disturbance" and set(ids) <= set(fo.STRATEGIES) and len(ids) <= fo.MAX_OPTIONS
    lo, hi = out["opf_vm_lower_used"], out["opf_vm_upper_used"]
    for o in out["options"]:
        assert lo - 1e-4 <= o["min_vm_pu"] and o["max_vm_pu"] <= hi + 1e-4    # every option meets the limits
        assert abs(sum(o["cost_breakdown_eur_per_h"].values()) - o["cost_eur_per_h"]) < 0.05
        assert o["above_cheapest_eur_per_h"] >= -0.05                         # nothing beats the cheapest
        assert o["dispatch"] and all("dp_mw" in row for row in o["dispatch"])
    assert out["costs"]["currency"] == "EUR" and out["costs"]["prices_from"]
    assert all(entry["reason"] for entry in out["not_offered"])


@needs_ipopt
def test_no_action_needed_is_one_answer_not_a_menu(client):
    out = _options(client, load_scaling_factor=1.0)
    assert out["status"] == "no_action_needed" and out["options"] == []


@needs_ipopt
def test_the_default_dispatch_is_unchanged(client):
    r = client.post("/api/flexibility/optimize", json={"timestamp": TS, "load_scaling_factor": 1.8})
    out = r.json()
    assert out["status"] == "optimal" and out["activated_resources"] and "options" not in out


@needs_ipopt
def test_costs_follow_the_cost_table(client):
    units = [u["unit"] for u in client.get("/api/costs").json()["units"] if u["unit"] != "External grid"]
    r = client.post("/api/costs", json={"changes": [{"unit": u, "q_eur_per_mvarh": 0} for u in units]})
    assert r.status_code == 200, r.text
    out = _options(client, load_scaling_factor=1.8)
    first = out["options"][0]
    assert first["cost_breakdown_eur_per_h"]["reactive"] == 0.0
    assert "operator" in out["costs"]["prices_from"]
