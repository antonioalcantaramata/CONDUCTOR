"""
Beyond unit redispatch, more alternatives, and options over time — on the
default demo (PGLib IEEE 30-bus):

- demand response and load relief: prices per load in the cost table, the
  "With demand response" option (offered only when it uses it), the last
  resort when redispatch alone cannot clear the violations, load reduced after
  an outage in an N-1 corrective plan;
- switching that may help, checked by power flow;
- near-optimal alternatives from several methods, with each unit's range;
- carrying a chosen option forward: the assessment, a single outage and the
  N-1 screen with it applied (held at a later moment), the next corrective
  action from it, holding it over the next hours, and keeping it across a
  restart;
- one corrective plan across a window, and the cost of N-1 security over one.

The backend runs in-process with its storage in a temporary folder. The OPF
checks need IPOPT.
"""

import os
import shutil

import pytest

pytest.importorskip("fastapi.testclient")

import flex_options as fo  # noqa: E402
import main_backend as mb  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

needs_ipopt = pytest.mark.skipif(shutil.which("ipopt") is None, reason="IPOPT not installed")
PEAK = "2000-01-06 20:00:00"
EVENING = "2000-01-06 18:00:00"


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    store = tmp_path_factory.mktemp("beyond")
    patch = pytest.MonkeyPatch()
    patch.delenv("GRID_PROFILE", raising=False)                      # the default demo
    patch.setattr(mb, "SYSTEMS_DIR", str(store / "systems"))
    patch.setattr(mb, "DATA_FILES_DIR", str(store / "data_files"))
    for name in ("_LAST_UPLOAD_SENTINEL", "_LAST_TIMESERIES_SENTINEL", "_LAST_FORECAST_SENTINEL",
                 "_PROPOSALS_FILE", "_ASSUMPTIONS_FILE", "_SCENARIO_FILE"):
        patch.setattr(mb, name, str(store / f"{name}.json"))
    saved = dict(mb.app_data)
    with TestClient(mb.app) as c:
        yield c
    mb.app_data.clear()
    mb.app_data.update(saved)
    patch.undo()


def _ok(r):
    assert r.status_code == 200, r.text
    return r.json()


def _flex(client, **body):
    return _ok(client.post("/api/flexibility/optimize", json=body))


@pytest.fixture(scope="module")
def evening_cheapest(client):
    """The cheapest option at 18:00, to carry forward."""
    if shutil.which("ipopt") is None:
        pytest.skip("IPOPT not installed")
    out = _flex(client, timestamp=EVENING, options=True)
    return next(o for o in out["options"] if o["id"] == "cheapest")


# ---------------------------------------------------------------------------
# Demand response and load relief: the prices
# ---------------------------------------------------------------------------

def test_every_load_has_synthetic_demand_response_and_relief_prices(client):
    costs = _ok(client.get("/api/costs"))
    loads = {row["unit"]: row for row in costs["loads"]}
    assert len(loads) == 21 and loads["Load 20"]["bus_name"] == "Bus_29"
    assert {(r["dr_share_pct"], r["dr_eur_per_mwh"], r["relief_eur_per_mwh"]) for r in loads.values()} == {
        (10.0, 150.0, 3000.0)}
    assert all(set(r["source"].values()) == {"synthetic"} for r in loads.values())
    assert all(u["kind"] != "load" for u in costs["units"])               # the units' table is as it was
    assert "demand response" in costs["loads_rule"]


def test_a_load_is_changed_by_name_or_by_its_bus_and_back(client):
    try:
        out = _ok(client.post("/api/costs", json={"changes": [{"unit": "Bus_29", "dr_eur_per_mwh": 90},
                                                              {"unit": "Load 3", "dr_share_pct": 0}]}))
        loads = {r["unit"]: r for r in out["loads"]}
        assert loads["Load 20"]["dr_eur_per_mwh"] == 90 and loads["Load 20"]["source"]["dr_eur_per_mwh"] == "operator"
        assert loads["Load 3"]["dr_share_pct"] == 0
        for wrong, says in (({"unit": "Load 20", "up_eur_per_mwh": 5}, "units only"),
                            ({"unit": "Gen_bus1", "dr_eur_per_mwh": 5}, "loads only"),
                            ({"unit": "Load 20", "dr_share_pct": 150}, "between"),
                            ({"unit": "Load 99", "dr_eur_per_mwh": 5}, "no unit or load")):
            r = client.post("/api/costs", json={"changes": [wrong]})
            assert r.status_code == 400 and says in r.json()["detail"], (wrong, r.json())
    finally:
        _ok(client.post("/api/costs", json={"reset": True}))


# ---------------------------------------------------------------------------
# The demand-response option and the last resort
# ---------------------------------------------------------------------------

@needs_ipopt
def test_demand_response_is_offered_only_when_it_is_used(client):
    out = _flex(client, timestamp=PEAK, options=True)
    why = next(n for n in out["not_offered"] if n["id"] == "demand_response")["reason"]
    assert "redispatching the units is cheaper" in why
    assert all("load_actions" not in o for o in out["options"])          # nothing else touches a load
    # Demand response far cheaper than redispatch: now it is worth it.
    changes = [{"unit": f"Load {i}", "dr_eur_per_mwh": 0.5} for i in range(21)]
    try:
        _ok(client.post("/api/costs", json={"changes": changes}))
        out = _flex(client, timestamp=PEAK, options=True)
        by_id = {o["id"]: o for o in out["options"]}
        assert "demand_response" in by_id, out["not_offered"]
        dr = by_id["demand_response"]
        assert dr["load_reduction_mw"] > 0 and dr["load_relief_mw"] == 0 and dr["load_actions"]
        assert dr["cost_breakdown_eur_per_h"]["demand_response"] > 0
        assert dr["cost_eur_per_h"] < by_id["cheapest"]["cost_eur_per_h"]
        assert dr["power_flow_check"]["confirmed"]
        assert all(a["demand_response_mw"] <= 0.1 * 200 for a in dr["load_actions"])
    finally:
        _ok(client.post("/api/costs", json={"reset": True}))


@pytest.fixture(scope="module")
def stressed(client):
    """1.25 × the evening peak's load: no redispatch of the units alone clears it."""
    if shutil.which("ipopt") is None:
        pytest.skip("IPOPT not installed")
    return _flex(client, timestamp=PEAK, load_scaling_factor=1.25)


def test_when_redispatch_fails_the_last_resort_reduces_the_least_load(stressed):
    assert stressed["feasible"] is False and stressed["status"] != "optimal"       # the failure stays the answer
    rescue = stressed["last_resort"]
    assert rescue["set"] == "last_resort" and rescue["redispatch_alone"]["clears_violations"] is False
    least = next(o for o in rescue["options"] if o["id"] == "last_resort_least")
    assert least["power_flow_check"]["confirmed"] and least["load_reduction_mw"] > 0
    assert least["load_relief_mw"] == 0                     # demand response is enough: no involuntary relief
    assert {a["bus"] for a in least["load_actions"]} <= {f"Bus_{b}" for b in range(30)}
    assert least["ref"].startswith(rescue["set_id"] + ":")


@needs_ipopt
def test_asked_for_options_the_last_resort_is_the_answer(client):
    out = _flex(client, timestamp=PEAK, load_scaling_factor=1.25, options=True)
    assert out["status"] == "options" and out["set"] == "last_resort"
    assert "Redispatching the units alone cannot" in out["message"]
    least = next(o for o in out["options"] if o["id"] == "last_resort_least")
    # Its load reductions go with it: the N-1 screen applies them too.
    screen = _ok(client.post("/api/contingency/simulate_all", json={"option": least["ref"]}))
    assert screen["option"]["load_reduction_mw"] == pytest.approx(least["load_reduction_mw"], abs=1e-3)


# ---------------------------------------------------------------------------
# Near-optimal alternatives: several methods, each unit's range
# ---------------------------------------------------------------------------

@needs_ipopt
def test_near_optimal_alternatives_come_from_several_methods_with_ranges(client):
    out = _flex(client, timestamp=PEAK, options=True, near_optimal=True)
    alts = [o for o in out["options"] if o["id"].startswith("near_optimal")]
    assert 3 <= len(alts) <= fo.MAX_ALTERNATIVES
    assert all(o["cost_eur_per_h"] <= out["budget_eur_per_h"] + 0.01 for o in alts)
    assert len({o["found_by"] for o in alts}) >= 2 and all(o["found_by"] in o["how"] for o in alts)
    for a, b in ((x, y) for i, x in enumerate(alts) for y in alts[i + 1:]):        # each shown differs
        pa, pb = ({r["unit"]: r["p_mw_after"] for r in o["dispatch"]} for o in (a, b))
        assert sum(abs(pa[u] - pb.get(u, 0)) for u in pa) >= fo.DIVERSE_MW
    ranges = {r["unit"]: r for r in out["near_optimal_ranges"]}
    assert [r["unit"] for r in out["near_optimal_ranges"]] == ["Gen_bus1", "Gen_bus4", "Gen_bus7", "Gen_bus10",
                                                               "Gen_bus12"]
    for r in ranges.values():
        assert r["min_mw"] - 1e-3 <= r["cheapest_mw"] <= r["max_mw"] + 1e-3
    assert out["near_optimal_methods"] == list(fo.NEAR_OPTIMAL_METHODS)


@needs_ipopt
def test_near_optimal_methods_and_count_are_the_operators(client):
    out = _flex(client, timestamp=PEAK, options=True, near_optimal=True, near_optimal_methods=["hsj"],
                near_optimal_count=2)
    alts = [o for o in out["options"] if o["id"].startswith("near_optimal")]
    assert 1 <= len(alts) <= 2 and all("Hop-Skip-Jump" in o["found_by"] for o in alts)
    assert "near_optimal_ranges" not in out
    r = client.post("/api/flexibility/optimize", json={"timestamp": PEAK, "options": True, "near_optimal": True,
                                                       "near_optimal_methods": ["guess"]})
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# N-1: load reduced after an outage, switching
# ---------------------------------------------------------------------------

@needs_ipopt
def test_a_corrective_plan_may_reduce_load_after_an_outage(client):
    with_loads = _flex(client, timestamp=PEAK, n1="corrective")
    n1 = next(o for o in with_loads["options"] if o["id"] == "n1_corrective")["n1"]
    assert n1["load_reduction_allowed"] is True and n1["secured_with_load_reduction"] >= 1
    for step in n1["corrective_plan"]:
        if step.get("loads"):
            assert step["checked_by_power_flow"], step                     # load reduced, and it holds
            assert all(a["demand_response_mw"] >= 0 and a["load_relief_mw"] >= 0 for a in step["loads"])
    without = _flex(client, timestamp=PEAK, n1="corrective", n1_load_reduction=False)
    n1_without = next(o for o in without["options"] if o["id"] == "n1_corrective")["n1"]
    assert all("loads" not in p for p in n1_without["corrective_plan"])
    assert n1_without["secured"] < n1["secured"]


@needs_ipopt
def test_switching_is_offered_where_a_dispatch_cannot_secure_an_outage(client):
    out = _flex(client, timestamp=PEAK, n1="preventive")
    n1 = next(o for o in out["options"] if o["id"] == "n1_preventive")["n1"]
    candidates = [s for x in n1["not_secured"] for s in x.get("switching") or []]
    assert candidates and all(s["checked_by_power_flow"] and s["open"] for s in candidates)
    assert all(s["clears_all"] or s["relieves_pct"] >= mb._SWITCH_GAIN_PCT or s["remaining"] for s in candidates)
    for x in n1["not_secured"]:                                   # at most three, those that clear everything first
        found = x.get("switching") or []
        assert len(found) <= mb._SWITCH_SHOWN
        assert [s["clears_all"] for s in found] == sorted((s["clears_all"] for s in found), reverse=True)


# ---------------------------------------------------------------------------
# Carrying a chosen option forward
# ---------------------------------------------------------------------------

def test_the_assessment_with_an_option_at_its_moment_and_held_later(client, evening_cheapest):
    ref = evening_cheapest["ref"]
    own = _ok(client.post("/api/grid/rsa", json={"option": ref}))
    assert own["timestamp"] == EVENING and own["option"]["held"] is False
    assert own["total_violations"] == 0              # an option on a limit is within it (no float noise)
    later = _ok(client.post("/api/grid/rsa", json={"option": ref, "timestamp": "2000-01-06 19:00:00"}))
    assert later["option"]["held"] is True and later["option"]["applied_at"] == "2000-01-06 19:00:00"
    assert later["option"]["held_instructions"] == evening_cheapest["instructions"]
    r = client.post("/api/grid/rsa", json={"option": ref, "timestamp": "2000-01-06 17:00:00"})
    assert r.status_code == 400 and "before it was computed" in r.json()["detail"]


def test_a_single_outage_and_the_n1_screen_with_an_option(client, evening_cheapest):
    ref = evening_cheapest["ref"]
    one = _ok(client.post("/api/contingency/simulate", json={"option": ref, "element_type": "line",
                                                             "element_index": 31}))
    assert one["option"]["ref"] == ref
    sweep = _ok(client.post("/api/contingency/simulate_all", json={"option": ref,
                                                                   "timestamp": "2000-01-06 19:00:00"}))
    assert sweep["option"]["held"] is True and sweep["timestamp"] == "2000-01-06 19:00:00"
    r = client.post("/api/proposals/compare", json={"proposal": "P1", "option": ref})
    assert r.status_code == 422 and "option is not applied by this study" in r.text   # refused, not ignored


@needs_ipopt
def test_the_next_corrective_action_starts_from_the_option(client, evening_cheapest):
    ref = evening_cheapest["ref"]
    out = _flex(client, timestamp=PEAK, from_option=ref, options=True)
    assert out["from_option"]["ref"] == ref and out["from_option"]["held"] is True
    plain = _flex(client, timestamp=PEAK, options=True)
    held_cheapest = next(o for o in out["options"] if o["id"] == "cheapest")
    plain_cheapest = next(o for o in plain["options"] if o["id"] == "cheapest")
    assert held_cheapest["cost_eur_per_h"] != plain_cheapest["cost_eur_per_h"]    # another starting point
    after = _flex(client, timestamp="2000-01-03 12:00:00", options=True)
    outage = _ok(client.post("/api/contingency/optimize", json={
        "timestamp": "2000-01-03 12:00:00", "element_type": "line", "element_index": 36, "options": True}))
    r = client.post("/api/flexibility/optimize", json={"from_option": outage["options"][0]["ref"]})
    assert r.status_code == 400 and "optimize_contingency" in r.json()["detail"]
    assert after["status"] in ("options", "no_action_needed")


def test_holding_an_option_over_the_next_hours(client, evening_cheapest):
    out = _ok(client.post("/api/flexibility/hold", json={"option": evening_cheapest["ref"], "hours": 2}))
    hours = out["hours"]
    assert hours[0]["timestamp"] == EVENING and hours[0]["held"] is False and all(h["held"] for h in hours[1:])
    assert hours[-1]["timestamp"] <= "2000-01-06 20:00:00" and len(hours) == 9          # every 15 minutes
    first_bad = next((h["timestamp"] for h in hours if h["with_option"]["violations"]), None)
    assert out["first_failure"] == first_bad
    if first_bad:
        assert out["holds_until"] is None or out["holds_until"] < first_bad
        assert "from_option=" in out["message"]
    with_n1 = _ok(client.post("/api/flexibility/hold", json={"option": evening_cheapest["ref"], "hours": 1,
                                                             "n1": True}))
    assert all(h["with_option"]["n1_outages_tested"] == 41 for h in with_n1["hours"])
    assert client.post("/api/flexibility/hold", json={"hours": 2}).status_code == 400


def test_options_are_kept_across_a_restart(client, evening_cheapest):
    path = mb._option_sets_path()
    assert os.path.exists(path) and os.path.dirname(path) == os.path.dirname(mb._PROPOSALS_FILE)
    saved = dict(mb.app_data["option_sets"])
    mb.app_data["option_sets"] = {}
    mb._restore_option_sets()
    assert set(mb.app_data["option_sets"]) == set(saved)
    restored = mb._resolve_option(evening_cheapest["ref"])[1]
    original = saved[evening_cheapest["ref"].split(":")[0]]["options"]["cheapest"]
    assert restored == original                               # every field, its keys back to numbers


# ---------------------------------------------------------------------------
# Over a window: one plan, and the cost of security
# ---------------------------------------------------------------------------

@needs_ipopt
def test_a_plan_keeps_instructions_steady_across_the_peak(client):
    out = _ok(client.post("/api/flexibility/plan", json={"start": "2000-01-06 17:00:00",
                                                         "end": "2000-01-06 22:00:00", "steadiness": 2}))
    plan, alone = out["plan"], out["each_moment_alone"]
    assert out["moments"][0] == "2000-01-06 17:00:00" and len(out["moments"]) == 6
    assert sum(out["hours_each"]) == pytest.approx(5.0)          # 17:00 to 22:00
    assert plan["instructions_issued"] < alone["instructions_issued"]
    assert all(step["power_flow_check"] for step in plan["steps"])
    assert out["units"] == ["Gen_bus1", "Gen_bus4", "Gen_bus7", "Gen_bus10", "Gen_bus12"]
    ramped = _ok(client.post("/api/flexibility/plan", json={"start": "2000-01-06 17:00:00", "end": "2000-01-06 22:00:00",
                                                            "objective": "cheapest", "ramp_pct_per_h": 30}))
    assert ramped["objective"] == "cheapest" and ramped["ramp_pct_per_h"] == 30
    loose = _ok(client.post("/api/flexibility/plan", json={"start": "2000-01-06 17:00:00",
                                                           "end": "2000-01-06 22:00:00", "steadiness": 0}))
    assert loose["plan"]["instructions_issued"] == loose["each_moment_alone"]["instructions_issued"]
    r = client.post("/api/flexibility/plan", json={"start": "2000-01-06 17:00:00", "end": "2000-01-06 16:00:00"})
    assert r.status_code == 400


@needs_ipopt
def test_the_cost_of_security_over_a_window(client, monkeypatch):
    window = {"start": "2000-01-06 23:00:00", "end": "2000-01-07 00:00:00", "moments": 2}
    monkeypatch.setenv("CONDUCTOR_STUDY_WORKERS", "1")                  # in turn ...
    alone = _ok(client.post("/api/flexibility/security_over_time", json=window))
    monkeypatch.delenv("CONDUCTOR_STUDY_WORKERS")                       # ... and in parallel, where it can
    out = _ok(client.post("/api/flexibility/security_over_time", json={**window, "progress_id": "t1"}))
    assert alone["computed_in"]["workers"] == 1
    assert out["computed_in"]["workers"] == mb._study_workers(2)
    assert [r["preventive"]["cost_eur_per_h"] for r in out["moments"]] == \
        [r["preventive"]["cost_eur_per_h"] for r in alone["moments"]]            # the same answer either way
    progress = _ok(client.get("/api/progress/t1"))
    assert progress["known"] and progress["done"] == progress["total"] == 2
    assert _ok(client.get("/api/progress/never"))["known"] is False
    rows = out["moments"]
    for row in rows:                                     # a worker's options are kept here, under their refs
        screened = _ok(client.post("/api/contingency/simulate_all", json={"option": row["preventive"]["ref"]}))
        assert screened["total_outages_causing_violations"] == row["preventive"]["n1_outages_with_violations"]
    assert [r["timestamp"] for r in rows] == ["2000-01-06 23:00:00", "2000-01-07 00:00:00"]
    for row in rows:
        assert row["status"] == "options" and row["preventive"]["ref"].endswith(":n1_preventive")
        assert row["preventive"]["security_cost_eur_per_h"] == pytest.approx(
            row["preventive"]["cost_eur_per_h"] - row["cheapest"]["cost_eur_per_h"], abs=0.01)
    assert out["hours_each"] == [0.5, 0.5]                # one hour, 23:00 to midnight — not two
    total = sum(r["preventive"]["security_cost_eur_per_h"] * h for r, h in zip(rows, out["hours_each"]))
    assert out["totals"]["preventive"]["security_cost_eur"] == pytest.approx(total, abs=0.05)
    assert "above the cheapest over the window" in out["message"]
    assert "in the full N-1 screen" in out["message"] and "outages_studied_by_the_opf" in out["message"]
    for row in rows:                                      # the screen and the studied set, apart
        p = row["preventive"]
        assert p["of_those_secured"] <= p["outages_studied_by_the_opf"] and "secured" not in p
        assert 0 <= p["n1_outages_with_violations"] <= row["outages_tested"]
    assert all(isinstance(r["schedule_meets_limits"], bool) for r in rows)
    assert out["schedule_breaks_limits_at"] == [r["timestamp"] for r in rows if not r["schedule_meets_limits"]]


# ---------------------------------------------------------------------------
# Pieces of the parallel run, without a solver
# ---------------------------------------------------------------------------

def test_a_study_coroutine_runs_without_an_event_loop():
    async def study():
        return 42

    assert mb._run_coroutine(study()) == 42


def test_a_workers_option_sets_are_kept_under_fresh_ids(monkeypatch):
    monkeypatch.setitem(mb.app_data, "option_sets", {"S1": {"id": "S1", "options": {}}})
    monkeypatch.setitem(mb.app_data, "option_set_counter", 1)
    renamed = mb._adopt_option_sets({"S2": {"id": "S2", "options": {"n1_preventive": None}},
                                     "S3": {"id": "S3", "options": {}}})
    assert renamed == {"S2": "S2", "S3": "S3"}
    renamed = mb._adopt_option_sets({"S2": {"id": "S2", "options": {}}})       # another worker's "S2"
    assert renamed == {"S2": "S4"} and mb.app_data["option_sets"]["S4"]["id"] == "S4"
    row = mb._renamed_refs({"cheapest": {"ref": "S2:cheapest"}, "preventive": {"ref": "S2:n1_preventive"},
                            "timestamp": "t"}, renamed)
    assert row["cheapest"]["ref"] == "S4:cheapest" and row["preventive"]["ref"] == "S4:n1_preventive"


def test_workers_one_per_moment_and_never_more_than_asked(monkeypatch):
    monkeypatch.setenv("CONDUCTOR_STUDY_WORKERS", "1")
    assert mb._study_workers(9) == 1
    monkeypatch.setenv("CONDUCTOR_STUDY_WORKERS", "3")
    assert mb._study_workers(9) in (1, 3)                 # 1 where processes cannot be forked
    assert mb._study_workers(1) == 1



def test_a_window_total_covers_the_window_and_a_ramp_the_gap():
    ticks = ["2000-01-06 16:00:00", "2000-01-06 17:00:00", "2000-01-06 18:00:00", "2000-01-06 20:00:00"]
    assert mb._gaps(ticks) == [1.0, 1.0, 2.0]
    assert mb._weights(ticks) == [0.5, 1.0, 1.5, 1.0]      # half the time to each neighbour
    assert sum(mb._weights(ticks)) == 4.0                 # 16:00 to 20:00
    assert mb._weights(ticks[:1]) == [0.0] and mb._gaps(ticks[:1]) == []


def test_a_window_keeps_its_first_and_last_moment():
    import pandas as pd
    quarter = [str(t) for t in pd.date_range("2000-01-06 16:00", "2000-01-07 02:00", freq="15min")]
    ticks = mb._ticks_within(quarter, "2000-01-06 16:00:00", 8, 6)
    assert ticks == ["2000-01-06 16:00:00", "2000-01-06 18:00:00", "2000-01-06 20:00:00",
                     "2000-01-06 22:00:00", "2000-01-07 00:00:00"]       # a round step that lands on the end
    assert mb._ticks_within(quarter, "2000-01-06 16:00:00", 8, 9)[1] == "2000-01-06 17:00:00"
    odd = mb._ticks_within(quarter, "2000-01-06 16:00:00", 7.75, 6)    # 31 steps: no round one fits
    assert len(odd) == 6 and odd[0] == "2000-01-06 16:00:00" and odd[-1] == "2000-01-06 23:45:00"
    assert mb._ticks_within(quarter, "2000-01-06 16:00:00", 1, 6) == quarter[:5]   # few enough: all of them
