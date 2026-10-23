"""
Key performance indicators (`backend/kpis.py`, `/api/kpi/evaluate`,
`/api/kpi/over_time`): the schedule against a corrective action at a moment,
and over a window.

The three indicators of the project notebook were uninformative: KPI-2 read
0 % at every hour (it compared the MW moved with the net change in total
generation, which redispatch keeps near zero), KPI-3 read 100 % whenever the
solver converged (even with nothing to prevent), and "no solution" read as
0 %. These tests pin the fixed definitions, the new indicators (N-0 and N-1
margins, deliverable flexibility, flexibility required, costs) and that a
moment without a result is unknown, never zero.

The formulas run without a solver; the studies on the default demo (PGLib
IEEE 30-bus) and on IEEE 14 need IPOPT.
"""

import shutil

import pytest

import kpis as kp

needs_ipopt = pytest.mark.skipif(shutil.which("ipopt") is None, reason="IPOPT not installed")
PEAK = "2000-01-06 20:00:00"
NIGHT = "2000-01-06 04:00:00"


# ---------------------------------------------------------------------------
# The three indicators of the project notebook, fixed
# ---------------------------------------------------------------------------

def test_kpi1_is_spare_capacity_over_supply():
    out = kp.target_demand_flexibility({"G1": 10.0, "G2": 30.0, "PV": 5.0}, {"G1": 40.0, "G2": 30.0}, 200.0)
    assert out == {"spare_mw": 30.0, "supply_mw": 200.0, "pct": 15.0}       # a unit without a maximum: left out
    assert kp.target_demand_flexibility({"G1": 10.0}, {"G1": 40.0}, 0.0)["pct"] is None


@pytest.mark.parametrize("required, moved, violations, pct, note", [
    (9.0, 36.0, 2, 25.0, None),                                    # a quarter of what moved was needed
    (0.0, 0.0, 0, 100.0, "nothing needed, nothing moved"),
    (0.0, 5.0, 1, 0.0, "reactive power or voltage set-points"),    # violations cleared without moving MW
    (0.0, 5.0, 0, 0.0, "though nothing needs to move"),
    (5.0, 0.0, 1, 0.0, "moves nothing"),
    (12.0, 10.0, 1, 100.0, None),                                  # never above 100 %
    (None, 5.0, 1, None, "redispatching the units alone cannot"),  # unknown, not 0 %
])
def test_kpi2_is_the_share_of_the_activated_flexibility_that_was_required(required, moved, violations, pct, note):
    out = kp.flexibility_utilisation(required, moved, violations=violations)
    assert out["pct"] == pct
    assert (note in out["note"]) if note else "note" not in out


@pytest.mark.parametrize("before, after, pct, note", [
    (2, 0, 100.0, None), (4, 1, 75.0, "leaves 1 of the 4"), (1, 3, 0.0, "leaves 3 of the 1"),
    (0, 0, None, "nothing to prevent"), (0, 2, None, "the action breaks 2"), (2, None, None, "could not"),
])
def test_kpi3_counts_the_violations_that_existed(before, after, pct, note):
    out = kp.prevented_violation_ratio(before, after)
    assert out["pct"] == pct
    assert (note in out["note"]) if note else "note" not in out


def test_indicators_keep_their_order_and_say_which_way_is_better():
    rows = kp.indicator_rows({"cost_eur_per_h": (0.0, 12.5), "violations": (2, 0),
                              "kpi_1_target_demand_flex_pct": (31.2, 20.7)})
    assert [r["key"] for r in rows] == ["violations", "kpi_1_target_demand_flex_pct", "cost_eur_per_h"]
    assert rows[0]["better"] == "lower" and rows[1]["better"] == "higher" and rows[2]["unit"] == "€/h"
    assert kp.flat_metrics(rows) == {"violations": 0, "kpi_1_target_demand_flex_pct": 20.7, "cost_eur_per_h": 12.5}


def test_numbers_never_read_minus_zero():
    assert kp._fmt(-0.00004, "p.u.") == "+0.000 p.u." and kp._fmt(-0.0, "pp") == "+0.0 pp"
    assert kp._fmt(None, "MW") == "—"


def _row(ts, **values):
    return {"timestamp": ts, "status": "ok", "after_is": "cheapest", "schedule": {"violations": 1},
            "indicators": kp.indicator_rows(values)}


def test_a_window_summary_finds_the_worst_moment_each_way_and_weights_the_totals():
    rows = [_row("t1", deliverable_up_mw=(None, 80.0), cost_eur_per_h=(0.0, 10.0), moved_mw=(0.0, 4.0),
                 required_mw=(None, 2.0), violations=(1, 0)),
            _row("t2", deliverable_up_mw=(None, 60.0), cost_eur_per_h=(0.0, -20.0), moved_mw=(0.0, 0.0),
                 required_mw=(None, 0.0), violations=(3, 1)),
            {"timestamp": "t3", "status": "error", "why": "no power flow"}]
    out = kp.window_summary(rows, [0.5, 1.0, 0.5])
    up, viol = out["indicators"]["deliverable_up_mw"], out["indicators"]["violations"]
    assert up["worst"] == {"timestamp": "t2", "value": 60.0} and up["best"]["timestamp"] == "t1"
    assert viol["worst"]["timestamp"] == "t2" and viol["before_max"] == 3        # lower is better here
    assert out["totals"] == {"cost_eur": -15.0, "mwh_moved": 2.0, "mwh_required": 1.0, "hours": 2.0}
    assert out["moments_failed"] == ["t3"] and out["moments_with_violations"] == ["t1", "t2"]
    message = kp.window_message(["t1", "t2", "t3"], out)
    assert "save 15.00 € net" in message and "not zero, unknown" in message


def test_a_window_result_describes_each_indicator_once():
    rows = [_row("t1", violations=(1, 0), kpi_1_target_demand_flex_pct=(30.0, 20.0)),
            {**_row("t2", violations=(0, 0)), "after_is": "schedule", "after": {"violations": 0},
             "deliverable": {"with_action": {"up_mw": 5.0}, "schedule": {"up_mw": 5.0}}}]
    out = kp.compact_moments(rows)
    assert [i["key"] for i in out["indicators"]] == ["violations", "kpi_1_target_demand_flex_pct"]
    assert out["moments"][0]["values"] == {"violations": [1, 0], "kpi_1_target_demand_flex_pct": [30.0, 20.0]}
    assert "indicators" not in out["moments"][0]
    assert out["moments"][1]["after"] == "the schedule"                        # the same state, said once
    assert out["moments"][1]["deliverable"]["schedule"] == "as with_action"


def test_a_moment_says_what_it_compares_with_every_number():
    row = {"timestamp": PEAK, "after_is": "cheapest", "after_label": "Cheapest", "after_ref": "S1:cheapest",
           "schedule": {"violations": 2, "worst": ["Bus_29 at 0.943 p.u."]},
           "indicators": kp.indicator_rows({"violations": (2, 0), "kpi_2_flex_utilization_pct": (None, 26.3)}),
           "kpi_details": {"kpi_2": {"note": "x"}}}
    text = kp.moment_message(row)
    assert "breaks 2 limit(s) (Bus_29 at 0.943 p.u.)" in text and "Cheapest (S1:cheapest)" in text
    assert "Violations (N-0) 2 → 0" in text and "KPI-2 Flexibility utilisation 26.3 %" in text and "(x)" in text
    assert "no indicators" in kp.moment_message({"timestamp": "t", "status": "error", "why": "w"})


# ---------------------------------------------------------------------------
# The studies, on the default demo
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def demo(tmp_path_factory):
    pytest.importorskip("fastapi.testclient")
    if shutil.which("ipopt") is None:
        pytest.skip("IPOPT not installed")
    import main_backend as mb
    from fastapi.testclient import TestClient
    store = tmp_path_factory.mktemp("kpis")
    patch = pytest.MonkeyPatch()
    patch.delenv("GRID_PROFILE", raising=False)
    patch.setattr(mb, "SYSTEMS_DIR", str(store / "systems"))
    patch.setattr(mb, "DATA_FILES_DIR", str(store / "data_files"))
    for name in ("_LAST_UPLOAD_SENTINEL", "_LAST_TIMESERIES_SENTINEL", "_LAST_FORECAST_SENTINEL",
                 "_PROPOSALS_FILE", "_ASSUMPTIONS_FILE", "_SCENARIO_FILE"):
        patch.setattr(mb, name, str(store / f"{name}.json"))
    saved = dict(mb.app_data)
    with TestClient(mb.app) as client:
        yield client, mb
    mb.app_data.clear()
    mb.app_data.update(saved)
    patch.undo()


def _ok(r):
    assert r.status_code == 200, r.text
    return r.json()


def _values(result):
    return {r["key"]: (r["before"], r["after"]) for r in result["indicators"]}


@pytest.fixture(scope="module")
def peak(demo):
    client, _ = demo
    return _ok(client.post("/api/kpi/evaluate", json={"timestamp": PEAK}))


@needs_ipopt
def test_a_quiet_hour_needs_nothing_and_says_so(demo):
    client, _ = demo
    out = _ok(client.post("/api/kpi/evaluate", json={"timestamp": NIGHT}))
    v = _values(out)
    assert out["after_is"] == "schedule" and v["violations"] == (0, 0)
    assert v["kpi_2_flex_utilization_pct"][1] == 100.0 and v["kpi_3_prevented_violation_ratio_pct"][1] is None
    assert "nothing to prevent" in out["kpi_details"]["kpi_3"]["note"]
    assert v["required_mw"][1] == 0.0 and v["moved_mw"][1] == 0.0 and v["cost_eur_per_h"] == (0.0, 0.0)
    units = out["deliverable"]["with_action"]["units"]
    assert set(units) == {"Gen_bus1", "Gen_bus4", "Gen_bus7", "Gen_bus10", "Gen_bus12"}
    for u in units.values():                         # a unit can deliver at most the capacity it has spare
        assert u["up_mw"] is not None and 0 <= u["up_mw"] <= u["spare_mw"] + 1e-3 and u["down_mw"] >= 0
    assert v["deliverable_up_mw"][1] <= v["spare_up_mw"][1] + 1e-3
    assert out["deliverable"]["schedule"] == out["deliverable"]["with_action"]


@needs_ipopt
def test_the_peak_against_its_cheapest_action(demo, peak):
    client, _ = demo
    v = _values(peak)
    assert peak["after_is"] == "cheapest" and peak["after_ref"].endswith(":cheapest")
    assert v["violations"][0] > 0 and v["violations"][1] == 0
    assert v["kpi_3_prevented_violation_ratio_pct"][1] == 100.0
    assert v["loading_headroom_pct"][0] < 0 <= v["loading_headroom_pct"][1]
    # KPI-2: the fewest MW (an L1 minimum) over what the action moves.
    required, moved = v["required_mw"][1], v["moved_mw"][1]
    assert 0 < required <= moved and v["kpi_2_flex_utilization_pct"][1] == pytest.approx(
        100 * required / moved, abs=0.05)
    # The same action the options study offers: what it moves and what it costs.
    options = _ok(client.post("/api/flexibility/optimize", json={"timestamp": PEAK, "options": True}))["options"]
    by_id = {o["id"]: o for o in options}
    assert moved == pytest.approx(by_id["cheapest"]["moved_mw"], abs=0.05)
    assert v["cost_eur_per_h"][1] == pytest.approx(by_id["cheapest"]["cost_eur_per_h"], abs=0.01)
    assert required <= by_id["least_disturbance"]["moved_mw"] + 1e-3     # no dispatch moves fewer MW
    k1 = peak["kpi_details"]["kpi_1"]["after"]
    assert v["kpi_1_target_demand_flex_pct"][1] == pytest.approx(100 * k1["spare_mw"] / k1["supply_mw"], abs=0.01)
    assert peak["metrics"]["violations"] == 0 and "→" in peak["message"]
    assert peak["after"]["n1_outages_tested"] == peak["schedule"]["n1_outages_tested"] > 0


@needs_ipopt
def test_an_option_held_from_its_moment(demo, peak):
    client, _ = demo
    out = _ok(client.post("/api/kpi/evaluate", json={"timestamp": "2000-01-06 21:00:00",
                                                     "option": peak["after_ref"]}))
    assert out["after_is"] == "option" and out["held"] is True and out["held_since"] == PEAK
    assert out["option"]["ref"] == peak["after_ref"] and out["option"]["held_instructions"]
    assert _values(out)["cost_eur_per_h"][1] is not None                 # priced from this moment's schedule
    early = client.post("/api/kpi/evaluate", json={"timestamp": "2000-01-06 19:00:00", "option": peak["after_ref"]})
    assert early.status_code == 400 and "before it was computed" in early.json()["detail"]


@needs_ipopt
def test_an_option_computed_after_an_outage_is_refused_with_the_reason(demo, peak, monkeypatch):
    client, mb = demo
    sid, oid = peak["after_ref"].split(":")
    sets = mb.app_data["option_sets"]
    monkeypatch.setitem(sets, "S990", {**sets[sid], "id": "S990", "outage": {"element": "Bus_1 -> Bus_3 [L2]"}})
    r = client.post("/api/kpi/evaluate", json={"timestamp": PEAK, "option": f"S990:{oid}"})
    assert r.status_code == 400 and "after the outage of Bus_1 -> Bus_3 [L2]" in r.json()["detail"]


@needs_ipopt
def test_a_window_in_parallel_matches_one_moment_at_a_time(demo, monkeypatch):
    client, mb = demo
    body = {"start": "2000-01-06 22:00:00", "end": "2000-01-07 00:00:00", "moments": 3}
    out = _ok(client.post("/api/kpi/over_time", json={**body, "progress_id": "kpi-w"}))
    assert [m["timestamp"] for m in out["moments"]] == ["2000-01-06 22:00:00", "2000-01-06 23:00:00",
                                                        "2000-01-07 00:00:00"]
    assert out["values_are"] == "[as scheduled, with the action]" and out["indicators"][0]["key"] == "violations"
    assert out["summary"]["totals"]["hours"] == 2.0 and out["hours_each"] == [0.5, 1.0, 0.5]
    progress = _ok(client.get("/api/progress/kpi-w"))
    assert progress["done"] == progress["total"] == 3
    monkeypatch.setenv("CONDUCTOR_STUDY_WORKERS", "1")
    alone = _ok(client.post("/api/kpi/over_time", json=body))
    assert alone["computed_in"]["workers"] == 1
    assert [m["values"] for m in alone["moments"]] == [m["values"] for m in out["moments"]]
    single = _ok(client.post("/api/kpi/evaluate", json={"timestamp": "2000-01-06 23:00:00"}))
    assert out["moments"][1]["values"] == {r["key"]: [r["before"], r["after"]] for r in single["indicators"]}


@needs_ipopt
def test_a_moment_without_a_result_is_unknown_never_zero(demo, monkeypatch):
    client, mb = demo
    monkeypatch.setenv("CONDUCTOR_STUDY_WORKERS", "1")
    real = mb._kpi_state

    def failing(ts, req, applied=None):
        if ts.startswith("2000-01-06 23"):
            raise mb.HTTPException(status_code=503, detail="The power flow did not converge.")
        return real(ts, req, applied)
    monkeypatch.setattr(mb, "_kpi_state", failing)
    out = _ok(client.post("/api/kpi/over_time", json={"start": "2000-01-06 22:00:00", "end": "2000-01-07 00:00:00",
                                                      "moments": 3}))
    failed = out["moments"][1]
    assert failed["status"] == "error" and "did not converge" in failed["why"] and "values" not in failed
    assert out["summary"]["moments_failed"] == ["2000-01-06 23:00:00"] and "not zero, unknown" in out["message"]
    r = client.post("/api/kpi/evaluate", json={"timestamp": "2000-01-06 23:00:00"})
    assert r.status_code == 503 and "did not converge" in r.json()["detail"]


@needs_ipopt
def test_the_conditions_reach_the_options_study(demo, monkeypatch):
    client, mb = demo
    seen = []

    class Study:
        @staticmethod
        async def __wrapped__(study):
            seen.append(study)
            return {"status": "infeasible", "message": "no dispatch at all"}
    monkeypatch.setattr(mb, "optimise_flexibility", Study)
    out = _ok(client.post("/api/kpi/evaluate", json={
        "timestamp": PEAK, "fixed_setpoints": {"Gen_bus1": 10.0}, "hold_units": ["Gen_bus4"], "n1": "preventive",
        "load_scaling_factor": 1.1, "opf_vm_lower": 0.96}))
    study = seen[0]
    assert study.fixed_setpoints == {"Gen_bus1": 10.0} and study.hold_units == ["Gen_bus4"]
    assert study.n1 == "preventive" and study.load_scaling_factor == 1.1 and study.opf_vm_lower == 0.96
    assert study.options is True
    v = _values(out)                                   # no action found: the action's indicators are unknown
    assert out["after_is"] == "none" and "no dispatch at all" in out["why"]
    assert v["violations"][1] is None and v["kpi_2_flex_utilization_pct"][1] is None
    assert v["kpi_3_prevented_violation_ratio_pct"][1] is None and v["violations"][0] > 0


# ---------------------------------------------------------------------------
# Charts and the tools
# ---------------------------------------------------------------------------

def _result(after_is="cheapest"):
    values = {"violations": (2, 0), "vm_margin_pu": (-0.007, 0.0), "loading_headroom_pct": (-14.7, 10.0),
              "n1_outages_with_violations": (41, 29), "kpi_3_prevented_violation_ratio_pct": (None, 100.0),
              "kpi_1_target_demand_flex_pct": (31.2, 20.7), "spare_up_mw": (99.9, 65.5),
              "deliverable_up_mw": (None, 65.5), "deliverable_down_mw": (None, None),
              "required_mw": (None, 9.0), "moved_mw": (0.0, 34.4), "kpi_2_flex_utilization_pct": (None, 26.2),
              "cost_eur_per_h": (0.0, 26.48)}
    return {"timestamp": PEAK, "after_is": after_is, "after_label": "Cheapest", "after_ref": "S1:cheapest",
            "indicators": kp.indicator_rows(values),
            "kpi_details": {"kpi_2": {"pct": 26.2}, "kpi_3": {"pct": 100.0}},
            "deliverable": {"with_action": {"up_mw": 65.5, "down_mw": None, "units": {
                "G1": {"up_mw": 26.0, "down_mw": 22.0, "spare_mw": 26.0},
                "G2": {"up_mw": None, "down_mw": None, "spare_mw": 3.0}}}}}


def test_one_moment_renders_tiles_the_units_and_a_table():
    pytest.importorskip("plotly")
    from llm_agent.agent.renderers import RENDERER_MAP
    tiles, units, table = RENDERER_MAP["evaluate_kpis"](_result())
    assert len(tiles.data) == 12 and "Cheapest (S1:cheapest)" in tiles.layout.title.text
    by_title = {t.title.text.split("</b>")[0][3:]: t for t in tiles.data}
    assert by_title["Violations"].mode == "number+delta" and by_title["Violations"].delta.reference == 2
    assert by_title["Action cost"].mode == "number"                          # a cost has no "before" to beat
    assert by_title["Violations"].delta.decreasing.color != by_title["KPI-1"].delta.decreasing.color
    assert list(units.data[1].x) == [None, 26.0] and "no range within the limits" in units.layout.annotations[0].text
    assert table.data[0].cells.values[0][0] == "Violations (N-0)"


def test_a_moment_needing_nothing_shows_no_changes():
    pytest.importorskip("plotly")
    from llm_agent.agent.renderers import RENDERER_MAP
    result = _result("schedule")
    for r in result["indicators"]:
        r["before"] = r["after"]
    tiles = RENDERER_MAP["evaluate_kpis"](result)[0]
    assert all(t.mode == "number" for t in tiles.data) and "▲▼" not in tiles.layout.title.text
    assert len(RENDERER_MAP["evaluate_kpis"]({"message": "No options kept."})) == 1


def test_a_window_renders_five_charts_and_gaps_where_nothing_happened():
    pytest.importorskip("plotly")
    from llm_agent.agent.renderers import RENDERER_MAP
    rows = [_row("2000-01-06 18:00:00", kpi_2_flex_utilization_pct=(None, 100.0), required_mw=(None, 0.0),
                 moved_mw=(0.0, 0.0), deliverable_up_mw=(None, 90.0)),
            _row("2000-01-06 20:00:00", kpi_2_flex_utilization_pct=(None, 26.0), required_mw=(None, 9.0),
                 moved_mw=(0.0, 34.0), deliverable_up_mw=(None, 65.0))]
    summary = kp.window_summary(rows, [1.0, 1.0])               # before compacting: it reads the indicators
    result = {**kp.compact_moments(rows), "summary": summary, "hours_each": [1, 1]}
    figs = RENDERER_MAP["kpis_over_time"](result)
    assert len(figs) == 5
    kpi2 = next(t for t in figs[0].data if t.name.startswith("KPI-2"))
    assert list(kpi2.y) == [None, 26.0]                   # a trivial 100 % where nothing moved: not drawn
    assert "Nothing to show" in RENDERER_MAP["kpis_over_time"]({"moments": []})[0].layout.title.text


def test_the_tools_send_one_voltage_limit_to_both_checks_and_follow_the_window(monkeypatch):
    from llm_agent.agent import tools
    sent = []
    monkeypatch.setattr(tools, "_post", lambda path, body: sent.append((path, body)) or {"ok": True})
    tools.evaluate_kpis(vm_lower_pu=0.96, option="S3:cheapest", timestamp="2000-01-06 20:00")
    path, body = sent[-1]
    assert path == "/api/kpi/evaluate" and body["vm_lower_pu"] == body["opf_vm_lower"] == 0.96
    assert body["option"] == "S3:cheapest" and "vm_upper_pu" not in body             # not sent: the network's own
    assert tools.evaluate_kpis(disabled_generators=["G1"]).get("refusal_required")
    tools.kpis_over_time(start="2000-01-06 16", end="2000-01-07 00", moments=9, n1="preventive")
    path, body = sent[-1]
    assert path == "/api/kpi/over_time" and body["moments"] == 9 and body["progress_id"]
    assert tools._timeout_for("/api/kpi/over_time") == tools.LONG_STUDY_TIMEOUT_S
