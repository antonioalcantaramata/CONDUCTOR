"""
The investigation: why an element's voltage or loading is what it is, as its
change since a reference state divided among the changes that caused it
(backend/investigation.py, POST /api/investigate, the `investigate` tool).

The decomposition is checked on its own (Shapley values, verdicts, the
difference between two states), then end to end on the IEEE 30-bus demo,
including the bench of injected causes (evaluation/investigation.py): each
injection must come back as the cause, with nothing else blamed.
"""

import copy

import pandapower.networks as pn
import pytest

import investigation as inv

pytest.importorskip("fastapi.testclient")

import main_backend as mb  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

PEAK = "2000-01-06 20:00:00"


# ---------------------------------------------------------------------------
# The decomposition on its own
# ---------------------------------------------------------------------------

class TestShapley:
    def test_an_additive_quantity_gives_each_player_its_own_effect(self):
        effects = [0.5, -0.2, 0.1]
        v = lambda mask: sum(e for i, e in enumerate(effects) if mask >> i & 1)  # noqa: E731
        assert inv.shapley(3, v) == pytest.approx(effects)

    def test_an_interaction_is_shared_and_the_values_add_up(self):
        # Each alone does nothing; together they give 1: an even split.
        v = lambda mask: 1.0 if mask == 0b11 else 0.0  # noqa: E731
        assert inv.shapley(2, v) == pytest.approx([0.5, 0.5])
        w = lambda mask: [0.0, 0.3, 0.1, 1.0][mask & 0b11] + (mask >> 2) * 0.2 * (1 + (mask & 1))  # noqa: E731
        assert sum(inv.shapley(3, w)) == pytest.approx(w(0b111) - w(0))

    @pytest.mark.parametrize("share, expected", [(0.6, "cause"), (0.15, "cause"), (0.1, "contributes"),
                                                 (0.01, "ruled_out"), (-0.02, "ruled_out"),
                                                 (-0.2, "counteracts")])
    def test_verdicts(self, share, expected):
        assert inv.verdict(share) == expected


class TestDiff:
    def test_every_difference_a_power_flow_sees_and_nothing_else(self):
        ref = pn.case14()
        cur = copy.deepcopy(ref)
        cur.load.at[3, "p_mw"] *= 1.2
        cur.gen.at[0, "vm_pu"] = 1.0
        cur.line.at[5, "in_service"] = False
        cur.gen.at[1, "max_p_mw"] = 1.0                    # an OPF limit: no power-flow difference
        template, changes = inv.diff(ref, cur)
        assert template == "current"
        assert {(c.table, c.idx, tuple(c.cur)) for c in changes} == {
            ("load", 3, ("p_mw",)), ("gen", 0, ("vm_pu",)), ("line", 5, ("in_service",))}

    def test_an_added_element_is_out_of_service_in_the_other_state(self):
        ref = pn.case14()
        cur = copy.deepcopy(ref)
        import pandapower as pp
        pp.create_load(cur, bus=9, p_mw=10)
        template, changes = inv.diff(ref, cur)
        added = [c for c in changes if c.table == "load"]
        assert template == "current" and added[0].ref == {"in_service": False}

    def test_two_states_each_with_its_own_elements_are_refused(self):
        import pandapower as pp
        a, b = pn.case14(), pn.case14()
        pp.create_load(a, bus=9, p_mw=10)
        pp.create_sgen(b, bus=9, p_mw=10)
        with pytest.raises(inv.InvestigationError):
            inv.diff(a, b)

    def test_a_decomposition_reproduces_both_states_and_adds_up(self):
        ref = pn.case14()
        cur = copy.deepcopy(ref)
        cur.load["p_mw"] *= 1.15
        cur.load["q_mvar"] *= 1.15
        cur.ext_grid.at[0, "vm_pu"] = 1.02
        limits = {"vm_lower_pu": 0.95, "vm_upper_pu": 1.05, "max_line_loading_pct": 100.0,
                  "max_trafo_loading_pct": 100.0}
        out = inv.decompose(ref, cur, inv.element_target("bus", 13, "Bus_13", {13}, limits))
        assert out["status"] == "explained" and out["closure"]["reproduces_both_states"]
        assert sum(f["share_pct"] for f in out["factors"]) == pytest.approx(100.0, abs=0.5)
        kinds = {f["kind"] for f in out["factors"]}
        assert "setpoint" in kinds and "demand" in kinds


# ---------------------------------------------------------------------------
# End to end on the demo
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client(tmp_path_factory):
    store = tmp_path_factory.mktemp("investigation")
    patch = pytest.MonkeyPatch()
    patch.delenv("GRID_PROFILE", raising=False)                      # the default demo (IEEE 30)
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


def test_the_peak_undervoltage_is_the_demand_near_it(client):
    clock = mb.app_data["current_index"]
    out = _ok(client.post("/api/investigate", json={"timestamp": PEAK, "element": "Bus_29"}))
    assert out["status"] == "explained" and out["closure"]["reproduces_both_states"]
    assert out["symptom"]["violated"] and out["reference"]["within_limits"]
    assert out["reference"]["timestamp"] < PEAK and out["outside_limits_since"] > out["reference"]["timestamp"]
    assert out["main_cause"].startswith("Demand near Bus_29")
    assert sum(f["share_pct"] for f in out["factors"]) == pytest.approx(100.0, abs=0.5)
    assert out["drill_down"] and out["drill_down"][0]["members"][0]["member"] == "Demand at Bus_29"
    # The lever for an undervoltage where every unit holds its voltage is a
    # set-point, checked by a power flow (not "raise Gen_bus7 by 193 MW").
    action = out["levers"]["recommended_action"]
    assert action["kind"] == "setpoint_change" and action["source"] == "Gen_bus7"
    assert out["levers"]["setpoint_levers"][0]["verified"]
    assert out["message"].startswith("Bus_29 is at 0.943 p.u.")
    assert mb.app_data["current_index"] == clock                      # the clock never moves


def test_several_violations_and_none_named_ask_which(client):
    out = _ok(client.post("/api/investigate", json={"timestamp": PEAK}))
    assert out["status"] == "choose" and len(out["violations"]) == 2


def test_a_symptom_that_is_not_there_is_said_not_explained(client):
    out = _ok(client.post("/api/investigate", json={"timestamp": "2000-01-06 04:00", "element": "Bus_29"}))
    assert out["status"] == "no_symptom" and "within its limits" in out["message"]
    assert "factors" not in out


def test_a_change_within_limits_is_explained_against_the_moment_given(client):
    out = _ok(client.post("/api/investigate", json={"timestamp": "2000-01-06 18:00", "element": "Bus_29",
                                                    "reference_timestamp": "2000-01-06 04:00"}))
    assert out["status"] == "explained" and not out["symptom"]["violated"]
    assert out["reference"]["why"] == "the moment asked for" and "levers" not in out


def test_the_forecast_against_the_measurements(client):
    out = _ok(client.post("/api/investigate", json={"timestamp": PEAK, "element": "Bus_29",
                                                    "data_source": "forecasts",
                                                    "reference_data_source": "measurements"}))
    assert out["status"] == "explained"
    assert out["reference"]["why"] == "the measurements for the same moment"


def test_a_reference_naming_the_state_itself_leaves_it_to_the_search(client):
    # A model that fills every field sends the current moment, dataset and network, or "".
    for extra in ({"reference_timestamp": PEAK, "reference_data_source": "measurements", "reference_network": "base"},
                  {"reference_timestamp": "", "reference_data_source": "", "reference_network": "", "element_type": ""}):
        out = _ok(client.post("/api/investigate", json={"timestamp": PEAK, "element": "Bus_29", **extra}))
        assert out["status"] == "explained" and out["reference"]["why"].startswith("the last moment before")
    out = _ok(client.post("/api/investigate", json={"timestamp": "2000-01-06 04:00", "element": "Bus_29",
                                                    "reference_timestamp": "2000-01-06 04:00",
                                                    "reference_network": "base"}))
    assert out["status"] == "no_symptom"


def test_an_unknown_element_is_refused(client):
    assert client.post("/api/investigate", json={"timestamp": PEAK, "element": "Bus_999"}).status_code == 404


def test_the_injected_causes_are_found_and_nothing_else_is_blamed(client):
    from evaluation import investigation as ev

    def call(method, path, body):
        r = getattr(client, method)(path, **({"json": body} if body is not None else {}))
        return r.status_code, r.json()

    before = _ok(client.get("/api/proposals"))["proposals"]
    out = ev.bench(call)
    s = out["summary"]
    assert s["errors"] == 0, [r for r in out["cases"] if "error" in r]
    assert s["top1"] == 1.0 and s["recall"] == 1.0 and s["false_causes"] == 0 and s["closure"] == 1.0
    assert s["wrong_hypotheses_ruled_out"] == 1.0
    # The linear (sensitivity) estimate: close where it applies, blind to a line taken out.
    assert s["linear_mae_points"] < 2 and s["linear_blind_cases"] >= 3 and s["linear_top1"] < 1.0
    # Undoing each factor alone double-counts once the causes interact; the
    # shares never do.
    interacting = next(r for r in out["cases"] if r["case"] == "line_out_and_load")
    assert interacting["one_at_a_time_sum_pct"] > 110
    # The attribution alone cannot name a set-point, a line out or the grid's voltage.
    assert s["attribution_top1"] < 0.5
    assert _ok(client.get("/api/proposals"))["proposals"] == before   # the bench leaves nothing behind


def test_the_chart_is_a_waterfall_of_the_change(client):
    pytest.importorskip("plotly")  # the chart is UI-only; CI does not install plotly
    from llm_agent.agent.renderers import render_investigation

    out = _ok(client.post("/api/investigate", json={"timestamp": PEAK, "element": "Bus_0 -> Bus_1 [L0]"}))
    figures = render_investigation(out)
    waterfall = figures[0].data[0]
    assert waterfall.type == "waterfall" and waterfall.measure[-1] == "total"
    assert waterfall.y[-1] == pytest.approx(out["change"])
    assert len(figures) == 1 + len(out["drill_down"])
    assert render_investigation({"status": "no_symptom"}) == []


def test_the_tool_sends_what_it_is_given(monkeypatch):
    from llm_agent.agent import tools

    sent = {}
    monkeypatch.setattr(tools, "_post", lambda path, body: sent.update(path=path, body=body) or {"status": "x"})
    tools.investigate(element="Bus_29", reference_timestamp="2000-01-06", network="p1")
    assert sent["path"] == "/api/investigate"
    assert sent["body"]["reference_timestamp"] == "2000-01-06 00:00:00" and sent["body"]["network"] == "p1"
    tools.investigate(element="Bus_29", element_type="", reference_network="")
    assert "element_type" not in sent["body"] and "reference_network" not in sent["body"]
    refused = tools.investigate(element="Bus_29", top_n_sources=5)
    assert refused.get("refusal_required") or "Unsupported" in str(refused.get("error"))


# ---------------------------------------------------------------------------
# The agent's own hypotheses, and its record of the investigation
# ---------------------------------------------------------------------------

def test_hypotheses_are_the_callers_own_factors(client):
    _ok(client.post("/api/proposals/edit", json={"proposal": "two_loads", "replace": True, "edits": [
        {"op": "scale", "element_type": "load", "bus": 23, "factor": 2.0},
        {"op": "scale", "element_type": "load", "bus": 25, "factor": 1.6}]}))
    try:
        hyps = [{"name": "Demand at Bus_23", "elements": ["Bus_23"]},
                {"name": "Demand at Bus_25", "elements": ["Bus_25"]},
                {"name": "The line L38", "elements": ["L38"]}]
        out = _ok(client.post("/api/investigate", json={
            "timestamp": "2000-01-06 18:00", "network": "two_loads", "reference_network": "base",
            "element": "Bus_29", "hypotheses": hyps}))
        rows = {f["factor"]: f for f in out["factors"]}
        assert out["grouping"] == "hypotheses"
        assert rows["Demand at Bus_23"]["verdict"] == rows["Demand at Bus_25"]["verdict"] == "cause"
        assert rows["The line L38"]["nothing_changed"] and rows["The line L38"]["verdict"] == "ruled_out"
        assert "nothing it names changed" in out["message"]
        # The standard grouping cannot tell the two loads apart.
        standard = _ok(client.post("/api/investigate", json={
            "timestamp": "2000-01-06 18:00", "network": "two_loads", "reference_network": "base",
            "element": "Bus_29"}))
        assert len(standard["factors"]) == 1
        too_many = client.post("/api/investigate", json={"timestamp": PEAK, "element": "Bus_29",
                                                         "hypotheses": [{"name": str(i), "elements": ["Bus_29"]}
                                                                        for i in range(6)]})
        assert too_many.status_code == 422
    finally:
        client.delete("/api/proposals/two_loads")


class TestRecord:
    @pytest.fixture(autouse=True)
    def turn(self, monkeypatch):
        from llm_agent.agent import tools
        monkeypatch.setattr(tools, "_last_tool_results", [("investigate", {"status": "explained"}),
                                                          ("run_rsa", {"error": "boom"})])
        return tools

    def test_a_verdict_must_rest_on_a_study_that_ran(self, turn):
        refused = turn.record_investigation("Bus_29 low", [
            {"hypothesis": "Demand near Bus_29", "verdict": "supported", "evidence": []},
            {"hypothesis": "A line out", "verdict": "refuted", "evidence": ["simulate_contingency"]},
            {"hypothesis": "A failed study", "verdict": "refuted", "evidence": ["run_rsa"]},
            {"hypothesis": "The weather", "verdict": "untested"}])
        assert "not accepted" in refused["error"] and len(refused["problems"]) == 4

    def test_an_accepted_record_keeps_the_tools_it_needed(self, turn, tmp_path, monkeypatch):
        import json
        monkeypatch.setenv("CONDUCTOR_CACHE_DIR", str(tmp_path))
        out = turn.record_investigation("L0 overloaded", [
            {"hypothesis": "Demand near the line", "verdict": "supported", "evidence": ["investigate"],
             "finding": "47 % of the rise."},
            {"hypothesis": "Hot weather", "verdict": "untested", "evidence": ["investigate"],
             "missing_tool": "Weather data (temperature) for the network's area, per moment."}])
        assert out["status"] == "recorded" and out["counts"] == {"supported": 1, "refuted": 0, "untested": 1}
        assert out["hypotheses"][1]["evidence"] == []        # an untested hypothesis cites nothing
        saved = [json.loads(line) for line in (tmp_path / turn.TOOL_REQUESTS_FILE).read_text().splitlines()]
        assert saved[0]["hypothesis"] == "Hot weather" and "Weather data" in saved[0]["missing_tool"]

    def test_a_step_not_taken_is_not_a_missing_tool(self, turn):
        refused = turn.record_investigation("Bus_29 low", [
            {"hypothesis": "Demand", "verdict": "untested",
             "missing_tool": "After the operator chooses, run investigate on that element."},
            {"hypothesis": "A line out", "verdict": "untested",
             "missing_tool": "A single N-1 contingency study of each line."}])
        assert len(refused["problems"]) == 2 and "investigate" in refused["problems"][0]

    def test_inside_a_check_nothing_is_kept(self, turn, tmp_path, monkeypatch):
        monkeypatch.setenv("CONDUCTOR_CACHE_DIR", str(tmp_path))
        with turn.unrecorded():
            turn.record_investigation("x", [{"hypothesis": "h", "verdict": "untested", "missing_tool": "t"}])
        assert not (tmp_path / turn.TOOL_REQUESTS_FILE).exists()


def test_the_record_is_never_a_source_for_the_answer():
    from llm_agent.agent.provenance import check_turn
    # The only place 61.2 appears is the agent's own note: it stays ungrounded.
    record = {"user": "why?", "assistant": "Demand explains 61.2 % of the fall.",
              "tool_calls": [{"name": "record_investigation", "args": {"hypotheses": [{"finding": "61.2 %"}]}}],
              "tool_results": [{"name": "record_investigation", "result": {
                  "status": "recorded", "hypotheses": [{"finding": "61.2 % of the fall"}]}}]}
    assert len(check_turn(record).ungrounded) == 1


def test_the_tool_requests_report_groups_wordings():
    from evaluation.tool_requests import summarise
    groups = summarise([{"missing_tool": "Weather data (temperature)", "hypothesis": "heat"},
                        {"missing_tool": "Matched temperature observations for that evening", "hypothesis": "heat"},
                        {"missing_tool": "Weather, calendar and customer records", "hypothesis": "demand rose"},
                        {"missing_tool": "Protection relay event log", "hypothesis": "a trip"},
                        {"missing_tool": "A gas pipeline pressure study", "hypothesis": "gas"},
                        {"missing_tool": "a gas pipeline pressure study.", "hypothesis": "gas again"}])
    by = {g["need"] or g["tool"]: g for g in groups}
    # Three wordings of one need count together; a request for three needs counts under each.
    assert by["Weather data"]["count"] == 3 and len(by["Weather data"]["wordings"]) == 3
    assert by["Calendar and events"]["count"] == by["Customer and consumption data"]["count"] == 1
    from llm_agent.agent.investigation_record import needs_of
    # A need named only in saying why CONDUCTOR cannot do it is not asked for (observed live).
    assert needs_of("Customer-level metering for Bus_29; no CONDUCTOR tool provides weather or customer-level "
                    "data.") == ["Customer and consumption data"]
    assert needs_of("A continuation power-flow study and a time-domain ramp feasibility study", "analysis") == [
        "Voltage stability margin", "Dynamic simulation", "Time-coupled studies"]
    assert by["Protection and fault events"]["count"] == 1
    # A need no group names: by its wording, loosely.
    gas = next(g for g in groups if g["need"] is None)
    assert gas["count"] == 2 and len(gas["wordings"]) == 2 and groups[0]["need"] == "Weather data"
    # Shown: the most specific wording, not the one naming three needs.
    assert by["Weather data"]["tool"] == "Matched temperature observations for that evening"


# ---------------------------------------------------------------------------
# Other scopes and references
# ---------------------------------------------------------------------------

def test_the_whole_grid(client):
    out = _ok(client.post("/api/investigate", json={"timestamp": PEAK, "element": "all"}))
    assert out["status"] == "explained" and out["quantity"] == "violations"
    assert out["symptom"]["value"] == 2 and out["reference"]["value"] == 0
    assert out["message"].startswith("2 elements are outside their limits")
    assert "levers" not in out


def test_a_comfortable_reference_and_the_previous_day(client):
    near = _ok(client.post("/api/investigate", json={"timestamp": PEAK, "element": "Bus_29"}))
    comfy = _ok(client.post("/api/investigate", json={"timestamp": PEAK, "element": "Bus_29",
                                                      "reference_mode": "comfortable"}))
    assert comfy["reference"]["value"] >= 0.95 + 0.01 and comfy["reference"]["timestamp"] < near["reference"]["timestamp"]
    assert abs(comfy["change"]) > abs(near["change"]) and "outside_limits_since" not in comfy
    day = _ok(client.post("/api/investigate", json={"timestamp": PEAK, "element": "Bus_0 -> Bus_1 [L0]",
                                                    "reference_mode": "previous_day"}))
    assert day["reference"]["timestamp"] == "2000-01-05 20:00:00"
    assert day["reference"]["why"] == "the same moment the previous day"


def test_a_reference_close_to_the_limit_is_pointed_out(client):
    out = _ok(client.post("/api/investigate", json={"timestamp": "2000-01-06 18:00", "element": "Bus_29",
                                                    "outage": "L38"}))
    assert "reference_note" in out and "comfortable" in out["reference_note"]


def test_after_an_outage(client):
    out = _ok(client.post("/api/investigate", json={"timestamp": "2000-01-06 18:00", "element": "Bus_29",
                                                    "outage": "L38"}))
    assert out["outage"] == "Bus_28 -> Bus_29 [L38]" and "after losing" in out["symptom"]["label"]
    assert out["status"] == "explained" and out["main_cause"].startswith("Demand near Bus_29")
    intact = _ok(client.post("/api/investigate", json={"timestamp": "2000-01-06 18:00", "element": "Bus_29",
                                                       "outage": "L38", "reference_outage": "none"}))
    assert intact["factors"][0]["factor"] == "Bus_28 -> Bus_29 [L38] out of service"
    both = _ok(client.post("/api/investigate", json={"timestamp": "2000-01-06 18:00", "element": "Bus_29",
                                                     "outage": "L38", "reference_outage": "L37"}))
    assert both["status"] == "not_decomposed" and "both outages at once" in both["message"]


def test_over_a_window(client):
    out = _ok(client.post("/api/investigate", json={"start_timestamp": "2000-01-06 17",
                                                    "end_timestamp": "2000-01-06 21",
                                                    "element": "Bus_0 -> Bus_1 [L0]"}))
    assert out["mode"] == "window" and out["quantity"] == "exceedance_pct_h"
    assert out["symptom"]["moments_outside_limits"] > out["reference"]["moments_outside_limits"] == 0
    assert out["symptom"]["hours_outside_limits"] == pytest.approx(out["symptom"]["moments_outside_limits"] * 0.25)
    assert out["reference"]["why"] == "the window of the same length just before"
    assert out["closure"]["reproduces_both_states"]
    assert sum(f["share_pct"] for f in out["factors"]) == pytest.approx(100.0, abs=0.5)
    grid = _ok(client.post("/api/investigate", json={"start_timestamp": "2000-01-06", "end_timestamp": "2000-01-06"}))
    assert grid["quantity"] == "violation_hours" and grid["status"] == "explained"
    forecast = _ok(client.post("/api/investigate", json={"start_timestamp": "2000-01-06 17",
                                                         "end_timestamp": "2000-01-06 21", "element": "Bus_29",
                                                         "data_source": "forecasts",
                                                         "reference_data_source": "measurements"}))
    assert forecast["reference"]["why"] == "the measurements over the same window"
    pytest.importorskip("plotly")  # the chart is UI-only; CI does not install plotly
    from llm_agent.agent.renderers import render_investigation
    fig = render_investigation(out)[0]
    assert "outside limits" in fig.layout.yaxis.title.text


def test_causes_that_amplify_each_other_are_named(client):
    _ok(client.post("/api/proposals/edit", json={"proposal": "amp", "replace": True, "edits": [
        {"op": "set_in_service", "element_type": "line", "element": 38, "in_service": False},
        {"op": "scale", "element_type": "load", "bus": 29, "factor": 1.4}]}))
    try:
        out = _ok(client.post("/api/investigate", json={"timestamp": "2000-01-06 18:00", "element": "Bus_29",
                                                        "network": "amp", "reference_network": "base"}))
        pair = out["interactions"][0]
        assert pair["kind"] == "amplify" and pair["share_pct"] > 10
        assert "amplify each other" in out["message"]
    finally:
        client.delete("/api/proposals/amp")


# ---------------------------------------------------------------------------
# The record against the study, the answer against the record
# ---------------------------------------------------------------------------

_STUDY = {"status": "explained", "reference": {"label": "2000-01-06 18:30"},
          "factors": [{"factor": "Demand near Bus_29 (2 buses)", "verdict": "cause", "share_pct": 71.1},
                      {"factor": "Units' output (5 units)", "verdict": "ruled_out", "share_pct": 0.2}]}


class TestRecordAgainstStudy:
    @pytest.fixture(autouse=True)
    def turn(self, monkeypatch):
        from llm_agent.agent import tools
        monkeypatch.setattr(tools, "_last_tool_results", [("investigate", _STUDY)])
        return tools

    def test_a_verdict_the_study_contradicts_is_refused(self, turn):
        out = turn.record_investigation("Bus_29 low", [
            {"hypothesis": "The units' output caused it", "verdict": "supported", "evidence": ["investigate"],
             "factor": "Units' output (5 units)"}])
        assert "ruled out" in out["error"] and "0.2 %" in out["error"]

    def test_the_factor_is_found_in_the_text_too(self, turn):
        out = turn.record_investigation("Bus_29 low", [
            {"hypothesis": "Demand near Bus_29 is the cause", "verdict": "refuted", "evidence": ["investigate"]}])
        assert "error" in out and "cause" in out["error"]

    def test_a_factor_no_study_returned_is_refused(self, turn):
        out = turn.record_investigation("Bus_29 low", [
            {"hypothesis": "A tap moved", "verdict": "refuted", "evidence": ["investigate"], "factor": "Tap of T1"}])
        assert "no investigation" in out["error"]

    def test_an_agreeing_record_keeps_its_factor(self, turn, tmp_path, monkeypatch):
        monkeypatch.setenv("CONDUCTOR_CACHE_DIR", str(tmp_path))
        out = turn.record_investigation("Bus_29 low", [
            {"hypothesis": "Demand near Bus_29 rose", "verdict": "supported", "evidence": ["investigate"]},
            {"hypothesis": "The units' output", "verdict": "refuted", "evidence": ["investigate"]}])
        assert out["status"] == "recorded"
        assert out["hypotheses"][0]["factor"] == "Demand near Bus_29 (2 buses)"


def _with_record(answer: str, hypotheses: list[dict]) -> dict:
    return {"user": "why?", "assistant": answer, "tool_calls": [{"name": "record_investigation", "args": {}}],
            "tool_results": [{"name": "record_investigation",
                              "result": {"status": "recorded", "hypotheses": hypotheses}}]}


class TestAnswerAgainstRecord:
    WEATHER = {"hypothesis": "Hot weather raised the demand", "verdict": "untested", "missing_tool": "weather data"}
    DEMAND = {"hypothesis": "Demand at Bus_29 rose", "verdict": "supported", "evidence": ["investigate"]}

    def _issues(self, answer, hyps):
        from llm_agent.agent.investigation_record import check_answer
        return [i.kind for i in check_answer(answer, _with_record(answer, hyps))]

    def test_an_untested_cause_stated_as_a_finding(self):
        assert "untested_as_finding" in self._issues(
            "The hot weather caused the overload at Bus_29. It was not tested further.", [self.WEATHER, self.DEMAND])

    def test_a_hedged_or_negated_one_is_fine(self):
        answer = ("Demand at Bus_29 explains most of it. Hot weather could have driven that demand, but there is "
                  "no weather data, so it remains untested.")
        assert self._issues(answer, [self.WEATHER, self.DEMAND]) == []

    def test_untested_causes_must_be_said(self):
        assert "untested_unsaid" in self._issues("Demand at Bus_29 rose and explains it.", [self.WEATHER, self.DEMAND])

    def test_a_supported_cause_must_be_named(self):
        assert "hypothesis_unsaid" in self._issues("The demand rose. Weather was not tested.", [self.WEATHER, self.DEMAND])

    def test_the_grader_fails_a_finding_never_tested(self):
        from evaluation.report import grade_record
        answer = "The hot weather caused the overload at Bus_29."
        codes = {f.code: f.severity for f in grade_record(_with_record(answer, [self.WEATHER, self.DEMAND])).findings}
        assert codes.get("untested_as_finding") == "fail"


def test_the_reading_check_judges_an_investigation_by_its_main_cause():
    from llm_agent.agent import robustness as rb
    study = {**_STUDY, "factors": [{**f, "kind": "demand" if f["factor"].startswith("Demand") else "units"}
                                   for f in _STUDY["factors"]]}
    v = rb.verdict("investigate", study | {"main_cause": "Demand near Bus_29 (2 buses)"})
    w = rb.verdict("investigate", {**study, "factors": [{"factor": "Demand (21 buses)", "kind": "demand",
                                                         "share_pct": 100.0}], "main_cause": "Demand (21 buses)"})
    assert v.key == w.key == "explained:demand"
    u = rb.verdict("investigate", study | {"main_cause": "Units' output (5 units)"})
    assert rb.compare(v, u) == rb.CHANGES
    assert rb.compare(v, rb.verdict("investigate", {"status": "no_symptom"})) == rb.CHANGES
    window = {"tool_calls": [{"name": "investigate", "args": {"start_timestamp": "2000-01-06"}}],
              "tool_results": [{"name": "investigate", "result": _STUDY | {"main_cause": "x"}}]}
    assert rb._decisive(window) == []


def test_comfortable_means_at_least_twice_the_margin_it_has(client):
    # At 16:30 Bus_29 is 0.015 p.u. inside its band: a fixed 0.01 p.u. would
    # accept a moment 15 minutes earlier and explain nothing.
    out = _ok(client.post("/api/investigate", json={"timestamp": "2000-01-06 16:30", "element": "Bus_29",
                                                    "reference_mode": "comfortable"}))
    margin = out["symptom"]["margin"]
    assert 0 < margin < 0.02
    assert out["reference"]["value"] - 0.95 >= 2 * margin - 1e-9


def test_the_mw_behind_a_factor_are_numbers_not_only_text(client):
    out = _ok(client.post("/api/investigate", json={"timestamp": PEAK, "element": "Bus_29"}))
    demand = out["factors"][0]
    assert demand["mw"]["change"] == pytest.approx(demand["mw"]["now"] - demand["mw"]["reference"], abs=0.02)
    assert all("mw" in m for m in out["drill_down"][0]["members"])


def test_a_window_of_one_moment_is_a_comparison_of_two_moments(client):
    # What a model filling every field sent: start = end, element 'all' with a type, the reference window one moment.
    out = _ok(client.post("/api/investigate", json={
        "element": "all", "element_type": "line", "outage": "Bus_28 -> Bus_29 [L38]",
        "start_timestamp": "2000-01-06 18:00", "end_timestamp": "2000-01-06 18:00",
        "reference_start_timestamp": "2000-01-06 15:00", "reference_end_timestamp": "2000-01-06 15:00"}))
    assert out.get("mode") != "window" and out["element"] == "Bus_29" and out["quantity"] == "vm_pu"
    assert out["reference"]["timestamp"] == "2000-01-06 15:00:00" and out["outage"] == "Bus_28 -> Bus_29 [L38]"


def test_an_intact_reference_at_another_moment_is_pointed_out(client):
    out = _ok(client.post("/api/investigate", json={"timestamp": "2000-01-06 18:00", "element": "Bus_29",
                                                    "outage": "L38", "reference_outage": "none",
                                                    "reference_timestamp": "2000-01-06 15:00"}))
    assert "mix the outage" in out["reference_note"]


def test_the_grid_with_one_element_outside_is_that_element(client):
    out = _ok(client.post("/api/investigate", json={"timestamp": "2000-01-06 18:00", "element": "all",
                                                    "outage": "L38", "reference_timestamp": "2000-01-06 15:00"}))
    assert out["element"] == "Bus_29" and out["quantity"] == "vm_pu" and "only element" in out["narrowed"]



# ---------------------------------------------------------------------------
# Classical views, when the agent asks for them
# ---------------------------------------------------------------------------

def test_classical_views_only_when_asked(client):
    plain = _ok(client.post("/api/investigate", json={"timestamp": PEAK, "element": "Bus_29"}))
    assert "classical" not in plain
    out = _ok(client.post("/api/investigate", json={"timestamp": PEAK, "element": "Bus_0 -> Bus_1 [L0]",
                                                    "classical": ["linear", "tracing"]}))
    linear = out["classical"]["linear"]
    assert linear["agrees_with_exact"] and linear["mean_abs_error_points"] < 1
    assert all(r["method"] == "sensitivities (trapezoid)" for r in linear["factors"])
    now = out["classical"]["tracing"]["current"]
    assert now["direction"] == "Bus_0 → Bus_1" and now["sources"][0]["name"] == "the external grid"
    assert sum(e["share_pct"] for e in now["sinks"]) == pytest.approx(100.0, abs=0.5)
    bus = _ok(client.post("/api/investigate", json={"timestamp": PEAK, "element": "Bus_29", "classical": ["tracing"]}))
    assert "not_applicable" in bus["classical"]["tracing"]
    assert client.post("/api/investigate", json={"timestamp": PEAK, "element": "Bus_29",
                                                 "classical": ["fourier"]}).status_code == 422


def test_a_switching_cause_has_no_sensitivity_but_a_branch_out_has_its_lodf(client):
    _ok(client.post("/api/proposals/edit", json={"proposal": "sw", "replace": True, "edits": [
        {"op": "set_in_service", "element_type": "line", "element": 1, "in_service": False}]}))
    try:
        volt = _ok(client.post("/api/investigate", json={"timestamp": "2000-01-06 18:00", "element": "Bus_29",
                                                         "network": "sw", "reference_network": "base",
                                                         "classical": ["linear"]}))
        assert volt["classical"]["linear"]["factors"][0]["why"] == "a switching change has no sensitivity"
        flow = _ok(client.post("/api/investigate", json={"timestamp": "2000-01-06 18:00",
                                                         "element": "Bus_0 -> Bus_1 [L0]", "network": "sw",
                                                         "reference_network": "base", "classical": ["linear"]}))
        row = flow["classical"]["linear"]["factors"][0]
        assert row["method"] == "LODF (DC)" and abs(row["error_points"]) < 50
    finally:
        client.delete("/api/proposals/sw")


def test_classical_figures_are_drawn_beside_the_exact_ones(client):
    pytest.importorskip("plotly")  # the chart is UI-only; CI does not install plotly
    from llm_agent.agent.renderers import render_investigation
    out = _ok(client.post("/api/investigate", json={"timestamp": PEAK, "element": "Bus_0 -> Bus_1 [L0]",
                                                    "classical": ["linear", "tracing"]}))
    titles = [f.layout.title.text for f in render_investigation(out)]
    assert any(t.startswith("Exact shares and the linear estimate") for t in titles)
    assert any(t.startswith("Where its flow comes from") for t in titles)


# ---------------------------------------------------------------------------
# Against other references; the agent's coverage; recurring causes; prevention; the report
# ---------------------------------------------------------------------------

def test_the_main_cause_is_checked_against_the_other_references(client):
    out = _ok(client.post("/api/investigate", json={"timestamp": PEAK, "element": "Bus_29"}))
    rr = out["reference_robustness"]
    assert rr["holds"] is True and {c["mode"] for c in rr["checked"]} == {"within_limits", "comfortable", "previous_day"}
    assert all(c["kind"] == "demand" for c in rr["checked"])
    assert "against every reference checked" in out["message"]
    asked = _ok(client.post("/api/investigate", json={"timestamp": PEAK, "element": "Bus_29",
                                                      "reference_timestamp": "2000-01-06 16:00"}))
    assert "reference_robustness" not in asked                     # the operator chose the comparison
    forced = _ok(client.post("/api/investigate", json={"timestamp": PEAK, "element": "Bus_29",
                                                       "reference_timestamp": "2000-01-06 16:00",
                                                       "check_references": True}))
    assert forced["reference_robustness"]["checked"][0]["mode"] == "asked"


def test_how_much_the_agents_hypotheses_explain(client):
    out = _ok(client.post("/api/investigate", json={"timestamp": PEAK, "element": "Bus_29",
                                                    "hypotheses": [{"name": "Demand at Bus_29", "elements": ["Bus_29"]}]}))
    assert out["hypotheses_coverage_pct"] < 75 and "Split or add hypotheses" in out["coverage_note"]
    from llm_agent.agent.investigation_record import check_answer
    record = {"tool_calls": [{"name": "investigate", "args": {}}],
              "tool_results": [{"name": "investigate", "result": out}]}
    assert [i.kind for i in check_answer("Demand at Bus_29 explains it.", record)] == ["coverage_unsaid"]
    assert check_answer("Demand at Bus_29 explains 62 %; the rest is in other changes.", record) == []


def test_recurring_causes_over_the_week(client):
    out = _ok(client.post("/api/investigate", json={"recurring": True}))
    assert out["element"] == "Bus_29" and "most often" in out["picked"]
    assert out["episodes_found"] >= 5 and out["recurs"] is True
    assert out["causes"][0]["cause"] == "demand" and out["causes"][0]["main_in_pct"] == 100.0
    assert all(e["reference"] < e["start"] for e in out["events"] if e.get("reference"))
    pytest.importorskip("plotly")  # the chart is UI-only; CI does not install plotly
    from llm_agent.agent.renderers import render_investigation
    assert render_investigation(out)[0].layout.barmode == "relative"


def test_prevention_finds_the_smallest_sufficient_action_and_the_warning(client):
    out = _ok(client.post("/api/investigate/prevention", json={
        "timestamp": "2000-01-05 18:30", "element": "Bus_29",
        "actions": [{"unit": "Gen_bus7", "setpoint_change": 0.01}, {"unit": "Gen_bus12", "setpoint_change": 0.01}],
        "combine": True}))
    assert out["warning"]["seen"] and out["warning"]["lead_minutes"] > 0
    gen7 = out["actions"][0]
    assert gen7["as_given"]["works"] and gen7["smallest"]["share_of_action"] <= 1.0
    assert out["actions"][-1]["action"].startswith("together:")
    assert len(out["series"]["with"]) == len(out["series"]["moments"]) and out["series"]["sufficient"]
    pytest.importorskip("plotly")  # the chart is UI-only; CI does not install plotly
    from llm_agent.agent.renderers import render_prevention
    assert len(render_prevention(out)[0].data) == 2


def test_prevention_says_why_nothing_works(client):
    out = _ok(client.post("/api/investigate/prevention", json={"timestamp": PEAK, "element": "Bus_29"}))
    assert not any("smallest" in a for a in out["actions"])
    assert "None of the actions tested" in out["message"] and "outside" in out["message"]
    quiet = _ok(client.post("/api/investigate/prevention", json={"timestamp": "2000-01-06 04:00", "element": "Bus_29"}))
    assert quiet["status"] == "no_symptom"


def test_the_tool_drops_the_zeros_a_model_fills_in(monkeypatch):
    from llm_agent.agent import tools
    sent = {}
    monkeypatch.setattr(tools, "_post", lambda path, body: sent.update(path=path, body=body) or {"status": "x"})
    tools.test_prevention(element="Bus_29", actions=[{"unit": "Gen_bus7", "setpoint_pu": 0, "setpoint_change": 0.01,
                                                      "p_change_mw": 0}])
    assert sent["path"] == "/api/investigate/prevention"
    assert sent["body"]["actions"] == [{"unit": "Gen_bus7", "setpoint_change": 0.01}]


def test_the_report_holds_everything_the_investigation_found(client):
    from llm_agent.agent import investigation_report as rep
    inv_out = _ok(client.post("/api/investigate", json={"timestamp": PEAK, "element": "Bus_0 -> Bus_1 [L0]",
                                                        "classical": ["linear", "tracing"]}))
    prev = _ok(client.post("/api/investigate/prevention", json={"timestamp": "2000-01-05 18:30", "element": "Bus_29",
                                                               "actions": [{"unit": "Gen_bus7", "setpoint_change": 0.01}]}))
    record = {"user": "Why is L0 overloaded?", "assistant": "Demand near the line.", "model": "m", "provider": "p",
              "tool_calls": [{"name": "investigate", "args": {"element": "L0"}},
                             {"name": "test_prevention", "args": {}},
                             {"name": "record_investigation", "args": {}}],
              "tool_results": [{"name": "investigate", "result": inv_out}, {"name": "test_prevention", "result": prev},
                               {"name": "record_investigation", "result": {
                                   "status": "recorded", "symptom": "L0 overloaded",
                                   "hypotheses": [{"hypothesis": "Hot weather", "verdict": "untested",
                                                   "missing_tool": "weather data"}]}}]}
    assert rep.has_investigation(record) and not rep.has_investigation({"tool_results": []})
    md = rep.build(record)
    for heading in ("# Investigation report", "## The question", "## The investigation", "**Factors**",
                    "Against other references", "linear estimate", "power-flow tracing", "### Prevention 2",
                    "## The hypotheses the agent tested", "would need: weather data", "## The studies that ran",
                    "## Method and limits"):
        assert heading in md, heading


def test_a_record_can_cite_the_factors_of_a_recurring_investigation(monkeypatch, tmp_path):
    # Observed live: the episodes' factors were not seen, and a correct record was refused.
    from llm_agent.agent import tools
    monkeypatch.setenv("CONDUCTOR_CACHE_DIR", str(tmp_path))
    recurring = {"mode": "recurring", "status": "explained",
                 "events": [{"start": "2000-01-05 18:00:00", "factors": [
                     {"factor": "Demand near Bus_29 (2 buses)", "verdict": "cause", "share_pct": 71.5}]}]}
    monkeypatch.setattr(tools, "_last_tool_results", [("investigate", recurring)])
    out = tools.record_investigation("Bus_29 recurring", [
        {"hypothesis": "Demand near Bus_29", "verdict": "supported", "evidence": ["investigate"],
         "factor": "Demand near Bus_29 (2 buses)"}])
    assert out["status"] == "recorded"


def test_a_share_of_an_absolute_setpoint_is_said_as_the_setpoint_it_gives(client):
    out = _ok(client.post("/api/investigate/prevention", json={
        "timestamp": "2000-01-05 18:30", "element": "Bus_29", "actions": [{"unit": "Gen_bus7", "setpoint_pu": 1.01}]}))
    action = out["actions"][0]
    assert action["setpoint_from"] == 1.0 and action["setpoint_change"] == 0.01
    smallest = action["smallest"]
    to = 1.0 + 0.01 * smallest["share_of_action"]
    assert f"to {to:.3f} p.u." in smallest["action"]


# ---------------------------------------------------------------------------
# Physical or a fault in the data
# ---------------------------------------------------------------------------

def test_measured_causes_on_clean_data_look_physical(client):
    out = _ok(client.post("/api/investigate", json={"timestamp": PEAK, "element": "Bus_29"}))
    assert out["data_check"]["suspicious"] == [] and "look physical" in out["data_check"]["message"]
    assert all("data_check" in f for f in out["factors"] if f["verdict"] == "cause")


def test_a_spike_in_the_measurements_is_flagged_before_anyone_acts_on_it(client):
    ts = "2000-01-06 14:00:00"
    table = mb.app_data["measurements"][ts]
    saved = table.copy()
    try:
        # A faulty meter: Bus_29's consumption reads four times its value for one moment.
        table.loc[table["substation_name"] == "Bus_29", "consumption"] *= 4
        mb.app_data.pop("_injection_series", None)                  # the history is cached per dataset
        out = _ok(client.post("/api/investigate", json={"timestamp": ts, "element": "Bus_29",
                                                        "reference_timestamp": "2000-01-06 13:45"}))
        assert out["main_cause"].startswith("Demand")
        doubts = out["data_check"]["suspicious"]
        assert doubts and doubts[0]["member"] == "Demand at Bus_29"
        assert any("jump" in f for f in doubts[0]["flags"]) and any("outside everything" in f for f in doubts[0]["flags"])
        assert "Check the data before acting" in out["message"]
    finally:
        mb.app_data["measurements"][ts] = saved
        mb.app_data.pop("_injection_series", None)


class TestPlausibility:
    def test_a_frozen_value_and_a_level_out_of_range(self):
        import numpy as np
        import plausibility
        x = np.sin(np.linspace(0, 20, 300)) * 5 + 10
        frozen = x.copy()
        frozen[100:112] = frozen[100]
        assert any("same value" in f for f in plausibility.check(frozen, 110, 111)["flags"])
        spike = x.copy()
        spike[200] = 60
        flags = plausibility.check(spike, 199, 200)["flags"]
        assert any("jump" in f for f in flags) and any("outside everything" in f for f in flags)
        assert plausibility.check(x, 199, 200)["plausible"] is True


def test_an_answer_that_hides_a_data_doubt_is_flagged():
    from llm_agent.agent.investigation_record import check_answer
    record = {"tool_calls": [{"name": "investigate"}], "tool_results": [{"name": "investigate", "result": {
        "data_check": {"suspicious": [{"member": "Demand at Bus_29", "flags": ["a jump"]}]}}}]}
    assert [i.kind for i in check_answer("Demand at Bus_29 caused it.", record)] == ["data_doubt_unsaid"]
    assert check_answer("Demand at Bus_29 caused it — but check the measurement first.", record) == []


# ---------------------------------------------------------------------------
# Looking ahead: what the forecast says is coming, and what would prevent it
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def ahead(client):
    """The forecast's first day (the clock is before the forecast starts)."""
    return _ok(client.post("/api/investigate/look_ahead", json={}))


def test_the_look_ahead_finds_the_episodes_when_to_decide_and_what_prevents_them(ahead):
    assert ahead["status"] == "expected" and ahead["basis"] == "forecasts" and "not facts" in ahead["note"]
    assert ahead["horizon"]["start"] == "2000-01-06 00:00:00" and ahead["horizon"]["hours"] == 24
    episodes = ahead["episodes"]
    assert [e["start"] for e in episodes] == sorted(e["start"] for e in episodes)
    for e in episodes:
        assert e["decide_by"] <= e["start"] <= e["peak"] <= e["end"]
        assert e["cause"]["against"] < e["decide_by"]
        assert e["action"] or e["no_action"].startswith("None of the actions tested")
    # Back inside its limits for under an hour: still one episode (L0's morning).
    morning = [e for e in episodes if e["element"] == "Bus_0 -> Bus_1 [L0]" and e["start"] < "2000-01-06 12"]
    assert len(morning) == 1 and morning[0]["end"] > morning[0]["start"]
    held = [h for e in episodes if e["action"] for h in e["action"]["holds_under"]]
    # Where the forecast errs against the action: more demand. The demo's units are all
    # conventional, so no error in renewable output is checked, and the result says so.
    assert {h["case"] for h in held} == {"demand_higher"} and "no renewable units" in ahead["forecast_errors_checked"]
    assert all(h["holds"] in (True, False) for h in held)
    assert all(h.get("why_not") for h in held if h["holds"] is False)
    assert "the forecast expects" in ahead["message"]


def test_the_look_ahead_chart_and_report(ahead):
    from llm_agent.agent import investigation_report as rep
    record = {"user": "What should I watch for?", "assistant": "…",
              "tool_calls": [{"name": "look_ahead", "args": {}}],
              "tool_results": [{"name": "look_ahead", "result": ahead}]}
    assert rep.has_investigation(record)
    md = rep.build(record)
    assert "Looking ahead" in md and "Holds if the forecast errs" in md and "A look-ahead is as good" in md
    pytest.importorskip("plotly")  # the chart is UI-only; CI does not install plotly
    from llm_agent.agent.renderers import render_look_ahead
    fig = render_look_ahead(ahead)[0]
    assert len(fig.layout.yaxis.categoryarray) == len(ahead["episodes"])


def test_a_quiet_horizon_and_one_element(client):
    quiet = _ok(client.post("/api/investigate/look_ahead", json={
        "start_timestamp": "2000-01-06 00:00", "hours": 6, "elements": ["Bus_29"]}))
    assert quiet["status"] == "clear" and quiet["episodes"] == [] and "Nothing is expected" in quiet["message"]
    assert client.post("/api/investigate/look_ahead", json={"hours": 100}).status_code == 422


def test_the_look_ahead_tool_sends_the_forecast_unless_told(monkeypatch):
    from llm_agent.agent import tools
    sent = {}
    monkeypatch.setattr(tools, "_post", lambda path, body: sent.update(path=path, body=body) or {"status": "x"})
    tools.look_ahead(start_timestamp="", elements=["", "Bus_29"], max_episodes=0, data_source="")
    assert sent["path"] == "/api/investigate/look_ahead"
    assert sent["body"]["data_source"] == "forecasts" and sent["body"]["elements"] == ["Bus_29"]
    assert "start_timestamp" not in sent["body"] and "max_episodes" not in sent["body"]


def test_a_start_before_the_forecast_begins_starts_where_it_does(client):
    # Observed live: the model passed the clock (1 January) and got a 404.
    out = _ok(client.post("/api/investigate/look_ahead", json={"start_timestamp": "2000-01-01 00:00:00",
                                                               "hours": 2}))
    assert out["horizon"]["start"] == "2000-01-06 00:00:00" and "before the forecast begins" in out["start_note"]


def test_a_data_fault_hypothesis_is_judged_by_the_data_check():
    # Observed live: "the demand change is a data anomaly", refuted by a clean data
    # check, was refused because the factor itself is a cause.
    from llm_agent.agent.investigation_record import study_factors, validate
    result = {"reference": {"label": "18:30"}, "factors": [
        {"factor": "Demand near Bus_29 (2 buses)", "verdict": "cause", "share_pct": 71.1,
         "data_check": {"checked": 2, "suspicious": []}}]}
    factors = study_factors([("investigate", result)])
    anomaly = {"hypothesis": "The measured demand change is a data-quality anomaly.", "verdict": "refuted",
               "evidence": ["investigate"], "factor": "Demand near Bus_29 (2 buses)"}
    rows, problems = validate([anomaly], {"investigate"}, factors)
    assert problems == [] and rows[0]["factor"] == "Demand near Bus_29 (2 buses)"
    assert "data check" in validate([{**anomaly, "verdict": "supported"}], {"investigate"}, factors)[1][0]
    # The share still decides a hypothesis about the cause itself.
    cause = {**anomaly, "hypothesis": "Rising demand near Bus_29 caused the fall."}
    assert "investigation found" in validate([cause], {"investigate"}, factors)[1][0]


def test_an_untested_hypothesis_says_whether_it_lacks_an_analysis_or_data():
    from llm_agent.agent.investigation_record import summarise_requests, validate
    stability = {"hypothesis": "Bus_29 is close to voltage collapse.", "verdict": "untested",
                 "missing_tool": "A continuation power flow giving the PV curve and loadability margin at Bus_29",
                 "missing_kind": "analysis"}
    weather = {"hypothesis": "Hot weather raised the demand.", "verdict": "untested",
               "missing_tool": "Weather observations aligned with the demand"}
    rows, problems = validate([stability, weather], set(), [])
    assert problems == [] and [r["missing_kind"] for r in rows] == ["analysis", "data"]   # the second inferred
    groups = summarise_requests(rows)
    assert {(g["kind"], g["need"]) for g in groups} == {("analysis", "Voltage stability margin"),
                                                         ("data", "Weather data")}
    # Something CONDUCTOR already computes is a study to run, not a tool to build…
    n1 = {**stability, "missing_tool": "An N-1 contingency analysis of the evening"}
    assert "already computes" in validate([n1], set(), [])[1][0]
    # …unless something new comes with it.
    ramps = {**stability, "missing_tool": "A multi-period OPF with the units' ramp limits"}
    assert validate([ramps], set(), [])[1] == []


def test_the_tool_requests_report_lists_tools_and_data_apart():
    from evaluation.tool_requests import render, summarise
    out = render(summarise([{"missing_tool": "A dynamic simulation of the motor loads", "missing_kind": "analysis"},
                            {"missing_tool": "Customer metering", "missing_kind": "data"}]))
    assert out.index("## Tools to build") < out.index("Dynamic simulation") < out.index("## Data to integrate") \
        < out.index("Customer and consumption data")


@pytest.mark.parametrize("answer", [
    "A voltage-stability-margin study would be needed to assess distance to voltage collapse; customer, weather, "
    "or calendar data would be needed to explain why demand increased.",
    "The analysis does not establish whether weather, calendar effects, or a customer event drove the demand rise; "
    "nor does it assess voltage-collapse margin.",
])
def test_ways_of_saying_something_was_not_tested(answer):
    # Observed live: both were flagged as hiding, or stating, an untested hypothesis.
    from llm_agent.agent.investigation_record import check_answer
    record = {"tool_results": [{"name": "record_investigation", "result": {"status": "recorded", "hypotheses": [
        {"hypothesis": "Weather drove the rise in demand.", "verdict": "untested", "missing_tool": "weather data"},
        {"hypothesis": "Bus_29 is close to voltage collapse.", "verdict": "untested",
         "missing_tool": "a continuation power flow"}]}}]}
    assert check_answer(answer, record) == []


def test_a_sentence_sharing_only_a_common_word_is_not_the_hypothesis():
    from llm_agent.agent.investigation_record import check_answer
    record = {"tool_results": [{"name": "record_investigation", "result": {"status": "recorded", "hypotheses": [
        {"hypothesis": "The undervoltage indicates Bus_29 is close to voltage collapse.", "verdict": "untested",
         "missing_tool": "a continuation power flow"}]}}]}
    answer = ("The week's measurements contained 12 Bus_29 undervoltage episodes, and demand was the main cause "
              "in all 10 episodes investigated. How close Bus_29 is to collapse was not tested.")
    assert check_answer(answer, record) == []


def test_with_renewables_the_look_ahead_also_checks_less_renewable_output(client):
    _ok(client.post("/api/proposals/edit", json={"proposal": "pv", "replace": True, "edits": [
        {"op": "add_generator", "name": "PV_29", "bus": 29, "p_mw": 3, "p_max_mw": 3, "kind": "PV"}]}))
    try:
        out = _ok(client.post("/api/investigate/look_ahead", json={"network": "pv", "hours": 12}))
        cases = {h["case"] for e in out["episodes"] if e["action"] for h in e["action"]["holds_under"]}
        assert cases == {"demand_higher", "renewables_lower"} and "renewable output" in out["forecast_errors_checked"]
    finally:
        client.delete("/api/proposals/pv")
