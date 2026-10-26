"""
What backs each claim of an answer (agent/evidence.py), and whether its
conclusion depends on how the question was read (agent/robustness.py).

Both are computed in code from the turn record — the evidence levels from
where each figure was traced, the reading check from the studies' own verdict
fields — so they are tested here without a model or a backend.
"""

import pytest

from llm_agent.agent import evidence as ev
from llm_agent.agent import loop as loop_module
from llm_agent.agent import robustness as rb
from llm_agent.agent import tools
from llm_agent.agent.providers.base import ModelResponse, ToolCall
from llm_agent.agent.reflection import check_all, format_message
from llm_agent.agent.trace import build as build_trace

RSA = {"timestamp": "2000-01-06 04:00:00", "secure": True, "total_violations": 0, "violations": [],
       "all_voltages": [{"bus_name": "Bus_29", "vm_pu": 0.989}],
       "load_power_factor": {"value": 0.95, "source": "default"},
       "thresholds_used": {"vm_lower_pu": 0.95, "vm_upper_pu": 1.05, "max_line_loading_pct": 100.0},
       "voltage_controlled_buses": [{"bus": 4}]}
OPTIONS = {"timestamp": "2000-01-06 20:00:00", "status": "options", "feasible": True,
           "message": "No dispatch of the units meeting these limits costs less than 22.48 €/h.",
           "options": [{"id": "cheapest", "cost_eur_per_h": 26.48,
                        "power_flow_check": {"confirmed": True, "max_loading_pct": 89.7}}],
           "optimality": {"cheapest_eur_per_h": 26.48, "lower_bound_eur_per_h": 22.48, "gap_eur_per_h": 4.0,
                          "best_possible": False}}
FAILED = {"status": "infeasible", "feasible": False,
          "no_dispatch_check": {"verdict": "not_proven", "meaning": "Not proven either way."}}
PROVED = {"status": "infeasible", "feasible": False,
          "no_dispatch_check": {"verdict": "no_dispatch_exists", "meaning": "Proved."}}
PROBABILISTIC = {"timestamp": "2000-01-06 04:00:00", "p_any_violation": 0.035, "n_samples": 200}


def _record(answer, *studies, user="What is the situation?"):
    return {"user": user, "assistant": answer,
            "tool_calls": [{"name": name, "args": dict(args)} for name, args, _ in studies],
            "tool_results": [{"name": name, "result": result} for name, _, result in studies]}


def _level_of(ledger, text):
    return next(c for c in ledger.claims if c.text.startswith(text)).level


# ---------------------------------------------------------------------------
# Evidence levels of figures
# ---------------------------------------------------------------------------

def test_figures_are_graded_by_what_produced_them():
    ledger = ev.build(_record(
        "The lowest voltage is 0.989 p.u. The cheapest option costs 26.48 €/h, and no dispatch costs less "
        "than €22.48/h. A violation has a 3.5 % chance.",
        ("run_rsa", {}, RSA), ("optimize_flexibility", {"options": True}, OPTIONS),
        ("run_probabilistic_rsa", {}, {**PROBABILISTIC, "p_any_violation_pct": 3.5})))
    assert _level_of(ledger, "0.989") == ev.POWER_FLOW
    assert _level_of(ledger, "26.48") == ev.OPTIMISED
    assert _level_of(ledger, "22.48") == ev.PROVED            # the bound, not the message repeating it
    assert _level_of(ledger, "3.5") == ev.ESTIMATED


def test_a_number_in_a_label_does_not_grade_a_figure():
    """4.00 €/h is the optimality gap, not bus 4 of the assessment."""
    ledger = ev.build(_record("No other dispatch saves more than €4.00/h.",
                              ("run_rsa", {}, RSA), ("optimize_flexibility", {}, OPTIONS)))
    claim = ledger.figures[0]
    assert claim.level == ev.PROVED and "gap" in claim.source


def test_a_setting_is_marked_assumed_unless_the_question_gave_it():
    assumed = ev.build(_record("Loads were at power factor 0.95.", ("run_rsa", {}, RSA)))
    assert assumed.figures[0].level == ev.INPUT and assumed.figures[0].note.startswith("assumed")
    given = ev.build(_record("Loads were at power factor 0.95.", ("run_rsa", {}, RSA),
                             user="Is it secure with loads at power factor 0.95?"))
    assert given.figures[0].note == "given in your question"


def test_an_invented_figure_is_unsupported():
    ledger = ev.build(_record("It moves 41.7 MW.", ("run_rsa", {}, RSA)))
    assert ledger.figures[0].level == ev.UNSUPPORTED
    assert ledger.counts()[ev.UNSUPPORTED] == 1


# ---------------------------------------------------------------------------
# Assertions, and the overstated ones
# ---------------------------------------------------------------------------

def test_no_dispatch_exists_needs_the_proof():
    claim = "No dispatch of the units exists that clears these violations."
    overstated = ev.build(_record(claim, ("optimize_flexibility", {}, FAILED))).overstated
    assert len(overstated) == 1 and "does not show that none exists" in overstated[0].note
    proved = ev.build(_record(claim, ("optimize_flexibility", {}, PROVED)))
    assert not proved.overstated and proved.assertions[0].level == ev.PROVED


def test_best_possible_needs_the_bound_to_meet_the_cost():
    claim = "The cheapest option is the best possible."
    assert ev.build(_record(claim, ("optimize_flexibility", {}, OPTIONS))).overstated
    tight = {**OPTIONS, "optimality": {**OPTIONS["optimality"], "best_possible": True}}
    assert not ev.build(_record(claim, ("optimize_flexibility", {}, tight))).overstated


def test_quoting_the_bound_is_not_overstating():
    ledger = ev.build(_record("No dispatch costs less than €22.48/h, so none saves more than €4.00/h.",
                              ("optimize_flexibility", {}, OPTIONS)))
    assert not ledger.overstated


def test_a_hedge_is_not_a_claim():
    ledger = ev.build(_record("This is a snapshot assessment, not an N-1 screen or a guarantee of security "
                              "at other times.", ("run_rsa", {}, RSA)))
    assert not ledger.overstated


def test_secure_needs_a_study():
    assert ev.build(_record("The grid is secure.")).overstated
    ledger = ev.build(_record("The grid is secure.", ("run_rsa", {}, RSA)))
    assert ledger.assertions[0].level == ev.POWER_FLOW and not ledger.overstated
    statistical = ev.build(_record("It is statistically secure.", ("run_probabilistic_rsa", {}, PROBABILISTIC)))
    assert statistical.assertions[0].level == ev.ESTIMATED


def test_the_agents_own_reading_and_advice_are_labelled():
    ledger = ev.build(_record("The low voltage is probably due to the evening peak. I recommend running an "
                              "N-1 screen next.", ("run_rsa", {}, RSA)))
    assert [c.level for c in ledger.assertions] == [ev.INTERPRETATION, ev.ADVICE]


def test_sentences_keep_their_decimals_and_dates():
    ledger = ev.build(_record("At 04:00 on 6 January 2000 the grid was secure. It found 3.88 MW of demand "
                              "response; the power-flow check had no violations.", ("run_rsa", {}, RSA)))
    texts = [c.text for c in ledger.assertions]
    assert texts[0].startswith("At 04:00 on 6 January 2000")
    assert any("3.88 MW" in t for t in texts)


# ---------------------------------------------------------------------------
# Where the findings go
# ---------------------------------------------------------------------------

def test_an_overstatement_reaches_the_review_and_the_audit_panel():
    record = _record("No dispatch exists that clears the violations.", ("optimize_flexibility", {}, FAILED))
    _f, _r, conflicts, _g = check_all(record["assistant"], record)
    assert any(c.path == "evidence" for c in conflicts)
    assert "Word it by what backs it" in format_message((), conflicts)
    assert any("does not show that none exists" in c for c in build_trace(record).concerns)


def test_groups_and_marks_for_display():
    ledger = ev.build(_record("Bus_29 is at 0.989 p.u.; the cheapest costs 26.48 €/h (`26.48` in code).",
                              ("run_rsa", {}, RSA), ("optimize_flexibility", {}, OPTIONS)))
    groups = ev.groups(ledger)
    assert [g.level for g in groups] == [ev.POWER_FLOW, ev.OPTIMISED]
    marked = ev.mark_figures(ledger_answer := "Bus_29 is at 0.989 p.u.; the cheapest costs 26.48 €/h "
                             "(`26.48` in code).", ledger)
    assert '<span class="cd-ev cd-ev-power_flow"' in marked and "`26.48`" in marked
    assert marked.count("cd-ev-optimised") == 1 and ledger_answer.count("26.48") == 2
    assert "2 power flow" not in ledger.headline() and "1 power flow" in ledger.headline()


# ---------------------------------------------------------------------------
# Which readings a question left open
# ---------------------------------------------------------------------------

GRID = {"vm_lower": 0.95, "vm_upper": 1.05, "max_loading_pct": 100.0,
        "forecasts": {"first_shared": "2000-01-06 00:00:00", "last_shared": "2000-01-07 23:45:00"}}
DEFAULTS = {"load_power_factor": None, "default_load_power_factor": 0.95}


def _keys(readings):
    return sorted({r.key for r in readings})


def test_the_agent_passing_a_default_does_not_settle_it():
    args = {"load_power_factor": 0.95, "max_line_loading_pct": 100, "data_source": "measurements"}
    found = rb.readings("run_rsa", args, RSA, "Is the grid secure at 04:00?", GRID, DEFAULTS)
    assert _keys(found) == ["data_source", "load_power_factor", "loading_limit"]
    assert [r.overrides["load_power_factor"] for r in found if r.key == "load_power_factor"] == [0.9, 1.0]


def test_what_the_question_or_the_session_settled_is_left_alone():
    found = rb.readings("run_rsa", {}, RSA, "Is it secure with forecasts, pf 0.9 and lines at 80 %?", GRID,
                        DEFAULTS)
    assert found == []
    session = rb.readings("run_rsa", {}, RSA, "Is it secure?", GRID,
                          {"load_power_factor": 0.95, "default_load_power_factor": 0.95})
    assert "load_power_factor" not in _keys(session)
    other = {**RSA, "load_power_factor": {"value": 0.9, "source": "this study"}}
    assert "load_power_factor" not in _keys(rb.readings("run_rsa", {}, other, "Is it secure?", GRID, DEFAULTS))


def test_a_load_scaling_is_not_a_loading_limit():
    assert "loading_limit" not in rb._stated("At 20:00 with loads at 125 %, can redispatch clear it?")
    assert "loading_limit" in rb._stated("Is it secure with lines at 80 %?")


def test_the_voltage_band_and_the_forecast_only_where_they_differ():
    own_band = {**RSA, "thresholds_used": {**RSA["thresholds_used"], "vm_lower_pu": 0.94, "vm_upper_pu": 1.06}}
    grid = {**GRID, "vm_lower": 0.94, "vm_upper": 1.06}
    assert "voltage_band" in _keys(rb.readings("run_rsa", {}, own_band, "secure?", grid, DEFAULTS))
    assert "voltage_band" not in _keys(rb.readings("run_rsa", {}, RSA, "secure?", GRID, DEFAULTS))
    late = {**RSA, "timestamp": "2000-01-01 04:00:00"}
    assert "data_source" not in _keys(rb.readings("run_rsa", {}, late, "secure?", GRID, DEFAULTS))


# ---------------------------------------------------------------------------
# The check
# ---------------------------------------------------------------------------

def _rsa_at(pf_breaks: float):
    """A fake assessment: insecure when the loads are at power factor `pf_breaks`."""
    def run(**kwargs):
        if abs(kwargs.get("load_power_factor", 0.95) - pf_breaks) < 1e-6:
            return {**RSA, "secure": False, "total_violations": 1, "violations": [{"element": "Bus_23"}]}
        return dict(RSA)
    return run


def test_a_reading_that_changes_the_conclusion_is_said_under_the_answer():
    record = _record("Yes, the grid is secure.", ("run_rsa", {"load_power_factor": 0.95}, RSA),
                     user="Is the grid secure at 04:00 on 6 January?")
    report = rb.check(record, grid=GRID, dispatch={"run_rsa": _rsa_at(1.0)}, settings=DEFAULTS)
    statuses = {o.reading.label: o.status for o in report.outcomes}
    assert statuses["loads at power factor 1.00"] == rb.CHANGES
    assert statuses["loads at power factor 0.90"] == rb.HOLDS
    note = rb.note(report)
    assert "loads at power factor 1.00 instead of 0.95 (the default)" in note
    assert "not secure, 1 violation" in note
    assert rb.from_record(report.as_record()).changes[0].after == report.changes[0].after


def test_a_figure_that_moves_shifts_and_a_failed_rerun_proves_nothing():
    cheap = {"status": "options", "options": [{"id": "cheapest", "cost_eur_per_h": 26.48}],
             "load_power_factor": {"value": 0.95, "source": "default"}}

    def flex(**kwargs):
        if kwargs.get("opf_current_safety_margin") == 1.0:
            return {"error": "solver stopped"}
        return {**cheap, "options": [{"id": "cheapest", "cost_eur_per_h": 95.16 if kwargs.get("load_power_factor")
                                      == 0.9 else 26.9}]}
    record = _record("Options.", ("optimize_flexibility", {"options": True}, {**cheap, "timestamp": "x"}),
                     user="What are my options?")
    report = rb.check(record, grid={}, dispatch={"optimize_flexibility": flex}, settings=DEFAULTS)
    by = {o.reading.key + str(o.reading.overrides.get("load_power_factor")): o.status for o in report.outcomes}
    assert by["load_power_factor0.9"] == rb.SHIFTS and by["load_power_factor1.0"] == rb.HOLDS
    assert by["loading_limitNone"] == rb.FAILED and rb.note(report) == "" and report.robust


def test_the_main_study_is_rechecked_first(monkeypatch):
    monkeypatch.setattr(rb, "MAX_RERUNS", 2)
    record = _record("x", ("run_rsa", {}, RSA), ("simulate_all_contingencies", {},
                                                 {**RSA, "system_n1_secure": True,
                                                  "total_outages_causing_violations": 0}),
                     user="secure?")
    ran = []

    def recorder(name):
        def run(**kwargs):
            ran.append(name)
            return {**RSA, "system_n1_secure": True, "total_outages_causing_violations": 0}
        return run
    rb.check(record, grid={}, dispatch={"run_rsa": recorder("rsa"), "simulate_all_contingencies": recorder("n1")},
             settings=DEFAULTS)
    assert ran == ["n1", "n1"]


def test_a_rerun_leaves_no_chart_and_tells_the_backend_to_keep_nothing(monkeypatch):
    sent = {}

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"ok": True}

    class Client:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def post(self, url, json):
            sent.update(json)
            return Response()
    monkeypatch.setattr(tools.httpx, "Client", Client)
    tools._last_tool_results.clear()
    with tools.unrecorded():
        tools._record("run_rsa", {"x": 1})
        tools._post("/api/flexibility/optimize", {"options": True})
    assert tools._last_tool_results == [] and sent["remember"] is False
    sent.clear()
    tools._post("/api/flexibility/optimize", {"options": True})
    assert sent == {"options": True}


def test_the_loop_runs_the_check_and_records_it(monkeypatch):
    records = []
    provider_responses = [ModelResponse(tool_calls=[ToolCall("run_rsa", {"timestamp": "2000-01-06 04:00:00"})]),
                          ModelResponse(text="Yes, the grid is secure.")]
    from tests.test_agent_loop import StubProvider
    provider = StubProvider(provider_responses)
    monkeypatch.setattr(loop_module, "get_provider", lambda *a, **k: provider)
    monkeypatch.setattr(loop_module, "get_system_prompt", lambda: "SYSTEM")
    monkeypatch.setattr(loop_module, "TOOLS", [])
    monkeypatch.setattr(loop_module, "TOOL_DISPATCH", {"run_rsa": _rsa_at(1.0)})
    monkeypatch.setattr(loop_module, "_append_to_log", records.append)
    monkeypatch.setattr(loop_module, "_grid_facts", lambda: GRID)
    monkeypatch.setattr(tools, "_get", lambda endpoint: DEFAULTS)
    text, _ = loop_module.run_agent_turn("Is the grid secure at 04:00 on 6 January?", [], reading_check=True)
    assert "**Reading check.**" in text and "power factor 1.00" in text
    assert records[-1]["reading_check"]["outcomes"]
    # CONDUCTOR's note is its own, not a claim of the answer.
    assert not any(c.text.startswith("1.00") for c in ev.build(records[-1]).figures)
    # Switched off: no check, no note.
    again = StubProvider([ModelResponse(tool_calls=[ToolCall("run_rsa", {"timestamp": "2000-01-06 04:00:00"})]),
                          ModelResponse(text="Yes.")])
    monkeypatch.setattr(loop_module, "get_provider", lambda *a, **k: again)
    off, _ = loop_module.run_agent_turn("Is the grid secure at 04:00 on 6 January?", [], reading_check=False)
    assert off == "Yes." and "reading_check" not in records[-1]


@pytest.mark.parametrize("level", ev.LEVELS)
def test_every_level_has_a_label_and_a_meaning(level):
    assert ev.LABELS[level] and ev.MEANINGS[level]


# ---------------------------------------------------------------------------
# Found in the live run across the tools
# ---------------------------------------------------------------------------

def test_a_figure_in_a_table_belongs_to_its_column():
    """0.943 p.u. sits in the Bus_29 column; the last name before it in the
    text is L0, from the header's last column."""
    rsa = {"violations": [{"element": "Bus_29", "value": 0.943}, {"element": "Bus_0 -> Bus_1 [L0]", "value": 114.7}],
           "total_violations": 2, "secure": False}
    answer = ("| Dataset | Bus_29 voltage | Line Bus_0–Bus_1 [L0] loading |\n|---|---:|---:|\n"
              "| Measurements | 0.943 p.u. | 114.7% |\n")
    from llm_agent.agent.provenance import check_turn
    verdicts = {v.claim.value: v for v in check_turn(_record(answer, ("run_rsa", {}, rsa))).verdicts}
    assert verdicts[0.943].status == "grounded" and verdicts[0.943].subject == "Bus_29"
    assert verdicts[114.7].subject == "Bus_0 -> Bus_1 [L0]"


def test_insecure_is_a_verdict_too():
    ledger = ev.build(_record("No limits are broken, but the system is not secure because load is unsupplied.",
                              ("simulate_contingency", {}, {"system_secure": False, "total_violations": 0})))
    assert ledger.assertions[0].level == ev.POWER_FLOW
    assert ev.build(_record("The grid is insecure.")).overstated


def test_a_zero_search_bound_is_left_to_the_backend(monkeypatch):
    sent = {}
    monkeypatch.setattr(tools, "_post", lambda endpoint, body: sent.update(body) or {"ok": True})
    tools.compute_hosting_capacity(bus="Bus_29", p_max_search_mw=0)
    assert "p_max_search_mw" not in sent
    tools.compute_hosting_capacity(bus="Bus_29", p_max_search_mw=12.5)
    assert sent["p_max_search_mw"] == 12.5


def test_a_count_written_as_a_share_of_a_total_is_derived():
    """“95 % (190 of 200 samples)”: 190 is 0.95 × 200, both written and traced."""
    from llm_agent.agent.provenance import check_turn
    record = _record("The probability of a violation is 95 % (190 of 200 samples).",
                     ("run_probabilistic_rsa", {}, {"p_any_violation": 0.95, "n_samples": 200}))
    verdict = next(v for v in check_turn(record).verdicts if v.claim.value == 190)
    assert verdict.status == "derived" and "p_any_violation" in verdict.source
    unwritten = _record("It failed in 190 of 200 samples.", ("run_probabilistic_rsa", {}, {"n_samples": 200}))
    assert next(v for v in check_turn(unwritten).verdicts if v.claim.value == 190).status == "ungrounded"


def test_an_n1_count_inside_kpis_is_backed_by_a_power_flow():
    kpis = {"after": {"n1_outages_with_violations": 29, "n1_outages_tested": 41}}
    ledger = ev.build(_record("The grid is not N-1 secure: 29 of 41 outages still have violations.",
                              ("evaluate_kpis", {}, kpis)))
    assert ledger.assertions[0].level == ev.POWER_FLOW


def test_a_figure_is_marked_where_it_stands_not_inside_a_name():
    from llm_agent.agent.provenance import extract_claims
    # Observed live: "0" marked inside "[L30]", and the "0" of "N-0" and the "1" of "N-1"
    # taken as figures.
    answer = ("Bus_29 is at 0.989 p.u. on Bus_21 -> Bus_23 [L30] and Bus_10; this is an N-0 assessment, "
              "not an N-1 check, with 0 violations.")
    assert [c.value for c in extract_claims(answer)] == [0.989, 0.0]
    marked = ev.mark_figures(answer, ev.build(_record(answer, ("run_rsa", {}, RSA))))
    assert "[L30]" in marked and "N-0" in marked and "N-1" in marked and "Bus_10" in marked
    assert marked.count('<span class="cd-ev') == 2
    assert marked.endswith('>0</span> violations.')
