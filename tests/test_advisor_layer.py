"""
The advisor and the CIM layer (`advisor_layer.py`), for any network:

1. it is shown what the layer says about the elements the latest results name, with how to use it — only when
   there is something to show, so its system prompt does not grow;
2. what it says (comment, suggestions, replies, notes) is checked against the layer like an answer;
3. when the conclusion moved with what units the network does not say are, it asks the operator — and if its
   reply did not, a question is added; it never assumes the fact;
4. a group its study names by meaning is resolved against the layer before the operator sees it.
"""

import json

import pytest

from llm_agent.agent import advisor_layer as al
from llm_agent.agent import loop, recommender
from llm_agent.agent.providers.base import ModelResponse


class FakeProvider:
    name, model = "fake", "fake-model"

    def __init__(self, text):
        self.text, self.calls = text, []

    def chat(self, messages, tools, system, on_event=None):
        self.calls.append({"messages": messages, "system": system})
        return ModelResponse(text=self.text)


BUS14 = {"name": "Bus 14", "kind": "bus", "kv": {"value": 20.0, "basis": "stated"},
         "substation": {"value": "Bus 14", "basis": "inferred"},
         "feeder": {"value": "Line 12-13", "basis": "inferred", "position": 2, "of": 2},
         "units": [], "unit_count": 0, "loads": [{"name": "Load R14", "value": "unknown", "basis": "unknown"}],
         "load_count": 1, "load_mw": 0.243}

READING = {"headline": "Depends on the reading", "outcomes": [
    {"tool": "run_rsa", "reading": "unit_sources", "label": "Gen_bus2, Gen_bus5 counted as wind or solar",
     "assumed": "not wind or solar (the network does not say what they are)",
     "overrides": {"assume_unit_sources": {"Gen_bus2": "wind", "Gen_bus5": "wind"}},
     "status": "changes", "before": "secure", "after": "not secure"},
    {"tool": "run_rsa", "reading": "load_power_factor", "label": "loads at power factor 0.90", "assumed": "0.95",
     "overrides": {}, "status": "changes", "before": "secure", "after": "not secure"}]}


def _record(**extra):
    return {"user": "Is the grid secure at 30 % renewables?", "turn_key": "t1",
            "tool_calls": [{"name": "run_rsa", "args": {"sgen_scaling_factor": 1.3}}],
            "tool_results": [{"name": "run_rsa", "result": {"timestamp": "2000-01-03 18:00:00", "secure": True,
                                                            "total_violations": 0, "violations": []}}],
            "assistant": "The grid is secure.", **extra}


# --- 1. What the advisor is shown -----------------------------------------------------------------------------

def test_the_advisor_is_shown_the_layer_only_when_there_is_something_to_show():
    rec = _record(cim_context={"elements": {"Bus 14": BUS14}, "where": ["feeder: all 2 on Line 12-13"],
                               "changed": []})
    block = al.context_block(rec)
    assert "- Bus 14: bus · 20 kV (stated) · feeder Line 12-13 (inferred), bus 2 of 2" in block
    assert "0.243 MW at the study's moment" in block and "Together: feeder: all 2 on Line 12-13" in block
    assert "never from its name or a guess" in block and "by meaning" in block
    assert al.context_block(_record()) == "" and al.gap_block(al.fact_gaps(_record())) == ""
    request = recommender.build_request([], rec, {}, [], [rec])
    assert "## What the CIM layer says about the elements" in request
    # The rules travel with the facts: the system prompt does not grow.
    assert len(recommender.build_system_prompt()) // 4 < 3000
    assert "CIM layer says about the elements" not in recommender.build_system_prompt()


# --- 3. What the network does not say -------------------------------------------------------------------------

def test_what_the_conclusion_depends_on_that_the_network_does_not_say():
    gaps = al.fact_gaps(_record(reading_check=READING))
    assert gaps == [{"tool": "run_rsa", "units": ["Gen_bus2", "Gen_bus5"],
                     "label": "Gen_bus2, Gen_bus5 counted as wind or solar", "status": "changes",
                     "assumed": "not wind or solar (the network does not say what they are)",
                     "before": "secure", "after": "not secure"}]           # the power factor reading is not a fact
    block = al.gap_block(gaps)
    assert "secure → not secure" in block and "Never assume what they are" in block
    assert "The units: Gen_bus2, Gen_bus5" in block                 # by name, not "… and 2 more"
    added = al.gap_suggestion(gaps, _record())
    assert added.angle == "step_back" and added.title == "Say what Gen_bus2 and Gen_bus5 are"
    assert added.prompt.startswith("If you know what Gen_bus2 and Gen_bus5 are") and "2000-01-03 18:00:00" in added.prompt
    assert al.asks_about(added.prompt, gaps) and not al.asks_about("Run N-1 at 18:00", gaps)


def _four(*extra):
    return [{"angle": a, "title": f"T {a}", "prompt": f"Run something {a} at 2000-01-03 18:00.", "why": ""}
            for a in ("deeper", "stress", "change", "step_back")][:4 - len(extra)] + list(extra)


def test_the_question_is_added_when_the_advisor_did_not_ask(monkeypatch):
    monkeypatch.setattr(loop, "_network_facts", lambda results, text: {})
    rec = _record(reading_check=READING)
    provider = FakeProvider(json.dumps({"comment": "Secure at 18:00.", "suggestions": _four()}))
    result = recommender.suggest([], rec, [rec], {}, provider=provider)
    assert len(result.suggestions) == 4 and result.suggestions[-1].title == "Say what Gen_bus2 and Gen_bus5 are"
    assert "What the conclusion depends on that the network does not say" in provider.calls[0]["messages"][0]["content"]
    # Asked already: nothing added.
    asked = {"angle": "step_back", "title": "What drives Gen_bus2?", "prompt": "If you know what Gen_bus2 is, "
             "declare it, then rerun run_rsa at 2000-01-03 18:00.", "why": "the conclusion changes"}
    result = recommender.suggest([], rec, [rec], {}, provider=FakeProvider(json.dumps({"suggestions": _four(asked)})))
    assert [s.title for s in result.suggestions].count("What drives Gen_bus2?") == 1
    assert not any(s.title.startswith("Say what") for s in result.suggestions)


# --- 2. What the advisor says, checked -----------------------------------------------------------------------

FACTS = {"base": {"per_unit_case": False, "voltage_levels_kv": [110.0, 20.0], "external_grids": [],
                  "feeders": ["Line 1-2", "Line 12-13"], "units_by_source": {"solar": ["PV 3"]}, "units_unknown": 1,
                  "buses": {"Bus 14": {"kv": 20.0, "kv_basis": "stated", "substation": "Bus 14",
                                       "feeder": "Line 12-13"}},
                  "branches": {}, "loads": {},
                  "units": {"Gen_bus2": {"source": "unknown", "basis": "unknown", "bus": "Bus 2", "type": "gen",
                                         "feeder": None}},
                  "names_not_in_network": []}}


def test_what_the_advisor_says_is_checked_against_the_layer(monkeypatch):
    monkeypatch.setattr(loop, "_network_facts", lambda results, text: FACTS)
    reply = json.dumps({"comment": "Gen_bus2, a wind farm, carries the evening peak.", "suggestions": [
        {"angle": "deeper", "title": "Bus 14 on feeder Line 1-2", "prompt": "Run N-1 at 2000-01-03 18:00 for Bus 14.",
         "why": "Bus 14 sits at the end of feeder Line 1-2."},
        {"angle": "stress", "title": "Check Bus 14", "prompt": "Run RSA at 2000-01-03 18:00.",
         "why": "Bus 14 is at the far end of feeder Line 12-13."}]})
    result = recommender.suggest([], _record(), [_record()], {}, provider=FakeProvider(reply))
    assert result.comment_layer and "does not say what drives it" in result.comment_layer[0]
    wrong, right = result.suggestions
    assert wrong.layer and "on feeder Line 12-13" in wrong.layer[0] and right.layer == ()
    assert wrong.as_record()["layer"] == list(wrong.layer)


def test_the_discussion_and_the_note_are_checked_too(monkeypatch):
    monkeypatch.setattr(loop, "_network_facts", lambda results, text: FACTS)
    text = json.dumps({"reply": "Gen_bus2 is a wind farm, in my view the cause.",
                       "studies": [{"title": "Scale Gen_bus2", "prompt": "Run RSA at 2000-01-03 18:00.",
                                    "why": "Gen_bus2, a solar plant, may matter."}]})
    reply = recommender.discuss("Why?", [], [], _record(), [_record()], {}, provider=FakeProvider(text))
    assert reply.layer and reply.studies[0].layer
    note = recommender.decision_note([], [], _record(), [_record()], {}, provider=FakeProvider(json.dumps({
        "decision": "Treat Gen_bus2 as a wind farm.", "evidence": [], "open_risks": [], "studies_run": []})))
    assert note.layer and note.as_record()["layer"] == list(note.layer)


# --- 4. Groups by meaning ---------------------------------------------------------------------------------------

VOCAB = {"feeders": ["Line 12-13", "Line 1-2"], "substations": ["Bus 5"], "levels": [110.0, 20.0]}


def test_group_phrases_are_found_by_their_place():
    text = ("Study curtailing the solar units on feeder Line 1-2 at 2000-01-03 18:00, then the loads at 20 kV, and "
            "compare with the units in substation Bus 5. Bus 14 is low.")
    assert al.candidates(text, VOCAB) == ["solar units on feeder Line 1-2", "loads at 20 kV",
                                          "units in substation Bus 5"]
    assert al.candidates("Run N-1 at 18:00 on Line 1-2.", VOCAB) == []
    # As the advisor wrote it live: the name before "feeder" — written back the way the layer reads it.
    assert al.candidates("report the loads on the Line 12-13 feeder and the units in the Bus 5 substation",
                         VOCAB) == ["loads on feeder Line 12-13", "units in substation Bus 5"]


def test_a_phrase_is_resolved_or_said_why_not(monkeypatch):
    answers = {"solar units on feeder Line 1-2": {"means": "solar units in feeder Line 1-2",
                                                  "elements": [{"name": f"PV {i}"} for i in range(3, 12)]},
               "thermal units on feeder Line 12-13": {"error": "400", "detail": "No thermal units in feeder Line 12-13."},
               "nice units on feeder Line 1-2": {"error": "400", "detail": "Could not read 'nice units'."},
               "units on feeder Line 1-2": {"means": "units in feeder Line 1-2",
                                            "elements": [{"name": "PV 3"}, {"name": "WKA 7"}]},
               "the moon units on feeder Line 1-2": {"error": "400", "detail": "Could not read 'the moon units'."}}

    def fake_get(endpoint):
        from urllib.parse import parse_qs, urlparse
        return answers[parse_qs(urlparse(endpoint).query)["element"][0]]
    monkeypatch.setattr(al, "_get", fake_get)
    out = al.group_phrases(["Curtail the solar units on feeder Line 1-2, then the thermal units on feeder "
                            "Line 12-13 and the nice units on feeder Line 1-2."], VOCAB)
    assert out[0]["count"] == 9 and out[0]["elements"] == ["PV 3", "PV 4", "PV 5", "PV 6", "PV 7", "PV 8"]
    assert out[1] == {"phrase": "thermal units on feeder Line 12-13", "error": "No thermal units in feeder Line 12-13."}
    # A word the layer cannot read is dropped, and the phrase shown is the one it resolved.
    assert out[2]["phrase"] == "units on feeder Line 1-2" and out[2]["count"] == 2 and len(out) == 3
    assert al.group_line(out[0]).endswith("PV 8 and 3 more") and "cannot resolve" in al.group_line(out[1])


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    """The backend in-process on CIGRE MV; every file it writes goes to a temporary folder."""
    pytest.importorskip("fastapi.testclient")
    import main_backend as mb
    from fastapi.testclient import TestClient

    scratch = tmp_path_factory.mktemp("advisor_layer")
    patch = pytest.MonkeyPatch()
    patch.setenv("GRID_PROFILE", "cigre_mv")
    for name in ("_LAST_UPLOAD_SENTINEL", "_LAST_TIMESERIES_SENTINEL", "_LAST_FORECAST_SENTINEL",
                 "_PROPOSALS_FILE", "_ASSUMPTIONS_FILE", "_SCENARIO_FILE"):
        patch.setattr(mb, name, str(scratch / f"{name}.json"))
    patch.setattr(mb, "DATA_FILES_DIR", str(scratch / "data_files"))
    patch.setattr(mb, "SYSTEMS_DIR", str(scratch / "systems"))
    saved = dict(mb.app_data)
    with TestClient(mb.app) as c:
        yield c
    mb.app_data.clear()
    mb.app_data.update(saved)
    patch.undo()


def test_phrases_resolve_against_the_real_layer(client, monkeypatch):
    def get(endpoint):
        r = client.get(endpoint)
        return r.json() if r.status_code == 200 else {"error": str(r.status_code), "detail": r.json().get("detail")}
    monkeypatch.setattr(al, "_get", get)
    vocab = al._vocabulary()
    assert "Line 1-2" in vocab["feeders"] and 20.0 in vocab["levels"]
    out = al.group_phrases(["Study curtailing the solar units on feeder Line 1-2 at 18:00 and the thermal units "
                            "on feeder Line 12-13."], vocab)
    solar, thermal = out
    assert solar["count"] >= 5 and all(n.startswith("PV") for n in solar["elements"])
    assert "No thermal units" in thermal["error"]
