"""
A study's result linked to the CIM layer, for any network (`grid_context`, `semantics.link_targets/context`):

- the elements a result names as a problem are found generically (a violation, an outage, a probability keyed by
  name), never the lists of every element;
- the logged result keeps only links (mRIDs and the layer's version); the model's copy gets what the layer says
  each element is; the advisor and the checks never read the mRIDs' digits as figures;
- the links resolve later, and say when the layer has changed since the study ran;
- the turn's record keeps the links resolved, and the answer checks read them: a figure from the layer traces, a
  wrong feeder is caught;
- off when the agent is not given the layer.
"""

import pandapower.networks as pn
import pytest

import semantics as sm
from llm_agent.agent import grid_context as gc
from llm_agent.agent import loop as loop_module
from llm_agent.agent.entities import CONTRADICTED, UNSUPPORTED, check_answer
from llm_agent.agent.provenance import check_turn
from llm_agent.agent.providers.base import ModelResponse, ToolCall


@pytest.fixture(scope="module")
def cigre():
    net = pn.create_cigre_network_mv(with_der="all")
    return net, sm.build(net, network_id="cigre")


# --- Finding the elements a result names as a problem ----------------------------------------------------------

def test_names_are_found_under_problem_fields_only():
    result = {"violations": [{"element_name": "Bus 14", "element_type": "bus", "vm_pu": 0.91}],
              "all_voltages": [{"bus_name": "Bus 1", "vm_pu": 1.0}],
              "gen_dispatch": [{"name": "PV 3", "Pg_mw": 0.02}],
              "bus_violation_probability": {"Bus 13": 0.4},
              "outages_disconnecting_load": [{"outage": "Line 12-13", "buses": ["Bus 13"]}],
              "voltage_percentiles": {"Bus 2": {"p5": 0.99}}}
    texts = [t for t, _h in gc.candidates(result)]
    assert "Bus 14" in texts and "Bus 13" in texts and "Line 12-13" in texts
    assert not {"Bus 1", "PV 3", "Bus 2"} & set(texts)
    assert ("Bus 14", "bus") in gc.candidates(result)              # the record's element_type is the hint


# --- The layer's side: links, facts, version ------------------------------------------------------------------

def test_texts_link_to_elements_by_name_or_by_their_start(cigre):
    _net, layer = cigre
    targets = sm.link_targets(layer, [("Bus 14 at 0.912 p.u.", "bus"), ("Line 12-13", None), ("Bus 1", None),
                                      ("Bus 14", None), ("the north side", None), ("Bus 12-13", None)])
    assert [(t["name"].split(" | ")[0], t["type"]) for t in targets] == [
        ("Bus 14", "bus"), ("Line 12-13", "line"), ("Bus 1", "bus")]     # each once; nothing invented
    assert len(sm.link_targets(layer, [(b["name"], None) for b in layer["buses"]])) == sm.MAX_LINKED


def test_what_the_layer_says_about_linked_elements(cigre):
    net, layer = cigre
    targets = sm.link_targets(layer, [("Bus 14", None), ("Line 12-13", None), ("PV 3", None), ("Bus 1", None)])
    ctx = {f["name"].split(" | ")[0]: f for f in sm.context(net, layer, targets).values()}
    bus = ctx["Bus 14"]
    assert bus["kv"] == {"value": 20.0, "basis": "stated"}
    assert bus["feeder"]["value"] == "Line 12-13" and bus["feeder"]["position"] == bus["feeder"]["of"] == 2
    assert {d["name"] for d in bus["loads"]} == {"Load R14", "Load CI14"} and bus["unit_count"] == 0
    line = ctx["Line 12-13"]
    assert line["without_it"]["buses_cut_off"] == 2 and line["without_it"]["load_count_cut_off"] == 3
    # No demand: it is the clock's moment, not the study's — the study reports the MW it cut, at its own moment.
    assert not [k for f in ctx.values() for k in (f.get("without_it") or {}) if "mw" in k.lower()]
    assert ctx["PV 3"]["source"] == {"value": "solar", "basis": "stated"}
    assert ctx["Bus 1"]["feeder"]["value"] is None and "busbar" in ctx["Bus 1"]["feeder"]["why"]
    # Where they sit together, from the full facts or from the cheap ones every named element gets.
    together = sm.where(sm.context(net, layer, targets))
    assert together == sm.where(sm.locate(layer, targets)) == [
        "feeder: 2 of 4 on Line 12-13, 1 on Line 1-2, 1 on no feeder",
        "voltage: all 3 at 20 kV"]                          # buses and lines only: a unit has no voltage here


def test_the_version_follows_the_layer_not_the_operating_point():
    net = pn.create_cigre_network_mv(with_der="all")
    before = sm.layer_version(sm.build(net, network_id="c"))
    net.load.loc[:, "p_mw"] *= 1.3
    assert sm.layer_version(sm.build(net, network_id="c")) == before
    assert sm.layer_version(sm.build(net, {"PV 3": "wind"}, network_id="c")) != before


def test_a_meshed_network_has_no_feeders_and_says_why():
    net = pn.case30()
    layer = sm.build(net, network_id="ieee")
    targets = sm.link_targets(layer, [(layer["buses"][20]["name"], None)])
    (facts,) = sm.context(net, layer, targets).values()
    assert facts["feeder"]["value"] is None and "meshed" in facts["feeder"]["why"]
    assert facts["next_to"] and "· no feeder · next to " in gc.describe(facts)      # where it is, without a feeder
    view = gc.view({"m": facts}, [], "v")
    assert "meshed" in view["no_feeder"] and "meshed" not in view["elements"][facts["name"]]   # said once


# --- What the model reads, and what the card shows ---------------------------------------------------------------

def test_every_named_element_is_placed_not_only_those_described(client):
    texts = [[f"Bus {i}", "bus"] for i in range(1, 15)]
    made = client.post("/api/network/cim/links", json={"names": texts}).json()
    assert len(made["elements"]) == sm.MAX_LINKED and made["named"] == 14
    assert made["where"][0] == "feeder: 10 of 14 on Line 1-2, 2 on Line 12-13, 2 on no feeder"
    view = gc.view(made["context"], made["where"], made["version"], made["named"])
    assert "of the 14 the result names" in view["covers"]


def test_the_model_reads_one_line_per_element(cigre):
    net, layer = cigre
    targets = sm.link_targets(layer, [("Bus 14", None), ("Line 12-13", None)])
    ctx = sm.context(net, layer, targets)
    view = gc.view(ctx, sm.where(ctx), "v1")
    assert view["elements"]["Bus 14"] == ("bus · 20 kV (stated) · feeder Line 12-13 (inferred), bus 2 of 2 from the "
                                          "substation, its far end · no units · loads: Load R14 (not stated), "
                                          "Load CI14 (not stated)")
    assert "head of feeder Line 12-13" in view["elements"]["Line 12-13"]
    assert "2 bus(es) and 3 load(s) lose supply" in view["elements"]["Line 12-13"]
    assert " MW)" not in view["elements"]["Line 12-13"] and "the MW lost is the study's" in view["elements"]["Line 12-13"]
    assert "version v1" in view["from"]
    rows = gc.rows({"context": ctx})
    assert rows[0]["Where"] == "feeder Line 12-13 · bus 2 of 2 (far end)" and rows[1]["Where"].endswith("(its head)")


# --- In the loop: links logged, facts to the model, the record resolved ---------------------------------------

LINKED = {"version": "abc123", "network": None,
          "elements": [{"name": "Bus 14", "type": "bus", "mrid": "76d18215-8353-5d9b-91ef-983e9aed2d12"}],
          "context": {"76d18215-8353-5d9b-91ef-983e9aed2d12": {
              "name": "Bus 14", "kind": "bus", "kv": {"value": 20.0, "basis": "stated"},
              "substation": {"value": "Bus 14", "basis": "inferred"},
              "feeder": {"value": "Line 12-13", "basis": "inferred", "position": 2, "of": 2},
              "units": [], "unit_count": 0, "loads": [{"name": "Load R14", "value": "unknown", "basis": "unknown"}],
              "load_count": 1}},
          "where": []}


class _Provider:
    name, model = "stub", "stub-1"

    def __init__(self, responses):
        self.responses, self.sent = list(responses), []

    def preflight(self):
        pass

    def chat(self, messages, tools, system, on_event=None):
        self.sent.append([dict(m) for m in messages])
        return self.responses.pop(0) if self.responses else ModelResponse(text="(done)")


def _rsa(**_kwargs):
    from llm_agent.agent import tools
    return tools._record("run_rsa", {"violations": [{"element_name": "Bus 14", "element_type": "bus",
                                                      "vm_pu": 0.912}], "total_violations": 1})


def _turn(monkeypatch, answer, resolved_changed=False, ontology="on"):
    monkeypatch.setenv("CONDUCTOR_ONTOLOGY", ontology)
    provider = _Provider([ModelResponse(tool_calls=[ToolCall("run_rsa", {})]), ModelResponse(text=answer)])
    records = []
    monkeypatch.setattr(loop_module, "get_provider", lambda *a, **k: provider)
    monkeypatch.setattr(loop_module, "get_system_prompt", lambda: "SYSTEM")
    monkeypatch.setattr(loop_module, "TOOLS", [])
    monkeypatch.setattr(loop_module, "TOOL_DISPATCH", {"run_rsa": _rsa})
    monkeypatch.setattr(loop_module, "_append_to_log", records.append)
    monkeypatch.setattr(loop_module, "_network_facts", lambda results, answer: {})
    monkeypatch.setattr(gc, "_backend_post", lambda endpoint, body: dict(LINKED))
    monkeypatch.setattr(gc, "_backend_get", lambda endpoint: {**LINKED, "changed": resolved_changed,
                                                              "version": "def456" if resolved_changed else "abc123"})
    loop_module.run_agent_turn("Where is the undervoltage?", [])
    return provider, records[-1]


def test_the_logged_result_keeps_links_and_the_model_reads_facts(monkeypatch):
    provider, record = _turn(monkeypatch, "Bus 14 is at 0.912 p.u., at the far end of feeder Line 12-13 (20 kV).")
    logged = record["tool_results"][0]["result"]
    assert logged["cim_links"] == {"version": "abc123", "elements": [
        {"name": "Bus 14", "mrid": "76d18215-8353-5d9b-91ef-983e9aed2d12"}]}
    assert "grid_context" not in logged                                # the facts live in the layer only
    sent = str(provider.sent[-1])
    assert "grid_context" in sent and "Line 12-13" in sent and "76d18215" not in sent
    # The record keeps the links resolved; the layer's figures trace, the mRIDs' digits never do.
    assert record["cim_context"]["elements"]["Bus 14"]["kv"]["value"] == 20.0
    # The answer card's copy carries the links too (its "Where" section resolves them).
    from llm_agent.agent import tools
    (name, chart), = tools._last_tool_results
    assert name == "run_rsa" and chart["cim_links"] == logged["cim_links"]
    verdicts = {v.claim.value: (v.status, v.source) for v in check_turn(record).verdicts}
    assert verdicts[20.0][0] == "grounded" and verdicts[20.0][1].startswith("cim layer:Bus 14")
    assert verdicts[0.912][0] == "grounded"


def test_a_change_of_the_layer_since_the_study_is_recorded(monkeypatch):
    _provider, record = _turn(monkeypatch, "Bus 14 is low.", resolved_changed=True)
    assert record["cim_context"]["changed"] == ["run_rsa"]


def test_nothing_is_linked_without_the_layer(monkeypatch):
    provider, record = _turn(monkeypatch, "Bus 14 is low.", ontology="off")
    assert "cim_links" not in record["tool_results"][0]["result"] and "cim_context" not in record
    assert "grid_context" not in str(provider.sent[-1])


def test_a_failing_backend_costs_only_the_context(monkeypatch):
    def down(endpoint, body):
        raise ConnectionError("backend down")
    monkeypatch.setenv("CONDUCTOR_ONTOLOGY", "on")
    monkeypatch.setattr(gc, "_backend_post", down)
    assert gc.link("run_rsa", {"violations": [{"element_name": "Bus 14"}]}) == (None, None)
    assert gc.link("describe_network", {"violations": [{"element_name": "Bus 14"}]}) == (None, None)


def test_the_advisor_never_reads_the_links():
    from llm_agent.agent.recommender import compact_turn
    record = {"user": "q", "tool_calls": [{"name": "run_rsa", "args": {}}],
              "tool_results": [{"name": "run_rsa", "result": {"total_violations": 1, "cim_links": LINKED}}]}
    assert "cim_links" not in str(compact_turn(record))


# --- The answer check on feeders ------------------------------------------------------------------------------

FACTS = {"base": {"per_unit_case": False, "voltage_levels_kv": [110.0, 20.0], "external_grids": [],
                  "feeders": ["Line 1-2", "Line 12-13"], "units_by_source": {}, "units_unknown": 0,
                  "buses": {"Bus 14": {"kv": 20.0, "kv_basis": "stated", "substation": "Bus 14", "feeder": "Line 12-13"},
                            "Bus 1": {"kv": 20.0, "kv_basis": "stated", "substation": "S0", "feeder": None},
                            "Bus 5": {"kv": 20.0, "kv_basis": "stated", "substation": "Bus 5", "feeder": "Line 1-2"}},
                  "branches": {}, "units": {},
                  "loads": {"Load R14": {"category": "unknown", "basis": "unknown", "bus": "Bus 14",
                                         "feeder": "Line 12-13"}}}}


@pytest.mark.parametrize("answer, kind", [
    ("Bus 14 sits at the far end of feeder Line 1-2.", CONTRADICTED),
    ("Load R14, on the Line 1-2 feeder, is the largest.", CONTRADICTED),
    ("Bus 1 is on feeder Line 12-13.", UNSUPPORTED),
    ("Bus 14 sits at the far end of feeder Line 12-13.", None),
    ("If Bus 14 were on feeder Line 1-2, it would be fine.", None),           # held back
    ("Bus 14 is low, while Bus 5 on feeder Line 1-2 is not.", None),           # another name in between
    ("Bus 14 is not on feeder Line 1-2.", None),                               # denied
])
def test_a_feeder_said_of_an_element_is_checked(answer, kind):
    issues = [i for i in check_answer(answer, {"network_facts": FACTS, "user": ""}) if "feeder" in i.note]
    assert [i.kind for i in issues] == ([kind] if kind else [])


# --- The backend: links made and resolved -----------------------------------------------------------------------

@pytest.fixture(scope="module")
def client(tmp_path_factory):
    """The backend in-process on CIGRE MV; every file it writes goes to a temporary folder."""
    pytest.importorskip("fastapi.testclient")
    import main_backend as mb
    from fastapi.testclient import TestClient

    scratch = tmp_path_factory.mktemp("grid_context")
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


def test_a_study_result_links_and_resolves(client):
    rsa = client.post("/api/grid/rsa", json={}).json()
    names = gc.candidates({"violations": [{"element_name": "Bus 14"}], "outage": "Line 12-13"})
    made = client.post("/api/network/cim/links", json={"names": [list(p) for p in names]}).json()
    assert [e["name"].split(" | ")[0] for e in made["elements"]] == ["Bus 14", "Line 12-13"]
    assert rsa.get("converged") is not False
    mrids = ",".join(e["mrid"] for e in made["elements"])
    now = client.get("/api/network/cim/resolve", params={"mrids": mrids + ",not-an-mrid",
                                                         "version": made["version"]}).json()
    assert now["changed"] is False and len(now["context"]) == 2 and now["missing"] == ["not-an-mrid"]
    # A declaration changes the layer: the links say so, and resolve to what the layer says now.
    client.post("/api/network/declare", json={"loads": [{"load": "Load R14", "category": "residential"}]})
    try:
        later = client.get("/api/network/cim/resolve", params={"mrids": mrids, "version": made["version"]}).json()
        assert later["changed"] is True
        bus = next(f for f in later["context"].values() if f["name"] == "Bus 14")
        r14 = next(d for d in bus["loads"] if d["name"] == "Load R14")
        assert (r14["value"], r14["basis"]) == ("residential", "stated")
        assert r14["p_kind"] in ("measured", "allocated") and r14["q_kind"] == "derived"   # how the data give it
    finally:
        client.post("/api/network/declare", json={"reset": True})


# --- MW at the study's moment, only when it is the study's ---------------------------------------------------

@pytest.mark.parametrize("tool, kwargs, has_moment", [
    ("run_rsa", {}, True),
    ("simulate_contingency", {"element_type": "line", "element_index": 10, "vm_lower_pu": 0.9}, True),
    ("simulate_all_contingencies", {"network": "base", "load_scaling_factor": 1.0}, True),
    # As a model sent it live: every default spelled out, an empty option.
    ("simulate_contingency", {"option": "", "element_type": "line", "element_index": 10, "load_scaling_factor": 1.0,
                              "vm_upper_pu": 1.05, "vm_lower_pu": 0.95, "max_line_loading_pct": 100,
                              "max_trafo_loading_pct": 100, "network": "base", "data_source": "measurements",
                              "timestamp": "2000-01-01 00:00:00"}, True),
    ("run_rsa", {"load_scaling_factor": 1.3}, False),               # the study's demand is not the data's
    ("run_rsa", {"network": "dc_bus9"}, False),                     # a proposal's loads are not the base's
    ("run_rsa", {"option": "A"}, False),                            # an option may relieve load
    ("optimize_contingency", {"element_type": "line", "element_index": 10}, False),
    ("run_probabilistic_rsa", {}, False),                           # sampled demand
    ("scan_scenarios", {}, False),                                  # many moments
    ("run_rsa", {"some_new_argument": 1}, False),                   # not known to leave demand as it is
])
def test_the_moment_is_sent_only_when_demand_is_the_datas(tool, kwargs, has_moment):
    moment = gc.study_moment(tool, {"timestamp": "2000-01-01 18:00:00", "data_source": "measurements"}, kwargs)
    assert (moment is not None) == has_moment
    if moment:
        assert moment == {"timestamp": "2000-01-01 18:00:00", "data_source": "measurements"}
    assert gc.study_moment("run_rsa", {"timestamps": ["a", "b"]}, {}) is None      # no single moment


def test_the_mw_a_branch_cuts_is_the_studys_at_its_moment(client):
    now = client.get("/api/time/current").json()
    ts = now.get("current_timestamp") or now.get("timestamp")
    study = client.post("/api/contingency/simulate", json={"element_type": "line", "element_index": 10,
                                                           "timestamp": ts}).json()
    made = client.post("/api/network/cim/links", json={
        "names": [["Line 12-13", "branch"], ["Bus 14", "bus"]],
        "moment": {"timestamp": study["timestamp"], "data_source": "measurements"}}).json()
    line = next(f for f in made["context"].values() if f["kind"] == "line")
    assert line["without_it"]["load_cut_off_mw"] == pytest.approx(study["unsupplied_load_mw"], abs=1e-3)
    assert made["moment"]["timestamp"] == study["timestamp"] and made["moment"]["data_key"]
    bus = next(f for f in made["context"].values() if f["kind"] == "bus")
    assert bus["load_mw"] > 0
    view = gc.view(made["context"], made["where"], made["version"], made["named"], made["moment"])
    assert "MW at the study's moment" in view["elements"]["Line 12-13"] and study["timestamp"] in view["demand"]
    # Resolved later through the links: the same MW at the same moment, whatever the clock says then.
    client.post("/api/time/advance", json={"steps": 40})
    assert (client.get("/api/time/current").json().get("current_timestamp") or "") != ts
    links = {"version": made["version"], "moment": made["moment"],
             "elements": [{"name": e["name"], "mrid": e["mrid"]} for e in made["elements"]]}
    again = client.get("/api/network/cim/resolve", params={
        "mrids": ",".join(e["mrid"] for e in links["elements"]), "version": links["version"],
        **{k: made["moment"][k] for k in ("timestamp", "data_source", "data_key")}}).json()
    line_again = next(f for f in again["context"].values() if f["kind"] == "line")
    assert line_again["without_it"]["load_cut_off_mw"] == line["without_it"]["load_cut_off_mw"]
    # Other data loaded since: no MW, and it says why.
    stale = client.get("/api/network/cim/resolve", params={
        "mrids": links["elements"][0]["mrid"], "timestamp": ts, "data_source": "measurements",
        "data_key": "other-data"}).json()
    assert "Other data" in stale["demand"] and "moment" not in stale
    assert not [k for f in stale["context"].values() for k in (f.get("without_it") or {}) if "mw" in k]


def test_no_moment_no_mw_and_a_moment_the_data_do_not_have_neither(client):
    names = [["Line 12-13", "branch"], ["Bus 14", "bus"]]
    for body in ({"names": names}, {"names": names, "moment": {"timestamp": "1999-01-01 00:00:00"}},
                 {"names": names, "network": "a-proposal", "moment": {"timestamp": "x"}}):
        out = client.post("/api/network/cim/links", json=body)
        if body.get("network"):
            assert out.status_code >= 400 or "moment" not in out.json()     # an unknown proposal is refused
            continue
        made = out.json()
        assert "moment" not in made
        assert all("load_mw" not in f and "load_cut_off_mw" not in (f.get("without_it") or {})
                   for f in made["context"].values())


def test_a_status_with_restoring_ties_reads_as_lines_not_markup():
    from llm_agent.agent.renderers import render_contingency_violations
    fig = render_contingency_violations({
        "violations": [], "system_secure": False, "message": "The outage cuts off 0.2588 MW of load: not secure.",
        "ties_that_could_restore": [{"tie": "Line 14-8 (switch S1, open at Bus 8)", "restores": 2,
                                     "within_limits": True}]})
    status = fig.layout.meta["conductor_status"]
    assert "<br>" not in status and status.startswith("The outage cuts off 0.2588 MW") and "  \n" in status


# --- The element studied, how values are obtained, normally open points, the report, the cache -----------------

def test_the_arguments_name_the_element_studied():
    texts, refs = gc.studied({"element_type": "line", "element_index": 10, "timestamp": "2000-01-01 00:00:00",
                              "network": "base", "disabled_generators": ["Gen_bus2", "unknown units"],
                              "hold_units": [{"unit": "PV 3"}], "bus": "Bus 14", "vm_lower_pu": 0.95})
    assert refs == [{"type": "line", "index": 10}]
    assert ("Bus 14", "bus") in texts and ("PV 3", "unit") in texts and ("Gen_bus2", "unit") in texts
    assert not any(t in ("2000-01-01 00:00:00", "base") for t, _ in texts)       # when and on what: not elements


def test_the_element_studied_is_linked_first_and_marked(client):
    made = client.post("/api/network/cim/links", json={
        "names": [["Bus 13", "bus"], ["Bus 14", "bus"]], "refs": [{"type": "line", "index": 10}],
        "studied": [["Bus 14", "bus"]]}).json()
    first, *rest = made["elements"]
    assert first["name"].startswith("Line 12-13") and first["role"] == "studied"
    assert [e["name"] for e in made["elements"]].count("Bus 14") == 1             # once, as studied
    assert {e["name"] for e in rest if e.get("role") == "studied"} == {"Bus 14"}
    view = gc.view(made["context"], made["where"], made["version"], made["named"], None,
                   roles={e["mrid"]: e.get("role") for e in made["elements"]}, ties=made.get("ties"))
    assert view["studied"] == ["Line 12-13", "Bus 14"]


def test_a_result_with_no_problem_still_links_the_element_studied(monkeypatch):
    monkeypatch.setenv("CONDUCTOR_ONTOLOGY", "on")
    sent = {}

    def post(endpoint, body):
        sent.update(body)
        return {"version": "v", "elements": [{"name": "Line 3-4 | x", "type": "line", "mrid": "m", "role": "studied"}],
                "context": {"m": {"name": "Line 3-4 | x", "kind": "line", "ends": ["Bus 3", "Bus 4"], "kv": 20.0}},
                "named": 1, "where": [], "ties": {}}
    monkeypatch.setattr(gc, "_backend_post", post)
    links, view = gc.link("simulate_contingency", {"timestamp": "t", "system_secure": True, "violations": []},
                          {"element_type": "line", "element_index": 2})
    assert sent["refs"] == [{"type": "line", "index": 2}] and sent["names"] == []
    assert links["elements"][0]["role"] == "studied" and view["studied"] == ["Line 3-4"]
    # Nothing named anywhere: nothing linked, no call.
    sent.clear()
    assert gc.link("run_rsa", {"secure": True, "violations": []}, {}) == (None, None) and not sent


def test_how_values_are_obtained_and_the_normally_open_points(client):
    made = client.post("/api/network/cim/links", json={"names": [["Bus 14", "bus"], ["PV 3", "unit"]]}).json()
    bus = next(f for f in made["context"].values() if f["kind"] == "bus")
    pv = next(f for f in made["context"].values() if f["kind"] == "unit")
    assert all(d["p_kind"] in ("measured", "allocated") and d["q_kind"] == "derived" for d in bus["loads"])
    assert pv["p_kind"] in ("measured", "allocated") and "P " in gc.describe(pv)
    assert "Q derived" in gc.describe(bus)
    # CIGRE's feeders: S1 ties Line 12-13 to Line 1-2 (normally open, on Line 14-8); S2 and S3 close loops in Line 1-2.
    s1 = made["ties"]["Line 12-13"][0]
    assert (s1["switch"], s1["on"], s1["to_feeder"]) == ("S1", "Line 14-8", "Line 1-2")
    lines = gc.tie_lines(made["ties"])
    assert lines["Line 12-13"].startswith("S1 on Line 14-8 at Bus 8: a tie to feeder Line 1-2")
    assert "closes a loop within the feeder" in lines["Line 1-2"]
    view = gc.view(made["context"], made["where"], made["version"], made["named"], ties=made["ties"])
    assert view["normally_open_points"] == lines


def test_the_turn_record_keeps_ties_and_the_element_studied(monkeypatch):
    resolved = {"version": "v", "changed": False, "context": {"m1": {"name": "Line 12-13 | x", "kind": "line"},
                                                              "m2": BUS14_CTX},
                "where": ["feeder: all 2 on Line 12-13"], "ties": {"Line 12-13": [{"switch": "S1"}]}}
    monkeypatch.setattr(gc, "_backend_get", lambda endpoint: resolved)
    links = {"version": "v", "elements": [{"name": "Line 12-13 | x", "mrid": "m1", "role": "studied"},
                                          {"name": "Bus 14", "mrid": "m2"}]}
    out = gc.turn_context([{"name": "simulate_contingency", "result": {"cim_links": links}}])
    assert out["studied"] == ["Line 12-13 | x"] and out["ties"] == {"Line 12-13": [{"switch": "S1"}]}


BUS14_CTX = {"name": "Bus 14", "kind": "bus", "kv": {"value": 20.0, "basis": "stated"},
             "substation": {"value": "Bus 14", "basis": "inferred"},
             "feeder": {"value": "Line 12-13", "basis": "inferred", "position": 2, "of": 2},
             "units": [], "unit_count": 0, "load_count": 0, "loads": []}


def _report_functions(resolved):
    """app.py's own report builder (the chart helpers, `_report_charts_html`, `_where_html`), with the layer's
    reading replaced by `resolved`."""
    import ast
    import html
    import json
    import pathlib

    import sys

    import pandas as pd
    sys.path.insert(0, str(pathlib.Path("llm_agent").resolve()))
    from agent import renderers, tool_registry
    from agent import report as report_module
    src = pathlib.Path("llm_agent/app.py").read_text(encoding="utf-8")
    ns = {"RENDERER_MAP": renderers.RENDERER_MAP, "logging": __import__("logging"), "_tools_reg": tool_registry,
          "re": __import__("re"), "pd": pd, "json": json, "html": html, "_grid_context": gc, "_report": report_module,
          "st": __import__("streamlit")}
    start, end = src.index("_CHARTS_PER_ROW = "), src.index("# Chart injection helper")
    exec(src[start:end].rsplit("# ----", 1)[0], ns)
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in ("_report_charts_html", "_where_html"):
            exec(ast.get_source_segment(src, node), ns)
    ns["_resolve_links"] = lambda links_json: resolved
    return ns


def test_the_report_carries_the_where_table_with_and_without_charts():
    pytest.importorskip("streamlit")
    resolved = {"version": "v", "changed": True, "context": {"m1": BUS14_CTX},
                "where": [], "ties": {"Line 12-13": [{"switch": "S1", "on": "Line 14-8", "at": "Bus 8",
                                                      "to_feeder": "Line 1-2", "open_now": True}]},
                "moment": {"timestamp": "2000-01-01 00:00:00", "data_source": "measurements"}}
    ns = _report_functions(resolved)
    links = {"version": "v", "elements": [{"name": "Bus 14", "mrid": "m1", "role": "studied"}]}
    with_status = {"violations": [], "system_secure": False, "message": "The outage cuts off 0.2588 MW.",
                   "timestamp": "2000-01-01 00:00:00", "cim_links": links}
    page = ns["_report_charts_html"]([("simulate_contingency", with_status)])
    assert "Where these elements are" in page and "bus (studied)" in page and "feeder Line 12-13" in page
    assert "S1 on Line 14-8 at Bus 8: a tie to feeder Line 1-2" in page and "has changed since" in page
    assert "the moment the study ran" in page
    # A study with no chart at all (no renderer) still gets its table.
    page = ns["_report_charts_html"]([("compute_hosting_capacity_unknown", {"cim_links": links})])
    assert "Where these elements are" in page
    assert ns["_report_charts_html"]([("run_rsa", {"timestamp": "t"})]).count("Where these") == 0


def test_the_demand_at_a_moment_is_read_once(client):
    import main_backend as mb
    now = client.get("/api/time/current").json()
    ts = now.get("current_timestamp") or now.get("timestamp")
    mb.app_data.pop("_demand_cache", None)
    calls = []
    real = mb.lg.assign_load_values_from_measurements

    def counted(*a, **k):
        calls.append(1)
        return real(*a, **k)
    patch = pytest.MonkeyPatch()
    patch.setattr(mb.lg, "assign_load_values_from_measurements", counted)
    try:
        first = mb._demand_at(None, {"timestamp": ts, "data_source": "measurements"})
        again = mb._demand_at(None, {"timestamp": ts, "data_source": "measurements"})
    finally:
        patch.undo()
    assert len(calls) == 1 and first == again and first[0]
    first[0].clear()                                         # a caller's copy: the cache is not touched
    assert mb._demand_at(None, {"timestamp": ts, "data_source": "measurements"})[0]
