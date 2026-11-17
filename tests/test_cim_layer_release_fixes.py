"""
Fixes to the CIM layer found by the release review: the element picker sends
mRIDs (a bus named "1" is not the bus at index 1), a CGMES model without
equipment is refused and a bus without a voltage is unknown rather than NaN,
the converter's unnamed switch nodes get names of their own, the supply route
ignores external grids in other islands, the export energises a feeder from
its head line's substation, the export cache and Start fresh follow
declarations, storage rows have no Edit, and two texts read as people say them.
"""

import json
import pathlib

import pandapower as pp
import pandapower.networks as pn
import pytest

import cim_export as ce
import main_backend as mb
import network_loader as nl
import semantics as sm
from llm_agent.agent import renderers
from llm_agent.agent import semantics_view as sv
from llm_agent.agent import system_tables as st_tables

FIXTURE = pathlib.Path(__file__).parent / "fixtures" / "cgmes" / "pandapower_example_multivoltage.zip"


# --- The picker ---------------------------------------------------------------------------------------------------

def test_the_picker_sends_mrids_so_a_bus_named_1_is_not_the_bus_at_index_1():
    net = pn.case30()
    net.bus["name"] = [str(i + 1) for i in range(len(net.bus))]      # "1" is at index 0, "2" at index 1, …
    layer = sm.build(net, network_id="numeric")
    sent = dict(sv.around_choices(layer))
    for label, expected_index in (("Bus · 1", 0), ("Bus · 2", 1), ("Bus · 30", 29)):
        assert mb._by_mrid(layer, sent[label]) == ("bus", expected_index)


def test_the_picker_labels_branches_by_their_short_name_and_keeps_labels_unique():
    layer = sm.build(pn.create_cigre_network_mv(with_der="all"), network_id="cigre")
    labels = [label for label, _ in sv.around_choices(layer)]
    assert "Line · Line 2-3" in labels and not any(" -> " in label for label in labels)
    assert len(labels) == len(set(labels))


def test_the_supply_route_shows_short_names():
    net = pn.create_cigre_network_mv(with_der="all")
    layer = sm.build(net, network_id="cigre")
    line = int(net.line.index[net.line.name == "Line 2-3"][0])
    cards = dict(sv.around_cards(sm.around(net, layer, "line", line)))
    route = cards["How it is supplied"][0]
    assert "Trafo 0-1" in route and "Line 2-3" in route and "[" not in route


# --- CGMES robustness ---------------------------------------------------------------------------------------------

def test_a_cgmes_model_without_equipment_is_refused(monkeypatch):
    bare = pp.create_empty_network()
    for _ in range(3):
        pp.create_bus(bare, vn_kv=1.0)
    bare.bus["vn_kv"] = float("nan")                                  # what a model without its EQ profile gives
    monkeypatch.setattr("pandapower.converter.cim.cim2pp.from_cim.from_cim", lambda **kw: bare)
    pytest.importorskip("lxml")
    with pytest.raises(nl.CgmesError, match="EQ profile"):
        nl.read_cgmes(str(FIXTURE))


def test_a_cgmes_model_without_its_operating_point_is_refused(tmp_path):
    """Without the SSH profile the loads draw nothing and the units produce nothing: the synthetic data
    and every study would be empty."""
    pytest.importorskip("lxml")
    import zipfile
    no_ssh = tmp_path / "no_ssh.zip"
    with zipfile.ZipFile(FIXTURE) as src, zipfile.ZipFile(no_ssh, "w") as out:
        for name in src.namelist():
            if "_SSH" not in name.upper():
                out.writestr(name, src.read(name))
    with pytest.raises(nl.CgmesError, match="include the SSH profile"):
        nl.read_cgmes(str(no_ssh))
    net = nl.read_cgmes(str(FIXTURE))                                 # the whole model still loads
    assert net.load["p_mw"].notna().all() and net.load["p_mw"].sum() > 0


def test_a_bus_without_a_voltage_is_unknown_and_the_layer_is_valid_json():
    net = pn.case9()
    net.bus["vn_kv"] = net.bus["vn_kv"].astype(float)
    net.bus.at[net.bus.index[0], "vn_kv"] = float("nan")
    layer = sm.build(net, network_id="nan")
    assert layer["buses"][0]["kv"]["basis"] == sm.UNKNOWN
    json.dumps(layer, allow_nan=False)                               # no NaN reaches the API


@pytest.fixture(scope="module")
def cgmes_net():
    pytest.importorskip("lxml")
    return nl.read_cgmes(str(FIXTURE))


def test_the_converter_s_switch_nodes_get_names_of_their_own(cgmes_net):
    names = cgmes_net.bus["name"].astype(str)
    assert not names.str.contains("nan").any()
    assert names.is_unique
    layer = sm.build(cgmes_net, network_id="x")
    stations = {b["substation"]["value"] for b in layer["buses"]}
    assert not any("nan" in s for s in stations)
    # No substation spans every voltage level, as the one named after "be-sw nan" did.
    for station in stations:
        levels = {b["kv"]["value"] for b in layer["buses"] if b["substation"]["value"] == station}
        assert len(levels) <= 2, (station, levels)


def test_the_export_places_each_node_at_its_own_voltage_level(cgmes_net):
    rdflib = pytest.importorskip("rdflib")
    layer = sm.build(cgmes_net, network_id="x")
    g = rdflib.Graph().parse(data=ce.to_rdf_xml(layer, "x"), format="xml")
    cim = rdflib.Namespace("http://iec.ch/TC57/CIM100#")
    for node in g.subjects(rdflib.RDF.type, cim.TopologicalNode):
        node_kv = g.value(g.value(node, cim["TopologicalNode.BaseVoltage"]), cim["BaseVoltage.nominalVoltage"])
        level = g.value(node, cim["TopologicalNode.ConnectivityNodeContainer"])
        level_kv = g.value(g.value(level, cim["VoltageLevel.BaseVoltage"]), cim["BaseVoltage.nominalVoltage"])
        assert node_kv == level_kv


# --- Supply and export --------------------------------------------------------------------------------------------

def test_the_supply_route_of_a_line_includes_the_line_with_two_external_grids_in_separate_islands():
    net = pn.mv_oberrhein()
    assert len(net.ext_grid) == 2
    layer = sm.build(net, network_id="oberrhein")
    names = {b["index"]: b["name"] for b in layer["branches"] if b["element_type"] == "line"}
    open_at_an_end = set(net.switch.element[(net.switch.et == "l") & ~net.switch.closed.astype(bool)].astype(int))
    missing = []
    # A line open at one end (a normally open tie) carries no supply: its route rightly goes around it.
    for idx in [i for i in net.line.index[net.line.in_service.astype(bool)] if int(i) not in open_at_an_end][::5]:
        supply = sm.around(net, layer, "line", int(idx))["supply"]
        route = supply.get("route_from_external_grid") or []
        if route and names[int(idx)] not in route:
            missing.append(names[int(idx)])
    assert missing == []


def test_a_feeder_is_energised_from_its_head_line_s_substation():
    rdflib = pytest.importorskip("rdflib")
    layer = sm.build(pn.create_cigre_network_mv(with_der="all"), network_id="cigre")
    g = rdflib.Graph().parse(data=ce.to_rdf_xml(layer, "cigre"), format="xml")
    cim = rdflib.Namespace("http://iec.ch/TC57/CIM100#")
    feeders = list(g.subjects(rdflib.RDF.type, cim.Feeder))
    assert len(feeders) == 2
    for f in feeders:
        sub = g.value(f, cim["Feeder.NormalEnergizingSubstation"])
        assert str(g.value(sub, cim["IdentifiedObject.name"])) == "Substation Bus 0"


# --- Declarations: the export cache, Start fresh, storage ---------------------------------------------------------

def test_the_assumptions_key_changes_with_a_declaration(monkeypatch):
    monkeypatch.setitem(mb.app_data, "unit_sources", {})
    monkeypatch.setitem(mb.app_data, "load_categories", {})
    before = mb._assumptions_key()
    monkeypatch.setitem(mb.app_data, "unit_sources", {"sgen:0": "wind"})
    assert mb._assumptions_key() != before
    monkeypatch.setitem(mb.app_data, "load_categories", {"load:0": "industrial"})
    assert mb._assumptions_key() != before


def test_declarations_count_among_what_is_kept(monkeypatch, tmp_path):
    store = tmp_path / "assumptions.json"
    store.write_text(json.dumps({"by_fingerprint": {"fp": {"unit_sources": {"sgen:0": "wind", "sgen:1": "solar"},
                                                           "load_categories": {"load:3": "industrial"}}}}))
    monkeypatch.setattr(mb, "_ASSUMPTIONS_FILE", str(store))
    kept = mb._kept_changes({"fingerprint": "fp"})
    assert kept == {"declarations": 3}
    assert st_tables.kept_line(kept) == "Your changes are kept: 3 declared facts"


def test_storage_rows_have_no_edit():
    layer = sm.build(pn.create_cigre_network_mv(with_der="all"), network_id="cigre")
    rows = {r["unit"]: r for r in sv.unit_rows(layer)}
    assert rows["Battery 1"]["editable"] is False and rows["PV 3"]["editable"] is True


# --- Texts --------------------------------------------------------------------------------------------------------

def test_an_outage_that_cuts_off_load_says_so_instead_of_no_data():
    fig = renderers.render_contingency_violations(
        {"system_secure": False, "violations": [], "total_violations": 0,
         "message": "No limit breaks, but the outage cuts off 1.95 MW of load: not secure."})
    figs = fig if isinstance(fig, list) else [fig]
    title = figs[0].layout.title.text
    assert "cuts off 1.95 MW" in title and "No violation data" not in title


def test_the_no_feeders_note_is_not_doubled():
    layer = sm.build(pn.case30(), network_id="meshed")
    note = sm.view(layer, "feeders").get("note") or ""
    assert note.startswith("No feeders: ") and "no feeder:" not in note


def test_an_external_grid_with_no_branch_of_its_own_is_said_plainly():
    tiles = dict((label, how) for label, _value, how in sv.headline(
        {"external_grids": [{"bus": "Double Busbar 1", "substation": "Substation Double Busbar 1", "branches": []}]}))
    assert tiles["External grid"] == "no line or transformer leaves its substation"
    tiles = dict((label, how) for label, _value, how in sv.headline(
        {"external_grids": [{"bus": "Bus 0", "substation": "Substation Bus 0", "branches": ["Line 1-2 | x [L0]"]}]}))
    assert tiles["External grid"] == "its power comes in over 1 branch"


def test_the_export_names_branches_and_feeders_as_people_do():
    rdflib = pytest.importorskip("rdflib")
    layer = sm.build(pn.create_cigre_network_mv(with_der="all"), network_id="cigre")
    g = rdflib.Graph().parse(data=ce.to_rdf_xml(layer, "cigre"), format="xml")
    cim = rdflib.Namespace("http://iec.ch/TC57/CIM100#")
    cdr = rdflib.Namespace("urn:conductor:semantics#")
    names = {cls: {str(g.value(s, cim["IdentifiedObject.name"])) for s in g.subjects(rdflib.RDF.type, cim[cls])}
             for cls in ("ACLineSegment", "PowerTransformer", "Feeder")}
    assert "Line 2-3" in names["ACLineSegment"] and "Trafo 0-1" in names["PowerTransformer"]
    assert names["Feeder"] == {"Line 1-2", "Line 12-13"}
    s1 = next(g.subjects(cim["IdentifiedObject.name"], rdflib.Literal("S1")))
    assert str(g.value(s1, cdr.role)) == "tie between feeders Line 1-2 and Line 12-13"
    assert not any(" -> " in str(o) for o in g.objects(None, cim["IdentifiedObject.name"]))


def test_resume_names_the_upload_as_the_operator_named_it():
    import scenarios as scn
    d = scn.resume_detail({"network_path": "/x/systems/uploaded_system__pandapower_example_multivoltage.zip"},
                          {"csv_path": "/x/data_files/measurements_synthetic.csv"})
    assert d == {"network": "pandapower_example_multivoltage.zip", "data": "a synthetic week", "available": True}
    assert scn.resume_detail(None, {"csv_path": "/x/data_files/my_week.csv"})["data"] == "my_week.csv"


# --- Lower-priority fixes -----------------------------------------------------------------------------------------

def test_one_saved_declaration_that_no_longer_applies_drops_alone(monkeypatch):
    net = pn.create_cigre_network_mv(with_der="all")
    monkeypatch.setitem(mb.app_data, "net_loaded", net)
    kept = mb._restore_declarations({"PV 3": "wind", "A unit that is gone": "solar", "WKA 7": "solar"},
                                    "unit", "source", sm.normalise_declarations, "Unit source")
    assert kept == {"PV 3": "wind", "WKA 7": "solar"}


def test_two_unnamed_units_on_one_bus_are_told_apart():
    import generators as gm
    import resolve as rv
    net = pn.case9()
    net.gen["name"] = None
    bus = int(net.gen.bus.iloc[0])
    pp.create_gen(net, bus=bus, p_mw=5, vm_pu=1.0)
    names = [gm.unit_name(net, "gen", i) for i in net.gen.index]
    assert len(set(names)) == len(names) and names[-1] == f"Gen_bus{bus}_2"
    assert rv.find_unit(net, f"Gen_bus{bus}_2").idx == int(net.gen.index[-1])
    assert sm.normalise_declarations(net, [{"unit": f"Gen_bus{bus}_2", "source": "wind"}], {}) == \
        {f"Gen_bus{bus}_2": "wind"}


def test_switches_shunts_and_the_external_grid_are_found_by_their_mrid():
    net = pn.create_cigre_network_mv(with_der="all")
    layer = sm.build(net, network_id="cigre")
    s1 = next(x for x in layer["switches"] if x["name"] == "S1")
    assert mb._by_mrid(layer, s1["mrid"]) == ("switch", s1["index"])
    line = mb._around_target(layer, ("switch", s1["index"]))
    assert line[0] == "line" and layer["branches"][[b["index"] for b in layer["branches"]].index(line[1])]["name"] \
        .startswith("Line 14-8")
    ext = layer["external_grids"][0]
    assert mb._around_target(layer, mb._by_mrid(layer, ext["mrid"])) == ("bus", ext["bus_index"])
    shunt_layer = sm.build(pn.case30(), network_id="ieee30")
    sh = shunt_layer["shunts"][0]
    assert mb._around_target(shunt_layer, mb._by_mrid(shunt_layer, sh["mrid"])) == ("bus", sh["bus_index"])


def test_a_follow_up_keeps_the_power_factor_asked_for_a_question_ago():
    from llm_agent.agent.validators import power_factor_in_play
    earlier = [{"role": "user", "content": "Is it secure at 18:00 with loads at pf 0.95?"},
               {"role": "assistant", "content": "No: …"}]
    assert power_factor_in_play("And at 19:00?", earlier)
    assert not power_factor_in_play("And at 19:00?", [{"role": "user", "content": "Is it secure at 18:00?"}])
    assert power_factor_in_play("Rerun at power factor 0.95", [])


def test_a_non_uuid_mrid_gets_a_valid_iri():
    assert ce._urn("a8b77418-f217-5432-a79f-df3a31309e5f") == "urn:uuid:a8b77418-f217-5432-a79f-df3a31309e5f"
    iri = ce._urn("Bus 7 / west")
    assert iri.startswith("urn:conductor:mrid:") and " " not in iri and "/" not in iri.split("mrid:")[1]


def test_names_as_loaded_count_as_real_elements_after_a_rename(monkeypatch):
    from llm_agent.agent.entities import names_not_in_network
    net = pn.create_cigre_network_mv(with_der="all")
    monkeypatch.setitem(mb.app_data, "net_loaded", net)
    monkeypatch.setitem(mb.app_data, "element_names", {"load": {"0": "Hospital"}})
    loaded = mb._names_as_loaded()
    assert "Load R1" in loaded and "Bus 3" in loaded
    current = ["Hospital", *[f"Load R{k}" for k in range(3, 12)]]
    assert names_not_in_network("Load R1 is the hospital.", current) == ["Load R1"]   # without them: flagged
    assert names_not_in_network("Load R1 is the hospital.", current + loaded) == []
    monkeypatch.setitem(mb.app_data, "element_names", {})
    assert mb._names_as_loaded() == []                                                  # nothing renamed
