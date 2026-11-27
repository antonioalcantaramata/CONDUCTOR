"""
A proposal as a change to the CIM model (backend/cim_changes.py): the difference between the base's CGMES model and
the proposal's, applied to the base network by mRID, gives the network the edits build — every column of every table,
the same row indices, the same power flow — for every kind of edit. Every proposal is kept as its difference and
studied through it (no switch); its edits are the language it is written in.
"""

import pandapower.networks as pn
import pytest
from fastapi.testclient import TestClient

import cim_changes as cc
import main_backend as mb
import sandbox as sb
from evaluation.cim_central import edit_catalogue, load

pytest.importorskip("lxml")
NID = "case14"


def _net():
    n = pn.case14()
    n.bus["name"] = None
    return n


EDITS = {
    "add_load": [{"op": "add_load", "name": "DC", "bus": 9, "p_mw": 5}],
    "add_line": [{"op": "add_line", "from_bus": 0, "to_bus": 4, "length_km": 5, "line_like": 0}],
    "reinforce_line": [{"op": "reinforce_line", "line": 3, "factor": 2.0}],
    "switching": [{"op": "set_parameter", "element_type": "line", "element": 5, "values": {"in_service": False}}],
    "add_bus_with_transformer": [{"op": "add_bus_with_transformer", "name": "DC 20 kV", "connect_to": 9, "vn_kv": 20,
                                  "sn_mva": 40, "vk_percent": 12, "vkr_percent": 0.4, "parallel": 2},
                                 {"op": "add_load", "name": "DC", "bus": "DC 20 kV", "p_mw": 10}],
}


@pytest.fixture(scope="module")
def base():
    return _net()


@pytest.fixture(scope="module")
def base_statements(base):
    return cc.statements(base, NID)


# -- the model ----------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("kind", list(EDITS))
def test_the_model_with_the_difference_read_whole_is_the_edits_network(kind, base, base_statements):
    """The whole proposal's model, read by the adapter: the same power flow element by element."""
    edited, _ = sb.materialise(base, EDITS[kind])
    diff = cc.difference(base_statements, cc.statements(edited, NID))
    projected, emap, _notes = cc.project(base_statements, diff, NID, kind)
    r = cc.compare(edited, projected, NID, emap)
    assert r["matches"], (kind, r["max_diff"], r["worst"])
    assert r["n_missing"] == 0


def test_applying_the_difference_gives_the_proposal_s_model(base, base_statements):
    edited, _ = sb.materialise(base, EDITS["add_bus_with_transformer"])
    target = cc.statements(edited, NID)
    diff = cc.difference(base_statements, target)
    assert cc.apply(base_statements, diff)["objects"] == target["objects"]
    s = cc.summary(diff)
    assert s["added"]["PowerTransformer"] == 1 and s["added"]["EnergyConsumer"] == 1


def test_a_difference_is_written_as_an_iec_61970_552_difference_model(base, base_statements):
    import rdflib
    edited, canonical = sb.materialise(base, EDITS["add_line"] + EDITS["switching"])
    xml = cc.to_xml(cc.translate(base, base_statements, canonical, NID), description="P1")
    assert "DifferenceModel" in xml and "forwardDifferences" in xml and "ACLineSegment" in xml
    rdflib.Graph().parse(data=xml, format="xml")


# -- every kind of edit, on networks with and without switches -------------------------------------------------------

_NETS = {"case14": load("case14"), "cigre_mv": load("cigre_mv")}
_BASES: dict = {}
_CASES = [(n, k) for n, net in _NETS.items() for k in edit_catalogue(net)]


@pytest.mark.parametrize("network,kind", _CASES)
def test_the_projected_difference_is_the_edits_network(network, kind):
    """Every column of every table, the costs, the row indices and the power flow: the base network with the
    proposal's difference applied is the network its edits build."""
    net = _NETS[network]
    base = _BASES.setdefault(network, cc.statements(net, network))
    try:
        edited, canonical = sb.materialise(net, edit_catalogue(net)[kind])
    except sb.EditError as exc:
        pytest.skip(f"not an edit this network takes: {exc}")
    diff = cc.translate(net, base, canonical, network)
    onto, _notes = cc.project_onto(net, base, diff, network, kind)
    assert cc.equivalent(edited, onto, network) == []
    assert cc.compare(edited, onto, network)["matches"]


def test_an_edit_with_no_translation_of_its_own_is_written_from_the_edit(base, base_statements):
    far = int(base.bus.index[base.bus.vn_kv != base.bus.at[int(base.load.at[0, "bus"]), "vn_kv"]][0])
    edited, canonical = sb.materialise(base, [{"op": "move_element", "element_type": "load", "element": 0,
                                               "to_bus": far}])
    diff = cc.translate(base, base_statements, canonical, NID)
    assert diff["translation"] == {"direct": 0, "rewritten": 1, "elements_rewritten": {"load": 1}}
    assert cc.equivalent(edited, cc.project_onto(base, base_statements, diff, NID)[0], NID) == []


def test_a_reinforcement_is_translated_directly(base, base_statements):
    _edited, canonical = sb.materialise(base, EDITS["reinforce_line"])
    assert cc.translate(base, base_statements, canonical, NID)["translation"]["direct"] == 1


def test_a_partial_write_states_what_a_full_one_does():
    for net in _NETS.values():
        full = cc.statements(net, "x")
        for table in ("line", "trafo", "load", "sgen", "switch", "bus"):
            if len(net[table]):
                part = cc.statements(net, "x", only={table: list(net[table].index[:3])})
                assert all(full["objects"][k]["props"] == v["props"] for k, v in part["objects"].items())


def test_a_projection_reads_only_what_the_difference_touches(base, base_statements):
    _edited, canonical = sb.materialise(base, EDITS["reinforce_line"])
    diff = cc.translate(base, base_statements, canonical, NID)
    owners, added = cc._touched(base_statements, diff)
    sub = cc._sub_model(base_statements, diff, owners, added)
    assert len(owners) == 1 and len(sub["objects"]) < len(base_statements["objects"]) / 5


def test_equivalent_finds_a_difference(base):
    other = base.deepcopy() if hasattr(base, "deepcopy") else __import__("copy").deepcopy(base)
    other.line.at[3, "length_km"] = 99.0
    assert any("line 3.length_km" in d for d in cc.equivalent(base, other, NID))


# -- the backend ------------------------------------------------------------------------------------------------------

@pytest.fixture
def client(tmp_path, monkeypatch):
    systems, data = tmp_path / "systems", tmp_path / "data_files"
    systems.mkdir()
    data.mkdir()
    monkeypatch.setenv("GRID_PROFILE", "pglib_case30")
    monkeypatch.setattr(mb, "SYSTEMS_DIR", str(systems))
    monkeypatch.setattr(mb, "DATA_FILES_DIR", str(data))
    for name, folder in (("_LAST_UPLOAD_SENTINEL", systems), ("_LAST_TIMESERIES_SENTINEL", data),
                         ("_LAST_FORECAST_SENTINEL", data), ("_PROPOSALS_FILE", systems),
                         ("_ASSUMPTIONS_FILE", systems), ("_SCENARIO_FILE", tmp_path)):
        monkeypatch.setattr(mb, name, str(folder / f"{name}.json"))
    saved = dict(mb.app_data)
    with TestClient(mb.app) as c:
        yield c
    mb.app_data.clear()
    mb.app_data.update(saved)


def _propose(client, name="P1"):
    bus = client.get("/api/network/topology").json()["buses"][5]["name"]
    r = client.post("/api/proposals/edit", json={"proposal": name, "edits": [
        {"op": "add_load", "name": "DC", "bus": bus, "p_mw": 4}]})
    assert r.status_code == 200, r.text
    return r.json()


def test_every_proposal_is_kept_as_its_cim_difference_and_studied_through_it(client):
    view = _propose(client)
    assert view["cim"]["summary"]["added"].get("EnergyConsumer") == 1
    assert view["cim"]["translation"]["rewritten"] == 1          # an added load: its element written from the edit
    entry = mb._materialised("P1")
    assert entry["via"] == "cim"
    edited, _ = sb.materialise(mb.app_data["net"], mb._proposals()["P1"]["edits"])
    assert cc.equivalent(edited, entry["net"], str(mb.app_data["loaded_fingerprint"])) == []
    info = client.get("/api/network/cim/proposal", params={"network": "P1", "format": "json"}).json()
    assert info["summary"]["added"]["EnergyConsumer"] == 1 and "mode" not in info
    xml = client.get("/api/network/cim/proposal", params={"network": "P1"})
    assert xml.status_code == 200 and b"DifferenceModel" in xml.content
    rsa = client.post("/api/grid/rsa", json={"network": "P1"})
    assert rsa.status_code == 200 and rsa.json().get("converged") is not False


def test_a_reinforcement_proposal_is_translated_directly(client):
    line = client.get("/api/network/topology").json()["branches"][2]["name"]
    r = client.post("/api/proposals/edit", json={"proposal": "P2", "edits": [
        {"op": "reinforce_line", "line": line, "factor": 2}]})
    assert r.status_code == 200, r.text
    assert r.json()["cim"]["translation"]["direct"] == 1 and r.json()["cim"]["translation"]["rewritten"] == 0


def test_without_a_difference_a_proposal_is_studied_by_its_edits(client, monkeypatch):
    def broken(*a, **k):
        raise RuntimeError("no model")
    monkeypatch.setattr(cc, "translate", broken)
    view = _propose(client)
    assert view["cim"]["error"].startswith("RuntimeError")
    assert mb._materialised("P1")["via"] == "edits"
    r = client.get("/api/network/cim/proposal", params={"network": "P1"})
    assert r.status_code == 404 and "studied by its edits" in r.json()["detail"]


def test_a_projection_that_fails_falls_back_to_the_edits(client, monkeypatch):
    _propose(client)

    def broken(*a, **k):
        raise RuntimeError("cannot project")
    monkeypatch.setattr(cc, "project_onto", broken)
    mb.app_data["_proposal_cache"] = {}
    assert mb._materialised("P1")["via"] == "edits"


# -- the operator's standing changes ---------------------------------------------------------------------------------

def test_a_unit_keeps_its_mrid_when_its_generator_model_moves_it():
    import generators as gm
    import semantics as smx
    net = pn.case30()
    net.sgen.drop(net.sgen.index, inplace=True)
    units = {u["unit"]: u for u in gm.describe(net)}
    before = {u: smx._row_mrid(net, *gm._row_of(net, u), NID) for u in units}
    target = next(u for u in units if gm._row_of(net, u)[0] == "gen")
    moved = gm.apply(net, {target: {"control": gm.FIXED_Q, "q_mvar": 0.0}})
    assert gm._row_of(moved, target)[0] == "sgen"
    assert {u: smx._row_mrid(moved, *gm._row_of(moved, u), NID) for u in units} == before


def test_the_operator_s_changes_are_a_cim_difference_and_go_in_the_package(client):
    import io
    import zipfile
    empty = client.get("/api/network/cim/operator", params={"format": "json"}).json()
    assert empty["empty"] is True
    unit = client.get("/api/generators").json()["units"][1]["unit"]
    r = client.post("/api/generators", json={"changes": [{"unit": unit, "control": "power_factor"}]})
    assert r.status_code == 200, r.text
    bus = client.get("/api/network/topology").json()["buses"][2]["name"]
    r = client.post("/api/names", json={"changes": [{"element_type": "bus", "element": bus, "name": "North"}]})
    assert r.status_code == 200, r.text
    info = client.get("/api/network/cim/operator", params={"format": "json"}).json()
    assert info["empty"] is False and info["summary"]["changed"]
    xml = client.get("/api/network/cim/operator")
    assert xml.status_code == 200 and b"DifferenceModel" in xml.content and b"North" in xml.content
    package = client.get("/api/network/cim/package")
    assert package.status_code == 200
    with zipfile.ZipFile(io.BytesIO(package.content)) as z:
        assert "operator_changes.xml" in z.namelist()


def test_the_base_network_is_the_model_with_the_operator_s_difference(client):
    import generators as gm
    import naming
    unit = client.get("/api/generators").json()["units"][1]["unit"]
    assert client.post("/api/generators", json={"changes": [{"unit": unit, "control": "power_factor"}]}).status_code == 200
    bus = client.get("/api/network/topology").json()["buses"][2]["name"]
    assert client.post("/api/names", json={"changes": [{"element_type": "bus", "element": bus,
                                                        "name": "North"}]}).status_code == 200
    direct = naming.apply(gm.apply(mb.app_data["net_loaded"], mb.app_data["generator_model"]),
                          mb.app_data["element_names"])
    nid = str(mb.app_data["loaded_fingerprint"])
    assert cc.equivalent(direct, mb.app_data["net"], nid) == []
    assert mb._operator_difference()["changed"]


def test_a_saved_operator_difference_is_projected_without_reapplying_the_changes(client, monkeypatch):
    import generators as gm
    unit = client.get("/api/generators").json()["units"][1]["unit"]
    assert client.post("/api/generators", json={"changes": [{"unit": unit, "control": "power_factor"}]}).status_code == 200
    model, names = mb.app_data["generator_model"], mb.app_data["element_names"]
    expected = mb.app_data["net"]
    mb.app_data["_saved_operator"] = {"basis": mb._cim_basis(model, names), "difference": mb._operator_difference()}

    def refused(*a, **k):
        raise AssertionError("the changes were applied again")
    monkeypatch.setattr(gm, "apply", refused)
    again = mb._operated_network(mb.app_data["net_loaded"], model, names)
    assert cc.equivalent(expected, again, str(mb.app_data["loaded_fingerprint"])) == []


def test_a_saved_proposal_keeps_its_difference_across_a_restart(client, monkeypatch):
    _propose(client)
    saved = dict(mb._proposals()["P1"])
    mb._proposals().clear()

    def refused(*a, **k):
        raise AssertionError("translated again")
    monkeypatch.setattr(cc, "translate", refused)
    assert mb._restore_proposals() == ["P1"]
    assert mb._proposals()["P1"]["cim_difference"] == saved["cim_difference"]


def test_a_difference_file_reads_back_as_the_same_difference(base, base_statements):
    for kind in EDITS:
        edited, canonical = sb.materialise(base, EDITS[kind])
        diff = cc.translate(base, base_statements, canonical, NID)
        back = cc.from_xml(cc.to_xml(diff, description=kind))
        assert cc.equivalent(edited, cc.project_onto(base, base_statements, back, NID)[0], NID) == [], kind


def test_a_moment_is_shared_by_networks_with_the_same_measured_elements():
    import operating_point as op
    from evaluation.cim_central import measurements
    net = _NETS["case14"]
    base = _BASES.setdefault("case14", cc.statements(net, "case14"))
    edited, canonical = sb.materialise(net, edit_catalogue(net)["add a load"])
    proposal, _ = cc.project_onto(net, base, cc.translate(net, base, canonical, "case14"), "case14")
    assert op.signature(proposal) == op.signature(net)
    meas = measurements(net, 1)
    subs = list(meas["substation_name"])
    own = op.allocate(__import__("copy").deepcopy(proposal), meas, subs, 0.95)[0]
    shared = __import__("copy").deepcopy(proposal)
    op.apply(shared, op.compute(net, meas, subs, 0.95, "case14"), meas, subs, "case14")
    assert cc.equivalent(own, shared, "case14") == []


def test_a_moment_is_exported_as_an_ssh_profile(client):
    import rdflib
    r = client.get("/api/network/cim/moment", params={"format": "json"})
    assert r.status_code == 200, r.text
    elements = r.json()["elements"]
    assert any(e["table"] == "load" and "p_mw" in e for e in elements.values())
    xml = client.get("/api/network/cim/moment")
    assert xml.status_code == 200 and b"EnergyConsumer.p" in xml.content
    rdflib.Graph().parse(data=xml.content, format="xml")


def test_the_package_carries_the_proposals_and_a_manifest(client):
    import io
    import json
    import zipfile
    _propose(client)
    package = client.get("/api/network/cim/package")
    with zipfile.ZipFile(io.BytesIO(package.content)) as z:
        names = z.namelist()
        manifest = json.loads(z.read("manifest.json"))
    assert "proposals/P1.xml" in names and manifest["proposals"][0]["id"] == "P1"
    assert manifest["basis"] == mb._cim_basis()


def test_the_changes_view_reads_the_proposal_s_cim_difference(client):
    line = client.get("/api/network/topology").json()["branches"][2]["name"]
    assert client.post("/api/proposals/edit", json={"proposal": "P2", "edits": [
        {"op": "reinforce_line", "line": line, "factor": 2}]}).status_code == 200
    out = client.get("/api/network/semantics", params={"network": "P2", "about": "changes"})
    assert out.status_code == 200, out.text
    model = out.json()["model"]
    assert model[0]["class"] == "ACLineSegment" and model[0]["change"] == "changed"
    assert any(s.get("property") == "cim:ACLineSegment.r" for s in model[0]["statements"])


def test_on_a_network_loaded_as_cim_a_difference_names_the_model_s_own_objects():
    """Terminals, ends, tap changers, controls, units, limits: the model's own ids, so the difference applies to its
    files; and the projected network is still the edits' network."""
    net = load("multivoltage_cgmes")
    model = cc._MODELS["multivoltage_cgmes"]
    base = cc.statements(net, "multivoltage_cgmes")
    assert base["parts"]
    for kind in ("reinforce a line", "a line out", "upgrade a transformer", "move a load"):
        edited, canonical = sb.materialise(net, edit_catalogue(net)[kind])
        diff = cc.translate(net, base, canonical, "multivoltage_cgmes")
        changed = [x["mrid"] for x in diff["changed"] + diff["removed"] if x["profile"] != cc.EXTRAS]
        assert changed and all(model.get(m) is not None for m in changed), kind
        onto, _ = cc.project_onto(net, base, diff, "multivoltage_cgmes")
        assert cc.equivalent(edited, onto, "multivoltage_cgmes") == [], kind


def test_a_transformer_can_be_added_to_a_network_loaded_as_cim():
    net = load("multivoltage_cgmes")
    edited, _ = sb.materialise(net, edit_catalogue(net)["add a transformer"])
    assert len(edited.trafo) == len(net.trafo) + 1
