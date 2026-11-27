"""
The CIM model kept whole (backend/cim_model.py) and what the CIM layer reads from it (semantics.build with
`cim`): names, voltage levels, bays, busbars, limits by duration, controls, transformer ends and taps, load
responses, the equipment pandapower's converter has no place for, the loss report of an import, and the check of
CONDUCTOR's power flow against the model's own solution. Also: a switch class the converter does not read (a
fuse) is added at import, so the topology stays the model's; two units sharing a name keep their own nominal values.

Only pandapower's example model is used (ENTSO-E's test models are not vendored); the cases that need other
content are built from it in a temporary folder.
"""

import copy
import io
import pathlib
import re
import zipfile

import pandapower as pp
import pandapower.topology as top
import pytest
from fastapi.testclient import TestClient

import cim_model as cm
import main_backend as mb
import network_loader as nl
import nominal as nom
import semantics as sm

pytest.importorskip("lxml")
FIXTURE = pathlib.Path(__file__).parent / "fixtures" / "cgmes" / "pandapower_example_multivoltage.zip"
STATUSES = {"described", "partly", "read_into", "solved_state", "model_only"}


@pytest.fixture(scope="module")
def model():
    return cm.CimModel.read(str(FIXTURE))


@pytest.fixture(scope="module")
def net():
    return nl.read_cgmes(str(FIXTURE))


@pytest.fixture(scope="module")
def layer(net, model):
    return sm.build(net, network_id="example", cim=model)


# -- the model as read -------------------------------------------------------------------------------------------

def test_every_profile_and_object_is_kept(model):
    s = model.summary()
    assert s["version"] == "2.4.15"
    assert set(s["profiles"]) >= {"EQ", "SSH", "TP", "SV", "EQ_BD", "TP_BD"}
    # Classes the converter has no table for are kept too.
    assert model.of("OperationalLimitSet") and model.of("SvVoltage") and model.of("ControlArea")
    assert all(model.cls(i) == c for c, ids in model.by_class.items() for i in ids[:3])


def test_lookups_follow_the_model_s_references(model):
    seg = model.of("ACLineSegment")[0]
    assert len(model.terminals(seg)) == 2
    limits = model.limits(seg)
    assert limits and all(x["value"] > 0 for x in limits)
    node = model.of("ConnectivityNode")[0]
    chain = model.container_chain(node)
    assert model.cls(chain["voltage_level"]) == "VoltageLevel" and model.cls(chain["substation"]) == "Substation"
    d = model.describe(seg)
    assert d["class"] == "ACLineSegment" and d["attributes"]
    # References carry what they refer to.
    ref = next(iter(d["references"].values()))
    assert {"mrid", "class", "name"} <= set(ref if isinstance(ref, dict) else ref[0])


def test_a_name_that_is_only_an_identifier_is_no_label(model):
    sub = model.of("Substation")[0]
    assert re.match(r"_?[0-9a-f]{8}-", model.name(sub))       # this example names substations by id
    assert model.label(sub) is None


def test_ids_are_compared_without_their_prefixes():
    assert cm.norm("#_abc-1") == "abc-1" and cm.norm("urn:uuid:abc-1") == "abc-1"


def test_a_zip_without_cim_objects_is_refused(tmp_path):
    p = tmp_path / "x.zip"
    with zipfile.ZipFile(p, "w") as z:
        z.writestr("a.txt", "nothing")
    with pytest.raises(cm.CimModelError):
        cm.CimModel.read(str(p))


# -- what the layer reads from it --------------------------------------------------------------------------------

def test_buses_carry_their_node_class_and_the_model_s_voltage_level(layer, model):
    buses = layer["buses"]
    assert {b["cim_class"] for b in buses} == {"ConnectivityNode"}
    with_level = [b for b in buses if b.get("voltage_level")]
    assert with_level and all(b["voltage_level"]["basis"] == "stated" for b in with_level)
    vl = with_level[0]["voltage_level"]["value"]
    assert model.cls(vl["mrid"]) == "VoltageLevel"


def test_substations_and_voltage_levels_keep_the_model_s_identity(layer, model):
    subs = layer["containers"]["substations"]
    assert all(model.cls(s["mrid"]) == "Substation" for s in subs)
    levels = [vl for s in subs for vl in s["voltage_levels"]]
    assert all(vl["basis"] == "stated" and model.cls(vl["mrid"]) == "VoltageLevel" for vl in levels)
    assert any(vl["name"] == "VL for Double Busbar 1" for vl in levels)
    # Names are unique even where the model names its substations by id (the label falls back to a bus).
    assert len({s["name"] for s in subs}) == len(subs)
    assert not any(re.match(r"_?[0-9a-f]{8}-", s["name"]) for s in subs)


def test_branches_carry_limits_ends_and_taps(layer):
    line = next(b for b in layer["branches"] if b.get("limits"))
    assert line["limits"]["basis"] == "stated" and "permanent" in line["limits"]["in_studies"]
    trafo = next(b for b in layer["branches"] if b["element_type"] == "trafo" and b.get("tap_changers"))
    assert len(trafo["transformer_ends"]["value"]) == 2 and trafo["transformer_ends"]["value"][0]["rated_kv"]
    assert trafo["ends"] == [str(e) for e in trafo["ends"]]      # the buses it joins, still
    assert "fixed" in trafo["tap_changers"]["in_studies"]


def test_controls_and_units_are_the_model_s(layer):
    unit = next(u for u in layer["units"] if u["cim_control"].get("in_model"))
    assert unit["cim_control"]["in_model"]["mode"] == "voltage" and unit["cim_control"]["in_model"]["target"]
    ext = layer["external_grids"][0]["in_model"]
    assert ext["class"] == "ExternalNetworkInjection" and ext["control"]["target"]


def test_other_equipment_is_listed_with_what_it_is(layer):
    other = layer["other_equipment"]
    by_class = {o["cim_class"] for o in other}
    assert {"EquivalentInjection", "EquivalentBranch", "PowerTransformer"} <= by_class
    three = next(o for o in other if o["element_type"] == "trafo3w")
    assert len(three["transformer_ends"]["value"]) == 3 and three["in_computation"]


def test_the_loss_report_covers_every_class(layer, model):
    rows = layer["cim_model"]["coverage"]["classes"]
    assert {r["class"] for r in rows} == {c for c, ids in model.by_class.items() if ids}
    assert {r["status"] for r in rows} <= STATUSES
    status = {r["class"]: r["status"] for r in rows}
    for cls in ("ACLineSegment", "ConnectivityNode", "VoltageLevel", "Substation", "CurrentLimit",
                "RegulatingControl", "RatioTapChanger"):
        assert status[cls] == "described", cls
    assert status["Terminal"] == "read_into" and status["SvVoltage"] == "solved_state"


def test_without_the_model_the_layer_is_built_from_pandapower_alone(net):
    plain = sm.build(net, network_id="example")
    assert "cim_model" not in plain
    assert all(vl["basis"] == "inferred" for s in plain["containers"]["substations"] for vl in s["voltage_levels"])
    # The model's substations are still stated through pandapower's ids.
    assert plain["substations"]["basis"] == "stated"


def test_the_model_view_says_what_the_studies_ignore(layer):
    view = sm.view(layer, "model")
    assert view["about"] == "model" and set(view["classes_by_status"]) <= STATUSES
    assert sm.view({"units": [], "buses": [], "units_by_source": {}}, "model")["note"].startswith("The network did not")


# -- what the studies ignore -------------------------------------------------------------------------------------

class _Model(cm.CimModel):
    """A hand-built model: objects given directly."""

    def __init__(self, objects):
        super().__init__()
        for k, o in objects.items():
            self.objects[k] = o
            self.by_class[o["class"]].append(k)


def test_a_voltage_dependent_load_is_said_to_be_studied_as_constant_power():
    m = _Model({"L1": {"class": "EnergyConsumer", "EnergyConsumer.LoadResponse": cm.Ref("R1")},
                "R1": {"class": "LoadResponseCharacteristic", "LoadResponseCharacteristic.exponentModel": "false",
                       "LoadResponseCharacteristic.pConstantPower": "0.6",
                       "LoadResponseCharacteristic.pConstantImpedance": "0.4",
                       "LoadResponseCharacteristic.qConstantPower": "1"}})
    resp = sm._model_load_response(m, "L1")
    assert resp["constant_power"] is False and resp["p_impedance"] == 0.4
    layer = {"cim_model": {"version": "x"}, "loads": [{"name": "L1", "load_response": resp}]}
    kinds = {c["kind"] for c in sm.model_caveats(layer)}
    assert kinds == {"load_response_not_used"}


def test_temporary_limits_and_followed_controls_are_caveats():
    layer = {"cim_model": {"version": "x"},
             "branches": [{"name": "L 1-2 | a -> b", "limits": {"value": [
                 {"permanent": True, "duration_s": None}, {"permanent": False, "duration_s": 900.0}]}}],
             "shunts": [{"name": "C1", "control": {"enabled": True}}]}
    kinds = {c["kind"] for c in sm.model_caveats(layer)}
    assert kinds == {"temporary_limits_after_outages_only", "shunt_control_not_followed"}


# -- a switch class the converter does not read -------------------------------------------------------------------

def _with_fuse(tmp_path, model, net) -> tuple[pathlib.Path, str]:
    """The example with a closed breaker whose opening cuts buses off turned into a Fuse."""
    for b in model.of("Breaker"):
        if str(model.get(b).get("Switch.open")).lower() == "true":
            continue
        rows = net.switch.index[net.switch.origin_id.astype(str).str.lstrip("_") == b]
        n = copy.deepcopy(net)
        n.switch.at[rows[0], "closed"] = False
        if len(top.unsupplied_buses(n)):
            break
    out = tmp_path / "with_fuse.zip"
    pat = re.compile(r'<cim:Breaker(\s+rdf:(?:ID|about)="#?_?' + re.escape(b) + r'")(.*?)</cim:Breaker>', re.S)
    with zipfile.ZipFile(FIXTURE) as zin, zipfile.ZipFile(out, "w") as zout:
        for item in zin.namelist():
            data = zin.read(item)
            if item.lower().endswith(".xml"):
                data = pat.sub(lambda mm: "<cim:Fuse" + mm.group(1) + mm.group(2) + "</cim:Fuse>",
                               data.decode("utf-8")).encode("utf-8")
            zout.writestr(item, data)
    return out, b


def test_a_fuse_is_added_so_the_topology_is_the_model_s(tmp_path, model, net):
    path, fuse = _with_fuse(tmp_path, model, net)
    loaded = nl.read_cgmes(str(path))
    assert len(loaded.switch) == len(net.switch)
    assert not len(top.unsupplied_buses(loaded))
    assert any("Fuse" in n for n in loaded["_import_notes"])
    layer = sm.build(loaded, cim=cm.CimModel.read(str(path)))
    sw = next(s for s in layer["switches"] if s["mrid"] == fuse)
    assert sw["cim_class"]["value"] == "Fuse" and sw["cim_class"]["basis"] == "stated"


def test_without_the_fix_the_lost_fuse_is_flagged_everywhere(tmp_path, model, net, monkeypatch):
    path, fuse = _with_fuse(tmp_path, model, net)
    monkeypatch.setattr(nl, "add_unconverted_switches", lambda net, path: [])
    loaded = nl.read_cgmes(str(path))
    assert len(top.unsupplied_buses(loaded)) > 50          # everything beyond the fuse looks cut off
    m = cm.CimModel.read(str(path))
    layer = sm.build(loaded, cim=m)
    lost = [o for o in layer["other_equipment"] if not o["in_computation"]]
    assert [o["mrid"] for o in lost] == [fuse] and len(lost[0]["buses"]) == 2
    assert "equipment_not_in_computation" in {c["kind"] for c in sm.model_caveats(layer)}
    solved = copy.deepcopy(loaded)
    pp.runpp(solved)
    check = sm.solved_state_check(solved, m)
    assert check["matches"] is False and check["buses_not_solved"] > 50


# -- CONDUCTOR's power flow against the model's own solution -------------------------------------------------------

def test_the_solved_state_check_finds_what_differs(model, net):
    solved = copy.deepcopy(net)
    pp.runpp(solved)
    check = sm.solved_state_check(solved, model)
    assert check["buses_compared"] > 40 and check["injections_compared"] > 30
    # Loads and units are as the model solved them; the example's equivalents are not.
    assert {d["table"] for d in check["largest_injection_differences"]} & {"ext_grid", "xward"}
    # A load changed by 50 MW is found among the largest differences (the external grid, which balances it, moves
    # with it; the example's equivalents differ by about 10 MW already).
    moved = copy.deepcopy(net)
    big = moved.load.p_mw.idxmax()
    moved.load.at[big, "p_mw"] += 50.0
    pp.runpp(moved)
    loads = [d for d in sm.solved_state_check(moved, model)["largest_injection_differences"] if d["table"] == "load"]
    assert loads and loads[0]["dp_mw"] == pytest.approx(50.0, abs=1e-6)


def test_no_check_without_a_solved_state(net):
    assert sm.solved_state_check(copy.deepcopy(net), _Model({})) is None


# -- two units with one name ----------------------------------------------------------------------------------------

def test_two_units_sharing_a_name_keep_their_own_nominal_values():
    n = pp.create_empty_network()
    b = [pp.create_bus(n, 20.0) for _ in range(3)]
    pp.create_ext_grid(n, b[0])
    pp.create_line_from_parameters(n, b[0], b[1], 1, 0.1, 0.1, 0, 1)
    pp.create_line_from_parameters(n, b[0], b[2], 1, 0.1, 0.1, 0, 1)
    pp.create_sgen(n, b[1], 3.0, name="Twin")
    pp.create_sgen(n, b[2], 7.0, name="Twin")
    snap = nom.snapshot(n)
    moved = copy.deepcopy(n)
    moved.sgen["p_mw"] = 0.0
    back = nom.apply(moved, snap)
    assert back.sgen.p_mw.tolist() == [3.0, 7.0]
    table = nom.as_measurements(n, snap, lambda bus: f"B{bus}")
    prod = dict(zip(table.substation_name, table.production))
    assert prod[f"B{b[1]}"] == 3.0 and prod[f"B{b[2]}"] == 7.0


# -- the backend ----------------------------------------------------------------------------------------------------

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


def test_an_upload_keeps_the_model_and_the_agent_can_read_it(client, model):
    r = client.post("/api/network/upload",
                    files={"file": ("example.zip", io.BytesIO(FIXTURE.read_bytes()), "application/zip")})
    assert r.status_code == 200, r.text
    view = client.get("/api/network/semantics", params={"about": "model"}).json()
    assert view["version"] == "2.4.15" and "solved_state" in view
    assert view["solved_state"]["buses_compared"] > 40
    summary = client.get("/api/network/semantics", params={"about": "summary"}).json()
    assert summary["cim_model"]["objects"] == model.summary()["objects"]
    # Any object by its mRID, even one the layer does not describe.
    limit_set = model.of("OperationalLimitSet")[0]
    found = client.get("/api/network/semantics", params={"about": "around", "element": limit_set}).json()
    assert found["in_model"]["class"] == "OperationalLimitSet"
    # A voltage level by its name in the model: its substation.
    around = client.get("/api/network/semantics",
                        params={"about": "around", "element": "VL for Double Busbar 1"}).json()
    assert around["about"] == "around" and "substation" in around
    issues = client.get("/api/network/semantics", params={"about": "issues"}).json()["issues"]
    assert any(i["kind"] == "solved_state_differs" for i in issues)


def test_the_model_applies_only_to_the_network_it_came_with(client):
    client.post("/api/network/upload",
                files={"file": ("example.zip", io.BytesIO(FIXTURE.read_bytes()), "application/zip")})
    assert mb._cim_model() is not None
    mb.app_data["loaded_fingerprint"] = "another network"
    assert mb._cim_model() is None and mb._cim_solved() is None


def test_the_package_keeps_the_model_untouched_beside_conductor_s_additions(client, model):
    import rdflib
    client.post("/api/network/upload",
                files={"file": ("example.zip", io.BytesIO(FIXTURE.read_bytes()), "application/zip")})
    r = client.get("/api/network/cim/package")
    assert r.status_code == 200 and r.headers["content-type"] == "application/zip"
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        names = z.namelist()
        original = next(n for n in names if n.startswith("model/"))
        assert z.read(original) == FIXTURE.read_bytes()          # byte for byte as uploaded
        additions = z.read("conductor_additions.xml").decode("utf-8")
    g = rdflib.Graph()
    g.parse(data=additions, format="xml", publicID="urn:model")
    cdr = rdflib.Namespace("urn:conductor:semantics#")
    subjects = {str(s).split("#", 1)[1] for s, _p, _o in g.triples((None, cdr.energySource, None))}
    assert subjects and all(model.get(s) is not None for s in subjects)    # the model's own objects
    assert "Model.DependentOn" in additions


def test_no_package_for_a_network_not_loaded_as_cim(client):
    mb.app_data["cim_model"] = None
    assert client.get("/api/network/cim/package").status_code == 404


def test_a_restart_records_the_network_as_loaded_not_after_the_first_tick(client):
    import json
    client.post("/api/network/upload",
                files={"file": ("example.zip", io.BytesIO(FIXTURE.read_bytes()), "application/zip")})
    sentinel = json.loads(pathlib.Path(mb._LAST_UPLOAD_SENTINEL).read_text(encoding="utf-8"))
    assert mb._lifespan_from_uploaded(sentinel)
    loaded, snap = mb.app_data["net_loaded"], mb.app_data["nominal"]["tables"]["load"]
    assert loaded.load.p_mw.to_numpy() == pytest.approx(snap["p_mw"].to_numpy())
    assert mb._cim_model() is not None and mb._cim_solved()["buses_compared"] > 40


# -- the map of a node-breaker model ---------------------------------------------------------------------------------

def _node_breaker_payload(node_breaker=True):
    return {"name": "nb", "buses": [{"index": i, "name": f"Bus {i}", "vn_kv": 110.0} for i in range(5)],
            "branches": [{"index": 0, "name": "L1", "from_bus": 1, "to_bus": 2, "kind": "line"},
                         {"index": 1, "name": "L2", "from_bus": 2, "to_bus": 4, "kind": "line"}],
            "external_grids": [{"name": "Grid", "bus": 0}],
            **({"node_breaker": True,
                "couplers": [{"from_bus": 0, "to_bus": 1, "closed": True},
                             {"from_bus": 2, "to_bus": 3, "closed": False},
                             {"from_bus": 3, "to_bus": 4, "closed": True}],
                "labels": {"1": "Busbar A"},
                "positions": {"0": [0, 0], "1": [10, 0], "2": [50, 40], "3": [80, 40], "4": [90, 50]}}
               if node_breaker else {})}


def test_nodes_joined_by_closed_switches_are_drawn_as_one():
    from llm_agent.agent import network_map as nmap
    net = nmap.from_payload(_node_breaker_payload())
    real = {i: n for i, n in net.nodes.items() if i >= 0}
    assert sorted(real) == [1, 2, 3]                     # 0 joins 1 (named), 4 joins 3; the open switch keeps 2 apart
    assert real[1].name == "Busbar A" and real[1].member_names == ["Bus 0", "Bus 1"]
    assert net.alias == {0: 1, 4: 3} and net.node(0) is real[1]
    assert {(e.from_bus, e.to_bus) for e in net.edges if e.kind == "line"} == {(1, 2), (2, 3)}
    # Results reported for a member, by name or by index, land on its node.
    nmap.apply_assessment(net, {"all_voltages": [{"bus_name": "Bus 0", "vm_pu": 1.01}]})
    assert real[1].vm_pu == pytest.approx(1.01)
    assert not nmap.apply_conditions(net, {"loads": [{"bus_index": 4, "p_mw": 2.0, "name": "D"}]})


def test_the_model_s_diagram_places_the_map():
    from llm_agent.agent import network_map as nmap
    net = nmap.from_payload(_node_breaker_payload())
    positions = nmap.compute_layout(net)
    assert net.drawn_from_diagram and "model's own diagram" in net.what_is_drawn
    # The diagram's arrangement is kept: bus 3 (x 80-90) right of bus 2 (x 50), the busbar (x 0-10) left of both.
    assert positions[1][0] < positions[2][0] < positions[3][0]
    # Another layout, asked for, ignores the diagram.
    other = nmap.from_payload(_node_breaker_payload())
    nmap.compute_layout(other, kind="spring")
    assert not other.drawn_from_diagram


def test_any_other_network_is_drawn_bus_by_bus():
    from llm_agent.agent import network_map as nmap
    net = nmap.from_payload(_node_breaker_payload(node_breaker=False))
    assert len([i for i in net.nodes if i >= 0]) == 5 and not net.alias
    nmap.compute_layout(net)
    assert not net.drawn_from_diagram


def test_a_merged_node_takes_its_members_colours():
    from llm_agent.agent import layer_map as lmap
    from llm_agent.agent import network_map as nmap
    net = nmap.from_payload(_node_breaker_payload())
    folded = lmap.fold_merged(net, {0: ("solar", "PV"), 1: ("wind", "WT"), 2: ("solar", "PV2")}, "source")
    assert folded[1][0] == "mixed" and folded[2][0] == "solar"
    known = lmap.fold_merged(net, {0: ("stated", ""), 1: ("unknown", "")}, "known")
    assert known[1][0] == "unknown"


# -- the System tab ---------------------------------------------------------------------------------------------------

def test_the_system_tab_shows_the_model(layer):
    from llm_agent.agent import semantics_view as view
    label, value, how = view.model_headline(layer)
    assert label == "CIM model" and "objects" in value and "classes in the layer" in how
    rows = view.model_table(layer)
    assert len(rows) == len(layer["cim_model"]["coverage"]["classes"])
    assert rows[0]["Status"] == "Described by the layer"
    h = view.model_hierarchy(layer)
    assert set(h["parents"]) - {""} <= set(view.STATUS_LABELS)
    assert {r["CIM class"] for r in view.other_table(layer)} >= {"EquivalentInjection"}
    assert "Voltage level" in view.bus_table(layer)[0]
    assert view.model_headline({"buses": []}) is None


def test_limits_read_as_people_say_them():
    from llm_agent.agent import semantics_view as view
    limits = {"value": [{"terminal": 1, "kind": "CurrentLimit", "value": 1000.0, "permanent": True, "duration_s": None,
                         "limit_type": "patl"},
                        {"terminal": 1, "kind": "CurrentLimit", "value": 1150.0, "permanent": False,
                         "duration_s": 900.0, "limit_type": "tatl"},
                        {"terminal": 1, "kind": "CurrentLimit", "value": 1000.0, "permanent": False,
                         "duration_s": 4000.0, "limit_type": "patlt"},
                        {"terminal": 2, "kind": "CurrentLimit", "value": 999.0, "permanent": True, "duration_s": None}]}
    assert view.limits_text(limits) == "1000 A permanent · 1150 A 15 min"
