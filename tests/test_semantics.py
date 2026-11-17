"""
What each element of the loaded network is, and how we know (backend/semantics.py).

The layer must adapt to whatever network is loaded: a MATPOWER case that says
nothing about its units, a pandapower network that types every generator, a
multi-voltage network with bus couplers — and never fill in what the network
does not say.
"""


import pandapower.networks as pn
import pytest

import semantics as sm


@pytest.fixture(scope="module")
def cigre():
    return pn.create_cigre_network_mv(with_der="all")


@pytest.fixture(scope="module")
def multivoltage():
    return pn.example_multivoltage()


def _by_name(units):
    return {u["name"]: u for u in units}


def test_a_case_that_says_nothing_about_its_units_leaves_them_unknown():
    s = sm.build(pn.case30())
    assert {u["source"]["value"] for u in s["units"]} == {"unknown"}
    assert all(u["source"]["basis"] == sm.UNKNOWN for u in s["units"])
    assert sm.view(s, "units")["wind_or_solar"] == [] and "never guessed" in sm.view(s, "units")["note"]


def test_a_network_that_types_its_units_is_read_as_it_states(cigre):
    s = sm.build(cigre)
    counts = {k: len(v) for k, v in s["units_by_source"].items()}
    assert counts == {"solar": 8, "wind": 1, "thermal": 4, "storage": 2}
    assert all(u["source"]["basis"] == sm.STATED for u in s["units"])
    assert s["voltage_levels_kv"] == [110.0, 20.0]


def test_a_name_is_a_weaker_basis_and_a_generic_type_is_no_statement(multivoltage):
    units = _by_name(sm.build(multivoltage)["units"])
    assert units["Wind Park"]["source"] == {"value": "wind", "basis": sm.STATED, "from": "its 'type' in the network ('WP')"}
    assert units["Biogas plant"]["source"]["value"] == "bioenergy" and units["Biogas plant"]["source"]["basis"] == sm.NAMED
    assert units["Further MV Generator"]["source"]["basis"] == sm.UNKNOWN      # type 'SGEN': no statement


def test_substations_are_buses_joined_by_transformers_or_couplers_and_the_external_grid_what_leaves_them(multivoltage):
    s = sm.build(multivoltage)
    ext = s["external_grids"][0]
    assert ext["substation"] == "Substation Double Busbar 1"
    assert any("[T0]" in b for b in ext["branches"]) and any("[L0]" in b for b in ext["branches"])
    assert s["substations"]["basis"] == sm.INFERRED and len(s["voltage_levels_kv"]) == 5


def test_a_per_unit_case_has_no_voltage_levels():
    net = pn.case14()
    net.bus["vn_kv"] = 1.0
    s = sm.build(net)
    assert s["per_unit_case"] and s["voltage_levels_kv"] == []
    assert all(b["kv"]["basis"] == sm.UNKNOWN for b in s["buses"])
    assert "placeholder" in sm.view(s, "voltage_levels")["note"]


def test_a_condenser_is_inferred_and_said_so():
    net = pn.case14()
    net.gen.loc[[1, 2, 3], "max_p_mw"] = 0.0          # as PGLib's case14 has them: no active power
    s = sm.build(net)
    condensers = [u for u in s["units"] if u["source"]["value"] == "condenser"]
    assert condensers and all(u["source"]["basis"] == sm.INFERRED for u in condensers)


def test_a_unit_added_by_a_proposal_is_what_the_proposal_says():
    import sandbox as sb
    net = pn.case14()
    net.bus["name"] = None
    prop, _ = sb.materialise(net, [{"op": "add_generator", "name": "PV 9", "bus": "Bus_9", "p_max_mw": 5,
                                    "kind": "pv"}])
    unit = _by_name(sm.build(prop)["units"])["PV 9"]
    assert unit["source"]["value"] == "solar" and unit["source"]["basis"] == sm.STATED


def test_the_operator_declares_what_the_network_does_not_say():
    net = pn.case30()
    declared = sm.normalise_declarations(net, [{"unit": "Gen_bus1", "source": "wind"},
                                               {"unit": "bus 22", "source": "solar"}])
    units = _by_name(sm.build(net, declared)["units"])
    assert units["Gen_bus1"]["source"] == {"value": "wind", "basis": sm.STATED, "from": "declared by the operator"}
    assert units["Gen_bus22"]["source"]["value"] == "solar"
    withdrawn = sm.normalise_declarations(net, [{"unit": "Gen_bus1", "source": "unknown"}], declared)
    assert "Gen_bus1" not in withdrawn and "Gen_bus22" in withdrawn
    with pytest.raises(sm.DeclarationError, match="source must be one of"):
        sm.normalise_declarations(net, [{"unit": "Gen_bus1", "source": "unicorns"}])
    with pytest.raises(sm.DeclarationError, match="no unit"):
        sm.normalise_declarations(net, [{"unit": "Nope", "source": "wind"}])


def test_the_facts_kept_with_an_answer_are_compact(cigre):
    facts = sm.compact(sm.build(cigre))
    assert set(facts) == {"per_unit_case", "voltage_levels_kv", "buses", "branches", "units", "loads",
                          "external_grids", "switches", "feeders"}
    assert facts["units"]["WKA 7"]["source"] == "wind"


def test_a_load_is_what_its_name_says_and_a_code_is_no_statement(multivoltage, cigre):
    loads = {d["name"]: d for d in sm.build(multivoltage)["loads"]}
    assert loads["Industry Load"]["category"]["value"] == "industrial"
    assert loads["Residential Load"]["category"]["basis"] == sm.NAMED
    assert loads["MV Net 0"]["category"]["value"] == "network"            # a downstream network as one load
    # CIGRE's "Load R1" / "Load CI1" are benchmark codes, not statements.
    assert {d["category"]["basis"] for d in sm.build(cigre)["loads"]} == {sm.UNKNOWN}
    assert "never guessed" in sm.view(sm.build(cigre), "loads")["note"]


def test_the_operator_declares_a_loads_category():
    net = pn.case30()
    declared = sm.normalise_load_declarations(net, [{"load": "Load 5", "category": "data centre"}])
    loads = {d["index"]: d for d in sm.build(net, declared_loads=declared)["loads"]}
    assert loads[5]["category"] == {"value": "data_centre", "basis": sm.STATED, "from": "declared by the operator"}
    assert sm.normalise_load_declarations(net, [{"load": "Load 5", "category": "unknown"}], declared) == {}
    with pytest.raises(sm.DeclarationError, match="category must be one of"):
        sm.normalise_load_declarations(net, [{"load": "Load 5", "category": "spaceport"}])


def test_the_backend_serves_the_layer_and_keeps_declarations(backend_client):
    summary = backend_client.get("/api/network/semantics", params={"about": "summary"}).json()
    assert summary["units_by_source"] and "external_grid_at" in summary
    units = backend_client.get("/api/network/semantics", params={"about": "units"}).json()
    name = units["unknown"][0]
    try:
        out = backend_client.post("/api/network/declare", json={"units": [{"unit": name, "source": "wind"}]}).json()
        assert name in out["units_by_source"]["wind"]
        assert name in backend_client.get("/api/network/semantics", params={"about": "units"}).json()["wind_or_solar"]
        load = backend_client.get("/api/network/semantics").json()["loads"][0]["name"]
        out = backend_client.post("/api/network/declare", json={"loads": [{"load": load, "category": "industrial"}]})
        assert out.json()["loads_by_category"]["industrial"] == 1
        found = backend_client.get("/api/network/resolve", params={"name": load, "kinds": "bus,load",
                                                                     "facts": True}).json()
        assert found["element_type"] == "load" and found["facts"]["category"]["value"] == "industrial"
        assert backend_client.post("/api/network/declare", json={}).status_code == 400
    finally:
        backend_client.post("/api/network/declare", json={"reset": True})
    assert backend_client.get("/api/network/semantics", params={"about": "nonsense"}).status_code == 400


def test_around_a_line_gives_both_ends_its_route_and_what_it_cuts_off(cigre):
    a = sm.around(cigre, sm.build(cigre), "line", 3)                     # Line 4-5
    assert [s["name"] for s in a["substations_at_its_ends"]] == ["Bus 4", "Bus 5"]
    assert a["supply"]["route_from_external_grid"][0].startswith("Trafo 0-1")
    assert a["supply"]["feeder"].startswith("Line 1-2") and a["supply"]["basis"] == sm.INFERRED
    out = a["without_it"]
    assert out["buses_cut_off"] == ["Bus 5", "Bus 6"] and "Load R5" in out["loads_cut_off"]
    assert any("Line 6-7" in t for t in out["open_ties_that_could_restore"])      # the normally open tie


def test_around_a_substation_lists_what_is_in_it(cigre):
    a = sm.around(cigre, sm.build(cigre), "substation", "Substation Bus 0")
    assert a["substation"]["external_grid_here"] and len(a["substation"]["transformers_inside"]) == 2
    assert a["without_it"]["load_cut_off_mw"] > 40


def test_a_feeder_belongs_to_its_voltage_level_and_a_meshed_one_has_none(multivoltage):
    s = sm.build(multivoltage)
    load = {d["name"]: d["index"] for d in s["loads"]}
    residential = sm.around(multivoltage, s, "load", load["Residential Load"])["supply"]
    assert residential["feeder"].startswith("LV Line1.1") and residential["radial_at_its_voltage_level"]
    assert "bus coupler" not in " ".join(residential["route_from_external_grid"])
    meshed = sm.around(pn.case30(), sm.build(pn.case30()), "line", 5)["supply"]
    assert meshed["feeder"] is None and "meshed" in meshed["note"]


def test_the_backend_answers_around_an_element(backend_client):
    bus = backend_client.get("/api/network/semantics").json()["buses"][3]["name"]
    out = backend_client.get("/api/network/semantics", params={"about": "around", "element": bus}).json()
    assert out["element"]["kind"] == "bus" and "supply" in out and "without_it" in out
    assert backend_client.get("/api/network/semantics", params={"about": "around"}).status_code == 400
    assert backend_client.get("/api/network/semantics",
                              params={"about": "around", "element": "Nowhere"}).status_code == 404
