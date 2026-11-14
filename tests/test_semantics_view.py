"""The network description pane's rows and picture (llm_agent/agent/semantics_view.py)."""

import pandapower.networks as pn

import semantics as sm
from llm_agent.agent import semantics_view as sv


def test_unknown_units_come_first_and_say_so():
    rows = sv.unit_rows(sm.build(pn.case30()))
    assert rows and all(r["value"] == "unknown" and r["cells"][3] == "Unknown" for r in rows)


def test_a_declared_source_is_told_apart_from_a_stated_one():
    net = pn.case30()
    declared = sm.normalise_declarations(net, [{"unit": "Gen_bus22", "source": "wind"}])
    rows = {r["unit"]: r for r in sv.unit_rows(sm.build(net, declared))}
    assert rows["Gen_bus22"]["declared"] and rows["Gen_bus22"]["cells"][3] == "Declared by you"
    assert not rows["Gen_bus1"]["declared"]


def test_the_picture_nests_units_under_their_substation_and_level():
    s = sm.build(pn.create_cigre_network_mv(with_der="all"))
    h = sv.hierarchy(s)
    assert {"110 kV", "20 kV"} & set(h["ids"])
    leaves = [i for i, p in zip(h["ids"], h["parents"]) if p.count("/") == 1]
    assert len(leaves) == len(s["units"])
    assert sv.colour("wind", "stated") == sv.SOURCE_COLOURS["wind"]
    assert sv.colour("wind", "named") != sv.SOURCE_COLOURS["wind"]
    sv.figure(s)


def test_a_per_unit_case_has_no_stated_level():
    net = pn.case14()
    net.bus["vn_kv"] = 1.0
    s = sm.build(net)
    assert sv.headline(s)[0][1] == "not stated"
    assert set(sv.hierarchy(s)["ids"]) >= {"voltage not stated"}


def test_loads_get_their_own_picture_and_rows():
    s = sm.build(pn.example_multivoltage())
    rows = {r["unit"]: r for r in sv.load_rows(s)}
    assert rows["Industry Load"]["cells"][2] == "industrial" and rows["Industry Load"]["cells"][3] == "From its name"
    h = sv.hierarchy(s, "loads")
    assert sum(p.count("/") == 1 for p in h["parents"]) == len(s["loads"])
    assert ("industrial", sv.CATEGORY_COLOURS["industrial"]) in sv.legend(s, "loads")
    assert len(sv.line_table(s)) + len(sv.trafo_table(s)) == len(s["branches"])
    assert sv.bus_table(s)[0]["kV basis"] == "Stated"


def test_looking_around_a_line_shows_both_ends_its_supply_and_what_it_cuts_off():
    net = pn.create_cigre_network_mv(with_der="all")
    s = sm.build(net)
    station = next(x for x in s["containers"]["substations"] if x["name"] == "Substation Bus 0")
    assert ("Substation · Substation Bus 0", station["mrid"]) in sv.around_choices(s)   # sent by its mRID
    cards = dict(sv.around_cards(sm.around(net, s, "line", 1)))                  # Line 2-3
    assert set(cards) == {"Where it sits", "How it is supplied", "Without it"}
    assert any("Feeder: **Line 1-2**" in x for x in cards["How it is supplied"])
    assert any("Line 14-8" in x for x in cards["Without it"])                     # the tie that restores it


def test_switches_feeders_and_data_tables():
    net = pn.create_cigre_network_mv(with_der="all")
    rows = __import__("pandas").DataFrame({"substation_name": ["Bus 13"], "consumption": [0.02], "production": [0.0]})
    s = sm.build(net, provenance=sm.value_provenance(net, rows, moment="2000-01-01 00:00:00"))
    sw = {r["Switch"]: r for r in sv.switch_table(s)}
    assert sw["S1"]["Normally"] == "open" and sw["S1"]["Role"].startswith("tie between feeders")
    assert {r["Feeder"] for r in sv.feeder_table(s)} == {"Line 1-2", "Line 12-13"}
    data = {r["Element"]: r for r in sv.data_table(s)}
    assert data["Load CI13"]["P"] == "Measured" and data["Load CI13"]["Q"] == "Derived (assumption)"
    assert "2000-01-01 00:00:00" in sv.data_caption(s) and "power factor" in sv.data_caption(s)
    assert sv.bus_table(s)[3]["Feeder"] == "Line 1-2"


def test_tables_show_branches_by_their_short_name():
    assert sv._short("tie between feeders Line 1-2 | Bus 1 -> Bus 2 [L0] and Line 12-13 | Bus 12 -> Bus 13 [L10]") \
        == "tie between feeders Line 1-2 and Line 12-13"


def test_shunt_sizes_are_unsigned_and_the_kind_gives_the_direction():
    rows = sv.shunt_table(sm.build(pn.case30()))
    assert rows and all(r["Kind"] == "capacitor" and r["Size at 1 p.u. (Mvar)"] > 0 for r in rows)
