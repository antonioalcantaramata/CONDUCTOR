"""
Switches, feeders and where each value comes from (backend/semantics.py):
what the network states about its switches and how they tie its feeders, and,
for every load and unit at the present operating point, whether a value is
measured, a share of a bus's measurement, derived from an assumption, solved
by the power flow, or the case's own.
"""

import pandapower.networks as pn
import pandas as pd
import pytest

import semantics as sm


@pytest.fixture(scope="module")
def cigre():
    net = pn.create_cigre_network_mv(with_der="all")
    return net, sm.build(net)


def _sw(s, name):
    return next(x for x in s["switches"] if x["name"] == name)


def test_switches_are_typed_as_the_network_states(cigre):
    _, s = cigre
    assert {x["cim_class"]["value"] for x in s["switches"]} == {"LoadBreakSwitch", "Breaker"}
    assert all(x["cim_class"]["basis"] == sm.STATED for x in s["switches"])


def test_the_tie_between_feeders_is_told_apart_from_a_loop_point(cigre):
    _, s = cigre
    s1 = _sw(s, "S1")                                          # Line 14-8, open at Bus 8
    assert s1["normal_open"]["value"] and s1["role"]["value"].startswith("tie between feeders")
    assert "Line 1-2" in s1["role"]["value"] and "Line 12-13" in s1["role"]["value"]
    assert _sw(s, "S2")["role"]["value"].startswith("normally open point closing a loop within feeder")
    assert s1["where"]["element"].startswith("Line 14-8")


def test_normal_and_present_states_are_kept_apart():
    net = pn.create_cigre_network_mv(with_der="all")
    normal = {int(i): not bool(c) for i, c in net.switch["closed"].items()}
    net.switch.loc[net.switch.name == "S1", "closed"] = True             # the tie closed now
    s1 = _sw(sm.build(net, normal_open=normal), "S1")
    assert s1["normal_open"]["value"] is True and s1["open"]["value"] is False
    assert "switched_from_normal" in sm.view(sm.build(net, normal_open=normal), "switches")
    assert sm.view(sm.build(net, normal_open=normal), "switches")["switched_from_normal"] == ["S1"]


def test_feeders_are_containers_with_what_they_supply(cigre):
    _, s = cigre
    feeders = {f["name"].split(" | ")[0]: f for f in s["containers"]["feeders"]}
    assert set(feeders) == {"Line 1-2", "Line 12-13"}
    assert "Load R11" in feeders["Line 1-2"]["loads"] and "WKA 7" in feeders["Line 1-2"]["units"]
    assert feeders["Line 1-2"]["ties"] == ["S1"] and feeders["Line 12-13"]["ties"] == ["S1"]
    hv = next(x for x in s["containers"]["substations"] if x["name"] == "Substation Bus 0")
    assert [v["kv"] for v in hv["voltage_levels"]] == [110.0, 20.0]
    assert all(v["cim_class"] == "VoltageLevel" for v in hv["voltage_levels"])
    bus1 = next(b for b in s["buses"] if b["name"] == "Bus 1")
    assert bus1["feeder"]["value"] is None and "busbar" in bus1["feeder"]["from"]


def test_a_meshed_network_has_no_feeders_and_says_why():
    s = sm.build(pn.case30())
    assert s["containers"]["feeders"] == [] and s["switches"] == []
    assert "meshed" in s["buses"][5]["feeder"]["from"] and "meshed" in sm.view(s, "feeders")["note"]


def _rows(**cols):
    return pd.DataFrame(cols)


def test_value_provenance_follows_the_data():
    net = pn.create_cigre_network_mv(with_der="all")
    # Bus 1 carries two loads (R1 and CI1); Bus 3 two (R3, CI3); Bus 13 one (CI13). Bus 9: PV 9, CHP, fuel cell.
    rows = _rows(substation_name=["Bus 1", "Bus 13", "Bus 9", "Bus 7"], consumption=[10.0, 0.02, 0.3, 0.05],
                 production=[0.0, 0.0, 0.4, 0.2])
    p = sm.value_provenance(net, rows, moment="t", synthetic=True, load_pf=0.97, load_pf_source="the network's profile")
    load = {net.load.at[int(i), "name"]: v for i, v in p["loads"].items()}
    assert load["Load CI13"]["p"]["kind"] == "measured" and load["Load R1"]["p"]["kind"] == "allocated"
    assert load["Load CI13"]["q"]["kind"] == "derived" and "0.97" in load["Load CI13"]["q"]["from"]
    assert load["Load R5"]["p"]["kind"] == "nominal"                          # Bus 5 not in these rows
    unit = {sm.gm.unit_name(net, *k.split(":")[:1], int(k.split(":")[1])): v
            for k, v in p["units"].items() if not k.startswith("storage")}
    assert unit["CHP diesel 1"]["p"]["kind"] == "allocated" and unit["WKA 7"]["p"]["kind"] == "measured"
    assert unit["WKA 7"]["q"]["kind"] == "derived"                            # a fixed power factor
    assert p["units"]["storage:0"]["p"]["kind"] == "set" and p["context"]["synthetic"] is True
    # Measured reactive demand wins over the assumption.
    rows_q = rows.assign(consumption_q=[2.0, 0.005, None, None])
    q = sm.value_provenance(net, rows_q)["loads"]
    assert {v["q"]["kind"] for i, v in q.items() if net.load.at[int(i), "name"] == "Load CI13"} == {"measured"}


def test_a_voltage_controlled_units_q_is_solved_and_a_proposal_is_its_own():
    import sandbox as sb
    net = pn.case14()
    net.bus["name"] = None
    prop, _ = sb.materialise(net, [{"op": "add_load", "name": "DC", "bus": "Bus_9", "p_mw": 5}])
    rows = _rows(substation_name=["Bus_1", "Bus_9"], consumption=[0.0, 20.0], production=[30.0, 0.0])
    p = sm.value_provenance(prop, rows)
    assert p["units"]["gen:0"]["q"]["kind"] == "solved"
    dc = next(v for i, v in p["loads"].items() if prop.load.at[int(i), "name"] == "DC")
    assert dc["p"]["kind"] == "proposal"


def test_the_backend_says_where_values_come_from(backend_client):
    data = backend_client.get("/api/network/semantics", params={"about": "data"}).json()
    assert data["dataset"] == "measurements" and data["moment"] and data["synthetic"] is True
    assert data["load_power_factor"]["value"] > 0
    assert set(data["loads"]["count_by_kind"]) <= {"p", "q"}
    assert "derived" in data["loads"]["count_by_kind"]["q"]                   # IEEE 30: no reactive data
    sws = backend_client.get("/api/network/semantics", params={"about": "switches"}).json()
    assert sws["count"] == 0
    assert backend_client.get("/api/network/semantics", params={"about": "feeders"}).json()["feeders"] == []


def test_units_are_classed_as_distribution_cim_has_them(cigre):
    _, s = cigre
    u = {x["name"]: x for x in s["units"]}
    assert (u["PV 3"]["cim_class"], u["PV 3"]["cim_unit"]) == ("PowerElectronicsConnection", "PhotovoltaicUnit")
    assert u["WKA 7"]["cim_unit"] == "PowerElectronicsWindUnit"
    assert (u["CHP diesel 1"]["cim_class"], u["CHP diesel 1"]["cim_unit"]) == ("SynchronousMachine",
                                                                               "ThermalGeneratingUnit")
    assert u["Fuel cell 1"]["cim_class"] == "PowerElectronicsConnection"         # inverter-based
    assert u["Battery 1"]["cim_unit"] == "BatteryUnit"
    assert u["PV 3"]["cim_control"] == {"class": "RegulatingControl", "mode": "powerFactor"}


def test_a_unit_whose_source_is_unknown_keeps_the_abstract_class():
    s = sm.build(pn.case30())
    gens = [x for x in s["units"] if x["element_type"] == "gen"]
    assert all(x["cim_class"] == "SynchronousMachine" and x["cim_unit"] is None for x in gens)
    assert all(x["cim_control"]["mode"] == "voltage" for x in gens)
    assert sm.cim_unit("sgen", "unknown") == ("GeneratingUnit", None)


def test_shunts_are_linear_shunt_compensators():
    s = sm.build(pn.case30())
    assert {x["cim_class"] for x in s["shunts"]} == {"LinearShuntCompensator"}
    assert all(x["kind"]["value"] == "capacitor" and x["kind"]["basis"] == sm.STATED for x in s["shunts"])
    assert sm.view(s, "summary")["shunts"]
