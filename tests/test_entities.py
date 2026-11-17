"""
What an answer says the network's elements are, checked against what the
network says (llm_agent/agent/entities.py) — on the facts the turn's record keeps.
"""

import pandapower.networks as pn
import pytest

import semantics as sm
from llm_agent.agent import entities as en


def _record(net, answer, declared=None, user="", declared_loads=None):
    """The record the loop keeps: the compact facts, filtered to what the answer names."""
    facts = sm.compact(sm.build(net, declared, declared_loads=declared_loads))
    units = facts["units"]
    kept = {
        "per_unit_case": facts["per_unit_case"], "voltage_levels_kv": facts["voltage_levels_kv"],
        "external_grids": facts["external_grids"],
        "units_by_source": {s: sorted(u for u, f in units.items() if f["source"] == s and f["basis"] != "unknown")
                            for s in {f["source"] for f in units.values()} if s != "unknown"},
        "buses": {k: v for k, v in facts["buses"].items() if k in answer},
        "branches": {k: v for k, v in facts["branches"].items() if k in answer},
        "units": {k: v for k, v in units.items() if k in answer},
        "loads": {k: v for k, v in facts["loads"].items() if k in answer},
        "names_not_in_network": en.names_not_in_network(answer, [
            *facts["buses"], *facts["loads"], *units, *facts["branches"],
            *(k.split(" | ")[0] for k in facts["branches"])]),
    }
    return {"user": user, "network_facts": {"base": kept}}


def test_a_loads_category_is_checked_like_a_units_source(case30):
    assert _kinds(case30, "Load 5, an industrial customer, peaks at 18:00.") == [en.UNSUPPORTED]
    declared = sm.normalise_load_declarations(case30, [{"load": "Load 5", "category": "data_centre"}])
    assert _kinds(case30, "Load 5, a data centre, peaks at 18:00.", declared_loads=declared) == []
    assert _kinds(case30, "Load 5, an industrial customer, peaks at 18:00.",
                  declared_loads=declared) == [en.CONTRADICTED]


def _kinds(net, answer, **kw):
    return [i.kind for i in en.check_answer(answer, _record(net, answer, **kw))]


@pytest.fixture(scope="module")
def case30():
    return pn.case30()


@pytest.fixture(scope="module")
def cigre():
    return pn.create_cigre_network_mv(with_der="all")


def test_a_source_the_network_does_not_state_is_unsupported(case30):
    assert _kinds(case30, "Gen_bus22, a wind farm, is at its limit.") == [en.UNSUPPORTED]


def test_a_declared_source_is_supported(case30):
    declared = sm.normalise_declarations(case30, [{"unit": "Gen_bus22", "source": "wind"}])
    assert _kinds(case30, "Gen_bus22, a wind farm, is at its limit.", declared=declared) == []


def test_a_source_against_the_stated_one_is_contradicted(cigre):
    assert _kinds(cigre, "The wind unit WKA 7 is at 2 MW.") == []
    assert _kinds(cigre, "WKA 7, a solar plant, is at 2 MW.") == [en.CONTRADICTED]


def test_held_back_or_denied_claims_are_not_claims(case30):
    assert _kinds(case30, "If Gen_bus22 were a wind farm, its output would vary.") == []
    assert _kinds(case30, "The network does not state whether Gen_bus22 is wind or solar.") == []
    assert _kinds(case30, "There are no wind units in this network.") == []


def test_a_group_with_no_known_member_is_unsupported(case30, cigre):
    assert _kinds(case30, "The renewable generation covers the peak.") == [en.UNSUPPORTED]
    assert _kinds(cigre, "The solar generation covers the peak.") == []


def test_voltages_are_checked_against_the_network(cigre, case30):
    assert _kinds(cigre, "The fault sits on the 20 kV feeder.") == []
    assert _kinds(cigre, "The fault sits on the 400 kV feeder.") == [en.UNSUPPORTED]
    assert _kinds(cigre, "Bus 5 at 110 kV is the weakest.") == [en.CONTRADICTED]
    assert _kinds(cigre, "The fault sits on the 400 kV feeder.", user="assume a 400 kV feeder") == []
    net = pn.case14()
    net.bus["vn_kv"] = 1.0
    issues = en.check_answer("Bus_9 at 132 kV is low.", _record(net, "Bus_9 at 132 kV is low."))
    assert [i.kind for i in issues] == [en.UNSUPPORTED] and "per-unit" in issues[0].note


def test_a_branch_called_by_the_wrong_kind_or_ends_is_contradicted(cigre):
    facts = sm.compact(sm.build(cigre))["branches"]
    line = next(k for k, v in facts.items() if v["type"] == "line")
    trafo = next(k for k, v in facts.items() if v["type"] == "trafo")
    assert _kinds(cigre, f"The transformer {line} is loaded at 80%.") == [en.CONTRADICTED]
    assert _kinds(cigre, f"The line {trafo} is loaded at 80%.") == [en.CONTRADICTED]
    a, b = facts[line]["ends"]
    assert _kinds(cigre, f"{line} between {a} and {b} is loaded at 80%.") == []


def test_no_facts_no_checks():
    assert en.check_answer("Gen_bus22, a wind farm.", {"network_facts": {}}) == []


def test_issues_reach_the_answer_review(case30):
    from llm_agent.agent import reflection
    rec = _record(case30, "Gen_bus22, a wind farm, is at its limit.")
    conflicts = en.as_conflicts("Gen_bus22, a wind farm, is at its limit.", rec)
    assert conflicts and conflicts[0].path == "entities"
    assert hasattr(reflection, "_entity_issues")


def test_a_list_of_units_says_nothing_about_each_one(cigre):
    text = "The units cut off are PV 3, PV 4, PV 5, WKA 7, Battery 1 and CHP diesel 1."
    assert _kinds(cigre, text) == []
    assert _kinds(cigre, "WKA 7, a solar plant next to PV 3, is at 2 MW.") == [en.CONTRADICTED]
    # Observed live: names shortened in a list.
    shortened = ("The units on the cut-off buses are PV 3–6, PV 8–11, WKA 7, Residential fuel cells 1 and 2, "
                 "CHP diesel 1, Fuel cell 1, and Batteries 1 and 2.")
    assert _kinds(cigre, shortened) == []
    assert _kinds(cigre, "WKA 7 is a solar unit.") == [en.CONTRADICTED]
    assert _kinds(cigre, "WKA 7 (solar) is at 2 MW.") == [en.CONTRADICTED]


def test_an_element_the_network_does_not_have_is_contradicted(cigre):
    # Observed live: "Load R9" in a list of the loads cut off; CIGRE MV has no R9.
    issues = en.check_answer("Loads cut off: Load R8, Load R9 and Load CI9.",
                             _record(cigre, "Loads cut off: Load R8, Load R9 and Load CI9."))
    assert [i.kind for i in issues] == [en.CONTRADICTED] and "Load R9" in issues[0].note
    assert _kinds(cigre, "There is no Load R9 in this network.") == []
    assert _kinds(cigre, "Load R9 is not in the network.", user="Is Load R9 overloaded?") == []
    assert _kinds(cigre, "PV 3–6 and Bus 4–11 lose supply.") == []                   # ranges of real names


def test_a_name_a_proposal_adds_is_not_missing(cigre):
    rec = _record(cigre, "Bus 15 is at 0.96 p.u.")
    rec["network_facts"]["new_site"] = {"names_not_in_network": []}                 # the proposal has it
    assert en.check_answer("Bus 15 is at 0.96 p.u.", rec) == []


def test_the_offline_grader_fails_a_contradiction_and_warns_on_the_unsupported(case30, cigre):
    from evaluation.graders.entities import grade_entities
    from evaluation.session_log import FAIL, WARN
    rec = {**_record(cigre, "WKA 7, a solar plant, is at 2 MW."), "assistant": "WKA 7, a solar plant, is at 2 MW."}
    assert [f.severity for f in grade_entities(rec)] == [FAIL]
    rec = {**_record(case30, "Gen_bus22, a wind farm, is at its limit."),
           "assistant": "Gen_bus22, a wind farm, is at its limit."}
    assert {f.severity for f in grade_entities(rec)} == {WARN}


def test_the_false_positives_seen_in_the_reasoning_ablation(case30, cigre):
    # A conjunction ends a description.
    text = "The weather-dependent units are the eight solar PV units—PV 3, PV 10 and PV 11—and the wind unit WKA 7."
    assert _kinds(cigre, text) == []
    # The operator's own statement, repeated.
    assert _kinds(case30, "Understood: Gen_bus22 is a wind farm.", user="Gen_bus22 is a wind farm.") == []
    # The operator's words, and a denial.
    q = "If wind and solar output dropped by 50 %, would the grid stay secure?"
    assert _kinds(case30, "With wind/solar output reduced to 50 %, there are no violations.", user=q) == []
    assert _kinds(case30, "The network does not identify any solar unit.") == []
    # Still caught.
    assert _kinds(case30, "Gen_bus22, a wind farm, is at its limit.") == [en.UNSUPPORTED]
    assert _kinds(cigre, "PV 11 — a wind unit — is at 2 MW.") == [en.CONTRADICTED]


def test_a_unit_listed_under_the_wrong_source_is_contradicted():
    # Observed in the reasoning ablation, without the layer: a CHP unit renamed "Solar Farm North"
    # and a fuel cell renamed "PV Park East" listed among the solar units.
    net = pn.create_cigre_network_mv(with_der="all")
    net.sgen.at[10, "name"], net.sgen.at[11, "name"] = "Solar Farm North", "PV Park East"
    text = "There are 10 solar units: PV 3, PV 4, PV 11, Solar Farm North, and PV Park East."
    issues = en.check_answer(text, _record(net, text))
    # Each unit listed wrongly is named (one finding per sentence hid the second).
    assert [i.kind for i in issues] == [en.CONTRADICTED, en.CONTRADICTED]
    assert "Solar Farm North" in issues[0].note and "PV Park East" in issues[1].note
    # The list stops at the next source word.
    ok = "The solar PV units are PV 3, PV 4 and PV 11, and the wind unit WKA 7."
    assert en.check_answer(ok, _record(net, ok)) == []


def test_mixed_lists_and_closed_lists_claim_nothing_wrong(cigre):
    for text in ("The affected units are the solar and wind units: PV 3, PV 4, PV 11, and WKA 7.",
                 "The renewable units are the PV units—PV 3, PV 4 and PV 11—and WKA 7 (wind).",
                 "The solar units are PV 3, PV 4, and WKA 7 (wind)."):
        assert _kinds(cigre, text) == [], text


def test_a_value_called_measured_must_be_measured():
    net = pn.create_cigre_network_mv(with_der="all")
    rows = __import__("pandas").DataFrame({"substation_name": ["Bus 13", "Bus 9", "Bus 1"],
                                           "consumption": [0.02, 0.3, 10.0], "production": [0.0, 0.4, 0.0]})
    prov = sm.value_provenance(net, rows, load_pf=0.97, load_pf_source="the network's profile")

    def kinds(text):
        facts = sm.compact(sm.build(net, provenance=prov))
        rec = {"user": "", "network_facts": {"base": {
            "per_unit_case": False, "voltage_levels_kv": facts["voltage_levels_kv"], "external_grids": [],
            "units_by_source": {}, "buses": {}, "branches": {},
            "units": {k: v for k, v in facts["units"].items() if k in text},
            "loads": {k: v for k, v in facts["loads"].items() if k in text}}}}
        return [(i.kind, i.note) for i in en.check_answer(text, rec)]

    issues = kinds("The measured reactive demand of Load CI13 is 0.005 Mvar.")
    assert [k for k, _ in issues] == [en.CONTRADICTED] and "0.97" in issues[0][1]
    assert [k for k, _ in kinds("CHP diesel 1's measured output is 0.1 MW.")] == [en.UNSUPPORTED]   # a share
    assert kinds("Load CI13's measured demand is 0.02 MW.") == []                                    # measured
    assert kinds("Load CI13's reactive demand is not measured: it comes from the power factor.") == []
