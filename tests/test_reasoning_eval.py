"""The reasoning evaluation's checks (evaluation/reasoning.py) — without model credit."""

import pytest

from evaluation import reasoning as rs


def _rec(text, results=()):
    return [{"assistant": text, "tool_results": [{"result": r} for r in results]}]


def test_a_misleading_name_must_be_seen_through():
    spec = {"kind": "name_handled", "name": "Solar Farm North", "ok_any": ["chp", "despite"]}
    assert rs.check(spec, _rec("The solar units are PV 3 … PV 11."))                       # not mentioned
    assert rs.check(spec, _rec("Solar Farm North is a CHP unit despite its name."))
    assert not rs.check(spec, _rec("Solar units: PV 3, PV 4 and Solar Farm North."))


def test_a_name_alone_must_be_said_to_be_a_name():
    spec = {"kind": "name_hedged", "name": "Solar Plant A", "hedge_any": ["its name", "not stated"]}
    assert rs.check(spec, _rec("Only Solar Plant A, and only by its name: the network does not say."))
    assert not rs.check(spec, _rec("Solar Plant A is the solar unit."))
    assert not rs.check(spec, _rec("No unit is solar."))
    assert rs.check(spec, _rec("Solar Plant A is the only solar unit. Its source is inferred from its name."))
    assert not rs.check(spec, _rec("Solar Plant A is solar. Gen_bus4 holds 1.0 p.u. The rest is not stated."))


def test_no_unit_affected_unless_denied():
    spec = {"kind": "none_affected", "names": ["Gen_bus12"]}
    assert rs.check(spec, _rec("Gen_bus12 is not affected: no unit is stated wind or solar."))
    assert not rs.check(spec, _rec("The drop would reduce Gen_bus12's output."))


def test_the_study_must_have_scaled_exactly_those_units():
    spec = {"kind": "scaled_units", "units": ["Gen_bus12"]}
    assert rs.check(spec, _rec("", [{"renewables_scaled": {"factor": 0.5, "units": ["Gen_bus12"]}}]))
    assert not rs.check(spec, _rec("", [{"renewables_scaled": {"factor": 0.5, "units": []}}]))
    assert not rs.check(spec, _rec("Gen_bus12 drops."))


def test_every_task_names_its_network_and_pairs_exist():
    assert {i.network for i in rs.ITEMS} == {"cigre", "ieee30"}
    assert all(i.pair in rs.TASKS for i in rs.ITEMS if i.pair)
    assert {i.family for i in rs.ITEMS} == {"names", "intervention", "composition", "switches", "data", "consistency"}


def test_a_unit_lowered_by_a_proposal_is_studied_too():
    spec = {"kind": "unit_studied", "units": ["Gen_bus12"], "gen_index": 1}
    by_proposal = [{"assistant": "", "tool_calls": [{"name": "edit_proposal"}], "tool_results": [
        {"result": {"edits": [{"op": "scale", "element_type": "gen", "elements": [1], "factor": 0.5}]}}]}]
    assert rs.check(spec, by_proposal)
    assert rs.check(spec, _rec("", [{"renewables_scaled": {"factor": 0.5, "units": ["Gen_bus12"]}}]))
    other = [{"assistant": "", "tool_calls": [{"name": "edit_proposal"}], "tool_results": [
        {"result": {"edits": [{"op": "scale", "element_type": "gen", "elements": [0, 1], "factor": 0.5}]}}]}]
    assert not rs.check(spec, other)


@pytest.mark.parametrize("task, right, wrong", [
    ("names_cigre_wind", "WKA 7 is the only wind turbine; Wind Park West is a PV unit despite its name.",
     "WKA 7 and Wind Park West are the wind turbines."),
    ("names_cigre_storage", "Battery 1 (Bus 5) and Battery 2 (Bus 10). Battery Storage 3 is a fuel cell, not storage.",
     "Battery 1, Battery 2 and Battery Storage 3 store energy."),
    ("names_ieee30_wind", "The network does not state any unit's source; Offshore Wind 1 suggests wind only by its name.",
     "Offshore Wind 1 is the wind unit."),
    ("switches_loops", "S2 and S3 close loops within feeder Line 1-2; S1 is a tie between two feeders.",
     "S1, S2 and S3 close loops within the feeder."),
    ("switches_breakers", "Switch 6 and Switch 7 are breakers on the transformers Trafo 0-1 and Trafo 0-12.",
     "No switch is a breaker."),
    ("switches_line148", "S1, on Line 14-8 at Bus 8, decides it; it is normally open, so the line carries no power.",
     "Line 14-8 is in service; its loading is 0.04 %."),
    ("data_cigre_wind", "WKA 7's output is measured directly: it is the only unit at Bus 7. PV 9's output is a share "
                        "of Bus 9's measured production.",
     "Both WKA 7 and PV 9 are measured individually."),
    ("data_cigre_battery", "No: Battery 1 is held at its set-point; the data carry no storage.",
     "Yes, Battery 1's output is measured every 15 minutes."),
    ("data_cigre_load_split", "No: Load R1's demand is a share of Bus 1's measured consumption, split with Load CI1.",
     "Yes, Load R1 is metered."),
    ("data_ieee30_solved", "No: Gen_bus12's reactive power is solved by the power flow (it holds its voltage).",
     "Yes, Gen_bus12's reactive power comes from the measurements."),
    ("data_ieee30_solved", "No. Gen_bus12's reactive power is modeled rather than taken from a measurement.",
     "Yes, it is in the data."),
])
def test_the_new_tasks_pass_a_right_answer_and_fail_a_wrong_one(task, right, wrong):
    item = rs.TASKS[task]
    assert all(rs.evaluate(item.task, [{"assistant": right}]).values()), task
    assert not all(rs.evaluate(item.task, [{"assistant": wrong}]).values()), task
