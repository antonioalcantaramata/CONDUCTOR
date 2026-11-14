"""
The answer checks and the CIM layer after the CIM layer paper's results:

1. a fact read only from a name, said as a fact, is flagged (the paper: 18 such
   answers on IEEE 30, none flagged);
2. a unit's source is read in more of the ways answers write it — a list whose
   first name the check never read, a bulleted list under a source, a table row,
   the answer to a question about a source (the paper: 18 of 30 contradicted
   answers missed);
3. a figure computed in one step from figures the answer or the tools give — a
   sum at a bus, a duration from intervals, a count, an apparent power — is
   derived, not invented (the paper: most warnings were correct figures);
4. the tie that could restore supply after an outage is checked with a power flow.

Measured on the paper's recorded answers with `evaluation.injection` (read only).
"""

import pandapower.networks as pn
import pytest

import main_backend as mb
import semantics as sm
from llm_agent.agent import entities as en
from llm_agent.agent import provenance as pv
from llm_agent.agent import semantics_view as sv


def _record(units: dict, question: str = "Which units are there?") -> dict:
    """A turn's record as the loop keeps it: the units named, and every unit by source."""
    by_source: dict = {}
    for name, u in units.items():
        if u["basis"] != "unknown":
            by_source.setdefault(u["source"], []).append(name)
    return {"user": question, "network_facts": {"base": {"units": units, "units_by_source": by_source}}}


CIGRE_RENAMED = {"Wind Park West": {"source": "solar", "basis": "stated"},
                 "WKA 7": {"source": "wind", "basis": "stated"},
                 "Solar Farm North": {"source": "thermal", "basis": "stated"},
                 "PV 3": {"source": "solar", "basis": "stated"},
                 "Battery Storage 3": {"source": "thermal", "basis": "stated"}}


def _kinds(answer: str, record: dict) -> list[str]:
    return [i.kind for i in en.check_answer(answer, record)]


# --- 1. Read only from a name -----------------------------------------------------------------------------------

def test_a_fact_only_its_name_gives_is_flagged_unless_the_answer_says_so():
    rec = _record({"Solar Plant A": {"source": "solar", "basis": "named"}})
    for said in ("**Solar Plant A** is the solar unit.", "Solar Plant A at Bus_1 is the solar unit."):
        issues = en.check_answer(said, rec)
        assert [i.kind for i in issues] == [en.UNSUPPORTED] and "only its name says so" in issues[0].note
    assert en.check_answer("Going by its name, Solar Plant A is presumably the solar unit.", rec) == []
    stated = _record({"Solar Plant A": {"source": "solar", "basis": "stated"}})
    assert en.check_answer("Solar Plant A is the solar unit.", stated) == []


# --- 2. Where a source is said ----------------------------------------------------------------------------------

def test_the_first_unit_of_a_list_is_read():
    rec = _record(CIGRE_RENAMED, "Which units in this network are wind turbines?")
    issues = en.check_answer("The wind units are **Wind Park West** and **WKA 7**.", rec)
    assert any("Wind Park West among the wind units" in i.note for i in issues)
    assert not any("WKA 7" in i.note for i in issues)


def test_a_bulleted_list_under_a_source_is_one_list():
    rec = _record(CIGRE_RENAMED)
    answer = "There are **3 solar units**:\n\n- **PV 3**\n- **Wind Park West**\n- **Solar Farm North**\n"
    notes = [i.note for i in en.check_answer(answer, rec)]
    assert any("Solar Farm North among the solar units" in n for n in notes)
    assert not any("PV 3" in n for n in notes)


def test_bullets_under_a_head_that_lists_no_one_source_are_their_own_claims():
    rec = _record(CIGRE_RENAMED)
    answer = "Tripping it disconnects 3 units:\n- 1 solar unit: PV 3\n- 1 wind unit: WKA 7\n"
    assert en.check_answer(answer, rec) == []


def test_a_table_row_with_a_source_column():
    rec = _record(CIGRE_RENAMED)
    table = "| Unit | Type | Bus |\n|---|---|---|\n| WKA 7 | wind | Bus 7 |\n| Solar Farm North | solar | Bus 9 |\n"
    issues = en.check_answer(table, rec)
    assert [i.note for i in issues] == ["gives Solar Farm North as solar, but it is thermal (stated)"]


def test_the_answer_to_a_question_about_a_source_is_read():
    rec = _record(CIGRE_RENAMED, "Where is the storage in this network?")
    assert _kinds("The **Battery Storage 3** unit is located at **Bus 5**.", rec) == [en.CONTRADICTED]
    # Named after the subject, or with its real source: no claim that it is storage.
    assert _kinds("The active power of Battery Storage 3 is measured.", rec) == []
    assert _kinds("Battery Storage 3 at Bus 5 is a CHP unit (thermal), not storage.", rec) == []


def test_two_problems_in_one_sentence_are_both_kept():
    rec = _record(CIGRE_RENAMED)
    rec["network_facts"]["base"]["names_not_in_network"] = ["PV 61"]
    notes = [i.note for i in en.check_answer("The solar units are PV 61 and Solar Farm North.", rec)]
    assert any("PV 61" in n for n in notes) and any("Solar Farm North" in n for n in notes)


# --- 3. One step from figures given -----------------------------------------------------------------------------

def _status(answer: str, value: float, sources=(), record: dict | None = None) -> str:
    report = pv.check_turn({"assistant": answer, **record}) if record else pv.check_answer(answer, list(sources))
    return next(v.status for v in report.verdicts if abs(v.claim.value - value) < 1e-9)


def test_the_load_at_a_bus_is_the_sum_of_its_loads_when_the_bus_is_named():
    record = {"tool_results": [{"name": "get_current_conditions", "result": {"loads": [
        {"bus": "Bus 3", "P_mw": 0.2143}, {"bus": "Bus 3", "P_mw": 0.0153}, {"bus": "Bus 4", "P_mw": 0.19}]}}]}
    assert _status("Supply is lost at Bus 3 (0.23 MW) and Bus 4 (0.19 MW).", 0.23, record=record) == pv.DERIVED
    assert _status("Supply is lost at Bus 4 (0.23 MW).", 0.23, record=record) == pv.UNGROUNDED


def test_a_duration_from_a_count_of_intervals():
    sources = [(10.0, "risk.n_violating"), (5.0, "risk.n_near")]
    answer = "It exceeded the limit for 10 quarter-hour intervals (2.50 hours), and came close in 5 intervals."
    assert _status(answer, 2.5, sources) == pv.DERIVED
    assert _status(answer.replace("2.50", "3.00"), 3.0, sources) == pv.UNGROUNDED


def test_the_other_n_and_counts_together():
    sources = [(41.0, "grid.n_lines"), (29.0, "risk.violating"), (27.0, "risk.near"), (672.0, "risk.ticks")]
    assert _status("It screened all 41 lines; plans cover these eight outages, and the other 33 lines got none.",
                   33, sources) == pv.DERIVED
    assert _status("It screened all 41 lines; plans cover these eight outages, and the other 31 lines got none.",
                   31, sources) == pv.UNGROUNDED
    answer = ("It violated in 29 of 672 intervals and came close in 27 intervals. So it was either violating or "
              "near the limit in 56 intervals (8.3%).")
    assert _status(answer, 56, sources) == pv.DERIVED
    assert _status(answer, 8.3, sources) == pv.DERIVED


def test_an_apparent_power_from_active_power_and_power_factor():
    sources = [(0.8, "edit_proposal.edits[0].p_mw"), (0.63, "trafo.sn_mva")]
    answer = "The data centre draws 0.8 MW at a power factor of 0.95, about 0.84 MVA, above the 0.63 MVA rating."
    assert _status(answer, 0.84, sources) == pv.DERIVED
    assert _status(answer.replace("0.84", "0.91"), 0.91, sources) == pv.UNGROUNDED


def test_a_change_written_as_its_two_ends():
    sources = [(29.4112, "base.import"), (29.8731, "proposal.import")]
    assert _status("Import rises from 29.41 MW to 29.87 MW (about 0.46 MW).", 0.46, sources) == pv.DERIVED


def test_values_offered_in_a_question_are_proposals():
    answer = "What should “comfortably below” mean for L30? I suggest 80%; you can choose 85% or 90% instead."
    assert pv.extract_claims(pv.answer_to_check(answer, [{"name": "run_rsa"}])) == []
    stated = "L30 is at 80%; you can choose to act later."            # no question: still checked
    assert [c.value for c in pv.extract_claims(pv.answer_to_check(stated, [{"name": "run_rsa"}]))] == [80.0]


# --- 4. The tie, checked --------------------------------------------------------------------------------------------

@pytest.fixture
def cigre(monkeypatch):
    net = pn.create_cigre_network_mv(with_der="all")
    monkeypatch.setitem(mb.app_data, "measurements", {})
    monkeypatch.setitem(mb.app_data, "timestamps", [])
    monkeypatch.setitem(mb.app_data, "default_vm_lower", 0.95)
    monkeypatch.setitem(mb.app_data, "default_vm_upper", 1.05)
    monkeypatch.setitem(mb.app_data, "grid_profile", {"max_loading_pct": 100.0})
    return net


def test_the_tie_that_restores_supply_is_checked_with_a_power_flow(cigre):
    layer = sm.build(cigre, network_id="cigre")
    line = int(cigre.line.index[cigre.line.name == "Line 2-3"][0])
    out = sm.around(cigre, layer, "line", line, check_tie=mb._tie_checker(cigre))["without_it"]
    [tie] = out["ties_checked"]
    assert tie["converged"] and tie["buses_restored"] == tie["buses_cut_off"] == 9 and "Line 14-8" in tie["tie"]
    # At the case's own (full) load, the restored feeder sags below 0.95 p.u.: said, with each bus.
    assert tie["within_limits"] == (not tie["violations"])
    assert all(float(v.split(" at ")[1].split()[0]) < 0.95 for v in tie["violations"])
    assert tie["lowest_voltage"]["pu"] == pytest.approx(min(float(v.split(" at ")[1].split()[0])
                                                           for v in tie["violations"]), abs=6e-4)
    assert sv.tie_line(tie).startswith("Closing **Line 14-8")


def test_at_a_light_load_the_same_tie_is_within_limits(cigre):
    cigre.load["p_mw"] *= 0.5
    cigre.load["q_mvar"] *= 0.5
    layer = sm.build(cigre, network_id="cigre")
    line = int(cigre.line.index[cigre.line.name == "Line 2-3"][0])
    [tie] = sm.around(cigre, layer, "line", line, check_tie=mb._tie_checker(cigre))["without_it"]["ties_checked"]
    assert tie["within_limits"] and "within limits" in sv.tie_line(tie)


def test_a_tie_that_overloads_says_so(cigre):
    cigre.load["p_mw"] *= 6
    layer = sm.build(cigre, network_id="cigre")
    line = int(cigre.line.index[cigre.line.name == "Line 2-3"][0])
    [tie] = sm.around(cigre, layer, "line", line, check_tie=mb._tie_checker(cigre))["without_it"]["ties_checked"]
    assert not tie.get("within_limits", False)
    assert "outside limits" in sv.tie_line(tie) or "does not converge" in sv.tie_line(tie)


def test_without_a_checker_the_ties_are_topology_only(cigre):
    layer = sm.build(cigre, network_id="cigre")
    line = int(cigre.line.index[cigre.line.name == "Line 2-3"][0])
    out = sm.around(cigre, layer, "line", line)["without_it"]
    assert "ties_checked" not in out and out["open_ties_that_could_restore"]
