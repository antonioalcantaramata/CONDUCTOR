"""Error injection (evaluation/injection.py), and the gaps in the answer checks it found — without model credit."""

import csv
import json
import random

import pandapower.networks as pn
import pytest

import network_loader as nl
import semantics as sm
from evaluation import injection as ij
from llm_agent.agent import entities as en
from llm_agent.agent.provenance import check_turn


@pytest.fixture(scope="module")
def ieee30_facts():
    net = nl.load_network(nl.load_profile("pglib_case30"))
    net, _, _ = nl.gen_to_sgen(net)
    return sm.compact(sm.build(net))


@pytest.fixture(scope="module")
def cigre_facts():
    return sm.compact(sm.build(pn.create_cigre_network_mv(with_der="all")))


def _record(facts, answer, user="", results=(), grid="CIGRE MV Benchmark (with DER)"):
    """A turn record as the loop keeps it, with every element the answer names in its facts."""
    names = [*facts["buses"], *facts["loads"], *facts["units"], *facts["branches"],
             *(k.split(" | ")[0] for k in facts["branches"])]

    def named(k):
        return k.split(" | ")[0] in answer or k in answer
    kept = {"per_unit_case": facts["per_unit_case"], "voltage_levels_kv": facts["voltage_levels_kv"],
            "external_grids": facts["external_grids"],
            "units_by_source": {s: sorted(u for u, f in facts["units"].items() if f["source"] == s
                                          and f["basis"] != "unknown")
                                for s in {f["source"] for f in facts["units"].values()} if s != "unknown"},
            **{part: {k: v for k, v in facts[part].items() if named(k)} for part in ("buses", "branches", "units",
                                                                                    "loads")},
            "names_not_in_network": en.names_not_in_network(answer, names)}
    return {"user": user, "assistant": answer, "network_facts": {"base": kept}, "grid": {"name": grid},
            "tool_calls": [{"name": n, "args": {}} for n, _ in results],
            "tool_results": [{"name": n, "result": r} for n, r in results]}


def _issues(rec):
    return [(i.kind, i.note) for i in en.check_answer(rec["assistant"], rec)]


# ---------------------------------------------------------------------------
# The gaps the injection found, each fixed and pinned here
# ---------------------------------------------------------------------------

def test_names_with_underscores_are_checked(ieee30_facts):
    # Cleaning the sentence stripped the underscore: "Gen_bus12" became "Genbus12" and no claim about any IEEE 30
    # unit, bus or invented name was ever checked.
    rec = _record(ieee30_facts, "Gen_bus12 is a wind turbine.", grid="PGLib IEEE 30")
    assert [k for k, _ in _issues(rec)] == [en.UNSUPPORTED]


def test_a_sentence_ends_where_markdown_closes_after_its_stop(cigre_facts):
    # "…0.942 p.u.** I can't confirm…" was one sentence; the next one's denial held back the first one's claim.
    rec = _record(cigre_facts, "The lowest voltage is at **Bus 64: 0.942 p.u.** I can't confirm what it doesn't say.")
    assert any("Bus 64" in note for _k, note in _issues(rec))


def test_a_branch_is_recognised_by_its_short_name_on_either_side(cigre_facts):
    for answer in ("The transformer Line 1-2 is the most loaded branch.", "Line 1-2 is a transformer.",
                   "Line 1-2 (transformer) is the most loaded branch."):
        assert [k for k, _ in _issues(_record(cigre_facts, answer))] == [en.CONTRADICTED], answer
    assert _issues(_record(cigre_facts, "Line 1-2 is the most loaded line.")) == []


def test_an_invented_name_is_not_excused_by_a_hedge_only_by_a_denial(cigre_facts):
    hedged = "If Line 2-3 trips, the loads at Bus 64 through Bus 11 are left without supply."
    assert any("Bus 64" in note for _k, note in _issues(_record(cigre_facts, hedged)))
    for denial in ("There is no Bus 64 in this network.", "Bus 64 does not exist."):
        assert _issues(_record(cigre_facts, denial)) == [], denial


def test_a_figure_from_another_violation_is_caught_with_spaced_names():
    # Blanking "Bus 11" so that "11" is not read as a figure also hid it from the misattribution check.
    viol = [{"element": "Bus 11", "violation_type": "bus_vm_pu", "value": 0.9421, "violated": True},
            {"element": "Bus 8", "violation_type": "bus_vm_pu", "value": 0.9443, "violated": True}]
    rec = {"assistant": "At 18:00, **Bus 11** had the lowest voltage: **0.944 p.u.**",
           "tool_calls": [{"name": "run_rsa", "args": {}}],
           "tool_results": [{"name": "run_rsa", "result": {"secure": False, "violations": viol}}]}
    (verdict,) = check_turn(rec).misattributed
    assert verdict.subject == "Bus 11" and verdict.belongs_to == "Bus 8"
    rec["assistant"] = "At 18:00, **Bus 11** had the lowest voltage: **0.942 p.u.**"
    assert check_turn(rec).misattributed == []


# ---------------------------------------------------------------------------
# False alarms on the agent's own answers (the CIM layer paper's runs, labelled by hand), each fixed and pinned here — with the
# error the same check must still catch beside it
# ---------------------------------------------------------------------------

def _ungrounded(rec):
    return [v.claim.value for v in check_turn(rec).verdicts if v.status == "ungrounded"]


def test_a_branch_name_ending_a_sentence_is_not_a_figure(cigre_facts):
    # "Line 14-8." was matched as "Line 14", and "-8" read as a figure no result held.
    for answer in ("S1 controls the tie on Line 14-8. It is normally open.", "Use Line 8-9, then Line 2-3."):
        assert _ungrounded(_cigre_turn(cigre_facts, answer)) == [], answer
    assert _ungrounded(_cigre_turn(cigre_facts, "On proposal gen_bus4_minus10_2000-01-06-20, the scan covered "
                                                "January 1–7, 2000, with Q = P tan(\\cos^{-1} 0.95).")) == []


def test_a_margin_stated_without_its_subtraction_is_derived_and_a_changed_figure_is_not(cigre_facts):
    rsa = [("run_rsa", {"secure": False, "violations": _VIOL, "thresholds_used": {"vm_lower_pu": 0.95}})]
    assert _ungrounded(_cigre_turn(cigre_facts, "Bus 3 is at 0.947 p.u., 0.003 p.u. below the limit.", rsa)) == []
    # The same figure with no step in the words, or a step from bare integers (names, counts), is still unexplained.
    assert _ungrounded(_cigre_turn(cigre_facts, "Bus 3 is at 0.947 p.u. and Bus 4 at 0.003 p.u.")) == [0.003]
    assert _ungrounded(_cigre_turn(cigre_facts, "PV 3 and PV 4 connect at 7 kV, more than before.")) == [7.0]


def test_a_follow_up_is_checked_against_the_turn_before(cigre_facts):
    rec = _cigre_turn(cigre_facts, "The set-point was raised from 1.000 to 1.009 p.u.", results=[("noop", {})])
    assert _ungrounded(rec) == [1.0, 1.009]
    rec["earlier_results"] = [{"name": "test_prevention", "result": {"setpoint_new": 1.009, "setpoint_from": 1.0}}]
    assert _ungrounded(rec) == []


def test_figures_proposed_in_a_question_are_not_checked(cigre_facts):
    rec = _cigre_turn(cigre_facts, "What should it mean? I suggest 80 %; alternatives are 70 % or 90 %.", results=[])
    rec["tool_calls"] = []
    assert _ungrounded(rec) == []


def test_a_value_said_not_to_be_measured_is_not_called_measured(cigre_facts):
    rec = _cigre_turn(cigre_facts, "Load R5")
    rec["network_facts"]["base"]["loads"]["Load R5"].update(
        {"q_kind": "derived", "q_from": "the power factor 0.97 (profile)", "p_kind": "allocated", "p_from": "a share"})
    for answer in ("Load R5's active demand is allocated—estimated as a share of the measured bus total.",
                   "Load R5's reactive demand is derived from the power factor, rather than measured."):
        assert en.check_answer(answer, rec) == [], answer
    assert [i.kind for i in en.check_answer("The reactive power of Load R5 is measured.", rec)] == [en.CONTRADICTED]


def test_a_study_setting_is_not_a_claim_about_units(ieee30_facts):
    rec = _record(ieee30_facts, "This uses 5 % load uncertainty; renewable-output uncertainty was disabled.",
                  grid="PGLib IEEE 30")
    assert _issues(rec) == []
    rec = _record(ieee30_facts, "The wind farms supply most of the load.", grid="PGLib IEEE 30")
    assert [k for k, _ in _issues(rec)] == [en.UNSUPPORTED]


def test_a_voltage_level_near_a_bus_is_not_the_bus_voltage(cigre_facts):
    assert _issues(_record(cigre_facts, "Bus 0 feeds the 20 kV network through Trafo 0-1 and Trafo 0-12.")) == []
    assert [k for k, _ in _issues(_record(cigre_facts, "Bus 3 is at 110 kV."))] == [en.CONTRADICTED]


def test_a_route_from_the_external_grid_names_every_branch_on_it(cigre_facts):
    rec = _record(cigre_facts, "The route from the external grid is Trafo 0-1 → Line 1-2 → Line 2-3 → Line 3-8.")
    assert _issues(rec) == []


def test_a_secure_verdict_about_another_state_is_not_read_against_the_study():
    from llm_agent.agent.characterisation import check_study_claims
    rsa = {"secure": False, "total_violations": 1, "timestamp": "2000-01-06 20:00:00",
           "violations": [{"element": "Bus_29", "violation_type": "bus_vm_pu", "value": 0.943}]}
    rec = {"tool_calls": [{"name": "run_rsa", "args": {}}], "tool_results": [{"name": "run_rsa", "result": rsa}]}
    for answer in ("At 20:00 Bus_29 is at 0.943 p.u. Compared with 19:00, when it was just within limits, demand rose.",
                   "Bus_29 is below its limit. No violations are resolved or newly introduced.",
                   "Bus_29 is at 0.943 p.u. Thermal loading is within limits."):
        assert check_study_claims(answer, rec) == [], answer
    assert [c.field for c in check_study_claims("Bus_29 is at 0.943 p.u. Overall, the grid is secure.", rec)] == [
        "secure"]


def test_a_follow_up_calling_the_grid_secure_is_read_against_the_turn_before():
    # The evidence check now counts the earlier assessment as behind a follow-up's verdict; the verdict must then
    # agree with it.
    from llm_agent.agent.characterisation import check_study_claims
    rsa = {"secure": False, "total_violations": 1, "timestamp": "2000-01-06 20:00:00"}
    rec = {"tool_calls": [], "tool_results": [], "earlier_results": [{"name": "run_rsa", "result": rsa}]}
    assert [c.field for c in check_study_claims("The grid is secure.", rec)] == ["secure"]
    assert check_study_claims("No violations are added or cleared.", rec) == []


def test_a_new_paragraph_ends_the_subject():
    viol = [{"element": "Bus 11", "violation_type": "bus_vm_pu", "value": 0.9421, "violated": True},
            {"element": "Bus 8", "violation_type": "bus_vm_pu", "value": 0.9443, "violated": True}]
    rec = {"assistant": "The assessment also found Bus 11 at 0.942 p.u.\n\nThe line's 0.944 p.u. end is Bus 8's.",
           "tool_calls": [{"name": "run_rsa", "args": {}}],
           "tool_results": [{"name": "run_rsa", "result": {"secure": False, "violations": viol}}]}
    assert check_turn(rec).misattributed == []


def test_a_structural_or_bounded_impossibility_is_not_overstated():
    from llm_agent.agent import evidence
    bound = {"optimality": {"lower_bound_eur_per_h": 22.48}, "status": "options"}
    rec = {"tool_calls": [{"name": "optimize_flexibility", "args": {}}],
           "tool_results": [{"name": "optimize_flexibility", "result": bound}]}
    for answer in ("The outage leaves Bus_25 without supply, so no dispatch can keep its load supplied.",
                   "A cheaper solution may exist, but no feasible dispatch can be more than €4.00/h cheaper.",
                   "That is a warning sign, but it is not, by itself, proof of the cause."):
        assert evidence.build({**rec, "assistant": answer}).overstated == (), answer
    plain = {"tool_calls": [], "tool_results": [], "assistant": "The grid is secure."}
    assert len(evidence.build(plain).overstated) == 1
    plain["earlier_results"] = [{"name": "run_rsa", "result": {"secure": True, "total_violations": 0}}]
    assert evidence.build(plain).overstated == ()


# ---------------------------------------------------------------------------
# The harness
# ---------------------------------------------------------------------------

_VIOL = [{"element": "Bus 3", "violation_type": "bus_vm_pu", "value": 0.9471, "violated": True},
         {"element": "Bus 4", "violation_type": "bus_vm_pu", "value": 0.9449, "violated": True}]


def _cigre_turn(cigre_facts, answer, results=None):
    return _record(cigre_facts, answer, results=results or [("run_rsa", {"secure": False, "total_violations": 2,
                                                                         "violations": _VIOL})])


def test_each_injection_is_caught_and_each_true_statement_is_not(cigre_facts):
    rec = _cigre_turn(cigre_facts, "Bus 3 is at 0.947 p.u. and Bus 4 at 0.945 p.u. PV 3 at Bus 3 feeds the "
                                   "feeder through Line 2-3.")
    rows = {}
    for inject in ij.INJECTORS:
        for inj in inject(rec, random.Random(0)):
            rows[inj.kind] = (inj, ij.caught(inj)[0], ij.undetectable(inj))
    for kind in ("figure_swapped", "insecure_called_secure", "unit_source_contradicted", "bus_voltage_wrong",
                 "line_called_transformer_before", "line_called_transformer_after", "invented_element"):
        assert kind in rows and rows[kind][1], kind
    for kind in ("unit_source_true", "bus_voltage_true"):
        assert kind in rows and not rows[kind][1], kind
    # The injected sentence goes where the model's text ends.
    inj = rows["unit_source_contradicted"][0]
    assert inj.answer.startswith(rec["assistant"]) and inj.target["sentence"] in inj.answer


def test_a_swap_is_caught_when_the_true_figure_comes_first_in_its_sentence(cigre_facts):
    # Swapping Bus 4's 0.945 for Bus 3's 0.947 leaves "Bus 3 at 0.947 …, Bus 4 at 0.947": the first 0.947 is
    # right, and the scorer once judged it instead of the injected one.
    rec = _cigre_turn(cigre_facts, "The violations are Bus 3 at 0.947 p.u. and Bus 4 at 0.945 p.u.")
    inj = ij.Injection("figure_swapped", "L1", "provenance", False,
                       "The violations are Bus 3 at 0.947 p.u. and Bus 4 at 0.947 p.u.",
                       {"value": 0.947, "decimals": 3, "about": "Bus 4", "belongs_to": "Bus 3"}, rec)
    assert ij.caught(inj) == (True, "misattributed")
    # The same value already wrong in the original answer is not credited to the injection.
    wrong = _cigre_turn(cigre_facts, "Bus 4 is at 0.947 p.u.")
    assert ij.caught(ij.Injection("figure_swapped", "L1", "provenance", False, wrong["assistant"],
                                  {"value": 0.947, "decimals": 3}, wrong))[0] is False


def test_an_injection_goes_before_conductors_own_notes():
    answer = "The grid is insecure.\n\n**Reading check.** CONDUCTOR re-ran the studies under readings the question left open."
    out = ij._append(answer, "X.")
    assert out.index("X.") < out.index("**Reading check.**")


def test_a_changed_figure_another_result_holds_is_undetectable_not_missed(cigre_facts):
    rec = _cigre_turn(cigre_facts, "Bus 3 is at 0.947 p.u.")
    inj = ij.Injection("figure_changed", "L1", "provenance", False, "Bus 3 is at 0.945 p.u.",
                       {"value": 0.945, "decimals": 3, "was": 0.947}, rec)
    assert ij.undetectable(inj)


def test_the_operators_own_units_are_not_injected_about(ieee30_facts):
    rec = _record(ieee30_facts, "Understood: Gen_bus12 is the wind farm.", user="Gen_bus12 is a wind farm.",
                  grid="PGLib IEEE 30")
    assert ij._named_units(rec) == {}


def test_the_command_writes_its_tables(tmp_path, cigre_facts):
    rec = _cigre_turn(cigre_facts, "Bus 3 is at 0.947 p.u. and Bus 4 at 0.945 p.u.")
    run = tmp_path / "runs" / "full" / "x"
    run.mkdir(parents=True)
    (run / "t-01.jsonl").write_text(json.dumps({"kind": "ablation"}) + "\n" + json.dumps(rec) + "\n")
    assert ij.main([str(tmp_path), "--seed", "1"]) == 0
    md = (tmp_path / "injection.md").read_text()
    assert "figure_swapped" in md and "answers as written" in md
    data = json.loads((tmp_path / "injection.json").read_text())
    assert data["records"] == 1 and data["summary"] and "baseline" in data
    assert ij.run(ij.load_records([tmp_path]), seed=1) == ij.run(ij.load_records([tmp_path]), seed=1)


def test_findings_on_the_answers_as_written_go_to_a_sheet_that_keeps_its_labels(tmp_path, cigre_facts):
    rec = _cigre_turn(cigre_facts, "The solar units are PV 3 and CHP diesel 1.")
    run = tmp_path / "runs" / "full" / "x"
    run.mkdir(parents=True)
    (run / "t-01.jsonl").write_text(json.dumps({"kind": "ablation"}) + "\n" + json.dumps(rec) + "\n")
    out = tmp_path / "out"
    assert ij.main([str(tmp_path), "--out", str(out)]) == 0
    sheet = out / "findings-to-label.csv"
    rows = list(csv.DictReader(open(sheet, encoding="utf-8")))
    assert rows and rows[0]["code"] == "entity_contradicted" and rows[0]["source"].endswith("t-01.jsonl")
    assert rows[0]["label"] == ""
    rows[0]["label"] = "error"
    with open(sheet, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    assert ij.main([str(tmp_path), "--out", str(out)]) == 0
    assert "error" in sheet.read_text()


def test_a_derived_value_called_measured_is_caught_and_a_measured_one_is_not(cigre_facts):
    rec = _cigre_turn(cigre_facts, "Load R5 draws 0.4 MW and PV 3 produces 0.02 MW.")
    unit = rec["network_facts"]["base"]["units"]["PV 3"]
    load = rec["network_facts"]["base"]["loads"]["Load R5"]
    load.update({"q_kind": "derived", "q_from": "the power factor 0.97 (profile)", "p_kind": "measured"})
    unit.update({"p_kind": "measured"})
    kinds = {}
    for seed in range(6):
        for inj in ij.inject_measured(rec, random.Random(seed)):
            kinds.setdefault(inj.kind, ij.caught(inj)[0])
    assert kinds == {"value_called_measured": True, "value_measured_true": False}
