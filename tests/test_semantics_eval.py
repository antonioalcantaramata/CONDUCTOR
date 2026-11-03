"""The semantics ablation (evaluation/semantics.py): the switch, and grading — without model credit."""

import json

from evaluation import semantics as se
from evaluation import whatif


def test_the_switch_hides_the_layer_from_the_model_only(monkeypatch):
    from llm_agent.agent.tool_schemas import tools_for_model
    names = lambda: {fd.name for t in tools_for_model() for fd in t.function_declarations}  # noqa: E731
    monkeypatch.setenv("CONDUCTOR_ONTOLOGY", "on")
    assert {"describe_network", "declare_network_facts"} <= names()
    monkeypatch.setenv("CONDUCTOR_ONTOLOGY", "off")
    assert not {"describe_network", "declare_network_facts"} & names() and "locate_network_element" in names()


def test_mentions_all_takes_alternatives():
    rec = {"assistant": "The feeder is **L0**; power comes over Line 3-8 and Line 10-11."}
    spec = {"kind": "mentions_all", "all": [["Line 1-2", "L0"], "Line 3-8", ["Line 10-11", "L8"]]}
    assert whatif.check(spec, [rec])
    assert not whatif.check({"kind": "mentions_all", "all": ["Line 1-2"]}, [rec])


def test_a_folder_is_graded_with_checks_entities_calls_and_time(tmp_path):
    rec = {"turn": 1, "user": "Which units in this network are renewable?", "tool_calls": [{"name": "describe_network"}],
           "tool_results": [], "timing": {"total_s": 12.0},
           "assistant": "WKA 7 (wind) and PV 3 … PV 11 (solar) are renewable.", "network_facts": {}}
    with open(tmp_path / "cigre_renewables-01.jsonl", "w") as fh:
        fh.write(json.dumps({"kind": "semantics-on", "task": "cigre_renewables", "repeat": 1}) + "\n")
        fh.write(json.dumps(rec) + "\n")
    g = se.grade_folder(tmp_path)["cigre_renewables"]
    assert g["runs"] == 1 and g["calls"] == 1 and g["seconds"] == 12.0 and g["used_layer"] == 1
    assert list(g["checks"].values()) == [1]
    assert "Checks passed | 1/1 | 1/1" in se.compare([(tmp_path, tmp_path)])


def test_a_name_mentioned_only_to_reject_it_does_not_count():
    rec = {"assistant": "The lookup lists Line 14-8 as in service, so I can’t present it as a normally open tie."}
    spec = next(c for t in se.CIGRE_TASKS if t.id == "cigre_outage" for c in t.checks if "tie" in c["label"])
    assert not whatif.check(spec, [rec])
    assert whatif.check(spec, [{"assistant": "Closing the open tie Line 14-8 restores them."}])


def _ontology_run(tmp_path, arms):
    """A run folder as `python -m evaluation.ablation run ontology` writes it: one correct answer per arm."""
    rec = {"turn": 1, "user": "q", "tool_calls": [], "tool_results": [], "timing": {"total_s": 3.0},
           "assistant": "WKA 7, PV 3 and PV 11 are renewable.", "network_facts": {}}
    for arm in arms:
        d = tmp_path / "runs" / arm / "semantics-cigre"
        d.mkdir(parents=True)
        with open(d / "cigre_renewables-01.jsonl", "w") as fh:
            fh.write(json.dumps({"kind": "ablation", "arm": arm, "arm_hash": "x", "suite": "semantics-cigre",
                                 "task": "cigre_renewables", "repeat": 1}) + "\n")
            fh.write(json.dumps(rec) + "\n")
    (tmp_path / "manifest.json").write_text(json.dumps({"study": "ontology", "baseline": arms[0],
                                                        "arms": [{"name": a} for a in arms]}))


def test_the_ablation_analysis_reads_a_run_folder(tmp_path):
    from evaluation import ablation
    _ontology_run(tmp_path, ("on", "off"))
    text = ablation.analyse([tmp_path])
    assert "cigre" in text and "Checks passed | 1/1 | 1/1" in text
    assert (tmp_path / "analysis.md").read_text().startswith("# Ablation: ontology")


def test_none_hides_the_layer_from_the_agent_too(monkeypatch):
    from llm_agent.agent.config import ontology_enabled
    monkeypatch.setenv("CONDUCTOR_ONTOLOGY", "none")
    assert not ontology_enabled()


def test_an_arm_refuses_a_backend_in_the_wrong_mode(monkeypatch):
    import pytest
    for uses, arm, ok in ((True, "on", True), (True, "none", False), (False, "none", True), (False, "off", False)):
        monkeypatch.setattr(whatif, "_backend", lambda m, p, u=uses: {"studies_use_semantic_layer": u})
        if ok:
            se.check_backend_mode(arm)
        else:
            with pytest.raises(RuntimeError, match="CONDUCTOR_ONTOLOGY"):
                se.check_backend_mode(arm)


def test_three_arms_give_three_comparisons(tmp_path):
    from evaluation import ablation
    _ontology_run(tmp_path, ("on", "off", "none"))
    text = ablation._ontology_sections(tmp_path)
    assert len(__import__("re").findall(r"(?m)^## ", text)) == 3 and "on vs off" in text and "off vs none" in text \
        and "on vs none" in text
