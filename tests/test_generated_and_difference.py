"""
A test set generated from any network's CIM layer (`evaluation.generated`), and a proposal as a CIM
difference against the base (`cim_export.difference`, IEC 61970-552).
"""

import copy

import pandapower.networks as pn
import pytest

import cim_export as ce
import semantics as sm
from evaluation import generated as gen


@pytest.fixture(scope="module")
def cigre():
    net = pn.create_cigre_network_mv(with_der="all")
    return net, sm.build(net, network_id="cigre")


# --- The generated test set ---------------------------------------------------------------------------------

def test_tasks_come_from_what_the_layer_knows(cigre):
    _net, layer = cigre
    tasks = gen.generate(layer, {"Line 3-4": ["Load R4", "Load R5"], "Line 1-2": []}, per_kind=2, seed=1)
    kinds = {t.id.split("-", 1)[0] for t in tasks}
    assert kinds == {"unit_source", "units_of", "bus_voltage", "load_feeder", "outage"}   # CIGRE states every unit
    wind = next(t for t in tasks if t.id == "units_of-wind") if any(t.id == "units_of-wind" for t in tasks) else None
    if wind:
        assert wind.checks[0]["all"] == ["WKA 7"]
    outage = next(t for t in tasks if t.id.startswith("outage-"))
    assert outage.checks[0]["all"] == ["Load R4", "Load R5"]           # a line that cuts nothing off is not asked
    assert gen.generate(layer, None, per_kind=2, seed=1) == [t for t in gen.generate(layer, None, 2, 1)]  # seeded


def test_a_network_that_states_nothing_is_asked_what_it_does_not_say():
    net = pn.case30()
    tasks = gen.generate(sm.build(net, network_id="x"), per_kind=3)
    unknown = [t for t in tasks if t.id.startswith("unit_unknown-")]
    assert len(unknown) == 3 and "does not say" in unknown[0].checks[0]["any"]
    assert not any(t.id.startswith(("unit_source-", "units_of-")) for t in tasks)


def test_tasks_round_trip_and_grade(tmp_path, cigre):
    _net, layer = cigre
    tasks = gen.generate(layer, per_kind=1, seed=0)
    data = gen.as_json(tasks, {"name": "CIGRE"})
    assert gen.from_json(data) == tasks
    task = next(t for t in tasks if t.id.startswith("bus_voltage-"))
    import json
    (tmp_path / "a-01.jsonl").write_text(json.dumps({"task": task.id}) + "\n"
                                         + json.dumps({"assistant": "It is at **20 kV**."}) + "\n")
    out = gen.grade(tmp_path, tasks)
    assert out["tasks"][task.id] == {"runs": 1, "passed": 1} and out["share"] == 1.0


# --- A proposal as a CIM difference ---------------------------------------------------------------------------

def _statements(semantics):
    return {(s, cls, frozenset(st)) for s, (cls, st) in ce._statements(semantics, "cigre").items()}


def test_the_difference_applied_to_the_base_gives_the_proposal(cigre):
    net, base = cigre
    changed = copy.deepcopy(net)
    line = int(changed.line.index[changed.line.name == "Line 6-7"][0])
    changed.line.at[line, "in_service"] = False                       # a line out: its facts change
    changed.load.at[changed.load.index[0], "name"] = "Load R1 north"   # a renamed load
    proposal = sm.build(changed, network_id="cigre")
    diff = ce.difference(base, proposal, "cigre")
    assert diff["changed"] and not diff["added"]
    before, after = ce._statements(base, "cigre"), ce._statements(proposal, "cigre")
    rebuilt = {s: set(st) for s, (_c, st) in before.items()}
    for sid, cls, stmts in diff["added"]:
        rebuilt[sid] = set(stmts)
    for sid, _cls, _stmts in diff["removed"]:
        rebuilt.pop(sid, None)
    for sid, _cls, plus, minus in diff["changed"]:
        rebuilt[sid] = (rebuilt[sid] - minus) | plus
    assert rebuilt == {s: st for s, (_c, st) in after.items()}
    summary = ce.difference_summary(diff)
    changes = {c["element"]: c["changes"] for c in summary["changed"]}
    assert changes["Load R1 north"] == ["IdentifiedObject.name: Load R1 → Load R1 north"]
    assert changes["Line 6-7"] == ["Equipment.inService: true → false"]      # named, though its name did not change


def test_the_difference_model_is_rdf(cigre):
    rdflib = pytest.importorskip("rdflib")
    net, base = cigre
    changed = copy.deepcopy(net)
    changed.load.at[changed.load.index[0], "name"] = "Load R1 north"
    text = ce.to_difference_rdf_xml(base, sm.build(changed, network_id="cigre"), "cigre", description="test",
                                    created="2026-11-14T00:00:00+00:00")
    g = rdflib.Graph().parse(data=text, format="xml")
    dm = rdflib.Namespace(ce.DM_NS)
    assert (None, rdflib.RDF.type, dm.DifferenceModel) in g
    assert "Load R1 north" in text and "dm:forwardDifferences" in text and "dm:reverseDifferences" in text
