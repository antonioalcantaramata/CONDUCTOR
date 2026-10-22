"""
What an answer changed in the session's settings (agent/settings_change.py),
and Undo (/api/settings/revert): it puts back only what that answer changed,
and refuses rather than overwrite a change made since.
"""

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "llm_agent"))

from agent import settings_change as sc  # noqa: E402

BASE = {"load_power_factor": None, "generator_model": {}, "unit_costs": {}, "element_names": {},
        "unit_names_now": {}, "names_as_loaded": {}, "default_load_power_factor": 0.95}


class TestLines:
    def test_nothing_changed(self):
        assert sc.lines(BASE, dict(BASE)) == [] and sc.lines(None, BASE) == []

    def test_power_factor(self):
        assert sc.lines(BASE, {**BASE, "load_power_factor": 0.9}) == ["load power factor 0.95 (default) → 0.9"]
        assert sc.lines({**BASE, "load_power_factor": 0.9}, BASE) == ["load power factor 0.9 → 0.95 (default)"]

    def test_generator_model_by_its_name_now(self):
        after = {**BASE, "generator_model": {"Gen_bus1": {"control": "voltage", "vm_pu": 1.02}},
                 "unit_names_now": {"Gen_bus1": "Plant A"}}
        assert sc.lines(BASE, after) == ["Plant A: voltage control, set-point 1.02 p.u."]
        assert sc.lines(after, BASE) == ["Plant A back to as loaded"]

    def test_only_what_differs_is_said(self):
        before = {**BASE, "generator_model": {"G": {"control": "voltage", "vm_pu": 1.0}}}
        after = {**BASE, "generator_model": {"G": {"control": "voltage", "vm_pu": 1.03, "dispatchable": False}}}
        assert sc.lines(before, after) == ["G: set-point 1.03 p.u., not dispatchable"]

    def test_costs(self):
        after = {**BASE, "unit_costs": {"Gen_bus1": {"up_eur_per_mwh": 90.0}}}
        assert sc.lines(BASE, after) == ["Gen_bus1 costs: up 90 €/MWh"]
        assert sc.lines(after, BASE) == ["Gen_bus1 costs: up back to default"]

    def test_units_given_the_same_change_share_a_line(self):
        zero = {"q_eur_per_mvarh": 0.0}
        after = {**BASE, "unit_costs": {"Gen_bus1": zero, "Gen_bus2": zero, "Gen_bus5": zero,
                                        "Gen_bus7": {"up_eur_per_mwh": 90.0}}}
        assert sc.lines(BASE, after) == ["Gen_bus1, Gen_bus2 and Gen_bus5 costs: reactive 0 €/Mvarh",
                                         "Gen_bus7 costs: up 90 €/MWh"]

    def test_renames(self):
        after = {**BASE, "element_names": {"bus": {"3": "North"}}, "names_as_loaded": {"bus:3": "Bus_3"}}
        assert sc.lines(BASE, after) == ["bus Bus_3 renamed North"]
        assert sc.lines(after, BASE) == ["North back to Bus_3"]

    def test_summary(self):
        assert sc.summary(["a", "b"]) == "a · b"
        assert sc.summary(list("abcdef")) == "a · b · c · d · and 2 more"


@pytest.fixture
def client(tmp_path, monkeypatch):
    pytest.importorskip("fastapi.testclient")
    import main_backend as mb
    from fastapi.testclient import TestClient
    systems, data = tmp_path / "systems", tmp_path / "data_files"
    systems.mkdir()
    data.mkdir()
    monkeypatch.setenv("GRID_PROFILE", "pglib_case14")
    monkeypatch.setattr(mb, "SYSTEMS_DIR", str(systems))
    monkeypatch.setattr(mb, "DATA_FILES_DIR", str(data))
    for name in ("_LAST_UPLOAD_SENTINEL", "_LAST_TIMESERIES_SENTINEL", "_LAST_FORECAST_SENTINEL",
                 "_PROPOSALS_FILE", "_ASSUMPTIONS_FILE", "_SCENARIO_FILE"):
        monkeypatch.setattr(mb, name, str(tmp_path / f"{name}.json"))
    saved = dict(mb.app_data)
    with TestClient(mb.app) as c:
        yield c
    mb.app_data.clear()
    mb.app_data.update(saved)


def _ok(r):
    assert r.status_code == 200, r.text
    return r.json()


def _an_answer_that_changes_everything(client):
    """What one answer might do: every kind of setting at once."""
    before = _ok(client.get("/api/settings"))
    _ok(client.post("/api/assumptions", json={"load_power_factor": 0.9}))
    _ok(client.post("/api/generators", json={"changes": [{"unit": "Gen_bus1", "dispatchable": False}]}))
    _ok(client.post("/api/costs", json={"changes": [{"unit": "Gen_bus1", "up_eur_per_mwh": 90}]}))
    _ok(client.post("/api/names", json={"changes": [{"element_type": "bus", "element": 3, "name": "North"}]}))
    after = _ok(client.get("/api/settings"))
    return before, after


def test_undo_puts_back_exactly_what_the_answer_changed(client):
    before, after = _an_answer_that_changes_everything(client)
    assert len(sc.lines(before, after)) == 4
    out = _ok(client.post("/api/settings/revert", json={"before": before, "after": after}))
    assert len(out["reverted"]) == 4
    now = _ok(client.get("/api/settings"))
    for key in ("load_power_factor", "generator_model", "unit_costs", "element_names"):
        assert now[key] == before[key]


def test_a_later_unrelated_change_survives_undo(client):
    before, after = _an_answer_that_changes_everything(client)
    _ok(client.post("/api/costs", json={"changes": [{"unit": "External grid", "cost_eur_per_mwh": 70}]}))
    _ok(client.post("/api/settings/revert", json={"before": before, "after": after}))
    assert _ok(client.get("/api/settings"))["unit_costs"] == {"External grid": {"cost_eur_per_mwh": 70.0}}


def test_a_change_made_since_is_never_overwritten(client):
    before, after = _an_answer_that_changes_everything(client)
    _ok(client.post("/api/assumptions", json={"load_power_factor": 0.85}))        # changed again by hand
    r = client.post("/api/settings/revert", json={"before": before, "after": after})
    assert r.status_code == 409 and "load power factor" in r.text
    now = _ok(client.get("/api/settings"))
    assert now["load_power_factor"] == 0.85 and now["element_names"] == after["element_names"]   # nothing undone


def test_undoing_twice_does_nothing_the_second_time(client):
    before, after = _an_answer_that_changes_everything(client)
    _ok(client.post("/api/settings/revert", json={"before": before, "after": after}))
    assert _ok(client.post("/api/settings/revert", json={"before": before, "after": after}))["reverted"] == []


def test_the_report_says_it():
    from agent import report
    page = report.build([{"role": "assistant", "text": "Done.", "settings_change": {
        "lines": ["load power factor 0.95 (default) → 0.9"], "undone": True}}], {"title": "t"})
    assert "Undone afterwards" in page and "load power factor 0.95 (default) → 0.9" in page
