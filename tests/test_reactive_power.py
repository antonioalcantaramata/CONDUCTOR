"""
The load power factor: how measured loads get their reactive power.

The measurements carry active power only, so every Mvar of demand comes from
an assumed power factor. It used to be a hardcoded 0.95; it is now the default
of a session setting the operator sees (and changes) in the sidebar, the agent
can change on request, and any study can override for itself — and every
result that assigned a tick says which value it used.
"""

import json
import math

import pandapower.networks as pn
import pandas as pd
import pytest

import load_gen_assignment as lg

TAN = {pf: math.tan(math.acos(pf)) for pf in (0.95, 0.9, 0.92, 0.85)}


def _q(snapshot: dict) -> float:
    return sum(l["Q_mvar"] for l in snapshot["loads"])


def _p(snapshot: dict) -> float:
    return sum(l["P_mw"] for l in snapshot["loads"])


def test_the_assignment_takes_the_power_factor():
    net = pn.case14()
    net.bus["name"] = None                      # as the backend loads it: Bus_<index>
    meas = pd.DataFrame({"substation_name": [f"Bus_{b}" for b in net.load.bus],
                         "consumption": [10.0] * len(net.load), "production": [0.0] * len(net.load)})
    subs = list(meas.substation_name)
    lg.assign_load_values_from_measurements(net, meas, subs)
    assert (net.load.q_mvar / net.load.p_mw).round(6).unique().tolist() == [round(TAN[0.95], 6)]
    lg.assign_load_values_from_measurements(net, meas, subs, power_factor=0.9)
    assert (net.load.q_mvar / net.load.p_mw).round(6).unique().tolist() == [round(TAN[0.9], 6)]
    assert lg.DEFAULT_LOAD_POWER_FACTOR == 0.95 and lg.MEASURED_GENERATION_POWER_FACTOR == 0.99


class TestThroughTheBackend:
    @pytest.fixture(autouse=True)
    def _default_after(self, backend_client):
        yield
        backend_client.post("/api/assumptions", json={"load_power_factor": None})

    def test_the_default_is_stated_in_every_result(self, backend_client):
        assert backend_client.get("/api/assumptions").json()["load_power_factor"]["value"] == 0.95
        snap = backend_client.post("/api/network/snapshot", json={}).json()
        assert snap["load_power_factor"]["value"] == 0.95 and snap["load_power_factor"]["source"] == "default"
        assert _q(snap) == pytest.approx(_p(snap) * TAN[0.95], abs=0.01)
        rsa = backend_client.post("/api/grid/rsa", json={}).json()
        assert rsa["load_power_factor"]["source"] == "default"

    def test_one_study_at_another_power_factor_leaves_the_setting_alone(self, backend_client):
        snap = backend_client.post("/api/network/snapshot", json={"load_power_factor": 0.9}).json()
        assert snap["load_power_factor"] == {**snap["load_power_factor"], "value": 0.9, "source": "this study"}
        assert _q(snap) == pytest.approx(_p(snap) * TAN[0.9], abs=0.01)
        assert backend_client.get("/api/assumptions").json()["load_power_factor"]["source"] == "default"
        again = backend_client.post("/api/network/snapshot", json={}).json()
        assert again["load_power_factor"]["value"] == 0.95

    def test_window_studies_and_comparisons_use_it_too(self, backend_client):
        worst = backend_client.post("/api/rsa/worst_case", json={"load_power_factor": 0.85, "step_size": 48})
        assert worst.status_code == 200 and worst.json()["load_power_factor"]["value"] == 0.85
        import main_backend as mb
        backend_client.post("/api/proposals/edit", json={"proposal": "dc", "edits": [
            {"op": "add_load", "name": "DC", "bus": "Bus 9", "p_mw": 10}]})
        cmp = backend_client.post("/api/proposals/compare", json={"proposal": "dc", "load_power_factor": 0.9}).json()
        assert cmp["load_power_factor"]["value"] == 0.9
        mb._discard_proposals()

    def test_the_session_setting_applies_to_later_studies(self, backend_client):
        out = backend_client.post("/api/assumptions", json={"load_power_factor": 0.92}).json()
        assert out["load_power_factor"]["source"] == "session setting"
        assert out["previous"] == {"value": 0.95, "source": "default"}
        snap = backend_client.post("/api/network/snapshot", json={}).json()
        assert snap["load_power_factor"]["value"] == 0.92 and snap["load_power_factor"]["source"] == "session setting"
        assert _q(snap) == pytest.approx(_p(snap) * TAN[0.92], abs=0.01)
        own = backend_client.post("/api/network/snapshot", json={"load_power_factor": 0.9}).json()
        assert own["load_power_factor"]["source"] == "this study"          # a study's own value wins
        reset = backend_client.post("/api/assumptions", json={"load_power_factor": None}).json()
        assert reset["load_power_factor"]["source"] == "default"

    def test_the_nominal_point_uses_none(self, backend_client):
        out = backend_client.post("/api/network/snapshot", json={"data_source": "nominal",
                                                                "load_power_factor": 0.8}).json()
        assert "load_power_factor" not in out

    @pytest.mark.parametrize("value", [1.2, 0.3, -1])
    def test_impossible_values_are_refused(self, backend_client, value):
        assert backend_client.post("/api/grid/rsa", json={"load_power_factor": value}).status_code == 422
        assert backend_client.post("/api/assumptions", json={"load_power_factor": value}).status_code == 422


def test_the_setting_survives_a_restart_for_the_same_network_only(tmp_path, monkeypatch):
    pytest.importorskip("fastapi.testclient")
    import main_backend as mb
    from fastapi.testclient import TestClient
    monkeypatch.setenv("GRID_PROFILE", "pglib_case14")
    monkeypatch.setattr(mb, "SYSTEMS_DIR", str(tmp_path / "systems"))
    monkeypatch.setattr(mb, "DATA_FILES_DIR", str(tmp_path / "data_files"))
    for name in ("_LAST_UPLOAD_SENTINEL", "_LAST_TIMESERIES_SENTINEL", "_LAST_FORECAST_SENTINEL",
                 "_PROPOSALS_FILE"):
        monkeypatch.setattr(mb, name, str(tmp_path / f"{name}.json"))
    store = tmp_path / "systems" / "assumptions.json"
    monkeypatch.setattr(mb, "_ASSUMPTIONS_FILE", str(store))
    monkeypatch.setattr(mb, "_SCENARIO_FILE", str(tmp_path / "scenario.json"))
    saved = dict(mb.app_data)
    try:
        with TestClient(mb.app) as c:
            c.post("/api/assumptions", json={"load_power_factor": 0.9})
        (entry,) = json.loads(store.read_text())["by_fingerprint"].values()
        assert entry["load_power_factor"] == 0.9
        with TestClient(mb.app) as c:
            assert c.get("/api/assumptions").json()["load_power_factor"] == {
                **c.get("/api/assumptions").json()["load_power_factor"], "value": 0.9, "source": "session setting"}
        data = json.loads(store.read_text())
        store.write_text(json.dumps({**data, "by_fingerprint": {"another network": entry}}))
        with TestClient(mb.app) as c:
            assert c.get("/api/assumptions").json()["load_power_factor"]["source"] == "default"
            c.post("/api/assumptions", json={"load_power_factor": 0.9})
            c.post("/api/assumptions", json={"load_power_factor": None})
        # The default is not stored; the other network's entry is kept.
        assert list(json.loads(store.read_text())["by_fingerprint"]) == ["another network"]
    finally:
        mb.app_data.clear()
        mb.app_data.update(saved)


def test_the_robust_opf_takes_its_power_factor_like_the_others(backend_client, monkeypatch):
    """It used to hardcode 0.95 while every other OPF took `opf_min_power_factor`."""
    import main_backend as mb
    seen = []

    def stop(*args, **kwargs):
        seen.append(kwargs.get("min_power_factor"))
        raise RuntimeError("stop here")

    monkeypatch.setattr(mb.fe, "optimization_model_base", stop)
    monkeypatch.setattr(mb.fe, "scenario_optimization_model_base", stop)
    for method in ("heuristic", "scenario"):
        try:
            backend_client.post("/api/flexibility/robust", json={
                "robust_method": method, "opf_min_power_factor": 0.8, "n_samples": 20,
                "n_scenarios": 5, "validation_samples": 20})
        except RuntimeError:
            pass                                    # stopped at the OPF, as intended
    assert seen and set(seen) == {0.8}


def test_the_tool_changes_the_setting_and_refuses_ambiguity(monkeypatch):
    from llm_agent.agent import tools
    sent = []
    monkeypatch.setattr(tools, "_post", lambda path, body: sent.append((path, body)) or {"ok": True})
    tools.set_load_power_factor(value=0.92)
    tools.set_load_power_factor(reset=True)
    assert sent == [("/api/assumptions", {"load_power_factor": 0.92}),
                    ("/api/assumptions", {"load_power_factor": None})]
    assert "error" in tools.set_load_power_factor()
    assert "error" in tools.set_load_power_factor(value=0.9, reset=True)


def test_every_study_tool_offers_it_and_the_setting_tool_is_separate():
    from llm_agent.agent import tool_schemas as ts
    from llm_agent.agent.system_prompt import build_system_prompt
    decl = {fd.name: fd for t in ts.TOOLS for fd in t.function_declarations}
    for name, fd in decl.items():
        props = fd.parameters.properties if fd.parameters else {}
        studies = "data_source" in props or name == "scan_rsa_over_time"
        assert ("load_power_factor" in props) == studies, name
    setting = decl["set_load_power_factor"].parameters.properties
    assert set(setting) == {"value", "reset"}
    assert "opf_min_power_factor" in decl["optimize_robust_flexibility"].parameters.properties
    assert "set_load_power_factor" in ts.TOOL_DISPATCH
    prompt = build_system_prompt({"name": "g"})
    assert "## Reactive power — the load power factor" in prompt and "opf_min_power_factor" in prompt


def test_results_name_the_units_modelled_differently(backend_client):
    """With a changed generator model, every result says so — and the UI's
    caches are keyed to the assumptions, so the map redraws."""
    key0 = backend_client.get("/api/assumptions").json()["key"]
    assert "generator_model" not in backend_client.post("/api/grid/rsa", json={}).json()
    backend_client.post("/api/generators", json={"changes": [{"unit": "Gen_bus1", "control": "voltage"}]})
    try:
        out = backend_client.post("/api/grid/rsa", json={}).json()
        assert out["generator_model"]["changed_units"] == {"Gen_bus1": {"control": "voltage"}}
        assert backend_client.get("/api/assumptions").json()["key"] != key0
    finally:
        backend_client.post("/api/generators", json={"reset": True})
    assert backend_client.get("/api/assumptions").json()["key"] == key0
