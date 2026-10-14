"""
Starting scenarios: Resume (the last upload) and the two IEEE 14 demos.

Storage — `systems/`, `data_files/`, the "last uploaded" records and the saved
choice — is redirected to a temporary directory, so nothing here touches the
developer's own uploads.
"""

import io

import pandapower as pp
import pandapower.networks as pn
import pytest

pytest.importorskip("fastapi.testclient")

import main_backend as mb  # noqa: E402
import scenarios as scn  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402


def _patch_storage(monkeypatch, tmp_path):
    systems, data = tmp_path / "systems", tmp_path / "data_files"
    systems.mkdir(exist_ok=True)
    data.mkdir(exist_ok=True)
    monkeypatch.setenv("GRID_PROFILE", "pglib_case14")
    monkeypatch.setattr(mb, "SYSTEMS_DIR", str(systems))
    monkeypatch.setattr(mb, "DATA_FILES_DIR", str(data))
    monkeypatch.setattr(mb, "_LAST_UPLOAD_SENTINEL", str(systems / "last_uploaded.json"))
    monkeypatch.setattr(mb, "_LAST_TIMESERIES_SENTINEL", str(data / "last_uploaded_timeseries.json"))
    monkeypatch.setattr(mb, "_LAST_FORECAST_SENTINEL", str(data / "last_uploaded_forecast.json"))
    monkeypatch.setattr(mb, "_PROPOSALS_FILE", str(systems / "proposals.json"))
    monkeypatch.setattr(mb, "_ASSUMPTIONS_FILE", str(systems / "assumptions.json"))
    monkeypatch.setattr(mb, "_SCENARIO_FILE", str(tmp_path / "scenario.json"))
    return systems, data


@pytest.fixture
def client(tmp_path, monkeypatch):
    systems, data = _patch_storage(monkeypatch, tmp_path)
    saved = dict(mb.app_data)
    with TestClient(mb.app) as c:
        c.storage = (systems, data)
        yield c
    mb.app_data.clear()
    mb.app_data.update(saved)


def _upload_case9(client):
    r = client.post("/api/network/upload",
                    files={"file": ("grid9.json", io.BytesIO(pp.to_json(pn.case9()).encode()), "application/json")})
    assert r.status_code == 200, r.text
    return r


def _option(payload, name):
    return next(o for o in payload["options"] if o["name"] == name)


class TestChoosing:
    def test_a_fresh_start_offers_the_demos_and_no_resume(self, client):
        payload = client.get("/api/scenario").json()
        assert payload["current"] == "ieee14"
        assert [o["name"] for o in payload["options"]] == ["resume", "ieee14", "ieee14_reactive"]
        assert _option(payload, "resume")["available"] is False
        assert "Nothing uploaded yet" in _option(payload, "resume")["description"]

    def test_resume_without_uploads_is_refused(self, client):
        r = client.post("/api/scenario", json={"name": "resume"})
        assert r.status_code == 400 and "Nothing to resume" in r.json()["detail"]

    def test_an_unknown_scenario_is_refused(self, client):
        assert client.post("/api/scenario", json={"name": "ieee118"}).status_code == 400

    def test_the_reactive_demo_carries_measured_reactive_power_with_gaps(self, client):
        r = client.post("/api/scenario", json={"name": "ieee14_reactive"})
        assert r.status_code == 200, r.text
        assert r.json()["current"] == "ieee14_reactive"
        quality = client.get("/api/data/quality").json()["measurements"]
        assert quality["reactive"] == {"consumption": True, "production": True}
        codes = {f["code"]: f for f in quality["findings"]}
        assert codes["reactive_consumption_coverage"]["level"] == "warning"     # the gaps, on purpose
        assert 0.95 < codes["reactive_consumption_coverage"]["coverage"] < 1.0
        assert "leading_load" in codes
        assert client.get("/api/assumptions").json()["reactive_measured"]["measurements"]["consumption"] is True

    def test_the_active_only_demo_has_no_reactive_measurements(self, client):
        client.post("/api/scenario", json={"name": "ieee14_reactive"})
        assert client.post("/api/scenario", json={"name": "ieee14"}).status_code == 200
        quality = client.get("/api/data/quality").json()["measurements"]
        assert quality["reactive"] == {"consumption": False, "production": False}
        assert not any(f["code"] == "active_interpolated" for f in quality["findings"])

    def test_the_demo_data_are_the_same_every_time(self, client):
        week = mb.app_data["measurements"]                      # the demo week, as loaded
        first, again = scn.demo_reactive_csv(week), scn.demo_reactive_csv(week)
        assert first == again
        assert first.splitlines()[0].decode().endswith("consumption_mvar,production_mvar")


class TestResume:
    def test_an_upload_makes_resume_current_and_a_demo_leaves_it_intact(self, client):
        systems, _ = client.storage
        _upload_case9(client)
        payload = client.get("/api/scenario").json()
        assert payload["current"] == "resume"
        assert "grid9" in _option(payload, "resume")["description"]
        sentinel = (systems / "last_uploaded.json").read_bytes()

        assert client.post("/api/scenario", json={"name": "ieee14_reactive"}).status_code == 200
        assert len(mb.app_data["net"].bus) == 14
        assert (systems / "last_uploaded.json").read_bytes() == sentinel   # the demo never touches it

        assert client.post("/api/scenario", json={"name": "resume"}).status_code == 200
        assert len(mb.app_data["net"].bus) == 9

    def test_each_network_keeps_its_own_proposals_and_assumptions(self, client):
        _upload_case9(client)
        client.post("/api/assumptions", json={"load_power_factor": 0.9})
        client.post("/api/scenario", json={"name": "ieee14"})
        assert client.get("/api/assumptions").json()["load_power_factor"]["value"] == 0.95
        client.post("/api/assumptions", json={"load_power_factor": 0.85})
        client.post("/api/scenario", json={"name": "resume"})
        assert client.get("/api/assumptions").json()["load_power_factor"]["value"] == 0.9
        client.post("/api/scenario", json={"name": "ieee14"})
        assert client.get("/api/assumptions").json()["load_power_factor"]["value"] == 0.85

    def test_reset_forgets_the_choice(self, client, tmp_path):
        _upload_case9(client)
        client.post("/api/scenario", json={"name": "ieee14_reactive"})
        assert (tmp_path / "scenario.json").exists()
        client.post("/api/network/reset_active")
        assert not (tmp_path / "scenario.json").exists()
        assert client.get("/api/scenario").json()["current"] == "ieee14"


class TestRestart:
    def test_the_choice_is_remembered_across_a_restart(self, tmp_path, monkeypatch):
        _patch_storage(monkeypatch, tmp_path)
        saved = dict(mb.app_data)
        try:
            with TestClient(mb.app) as c:
                _upload_case9(c)
                assert c.post("/api/scenario", json={"name": "ieee14_reactive"}).status_code == 200
            with TestClient(mb.app) as c:
                payload = c.get("/api/scenario").json()
                assert payload["current"] == "ieee14_reactive"
                assert len(mb.app_data["net"].bus) == 14
                assert c.get("/api/data/quality").json()["measurements"]["reactive"]["consumption"] is True
                assert c.post("/api/scenario", json={"name": "resume"}).status_code == 200
            with TestClient(mb.app) as c:
                assert c.get("/api/scenario").json()["current"] == "resume"
                assert len(mb.app_data["net"].bus) == 9
        finally:
            mb.app_data.clear()
            mb.app_data.update(saved)


class TestTemplateAndDataKey:
    def test_the_template_lists_the_network_and_uploads_back(self, client):
        r = client.get("/api/data/template")
        assert r.status_code == 200 and r.headers["content-type"].startswith("text/csv")
        assert "attachment" in r.headers["content-disposition"]
        lines = r.text.splitlines()
        assert lines[0] == "timestamp,substation_name,consumption_mw,production_mw,consumption_mvar,production_mvar"
        names = {line.split(",")[1] for line in lines[1:]}
        assert names == set(mb.app_data["measurements"][mb.app_data["timestamps"][0]]["substation_name"].astype(str))
        assert len(lines) - 1 == 4 * len(names)                     # one hour of 15-minute ticks
        assert all(line.endswith(",,") for line in lines[1:])        # no reactive data loaded: empty columns
        up = client.post("/api/data/upload", data={"kind": "measurements", "overwrite": "true"},
                         files={"file": ("template.csv", io.BytesIO(r.content), "text/csv")})
        assert up.status_code == 200, up.text

    def test_the_template_carries_reactive_values_when_loaded(self, client):
        client.post("/api/scenario", json={"name": "ieee14_reactive"})
        lines = client.get("/api/data/template?hours=2").text.splitlines()
        assert len(lines) - 1 == 8 * (len({x.split(",")[1] for x in lines[1:]}))
        assert any(not line.endswith(",,") for line in lines[1:])

    def test_the_data_key_follows_the_data_not_the_assumptions(self, client):
        key = client.get("/api/assumptions").json()["data_key"]
        client.post("/api/assumptions", json={"load_power_factor": 0.9})
        assert client.get("/api/assumptions").json()["data_key"] == key
        client.post("/api/scenario", json={"name": "ieee14_reactive"})
        reactive = client.get("/api/assumptions").json()["data_key"]
        assert reactive != key
        client.post("/api/scenario", json={"name": "ieee14"})
        assert client.get("/api/assumptions").json()["data_key"] == key      # same data, same key


def test_a_window_with_reactive_gaps_says_the_power_factor_filled_them(client):
    client.post("/api/scenario", json={"name": "ieee14_reactive"})
    worst = client.post("/api/rsa/worst_case", json={}).json()
    block = worst["load_power_factor"]["measured_reactive_power"]
    assert isinstance(block["share_of_load"], list) and block["share_of_load"][0] < 0.9999
    assert "applies only to the rest" in block["note"]


def test_a_network_with_numbered_buses_gives_text_names(client):
    """case33bw names its buses 1, 2, …: the agent's prompt joined them as text
    and every turn failed before reaching the model."""
    import sys
    r = client.post("/api/network/upload",
                    files={"file": ("case33bw.json", io.BytesIO(pp.to_json(pn.case33bw()).encode()), "application/json")})
    assert r.status_code == 200, r.text
    grid = client.get("/api/grid_constants").json()
    assert grid["substation_names"] and all(isinstance(n, str) for n in grid["substation_names"])
    assert client.post("/api/grid/rsa", json={}).json()["converged"] is True
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "llm_agent"))
    from agent.system_prompt import build_system_prompt
    gc = dict(grid, substation_names=[1, 2, 3])                     # tolerant of numbers, too
    assert "1, 2, 3" in build_system_prompt(gc)


def test_buses_and_substations_are_reported_apart(client):
    grid = client.get("/api/grid_constants").json()
    assert grid["n_buses"] == 14 and grid["n_substations"] == 11 and grid["n_lines"] == 17 and grid["n_trafos"] == 3
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "llm_agent"))
    from agent.system_prompt import build_system_prompt
    prompt = build_system_prompt(grid)
    assert "14 buses, 17 power lines" in prompt and "11 of the 14 buses are substations" in prompt
