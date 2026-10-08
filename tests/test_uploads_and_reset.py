"""
Uploads and resets leave the stored state consistent (code review batch D,
findings 3.3 and 3.4).

Storage — `systems/`, `data_files/` and the "last uploaded" records — is
redirected to a temporary directory, so nothing here touches the developer's
own uploads.
"""

import io

import pandapower as pp
import pandapower.networks as pn
import pytest

pytest.importorskip("fastapi.testclient")

import main_backend as mb  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture
def client(tmp_path, monkeypatch):
    systems, data = tmp_path / "systems", tmp_path / "data_files"
    systems.mkdir()
    data.mkdir()
    monkeypatch.setenv("GRID_PROFILE", "pglib_case14")
    monkeypatch.setattr(mb, "SYSTEMS_DIR", str(systems))
    monkeypatch.setattr(mb, "DATA_FILES_DIR", str(data))
    monkeypatch.setattr(mb, "_LAST_UPLOAD_SENTINEL", str(systems / "last_uploaded.json"))
    monkeypatch.setattr(mb, "_LAST_TIMESERIES_SENTINEL", str(data / "last_uploaded_timeseries.json"))
    monkeypatch.setattr(mb, "_LAST_FORECAST_SENTINEL", str(data / "last_uploaded_forecast.json"))
    saved = dict(mb.app_data)
    with TestClient(mb.app) as c:
        c.storage = (systems, data)
        yield c
    mb.app_data.clear()
    mb.app_data.update(saved)


def _network_json(net) -> bytes:
    return pp.to_json(net).encode("utf-8")


def _upload_network(client, name: str, content: bytes):
    return client.post("/api/network/upload",
                       files={"file": (name, io.BytesIO(content), "application/json")})


def _csv(client) -> bytes:
    """A valid timeseries for the loaded network, two ticks."""
    names = [f"Bus_{i}" for i in mb.app_data["net"].load.bus]
    rows = ["timestamp,substation_name,consumption_mw,production_mw"]
    for ts in ("2001-01-01 00:00:00", "2001-01-01 00:15:00"):
        rows += [f"{ts},{n},5.0,0.0" for n in names]
    return ("\n".join(rows) + "\n").encode()


class TestUploads:
    def test_a_failed_upload_does_not_replace_the_stored_network(self, client):
        systems, _ = client.storage
        assert _upload_network(client, "grid.json", _network_json(pn.case9())).status_code == 200
        stored = sorted(p for p in systems.iterdir() if p.suffix == ".json" and "grid" in p.name)
        before = stored[0].read_bytes()

        assert _upload_network(client, "grid.json", b"{not json").status_code == 400
        assert stored[0].read_bytes() == before

    def test_measurements_and_forecasts_with_one_name_do_not_collide(self, client):
        _, data = client.storage
        content = _csv(client)
        for kind in ("measurements", "forecasts"):
            r = client.post("/api/data/upload", data={"kind": kind, "overwrite": "true"},
                            files={"file": ("data.csv", io.BytesIO(content), "text/csv")})
            assert r.status_code == 200, r.text
        stored = sorted(p.name for p in data.iterdir() if p.name.endswith("data.csv"))
        assert stored == ["uploaded_forecasts__data.csv", "uploaded_measurements__data.csv"]

    def test_an_unparseable_csv_is_not_written(self, client):
        _, data = client.storage
        r = client.post("/api/data/upload", data={"kind": "measurements"},
                        files={"file": ("bad.csv", io.BytesIO(b"nonsense"), "text/csv")})
        assert r.status_code == 400
        assert not any("bad" in p.name for p in data.iterdir())


class TestReset:
    def test_reset_forgets_the_previous_network_and_its_data(self, client):
        systems, data = client.storage
        assert _upload_network(client, "case9.json", _network_json(pn.case9())).status_code == 200
        assert (systems / "last_uploaded.json").exists()

        r = client.post("/api/network/reset_active", json={})
        assert r.status_code == 200 and r.json()["n_buses"] == 14
        assert not (systems / "last_uploaded.json").exists()
        assert not (data / "last_uploaded_timeseries.json").exists()

        # The default network's own data again: load follows the time series.
        ts = mb.app_data["timestamps"]
        loads = {client.post("/api/grid/rsa", json={"timestamp": t}).json()["total_load_mw"]
                 for t in (ts[0], ts[30], ts[60])}
        assert len(loads) == 3
