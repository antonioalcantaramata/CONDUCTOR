"""
Switching as a proposal (sandbox `set_switch`): open or close a switch — by its name, index or mRID — kept as a CIM
change (its `Switch.open`), studied through it.
"""

import contextlib
import io

import pytest
from fastapi.testclient import TestClient

import main_backend as mb

pytest.importorskip("lxml")


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("sw")
    systems, data = tmp / "systems", tmp / "data_files"
    systems.mkdir()
    data.mkdir()
    mp = pytest.MonkeyPatch()
    mp.setenv("GRID_PROFILE", "cigre_mv")
    mp.setattr(mb, "SYSTEMS_DIR", str(systems))
    mp.setattr(mb, "DATA_FILES_DIR", str(data))
    for name, folder in (("_LAST_UPLOAD_SENTINEL", systems), ("_LAST_TIMESERIES_SENTINEL", data),
                         ("_LAST_FORECAST_SENTINEL", data), ("_PROPOSALS_FILE", systems),
                         ("_ASSUMPTIONS_FILE", systems), ("_SCENARIO_FILE", tmp)):
        mp.setattr(mb, name, str(folder / f"{name}.json"))
    saved = dict(mb.app_data)
    with contextlib.redirect_stdout(io.StringIO()), TestClient(mb.app) as c:
        yield c
    mb.app_data.clear()
    mb.app_data.update(saved)
    mp.undo()


def _tie():
    sw = mb.app_data["net"].switch
    i = int(sw.index[~sw["closed"].astype(bool)][0])
    return i, str(sw.at[i, "name"])


def test_closing_a_tie_is_a_proposal_kept_as_its_cim_change(client):
    i, name = _tie()
    r = client.post("/api/proposals/edit", json={"proposal": "tie", "edits": [
        {"op": "set_switch", "switch": name, "closed": True}]})
    assert r.status_code == 200, r.text
    view = r.json()
    assert view["edits"] == [{"op": "set_switch", "switch": i, "closed": True}]
    assert view["summary"]["edits"][0].startswith(f"Close {name}") and view["summary"]["operated_switches"] == 1
    assert view["cim"]["translation"]["direct"] == 1
    entry = mb._materialised("tie")
    assert entry["via"] == "cim" and bool(entry["net"].switch.at[i, "closed"]) is True
    assert bool(mb.app_data["net"].switch.at[i, "closed"]) is False          # the base is untouched
    xml = client.get("/api/network/cim/proposal", params={"network": "tie"})
    assert b"Switch.open" in xml.content
    rsa = client.post("/api/grid/rsa", json={"network": "tie"})
    assert rsa.status_code == 200 and rsa.json().get("converged") is not False


def test_a_switch_can_be_named_by_its_mrid_and_opened(client):
    import semantics as sm
    net = mb.app_data["net"]
    closed = int(net.switch.index[net.switch["closed"].astype(bool)][0])
    mrid = sm.row_mrids(net, "switch", str(mb.app_data["loaded_fingerprint"]))[closed]
    r = client.post("/api/proposals/edit", json={"proposal": "open1", "edits": [
        {"op": "set_switch", "switch": mrid, "closed": False}]})
    assert r.status_code == 200, r.text
    assert r.json()["edits"][0]["switch"] == closed and r.json()["summary"]["edits"][0].startswith("Open ")


def test_a_switch_edit_needs_a_state(client):
    _i, name = _tie()
    r = client.post("/api/proposals/edit", json={"proposal": "bad", "edits": [{"op": "set_switch", "switch": name}]})
    assert r.status_code == 400 and "closed must be true or false" in r.json()["detail"]
