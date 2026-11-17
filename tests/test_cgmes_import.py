"""
CGMES upload (backend/network_loader.read_cgmes): a CIM model read with
pandapower's converter keeps its own identifiers and substations, which the
semantic layer then states rather than infers; a model missing its boundary
set, or a zip of something else, is refused with what to add.
"""

import io
import pathlib
import zipfile

import pandapower as pp
import pytest
from fastapi.testclient import TestClient

import main_backend as mb
import network_loader as nl
import semantics as sm

pytest.importorskip("lxml")
FIXTURE = pathlib.Path(__file__).parent / "fixtures" / "cgmes" / "pandapower_example_multivoltage.zip"


@pytest.fixture(scope="module")
def net():
    return nl.read_cgmes(str(FIXTURE))


def test_the_model_is_read_and_solves(net):
    assert len(net.bus) == 111 and len(net.line) == 26 and len(net.ext_grid) == 1
    pp.runpp(net)
    assert net.converged


def test_the_layer_states_what_the_model_states(net):
    s = sm.build(net, network_id="x")
    assert s["mrid_basis"] == sm.STATED
    assert s["buses"][0]["mrid"] == str(net.bus.at[net.bus.index[0], "origin_id"]).lstrip("_")   # the model's mRID
    assert s["substations"]["basis"] == sm.STATED and "Substation" in s["substations"]["from"]
    units = {u["name"]: u for u in s["units"]}
    assert units["Wind Park"]["source"]["value"] == "wind"
    assert sum(x["normal_open"]["value"] for x in s["switches"]) == 5


def test_a_zip_of_something_else_is_refused(tmp_path):
    bad = tmp_path / "notes.zip"
    with zipfile.ZipFile(bad, "w") as z:
        z.writestr("readme.txt", "hello")
    with pytest.raises(nl.CgmesError, match="no CIM/CGMES profiles"):
        nl.read_cgmes(str(bad))


def test_a_model_missing_its_boundary_says_what_to_add(monkeypatch, net):
    import copy
    broken = copy.deepcopy(net)
    broken.line["to_bus"] = broken.line["to_bus"].astype(float)
    broken.line.at[broken.line.index[0], "to_bus"] = float("nan")
    monkeypatch.setattr("pandapower.converter.cim.cim2pp.from_cim.from_cim", lambda **kw: broken)
    with pytest.raises(nl.CgmesError, match="boundary set"):
        nl.read_cgmes(str(FIXTURE))
    # The layer itself reports such a branch instead of failing on it.
    s = sm.build(broken)
    assert s["unresolved_branches"] and len(s["branches"]) == len(broken.line) + len(broken.trafo) - 1


def test_a_reference_unit_becomes_the_supply_point(monkeypatch, net):
    import copy
    no_ext = copy.deepcopy(net)
    no_ext.ext_grid = no_ext.ext_grid.iloc[0:0]
    no_ext.gen["slack"] = True
    monkeypatch.setattr("pandapower.converter.cim.cim2pp.from_cim.from_cim", lambda **kw: no_ext)
    out = nl.read_cgmes(str(FIXTURE))
    assert len(out.ext_grid) == 1 and "(reference)" in out.ext_grid.name.iloc[0]
    assert out["_import_notes"] and "supply point" in out["_import_notes"][0]


@pytest.fixture
def client(tmp_path, monkeypatch):
    systems, data = tmp_path / "systems", tmp_path / "data_files"
    systems.mkdir()
    data.mkdir()
    monkeypatch.setenv("GRID_PROFILE", "pglib_case30")
    monkeypatch.setattr(mb, "SYSTEMS_DIR", str(systems))
    monkeypatch.setattr(mb, "DATA_FILES_DIR", str(data))
    for name, folder in (("_LAST_UPLOAD_SENTINEL", systems), ("_LAST_TIMESERIES_SENTINEL", data),
                         ("_LAST_FORECAST_SENTINEL", data), ("_PROPOSALS_FILE", systems),
                         ("_ASSUMPTIONS_FILE", systems), ("_SCENARIO_FILE", tmp_path)):
        monkeypatch.setattr(mb, name, str(folder / f"{name}.json"))
    saved = dict(mb.app_data)
    with TestClient(mb.app) as c:
        yield c
    mb.app_data.clear()
    mb.app_data.update(saved)


def test_a_cgmes_upload_is_studied_like_any_network(client):
    r = client.post("/api/network/upload",
                    files={"file": ("example.zip", io.BytesIO(FIXTURE.read_bytes()), "application/zip")})
    assert r.status_code == 200, r.text
    summary = client.get("/api/network/semantics", params={"about": "summary"}).json()
    assert summary["buses"] == 111 and summary["substations"]["basis"] == "stated"
    rsa = client.post("/api/grid/rsa", json={}).json()
    assert rsa.get("converged") is not False
    full = client.get("/api/network/semantics").json()
    bus = full["buses"][3]
    assert client.get("/api/network/resolve", params={"name": bus["mrid"]}).json()["index"] == bus["index"]
    bad = client.post("/api/network/upload", files={"file": ("x.zip", io.BytesIO(b"PK\x05\x06" + b"\0" * 18),
                                                             "application/zip")})
    assert bad.status_code == 400
