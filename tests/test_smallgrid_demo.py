"""
The ENTSO-E SmallGrid demo (backend/network_profiles/entsoe_smallgrid.yaml; the model in
backend/network_data/entsoe_smallgrid/, © ENTSO-E, CC BY-SA 4.0): a CGMES model loaded as CIM and kept whole — its
units keep their names and mRIDs, their source is the model's, the model's own solution matches, an N-1 overload
says how long the model's temporary limits allow it, and the solved state is written as the model's SV.
"""

import contextlib
import io

import pytest
from fastapi.testclient import TestClient

import main_backend as mb

pytest.importorskip("lxml")


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("sg")
    systems, data = tmp / "systems", tmp / "data_files"
    systems.mkdir()
    data.mkdir()
    mp = pytest.MonkeyPatch()
    mp.setenv("GRID_PROFILE", "entsoe_smallgrid")
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


def test_the_model_is_kept_and_its_own_solution_matches(client):
    assert mb._cim_model() is not None and len(mb.app_data["net"].bus) == 118
    solved = mb._cim_solved()
    assert solved["matches"] and solved["buses_compared"] == 118


def test_units_keep_their_model_names_identity_and_source(client):
    units = client.get("/api/generators").json()["units"]
    names = {u["unit"] for u in units}
    assert {"Breed SM", "Muskngum SM", "Muskngum SM 2"} <= names
    facts = client.get("/api/network/semantics", params={"about": "units"}).json()
    thermal = facts["by_source"]["thermal"]["units"]
    assert len(thermal) == 18 and all(u["basis"] == "stated" for u in thermal)
    assert "ThermalGeneratingUnit" in thermal[0]["from"]
    units_now = mb.app_data["net"].gen                                    # on voltage control, as the model has them
    assert len(units_now) == 18 and units_now["origin_id"].notna().all()  # each unit keeps its machine's mRID


def test_an_n1_overload_says_how_long_the_model_allows_it(client):
    r = client.post("/api/contingency/simulate_all", json={"timestamp": "2000-01-04 19:00",
                                                          "max_line_loading_pct": 100,
                                                          "max_trafo_loading_pct": 100}).json()
    over = [v for v in r["violations"] if v["violation_type"] == "line_loading" and v["value"] > 100]
    assert over and "time_to_act_s" in r
    annotated = [v for v in over if v.get("temporary_limit") or v.get("beyond_temporary_limits")]
    assert len(annotated) == len(over)
    held = [v["temporary_limit"] for v in over if v.get("temporary_limit")]
    assert held and {h["held_for_s"] for h in held} <= {900.0, 60.0}


def test_the_solved_state_is_the_model_s_sv(client):
    import rdflib

    import cim_model
    r = client.get("/api/network/cim/solved", params={"source": "nominal"})
    assert r.status_code == 200, r.text
    g = rdflib.Graph().parse(data=r.content, format="xml")
    cim = rdflib.Namespace(next(str(ns) for p, ns in g.namespaces() if p == "cim"))
    ours = {str(g.value(s, cim["SvVoltage.TopologicalNode"])).split("#_")[-1]: float(g.value(s, cim["SvVoltage.v"]))
            for s in g.subjects(rdflib.RDF.type, cim.SvVoltage)}
    model = mb._cim_model()
    theirs = {cim_model.norm(model.get(sv).get("SvVoltage.TopologicalNode")): float(model.get(sv).get("SvVoltage.v"))
              for sv in model.of("SvVoltage")}
    assert set(ours) == set(theirs)                                       # the model's own nodes
    assert max(abs(ours[n] - theirs[n]) for n in ours) < 0.1             # kV


def test_the_week_congests_the_220_kv_corridor_at_the_evening_peak(client):
    r = client.post("/api/grid/rsa", json={"timestamp": "2000-01-04 19:00"}).json()
    names = {v["element"] for v in r["violations"]}
    assert any("X9" in n and ("Breed" in n or "Olive1" in n) for n in names)
    night = client.post("/api/grid/rsa", json={"timestamp": "2000-01-04 03:00"}).json()
    assert night["total_violations"] == 0


def test_an_element_can_be_named_by_its_mrid(client):
    import resolve
    import semantics as sm
    net = mb.app_data["net"]
    line = int(net.line.index[3])
    own = sm.row_mrids(net, "line", str(mb.app_data["loaded_fingerprint"]))[line]
    hit = resolve.find(net, own)
    assert (hit.etype, hit.idx, hit.how) == ("line", line, "mrid")
    unit = int(net.gen.index[0])
    mrid = sm.row_mrids(net, "gen", str(mb.app_data["loaded_fingerprint"]))[unit]
    assert resolve.find_unit(net, mrid).idx == unit
