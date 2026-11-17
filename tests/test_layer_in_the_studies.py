"""
The CIM layer in the studies, for any network:

1. a unit the layer says is wind or solar is curtailable, not dispatchable up:
   0 €/MWh to produce, 40 €/MWh to curtail, and never above what the resource
   gives at the moment studied (as a proposal's own wind and pv units already
   were) — the optimisations could raise a PV unit to full capacity at night;
2. an outage that cuts off load says which normally open ties could restore it,
   each checked with a power flow at the study's moment;
3. in the synthetic data, a load follows its category (residential,
   commercial, industrial, a data centre…), as a unit follows its source;
4. an N-1 sweep can be scoped to a feeder, a substation or a voltage level.
"""

import pandapower.networks as pn
import pytest

import semantics as sm
import synthetic_timeseries as synth
from llm_agent.agent import renderers


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    """The backend in-process on the CIGRE MV demo profile; every file it would write goes to a temporary folder."""
    pytest.importorskip("fastapi.testclient")
    import main_backend as mb
    from fastapi.testclient import TestClient

    scratch = tmp_path_factory.mktemp("layer")
    patch = pytest.MonkeyPatch()
    patch.setenv("GRID_PROFILE", "cigre_mv")
    for name in ("_LAST_UPLOAD_SENTINEL", "_LAST_TIMESERIES_SENTINEL", "_LAST_FORECAST_SENTINEL",
                 "_PROPOSALS_FILE", "_ASSUMPTIONS_FILE", "_SCENARIO_FILE"):
        patch.setattr(mb, name, str(scratch / f"{name}.json"))
    patch.setattr(mb, "DATA_FILES_DIR", str(scratch / "data_files"))
    patch.setattr(mb, "SYSTEMS_DIR", str(scratch / "systems"))
    saved = dict(mb.app_data)
    with TestClient(mb.app) as c:
        yield c, mb
    mb.app_data.clear()
    mb.app_data.update(saved)
    patch.undo()


# --- 1. Wind and solar in the optimisations ---------------------------------------------------------------------

def test_wind_and_solar_are_costed_as_curtailable(client):
    c, _ = client
    rows = {r["unit"]: r for r in c.get("/api/costs").json()["units"]}
    pv, wind, chp = rows["PV 3"], rows["WKA 7"], rows["CHP diesel 1"]
    assert pv["kind"] == "pv" and wind["kind"] == "wind"
    for unit in (pv, wind):
        assert unit["cost_eur_per_mwh"] == 0.0 and unit["up_eur_per_mwh"] is None and unit["down_eur_per_mwh"] == 40.0
    assert chp["cost_eur_per_mwh"] == 50.0 and chp["up_eur_per_mwh"] > 0


def test_a_study_never_lets_wind_or_solar_exceed_the_resource(client):
    c, mb = client
    ts = mb.app_data["timestamps"]
    net = mb.copy.deepcopy(mb._net_for(None))
    raw = net.sgen.at[0, "max_p_mw"] if "max_p_mw" in net.sgen.columns else float("nan")
    capacity = float(raw) if raw == raw and raw > 0 else float("inf")      # PV 3, as loaded (CIGRE states none)
    net, _ = mb._assign_tick(net, mb.prepare_measurement_df(mb.app_data["measurements"][ts[0]]), ts[0])
    assert net.sgen.at[0, "max_p_mw"] == pytest.approx(max(net.sgen.at[0, "p_mw"], 0.0))
    assert net.sgen.at[0, mb.CAPACITY_COLUMN] == capacity
    noon = next(t for t in ts if t.endswith("12:00:00"))
    net, _ = mb._assign_tick(net, mb.prepare_measurement_df(mb.app_data["measurements"][noon]), noon)
    assert net.sgen.at[0, "max_p_mw"] == pytest.approx(min(net.sgen.at[0, "p_mw"], capacity))
    # A scenario with more wind and sun raises the cap with the output.
    mb._scale_renewables(net, 1.3)
    assert net.sgen.at[0, "max_p_mw"] == pytest.approx(net.sgen.at[0, "p_mw"])
    # A conventional unit keeps its capacity; the live network keeps its own limits.
    chp = int(net.sgen.index[net.sgen.name == "CHP diesel 1"][0])
    live = mb._net_for(None).sgen
    assert (net.sgen.at[chp, "max_p_mw"] == live.at[chp, "max_p_mw"]) if "max_p_mw" in live.columns \
        else net.sgen.at[chp, "max_p_mw"] != net.sgen.at[chp, "max_p_mw"]       # NaN: no limit of its own
    assert mb.CAPACITY_COLUMN not in mb._net_for(None).sgen.columns


def test_without_the_layer_the_studies_behave_as_before(client, monkeypatch):
    c, mb = client
    monkeypatch.setenv("CONDUCTOR_ONTOLOGY", "none")
    mb.app_data.pop("_variable_units", None)
    rows = {r["unit"]: r for r in c.get("/api/costs").json()["units"]}
    assert rows["PV 3"]["cost_eur_per_mwh"] == 50.0
    assert mb._variable_units(mb._net_for(None)) == {}


# --- 2. Restoring ties in an outage study -------------------------------------------------------------------------

def test_an_outage_that_cuts_off_load_names_the_ties_that_restore_it(client):
    c, mb = client
    net = mb._net_for(None)
    line = int(net.line.index[net.line.name == "Line 2-3"][0])
    out = c.post("/api/contingency/simulate", json={"element_type": "line", "element_index": line}).json()
    assert out["unsupplied_load_mw"] > 0
    [tie] = out["ties_that_could_restore"]
    assert "Line 14-8" in tie["tie"] and tie["converged"] and tie["buses_restored"] == tie["buses_cut_off"]
    assert tie["limits"]["vm_pu"] == [out["thresholds_used"]["vm_lower_pu"], out["thresholds_used"]["vm_upper_pu"]]
    if not out.get("violations"):
        text = renderers.render_contingency_violations(out).layout.annotations[0].text
        assert "Closing Line 14-8" in text


# --- 3. Loads follow their category -------------------------------------------------------------------------------

def test_a_load_follows_its_category():
    net = pn.case30()
    b = synth.basis(net)
    ts, plain = synth.generate(b, n_days=2)
    ts2, shaped = synth.generate(b, n_days=2, load_shapes={int(net.load.index[0]): "commercial"})
    bus = synth.basis(net).names[int(net.load.at[net.load.index[0], "bus"])]
    row = lambda data, t: data[t].set_index("substation_name").at[bus, "consumption"]   # noqa: E731
    night, office = ts[2 * 4], ts[11 * 4]                        # 02:00 and 11:00 on Saturday 1 January 2000
    # The commercial load is low at night, where the residential shape it replaced is not.
    assert row(shaped, night) != row(plain, night)
    # Every other bus is untouched by the one load's shape — their draws come first.
    other = synth.basis(net).names[int(net.load.at[net.load.index[5], "bus"])]
    assert plain[office].set_index("substation_name").at[other, "consumption"] == \
        shaped[office].set_index("substation_name").at[other, "consumption"]


def test_a_category_declared_after_the_data_is_flagged_and_followed(client):
    c, mb = client
    load = mb.sem.load_key(mb._net_for(None), int(mb._net_for(None).load.index[0]))
    assert c.post("/api/network/declare", json={"loads": [{"load": load, "category": "data_centre"}]}).status_code == 200
    data = c.get("/api/data/synthetic").json()
    assert {"unit": load, "drawn_as": "residential", "now": "flat", "kind": "load"} in data["stale"]
    assert data["loads"]["stated"] == {"flat": 1}
    after = c.post("/api/data/synthetic", json={"settings": {}}).json()
    assert after["stale"] == [] and mb.app_data["synthetic"]["load_shapes"] == {int(mb._net_for(None).load.index[0]): "flat"}
    c.post("/api/network/declare", json={"reset": True})


# --- 4. A scoped N-1 ----------------------------------------------------------------------------------------------

def test_a_scope_names_the_branches_of_a_level_a_feeder_or_a_substation():
    net = pn.create_cigre_network_mv(with_der="all")
    layer = sm.build(net, network_id="cigre")
    level, words = sm.scope_branches(layer, "20 kV")
    assert {t for t, _ in level} == {"line", "trafo"} and "20 kV" in words
    assert all(t == "trafo" for t, _ in sm.scope_branches(layer, "110 kV")[0])
    feeder = layer["containers"]["feeders"][0]["name"]
    branches, words = sm.scope_branches(layer, f"feeder {feeder}")
    assert branches and branches < level and feeder.split(" | ")[0] in words
    assert sm.scope_branches(layer, f"feeder {feeder.split(' | ')[0]}")[0] == branches      # by its short name too
    with pytest.raises(ValueError, match="Feeders:"):
        sm.scope_branches(layer, "the north feeder")
    with pytest.raises(ValueError, match="no 33 kV level"):
        sm.scope_branches(layer, "33 kV")


def test_a_scoped_sweep_tests_only_its_outages(client):
    c, mb = client
    out = c.post("/api/contingency/simulate_all", json={"scope": "110 kV"}).json()
    assert out["scope"]["outages_in_scope"] == 2 and out["total_outages_tested"] == 2
    full = c.post("/api/contingency/simulate_all", json={}).json()
    assert "scope" not in full and full["total_outages_tested"] > out["total_outages_tested"]
    bad = c.post("/api/contingency/simulate_all", json={"scope": "Feeder 99"})
    assert bad.status_code == 400 and "No feeder, substation or voltage level" in bad.json()["detail"]


def test_declaring_by_meaning_skips_what_cannot_be_declared(client):
    """A phrase may name storage too ("the units in substation Bus 5": PV 5, a fuel cell, Battery 1); a
    declaration, a cost or a generator model is for generators, so the battery is left out, not refused."""
    c, _ = client
    out = c.post("/api/network/declare", json={"units": [{"unit": "units in substation Bus 5", "source": "solar"}]})
    assert out.status_code == 200 and set(out.json()["declared"]) == {"PV 5", "Residential fuel cell 1"}
    c.post("/api/network/declare", json={"reset": True})
