"""
The CIM layer in the map, the reading check and the tools, for any network:

- the network map coloured by the layer: each bus by its units' energy source,
  its feeder, or how well the layer knows what is there;
- the reading check re-runs a study that depends on which units are wind or
  solar with the facts the layer does not know read the other way (unknown
  units counted as wind or solar; units known only by their name not counted),
  through the backend's `assume_unit_sources` — nothing kept;
- elements chosen by meaning ("the solar units on feeder Line 1-2"), in
  describe_network and wherever a study takes units;
- the layer checked against itself: what in the network's own data looks wrong.
"""

import pandapower.networks as pn
import pytest

import semantics as sm
from llm_agent.agent import layer_map as lm
from llm_agent.agent import network_map as nm
from llm_agent.agent import robustness as rb


@pytest.fixture(scope="module")
def cigre_layer():
    net = pn.create_cigre_network_mv(with_der="all")
    return net, sm.build(net, network_id="cigre")


# --- The map --------------------------------------------------------------------------------------------------

def test_buses_are_coloured_by_source_feeder_and_what_is_known(cigre_layer):
    _net, layer = cigre_layer
    source, order, colours = lm.bus_groups(layer, "source")
    assert source[7][0] == "wind" and source[3][0] == "solar"           # WKA 7; PV 3 alone at Bus 3
    assert set(order) <= set(lm.SOURCE_COLOURS) and all(colours[g] for g in order)
    feeder, order, colours = lm.bus_groups(layer, "feeder")
    assert len(order) == 2 and all(colours[g] in lm.FEEDER_COLOURS for g in order)
    known, order, _ = lm.bus_groups(layer, "known")
    assert {g for g, _d in known.values()} == {"unknown"}           # no load of CIGRE states its category
    with pytest.raises(ValueError, match="Unknown view"):
        lm.bus_groups(layer, "colour")


def test_the_layer_map_draws(cigre_layer):
    net, layer = cigre_layer
    payload = {"name": "CIGRE", "buses": [{"index": int(i), "name": str(r["name"]), "vn_kv": float(r["vn_kv"]),
                                           "in_service": True} for i, r in net.bus.iterrows()],
               "branches": [{"index": int(i), "name": str(r["name"]), "from_bus": int(r.from_bus),
                             "to_bus": int(r.to_bus), "kind": "line"} for i, r in net.line.iterrows()],
               "external_grids": [{"bus": int(net.ext_grid.bus.iloc[0])}]}
    network = nm.from_payload(payload)
    positions = nm.compute_layout(network)
    for by in lm.MODES:
        fig = lm.render_layer_map(network, positions, layer, by)
        names = {t.name for t in fig.data if t.name}
        assert "external grid" in names and len(fig.data) > len(network.edges)


# --- The reading check on facts -----------------------------------------------------------------------------

UNITS = {"unknown": {"units": [{"name": "Gen_bus2", "basis": "unknown"}, {"name": "Gen_bus5", "basis": "unknown"}]},
         "solar": {"units": [{"name": "Solar Plant A", "basis": "named"}, {"name": "PV 3", "basis": "stated"}]}}


def test_facts_the_layer_does_not_know_are_read_the_other_way():
    readings = rb.fact_readings("run_rsa", {"sgen_scaling_factor": 0.5}, {}, UNITS)
    assert [r.overrides["assume_unit_sources"] for r in readings] == [
        {"Gen_bus2": "wind", "Gen_bus5": "wind"}, {"Solar Plant A": "conventional"}]
    assert readings[0].label == "Gen_bus2, Gen_bus5 counted as wind or solar"
    assert "read only from their names" in readings[1].assumed
    # Only where the result depends on it: an unscaled assessment does not.
    assert rb.fact_readings("run_rsa", {}, {}, UNITS) == []
    assert rb.fact_readings("optimize_flexibility", {}, {}, UNITS)


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    """The backend in-process on IEEE 30 (no unit's source is stated); every file it writes goes to a temporary
    folder."""
    pytest.importorskip("fastapi.testclient")
    import main_backend as mb
    from fastapi.testclient import TestClient

    scratch = tmp_path_factory.mktemp("layer_tools")
    patch = pytest.MonkeyPatch()
    patch.setenv("GRID_PROFILE", "pglib_case30")
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


def test_a_study_runs_on_assumed_sources_and_keeps_nothing(client):
    c, mb = client
    units = c.get("/api/network/semantics?about=units").json()["by_source"]
    unknown = [u["name"] for u in units["unknown"]["units"]]
    plain = c.post("/api/grid/rsa", json={"sgen_scaling_factor": 0.5}).json()
    assert plain["renewables_scaled"]["units"] == []
    assumed = c.post("/api/grid/rsa", json={"sgen_scaling_factor": 0.5,
                                            "assume_unit_sources": {unknown[0]: "wind"}}).json()
    assert assumed["renewables_scaled"]["units"] == [unknown[0]]
    assert list(assumed["assumed_unit_sources"].values()) == ["wind"]
    # Nothing kept: the next study, and the layer, are as before.
    assert mb.app_data.get("unit_sources") in ({}, None)
    again = c.post("/api/grid/rsa", json={"sgen_scaling_factor": 0.5}).json()
    assert again["renewables_scaled"]["units"] == [] and "assumed_unit_sources" not in again


# --- Choosing by meaning --------------------------------------------------------------------------------------

def test_a_phrase_selects_by_what_the_layer_says(cigre_layer):
    _net, layer = cigre_layer
    assert [e["name"] for e in sm.select(layer, "wind units")["elements"]] == ["WKA 7"]
    assert len(sm.select(layer, "solar units")["elements"]) == 8
    on = sm.select(layer, "the renewable units on feeder Line 1-2")
    assert on["means"] == "renewable units in feeder Line 1-2" and {e["is"] for e in on["elements"]} == {"solar", "wind"}
    assert {e["name"] for e in sm.select(layer, "units in substation Bus 5")["elements"]} == \
        {"PV 5", "Residential fuel cell 1", "Battery 1"}
    assert [e["type"] for e in sm.select(layer, "transformers")["elements"]] == ["trafo", "trafo"]
    assert sm.select(layer, "unknown loads")["means"] == "loads whose kind the network does not say"
    # The name first, as people say it.
    assert sm.select(layer, "loads on the Line 12-13 feeder")["means"] == "loads in feeder Line 12-13"
    for bad, match in (("the north side", "Could not read"), ("tidal units", "not a kind of units"),
                       ("thermal units on feeder Line 12-13", "No thermal units"), ("loads at 33 kV", "no 33 kV")):
        with pytest.raises(ValueError, match=match):
            sm.select(layer, bad)


def test_studies_and_declarations_take_a_phrase(client):
    c, mb = client
    out = c.get("/api/network/semantics", params={"about": "select", "element": "unknown units"}).json()
    assert out["kind"] == "units" and len(out["elements"]) == 5
    assert c.get("/api/network/semantics", params={"about": "select", "element": "the moon"}).status_code == 400
    # Declared by meaning: every unit the phrase names.
    declared = c.post("/api/network/declare", json={"units": [{"unit": "units at 135 kV", "source": "thermal"}]}).json()
    assert len(declared["declared"]) == 5 and set(declared["declared"].values()) == {"thermal"}
    c.post("/api/network/declare", json={"reset": True})
    # Taken out of a flexibility study by meaning: the study names each unit it dropped, as for names.
    assert mb._expand_by_meaning(["Gen_bus2", "unknown units"])[:1] == ["Gen_bus2"]
    assert len(mb._expand_by_meaning(["unknown units"])) == 5


# --- The layer checked against itself -------------------------------------------------------------------------

def test_what_in_the_data_looks_wrong():
    net = pn.create_cigre_network_mv(with_der="all")
    net.sgen.at[int(net.sgen.index[net.sgen.name == "CHP diesel 1"][0]), "name"] = "Solar Farm North"
    net.load.at[net.load.index[0], "p_mw"] = -2.0
    net.bus.at[2, "vn_kv"] = 10.0                            # Line 1-2 now joins 20 and 10 kV
    kinds = {i["kind"]: i for i in sm.data_issues(net, sm.build(net, network_id="c"))}
    assert "Solar Farm North" in kinds["name_contradicts_type"]["element"]
    assert kinds["negative_load"]["level"] == "check"
    assert kinds["line_across_levels"]["element"] in ("Line 1-2", "Line 2-3")
    clean = pn.create_cigre_network_mv(with_der="all")
    assert sm.data_issues(clean, sm.build(clean, network_id="c")) == []


def test_the_agent_can_ask_for_the_issues(client):
    c, _ = client
    out = c.get("/api/network/semantics?about=issues").json()
    assert out["about"] == "issues" and isinstance(out["issues"], list) and out["note"]
