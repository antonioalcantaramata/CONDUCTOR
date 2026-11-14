"""
The synthetic data's settings, changed by the operator (the Data tab's panel,
/api/data/synthetic): checked settings, a preview that matches what is
generated, the network's own data unchanged when nothing is changed (the
demos and the papers' runs), the clock kept, the settings kept for the network
across a reload, units drawn with a shape their declared source no longer has,
uploaded data replaced only when confirmed, and Start fresh going back.
"""

import json

import pandapower.networks as pn
import pytest

import synthetic_timeseries as synth
from llm_agent.agent import synthetic_view as sv
from llm_agent.agent import system_tables as st_tables


# --- The generator ---------------------------------------------------------------------------------------------

def test_settings_are_checked_and_completed():
    assert synth.normalise_settings({}) == synth.DEFAULTS
    s = synth.normalise_settings({"n_days": 3.0, "peak_load_factor": 1})
    assert s["n_days"] == 3 and isinstance(s["n_days"], int) and s["peak_load_factor"] == 1.0
    assert s["forecast_overlap_days"] == 2.0
    assert synth.normalise_settings({"n_days": 1})["forecast_overlap_days"] == 1.0   # never longer than the data
    for bad, match in (({"days": 3}, "Unknown setting"), ({"n_days": 0}, "between 1 and 31"),
                       ({"n_days": 2.5}, "whole number"), ({"resolution_min": 20}, "one of 5, 10, 15, 30, 60"),
                       ({"n_days": 31, "resolution_min": 5}, "ticks"), ({"load_profile": "office"}, "one of"),
                       ({"peak_load_factor": 3}, "between"), ({"stress_events": "yes"}, "true or false"),
                       ({"seed": True}, "a number"), ({"generation_error": -0.1}, "between")):
        with pytest.raises(synth.SettingsError, match=match):
            synth.normalise_settings(bad)


def test_a_basis_taken_earlier_gives_the_same_data_as_the_network():
    net = pn.create_cigre_network_mv(with_der="all")
    shapes = {("sgen", 0): "solar", ("sgen", 8): "wind"}
    b = synth.basis(net)
    net.load["p_mw"] *= 3                     # what a tick does to the network after the basis is taken
    ts, from_basis = synth.generate(b, n_days=2, unit_shapes=shapes)
    ts2, from_net = synth.generate(pn.create_cigre_network_mv(with_der="all"), n_days=2, unit_shapes=shapes)
    assert ts == ts2
    assert all(from_basis[t].equals(from_net[t]) for t in ts)


def test_the_preview_is_the_total_of_what_is_generated():
    net = pn.case30()
    settings = synth.normalise_settings({"n_days": 2, "load_profile": "industrial", "peak_load_factor": 1.2,
                                         "generation_profile": "solar", "seed": 7})
    b = synth.basis(net)
    preview = synth.preview(b, settings)
    ts, data = synth.generate(b, n_days=2, load_profile="industrial", peak_load_factor=1.2,
                              generation_profile="solar", seed=7)
    assert preview["timestamps"] == ts
    for k in (0, 50, len(ts) - 1):
        assert preview["consumption"][k] == pytest.approx(data[ts[k]]["consumption"].sum(), abs=1e-3)
        assert preview["production"][k] == pytest.approx(data[ts[k]]["production"].sum(), abs=1e-3)
    assert set(preview["production_by_shape"]) == {"solar"}


def test_the_forecast_follows_the_stress_setting():
    net = pn.case30()
    ts, meas = synth.generate(net, n_days=3)
    on = synth.forecast(net, ts, meas, n_days=3)[1]
    off = synth.forecast(net, ts, meas, n_days=3, stress_events=False)[1]
    ahead = sorted(set(on) - set(ts))
    assert any(not on[t].equals(off[t]) for t in ahead)


def test_the_settings_in_words():
    words = synth.describe(synth.DEFAULTS)
    assert words.startswith("7 days at 15-minute steps; residential load")
    assert "100 % of the network's own" in words and "seed 42" in words


# --- The backend -----------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client(tmp_path_factory):
    """The backend in-process on the CIGRE MV demo profile; every file it would
    write goes to a temporary folder."""
    pytest.importorskip("fastapi.testclient")
    import main_backend as mb
    from fastapi.testclient import TestClient

    scratch = tmp_path_factory.mktemp("synthetic")
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


def test_unchanged_settings_give_the_data_the_network_always_had(client):
    """The demos and the papers' recorded runs depend on it: same profile, same week."""
    c, mb = client
    profile = mb.nl.load_profile("cigre_mv", backend_dir=mb.BACKEND_DIR)
    net = mb.nl.load_network(profile, backend_dir=mb.BACKEND_DIR)
    syn = mb.app_data["synthetic"]
    ts, meas = synth.generate(syn["basis"], n_days=7, resolution_min=15, load_profile="residential",
                              generation_profile=profile["synthetic_generation_profile"], seed=42,
                              peak_load_factor=profile["synthetic_peak_load_factor"], unit_shapes=syn["shapes"])
    assert len(net.load) == len(syn["basis"].loads)
    assert ts == mb.app_data["timestamps"]
    assert all(meas[t].equals(mb.app_data["measurements"][t]) for t in ts[:5])


def test_the_panel_shows_the_settings_the_network_loads_with(client):
    c, _ = client
    data = c.get("/api/data/synthetic").json()
    assert data["available"] and data["in_use"] and not data["custom"]
    assert data["settings"]["generation_profile"] == "flat" and data["settings"]["peak_load_factor"] == 0.9
    assert data["units"]["stated"]["solar"] == 8 and data["stale"] == []


def test_a_preview_changes_nothing_and_bad_settings_are_refused(client):
    c, mb = client
    before = mb._data_key()
    out = c.post("/api/data/synthetic/preview", json={"settings": {"n_days": 14, "load_profile": "flat"}}).json()
    assert out["n_ticks"] == 14 * 96 and len(out["timestamps"]) <= 700 and out["current"]["consumption"]
    assert mb._data_key() == before
    bad = c.post("/api/data/synthetic/preview", json={"settings": {"n_days": 40}})
    assert bad.status_code == 400 and "between 1 and 31" in bad.json()["detail"]


def test_generating_keeps_the_clock_and_is_kept_for_the_network(client):
    c, mb = client
    c.post("/api/time/advance", json={"target_timestamp": "2000-01-03 18:00"})
    key, data_key = mb._assumptions_key(), mb._data_key()
    out = c.post("/api/data/synthetic", json={"settings": {"n_days": 14, "load_profile": "industrial"}}).json()
    assert out["status"] == "success" and out["n_timestamps"] == 14 * 96 and out["clock"] == "2000-01-03 18:00:00"
    assert out["custom"] and out["kept"]
    assert mb._data_key() != data_key and mb._assumptions_key() != key      # earlier answers: "other data loaded"
    note = c.get("/api/network/semantics?about=data").json()["note"]
    assert "generated with: 14 days at 15-minute steps; industrial load" in note
    stored = json.load(open(mb._ASSUMPTIONS_FILE))["by_fingerprint"][mb.app_data["loaded_fingerprint"]]
    assert stored["synthetic_data"]["n_days"] == 14
    assert mb._kept_changes({"fingerprint": mb.app_data["loaded_fingerprint"]})["synthetic_data"] == 1
    # Loading the network again generates the same data from the kept settings.
    first = mb.app_data["measurements"]["2000-01-10 12:00:00"].copy()
    assert c.post("/api/scenario", json={"name": "cigre_mv"}).status_code == 200
    assert len(mb.app_data["timestamps"]) == 14 * 96
    assert mb.app_data["measurements"]["2000-01-10 12:00:00"].equals(first)


def test_a_unit_declared_after_the_data_is_flagged_until_they_are_generated_again(client):
    c, mb = client
    assert c.post("/api/network/declare", json={"units": [{"unit": "PV 3", "source": "wind"}]}).status_code == 200
    stale = c.get("/api/data/synthetic").json()["stale"]
    assert stale == [{"unit": "PV 3", "drawn_as": "solar", "now": "wind"}]
    assert c.post("/api/data/synthetic", json={"settings": {}}).json()["stale"] == []


def test_uploaded_data_are_replaced_only_when_confirmed(client, monkeypatch):
    c, mb = client
    monkeypatch.setitem(mb.app_data, "measurements_source", "uploaded")
    out = c.post("/api/data/synthetic", json={"settings": {}}).json()
    assert out["status"] == "confirm_required" and "the measurements in use (uploaded)" in out["message"]
    assert c.post("/api/data/synthetic", json={"settings": {}, "replace": True}).json()["status"] == "success"
    assert mb.app_data["measurements_source"] == "synthetic"


def test_start_fresh_goes_back_to_the_data_the_network_loads_with(client):
    c, mb = client
    assert c.post("/api/data/synthetic", json={"settings": {"n_days": 3}}).json()["custom"]
    assert c.post("/api/scenario/forget", json={"name": "cigre_mv"}).status_code == 200
    data = c.get("/api/data/synthetic").json()
    assert not data["custom"] and len(mb.app_data["timestamps"]) == 7 * 96
    assert mb._synthetic_chosen() is None


def test_back_to_the_defaults(client):
    c, mb = client
    c.post("/api/data/synthetic", json={"settings": {"seed": 9}})
    out = c.post("/api/data/synthetic", json={"reset": True}).json()
    assert out["settings"] == out["base"] and not out["kept"]


# --- The panel's words -----------------------------------------------------------------------------------------

def test_the_units_line_counts_each_shape_in_words():
    payload = {"units": {"stated": {"solar": 8, "wind": 1, "flat": 0}, "default": 0}}
    assert sv.units_line(payload, "flat") == ("As their source says: 8 follow the sun, 1 follows the wind. "
                                               "Every unit's source is known, so this shape applies to none.")
    assert sv.units_line({"units": {"stated": {}, "default": 1}}, "wind") == \
        "1 unit the network does not describe follows this shape."


def test_the_stale_line_and_the_differences():
    line = sv.stale_line([{"unit": "PV 3", "drawn_as": "solar", "now": "wind"}])
    assert "**PV 3** (now wind, drawn as solar)" in line
    assert sv.stale_line([]) == ""
    base = synth.DEFAULTS
    assert sv.differences({**base, "n_days": 14, "peak_load_factor": 1.2}, base) == [
        "days 7 → 14", "daily peak 100 % → 120 %"]


def test_data_of_another_period_are_not_drawn_beside_the_preview():
    ours = {"timestamps": ["2000-01-01 00:00:00", "2000-01-07 23:45:00"],
            "current": {"timestamps": ["2024-03-04 00:00:00", "2024-03-05 23:45:00"]}}
    assert not sv.comparable(ours)
    assert "2024-03-04 → 2024-03-05" in sv.other_period(ours)
    ours["current"]["timestamps"] = ["2000-01-01 00:00:00", "2000-01-07 23:45:00"]
    assert sv.comparable(ours)


def test_the_settings_count_among_what_is_kept():
    assert st_tables.kept_line({"synthetic_data": 1, "renames": 2}) == \
        "Your changes are kept: synthetic-data settings · 2 renames"
