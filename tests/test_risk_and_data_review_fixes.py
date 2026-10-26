"""
Risk studies, scans and network data — what a code review found wrong, fixed:

- substations were listed as "Bus 1" while the data and every result say
  "Bus_1", or as unit names on a network whose buses are unnamed; unit-keyed
  arguments were shown with a substation name as their example;
- the assessment listed no voltage-controlled unit (the default IEEE 30 showed
  only the external grid and 0 MW available), and crashed at the nominal point
  whenever there was a violation;
- worker threads of a study lost its load power factor;
- the worst-case scan's loading left transformers out and its import took the
  first external grid only;
- historical risk merged two episodes across a tick with no converged power
  flow, and called a violation count a "signed limit margin";
- the probabilistic assessment dropped samples whose power flow failed, which
  made the risk read lower;
- a window's security total over some moments read as the whole window's;
- the envelope's `post_opf` reference applied a dispatch computed for another
  network, other data or another load scaling;
- an uploaded network of static generators only had them rebuilt per load,
  injecting minus the bus's production.
"""

import io
import threading

import pandapower as pp
import pandapower.networks as pn
import pytest
from fastapi import HTTPException

import main_backend as mb
from llm_agent.agent.system_prompt import build_system_prompt

PEAK = "2000-01-06 20:00:00"


@pytest.fixture(scope="module")
def demo(tmp_path_factory):
    """The default profile (IEEE 30, every unit voltage-controlled)."""
    pytest.importorskip("fastapi.testclient")
    from fastapi.testclient import TestClient
    store = tmp_path_factory.mktemp("risk_and_data")
    patch = pytest.MonkeyPatch()
    patch.delenv("GRID_PROFILE", raising=False)
    patch.setattr(mb, "SYSTEMS_DIR", str(store / "systems"))
    patch.setattr(mb, "DATA_FILES_DIR", str(store / "data_files"))
    for name in ("_LAST_UPLOAD_SENTINEL", "_LAST_TIMESERIES_SENTINEL", "_LAST_FORECAST_SENTINEL",
                 "_PROPOSALS_FILE", "_ASSUMPTIONS_FILE", "_SCENARIO_FILE"):
        patch.setattr(mb, name, str(store / f"{name}.json"))
    saved = dict(mb.app_data)
    with TestClient(mb.app) as client:
        yield client
    mb.app_data.clear()
    mb.app_data.update(saved)
    patch.undo()


# --- names -----------------------------------------------------------------

def test_substations_are_named_as_the_data_names_buses():
    net = pn.case14()
    net.bus["name"] = None
    subs = mb.get_substation_names(net)
    assert subs[:3] == ["Bus_1", "Bus_2", "Bus_3"] and all(" " not in s for s in subs)
    net.gen["name"] = [f"Gen_bus{int(b)}" for b in net.gen.bus]       # named units, unnamed buses
    subs = mb.get_substation_names(net)
    assert not any(s.startswith("Gen_bus") for s in subs)
    assert f"Bus_{int(net.gen.bus.iloc[-1])}" in subs                 # a named unit's bus (no load there)


def test_unit_keyed_examples_use_a_unit():
    gc = {"substation_names": ["Bus_1", "Bus_2"], "unit_names": ["Gen_bus1", "Gen_bus4"]}
    prompt = build_system_prompt(gc)
    assert '`{"Gen_bus1": 5.0}`' in prompt and "Gen_bus1, Gen_bus4" in prompt
    assert '{"Bus_1": 5.0}' not in prompt


def test_the_grid_constants_list_units_and_buses_by_their_keys(demo):
    gc = demo.get("/api/grid_constants").json()
    assert gc["unit_names"] == ["Gen_bus1", "Gen_bus12", "Gen_bus4", "Gen_bus7", "Gen_bus10"]
    assert gc["substation_names"] and all(s.startswith("Bus_") for s in gc["substation_names"])


# --- the assessment ----------------------------------------------------------

def test_the_assessment_lists_voltage_controlled_units(demo):
    out = demo.post("/api/grid/rsa", json={"timestamp": PEAK}).json()
    names = [u["name"] for u in out["gen_dispatch"]]
    assert {"Gen_bus1", "Gen_bus4"} <= set(names)
    assert out["flexibility_metrics"]["total_available_capacity_mw"] > 0


def test_the_nominal_point_with_a_violation_is_assessed(demo):
    r = demo.post("/api/grid/rsa", json={"data_source": "nominal", "vm_lower_pu": 1.0, "vm_upper_pu": 1.01})
    assert r.status_code == 200, r.text
    assert r.json()["total_violations"] > 0 and r.json()["timestamp"] == "nominal"


# --- a study's settings in worker threads ------------------------------------

def test_a_worker_thread_keeps_the_study_power_factor():
    seen = {}

    def tick():
        seen["pf"] = mb._study.load_pf
        mb._study.pf_used = True
        mb._study.q_share = (0.4, 0.6)

    mb._study.load_pf, mb._study.pf_used, mb._study.q_share = 0.8, False, None
    try:
        worker = threading.Thread(target=mb._in_this_study(tick))
        worker.start()
        worker.join()
        assert seen["pf"] == 0.8
        assert mb._study.pf_used is True and mb._study.q_share == (0.4, 0.6)
    finally:
        mb._study.load_pf, mb._study.pf_used, mb._study.q_share = None, False, None


def test_the_scenario_scan_says_the_power_factor_it_used(demo):
    out = demo.post("/api/rsa/scan_scenarios", json={"start_timestamp": PEAK, "n_steps": 1,
                                                     "sgen_scales": [1.0], "load_power_factor": 0.8}).json()
    assert out["load_power_factor"]["value"] == 0.8 and out["load_power_factor"]["source"] == "this study"


# --- worst case and historical risk -------------------------------------------

def test_the_worst_case_loading_counts_transformers(demo):
    out = demo.post("/api/rsa/worst_case", json={"metric": "max_loading", "n_steps": 3}).json()
    series = out["series"]
    assert len(series["max_trafo_loading"]) == len(series["max_line_loading"]) == out["n_scanned"]
    assert out["worst_value"] == max(series["max_line_loading"] + series["max_trafo_loading"])


def _fake_snapshots(monkeypatch, plan: dict):
    """`_run_rsa_snapshot` answering from `plan` {timestamp: violations, or None for no power flow}."""
    def snapshot(ts, request, applied=None):
        if plan[ts] is None:
            raise HTTPException(status_code=503, detail="no power flow")
        return {"timestamp": ts, "total_violations": plan[ts], "all_voltages": [], "converged": True,
                "all_line_loading": [{"line_name": "L1", "loading_percent": 96.0}], "all_trafo_loading": []}
    monkeypatch.setattr(mb, "_run_rsa_snapshot", snapshot)


def test_an_unconverged_tick_ends_a_risk_episode(demo, monkeypatch):
    ticks = mb.app_data["timestamps"][:5]
    _fake_snapshots(monkeypatch, dict(zip(ticks, [2, 1, None, 3, 0])))
    out = demo.post("/api/rsa/historical_risk", json={"window_start": ticks[0], "window_end": ticks[-1]}).json()
    episodes = sorted((e["start"], e["duration_steps"]) for e in out["worst_episodes"])
    assert episodes == [(ticks[0], 2), (ticks[3], 1)]
    assert out["duration_curve_metric"] == "violation_count" and out["n_non_converged"] == 1


def test_a_branch_near_miss_follows_its_band(demo, monkeypatch):
    ticks = mb.app_data["timestamps"][:2]
    _fake_snapshots(monkeypatch, dict(zip(ticks, [0, 0])))
    body = {"window_start": ticks[0], "window_end": ticks[-1], "target": "line:L1", "max_line_loading_pct": 100}
    assert demo.post("/api/rsa/historical_risk", json=body).json()["near_miss_count"] == 2      # 4 pp ≤ 5
    tight = demo.post("/api/rsa/historical_risk", json={**body, "near_miss_band_pct": 2.0}).json()
    assert tight["near_miss_count"] == 0 and tight["duration_curve_metric"] == "signed_limit_margin"


# --- probabilistic -------------------------------------------------------------

def test_a_sample_with_no_power_flow_counts_as_violating(demo, monkeypatch):
    real = mb._run_probabilistic_sample
    calls = iter(range(10_000))

    def every_fourth_fails(*args, **kwargs):
        return None if next(calls) % 4 == 0 else real(*args, **kwargs)
    monkeypatch.setattr(mb, "_run_probabilistic_sample", every_fourth_fails)
    out = demo.post("/api/rsa/probabilistic", json={"timestamp": "2000-01-06 04:00:00", "n_samples": 20,
                                                    "load_sigma": 0.0}).json()
    assert out["n_non_converged"] == 5 and out["n_converged"] == 15
    assert out["p_any_violation"] >= 5 / 20


# --- a window's security total -----------------------------------------------------

def test_a_total_over_some_moments_says_so(demo, monkeypatch):
    def rows(request, ticks, job, moment=None):
        priced = {"preventive": {"security_cost_eur_per_h": 10.0, "n1_outages_with_violations": 0}}
        unpriced = {"preventive": {"not_offered": "no dispatch"}}
        return [{"timestamp": t, **(priced if k == 0 else unpriced)} for k, t in enumerate(ticks)], 1
    monkeypatch.setattr(mb, "_run_moments", rows)
    out = demo.post("/api/flexibility/security_over_time",
                    json={"start": PEAK, "hours": 2, "moments": 3}).json()
    total = out["totals"]["preventive"]
    assert total["moments_priced"] == 1 and total["hours_priced"] == 0.5
    assert "Covers 1 of 3 moments" in total["note"]


# --- the envelope's reference dispatch -------------------------------------------

def test_a_cached_dispatch_for_another_study_is_not_applied(demo):
    unit, night = "Gen_bus4", "2000-01-06 04:00:00"
    mb._remember_dispatch([{"element": unit, "Pg_new": 1.0, "Qg_new": 0.0}], night,
                          mb.FlexibilityRequest(timestamp=night, load_scaling_factor=1.3))
    out = demo.post("/api/flexibility/envelope", json={"gen_name": unit, "timestamp": night, "resolution": 5,
                                                       "reference_state": "post_opf"}).json()
    assert out["reference_state_used"] == "scada"
    assert out["reference_dispatch_summary"]["reason"] == "cached_dispatch_context_mismatch"
    low, high = out["safe_q_range_at_base_p"]
    # A voltage-controlled unit: within its own reactive limits (or its base point, where it runs).
    cap_lo, cap_hi = out["unit_q_limits_mvar"]
    base_q = out["base_point"]["q_mvar"]
    assert min(cap_lo, base_q) <= low <= high <= max(cap_hi, base_q)


# --- uploads --------------------------------------------------------------------------

def test_an_upload_of_static_generators_keeps_them(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    systems, data = tmp_path / "systems", tmp_path / "data_files"
    systems.mkdir()
    data.mkdir()
    monkeypatch.setenv("GRID_PROFILE", "pglib_case14")
    monkeypatch.setattr(mb, "SYSTEMS_DIR", str(systems))
    monkeypatch.setattr(mb, "DATA_FILES_DIR", str(data))
    for name in ("_LAST_UPLOAD_SENTINEL", "_LAST_TIMESERIES_SENTINEL", "_LAST_FORECAST_SENTINEL",
                 "_PROPOSALS_FILE", "_ASSUMPTIONS_FILE"):
        monkeypatch.setattr(mb, name, str(systems / f"{name}.json"))
    monkeypatch.setattr(mb, "_SCENARIO_FILE", str(tmp_path / "scenario.json"))
    net = pn.create_cigre_network_mv(with_der="pv_wind")
    units = sorted(net.sgen.name)
    saved = dict(mb.app_data)
    try:
        with TestClient(mb.app) as client:
            r = client.post("/api/network/upload", files={"file": (
                "cigre.json", io.BytesIO(pp.to_json(net).encode()), "application/json")})
            assert r.status_code == 200, r.text
            loaded = mb.app_data["net"]
            assert sorted(loaded.sgen.name) == units
            assert (loaded.sgen.p_mw >= 0).all() and loaded.sgen.p_mw.sum() > 0
    finally:
        mb.app_data.clear()
        mb.app_data.update(saved)



def test_the_robust_dispatch_names_the_buses_tightened_most():
    # Observed live: thirty per-bus back-offs, most of them 0, and the answer named none.
    import main_backend as mb
    out = mb._largest_back_offs({"Bus_0": 0.0, "Bus_14": 0.0046, "Bus_9": 0.004}, {"Bus_29": 0.006},
                                {"Bus_14": 1.0454, "Bus_9": 1.046}, {"Bus_29": 0.956})
    assert [r["bus"] for r in out["largest_back_offs"]] == ["Bus_29", "Bus_14", "Bus_9"]
    assert "Bus_29 (by up to 0.0060 p.u.; lower to 0.956 p.u.)" in out["back_off_summary"]
    # Both limits of one bus: one entry.
    both = mb._largest_back_offs({"Bus_29": 0.009}, {"Bus_29": 0.0098}, {"Bus_29": 1.041}, {"Bus_29": 0.96})
    assert len(both["largest_back_offs"]) == 1 and "upper to 1.041, lower to 0.960" in both["back_off_summary"]
    assert "small margin" in out["back_off_summary"]
    assert mb._largest_back_offs({"Bus_0": 0.0}, {}, {}, {})["largest_back_offs"] == []



def test_a_missed_risk_target_says_why():
    import main_backend as mb
    out = mb._why_not_met({"line_violation_probability_after": {"Bus_0 -> Bus_1 [L0]": 0.99},
                           "bus_undervoltage_probability_after": {"Bus_28": 0.09, "Bus_29": 0.975, "Bus_5": 0.01}},
                          0.05)
    assert "voltage limits only, so it cannot lower the overload risk of Bus_0 -> Bus_1 [L0] (99.0 %)" in out["why_not_met"]
    assert "Bus_29 (97.5 %), Bus_28 (9.0 %) remains" in out["why_not_met"] and "Bus_5" not in out["why_not_met"]
