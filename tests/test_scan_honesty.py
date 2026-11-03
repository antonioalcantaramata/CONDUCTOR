"""
Scans and N-1 screens report what they could not assess (code review batch D,
findings 3.5 and 3.6).

A post-outage power flow that does not converge used to come back as an empty
violation table, which both N-1 endpoints read as "no violations": at 3.2×
load, where every outage collapses the power flow, the sweep reported the
network N-1 secure with 0 violations.
"""

import main_backend as mb


def _tick():
    return mb.app_data["timestamps"][40]


def test_collapsing_outages_are_insecure_not_clean(backend_client):
    # 4.5×: every one of the IEEE 30 demo's 41 outages collapses (at 3.6×,
    # 6 of 41 do; at 4.0×, 12).
    r = backend_client.post("/api/contingency/simulate_all", json={
        "timestamp": _tick(), "load_scaling_factor": 4.5}).json()
    assert r["system_n1_secure"] is False
    assert r["n_non_converged_outages"] == r["total_outages_tested"]
    assert "Bus_0 -> Bus_1 [L0]" in r["non_converged_outages"]


def test_a_single_collapsing_outage_says_so(backend_client):
    r = backend_client.post("/api/contingency/simulate", json={
        "timestamp": _tick(), "element_type": "line", "element_index": 0,
        "load_scaling_factor": 3.2}).json()
    assert r["system_secure"] is False and r["converged"] is False
    assert r["total_violations"] is None


def test_a_converging_sweep_reports_none_missing(backend_client):
    r = backend_client.post("/api/contingency/simulate_all", json={"timestamp": _tick()}).json()
    assert r["n_non_converged_outages"] == 0


def test_the_sweep_applies_renewable_scaling_to_the_wind_and_solar_units(backend_client):
    def sweep(scale):
        return backend_client.post("/api/contingency/simulate_all", json={
            "timestamp": _tick(), "sgen_scaling_factor": scale}).json()
    # The IEEE 30 demo loads every unit on voltage control (the `gen` table);
    # one at a fixed power factor is a static generator, as a converted
    # conventional unit is — the case renewable scaling used to get wrong.
    unit = "Gen_bus12"
    try:
        assert backend_client.post("/api/generators", json={"changes": [
            {"unit": unit, "control": "power_factor"}]}).status_code == 200
        # No unit is stated to be wind or solar: the scaling changes nothing, and says so
        # (it used to scale every static generator — conventional units included).
        nothing = sweep(0.0)
        assert nothing["total_violations"] == sweep(1.0)["total_violations"]
        assert "No unit of this network is stated to be wind or solar" in nothing["renewables_scaled"]["note"]
        backend_client.post("/api/network/declare", json={"units": [{"unit": unit, "source": "wind"}]})
        declared = sweep(0.0)
        assert declared["renewables_scaled"]["units"] == [unit]
        assert declared["total_violations"] != sweep(1.0)["total_violations"]
    finally:
        backend_client.post("/api/network/declare", json={"reset": True})
        backend_client.post("/api/generators", json={"reset": True})


def test_scans_report_their_non_converged_ticks(backend_client):
    r = backend_client.post("/api/rsa/worst_case", json={"n_steps": 3}).json()
    assert r["n_non_converged"] == 0 and r["non_converged_timestamps"] == []


def test_shared_bus_names_never_make_a_probability_above_one(backend_client):
    """Five buses called "Oscar" used to give a violation probability of 5.0."""
    net = mb.app_data["net"]
    saved = net.bus["name"].copy()
    net.bus.loc[net.bus.index[2:7], "name"] = "Oscar"
    try:
        r = backend_client.post("/api/rsa/probabilistic", json={
            "timestamp": mb.app_data["timestamps"][100], "n_samples": 30,
            "vm_lower_pu": 0.99, "vm_upper_pu": 1.01}).json()
    finally:
        net.bus["name"] = saved
    probabilities = r["bus_violation_probability"]
    assert probabilities and all(0.0 <= p <= 1.0 for p in probabilities.values())


def test_two_generators_on_one_bus_keep_distinct_names():
    import network_loader as nl
    import pandapower as pp
    import pandapower.networks as pn

    net = pn.case9()
    pp.create_gen(net, bus=1, p_mw=10, max_p_mw=40, vm_pu=1.0)
    _, names, _ = nl.gen_to_sgen(net)
    assert len(names) == len(set(names)) == 3


def test_historical_risk_reports_a_bad_tick_instead_of_failing_the_window(backend_client):
    """One non-converging tick used to fail the whole window with HTTP 503."""
    ts = mb.app_data["timestamps"]
    bad = ts[2]
    original = mb.app_data["measurements"][bad]
    mb.app_data["measurements"][bad] = original.assign(consumption=original["consumption"] * 8)
    try:
        r = backend_client.post("/api/rsa/historical_risk", json={
            "window_start": ts[0], "window_end": ts[5], "target": "bus:Bus_3"})
    finally:
        mb.app_data["measurements"][bad] = original
    body = r.json()
    assert r.status_code == 200
    assert body["n_non_converged"] == 1 and body["non_converged_timestamps"] == [bad]
    assert body["n_timestamps"] == 5
    assert body["exceedance_frequency_upper_bound"] >= body["exceedance_frequency"]
