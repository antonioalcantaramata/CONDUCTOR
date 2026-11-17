"""A capped optimisation that cannot meet its cap is reported as failed
(code review 2.1).

It used to return status "optimal" and "Grid restored at slack cap ±10 MW"
with the uncapped dispatch — 190 MW of import — as the answer, cached for the
envelope tool. Needs the IPOPT solver; skipped where it is not installed (CI).
"""

import shutil

import pytest

import main_backend as mb

pytestmark = pytest.mark.skipif(shutil.which("ipopt") is None, reason="IPOPT not installed")


def test_an_unmeetable_cap_is_a_failed_result(backend_client):
    # On the IEEE 30 demo the five units cover a 10 MW cap at this hour as measured;
    # at 1.3× load they cannot (the uncapped dispatch is feasible).
    ts = mb.app_data["timestamps"][40]
    r = backend_client.post("/api/flexibility/optimize",
                            json={"timestamp": ts, "slack_max_mw": 10, "load_scaling_factor": 1.3}).json()
    assert r["status"] == "cap_infeasible" and r["feasible"] is False
    assert r["slack_cap_respected"] is False and r["activated_resources"] == []
    assert r["uncapped_reference_dispatch"]
    assert "Grid restored" not in r["message"]


def test_a_met_cap_says_so(backend_client):
    ts = mb.app_data["timestamps"][40]
    r = backend_client.post("/api/flexibility/optimize", json={"timestamp": ts}).json()
    assert r["status"] == "optimal" and r["feasible"] is True and r["slack_cap_respected"] is True


def test_the_optimiser_uses_the_networks_band_by_default(backend_client):
    ts = mb.app_data["timestamps"][40]
    r = backend_client.post("/api/flexibility/optimize", json={"timestamp": ts}).json()
    assert (r["opf_vm_lower_used"], r["opf_vm_upper_used"]) == (
        mb.app_data["default_vm_lower"], mb.app_data["default_vm_upper"])
    asked = backend_client.post("/api/flexibility/optimize",
                                json={"timestamp": ts, "opf_vm_upper": 1.045}).json()
    assert asked["opf_vm_upper_used"] == 1.045


class TestInfeasibilityIsExplained:
    """A failed OPF says why (diagnosis re-solve with the grid voltage free).

    Found at 2000-01-03 14:15 on the old default data, where the OPF answered
    only "Infeasible.". On the IEEE 30 demo the units' reactive ranges are
    wide (Gen_bus1 up to 100 Mvar), so the shortage is made by limiting each
    to ±10 Mvar: at the evening peak local reactive power runs out at the grid
    connection's 1.0 p.u., and a raise inside the band fixes it.
    """

    UNITS = ("Gen_bus1", "Gen_bus4", "Gen_bus7", "Gen_bus10", "Gen_bus12")

    @pytest.fixture
    def short_of_reactive(self, backend_client):
        r = backend_client.post("/api/generators", json={"changes": [
            {"unit": u, "q_min_mvar": -10, "q_max_mvar": 10} for u in self.UNITS]})
        assert r.status_code == 200, r.text
        yield backend_client
        backend_client.post("/api/generators", json={"reset": True})

    def test_short_of_local_reactive_power(self, short_of_reactive):
        r = short_of_reactive.post("/api/flexibility/optimize", json={
            "timestamp": "2000-01-03 19:00:00"}).json()
        assert r["feasible"] is False and r["activated_resources"] == []
        d = r["infeasibility_diagnosis"]
        assert d["solvable_with_grid_voltage_free"] is True
        assert d["grid_voltage_fixed_pu"] < d["grid_voltage_needed_pu"] <= d["voltage_band_pu"][1]
        assert {u["name"] for u in d["reactive_sources_at_limit"]} == set(self.UNITS)
        assert "tap-changer" in r["message"] and r["message"] != "Infeasible."

    def test_beyond_either_lever(self, short_of_reactive):
        """Losing line 0 (a main import line) at the evening peak with the
        units' reactive power so limited: neither the grid voltage nor lifting
        branch limits — even both — gives a dispatch."""
        r = short_of_reactive.post("/api/contingency/optimize", json={
            "element_type": "line", "element_index": 0,
            "timestamp": "2000-01-03 19:00:00"}).json()
        assert r["feasible"] is False
        d = r["infeasibility_diagnosis"]
        assert d["solvable_with_grid_voltage_free"] is False and d["grid_voltage_needed_pu"] is None
        assert d["binding_limit"] == "beyond_voltage_and_branch_levers"
        assert "even both together" in r["message"]

    def test_line_loading_is_named(self, backend_client):
        """With branches allowed only 30 % of their rating (margin 0.3),
        loading is the binding limit and the branches carrying more than
        allowed are named. Judged against the OPF's own bound, not the rating:
        checking against 100 % of the rating found nothing and fell through to
        "beyond either lever"."""
        r = backend_client.post("/api/flexibility/optimize", json={
            "timestamp": "2000-01-04 03:15:00", "opf_current_safety_margin": 0.3}).json()
        assert r["feasible"] is False
        d = r["infeasibility_diagnosis"]
        assert d["binding_limit"] in ("branch_loading", "voltage_support_and_branch_loading")
        assert d["branch_overloads"] and all(
            o["loading_pct_of_rating"] > o["allowed_pct_of_rating"] for o in d["branch_overloads"])
        assert d["branch_overloads"][0]["allowed_pct_of_rating"] == pytest.approx(30.0, abs=0.1)
        assert "[L" in d["branch_overloads"][0]["branch"]  # the network's own branch name
        assert "line loading" in r["message"] or "overloaded" in r["message"]

    def test_a_feasible_solve_carries_no_diagnosis(self, backend_client):
        ts = mb.app_data["timestamps"][0]
        r = backend_client.post("/api/flexibility/optimize", json={"timestamp": ts}).json()
        assert r["feasible"] is True and "infeasibility_diagnosis" not in r


class TestOutagesAreSwitchedNotCut:
    """Contingency optimisation switches the element out of service, as the
    N-1 power flows do. It used to cut it out of the tables and renumber the
    buses: an islanding line and every transformer then crashed the solver with
    a 500 ("Solver crashed: ((3, 7), 1)") — code review 5.2. The IEEE 30 demo
    has no transformer, so one is added by a proposal (a 20 kV substation at
    Bus 9, the only way its new bus is supplied) and its outage studied there."""

    T = "2000-01-04 03:15:00"
    SUBSTATION = [{"op": "add_bus_with_transformer", "name": "DC 20 kV", "connect_to": "Bus 9", "vn_kv": 20,
                   "sn_mva": 40, "vk_percent": 12, "vkr_percent": 0.4, "parallel": 2},
                  {"op": "add_load", "bus": "DC 20 kV", "p_mw": 10, "pf": 0.98, "name": "Data centre"}]

    @pytest.mark.parametrize("element_type, index, network", [("line", 12, None), ("trafo", 0, "sub")])
    def test_the_outages_that_crashed_now_answer(self, backend_client, element_type, index, network):
        if network:
            assert backend_client.post("/api/proposals/edit", json={
                "proposal": network, "edits": self.SUBSTATION}).status_code == 200
        try:
            r = backend_client.post("/api/contingency/optimize", json={
                "element_type": element_type, "element_index": index, "timestamp": self.T,
                **({"network": network} if network else {})})
            assert r.status_code == 200, r.text
            assert r.json()["status"] in ("optimal", "infeasible", "maxIterations")
            if network:                              # the new bus hangs on the transformer alone
                assert r.json()["islanded_buses"] == ["DC 20 kV"] and r.json()["load_not_supplied_mw"] == 10.0
        finally:
            mb._discard_proposals()

    def test_an_islanding_outage_says_what_it_islands(self, backend_client):
        # Line 12 (Bus_8 -> Bus_10) is Bus_10's only branch: it carries Gen_bus10 and no load.
        r = backend_client.post("/api/contingency/optimize", json={
            "element_type": "line", "element_index": 12, "timestamp": self.T}).json()
        assert r["islanded_buses"] == ["Bus_10"]
        assert r["load_not_supplied_mw"] == 0.0     # bus 10 carries a unit, no load
        assert "Gen_bus10" not in {a["element"] for a in r["activated_resources"]}

    def test_a_meshed_outage_islands_nothing(self, backend_client):
        r = backend_client.post("/api/contingency/optimize", json={
            "element_type": "line", "element_index": 3, "timestamp": self.T}).json()
        assert r["islanded_buses"] == [] and r["load_not_supplied_mw"] == 0.0


class TestOneTapPositionEverywhere:
    """The OPF works on the network as loaded, transformer taps included
    (review 2.2). Its network equations always used the loaded taps (the
    admittance database is built from them), but the endpoints reset taps to
    neutral for the power flow that sets the starting point: the base dispatch
    and base voltages it reported, and departed from, belonged to a neutral-tap
    grid that the assessments never showed."""

    def test_its_base_point_is_the_one_the_assessment_shows(self, backend_client):
        ts = "2000-01-04 03:15:00"
        rsa = backend_client.post("/api/grid/rsa", json={"timestamp": ts}).json()
        opf = backend_client.post("/api/flexibility/optimize", json={"timestamp": ts}).json()
        assessed = {v["bus_name"] if "bus_name" in v else v["bus"]: v["vm_pu"] for v in rsa["all_voltages"]}
        base = {int(v["bus"]): v["vm_pu"] for v in opf["bus_voltages_base"]}
        assessed_by_index = {int(str(k).split("_")[-1]) if str(k).startswith("Bus_") else int(k): v
                             for k, v in assessed.items()}
        assert max(abs(base[b] - assessed_by_index[b]) for b in base) < 1e-4

    def test_a_dispatch_replays_on_the_network_as_loaded(self, backend_client):
        import copy
        import pandapower as pp
        import load_gen_assignment as lg
        ts = "2000-01-04 03:15:00"
        r = backend_client.post("/api/flexibility/optimize",
                                json={"timestamp": ts, "load_scaling_factor": 1.15}).json()
        assert r["feasible"] is True
        net = copy.deepcopy(mb.app_data["net"])            # taps as loaded, untouched
        meas = mb.prepare_measurement_df(mb._lookup_measurement_df(ts, "measurements"))
        net, _ = lg.assign_load_values_from_measurements(net, meas, mb.substations)
        net, _ = lg.assign_generators_values_from_measurements(net, meas, mb.substations)
        net.load["p_mw"] *= 1.15
        net.load["q_mvar"] *= 1.15
        volts = {int(v["bus"]): v["vm_pu"] for v in r["bus_voltages_post_opf"]}
        for a in r["activated_resources"]:
            rows = net.sgen.index[net.sgen["name"] == a["element"]].tolist()
            if rows:
                net.sgen.at[rows[0], "p_mw"] = a["Pg_new"]
                net.sgen.at[rows[0], "q_mvar"] = a["Qg_new"]
        for k, row in net.gen.iterrows():
            net.gen.at[k, "vm_pu"] = volts[int(row.bus)]
        net.ext_grid["vm_pu"] = volts[int(net.ext_grid.bus.iloc[0])]
        pp.runpp(net)
        assert max(abs(float(net.res_bus.vm_pu[b]) - v) for b, v in volts.items()) < 1e-3


def test_the_current_tap_policy_keeps_imported_ratios():
    """"current" set tap_neutral = tap_pos, which made every imported
    off-nominal ratio 1.0 (review 2.3)."""
    import pandapower.networks as pn
    # MATPOWER's case14: imported off-nominal ratios on its three tap-changing
    # transformers (the other two have no tap changer: both columns empty).
    net = pn.case14()
    tapped = net.trafo.tap_step_percent.notna()
    before = net.trafo[["tap_pos", "tap_neutral"]].copy()
    assert tapped.sum() == 3 and (before.tap_pos[tapped] != before.tap_neutral[tapped]).all()
    mb._apply_tap_policy(net, "current")
    assert net.trafo[["tap_pos", "tap_neutral"]].equals(before)
    mb._apply_tap_policy(net, "neutral")
    assert (net.trafo.tap_pos[tapped] == net.trafo.tap_neutral[tapped]).all()


def test_a_contingency_dispatch_can_be_replayed_from_the_result(backend_client):
    """The result reported active power only, and only for units whose active
    power moved: a unit that changed only its reactive output was missing, so
    the dispatch could not be checked against a power flow. Whatever line 7's
    outage at 02:15 moves, the reported dispatch must replay in pandapower
    (first observed on IEEE 14, where it moved only Gen_bus1's reactive output)."""
    import copy
    import pandapower as pp
    import pandapower.topology as top
    import load_gen_assignment as lg
    ts = "2000-01-03 02:15:00"
    r = backend_client.post("/api/contingency/optimize",
                            json={"element_type": "line", "element_index": 7, "timestamp": ts}).json()
    assert r["feasible"] is True
    assert all("Qg_new" in a for a in r["activated_resources"])
    net = copy.deepcopy(mb.app_data["net"])
    meas = mb.prepare_measurement_df(mb._lookup_measurement_df(ts, "measurements"))
    net, _ = lg.assign_load_values_from_measurements(net, meas, mb.substations)
    net, _ = lg.assign_generators_values_from_measurements(net, meas, mb.substations)
    net.line.at[7, "in_service"] = False
    for b in top.unsupplied_buses(net):
        net.bus.at[b, "in_service"] = False
    volts = {int(v["bus"]): v["vm_pu"] for v in r["bus_voltages_post_opf"]}
    for a in r["activated_resources"]:
        rows = net.sgen.index[net.sgen["name"] == a["element"]].tolist()
        if rows:
            net.sgen.at[rows[0], "p_mw"] = a["Pg_new"]
            net.sgen.at[rows[0], "q_mvar"] = a["Qg_new"]
    for k, row in net.gen.iterrows():
        net.gen.at[k, "vm_pu"] = volts[int(row.bus)]
    net.ext_grid["vm_pu"] = volts[int(net.ext_grid.bus.iloc[0])]
    pp.runpp(net)
    assert max(abs(float(net.res_bus.vm_pu[b]) - v) for b, v in volts.items()) < 1e-3


class TestScenarioRobustOPF:
    """Scenario robust OPF (code review 2.6)."""

    def _run(self, backend_client, monkeypatch, **body):
        seen = []
        real = mb.fe.scenario_optimization_model_base

        def spy(*args, **kwargs):
            seen.append(len(kwargs["load_multipliers"]))
            return real(*args, **kwargs)

        monkeypatch.setattr(mb.fe, "scenario_optimization_model_base", spy)
        r = backend_client.post("/api/flexibility/robust", json={
            "robust_method": "scenario", "timestamp": "2000-01-04 03:15:00", **body}).json()
        return r, seen

    def test_it_trains_on_the_scenarios_it_reports(self, backend_client, monkeypatch):
        """K scenarios were sliced from a pool of n_samples draws: with K above
        n_samples it trained on fewer than it reported."""
        r, seen = self._run(backend_client, monkeypatch, n_samples=10, n_scenarios=20,
                            validation_samples=10, load_sigma=0.03)
        assert seen and seen[0] == 20
        assert r["K"] == r["K_trained"] == 20

    def test_its_bound_is_called_nominal_not_certified(self, backend_client, monkeypatch):
        r, _ = self._run(backend_client, monkeypatch, n_samples=10, n_scenarios=12,
                         validation_samples=10, load_sigma=0.03)
        assert "not_certified" in r["guarantee_interpretation"]
        assert "not convex" in r["guarantee_note"] and "p_any_violation_after_validation" in r["guarantee_note"]


def test_a_scenario_run_reports_the_curtailment_it_relies_on(backend_client):
    """Code review 2.5. 2000-01-02 04:00 with 20 % wind and 5 % load
    uncertainty at 1.1× load: the plan holds back output at Gen_bus1, which the
    setpoint-only dispatch never showed. The IEEE 30 demo's units are
    synchronous machines; Gen_bus1 is made a wind unit at a fixed power factor
    (a static generator, declared wind) — then it curtails in every scenario."""
    backend_client.post("/api/generators", json={"changes": [{"unit": "Gen_bus1", "control": "power_factor"}]})
    backend_client.post("/api/network/declare", json={"units": [{"unit": "Gen_bus1", "source": "wind"}]})
    try:
        r = backend_client.post("/api/flexibility/robust", json={
            "robust_method": "scenario", "timestamp": "2000-01-02 04:00:00", "n_samples": 40,
            "n_scenarios": 40, "validation_samples": 40, "sgen_sigma": 0.2, "load_sigma": 0.05,
            "load_scaling_factor": 1.1}).json()
    finally:
        backend_client.post("/api/network/declare", json={"reset": True})
        backend_client.post("/api/generators", json={"reset": True})
    assert r["feasible"] is True
    c = r["scenario_curtailment"]
    assert c["n_scenarios"] == r["K_trained"] == 40
    assert c["scenarios_with_curtailment"] >= 1 and c["max_total_mw"] > 1.0
    assert c["by_unit"][0]["unit"] == "Gen_bus1"
    assert "curtailing renewable output" in r["message"]


def test_a_thermal_diagnosis_says_how_far_local_flexibility_gets(backend_client):
    """Line 0 out at an evening peak (IEEE 30 demo, 1.3× load): line 1 is at
    ~154 % at the least-redispatch solution, which the agent would read as the
    best possible. At best local flexibility brings it to ~122 %."""
    r = backend_client.post("/api/contingency/optimize", json={
        "element_type": "line", "element_index": 0, "timestamp": "2000-01-03 19:00:00",
        "load_scaling_factor": 1.3}).json()
    (worst, *_) = r["infeasibility_diagnosis"]["branch_overloads"]
    assert worst["branch"].endswith("[L1]")
    best = worst["best_achievable_pct_of_rating"]
    assert worst["allowed_pct_of_rating"] < best < worst["loading_pct_of_rating"]
    assert 119 < best < 125
    assert "At best, local flexibility brings" in r["message"]


class TestContingencyResultAt0200:
    """2000-01-01 02:00, line 1 out. Two reporting defects found on it (on the
    IEEE 14 case; the IEEE 30 demo is exporting a little at that hour)."""

    def _result(self, backend_client):
        return backend_client.post("/api/contingency/optimize", json={
            "element_type": "line", "element_index": 1, "timestamp": "2000-01-01 02:00:00"}).json()

    def test_the_grid_exchange_is_compared_with_this_hour(self, backend_client):
        """The external grid's "before" came from the stored network's startup
        power flow (PGLib's design load) rather than this hour's, so a small
        change read as a large one — and as load flexibility the OPF does not
        have. On the IEEE 30 demo: about 141 MW imported at startup, −1.67 MW
        (an export) at this hour after the outage."""
        r = self._result(backend_client)
        assert r["feasible"] is True
        (ext,) = [a for a in r["activated_resources"] if a["type"] == "External Grid"]
        assert ext["Pg_base"] == pytest.approx(-1.6664, abs=0.05)  # post-outage, before redispatch
        assert abs(ext["Pg_base"] - float(mb.app_data["net"].res_ext_grid.p_mw.sum())) > 100   # not startup's
        assert abs(ext["Pg_new"] - ext["Pg_base"]) < 1.0

    def test_every_bus_has_its_own_label(self, backend_client):
        """PGLib's buses have no names; str(None) labelled all of them "None"
        and the voltage chart drew every bus at one x position."""
        for r in (self._result(backend_client),
                  backend_client.post("/api/flexibility/optimize",
                                      json={"timestamp": "2000-01-01 02:00:00"}).json()):
            names = [v["bus_name"] for v in r["bus_voltages_post_opf"]]
            assert "None" not in names and len(set(names)) == len(names)
            assert names[0] == "Bus_0"

    def test_each_unit_names_its_bus(self, backend_client):
        """The answer called Gen_bus7 "Bus 8": unit and substation names
        number buses differently, and the result did not say which bus."""
        r = self._result(backend_client)
        buses = {a["element"]: a["bus"] for a in r["activated_resources"]}
        assert buses["Gen_bus7"] == "Bus_7" and buses["ExtGrid_0"] == "Bus_0"
        units = {e: b for e, b in buses.items() if e.startswith("Gen_bus")}
        assert units and all(b == f"Bus_{e.removeprefix('Gen_bus')}" for e, b in units.items())


class TestRobustUsesTheLoadedNetworksLimits:
    """The robust request declares 0.95–1.05 p.u. as its defaults, but a band
    the caller did not send is the loaded network's own — and follows a new
    network, because it is read from it on every request."""

    BODY = {"robust_method": "heuristic", "timestamp": "2000-01-04 03:15:00",
            "n_samples": 10, "validation_samples": 10, "max_iter": 1}

    def _band(self, backend_client, **extra):
        r = backend_client.post("/api/flexibility/robust", json={**self.BODY, **extra}).json()
        return r["opf_vm_lower_used"], r["opf_vm_upper_used"]

    def test_unset_is_the_networks_band(self, backend_client):
        # The IEEE 30 demo's band equals the request's declared defaults; the
        # next test gives the network another band, which must be followed.
        assert self._band(backend_client) == (mb.app_data["default_vm_lower"],
                                              mb.app_data["default_vm_upper"]) == (0.95, 1.05)

    def test_a_new_networks_band_is_followed(self, backend_client, monkeypatch):
        monkeypatch.setitem(mb.app_data, "default_vm_lower", 0.92)   # as an upload sets them
        monkeypatch.setitem(mb.app_data, "default_vm_upper", 1.08)
        assert self._band(backend_client) == (0.92, 1.08)

    def test_an_explicit_band_wins(self, backend_client):
        assert self._band(backend_client, vm_lower_pu=0.95, vm_upper_pu=1.05) == (0.95, 1.05)
