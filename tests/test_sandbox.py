"""
Network proposals: the base network plus a list of edits (sandbox, phase 3 of
`sandbox_and_recommender.md`).

The properties that make a proposal trustworthy:
- the base is never modified, and base indices stay the same in a proposal;
- every element an edit adds injects what its edit says at every tick —
  measurement assignment must not overwrite it (hazard A.3 (1));
- every result computed on a proposal says so;
- base and proposal are compared at one operating point, in one call.
"""

import copy
import json
import shutil
import types

import pandapower as pp
import pandapower.toolbox  # noqa: F401 — pp.toolbox.nets_equal
import pandapower.networks as pn
import pandas as pd
import pytest

import flex_engine as fe
import network_loader as nl
import sandbox as sb


@pytest.fixture
def net():
    """IEEE 14-bus without bus names, as the backend's PGLib profile loads it
    (pandapower's own copy names buses 1–14, one off their indices)."""
    n = pn.case14()
    n.bus["name"] = None
    return n


def _meas(cons: dict, prod: dict | None = None) -> pd.DataFrame:
    prod = prod or {}
    keys = sorted(set(cons) | set(prod))
    return pd.DataFrame({"substation_name": keys,
                         "consumption": [cons.get(k, 0.0) for k in keys],
                         "production": [prod.get(k, 0.0) for k in keys]})


DC = {"op": "add_load", "bus": "Bus 9", "p_mw": 50, "pf": 0.98, "name": "Data centre"}


class TestMaterialise:
    def test_the_base_is_untouched_and_indices_are_kept(self, net):
        before = copy.deepcopy(net)
        prop, edits = sb.materialise(net, [DC, {"op": "set_in_service", "element_type": "line",
                                                "element": 3, "in_service": False}])
        assert pp.toolbox.nets_equal(net, before)
        assert list(prop.line.index) == list(net.line.index)
        assert list(prop.load.index[:len(net.load)]) == list(net.load.index)
        assert len(prop.load) == len(net.load) + 1 and not prop.line.at[3, "in_service"]
        added = prop.load.iloc[-1]
        assert added["proposed"] and added["name"] == "Data centre" and int(added["bus"]) == 9
        assert edits[0]["bus"] == 9 and edits[0]["profile"] == "flat"

    def test_reactive_power_follows_the_edits_power_factor(self, net):
        prop, _ = sb.materialise(net, [DC])
        q = prop.load.iloc[-1]["q_mvar"]
        assert q == pytest.approx(50 * (1 / 0.98 ** 2 - 1) ** 0.5)

    @pytest.mark.parametrize("edit, fragment", [
        ({**DC, "bus": "Bus 99"}, "no bus"),
        ({**DC, "p_mw": 0}, "above 0"),
        ({**DC, "p_mw": 1e6}, "at most"),
        ({**DC, "name": ""}, "name is required"),
        ({**DC, "profile": "weekly"}, "profile must be"),
        ({**DC, "profile": "shift"}, "start_hour is required"),
        ({**DC, "pf": 1.2}, "at most 1.0"),
        ({"op": "add_generator", "bus": 3, "p_max_mw": 5, "kind": "nuclear", "name": "G"}, "What kind of unit"),
        ({"op": "add_generator", "bus": 3, "p_max_mw": 5, "kind": "dispatchable", "p_mw": 9,
          "name": "G"}, "at most 5"),
        ({"op": "reinforce_line", "line": 3, "factor": 1.0}, "above 1.0"),
        ({"op": "reinforce_line", "line": "nowhere", "factor": 2}, "no line"),
        ({"op": "set_in_service", "element_type": "bus", "element": 3, "in_service": False}, "element_type"),
        ({"op": "set_in_service", "element_type": "line", "element": 3, "in_service": "no"}, "true or false"),
        ({"op": "delete_line", "line": 3}, "Unknown edit"),
    ])
    def test_invalid_edits_are_refused_with_the_reason(self, net, edit, fragment):
        with pytest.raises(sb.EditError, match=fragment):
            sb.materialise(net, [edit])

    def test_names_must_be_unique_including_earlier_edits(self, net):
        with pytest.raises(sb.EditError, match="Edit 2: .*already used"):
            sb.materialise(net, [DC, {**DC, "bus": 4}])

    def test_lines_are_found_by_the_names_results_use(self, net):
        from element_names import line_display_name
        label = line_display_name(net, 12)
        assert label == "Bus_9 -> Bus_10 [L12]"
        _, edits = sb.materialise(net, [{"op": "reinforce_line", "line": label, "factor": 2}])
        assert edits[0]["line"] == 12

    def test_the_hash_identifies_the_edit_list(self, net):
        _, a = sb.materialise(net, [DC])
        _, b = sb.materialise(net, [dict(DC)])
        _, c = sb.materialise(net, [{**DC, "p_mw": 51}])
        assert sb.edits_hash(a) == sb.edits_hash(b) != sb.edits_hash(c)


class TestReinforceLine:
    """×2 is two identical circuits: the same admittances and limits as a
    line with `parallel = 2` (the representation phase 0 made exact)."""

    def test_admittances_match_two_parallel_circuits(self, net):
        doubled = copy.deepcopy(net)
        doubled.line.at[12, "parallel"] = 2
        reinforced, _ = sb.materialise(net, [{"op": "reinforce_line", "line": 12, "factor": 2}])
        a, b = (nl.build_ybus_full(n) for n in (doubled, reinforced))
        i, j = int(net.line.from_bus[12]), int(net.line.to_bus[12])
        for k in ("Yff_r", "Yff_i", "Yft_r", "Yft_i"):
            assert a[k][(i, j), 1] == pytest.approx(b[k][(i, j), 1])
        assert fe._branch_current_data(reinforced)[(i, j)]["I_from_max"] == pytest.approx(
            fe._branch_current_data(doubled)[(i, j)]["I_from_max"])


class TestProfiles:
    PEAKS = {"consumption": {"Bus_9": 20.0}, "production": {"Bus_1": 40.0}, "total_production": 40.0}

    def _report(self, net, edits, ts="2000-01-03 12:00:00", meas=None):
        prop, _ = sb.materialise(net, edits)
        meas = _meas({"Bus_9": 10.0}, {"Bus_1": 30.0}) if meas is None else meas
        return prop, sb.apply_profiles(prop, ts, meas, self.PEAKS)

    def test_flat(self, net):
        _, rep = self._report(net, [DC])
        assert rep[0]["p_mw"] == 50

    def test_follow_bus_scales_to_the_buss_peak(self, net):
        _, rep = self._report(net, [{**DC, "profile": "follow_bus"}])
        assert rep[0]["p_mw"] == pytest.approx(25.0)      # 10 of a 20 MW peak → half of 50
        assert rep[0]["source"] == "demand of Bus_9"

    @pytest.mark.parametrize("hours, ts, on", [
        ((8, 18), "2000-01-03 12:00:00", True), ((8, 18), "2000-01-03 18:00:00", False),
        ((22, 6), "2000-01-03 23:15:00", True), ((22, 6), "2000-01-03 05:45:00", True),
        ((22, 6), "2000-01-03 12:00:00", False), ((6, 24), "2000-01-03 23:45:00", True),
    ])
    def test_shift(self, net, hours, ts, on):
        edit = {**DC, "profile": "shift", "start_hour": hours[0], "end_hour": hours[1]}
        _, rep = self._report(net, [edit], ts=ts)
        assert rep[0]["p_mw"] == (50 if on else 0)

    def test_variable_generation_follows_measured_production_and_caps_the_optimiser(self, net):
        wind = {"op": "add_generator", "bus": 9, "p_max_mw": 20, "kind": "wind", "name": "Wind 9"}
        prop, rep = self._report(net, [wind])
        assert rep[0]["p_mw"] == pytest.approx(15.0)       # 30 of 40 MW → 75 % of 20
        assert rep[0]["source"] == "total measured production"  # Bus_9 produces nothing
        row = prop.sgen.iloc[-1]
        assert row["max_p_mw"] == pytest.approx(15.0)      # curtail, never exceed the wind

    def test_a_unit_on_a_producing_bus_follows_that_bus(self, net):
        pv = {"op": "add_generator", "bus": 1, "p_max_mw": 10, "kind": "pv", "name": "PV 1"}
        _, rep = self._report(net, [pv], meas=_meas({"Bus_9": 10.0}, {"Bus_1": 20.0, "Bus_2": 20.0}))
        assert rep[0]["p_mw"] == pytest.approx(5.0) and rep[0]["source"] == "measured production of Bus_1"

    def test_dispatchable_keeps_its_set_point_and_capacity(self, net):
        gen = {"op": "add_generator", "bus": 9, "p_max_mw": 20, "kind": "dispatchable", "p_mw": 8, "name": "CHP",
               "control": "power_factor"}
        prop, rep = self._report(net, [gen])
        assert rep[0]["p_mw"] == pytest.approx(8.0) and prop.sgen.iloc[-1]["max_p_mw"] == 20

    def test_a_profile_that_cannot_be_followed_is_an_error_not_zero(self, net):
        with pytest.raises(sb.EditError, match="no measured demand"):
            self._report(net, [{**DC, "bus": 7, "profile": "follow_bus"}])
        wind = {"op": "add_generator", "bus": 9, "p_max_mw": 20, "kind": "wind", "name": "W"}
        prop, _ = sb.materialise(net, [wind])
        with pytest.raises(sb.EditError, match="has none"):
            sb.apply_profiles(prop, "2000-01-03 12:00:00", _meas({"Bus_9": 1.0}),
                              {"consumption": {}, "production": {}, "total_production": 0.0})

    def test_the_check_catches_an_overwritten_element(self, net):
        prop, rep = self._report(net, [DC])
        assert sb.check_profiles(prop, rep) == []
        prop.load.iloc[-1, prop.load.columns.get_loc("p_mw")] = 3.0   # what the old assignment did
        assert sb.check_profiles(prop, rep) == ["load 'Data centre': 3.0000 MW instead of 50.0000 MW"]

    def test_peaks_over_a_dataset(self):
        ds = {"t1": _meas({"A": 1.0}, {"A": -2.0}), "t2": _meas({"A": 3.0}, {"A": -1.0, "B": -4.0})}
        peaks = sb.dataset_peaks(ds)
        assert peaks["consumption"]["A"] == 3.0 and peaks["production"]["B"] == 4.0
        assert peaks["total_production"] == 5.0


class TestMeasuredSubstationNetworks:
    """On a network whose generation is rebuilt from the measurements every
    tick (no `gen` table), an added generator survives the rebuild and is
    then set by its profile."""

    def test_added_unit_survives_the_sgen_rebuild(self):
        import load_gen_assignment as lg
        net = pp.create_empty_network()
        b0 = pp.create_bus(net, 20, name="Grid")
        b1 = pp.create_bus(net, 20, name="Alpha")
        pp.create_ext_grid(net, b0)
        pp.create_line(net, b0, b1, 5, "NA2XS2Y 1x95 RM/25 12/20 kV")
        pp.create_load(net, b1, p_mw=2.0, name="Alpha Load")
        gen = {"op": "add_generator", "bus": "Alpha", "p_max_mw": 3, "kind": "dispatchable",
               "p_mw": 1.5, "name": "Battery", "control": "power_factor"}
        prop, _ = sb.materialise(net, [gen])
        meas = _meas({"Alpha": 2.5}, {"Alpha": -0.7})
        prop, _ = lg.assign_load_values_from_measurements(prop, meas, ["Alpha"])
        prop, _ = lg.assign_generators_values_from_measurements(prop, meas, ["Alpha"])
        report = sb.apply_profiles(prop, "2000-01-01 10:00:00", meas, sb.dataset_peaks({"t": meas}))
        assert sb.check_profiles(prop, report) == []
        assert sorted(prop.sgen["name"]) == ["Alpha Sgen", "Battery"]
        assert float(prop.sgen.loc[prop.sgen.name == "Battery", "p_mw"].iloc[0]) == pytest.approx(1.5)


# ---------------------------------------------------------------------------
# Through the backend
# ---------------------------------------------------------------------------

@pytest.fixture
def client(backend_client):
    import main_backend as mb
    yield backend_client
    mb._discard_proposals()


def _create(client, name="dc_bus9", edits=None):
    r = client.post("/api/proposals/edit", json={"proposal": name, "edits": edits or [
        {**DC, "profile": "follow_bus"},
        {"op": "add_generator", "bus": 9, "p_max_mw": 20, "kind": "wind", "name": "Wind 9"}]})
    assert r.status_code == 200, r.text
    return r.json()


class TestProposalEndpoints:
    def test_create_list_and_discard(self, client):
        out = _create(client)
        assert out["status"] == "created" and out["summary"]["added_load_mw"] == 50
        assert out["profiles_checked"]
        assert [p["id"] for p in client.get("/api/proposals").json()["proposals"]] == ["dc_bus9"]
        more = client.post("/api/proposals/edit", json={"proposal": "dc_bus9", "edits": [
            {"op": "reinforce_line", "line": 12, "factor": 2}]}).json()
        assert more["status"] == "updated" and more["summary"]["n_edits"] == 3
        less = client.delete("/api/proposals/dc_bus9/edits/1").json()
        assert [e["op"] for e in less["edits"]] == ["add_generator", "reinforce_line"]
        assert client.delete("/api/proposals/dc_bus9").json()["remaining"] == []

    def test_nothing_is_stored_when_an_edit_fails(self, client):
        r = client.post("/api/proposals/edit", json={"proposal": "bad", "edits": [
            DC, {**DC, "bus": "Bus 99", "name": "x"}]})
        assert r.status_code == 400 and "Edit 2" in r.json()["detail"]
        r = client.post("/api/proposals/edit", json={"proposal": "bad", "edits": [
            {**DC, "bus": 7, "profile": "follow_bus"}]})
        assert r.status_code == 422 and "no measured demand" in r.json()["detail"]
        assert client.get("/api/proposals").json()["proposals"] == []

    @pytest.mark.parametrize("name", ["base", "", "has space", "x" * 49, "-lead"])
    def test_ids_are_checked(self, client, name):
        r = client.post("/api/proposals/edit", json={"proposal": name, "edits": [DC]})
        assert r.status_code == 400

    def test_replacing_the_base_discards_every_proposal(self, client):
        import main_backend as mb
        _create(client)
        # The reload also empties the shared tables the later tests read.
        saved = (dict(mb.app_data), list(mb.substations), dict(mb.PG_MAX_DATA))
        try:
            mb._prepare_runtime_reload()
            assert mb.app_data["proposals"] == {} and mb.app_data["_proposal_cache"] == {}
        finally:
            mb.app_data.update(saved[0])
            mb.substations[:] = saved[1]
            mb.PG_MAX_DATA.clear()
            mb.PG_MAX_DATA.update(saved[2])


class TestStudiesOnAProposal:
    def test_the_base_answers_as_before(self, client):
        before = client.post("/api/grid/rsa", json={}).json()
        _create(client)
        after = client.post("/api/grid/rsa", json={}).json()
        assert after == before and "network" not in after

    def test_a_result_on_a_proposal_says_so(self, client):
        base = client.post("/api/grid/rsa", json={}).json()
        _create(client)
        prop = client.post("/api/grid/rsa", json={"network": "dc_bus9"}).json()
        card = prop["network"]
        assert card["id"] == "dc_bus9" and len(card["edits"]) == 2 and card["edits_hash"]
        added = {e["element"]: e for e in card["added_elements_at_tick"]["elements"]}
        assert set(added) == {"Data centre", "Wind 9"}
        assert prop["total_load_mw"] == pytest.approx(base["total_load_mw"] + added["Data centre"]["p_mw"], abs=1e-3)

    def test_an_unknown_network_is_refused(self, client):
        r = client.post("/api/grid/rsa", json={"network": "nope"})
        assert r.status_code == 400 and "nope" in r.json()["detail"]

    def test_the_snapshot_lists_the_added_unit_with_its_available_power(self, client):
        _create(client)
        gens = client.post("/api/network/snapshot", json={"network": "dc_bus9"}).json()["generators"]
        wind = next(g for g in gens if g["name"] == "Wind 9")
        assert wind["bus"] == "Bus_9" and wind["Pg_max_mw"] == pytest.approx(wind["Pg_mw"])

    def test_scans_follow_the_profile_at_every_tick(self, client):
        _create(client, edits=[{**DC, "p_mw": 30, "profile": "shift", "start_hour": 8, "end_hour": 18}])
        body = {"metric": "slack_import", "n_steps": 96, "step_size": 4}
        base = client.post("/api/rsa/worst_case", json=body).json()
        prop = client.post("/api/rsa/worst_case", json={**body, "network": "dc_bus9"}).json()
        assert base["series"]["timestamps"] == prop["series"]["timestamps"]
        hours = [int(t[11:13]) for t in prop["series"]["timestamps"]]
        b, p = base["series"]["slack_import_mw"], prop["series"]["slack_import_mw"]
        assert len(b) == len(p) == 24 and len(set(hours)) > 12
        for h, vb, vp in zip(hours, b, p):
            assert (vp - vb > 25) == (8 <= h < 18), (h, vb, vp)

    def test_an_opened_line_is_not_tested_again_nor_put_back(self, client):
        _create(client, "l3_open", [{"op": "set_in_service", "element_type": "line", "element": 3,
                                     "in_service": False}])
        base = client.post("/api/contingency/simulate_all", json={}).json()
        prop = client.post("/api/contingency/simulate_all", json={"network": "l3_open"}).json()
        assert prop["total_outages_tested"] == base["total_outages_tested"] - 1
        assert not any(v.get("outage_cause", "").endswith("[L3]") for v in prop["violations"])


class TestCompare:
    def test_one_operating_point_and_the_difference(self, client):
        _create(client)
        r = client.post("/api/proposals/compare", json={"proposal": "dc_bus9", "timestamp": "2000-01-02 18:00"})
        assert r.status_code == 200, r.text
        out = r.json()
        assert out["timestamp"].startswith("2000-01-02 18:00") and out["study"] == "rsa"
        assert out["network"]["compared_with"] == "base"
        dc = next(e for e in out["network"]["added_elements_at_tick"]["elements"] if e["element"] == "Data centre")
        assert out["proposal"]["total_load_mw"] - out["base"]["total_load_mw"] == pytest.approx(dc["p_mw"], abs=1e-3)
        worst = out["largest_voltage_changes"][0]
        assert worst["element"] == "Bus_9" and worst["change"] < 0

    def test_n1(self, client):
        _create(client)
        out = client.post("/api/proposals/compare", json={"proposal": "dc_bus9", "study": "n1"}).json()
        assert out["base"]["total_outages_tested"] == out["proposal"]["total_outages_tested"]
        assert {"outage", "type", "element"} <= set((out["new_violations"] or [{"outage": 1, "type": 1, "element": 1}])[0])

    def test_it_needs_a_proposal(self, client):
        assert client.post("/api/proposals/compare", json={"proposal": "base"}).status_code == 400
        assert client.post("/api/proposals/compare", json={"proposal": "x"}).status_code == 400
        _create(client)
        assert client.post("/api/proposals/compare", json={"proposal": "dc_bus9", "study": "load_flow"}).status_code == 400


@pytest.mark.skipif(shutil.which("ipopt") is None, reason="IPOPT not installed")
def test_the_optimiser_may_curtail_an_added_wind_unit_but_not_exceed_the_wind(client):
    _create(client)
    out = client.post("/api/flexibility/optimize", json={"network": "dc_bus9"}).json()
    assert out["feasible"] and out["network"]["id"] == "dc_bus9"
    avail = next(e["p_mw"] for e in out["network"]["added_elements_at_tick"]["elements"]
                 if e["element"] == "Wind 9")
    wind = next(r for r in out["activated_resources"] if r["element"] == "Wind 9")
    assert wind["Pg_base"] == pytest.approx(avail, abs=1e-3)
    assert wind["Pg_new"] <= avail + 1e-3


# ---------------------------------------------------------------------------
# The agent's guards
# ---------------------------------------------------------------------------

class TestAgentGuards:
    def _prop(self, **extra):
        return {"timestamp": "2000-01-01 00:00:00", "data_source": "measurements",
                "thresholds_used": {}, "network": {"id": "dc_bus9", "edits": ["Add load 'DC'"]}, **extra}

    def _base(self):
        return {"timestamp": "2000-01-01 00:00:00", "data_source": "measurements", "thresholds_used": {}}

    def test_a_turn_inherits_the_network(self):
        from llm_agent.agent.validators import TurnOperatingPoint
        top = TurnOperatingPoint()
        top.resolve("run_rsa", {"network": "dc_bus9"}, {"network", "timestamp"})
        kwargs, inherited = top.resolve("simulate_all_contingencies", {}, {"network"})
        assert kwargs["network"] == "dc_bus9" and inherited["network"]["from"] == "run_rsa"
        kwargs, _ = top.resolve("edit_proposal", {"proposal": "x"}, {"proposal", "edits"})
        assert "network" not in kwargs

    def test_mixing_base_and_proposal_is_flagged_unless_asked(self):
        from llm_agent.agent.validators import TurnConsistency, compares_networks
        tc = TurnConsistency()
        assert tc.observe("run_rsa", self._base()) == []
        issues = tc.observe("simulate_all_contingencies", self._prop())
        assert [i.field for i in issues] == ["network"]
        asked = "Would the data centre proposal cause overloads compared with today?"
        assert compares_networks(asked) and not compares_networks("Run an N-1 screen")
        tc = TurnConsistency(networks_compared=True)
        tc.observe("run_rsa", self._base())
        assert tc.observe("run_rsa", self._prop()) == []

    def test_a_comparison_is_not_one_network(self):
        from llm_agent.agent.validators import operating_context
        cmp_ = self._prop()
        cmp_["network"]["compared_with"] = "base"
        assert "network" not in operating_context(cmp_)
        assert operating_context(self._base())["network"] == "base"

    def test_an_answer_that_does_not_name_its_proposal_gets_a_note(self):
        from llm_agent.agent.validators import proposal_note
        results = [{"name": "run_rsa", "result": self._prop()}, {"name": "run_rsa", "result": self._base()}]
        note = proposal_note("The grid stays secure.", results)
        assert "dc_bus9" in note and "Add load 'DC'" in note and "not the loaded network" in note
        assert proposal_note("On proposal dc_bus9 the grid stays secure.", results) is None
        # One proposal in the turn: "the proposal" can only mean it.
        assert proposal_note("With the proposal the grid stays secure.", results) is None
        two = results + [{"name": "run_rsa", "result": {**self._prop(), "network": {"id": "alt", "edits": ["x"]}}}]
        note = proposal_note("The proposal is secure; dc_bus9 is not.", two)
        assert "**alt**" in note and "dc_bus9" not in note
        assert proposal_note("Secure.", [{"name": "run_rsa", "result": self._base()}]) is None
        failed = [{"name": "run_rsa", "result": {**self._prop(), "error": "x"}}]
        assert proposal_note("Failed.", failed) is None


class TestComparisonChart:
    def test_rsa_and_n1_charts_name_the_proposal(self):
        pytest.importorskip("plotly")
        from llm_agent.agent.renderers import render_network_comparison
        rsa = {"study": "rsa", "timestamp": "2000-01-01 00:00:00", "network": {"id": "dc_bus9"},
               "largest_voltage_changes": [{"element": "Bus_9", "base": 1.0, "proposal": 0.98, "change": -0.02}],
               "largest_line_loading_changes": [{"element": "L1", "base": 40, "proposal": 46, "change": 6}],
               "largest_trafo_loading_changes": []}
        figs = render_network_comparison(rsa)
        assert len(figs) == 2 and all("dc_bus9 − base at 2000-01-01 00:00" in f.layout.title.text for f in figs)
        n1 = {"study": "n1", "timestamp": "2000-01-01 00:00:00", "network": {"id": "dc_bus9"},
              "new_violations": [{"outage": "L1", "type": "bus_vm_pu", "element": "Bus_3"}],
              "resolved_violations": [{"outage": "L7", "type": "line_loading", "element": "L8"}]}
        (fig,) = render_network_comparison(n1)
        assert [list(t.x) for t in fig.data] == [[0, 1], [-1, 0]]
        same = render_network_comparison({**n1, "new_violations": [], "resolved_violations": []})
        assert "Same N-1 violations" in same[0].layout.title.text


def test_a_scripted_turn_carries_the_proposal_and_names_it(client, monkeypatch):
    """The agent loop end to end, with a scripted model instead of an API:
    create a proposal, assess it, run N-1 without repeating `network` — the
    turn's network is inherited — then answer without naming the proposal,
    which earns the answer a note saying which grid the figures are from."""
    from llm_agent.agent import loop, tools
    from llm_agent.agent.providers.base import ModelResponse, ToolCall

    def post(endpoint, body):
        body = {k: v for k, v in body.items() if v is not None}
        r = client.post(endpoint, json=body)
        return r.json() if r.status_code < 400 else {"error": r.text, "status_code": r.status_code}

    script = [
        [ToolCall("edit_proposal", {"proposal": "dc_bus9", "edits": [{**DC, "profile": "follow_bus"}]})],
        [ToolCall("run_rsa", {"network": "dc_bus9"})],
        [ToolCall("simulate_all_contingencies", {})],
        "The grid stays secure in N-0.",
    ]

    class Scripted:
        def chat(self, messages, tools, system, on_event=None):
            step = script.pop(0)
            return ModelResponse(text=step) if isinstance(step, str) else ModelResponse(text="", tool_calls=step)

    written = []
    monkeypatch.setattr(tools, "_post", post)
    monkeypatch.setattr(loop, "get_provider", lambda: Scripted())
    monkeypatch.setattr(loop, "describe_provider", lambda p: {"provider": "scripted"})
    monkeypatch.setattr(loop, "get_system_prompt", lambda: "test")
    monkeypatch.setattr(loop, "_append_to_log", written.append)

    text, _ = loop.run_agent_turn("Is the grid secure with the data centre?", [], reflection="off")
    results = {r["name"]: r["result"] for r in written[0]["tool_results"]}
    assert results["edit_proposal"]["status"] == "created"
    assert results["simulate_all_contingencies"]["network"]["id"] == "dc_bus9"   # inherited
    drift = [w for r in results.values() for w in r.get("_integrity_warnings", []) if w["field"] == "network"]
    assert drift == []                                                           # one network, no drift
    assert text.startswith("The grid stays secure in N-0.")
    assert "Results above are for proposal **dc_bus9**" in text


class TestSwitchedOff:
    """`set_in_service` is honoured by every study, base and proposal alike."""

    def test_a_measured_unit_switched_off_stays_off_through_the_rebuild(self):
        import load_gen_assignment as lg
        net = pp.create_empty_network()
        b0, b1 = pp.create_bus(net, 20, name="Grid"), pp.create_bus(net, 20, name="Alpha")
        pp.create_ext_grid(net, b0)
        pp.create_line(net, b0, b1, 5, "NA2XS2Y 1x95 RM/25 12/20 kV")
        pp.create_load(net, b1, p_mw=2.0, name="Alpha Load")
        meas = _meas({"Alpha": 2.5}, {"Alpha": -0.7})
        net, _ = lg.assign_load_values_from_measurements(net, meas, ["Alpha"])
        net, _ = lg.assign_generators_values_from_measurements(net, meas, ["Alpha"])  # base, as loaded
        prop, _ = sb.materialise(net, [{"op": "set_in_service", "element_type": "sgen",
                                        "element": "Alpha Sgen", "in_service": False}])
        for _ in range(2):  # two ticks
            prop, _ = lg.assign_load_values_from_measurements(prop, meas, ["Alpha"])
            prop, _ = lg.assign_generators_values_from_measurements(prop, meas, ["Alpha"])
        assert not bool(prop.sgen.loc[prop.sgen.name == "Alpha Sgen", "in_service"].iloc[0])

    def test_results_mark_what_is_switched_off(self, client):
        _create(client, "off", [
            {"op": "set_in_service", "element_type": "sgen", "element": "Gen_bus1", "in_service": False},
            {"op": "set_in_service", "element_type": "line", "element": 3, "in_service": False},
            {"op": "set_in_service", "element_type": "load", "element": 1, "in_service": False}])
        rsa = client.post("/api/grid/rsa", json={"network": "off"}).json()
        line = next(x for x in rsa["all_line_loading"] if x["line_name"].endswith("[L3]"))
        assert line["in_service"] is False and line["loading_percent"] == 0
        assert "Gen_bus1" not in [g["name"] for g in rsa["gen_dispatch"]]
        snap = client.post("/api/network/snapshot", json={"network": "off"}).json()
        unit = next(g for g in snap["generators"] if g["name"] == "Gen_bus1")
        assert unit["in_service"] is False and unit["Pg_mw"] == 0
        load = next(x for x in snap["loads"] if x["bus_index"] == 2)
        assert load["in_service"] is False and load["P_mw"] == 0
        base = client.post("/api/grid/rsa", json={}).json()
        assert all("in_service" not in x for x in base["all_line_loading"])

    def test_the_rsa_names_units_as_the_optimiser_does(self, client):
        rsa = client.post("/api/grid/rsa", json={"timestamp": "2000-01-03 12:00"}).json()
        names = [g["name"] for g in rsa["gen_dispatch"]]
        assert "Gen_bus1" in names and "0" not in names
        gen1 = next(g for g in rsa["gen_dispatch"] if g["name"] == "Gen_bus1")
        assert rsa["flexibility_metrics"]["total_available_capacity_mw"] == pytest.approx(59.0 - gen1["Pg_mw"], abs=1e-2)


class TestDiffMap:
    def _maps(self):
        from llm_agent.agent.network_map import Edge, NetworkMap, Node
        def make(v9, load9, l1_on, l1, l12):
            nodes = {i: Node(i, f"Bus_{i}", 20.0, True, "substation") for i in (0, 1, 9)}
            nodes[0].vm_pu, nodes[1].vm_pu, nodes[9].vm_pu = 1.0, 0.99, v9
            nodes[9].load_mw = load9
            edges = [Edge(1, "L1", 0, 1, "line", l1_on, l1), Edge(12, "L12", 1, 9, "line", True, l12)]
            return NetworkMap("Grid", nodes, edges)
        return make(1.0, 10.0, True, 40.0, 10.0), make(0.93, 70.0, False, 0.0, 60.0)

    def test_the_difference_is_what_is_drawn(self):
        pytest.importorskip("plotly")
        from llm_agent.agent.renderers import render_network_diff
        base, prop = self._maps()
        pos = {0: (0.0, 0.0), 1: (1.0, 0.0), 9: (2.0, 0.0)}
        fig = render_network_diff(base, prop, pos, new_violations={"Bus_9"}, subtitle="x")
        hover = [h for d in fig.data for h in (d.hovertext or []) if isinstance(h, str)]
        assert any("L1</b><br>line · opened by the proposal" in h for h in hover)
        assert any("10.0% → 60.0% (+50.0 pts)" in h for h in hover)
        assert any("1.0000 → 0.9300 pu (-0.0700)" in h and "newly violating" in h for h in hover)
        added = next(d for d in fig.data if d.name == "added demand")
        assert list(added.hovertext) == ["<b>Bus_9</b><br>added demand: +60.00 MW"]
        assert "difference from the base" in fig.layout.title.text

    def test_an_open_branch_in_a_result_is_drawn_open(self):
        from llm_agent.agent import network_map as nm
        base, _ = self._maps()
        nm.apply_assessment(base, {"all_line_loading": [{"line_name": "L1", "loading_percent": 0.0,
                                                         "in_service": False}]})
        assert next(e for e in base.edges if e.name == "L1").in_service is False


class TestRecommender:
    """The suggestion agent sees which network each turn studied, and the
    button lights up when a proposal was judged on less than a planner would
    accept."""

    def _record(self, *calls):
        return {"user": "What if we add a data centre at Bus 9?",
                "tool_calls": [{"name": n, "args": a} for n, a, _ in calls],
                "tool_results": [{"name": n, "result": r} for n, _, r in calls]}

    NET = {"id": "dc_bus9", "edits": ["Add load 'Data centre': 50 MW at Bus_9"]}

    def test_a_single_moment_compare_is_flagged_with_its_new_violations(self):
        from llm_agent.agent.recommender import needs_attention, proposal_gaps
        rec = self._record(("compare_networks", {"proposal": "dc_bus9"},
                            {"study": "rsa", "network": {**self.NET, "compared_with": "base"},
                             "new_violations": [{"type": "bus_vm_pu", "element": "Bus_9"}]}))
        gaps = proposal_gaps(rec)
        assert gaps == ["proposal dc_bus9: 1 violation(s) the base does not have",
                        "proposal dc_bus9: judged at a single moment — no study over time",
                        "proposal dc_bus9: no N-1 screen of the proposal in this turn"]
        assert set(gaps) <= set(needs_attention(rec))

    def test_a_thorough_study_is_not_flagged(self):
        from llm_agent.agent.recommender import proposal_gaps
        rec = self._record(
            ("find_worst_case_timestamp", {"network": "dc_bus9"}, {"network": self.NET, "worst_value": 3}),
            ("compare_networks", {"proposal": "dc_bus9", "study": "n1"},
             {"study": "n1", "network": {**self.NET, "compared_with": "base"}, "new_violations": []}))
        assert proposal_gaps(rec) == []
        assert proposal_gaps(self._record(("run_rsa", {}, {"secure": True}))) == []

    def test_the_digest_and_the_latest_turn_keep_the_network(self):
        from llm_agent.agent.recommender import compact_turn, turn_digest
        rec = self._record(("run_rsa", {"network": "dc_bus9", "timestamp": "2000-01-01 00:00"},
                            {"secure": True, "network": self.NET,
                             "all_voltages": [{"bus_name": f"Bus_{i}", "vm_pu": 1.0} for i in range(400)]}))
        assert "run_rsa(network=dc_bus9, timestamp=2000-01-01 00:00)" in turn_digest(rec, 1)
        view = compact_turn(rec, max_chars_per_tool=200)["tools"][0]["result"]
        assert view["network"]["id"] == "dc_bus9" and "all_voltages" not in view

    def test_the_prompt_asks_for_alternatives_to_a_proposal(self):
        from llm_agent.agent.recommender import build_system_prompt
        prompt = build_system_prompt([("edit_proposal", "Create a network proposal.")])
        assert "network proposal" in prompt and "ALTERNATIVE to the proposal" in prompt
        assert "never claim" in prompt


def test_a_proposal_discarded_mid_study_is_a_clear_error(client):
    import main_backend as mb
    _create(client)
    req = types.SimpleNamespace(network="dc_bus9")
    mb._net_for(req)                       # the study has started on it
    del mb.app_data["proposals"]["dc_bus9"]  # the sidebar discards it
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as exc:
        mb._materialised("dc_bus9")
    assert exc.value.status_code == 409 and "discarded" in exc.value.detail


# ---------------------------------------------------------------------------
# A larger transformer, a new site
# ---------------------------------------------------------------------------

class TestUpgradeTrafo:
    def test_double_rating_is_two_units_in_parallel(self, net):
        doubled = copy.deepcopy(net)
        doubled.trafo.at[1, "parallel"] = 2
        upgraded, edits = sb.materialise(net, [{"op": "upgrade_trafo", "trafo": 1,
                                                "new_sn_mva": 2 * float(net.trafo.sn_mva[1])}])
        a, b = (nl.build_ybus_full(n) for n in (doubled, upgraded))
        i, j = int(net.trafo.hv_bus[1]), int(net.trafo.lv_bus[1])
        keys = [key for key in a["Yff_r"] if set(key[0]) == {i, j}]
        assert keys
        for k in ("Yff_r", "Yff_i", "Yft_r", "Yft_i"):
            for key in keys:
                assert a[k][key] == pytest.approx(b[k][key])
        assert fe._branch_current_data(upgraded)[(i, j)]["I_from_max"] == pytest.approx(
            fe._branch_current_data(doubled)[(i, j)]["I_from_max"])
        assert "to 19800 MVA" in sb.describe(upgraded, edits[0])

    def test_only_upward_and_within_reason(self, net):
        now = float(net.trafo.sn_mva[1])
        for value, fragment in ((now, "above"), (now * 11, "at most")):
            with pytest.raises(sb.EditError, match=fragment):
                sb.materialise(net, [{"op": "upgrade_trafo", "trafo": 1, "new_sn_mva": value}])


SITE = {"op": "add_bus_with_connection", "name": "DC site", "connect_to": "Bus_9",
        "length_km": 8, "line_like": 12}


class TestNewSite:
    def test_a_site_joined_by_a_copied_line_then_a_load_on_it(self, net):
        prop, edits = sb.materialise(net, [SITE, {**DC, "bus": "DC site", "p_mw": 5}])
        bus = int(prop.bus.index[-1])
        assert prop.bus.at[bus, "name"] == "DC site" and prop.bus.at[bus, "vn_kv"] == net.bus.vn_kv[9]
        assert prop.bus.at[bus, "max_vm_pu"] == net.bus.max_vm_pu[9]
        line = prop.line.iloc[-1]
        assert (int(line.from_bus), int(line.to_bus), line.length_km) == (9, bus, 8)
        for col in ("r_ohm_per_km", "x_ohm_per_km", "c_nf_per_km", "max_i_ka"):
            assert line[col] == pytest.approx(net.line.at[12, col])
        assert int(prop.load.iloc[-1]["bus"]) == bus and edits[1]["bus"] == bus
        pp.runpp(prop)
        assert prop.res_bus.vm_pu[bus] < prop.res_bus.vm_pu[9]   # the load draws through the line
        assert list(prop.bus.index[:len(net.bus)]) == list(net.bus.index)

    def test_a_standard_type_works_too(self, net):
        prop, _ = sb.materialise(net, [{**SITE, "line_like": None, "std_type": "NA2XS2Y 1x240 RM/25 12/20 kV"}])
        assert prop.line.iloc[-1]["std_type"] == "NA2XS2Y 1x240 RM/25 12/20 kV"

    @pytest.mark.parametrize("change, fragment", [
        ({"name": "Bus_3"}, "already named"),
        ({"line_like": None}, "exactly one of"),
        ({"std_type": "NA2XS2Y 1x240 RM/25 12/20 kV"}, "exactly one of"),
        ({"line_like": None, "std_type": "nope"}, "not a known line type"),
        ({"vn_kv": 999}, "one voltage"),
        ({"length_km": 0}, "above 0"),
        ({"length_km": 500}, "at most"),
    ])
    def test_invalid_sites_are_refused(self, net, change, fragment):
        with pytest.raises(sb.EditError, match=fragment):
            sb.materialise(net, [{**SITE, **change}])

    def test_a_site_has_no_measured_demand_to_follow(self, net):
        prop, _ = sb.materialise(net, [SITE, {**DC, "bus": "DC site", "profile": "follow_bus"}])
        with pytest.raises(sb.EditError, match="no measured demand"):
            sb.apply_profiles(prop, "2000-01-03 12:00:00", _meas({"Bus_9": 10.0}),
                              TestProfiles.PEAKS)


class TestSiteThroughTheBackend:
    def test_topology_lookup_and_studies_see_the_site(self, client):
        _create(client, "site", [{**SITE, "connect_to": "Bus 9", "line_like": "L12"},
                                 {**DC, "bus": "DC site", "p_mw": 10}])
        base = client.get("/api/network/topology").json()
        topo = client.get("/api/network/topology", params={"network": "site"}).json()
        assert len(topo["buses"]) == len(base["buses"]) + 1 and "network" not in base
        assert topo["network"]["id"] == "site"
        added = [b["name"] for b in topo["buses"] if b.get("added")]
        assert added == ["DC site"]
        assert [b["name"] for b in topo["branches"] if b.get("added")] == [
            next(b["name"] for b in topo["branches"] if "DC site connection" in b["name"])]
        rsa = client.post("/api/grid/rsa", json={"network": "site"}).json()
        assert any(v["bus_name"] == "DC site" for v in rsa["all_voltages"])
        n1 = client.post("/api/proposals/compare", json={"proposal": "site", "study": "n1"}).json()
        assert n1["proposal"]["total_outages_tested"] == n1["base"]["total_outages_tested"] + 1
        assert client.get("/api/network/topology", params={"network": "nope"}).status_code == 400

    def test_the_element_lookup_finds_the_site_on_the_proposal(self, client, monkeypatch):
        from llm_agent.agent import tools
        _create(client, "site", [{**SITE, "connect_to": "Bus 9", "line_like": "L12"}])
        monkeypatch.setattr(tools, "_get", lambda path: client.get(path).json())
        found = tools.locate_network_element("DC site", network="site")
        assert found["kind"] == "bus" and found["network"]["id"] == "site"
        assert found["n_connections"] == 1 and found["connected_to"][0]["name"] == "Bus_9"
        assert "DC site connection" in found["connected_to"][0]["via"]
        assert "error" in tools.locate_network_element("DC site")   # not on the base


class TestAddedOnTheDiffMap:
    def test_a_new_bus_and_line_are_drawn_as_added(self):
        pytest.importorskip("plotly")
        from llm_agent.agent.network_map import Edge, NetworkMap, Node
        from llm_agent.agent.renderers import render_network_diff
        base = NetworkMap("Grid", {0: Node(0, "Bus_0", 20.0, True, "substation", vm_pu=1.0)}, [])
        prop = NetworkMap("Grid", {0: Node(0, "Bus_0", 20.0, True, "substation", vm_pu=0.99),
                                   1: Node(1, "Site", 20.0, True, "substation", vm_pu=0.97, load_mw=10.0)},
                          [Edge(5, "Site connection", 0, 1, "line", True, 30.0)])
        fig = render_network_diff(base, prop, {0: (0.0, 0.0), 1: (1.0, 0.0)})
        hover = [h for d in fig.data for h in (d.hovertext or []) if isinstance(h, str)]
        assert any("Site connection</b><br>line · added by the proposal · 30.0% loaded" in h for h in hover)
        assert any("Site</b>" in h and "added by the proposal · 0.9700 pu" in h for h in hover)


# ---------------------------------------------------------------------------
# A new bus at another voltage behind a new transformer
# ---------------------------------------------------------------------------

@pytest.fixture
def mv():
    """A small 110/20 kV network with real voltages: HV — T — MV — line — MV2 (load)."""
    n = pp.create_empty_network()
    hv, b1, b2 = (pp.create_bus(n, vn_kv=kv, name=name, min_vm_pu=0.9, max_vm_pu=1.1)
                  for kv, name in ((110, "HV"), (20, "MV"), (20, "MV2")))
    pp.create_ext_grid(n, hv)
    pp.create_transformer(n, hv, b1, std_type="25 MVA 110/20 kV", name="Main")
    pp.create_line(n, b1, b2, length_km=4, std_type="NA2XS2Y 1x240 RM/25 12/20 kV", name="Feeder")
    pp.create_load(n, b2, p_mw=6, q_mvar=1, name="Town")
    return n


TRAFO = {"op": "add_bus_with_transformer", "name": "Site"}


class TestNewTransformer:
    def test_a_copy_at_double_rating_is_two_copies_in_parallel(self, net):
        like = {**TRAFO, "connect_to": 3, "trafo_like": 1}
        resized, _ = sb.materialise(net, [{**like, "sn_mva": 2 * float(net.trafo.sn_mva[1])}])
        doubled, _ = sb.materialise(net, [{**like, "parallel": 2}])
        a, b = (nl.build_ybus_full(n) for n in (doubled, resized))
        new = int(resized.bus.index[-1])
        keys = [key for key in a["Yff_r"] if set(key[0]) == {3, new}]
        assert keys
        for k in ("Yff_r", "Yff_i", "Yft_r", "Yft_i"):
            for key in keys:
                assert a[k][key] == pytest.approx(b[k][key])
        assert fe._branch_current_data(resized)[(3, new)]["I_from_max"] == pytest.approx(
            fe._branch_current_data(doubled)[(3, new)]["I_from_max"])

    def test_a_copy_is_the_same_unit_at_neutral_tap(self, net):
        prop, edits = sb.materialise(net, [{**TRAFO, "connect_to": 3, "trafo_like": "T1"}])
        new, t = int(prop.bus.index[-1]), prop.trafo.iloc[-1]
        assert edits[0]["vn_kv"] == net.bus.vn_kv[8]               # the copied unit's other side
        assert (int(t.hv_bus), int(t.lv_bus)) == (3, new) and t["proposed"] is True
        for col in ("sn_mva", "vk_percent", "vkr_percent", "tap_step_percent", "shift_degree"):
            assert t[col] == pytest.approx(net.trafo.at[1, col])
        assert t.tap_pos == 0 and net.trafo.tap_pos[1] == -1
        assert any("runs at tap -1" in n and "neutral tap" in n for n in sb.notes(prop, edits))
        pp.runpp(prop)
        assert prop.res_bus.vm_pu[new] == pytest.approx(prop.res_bus.vm_pu[3], abs=0.05)

    def test_a_catalogue_type_with_its_other_voltage(self, mv):
        prop, edits = sb.materialise(mv, [{**TRAFO, "connect_to": "MV2", "trafo_std_type": "0.63 MVA 20/0.4 kV"},
                                          {**DC, "bus": "Site", "p_mw": 0.3, "name": "Shop"}])
        t, new = prop.trafo.iloc[-1], int(prop.bus.index[-1])
        assert edits[0]["vn_kv"] == 0.4 and prop.bus.vn_kv[new] == 0.4
        assert t.std_type == "0.63 MVA 20/0.4 kV" and t.sn_mva == 0.63 and t.vk_percent == 6.0
        assert prop.bus.max_vm_pu[new] == 1.1                       # limits from the bus it joins
        pp.runpp(prop)
        assert 40 < prop.res_trafo.loading_percent.iloc[-1] < 60
        assert prop.res_bus.vm_pu[new] < prop.res_bus.vm_pu[2]
        assert sb.describe(prop, edits[0]) == ("New bus 'Site' (0.4 kV) behind a new 0.63 MVA 20/0.4 kV "
                                               "transformer (type '0.63 MVA 20/0.4 kV') at MV2")

    def test_explicit_values_and_the_tap_position(self, mv):
        unit = {**TRAFO, "connect_to": "MV2", "vn_kv": 0.4, "sn_mva": 1, "vk_percent": 6,
                "vkr_percent": 1, "pfe_kw": 1.5, "tap_step_percent": 2.5, "tap_min": -2, "tap_max": 2}
        vm = {}
        for pos in (0, 2):
            prop, edits = sb.materialise(mv, [{**unit, "tap_pos": pos}])
            pp.runpp(prop)
            vm[pos] = prop.res_bus.vm_pu.iloc[-1]
        t = prop.trafo.iloc[-1]
        assert (t.tap_side, t.tap_neutral, t.tap_pos, t.i0_percent) == ("hv", 0, 2, 0.15)
        assert vm[0] / vm[2] == pytest.approx(1.05, abs=0.003)     # +2 steps of 2.5 % on the HV side
        assert "tap +2" in sb.describe(prop, edits[0])

    def test_step_up_puts_the_new_bus_on_the_high_side(self, mv):
        prop, edits = sb.materialise(mv, [{**TRAFO, "connect_to": "MV2", "vn_kv": 66, "sn_mva": 40,
                                           "vk_percent": 10, "vkr_percent": 0.3, "shift_degree": 150}])
        t, new = prop.trafo.iloc[-1], int(prop.bus.index[-1])
        assert (int(t.hv_bus), int(t.lv_bus), t.vn_hv_kv, t.vn_lv_kv) == (new, 2, 66, 20)
        assert "66/20 kV" in sb.describe(prop, edits[0])
        pp.runpp(prop)

    def test_a_line_first_then_the_transformer(self, mv):
        prop, edits = sb.materialise(mv, [{
            **TRAFO, "connect_to": "MV", "length_km": 10, "line_std_type": "NA2XS2Y 1x240 RM/25 12/20 kV",
            "trafo_std_type": "0.63 MVA 20/0.4 kV", "parallel": 2},
            {**DC, "bus": "Site", "p_mw": 0.8, "name": "Factory"}])
        sub, new = (int(i) for i in prop.bus.index[-2:])
        assert list(prop.bus.name[-2:]) == ["Site 20 kV", "Site"] and prop.bus.vn_kv[sub] == 20
        line, t = prop.line.iloc[-1], prop.trafo.iloc[-1]
        assert (int(line.from_bus), int(line.to_bus), line.length_km) == (1, sub, 10)
        assert (int(t.hv_bus), int(t.lv_bus), int(t.parallel)) == (sub, new, 2)
        assert all(tbl["proposed"].iloc[-1] is True for tbl in (prop.line, prop.trafo))
        assert prop.bus["proposed"].iloc[-2:].tolist() == [True, True]
        summary = sb.summary(prop, edits)
        assert (summary["new_buses"], summary["new_trafos"], summary["new_transformer_mva"]) == (2, 1, 1.26)
        assert "2 × 0.63 MVA" in summary["edits"][0] and "end of a 10 km line from MV" in summary["edits"][0]
        assert any("2 units in parallel" in n for n in summary["notes"])
        pp.runpp(prop)
        assert prop.res_bus.vm_pu[new] < prop.res_bus.vm_pu[sub] < prop.res_bus.vm_pu[1]

    def test_overrides_on_a_copy_and_the_canonical_form_replays(self, mv):
        raw = {**TRAFO, "connect_to": "HV", "trafo_like": "Main", "sn_mva": 50, "vk_percent": 14,
               "max_loading_percent": 120, "min_vm_pu": 0.95}
        prop, edits = sb.materialise(mv, [raw])
        t = prop.trafo.iloc[-1]
        src = mv.trafo.iloc[0]
        assert (t.sn_mva, t.vk_percent, t.vkr_percent, t.max_loading_percent) == (50, 14, src.vkr_percent, 120)
        assert t.pfe_kw == pytest.approx(2 * src.pfe_kw)            # no-load losses grow with size
        assert prop.bus.min_vm_pu.iloc[-1] == 0.95 and prop.bus.max_vm_pu.iloc[-1] == 1.1
        again, same = sb.materialise(mv, edits)
        assert same == edits and pp.toolbox.nets_equal(prop, again)

    @pytest.mark.parametrize("change, fragment", [
        ({"connect_to": "MV2", "vn_kv": 0.4}, "give trafo_like"),
        ({"connect_to": "MV2", "vn_kv": 20, "sn_mva": 1, "vk_percent": 6, "vkr_percent": 1}, "same voltage"),
        ({"connect_to": "MV2", "trafo_like": "Main", "trafo_std_type": "0.63 MVA 20/0.4 kV"}, "not both"),
        ({"connect_to": "MV2", "trafo_std_type": "nope"}, "not a known transformer type"),
        ({"connect_to": "MV2", "trafo_std_type": "160 MVA 380/110 kV"}, "neither side"),
        ({"connect_to": "MV2", "vn_kv": 0.4, "trafo_std_type": "25 MVA 110/20 kV"}, "cannot join"),
        ({"connect_to": "MV2", "trafo_std_type": "0.63 MVA 20/0.4 kV", "vk_percent": 4}, "fixes vk_percent"),
        ({"connect_to": "MV2", "vn_kv": 0.4, "sn_mva": 1, "vk_percent": 6, "vkr_percent": 6}, "must be below"),
        ({"connect_to": "MV2", "vn_kv": 0.4, "sn_mva": 1, "vk_percent": 6, "vkr_percent": 1, "tap_pos": 1},
         "needs a tap changer"),
        ({"connect_to": "MV2", "vn_kv": 0.4, "sn_mva": 1, "vk_percent": 6, "vkr_percent": 1,
          "tap_step_percent": 2.5}, "needs tap_step_percent, tap_min and tap_max"),
        ({"connect_to": "MV2", "trafo_std_type": "0.63 MVA 20/0.4 kV", "tap_pos": 5}, "outside the tap range"),
        ({"connect_to": "MV2", "trafo_std_type": "0.63 MVA 20/0.4 kV", "parallel": 5}, "at most 4"),
        ({"connect_to": "MV2", "trafo_std_type": "0.63 MVA 20/0.4 kV", "parallel": 1.5}, "whole number"),
        ({"connect_to": "MV2", "trafo_std_type": "0.63 MVA 20/0.4 kV", "line_like": "Feeder"},
         "length_km is required"),
        ({"connect_to": "MV2", "trafo_std_type": "0.63 MVA 20/0.4 kV", "length_km": 3}, "exactly one of"),
        ({"connect_to": "MV2", "vn_kv": 0.4, "sn_mva": 1, "vk_percent": 6, "vkr_percent": 1,
          "vn_lv_kv": 0.6}, "vn_lv_kv must be at most"),
        ({"connect_to": "MV2", "vn_kv": 0.4, "sn_mva": 1, "vk_percent": 6, "vkr_percent": 1,
          "pfe_kw": 80}, "over 5 %"),
        ({"connect_to": "MV2", "trafo_std_type": "0.63 MVA 20/0.4 kV", "min_vm_pu": 1.05,
          "max_vm_pu": 1.0}, "below max_vm_pu"),
        ({"connect_to": "MV2", "trafo_std_type": "0.63 MVA 20/0.4 kV", "name": "MV"}, "already named"),
    ])
    def test_invalid_units_are_refused(self, mv, change, fragment):
        with pytest.raises(sb.EditError, match=fragment):
            sb.materialise(mv, [{**TRAFO, **change}])

    def test_notes_say_how_the_unit_was_modelled(self, mv, net):
        prop, edits = sb.materialise(mv, [{**TRAFO, "connect_to": "MV2", "vn_kv": 0.4, "sn_mva": 1,
                                           "vk_percent": 6, "vkr_percent": 1, "vn_lv_kv": 0.42}])
        notes = " ".join(sb.notes(prop, edits))
        assert "no no-load losses" in notes and "off nominal" in notes
        prop, edits = sb.materialise(net, [{**TRAFO, "connect_to": 3, "vn_kv": 20, "trafo_like": "T0",
                                            "sn_mva": 50}])
        notes = " ".join(sb.notes(prop, edits))
        assert "not a physical unit" in notes and "copied in per unit to a 135/20 kV unit" in notes

    def test_a_unit_between_equal_voltages_needs_the_new_voltage(self, net):
        net = copy.deepcopy(net)
        net.bus["vn_kv"] = 1.0                                       # a converted per-unit case
        with pytest.raises(sb.EditError, match="joins two 1 kV buses"):
            sb.materialise(net, [{**TRAFO, "connect_to": 3, "trafo_like": 1}])
        prop, edits = sb.materialise(net, [{**TRAFO, "connect_to": 3, "trafo_like": 1, "vn_kv": 20}])
        assert any("is a label" in n for n in sb.notes(prop, edits))


SUBSTATION = [{"op": "add_bus_with_transformer", "name": "DC 20 kV", "connect_to": "Bus 9", "vn_kv": 20,
               "sn_mva": 40, "vk_percent": 12, "vkr_percent": 0.4, "parallel": 2},
              {**DC, "bus": "DC 20 kV", "p_mw": 10}]


class TestTransformerSiteThroughTheBackend:
    def test_topology_and_studies_see_the_new_transformer(self, client):
        out = _create(client, "sub", SUBSTATION)
        assert out["summary"]["new_trafos"] == 1 and out["summary"]["new_transformer_mva"] == 80
        topo = client.get("/api/network/topology", params={"network": "sub"}).json()
        assert [b["name"] for b in topo["buses"] if b.get("added")] == ["DC 20 kV"]
        added = [b for b in topo["branches"] if b.get("added")]
        assert [b["kind"] for b in added] == ["trafo"] and "DC 20 kV transformer" in added[0]["name"]
        rsa = client.post("/api/grid/rsa", json={"network": "sub"}).json()
        assert any(v["bus_name"] == "DC 20 kV" for v in rsa["all_voltages"])
        assert any("DC 20 kV transformer" in (t.get("trafo_name") or "") for t in rsa["all_trafo_loading"])
        n1 = client.post("/api/proposals/compare", json={"proposal": "sub", "study": "n1"}).json()
        assert n1["proposal"]["total_outages_tested"] == n1["base"]["total_outages_tested"] + 1
        scan = client.post("/api/proposals/compare", json={
            "proposal": "sub", "study": "scan", "start_timestamp": "2000-01-03",
            "end_timestamp": "2000-01-03", "step_size": 16}).json()
        assert scan["study"] == "scan" and scan["network"]["id"] == "sub"

    @pytest.mark.skipif(shutil.which("ipopt") is None, reason="IPOPT not installed")
    def test_the_opf_runs_on_the_new_transformer(self, client):
        _create(client, "sub", SUBSTATION)
        out = client.post("/api/proposals/compare", json={"proposal": "sub", "study": "opf",
                                                         "timestamp": "2000-01-04 03:15"}).json()
        assert out["base"]["feasible"] and out["proposal"]["feasible"], out

    def test_it_survives_a_restart(self, restartable):
        mb, start = restartable
        with start() as c:
            before = _create(c, "sub", SUBSTATION)["edits"]
        mb.app_data["proposals"] = {}
        with start() as c:
            [after] = c.get("/api/proposals").json()["proposals"]
            assert after["id"] == "sub" and after["edits"] == before


def test_an_added_transformer_is_drawn_as_added_and_dotted():
    pytest.importorskip("plotly")
    from llm_agent.agent.network_map import Edge, NetworkMap, Node
    from llm_agent.agent.renderers import render_network_diff
    base = NetworkMap("Grid", {0: Node(0, "Bus_0", 20.0, True, "substation", vm_pu=1.0)}, [])
    prop = NetworkMap("Grid", {0: Node(0, "Bus_0", 20.0, True, "substation", vm_pu=0.99),
                               1: Node(1, "Site", 0.4, True, "distribution", vm_pu=0.97)},
                      [Edge(3, "Site transformer", 0, 1, "trafo", True, 45.0)])
    fig = render_network_diff(base, prop, {0: (0.0, 0.0), 1: (1.0, 0.0)})
    hover = [h for d in fig.data for h in (d.hovertext or []) if isinstance(h, str)]
    assert any("Site transformer</b><br>trafo · added by the proposal · 45.0% loaded" in h for h in hover)
    assert any(d.mode == "lines" and d.line.dash == "dot" for d in fig.data)


class TestGraderAndPrompt:
    def _turn(self, user, assistant="On proposal dc_bus9 the grid is secure."):
        net = {"id": "dc_bus9", "edits": ["Add load"]}
        return {"turn": 1, "status": "completed", "user": user, "assistant": assistant,
                "tool_calls": [{"name": "run_rsa", "args": {}}, {"name": "run_rsa", "args": {"network": "dc_bus9"}}],
                "tool_results": [
                    {"name": "run_rsa", "result": {"timestamp": "2000-01-01 00:00:00", "data_source": "measurements",
                                                   "thresholds_used": {}}},
                    {"name": "run_rsa", "result": {"timestamp": "2000-01-01 00:00:00", "data_source": "measurements",
                                                   "thresholds_used": {}, "network": net}}]}

    def test_a_requested_comparison_is_not_graded_as_drift(self):
        from evaluation.graders.integrity import grade_composition
        codes = lambda r: [f.code for f in grade_composition(r)]  # noqa: E731
        assert "inconsistent_context" not in codes(self._turn("Compare the data centre proposal with today"))
        assert "inconsistent_context" in codes(self._turn("Is the grid secure?"))

    def test_an_answer_the_guard_had_to_label_is_counted(self):
        from evaluation.graders.integrity import grade_composition
        from llm_agent.agent.validators import proposal_note
        rec = self._turn("Compare the data centre proposal with today", "Secure.")
        rec["assistant"] = "Secure.\n\n" + proposal_note("Secure.", rec["tool_results"])
        assert "proposal_unnamed" in [f.code for f in grade_composition(rec)]

    def test_the_prompt_limits_a_single_moment_verdict(self):
        from llm_agent.agent.config import DEFAULT_GRID_CONSTANTS
        from llm_agent.agent.system_prompt import build_system_prompt
        prompt = build_system_prompt(dict(DEFAULT_GRID_CONSTANTS))
        assert "verdict for **that moment**" in prompt and "add_bus_with_connection" in prompt


class TestFromTheLiveTest:
    """Found in the first live session: a new substation 5 km from Bus 9 with
    a 15 MW data centre."""

    LIVE = [{"op": "add_bus_with_connection", "name": "New Substation near Bus 9", "connect_to": "Bus_9",
             "length_km": 5, "line_like": "Bus_8 -> Bus_9 [L12]"},
            {"op": "add_load", "name": "Data centre", "bus": "New Substation near Bus 9", "p_mw": 15,
             "pf": 0.95, "profile": "flat"}]

    def test_the_comparison_names_the_violated_new_bus_and_reports_its_voltage(self, client):
        _create(client, "live", self.LIVE)
        out = client.post("/api/proposals/compare", json={"proposal": "live",
                                                         "timestamp": "2000-01-01 00:00"}).json()
        assert out["new_violations"] == [{"type": "bus_vm_pu", "element": "New Substation near Bus 9"}]
        (site,) = out["added_buses"]
        assert site["bus"] == "New Substation near Bus 9" and site["vm_pu"] < 0.94

    def test_placeholder_lengths_are_noted(self, client):
        out = _create(client, "live", self.LIVE)
        (note,) = out["summary"]["notes"]
        assert "length 1 km" in note and "5× that line's impedance" in note

    def test_real_lengths_need_no_note(self, net):
        net.line["length_km"] = [1.0 + i for i in range(len(net.line))]
        prop, edits = sb.materialise(net, [SITE])
        assert sb.notes(prop, edits) == []

    def test_a_load_at_a_new_bus_can_follow_another_buss_demand(self, net):
        prop, edits = sb.materialise(net, [SITE, {**DC, "bus": "DC site", "profile": "follow_bus",
                                                  "follow": "Bus_9"}])
        assert edits[1]["follow"] == 9 and "demand shape of Bus_9" in sb.describe(prop, edits[1])
        (rep,) = sb.apply_profiles(prop, "2000-01-03 12:00:00", _meas({"Bus_9": 10.0}), TestProfiles.PEAKS)
        assert rep["p_mw"] == pytest.approx(25.0) and rep["source"] == "demand of Bus_9"
        prop, _ = sb.materialise(net, [SITE, {**DC, "bus": "DC site", "profile": "follow_bus"}])
        with pytest.raises(sb.EditError, match="give `follow` a bus that has"):
            sb.apply_profiles(prop, "2000-01-03 12:00:00", _meas({"Bus_9": 10.0}), TestProfiles.PEAKS)

    def test_the_lookup_accepts_bus_8_for_bus_8(self, client, monkeypatch):
        from llm_agent.agent import tools
        monkeypatch.setattr(tools, "_get", lambda path: client.get(path).json())
        assert tools.locate_network_element("Bus 8")["element"] == "Bus_8"
        assert tools.locate_network_element("bus_8 -> bus 9")["element_index"] == 12


def test_a_retried_lookup_is_not_a_concern():
    from llm_agent.agent.recommender import needs_attention
    rec = {"tool_calls": [{"name": "locate_network_element", "args": {"element": "Bus 8"}},
                          {"name": "locate_network_element", "args": {"element": "Bus_8"}}],
           "tool_results": [{"name": "locate_network_element", "result": {"error": "No bus named 'Bus 8'."}},
                            {"name": "locate_network_element", "result": {"element": "Bus_8"}}]}
    assert needs_attention(rec) == []
    rec["tool_results"] = rec["tool_results"][:1]
    assert needs_attention(rec) == ["locate_network_element: returned an error"]


# ---------------------------------------------------------------------------
# Proposals survive a backend restart (for the same network)
# ---------------------------------------------------------------------------

@pytest.fixture
def restartable(tmp_path, monkeypatch):
    """Start the backend, stop it, start it again — all storage in tmp_path."""
    pytest.importorskip("fastapi.testclient")
    import main_backend as mb
    from fastapi.testclient import TestClient
    monkeypatch.setenv("GRID_PROFILE", "pglib_case14")
    for name, sub in (("SYSTEMS_DIR", "systems"), ("DATA_FILES_DIR", "data_files")):
        monkeypatch.setattr(mb, name, str(tmp_path / sub))
    for name in ("_LAST_UPLOAD_SENTINEL", "_LAST_TIMESERIES_SENTINEL", "_LAST_FORECAST_SENTINEL"):
        monkeypatch.setattr(mb, name, str(tmp_path / f"{name}.json"))
    monkeypatch.setattr(mb, "_PROPOSALS_FILE", str(tmp_path / "systems" / "proposals.json"))
    monkeypatch.setattr(mb, "_ASSUMPTIONS_FILE", str(tmp_path / "systems" / "assumptions.json"))
    monkeypatch.setattr(mb, "_SCENARIO_FILE", str(tmp_path / "scenario.json"))
    saved = dict(mb.app_data)

    def start():
        return TestClient(mb.app)

    yield mb, start
    mb.app_data.clear()
    mb.app_data.update(saved)


class TestPersistence:
    def test_proposals_come_back_after_a_restart(self, restartable):
        mb, start = restartable
        with start() as c:
            _create(c, "dc_bus9")
            _create(c, "site", [SITE | {"connect_to": "Bus 9", "line_like": "L12"}])
            before = c.get("/api/proposals").json()["proposals"]
        mb.app_data["proposals"] = {}
        with start() as c:
            after = c.get("/api/proposals").json()["proposals"]
            assert [(p["id"], p["edits_hash"]) for p in after] == [(p["id"], p["edits_hash"]) for p in before]
            assert c.post("/api/grid/rsa", json={"network": "site"}).status_code == 200
            c.delete("/api/proposals/site")
        mb.app_data["proposals"] = {}
        with start() as c:
            assert [p["id"] for p in c.get("/api/proposals").json()["proposals"]] == ["dc_bus9"]

    def test_not_restored_onto_a_different_network(self, restartable):
        mb, start = restartable
        with start() as c:
            _create(c)
        saved = json.loads(open(mb._PROPOSALS_FILE).read())
        (entry,) = saved["by_fingerprint"].values()             # one entry per network
        saved["by_fingerprint"] = {"another network": entry}
        open(mb._PROPOSALS_FILE, "w").write(json.dumps(saved))
        mb.app_data["proposals"] = {}
        with start() as c:
            assert c.get("/api/proposals").json()["proposals"] == []

    def test_a_file_in_the_older_one_network_format_is_still_read(self, restartable):
        mb, start = restartable
        with start() as c:
            _create(c)
        saved = json.loads(open(mb._PROPOSALS_FILE).read())
        ((fp, entry),) = saved["by_fingerprint"].items()
        open(mb._PROPOSALS_FILE, "w").write(json.dumps({"version": 1, "base_fingerprint": fp, **entry}))
        mb.app_data["proposals"] = {}
        with start() as c:
            assert [p["id"] for p in c.get("/api/proposals").json()["proposals"]] == ["dc_bus9"]

    def test_an_edit_that_no_longer_applies_is_dropped(self, restartable):
        mb, start = restartable
        with start() as c:
            _create(c)
        saved = json.loads(open(mb._PROPOSALS_FILE).read())
        (entry,) = saved["by_fingerprint"].values()
        entry["proposals"]["broken"] = {**entry["proposals"]["dc_bus9"], "id": "broken",
                                        "edits": [{**DC, "bus": 999}]}
        open(mb._PROPOSALS_FILE, "w").write(json.dumps(saved))
        mb.app_data["proposals"] = {}
        with start() as c:
            assert [p["id"] for p in c.get("/api/proposals").json()["proposals"]] == ["dc_bus9"]

    def test_replacing_the_network_forgets_them(self, restartable):
        import os
        mb, start = restartable
        with start() as c:
            _create(c)
            assert os.path.isfile(mb._PROPOSALS_FILE)
            assert c.post("/api/network/reset_active", json={}).status_code == 200
            assert not os.path.isfile(mb._PROPOSALS_FILE)
            assert c.get("/api/proposals").json()["proposals"] == []

    def test_the_fingerprint_ignores_measured_values_but_not_structure(self, net):
        a = sb.network_fingerprint(net)
        changed = copy.deepcopy(net)
        changed.load["p_mw"] *= 1.7                    # another tick
        assert sb.network_fingerprint(changed) == a
        prop, _ = sb.materialise(net, [DC])            # added rows are not structure of the base
        assert sb.network_fingerprint(prop) == a
        changed.line.at[3, "max_i_ka"] *= 2
        assert sb.network_fingerprint(changed) != a


class TestCompareOverTime:
    FACTORY = [{**DC, "name": "Factory", "p_mw": 45, "profile": "shift", "start_hour": 8, "end_hour": 18}]

    def test_violations_follow_the_proposals_profile(self, client):
        _create(client, "factory", self.FACTORY)
        out = client.post("/api/proposals/compare", json={
            "proposal": "factory", "study": "scan",
            "start_timestamp": "2000-01-02", "end_timestamp": "2000-01-02"}).json()
        assert out["study"] == "scan" and out["window"]["ticks_scanned"] == 96 and out["window"]["step"] == 1
        only = out["ticks_violating_only_on_proposal"]
        assert only["count"] > 0
        assert all(8 <= int(t[11:13]) < 18 for t in only["first"])       # only while the factory runs
        assert out["proposal"]["ticks_with_violations"] >= out["base"]["ticks_with_violations"]
        assert out["new_violations"] and out["resolved_violations"] == []
        assert out["suggested_next"].startswith("An N-1 screen of the proposal at 2000-01-02")
        assert len(out["series"]["timestamps"]) == 96

    def test_a_long_window_is_thinned_and_says_so(self, client, monkeypatch):
        import main_backend as mb
        monkeypatch.setattr(mb, "_SCAN_MAX_TICKS", 50)
        _create(client, "factory", self.FACTORY)
        out = client.post("/api/proposals/compare", json={"proposal": "factory", "study": "scan",
                                                         "end_timestamp": "2000-01-02"}).json()
        assert out["window"]["ticks_in_window"] == 192 and out["window"]["step"] == 4
        assert "every 4th tick" in out["window"]["note"]
        exact = client.post("/api/proposals/compare", json={"proposal": "factory", "study": "scan",
                                                           "end_timestamp": "2000-01-01", "step_size": 1}).json()
        assert exact["window"]["step"] == 1 and "note" not in exact["window"]

    def test_bad_windows_are_refused(self, client):
        _create(client, "factory", self.FACTORY)
        bad = [{"start_timestamp": "2000-01-03", "end_timestamp": "2000-01-02"},
               {"start_timestamp": "1999"}, {"step_size": 0}]
        for extra in bad:
            r = client.post("/api/proposals/compare", json={"proposal": "factory", "study": "scan", **extra})
            assert r.status_code in (400, 404), (extra, r.status_code)

    def test_the_model_gets_the_summary_and_the_chart_gets_the_series(self):
        from llm_agent.agent.loop import _prepare_tool_result_for_model
        result = {"study": "scan", "network": {"id": "f"}, "proposal": {"ticks_with_violations": 3},
                  "series": {"timestamps": ["2000-01-01 00:00:00"], "base_min_vm": [1.0]}}
        assert "series" not in _prepare_tool_result_for_model("compare_networks", result)
        assert "series" in result

    def test_a_scan_counts_as_a_study_over_time(self):
        from llm_agent.agent.recommender import proposal_gaps
        net = {"id": "f", "edits": [], "compared_with": "base"}
        rec = {"tool_calls": [{"name": "compare_networks", "args": {"study": "scan"}}],
               "tool_results": [{"name": "compare_networks",
                                 "result": {"study": "scan", "network": net, "new_violations": []}}]}
        assert proposal_gaps(rec) == ["proposal f: no N-1 screen of the proposal in this turn"]

    def test_the_chart(self):
        pytest.importorskip("plotly")
        from llm_agent.agent.renderers import render_network_comparison
        stamps = ["2000-01-01 00:00:00", "2000-01-01 00:15:00"]
        figs = render_network_comparison({
            "study": "scan", "network": {"id": "f"}, "thresholds_used": {"vm_lower_pu": 0.94,
                                                                         "max_line_loading_pct": 100},
            "window": {"start": stamps[0], "end": stamps[1]},
            "series": {"timestamps": stamps, "base_min_vm": [1.0, 0.99], "proposal_min_vm": [0.95, 0.93],
                       "base_max_line_loading": [40, 41], "proposal_max_line_loading": [60, 70]}})
        assert [f.layout.title.text.split(" over time")[0] for f in figs] == [
            "Lowest bus voltage", "Highest line loading"]
        assert list(figs[0].data[1].y) == [0.95, 0.93] and figs[0].data[1].name == "f"


class TestCharacterisationAcrossNetworks:
    """Found grading the first live session: "Secure" in a base-vs-proposal
    table was read as a claim about the proposal's insecure assessment."""

    PROP = {"id": "p", "edits": ["x"]}

    def _conflicts(self, answer, *results):
        from llm_agent.agent.characterisation import check_study_claims
        return check_study_claims(answer, {"tool_results": [{"name": n, "result": r} for n, r in results]})

    def test_a_comparison_with_one_secure_network_is_not_a_conflict(self):
        compare = {"study": "rsa", "network": {**self.PROP, "compared_with": "base"},
                   "base": {"secure": True}, "proposal": {"secure": False}}
        rsa_on_p = {"secure": False, "total_violations": 1, "network": self.PROP}
        assert self._conflicts("| Security | Secure | Not secure |", ("compare_networks", compare),
                               ("run_rsa", rsa_on_p)) == []

    def test_one_network_alone_is_still_judged(self):
        rsa_on_p = {"secure": False, "total_violations": 1, "network": self.PROP}
        assert [c.field for c in self._conflicts("The proposal is secure.", ("run_rsa", rsa_on_p))] == ["secure"]
        assert [c.field for c in self._conflicts("The grid is secure.",
                                                 ("run_rsa", {"secure": False, "total_violations": 2}))] == ["secure"]


def test_the_comparison_states_its_differences(client):
    _create(client)
    out = client.post("/api/proposals/compare", json={"proposal": "dc_bus9"}).json()
    ch = out["changes"]
    assert ch["total_load_mw"] == pytest.approx(out["proposal"]["total_load_mw"] - out["base"]["total_load_mw"], abs=1e-3)
    assert ch["slack_import_mw"] == pytest.approx(out["proposal"]["slack_import_mw"] - out["base"]["slack_import_mw"], abs=1e-3)


class TestFlatAddedLoadsAreFixed:
    """Load stress and demand-uncertainty draws leave a flat added load at its
    stated size; an added load that follows its bus's demand scales with it."""

    EDITS = [{**DC, "p_mw": 30, "profile": "flat"},
             {**DC, "name": "Factory", "bus": "Bus 4", "p_mw": 10, "profile": "follow_bus"}]

    def test_the_helper(self, net):
        import main_backend as mb
        prop, _ = sb.materialise(net, self.EDITS)
        before = prop.load.p_mw.copy()
        mb._scale_loads(prop, 1.5)
        flat, follow = prop.load.index[-2], prop.load.index[-1]
        assert prop.load.p_mw[flat] == pytest.approx(before[flat])
        assert prop.load.p_mw[follow] == pytest.approx(1.5 * before[follow])
        assert prop.load.p_mw[0] == pytest.approx(1.5 * before[0])

    def test_through_the_rsa(self, client):
        _create(client, "fixed", self.EDITS)
        ts = "2000-01-03 14:45:00"
        base1 = client.post("/api/grid/rsa", json={"timestamp": ts}).json()["total_load_mw"]
        prop = client.post("/api/grid/rsa", json={"timestamp": ts, "network": "fixed",
                                                  "load_scaling_factor": 1.2}).json()
        added = {e["element"]: e["p_mw"] for e in prop["network"]["added_elements_at_tick"]["elements"]}
        expected = (base1 + added["Factory"]) * 1.2 + added["Data centre"]
        assert prop["total_load_mw"] == pytest.approx(expected, abs=1e-2)


@pytest.mark.skipif(shutil.which("ipopt") is None, reason="IPOPT not installed")
def test_the_opf_comparison(client):
    import main_backend as mb
    _create(client, "small", [{**DC, "p_mw": 10}])
    mb.app_data["last_dispatch_result"], mb.app_data["last_dispatch_timestamp"] = ["kept"], "t0"
    out = client.post("/api/proposals/compare", json={"proposal": "small", "study": "opf",
                                                     "timestamp": "2000-01-04 03:15"}).json()
    assert out["study"] == "opf" and out["network"]["id"] == "small"
    for side in ("base", "proposal"):
        assert {"status", "feasible", "units_moved", "redispatch_mw", "grid_import_after_mw"} <= set(out[side])
    assert out["base"]["feasible"] and out["proposal"]["feasible"]
    assert out["changes"]["grid_import_after_mw"] == pytest.approx(
        out["proposal"]["grid_import_after_mw"] - out["base"]["grid_import_after_mw"], abs=1e-3)
    assert isinstance(out["units"], list)
    assert mb.app_data["last_dispatch_result"] == ["kept"]      # the comparison left it alone
