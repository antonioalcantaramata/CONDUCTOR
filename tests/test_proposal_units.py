"""
New units in a proposal: generators modelled as the operator chooses — a
synchronous machine on voltage control, or an inverter-based unit at a fixed
power factor — capacitor banks and reactors, and loads at a leading power
factor. When a decision is missing, the edit asks (with options) instead of
guessing; every default it fills in is reported as assumed.
"""

import math
import shutil

import pandapower as pp
import pandapower.networks as pn
import pytest

import sandbox as sb

TAN95 = math.tan(math.acos(0.95))


@pytest.fixture
def net():
    n = pn.case14()
    n.bus["name"] = None
    return n


def _gen(**kw):
    return {"op": "add_generator", "name": "G", "bus": 4, "p_max_mw": 30, **kw}


class TestQuestions:
    @pytest.mark.parametrize("edit, fragment, n_options", [
        (_gen(), "What kind of unit", 3),
        (_gen(kind="dispatchable"), "behave in a power flow", 2),
        ({"op": "add_shunt", "name": "C", "bus": 4, "q_mvar": 5}, "capacitor bank or a reactor", 2),
    ])
    def test_a_missing_decision_is_a_question_with_options(self, net, edit, fragment, n_options):
        with pytest.raises(sb.EditQuestion, match=fragment) as exc:
            sb.materialise(net, [edit])
        assert len(exc.value.options) == n_options and exc.value.edit == 1

    def test_two_set_points_on_one_bus(self, net):
        held = float(net.gen.vm_pu[net.gen.bus == 1].iloc[0])
        with pytest.raises(sb.EditQuestion, match="already held at") as exc:
            sb.materialise(net, [_gen(bus=1, kind="dispatchable", control="voltage", vm_pu=held + 0.03)])
        assert exc.value.options[0].startswith(f"hold {held:g} p.u.")
        prop, edits = sb.materialise(net, [_gen(bus=1, kind="dispatchable", control="voltage")])
        assert edits[0]["vm_pu"] == held                      # left out: it takes the bus's own

    def test_a_measured_substation_network_offers_a_fixed_power_factor(self):
        n = pp.create_empty_network()
        b0, b1 = pp.create_bus(n, 20, name="Grid"), pp.create_bus(n, 20, name="Alpha")
        pp.create_ext_grid(n, b0)
        pp.create_line(n, b0, b1, 5, "NA2XS2Y 1x95 RM/25 12/20 kV")
        pp.create_load(n, b1, p_mw=5.0, name="Alpha Load")
        with pytest.raises(sb.EditQuestion, match="cannot control voltage here"):
            sb.materialise(n, [_gen(bus="Alpha", kind="dispatchable", control="voltage", p_max_mw=3)])
        sb.materialise(n, [_gen(bus="Alpha", kind="dispatchable", control="power_factor", p_max_mw=3)])


class TestGenerators:
    def test_a_synchronous_machine_holds_its_bus(self, net):
        prop, edits = sb.materialise(net, [_gen(kind="dispatchable", control="voltage", vm_pu=1.03,
                                                p_mw=20, p_min_mw=5)])
        row = prop.gen.iloc[-1]
        assert (row["name"], row.vm_pu, row.p_mw, row.min_p_mw, row.max_p_mw) == ("G", 1.03, 20, 5, 30)
        assert row["proposed"] is True and row["controllable"] is True
        q = round(30 * math.tan(math.acos(sb.ASSUMED_RATED_PF)), 4)
        assert (edits[0]["q_min_mvar"], edits[0]["q_max_mvar"]) == (-q, q)
        pp.runpp(prop)
        assert prop.res_bus.vm_pu[4] == pytest.approx(1.03)
        assert "holds 1.03 p.u." in sb.describe(prop, edits[0])

    def test_an_inverter_unit_at_a_fixed_power_factor(self, net):
        prop, edits = sb.materialise(net, [_gen(kind="wind", pf=0.95, absorbing=True)])
        assert edits[0]["control"] == "power_factor" and "pf 0.95 absorbing" in sb.describe(prop, edits[0])
        import pandas as pd
        tick = pd.DataFrame({"substation_name": ["Bus_4"], "consumption": [0.0], "production": [10.0]})
        report = sb.apply_profiles(prop, "2000-01-03 12:00:00", tick,
                                   {"consumption": {}, "production": {"Bus_4": 20.0}, "total_production": 20.0})
        i = prop.sgen.index[-1]
        assert prop.sgen.at[i, "p_mw"] == pytest.approx(15.0)       # 10/20 of 30 MW
        assert prop.sgen.at[i, "q_mvar"] == pytest.approx(-15.0 * TAN95)
        assert report[0]["control"] == "power_factor"

    def test_minimum_output_and_not_dispatchable(self, net):
        prop, edits = sb.materialise(net, [_gen(kind="dispatchable", control="power_factor", p_min_mw=8,
                                                dispatchable=False)])
        assert edits[0]["p_mw"] == 8 and prop.sgen.iloc[-1]["controllable"] is False
        with pytest.raises(sb.EditError, match="at least 8"):
            sb.materialise(net, [_gen(kind="dispatchable", control="power_factor", p_min_mw=8, p_mw=4)])

    def test_a_proposal_stored_before_control_existed_still_replays(self, net):
        old = {"op": "add_generator", "name": "Old", "bus": 4, "p_max_mw": 10, "kind": "dispatchable", "p_mw": 2}
        with pytest.raises(sb.EditQuestion):
            sb.materialise(net, [old])
        prop, edits = sb.materialise(net, sb.upgrade_stored([old]))
        assert edits[0]["control"] == "power_factor" and edits[0]["pf"] == 1.0


class TestShuntsAndLoads:
    def test_a_capacitor_raises_and_a_reactor_lowers(self, net):
        pp.runpp(net)
        v0 = net.res_bus.vm_pu[13]
        for kind, sign in (("capacitor", 1), ("reactor", -1)):
            prop, edits = sb.materialise(net, [{"op": "add_shunt", "name": "S", "bus": 13, "q_mvar": 10,
                                                "kind": kind}])
            pp.runpp(prop)
            assert sign * (prop.res_bus.vm_pu[13] - v0) > 1e-4
            assert prop.shunt["proposed"].iloc[-1] is True
            assert sb.summary(prop, edits)["new_shunts_mvar"][kind] == 10
        assert sb.network_fingerprint(prop) == sb.network_fingerprint(net)

    def test_a_leading_load_injects_reactive_power(self, net):
        prop, edits = sb.materialise(net, [{"op": "add_load", "name": "DC", "bus": 9, "p_mw": 10,
                                            "pf": 0.95, "leading": True}])
        assert prop.load.iloc[-1].q_mvar == pytest.approx(-10 * TAN95)
        assert "pf 0.95 leading" in sb.describe(prop, edits[0])


class TestThroughTheBackend:
    @pytest.fixture
    def client(self, backend_client):
        yield backend_client
        import main_backend as mb
        mb._discard_proposals()

    def test_the_question_reaches_the_agent_with_its_options(self, client):
        r = client.post("/api/proposals/edit", json={"proposal": "g", "edits": [
            {"op": "add_load", "name": "DC", "bus": "Bus 9", "p_mw": 5},
            {"op": "add_generator", "name": "G", "bus": "Bus 4", "p_max_mw": 30, "kind": "dispatchable"}]})
        d = r.json()["detail"]
        assert r.status_code == 400 and d["needs_operator"] is True and d["edit"] == 2
        assert d["question"].startswith("How should 'G' behave") and len(d["options"]) == 2
        assert "do not choose for them" in d["instruction"]
        assert client.get("/api/proposals").json()["proposals"] == []          # nothing stored

    def test_defaults_are_reported_as_assumed(self, client):
        out = client.post("/api/proposals/edit", json={"proposal": "g", "edits": [
            {"op": "add_load", "name": "DC", "bus": "Bus 9", "p_mw": 5},
            {"op": "add_generator", "name": "G", "bus": "Bus 4", "p_max_mw": 30, "kind": "dispatchable",
             "control": "voltage"}]}).json()
        assumed = {a["edit"]: a["defaults"] for a in out["assumed"]}
        assert assumed[1] == {"pf": 0.95, "profile": "flat"}
        assert {"vm_pu", "q_min_mvar", "q_max_mvar", "dispatchable", "p_min_mw", "p_mw"} <= set(assumed[2])

    def test_studies_see_a_new_synchronous_machine(self, client):
        client.post("/api/proposals/edit", json={"proposal": "g", "edits": [
            {"op": "add_generator", "name": "G", "bus": "Bus 4", "p_max_mw": 30, "p_mw": 10,
             "kind": "dispatchable", "control": "voltage", "vm_pu": 1.02}]})
        rsa = client.post("/api/grid/rsa", json={"network": "g"}).json()
        held = {v["unit"]: v for v in rsa["voltage_controlled_buses"]}
        assert held["G"]["setpoint_pu"] == pytest.approx(1.02)
        added = rsa["network"]["added_elements_at_tick"]["elements"]
        assert added == [{**added[0], "element": "G", "p_mw": 10.0, "control": "voltage"}]
        snap = client.post("/api/network/snapshot", json={"network": "g"}).json()
        assert any("G" in str(g.get("name")) for g in snap["generators"])

    @pytest.mark.skipif(shutil.which("ipopt") is None, reason="IPOPT not installed")
    def test_the_opf_uses_a_new_units_limits_and_keeps_a_fixed_one(self, client):
        client.post("/api/proposals/edit", json={"proposal": "g", "edits": [
            {"op": "add_generator", "name": "G", "bus": "Bus 4", "p_max_mw": 30, "p_mw": 10,
             "kind": "dispatchable", "control": "voltage", "q_min_mvar": -2, "q_max_mvar": 2,
             "dispatchable": False}]})
        out = client.post("/api/flexibility/optimize", json={"network": "g", "timestamp": "2000-01-03 14:15"}).json()
        assert out["feasible"], out
        g = next(a for a in out["activated_resources"] if a.get("element") == "G")
        assert -2 - 1e-3 <= g["Qg_new"] <= 2 + 1e-3
        assert g["Pg_new"] == pytest.approx(g["Pg_base"], abs=1e-3)


def test_the_agent_is_told_to_ask_and_to_state_defaults():
    from llm_agent.agent import tool_schemas as ts
    from llm_agent.agent.system_prompt import build_system_prompt
    fd = next(fd for t in ts.TOOLS for fd in t.function_declarations if fd.name == "edit_proposal")
    item = fd.parameters.properties["edits"].items.properties
    assert {"control", "vm_pu", "q_min_mvar", "q_max_mvar", "absorbing", "dispatchable", "p_min_mw",
            "leading", "q_mvar"} <= set(item)
    assert "add_shunt" in item["op"].enum and "capacitor" in item["kind"].enum
    assert "needs_operator" in fd.description and "assumed" in fd.description
    prompt = build_system_prompt({"name": "g"})
    assert "`needs_operator`" in prompt and "never pick an option yourself" in prompt


def test_a_new_unit_at_a_fixed_reactive_output(net):
    import pandas as pd
    with pytest.raises(sb.EditQuestion, match="What reactive power"):
        sb.materialise(net, [_gen(kind="dispatchable", control="fixed_q")])
    prop, edits = sb.materialise(net, [_gen(kind="wind", control="fixed_q", q_mvar=5)])
    assert "fixed Q 5 Mvar injected" in sb.describe(prop, edits[0])
    tick = pd.DataFrame({"substation_name": ["Bus_4"], "consumption": [0.0], "production": [2.0]})
    sb.apply_profiles(prop, "2000-01-03 12:00:00", tick,
                      {"consumption": {}, "production": {"Bus_4": 20.0}, "total_production": 20.0})
    assert prop.sgen.iloc[-1]["q_mvar"] == pytest.approx(5.0)     # whatever its output (3 MW here)
