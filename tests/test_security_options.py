"""
Options with the operator's limits and with N-1 security, on the default demo
(PGLib IEEE 30-bus):

- per-branch loading caps (`branch_limits`): every option keeps the branch
  below its cap, and the power-flow check holds it to that cap;
- the most-headroom option: the lowest highest loading of all options;
- the N-1 study (`n1`, scopf.py): preventive and corrective dispatches that
  stay within limits after the critical outages, each screened under N-1 the
  same way as the least-disturbance and cheapest options, with a corrective
  plan checked by a power flow per outage.

The backend runs in-process with its storage in a temporary folder. The OPF
checks need IPOPT.
"""

import shutil

import pytest

pytest.importorskip("fastapi.testclient")

import main_backend as mb  # noqa: E402
import scopf  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

needs_ipopt = pytest.mark.skipif(shutil.which("ipopt") is None, reason="IPOPT not installed")
PEAK = "2000-01-06 20:00:00"            # the worst hour of the week
NIGHT = "2000-01-01 00:00:00"           # secure with everything in service, not after every outage


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    store = tmp_path_factory.mktemp("security")
    patch = pytest.MonkeyPatch()
    patch.delenv("GRID_PROFILE", raising=False)                      # the default demo
    patch.setattr(mb, "SYSTEMS_DIR", str(store / "systems"))
    patch.setattr(mb, "DATA_FILES_DIR", str(store / "data_files"))
    for name in ("_LAST_UPLOAD_SENTINEL", "_LAST_TIMESERIES_SENTINEL", "_LAST_FORECAST_SENTINEL",
                 "_PROPOSALS_FILE", "_ASSUMPTIONS_FILE", "_SCENARIO_FILE"):
        patch.setattr(mb, name, str(store / f"{name}.json"))
    saved = dict(mb.app_data)
    with TestClient(mb.app) as c:
        yield c
    mb.app_data.clear()
    mb.app_data.update(saved)
    patch.undo()


def _ok(r):
    assert r.status_code == 200, r.text
    return r.json()


def _options(client, **body):
    return _ok(client.post("/api/flexibility/optimize", json={"options": True, **body}))


# ---------------------------------------------------------------------------
# Per-branch caps
# ---------------------------------------------------------------------------

@needs_ipopt
def test_a_capped_branch_stays_below_its_cap_in_every_option(client):
    out = _options(client, timestamp=PEAK, branch_limits={"L30": 70})
    assert out["branch_limits"] == {"Bus_21 -> Bus_23 [L30]": 70.0}            # named as in every result
    assert len(out["options"]) >= 3
    for option in out["options"]:                       # the power flow holds each to the cap as well
        assert option["power_flow_check"]["confirmed"], (option["id"], option["power_flow_check"])
    # The same check, on a dispatch kept in memory, flags a cap it does not meet.
    _found, option = mb._resolve_option(out["options"][0]["ref"])
    tight = mb._power_flow_check(mb._net_for(None), option, 0.95, 1.05, caps={"Bus_21 -> Bus_23 [L30]": 10.0})
    assert any("above its 10 % cap" in v for v in tight["violations"])


def test_a_cap_on_an_unknown_branch_or_out_of_range_is_refused(client):
    r = client.post("/api/flexibility/optimize", json={"timestamp": PEAK, "options": True,
                                                       "branch_limits": {"L999": 80}})
    assert r.status_code == 400 and "L999" in r.json()["detail"]
    r = client.post("/api/flexibility/optimize", json={"timestamp": PEAK, "options": True,
                                                       "branch_limits": {"L30": 5}})
    assert r.status_code == 400 and "L30" in r.json()["detail"]


# ---------------------------------------------------------------------------
# Most headroom
# ---------------------------------------------------------------------------

@needs_ipopt
def test_the_most_headroom_option_has_the_lowest_highest_loading(client):
    out = _options(client, timestamp=PEAK)
    by_id = {o["id"]: o for o in out["options"]}
    assert "most_headroom" in by_id, [d["id"] for d in out["not_offered"]]
    lowest = min(o["max_loading_pct"] for o in out["options"])
    assert by_id["most_headroom"]["max_loading_pct"] <= lowest + 0.1
    assert by_id["most_headroom"]["max_loading_pct"] < by_id["cheapest"]["max_loading_pct"] - 5
    assert by_id["most_headroom"]["power_flow_check"]["confirmed"]


# ---------------------------------------------------------------------------
# N-1 security
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def corrective(client):
    if shutil.which("ipopt") is None:
        pytest.skip("IPOPT not installed")
    # No `options` flag: an N-1 study is a set of options anyway.
    return _ok(client.post("/api/flexibility/optimize", json={"timestamp": NIGHT, "n1": "corrective"}))


@pytest.fixture(scope="module")
def preventive(client):
    if shutil.which("ipopt") is None:
        pytest.skip("IPOPT not installed")
    return _options(client, timestamp=PEAK, n1="preventive")


def test_an_n1_study_is_a_set_of_options_compared_by_their_screens(corrective):
    out = corrective
    assert out["status"] == "options" and out["set"] == "n1"
    assert [o["id"] for o in out["options"]] == ["least_disturbance", "cheapest", "n1_corrective"]
    study = out["n1_study"]
    assert study["modes"] == ["corrective"] and study["emergency_loading_pct"] == 100.0
    assert study["outages_studied"] and study["outages_tested_by_screen"] > len(study["outages_studied"])
    for option in out["options"]:                       # every option screened the same way
        assert option["n1"]["screen"]["outages_tested"] == study["outages_tested_by_screen"]
        assert option["power_flow_check"]["confirmed"]
    by_id = {o["id"]: o for o in out["options"]}
    assert by_id["n1_corrective"]["n1"]["screen"]["outages_with_violations"] <= \
        by_id["least_disturbance"]["n1"]["screen"]["outages_with_violations"]


def test_the_corrective_plan_says_what_to_do_after_each_outage(corrective):
    n1 = next(o for o in corrective["options"] if o["id"] == "n1_corrective")["n1"]
    plan = n1["corrective_plan"]
    assert n1["screen_is"] == "before any corrective action"
    assert [p["outage"] for p in plan] == n1["studied"]
    not_secured = {x["outage"] for x in n1["not_secured"]}
    assert n1["secured"] == len(n1["studied"]) - len(not_secured)
    for p in plan:
        if p["no_action_needed"]:
            assert p["actions"] == [] and p["checked_by_power_flow"]
        elif p["outage"] not in not_secured:            # a plan the study counts as securing it holds
            assert p["actions"] and p["checked_by_power_flow"], p
        for a in p["actions"]:                          # an instruction, not solver residue
            assert abs(a["dp_mw"]) >= mb._PLAN_MW or "setpoint_vm_pu_after" in a
        units = [a["unit"] for a in p["actions"]]
        assert units == sorted(units, key=lambda u: int(u.split("bus")[1]))  # Gen_bus4 before Gen_bus10
    assert any(p["no_action_needed"] for p in plan) and any(p["actions"] for p in plan)


def test_the_n1_option_screens_the_same_through_the_contingency_endpoint(client, corrective):
    """The screen the study reports is the ordinary N-1 screen of that dispatch:
    the same limits (the network's own, 100 % here) unless one is set."""
    option = next(o for o in corrective["options"] if o["id"] == "n1_corrective")
    assert corrective["n1_study"]["emergency_loading_pct"] == 100.0
    screen = _ok(client.post("/api/contingency/simulate_all", json={"option": option["ref"]}))
    assert screen["thresholds_used"]["max_line_loading_pct"] == 100.0
    assert screen["total_outages_tested"] == option["n1"]["screen"]["outages_tested"]
    assert screen["total_outages_causing_violations"] == option["n1"]["screen"]["outages_with_violations"]


def test_a_preventive_dispatch_is_more_secure_and_says_what_it_could_not_secure(preventive):
    by_id = {o["id"]: o for o in preventive["options"]}
    secure, reference = by_id["n1_preventive"]["n1"], by_id["least_disturbance"]["n1"]
    assert secure["screen"]["outages_with_violations"] < reference["screen"]["outages_with_violations"]
    assert by_id["n1_preventive"]["cost_eur_per_h"] > by_id["cheapest"]["cost_eur_per_h"]   # security costs
    assert "corrective_plan" not in secure
    for entry in secure["not_secured"]:                 # named as everywhere, each branch once
        assert entry["outage"] in secure["studied"] and entry["remaining"]
        names = [r.rsplit(" at ", 1)[0] for r in entry["remaining"]]
        assert len(names) == len(set(names))
        assert all(" [L" in n or " [T" in n or n.startswith("Bus_") for n in names)
    skipped = preventive["n1_study"]["skipped"]
    assert any("without supply" in s["reason"] for s in skipped)          # an outage that islands a bus


@needs_ipopt
def test_chosen_outages_and_their_refusals(client):
    out = _options(client, timestamp=NIGHT, n1="preventive", n1_outages=["L31"])
    assert out["n1_study"]["outages_studied"] == ["Bus_22 -> Bus_23 [L31]"]
    r = client.post("/api/flexibility/optimize", json={"timestamp": NIGHT, "n1": "preventive",
                                                       "n1_outages": ["L999"]})
    assert r.status_code == 400 and "L999" in r.json()["detail"]
    r = client.post("/api/flexibility/optimize", json={"timestamp": NIGHT, "n1": "preventive",
                                                       "near_optimal": True})
    assert r.status_code == 400 and "one at a time" in r.json()["detail"]
    r = client.post("/api/flexibility/optimize", json={"timestamp": NIGHT, "n1": "sometimes"})
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# scopf.residuals: what a post-outage state still breaks, in words
# ---------------------------------------------------------------------------

def test_residuals_name_each_branch_once_worst_first():
    from pyomo.environ import ConcreteModel, NonNegativeReals, Var
    m = ConcreteModel()
    m.s_branch = Var(range(3), within=NonNegativeReals)
    m.s_vlo = Var([4, 5], within=NonNegativeReals, initialize=0.0)       # one pair per free bus, as soften makes
    m.s_vhi = Var([4, 5], within=NonNegativeReals, initialize=0.0)
    m._soft_rows = [(1, 2), (2, 1), (3, 4)]             # one line measured at both ends, another line
    m._emergency, m._band = 1.0, (0.95, 1.05)
    m.s_branch[0].value, m.s_branch[1].value, m.s_branch[2].value = 0.21, 0.44, 0.0
    m.s_vlo[4].value, m.s_vhi[5].value = 0.02, 0.00001
    names = {frozenset((1, 2)): "Bus_1 -> Bus_2 [L7]"}
    out = scopf.residuals(m, lambda i, j: names.get(frozenset((i, j)), f"{i}-{j}"), lambda b: f"Bus_{b}")
    assert out == ["Bus_1 -> Bus_2 [L7] at 120.0 %", "Bus_4 at 0.930 p.u."]  # the worse end; 1e-5 is tolerance
