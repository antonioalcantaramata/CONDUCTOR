"""The security assessment says which buses a unit holds at a setpoint, and
that unit's margin. Bus_2 sits at its condenser's 0.942 p.u. setpoint in every
hour; reported only as "0.002 above the 0.94 floor", the agent and the
suggestion agent took it for a weak spot each time."""

import pytest

pytest.importorskip("fastapi.testclient")


def test_the_condenser_bus_is_reported_with_its_reactive_headroom(backend_client):
    r = backend_client.post("/api/grid/rsa", json={"timestamp": "2000-01-04 03:15:00"}).json()
    held = {v["bus"]: v for v in r["voltage_controlled_buses"]}
    bus2 = held["Bus_2"]
    assert bus2["unit"] == "Gen_bus2" and bus2["setpoint_pu"] == pytest.approx(0.942)
    voltage = {v["bus_name"]: v["vm_pu"] for v in r["all_voltages"]}["Bus_2"]
    assert voltage == pytest.approx(bus2["setpoint_pu"], abs=1e-6)       # held exactly
    assert bus2["q_headroom_up_mvar"] == pytest.approx(bus2["q_max_mvar"] - bus2["q_mvar"], abs=1e-3)
    snap = backend_client.post("/api/network/snapshot", json={"timestamp": "2000-01-04 03:15:00"}).json()
    q_snap = {g["name"]: g["Qg_mvar"] for g in snap["generators"]}["Gen_bus2"]
    assert bus2["q_mvar"] == pytest.approx(q_snap, abs=1e-3)


def test_the_external_grid_bus_is_listed_without_a_reactive_limit(backend_client):
    r = backend_client.post("/api/grid/rsa", json={}).json()
    (grid,) = [v for v in r["voltage_controlled_buses"] if v["unit"] == "external grid"]
    assert grid["q_max_mvar"] is None and grid["q_headroom_up_mvar"] is None
