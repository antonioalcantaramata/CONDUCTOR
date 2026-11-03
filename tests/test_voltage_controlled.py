"""The security assessment says which buses a unit holds at a setpoint, and
that unit's margin. On the IEEE 14 case Bus_2 sat at its condenser's 0.942
p.u. setpoint in every hour; reported only as "0.002 above the 0.94 floor",
the agent and the suggestion agent took it for a weak spot each time. On the
IEEE 30 demo every unit holds its bus (Gen_bus1 at 1.025 p.u.)."""

import pytest

pytest.importorskip("fastapi.testclient")


def test_a_held_bus_is_reported_with_its_units_reactive_headroom(backend_client):
    r = backend_client.post("/api/grid/rsa", json={"timestamp": "2000-01-04 03:15:00"}).json()
    held = {v["bus"]: v for v in r["voltage_controlled_buses"]}
    assert set(held) == {"Bus_0", "Bus_1", "Bus_4", "Bus_7", "Bus_10", "Bus_12"}  # five units, the grid
    bus1 = held["Bus_1"]
    assert bus1["unit"] == "Gen_bus1" and bus1["setpoint_pu"] == pytest.approx(1.025)
    voltage = {v["bus_name"]: v["vm_pu"] for v in r["all_voltages"]}["Bus_1"]
    assert voltage == pytest.approx(bus1["setpoint_pu"], abs=1e-6)       # held exactly
    assert bus1["q_headroom_up_mvar"] == pytest.approx(bus1["q_max_mvar"] - bus1["q_mvar"], abs=1e-3)
    assert bus1["q_headroom_down_mvar"] == pytest.approx(bus1["q_mvar"] - bus1["q_min_mvar"], abs=1e-3)
    snap = backend_client.post("/api/network/snapshot", json={"timestamp": "2000-01-04 03:15:00"}).json()
    q_snap = {g["name"]: g["Qg_mvar"] for g in snap["generators"]}["Gen_bus1"]
    assert bus1["q_mvar"] == pytest.approx(q_snap, abs=1e-3)


def test_the_external_grid_bus_is_listed_without_a_reactive_limit(backend_client):
    r = backend_client.post("/api/grid/rsa", json={}).json()
    (grid,) = [v for v in r["voltage_controlled_buses"] if v["unit"] == "external grid"]
    assert grid["q_max_mvar"] is None and grid["q_headroom_up_mvar"] is None
