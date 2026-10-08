"""Profiles state voltage setpoints their case can hold (network_loader).

PGLib case14 ships 1.0 p.u. everywhere — OPF placeholders. Held in a power
flow, they made the condensers deliver up to 3.8× their reactive limit in every
hour of the synthetic week.
"""

import network_loader as nl


def test_profile_setpoints_are_applied(case14_net):
    gen_bus = int(case14_net.gen.bus.iloc[1])
    applied = nl.apply_voltage_setpoints(case14_net, {"ext_grid_vm_pu": 1.03,
                                                      "gen_vm_pu": {gen_bus: 0.99}})
    assert case14_net.ext_grid.vm_pu.iloc[0] == 1.03
    assert case14_net.gen.loc[case14_net.gen.bus == gen_bus, "vm_pu"].iloc[0] == 0.99
    assert applied == {"ext_grid": 1.03, f"gen@bus{gen_bus}": 0.99}


def test_a_profile_without_setpoints_changes_nothing(case14_net):
    before = case14_net.gen.vm_pu.tolist(), case14_net.ext_grid.vm_pu.tolist()
    assert nl.apply_voltage_setpoints(case14_net, {}) == {}
    assert (case14_net.gen.vm_pu.tolist(), case14_net.ext_grid.vm_pu.tolist()) == before


def test_an_unknown_bus_is_ignored(case14_net):
    assert nl.apply_voltage_setpoints(case14_net, {"gen_vm_pu": {999: 1.0}}) == {}
