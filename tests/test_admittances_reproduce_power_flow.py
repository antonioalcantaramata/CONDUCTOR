"""
The optimisation's admittances reproduce pandapower's power flow.

The OPF does not use pandapower's own model: it reads a database of branch
π-model admittances built once from it (`network_loader.build_ybus_full`).
Evaluated at a converged power flow, those admittances must give back every
bus's injection — the claim the documentation makes ("How it works", the
optimal power flow): within 1e-6 MVA on the bundled networks.
"""

import cmath
from collections import defaultdict

import pandapower as pp
import pytest

import network_loader as nl


def _distribution_net():
    """A transformer with iron losses (pfe_kw) and a cable: the shunt
    conductance the bundled cases do not have."""
    net = pp.create_empty_network()
    hv, mv, end = (pp.create_bus(net, vn_kv=kv) for kv in (110.0, 20.0, 20.0))
    pp.create_ext_grid(net, hv, vm_pu=1.02)
    pp.create_transformer(net, hv, mv, std_type="25 MVA 110/20 kV")
    pp.create_line(net, mv, end, length_km=8.0, std_type="NA2XS2Y 1x240 RM/25 12/20 kV")
    pp.create_load(net, end, p_mw=6.0, q_mvar=2.0)
    return net


@pytest.mark.parametrize("profile", ["pglib_case30", "pglib_case14", "ieee14", "distribution"])
def test_each_bus_injection_is_given_back(profile):
    net = _distribution_net() if profile == "distribution" else nl.load_network(nl.load_profile(profile))
    db = nl.build_ybus_full(net)
    pp.runpp(net, numba=False, calculate_voltage_angles=True)
    v = {b: cmath.rect(net.res_bus.at[b, "vm_pu"], net.res_bus.va_degree.at[b] * cmath.pi / 180)
         for b in net.bus.index[net.bus.in_service.astype(bool)]}
    flows = defaultdict(complex)
    for ((i, j), tap), g_ff in db["Yff_r"].items():
        yff = complex(g_ff, db["Yff_i"][(i, j), tap])
        yft = complex(db["Yft_r"][(i, j), tap], db["Yft_i"][(i, j), tap])
        flows[i] += v[i] * (yff * v[i] + yft * v[j]).conjugate() * net.sn_mva
    for b, s in flows.items():
        # pandapower's bus result is what the bus consumes (shunts included): the branches carry the rest away.
        consumed = complex(net.res_bus.at[b, "p_mw"], net.res_bus.at[b, "q_mvar"])
        assert abs(s + consumed) < 1e-6, (profile, b, s, consumed)
