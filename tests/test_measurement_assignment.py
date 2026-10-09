"""
Measured demand and production reach the network once, and only where they
belong (code review batch D, findings 3.1 and 3.2).

A measurement is the total of its substation or bus. Every load matched to it
used to receive that whole total, a bus with both a `gen` and an `sgen` got its
production twice, and a missing measurement row was filled from a different bus
by a three-letter fuzzy match — each silently, with nothing reported unmatched.
"""

import pandas as pd
import pandapower as pp
import pandapower.networks as pn
import pytest

import load_gen_assignment as lg


def _measurements(net, scale: float = 1.3, drop: tuple = ()) -> pd.DataFrame:
    """One row per load bus, the way the synthetic series is built: the bus
    total of the network's loads, scaled to stand in for a measured tick."""
    totals = net.load.groupby("bus")["p_mw"].sum() * scale
    return pd.DataFrame({
        "substation_name": [f"Bus_{b}" for b in totals.index if b not in drop],
        "consumption": [v for b, v in totals.items() if b not in drop],
        "production": [0.0 for b in totals.index if b not in drop],
    })


@pytest.fixture
def net():
    n = pn.case14()
    n.bus["name"] = None  # MATPOWER cases carry no names → "Bus_<idx>"
    return n


class TestLoads:
    def test_two_loads_on_a_bus_share_its_measurement(self, net):
        pp.create_load(net, bus=2, p_mw=30.0, name="second load on bus 2")
        meas = _measurements(net)
        net, unmatched = lg.assign_load_values_from_measurements(net, meas, [])
        assert net.load["p_mw"].sum() == pytest.approx(meas["consumption"].sum())
        on_bus2 = net.load[net.load.bus == 2]["p_mw"]
        measured = meas.set_index("substation_name").at["Bus_2", "consumption"]
        assert on_bus2.sum() == pytest.approx(measured)
        assert not unmatched

    def test_the_split_follows_the_networks_own_shares(self, net):
        first = float(net.load.loc[net.load.bus == 2, "p_mw"].iloc[0])
        pp.create_load(net, bus=2, p_mw=first, name="equal twin")
        net, _ = lg.assign_load_values_from_measurements(net, _measurements(net), [])
        a, b = net.load.loc[net.load.bus == 2, "p_mw"]
        assert a == pytest.approx(b)

    def test_shares_survive_repeated_ticks(self, net):
        """Later ticks must split by the original shares, not by values an
        earlier tick wrote into p_mw."""
        pp.create_load(net, bus=2, p_mw=10.0)
        first = lg.assign_load_values_from_measurements(net, _measurements(net, 1.3), [])[0]
        ratio = first.load.loc[first.load.bus == 2, "p_mw"].tolist()
        second = lg.assign_load_values_from_measurements(first, _measurements(pn.case14(), 0.7), [])[0]
        again = second.load.loc[second.load.bus == 2, "p_mw"].tolist()
        assert ratio[0] / ratio[1] == pytest.approx(again[0] / again[1])

    def test_one_load_per_bus_is_unchanged(self, net):
        meas = _measurements(net)
        net, _ = lg.assign_load_values_from_measurements(net, meas, [])
        expected = meas.set_index("substation_name")["consumption"]
        for _, row in net.load.iterrows():
            assert row.p_mw == pytest.approx(expected[f"Bus_{int(row.bus)}"])

    def test_a_missing_row_is_reported_not_borrowed(self, net):
        """Bus_13's load used to receive Bus_1's measurement."""
        static = float(net.load.loc[net.load.bus == 13, "p_mw"].iloc[0])
        net, unmatched = lg.assign_load_values_from_measurements(
            net, _measurements(net, drop=(13,)), [])
        assert "Bus_13" in unmatched
        assert float(net.load.loc[net.load.bus == 13, "p_mw"].iloc[0]) == pytest.approx(static)


class TestMatching:
    def test_generic_names_do_not_fuzzy_match(self):
        assert lg.match_substation("Bus_13", ["Bus_1", "Bus_2"]) is None

    def test_measured_style_names_still_match(self):
        assert lg.match_substation("Allinge 10 kV", ["Allinge", "Bravo"]) == "Allinge"

    def test_a_unique_prefix_still_matches(self):
        assert lg.match_substation("Rønne Syd", ["Rønnevang", "Bravo"]) == "Rønnevang"

    def test_an_ambiguous_prefix_does_not(self):
        assert lg.match_substation("Rønne Syd", ["Rønnevang", "Rønnebakke"]) is None


class TestGeneration:
    def test_gen_and_sgen_on_one_bus_share_its_production(self):
        shares = lg._split_between_tables(
            pd.DataFrame({"bus": [1], "max_p_mw": [60.0]}),
            pd.DataFrame({"bus": [1, 3], "max_p_mw": [20.0, 50.0]}),
            {1: 40.0, 3: 10.0},
        )
        gen_share, sgen_share = shares
        assert gen_share[1] + sgen_share[1] == pytest.approx(40.0)
        assert gen_share[1] == pytest.approx(30.0)  # 60 of 80 MW installed
        assert sgen_share[3] == pytest.approx(10.0)  # only sgens there


class TestProposedElements:
    """Rows a network proposal added follow their own profile, never the
    measurements (sandbox hazard A.3 (1)): a 50 MW data centre on a measured
    bus used to receive that bus's measured consumption and vanish."""

    def test_a_proposed_load_keeps_its_own_value(self, net):
        meas = _measurements(net)
        idx = pp.create_load(net, bus=2, p_mw=50.0, q_mvar=10.0, name="Data centre")
        net.load["proposed"] = False
        net.load.at[idx, "proposed"] = True
        net, unmatched = lg.assign_load_values_from_measurements(net, meas, [])
        assert net.load.at[idx, "p_mw"] == pytest.approx(50.0)
        assert net.load.at[idx, "q_mvar"] == pytest.approx(10.0)
        # The measured loads on that bus still carry the whole measurement.
        measured = meas.set_index("substation_name").at["Bus_2", "consumption"]
        others = net.load[(net.load.bus == 2) & ~net.load.proposed]["p_mw"].sum()
        assert others == pytest.approx(measured)
        assert not unmatched

    def test_a_proposed_unit_keeps_its_output_on_a_generic_network(self):
        n = pn.case14()
        n.bus["name"] = None
        bus = int(n.gen.bus.iloc[0])
        idx = pp.create_sgen(n, bus=bus, p_mw=7.0, max_p_mw=100.0, name="New wind")
        n.sgen["proposed"] = False
        n.sgen.at[idx, "proposed"] = True
        meas = pd.DataFrame({"substation_name": [f"Bus_{bus}"], "consumption": [0.0],
                             "production": [30.0]})
        n, _ = lg.assign_generators_values_from_measurements(n, meas, [])
        assert n.sgen.at[idx, "p_mw"] == pytest.approx(7.0)
        # The measured unit gets the whole measurement, not a share diluted by
        # the proposal's 100 MW of capacity.
        assert n.gen.loc[n.gen.bus == bus, "p_mw"].sum() == pytest.approx(30.0)

    def test_a_proposed_unit_survives_the_measured_path_rebuild(self, net):
        net.gen = net.gen.iloc[0:0]          # measured-substation path
        net.load["substation_name"] = [f"Bus_{b}" for b in net.load.bus]
        pp.create_sgen(net, bus=3, p_mw=12.0, name="New PV")
        net.sgen["proposed"] = True
        meas = _measurements(net)
        net, _ = lg.assign_generators_values_from_measurements(net, meas, [])
        kept = net.sgen[lg._proposed_mask(net.sgen)]
        assert len(kept) == 1 and kept["p_mw"].iloc[0] == pytest.approx(12.0)
