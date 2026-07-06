import numpy as np
import pandas as pd
import pytest

import synthetic_timeseries as st


class TestBusName:
    def test_uses_name_column_when_present(self, case14_net):
        net = case14_net
        net.bus.at[0, "name"] = "Nexo"
        assert st._bus_name(net, 0) == "Nexo"

    def test_falls_back_to_bus_id_when_blank(self, case14_net):
        net = case14_net
        net.bus.at[0, "name"] = "   "
        assert st._bus_name(net, 0) == "Bus_0"

    def test_falls_back_when_no_name_column(self, case14_net):
        net = case14_net
        net.bus = net.bus.drop(columns=["name"])
        assert st._bus_name(net, 3) == "Bus_3"


class TestGetMaxRef:
    def test_prefers_max_p_mw(self):
        row = pd.Series({"max_p_mw": 50.0, "p_mw": 10.0})
        assert st._get_max_ref(row) == 50.0

    def test_falls_back_to_p_mw_when_max_missing(self):
        row = pd.Series({"p_mw": 10.0})
        assert st._get_max_ref(row) == 10.0

    def test_falls_back_to_p_mw_when_max_is_nan(self):
        row = pd.Series({"max_p_mw": float("nan"), "p_mw": 10.0})
        assert st._get_max_ref(row) == 10.0

    def test_returns_zero_when_nothing_valid(self):
        row = pd.Series({"max_p_mw": 0.0, "p_mw": 0.0})
        assert st._get_max_ref(row) == 0.0


class TestLoadScale:
    @pytest.mark.parametrize("profile", ["residential", "industrial", "flat"])
    def test_stays_within_bounds(self, profile):
        rng = np.random.default_rng(0)
        hours = np.arange(24, dtype=float)
        dow = np.zeros(24)
        alpha = st._load_scale(hours, dow, profile, rng)
        assert alpha.shape == (24,)
        assert (alpha >= 0.05).all() and (alpha <= 1.5).all()


class TestGenScale:
    @pytest.mark.parametrize("profile", ["wind", "solar", "flat"])
    def test_stays_within_bounds(self, profile):
        rng = np.random.default_rng(0)
        hours = np.arange(24, dtype=float)
        alpha = st._gen_scale(profile, 24, hours, rng)
        assert alpha.shape == (24,)
        assert (alpha >= 0.0).all() and (alpha <= 1.0).all()

    def test_solar_is_zero_outside_daylight_hours(self):
        rng = np.random.default_rng(0)
        hours = np.array([0.0, 2.0, 22.0, 23.0])
        alpha = st._gen_scale("solar", 4, hours, rng)
        assert (alpha == 0.0).all()


class TestGenerate:
    def test_is_deterministic_given_seed(self, case14_net):
        ts1, meas1 = st.generate(case14_net, n_days=1, resolution_min=60, seed=1, stress_events=False)
        ts2, meas2 = st.generate(case14_net, n_days=1, resolution_min=60, seed=1, stress_events=False)
        assert ts1 == ts2
        pd.testing.assert_frame_equal(meas1[ts1[0]], meas2[ts2[0]])

    def test_timestamps_match_requested_resolution(self, case14_net):
        ts, _ = st.generate(case14_net, n_days=1, resolution_min=60, seed=1, stress_events=False)
        assert len(ts) == 24
        assert ts[0] == "2000-01-01 00:00:00"

    def test_measurement_columns_and_nonnegativity(self, case14_net):
        _, meas = st.generate(case14_net, n_days=1, resolution_min=60, seed=1, stress_events=False)
        df = next(iter(meas.values()))
        assert list(df.columns) == ["substation_name", "production", "consumption"]
        assert (df["production"] >= 0).all()
        assert (df["consumption"] >= 0).all()

    def test_degenerate_network_with_no_load_or_gen_returns_flat_row(self, case14_net):
        net = case14_net
        net.load = net.load.iloc[0:0]
        net.gen = net.gen.iloc[0:0]
        net.sgen = net.sgen.iloc[0:0]

        ts, meas = st.generate(net, n_days=2, resolution_min=60, seed=1, stress_events=False)

        assert len(ts) == 1
        row = meas[ts[0]].iloc[0]
        assert row["substation_name"] == "EXT_GRID"
        assert row["production"] == 0.0
        assert row["consumption"] == 0.0
