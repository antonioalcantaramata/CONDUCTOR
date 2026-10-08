import numpy as np
import pandas as pd
import pytest

import synthetic_timeseries as st


class TestBusName:
    def test_uses_name_column_when_present(self, case14_net):
        net = case14_net
        net.bus.at[0, "name"] = "Foxtrot"
        assert st._bus_name(net, 0) == "Foxtrot"

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


class TestPeakLoadFactor:
    """The network's own load is its design point; the usual daily peak sits
    there, not at the profile shape's ~1.22× (which ran PGLib case14 past its
    reactive capability every afternoon)."""

    def _ratio(self, net, **kw):
        base = net.load.groupby("bus")["p_mw"].sum()
        _, meas = st.generate(net, n_days=7, resolution_min=60, seed=3, stress_events=False, **kw)
        totals = [df["consumption"].sum() for df in meas.values()]
        return max(totals) / base.sum(), min(totals) / base.sum()

    def test_default_peaks_at_the_networks_load(self, case14_net):
        high, low = self._ratio(case14_net)
        assert 0.97 <= high <= 1.06 and low < 0.75

    def test_explicit_factor(self, case14_net):
        high, _ = self._ratio(case14_net, peak_load_factor=0.9)
        assert 0.87 <= high <= 0.95

    def test_none_keeps_the_raw_shape(self, case14_net):
        high, _ = self._ratio(case14_net, peak_load_factor=None)
        assert high > 1.15

    def test_same_seed_same_series_up_to_scale(self, case14_net):
        _, a = st.generate(case14_net, n_days=1, resolution_min=60, seed=5, peak_load_factor=None)
        _, b = st.generate(case14_net, n_days=1, resolution_min=60, seed=5)
        ratio = {round(b[t]["consumption"].sum() / a[t]["consumption"].sum(), 6) for t in a}
        assert len(ratio) == 1


class TestForecast:
    """The synthetic forecast covers the last measured days (measured values
    with a day-ahead-sized error) and the days after — so forecast and
    measured can be compared on shared timestamps."""

    def _series(self, net, **kw):
        ts, meas = st.generate(net, n_days=3, resolution_min=60, seed=2)
        fts, fc = st.forecast(net, ts, meas, n_days=2, resolution_min=60, seed=9, **kw)
        return ts, meas, fts, fc

    def test_overlaps_the_last_days_then_looks_ahead(self, case14_net):
        ts, _, fts, _ = self._series(case14_net, overlap_days=1)
        shared = sorted(set(ts) & set(fts))
        assert shared == ts[-24:]
        assert fts[24] > ts[-1] and len(fts) == 24 + 48
        assert fts == sorted(fts)

    def test_shared_hours_are_a_forecast_of_what_was_measured(self, case14_net):
        ts, meas, fts, fc = self._series(case14_net, overlap_days=1, load_error=0.04)
        ratios = [fc[t]["consumption"].sum() / meas[t]["consumption"].sum() for t in ts[-24:]]
        assert all(0.8 < r < 1.2 for r in ratios)
        assert len({round(r, 6) for r in ratios}) > 1  # an error, not a copy

    def test_no_overlap_is_the_old_look_ahead(self, case14_net):
        ts, _, fts, _ = self._series(case14_net, overlap_days=0)
        assert not set(ts) & set(fts) and fts[0] > ts[-1]

    def test_deterministic(self, case14_net):
        a = self._series(case14_net)[3]
        b = self._series(case14_net)[3]
        assert all(a[t].equals(b[t]) for t in a)
