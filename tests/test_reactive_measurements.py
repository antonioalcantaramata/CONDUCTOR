"""
Measured reactive power in uploaded data.

Two optional columns, `consumption_mvar` and `production_mvar`, carry the
reactive power the data used to lack. When given they are always used — the
load power factor only fills cells left empty — and every upload gets a quality
report: coverage, empty columns, implausible power factors, kvar instead of
Mvar, filled-in zeros, missing-value markers.
"""

import io
import math

import numpy as np
import pandapower.networks as pn
import pandas as pd
import pytest

import data_quality as dq
import generators as gm
import load_gen_assignment as lg

TAN95 = math.tan(math.acos(0.95))


def _csv(rows, cols="timestamp,substation_name,consumption_mw,production_mw,consumption_mvar,production_mvar"):
    return ("\n".join([cols, *rows]) + "\n").encode()


class TestParsing:
    def test_optional_columns_kept_missing_stays_missing(self):
        m, ts, notes = dq.parse(_csv(["2001-01-01 00:00,A,10,0,3,", "2001-01-01 00:00,B,5,2,,0.5",
                                      "2001-01-01 00:15,A,10,0,-9999,"]))
        assert ts == ["2001-01-01 00:00:00", "2001-01-01 00:15:00"]
        first = m[ts[0]].set_index("substation_name")
        assert first.at["A", "consumption_q"] == 3 and np.isnan(first.at["B", "consumption_q"])
        assert first.at["B", "production_q"] == 0.5
        assert np.isnan(m[ts[1]].set_index("substation_name").at["A", "consumption_q"])   # a marker: missing
        assert notes["markers"] == {"consumption_mvar": 1}
        assert notes["reactive_columns"] == ["consumption_mvar", "production_mvar"]
        assert dq.has_reactive(m) == {"consumption": True, "production": True}

    def test_without_them_nothing_changes_and_an_empty_one_is_ignored(self):
        m, _, notes = dq.parse(_csv(["2001-01-01,A,10,0"], "timestamp,substation_name,consumption_mw,production_mw"))
        assert list(next(iter(m.values())).columns) == ["substation_name", "production", "consumption"]
        m, _, notes = dq.parse(_csv(["2001-01-01,A,10,0,,"]))
        assert notes["empty_columns"] == ["consumption_mvar", "production_mvar"]
        assert dq.has_reactive(m) == {"consumption": False, "production": False}
        with pytest.raises(ValueError, match="optional: consumption_mvar"):
            dq.parse(b"timestamp,substation_name\n2001,A\n")


def _codes(content, voltage=None):
    m, _, notes = dq.parse(content)
    return {f["code"]: f for f in dq.quality(m, notes, voltage)["findings"]}


class TestQuality:
    def test_coverage_and_the_fallback_said(self):
        f = _codes(_csv(["2001-01-01,A,10,0,3,", "2001-01-01,B,10,0,,"]))
        assert f["reactive_consumption_coverage"]["coverage"] == 0.5
        assert f["reactive_consumption_coverage"]["level"] == "warning"

    def test_implausible_values_are_flagged(self):
        f = _codes(_csv(["2001-01-01,A,10,0,40,", "2001-01-01,B,10,0,-2,", "2001-01-01,C,0,0,1,"]))
        assert f["low_power_factor"]["count"] == 1 and f["leading_load"]["count"] == 1
        assert f["reactive_without_load"]["count"] == 1
        kvar = _codes(_csv([f"2001-01-01,{n},10,0,3000," for n in "ABC"]))
        assert kvar["reactive_units"]["level"] == "error"
        zero = _codes(_csv([f"2001-01-01,{n},10,0,0," for n in "ABC"]))
        assert "reactive_all_zero" in zero

    def test_missing_active_power_is_interpolated_in_time_and_reported(self):
        content = (b"timestamp,substation_name,consumption_mw,production_mw\n"
                   b"2001-01-01 00:00,A,10,0\n2001-01-01 00:15,A,-9999,0\n2001-01-01 00:30,A,,0\n"
                   b"2001-01-01 00:45,A,16,0\n2001-01-01 00:00,B,,0\n2001-01-01 00:15,B,-9999,0\n")
        m, ts, notes = dq.parse(content)
        a = [m[t].set_index("substation_name").at["A", "consumption"] for t in ts]
        assert a == pytest.approx([10, 12, 14, 16])                    # between its neighbours
        assert m[ts[0]].set_index("substation_name").at["B", "consumption"] == 0.0
        f = {x["code"]: x for x in dq.quality(m, notes)["findings"]}
        assert f["active_interpolated"]["count"] == 2
        assert "1 missing-value marker(s) and 1 empty cell(s)" in f["active_interpolated"]["message"]
        assert f["active_no_values"]["substations"] == ["B"]

    def test_measured_generation_at_a_voltage_unit_is_not_imposed(self):
        f = _codes(_csv(["2001-01-01,A,0,10,,3"]), voltage={"A"})
        assert f["reactive_at_voltage_units"]["substations"] == ["A"]

    def test_no_reactive_columns_is_said_plainly(self):
        m, _, notes = dq.parse(_csv(["2001-01-01,A,10,0"], "timestamp,substation_name,consumption_mw,production_mw"))
        report = dq.quality(m, notes)
        assert [x["code"] for x in report["findings"]] == ["no_reactive_consumption"]
        assert report["summary"].startswith("no reactive measurements")


@pytest.fixture
def net():
    n = pn.case14()
    n.bus["name"] = None
    n, _, _ = __import__("network_loader").gen_to_sgen(n)
    return n


def _meas(rows):
    return pd.DataFrame(rows, columns=["substation_name", "consumption", "production", "consumption_q",
                                       "production_q"])


class TestAssignment:
    def test_loads_use_the_measurement_and_the_power_factor_only_fills_gaps(self, net):
        names = [f"Bus_{b}" for b in sorted(set(net.load.bus))]
        rows = [[n, 10.0, 0.0, (2.0 if n != "Bus_9" else np.nan), np.nan] for n in names]
        net, _ = lg.assign_load_values_from_measurements(net, _meas(rows), names, power_factor=0.95)
        by_bus = net.load.set_index("bus")
        assert by_bus.at[4, "q_mvar"] == pytest.approx(2.0)
        assert by_bus.at[9, "q_mvar"] == pytest.approx(10 * TAN95)          # the gap: the power factor
        assert net["_q_measured_share"] == pytest.approx((len(names) - 1) / len(names), abs=1e-3)

    def test_a_fixed_pf_unit_follows_measured_generation_and_its_model_does_not_overwrite(self, net):
        model = gm.merge({}, gm.normalise(net, [{"unit": "Gen_bus1", "pf": 0.9}]))
        net = gm.apply(net, model)
        names = [f"Bus_{b}" for b in range(14)]
        rows = [[n, 0.0, (20.0 if n == "Bus_1" else 0.0), np.nan, (4.0 if n == "Bus_1" else np.nan)] for n in names]
        net, _ = lg.assign_generators_values_from_measurements(net, _meas(rows), names)
        gm.set_reactive(net)
        i = net.sgen.index[net.sgen.name == "Gen_bus1"][0]
        assert net.sgen.at[i, "p_mw"] == pytest.approx(20.0) and net.sgen.at[i, "q_mvar"] == pytest.approx(4.0)
        rows[1][4] = np.nan                                                  # no measurement: its model again
        net, _ = lg.assign_generators_values_from_measurements(net, _meas(rows), names)
        gm.set_reactive(net)
        assert net.sgen.at[i, "q_mvar"] == pytest.approx(20.0 * math.tan(math.acos(0.9)))


class TestThroughTheBackend:
    @pytest.fixture
    def upload(self, tmp_path, monkeypatch):
        pytest.importorskip("fastapi.testclient")
        import main_backend as mb
        from fastapi.testclient import TestClient
        monkeypatch.setenv("GRID_PROFILE", "pglib_case14")
        monkeypatch.setattr(mb, "SYSTEMS_DIR", str(tmp_path / "systems"))
        monkeypatch.setattr(mb, "DATA_FILES_DIR", str(tmp_path / "data_files"))
        for name in ("_LAST_UPLOAD_SENTINEL", "_LAST_TIMESERIES_SENTINEL", "_LAST_FORECAST_SENTINEL",
                     "_PROPOSALS_FILE", "_ASSUMPTIONS_FILE", "_SCENARIO_FILE"):
            monkeypatch.setattr(mb, name, str(tmp_path / f"{name}.json"))
        saved = dict(mb.app_data)
        with TestClient(mb.app) as c:
            names = sorted({f"Bus_{b}" for b in mb.app_data["net"].load.bus})
            rows = []
            for k, ts in enumerate(("2001-01-01 00:00:00", "2001-01-01 00:15:00")):
                rows += [f"{ts},{n},20,{10 if n == 'Bus_1' else 0},{'' if (n == 'Bus_9' and k) else 4},"
                         f"{3 if n == 'Bus_1' else ''}" for n in names]
            r = c.post("/api/data/upload", data={"kind": "measurements", "overwrite": "true"},
                       files={"file": ("q.csv", io.BytesIO(_csv(rows)), "text/csv")})
            assert r.status_code == 200, r.text
            yield c, r.json()
        mb.app_data.clear()
        mb.app_data.update(saved)

    def test_studies_use_the_measured_reactive_power(self, upload):
        c, out = upload
        assert out["quality"]["reactive"] == {"consumption": True, "production": True}
        snap = c.post("/api/network/snapshot", json={"timestamp": "2001-01-01 00:00"}).json()
        assert {round(l["Q_mvar"], 4) for l in snap["loads"]} == {4.0}
        assert snap["load_power_factor"]["measured_reactive_power"]["share_of_load"] == 1.0
        g1 = next(g for g in snap["generators"] if g["name"] == "Gen_bus1")
        assert (g1["Pg_mw"], g1["Qg_mvar"]) == (10.0, 3.0)
        gap = c.post("/api/network/snapshot", json={"timestamp": "2001-01-01 00:15", "load_power_factor": 0.9}).json()
        q9 = next(l["Q_mvar"] for l in gap["loads"] if l["bus"] == "Bus_9")
        assert q9 == pytest.approx(20 * math.tan(math.acos(0.9)), abs=1e-3)   # an override reaches only gaps
        assert {round(l["Q_mvar"], 4) for l in gap["loads"] if l["bus"] != "Bus_9"} == {4.0}

    def test_the_operator_and_the_agent_can_see_it(self, upload):
        c, _ = upload
        assert c.get("/api/time/timeline").json()["reactive_measured"]["measurements"]["consumption"] is True
        assert c.get("/api/grid_constants").json()["measurements"]["reactive_measured"]["consumption"] is True
        quality = c.get("/api/data/quality").json()["measurements"]
        assert quality["source"] == "uploaded" and any(f["code"] == "reactive_consumption_coverage"
                                                       for f in quality["findings"])
        series = c.post("/api/data/series", json={"substations": ["Bus_4"]}).json()
        assert series["totals"]["consumption_q"][0] == pytest.approx(4.0 * len(series["available_substations"]))
        assert series["substations"]["Bus_4"]["consumption_q"][0] == 4.0
        assert c.get("/api/assumptions").json()["reactive_measured"]["measurements"]["consumption"] is True


def test_the_data_tab_offers_reactive_quantities_only_when_present():
    from llm_agent.agent import data_view as dv
    plain = {"totals": {"consumption": [1.0], "production": [0.0]}}
    rich = {"totals": {"consumption": [1.0], "production": [0.0], "consumption_q": [0.3]},
            "substations": {"A": {"consumption_q": [0.1]}}}
    assert "consumption_q" not in dv.quantities(plain) and "consumption_q" in dv.quantities(plain, rich)
    assert dv.values(rich, dv.TOTAL, "consumption_q") == [0.3] and dv.unit("consumption_q") == "Mvar"
    assert dv.values(rich, "A", "consumption_q") == [0.1] and dv.unit("net") == "MW"



def test_voltage_units_can_be_switched_to_follow_the_measured_reactive_power(tmp_path, monkeypatch):
    pytest.importorskip("fastapi.testclient")
    import main_backend as mb
    from fastapi.testclient import TestClient
    monkeypatch.setenv("GRID_PROFILE", "pglib_case14")
    monkeypatch.setattr(mb, "SYSTEMS_DIR", str(tmp_path / "systems"))
    monkeypatch.setattr(mb, "DATA_FILES_DIR", str(tmp_path / "data_files"))
    for name in ("_LAST_UPLOAD_SENTINEL", "_LAST_TIMESERIES_SENTINEL", "_LAST_FORECAST_SENTINEL",
                 "_PROPOSALS_FILE", "_ASSUMPTIONS_FILE", "_SCENARIO_FILE"):
        monkeypatch.setattr(mb, name, str(tmp_path / f"{name}.json"))
    saved = dict(mb.app_data)
    try:
        with TestClient(mb.app) as c:
            c.post("/api/generators", json={"changes": [{"unit": "Gen_bus1", "control": "voltage"}]})
            names = sorted({f"Bus_{b}" for b in mb.app_data["net"].load.bus} | {"Bus_1"})
            rows = [f"2001-01-01 00:{m:02d}:00,{n},10,{20 if n == 'Bus_1' else 0},,{(5 + m / 15) if n == 'Bus_1' else ''}"
                    for m in (0, 15, 30) for n in names]
            out = c.post("/api/data/upload", data={"kind": "measurements", "overwrite": "true"},
                         files={"file": ("q.csv", io.BytesIO(_csv(rows)), "text/csv")}).json()
            held = next(f for f in out["quality"]["findings"] if f["code"] == "reactive_at_voltage_units")
            assert "Bus_1" in held["substations"]
            gc = c.get("/api/grid_constants").json()["measurements"]
            assert "Bus_1" in gc["voltage_units_not_following_measured_q"]
            r = c.post("/api/generators/follow_measured", json={"substations": held["substations"]}).json()
            assert r["switched"] == ["Gen_bus1"] and r["changed"]["Gen_bus1"]["control"] == "fixed_q"
            assert r["changed"]["Gen_bus1"]["q_mvar"] == pytest.approx(6.0)        # the median
            snap = c.post("/api/network/snapshot", json={"timestamp": "2001-01-01 00:30"}).json()
            g1 = next(g for g in snap["generators"] if g["name"] == "Gen_bus1")
            assert g1["Qg_mvar"] == pytest.approx(7.0)                               # the measurement
            q = c.get("/api/data/quality").json()["measurements"]
            assert not any(f["code"] == "reactive_at_voltage_units" for f in q["findings"])
            bad = c.post("/api/generators/follow_measured", json={"substations": ["Bus_2"]})
            assert bad.status_code == 400 and "condenser" in bad.json()["detail"]
    finally:
        mb.app_data.clear()
        mb.app_data.update(saved)
