import os
import pathlib

import pandas as pd
import pytest

import pipeline_functions as pf

EXAMPLE_MAP = pathlib.Path(pf.__file__).parent / "datastream_map.example.json"


@pytest.fixture(autouse=True)
def synthetic_map(monkeypatch):
    """Pin every test to the example map.

    Without this the tests read whichever `datastream_map.local.json` the
    developer happens to have, so they would pass on one machine and fail on
    another — and the real map is deliberately not in the repository.
    """
    monkeypatch.setenv("CONDUCTOR_DATASTREAM_MAP", str(EXAMPLE_MAP))
    pf._datastream_map.cache_clear()
    monkeypatch.setattr(pf, "metadata", pf._datastream_map()["substations"])
    yield
    pf._datastream_map.cache_clear()


class TestDefineTimespan:
    def test_reformats_dates_to_midnight_isoformat(self):
        start, end = pf.define_timespan("2022-08-01", "2022-08-02")
        assert start == "2022-08-01T00:00:00"
        assert end == "2022-08-02T00:00:00"


class TestConvertDatatypes:
    def test_raises_on_missing_columns(self):
        df = pd.DataFrame({"value": [1.0]})
        with pytest.raises(ValueError, match="Missing required columns"):
            pf.convert_datatypes(df)

    def test_converts_columns_to_expected_types(self):
        df = pd.DataFrame({
            "datastream_id": ["123"],
            "value": ["4.5"],
            "substation": ["BRAVO"],
            "parameter": ["t1s_belastning"],
            "timestamp": ["2022-08-01T00:00:00"],
        })

        result = pf.convert_datatypes(df, verbose=False)

        assert result["datastream_id"].dtype == "int64"
        assert result["value"].dtype == "float64"
        assert pd.api.types.is_datetime64_any_dtype(result["timestamp"])

    def test_wraps_conversion_errors_in_value_error(self):
        df = pd.DataFrame({
            "datastream_id": ["not_a_number"],
            "value": ["4.5"],
            "substation": ["BRAVO"],
            "parameter": ["t1s_belastning"],
            "timestamp": ["2022-08-01T00:00:00"],
        })
        with pytest.raises(ValueError, match="Error converting data types"):
            pf.convert_datatypes(df)


class TestShiftTimespan:
    def test_empty_data_returns_empty_list(self):
        assert pf.shift_timespan([], "2023-01-01T00:00:00") == []

    def test_shifts_all_timestamps_by_same_delta(self):
        data = [
            {"timestamp": "2022-08-01T00:00:00", "value": 1},
            {"timestamp": "2022-08-01T01:00:00", "value": 2},
        ]

        result = pf.shift_timespan(data, "2023-01-01T00:00:00")

        assert result[0]["timestamp"] == "2023-01-01T00:00:00"
        assert result[1]["timestamp"] == "2023-01-01T01:00:00"


class TestGetIdsBySubstation:
    def test_returns_ids_for_named_substation(self):
        ids = pf.get_ids_by_substation("BRAVO")
        assert ids == list(pf.metadata["BRAVO"].values())

    def test_star_returns_ids_from_all_substations(self):
        ids = pf.get_ids_by_substation("*")
        expected_total = sum(len(v) for v in pf.metadata.values())
        assert len(ids) == expected_total

    def test_unknown_substation_returns_empty_list(self):
        assert pf.get_ids_by_substation("NOT_A_REAL_SUBSTATION") == []


class TestGetIdBySubstationAndParameter:
    def test_returns_matching_id(self):
        # get_datastream_metadata() uses title-cased substation names
        # ("Bravo"), unlike the module-level `metadata` dict which uses
        # all-caps keys ("BRAVO") — the two shapes really do differ.
        expected = pf.metadata["BRAVO"]["t1s_belastning"]
        assert pf.get_id_by_substation_and_parameter("Bravo", "t1s_belastning") == expected

    def test_returns_none_when_not_found(self):
        assert pf.get_id_by_substation_and_parameter("Bravo", "not_a_parameter") is None
