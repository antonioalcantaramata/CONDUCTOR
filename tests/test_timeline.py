"""
The operating point: measurements, forecasts, or both — and both at the same
timestamp (agent/timeline.py, the sidebar card, and what the agent is told).

The old clock built one timeline from the union of both datasets and resolved
a timestamp present in both to the measurement, so a forecast at that time
could not be selected; the system prompt said the two never overlap.
"""

import pytest

from llm_agent.agent import timeline as tl
from llm_agent.agent.system_prompt import build_system_prompt


def _ticks(day: int, hours: range, minute: int = 0) -> list[str]:
    return [f"2000-01-{day:02d} {h:02d}:{minute:02d}:00" for h in hours]


MEAS = _ticks(1, range(0, 12))          # 00:00 … 11:00, hourly
FC = _ticks(1, range(8, 24))            # 08:00 … 23:00, overlaps 08–11
SHARED = _ticks(1, range(8, 12))


def _payload(meas=MEAS, fc=FC, clock=None):
    return {"timestamps": meas, "forecast_timestamps": fc,
            "current_timestamp": clock or (meas[0] if meas else None),
            "measurements_source": "synthetic", "forecast": {"source": "uploaded"}}


class TestDatasets:
    def test_both_with_overlap_offer_compare(self):
        timeline = tl.Timeline.from_payload(_payload())
        assert timeline.modes() == [tl.MEASURED, tl.FORECAST, tl.COMPARE]
        assert list(timeline.ticks(tl.COMPARE)) == SHARED

    def test_a_shared_timestamp_is_reachable_as_a_forecast(self):
        timeline = tl.Timeline.from_payload(_payload())
        assert "2000-01-01 09:00:00" in timeline.ticks(tl.FORECAST)
        assert tl.agent_note(tl.FORECAST, "2000-01-01 09:00:00") is not None

    def test_no_overlap_no_compare(self):
        timeline = tl.Timeline.from_payload(_payload(fc=_ticks(2, range(0, 5))))
        assert timeline.modes() == [tl.MEASURED, tl.FORECAST]

    def test_measurements_only(self):
        assert tl.Timeline.from_payload(_payload(fc=[])).modes() == [tl.MEASURED]

    def test_forecasts_only(self):
        assert tl.Timeline.from_payload(_payload(meas=[])).modes() == [tl.FORECAST]

    def test_backend_down(self):
        assert tl.Timeline.from_payload({}).modes() == []


class TestMovingInTime:
    def test_nearest_keeps_an_exact_tick(self):
        assert tl.nearest(MEAS, "2000-01-01 05:00:00") == "2000-01-01 05:00:00"

    def test_nearest_snaps_across_a_dataset_switch(self):
        # 23:00 is a forecast tick; the measured view snaps to its last tick.
        assert tl.nearest(MEAS, "2000-01-01 23:00:00") == "2000-01-01 11:00:00"
        assert tl.nearest(tuple(FC), "2000-01-01 02:00:00") == "2000-01-01 08:00:00"

    def test_nearest_between_ticks(self):
        quarter = _ticks(1, range(0, 3)) + ["2000-01-01 02:15:00"]
        assert tl.nearest(sorted(quarter), "2000-01-01 02:10:00") == "2000-01-01 02:15:00"

    def test_step_is_clamped(self):
        assert tl.step(MEAS, MEAS[0], -1) == MEAS[0]
        assert tl.step(MEAS, MEAS[-1], 1) == MEAS[-1]
        assert tl.step(MEAS, MEAS[3], 1) == MEAS[4]

    @pytest.mark.parametrize("ticks, label", [
        (MEAS, "1 h"),
        ([f"2000-01-01 00:{m:02d}:00" for m in (0, 15, 30, 45)], "15 min"),
        ([f"2000-01-{d:02d} 00:00:00" for d in (1, 2, 3)], "1 day"),
        (MEAS[:1], None),
    ])
    def test_resolution(self, ticks, label):
        assert tl.resolution(ticks) == label


class TestCoverage:
    def test_rows_span_the_whole_window(self):
        cov = tl.coverage(tl.Timeline.from_payload(_payload()), "2000-01-01 23:00:00")
        meas, fc = cov["rows"]
        assert meas["start"] == 0.0 and fc["end"] == 1.0
        assert meas["end"] > fc["start"]  # the overlap is drawn as overlap
        assert cov["cursor"] == 1.0
        assert (meas["source"], fc["source"]) == ("synthetic", "uploaded")

    def test_empty(self):
        assert tl.coverage(tl.Timeline(), None) == {"rows": [], "cursor": None}


class TestWhatTheAgentIsTold:
    def test_nothing_extra_on_measurements(self):
        assert tl.agent_note(tl.MEASURED, MEAS[0]) is None

    def test_forecast_names_dataset_and_time(self):
        note = tl.agent_note(tl.FORECAST, FC[0])
        assert 'data_source="forecasts"' in note and FC[0] in note

    def test_compare_asks_for_both(self):
        note = tl.agent_note(tl.COMPARE, SHARED[0])
        assert 'data_source="measurements"' in note and 'data_source="forecasts"' in note
        assert SHARED[0] in note


class TestSystemPrompt:
    BASE = {"name": "Test grid", "measurements": {"loaded": True, "n_timestamps": 12,
                                                  "first_timestamp": MEAS[0], "last_timestamp": MEAS[-1],
                                                  "source": "synthetic"}}

    def test_overlap_is_stated_when_there_is_one(self):
        prompt = build_system_prompt({**self.BASE, "forecasts": {
            "loaded": True, "n_timestamps": 16, "first_timestamp": FC[0], "last_timestamp": FC[-1],
            "source": "uploaded", "n_shared_with_measurements": 4,
            "first_shared": SHARED[0], "last_shared": SHARED[-1]}})
        assert "share 4 timestamps" in prompt
        assert "forecasts begin after the measurement window" not in prompt
        assert "(uploaded)" in prompt and "(synthetic)" in prompt

    def test_no_overlap_is_stated_too(self):
        prompt = build_system_prompt({**self.BASE, "forecasts": {
            "loaded": True, "n_timestamps": 5, "first_timestamp": "2000-01-02 00:00:00",
            "last_timestamp": "2000-01-02 04:00:00", "n_shared_with_measurements": 0}})
        assert "share no timestamp" in prompt


def test_grid_constants_report_overlap_and_source(backend_client):
    body = backend_client.get("/api/grid_constants").json()
    assert body["measurements"]["source"] in ("synthetic", "uploaded")
    assert body["forecasts"]["n_shared_with_measurements"] >= 0
