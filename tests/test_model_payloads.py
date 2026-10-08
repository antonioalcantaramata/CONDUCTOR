"""What the model is sent stays bounded (code review 7.1).

The full result still feeds the charts and the guard layer; only the copy sent
to the model drops per-tick series, which a week-long scan made ~9.5k tokens
and a year-long one would make larger than the context.
"""

from llm_agent.agent.loop import _prepare_tool_result_for_model as prep

N = 672
STAMPS = [f"2000-01-0{1 + i // 96} {i % 96:02d}" for i in range(N)]


def test_worst_case_keeps_the_answer_and_drops_the_series():
    full = {"worst_timestamp": "t", "worst_value": 2, "worst_per_metric": {"violations": {}},
            "n_scanned": N, "series": {"timestamps": STAMPS, "min_voltage": [0.95] * N}}
    sent = prep("find_worst_case_timestamp", full)
    assert "series" not in sent and sent["worst_timestamp"] == "t"
    assert "series" in full  # the chart's copy is untouched


def test_element_timeseries_is_summarised_with_its_extremes():
    values = [1.0] * N
    values[100], values[200] = 0.93, 1.07
    sent = prep("get_element_timeseries", {"timestamps": STAMPS, "series": {"vm_pu": values},
                                           "element_name": "Bus_3"})
    summary = sent["series_summary"]["vm_pu"]
    assert (summary["min"], summary["min_at"]) == (0.93, STAMPS[100])
    assert (summary["max"], summary["max_at"]) == (1.07, STAMPS[200])
    assert "series" not in sent and "timestamps" not in sent and sent["n_points"] == N


def test_scenario_ticks_are_capped_with_a_count():
    ticks = [{"timestamp": s, "buses": ["Bus_3"]} for s in STAMPS[:50]]
    sent = prep("scan_scenarios", {"scenarios": [{"label": "x1.0", "violations_per_tick": ticks,
                                                  "series": {"a": [1] * N}}]})
    scenario = sent["scenarios"][0]
    assert len(scenario["violations_per_tick"]) == 20
    assert scenario["violations_per_tick_truncated"] == 50 and "series" not in scenario


def test_errors_pass_through_unchanged():
    err = {"error": "boom", "series": [1]}
    assert prep("find_worst_case_timestamp", err) is err
