"""
The tailored session report (`llm_agent/agent/tailored_report.py`): written by
the model from the session's record, with the session's own charts placed by
reference and every figure checked against the results.
"""

from llm_agent.agent import tailored_report as tr
from llm_agent.agent.providers.base import ModelResponse


def _rec(user, results):
    return {"user": user, "turn_key": "k", "grid": {},
            "tool_calls": [{"name": n, "args": {"timestamp": "2000-01-06 20:00"}} for n, _ in results],
            "tool_results": [{"name": n, "result": r} for n, r in results]}


MESSAGES = [
    {"role": "user", "text": "Is the grid secure at 20:00?"},
    {"role": "assistant", "text": "Two violations: the feeder line at 114.7 %.", "charts": [("run_rsa", {})],
     "record": _rec("Is the grid secure at 20:00?", [("run_rsa", {"secure": False, "total_violations": 2,
                                                                  "max_line_loading_pct": 114.7})])},
    {"role": "user", "text": "Thanks"},
    {"role": "assistant", "text": "You're welcome."},                     # no record: not a turn
    {"role": "user", "text": "Give me options."},
    {"role": "assistant", "text": "Four options; the cheapest costs 26.48 €/h.", "charts": None,
     "record": _rec("Give me options.", [("optimize_flexibility", {"mode": "options", "status": "options",
                                                                   "cost_eur_per_h": 26.48})])},
]
META = {"network": "IEEE 30", "scenario": "Demo", "data": "measurements", "model": "gpt-test",
        "settings": ["Gen_bus1 costs: reactive 0 €/Mvarh"], "title": "CONDUCTOR session report — IEEE 30"}


class FakeProvider:
    name, model = "fake", "fake-model"

    def __init__(self, text="", raises=None):
        self.text, self.raises, self.calls = text, raises, []

    def chat(self, messages, tools, system, on_event=None):
        self.calls.append({"messages": messages, "tools": tools, "system": system})
        if self.raises:
            raise self.raises
        return ModelResponse(text=self.text)


def test_turns_are_numbered_as_the_advisor_numbers_them():
    assert [n for n, _, _ in tr.turn_records(MESSAGES)] == [1, 2]
    text = tr.request_text(MESSAGES, META, notes=[{"decision": "Rule out Gen_bus1.", "criteria": ["trust"]}])
    assert "## T1\nQuestion: Is the grid secure at 20:00?" in text and "## T2" in text and "## T3" not in text
    assert "Charts: yes" in text and "Charts: no" in text
    assert '- run_rsa arguments: {"timestamp": "2000-01-06 20:00"}' in text            # every argument
    assert "Setting: Gen_bus1 costs: reactive 0 €/Mvarh" in text and "Decision: Rule out Gen_bus1." in text


def test_a_long_session_keeps_its_newest_turns():
    long = [m for i in range(60) for m in (
        {"role": "user", "text": "q"},
        {"role": "assistant", "text": "a" * 3500, "record": _rec(f"Question {i}", [("run_rsa", {})])})]
    text = tr.request_text(long, META)
    assert len(text) <= tr._MAX_INPUT_CHARS + 500 and "earlier turns omitted" in text and "Question 59" in text


def test_one_tool_less_call_with_the_rules():
    provider = FakeProvider("```markdown\n## Summary\nTwo violations (T1).\n```")
    body, error = tr.write("record", provider=provider)
    assert body == "## Summary\nTwo violations (T1)." and not error
    call = provider.calls[0]
    assert call["tools"] == [] and "cites the turns" in call["system"] and "{{chart T3}}" in call["system"]
    assert tr.write("record", provider=FakeProvider(raises=RuntimeError("quota")))[1].endswith("quota")


def test_the_session_charts_are_placed_and_nothing_else():
    body = ("## Findings\nThe feeder line overloads (T1).\n\n{{chart T1}}\n\n{{chart T1}}\n\n{{chart T2}}\n\n"
            "The cheapest option costs 26.48 €/h (T2).")
    page = tr.render(body, META, charts_by_turn={1: "<div id='chart-t1'></div>"},
                     questions=[(1, "Is the grid secure at 20:00?"), (2, "Give me options.")])
    assert page.count("id='chart-t1'") == 1 and "Chart from T1, as the session drew it." in page
    assert "(No chart for T2.)" in page and "{{chart" not in page
    assert "Written by gpt-test from the session's record" in page and "raw report is the complete record" in page
    assert "<b>T2</b> Give me options." in page
    assert "<h2" in page and "Findings</h2>" in page                               # sections, not small print


def test_figures_not_in_the_results_are_flagged_on_the_page():
    records = [r for _, _, r in tr.turn_records(MESSAGES)]
    assert tr.unverified_figures("The feeder line at 114.7 % (T1); the cheapest 26.48 €/h (T2).", records) == ()
    assert tr.unverified_figures("A loss of 3.9 MW (T1).", records) == ("3.9 MW",)
    page = tr.render("## Summary\nA loss of 3.9 MW.", META, charts_by_turn={}, unverified=("3.9 MW",))
    assert "Figures in this report that the session's results do not contain: 3.9 MW" in page
