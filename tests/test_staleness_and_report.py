"""
Answers computed under other settings (agent/staleness.py), and the session
report (agent/report.py). Pure modules: no backend, no Streamlit.
"""

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "llm_agent"))

from agent import report, staleness  # noqa: E402
from agent.trace import FigureTrace, ToolStep, Trace  # noqa: E402


def _result(pf=0.95, source="default", model=None):
    out = {"load_power_factor": {"value": pf, "source": source}}
    if model:
        out["generator_model"] = {"changed_units": model}
    return out


ASSUMPTIONS = {"load_power_factor": {"value": 0.95}, "generator_model": {}, "data_key": "d1"}


class TestStamp:
    def test_an_answer_without_studies_has_no_stamp(self):
        assert staleness.stamp([], ASSUMPTIONS) is None
        assert staleness.stamp([("run_rsa", {"error": "boom"})], ASSUMPTIONS) is None

    def test_the_stamp_comes_from_the_results(self):
        stamp = staleness.stamp([("run_rsa", _result(0.9, "session setting", {"Gen_bus1": {"control": "voltage"}}))],
                                ASSUMPTIONS)
        assert stamp == {"pf": 0.9, "gm": {"Gen_bus1": {"control": "voltage"}}, "data_key": "d1", "names": None}

    def test_a_study_with_its_own_power_factor_is_not_compared_on_it(self):
        stamp = staleness.stamp([("run_rsa", _result(0.8, "this study"))], ASSUMPTIONS)
        assert stamp["pf"] is None and stamp["gm"] == {}

    def test_a_result_that_used_no_tick_leaves_the_model_unknown(self):
        stamp = staleness.stamp([("get_grid_info", {"name": "x"})], ASSUMPTIONS)
        assert stamp == {"pf": None, "gm": None, "data_key": "d1", "names": None}


class TestRenames:
    """A rename changes no figure: a note, not a reason to rerun."""
    NAMED = {**ASSUMPTIONS, "names_key": "n1"}

    def test_a_rename_is_a_note_not_a_reason(self):
        then = staleness.stamp([("run_rsa", _result())], self.NAMED)
        now = staleness.current({**self.NAMED, "names_key": "n2"})
        assert staleness.reasons(then, now) == []
        assert staleness.notes(then, now) == ["elements renamed since — this answer shows the names it ran with"]
        assert staleness.notes(then, staleness.current(self.NAMED)) == []

    def test_a_renamed_unit_is_not_a_model_change(self):
        result = {"load_power_factor": {"value": 0.95, "source": "default"},
                  "generator_model": {"changed_units": {"Plant A": {"dispatchable": False}},
                                      "changed_units_as_loaded": {"Gen_bus1": {"dispatchable": False}}}}
        then = staleness.stamp([("run_rsa", result)], self.NAMED)
        now = staleness.current({**self.NAMED, "generator_model": {"Gen_bus1": {"dispatchable": False}}})
        assert staleness.reasons(then, now) == []

    def test_unknown_names_are_never_a_note(self):
        assert staleness.notes({"names": None}, {"names": "n2"}) == []


class TestReasons:
    def test_nothing_changed(self):
        then = staleness.stamp([("run_rsa", _result())], ASSUMPTIONS)
        assert staleness.reasons(then, staleness.current(ASSUMPTIONS)) == []

    def test_each_change_is_named(self):
        then = staleness.stamp([("run_rsa", _result())], ASSUMPTIONS)
        now = staleness.current({"load_power_factor": {"value": 0.9},
                                 "generator_model": {"Gen_bus1": {"control": "voltage"}}, "data_key": "d2"})
        assert staleness.reasons(then, now) == [
            "load power factor 0.95 → 0.90", "generator model changed (1 unit)", "other data loaded"]

    def test_unknown_is_never_a_change(self):
        assert staleness.reasons(None, staleness.current(ASSUMPTIONS)) == []
        assert staleness.reasons({"pf": 0.9, "gm": None, "data_key": None}, None) == []
        assert staleness.reasons({"pf": None, "gm": None, "data_key": None},
                                 staleness.current({"load_power_factor": {"value": 0.8},
                                                    "generator_model": {"G": {}}, "data_key": "x"})) == []

    def test_the_rerun_asks_for_fresh_studies(self):
        prompt = staleness.rerun_prompt("Is the grid secure?", ["load power factor 0.95 → 0.90"])
        assert prompt.endswith("Is the grid secure?")
        assert "re-run the studies" in prompt and "0.95 → 0.90" in prompt


class TestMarkdown:
    def test_text_is_escaped(self):
        assert "&lt;script&gt;" in report.markdown("<script>alert(1)</script>")
        assert "<script>" not in report.markdown("<script>alert(1)</script>")

    def test_the_subset_answers_use(self):
        out = report.markdown("## Result\n\n**Secure** at *14:45*, see `rsa`.\n\n- one\n- two\n\n"
                              "| Bus | V |\n|---|---|\n| 4 | 1.02 |\n\n1. first\n2. second")
        assert "<h4>Result</h4>" in out
        assert "<strong>Secure</strong>" in out and "<em>14:45</em>" in out and "<code>rsa</code>" in out
        assert "<ul><li>one</li><li>two</li></ul>" in out
        assert "<th>Bus</th>" in out and "<td>1.02</td>" in out
        assert "<ol><li>first</li><li>second</li></ol>" in out

    def test_code_spans_are_left_alone(self):
        assert "<code>a**b**c</code>" in report.markdown("`a**b**c`")

    def test_underscores_in_names_are_not_italics(self):
        assert "Gen_bus1" in report.markdown("Gen_bus1 and load_power_factor")

    def test_a_quote_in_a_link_cannot_leave_the_attribute(self):
        out = report.markdown('[guide](https://example.org/a"onmouseover="alert(1))')
        assert 'href="https://example.org/a&quot;onmouseover=&quot;alert(1"' in out
        assert '"onmouseover="' not in out

    def test_only_web_and_relative_links_become_links(self):
        assert '<a href="https://example.org/?a=1&amp;b=2">x</a>' in report.markdown(
            "[x](https://example.org/?a=1&b=2)")
        assert '<a href="charts/n1.html">screen</a>' in report.markdown("[screen](charts/n1.html)")
        out = report.markdown("[click](javascript:alert)")
        assert "<a" not in out and "click" in out


class TestReport:
    def _trace(self):
        return Trace(steps=(ToolStep(1, "run_rsa", {"timestamp": "now"}, {}, True, None, []),),
                     figures=(FigureTrace(1.02, "pu", "grounded", "run_rsa", "", ""),),
                     concerns=("One figure could not be traced.",), limits=("Provenance is not truth.",))

    def test_the_report_has_everything(self):
        messages = [{"role": "user", "text": "Is it secure?"},
                    {"role": "assistant", "text": "Yes — **secure**.", "charts": [("run_rsa", {})],
                     "trace": self._trace(), "context": {"pf": 0.9}}]
        meta = {"title": "CONDUCTOR session — IEEE 14", "scenario": "Demo: IEEE 14 · active only",
                "network": "IEEE 14", "generated": "2026-10-13 21:00",
                "assumptions": {"load_power_factor": {"value": 0.95, "source": "default"},
                                "generator_model": {"Gen_bus1": {"control": "voltage"}},
                                "reactive_measured": {"measurements": {"consumption": True}}},
                "quality": "measured reactive demand · ok",
                "proposals": [{"id": "pv9", "summary": {"edits": ["add 30 MW PV at Bus 9"]}}]}
        page = report.build(messages, meta, charts_html=lambda results: "<div id='chart'></div>",
                            plotly_js="/*plotly*/", stale=lambda m: ["load power factor 0.90 → 0.95"])
        assert page.startswith("<!doctype html>")
        for text in ("Demo: IEEE 14 · active only", "Is it secure?", "<strong>secure</strong>",
                     "<div id='chart'></div>", "/*plotly*/", "add 30 MW PV at Bus 9", "Gen_bus1",
                     "measured reactive demand · ok", "Computed under other settings", "One figure could not be traced.",
                     "Provenance is not truth.", "run_rsa", "Measured reactive power"):
            assert text in page, text

    def test_an_empty_session_still_builds(self):
        page = report.build([], {})
        assert "No questions yet." in page and "Not available" in page and "None —" in page

    def test_the_header_describes_and_does_not_certify(self):
        """"Every figure from a solver run" is what the evidence below measures,
        not a fact it can state: some figures are derived or untraced."""
        page = report.build([], {})
        assert "every figure from a solver run" not in page.lower()
        assert "where its figures came from" in page


def test_a_settings_tool_is_not_a_study():
    """An answer that only set the power factor ran no study under any setting."""
    setting = ("set_load_power_factor", {"load_power_factor": {"value": 0.9, "source": "session setting"}})
    assert staleness.stamp([setting], ASSUMPTIONS) is None
    study = ("run_rsa", _result(0.9, "session setting"))
    assert staleness.stamp([setting, study], ASSUMPTIONS)["pf"] == 0.9
