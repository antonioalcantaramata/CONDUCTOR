"""
Guard checks and tool wrappers say what the results say (code review batch D,
findings 4.2–4.6).
"""

import inspect

from llm_agent.agent import tools
from llm_agent.agent.characterisation import check_answer as conflicts
from llm_agent.agent.completeness import check_answer as gaps
from llm_agent.agent.tool_schemas import TOOLS
from llm_agent.agent.providers.schema import function_declarations

ATTRIBUTION = {"tool_results": [{"name": "compute_violation_attribution", "result": {
    "violations": [{"element": "Bus_3", "drivers": [
        {"source": "Bravo", "relief_mw": 6.24, "relief_mw_feasible": False}]}]}}]}


class TestCharacterisationReadsTheRightSentence:
    """Claim positions are offsets into the cleaned answer; the sentence used to
    be cut from the raw one, shifted by every `**` and timestamp before it."""

    def test_a_negated_figure_after_markup_is_not_flagged(self):
        answer = ("At **2022-01-02 21:45:00** reduce Bravo by **1 MW**. "
                  "A cut of 6.24 MW is not deliverable.")
        assert conflicts(answer, ATTRIBUTION) == []

    def test_an_asserted_figure_after_markup_is_flagged(self):
        answer = ("At **2022-01-02 21:45:00**, the fix is clear. "
                  "**Reduce Bravo by 6.24 MW** to clear it.")
        assert len(conflicts(answer, ATTRIBUTION)) == 1


class TestCompletenessReadsWhen:
    def test_a_conditional_when_does_not_ask_for_a_time(self):
        assert gaps("What happens to voltages when wind drops 30%?", "They stay within limits.") == []

    def test_a_question_when_does(self):
        assert [g.kind for g in gaps("When is the grid most stressed?", "Often.")] == ["timestamp"]


class TestCompareResults:
    RSA = {"_source": "rsa", "timestamp": "t"}

    def _opf(self, **pg_new):
        return {"status": "optimal", "activated_resources": [
            {"element": name, "Pg_base": 4.0, "Pg_new": value, "Qg_base": 0.0, "Qg_new": 0.0}
            for name, value in pg_new.items()]}

    def test_curtailment_to_zero_is_a_decrease(self):
        """`Pg_new or Pg_base` read 0.0 as missing; the sign even flipped."""
        diff = tools.compare_results(result_a=self._opf(Echo=4.0), result_b=self._opf(Echo=0.0))
        (row,) = diff["dispatch_diff"]
        assert (row["Pg_new_a"], row["Pg_new_b"], row["delta_Pg"]) == (4.0, 0.0, -4.0)

    def test_a_unit_missing_from_one_side_is_at_its_base(self):
        a = self._opf(Echo=5.0, Foxtrot=4.0)
        b = self._opf(Echo=3.0)  # contingency results list only units that moved
        rows = {r["name"]: r for r in tools.compare_results(result_a=a, result_b=b)["dispatch_diff"]}
        assert rows["Foxtrot"]["delta_Pg"] == 0.0

    def test_an_infeasible_result_is_refused(self):
        infeasible = {"status": "infeasible", "feasible": False, "activated_resources": []}
        out = tools.compare_results(result_a=self.RSA, result_b=infeasible)
        assert "error" in out and "infeasible" in out["error"]

    def test_auto_pairing_skips_failed_results(self, monkeypatch):
        good_a, good_b = self._opf(Echo=4.0), self._opf(Echo=1.0)
        monkeypatch.setattr(tools, "_last_tool_results", [
            ("optimize_flexibility", good_a),
            ("optimize_flexibility", good_b),
            ("optimize_flexibility", {"status": "infeasible", "feasible": False,
                                      "activated_resources": []}),
        ])
        (row,) = tools.compare_results()["dispatch_diff"]
        assert row["delta_Pg"] == -3.0  # 4 → 1, not 1 → the infeasible result


class TestContingencySlackCap:
    def test_it_is_not_offered(self):
        decl = next(fd for fd in function_declarations(TOOLS) if fd.name == "optimize_contingency")
        assert "slack_max_mw" not in decl.parameters.properties

    def test_it_is_refused_not_silently_dropped(self, monkeypatch):
        sent = []
        monkeypatch.setattr(tools, "_post", lambda e, b: sent.append(b) or {})
        out = tools.optimize_contingency("line", 0, slack_max_mw=20)
        assert out.get("refusal_required") is True and not sent


class TestWrapperDefaultsMatchTheBackend:
    def test_envelope_resolution(self):
        assert inspect.signature(tools.compute_flexibility_envelope).parameters["resolution"].default == 20

    def test_near_miss_band(self):
        assert inspect.signature(tools.compute_historical_risk).parameters["near_miss_band"].default == 0.005
