"""Charts never take the conversation down (code review batch B).

Renderers crashed on None values — the backend reports an out-of-service
element's loading as None — and an empty renderer result reached
`st.columns(0)`. Because a message is stored before its charts are drawn, either
failure repeated on every rerun and left the chat unusable until reset.

Plotly and Streamlit are UI packages outside the CI requirements, so these run
wherever the full environment is installed and skip elsewhere.
"""

import sys
import pathlib

import pytest

pytest.importorskip("plotly")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "llm_agent"))

from agent import renderers  # noqa: E402
from agent.renderers import RENDERER_MAP  # noqa: E402

RSA = {
    "timestamp": "2000-01-01 00:00:00",
    "all_voltages": [{"bus_name": "Bus_0", "vm_pu": 1.0}, {"bus_name": "Bus_1", "vm_pu": None}],
    "all_line_loading": [{"line_name": "L0", "loading_percent": 50.0},
                         {"line_name": "L1", "loading_percent": None}],
    "all_trafo_loading": [{"trafo_name": "T0", "loading_percent": None}],
    "violations": [], "total_violations": 0, "secure": True,
    "thresholds_used": {"vm_upper_pu": 1.06, "vm_lower_pu": 0.94,
                        "max_line_loading_pct": 100.0, "max_trafo_loading_pct": 100.0},
}


def test_over_is_none_safe():
    assert renderers._over(1.1, 1.0) and not renderers._over(None, 1.0)
    assert not renderers._over(1.1, None) and not renderers._over(0.9, 1.0)


def test_rsa_with_unsolved_elements_renders():
    figs = RENDERER_MAP["run_rsa"](RSA)
    assert figs


def test_kpis_without_values_render():
    RENDERER_MAP["evaluate_kpis"]({"metrics": {
        "kpi_1_target_demand_flex_pct": None, "kpi_2_flex_utilization_pct": None,
        "kpi_3_prevented_violation_ratio_pct": None}, "timestamp": "t"})


def test_conditions_with_unknown_output_render():
    RENDERER_MAP["get_current_conditions"]({
        "timestamp": "t", "generators": [{"name": "G", "Pg_mw": None, "Pg_max_mw": 10}],
        "loads": []})


def test_attribution_with_nothing_to_attribute_returns_no_figures():
    assert not RENDERER_MAP["compute_violation_attribution"]({"violations": []})


def test_render_charts_survives_empty_and_failing_renderers(monkeypatch):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    def script():
        import sys
        sys.path.insert(0, "llm_agent")
        import streamlit as st
        from agent import renderers

        def broken(result):
            raise ValueError("boom")

        renderers.RENDERER_MAP["run_rsa"] = broken
        # A trimmed copy of app.render_charts' contract: the real function is
        # exercised through the app tests; here only its guards matter.
        src = open("llm_agent/app.py", encoding="utf-8").read()
        start = src.index("_CHARTS_PER_ROW = ")
        end = src.index("# Chart injection helper")
        namespace = {"st": st, "RENDERER_MAP": renderers.RENDERER_MAP,
                     "logging": __import__("logging"), "_TOOL_LABELS": {},
                     "re": __import__("re"), "pd": __import__("pandas")}
        exec(src[start:end].rsplit("# ----", 1)[0], namespace)
        namespace["render_charts"]([
            ("compute_violation_attribution", {"violations": []}),
            ("run_rsa", {"timestamp": "t"}),
            # A status figure becomes a message, not an empty canvas.
            ("simulate_contingency", {"timestamp": "2000-01-01 00:00:00",
                                      "violations": [], "system_secure": True}),
        ])
        st.write("still here")

    at = AppTest.from_function(script, default_timeout=30)
    at.run()
    assert not at.exception
    assert any("still here" in m.value for m in at.markdown)
    assert any("could not be drawn" in c.value for c in at.caption)
    assert any("No violations" in m.value for m in at.success)
    assert any("2000-01-01 00:00" in c.value for c in at.caption)


class TestN1Table:
    LIMITS = {"vm_lower_pu": 0.94, "vm_upper_pu": 1.06,
              "max_line_loading_pct": 100.0, "max_trafo_loading_pct": 100.0}

    def _sweep(self, violations, **extra):
        return {"total_outages_tested": 20, "total_outages_causing_violations": 2,
                "total_violations": len(violations), "system_n1_secure": False,
                "violations": violations, "thresholds_used": self.LIMITS, **extra}

    def test_rows_name_the_outage_and_sort_worst_first(self):
        fig = RENDERER_MAP["simulate_all_contingencies"](self._sweep([
            {"violation_type": "bus_vm_pu", "value": 0.93, "element_name": "Bus_3", "outage_cause": "L1"},
            {"violation_type": "line_loading", "value": 125.0, "element_name": "L5", "outage_cause": "L2"},
        ]))
        rows = list(zip(*fig.data[0].cells.values))
        assert rows[0][:3] == ("L2", "L5", "Line overload")
        assert rows[1][:3] == ("L1", "Bus_3", "Undervoltage")

    def test_a_rounded_value_on_the_floor_is_still_undervoltage(self):
        """0.9396 p.u. arrives as 0.940 — exactly the floor."""
        fig = RENDERER_MAP["simulate_all_contingencies"](self._sweep([
            {"violation_type": "bus_vm_pu", "value": 0.94, "element_name": "Bus_13", "outage_cause": "L2"}]))
        assert fig.data[0].cells.values[2][0] == "Undervoltage"

    def test_every_violation_is_listed(self):
        """The table scrolls in the app, so the old 30-row cap is gone."""
        many = [{"violation_type": "line_loading", "value": 101.0 + i, "element_name": f"L{i}",
                 "outage_cause": f"L{i + 1}"} for i in range(45)]
        fig = RENDERER_MAP["simulate_all_contingencies"](self._sweep(many))
        assert len(fig.data[0].cells.values[0]) == 45

    def test_the_app_shows_it_as_a_copyable_table(self):
        """A Plotly table is a picture; the app draws it with st.dataframe."""
        pytest.importorskip("streamlit")
        from streamlit.testing.v1 import AppTest

        def script():
            import sys
            sys.path.insert(0, "llm_agent")
            import streamlit as st
            from agent import renderers
            src = open("llm_agent/app.py", encoding="utf-8").read()
            start = src.index("_CHARTS_PER_ROW = ")
            end = src.index("# Chart injection helper")
            namespace = {"st": st, "RENDERER_MAP": renderers.RENDERER_MAP,
                         "logging": __import__("logging"), "_TOOL_LABELS": {},
                         "re": __import__("re"), "pd": __import__("pandas")}
            exec(src[start:end].rsplit("# ----", 1)[0], namespace)
            namespace["render_charts"]([("simulate_all_contingencies", {
                "timestamp": "2000-01-03 02:15:00", "total_outages_tested": 20,
                "total_outages_causing_violations": 1, "system_n1_secure": False,
                "thresholds_used": {"vm_lower_pu": 0.94, "vm_upper_pu": 1.06,
                                    "max_line_loading_pct": 100.0, "max_trafo_loading_pct": 100.0},
                "violations": [{"violation_type": "line_loading", "value": 109.8,
                                "element_name": "L5", "outage_cause": "L2"}]})])

        at = AppTest.from_function(script, default_timeout=30)
        at.run()
        assert not at.exception
        (table,) = at.dataframe
        assert list(table.value.columns) == ["Outage", "Violated element", "Problem", "Value", "Limit"]
        assert table.value.iloc[0].tolist() == ["L2", "L5", "Line overload", "109.8 %", "100 %"]
        assert any("N-1 screen" in m.value for m in at.markdown)

    def test_non_converged_outages_are_stated_first(self):
        figs = RENDERER_MAP["simulate_all_contingencies"](self._sweep(
            [{"violation_type": "line_loading", "value": 101.0, "element_name": "L5", "outage_cause": "L2"}],
            non_converged_outages=["Bus_0 -> Bus_1 [L0]"]))
        assert "no converged power flow" in figs[0].layout.meta["conductor_status"]

    def test_a_collapsed_single_outage_is_not_secure(self):
        fig = RENDERER_MAP["simulate_contingency"]({"converged": False, "system_secure": False,
                                                    "violations": []})
        assert fig.layout.meta["tone"] == "error"


def test_any_failed_dispatch_is_shown_as_failed_with_its_reason():
    """Only status "infeasible" was recognised; other failures read as
    "No dispatch data — run the optimizer first"."""
    fig = RENDERER_MAP["optimize_flexibility"]({
        "status": "cap_infeasible", "feasible": False, "activated_resources": [],
        "message": "No dispatch keeps the external-grid exchange within ±10 MW."})[0]
    assert fig.layout.meta["tone"] == "error" and "±10 MW" in fig.layout.meta["conductor_status"]


def test_scans_with_skipped_ticks_carry_a_warning():
    src = (pathlib.Path(__file__).resolve().parents[1] / "llm_agent" / "app.py").read_text()
    ns: dict = {}
    exec(src[src.index("def _non_converged_note("):src.index("def _chart_caption(")], ns)
    note = ns["_non_converged_note"]
    assert "2 tick(s)" in note({"n_non_converged": 2, "non_converged_timestamps": ["a", "b"]})
    assert "1 tick(s)" in note({"scenarios": [{"n_non_converged": 1}]})
    assert note({"n_non_converged": 0}) is None
