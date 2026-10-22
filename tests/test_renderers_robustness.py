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
from agent import tool_registry  # noqa: E402
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
        from agent import renderers, tool_registry

        def broken(result):
            raise ValueError("boom")

        renderers.RENDERER_MAP["run_rsa"] = broken
        # A trimmed copy of app.render_charts' contract: the real function is
        # exercised through the app tests; here only its guards matter.
        src = open("llm_agent/app.py", encoding="utf-8").read()
        start = src.index("_CHARTS_PER_ROW = ")
        end = src.index("# Chart injection helper")
        namespace = {"st": st, "RENDERER_MAP": renderers.RENDERER_MAP,
                     "logging": __import__("logging"), "_tools_reg": tool_registry,
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
            from agent import renderers, tool_registry
            src = open("llm_agent/app.py", encoding="utf-8").read()
            start = src.index("_CHARTS_PER_ROW = ")
            end = src.index("# Chart injection helper")
            namespace = {"st": st, "RENDERER_MAP": renderers.RENDERER_MAP,
                         "logging": __import__("logging"), "_tools_reg": tool_registry,
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


def _caption_functions():
    src = (pathlib.Path(__file__).resolve().parents[1] / "llm_agent" / "app.py").read_text()
    ns: dict = {"_tools_reg": tool_registry}
    exec(src[src.index("def _non_default_pf("):src.index("def _table_of(")], ns)
    return ns["_chart_caption"], ns["_non_default_pf"]


def test_charts_from_different_power_factors_say_which():
    caption, non_default = _caption_functions()
    runs = [{"timestamp": "2000-01-01 00:00:00", "load_power_factor": {"value": v, "source": "this study"}}
            for v in (0.9, 0.95, 1.0)]
    show = any(non_default(r) for r in runs)
    assert [caption("run_rsa", r, show_pf=show) for r in runs] == [
        f"**Security assessment (RSA)** · 2000-01-01 00:00 · loads at pf {v}" for v in ("0.90", "0.95", "1.00")]


def test_the_default_power_factor_is_not_repeated_on_every_chart():
    caption, _ = _caption_functions()
    result = {"timestamp": "2000-01-01 00:00:00", "load_power_factor": {"value": 0.95, "source": "default"}}
    assert caption("run_rsa", result) == "**Security assessment (RSA)** · 2000-01-01 00:00"


def test_measured_reactive_demand_and_other_conditions_are_named():
    caption, _ = _caption_functions()
    measured = {"timestamp": "2000-01-01 00:00:00", "load_power_factor": {
        "value": 0.9, "source": "this study", "measured_reactive_power": {"share_of_load": 1.0}}}
    assert caption("run_rsa", measured).endswith("reactive demand as measured")
    other = {"data_source": "nominal", "network": {"id": "pv9"}, "generator_model": {"changed_units": {}}}
    assert caption("run_rsa", other) == ("**Security assessment (RSA)** · nominal operating point · proposal `pv9` · "
                                         "generator model changed")


def _arg_functions():
    import json
    src = (pathlib.Path(__file__).resolve().parents[1] / "llm_agent" / "app.py").read_text()
    ns: dict = {"json": json, "_tools_reg": tool_registry}
    exec(src[src.index("_ARG_SHOWN_ELSEWHERE = "):src.index("def _table_of(")], ns)
    return ns


def _run(**args):
    return {"timestamp": "2000-01-01 00:00:00", "_call_args": args}


def test_any_parameter_that_differs_between_runs_is_on_the_caption():
    ns = _arg_functions()
    runs = [("run_rsa", _run(load_scaling_factor=1.0, timestamp="2000-01-01 00")),
            ("run_rsa", _run(load_scaling_factor=1.2, timestamp="2000-01-01 00"))]
    varying = ns["_varying_args"](runs)
    assert varying == {"run_rsa": ["load_scaling_factor"]}           # the shared timestamp is not repeated
    captions = [ns["_chart_caption"](n, r, extra=ns["_arg_bits"](n, r, varying)) for n, r in runs]
    assert captions == ["**Security assessment (RSA)** · 2000-01-01 00:00 · load ×1",
                        "**Security assessment (RSA)** · 2000-01-01 00:00 · load ×1.2"]


def test_readable_labels_for_common_parameters():
    ns = _arg_functions()
    bit = ns["_arg_bit"]
    assert bit("vm_lower_pu", 0.95, {"vm_lower_pu": 0.95, "vm_upper_pu": 1.05}) == "V limits 0.95–1.05"
    assert bit("element_index", 7, {"element_type": "line", "element_index": 7}) == "outage line 7"
    assert bit("n_samples", 500, {}) == "500 samples"
    assert bit("max_line_loading_pct", 80, {}) == "line limit 80 %"
    assert bit("strategy", "unity", {}) == "strategy unity"                 # anything else, by name
    assert bit("load_scaling_factor", None, {}) == "load scaling factor default"


def test_a_single_run_or_identical_runs_add_nothing():
    ns = _arg_functions()
    assert ns["_varying_args"]([("run_rsa", _run(load_scaling_factor=1.2))]) == {}
    assert ns["_varying_args"]([("run_rsa", _run(a=1)), ("run_rsa", _run(a=1))]) == {}
    assert ns["_varying_args"]([("run_rsa", _run(a=1)), ("scan_rsa_over_time", _run(a=2))]) == {}


def test_every_tool_with_charts_gets_the_same_captions():
    ns = _arg_functions()
    runs = [("simulate_contingency", _run(element_type="line", element_index=3)),
            ("simulate_contingency", _run(element_type="line", element_index=7)),
            ("run_probabilistic_rsa", _run(n_samples=200)), ("run_probabilistic_rsa", _run(n_samples=1000))]
    varying = ns["_varying_args"](runs)
    assert varying == {"simulate_contingency": ["element_index"], "run_probabilistic_rsa": ["n_samples"]}
    bits = [ns["_arg_bits"](n, r, varying) for n, r in runs]
    assert bits == [["outage line 3"], ["outage line 7"], ["200 samples"], ["1000 samples"]]


class TestSeveralProposals:
    """compare_networks(proposals=[...]): one chart per figure of the ranking
    table, or every network's curve on one scan chart."""

    TABLE = [{"network": "base", "role": "reference", "total_violations": 3, "min_vm_pu": 0.93,
              "max_line_loading_pct": 88.0},
             {"network": "A", "role": "proposal", "total_violations": 1, "min_vm_pu": 0.96,
              "max_line_loading_pct": 70.0}]

    def test_rsa_table(self):
        figs = renderers.render_network_comparison({"several": True, "study": "rsa", "table": self.TABLE,
                                                    "timestamp": "2000-01-01 00:00"})
        assert len(figs) == 3 and "by network" in figs[0].layout.title.text

    def test_scan_curves_and_the_model_copy_drops_them(self):
        from agent.loop import _prepare_tool_result_for_model
        result = {"several": True, "study": "scan", "compared_with": "A", "comparisons": [
            {"network": {"id": "B"}, "series": {"timestamps": ["t1"], "base_min_vm": [1.0],
                                               "proposal_min_vm": [0.9]}}]}
        figs = renderers.render_network_comparison(result)
        assert [t.name for t in figs[0].data] == ["A", "B"]
        assert "series" not in str(_prepare_tool_result_for_model("compare_networks", result))

    def test_empty(self):
        assert len(renderers.render_network_comparison({"several": True, "study": "rsa", "table": []})) == 1

    def test_a_proposal_against_another_is_titled_so(self):
        figs = renderers.render_network_comparison({
            "study": "rsa", "timestamp": "2000-01-01 00:00", "network": {"id": "B", "compared_with": "A"},
            "largest_voltage_changes": [{"element": "Bus_3", "change": 0.01, "base": 1.0, "proposal": 1.01}]})
        assert "B − A" in figs[0].layout.title.text


class TestFlexibilityOptions:
    @staticmethod
    def option(sid, label, dp, dq=0.0, also=()):
        return {"id": sid, "label": label, "how": "…", "also": list(also), "n_units_called": 1,
                "moved_mw": abs(dp), "moved_mvar": abs(dq), "import_mw": 250.0, "cost_eur_per_h": 30.0,
                "above_cheapest_eur_per_h": 0.0, "min_vm_pu": 0.94, "max_loading_pct": 60.0,
                "dispatch": [{"unit": "Gen_bus1", "dp_mw": dp, "dq_mvar": dq},
                             {"unit": "External grid", "dp_mw": -dp, "dq_mvar": -dq}]}

    def test_a_table_and_one_bar_chart_per_quantity_moved(self):
        result = {"mode": "options", "status": "options", "timestamp": "2000-01-03 18:00",
                  "costs": {"prices_from": {"synthetic": 5, "case": 1}},
                  "options": [self.option("least_disturbance", "Least disturbance", 15.0, 5.0, also=["Cheapest"]),
                              self.option("least_import", "Least import", 44.0)]}
        figs = RENDERER_MAP["optimize_flexibility"](result)
        names = figs[0].data[0].cells.values[0]
        assert names[0] == "Least disturbance (also Cheapest)" and names[1] == "Least import"
        assert len(figs) == 3                       # the table, active power, reactive power
        assert "with" not in figs[0].layout.title.text.split("<br>")[1].split("·")[0]
        assert list(figs[0].data[0].cells.values[-1]) == ["—", "—"]           # no check in this result
        assert [t.name for t in figs[1].data] == ["Least disturbance", "Least import"]

    def test_the_operators_conditions_are_in_the_subtitle(self):
        result = {"mode": "options", "status": "options", "timestamp": "2000-01-06 20:00",
                  "held_units": {"Gen_bus1": 48.8}, "options": [self.option("least_disturbance", "Least disturbance", 5.0)]}
        title = RENDERER_MAP["optimize_flexibility"](result)[0].layout.title.text
        assert "Each meets every limit with Gen_bus1 at its schedule" in title

    def test_after_an_outage_the_title_names_it(self):
        result = {"mode": "options", "status": "options", "timestamp": "2000-01-03 12:00",
                  "outage": {"element": "Bus_26 -> Bus_28 [L36]"},
                  "options": [self.option("cheapest", "Cheapest", 5.0)]}
        title = RENDERER_MAP["optimize_contingency"](result)[0].layout.title.text
        assert title.startswith("Corrective options after the outage of Bus_26 -> Bus_28 [L36] at 2000-01-03 12:00")

    def test_the_power_flow_column(self):
        ok = {**self.option("cheapest", "Cheapest", 5.0), "power_flow_check": {"confirmed": True}}
        bad = {**self.option("least_import", "Least import", 6.0),
               "power_flow_check": {"confirmed": False, "violations": ["Bus_0 -> Bus_1 [L0] at 101.3 %"]}}
        figs = RENDERER_MAP["optimize_flexibility"]({"mode": "options", "status": "options",
                                                     "timestamp": "t", "options": [ok, bad]})
        assert list(figs[0].data[0].cells.values[-1]) == ["✓ confirmed", "✗ Bus_0 -> Bus_1 [L0] at 101.3 %"]

    def test_a_near_optimal_set_says_its_margin(self):
        result = {"mode": "options", "status": "options", "timestamp": "2000-01-06 20:00", "set": "near_optimal",
                  "near_optimal_pct": 10, "budget_eur_per_h": 29.13,
                  "options": [self.option("cheapest", "Cheapest", 5.0), self.option("near_optimal_1", "Near-optimal 1", 6.0)]}
        figs = RENDERER_MAP["optimize_flexibility"](result)
        assert figs[0].layout.title.text.startswith("Near-optimal alternatives at 2000-01-06 20:00")
        assert "costs at most 29.13 €/h (10 % above the cheapest)" in figs[0].layout.title.text
        assert [t.marker.color for t in figs[1].data] == [renderers.OPTION_COLOURS["cheapest"],
                                                           renderers.OPTION_COLOURS["near_optimal_1"]]

    def test_branch_caps_are_in_the_subtitle(self):
        result = {"mode": "options", "status": "options", "timestamp": "2000-01-06 20:00",
                  "held_units": {"Gen_bus1": 48.8}, "branch_limits": {"Bus_21 -> Bus_23 [L30]": 80.0},
                  "options": [self.option("least_disturbance", "Least disturbance", 5.0)]}
        title = RENDERER_MAP["optimize_flexibility"](result)[0].layout.title.text
        assert "with Gen_bus1 at its schedule and L30 at most 80 %" in title

    def test_every_strategy_has_its_own_colour(self):
        default = [s for s in renderers.OPTION_COLOURS if not s.startswith(("near_optimal", "n1_"))]
        assert len({renderers.OPTION_COLOURS[s] for s in default}) == len(default) == 6
        n1 = ["least_disturbance", "cheapest", "n1_preventive", "n1_corrective"]
        assert len({renderers.OPTION_COLOURS[s] for s in n1}) == 4

    N1_STUDY = {"modes": ["preventive", "corrective"], "outages_studied": ["Bus_0 -> Bus_1 [L0]", "Bus_9 -> Bus_19 [L24]"],
                "skipped": [], "emergency_loading_pct": 100.0, "post_outage_band_pu": [0.95, 1.05], "reach_pct": 20.0}

    def n1_result(self):
        screen = {"outages_tested": 41, "outages_with_violations": 29, "violations": 57, "not_converged": 0}
        plan = [{"outage": "Bus_0 -> Bus_1 [L0]", "no_action_needed": False, "checked_by_power_flow": True,
                 "after_action": [], "actions": [{"unit": "Gen_bus4", "dp_mw": 3.8},
                                                 {"unit": "Gen_bus10", "dp_mw": 0.0, "setpoint_vm_pu_before": 1.041,
                                                  "setpoint_vm_pu_after": 1.05}]},
                {"outage": "Bus_9 -> Bus_19 [L24]", "no_action_needed": True, "checked_by_power_flow": True,
                 "after_action": [], "actions": []}]
        return {"mode": "options", "status": "options", "timestamp": "2000-01-06 20:00", "set": "n1",
                "n1_study": self.N1_STUDY,
                "options": [{**self.option("least_disturbance", "Least disturbance", 5.0), "n1": {"screen": screen}},
                            {**self.option("n1_preventive", "N-1 preventive", 9.0),
                             "n1": {"screen": {**screen, "outages_with_violations": 16}, "studied": self.N1_STUDY[
                                 "outages_studied"], "secured": 1,
                                    "not_secured": [{"outage": "Bus_9 -> Bus_19 [L24]", "remaining": ["x"]}]}},
                            {**self.option("n1_corrective", "N-1 corrective", 7.0),
                             "n1": {"screen": {**screen, "outages_with_violations": 17}, "studied": self.N1_STUDY[
                                 "outages_studied"], "secured": 2, "not_secured": [], "corrective_plan": plan,
                                    "screen_is": "before any corrective action"}}]}

    def test_an_n1_set_shows_each_options_screen_and_the_corrective_plan(self):
        figs = RENDERER_MAP["optimize_flexibility"](self.n1_result())
        assert figs[0].layout.title.text.startswith("Options for N-1 security at 2000-01-06 20:00<br><sup>2 critical "
                                                    "outage(s) studied (preventive and corrective)")
        screen = figs[1].data[0].cells.values
        assert figs[1].layout.title.text.startswith("N-1 screen of each option")
        assert list(screen[1]) == ["29 of 41", "16 of 41", "17 of 41 before its action"]
        assert list(screen[3]) == ["not studied", "1 of 2", "2 of 2"]
        assert list(screen[4]) == ["—", "L24", "none"]
        plan = figs[2]
        assert plan.layout.title.text.startswith("Corrective plan of N-1 corrective")
        headers = [h.replace("<b>", "").replace("</b>", "") for h in plan.data[0].header.values]
        assert headers == ["After the outage of", "Gen_bus4", "Gen_bus10", "Power flow"]     # Gen_bus4 first
        rows = list(zip(*plan.data[0].cells.values))
        assert rows[0] == ("Bus_0 -> Bus_1 [L0]", "+3.80 MW", "V 1.041→1.050", "✓ confirmed")
        assert rows[1] == ("Bus_9 -> Bus_19 [L24]", "—", "—", "✓ no action needed")
        assert len(figs) == 4                     # and the active-power bars

    def test_no_action_needed_is_one_message(self):
        figs = RENDERER_MAP["optimize_flexibility"]({"mode": "options", "status": "no_action_needed",
                                                     "timestamp": "2000-01-03 18:00", "options": []})
        assert len(figs) == 1 and "No corrective action needed" in figs[0].layout.title.text


class TestScreenOfAnOption:
    def test_the_title_says_what_was_screened(self):
        result = {"total_outages_tested": 41, "total_outages_causing_violations": 31, "system_n1_secure": False,
                  "thresholds_used": {"max_line_loading_pct": 90},
                  "violations": [{"violation_type": "line_loading", "value": 95.0, "element_name": "L0",
                                  "outage_cause": "L1"}],
                  "option": {"ref": "S1:cheapest", "label": "Cheapest", "screen": "N-1"}}
        figs = RENDERER_MAP["simulate_all_contingencies"](result)
        fig = figs if not isinstance(figs, list) else figs[-1]
        assert fig.layout.title.text.startswith("N-1 screen with option S1:cheapest (Cheapest) applied — 31 of 41")
        after = dict(result, option={**result["option"], "after_outage": "L36", "screen": "N-1-1"})
        fig = RENDERER_MAP["simulate_all_contingencies"](after)
        fig = fig if not isinstance(fig, list) else fig[-1]
        assert fig.layout.title.text.startswith("N-1-1 screen after the outage of L36, with option S1:cheapest")
