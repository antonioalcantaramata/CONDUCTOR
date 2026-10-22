"""
The tool registry (llm_agent/agent/tool_registry.py) is the one place that
describes each tool: its label, its documentation section, what calling it
changes, and the shared arguments it takes. These tests keep it complete and
in step with the schemas, the functions and the charts.
"""

import pathlib

import pytest

from llm_agent.agent import tool_registry as reg
from llm_agent.agent.providers.schema import to_json_schema_tools
from llm_agent.agent.tool_schemas import TOOL_DISPATCH, TOOLS

SCHEMAS = {t["function"]["name"]: t["function"] for t in to_json_schema_tools(TOOLS)}
APP = pathlib.Path(__file__).resolve().parents[1] / "llm_agent" / "app.py"


def test_every_tool_has_a_row_a_schema_and_a_function():
    assert set(reg.REGISTRY) == set(SCHEMAS) == set(TOOL_DISPATCH)


def test_every_chart_belongs_to_a_tool():
    pytest.importorskip("plotly")
    from llm_agent.agent.renderers import RENDERER_MAP
    assert set(RENDERER_MAP) <= set(reg.REGISTRY)


def test_rows_are_well_formed():
    labels = [t.label for t in reg.REGISTRY.values()]
    assert all(labels) and len(set(labels)) == len(labels)
    assert all(t.group in reg.GROUPS for t in reg.REGISTRY.values())
    assert all(reg.tools_in(g) for g in reg.GROUPS)
    assert all(t.effect in reg.EFFECT_TEXT for t in reg.REGISTRY.values())
    assert all(t.effect == reg.CHANGES_SETTINGS for t in reg.REGISTRY.values() if t.reads_without_changes)
    assert all(t.data_source in (None, reg.SERIES, reg.POINT) for t in reg.REGISTRY.values())


def test_the_app_keeps_no_label_list_of_its_own():
    assert "_TOOL_LABELS" not in APP.read_text(encoding="utf-8")


def test_the_deterministic_opf_is_not_called_robust():
    assert "robust" not in reg.label("optimize_flexibility").lower()


def test_an_unknown_tool_still_gets_a_readable_label():
    assert reg.label("brand_new_tool") == "Brand New Tool"


@pytest.mark.parametrize("name", sorted(reg.REGISTRY))
def test_the_schema_has_the_shared_arguments_the_row_says(name):
    row = reg.REGISTRY[name]
    props = (SCHEMAS[name].get("parameters") or {}).get("properties") or {}
    assert ("network" in props) == row.network
    assert ("load_power_factor" in props) == row.load_pf
    assert ("timestamp" in props) == row.timestamp
    assert ("data_source" in props) == bool(row.data_source)
    if row.data_source:
        assert ("nominal" in props["data_source"]["description"]) == (row.data_source == reg.POINT)


@pytest.mark.parametrize("name, kwargs, changes", [
    ("run_rsa", {}, False),
    ("get_current_conditions", {"timestamp": "x"}, False),
    ("advance_timestamp", {"steps": 1}, True),
    ("scan_rsa_over_time", {"n_steps": 4}, True),          # it steps the clock forward as it scans
    ("edit_proposal", {"proposal": "p", "edits": []}, True),
    ("set_load_power_factor", {"value": 0.9}, True),
    ("set_generator_model", {}, False),                    # read the models
    ("set_generator_model", {"changes": [{"unit": "G"}]}, True),
    ("set_unit_costs", {"reset": True}, True),
    ("rename_elements", {}, False),
    ("not_a_tool", {}, False),
])
def test_what_changes_state(name, kwargs, changes):
    assert reg.changes_state(name, kwargs) is changes
