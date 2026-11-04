"""
The raw-tables baseline (`CONDUCTOR_ONTOLOGY=tables`): the agent is given the
network's own data tables instead of the semantic layer, and the studies keep
reading the layer — the arm that asks what the layer adds over the data it is
built from. Without model credit.
"""

import json

import pandapower.networks as pn
import pytest

import raw_tables as rt
from evaluation import arms
from llm_agent.agent import config, system_prompt, tool_schemas, tools
from llm_agent.agent.tool_registry import EVALUATION_ONLY


def _declared():
    return {fd.name for t in tool_schemas.tools_for_model() for fd in t.function_declarations}


@pytest.mark.parametrize("mode, layer, tables", [("on", True, False), ("off", False, False),
                                                 ("none", False, False), ("tables", False, True)])
def test_the_switch(monkeypatch, mode, layer, tables):
    monkeypatch.setenv("CONDUCTOR_ONTOLOGY", mode)
    monkeypatch.delenv("CONDUCTOR_HIDE_TOOLS", raising=False)
    assert config.ontology_enabled() is layer and config.raw_tables_enabled() is tables
    declared = _declared()
    assert ("describe_network" in declared) is layer and ("declare_network_facts" in declared) is layer
    assert ("read_network_tables" in declared) is tables
    assert "locate_network_element" in declared


def test_the_arm_keeps_the_studies_on_the_layer():
    arm = arms.parse("tables: ontology=tables")
    assert arm.env()["CONDUCTOR_ONTOLOGY"] == "tables" and arm.backend_mode() == "on"
    footprint = arms.footprint(arm)
    assert {"describe_network", "declare_network_facts"} <= set(footprint["tools_hidden"])
    assert footprint["tools_added"] == ["read_network_tables"]
    assert arms.PRESETS["raw_tables"].arms == ("on: ontology=on", "off: ontology=off", "tables: ontology=tables")


class _Down:
    status_code = 503

    def json(self):
        return {}


def test_the_prompt_points_to_the_tables_with_the_same_care_about_names(monkeypatch):
    monkeypatch.setattr(config.httpx, "get", lambda *a, **k: _Down())
    monkeypatch.delenv("CONDUCTOR_DROP_RULES", raising=False)
    monkeypatch.setenv("CONDUCTOR_ONTOLOGY", "on")
    with_layer = system_prompt.get_system_prompt()
    monkeypatch.setenv("CONDUCTOR_ONTOLOGY", "tables")
    with_tables = system_prompt.get_system_prompt()
    monkeypatch.setenv("CONDUCTOR_ONTOLOGY", "off")
    without = system_prompt.get_system_prompt()
    assert "read_network_tables" not in with_layer and "A name is only a hint" in with_tables
    assert "**What an element is**" not in without
    # Only that rule differs: the rest of the prompt is the layer arm's.
    assert with_tables == system_prompt.replace_part(with_layer, "element_identity", system_prompt.RAW_TABLES_RULE)


def test_the_tables_are_pandapowers_standard_columns_only():
    net = pn.create_cigre_network_mv(with_der="all")
    net.sgen["_q_from_meas"] = False                     # a column CONDUCTOR adds while loading data
    net.load["substation_name"] = "X"
    out = rt.raw_tables(net)
    assert set(out) == set(rt.TABLES)
    assert all(not k.startswith("_") and k != "substation_name" for rows in out.values() for row in rows for k in row)
    types = {row["name"]: row["type"] for row in out["sgen"]}
    assert "CHP diesel" in types.values() and any(t == "PV" for t in types.values())
    assert {"bus", "element", "et", "closed"} <= set(out["switch"][0])
    json.dumps(out)                                       # plain JSON: no numpy scalars, no NaN


def test_an_unknown_table_is_refused_and_a_long_one_cut(monkeypatch):
    net = pn.create_cigre_network_mv()
    with pytest.raises(ValueError):
        rt.raw_tables(net, ["sgens"])
    monkeypatch.setattr(rt, "MAX_ROWS", 5)
    out = rt.raw_tables(net, ["bus"])
    assert len(out["bus"]) == 5 and out["bus_truncated"] == f"first 5 of {len(net.bus)} rows"


def test_the_tool_asks_the_backend_for_the_tables(monkeypatch):
    calls = []
    monkeypatch.setattr(tools, "_get", lambda endpoint: calls.append(endpoint) or {"tables": {}})
    assert tools.read_network_tables() == {"tables": {}}
    tools.read_network_tables(tables=["SGEN", "switch"], network="dc_bus9")
    assert calls == ["/api/network/tables", "/api/network/tables?tables=sgen%2Cswitch&network=dc_bus9"]


def test_the_tool_is_evaluation_only():
    assert EVALUATION_ONLY == {"read_network_tables"}
    assert "read_network_tables" in tool_schemas.TOOL_DISPATCH
