from llm_agent.agent import tool_schemas as ts


class TestPropBuilders:
    def test_str_prop(self):
        assert ts._str_prop("a description") == {"type": "string", "description": "a description"}

    def test_num_prop(self):
        assert ts._num_prop("a number") == {"type": "number", "description": "a number"}

    def test_int_prop(self):
        assert ts._int_prop("an int") == {"type": "integer", "description": "an int"}

    def test_bool_prop(self):
        assert ts._bool_prop("a bool") == {"type": "boolean", "description": "a bool"}

    def test_arr_str_prop(self):
        assert ts._arr_str_prop("a list") == {
            "type": "array",
            "items": {"type": "string"},
            "description": "a list",
        }

class TestSharedParameterRegistration:
    """Point-in-time tools must declare the parameters that select an operating
    point.

    A tool registered in the dispatch table but omitted from
    `_POINT_IN_TIME_TOOLS` never has `timestamp` / `data_source` injected into
    its schema. The model then cannot reliably target an operating point, and
    the tool silently analyses the simulation clock instead — observed live as
    answers about the wrong day in 4 of 10 runs, with every figure correct.
    """

    def _properties(self, tools):
        from llm_agent.agent.providers.schema import to_json_schema_tools
        return {
            t["function"]["name"]: set((t["function"].get("parameters") or {})
                                       .get("properties") or {})
            for t in to_json_schema_tools(tools)
        }

    def test_point_in_time_tools_declare_timestamp_and_data_source(self):
        from llm_agent.agent import tool_schemas

        props = self._properties(tool_schemas.TOOLS)
        for name in tool_schemas._POINT_IN_TIME_TOOLS:
            assert "timestamp" in props[name], f"{name} lacks timestamp"
            assert "data_source" in props[name], f"{name} lacks data_source"

    def test_every_analysis_tool_can_select_a_dataset(self):
        """Only clock control and pure post-processing may omit data_source."""
        from llm_agent.agent import tool_schemas

        props = self._properties(tool_schemas.TOOLS)
        for name in props:
            if name in tool_schemas._NO_DATA_SOURCE_TOOLS:
                continue
            assert "data_source" in props[name], f"{name} lacks data_source"
