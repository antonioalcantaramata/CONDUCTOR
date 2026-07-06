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

    def test_options_prop(self):
        result = ts._options_prop()
        assert result["type"] == "object"
        assert isinstance(result["description"], str) and result["description"]
