"""Tool-schema conversion from google-genai declarations to plain JSON Schema.

`tool_schemas.TOOLS` is the single source of truth and is expressed in genai
types; non-Gemini backends need the same tools as JSON Schema. These tests pin
that conversion so the two backends can never silently disagree about what tools
exist or what arguments they take.
"""

import json

from llm_agent.agent.providers import schema
from llm_agent.agent.tool_schemas import TOOL_DISPATCH, TOOLS


class TestFunctionDeclarations:
    def test_flattens_every_declared_tool(self):
        assert len(schema.function_declarations(TOOLS)) == len(TOOL_DISPATCH)

    def test_handles_a_tool_with_no_declarations(self):
        class Empty:
            function_declarations = None

        assert schema.function_declarations([Empty()]) == []


class TestNormalize:
    def test_lowercases_genai_type_names(self):
        node = {"type": "OBJECT", "properties": {"x": {"type": "STRING"}}}
        assert schema._normalize(node) == {
            "type": "object", "properties": {"x": {"type": "string"}}
        }

    def test_recurses_through_lists(self):
        node = {"anyOf": [{"type": "NUMBER"}, {"type": "INTEGER"}]}
        assert schema._normalize(node)["anyOf"] == [
            {"type": "number"}, {"type": "integer"}
        ]

    def test_leaves_non_type_values_untouched(self):
        node = {"description": "An OBJECT of things", "type": "STRING"}
        out = schema._normalize(node)
        assert out["description"] == "An OBJECT of things"
        assert out["type"] == "string"

    def test_handles_enum_valued_type_attribute(self):
        class FakeEnum:
            value = "ARRAY"

        assert schema._normalize({"type": FakeEnum()})["type"] == "array"


class TestToJsonSchemaTools:
    def test_converts_every_tool(self):
        assert len(schema.to_json_schema_tools(TOOLS)) == len(TOOL_DISPATCH)

    def test_names_match_the_dispatch_table(self):
        # A tool the model can call but the loop cannot dispatch is a dead end,
        # and vice versa; these two lists must stay identical.
        converted = {t["function"]["name"] for t in schema.to_json_schema_tools(TOOLS)}
        assert converted == set(TOOL_DISPATCH)

    def test_output_is_json_serialisable(self):
        # It is POSTed to Ollama as a JSON body; a stray enum would 500 there.
        json.dumps(schema.to_json_schema_tools(TOOLS))

    def test_no_uppercase_types_survive_anywhere(self):
        blob = json.dumps(schema.to_json_schema_tools(TOOLS))
        assert '"type": "OBJECT"' not in blob
        assert '"type": "STRING"' not in blob

    def test_every_tool_has_openai_shape(self):
        for tool in schema.to_json_schema_tools(TOOLS):
            assert tool["type"] == "function"
            assert set(tool["function"]) == {"name", "description", "parameters"}
            assert tool["function"]["name"]

    def test_parameters_always_present_even_for_argumentless_tools(self):
        for tool in schema.to_json_schema_tools(TOOLS):
            params = tool["function"]["parameters"]
            assert params.get("type") == "object"

    def test_nested_object_properties_are_preserved(self):
        by_name = {
            t["function"]["name"]: t for t in schema.to_json_schema_tools(TOOLS)
        }
        # optimize_flexibility is the most deeply structured schema in the set.
        params = by_name["optimize_flexibility"]["function"]["parameters"]
        assert params["properties"]
        nested = [
            v for v in params["properties"].values()
            if isinstance(v, dict) and v.get("type") == "object"
        ]
        assert nested, "expected at least one nested object property"
