"""
Turning a name into an element index.

Asked which substations would lose supply if the line between Golf and
Hotel were out, the agent had no way to look that line up. It guessed
`element_index=11` — the Foxtrot–Charlie line — ran a perfectly good
contingency study on it, and reported the result as the answer. Every figure
in the reply came from a real solve, the timestamps agreed, and nothing in the
output revealed that a different line had been simulated.

None of the existing guards can see that: provenance checks where a number
came from, and cross-tool consistency checks when it was computed. Neither
asks whether the element was the one the user named.

The other property these pin down is that the tool answers a question rather
than returning a network. The full topology is ~2k tokens on a small grid and
~87k on a 1355-bus one, so it must never reach the model.
"""

import pytest

from llm_agent.agent import tools


TOPOLOGY = {
    "name": "test grid",
    "buses": [
        {"index": 0, "name": "Golf", "vn_kv": 63.0, "in_service": True},
        {"index": 1, "name": "Hotel", "vn_kv": 63.0, "in_service": True},
        {"index": 2, "name": "Bravo", "vn_kv": 63.0, "in_service": True},
        {"index": 3, "name": "Golf 10 kV", "vn_kv": 10.0, "in_service": True},
        {"index": 4, "name": "Aux", "vn_kv": 63.0, "in_service": False},
        {"index": 5, "name": "Aux", "vn_kv": 63.0, "in_service": False},
    ],
    "branches": [
        {"index": 7, "name": "OLS-ØST | Golf -> Hotel [L7]", "from_bus": 0,
         "to_bus": 1, "kind": "line", "in_service": True},
        {"index": 6, "name": "OLS-ALL | Golf -> Bravo [L6]", "from_bus": 0,
         "to_bus": 2, "kind": "line", "in_service": True},
        {"index": 11, "name": "BOD-NEX | Foxtrot -> Charlie [L11]", "from_bus": 1,
         "to_bus": 2, "kind": "line", "in_service": True},
        {"index": 3, "name": "Golf Trf", "from_bus": 0, "to_bus": 3,
         "kind": "trafo", "in_service": True},
    ],
    "external_grids": [],
}


@pytest.fixture(autouse=True)
def topology(monkeypatch):
    monkeypatch.setattr(tools, "_get", lambda endpoint: TOPOLOGY)


class TestResolvingAnElement:
    def test_a_named_bus_yields_the_index_of_every_branch_on_it(self):
        # The lookup the guessed index needed: Golf → Hotel is line 7,
        # and 7 is what `simulate_contingency` has to be called with.
        result = tools.locate_network_element("Golf")
        assert result["bus_index"] == 0
        by_neighbour = {c["name"]: c for c in result["connected_to"]}
        assert by_neighbour["Hotel"]["element_index"] == 7
        assert by_neighbour["Hotel"]["element_type"] == "line"
        assert by_neighbour["Golf 10 kV"]["element_type"] == "trafo"

    def test_the_index_is_named_as_the_other_tools_expect_it(self):
        # `element_type` / `element_index` verbatim, so the model does not have
        # to translate between one tool's vocabulary and another's.
        (first, *_) = tools.locate_network_element("Golf")["connected_to"]
        assert {"element_type", "element_index"} <= set(first)

    def test_a_named_branch_yields_its_index_and_endpoints(self):
        result = tools.locate_network_element("OLS-ØST")
        assert result["element_index"] == 7
        assert set(result["between"]) == {"Golf", "Hotel"}

    def test_matching_ignores_case(self):
        assert tools.locate_network_element("golf")["bus_index"] == 0

    def test_a_bus_and_its_transformer_bus_are_not_confused(self):
        # "Golf" and "Golf 10 kV" are different buses; an exact match on
        # the shorter name must not sweep in the longer one.
        assert tools.locate_network_element("Golf 10 kV")["bus_index"] == 3


class TestRefusingToGuess:
    def test_a_repeated_bus_name_is_refused_with_its_candidates(self):
        # Bus names repeat on real networks — one has four buses called
        # `Mike A_aux`. Picking one is the error this tool exists to stop.
        result = tools.locate_network_element("Aux")
        assert "error" in result
        assert {c["index"] for c in result["candidates"]} == {4, 5}

    def test_an_ambiguous_branch_name_is_refused(self):
        result = tools.locate_network_element("OLS-")
        assert "error" in result
        assert len(result["candidates"]) == 2

    def test_an_unknown_name_says_so_and_offers_the_real_ones(self):
        result = tools.locate_network_element("Nowhere")
        assert "error" in result
        assert "Golf" in result["known_buses"]

    def test_an_empty_name_is_refused(self):
        assert "error" in tools.locate_network_element("  ")

    def test_a_backend_error_is_passed_through(self, monkeypatch):
        monkeypatch.setattr(tools, "_get", lambda endpoint: {"error": "backend down"})
        assert tools.locate_network_element("Golf")["error"] == "backend down"


class TestItAnswersRatherThanDumping:
    def test_the_network_itself_is_never_returned(self):
        # ~2k tokens on a small grid, ~87k on a 1355-bus one. It must not reach
        # the model under any name.
        result = tools.locate_network_element("Golf")
        assert not {"buses", "branches"} & set(result)

    def test_a_busy_bus_is_truncated_and_says_so(self, monkeypatch):
        crowded = dict(TOPOLOGY)
        crowded["buses"] = TOPOLOGY["buses"] + [
            {"index": 100 + i, "name": f"B{i}", "vn_kv": 63.0, "in_service": True}
            for i in range(20)
        ]
        crowded["branches"] = TOPOLOGY["branches"] + [
            {"index": 200 + i, "name": f"L{i}", "from_bus": 0, "to_bus": 100 + i,
             "kind": "line", "in_service": True} for i in range(20)
        ]
        monkeypatch.setattr(tools, "_get", lambda endpoint: crowded)

        result = tools.locate_network_element("Golf")
        assert len(result["connected_to"]) == tools._MAX_NEIGHBOURS
        assert result["truncated"] is True
        assert result["n_connections"] == 23


class TestRegistration:
    def test_the_tool_is_dispatchable_and_declared(self):
        from llm_agent.agent.tool_schemas import TOOL_DISPATCH, _NO_DATA_SOURCE_TOOLS
        from llm_agent.agent.providers.schema import to_json_schema_tools
        from llm_agent.agent.tool_schemas import TOOLS

        assert TOOL_DISPATCH["locate_network_element"] is tools.locate_network_element
        # Topology does not vary with the clock, so it takes no dataset.
        assert "locate_network_element" in _NO_DATA_SOURCE_TOOLS

        declared = {t["function"]["name"]: t["function"] for t in to_json_schema_tools(TOOLS)}
        parameters = declared["locate_network_element"]["parameters"]
        assert parameters["required"] == ["element"]
        # One argument, because every schema is sent on every request.
        assert set(parameters["properties"]) == {"element"}
