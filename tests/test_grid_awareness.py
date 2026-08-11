"""Nothing may describe the grid from constants frozen at import time.

The agent's view of the network comes from the backend and changes when a user
uploads a different one. Two places used to ignore that by capturing the bundled
example network's constants at import:

  * chart threshold lines (renderers), drawn at the wrong voltages;
  * substation and slack names baked into tool descriptions, naming buses that
    do not exist in the loaded network.

Neither failed loudly — they produced confident, plausible, wrong output — so
these tests exist to keep them honest.
"""

import json

from llm_agent.agent import config, renderers
from llm_agent.agent.providers.schema import to_json_schema_tools
from llm_agent.agent.tool_schemas import TOOLS

import pytest


@pytest.fixture(autouse=True)
def _reset_constants_cache():
    config._last_fetch = None
    yield
    config._last_fetch = None


def _loaded(**values):
    """Pretend the backend reported a network with these constants."""
    config._last_fetch = config.GridConstants(values=values, live=True)


class TestChartLimitsFollowTheLoadedNetwork:
    def test_defaults_apply_before_any_fetch(self):
        limits = renderers._limits()
        assert limits.vm_lower == config.DEFAULT_GRID_CONSTANTS["vm_lower"]
        assert limits.vm_upper == config.DEFAULT_GRID_CONSTANTS["vm_upper"]

    def test_limits_track_the_loaded_network(self):
        _loaded(vm_lower=0.90, vm_upper=1.10, max_loading_pct=80.0)
        assert renderers._limits() == (0.90, 1.10, 80.0)

    def test_limits_are_not_frozen_at_import(self):
        """A network swap mid-session must move the threshold lines."""
        _loaded(vm_lower=0.90, vm_upper=1.10, max_loading_pct=80.0)
        first = renderers._limits()
        _loaded(vm_lower=0.95, vm_upper=1.05, max_loading_pct=120.0)
        assert renderers._limits() != first

    def test_missing_keys_fall_back_individually(self):
        # A sparse payload must not wipe out the other limits.
        _loaded(vm_lower=0.85)
        limits = renderers._limits()
        assert limits.vm_lower == 0.85
        assert limits.vm_upper == config.DEFAULT_GRID_CONSTANTS["vm_upper"]

    def test_degraded_fetch_still_yields_usable_limits(self):
        config._last_fetch = config.GridConstants(
            values=config.DEFAULT_GRID_CONSTANTS, live=False, error="down"
        )
        limits = renderers._limits()
        assert limits.vm_lower < limits.vm_upper


class TestToolSchemasAreGridAgnostic:
    @pytest.fixture
    def schema_text(self):
        return json.dumps(to_json_schema_tools(TOOLS))

    def test_no_example_network_bus_names(self, schema_text):
        # Bus_1..Bus_14 belong to the bundled case, not to whatever is loaded.
        for name in config.DEFAULT_GRID_CONSTANTS["substation_names"]:
            assert f"'{name}'" not in schema_text

    def test_no_hardcoded_slack_name(self, schema_text):
        assert config.DEFAULT_GRID_CONSTANTS["slack_name"] not in schema_text

    def test_descriptions_defer_to_the_system_prompt(self, schema_text):
        # The prompt is rebuilt per turn from live constants, so it is the one
        # place allowed to name buses.
        assert "system prompt" in schema_text

    def test_schemas_do_not_change_with_the_loaded_network(self):
        """They are built once at import, so they must contain nothing grid-specific."""
        before = json.dumps(to_json_schema_tools(TOOLS))
        _loaded(substation_names=["ZZZ"], slack_name="Some Other Slack")
        assert json.dumps(to_json_schema_tools(TOOLS)) == before


class TestSystemPromptCarriesTheNames:
    """Removing names from schemas is only safe because the prompt has them."""

    def test_prompt_lists_the_live_substations(self):
        from llm_agent.agent.system_prompt import build_system_prompt

        prompt = build_system_prompt({
            **config.DEFAULT_GRID_CONSTANTS,
            "substation_names": ["ALPHA", "BETA"],
            "n_substations": 2,
        })
        assert "ALPHA" in prompt and "BETA" in prompt

    def test_prompt_names_the_live_slack(self):
        from llm_agent.agent.system_prompt import build_system_prompt

        prompt = build_system_prompt({
            **config.DEFAULT_GRID_CONSTANTS, "slack_name": "Cable Connection",
        })
        assert "Cable Connection" in prompt
