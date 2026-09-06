"""The agent must never describe a grid it cannot verify.

`get_grid_constants()` falls back to constants for the bundled IEEE 14-bus case
when the backend is unreachable. Those constants are not a degraded version of
the real network — they are a *different* network, with different bus names and
different voltage limits. Handing them to the model unannounced produces
confident, specific, wrong operational statements.

These tests pin the two halves of the defence: the fetch reports whether it
succeeded, and the system prompt tells the model to stay quiet when it didn't.
"""

import httpx
import pytest

from llm_agent.agent import config, system_prompt


class _FakeResponse:
    def __init__(self, status_code, payload=None):
        self.status_code = status_code
        self._payload = payload or {}

    def json(self):
        return self._payload


LIVE = {
    "name": "Coastal Distribution Network",
    "n_substations": 2,
    "substation_names": ["SUB_A", "SUB_B"],
    "n_lines": 3,
    "n_trafos": 1,
    "vm_lower": 0.90,
    "vm_upper": 1.10,
    "max_loading_pct": 80.0,
    "slack_name": "Grid Connection",
}


@pytest.fixture(autouse=True)
def _clear_status():
    config._last_fetch = None
    yield
    config._last_fetch = None


class TestFetchReportsProvenance:
    def test_live_fetch_is_marked_live(self, monkeypatch):
        monkeypatch.setattr(
            config.httpx, "get", lambda *a, **k: _FakeResponse(200, LIVE)
        )
        result = config.fetch_grid_constants()
        assert result.live is True
        assert result.values == LIVE
        assert result.error is None

    def test_unreachable_backend_is_marked_not_live(self, monkeypatch):
        def boom(*a, **k):
            raise httpx.ConnectError("connection refused")

        monkeypatch.setattr(config.httpx, "get", boom)
        result = config.fetch_grid_constants()
        assert result.live is False
        assert "connection refused" in result.error

    def test_non_200_is_marked_not_live(self, monkeypatch):
        monkeypatch.setattr(config.httpx, "get", lambda *a, **k: _FakeResponse(503))
        result = config.fetch_grid_constants()
        assert result.live is False
        assert "503" in result.error

    def test_fallback_values_are_still_returned(self, monkeypatch):
        monkeypatch.setattr(config.httpx, "get", lambda *a, **k: _FakeResponse(503))
        assert config.fetch_grid_constants().values == config.DEFAULT_GRID_CONSTANTS

    def test_status_is_queryable_without_refetching(self, monkeypatch):
        monkeypatch.setattr(config.httpx, "get", lambda *a, **k: _FakeResponse(503))
        config.fetch_grid_constants()
        assert config.last_grid_constants_status().live is False

    def test_status_is_none_before_any_fetch(self):
        assert config.last_grid_constants_status() is None

    def test_generous_timeout(self):
        # A 3s timeout meant a briefly busy backend silently swapped the agent
        # onto a different network; this must stay well above that.
        assert config.GRID_CONSTANTS_TIMEOUT >= 10


class TestBackwardCompatibility:
    def test_get_grid_constants_still_returns_a_plain_dict(self, monkeypatch):
        monkeypatch.setattr(
            config.httpx, "get", lambda *a, **k: _FakeResponse(200, LIVE)
        )
        assert config.get_grid_constants() == LIVE


class TestSystemPromptGuardsIdentity:
    def test_live_prompt_carries_no_warning(self, monkeypatch):
        monkeypatch.setattr(
            config.httpx, "get", lambda *a, **k: _FakeResponse(200, LIVE)
        )
        prompt = system_prompt.get_system_prompt()
        assert system_prompt.STALE_CONSTANTS_WARNING not in prompt
        assert "Coastal Distribution Network" in prompt

    def test_degraded_prompt_carries_the_warning(self, monkeypatch):
        monkeypatch.setattr(config.httpx, "get", lambda *a, **k: _FakeResponse(503))
        prompt = system_prompt.get_system_prompt()
        assert system_prompt.STALE_CONSTANTS_WARNING in prompt

    def test_warning_forbids_naming_the_grid(self, monkeypatch):
        monkeypatch.setattr(config.httpx, "get", lambda *a, **k: _FakeResponse(503))
        prompt = system_prompt.get_system_prompt()
        assert "Never state the grid" in prompt

    def test_warning_tells_the_model_to_disclose_the_outage(self, monkeypatch):
        monkeypatch.setattr(config.httpx, "get", lambda *a, **k: _FakeResponse(503))
        assert "backend is unreachable" in system_prompt.get_system_prompt()

    def test_warning_appears_after_the_placeholder_description(self, monkeypatch):
        # It has to override what precedes it, so ordering is load-bearing.
        monkeypatch.setattr(config.httpx, "get", lambda *a, **k: _FakeResponse(503))
        prompt = system_prompt.get_system_prompt()
        placeholder = config.DEFAULT_GRID_CONSTANTS["name"]
        assert prompt.index(placeholder) < prompt.index("GRID IDENTITY UNVERIFIED")

    def test_recovery_drops_the_warning_again(self, monkeypatch):
        monkeypatch.setattr(config.httpx, "get", lambda *a, **k: _FakeResponse(503))
        assert system_prompt.STALE_CONSTANTS_WARNING in system_prompt.get_system_prompt()

        monkeypatch.setattr(
            config.httpx, "get", lambda *a, **k: _FakeResponse(200, LIVE)
        )
        assert system_prompt.STALE_CONSTANTS_WARNING not in system_prompt.get_system_prompt()

    def test_prompt_tracks_a_network_swap(self, monkeypatch):
        """Uploading a new network must change the prompt, not just the tools."""
        monkeypatch.setattr(
            config.httpx, "get", lambda *a, **k: _FakeResponse(200, LIVE)
        )
        assert "SUB_A" in system_prompt.get_system_prompt()

        other = {**LIVE, "name": "Other Grid", "substation_names": ["ZZZ"]}
        monkeypatch.setattr(
            config.httpx, "get", lambda *a, **k: _FakeResponse(200, other)
        )
        prompt = system_prompt.get_system_prompt()
        assert "ZZZ" in prompt and "SUB_A" not in prompt
