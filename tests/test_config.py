import httpx

from llm_agent.agent import config


class TestGetGridConstants:
    def test_returns_backend_json_on_200(self, monkeypatch):
        live_constants = {"name": "Live Network", "n_lines": 99}

        class FakeResponse:
            status_code = 200

            def json(self):
                return live_constants

        monkeypatch.setattr(config.httpx, "get", lambda *a, **k: FakeResponse())

        assert config.get_grid_constants() == live_constants

    def test_falls_back_to_default_on_non_200(self, monkeypatch):
        class FakeResponse:
            status_code = 503

            def json(self):
                raise AssertionError("should not be called")

        monkeypatch.setattr(config.httpx, "get", lambda *a, **k: FakeResponse())

        assert config.get_grid_constants() == config.DEFAULT_GRID_CONSTANTS

    def test_falls_back_to_default_when_backend_unreachable(self, monkeypatch):
        def raise_connect_error(*a, **k):
            raise httpx.ConnectError("connection refused")

        monkeypatch.setattr(config.httpx, "get", raise_connect_error)

        assert config.get_grid_constants() == config.DEFAULT_GRID_CONSTANTS


class TestEnvOverrides:
    def test_base_url_defaults_to_localhost(self):
        assert config.BASE_URL.startswith("http")

    def test_default_grid_constants_has_expected_shape(self):
        constants = config.DEFAULT_GRID_CONSTANTS
        assert constants["n_substations"] == len(constants["substation_names"])
        assert constants["vm_lower"] < constants["vm_upper"]
