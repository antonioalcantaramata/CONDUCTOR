"""
The provider catalog (llm_agent/provider_catalog.yaml) loads, validates, and
selects the option an environment means — including existing `.env` files
that predate tiers.
"""

import copy

import pytest

from llm_agent.agent import catalog as cat


@pytest.fixture
def data():
    return copy.deepcopy(cat.load())


class TestTheShippedCatalog:
    def test_it_loads_and_validates(self, data):
        ids = [o["id"] for o in data["options"]]
        assert ids[:3] == ["gemini_free", "nvidia_free", "openrouter_free"]
        assert {o["tier"] for o in data["options"]} == set(cat.TIERS)

    def test_the_free_tier_is_gemini_nvidia_and_openrouter_only(self, data):
        free = [o.provider for o in cat.options(data, tier="free")]
        assert free == ["google", "nvidia", "openrouter"]
        assert not any(o["provider"] in ("mistral", "zai") for o in data["options"])

    def test_every_free_option_has_a_data_policy_and_limits_with_sources(self, data):
        for o in cat.options(data):
            if o.is_cloud_free:
                assert o["data_policy_url"] and o["data_policy_summary"]
                limits = [o.get("account_limits")] + [m.get("limits") for m in o.models]
                for lim in filter(None, limits):
                    assert lim["source_url"] and lim["last_verified"] and lim["verified"] in cat.VERIFIED

    def test_gemma_is_never_offered_on_the_gemini_api(self, data):
        gemini = cat.option("gemini_free", data)
        assert not any("gemma" in m["id"] for m in gemini.models)
        assert all(m["limits"]["tpm"] == 16000 for m in gemini["incompatible_models"])

    def test_nvidia_ids_are_the_api_ids(self, data):
        ids = [m["id"] for m in cat.option("nvidia_free", data).models]
        assert ids == ["nvidia/nemotron-3-super-120b-a12b", "nvidia/nemotron-3-ultra-550b-a55b",
                       "nvidia/nemotron-3.5-lightning-30b-a3b", "z-ai/glm-5.3", "z-ai/glm-5.3-flash",
                       "deepseek-ai/deepseek-v4.1-flash"]
        assert cat.option("nvidia_free", data)["default_model"] == "nvidia/nemotron-3-super-120b-a12b"

    def test_the_gemini_fallback_chain(self, data):
        assert cat.option("gemini_free", data)["fallback_chain"] == ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite"]

    def test_limits_come_from_the_model_or_the_account(self, data):
        assert cat.option("gemini_free", data).limits_for("gemini-3.8-flash")["rpd"] == 20
        assert cat.option("nvidia_free", data).limits_for("z-ai/glm-5.3")["rpm"] == 40
        assert cat.option("openrouter_free", data).limits_for("any:free")["rpd"] == 50


class TestValidation:
    def _first(self, data, option_id="nvidia_free"):
        return next(o for o in data["options"] if o["id"] == option_id)

    def test_a_limit_without_a_source_is_refused(self, data):
        self._first(data)["account_limits"].pop("source_url")
        with pytest.raises(cat.CatalogError, match="source_url"):
            cat.validate(data)

    def test_an_unknown_tier_is_refused(self, data):
        self._first(data)["tier"] = "gratis"
        with pytest.raises(cat.CatalogError, match="tier"):
            cat.validate(data)

    def test_a_model_without_tool_calling_is_refused(self, data):
        self._first(data)["models"][0]["tools"] = False
        with pytest.raises(cat.CatalogError, match="tool calling"):
            cat.validate(data)

    def test_a_fallback_model_must_be_listed(self, data):
        self._first(data, "gemini_free")["fallback_chain"].append("gemini-9-ultra")
        with pytest.raises(cat.CatalogError, match="fallback"):
            cat.validate(data)

    def test_duplicate_ids_are_refused(self, data):
        data["options"].append(copy.deepcopy(self._first(data)))
        with pytest.raises(cat.CatalogError, match="duplicate"):
            cat.validate(data)

    def test_a_negative_limit_is_refused(self, data):
        self._first(data)["account_limits"]["rpm"] = -1
        with pytest.raises(cat.CatalogError, match="rpm"):
            cat.validate(data)

    def test_a_free_option_needs_a_data_policy_link(self, data):
        self._first(data, "openrouter_free")["data_policy_url"] = None
        with pytest.raises(cat.CatalogError, match="data_policy_url"):
            cat.validate(data)

    def test_a_broken_file_names_the_problem(self, tmp_path):
        bad = tmp_path / "catalog.yaml"
        bad.write_text("schema_version: 1\noptions: [\n", encoding="utf-8")
        with pytest.raises(cat.CatalogError, match="not valid YAML"):
            cat.load(bad)


class TestWhichOptionAnEnvironmentMeans:
    @pytest.mark.parametrize("env, expected", [
        ({}, "gemini_free"),                                                     # default
        ({"LLM_PROVIDER": "google"}, "gemini_free"),                             # an old .env: free
        ({"LLM_PROVIDER": "google", "GEMINI_TIER": "paid"}, "gemini_paid"),
        ({"LLM_PROVIDER": "openai", "OPENAI_BASE_URL": "https://api.openai.com/v1"}, "openai_paid"),
        ({"LLM_PROVIDER": "openai"}, "openai_paid"),
        ({"LLM_PROVIDER": "openai", "OPENAI_BASE_URL": "https://my-vllm:8000/v1"}, "openai_compatible"),
        ({"LLM_PROVIDER": "anthropic"}, "anthropic_paid"),
        ({"LLM_PROVIDER": "ollama"}, "ollama_local"),
        ({"LLM_PROVIDER": "openrouter"}, "openrouter_free"),
        ({"LLM_PROVIDER": "openrouter", "OPENROUTER_TIER": "paid"}, "openrouter_paid"),
        ({"LLM_PROVIDER": "nvidia"}, "nvidia_free"),
        ({"LLM_PROVIDER": "deepseek"}, "deepseek_paid"),
        ({"LLM_PROVIDER": "deepseek", "DEEPSEEK_TIER": "trial"}, "deepseek_paid"),   # an old .env
        ({"LLM_PROVIDER": "mistral"}, "gemini_free"),                            # unknown → first option
    ])
    def test_mapping(self, data, env, expected):
        assert cat.active_option_id(env, data) == expected


class TestCatalogAge:
    def test_days_since_verified(self, data):
        import datetime as dt
        option = cat.option("nvidia_free", data)
        assert cat.days_since_verified(option, dt.date(2026, 10, 24)) == 10

    def test_stale_after_the_threshold(self, data, monkeypatch):
        import datetime as dt
        monkeypatch.delenv("CATALOG_STALE_DAYS", raising=False)
        option = cat.option("nvidia_free", data)
        assert cat.stale_days(option, dt.date(2026, 11, 1)) is None
        assert cat.stale_days(option, dt.date(2027, 1, 1)) == 79
        monkeypatch.setenv("CATALOG_STALE_DAYS", "5")
        assert cat.stale_days(option, dt.date(2026, 10, 24)) == 10


def test_thinking_must_map_levels_to_fields(data):
    nv = next(o for o in data["options"] if o["id"] == "nvidia_free")
    nv["models"][0]["thinking"] = {"off": "enable_thinking=false"}
    with pytest.raises(cat.CatalogError, match="thinking"):
        cat.validate(data)


def test_every_hosted_option_explains_how_to_get_a_key(data):
    for o in cat.options(data):
        if o.tier != "local":
            assert o.get("setup_steps"), o.id
    nv = cat.option("nvidia_free", data)
    assert any("phone" in step for step in nv["setup_steps"])


def test_setup_steps_must_be_sentences(data):
    next(o for o in data["options"] if o["id"] == "nvidia_free")["setup_steps"] = "sign up"
    with pytest.raises(cat.CatalogError, match="setup_steps"):
        cat.validate(data)
