"""
Model discovery (llm_agent/agent/discovery/): OpenRouter's free models from a
saved live response plus synthetic cases, NVIDIA's allowlist validated
against its model list, and the local cache. All HTTP is mocked.
"""

import copy
import datetime as dt
import json
import pathlib

import pytest

from llm_agent.agent.discovery import cache, nvidia, openrouter

FIXTURES = pathlib.Path(__file__).parent / "fixtures" / "providers"
TODAY = dt.date(2026, 10, 14)


@pytest.fixture(autouse=True)
def own_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("CONDUCTOR_CACHE_DIR", str(tmp_path))
    for var in ("OPENROUTER_MIN_CONTEXT", "FREE_MODEL_EXPIRY_GRACE_DAYS", "DISCOVERY_CACHE_TTL_H"):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def or_payload():
    return json.loads((FIXTURES / "openrouter_models.json").read_text())


@pytest.fixture
def nv_payload():
    return json.loads((FIXTURES / "nvidia_models.json").read_text())


def _model(**kw):
    base = {"id": "acme/model:free", "name": "Acme", "created": 1, "context_length": 200000,
            "description": "", "pricing": {"prompt": "0", "completion": "0"},
            "architecture": {"input_modalities": ["text"], "output_modalities": ["text"]},
            "supported_parameters": ["tools", "max_tokens"], "expiration_date": None}
    base.update(kw)
    return base


class TestOpenRouterFilter:
    def test_the_live_response_yields_the_qualifying_free_models(self, or_payload):
        result = openrouter.qualifying(or_payload, today=TODAY)
        ids = [m["id"] for m in result["models"]]
        assert len(ids) == 12 and ids[:2] == ["thinkingmachines/inkling-small:free", "thinkingmachines/inkling:free"]
        assert result["excluded"] == {
            "liquid/lfm-2.5-2.6b:free": "context below 131,072 tokens",
            "nvidia/nemotron-3.5-content-safety:free": "no tool calling",
            "google/lyria-3-pro-preview": "no tool calling",
            "google/lyria-3-clip-preview": "no tool calling",
            "openrouter/free": "a router or alias, not a model",
        }
        assert all(m["free"] for m in result["models"])

    def test_free_by_suffix_or_by_zero_price(self):
        assert openrouter.is_free(_model(id="a/b:free", pricing={"prompt": "0.1", "completion": "0.1"}))
        assert openrouter.is_free(_model(id="a/b", pricing={"prompt": "0", "completion": "0.0"}))
        assert not openrouter.is_free(_model(id="a/b", pricing={"prompt": "0", "completion": "0.000001"}))
        assert not openrouter.is_free(_model(id="a/b", pricing={}))

    def test_each_requirement(self):
        assert openrouter.check(_model(), today=TODAY) is None
        assert openrouter.check(_model(supported_parameters=["max_tokens"]), today=TODAY) == "no tool calling"
        assert openrouter.check(_model(architecture={"input_modalities": ["image"], "output_modalities": ["text"]}),
                                today=TODAY) == "not text in and text out"
        assert openrouter.check(_model(context_length=65536), today=TODAY).startswith("context below")
        assert openrouter.check(_model(id="a/paid", pricing={"prompt": "1", "completion": "1"}), today=TODAY) \
            == "not free"
        assert openrouter.check(_model(id="a/paid", pricing={"prompt": "1", "completion": "1"}), free_only=False,
                                today=TODAY) is None

    def test_the_context_floor_is_configurable(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_MIN_CONTEXT", "300000")
        assert openrouter.check(_model(context_length=262144), today=TODAY).startswith("context below 300,000")

    def test_retiring_models_within_the_grace_period_are_hidden(self, or_payload):
        kept = openrouter.qualifying(or_payload, today=TODAY, grace=14)["models"]
        laguna = next(m for m in kept if m["id"] == "poolside/laguna-s-2.1:free")
        assert laguna["expiration_date"] == "2026-10-31" and laguna["days_left"] == 17
        hidden = openrouter.qualifying(or_payload, today=TODAY, grace=20)
        assert hidden["excluded"]["poolside/laguna-s-2.1:free"] == "retires on 2026-10-31"

    def test_data_use_wording_is_flagged_as_a_heuristic(self):
        assert openrouter.data_warning(_model(description="Prompts and outputs may be used to train our models."))
        assert openrouter.data_warning(_model(description="The provider may retain prompts for 30 days."))
        assert openrouter.data_warning(_model(description="Free; prompts may be logged by the provider."))
        assert openrouter.data_warning(_model(description="A strong model for RAG and data extraction.")) is None

    def test_ranking_context_then_recency_then_id(self):
        models = [{"id": "c", "context_length": 200, "created": 5}, {"id": "a", "context_length": 300, "created": 1},
                  {"id": "b", "context_length": 200, "created": 9}, {"id": "d", "context_length": 200, "created": 9}]
        assert [m["id"] for m in openrouter.rank(models)] == ["a", "b", "d", "c"]

    def test_fallbacks_and_next_model(self):
        models = [{"id": i} for i in ("a", "b", "c", "d")]
        assert openrouter.fallbacks(models, "b", 2) == ["a", "c"]
        assert openrouter.next_model(models, {"a", "b"}) == "c"
        assert openrouter.next_model(models, {"a", "b", "c", "d"}) is None

    def test_zero_qualifying_models_is_an_empty_list_not_an_error(self, or_payload):
        none = {"data": [m for m in or_payload["data"] if "tools" not in m["supported_parameters"]]}
        assert openrouter.qualifying(none, today=TODAY)["models"] == []


class TestOpenRouterCache:
    def test_cached_for_the_ttl_and_refreshed_on_demand(self, or_payload, monkeypatch):
        calls = []
        monkeypatch.setattr(openrouter, "fetch", lambda timeout=20.0: calls.append(1) or copy.deepcopy(or_payload))
        first = openrouter.discover()
        again = openrouter.discover()
        assert len(calls) == 1 and not first["from_cache"] and again["from_cache"]
        assert [m["id"] for m in again["models"]] == [m["id"] for m in first["models"]]
        openrouter.discover(force=True)
        assert len(calls) == 2

    def test_a_stale_cache_is_refetched(self, or_payload, monkeypatch):
        calls = []
        monkeypatch.setattr(openrouter, "fetch", lambda timeout=20.0: calls.append(1) or copy.deepcopy(or_payload))
        cache.put("openrouter_free", {"models": [], "excluded": {}, "n_free": 0}, now=0)
        openrouter.discover()
        assert calls == [1]

    def test_a_failed_fetch_falls_back_to_the_last_list(self, monkeypatch):
        cache.put("openrouter_free", {"models": [{"id": "x:free"}], "excluded": {}, "n_free": 1}, now=0)

        def boom(timeout=20.0):
            raise OSError("offline")
        monkeypatch.setattr(openrouter, "fetch", boom)
        result = openrouter.discover()
        assert result["models"] == [{"id": "x:free"}] and "offline" in result["error"]

    def test_a_failed_fetch_without_a_cache_is_empty_with_the_error(self, monkeypatch):
        def boom(timeout=20.0):
            raise OSError("offline")
        monkeypatch.setattr(openrouter, "fetch", boom)
        result = openrouter.discover()
        assert result["models"] == [] and "offline" in result["error"]

    def test_free_and_paid_lists_are_cached_apart(self, or_payload, monkeypatch):
        monkeypatch.setattr(openrouter, "fetch", lambda timeout=20.0: copy.deepcopy(or_payload))
        free = openrouter.discover(free_only=True)["models"]
        paid = openrouter.discover(free_only=False)["models"]
        assert len(paid) > len(free)


ALLOWLIST = [{"id": "z-ai/glm-5.3", "context_window": 1048576, "tools": True},
             {"id": "deepseek-ai/deepseek-v4.1-flash", "context_window": 1048576, "tools": True},
             {"id": "z-ai/glm-5.3-flash", "context_window": 1048576, "tools": True, "smoke_test_required": True}]


class TestNvidiaAllowlist:
    def test_all_present_in_the_live_list(self, nv_payload):
        live = [m["id"] for m in nv_payload["data"]]
        result = nvidia.validate(ALLOWLIST, live)
        assert [m["id"] for m in result["available"]] == [m["id"] for m in ALLOWLIST]
        assert result["missing"] == [] and result["validated"]

    def test_a_model_missing(self, nv_payload):
        live = [m["id"] for m in nv_payload["data"] if m["id"] != "deepseek-ai/deepseek-v4.1-flash"]
        result = nvidia.validate(ALLOWLIST, live)
        assert result["missing"] == ["deepseek-ai/deepseek-v4.1-flash"]
        assert [m["id"] for m in result["available"]] == ["z-ai/glm-5.3", "z-ai/glm-5.3-flash"]

    def test_all_missing(self):
        result = nvidia.validate(ALLOWLIST, ["01-ai/yi-large"])
        assert result["available"] == [] and len(result["missing"]) == 3

    def test_without_a_live_list_the_allowlist_is_unvalidated(self):
        result = nvidia.validate(ALLOWLIST, None)
        assert len(result["available"]) == 3 and not result["validated"]

    def test_a_model_needing_a_test_is_offered_after_it_passes(self):
        flash = ALLOWLIST[2]
        assert nvidia.needs_test(flash) and not nvidia.needs_test(ALLOWLIST[0])
        cache.put_smoke("nvidia", flash["id"], {"ok": False})
        assert nvidia.needs_test(flash)
        cache.put_smoke("nvidia", flash["id"], {"ok": True})
        assert not nvidia.needs_test(flash)

    def test_discover_uses_and_caches_the_live_list(self, nv_payload, monkeypatch):
        calls = []
        monkeypatch.setattr(nvidia, "fetch_ids",
                            lambda timeout=20.0: calls.append(1) or [m["id"] for m in nv_payload["data"]])
        assert len(nvidia.discover(ALLOWLIST)["available"]) == 3
        nvidia.discover(ALLOWLIST)
        assert calls == [1]
        nvidia.discover(ALLOWLIST, force=True)
        assert calls == [1, 1]

    def test_next_model_skips_failed_and_untested(self):
        assert nvidia.next_model(ALLOWLIST, {"z-ai/glm-5.3"}) == "deepseek-ai/deepseek-v4.1-flash"
        assert nvidia.next_model(ALLOWLIST, {"z-ai/glm-5.3", "deepseek-ai/deepseek-v4.1-flash"}) is None


class TestNewerVersions:
    def test_nvidia_families(self):
        assert nvidia.family("z-ai/glm-5.3-flash") == (("z-ai", "glm-", "-flash"), (5, 3))
        assert nvidia.family("deepseek-ai/deepseek-v4.1-flash") == (("deepseek-ai", "deepseek-v", "-flash"), (4, 1))
        assert nvidia.family("01-ai/yi-large") is None

    def test_nvidia_candidates_are_newer_versions_of_the_same_family(self, nv_payload):
        live = [m["id"] for m in nv_payload["data"]] + [
            "z-ai/glm-5.4", "z-ai/glm-5.2", "z-ai/glm-5.4-flash", "deepseek-ai/deepseek-v4.2-flash",
            "deepseek-ai/deepseek-v5-pro", "z-ai/glm-6"]
        ids = [c["id"] for c in nvidia.newer_candidates(ALLOWLIST, live)]
        assert ids == ["deepseek-ai/deepseek-v4.2-flash", "z-ai/glm-5.4", "z-ai/glm-5.4-flash", "z-ai/glm-6"]
        assert all(c["smoke_test_required"] and c["context_window"] is None for c in
                   nvidia.newer_candidates(ALLOWLIST, live))

    def test_no_candidates_in_todays_list(self, nv_payload):
        assert nvidia.newer_candidates(ALLOWLIST, [m["id"] for m in nv_payload["data"]]) == []

    def test_discover_reports_candidates_needing_a_test(self, nv_payload, monkeypatch):
        monkeypatch.setattr(nvidia, "fetch_ids",
                            lambda timeout=20.0: [m["id"] for m in nv_payload["data"]] + ["z-ai/glm-5.4"])
        found = nvidia.discover(ALLOWLIST)
        assert [c["id"] for c in found["candidates"]] == ["z-ai/glm-5.4"]
        assert nvidia.needs_test(found["candidates"][0])
        cache.put_smoke("nvidia", "z-ai/glm-5.4", {"ok": True})
        assert not nvidia.needs_test(found["candidates"][0])


class TestGeminiNewerModels:
    @pytest.fixture
    def listed(self):
        raw = json.loads((FIXTURES / "gemini_models.json").read_text())["models"]
        return [{"id": m["name"].removeprefix("models/"), "input_token_limit": m.get("inputTokenLimit"),
                 "methods": m.get("supportedGenerationMethods") or []} for m in raw]

    @pytest.fixture
    def catalog_models(self):
        from llm_agent.agent import catalog
        return catalog.option("gemini_free").models

    def test_families(self):
        from llm_agent.agent.discovery import gemini
        assert gemini.family("gemini-3-flash-preview") == ("flash", (3,), True)
        assert gemini.family("gemini-3.5-flash-lite") == ("flash-lite", (3, 5), False)
        for alias in ("gemini-flash-latest", "gemini-3.8-flash-tts", "gemini-3.1-flash-image", "gemma-4-31b-it"):
            assert gemini.family(alias) is None

    def test_none_newer_today(self, listed, catalog_models):
        from llm_agent.agent.discovery import gemini
        assert gemini.newer_candidates(catalog_models, listed) == []

    def test_a_newer_flash_lite_and_flash(self, listed, catalog_models):
        from llm_agent.agent.discovery import gemini
        listed += [{"id": "gemini-3.9-flash-lite", "input_token_limit": 1048576, "methods": ["generateContent"]},
                   {"id": "gemini-4-flash-preview", "input_token_limit": 2097152, "methods": ["generateContent"]},
                   {"id": "gemini-3.9-flash-lite-tts", "input_token_limit": 8192, "methods": ["generateContent"]},
                   {"id": "gemini-5-flash", "input_token_limit": 1048576, "methods": ["embedContent"]}]
        found = gemini.newer_candidates(catalog_models, listed)
        assert [(c["id"], c["context_window"]) for c in found] == [
            ("gemini-3.9-flash-lite", 1048576), ("gemini-4-flash-preview", 2097152)]
        assert "preview" in found[1]["label"] and all(c["smoke_test_required"] for c in found)

    def test_discover_without_a_key_does_nothing(self, catalog_models, monkeypatch):
        from llm_agent.agent.discovery import gemini
        monkeypatch.setattr(gemini, "fetch", lambda *a, **k: pytest.fail("no key, no call"))
        assert gemini.discover("", catalog_models)["candidates"] == []

    def test_discover_is_cached_and_survives_errors(self, listed, catalog_models, monkeypatch):
        from llm_agent.agent.discovery import gemini
        calls = []
        monkeypatch.setattr(gemini, "fetch", lambda key, timeout=20.0: calls.append(1) or listed)
        gemini.discover("k", catalog_models)
        gemini.discover("k", catalog_models)
        assert calls == [1]

        def boom(key, timeout=20.0):
            raise OSError("offline")
        monkeypatch.setattr(gemini, "fetch", boom)
        found = gemini.discover("k", catalog_models, force=True)
        assert "offline" in found["error"] and found["candidates"] == []


class TestPaidModelLists:
    def test_openai_chat_models_newest_first(self):
        from llm_agent.agent.discovery import paid
        ids = ["gpt-6-luna", "text-embedding-3-large", "gpt-6", "tts-1", "dall-e-3", "gpt-5.6-luna", "omni-moderation", "gpt-live-1"]
        created = {"gpt-6-luna": 300, "gpt-6": 200, "gpt-5.6-luna": 100}
        assert [m["id"] for m in paid.openai_chat_models(ids, created)] == ["gpt-6-luna", "gpt-6", "gpt-5.6-luna"]

    def test_gemini_text_models_newest_and_biggest_first(self):
        from llm_agent.agent.discovery import paid
        raw = json.loads((FIXTURES / "gemini_models.json").read_text())["models"]
        listed = [{"id": m["name"].removeprefix("models/"), "input_token_limit": m.get("inputTokenLimit"),
                   "methods": m.get("supportedGenerationMethods") or []} for m in raw]
        ids = [m["id"] for m in paid.gemini_text_models(listed)]
        assert ids[0] == "gemini-3.8-flash" and "gemini-3.5-flash-lite" in ids
        assert not any("tts" in i or "image" in i or "gemma" in i or "latest" in i for i in ids)

    def test_discover_without_a_key_is_empty_and_quiet(self, monkeypatch):
        from llm_agent.agent.discovery import paid
        monkeypatch.setattr(paid, "_list", lambda *a: pytest.fail("no key, no call"))
        assert paid.discover("openai", "", None) == {"models": [], "fetched_at": None, "error": None}

    def test_discover_caches_per_provider_and_key(self, monkeypatch):
        from llm_agent.agent.discovery import paid
        calls = []
        monkeypatch.setattr(paid, "_list", lambda provider, key, base: calls.append(provider) or [{"id": provider}])
        assert paid.discover("anthropic", "sk-a")["models"] == [{"id": "anthropic"}]
        paid.discover("anthropic", "sk-a")
        paid.discover("openai", "sk-o", "https://api.openai.com/v1")
        assert calls == ["anthropic", "openai"]
        paid.discover("anthropic", "sk-a", force=True)
        assert calls == ["anthropic", "openai", "anthropic"]

    def test_a_failed_listing_is_an_error_not_a_crash(self, monkeypatch):
        from llm_agent.agent.discovery import paid

        def boom(*a):
            raise RuntimeError("HTTP 401")
        monkeypatch.setattr(paid, "_list", boom)
        found = paid.discover("openai", "bad", "https://api.openai.com/v1")
        assert found["models"] == [] and "401" in found["error"]


def test_a_rejected_key_is_said_plainly(monkeypatch):
    from llm_agent.agent.discovery import paid

    def rejected(*a):
        raise RuntimeError("Error code: 401 - {'type': 'error', 'error': {'type': 'authentication_error', "
                           "'message': 'invalid x-api-key'}, 'request_id': 'req_x'}")
    monkeypatch.setattr(paid, "_list", rejected)
    error = paid.discover("anthropic", "bad")["error"]
    assert error == "Could not list models: the API key was rejected (HTTP 401) — update it in the settings below."


def test_openrouter_models_say_whether_they_take_reasoning(or_payload):
    found = {m["id"]: m for m in openrouter.qualifying(or_payload, today=TODAY)["models"]}
    assert found["dots-studio/dots-3-note-preview:free"]["reasoning"] is True
