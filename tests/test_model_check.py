"""
The launch-time model check (`llm_agent/agent/model_check.py`).

Hosted providers retire models; a saved `.env` keeps naming the old one. These
tests pin what the check promises:

  - a model the key cannot use blocks starting, with alternatives offered;
  - a rejected key blocks starting;
  - not being able to ask (network, a server without a model list) never
    blocks — it costs a warning, not the ability to run;
  - a newer model of the same family is reported, never applied, and a fresh
    dated snapshot of the configured model is not mistaken for a newer model;
  - the check itself can never raise into the launch screen.

No test contacts a provider: every listing call is stubbed.
"""

import pathlib
import re
import types

import pytest

from llm_agent.agent import model_check as mc

ROOT = pathlib.Path(__file__).resolve().parents[1]


class TestNaming:
    @pytest.mark.parametrize("model,expected", [
        ("gpt-5.6-luna", "luna"),
        ("gpt-6-luna", "luna"),
        ("gpt-6-luna-2026-09-22", "luna"),
        ("claude-haiku-4-5", "haiku"),
        ("claude-haiku-4-5-20251001", "haiku"),
        ("claude-haiku-5-5", "haiku"),
        ("gemini-3.5-flash-lite", "flash-lite"),
        ("models/gemini-3.5-flash-lite", "flash-lite"),
        ("gpt-4o", ""),
    ])
    def test_family(self, model, expected):
        assert mc.family(model) == expected

    @pytest.mark.parametrize("model,expected", [
        ("gpt-6-luna-2026-09-22", "gpt-6-luna"),
        ("claude-haiku-4-5-20251001", "claude-haiku-4-5"),
        ("models/gemini-3.5-flash", "gemini-3.5-flash"),
        ("gpt-6-luna", "gpt-6-luna"),
    ])
    def test_base_id(self, model, expected):
        assert mc.base_id(model) == expected


class TestNewerInFamily:
    def test_newer_model_of_the_same_family(self):
        catalog = {"gpt-5.6-luna": 100, "gpt-6-luna": 200, "gpt-6.1-sol": 300}
        assert mc.newer_in_family("gpt-5.6-luna", catalog) == "gpt-6-luna"

    def test_already_newest(self):
        assert mc.newer_in_family("gpt-6-luna", {"gpt-5.6-luna": 100, "gpt-6-luna": 200}) is None

    def test_a_fresh_snapshot_of_the_same_model_is_not_newer(self):
        catalog = {"gpt-6-luna": 200, "gpt-6-luna-2026-10-01": 250}
        assert mc.newer_in_family("gpt-6-luna", catalog) is None

    def test_alias_matches_its_dated_snapshot(self):
        # Anthropic lists the dated id; the configured value is the alias.
        catalog = {"claude-haiku-4-5-20251001": 1, "claude-haiku-5-5": 2, "claude-opus-5-5": 3}
        assert mc.newer_in_family("claude-haiku-4-5", catalog) == "claude-haiku-5-5"

    def test_unknown_dates_say_nothing(self):
        assert mc.newer_in_family("gpt-5.6-luna", {"gpt-6-luna": None}) is None

    def test_a_model_without_a_family_says_nothing(self):
        assert mc.newer_in_family("gpt-4o", {"gpt-4o": 1, "gpt-5o": 2}) is None


class TestAlternatives:
    def test_same_family_first_and_undated(self):
        ids = ["gpt-6.1-sol", "gpt-6-luna-2026-09-22", "gpt-6-luna", "gpt-5-nano"]
        assert mc.alternatives("gpt-5.6-luna", ids) == ("gpt-6-luna", "gpt-5-nano", "gpt-6.1-sol")

    def test_newest_of_the_family_first_when_dates_are_known(self):
        ids = ["gpt-5.6-luna", "gpt-6-luna", "gpt-6.1-sol"]
        created = {"gpt-5.6-luna": 1, "gpt-6-luna": 2, "gpt-6.1-sol": 3}
        assert mc.alternatives("gpt-99-luna", ids, created)[:2] == ("gpt-6-luna", "gpt-5.6-luna")

    def test_bounded(self):
        ids = [f"model-{i}" for i in range(50)]
        assert len(mc.alternatives("x", ids)) == 12


# ---------------------------------------------------------------------------


def _probe(monkeypatch, result):
    from llm_agent.agent.providers import openai as openai_mod

    monkeypatch.setattr(openai_mod, "probe", lambda base_url=None, api_key=None: result)


class TestOpenAI:
    def test_offered_and_newest(self, monkeypatch):
        _probe(monkeypatch, {"reachable": True, "error": "", "models": ["gpt-6-luna"],
                             "created": {"gpt-6-luna": 2}})
        result = mc.check_openai("gpt-6-luna")
        assert result.status == mc.OK and result.newer is None and not result.blocks_start

    def test_offered_with_a_newer_one(self, monkeypatch):
        _probe(monkeypatch, {"reachable": True, "error": "",
                             "models": ["gpt-5.6-luna", "gpt-6-luna"],
                             "created": {"gpt-5.6-luna": 1, "gpt-6-luna": 2}})
        result = mc.check_openai("gpt-5.6-luna")
        assert result.status == mc.OK and result.newer == "gpt-6-luna"

    def test_retired_blocks_and_offers_alternatives(self, monkeypatch):
        _probe(monkeypatch, {"reachable": True, "error": "",
                             "models": ["gpt-6-luna", "gpt-6.1-sol"], "created": {}})
        result = mc.check_openai("gpt-5.6-luna")
        assert result.status == mc.MISSING and result.blocks_start
        assert result.alternatives[0] == "gpt-6-luna"

    def test_non_chat_models_are_not_offered(self, monkeypatch):
        _probe(monkeypatch, {"reachable": True, "error": "", "created": {},
                             "models": ["gpt-6-luna", "babbage-002", "chat-latest",
                                        "text-embedding-3-small", "gpt-6-tts", "whisper-1"]})
        assert mc.check_openai("gpt-99-luna").alternatives == ("gpt-6-luna",)

    def test_rejected_key_blocks(self, monkeypatch):
        _probe(monkeypatch, {"reachable": False, "models": [], "created": {},
                             "error": "The API key was rejected (HTTP 401)."})
        assert mc.check_openai("gpt-6-luna").status == mc.KEY_REJECTED

    def test_unreachable_does_not_block(self, monkeypatch):
        _probe(monkeypatch, {"reachable": False, "models": [], "created": {},
                             "error": "Cannot reach https://api.openai.com/v1: timeout"})
        result = mc.check_openai("gpt-6-luna")
        assert result.status == mc.UNVERIFIED and not result.blocks_start

    def test_a_compatible_server_without_a_model_list_does_not_block(self, monkeypatch):
        _probe(monkeypatch, {"reachable": True, "error": "", "models": [], "created": {}})
        result = mc.check_openai("kimi-k3", base_url="https://compatible.test/v1")
        assert result.status == mc.UNVERIFIED and "compatible.test" in result.message

    def test_probe_reports_creation_times(self, monkeypatch):
        import httpx

        from llm_agent.agent.providers.openai import probe

        def fake_get(url, headers=None, timeout=None):
            return httpx.Response(200, json={"data": [{"id": "gpt-6-luna", "created": 7}]},
                                  request=httpx.Request("GET", url))

        monkeypatch.setattr(httpx, "get", fake_get)
        out = probe(base_url="https://api.openai.com/v1", api_key="sk-test")
        assert out["models"] == ["gpt-6-luna"] and out["created"] == {"gpt-6-luna": 7}


# ---------------------------------------------------------------------------


def _anthropic_error(cls, status):
    # The 1.x SDK is built on httpx2; older ones on httpx.
    try:
        import httpx2 as http
    except ImportError:
        import httpx as http

    request = http.Request("GET", "https://api.anthropic.com/v1/models/x")
    return cls("err", response=http.Response(status, request=request), body=None)


class _FakeModels:
    def __init__(self, found=True, error=None, listing=()):
        self.found, self.error, self.listing = found, error, listing

    def retrieve(self, model):
        import anthropic

        if self.error is not None:
            raise self.error
        if not self.found:
            raise _anthropic_error(anthropic.NotFoundError, 404)
        return types.SimpleNamespace(id=model)

    def list(self, limit=100):
        return [types.SimpleNamespace(id=i, created_at=t) for i, t in self.listing]


def _anthropic(monkeypatch, models):
    from llm_agent.agent.providers.claude import AnthropicProvider

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setattr(AnthropicProvider, "client",
                        property(lambda self: types.SimpleNamespace(models=models)))


class TestAnthropic:
    def test_alias_found_with_a_newer_haiku(self, monkeypatch):
        _anthropic(monkeypatch, _FakeModels(listing=[
            ("claude-haiku-4-5-20251001", 1), ("claude-haiku-5-5", 2), ("claude-opus-5-5", 3)]))
        result = mc.check_anthropic("claude-haiku-4-5")
        assert result.status == mc.OK and result.newer == "claude-haiku-5-5"

    def test_retired_blocks(self, monkeypatch):
        _anthropic(monkeypatch, _FakeModels(found=False, listing=[("claude-haiku-5-5", 2)]))
        result = mc.check_anthropic("claude-haiku-3")
        assert result.status == mc.MISSING and result.alternatives == ("claude-haiku-5-5",)

    def test_rejected_key_blocks(self, monkeypatch):
        import anthropic

        _anthropic(monkeypatch, _FakeModels(
            error=_anthropic_error(anthropic.AuthenticationError, 401)))
        assert mc.check_anthropic("claude-haiku-5-5").status == mc.KEY_REJECTED

    def test_connection_trouble_does_not_block(self, monkeypatch):
        _anthropic(monkeypatch, _FakeModels(error=TimeoutError("slow")))
        assert mc.check_anthropic("claude-haiku-5-5").status == mc.UNVERIFIED

    def test_no_key(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        assert mc.check_anthropic("claude-haiku-5-5", api_key="").status == mc.KEY_REJECTED


# ---------------------------------------------------------------------------


class _FakeGenaiModels:
    def __init__(self, error=None, listing=()):
        self.error, self.listing = error, listing

    def get(self, model):
        if self.error is not None:
            raise self.error
        return types.SimpleNamespace(name=f"models/{model}")

    def list(self):
        return [types.SimpleNamespace(name=n, supported_actions=a) for n, a in self.listing]


def _gemini(monkeypatch, models):
    from google import genai

    monkeypatch.setattr(genai, "Client", lambda api_key=None, **kw: types.SimpleNamespace(models=models))


class TestGemini:
    def test_found(self, monkeypatch):
        _gemini(monkeypatch, _FakeGenaiModels())
        assert mc.check_gemini("gemini-3.5-flash-lite", api_key="k").status == mc.OK

    def test_retired_offers_only_chat_models(self, monkeypatch):
        from google.genai import errors

        _gemini(monkeypatch, _FakeGenaiModels(
            error=errors.ClientError(404, {"error": {"message": "not found"}}),
            listing=[("models/gemini-4-flash-lite", ["generateContent"]),
                     ("models/text-embedding-5", ["embedContent"])]))
        result = mc.check_gemini("gemini-3.5-flash-lite", api_key="k")
        assert result.status == mc.MISSING and result.alternatives == ("gemini-4-flash-lite",)

    def test_rejected_key(self, monkeypatch):
        from google.genai import errors

        _gemini(monkeypatch, _FakeGenaiModels(
            error=errors.ClientError(400, {"error": {"message": "API key not valid."}})))
        assert mc.check_gemini("gemini-3.5-flash-lite", api_key="bad").status == mc.KEY_REJECTED


# ---------------------------------------------------------------------------


class TestDispatch:
    def test_ollama_is_not_checked_here(self):
        assert mc.check("ollama") is None

    def test_never_raises(self, monkeypatch):
        def broken(*a, **k):
            raise RuntimeError("boom")

        monkeypatch.setitem(mc._CHECKS, "openai", broken)
        result = mc.check("openai", "gpt-6-luna")
        assert result.status == mc.UNVERIFIED and not result.blocks_start


class TestDefaultsStayInSync:
    """The default model is written in config.py and .env.example; the two
    drifting apart is how a stale default survives an update."""

    @pytest.mark.parametrize("var", ["OPENAI_MODEL", "ANTHROPIC_MODEL", "GEMINI_MODEL"])
    def test_config_matches_env_example(self, var):
        config = (ROOT / "llm_agent/agent/config.py").read_text(encoding="utf-8")
        example = (ROOT / "llm_agent/.env.example").read_text(encoding="utf-8")
        default = re.search(rf'os\.environ\.get\("{var}", "([^"]+)"\)', config).group(1)
        assert re.search(rf"^#?\s*{var}=(\S+)", example, re.MULTILINE).group(1) == default

    def test_current_defaults(self):
        config = (ROOT / "llm_agent/agent/config.py").read_text(encoding="utf-8")
        assert 'get("OPENAI_MODEL", "gpt-6-luna")' in config
        assert 'get("ANTHROPIC_MODEL", "claude-haiku-5-5")' in config
