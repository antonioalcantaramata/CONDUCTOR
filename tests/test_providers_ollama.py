"""Ollama provider: preflight, the context guard, and message conversion.

The guard is the important part. Ollama silently truncates the *front* of an
over-long prompt — dropping the system prompt first — so a prompt that doesn't
fit produces a confident, uninstructed model rather than an error. Every test
here that asserts a refusal is protecting against that.

No test contacts a real server: hosts point at a closed port, or httpx is
monkeypatched.
"""

import json

import pytest

from llm_agent.agent.providers.ollama import OllamaProvider, probe

CLOSED = "http://localhost:59999"


@pytest.fixture
def provider(monkeypatch):
    """A provider with fixed settings, insulated from the developer's .env."""
    for var in (
        "OLLAMA_MODEL", "OLLAMA_HOST", "OLLAMA_NUM_CTX",
        "OLLAMA_OUTPUT_RESERVE", "OLLAMA_KEEP_ALIVE", "OLLAMA_THINK",
        "OLLAMA_TIMEOUT_S",
    ):
        monkeypatch.delenv(var, raising=False)
    p = OllamaProvider(model="test-model", host=CLOSED)
    p._preflighted = True
    return p


class TestSettingsAreLive:
    def test_reads_settings_from_the_environment(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_MODEL", "m2")
        monkeypatch.setenv("OLLAMA_NUM_CTX", "16384")
        monkeypatch.setenv("OLLAMA_KEEP_ALIVE", "5m")
        monkeypatch.setenv("OLLAMA_THINK", "false")
        monkeypatch.setenv("OLLAMA_TIMEOUT_S", "42")
        p = OllamaProvider()
        assert (p.model, p.num_ctx, p.keep_alive, p.think, p.timeout_s) == (
            "m2", 16384, "5m", False, 42.0
        )

    def test_explicit_arguments_beat_the_environment(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_MODEL", "from-env")
        assert OllamaProvider(model="explicit").model == "explicit"

    @pytest.mark.parametrize("raw,expected", [("true", True), ("TRUE", True),
                                              ("false", False), ("False", False)])
    def test_think_flag_parsing(self, monkeypatch, raw, expected):
        monkeypatch.setenv("OLLAMA_THINK", raw)
        assert OllamaProvider().think is expected


class TestContextGuard:
    def _payload(self, tokens):
        # ~4 bytes per token is the estimator's assumption.
        return {"filler": "x" * (tokens * 4)}

    def test_allows_a_prompt_that_fits(self, provider):
        provider.num_ctx = 32768
        provider.output_reserve = 4096
        provider._check_fits(self._payload(1000))  # must not raise

    def test_refuses_a_prompt_that_would_be_truncated(self, provider):
        provider.num_ctx = 4096  # Ollama's dangerous default
        provider.output_reserve = 1024
        with pytest.raises(RuntimeError) as exc:
            provider._check_fits(self._payload(25_000))
        assert "silently discard" in str(exc.value)

    def test_refusal_names_the_setting_to_change(self, provider):
        provider.num_ctx = 4096
        with pytest.raises(RuntimeError, match="OLLAMA_NUM_CTX"):
            provider._check_fits(self._payload(25_000))

    def test_budget_accounts_for_the_output_reserve(self, provider):
        provider.num_ctx = 10_000
        provider.output_reserve = 9_000
        # Only ~1000 tokens of input are permitted despite a 10k window.
        with pytest.raises(RuntimeError):
            provider._check_fits(self._payload(5_000))

    def test_reserve_is_reported_in_the_message(self, provider):
        provider.num_ctx = 8192
        provider.output_reserve = 2048
        with pytest.raises(RuntimeError, match="2,048 reserved"):
            provider._check_fits(self._payload(25_000))

    def test_the_real_prompt_fits_the_default_window(self):
        """The shipped defaults must accommodate the actual system prompt."""
        from llm_agent.agent.providers.schema import to_json_schema_tools
        from llm_agent.agent.system_prompt import build_system_prompt
        from llm_agent.agent.config import DEFAULT_GRID_CONSTANTS, OLLAMA_NUM_CTX

        p = OllamaProvider(model="m", host=CLOSED)
        p.num_ctx = OLLAMA_NUM_CTX
        payload = {
            "messages": p._to_ollama_messages(
                [{"role": "user", "content": "run rsa"}],
                build_system_prompt(DEFAULT_GRID_CONSTANTS),
            ),
            "tools": to_json_schema_tools(__import__(
                "llm_agent.agent.tool_schemas", fromlist=["TOOLS"]).TOOLS),
        }
        p._check_fits(payload)  # must not raise with shipped defaults


class TestPreflight:
    def test_unreachable_server_explains_how_to_start_it(self):
        p = OllamaProvider(model="m", host=CLOSED)
        with pytest.raises(RuntimeError, match="ollama serve"):
            p.preflight()

    def test_missing_model_names_the_pull_command(self, monkeypatch):
        p = OllamaProvider(model="absent-model", host=CLOSED)
        monkeypatch.setattr(p, "_installed_models", lambda: {"other-model"})
        monkeypatch.setattr(
            "llm_agent.agent.providers.ollama.httpx.get",
            lambda *a, **k: _FakeResponse(200, {"version": "1"}),
        )
        with pytest.raises(RuntimeError, match="ollama pull absent-model"):
            p.preflight()

    def test_model_without_tool_support_is_rejected(self, monkeypatch):
        p = OllamaProvider(model="m", host=CLOSED)
        monkeypatch.setattr(
            "llm_agent.agent.providers.ollama.httpx.get",
            lambda *a, **k: _FakeResponse(200, {"version": "1"}),
        )
        monkeypatch.setattr(p, "_installed_models", lambda: {"m"})
        monkeypatch.setattr(p, "_capabilities", lambda: ["completion"])
        with pytest.raises(RuntimeError, match="does not support tool calling"):
            p.preflight()

    def test_unknown_capabilities_do_not_block(self, monkeypatch):
        # Older Ollama builds omit the field; refusing then would be wrong.
        p = OllamaProvider(model="m", host=CLOSED)
        monkeypatch.setattr(
            "llm_agent.agent.providers.ollama.httpx.get",
            lambda *a, **k: _FakeResponse(200, {"version": "1"}),
        )
        monkeypatch.setattr(p, "_installed_models", lambda: {"m"})
        monkeypatch.setattr(p, "_capabilities", lambda: None)
        p.preflight()
        assert p._preflighted

    def test_failed_model_query_is_distinct_from_no_models(self):
        """None (query failed) must not be read as 'nothing installed'."""
        p = OllamaProvider(model="m", host=CLOSED)
        assert p._installed_models() is None
        assert p._capabilities() is None

    def test_bare_name_matches_the_latest_tag(self):
        p = OllamaProvider(model="m", host=CLOSED)
        assert p._model_installed({"m:latest"})


class TestMessageConversion:
    def test_system_prompt_leads_the_message_list(self, provider):
        out = provider._to_ollama_messages([], "SYSTEM")
        assert out[0] == {"role": "system", "content": "SYSTEM"}

    def test_roles_map_across(self, provider):
        out = provider._to_ollama_messages([
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a"},
            {"role": "tool", "name": "t", "content": {}},
        ], "sys")
        assert [m["role"] for m in out] == ["system", "user", "assistant", "tool"]

    def test_tool_calls_use_the_ollama_function_shape(self, provider):
        out = provider._to_ollama_messages([
            {"role": "assistant", "content": "",
             "tool_calls": [{"name": "run_rsa", "args": {"f": 1.0}}]},
        ], "sys")
        assert out[1]["tool_calls"] == [
            {"function": {"name": "run_rsa", "arguments": {"f": 1.0}}}
        ]

    def test_tool_results_are_serialised_to_text(self, provider):
        out = provider._to_ollama_messages(
            [{"role": "tool", "name": "t", "content": {"violations": 0}}], "sys"
        )
        assert json.loads(out[1]["content"]) == {"violations": 0}

    def test_gemini_only_fields_are_not_leaked(self, provider):
        out = provider._to_ollama_messages([
            {"role": "assistant", "content": "",
             "tool_calls": [{"name": "a", "args": {}, "signature": "sig", "id": "i"}]},
        ], "sys")
        assert "signature" not in json.dumps(out)


class TestResponseParsing:
    def test_parses_prose(self, provider):
        r = provider._to_model_response({"message": {"content": "  Done.  "}})
        assert r.text == "Done." and not r.wants_tools

    def test_parses_tool_calls(self, provider):
        r = provider._to_model_response({"message": {"tool_calls": [
            {"function": {"name": "run_rsa", "arguments": {"a": 1}}}
        ]}})
        assert r.tool_calls[0].name == "run_rsa"
        assert r.tool_calls[0].args == {"a": 1}

    def test_accepts_arguments_as_a_json_string(self, provider):
        r = provider._to_model_response({"message": {"tool_calls": [
            {"function": {"name": "t", "arguments": '{"a": 1}'}}
        ]}})
        assert r.tool_calls[0].args == {"a": 1}

    def test_unparseable_arguments_degrade_to_empty(self, provider):
        r = provider._to_model_response({"message": {"tool_calls": [
            {"function": {"name": "t", "arguments": "not json"}}
        ]}})
        assert r.tool_calls[0].args == {}

    def test_nameless_tool_calls_are_dropped(self, provider):
        r = provider._to_model_response({"message": {"tool_calls": [{"function": {}}]}})
        assert r.tool_calls == []

    def test_empty_body_is_safe(self, provider):
        assert provider._to_model_response({}).text == ""

    def test_reasoning_is_used_when_content_is_empty(self, provider):
        # Otherwise the loop reads a blank turn as a finished answer and the
        # user is shown nothing at all.
        r = provider._to_model_response(
            {"message": {"content": "", "thinking": "The answer is 42."}}
        )
        assert r.text == "The answer is 42."

    def test_content_wins_over_reasoning(self, provider):
        r = provider._to_model_response(
            {"message": {"content": "Real.", "thinking": "noise"}}
        )
        assert r.text == "Real."

    def test_reasoning_is_not_substituted_when_tools_were_called(self, provider):
        r = provider._to_model_response({"message": {
            "content": "", "thinking": "noise",
            "tool_calls": [{"function": {"name": "t", "arguments": {}}}],
        }})
        assert r.text == "" and r.wants_tools


class TestProbe:
    def test_unreachable_host_reports_cleanly(self):
        info = probe(host=CLOSED)
        assert info == {"reachable": False, "version": "", "models": []}


class _FakeResponse:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload
        self.text = json.dumps(payload)

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")
