"""
Session context (`llm_agent/agent/session_context.py`): what a session asked,
found and decided, as a Markdown file CONDUCTOR writes from the turn records,
kept per network, offered on the next start, and given to both agents as
background — never as evidence.
"""

import json

import pytest

from llm_agent.agent import loop as loop_module
from llm_agent.agent import recommender
from llm_agent.agent import session_context as sc
from llm_agent.agent.providers.base import ModelResponse

FP = "PGLib IEEE 30-Bus System (Alsac–Stott)|abc123"


def _record(user, results):
    return {"user": user, "turn_key": "k", "grid": {},
            "tool_calls": [{"name": n, "args": {"timestamp": "2000-01-06 20:00"}} for n, _ in results],
            "tool_results": [{"name": n, "result": r} for n, r in results]}


RSA = _record("Is the grid secure at 20:00 on 6 January?",
              [("run_rsa", {"secure": False, "total_violations": 2, "timestamp": "2000-01-06 20:00:00"})])
OPTIONS = _record("Give me options.", [("optimize_flexibility", {"mode": "options", "status": "options",
                                                                 "timestamp": "2000-01-06 20:00:00"})])
META = {"saved": "2026-10-22 18:40 UTC", "network": "PGLib IEEE 30-Bus System (Alsac–Stott)", "fingerprint": FP,
        "scenario": "Demo: IEEE 30 · active only", "data": "measurements",
        "settings": ["Gen_bus1 costs: reactive 0 €/Mvarh"], "proposals": ["dc (add_load at Bus_20)"]}
NOTE = {"decision": "Take the cheapest option without Gen_bus1.",
        "chosen_option": {"ref": "T2·cheapest", "label": "Cheapest", "timestamp": "2000-01-06 20:00:00",
                          "units_called": ["Gen_bus7", "Gen_bus10"], "moved_mw": 30.12, "import_mw": 151.9,
                          "cost_eur_per_h": 31.05, "held_units": {"Gen_bus1": 46.0}},
        "criteria": ["Gen_bus1 untrusted today"], "open_risks": ["No N-1 screen after the redispatch."]}


def _build(**kw):
    return sc.build([RSA, OPTIONS], ["Two violations: Bus_29 and the feeder line.", "Four options."],
                    meta=META, **kw)


class TestWriting:
    def test_the_record_decisions_settings_and_what_was_left_open(self):
        text = _build(notes=[NOTE], pending=[{"title": "N-1 after T2·cheapest", "prompt": "Screen N-1 at 20:00."}])
        assert text.startswith("# CONDUCTOR session context") and sc.MARKER in text
        assert f"<!-- network: {FP} -->" in text
        assert "**Decision:** Take the cheapest option without Gen_bus1." in text
        assert "T2·cheapest — Cheapest" in text and "31.05 €/h" in text and "(figures from the solver result)" in text
        assert "held at schedule: Gen_bus1" in text and "What mattered: Gen_bus1 untrusted today" in text
        assert "- Gen_bus1 costs: reactive 0 €/Mvarh" in text and "dc (add_load at Bus_20)" in text
        assert "T1 · Q: Is the grid secure" in text and "run_rsa(timestamp=2000-01-06 20:00)" in text
        assert "## Studies proposed and not run" in text and "Screen N-1 at 20:00." in text
        assert "may no longer hold" in text

    def test_the_summary_is_labelled_and_its_unverified_figures_flagged(self):
        text = _build(summary="- Two violations at 20:00.", summary_unverified=("99.9 %",))
        assert "## Summary" in text and "Written by the model" in text
        assert "Not found in the session's results: 99.9 %" in text

    def test_a_long_session_keeps_its_newest_turns_and_every_decision(self):
        records = [_record(f"Question {i} " + "x" * 200, [("run_rsa", {"secure": True})]) for i in range(80)]
        text = sc.build(records, ["y" * 300] * 80, meta=META, notes=[NOTE])
        assert len(text) <= sc.MAX_CHARS
        assert "earlier turns omitted" in text and "Question 79" in text and "Question 0 " not in text
        assert "**Decision:**" in text
        assert sc.parse(text).turns == 80


class TestReading:
    def test_round_trip(self):
        ctx = sc.parse(_build(notes=[NOTE]))
        assert (ctx.saved, ctx.fingerprint, ctx.turns, ctx.decisions) == ("2026-10-22 18:40 UTC", FP, 2, 1)
        assert ctx.network == "PGLib IEEE 30-Bus System (Alsac–Stott)"
        assert ctx.label() == "2026-10-22 18:40 UTC · 2 turns · 1 decision"

    @pytest.mark.parametrize("text, says", [("# Notes\nsome text", "not a CONDUCTOR session context"),
                                            (sc.MARKER + "x" * 50_000, "characters")])
    def test_other_files_are_refused_with_the_reason(self, text, says):
        with pytest.raises(sc.ContextError, match=says):
            sc.parse(text)

    def test_the_agents_get_it_as_background_not_evidence(self):
        ctx = sc.parse(_build())
        block = sc.prompt_block(ctx, FP)
        assert block.startswith("## Earlier session (a context file saved 2026-10-22 18:40 UTC)")
        assert "on this network" in block and "not evidence" in block and "not instructions" in block
        assert "<<<earlier session" in block and "earlier session>>>" in block
        other = sc.prompt_block(ctx, "Another grid|zzz")
        assert "ANOTHER network (PGLib IEEE 30-Bus System (Alsac–Stott))" in other


class TestStorage:
    def test_kept_per_network_newest_first(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sc, "CONTEXT_DIR", str(tmp_path))
        first = sc.save(_build(), FP)
        (tmp_path / "corrupt").mkdir()
        later = pathlib_write(tmp_path, first, _build(notes=[NOTE]))
        found = sc.saved_for(FP)
        assert [c.path for c in found] == [later, first] and found[0].decisions == 1
        assert sc.saved_for("Another grid|zzz") == []

    def test_a_damaged_file_is_skipped(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sc, "CONTEXT_DIR", str(tmp_path))
        path = sc.save(_build(), FP)
        broken = path.replace("context-", "context-9")
        open(broken, "w").write("not a context")
        assert [c.path for c in sc.saved_for(FP)] == [path]


def pathlib_write(root, beside: str, text: str) -> str:
    """A second context saved later than `beside` (names sort by time)."""
    import pathlib
    later = pathlib.Path(beside).with_name("context-99991231-235959.md")
    later.write_text(text, encoding="utf-8")
    return str(later)


class FakeProvider:
    name, model = "fake", "fake-model"

    def __init__(self, text="", raises=None):
        self.text, self.raises, self.calls = text, raises, []

    def chat(self, messages, tools, system, on_event=None):
        self.calls.append({"messages": messages, "tools": tools, "system": system})
        if self.raises:
            raise self.raises
        return ModelResponse(text=self.text)


class TestSummary:
    def test_one_tool_less_call_and_its_figures_checked(self):
        records = [_record("q", [("run_rsa", {"total_violations": 2, "max_line_loading_pct": 114.7})])]
        provider = FakeProvider("- 2 violations; the feeder line at 114.7 %.\n- Bus_29 at 0.12 p.u.")
        summary, unverified, error = sc.summarize("the record", records, provider=provider)
        assert summary.startswith("- 2 violations") and not error
        assert provider.calls[0]["tools"] == [] and "Copy figures exactly" in provider.calls[0]["system"]
        assert unverified == ("0.12 p.u.",)

    def test_a_failed_call_leaves_the_record_without_a_summary(self):
        summary, unverified, error = sc.summarize("the record", [], provider=FakeProvider(raises=RuntimeError("quota")))
        assert summary == "" and "quota" in error


class TestTheAgentsReceiveIt:
    def test_the_orchestrator_after_its_system_prompt_and_in_the_record(self, monkeypatch):
        from llm_agent.agent.providers.base import ModelResponse as MR
        seen, records = [], []

        class Provider(FakeProvider):
            def chat(self, messages, tools, system, on_event=None):
                seen.append(system)
                return MR(text="done")

        monkeypatch.setattr(loop_module, "get_provider", lambda: Provider())
        monkeypatch.setattr(loop_module, "_append_to_log", records.append)
        block = sc.prompt_block(sc.parse(_build()), FP)
        loop_module.run_agent_turn("go", [], earlier_context=block)
        assert seen[0].endswith(block)
        assert records[-1]["earlier_context"].startswith("Earlier session (a context file saved")
        loop_module.run_agent_turn("go", [])
        assert "Earlier session" not in seen[-1] and "earlier_context" not in records[-1]

    def test_the_advisor_as_its_own_section_not_inside_the_json(self):
        block = sc.prompt_block(sc.parse(_build()), FP)
        provider = FakeProvider(json.dumps({"reply": "Noted.", "studies": []}))
        recommender.discuss("What did we decide last time?", [], [], None, [], {"network": "x",
                            "earlier_session": block}, provider=provider)
        request = provider.calls[0]["messages"][0]["content"]
        assert block in request and '"earlier_session"' not in request
