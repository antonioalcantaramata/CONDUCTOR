"""
The chat app's flows, driven headlessly: the advisor panel, the discussion,
the waiting states, interrupted work, collapsing the panel, and saving a turn.

These broke more than once while they were built, in ways only the running
app shows. Each test runs the real app (Streamlit's AppTest) against the real
backend, started in-process on a free port with its storage in a temporary
folder. The models are scripted: no model credit is spent. Skipped where
Streamlit is not installed (CI).
"""

import pathlib
import socket
import sys
import threading
import time

import pytest

pytest.importorskip("streamlit.testing.v1")
pytest.importorskip("uvicorn")

ROOT = pathlib.Path(__file__).resolve().parents[1]
APP = str(ROOT / "llm_agent" / "app.py")
REC = {"user": "q", "turn_key": "k", "grid": {}, "tool_calls": [], "tool_results": [], "assistant": "a"}


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def backend_url(tmp_path_factory):
    """The real backend on a free port, its uploads and proposals in tmp."""
    import uvicorn

    import main_backend as mb

    store = tmp_path_factory.mktemp("app_flows")
    patch = pytest.MonkeyPatch()
    patch.setenv("GRID_PROFILE", "pglib_case14")
    patch.setattr(mb, "SYSTEMS_DIR", str(store / "systems"))
    patch.setattr(mb, "DATA_FILES_DIR", str(store / "data_files"))
    for name in ("_LAST_UPLOAD_SENTINEL", "_LAST_TIMESERIES_SENTINEL", "_LAST_FORECAST_SENTINEL",
                 "_PROPOSALS_FILE"):
        patch.setattr(mb, name, str(store / f"{name}.json"))
    saved = dict(mb.app_data)
    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(mb.app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    import urllib.request
    for _ in range(200):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1)
            break
        except OSError:
            time.sleep(0.1)
    url = f"http://127.0.0.1:{port}"
    yield url
    server.should_exit = True
    thread.join(timeout=10)
    mb.app_data.clear()
    mb.app_data.update(saved)
    patch.undo()


@pytest.fixture
def agent(backend_url, monkeypatch):
    """The app's agent modules, pointed at the test backend and scripted."""
    monkeypatch.setenv("DT_BACKEND_URL", backend_url)
    sys.path.insert(0, str(ROOT / "llm_agent"))
    import agent.config as config
    import agent.loop as loop
    import agent.recommender as recommender
    import agent.tools as tools
    for module in (config, tools):
        monkeypatch.setattr(module, "BASE_URL", backend_url, raising=False)
    monkeypatch.setattr(loop, "log_event", lambda record: None)
    monkeypatch.setattr(loop, "_write_record", lambda record: None)
    yield {"loop": loop, "recommender": recommender}
    if str(ROOT / "llm_agent") in sys.path:
        sys.path.remove(str(ROOT / "llm_agent"))


def _app(attention=(), extra=None, messages=None):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(APP, default_timeout=120)
    at.session_state["_launched"] = True
    at.session_state["history"] = []
    at.session_state["auto_suggest"] = False
    at.session_state["agent_panel_open"] = True          # it starts closed; these tests use it
    at.session_state["messages"] = messages if messages is not None else [
        {"role": "user", "text": "Is the system secure?", "charts": None},
        {"role": "assistant", "text": "a", "charts": None, "trace": None, "record": REC,
         "attention": list(attention)}]
    for k, v in (extra or {}).items():
        at.session_state[k] = v
    at.run()
    return at


def _button(at, label=None, key_prefix=None):
    for b in at.button:
        if (label is not None and b.label == label) or (key_prefix and str(b.key or "").startswith(key_prefix)):
            return b
    raise AssertionError(f"no button {label or key_prefix}: {[b.label for b in at.button]}")


def _mode(at, value):
    [s for s in at.get("segmented_control") if s.label == "Advisor"][0].set_value(value).run()


def _ask(at, text, k=0):
    at.text_input(key=f"adv_q_{k}").input(text)
    _button(at, "Discuss").click().run()


def _send(at, text):
    box = [t for t in at.text_input if str(t.key).startswith("chat_box_")][0]
    box.input(text)
    _button(at, "Run").click().run()


def _busy_text(at):
    return [m.value for m in at.markdown if "class='conductor-busy" in m.value]


def _script(agent, monkeypatch, *, freeze=None, reply="Reply.", studies=()):
    """Scripted models; `freeze` names calls that stop the page as they start
    (as a click elsewhere would), so the screen at that moment can be read."""
    import streamlit as st
    R, L = agent["recommender"], agent["loop"]
    freeze = freeze if freeze is not None else {}
    calls = {"turn": [], "discuss": [], "suggest": 0, "note": 0}

    def stop_if(kind):
        if freeze.get(kind, 0) > 0:
            freeze[kind] -= 1
            st.stop()

    def turn(user_message, history, conversation_id=None, on_event=None, reflection=None):
        calls["turn"].append(user_message)
        stop_if("turn")
        return ("Answer to: " + user_message, list(history))

    def discuss(q, thread, messages, latest, records, context, provider=None):
        calls["discuss"].append({"q": q, "sent": [t.get("answered_turn") for t in thread if t["role"] == "sent"]})
        stop_if("discuss")
        return R.AdvisorReply(reply, tuple(R.Study(t, f"Run {t}.", "why") for t in studies))

    def suggest(*a, **k):
        calls["suggest"] += 1
        stop_if("suggest")
        return R.SuggestionResult((R.Suggestion("stress", "Scan the week", "Compare over the week.", "w"),),
                                  comment="**One** moment only.")

    def note(*a, **k):
        calls["note"] += 1
        return R.DecisionNote("Do not approve yet.", ("Evidence (turn 2).",), ("Risk.",), ("scan (turn 2)",))

    monkeypatch.setattr(L, "run_agent_turn", turn)
    monkeypatch.setattr(R, "discuss", discuss)
    monkeypatch.setattr(R, "suggest", suggest)
    monkeypatch.setattr(R, "decision_note", note)
    return calls


def test_the_panel_suggestions(agent, monkeypatch):
    calls = _script(agent, monkeypatch)
    at = _app(attention=["run_rsa: 1 violation(s)"])
    assert any("advisor (right) can help" in c.value for c in at.caption)
    assert [s.value for s in at.get("segmented_control") if s.label == "Advisor"] == ["suggest"]
    _button(at, "✨ Suggest next steps").click().run()
    assert calls["suggest"] == 1
    assert any("Suggestions" in e.label for e in at.expander)
    assert _button(at, "✨ More ideas")
    assert any(m.value == "**One** moment only." for m in at.markdown)        # rendered as Markdown
    assert not at.exception


def test_a_discussion_from_question_to_decision(agent, monkeypatch):
    calls = _script(agent, monkeypatch, studies=("N-1 now",))
    at = _app()
    _mode(at, "discuss")
    _ask(at, "Would you trust this result?")
    assert any("conductor-you'>Would you trust this result?" in m.value for m in at.markdown)
    _button(at, key_prefix="adv_use_").click().run()
    assert at.session_state["_box_value"] == "Run N-1 now."
    assert any("· not sent yet" in m.value for m in at.markdown if "conductor-sent" in m.value)
    _send(at, "Run N-1 now.")
    assert calls["turn"] == ["Run N-1 now."]
    texts = [(m["role"], m["text"]) for m in at.session_state["messages"]]
    assert texts[-2:] == [("user", "Run N-1 now."), ("assistant", "Answer to: Run N-1 now.")]
    assert any("answered in turn 2" in m.value for m in at.markdown if "conductor-sent" in m.value)
    _ask(at, "So?")
    assert calls["discuss"][-1]["sent"] == [2]
    assert any(e.label == "1 earlier question and the replies" for e in at.expander)
    _button(at, "📝 Close and record a decision").click().run()
    assert calls["note"] == 1
    assert sum("Decision note" in m.value for m in at.markdown) >= 2      # under the answer and in history
    assert any(e.label == "Earlier in this conversation" for e in at.expander)
    assert not at.exception


@pytest.mark.parametrize("kind", ["suggest", "discuss", "turn"])
def test_while_anything_runs_the_chat_box_waits_and_the_panel_is_locked(agent, monkeypatch, kind):
    _script(agent, monkeypatch, freeze={kind: 1})
    at = _app()
    if kind == "suggest":
        _button(at, "✨ Suggest next steps").click().run()
    elif kind == "discuss":
        _mode(at, "discuss")
        _ask(at, "Is it N-1 secure?")
        assert any("conductor-you'>Is it N-1 secure?" in m.value for m in at.markdown)
    else:
        _send(at, "Run an RSA")
    busy = _busy_text(at)
    assert busy and ("CONDUCTOR is working" in busy[0] if kind == "turn" else "advisor is thinking" in busy[0])
    assert not any(b.label == "Run" for b in at.button)
    assert [b.disabled for b in at.button if b.key == "agent_hide"] == [True]
    assert [s.disabled for s in at.get("segmented_control") if s.label == "Advisor"] == [True]


def test_interrupted_work_resumes_and_stops_after_two_attempts(agent, monkeypatch):
    calls = _script(agent, monkeypatch, freeze={"turn": 1})
    at = _app()
    _send(at, "Interrupted")
    assert at.session_state["_pending_turn"] == "Interrupted"
    assert [m["text"] for m in at.session_state["messages"]][-1] == "a"   # nothing half-saved
    at.run()
    assert [m["text"] for m in at.session_state["messages"]][-2:] == ["Interrupted", "Answer to: Interrupted"]
    assert len(calls["turn"]) == 2
    _script(agent, monkeypatch, freeze={"turn": 9})
    at = _app()
    _send(at, "Never")
    at.run()
    at.run()
    assert any("interrupted twice" in w.value for w in at.warning)
    assert [m["text"] for m in at.session_state["messages"]][-1] == "a"


def test_an_interrupted_discussion_reply_resumes(agent, monkeypatch):
    calls = _script(agent, monkeypatch, freeze={"discuss": 1})
    at = _app()
    _mode(at, "discuss")
    _ask(at, "Q1")
    assert at.session_state["_advisor_request"] == ("ask", "Q1", 1)
    at.run()
    assert [t["role"] for t in at.session_state["discussions"][0]["thread"]] == ["operator", "advisor"]
    assert len(calls["discuss"]) == 2


@pytest.mark.parametrize("attention, key", [(["run_rsa: 1 violation(s)"], "agent_open_flag"), ([], "agent_open")])
def test_collapse_and_reopen_on_the_right(agent, monkeypatch, attention, key):
    _script(agent, monkeypatch)
    at = _app(attention=attention)
    _button(at, key_prefix="agent_hide").click().run()
    assert not any("Advisor**" in m.value for m in at.markdown)
    opener = [b for b in at.button if b.key == key]
    assert opener and opener[0].label == "✨"
    opener[0].click().run()
    assert any("Advisor**" in m.value for m in at.markdown)
    assert not at.exception


def test_planning_with_the_agent_before_the_first_question(agent, monkeypatch):
    calls = _script(agent, monkeypatch, studies=("worst hour",))
    at = _app(messages=[])
    assert any(m.value.startswith("**Before you ask**") for m in at.markdown)
    assert not any(s.label == "Advisor" for s in at.get("segmented_control"))   # no suggestions yet
    _ask(at, "How should we assess a 30 MW data centre?")
    assert calls["discuss"] and calls["discuss"][0]["q"] == "How should we assess a 30 MW data centre?"
    assert not any(b.key == "adv_close_0" for b in at.button)        # no decision without evidence
    _button(at, key_prefix="adv_use_").click().run()
    _send(at, "Run worst hour.")
    assert calls["turn"] == ["Run worst hour."]
    # The same discussion carries on beside the answer, and can now be closed.
    assert len(at.session_state["discussions"]) == 1
    assert any("**About the latest answer**" in m.value for m in at.markdown)
    assert [s.value for s in at.get("segmented_control") if s.label == "Advisor"] == ["discuss"]
    assert any("answered in turn 1" in m.value for m in at.markdown if "conductor-sent" in m.value)
    assert any(b.key == "adv_close_0" for b in at.button)
    assert not at.exception


def test_the_panel_can_be_hidden_before_the_first_question(agent, monkeypatch):
    _script(agent, monkeypatch)
    at = _app(messages=[])
    _button(at, key_prefix="agent_hide").click().run()
    opener = [b for b in at.button if b.key == "agent_open"]
    assert opener and not at.exception
    opener[0].click().run()
    assert any(m.value.startswith("**Before you ask**") for m in at.markdown)


def test_the_agent_starts_closed_and_opens_with_its_launcher(agent, monkeypatch):
    from streamlit.testing.v1 import AppTest
    _script(agent, monkeypatch)
    at = AppTest.from_file(APP, default_timeout=120)
    at.session_state["_launched"] = True
    at.run()
    assert any("**Open the advisor** (✨ on the right)" in m.value for m in at.markdown)
    assert not any(m.value.startswith("**Before you ask**") for m in at.markdown)
    _button(at, key_prefix="agent_open").click().run()
    assert any(m.value.startswith("**Before you ask**") for m in at.markdown)
    assert not at.exception


def test_a_flag_under_the_answer_points_to_the_closed_agent(agent, monkeypatch):
    _script(agent, monkeypatch)
    at = _app(attention=["run_rsa: 1 violation(s)"], extra={"agent_panel_open": False})
    assert any("open the advisor (✨ on the right)" in c.value for c in at.caption)
    assert [b.key for b in at.button if b.key and b.key.startswith("agent_open")] == ["agent_open_flag"]


def test_each_side_says_what_it_is_for_and_they_hand_over(agent, monkeypatch):
    """CONDUCTOR runs studies; the advisor discusses. Each box says whose it
    is, an answer leads to the advisor, and the advisor's studies lead back."""
    calls = _script(agent, monkeypatch, studies=("N-1 now",))
    at = _app(extra={"agent_panel_open": False})
    assert any("conductor-who'>⚡ CONDUCTOR" in m.value for m in at.markdown)
    box = [t for t in at.text_input if str(t.key).startswith("chat_box_")][0]
    assert box.placeholder == "Ask CONDUCTOR to run, compare or explain a study…"
    _button(at, "💬 Discuss with the advisor").click().run()
    assert at.session_state["agent_panel_open"] and at.session_state["agent_mode"] == "discuss"
    assert not any(b.label == "💬 Discuss with the advisor" for b in at.button)   # already there
    assert any("it runs no studies; CONDUCTOR does" in c.value for c in at.caption)
    _ask(at, "Would you trust it?")
    card = _button(at, key_prefix="adv_use_")
    assert card.label == "Run in CONDUCTOR →"
    card.click().run()
    assert at.session_state["_box_value"] == "Run N-1 now."
    _send(at, "Run N-1 now.")
    assert calls["turn"] == ["Run N-1 now."]
    assert not at.exception
