"""
Session log retention.

The turn record is the only durable evidence of what the agent did, and the
offline graders read nothing else. It used to be a single file truncated at
process start, so every run destroyed the previous one — which happened during
development and cost a real session. These tests pin the two properties that
matter: a run never overwrites another run's log, and no existing log is
deleted to make room.
"""

import json
import pathlib
import tempfile

import pytest

from llm_agent.agent import loop


def _symlinks_work() -> bool:
    """Windows refuses symlinks without elevation; the pointer is optional there."""
    with tempfile.TemporaryDirectory() as scratch:
        try:
            (pathlib.Path(scratch) / "link").symlink_to("target")
        except OSError:
            return False
    return True


SYMLINKS = pytest.mark.skipif(
    not _symlinks_work(),
    reason="platform does not allow symlinks; per-run log files still apply",
)


@pytest.fixture
def log_dir(tmp_path, monkeypatch):
    """Point the loop's logging at a temporary directory, freshly initialised."""
    monkeypatch.setattr(loop, "_LOG_DIR", tmp_path)
    monkeypatch.setattr(loop, "_LATEST", tmp_path / "last_session.jsonl")
    monkeypatch.setattr(loop, "SESSION_LOG_PATH", "")
    monkeypatch.setattr(loop, "_log_path", None)
    monkeypatch.setattr(loop, "_log_initialized", False)
    monkeypatch.setattr(loop, "session_log", [])
    return tmp_path


class TestPathSelection:
    def test_explicit_path_wins(self, log_dir, monkeypatch):
        # A batch sweep directs every turn it runs into one file.
        target = log_dir / "sweep.jsonl"
        monkeypatch.setattr(loop, "SESSION_LOG_PATH", str(target))
        assert loop._session_log_path() == target

    def test_default_is_unique_per_run(self, log_dir):
        path = loop._session_log_path()
        assert path.parent == log_dir
        assert path.name.startswith("session-") and path.suffix == ".jsonl"
        # Chosen once and reused for the life of the process.
        assert loop._session_log_path() == path


class TestWriting:
    def test_turns_accumulate_rather_than_overwrite(self, log_dir):
        loop._append_to_log({"turn": 1})
        loop._append_to_log({"turn": 2})
        written = loop._session_log_path().read_text(encoding="utf-8").splitlines()
        assert [json.loads(line)["turn"] for line in written] == [1, 2]

    def test_a_second_run_does_not_touch_the_first(self, log_dir, monkeypatch):
        loop._append_to_log({"turn": 1})
        first = loop._session_log_path()

        # Simulate a fresh process, one second later.
        monkeypatch.setattr(loop, "_log_path", log_dir / "session-later.jsonl")
        monkeypatch.setattr(loop, "_log_initialized", False)
        loop._append_to_log({"turn": 1})

        assert first.read_text(encoding="utf-8").strip() == '{"turn": 1}'
        assert len(list(log_dir.glob("session-*.jsonl"))) == 2

    def test_a_write_failure_is_not_fatal(self, log_dir, monkeypatch):
        # Logging is evidence, not function: losing it must not lose the turn.
        monkeypatch.setattr(loop, "_log_path", log_dir / "missing" / "x" / "s.jsonl")
        monkeypatch.setattr(pathlib.Path, "mkdir", lambda *a, **k: (_ for _ in ()).throw(OSError))
        loop._append_to_log({"turn": 1})
        assert loop.session_log == [{"turn": 1}]


@SYMLINKS
class TestLatestPointer:
    def test_points_at_the_current_run(self, log_dir):
        loop._append_to_log({"turn": 1})
        latest = log_dir / "last_session.jsonl"
        assert latest.is_symlink()
        assert json.loads(latest.read_text(encoding="utf-8"))["turn"] == 1

    def test_an_existing_real_file_is_archived_not_deleted(self, log_dir):
        # Left over from the old truncating scheme, and it holds a session
        # somebody may still want.
        stale = log_dir / "last_session.jsonl"
        stale.write_text('{"turn": 99}\n', encoding="utf-8")

        loop._append_to_log({"turn": 1})

        archived = list(log_dir.glob("*-archived.jsonl"))
        assert len(archived) == 1
        assert json.loads(archived[0].read_text(encoding="utf-8"))["turn"] == 99

    def test_a_stale_symlink_is_replaced(self, log_dir):
        latest = log_dir / "last_session.jsonl"
        latest.symlink_to("session-gone.jsonl")
        loop._append_to_log({"turn": 1})
        assert latest.resolve() == loop._session_log_path().resolve()
