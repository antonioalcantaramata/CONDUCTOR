"""The progress of an ablation run (evaluation/progress.py)."""

import io
import time

from evaluation import progress as pg


def test_the_line_counts_turns_and_estimates_from_the_turns_run_now(tmp_path):
    state = pg.start(tmp_path, total=100, done=40)          # a resume: 40 turns already done
    state["started"] = time.time() - 600
    pg.save(tmp_path, state)
    pg.record(tmp_path, 10, seconds=100.0, current="full · reasoning/names_cigre #2")
    state = pg.record(tmp_path, 10, seconds=100.0, failed=True)
    text = pg.line(state)
    assert "60/100 turns" in text and " 60%" in text and "1 failed" in text
    # 20 turns took 200 s: 10 s a turn, 40 turns left → ~7 min (the 40 found done do not count).
    assert "~7 min left" in text and "10 min" in text


def test_skipped_runs_advance_the_bar_but_not_the_estimate(tmp_path):
    pg.start(tmp_path, total=10, done=0)
    state = pg.record(tmp_path, 5)                          # found finished on resume
    assert state["done"] == 5 and state["ran"] == 0 and "left" not in pg.line(state)


def test_a_log_gets_one_line_per_call_and_a_terminal_one_line_redrawn(tmp_path):
    state = pg.start(tmp_path, total=4, done=1)
    log = io.StringIO()
    pg.show(state, "a · t #1", log)
    pg.show(state, "a · t #2", log)
    assert log.getvalue().count("\n") == 2 and "a · t #2" in log.getvalue()

    class Tty(io.StringIO):
        def isatty(self):
            return True
    tty = Tty()
    pg.show(state, "x", tty)
    pg.show(state, "y", tty, final=True)
    assert tty.getvalue().startswith("\r") and tty.getvalue().count("\n") == 1


def test_a_missing_or_broken_file_reads_as_none(tmp_path):
    assert pg.load(tmp_path) is None
    (tmp_path / pg.FILE).write_text("{not json")
    assert pg.load(tmp_path) is None
