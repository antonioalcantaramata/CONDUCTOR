"""
A long scan no longer stalls the backend (code review 7.6).

The endpoints are `async def` doing blocking work; on the event loop one scan
held up every request, `/health` and the clock included. They now run in a
worker thread under one lock: cheap reads answer meanwhile, while the
endpoints that share `app_data` still run one at a time.
"""

import threading
import time

import pytest

pytest.importorskip("fastapi.testclient")

import main_backend as mb  # noqa: E402

SLOW = 1.5


@pytest.fixture
def slow_scan(backend_client, monkeypatch):
    """`/api/grid/rsa` takes SLOW seconds longer; records when each call ran."""
    spans = []
    real = mb._resolve_tick

    def slow(*args, **kwargs):
        start = time.monotonic()
        time.sleep(SLOW)
        spans.append((start, time.monotonic()))
        return real(*args, **kwargs)

    monkeypatch.setattr(mb, "_resolve_tick", slow)
    return spans


def _in_background(call):
    out = {}
    thread = threading.Thread(target=lambda: out.setdefault("r", call()))
    thread.start()
    return thread, out


def test_health_answers_during_a_scan(backend_client, slow_scan):
    thread, out = _in_background(lambda: backend_client.post("/api/grid/rsa", json={}))
    time.sleep(0.3)
    start = time.monotonic()
    for path in ("/health", "/api/time/current", "/api/time/timeline"):
        assert backend_client.get(path).status_code == 200
    assert time.monotonic() - start < SLOW / 2
    thread.join()
    assert out["r"].status_code == 200


def test_scans_do_not_interleave(backend_client, slow_scan):
    threads = [_in_background(lambda: backend_client.post("/api/grid/rsa", json={}))[0]
               for _ in range(2)]
    for thread in threads:
        thread.join()
    (a_start, a_end), (b_start, b_end) = sorted(slow_scan)
    assert b_start >= a_end


def test_a_locked_endpoint_may_call_another(backend_client):
    """robust → optimise runs on the thread that already holds the lock."""
    calls = []

    @mb._off_event_loop
    async def inner():
        calls.append(threading.get_ident())
        return "inner"

    @mb._off_event_loop
    async def outer():
        calls.append(threading.get_ident())
        return await inner()

    import anyio
    assert anyio.run(outer) == "inner"
    assert len(set(calls)) == 1 and not mb._STATE_LOCK.locked()


def test_errors_still_reach_the_client(backend_client):
    r = backend_client.post("/api/grid/rsa", json={"timestamp": "1900-01-01 00:00:00"})
    assert r.status_code in (400, 404, 422)
    assert not mb._STATE_LOCK.locked()
