"""Unit tests for the cross-platform launcher (start.py).

These run on macOS, Linux, and Windows CI. They never invoke conda or the
launcher's main() — only the pure helpers and the process-tree termination
logic (with plain Python subprocesses standing in for `conda run` + service).
"""

import hashlib
import os
import socket
import subprocess
import sys
import time
import types

import pytest

import start


def _pid_alive(pid: int) -> bool:
    if os.name == "nt":
        out = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}"], capture_output=True, text=True
        ).stdout
        return str(pid) in out
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


class TestCheckPortFree:
    def test_free_port_passes(self):
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        start.check_port_free(port, "test service", "TEST_PORT")  # no exception

    def test_busy_port_exits_with_actionable_message(self, capsys):
        with socket.socket() as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("127.0.0.1", 0))
            s.listen(1)
            port = s.getsockname()[1]
            with pytest.raises(SystemExit):
                start.check_port_free(port, "test service", "TEST_PORT")
        err = capsys.readouterr().err
        assert str(port) in err
        assert "TEST_PORT" in err  # tells the user exactly how to override


class TestEnvYmlHash:
    def test_matches_sha256_of_file_bytes(self, tmp_path, monkeypatch):
        yml = tmp_path / "environment.yml"
        yml.write_bytes(b"name: test\ndependencies: []\n")
        monkeypatch.setattr(start, "ENV_YML", yml)
        assert start.env_yml_hash() == hashlib.sha256(yml.read_bytes()).hexdigest()


class TestTail:
    def test_returns_last_n_lines(self, tmp_path):
        f = tmp_path / "log.txt"
        f.write_text("\n".join(f"line{i}" for i in range(50)))
        result = start.tail(f, n=3)
        assert result == "line47\nline48\nline49"

    def test_missing_file_returns_placeholder(self, tmp_path):
        assert start.tail(tmp_path / "nope.log") == "(no log output)"


class TestCheckApiKey:
    def test_warns_when_env_file_missing(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(start, "AGENT_DIR", tmp_path)
        start.check_api_key()
        assert "No GEMINI_API_KEY" in capsys.readouterr().out

    def test_warns_when_key_empty(self, tmp_path, monkeypatch, capsys):
        (tmp_path / ".env").write_text("GEMINI_API_KEY=\n")
        monkeypatch.setattr(start, "AGENT_DIR", tmp_path)
        start.check_api_key()
        assert "No GEMINI_API_KEY" in capsys.readouterr().out

    def test_silent_when_key_present(self, tmp_path, monkeypatch, capsys):
        (tmp_path / ".env").write_text("GEMINI_API_KEY=abc123\n")
        monkeypatch.setattr(start, "AGENT_DIR", tmp_path)
        start.check_api_key()
        assert "No GEMINI_API_KEY" not in capsys.readouterr().out


class TestWaitForBackend:
    def test_any_http_response_counts_as_ready(self, tmp_path, monkeypatch):
        # A plain http.server returns 404 for unknown paths — the poll must
        # treat that as "server is up" (backends without /health still work).
        import http.server
        import threading

        httpd = http.server.HTTPServer(
            ("127.0.0.1", 0), http.server.BaseHTTPRequestHandler
        )
        port = httpd.server_address[1]
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        try:
            monkeypatch.setattr(start, "BACKEND_HOST", "127.0.0.1")
            monkeypatch.setattr(start, "BACKEND_PORT", port)
            fake_proc = types.SimpleNamespace(poll=lambda: None)
            t0 = time.monotonic()
            start.wait_for_backend(fake_proc, tmp_path / "backend.log", timeout_s=10)
            assert time.monotonic() - t0 < 8  # returned early, not via timeout
        finally:
            httpd.shutdown()

    def test_dead_process_exits_with_log_tail(self, tmp_path, monkeypatch, capsys):
        log_path = tmp_path / "backend.log"
        log_path.write_text("boom: address already in use\n")
        # An unbound port + a "dead" process → immediate failure path.
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            free_port = s.getsockname()[1]
        monkeypatch.setattr(start, "BACKEND_HOST", "127.0.0.1")
        monkeypatch.setattr(start, "BACKEND_PORT", free_port)
        fake_proc = types.SimpleNamespace(poll=lambda: 1)
        with pytest.raises(SystemExit):
            start.wait_for_backend(fake_proc, log_path, timeout_s=5)
        assert "address already in use" in capsys.readouterr().err


class TestTerminateKillsProcessTree:
    def test_parent_and_grandchild_both_die(self):
        """Simulate the real topology: launcher → conda-run wrapper → service.

        terminate() must kill the *whole* tree, not just the wrapper —
        `conda run` does not forward signals (the bug found in live testing).
        """
        parent_script = (
            "import subprocess, sys, time; "
            "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)']); "
            "print(p.pid, flush=True); "
            "time.sleep(120)"
        )
        proc = subprocess.Popen(
            [sys.executable, "-c", parent_script],
            stdout=subprocess.PIPE, text=True,
            **start._process_group_kwargs(),
        )
        try:
            grandchild_pid = int(proc.stdout.readline().strip())
            assert _pid_alive(grandchild_pid)

            start.terminate(proc, "test-tree")

            assert proc.poll() is not None  # wrapper is dead
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline and _pid_alive(grandchild_pid):
                time.sleep(0.2)
            assert not _pid_alive(grandchild_pid), (
                f"grandchild {grandchild_pid} survived terminate() — "
                "process-tree kill is broken on this platform"
            )
        finally:
            # Belt-and-braces cleanup if the assertion above ever fails.
            if proc.poll() is None:
                proc.kill()
            try:
                if _pid_alive(grandchild_pid):
                    if os.name == "nt":
                        subprocess.run(
                            ["taskkill", "/PID", str(grandchild_pid), "/F"],
                            capture_output=True,
                        )
                    else:
                        os.kill(grandchild_pid, 9)
            except (NameError, ProcessLookupError):
                pass
