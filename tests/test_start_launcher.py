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


class TestCheckLlmConfig:
    def test_warns_when_env_file_missing(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(start, "AGENT_DIR", tmp_path)
        start.check_llm_config()
        assert "No GEMINI_API_KEY" in capsys.readouterr().out

    def test_warns_when_key_empty(self, tmp_path, monkeypatch, capsys):
        (tmp_path / ".env").write_text("GEMINI_API_KEY=\n")
        monkeypatch.setattr(start, "AGENT_DIR", tmp_path)
        start.check_llm_config()
        assert "No GEMINI_API_KEY" in capsys.readouterr().out

    def test_silent_when_key_present(self, tmp_path, monkeypatch, capsys):
        (tmp_path / ".env").write_text("GEMINI_API_KEY=abc123\n")
        monkeypatch.setattr(start, "AGENT_DIR", tmp_path)
        start.check_llm_config()
        assert "No GEMINI_API_KEY" not in capsys.readouterr().out

    def test_ollama_user_is_not_nagged_about_a_gemini_key(
        self, tmp_path, monkeypatch, capsys
    ):
        # A local-model user has no API key and needs none.
        (tmp_path / ".env").write_text(
            "LLM_PROVIDER=ollama\nOLLAMA_MODEL=some-model\n"
        )
        monkeypatch.setattr(start, "AGENT_DIR", tmp_path)
        start.check_llm_config()
        assert capsys.readouterr().out.strip() == ""

    @pytest.mark.parametrize("provider,key", [("openai", "OPENAI_API_KEY"),
                                              ("anthropic", "ANTHROPIC_API_KEY")])
    def test_hosted_providers_are_asked_for_their_own_key(
        self, tmp_path, monkeypatch, capsys, provider, key
    ):
        (tmp_path / ".env").write_text(f"LLM_PROVIDER={provider}\n")
        monkeypatch.setattr(start, "AGENT_DIR", tmp_path)
        start.check_llm_config()
        out = capsys.readouterr().out
        assert f"No {key}" in out and "GEMINI" not in out

    def test_a_quoted_key_counts_as_present(self, tmp_path, monkeypatch, capsys):
        (tmp_path / ".env").write_text('LLM_PROVIDER=openai\nOPENAI_API_KEY="sk-x"\n')
        monkeypatch.setattr(start, "AGENT_DIR", tmp_path)
        start.check_llm_config()
        assert capsys.readouterr().out.strip() == ""

    def test_warns_when_ollama_selected_without_a_model(
        self, tmp_path, monkeypatch, capsys
    ):
        (tmp_path / ".env").write_text("LLM_PROVIDER=ollama\n")
        monkeypatch.setattr(start, "AGENT_DIR", tmp_path)
        start.check_llm_config()
        assert "No OLLAMA_MODEL" in capsys.readouterr().out

    def test_comments_and_blank_lines_are_ignored(
        self, tmp_path, monkeypatch, capsys
    ):
        (tmp_path / ".env").write_text(
            "# GEMINI_API_KEY=commented-out\n\nLLM_PROVIDER=google\n"
            "GEMINI_API_KEY=real\n"
        )
        monkeypatch.setattr(start, "AGENT_DIR", tmp_path)
        start.check_llm_config()
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


class TestRemoteAccess:
    """Over SSH the browser is on another computer: the launcher says how to
    reach the app, waits until it answers, and keeps it off the network."""

    def test_ssh_is_detected(self, monkeypatch):
        for var in ("SSH_CONNECTION", "SSH_CLIENT"):
            monkeypatch.delenv(var, raising=False)
        assert not start.ssh_session()
        monkeypatch.setenv("SSH_CONNECTION", "10.0.0.2 52063 10.0.0.1 22")
        assert start.ssh_session()

    def test_instructions(self):
        local = start.access_instructions(8501, 8000, remote=False)
        assert local == ["Local URL: http://localhost:8501"]   # bare, no colour codes
        remote = "\n".join(start.access_instructions(8502, 8010, remote=True))
        assert "Forward a Port -> 8502, then 8010" in remote
        assert "ssh -L 8502:localhost:8502 -L 8010:localhost:8010" in remote

    def test_waiting_for_the_app_returns_once_it_answers(self, monkeypatch):
        import http.server
        import threading

        class Health(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200 if self.path == "/_stcore/health" else 404)
                self.end_headers()

            def log_message(self, *args):
                pass

        server = http.server.HTTPServer(("127.0.0.1", 0), Health)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        monkeypatch.setattr(start, "STREAMLIT_PORT", server.server_address[1])
        proc = types.SimpleNamespace(poll=lambda: None)
        t0 = time.time()
        start.wait_for_streamlit(proc, log_path=None, timeout_s=5)
        assert time.time() - t0 < 2
        server.shutdown()


class TestServedOnOneAddress:
    def test_a_specific_address_is_where_the_app_is_reached(self):
        assert start.streamlit_url_host("localhost") == "localhost"
        assert start.streamlit_url_host("0.0.0.0") == "localhost"
        assert start.streamlit_url_host("100.110.91.32") == "100.110.91.32"

    def test_instructions_for_an_address(self):
        lines = start.access_instructions(8501, 8000, remote=True, host="100.110.91.32")
        assert lines[0] == "URL: http://100.110.91.32:8501"
        assert "no port forwarding needed" in lines[1]
