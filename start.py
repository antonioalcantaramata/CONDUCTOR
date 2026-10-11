#!/usr/bin/env python3
"""start.py — Cross-platform launcher for the CONDUCTOR backend + chat app.

Works on macOS, Linux, and Windows (PowerShell, cmd, Anaconda Prompt, Git
Bash) — any shell where `conda` and a Python interpreter are on the PATH.

Usage:
    python start.py            # macOS / Linux (or ./start.sh, which wraps this)
    py start.py                # Windows

Environment overrides:
    GRID_PROFILE=ieee14        # bundled network profile (default: pglib_case14)
    BACKEND_PORT=8010          # FastAPI port   (default: 8000)
    STREAMLIT_PORT=8502        # Streamlit port (default: 8501)
    BACKEND_HOST=0.0.0.0       # bind address   (default: 127.0.0.1 — keep it
                               # local unless you need LAN access; 0.0.0.0
                               # triggers OS firewall prompts)
    CONDUCTOR_UPDATE_CHECK=0   # skip the check for a newer version on GitHub
    STREAMLIT_HOST=0.0.0.0     # chat app bind address (default: localhost;
                               # Streamlit itself would listen on every
                               # interface, LAN and public addresses included).
                               # A specific address — e.g. the machine's
                               # Tailscale IP — serves the app on that network
                               # only, without any port forwarding.

Over SSH (e.g. VS Code Remote-SSH), forward ports STREAMLIT_PORT and
BACKEND_PORT to your own machine and open http://localhost:<STREAMLIT_PORT>;
the launcher prints the exact steps when it detects an SSH session.

Press Ctrl+C once to stop both services cleanly.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from llm_agent.agent import update_check  # noqa: E402 — standard library only

BACKEND_DIR = SCRIPT_DIR / "backend"
AGENT_DIR = SCRIPT_DIR / "llm_agent"
ENV_NAME = "conductor_env"
ENV_YML = SCRIPT_DIR / "environment.yml"
HASH_FILE = SCRIPT_DIR / ".conductor_env_hash"

BACKEND_HOST = os.environ.get("BACKEND_HOST", "127.0.0.1")
BACKEND_PORT = int(os.environ.get("BACKEND_PORT", "8000"))
STREAMLIT_PORT = int(os.environ.get("STREAMLIT_PORT", "8501"))
STREAMLIT_HOST = os.environ.get("STREAMLIT_HOST", "localhost")
_WILDCARD_OR_LOCAL = {"localhost", "127.0.0.1", "0.0.0.0", "::", "::1", ""}


def streamlit_url_host(host: str = STREAMLIT_HOST) -> str:
    """Where the chat app is reached: localhost, unless it is bound to one
    specific address — then only that address answers."""
    return "localhost" if host in _WILDCARD_OR_LOCAL else host
GRID_PROFILE = os.environ.get("GRID_PROFILE", "pglib_case14")

# Enable ANSI colours (the empty os.system call switches on VT processing in
# legacy Windows consoles; harmless elsewhere).
if os.name == "nt":
    os.system("")
_USE_COLOUR = sys.stdout.isatty()
GREEN = "\033[0;32m" if _USE_COLOUR else ""
YELLOW = "\033[1;33m" if _USE_COLOUR else ""
RED = "\033[0;31m" if _USE_COLOUR else ""
NC = "\033[0m" if _USE_COLOUR else ""


def log(msg: str) -> None:
    print(f"{GREEN}[start]{NC} {msg}", flush=True)


def warn(msg: str) -> None:
    print(f"{YELLOW}[start]{NC} {msg}", flush=True)


def die(msg: str, code: int = 1) -> None:
    print(f"{RED}[start] ERROR:{NC} {msg}", file=sys.stderr, flush=True)
    sys.exit(code)


def tail(path: Path, n: int = 20) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(lines[-n:])
    except OSError:
        return "(no log output)"


# ---------------------------------------------------------------------------
# Conda environment management
# ---------------------------------------------------------------------------

def find_conda() -> str:
    conda = shutil.which("conda")
    if conda:
        return conda
    die(
        "conda not found on PATH.\n\n"
        "  Miniconda (free, ~100 MB) is the only prerequisite.\n"
        "  Install it from: https://www.anaconda.com/docs/getting-started/miniconda/install\n\n"
        "  On Windows, run this script from the 'Anaconda Prompt' that the\n"
        "  installer creates, or tick 'Add Miniconda to PATH' during install.\n"
        "  On macOS/Linux, close and reopen your terminal after installing."
    )
    raise SystemExit  # unreachable; keeps type-checkers happy


def env_exists(conda: str) -> bool:
    out = subprocess.run(
        [conda, "env", "list", "--json"], capture_output=True, text=True, check=True
    ).stdout
    prefixes = json.loads(out).get("envs", [])
    return any(Path(p).name == ENV_NAME for p in prefixes)


def env_yml_hash() -> str:
    return hashlib.sha256(ENV_YML.read_bytes()).hexdigest()


def sanity_check_env(conda: str) -> bool:
    """Quick import probe to detect a half-created (interrupted) env."""
    probe = subprocess.run(
        [conda, "run", "-n", ENV_NAME, "python", "-c",
         "import pandapower, fastapi, streamlit"],
        capture_output=True,
    )
    return probe.returncode == 0


def ensure_env(conda: str) -> None:
    if not ENV_YML.is_file():
        die(f"environment.yml not found at {ENV_YML}")

    if not env_exists(conda):
        warn(f"Conda env '{ENV_NAME}' not found — creating it from environment.yml…")
        warn("First-time setup: expect ~3–5 minutes and a ~2 GB download. Only happens once.")
        try:
            subprocess.run([conda, "env", "create", "-f", str(ENV_YML)], check=True)
        except KeyboardInterrupt:
            die(
                "Environment creation interrupted. The half-created env is likely broken;\n"
                f"  remove it before retrying:  conda env remove -n {ENV_NAME}"
            )
        except subprocess.CalledProcessError:
            die(
                "Failed to create the conda env (see output above).\n"
                f"  If it was partially created, remove it first:  conda env remove -n {ENV_NAME}"
            )
        HASH_FILE.write_text(env_yml_hash(), encoding="utf-8")
        log(f"Conda env '{ENV_NAME}' created ✓")
        return

    current = env_yml_hash()
    stored = HASH_FILE.read_text(encoding="utf-8").strip() if HASH_FILE.is_file() else ""

    if not stored:
        # Env exists but we've never verified it — probe for an interrupted create.
        log("Verifying existing environment…")
        if not sanity_check_env(conda):
            die(
                f"Conda env '{ENV_NAME}' exists but core packages fail to import — it is\n"
                "  probably a leftover from an interrupted install. Rebuild it with:\n"
                f"    conda env remove -n {ENV_NAME}\n"
                "  then run this launcher again."
            )
        HASH_FILE.write_text(current, encoding="utf-8")
        return

    if current != stored:
        warn(f"environment.yml changed — updating '{ENV_NAME}'…")
        try:
            subprocess.run(
                [conda, "env", "update", "-n", ENV_NAME, "-f", str(ENV_YML), "--prune"],
                check=True,
            )
        except subprocess.CalledProcessError:
            die("conda env update failed. Check the output above.")
        HASH_FILE.write_text(current, encoding="utf-8")
        log("Env updated ✓")


# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------

def check_port_free(port: int, what: str, override_var: str) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        if s.connect_ex(("127.0.0.1", port)) == 0:
            die(
                f"Port {port} is already in use, so the {what} cannot start.\n"
                f"  Stop the process using it, or pick another port:\n"
                f"    {override_var}={port + 10} python start.py"
            )


def check_llm_config() -> None:
    """
    Warn only if the *selected* backend is unconfigured.

    An Ollama user has no Gemini key and needs none, so checking for one
    unconditionally would warn at every launch about a non-problem.
    """
    env_file = AGENT_DIR / ".env"
    values: dict[str, str] = {}
    if env_file.is_file():
        for line in env_file.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                values[key.strip()] = value.strip().strip('"').strip("'")

    provider = values.get("LLM_PROVIDER", "").lower() or "google"

    if provider == "ollama":
        if not values.get("OLLAMA_MODEL"):
            warn(f"No OLLAMA_MODEL set in {env_file}")
            warn("The app will let you pick one from your installed models.")
        return
    # Each hosted backend has its own key; asking an OpenAI or Anthropic user
    # for a Gemini key was a warning about the wrong thing.
    key_var = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}.get(provider, "GEMINI_API_KEY")
    if not values.get(key_var):
        warn(f"No {key_var} found in {env_file}")
        warn("The app will guide you through setup on first open.")


# ---------------------------------------------------------------------------
# Service management
# ---------------------------------------------------------------------------

def _process_group_kwargs() -> dict:
    """Popen kwargs that make the child the root of its own process group.

    Needed so terminate() can signal the whole tree — `conda run` does not
    forward signals to its child, so signalling just the wrapper PID would
    orphan uvicorn/streamlit.
    """
    if os.name == "nt":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


def launch(conda: str, args: list[str], cwd: Path, log_path: Path,
           extra_env: dict[str, str] | None = None) -> subprocess.Popen:
    env = {**os.environ, **(extra_env or {})}
    log_fh = open(log_path, "wb")
    return subprocess.Popen(
        [conda, "run", "-n", ENV_NAME, "--no-capture-output", *args],
        cwd=str(cwd), stdout=log_fh, stderr=subprocess.STDOUT, env=env,
        **_process_group_kwargs(),
    )


def wait_for_streamlit(proc: subprocess.Popen, log_path: Path, timeout_s: int = 60) -> None:
    """Until the chat app answers its health check. The browser used to be
    opened after a fixed two seconds — a page that cannot load if the app is
    slower than that (and, over SSH, a forwarded port with nothing behind it)."""
    probe = streamlit_url_host()
    url = f"http://{'127.0.0.1' if probe == 'localhost' else probe}:{STREAMLIT_PORT}/_stcore/health"
    for _ in range(timeout_s * 4):
        if proc.poll() is not None:
            warn("Streamlit exited unexpectedly. Last lines of streamlit.log:")
            print(tail(log_path), file=sys.stderr)
            die("Streamlit failed to start.")
        try:
            with urllib.request.urlopen(url, timeout=1) as resp:
                if resp.status == 200:
                    return
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(0.25)
    warn(f"Streamlit did not answer within {timeout_s} s; opening the browser anyway.")


def ssh_session() -> bool:
    """Whether this launcher runs on a machine reached over SSH — where the
    browser that opens is on another computer, and `localhost` there is not
    this machine unless the ports are forwarded."""
    return bool(os.environ.get("SSH_CONNECTION") or os.environ.get("SSH_CLIENT"))


def access_instructions(streamlit_port: int, backend_port: int, remote: bool,
                        host: str = "localhost") -> list[str]:
    """What to open, and over SSH how to reach it. The URL is printed bare —
    no colour codes — so terminals (and VS Code's port detection) recognise it."""
    if host != "localhost":
        return [
            f"URL: http://{host}:{streamlit_port}",
            f"Served on {host} only: open it from any computer that can reach that "
            "address (e.g. on the same Tailscale network) — no port forwarding needed.",
            f"Uploading a network from the browser also needs port {backend_port} "
            "forwarded; everything else works without it.",
        ]
    lines = [f"Local URL: http://localhost:{streamlit_port}"]
    if remote:
        lines += [
            "",
            "This is an SSH session: the browser runs on your own computer, so both ports",
            f"must be forwarded to it — {streamlit_port} (chat app) and {backend_port} (backend;",
            "network uploads go from the browser straight to it).",
            f"  VS Code Remote-SSH: Ports panel -> Forward a Port -> {streamlit_port}, then {backend_port}.",
            f"  Plain ssh: ssh -L {streamlit_port}:localhost:{streamlit_port} "
            f"-L {backend_port}:localhost:{backend_port} <user>@<this machine>",
            f"Then open http://localhost:{streamlit_port} on your own computer.",
        ]
    return lines


def wait_for_backend(proc: subprocess.Popen, log_path: Path, timeout_s: int = 120) -> None:
    log("Waiting for backend to be ready (this can take 60–120 s on first load)…")
    url = f"http://{'127.0.0.1' if BACKEND_HOST == '0.0.0.0' else BACKEND_HOST}:{BACKEND_PORT}/health"
    for i in range(1, timeout_s + 1):
        if proc.poll() is not None:
            print()
            warn("Backend process exited unexpectedly. Last lines of backend.log:")
            print(tail(log_path), file=sys.stderr)
            die("Backend failed to start.")
        try:
            with urllib.request.urlopen(url, timeout=1) as resp:
                if resp.status == 200:
                    print()
                    log(f"Backend is up ✓  (took {i}s)")
                    return
        except urllib.error.HTTPError:
            # The server answered (even if /health is missing) — it's up.
            print()
            log(f"Backend is up ✓  (took {i}s)")
            return
        except OSError:
            pass
        print(".", end="", flush=True)
        time.sleep(1)
    print()
    warn(f"Backend did not respond after {timeout_s} s — Streamlit will start anyway.")
    warn("Check backend.log for errors.")


def terminate(proc: subprocess.Popen | None, name: str) -> None:
    if proc is None or proc.poll() is not None:
        return
    if os.name == "nt":
        # /T kills the whole tree (conda run wrapper + the actual service).
        subprocess.run(
            ["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True
        )
    else:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except ProcessLookupError:
            return
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        if os.name != "nt":
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
        proc.kill()
    log(f"{name} stopped (PID {proc.pid})")


class UpdateCheck:
    """The GitHub version check, run in the background while the services
    start so it never delays them."""

    def __init__(self) -> None:
        self.status: dict | None = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        if update_check.enabled():
            self._thread.start()

    def _run(self) -> None:
        self.status = update_check.check(SCRIPT_DIR)

    def result(self, wait_s: float = 0.0) -> tuple[bool, dict | None]:
        """(finished, status) — waits at most `wait_s`."""
        if not self._thread.is_alive() and self._thread.ident is None:
            return False, None                       # disabled: never started
        self._thread.join(timeout=wait_s)
        return (not self._thread.is_alive()), self.status


def main() -> None:
    # Ensure the cleanup in `finally` also runs when the launcher is killed
    # (SIGTERM), not just on Ctrl+C.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    updates = UpdateCheck()

    conda = find_conda()
    ensure_env(conda)
    check_llm_config()
    check_port_free(BACKEND_PORT, "FastAPI backend", "BACKEND_PORT")
    check_port_free(STREAMLIT_PORT, "Streamlit app", "STREAMLIT_PORT")

    backend_proc: subprocess.Popen | None = None
    streamlit_proc: subprocess.Popen | None = None
    backend_log = SCRIPT_DIR / "backend.log"
    streamlit_log = SCRIPT_DIR / "streamlit.log"
    backend_url = f"http://{'127.0.0.1' if BACKEND_HOST == '0.0.0.0' else BACKEND_HOST}:{BACKEND_PORT}"

    try:
        log(f"Starting FastAPI backend on {backend_url} …")
        log(f"Grid profile: {GRID_PROFILE}")
        backend_proc = launch(
            conda,
            ["uvicorn", "main_backend:app", "--host", BACKEND_HOST, "--port", str(BACKEND_PORT)],
            cwd=BACKEND_DIR, log_path=backend_log,
            extra_env={"GRID_PROFILE": GRID_PROFILE},
        )
        log(f"Backend PID: {backend_proc.pid}  (logs → backend.log)")
        wait_for_backend(backend_proc, backend_log)

        # The app shows the same notice; it is handed the result rather than
        # asking GitHub again. Not finished yet → the app checks on its own.
        checked, status = updates.result(wait_s=1.0)
        app_env = {"DT_BACKEND_URL": backend_url}
        if checked:
            app_env[update_check.ENV_RESULT] = update_check.to_env(status)

        log(f"Starting Streamlit chat app on http://localhost:{STREAMLIT_PORT} …")
        streamlit_proc = launch(
            conda,
            ["streamlit", "run", "app.py",
             "--server.port", str(STREAMLIT_PORT), "--server.headless", "true",
             "--server.address", STREAMLIT_HOST],
            cwd=AGENT_DIR, log_path=streamlit_log,
            extra_env=app_env,
        )
        log(f"Streamlit PID: {streamlit_proc.pid}  (logs → streamlit.log)")
        wait_for_streamlit(streamlit_proc, streamlit_log)

        print()
        log("Both services running. Press Ctrl+C to stop.")
        print(f"  Backend:   {GREEN}{backend_url}/docs{NC}")
        url_host = streamlit_url_host()
        for line in access_instructions(STREAMLIT_PORT, BACKEND_PORT, ssh_session(), url_host):
            print(f"  {line}" if line else "")
        notice = update_check.message(updates.result(wait_s=2.0)[1])
        if notice:
            warn(notice)
        print(flush=True)  # shown at once even when the output is not a terminal
        # Over VS Code Remote-SSH this opens the browser on your own computer.
        webbrowser.open(f"http://{url_host}:{STREAMLIT_PORT}")

        while True:
            time.sleep(1)
            for proc, name, log_path in (
                (backend_proc, "Backend", backend_log),
                (streamlit_proc, "Streamlit", streamlit_log),
            ):
                if proc.poll() is not None:
                    warn(f"{name} exited unexpectedly. Last lines of {log_path.name}:")
                    print(tail(log_path), file=sys.stderr)
                    raise SystemExit(1)
    except KeyboardInterrupt:
        print()
        warn("Shutting down…")
    finally:
        terminate(streamlit_proc, "Streamlit")
        terminate(backend_proc, "Backend")


if __name__ == "__main__":
    main()
