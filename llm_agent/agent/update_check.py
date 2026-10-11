"""
update_check.py — is a newer CONDUCTOR on GitHub than the one running?

Run once at start (by `start.py`, and by the app when it was started some
other way). Standard library only: the launcher imports it before the conda
environment is active.

Read-only: `git ls-remote` asks the remote which commit its branch is at and
changes nothing in the checkout — no fetch, no refs moved. It never prompts
for credentials (a private repository uses whatever helper or key is already
set up, or the check is skipped), gives up after a few seconds, and returns
None whenever it cannot answer: no git, not a clone, offline, or disabled
with CONDUCTOR_UPDATE_CHECK=0. Starting the app must never wait on, or fail
because of, this check.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

ENV_DISABLE = "CONDUCTOR_UPDATE_CHECK"   # "0" / "false" / "off" turns it off
ENV_RESULT = "CONDUCTOR_UPDATE_STATUS"   # the launcher's result, handed to the app

_TIMEOUT_S = 6.0


def enabled() -> bool:
    return os.environ.get(ENV_DISABLE, "1").strip().lower() not in ("0", "false", "no", "off")


def _git(repo: Path | str, *args: str, timeout: float = _TIMEOUT_S) -> str | None:
    """stdout of a git command, "" when it succeeds silently, None on failure."""
    env = {
        **os.environ,
        "GIT_TERMINAL_PROMPT": "0",          # no username/password prompt
        "GCM_INTERACTIVE": "never",          # Git Credential Manager: no window
        "GIT_SSH_COMMAND": "ssh -o BatchMode=yes -o ConnectTimeout=5",
    }
    try:
        out = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True,
                             timeout=timeout, env=env, stdin=subprocess.DEVNULL)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def check(repo: Path | str, timeout: float = _TIMEOUT_S) -> dict | None:
    """The running checkout against its branch on the remote.

    Compares with the current branch's upstream; a branch without one (a
    local feature branch) is compared with the remote's default branch.
    Returns {"update_available", "behind", "count_known", "branch",
    "tracking", "current", "local", "remote"} or None when it cannot tell.
    """
    if not enabled():
        return None
    head = _git(repo, "rev-parse", "HEAD")
    if not head:
        return None
    current = _git(repo, "rev-parse", "--abbrev-ref", "HEAD") or "HEAD"
    upstream = _git(repo, "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}")
    tracking = bool(upstream)
    if not upstream:
        upstream = _git(repo, "symbolic-ref", "-q", "--short", "refs/remotes/origin/HEAD") or "origin/main"
    remote, _, branch = upstream.partition("/")
    if not branch:
        return None
    line = _git(repo, "ls-remote", remote, f"refs/heads/{branch}", timeout=timeout)
    if not line:
        return None
    remote_sha = line.split()[0]
    status = {"update_available": False, "behind": 0, "count_known": True, "branch": upstream,
              "tracking": tracking, "current": current, "local": head[:7], "remote": remote_sha[:7]}
    if remote_sha == head:
        return status
    if _git(repo, "cat-file", "-e", f"{remote_sha}^{{commit}}") is None:
        # The remote has a commit this checkout has never seen: newer, but
        # how much newer is unknown without fetching (which this does not do).
        status.update(update_available=True, count_known=False)
        return status
    if _git(repo, "merge-base", "--is-ancestor", remote_sha, "HEAD") is not None:
        return status                                   # up to date, or ahead
    count = _git(repo, "rev-list", "--count", f"HEAD..{remote_sha}")
    status.update(update_available=True, behind=int(count) if count and count.isdigit() else 0,
                  count_known=bool(count and count.isdigit()))
    return status


def message(status: dict | None) -> str | None:
    """One line for the console and the UI, or None when there is nothing to say."""
    if not status or not status.get("update_available"):
        return None
    what = (f"{status['behind']} new commit{'s' if status['behind'] != 1 else ''}"
            if status.get("count_known") and status.get("behind") else "a newer version")
    head = (f"Update available: {what} on {status['branch']} "
            f"(running {status['local']}, GitHub has {status['remote']}).")
    if status.get("tracking"):
        return head + " Update with `git pull`, then restart."
    return (head + f" You are on branch {status.get('current')}; bring in "
            f"{status['branch']} when ready (`git fetch` and `git merge {status['branch']}`).")


def to_env(status: dict | None) -> str:
    """The launcher's result for the app: "null" means checked, nothing to report."""
    return json.dumps(status)


def from_env() -> tuple[bool, dict | None]:
    """(the launcher ran the check, its result)."""
    raw = os.environ.get(ENV_RESULT)
    if raw is None:
        return False, None
    try:
        value = json.loads(raw)
    except ValueError:
        return False, None
    return True, value if isinstance(value, dict) else None
