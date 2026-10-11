"""
The startup check for a newer version on GitHub.

Each test builds a throwaway "GitHub" (a local bare repository) and a clone of
it, so nothing touches the network or this checkout. The properties that
matter: it is right about being behind, it stays silent when it cannot tell,
and it never moves anything in the checkout.
"""

import shutil
import subprocess

import pytest

from llm_agent.agent import update_check as uc

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")


def _git(cwd, *args):
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false",
                    *args], cwd=cwd, check=True, capture_output=True)


@pytest.fixture
def repos(tmp_path, monkeypatch):
    """(github, clone, pusher): a bare remote, the running checkout, and a
    second clone that publishes new commits."""
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(tmp_path / "gitconfig"))  # no personal config
    monkeypatch.delenv(uc.ENV_DISABLE, raising=False)
    github, clone, pusher = tmp_path / "github.git", tmp_path / "clone", tmp_path / "pusher"
    _git(tmp_path, "init", "--bare", "-b", "main", str(github))
    _git(tmp_path, "clone", str(github), str(pusher))
    _git(pusher, "commit", "--allow-empty", "-m", "first")
    _git(pusher, "push", "origin", "main")
    _git(tmp_path, "clone", str(github), str(clone))
    return github, clone, pusher


def _publish(pusher, n=1):
    for i in range(n):
        _git(pusher, "commit", "--allow-empty", "-m", f"new {i}")
    _git(pusher, "push", "origin", "main")


def test_up_to_date(repos):
    _, clone, _ = repos
    status = uc.check(clone)
    assert status["update_available"] is False and status["tracking"] is True
    assert uc.message(status) is None


def test_newer_on_github_without_fetching_anything(repos):
    _, clone, pusher = repos
    _publish(pusher, 2)
    refs_before = subprocess.run(["git", "show-ref"], cwd=clone, capture_output=True, text=True).stdout
    status = uc.check(clone)
    assert status["update_available"] and status["count_known"] is False
    assert "a newer version on origin/main" in uc.message(status)
    assert "`git pull`" in uc.message(status)
    # Read-only: no ref moved in the checkout.
    assert subprocess.run(["git", "show-ref"], cwd=clone, capture_output=True, text=True).stdout == refs_before


def test_counts_commits_when_they_are_already_fetched(repos):
    _, clone, pusher = repos
    _publish(pusher, 3)
    _git(clone, "fetch")
    status = uc.check(clone)
    assert status["update_available"] and status["behind"] == 3
    assert uc.message(status).startswith("Update available: 3 new commits on origin/main")


def test_ahead_is_not_an_update(repos):
    _, clone, _ = repos
    _git(clone, "commit", "--allow-empty", "-m", "local work")
    assert uc.check(clone)["update_available"] is False


def test_a_local_branch_is_compared_with_the_default_branch(repos):
    _, clone, pusher = repos
    _git(clone, "checkout", "-b", "phase3/sandbox")
    _publish(pusher)
    status = uc.check(clone)
    assert status["tracking"] is False and status["branch"] == "origin/main"
    text = uc.message(status)
    assert "You are on branch phase3/sandbox" in text and "git merge origin/main" in text


def test_silent_when_it_cannot_tell(repos, tmp_path, monkeypatch):
    _, clone, _ = repos
    assert uc.check(tmp_path / "not-a-repo-at-all") is None
    _git(clone, "remote", "set-url", "origin", str(tmp_path / "gone.git"))
    assert uc.check(clone) is None                      # offline / remote unreachable
    monkeypatch.setenv(uc.ENV_DISABLE, "0")
    assert uc.check(repos[1]) is None and not uc.enabled()


def test_the_launcher_hands_its_result_to_the_app(monkeypatch):
    status = {"update_available": True, "behind": 1, "count_known": True, "branch": "origin/main",
              "tracking": True, "current": "main", "local": "abc1234", "remote": "def5678"}
    monkeypatch.setenv(uc.ENV_RESULT, uc.to_env(status))
    assert uc.from_env() == (True, status)
    monkeypatch.setenv(uc.ENV_RESULT, uc.to_env(None))
    assert uc.from_env() == (True, None)                # checked, nothing to report
    monkeypatch.delenv(uc.ENV_RESULT)
    assert uc.from_env() == (False, None)               # not checked: the app asks itself
    assert uc.message(status) == ("Update available: 1 new commit on origin/main (running abc1234, "
                                  "GitHub has def5678). Update with `git pull`, then restart.")


def test_the_launcher_never_waits_when_disabled(monkeypatch):
    import start
    monkeypatch.setenv(uc.ENV_DISABLE, "off")
    assert start.UpdateCheck().result(wait_s=5) == (False, None)


def test_git_is_never_allowed_to_prompt(monkeypatch):
    """Without a GitHub login (a private repository, or an SSH clone without a
    key) git would ask for a username or passphrase and hang the start. It
    must fail at once instead, and the check then says nothing."""
    seen = {}

    def fake_run(cmd, **kwargs):
        seen.update(kwargs)
        return subprocess.CompletedProcess(cmd, 128, "", "fatal: terminal prompts disabled")

    monkeypatch.setattr(uc.subprocess, "run", fake_run)
    assert uc._git(".", "ls-remote", "origin") is None
    env = seen["env"]
    assert env["GIT_TERMINAL_PROMPT"] == "0" and env["GCM_INTERACTIVE"] == "never"
    assert "BatchMode=yes" in env["GIT_SSH_COMMAND"]
    assert seen["stdin"] is subprocess.DEVNULL and seen["timeout"] <= 10
