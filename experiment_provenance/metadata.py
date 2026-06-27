from __future__ import annotations

import subprocess
from pathlib import Path


def _git(repo_root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo_root), *args],
        text=True,
    ).strip()


def get_git_metadata(repo_root: Path | None = None) -> dict[str, str | bool | None]:
    repo_root = Path(repo_root or Path.cwd()).resolve()
    commit = _git(repo_root, "rev-parse", "HEAD")
    branch = _git(repo_root, "rev-parse", "--abbrev-ref", "HEAD")
    status = _git(repo_root, "status", "--porcelain")
    remote_url = None
    try:
        remote_url = _git(repo_root, "remote", "get-url", "origin")
    except subprocess.CalledProcessError:
        remote_url = None
    return {
        "code_commit": commit,
        "short_commit": commit[:7],
        "branch": branch,
        "dirty": bool(status),
        "remote_url": remote_url,
    }
