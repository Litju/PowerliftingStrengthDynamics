"""Execution-environment and code-state capture for provenance manifests.

Artifacts must be traceable to the code and dependency lock that produced them.
The helpers here are deliberately read-only and best-effort: when Git metadata
is unavailable (an sdist, a wheel, a copied tree), the fact is recorded as
``None`` rather than guessed.
"""

from __future__ import annotations

import hashlib
import platform
import subprocess
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as distribution_version
from pathlib import Path

__all__ = (
    "DISTRIBUTION_NAME",
    "EnvironmentSnapshot",
    "GitState",
    "detect_git_state",
    "lockfile_fingerprint",
    "runtime_snapshot",
)

DISTRIBUTION_NAME = "powerlifting-strength-dynamics"

_GIT_TIMEOUT_SECONDS = 10


@dataclass(frozen=True, slots=True)
class GitState:
    """Code state of the working tree that produced an artifact.

    Attributes:
        commit: Full commit SHA, or ``None`` when unavailable.
        is_dirty: Whether the tree had uncommitted changes, or ``None`` when
            unknown. A dirty tree means the artifact is not reproducible from
            the commit alone.
    """

    commit: str | None
    is_dirty: bool | None


@dataclass(frozen=True, slots=True)
class EnvironmentSnapshot:
    """Platform, interpreter, package, and lock state for a run.

    Attributes:
        python_version: Interpreter version string.
        python_implementation: CPython, PyPy, and so on.
        platform: ``platform.platform()`` string.
        package_version: Installed PSD distribution version.
        lockfile_sha256: SHA-256 of ``uv.lock``, or ``None`` when absent.
        code_commit: Git commit SHA, or ``None`` when unavailable.
        is_dirty_tree: Working-tree dirtiness, or ``None`` when unknown.
    """

    python_version: str
    python_implementation: str
    platform: str
    package_version: str | None
    lockfile_sha256: str | None
    code_commit: str | None
    is_dirty_tree: bool | None


def _git(args: list[str], *, cwd: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            check=False,
            timeout=_GIT_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def detect_git_state(cwd: Path | None = None) -> GitState:
    """Return the Git state of the repository containing *cwd*.

    Args:
        cwd: Directory to inspect; defaults to the current working directory.

    Returns:
        A :class:`GitState`. Unavailable Git metadata yields ``(None, None)``
        rather than an error.
    """
    root = cwd if cwd is not None else Path.cwd()
    commit = _git(["rev-parse", "HEAD"], cwd=root)
    if commit is None:
        return GitState(commit=None, is_dirty=None)
    status = _git(["status", "--porcelain"], cwd=root)
    if status is None:
        return GitState(commit=commit, is_dirty=None)
    return GitState(commit=commit, is_dirty=bool(status))


def lockfile_fingerprint(lockfile: Path) -> str | None:
    """Return the SHA-256 of a dependency lockfile, or ``None`` when absent."""
    if not lockfile.is_file():
        return None
    return hashlib.sha256(lockfile.read_bytes()).hexdigest()


def runtime_snapshot(
    *,
    lockfile: Path | None = None,
    repo_root: Path | None = None,
) -> EnvironmentSnapshot:
    """Capture the environment needed to interpret and reproduce an artifact."""
    try:
        package_version: str | None = distribution_version(DISTRIBUTION_NAME)
    except PackageNotFoundError:  # pragma: no cover - package is always installed
        package_version = None

    root = repo_root if repo_root is not None else Path.cwd()
    git_state = detect_git_state(root)
    lock_sha256 = lockfile_fingerprint(lockfile) if lockfile is not None else None

    return EnvironmentSnapshot(
        python_version=platform.python_version(),
        python_implementation=platform.python_implementation(),
        platform=platform.platform(),
        package_version=package_version,
        lockfile_sha256=lock_sha256,
        code_commit=git_state.commit,
        is_dirty_tree=git_state.is_dirty,
    )
