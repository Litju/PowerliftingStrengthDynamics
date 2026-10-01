"""External data-root boundary.

PSD never stores datasets inside the Git repository. Every artifact that is not
source code resolves through a single typed boundary: the ``PSD_DATA_ROOT``
environment variable (or an explicitly supplied path), always handled with
:class:`pathlib.Path`.

Rules enforced here:

* the data root must be an absolute path, so behavior never depends on the
  current working directory;
* every write path must resolve *inside* the data root, so a malformed
  relative path cannot escape the boundary;
* the canonical local layout is created on demand and never silently replaced.
"""

from __future__ import annotations

import os
from collections.abc import Iterator, Mapping
from enum import StrEnum
from pathlib import Path

__all__ = (
    "DATA_ROOT_ENV_VAR",
    "DataRootError",
    "DataRootEscapeError",
    "DataRootNotConfiguredError",
    "DataRootSubdirectory",
    "data_root_children",
    "ensure_data_root",
    "resolve_data_root",
    "resolve_within_data_root",
)

DATA_ROOT_ENV_VAR = "PSD_DATA_ROOT"


class DataRootError(RuntimeError):
    """Base class for data-root boundary failures."""


class DataRootNotConfiguredError(DataRootError):
    """Raised when the data-root boundary cannot be resolved."""


class DataRootEscapeError(DataRootError):
    """Raised when a requested path escapes the data-root boundary."""


class DataRootSubdirectory(StrEnum):
    """Canonical local layout under the external data root."""

    RAW = "raw"
    EXTERNAL = "external"
    CANONICAL = "canonical"
    SYNTHETIC = "synthetic"
    PROCESSED = "processed"
    MANIFESTS = "manifests"
    CACHE = "cache"
    RUNS = "runs"


def resolve_data_root(
    explicit: Path | str | None = None,
    *,
    env: Mapping[str, str] | None = None,
) -> Path:
    """Resolve the external data root.

    Args:
        explicit: An explicit data root. When given it wins over the
            environment; used by tests and by callers that already own a
            resolved boundary.
        env: Environment mapping to read instead of :data:`os.environ`.

    Returns:
        The absolute, resolved data-root path.

    Raises:
        DataRootNotConfiguredError: The boundary is unset or empty.
        DataRootError: The configured boundary is not an absolute path.
    """
    environment: Mapping[str, str] = os.environ if env is None else env
    candidate: Path | str | None = explicit
    if candidate is None:
        candidate = environment.get(DATA_ROOT_ENV_VAR)

    if candidate is None or (isinstance(candidate, str) and not candidate.strip()):
        msg = (
            f"External data boundary is not configured. Set the {DATA_ROOT_ENV_VAR} "
            "environment variable to the absolute path of your PSD data root, for "
            "example 'E:\\Data\\Databases\\PowerliftingStrengthDynamics' on Windows. "
            "The Git repository never stores datasets."
        )
        raise DataRootNotConfiguredError(msg)

    path = Path(candidate).expanduser()
    if not path.is_absolute():
        msg = (
            f"{DATA_ROOT_ENV_VAR} must be an absolute path; got {str(path)!r}. "
            "Relative roots make artifact locations depend on the current working "
            "directory."
        )
        raise DataRootError(msg)
    return path.resolve()


def resolve_within_data_root(
    relative: Path | str,
    *,
    data_root: Path | None = None,
    create: bool = False,
) -> Path:
    """Resolve *relative* inside the data root and enforce containment.

    Args:
        relative: Path relative to the data root.
        data_root: Explicit data root; resolved from the environment otherwise.
        create: Create the containing directories when ``True``.

    Returns:
        The absolute, resolved path.

    Raises:
        DataRootEscapeError: The resolved path is not inside the data root.
    """
    root = data_root if data_root is not None else resolve_data_root()
    root = root.resolve()
    candidate = Path(relative)
    if candidate.is_absolute():
        msg = (
            f"Refusing absolute path {str(candidate)!r} inside the data-root boundary; "
            "pass a path relative to the data root."
        )
        raise DataRootEscapeError(msg)
    target = (root / candidate).resolve()
    if target != root and not target.is_relative_to(root):
        msg = (
            f"Refusing path {str(relative)!r}: it resolves to {str(target)!r}, which "
            f"escapes the data root {str(root)!r}."
        )
        raise DataRootEscapeError(msg)
    if create:
        target.parent.mkdir(parents=True, exist_ok=True)
    return target


def data_root_children(data_root: Path | None = None) -> Iterator[tuple[str, Path]]:
    """Yield ``(name, path)`` for each canonical subdirectory."""
    root = data_root if data_root is not None else resolve_data_root()
    for subdirectory in DataRootSubdirectory:
        yield subdirectory.value, root / subdirectory.value


def ensure_data_root(
    data_root: Path | None = None,
    *,
    subdirectories: bool = True,
) -> Path:
    """Create the data root (and optionally the canonical layout) if absent.

    Returns:
        The resolved, existing data root.
    """
    root = data_root if data_root is not None else resolve_data_root()
    root = root.expanduser()
    if not root.is_absolute():
        msg = f"Data root must be an absolute path; got {str(root)!r}."
        raise DataRootError(msg)
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if subdirectories:
        for _name, path in data_root_children(root):
            path.mkdir(parents=True, exist_ok=True)
    return root
