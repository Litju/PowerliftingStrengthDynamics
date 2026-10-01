"""Tests for the external data-root boundary."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from psd.config import get_settings, reset_settings_cache
from psd.paths import (
    DATA_ROOT_ENV_VAR,
    DataRootEscapeError,
    DataRootNotConfiguredError,
    DataRootSubdirectory,
    ensure_data_root,
    resolve_data_root,
    resolve_within_data_root,
)


def test_resolves_explicit_path_without_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(DATA_ROOT_ENV_VAR, raising=False)
    assert resolve_data_root(tmp_path) == tmp_path.resolve()


def test_resolves_from_environment(tmp_path: Path) -> None:
    assert resolve_data_root(env={DATA_ROOT_ENV_VAR: str(tmp_path)}) == tmp_path.resolve()


def test_unconfigured_boundary_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(DATA_ROOT_ENV_VAR, raising=False)
    with pytest.raises(DataRootNotConfiguredError, match="not configured"):
        resolve_data_root()


def test_blank_environment_value_raises() -> None:
    with pytest.raises(DataRootNotConfiguredError):
        resolve_data_root(env={DATA_ROOT_ENV_VAR: "   "})


def test_relative_root_is_rejected() -> None:
    with pytest.raises(Exception, match="absolute"):
        resolve_data_root(env={DATA_ROOT_ENV_VAR: "relative/dir"})


def test_settings_require_absolute_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(DATA_ROOT_ENV_VAR, str(tmp_path))
    reset_settings_cache()
    assert get_settings().data_root == tmp_path


def test_within_data_root_joins_and_contains(tmp_path: Path) -> None:
    target = resolve_within_data_root(Path("canonical") / "athlete", data_root=tmp_path)
    assert target == (tmp_path.resolve() / "canonical" / "athlete")
    assert target.is_relative_to(tmp_path.resolve())


def test_within_data_root_rejects_parent_traversal(tmp_path: Path) -> None:
    with pytest.raises(DataRootEscapeError, match="escapes the data root"):
        resolve_within_data_root(Path("canonical") / ".." / ".." / "outside", data_root=tmp_path)


def test_within_data_root_rejects_absolute_paths(tmp_path: Path) -> None:
    # The escape has to be absolute on the platform running the test: a literal
    # "C:/secrets" is absolute on Windows but relative on Linux.
    outside = tmp_path.parent / "secrets"
    assert outside.is_absolute()

    with pytest.raises(DataRootEscapeError, match="absolute"):
        resolve_within_data_root(outside, data_root=tmp_path)


def test_create_flag_materializes_parent(tmp_path: Path) -> None:
    target = resolve_within_data_root(
        Path("canonical") / "ds" / "performed_set.parquet", data_root=tmp_path, create=True
    )
    assert target.parent.is_dir()


def test_ensure_data_root_creates_canonical_layout(tmp_path: Path) -> None:
    root = ensure_data_root(tmp_path / "psd")
    for subdirectory in DataRootSubdirectory:
        assert (root / subdirectory.value).is_dir()


def test_ensure_data_root_does_not_depend_on_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    os.environ[DATA_ROOT_ENV_VAR] = str(tmp_path / "elsewhere")
    try:
        root = ensure_data_root()
    finally:
        del os.environ[DATA_ROOT_ENV_VAR]
    assert root == (tmp_path / "elsewhere").resolve()


def test_test_suite_never_requires_machine_specific_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PSD tests must not require the maintainer's external data directory."""
    monkeypatch.delenv(DATA_ROOT_ENV_VAR, raising=False)
    resolved = resolve_data_root(tmp_path)
    assert str(resolved).startswith(str(tmp_path))
