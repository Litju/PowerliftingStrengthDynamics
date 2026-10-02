"""Tests for the ``psd paths`` boundary command."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from psd.cli.main import app as root_app
from psd.cli.paths_cmd import app
from psd.paths import DATA_ROOT_ENV_VAR, ensure_data_root

pytestmark = pytest.mark.windows_parity

runner = CliRunner()


def test_show_reports_resolved_root_and_layout(tmp_path: Path) -> None:
    ensure_data_root(tmp_path)
    result = runner.invoke(app, ["show", "--data-root", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert DATA_ROOT_ENV_VAR in result.output
    for name in ("raw", "canonical", "synthetic", "manifests", "runs"):
        assert name in result.output


def test_show_marks_uncreated_subdirectories(tmp_path: Path) -> None:
    result = runner.invoke(app, ["show", "--data-root", str(tmp_path / "fresh")])
    assert result.exit_code == 0, result.output
    assert "not created" in result.output


def test_show_json_output_is_machine_readable(tmp_path: Path) -> None:
    result = runner.invoke(app, ["show", "--data-root", str(tmp_path), "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["data_root"] == str(tmp_path.resolve())
    assert payload["environment_variable"] == DATA_ROOT_ENV_VAR
    assert {entry["name"] for entry in payload["layout"]} >= {"raw", "canonical"}


def test_missing_boundary_exits_with_code_two(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(DATA_ROOT_ENV_VAR, raising=False)
    result = runner.invoke(app, ["show"])
    assert result.exit_code == 2
    assert "not configured" in result.output


def test_paths_group_is_exposed_from_root_cli() -> None:
    result = runner.invoke(root_app, ["paths", "--help"])
    assert result.exit_code == 0, result.output
    assert "PSD data root" in result.output
