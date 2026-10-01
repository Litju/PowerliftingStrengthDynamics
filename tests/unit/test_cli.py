"""CLI smoke tests."""

from __future__ import annotations

from typer.testing import CliRunner

from psd import __version__
from psd.cli.main import app

runner = CliRunner()


def test_help_lists_commands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0, result.output
    assert "Powerlifting Strength Dynamics" in result.output


def test_version_command_prints_package_version() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0, result.output
    assert result.output.strip() == __version__


def test_bare_invocation_shows_help() -> None:
    result = runner.invoke(app, [])
    assert "Usage" in result.output


def test_unknown_command_fails_without_traceback() -> None:
    result = runner.invoke(app, ["definitely-not-a-command"])
    assert result.exit_code != 0
    assert "Traceback" not in result.output
