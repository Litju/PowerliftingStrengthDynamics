"""End-to-end CLI tests for the canonical dataset commands.

Every test points ``PSD_DATA_ROOT`` at pytest's own temporary directory, so the
suite never touches the maintainer's external data drive.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from psd.cli.canonical_cmd import app as canonical_app
from psd.cli.inspect_cmd import app as inspect_app
from psd.cli.main import app as root_app
from psd.cli.paths_cmd import app as paths_app
from psd.paths import DATA_ROOT_ENV_VAR, resolve_within_data_root
from psd.schema.registry import table_names

runner = CliRunner()

SOURCE_ID = "src_22222222222222222222222222222222"
ATHLETE_ID = "ath_11111111111111111111111111111111"
INGESTED_AT = datetime(2026, 1, 5, 9, 30, tzinfo=UTC)
SESSION_AT = datetime(2026, 3, 1, 18, 0, tzinfo=UTC)


@pytest.fixture(autouse=True)
def _data_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point every CLI test at an isolated data root."""
    root = tmp_path / "data"
    root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv(DATA_ROOT_ENV_VAR, str(root))
    return root


def _source() -> dict[str, Any]:
    return {
        "source_id": SOURCE_ID,
        "nature": "real",
        "regime": "psd_real",
        "origin_system": "hevy",
        "display_name": "Athlete export",
        "dataset_version": "2026-01",
        "snapshot_date": None,
        "snapshot_sha256": None,
        "license_id": "proprietary",
        "license_url": None,
        "consent_basis": "user_consent",
        "redistribution": "not_allowed",
        "ingested_at": INGESTED_AT,
        "notes": None,
    }


def _athlete() -> dict[str, Any]:
    return {
        "athlete_id": ATHLETE_ID,
        "pseudonym": "athlete-001",
        "identity_status": "single_source_verified",
        "ambiguity_group_id": None,
        "is_synthetic": False,
        "synthetic_regime": None,
        "sex_category_raw": "Male",
        "sex_category": "male",
        "birth_year": 1995,
        "country_code": "ES",
        "created_at": INGESTED_AT,
        "source_id": SOURCE_ID,
        "source_record_key": "athlete-001",
        "source_record_hash": None,
        "ingested_at": INGESTED_AT,
        "quality_flags": [],
        "missingness_reason": None,
    }


def _performed_session() -> dict[str, Any]:
    return {
        "performed_session_id": "pses_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "athlete_id": ATHLETE_ID,
        "planned_session_id": None,
        "started_at": SESSION_AT,
        "session_order_index": 0,
        "session_type": "training",
        "ended_at": None,
        "duration_seconds": None,
        "is_completed": None,
        "created_at": None,
        "scheduled_at": None,
        "performed_at": SESSION_AT,
        "observed_at": None,
        "modified_at": None,
        "source_id": SOURCE_ID,
        "source_record_key": "session-1",
        "source_record_hash": None,
        "ingested_at": INGESTED_AT,
        "quality_flags": [],
        "missingness_reason": "not_recorded_in_source",
    }


def _records_dir(tmp_path: Path, tables: dict[str, list[dict[str, Any]]]) -> Path:
    directory = tmp_path / "records"
    directory.mkdir(parents=True, exist_ok=True)
    for table, rows in tables.items():
        (directory / f"{table}.json").write_text(json.dumps(rows, default=str), encoding="utf-8")
    return directory


def test_root_cli_exposes_the_v0_command_surface() -> None:
    result = runner.invoke(root_app, ["--help"])
    assert result.exit_code == 0, result.output
    for command in ("schema", "validate", "canonical", "inspect", "provenance", "paths"):
        if command == "provenance":
            continue
        assert command in result.output


def test_schema_version_command() -> None:
    result = runner.invoke(root_app, ["schema", "version"])
    assert result.exit_code == 0, result.output
    assert "psd-canonical/0.2.0" in result.output


def test_schema_list_command() -> None:
    result = runner.invoke(root_app, ["schema", "list"])
    assert result.exit_code == 0, result.output
    assert "performed_set" in result.output
    assert "competition_attempt" in result.output


def test_schema_list_json() -> None:
    result = runner.invoke(root_app, ["schema", "list", "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert any(entry["table"] == "athlete" for entry in payload)


def test_schema_show_command() -> None:
    result = runner.invoke(root_app, ["schema", "show", "planned_set"])
    assert result.exit_code == 0, result.output
    assert "target_load_kg" in result.output
    assert "primary key" in result.output


def test_schema_show_unknown_table_fails() -> None:
    result = runner.invoke(root_app, ["schema", "show", "nope"])
    assert result.exit_code == 2


def test_schema_dump_writes_a_document(tmp_path: Path) -> None:
    out = tmp_path / "schema.json"
    result = runner.invoke(root_app, ["schema", "dump", "--out", str(out)])
    assert result.exit_code == 0, result.output
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "psd-canonical/0.2.0"
    assert len(payload["tables"]) == len(table_names())


def test_schema_vocabularies_command() -> None:
    result = runner.invoke(root_app, ["schema", "vocabularies", "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert "not_recorded_in_source" in payload["missingness_reason"]


@pytest.mark.usefixtures("_data_root")
def test_canonical_build_verify_inspect_validate(tmp_path: Path) -> None:
    records = _records_dir(
        tmp_path,
        {
            "source": [_source()],
            "athlete": [_athlete()],
            "performed_session": [_performed_session()],
        },
    )
    relative = "canonical/ds_cli"

    built = runner.invoke(
        canonical_app,
        [
            "build",
            "--input",
            str(records),
            "--output",
            relative,
            "--dataset-id",
            "ds_cli",
            "--name",
            "CLI fixture",
        ],
    )
    assert built.exit_code == 0, built.output
    assert "ds_cli" in built.output

    verified = runner.invoke(canonical_app, ["verify", relative])
    assert verified.exit_code == 0, verified.output
    assert "OK" in verified.output

    manifest = runner.invoke(canonical_app, ["manifest", relative])
    assert manifest.exit_code == 0, manifest.output
    payload = json.loads(manifest.output)
    assert payload["schema_version"] == "psd-canonical/0.2.0"
    assert len(payload["artifacts"]) == len(table_names())

    tables = runner.invoke(inspect_app, ["tables", relative])
    assert tables.exit_code == 0, tables.output
    assert "performed_session" in tables.output

    rows = runner.invoke(inspect_app, ["table", relative, "performed_session", "--json"])
    assert rows.exit_code == 0, rows.output
    session_rows = json.loads(rows.output)
    assert session_rows[0]["athlete_id"] == ATHLETE_ID

    timeline = runner.invoke(inspect_app, ["timeline", relative, "--athlete", ATHLETE_ID, "--json"])
    assert timeline.exit_code == 0, timeline.output
    events = json.loads(timeline.output)
    assert any(event["table"] == "performed_session" for event in events)

    validated = runner.invoke(root_app, ["validate", relative])
    assert validated.exit_code == 0, validated.output
    assert "0 error(s)" in validated.output


def test_canonical_build_rejects_unknown_kind(tmp_path: Path) -> None:
    records = _records_dir(tmp_path, {"source": [_source()]})
    result = runner.invoke(
        canonical_app,
        [
            "build",
            "--input",
            str(records),
            "--output",
            "canonical/x",
            "--dataset-id",
            "x",
            "--kind",
            "not_a_kind",
        ],
    )
    assert result.exit_code == 2


def test_canonical_build_reports_invalid_records(tmp_path: Path) -> None:
    records = _records_dir(tmp_path, {"source": [{"source_id": "only"}]})
    result = runner.invoke(
        canonical_app,
        ["build", "--input", str(records), "--output", "canonical/x", "--dataset-id", "x"],
    )
    assert result.exit_code == 2
    assert "violates" in result.output


def test_validate_reports_a_dangling_reference_and_exits_nonzero(tmp_path: Path) -> None:
    """A row pointing at an athlete that does not exist is an error, not a warning."""
    orphan = {**_performed_session(), "athlete_id": "ath_ffffffffffffffffffffffffffffffff"}
    records = _records_dir(
        tmp_path,
        {"source": [_source()], "athlete": [_athlete()], "performed_session": [orphan]},
    )
    relative = "canonical/ds_bad"
    built = runner.invoke(
        canonical_app,
        ["build", "--input", str(records), "--output", relative, "--dataset-id", "ds_bad"],
    )
    assert built.exit_code == 0, built.output

    validated = runner.invoke(root_app, ["validate", relative])
    assert validated.exit_code == 1
    assert "dangling_athlete" in validated.output

    as_json = runner.invoke(root_app, ["validate", relative, "--json"])
    payload = json.loads(as_json.output)
    assert payload["ok"] is False
    assert any(issue["code"] == "dangling_athlete" for issue in payload["issues"])


def test_validate_strict_mode_promotes_warnings(tmp_path: Path) -> None:
    """An undeclared absence is a warning, and strict mode turns it into an error."""
    row = {**_performed_session(), "duration_seconds": None, "missingness_reason": None}
    records = _records_dir(
        tmp_path,
        {"source": [_source()], "athlete": [_athlete()], "performed_session": [row]},
    )
    relative = "canonical/ds_warn"
    built = runner.invoke(
        canonical_app,
        ["build", "--input", str(records), "--output", relative, "--dataset-id", "ds_warn"],
    )
    assert built.exit_code == 0, built.output

    lenient = runner.invoke(root_app, ["validate", relative])
    assert lenient.exit_code == 0
    strict = runner.invoke(root_app, ["validate", relative, "--strict"])
    assert strict.exit_code == 1


@pytest.mark.usefixtures("_data_root")
def test_verify_fails_after_tampering(tmp_path: Path) -> None:
    records = _records_dir(tmp_path, {"source": [_source()], "athlete": [_athlete()]})
    relative = "canonical/ds_tamper"
    built = runner.invoke(
        canonical_app,
        ["build", "--input", str(records), "--output", relative, "--dataset-id", "ds_tamper"],
    )
    assert built.exit_code == 0, built.output

    target = resolve_within_data_root(Path(relative) / "tables" / "athlete.parquet")
    target.write_bytes(b"tampered")
    verified = runner.invoke(canonical_app, ["verify", relative])
    assert verified.exit_code == 1
    assert "PROBLEMS" in verified.output


def test_inspect_unknown_table_exits_nonzero(tmp_path: Path) -> None:
    records = _records_dir(tmp_path, {"source": [_source()]})
    relative = "canonical/ds_cli2"
    built = runner.invoke(
        canonical_app,
        ["build", "--input", str(records), "--output", relative, "--dataset-id", "ds_cli2"],
    )
    assert built.exit_code == 0, built.output
    result = runner.invoke(inspect_app, ["table", relative, "nope"])
    assert result.exit_code == 2


def test_paths_show_reports_the_configured_root(_data_root: Path) -> None:  # noqa: PT019
    result = runner.invoke(paths_app, ["show"])
    assert result.exit_code == 0, result.output
    assert str(_data_root) in result.output
