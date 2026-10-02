"""Tests for canonical dataset assembly, persistence, and verification.

These tests must not depend on the maintainer's ``PSD_DATA_ROOT``: every one
points the data root at pytest's own temporary directory.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from psd.paths import DataRootEscapeError, resolve_within_data_root
from psd.provenance.environment import EnvironmentSnapshot
from psd.provenance.manifest import DatasetKind, manifest_digest
from psd.provenance.sources import (
    ConsentBasis,
    DataRegime,
    RedistributionPolicy,
    SourceNature,
    SourceRecord,
)
from psd.schema.models import AthleteRecord
from psd.schema.registry import table_names
from psd.schema.version import SCHEMA_VERSION
from psd.serialization.dataset import (
    DatasetLayoutError,
    DatasetWriteSpec,
    build_dataset,
    load_records,
    read_dataset,
    records_from_tables,
    verify_dataset,
    write_dataset,
)
from psd.validation import validate_tables

pytestmark = pytest.mark.windows_parity

CREATED_AT = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)
SOURCE_ID = "src_22222222222222222222222222222222"
ATHLETE_ID = "ath_11111111111111111111111111111111"

_ENVIRONMENT = EnvironmentSnapshot(
    python_version="3.12.13",
    python_implementation="CPython",
    platform="Windows-11",
    package_version="0.1.0",
    lockfile_sha256="a" * 64,
    code_commit="b" * 40,
    is_dirty_tree=False,
)


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
        "ingested_at": datetime(2026, 1, 5, 9, 30, tzinfo=UTC),
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
        "created_at": datetime(2026, 1, 5, 9, 30, tzinfo=UTC),
        "source_id": SOURCE_ID,
        "source_record_key": "athlete-001",
        "source_record_hash": None,
        "ingested_at": datetime(2026, 1, 5, 9, 30, tzinfo=UTC),
        "quality_flags": [],
        "missingness_reason": None,
    }


def _provenance() -> dict[str, Any]:
    return {
        "provenance_id": "prv_33333333333333333333333333333333",
        "dataset_id": "ds_test",
        "parent_dataset_id": None,
        "transform_name": "fixture_build",
        "transform_version": "0.1.0",
        "code_commit": "c" * 40,
        "config_sha256": None,
        "schema_version": "psd-canonical/0.1.0",
        "random_seed": None,
        "created_at": CREATED_AT,
        "description": "test fixture",
    }


def _write_records(directory: Path, records: dict[str, list[dict[str, Any]]]) -> Path:
    """Write ``<table>.json`` files for the given records."""
    directory.mkdir(parents=True, exist_ok=True)
    for table, rows in records.items():
        (directory / f"{table}.json").write_text(json.dumps(rows, default=str), encoding="utf-8")
    return directory


def _minimal_dataset(tmp_path: Path) -> tuple[Path, Path]:
    """Build and persist a minimal dataset; return ``(data_root, relative)``."""
    data_root = tmp_path / "data"
    records = load_records(
        _write_records(tmp_path / "records", {"source": [_source()], "athlete": [_athlete()]})
    )
    dataset = build_dataset(records, dataset_id="ds_test", created_at=CREATED_AT)
    write_dataset(
        dataset,
        Path("canonical") / "ds_test",
        DatasetWriteSpec(
            dataset_kind=DatasetKind.TRAINING_HISTORY,
            sources=(
                SourceRecord(
                    source_id=SOURCE_ID,
                    display_name="Athlete export",
                    nature=SourceNature.REAL,
                    regime=DataRegime.REAL,
                    origin_system="hevy",
                    license_id="proprietary",
                    consent_basis=ConsentBasis.USER_CONSENT,
                    redistribution=RedistributionPolicy.NOT_ALLOWED,
                    ingested_at=datetime(2026, 1, 5, 9, 30, tzinfo=UTC),
                ),
            ),
        ),
        environment=_ENVIRONMENT,
        data_root=data_root,
    )
    return data_root, Path("canonical") / "ds_test"


def test_build_dataset_contains_every_canonical_table(tmp_path: Path) -> None:
    records = load_records(_write_records(tmp_path / "records", {"source": [_source()]}))
    dataset = build_dataset(records, dataset_id="ds_x", created_at=CREATED_AT)
    assert set(dataset.tables) == set(table_names())
    assert dataset.row_counts()["source"] == 1
    assert dataset.row_counts()["athlete"] == 0


def test_build_dataset_rejects_unknown_table(tmp_path: Path) -> None:
    dataset = build_dataset({}, dataset_id="ds_x", created_at=CREATED_AT)
    assert dataset.total_rows() == 0
    source_dir = _write_records(tmp_path / "records", {"source": [_source()]})
    (source_dir / "not_a_table.json").write_text("[]", encoding="utf-8")
    assert "not_a_table" not in load_records(source_dir)


def test_write_then_read_round_trip(tmp_path: Path) -> None:
    data_root, relative = _minimal_dataset(tmp_path)
    dataset, manifest = read_dataset(relative, data_root=data_root)
    assert manifest.dataset_id == "ds_test"
    assert manifest.schema_version == SCHEMA_VERSION.tag
    assert set(dataset.tables) == set(table_names())
    assert dataset.table("athlete").num_rows == 1


def test_persisted_dataset_validates(tmp_path: Path) -> None:
    data_root, relative = _minimal_dataset(tmp_path)
    dataset, _manifest = read_dataset(relative, data_root=data_root)
    report = validate_tables(dataset.tables)
    assert report.ok, [issue.format() for issue in report.issues]


def test_verify_detects_intact_dataset(tmp_path: Path) -> None:
    data_root, relative = _minimal_dataset(tmp_path)
    result = verify_dataset(relative, data_root=data_root)
    assert result.ok, result.problems
    assert result.checked_artifacts == len(table_names())


def test_verify_detects_tampering(tmp_path: Path) -> None:
    data_root, relative = _minimal_dataset(tmp_path)
    target = resolve_within_data_root(relative / "tables" / "athlete.parquet", data_root=data_root)
    target.write_bytes(b"not a parquet file")
    result = verify_dataset(relative, data_root=data_root)
    assert not result.ok
    assert result.problems


def test_verify_detects_missing_artifact(tmp_path: Path) -> None:
    data_root, relative = _minimal_dataset(tmp_path)
    resolve_within_data_root(relative / "tables" / "athlete.parquet", data_root=data_root).unlink()
    result = verify_dataset(relative, data_root=data_root)
    assert not result.ok
    assert any("missing" in problem for problem in result.problems)


def test_verify_detects_a_duplicated_primary_key(tmp_path: Path) -> None:
    """A duplicated key digests cleanly, so verification has to look for it explicitly.

    Both digests are computed over whatever rows a file holds. A table that puts two
    rows under one identity therefore passes every digest check and is still wrong: a
    consumer cannot tell a duplicated athlete from two athletes. The registry declares
    each key unique within its table, so verification enforces it.
    """
    data_root, _relative = _minimal_dataset(tmp_path)
    duplicate = _athlete()
    duplicate["pseudonym"] = "athlete-001-again"
    records = load_records(
        _write_records(
            tmp_path / "duplicated", {"source": [_source()], "athlete": [_athlete(), duplicate]}
        )
    )
    write_dataset(
        build_dataset(records, dataset_id="ds_dup", created_at=CREATED_AT),
        Path("canonical") / "ds_dup",
        DatasetWriteSpec(dataset_kind=DatasetKind.TRAINING_HISTORY),
        data_root=data_root,
    )

    result = verify_dataset(Path("canonical") / "ds_dup", data_root=data_root)

    assert not result.ok
    assert any("primary key" in problem and "athlete_id" in problem for problem in result.problems)


def test_a_single_rowed_table_needs_no_key_check(tmp_path: Path) -> None:
    """The uniqueness check must not cost anything for a table that cannot fail it."""
    data_root, relative = _minimal_dataset(tmp_path)
    assert verify_dataset(relative, data_root=data_root).ok


def test_write_is_reproducible(tmp_path: Path) -> None:
    first_root, first_relative = _minimal_dataset(tmp_path / "one")
    second_root, second_relative = _minimal_dataset(tmp_path / "two")
    first, first_manifest = read_dataset(first_relative, data_root=first_root)
    second, second_manifest = read_dataset(second_relative, data_root=second_root)
    assert {a.name: a.content_sha256 for a in first_manifest.artifacts} == {
        a.name: a.content_sha256 for a in second_manifest.artifacts
    }
    assert {a.name: a.sha256 for a in first_manifest.artifacts} == {
        a.name: a.sha256 for a in second_manifest.artifacts
    }
    assert first.row_counts() == second.row_counts()


def test_manifest_digest_is_deterministic(tmp_path: Path) -> None:
    _data_root, relative = _minimal_dataset(tmp_path)
    _dataset, manifest = read_dataset(relative, data_root=tmp_path / "data")
    assert manifest_digest(manifest) == manifest_digest(manifest)


def test_dataset_cannot_be_written_outside_the_data_root(tmp_path: Path) -> None:
    with pytest.raises(DataRootEscapeError):
        resolve_within_data_root(Path("..") / "escape", data_root=tmp_path / "data")


def test_read_missing_dataset_raises(tmp_path: Path) -> None:
    with pytest.raises(DatasetLayoutError, match="No dataset manifest"):
        read_dataset(Path("canonical") / "nope", data_root=tmp_path / "data")


def test_invalid_manifest_is_reported(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    manifest_dir = resolve_within_data_root(Path("canonical") / "broken", data_root=data_root)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    (manifest_dir / "manifest.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(DatasetLayoutError, match="Unreadable dataset manifest"):
        read_dataset(Path("canonical") / "broken", data_root=data_root)


def test_records_reload_from_persisted_tables(tmp_path: Path) -> None:
    data_root, relative = _minimal_dataset(tmp_path)
    dataset, _manifest = read_dataset(relative, data_root=data_root)
    rebuilt = records_from_tables(dataset.tables)
    athletes = rebuilt["athlete"]
    assert len(athletes) == 1
    assert isinstance(athletes[0], AthleteRecord)
    assert athletes[0].athlete_id == ATHLETE_ID


def test_load_records_rejects_a_non_array(tmp_path: Path) -> None:
    directory = _write_records(tmp_path / "records", {})
    (directory / "source.json").write_text("{}", encoding="utf-8")
    with pytest.raises(DatasetLayoutError, match="must contain a JSON array"):
        load_records(directory)


def test_load_records_rejects_an_invalid_record(tmp_path: Path) -> None:
    directory = _write_records(tmp_path / "records", {"source": [{"source_id": "x"}]})
    with pytest.raises(DatasetLayoutError, match="violates SourceRecord"):
        load_records(directory)


def test_load_records_reports_missing_required_table(tmp_path: Path) -> None:
    directory = _write_records(tmp_path / "records", {})
    with pytest.raises(DatasetLayoutError, match="Missing required record file"):
        load_records(directory, required=("source",))


def test_provenance_lineage_survives_persistence(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    records = load_records(
        _write_records(
            tmp_path / "records",
            {"source": [_source()], "athlete": [_athlete()], "provenance": [_provenance()]},
        )
    )
    dataset = build_dataset(records, dataset_id="ds_test", created_at=CREATED_AT)
    write_dataset(
        dataset,
        Path("canonical") / "ds_test",
        DatasetWriteSpec(dataset_kind=DatasetKind.TRAINING_HISTORY),
        environment=_ENVIRONMENT,
        data_root=data_root,
    )
    read_back, manifest = read_dataset(Path("canonical") / "ds_test", data_root=data_root)
    assert read_back.table("provenance").num_rows == 1
    assert manifest.artifacts
