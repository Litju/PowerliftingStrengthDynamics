"""Tests for provenance sources, manifests, and environment capture."""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from psd.provenance.environment import (
    EnvironmentSnapshot,
    GitState,
    detect_git_state,
    lockfile_fingerprint,
    runtime_snapshot,
)
from psd.provenance.manifest import (
    ArtifactRef,
    DatasetKind,
    DatasetManifest,
    LineageEntry,
    manifest_digest,
)
from psd.provenance.sources import (
    ConsentBasis,
    DataRegime,
    RedistributionPolicy,
    SourceNature,
    SourceRecord,
)

INGESTED_AT = datetime(2026, 1, 5, 9, 30, tzinfo=UTC)


def _source(**overrides: object) -> SourceRecord:
    payload: dict[str, object] = {
        "source_id": "hevy_export_2024",
        "display_name": "Athlete training export",
        "nature": SourceNature.REAL,
        "regime": DataRegime.REAL,
        "origin_system": "hevy",
        "license_id": "proprietary",
        "consent_basis": ConsentBasis.USER_CONSENT,
        "redistribution": RedistributionPolicy.NOT_ALLOWED,
        "ingested_at": INGESTED_AT,
    }
    payload.update(overrides)
    return SourceRecord(**payload)  # type: ignore[arg-type]


def _environment(**overrides: object) -> EnvironmentSnapshot:
    payload: dict[str, object] = {
        "python_version": "3.12.13",
        "python_implementation": "CPython",
        "platform": "Windows-11",
        "package_version": "0.1.0",
        "lockfile_sha256": "a" * 64,
        "code_commit": "b" * 40,
        "is_dirty_tree": False,
    }
    payload.update(overrides)
    return EnvironmentSnapshot(**payload)  # type: ignore[arg-type]


def _artifact(name: str) -> ArtifactRef:
    return ArtifactRef(
        name=name,
        relative_path=f"{name}.parquet",
        sha256="c" * 64,
        content_sha256="d" * 64,
        byte_size=1024,
        row_count=3,
    )


def _manifest(**overrides: object) -> DatasetManifest:
    payload: dict[str, object] = {
        "dataset_id": "ds_athlete_001",
        "dataset_name": "Athlete 001 canonical history",
        "dataset_kind": DatasetKind.TRAINING_HISTORY,
        "schema_version": "psd-canonical/0.1.0",
        "created_at": INGESTED_AT,
        "environment": _environment(),
        "sources": (_source(),),
        "artifacts": (_artifact("athlete"), _artifact("performed_set")),
        "lineage": (
            LineageEntry(
                provenance_id="lin_1",
                dataset_id="ds_athlete_001",
                transform_name="canonicalize",
                transform_version="0.1.0",
                schema_version="psd-canonical/0.1.0",
                created_at=INGESTED_AT,
            ),
        ),
    }
    payload.update(overrides)
    return DatasetManifest(**payload)  # type: ignore[arg-type]


def test_real_source_requires_explicit_consent_basis() -> None:
    with pytest.raises(ValidationError, match="consent basis"):
        _source(consent_basis=ConsentBasis.NONE_DECLARED)


def test_synthetic_source_must_declare_sim_regime() -> None:
    with pytest.raises(ValidationError, match="synthetic sources must declare"):
        _source(nature=SourceNature.SYNTHETIC, regime=DataRegime.REAL)


def test_real_source_cannot_declare_sim_regime() -> None:
    with pytest.raises(ValidationError, match=re.escape("regime 'psd_sim' but is real")):
        _source(regime=DataRegime.SIM)


def test_real_source_cannot_declare_reference_regime() -> None:
    with pytest.raises(ValidationError, match=re.escape("regime 'psd_reference' but is real")):
        _source(regime=DataRegime.REFERENCE)


def test_reference_source_must_declare_reference_regime() -> None:
    """Authored vocabulary is its own regime, not a simulator regime."""
    with pytest.raises(ValidationError, match="reference sources must declare"):
        _source(
            nature=SourceNature.REFERENCE,
            regime=DataRegime.SIM,
            consent_basis=ConsentBasis.NONE_DECLARED,
            license_id="apache-2.0",
            redistribution=RedistributionPolicy.ALLOWED,
        )


def test_synthetic_source_cannot_declare_reference_regime() -> None:
    with pytest.raises(ValidationError, match="synthetic sources must declare"):
        _source(
            nature=SourceNature.SYNTHETIC,
            regime=DataRegime.REFERENCE,
            consent_basis=ConsentBasis.PUBLIC_LICENSE,
            license_id="apache-2.0",
            redistribution=RedistributionPolicy.ALLOWED,
        )


def _reference_source() -> SourceRecord:
    return _source(  # type: ignore[return-value]
        source_id="src_psd_exercise_ontology",
        display_name="PSD exercise ontology registry",
        nature=SourceNature.REFERENCE,
        regime=DataRegime.REFERENCE,
        origin_system="psd-ontology",
        license_id="apache-2.0",
        consent_basis=ConsentBasis.NONE_DECLARED,
        redistribution=RedistributionPolicy.ALLOWED,
    )


def test_reference_provenance_describes_no_athlete() -> None:
    """A reference source needs no athlete consent, because it holds no athletes."""
    record = _reference_source()
    assert record.nature is SourceNature.REFERENCE
    assert record.regime is DataRegime.REFERENCE
    assert record.consent_basis is ConsentBasis.NONE_DECLARED


def test_reference_and_simulator_regimes_stay_distinguishable() -> None:
    """The point of the third nature: "generated" no longer means "not real"."""
    simulator = _source(
        source_id="psd_sim_gen",
        nature=SourceNature.SYNTHETIC,
        regime=DataRegime.SIM,
        consent_basis=ConsentBasis.PUBLIC_LICENSE,
        redistribution=RedistributionPolicy.ALLOWED,
        license_id="apache-2.0",
    )
    assert simulator.nature is not _reference_source().nature
    assert simulator.regime is not _reference_source().regime


def test_non_redistributable_source_requires_license() -> None:
    with pytest.raises(ValidationError, match="declares no license"):
        _source(license_id=None)


def test_naive_ingested_at_is_rejected() -> None:
    with pytest.raises(ValidationError, match="naive timestamp"):
        _source(ingested_at=datetime(2026, 1, 5, 9, 30))


def test_uppercase_snapshot_digest_is_rejected() -> None:
    with pytest.raises(ValidationError, match="lowercase hex"):
        _source(snapshot_sha256="E" * 64)


def test_synthetic_dataset_cannot_cite_real_source() -> None:
    with pytest.raises(ValidationError, match="synthetic dataset cannot cite real"):
        _manifest(dataset_kind=DatasetKind.SYNTHETIC)


def test_real_dataset_cannot_cite_synthetic_source() -> None:
    synthetic = _source(
        source_id="psd_sim_gen",
        nature=SourceNature.SYNTHETIC,
        regime=DataRegime.SIM,
        consent_basis=ConsentBasis.PUBLIC_LICENSE,
        redistribution=RedistributionPolicy.ALLOWED,
        license_id="apache-2.0",
    )
    with pytest.raises(ValidationError, match="cannot cite synthetic sources"):
        _manifest(sources=(synthetic,))


def test_ingestion_dataset_cannot_cite_reference_vocabulary() -> None:
    """Authored vocabulary is not an ingested observation source.

    Rows derived from the registry are re-stamped on ingest with the ingest source, so
    an athlete-history dataset citing the ontology would be claiming an external
    ingestion it never performed.
    """
    with pytest.raises(ValidationError, match="cannot cite reference sources"):
        _manifest(sources=(_source(), _reference_source()))


def test_reference_dataset_may_not_cite_even_reference_sources() -> None:
    """A reference artifact is self-describing, so it cites nothing at all."""
    with pytest.raises(ValidationError, match="reference dataset cites no sources"):
        _manifest(dataset_kind=DatasetKind.REFERENCE, sources=(_reference_source(),))


def test_mixed_real_and_synthetic_dataset_is_rejected() -> None:
    synthetic = _source(
        source_id="psd_sim_gen",
        nature=SourceNature.SYNTHETIC,
        regime=DataRegime.SIM,
        consent_basis=ConsentBasis.PUBLIC_LICENSE,
        redistribution=RedistributionPolicy.ALLOWED,
        license_id="apache-2.0",
    )
    with pytest.raises(ValidationError, match="Mixed real/synthetic datasets are rejected"):
        _manifest(dataset_kind=DatasetKind.MIXED, sources=(_source(), synthetic))


def test_duplicate_source_ids_are_rejected() -> None:
    with pytest.raises(ValidationError, match="Duplicate source_id"):
        _manifest(sources=(_source(), _source()))


@pytest.mark.parametrize(
    "candidate",
    [
        "C:/data/athlete.parquet",
        "/data/athlete.parquet",
        r"\\server\share\athlete.parquet",
        r"tables\performed_set.parquet",
    ],
)
def test_artifact_path_must_be_relative(candidate: str) -> None:
    """The same manifest must be rejected everywhere, not just on Windows.

    ``C:/data/athlete.parquet`` is absolute on Windows and relative on Linux, so a
    check that relied on ``Path.is_absolute`` alone would accept it on one
    platform and refuse it on the other.
    """
    with pytest.raises(ValidationError, match="relative POSIX path"):
        ArtifactRef(
            name="athlete",
            relative_path=candidate,
            sha256="c" * 64,
            content_sha256="d" * 64,
            byte_size=1,
            row_count=1,
        )


@pytest.mark.parametrize(
    "candidate",
    [
        "athlete.parquet",
        "tables/athlete.parquet",
        "tables/nested/deeper/athlete.parquet",
    ],
)
def test_artifact_path_accepts_portable_relative_paths(candidate: str) -> None:
    reference = ArtifactRef(
        name="athlete",
        relative_path=candidate,
        sha256="c" * 64,
        content_sha256="d" * 64,
        byte_size=1,
        row_count=1,
    )

    assert reference.relative_path == candidate


def test_manifest_json_is_deterministic_and_parseable() -> None:
    manifest = _manifest()
    first = manifest.to_json_bytes()
    second = manifest.to_json_bytes()
    assert first == second
    payload = json.loads(first)
    assert payload["schema_version"] == "psd-canonical/0.1.0"
    assert payload["sources"][0]["nature"] == "real"
    assert manifest_digest(manifest) == manifest_digest(manifest)


def test_manifest_lookup_helpers() -> None:
    manifest = _manifest()
    assert manifest.source("hevy_export_2024").origin_system == "hevy"
    assert manifest.artifact("performed_set").row_count == 3
    with pytest.raises(KeyError):
        manifest.artifact("planned_set")


def test_manifest_write_json(tmp_path: Path) -> None:
    manifest = _manifest()
    target = tmp_path / "nested" / "manifest.json"
    written = manifest.write_json(target)
    assert target.read_bytes() == written


def test_malformed_schema_version_is_rejected() -> None:
    with pytest.raises(ValidationError):
        _manifest(schema_version="0.1")


def test_lockfile_fingerprint(tmp_path: Path) -> None:
    lock = tmp_path / "uv.lock"
    lock.write_text("version = 1\n", encoding="utf-8")
    assert lockfile_fingerprint(lock) is not None
    assert lockfile_fingerprint(tmp_path / "absent.lock") is None


def test_detect_git_state_in_repository() -> None:
    state = detect_git_state(Path(__file__).resolve().parent)
    assert isinstance(state, GitState)
    if state.commit is not None:
        assert len(state.commit) == 40


def test_detect_git_state_falls_back_when_git_unavailable(tmp_path: Path) -> None:
    """Unavailable Git metadata must yield None, never a guessed commit."""
    state = detect_git_state(tmp_path / "definitely-not-a-directory")
    assert state.commit is None
    assert state.is_dirty is None


def test_runtime_snapshot_records_lock_and_code_state(tmp_path: Path) -> None:
    lock = tmp_path / "uv.lock"
    lock.write_text("version = 1\n", encoding="utf-8")
    snapshot = runtime_snapshot(lockfile=lock, repo_root=Path(__file__).resolve().parent)
    assert snapshot.lockfile_sha256 is not None
    assert snapshot.python_version.startswith("3.12")
