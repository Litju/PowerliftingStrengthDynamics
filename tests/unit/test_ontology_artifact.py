"""Tests for the persisted exercise-ontology artifact.

The ontology is persisted as an ordinary canonical dataset, so these tests assert
that it inherits the guarantees PSD already relies on -- explicit ordering, Arrow
schema enforcement, writer-independent content digests, and manifest verification --
and that it is deterministic enough to compare in CI across two operating systems.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pytest

from psd.ontology import default_ontology
from psd.ontology.artifact import (
    ONTOLOGY_EPOCH,
    ONTOLOGY_SOURCE_SYSTEM,
    build_ontology_dataset,
    ontology_dataset_id,
    ontology_relative_path,
    ontology_source_record,
    write_ontology,
)
from psd.ontology.records import alias_records
from psd.paths import resolve_within_data_root
from psd.provenance.environment import EnvironmentSnapshot
from psd.provenance.manifest import DatasetKind, DatasetManifest
from psd.provenance.sources import DataRegime, SourceNature
from psd.schema.registry import table_names, table_spec
from psd.schema.vocabulary import ResolutionStatus
from psd.serialization.canonical import content_digest
from psd.serialization.dataset import CanonicalDataset, read_dataset, verify_dataset
from psd.serialization.ordering import canonical_order
from psd.serialization.parquet import read_parquet
from psd.serialization.table import records_to_table, table_to_records, table_to_rows
from psd.validation import validate_tables

ONTOLOGY = default_ontology()

EXERCISE_TABLES: tuple[str, ...] = (
    "exercise_definition",
    "exercise_alias",
    "exercise_normalization",
)


@pytest.fixture(name="dataset")
def _dataset() -> CanonicalDataset:
    return build_ontology_dataset(ONTOLOGY)


# ---------------------------------------------------------------------------
# shape
# ---------------------------------------------------------------------------


def test_the_artifact_carries_every_canonical_table(dataset: CanonicalDataset) -> None:
    """A canonical dataset always has the full table set, so discovery stays uniform."""
    assert set(dataset.tables) == set(table_names())


def test_the_artifact_populates_the_three_exercise_tables(dataset: CanonicalDataset) -> None:
    counts = dataset.row_counts()
    for table in EXERCISE_TABLES:
        assert counts[table] > 0, table
    assert counts["source"] == 1


def test_the_artifact_leaves_the_athlete_tables_empty(dataset: CanonicalDataset) -> None:
    counts = dataset.row_counts()
    for table in ("performed_set", "competition", "planned_session", "athlete"):
        assert counts[table] == 0, table


def test_the_artifact_registers_its_own_source(dataset: CanonicalDataset) -> None:
    rows = table_to_rows(dataset.table("source"), table_name="source")
    assert len(rows) == 1
    record = rows[0]
    assert record["origin_system"] == ONTOLOGY_SOURCE_SYSTEM
    assert record["nature"] == SourceNature.REFERENCE.value
    assert record["regime"] == DataRegime.REFERENCE.value


def test_the_registry_source_is_reference_data_and_not_simulator_output() -> None:
    """A vocabulary must be neither an observation nor a generated athlete record.

    It was previously filed ``synthetic``/``psd_sim``, which made authored reference
    data look like simulator output and left ``synthetic`` meaning "not real" instead
    of "generated athlete observations".
    """
    record = ontology_source_record(ONTOLOGY.ontology_version.tag)
    assert record.nature is SourceNature.REFERENCE
    assert record.regime is DataRegime.REFERENCE
    assert record.nature is not SourceNature.REAL
    assert record.nature is not SourceNature.SYNTHETIC
    assert record.regime is not DataRegime.SIM


def test_the_dataset_id_and_path_derive_from_the_ontology_version() -> None:
    assert ontology_dataset_id(ONTOLOGY) == "psd-ontology-1.0.0"
    relative = ontology_relative_path(ONTOLOGY)
    assert relative == Path("canonical") / "ontology" / "psd-ontology-1.0.0"
    # The leaf carries a filesystem-safe rendering of the version tag: a literal "/"
    # would be a separator on one platform and illegal in a Windows filename.
    assert relative.name == "psd-ontology-1.0.0"
    assert relative.parts == ("canonical", "ontology", "psd-ontology-1.0.0")


def test_a_build_uses_a_fixed_epoch_by_default(dataset: CanonicalDataset) -> None:
    """Otherwise a rebuild would change the manifest bytes without changing any table."""
    assert dataset.created_at == ONTOLOGY_EPOCH
    assert dataset.created_at.tzinfo is UTC


def test_the_artifact_passes_full_validation(
    dataset: CanonicalDataset,
) -> None:
    report = validate_tables(dataset.tables, strict=True)
    assert report.errors == (), [issue.message for issue in report.errors]
    assert report.warnings == (), [issue.message for issue in report.warnings]


# ---------------------------------------------------------------------------
# completeness: the artifact publishes its refusals too
# ---------------------------------------------------------------------------


def test_the_artifact_records_every_alias_and_its_outcome(dataset: CanonicalDataset) -> None:
    aliases = table_to_rows(dataset.table("exercise_alias"), table_name="exercise_alias")
    outcomes = table_to_rows(
        dataset.table("exercise_normalization"), table_name="exercise_normalization"
    )
    resolved_by_label = {(row["source_system"], row["raw_label"]): row for row in outcomes}
    for alias in aliases:
        key = (alias["source_system"], alias["alias_raw"])
        outcome = resolved_by_label[key]
        assert outcome["exercise_id"] == alias["exercise_id"], key
        assert outcome["resolution_status"] in {
            ResolutionStatus.EXACT_CANONICAL.value,
            ResolutionStatus.RESOLVED_ALIAS.value,
        }, key


def test_the_artifact_persists_the_deliberate_refusals(
    dataset: CanonicalDataset,
) -> None:
    rows = _normalization_rows(dataset)
    by_label = {row["raw_label"]: row for row in rows}
    for label, status in (
        ("Bench Variation", ResolutionStatus.PARTIAL_FAMILY.value),
        ("Machine Press", ResolutionStatus.AMBIGUOUS.value),
        ("Squat Machine", ResolutionStatus.AMBIGUOUS.value),
        ("Leg Press?", ResolutionStatus.AMBIGUOUS.value),
        ("Zorblax Lever Exercise", ResolutionStatus.UNMAPPED.value),
    ):
        assert by_label[label]["resolution_status"] == status, label
        assert by_label[label]["exercise_id"] is None, label
        assert by_label[label]["ambiguity_reason"], label


def test_the_artifact_persists_the_mapping_evidence_for_every_outcome(
    dataset: CanonicalDataset,
) -> None:
    """Method, alias row, candidates, reason: the whole record, and no score.

    The retired ``confidence`` column was a fixed float per method with no empirical
    calibration. What replaces it is not another number but the provenance already
    carried: every row names the stage that produced it, and a lookup cites its row.
    """
    resolved = {
        ResolutionStatus.EXACT_CANONICAL.value,
        ResolutionStatus.RESOLVED_ALIAS.value,
    }
    lookup_methods = {"registered_alias", "cross_source_alias"}
    for row in _normalization_rows(dataset):
        assert row["resolution_method"], row["raw_label"]
        if row["resolution_method"] in lookup_methods:
            assert row["source_alias_id"], row["raw_label"]
            assert row["resolution_status"] in resolved, row["raw_label"]
        if row["resolution_status"] not in resolved:
            assert row["source_alias_id"] is None, row["raw_label"]
            assert row["ambiguity_reason"], row["raw_label"]
    assert "confidence" not in dataset.table("exercise_normalization").schema.names


def test_the_artifact_preserves_every_raw_label_verbatim(
    dataset: CanonicalDataset,
) -> None:
    """The persisted row keeps the source string, not the normalized lookup key."""
    rows = _normalization_rows(dataset)
    raw = {row["raw_label"] for row in rows}
    assert "Leg Press?" in raw
    assert "Machine Press" in raw
    assert "  " in raw, "a blank source label must be persisted, not dropped"
    questioned = next(row for row in rows if row["raw_label"] == "Leg Press?")
    assert questioned["normalized_label"] == "legpress"
    assert questioned["resolution_status"] == ResolutionStatus.AMBIGUOUS.value


def test_every_persisted_candidate_is_a_persisted_exercise(
    dataset: CanonicalDataset,
) -> None:
    known = {
        str(row["exercise_id"])
        for row in table_to_rows(
            dataset.table("exercise_definition"), table_name="exercise_definition"
        )
    }
    for row in _normalization_rows(dataset):
        candidates: list[str] = list(row["candidate_exercise_ids"])
        for candidate in candidates:
            assert candidate in known, row["raw_label"]


# ---------------------------------------------------------------------------
# determinism
# ---------------------------------------------------------------------------


def test_two_builds_produce_identical_content_digests() -> None:
    """Determinism is the property that lets a CI diff catch a vocabulary change."""
    first = build_ontology_dataset(ONTOLOGY)
    second = build_ontology_dataset(ONTOLOGY)
    for table in EXERCISE_TABLES:
        assert content_digest(first.table(table), table_name=table) == content_digest(
            second.table(table), table_name=table
        ), table


def test_two_writes_produce_byte_identical_files(tmp_path: Path) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    environment = EnvironmentSnapshot(
        python_version="3.12.0",
        python_implementation="cpython",
        platform="test-platform",
        package_version="0.1.0",
        lockfile_sha256="0" * 64,
        code_commit="0" * 40,
        is_dirty_tree=False,
    )
    write_ontology(ONTOLOGY, data_root=first_root, environment=environment)
    write_ontology(ONTOLOGY, data_root=second_root, environment=environment)
    relative = ontology_relative_path(ONTOLOGY) / "tables"
    for table in table_names():
        left = (first_root / relative / f"{table}.parquet").read_bytes()
        right = (second_root / relative / f"{table}.parquet").read_bytes()
        assert left == right, table


def test_persisted_ordering_is_the_registry_canonical_ordering(
    dataset: CanonicalDataset,
) -> None:
    """Row order must come from the registry, not from the order rows were built."""
    table = dataset.table("exercise_alias")
    order = (
        "source_system",
        "alias_normalized",
        "exercise_id",
        "alias_raw",
        "exercise_alias_id",
    )
    expected = canonical_order(table, order, table_name="exercise_alias")
    keys = [
        (row["source_system"], row["alias_normalized"], row["alias_raw"])
        for row in table_to_rows(expected, table_name="exercise_alias")
    ]
    assert keys == sorted(keys)


def test_content_digest_does_not_depend_on_row_input_order() -> None:
    """A table built in any order must digest identically, or ordering is not canonical."""

    records = alias_records(ONTOLOGY, ingested_at=ONTOLOGY_EPOCH)
    forward = records_to_table(list(records), table_name="exercise_alias")
    backward = records_to_table(list(reversed(records)), table_name="exercise_alias")
    order = (
        "source_system",
        "alias_normalized",
        "exercise_id",
        "alias_raw",
        "exercise_alias_id",
    )
    assert content_digest(
        canonical_order(forward, order, table_name="exercise_alias"), table_name="exercise_alias"
    ) == content_digest(
        canonical_order(backward, order, table_name="exercise_alias"), table_name="exercise_alias"
    )


# ---------------------------------------------------------------------------
# persistence and manifest
# ---------------------------------------------------------------------------


def test_the_artifact_is_persisted_as_a_reference_dataset(tmp_path: Path) -> None:
    manifest = write_ontology(ONTOLOGY, data_root=tmp_path)
    assert isinstance(manifest, DatasetManifest)
    assert manifest.dataset_kind is DatasetKind.REFERENCE
    assert manifest.schema_version == "psd-canonical/1.1.0"
    assert manifest.sources == ()
    assert len(manifest.artifacts) == len(table_names())


def test_the_manifest_paths_are_portable_relative_posix(tmp_path: Path) -> None:
    manifest = write_ontology(ONTOLOGY, data_root=tmp_path)
    for artifact in manifest.artifacts:
        assert "\\" not in artifact.relative_path
        assert not Path(artifact.relative_path).is_absolute()


def test_the_persisted_artifact_verifies_against_its_manifest(tmp_path: Path) -> None:
    write_ontology(ONTOLOGY, data_root=tmp_path)
    result = verify_dataset(ontology_relative_path(ONTOLOGY), data_root=tmp_path)
    assert result.ok, result.problems
    assert result.checked_artifacts == len(table_names())


def test_the_persisted_artifact_reads_back_through_the_manifest(tmp_path: Path) -> None:
    """Reading verifies the schema, the recorded table, and the recorded self-description."""
    write_ontology(ONTOLOGY, data_root=tmp_path)
    restored, manifest = read_dataset(ontology_relative_path(ONTOLOGY), data_root=tmp_path)
    assert manifest.dataset_kind is DatasetKind.REFERENCE
    for table in EXERCISE_TABLES:
        assert restored.table(table).equals(build_ontology_dataset(ONTOLOGY).table(table))


def test_persisted_artifacts_carry_schema_and_version_metadata(tmp_path: Path) -> None:
    manifest = write_ontology(ONTOLOGY, data_root=tmp_path)
    path = resolve_within_data_root(
        ontology_relative_path(ONTOLOGY) / "tables" / "exercise_alias.parquet",
        data_root=tmp_path,
    )
    table = read_parquet(path, table_name="exercise_alias")
    metadata = table.schema.metadata or {}
    assert metadata[b"psd_table"] == b"exercise_alias"
    assert metadata[b"psd_schema_version"] == b"psd-canonical/1.1.0"
    assert metadata[b"psd_id_scheme"] == b"psd-ids-v1"
    assert b"psd_column_order" in metadata
    artifact = next(item for item in manifest.artifacts if item.name == "exercise_alias")
    assert metadata[b"psd_content_sha256"].decode() == artifact.content_sha256


def test_persisted_exercise_records_re_validate_against_their_contracts(tmp_path: Path) -> None:
    """A persisted ontology row must survive the contract that produced it."""
    write_ontology(ONTOLOGY, data_root=tmp_path)
    restored, _manifest = read_dataset(ontology_relative_path(ONTOLOGY), data_root=tmp_path)
    for table in EXERCISE_TABLES:
        records = table_to_records(restored.table(table), table_name=table)
        assert records, table
        sources = [str(record.model_dump()["source_id"]) for record in records]
        assert all(sources), table


def test_a_tampered_ontology_artifact_fails_verification(tmp_path: Path) -> None:
    """Verification has to actually fail, or the digests are decoration."""
    write_ontology(ONTOLOGY, data_root=tmp_path)
    path = resolve_within_data_root(
        ontology_relative_path(ONTOLOGY) / "tables" / "exercise_alias.parquet",
        data_root=tmp_path,
    )
    path.write_bytes(path.read_bytes() + b"\x00")
    result = verify_dataset(ontology_relative_path(ONTOLOGY), data_root=tmp_path)
    assert result.ok is False
    assert result.problems


def test_the_arrow_schema_of_every_exercise_table_is_the_canonical_one(
    dataset: CanonicalDataset,
) -> None:
    for table in EXERCISE_TABLES:
        assert isinstance(dataset.table(table), pa.Table)
        assert dataset.table(table).schema.equals(table_spec(table).arrow_schema())


def test_a_custom_creation_instant_is_honoured() -> None:
    """A caller that wants wall-clock provenance passes its own instant."""
    moment = datetime(2030, 5, 4, 3, 2, 1, tzinfo=UTC)
    dataset = build_ontology_dataset(ONTOLOGY, created_at=moment)
    assert dataset.created_at == moment
    assert dataset.tables["exercise_alias"].num_rows > 0


def test_a_custom_probe_namespace_changes_the_persisted_probe_rows() -> None:
    default_rows = table_to_rows(
        build_ontology_dataset(ONTOLOGY).table("exercise_normalization"),
        table_name="exercise_normalization",
    )
    hevy_rows = table_to_rows(
        build_ontology_dataset(ONTOLOGY, probe_source_system="hevy").table(
            "exercise_normalization"
        ),
        table_name="exercise_normalization",
    )
    assert {row["source_system"] for row in default_rows} != {
        row["source_system"] for row in hevy_rows
    }


def _normalization_rows(dataset: CanonicalDataset) -> list[dict[str, Any]]:
    """Return the persisted normalization rows as plain dictionaries."""
    return table_to_rows(
        dataset.table("exercise_normalization"), table_name="exercise_normalization"
    )
