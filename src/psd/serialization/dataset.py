"""Canonical dataset assembly, persistence, and verification.

A canonical dataset on disk is::

    <dataset-dir>/
        manifest.json
        tables/
            source.parquet
            athlete.parquet
            ...

Invariants
----------

* **Every canonical table is present**, even when empty, so schema discovery and
  digest coverage are uniform across datasets.
* **Ordering happens before persistence.** Every table is sorted by its declared
  total ordering here, so the writer and the digests both see canonical order.
* **Artifacts live inside the data root.** Paths are resolved through
  :func:`psd.paths.resolve_within_data_root`, so a dataset cannot be written
  outside ``PSD_DATA_ROOT``.
* **The manifest is the index.** It records the schema version, the sources, the
  lineage, the environment, and both digests per artifact, and reading verifies
  all of it before returning data.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pyarrow as pa
from pydantic import BaseModel

from psd.paths import resolve_within_data_root
from psd.provenance.environment import EnvironmentSnapshot, runtime_snapshot
from psd.provenance.manifest import (
    ArtifactRef,
    DatasetKind,
    DatasetManifest,
    LineageEntry,
)
from psd.provenance.sources import SourceRecord
from psd.schema.registry import TableSpec, table_names, table_spec
from psd.schema.version import SCHEMA_VERSION, SchemaVersion, assert_schema_readable
from psd.serialization.canonical import content_digest
from psd.serialization.ordering import canonical_order, from_arrow_frame
from psd.serialization.parquet import (
    ParquetWriteResult,
    read_parquet,
    sha256_file,
    write_parquet,
)
from psd.serialization.table import empty_table, records_to_table, table_to_records

__all__ = (
    "CanonicalDataset",
    "DatasetLayoutError",
    "VerificationResult",
    "artifact_paths",
    "build_dataset",
    "dataset_directory",
    "read_dataset",
    "read_manifest",
    "verify_dataset",
    "write_dataset",
)

MANIFEST_NAME = "manifest.json"
TABLES_DIRNAME = "tables"

#: A table with fewer rows than this cannot hold a duplicate, so the key check skips it.
_MINIMUM_ROWS_FOR_DUPLICATES = 2


class DatasetLayoutError(ValueError):
    """Raised when a dataset directory does not match the canonical layout."""


@dataclass(frozen=True, slots=True)
class CanonicalDataset:
    """A validated, canonically ordered set of canonical tables.

    Attributes:
        dataset_id: Stable dataset identifier.
        tables: Canonical table name to Arrow table, all in canonical order.
        created_at: Timezone-aware UTC creation timestamp.
    """

    dataset_id: str
    tables: Mapping[str, pa.Table]
    created_at: datetime

    def table(self, name: str) -> pa.Table:
        """Return one table.

        Raises:
            DatasetLayoutError: The table is not part of this dataset.
        """
        found = self.tables.get(name)
        if found is None:
            msg = f"Dataset {self.dataset_id!r} has no table {name!r}."
            raise DatasetLayoutError(msg)
        return found

    def row_counts(self) -> dict[str, int]:
        """Return row counts per table, sorted by table name."""
        return {name: table.num_rows for name, table in sorted(self.tables.items())}

    def total_rows(self) -> int:
        """Return the total number of rows across all tables."""
        return sum(table.num_rows for table in self.tables.values())


@dataclass(frozen=True, slots=True)
class VerificationResult:
    """Outcome of verifying a persisted dataset against its manifest.

    Attributes:
        dataset_dir: Directory verified.
        manifest_digest_ok: Whether the manifest matches its recorded schema and
            artifact set.
        artifacts_ok: Whether every artifact digest matched.
        checked_artifacts: Number of artifacts checked.
        problems: Human-readable problems found.
    """

    dataset_dir: Path
    manifest_digest_ok: bool
    artifacts_ok: bool
    checked_artifacts: int
    problems: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        """Whether the dataset verified cleanly."""
        return self.manifest_digest_ok and self.artifacts_ok and not self.problems


def build_dataset(
    records: Mapping[str, Sequence[BaseModel]],
    *,
    dataset_id: str,
    created_at: datetime | None = None,
) -> CanonicalDataset:
    """Build a canonically ordered dataset from validated records.

    Tables with no records become empty tables that still carry the canonical
    schema, so every dataset has the same table set.

    Args:
        records: Canonical table name to validated records.
        dataset_id: Stable dataset identifier.
        created_at: Creation timestamp; defaults to now in UTC.

    Returns:
        The ordered dataset.

    Raises:
        DatasetLayoutError: An unknown table name was supplied.
    """
    unknown = sorted(set(records) - set(table_names()))
    if unknown:
        msg = f"Unknown canonical table(s): {', '.join(unknown)}."
        raise DatasetLayoutError(msg)
    tables: dict[str, pa.Table] = {}
    for name in table_names():
        supplied = records.get(name, ())
        table = records_to_table(list(supplied), table_name=name)
        tables[name] = _ordered(table, name)
    return CanonicalDataset(
        dataset_id=dataset_id,
        tables=tables,
        created_at=created_at if created_at is not None else datetime.now(tz=UTC),
    )


def _ordered(table: pa.Table, name: str) -> pa.Table:
    """Sort a table into its canonical total ordering."""
    spec = table_spec(name)
    return canonical_order(table, spec.order_by, table_name=name)


def dataset_directory(relative: Path | str, *, data_root: Path | None = None) -> Path:
    """Resolve a dataset directory inside the data root, creating it."""
    return resolve_within_data_root(
        Path(relative) / TABLES_DIRNAME, data_root=data_root, create=True
    )


@dataclass(frozen=True, slots=True)
class DatasetWriteSpec:
    """Metadata recorded alongside a persisted dataset.

    Attributes:
        dataset_kind: Competition, training, prospective, or synthetic.
        dataset_name: Human-readable name; defaults to the dataset id.
        sources: Every source the dataset draws on.
        lineage: Transformation steps.
        description: Manifest notes.
    """

    dataset_kind: DatasetKind
    dataset_name: str | None = None
    sources: tuple[SourceRecord, ...] = ()
    lineage: tuple[LineageEntry, ...] = ()
    description: str | None = None


def write_dataset(
    dataset: CanonicalDataset,
    relative: Path | str,
    spec: DatasetWriteSpec,
    *,
    environment: EnvironmentSnapshot | None = None,
    data_root: Path | None = None,
) -> DatasetManifest:
    """Persist a dataset and return its manifest.

    Args:
        dataset: The ordered dataset.
        relative: Dataset directory relative to the data root.
        spec: Metadata recorded in the manifest.
        environment: Environment snapshot; captured live when omitted.
        data_root: Explicit data root; resolved from ``PSD_DATA_ROOT`` otherwise.

    Returns:
        The manifest, which is also written to ``manifest.json``.
    """
    tables_dir = dataset_directory(relative, data_root=data_root)
    artifacts: list[ArtifactRef] = []
    for name in table_names():
        written: ParquetWriteResult = write_parquet(
            dataset.tables[name], tables_dir / f"{name}.parquet", table_name=name
        )
        artifacts.append(
            ArtifactRef(
                name=name,
                relative_path=f"{TABLES_DIRNAME}/{name}.parquet",
                sha256=written.sha256,
                content_sha256=written.content_sha256,
                byte_size=written.byte_size,
                row_count=written.row_count,
            )
        )
    manifest = DatasetManifest(
        dataset_id=dataset.dataset_id,
        dataset_name=spec.dataset_name if spec.dataset_name is not None else dataset.dataset_id,
        dataset_kind=spec.dataset_kind,
        schema_version=SCHEMA_VERSION.tag,
        created_at=dataset.created_at,
        environment=environment if environment is not None else runtime_snapshot(),
        sources=spec.sources,
        artifacts=tuple(artifacts),
        lineage=spec.lineage,
        notes=spec.description,
    )
    manifest.write_json(_manifest_path(relative, data_root=data_root))
    return manifest


def _manifest_path(relative: Path | str, *, data_root: Path | None = None) -> Path:
    return resolve_within_data_root(
        Path(relative) / MANIFEST_NAME, data_root=data_root, create=True
    )


def read_dataset(
    relative: Path | str,
    *,
    data_root: Path | None = None,
) -> tuple[CanonicalDataset, DatasetManifest]:
    """Read a persisted dataset and its manifest.

    The recorded schema version is checked before any table is loaded, so an
    artifact written by an incompatible schema is refused rather than
    reinterpreted.

    Args:
        relative: Dataset directory relative to the data root.
        data_root: Explicit data root; resolved from ``PSD_DATA_ROOT`` otherwise.

    Returns:
        The dataset and its manifest.

    Raises:
        DatasetLayoutError: The manifest is missing, unreadable, or unreadable at
            this schema version.
    """
    manifest_path = _manifest_path(relative, data_root=data_root)
    if not manifest_path.is_file():
        msg = f"No dataset manifest at {manifest_path}."
        raise DatasetLayoutError(msg)
    manifest = _parse_manifest(manifest_path)
    assert_schema_readable(SchemaVersion.parse(manifest.schema_version))

    tables: dict[str, pa.Table] = {}
    for name in table_names():
        artifact = manifest.artifact(name)
        path = resolve_within_data_root(
            Path(relative) / artifact.relative_path, data_root=data_root
        )
        tables[name] = read_parquet(path, table_name=name)
    dataset = CanonicalDataset(
        dataset_id=manifest.dataset_id,
        tables=tables,
        created_at=manifest.created_at,
    )
    return dataset, manifest


def _parse_manifest(path: Path) -> DatasetManifest:
    """Parse a manifest file, reporting layout problems precisely."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        msg = f"Unreadable dataset manifest at {path}: {error}"
        raise DatasetLayoutError(msg) from error
    try:
        return DatasetManifest.model_validate(payload)
    except ValueError as error:
        msg = f"Invalid dataset manifest at {path}: {error}"
        raise DatasetLayoutError(msg) from error


def read_manifest(
    relative: Path | str,
    *,
    data_root: Path | None = None,
) -> DatasetManifest:
    """Read a dataset's manifest without loading any table.

    A multi-million-row corpus cannot be read into memory just to learn what it
    contains, and a caller that only needs the declared artifact set -- an audit, a
    projection builder, a size report -- should not have to pay for the rows. This is
    the manifest half of :func:`read_dataset`, with the same schema-version refusal.

    Args:
        relative: Dataset directory relative to the data root.
        data_root: Explicit data root; resolved from the environment otherwise.

    Returns:
        The dataset's manifest.

    Raises:
        DatasetLayoutError: The manifest is missing, unreadable, or written at an
            unreadable schema version.
    """
    manifest_path = _manifest_path(relative, data_root=data_root)
    if not manifest_path.is_file():
        msg = f"No dataset manifest at {manifest_path}."
        raise DatasetLayoutError(msg)
    manifest = _parse_manifest(manifest_path)
    assert_schema_readable(SchemaVersion.parse(manifest.schema_version))
    return manifest


def artifact_paths(
    manifest: DatasetManifest,
    relative: Path | str,
    *,
    data_root: Path | None = None,
) -> dict[str, Path]:
    """Return the resolved on-disk path of every artifact a manifest declares.

    The paths come from the manifest rather than from a naming convention, so a reader
    follows the manifest even if an artifact was written under an unexpected name.
    """
    return {
        artifact.name: resolve_within_data_root(
            Path(relative) / artifact.relative_path, data_root=data_root
        )
        for artifact in manifest.artifacts
    }


def verify_dataset(
    relative: Path | str,
    *,
    data_root: Path | None = None,
) -> VerificationResult:
    """Verify a persisted dataset against its manifest.

    Both digests are checked: ``sha256`` for byte-level tamper evidence and
    ``content_sha256`` for logical reproducibility. The manifest is read directly
    and every artifact is checked independently, so one damaged or missing file is
    reported as a problem rather than aborting the verification.

    Args:
        relative: Dataset directory relative to the data root.
        data_root: Explicit data root; resolved from ``PSD_DATA_ROOT`` otherwise.

    Returns:
        The verification result. Problems are collected rather than raised, so one
        pass reports everything that is wrong.
    """
    manifest_path = _manifest_path(relative, data_root=data_root)
    if not manifest_path.is_file():
        msg = f"No dataset manifest at {manifest_path}."
        raise DatasetLayoutError(msg)
    manifest = _parse_manifest(manifest_path)
    expected_tables = set(table_names())
    manifest_ok = {artifact.name for artifact in manifest.artifacts} == expected_tables
    problems: list[str] = []
    if not manifest_ok:
        problems.append(
            "manifest artifacts do not cover the canonical table set exactly; "
            f"expected {len(expected_tables)} tables"
        )
    artifacts_ok = True
    for artifact in manifest.artifacts:
        artifacts_ok, artifact_problems = _verify_artifact(relative, artifact, data_root=data_root)
        problems.extend(artifact_problems)
        artifacts_ok = artifacts_ok and not artifact_problems
    return VerificationResult(
        dataset_dir=resolve_within_data_root(Path(relative), data_root=data_root),
        manifest_digest_ok=manifest_ok,
        artifacts_ok=artifacts_ok,
        checked_artifacts=len(manifest.artifacts),
        problems=tuple(problems),
    )


def _verify_artifact(
    relative: Path | str,
    artifact: ArtifactRef,
    *,
    data_root: Path | None,
) -> tuple[bool, list[str]]:
    """Check one artifact's file digest, canonical content digest, and key uniqueness.

    The registry declares each table's key columns to be *unique within the table*, and
    nothing else enforces it: both digests are computed over whatever rows the file
    holds, so a table that duplicates a key still digests cleanly. A corpus with two rows
    claiming one identity is internally consistent and scientifically wrong, which is the
    failure mode digests cannot see.
    """
    problems: list[str] = []
    path = resolve_within_data_root(Path(relative) / artifact.relative_path, data_root=data_root)
    if not path.is_file():
        return False, [f"{artifact.name}: artifact file is missing at {path}"]
    actual = sha256_file(path)
    if actual != artifact.sha256:
        problems.append(
            f"{artifact.name}: file digest {actual} does not match manifest {artifact.sha256}"
        )
    try:
        table = read_parquet(path, table_name=artifact.name)
    except (OSError, ValueError, pa.ArrowException) as error:
        problems.append(f"{artifact.name}: artifact is unreadable ({error})")
        return False, problems
    expected = content_digest(table, table_name=artifact.name)
    if expected != artifact.content_sha256:
        problems.append(
            f"{artifact.name}: content digest {expected} does not match manifest "
            f"{artifact.content_sha256}"
        )
    problems.extend(_duplicate_key_problems(table, table_name=artifact.name))
    return not problems, problems


def _duplicate_key_problems(table: pa.Table, *, table_name: str) -> list[str]:
    """Return a problem description if the table's declared key is not unique.

    Polars performs the check because it hashes rows in one streaming pass rather than
    materialising a Python set of every key, which at corpus scale would be the most
    expensive thing verification does. Fewer than two rows are trivially unique.
    """
    spec = table_spec(table_name)
    if table.num_rows < _MINIMUM_ROWS_FOR_DUPLICATES:
        return []
    duplicated = from_arrow_frame(table.select(list(spec.primary_key))).is_duplicated().any()
    if not bool(duplicated):
        return []
    key = ", ".join(spec.primary_key)
    return [
        f"{spec.name}: primary key ({key}) is not unique across {table.num_rows} rows; the "
        f"registry declares those columns unique within the table"
    ]


def load_records(
    source_dir: Path,
    *,
    required: Sequence[str] = (),
) -> dict[str, list[BaseModel]]:
    """Load JSON record files from a directory, validating against contracts.

    One JSON array per table, named ``<table>.json``. This is the machine-agnostic
    interchange used to hand a dataset to the builder; it is not a canonical
    artifact and carries no ordering or digest guarantees.

    Args:
        source_dir: Directory holding ``<table>.json`` files.
        required: Tables that must be present.

    Returns:
        Table name to validated records.

    Raises:
        DatasetLayoutError: A required file is missing, unreadable, or invalid.
    """
    records: dict[str, list[BaseModel]] = {}
    for name in table_names():
        path = source_dir / f"{name}.json"
        if not path.is_file():
            if name in required:
                msg = f"Missing required record file {path}."
                raise DatasetLayoutError(msg)
            continue
        records[name] = _load_table_records(path, name)
    return records


def _json_array(payload: Any) -> list[object] | None:
    """Return a JSON array as a list of objects, or ``None`` for any other shape."""
    if not isinstance(payload, list):
        return None
    elements: list[object] = list(cast("list[object]", payload))
    return elements


def _load_table_records(path: Path, name: str) -> list[BaseModel]:
    """Parse and validate one table's JSON records."""
    spec: TableSpec = table_spec(name)
    try:
        payload: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        msg = f"Unreadable record file {path}: {error}"
        raise DatasetLayoutError(msg) from error
    elements = _json_array(payload)
    if elements is None:
        msg = f"Record file {path} must contain a JSON array."
        raise DatasetLayoutError(msg)
    records: list[BaseModel] = []
    for index, element in enumerate(elements):
        if not isinstance(element, dict):
            msg = f"Record {index} in {path} is not a JSON object."
            raise DatasetLayoutError(msg)
        row = cast("dict[str, object]", element)
        try:
            records.append(spec.model.model_validate(row))
        except ValueError as error:
            msg = f"Record {index} in {path} violates {spec.model.__name__}: {error}"
            raise DatasetLayoutError(msg) from error
    return records


def records_from_tables(tables: Mapping[str, pa.Table]) -> dict[str, list[BaseModel]]:
    """Rebuild validated records from canonical tables.

    Used to prove that a persisted dataset re-reads into exactly the contracts
    that produced it.
    """
    return {name: table_to_records(table, table_name=name) for name, table in tables.items()}


def empty_dataset(dataset_id: str, *, created_at: datetime | None = None) -> CanonicalDataset:
    """Return a dataset with every canonical table present and empty."""
    tables = {name: empty_table(name) for name in table_names()}
    return CanonicalDataset(
        dataset_id=dataset_id,
        tables=tables,
        created_at=created_at if created_at is not None else datetime.now(tz=UTC),
    )
