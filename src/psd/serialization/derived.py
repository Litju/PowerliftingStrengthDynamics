"""Deterministic persistence for derived artifacts.

What this is for
----------------

A **canonical** table is one PSD's schema owns: its columns, its ordering, and its
contract live in :mod:`psd.schema.registry`, and it is the only thing the benchmark
treats as source of truth. A **derived** artifact is a read-optimised projection *of*
canonical tables -- an athlete-history summary, a coverage audit -- that a consumer
wants to load without re-deriving it.

The distinction is load-bearing, so this module is deliberately separate from
:mod:`psd.serialization.parquet`:

* a derived artifact declares its own Arrow schema and ordering here, and adding one
  never touches the canonical registry or the canonical schema version;
* a derived artifact carries the same ``content_sha256`` as a canonical one, computed
  by the same :mod:`psd.serialization.canonical` encoding, so "did this change?" is one
  comparable question across both;
* a derived artifact records the canonical dataset it was derived from, so it can never
  be mistaken for an independent observation of an athlete.

The last point is the one that matters scientifically. A longitudinal summary is a
*view* of canonical events. If a consumer could read the summary without knowing that,
the corpus would have two representations of the same competition and no way to tell
which one is authoritative. PSD keeps exactly one representation of an event and marks
everything else as a projection of it.

Both digests are written into the file's Arrow key/value block, so a derived artifact
is self-describing without its manifest: the schema version, the artifact name, the
declared column order, the digest, and the canonical dataset it came from.

Typing note
-----------

PyArrow ships no inline types, and the community stubs describe the Parquet
read/write surface with parameter unions narrower than the runtime accepts. The two
entry points are therefore reached through narrow, documented shims with explicit local
signatures, as in :mod:`psd.serialization.parquet` and :mod:`psd.serialization.stream`,
so project-wide strictness is kept rather than relaxed.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict, Field, field_validator

from psd.provenance.manifest import DatasetManifest, LineageEntry, manifest_digest
from psd.schema.version import ID_SCHEME, SCHEMA_VERSION
from psd.serialization.canonical import content_digest
from psd.serialization.ordering import canonical_order_polars, from_arrow_frame
from psd.serialization.parquet import (
    PARQUET_PROFILE,
    ParquetProfile,
    ParquetWriteResult,
    sha256_file,
)
from psd.timeutil import require_aware

__all__ = (
    "DERIVED_ARTIFACT_ENCODING",
    "DerivedArtifactError",
    "DerivedArtifactManifest",
    "DerivedParent",
    "DerivedTableSpec",
    "derived_metadata",
    "derived_schema_with_metadata",
    "read_derived_manifest",
    "read_derived_table",
    "write_derived_table",
)

#: Names the encoding of the key/value block a derived artifact carries, so a reader
#: can tell a derived projection from a canonical table by metadata alone.
DERIVED_ARTIFACT_ENCODING: str = "psd-derived-artifact/1"

_READ_TABLE: Callable[..., pa.Table] = getattr(pq, "read_table")
_WRITE_TABLE: Callable[..., None] = getattr(pq, "write_table")
_WITH_METADATA: Callable[[pa.Schema, dict[bytes, bytes]], pa.Schema] = getattr(
    pa.Schema, "with_metadata"
)


class DerivedArtifactError(ValueError):
    """Raised when a derived artifact is declared or read inconsistently."""


@dataclass(frozen=True, slots=True)
class DerivedParent:
    """The canonical dataset a derived artifact projects.

    Recorded on the artifact and in its manifest. A projection with no declared parent
    would be an observation in everything but name, and that is precisely the confusion
    a longitudinal summary invites: a summary row looks like evidence, and it is not.

    Attributes:
        dataset_id: The canonical dataset identifier.
        manifest_digest: SHA-256 of that dataset's manifest bytes, so the projection is
            pinned to the exact corpus it was derived from and not merely to a name.
    """

    dataset_id: str
    manifest_digest: str

    @classmethod
    def of(cls, manifest: DatasetManifest) -> DerivedParent:
        """Return the parent a canonical dataset's own manifest declares."""
        return cls(dataset_id=manifest.dataset_id, manifest_digest=manifest_digest(manifest))


@dataclass(frozen=True, slots=True)
class DerivedTableSpec:
    """The declared shape of one derived artifact.

    Attributes:
        name: Logical artifact name, recorded in the file's metadata.
        columns: Explicit persisted column order.
        primary_key: Columns that are unique within the artifact.
        order_by: Canonical total ordering, ascending. It ends with the primary key,
            exactly as a canonical table's ordering does, so that no two rows can
            compare equal and the result cannot depend on input row order.
        arrow_schema: The artifact's Arrow schema, in ``columns`` order.
        summary: One-line description for documentation and CLI output.
    """

    name: str
    columns: tuple[str, ...]
    primary_key: tuple[str, ...]
    order_by: tuple[str, ...]
    arrow_schema: pa.Schema
    summary: str

    @classmethod
    def build(
        cls,
        *,
        name: str,
        arrow_schema: pa.Schema,
        primary_key: Sequence[str],
        order_by: Sequence[str] | None = None,
        summary: str,
    ) -> DerivedTableSpec:
        """Return a spec for *arrow_schema*, validating the ordering.

        Args:
            name: Logical artifact name.
            arrow_schema: The artifact's Arrow schema. Its field order *is* the
                declared column order, so the two cannot drift apart.
            primary_key: Columns that are unique within the artifact.
            order_by: Canonical ordering; defaults to the primary key.
            summary: One-line description.

        Returns:
            The validated spec.

        Raises:
            DerivedArtifactError: The ordering names an absent column, or omits a
                primary-key column and so is not a total order.
        """
        columns = tuple(arrow_schema.names)
        ordering = tuple(primary_key) if order_by is None else tuple(order_by)
        unknown = sorted(set(ordering) - set(columns))
        if unknown:
            msg = f"{name}: ordering names column(s) absent from the schema: {', '.join(unknown)}."
            raise DerivedArtifactError(msg)
        missing = [column for column in primary_key if column not in ordering]
        if missing:
            msg = (
                f"{name}: ordering must contain the primary key {tuple(primary_key)!r} so it "
                f"is total; {', '.join(missing)} is missing from {ordering!r}."
            )
            raise DerivedArtifactError(msg)
        return cls(
            name=name,
            columns=columns,
            primary_key=tuple(primary_key),
            order_by=ordering,
            arrow_schema=arrow_schema,
            summary=summary,
        )

    def empty(self) -> pa.Table:
        """Return an empty table carrying this artifact's schema."""
        return pa.Table.from_batches([], schema=self.arrow_schema)

    def order(self, table: pa.Table) -> pa.Table:
        """Return *table* sorted into this artifact's canonical ordering.

        Args:
            table: The rows to sort.

        Returns:
            A new table in canonical order, with the declared schema preserved.

        Raises:
            DerivedArtifactError: The table does not match the declared schema, so the
                ordering could not describe it.
        """
        if not table.schema.equals(self.arrow_schema, check_metadata=False):
            msg = (
                f"{self.name}: table schema does not match the declared artifact schema; "
                f"expected=[{', '.join(self.columns)}] actual=[{', '.join(table.schema.names)}]"
            )
            raise DerivedArtifactError(msg)
        # Polars sorts; it also widens Arrow strings to its own 64-bit representation,
        # so the sorted frame is cast back or the artifact would have two physical forms
        # and two digests.
        frame = canonical_order_polars(from_arrow_frame(table), self.order_by)
        return frame.to_arrow().cast(self.arrow_schema)

    def metadata(self, *, content_sha256: str, parent: DerivedParent) -> dict[bytes, bytes]:
        """Return the Arrow key/value block this artifact carries.

        The parent fields are what stop a derived artifact from reading as an
        independent observation: the file itself says which canonical corpus it
        projects, so a consumer cannot mistake a projection for evidence.
        """
        return {
            b"psd_table": self.name.encode("utf-8"),
            b"psd_schema_version": SCHEMA_VERSION.tag.encode("utf-8"),
            b"psd_id_scheme": ID_SCHEME.encode("utf-8"),
            b"psd_column_order": ",".join(self.columns).encode("utf-8"),
            b"psd_content_sha256": content_sha256.encode("ascii"),
            b"psd_derived_encoding": DERIVED_ARTIFACT_ENCODING.encode("utf-8"),
            b"psd_derived_parent_dataset": parent.dataset_id.encode("utf-8"),
            b"psd_derived_parent_digest": parent.manifest_digest.encode("ascii"),
        }


def derived_schema_with_metadata(spec: DerivedTableSpec, metadata: dict[bytes, bytes]) -> pa.Schema:
    """Return *spec*'s Arrow schema carrying *metadata*."""
    return _WITH_METADATA(spec.arrow_schema, metadata)


def write_derived_table(
    table: pa.Table,
    path: Path,
    spec: DerivedTableSpec,
    *,
    parent: DerivedParent,
    profile: ParquetProfile = PARQUET_PROFILE,
) -> ParquetWriteResult:
    """Order, digest, and write one derived artifact.

    The ordering happens here, before the digest, so the digest describes the artifact
    as written rather than as handed over.

    Args:
        table: The artifact's rows, in any order.
        path: Destination path. Parent directories are created.
        spec: The artifact's declared shape.
        parent: The canonical dataset this artifact projects.
        profile: The pinned Parquet writer options, shared with canonical tables so both
            artifact kinds have one on-disk profile.

    Returns:
        The write result, including both digests and the row count.

    Raises:
        DerivedArtifactError: The table does not match the declared schema.
    """
    ordered = spec.order(table)
    digest = content_digest(ordered, table_name=spec.name)
    schema = derived_schema_with_metadata(spec, spec.metadata(content_sha256=digest, parent=parent))
    payload = pa.Table.from_batches(ordered.to_batches(), schema=schema)
    path.parent.mkdir(parents=True, exist_ok=True)
    _WRITE_TABLE(
        payload,
        path,
        compression=profile.compression,
        compression_level=profile.compression_level,
        row_group_size=profile.row_group_size,
        version=profile.version,
        data_page_version=profile.data_page_version,
        use_dictionary=profile.use_dictionary,
        write_statistics=profile.write_statistics,
    )
    return ParquetWriteResult(
        path=path,
        sha256=sha256_file(path),
        content_sha256=digest,
        byte_size=path.stat().st_size,
        row_count=ordered.num_rows,
        compression=profile.compression,
    )


def read_derived_table(path: Path, spec: DerivedTableSpec) -> pa.Table:
    """Read a derived artifact and check it against its declaration.

    Args:
        path: The artifact.
        spec: The declared shape.

    Returns:
        The artifact's rows, in canonical order.

    Raises:
        DerivedArtifactError: The file is missing, has a different schema, or its
            recorded digest does not describe the rows it contains.
    """
    if not path.is_file():
        msg = f"No derived artifact at {path}."
        raise DerivedArtifactError(msg)
    table = _READ_TABLE(path)
    if not table.schema.equals(spec.arrow_schema, check_metadata=False):
        msg = (
            f"{path} does not match the declared schema for {spec.name!r}: "
            f"expected=[{', '.join(spec.columns)}] actual=[{', '.join(table.schema.names)}]"
        )
        raise DerivedArtifactError(msg)
    recorded = (table.schema.metadata or {}).get(b"psd_content_sha256", b"").decode("ascii")
    actual = content_digest(table, table_name=spec.name)
    if recorded and recorded != actual:
        msg = (
            f"{path} records content digest {recorded} but its rows digest to {actual}; "
            "the artifact does not describe itself."
        )
        raise DerivedArtifactError(msg)
    return spec.order(table)


def derived_metadata(path: Path) -> dict[str, str]:
    """Return a derived artifact's recorded key/value block as decoded text.

    Reads only the metadata, so a caller can see what an artifact claims about itself
    without loading its rows.
    """
    handle = pq.ParquetFile(path)
    try:
        block = handle.schema_arrow
    finally:
        handle.close()
    metadata = block.metadata or {}
    return {
        key.decode("utf-8", errors="replace"): value.decode("utf-8", errors="replace")
        for key, value in metadata.items()
    }


class DerivedArtifactManifest(BaseModel):
    """Machine-readable record of one derived artifact.

    A canonical dataset's manifest is its own authority. A derived artifact has no
    independent authority at all -- it is a projection -- so its manifest says so: it
    names the canonical dataset it projects, that dataset's manifest digest, and the
    transformation that produced it. Without those fields a derived artifact would be
    indistinguishable from an independent observation of the same athletes.

    Attributes:
        artifact_name: Logical artifact name, matching the table's declared name.
        artifact_relative_path: Path relative to the dataset directory, as a portable
            relative POSIX path.
        artifact_sha256: Lowercase hex SHA-256 of the artifact's file bytes.
        artifact_content_sha256: SHA-256 of the canonical logical encoding, computed by
            the same encoder canonical tables use, so the two are comparable.
        artifact_byte_size: Artifact size in bytes.
        row_count: Rows in the artifact.
        ordering: The artifact's canonical total ordering.
        parent_dataset_id: The canonical dataset this artifact projects.
        parent_manifest_digest: SHA-256 of that dataset's manifest bytes.
        lineage: Transformation steps, in the same vocabulary canonical manifests use.
        created_at: Timezone-aware UTC creation instant.
        notes: Human-readable notes.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    artifact_name: str = Field(min_length=1, max_length=128)
    artifact_relative_path: str = Field(min_length=1, max_length=1024)
    artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    artifact_content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    artifact_byte_size: int = Field(ge=0)
    row_count: int = Field(ge=0)
    ordering: tuple[str, ...]
    parent_dataset_id: str = Field(min_length=1, max_length=160)
    parent_manifest_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    lineage: tuple[LineageEntry, ...] = ()
    created_at: datetime
    notes: str | None = Field(default=None, max_length=4096)

    @field_validator("created_at")
    @classmethod
    def _require_aware_created_at(cls, value: datetime) -> datetime:
        return require_aware(value, field="created_at")

    def to_json_bytes(self) -> bytes:
        """Serialize deterministically: sorted keys, fixed indentation, UTF-8."""
        text = json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            separators=(",", ": "),
        )
        return (text + "\n").encode("utf-8")

    def write_json(self, path: Path) -> bytes:
        """Write the manifest to *path* and return the bytes written."""
        data = self.to_json_bytes()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return data

    def digest(self) -> str:
        """Return the SHA-256 of this manifest's own canonical JSON bytes."""
        return hashlib.sha256(self.to_json_bytes()).hexdigest()


def read_derived_manifest(path: Path) -> DerivedArtifactManifest:
    """Read a derived artifact's manifest.

    Args:
        path: The manifest file.

    Returns:
        The parsed manifest.

    Raises:
        DerivedArtifactError: The file is missing or does not describe a derived
            artifact, which would mean the path is not what the caller thinks it is.
    """
    if not path.is_file():
        msg = f"No derived artifact manifest at {path}."
        raise DerivedArtifactError(msg)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        msg = f"Unreadable derived artifact manifest at {path}: {error}"
        raise DerivedArtifactError(msg) from error
    try:
        return DerivedArtifactManifest.model_validate(payload)
    except ValueError as error:
        msg = f"Invalid derived artifact manifest at {path}: {error}"
        raise DerivedArtifactError(msg) from error
