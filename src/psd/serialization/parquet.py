"""Deterministic Parquet persistence.

Zstandard is the default compression per the locked stack. The writer options
below are pinned because each of them can otherwise vary the byte stream without
changing the logical table:

``use_dictionary=False``
    Dictionary page ordering depends on first-seen value order.
``write_statistics=False``
    Statistics are redundant here and are an avoidable source of variation.
``data_page_version="1.0"``
    The newer page header layout is not needed for these artifacts.
``row_group_size``
    Fixed, so identical data always produces identical row groups.
``version="2.6"``
    A single, explicit Parquet format version.

Byte-level determinism still depends on the PyArrow build, because Parquet embeds
a ``created_by`` string. Use ``content_sha256`` (see
:mod:`psd.serialization.canonical`) when comparing artifacts across environments;
use ``sha256`` for tamper evidence within one environment.

Typing note
-----------

PyArrow ships no inline types, and the community stubs describe the Parquet
writer/reader surface with closed ``Literal`` parameter unions that do not match
the runtime API's accepted values. Rather than relaxing project-wide strictness,
this module reaches the two untyped entry points through narrow, documented
:func:`getattr` shims with explicit local signatures. Everything else stays fully
typed.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pyarrow as pa
import pyarrow.parquet as pq

from psd.schema.registry import table_spec
from psd.schema.version import ID_SCHEME, SCHEMA_VERSION
from psd.serialization.canonical import content_digest

__all__ = (
    "PARQUET_PROFILE",
    "ParquetProfile",
    "ParquetWriteResult",
    "read_parquet",
    "sha256_file",
    "write_parquet",
)

type ParquetVersion = Literal["1.0", "2.4", "2.6"]
type DataPageVersion = Literal["1.0", "2.0"]
type CompressionCodec = str

_WRITE_TABLE: Callable[..., None] = getattr(pq, "write_table")
_READ_TABLE: Callable[..., pa.Table] = getattr(pq, "read_table")

_DEFAULT_COMPRESSION: CompressionCodec = "zstd"
_DEFAULT_COMPRESSION_LEVEL = 3
_DEFAULT_ROW_GROUP_SIZE = 65536


@dataclass(frozen=True, slots=True)
class ParquetWriteResult:
    """Outcome of one deterministic Parquet write.

    Attributes:
        path: Absolute path written.
        sha256: SHA-256 of the file bytes.
        content_sha256: SHA-256 of the canonical logical encoding.
        byte_size: File size in bytes.
        row_count: Rows written.
        compression: Compression codec used.
    """

    path: Path
    sha256: str
    content_sha256: str
    byte_size: int
    row_count: int
    compression: str


@dataclass(frozen=True, slots=True)
class ParquetProfile:
    """Pinned writer options.

    Attributes:
        compression: Compression codec.
        compression_level: Codec level.
        row_group_size: Rows per row group.
        version: Parquet format version.
        data_page_version: Data page layout version.
        use_dictionary: Whether dictionary encoding is enabled.
        write_statistics: Whether column statistics are written.
    """

    compression: CompressionCodec = _DEFAULT_COMPRESSION
    compression_level: int = _DEFAULT_COMPRESSION_LEVEL
    row_group_size: int = _DEFAULT_ROW_GROUP_SIZE
    version: ParquetVersion = "2.6"
    data_page_version: DataPageVersion = "1.0"
    use_dictionary: bool = False
    write_statistics: bool = False


#: The pinned default writer profile.
PARQUET_PROFILE = ParquetProfile()


def write_parquet(
    table: pa.Table,
    path: Path,
    *,
    table_name: str,
    profile: ParquetProfile = PARQUET_PROFILE,
) -> ParquetWriteResult:
    """Write *table* deterministically and return its digests.

    The stored Arrow schema metadata records the canonical schema version, table
    name, identifier scheme, declared column order, and content digest, so an
    artifact is self-describing even without its manifest.

    Args:
        table: Table to write. Sort it with
            :func:`psd.serialization.ordering.canonical_order` first.
        path: Destination path. Parent directories are created.
        table_name: Canonical table name.
        profile: Writer options.

    Returns:
        The write result, including both digests.
    """
    spec = table_spec(table_name)
    digest = content_digest(table, table_name=table_name)
    metadata = {
        b"psd_table": table_name.encode("utf-8"),
        b"psd_schema_version": SCHEMA_VERSION.tag.encode("utf-8"),
        b"psd_id_scheme": ID_SCHEME.encode("utf-8"),
        b"psd_column_order": ",".join(spec.columns).encode("utf-8"),
        b"psd_content_sha256": digest.encode("ascii"),
    }
    with_metadata: Callable[[dict[bytes, bytes]], pa.Schema] = getattr(
        spec.arrow_schema(), "with_metadata"
    )
    schema = with_metadata(metadata)
    payload = pa.Table.from_batches(table.to_batches(), schema=schema)
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
        row_count=table.num_rows,
        compression=profile.compression,
    )


def read_parquet(path: Path, *, table_name: str) -> pa.Table:
    """Read a canonical Parquet artifact and verify its recorded metadata.

    The stored schema is checked against the canonical schema before the data is
    returned, so a stale or foreign artifact fails fast and loudly instead of
    being reinterpreted.

    Args:
        path: Artifact path.
        table_name: Expected canonical table name.

    Returns:
        The Arrow table.

    Raises:
        ValueError: The file's schema or self-description disagrees with the
            canonical schema.
    """
    parquet_file = pq.ParquetFile(path)
    schema = parquet_file.schema_arrow
    expected = table_spec(table_name).arrow_schema()
    if not schema.equals(expected, check_metadata=False):
        msg = (
            f"{path} does not match the canonical schema for {table_name!r}: "
            f"expected=[{', '.join(expected.names)}] actual=[{', '.join(schema.names)}]"
        )
        raise ValueError(msg)
    _require_metadata(path, schema, table_name)
    return _READ_TABLE(path)


def _require_metadata(path: Path, schema: pa.Schema, table_name: str) -> None:
    """Reject an artifact whose recorded self-description disagrees."""
    metadata = schema.metadata or {}
    recorded_table = metadata.get(b"psd_table", b"").decode("utf-8")
    if recorded_table and recorded_table != table_name:
        msg = f"{path} declares table {recorded_table!r} but was opened as {table_name!r}."
        raise ValueError(msg)
    recorded_version = metadata.get(b"psd_schema_version", b"").decode("utf-8")
    if recorded_version and recorded_version != SCHEMA_VERSION.tag:
        msg = (
            f"{path} was written with schema {recorded_version!r} but this install is "
            f"{SCHEMA_VERSION.tag!r}."
        )
        raise ValueError(msg)


def sha256_file(path: Path) -> str:
    """Return the lowercase hex SHA-256 of a file's bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
