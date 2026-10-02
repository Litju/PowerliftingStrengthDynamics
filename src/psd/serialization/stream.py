"""Streaming canonical Parquet persistence.

Why a second writer
-------------------

:func:`psd.serialization.parquet.write_parquet` takes a complete Arrow table. That is
the right shape for a fixture or a single athlete's history, and it stays the writer
used everywhere except one place: a multi-million-row corpus, where holding a whole
table in memory just to hash it and then write it is the difference between a build
that finishes and one that does not.

This module writes the same artifact from ordered batches, and the equivalence is
checkable rather than asserted: a test writes the same table both ways and compares
the file bytes, the content digest, and the row count.

The guarantees are identical:

* the same pinned :class:`~psd.serialization.parquet.ParquetProfile` options;
* the same self-describing Arrow schema metadata, including the content digest;
* the same canonical ordering, supplied by the caller batch by batch;
* the same ``content_sha256``, computed incrementally over exactly the bytes
  :func:`~psd.serialization.canonical.canonical_bytes` would produce.

The row count is part of that encoding's header, so a digest over "header then rows" can
only be opened once the count is final. That is what shapes this module: a cheap counting
pass over the batches, then a single pass that both hashes and writes them.
:meth:`StreamingTableWriter.count_batch` encodes nothing -- it validates and counts -- and
:meth:`StreamingTableWriter.write_batch` folds each batch into a digest that
:meth:`StreamingTableWriter.open` has already primed with the header, while appending the
same rows to the Parquet file. The digest therefore cannot describe rows the file does not
contain, and the two are never computed by separate passes that could disagree.

The content digest still has to reach the file's metadata *after* the rows, so
:meth:`StreamingTableWriter.close` finishes the Parquet file and then re-attaches it with
the final digest in its key/value block.

The counting requirement is the writer's one sharp edge. It is deliberate: it means the
content digest inside the file is a value computed *from* the rows rather than a claim made
before them.

Batches and row groups
----------------------

Parquet splits a write into row groups, and where a batch ends decides where a row group
ends. That only affects the file's *layout*, never its content: a table written in batches
of 1000 rows has the same rows, the same values, and the same ``content_sha256`` as the same
table written in one piece, whatever the batch size. It is the ``sha256`` -- byte-level
tamper evidence -- that reflects the batching.

So when batches are whole multiples of the profile's ``row_group_size``, this writer's
output is byte-identical to :func:`~psd.serialization.parquet.write_parquet`'s, and a test
asserts exactly that. When they are not, the artifact is still correct and still
self-describing; it is merely laid out in more, smaller row groups. A writer that refused
misaligned batches would be refusing a legitimate memory choice, so it does not.

:func:`psd.ingest.openpowerlifting.transform.BuildConfig` still defaults ``batch_rows`` to a
whole number of row groups, because the corpus build is the case where byte identity with
the single-shot writer is worth having.

Typing note
-----------

PyArrow ships no inline types and the community stubs describe ``ParquetWriter`` with
parameter unions that do not match the runtime signature. As in
:mod:`psd.serialization.parquet`, the untyped entry points are reached through narrow,
documented :func:`getattr` shims with explicit local signatures, so project-wide
strictness is kept rather than relaxed.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import pyarrow as pa
import pyarrow.parquet as pq

from psd.paths import resolve_within_data_root
from psd.provenance.environment import EnvironmentSnapshot, runtime_snapshot
from psd.provenance.manifest import (
    ArtifactRef,
    DatasetKind,
    DatasetManifest,
    LineageEntry,
)
from psd.provenance.sources import SourceRecord
from psd.schema.registry import table_names, table_spec
from psd.schema.version import ID_SCHEME, SCHEMA_VERSION
from psd.serialization.canonical import content_header, encode_rows
from psd.serialization.parquet import (
    PARQUET_PROFILE,
    ParquetProfile,
    ParquetWriteResult,
    sha256_file,
)
from psd.serialization.table import empty_table

__all__ = (
    "StreamingDatasetSpec",
    "StreamingDatasetWriter",
    "StreamingTableWriter",
    "StreamingWriteError",
    "canonical_schema_with_metadata",
)

_ParquetWriterFactory = Callable[..., "pq.ParquetWriter"]
_ParquetFileFactory = Callable[..., "pq.ParquetFile"]
_WRITE_TABLE: _ParquetWriterFactory = cast("_ParquetWriterFactory", getattr(pq, "ParquetWriter"))
_PARQUET_FILE: _ParquetFileFactory = cast("_ParquetFileFactory", getattr(pq, "ParquetFile"))
_READ_ROW_GROUP: Callable[..., pa.Table] = getattr(pq.ParquetFile, "read_row_group")


class StreamingWriteError(RuntimeError):
    """Raised when a streaming canonical write is used incorrectly."""


def _writer_options(profile: ParquetProfile) -> dict[str, Any]:
    """Return the pinned writer options, identical to the single-shot writer's.

    ``row_group_size`` is the one option Parquet takes per write rather than per writer,
    so it is returned separately by :func:`_write_options`.
    """
    return {
        "compression": profile.compression,
        "compression_level": profile.compression_level,
        "version": profile.version,
        "data_page_version": profile.data_page_version,
        "use_dictionary": profile.use_dictionary,
        "write_statistics": profile.write_statistics,
    }


def _write_options(profile: ParquetProfile) -> dict[str, Any]:
    """Return the per-write pinned options."""
    return {"row_group_size": profile.row_group_size}


def canonical_schema_with_metadata(table_name: str, *, content_sha256: str) -> pa.Schema:
    """Return the canonical Arrow schema with its self-describing metadata.

    ``content_sha256`` is supplied by the caller because it is only known once every
    batch has been written: the canonical encoding declares the row count in its
    header, so the digest cannot be opened before the count is final.
    """
    spec = table_spec(table_name)
    schema = spec.arrow_schema()
    with_metadata: Callable[[dict[bytes, bytes]], pa.Schema] = getattr(schema, "with_metadata")
    return with_metadata(
        {
            b"psd_table": table_name.encode("utf-8"),
            b"psd_schema_version": SCHEMA_VERSION.tag.encode("utf-8"),
            b"psd_id_scheme": ID_SCHEME.encode("utf-8"),
            b"psd_column_order": ",".join(spec.columns).encode("utf-8"),
            b"psd_content_sha256": content_sha256.encode("ascii"),
        }
    )


def _require_canonical_batch(table_name: str, batch: pa.Table) -> None:
    """Reject a batch that is empty or does not carry the canonical schema.

    Raises:
        StreamingWriteError: The batch is unusable for this table.
    """
    if batch.num_rows == 0:
        msg = f"{table_name}: refusing to hash or write an empty batch."
        raise StreamingWriteError(msg)
    expected = table_spec(table_name).arrow_schema()
    if not batch.schema.equals(expected, check_metadata=False):
        msg = (
            f"{table_name}: batch schema does not match the canonical schema; "
            f"expected=[{', '.join(expected.names)}] "
            f"actual=[{', '.join(batch.schema.names)}]"
        )
        raise StreamingWriteError(msg)


class StreamingTableWriter:
    """Write one canonical table from ordered batches.

    The canonical encoding declares its row count in a header that precedes the rows, so a
    digest over "header then rows" cannot be opened until the count is final. That gives the
    writer its shape: a cheap counting pass, then a single pass that both hashes and writes.

    * :meth:`count_batch` validates each batch and records its size. It encodes nothing, so
      a corpus's counting pass costs a schema comparison per batch and nothing else.
    * :meth:`open` primes the digest with the header, now that the count is known.
    * :meth:`write_batch` folds each batch's rows into that digest *and* appends them to the
      Parquet file, so the digest and the bytes can never describe different rows.
    * :meth:`close` attaches the digest to the file's metadata.

    An earlier design fed the same batches twice, once to hash and once to write, and
    combined a digest-of-the-body with the header. That was wrong: hashing a body digest is
    not the same as hashing the bytes, so the two writers produced different ``content``
    digests over identical rows, and a manifest would have recorded a digest no other tool
    could reproduce.

    The writer concatenates batches and never reorders them. That is deliberate: the
    caller's partitioning strategy is how a corpus build keeps memory bounded, and a writer
    that silently sorted would hide whether that strategy was right.
    """

    def __init__(
        self,
        table_name: str,
        path: Path,
        *,
        profile: ParquetProfile = PARQUET_PROFILE,
    ) -> None:
        self.table_name = table_name
        self.path = path
        self._profile = profile
        self._columns = tuple(table_spec(table_name).arrow_schema().names)
        self._writer: pq.ParquetWriter | None = None
        self._digest: Any = hashlib.sha256()
        self._counted_rows = 0
        self._written_rows = 0
        self._written_batches = 0
        self._batch_rows: list[int] = []

    @property
    def row_count(self) -> int:
        """Rows counted so far, which is the artifact's final row count."""
        return self._counted_rows

    def count_batch(self, batch: pa.Table) -> None:
        """Validate one batch and record its row count.

        Raises:
            StreamingWriteError: The batch is empty or does not carry the canonical
                schema for this table.
        """
        _require_canonical_batch(self.table_name, batch)
        self._counted_rows += batch.num_rows
        self._batch_rows.append(batch.num_rows)

    def write_batch(self, batch: pa.Table) -> None:
        """Hash one canonically ordered batch and append it to the artifact.

        Raises:
            StreamingWriteError: The writer is not open, the batch does not carry the
                canonical schema for this table, or the batch is one this writer never
                counted.
        """
        if self._writer is None:
            msg = f"{self.table_name}: write_batch() called before open()."
            raise StreamingWriteError(msg)
        _require_canonical_batch(self.table_name, batch)
        if self._written_batches >= len(self._batch_rows):
            msg = (
                f"{self.table_name}: write_batch() received a batch the counting pass never "
                f"saw. {len(self._batch_rows)} batches were counted and "
                f"{self._written_batches} written, so the digest's declared row count would "
                "not describe the rows the file contains."
            )
            raise StreamingWriteError(msg)
        self._digest.update(b"".join(encode_rows(batch)))
        self._writer.write_table(batch, **_write_options(self._profile))
        self._written_batches += 1
        self._written_rows += batch.num_rows

    def open(self) -> StreamingTableWriter:
        """Prime the content digest with its header and create the Parquet file.

        The header declares the row count the counting pass found, so the digest is now
        open over exactly the bytes that follow it. The file itself is written under the
        bare canonical schema, because the digest is not final until the last batch; the
        metadata is attached by the rewrite in :meth:`close`.

        Raises:
            StreamingWriteError: The writer is already open.
        """
        if self._writer is not None:
            msg = f"{self.table_name}: open() called on a writer that is already open."
            raise StreamingWriteError(msg)
        self._digest = hashlib.sha256(
            content_header(self._columns, table_name=self.table_name, row_count=self._counted_rows)
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = _WRITE_TABLE(
            self.path,
            table_spec(self.table_name).arrow_schema(),
            **_writer_options(self._profile),
        )
        return self

    def close(self) -> ParquetWriteResult:
        """Finalize the artifact and return its digests.

        A table that received no batch is still written, carrying the canonical schema
        with zero rows: a dataset's table set is uniform, so a reader walking it never
        has to tell "no rows" apart from "lost". An empty table is written directly rather
        than rewritten, and still carries its self-description.

        Raises:
            StreamingWriteError: The writer was never opened, or the written rows are not
                the counted rows.
        """
        if self._writer is None:
            msg = f"{self.table_name}: close() called before open()."
            raise StreamingWriteError(msg)
        if self._written_rows != self._counted_rows:
            msg = (
                f"{self.table_name}: counted {self._counted_rows} rows but wrote "
                f"{self._written_rows}. The digest's header declares the counted total, so "
                "the file's self-description would not describe its contents."
            )
            raise StreamingWriteError(msg)

        content_sha256 = self._digest.hexdigest()

        self._writer.close()
        self._writer = None
        schema = canonical_schema_with_metadata(self.table_name, content_sha256=content_sha256)
        if self._counted_rows == 0:
            # An empty table still gets one empty row group, because that is what the
            # single-shot writer produces for it, and the two must not differ.
            empty_writer = _WRITE_TABLE(self.path, schema, **_writer_options(self._profile))
            empty_writer.write_table(empty_table(self.table_name), **_write_options(self._profile))
            empty_writer.close()
        else:
            _rewrite_with_metadata(self.path, schema, profile=self._profile)
        return ParquetWriteResult(
            path=self.path,
            sha256=sha256_file(self.path),
            content_sha256=content_sha256,
            byte_size=self.path.stat().st_size,
            row_count=self._counted_rows,
            compression=self._profile.compression,
        )


def _rewrite_with_metadata(path: Path, schema: pa.Schema, *, profile: ParquetProfile) -> None:
    """Re-write a finished Parquet file with its final self-describing metadata.

    The file is streamed one row group at a time rather than read whole. A reported-result
    table of twenty-eight million rows does not fit in memory, and a rewrite that needed
    it to would defeat the reason this module exists. Each source row group is written back
    as exactly one output row group, so the layout the streaming pass produced is preserved
    and only the Arrow key/value block changes.

    Attaching the metadata *before* the rows is not available, because the content digest
    depends on a row count that is not final until the last batch has been hashed.
    """
    staged = path.with_name(f"{path.name}.metadata")
    source: pq.ParquetFile | None = None
    try:
        source = _PARQUET_FILE(path)
        groups = source.metadata.num_row_groups
        writer = _WRITE_TABLE(staged, schema, **_writer_options(profile))
        for index in range(groups):
            group = _READ_ROW_GROUP(source, index)
            writer.write_table(group, row_group_size=group.num_rows)
        writer.close()
        # PyArrow memory-maps the source, and Windows will not replace a file that is
        # still mapped. The handle has to go before the new file can take its name.
        source.close()
        source = None
        staged.replace(path)
    except BaseException:
        if source is not None:
            source.close()
        staged.unlink(missing_ok=True)
        raise


@dataclass(frozen=True, slots=True)
class StreamingDatasetSpec:
    """Metadata recorded alongside a streaming-written dataset.

    Attributes:
        dataset_id: Stable dataset identifier.
        dataset_kind: Competition, training, prospective, synthetic, or reference.
        created_at: Timezone-aware UTC creation timestamp.
        sources: Every source the dataset draws on.
        lineage: Transformation steps.
        dataset_name: Human-readable name; defaults to the dataset id.
        notes: Manifest notes.
        environment: Environment snapshot; captured live when omitted.
    """

    dataset_id: str
    dataset_kind: DatasetKind
    created_at: datetime
    sources: tuple[SourceRecord, ...] = ()
    lineage: tuple[LineageEntry, ...] = ()
    dataset_name: str | None = None
    notes: str | None = None
    environment: EnvironmentSnapshot | None = field(default=None)


class StreamingDatasetWriter:
    """Write a whole canonical dataset from ordered batches, one table at a time."""

    def __init__(self, relative: Path | str, *, data_root: Path | None = None) -> None:
        self._tables_dir = resolve_within_data_root(
            Path(relative) / "tables", data_root=data_root, create=True
        )
        self._writers: dict[str, StreamingTableWriter] = {
            name: StreamingTableWriter(name, self._tables_dir / f"{name}.parquet")
            for name in table_names()
        }
        self._opened = False

    def _require(self, table_name: str) -> StreamingTableWriter:
        """Return the writer for *table_name*.

        Raises:
            StreamingWriteError: The table is not part of this dataset.
        """
        writer = self._writers.get(table_name)
        if writer is None:
            msg = (
                f"Unknown canonical table {table_name!r}. Known tables: {', '.join(table_names())}."
            )
            raise StreamingWriteError(msg)
        return writer

    def open(self) -> StreamingDatasetWriter:
        """Prime every table's content digest and create every table file.

        Call this after the counting pass, so each digest opens over a header that declares
        the row count that pass found.
        """
        for writer in self._writers.values():
            writer.open()
        self._opened = True
        return self

    def count_batch(self, table_name: str, batch: pa.Table) -> None:
        """Validate one batch of *table_name* and record its row count."""
        self._require(table_name).count_batch(batch)

    def write_batch(self, table_name: str, batch: pa.Table) -> None:
        """Append one canonically ordered batch to *table_name*'s Parquet file.

        Raises:
            StreamingWriteError: The table is not part of the dataset, or the writer is
                not open.
        """
        self._require(table_name).write_batch(batch)

    def row_count(self, table_name: str) -> int:
        """Rows hashed so far for *table_name*."""
        return self._require(table_name).row_count

    def close(self, spec: StreamingDatasetSpec) -> DatasetManifest:
        """Finalize every table and write the dataset manifest.

        Raises:
            StreamingWriteError: The writer was never opened.
        """
        if not self._opened:
            msg = "close() called before open()."
            raise StreamingWriteError(msg)
        artifacts: list[ArtifactRef] = []
        for name in table_names():
            result = self._writers[name].close()
            artifacts.append(
                ArtifactRef(
                    name=name,
                    relative_path=f"tables/{name}.parquet",
                    sha256=result.sha256,
                    content_sha256=result.content_sha256,
                    byte_size=result.byte_size,
                    row_count=result.row_count,
                )
            )
        manifest = DatasetManifest(
            dataset_id=spec.dataset_id,
            dataset_name=spec.dataset_name or spec.dataset_id,
            dataset_kind=spec.dataset_kind,
            schema_version=SCHEMA_VERSION.tag,
            created_at=spec.created_at,
            environment=spec.environment if spec.environment is not None else runtime_snapshot(),
            sources=spec.sources,
            artifacts=tuple(artifacts),
            lineage=spec.lineage,
            notes=spec.notes,
        )
        manifest.write_json(self._tables_dir.parent / "manifest.json")
        return manifest
