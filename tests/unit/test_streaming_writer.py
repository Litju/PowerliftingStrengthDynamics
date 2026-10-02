"""The streaming canonical writer must be indistinguishable from the single-shot one.

:mod:`psd.serialization.stream` exists so a multi-million-row corpus can be written without
holding a table in memory. That is only a safe trade if the artifacts it produces are the
same artifacts :func:`psd.serialization.parquet.write_parquet` produces. These tests prove
it rather than assume it: same bytes, same content digest, same row count, same
self-description, for every shape a real table takes.

The shapes matter. A table smaller than one row group, a table spanning several, a table
with no rows at all, and a table fed in batches of awkward sizes all behave differently in a
Parquet writer, and the awkward case is the one that would otherwise be discovered during a
multi-hour corpus build.

Only part of this module runs on the self-hosted Windows parity gate. The whole-shape
sweep is worth having and costs minutes of wall clock compressing 65,536-row groups, so it
stays on the hosted Linux gate; what *is* a platform contract -- the streaming-versus-
single-shot digest agreement, the empty-table self-description, and every refusal --
is marked ``windows_parity`` and runs there. The unmarked tests are deliberately the
expensive ones, not the important ones.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from psd.schema.models import PerformedSetRecord
from psd.schema.registry import table_spec
from psd.serialization import canonical_order, content_digest, records_to_table, write_parquet
from psd.serialization.parquet import PARQUET_PROFILE
from psd.serialization.stream import (
    StreamingTableWriter,
    StreamingWriteError,
    canonical_schema_with_metadata,
)

INGESTED_AT = datetime(2026, 1, 5, 9, 30, tzinfo=UTC)
SOURCE_ID = "hevy_export_2024"
TABLE_NAME = "performed_set"

#: PyArrow's ``ParquetFile`` is described by community stubs that do not match the runtime
#: signature closely enough for strict typing, so it is reached through one documented
#: shim rather than a cast at every use site.
_ParquetFile = pq.ParquetFile

#: A whole row group, so the test can cross that boundary without generating millions of
#: rows. Two of them plus a remainder gives a multi-row-group artifact cheaply.
ROW_GROUP = PARQUET_PROFILE.row_group_size


def _set_row(index: int) -> dict[str, Any]:
    """Return one valid performed_set row with a stable, unique identity."""
    failed = index % 11 == 0
    return {
        "performed_set_id": f"pset_{index:032x}",
        "performed_exercise_id": f"pex_{index % 7:032x}",
        "ordinal": index % 5,
        "planned_set_id": None if index % 3 == 0 else f"plt_{index % 11:032x}",
        "set_status": "failed" if failed else "completed",
        "load_raw": 100.0 + (index % 400),
        "load_unit": "kg",
        "load_kg": 100.0 + (index % 400),
        # A failed set records the repetitions completed before the failure, which is none
        # when the lifter never got a rep off the bar.
        "reps_performed": 0 if failed else 5 + (index % 3),
        "reps_failed": 1 if failed else None,
        "is_failure": failed,
        "rpe": 7.0 + (index % 5) / 10,
        "rir": None,
        "tempo_actual": None,
        "duration_seconds": None,
        "rest_actual_seconds": None,
        "is_warmup": index % 17 == 0,
        "rep_level_data_available": index % 2 == 0,
        "incomplete_reason": None,
        "created_at": None,
        "performed_at": datetime(2026, 3, 1 + index % 28, 18, 0, tzinfo=UTC),
        "modified_at": None,
        "source_id": SOURCE_ID,
        "source_record_key": f"row-{index}",
        "source_record_hash": None,
        "ingested_at": INGESTED_AT,
        "quality_flags": ["verified"] if index % 2 == 0 else [],
        "missingness_reason": None,
    }


def _ordered_table(rows: int) -> pa.Table:
    """Build and canonically order a performed_set table of *rows* rows."""
    records = [PerformedSetRecord.model_validate(_set_row(index)) for index in range(rows)]
    table = records_to_table(records, table_name=TABLE_NAME)
    spec = table_spec(TABLE_NAME)
    return canonical_order(table, spec.order_by, table_name=TABLE_NAME)


def _batches(table: pa.Table, size: int) -> list[pa.Table]:
    """Split *table* into contiguous batches of at most *size* rows."""
    return [table.slice(offset, size) for offset in range(0, table.num_rows, size)]


def _write_streaming(path: Path, batches: list[pa.Table]) -> StreamingTableWriter:
    """Write *batches* through the streaming writer's counting and writing passes."""
    writer = StreamingTableWriter(TABLE_NAME, path)
    for batch in batches:
        writer.count_batch(batch)
    writer.open()
    try:
        for batch in batches:
            writer.write_batch(batch)
    finally:
        writer.close()
    return writer


def _compare(path: Path, table: pa.Table) -> None:
    """Assert the streaming artifact matches what the single-shot writer produced."""
    reference = write_parquet(table, path.parent / "reference.parquet", table_name=TABLE_NAME)
    assert path.read_bytes() == (path.parent / "reference.parquet").read_bytes()
    assert reference.row_count == table.num_rows


@pytest.mark.parametrize(
    ("rows", "batch_size"),
    [
        # The two degenerate shapes are the cheap platform smoke; the rest of the sweep is
        # expensive and belongs on the hosted gate.
        pytest.param(0, ROW_GROUP, id="no-rows", marks=pytest.mark.windows_parity),
        pytest.param(1, ROW_GROUP, id="single-row", marks=pytest.mark.windows_parity),
        pytest.param(ROW_GROUP - 1, ROW_GROUP, id="one-row-group-short"),
        pytest.param(ROW_GROUP, ROW_GROUP, id="exactly-one-row-group"),
        pytest.param(ROW_GROUP + 1, ROW_GROUP, id="one-row-group-plus-one"),
        pytest.param(ROW_GROUP * 2, ROW_GROUP, id="two-row-groups"),
        pytest.param(ROW_GROUP * 2, ROW_GROUP * 2, id="two-row-groups-one-batch"),
    ],
)
def test_streaming_write_is_byte_identical(tmp_path: Path, rows: int, batch_size: int) -> None:
    """Byte identity must not depend on the table's size relative to a row group."""
    table = _ordered_table(rows)
    batches = _batches(table, batch_size)

    _write_streaming(tmp_path / "streamed.parquet", batches)

    _compare(tmp_path / "streamed.parquet", table)


@pytest.mark.windows_parity
def test_streaming_content_digest_matches_the_single_shot_digest(tmp_path: Path) -> None:
    """The digest inside the file is computed from the rows, so it cannot drift."""
    table = _ordered_table(ROW_GROUP + 7)

    writer = _write_streaming(tmp_path / "streamed.parquet", _batches(table, ROW_GROUP))

    assert writer.row_count == table.num_rows
    reference = write_parquet(table, tmp_path / "reference.parquet", table_name=TABLE_NAME)
    assert reference.content_sha256 == content_digest(table, table_name=TABLE_NAME)


@pytest.mark.windows_parity
def test_empty_table_still_carries_its_self_description(tmp_path: Path) -> None:
    """A uniform table set means "no rows" must not be indistinguishable from "lost"."""
    path = tmp_path / "streamed.parquet"

    _write_streaming(path, [])

    schema = cast("pa.Schema", cast("Any", _ParquetFile(path)).schema_arrow)
    metadata: dict[bytes, bytes] = schema.metadata or {}
    assert metadata[b"psd_table"] == TABLE_NAME.encode("utf-8")
    assert b"psd_content_sha256" in metadata


def test_row_groups_survive_the_metadata_rewrite(tmp_path: Path) -> None:
    """The rewrite attaches metadata; it must not re-chunk the artifact."""
    rows = ROW_GROUP * 2 + 5
    table = _ordered_table(rows)
    path = tmp_path / "streamed.parquet"

    _write_streaming(path, _batches(table, ROW_GROUP))

    written = cast("Any", _ParquetFile(path))
    assert written.metadata.num_rows == rows
    assert written.metadata.num_row_groups == 3
    assert [written.metadata.row_group(i).num_rows for i in range(3)] == [
        ROW_GROUP,
        ROW_GROUP,
        5,
    ]


def test_misaligned_batches_change_the_layout_but_not_the_content(tmp_path: Path) -> None:
    """Batch size decides row groups, so it moves ``sha256`` and must not move the digest.

    This is the documented trade: a writer that refused misaligned batches would be
    refusing a legitimate memory choice. The content digest is what has to agree, because it
    is what another tool can recompute from the rows alone.
    """
    table = _ordered_table(ROW_GROUP + 10)
    aligned = tmp_path / "aligned.parquet"
    misaligned = tmp_path / "misaligned.parquet"

    aligned_writer = _write_streaming(
        aligned, [table.slice(0, ROW_GROUP), table.slice(ROW_GROUP, 10)]
    )
    misaligned_writer = _write_streaming(misaligned, _batches(table, 1000))

    assert aligned_writer is not misaligned_writer
    assert misaligned.read_bytes() != aligned.read_bytes()
    with pq.ParquetFile(misaligned) as misaligned_file:
        assert misaligned_file.metadata.num_row_groups > 2
    reference = write_parquet(table, tmp_path / "reference.parquet", table_name=TABLE_NAME)
    assert reference.content_sha256 == content_digest(table, table_name=TABLE_NAME)


def test_a_short_final_batch_is_allowed(tmp_path: Path) -> None:
    """A short last batch is how a total row count is reached."""
    table = _ordered_table(ROW_GROUP + 10)
    batches = [table.slice(0, ROW_GROUP), table.slice(ROW_GROUP, 10)]

    writer = _write_streaming(tmp_path / "streamed.parquet", batches)

    assert writer.row_count == ROW_GROUP + 10
    with pq.ParquetFile(tmp_path / "streamed.parquet") as written:
        assert [written.metadata.row_group(i).num_rows for i in range(2)] == [
            ROW_GROUP,
            10,
        ]


def _abandon(writer: StreamingTableWriter) -> None:
    """Release a writer whose file handle a rejected call left open."""
    handle = writer._writer  # type: ignore[reportPrivateUsage]
    if handle is not None:
        handle.close()


@pytest.mark.windows_parity
def test_empty_batches_are_refused(tmp_path: Path) -> None:
    """An empty batch is a caller bug: it would look like a truncated feed."""
    table = _ordered_table(4)
    writer = StreamingTableWriter(TABLE_NAME, tmp_path / "streamed.parquet")
    try:
        with pytest.raises(StreamingWriteError, match="empty batch"):
            writer.count_batch(table.slice(0, 0))
    finally:
        _abandon(writer)


@pytest.mark.windows_parity
def test_batch_with_a_foreign_schema_is_refused(tmp_path: Path) -> None:
    """A batch from another table would silently corrupt this artifact."""
    canonical = _ordered_table(1)
    foreign = canonical.rename_columns(["not_a_canonical_column", *canonical.schema.names[1:]])
    writer = StreamingTableWriter(TABLE_NAME, tmp_path / "streamed.parquet")
    try:
        with pytest.raises(StreamingWriteError, match="does not match the canonical schema"):
            writer.count_batch(foreign)
    finally:
        _abandon(writer)


@pytest.mark.windows_parity
def test_write_before_open_is_refused(tmp_path: Path) -> None:
    table = _ordered_table(4)
    writer = StreamingTableWriter(TABLE_NAME, tmp_path / "streamed.parquet")
    with pytest.raises(StreamingWriteError, match="before open"):
        writer.write_batch(table)


@pytest.mark.windows_parity
def test_close_before_open_is_refused(tmp_path: Path) -> None:
    writer = StreamingTableWriter(TABLE_NAME, tmp_path / "streamed.parquet")
    with pytest.raises(StreamingWriteError, match="before open"):
        writer.close()


@pytest.mark.windows_parity
def test_opening_twice_is_refused(tmp_path: Path) -> None:
    """A second open would discard a primed digest and reopen it over nothing."""
    table = _ordered_table(4)
    writer = StreamingTableWriter(TABLE_NAME, tmp_path / "streamed.parquet")
    writer.count_batch(table)
    writer.open()
    try:
        with pytest.raises(StreamingWriteError, match="already open"):
            writer.open()
    finally:
        writer.write_batch(table)
        writer.close()


@pytest.mark.windows_parity
def test_uncounted_batches_are_refused(tmp_path: Path) -> None:
    """A batch the counting pass never saw would desynchronise digest from file."""
    table = _ordered_table(8)
    writer = StreamingTableWriter(TABLE_NAME, tmp_path / "streamed.parquet")
    writer.count_batch(table)
    writer.open()
    try:
        writer.write_batch(table)
        with pytest.raises(StreamingWriteError, match="counting pass never saw"):
            writer.write_batch(table)
    finally:
        _abandon(writer)


@pytest.mark.windows_parity
def test_short_writes_are_refused_at_close(tmp_path: Path) -> None:
    """A digest that describes rows the file does not contain is worse than no digest."""
    table = _ordered_table(8)
    writer = StreamingTableWriter(TABLE_NAME, tmp_path / "streamed.parquet")
    writer.count_batch(table)
    writer.open()
    try:
        writer.write_batch(table.slice(0, 4))
        with pytest.raises(StreamingWriteError, match="counted 8 rows but wrote 4"):
            writer.close()
    finally:
        _abandon(writer)


@pytest.mark.windows_parity
def test_metadata_schema_declares_the_digest_it_is_given(tmp_path: Path) -> None:
    """The schema helper is what makes a streaming artifact self-describing."""
    digest = "a" * 64
    schema = canonical_schema_with_metadata(TABLE_NAME, content_sha256=digest)
    metadata = schema.metadata or {}

    assert metadata[b"psd_content_sha256"] == digest.encode("ascii")
    assert metadata[b"psd_table"] == TABLE_NAME.encode("utf-8")
    assert metadata[b"psd_column_order"] == ",".join(table_spec(TABLE_NAME).columns).encode("utf-8")
