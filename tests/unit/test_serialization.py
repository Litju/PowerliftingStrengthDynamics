"""Unit and property tests for deterministic serialization.

Proves three things the locked stack requires:

* an artifact is sorted by an explicit total ordering before it is hashed;
* the content digest depends only on logical values, not on row order;
* the Parquet write is reproducible byte-for-byte on one platform.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from psd.schema.models import PerformedSetRecord
from psd.schema.registry import table_spec
from psd.serialization import (
    CanonicalEncodingError,
    OrderingError,
    TableSchemaMismatchError,
    canonical_bytes,
    canonical_order,
    content_digest,
    empty_table,
    read_parquet,
    records_to_table,
    sha256_file,
    table_to_records,
    write_parquet,
)
from psd.serialization.canonical import CONTENT_ENCODING
from psd.serialization.ordering import canonical_order_polars
from psd.units import LB_TO_KG

INGESTED_AT = datetime(2026, 1, 5, 9, 30, tzinfo=UTC)
SOURCE_ID = "hevy_export_2024"

SET_TEMPLATES: list[dict[str, Any]] = [
    {
        "performed_set_id": f"pset_{index:032x}",
        "performed_exercise_id": f"pex_{index:032x}",
        "ordinal": index,
        "planned_set_id": None,
        "set_status": "completed",
        "load_raw": 100.0 + index,
        "load_unit": "kg",
        "load_kg": 100.0 + index,
        "reps_performed": 5,
        "reps_failed": None,
        "is_failure": None,
        "rpe": 7.0,
        "rir": None,
        "tempo_actual": None,
        "duration_seconds": None,
        "rest_actual_seconds": None,
        "is_warmup": None,
        "rep_level_data_available": False,
        "incomplete_reason": None,
        "created_at": None,
        "performed_at": datetime(2026, 3, 1 + index, 18, 0, tzinfo=UTC),
        "modified_at": None,
        "source_id": SOURCE_ID,
        "source_record_key": f"row-{index}",
        "source_record_hash": None,
        "ingested_at": INGESTED_AT,
        "quality_flags": ["verified"] if index % 2 == 0 else [],
        "missingness_reason": None,
    }
    for index in range(6)
]


def _table(rows: list[dict[str, Any]] | None = None) -> pa.Table:
    """Build a valid performed_set table through the record contracts."""
    selected = SET_TEMPLATES if rows is None else rows
    records = [PerformedSetRecord.model_validate(row) for row in selected]
    return records_to_table(records, table_name="performed_set")


def _ordered(table: pa.Table) -> pa.Table:
    spec = table_spec("performed_set")
    return canonical_order(table, spec.order_by, table_name="performed_set")


def _digest(table: pa.Table) -> str:
    return content_digest(_ordered(table), table_name="performed_set")


def test_empty_table_carries_canonical_schema() -> None:
    table = empty_table("athlete")
    assert table.num_rows == 0
    assert table.schema.equals(table_spec("athlete").arrow_schema())


def test_canonical_order_preserves_the_physical_schema() -> None:
    """Polars widens strings; canonical ordering must not."""
    ordered = _ordered(_table())
    assert ordered.schema.equals(table_spec("performed_set").arrow_schema())


def test_canonical_order_is_independent_of_input_order() -> None:
    reference = _digest(_table())
    for rotation in range(len(SET_TEMPLATES)):
        shuffled = SET_TEMPLATES[rotation:] + SET_TEMPLATES[:rotation]
        assert _digest(_table(list(shuffled))) == reference, f"rotation {rotation} changed it"


def test_canonical_order_puts_nulls_last() -> None:
    """Null placement is fixed, so "not recorded" rows sort predictably."""
    rows = [
        {
            **SET_TEMPLATES[index],
            "planned_set_id": None if index % 2 == 0 else f"plt_{index:032x}",
        }
        for index in range(6)
    ]
    table = _table(rows)
    ordered = canonical_order(
        table,
        ("planned_set_id", "performed_set_id"),
        table_name="performed_set",
    )
    planned_ids = [row["planned_set_id"] for row in ordered.to_pylist()]
    nulls = [index for index, value in enumerate(planned_ids) if value is None]
    assert nulls, "fixture must contain nulls for this ordering rule to be observable"
    assert nulls == list(range(len(nulls), len(planned_ids)))


def test_ordering_must_be_total() -> None:
    spec = table_spec("performed_set")
    partial = tuple(column for column in spec.order_by if column != "performed_set_id")
    with pytest.raises(OrderingError, match="not be total"):
        canonical_order(_table(), partial, table_name="performed_set")


def test_ordering_rejects_unknown_column() -> None:
    with pytest.raises(OrderingError, match="unknown column"):
        canonical_order_polars(pl.DataFrame({"a": [1]}), ["nope"])


def test_content_digest_is_table_name_sensitive() -> None:
    ordered = _ordered(_table())
    assert content_digest(ordered, table_name="performed_set") != content_digest(
        ordered, table_name="performed_exercise"
    )


def test_canonical_encoding_header_is_stable() -> None:
    encoded = canonical_bytes(_ordered(_table()), table_name="performed_set")
    assert encoded.startswith(CONTENT_ENCODING.encode())
    assert b"table:performed_set\n" in encoded[:200]


def test_canonical_encoding_distinguishes_string_and_int() -> None:
    """Type tags prevent value collisions."""
    as_reps = canonical_bytes(
        _ordered(_table([{**SET_TEMPLATES[0], "reps_performed": 1}])),
        table_name="performed_set",
    )
    as_ordinal = canonical_bytes(
        _ordered(_table([{**SET_TEMPLATES[0], "ordinal": 1}])),
        table_name="performed_set",
    )
    assert as_reps != as_ordinal


def test_canonical_encoding_rejects_non_finite_floats() -> None:
    table = pa.Table.from_arrays([pa.array([float("nan")], pa.float64())], names=["value"])
    with pytest.raises(CanonicalEncodingError, match="Non-finite float"):
        canonical_bytes(table, table_name="probe")


def test_parquet_round_trip_is_lossless(tmp_path: Path) -> None:
    original = _ordered(_table())
    result = write_parquet(original, tmp_path / "performed_set.parquet", table_name="performed_set")
    assert result.content_sha256 == content_digest(original, table_name="performed_set")
    assert result.row_count == len(SET_TEMPLATES)
    assert result.sha256 == sha256_file(result.path)
    read_back = read_parquet(result.path, table_name="performed_set")
    assert read_back.to_pylist() == original.to_pylist()


def test_parquet_write_is_byte_reproducible(tmp_path: Path) -> None:
    original = _ordered(_table())
    first = write_parquet(original, tmp_path / "a.parquet", table_name="performed_set")
    second = write_parquet(original, tmp_path / "b.parquet", table_name="performed_set")
    assert first.sha256 == second.sha256
    assert first.content_sha256 == second.content_sha256


def test_parquet_artifact_is_self_describing(tmp_path: Path) -> None:
    result = write_parquet(_ordered(_table()), tmp_path / "ps.parquet", table_name="performed_set")
    metadata = pq.ParquetFile(result.path).schema_arrow.metadata or {}
    assert metadata[b"psd_table"] == b"performed_set"
    assert metadata[b"psd_schema_version"] == b"psd-canonical/0.2.0"
    assert metadata[b"psd_content_sha256"].decode() == result.content_sha256


def test_read_parquet_rejects_wrong_table(tmp_path: Path) -> None:
    result = write_parquet(_ordered(_table()), tmp_path / "ps.parquet", table_name="performed_set")
    with pytest.raises(ValueError, match="does not match the canonical schema"):
        read_parquet(result.path, table_name="performed_session")


def test_table_to_records_revalidates(tmp_path: Path) -> None:
    original = _ordered(_table())
    path = tmp_path / "ps.parquet"
    write_parquet(original, path, table_name="performed_set")
    records = table_to_records(
        read_parquet(path, table_name="performed_set"), table_name="performed_set"
    )
    assert len(records) == len(SET_TEMPLATES)
    assert all(isinstance(record, PerformedSetRecord) for record in records)


def test_table_schema_mismatch_is_detected() -> None:
    foreign = pa.Table.from_arrays([pa.array([1], pa.int64())], names=["unexpected"])
    with pytest.raises(TableSchemaMismatchError):
        table_to_records(foreign, table_name="performed_set")


def test_nulls_survive_the_round_trip(tmp_path: Path) -> None:
    """Absence must never come back as zero."""
    path = tmp_path / "ps.parquet"
    write_parquet(_ordered(_table()), path, table_name="performed_set")
    rows = read_parquet(path, table_name="performed_set").to_pylist()
    assert all(row["rir"] is None for row in rows)
    assert all(row["reps_failed"] is None for row in rows)
    assert all(row["planned_set_id"] is None for row in rows)


def test_raw_and_normalized_values_both_survive(tmp_path: Path) -> None:
    path = tmp_path / "ps.parquet"
    write_parquet(_ordered(_table()), path, table_name="performed_set")
    row = read_parquet(path, table_name="performed_set").to_pylist()[0]
    assert row["load_raw"] is not None
    assert row["load_kg"] == row["load_raw"]


def test_pound_source_values_survive_normalization(tmp_path: Path) -> None:
    rows = [
        {
            **SET_TEMPLATES[0],
            "performed_set_id": f"pset_{index:032x}",
            "performed_exercise_id": f"pex_{index:032x}",
            "ordinal": index,
            "load_raw": 225.0,
            "load_unit": "lb",
            "load_kg": 225.0 * LB_TO_KG,
        }
        for index in range(3)
    ]
    path = tmp_path / "ps.parquet"
    write_parquet(_ordered(_table(rows)), path, table_name="performed_set")
    read_back = read_parquet(path, table_name="performed_set").to_pylist()
    assert {row["load_unit"] for row in read_back} == {"lb"}
    assert read_back[0]["load_kg"] == 225.0 * LB_TO_KG


@settings(max_examples=20)
@given(
    value=st.floats(min_value=20.0, max_value=400.0, allow_nan=False, allow_infinity=False),
    unit=st.sampled_from(["kg", "lb"]),
)
def test_content_digest_survives_row_reordering(value: float, unit: str) -> None:
    normalized = value if unit == "kg" else value * LB_TO_KG
    rows = [
        {
            **SET_TEMPLATES[0],
            "performed_set_id": f"pset_{index:032x}",
            "performed_exercise_id": f"pex_{index:032x}",
            "ordinal": index,
            "load_raw": value,
            "load_unit": unit,
            "load_kg": normalized,
        }
        for index in range(3)
    ]
    forward = _digest(_table(list(rows)))
    backward = _digest(_table(list(reversed(rows))))
    assert forward == backward


@given(index=st.integers(min_value=0, max_value=len(SET_TEMPLATES) - 1))
def test_digest_is_stable_for_a_single_row(index: int) -> None:
    row = [SET_TEMPLATES[index]]
    assert _digest(_table(list(row))) == _digest(_table(list(row)))
