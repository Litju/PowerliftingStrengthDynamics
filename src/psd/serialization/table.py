"""Conversion between canonical record contracts and Arrow tables."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pyarrow as pa
from pydantic import BaseModel

from psd.schema.registry import TableSpec, table_spec

__all__ = (
    "SCHEMA_TAG",
    "TableSchemaMismatchError",
    "empty_table",
    "records_to_table",
    "table_to_records",
    "table_to_rows",
)

SCHEMA_TAG = "psd-canonical"


class TableSchemaMismatchError(ValueError):
    """Raised when an Arrow table does not match its canonical schema."""


def empty_table(table_name: str) -> pa.Table:
    """Return an empty table carrying the canonical schema for *table_name*.

    A canonical dataset always contains every table, even when a table is empty
    for a given athlete. A consistent table set keeps schema discovery and digest
    coverage uniform.
    """
    return pa.Table.from_batches([], schema=table_spec(table_name).arrow_schema())


def records_to_table(records: Sequence[BaseModel], *, table_name: str) -> pa.Table:
    """Build an Arrow table from validated records.

    Args:
        records: Records, each an instance of the table's contract.
        table_name: Canonical table name.

    Returns:
        The Arrow table, in the order supplied. Sort before persisting.

    Raises:
        TableSchemaMismatchError: A record belongs to a different table's contract.
    """
    spec = table_spec(table_name)
    rows: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, spec.model):
            msg = (
                f"Expected {spec.model.__name__} records for table {table_name!r}; got "
                f"{type(record).__name__}."
            )
            raise TableSchemaMismatchError(msg)
        rows.append(record.model_dump())
    if not rows:
        return empty_table(table_name)
    return pa.Table.from_pylist(rows, schema=spec.arrow_schema())


def table_to_rows(table: pa.Table, *, table_name: str) -> list[dict[str, Any]]:
    """Return the table's rows as plain dictionaries.

    The schema is checked before any row is read, so a truncated or re-typed
    artifact fails with a precise message instead of an error about one arbitrary
    row.

    Args:
        table: Arrow table read from disk or received from a transform.
        table_name: Canonical table name.

    Returns:
        One dictionary per row.

    Raises:
        TableSchemaMismatchError: The Arrow schema differs from the canonical schema.
    """
    _require_schema(table_spec(table_name), table)
    rows: list[dict[str, Any]] = table.to_pylist()
    return rows


def table_to_records(table: pa.Table, *, table_name: str) -> list[BaseModel]:
    """Rebuild and re-validate records from an Arrow table.

    This is the semantic round-trip entry point: an artifact that survives this
    call has every column re-checked against the contract that produced it.

    Args:
        table: Arrow table read from disk.
        table_name: Canonical table name.

    Returns:
        Records validated against the table's contract. Callers narrow the type
        with ``isinstance`` against the known contract.

    Raises:
        TableSchemaMismatchError: The Arrow schema differs from the canonical schema.
    """
    model = table_spec(table_name).model
    return [model(**row) for row in table_to_rows(table, table_name=table_name)]


def _require_schema(spec: TableSpec, table: pa.Table) -> None:
    expected = spec.arrow_schema()
    actual = table.schema
    if actual.equals(expected, check_metadata=False):
        return
    msg = (
        f"Table {spec.name!r} does not match {SCHEMA_TAG} schema. "
        f"expected=[{', '.join(expected.names)}] actual=[{', '.join(actual.names)}]"
    )
    raise TableSchemaMismatchError(msg)
