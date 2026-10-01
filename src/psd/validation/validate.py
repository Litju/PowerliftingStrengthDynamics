"""Dataset-level validation entry point.

Validation runs in two layers:

1. **Declarative column checks** (:mod:`psd.validation.declarative`) over Polars
   frames -- vectorized, whole-column, derived from the Arrow schema plus
   vocabulary and range overlays.
2. **Cross-record rules** (:mod:`psd.validation.rules`) over an index built once
   -- referential integrity, prescription/execution separation, real/synthetic
   separation, explicit missingness, provenance integrity, and physical
   consistency.

The result is a :class:`~psd.validation.issues.ValidationReport` listing every
finding, so one pass describes an artifact completely.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pyarrow as pa

from psd.schema.registry import table_names, table_spec
from psd.serialization.ordering import from_arrow_frame
from psd.serialization.table import table_to_rows
from psd.validation.declarative import column_issues as declarative_column_issues
from psd.validation.issues import Severity, ValidationIssue, ValidationReport
from psd.validation.rules import TableRows, all_issues

__all__ = ("validate_tables",)


def validate_tables(
    tables: Mapping[str, pa.Table],
    *,
    strict: bool = False,
    max_column_issues: int = 50,
) -> ValidationReport:
    """Validate a set of canonical Arrow tables.

    Every canonical table is checked; a table that is absent is reported by
    :meth:`ValidationReport.tables_checked` so a caller can see what was covered.
    Tables may be empty -- absence of records is not a defect -- but an empty
    table still carries the canonical schema, which is verified before any row is
    read.

    Args:
        tables: Canonical table name to Arrow table.
        strict: Treat warnings as errors in the returned report's ``ok``.
        max_column_issues: Cap on declarative column issues per table.

    Returns:
        The combined report.

    Raises:
        ValueError: A table's Arrow schema does not match the canonical schema.
    """
    rows: dict[str, list[dict[str, Any]]] = {}
    row_counts: dict[str, int] = {}
    issues: list[ValidationIssue] = []
    checked: list[str] = []

    for name in table_names():
        table = tables.get(name)
        if table is None:
            continue
        checked.append(name)
        table_rows = table_to_rows(table, table_name=name)
        rows[name] = table_rows
        row_counts[name] = len(table_rows)
        issues.extend(_column_issues(table, name, limit=max_column_issues))

    table_rows_map: TableRows = rows
    issues.extend(all_issues(table_rows_map))

    report = ValidationReport(
        issues=tuple(issues),
        tables_checked=tuple(checked),
        row_counts=row_counts,
    )
    if strict and report.warnings:
        promoted = tuple(
            ValidationIssue(
                code=issue.code,
                severity=Severity.ERROR,
                message=f"{issue.message} (strict mode)",
                table=issue.table,
                record_id=issue.record_id,
            )
            if issue.severity is Severity.WARNING
            else issue
            for issue in report.issues
        )
        return ValidationReport(
            issues=promoted,
            tables_checked=report.tables_checked,
            row_counts=report.row_counts,
        )
    return report


def _column_issues(table: pa.Table, table_name: str, *, limit: int) -> list[ValidationIssue]:
    """Run the declarative layer over one table."""
    spec = table_spec(table_name)
    frame = from_arrow_frame(table.select(spec.columns))
    if frame.is_empty():
        return []
    return declarative_column_issues(frame, table_name=table_name, limit=limit)
