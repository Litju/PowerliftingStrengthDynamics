"""Fixture package contract tests.

Each fixture is asserted to be *clean*: it must satisfy every declarative and
cross-record invariant PSD enforces, while still exercising the edge cases the
design documents require. A fixture that drifts into an invalid state is a bug in
the fixture, so these tests fail loudly rather than tolerating issues.
"""

from __future__ import annotations

from collections.abc import Callable

import pyarrow as pa
import pytest
from pydantic import BaseModel

from psd.schema.registry import table_names
from psd.serialization.table import records_to_table
from psd.validation import validate_tables
from tests.fixtures import (
    adversarial_history_records,
    counts,
    messy_timeline_records,
    normal_history_records,
    synthetic_records,
)

ALL_TABLES = frozenset(table_names())

FIXTURES: dict[str, Callable[[], dict[str, list[BaseModel]]]] = {
    "normal": normal_history_records,
    "adversarial": adversarial_history_records,
    "synthetic": synthetic_records,
    "messy": messy_timeline_records,
}


@pytest.fixture(scope="module", params=sorted(FIXTURES), name="tables")
def _tables(request: pytest.FixtureRequest) -> dict[str, list[BaseModel]]:
    """Each canonical fixture, built once per module."""
    return FIXTURES[request.param]()


def _as_arrow(tables: dict[str, list[BaseModel]]) -> dict[str, pa.Table]:
    return {name: records_to_table(rows, table_name=name) for name, rows in tables.items()}


def test_fixture_uses_only_canonical_tables(tables: dict[str, list[BaseModel]]) -> None:
    assert set(tables) == ALL_TABLES


def test_fixture_reports_no_validation_issues(tables: dict[str, list[BaseModel]]) -> None:
    report = validate_tables(_as_arrow(tables))

    assert report.errors == (), [issue.message for issue in report.errors]
    assert report.warnings == (), [issue.message for issue in report.warnings]


def test_fixture_populates_at_least_one_row(tables: dict[str, list[BaseModel]]) -> None:
    total = sum(counts(tables).values())

    assert total > 0


def test_arrow_round_trip_preserves_records(tables: dict[str, list[BaseModel]]) -> None:
    for name, rows in tables.items():
        round_tripped = records_to_table(rows, table_name=name).to_pylist()

        assert len(round_tripped) == len(rows), name
