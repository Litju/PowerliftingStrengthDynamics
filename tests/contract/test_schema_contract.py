"""Structural contract tests for the canonical schema registry.

These tests are the executable form of the PSD design invariants: they assert
that the persisted schema cannot accidentally blur prescription into execution,
cannot encode absence as zero, and cannot be written without an explicit
canonical ordering.
"""

from __future__ import annotations

import re

import pyarrow as pa
import pytest

from psd.schema import SCHEMA_VERSION, TABLE_SPECS
from psd.schema.arrow import schema_for_model, undeclared_fields
from psd.schema.registry import (
    PROVENANCE_COLUMNS,
    TEMPORAL_COLUMNS,
    TableSpec,
    arrow_schema_for,
    column_order,
    table_categories,
    table_names,
    table_spec,
    temporal_columns,
)

ALL_SPECS = TABLE_SPECS


def test_registry_covers_every_required_entity() -> None:
    required = {
        "source",
        "provenance",
        "athlete",
        "athlete_source_link",
        "body_measurement",
        "equipment_state",
        "exercise_definition",
        "exercise_alias",
        "program",
        "program_version",
        "program_modification",
        "planned_session",
        "planned_exercise",
        "planned_set",
        "performed_session",
        "performed_exercise",
        "performed_set",
        "performed_rep",
        "observation",
        "performance_test",
        "velocity_observation",
        "competition",
        "competition_attempt",
        "competition_reported_result",
    }
    assert required <= set(table_names())


def test_table_names_are_unique_and_sorted_within_categories() -> None:
    names = table_names()
    assert len(names) == len(set(names))
    for category, members in table_categories().items():
        assert members, category
        assert len(members) == len(set(members))


@pytest.mark.parametrize("spec", ALL_SPECS, ids=lambda spec: spec.name)
def test_columns_are_a_permitted_subset_of_model_fields(spec: TableSpec) -> None:
    declared = column_order(spec.name)
    assert len(declared) == len(set(declared)), "duplicate columns"
    assert set(declared) <= set(spec.model.model_fields)
    withheld = undeclared_fields(spec.model, declared)
    assert set(withheld) <= set(TEMPORAL_COLUMNS), f"{spec.name} drops non-temporal fields"
    assert set(withheld) <= set(TEMPORAL_COLUMNS) - set(temporal_columns(spec.model))


@pytest.mark.parametrize("spec", ALL_SPECS, ids=lambda spec: spec.name)
def test_ordering_is_total_and_ends_with_primary_key(spec: TableSpec) -> None:
    assert spec.order_by[-len(spec.primary_key) :] == spec.primary_key
    assert set(spec.order_by) <= set(spec.columns)


@pytest.mark.parametrize("spec", ALL_SPECS, ids=lambda spec: spec.name)
def test_schema_derives_cleanly_from_model(spec: TableSpec) -> None:
    schema = arrow_schema_for(spec.name)
    assert isinstance(schema, pa.Schema)
    assert tuple(schema.names) == spec.columns
    assert schema.equals(schema_for_model(spec.model, spec.columns))


@pytest.mark.parametrize("spec", ALL_SPECS, ids=lambda spec: spec.name)
def test_every_column_is_nullable(spec: TableSpec) -> None:
    """A column that cannot be null cannot express absence."""
    schema = arrow_schema_for(spec.name)
    non_nullable = pa.schema(
        [
            pa.field(name, data_type, nullable=False)
            for name, data_type in zip(schema.names, schema.types, strict=True)
        ]
    )
    assert not schema.equals(non_nullable), f"{spec.name} declares a non-nullable column"


@pytest.mark.parametrize("spec", ALL_SPECS, ids=lambda spec: spec.name)
def test_timestamp_columns_are_timezone_aware_utc(spec: TableSpec) -> None:
    schema = arrow_schema_for(spec.name)
    for data_type in schema.types:
        if pa.types.is_timestamp(data_type):
            assert data_type.unit == "us"
            assert data_type.tz == "UTC"


EVENT_TABLES = {
    spec.name
    for spec in ALL_SPECS
    if spec.category in {"context", "programming", "execution", "observation", "competition"}
}


@pytest.mark.parametrize("name", sorted(EVENT_TABLES))
def test_event_tables_carry_full_provenance(name: str) -> None:
    columns = set(column_order(name))
    assert set(PROVENANCE_COLUMNS) <= columns, f"{name} is missing row-level provenance"


@pytest.mark.parametrize(
    "name",
    sorted(spec.name for spec in ALL_SPECS if spec.category != "provenance"),
)
def test_event_tables_distinguish_temporal_provenance(name: str) -> None:
    """Each record persists the temporal fields its semantics allow, and only those."""
    spec = table_spec(name)
    declared = set(spec.columns)
    permitted = set(temporal_columns(spec.model)) & set(spec.model.model_fields)
    assert permitted, f"{name} declares no temporal provenance at all"
    assert permitted <= declared
    withheld = set(TEMPORAL_COLUMNS) - permitted
    assert not (withheld & declared), (
        f"{name} persists forbidden temporal columns: {withheld & declared}"
    )


PRESCRIPTION_TABLES = sorted(spec.name for spec in ALL_SPECS if spec.category == "programming")
EXECUTION_TABLES = sorted(spec.name for spec in ALL_SPECS if spec.category == "execution")

FORBIDDEN_ON_PRESCRIPTION = (
    "performed_",
    "actual_",
    "reps_performed",
    "reps_failed",
    "set_status",
    "rep_status",
    "is_failure",
    "tempo_actual",
    "rpe",
    "rir",
    "ended_at",
    "started_at",
)


@pytest.mark.parametrize("name", PRESCRIPTION_TABLES)
def test_prescription_tables_contain_no_execution_columns(name: str) -> None:
    offenders = [
        column
        for column in column_order(name)
        if column.startswith(FORBIDDEN_ON_PRESCRIPTION) or column in {"rpe", "rir"}
    ]
    assert not offenders, f"{name} leaks execution state: {offenders}"


FORBIDDEN_ON_EXECUTION = (
    "target_",
    "rest_target_seconds",
    "prescription_basis",
)


@pytest.mark.parametrize("name", EXECUTION_TABLES)
def test_execution_tables_contain_no_prescription_targets(name: str) -> None:
    offenders = [
        column for column in column_order(name) if column.startswith(FORBIDDEN_ON_EXECUTION)
    ]
    assert not offenders, f"{name} leaks prescription into execution: {offenders}"


PLAN_LINK_COLUMNS = {
    "planned_session_id",
    "planned_exercise_id",
    "planned_set_id",
}


@pytest.mark.parametrize("name", EXECUTION_TABLES)
def test_execution_tables_only_link_to_plans_explicitly(name: str) -> None:
    """A plan may only be referenced by an explicit, nullable link column."""
    plan_like = {column for column in column_order(name) if "planned" in column}
    assert plan_like <= PLAN_LINK_COLUMNS, f"{name} has implicit plan columns: {plan_like}"


def test_prescription_is_never_derived_from_execution() -> None:
    """No table outside ``programming`` carries prescription targets."""
    for spec in ALL_SPECS:
        if spec.category == "programming":
            continue
        offenders = [column for column in spec.columns if column.startswith("target_")]
        assert not offenders, f"{spec.name} stores prescription targets: {offenders}"


def test_exercise_semantics_carry_no_transfer_coefficient() -> None:
    columns = column_order("exercise_definition")
    forbidden = [column for column in columns if "transfer" in column or "coefficient" in column]
    assert not forbidden


def test_no_daily_or_weekly_aggregate_table_exists() -> None:
    """Canonical storage is event-time; daily/weekly aggregates are derived only."""
    forbidden = re.compile(r"(^|_)(daily|weekly|week|day)_(total|volume|summary|aggregate)")
    offenders = [
        spec.name for spec in ALL_SPECS if any(forbidden.search(column) for column in spec.columns)
    ]
    assert not offenders


def test_competition_attempts_and_reported_results_are_distinct() -> None:
    assert "competition_attempt" in table_names()
    assert "competition_reported_result" in table_names()
    reported = column_order("competition_reported_result")
    assert "is_derived" in reported
    assert "derivation_note" in reported


def test_unknown_table_raises() -> None:
    with pytest.raises(KeyError, match="Unknown canonical table"):
        table_spec("not_a_table")


def test_schema_version_is_reported() -> None:
    assert SCHEMA_VERSION.tag == "psd-canonical/0.3.0"
