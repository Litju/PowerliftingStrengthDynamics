"""Unit tests for the validation layer.

Each test injects exactly one defect and asserts the matching rule fires, so a
rule that silently stops working is caught rather than masked by another rule.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any

import polars as pl
import pyarrow as pa

from psd.schema.registry import column_order, table_names, table_spec
from psd.serialization.ordering import from_arrow_frame
from psd.serialization.table import empty_table, records_to_table
from psd.validation import Severity, validate_tables
from psd.validation.declarative import column_issues, vocabulary_issues

ATHLETE_ID = "ath_11111111111111111111111111111111"
SOURCE_ID = "src_22222222222222222222222222222222"
INGESTED_AT = datetime(2026, 1, 5, 9, 30, tzinfo=UTC)
SESSION_AT = datetime(2026, 3, 1, 18, 0, tzinfo=UTC)


def _source(**overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "source_id": SOURCE_ID,
        "nature": "real",
        "regime": "psd_real",
        "origin_system": "hevy",
        "display_name": "Athlete export",
        "dataset_version": "2026-01",
        "snapshot_date": None,
        "snapshot_sha256": None,
        "license_id": "proprietary",
        "license_url": None,
        "consent_basis": "user_consent",
        "redistribution": "not_allowed",
        "ingested_at": INGESTED_AT,
        "notes": None,
    }
    row.update(overrides)
    return row


def _athlete(**overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "athlete_id": ATHLETE_ID,
        "pseudonym": "athlete-001",
        "identity_status": "single_source_verified",
        "ambiguity_group_id": None,
        "is_synthetic": False,
        "synthetic_regime": None,
        "sex_category_raw": "Male",
        "sex_category": "male",
        "birth_year": 1995,
        "country_code": "ES",
        "created_at": INGESTED_AT,
        "source_id": SOURCE_ID,
        "source_record_key": "athlete-001",
        "source_record_hash": None,
        "ingested_at": INGESTED_AT,
        "quality_flags": [],
        "missingness_reason": None,
    }
    row.update(overrides)
    return row


def _set_row(**overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "performed_set_id": "pset_33333333333333333333333333333333",
        "performed_exercise_id": "pex_44444444444444444444444444444444",
        "ordinal": 1,
        "planned_set_id": None,
        "set_status": "completed",
        "load_raw": 100.0,
        "load_unit": "kg",
        "load_kg": 100.0,
        "reps_performed": 5,
        "reps_failed": None,
        "is_failure": None,
        "rpe": 7.5,
        "rir": None,
        "tempo_actual": None,
        "duration_seconds": None,
        "rest_actual_seconds": None,
        "is_warmup": None,
        "rep_level_data_available": False,
        "incomplete_reason": None,
        "created_at": None,
        "performed_at": SESSION_AT,
        "modified_at": None,
        "source_id": SOURCE_ID,
        "source_record_key": "row-1",
        "source_record_hash": None,
        "ingested_at": INGESTED_AT,
        "quality_flags": [],
        "missingness_reason": None,
    }
    row.update(overrides)
    return row


def _performed_session(**overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "performed_session_id": "pses_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "athlete_id": ATHLETE_ID,
        "planned_session_id": None,
        "started_at": SESSION_AT,
        "session_order_index": 0,
        "session_type": "training",
        "ended_at": SESSION_AT + timedelta(hours=1),
        "duration_seconds": 3600.0,
        "is_completed": True,
        "created_at": None,
        "scheduled_at": None,
        "performed_at": SESSION_AT,
        "observed_at": None,
        "modified_at": None,
        "source_id": SOURCE_ID,
        "source_record_key": "session-1",
        "source_record_hash": None,
        "ingested_at": INGESTED_AT,
        "quality_flags": [],
        "missingness_reason": None,
    }
    row.update(overrides)
    return row


def _performed_exercise(**overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "performed_exercise_id": "pex_44444444444444444444444444444444",
        "performed_session_id": "pses_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "exercise_id": "exd_bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        "ordinal": 1,
        "planned_exercise_id": None,
        "is_substitution": None,
        "substituted_from_exercise_id": None,
        "created_at": None,
        "scheduled_at": None,
        "performed_at": SESSION_AT,
        "observed_at": None,
        "modified_at": None,
        "source_id": SOURCE_ID,
        "source_record_key": "exercise-1",
        "source_record_hash": None,
        "ingested_at": INGESTED_AT,
        "quality_flags": [],
        "missingness_reason": None,
    }
    row.update(overrides)
    return row


def _exercise(**overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "exercise_id": "exd_bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        "canonical_name": "Competition Bench Press",
        "parent_lift": "bench",
        "specificity_level": "competition_lift",
        "implement": "barbell",
        "laterality": "bilateral",
        "stance": None,
        "grip": None,
        "range_of_motion": "competition",
        "pause": True,
        "tempo": None,
        "equipment_note": None,
        "definition_note": None,
        "created_at": None,
        "source_id": SOURCE_ID,
        "source_record_key": "bench-press",
        "source_record_hash": None,
        "ingested_at": INGESTED_AT,
        "quality_flags": [],
        "missingness_reason": None,
    }
    row.update(overrides)
    return row


def _tables(**overrides: Any) -> dict[str, pa.Table]:
    """Build a small but complete and valid dataset, with per-table overrides."""
    rows: dict[str, list[dict[str, Any]]] = {
        "source": [_source()],
        "athlete": [_athlete()],
        "exercise_definition": [_exercise()],
        "performed_session": [_performed_session()],
        "performed_exercise": [_performed_exercise()],
        "performed_set": [_set_row()],
    }
    for table, table_rows in overrides.items():
        rows[table] = _apply(table, table_rows, rows.get(table, []))
    tables: dict[str, pa.Table] = {}
    for table, table_rows in rows.items():
        tables[table] = _table_for(table, table_rows)
    return tables


def _apply(table: str, override: object, defaults: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return the rows for a table, applying a dict override or a replacement list."""
    if isinstance(override, dict):
        builders: dict[str, Callable[..., dict[str, Any]]] = {
            "source": _source,
            "athlete": _athlete,
            "exercise_definition": _exercise,
            "performed_session": _performed_session,
            "performed_exercise": _performed_exercise,
            "performed_set": _set_row,
        }
        builder = builders.get(table)
        if builder is None:
            return [{**row, **override} for row in defaults]  # type: ignore[dict-item]
        return [builder(**override)]
    return list(override)  # type: ignore[call-overload]


def _table_for(table: str, rows: list[dict[str, Any]]) -> pa.Table:
    """Build an Arrow table through the table's contract."""
    spec = table_spec(table)
    model = spec.model
    records = [model.model_validate(row) for row in rows]
    return records_to_table(records, table_name=table)


def _codes(report: Any) -> set[str]:
    return {issue.code for issue in report.issues}


def test_valid_dataset_has_no_errors() -> None:
    report = validate_tables(_tables())
    assert report.ok, report.summary()
    assert "performed_set" in report.tables_checked
    assert report.row_counts is not None
    assert report.row_counts["performed_set"] == 1


def test_empty_tables_are_valid() -> None:
    tables = {"source": _table_for("source", [_source()])}
    report = validate_tables(tables)
    assert report.ok


def test_dangling_athlete_reference_is_an_error() -> None:
    tables = _tables(body_measurement=[_body_measurement(athlete_id="ath_deadbeef")])
    report = validate_tables(tables)
    assert not report.ok
    assert "dangling_athlete" in _codes(report)


def _body_measurement(**overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "body_measurement_id": "bms_55555555555555555555555555555555",
        "athlete_id": ATHLETE_ID,
        "measured_at": datetime(2026, 3, 1, 7, 0, tzinfo=UTC),
        "measurement_type": "body_mass",
        "event_time_precision": "minute",
        "measurement_context": "morning_fasted",
        "method": "bathroom_scale",
        "raw_value": 93.2,
        "raw_unit": "kg",
        "value_normalized": 93.2,
        "unit_normalized": "kg",
        "created_at": None,
        "scheduled_at": None,
        "observed_at": None,
        "modified_at": None,
        "source_id": SOURCE_ID,
        "source_record_key": "weight-1",
        "source_record_hash": None,
        "ingested_at": INGESTED_AT,
        "quality_flags": [],
        "missingness_reason": None,
    }
    row.update(overrides)
    return row


def test_unregistered_source_is_an_error() -> None:
    tables = _tables()
    tables["source"] = empty_table("source")
    report = validate_tables(tables)
    assert not report.ok
    assert "unregistered_source" in _codes(report)


def test_synthetic_athlete_from_real_source_is_an_error() -> None:
    synthetic_athlete = _athlete(
        athlete_id="ath_66666666666666666666666666666666",
        is_synthetic=True,
        synthetic_regime="psd_sim",
        sex_category=None,
        sex_category_raw=None,
        birth_year=None,
    )
    body = _body_measurement(athlete_id=synthetic_athlete["athlete_id"])
    tables = _tables(athlete=[_athlete(), synthetic_athlete], body_measurement=[body])
    report = validate_tables(tables)
    assert not report.ok
    assert "synthetic_athlete_from_real_source" in _codes(report)


def test_real_athlete_from_synthetic_source_is_an_error() -> None:
    tables = _tables(source=[_source(nature="synthetic", regime="psd_sim")])
    report = validate_tables(tables)
    assert not report.ok
    assert "real_athlete_from_synthetic_source" in _codes(report)


def test_edit_flag_without_timestamp_is_an_error() -> None:
    tables = _tables(
        performed_set={"quality_flags": ["edited_after_the_fact"], "modified_at": None}
    )
    report = validate_tables(tables)
    assert not report.ok
    assert "edit_without_timestamp" in _codes(report)


def test_edit_flag_with_timestamp_passes() -> None:
    tables = _tables(
        performed_set={
            "quality_flags": ["edited_after_the_fact"],
            "modified_at": SESSION_AT + timedelta(days=1),
        }
    )
    report = validate_tables(tables)
    assert report.ok, [issue.format() for issue in report.issues]


def test_rpe_rir_inconsistency_is_a_warning() -> None:
    """RPE 5 with RIR 2 implies a total of 7, not the usual 10."""
    tables = _tables(performed_set={"rpe": 5.0, "rir": 2.0})
    report = validate_tables(tables)
    assert report.ok
    assert "rpe_rir_inconsistent" in _codes(report)


def test_consistent_rpe_rir_produces_no_warning() -> None:
    tables = _tables(performed_set={"rpe": 8.0, "rir": 2.0})
    report = validate_tables(tables)
    assert "rpe_rir_inconsistent" not in _codes(report)


def test_strict_mode_promotes_warnings() -> None:
    tables = _tables(performed_set={"rpe": 5.0, "rir": 2.0})
    report = validate_tables(tables, strict=True)
    assert not report.ok


def test_missingness_undeclared_is_a_warning() -> None:
    tables = _tables(performed_set={"rir": None, "missingness_reason": None})
    report = validate_tables(tables)
    assert "missingness_undeclared" in _codes(report)
    assert report.ok


def test_declared_missingness_silences_the_warning() -> None:
    tables = _tables(performed_set={"rir": None, "missingness_reason": "not_recorded_in_source"})
    report = validate_tables(tables)
    assert "missingness_undeclared" not in _codes(report)


def test_duplicate_primary_key_is_an_error() -> None:
    duplicate = _set_row(source_record_key="row-2")
    tables = _tables(performed_set=[_set_row(), duplicate])  # type: ignore[arg-type]
    report = validate_tables(tables)
    assert not report.ok
    assert "duplicate_primary_key" in _codes(report)


def test_overlapping_equipment_intervals_are_an_error() -> None:
    base: dict[str, Any] = {
        "equipment_state_id": "eqs_77777777777777777777777777777777",
        "athlete_id": ATHLETE_ID,
        "equipment_item": "shoes",
        "effective_from": datetime(2026, 1, 1, tzinfo=UTC),
        "effective_to": None,
        "identifier": "model-a",
        "version_label": None,
        "created_at": None,
        "source_id": SOURCE_ID,
        "source_record_key": None,
        "source_record_hash": None,
        "ingested_at": INGESTED_AT,
        "quality_flags": [],
        "missingness_reason": None,
    }
    second: dict[str, Any] = {
        **base,
        "equipment_state_id": "eqs_88888888888888888888888888888888",
        "effective_from": datetime(2026, 6, 1, tzinfo=UTC),
    }
    tables = _tables(equipment_state=[base, second])
    report = validate_tables(tables)
    assert not report.ok
    assert "overlapping_interval" in _codes(report)


def test_reported_total_mismatch_is_a_warning() -> None:
    results = [
        _reported_result(kind="squat_best", value=200.0, identifier="c1"),
        _reported_result(kind="bench_best", value=140.0, identifier="c2"),
        _reported_result(kind="deadlift_best", value=240.0, identifier="c3"),
        _reported_result(kind="total", value=999.0, identifier="c4"),
    ]
    tables = _tables(
        competition=[_competition()],
        competition_reported_result=results,
    )
    report = validate_tables(tables)
    assert "reported_total_mismatch" in _codes(report)


def _competition(**overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "competition_id": "cmp_99999999999999999999999999999999",
        "athlete_id": ATHLETE_ID,
        "competition_date": datetime(2026, 4, 11, tzinfo=UTC),
        "event_time_precision": "date_only",
        "name": "Regional Open",
        "federation": "test-federation",
        "sanctioning_body": None,
        "location": "Somewhere",
        "equipment_class_raw": "raw",
        "equipment_class": "raw",
        "weight_class_raw": "-83",
        "bodyweight_raw": 82.4,
        "bodyweight_unit": "kg",
        "bodyweight_kg": 82.4,
        "participation_status": "competed",
        "is_championship": False,
        "created_at": None,
        "scheduled_at": None,
        "performed_at": None,
        "observed_at": None,
        "modified_at": None,
        "source_id": SOURCE_ID,
        "source_record_key": "meet-1",
        "source_record_hash": None,
        "ingested_at": INGESTED_AT,
        "quality_flags": [],
        "missingness_reason": None,
    }
    row.update(overrides)
    return row


def _reported_result(*, kind: str, value: float, identifier: str) -> dict[str, Any]:
    return {
        "competition_reported_result_id": f"cres_{identifier}",
        "competition_id": _competition()["competition_id"],
        "athlete_id": ATHLETE_ID,
        "result_kind": kind,
        "value": value,
        "unit": "kg",
        "is_derived": False,
        "derivation_note": None,
        "created_at": None,
        "scheduled_at": None,
        "performed_at": None,
        "observed_at": None,
        "modified_at": None,
        "source_id": SOURCE_ID,
        "source_record_key": identifier,
        "source_record_hash": None,
        "ingested_at": INGESTED_AT,
        "quality_flags": [],
        "missingness_reason": None,
    }


def test_derived_totals_are_not_compared() -> None:
    results = [
        _reported_result(kind="squat_best", value=200.0, identifier="c1"),
        _reported_result(kind="bench_best", value=140.0, identifier="c2"),
        _reported_result(kind="deadlift_best", value=240.0, identifier="c3"),
        {**_reported_result(kind="total", value=999.0, identifier="c4"), "is_derived": True},
    ]
    tables = _tables(competition=[_competition()], competition_reported_result=results)
    report = validate_tables(tables)
    assert "reported_total_mismatch" not in _codes(report)


def test_declarative_layer_rejects_an_out_of_range_column() -> None:
    """A value the contract could not produce is caught column-wise."""
    table = _table_for("performed_set", [_set_row()])
    corrupted = table.set_column(
        table.schema.get_field_index("rpe"),
        "rpe",
        pa.array([99.0], pa.float64()),
    )
    frame = from_arrow_frame(corrupted.select(column_order("performed_set")))
    issues = column_issues(frame, table_name="performed_set")
    assert issues
    assert all(issue.severity is Severity.ERROR for issue in issues)


def test_declarative_layer_rejects_a_zero_load() -> None:
    table = _table_for("performed_set", [_set_row()])
    frame = from_arrow_frame(table.select(column_order("performed_set")))
    corrupted = frame.with_columns(pl.lit(0.0).alias("load_kg"))
    issues = column_issues(corrupted, table_name="performed_set")
    assert any(issue.code == "range_load_kg" for issue in issues)


def test_every_table_has_a_polars_dtype_mapping() -> None:
    """A canonical column must be checkable after the Arrow to Polars hop."""
    for name in table_names():
        frame = from_arrow_frame(empty_table(name))
        assert column_issues(frame, table_name=name) == []


def test_vocabulary_membership_is_enforced() -> None:
    frame = from_arrow_frame(empty_table("performed_set"))
    assert vocabulary_issues(frame, table_name="performed_set") == []
    corrupted = frame.clear().select(pl.lit("teleported").alias("set_status"))
    issues = vocabulary_issues(corrupted, table_name="performed_set")
    assert any(issue.code == "vocabulary_set_status" for issue in issues)


def test_missing_column_is_reported() -> None:
    frame = from_arrow_frame(empty_table("performed_set")).select("performed_set_id")
    issues = column_issues(frame, table_name="performed_set")
    assert any(issue.code == "missing_columns" for issue in issues)


def test_unexpected_column_is_reported() -> None:
    frame = from_arrow_frame(empty_table("performed_set")).with_columns(pl.lit(1).alias("surprise"))
    issues = column_issues(frame, table_name="performed_set")
    assert any(issue.code == "unexpected_columns" for issue in issues)


def test_report_serializes_to_json_ready_mapping() -> None:
    report = validate_tables(_tables(performed_set={"rpe": 5.0, "rir": 2.0}))
    payload = report.to_dict()
    assert payload["ok"] is True
    assert isinstance(payload["issues"], list)
    summary: str = str(payload["summary"])
    assert "0 error(s)" in summary


def test_issue_formatting_includes_location() -> None:
    report = validate_tables(_tables(performed_set={"quality_flags": ["edited_after_the_fact"]}))
    rendered = report.errors[0].format()
    assert "performed_set" in rendered
    assert "edit_without_timestamp" in rendered
