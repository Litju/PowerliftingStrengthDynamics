"""Tests for the competition cross-record rules.

These are the relationships a per-column check cannot express: an attempt on a lift
the declared event never contested, duplicated meet facts that tell two different
stories, and a dangling meet reference.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from psd.schema.registry import table_names
from psd.schema.vocabulary import ReportedBestSemantics
from psd.validation import validate_tables
from psd.validation.rules import (
    TableRows,
    all_issues,
    check_attempts_belong_to_the_declared_event,
    check_meet_fields_agree_with_the_referenced_meet,
    check_reported_best_semantics,
)

MEET_DATE = datetime(2025, 11, 8, 0, 0, tzinfo=UTC)
INGESTED_AT = datetime(2026, 3, 1, 12, 0, tzinfo=UTC)

ATHLETE_ID = "ath_11111111111111111111111111111111"
COMPETITION_ID = "cmp_22222222222222222222222222222222"
COMPETITION_ID_B = "cmp_66666666666666666666666666666666"
MEET_ID = "cmeet_44444444444444444444444444444444"
SOURCE_ID = "openpowerlifting_probe"


def _rows(**tables: Any) -> TableRows:
    base: dict[str, list[dict[str, Any]]] = {name: [] for name in table_names()}
    for name, rows in tables.items():
        base[name] = list(rows)
    return base


def _meet_row(**overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "competition_meet_id": MEET_ID,
        "meet_name": "Raw National Championships",
        "meet_date": MEET_DATE,
        "event_time_precision": "date_only",
        "meet_federation": "USPA",
        "meet_parent_federation": "IPF",
        "meet_country": "USA",
        "meet_state": "TX",
        "sanctioned_status_raw": "Yes",
        "is_sanctioned": True,
        "created_at": MEET_DATE,
        "source_id": SOURCE_ID,
        "source_record_key": "opl:row=1",
        "source_record_hash": None,
        "ingested_at": INGESTED_AT,
        "quality_flags": [],
        "missingness_reason": None,
    }
    row.update(overrides)
    return row


def _competition_row(**overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "competition_id": COMPETITION_ID,
        "athlete_id": ATHLETE_ID,
        "competition_meet_id": MEET_ID,
        "competition_date": MEET_DATE,
        "event_time_precision": "date_only",
        "competition_event": "sbd",
        "name": "Raw National Championships",
        "federation": "USPA",
        "sanctioning_body": "IPF",
        "location": None,
        "equipment_class_raw": "Raw",
        "equipment_class": "raw",
        "weight_class_raw": "-93",
        "bodyweight_raw": 91.4,
        "bodyweight_unit": "kg",
        "bodyweight_kg": 91.4,
        "participation_status": "2",
        "participation_place": 2,
        "participation_status_kind": "placed",
        "is_drug_tested_category": True,
        "age_reported": 23.5,
        "age_precision": "approximate",
        "age_class_raw": "40-49",
        "birth_year_class_raw": "40-49",
        "division_raw": "Open",
        "athlete_country_raw": "USA",
        "athlete_region_raw": "TX",
        "is_championship": None,
        "created_at": None,
        "scheduled_at": None,
        "performed_at": None,
        "observed_at": MEET_DATE,
        "modified_at": None,
        "source_id": SOURCE_ID,
        "source_record_key": "opl:row=1",
        "source_record_hash": None,
        "ingested_at": INGESTED_AT,
        "quality_flags": [],
        "missingness_reason": None,
    }
    row.update(overrides)
    return row


def _attempt_row(**overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "competition_attempt_id": "catt_33333333333333333333333333333333",
        "competition_id": COMPETITION_ID,
        "athlete_id": ATHLETE_ID,
        "lift": "bench",
        "attempt_number": 1,
        "attempt_role": "ordered",
        "attempt_order_basis": "source_explicit",
        "attempt_time": None,
        "source_attempt_raw": 100.0,
        "load_raw": 100.0,
        "load_unit": "kg",
        "load_kg": 100.0,
        "result": "good_lift",
        "is_opener": True,
        "created_at": None,
        "scheduled_at": None,
        "performed_at": None,
        "observed_at": MEET_DATE,
        "modified_at": None,
        "source_id": SOURCE_ID,
        "source_record_key": "opl:row=1",
        "source_record_hash": None,
        "ingested_at": INGESTED_AT,
        "quality_flags": [],
        "missingness_reason": None,
    }
    row.update(overrides)
    return row


def _reported_row(**overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "competition_reported_result_id": "cres_55555555555555555555555555555555",
        "competition_id": COMPETITION_ID,
        "athlete_id": ATHLETE_ID,
        "result_kind": "squat_best",
        "value": 120.0,
        "unit": "kg",
        "source_value_raw": -120.0,
        "result_source_field": "Best3SquatKg",
        "reported_best_semantics": "failed_attempt_only",
        "is_derived": False,
        "derivation_note": None,
        "created_at": None,
        "scheduled_at": None,
        "performed_at": None,
        "observed_at": MEET_DATE,
        "modified_at": None,
        "source_id": SOURCE_ID,
        "source_record_key": "opl:row=1",
        "source_record_hash": None,
        "ingested_at": INGESTED_AT,
        "quality_flags": [],
        "missingness_reason": None,
    }
    row.update(overrides)
    return row


def test_an_attempt_must_belong_to_the_declared_event() -> None:
    """A bench attempt in a squat-only event is a contradiction, not a fact."""
    tables = _rows(
        competition=[_competition_row(competition_event="s")],
        competition_attempt=[_attempt_row(lift="bench")],
    )
    codes = {issue.code for issue in check_attempts_belong_to_the_declared_event(tables)}
    assert codes == {"attempt_outside_declared_event"}


def test_a_reduced_event_accepts_only_its_own_lifts() -> None:
    tables = _rows(
        competition=[_competition_row(competition_event="bd")],
        competition_attempt=[
            _attempt_row(lift="bench", competition_attempt_id="catt_1" + "1" * 31),
            _attempt_row(lift="deadlift", competition_attempt_id="catt_2" + "2" * 31),
        ],
    )
    assert check_attempts_belong_to_the_declared_event(tables) == []


def test_a_missing_lift_is_not_a_failed_lift() -> None:
    """No attempt row exists for an uncontested lift, which is not a defect."""
    tables = _rows(
        competition=[_competition_row(competition_event="b")],
        competition_attempt=[_attempt_row(lift="bench")],
    )
    assert check_attempts_belong_to_the_declared_event(tables) == []


def test_agreeing_meet_facts_produce_no_issue() -> None:
    tables = _rows(
        competition_meet=[_meet_row()],
        competition=[_competition_row()],
    )
    assert check_meet_fields_agree_with_the_referenced_meet(tables) == []


@pytest.mark.parametrize(
    ("competition_column", "value"),
    [
        ("name", "Some Other Open"),
        ("federation", "CPU"),
        ("sanctioning_body", "USAPL"),
    ],
)
def test_disagreeing_meet_facts_are_an_error(competition_column: str, value: str) -> None:
    """The duplicated meet facts must not tell two different stories."""
    tables = _rows(
        competition_meet=[_meet_row()],
        competition=[_competition_row(**{competition_column: value})],
    )
    issues = check_meet_fields_agree_with_the_referenced_meet(tables)
    assert [issue.code for issue in issues] == ["meet_field_disagreement"]
    assert competition_column in issues[0].message


def test_a_dangling_meet_reference_is_an_error() -> None:
    tables = _rows(competition=[_competition_row(competition_meet_id="cmeet_" + "9" * 32)])
    codes = {issue.code for issue in all_issues(tables)}
    assert "dangling_meet" in codes


def test_a_negative_reported_best_must_declare_its_meaning() -> None:
    tables = _rows(
        competition_reported_result=[
            _reported_row(reported_best_semantics=ReportedBestSemantics.SUCCESSFUL_BEST.value)
        ]
    )
    codes = {issue.code for issue in check_reported_best_semantics(tables)}
    assert codes == {"reported_best_semantics_mismatch"}


def test_a_declared_negative_reported_best_is_accepted() -> None:
    assert check_reported_best_semantics(_rows(competition_reported_result=[_reported_row()])) == []


def test_a_positive_reported_best_ignores_the_semantics_rule() -> None:
    tables = _rows(
        competition_reported_result=[
            _reported_row(value=187.5, source_value_raw=187.5, reported_best_semantics=None)
        ]
    )
    assert check_reported_best_semantics(tables) == []


def test_a_non_best_result_kind_is_out_of_scope_for_the_semantics_rule() -> None:
    tables = _rows(
        competition_reported_result=[
            _reported_row(
                result_kind="total",
                value=500.0,
                source_value_raw=-500.0,
                reported_best_semantics=None,
            )
        ]
    )
    assert check_reported_best_semantics(tables) == []


def test_two_lifters_at_one_meet_agree_on_the_meet_facts() -> None:
    tables = _rows(
        competition_meet=[_meet_row()],
        competition=[
            _competition_row(),
            _competition_row(competition_id=COMPETITION_ID_B, athlete_id=ATHLETE_ID[:-1] + "9"),
        ],
    )
    assert check_meet_fields_agree_with_the_referenced_meet(tables) == []


def test_the_new_rules_are_wired_into_the_dataset_entry_point() -> None:
    """The rules run through ``all_issues``, not only when called directly."""
    tables = _rows(
        competition=[_competition_row(competition_event="s")],
        competition_attempt=[_attempt_row(lift="bench")],
    )
    assert any(issue.code == "attempt_outside_declared_event" for issue in all_issues(tables))
    assert validate_tables is not None
