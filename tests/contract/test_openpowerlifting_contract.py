"""Tests for the OpenPowerlifting source schema contract.

The contract is what stops schema drift from silently changing the corpus. These
tests pin the declared header, the classification of every column, and the failure
behaviour when a snapshot's columns do not match.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from psd.ingest.openpowerlifting.contract import (
    SOURCE_COLUMNS,
    SourceColumnDisposition,
    SourceSchemaError,
    attempt_source_columns,
    column_spec,
    expected_columns,
    read_source_header,
    reported_result_source_columns,
    require_source_schema,
    review_source_schema,
)

#: The header as published, in order. Written out here rather than imported so a
#: change to the contract has to be a deliberate edit to this test.
PUBLISHED_HEADER: tuple[str, ...] = (
    "Name",
    "Sex",
    "Event",
    "Equipment",
    "Age",
    "AgeClass",
    "BirthYearClass",
    "Division",
    "BodyweightKg",
    "WeightClassKg",
    "Squat1Kg",
    "Bench1Kg",
    "Deadlift1Kg",
    "Squat2Kg",
    "Bench2Kg",
    "Deadlift2Kg",
    "Squat3Kg",
    "Bench3Kg",
    "Deadlift3Kg",
    "Squat4Kg",
    "Bench4Kg",
    "Deadlift4Kg",
    "Best3SquatKg",
    "Best3BenchKg",
    "Best3DeadliftKg",
    "TotalKg",
    "Place",
    "Dots",
    "Wilks",
    "Glossbrenner",
    "Goodlift",
    "Tested",
    "Country",
    "State",
    "Federation",
    "ParentFederation",
    "Date",
    "MeetCountry",
    "MeetState",
    "MeetName",
    "Sanctioned",
)


def test_contract_declares_the_published_header_in_order() -> None:
    assert expected_columns() == PUBLISHED_HEADER


def test_every_column_is_declared_exactly_once() -> None:
    names = [spec.name for spec in SOURCE_COLUMNS]
    assert len(names) == len(set(names))
    assert len(names) == len(PUBLISHED_HEADER)


def test_every_column_is_classified() -> None:
    assert all(spec.disposition in set(SourceColumnDisposition) for spec in SOURCE_COLUMNS)


def test_every_mapped_or_preserved_column_names_a_destination() -> None:
    for spec in SOURCE_COLUMNS:
        if spec.disposition is SourceColumnDisposition.IGNORED:
            continue
        assert spec.destination is not None, spec.name


def test_every_ignored_column_states_a_reason() -> None:
    for spec in SOURCE_COLUMNS:
        if spec.disposition is not SourceColumnDisposition.IGNORED:
            continue
        assert spec.reason, spec.name


def test_no_column_is_ignored_today() -> None:
    """Every published column currently has a canonical home; none is dropped."""
    ignored = [
        spec.name for spec in SOURCE_COLUMNS if spec.disposition is SourceColumnDisposition.IGNORED
    ]
    assert ignored == []


def test_the_load_bearing_columns_explain_themselves() -> None:
    """The semantics that motivated the contract must be recorded, not implied."""
    explained = {
        "Name",
        "Sex",
        "Event",
        "Equipment",
        "Age",
        "AgeClass",
        "BirthYearClass",
        "Division",
        "WeightClassKg",
        "Squat4Kg",
        "Best3SquatKg",
        "TotalKg",
        "Place",
        "Tested",
        "Federation",
        "ParentFederation",
        "Date",
        "MeetName",
        "Sanctioned",
    }
    for name in explained:
        spec = column_spec(name)
        assert spec.note is not None, name
        assert len(spec.note) > 40, name


def test_a_matching_header_classifies_without_drift() -> None:
    review = review_source_schema(PUBLISHED_HEADER)
    assert not review.drifted
    assert review.unknown == ()
    assert review.missing == ()
    assert review.reordered is False
    assert set(review.mapped) | set(review.preserved) == set(PUBLISHED_HEADER)
    assert "Name" in review.mapped
    assert "Division" in review.preserved


def test_a_new_column_is_unknown_drift() -> None:
    header = (*PUBLISHED_HEADER, "Squat5Kg")
    review = review_source_schema(header)
    assert review.unknown == ("Squat5Kg",)
    assert review.drifted


def test_a_renamed_column_is_both_unknown_and_missing() -> None:
    header = tuple("SquatKg1" if name == "Squat1Kg" else name for name in PUBLISHED_HEADER)
    review = review_source_schema(header)
    assert review.unknown == ("SquatKg1",)
    assert review.missing == ("Squat1Kg",)


def test_a_removed_column_is_missing_drift() -> None:
    header = tuple(name for name in PUBLISHED_HEADER if name != "Glossbrenner")
    review = review_source_schema(header)
    assert review.missing == ("Glossbrenner",)
    assert review.drifted


def test_a_reordered_header_is_drift() -> None:
    header = (PUBLISHED_HEADER[1], PUBLISHED_HEADER[0], *PUBLISHED_HEADER[2:])
    review = review_source_schema(header)
    assert review.reordered is True
    assert review.drifted


def test_require_source_schema_accepts_the_published_header() -> None:
    assert require_source_schema(PUBLISHED_HEADER).drifted is False


def test_require_source_schema_refuses_an_unknown_column() -> None:
    with pytest.raises(SourceSchemaError, match="unknown source column"):
        require_source_schema((*PUBLISHED_HEADER, "NewField"))


def test_require_source_schema_refuses_a_missing_column() -> None:
    header = tuple(name for name in PUBLISHED_HEADER if name != "TotalKg")
    with pytest.raises(SourceSchemaError, match="missing source column"):
        require_source_schema(header)


def test_require_source_schema_refuses_a_reorder() -> None:
    header = (PUBLISHED_HEADER[1], PUBLISHED_HEADER[0], *PUBLISHED_HEADER[2:])
    with pytest.raises(SourceSchemaError, match="different order"):
        require_source_schema(header)


def test_a_reviewed_unknown_column_can_be_accepted_explicitly() -> None:
    """Drift may be tolerated, but only by naming the column that was reviewed."""
    header = (*PUBLISHED_HEADER, "Squat5Kg")
    review = require_source_schema(header, allow_unknown=("Squat5Kg",))
    assert review.unknown == ("Squat5Kg",)


def test_accepting_one_unknown_column_does_not_accept_another() -> None:
    header = (*PUBLISHED_HEADER, "Squat5Kg", "Bench5Kg")
    with pytest.raises(SourceSchemaError, match="Bench5Kg"):
        require_source_schema(header, allow_unknown=("Squat5Kg",))


def test_unknown_column_spec_lookup_raises() -> None:
    with pytest.raises(KeyError, match="Unknown OpenPowerlifting source column"):
        column_spec("NotAColumn")


def test_twelve_attempt_columns_cover_three_lifts_and_four_numbers() -> None:
    columns = attempt_source_columns()
    assert len(columns) == 12
    assert {number for _lift, _column, number in columns} == {1, 2, 3, 4}
    assert {lift for lift, _column, _number in columns} == {"squat", "bench", "deadlift"}


def test_the_fourth_attempt_is_declared_as_a_record_attempt() -> None:
    fourths = [column for _lift, column, number in attempt_source_columns() if number == 4]
    assert sorted(fourths) == ["Bench4Kg", "Deadlift4Kg", "Squat4Kg"]
    for column in fourths:
        spec = column_spec(column)
        assert spec.note is not None
        assert "record" in spec.note.lower()


def test_reported_result_columns_name_their_kinds() -> None:
    pairs = dict(reported_result_source_columns())
    assert pairs["TotalKg"] == "total"
    assert pairs["Best3SquatKg"] == "squat_best"
    assert pairs["Best3BenchKg"] == "bench_best"
    assert pairs["Best3DeadliftKg"] == "deadlift_best"
    assert pairs["Dots"] == "dots"
    assert pairs["Wilks"] == "wilks"
    assert pairs["Goodlift"] == "gl_points"
    assert pairs["Glossbrenner"] == "other"


def test_glossbrenner_is_not_mistaken_for_ipf_gl_points() -> None:
    """Two different scoring systems share this table; the source field separates them."""
    pairs = dict(reported_result_source_columns())
    assert pairs["Glossbrenner"] != pairs["Goodlift"]


def test_read_header_from_a_file(tmp_path: Path) -> None:
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text(",".join(PUBLISHED_HEADER) + "\nJohn Doe,M,BD,Wraps,23,,,,,\n")
    assert read_source_header(csv_path) == PUBLISHED_HEADER


def test_read_header_from_a_missing_file_fails(tmp_path: Path) -> None:
    with pytest.raises(SourceSchemaError, match="No OpenPowerlifting CSV"):
        read_source_header(tmp_path / "absent.csv")


def test_read_header_from_an_empty_file_fails(tmp_path: Path) -> None:
    csv_path = tmp_path / "empty.csv"
    csv_path.write_bytes(b"")
    with pytest.raises(SourceSchemaError, match="no header row"):
        read_source_header(csv_path)


def test_review_summary_mentions_drift() -> None:
    review = review_source_schema((*PUBLISHED_HEADER, "Extra"))
    assert "DRIFT" in review.summary()


def test_review_is_json_serializable() -> None:
    payload = review_source_schema(PUBLISHED_HEADER).to_dict()
    assert payload["drifted"] is False
    assert payload["declared"] == list(PUBLISHED_HEADER)
    assert payload["ignored"] == {}
