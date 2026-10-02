"""Tests for the competition, attempt, and reported-result semantics.

These are the contracts that keep a source's own meaning intact: a signed attempt
value, a fourth attempt that counts toward nothing, a negative reported best, an
approximate age, a placement that is not a number, and a meet that is not an
athlete outcome.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from psd.schema.models import (
    ORDERED_ATTEMPT_NUMBERS,
    RECORD_FOURTH_ATTEMPT_NUMBER,
    CompetitionAttemptRecord,
    CompetitionMeetRecord,
    CompetitionRecord,
    CompetitionReportedResultRecord,
)
from psd.schema.vocabulary import (
    AgePrecision,
    AttemptOrderBasis,
    AttemptResult,
    AttemptRole,
    CompetitionEvent,
    CompetitionResultKind,
    EquipmentClass,
    EventTimePrecision,
    LiftType,
    ParticipationStatus,
    ReportedBestSemantics,
)

INGESTED_AT = datetime(2026, 3, 1, 12, 0, tzinfo=UTC)
MEET_DATE = datetime(2025, 11, 8, 0, 0, tzinfo=UTC)
ATHLETE_ID = "ath_11111111111111111111111111111111"
COMPETITION_ID = "cmp_22222222222222222222222222222222"
ATTEMPT_ID = "catt_33333333333333333333333333333333"
MEET_ID = "cmeet_44444444444444444444444444444444"
REPORTED_ID = "cres_55555555555555555555555555555555"


def _provenance() -> dict[str, object]:
    return {
        "source_id": "openpowerlifting_probe",
        "source_record_key": "opl:row=1",
        "source_record_hash": None,
        "ingested_at": INGESTED_AT,
        "quality_flags": (),
        "missingness_reason": None,
    }


def _meet(**overrides: object) -> CompetitionMeetRecord:
    payload: dict[str, object] = {
        "competition_meet_id": MEET_ID,
        "meet_name": "Raw National Championships",
        "meet_date": MEET_DATE,
        "event_time_precision": EventTimePrecision.DATE_ONLY,
        "meet_federation": "USPA",
        "meet_parent_federation": "IPF",
        "meet_country": "USA",
        "meet_state": "TX",
        "sanctioned_status_raw": "Yes",
        "is_sanctioned": True,
        "created_at": MEET_DATE,
        **_provenance(),
    }
    payload.update(overrides)
    return CompetitionMeetRecord(**payload)  # type: ignore[arg-type]


def _competition(**overrides: object) -> CompetitionRecord:
    payload: dict[str, object] = {
        "competition_id": COMPETITION_ID,
        "athlete_id": ATHLETE_ID,
        "competition_meet_id": MEET_ID,
        "competition_date": MEET_DATE,
        "event_time_precision": EventTimePrecision.DATE_ONLY,
        "competition_event": CompetitionEvent.SQUAT_BENCH_DEADLIFT,
        "name": "Raw National Championships",
        "federation": "USPA",
        "sanctioning_body": "IPF",
        "equipment_class_raw": "Raw",
        "equipment_class": EquipmentClass.RAW,
        "weight_class_raw": "-93",
        "bodyweight_raw": 91.4,
        "bodyweight_unit": "kg",
        "bodyweight_kg": 91.4,
        "participation_status": "2",
        "participation_place": 2,
        "participation_status_kind": ParticipationStatus.PLACED,
        "is_drug_tested_category": True,
        "age_reported": 23.5,
        "age_precision": AgePrecision.APPROXIMATE,
        "age_class_raw": "40-49",
        "birth_year_class_raw": "40-49",
        "division_raw": "Open",
        "athlete_country_raw": "USA",
        "athlete_region_raw": "TX",
        "observed_at": MEET_DATE,
        **_provenance(),
    }
    payload.update(overrides)
    return CompetitionRecord(**payload)  # type: ignore[arg-type]


def _attempt(**overrides: object) -> CompetitionAttemptRecord:
    payload: dict[str, object] = {
        "competition_attempt_id": ATTEMPT_ID,
        "competition_id": COMPETITION_ID,
        "athlete_id": ATHLETE_ID,
        "lift": LiftType.SQUAT,
        "attempt_number": 2,
        "attempt_role": AttemptRole.ORDERED,
        "attempt_order_basis": AttemptOrderBasis.SOURCE_EXPLICIT,
        "attempt_time": None,
        "source_attempt_raw": -200.0,
        "load_raw": 200.0,
        "load_unit": "kg",
        "load_kg": 200.0,
        "result": AttemptResult.BAD_LIFT,
        "is_opener": False,
        **_provenance(),
    }
    payload.update(overrides)
    return CompetitionAttemptRecord(**payload)  # type: ignore[arg-type]


def _reported(**overrides: object) -> CompetitionReportedResultRecord:
    payload: dict[str, object] = {
        "competition_reported_result_id": REPORTED_ID,
        "competition_id": COMPETITION_ID,
        "athlete_id": ATHLETE_ID,
        "result_kind": CompetitionResultKind.SQUAT_BEST,
        "value": 187.5,
        "unit": "kg",
        "source_value_raw": 187.5,
        "result_source_field": "Best3SquatKg",
        "reported_best_semantics": ReportedBestSemantics.SUCCESSFUL_BEST,
        "is_derived": False,
        "derivation_note": None,
        "observed_at": MEET_DATE,
        **_provenance(),
    }
    payload.update(overrides)
    return CompetitionReportedResultRecord(**payload)  # type: ignore[arg-type]


# -- meet ------------------------------------------------------------------


def test_meet_keeps_hosting_and_sanctioning_bodies_apart() -> None:
    meet = _meet()
    assert meet.meet_federation == "USPA"
    assert meet.meet_parent_federation == "IPF"
    assert meet.meet_federation != meet.meet_parent_federation


def test_meet_date_is_a_start_date_with_date_only_precision() -> None:
    meet = _meet()
    assert meet.event_time_precision is EventTimePrecision.DATE_ONLY


def test_parsed_sanctioned_status_requires_the_sources_own_wording() -> None:
    with pytest.raises(ValidationError, match="requires sanctioned_status_raw"):
        _meet(sanctioned_status_raw=None)


def test_meet_rejects_a_naive_date() -> None:
    with pytest.raises(ValidationError, match="naive timestamp"):
        _meet(meet_date=datetime(2025, 11, 8, 0, 0))


# -- competition -----------------------------------------------------------


def test_competition_links_to_its_meet() -> None:
    assert _competition().competition_meet_id == MEET_ID


def test_approximate_age_survives_unrounded() -> None:
    """23.5 means "23 or 24"; rounding it would invent a precision the source lacks."""
    competition = _competition(age_reported=23.5, age_precision=AgePrecision.APPROXIMATE)
    assert competition.age_reported == 23.5
    assert competition.age_precision is AgePrecision.APPROXIMATE


def test_exact_age_states_its_own_precision() -> None:
    competition = _competition(age_reported=23.0, age_precision=AgePrecision.EXACT)
    assert competition.age_reported == 23.0
    assert competition.age_precision is AgePrecision.EXACT


def test_age_without_precision_is_rejected() -> None:
    with pytest.raises(ValidationError, match="age_reported and age_precision"):
        _competition(age_precision=None)


def test_precision_without_age_is_rejected() -> None:
    with pytest.raises(ValidationError, match="age_reported and age_precision"):
        _competition(age_reported=None)


def test_negative_age_is_rejected() -> None:
    with pytest.raises(ValidationError, match="must not be negative"):
        _competition(age_reported=-1.0, age_precision=AgePrecision.EXACT)


def test_age_class_birth_year_class_and_division_stay_distinct() -> None:
    competition = _competition()
    assert competition.age_class_raw == "40-49"
    assert competition.birth_year_class_raw == "40-49"
    assert competition.division_raw == "Open"


def test_weight_class_keeps_an_open_ended_class_verbatim() -> None:
    assert _competition(weight_class_raw="90+").weight_class_raw == "90+"
    assert _competition(weight_class_raw="-93").weight_class_raw == "-93"


def test_padding_weight_class_is_rejected() -> None:
    with pytest.raises(ValidationError, match="must not be padding"):
        _competition(weight_class_raw="   ")


def test_equipment_category_is_recorded_as_a_category() -> None:
    assert _competition().equipment_class is EquipmentClass.RAW


def test_equipment_class_names_the_federation_categories_that_exist() -> None:
    """Multi-ply and strap-allowed categories must not be folded into another."""
    assert EquipmentClass.MULTI_PLY.value == "multi_ply"
    assert EquipmentClass.STRAPS_ALLOWED.value == "straps_allowed"
    assert EquipmentClass.MULTI_PLY is not EquipmentClass.SINGLE_PLY
    assert EquipmentClass.MULTI_PLY is not EquipmentClass.OTHER


@pytest.mark.parametrize(
    ("status", "kind"),
    [
        ("G", ParticipationStatus.GUEST),
        ("DQ", ParticipationStatus.DISQUALIFIED),
        ("DD", ParticipationStatus.DRUG_DISQUALIFIED),
        ("NS", ParticipationStatus.NO_SHOW),
        ("1", ParticipationStatus.PLACED),
    ],
)
def test_non_numeric_participation_codes_are_not_placings(
    status: str, kind: ParticipationStatus
) -> None:
    competition = _competition(
        participation_status=status,
        participation_place=1 if kind is ParticipationStatus.PLACED else None,
        participation_status_kind=kind,
    )
    assert competition.participation_status == status
    assert competition.participation_status_kind is kind
    if kind is ParticipationStatus.PLACED:
        assert competition.participation_place == 1
    else:
        assert competition.participation_place is None


def test_a_placing_without_a_placed_status_is_rejected() -> None:
    with pytest.raises(ValidationError, match="participation_place is present exactly when"):
        _competition(participation_status_kind=ParticipationStatus.NO_SHOW)


def test_a_placed_status_without_a_placing_is_rejected() -> None:
    with pytest.raises(ValidationError, match="participation_place is present exactly when"):
        _competition(participation_place=None)


def test_tested_category_is_not_an_athlete_attribute() -> None:
    assert _competition().is_drug_tested_category is True


@pytest.mark.parametrize(
    "event",
    [
        CompetitionEvent.SQUAT_BENCH_DEADLIFT,
        CompetitionEvent.BENCH_DEADLIFT,
        CompetitionEvent.SQUAT_DEADLIFT,
        CompetitionEvent.SQUAT_BENCH,
        CompetitionEvent.SQUAT,
        CompetitionEvent.BENCH,
        CompetitionEvent.DEADLIFT,
    ],
)
def test_every_declared_event_is_storable(event: CompetitionEvent) -> None:
    assert _competition(competition_event=event).competition_event is event


def test_body_mass_requires_raw_unit_and_normalized_value_together() -> None:
    with pytest.raises(ValidationError, match="raw value and unit must be recorded together"):
        _competition(bodyweight_unit=None)


# -- attempts --------------------------------------------------------------


def test_a_failed_attempt_keeps_the_signed_source_value() -> None:
    """The sign is the source's result encoding, so it must survive verbatim."""
    attempt = _attempt(source_attempt_raw=-200.0, result=AttemptResult.BAD_LIFT)
    assert attempt.source_attempt_raw == -200.0
    assert attempt.load_kg == 200.0
    assert attempt.result is AttemptResult.BAD_LIFT


def test_a_successful_attempt_keeps_a_positive_load() -> None:
    attempt = _attempt(source_attempt_raw=200.0, load_raw=200.0, result=AttemptResult.GOOD_LIFT)
    assert attempt.load_kg == 200.0
    assert attempt.result is AttemptResult.GOOD_LIFT


def test_a_negative_load_is_never_stored_as_a_physical_load() -> None:
    with pytest.raises(ValidationError, match="is not a plausible mass"):
        _attempt(load_raw=-200.0, load_kg=-200.0)


def test_a_sign_that_contradicts_the_result_is_rejected() -> None:
    with pytest.raises(ValidationError, match="implies a 'bad_lift' attempt"):
        _attempt(source_attempt_raw=-200.0, result=AttemptResult.GOOD_LIFT)


def test_a_load_that_is_not_the_magnitude_of_the_source_value_is_rejected() -> None:
    with pytest.raises(ValidationError, match="is not the magnitude"):
        _attempt(load_raw=180.0, load_kg=180.0)


def test_a_zero_source_attempt_is_rejected() -> None:
    """Zero is not a load and not a missing-value encoding in PSD."""
    with pytest.raises(ValidationError, match="which is not a load"):
        _attempt(source_attempt_raw=0.0)


def test_a_fourth_attempt_is_a_record_attempt() -> None:
    attempt = _attempt(
        attempt_number=RECORD_FOURTH_ATTEMPT_NUMBER,
        attempt_role=AttemptRole.RECORD_FOURTH,
    )
    assert attempt.attempt_role is AttemptRole.RECORD_FOURTH
    assert attempt.attempt_number == 4


def test_a_fourth_attempt_ordered_would_be_mistaken_for_an_ordinary_one() -> None:
    with pytest.raises(ValidationError, match="requires attempt_number in"):
        _attempt(attempt_number=4, attempt_role=AttemptRole.ORDERED)


def test_a_record_attempt_numbered_wrongly_is_rejected() -> None:
    with pytest.raises(ValidationError, match="requires attempt_number=4"):
        _attempt(attempt_number=3, attempt_role=AttemptRole.RECORD_FOURTH)


def test_an_attempt_number_beyond_the_fourth_is_rejected() -> None:
    with pytest.raises(ValidationError):
        _attempt(attempt_number=5)


@pytest.mark.parametrize("number", ORDERED_ATTEMPT_NUMBERS)
def test_the_ordinary_attempt_numbers_remain_ordered_attempts(number: int) -> None:
    assert _attempt(attempt_number=number).attempt_role is AttemptRole.ORDERED


def test_an_attempt_with_no_time_records_no_time() -> None:
    """The source publishes no per-attempt instant, so none is invented."""
    assert _attempt().attempt_time is None


# -- reported results ------------------------------------------------------


def test_a_positive_reported_best_is_a_successful_lift() -> None:
    reported = _reported()
    assert reported.reported_best_semantics is ReportedBestSemantics.SUCCESSFUL_BEST
    assert reported.value == 187.5


def test_a_negative_reported_best_means_a_failed_opener() -> None:
    """Some federations publish the lowest failed weight as a negative best."""
    reported = _reported(
        value=120.0,
        source_value_raw=-120.0,
        reported_best_semantics=ReportedBestSemantics.FAILED_ATTEMPT_ONLY,
    )
    assert reported.value == 120.0
    assert reported.source_value_raw == -120.0
    assert reported.reported_best_semantics is ReportedBestSemantics.FAILED_ATTEMPT_ONLY


def test_a_negative_reported_best_cannot_be_a_successful_lift() -> None:
    with pytest.raises(ValidationError, match="a negative successful lift does not exist"):
        _reported(
            value=120.0,
            source_value_raw=-120.0,
            reported_best_semantics=ReportedBestSemantics.SUCCESSFUL_BEST,
        )


def test_reported_best_semantics_must_match_the_source_sign() -> None:
    """A positive reported best is a successful lift, whatever the column claims."""
    with pytest.raises(ValidationError, match="contradicts source_value_raw"):
        _reported(reported_best_semantics=ReportedBestSemantics.FAILED_ATTEMPT_ONLY)


def test_a_reported_best_never_stores_a_negative_mass() -> None:
    """The sign lives in source_value_raw; value is the magnitude, always positive."""
    with pytest.raises(ValidationError, match="must be a positive magnitude"):
        _reported(value=-120.0, source_value_raw=-120.0)


def test_reported_best_semantics_is_meaningless_for_a_total() -> None:
    with pytest.raises(ValidationError, match="only meaningful for a reported best"):
        _reported(
            result_kind=CompetitionResultKind.TOTAL,
            reported_best_semantics=ReportedBestSemantics.SUCCESSFUL_BEST,
        )


def test_reported_best_semantics_may_be_omitted_for_a_positive_best() -> None:
    """Absence stays absence; the value already fixes the only possible reading."""
    assert _reported(reported_best_semantics=None).reported_best_semantics is None


def test_a_total_without_component_lifts_is_recorded_as_given() -> None:
    """PSD records the total it was given and never manufactures the components."""
    total = _reported(
        result_kind=CompetitionResultKind.TOTAL,
        value=500.0,
        source_value_raw=500.0,
        result_source_field="TotalKg",
        reported_best_semantics=None,
        unit="kg",
    )
    assert total.value == 500.0
    assert total.result_source_field == "TotalKg"


def test_source_field_names_the_scoring_system() -> None:
    assert _reported(result_source_field="Best3SquatKg").result_source_field == "Best3SquatKg"


def test_a_zero_reported_value_is_rejected() -> None:
    with pytest.raises(ValidationError, match="is not a reported value"):
        _reported(source_value_raw=0.0)


def test_a_value_that_is_not_the_magnitude_of_its_source_is_rejected() -> None:
    with pytest.raises(ValidationError, match="is not the magnitude"):
        _reported(value=180.0, source_value_raw=187.5)
