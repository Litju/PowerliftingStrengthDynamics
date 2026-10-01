"""The adversarial athlete history.

Every case the PSD design documents name explicitly is represented here, and each
one is a case a real ingestion must survive without semantic loss:

* **missingness** -- absent RPE, absent velocity, absent body mass, and a sparse
  partial log, each with an explicit reason rather than a zero;
* **planned versus performed** -- a plan exists, is modified mid-week, and is
  executed differently; another session is performed with *no* plan at all, and no
  prescription is manufactured for it;
* **failures** -- a failed set with ``reps_performed = 0``, which is a recorded
  zero and not a missing value, plus a partial set and an incomplete set;
* **program modification** -- a versioned change stamped with ``known_at`` after
  the session was scheduled, which is what makes leakage control decidable;
* **multiple same-day sessions** -- two sessions on one date distinguished by
  ``session_order_index``;
* **tests** -- a structured test and a meet-derived result kept distinct;
* **body mass** -- recorded in pounds for part of the history and kilograms for
  the rest, with both raw and normalized values preserved, plus a
  percentage-based measurement that has no mass unit at all;
* **velocity when present** -- present for one exercise only;
* **competition attempts** -- a meet with only two squat attempts, an explicit
  ``no_attempt`` slot, and date-only event precision;
* **source provenance** -- an unresolved cross-source link, ambiguous identity
  status, and a post-hoc edit carrying ``modified_at``;
* **equipment transitions** -- a bar and a shoe change mid-history;
* **substitutions** -- a prescribed exercise replaced during execution.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from pydantic import BaseModel

from psd.provenance.manifest import LineageEntry
from psd.schema.identifiers import IdPrefix, derive_id, make_id
from psd.schema.models import (
    AthleteSourceLinkRecord,
    BodyMeasurementRecord,
    EquipmentStateRecord,
    ExerciseDefinitionRecord,
    ObservationRecord,
    PerformanceTestRecord,
    PerformedExerciseRecord,
    PerformedRepRecord,
    PerformedSessionRecord,
    PerformedSetRecord,
    PlannedExerciseRecord,
    PlannedSessionRecord,
    PlannedSetRecord,
    ProgramModificationRecord,
    ProgramVersionRecord,
    VelocityObservationRecord,
)
from psd.schema.vocabulary import (
    AttemptResult,
    BodyMassContext,
    BodyMeasurementMethod,
    BodyMeasurementType,
    CompetitionResultKind,
    EquipmentItem,
    EventTimePrecision,
    IdentityLinkMethod,
    IdentityStatus,
    LiftType,
    MissingnessReason,
    NotPerformedReason,
    ObservationMethod,
    ObservationScope,
    ObservationType,
    PrescriptionBasis,
    ProgramModificationKind,
    QualityFlag,
    ReporterRole,
    RepStatus,
    SessionStatus,
    SessionType,
    SetStatus,
    TestType,
    ValueEncoding,
    VelocityMethod,
)
from tests.fixtures.builders import (
    ATHLETE_ID,
    HistoryBuilder,
    SourceIds,
    add_athlete,
    add_competition,
    add_exercises,
    add_program,
    add_source_records,
    at,
    kg,
    lb,
)

__all__ = (
    "ADVERSARIAL_ATTEMPTS",
    "ADVERSARIAL_REPORTED",
    "SOURCE_IDS",
    "adversarial_history_records",
    "synthetic_records",
)

SOURCE_IDS = SourceIds()

ADVERSARIAL_ATTEMPTS: tuple[tuple[LiftType, int, float, AttemptResult], ...] = (
    # Only two squat attempts: the third was never taken and simply has no row.
    (LiftType.SQUAT, 1, 150.0, AttemptResult.GOOD_LIFT),
    (LiftType.SQUAT, 2, 157.5, AttemptResult.GOOD_LIFT),
    (LiftType.BENCH, 1, 100.0, AttemptResult.GOOD_LIFT),
    (LiftType.BENCH, 2, 105.0, AttemptResult.BAD_LIFT),
    (LiftType.BENCH, 3, 105.0, AttemptResult.GOOD_LIFT),
)

ADVERSARIAL_REPORTED: tuple[tuple[CompetitionResultKind, float, bool], ...] = (
    (CompetitionResultKind.SQUAT_BEST, 157.5, False),
    (CompetitionResultKind.BENCH_BEST, 105.0, False),
    # No deadlift best is reported at all: the athlete withdrew with no attempts.
    # The total below is derived by PSD from the two reported bests.
    (CompetitionResultKind.TOTAL, 262.5, True),
)

#: An attempt slot the source explicitly reported as empty, after a withdrawal.
#: It carries no load, because zero is not how PSD encodes "not attempted".
WITHDRAWN_ATTEMPT: tuple[LiftType, int, float | None, AttemptResult] = (
    LiftType.DEADLIFT,
    1,
    None,
    AttemptResult.NO_ATTEMPT,
)

SESSION_COUNT = 10
SESSION_STRIDE = 4
SKIPS: dict[int, str] = {3: "injury", 4: "scheduled_rest"}


def adversarial_history_records() -> dict[str, list[BaseModel]]:
    """Return the adversarial athlete history.

    Returns:
        Table name to validated records, covering every canonical table.
    """
    builder = HistoryBuilder(athlete_id=ATHLETE_ID, source_id=SOURCE_IDS.training)
    add_source_records(builder, ids=SOURCE_IDS)
    add_athlete(builder, identity_status=IdentityStatus.AMBIGUOUS_UNRESOLVED)
    _add_unresolved_link(builder)
    exercises = add_exercises(builder)
    program = add_program(builder, versions=(("v1", 0), ("v2", 28)))
    _add_equipment(builder)
    _add_sparse_body_mass(builder)
    _add_plan_with_skips(builder, program["v1"], program["v2"])
    _add_execution(builder, exercises)
    _add_observations(builder)
    _add_tests_and_velocity(builder, exercises)
    add_competition(
        builder,
        competition_id=make_id(IdPrefix.COMPETITION, "regional-open", "2021-05"),
        athlete_id=ATHLETE_ID,
        offset=95,
        attempts=(*ADVERSARIAL_ATTEMPTS, WITHDRAWN_ATTEMPT),
        source_id=SOURCE_IDS.competition,
        reported=ADVERSARIAL_REPORTED,
    )
    _add_lineage(builder)
    return builder.build()


def synthetic_records() -> dict[str, list[BaseModel]]:
    """Return a small synthetic (PSD-Sim style) history.

    Returned separately from the real history so the two regimes are never mixed
    in one canonical dataset.

    Returns:
        Table name to validated records.
    """
    synthetic_athlete = make_id(IdPrefix.ATHLETE, "sim", "athlete-007")
    builder = HistoryBuilder(athlete_id=synthetic_athlete, source_id=SOURCE_IDS.synthetic)
    add_source_records(builder, ids=SOURCE_IDS)
    add_athlete(
        builder,
        athlete_id=synthetic_athlete,
        pseudonym="sim-athlete-007",
        is_synthetic=True,
        source_id=SOURCE_IDS.synthetic,
    )
    exercises = add_exercises(builder)
    _add_simulated_sets(builder, exercises)
    return builder.build()


def _add_unresolved_link(builder: HistoryBuilder) -> None:
    """Record a cross-source link that could not be resolved."""

    builder.add(
        "athlete_source_link",
        AthleteSourceLinkRecord(
            athlete_id=builder.athlete_id,
            source_athlete_key=None,
            link_method=IdentityLinkMethod.UNRESOLVED,
            link_confidence=None,
            is_primary=False,
            **builder.provenance(source_id=SOURCE_IDS.competition),
        ),
    )


def _add_equipment(builder: HistoryBuilder) -> None:
    """Record non-overlapping equipment transitions."""

    schedule: tuple[tuple[EquipmentItem, str, int, int | None], ...] = (
        (EquipmentItem.BAR, "bar-old", 0, 150),
        (EquipmentItem.BAR, "bar-new", 150, None),
        (EquipmentItem.SHOES, "shoes-a", 0, 90),
        (EquipmentItem.SHOES, "shoes-b", 90, None),
    )
    for item, identifier, start, end in schedule:
        builder.add(
            "equipment_state",
            EquipmentStateRecord(
                equipment_state_id=make_id(IdPrefix.EQUIPMENT_STATE, item.value, identifier),
                athlete_id=builder.athlete_id,
                equipment_item=item,
                effective_from=at(start),
                effective_to=None if end is None else at(end),
                identifier=identifier,
                **builder.provenance(source_record_key=f"{item.value}:{identifier}"),
            ),
        )


def _add_sparse_body_mass(builder: HistoryBuilder) -> None:
    """Record body mass in pounds, then kilograms, plus a percentage-only measure."""

    raw, normalized = lb(205.0)
    builder.add(
        "body_measurement",
        BodyMeasurementRecord(
            body_measurement_id=make_id(IdPrefix.BODY_MEASUREMENT, "weigh-in", 0),
            athlete_id=builder.athlete_id,
            measured_at=at(1, 7),
            measurement_type=BodyMeasurementType.BODY_MASS,
            event_time_precision=EventTimePrecision.MINUTE,
            measurement_context=BodyMassContext.MORNING_FASTED,
            method=BodyMeasurementMethod.BATHROOM_SCALE,
            raw_value=raw,
            raw_unit="lb",
            value_normalized=normalized,
            unit_normalized="kg",
            **builder.provenance(source_record_key="weigh-in:lb"),
        ),
    )
    builder.add(
        "body_measurement",
        BodyMeasurementRecord(
            body_measurement_id=make_id(IdPrefix.BODY_MEASUREMENT, "weigh-in", 1),
            athlete_id=builder.athlete_id,
            measured_at=at(30, 7),
            measurement_type=BodyMeasurementType.BODY_MASS,
            event_time_precision=EventTimePrecision.MINUTE,
            measurement_context=BodyMassContext.WEIGH_IN,
            method=BodyMeasurementMethod.CLINICAL_SCALE,
            raw_value=kg(93.0),
            raw_unit="kg",
            value_normalized=kg(93.0),
            unit_normalized="kg",
            **builder.provenance(source_record_key="weigh-in:kg"),
        ),
    )
    # A body-fat percentage has no mass unit at all, so nothing is normalized.
    builder.add(
        "body_measurement",
        BodyMeasurementRecord(
            body_measurement_id=make_id(IdPrefix.BODY_MEASUREMENT, "body-fat", 0),
            athlete_id=builder.athlete_id,
            measured_at=at(60, 7),
            measurement_type=BodyMeasurementType.BODY_FAT_PERCENTAGE,
            event_time_precision=EventTimePrecision.MINUTE,
            measurement_context=BodyMassContext.UNSPECIFIED,
            method=BodyMeasurementMethod.BIOIMPEDANCE,
            raw_value=14.2,
            raw_unit=None,
            value_normalized=14.2,
            unit_normalized="percent",
            **builder.provenance(
                source_record_key="body-fat:0",
                missingness_reason=MissingnessReason.NOT_APPLICABLE,
            ),
        ),
    )


def _planned_session_id(session_index: int) -> str:
    """Return the adversarial planned-session id for a session index."""
    return make_id(IdPrefix.PLANNED_SESSION, "adv", session_index)


def _planned_exercise_id(session_index: int) -> str:
    """Return the planned-exercise id for a session index.

    The plan prescribes a single exercise per session, so the ordinal is fixed.
    Execution has to derive the same id to link back without inventing a
    prescription; deriving it from the wrong parent would produce a link that
    resolves to nothing.
    """
    return derive_id(_planned_session_id(session_index), IdPrefix.PLANNED_EXERCISE, 0)


def _planned_set_id(session_index: int, ordinal: int = 0) -> str:
    """Return the planned-set id for a session index and set ordinal."""
    return derive_id(_planned_exercise_id(session_index), IdPrefix.PLANNED_SET, ordinal)


def _set_missingness_reason(rpe: float | None, rir: float | None) -> MissingnessReason | None:
    """Return why a set's effort data is incomplete, or ``None`` when it is complete.

    PSD requires every null ``rpe`` or ``rir`` to declare *why* it is null, so an
    absent value is never silently ambiguous.
    """
    if rpe is not None and rir is not None:
        return None
    return MissingnessReason.NOT_RECORDED_IN_SOURCE


def _add_plan_with_skips(
    builder: HistoryBuilder, version_one: ProgramVersionRecord, version_two: ProgramVersionRecord
) -> None:
    """Prescribe ten sessions, two of which are skipped with a reason."""

    for session_index in range(SESSION_COUNT):
        planned_session_id = _planned_session_id(session_index)
        skip_reason = SKIPS.get(session_index)
        builder.add(
            "planned_session",
            PlannedSessionRecord(
                planned_session_id=planned_session_id,
                athlete_id=builder.athlete_id,
                program_version_id=version_one.program_version_id
                if session_index < 7
                else version_two.program_version_id,
                scheduled_at=at(session_index * SESSION_STRIDE, 9),
                session_order_index=0,
                session_status=SessionStatus.SKIPPED
                if skip_reason is not None
                else SessionStatus.PLANNED,
                not_performed_reason=NotPerformedReason(skip_reason) if skip_reason else None,
                template_label="day_b",
                **builder.provenance(source_record_key=f"adv-plan:{session_index}"),
            ),
        )
        if skip_reason is not None:
            continue
        planned_exercise_id = _planned_exercise_id(session_index)
        builder.add(
            "planned_exercise",
            PlannedExerciseRecord(
                planned_exercise_id=planned_exercise_id,
                planned_session_id=planned_session_id,
                exercise_id=make_id(IdPrefix.EXERCISE_DEFINITION, "bench"),
                ordinal=1,
                **builder.provenance(source_record_key=f"adv-plan:{session_index}:bench"),
            ),
        )
        builder.add(
            "planned_set",
            PlannedSetRecord(
                planned_set_id=_planned_set_id(session_index),
                planned_exercise_id=planned_exercise_id,
                ordinal=1,
                target_reps_min=4,
                target_reps_max=6,
                target_rpe=8.0,
                prescription_basis=PrescriptionBasis.RPE_ANCHORED,
                **builder.provenance(source_record_key=f"adv-plan:{session_index}:set"),
            ),
        )

    # An autoregulated load change selected *after* the session was scheduled.
    builder.add(
        "program_modification",
        ProgramModificationRecord(
            program_modification_id=make_id(IdPrefix.PROGRAM_MODIFICATION, "load-drop", 2),
            athlete_id=builder.athlete_id,
            planned_session_id=_planned_session_id(2),
            program_version_id=version_one.program_version_id,
            known_at=at(8, 2),
            change_kind=ProgramModificationKind.LOAD_CHANGE,
            field_changed="target_load_kg",
            previous_value="102.5",
            new_value="97.5",
            value_encoding=ValueEncoding.NUMBER,
            rationale="Sore elbow after the previous session.",
            **builder.provenance(source_record_key="modification:load-drop"),
        ),
    )
    builder.add(
        "program_modification",
        ProgramModificationRecord(
            program_modification_id=make_id(IdPrefix.PROGRAM_MODIFICATION, "substitute", 5),
            athlete_id=builder.athlete_id,
            planned_session_id=_planned_session_id(5),
            program_version_id=version_two.program_version_id,
            known_at=at(21, 1),
            change_kind=ProgramModificationKind.EXERCISE_SUBSTITUTION,
            field_changed="exercise_id",
            previous_value=make_id(IdPrefix.EXERCISE_DEFINITION, "bench"),
            new_value=make_id(IdPrefix.EXERCISE_DEFINITION, "pause_bench"),
            value_encoding=ValueEncoding.TEXT,
            **builder.provenance(source_record_key="modification:substitute"),
        ),
    )


def _add_execution(builder: HistoryBuilder, exercises: dict[str, ExerciseDefinitionRecord]) -> None:
    """Execute sessions including failures, a substitution, and an unplanned day."""

    for session_index in range(SESSION_COUNT):
        if session_index in SKIPS:
            # A skipped session has no performed record at all.
            continue
        started = at(session_index * SESSION_STRIDE, 12)
        performed_session_id = make_id(IdPrefix.PERFORMED_SESSION, "adv", session_index)
        unplanned = session_index == 9
        builder.add(
            "performed_session",
            PerformedSessionRecord(
                performed_session_id=performed_session_id,
                athlete_id=builder.athlete_id,
                planned_session_id=None if unplanned else _planned_session_id(session_index),
                started_at=started,
                session_order_index=0,
                session_type=SessionType.TRAINING,
                ended_at=None if session_index == 6 else started + timedelta(hours=1),
                duration_seconds=None if session_index == 6 else 3600.0,
                is_completed=None if session_index == 6 else True,
                performed_at=started,
                **builder.provenance(
                    source_record_key=f"adv-session:{session_index}",
                    quality_flags=(QualityFlag.SELF_REPORTED,),
                    missingness_reason=MissingnessReason.NOT_RECORDED_IN_SOURCE
                    if session_index == 6
                    else None,
                ),
            ),
        )
        exercise_name = "pause_bench" if session_index == 5 else "bench"
        planned_name = "bench" if session_index == 5 else exercise_name
        performed_exercise_id = derive_id(performed_session_id, IdPrefix.PERFORMED_EXERCISE, 0)
        builder.add(
            "performed_exercise",
            PerformedExerciseRecord(
                performed_exercise_id=performed_exercise_id,
                performed_session_id=performed_session_id,
                exercise_id=exercises[exercise_name].exercise_id,
                ordinal=1,
                planned_exercise_id=None if unplanned else _planned_exercise_id(session_index),
                is_substitution=True if session_index == 5 else None,
                substituted_from_exercise_id=exercises[planned_name].exercise_id
                if session_index == 5
                else None,
                performed_at=started,
                **builder.provenance(
                    source_record_key=f"adv-session:{session_index}:{exercise_name}"
                ),
            ),
        )
        _add_performed_sets(
            builder,
            performed_exercise_id=performed_exercise_id,
            started=started,
            session_index=session_index,
            planned_session_id=None if unplanned else _planned_session_id(session_index),
        )

    _add_same_day_pair(builder, exercises)


def _add_performed_sets(
    builder: HistoryBuilder,
    *,
    performed_exercise_id: str,
    started: datetime,
    session_index: int,
    planned_session_id: str | None,
) -> None:
    """Add the sets of one session, covering failure, partial, and absent RPE."""

    plans: tuple[tuple[SetStatus, int, float | None, float | None], ...] = (
        (SetStatus.COMPLETED, 5, 8.0, 2.0),
        (SetStatus.COMPLETED, 5, None, None),
        # A partial set: three reps at RPE 9.5 with nothing left in reserve. RIR
        # is recorded as null rather than 0.0, because zero is how a source spells
        # "did not record this", and PSD refuses to read it as a measurement.
        (SetStatus.PARTIAL, 3, 9.5, None),
    )
    if session_index == 2:
        # A failed set: zero repetitions is the recorded observation, not a
        # missing value, so the status carries it and the load stays populated.
        plans = (
            (SetStatus.FAILED, 0, 9.5, None),
            (SetStatus.COMPLETED, 5, 9.0, 1.0),
        )
    pounds_raw, pounds_kg = lb(225.0)

    for set_index, (status, reps, rpe, rir) in enumerate(plans):
        load = kg(100.0 + 2.5 * session_index)
        use_pounds = session_index == 1 and set_index == 0
        performed_set_id = derive_id(performed_exercise_id, IdPrefix.PERFORMED_SET, set_index)
        edited = session_index == 7
        builder.add(
            "performed_set",
            PerformedSetRecord(
                performed_set_id=performed_set_id,
                performed_exercise_id=performed_exercise_id,
                ordinal=set_index + 1,
                planned_set_id=_planned_set_id(session_index)
                if planned_session_id is not None and set_index == 0
                else None,
                set_status=status,
                load_raw=pounds_raw if use_pounds else load,
                load_unit="lb" if use_pounds else "kg",
                load_kg=pounds_kg if use_pounds else load,
                reps_performed=reps,
                reps_failed=1 if status is SetStatus.PARTIAL else None,
                is_failure=True if status is SetStatus.FAILED else None,
                rpe=rpe,
                rir=rir,
                rep_level_data_available=session_index == 0,
                incomplete_reason=MissingnessReason.NOT_RECORDED_IN_SOURCE if rpe is None else None,
                performed_at=started,
                modified_at=started + timedelta(hours=8) if edited else None,
                **builder.provenance(
                    source_record_key=f"adv-set:{session_index}:{set_index}",
                    quality_flags=(QualityFlag.EDITED_AFTER_THE_FACT,) if edited else (),
                    missingness_reason=_set_missingness_reason(rpe, rir),
                ),
            ),
        )
        if session_index != 0:
            continue
        for rep_index in range(reps):
            builder.add(
                "performed_rep",
                PerformedRepRecord(
                    performed_rep_id=derive_id(performed_set_id, IdPrefix.PERFORMED_REP, rep_index),
                    performed_set_id=performed_set_id,
                    rep_ordinal=rep_index + 1,
                    rep_status=RepStatus.COMPLETED,
                    is_success=True,
                    load_raw=load,
                    load_unit="kg",
                    load_kg=load,
                    rpe=None if rep_index < reps - 1 else 8.0,
                    velocity_mps=round(0.5 - 0.01 * rep_index, 3),
                    performed_at=started,
                    **builder.provenance(
                        source_record_key=f"adv-rep:{session_index}:{set_index}:{rep_index}",
                        missingness_reason=MissingnessReason.NOT_RECORDED_IN_SOURCE
                        if rep_index < reps - 1
                        else None,
                    ),
                ),
            )


def _add_same_day_pair(
    builder: HistoryBuilder, exercises: dict[str, ExerciseDefinitionRecord]
) -> None:
    """Record two sessions on one calendar day, distinguished only by order index."""

    for order, (label, exercise_name) in enumerate((("am", "squat"), ("pm", "deadlift"))):
        started = at(200, 9 + 6 * order)
        performed_session_id = make_id(IdPrefix.PERFORMED_SESSION, "twice", order)
        builder.add(
            "performed_session",
            PerformedSessionRecord(
                performed_session_id=performed_session_id,
                athlete_id=builder.athlete_id,
                planned_session_id=None,
                started_at=started,
                session_order_index=order,
                session_type=SessionType.TRAINING,
                ended_at=started + timedelta(hours=1),
                duration_seconds=3600.0,
                is_completed=True,
                performed_at=started,
                **builder.provenance(source_record_key=f"twice:{label}"),
            ),
        )
        performed_exercise_id = derive_id(performed_session_id, IdPrefix.PERFORMED_EXERCISE, 0)
        builder.add(
            "performed_exercise",
            PerformedExerciseRecord(
                performed_exercise_id=performed_exercise_id,
                performed_session_id=performed_session_id,
                exercise_id=exercises[exercise_name].exercise_id,
                ordinal=1,
                planned_exercise_id=None,
                performed_at=started,
                **builder.provenance(source_record_key=f"twice:{label}:{exercise_name}"),
            ),
        )
        builder.add(
            "performed_set",
            PerformedSetRecord(
                performed_set_id=derive_id(performed_exercise_id, IdPrefix.PERFORMED_SET, 0),
                performed_exercise_id=performed_exercise_id,
                ordinal=1,
                planned_set_id=None,
                set_status=SetStatus.COMPLETED,
                load_raw=kg(120.0),
                load_unit="kg",
                load_kg=kg(120.0),
                reps_performed=5,
                rpe=7.5,
                rir=3.0,
                rep_level_data_available=False,
                performed_at=started,
                **builder.provenance(source_record_key=f"twice:{label}:set"),
            ),
        )


def _add_observations(builder: HistoryBuilder) -> None:
    """Record athlete-level, set-level, and rep-level observations."""

    builder.add(
        "observation",
        ObservationRecord(
            observation_id=make_id(IdPrefix.OBSERVATION, "soreness", 0),
            athlete_id=builder.athlete_id,
            observed_at=at(8, 8),
            observation_type=ObservationType.SORENESS_SCORE,
            observation_method=ObservationMethod.SELF_REPORT,
            observation_scope=ObservationScope.ATHLETE,
            reporter_role=ReporterRole.ATHLETE,
            numeric_value=6.0,
            **builder.provenance(source_record_key="soreness:0"),
        ),
    )
    # A set whose RPE the source did not capture, so the athlete's report of it is
    # new information rather than a duplicate of the set's own column.
    first_set_id = derive_id(
        derive_id(make_id(IdPrefix.PERFORMED_SESSION, "adv", 0), IdPrefix.PERFORMED_EXERCISE, 0),
        IdPrefix.PERFORMED_SET,
        1,
    )
    builder.add(
        "observation",
        ObservationRecord(
            observation_id=make_id(IdPrefix.OBSERVATION, "rpe-set", 0),
            athlete_id=builder.athlete_id,
            observed_at=at(0, 13),
            observation_type=ObservationType.RPE,
            observation_method=ObservationMethod.SELF_REPORT,
            observation_scope=ObservationScope.SET,
            reporter_role=ReporterRole.ATHLETE,
            numeric_value=8.0,
            parent_set_id=first_set_id,
            **builder.provenance(source_record_key="rpe-set:0"),
        ),
    )
    third_set_id = derive_id(
        derive_id(make_id(IdPrefix.PERFORMED_SESSION, "adv", 2), IdPrefix.PERFORMED_EXERCISE, 0),
        IdPrefix.PERFORMED_SET,
        0,
    )
    builder.add(
        "observation",
        ObservationRecord(
            observation_id=make_id(IdPrefix.OBSERVATION, "rir-rep", 0),
            athlete_id=builder.athlete_id,
            observed_at=at(8, 13),
            observation_type=ObservationType.RIR,
            observation_method=ObservationMethod.COACH_REPORT,
            observation_scope=ObservationScope.REP,
            reporter_role=ReporterRole.COACH,
            numeric_value=0.0,
            parent_set_id=third_set_id,
            parent_rep_id=derive_id(third_set_id, IdPrefix.PERFORMED_REP, 0),
            **builder.provenance(source_record_key="rir-rep:0"),
        ),
    )


def _add_tests_and_velocity(
    builder: HistoryBuilder, exercises: dict[str, ExerciseDefinitionRecord]
) -> None:
    """Record a maximal single, an isometric test, and velocity for one lift."""

    builder.add(
        "performance_test",
        PerformanceTestRecord(
            performance_test_id=make_id(IdPrefix.PERFORMANCE_TEST, "squat", "single"),
            athlete_id=builder.athlete_id,
            observed_at=at(120, 17),
            test_type=TestType.MAXIMAL_SINGLE,
            exercise_id=exercises["squat"].exercise_id,
            protocol_label="single attempt to failure",
            result_metric="load",
            result_raw=kg(160.0),
            result_unit="kg",
            result_normalized=kg(160.0),
            result_unit_normalized="kg",
            **builder.provenance(source_record_key="test:squat:single"),
        ),
    )
    builder.add(
        "performance_test",
        PerformanceTestRecord(
            performance_test_id=make_id(IdPrefix.PERFORMANCE_TEST, "mid-pull", "isometric"),
            athlete_id=builder.athlete_id,
            observed_at=at(150, 17),
            test_type=TestType.ISOMETRIC,
            exercise_id=exercises["deadlift"].exercise_id,
            protocol_label="mid-pull on a force plate",
            result_metric="peak_force",
            result_raw=9200.0,
            result_unit="N",
            result_normalized=9200.0,
            result_unit_normalized="N",
            **builder.provenance(source_record_key="test:mid-pull"),
        ),
    )
    measured_set_id = derive_id(
        derive_id(make_id(IdPrefix.PERFORMED_SESSION, "adv", 0), IdPrefix.PERFORMED_EXERCISE, 0),
        IdPrefix.PERFORMED_SET,
        1,
    )
    builder.add(
        "velocity_observation",
        VelocityObservationRecord(
            velocity_observation_id=make_id(IdPrefix.VELOCITY_OBSERVATION, "bench", 1),
            athlete_id=builder.athlete_id,
            observed_at=at(0, 13),
            performed_set_id=measured_set_id,
            method=VelocityMethod.ACCELEROMETER,
            mean_velocity_mps=0.61,
            peak_velocity_mps=0.79,
            velocity_loss_percent=12.0,
            load_raw=kg(102.5),
            load_unit="kg",
            load_kg=kg(102.5),
            sampling_hz=100.0,
            **builder.provenance(source_record_key="velocity:bench"),
        ),
    )


def _add_simulated_sets(
    builder: HistoryBuilder, exercises: dict[str, ExerciseDefinitionRecord]
) -> None:
    """Record a short synthetic execution history for the PSD-Sim regime."""

    for session_index in range(4):
        started = at(session_index * 7, 17)
        performed_session_id = make_id(
            IdPrefix.PERFORMED_SESSION, "sim", builder.athlete_id, session_index
        )
        builder.add(
            "performed_session",
            PerformedSessionRecord(
                performed_session_id=performed_session_id,
                athlete_id=builder.athlete_id,
                planned_session_id=None,
                started_at=started,
                session_order_index=0,
                session_type=SessionType.TRAINING,
                ended_at=started + timedelta(minutes=50),
                duration_seconds=3000.0,
                is_completed=True,
                performed_at=started,
                **builder.provenance(source_record_key=f"sim-session:{session_index}"),
            ),
        )
        performed_exercise_id = derive_id(performed_session_id, IdPrefix.PERFORMED_EXERCISE, 0)
        builder.add(
            "performed_exercise",
            PerformedExerciseRecord(
                performed_exercise_id=performed_exercise_id,
                performed_session_id=performed_session_id,
                exercise_id=exercises["squat"].exercise_id,
                ordinal=1,
                planned_exercise_id=None,
                performed_at=started,
                **builder.provenance(source_record_key=f"sim-session:{session_index}:ex"),
            ),
        )
        builder.add(
            "performed_set",
            PerformedSetRecord(
                performed_set_id=derive_id(performed_exercise_id, IdPrefix.PERFORMED_SET, 0),
                performed_exercise_id=performed_exercise_id,
                ordinal=1,
                planned_set_id=None,
                set_status=SetStatus.COMPLETED,
                load_raw=kg(120.0 + 2.5 * session_index),
                load_unit="kg",
                load_kg=kg(120.0 + 2.5 * session_index),
                reps_performed=5,
                rpe=7.0 + 0.25 * session_index,
                rir=3.0 - 0.25 * session_index,
                rep_level_data_available=False,
                performed_at=started,
                **builder.provenance(source_record_key=f"sim-session:{session_index}:set"),
            ),
        )


def _add_lineage(builder: HistoryBuilder) -> None:
    """Record dataset-level lineage for the fixture."""

    builder.add(
        "provenance",
        LineageEntry(
            provenance_id=make_id(IdPrefix.PROVENANCE, "adversarial", "ingest"),
            dataset_id="fixture_adversarial",
            transform_name="fixture_ingest",
            transform_version="0.1.0",
            code_commit="0" * 40,
            schema_version="psd-canonical/0.1.0",
            random_seed=None,
            created_at=builder.ingested_at,
            description="Adversarial fixture ingest",
        ),
    )
