"""A clean, well-formed athlete history.

The normal fixture is the counterweight to the adversarial one: a plan, its
execution, athlete-level and session-level observations, body-mass measurements,
equipment transitions, a structured test, one velocity observation, and a meet
with a complete three-attempt sequence per lift.

Everything is consistent: plan links resolve, execution matches the plan, attempt
numbers are unique, and every absent value carries a missingness reason.
"""

from __future__ import annotations

from datetime import timedelta

from pydantic import BaseModel

from psd.schema.identifiers import IdPrefix, derive_id, make_id
from psd.schema.models import (
    BodyMeasurementRecord,
    EquipmentStateRecord,
    ExerciseDefinitionRecord,
    ObservationRecord,
    PerformanceTestRecord,
    PerformedExerciseRecord,
    PerformedSessionRecord,
    PerformedSetRecord,
    PlannedExerciseRecord,
    PlannedSessionRecord,
    PlannedSetRecord,
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
    LiftType,
    ObservationMethod,
    ObservationScope,
    ObservationType,
    PrescriptionBasis,
    ReporterRole,
    SessionStatus,
    SessionType,
    SetStatus,
    TestType,
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
)

__all__ = ("SOURCE_IDS", "normal_history_records")

SOURCE_IDS = SourceIds()

MEET_ATTEMPTS: tuple[tuple[LiftType, int, float, AttemptResult], ...] = (
    (LiftType.SQUAT, 1, 170.0, AttemptResult.GOOD_LIFT),
    (LiftType.SQUAT, 2, 180.0, AttemptResult.GOOD_LIFT),
    (LiftType.SQUAT, 3, 187.5, AttemptResult.BAD_LIFT),
    (LiftType.BENCH, 1, 115.0, AttemptResult.GOOD_LIFT),
    (LiftType.BENCH, 2, 122.5, AttemptResult.GOOD_LIFT),
    (LiftType.BENCH, 3, 125.0, AttemptResult.GOOD_LIFT),
    (LiftType.DEADLIFT, 1, 210.0, AttemptResult.GOOD_LIFT),
    (LiftType.DEADLIFT, 2, 225.0, AttemptResult.GOOD_LIFT),
    (LiftType.DEADLIFT, 3, 232.5, AttemptResult.GOOD_LIFT),
)

MEET_REPORTED: tuple[tuple[CompetitionResultKind, float, bool], ...] = (
    (CompetitionResultKind.SQUAT_BEST, 180.0, False),
    (CompetitionResultKind.BENCH_BEST, 125.0, False),
    (CompetitionResultKind.DEADLIFT_BEST, 232.5, False),
    (CompetitionResultKind.TOTAL, 537.5, False),
    (CompetitionResultKind.PLACING, 2.0, True),
)

#: ``(exercise short name, top load, reps, set count)`` per training day.
DAY_PLAN: tuple[tuple[str, float, int, int], ...] = (
    ("squat", 140.0, 5, 3),
    ("bench", 100.0, 5, 3),
    ("deadlift", 170.0, 5, 3),
)

SESSION_COUNT = 8
SESSION_STRIDE = 3


def normal_history_records() -> dict[str, list[BaseModel]]:
    """Return a clean, self-consistent athlete history.

    Returns:
        Table name to validated records, covering every canonical table.
    """
    builder = HistoryBuilder(athlete_id=ATHLETE_ID, source_id=SOURCE_IDS.training)
    add_source_records(builder, ids=SOURCE_IDS)
    add_athlete(builder)
    exercises = add_exercises(builder)
    program = add_program(builder, versions=(("v1", 0),))
    version_id = program["v1"].program_version_id
    _add_equipment(builder)
    _add_body_mass(builder)
    _add_planned_block(builder, exercises, version_id)
    _add_performed_block(builder, exercises)
    _add_observations(builder)
    _add_test_and_velocity(builder, exercises)
    add_competition(
        builder,
        competition_id=make_id(IdPrefix.COMPETITION, "regional-open", "2021-03"),
        athlete_id=ATHLETE_ID,
        offset=70,
        attempts=MEET_ATTEMPTS,
        source_id=SOURCE_IDS.competition,
        reported=MEET_REPORTED,
    )
    return builder.build()


def _add_equipment(builder: HistoryBuilder) -> None:
    """Record a shoe transition and a single belt."""

    for item, identifier, start, end in (
        (EquipmentItem.SHOES, "old-shoes", 0, 120),
        (EquipmentItem.SHOES, "new-shoes", 120, None),
        (EquipmentItem.BELT, "belt-13mm", 0, None),
    ):
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


def _add_body_mass(builder: HistoryBuilder) -> None:
    """Record a weekly body-mass series."""

    for week in range(8):
        mass = kg(round(91.0 + 0.1 * week, 1))
        builder.add(
            "body_measurement",
            BodyMeasurementRecord(
                body_measurement_id=make_id(IdPrefix.BODY_MEASUREMENT, "morning", week),
                athlete_id=builder.athlete_id,
                measured_at=at(7 * week, 7),
                measurement_type=BodyMeasurementType.BODY_MASS,
                event_time_precision=EventTimePrecision.MINUTE,
                measurement_context=BodyMassContext.MORNING_FASTED,
                method=BodyMeasurementMethod.BATHROOM_SCALE,
                raw_value=mass,
                raw_unit="kg",
                value_normalized=mass,
                unit_normalized="kg",
                **builder.provenance(source_record_key=f"weigh-in:{week}"),
            ),
        )


def _add_planned_block(
    builder: HistoryBuilder, exercises: dict[str, ExerciseDefinitionRecord], version_id: str
) -> None:
    """Prescribe eight sessions of three exercises."""

    for session_index in range(SESSION_COUNT):
        planned_session_id = _planned_session_id(session_index)
        builder.add(
            "planned_session",
            PlannedSessionRecord(
                planned_session_id=planned_session_id,
                athlete_id=builder.athlete_id,
                program_version_id=version_id,
                scheduled_at=at(session_index * SESSION_STRIDE, 9),
                session_order_index=0,
                session_status=SessionStatus.PLANNED,
                template_label="day_a",
                **builder.provenance(source_record_key=f"plan:{session_index}"),
            ),
        )
        for exercise_index, (name, load, reps, sets) in enumerate(DAY_PLAN):
            planned_exercise_id = derive_id(
                planned_session_id, IdPrefix.PLANNED_EXERCISE, exercise_index
            )
            builder.add(
                "planned_exercise",
                PlannedExerciseRecord(
                    planned_exercise_id=planned_exercise_id,
                    planned_session_id=planned_session_id,
                    exercise_id=exercises[name].exercise_id,
                    ordinal=exercise_index + 1,
                    **builder.provenance(source_record_key=f"plan:{session_index}:{name}"),
                ),
            )
            for set_index in range(sets):
                target = kg(load + 2.5 * session_index + 2.5 * set_index)
                builder.add(
                    "planned_set",
                    PlannedSetRecord(
                        planned_set_id=derive_id(
                            planned_exercise_id, IdPrefix.PLANNED_SET, set_index
                        ),
                        planned_exercise_id=planned_exercise_id,
                        ordinal=set_index + 1,
                        target_reps=reps,
                        target_load_raw=target,
                        target_load_unit="kg",
                        target_load_kg=target,
                        prescription_basis=PrescriptionBasis.ABSOLUTE_LOAD,
                        rest_target_seconds=180.0,
                        **builder.provenance(
                            source_record_key=f"plan:{session_index}:{name}:{set_index}"
                        ),
                    ),
                )


def _planned_session_id(session_index: int) -> str:
    """Return the planned-session id for a session index."""
    return make_id(IdPrefix.PLANNED_SESSION, "normal", session_index)


def _performed_session_id(session_index: int) -> str:
    """Return the performed-session id for a session index."""
    return make_id(IdPrefix.PERFORMED_SESSION, "normal", session_index)


def _add_performed_block(
    builder: HistoryBuilder, exercises: dict[str, ExerciseDefinitionRecord]
) -> None:
    """Execute eight sessions matching the prescription."""

    for session_index in range(SESSION_COUNT):
        started = at(session_index * SESSION_STRIDE, 11)
        performed_session_id = _performed_session_id(session_index)
        builder.add(
            "performed_session",
            PerformedSessionRecord(
                performed_session_id=performed_session_id,
                athlete_id=builder.athlete_id,
                planned_session_id=_planned_session_id(session_index),
                started_at=started,
                session_order_index=0,
                session_type=SessionType.TRAINING,
                ended_at=started + timedelta(hours=1),
                duration_seconds=3600.0,
                is_completed=True,
                performed_at=started,
                **builder.provenance(source_record_key=f"session:{session_index}"),
            ),
        )
        for exercise_index, (name, load, reps, sets) in enumerate(DAY_PLAN):
            planned_exercise_id = derive_id(
                _planned_session_id(session_index), IdPrefix.PLANNED_EXERCISE, exercise_index
            )
            performed_exercise_id = derive_id(
                performed_session_id, IdPrefix.PERFORMED_EXERCISE, exercise_index
            )
            builder.add(
                "performed_exercise",
                PerformedExerciseRecord(
                    performed_exercise_id=performed_exercise_id,
                    performed_session_id=performed_session_id,
                    exercise_id=exercises[name].exercise_id,
                    ordinal=exercise_index + 1,
                    planned_exercise_id=planned_exercise_id,
                    performed_at=started,
                    **builder.provenance(source_record_key=f"session:{session_index}:{name}"),
                ),
            )
            for set_index in range(sets):
                performed = kg(load + 2.5 * session_index + 2.5 * set_index)
                builder.add(
                    "performed_set",
                    PerformedSetRecord(
                        performed_set_id=derive_id(
                            performed_exercise_id, IdPrefix.PERFORMED_SET, set_index
                        ),
                        performed_exercise_id=performed_exercise_id,
                        ordinal=set_index + 1,
                        planned_set_id=derive_id(
                            planned_exercise_id, IdPrefix.PLANNED_SET, set_index
                        ),
                        set_status=SetStatus.COMPLETED,
                        load_raw=performed,
                        load_unit="kg",
                        load_kg=performed,
                        reps_performed=reps,
                        rpe=round(7.0 + 0.1 * session_index, 1),
                        rir=round(3.0 - 0.1 * session_index, 1),
                        rep_level_data_available=False,
                        performed_at=started,
                        **builder.provenance(
                            source_record_key=f"session:{session_index}:{name}:{set_index}"
                        ),
                    ),
                )


def _add_observations(builder: HistoryBuilder) -> None:
    """Record session-level readiness and athlete-level sleep."""

    for session_index in range(SESSION_COUNT):
        builder.add(
            "observation",
            ObservationRecord(
                observation_id=make_id(IdPrefix.OBSERVATION, "readiness", session_index),
                athlete_id=builder.athlete_id,
                observed_at=at(session_index * SESSION_STRIDE, 10),
                observation_type=ObservationType.READINESS_SCORE,
                observation_method=ObservationMethod.SELF_REPORT,
                observation_scope=ObservationScope.SESSION,
                reporter_role=ReporterRole.ATHLETE,
                instrument="readiness-1-10",
                numeric_value=round(6.0 + 0.2 * session_index, 1),
                parent_session_id=_performed_session_id(session_index),
                **builder.provenance(source_record_key=f"readiness:{session_index}"),
            ),
        )
    for week in range(4):
        builder.add(
            "observation",
            ObservationRecord(
                observation_id=make_id(IdPrefix.OBSERVATION, "sleep", week),
                athlete_id=builder.athlete_id,
                observed_at=at(7 * week, 7),
                observation_type=ObservationType.SLEEP_DURATION,
                observation_method=ObservationMethod.SELF_REPORT,
                observation_scope=ObservationScope.ATHLETE,
                reporter_role=ReporterRole.ATHLETE,
                numeric_value=round(6.5 + 0.1 * week, 2),
                unit="h",
                **builder.provenance(source_record_key=f"sleep:{week}"),
            ),
        )


def _add_test_and_velocity(
    builder: HistoryBuilder, exercises: dict[str, ExerciseDefinitionRecord]
) -> None:
    """Record one structured test and one velocity observation."""

    builder.add(
        "performance_test",
        PerformanceTestRecord(
            performance_test_id=make_id(IdPrefix.PERFORMANCE_TEST, "bench", "e1rm"),
            athlete_id=builder.athlete_id,
            observed_at=at(20, 17),
            test_type=TestType.ESTIMATED_ONE_RM_FROM_WORK_SETS,
            exercise_id=exercises["bench"].exercise_id,
            protocol_label="5x5 e1RM at RPE 8",
            result_metric="estimated_one_rm",
            result_raw=kg(135.0),
            result_unit="kg",
            result_normalized=kg(135.0),
            result_unit_normalized="kg",
            **builder.provenance(source_record_key="test:bench"),
        ),
    )
    performed_set_id = derive_id(
        derive_id(_performed_session_id(2), IdPrefix.PERFORMED_EXERCISE, 1),
        IdPrefix.PERFORMED_SET,
        0,
    )
    builder.add(
        "velocity_observation",
        VelocityObservationRecord(
            velocity_observation_id=make_id(IdPrefix.VELOCITY_OBSERVATION, "bench", 0),
            athlete_id=builder.athlete_id,
            observed_at=at(6, 18),
            performed_set_id=performed_set_id,
            method=VelocityMethod.LINEAR_POSITION_TRANSDUCER,
            mean_velocity_mps=0.52,
            peak_velocity_mps=0.68,
            velocity_loss_percent=18.4,
            load_raw=kg(105.0),
            load_unit="kg",
            load_kg=kg(105.0),
            sampling_hz=100.0,
            **builder.provenance(source_record_key="velocity:bench"),
        ),
    )
