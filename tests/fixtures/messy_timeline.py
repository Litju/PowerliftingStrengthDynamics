"""The messy multi-year athlete timeline -- the RES-235 exit gate.

The exit gate asks for a messy, multi-year athlete timeline containing
programming changes, training execution, missing fields, tests, and meets that can
be ingested, validated, serialized, and re-read without material semantic or
provenance loss. This fixture is that timeline.

Shape of the history
--------------------

* **3.5 years** (``0..1290`` days) with real calendar structure: four-week
  blocks, deload weeks, and taper weeks before meets.
* **Four program versions**, each superseding the previous one, plus mid-block
  autoregulated modifications stamped with ``known_at`` *after* the session they
  affect was scheduled.
* **Gaps** -- a six-week injury layoff, two holiday breaks, and a short
  interruption where training simply stopped being logged.
* **Uneven logging** -- some sessions were performed without a prescription, some
  sets have no RPE, some sessions have no end time, some sets were never recorded,
  and velocity exists for only a subset of blocks.
* **Six meets** with three-attempt sequences, one with a missed third attempt and
  one with an explicit ``no_attempt`` slot, plus reported bests and totals.
* **Monthly body mass**, with one month reported in pounds.
* **Quarterly tests**: estimated one-rep maxima from work sets, plus maximal
  singles.

Everything is derived from the day index, so the fixture is a pure function of its
constants and produces identical artifacts on every run.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from pydantic import BaseModel

from psd.provenance.manifest import LineageEntry
from psd.schema.identifiers import IdPrefix, derive_id, make_id
from psd.schema.models import (
    BodyMeasurementRecord,
    EquipmentStateRecord,
    ExerciseDefinitionRecord,
    PerformanceTestRecord,
    PerformedExerciseRecord,
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
    LiftType,
    MissingnessReason,
    NotPerformedReason,
    PrescriptionBasis,
    ProgramModificationKind,
    QualityFlag,
    SessionStatus,
    SessionType,
    SetStatus,
    TestType,
    ValueEncoding,
    VelocityMethod,
)
from tests.fixtures.builders import (
    ATHLETE_ID,
    BASE_INSTANT,
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
    program_version_of,
)

__all__ = ("MEETS", "TOTAL_DAYS", "messy_timeline_records", "messy_timeline_span")

SOURCE_IDS = SourceIds()

TOTAL_DAYS = 1290
BLOCK_LENGTH = 28

MEETS: tuple[tuple[int, str], ...] = (
    (150, "regional-open-spring"),
    (430, "nationals-summer"),
    (700, "regional-open-autumn"),
    (905, "invitational-winter"),
    (1120, "regional-open-spring-final"),
    (1265, "championship-finals"),
)

#: Day ranges where the athlete was not training at all.
GAPS: tuple[tuple[int, int], ...] = ((300, 342), (620, 634), (990, 1004))

#: Blocks in which velocity was measured.
VELOCITY_BLOCKS: frozenset[int] = frozenset({1, 3, 5, 8})

#: ``(exercise short name, canonical lift name, base load)`` per training day.
LIFTS: tuple[tuple[LiftType, str, float], ...] = (
    (LiftType.SQUAT, "squat", 150.0),
    (LiftType.BENCH, "bench", 105.0),
    (LiftType.DEADLIFT, "deadlift", 185.0),
)

SESSIONS_PER_WEEK = 3
TAPER_WINDOW_DAYS = 10


def _in_gap(offset: int) -> bool:
    """Return whether *offset* falls inside a layoff."""
    return any(start <= offset < end for start, end in GAPS)


def _is_deload(offset: int) -> bool:
    """Return whether *offset* falls in a deload week."""
    return (offset // 7) % 4 == 3


def _taper_meet(offset: int) -> str | None:
    """Return the meet name when *offset* is inside a taper window."""
    for meet_offset, name in MEETS:
        if 0 <= meet_offset - offset <= TAPER_WINDOW_DAYS:
            return name
    return None


def _skip_reason(offset: int, day_index: int) -> str | None:
    """Return why a planned session was skipped, if it was."""
    selector = (offset + day_index) % 23
    if 300 <= offset < 342:
        return "injury"
    if selector == 3:
        return "scheduled_rest"
    if selector == 7:
        return "time_constraint"
    if selector == 11:
        return "travel"
    return None


def _has_modification(offset: int, day_index: int) -> bool:
    """Return whether this session received an autoregulated change."""
    return (offset + day_index) % 13 == 0


def _target_load(base: float, block: int, offset: int, deload: bool, tapering: bool) -> float:
    """Return the prescribed top load in kilograms for a session."""
    load = base + 2.5 * (block % 8) + 1.25 * ((offset // 7) % 4)
    if deload:
        load -= 10.0
    if tapering:
        load -= 7.5
    return round(load, 1)


def messy_timeline_records() -> dict[str, list[BaseModel]]:
    """Return the multi-year messy athlete history.

    Returns:
        Table name to validated records, covering every canonical table.
    """
    builder = HistoryBuilder(athlete_id=ATHLETE_ID, source_id=SOURCE_IDS.training)
    add_source_records(builder, ids=SOURCE_IDS)
    add_athlete(builder)
    exercises = add_exercises(builder)
    program = add_program(builder, versions=(("v1", 0), ("v2", 280), ("v3", 560), ("v4", 980)))
    _add_equipment(builder)
    _add_lineage(builder)
    _add_body_mass(builder)
    for week_start in range(0, TOTAL_DAYS, 7):
        _add_week(builder, exercises, program, week_start)
    _add_tests(builder, exercises)
    _add_meets(builder)
    return builder.build()


def _add_equipment(builder: HistoryBuilder) -> None:
    """Record equipment transitions across the history."""

    schedule: tuple[tuple[Any, str, int, int | None], ...] = (
        (EquipmentItem.BAR, "bar-2017", 0, 280),
        (EquipmentItem.BAR, "bar-2021", 280, None),
        (EquipmentItem.SHOES, "shoes-heel-1", 0, 430),
        (EquipmentItem.SHOES, "shoes-flat", 430, None),
        (EquipmentItem.BELT, "belt-10mm", 0, 700),
        (EquipmentItem.BELT, "belt-13mm", 700, None),
        (EquipmentItem.HEEL_RAISERS, "no-raisers", 0, 430),
        (EquipmentItem.HEEL_RAISERS, "no-raisers", 430, None),
        (EquipmentItem.KNEE_SLEEVES, "sleeves-7mm", 560, None),
    )
    for index, (item, identifier, start, end) in enumerate(schedule):
        builder.add(
            "equipment_state",
            EquipmentStateRecord(
                equipment_state_id=make_id(IdPrefix.EQUIPMENT_STATE, item.value, identifier, index),
                athlete_id=builder.athlete_id,
                equipment_item=item,
                effective_from=at(start),
                effective_to=None if end is None else at(end),
                identifier=identifier,
                **builder.provenance(source_record_key=f"{item.value}:{identifier}:{index}"),
            ),
        )


def _add_lineage(builder: HistoryBuilder) -> None:
    """Record the dataset-level lineage of the fixture."""

    builder.add(
        "provenance",
        LineageEntry(
            provenance_id=make_id(IdPrefix.PROVENANCE, "messy", "ingest"),
            dataset_id="fixture_messy_timeline",
            transform_name="fixture_messy_ingest",
            transform_version="0.1.0",
            code_commit="0" * 40,
            schema_version="psd-canonical/0.1.0",
            random_seed=None,
            created_at=builder.ingested_at,
            description="Multi-year messy timeline used for the RES-235 exit gate",
        ),
    )


def _add_body_mass(builder: HistoryBuilder) -> None:
    """Record monthly body mass, with one month reported in pounds."""

    for month in range(TOTAL_DAYS // 30):
        offset = 30 * month + 2
        use_pounds = month == 6
        kilograms = kg(round(93.5 + 0.12 * month, 1))
        raw, normalized = lb(207.5 + 0.4 * month) if use_pounds else (kilograms, kilograms)
        builder.add(
            "body_measurement",
            BodyMeasurementRecord(
                body_measurement_id=make_id(IdPrefix.BODY_MEASUREMENT, "monthly", month),
                athlete_id=builder.athlete_id,
                measured_at=at(offset, 7),
                measurement_type=BodyMeasurementType.BODY_MASS,
                event_time_precision=EventTimePrecision.MINUTE,
                measurement_context=BodyMassContext.MORNING_FASTED,
                method=BodyMeasurementMethod.BATHROOM_SCALE,
                raw_value=raw,
                raw_unit="lb" if use_pounds else "kg",
                value_normalized=normalized,
                unit_normalized="kg",
                **builder.provenance(source_record_key=f"monthly:{month}"),
            ),
        )


def _add_week(
    builder: HistoryBuilder,
    exercises: dict[str, ExerciseDefinitionRecord],
    program: dict[str, ProgramVersionRecord],
    week_start: int,
) -> None:
    """Plan and execute the sessions of one training week."""

    block = week_start // BLOCK_LENGTH
    for day_index, (name, _lift, base) in enumerate(LIFTS):
        offset = week_start + day_index
        if _in_gap(offset):
            continue
        deload = _is_deload(offset)
        tapering = _taper_meet(offset) is not None
        skip = _skip_reason(offset, day_index)
        version = program_version_of(program, offset)
        planned_session_id = make_id(IdPrefix.PLANNED_SESSION, "messy", offset, day_index)
        modified = _has_modification(offset, day_index)

        builder.add(
            "planned_session",
            PlannedSessionRecord(
                planned_session_id=planned_session_id,
                athlete_id=builder.athlete_id,
                program_version_id=version.program_version_id,
                scheduled_at=at(offset, 9),
                session_order_index=day_index,
                session_status=SessionStatus.SKIPPED if skip else SessionStatus.PLANNED,
                not_performed_reason=NotPerformedReason(skip) if skip else None,
                template_label="taper" if tapering else ("deload" if deload else "accumulation"),
                modified_at=at(offset - 1, 20) if modified else None,
                **builder.provenance(source_record_key=f"messy-plan:{offset}:{day_index}"),
            ),
        )
        if modified:
            builder.add(
                "program_modification",
                ProgramModificationRecord(
                    program_modification_id=make_id(
                        IdPrefix.PROGRAM_MODIFICATION, offset, day_index
                    ),
                    athlete_id=builder.athlete_id,
                    planned_session_id=planned_session_id,
                    program_version_id=version.program_version_id,
                    known_at=at(offset - 1, 21),
                    change_kind=ProgramModificationKind.LOAD_CHANGE,
                    field_changed="target_load_kg",
                    previous_value=f"{base:.1f}",
                    new_value=f"{base - 5.0:.1f}",
                    value_encoding=ValueEncoding.NUMBER,
                    rationale="Bar speed below target; reduced top sets.",
                    **builder.provenance(source_record_key=f"messy-mod:{offset}:{day_index}"),
                ),
            )
        if skip:
            continue

        top_load = _target_load(base, block, offset, deload, tapering)
        planned_exercise_id = derive_id(planned_session_id, IdPrefix.PLANNED_EXERCISE, 0)
        builder.add(
            "planned_exercise",
            PlannedExerciseRecord(
                planned_exercise_id=planned_exercise_id,
                planned_session_id=planned_session_id,
                exercise_id=exercises[name].exercise_id,
                ordinal=1,
                **builder.provenance(source_record_key=f"messy-plan:{offset}:{day_index}:ex"),
            ),
        )
        set_count = 2 if tapering else (3 if deload else 4)
        for set_index in range(set_count):
            planned_load = top_load + 2.5 * set_index
            builder.add(
                "planned_set",
                PlannedSetRecord(
                    planned_set_id=derive_id(planned_exercise_id, IdPrefix.PLANNED_SET, set_index),
                    planned_exercise_id=planned_exercise_id,
                    ordinal=set_index + 1,
                    target_reps=3 if tapering else 5,
                    target_load_raw=planned_load,
                    target_load_unit="kg",
                    target_load_kg=planned_load,
                    prescription_basis=PrescriptionBasis.PERCENT_ONE_RM,
                    target_percent_one_rm=round(
                        70.0 + 2.5 * (block % 8) - (10.0 if deload else 0.0), 1
                    ),
                    rest_target_seconds=120.0 if tapering else 240.0,
                    **builder.provenance(
                        source_record_key=f"messy-plan:{offset}:{day_index}:{set_index}"
                    ),
                ),
            )

        # Roughly one session in seventeen was performed without a prescription
        # being captured by the source.
        unplanned = (offset + day_index) % 17 == 0
        performed_session_id = make_id(IdPrefix.PERFORMED_SESSION, "messy", offset, day_index)
        started = at(offset, 11)
        open_ended = (offset + day_index) % 29 == 0
        builder.add(
            "performed_session",
            PerformedSessionRecord(
                performed_session_id=performed_session_id,
                athlete_id=builder.athlete_id,
                planned_session_id=None if unplanned else planned_session_id,
                started_at=started,
                session_order_index=day_index,
                session_type=SessionType.TRAINING,
                ended_at=None if open_ended else started + timedelta(minutes=55 + 10 * day_index),
                duration_seconds=None if open_ended else float(3300 + 600 * day_index),
                is_completed=None if open_ended else True,
                performed_at=started,
                **builder.provenance(
                    source_record_key=f"messy-session:{offset}:{day_index}",
                    missingness_reason=MissingnessReason.NOT_RECORDED_IN_SOURCE
                    if open_ended
                    else None,
                ),
            ),
        )
        performed_exercise_id = derive_id(performed_session_id, IdPrefix.PERFORMED_EXERCISE, 0)
        builder.add(
            "performed_exercise",
            PerformedExerciseRecord(
                performed_exercise_id=performed_exercise_id,
                performed_session_id=performed_session_id,
                exercise_id=exercises[name].exercise_id,
                ordinal=1,
                planned_exercise_id=None if unplanned else planned_exercise_id,
                performed_at=started,
                **builder.provenance(source_record_key=f"messy-session:{offset}:{day_index}:ex"),
            ),
        )
        recorded_sets = set_count if (offset + day_index) % 11 else set_count - 1
        first_set_id: str | None = None
        for set_index in range(recorded_sets):
            performed_set_id = derive_id(performed_exercise_id, IdPrefix.PERFORMED_SET, set_index)
            if set_index == 0:
                first_set_id = performed_set_id
            load = top_load + 2.5 * set_index
            has_rpe = (offset + day_index + set_index) % 5 != 0
            edited = (offset + day_index + set_index) % 53 == 0
            # RPE climbs across the block and RIR falls with it, so the pair keeps
            # the usual RPE + RIR == 10 relationship PSD cross-checks.
            effort = round(6.5 + 0.06 * (offset % 40) + 0.1 * set_index, 1)
            builder.add(
                "performed_set",
                PerformedSetRecord(
                    performed_set_id=performed_set_id,
                    performed_exercise_id=performed_exercise_id,
                    ordinal=set_index + 1,
                    planned_set_id=None
                    if unplanned or set_index > 0
                    else derive_id(planned_exercise_id, IdPrefix.PLANNED_SET, 0),
                    set_status=SetStatus.COMPLETED,
                    load_raw=load,
                    load_unit="kg",
                    load_kg=load,
                    reps_performed=3 if tapering else 5,
                    rpe=effort if has_rpe else None,
                    rir=round(10.0 - effort, 1) if has_rpe else None,
                    rep_level_data_available=False,
                    performed_at=started,
                    modified_at=at(offset + 2, 8) if edited else None,
                    **builder.provenance(
                        source_record_key=f"messy-set:{offset}:{day_index}:{set_index}",
                        quality_flags=(QualityFlag.EDITED_AFTER_THE_FACT,) if edited else (),
                        missingness_reason=None
                        if has_rpe
                        else MissingnessReason.NOT_RECORDED_IN_SOURCE,
                    ),
                ),
            )
        if block in VELOCITY_BLOCKS and first_set_id is not None:
            _add_velocity(builder, first_set_id, offset, day_index)


def _add_velocity(
    builder: HistoryBuilder, performed_set_id: str, offset: int, day_index: int
) -> None:
    """Record one velocity measurement for a session."""

    builder.add(
        "velocity_observation",
        VelocityObservationRecord(
            velocity_observation_id=make_id(IdPrefix.VELOCITY_OBSERVATION, offset, day_index),
            athlete_id=builder.athlete_id,
            observed_at=at(offset, 11),
            performed_set_id=performed_set_id,
            method=VelocityMethod.LINEAR_POSITION_TRANSDUCER,
            mean_velocity_mps=round(0.45 + 0.002 * (offset % 100), 3),
            peak_velocity_mps=round(0.6 + 0.002 * (offset % 100), 3),
            velocity_loss_percent=round(18.0 - 0.05 * (offset % 20), 1),
            load_raw=kg(100.0 + 1.25 * (offset % 30)),
            load_unit="kg",
            load_kg=kg(100.0 + 1.25 * (offset % 30)),
            sampling_hz=100.0,
            **builder.provenance(source_record_key=f"messy-velocity:{offset}:{day_index}"),
        ),
    )


def _add_tests(builder: HistoryBuilder, exercises: dict[str, ExerciseDefinitionRecord]) -> None:
    """Record quarterly estimated-max tests and three maximal singles."""

    for index in range(14):
        offset = 90 * index
        if offset >= TOTAL_DAYS:
            break
        for lift_index, (name, _lift, base) in enumerate(LIFTS[:2]):
            result = round(base + 12.5 * index + 2.5 * lift_index, 1)
            builder.add(
                "performance_test",
                PerformanceTestRecord(
                    performance_test_id=make_id(
                        IdPrefix.PERFORMANCE_TEST, "quarterly", index, name
                    ),
                    athlete_id=builder.athlete_id,
                    observed_at=at(offset, 17),
                    test_type=TestType.ESTIMATED_ONE_RM_FROM_WORK_SETS,
                    exercise_id=exercises[name].exercise_id,
                    protocol_label="top work set, e1RM formula",
                    result_metric="estimated_one_rm",
                    result_raw=kg(result),
                    result_unit="kg",
                    result_normalized=kg(result),
                    result_unit_normalized="kg",
                    **builder.provenance(source_record_key=f"quarterly:{index}:{name}"),
                ),
            )
    for name, _lift, base in LIFTS:
        offset = MEETS[-1][0] + 7
        builder.add(
            "performance_test",
            PerformanceTestRecord(
                performance_test_id=make_id(IdPrefix.PERFORMANCE_TEST, "max-single", name),
                athlete_id=builder.athlete_id,
                observed_at=at(offset, 17),
                test_type=TestType.MAXIMAL_SINGLE,
                exercise_id=exercises[name].exercise_id,
                protocol_label="single attempt to failure",
                result_metric="load",
                result_raw=kg(base + 45.0),
                result_unit="kg",
                result_normalized=kg(base + 45.0),
                result_unit_normalized="kg",
                **builder.provenance(source_record_key=f"max-single:{name}"),
            ),
        )


def _best_kind_of(lift: LiftType) -> CompetitionResultKind:
    """Map a lift to the reported best that a meet result carries for it."""
    return {
        LiftType.SQUAT: CompetitionResultKind.SQUAT_BEST,
        LiftType.BENCH: CompetitionResultKind.BENCH_BEST,
        LiftType.DEADLIFT: CompetitionResultKind.DEADLIFT_BEST,
    }[lift]


def _add_meets(builder: HistoryBuilder) -> None:
    """Record six meets with attempt sequences and reported results."""
    for index, (offset, name) in enumerate(MEETS):
        attempts: list[tuple[LiftType, int, float | None, AttemptResult]] = []
        bests: list[tuple[CompetitionResultKind, float, bool]] = []
        total = 0.0
        for lift, _canonical, base in LIFTS:
            opener = base + 20.0 + 5.0 * index
            second = opener + 7.5
            third = None if index == 2 else opener + 10.0
            attempts.append((lift, 1, opener, AttemptResult.GOOD_LIFT))
            attempts.append((lift, 2, second, AttemptResult.GOOD_LIFT))
            if lift is LiftType.DEADLIFT and index == 4:
                # The third deadlift slot was recorded but never taken, so it
                # carries no load and occupies the same slot a bad lift would.
                attempts.append((lift, 3, None, AttemptResult.NO_ATTEMPT))
            elif third is not None:
                attempts.append((lift, 3, third, AttemptResult.BAD_LIFT))
            bests.append((_best_kind_of(lift), second, False))
            total += second
        bests.append((CompetitionResultKind.TOTAL, round(total, 1), False))
        add_competition(
            builder,
            competition_id=make_id(IdPrefix.COMPETITION, name, offset),
            athlete_id=builder.athlete_id,
            offset=offset,
            attempts=tuple(attempts),
            source_id=SOURCE_IDS.competition,
            reported=tuple(bests),
        )


def messy_timeline_span() -> tuple[datetime, datetime]:
    """Return the first and last instants covered by the messy timeline."""
    return BASE_INSTANT, BASE_INSTANT + timedelta(days=TOTAL_DAYS)
