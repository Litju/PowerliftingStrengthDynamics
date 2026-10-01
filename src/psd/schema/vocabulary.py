"""Controlled vocabularies for the canonical schema.

These vocabularies constrain *how a value was recorded*, never what it means
physiologically. That distinction is deliberate:

* ``ObservationType`` names reporting instruments (``rpe``, ``sleep_duration``),
  not latent physiological constructs. Adding a member must never require the
  benchmark to assume the underlying physiology exists or is identifiable.
* ``QualityFlag`` marks trust and provenance of a value.
* ``MissingnessReason`` makes absence explicit so that ``null`` is never silently
  reinterpreted, and never confused with a recorded zero.

Vocabularies are stored as plain strings in Arrow/Parquet so that artifacts stay
readable without decoding dictionaries, and are validated on the way in.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Final

__all__ = (
    "VOCABULARIES",
    "AttemptOrderBasis",
    "AttemptResult",
    "BodyMassContext",
    "BodyMeasurementMethod",
    "BodyMeasurementType",
    "CompetitionResultKind",
    "EquipmentClass",
    "EquipmentItem",
    "EventTimePrecision",
    "IdentityLinkMethod",
    "IdentityStatus",
    "ImplementType",
    "Laterality",
    "LiftType",
    "MissingnessReason",
    "NotPerformedReason",
    "ObservationMethod",
    "ObservationScope",
    "ObservationType",
    "ParentLift",
    "PrescriptionBasis",
    "ProgramModificationKind",
    "QualityFlag",
    "RepStatus",
    "ReporterRole",
    "SessionStatus",
    "SessionType",
    "SetStatus",
    "SexCategory",
    "SpecificityLevel",
    "TestType",
    "ValueEncoding",
    "VelocityMethod",
    "vocabulary_members",
)


class MissingnessReason(StrEnum):
    """Why a value is absent.

    ``null`` always means "not recorded here", never "recorded as zero".
    """

    NOT_RECORDED_IN_SOURCE = "not_recorded_in_source"
    NOT_APPLICABLE = "not_applicable"
    NOT_YET_RECORDED = "not_yet_recorded"
    WITHHELD_BY_SOURCE = "withheld_by_source"
    REDACTED_FOR_PRIVACY = "redacted_for_privacy"
    AMBIGUOUS_SOURCE_VALUE = "ambiguous_source_value"
    SOURCE_ERROR = "source_error"
    UNKNOWN = "unknown"


class QualityFlag(StrEnum):
    """Trust and provenance markers for a value.

    Flags are stored as a sorted, de-duplicated list so that two equal flag sets
    serialize to identical bytes regardless of insertion order.
    """

    VERIFIED = "verified"
    SELF_REPORTED = "self_reported"
    THIRD_PARTY_REPORTED = "third_party_reported"
    ESTIMATED = "estimated"
    ROUNDED_TO_INCREMENT = "rounded_to_increment"
    DEVICE_MEASURED = "device_measured"
    LAB_MEASURED = "lab_measured"
    IMPORTED_WITHOUT_SOURCE_TIME = "imported_without_source_time"
    AMBIGUOUS_IDENTITY = "ambiguous_identity"
    DISPUTED = "disputed"
    OUT_OF_PLAUSIBLE_RANGE = "out_of_plausible_range"
    EDITED_AFTER_THE_FACT = "edited_after_the_fact"
    UNVERIFIED = "unverified"


class SessionStatus(StrEnum):
    """Lifecycle state of a planned session."""

    PLANNED = "planned"
    COMPLETED = "completed"
    PARTIALLY_COMPLETED = "partially_completed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"


class NotPerformedReason(StrEnum):
    """Why a planned session was not performed.

    A skipped session is recorded as a *planned* fact. PSD never manufactures a
    performed record, and never treats a skip as a zero-load session.
    """

    INJURY = "injury"
    ILLNESS = "illness"
    TIME_CONSTRAINT = "time_constraint"
    TRAVEL = "travel"
    SCHEDULED_REST = "scheduled_rest"
    REPLACED_BY_OTHER_SESSION = "replaced_by_other_session"
    PROGRAM_CANCELLED = "program_cancelled"
    UNKNOWN = "unknown"


class SessionType(StrEnum):
    """What kind of session a performed session represents."""

    TRAINING = "training"
    TESTING = "testing"
    COMPETITION = "competition"
    MEET_WARMUP = "meet_warmup"
    OTHER = "other"
    UNKNOWN = "unknown"


class SetStatus(StrEnum):
    """Outcome state of a performed set."""

    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"
    ABORTED = "aborted"
    UNKNOWN = "unknown"


class RepStatus(StrEnum):
    """Outcome state of a single performed repetition."""

    COMPLETED = "completed"
    FAILED = "failed"
    ASSISTED = "assisted"
    ABANDONED = "abandoned"
    UNKNOWN = "unknown"


class PrescriptionBasis(StrEnum):
    """How a prescribed target was anchored.

    ``AUTOREGULATED`` and ``UNKNOWN`` matter for leakage analysis: an
    autoregulated target is only known once the athlete or coach chose it, so
    its ``known_at`` time may be later than the session's scheduled time.
    """

    ABSOLUTE_LOAD = "absolute_load"
    PERCENT_ONE_RM = "percent_one_rm"
    RPE_ANCHORED = "rpe_anchored"
    RIR_ANCHORED = "rir_anchored"
    BODYWEIGHT = "bodyweight"
    AUTOREGULATED = "autoregulated"
    UNKNOWN = "unknown"


class ProgramModificationKind(StrEnum):
    """What a program modification changed."""

    LOAD_CHANGE = "load_change"
    REP_CHANGE = "rep_change"
    SET_COUNT_CHANGE = "set_count_change"
    EXERCISE_SUBSTITUTION = "exercise_substitution"
    SESSION_RESCHEDULED = "session_rescheduled"
    SESSION_CANCELLED = "session_cancelled"
    SESSION_ADDED = "session_added"
    AUTOREGULATION_RULE = "autoregulation_rule"
    PROGRAM_VERSION_CHANGE = "program_version_change"
    OTHER = "other"


class ParentLift(StrEnum):
    """Competition lift an exercise belongs to."""

    SQUAT = "squat"
    BENCH = "bench"
    DEADLIFT = "deadlift"
    ACCESSORY = "accessory"
    OTHER = "other"
    UNKNOWN = "unknown"


class SpecificityLevel(StrEnum):
    """Observable competition specificity of an exercise.

    This describes observable semantics only. It carries **no** transfer
    coefficient: PSD must not hard-code how one variation substitutes for
    another, because that would impose a latent ontology on the ontology-free
    benchmark.
    """

    COMPETITION_LIFT = "competition_lift"
    COMPETITION_VARIATION = "competition_variation"
    SPORT_SPECIFIC = "sport_specific"
    GENERAL_STRENGTH = "general_strength"
    ACCESSORY = "accessory"
    UNKNOWN = "unknown"


class Laterality(StrEnum):
    """Unilateral or bilateral movement."""

    BILATERAL = "bilateral"
    UNILATERAL = "unilateral"
    UNKNOWN = "unknown"


class LiftType(StrEnum):
    """Competition lift for attempts and reported results."""

    SQUAT = "squat"
    BENCH = "bench"
    DEADLIFT = "deadlift"


class ObservationType(StrEnum):
    """Reported measurement instruments.

    Every member names something a source *recorded*. None of them is a latent
    physiological state, and PSD does not require any of them to be present.
    """

    RPE = "rpe"
    RIR = "rir"
    READINESS_SCORE = "readiness_score"
    SORENESS_SCORE = "soreness_score"
    FATIGUE_SCORE = "fatigue_score"
    SLEEP_DURATION = "sleep_duration"
    SLEEP_QUALITY_SCORE = "sleep_quality_score"
    STRESS_SCORE = "stress_score"
    MOTIVATION_SCORE = "motivation_score"
    MOOD_SCORE = "mood_score"
    PAIN_SCORE = "pain_score"
    CONFIDENCE_SCORE = "confidence_score"
    OTHER_REPORTED = "other_reported"
    UNKNOWN = "unknown"


class ObservationMethod(StrEnum):
    """How an observation was obtained."""

    SELF_REPORT = "self_report"
    COACH_REPORT = "coach_report"
    THIRD_PARTY_REPORT = "third_party_report"
    DEVICE_MEASURED = "device_measured"
    INSTRUMENTED_TEST = "instrumented_test"
    VIDEO_REVIEW = "video_review"
    SOURCE_REPORTED = "source_reported"
    UNKNOWN = "unknown"


class ObservationScope(StrEnum):
    """What an observation applies to."""

    ATHLETE = "athlete"
    SESSION = "session"
    EXERCISE = "exercise"
    SET = "set"
    REP = "rep"
    COMPETITION = "competition"
    UNKNOWN = "unknown"


class ReporterRole(StrEnum):
    """Who reported an observation."""

    ATHLETE = "athlete"
    COACH = "coach"
    SPOTTER = "spotter"
    OTHER_TRAINER = "other_trainer"
    UNKNOWN = "unknown"


class TestType(StrEnum):
    """What kind of performance test was performed.

    A test result is an *observation*, not an estimate of latent capacity: a
    meet best and a maximal single are not interchangeable with a true latent
    1RM.
    """

    MAXIMAL_SINGLE = "maximal_single"
    ESTIMATED_ONE_RM_FROM_WORK_SETS = "estimated_one_rm_from_work_sets"
    REPETITION_TEST = "repetition_test"
    ISOMETRIC = "isometric"
    JUMP_TEST = "jump_test"
    STRENGTH_ENDURANCE_TEST = "strength_endurance_test"
    OTHER = "other"
    UNKNOWN = "unknown"


class VelocityMethod(StrEnum):
    """Instrument used for a velocity observation."""

    LINEAR_POSITION_TRANSDUCER = "linear_position_transducer"
    ACCELEROMETER = "accelerometer"
    OPTICAL = "optical"
    RADAR = "radar"
    MANUAL_VIDEO = "manual_video"
    ESTIMATED_BY_COACH = "estimated_by_coach"
    UNKNOWN = "unknown"


class AttemptResult(StrEnum):
    """Outcome of a competition attempt.

    ``NO_ATTEMPT`` is not a failed lift and is not a zero load. Missing
    attempts stay missing.
    """

    GOOD_LIFT = "good_lift"
    BAD_LIFT = "bad_lift"
    NO_ATTEMPT = "no_attempt"
    WITHDRAWN = "withdrawn"
    UNKNOWN = "unknown"


class AttemptOrderBasis(StrEnum):
    """How the attempt sequence number was established.

    When a source exports attempts without an explicit order, PSD records the
    basis rather than silently trusting row order.
    """

    SOURCE_EXPLICIT = "source_explicit"
    DERIVED_FROM_SOURCE_ORDER = "derived_from_source_order"
    DERIVED_FROM_LOAD_ORDER = "derived_from_load_order"
    UNKNOWN = "unknown"


class CompetitionResultKind(StrEnum):
    """Reported or derived competition results.

    ``IS_DERIVED`` on the record marks values computed by PSD or a source rather
    than observed, keeping "reported best" separate from attempt observations.
    """

    SQUAT_BEST = "squat_best"
    BENCH_BEST = "bench_best"
    DEADLIFT_BEST = "deadlift_best"
    TOTAL = "total"
    PLACING = "placing"
    DOTS = "dots"
    GL_POINTS = "gl_points"
    WILKS = "wilks"
    OTHER = "other"


class EquipmentClass(StrEnum):
    """Normalized equipment class.

    Federations and apps use different vocabularies; the raw value is always
    preserved alongside the normalized member.
    """

    RAW = "raw"
    RAW_EQUIP = "raw_equip"
    CLASSIC_POWERLIFTING = "classic_powerlifting"
    SINGLE_PLY = "single_ply"
    OTHER = "other"
    UNKNOWN = "unknown"


class IdentityStatus(StrEnum):
    """How confidently an athlete record identifies one person.

    Ambiguous identity is never silently resolved: PSD links records
    optimistically at ingest and records the status so downstream evaluation can
    exclude or down-weight it.
    """

    SINGLE_SOURCE_UNVERIFIED = "single_source_unverified"
    SINGLE_SOURCE_VERIFIED = "single_source_verified"
    CROSS_SOURCE_LINKED = "cross_source_linked"
    AMBIGUOUS_UNRESOLVED = "ambiguous_unresolved"
    AMBIGUOUS_CONFLICT = "ambiguous_conflict"
    UNKNOWN = "unknown"


class IdentityLinkMethod(StrEnum):
    """How an athlete was linked to a source record."""

    SOURCE_EXPLICIT_ID = "source_explicit_id"
    EXACT_ATTRIBUTE_MATCH = "exact_attribute_match"
    FUZZY_ATTRIBUTE_MATCH = "fuzzy_attribute_match"
    MANUAL_REVIEW = "manual_review"
    UNRESOLVED = "unresolved"
    UNKNOWN = "unknown"


class BodyMeasurementType(StrEnum):
    """Which body measurement a record holds."""

    BODY_MASS = "body_mass"
    HEIGHT = "height"
    BODY_FAT_PERCENTAGE = "body_fat_percentage"
    SKELETAL_MUSCLE_MASS = "skeletal_muscle_mass"
    WAIST_CIRCUMFERENCE = "waist_circumference"
    CHEST_CIRCUMFERENCE = "chest_circumference"
    OTHER = "other"


class BodyMeasurementMethod(StrEnum):
    """How a body measurement was obtained."""

    BATHROOM_SCALE = "bathroom_scale"
    CLINICAL_SCALE = "clinical_scale"
    SMART_SCALE = "smart_scale"
    BIOIMPEDANCE = "bioimpedance"
    TAPE_MEASURE = "tape_measure"
    CALIPER = "caliper"
    SELF_REPORTED = "self_reported"
    SOURCE_REPORTED = "source_reported"
    UNKNOWN = "unknown"


class BodyMassContext(StrEnum):
    """When a body-mass measurement was taken relative to activity."""

    MORNING_FASTED = "morning_fasted"
    PRE_SESSION = "pre_session"
    POST_SESSION = "post_session"
    WEIGH_IN = "weigh_in"
    PRE_COMPETITION = "pre_competition"
    UNSPECIFIED = "unspecified"
    UNKNOWN = "unknown"


class SexCategory(StrEnum):
    """Source-reported sex category.

    Kept as a reported attribute with the raw source value preserved. PSD makes
    no claim about how sex category relates to performance, and this field
    carries no physiological meaning in the benchmark.
    """

    FEMALE = "female"
    MALE = "male"
    INTERSEX = "intersex"
    OTHER_SELF_DESCRIBED = "other_self_described"
    NOT_RECORDED = "not_recorded"
    UNKNOWN = "unknown"


class EventTimePrecision(StrEnum):
    """Granularity of an event timestamp.

    Recorded so that a date-only competition is never silently treated as a
    midnight-UTC instant of known precision.
    """

    DATE_ONLY = "date_only"
    MINUTE = "minute"
    SECOND = "second"
    MILLISECOND = "millisecond"
    UNKNOWN = "unknown"


class EquipmentItem(StrEnum):
    """Equipment whose state changes over an athlete's history."""

    BAR = "bar"
    PLATES = "plates"
    SHOES = "shoes"
    HEEL_RAISERS = "heel_raisers"
    BELT = "belt"
    KNEE_SLEEVES = "knee_sleeves"
    WRIST_WRAPS = "wrist_wraps"
    SHIRT = "shirt"
    SUIT = "suit"
    STOOL = "stool"
    RACK = "rack"
    IMPLEMENTS = "implements"
    MACHINE = "machine"
    OTHER = "other"
    UNKNOWN = "unknown"


class ImplementType(StrEnum):
    """Implement used for an exercise."""

    BARBELL = "barbell"
    PLATE_LOADED_BARBELL = "plate_loaded_barbell"
    DUMBBELL = "dumbbell"
    KETTLEBELL = "kettlebell"
    MACHINE = "machine"
    CABLE = "cable"
    BODYWEIGHT = "bodyweight"
    BAND = "band"
    SMITH_MACHINE = "smith_machine"
    SPECIALTY_BAR = "specialty_bar"
    OTHER = "other"
    UNKNOWN = "unknown"


class ValueEncoding(StrEnum):
    """How a before/after value in a program modification is encoded."""

    TEXT = "text"
    NUMBER = "number"
    JSON = "json"


#: Every controlled vocabulary, keyed by the field it constrains.
VOCABULARIES: Final[dict[str, tuple[str, ...]]] = {
    "missingness_reason": tuple(member.value for member in MissingnessReason),
    "quality_flag": tuple(member.value for member in QualityFlag),
    "session_status": tuple(member.value for member in SessionStatus),
    "not_performed_reason": tuple(member.value for member in NotPerformedReason),
    "session_type": tuple(member.value for member in SessionType),
    "set_status": tuple(member.value for member in SetStatus),
    "rep_status": tuple(member.value for member in RepStatus),
    "prescription_basis": tuple(member.value for member in PrescriptionBasis),
    "program_modification_kind": tuple(member.value for member in ProgramModificationKind),
    "parent_lift": tuple(member.value for member in ParentLift),
    "specificity_level": tuple(member.value for member in SpecificityLevel),
    "laterality": tuple(member.value for member in Laterality),
    "lift_type": tuple(member.value for member in LiftType),
    "observation_type": tuple(member.value for member in ObservationType),
    "observation_method": tuple(member.value for member in ObservationMethod),
    "observation_scope": tuple(member.value for member in ObservationScope),
    "reporter_role": tuple(member.value for member in ReporterRole),
    "test_type": tuple(member.value for member in TestType),
    "velocity_method": tuple(member.value for member in VelocityMethod),
    "attempt_result": tuple(member.value for member in AttemptResult),
    "attempt_order_basis": tuple(member.value for member in AttemptOrderBasis),
    "competition_result_kind": tuple(member.value for member in CompetitionResultKind),
    "equipment_class": tuple(member.value for member in EquipmentClass),
    "identity_status": tuple(member.value for member in IdentityStatus),
    "identity_link_method": tuple(member.value for member in IdentityLinkMethod),
    "body_measurement_type": tuple(member.value for member in BodyMeasurementType),
    "body_measurement_method": tuple(member.value for member in BodyMeasurementMethod),
    "body_mass_context": tuple(member.value for member in BodyMassContext),
    "sex_category": tuple(member.value for member in SexCategory),
    "event_time_precision": tuple(member.value for member in EventTimePrecision),
    "equipment_item": tuple(member.value for member in EquipmentItem),
    "implement_type": tuple(member.value for member in ImplementType),
    "value_encoding": tuple(member.value for member in ValueEncoding),
    "mass_unit": ("kg", "lb"),
    "length_unit": ("cm", "in", "m"),
}


def vocabulary_members(name: str) -> tuple[str, ...]:
    """Return the members of a named vocabulary.

    Args:
        name: Vocabulary name, for example ``set_status``.

    Returns:
        The member values.

    Raises:
        KeyError: The vocabulary is unknown.
    """
    return VOCABULARIES[name]
