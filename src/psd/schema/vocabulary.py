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

from psd.provenance.sources import (
    ConsentBasis,
    DataRegime,
    PublicationBasis,
    RedistributionPolicy,
    SourceNature,
)

__all__ = (
    "TEXT_NORMALIZATION_RULES",
    "VOCABULARIES",
    "AgePrecision",
    "AliasSourceSystem",
    "AmbiguityReason",
    "AttemptOrderBasis",
    "AttemptResult",
    "AttemptRole",
    "BarType",
    "BodyMassContext",
    "BodyMeasurementMethod",
    "BodyMeasurementType",
    "CompetitionEvent",
    "CompetitionResultKind",
    "ConfigurationFlag",
    "ConsentBasis",
    "DataRegime",
    "EquipmentClass",
    "EquipmentItem",
    "EventTimePrecision",
    "ExerciseEquipment",
    "Grip",
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
    "ParticipationStatus",
    "PauseRule",
    "PrescriptionBasis",
    "ProgramModificationKind",
    "PublicationBasis",
    "QualityFlag",
    "RangeOfMotion",
    "RedistributionPolicy",
    "RepStatus",
    "ReportedBestSemantics",
    "ReporterRole",
    "ResolutionMethod",
    "ResolutionStatus",
    "SessionStatus",
    "SessionType",
    "SetStatus",
    "SexCategory",
    "SourceNature",
    "SpecificityLevel",
    "Stance",
    "TempoPattern",
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

    ``COMPETITION_LIFT`` is the lift *as the rules define it*, so it may only be
    claimed where the rules leave the observable open. ``COMPETITION_VARIATION``
    is a way of performing that lift which the rules admit but do not fix. Both
    stance-qualified deadlifts sit in the second class symmetrically: the rules
    prescribe neither a conventional nor a sumo stance, so neither style is the
    competition lift and neither is more of it than the other.
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


class Stance(StrEnum):
    """Foot placement, where the label distinguishes one.

    ``NOT_SPECIFIED`` is the default and is deliberately distinct from
    ``UNKNOWN``: an unqualified label is normal in powerlifting logs, and PSD
    refuses to guess a stance the source never named. Sumo and conventional
    deadlifts are separate entities rather than one entity with a stance
    attribute, because the distinction may matter to a model even though both
    share the same parent lift -- and because competition rules admit both
    stances equally, so neither can be the unqualified deadlift.
    """

    NOT_SPECIFIED = "not_specified"
    HIGH = "high"
    LOW = "low"
    WIDE = "wide"
    MODERATE = "moderate"
    NARROW = "narrow"
    SPLIT = "split"
    UNKNOWN = "unknown"


class Grip(StrEnum):
    """Hand placement, where the label distinguishes one.

    Close-grip and competition-grip bench presses stay separate entities. They
    share a parent lift and nothing else, and collapsing them would assert an
    equivalence the source never made.
    """

    NOT_SPECIFIED = "not_specified"
    COMPETITION = "competition"
    FRONT_RACK = "front_rack"
    CLOSE = "close"
    WIDE = "wide"
    NEUTRAL = "neutral"
    MIXED = "mixed"
    OVERHAND = "overhand"
    UNDERHAND = "underhand"
    SUPINATED = "supinated"
    PRONATED = "pronated"
    UNKNOWN = "unknown"


class RangeOfMotion(StrEnum):
    """Range of motion the exercise actually traverses.

    ``ELEVATED_START`` covers lifts that begin from an elevated position rather
    than the floor, such as a deficit deadlift. ``REDUCED`` is shorter than
    ``PARTIAL``: a floor press or a Spoto press stops well short of a full
    competition repetition.
    """

    NOT_SPECIFIED = "not_specified"
    COMPETITION = "competition"
    FULL = "full"
    PARTIAL = "partial"
    REDUCED = "reduced"
    ELEVATED_START = "elevated_start"
    UNKNOWN = "unknown"


class PauseRule(StrEnum):
    """Pause characteristics.

    The distinction that matters here is between a pause the competition rules
    *mandate* -- which is part of the definition of the lift itself -- and a pause
    a program *adds* on top of it. ``COMPETITION`` is the mandated pause; ``BRIEF``,
    ``COUNT_2``, ``COUNT_3`` and ``LONG`` are added pauses. That is why
    ``Paused Bench Press`` (``BRIEF``) stays distinct from ``Bench Press``
    (``COMPETITION``) even though both record ``pause = True`` at the coarser
    ``Competition Bench Press`` family level.

    A 2-count and a long pause are deliberately different members: a counted
    pause and an open-ended one are different prescriptions and PSD does not
    collapse them.
    """

    NOT_SPECIFIED = "not_specified"
    NONE = "none"
    COMPETITION = "competition"
    BRIEF = "brief"
    COUNT_2 = "count_2"
    COUNT_3 = "count_3"
    LONG = "long"
    UNKNOWN = "unknown"


class TempoPattern(StrEnum):
    """Prescribed lifting tempo.

    ``TEMPO_COUNT`` is a numeric scheme such as ``3-0-1``; ``TEMPO_PRESCRIBED``
    is a controlled tempo whose exact scheme the label did not give. Tempo is a
    separate axis from :class:`PauseRule`: a lift may be tempo-controlled without
    being paused.
    """

    NOT_SPECIFIED = "not_specified"
    CONTROLLED_ECCENTRIC = "controlled_eccentric"
    TEMPO_COUNT = "tempo_count"
    TEMPO_PRESCRIBED = "tempo_prescribed"
    UNKNOWN = "unknown"


class BarType(StrEnum):
    """Which bar or shaft carries the load.

    Kept separate from :class:`ImplementType`: the implement says *how* the load
    is held, the bar says *which* bar. ``NOT_APPLICABLE`` is the honest default for
    bodyweight, dumbbell, cable and machine work, where there is no bar at all.
    """

    NOT_APPLICABLE = "not_applicable"
    OLYMPIC_BAR = "olympic_bar"
    SPECIALTY_BAR = "specialty_bar"
    SAFETY_BAR = "safety_bar"
    TRAP_BAR = "trap_bar"
    EZ_BAR = "ez_bar"
    CURL_BAR = "curl_bar"
    T_BAR = "t_bar"
    UNKNOWN = "unknown"


class ExerciseEquipment(StrEnum):
    """Fixed apparatus the exercise requires, beyond the implement.

    This is the *apparatus* axis, not the athlete's chosen equipment: shoes,
    belt and suit belong to ``equipment_state`` on the athlete's history, while
    this says what machine or bench the exercise itself needs. ``NONE`` is the
    honest value for a free-standing barbell or bodyweight exercise that needs no
    apparatus, which is different from not having checked.
    """

    NONE = "none"
    FLAT_BENCH = "flat_bench"
    INCLINE_BENCH = "incline_bench"
    DECLINE_BENCH = "decline_bench"
    RACK = "rack"
    BOX = "box"
    PLATFORMS = "platforms"
    MACHINE = "machine"
    CABLE_STATION = "cable_station"
    LANDMINE = "landmine"
    DIP_BARS = "dip_bars"
    HYPEREXTENSION_BENCH = "hyperextension_bench"
    UNKNOWN = "unknown"


class ConfigurationFlag(StrEnum):
    """Positional or configuration descriptors that need a flag rather than a level.

    Kept as a sorted, de-duplicated list so two equal configurations serialize to
    identical bytes regardless of authoring order. These describe *how* the
    movement is performed as part of its definition -- seated, from blocks, with
    a safety bar -- and never say how much the variation is worth.
    """

    ASSISTED = "assisted"
    BLOCK_SUPPORTED = "block_supported"
    COMPETITION_TOUCH_POINT = "competition_touch_point"
    FEET_ELEVATED = "feet_elevated"
    FEET_FIXED = "feet_fixed"
    HAND_SPACING_REDUCED = "hand_spacing_reduced"
    HAND_SPACING_WIDE = "hand_spacing_wide"
    KNEE_PADDED = "knee_padded"
    SAFETY_BARS = "safety_bars"
    SEATED = "seated"
    SHORT_RANGE_OF_MOTION = "short_range_of_motion"
    TOUCH_AND_GO = "touch_and_go"
    UNKNOWN = "unknown"


class AliasSourceSystem(StrEnum):
    """Well-known namespaces for source-specific exercise aliases.

    This is *not* a closed vocabulary for the persisted
    ``exercise_alias.source_system`` column, which is free text so that a future
    adapter can contribute a new namespace without a canonical schema change.
    These members exist so that PSD's own artifacts can be validated against the
    namespaces it ships with, and so a CLI reader can spell them correctly.
    """

    PSD_REGISTRY = "psd_registry"
    HEVY = "hevy"
    STRONG = "strong"
    TRAINHEROIC = "trainheroic"
    GENERIC_CSV = "generic_csv"
    SPREADSHEET = "spreadsheet"


class ResolutionStatus(StrEnum):
    """How far a raw source label could be resolved.

    The ladder is deliberately explicit. ``EXACT_CANONICAL`` and
    ``RESOLVED_ALIAS`` name exactly one canonical exercise; ``PARTIAL_FAMILY``
    names only a parent lift; ``AMBIGUOUS`` names several defensible candidates;
    ``UNMAPPED`` admits nothing. Coverage is never improved by promoting a
    ``PARTIAL_FAMILY`` or ``AMBIGUOUS`` result to a resolved one, because a
    confident wrong mapping is worse than an explicit gap.
    """

    EXACT_CANONICAL = "exact_canonical"
    RESOLVED_ALIAS = "resolved_alias"
    PARTIAL_FAMILY = "partial_family"
    AMBIGUOUS = "ambiguous"
    UNMAPPED = "unmapped"


class ResolutionMethod(StrEnum):
    """Which stage of the pipeline produced the outcome.

    Recorded so that a resolved mapping can be audited: a label that resolved
    because an alias registry entry matched is a weaker claim than a label that
    already *was* the canonical identity.
    """

    CANONICAL_IDENTITY = "canonical_identity"
    REGISTERED_ALIAS = "registered_alias"
    CROSS_SOURCE_ALIAS = "cross_source_alias"
    STRUCTURED_INTERPRETATION = "structured_interpretation"
    #: The lookup found an exercise, but the source phrased the label as a
    #: question, so the answer is recorded as an ambiguity rather than a mapping.
    QUESTION_FORM = "question_form"
    GENERIC_QUALIFIER = "generic_qualifier"
    FAMILY_KEYWORD = "family_keyword"
    CURATED_AMBIGUOUS = "curated_ambiguous"
    NO_MATCH = "no_match"


class AmbiguityReason(StrEnum):
    """Why a label was not forced into a canonical exercise.

    ``UNKNOWN_SOURCE_TAXONOMY`` means the registry has never seen the label.
    The others mean the registry *has* seen it and is recording, on purpose,
    that the source did not say enough.
    """

    UNSPECIFIED_VARIATION = "unspecified_variation"
    UNSPECIFIED_MACHINE = "unspecified_machine"
    UNSPECIFIED_IMPLEMENT = "unspecified_implement"
    MULTIPLE_DEFENSIBLE_MATCHES = "multiple_defensible_matches"
    QUESTION_FORM_LABEL = "question_form_label"
    CONFLICTING_ALIAS_BINDINGS = "conflicting_alias_bindings"
    UNKNOWN_SOURCE_TAXONOMY = "unknown_source_taxonomy"


#: Identifiers of the deterministic text-normalization rules, in application order.
#:
#: Declared here rather than in the normalizer so that the controlled-vocabulary
#: registry stays the single authority for every enumerated column value, and so
#: the declarative validator can check a persisted ``normalization_rules`` list
#: without importing the ontology.
TEXT_NORMALIZATION_RULES: Final[tuple[str, ...]] = (
    "unicode_nfkc",
    "unicode_casefold",
    "unicode_nfkc_recheck",
    "strip_diacritics",
    "punctuation_to_space",
    "collapse_whitespace",
    "expand_abbreviation",
    "fold_plural",
    "join_compound",
    "question_form",
)


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


class AttemptRole(StrEnum):
    """What role an attempt played in the meet.

    This is the distinction a three-attempt model cannot express on its own. The
    first three attempts of a lift are the ones the rules count; a fourth attempt
    exists only to attempt a single-lift record, and it **does not contribute to the
    total**. Without this, a fourth attempt is indistinguishable from an ordinary
    third attempt and a naive reading of the corpus inflates totals.

    ``RECORD_FOURTH`` is deliberately not modelled as "attempt number 4" alone: the
    number and the role must agree, and the contract enforces that they do.
    """

    #: One of the three attempts the rules count toward the lift and the total.
    ORDERED = "ordered"
    #: A fourth attempt taken solely for a single-lift record. It counts toward
    #: nothing.
    RECORD_FOURTH = "record_fourth"


class CompetitionEvent(StrEnum):
    """Which competition event a result belongs to.

    An event is a *declared* event type, not a description of which lifts happened.
    A reduced event such as ``B`` or ``BD`` means the lifter never attempted the
    other lifts, and that is categorically different from attempting them and
    failing: one is not in the meet, the other is a bad lift.
    """

    SQUAT_BENCH_DEADLIFT = "sbd"
    BENCH_DEADLIFT = "bd"
    SQUAT_DEADLIFT = "sd"
    SQUAT_BENCH = "sb"
    SQUAT = "s"
    BENCH = "b"
    DEADLIFT = "d"


class ParticipationStatus(StrEnum):
    """How a lifter's participation ended, independent of the placing.

    A source that reports ``DQ``, ``DD``, ``G`` or ``NS`` is not reporting a rank.
    Coercing those codes to a number would invent a placing nobody earned, so the
    numeric placing lives in its own nullable column and these codes live here.
    """

    PLACED = "placed"
    #: Succeeded, but not eligible for awards.
    GUEST = "guest"
    #: Disqualified, possibly for procedural reasons rather than failed lifts.
    DISQUALIFIED = "disqualified"
    #: Disqualified by a failed drug test.
    DRUG_DISQUALIFIED = "drug_disqualified"
    #: Did not appear on the meet day.
    NO_SHOW = "no_show"
    UNKNOWN = "unknown"


class AgePrecision(StrEnum):
    """Whether a reported age is exact or an approximation.

    Some federations publish only a birth year, so the age is known to lie between
    two integers. Collapsing that into one integer would invent a precision the
    source did not have, and rounding is exactly as wrong as guessing.
    """

    EXACT = "exact"
    APPROXIMATE = "approximate"


class ReportedBestSemantics(StrEnum):
    """What a source-reported best lift actually reports.

    Most reported bests are the best *successful* attempt. A small number of
    federations instead report the lowest weight the lifter attempted and failed,
    which the source publishes as a negative best. That is a completely different
    fact, and reading it as a negative successful lift -- or as a missing value --
    would both be wrong.
    """

    SUCCESSFUL_BEST = "successful_best"
    FAILED_ATTEMPT_ONLY = "failed_attempt_only"


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

    This is the **competition equipment category** -- the class the lifts were
    performed under, and therefore the equipment the rules *allowed*. It is not a
    statement that the athlete wore any particular item: a federation with no
    wrap-free category puts every lifter, sleeve-wearer or not, in the same
    category. Federations and apps use different vocabularies; the raw value is
    always preserved alongside the normalized member.

    ``MULTI_PLY`` and ``STRAPS_ALLOWED`` exist because federation vocabularies name
    categories that ``CLASSIC_POWERLIFTING`` and ``SINGLE_PLY`` do not cover.
    Folding equipped multi-ply into ``SINGLE_PLY`` would assert a distinction the
    source explicitly draws, so the categories stay separate.
    """

    RAW = "raw"
    RAW_EQUIP = "raw_equip"
    CLASSIC_POWERLIFTING = "classic_powerlifting"
    SINGLE_PLY = "single_ply"
    MULTI_PLY = "multi_ply"
    #: Competition classes where straps were permitted on the deadlift.
    STRAPS_ALLOWED = "straps_allowed"
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
    "stance": tuple(member.value for member in Stance),
    "grip": tuple(member.value for member in Grip),
    "range_of_motion": tuple(member.value for member in RangeOfMotion),
    "pause_rule": tuple(member.value for member in PauseRule),
    "tempo": tuple(member.value for member in TempoPattern),
    "bar_type": tuple(member.value for member in BarType),
    "equipment": tuple(member.value for member in ExerciseEquipment),
    "configuration_flag": tuple(member.value for member in ConfigurationFlag),
    "alias_source_system": tuple(member.value for member in AliasSourceSystem),
    "resolution_status": tuple(member.value for member in ResolutionStatus),
    "resolution_method": tuple(member.value for member in ResolutionMethod),
    "ambiguity_reason": tuple(member.value for member in AmbiguityReason),
    "text_rule": TEXT_NORMALIZATION_RULES,
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
    "attempt_role": tuple(member.value for member in AttemptRole),
    "competition_event": tuple(member.value for member in CompetitionEvent),
    "participation_status": tuple(member.value for member in ParticipationStatus),
    "age_precision": tuple(member.value for member in AgePrecision),
    "reported_best_semantics": tuple(member.value for member in ReportedBestSemantics),
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
    "source_nature": tuple(member.value for member in SourceNature),
    "data_regime": tuple(member.value for member in DataRegime),
    "consent_basis": tuple(member.value for member in ConsentBasis),
    "publication_basis": tuple(member.value for member in PublicationBasis),
    "redistribution": tuple(member.value for member in RedistributionPolicy),
    "mass_unit": ("kg", "lb"),
    "length_unit": ("cm", "in", "m"),
    "measurement_unit": ("kg", "lb", "cm", "in", "m"),
    "percentage_unit": ("percent", "%"),
    # A normalized measurement unit can be either a mass/length unit or percent,
    # depending on the measurement type, so it needs the union.
    "normalized_measurement_unit": ("kg", "lb", "cm", "in", "m", "percent", "%"),
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
