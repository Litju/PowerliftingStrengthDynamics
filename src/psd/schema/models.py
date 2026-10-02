"""Pydantic contracts for every canonical entity.

These models are the single source of truth for the canonical schema. The Arrow
schemas are derived from them (:mod:`psd.schema.arrow`) and the persisted column
order is declared per table (:mod:`psd.schema.registry`), so the contract, the
persisted schema, and the CLI-facing validation surface cannot drift apart.

Structural separation
---------------------

``planned_*`` records hold prescription only. ``performed_*`` records hold
execution only. ``observation``/``performance_test``/``velocity_observation``
hold measurement. ``competition*`` records hold outcomes. No model on one side
carries a field belonging to another, and cross-table linkage is an explicit
nullable foreign key. A ``planned_*`` reference that is ``null`` means no
prescription existed in the source: PSD never reconstructs one from execution.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from datetime import datetime
from typing import Final, Self

from pydantic import Field, model_validator

from psd.schema.base import (
    ContextRecord,
    EventRecord,
    normalize_configuration_flags,
    normalize_timestamp,
)
from psd.schema.vocabulary import (
    AgePrecision,
    AmbiguityReason,
    AttemptOrderBasis,
    AttemptResult,
    AttemptRole,
    BarType,
    BodyMassContext,
    BodyMeasurementMethod,
    BodyMeasurementType,
    CompetitionEvent,
    CompetitionResultKind,
    ConfigurationFlag,
    EquipmentClass,
    EquipmentItem,
    EventTimePrecision,
    ExerciseEquipment,
    Grip,
    IdentityLinkMethod,
    IdentityStatus,
    ImplementType,
    Laterality,
    LiftType,
    MissingnessReason,
    NotPerformedReason,
    ObservationMethod,
    ObservationScope,
    ObservationType,
    ParentLift,
    ParticipationStatus,
    PauseRule,
    PrescriptionBasis,
    ProgramModificationKind,
    RangeOfMotion,
    ReportedBestSemantics,
    ReporterRole,
    RepStatus,
    ResolutionMethod,
    ResolutionStatus,
    SessionStatus,
    SessionType,
    SetStatus,
    SexCategory,
    SpecificityLevel,
    Stance,
    TempoPattern,
    TestType,
    ValueEncoding,
    VelocityMethod,
)
from psd.units import (
    MassUnit,
    NonNormalizableValueError,
    normalize_length,
    normalize_mass,
)

__all__ = (
    "ORDERED_ATTEMPT_NUMBERS",
    "RECORD_FOURTH_ATTEMPT_NUMBER",
    "STATUS_BY_RESOLUTION_METHOD",
    "AthleteRecord",
    "AthleteSourceLinkRecord",
    "BodyMeasurementRecord",
    "CompetitionAttemptRecord",
    "CompetitionMeetRecord",
    "CompetitionRecord",
    "CompetitionReportedResultRecord",
    "EquipmentStateRecord",
    "ExerciseAliasRecord",
    "ExerciseDefinitionRecord",
    "ExerciseNormalizationRecord",
    "ObservationRecord",
    "PerformanceTestRecord",
    "PerformedExerciseRecord",
    "PerformedRepRecord",
    "PerformedSessionRecord",
    "PerformedSetRecord",
    "PlannedExerciseRecord",
    "PlannedSessionRecord",
    "PlannedSetRecord",
    "ProgramModificationRecord",
    "ProgramRecord",
    "ProgramVersionRecord",
    "VelocityObservationRecord",
    "pause_flag_for_rule",
)

NORMALIZED_MASS_UNIT: Final[str] = MassUnit.KG.value
NORMALIZED_LENGTH_UNIT: Final[str] = "cm"
PERCENT_UNIT: Final[str] = "percent"

LENGTH_TYPES: Final[frozenset[BodyMeasurementType]] = frozenset(
    {
        BodyMeasurementType.HEIGHT,
        BodyMeasurementType.WAIST_CIRCUMFERENCE,
        BodyMeasurementType.CHEST_CIRCUMFERENCE,
    }
)

PERCENTAGE_TYPES: Final[frozenset[BodyMeasurementType]] = frozenset(
    {
        BodyMeasurementType.BODY_FAT_PERCENTAGE,
    }
)

PERCENTAGE_UNITS: Final[frozenset[str]] = frozenset({"percent", "%"})

RPE_MIN: Final[float] = 0.0
RPE_MAX: Final[float] = 10.0
PERCENT_MIN: Final[float] = 0.0
PERCENT_MAX: Final[float] = 100.0
ATTEMPT_NUMBERS: Final[tuple[int, ...]] = (1, 2, 3)

#: The attempts the competition rules count toward the lift and toward the total.
ORDERED_ATTEMPT_NUMBERS: Final[tuple[int, ...]] = (1, 2, 3)

#: The one further attempt a ruleset permits, taken solely for a single-lift
#: record. It is not part of :data:`ORDERED_ATTEMPT_NUMBERS` and contributes to
#: no total.
RECORD_FOURTH_ATTEMPT_NUMBER: Final[int] = 4

#: The reported-result kinds whose value is a best lift rather than a total or a
#: score. Used to require the negative-best semantics to be stated explicitly.
REPORTED_BEST_KINDS: Final[frozenset[CompetitionResultKind]] = frozenset(
    {
        CompetitionResultKind.SQUAT_BEST,
        CompetitionResultKind.BENCH_BEST,
        CompetitionResultKind.DEADLIFT_BEST,
    }
)

#: ``Event`` codes map onto the lifts a lifter actually attempted. A lift missing
#: from its event was never attempted, which is a different fact from attempting it
#: and failing, and the cross-record rules use this to keep the two apart.
LIFTS_BY_COMPETITION_EVENT: Final[Mapping[CompetitionEvent, frozenset[LiftType]]] = {
    CompetitionEvent.SQUAT_BENCH_DEADLIFT: frozenset(
        {LiftType.SQUAT, LiftType.BENCH, LiftType.DEADLIFT}
    ),
    CompetitionEvent.BENCH_DEADLIFT: frozenset({LiftType.BENCH, LiftType.DEADLIFT}),
    CompetitionEvent.SQUAT_DEADLIFT: frozenset({LiftType.SQUAT, LiftType.DEADLIFT}),
    CompetitionEvent.SQUAT_BENCH: frozenset({LiftType.SQUAT, LiftType.BENCH}),
    CompetitionEvent.SQUAT: frozenset({LiftType.SQUAT}),
    CompetitionEvent.BENCH: frozenset({LiftType.BENCH}),
    CompetitionEvent.DEADLIFT: frozenset({LiftType.DEADLIFT}),
}

NORMALIZATION_TOLERANCE: Final[float] = 1e-9

#: Pause rules that represent a pause *added* on top of the competition rules of
#: the parent lift, mapped to the coarse ``pause`` flag RES-235 defined.
_ADDED_PAUSE_RULES: Final[dict[PauseRule, bool]] = {
    PauseRule.BRIEF: True,
    PauseRule.COUNT_2: True,
    PauseRule.COUNT_3: True,
    PauseRule.LONG: True,
    PauseRule.NONE: False,
    PauseRule.COMPETITION: False,
    PauseRule.NOT_SPECIFIED: False,
    PauseRule.UNKNOWN: False,
}

#: Specificity levels that must name a real competition lift as their parent.
COMPETITION_PARENT_LIFTS: Final[frozenset[ParentLift]] = frozenset(
    {ParentLift.SQUAT, ParentLift.BENCH, ParentLift.DEADLIFT}
)

#: ``source_system`` is deliberately free text so a future adapter can contribute
#: a new namespace without a canonical schema change. This is the shape PSD's own
#: namespaces follow, and what ``exercise_alias`` validates against.
SOURCE_SYSTEM_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[a-z][a-z0-9_]{0,63}$")

#: Alias registries are keyed by this version tag, recorded on every persisted row.
ALIAS_VERSION_FIELD: Final[str] = "alias_registry_version"
ONTOLOGY_VERSION_FIELD: Final[str] = "ontology_version"


def pause_flag_for_rule(pause_rule: PauseRule) -> bool | None:
    """Return the coarse ``pause`` flag for a pause rule.

    The two views answer different questions. ``pause_rule`` says *which* pause the
    exercise has; ``pause`` says whether a pause is *added* on top of the one the
    competition rules already mandate. The competition bench press therefore records
    ``pause=False`` with ``pause_rule=competition`` -- the mandated touch-and-pause is
    part of the lift, not a variation of it -- while a paused bench press records
    ``pause=True`` with ``pause_rule=brief``.

    ``not_specified`` and ``unknown`` return ``None``: the source never said, so the
    flag stays absent rather than claiming either answer.
    """
    if pause_rule in (PauseRule.NOT_SPECIFIED, PauseRule.UNKNOWN):
        return None
    return _ADDED_PAUSE_RULES[pause_rule]


def _validate_percentage_triplet(
    *,
    raw_value: float | None,
    raw_unit: str | None,
    normalized: float | None,
) -> None:
    """Validate a raw/unit/normalized percentage triplet.

    A percentage is dimensionless, so a source may legitimately report
    ``14.2`` with no unit at all. The raw value is still required: PSD never
    synthesizes a measurement.
    """
    if raw_value is None:
        if raw_unit is not None or normalized is not None:
            msg = "body_measurement: a unit or normalized value requires a raw value."
            raise ValueError(msg)
        return
    if not (PERCENT_MIN < raw_value <= PERCENT_MAX):
        msg = (
            f"body_measurement: percentage {raw_value!r} must be within "
            f"({PERCENT_MIN}, {PERCENT_MAX}]."
        )
        raise ValueError(msg)
    if raw_unit is not None and raw_unit.strip().lower() not in PERCENTAGE_UNITS:
        msg = (
            f"body_measurement: unit {raw_unit!r} is not a percentage unit; expected "
            f"one of {sorted(PERCENTAGE_UNITS)}."
        )
        raise ValueError(msg)
    if normalized is None:
        msg = "body_measurement: normalized value is required when a raw value is present."
        raise ValueError(msg)
    if abs(normalized - raw_value) > NORMALIZATION_TOLERANCE:
        msg = (
            f"body_measurement: normalized percentage {normalized!r} disagrees with "
            f"raw {raw_value!r}."
        )
        raise ValueError(msg)


# --------------------------------------------------------------------------
# shared validation helpers
# --------------------------------------------------------------------------


def _validate_mass_triplet(
    *,
    raw_value: float | None,
    raw_unit: str | None,
    normalized: float | None,
    prefix: str,
) -> None:
    """Validate a raw/unit/normalized mass triplet.

    Rules: a raw value requires a unit; a normalized value requires a raw value
    (normalization alone would destroy the audit trail); a raw mass of zero is
    rejected because it is not a load, it is a missing-value sentinel.
    """
    if raw_value is None and raw_unit is None:
        if normalized is not None:
            msg = (
                f"{prefix}: normalized value present without a raw source value; "
                "PSD never synthesizes values."
            )
            raise ValueError(msg)
        return
    if raw_value is None or raw_unit is None:
        msg = f"{prefix}: raw value and unit must be recorded together."
        raise ValueError(msg)
    if raw_value <= 0.0:
        msg = (
            f"{prefix}: raw value {raw_value!r} is not a plausible mass. Zero is not a "
            "missing-value encoding in PSD; use null with a missingness reason."
        )
        raise ValueError(msg)
    try:
        expected = normalize_mass(raw_value, raw_unit)
    except NonNormalizableValueError as error:
        msg = f"{prefix}: {error}"
        raise ValueError(msg) from error
    if normalized is None:
        msg = f"{prefix}: normalized value is required when a raw value is present."
        raise ValueError(msg)
    if abs(normalized - expected) > NORMALIZATION_TOLERANCE:
        msg = (
            f"{prefix}: normalized value {normalized!r} disagrees with "
            f"{raw_value!r} {raw_unit!r} ({expected!r} kg)."
        )
        raise ValueError(msg)


def _validate_length_triplet(
    *,
    raw_value: float | None,
    raw_unit: str | None,
    normalized: float | None,
    prefix: str,
) -> None:
    """Validate a raw/unit/normalized length triplet."""
    if raw_value is None and raw_unit is None:
        if normalized is not None:
            msg = f"{prefix}: normalized value present without a raw source value."
            raise ValueError(msg)
        return
    if raw_value is None or raw_unit is None:
        msg = f"{prefix}: raw value and unit must be recorded together."
        raise ValueError(msg)
    if raw_value <= 0.0:
        msg = f"{prefix}: raw value {raw_value!r} is not a plausible length."
        raise ValueError(msg)
    try:
        expected = normalize_length(raw_value, raw_unit)
    except NonNormalizableValueError as error:
        msg = f"{prefix}: {error}"
        raise ValueError(msg) from error
    if normalized is None:
        msg = f"{prefix}: normalized value is required when a raw value is present."
        raise ValueError(msg)
    if abs(normalized - expected) > NORMALIZATION_TOLERANCE:
        msg = (
            f"{prefix}: normalized value {normalized!r} disagrees with "
            f"{raw_value!r} {raw_unit!r} ({expected!r} cm)."
        )
        raise ValueError(msg)


def _validate_rpe(value: float | None, *, prefix: str) -> None:
    if value is None:
        return
    if not (RPE_MIN <= value <= RPE_MAX):
        msg = f"{prefix}: RPE {value!r} is outside the plausible range [{RPE_MIN}, {RPE_MAX}]."
        raise ValueError(msg)


def _validate_rir(value: float | None, *, prefix: str) -> None:
    if value is None:
        return
    if not (RPE_MIN <= value <= RPE_MAX):
        msg = f"{prefix}: RIR {value!r} is outside the plausible range [0, 10]."
        raise ValueError(msg)


# --------------------------------------------------------------------------
# context
# --------------------------------------------------------------------------


class AthleteRecord(ContextRecord):
    """One athlete identity, real or synthetic.

    ``is_synthetic`` and ``identity_status`` are load-bearing: PSD-Sim ground truth
    must never be projected onto real athletes, and ambiguous cross-source identity
    must never be silently resolved.
    """

    athlete_id: str = Field(min_length=1, max_length=160)
    pseudonym: str | None = Field(default=None, max_length=128)
    identity_status: IdentityStatus = IdentityStatus.UNKNOWN
    ambiguity_group_id: str | None = Field(default=None, max_length=160)
    is_synthetic: bool | None = None
    synthetic_regime: str | None = Field(default=None, max_length=32)
    sex_category_raw: str | None = Field(default=None, max_length=64)
    sex_category: SexCategory | None = None
    birth_year: int | None = Field(default=None, ge=1900, le=2100)
    country_code: str | None = Field(default=None, max_length=2)

    @model_validator(mode="after")
    def _check_consistency(self) -> Self:
        if self.is_synthetic is True and self.synthetic_regime is None:
            msg = "A synthetic athlete must declare its synthetic regime."
            raise ValueError(msg)
        if self.is_synthetic is True and self.synthetic_regime != "psd_sim":
            msg = f"synthetic_regime must be 'psd_sim'; got {self.synthetic_regime!r}."
            raise ValueError(msg)
        if self.is_synthetic is not True and self.synthetic_regime is not None:
            msg = "Only synthetic athletes may declare a synthetic regime."
            raise ValueError(msg)
        if self.sex_category is None and self.sex_category_raw is None:
            object.__setattr__(
                self,
                "missingness_reason",
                self.missingness_reason or MissingnessReason.NOT_RECORDED_IN_SOURCE,
            )
        return self


class AthleteSourceLinkRecord(ContextRecord):
    """Link between an athlete identity and one source's own athlete key.

    An unresolvable link is stored with ``link_method = unresolved`` and a null
    confidence, never dropped and never asserted.
    """

    athlete_id: str = Field(min_length=1, max_length=160)
    source_athlete_key: str | None = Field(default=None, max_length=512)
    link_method: IdentityLinkMethod = IdentityLinkMethod.UNKNOWN
    link_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    is_primary: bool = True

    @model_validator(mode="after")
    def _check_consistency(self) -> Self:
        unlinked = {IdentityLinkMethod.UNRESOLVED, IdentityLinkMethod.UNKNOWN}
        if self.link_method in unlinked and self.link_confidence is not None:
            msg = (
                f"link_confidence must be null when link_method is "
                f"{self.link_method.value!r}; an unresolved link has no confidence."
            )
            raise ValueError(msg)
        if self.link_method not in unlinked and self.source_athlete_key is None:
            # An unresolved link has, by definition, no key to point at.
            msg = "source_athlete_key is required for every resolved link method."
            raise ValueError(msg)
        return self


class BodyMeasurementRecord(EventRecord):
    """A body measurement with its raw source value preserved.

    Body mass is a first-class observation: it appears in longitudinal models and
    is frequently the only available covariate for weight-class context. Mass and
    length measurements normalize to kilograms and centimeters; percentage
    measurements such as body fat normalize to percent.
    """

    body_measurement_id: str = Field(min_length=1, max_length=160)
    athlete_id: str = Field(min_length=1, max_length=160)
    measured_at: datetime
    measurement_type: BodyMeasurementType
    event_time_precision: EventTimePrecision = EventTimePrecision.UNKNOWN
    measurement_context: BodyMassContext = BodyMassContext.UNSPECIFIED
    method: BodyMeasurementMethod = BodyMeasurementMethod.UNKNOWN
    raw_value: float | None = None
    raw_unit: str | None = Field(default=None, max_length=16)
    value_normalized: float | None = None
    unit_normalized: str | None = Field(default=None, max_length=16)

    @model_validator(mode="after")
    def _check_consistency(self) -> Self:
        object.__setattr__(
            self, "measured_at", normalize_timestamp(self.measured_at, field="measured_at")
        )
        if self.measurement_type in LENGTH_TYPES:
            _validate_length_triplet(
                raw_value=self.raw_value,
                raw_unit=self.raw_unit,
                normalized=self.value_normalized,
                prefix="body_measurement",
            )
            if self.value_normalized is not None and self.unit_normalized != NORMALIZED_LENGTH_UNIT:
                msg = "Length measurements normalize to centimeters."
                raise ValueError(msg)
        elif self.measurement_type in PERCENTAGE_TYPES:
            _validate_percentage_triplet(
                raw_value=self.raw_value,
                raw_unit=self.raw_unit,
                normalized=self.value_normalized,
            )
            if self.value_normalized is not None and self.unit_normalized != PERCENT_UNIT:
                msg = "Percentage measurements normalize to percent."
                raise ValueError(msg)
        else:
            _validate_mass_triplet(
                raw_value=self.raw_value,
                raw_unit=self.raw_unit,
                normalized=self.value_normalized,
                prefix="body_measurement",
            )
            if self.value_normalized is not None and self.unit_normalized != NORMALIZED_MASS_UNIT:
                msg = "Mass measurements normalize to kilograms."
                raise ValueError(msg)
        return self


class EquipmentStateRecord(ContextRecord):
    """Equipment in use over an interval, so equipment transitions are visible."""

    equipment_state_id: str = Field(min_length=1, max_length=160)
    athlete_id: str = Field(min_length=1, max_length=160)
    equipment_item: EquipmentItem
    effective_from: datetime
    effective_to: datetime | None = None
    identifier: str | None = Field(default=None, max_length=256)
    version_label: str | None = Field(default=None, max_length=128)

    @model_validator(mode="after")
    def _check_interval(self) -> Self:
        object.__setattr__(
            self, "effective_from", normalize_timestamp(self.effective_from, field="effective_from")
        )
        if self.effective_to is not None:
            object.__setattr__(
                self, "effective_to", normalize_timestamp(self.effective_to, field="effective_to")
            )
            if self.effective_to <= self.effective_from:
                msg = "effective_to must be strictly after effective_from."
                raise ValueError(msg)
        return self


class ExerciseDefinitionRecord(ContextRecord):
    """Observable exercise semantics for one canonical exercise.

    What an exercise **is**, and nothing about what it is worth. Every column here
    is a property of the movement itself -- which lift it belongs to, what carries
    the load, how the feet and hands are placed, how far it travels, what the pause
    and tempo are. There is deliberately no transfer coefficient, no specificity
    score, and no effectiveness number, because those are claims about athlete
    response that PSD must not assert on a model's behalf; a model is free to learn
    how response transfers between these descriptors.

    ``canonical_key`` is the stable, human-readable identity used to derive
    ``exercise_id`` and to reference the exercise from fixtures, adapters, and
    tests. Two records with the same key are the same exercise, which the
    cross-record rules enforce.

    ``pause`` and ``pause_rule`` are two views of one fact, kept deliberately.
    ``pause`` is the coarse RES-235 flag meaning "a pause is *added* on top of the
    competition rules of the parent lift"; ``pause_rule`` says which pause. That is
    why ``Competition Bench Press`` records ``pause=False`` with
    ``pause_rule=competition`` while ``Paused Bench Press`` records ``pause=True``
    with ``pause_rule=brief`` -- the rules-mandated pause is part of the lift, not
    a variation of it.
    """

    exercise_id: str = Field(min_length=1, max_length=160)
    canonical_key: str = Field(min_length=1, max_length=128, pattern=r"^[a-z][a-z0-9_]*$")
    canonical_name: str = Field(min_length=1, max_length=256)
    parent_lift: ParentLift = ParentLift.UNKNOWN
    specificity_level: SpecificityLevel = SpecificityLevel.UNKNOWN
    implement: ImplementType = ImplementType.UNKNOWN
    bar_type: BarType = BarType.UNKNOWN
    equipment: ExerciseEquipment = ExerciseEquipment.UNKNOWN
    laterality: Laterality = Laterality.UNKNOWN
    stance: Stance = Stance.UNKNOWN
    grip: Grip = Grip.UNKNOWN
    range_of_motion: RangeOfMotion = RangeOfMotion.UNKNOWN
    pause: bool | None = None
    pause_rule: PauseRule = PauseRule.UNKNOWN
    tempo: TempoPattern = TempoPattern.UNKNOWN
    configuration: tuple[ConfigurationFlag, ...] = ()
    equipment_note: str | None = Field(default=None, max_length=256)
    definition_note: str | None = Field(default=None, max_length=1024)

    @model_validator(mode="after")
    def _check_consistency(self) -> Self:
        """Enforce the descriptor relationships PSD can assert as facts.

        These are structural facts about the *ontology*, not judgements about
        training value: a competition lift is by definition one of the three
        regulated lifts, and an accessory is by definition not inside a competition
        lift's family.
        """
        object.__setattr__(self, "configuration", normalize_configuration_flags(self.configuration))
        expected_pause = _ADDED_PAUSE_RULES.get(self.pause_rule)
        if (
            expected_pause is not None
            and self.pause is not None
            and self.pause is not expected_pause
        ):
            msg = (
                f"pause={self.pause!r} contradicts pause_rule={self.pause_rule.value!r}; the "
                "coarse pause flag marks a pause added on top of the competition rules"
            )
            raise ValueError(msg)
        if (
            self.specificity_level
            in (
                SpecificityLevel.COMPETITION_LIFT,
                SpecificityLevel.COMPETITION_VARIATION,
            )
            and self.parent_lift not in COMPETITION_PARENT_LIFTS
        ):
            msg = (
                f"specificity_level={self.specificity_level.value!r} requires parent_lift to be "
                f"one of squat/bench/deadlift; got {self.parent_lift.value!r}"
            )
            raise ValueError(msg)
        if self.parent_lift is ParentLift.ACCESSORY and self.specificity_level not in (
            SpecificityLevel.SPORT_SPECIFIC,
            SpecificityLevel.GENERAL_STRENGTH,
            SpecificityLevel.ACCESSORY,
        ):
            msg = (
                "An exercise in the accessory family cannot be a competition lift or a "
                f"competition variation; got specificity_level="
                f"{self.specificity_level.value!r}"
            )
            raise ValueError(msg)
        if (
            self.specificity_level is SpecificityLevel.COMPETITION_LIFT
            and self.range_of_motion is not RangeOfMotion.COMPETITION
        ):
            msg = (
                "A competition lift is defined by the depth its rules require, so it must record "
                f"range_of_motion=competition; got {self.range_of_motion.value!r}"
            )
            raise ValueError(msg)
        if self.pause is True and self.pause_rule is PauseRule.UNKNOWN:
            msg = (
                "pause=True requires a pause_rule; a coarse pause flag without the rule that "
                "produced it cannot distinguish a 2-count from a long pause"
            )
            raise ValueError(msg)
        return self


class ExerciseAliasRecord(ContextRecord):
    """One source-specific exercise label bound to a canonical exercise.

    The alias layer is what makes heterogeneous logs interoperable without
    pretending the exercises are equivalent: ``Close Grip Bench Press`` and
    ``Competition Bench Press`` both live here, both point at *different*
    ``exercise_id`` values, and both keep their verbatim ``alias_raw`` spelling.

    ``source_system`` is the namespace the alias belongs to, and it is free text
    on purpose: a future adapter may contribute a namespace PSD has never seen
    without changing the canonical schema or any existing exercise identity.

    ``mapping_version`` and ``ontology_version`` record what the binding was made
    against, so a consumer can refuse an alias registry it cannot honour instead of
    silently reinterpreting it.
    """

    exercise_alias_id: str = Field(min_length=1, max_length=160)
    exercise_id: str = Field(min_length=1, max_length=160)
    source_system: str = Field(min_length=1, max_length=64, pattern=SOURCE_SYSTEM_PATTERN.pattern)
    alias_raw: str = Field(min_length=1, max_length=512)
    alias_normalized: str = Field(min_length=1, max_length=512)
    mapping_status: ResolutionStatus = ResolutionStatus.RESOLVED_ALIAS
    mapping_version: str = Field(min_length=1, max_length=64)
    ontology_version: str = Field(min_length=1, max_length=64)
    note: str | None = Field(default=None, max_length=512)

    @model_validator(mode="after")
    def _check_consistency(self) -> Self:
        """Reject an alias row that claims a resolution its own fields contradict."""
        if self.alias_normalized != self.alias_normalized.strip():
            msg = (
                f"alias_normalized {self.alias_normalized!r} has surrounding whitespace; the "
                "normalized form is compared and indexed, so it may not carry padding"
            )
            raise ValueError(msg)
        if self.mapping_status not in _ALIAS_ROW_STATUSES:
            msg = (
                f"mapping_status={self.mapping_status.value!r} cannot describe an alias row; an "
                "alias row is a resolved binding by construction, and unresolved labels are "
                "recorded in exercise_normalization"
            )
            raise ValueError(msg)
        if self.mapping_status is ResolutionStatus.EXACT_CANONICAL and self.alias_raw != (
            self.alias_normalized
        ):
            msg = (
                "mapping_status=exact_canonical requires alias_raw to equal alias_normalized; a "
                "spelling variant is a known alias, not the canonical identity itself"
            )
            raise ValueError(msg)
        return self


_RESOLVED_STATUSES: Final[frozenset[ResolutionStatus]] = frozenset(
    {ResolutionStatus.EXACT_CANONICAL, ResolutionStatus.RESOLVED_ALIAS}
)

_ALIAS_ROW_STATUSES: Final[frozenset[ResolutionStatus]] = frozenset(
    {ResolutionStatus.EXACT_CANONICAL, ResolutionStatus.RESOLVED_ALIAS}
)

#: Which resolution status each resolution method is allowed to produce.
#: Enforced so a method can never be recorded as the justification for an outcome it
#: could not have produced, which is what makes the method column auditable.
STATUS_BY_RESOLUTION_METHOD: Final[Mapping[ResolutionMethod, frozenset[ResolutionStatus]]] = {
    ResolutionMethod.CANONICAL_IDENTITY: frozenset({ResolutionStatus.EXACT_CANONICAL}),
    ResolutionMethod.REGISTERED_ALIAS: frozenset({ResolutionStatus.RESOLVED_ALIAS}),
    ResolutionMethod.CROSS_SOURCE_ALIAS: frozenset({ResolutionStatus.RESOLVED_ALIAS}),
    ResolutionMethod.STRUCTURED_INTERPRETATION: frozenset(
        {
            ResolutionStatus.RESOLVED_ALIAS,
            ResolutionStatus.AMBIGUOUS,
        }
    ),
    ResolutionMethod.QUESTION_FORM: frozenset({ResolutionStatus.AMBIGUOUS}),
    ResolutionMethod.GENERIC_QUALIFIER: frozenset({ResolutionStatus.PARTIAL_FAMILY}),
    ResolutionMethod.FAMILY_KEYWORD: frozenset({ResolutionStatus.PARTIAL_FAMILY}),
    ResolutionMethod.CURATED_AMBIGUOUS: frozenset(
        {
            ResolutionStatus.PARTIAL_FAMILY,
            ResolutionStatus.AMBIGUOUS,
            ResolutionStatus.UNMAPPED,
        }
    ),
    ResolutionMethod.NO_MATCH: frozenset({ResolutionStatus.UNMAPPED}),
}


def _validate_resolution_ladder(record: ExerciseNormalizationRecord) -> None:
    """Check an outcome never claims more than it established.

    The ladder has one rule: an outcome that did not identify exactly one canonical
    exercise must not carry that exercise, an alias reference, or an ambiguity reason
    that implies it did. Every check below exists to make overstating an outcome
    impossible rather than merely discouraged.
    """
    status = record.resolution_status
    resolved = status in _RESOLVED_STATUSES

    if resolved != (record.exercise_id is not None):
        verb = "requires" if resolved else "may not carry"
        msg = (
            f"resolution_status={status.value!r} {verb} an exercise_id; naming a canonical "
            "exercise would be a mapping this outcome did not make"
        )
        raise ValueError(msg)
    if resolved and record.candidate_exercise_ids:
        msg = (
            "A resolved outcome names one exercise and therefore has no candidates; listing "
            "candidates alongside a resolution would present the refusal and the answer together"
        )
        raise ValueError(msg)
    if resolved and record.ambiguity_reason is not None:
        msg = "A resolved outcome has no ambiguity reason."
        raise ValueError(msg)
    if status is ResolutionStatus.PARTIAL_FAMILY and record.parent_lift is ParentLift.UNKNOWN:
        msg = (
            "resolution_status=partial_family must name the parent lift it did resolve; without "
            "one it is indistinguishable from unmapped"
        )
        raise ValueError(msg)
    if not resolved and record.candidate_exercise_ids and status is not ResolutionStatus.AMBIGUOUS:
        msg = (
            f"resolution_status={status.value!r} is not an ambiguity, so it has no canonical "
            "candidates"
        )
        raise ValueError(msg)
    if (status is ResolutionStatus.AMBIGUOUS) != (bool(record.candidate_exercise_ids)):
        msg = (
            "resolution_status=ambiguous requires at least one candidate exercise, and no other "
            "status may carry one; otherwise a resolved result would be hidden inside a refusal"
        )
        raise ValueError(msg)


def _validate_resolution_evidence(record: ExerciseNormalizationRecord) -> None:
    """Check the cited alias reference matches the resolution outcome."""
    resolved = record.resolution_status in _RESOLVED_STATUSES
    if record.source_alias_id is not None and not resolved:
        msg = (
            "source_alias_id points at a registered alias, so it is only meaningful for a "
            "resolved outcome"
        )
        raise ValueError(msg)


def _validate_resolution_method(record: ExerciseNormalizationRecord) -> None:
    """Check the recorded method could have produced the recorded status."""
    permitted = STATUS_BY_RESOLUTION_METHOD.get(record.resolution_method)
    if permitted is None:
        msg = f"Unknown resolution method {record.resolution_method!r}."
        raise ValueError(msg)
    if record.resolution_status not in permitted:
        allowed = ", ".join(sorted(item.value for item in permitted))
        msg = (
            f"resolution_method={record.resolution_method.value!r} can produce "
            f"[{allowed}]; got resolution_status={record.resolution_status.value!r}"
        )
        raise ValueError(msg)


class ExerciseNormalizationRecord(ContextRecord):
    """The outcome of normalizing one raw source exercise label.

    This is the audit record for every mapping decision PSD makes about a
    heterogeneous label, including the ones it refuses to make. Its whole purpose
    is that the *refusals* are as inspectable as the successes:

    * ``raw_label`` is always the verbatim source string, never rewritten;
    * ``resolution_status`` separates an exact identity from a known alias, a
      family-only reading, an ambiguity, and an honest "not mapped";
    * ``candidate_exercise_ids`` records the defensible readings PSD declined to
      choose between, sorted so the list is deterministic;
    * ``ambiguity_reason`` says *why* no canonical exercise was forced, so a
      reviewer can tell "the source was vague" apart from "PSD has never seen this
      label".

    The evidence for a mapping is entirely symbolic: ``resolution_method`` names the
    stage that produced it, ``source_alias_id`` cites the alias row a lookup matched,
    and the two version columns name the ontology and registry that were in force.
    There is deliberately **no** numeric mapping score. A float such as ``0.8`` would
    read as a calibrated probability without ever having been calibrated against
    held-out labels, so the contract omits the field rather than inventing a
    calibration PSD cannot perform.
    """

    normalization_id: str = Field(min_length=1, max_length=160)
    raw_label: str = Field(min_length=1, max_length=512)
    normalized_label: str = Field(min_length=1, max_length=512)
    normalization_rules: tuple[str, ...] = ()
    source_system: str = Field(min_length=1, max_length=64, pattern=SOURCE_SYSTEM_PATTERN.pattern)
    source_alias_id: str | None = Field(default=None, max_length=160)
    resolution_status: ResolutionStatus
    resolution_method: ResolutionMethod
    exercise_id: str | None = Field(default=None, max_length=160)
    parent_lift: ParentLift = ParentLift.UNKNOWN
    candidate_exercise_ids: tuple[str, ...] = ()
    ambiguity_reason: AmbiguityReason | None = None
    mapping_version: str = Field(min_length=1, max_length=64)
    ontology_version: str = Field(min_length=1, max_length=64)
    note: str | None = Field(default=None, max_length=512)

    @model_validator(mode="after")
    def _check_consistency(self) -> Self:
        """Enforce the resolution ladder, so an outcome cannot overstate itself."""
        object.__setattr__(
            self, "candidate_exercise_ids", tuple(sorted(set(self.candidate_exercise_ids)))
        )
        object.__setattr__(
            self, "normalization_rules", tuple(sorted(set(self.normalization_rules)))
        )
        _validate_resolution_ladder(self)
        _validate_resolution_evidence(self)
        _validate_resolution_method(self)
        return self


# --------------------------------------------------------------------------
# programming (prescription only)
# --------------------------------------------------------------------------


class ProgramRecord(ContextRecord):
    """A named training program."""

    program_id: str = Field(min_length=1, max_length=160)
    name: str = Field(min_length=1, max_length=256)
    description: str | None = Field(default=None, max_length=2048)


class ProgramVersionRecord(ContextRecord):
    """An immutable version of a program, effective over an interval.

    Versioning prescriptions is what makes "which plan was known when" answerable
    for an autoregulated program that changed mid-block.
    """

    program_version_id: str = Field(min_length=1, max_length=160)
    program_id: str = Field(min_length=1, max_length=160)
    version_label: str = Field(min_length=1, max_length=64)
    effective_from: datetime
    effective_to: datetime | None = None
    authored_at: datetime | None = None
    change_summary: str | None = Field(default=None, max_length=2048)
    supersedes_program_version_id: str | None = Field(default=None, max_length=160)

    @model_validator(mode="after")
    def _check_interval(self) -> Self:
        for name in ("effective_from", "effective_to", "authored_at"):
            object.__setattr__(self, name, normalize_timestamp(getattr(self, name), field=name))
        if self.effective_to is not None and self.effective_to <= self.effective_from:
            msg = "effective_to must be strictly after effective_from."
            raise ValueError(msg)
        return self


class ProgramModificationRecord(EventRecord):
    """A change to a prescription, stamped with when it became known.

    ``known_at`` is the leakage-critical field: an autoregulated change is only
    knowable from the moment it was selected, so it may never be exposed to a
    benchmark episode before that instant.
    """

    program_modification_id: str = Field(min_length=1, max_length=160)
    program_version_id: str | None = Field(default=None, max_length=160)
    athlete_id: str | None = Field(default=None, max_length=160)
    planned_session_id: str | None = Field(default=None, max_length=160)
    known_at: datetime
    change_kind: ProgramModificationKind
    field_changed: str | None = Field(default=None, max_length=128)
    previous_value: str | None = Field(default=None, max_length=1024)
    new_value: str | None = Field(default=None, max_length=1024)
    value_encoding: ValueEncoding = ValueEncoding.TEXT
    rationale: str | None = Field(default=None, max_length=2048)

    @model_validator(mode="after")
    def _check_consistency(self) -> Self:
        object.__setattr__(self, "known_at", normalize_timestamp(self.known_at, field="known_at"))
        if self.previous_value is None and self.new_value is None:
            msg = "A program modification must record at least one of previous_value/new_value."
            raise ValueError(msg)
        if self.athlete_id is None and self.planned_session_id is None:
            msg = "A program modification must be scoped to an athlete or a planned session."
            raise ValueError(msg)
        return self


class PlannedSessionRecord(EventRecord):
    """A scheduled session: prescription, not execution.

    A skipped session is recorded here with a reason. PSD never creates a
    performed record for it, and never treats the skip as a zero-load session.
    """

    planned_session_id: str = Field(min_length=1, max_length=160)
    athlete_id: str = Field(min_length=1, max_length=160)
    program_version_id: str | None = Field(default=None, max_length=160)
    session_order_index: int = Field(default=0, ge=0)
    session_status: SessionStatus = SessionStatus.PLANNED
    not_performed_reason: NotPerformedReason | None = None
    template_label: str | None = Field(default=None, max_length=128)

    @model_validator(mode="after")
    def _check_consistency(self) -> Self:
        if self.scheduled_at is None:
            msg = "planned_session.scheduled_at is required."
            raise ValueError(msg)
        needs_reason = {
            SessionStatus.SKIPPED,
            SessionStatus.CANCELLED,
        }
        if self.session_status in needs_reason and self.not_performed_reason is None:
            msg = (
                f"session_status {self.session_status.value!r} requires not_performed_reason; "
                "an unexplained skip must be recorded as unknown, not omitted."
            )
            raise ValueError(msg)
        if self.session_status not in needs_reason and self.not_performed_reason is not None:
            msg = "not_performed_reason is only meaningful for skipped or cancelled sessions."
            raise ValueError(msg)
        return self


class PlannedExerciseRecord(EventRecord):
    """A prescribed exercise within a planned session."""

    planned_exercise_id: str = Field(min_length=1, max_length=160)
    planned_session_id: str = Field(min_length=1, max_length=160)
    exercise_id: str = Field(min_length=1, max_length=160)
    ordinal: int = Field(ge=0)
    supersedes_planned_exercise_id: str | None = Field(default=None, max_length=160)


class PlannedSetRecord(EventRecord):
    """A prescribed set: targets only.

    No field on this record describes what actually happened. ``prescription !=
    execution`` is enforced structurally by the absence of any executed column,
    which the schema contract test asserts.
    """

    planned_set_id: str = Field(min_length=1, max_length=160)
    planned_exercise_id: str = Field(min_length=1, max_length=160)
    ordinal: int = Field(ge=0)
    target_reps: int | None = Field(default=None, ge=0, le=1000)
    target_reps_min: int | None = Field(default=None, ge=0, le=1000)
    target_reps_max: int | None = Field(default=None, ge=0, le=1000)
    target_load_raw: float | None = None
    target_load_unit: str | None = Field(default=None, max_length=16)
    target_load_kg: float | None = None
    target_rpe: float | None = None
    target_rir: float | None = None
    target_percent_one_rm: float | None = None
    target_duration_seconds: float | None = Field(default=None, gt=0.0)
    rest_target_seconds: float | None = Field(default=None, ge=0.0)
    prescription_basis: PrescriptionBasis = PrescriptionBasis.UNKNOWN

    @model_validator(mode="after")
    def _check_consistency(self) -> Self:
        has_single = self.target_reps is not None
        has_range = self.target_reps_min is not None or self.target_reps_max is not None
        if has_single and has_range:
            msg = "Record either target_reps or a target_reps_min/max range, not both."
            raise ValueError(msg)
        if has_range and (self.target_reps_min is None or self.target_reps_max is None):
            msg = "A rep range requires both target_reps_min and target_reps_max."
            raise ValueError(msg)
        if (
            has_range
            and self.target_reps_min is not None
            and self.target_reps_max is not None
            and self.target_reps_min > self.target_reps_max
        ):
            msg = "target_reps_min must not exceed target_reps_max."
            raise ValueError(msg)
        _validate_mass_triplet(
            raw_value=self.target_load_raw,
            raw_unit=self.target_load_unit,
            normalized=self.target_load_kg,
            prefix="planned_set.target_load",
        )
        _validate_rpe(self.target_rpe, prefix="planned_set")
        _validate_rir(self.target_rir, prefix="planned_set")
        if self.target_percent_one_rm is not None and not (
            PERCENT_MIN < self.target_percent_one_rm <= PERCENT_MAX
        ):
            msg = f"target_percent_one_rm {self.target_percent_one_rm!r} must be in (0, 100]."
            raise ValueError(msg)
        return self


# --------------------------------------------------------------------------
# execution
# --------------------------------------------------------------------------


class PerformedSessionRecord(EventRecord):
    """A session that actually happened.

    ``planned_session_id`` is null when no prescription existed. That null is the
    whole point: PSD must not manufacture an ex-ante plan from an export that
    only ever recorded execution.
    """

    performed_session_id: str = Field(min_length=1, max_length=160)
    athlete_id: str = Field(min_length=1, max_length=160)
    planned_session_id: str | None = Field(default=None, max_length=160)
    session_type: SessionType = SessionType.TRAINING
    started_at: datetime
    ended_at: datetime | None = None
    session_order_index: int = Field(default=0, ge=0)
    duration_seconds: float | None = Field(default=None, ge=0.0)
    is_completed: bool | None = None

    @model_validator(mode="after")
    def _check_consistency(self) -> Self:
        object.__setattr__(
            self, "started_at", normalize_timestamp(self.started_at, field="started_at")
        )
        if self.ended_at is not None:
            object.__setattr__(
                self, "ended_at", normalize_timestamp(self.ended_at, field="ended_at")
            )
            if self.ended_at < self.started_at:
                msg = "ended_at must not precede started_at."
                raise ValueError(msg)
        if (
            self.ended_at is not None
            and self.duration_seconds is not None
            and self.duration_seconds < 0.0
        ):
            msg = "duration_seconds must not be negative."
            raise ValueError(msg)
        return self


class PerformedExerciseRecord(EventRecord):
    """An exercise that was actually performed."""

    performed_exercise_id: str = Field(min_length=1, max_length=160)
    performed_session_id: str = Field(min_length=1, max_length=160)
    exercise_id: str = Field(min_length=1, max_length=160)
    ordinal: int = Field(ge=0)
    planned_exercise_id: str | None = Field(default=None, max_length=160)
    is_substitution: bool | None = None
    substituted_from_exercise_id: str | None = Field(default=None, max_length=160)

    @model_validator(mode="after")
    def _check_consistency(self) -> Self:
        if self.is_substitution is True and self.substituted_from_exercise_id is None:
            msg = "A substitution must name the exercise it replaced."
            raise ValueError(msg)
        if self.is_substitution is not True and self.substituted_from_exercise_id is not None:
            msg = "substituted_from_exercise_id is only meaningful when is_substitution is true."
            raise ValueError(msg)
        return self


class PerformedSetRecord(EventRecord):
    """A set that was actually performed.

    ``reps_performed`` may legitimately be ``0`` for a failed set: zero there is a
    recorded observation, not a missing-value sentinel. A set that was never
    performed has no record here at all.
    """

    performed_set_id: str = Field(min_length=1, max_length=160)
    performed_exercise_id: str = Field(min_length=1, max_length=160)
    ordinal: int = Field(ge=0)
    planned_set_id: str | None = Field(default=None, max_length=160)
    set_status: SetStatus = SetStatus.UNKNOWN
    load_raw: float | None = None
    load_unit: str | None = Field(default=None, max_length=16)
    load_kg: float | None = None
    reps_performed: int | None = Field(default=None, ge=0, le=1000)
    reps_failed: int | None = Field(default=None, ge=0, le=1000)
    is_failure: bool | None = None
    rpe: float | None = None
    rir: float | None = None
    tempo_actual: str | None = Field(default=None, max_length=64)
    duration_seconds: float | None = Field(default=None, ge=0.0)
    rest_actual_seconds: float | None = Field(default=None, ge=0.0)
    is_warmup: bool | None = None
    rep_level_data_available: bool = False
    incomplete_reason: MissingnessReason | None = None

    @model_validator(mode="after")
    def _check_consistency(self) -> Self:
        _validate_mass_triplet(
            raw_value=self.load_raw,
            raw_unit=self.load_unit,
            normalized=self.load_kg,
            prefix="performed_set.load",
        )
        _validate_rpe(self.rpe, prefix="performed_set")
        _validate_rir(self.rir, prefix="performed_set")
        if self.reps_performed is None and self.set_status is not SetStatus.UNKNOWN:
            object.__setattr__(
                self,
                "incomplete_reason",
                self.incomplete_reason or MissingnessReason.NOT_RECORDED_IN_SOURCE,
            )
        if self.is_failure is True and self.reps_performed is not None and self.reps_performed > 0:
            msg = (
                "is_failure is true but reps_performed is positive; a failed set records the "
                "repetitions completed before the failure, and the failure is captured by "
                "is_failure/reps_failed."
            )
            raise ValueError(msg)
        return self


class PerformedRepRecord(EventRecord):
    """A single performed repetition, when rep-level data exist.

    Rep-level data are frequently absent. Absence is expressed by the absence of
    these rows together with ``rep_level_data_available`` on the parent set, never
    by rows with zeroed values.
    """

    performed_rep_id: str = Field(min_length=1, max_length=160)
    performed_set_id: str = Field(min_length=1, max_length=160)
    rep_ordinal: int = Field(ge=1, le=1000)
    rep_status: RepStatus = RepStatus.UNKNOWN
    is_success: bool | None = None
    load_raw: float | None = None
    load_unit: str | None = Field(default=None, max_length=16)
    load_kg: float | None = None
    rpe: float | None = None
    velocity_mps: float | None = Field(default=None, gt=0.0)

    @model_validator(mode="after")
    def _check_consistency(self) -> Self:
        _validate_mass_triplet(
            raw_value=self.load_raw,
            raw_unit=self.load_unit,
            normalized=self.load_kg,
            prefix="performed_rep.load",
        )
        _validate_rpe(self.rpe, prefix="performed_rep")
        if self.is_success is False and self.rep_status is RepStatus.COMPLETED:
            msg = "is_success false contradicts rep_status completed."
            raise ValueError(msg)
        if self.is_success is True and self.rep_status is RepStatus.FAILED:
            msg = "is_success true contradicts rep_status failed."
            raise ValueError(msg)
        return self


# --------------------------------------------------------------------------
# observations
# --------------------------------------------------------------------------


class ObservationRecord(EventRecord):
    """A reported measurement, scoped to whatever level the source recorded it.

    ``numeric_value`` and ``text_value`` are separate columns so that a numeric
    reading is never flattened into text. Exactly one of them is populated when
    the observation has a payload.
    """

    observation_id: str = Field(min_length=1, max_length=160)
    athlete_id: str = Field(min_length=1, max_length=160)
    observation_type: ObservationType
    observation_method: ObservationMethod = ObservationMethod.UNKNOWN
    observation_scope: ObservationScope = ObservationScope.ATHLETE
    reporter_role: ReporterRole = ReporterRole.UNKNOWN
    instrument: str | None = Field(default=None, max_length=128)
    numeric_value: float | None = None
    text_value: str | None = Field(default=None, max_length=1024)
    unit: str | None = Field(default=None, max_length=32)
    parent_session_id: str | None = Field(default=None, max_length=160)
    parent_exercise_id: str | None = Field(default=None, max_length=160)
    parent_set_id: str | None = Field(default=None, max_length=160)
    parent_rep_id: str | None = Field(default=None, max_length=160)
    parent_competition_id: str | None = Field(default=None, max_length=160)

    @model_validator(mode="after")
    def _check_consistency(self) -> Self:
        if self.observed_at is None:
            msg = (
                "observation.observed_at is required: an unplaced observation has no "
                "position on the event timeline."
            )
            raise ValueError(msg)
        if self.numeric_value is not None and self.text_value is not None:
            msg = "An observation carries either numeric_value or text_value, not both."
            raise ValueError(msg)
        scope_to_parent: dict[ObservationScope, str] = {
            ObservationScope.SESSION: "parent_session_id",
            ObservationScope.EXERCISE: "parent_exercise_id",
            ObservationScope.SET: "parent_set_id",
            ObservationScope.REP: "parent_rep_id",
            ObservationScope.COMPETITION: "parent_competition_id",
        }
        expected_field = scope_to_parent.get(self.observation_scope)
        parent_fields = set(scope_to_parent.values())
        if expected_field is None:
            stray = [name for name in parent_fields if getattr(self, name) is not None]
            if stray:
                msg = (
                    f"observation_scope {self.observation_scope.value!r} does not scope to a "
                    f"parent record, but {', '.join(sorted(stray))} is set."
                )
                raise ValueError(msg)
        elif getattr(self, expected_field) is None:
            msg = f"observation_scope {self.observation_scope.value!r} requires {expected_field}."
            raise ValueError(msg)
        elif self.observation_type is ObservationType.RPE:
            _validate_rpe(self.numeric_value, prefix="observation")
        elif self.observation_type is ObservationType.RIR:
            _validate_rir(self.numeric_value, prefix="observation")
        return self


class PerformanceTestRecord(EventRecord):
    """A structured performance test.

    A test result is an observation of what was measured on a day. It is not an
    estimate of latent capacity: a maximal single, a meet best, and an estimated
    1RM are different observables and PSD keeps them distinct.
    """

    performance_test_id: str = Field(min_length=1, max_length=160)
    athlete_id: str = Field(min_length=1, max_length=160)
    test_type: TestType
    exercise_id: str | None = Field(default=None, max_length=160)
    performed_set_id: str | None = Field(default=None, max_length=160)
    competition_id: str | None = Field(default=None, max_length=160)
    protocol_label: str | None = Field(default=None, max_length=256)
    result_metric: str = Field(min_length=1, max_length=128)
    result_raw: float | None = None
    result_unit: str | None = Field(default=None, max_length=32)
    result_normalized: float | None = None
    result_unit_normalized: str | None = Field(default=None, max_length=32)

    @model_validator(mode="after")
    def _check_consistency(self) -> Self:
        if self.observed_at is None:
            msg = "performance_test.observed_at is required."
            raise ValueError(msg)
        if self.result_raw is not None and self.result_normalized is None:
            msg = "A raw test result requires a normalized result."
            raise ValueError(msg)
        if self.result_normalized is not None and self.result_raw is None:
            msg = (
                "PSD never synthesizes a test result: a normalized value without a raw "
                "source value is rejected."
            )
            raise ValueError(msg)
        return self


class VelocityObservationRecord(EventRecord):
    """A bar-velocity measurement, where the instrument and context are recorded.

    Velocity is not comparable across instruments, so ``method`` is mandatory
    rather than optional metadata.
    """

    velocity_observation_id: str = Field(min_length=1, max_length=160)
    athlete_id: str = Field(min_length=1, max_length=160)
    performed_set_id: str | None = Field(default=None, max_length=160)
    performed_rep_id: str | None = Field(default=None, max_length=160)
    method: VelocityMethod = VelocityMethod.UNKNOWN
    mean_velocity_mps: float | None = Field(default=None, gt=0.0, le=20.0)
    peak_velocity_mps: float | None = Field(default=None, gt=0.0, le=30.0)
    velocity_loss_percent: float | None = Field(default=None, ge=0.0, le=100.0)
    load_raw: float | None = None
    load_unit: str | None = Field(default=None, max_length=16)
    load_kg: float | None = None
    sampling_hz: float | None = Field(default=None, gt=0.0, le=1000.0)

    @model_validator(mode="after")
    def _check_consistency(self) -> Self:
        if self.observed_at is None:
            msg = "velocity_observation.observed_at is required."
            raise ValueError(msg)
        _validate_mass_triplet(
            raw_value=self.load_raw,
            raw_unit=self.load_unit,
            normalized=self.load_kg,
            prefix="velocity_observation.load",
        )
        if self.performed_set_id is None and self.performed_rep_id is None:
            object.__setattr__(
                self,
                "missingness_reason",
                self.missingness_reason or MissingnessReason.NOT_APPLICABLE,
            )
        return self


# --------------------------------------------------------------------------
# competition
# --------------------------------------------------------------------------


class CompetitionMeetRecord(ContextRecord):
    """One competition, as a meet.

    A meet is context rather than an outcome: it has an identity, a start date, a
    hosting body, a sanctioning body, and a sanctioned/unsanctioned status, none of
    which is a thing an athlete performed. The lifter's outcome at that meet lives
    on :class:`CompetitionRecord`.

    Two source facts drive most of this record's shape, and both are easy to lose:

    * A meet that runs over more than one day publishes only its **start** date, so
      ``event_time_precision`` is ``DATE_ONLY`` and nothing here claims to know when
      a particular lifter walked on. Two meets may share a start date; that is a fact
      about the source, not a defect to be repaired.
    * A meet is hosted by one body and sanctioned by another. Those are separate
      fields and stay separate: the hosting federation is frequently an affiliate
      rather than the top-level sanctioning body.

    ``sanctioned_status_raw`` keeps the source's own wording beside the parsed
    boolean. "Sanctioned" is the source's competition status; it is not a quality
    score, and PSD does not rank meets by it.
    """

    competition_meet_id: str = Field(min_length=1, max_length=160)
    meet_name: str = Field(min_length=1, max_length=512)
    #: The meet's start date, as a date-only UTC instant.
    meet_date: datetime
    event_time_precision: EventTimePrecision = EventTimePrecision.DATE_ONLY
    #: The federation that hosted the meet.
    meet_federation: str = Field(min_length=1, max_length=128)
    #: The top-level body that sanctioned the meet, when the source states one.
    meet_parent_federation: str | None = Field(default=None, max_length=128)
    meet_country: str | None = Field(default=None, max_length=128)
    meet_state: str | None = Field(default=None, max_length=128)
    sanctioned_status_raw: str | None = Field(default=None, max_length=64)
    is_sanctioned: bool | None = None

    @model_validator(mode="after")
    def _check_consistency(self) -> Self:
        object.__setattr__(
            self, "meet_date", normalize_timestamp(self.meet_date, field="meet_date")
        )
        if self.is_sanctioned is not None and self.sanctioned_status_raw is None:
            msg = (
                "competition_meet: a parsed is_sanctioned requires sanctioned_status_raw, so a "
                "reader can see the source's own wording rather than only PSD's reading of it"
            )
            raise ValueError(msg)
        return self


class CompetitionRecord(EventRecord):
    """One athlete's result at one competition.

    Everything here is what the *source recorded about this lifter at this meet*.
    Meet-level facts are stated once on :class:`CompetitionMeetRecord`, which
    ``competition_meet_id`` points at; this record repeats the ones that a reader
    would otherwise have to join for, and a cross-record rule requires the two
    views to agree.

    Four columns carry most of the source's meaning and are worth reading together:

    ``equipment_class`` / ``equipment_class_raw``
        The competition equipment **category**, i.e. the equipment the rules allowed.
        A lifter in a wrap-free-permitting category may still have competed in
        knee sleeves, and PSD does not claim they wore anything: no athlete-level
        equipment observation is derived from this field.
    ``age_reported`` / ``age_precision``
        A reported age, with its precision stated. An approximate age is known to
        lie between two integers; rounding it would invent a precision the source
        never had, and no birth date or birth year is inferred from it.
    ``participation_place`` / ``participation_status`` / ``participation_status_kind``
        The placing, when there is one. Sources also report guest, disqualified,
        doping-disqualified, and no-show codes, which are *not* placings and are
        never coerced into numbers.
    ``is_drug_tested_category``
        Whether the result belongs to a drug-tested competition category. It says
        nothing about whether this individual was tested -- federations do not
        publish which lifters were -- so it is a property of the category, not of
        the athlete.
    """

    competition_id: str = Field(min_length=1, max_length=160)
    athlete_id: str = Field(min_length=1, max_length=160)
    competition_meet_id: str | None = Field(default=None, max_length=160)
    competition_date: datetime
    event_time_precision: EventTimePrecision = EventTimePrecision.UNKNOWN
    #: Which competition event was entered: which lifts were contested at all.
    competition_event: CompetitionEvent | None = None
    name: str | None = Field(default=None, max_length=512)
    federation: str | None = Field(default=None, max_length=128)
    sanctioning_body: str | None = Field(default=None, max_length=128)
    location: str | None = Field(default=None, max_length=512)
    equipment_class_raw: str | None = Field(default=None, max_length=64)
    equipment_class: EquipmentClass = EquipmentClass.UNKNOWN
    weight_class_raw: str | None = Field(default=None, max_length=64)
    bodyweight_raw: float | None = None
    bodyweight_unit: str | None = Field(default=None, max_length=16)
    bodyweight_kg: float | None = None
    #: The source's participation code, verbatim, whether numeric or not.
    participation_status: str | None = Field(default=None, max_length=64)
    #: The numeric placing, present only when the lifter was actually placed.
    participation_place: int | None = Field(default=None, ge=1)
    participation_status_kind: ParticipationStatus = ParticipationStatus.UNKNOWN
    #: Whether the meet sanctioned this result under a drug-tested category.
    is_drug_tested_category: bool | None = None
    #: The source-reported age. A half-integer age is approximate, not rounded.
    age_reported: float | None = None
    age_precision: AgePrecision | None = None
    #: The source's explicit age-class label, when it published one.
    age_class_raw: str | None = Field(default=None, max_length=64)
    #: The source's birth-year class, a different bucket from the age class.
    birth_year_class_raw: str | None = Field(default=None, max_length=64)
    #: Free-form division text. Never mined for an age: dedicated age fields exist.
    division_raw: str | None = Field(default=None, max_length=256)
    #: The lifter's own country/state as reported for this result, verbatim.
    athlete_country_raw: str | None = Field(default=None, max_length=128)
    athlete_region_raw: str | None = Field(default=None, max_length=128)
    is_championship: bool | None = None

    @model_validator(mode="after")
    def _check_consistency(self) -> Self:
        object.__setattr__(
            self,
            "competition_date",
            normalize_timestamp(self.competition_date, field="competition_date"),
        )
        _validate_mass_triplet(
            raw_value=self.bodyweight_raw,
            raw_unit=self.bodyweight_unit,
            normalized=self.bodyweight_kg,
            prefix="competition.bodyweight",
        )
        if (self.participation_place is None) != (
            self.participation_status_kind is not ParticipationStatus.PLACED
        ):
            msg = (
                "competition: participation_place is present exactly when "
                "participation_status_kind is 'placed'; a guest, disqualified, "
                "doping-disqualified, or no-show result has no placing to record, and a "
                "placing that exists without a placed status contradicts itself"
            )
            raise ValueError(msg)
        if (self.age_reported is None) != (self.age_precision is None):
            msg = (
                "competition: age_reported and age_precision must be recorded together; an "
                "age with no stated precision would let an approximation read as exact"
            )
            raise ValueError(msg)
        if self.age_reported is not None and self.age_reported < 0.0:
            msg = f"competition: age_reported {self.age_reported!r} must not be negative."
            raise ValueError(msg)
        if self.weight_class_raw is not None and not self.weight_class_raw.strip():
            msg = "competition: weight_class_raw must not be padding; use null instead."
            raise ValueError(msg)
        return self


class CompetitionAttemptRecord(EventRecord):
    """One competition attempt.

    Missing attempts stay missing: PSD creates a row only for attempts the source
    actually recorded. ``NO_ATTEMPT`` exists solely for sources that explicitly
    report an empty attempt slot (for example a withdrawal), and is distinct from
    ``BAD_LIFT``.

    Three columns together carry a source's signed-attempt encoding without ever
    storing a negative physical load:

    ``source_attempt_raw``
        The source value verbatim, sign included. Several sources encode success as a
        positive number and failure as that number negated, so the sign *is* the
        result and this column is the only place the original encoding survives.
    ``load_raw`` / ``load_kg``
        The magnitude that was physically on the bar: ``abs(source_attempt_raw)``, in
        kilograms. ``-200 kg`` is not a load anybody attempted.
    ``result``
        The sign, read as an outcome: ``GOOD_LIFT`` above zero, ``BAD_LIFT`` below.

    ``attempt_role`` separates the three attempts the rules count from a fourth
    attempt taken only for a single-lift record. A fourth attempt does not contribute
    to the total, and this contract will not let it be mistaken for one.
    """

    competition_attempt_id: str = Field(min_length=1, max_length=160)
    competition_id: str = Field(min_length=1, max_length=160)
    athlete_id: str = Field(min_length=1, max_length=160)
    lift: LiftType
    attempt_number: int | None = Field(default=None, ge=1, le=4)
    attempt_role: AttemptRole = AttemptRole.ORDERED
    attempt_order_basis: AttemptOrderBasis = AttemptOrderBasis.UNKNOWN
    attempt_time: datetime | None = None
    source_attempt_raw: float | None = None
    load_raw: float | None = None
    load_unit: str | None = Field(default=None, max_length=16)
    load_kg: float | None = None
    result: AttemptResult = AttemptResult.UNKNOWN
    is_opener: bool | None = None

    @model_validator(mode="after")
    def _check_consistency(self) -> Self:
        object.__setattr__(
            self, "attempt_time", normalize_timestamp(self.attempt_time, field="attempt_time")
        )
        if self.attempt_number is not None and self.attempt_number not in (
            *ORDERED_ATTEMPT_NUMBERS,
            RECORD_FOURTH_ATTEMPT_NUMBER,
        ):
            msg = (
                f"attempt_number must be one of "
                f"{(*ORDERED_ATTEMPT_NUMBERS, RECORD_FOURTH_ATTEMPT_NUMBER)}; got "
                f"{self.attempt_number!r}."
            )
            raise ValueError(msg)
        _validate_mass_triplet(
            raw_value=self.load_raw,
            raw_unit=self.load_unit,
            normalized=self.load_kg,
            prefix="competition_attempt.load",
        )
        _check_attempt_role(self.attempt_role, self.attempt_number)
        completed = {AttemptResult.GOOD_LIFT, AttemptResult.BAD_LIFT}
        if self.result in completed and self.load_kg is None:
            msg = f"A {self.result.value!r} attempt must record the load that was attempted."
            raise ValueError(msg)
        _check_signed_attempt(self.source_attempt_raw, self.result, self.load_raw)
        return self


def _check_attempt_role(role: AttemptRole, attempt_number: int | None) -> None:
    """Check the attempt role and the attempt number tell the same story."""
    if role is AttemptRole.RECORD_FOURTH and attempt_number != RECORD_FOURTH_ATTEMPT_NUMBER:
        msg = (
            f"attempt_role='record_fourth' requires attempt_number="
            f"{RECORD_FOURTH_ATTEMPT_NUMBER}; a record attempt is the fourth, and one "
            "numbered differently would be indistinguishable from an ordinary attempt"
        )
        raise ValueError(msg)
    if (
        role is AttemptRole.ORDERED
        and attempt_number is not None
        and attempt_number not in ORDERED_ATTEMPT_NUMBERS
    ):
        msg = (
            f"attempt_role='ordered' requires attempt_number in {ORDERED_ATTEMPT_NUMBERS}; "
            f"only attempt number {RECORD_FOURTH_ATTEMPT_NUMBER} is a record attempt and it "
            "must be labelled as one"
        )
        raise ValueError(msg)


def _check_signed_attempt(
    source_attempt_raw: float | None, result: AttemptResult, load_raw: float | None
) -> None:
    """Check the signed source value and the canonical load agree.

    This check *is* the reason both columns exist. A negative source value must be a
    bad lift, a positive one a good lift, and the physical load must be its magnitude.
    A source value of zero is not a load at all and is rejected rather than stored.
    """
    if source_attempt_raw is None:
        return
    if source_attempt_raw == 0.0:
        msg = (
            "competition_attempt: source_attempt_raw is 0.0, which is not a load. A source "
            "that has no attempt to report leaves the attempt absent; it does not report zero."
        )
        raise ValueError(msg)
    expected = AttemptResult.GOOD_LIFT if source_attempt_raw > 0 else AttemptResult.BAD_LIFT
    if result in {AttemptResult.GOOD_LIFT, AttemptResult.BAD_LIFT} and result is not expected:
        msg = (
            f"source_attempt_raw={source_attempt_raw!r} implies a {expected.value!r} attempt, "
            f"but result={result.value!r} says otherwise"
        )
        raise ValueError(msg)
    if load_raw is not None and abs(load_raw - abs(source_attempt_raw)) > NORMALIZATION_TOLERANCE:
        msg = (
            f"load_raw={load_raw!r} is not the magnitude of source_attempt_raw="
            f"{source_attempt_raw!r}; the attempted load is abs() of the source value and "
            "never its signed form"
        )
        raise ValueError(msg)


class CompetitionReportedResultRecord(EventRecord):
    """A reported or derived competition result such as a best lift or a total.

    ``is_derived`` marks values computed by a source or by PSD rather than observed.
    Reported bests and totals stay separate from the attempt observations they
    summarize, so a meet best is never mistaken for a measured latent 1RM. Nothing
    here is an estimate of latent capacity.

    Three columns keep the source's own numbers recoverable:

    ``source_value_raw``
        The value exactly as the source published it, sign and precision included.
    ``result_source_field``
        Which source column supplied the value. Several distinct scoring systems share
        this table and the field name is what tells them apart.
    ``reported_best_semantics``
        Required for every best lift, because a *negative* reported best is not a
        negative lift: a few federations publish the lowest weight a lifter attempted
        and failed. ``value`` holds the magnitude and this column states which of the
        two meanings applies, so neither reading can be inferred by accident.

    A source may report a total while omitting the component lifts entirely. PSD
    records the total it was given and never manufactures components to balance it.
    """

    competition_reported_result_id: str = Field(min_length=1, max_length=160)
    competition_id: str = Field(min_length=1, max_length=160)
    athlete_id: str = Field(min_length=1, max_length=160)
    result_kind: CompetitionResultKind
    value: float
    unit: str | None = Field(default=None, max_length=32)
    source_value_raw: float | None = None
    result_source_field: str | None = Field(default=None, max_length=64)
    reported_best_semantics: ReportedBestSemantics | None = None
    is_derived: bool = False
    derivation_note: str | None = Field(default=None, max_length=512)

    @model_validator(mode="after")
    def _check_consistency(self) -> Self:
        """Keep the recorded value and the recorded meaning from contradicting.

        The rule is symmetric with attempts: ``value`` is the magnitude and
        ``source_value_raw`` is the source's signed number. The sign lives only in the
        raw column, which is what makes "lowest weight attempted and failed"
        representable without ever storing a negative lift.
        """
        if self.source_value_raw == 0.0:
            msg = "competition_reported_result: source_value_raw 0.0 is not a reported value."
            raise ValueError(msg)
        if self.source_value_raw is not None and (
            abs(abs(self.source_value_raw) - abs(self.value)) > NORMALIZATION_TOLERANCE
        ):
            msg = (
                f"value={self.value!r} is not the magnitude of source_value_raw="
                f"{self.source_value_raw!r}"
            )
            raise ValueError(msg)
        if self.result_kind in REPORTED_BEST_KINDS:
            if self.value <= 0.0:
                msg = (
                    f"value={self.value!r} must be a positive magnitude; a reported best that "
                    "the source published with a negative sign is recorded as its magnitude "
                    "here, with the sign kept in source_value_raw"
                )
                raise ValueError(msg)
            if self.source_value_raw is not None and self.source_value_raw < 0.0:
                required = ReportedBestSemantics.FAILED_ATTEMPT_ONLY
                if self.reported_best_semantics is not required:
                    msg = (
                        f"source_value_raw={self.source_value_raw!r} is negative, which is how "
                        "a few federations publish the lowest weight a lifter attempted and "
                        "failed. Record reported_best_semantics="
                        f"{required.value!r}, because a negative successful lift does not exist"
                    )
                    raise ValueError(msg)
            if (
                self.reported_best_semantics is ReportedBestSemantics.FAILED_ATTEMPT_ONLY
                and self.source_value_raw is not None
                and self.source_value_raw > 0.0
            ):
                msg = (
                    f"reported_best_semantics='failed_attempt_only' contradicts "
                    f"source_value_raw={self.source_value_raw!r}: a positive reported best is "
                    "a successful lift. Only a source that publishes a negative best means the "
                    "lowest failed weight."
                )
                raise ValueError(msg)
        elif self.reported_best_semantics is not None:
            msg = (
                "reported_best_semantics is only meaningful for a reported best lift; "
                f"result_kind={self.result_kind.value!r} is not one"
            )
            raise ValueError(msg)
        return self
