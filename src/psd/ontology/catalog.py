"""The PSD exercise ontology: canonical exercises, source aliases, and refusals.

This module is *data*. It declares what an exercise is and how sources spell it,
and it holds no logic beyond the shape of the declarations.

Three tables live here
----------------------

``EXERCISES``
    The canonical exercise identities. Every entry is a set of observable
    descriptors -- parent lift, specificity, implement, bar, apparatus, stance,
    grip, range of motion, pause, tempo, laterality, and configuration flags. There
    is deliberately no transfer coefficient, no effectiveness number, and no
    similarity score: those are claims about athlete response that PSD leaves for
    models to learn.

``ALIASES``
    Source-specific label spellings bound to a canonical exercise. Aliases are
    where *spelling* variation lives; entities are where *semantic* variation
    lives. "Bench", "Bench Press", "Benchpress", and "Competition Bench" are four
    spellings of one entity, while "Close Grip Bench Press" is a different entity
    that happens to share a parent lift.

``CURATED_AMBIGUOUS_LABELS``
    Labels PSD has seen and is *refusing* to resolve: "Bench variation", "Machine
    press", "Squat machine". This table is the ambiguity policy made data. It
    exists so that raising coverage can never become the reason a vague label gets
    a canonical exercise.

A note on source systems
------------------------

``AliasSourceSystem`` namespaces mark where a spelling came from. The ``psd_registry``
and ``generic_csv``/``spreadsheet`` entries describe labels PSD can expect to meet
in an export. The entries under ``hevy``, ``strong`` and ``trainheroic`` are
*representative* label sets assembled from publicly documented powerlifting logging
vocabulary; they are **not** verified vendor exports, and the persisted alias rows
are stamped ``QualityFlag.UNVERIFIED`` to say so. Adapters must reconcile them
against real data before relying on them. A new namespace may be added without
touching this file's semantics, because ``source_system`` is free text on the
persisted column.

What a competition-discipline label names
-----------------------------------------

A label that names the competition *discipline* -- "Competition Squat",
"Competition Bench", "Competition Deadlift" -- resolves to the bare competition lift
and nothing narrower, because the rules define each lift by grip, start, and the
completed erect position rather than by every observable a label could mention. The
deadlift is the case that makes this obvious: the current IPF Technical Rulebook
defines the deadlift by bar position, grip, the start from the floor, and the
completed erect position, and it prescribes **neither** a conventional nor a sumo
stance. Both are legal performances of the same lift.

So ``deadlift`` is the stance-unspecified competition lift, and ``conventional_deadlift``
and ``sumo_deadlift`` are two stance-qualified variations of it -- carrying the same
specificity class, symmetrically. Neither is privileged for being the stance a lifter
more often meets, and a bare "Competition Deadlift" does not pick one. A source that
does name a stance ("Conventional Deadlift", "Sumo Deadlift") gets the stance-qualified
entity, exactly as "Low Bar Squat" gets ``low_bar_squat``.

What is deliberately absent
---------------------------

No entry asserts that one exercise is worth more than another, and no entry merges
two exercises because their names look alike. Where a distinction could matter to a
model -- sumo versus conventional, low bar versus high bar, paused versus ordinary,
competition bench versus close grip -- PSD keeps separate canonical exercises that
share a parent lift.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from psd.schema.vocabulary import (
    AliasSourceSystem,
    AmbiguityReason,
    BarType,
    ConfigurationFlag,
    ExerciseEquipment,
    Grip,
    ImplementType,
    Laterality,
    ParentLift,
    PauseRule,
    RangeOfMotion,
    ResolutionStatus,
    SpecificityLevel,
    Stance,
    TempoPattern,
)

__all__ = (
    "ALIASES",
    "CURATED_AMBIGUOUS_LABELS",
    "EXERCISES",
    "PROBE_LABELS",
    "PROVISIONAL_SOURCE_SYSTEMS",
    "SOURCE_SYSTEMS",
    "AliasSpec",
    "ExerciseSpec",
    "UnresolvedLabelSpec",
)


@dataclass(frozen=True, slots=True)
class ExerciseSpec:
    """One canonical exercise, declared by its observable descriptors.

    Attributes:
        key: Stable identity slug. ``exercise_id`` is derived from it, so
            renaming a key is a breaking change to every artifact that cites the
            exercise.
        canonical_name: Human-readable name.
        parent_lift: Which competition lift this exercise belongs to, or the
            accessory family.
        specificity_level: Whether this is the competition lift itself, a
            variation of it, sport-specific assistance, general strength work, or
            an accessory.
        implement: What carries the load.
        bar_type: Which bar carries it, where there is a bar.
        equipment: Which fixed apparatus the exercise requires.
        laterality: Unilateral or bilateral.
        stance: Foot placement, when the label distinguishes one.
        grip: Hand placement, when the label distinguishes one.
        range_of_motion: How far the movement traverses.
        pause_rule: Which pause the exercise has.
        tempo: Which tempo the exercise prescribes.
        configuration: Positional and setup descriptors that need a flag.
        note: Why the entry is shaped the way it is, for a reviewer.
    """

    key: str
    canonical_name: str
    parent_lift: ParentLift
    specificity_level: SpecificityLevel
    implement: ImplementType = ImplementType.UNKNOWN
    bar_type: BarType = BarType.UNKNOWN
    equipment: ExerciseEquipment = ExerciseEquipment.UNKNOWN
    laterality: Laterality = Laterality.BILATERAL
    stance: Stance = Stance.UNKNOWN
    grip: Grip = Grip.UNKNOWN
    range_of_motion: RangeOfMotion = RangeOfMotion.UNKNOWN
    pause_rule: PauseRule = PauseRule.UNKNOWN
    tempo: TempoPattern = TempoPattern.UNKNOWN
    configuration: tuple[ConfigurationFlag, ...] = ()
    note: str | None = None


@dataclass(frozen=True, slots=True)
class AliasSpec:
    """One source-specific spelling bound to a canonical exercise.

    Attributes:
        source_system: Namespace the spelling belongs to.
        raw_label: The spelling exactly as a source would write it.
        exercise_key: Which canonical exercise it names.
        note: Why this spelling maps here, for a reviewer.
    """

    source_system: str
    raw_label: str
    exercise_key: str
    note: str | None = None


@dataclass(frozen=True, slots=True)
class UnresolvedLabelSpec:
    """A label PSD has seen and is deliberately not resolving.

    Attributes:
        raw_label: The label as written.
        resolution_status: ``PARTIAL_FAMILY`` when the family is identifiable but
            no canonical exercise is, or ``AMBIGUOUS`` when several canonical
            exercises are defensible readings.
        reason: Why no canonical mapping is forced.
        parent_lift: The family, when the label identifies one.
        candidate_keys: The defensible readings, when there are several.
        note: What the source would have had to say, for a reviewer.
    """

    raw_label: str
    resolution_status: ResolutionStatus
    reason: AmbiguityReason
    parent_lift: ParentLift = ParentLift.UNKNOWN
    candidate_keys: tuple[str, ...] = ()
    note: str | None = None


_REGISTRY: Final[str] = AliasSourceSystem.PSD_REGISTRY.value
_CSV: Final[str] = AliasSourceSystem.GENERIC_CSV.value
_SHEET: Final[str] = AliasSourceSystem.SPREADSHEET.value
_HEVY: Final[str] = AliasSourceSystem.HEVY.value
_STRONG: Final[str] = AliasSourceSystem.STRONG.value
_HEROIC: Final[str] = AliasSourceSystem.TRAINHEROIC.value

#: Namespaces whose spellings are representative rather than verified exports.
PROVISIONAL_SOURCE_SYSTEMS: Final[frozenset[str]] = frozenset({_HEVY, _STRONG, _HEROIC})

#: Every namespace this catalog contributes.
SOURCE_SYSTEMS: Final[frozenset[str]] = frozenset(
    {_REGISTRY, _CSV, _SHEET, *PROVISIONAL_SOURCE_SYSTEMS}
)

_BARBELL: Final[ImplementType] = ImplementType.BARBELL
_OLYMPIC: Final[BarType] = BarType.OLYMPIC_BAR
_SPECIALTY: Final[BarType] = BarType.SPECIALTY_BAR
_NO_BAR: Final[BarType] = BarType.NOT_APPLICABLE
_NO_APPARATUS: Final[ExerciseEquipment] = ExerciseEquipment.NONE
_RULES_PAUSE: Final[PauseRule] = PauseRule.COMPETITION
_ADDED_PAUSE: Final[PauseRule] = PauseRule.BRIEF
_NO_PAUSE: Final[PauseRule] = PauseRule.NONE
_ROM_RULES: Final[RangeOfMotion] = RangeOfMotion.COMPETITION
_ROM_FULL: Final[RangeOfMotion] = RangeOfMotion.FULL
_ROM_PARTIAL: Final[RangeOfMotion] = RangeOfMotion.PARTIAL
_ROM_REDUCED: Final[RangeOfMotion] = RangeOfMotion.REDUCED
_ROM_ELEVATED: Final[RangeOfMotion] = RangeOfMotion.ELEVATED_START
_NO_STANCE: Final[Stance] = Stance.NOT_SPECIFIED
_NO_GRIP: Final[Grip] = Grip.NOT_SPECIFIED
_NO_ROM: Final[RangeOfMotion] = RangeOfMotion.NOT_SPECIFIED


EXERCISES: Final[tuple[ExerciseSpec, ...]] = (
    # ---------------------------------------------------------------- squat
    ExerciseSpec(
        key="squat",
        canonical_name="Squat",
        parent_lift=ParentLift.SQUAT,
        specificity_level=SpecificityLevel.COMPETITION_LIFT,
        implement=_BARBELL,
        bar_type=_OLYMPIC,
        equipment=_NO_APPARATUS,
        stance=_NO_STANCE,
        grip=_NO_GRIP,
        range_of_motion=_ROM_RULES,
        pause_rule=_RULES_PAUSE,
        note=(
            "The regulation squat. Stance stays unspecified because competition rules permit "
            "both low and high bar, so an unqualified 'Squat' does not pick one."
        ),
    ),
    ExerciseSpec(
        key="low_bar_squat",
        canonical_name="Low Bar Squat",
        parent_lift=ParentLift.SQUAT,
        specificity_level=SpecificityLevel.SPORT_SPECIFIC,
        implement=_BARBELL,
        bar_type=_SPECIALTY,
        equipment=_NO_APPARATUS,
        stance=Stance.LOW,
        range_of_motion=_ROM_RULES,
        pause_rule=_RULES_PAUSE,
    ),
    ExerciseSpec(
        key="high_bar_squat",
        canonical_name="High Bar Squat",
        parent_lift=ParentLift.SQUAT,
        specificity_level=SpecificityLevel.SPORT_SPECIFIC,
        implement=_BARBELL,
        bar_type=_SPECIALTY,
        equipment=_NO_APPARATUS,
        stance=Stance.HIGH,
        range_of_motion=_ROM_RULES,
        pause_rule=_RULES_PAUSE,
    ),
    ExerciseSpec(
        key="front_squat",
        canonical_name="Front Squat",
        parent_lift=ParentLift.SQUAT,
        specificity_level=SpecificityLevel.SPORT_SPECIFIC,
        implement=_BARBELL,
        bar_type=_OLYMPIC,
        equipment=_NO_APPARATUS,
        stance=_NO_STANCE,
        grip=Grip.FRONT_RACK,
        range_of_motion=_ROM_RULES,
        pause_rule=_RULES_PAUSE,
    ),
    ExerciseSpec(
        key="pause_squat",
        canonical_name="Paused Squat",
        parent_lift=ParentLift.SQUAT,
        specificity_level=SpecificityLevel.COMPETITION_VARIATION,
        implement=_BARBELL,
        bar_type=_OLYMPIC,
        equipment=_NO_APPARATUS,
        range_of_motion=_ROM_RULES,
        pause_rule=_ADDED_PAUSE,
        note="A pause added on top of the pause the squat rules already mandate.",
    ),
    ExerciseSpec(
        key="tempo_squat",
        canonical_name="Tempo Squat",
        parent_lift=ParentLift.SQUAT,
        specificity_level=SpecificityLevel.COMPETITION_VARIATION,
        implement=_BARBELL,
        bar_type=_OLYMPIC,
        equipment=_NO_APPARATUS,
        range_of_motion=_ROM_RULES,
        pause_rule=_RULES_PAUSE,
        tempo=TempoPattern.TEMPO_PRESCRIBED,
    ),
    ExerciseSpec(
        key="pin_squat",
        canonical_name="Pin Squat",
        parent_lift=ParentLift.SQUAT,
        specificity_level=SpecificityLevel.SPORT_SPECIFIC,
        implement=_BARBELL,
        bar_type=BarType.SAFETY_BAR,
        equipment=ExerciseEquipment.RACK,
        range_of_motion=_ROM_PARTIAL,
        pause_rule=_RULES_PAUSE,
        configuration=(ConfigurationFlag.SAFETY_BARS,),
    ),
    ExerciseSpec(
        key="box_squat",
        canonical_name="Box Squat",
        parent_lift=ParentLift.SQUAT,
        specificity_level=SpecificityLevel.SPORT_SPECIFIC,
        implement=_BARBELL,
        bar_type=_SPECIALTY,
        equipment=ExerciseEquipment.BOX,
        range_of_motion=_ROM_PARTIAL,
        pause_rule=_RULES_PAUSE,
        configuration=(ConfigurationFlag.BLOCK_SUPPORTED,),
        note="The box defines the depth, so the range of motion is set by the apparatus.",
    ),
    ExerciseSpec(
        key="safety_bar_squat",
        canonical_name="Safety Bar Squat",
        parent_lift=ParentLift.SQUAT,
        specificity_level=SpecificityLevel.SPORT_SPECIFIC,
        implement=_BARBELL,
        bar_type=BarType.SAFETY_BAR,
        equipment=ExerciseEquipment.RACK,
        range_of_motion=_ROM_RULES,
        pause_rule=_RULES_PAUSE,
        configuration=(ConfigurationFlag.SAFETY_BARS,),
    ),
    # ---------------------------------------------------------------- bench
    ExerciseSpec(
        key="bench",
        canonical_name="Bench Press",
        parent_lift=ParentLift.BENCH,
        specificity_level=SpecificityLevel.COMPETITION_LIFT,
        implement=_BARBELL,
        bar_type=_OLYMPIC,
        equipment=ExerciseEquipment.FLAT_BENCH,
        grip=_NO_GRIP,
        range_of_motion=_ROM_RULES,
        pause_rule=_RULES_PAUSE,
        note=(
            "The regulation bench press. The rules set hand spacing, so an unqualified "
            "'Bench Press' leaves grip unspecified rather than assuming a narrower one."
        ),
    ),
    ExerciseSpec(
        key="pause_bench",
        canonical_name="Paused Bench Press",
        parent_lift=ParentLift.BENCH,
        specificity_level=SpecificityLevel.COMPETITION_VARIATION,
        implement=_BARBELL,
        bar_type=_OLYMPIC,
        equipment=ExerciseEquipment.FLAT_BENCH,
        range_of_motion=_ROM_RULES,
        pause_rule=_ADDED_PAUSE,
        note="A pause added beyond the touch-and-pause the bench rules already require.",
    ),
    ExerciseSpec(
        key="two_count_bench",
        canonical_name="2-Count Bench Press",
        parent_lift=ParentLift.BENCH,
        specificity_level=SpecificityLevel.COMPETITION_VARIATION,
        implement=_BARBELL,
        bar_type=_OLYMPIC,
        equipment=ExerciseEquipment.FLAT_BENCH,
        range_of_motion=_ROM_RULES,
        pause_rule=PauseRule.COUNT_2,
        note="A counted pause, kept distinct from an open-ended long pause.",
    ),
    ExerciseSpec(
        key="three_count_bench",
        canonical_name="3-Count Bench Press",
        parent_lift=ParentLift.BENCH,
        specificity_level=SpecificityLevel.COMPETITION_VARIATION,
        implement=_BARBELL,
        bar_type=_OLYMPIC,
        equipment=ExerciseEquipment.FLAT_BENCH,
        range_of_motion=_ROM_RULES,
        pause_rule=PauseRule.COUNT_3,
    ),
    ExerciseSpec(
        key="long_pause_bench",
        canonical_name="Long Pause Bench Press",
        parent_lift=ParentLift.BENCH,
        specificity_level=SpecificityLevel.COMPETITION_VARIATION,
        implement=_BARBELL,
        bar_type=_OLYMPIC,
        equipment=ExerciseEquipment.FLAT_BENCH,
        range_of_motion=_ROM_RULES,
        pause_rule=PauseRule.LONG,
    ),
    ExerciseSpec(
        key="close_grip_bench",
        canonical_name="Close-Grip Bench Press",
        parent_lift=ParentLift.BENCH,
        specificity_level=SpecificityLevel.COMPETITION_VARIATION,
        implement=_BARBELL,
        bar_type=_OLYMPIC,
        equipment=ExerciseEquipment.FLAT_BENCH,
        grip=Grip.CLOSE,
        range_of_motion=_ROM_RULES,
        pause_rule=_RULES_PAUSE,
        configuration=(ConfigurationFlag.HAND_SPACING_REDUCED,),
        note="Hand spacing narrower than the rules require. A different exercise, not a synonym.",
    ),
    ExerciseSpec(
        key="wide_grip_bench",
        canonical_name="Wide-Grip Bench Press",
        parent_lift=ParentLift.BENCH,
        specificity_level=SpecificityLevel.COMPETITION_VARIATION,
        implement=_BARBELL,
        bar_type=_OLYMPIC,
        equipment=ExerciseEquipment.FLAT_BENCH,
        grip=Grip.WIDE,
        range_of_motion=_ROM_RULES,
        pause_rule=_RULES_PAUSE,
        configuration=(ConfigurationFlag.HAND_SPACING_WIDE,),
    ),
    ExerciseSpec(
        key="spoto_press",
        canonical_name="Spoto Press",
        parent_lift=ParentLift.BENCH,
        specificity_level=SpecificityLevel.SPORT_SPECIFIC,
        implement=_BARBELL,
        bar_type=_OLYMPIC,
        equipment=ExerciseEquipment.FLAT_BENCH,
        range_of_motion=_ROM_REDUCED,
        pause_rule=_RULES_PAUSE,
        configuration=(
            ConfigurationFlag.COMPETITION_TOUCH_POINT,
            ConfigurationFlag.SEATED,
            ConfigurationFlag.SHORT_RANGE_OF_MOTION,
        ),
    ),
    ExerciseSpec(
        key="feet_up_bench",
        canonical_name="Feet-Up Bench Press",
        parent_lift=ParentLift.BENCH,
        specificity_level=SpecificityLevel.SPORT_SPECIFIC,
        implement=_BARBELL,
        bar_type=_OLYMPIC,
        equipment=ExerciseEquipment.FLAT_BENCH,
        range_of_motion=_ROM_RULES,
        pause_rule=_RULES_PAUSE,
        configuration=(ConfigurationFlag.FEET_ELEVATED, ConfigurationFlag.SEATED),
    ),
    ExerciseSpec(
        key="board_press",
        canonical_name="Board Press",
        parent_lift=ParentLift.BENCH,
        specificity_level=SpecificityLevel.SPORT_SPECIFIC,
        implement=_BARBELL,
        bar_type=_OLYMPIC,
        equipment=_NO_APPARATUS,
        range_of_motion=_ROM_PARTIAL,
        pause_rule=_NO_PAUSE,
        configuration=(
            ConfigurationFlag.BLOCK_SUPPORTED,
            ConfigurationFlag.SHORT_RANGE_OF_MOTION,
        ),
    ),
    ExerciseSpec(
        key="tempo_bench",
        canonical_name="Tempo Bench Press",
        parent_lift=ParentLift.BENCH,
        specificity_level=SpecificityLevel.COMPETITION_VARIATION,
        implement=_BARBELL,
        bar_type=_OLYMPIC,
        equipment=ExerciseEquipment.FLAT_BENCH,
        range_of_motion=_ROM_RULES,
        pause_rule=_RULES_PAUSE,
        tempo=TempoPattern.TEMPO_PRESCRIBED,
    ),
    ExerciseSpec(
        key="incline_bench",
        canonical_name="Incline Bench Press",
        parent_lift=ParentLift.BENCH,
        specificity_level=SpecificityLevel.GENERAL_STRENGTH,
        implement=_BARBELL,
        bar_type=_OLYMPIC,
        equipment=ExerciseEquipment.INCLINE_BENCH,
        range_of_motion=_ROM_PARTIAL,
        pause_rule=_NO_PAUSE,
    ),
    ExerciseSpec(
        key="decline_bench",
        canonical_name="Decline Bench Press",
        parent_lift=ParentLift.BENCH,
        specificity_level=SpecificityLevel.GENERAL_STRENGTH,
        implement=_BARBELL,
        bar_type=_OLYMPIC,
        equipment=ExerciseEquipment.DECLINE_BENCH,
        range_of_motion=_ROM_PARTIAL,
        pause_rule=_NO_PAUSE,
    ),
    ExerciseSpec(
        key="floor_press",
        canonical_name="Floor Press",
        parent_lift=ParentLift.BENCH,
        specificity_level=SpecificityLevel.GENERAL_STRENGTH,
        implement=_BARBELL,
        bar_type=_OLYMPIC,
        equipment=_NO_APPARATUS,
        range_of_motion=_ROM_REDUCED,
        pause_rule=_NO_PAUSE,
        configuration=(ConfigurationFlag.SHORT_RANGE_OF_MOTION,),
    ),
    ExerciseSpec(
        key="dumbbell_bench",
        canonical_name="Dumbbell Bench Press",
        parent_lift=ParentLift.BENCH,
        specificity_level=SpecificityLevel.GENERAL_STRENGTH,
        implement=ImplementType.DUMBBELL,
        bar_type=_NO_BAR,
        equipment=ExerciseEquipment.FLAT_BENCH,
        range_of_motion=_ROM_PARTIAL,
        pause_rule=_NO_PAUSE,
    ),
    ExerciseSpec(
        key="smith_bench",
        canonical_name="Smith Machine Bench Press",
        parent_lift=ParentLift.BENCH,
        specificity_level=SpecificityLevel.GENERAL_STRENGTH,
        implement=ImplementType.SMITH_MACHINE,
        bar_type=_NO_BAR,
        equipment=ExerciseEquipment.MACHINE,
        range_of_motion=_ROM_RULES,
        pause_rule=_NO_PAUSE,
    ),
    # ------------------------------------------------------------ deadlift
    ExerciseSpec(
        key="deadlift",
        canonical_name="Deadlift",
        parent_lift=ParentLift.DEADLIFT,
        specificity_level=SpecificityLevel.COMPETITION_LIFT,
        implement=_BARBELL,
        bar_type=_OLYMPIC,
        equipment=_NO_APPARATUS,
        stance=_NO_STANCE,
        range_of_motion=_ROM_RULES,
        pause_rule=_RULES_PAUSE,
        note=(
            "The regulation deadlift, stance unspecified. The rules define it by bar "
            "position, grip, the start from the floor, and the completed erect position; "
            "they do not prescribe a conventional or sumo stance, so an unqualified "
            "'Deadlift' does not pick one."
        ),
    ),
    ExerciseSpec(
        key="conventional_deadlift",
        canonical_name="Conventional Deadlift",
        parent_lift=ParentLift.DEADLIFT,
        specificity_level=SpecificityLevel.COMPETITION_VARIATION,
        implement=_BARBELL,
        bar_type=_OLYMPIC,
        equipment=_NO_APPARATUS,
        stance=Stance.MODERATE,
        range_of_motion=_ROM_RULES,
        pause_rule=_RULES_PAUSE,
        note=(
            "A stance-qualified way of performing the competition deadlift. Stance is an "
            "observable descriptor rather than something the rules fix, so this is a "
            "variation of the competition lift and is classified exactly as sumo is."
        ),
    ),
    ExerciseSpec(
        key="sumo_deadlift",
        canonical_name="Sumo Deadlift",
        parent_lift=ParentLift.DEADLIFT,
        specificity_level=SpecificityLevel.COMPETITION_VARIATION,
        implement=_BARBELL,
        bar_type=_OLYMPIC,
        equipment=_NO_APPARATUS,
        stance=Stance.WIDE,
        range_of_motion=_ROM_RULES,
        pause_rule=_RULES_PAUSE,
        note=(
            "The mirror image of conventional: a stance-qualified way of performing the "
            "competition deadlift, classified identically, because the rules prescribe "
            "neither stance."
        ),
    ),
    ExerciseSpec(
        key="pause_deadlift",
        canonical_name="Paused Deadlift",
        parent_lift=ParentLift.DEADLIFT,
        specificity_level=SpecificityLevel.COMPETITION_VARIATION,
        implement=_BARBELL,
        bar_type=_OLYMPIC,
        equipment=_NO_APPARATUS,
        stance=_NO_STANCE,
        range_of_motion=_ROM_RULES,
        pause_rule=_ADDED_PAUSE,
    ),
    ExerciseSpec(
        key="tempo_deadlift",
        canonical_name="Tempo Deadlift",
        parent_lift=ParentLift.DEADLIFT,
        specificity_level=SpecificityLevel.COMPETITION_VARIATION,
        implement=_BARBELL,
        bar_type=_OLYMPIC,
        equipment=_NO_APPARATUS,
        stance=_NO_STANCE,
        range_of_motion=_ROM_RULES,
        pause_rule=_RULES_PAUSE,
        tempo=TempoPattern.TEMPO_PRESCRIBED,
    ),
    ExerciseSpec(
        key="deficit_deadlift",
        canonical_name="Deficit Deadlift",
        parent_lift=ParentLift.DEADLIFT,
        specificity_level=SpecificityLevel.SPORT_SPECIFIC,
        implement=_BARBELL,
        bar_type=_OLYMPIC,
        equipment=ExerciseEquipment.PLATFORMS,
        stance=_NO_STANCE,
        range_of_motion=_ROM_ELEVATED,
        pause_rule=_RULES_PAUSE,
    ),
    ExerciseSpec(
        key="block_pull",
        canonical_name="Block Pull",
        parent_lift=ParentLift.DEADLIFT,
        specificity_level=SpecificityLevel.SPORT_SPECIFIC,
        implement=_BARBELL,
        bar_type=_OLYMPIC,
        equipment=ExerciseEquipment.BOX,
        stance=_NO_STANCE,
        range_of_motion=_ROM_PARTIAL,
        pause_rule=_RULES_PAUSE,
        configuration=(
            ConfigurationFlag.BLOCK_SUPPORTED,
            ConfigurationFlag.SHORT_RANGE_OF_MOTION,
        ),
    ),
    ExerciseSpec(
        key="rack_pull",
        canonical_name="Rack Pull",
        parent_lift=ParentLift.DEADLIFT,
        specificity_level=SpecificityLevel.SPORT_SPECIFIC,
        implement=_BARBELL,
        bar_type=_OLYMPIC,
        equipment=ExerciseEquipment.RACK,
        stance=_NO_STANCE,
        range_of_motion=_ROM_PARTIAL,
        pause_rule=_RULES_PAUSE,
        configuration=(
            ConfigurationFlag.BLOCK_SUPPORTED,
            ConfigurationFlag.SHORT_RANGE_OF_MOTION,
        ),
    ),
    ExerciseSpec(
        key="rdl",
        canonical_name="Romanian Deadlift",
        parent_lift=ParentLift.DEADLIFT,
        specificity_level=SpecificityLevel.SPORT_SPECIFIC,
        implement=_BARBELL,
        bar_type=_OLYMPIC,
        equipment=_NO_APPARATUS,
        stance=_NO_STANCE,
        range_of_motion=_ROM_PARTIAL,
        pause_rule=_NO_PAUSE,
        note="A hinge with a shortened range of motion, not a partial deadlift repetition.",
    ),
    ExerciseSpec(
        key="stiff_leg_deadlift",
        canonical_name="Stiff-Leg Deadlift",
        parent_lift=ParentLift.DEADLIFT,
        specificity_level=SpecificityLevel.SPORT_SPECIFIC,
        implement=_BARBELL,
        bar_type=_OLYMPIC,
        equipment=_NO_APPARATUS,
        stance=_NO_STANCE,
        range_of_motion=_ROM_PARTIAL,
        pause_rule=_NO_PAUSE,
    ),
    ExerciseSpec(
        key="trap_bar_deadlift",
        canonical_name="Trap-Bar Deadlift",
        parent_lift=ParentLift.DEADLIFT,
        specificity_level=SpecificityLevel.SPORT_SPECIFIC,
        implement=_BARBELL,
        bar_type=BarType.TRAP_BAR,
        equipment=_NO_APPARATUS,
        stance=Stance.MODERATE,
        range_of_motion=_ROM_RULES,
        pause_rule=_RULES_PAUSE,
    ),
    # -------------------------------------------------- squat-side accessory
    ExerciseSpec(
        key="good_morning",
        canonical_name="Good Morning",
        parent_lift=ParentLift.DEADLIFT,
        specificity_level=SpecificityLevel.GENERAL_STRENGTH,
        implement=_BARBELL,
        bar_type=_OLYMPIC,
        equipment=_NO_APPARATUS,
        stance=_NO_STANCE,
        range_of_motion=_ROM_PARTIAL,
        pause_rule=_NO_PAUSE,
    ),
    ExerciseSpec(
        key="hack_squat",
        canonical_name="Hack Squat",
        parent_lift=ParentLift.SQUAT,
        specificity_level=SpecificityLevel.GENERAL_STRENGTH,
        implement=ImplementType.MACHINE,
        bar_type=_NO_BAR,
        equipment=ExerciseEquipment.MACHINE,
        range_of_motion=_ROM_PARTIAL,
        pause_rule=_NO_PAUSE,
    ),
    ExerciseSpec(
        key="leg_press",
        canonical_name="Leg Press",
        parent_lift=ParentLift.ACCESSORY,
        specificity_level=SpecificityLevel.ACCESSORY,
        implement=ImplementType.MACHINE,
        bar_type=_NO_BAR,
        equipment=ExerciseEquipment.MACHINE,
        range_of_motion=_ROM_PARTIAL,
        pause_rule=_NO_PAUSE,
    ),
    ExerciseSpec(
        key="leg_extension",
        canonical_name="Leg Extension",
        parent_lift=ParentLift.ACCESSORY,
        specificity_level=SpecificityLevel.ACCESSORY,
        implement=ImplementType.MACHINE,
        bar_type=_NO_BAR,
        equipment=ExerciseEquipment.MACHINE,
        range_of_motion=_ROM_PARTIAL,
        pause_rule=_NO_PAUSE,
    ),
    ExerciseSpec(
        key="leg_curl",
        canonical_name="Leg Curl",
        parent_lift=ParentLift.ACCESSORY,
        specificity_level=SpecificityLevel.ACCESSORY,
        implement=ImplementType.MACHINE,
        bar_type=_NO_BAR,
        equipment=ExerciseEquipment.MACHINE,
        range_of_motion=_ROM_PARTIAL,
        pause_rule=_NO_PAUSE,
    ),
    ExerciseSpec(
        key="goblet_squat",
        canonical_name="Goblet Squat",
        parent_lift=ParentLift.SQUAT,
        specificity_level=SpecificityLevel.ACCESSORY,
        implement=ImplementType.KETTLEBELL,
        bar_type=_NO_BAR,
        equipment=_NO_APPARATUS,
        stance=_NO_STANCE,
        range_of_motion=_ROM_PARTIAL,
        pause_rule=_NO_PAUSE,
    ),
    ExerciseSpec(
        key="walking_lunge",
        canonical_name="Walking Lunge",
        parent_lift=ParentLift.SQUAT,
        specificity_level=SpecificityLevel.ACCESSORY,
        implement=ImplementType.DUMBBELL,
        bar_type=_NO_BAR,
        equipment=_NO_APPARATUS,
        stance=Stance.SPLIT,
        range_of_motion=_ROM_PARTIAL,
        pause_rule=_NO_PAUSE,
        laterality=Laterality.UNILATERAL,
    ),
    ExerciseSpec(
        key="bulgarian_split_squat",
        canonical_name="Bulgarian Split Squat",
        parent_lift=ParentLift.SQUAT,
        specificity_level=SpecificityLevel.ACCESSORY,
        implement=ImplementType.DUMBBELL,
        bar_type=_NO_BAR,
        equipment=_NO_APPARATUS,
        stance=Stance.SPLIT,
        range_of_motion=_ROM_PARTIAL,
        pause_rule=_NO_PAUSE,
        laterality=Laterality.UNILATERAL,
        configuration=(ConfigurationFlag.FEET_ELEVATED,),
    ),
    # ------------------------------------------------------- pull accessory
    ExerciseSpec(
        key="lat_pulldown",
        canonical_name="Lat Pulldown",
        parent_lift=ParentLift.ACCESSORY,
        specificity_level=SpecificityLevel.ACCESSORY,
        implement=ImplementType.CABLE,
        bar_type=_NO_BAR,
        equipment=ExerciseEquipment.CABLE_STATION,
        range_of_motion=_ROM_FULL,
        pause_rule=_NO_PAUSE,
    ),
    ExerciseSpec(
        key="pull_up",
        canonical_name="Pull-Up",
        parent_lift=ParentLift.ACCESSORY,
        specificity_level=SpecificityLevel.ACCESSORY,
        implement=ImplementType.BODYWEIGHT,
        bar_type=_NO_BAR,
        equipment=_NO_APPARATUS,
        range_of_motion=_ROM_FULL,
        pause_rule=_NO_PAUSE,
    ),
    ExerciseSpec(
        key="chin_up",
        canonical_name="Chin-Up",
        parent_lift=ParentLift.ACCESSORY,
        specificity_level=SpecificityLevel.ACCESSORY,
        implement=ImplementType.BODYWEIGHT,
        bar_type=_NO_BAR,
        equipment=_NO_APPARATUS,
        grip=Grip.UNDERHAND,
        range_of_motion=_ROM_FULL,
        pause_rule=_NO_PAUSE,
    ),
    ExerciseSpec(
        key="barbell_row",
        canonical_name="Barbell Row",
        parent_lift=ParentLift.ACCESSORY,
        specificity_level=SpecificityLevel.GENERAL_STRENGTH,
        implement=_BARBELL,
        bar_type=_OLYMPIC,
        equipment=_NO_APPARATUS,
        range_of_motion=_ROM_FULL,
        pause_rule=_NO_PAUSE,
    ),
    ExerciseSpec(
        key="pendlay_row",
        canonical_name="Pendlay Row",
        parent_lift=ParentLift.ACCESSORY,
        specificity_level=SpecificityLevel.SPORT_SPECIFIC,
        implement=_BARBELL,
        bar_type=_OLYMPIC,
        equipment=_NO_APPARATUS,
        range_of_motion=_ROM_FULL,
        pause_rule=_NO_PAUSE,
        configuration=(ConfigurationFlag.TOUCH_AND_GO,),
    ),
    ExerciseSpec(
        key="t_bar_row",
        canonical_name="T-Bar Row",
        parent_lift=ParentLift.ACCESSORY,
        specificity_level=SpecificityLevel.ACCESSORY,
        implement=_BARBELL,
        bar_type=BarType.T_BAR,
        equipment=_NO_APPARATUS,
        range_of_motion=_ROM_FULL,
        pause_rule=_NO_PAUSE,
    ),
    ExerciseSpec(
        key="seated_cable_row",
        canonical_name="Seated Cable Row",
        parent_lift=ParentLift.ACCESSORY,
        specificity_level=SpecificityLevel.ACCESSORY,
        implement=ImplementType.CABLE,
        bar_type=_NO_BAR,
        equipment=ExerciseEquipment.CABLE_STATION,
        range_of_motion=_ROM_PARTIAL,
        pause_rule=_NO_PAUSE,
        configuration=(ConfigurationFlag.SEATED,),
    ),
    # ----------------------------------------------------- press accessory
    ExerciseSpec(
        key="overhead_press",
        canonical_name="Overhead Press",
        parent_lift=ParentLift.ACCESSORY,
        specificity_level=SpecificityLevel.GENERAL_STRENGTH,
        implement=_BARBELL,
        bar_type=_OLYMPIC,
        equipment=_NO_APPARATUS,
        range_of_motion=_ROM_FULL,
        pause_rule=_NO_PAUSE,
    ),
    ExerciseSpec(
        key="push_up",
        canonical_name="Push-Up",
        parent_lift=ParentLift.ACCESSORY,
        specificity_level=SpecificityLevel.ACCESSORY,
        implement=ImplementType.BODYWEIGHT,
        bar_type=_NO_BAR,
        equipment=_NO_APPARATUS,
        range_of_motion=_ROM_FULL,
        pause_rule=_NO_PAUSE,
    ),
    ExerciseSpec(
        key="dips",
        canonical_name="Dips",
        parent_lift=ParentLift.ACCESSORY,
        specificity_level=SpecificityLevel.ACCESSORY,
        implement=ImplementType.BODYWEIGHT,
        bar_type=_NO_BAR,
        equipment=ExerciseEquipment.DIP_BARS,
        range_of_motion=_ROM_PARTIAL,
        pause_rule=_NO_PAUSE,
    ),
    ExerciseSpec(
        key="cable_fly",
        canonical_name="Cable Fly",
        parent_lift=ParentLift.ACCESSORY,
        specificity_level=SpecificityLevel.ACCESSORY,
        implement=ImplementType.CABLE,
        bar_type=_NO_BAR,
        equipment=ExerciseEquipment.CABLE_STATION,
        range_of_motion=_ROM_PARTIAL,
        pause_rule=_NO_PAUSE,
    ),
    # ------------------------------------------------------ upper accessory
    ExerciseSpec(
        key="lateral_raise",
        canonical_name="Lateral Raise",
        parent_lift=ParentLift.ACCESSORY,
        specificity_level=SpecificityLevel.ACCESSORY,
        implement=ImplementType.DUMBBELL,
        bar_type=_NO_BAR,
        equipment=_NO_APPARATUS,
        range_of_motion=_ROM_PARTIAL,
        pause_rule=_NO_PAUSE,
    ),
    ExerciseSpec(
        key="rear_delt_fly",
        canonical_name="Rear Delt Fly",
        parent_lift=ParentLift.ACCESSORY,
        specificity_level=SpecificityLevel.ACCESSORY,
        implement=ImplementType.DUMBBELL,
        bar_type=_NO_BAR,
        equipment=_NO_APPARATUS,
        range_of_motion=_ROM_PARTIAL,
        pause_rule=_NO_PAUSE,
    ),
    ExerciseSpec(
        key="face_pull",
        canonical_name="Face Pull",
        parent_lift=ParentLift.ACCESSORY,
        specificity_level=SpecificityLevel.ACCESSORY,
        implement=ImplementType.CABLE,
        bar_type=_NO_BAR,
        equipment=ExerciseEquipment.CABLE_STATION,
        range_of_motion=_ROM_PARTIAL,
        pause_rule=_NO_PAUSE,
    ),
    ExerciseSpec(
        key="barbell_curl",
        canonical_name="Barbell Curl",
        parent_lift=ParentLift.ACCESSORY,
        specificity_level=SpecificityLevel.ACCESSORY,
        implement=_BARBELL,
        bar_type=_OLYMPIC,
        equipment=_NO_APPARATUS,
        range_of_motion=_ROM_PARTIAL,
        pause_rule=_NO_PAUSE,
    ),
    ExerciseSpec(
        key="dumbbell_curl",
        canonical_name="Dumbbell Curl",
        parent_lift=ParentLift.ACCESSORY,
        specificity_level=SpecificityLevel.ACCESSORY,
        implement=ImplementType.DUMBBELL,
        bar_type=_NO_BAR,
        equipment=_NO_APPARATUS,
        range_of_motion=_ROM_PARTIAL,
        pause_rule=_NO_PAUSE,
    ),
    ExerciseSpec(
        key="hammer_curl",
        canonical_name="Hammer Curl",
        parent_lift=ParentLift.ACCESSORY,
        specificity_level=SpecificityLevel.ACCESSORY,
        implement=ImplementType.DUMBBELL,
        bar_type=_NO_BAR,
        equipment=_NO_APPARATUS,
        grip=Grip.NEUTRAL,
        range_of_motion=_ROM_PARTIAL,
        pause_rule=_NO_PAUSE,
    ),
    ExerciseSpec(
        key="triceps_pushdown",
        canonical_name="Triceps Pushdown",
        parent_lift=ParentLift.ACCESSORY,
        specificity_level=SpecificityLevel.ACCESSORY,
        implement=ImplementType.CABLE,
        bar_type=_NO_BAR,
        equipment=ExerciseEquipment.CABLE_STATION,
        range_of_motion=_ROM_PARTIAL,
        pause_rule=_NO_PAUSE,
    ),
    # ------------------------------------------------------ core accessory
    ExerciseSpec(
        key="plank",
        canonical_name="Plank",
        parent_lift=ParentLift.ACCESSORY,
        specificity_level=SpecificityLevel.ACCESSORY,
        implement=ImplementType.BODYWEIGHT,
        bar_type=_NO_BAR,
        equipment=_NO_APPARATUS,
        range_of_motion=_NO_ROM,
        pause_rule=_NO_PAUSE,
    ),
    ExerciseSpec(
        key="side_plank",
        canonical_name="Side Plank",
        parent_lift=ParentLift.ACCESSORY,
        specificity_level=SpecificityLevel.ACCESSORY,
        implement=ImplementType.BODYWEIGHT,
        bar_type=_NO_BAR,
        equipment=_NO_APPARATUS,
        range_of_motion=_NO_ROM,
        pause_rule=_NO_PAUSE,
        laterality=Laterality.UNILATERAL,
    ),
    ExerciseSpec(
        key="hanging_leg_raise",
        canonical_name="Hanging Leg Raise",
        parent_lift=ParentLift.ACCESSORY,
        specificity_level=SpecificityLevel.ACCESSORY,
        implement=ImplementType.BODYWEIGHT,
        bar_type=_NO_BAR,
        equipment=_NO_APPARATUS,
        range_of_motion=_ROM_FULL,
        pause_rule=_NO_PAUSE,
    ),
    ExerciseSpec(
        key="back_extension",
        canonical_name="Back Extension",
        parent_lift=ParentLift.ACCESSORY,
        specificity_level=SpecificityLevel.ACCESSORY,
        implement=ImplementType.BODYWEIGHT,
        bar_type=_NO_BAR,
        equipment=ExerciseEquipment.HYPEREXTENSION_BENCH,
        range_of_motion=_ROM_FULL,
        pause_rule=_NO_PAUSE,
    ),
)

# ---------------------------------------------------------------------------
# Source aliases
# ---------------------------------------------------------------------------
#
# ``psd_registry`` entries are generated from each canonical key and name by the
# registry, so they are not listed here: a canonical exercise is always reachable
# by its own identity. What follows are the *extra* spellings sources use.

ALIASES: Final[tuple[AliasSpec, ...]] = (
    # -- squat ---------------------------------------------------------------
    AliasSpec(_CSV, "Comp Squat", "squat"),
    AliasSpec(_CSV, "Squat (Competition)", "squat"),
    AliasSpec(_HEVY, "Competition Squat", "squat"),
    AliasSpec(_STRONG, "Squat", "squat"),
    AliasSpec(_CSV, "Low-Bar Squat", "low_bar_squat"),
    AliasSpec(_CSV, "Lowbar Squat", "low_bar_squat"),
    AliasSpec(_CSV, "High-Bar Squat", "high_bar_squat"),
    AliasSpec(_CSV, "Highbar Squat", "high_bar_squat"),
    AliasSpec(_CSV, "Front-Squat", "front_squat"),
    AliasSpec(_HEVY, "Paused Squat", "pause_squat"),
    AliasSpec(_CSV, "Pause Squat", "pause_squat"),
    AliasSpec(_CSV, "Squat Pause", "pause_squat"),
    AliasSpec(_CSV, "Tempo Squat Press", "tempo_squat", "A longer spelling found in the wild."),
    AliasSpec(_CSV, "Tempo Squat", "tempo_squat"),
    AliasSpec(_CSV, "Safety Bar Squat", "safety_bar_squat"),
    AliasSpec(_CSV, "SSB Squat", "safety_bar_squat"),
    AliasSpec(_CSV, "Pin Squat", "pin_squat"),
    AliasSpec(_CSV, "Box Squat", "box_squat"),
    # -- bench ---------------------------------------------------------------
    AliasSpec(_CSV, "Comp Bench", "bench"),
    AliasSpec(_CSV, "Bench Press (Competition)", "bench"),
    AliasSpec(_HEVY, "Competition Bench Press", "bench"),
    AliasSpec(_HEVY, "Competition Bench", "bench"),
    AliasSpec(_CSV, "Benchpress", "bench"),
    AliasSpec(_STRONG, "Flat Bench Press", "bench"),
    AliasSpec(_CSV, "Pause Bench", "pause_bench"),
    AliasSpec(_HEVY, "Paused Bench Press", "pause_bench"),
    AliasSpec(_HEVY, "Paused Bench", "pause_bench"),
    AliasSpec(_STRONG, "Paused Bench", "pause_bench"),
    AliasSpec(_CSV, "Bench Pause", "pause_bench"),
    AliasSpec(_CSV, "2ct Bench", "two_count_bench"),
    AliasSpec(_HEVY, "2ct Bench", "two_count_bench"),
    AliasSpec(_CSV, "2-count Bench", "two_count_bench"),
    AliasSpec(_CSV, "2 Count Bench Press", "two_count_bench"),
    AliasSpec(_CSV, "Two Count Bench", "two_count_bench"),
    AliasSpec(_CSV, "3ct Bench", "three_count_bench"),
    AliasSpec(_CSV, "3 Count Bench Press", "three_count_bench"),
    AliasSpec(_CSV, "Long Pause Bench", "long_pause_bench"),
    AliasSpec(_CSV, "2s Pause Bench", "long_pause_bench"),
    AliasSpec(_CSV, "Close Grip Bench", "close_grip_bench"),
    AliasSpec(_CSV, "Close-Grip Bench Press", "close_grip_bench"),
    AliasSpec(_CSV, "CGBP", "close_grip_bench"),
    AliasSpec(_HEVY, "CGBP", "close_grip_bench"),
    AliasSpec(_STRONG, "CGBP", "close_grip_bench"),
    AliasSpec(_HEVY, "Close Grip Bench Press", "close_grip_bench"),
    AliasSpec(_CSV, "Wide Grip Bench", "wide_grip_bench"),
    AliasSpec(_CSV, "Wide-Grip Bench Press", "wide_grip_bench"),
    AliasSpec(_CSV, "Spoto Press", "spoto_press"),
    AliasSpec(_CSV, "Spoto", "spoto_press"),
    AliasSpec(_CSV, "Feet Up Bench", "feet_up_bench"),
    AliasSpec(_CSV, "Feet-Up Bench Press", "feet_up_bench"),
    AliasSpec(_CSV, "Feet Elevated Bench", "feet_up_bench"),
    AliasSpec(_CSV, "Board Press", "board_press"),
    AliasSpec(_CSV, "Tempo Bench", "tempo_bench"),
    AliasSpec(_CSV, "Tempo Bench Press", "tempo_bench"),
    AliasSpec(_CSV, "Incline Bench", "incline_bench"),
    AliasSpec(_CSV, "Incline Bench Press", "incline_bench"),
    AliasSpec(_CSV, "Decline Bench", "decline_bench"),
    AliasSpec(_CSV, "Decline Bench Press", "decline_bench"),
    AliasSpec(_CSV, "Floor Press", "floor_press"),
    AliasSpec(_CSV, "Dumbbell Bench", "dumbbell_bench"),
    AliasSpec(_CSV, "DB Bench Press", "dumbbell_bench"),
    AliasSpec(_CSV, "Smith Machine Bench", "smith_bench"),
    AliasSpec(_CSV, "Smith Bench", "smith_bench"),
    # -- deadlift ------------------------------------------------------------
    AliasSpec(_CSV, "Comp Deadlift", "deadlift"),
    # A competition-discipline label names the lift, not a stance. Binding it to the
    # stance-unspecified ``deadlift`` is what keeps the deadlift family consistent with
    # "Competition Squat" -> ``squat`` and "Competition Bench" -> ``bench``.
    AliasSpec(_HEVY, "Competition Deadlift", "deadlift"),
    # This spelling *does* name a stance, so it resolves to the stance-qualified
    # entity rather than to the stance-unspecified competition deadlift.
    AliasSpec(_CSV, "Deadlift (Conventional)", "conventional_deadlift"),
    AliasSpec(_CSV, "Conventional Pull", "conventional_deadlift"),
    AliasSpec(_CSV, "Conv DL", "conventional_deadlift"),
    AliasSpec(_CSV, "Conventional DL", "conventional_deadlift"),
    AliasSpec(_CSV, "Standard Deadlift", "conventional_deadlift"),
    AliasSpec(_CSV, "Sumo DL", "sumo_deadlift"),
    AliasSpec(_CSV, "Sumo Pull", "sumo_deadlift"),
    AliasSpec(_HEVY, "Sumo Deadlift", "sumo_deadlift"),
    AliasSpec(_CSV, "Pause Deadlift", "pause_deadlift"),
    AliasSpec(_HEVY, "Paused Deadlift", "pause_deadlift"),
    AliasSpec(_CSV, "Tempo Deadlift", "tempo_deadlift"),
    AliasSpec(_CSV, "Deficit Deadlift", "deficit_deadlift"),
    AliasSpec(_CSV, "Deficit Pull", "deficit_deadlift"),
    AliasSpec(_CSV, "Block Pull", "block_pull"),
    AliasSpec(_CSV, "Rack Pull", "rack_pull"),
    AliasSpec(_CSV, "Rack Pulls", "rack_pull"),
    AliasSpec(_CSV, "RDL", "rdl"),
    AliasSpec(_CSV, "Romanian Deadlift", "rdl"),
    AliasSpec(_CSV, "Stiff Leg Deadlift", "stiff_leg_deadlift"),
    AliasSpec(_CSV, "Stiff-Leg DL", "stiff_leg_deadlift"),
    AliasSpec(_CSV, "Trap Bar Deadlift", "trap_bar_deadlift"),
    AliasSpec(_CSV, "Trap-Bar DL", "trap_bar_deadlift"),
    AliasSpec(_CSV, "Good Morning", "good_morning"),
    AliasSpec(_CSV, "Goodmorning", "good_morning"),
    # -- machine and accessory ----------------------------------------------
    AliasSpec(_CSV, "Hack Squat Machine", "hack_squat"),
    AliasSpec(_CSV, "Leg Press Machine", "leg_press"),
    AliasSpec(_SHEET, "Legpress", "leg_press"),
    AliasSpec(_SHEET, "Leg Curl Machine", "leg_curl"),
    AliasSpec(_SHEET, "Leg Extension Machine", "leg_extension"),
    AliasSpec(_CSV, "Goblet Squat", "goblet_squat"),
    AliasSpec(_CSV, "Goblet Squat Kettlebell", "goblet_squat"),
    AliasSpec(_CSV, "Walking Lunge", "walking_lunge"),
    AliasSpec(_CSV, "Bulgarian Split Squat", "bulgarian_split_squat"),
    AliasSpec(_CSV, "BSS", "bulgarian_split_squat"),
    AliasSpec(_CSV, "Lat Pulldown", "lat_pulldown"),
    AliasSpec(_CSV, "Lat Pull Down", "lat_pulldown"),
    AliasSpec(_CSV, "Pull Up", "pull_up"),
    AliasSpec(_CSV, "Pullup", "pull_up"),
    AliasSpec(_CSV, "Chin Up", "chin_up"),
    AliasSpec(_CSV, "Chinup", "chin_up"),
    AliasSpec(_CSV, "Barbell Row", "barbell_row"),
    AliasSpec(_CSV, "Bent Over Row", "barbell_row"),
    AliasSpec(_CSV, "Pendlay Row", "pendlay_row"),
    AliasSpec(_CSV, "T Bar Row", "t_bar_row"),
    AliasSpec(_CSV, "T-Bar Row", "t_bar_row"),
    AliasSpec(_CSV, "Seated Cable Row", "seated_cable_row"),
    AliasSpec(_CSV, "Cable Row", "seated_cable_row"),
    AliasSpec(_CSV, "Overhead Press", "overhead_press"),
    AliasSpec(_CSV, "OHP", "overhead_press"),
    AliasSpec(_CSV, "Military Press", "overhead_press"),
    AliasSpec(_CSV, "Push Up", "push_up"),
    AliasSpec(_CSV, "Pushup", "push_up"),
    AliasSpec(_CSV, "Chest Dip", "dips"),
    AliasSpec(_CSV, "Weighted Dip", "dips"),
    AliasSpec(_CSV, "Cable Fly", "cable_fly"),
    AliasSpec(_CSV, "Cable Crossover", "cable_fly"),
    AliasSpec(_CSV, "Lat Raise", "lateral_raise"),
    AliasSpec(_CSV, "Dumbbell Lateral Raise", "lateral_raise"),
    AliasSpec(_CSV, "Lateral Raises", "lateral_raise"),
    AliasSpec(_CSV, "Side Lateral Raise", "lateral_raise"),
    AliasSpec(_CSV, "Rear Delt Fly", "rear_delt_fly"),
    AliasSpec(_CSV, "Rear Delt Raise", "rear_delt_fly"),
    AliasSpec(_CSV, "Face Pull", "face_pull"),
    AliasSpec(_CSV, "Barbell Curl", "barbell_curl"),
    AliasSpec(_CSV, "BB Curl", "barbell_curl"),
    AliasSpec(_CSV, "Dumbbell Curl", "dumbbell_curl"),
    AliasSpec(_CSV, "DB Curl", "dumbbell_curl"),
    AliasSpec(_CSV, "Hammer Curl", "hammer_curl"),
    AliasSpec(_CSV, "Triceps Pushdown", "triceps_pushdown"),
    AliasSpec(_CSV, "Tricep Pushdown", "triceps_pushdown"),
    AliasSpec(_CSV, "Pushdown", "triceps_pushdown"),
    AliasSpec(_CSV, "Plank", "plank"),
    AliasSpec(_CSV, "Front Plank", "plank"),
    AliasSpec(_CSV, "Side Plank", "side_plank"),
    AliasSpec(_CSV, "Hanging Leg Raise", "hanging_leg_raise"),
    AliasSpec(_CSV, "Leg Raise", "hanging_leg_raise"),
    AliasSpec(_CSV, "Back Extension", "back_extension"),
    AliasSpec(_CSV, "45 Degree Back Extension", "back_extension"),
)

# ---------------------------------------------------------------------------
# Labels PSD refuses to resolve
# ---------------------------------------------------------------------------
#
# The ambiguity policy as data. A label may only be listed here when PSD has seen
# it and knows that forcing a canonical exercise would invent information, and a
# label may never appear both here and in ``ALIASES``: the registry rejects that
# contradiction at construction, because it would mean PSD was arguing with
# itself about the same string.

CURATED_AMBIGUOUS_LABELS: Final[tuple[UnresolvedLabelSpec, ...]] = (
    UnresolvedLabelSpec(
        raw_label="Bench Variation",
        resolution_status=ResolutionStatus.PARTIAL_FAMILY,
        reason=AmbiguityReason.UNSPECIFIED_VARIATION,
        parent_lift=ParentLift.BENCH,
        note="A variation of the bench press without naming which one.",
    ),
    UnresolvedLabelSpec(
        raw_label="Squat Variation",
        resolution_status=ResolutionStatus.PARTIAL_FAMILY,
        reason=AmbiguityReason.UNSPECIFIED_VARIATION,
        parent_lift=ParentLift.SQUAT,
    ),
    UnresolvedLabelSpec(
        raw_label="Deadlift Variation",
        resolution_status=ResolutionStatus.PARTIAL_FAMILY,
        reason=AmbiguityReason.UNSPECIFIED_VARIATION,
        parent_lift=ParentLift.DEADLIFT,
    ),
    UnresolvedLabelSpec(
        raw_label="Variation",
        resolution_status=ResolutionStatus.UNMAPPED,
        reason=AmbiguityReason.UNSPECIFIED_VARIATION,
        note="A variation of nothing in particular.",
    ),
    UnresolvedLabelSpec(
        raw_label="Machine Press",
        resolution_status=ResolutionStatus.AMBIGUOUS,
        reason=AmbiguityReason.UNSPECIFIED_MACHINE,
        candidate_keys=("bench", "dumbbell_bench", "incline_bench"),
        note="A machine press could be a chest, shoulder, or leg machine; the label names none.",
    ),
    UnresolvedLabelSpec(
        raw_label="Chest Press",
        resolution_status=ResolutionStatus.AMBIGUOUS,
        reason=AmbiguityReason.UNSPECIFIED_IMPLEMENT,
        candidate_keys=("bench", "dumbbell_bench", "incline_bench"),
        note="Machine or free weights, and flat or incline: the label says neither.",
    ),
    UnresolvedLabelSpec(
        raw_label="Press",
        resolution_status=ResolutionStatus.AMBIGUOUS,
        reason=AmbiguityReason.UNSPECIFIED_IMPLEMENT,
        candidate_keys=("bench", "overhead_press"),
        note="Bench or overhead. Resolving this would invent the pressing pattern.",
    ),
    UnresolvedLabelSpec(
        raw_label="Squat Machine",
        resolution_status=ResolutionStatus.AMBIGUOUS,
        reason=AmbiguityReason.UNSPECIFIED_MACHINE,
        candidate_keys=("hack_squat", "leg_press"),
        note="Named as a squat machine without naming the machine.",
    ),
)

#: Labels deliberately fed to the pipeline to demonstrate that it refuses to guess.
#:
#: These are not curated knowledge -- the pipeline reaches the same verdict for
#: them on its own. Persisting their outcomes in the ontology artifact is what
#: turns "the ambiguity policy exists" into a claim a reviewer can check.
PROBE_LABELS: Final[tuple[str, ...]] = (
    "Leg Press?",
    "Squat Machine?",
    "Zorblax Lever Exercise",
    "Not An Exercise At All",
    "  ",
)
