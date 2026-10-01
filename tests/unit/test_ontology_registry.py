"""Unit tests for the ontology resolver: the ladder, ambiguity, and collisions.

Every test here maps to one of the policies the resolver promises. The group that
matters most is the ambiguity group: those tests assert that labels PSD *could* have
mapped are still left open, because raising coverage is never a reason to invent a
mapping.
"""

from __future__ import annotations

import pytest

from psd.ontology import default_ontology
from psd.ontology.catalog import (
    ALIASES,
    CURATED_AMBIGUOUS_LABELS,
    EXERCISES,
    AliasSpec,
    ExerciseSpec,
    UnresolvedLabelSpec,
)
from psd.ontology.registry import (
    DEFAULT_SOURCE_SYSTEM,
    MAPPING_CONFIDENCE,
    AliasCollisionError,
    ExerciseOntology,
    OntologyError,
    alias_id_for,
    exercise_id_for,
)
from psd.schema.vocabulary import (
    AmbiguityReason,
    ParentLift,
    PauseRule,
    RangeOfMotion,
    ResolutionMethod,
    ResolutionStatus,
    SpecificityLevel,
)

ONTOLOGY = default_ontology()


# ---------------------------------------------------------------------------
# the ladder
# ---------------------------------------------------------------------------


def test_a_canonical_name_resolves_by_identity() -> None:
    outcome = ONTOLOGY.resolve("Low Bar Squat")
    assert outcome.resolution_status is ResolutionStatus.EXACT_CANONICAL
    assert outcome.resolution_method is ResolutionMethod.CANONICAL_IDENTITY
    assert outcome.exercise_key == "low_bar_squat"
    assert outcome.confidence == 1.0


def test_a_canonical_key_resolves_by_identity() -> None:
    """An adapter may hand PSD the key it already knows; that must be idempotent."""
    assert ONTOLOGY.resolve("low_bar_squat").exercise_key == "low_bar_squat"
    assert ONTOLOGY.resolve("LOW_BAR_SQUAT").exercise_key == "low_bar_squat"


def test_a_canonical_identifier_resolves_by_identity() -> None:
    outcome = ONTOLOGY.resolve(exercise_id_for("close_grip_bench"))
    assert outcome.resolution_status is ResolutionStatus.EXACT_CANONICAL
    assert outcome.exercise_key == "close_grip_bench"


def test_an_unknown_identifier_is_not_treated_as_an_identity() -> None:
    """A well-formed identifier PSD never minted resolves to nothing."""
    outcome = ONTOLOGY.resolve("exd_" + "0" * 32)
    assert outcome.is_resolved is False


def test_a_registered_alias_reports_its_namespace() -> None:
    outcome = ONTOLOGY.resolve("CGBP", source_system="hevy")
    assert outcome.resolution_status is ResolutionStatus.RESOLVED_ALIAS
    assert outcome.resolution_method is ResolutionMethod.REGISTERED_ALIAS
    assert outcome.source_alias_id is not None


def test_an_alias_only_another_namespace_knows_is_reported_as_cross_source() -> None:
    """A weaker claim, and visibly weaker: the confidence drops and the method says so.

    A different app's vocabulary may mean something subtly different, so the outcome
    names a concrete alias row rather than pretending the registry always knew this
    label in this namespace. The label is deliberately one PSD's own vocabulary does
    not contain, so the identity stage cannot short-circuit it.
    """
    outcome = ONTOLOGY.resolve("SSB Squat", source_system="hevy")
    expected = MAPPING_CONFIDENCE[ResolutionMethod.CROSS_SOURCE_ALIAS]
    assert outcome.resolution_method is ResolutionMethod.CROSS_SOURCE_ALIAS
    assert outcome.confidence == expected
    assert expected < MAPPING_CONFIDENCE[ResolutionMethod.REGISTERED_ALIAS]


def test_the_requesting_namespace_wins_over_a_cross_source_hit() -> None:
    with_namespace = ONTOLOGY.resolve("Paused Bench", source_system="hevy")
    without_namespace = ONTOLOGY.resolve("Paused Bench", source_system="generic_csv")
    assert with_namespace.resolution_method is ResolutionMethod.REGISTERED_ALIAS
    assert without_namespace.resolution_method is ResolutionMethod.CROSS_SOURCE_ALIAS
    assert with_namespace.exercise_key is not None
    assert without_namespace.exercise_key is not None
    assert with_namespace.exercise_key == without_namespace.exercise_key


def test_structured_interpretation_resolves_an_unregistered_spelling() -> None:
    outcome = ONTOLOGY.resolve("Paused Back Squat")
    assert outcome.resolution_method is ResolutionMethod.STRUCTURED_INTERPRETATION
    assert outcome.exercise_key == "pause_squat"


def test_structured_interpretation_refuses_to_synthesize_a_new_combination() -> None:
    """No exercise exists for "high bar paused squat", so none is created.

    A parser that produced one would be asserting that the combination is a real
    training variation with its own identity -- which is a scientific claim PSD has
    no source for.
    """
    outcome = ONTOLOGY.resolve("high bar paused squat")
    assert outcome.is_resolved is False
    assert "high_bar_squat" not in outcome.candidate_keys


def test_family_keyword_resolves_only_the_family() -> None:
    outcome = ONTOLOGY.resolve("bench")
    assert outcome.resolution_status is ResolutionStatus.EXACT_CANONICAL
    unregistered = ONTOLOGY.resolve("Bench Exercise")
    assert unregistered.resolution_status is ResolutionStatus.PARTIAL_FAMILY
    assert unregistered.parent_lift is ParentLift.BENCH
    assert unregistered.exercise_key is None


def test_generic_qualifier_caps_a_label_at_the_family() -> None:
    """A qualifier PSD has never curated still caps the label at the family.

    The curated table runs first, so this uses a qualifier spelling that is not
    itself curated: the generic rule has to hold for labels nobody wrote down.
    """
    outcome = ONTOLOGY.resolve("Deadlift Vars")
    assert outcome.resolution_status is ResolutionStatus.PARTIAL_FAMILY
    assert outcome.resolution_method is ResolutionMethod.GENERIC_QUALIFIER
    assert outcome.ambiguity_reason is AmbiguityReason.UNSPECIFIED_VARIATION
    assert outcome.parent_lift is ParentLift.DEADLIFT


# ---------------------------------------------------------------------------
# ambiguity policy
# ---------------------------------------------------------------------------

#: ``(label, status, reason)`` for the labels the issue requires to stay open.
REQUIRED_UNRESOLVED: tuple[tuple[str, ResolutionStatus, AmbiguityReason], ...] = (
    ("Bench variation", ResolutionStatus.PARTIAL_FAMILY, AmbiguityReason.UNSPECIFIED_VARIATION),
    ("Deadlift variation", ResolutionStatus.PARTIAL_FAMILY, AmbiguityReason.UNSPECIFIED_VARIATION),
    ("Machine press", ResolutionStatus.AMBIGUOUS, AmbiguityReason.UNSPECIFIED_MACHINE),
    ("Squat machine", ResolutionStatus.AMBIGUOUS, AmbiguityReason.UNSPECIFIED_MACHINE),
    ("Chest Press", ResolutionStatus.AMBIGUOUS, AmbiguityReason.UNSPECIFIED_IMPLEMENT),
    ("Press", ResolutionStatus.AMBIGUOUS, AmbiguityReason.UNSPECIFIED_IMPLEMENT),
    ("Leg press?", ResolutionStatus.AMBIGUOUS, AmbiguityReason.QUESTION_FORM_LABEL),
    ("Zorblax Lever Exercise", ResolutionStatus.UNMAPPED, AmbiguityReason.UNKNOWN_SOURCE_TAXONOMY),
    ("   ", ResolutionStatus.UNMAPPED, AmbiguityReason.UNKNOWN_SOURCE_TAXONOMY),
    ("", ResolutionStatus.UNMAPPED, AmbiguityReason.UNKNOWN_SOURCE_TAXONOMY),
)


@pytest.mark.parametrize(("label", "status", "reason"), REQUIRED_UNRESOLVED)
@pytest.mark.parametrize("source_system", ["psd_registry", "hevy", "generic_csv"])
def test_underspecified_labels_are_never_forced_into_an_exercise(
    label: str, status: ResolutionStatus, reason: AmbiguityReason, source_system: str
) -> None:
    outcome = ONTOLOGY.resolve(label, source_system=source_system)
    assert outcome.resolution_status is status
    assert outcome.ambiguity_reason is reason
    assert outcome.exercise_key is None
    assert outcome.exercise_id is None
    assert outcome.confidence is None


def test_an_ambiguity_lists_its_defensible_readings() -> None:
    outcome = ONTOLOGY.resolve("Machine press")
    assert outcome.candidate_keys == ("bench", "dumbbell_bench", "incline_bench")
    assert all(key != outcome.exercise_key for key in outcome.candidate_keys)


def test_a_partial_family_resolution_names_no_candidates() -> None:
    """Family-only and ambiguous are different rungs, not two ways of saying "maybe"."""
    outcome = ONTOLOGY.resolve("Bench variation")
    assert outcome.candidate_keys == ()
    assert outcome.parent_lift is ParentLift.BENCH


def test_a_question_form_label_keeps_the_candidate_it_refused_to_choose() -> None:
    """The answer PSD could have given stays visible next to the doubt it recorded."""
    outcome = ONTOLOGY.resolve("Leg press?")
    assert outcome.candidate_keys == ("leg_press",)
    assert outcome.resolution_method is ResolutionMethod.QUESTION_FORM


def test_a_question_form_on_an_unknown_label_stays_unmapped() -> None:
    outcome = ONTOLOGY.resolve("Zorblax Lever?")
    assert outcome.resolution_status is ResolutionStatus.UNMAPPED
    assert outcome.candidate_keys == ()


def test_the_raw_label_survives_verbatim() -> None:
    """Normalization reduces the lookup key; it never rewrites the source string."""
    for label in ("  Low-Bar   Squat  ", "Machine press", "Leg press?", "DB Bench"):
        assert ONTOLOGY.resolve(label).raw_label == label


def test_refusals_are_stable_across_case_and_spacing() -> None:
    baseline = ONTOLOGY.resolve("machine press")
    for variant in ("MACHINE PRESS", "  Machine   Press  ", "Machine-Press"):
        outcome = ONTOLOGY.resolve(variant)
        assert outcome.resolution_status == baseline.resolution_status
        assert outcome.candidate_keys == baseline.candidate_keys


def test_coverage_is_not_improved_by_forcing() -> None:
    """Forcing every curated refusal to resolve would be a regression, not progress.

    This test states the policy in the only terms a future change can be measured
    against: the number of labels PSD declines to map must not shrink without an
    explicit decision in the catalog.
    """
    assert len(CURATED_AMBIGUOUS_LABELS) == 8
    unresolved = sum(
        1 for spec in CURATED_AMBIGUOUS_LABELS if not ONTOLOGY.resolve(spec.raw_label).is_resolved
    )
    assert unresolved == len(CURATED_AMBIGUOUS_LABELS)


# ---------------------------------------------------------------------------
# determinism and idempotence
# ---------------------------------------------------------------------------

#: Every label the artifact and catalog exercise, resolved repeatedly.
ALL_KNOWN_LABELS: tuple[str, ...] = (
    *(binding.raw_label for binding in ONTOLOGY.aliases),
    *(spec.raw_label for spec in CURATED_AMBIGUOUS_LABELS),
    *(spec.canonical_name for spec in EXERCISES),
    *(spec.key.replace("_", " ") for spec in EXERCISES),
    "Leg press?",
    "Machine press",
    "Zorblax Lever Exercise",
    "   ",
)


@pytest.mark.parametrize("label", ALL_KNOWN_LABELS)
def test_resolution_is_repeatable(label: str) -> None:
    first = ONTOLOGY.resolve(label, source_system="hevy")
    second = ONTOLOGY.resolve(label, source_system="hevy")
    assert first == second


@pytest.mark.parametrize("label", ALL_KNOWN_LABELS)
def test_resolving_a_canonical_exercise_is_idempotent(label: str) -> None:
    """Normalizing a resolved exercise again must not change what it means.

    An ingestion pipeline may re-normalize a label it has already resolved -- after a
    schema migration, say -- and that second pass must land on the same exercise.
    """
    first = ONTOLOGY.resolve(label, source_system="hevy")
    if first.exercise_key is None:
        pytest.skip("label is deliberately unresolved")
    canonical = ONTOLOGY.spec(first.exercise_key)
    for representation in (
        canonical.canonical_name,
        canonical.key,
        canonical.canonical_name.upper(),
        f"  {canonical.canonical_name}  ",
        ONTOLOGY.resolve(label, source_system="hevy").normalized_label,
    ):
        again = ONTOLOGY.resolve(representation, source_system="hevy")
        assert again.exercise_key == first.exercise_key, representation


def test_every_registered_alias_resolves_to_the_exercise_it_claims() -> None:
    """The registry must not assert a binding its own resolver contradicts.

    Iterating the bindings rather than parametrizing over labels keeps the check
    honest for labels that appear more than once: a spelling registered in two
    namespaces must resolve to the same exercise from both.
    """
    for binding in ONTOLOGY.aliases:
        outcome = ONTOLOGY.resolve(binding.raw_label, source_system=binding.source_system)
        assert outcome.exercise_key == binding.exercise_key, binding.raw_label


def test_every_canonical_exercise_reaches_itself_by_name_and_key() -> None:
    for spec in EXERCISES:
        for representation in (spec.canonical_name, spec.key):
            outcome = ONTOLOGY.resolve(representation)
            assert outcome.exercise_key == spec.key, representation


def test_resolution_does_not_depend_on_query_order() -> None:
    forward = [ONTOLOGY.resolve(label).resolution_status for label in ALL_KNOWN_LABELS]
    backward = [ONTOLOGY.resolve(label).resolution_status for label in reversed(ALL_KNOWN_LABELS)]
    assert forward == list(reversed(backward))


def test_candidate_lists_are_sorted_and_deduplicated() -> None:
    for label in ALL_KNOWN_LABELS:
        candidates = ONTOLOGY.resolve(label).candidate_keys
        assert list(candidates) == sorted(set(candidates))


# ---------------------------------------------------------------------------
# alias collisions and registry integrity
# ---------------------------------------------------------------------------


def _squat() -> ExerciseSpec:
    return ExerciseSpec(
        key="squat",
        canonical_name="Squat",
        parent_lift=ParentLift.SQUAT,
        specificity_level=SpecificityLevel.COMPETITION_LIFT,
        range_of_motion=RangeOfMotion.COMPETITION,
        pause_rule=PauseRule.COMPETITION,
    )


def _other() -> ExerciseSpec:
    return ExerciseSpec(
        key="wide_grip_bench",
        canonical_name="Wide-Grip Bench Press",
        parent_lift=ParentLift.BENCH,
        specificity_level=SpecificityLevel.COMPETITION_VARIATION,
        range_of_motion=RangeOfMotion.COMPETITION,
        pause_rule=PauseRule.COMPETITION,
    )


def test_one_label_cannot_bind_two_exercises_in_one_namespace() -> None:
    """The registry refuses rather than letting coverage decide the semantics."""
    with pytest.raises(AliasCollisionError, match="conflicting canonical identities"):
        ExerciseOntology(
            [_squat(), _other()],
            [
                AliasSpec("hevy", "The Lift", "squat"),
                AliasSpec("hevy", "the  lift", "wide_grip_bench"),
            ],
            (),
        )


def test_one_label_cannot_bind_two_exercises_across_namespaces() -> None:
    """Two apps may spell things differently, but not identically-but-differently.

    Scoping the rule per source would make a lookup depend on which namespace happened
    to register the spelling first, which is exactly the nondeterminism the registry
    exists to remove.
    """
    with pytest.raises(AliasCollisionError, match="conflicting canonical identities"):
        ExerciseOntology(
            [_squat(), _other()],
            [
                AliasSpec("hevy", "The Lift", "squat"),
                AliasSpec("strong", "the-lift", "wide_grip_bench"),
            ],
            (),
        )


def test_two_spellings_of_one_exercise_are_both_kept() -> None:
    """The opposite case: this is not a collision, it is provenance."""
    registry = ExerciseOntology(
        [_squat(), _other()],
        [AliasSpec("hevy", "Lateral Raise", "squat"), AliasSpec("hevy", "lateral raises", "squat")],
        (),
    )
    spellings = {binding.raw_label for binding in registry.aliases}
    assert spellings == {"Lateral Raise", "lateral raises"}


def test_a_label_cannot_be_both_an_alias_and_a_curated_refusal() -> None:
    with pytest.raises(AliasCollisionError, match="curated as unresolved"):
        ExerciseOntology(
            [_squat(), _other()],
            [AliasSpec("hevy", "Squat Variation", "squat")],
            (
                UnresolvedLabelSpec(
                    raw_label="Squat Variation",
                    resolution_status=ResolutionStatus.PARTIAL_FAMILY,
                    reason=AmbiguityReason.UNSPECIFIED_VARIATION,
                    parent_lift=ParentLift.SQUAT,
                ),
            ),
        )


def test_a_curated_refusal_cannot_name_an_undeclared_exercise() -> None:
    with pytest.raises(OntologyError, match="unknown canonical key"):
        ExerciseOntology(
            [_squat()],
            (),
            (
                UnresolvedLabelSpec(
                    raw_label="Something",
                    resolution_status=ResolutionStatus.AMBIGUOUS,
                    reason=AmbiguityReason.UNSPECIFIED_MACHINE,
                    candidate_keys=("wide_grip_bench",),
                ),
            ),
        )


def _other_spec(
    key: str,
    canonical_name: str,
    parent_lift: ParentLift,
    specificity_level: SpecificityLevel,
) -> ExerciseSpec:
    """Build a second competition lift with the same shape as :func:`_other`."""
    return ExerciseSpec(
        key=key,
        canonical_name=canonical_name,
        parent_lift=parent_lift,
        specificity_level=specificity_level,
        range_of_motion=RangeOfMotion.COMPETITION,
        pause_rule=PauseRule.COMPETITION,
    )


def test_two_canonical_keys_cannot_normalize_to_one_spelling() -> None:
    """A second key whose name reads as the first would make lookup ambiguous."""
    with pytest.raises(AliasCollisionError, match="one spelling may identify only one"):
        ExerciseOntology(
            [
                _squat(),
                _other_spec(
                    key="squat_2",
                    canonical_name="squat",
                    parent_lift=ParentLift.SQUAT,
                    specificity_level=SpecificityLevel.COMPETITION_LIFT,
                ),
            ],
            (),
            (),
        )


def test_duplicate_canonical_keys_are_rejected() -> None:
    with pytest.raises(OntologyError, match="Duplicate canonical exercise key"):
        ExerciseOntology([_squat(), _squat()], (), ())


def test_an_empty_ontology_is_rejected() -> None:
    with pytest.raises(OntologyError, match="no canonical exercises"):
        ExerciseOntology([], (), ())


def test_an_alias_to_an_undeclared_exercise_is_rejected() -> None:
    with pytest.raises(OntologyError, match="unknown canonical key"):
        ExerciseOntology([_squat()], [AliasSpec("hevy", "Rack Pull", "rack_pull")], ())


def test_resolving_a_non_string_raises() -> None:
    with pytest.raises(TypeError, match="expects str"):
        ONTOLOGY.resolve(None)  # type: ignore[arg-type]


def test_unknown_canonical_key_lookup_raises() -> None:
    with pytest.raises(KeyError, match="Unknown canonical exercise key"):
        ONTOLOGY.spec("not_an_exercise")


def test_alias_ids_are_distinct_per_spelling() -> None:
    first = alias_id_for("hevy", "Lateral Raise", "lateral_raise")
    second = alias_id_for("hevy", "Lateral Raises", "lateral_raise")
    assert first != second
    assert alias_id_for("hevy", "Lateral Raise", "lateral_raise") == first


# ---------------------------------------------------------------------------
# the shipped registry
# ---------------------------------------------------------------------------


def test_shipped_registry_has_the_three_competition_lifts() -> None:
    for key, parent in (
        ("squat", ParentLift.SQUAT),
        ("bench", ParentLift.BENCH),
        ("deadlift", ParentLift.DEADLIFT),
    ):
        spec = ONTOLOGY.spec(key)
        assert spec.parent_lift is parent
        assert spec.specificity_level is SpecificityLevel.COMPETITION_LIFT


def test_shipped_registry_covers_every_parent_lift() -> None:
    for parent in (ParentLift.SQUAT, ParentLift.BENCH, ParentLift.DEADLIFT, ParentLift.ACCESSORY):
        assert ONTOLOGY.keys_for_parent(parent), parent


def test_shipped_aliases_all_carry_a_namespace_and_a_note_free_or_reasoned_binding() -> None:
    for binding in ONTOLOGY.aliases:
        assert binding.source_system
        assert binding.raw_label.strip()
        assert binding.normalized_label.strip()
    assert sum(1 for binding in ONTOLOGY.aliases if binding.note) >= 1


def test_shipped_alias_count_is_at_least_the_required_coverage() -> None:
    assert len(ONTOLOGY.aliases) >= 100
    assert len(EXERCISES) >= 50
    assert len(ALIASES) == len(ONTOLOGY.aliases)


def test_default_namespace_is_psds_own() -> None:
    assert DEFAULT_SOURCE_SYSTEM == "psd_registry"


def test_ontology_reports_its_versions() -> None:
    assert ONTOLOGY.ontology_version.tag == "psd-ontology/0.1.0"
    assert ONTOLOGY.alias_registry_version.tag == "psd-ontology-alias/0.1.0"
