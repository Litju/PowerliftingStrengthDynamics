"""Unit tests for the exercise-ontology descriptor and resolution vocabularies.

These assert that every vocabulary the ontology persists is a declared member, that
the neutral "not stated" member is distinct from every meaningful value, and that
the whole registry is reachable through :func:`vocabulary_members` so the declarative
column validator can check an artifact without importing the ontology.
"""

from __future__ import annotations

from enum import StrEnum

import pytest

from psd.schema.vocabulary import (
    TEXT_NORMALIZATION_RULES,
    VOCABULARIES,
    AliasSourceSystem,
    AmbiguityReason,
    BarType,
    ConfigurationFlag,
    ExerciseEquipment,
    Grip,
    PauseRule,
    RangeOfMotion,
    ResolutionMethod,
    ResolutionStatus,
    Stance,
    TempoPattern,
    vocabulary_members,
)

EXERCISE_VOCABULARIES: tuple[str, ...] = (
    "stance",
    "grip",
    "range_of_motion",
    "pause_rule",
    "tempo",
    "bar_type",
    "equipment",
    "configuration_flag",
    "resolution_status",
    "resolution_method",
    "ambiguity_reason",
    "text_rule",
)


@pytest.mark.parametrize("name", EXERCISE_VOCABULARIES)
def test_every_exercise_vocabulary_is_registered(name: str) -> None:
    assert vocabulary_members(name)
    assert name in VOCABULARIES


@pytest.mark.parametrize(
    ("vocabulary", "member_type"),
    [
        ("stance", Stance),
        ("grip", Grip),
        ("range_of_motion", RangeOfMotion),
        ("pause_rule", PauseRule),
        ("tempo", TempoPattern),
        ("bar_type", BarType),
        ("equipment", ExerciseEquipment),
        ("configuration_flag", ConfigurationFlag),
        ("resolution_status", ResolutionStatus),
        ("resolution_method", ResolutionMethod),
        ("ambiguity_reason", AmbiguityReason),
    ],
)
def test_vocabulary_members_match_their_enum(vocabulary: str, member_type: type[StrEnum]) -> None:
    assert set(vocabulary_members(vocabulary)) == {item.value for item in member_type}


@pytest.mark.parametrize(
    ("vocabulary", "member_type"),
    [
        ("stance", Stance),
        ("grip", Grip),
        ("range_of_motion", RangeOfMotion),
        ("pause_rule", PauseRule),
        ("tempo", TempoPattern),
    ],
)
def test_every_descriptor_has_a_not_specified_member(
    vocabulary: str, member_type: type[StrEnum]
) -> None:
    """A descriptor must be able to say "the source did not say".

    Without a neutral member, every label would have to invent a stance and a grip,
    and the ontology would encode those inventions as facts.
    """
    assert "not_specified" in {item.value for item in member_type}, vocabulary


@pytest.mark.parametrize(
    ("vocabulary", "member_type"),
    [
        ("bar_type", BarType),
        ("equipment", ExerciseEquipment),
    ],
)
def test_apparatus_vocabularies_separate_not_applicable_from_none(
    vocabulary: str, member_type: type[StrEnum]
) -> None:
    """Bar and apparatus separate "does not apply" from "not needed but applicable".

    A dumbbell curl has no bar at all, which is a fact about the exercise, whereas an
    olympic-bar squat needing no rack is the absence of an apparatus requirement.
    Collapsing the two would assert that every apparatus-free exercise lacks a bar.
    """
    values = {item.value for item in member_type}
    neutral = "not_applicable" if member_type is BarType else "none"
    assert neutral in values, vocabulary
    assert "none" in values or member_type is BarType, vocabulary


@pytest.mark.parametrize(
    ("vocabulary", "member_type"),
    [
        ("stance", Stance),
        ("grip", Grip),
        ("range_of_motion", RangeOfMotion),
        ("pause_rule", PauseRule),
        ("tempo", TempoPattern),
        ("bar_type", BarType),
        ("equipment", ExerciseEquipment),
    ],
)
def test_every_descriptor_vocabulary_has_an_unknown_member(
    vocabulary: str, member_type: type[StrEnum]
) -> None:
    """``unknown`` means "PSD has never seen this", which differs from "not said"."""
    assert "unknown" in {item.value for item in member_type}


def test_resolution_vocabularies_have_no_unknown_member() -> None:
    """An outcome is never "unknown": the resolver always reaches a verdict.

    ``unknown`` exists on descriptor vocabularies because a source may simply not
    have said. A normalization outcome is PSD's own decision, and declining to record
    one would defeat the purpose of the audit trail.
    """
    for member_type in (ResolutionStatus, ResolutionMethod):
        assert "unknown" not in {item.value for item in member_type}


def test_resolution_status_covers_every_ladder_rung() -> None:
    assert {member.value for member in ResolutionStatus} == {
        "exact_canonical",
        "resolved_alias",
        "partial_family",
        "ambiguous",
        "unmapped",
    }


def test_resolution_status_separates_resolved_rungs_from_refusal_rungs() -> None:
    """The ladder's whole point: a refusal is a distinct outcome, not a weak yes."""
    resolved = {ResolutionStatus.EXACT_CANONICAL, ResolutionStatus.RESOLVED_ALIAS}
    refused = {
        ResolutionStatus.PARTIAL_FAMILY,
        ResolutionStatus.AMBIGUOUS,
        ResolutionStatus.UNMAPPED,
    }
    assert resolved.isdisjoint(refused)
    assert resolved | refused == set(ResolutionStatus)


def test_ambiguity_reason_covers_the_deliberate_refusals() -> None:
    values = {member.value for member in AmbiguityReason}
    assert {
        "unspecified_variation",
        "unspecified_machine",
        "unspecified_implement",
        "multiple_defensible_matches",
        "question_form_label",
        "unknown_source_taxonomy",
    } <= values


def test_configuration_flag_values_are_sorted_and_unique() -> None:
    values = [member.value for member in ConfigurationFlag]
    assert values == sorted(values)
    assert len(values) == len(set(values))


def test_alias_source_systems_are_lowercase_namespace_tokens() -> None:
    for member in AliasSourceSystem:
        assert member.value == member.value.lower()
        assert member.value.replace("_", "").isalnum()


def test_text_normalization_rules_are_unique_lowercase_tokens() -> None:
    """Rules are declared in application order, so uniqueness is what is asserted."""
    assert len(set(TEXT_NORMALIZATION_RULES)) == len(TEXT_NORMALIZATION_RULES)
    for rule in TEXT_NORMALIZATION_RULES:
        assert rule == rule.lower()
        assert rule.replace("_", "").isalnum()
    assert vocabulary_members("text_rule") == TEXT_NORMALIZATION_RULES


def test_text_normalization_rules_are_declared_in_application_order() -> None:
    assert TEXT_NORMALIZATION_RULES.index("unicode_nfkc") < TEXT_NORMALIZATION_RULES.index(
        "unicode_casefold"
    )
    assert TEXT_NORMALIZATION_RULES.index("expand_abbreviation") < TEXT_NORMALIZATION_RULES.index(
        "fold_plural"
    )
    # Plural folding must precede compound joining, or "bench presses" would miss the
    # join table and become a second lookup key for one exercise.
    assert TEXT_NORMALIZATION_RULES.index("fold_plural") < TEXT_NORMALIZATION_RULES.index(
        "join_compound"
    )


def test_unknown_vocabulary_raises() -> None:
    with pytest.raises(KeyError):
        vocabulary_members("not_a_vocabulary")
