"""Tests for the controlled vocabularies."""

from __future__ import annotations

import pytest

from psd.schema.vocabulary import (
    VOCABULARIES,
    AttemptResult,
    MissingnessReason,
    ObservationType,
    QualityFlag,
    SpecificityLevel,
    vocabulary_members,
)


def test_every_vocabulary_is_non_empty_and_lowercase() -> None:
    assert VOCABULARIES
    for name, members in VOCABULARIES.items():
        assert members, name
        assert len(members) == len(set(members)), name
        assert all(member == member.lower() for member in members), name
        assert all(member.strip() == member for member in members), name


def test_vocabulary_lookup() -> None:
    assert "good_lift" in vocabulary_members("attempt_result")
    assert "kg" in vocabulary_members("mass_unit")
    with pytest.raises(KeyError):
        vocabulary_members("does_not_exist")


def test_missingness_has_explicit_not_recorded_member() -> None:
    """Absence must be representable without borrowing a value."""
    assert MissingnessReason.NOT_RECORDED_IN_SOURCE.value == "not_recorded_in_source"
    assert MissingnessReason.UNKNOWN.value == "unknown"


def test_no_attempt_is_distinct_from_failed_lift() -> None:
    assert AttemptResult.NO_ATTEMPT is not AttemptResult.BAD_LIFT


def test_specificity_carries_no_transfer_coefficient() -> None:
    """Exercise semantics describe observables only; no transfer information."""
    assert SpecificityLevel.COMPETITION_LIFT.value == "competition_lift"
    members = vocabulary_members("specificity_level")
    assert not any("transfer" in member or "coefficient" in member for member in members)


def test_observation_types_are_instruments_not_latent_states() -> None:
    """Observations name recording instruments, never physiological ontology."""
    members = set(vocabulary_members("observation_type"))
    forbidden = {"fatigue_state", "readiness_state", "recovery", "form", "arousal_state"}
    assert not members & forbidden
    assert ObservationType.RPE.value == "rpe"
    assert ObservationType.SLEEP_DURATION.value == "sleep_duration"


def test_quality_flags_cover_trust_and_edit_history() -> None:
    members = set(vocabulary_members("quality_flag"))
    assert {"verified", "self_reported", "estimated", "edited_after_the_fact"} <= members


def test_vocabulary_enum_values_match_registry() -> None:
    assert tuple(member.value for member in QualityFlag) == vocabulary_members("quality_flag")
