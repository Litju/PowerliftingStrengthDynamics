"""Unit tests for converting the ontology into canonical records.

The bridge between the two must be lossless in both directions: every descriptor the
ontology declares has to reach the canonical record, and every outcome the resolver
produces has to persist without losing the reason it reached its verdict.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from psd.ontology import default_ontology, exercise_id_for
from psd.ontology.records import (
    REGISTRY_SOURCE_ID,
    alias_records,
    definition_record_for,
    definition_records,
    normalization_id_for,
    normalization_records,
    probe_outcomes,
    resolution_for,
)
from psd.ontology.registry import DEFAULT_SOURCE_SYSTEM
from psd.ontology.text import normalize_label
from psd.schema.models import ExerciseNormalizationRecord
from psd.schema.vocabulary import (
    AmbiguityReason,
    ResolutionMethod,
    ResolutionStatus,
)

ONTOLOGY = default_ontology()
INGESTED_AT = datetime(2021, 1, 4, 7, 0, tzinfo=UTC)


# ---------------------------------------------------------------------------
# definitions
# ---------------------------------------------------------------------------


def test_definition_record_carries_every_declared_descriptor() -> None:
    spec = ONTOLOGY.spec("feet_up_bench")
    record = definition_record_for(ONTOLOGY, "feet_up_bench", ingested_at=INGESTED_AT)
    assert record.canonical_key == spec.key
    assert record.canonical_name == spec.canonical_name
    assert record.parent_lift == spec.parent_lift
    assert record.specificity_level == spec.specificity_level
    assert record.implement == spec.implement
    assert record.bar_type == spec.bar_type
    assert record.equipment == spec.equipment
    assert record.laterality == spec.laterality
    assert record.stance == spec.stance
    assert record.grip == spec.grip
    assert record.range_of_motion == spec.range_of_motion
    assert record.pause_rule == spec.pause_rule
    assert record.tempo == spec.tempo
    assert record.configuration == spec.configuration
    assert record.definition_note == spec.note


def test_definition_records_are_sorted_by_canonical_key() -> None:
    records = definition_records(ONTOLOGY, ingested_at=INGESTED_AT)
    keys = [record.canonical_key for record in records]
    assert keys == sorted(keys)


def test_definition_records_ignore_the_order_keys_were_supplied_in() -> None:
    """The result must not depend on a caller's ordering of the same set."""
    forward = definition_records(ONTOLOGY, ingested_at=INGESTED_AT, keys=["squat", "bench", "rdl"])
    backward = definition_records(ONTOLOGY, ingested_at=INGESTED_AT, keys=["rdl", "bench", "squat"])
    assert [record.exercise_id for record in forward] == [record.exercise_id for record in backward]


def test_definition_records_can_be_restricted_to_a_subset() -> None:
    records = definition_records(ONTOLOGY, ingested_at=INGESTED_AT, keys=["bench", "pause_bench"])
    assert {record.canonical_key for record in records} == {"bench", "pause_bench"}


def test_definition_records_carry_registry_provenance_by_default() -> None:
    record = definition_record_for(ONTOLOGY, "squat", ingested_at=INGESTED_AT)
    assert record.source_id == REGISTRY_SOURCE_ID
    assert record.source_record_key == "ontology:exercise:squat"
    assert record.ingested_at == INGESTED_AT


def test_definition_records_accept_a_caller_supplied_source() -> None:
    record = definition_record_for(
        ONTOLOGY, "squat", source_id="src_hevy_export", ingested_at=INGESTED_AT
    )
    assert record.source_id == "src_hevy_export"


# ---------------------------------------------------------------------------
# aliases
# ---------------------------------------------------------------------------


def test_alias_records_carry_namespace_versions_and_exercise_link() -> None:
    record = next(
        item
        for item in alias_records(ONTOLOGY, ingested_at=INGESTED_AT)
        if item.alias_raw == "CGBP"
    )
    assert record.source_system
    assert record.exercise_id == exercise_id_for("close_grip_bench")
    assert record.mapping_version == ONTOLOGY.alias_registry_version.tag
    assert record.ontology_version == ONTOLOGY.ontology_version.tag
    assert record.mapping_status is ResolutionStatus.RESOLVED_ALIAS


def test_alias_records_can_be_filtered_by_namespace_and_exercise() -> None:
    records = alias_records(
        ONTOLOGY,
        ingested_at=INGESTED_AT,
        source_systems=["hevy"],
        exercise_keys=["close_grip_bench"],
    )
    assert {record.alias_raw for record in records} == {"Close Grip Bench Press", "CGBP"}
    assert all(record.source_system == "hevy" for record in records)


def test_alias_records_are_sorted_deterministically() -> None:
    records = alias_records(ONTOLOGY, ingested_at=INGESTED_AT)
    keys = [(r.source_system, r.alias_normalized, r.alias_raw) for r in records]
    assert keys == sorted(keys)


def test_every_alias_normalized_label_recomputes_from_its_raw_label() -> None:
    """The persisted normalized form must match what the normalizer produces today.

    If these ever diverge, a persisted artifact would index aliases under a key the
    resolver can no longer reach.
    """

    for record in alias_records(ONTOLOGY, ingested_at=INGESTED_AT):
        assert record.alias_normalized == normalize_label(record.alias_raw).text


# ---------------------------------------------------------------------------
# normalization outcomes
# ---------------------------------------------------------------------------


def test_resolution_record_preserves_the_raw_label_and_the_rules_that_fired() -> None:
    outcome = ONTOLOGY.resolve("  Low-Bar   Squat  ", source_system="hevy")
    record = resolution_for(outcome, ontology=ONTOLOGY, ingested_at=INGESTED_AT)
    assert record.raw_label == "  Low-Bar   Squat  "
    assert record.normalized_label == "lowbar squat"
    assert "collapse_whitespace" in record.normalization_rules
    assert record.normalization_rules == tuple(sorted(set(record.normalization_rules)))


def test_resolved_record_names_the_canonical_exercise_and_a_confidence() -> None:
    outcome = ONTOLOGY.resolve("Low Bar Squat")
    record = resolution_for(outcome, ontology=ONTOLOGY, ingested_at=INGESTED_AT)
    assert record.exercise_id == exercise_id_for("low_bar_squat")
    assert record.confidence == outcome.confidence
    assert record.candidate_exercise_ids == ()
    assert record.ambiguity_reason is None


def test_ambiguous_record_names_candidates_and_no_exercise() -> None:
    outcome = ONTOLOGY.resolve("Machine press")
    record = resolution_for(outcome, ontology=ONTOLOGY, ingested_at=INGESTED_AT)
    assert record.exercise_id is None
    assert record.confidence is None
    assert record.candidate_exercise_ids == tuple(
        sorted(exercise_id_for(key) for key in outcome.candidate_keys)
    )
    assert record.ambiguity_reason is AmbiguityReason.UNSPECIFIED_MACHINE


def test_partial_family_record_names_the_family_only() -> None:
    outcome = ONTOLOGY.resolve("Bench Variation")
    record = resolution_for(outcome, ontology=ONTOLOGY, ingested_at=INGESTED_AT)
    assert record.resolution_status is ResolutionStatus.PARTIAL_FAMILY
    assert record.parent_lift.value == "bench"
    assert record.exercise_id is None
    assert record.candidate_exercise_ids == ()


def test_blank_label_record_stays_usable() -> None:
    """A whitespace-only source label is unmapped, not an error.

    The normalized form would be empty, which the column forbids, so the verbatim
    string stands in; the row still says ``unmapped``.
    """
    outcome = ONTOLOGY.resolve("   ")
    record = resolution_for(outcome, ontology=ONTOLOGY, ingested_at=INGESTED_AT)
    assert record.resolution_status is ResolutionStatus.UNMAPPED
    assert record.normalized_label == "   "
    assert record.raw_label == "   "


def test_normalization_ids_are_distinct_per_raw_spelling() -> None:
    outcomes = [ONTOLOGY.resolve(label) for label in ("Bench Press", "Benchpress", "benchpress")]
    records = normalization_records(outcomes, ontology=ONTOLOGY, ingested_at=INGESTED_AT)
    assert len({record.normalization_id for record in records}) == 3
    assert len({record.normalized_label for record in records}) == 1


def test_normalization_id_depends_on_the_verbatim_label() -> None:
    first = normalization_id_for("hevy", "Bench Press", "benchpress")
    second = normalization_id_for("hevy", "Benchpress", "benchpress")
    assert first != second
    assert normalization_id_for("strong", "Bench Press", "benchpress") != first


def test_normalization_records_are_sorted_deterministically() -> None:
    outcomes = probe_outcomes(ONTOLOGY, source_system=DEFAULT_SOURCE_SYSTEM)
    records = normalization_records(outcomes, ontology=ONTOLOGY, ingested_at=INGESTED_AT)
    keys = [(r.source_system, r.normalized_label, r.raw_label) for r in records]
    assert keys == sorted(keys)


def test_every_probe_outcome_persists() -> None:
    """Including the refusals: an artifact that hid them could not be reviewed."""
    outcomes = probe_outcomes(ONTOLOGY, source_system=DEFAULT_SOURCE_SYSTEM)
    records = normalization_records(outcomes, ontology=ONTOLOGY, ingested_at=INGESTED_AT)
    statuses = {record.resolution_status for record in records}
    assert statuses & {ResolutionStatus.UNMAPPED, ResolutionStatus.AMBIGUOUS}
    assert all(isinstance(record, ExerciseNormalizationRecord) for record in records)


# ---------------------------------------------------------------------------
# the normalization contract cannot be overstate itself
# ---------------------------------------------------------------------------


def _record(**overrides: object) -> ExerciseNormalizationRecord:
    base: dict[str, object] = {
        "normalization_id": "exn_" + "0" * 32,
        "raw_label": "Low Bar Squat",
        "normalized_label": "lowbar squat",
        "source_system": "hevy",
        "resolution_status": ResolutionStatus.RESOLVED_ALIAS,
        "resolution_method": ResolutionMethod.REGISTERED_ALIAS,
        "exercise_id": exercise_id_for("low_bar_squat"),
        "confidence": 1.0,
        "mapping_version": ONTOLOGY.alias_registry_version.tag,
        "ontology_version": ONTOLOGY.ontology_version.tag,
        "ingested_at": INGESTED_AT,
        "source_id": REGISTRY_SOURCE_ID,
    }
    base.update(overrides)
    return ExerciseNormalizationRecord.model_validate(base)


def test_a_resolved_outcome_requires_a_confidence() -> None:
    with pytest.raises(ValueError, match="must record its mapping confidence"):
        _record(confidence=None)


def test_an_unresolved_outcome_rejects_a_confidence() -> None:
    with pytest.raises(ValueError, match="may only be present on a resolved outcome"):
        _record(
            resolution_status=ResolutionStatus.AMBIGUOUS,
            resolution_method=ResolutionMethod.CURATED_AMBIGUOUS,
            exercise_id=None,
            confidence=0.9,
            ambiguity_reason=AmbiguityReason.UNSPECIFIED_MACHINE,
            candidate_exercise_ids=(exercise_id_for("bench"),),
        )


def test_an_unresolved_outcome_rejects_naming_an_exercise() -> None:
    with pytest.raises(ValueError, match="may not carry an exercise_id"):
        _record(
            resolution_status=ResolutionStatus.UNMAPPED,
            resolution_method=ResolutionMethod.NO_MATCH,
            confidence=None,
            ambiguity_reason=AmbiguityReason.UNKNOWN_SOURCE_TAXONOMY,
        )


def test_a_resolved_outcome_rejects_candidates() -> None:
    with pytest.raises(ValueError, match="no candidates"):
        _record(candidate_exercise_ids=(exercise_id_for("bench"),))


def test_an_ambiguity_requires_at_least_one_candidate() -> None:
    with pytest.raises(ValueError, match="ambiguous requires at least one candidate"):
        _record(
            resolution_status=ResolutionStatus.AMBIGUOUS,
            resolution_method=ResolutionMethod.CURATED_AMBIGUOUS,
            exercise_id=None,
            confidence=None,
            ambiguity_reason=AmbiguityReason.UNSPECIFIED_MACHINE,
        )


def test_an_unresolved_outcome_rejects_an_alias_reference() -> None:
    with pytest.raises(ValueError, match="only meaningful for a resolved outcome"):
        _record(
            resolution_status=ResolutionStatus.UNMAPPED,
            resolution_method=ResolutionMethod.NO_MATCH,
            exercise_id=None,
            confidence=None,
            ambiguity_reason=AmbiguityReason.UNKNOWN_SOURCE_TAXONOMY,
            source_alias_id="exa_" + "0" * 32,
        )


def test_a_method_cannot_justify_a_status_it_cannot_produce() -> None:
    with pytest.raises(ValueError, match="can produce"):
        _record(resolution_method=ResolutionMethod.NO_MATCH)


def test_the_question_form_method_never_records_a_mapping() -> None:
    with pytest.raises(ValueError, match="question_form"):
        _record(resolution_method=ResolutionMethod.QUESTION_FORM)


def test_candidate_and_rule_lists_are_deduplicated_and_sorted() -> None:
    record = _record(
        resolution_status=ResolutionStatus.AMBIGUOUS,
        resolution_method=ResolutionMethod.CURATED_AMBIGUOUS,
        exercise_id=None,
        confidence=None,
        ambiguity_reason=AmbiguityReason.UNSPECIFIED_MACHINE,
        candidate_exercise_ids=tuple(
            reversed([exercise_id_for("bench"), exercise_id_for("incline_bench")])
        ),
        normalization_rules=("fold_plural", "collapse_whitespace", "fold_plural"),
    )
    assert record.candidate_exercise_ids == tuple(
        sorted([exercise_id_for("bench"), exercise_id_for("incline_bench")])
    )
    assert record.normalization_rules == ("collapse_whitespace", "fold_plural")


def test_source_system_must_be_a_namespace_token() -> None:
    with pytest.raises(ValueError, match="source_system"):
        _record(source_system="Hevy Export 2024")


def test_the_timestamp_is_normalized_to_utc() -> None:
    """A naive or non-UTC instant must not reach the persisted artifact."""
    record = _record(ingested_at=datetime(2021, 1, 4, 7, 0, tzinfo=UTC))
    offset = record.ingested_at.utcoffset()
    assert offset is not None
    assert offset.total_seconds() == 0
