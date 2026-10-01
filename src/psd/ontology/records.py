"""Turning an ontology into canonical records.

The ontology is reference data; the canonical schema is what PSD persists. This
module is the one-way bridge between them, and it exists so the ontology can never
drift from the artifacts it produces: every field written here comes either from a
descriptor the ontology declared or from provenance the caller supplied.

Provenance policy
-----------------

Registry rows are authored by PSD rather than ingested from an athlete's log, so
they are stamped with a dedicated registry source id by default. Callers that want
the rows to carry their own provenance -- the canonical fixtures, or a future
adapter -- pass ``source_id`` and get exactly that.

Alias rows for a *provisional* namespace carry
:class:`~psd.schema.vocabulary.QualityFlag.UNVERIFIED`, because those spellings are
representative of a logging vocabulary rather than confirmed vendor exports. The
flag is data, not a comment: it survives into the Parquet artifact where a
downstream consumer can act on it.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import datetime
from typing import Final

from psd.ontology.catalog import PROBE_LABELS, PROVISIONAL_SOURCE_SYSTEMS
from psd.ontology.registry import ExerciseOntology, NormalizationOutcome, exercise_id_for
from psd.schema.identifiers import IdPrefix, make_id
from psd.schema.models import (
    ExerciseAliasRecord,
    ExerciseDefinitionRecord,
    ExerciseNormalizationRecord,
    pause_flag_for_rule,
)
from psd.schema.vocabulary import QualityFlag, ResolutionStatus

__all__ = (
    "REGISTRY_SOURCE_ID",
    "alias_records",
    "definition_record_for",
    "definition_records",
    "normalization_id_for",
    "normalization_records",
    "probe_outcomes",
    "resolution_for",
)

#: Source id used for rows the ontology authors rather than ingests.
REGISTRY_SOURCE_ID: Final[str] = "src_psd_exercise_ontology"

_KEY_PREFIX: Final[str] = "ontology:"


def definition_record_for(
    ontology: ExerciseOntology,
    key: str,
    *,
    source_id: str = REGISTRY_SOURCE_ID,
    ingested_at: datetime,
    quality_flags: tuple[QualityFlag, ...] = (),
) -> ExerciseDefinitionRecord:
    """Return the canonical record for one canonical exercise.

    Args:
        ontology: Ontology declaring the exercise.
        key: Canonical exercise key.
        source_id: Source the row is attributed to.
        ingested_at: Timezone-aware UTC ingestion instant.
        quality_flags: Trust markers to persist.

    Returns:
        A validated :class:`~psd.schema.models.ExerciseDefinitionRecord`.

    Raises:
        KeyError: The ontology does not declare *key*.
    """
    spec = ontology.spec(key)
    return ExerciseDefinitionRecord(
        exercise_id=ontology.exercise_id(key),
        canonical_key=spec.key,
        canonical_name=spec.canonical_name,
        parent_lift=spec.parent_lift,
        specificity_level=spec.specificity_level,
        implement=spec.implement,
        bar_type=spec.bar_type,
        equipment=spec.equipment,
        laterality=spec.laterality,
        stance=spec.stance,
        grip=spec.grip,
        range_of_motion=spec.range_of_motion,
        pause=pause_flag_for_rule(spec.pause_rule),
        pause_rule=spec.pause_rule,
        tempo=spec.tempo,
        configuration=spec.configuration,
        equipment_note=None,
        definition_note=spec.note,
        source_id=source_id,
        source_record_key=f"{_KEY_PREFIX}exercise:{key}",
        source_record_hash=None,
        ingested_at=ingested_at,
        quality_flags=quality_flags,
        missingness_reason=None,
    )


def definition_records(
    ontology: ExerciseOntology,
    *,
    ingested_at: datetime,
    keys: Sequence[str] | None = None,
    source_id: str = REGISTRY_SOURCE_ID,
    quality_flags: tuple[QualityFlag, ...] = (),
) -> tuple[ExerciseDefinitionRecord, ...]:
    """Return canonical records for the whole ontology or a subset of keys.

    Args:
        ontology: Source ontology.
        ingested_at: Timezone-aware UTC ingestion instant.
        keys: Restrict to these canonical keys; every exercise when omitted.
        source_id: Source the rows are attributed to.
        quality_flags: Trust markers to persist.

    Returns:
        Records sorted by canonical key, so the result does not depend on the order
        *keys* was supplied in.
    """
    selected = ontology.specs if keys is None else tuple(ontology.spec(key) for key in keys)
    return tuple(
        definition_record_for(
            ontology,
            spec.key,
            source_id=source_id,
            ingested_at=ingested_at,
            quality_flags=quality_flags,
        )
        for spec in sorted(selected, key=lambda item: item.key)
    )


def alias_records(
    ontology: ExerciseOntology,
    *,
    ingested_at: datetime,
    source_systems: Sequence[str] | None = None,
    exercise_keys: Sequence[str] | None = None,
    source_id: str = REGISTRY_SOURCE_ID,
) -> tuple[ExerciseAliasRecord, ...]:
    """Return canonical alias records, optionally filtered.

    Args:
        ontology: Source ontology.
        ingested_at: Timezone-aware UTC ingestion instant.
        source_systems: Restrict to these namespaces; all namespaces when omitted.
        exercise_keys: Restrict to aliases of these canonical exercises.
        source_id: Source the rows are attributed to.

    Returns:
        Records sorted by ``(source_system, alias_normalized, raw_label)``.
    """
    wanted_systems = None if source_systems is None else set(source_systems)
    wanted_keys = None if exercise_keys is None else set(exercise_keys)
    selected = [
        binding
        for binding in ontology.aliases
        if (wanted_systems is None or binding.source_system in wanted_systems)
        and (wanted_keys is None or binding.exercise_key in wanted_keys)
    ]
    return tuple(
        ExerciseAliasRecord(
            exercise_alias_id=binding.alias_id,
            exercise_id=exercise_id_for(binding.exercise_key),
            source_system=binding.source_system,
            alias_raw=binding.raw_label,
            alias_normalized=binding.normalized_label,
            mapping_status=ResolutionStatus.RESOLVED_ALIAS,
            mapping_version=ontology.alias_registry_version.tag,
            ontology_version=ontology.ontology_version.tag,
            note=binding.note,
            source_id=source_id,
            source_record_key=(f"{_KEY_PREFIX}alias:{binding.source_system}:{binding.raw_label}"),
            source_record_hash=None,
            ingested_at=ingested_at,
            quality_flags=_alias_quality_flags(binding.source_system),
            missingness_reason=None,
        )
        for binding in selected
    )


def normalization_id_for(source_system: str, raw_label: str, normalized_label: str) -> str:
    """Return the deterministic identifier for one normalization outcome.

    Keyed on the *verbatim* label, so every distinct spelling a source used keeps
    its own audit row even when several of them normalize to one lookup key.
    """
    return make_id(IdPrefix.EXERCISE_NORMALIZATION, source_system, raw_label, normalized_label)


def resolution_for(
    outcome: NormalizationOutcome,
    *,
    ontology: ExerciseOntology,
    source_id: str = REGISTRY_SOURCE_ID,
    ingested_at: datetime,
) -> ExerciseNormalizationRecord:
    """Return the canonical record for one normalization outcome.

    Args:
        outcome: Result of :meth:`ExerciseOntology.resolve`.
        ontology: Ontology the outcome was produced against, for its versions.
        source_id: Source the row is attributed to.
        ingested_at: Timezone-aware UTC ingestion instant.

    Returns:
        A validated :class:`~psd.schema.models.ExerciseNormalizationRecord`.

    Raises:
        ValueError: The outcome cannot be persisted without overstating itself. The
            contract rejects that, and this function does not paper over it.
    """
    return ExerciseNormalizationRecord(
        normalization_id=normalization_id_for(
            outcome.source_system, outcome.raw_label, outcome.normalized_label
        ),
        # A blank or whitespace-only source label normalizes to nothing, and the
        # column is non-empty, so the verbatim string is kept as the normalized
        # form too. The row still says ``unmapped``, which is the point.
        raw_label=outcome.raw_label,
        normalized_label=outcome.normalized_label or outcome.raw_label,
        normalization_rules=outcome.normalization_rules,
        source_system=outcome.source_system,
        source_alias_id=outcome.source_alias_id,
        resolution_status=outcome.resolution_status,
        resolution_method=outcome.resolution_method,
        exercise_id=outcome.exercise_id,
        parent_lift=outcome.parent_lift,
        candidate_exercise_ids=tuple(
            sorted(exercise_id_for(key) for key in outcome.candidate_keys)
        ),
        confidence=outcome.confidence,
        ambiguity_reason=outcome.ambiguity_reason,
        mapping_version=ontology.alias_registry_version.tag,
        ontology_version=ontology.ontology_version.tag,
        note=None,
        source_id=source_id,
        source_record_key=f"{_KEY_PREFIX}label:{outcome.source_system}",
        source_record_hash=None,
        ingested_at=ingested_at,
        quality_flags=_alias_quality_flags(outcome.source_system),
        missingness_reason=None,
    )


def probe_outcomes(
    ontology: ExerciseOntology,
    *,
    source_system: str,
    labels: Iterable[str] = PROBE_LABELS,
) -> tuple[NormalizationOutcome, ...]:
    """Resolve the labels used to demonstrate the ambiguity policy.

    These go through the public pipeline rather than being hand-written, so the
    persisted outcomes cannot drift away from what the resolver actually does.
    """
    return tuple(
        ontology.resolve(label, source_system=source_system) for label in sorted(set(labels))
    )


def normalization_records(
    outcomes: Sequence[NormalizationOutcome],
    *,
    ontology: ExerciseOntology,
    source_id: str = REGISTRY_SOURCE_ID,
    ingested_at: datetime,
) -> tuple[ExerciseNormalizationRecord, ...]:
    """Return canonical records for normalization outcomes, in a canonical order."""
    records = (
        resolution_for(outcome, ontology=ontology, source_id=source_id, ingested_at=ingested_at)
        for outcome in outcomes
    )
    return tuple(
        sorted(
            records,
            key=lambda item: (item.source_system, item.normalized_label, item.raw_label),
        )
    )


def _alias_quality_flags(source_system: str) -> tuple[QualityFlag, ...]:
    """Mark spellings from a namespace PSD has not verified against a real export."""
    return (QualityFlag.UNVERIFIED,) if source_system in PROVISIONAL_SOURCE_SYSTEMS else ()
