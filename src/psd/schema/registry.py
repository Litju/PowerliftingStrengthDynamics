"""Canonical table registry.

The registry is the machine-readable description of the persisted schema: the
table name, its Pydantic contract, its explicit column order, its primary key,
and its **canonical ordering**. Every table declares an explicit total order
because persisted artifacts must be sorted before they are hashed or written.

``order_by`` always ends with the primary key, so the order is total and no two
rows can compare equal. DuckDB and SQL do not guarantee row order without
``ORDER BY``, and persisted artifacts must not depend on incidental row order.

Every per-athlete competition table is ordered ``athlete_id`` first, so the whole
corpus walks in longitudinal athlete order and one partition walk can serve all of
them. That is not only tidier than a competition-major order: it is what lets a
corpus build keep memory bounded, because partitioning on the leading sort column
makes the partition walk *be* the canonical order.

Category labels mirror the four semantic separations PSD requires:

``context``
    athlete identity, body measurements, equipment state, and the meets
    competitions took place in.
``semantics``
    exercise definitions, source aliases, and normalization outcomes.
``programming``
    prescription only.
``execution``
    performed work only.
``observation``
    measurements, not prescriptions and not outcomes.
``competition``
    outcomes, including reported/derived bests and totals.
``provenance``
    sources and dataset lineage.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import pyarrow as pa
from pydantic import BaseModel

from psd.provenance.manifest import LineageEntry
from psd.provenance.sources import SourceRecord
from psd.schema.arrow import schema_for_model, undeclared_fields
from psd.schema.models import (
    AthleteRecord,
    AthleteSourceLinkRecord,
    BodyMeasurementRecord,
    CompetitionAttemptRecord,
    CompetitionMeetRecord,
    CompetitionRecord,
    CompetitionReportedResultRecord,
    EquipmentStateRecord,
    ExerciseAliasRecord,
    ExerciseDefinitionRecord,
    ExerciseNormalizationRecord,
    ObservationRecord,
    PerformanceTestRecord,
    PerformedExerciseRecord,
    PerformedRepRecord,
    PerformedSessionRecord,
    PerformedSetRecord,
    PlannedExerciseRecord,
    PlannedSessionRecord,
    PlannedSetRecord,
    ProgramModificationRecord,
    ProgramRecord,
    ProgramVersionRecord,
    VelocityObservationRecord,
)

__all__ = (
    "PROVENANCE_COLUMNS",
    "TABLE_SPECS",
    "TEMPORAL_COLUMNS",
    "TableSpec",
    "arrow_schema_for",
    "column_order",
    "table_categories",
    "table_names",
    "table_spec",
    "temporal_columns",
    "validate_schema_columns",
)

PROVENANCE_COLUMNS: tuple[str, ...] = (
    "source_id",
    "source_record_key",
    "source_record_hash",
    "ingested_at",
    "quality_flags",
    "missingness_reason",
)

TEMPORAL_COLUMNS: tuple[str, ...] = (
    "created_at",
    "scheduled_at",
    "performed_at",
    "observed_at",
    "modified_at",
)

# Temporal provenance is restricted per semantic category so that
# ``prescription != execution`` is enforced by the physical schema, not only by
# convention. A prescription record cannot carry ``performed_at``: the moment
# something was performed is execution state, and storing it on a plan would
# invite exactly the leakage PSD forbids.
_CONTEXT_TEMPORAL: tuple[str, ...] = ("created_at", "observed_at", "modified_at")
_SEMANTICS_TEMPORAL: tuple[str, ...] = ("created_at", "modified_at")
_PROGRAMMING_TEMPORAL: tuple[str, ...] = ("created_at", "scheduled_at", "modified_at")
_EXECUTION_TEMPORAL: tuple[str, ...] = ("created_at", "performed_at", "modified_at")
_OBSERVATION_TEMPORAL: tuple[str, ...] = ("created_at", "observed_at", "modified_at")
_COMPETITION_TEMPORAL: tuple[str, ...] = (
    "created_at",
    "observed_at",
    "performed_at",
    "modified_at",
)

_PROVENANCE_ONLY_TEMPORAL: tuple[str, ...] = ()

#: Which temporal-provenance columns each model may persist.
_TEMPORAL_BY_MODEL: dict[str, tuple[str, ...]] = {
    AthleteRecord.__name__: _CONTEXT_TEMPORAL,
    AthleteSourceLinkRecord.__name__: _CONTEXT_TEMPORAL,
    BodyMeasurementRecord.__name__: _CONTEXT_TEMPORAL,
    EquipmentStateRecord.__name__: _CONTEXT_TEMPORAL,
    CompetitionMeetRecord.__name__: _CONTEXT_TEMPORAL,
    ExerciseDefinitionRecord.__name__: _SEMANTICS_TEMPORAL,
    ExerciseAliasRecord.__name__: _SEMANTICS_TEMPORAL,
    ExerciseNormalizationRecord.__name__: _SEMANTICS_TEMPORAL,
    ProgramRecord.__name__: _PROGRAMMING_TEMPORAL,
    ProgramVersionRecord.__name__: _PROGRAMMING_TEMPORAL,
    ProgramModificationRecord.__name__: _PROGRAMMING_TEMPORAL,
    PlannedSessionRecord.__name__: _PROGRAMMING_TEMPORAL,
    PlannedExerciseRecord.__name__: _PROGRAMMING_TEMPORAL,
    PlannedSetRecord.__name__: _PROGRAMMING_TEMPORAL,
    PerformedSessionRecord.__name__: _EXECUTION_TEMPORAL,
    PerformedExerciseRecord.__name__: _EXECUTION_TEMPORAL,
    PerformedSetRecord.__name__: _EXECUTION_TEMPORAL,
    PerformedRepRecord.__name__: _EXECUTION_TEMPORAL,
    ObservationRecord.__name__: _OBSERVATION_TEMPORAL,
    PerformanceTestRecord.__name__: _OBSERVATION_TEMPORAL,
    VelocityObservationRecord.__name__: _OBSERVATION_TEMPORAL,
    CompetitionRecord.__name__: _COMPETITION_TEMPORAL,
    CompetitionAttemptRecord.__name__: _COMPETITION_TEMPORAL,
    CompetitionReportedResultRecord.__name__: _COMPETITION_TEMPORAL,
    SourceRecord.__name__: _PROVENANCE_ONLY_TEMPORAL,
    LineageEntry.__name__: _PROVENANCE_ONLY_TEMPORAL,
}


def temporal_columns(model: type[BaseModel]) -> tuple[str, ...]:
    """Return the temporal-provenance columns *model* may persist."""
    return _TEMPORAL_BY_MODEL[model.__name__]


def _columns(model: type[BaseModel], *leading: str) -> tuple[str, ...]:
    """Build a table's persisted column order.

    Args:
        model: The Pydantic contract whose fields define the full column set.
        *leading: Columns in the order they should be persisted. Temporal and
            provenance columns follow automatically, skipping any already listed.
            Duplicate columns are therefore impossible by construction.

    Returns:
        The full, duplicate-free column order.
    """
    seen = set(leading)
    tail: list[str] = [
        name
        for name in (*temporal_columns(model), *PROVENANCE_COLUMNS)
        if name in model.model_fields and name not in seen
    ]
    return (*leading, *tail)


@dataclass(frozen=True, slots=True)
class TableSpec:
    """Persisted description of one canonical table.

    Attributes:
        name: Table name and file stem.
        model: Pydantic contract.
        columns: Explicit persisted column order.
        primary_key: Key columns, unique within the table.
        order_by: Canonical total ordering used before hashing and writing.
        category: Semantic category label.
        summary: One-line description for CLI and documentation output.
    """

    name: str
    model: type[BaseModel]
    columns: tuple[str, ...]
    primary_key: tuple[str, ...]
    order_by: tuple[str, ...]
    category: str
    summary: str

    def arrow_schema(self) -> pa.Schema:
        """Return the derived Arrow schema for this table."""
        return schema_for_model(self.model, self.columns)

    def __post_init__(self) -> None:
        if not set(self.primary_key) <= set(self.columns):
            msg = f"{self.name}: primary key columns must be declared columns."
            raise ValueError(msg)
        if not set(self.order_by) <= set(self.columns):
            msg = f"{self.name}: order_by columns must be declared columns."
            raise ValueError(msg)
        if self.order_by[-len(self.primary_key) :] != self.primary_key:
            msg = (
                f"{self.name}: order_by must end with the primary key "
                f"{self.primary_key!r} so the ordering is total."
            )
            raise ValueError(msg)
        declared = set(self.columns)
        for column in (*PROVENANCE_COLUMNS, *temporal_columns(self.model)):
            if column in self.model.model_fields and column not in declared:
                msg = (
                    f"{self.name}: column {column!r} is required by the record's provenance "
                    "contract but missing from the persisted column order."
                )
                raise ValueError(msg)
        permitted_withheld = set(TEMPORAL_COLUMNS) - set(temporal_columns(self.model))
        withheld = set(undeclared_fields(self.model, self.columns))
        unexpected = sorted(withheld - permitted_withheld)
        if unexpected:
            msg = (
                f"{self.name}: field(s) {', '.join(unexpected)} are declared on the model but "
                "neither persisted nor withholdable as temporal provenance."
            )
            raise ValueError(msg)
        # Fail loudly at import time if the model and the declaration disagree,
        # and therefore before any artifact can be written with a stale schema.
        schema_for_model(self.model, self.columns)


TABLE_SPECS: tuple[TableSpec, ...] = (
    TableSpec(
        name="source",
        model=SourceRecord,
        columns=_columns(
            SourceRecord,
            "source_id",
            "nature",
            "regime",
            "origin_system",
            "display_name",
            "dataset_version",
            "snapshot_date",
            "snapshot_sha256",
            "license_id",
            "license_url",
            "publication_basis",
            "consent_basis",
            "redistribution",
            "ingested_at",
            "notes",
        ),
        primary_key=("source_id",),
        order_by=("source_id",),
        category="provenance",
        summary="External data source with license, consent, and real/synthetic semantics.",
    ),
    TableSpec(
        name="provenance",
        model=LineageEntry,
        columns=_columns(
            LineageEntry,
            "provenance_id",
            "dataset_id",
            "parent_dataset_id",
            "transform_name",
            "transform_version",
            "code_commit",
            "config_sha256",
            "schema_version",
            "random_seed",
            "created_at",
            "description",
        ),
        primary_key=("provenance_id",),
        order_by=("dataset_id", "transform_name", "provenance_id"),
        category="provenance",
        summary="Dataset-level transformation lineage for reproducibility.",
    ),
    TableSpec(
        name="athlete",
        model=AthleteRecord,
        columns=_columns(
            AthleteRecord,
            "athlete_id",
            "pseudonym",
            "identity_status",
            "ambiguity_group_id",
            "is_synthetic",
            "synthetic_regime",
            "sex_category_raw",
            "sex_category",
            "birth_year",
            "country_code",
            "created_at",
        ),
        primary_key=("athlete_id",),
        order_by=("athlete_id",),
        category="context",
        summary="Athlete identity with synthetic flag and identity-link status.",
    ),
    TableSpec(
        name="athlete_source_link",
        model=AthleteSourceLinkRecord,
        columns=_columns(
            AthleteSourceLinkRecord,
            "athlete_id",
            "source_athlete_key",
            "link_method",
            "link_confidence",
            "is_primary",
            "created_at",
        ),
        primary_key=("athlete_id", "source_id"),
        order_by=("athlete_id", "source_id"),
        category="context",
        summary="Athlete-to-source key linkage, including unresolved ambiguity.",
    ),
    TableSpec(
        name="body_measurement",
        model=BodyMeasurementRecord,
        columns=_columns(
            BodyMeasurementRecord,
            "body_measurement_id",
            "athlete_id",
            "measured_at",
            "measurement_type",
            "event_time_precision",
            "measurement_context",
            "method",
            "raw_value",
            "raw_unit",
            "value_normalized",
            "unit_normalized",
        ),
        primary_key=("body_measurement_id",),
        order_by=("athlete_id", "measured_at", "body_measurement_id"),
        category="context",
        summary="Body mass, height, and composition with raw values preserved.",
    ),
    TableSpec(
        name="equipment_state",
        model=EquipmentStateRecord,
        columns=_columns(
            EquipmentStateRecord,
            "equipment_state_id",
            "athlete_id",
            "equipment_item",
            "effective_from",
            "effective_to",
            "identifier",
            "version_label",
            "created_at",
        ),
        primary_key=("equipment_state_id",),
        order_by=("athlete_id", "equipment_item", "effective_from", "equipment_state_id"),
        category="context",
        summary="Equipment in use over time, so equipment transitions are visible.",
    ),
    TableSpec(
        name="exercise_definition",
        model=ExerciseDefinitionRecord,
        columns=_columns(
            ExerciseDefinitionRecord,
            "exercise_id",
            "canonical_key",
            "canonical_name",
            "parent_lift",
            "specificity_level",
            "implement",
            "bar_type",
            "equipment",
            "laterality",
            "stance",
            "grip",
            "range_of_motion",
            "pause",
            "pause_rule",
            "tempo",
            "configuration",
            "equipment_note",
            "definition_note",
            "created_at",
        ),
        primary_key=("exercise_id",),
        order_by=("canonical_key", "exercise_id"),
        category="semantics",
        summary="Observable exercise semantics without transfer coefficients.",
    ),
    TableSpec(
        name="exercise_alias",
        model=ExerciseAliasRecord,
        columns=_columns(
            ExerciseAliasRecord,
            "exercise_alias_id",
            "exercise_id",
            "source_system",
            "alias_raw",
            "alias_normalized",
            "mapping_status",
            "mapping_version",
            "ontology_version",
            "note",
            "created_at",
        ),
        primary_key=("exercise_alias_id",),
        order_by=("source_system", "alias_normalized", "exercise_id", "exercise_alias_id"),
        category="semantics",
        summary="Source-specific exercise aliases mapped to canonical exercises.",
    ),
    TableSpec(
        name="exercise_normalization",
        model=ExerciseNormalizationRecord,
        columns=_columns(
            ExerciseNormalizationRecord,
            "normalization_id",
            "raw_label",
            "normalized_label",
            "normalization_rules",
            "source_system",
            "source_alias_id",
            "resolution_status",
            "resolution_method",
            "exercise_id",
            "parent_lift",
            "candidate_exercise_ids",
            "ambiguity_reason",
            "mapping_version",
            "ontology_version",
            "note",
            "created_at",
        ),
        primary_key=("normalization_id",),
        order_by=("source_system", "normalized_label", "raw_label", "normalization_id"),
        category="semantics",
        summary=(
            "Deterministic normalization outcomes for raw labels, including the "
            "ambiguous and unmapped ones. Mapping evidence is symbolic: the resolution "
            "method plus, for a lookup, the alias row it matched."
        ),
    ),
    TableSpec(
        name="program",
        model=ProgramRecord,
        columns=_columns(ProgramRecord, "program_id", "name", "description", "created_at"),
        primary_key=("program_id",),
        order_by=("program_id",),
        category="programming",
        summary="Named training programs.",
    ),
    TableSpec(
        name="program_version",
        model=ProgramVersionRecord,
        columns=_columns(
            ProgramVersionRecord,
            "program_version_id",
            "program_id",
            "version_label",
            "effective_from",
            "effective_to",
            "authored_at",
            "change_summary",
            "supersedes_program_version_id",
            "created_at",
        ),
        primary_key=("program_version_id",),
        order_by=("program_id", "effective_from", "program_version_id"),
        category="programming",
        summary="Immutable program versions effective over an interval.",
    ),
    TableSpec(
        name="program_modification",
        model=ProgramModificationRecord,
        columns=_columns(
            ProgramModificationRecord,
            "program_modification_id",
            "athlete_id",
            "planned_session_id",
            "program_version_id",
            "known_at",
            "change_kind",
            "field_changed",
            "previous_value",
            "new_value",
            "value_encoding",
            "rationale",
        ),
        primary_key=("program_modification_id",),
        order_by=("athlete_id", "known_at", "program_modification_id"),
        category="programming",
        summary="Prescription changes stamped with when they became known.",
    ),
    TableSpec(
        name="planned_session",
        model=PlannedSessionRecord,
        columns=_columns(
            PlannedSessionRecord,
            "planned_session_id",
            "athlete_id",
            "program_version_id",
            "scheduled_at",
            "session_order_index",
            "session_status",
            "not_performed_reason",
            "template_label",
        ),
        primary_key=("planned_session_id",),
        order_by=("athlete_id", "scheduled_at", "session_order_index", "planned_session_id"),
        category="programming",
        summary="Scheduled sessions, including skipped sessions and their reasons.",
    ),
    TableSpec(
        name="planned_exercise",
        model=PlannedExerciseRecord,
        columns=_columns(
            PlannedExerciseRecord,
            "planned_exercise_id",
            "planned_session_id",
            "exercise_id",
            "ordinal",
            "supersedes_planned_exercise_id",
        ),
        primary_key=("planned_exercise_id",),
        order_by=("planned_session_id", "ordinal", "planned_exercise_id"),
        category="programming",
        summary="Prescribed exercises within planned sessions.",
    ),
    TableSpec(
        name="planned_set",
        model=PlannedSetRecord,
        columns=_columns(
            PlannedSetRecord,
            "planned_set_id",
            "planned_exercise_id",
            "ordinal",
            "target_reps",
            "target_reps_min",
            "target_reps_max",
            "target_load_raw",
            "target_load_unit",
            "target_load_kg",
            "target_rpe",
            "target_rir",
            "target_percent_one_rm",
            "target_duration_seconds",
            "rest_target_seconds",
            "prescription_basis",
        ),
        primary_key=("planned_set_id",),
        order_by=("planned_exercise_id", "ordinal", "planned_set_id"),
        category="programming",
        summary="Prescribed set targets: prescription only, never execution.",
    ),
    TableSpec(
        name="performed_session",
        model=PerformedSessionRecord,
        columns=_columns(
            PerformedSessionRecord,
            "performed_session_id",
            "athlete_id",
            "planned_session_id",
            "started_at",
            "session_order_index",
            "session_type",
            "ended_at",
            "duration_seconds",
            "is_completed",
        ),
        primary_key=("performed_session_id",),
        order_by=("athlete_id", "started_at", "session_order_index", "performed_session_id"),
        category="execution",
        summary="Sessions actually performed, optionally linked to a plan.",
    ),
    TableSpec(
        name="performed_exercise",
        model=PerformedExerciseRecord,
        columns=_columns(
            PerformedExerciseRecord,
            "performed_exercise_id",
            "performed_session_id",
            "exercise_id",
            "ordinal",
            "planned_exercise_id",
            "is_substitution",
            "substituted_from_exercise_id",
        ),
        primary_key=("performed_exercise_id",),
        order_by=("performed_session_id", "ordinal", "performed_exercise_id"),
        category="execution",
        summary="Exercises actually performed, including substitutions.",
    ),
    TableSpec(
        name="performed_set",
        model=PerformedSetRecord,
        columns=_columns(
            PerformedSetRecord,
            "performed_set_id",
            "performed_exercise_id",
            "ordinal",
            "planned_set_id",
            "set_status",
            "load_raw",
            "load_unit",
            "load_kg",
            "reps_performed",
            "reps_failed",
            "is_failure",
            "rpe",
            "rir",
            "tempo_actual",
            "duration_seconds",
            "rest_actual_seconds",
            "is_warmup",
            "rep_level_data_available",
            "incomplete_reason",
        ),
        primary_key=("performed_set_id",),
        order_by=("performed_exercise_id", "ordinal", "performed_set_id"),
        category="execution",
        summary="Sets actually performed, including failures and incomplete sets.",
    ),
    TableSpec(
        name="performed_rep",
        model=PerformedRepRecord,
        columns=_columns(
            PerformedRepRecord,
            "performed_rep_id",
            "performed_set_id",
            "rep_ordinal",
            "rep_status",
            "is_success",
            "load_raw",
            "load_unit",
            "load_kg",
            "rpe",
            "velocity_mps",
        ),
        primary_key=("performed_rep_id",),
        order_by=("performed_set_id", "rep_ordinal", "performed_rep_id"),
        category="execution",
        summary="Per-repetition records, present only when rep-level data exist.",
    ),
    TableSpec(
        name="observation",
        model=ObservationRecord,
        columns=_columns(
            ObservationRecord,
            "observation_id",
            "athlete_id",
            "observed_at",
            "observation_type",
            "observation_method",
            "observation_scope",
            "reporter_role",
            "instrument",
            "numeric_value",
            "text_value",
            "unit",
            "parent_session_id",
            "parent_exercise_id",
            "parent_set_id",
            "parent_rep_id",
            "parent_competition_id",
        ),
        primary_key=("observation_id",),
        order_by=("athlete_id", "observed_at", "observation_id"),
        category="observation",
        summary="Reported measurements scoped to their level of observation.",
    ),
    TableSpec(
        name="performance_test",
        model=PerformanceTestRecord,
        columns=_columns(
            PerformanceTestRecord,
            "performance_test_id",
            "athlete_id",
            "observed_at",
            "test_type",
            "exercise_id",
            "performed_set_id",
            "competition_id",
            "protocol_label",
            "result_metric",
            "result_raw",
            "result_unit",
            "result_normalized",
            "result_unit_normalized",
        ),
        primary_key=("performance_test_id",),
        order_by=("athlete_id", "observed_at", "performance_test_id"),
        category="observation",
        summary="Structured performance tests, kept distinct from meet results.",
    ),
    TableSpec(
        name="velocity_observation",
        model=VelocityObservationRecord,
        columns=_columns(
            VelocityObservationRecord,
            "velocity_observation_id",
            "athlete_id",
            "observed_at",
            "performed_set_id",
            "performed_rep_id",
            "method",
            "mean_velocity_mps",
            "peak_velocity_mps",
            "velocity_loss_percent",
            "load_raw",
            "load_unit",
            "load_kg",
            "sampling_hz",
        ),
        primary_key=("velocity_observation_id",),
        order_by=("athlete_id", "observed_at", "velocity_observation_id"),
        category="observation",
        summary="Bar-velocity measurements with instrument context.",
    ),
    TableSpec(
        name="competition_meet",
        model=CompetitionMeetRecord,
        columns=_columns(
            CompetitionMeetRecord,
            "competition_meet_id",
            "meet_name",
            "meet_date",
            "event_time_precision",
            "meet_federation",
            "meet_parent_federation",
            "meet_country",
            "meet_state",
            "meet_town",
            "sanctioned_status_raw",
            "is_sanctioned",
            "created_at",
        ),
        primary_key=("competition_meet_id",),
        order_by=("meet_date", "meet_federation", "meet_name", "competition_meet_id"),
        category="context",
        summary=(
            "A competition as a meet: start date, hosting and sanctioning bodies, "
            "location, and sanctioned status. Distinct from the per-athlete outcome."
        ),
    ),
    TableSpec(
        name="competition",
        model=CompetitionRecord,
        columns=_columns(
            CompetitionRecord,
            "competition_id",
            "athlete_id",
            "competition_meet_id",
            "competition_date",
            "event_time_precision",
            "competition_event",
            "name",
            "federation",
            "sanctioning_body",
            "location",
            "equipment_class_raw",
            "equipment_class",
            "weight_class_raw",
            "bodyweight_raw",
            "bodyweight_unit",
            "bodyweight_kg",
            "participation_status",
            "participation_place",
            "participation_status_kind",
            "is_drug_tested_category",
            "age_reported",
            "age_precision",
            "age_class_raw",
            "birth_year_class_raw",
            "division_raw",
            "athlete_country_raw",
            "athlete_region_raw",
            "is_championship",
        ),
        primary_key=("competition_id",),
        order_by=("athlete_id", "competition_date", "competition_id"),
        category="competition",
        summary=(
            "One athlete's result at one meet, with equipment category, weight class, "
            "body mass, age and its precision, participation status, and drug-tested "
            "category coverage."
        ),
    ),
    TableSpec(
        name="competition_attempt",
        model=CompetitionAttemptRecord,
        columns=_columns(
            CompetitionAttemptRecord,
            "competition_attempt_id",
            "competition_id",
            "athlete_id",
            "lift",
            "attempt_number",
            "attempt_role",
            "attempt_order_basis",
            "attempt_time",
            "source_attempt_raw",
            "load_raw",
            "load_unit",
            "load_kg",
            "result",
            "is_opener",
        ),
        primary_key=("competition_attempt_id",),
        order_by=(
            "athlete_id",
            "competition_id",
            "lift",
            "attempt_number",
            "competition_attempt_id",
        ),
        category="competition",
        summary=(
            "Individual competition attempts; missing attempts stay missing. The "
            "signed source value is preserved beside the positive attempted load, and "
            "a record fourth attempt is distinguishable from the three counted ones."
        ),
    ),
    TableSpec(
        name="competition_reported_result",
        model=CompetitionReportedResultRecord,
        columns=_columns(
            CompetitionReportedResultRecord,
            "competition_reported_result_id",
            "competition_id",
            "athlete_id",
            "result_kind",
            "value",
            "unit",
            "source_value_raw",
            "result_source_field",
            "reported_best_semantics",
            "is_derived",
            "derivation_note",
        ),
        primary_key=("competition_reported_result_id",),
        order_by=(
            "athlete_id",
            "competition_id",
            "result_kind",
            "competition_reported_result_id",
        ),
        category="competition",
        summary=(
            "Reported or derived bests, totals, and scoring-system points, flagged as "
            "derived and carrying the source field and signed value they came from."
        ),
    ),
)

_SPECS_BY_NAME: dict[str, TableSpec] = {spec.name: spec for spec in TABLE_SPECS}


def table_names() -> tuple[str, ...]:
    """Return every canonical table name in registry order."""
    return tuple(spec.name for spec in TABLE_SPECS)


def table_categories() -> dict[str, tuple[str, ...]]:
    """Return table names grouped by semantic category."""
    grouped: dict[str, list[str]] = {}
    for spec in TABLE_SPECS:
        grouped.setdefault(spec.category, []).append(spec.name)
    return {category: tuple(names) for category, names in grouped.items()}


def table_spec(name: str) -> TableSpec:
    """Return the spec for *name*.

    Raises:
        KeyError: No such canonical table.
    """
    spec = _SPECS_BY_NAME.get(name)
    if spec is None:
        msg = f"Unknown canonical table {name!r}. Known tables: {', '.join(table_names())}."
        raise KeyError(msg)
    return spec


def arrow_schema_for(name: str) -> pa.Schema:
    """Return the derived Arrow schema for a canonical table."""
    return table_spec(name).arrow_schema()


def column_order(name: str) -> tuple[str, ...]:
    """Return the explicit persisted column order for a canonical table."""
    return table_spec(name).columns


def validate_schema_columns(names: Sequence[str] | None = None) -> None:
    """Verify that every declared table's columns match its model.

    Args:
        names: Restrict the check to these tables; all tables by default.

    Raises:
        ArrowTypeMappingError: A declaration and its model disagree.
    """
    selected = table_names() if names is None else tuple(names)
    for name in selected:
        table_spec(name).arrow_schema()
