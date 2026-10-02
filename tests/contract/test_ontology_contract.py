"""Structural contract tests for the exercise ontology.

These assert properties of the *schema and vocabulary* rather than of one label: that
the ontology's tables are part of the canonical registry, that every descriptor
column is validated, that the ontology cannot encode an effectiveness claim, and that
the ambiguity policy is represented in the persisted schema rather than only in code.

A structural test here is worth more than an example test, because it keeps holding as
the vocabulary grows.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, get_args

import pyarrow as pa
import pytest
from pydantic import BaseModel
from pydantic.fields import FieldInfo

from psd.ontology import default_ontology
from psd.ontology.artifact import ontology_source_record
from psd.provenance.environment import EnvironmentSnapshot
from psd.provenance.manifest import DatasetKind, DatasetManifest
from psd.provenance.sources import DataRegime, SourceNature
from psd.schema.models import (
    STATUS_BY_RESOLUTION_METHOD,
    ExerciseAliasRecord,
    ExerciseDefinitionRecord,
    ExerciseNormalizationRecord,
)
from psd.schema.registry import PROVENANCE_COLUMNS, table_categories, table_names, table_spec
from psd.schema.vocabulary import VOCABULARIES, ResolutionMethod, ResolutionStatus
from psd.validation.declarative import COLUMN_RANGES, VOCABULARY_BY_COLUMN
from psd.versions import SCHEMA_VERSION

ONTOLOGY = default_ontology()

ONTOLOGY_TABLES: tuple[str, ...] = (
    "exercise_definition",
    "exercise_alias",
    "exercise_normalization",
)

#: Column-name fragments that would mean the ontology was prescribing training.
PRESCRIPTIVE_FRAGMENTS: tuple[str, ...] = (
    "transfer",
    "coefficient",
    "effectiveness",
    "specificity_score",
    "similarity",
    "recommend",
    "prescri",
    "target",
    "value",
)


# ---------------------------------------------------------------------------
# the tables exist and are wired into the canonical machinery
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("table", ONTOLOGY_TABLES)
def test_ontology_tables_are_registered(table: str) -> None:
    assert table in table_names()
    assert table_spec(table).category == "semantics"


def test_normalization_is_the_third_semantics_table() -> None:
    """The audit table is a first-class canonical table, not a side channel."""
    assert set(table_categories()["semantics"]) == set(ONTOLOGY_TABLES)


@pytest.mark.parametrize("table", ONTOLOGY_TABLES)
def test_every_enum_backed_column_is_validated_by_a_vocabulary_rule(table: str) -> None:
    """Every controlled-vocabulary column must be checked by the declarative layer.

    This is what stops a descriptor from drifting back to free text: adding an enum
    field without a rule would leave it outside this set and fail here.
    """
    checked = {
        column for (declared_table, column) in VOCABULARY_BY_COLUMN if declared_table == table
    }
    fields: dict[str, FieldInfo] = dict(table_spec(table).model.model_fields)
    enum_backed = {name for name, info in fields.items() if _is_enum_annotation(info.annotation)}
    # ``quality_flags`` is a RES-235 provenance column validated by its own contract
    # rather than by a column rule, so it is not this layer's business.
    assert enum_backed - set(PROVENANCE_COLUMNS) <= checked, sorted(enum_backed - checked)


def _is_enum_annotation(annotation: object) -> bool:
    """Whether an annotation is a controlled vocabulary or a tuple of one."""
    if isinstance(annotation, type) and issubclass(annotation, StrEnum):
        return True
    args = get_args(annotation)
    return bool(args) and all(
        isinstance(arg, type) and issubclass(arg, StrEnum) for arg in args if arg is not Ellipsis
    )


@pytest.mark.parametrize("table", ONTOLOGY_TABLES)
def test_every_vocabulary_rule_targets_an_existing_column(table: str) -> None:
    """A rule pointing at a renamed column would validate nothing."""
    columns = set(table_spec(table).columns)
    for declared_table, column in VOCABULARY_BY_COLUMN:
        if declared_table != table:
            continue
        assert column in columns, f"{table}.{column}"
        assert VOCABULARY_BY_COLUMN[(declared_table, column)] in VOCABULARIES


def test_no_numeric_column_remains_to_score_a_mapping() -> None:
    """The ontology holds no float at all, so there is no back door for a score.

    ``exercise_normalization`` once carried a fixed ``confidence`` per resolution
    method. It was never calibrated against held-out labels, so a reader could take
    ``0.8`` for a probability, and the field has been removed rather than replaced
    with different numbers. With no floating column left, an effectiveness value would
    have to be added as a new column and would fail this test first.
    """
    floats: set[str] = {
        column
        for table in ONTOLOGY_TABLES
        for column, data_type in zip(
            table_spec(table).arrow_schema().names,
            table_spec(table).arrow_schema().types,
            strict=True,
        )
        if pa.types.is_floating(data_type)
    }
    assert floats == set()
    assert "confidence" not in COLUMN_RANGES


def test_mapping_evidence_is_symbolic_and_complete() -> None:
    """Method, alias row, candidates, and reason: the whole evidence record.

    Every resolved outcome names the stage that produced it, and the two lookup
    methods cite the alias row they read. Nothing else is needed to audit a mapping,
    which is why no second score was introduced in place of the removed float.
    """
    lookup_methods = {
        ResolutionMethod.REGISTERED_ALIAS,
        ResolutionMethod.CROSS_SOURCE_ALIAS,
    }
    assert lookup_methods <= set(ResolutionMethod)
    for label, expected in (
        ("CGBP", ResolutionMethod.REGISTERED_ALIAS),
        ("SSB Squat", ResolutionMethod.CROSS_SOURCE_ALIAS),
    ):
        outcome = ONTOLOGY.resolve(label, source_system="hevy")
        assert outcome.resolution_method is expected
        assert outcome.source_alias_id is not None
    columns = set(table_spec("exercise_normalization").columns)
    assert {
        "resolution_method",
        "resolution_status",
        "source_alias_id",
        "candidate_exercise_ids",
        "ambiguity_reason",
        "mapping_version",
        "ontology_version",
    } <= columns


@pytest.mark.parametrize("table", ONTOLOGY_TABLES)
def test_ontology_tables_carry_row_level_provenance(table: str) -> None:
    assert set(PROVENANCE_COLUMNS) <= set(table_spec(table).columns)


# ---------------------------------------------------------------------------
# the ontology cannot prescribe training
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("table", ONTOLOGY_TABLES)
def test_no_ontology_column_prescribes_a_training_value(table: str) -> None:
    for column in table_spec(table).columns:
        for fragment in PRESCRIPTIVE_FRAGMENTS:
            assert fragment not in column, f"{table}.{column}"


@pytest.mark.parametrize(
    "model", [ExerciseDefinitionRecord, ExerciseAliasRecord, ExerciseNormalizationRecord]
)
def test_no_ontology_field_prescribes_a_training_value(model: type[BaseModel]) -> None:
    for field in model.model_fields:
        for fragment in PRESCRIPTIVE_FRAGMENTS:
            assert fragment not in field, f"{model.__name__}.{field}"


def test_the_ontology_ships_no_numeric_effectiveness_table() -> None:
    """A table whose name implies a value would be the back door this guards."""
    for name in table_names():
        for fragment in ("transfer", "coefficient", "effectiveness", "similarity", "value"):
            assert fragment not in name, name


def test_no_resolution_method_may_produce_a_mapping_it_did_not_justify() -> None:
    """The contract table that ties each method to the statuses it may produce."""
    assert set(STATUS_BY_RESOLUTION_METHOD) == set(ResolutionMethod)
    for method, statuses in STATUS_BY_RESOLUTION_METHOD.items():
        assert statuses, method
        if ResolutionStatus.EXACT_CANONICAL in statuses:
            assert statuses == frozenset({ResolutionStatus.EXACT_CANONICAL}), method
    for method, statuses in STATUS_BY_RESOLUTION_METHOD.items():
        if method in {
            ResolutionMethod.CURATED_AMBIGUOUS,
            ResolutionMethod.FAMILY_KEYWORD,
            ResolutionMethod.GENERIC_QUALIFIER,
            ResolutionMethod.QUESTION_FORM,
        }:
            assert ResolutionStatus.RESOLVED_ALIAS not in statuses, method
            assert ResolutionStatus.EXACT_CANONICAL not in statuses, method


# ---------------------------------------------------------------------------
# ambiguity is representable in the persisted schema
# ---------------------------------------------------------------------------


def test_the_resolution_ladder_is_fully_representable_in_the_schema() -> None:
    for status in ResolutionStatus:
        assert status.value in VOCABULARIES["resolution_status"]


def test_every_ambiguity_reason_is_persistable() -> None:
    reasons = set(VOCABULARIES["ambiguity_reason"])
    for row in ONTOLOGY.curated_ambiguities:
        assert row.reason.value in reasons


def test_candidate_lists_are_persisted_as_a_list_column() -> None:
    """Ambiguity has to survive into the artifact, not just into a return value."""
    schema = table_spec("exercise_normalization").arrow_schema()
    # Read the column type by index rather than through ``Schema.field``, whose
    # ``Field`` type is unparameterised in the community stubs.
    index = schema.names.index("candidate_exercise_ids")
    data_type = schema.types[index]
    assert pa.types.is_list(data_type)
    assert pa.types.is_string(data_type.value_type)


def test_canonical_key_is_persisted_and_unique_by_construction() -> None:
    assert "canonical_key" in table_spec("exercise_definition").columns
    keys = [spec.key for spec in ONTOLOGY.specs]
    assert len(keys) == len(set(keys))


def test_alias_rows_carry_namespace_and_both_versions() -> None:
    columns = set(table_spec("exercise_alias").columns)
    assert {"source_system", "mapping_status", "mapping_version", "ontology_version"} <= columns
    normalization_columns = set(table_spec("exercise_normalization").columns)
    assert {"source_system", "mapping_version", "ontology_version", "raw_label"} <= (
        normalization_columns
    )


def test_the_alias_namespace_is_free_text_so_new_adapters_need_no_schema_change() -> None:
    """A future source must be addable without bumping the canonical schema.

    ``source_system`` is therefore a pattern-constrained string rather than a closed
    vocabulary: a closed one would mean every new logging app is a breaking schema
    change, which is exactly the coupling the issue asks the alias layer to avoid.
    """

    def _alias(source_system: str) -> dict[str, Any]:
        return {
            "exercise_alias_id": "exa_" + "0" * 32,
            "exercise_id": "exd_" + "0" * 32,
            "source_system": source_system,
            "alias_raw": "Bench Press",
            "alias_normalized": "benchpress",
            "mapping_version": ONTOLOGY.alias_registry_version.tag,
            "ontology_version": ONTOLOGY.ontology_version.tag,
            "ingested_at": datetime(2021, 1, 4, tzinfo=UTC),
            "source_id": "src_ontology",
        }

    assert ExerciseAliasRecord.model_validate(_alias("some_future_app")).source_system == (
        "some_future_app"
    )
    with pytest.raises(ValueError, match="source_system"):
        ExerciseAliasRecord.model_validate(_alias("Hevy Export 2024"))


# ---------------------------------------------------------------------------
# the ambiguity policy is data, not just code
# ---------------------------------------------------------------------------


def test_the_ontology_publishes_labels_it_refuses_to_resolve() -> None:
    """A published refusal count is what keeps the policy auditable."""
    assert ONTOLOGY.curated_ambiguities
    statuses = {
        ONTOLOGY.resolve(spec.raw_label).resolution_status for spec in ONTOLOGY.curated_ambiguities
    }
    assert ResolutionStatus.RESOLVED_ALIAS not in statuses
    assert ResolutionStatus.EXACT_CANONICAL not in statuses


def test_every_refusal_reaches_a_rung_below_a_mapping() -> None:
    refused = {
        ResolutionStatus.PARTIAL_FAMILY,
        ResolutionStatus.AMBIGUOUS,
        ResolutionStatus.UNMAPPED,
    }
    for spec in ONTOLOGY.curated_ambiguities:
        assert ONTOLOGY.resolve(spec.raw_label).resolution_status in refused


def test_the_reference_dataset_kind_exists_for_the_ontology() -> None:
    assert DatasetKind.REFERENCE.value == "reference"


def test_a_reference_dataset_may_not_cite_sources() -> None:
    """The ontology was authored, so it must not be able to claim an external source."""
    environment = EnvironmentSnapshot(
        python_version="3.12.0",
        python_implementation="cpython",
        platform="test",
        package_version="0.1.0",
        lockfile_sha256=None,
        code_commit=None,
        is_dirty_tree=None,
    )
    with pytest.raises(ValueError, match="reference dataset cites no sources"):
        DatasetManifest(
            dataset_id="x",
            dataset_name="x",
            dataset_kind=DatasetKind.REFERENCE,
            schema_version=SCHEMA_VERSION.tag,
            manifest_version="psd-manifest/0.1.0",
            created_at=datetime(2021, 1, 1, tzinfo=UTC),
            environment=environment,
            sources=(ontology_source_record(ONTOLOGY.ontology_version.tag),),
        )


# ---------------------------------------------------------------------------
# authored vocabulary is reference data, not simulator output
# ---------------------------------------------------------------------------


def test_the_ontology_source_record_is_classified_as_reference() -> None:
    """The vocabulary was authored, so it is neither real nor generated athlete data.

    Calling it ``synthetic``/``psd_sim`` would assert that a simulator produced it,
    and would redefine ``synthetic`` as "not real athlete data" -- which is exactly
    the distinction that has to stay sharp.
    """
    record = ontology_source_record(ONTOLOGY.ontology_version.tag)
    assert record.nature is SourceNature.REFERENCE
    assert record.regime is DataRegime.REFERENCE
    assert record.nature is not SourceNature.SYNTHETIC
    assert record.regime is not DataRegime.SIM


def test_reference_and_sim_are_separate_classifications_in_the_vocabularies() -> None:
    """Both are first-class members, and neither is a synonym for the other."""
    assert "reference" in VOCABULARIES["source_nature"]
    assert "synthetic" in VOCABULARIES["source_nature"]
    assert "psd_reference" in VOCABULARIES["data_regime"]
    assert "psd_sim" in VOCABULARIES["data_regime"]
    assert SourceNature.REFERENCE.value != SourceNature.SYNTHETIC.value
    assert DataRegime.REFERENCE.value != DataRegime.SIM.value


def test_an_ingestion_dataset_may_not_cite_the_reference_registry() -> None:
    """Registry rows are re-stamped on ingest, so the vocabulary is not a data source."""
    environment = EnvironmentSnapshot(
        python_version="3.12.0",
        python_implementation="cpython",
        platform="test",
        package_version="0.1.0",
        lockfile_sha256=None,
        code_commit=None,
        is_dirty_tree=None,
    )
    with pytest.raises(ValueError, match="cannot cite reference sources"):
        DatasetManifest(
            dataset_id="x",
            dataset_name="x",
            dataset_kind=DatasetKind.TRAINING_HISTORY,
            schema_version=SCHEMA_VERSION.tag,
            manifest_version="psd-manifest/0.1.0",
            created_at=datetime(2021, 1, 1, tzinfo=UTC),
            environment=environment,
            sources=(ontology_source_record(ONTOLOGY.ontology_version.tag),),
        )
