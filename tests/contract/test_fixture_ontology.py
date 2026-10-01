"""The fixtures must consume the shared ontology, not a parallel vocabulary.

The failure this guards against is subtle and easy to reintroduce: a fixture that
declares its own exercises stays internally consistent and passes every validation
test, while silently disagreeing with the ontology a real ingestion would use. These
tests therefore compare the fixture rows against the registry rather than merely
checking they validate.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime

import pytest
from pydantic import BaseModel

from psd.ontology import default_ontology
from psd.ontology.records import definition_record_for
from psd.schema.registry import table_names
from tests.fixtures import (
    FIXTURE_EXERCISE_KEYS,
    adversarial_history_records,
    counts,
    messy_timeline_records,
    normal_history_records,
    synthetic_records,
)
from tests.fixtures.builders import (
    HistoryBuilder,
    SourceIds,
    add_exercises,
    add_source_records,
)

ONTOLOGY = default_ontology()
INGESTED_AT = datetime(2021, 1, 4, 7, 0, tzinfo=UTC)

FIXTURES: tuple[Callable[[], dict[str, list[BaseModel]]], ...] = (
    normal_history_records,
    adversarial_history_records,
    synthetic_records,
    messy_timeline_records,
)


def _builder() -> HistoryBuilder:
    ids = SourceIds()
    builder = HistoryBuilder(athlete_id="ath_fixture", source_id=ids.training)
    add_source_records(builder, ids=ids)
    return builder


def test_every_fixture_exercise_key_exists_in_the_ontology() -> None:
    for key in FIXTURE_EXERCISE_KEYS:
        ONTOLOGY.spec(key)


def test_fixture_exercises_are_ontology_keys_not_invented_names() -> None:
    assert set(FIXTURE_EXERCISE_KEYS) <= {spec.key for spec in ONTOLOGY.specs}


def test_fixture_rows_match_the_ontology_descriptors_exactly() -> None:
    """The registry is the single source of truth for what each exercise is.

    Any descriptor the fixture disagrees on would mean the fixtures exercise a
    vocabulary no real ingestion would ever produce.
    """
    builder = _builder()
    exercises = add_exercises(builder)
    for key, record in exercises.items():
        expected = definition_record_for(ONTOLOGY, key, ingested_at=INGESTED_AT)
        for field in (
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
            "definition_note",
        ):
            assert getattr(record, field) == getattr(expected, field), (key, field)


def test_fixture_exercise_identifiers_are_the_ontology_identifiers() -> None:
    """A cited exercise must be the one the registry derives, not a look-alike."""
    builder = _builder()
    exercises = add_exercises(builder)
    for key, record in exercises.items():
        assert record.exercise_id == ONTOLOGY.exercise_id(key), key


def test_fixture_aliases_come_from_the_registry() -> None:
    """Alias rows must be real bindings, so a fixture log resolves like a real one."""
    builder = _builder()
    add_exercises(builder)
    aliases = builder.build()["exercise_alias"]
    assert aliases
    registered = {
        (binding.source_system, binding.raw_label): binding.exercise_key
        for binding in ONTOLOGY.aliases
    }
    for alias in aliases:
        dumped = alias.model_dump()
        raw_label = str(dumped["alias_raw"])
        key = registered.get((str(dumped["source_system"]), raw_label))
        assert key is not None, raw_label
        assert dumped["exercise_id"] == ONTOLOGY.exercise_id(key), raw_label


def test_fixture_alias_rows_carry_the_registry_versions() -> None:
    builder = _builder()
    add_exercises(builder)
    for alias in builder.build()["exercise_alias"]:
        dumped = alias.model_dump()
        assert dumped["mapping_version"] == ONTOLOGY.alias_registry_version.tag
        assert dumped["ontology_version"] == ONTOLOGY.ontology_version.tag


@pytest.mark.parametrize("build", FIXTURES, ids=lambda call: call.__name__)
def test_every_fixture_references_ontologically_declared_exercises(
    build: Callable[[], dict[str, list[BaseModel]]],
) -> None:
    """No fixture row may cite an exercise the registry does not declare."""
    rows = build()
    declared = {str(record.model_dump()["exercise_id"]) for record in rows["exercise_definition"]}
    assert declared
    for table in ("planned_exercise", "performed_exercise", "performance_test"):
        for row in rows[table]:
            cited = row.model_dump().get("exercise_id")
            if cited is not None:
                assert str(cited) in declared, (table, cited)


@pytest.mark.parametrize("build", FIXTURES, ids=lambda call: call.__name__)
def test_every_fixture_populates_the_exercise_tables(
    build: Callable[[], dict[str, list[BaseModel]]],
) -> None:
    rows = build()
    assert set(rows) == set(table_names())
    row_counts = counts(rows)
    assert row_counts.get("exercise_definition", 0) == len(FIXTURE_EXERCISE_KEYS)
    assert row_counts.get("exercise_alias", 0) >= 1


@pytest.mark.parametrize("build", FIXTURES, ids=lambda call: call.__name__)
def test_fixture_exercise_rows_carry_the_fixture_source_provenance(
    build: Callable[[], dict[str, list[BaseModel]]],
) -> None:
    """Registry rows are re-stamped on ingest, so a dataset stays traceable."""
    rows = build()
    sources = {str(record.model_dump()["source_id"]) for record in rows["source"]}
    for record in rows["exercise_definition"]:
        dumped = record.model_dump()
        assert str(dumped["source_id"]) in sources
        assert str(dumped["source_record_key"]).startswith("exercise:")
    for record in rows["exercise_alias"]:
        assert str(record.model_dump()["source_id"]) in sources
