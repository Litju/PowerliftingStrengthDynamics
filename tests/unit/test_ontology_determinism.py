"""Cross-platform determinism for the exercise ontology.

Windows and Ubuntu must produce identical normalization results *and* identical
persisted ordering. This module asserts both halves of that claim from inside one
process, by proving that neither the resolver nor the canonical ordering can depend on
anything platform-specific: locale, path separators, filesystem enumeration order, or
hash seed.

The complementary proof -- that the artifacts really do match between a Windows and a
Linux CI job -- is the CI job itself. What this module rules out is a *latent*
platform dependency, so a failure there points at a real cause rather than at
"the two platforms disagreed somewhere".
"""

from __future__ import annotations

import hashlib
import locale
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from psd.ontology import default_ontology
from psd.ontology.artifact import ONTOLOGY_EPOCH, build_ontology_dataset, ontology_relative_path
from psd.ontology.catalog import ALIASES, CURATED_AMBIGUOUS_LABELS, EXERCISES
from psd.ontology.records import (
    alias_records,
    definition_records,
    normalization_records,
    probe_outcomes,
)
from psd.ontology.registry import DEFAULT_SOURCE_SYSTEM, ExerciseOntology
from psd.schema.registry import table_spec
from psd.schema.vocabulary import Grip
from psd.serialization.canonical import content_digest
from psd.serialization.dataset import CanonicalDataset
from psd.serialization.ordering import canonical_order
from psd.serialization.table import records_to_table

ONTOLOGY = default_ontology()

EXERCISE_TABLES: tuple[str, ...] = (
    "exercise_definition",
    "exercise_alias",
    "exercise_normalization",
)


def test_normalization_ignores_the_active_locale() -> None:
    """Locale-sensitive case folding would make the lookup key machine-dependent.

    Which locales a given platform actually has varies -- Windows ships far fewer than
    a typical Linux image -- so the test switches to each candidate it can reach and
    asserts the results never change, rather than asserting a particular failure.
    """
    baseline = {label: ONTOLOGY.resolve(label) for label in LABELS}
    applied: list[str] = []
    for candidate in ("C", "C.UTF-8", "tr_TR.UTF-8", "de_DE.UTF-8", "en_US.UTF-8"):
        try:
            locale.setlocale(locale.LC_ALL, candidate)
        except locale.Error:
            continue
        applied.append(candidate)
        assert {label: ONTOLOGY.resolve(label) for label in LABELS} == baseline, candidate
    assert applied, "no locale could be selected, so the check proved nothing"


def test_normalization_is_identical_under_a_different_hash_seed() -> None:
    """Dictionary iteration order must not leak into a result.

    Set iteration order over strings varies with the hash seed between processes, so
    re-running resolution in a subprocess with two fixed seeds is the only way to prove
    the result does not depend on it. Windows and Ubuntu start with different seeds by
    default, which is why this matters for the two-platform determinism claim.
    """
    script = (
        "import json\n"
        "from psd.ontology import default_ontology\n"
        "ontology = default_ontology()\n"
        f"labels = {list(LABELS)!r}\n"
        "print(json.dumps([[label, ontology.resolve(label).resolution_status.value,"
        " ontology.resolve(label).exercise_key,"
        " list(ontology.resolve(label).candidate_keys)] for label in labels]))\n"
    )
    outputs: list[str] = []
    for seed in ("0", "12345"):
        completed = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            check=True,
            env={"PYTHONHASHSEED": seed, "PYTHONPATH": os.pathsep.join(sys.path)},
        )
        outputs.append(completed.stdout.strip())
    assert outputs[0] == outputs[1], "normalization depends on the hash seed"


#: Labels spanning every ladder rung, so the determinism claim covers all of them.
LABELS: tuple[str, ...] = (
    "Squat",
    "Competition Squat",
    "Low Bar Squat",
    "2ct Bench",
    "Close Grip Bench",
    "CGBP",
    "Machine press",
    "Bench Variation",
    "Leg Press?",
    "RDL",
    "  low   bar  squat  ",
    "Zorblax Lever Exercise",
)


@pytest.mark.parametrize("table", EXERCISE_TABLES)
def test_persisted_ordering_comes_only_from_the_registry(table: str) -> None:
    """Sorting must be a function of the declared order, not of insertion order."""
    spec = table_spec(table)
    dataset = build_ontology_dataset(ONTOLOGY)
    resorted = canonical_order(dataset.table(table), spec.order_by, table_name=table)
    assert resorted.to_pylist() == dataset.table(table).to_pylist()
    assert content_digest(resorted, table_name=table) == content_digest(
        dataset.table(table), table_name=table
    )


@pytest.mark.parametrize("table", EXERCISE_TABLES)
def test_table_order_does_not_affect_the_content_digest(table: str) -> None:
    """A reversed table must digest the same as the canonical one once sorted."""
    if table == "exercise_definition":
        records = list(definition_records(ONTOLOGY, ingested_at=ONTOLOGY_EPOCH))
    elif table == "exercise_alias":
        records = list(alias_records(ONTOLOGY, ingested_at=ONTOLOGY_EPOCH))
    else:
        outcomes = probe_outcomes(ONTOLOGY, source_system=DEFAULT_SOURCE_SYSTEM)
        records = list(
            normalization_records(outcomes, ontology=ONTOLOGY, ingested_at=ONTOLOGY_EPOCH)
        )

    spec = table_spec(table)
    forward = canonical_order(
        records_to_table(records, table_name=table), spec.order_by, table_name=table
    )
    backward = canonical_order(
        records_to_table(list(reversed(records)), table_name=table),
        spec.order_by,
        table_name=table,
    )
    assert content_digest(forward, table_name=table) == content_digest(backward, table_name=table)


def test_two_ontologies_built_from_the_same_data_are_equal() -> None:
    """Construction must not depend on the order the declaration arrives in."""
    forward = ExerciseOntology(EXERCISES, ALIASES, CURATED_AMBIGUOUS_LABELS)
    backward = ExerciseOntology(
        tuple(reversed(EXERCISES)),
        tuple(reversed(ALIASES)),
        tuple(reversed(CURATED_AMBIGUOUS_LABELS)),
    )
    assert forward.specs == backward.specs
    assert forward.aliases == backward.aliases
    assert forward.curated_ambiguities == backward.curated_ambiguities
    assert forward.source_systems == backward.source_systems


def test_the_artifact_relative_path_is_identical_on_every_platform() -> None:
    """A path with a ``/`` inside a component would diverge between the two."""
    relative = ontology_relative_path(ONTOLOGY)
    assert relative.as_posix() == "canonical/ontology/psd-ontology-0.1.0"
    assert all("/" not in part and "\\" not in part for part in relative.parts)


def test_the_artifact_digest_is_stable_for_a_given_ontology() -> None:
    """A vocabulary change must show up as a digest change, and nothing else may."""
    assert _exercise_digest(build_ontology_dataset(ONTOLOGY)) == _exercise_digest(
        build_ontology_dataset(ONTOLOGY)
    )


def _exercise_digest(dataset: CanonicalDataset) -> str:
    """Return a single digest over all three exercise tables, order-independent."""
    digest = hashlib.sha256()
    for table in EXERCISE_TABLES:
        digest.update(content_digest(dataset.table(table), table_name=table).encode("ascii"))
    return digest.hexdigest()


def test_a_changed_ontology_changes_the_digest() -> None:
    """Otherwise the digest gate in CI would be checking nothing.

    A synthetic ontology differing in exactly one descriptor must not digest the same
    as the shipped one, so the change is minimal and attributable.
    """
    altered = tuple(
        replace(spec, grip=Grip.WIDE) if spec.key == "close_grip_bench" else spec
        for spec in EXERCISES
    )
    assert altered != EXERCISES
    baseline = _exercise_digest(build_ontology_dataset(ONTOLOGY))
    changed = _exercise_digest(
        build_ontology_dataset(ExerciseOntology(altered, ALIASES, CURATED_AMBIGUOUS_LABELS))
    )
    assert baseline != changed


def test_the_artifact_never_embeds_a_host_path() -> None:
    """A host path in a persisted artifact would break byte-identical rebuilds."""
    dataset = build_ontology_dataset(ONTOLOGY)
    for table in EXERCISE_TABLES:
        for row in dataset.table(table).to_pylist():
            for value in row.values():
                if isinstance(value, str):
                    assert str(Path.cwd()) not in value
                    assert "\\" not in value
