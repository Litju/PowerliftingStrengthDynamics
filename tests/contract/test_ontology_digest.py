"""The committed ontology digest must not drift.

A golden test rather than a derived one: the expected digests are committed in
:mod:`tests.fixtures.ontology_digest` and asserted here, so an accidental change to the
canonical vocabulary, the alias bindings, or the curated refusals fails on both the
Windows and the Ubuntu CI job, naming the table that moved.

Because ``content_digest`` is defined over logical values rather than Parquet bytes,
the two platforms must produce identical digests. This is therefore also the
cross-platform determinism gate for the ontology.
"""

from __future__ import annotations

import pytest

from tests.fixtures.ontology_digest import (
    EXPECTED_ONTOLOGY_DIGEST,
    EXPECTED_TABLE_DIGESTS,
    ONTOLOGY_TABLES,
    ontology_digest,
    table_digests,
)


def test_the_ontology_matches_its_committed_digest() -> None:
    """Any change to the vocabulary must be a deliberate, reviewed act."""
    actual = ontology_digest()
    assert actual == EXPECTED_ONTOLOGY_DIGEST, (
        "the exercise ontology content changed; if that was intended, update "
        "EXPECTED_ONTOLOGY_DIGEST and say why in the commit message"
    )


@pytest.mark.parametrize("table", ONTOLOGY_TABLES)
def test_each_ontology_table_matches_its_committed_digest(table: str) -> None:
    """Per-table digests, so a divergence names the table rather than the whole set."""
    actual = table_digests()[table]
    assert actual == EXPECTED_TABLE_DIGESTS[table], table


def test_the_committed_digests_are_lower_case_hex() -> None:
    for value in (EXPECTED_ONTOLOGY_DIGEST, *EXPECTED_TABLE_DIGESTS.values()):
        assert len(value) == 64
        assert value == value.lower()
        assert all(char in "0123456789abcdef" for char in value)


def test_the_expected_table_digest_set_is_complete() -> None:
    assert set(EXPECTED_TABLE_DIGESTS) == set(ONTOLOGY_TABLES)


def test_the_ontology_digest_is_stable_within_one_process() -> None:
    """The committed value must be reproducible, not merely currently correct."""
    assert ontology_digest() == ontology_digest()
    assert table_digests() == table_digests()
