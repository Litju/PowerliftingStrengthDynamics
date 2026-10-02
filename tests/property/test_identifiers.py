"""Unit and property tests for deterministic identifiers."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from psd.schema.identifiers import (
    IDENTIFIER_LENGTH,
    IdPrefix,
    derive_id,
    id_key,
    is_valid_id,
    make_id,
)

pytestmark = pytest.mark.windows_parity


def test_identifier_shape() -> None:
    value = make_id(IdPrefix.ATHLETE, "source-athlete-42")
    assert value.startswith("ath_")
    assert is_valid_id(value, IdPrefix.ATHLETE)


def test_identifier_is_deterministic() -> None:
    first = make_id(IdPrefix.PERFORMED_SET, "pses_abc", 3)
    second = make_id(IdPrefix.PERFORMED_SET, "pses_abc", 3)
    assert first == second


def test_different_ordinals_produce_different_identifiers() -> None:
    assert make_id(IdPrefix.PERFORMED_SET, "pses_abc", 1) != make_id(
        IdPrefix.PERFORMED_SET, "pses_abc", 2
    )


def test_absent_parts_are_encoded_distinctly() -> None:
    """A missing part must not collide with the literal text of its marker."""
    assert make_id(IdPrefix.PERFORMED_SET, "parent", None) != make_id(
        IdPrefix.PERFORMED_SET, "parent", "n"
    )
    assert make_id(IdPrefix.PERFORMED_SET, "parent", None) == make_id(
        IdPrefix.PERFORMED_SET, "parent", None
    )


def test_numeric_and_string_parts_are_not_interchangeable() -> None:
    assert make_id(IdPrefix.PERFORMED_SET, "parent", 1) != make_id(
        IdPrefix.PERFORMED_SET, "parent", "1"
    )


def test_length_prefixed_keys_remove_separator_ambiguity() -> None:
    assert id_key(["ab", "c"]) != id_key(["a", "bc"])
    assert make_id(IdPrefix.OBSERVATION, "ab", "c") != make_id(IdPrefix.OBSERVATION, "a", "bc")


def test_separator_characters_in_parts_do_not_collide() -> None:
    assert make_id(IdPrefix.OBSERVATION, "a\x1fb", "c") != make_id(
        IdPrefix.OBSERVATION, "a", "b", "c"
    )


def test_prefixes_are_disjoint() -> None:
    prefixes = [prefix.value for prefix in IdPrefix]
    assert len(prefixes) == len(set(prefixes))
    assert all("_" not in prefix for prefix in prefixes)


def test_derive_id_chains_from_parent() -> None:
    parent = make_id(IdPrefix.PERFORMED_EXERCISE, "pses_abc", 1)
    child = derive_id(parent, IdPrefix.PERFORMED_SET, 2)
    assert child.startswith("pset_")
    assert child == derive_id(parent, IdPrefix.PERFORMED_SET, 2)


@pytest.mark.parametrize(
    "value",
    ["", "ath", "ath_", "ath_xyz", "ath_" + "g" * IDENTIFIER_LENGTH, "ath_" + "0" * 31],
)
def test_is_valid_id_rejects_malformed_values(value: str) -> None:
    assert not is_valid_id(value)


def test_is_valid_id_without_prefix_accepts_any_known_prefix() -> None:
    assert is_valid_id(make_id(IdPrefix.SOURCE, "x"))
    assert not is_valid_id("zzz_" + "0" * IDENTIFIER_LENGTH)


@given(
    parts=st.lists(
        st.one_of(st.none(), st.text(min_size=0, max_size=32), st.integers(-1000, 1000)),
        min_size=1,
        max_size=6,
    ),
    prefix=st.sampled_from(list(IdPrefix)),
)
def test_identifiers_always_well_formed(parts: list[str | int | None], prefix: IdPrefix) -> None:
    value = make_id(prefix, *parts)
    assert is_valid_id(value, prefix)
    assert len(value) == len(prefix.value) + 1 + IDENTIFIER_LENGTH


@given(
    left=st.text(min_size=0, max_size=24),
    right=st.text(min_size=0, max_size=24),
)
def test_distinct_part_lists_produce_distinct_identifiers(left: str, right: str) -> None:
    assert make_id(IdPrefix.EXERCISE_ALIAS, left) != make_id(IdPrefix.EXERCISE_ALIAS, left, right)
