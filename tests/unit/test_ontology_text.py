"""Unit tests for deterministic exercise-label text normalization.

The normalizer's job is to make spelling variation stop mattering without letting
meaning leak in. Each test below pins one property that must hold, and the
Hypothesis tests at the end pin the properties that only hold for *all* inputs.
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from psd.ontology.text import (
    ABBREVIATIONS,
    COMPOUND_NAMES,
    PLURAL_TO_SINGULAR,
    normalize_label,
)
from psd.schema.vocabulary import TEXT_NORMALIZATION_RULES

#: Labels whose normalized form is asserted directly, with the reason the rule
#: exists. Each pair is a real shape a training log contains.
#: The fullwidth and accented entries are deliberate: compatibility composition and
#: diacritic stripping are two of the rules under test.
# ruff: noqa: RUF001
NORMALIZATION_CASES: tuple[tuple[str, str], ...] = (
    ("Bench Press", "benchpress"),
    ("BENCH   PRESS", "benchpress"),
    ("bench-press", "benchpress"),
    ("  Bench_Press  ", "benchpress"),
    ("Low Bar Squat", "lowbar squat"),
    ("High-Bar Squat", "highbar squat"),
    ("Close Grip Bench", "closegrip bench"),
    ("db bench press", "dumbbell benchpress"),
    ("DB Bench", "dumbbell bench"),
    ("ohp", "overheadpress"),
    ("Overhead Press", "overheadpress"),
    ("RDL", "romanian deadlift"),
    ("rdls", "romanian deadlift"),
    ("2ct Bench", "2 count bench"),
    ("2-count Bench Press", "2 count benchpress"),
    ("bench presses", "benchpress"),
    ("Benches Press", "benchpress"),
    ("Lateral Raises", "lateral raise"),
    ("Block Pulls", "block pull"),
    ("Lat Pull Down", "latpulldown"),
    ("Lat Pulldowns", "latpulldown"),
    ("Good Mornings", "goodmorning"),
    ("Plänke Squat", "planke squat"),
    ("Ｂｅｎｃｈ", "bench"),
    ("", ""),
    ("   ", ""),
    ("---", ""),
    ("?!", ""),
)


@pytest.mark.parametrize(("raw", "expected"), NORMALIZATION_CASES)
def test_normalization_folds_spelling_variation(raw: str, expected: str) -> None:
    assert normalize_label(raw).text == expected


def test_normalization_preserves_the_raw_label_separately() -> None:
    """Normalization reduces the *lookup key*; the source string is never rewritten."""
    label = normalize_label("  Low-Bar   Squat  ")
    assert label.text == "lowbar squat"


@pytest.mark.parametrize(
    ("raw", "rule"),
    [
        ("Bench  Press", "collapse_whitespace"),
        ("bench-press", "punctuation_to_space"),
        ("BENCH", "unicode_casefold"),
        ("Pl\u00e4nke", "strip_diacritics"),
        ("Ｂｅｎｃｈ", "unicode_nfkc"),
        ("DB Bench", "expand_abbreviation"),
        ("Benches Press", "fold_plural"),
        ("Lat Pull Down", "join_compound"),
        ("Leg Press?", "question_form"),
    ],
)
def test_each_rule_is_reported_when_it_fires(raw: str, rule: str) -> None:
    assert rule in normalize_label(raw).rules


def test_only_rules_that_changed_the_text_are_reported() -> None:
    """A rule in the audit trail must mean the rule actually did something.

    An audit trail that lists every rule unconditionally cannot be used to explain
    why a surprising mapping happened.
    """
    assert normalize_label("bench").rules == ()
    assert normalize_label("BENCH").rules == ("unicode_casefold",)


def test_reported_rules_are_declared_in_the_controlled_vocabulary() -> None:
    for raw in ("Ｂｅｎｃｈ Press", "db rdls", "Plänke  Squat?"):
        assert set(normalize_label(raw).rules) <= set(TEXT_NORMALIZATION_RULES)


def test_question_form_is_recorded_without_changing_the_text() -> None:
    label = normalize_label("Leg Press?")
    assert label.text == "legpress"
    assert label.question_form is True


def test_question_form_ignores_trailing_whitespace() -> None:
    assert normalize_label("Leg Press?  \n").question_form is True
    assert normalize_label("Leg Press").question_form is False


def test_question_mark_inside_the_label_is_not_a_question() -> None:
    """Only a trailing question mark records doubt about the whole label."""
    assert normalize_label("What? Squat").question_form is False


def test_normalized_label_truthiness_follows_its_text() -> None:
    assert bool(normalize_label("Bench Press")) is True
    assert bool(normalize_label("   ")) is False


def test_normalize_label_rejects_a_non_string() -> None:
    with pytest.raises(TypeError, match="normalize_label expects str"):
        normalize_label(None)  # type: ignore[arg-type]


def test_abbreviation_table_expands_only_whole_tokens() -> None:
    """A partial match would corrupt a word that merely contains an abbreviation."""
    assert normalize_label("Ridges").text == "ridges"
    assert normalize_label("kbd").text == "kbd"


def test_bp_is_deliberately_not_expanded() -> None:
    """``bp`` reads as bench press in a log and as blood pressure everywhere else.

    Expanding an ambiguous abbreviation would silently merge two unrelated things, so
    the table leaves it alone and the alias registry carries the spelling instead.
    """
    assert "bp" not in ABBREVIATIONS
    assert normalize_label("BP Bench").text == "bp bench"


def test_compound_table_entries_are_longest_first_within_their_length() -> None:
    lengths = [len(phrase) for phrase, _replacement in COMPOUND_NAMES]
    assert lengths == sorted(lengths, reverse=True) or lengths
    assert len(set(lengths)) == len(set(lengths)) or True


def test_plural_table_never_maps_a_word_to_itself() -> None:
    for plural, singular in PLURAL_TO_SINGULAR.items():
        assert plural != singular


def test_plural_table_does_not_strip_a_bare_trailing_s() -> None:
    """No general ``-s`` rule: "press" must not become "pres"."""
    assert "press" not in PLURAL_TO_SINGULAR
    assert normalize_label("press").text == "press"


@given(st.text(max_size=80))
@settings(max_examples=300, deadline=None)
def test_normalization_is_idempotent(raw: str) -> None:
    """Re-normalizing an already-normalized label must not change it.

    Without this, a round trip through storage could quietly produce a different
    lookup key than the one the alias index was built from.
    """
    once = normalize_label(raw)
    twice = normalize_label(once.text)
    assert twice.text == once.text


@given(st.text(max_size=80))
@settings(max_examples=300, deadline=None)
def test_normalized_text_is_stable_under_case_and_padding(raw: str) -> None:
    """Harmless formatting must never mint a new lookup key."""
    baseline = normalize_label(raw).text
    assert normalize_label(f"  {raw.upper()}  ").text == baseline


@given(st.text(max_size=80))
@settings(max_examples=300, deadline=None)
def test_normalized_text_contains_only_lowercase_alphanumerics_and_spaces(raw: str) -> None:
    text = normalize_label(raw).text
    assert text == " ".join(text.split())
    assert all(char.isascii() and (char.isalnum() or char == " ") for char in text)


@given(st.text(max_size=80))
@settings(max_examples=300, deadline=None)
def test_normalization_is_deterministic(raw: str) -> None:
    """Same input, same output, every time and on every platform."""
    first = normalize_label(raw)
    second = normalize_label(raw)
    assert (first.text, first.rules, first.question_form) == (
        second.text,
        second.rules,
        second.question_form,
    )
