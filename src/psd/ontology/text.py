"""Deterministic exercise-label text normalization.

Every heterogeneous label reaches the ontology as free text, so the first thing
PSD does is reduce it to one canonical spelling *without changing what it says*.
This module owns that reduction and nothing else: it never looks at the exercise
vocabulary and never decides that two labels mean the same exercise.

Why a dedicated step
--------------------

Source labels differ in ways that carry no meaning: case, surrounding and
repeated whitespace, punctuation, diacritics, plural endings, and a small set of
domain abbreviations that logs write interchangeably ("db" / "dumbbell",
"2ct" / "2 count"). If those differences minted new identities, the alias
registry would have to enumerate every spelling permutation of every exercise.
Normalizing first is what keeps the registry small enough to review by hand.

Determinism contract
--------------------

Given the same raw string, this module returns the same
:class:`NormalizedLabel` on every platform and in every process, because:

* no locale-sensitive operation is used (``str.lower`` depends on nothing, but
  ``str.casefold`` is applied *after* NFKC so that compatibility decomposition
  happens before folding);
* no dictionary iteration order, hash seed, or filesystem is involved;
* every rule is a frozen table declared in this module;
* the function is idempotent: ``normalize_label(normalize_label(x).text).text``
  equals ``normalize_label(x).text``.

Rule order
----------

The order is part of the contract, not an implementation detail::

    unicode_nfkc           compatibility composition first, so that a fullwidth
                           or ligature form folds like its ASCII counterpart
    unicode_casefold       then case, which may expand a character (ß -> ss)
    unicode_nfkc_recheck   the expansion can be re-composed; do it once, explicitly
    strip_diacritics       marks are removed before punctuation becomes a space
    punctuation_to_space   every non-alphanumeric character becomes a separator
    collapse_whitespace    runs of separators become one
    expand_abbreviation    token-wise, from a frozen table
    fold_plural            trailing plurals become singular, from a frozen table
    join_compound          multi-token names become one token, longest match first

Plural folding runs *before* compound joining so that "bench presses" reaches the
join table already spelled "bench press" and becomes "benchpress" like every other
spelling. The reverse order would leave "bench presses" as its own lookup key and
mint a second identity for one exercise.

``question_form`` is recorded rather than applied. A source label ending in a
question mark is the *source* expressing doubt ("Leg Press?"), not PSD's, so the
fact is carried to the resolver instead of being silently normalized away.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from typing import Final

__all__ = (
    "ABBREVIATIONS",
    "COMPOUND_NAMES",
    "PLURAL_TO_SINGULAR",
    "NormalizedLabel",
    "normalize_label",
)

#: Characters kept as part of a token. Everything else becomes a separator.
_TOKEN_ALPHABET: Final[frozenset[str]] = frozenset("abcdefghijklmnopqrstuvwxyz0123456789")

#: Token-level abbreviation expansions, applied to the whole token only.
#:
#: Deliberately tiny. Every entry has to be a spelling that unambiguously names
#: the same thing as its expansion in a training log, because an abbreviation that
#: means different things in different apps would silently merge two exercises.
#: ``bp`` is the obvious omission: it reads as bench press in a lifting log and as
#: blood pressure everywhere else, so PSD leaves it alone.
ABBREVIATIONS: Final[dict[str, tuple[str, ...]]] = {
    "db": ("dumbbell",),
    "dbs": ("dumbbell",),
    "dbel": ("dumbbell",),
    "dbell": ("dumbbell",),
    "dbells": ("dumbbell",),
    "dumbels": ("dumbbell",),
    "dumbells": ("dumbbell",),
    "kb": ("kettlebell",),
    "kbs": ("kettlebell",),
    "kbel": ("kettlebell",),
    "kbells": ("kettlebell",),
    "ohp": ("overhead", "press"),
    "rdl": ("romanian", "deadlift"),
    "rdls": ("romanian", "deadlift"),
    "dl": ("deadlift",),
    "tbar": ("t", "bar"),
    "1ct": ("1", "count"),
    "2ct": ("2", "count"),
    "3ct": ("3", "count"),
    "ct": ("count",),
}

#: Multi-token names joined into a single token, longest phrase first.
#:
#: Joining is what makes "Bench Press", "Benchpress", and "bench-press" one
#: lookup. It is applied *after* abbreviation expansion so that ``OHP`` becomes
#: ``overhead press`` and then ``overheadpress``.
COMPOUND_NAMES: Final[tuple[tuple[tuple[str, ...], str], ...]] = (
    (("bench", "press"), "benchpress"),
    (("overhead", "press"), "overheadpress"),
    (("lat", "pulldown"), "latpulldown"),
    (("leg", "press"), "legpress"),
    (("chest", "press"), "chestpress"),
    (("close", "grip"), "closegrip"),
    (("wide", "grip"), "widegrip"),
    (("low", "bar"), "lowbar"),
    (("high", "bar"), "highbar"),
)

#: Explicit plural-to-singular token map.
#:
#: Not a general ``-s`` stripper. English plurals in exercise names are irregular
#: enough -- "Bench Press" -> "Benches", "Bench Presses" -- that a mechanical rule
#: would also "correct" words it should not, so only spellings that genuinely
#: occur in training logs are listed.
PLURAL_TO_SINGULAR: Final[dict[str, str]] = {
    "barbells": "barbell",
    "benches": "bench",
    "cables": "cable",
    "dumbbells": "dumbbell",
    "kettlebells": "kettlebell",
    "presses": "press",
    "squats": "squat",
    "deadlifts": "deadlift",
    "curls": "curl",
    "pulls": "pull",
    "rows": "row",
    "raises": "raise",
    "flies": "fly",
    "dips": "dip",
    "extensions": "extension",
    "pushdowns": "pushdown",
    "pulldowns": "pulldown",
    "pullups": "pullup",
    "lunges": "lunge",
    "thrusts": "thrust",
    "shrugs": "shrug",
    "crunches": "crunch",
    "situps": "situp",
    "stepups": "stepup",
    "legpresses": "legpress",
    "goodmornings": "goodmorning",
    "sideplanks": "sideplank",
    "planks": "plank",
}

_COMPOUND_BY_LENGTH: Final[dict[int, tuple[tuple[tuple[str, ...], str], ...]]] = {
    size: tuple(entry for entry in COMPOUND_NAMES if len(entry[0]) == size)
    for size in {len(phrase) for phrase, _replacement in COMPOUND_NAMES}
}
_COMPOUND_LENGTHS: Final[tuple[int, ...]] = tuple(sorted(_COMPOUND_BY_LENGTH, reverse=True))

_QUESTION_MARK: Final[str] = "?"

# Every rule identifier this module can emit is declared in the controlled
# vocabulary, so ``exercise_normalization.normalization_rules`` can be validated
# as a column without importing the ontology.


@dataclass(frozen=True, slots=True)
class NormalizedLabel:
    """One normalized label and the audit trail of how it got there.

    Attributes:
        text: The normalized text, lowercase, alphanumeric, single-spaced.
        rules: Identifiers of the rules that changed the text, in application
            order. Two labels that normalize to the same text through different
            rule sets are still the same lookup key; the rules exist so that a
            surprising mapping can be traced back to the step that caused it.
        question_form: Whether the source label ended in a question mark, which
            records that the *source* was unsure rather than asserting anything.
    """

    text: str
    rules: tuple[str, ...]
    question_form: bool = False

    def __bool__(self) -> bool:
        return bool(self.text)


def normalize_label(raw: str) -> NormalizedLabel:
    """Return the canonical spelling of *raw*.

    Args:
        raw: The verbatim source label. It is never modified.

    Returns:
        The normalized text together with the rules that produced it.

    Raises:
        TypeError: *raw* is not a string. Callers holding a possibly-absent source
            value must decide what a missing label means; normalization cannot.
    """
    rules: list[str] = []

    try:
        composed = unicodedata.normalize("NFKC", raw)
    except TypeError as error:
        msg = f"normalize_label expects str; got {type(raw).__name__!r}."
        raise TypeError(msg) from error
    if composed != raw:
        rules.append("unicode_nfkc")

    folded = composed.casefold()
    if folded != composed:
        rules.append("unicode_casefold")

    recheck = unicodedata.normalize("NFKC", folded)
    if recheck != folded:
        rules.append("unicode_nfkc_recheck")

    question_form = _ends_with_question_mark(recheck)
    if question_form:
        rules.append("question_form")

    decomposed = unicodedata.normalize("NFKD", recheck)
    stripped = "".join(char for char in decomposed if not unicodedata.combining(char))
    if stripped != recheck:
        rules.append("strip_diacritics")

    separated = "".join(char if char in _TOKEN_ALPHABET else " " for char in stripped)
    if separated != stripped:
        rules.append("punctuation_to_space")

    collapsed = " ".join(separated.split())
    if collapsed != separated:
        rules.append("collapse_whitespace")

    tokens = collapsed.split() if collapsed else []
    expanded = _expand_abbreviations(tokens)
    if expanded != tokens:
        rules.append("expand_abbreviation")

    folded_plural = _fold_plurals(expanded)
    if folded_plural != expanded:
        rules.append("fold_plural")

    joined = _join_compounds(folded_plural)
    if joined != folded_plural:
        rules.append("join_compound")

    return NormalizedLabel(
        text=" ".join(folded_plural),
        rules=tuple(rules),
        question_form=question_form,
    )


def _ends_with_question_mark(text: str) -> bool:
    """Return whether *text*'s last non-whitespace character is a question mark."""
    stripped = text.rstrip()
    return stripped.endswith(_QUESTION_MARK)


def _expand_abbreviations(tokens: list[str]) -> list[str]:
    """Replace whole-token abbreviations with their expansions."""
    expanded: list[str] = []
    for token in tokens:
        expanded.extend(ABBREVIATIONS.get(token, (token,)))
    return expanded


def _join_compounds(tokens: list[str]) -> list[str]:
    """Join recognized multi-token names, preferring the longest phrase."""
    joined: list[str] = []
    index = 0
    length = len(tokens)
    while index < length:
        matched = False
        for size in _COMPOUND_LENGTHS:
            end = index + size
            if end > length:
                continue
            window = tuple(tokens[index:end])
            for phrase, replacement in _COMPOUND_BY_LENGTH[size]:
                if window == phrase:
                    joined.append(replacement)
                    index = end
                    matched = True
                    break
            if matched:
                break
        if not matched:
            joined.append(tokens[index])
            index += 1
    return joined


def _fold_plurals(tokens: list[str]) -> list[str]:
    """Replace known plural spellings with their singular form."""
    return [PLURAL_TO_SINGULAR.get(token, token) for token in tokens]
