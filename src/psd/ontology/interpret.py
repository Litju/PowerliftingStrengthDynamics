"""Conservative structured interpretation of an exercise label.

The resolver in :mod:`psd.ontology.registry` resolves a label in several stages.
Three of them are lookups. This module is the fourth: after text normalization
and an exact-alias lookup have both failed, it reads the label as a set of
descriptors and asks whether exactly one canonical exercise matches all of them.

Why it is deliberately weak
---------------------------

A compositional parser is where an ontology quietly grows a transfer function.
The temptation is to parse "paused high bar squat" and synthesize a
high-bar-paused-squat entry. PSD must not: that entry would be an assertion about
how much the variation is worth, and no source ever said it.

So the parser only ever *selects* an exercise that already exists:

* every parsed descriptor must equal the candidate's descriptor, and a candidate
  whose descriptor is ``UNKNOWN`` never matches a label that specified one;
* at least one descriptor beyond the family must be parsed, otherwise the label
  says nothing a family-level answer would not say better;
* the family token must name exactly one family;
* exactly one candidate may survive.

Anything else is not resolved. A label the parser cannot place exactly stays
ambiguous or unmapped, which is the intended outcome: the parser is allowed to say
"I don't know", and it says so often.

Phrase matching
---------------

Every table lists both the multi-token spelling and the joined spelling, because
:mod:`psd.ontology.text` has already joined compound names by the time a label
reaches the parser: ``close grip bench press`` arrives as ``closegrip
benchpress``. A matched phrase has its tokens removed, so ``wide grip bench`` is
read as a wide *grip* and never re-read as a wide *stance*.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from psd.schema.vocabulary import (
    Grip,
    ImplementType,
    ParentLift,
    PauseRule,
    RangeOfMotion,
    Stance,
    TempoPattern,
)

__all__ = ("ParsedFeatures", "parse_features")


@dataclass(frozen=True, slots=True)
class ParsedFeatures:
    """Descriptors read out of a normalized label.

    Attributes:
        family: The parent lift the label names, if it names exactly one.
        implement: The implement the label names.
        stance: The stance the label names.
        grip: The grip the label names.
        range_of_motion: The range-of-motion modifier the label names.
        pause_rule: The pause the label names.
        tempo: The tempo the label names.
        has_descriptor: Whether at least one descriptor beyond the family was
            read. This is the precondition for any structured resolution.
    """

    family: ParentLift = ParentLift.UNKNOWN
    implement: ImplementType = ImplementType.UNKNOWN
    stance: Stance = Stance.UNKNOWN
    grip: Grip = Grip.UNKNOWN
    range_of_motion: RangeOfMotion = RangeOfMotion.UNKNOWN
    pause_rule: PauseRule = PauseRule.UNKNOWN
    tempo: TempoPattern = TempoPattern.UNKNOWN
    has_descriptor: bool = False


_IMPLEMENT_TOKENS: Final[dict[str, ImplementType]] = {
    "barbell": ImplementType.BARBELL,
    "dumbbell": ImplementType.DUMBBELL,
    "kettlebell": ImplementType.KETTLEBELL,
    "cable": ImplementType.CABLE,
    "band": ImplementType.BAND,
}

#: Tokens that name the accessory family.
#:
#: ``press`` is deliberately absent. "Press" in a training log may mean the bench
#: press or the overhead press, and guessing would be exactly the forced mapping
#: this ontology refuses to make.
_ACCESSORY_TOKENS: Final[frozenset[str]] = frozenset(
    {
        "calf",
        "carry",
        "clean",
        "crunch",
        "curl",
        "dip",
        "extension",
        "farmer",
        "fly",
        "goodmorning",
        "jerk",
        "latpulldown",
        "lunge",
        "plank",
        "pulldown",
        "pullup",
        "pushdown",
        "raise",
        "reverse",
        "row",
        "shrug",
        "situp",
        "sissy",
        "sideplank",
        "snatch",
        "stepup",
        "thrust",
        "upright",
    }
)

_FAMILY_TOKENS: Final[dict[str, ParentLift]] = {
    "squat": ParentLift.SQUAT,
    "bench": ParentLift.BENCH,
    "benchpress": ParentLift.BENCH,
    "deadlift": ParentLift.DEADLIFT,
}

_GRIP_PHRASES: Final[tuple[tuple[tuple[str, ...], Grip], ...]] = (
    (("close", "grip"), Grip.CLOSE),
    (("closegrip",), Grip.CLOSE),
    (("wide", "grip"), Grip.WIDE),
    (("widegrip",), Grip.WIDE),
    (("neutral", "grip"), Grip.NEUTRAL),
    (("mixed", "grip"), Grip.MIXED),
    (("competition", "grip"), Grip.COMPETITION),
    (("overhand",), Grip.OVERHAND),
    (("underhand",), Grip.UNDERHAND),
    (("supinated",), Grip.SUPINATED),
    (("pronated",), Grip.PRONATED),
)

_STANCE_PHRASES: Final[tuple[tuple[tuple[str, ...], Stance], ...]] = (
    (("low", "bar"), Stance.LOW),
    (("lowbar",), Stance.LOW),
    (("high", "bar"), Stance.HIGH),
    (("highbar",), Stance.HIGH),
    (("sumo",), Stance.WIDE),
    (("conventional",), Stance.MODERATE),
    (("wide",), Stance.WIDE),
    (("moderate",), Stance.MODERATE),
    (("narrow",), Stance.NARROW),
    (("split",), Stance.SPLIT),
)

_RANGE_PHRASES: Final[tuple[tuple[tuple[str, ...], RangeOfMotion], ...]] = (
    (("short",), RangeOfMotion.REDUCED),
    (("reduced",), RangeOfMotion.REDUCED),
    (("partial",), RangeOfMotion.PARTIAL),
    (("deficit",), RangeOfMotion.ELEVATED_START),
    (("competition",), RangeOfMotion.COMPETITION),
    (("full",), RangeOfMotion.FULL),
)

#: A 1-count is absent on purpose: "1 count" is used both for a pause and for a
#: tempo scheme, so it is not a pause descriptor PSD can defend.
_PAUSE_PHRASES: Final[tuple[tuple[tuple[str, ...], PauseRule], ...]] = (
    (("touch", "and", "go"), PauseRule.NONE),
    (("long", "pause"), PauseRule.LONG),
    (("3", "count"), PauseRule.COUNT_3),
    (("2", "count"), PauseRule.COUNT_2),
    (("paused",), PauseRule.BRIEF),
    (("pause",), PauseRule.BRIEF),
)

_TEMPO_PHRASES: Final[tuple[tuple[tuple[str, ...], TempoPattern], ...]] = (
    (("controlled", "eccentric"), TempoPattern.CONTROLLED_ECCENTRIC),
    (("eccentric",), TempoPattern.CONTROLLED_ECCENTRIC),
    (("tempo",), TempoPattern.TEMPO_PRESCRIBED),
)

#: Implement phrases, applied after the named tables so their tokens are already
#: consumed by the time these run.
_IMPLEMENT_PHRASES: Final[tuple[tuple[tuple[str, ...], ImplementType], ...]] = (
    (("body", "weight"), ImplementType.BODYWEIGHT),
    (("bodyweight",), ImplementType.BODYWEIGHT),
    (("smith", "machine"), ImplementType.SMITH_MACHINE),
    (("smith",), ImplementType.SMITH_MACHINE),
    (("machine",), ImplementType.MACHINE),
    (("cable",), ImplementType.CABLE),
    (("barbell",), ImplementType.BARBELL),
    (("dumbbell",), ImplementType.DUMBBELL),
    (("kettlebell",), ImplementType.KETTLEBELL),
    (("band",), ImplementType.BAND),
)


def parse_features(normalized_text: str) -> ParsedFeatures:
    """Read descriptors out of already-normalized label text.

    Args:
        normalized_text: Output of :func:`psd.ontology.text.normalize_label`.

    Returns:
        The descriptors the label states. A descriptor the label does not state
        comes back as ``UNKNOWN``, which matches no candidate.
    """
    tokens = normalized_text.split()
    if not tokens:
        return ParsedFeatures()

    grip, rest = _take(tokens, _GRIP_PHRASES, Grip.UNKNOWN)
    stance, rest = _take(rest, _STANCE_PHRASES, Stance.UNKNOWN)
    pause_rule, rest = _take(rest, _PAUSE_PHRASES, PauseRule.UNKNOWN)
    tempo, rest = _take(rest, _TEMPO_PHRASES, TempoPattern.UNKNOWN)
    range_of_motion, rest = _take(rest, _RANGE_PHRASES, RangeOfMotion.UNKNOWN)
    implement, rest = _take(rest, _IMPLEMENT_PHRASES, ImplementType.UNKNOWN)

    return ParsedFeatures(
        family=_family_of(rest),
        implement=implement,
        stance=stance,
        grip=grip,
        range_of_motion=range_of_motion,
        pause_rule=pause_rule,
        tempo=tempo,
        has_descriptor=(
            implement is not ImplementType.UNKNOWN
            or stance is not Stance.UNKNOWN
            or grip is not Grip.UNKNOWN
            or range_of_motion is not RangeOfMotion.UNKNOWN
            or pause_rule is not PauseRule.UNKNOWN
            or tempo is not TempoPattern.UNKNOWN
        ),
    )


def _take[EnumT: StrEnum](
    tokens: list[str], phrases: tuple[tuple[tuple[str, ...], EnumT], ...], default: EnumT
) -> tuple[EnumT, list[str]]:
    """Extract the first matching phrase, returning it and the leftover tokens.

    Args:
        tokens: Remaining tokens, scanned left to right.
        phrases: Token sequences paired with the descriptor they name. Longer
            phrases must be listed before shorter ones that they could shadow.
        default: Descriptor returned when nothing matches.

    Returns:
        The descriptor and the tokens it did not consume.
    """
    for phrase, value in phrases:
        span = _find(tokens, phrase)
        if span is not None:
            start, end = span
            return value, [*tokens[:start], *tokens[end:]]
    return default, list(tokens)


def _find(tokens: list[str], phrase: tuple[str, ...]) -> tuple[int, int] | None:
    """Return the half-open span of the leftmost occurrence of *phrase*."""
    width = len(phrase)
    for start in range(len(tokens) - width + 1):
        if tuple(tokens[start : start + width]) == phrase:
            return start, start + width
    return None


def _family_of(tokens: list[str]) -> ParentLift:
    """Return the single parent lift the tokens name, or ``UNKNOWN``.

    More than one family is not an error; it means the label names no single
    family and therefore supports no structured resolution.
    """
    families: set[ParentLift] = set()
    for token in tokens:
        named = _FAMILY_TOKENS.get(token)
        if named is not None:
            families.add(named)
        elif token in _ACCESSORY_TOKENS:
            families.add(ParentLift.ACCESSORY)
    if len(families) != 1:
        return ParentLift.UNKNOWN
    return next(iter(families))
