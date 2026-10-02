"""OpenPowerlifting value semantics.

Every mapping the transform applies lives here, as data, with a reason attached. That
is deliberate: a mapping buried inside a Polars expression is a mapping nobody can
review, and the ones that matter most here are exactly the ones a reviewer would want
to argue with.

    Equipment = Wraps
        means the rules *allowed* wraps, not that the lifter wore them. This is a
        competition-category fact and stays on the competition row. No athlete-level
    equipment observation is ever derived from it.

    Age = 23.5
        means "23 or 24, we do not know which", because the source only had a birth
        year. The value is kept and its approximation is stated; rounding it would
        invent a precision the source did not have.

    Place = DQ
        is not a placing. Guest, disqualified, doping-disqualified, and no-show are
        statuses, and coercing any of them to a number would invent a rank.

    WeightClassKg = 90+
        is an open-ended class. It is kept verbatim; collapsing it to the number beside
        it would claim a maximum the source never stated.

Unknown values are never silently folded into a default. Each mapping yields ``None``
for a value it does not recognise, and the audit counts how often that happened, so a
new federation category shows up as a number rather than as an absence.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Final

from psd.ingest.openpowerlifting.contract import attempt_source_columns
from psd.schema.models import RECORD_FOURTH_ATTEMPT_NUMBER
from psd.schema.vocabulary import (
    AgePrecision,
    AttemptResult,
    AttemptRole,
    CompetitionEvent,
    EquipmentClass,
    ParticipationStatus,
    ReportedBestSemantics,
    SexCategory,
)

__all__ = (
    "AGE_APPROXIMATION_EPSILON",
    "ATTEMPT_SPECS",
    "EQUIPMENT_CLASS_BY_SOURCE",
    "EVENT_BY_SOURCE",
    "PLACE_KIND_BY_SOURCE",
    "REPORTED_RESULT_SPECS",
    "SANCTIONED_BY_SOURCE",
    "SEX_CATEGORY_BY_SOURCE",
    "age_precision",
    "attempt_result_for",
    "participation_kind",
    "participation_place",
    "reported_best_semantics",
    "sanctioned_flag",
    "tested_category_flag",
)

#: ``Age`` values are half-integers when approximate. The comparison is against a
#: tolerance because the source publishes two decimal places and floating point does
#: not respect that intent exactly.
AGE_APPROXIMATION_EPSILON: Final[float] = 1e-9

#: ``Sex`` as OpenPowerlifting publishes it, mapped onto PSD's vocabulary. ``Mx`` is the
#: source's catch-all neutral title; PSD records it as a self-described category and
#: does *not* reinterpret it as intersex, which is a different fact about a person.
SEX_CATEGORY_BY_SOURCE: Final[Mapping[str, SexCategory]] = {
    "M": SexCategory.MALE,
    "F": SexCategory.FEMALE,
    "Mx": SexCategory.OTHER_SELF_DESCRIBED,
}

#: ``Event`` is a declared competition event: which lifts were contested at all.
EVENT_BY_SOURCE: Final[Mapping[str, CompetitionEvent]] = {
    "SBD": CompetitionEvent.SQUAT_BENCH_DEADLIFT,
    "BD": CompetitionEvent.BENCH_DEADLIFT,
    "SD": CompetitionEvent.SQUAT_DEADLIFT,
    "SB": CompetitionEvent.SQUAT_BENCH,
    "S": CompetitionEvent.SQUAT,
    "B": CompetitionEvent.BENCH,
    "D": CompetitionEvent.DEADLIFT,
}

#: ``Equipment`` is the competition category, i.e. what the rules allowed.
EQUIPMENT_CLASS_BY_SOURCE: Final[Mapping[str, EquipmentClass]] = {
    "Raw": EquipmentClass.RAW,
    "Wraps": EquipmentClass.RAW_EQUIP,
    "Single-ply": EquipmentClass.CLASSIC_POWERLIFTING,
    "Multi-ply": EquipmentClass.MULTI_PLY,
    "Unlimited": EquipmentClass.EQUIPPED_UNLIMITED,
    "Straps": EquipmentClass.STRAPS_ALLOWED,
}

#: Non-numeric ``Place`` codes. Everything else numeric is a placing.
PLACE_KIND_BY_SOURCE: Final[Mapping[str, ParticipationStatus]] = {
    "G": ParticipationStatus.GUEST,
    "DQ": ParticipationStatus.DISQUALIFIED,
    "DD": ParticipationStatus.DRUG_DISQUALIFIED,
    "NS": ParticipationStatus.NO_SHOW,
}

#: ``Sanctioned`` is the source's competition status, not a quality score.
SANCTIONED_BY_SOURCE: Final[Mapping[str, bool]] = {"Yes": True, "No": False}

#: ``(lift, source column, attempt number)`` for the twelve attempt columns, plus the
#: role each number carries. Attempt four is a record attempt: it contributes to no
#: total, and PSD labels it rather than letting it be read as a third.
ATTEMPT_SPECS: Final[tuple[tuple[str, str, int, AttemptRole], ...]] = tuple(
    (
        lift,
        column,
        number,
        AttemptRole.RECORD_FOURTH
        if number == RECORD_FOURTH_ATTEMPT_NUMBER
        else AttemptRole.ORDERED,
    )
    for lift, column, number in attempt_source_columns()
)

#: ``(source column, result kind, carries best-lift semantics)`` for the reported
#: results. ``Glossbrenner`` is a scoring system of its own and deliberately does not
#: share a kind with IPF GL points; the recorded source field is what tells them apart.
REPORTED_RESULT_SPECS: Final[tuple[tuple[str, str, bool], ...]] = (
    ("Best3SquatKg", "squat_best", True),
    ("Best3BenchKg", "bench_best", True),
    ("Best3DeadliftKg", "deadlift_best", True),
    ("TotalKg", "total", False),
    ("Dots", "dots", False),
    ("Wilks", "wilks", False),
    ("Glossbrenner", "other", False),
    ("Goodlift", "gl_points", False),
)


def attempt_result_for(source_value: float) -> AttemptResult:
    """Return the outcome a signed source attempt value encodes.

    The sign *is* the result in this source's convention, so a negative value is a
    failed attempt rather than a negative load.

    Args:
        source_value: The published attempt value, sign included.

    Returns:
        ``GOOD_LIFT`` above zero, ``BAD_LIFT`` below.

    Raises:
        ValueError: The value is zero, which is not a load at all.
    """
    if source_value > 0.0:
        return AttemptResult.GOOD_LIFT
    if source_value < 0.0:
        return AttemptResult.BAD_LIFT
    msg = f"Attempt value {source_value!r} is not a load; a source reports no attempt by omission."
    raise ValueError(msg)


def age_precision(value: float) -> AgePrecision:
    """Return whether a reported age is exact or an approximation.

    A whole number is the source's exact age. A fractional one is the source's
    approximation, which happens because some federations publish only a birth year:
    ``23.5`` means 23 or 24, and rounding it would pick one.

    Args:
        value: The published age.

    Returns:
        ``EXACT`` for a whole number, ``APPROXIMATE`` otherwise.
    """
    return (
        AgePrecision.EXACT
        if abs(value - round(value)) <= AGE_APPROXIMATION_EPSILON
        else AgePrecision.APPROXIMATE
    )


def participation_kind(source_value: str) -> ParticipationStatus:
    """Return the participation status a ``Place`` code states.

    Args:
        source_value: The published ``Place`` value.

    Returns:
        The matching status. A positive integer is ``PLACED``; a code the source
        documents maps to its own status; anything unrecognised is ``UNKNOWN`` rather
        than a default that would look like a decision.
    """
    if source_value.isdigit():
        return ParticipationStatus.PLACED
    return PLACE_KIND_BY_SOURCE.get(source_value, ParticipationStatus.UNKNOWN)


def participation_place(source_value: str, kind: ParticipationStatus) -> int | None:
    """Return the numeric placing, only when there is one.

    Args:
        source_value: The published ``Place`` value.
        kind: The status :func:`participation_kind` assigned.

    Returns:
        The placing as an integer, or ``None`` for guest, disqualified,
        doping-disqualified, and no-show codes.
    """
    if kind is not ParticipationStatus.PLACED or not source_value.isdigit():
        return None
    return int(source_value)


def reported_best_semantics(source_value: float) -> ReportedBestSemantics:
    """Return the meaning a signed reported best carries.

    A handful of federations publish a negative best to mean "the lowest weight this
    lifter attempted and failed". That is a real, reportable fact about the meet and
    not a negative lift, so it is labelled rather than reinterpreted.

    Args:
        source_value: The published value, sign included.

    Returns:
        ``FAILED_ATTEMPT_ONLY`` below zero, ``SUCCESSFUL_BEST`` above.
    """
    if source_value < 0.0:
        return ReportedBestSemantics.FAILED_ATTEMPT_ONLY
    return ReportedBestSemantics.SUCCESSFUL_BEST


def sanctioned_flag(source_value: str) -> bool | None:
    """Return whether the source counts the meet as sanctioned.

    Args:
        source_value: The published ``Sanctioned`` value.

    Returns:
        The parsed flag, or ``None`` when the source said nothing.
    """
    return SANCTIONED_BY_SOURCE.get(source_value.strip())


def tested_category_flag(source_value: str) -> bool | None:
    """Return whether the result belongs to a drug-tested competition category.

    This says nothing about whether the individual was tested: federations do not
    publish which lifters were subject to testing, so the flag is a property of the
    category and never becomes an athlete-level claim.

    Args:
        source_value: The published ``Tested`` value.

    Returns:
        ``True`` only for the source's explicit affirmative marker; ``None`` otherwise.
    """
    return True if source_value.strip() == "Yes" else None


def unknown_values(mapping: Mapping[str, object], observed: Sequence[str]) -> tuple[str, ...]:
    """Return the observed values a declared mapping does not cover.

    Used by the audit so a new federation category is reported as a number rather than
    silently disappearing into a default.

    Args:
        mapping: The declared source-to-canonical mapping.
        observed: Distinct source values seen in the corpus.

    Returns:
        The unrecognised values, sorted.
    """
    return tuple(sorted(value for value in set(observed) if value.strip() not in mapping))
