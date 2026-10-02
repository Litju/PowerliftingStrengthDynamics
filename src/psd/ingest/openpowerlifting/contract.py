"""The OpenPowerlifting bulk-CSV schema contract.

PSD reads the official bulk CSV through an explicitly declared contract rather than
by whatever columns happen to be present. Every source column is classified into
exactly one of four states, and the four states are the point:

``mapped``
    PSD has a canonical destination for the column and reads it.
``preserved``
    PSD keeps the column but derives nothing from it. Free text such as a division
    label is kept verbatim because it is context, and never mined for structured
    facts.
``ignored``
    PSD deliberately drops the column, and records *why*. An ignored column is a
    decision with a stated reason, not an oversight.
``unknown``
    The column is not in the contract. This is schema drift, and it is an error.

Drift fails loudly. A new, renamed, or removed column would otherwise be read with
the wrong dtype, mapped to the wrong field, or silently discarded -- and a silently
discarded column in a multi-million-row competition corpus is a silent change to
the scientific object. A dropped column would be worse still: the transform would
run and produce an artifact that claims to be the same corpus while quietly
missing a field.

The contract is a *declaration of intent against the documented source*, not a
record of what one particular download happened to contain. Both are needed: the
contract says what PSD expects, and :func:`review_source_schema` compares it against
what the pinned snapshot actually has.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Final

from psd.schema.vocabulary import CompetitionResultKind

__all__ = (
    "SOURCE_COLUMNS",
    "SourceColumnDisposition",
    "SourceColumnSpec",
    "SourceSchemaError",
    "SourceSchemaReview",
    "read_source_header",
    "review_source_schema",
)


class SourceColumnDisposition(StrEnum):
    """What PSD does with a declared source column."""

    #: Read into a canonical field.
    MAPPED = "mapped"
    #: Kept verbatim on a canonical row, with nothing derived from it.
    PRESERVED = "preserved"
    #: Deliberately dropped, with a stated reason.
    IGNORED = "ignored"
    #: Not in the contract at all: schema drift.
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class SourceColumnSpec:
    """One declared source column.

    Attributes:
        name: Exact column name as published by the source.
        disposition: What PSD does with it.
        destination: Canonical column the value is read into, for mapped and
            preserved columns.
        reason: Why the column is ignored, for ignored columns.
        note: Why the mapping is what it is. Recorded rather than assumed, because
            each of these columns encodes a specific published semantic.
    """

    name: str
    disposition: SourceColumnDisposition
    destination: str | None = None
    reason: str | None = None
    note: str | None = None


def _spec(
    name: str,
    disposition: SourceColumnDisposition,
    destination: str | None = None,
    *,
    reason: str | None = None,
    note: str | None = None,
) -> SourceColumnSpec:
    return SourceColumnSpec(
        name=name, disposition=disposition, destination=destination, reason=reason, note=note
    )


#: The exact header of the OpenPowerlifting bulk CSV, in published order, with every
#: column's disposition. Order is significant: the snapshot records this header, and
#: a reorder is drift even when the column set is unchanged, because it changes what
#: "row N" means for source-record keys.
SOURCE_COLUMNS: Final[tuple[SourceColumnSpec, ...]] = (
    _spec(
        "Name",
        SourceColumnDisposition.MAPPED,
        "athlete.athlete_id",
        note=(
            "Mandatory. Lifters sharing a name are disambiguated by the source with a "
            "'#N' suffix. The value is preserved verbatim, suffix included: the suffix is "
            "the source's own assertion that two people are distinct, and stripping it "
            "would merge them. The athlete identity is derived from this exact string."
        ),
    ),
    _spec(
        "Sex",
        SourceColumnDisposition.MAPPED,
        "athlete.sex_category",
        note=(
            "Mandatory. M, F, or Mx. Mapped losslessly onto PSD's vocabulary and kept "
            "verbatim in sex_category_raw. Mx is the source's catch-all neutral title and "
            "is not reinterpreted as intersex or as a physiological claim."
        ),
    ),
    _spec(
        "Event",
        SourceColumnDisposition.MAPPED,
        "competition.competition_event",
        note=(
            "Mandatory. SBD, BD, SD, SB, S, B, or D. A declared event type: which lifts "
            "were contested at all. A lift outside the event was never attempted, which is "
            "a different fact from attempting it and failing."
        ),
    ),
    _spec(
        "Equipment",
        SourceColumnDisposition.MAPPED,
        "competition.equipment_class",
        note=(
            "Mandatory. The competition equipment category, i.e. what the rules allowed. "
            "It is not proof the athlete wore any item: a federation with no wrap-free "
            "category puts every lifter in the wrap category regardless of what they wore. "
            "No athlete-level equipment observation is derived from it."
        ),
    ),
    _spec(
        "Age",
        SourceColumnDisposition.MAPPED,
        "competition.age_reported",
        note=(
            "Optional. An integer is exact; a half-integer is the source's approximation "
            "for a lifter whose age is known only to lie between two years. The precision "
            "is recorded beside the value and the value is never rounded."
        ),
    ),
    _spec(
        "AgeClass",
        SourceColumnDisposition.PRESERVED,
        "competition.age_class_raw",
        note=(
            "Optional. The source's explicit age-class bucket. Kept as its own field and "
            "never merged with BirthYearClass, which is a different bucket with different "
            "endpoints."
        ),
    ),
    _spec(
        "BirthYearClass",
        SourceColumnDisposition.PRESERVED,
        "competition.birth_year_class_raw",
        note=(
            "Optional. A birth-year bucket, not an age bucket. It does not identify a birth "
            "year, so no birth_year is derived from it for the athlete."
        ),
    ),
    _spec(
        "Division",
        SourceColumnDisposition.PRESERVED,
        "competition.division_raw",
        note=(
            "Optional free-form text, sometimes carrying configuration hints. Kept "
            "verbatim and deliberately not parsed: dedicated age fields exist, and mining "
            "an age out of a division label would be guessing."
        ),
    ),
    _spec(
        "BodyweightKg",
        SourceColumnDisposition.MAPPED,
        "competition.bodyweight_kg",
        note=(
            "Optional. The weigh-in, recorded on the competition row. PSD does not also emit "
            "a body_measurement row for it: one observation gets one canonical record."
        ),
    ),
    _spec(
        "WeightClassKg",
        SourceColumnDisposition.PRESERVED,
        "competition.weight_class_raw",
        note=(
            "Optional. Kept verbatim, including an open-ended form such as '90+'. Collapsing "
            "an open-ended class to the number beside it would claim a maximum the source "
            "did not state."
        ),
    ),
    # Attempts and reported bests, grouped by lift exactly as the source publishes them.
    _spec(
        "Squat1Kg",
        SourceColumnDisposition.MAPPED,
        "competition_attempt.load_kg",
        note="Optional. Negative means a failed attempt; the sign is the result encoding.",
    ),
    _spec("Squat2Kg", SourceColumnDisposition.MAPPED, "competition_attempt.load_kg"),
    _spec("Squat3Kg", SourceColumnDisposition.MAPPED, "competition_attempt.load_kg"),
    _spec(
        "Squat4Kg",
        SourceColumnDisposition.MAPPED,
        "competition_attempt.load_kg",
        note=(
            "Optional. A fourth attempt exists only for a single-lift record and does not "
            "contribute to TotalKg. It is stored with attempt_role=record_fourth so it "
            "cannot be mistaken for one of the three counted attempts."
        ),
    ),
    _spec(
        "Best3SquatKg",
        SourceColumnDisposition.MAPPED,
        "competition_reported_result.value",
        note=(
            "Optional. A source-reported outcome, not an estimate of latent capacity. Rarely "
            "negative, which means the source is publishing the lowest weight attempted and "
            "failed; that meaning is recorded explicitly rather than read as a negative lift."
        ),
    ),
    _spec("Bench1Kg", SourceColumnDisposition.MAPPED, "competition_attempt.load_kg"),
    _spec("Bench2Kg", SourceColumnDisposition.MAPPED, "competition_attempt.load_kg"),
    _spec("Bench3Kg", SourceColumnDisposition.MAPPED, "competition_attempt.load_kg"),
    _spec(
        "Bench4Kg",
        SourceColumnDisposition.MAPPED,
        "competition_attempt.load_kg",
        note="Optional. A record attempt; contributes to no total.",
    ),
    _spec("Best3BenchKg", SourceColumnDisposition.MAPPED, "competition_reported_result.value"),
    _spec("Deadlift1Kg", SourceColumnDisposition.MAPPED, "competition_attempt.load_kg"),
    _spec("Deadlift2Kg", SourceColumnDisposition.MAPPED, "competition_attempt.load_kg"),
    _spec("Deadlift3Kg", SourceColumnDisposition.MAPPED, "competition_attempt.load_kg"),
    _spec(
        "Deadlift4Kg",
        SourceColumnDisposition.MAPPED,
        "competition_attempt.load_kg",
        note="Optional. A record attempt; contributes to no total.",
    ),
    _spec("Best3DeadliftKg", SourceColumnDisposition.MAPPED, "competition_reported_result.value"),
    _spec(
        "TotalKg",
        SourceColumnDisposition.MAPPED,
        "competition_reported_result.value",
        note=(
            "Optional. Recorded exactly as reported. A source may publish a total with no "
            "component lifts at all, and PSD never manufactures the components to balance it."
        ),
    ),
    _spec(
        "Place",
        SourceColumnDisposition.MAPPED,
        "competition.participation_status",
        note=(
            "Mandatory. A positive number, or one of G, DQ, DD, NS. Those are guest, "
            "disqualified, doping-disqualified, and no-show -- not placings, and never "
            "coerced into numbers."
        ),
    ),
    _spec(
        "Dots",
        SourceColumnDisposition.MAPPED,
        "competition_reported_result.value",
        note="Optional. A scoring-system score, empty when it could not be calculated.",
    ),
    _spec("Wilks", SourceColumnDisposition.MAPPED, "competition_reported_result.value"),
    _spec(
        "Glossbrenner",
        SourceColumnDisposition.MAPPED,
        "competition_reported_result.value",
        note=(
            "Optional. A scoring system distinct from IPF GL points; the recorded source "
            "field is what distinguishes the two, so they are not merged."
        ),
    ),
    _spec(
        "Goodlift",
        SourceColumnDisposition.MAPPED,
        "competition_reported_result.value",
        note="Optional. IPF GL points.",
    ),
    _spec(
        "Tested",
        SourceColumnDisposition.MAPPED,
        "competition.is_drug_tested_category",
        note=(
            "Optional. 'Yes' means the result belongs to a drug-tested competition category. "
            "It does not establish that the individual was tested -- federations do not "
            "publish which lifters were -- so it is recorded as a property of the category "
            "and never promoted to an athlete-level flag."
        ),
    ),
    _spec(
        "Country",
        SourceColumnDisposition.MAPPED,
        "competition.athlete_country_raw",
        note=(
            "Optional. The lifter's own country as reported for this result. Kept verbatim "
            "per result row: the source states it per result, so PSD does not invent a single "
            "modal country for the athlete across meets where it varies."
        ),
    ),
    _spec(
        "State",
        SourceColumnDisposition.MAPPED,
        "competition.athlete_region_raw",
        note="Optional. The lifter's own state or region as reported for this result.",
    ),
    _spec(
        "Federation",
        SourceColumnDisposition.MAPPED,
        "competition.federation",
        note=(
            "Mandatory. The federation that hosted the meet, which is frequently an "
            "affiliate rather than the sanctioning body. Never collapsed with "
            "ParentFederation."
        ),
    ),
    _spec(
        "ParentFederation",
        SourceColumnDisposition.MAPPED,
        "competition.sanctioning_body",
        note=(
            "Optional. The top-level sanctioning body when the source states one. A distinct "
            "field from Federation, not a fallback for it."
        ),
    ),
    _spec(
        "Date",
        SourceColumnDisposition.MAPPED,
        "competition.competition_date",
        note=(
            "Mandatory. The meet's START date. A multi-day meet publishes only that, so the "
            "date is recorded with date-only precision and nothing claims to know when a "
            "particular lifter competed within the meet."
        ),
    ),
    _spec("MeetCountry", SourceColumnDisposition.MAPPED, "competition_meet.meet_country"),
    _spec("MeetState", SourceColumnDisposition.MAPPED, "competition_meet.meet_state"),
    _spec(
        "MeetTown",
        SourceColumnDisposition.PRESERVED,
        "competition_meet.meet_town",
        note=(
            "The town or city the meet was held in. Kept as its own column rather than "
            "concatenated into a location string: PSD stores what the source states and "
            "does not synthesise a formatted address."
        ),
    ),
    _spec(
        "MeetName",
        SourceColumnDisposition.MAPPED,
        "competition_meet.meet_name",
        note=(
            "Mandatory. By the source's own convention the name excludes the year and the "
            "federation, so it is never unique on its own and is never used alone as a meet "
            "identity."
        ),
    ),
    _spec(
        "Sanctioned",
        SourceColumnDisposition.MAPPED,
        "competition_meet.is_sanctioned",
        note=(
            "Optional. Whether the source counts the meet as sanctioned by a body it "
            "recognises. A competition status, not a quality score, and kept beside the "
            "source's own wording."
        ),
    ),
)

_COLUMN_BY_NAME: Final[Mapping[str, SourceColumnSpec]] = {
    spec.name: spec for spec in SOURCE_COLUMNS
}


def _column_names() -> tuple[str, ...]:
    return tuple(spec.name for spec in SOURCE_COLUMNS)


class SourceSchemaError(ValueError):
    """Raised when a source's columns do not match the declared contract."""


@dataclass(frozen=True, slots=True)
class SourceSchemaReview:
    """The result of comparing a snapshot's header against the contract.

    Attributes:
        declared: The contract's columns, in order.
        observed: The snapshot's columns, in order.
        mapped: Declared columns PSD reads.
        preserved: Declared columns PSD keeps without deriving from them.
        ignored: Declared columns PSD drops, with the reason for each.
        unknown: Columns present in the snapshot but absent from the contract.
        missing: Contract columns absent from the snapshot.
        reordered: Whether the same column set appears in a different order.
    """

    declared: tuple[str, ...]
    observed: tuple[str, ...]
    mapped: tuple[str, ...]
    preserved: tuple[str, ...]
    ignored: Mapping[str, str]
    unknown: tuple[str, ...]
    missing: tuple[str, ...]
    reordered: bool

    @property
    def drifted(self) -> bool:
        """Whether the snapshot differs from the contract in any way."""
        return bool(self.unknown or self.missing or self.reordered)

    def summary(self) -> str:
        """Return a one-line rendering suitable for CLI and report output."""
        parts = [
            f"{len(self.declared)} declared",
            f"{len(self.observed)} observed",
            f"{len(self.mapped)} mapped",
            f"{len(self.preserved)} preserved",
            f"{len(self.ignored)} ignored",
        ]
        if self.drifted:
            parts.append(
                f"DRIFT unknown={list(self.unknown)} missing={list(self.missing)} "
                f"reordered={self.reordered}"
            )
        return "; ".join(parts)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable rendering of the review."""
        return {
            "declared": list(self.declared),
            "observed": list(self.observed),
            "mapped": list(self.mapped),
            "preserved": list(self.preserved),
            "ignored": dict(sorted(self.ignored.items())),
            "unknown": list(self.unknown),
            "missing": list(self.missing),
            "reordered": self.reordered,
            "drifted": self.drifted,
        }


def review_source_schema(header: Sequence[str]) -> SourceSchemaReview:
    """Compare a snapshot's header against the declared contract.

    Args:
        header: Column names read from the snapshot, in file order.

    Returns:
        The full review, including every classification. Nothing is rejected here, so
        a caller can report drift rather than discover it from an exception.
    """
    observed = tuple(header)
    mapped: list[str] = []
    preserved: list[str] = []
    ignored: dict[str, str] = {}
    for spec in SOURCE_COLUMNS:
        if spec.name not in observed:
            continue
        if spec.disposition is SourceColumnDisposition.MAPPED:
            mapped.append(spec.name)
        elif spec.disposition is SourceColumnDisposition.PRESERVED:
            preserved.append(spec.name)
        else:
            ignored[spec.name] = spec.reason or "declared ignored with no stated reason"
    unknown = tuple(name for name in observed if name not in _COLUMN_BY_NAME)
    declared = _column_names()
    missing = tuple(name for name in declared if name not in set(observed))
    present = [name for name in observed if name in _COLUMN_BY_NAME]
    reordered = present != [name for name in declared if name in set(present)]
    return SourceSchemaReview(
        declared=declared,
        observed=observed,
        mapped=tuple(mapped),
        preserved=tuple(preserved),
        ignored=ignored,
        unknown=unknown,
        missing=missing,
        reordered=reordered,
    )


def require_source_schema(
    header: Sequence[str],
    *,
    allow_unknown: Sequence[str] = (),
) -> SourceSchemaReview:
    """Compare the header against the contract and refuse drift.

    Args:
        header: Column names read from the snapshot, in file order.
        allow_unknown: Unknown column names the caller has explicitly reviewed and
            accepted. Names outside this list still fail. An explicit opt-in is the
            only way through, because the alternative -- ignoring a new column --
            changes the scientific object without saying so.

    Returns:
        The review, which is drift-free apart from the explicitly accepted columns.

    Raises:
        SourceSchemaError: The snapshot has unknown, missing, or reordered columns.
    """
    review = review_source_schema(header)
    tolerated = set(allow_unknown)
    unexpected = sorted(set(review.unknown) - tolerated)
    problems: list[str] = []
    if unexpected:
        problems.append(
            f"unknown source column(s): {', '.join(unexpected)}. The declared contract is "
            f"{len(review.declared)} columns; a new, renamed, or misspelled column would be "
            "read with the wrong meaning or dropped without notice"
        )
    if review.missing:
        problems.append(
            f"missing source column(s): {', '.join(review.missing)}. The transform would run "
            "and produce an artifact that is not the corpus the contract declares"
        )
    if review.reordered:
        problems.append(
            "the same columns appear in a different order. Row ordinals are used as "
            "source-record keys, so a reorder changes what row N means"
        )
    if problems:
        raise SourceSchemaError(
            "OpenPowerlifting source schema does not match the declared contract. "
            + "; ".join(problems)
            + ". Review the snapshot and either update the contract or pass the reviewed "
            "column names explicitly."
        )
    return review


def read_source_header(csv_path: Path) -> tuple[str, ...]:
    """Return a CSV's header without reading its rows.

    The header is read from the file itself rather than taken from a constant,
    because the whole purpose of the contract is to detect the case where the two
    disagree.

    Args:
        csv_path: Path to the CSV.

    Returns:
        The column names, in file order.

    Raises:
        SourceSchemaError: The file is missing, empty, or has no header row.
    """
    if not csv_path.is_file():
        msg = f"No OpenPowerlifting CSV at {csv_path}."
        raise SourceSchemaError(msg)
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        header = next(csv.reader(handle), None)
    if not header:
        msg = f"{csv_path} has no header row."
        raise SourceSchemaError(msg)
    return tuple(header)


def expected_columns() -> tuple[str, ...]:
    """Return the contract's column names, in published order."""
    return _column_names()


def column_spec(name: str) -> SourceColumnSpec:
    """Return the declared spec for a source column.

    Raises:
        KeyError: The column is not in the contract.
    """
    spec = _COLUMN_BY_NAME.get(name)
    if spec is None:
        msg = f"Unknown OpenPowerlifting source column {name!r}."
        raise KeyError(msg)
    return spec


def attempt_source_columns() -> tuple[tuple[str, str, int], ...]:
    """Return ``(lift, column, attempt_number)`` for the twelve attempt columns.

    Attempt number 4 is the record attempt: it is returned here so the transform can
    label it, and only so it can be labelled. It is never counted toward a total.
    """
    mapping: tuple[tuple[str, str, int], ...] = (
        ("squat", "Squat1Kg", 1),
        ("bench", "Bench1Kg", 1),
        ("deadlift", "Deadlift1Kg", 1),
        ("squat", "Squat2Kg", 2),
        ("bench", "Bench2Kg", 2),
        ("deadlift", "Deadlift2Kg", 2),
        ("squat", "Squat3Kg", 3),
        ("bench", "Bench3Kg", 3),
        ("deadlift", "Deadlift3Kg", 3),
        ("squat", "Squat4Kg", 4),
        ("bench", "Bench4Kg", 4),
        ("deadlift", "Deadlift4Kg", 4),
    )
    return mapping


def reported_result_source_columns() -> tuple[tuple[str, str], ...]:
    """Return ``(column, result_kind)`` for the reported result columns."""
    pairs: tuple[tuple[str, str], ...] = (
        ("Best3SquatKg", CompetitionResultKind.SQUAT_BEST.value),
        ("Best3BenchKg", CompetitionResultKind.BENCH_BEST.value),
        ("Best3DeadliftKg", CompetitionResultKind.DEADLIFT_BEST.value),
        ("TotalKg", CompetitionResultKind.TOTAL.value),
        ("Dots", CompetitionResultKind.DOTS.value),
        ("Wilks", CompetitionResultKind.WILKS.value),
        ("Glossbrenner", CompetitionResultKind.OTHER.value),
        ("Goodlift", CompetitionResultKind.GL_POINTS.value),
    )
    return pairs


def source_columns_for_report() -> Iterable[SourceColumnSpec]:
    """Iterate the declared columns, for contract reports."""
    return SOURCE_COLUMNS
