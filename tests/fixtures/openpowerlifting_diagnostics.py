"""Deterministic source fixtures for the four closure diagnostics.

The general-purpose OpenPowerlifting fixture is one meet, one entry per lifter, and a
perfectly well-formed corpus. That is the right shape for testing the transform, and the
wrong shape for testing diagnostics: an audit that reports zero for everything looks
identical whether the checks are correct or vacuous. So these fixtures are written to make
each diagnostic report a *positive* count, and to place a zero-count case beside it.

One corpus per concern, each built from named row sets:

``identity_rows``
    A base name published bare *and* as two ``#N`` variants, plus a name the source
    reports under two sex categories. Produces a base-name collision group, a
    ``base_names_also_published_unsuffixed`` finding, and a sex conflict.
``future_dated_rows``
    Meets dated after the pinned snapshot date, checked against a snapshot whose
    service-reported date is deliberately earlier than the meets.
``age_consistent_rows`` / ``age_inconsistent_rows``
    One lifter whose ages over four seasons are mutually compatible and one whose are
    not. The compatible set is computed by the audit, so these two rows are what prove
    the birth-year intersection is really being intersected.
``transition_rows``
    A lifter whose equipment and federation change twice, a lifter whose single meet
    contradicts itself across two event entries, and a lifter whose equipment is absent
    at one meet -- the case where a naive implementation would report a change.

Every row is written from a mapping, so a row always has exactly the contract's column
count and a drift fails with "too many fields" rather than silently shifting every
subsequent column.
"""

from __future__ import annotations

import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import polars as pl

from psd.ingest.openpowerlifting.acquire import (
    acquire_snapshot_from_local_file,
    resolve_snapshot_csv,
)
from psd.ingest.openpowerlifting.audit import AuditRequest, audit_corpus
from psd.ingest.openpowerlifting.contract import expected_columns
from psd.ingest.openpowerlifting.snapshot import ServiceSnapshotFacts
from psd.ingest.openpowerlifting.transform import BuildConfig, BuildRequest, build_corpus
from psd.schema.registry import table_names
from psd.serialization.dataset import read_manifest
from psd.serialization.parquet import read_parquet, write_parquet

if TYPE_CHECKING:
    from psd.ingest.openpowerlifting.audit import CorpusAudit

__all__ = (
    "DIAGNOSTIC_BUILD_CONFIG",
    "DIAGNOSTIC_CSV_NAME",
    "STAMP",
    "DiagnosticCorpus",
    "age_consistent_rows",
    "age_inconsistent_rows",
    "audit_rows",
    "build_diagnostic_corpus",
    "copy_corpus_with_modified_table",
    "csv_text",
    "future_dated_rows",
    "identity_rows",
    "transition_rows",
)

DIAGNOSTIC_CSV_NAME = "openpowerlifting-diagnostics.csv"

#: Fixed clock for every derived artifact, so two runs of the same fixture produce the
#: same bytes and a diff means something changed.
STAMP = datetime(2026, 10, 2, tzinfo=UTC)

#: Small partitions and batches on purpose: the fixtures are a handful of rows, and the
#: smallest legal configuration exercises the same code path the full corpus does.
DIAGNOSTIC_BUILD_CONFIG = BuildConfig(chunk_rows=3, batch_rows=4, partitions=2)

#: The default meet a row inherits unless it overrides the field. Distinct meets are made
#: by changing the date and the meet name, which is what the six-field meet rule keys on.
DEFAULT_MEET: dict[str, str] = {
    "Date": "2025-11-08",
    "Federation": "USPA",
    "MeetCountry": "USA",
    "MeetState": "TX",
    "MeetTown": "Dallas",
    "MeetName": "Diagnostics Open",
    "Sanctioned": "Yes",
    "Sex": "M",
    "Event": "SBD",
    "Equipment": "Raw",
    "Division": "Open",
    "Country": "USA",
    "State": "TX",
    "ParentFederation": "IPF",
}


def csv_text(rows: Sequence[dict[str, Any]]) -> str:
    """Return the fixture CSV's full text for *rows*, header included."""
    columns = expected_columns()
    lines = [",".join(columns)]
    for row in rows:
        values = dict(DEFAULT_MEET)
        values.update(row)
        lines.append(",".join(str(values.get(column, "")) for column in columns))
    return "\n".join(lines) + "\n"


def identity_rows() -> tuple[dict[str, Any], ...]:
    """Rows that exercise the identity diagnostic's positive cases.

    ``Ada Liftwell`` is published bare *and* as ``#1`` and ``#2``. The bare key is a real
    ambiguity rather than a curiosity: it is one identity the source may or may not have
    merged with a numbered variant, and PSD cannot tell, so the diagnostic reports the
    collision group rather than resolving it. ``Hal Twofold`` is one name key the source
    reports under two sex categories.
    """
    return (
        {
            "Name": "Ada Liftwell",
            "Sex": "F",
            "Age": "29",
            "BodyweightKg": "63.0",
            "WeightClassKg": "-69",
            "Squat1Kg": "120",
            "Best3SquatKg": "120",
            "Place": "1",
        },
        {
            "Name": "Ada Liftwell#1",
            "Sex": "F",
            "Age": "29",
            "BodyweightKg": "64.0",
            "WeightClassKg": "-69",
            "Squat1Kg": "125",
            "Best3SquatKg": "125",
            "Place": "2",
        },
        {
            "Name": "Ada Liftwell#2",
            "Sex": "F",
            "Age": "30",
            "BodyweightKg": "65.0",
            "WeightClassKg": "-69",
            "Squat1Kg": "130",
            "Best3SquatKg": "130",
            "Place": "3",
        },
        {
            "Name": "Hal Twofold",
            "Sex": "F",
            "Age": "31",
            "BodyweightKg": "70.0",
            "WeightClassKg": "-74",
            "Squat1Kg": "100",
            "Best3SquatKg": "100",
            "Place": "1",
        },
        {
            "Name": "Hal Twofold",
            "Sex": "M",
            "Age": "31",
            "BodyweightKg": "80.0",
            "WeightClassKg": "-83",
            "Squat1Kg": "150",
            "Best3SquatKg": "150",
            "Place": "2",
        },
    )


def future_dated_rows() -> tuple[dict[str, Any], ...]:
    """Rows dated after the pinned snapshot date used by the future-dating scenario.

    The snapshot for this scenario is pinned with a service-reported date of
    :data:`EARLY_SNAPSHOT_DATE`, so every meet here is in the source's future and the
    diagnostic must say so rather than reporting a clean chronology.
    """
    return (
        {
            "Name": "Ivy Tomorrow",
            "Age": "25",
            "BodyweightKg": "70.0",
            "WeightClassKg": "-74",
            "Squat1Kg": "140",
            "Best3SquatKg": "140",
            "Place": "1",
        },
        {
            "Name": "Jan Laterstill",
            "Age": "26",
            "BodyweightKg": "72.0",
            "WeightClassKg": "-74",
            "Squat1Kg": "145",
            "Best3SquatKg": "145",
            "Place": "1",
            "Date": "2026-02-14",
            "MeetName": "Diagnostics Spring",
            "MeetTown": "Austin",
        },
    )


#: The service-reported snapshot date the future-dating scenario is measured against.
EARLY_SNAPSHOT_DATE = "2025-06-01"

#: The season the ageing fixtures start in.
_FIRST_SEASON = {"Date": "2020-03-07", "MeetName": "Diagnostics Spring 2020", "MeetTown": "Austin"}

#: Four seasons later. Fixed rather than computed so a reader can check the arithmetic.
_LAST_SEASON = {
    "Date": "2024-03-09",
    "MeetName": "Diagnostics Spring 2024",
    "MeetTown": "Austin",
}


def age_consistent_rows() -> tuple[dict[str, Any], ...]:
    """Rows whose ages are mutually compatible across seasons.

    ``Ada Aging`` is 40 in 2020 and 44 in 2024, which admits birth years 1980/1979 at both
    ends. ``Eve Midpoint`` is reported approximately -- 40.5 then 44.5 -- which under the
    source's documented convention implies exactly one birth year each, also 1979. If the
    intersection were wrong these two would be reported alongside
    :func:`age_inconsistent_rows` rather than silently agreeing.
    """
    return (
        {
            "Name": "Ada Aging",
            "Age": "40",
            "BodyweightKg": "80.0",
            "WeightClassKg": "-83",
            "Squat1Kg": "180",
            "Best3SquatKg": "180",
            "Place": "1",
            **_FIRST_SEASON,
        },
        {
            "Name": "Ada Aging",
            "Age": "44",
            "BodyweightKg": "82.0",
            "WeightClassKg": "-83",
            "Squat1Kg": "195",
            "Best3SquatKg": "195",
            "Place": "1",
            **_LAST_SEASON,
        },
        {
            "Name": "Eve Midpoint",
            "Age": "40.5",
            "BodyweightKg": "61.0",
            "WeightClassKg": "-63",
            "Squat1Kg": "110",
            "Best3SquatKg": "110",
            "Place": "1",
            **_FIRST_SEASON,
        },
        {
            "Name": "Eve Midpoint",
            "Age": "44.5",
            "BodyweightKg": "62.0",
            "WeightClassKg": "-63",
            "Squat1Kg": "120",
            "Best3SquatKg": "120",
            "Place": "1",
            **_LAST_SEASON,
        },
    )


def age_inconsistent_rows() -> tuple[dict[str, Any], ...]:
    """Rows whose ages cannot all describe the same person.

    ``Gil Year`` is reported as 40 in 2020 and again as 40 in 2024. Age 40 in 2020 admits
    birth years 1980 or 1979; age 40 in 2024 admits 1984 or 1983. The intersection is
    empty, which is a genuine finding about the source -- a transcription slip, or two
    different lifters sharing one name key -- and is reported rather than repaired.

    ``Hal Fifties`` is the approximate counterpart: 40.5 in 2020 implies 1979 exactly, and
    40.5 in 2024 implies 1983 exactly, so the two cannot be reconciled either.
    """
    return (
        {
            "Name": "Gil Year",
            "Age": "40",
            "BodyweightKg": "90.0",
            "WeightClassKg": "-93",
            "Squat1Kg": "200",
            "Best3SquatKg": "200",
            "Place": "1",
            **_FIRST_SEASON,
        },
        {
            "Name": "Gil Year",
            "Age": "40",
            "BodyweightKg": "92.0",
            "WeightClassKg": "-93",
            "Squat1Kg": "210",
            "Best3SquatKg": "210",
            "Place": "1",
            **_LAST_SEASON,
        },
        {
            "Name": "Hal Fifties",
            "Age": "40.5",
            "BodyweightKg": "85.0",
            "WeightClassKg": "-93",
            "Squat1Kg": "190",
            "Best3SquatKg": "190",
            "Place": "1",
            **_FIRST_SEASON,
        },
        {
            "Name": "Hal Fifties",
            "Age": "40.5",
            "BodyweightKg": "86.0",
            "WeightClassKg": "-93",
            "Squat1Kg": "195",
            "Best3SquatKg": "195",
            "Place": "1",
            **_LAST_SEASON,
        },
    )


def transition_rows() -> tuple[dict[str, Any], ...]:
    """Rows that exercise the transition diagnostic's positive and negative cases.

    ``Rae Switcher`` changes equipment twice and federation once across three meets, and
    the ordering matters: meets are ordered by date, so a transition count that came out
    in the wrong order would not match. ``Gil Contradiction`` enters two events at one
    meet and the source states a different equipment for each, which is an intra-meet
    conflict rather than a switch. ``Jan Silent`` states no equipment at the second meet,
    which is an absence, not a change.
    """
    first = {"Date": "2022-05-14", "MeetName": "Diagnostics May 2022", "MeetTown": "Houston"}
    second = {"Date": "2023-05-13", "MeetName": "Diagnostics May 2023", "MeetTown": "Houston"}
    third = {"Date": "2024-05-11", "MeetName": "Diagnostics May 2024", "MeetTown": "Houston"}
    return (
        {
            "Name": "Rae Switcher",
            "Age": "33",
            "BodyweightKg": "80.0",
            "WeightClassKg": "-83",
            "Equipment": "Raw",
            "Federation": "USPA",
            "Squat1Kg": "200",
            "Best3SquatKg": "200",
            "Place": "1",
            **first,
        },
        {
            "Name": "Rae Switcher",
            "Age": "34",
            "BodyweightKg": "81.0",
            "WeightClassKg": "-83",
            "Equipment": "Wraps",
            "Federation": "USPA",
            "Squat1Kg": "210",
            "Best3SquatKg": "210",
            "Place": "1",
            **second,
        },
        {
            "Name": "Rae Switcher",
            "Age": "35",
            "BodyweightKg": "82.0",
            "WeightClassKg": "-83",
            "Equipment": "Multi-ply",
            "Federation": "CPU",
            "Squat1Kg": "220",
            "Best3SquatKg": "220",
            "Place": "1",
            **third,
        },
        {
            "Name": "Gil Contradiction",
            "Age": "28",
            "BodyweightKg": "88.0",
            "WeightClassKg": "-93",
            "Equipment": "Raw",
            "Event": "SBD",
            "Squat1Kg": "220",
            "Best3SquatKg": "220",
            "Place": "1",
            "Date": "2023-11-04",
            "MeetName": "Diagnostics Nov 2023",
            "MeetTown": "Fresno",
        },
        {
            "Name": "Gil Contradiction",
            "Age": "28",
            "BodyweightKg": "88.0",
            "WeightClassKg": "-93",
            "Equipment": "Wraps",
            "Event": "B",
            "Bench1Kg": "140",
            "Best3BenchKg": "140",
            "Place": "1",
            "Date": "2023-11-04",
            "MeetName": "Diagnostics Nov 2023",
            "MeetTown": "Fresno",
        },
        {
            "Name": "Jan Silent",
            "Age": "27",
            "BodyweightKg": "75.0",
            "WeightClassKg": "-83",
            "Equipment": "Single-ply",
            "Squat1Kg": "180",
            "Best3SquatKg": "180",
            "Place": "1",
            "Date": "2022-03-12",
            "MeetName": "Diagnostics Mar 2022",
            "MeetTown": "Denver",
        },
        {
            "Name": "Jan Silent",
            "Age": "28",
            "BodyweightKg": "76.0",
            "WeightClassKg": "-83",
            "Equipment": "",
            "Squat1Kg": "185",
            "Best3SquatKg": "185",
            "Place": "1",
            "Date": "2023-03-11",
            "MeetName": "Diagnostics Mar 2023",
            "MeetTown": "Denver",
        },
    )


def audit_rows() -> tuple[dict[str, Any], ...]:
    """Every diagnostic row, which is the corpus the classification tests read."""
    return (
        *identity_rows(),
        *future_dated_rows(),
        *age_consistent_rows(),
        *age_inconsistent_rows(),
        *transition_rows(),
    )


@dataclass(frozen=True, slots=True)
class DiagnosticCorpus:
    """One fixture corpus, built and audited.

    Attributes:
        data_root: The external PSD data root holding it.
        relative: Where the corpus lives, relative to that root.
        audit: Its audit report.
        service: The service-reported snapshot facts it was pinned with.
    """

    data_root: Path
    relative: Path
    audit: CorpusAudit
    service: ServiceSnapshotFacts


def build_diagnostic_corpus(
    rows: Sequence[dict[str, Any]],
    *,
    root: Path,
    service: ServiceSnapshotFacts | None = None,
) -> DiagnosticCorpus:
    """Pin, build, and audit a fixture corpus.

    Args:
        rows: Source rows, in the order the CSV will hold them.
        root: Scratch directory. Removed first, so a rerun never inherits staging state
            from a previous one.
        service: Service-reported snapshot facts. Defaults to none, which leaves the
            snapshot date unavailable and is itself a case worth testing.

    Returns:
        The corpus and its report.
    """
    if root.exists():
        shutil.rmtree(root)
    data_root = root / "data"
    csv_path = root / "source" / DIAGNOSTIC_CSV_NAME
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.write_text(csv_text(rows), encoding="utf-8")
    facts = service if service is not None else ServiceSnapshotFacts()
    snapshot = acquire_snapshot_from_local_file(csv_path, data_root=data_root, service=facts)
    build_corpus(
        BuildRequest(
            csv_path=resolve_snapshot_csv(snapshot, data_root=data_root),
            snapshot=snapshot,
            data_root=data_root,
            config=DIAGNOSTIC_BUILD_CONFIG,
            ingested_at=STAMP,
        )
    )
    relative = Path("canonical") / "psd_comp" / snapshot.archive_sha256
    report = audit_corpus(
        AuditRequest(dataset_dir=relative, data_root=data_root, generated_at=STAMP)
    ).audit
    return DiagnosticCorpus(data_root=data_root, relative=relative, audit=report, service=facts)


def copy_corpus_with_modified_table(
    corpus: DiagnosticCorpus,
    destination: Path,
    *,
    table: str,
    modify: pl.Expr,
) -> Path:
    """Copy a fixture corpus into *destination*, rewriting exactly one table.

    A deliberately broken corpus is how a diagnostic proves it is not vacuous: the check
    must fail on data that is *almost* right, and it cannot do that on a corpus the
    transform already produced correctly. Every other table is copied byte-for-byte and
    the manifest is carried over unchanged, so nothing but the named column can be
    responsible for a finding.

    Args:
        corpus: The corpus to copy.
        destination: Directory to build the copy in; it is created.
        table: Canonical table to rewrite.
        modify: A Polars expression replacing a column.

    Returns:
        The copied corpus directory, relative to the *copy's* data root.
    """
    target = destination / "data" / "broken"
    (target / "tables").mkdir(parents=True)
    source_tables = corpus.data_root / corpus.relative / "tables"
    frame = pl.read_parquet(source_tables / f"{table}.parquet")
    payload = frame.with_columns(modify)
    for name in table_names():
        if name == table:
            schema = read_parquet(source_tables / f"{name}.parquet", table_name=name).schema
            write_parquet(
                payload.to_arrow().cast(schema),
                target / "tables" / f"{name}.parquet",
                table_name=name,
            )
        else:
            shutil.copyfile(
                source_tables / f"{name}.parquet", target / "tables" / f"{name}.parquet"
            )
    shutil.copyfile(corpus.data_root / corpus.relative / "manifest.json", target / "manifest.json")
    read_manifest("broken", data_root=destination / "data")
    return Path("broken")
