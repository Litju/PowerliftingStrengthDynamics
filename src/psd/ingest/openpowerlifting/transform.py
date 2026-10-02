"""OpenPowerlifting to canonical transformation.

Bounded memory by construction
------------------------------

The full export is multi-million rows and its attempt and reported-result tables are
tens of millions, so the build is a sequence of streaming passes over data on disk
rather than over a list of rows:

1. **Stage.** The CSV is read exactly once, in bounded chunks. Each row gains its three
   derived identities -- athlete, meet, competition -- and is written to a staged Parquet
   partition named by the first hex character of its *athlete* identity. Memory is one
   chunk plus one open writer per partition, never the corpus.

2. **Build.** The partitions are walked in ascending order, and every per-athlete
   canonical table is produced from them: ``athlete``, ``athlete_source_link``,
   ``competition``, ``competition_attempt`` and ``competition_reported_result``.

Walking ascending partitions, each sorted internally, is the global canonical order.
That holds because every one of those tables is ordered ``athlete_id`` first, and the
partition key is a prefix of that leading sort column rather than an arbitrary hash
bucket. The property is what makes the partition walk and the canonical order the same
order, and it is why the canonical order of these tables is athlete-major in the first
place: a longitudinal competition corpus wants athlete-major order anyway.

``competition_meet`` is the one exception. Meet identity is not athlete-major, the table
is three orders of magnitude smaller than the others, and it is built once from a
streaming distinct over the staged identity columns.

Two fed passes
--------------

Each pass runs twice, because a canonical artifact's content digest declares its row
count in its own header and therefore cannot be final until the last partition has been
walked. See :mod:`psd.serialization.stream`.

Contract enforcement at this scale
----------------------------------

The build does **not** construct Pydantic records row by row. At tens of millions of rows
that would dominate the entire build, and the locked stack's answer at this scale is
vectorized Arrow/Polars validation. Every invariant the contracts enforce is enforced
here as a vectorized predicate instead, and :mod:`psd.ingest.openpowerlifting.audit`
re-derives the same checks over the persisted tables. The contracts stay authoritative
for fixtures and for record-level construction.

What is never done, anywhere in this module
-------------------------------------------

* A missing value is never turned into a zero.
* A signed attempt is never stored as a negative load.
* A negative reported best is never read as a successful lift.
* An approximate age is never rounded.
* An open-ended weight class is never collapsed to its bound.
* A participation code is never coerced to a placing.
* Attempts are never reconstructed from reported bests, and component lifts are never
  manufactured to make a total balance.
* An equipment category is never turned into an athlete equipment observation.
* One source identity's conflicting reports are never merged into a single attribute.
"""

from __future__ import annotations

import shutil
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from psd.ingest.openpowerlifting.contract import (
    expected_columns,
    read_source_header,
    require_source_schema,
)
from psd.ingest.openpowerlifting.mapping import (
    ATTEMPT_SPECS,
    EQUIPMENT_CLASS_BY_SOURCE,
    EVENT_BY_SOURCE,
    PLACE_KIND_BY_SOURCE,
    REPORTED_RESULT_SPECS,
    SEX_CATEGORY_BY_SOURCE,
    sanctioned_flag,
)
from psd.ingest.openpowerlifting.snapshot import OpenPowerliftingSnapshot
from psd.ingest.openpowerlifting.source import openpowerlifting_source_record
from psd.paths import resolve_within_data_root
from psd.provenance.environment import detect_git_state
from psd.provenance.manifest import DatasetKind, DatasetManifest, LineageEntry
from psd.schema.identifiers import IdPrefix, bulk_make_id
from psd.schema.registry import TableSpec, table_names, table_spec
from psd.schema.version import SCHEMA_VERSION
from psd.schema.vocabulary import (
    AgePrecision,
    AttemptOrderBasis,
    AttemptResult,
    EquipmentClass,
    EventTimePrecision,
    IdentityLinkMethod,
    IdentityStatus,
    ParticipationStatus,
    QualityFlag,
)
from psd.serialization.ordering import canonical_order
from psd.serialization.stream import StreamingDatasetSpec, StreamingDatasetWriter
from psd.serialization.table import empty_table

__all__ = (
    "BuildConfig",
    "BuildCounters",
    "BuildRequest",
    "BuildResult",
    "TransformError",
    "build_corpus",
    "normalize_meet_field",
    "partition_key",
    "partition_label",
    "partition_label_expression",
    "require_partition_count",
    "stage_snapshot",
    "staged_partitions",
)

#: A digest begins with one of these, so the first character is a partition key that
#: sorts in the same order as the identities it holds.
_PARTITION_ALPHABET: Final[str] = "0123456789abcdef"

#: Row-level source keys name the source and the row's 1-based position in the pinned
#: CSV. A source row carries no identifier of its own, so its ordinal inside a
#: digest-pinned file is the most direct key available, and it is stable exactly as long
#: as the digest is.
_SOURCE_ROW_KEY_PREFIX: Final[str] = "opl:row="

ROW_ORDINAL_COLUMN: Final[str] = "__psd_row"
ATHLETE_ID_COLUMN: Final[str] = "__psd_athlete_id"
MEET_ID_COLUMN: Final[str] = "__psd_meet_id"
COMPETITION_ID_COLUMN: Final[str] = "__psd_competition_id"
ATHLETE_KEY_COLUMN: Final[str] = "__psd_athlete_key"
MEET_KEY_COLUMN: Final[str] = "__psd_meet_key"
COMPETITION_KEY_COLUMN: Final[str] = "__psd_competition_key"
PARTITION_COLUMN: Final[str] = "__psd_partition"
NORMALIZED_PREFIX: Final[str] = "__psd_norm_"

_UNIT_SEPARATOR: Final[str] = "\x1f"

#: Local name for the row-position column a staging split materialises. Distinct from
#: ``ROW_ORDINAL_COLUMN``, which is the *source* row ordinal and is part of the staged
#: schema: reusing that name here would shadow the source ordinal it is meant to index.
_SPLIT_INDEX_COLUMN: Final[str] = "__psd_split_index"

#: PyArrow ships no inline types and the community stubs describe the Parquet writer's
#: parameter unions more narrowly than the runtime accepts. The untyped entry points are
#: reached through documented shims with explicit local signatures, as in
#: :mod:`psd.serialization.parquet`, so project-wide strictness is kept.
_PARQUET_WRITER_FACTORY: Any = pq.ParquetWriter
_WRITE_TABLE: Callable[[Any, pa.Table], None] = pq.ParquetWriter.write_table

#: The meet identity rule, stated once. A meet is identified by its start date, the
#: federation that hosted it, and where and under what name it was held -- each
#: normalized to lowercase with whitespace collapsed. ``MeetName`` is part of the rule
#: even though the source excludes the year and federation from it, because the other
#: five fields alone are nowhere near unique.
MEET_IDENTITY_COLUMNS: Final[tuple[str, ...]] = (
    "Date",
    "Federation",
    "MeetCountry",
    "MeetState",
    "MeetTown",
    "MeetName",
)

#: Source columns the meet record also states, beyond the identity itself.
MEET_CONTEXT_COLUMNS: Final[tuple[str, ...]] = ("ParentFederation", "Sanctioned")

_ATTEMPT_COLUMNS: Final[tuple[str, ...]] = tuple(column for _l, column, _n, _r in ATTEMPT_SPECS)
_REPORTED_COLUMNS: Final[tuple[str, ...]] = tuple(
    column for column, _kind, _best in REPORTED_RESULT_SPECS
)

#: Canonical column width limits. A value that cannot fit is nulled and counted rather
#: than truncated: a truncation would make two different source values read as one, while
#: a counted absence is recoverable and a silent truncation is not.
_WIDTH_LIMITS: Final[Mapping[str, int]] = {
    "Name": 512,
    "Division": 256,
    "Country": 128,
    "State": 128,
    "MeetName": 512,
    "MeetCountry": 128,
    "MeetState": 128,
    "MeetTown": 128,
    "Federation": 128,
    "ParentFederation": 128,
    "AgeClass": 64,
    "BirthYearClass": 64,
    "Equipment": 64,
    "Place": 64,
    "Sanctioned": 64,
    "WeightClassKg": 64,
}

#: An age is exact when the published value is a whole number. The comparison runs
#: against a tolerance because the source publishes two decimal places and floating point
#: does not respect that intent exactly.
_AGE_WHOLE_NUMBER_TOLERANCE: Final[float] = 1e-9

#: The lift a value belongs to, as a canonical ``lift_type`` member.
_LIFT_BY_ATTEMPT_COLUMN: Final[Mapping[str, str]] = {
    column: lift for lift, column, _number, _role in ATTEMPT_SPECS
}
_NUMBER_BY_ATTEMPT_COLUMN: Final[Mapping[str, int]] = {
    column: number for _lift, column, number, _role in ATTEMPT_SPECS
}
_ROLE_BY_ATTEMPT_COLUMN: Final[Mapping[str, str]] = {
    column: role.value for _lift, column, _number, role in ATTEMPT_SPECS
}
_KIND_BY_REPORTED_COLUMN: Final[Mapping[str, str]] = {
    column: kind for column, kind, _best in REPORTED_RESULT_SPECS
}
_BEST_LIFT_COLUMNS: Final[frozenset[str]] = frozenset(
    column for column, _kind, best in REPORTED_RESULT_SPECS if best
)

#: Reported results whose value is a mass in kilograms: the three best lifts and the
#: total. Scoring-system points are dimensionless and carry no unit.
_MASS_RESULT_COLUMNS: Final[frozenset[str]] = _BEST_LIFT_COLUMNS | {"TotalKg"}


class TransformError(RuntimeError):
    """Raised when the source cannot be converted faithfully."""


@dataclass(frozen=True, slots=True)
class BuildConfig:
    """Knobs for one corpus build.

    Attributes:
        chunk_rows: Source rows read per staged chunk. This is the primary memory bound
            of the staging pass.
        batch_rows: Rows per canonical batch handed to the writer. This bounds the
            transient Python objects the content encoder materializes. It defaults to a
            whole number of the Parquet profile's row groups, which makes the corpus's
            artifacts byte-identical to what the single-shot writer would produce from the
            same rows.
        partitions: Partition count. It must divide the sixteen a digest can start with,
            which is what keeps the partition walk in identity order: buckets are produced
            by scaling the leading hex digit, and that only preserves order for a divisor
            of sixteen. Checked here rather than at the point of use, so a caller finds out
            before a multi-hour build rather than after its first staging pass.
    """

    chunk_rows: int = 400_000
    batch_rows: int = 262_144
    partitions: int = 16

    def __post_init__(self) -> None:
        for name in ("chunk_rows", "batch_rows", "partitions"):
            value = getattr(self, name)
            if value < 1:
                msg = f"{name} must be at least 1; got {value}."
                raise ValueError(msg)
        alphabet = len(_PARTITION_ALPHABET)
        if self.partitions > alphabet:
            msg = (
                f"partitions must not exceed the identity alphabet size ({alphabet}); "
                f"got {self.partitions}."
            )
            raise ValueError(msg)
        require_partition_count(self.partitions)


@dataclass(slots=True)
class BuildCounters:
    """What the build saw, including everything it refused to keep.

    Each counter here corresponds to a source value that cannot be represented
    faithfully. They are reported, never repaired: a corpus whose irregularities have
    been quietly normalized is a corpus nobody can trust.

    Attributes:
        source_rows: Data rows read from the pinned CSV.
        attempt_values_unparseable: Attempt cells present but not numbers.
        attempt_values_zero: Attempt cells reported as ``0``, which is not a load.
        attempt_values_dropped: Attempt rows excluded, being unparseable or zero.
        age_values_unparseable: Age cells present but not numbers.
        bodyweight_values_unparseable: Body-mass cells present but not numbers.
        bodyweight_values_nonpositive: Body masses that were not above zero.
        dates_unparseable: Meet dates present but not ``YYYY-MM-DD``.
        values_too_long: Source values nulled for exceeding a canonical width.
        rows_without_federation: Rows whose hosting federation was absent, although the
            source declares it mandatory.
        unrecognised_values: Distinct values no declared mapping covers, per column.
    """

    source_rows: int = 0
    attempt_values_unparseable: int = 0
    attempt_values_zero: int = 0
    attempt_values_dropped: int = 0
    age_values_unparseable: int = 0
    bodyweight_values_unparseable: int = 0
    bodyweight_values_nonpositive: int = 0
    dates_unparseable: int = 0
    values_too_long: int = 0
    rows_without_federation: int = 0
    unrecognised_values: dict[str, tuple[str, ...]] = field(
        default_factory=dict[str, tuple[str, ...]]
    )

    def note_unrecognised(self, column: str, values: Sequence[str]) -> None:
        """Record the distinct unrecognised values seen for *column*."""
        cleaned = tuple(sorted(value for value in set(values) if value.strip()))
        if cleaned:
            self.unrecognised_values[column] = cleaned

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable rendering of the counters."""
        return {
            "source_rows": self.source_rows,
            "attempt_values_unparseable": self.attempt_values_unparseable,
            "attempt_values_zero": self.attempt_values_zero,
            "attempt_values_dropped": self.attempt_values_dropped,
            "age_values_unparseable": self.age_values_unparseable,
            "bodyweight_values_unparseable": self.bodyweight_values_unparseable,
            "bodyweight_values_nonpositive": self.bodyweight_values_nonpositive,
            "dates_unparseable": self.dates_unparseable,
            "values_too_long": self.values_too_long,
            "rows_without_federation": self.rows_without_federation,
            "unrecognised_values": {
                key: list(values) for key, values in sorted(self.unrecognised_values.items())
            },
        }


@dataclass(frozen=True, slots=True)
class BuildResult:
    """Outcome of one corpus build.

    Attributes:
        manifest: The persisted dataset manifest.
        counters: What the build saw and refused.
        staging_directory: Where the staged rows live. Safe to delete once the dataset
            has been verified.
        build_seconds: Wall-clock seconds the build took.
        row_counts: Canonical row counts per table, as persisted.
    """

    manifest: DatasetManifest
    counters: BuildCounters
    staging_directory: Path
    build_seconds: float
    row_counts: dict[str, int]


def normalize_meet_field(value: str | None) -> str:
    """Return the normalized form of one meet identity field.

    Normalization is lowercasing plus surrounding and repeated whitespace collapsed. It
    exists to stop cosmetic differences from minting two identities for one meet, and it
    is deliberately minimal: an aggressive cleaner would eventually merge genuinely
    different meets, whereas a variant that survives this normalization is reported as a
    meet-identity variant rather than silently merged.

    Args:
        value: A source value, possibly empty.

    Returns:
        The normalized value, or the empty string for an absent one.
    """
    if not value:
        return ""
    return " ".join(value.split()).lower()


def partition_key(identifier: str) -> str:
    """Return the first digest character of a derived identifier.

    That character is the partition key when all sixteen are used, because it sorts in
    the same order as the identifiers that carry it.
    """
    _prefix, separator, digest = identifier.partition("_")
    if not separator or not digest:
        msg = f"{identifier!r} is not a derived identifier."
        raise TransformError(msg)
    return digest[0]


#: The first hex character of an identifier's digest, as a vectorized extraction.
#:
#: Anchored on the underscore separator so it cannot match a hex-looking character inside
#: the prefix. Derived identifiers are ``<tag>_<32 hex>`` and no tag contains an
#: underscore, so the first ``_<hex>`` in one is always the start of the digest.
_DIGEST_FIRST_HEX: Final[str] = r"_([0-9a-f])"


def require_partition_count(partitions: int) -> None:
    """Refuse a partition count that cannot preserve identity order.

    Buckets are produced by scaling the leading hex digit onto the requested count, which
    keeps the partition walk in identity order only for a divisor of sixteen. Taken at
    face value, a count of five would produce buckets that interleave, and the build would
    silently write a canonically-ordered table in the wrong order.

    Raises:
        TransformError: The count is not a divisor of sixteen.
    """
    if 16 % partitions != 0:
        msg = (
            f"partitions must divide the sixteen a digest can start with; got {partitions}. "
            "Bucketing by scaling the leading hex digit is what keeps the partition walk "
            "in identity order, and that only works for a divisor of sixteen."
        )
        raise TransformError(msg)


def _partition_buckets(partitions: int) -> dict[str, str]:
    """Return the leading-hex-character to partition-label mapping.

    Declared once and used by both the vectorized expression and the single-identifier
    helper, so the two cannot disagree. They did once, and the disagreement was silent:
    every staged row landed in one partition while the configuration asked for sixteen.
    """
    return {
        character: _PARTITION_ALPHABET[int(character, 16) * partitions // 16]
        for character in _PARTITION_ALPHABET
    }


def partition_label(identifier: str, partitions: int) -> str:
    """Return the partition a derived identifier belongs to.

    Identities are bucketed by scaling the leading hex digit onto the requested count,
    which requires the count to divide sixteen. Bucketing this way -- rather than taking
    the first ``n`` characters of the alphabet -- keeps the partition walk in identity
    order for every valid count, which is the property the canonical ordering relies on.

    Args:
        identifier: A derived identifier.
        partitions: Partition count; must divide sixteen.

    Returns:
        A partition label in ``0``-``9``/``a``-``f``.

    Raises:
        TransformError: The partition count does not divide sixteen, or the identifier
            is not a derived identifier.
    """
    require_partition_count(partitions)
    return _partition_buckets(partitions)[partition_key(identifier)]


def partition_label_expression(partitions: int) -> pl.Expr:
    """Return the vectorized partition label of each row's athlete identity.

    Reads the identifier's **digest**, not its first character. An identifier is
    ``<tag>_<32 hex>``, so taking ``str.slice(0, 1)`` reads the ``a`` of ``ath`` and every
    row in the corpus lands in one partition -- which turns a sixteen-way memory bound
    into a no-op and hands the whole corpus to a single pass. That is exactly what the
    pinned 4,036,909-row snapshot did until this was fixed.

    Args:
        partitions: Partition count; must divide sixteen.

    Returns:
        An expression over ``athlete_id`` producing each row's partition label.

    Raises:
        TransformError: The partition count does not divide sixteen.
    """
    require_partition_count(partitions)
    return (
        pl.col(ATHLETE_ID_COLUMN)
        .str.extract(_DIGEST_FIRST_HEX, 1)
        .replace_strict(_partition_buckets(partitions), default=None, return_dtype=pl.String)
    )


# --------------------------------------------------------------------------
# vectorized expressions
# --------------------------------------------------------------------------


def _nullable(column: str) -> pl.Expr:
    """Return an expression mapping an absent or blank source value to null."""
    return (
        pl.when(pl.col(column).fill_null("").str.strip_chars() == "")
        .then(None)
        .otherwise(pl.col(column))
    )


def _fit(column: str, limit: int) -> pl.Expr:
    """Return *column* as text, nulling a value that exceeds the canonical width."""
    text = _nullable(column)
    return pl.when(text.str.len_chars() <= limit).then(text).otherwise(None).alias(column)


def _float(column: str, alias: str) -> pl.Expr:
    """Return an expression converting a text column to a float, or null."""
    return _nullable(column).str.strip_chars().cast(pl.Float64, strict=False).alias(alias)


def _part(column: str | pl.Expr, alias: str) -> pl.Expr:
    """Return an expression encoding *column* as one length-prefixed identity part.

    The length prefix is computed from the value rather than assumed, so a part of any
    width is encoded exactly as :func:`psd.schema.identifiers.id_key` encodes it.
    Building keys with vectorized string operations is what keeps identity generation
    from dominating a multi-million-row build.

    Args:
        column: Column name, or an expression already producing the string to encode.
        alias: Output column name.
    """
    values = pl.col(column) if isinstance(column, str) else column
    return pl.format("s{}:{}", values.str.len_chars(), values).alias(alias)


def _constant_part(value: str, alias: str) -> pl.Expr:
    """Return an expression encoding a known constant as one identity part."""
    return pl.lit(f"s{len(value)}:{value}").alias(alias)


def _mapped(column: str, mapping: Mapping[str, str]) -> pl.Expr:
    """Return an expression mapping a text column through a declared vocabulary.

    An unmapped value becomes null rather than a default, so a new federation category
    shows up in the audit as a count instead of disappearing into ``unknown``.
    """
    return (
        pl.col(column)
        .fill_null("")
        .str.strip_chars()
        .replace_strict(mapping, default=None, return_dtype=pl.String)
    )


def _empty_quality_flags() -> pl.Expr:
    """Return an empty quality-flag list expression.

    Written as a typed empty list rather than a cast of an empty tuple, because the
    canonical column is ``list<string>`` and Polars has to be told the element type
    rather than inferring it from nothing.
    """
    return pl.lit([], dtype=pl.List(pl.String))


def _quality_flags(*flags: str) -> pl.Expr:
    """Return a quality-flag list expression holding *flags*."""
    return pl.lit(sorted(flags), dtype=pl.List(pl.String))


def _utc_midnight(column: str, alias: str) -> pl.Expr:
    """Return an expression reading an ISO ``YYYY-MM-DD`` column as a UTC instant.

    A date-only source has no instant, so midnight UTC is the convention and the
    record's ``event_time_precision`` says the value is a day rather than a moment.
    """
    return (
        pl.when(pl.col(column).fill_null("").str.strip_chars() == "")
        .then(None)
        .otherwise(pl.col(column).str.strptime(pl.Date, "%Y-%m-%d", strict=False))
        .cast(pl.Datetime("us"))
        .dt.replace_time_zone("UTC")
        .alias(alias)
    )


# --------------------------------------------------------------------------
# staging
# --------------------------------------------------------------------------


def _staged_schema() -> pa.Schema:
    """Return the Arrow schema of a staged partition.

    Every staged column is nullable because a source cell may legitimately be empty and
    staging represents that as null rather than as an empty string. Staging is not a
    canonical artifact, so it does not need the canonical schema's contract; it needs to
    hold what the source said without changing it.
    """
    return pa.schema(
        [
            (ROW_ORDINAL_COLUMN, pa.uint32()),
            *((name, pa.string()) for name in expected_columns()),
            (ATHLETE_ID_COLUMN, pa.string()),
            (MEET_ID_COLUMN, pa.string()),
            (COMPETITION_ID_COLUMN, pa.string()),
        ]
    )


_STAGED_SELECT: Final[tuple[str, ...]] = (
    ROW_ORDINAL_COLUMN,
    *expected_columns(),
    ATHLETE_ID_COLUMN,
    MEET_ID_COLUMN,
    COMPETITION_ID_COLUMN,
)


def _normalized_meet_expressions() -> list[pl.Expr]:
    """Return the normalized copies of the meet identity fields.

    Normalization is applied identically wherever a meet identity is derived, so the
    staging pass and the meet pass cannot disagree about which raw spellings name the
    same meet. It is deliberately minimal -- lowercasing and whitespace collapse -- because
    an aggressive cleaner would eventually merge genuinely different meets.
    """
    return [
        pl.col(column)
        .fill_null("")
        .str.replace_all(r"\s+", " ")
        .str.strip_chars()
        .str.to_lowercase()
        .alias(f"{NORMALIZED_PREFIX}{column}")
        for column in MEET_IDENTITY_COLUMNS
    ]


def _prepare_chunk(chunk: pl.DataFrame, *, source_id: str, first_row: int) -> pl.DataFrame:
    """Return one CSV chunk with width guards, normalized fields, and identity keys."""
    frame = chunk.with_columns(
        [_fit(column, limit) for column, limit in sorted(_WIDTH_LIMITS.items())]
    ).with_columns(_normalized_meet_expressions())
    with_row = frame.with_row_index(ROW_ORDINAL_COLUMN, offset=first_row).with_columns(
        pl.col(ROW_ORDINAL_COLUMN).cast(pl.UInt32)
    )
    return with_row.with_columns(
        pl.concat_str(
            [
                _constant_part(source_id, "__psd_src"),
                pl.lit(_UNIT_SEPARATOR),
                _part("Name", "__psd_nm"),
            ],
        ).alias(ATHLETE_KEY_COLUMN),
        pl.concat_str(_meet_key_expressions(source_id)).alias(MEET_KEY_COLUMN),
    )


def _meet_key_expressions(source_id: str) -> list[pl.Expr]:
    """Return the identity parts of a meet key, in order.

    Declared once and used both where a staged row's meet identity is derived and where
    the meet table is built from the staged rows. The two derivations must agree
    byte-for-byte: when they disagree, every ``competition`` row points at a
    ``competition_meet_id`` that does not exist, and **no digest catches it**, because both
    tables are internally consistent. The audit's ``competitions_reference_a_meet``
    invariant is what found that, and it can only exist because the check is cheap.

    The source identifier leads, as it does for every other identity here, so two snapshots
    of the same service cannot mint colliding meet identities and a meet's identity cannot
    be derived without knowing which source it came from.
    """
    return [
        _constant_part(source_id, "__psd_src"),
        pl.lit(_UNIT_SEPARATOR),
        *(
            _part(f"{NORMALIZED_PREFIX}{column}", f"__psd_m{index}")
            for index, column in enumerate(MEET_IDENTITY_COLUMNS)
        ),
    ]


def _assign(
    frame: pl.DataFrame, *, key_column: str, prefix: IdPrefix, id_column: str
) -> pl.DataFrame:
    """Return *frame* with *id_column* derived from *key_column*."""
    identifiers = bulk_make_id(prefix, frame.get_column(key_column).to_list())
    return frame.with_columns(pl.Series(id_column, identifiers, dtype=pl.String))


def stage_snapshot(
    csv_path: Path,
    staging: Path,
    *,
    source_id: str,
    config: BuildConfig,
    counters: BuildCounters,
) -> None:
    """Read the pinned CSV once and write it to partitioned staged Parquet.

    The staging directory is rebuilt from nothing every time. Partition files are appended
    to as the CSV streams past, so staging on top of a previous build's leftovers would
    double every table -- silently, and only the second time the same snapshot is built.
    Staged rows are a derived artifact of one CSV; the build re-reads the CSV rather than
    reusing them, so there is nothing to lose by starting clean.

    This is public because the staged layout is an observable property of the corpus, and
    asserting it -- that rows really are spread across the partitions asked for, and that
    walking them yields the canonical order -- is the only way to catch a staging pass that
    quietly stops partitioning.
    """
    schema = _staged_schema()
    alphabet = _PARTITION_ALPHABET[: config.partitions]
    writers: dict[str, Any] = {}
    directory = staging / "athlete"
    _reset_staging(staging)
    directory.mkdir(parents=True, exist_ok=True)
    row_offset = 0
    try:
        for chunk in _read_csv_chunks(csv_path, chunk_rows=config.chunk_rows):
            prepared = _prepare_chunk(chunk, source_id=source_id, first_row=row_offset)
            prepared = _assign(
                prepared,
                key_column=ATHLETE_KEY_COLUMN,
                prefix=IdPrefix.ATHLETE,
                id_column=ATHLETE_ID_COLUMN,
            )
            prepared = _assign(
                prepared,
                key_column=MEET_KEY_COLUMN,
                prefix=IdPrefix.COMPETITION_MEET,
                id_column=MEET_ID_COLUMN,
            )
            prepared = _derive_competition(prepared)
            counters.source_rows += chunk.height
            _write_partitions(
                prepared,
                writers=writers,
                directory=directory,
                alphabet=alphabet,
                schema=schema,
            )
            row_offset += chunk.height
    finally:
        for writer in writers.values():
            writer.close()


def _reset_staging(staging: Path) -> None:
    """Remove any previously staged partition files.

    Raises:
        TransformError: The staging directory exists but is not a directory, which would
            mean something other than this build put it there.
    """
    if not staging.exists():
        staging.mkdir(parents=True)
        return
    if not staging.is_dir():
        msg = (
            f"Staging path {staging} exists and is not a directory. Refusing to stage over "
            "it; the build needs its own staging directory."
        )
        raise TransformError(msg)
    for entry in staging.iterdir():
        if entry.is_dir():
            shutil.rmtree(entry)
        else:
            entry.unlink()


def _derive_competition(frame: pl.DataFrame) -> pl.DataFrame:
    """Return *frame* with the competition identity derived from athlete, meet and entry.

    A competition is one entry: one source row describing one lifter at one meet. The source
    row is therefore part of the identity, not something to fold away.

    Keying on athlete and meet alone looks tidier and was the obvious first design, but on
    the pinned snapshot it collapses 209,724 groups of source rows that share a name, meet
    and event -- 433,341 rows -- into single entries. Some of those are one lifter entered
    twice and merging them is harmless; in 171 of them the source publishes different sex
    categories for the same name at the same meet, and merging then attributes one person's
    attempts and placings to another.

    Neither reading is decidable from this source, so PSD does not decide. Every source entry
    keeps its own row, each carries the ``source_record_key`` it came from, and the shared
    ``competition_meet_id`` still lets a consumer count the meets a lifter attended. Where the
    name itself is ambiguous, the athlete row carries an ``ambiguity_group_id``; the two
    facts are kept apart on purpose, because an entry is what the source published and an
    identity is what PSD inferred.
    """
    keyed = frame.with_columns(
        pl.concat_str(
            [
                _part(ATHLETE_ID_COLUMN, "__psd_ai"),
                pl.lit(_UNIT_SEPARATOR),
                _part(MEET_ID_COLUMN, "__psd_mi"),
                pl.lit(_UNIT_SEPARATOR),
                _part(
                    pl.col(ROW_ORDINAL_COLUMN).cast(pl.String),
                    "__psd_rk",
                ),
            ],
        ).alias(COMPETITION_KEY_COLUMN)
    )
    return _assign(
        keyed,
        key_column=COMPETITION_KEY_COLUMN,
        prefix=IdPrefix.COMPETITION,
        id_column=COMPETITION_ID_COLUMN,
    )


def _read_csv_chunks(csv_path: Path, *, chunk_rows: int) -> Iterator[pl.DataFrame]:
    """Yield the pinned CSV in bounded chunks, with every column read as text.

    Every column is read as text and converted explicitly afterwards. Letting Polars
    infer dtypes would make the transform's behaviour depend on the first chunk's
    contents, which is not a property a reproducible build can have. ``maintain_order``
    keeps the chunk sequence in file order, which is what makes a staged row ordinal a
    faithful reference to a source row.
    """
    scan = pl.scan_csv(
        csv_path,
        has_header=True,
        infer_schema_length=0,
        schema_overrides=dict.fromkeys(expected_columns(), pl.String),
        low_memory=True,
        raise_if_empty=False,
    )
    yield from scan.collect_batches(chunk_size=chunk_rows, maintain_order=True, lazy=False)


def _write_partitions(
    frame: pl.DataFrame,
    *,
    writers: dict[str, Any],
    directory: Path,
    alphabet: str,
    schema: pa.Schema,
) -> None:
    """Append each partition of *frame* to its staged file.

    The partition label is computed on a one-column frame and the split yields *row
    positions*, so the staged frame is never copied in order to hold one more column. That
    matters: adding the label inline copies every column of the chunk -- the raw source
    columns and the length-prefixed identity keys included -- and the difference between
    that and a position list is several times the chunk the pass was told to bound.
    """
    positions = frame.select(
        partition_label_expression(len(alphabet)).alias(PARTITION_COLUMN),
    ).with_row_index(_SPLIT_INDEX_COLUMN)
    for key, group in positions.partition_by(PARTITION_COLUMN, as_dict=True).items():
        label = str(key[0])
        if label not in alphabet:
            continue
        table = frame[group.get_column(_SPLIT_INDEX_COLUMN)].select(_STAGED_SELECT).to_arrow()
        table = table.cast(schema)
        writer = writers.get(label)
        if writer is None:
            path = directory / f"{label}.parquet"
            writer = _PARQUET_WRITER_FACTORY(path, schema, compression="zstd")
            writers[label] = writer
        _WRITE_TABLE(writer, table)


def staged_partitions(staging: Path) -> list[tuple[str, Path]]:
    """Return the staged partitions in ascending key order.

    This walk is the corpus's order, not an implementation detail of one build. Every
    athlete-major canonical table is produced by reading the partitions in exactly this
    sequence and concatenating what each yields, so ascending key order here is what puts
    a table's rows in the order its digest was computed from.

    Ascending key order is ascending order of the identity's leading digest character
    (see `partition_label`), which is why it is a global order and not merely a per-partition
    one: partition ranges are disjoint, so the walk cannot move backwards.
    """
    directory = staging / "athlete"
    if not directory.is_dir():
        return []
    return [(path.stem, path) for path in sorted(directory.glob("*.parquet"))]


def _read_partition(path: Path) -> pl.DataFrame:
    """Read one staged partition."""
    return pl.read_parquet(path)


# --------------------------------------------------------------------------
# canonical batches
# --------------------------------------------------------------------------


def _ordered(table_name: str, frame: pl.DataFrame) -> pa.Table:
    """Return *frame* as a canonical, canonically ordered Arrow table.

    Sorting is delegated to the registry's declared ordering, so a table can never be
    persisted in an order the canonical contract does not describe.
    """
    spec = table_spec(table_name)
    payload = frame.select(list(spec.columns)).to_arrow().cast(spec.arrow_schema())
    return canonical_order(payload, spec.order_by, table_name=table_name)


def _batches(table: pa.Table, *, rows: int) -> Iterator[pa.Table]:
    """Yield *table* in slices of at most *rows* rows, skipping empties."""
    offset = 0
    while offset < table.num_rows:
        chunk = table.slice(offset, rows)
        offset += rows
        yield chunk


def _feed(
    writer: StreamingDatasetWriter,
    table_name: str,
    table: pa.Table,
    *,
    counting: bool,
    rows: int,
) -> None:
    """Hand a partition's canonical table to the writer, one bounded batch at a time.

    The slice size is what bounds the transient Python objects the canonical content
    encoder materializes. A whole four-million-row table fed in one piece would make the
    encoder itself the peak-memory problem the partitioning was meant to avoid.

    The counting pass validates and sizes each batch. The writing pass hashes and writes the
    same batches, and is why the tables are built twice: the digest's header needs the final
    row count before any row can be hashed, and building is far cheaper than hashing.
    """
    if table.num_rows == 0:
        return
    for chunk in _batches(table, rows=rows):
        if counting:
            writer.count_batch(table_name, chunk)
        else:
            writer.write_batch(table_name, chunk)


def _source_row_key(ordinal: pl.Expr) -> pl.Expr:
    """Return the 1-based source row key for a staged ordinal column."""
    return pl.lit(_SOURCE_ROW_KEY_PREFIX) + (ordinal.cast(pl.Int64) + 1).cast(pl.String)


def _build_athletes(
    partition: pl.DataFrame, *, source_id: str, ingested_at: datetime
) -> tuple[pa.Table, pa.Table]:
    """Return the athlete and athlete-source-link tables for one partition.

    One athlete per source identity representation. The ``#N`` disambiguator the source
    appends to a shared name is part of that representation and is never stripped, so
    two lifters the source says are distinct never merge.

    A name the source reports under more than one sex category is *not* resolved by
    picking one. Its sex is left absent, the record is flagged, and the name is given an
    ambiguity group: the source contradicts itself, and collapsing that into an attribute
    would hide the contradiction rather than record it.

    One athlete row per **name**, not per name-and-sex pair. The identity is derived from
    the name alone, so a name reported under two sex categories is one identity wearing
    two rows -- which duplicates the ``athlete`` primary key and the
    ``athlete_source_link`` key, and leaves a consumer unable to tell a duplicated
    athlete from two athletes. Aggregating to the name first is what makes the declared
    uniqueness true.

    A conflict requires two *reported* sexes. A row that omits ``Sex`` and a row that
    states one are not a contradiction: the source said nothing on one side, and an
    absence is not a competing fact. Counting an absent sex as a variant would flag
    ordinary partial reporting as an identity conflict.
    """
    reported = partition.select(
        _nullable("Sex").alias("Sex"),
        pl.col("Name"),
    ).unique()
    resolved = (
        reported.group_by("Name", maintain_order=True)
        .agg(
            # Only non-absent sexes are variants, so a blank Sex cannot invent a conflict.
            pl.col("Sex").drop_nulls().n_unique().cast(pl.Int64).alias("__sex_variants"),
            pl.col("Sex").drop_nulls().min().alias("__reported_sex"),
        )
        .sort("Name")
    )
    keys = resolved.select(
        pl.concat_str(
            [
                _constant_part(source_id, "__psd_s"),
                pl.lit(_UNIT_SEPARATOR),
                _part("Name", "__psd_n"),
            ],
        ).alias("__link_key")
    )
    athlete_ids = bulk_make_id(IdPrefix.ATHLETE, keys.get_column("__link_key").to_list())
    stamp = pl.lit(ingested_at)
    ambiguous = pl.col("__sex_variants") > 1
    athletes = resolved.select(
        pl.Series(ATHLETE_ID_COLUMN, athlete_ids, dtype=pl.String).alias("athlete_id"),
        pl.lit(None, dtype=pl.String).alias("pseudonym"),
        pl.lit(IdentityStatus.SINGLE_SOURCE_UNVERIFIED.value).alias("identity_status"),
        pl.when(ambiguous)
        .then(pl.concat_str([pl.lit("opl-name-conflict:"), pl.col("Name")]))
        .otherwise(None)
        .alias("ambiguity_group_id"),
        pl.lit(False).alias("is_synthetic"),
        pl.lit(None, dtype=pl.String).alias("synthetic_regime"),
        pl.when(ambiguous).then(None).otherwise(pl.col("__reported_sex")).alias("sex_category_raw"),
        pl.when(ambiguous)
        .then(None)
        .otherwise(
            _mapped("__reported_sex", {k: v.value for k, v in SEX_CATEGORY_BY_SOURCE.items()})
        )
        .alias("sex_category"),
        # No birth year: a birth-year *class* is a bucket, not a year, and inferring one
        # from it would invent a fact the source never stated.
        pl.lit(None, dtype=pl.Int64).alias("birth_year"),
        # No ISO code: the source's country vocabulary is its own, and PSD does not
        # remap names to a standard it has not been given.
        pl.lit(None, dtype=pl.String).alias("country_code"),
        pl.lit(None, dtype=pl.Datetime("us", "UTC")).alias("created_at"),
        pl.lit(source_id).alias("source_id"),
        pl.lit(None, dtype=pl.String).alias("source_record_key"),
        pl.lit(None, dtype=pl.String).alias("source_record_hash"),
        stamp.alias("ingested_at"),
        pl.when(ambiguous)
        .then(_quality_flags(QualityFlag.AMBIGUOUS_IDENTITY.value))
        .otherwise(_empty_quality_flags())
        .alias("quality_flags"),
        pl.lit(None, dtype=pl.String).alias("missingness_reason"),
    )
    links = resolved.select(
        pl.Series(ATHLETE_ID_COLUMN, athlete_ids, dtype=pl.String).alias("athlete_id"),
        _nullable("Name").alias("source_athlete_key"),
        pl.lit(IdentityLinkMethod.EXACT_ATTRIBUTE_MATCH.value).alias("link_method"),
        # No confidence: PSD performs no calibrated identity model, and a float here
        # would read as a calibrated probability.
        pl.lit(None, dtype=pl.Float64).alias("link_confidence"),
        pl.lit(True).alias("is_primary"),
        pl.lit(None, dtype=pl.Datetime("us", "UTC")).alias("created_at"),
        pl.lit(source_id).alias("source_id"),
        pl.col("Name").alias("source_record_key"),
        pl.lit(None, dtype=pl.String).alias("source_record_hash"),
        stamp.alias("ingested_at"),
        _empty_quality_flags().alias("quality_flags"),
        pl.lit(None, dtype=pl.String).alias("missingness_reason"),
    )
    return _ordered("athlete", athletes), _ordered("athlete_source_link", links)


def _build_competitions(
    partition: pl.DataFrame, *, source_id: str, ingested_at: datetime
) -> pa.Table:
    """Return the canonical competition table for one athlete partition."""
    prepared = partition.with_columns(
        [
            _utc_midnight("Date", "competition_date"),
            _mapped("Event", {key: value.value for key, value in EVENT_BY_SOURCE.items()}).alias(
                "competition_event"
            ),
            _mapped("Equipment", {k: v.value for k, v in EQUIPMENT_CLASS_BY_SOURCE.items()}).alias(
                "__equipment_class"
            ),
            _nullable("Equipment").alias("equipment_class_raw"),
            _nullable("WeightClassKg").alias("weight_class_raw"),
            _nullable("AgeClass").alias("age_class_raw"),
            _nullable("BirthYearClass").alias("birth_year_class_raw"),
            _nullable("Division").alias("division_raw"),
            _nullable("Country").alias("athlete_country_raw"),
            _nullable("State").alias("athlete_region_raw"),
            _nullable("Federation").alias("federation"),
            _nullable("ParentFederation").alias("sanctioning_body"),
            _nullable("MeetName").alias("name"),
            _float("Age", "__age_raw"),
            _float("BodyweightKg", "__bodyweight_raw"),
            _float("Place", "__place_raw"),
        ]
    )
    exact = pl.col("__age_raw").is_not_null() & (
        (pl.col("__age_raw") - pl.col("__age_raw").floor()).abs() < _AGE_WHOLE_NUMBER_TOLERANCE
    )
    # Guest, disqualified, doping-disqualified and no-show are statuses, not ranks.
    # Coercing any of them to a number would invent a placing nobody earned, so the
    # non-numeric codes are matched before the numeric placing is considered.
    place_kind = _mapped(
        "Place", {key: value.value for key, value in PLACE_KIND_BY_SOURCE.items()}
    ).fill_null(
        pl.when(pl.col("__place_raw") > 0.0)
        .then(pl.lit(ParticipationStatus.PLACED.value))
        .otherwise(pl.lit(ParticipationStatus.UNKNOWN.value))
    )
    frame = prepared.select(
        pl.col(COMPETITION_ID_COLUMN).alias("competition_id"),
        pl.col(ATHLETE_ID_COLUMN).alias("athlete_id"),
        pl.col(MEET_ID_COLUMN).alias("competition_meet_id"),
        pl.col("competition_date"),
        pl.lit(EventTimePrecision.DATE_ONLY.value).alias("event_time_precision"),
        pl.col("competition_event"),
        pl.col("name"),
        pl.col("federation"),
        pl.col("sanctioning_body"),
        # Country, state and town are stated once on the meet. Concatenating them here
        # would invent a formatted address the source never published.
        pl.lit(None, dtype=pl.String).alias("location"),
        pl.col("equipment_class_raw"),
        pl.col("__equipment_class")
        .fill_null(EquipmentClass.UNKNOWN.value)
        .alias("equipment_class"),
        pl.col("weight_class_raw"),
        pl.when(pl.col("__bodyweight_raw") > 0.0)
        .then(pl.col("__bodyweight_raw"))
        .otherwise(None)
        .alias("bodyweight_raw"),
        pl.when(pl.col("__bodyweight_raw") > 0.0)
        .then(pl.lit("kg"))
        .otherwise(None)
        .alias("bodyweight_unit"),
        pl.when(pl.col("__bodyweight_raw") > 0.0)
        .then(pl.col("__bodyweight_raw"))
        .otherwise(None)
        .alias("bodyweight_kg"),
        pl.col("Place").alias("participation_status"),
        pl.when(place_kind == pl.lit("placed"))
        .then(pl.col("__place_raw").cast(pl.Int64))
        .otherwise(None)
        .alias("participation_place"),
        place_kind.alias("participation_status_kind"),
        pl.when(pl.col("Tested").str.strip_chars() == pl.lit("Yes"))
        .then(True)
        .otherwise(None)
        .alias("is_drug_tested_category"),
        pl.col("__age_raw").alias("age_reported"),
        pl.when(pl.col("__age_raw").is_null())
        .then(None)
        .when(exact)
        .then(pl.lit(AgePrecision.EXACT.value))
        .otherwise(pl.lit(AgePrecision.APPROXIMATE.value))
        .alias("age_precision"),
        pl.col("age_class_raw"),
        pl.col("birth_year_class_raw"),
        pl.col("division_raw"),
        pl.col("athlete_country_raw"),
        pl.col("athlete_region_raw"),
        pl.lit(None, dtype=pl.Boolean).alias("is_championship"),
        pl.lit(None, dtype=pl.Datetime("us", "UTC")).alias("created_at"),
        pl.lit(None, dtype=pl.Datetime("us", "UTC")).alias("scheduled_at"),
        pl.lit(None, dtype=pl.Datetime("us", "UTC")).alias("performed_at"),
        pl.col("competition_date").alias("observed_at"),
        pl.lit(None, dtype=pl.Datetime("us", "UTC")).alias("modified_at"),
        pl.lit(source_id).alias("source_id"),
        _source_row_key(pl.col(ROW_ORDINAL_COLUMN)).alias("source_record_key"),
        pl.lit(None, dtype=pl.String).alias("source_record_hash"),
        pl.lit(ingested_at).alias("ingested_at"),
        _empty_quality_flags().alias("quality_flags"),
        pl.lit(None, dtype=pl.String).alias("missingness_reason"),
    )
    return _ordered("competition", frame)


def _melt(
    partition: pl.DataFrame, columns: Sequence[str], *, value_alias: str, column_alias: str
) -> pl.DataFrame:
    """Return one long row per non-absent cell across *columns*.

    Long rather than wide is what keeps the transform honest about absence: a source that
    publishes no attempts produces no attempt rows, because an absent cell never becomes
    a row at all. It is also never a zero.
    """
    return (
        partition.select(
            pl.col(ATHLETE_ID_COLUMN),
            pl.col(COMPETITION_ID_COLUMN),
            pl.col(ROW_ORDINAL_COLUMN),
            pl.col("Date"),
            *columns,
        )
        .unpivot(
            on=list(columns),
            index=[ATHLETE_ID_COLUMN, COMPETITION_ID_COLUMN, ROW_ORDINAL_COLUMN, "Date"],
            variable_name=column_alias,
            value_name=value_alias,
        )
        .filter(pl.col(value_alias).is_not_null())
    )


def _build_attempts(
    partition: pl.DataFrame, *, source_id: str, ingested_at: datetime, counters: BuildCounters
) -> pa.Table:
    """Return the canonical attempt table for one athlete partition.

    The sign is the source's result encoding, so ``-200`` is a failed attempt at 200 kg
    and never a negative load. A cell reported as ``0`` is not a load at all: it is
    counted and dropped, because ``0 kg`` is not an attempt and not a missing value
    either, and storing either reading would invent one.
    """
    long = _melt(partition, _ATTEMPT_COLUMNS, value_alias="__value", column_alias="__column")
    if long.height == 0:
        return _empty_table("competition_attempt")
    values = pl.col("__value").str.strip_chars().cast(pl.Float64, strict=False).alias("__attempt")
    long = long.with_columns(values).with_columns(
        [
            _mapped("__column", _LIFT_BY_ATTEMPT_COLUMN).alias("lift"),
            pl.col("__column")
            .replace_strict(_NUMBER_BY_ATTEMPT_COLUMN, default=None, return_dtype=pl.Int64)
            .alias("attempt_number"),
            _mapped("__column", _ROLE_BY_ATTEMPT_COLUMN).alias("attempt_role"),
        ]
    )
    unparseable = int(long.filter(pl.col("__attempt").is_null()).height)
    zero = int(long.filter(pl.col("__attempt") == 0.0).height)
    kept = long.filter(pl.col("__attempt").is_not_null() & (pl.col("__attempt") != 0.0))
    counters.attempt_values_unparseable += unparseable
    counters.attempt_values_zero += zero
    counters.attempt_values_dropped += unparseable + zero
    if kept.height == 0:
        return _empty_table("competition_attempt")

    keyed = kept.with_columns(
        pl.concat_str(
            [
                _part(COMPETITION_ID_COLUMN, "__psd_ci"),
                pl.lit(_UNIT_SEPARATOR),
                _part("lift", "__psd_lf"),
                pl.lit(_UNIT_SEPARATOR),
                pl.format("i{}", pl.col("attempt_number")).alias("__psd_n"),
            ],
        ).alias("__attempt_key")
    )
    attempt_ids = bulk_make_id(
        IdPrefix.COMPETITION_ATTEMPT, keyed.get_column("__attempt_key").to_list()
    )
    result = (
        pl.when(pl.col("__attempt") > 0.0)
        .then(pl.lit(AttemptResult.GOOD_LIFT.value))
        .otherwise(pl.lit(AttemptResult.BAD_LIFT.value))
    )
    frame = (
        keyed.select(
            pl.Series("competition_attempt_id", attempt_ids, dtype=pl.String),
            pl.col(COMPETITION_ID_COLUMN).alias("competition_id"),
            pl.col(ATHLETE_ID_COLUMN).alias("athlete_id"),
            pl.col("lift"),
            pl.col("attempt_number"),
            pl.col("attempt_role"),
            # The column each cell came from names the attempt's position in the source
            # sequence, so the basis is the source's own ordering rather than PSD's.
            pl.lit(AttemptOrderBasis.SOURCE_EXPLICIT.value).alias("attempt_order_basis"),
            # The source publishes no per-attempt instant. Absent, not invented from the
            # meet date: the record's observed_at says which day the attempt happened on.
            pl.lit(None, dtype=pl.Datetime("us", "UTC")).alias("attempt_time"),
            pl.col("__attempt").alias("source_attempt_raw"),
            pl.col("__attempt").abs().alias("load_raw"),
            pl.lit("kg").alias("load_unit"),
            pl.col("__attempt").abs().alias("load_kg"),
            result.alias("result"),
            pl.col("attempt_number").eq(1).alias("is_opener"),
            pl.lit(None, dtype=pl.Datetime("us", "UTC")).alias("created_at"),
            pl.lit(None, dtype=pl.Datetime("us", "UTC")).alias("scheduled_at"),
            pl.lit(None, dtype=pl.Datetime("us", "UTC")).alias("performed_at"),
            _utc_midnight("Date", "__attempt_date"),
            pl.lit(None, dtype=pl.Datetime("us", "UTC")).alias("modified_at"),
            pl.lit(source_id).alias("source_id"),
            _source_row_key(pl.col(ROW_ORDINAL_COLUMN)).alias("source_record_key"),
            pl.lit(None, dtype=pl.String).alias("source_record_hash"),
            pl.lit(ingested_at).alias("ingested_at"),
            _empty_quality_flags().alias("quality_flags"),
            pl.lit(None, dtype=pl.String).alias("missingness_reason"),
        )
        .with_columns(pl.col("__attempt_date").alias("observed_at"))
        .drop("__attempt_date")
    )
    return _ordered("competition_attempt", frame)


def _build_reported_results(
    partition: pl.DataFrame, *, source_id: str, ingested_at: datetime
) -> pa.Table:
    """Return the canonical reported-result table for one athlete partition.

    Every value here is a *source-reported outcome*, recorded as such: ``is_derived`` is
    false for every row, because PSD derived none of them. A component lift the source
    did not publish is never manufactured, and a total the source published without its
    components is recorded on its own.
    """
    long = _melt(partition, _REPORTED_COLUMNS, value_alias="__value", column_alias="__column")
    if long.height == 0:
        return _empty_table("competition_reported_result")
    long = long.with_columns(
        pl.col("__value").str.strip_chars().cast(pl.Float64, strict=False).alias("__reported")
    ).filter(pl.col("__reported").is_not_null() & (pl.col("__reported") != 0.0))
    if long.height == 0:
        return _empty_table("competition_reported_result")
    keyed = long.with_columns(
        [
            _mapped("__column", _KIND_BY_REPORTED_COLUMN).alias("result_kind"),
            pl.col("__column").is_in(list(_BEST_LIFT_COLUMNS)).alias("__is_best"),
            pl.concat_str(
                [
                    _part(COMPETITION_ID_COLUMN, "__psd_ci"),
                    pl.lit(_UNIT_SEPARATOR),
                    _part("__column", "__psd_cn"),
                ]
            ).alias("__reported_key"),
        ]
    )
    identifiers = bulk_make_id(
        IdPrefix.COMPETITION_REPORTED_RESULT, keyed.get_column("__reported_key").to_list()
    )
    frame = keyed.select(
        pl.Series("competition_reported_result_id", identifiers, dtype=pl.String),
        pl.col(COMPETITION_ID_COLUMN).alias("competition_id"),
        pl.col(ATHLETE_ID_COLUMN).alias("athlete_id"),
        pl.col("result_kind"),
        pl.col("__reported").abs().alias("value"),
        # Best lifts and totals are masses; scoring-system points are dimensionless, and
        # saying so is what stops a consumer reading Dots as kilograms.
        pl.when(pl.col("__column").is_in(_MASS_RESULT_COLUMNS))
        .then(pl.lit("kg"))
        .otherwise(None)
        .alias("unit"),
        pl.col("__reported").alias("source_value_raw"),
        pl.col("__column").alias("result_source_field"),
        pl.when(pl.col("__is_best") & (pl.col("__reported") < 0.0))
        .then(pl.lit("failed_attempt_only"))
        .when(pl.col("__is_best"))
        .then(pl.lit("successful_best"))
        .otherwise(None)
        .alias("reported_best_semantics"),
        pl.lit(False).alias("is_derived"),
        pl.lit(None, dtype=pl.String).alias("derivation_note"),
        pl.lit(None, dtype=pl.Datetime("us", "UTC")).alias("created_at"),
        pl.lit(None, dtype=pl.Datetime("us", "UTC")).alias("scheduled_at"),
        pl.lit(None, dtype=pl.Datetime("us", "UTC")).alias("performed_at"),
        _utc_midnight("Date", "observed_at"),
        pl.lit(None, dtype=pl.Datetime("us", "UTC")).alias("modified_at"),
        pl.lit(source_id).alias("source_id"),
        _source_row_key(pl.col(ROW_ORDINAL_COLUMN)).alias("source_record_key"),
        pl.lit(None, dtype=pl.String).alias("source_record_hash"),
        pl.lit(ingested_at).alias("ingested_at"),
        _empty_quality_flags().alias("quality_flags"),
        pl.lit(None, dtype=pl.String).alias("missingness_reason"),
    )
    return _ordered("competition_reported_result", frame)


def _build_meets(staging: Path, *, source_id: str, ingested_at: datetime) -> pa.Table:
    """Return the canonical meet table.

    A meet is context, not an outcome, and its identity is not athlete-major, so it is
    built once from a streaming distinct over the staged identity columns. Distinct
    *raw* spellings that normalize to one identity survive as separate rows here, which is
    how a later pass can report a meet-identity variant instead of hiding it.
    """
    selected = [
        MEET_ID_COLUMN,
        "Date",
        "Federation",
        "ParentFederation",
        "MeetCountry",
        "MeetState",
        "MeetTown",
        "MeetName",
        "Sanctioned",
    ]
    scan = pl.scan_parquet(str(staging / "athlete" / "*.parquet")).select(selected)
    distinct = (
        scan.with_columns(_normalized_meet_expressions()).unique().collect(engine="streaming")
    )
    prepared = distinct.with_columns(
        [
            _utc_midnight("Date", "meet_date"),
            _nullable("Federation").alias("meet_federation"),
            _nullable("ParentFederation").alias("meet_parent_federation"),
            _nullable("MeetCountry").alias("meet_country"),
            _nullable("MeetState").alias("meet_state"),
            _nullable("MeetTown").alias("meet_town"),
            _nullable("MeetName").alias("meet_name"),
            _nullable("Sanctioned").alias("sanctioned_status_raw"),
        ]
    )
    keys = prepared.select(pl.concat_str(_meet_key_expressions(source_id)).alias("__meet_key"))
    meet_ids = bulk_make_id(IdPrefix.COMPETITION_MEET, keys.get_column("__meet_key").to_list())
    _require_staged_meet_identities(prepared, meet_ids)
    sanctioned = prepared.get_column("sanctioned_status_raw").to_list()
    frame = prepared.select(
        pl.Series("competition_meet_id", meet_ids, dtype=pl.String),
        pl.col("meet_name"),
        pl.col("meet_date"),
        pl.lit(EventTimePrecision.DATE_ONLY.value).alias("event_time_precision"),
        pl.col("meet_federation"),
        pl.col("meet_parent_federation"),
        pl.col("meet_country"),
        pl.col("meet_state"),
        pl.col("meet_town"),
        pl.col("sanctioned_status_raw"),
        pl.Series("is_sanctioned", _sanctioned_flags(sanctioned), dtype=pl.Boolean),
        pl.lit(None, dtype=pl.Datetime("us", "UTC")).alias("created_at"),
        pl.lit(source_id).alias("source_id"),
        pl.col(MEET_ID_COLUMN).alias("source_record_key"),
        pl.lit(None, dtype=pl.String).alias("source_record_hash"),
        pl.lit(ingested_at).alias("ingested_at"),
        _empty_quality_flags().alias("quality_flags"),
        pl.lit(None, dtype=pl.String).alias("missingness_reason"),
    )
    ordered = _ordered("competition_meet", frame)
    return _collapse_meet_variants(ordered)


def _require_staged_meet_identities(prepared: pl.DataFrame, meet_ids: Sequence[str]) -> None:
    """Refuse a meet table whose identities disagree with the staged rows'.

    The staged rows already carry the meet identity the staging pass derived, and every
    ``competition`` row references it. Recomputing the identity here must reproduce those
    values exactly. When it does not, the meet table and the competitions that point at it
    disagree about what a meet is -- and nothing in the build notices, because each table
    is separately well-formed and each digest is computed over its own rows.

    Raises:
        TransformError: A staged meet identity does not match the recomputed one.
    """
    staged = prepared.get_column(MEET_ID_COLUMN).to_list()
    mismatched = [
        (staged[index], derived)
        for index, derived in enumerate(meet_ids)
        if staged[index] != derived
    ]
    if mismatched:
        example_staged, example_derived = mismatched[0]
        msg = (
            f"{len(mismatched)} staged meet identities disagree with the ones this pass "
            f"derived from the same fields (for example staged {example_staged!r} vs "
            f"derived {example_derived!r}). Every competition referencing a disagreeing "
            "identity would point at a meet that does not exist, and both tables would "
            "still pass every digest."
        )
        raise TransformError(msg)


def _sanctioned_flags(values: Sequence[str | None]) -> list[bool | None]:
    """Return the sanctioned flag for each raw ``Sanctioned`` value."""
    return [sanctioned_flag(value) if value is not None else None for value in values]


def _collapse_meet_variants(table: pa.Table) -> pa.Table:
    """Collapse normalized variants of one meet identity into a single row.

    Two raw spellings that normalize to one identity are the same meet, and a meet table
    with two rows for it would break every foreign key pointing at it. Which spelling
    survives is fixed rather than chosen: the variant with the lexicographically smallest
    raw tuple, so the outcome depends on neither row order nor chunking.
    """
    rows = table.to_pylist()
    by_meet: dict[str, dict[str, object]] = {}
    for row in rows:
        identifier = str(row["competition_meet_id"])
        existing = by_meet.get(identifier)
        if existing is None or _variant_sort_key(row) < _variant_sort_key(existing):
            by_meet[identifier] = row
    collapsed = pa.Table.from_pylist([by_meet[key] for key in sorted(by_meet)], schema=table.schema)
    return canonical_order(
        collapsed, table_spec("competition_meet").order_by, table_name="competition_meet"
    )


_VARIANT_KEY_COLUMNS: Final[tuple[str, ...]] = (
    "meet_name",
    "meet_federation",
    "meet_parent_federation",
    "meet_country",
    "meet_state",
    "meet_town",
    "sanctioned_status_raw",
)


def _variant_sort_key(row: Mapping[str, object]) -> tuple[str, ...]:
    """Return the deterministic tiebreak used when collapsing meet variants."""
    return tuple(
        "" if row.get(name) is None else str(row.get(name)) for name in _VARIANT_KEY_COLUMNS
    )


def _empty_table(table_name: str) -> pa.Table:
    """Return an empty Arrow table carrying *table_name*'s canonical schema."""
    return empty_table(table_name)

    # --------------------------------------------------------------------------
    # orchestration
    # --------------------------------------------------------------------------

    """Return the version tag recorded on the transformation's lineage entry."""
    return "psd-comp-transform/1"


def _count_unrecognised(partition: pl.DataFrame, counters: BuildCounters) -> None:
    """Record distinct values no declared mapping covers, so new ones stay visible."""
    for column, mapping in (
        ("Sex", SEX_CATEGORY_BY_SOURCE),
        ("Event", EVENT_BY_SOURCE),
        ("Equipment", EQUIPMENT_CLASS_BY_SOURCE),
    ):
        observed = partition.get_column(column).unique().to_list()
        unknown = [value for value in observed if value.strip() and value.strip() not in mapping]
        counters.note_unrecognised(column, unknown)


def _unparseable_counts(partition: pl.DataFrame) -> dict[str, int]:
    """Return how many present values in a partition failed to convert to a number.

    A source value that is present but unreadable is counted, never silently nulled and
    forgotten: a review of the corpus has to be able to see that the export contained
    something PSD could not read.
    """
    body = _float("BodyweightKg", "__bw")
    tallied = partition.select(
        [
            _present_but_unreadable(column).sum().alias(f"{alias}_unreadable")
            for column, alias in (("Age", "age"), ("BodyweightKg", "bodyweight"))
        ]
        + [
            _unreadable_date().sum().alias("date_unreadable"),
            (body.is_not_null() & (body <= 0.0)).sum().alias("bodyweight_nonpositive"),
        ]
    )
    return {name: int(tallied.row(0, named=True)[name]) for name in tallied.columns}


def _present_but_unreadable(column: str) -> pl.Expr:
    """Return an expression flagging present, non-blank values that are not numbers."""
    present = pl.col(column).fill_null("")
    numeric = (
        pl.when(present.str.strip_chars() == "")
        .then(None)
        .otherwise(present.str.strip_chars().cast(pl.Float64, strict=False))
    )
    return (present != "") & numeric.is_null()


def _unreadable_date() -> pl.Expr:
    """Return an expression flagging meet dates that are not ``YYYY-MM-DD``."""
    present = pl.col("Date").fill_null("")
    parsed = (
        pl.when(present.str.strip_chars() == "")
        .then(None)
        .otherwise(present.str.strip_chars().str.strptime(pl.Date, "%Y-%m-%d", strict=False))
    )
    return (present != "") & parsed.is_null()


def _accumulate(counters: BuildCounters, key: str, count: int) -> None:
    """Add *count* to the counter named *key*."""
    if key == "age_unreadable":
        counters.age_values_unparseable += count
    elif key == "bodyweight_unreadable":
        counters.bodyweight_values_unparseable += count
    elif key == "date_unreadable":
        counters.dates_unparseable += count
    elif key == "bodyweight_nonpositive":
        counters.bodyweight_values_nonpositive += count


def _dataset_relative(snapshot: OpenPowerliftingSnapshot) -> Path:
    """Return the dataset directory for a pinned snapshot, keyed by its digest.

    The digest is the directory name, so two snapshots never overwrite one another. It is
    the same identity rule the acquisition layer uses.
    """
    return Path("canonical") / "psd_comp" / snapshot.archive_sha256


def _dataset_id(snapshot: OpenPowerliftingSnapshot) -> str:
    """Return the dataset identifier for a pinned snapshot."""
    return f"psd_comp_{snapshot.archive_sha256[:16]}"


def _transform_version() -> str:
    """Return the version tag recorded on the transformation's lineage entry."""
    return "psd-comp-transform/1"


def _lineage(
    snapshot: OpenPowerliftingSnapshot, *, ingested_at: datetime
) -> tuple[LineageEntry, ...]:
    """Return the transformation lineage recorded on the dataset."""
    commit = snapshot.code_commit or detect_git_state(Path.cwd()).commit
    return (
        LineageEntry(
            provenance_id=f"lin_{snapshot.archive_sha256[:24]}",
            dataset_id=_dataset_id(snapshot),
            parent_dataset_id=None,
            transform_name="openpowerlifting_to_canonical",
            transform_version=_transform_version(),
            code_commit=commit,
            config_sha256=None,
            schema_version=SCHEMA_VERSION.tag,
            random_seed=None,
            created_at=ingested_at,
            description=(
                "Deterministic conversion of one pinned OpenPowerlifting bulk snapshot into "
                "the canonical competition tables. Source identities are derived from the "
                "verbatim Name including its #N disambiguator; attempt signs, approximate "
                "ages, open-ended weight classes, and non-numeric participation codes keep "
                "their source semantics."
            ),
        ),
    )


@dataclass(frozen=True, slots=True)
class BuildRequest:
    """One request to build the PSD-COMP corpus.

    Attributes:
        csv_path: The pinned snapshot's extracted CSV.
        snapshot: The pinned snapshot the CSV belongs to. Its digest names the dataset
            directory, so two snapshots never overwrite one another.
        data_root: External PSD data root; resolved from ``PSD_DATA_ROOT`` otherwise.
        config: Build knobs.
        ingested_at: Timezone-aware UTC instant stamped on every canonical row. Defaults
            to now, which is the only non-deterministic input the build has, and it is
            provenance rather than content: it does not enter a content digest.
        keep_staging: Keep the staged rows after the build. They are only needed to
            rebuild without re-reading the CSV.
    """

    csv_path: Path
    snapshot: OpenPowerliftingSnapshot
    data_root: Path | None = None
    config: BuildConfig = field(default_factory=BuildConfig)
    ingested_at: datetime | None = None
    keep_staging: bool = True


def build_corpus(request: BuildRequest) -> BuildResult:
    """Build the canonical PSD-COMP corpus from one pinned snapshot.

    The build is deterministic for the same CSV bytes, schema version, and configuration.
    It reads the CSV exactly once, keeps at most one staged chunk plus one partition's
    canonical tables in memory, and writes every canonical table with the same pinned
    Parquet profile the rest of PSD uses.

    The declared source contract is enforced before any row is read. Without that check a
    drifted header would be read with the transform's own expectations: a new column would
    be inferred and then ignored, and a removed one would become nulls across a
    multi-million-row corpus -- either way producing an artifact that claims to be the
    same corpus while quietly describing a different one.

    Args:
        request: What to build, and where.

    Returns:
        The build outcome, including the persisted manifest.

    Raises:
        SourceSchemaError: The snapshot's header does not match the declared contract.
        TransformError: The source cannot be converted faithfully.
    """
    csv_path = request.csv_path
    snapshot = request.snapshot
    settings = request.config
    require_source_schema(read_source_header(csv_path))
    started = time.perf_counter()
    stamp = request.ingested_at if request.ingested_at is not None else datetime.now(tz=UTC)
    source = openpowerlifting_source_record(snapshot, ingested_at=stamp)
    source_id = source.source_id

    relative = _dataset_relative(snapshot)
    staging = resolve_within_data_root(
        Path("processed") / "psd_comp" / snapshot.archive_sha256,
        data_root=request.data_root,
        create=True,
    )
    counters = BuildCounters()

    stage_snapshot(csv_path, staging, source_id=source_id, config=settings, counters=counters)

    partitions = staged_partitions(staging)
    if not partitions:
        msg = f"Staging produced no partitions for {csv_path}; the source appears to be empty."
        raise TransformError(msg)

    writer = StreamingDatasetWriter(relative, data_root=request.data_root)
    meets = _build_meets(staging, source_id=source_id, ingested_at=stamp)
    batch_rows = settings.batch_rows

    def build(pass_index: int, _label: str, path: Path) -> None:
        """Feed one partition's canonical tables for *pass_index*."""
        counting = pass_index == 0
        partition = _read_partition(path)
        if counting:
            _count_unrecognised(partition, counters)
            for key, count in _unparseable_counts(partition).items():
                _accumulate(counters, key, count)
        athletes, links = _build_athletes(partition, source_id=source_id, ingested_at=stamp)
        _feed(writer, "athlete", athletes, counting=counting, rows=batch_rows)
        _feed(writer, "athlete_source_link", links, counting=counting, rows=batch_rows)
        _feed(
            writer,
            "competition",
            _build_competitions(partition, source_id=source_id, ingested_at=stamp),
            counting=counting,
            rows=batch_rows,
        )
        _feed(
            writer,
            "competition_attempt",
            _build_attempts(partition, source_id=source_id, ingested_at=stamp, counters=counters),
            counting=counting,
            rows=batch_rows,
        )
        _feed(
            writer,
            "competition_reported_result",
            _build_reported_results(partition, source_id=source_id, ingested_at=stamp),
            counting=counting,
            rows=batch_rows,
        )

    for pass_index in range(2):
        counting = pass_index == 0
        _feed(writer, "competition_meet", meets, counting=counting, rows=batch_rows)
        for _label, path in partitions:
            build(pass_index, _label, path)
        if counting:
            # The counting pass is complete, so each table's digest can now open over a
            # header that declares the final row count.
            writer.open()

    manifest = writer.close(
        StreamingDatasetSpec(
            dataset_id=_dataset_id(snapshot),
            dataset_kind=DatasetKind.COMPETITION_HISTORY,
            created_at=stamp,
            sources=(source,),
            lineage=_lineage(snapshot, ingested_at=stamp),
            dataset_name="PSD-COMP: OpenPowerlifting longitudinal competition corpus",
            notes=(
                f"Pinned OpenPowerlifting snapshot {snapshot.archive_sha256} "
                f"(CSV {snapshot.csv_sha256}, {snapshot.row_count} source rows) converted "
                f"deterministically by {_transform_version()}."
            ),
        )
    )
    if not request.keep_staging:
        shutil.rmtree(staging, ignore_errors=True)
    return BuildResult(
        manifest=manifest,
        counters=counters,
        staging_directory=staging,
        build_seconds=time.perf_counter() - started,
        row_counts={artifact.name: artifact.row_count for artifact in manifest.artifacts},
    )


def corpus_table_names() -> tuple[str, ...]:
    """Return the canonical tables a PSD-COMP build populates."""
    return (
        "athlete",
        "athlete_source_link",
        "competition",
        "competition_attempt",
        "competition_meet",
        "competition_reported_result",
    )


def corpus_table_specs() -> tuple[TableSpec, ...]:
    """Return the registry specs for the tables a PSD-COMP build populates."""
    return tuple(table_spec(name) for name in corpus_table_names())


def all_table_names() -> tuple[str, ...]:
    """Return every canonical table name, including those a corpus build leaves empty."""
    return table_names()
