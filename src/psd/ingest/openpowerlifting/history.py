"""Longitudinal athlete histories, derived from canonical PSD-COMP.

What this is
------------

A one-row-per-source-identity summary of how a lifter's competition record accumulates:
when it starts, when it ends, how many meets it spans, which lifts were contested, how
much of it is observed at attempt detail, and which federations and equipment categories
it spans.

What this is not
----------------

**Not a second event representation.** The canonical competition, attempt, and
reported-result tables remain the only description of what happened at a meet. Every
number here is a projection over them, recomputed from the persisted tables rather than
accumulated during the build. That distinction is enforced structurally, not by
convention:

* the history is written outside ``tables/``, with its own manifest, so it can never be
  mistaken for a canonical table or enter the canonical schema version;
* the artifact records the canonical dataset and that dataset's manifest digest inside
  its own Arrow metadata, so the file itself says it is a projection;
* the history is derived by a *separate* pass that reads the built corpus, so it cannot
  drift from the events it summarises even if a future build changes.

Identity is not solved here
---------------------------

Longitudinal continuity is the most tempting way to "fix" an ambiguous identity: two
source rows share a name, they share a meet, and joining their rows makes the record
look coherent. That is exactly the reasoning this module refuses.

The history is therefore a history *of the source identity PSD can justify*, which is
not necessarily one biological person across all time. An athlete whose source name
carries a ``#N`` disambiguator is that disambiguated identity and no other, and a name
the source reports under two sex categories stays one flagged identity rather than two
confident ones. ``identity_status`` and ``ambiguity_group_id`` travel on every history
row for exactly this reason: a consumer that groups histories must be able to see
which rows rest on an identity the source itself could not settle.

Ordering
--------

An athlete's competitions are ordered ``competition_date``, ``competition_meet_id``,
``competition_id``. That total order is what resolves "first" and "last" -- two entries
on the same meet date are ordered by meet and then by entry, so the boundary entries are
reproducible rather than whichever row the engine happened to see. The persisted
artifact is itself ordered ``athlete_id``, its primary key.

Field semantics
---------------

``meet_count``
    Distinct ``competition_meet_id`` values. Two entries by one source identity at one
    meet are two participations but one meet, which is why both ``meet_count`` and
    ``competition_count`` are reported.
``squat_event_count`` / ``bench_event_count`` / ``deadlift_event_count``
    Participations whose *declared event* contested that lift. An event says which lifts
    were contested at all, so a lifter entered in ``BD`` contributes to the bench and
    deadlift counts and not the squat count. These are not attempt counts and not
    best-lift counts.
``attempt_detail_meet_count`` / ``attempt_detail_fraction``
    Meets at which at least one attempt exists, and that count over ``meet_count``.
    Most federations publish only best lifts, so this is the field that says how much of
    a history is observed rather than summarised, and a fraction near zero is the normal
    case rather than a defect.
``bodyweight_observation_count`` / ``age_observation_count``
    Participations that actually carried the value. Absence is never counted as an
    observation and never as zero.
``federation_count`` / ``parent_federation_count`` / ``equipment_category_count``
    Distinct non-absent values. A lifter who competes under several affiliates has a
    count above one; ``equipment_category_count`` counts *competition categories*, which
    say what the rules allowed rather than what the lifter wore.
``reported_total_count``
    Participations carrying a source-reported ``TotalKg``. It counts reported totals
    only: PSD derives none, so a competition without a reported total contributes nothing
    and no component lift is ever invented to make one.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import polars as pl
import pyarrow as pa

from psd.paths import resolve_within_data_root
from psd.provenance.environment import detect_git_state
from psd.provenance.manifest import DatasetManifest, LineageEntry
from psd.schema.version import SCHEMA_VERSION
from psd.serialization.dataset import artifact_paths, read_manifest
from psd.serialization.derived import (
    DerivedArtifactManifest,
    DerivedParent,
    DerivedTableSpec,
    write_derived_table,
)

__all__ = (
    "ATHLETE_HISTORY_ARTIFACT",
    "ATHLETE_HISTORY_DIRNAME",
    "ATHLETE_HISTORY_MANIFEST",
    "ATHLETE_HISTORY_SCHEMA",
    "ATHLETE_HISTORY_SPEC",
    "HISTORY_TRANSFORM_NAME",
    "HISTORY_TRANSFORM_VERSION",
    "CompetitionEventMembership",
    "HistoryBuildError",
    "HistoryBuildResult",
    "HistoryRequest",
    "athlete_history_spec",
    "build_athlete_history",
    "lifts_of_declared_event",
)

#: Logical artifact name. Distinct from every canonical table name, which is what keeps
#: a projection from being read as one more table of the corpus.
ATHLETE_HISTORY_ARTIFACT: Final[str] = "athlete_history"

#: Where derived artifacts live inside a dataset directory. Outside ``tables/`` on
#: purpose: the canonical table set is uniform and closed, and a derived projection
#: joining it would change what ``psd canonical verify`` means.
ATHLETE_HISTORY_DIRNAME: Final[str] = "derived"

ATHLETE_HISTORY_MANIFEST: Final[str] = "athlete_history.manifest.json"

HISTORY_TRANSFORM_NAME: Final[str] = "openpowerlifting_longitudinal_history"
HISTORY_TRANSFORM_VERSION: Final[str] = "psd-comp-history/1"

#: The declared-event membership, as data. It mirrors
#: :class:`psd.schema.vocabulary.CompetitionEvent` without importing the enum: the
#: corpus is read as a projection of a snapshot, and a vocabulary member added after the
#: fact must show up as an event with no counted lifts rather than as a crash.
CompetitionEventMembership: Final[Mapping[str, frozenset[str]]] = {
    "sbd": frozenset({"squat", "bench", "deadlift"}),
    "bd": frozenset({"bench", "deadlift"}),
    "sd": frozenset({"squat", "deadlift"}),
    "sb": frozenset({"squat", "bench"}),
    "s": frozenset({"squat"}),
    "b": frozenset({"bench"}),
    "d": frozenset({"deadlift"}),
}

_LIFTS: Final[tuple[str, str, str]] = ("squat", "bench", "deadlift")

_LIFT_BY_COMPETITION: Final[Mapping[str, frozenset[str]]] = {
    lift: frozenset(event for event, lifts in CompetitionEventMembership.items() if lift in lifts)
    for lift in _LIFTS
}


def lifts_of_declared_event(event: str | None) -> frozenset[str]:
    """Return the lifts a declared competition event contests.

    Args:
        event: A canonical ``competition_event`` value, possibly null or unmapped.

    Returns:
        The contested lifts. An event PSD does not recognise contests nothing known,
        which is a visible zero rather than a guess.
    """
    return CompetitionEventMembership.get(event or "", frozenset())


ATHLETE_HISTORY_SCHEMA: Final[pa.Schema] = pa.schema(
    [
        ("athlete_id", pa.string()),
        ("identity_status", pa.string()),
        ("ambiguity_group_id", pa.string()),
        ("sex_category", pa.string()),
        ("first_observed_meet_date", pa.timestamp("us", tz="UTC")),
        ("last_observed_meet_date", pa.timestamp("us", tz="UTC")),
        ("first_competition_id", pa.string()),
        ("last_competition_id", pa.string()),
        ("meet_count", pa.int64()),
        ("competition_count", pa.int64()),
        ("observed_span_days", pa.int64()),
        ("squat_event_count", pa.int64()),
        ("bench_event_count", pa.int64()),
        ("deadlift_event_count", pa.int64()),
        ("event_category_count", pa.int64()),
        ("attempt_detail_meet_count", pa.int64()),
        ("attempt_detail_fraction", pa.float64()),
        ("bodyweight_observation_count", pa.int64()),
        ("age_observation_count", pa.int64()),
        ("federation_count", pa.int64()),
        ("parent_federation_count", pa.int64()),
        ("equipment_category_count", pa.int64()),
        ("reported_total_count", pa.int64()),
        ("drug_tested_category_count", pa.int64()),
    ]
)


def athlete_history_spec() -> DerivedTableSpec:
    """Return the declared shape of the longitudinal athlete-history artifact."""
    return DerivedTableSpec.build(
        name=ATHLETE_HISTORY_ARTIFACT,
        arrow_schema=ATHLETE_HISTORY_SCHEMA,
        primary_key=("athlete_id",),
        summary=(
            "One row per canonical source identity: the span and coverage of its "
            "competition record, derived from canonical competition, attempt, and "
            "reported-result tables."
        ),
    )


ATHLETE_HISTORY_SPEC: Final[DerivedTableSpec] = athlete_history_spec()


class HistoryBuildError(RuntimeError):
    """Raised when a longitudinal history cannot be derived faithfully."""


@dataclass(frozen=True, slots=True)
class HistoryRequest:
    """One request to derive longitudinal athlete histories.

    Attributes:
        dataset_dir: The built PSD-COMP dataset, relative to the data root.
        data_root: External PSD data root; resolved from ``PSD_DATA_ROOT`` otherwise.
        created_at: Timezone-aware UTC instant stamped on the derived manifest. Defaults
            to now, and it is provenance rather than content: it does not enter the
            artifact's content digest.
        output_dir: Where to write the derived artifact; defaults to
            ``<dataset_dir>/derived``.
    """

    dataset_dir: Path
    data_root: Path | None = None
    created_at: datetime | None = None
    output_dir: Path | None = None


@dataclass(frozen=True, slots=True)
class HistoryBuildResult:
    """Outcome of one history derivation.

    Attributes:
        manifest: The persisted derived-artifact manifest.
        table: The derived artifact, already in canonical order.
        artifact_path: Where the artifact was written.
        build_seconds: Wall-clock seconds the derivation took.
    """

    manifest: DerivedArtifactManifest
    table: pa.Table
    artifact_path: Path
    build_seconds: float


def _history_path(request: HistoryRequest) -> Path:
    """Return the artifact path for *request*."""
    base = (
        request.output_dir
        if request.output_dir is not None
        else Path(request.dataset_dir) / ATHLETE_HISTORY_DIRNAME
    )
    return resolve_within_data_root(base, data_root=request.data_root, create=True) / (
        f"{ATHLETE_HISTORY_ARTIFACT}.parquet"
    )


def _history_manifest_path(request: HistoryRequest) -> Path:
    """Return the derived manifest path for *request*."""
    base = (
        request.output_dir
        if request.output_dir is not None
        else Path(request.dataset_dir) / ATHLETE_HISTORY_DIRNAME
    )
    return resolve_within_data_root(base, data_root=request.data_root, create=True) / (
        ATHLETE_HISTORY_MANIFEST
    )


def _scan(paths: Mapping[str, Path], table: str, columns: Sequence[str]) -> pl.LazyFrame:
    """Return a lazy scan of *columns* of one canonical artifact.

    Only the named columns are read. A projection is what keeps this pass bounded: the
    canonical tables hold every provenance and temporal column a consumer may need, and
    a summary needs a fraction of them.
    """
    return pl.scan_parquet(paths[table]).select(list(columns))


#: The columns a history needs from ``competition``. Deliberately short: identity is not
#: among them, because an athlete's identity is a row of its own, and duplicating it here
#: would be a second place for it to be wrong.
_COMPETITION_COLUMNS: Final[tuple[str, ...]] = (
    "athlete_id",
    "competition_id",
    "competition_meet_id",
    "competition_date",
    "competition_event",
    "bodyweight_kg",
    "age_reported",
    "federation",
    "sanctioning_body",
    "equipment_class_raw",
    "is_drug_tested_category",
)

#: The total order that resolves an athlete's first and last observed meet. The meet
#: and the entry break ties on a shared date, so "first" is a fact about the corpus
#: rather than a fact about row order.
_EVENT_TIME_ORDER: Final[tuple[str, str, str]] = (
    "competition_date",
    "competition_meet_id",
    "competition_id",
)


def _sorted_by_event_time(column: str) -> pl.Expr:
    """Return *column*'s values ordered by the canonical event-time total order."""
    return pl.col(column).sort_by(*_EVENT_TIME_ORDER, nulls_last=True)


def _competition_aggregate(paths: Mapping[str, Path]) -> pl.LazyFrame:
    """Return per-athlete aggregates over the canonical competition table."""
    ordered = _scan(paths, "competition", _COMPETITION_COLUMNS)
    aggregations: list[pl.Expr] = [
        pl.col("competition_date").min().alias("first_observed_meet_date"),
        pl.col("competition_date").max().alias("last_observed_meet_date"),
        _sorted_by_event_time("competition_id").first().alias("first_competition_id"),
        _sorted_by_event_time("competition_id").last().alias("last_competition_id"),
        pl.col("competition_meet_id").drop_nulls().n_unique().alias("meet_count"),
        pl.len().cast(pl.Int64).alias("competition_count"),
        *(  # An event says which lifts were contested, not which were attempted.
            pl.col("competition_event")
            .is_in(sorted(_LIFT_BY_COMPETITION[lift]))
            .sum()
            .cast(pl.Int64)
            .alias(f"{lift}_event_count")
            for lift in _LIFTS
        ),
        pl.col("competition_event").drop_nulls().n_unique().alias("event_category_count"),
        pl.col("bodyweight_kg")
        .is_not_null()
        .sum()
        .cast(pl.Int64)
        .alias("bodyweight_observation_count"),
        pl.col("age_reported").is_not_null().sum().cast(pl.Int64).alias("age_observation_count"),
        pl.col("federation").drop_nulls().n_unique().alias("federation_count"),
        pl.col("sanctioning_body").drop_nulls().n_unique().alias("parent_federation_count"),
        pl.col("equipment_class_raw").drop_nulls().n_unique().alias("equipment_category_count"),
        pl.col("is_drug_tested_category")
        .fill_null(False)
        .sum()
        .cast(pl.Int64)
        .alias("drug_tested_category_count"),
    ]
    return ordered.group_by("athlete_id", maintain_order=True).agg(aggregations)


def _attempt_detail_meets(paths: Mapping[str, Path]) -> pl.LazyFrame:
    """Return, per athlete, how many meets carry at least one attempt.

    Attempts are collapsed to their competition first. A meet counts once however many
    attempts happened there, because the question is *where the source reported attempt
    detail*, not how much of it there was -- and the two are different: a lifter with
    nine attempts at one meet and a lifter with three each at three meets both have
    attempt detail at exactly one meet.
    """
    attempt_competitions = _scan(paths, "competition_attempt", ("athlete_id", "competition_id"))
    competitions = _scan(paths, "competition", ("competition_id", "competition_meet_id"))
    return (
        attempt_competitions.unique()
        .join(competitions, on="competition_id", how="inner")
        .group_by("athlete_id")
        .agg(
            pl.col("competition_meet_id").drop_nulls().n_unique().alias("attempt_detail_meet_count")
        )
    )


def _reported_totals(paths: Mapping[str, Path]) -> pl.LazyFrame:
    """Return, per athlete, how many participations carry a source-reported total."""
    return (
        _scan(paths, "competition_reported_result", ("athlete_id", "result_kind"))
        .filter(pl.col("result_kind") == "total")
        .group_by("athlete_id")
        .agg(pl.len().cast(pl.Int64).alias("reported_total_count"))
    )


def _identity(paths: Mapping[str, Path]) -> pl.LazyFrame:
    """Return the identity columns every history row travels with.

    An identity caveat that lives only in the athlete table can be lost by a consumer
    that reads a summary. These two columns travel with every history row so the caveat
    cannot be separated from the record it qualifies.
    """
    return _scan(
        paths,
        "athlete",
        ("athlete_id", "identity_status", "ambiguity_group_id", "sex_category"),
    )


def _derive(paths: Mapping[str, Path]) -> pa.Table:
    """Return the longitudinal athlete-history table for one built corpus.

    Every aggregate is a projection over canonical tables. Nothing here reconstructs an
    event, and nothing here uses continuity to settle an identity.
    """
    if not paths:
        msg = "Cannot derive athlete histories: the corpus manifest declares no artifacts."
        raise HistoryBuildError(msg)
    plan = (
        _competition_aggregate(paths)
        .join(_attempt_detail_meets(paths), on="athlete_id", how="left")
        .join(_reported_totals(paths), on="athlete_id", how="left")
        .join(_identity(paths), on="athlete_id", how="inner")
        .with_columns(
            [
                pl.col("attempt_detail_meet_count").fill_null(0).cast(pl.Int64),
                pl.col("reported_total_count").fill_null(0).cast(pl.Int64),
            ]
        )
        .with_columns(
            [
                # A history with no observed dates has no span, not a span of zero: zero
                # days would claim two things happened on one day that never happened.
                (
                    (pl.col("last_observed_meet_date") - pl.col("first_observed_meet_date"))
                    .dt.total_days()
                    .cast(pl.Int64)
                    .alias("observed_span_days")
                ),
                (
                    pl.when(pl.col("meet_count") > 0)
                    .then(
                        pl.col("attempt_detail_meet_count").cast(pl.Float64)
                        / pl.col("meet_count").cast(pl.Float64)
                    )
                    .otherwise(0.0)
                    .alias("attempt_detail_fraction")
                ),
            ]
        )
        .select(list(ATHLETE_HISTORY_SCHEMA.names))
    )
    collected = plan.collect(engine="streaming")
    return ATHLETE_HISTORY_SPEC.order(collected.to_arrow().cast(ATHLETE_HISTORY_SCHEMA))


def _lineage(
    manifest: DatasetManifest, *, created_at: datetime, row_count: int
) -> tuple[LineageEntry, ...]:
    """Return the lineage recorded for the derived artifact.

    The transform records the canonical corpus it read and the row count it produced, so
    a reader can tell "the history is stale" from "the history is empty" without
    re-deriving either.
    """
    commit = manifest.lineage[0].code_commit if manifest.lineage else None
    if commit is None:
        commit = detect_git_state(Path.cwd()).commit
    return (
        LineageEntry(
            provenance_id=f"hist_{manifest.dataset_id[:32]}",
            dataset_id=f"{manifest.dataset_id}_history",
            parent_dataset_id=manifest.dataset_id,
            transform_name=HISTORY_TRANSFORM_NAME,
            transform_version=HISTORY_TRANSFORM_VERSION,
            code_commit=commit,
            config_sha256=None,
            schema_version=SCHEMA_VERSION.tag,
            random_seed=None,
            created_at=created_at,
            description=(
                f"Longitudinal summary of {row_count} canonical source identities, derived "
                f"from PSD-COMP {manifest.dataset_id} by aggregating the canonical "
                "competition, attempt, and reported-result tables. No event is "
                "reconstructed and no ambiguous identity is resolved."
            ),
        ),
    )


def build_athlete_history(request: HistoryRequest) -> HistoryBuildResult:
    """Derive and persist longitudinal athlete histories for a built corpus.

    Args:
        request: Which corpus to summarise, and where to write the result.

    Returns:
        The derived artifact, its manifest, and how long the derivation took.

    Raises:
        HistoryBuildError: The corpus manifest declares none of the tables a history
            needs, so the projection would be fabricated rather than derived.
    """
    started = time.perf_counter()
    stamp = request.created_at if request.created_at is not None else datetime.now(tz=UTC)
    manifest = read_manifest(request.dataset_dir, data_root=request.data_root)
    paths = artifact_paths(manifest, request.dataset_dir, data_root=request.data_root)
    required = {"competition", "competition_attempt", "competition_reported_result", "athlete"}
    missing = sorted(required - set(paths))
    if missing:
        msg = (
            f"PSD-COMP corpus {manifest.dataset_id} does not declare the canonical "
            f"table(s) a history needs: {', '.join(missing)}."
        )
        raise HistoryBuildError(msg)

    table = _derive(paths)
    artifact_path = _history_path(request)
    written = write_derived_table(
        table,
        artifact_path,
        ATHLETE_HISTORY_SPEC,
        parent=DerivedParent.of(manifest),
    )
    derived_manifest = DerivedArtifactManifest(
        artifact_name=ATHLETE_HISTORY_ARTIFACT,
        artifact_relative_path=f"{ATHLETE_HISTORY_DIRNAME}/{written.path.name}",
        artifact_sha256=written.sha256,
        artifact_content_sha256=written.content_sha256,
        artifact_byte_size=written.byte_size,
        row_count=written.row_count,
        ordering=ATHLETE_HISTORY_SPEC.order_by,
        parent_dataset_id=manifest.dataset_id,
        parent_manifest_digest=DerivedParent.of(manifest).manifest_digest,
        lineage=_lineage(manifest, created_at=stamp, row_count=written.row_count),
        created_at=stamp,
        notes=(
            "Derived projection of canonical competition events. It is not an independent "
            "observation set: a history row is a summary of the canonical tables named in "
            f"the artifact metadata, and {ATHLETE_HISTORY_ARTIFACT} rows are histories of "
            "the source identity PSD can justify, not of proven biological continuity."
        ),
    )
    derived_manifest.write_json(_history_manifest_path(request))
    return HistoryBuildResult(
        manifest=derived_manifest,
        table=table,
        artifact_path=artifact_path,
        build_seconds=time.perf_counter() - started,
    )
