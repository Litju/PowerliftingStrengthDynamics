"""``psd openpowerlifting`` -- acquire, build, audit, and verify PSD-COMP.

The command surface is orchestration over importable APIs and nothing else: canonical
schema, validation, and corpus semantics live in :mod:`psd.ingest.openpowerlifting` and
:mod:`psd.serialization`, and a reader who needs behaviour this file does not show can get
it from Python. Every command here is a thin shell over one call.

Five commands, in the order an operator runs them:

``acquire``
    Download and pin the official bulk snapshot, or pin a local ``.zip`` or ``.csv``.
``inspect``
    State what a pinned snapshot is: its digest, its date and revision, its size, its row
    count, and whether its header still matches the declared contract. Runs before a
    multi-hour build so a drifted snapshot is discovered in a second rather than an hour.
``build``
    Convert a pinned snapshot into the canonical corpus and derive the longitudinal
    athlete histories from the persisted tables.
``audit``
    Produce the durable machine-readable corpus audit and its human-readable summary.
``verify``
    Check the pinned source and the canonical artifacts against their recorded digests,
    their declared primary keys, and the expansion invariants.

Exit codes are the contract with a pipeline: ``0`` for success, ``1`` for a failed
acquisition, schema, build, or verification, and ``2`` for a usage or layout error. A
command that cannot do its job exits non-zero rather than printing a warning.

No test requires the network. ``acquire`` reaches it only when asked for a URL and given
no local file, and every other command works entirely from a pinned local snapshot.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import NoReturn

import typer

from psd.ingest.openpowerlifting.acquire import (
    OPENPOWERLIFTING_BULK_URL,
    AcquisitionError,
    acquire_snapshot,
    acquire_snapshot_from_local_file,
    read_pinned_snapshot,
    resolve_snapshot_csv,
    snapshot_directory,
)
from psd.ingest.openpowerlifting.audit import (
    AuditError,
    AuditRequest,
    audit_corpus,
    corpus_expansion_invariants,
)
from psd.ingest.openpowerlifting.contract import SourceSchemaError, review_source_schema
from psd.ingest.openpowerlifting.history import (
    HistoryBuildError,
    HistoryRequest,
    build_athlete_history,
)
from psd.ingest.openpowerlifting.snapshot import OpenPowerliftingSnapshot
from psd.ingest.openpowerlifting.transform import (
    BuildConfig,
    BuildRequest,
    TransformError,
    build_corpus,
)
from psd.paths import (
    DATA_ROOT_ENV_VAR,
    DataRootError,
    resolve_data_root,
    resolve_within_data_root,
)
from psd.serialization.dataset import DatasetLayoutError, verify_dataset
from psd.serialization.derived import DerivedArtifactError, read_derived_manifest
from psd.serialization.parquet import sha256_file

app = typer.Typer(
    name="openpowerlifting",
    help="Acquire, build, audit, and verify the PSD-COMP competition corpus.",
    no_args_is_help=True,
)

#: The dataset directory a pinned snapshot builds into, relative to the data root.
DATASET_PREFIX = "canonical/psd_comp"

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE = 2


def _data_root(explicit: Path | None) -> Path:
    """Resolve the data root or exit with a clear message."""
    try:
        return resolve_data_root(explicit)
    except DataRootError as error:
        typer.secho(str(error), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=EXIT_USAGE) from error


def _fail(message: str, *, code: int = EXIT_FAILED) -> NoReturn:
    """Print *message* as an error and exit.

    Every failure path goes through here, so a command's failure always looks the same and
    always lands on a stream a pipeline can capture. It exits rather than returning, so no
    call site can forget to raise and carry on past a failure as though it had not
    happened.
    """
    typer.secho(message, fg=typer.colors.RED, err=True)
    raise typer.Exit(code=code)


def _dataset_dir(archive_sha256: str) -> str:
    """Return the dataset directory a snapshot digest builds into."""
    return f"{DATASET_PREFIX}/{archive_sha256}"


def _load_snapshot(
    *, digest: str | None, source: Path | None, data_root: Path
) -> OpenPowerliftingSnapshot:
    """Return the pinned snapshot named by *digest* or *source*.

    Raises:
        typer.Exit: Neither was given, or the named snapshot is not pinned.
    """
    if digest is None and source is None:
        _fail(
            "Name a pinned snapshot with --digest, or a local file with --source.",
            code=EXIT_USAGE,
        )
    try:
        if digest is not None:
            return read_pinned_snapshot(
                snapshot_directory(digest, data_root=data_root), data_root=data_root
            )
        if source is None:
            _fail("No source file was given.", code=EXIT_USAGE)
        return acquire_snapshot_from_local_file(source, data_root=data_root)
    except AcquisitionError as error:
        _fail(str(error), code=EXIT_USAGE)


@app.command("acquire")
def acquire(
    source: Path | None = typer.Option(
        None,
        "--source",
        help="Local .zip or .csv to pin instead of downloading. Required for offline use.",
    ),
    url: str = typer.Option(
        OPENPOWERLIFTING_BULK_URL, "--url", help="Bulk download URL when not using --source."
    ),
    digest: str | None = typer.Option(
        None, "--digest", help="Report an already-pinned snapshot instead of acquiring."
    ),
    data_root: Path | None = typer.Option(
        None, "--data-root", help=f"Override {DATA_ROOT_ENV_VAR}."
    ),
    as_json: bool = typer.Option(False, "--json", help="Emit a JSON summary."),
) -> None:
    """Pin the OpenPowerlifting bulk snapshot by digest.

    \b
    The published filename is mutable and is never the identity: a snapshot
    is named by the SHA-256 of its bytes, so two nightlies are two corpora and
    neither overwrites the other. Re-running against unchanged bytes reuses the
    existing pin rather than writing a second copy.
    \b
    Pass --source to pin a local .zip or .csv without touching the network.
    """
    root = _data_root(data_root)
    try:
        if digest is not None:
            snapshot = read_pinned_snapshot(
                snapshot_directory(digest, data_root=root), data_root=root
            )
        elif source is not None:
            snapshot = acquire_snapshot_from_local_file(source, data_root=root)
        else:
            snapshot = acquire_snapshot(data_root=root, url=url)
    except AcquisitionError as error:
        _fail(str(error))
    _emit_snapshot(snapshot, as_json=as_json)


def _emit_snapshot(snapshot: OpenPowerliftingSnapshot, *, as_json: bool) -> None:
    """Print a pinned snapshot's identity and licensing."""
    review = review_source_schema(snapshot.source_columns)
    if as_json:
        typer.echo(
            json.dumps(
                {
                    "archive_sha256": snapshot.archive_sha256,
                    "archive_byte_size": snapshot.archive_byte_size,
                    "csv_member_name": snapshot.csv_member_name,
                    "csv_sha256": snapshot.csv_sha256,
                    "csv_byte_size": snapshot.csv_byte_size,
                    "row_count": snapshot.row_count,
                    "dataset_version": snapshot.dataset_version(),
                    "service": snapshot.service.model_dump(mode="json"),
                    "source_columns": list(snapshot.source_columns),
                    "schema_review": review.to_dict(),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    typer.echo(f"archive_sha256  {snapshot.archive_sha256}")
    typer.echo(f"archive_bytes   {snapshot.archive_byte_size:,}")
    typer.echo(f"csv_sha256      {snapshot.csv_sha256}")
    typer.echo(f"csv_bytes       {snapshot.csv_byte_size:,}")
    typer.echo(f"csv_member      {snapshot.csv_member_name}")
    typer.echo(f"rows            {snapshot.row_count:,}")
    typer.echo(f"version         {snapshot.dataset_version()}")
    typer.echo(f"updated         {snapshot.service.updated_date or 'not reported'}")
    typer.echo(f"revision        {snapshot.service.revision or 'not reported'}")
    typer.echo(f"url             {snapshot.source_url}")
    typer.echo(f"schema          {review.summary()}")
    for disagreement in snapshot.service.disagreements():
        typer.secho(f"  note: {disagreement}", fg=typer.colors.YELLOW, err=True)
    typer.echo(f"dataset         {_dataset_dir(snapshot.archive_sha256)}")
    if review.drifted:
        raise typer.Exit(code=EXIT_FAILED)


@app.command("inspect")
def inspect(
    digest: str | None = typer.Option(None, "--digest", help="Pinned snapshot digest."),
    source: Path | None = typer.Option(
        None, "--source", help="Local .zip or .csv to pin and inspect."
    ),
    data_root: Path | None = typer.Option(
        None, "--data-root", help=f"Override {DATA_ROOT_ENV_VAR}."
    ),
    as_json: bool = typer.Option(False, "--json", help="Emit a JSON summary."),
) -> None:
    """Report a pinned snapshot's identity, revision, header, and schema agreement.

    \b
    Run this before a multi-hour build, so a drifted snapshot is found in a
    second rather than an hour. Exits non-zero on drift.
    """
    root = _data_root(data_root)
    snapshot = _load_snapshot(digest=digest, source=source, data_root=root)
    _emit_snapshot(snapshot, as_json=as_json)


@app.command("build")
def build(
    digest: str | None = typer.Option(None, "--digest", help="Pinned snapshot digest."),
    source: Path | None = typer.Option(
        None, "--source", help="Local .zip or .csv to pin and build."
    ),
    chunk_rows: int = typer.Option(
        BuildConfig().chunk_rows, "--chunk-rows", min=1, help="Source rows read per staged chunk."
    ),
    batch_rows: int = typer.Option(
        BuildConfig().batch_rows,
        "--batch-rows",
        min=1,
        help="Rows per canonical batch handed to the writer.",
    ),
    partitions: int = typer.Option(
        BuildConfig().partitions,
        "--partitions",
        min=1,
        help="Athlete partitions; must divide sixteen.",
    ),
    keep_staging: bool = typer.Option(
        True,
        "--keep-staging/--drop-staging",
        help="Keep the staged rows after the build.",
    ),
    skip_histories: bool = typer.Option(
        False, "--skip-histories", help="Do not derive the longitudinal athlete histories."
    ),
    data_root: Path | None = typer.Option(
        None, "--data-root", help=f"Override {DATA_ROOT_ENV_VAR}."
    ),
    as_json: bool = typer.Option(False, "--json", help="Emit a JSON summary."),
) -> None:
    """Build the canonical PSD-COMP corpus and its longitudinal athlete histories.

    \b
    Reads the pinned CSV once and converts it into the canonical competition
    tables, then derives the athlete histories from the persisted tables in a
    separate pass, so the histories cannot drift from the events they summarise.
    """
    root = _data_root(data_root)
    snapshot = _load_snapshot(digest=digest, source=source, data_root=root)
    relative = _dataset_dir(snapshot.archive_sha256)
    try:
        csv_path = resolve_snapshot_csv(snapshot, data_root=root)
    except AcquisitionError as error:
        _fail(str(error))
    try:
        config = BuildConfig(chunk_rows=chunk_rows, batch_rows=batch_rows, partitions=partitions)
    except ValueError as error:
        # A configuration that cannot be honoured is a usage error, and it is worth finding
        # out in a second rather than after the staging pass has begun.
        _fail(str(error), code=EXIT_USAGE)
    try:
        result = build_corpus(
            BuildRequest(
                csv_path=csv_path,
                snapshot=snapshot,
                data_root=root,
                config=config,
                keep_staging=keep_staging,
            )
        )
    except (SourceSchemaError, TransformError) as error:
        _fail(str(error))

    history: dict[str, object] | None = None
    if not skip_histories:
        try:
            derived = build_athlete_history(
                HistoryRequest(dataset_dir=Path(relative), data_root=root)
            )
        except (
            DatasetLayoutError,
            DerivedArtifactError,
            HistoryBuildError,
        ) as error:
            _fail(str(error))
        history = {
            "rows": derived.manifest.row_count,
            "content_sha256": derived.manifest.artifact_content_sha256,
            "sha256": derived.manifest.artifact_sha256,
            "relative_path": derived.manifest.artifact_relative_path,
        }

    if as_json:
        typer.echo(
            json.dumps(
                {
                    "dataset_id": result.manifest.dataset_id,
                    "dataset_relative_path": relative,
                    "schema_version": result.manifest.schema_version,
                    "archive_sha256": snapshot.archive_sha256,
                    "csv_sha256": snapshot.csv_sha256,
                    "row_counts": result.row_counts,
                    "content_digests": {
                        artifact.name: artifact.content_sha256
                        for artifact in result.manifest.artifacts
                    },
                    "counters": result.counters.to_dict(),
                    "build_seconds": round(result.build_seconds, 3),
                    "athlete_history": history,
                },
                indent=2,
                sort_keys=True,
                default=str,
            )
        )
        return
    typer.echo(f"dataset        {result.manifest.dataset_id}")
    typer.echo(f"path           {relative}")
    typer.echo(f"schema         {result.manifest.schema_version}")
    typer.echo(f"source rows    {result.counters.source_rows:,}")
    typer.echo(f"build seconds  {result.build_seconds:.1f}")
    for table, count in sorted(result.row_counts.items()):
        if count:
            typer.echo(f"  {table:<32} {count:>12,}")
    typer.echo(f"athlete history  {history['rows']:,}" if history else "athlete history  skipped")
    typer.echo(f"audit           psd openpowerlifting audit --digest {snapshot.archive_sha256}")
    typer.echo(f"verify          psd openpowerlifting verify --digest {snapshot.archive_sha256}")


@app.command("audit")
def audit(
    digest: str | None = typer.Option(None, "--digest", help="Pinned snapshot digest."),
    data_root: Path | None = typer.Option(
        None, "--data-root", help=f"Override {DATA_ROOT_ENV_VAR}."
    ),
    memory_limit: str = typer.Option("4GB", "--memory-limit", help="DuckDB memory ceiling."),
    as_json: bool = typer.Option(False, "--json", help="Emit the report instead of a summary."),
) -> None:
    """Produce the durable corpus audit and its human-readable summary.

    \b
    Writes <dataset>/audit/corpus_audit.json and corpus_audit.md. Reports source
    irregularities without repairing them, and exits zero whenever it has a
    complete report, however unusual the corpus is.
    \b
    A failing expansion invariant is a verification failure, not an audit
    failure: use verify.
    """
    root = _data_root(data_root)
    if digest is None:
        _fail("Name a pinned snapshot with --digest.", code=EXIT_USAGE)
    try:
        result = audit_corpus(
            AuditRequest(
                dataset_dir=Path(_dataset_dir(digest)),
                data_root=root,
                archive_sha256=digest,
                memory_limit=memory_limit,
            )
        )
    except (AuditError, DatasetLayoutError, DataRootError) as error:
        _fail(str(error))
    if as_json:
        typer.echo(result.audit.to_json_bytes().decode("utf-8"), nl=False)
        return
    typer.echo(result.audit.summary())
    typer.echo("")
    typer.echo(f"report    {result.json_path}")
    typer.echo(f"document  {result.markdown_path}")
    typer.echo(f"seconds   {result.audit_seconds:.1f}")


@app.command("verify")
def verify(
    digest: str | None = typer.Option(None, "--digest", help="Pinned snapshot digest."),
    source: Path | None = typer.Option(
        None, "--source", help="Local .zip or .csv to pin and verify."
    ),
    skip_invariants: bool = typer.Option(
        False, "--skip-invariants", help="Check digests and keys only."
    ),
    data_root: Path | None = typer.Option(
        None, "--data-root", help=f"Override {DATA_ROOT_ENV_VAR}."
    ),
    as_json: bool = typer.Option(False, "--json", help="Emit a JSON summary."),
) -> None:
    """Verify the pinned source, the canonical artifacts, and the expansion invariants.

    \b
    Three independent checks, because they fail for different reasons.
    \b
    The source must still hash to what the snapshot recorded. Every canonical
    artifact must match both its recorded digests and hold a unique primary key.
    And the corpus must satisfy every expansion invariant, which is what catches
    a corpus that is internally consistent and semantically wrong.
    """
    root = _data_root(data_root)
    snapshot = _load_snapshot(digest=digest, source=source, data_root=root)
    relative = _dataset_dir(snapshot.archive_sha256)
    problems: list[str] = []

    try:
        source_problems = _verify_source(snapshot, data_root=root)
    except (AcquisitionError, DataRootError) as error:
        _fail(str(error), code=EXIT_USAGE)
    problems.extend(source_problems)

    try:
        result = verify_dataset(relative, data_root=root)
    except DatasetLayoutError as error:
        _fail(str(error), code=EXIT_USAGE)
    problems.extend(result.problems)

    derived_checked = False
    history_path = Path(relative) / "derived" / "athlete_history.manifest.json"
    try:
        resolved_history = resolve_within_data_root(history_path, data_root=root)
        history = read_derived_manifest(resolved_history)
        derived_checked = True
    except (DataRootError, DerivedArtifactError):
        history = None

    invariants: tuple[Mapping[str, object], ...] = ()
    if not skip_invariants:
        try:
            invariants, invariant_problems = _verify_invariants(
                relative, data_root=root, digest=snapshot.archive_sha256
            )
        except AuditError as error:
            # An unreadable artifact is reported as a problem rather than raised: the
            # digests above have already said what is wrong, and a verification command
            # that crashes instead of reporting is less useful than one that does not.
            invariants = ()
            invariant_problems = [str(error)]
        problems.extend(invariant_problems)

    payload = {
        "dataset_relative_path": relative,
        "source_ok": not source_problems,
        "artifacts_ok": result.artifacts_ok,
        "checked_artifacts": result.checked_artifacts,
        "athlete_history_checked": derived_checked,
        "athlete_history_rows": history.row_count if history else None,
        "invariants_checked": len(invariants),
        "invariants_failed": sum(1 for item in invariants if not item["holds"]),
        "ok": not problems,
        "problems": problems,
    }
    if as_json:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
    else:
        typer.echo(f"dataset          {relative}")
        typer.echo(f"source           {'OK' if not source_problems else 'PROBLEMS'}")
        typer.echo(
            f"artifacts        {result.checked_artifacts} checked"
            f" ({'OK' if result.artifacts_ok else 'PROBLEMS'})"
        )
        typer.echo(
            f"athlete history  {'checked' if derived_checked else 'not built'}"
            + (f", {history.row_count:,} rows" if history else "")
        )
        if skip_invariants:
            typer.echo("invariants       skipped")
        else:
            typer.echo(
                f"invariants       {len(invariants)} checked,"
                f" {payload['invariants_failed']} failing"
            )
        typer.echo(f"status           {'OK' if not problems else 'PROBLEMS'}")
        for problem in problems:
            typer.secho(f"  {problem}", fg=typer.colors.RED, err=True)
    if problems:
        raise typer.Exit(code=EXIT_FAILED)


def _verify_source(snapshot: OpenPowerliftingSnapshot, *, data_root: Path) -> list[str]:
    """Return problems with the pinned source itself."""
    problems: list[str] = []
    csv_path = resolve_snapshot_csv(snapshot, data_root=data_root)
    actual_csv = sha256_file(csv_path)
    if actual_csv != snapshot.csv_sha256:
        problems.append(
            f"source: CSV digest {actual_csv} does not match the pinned snapshot "
            f"{snapshot.csv_sha256}; the pinned bytes have changed since acquisition"
        )
    try:
        review = review_source_schema(snapshot.source_columns)
    except (TypeError, ValueError) as error:
        problems.append(f"source: unreadable header ({error})")
        return problems
    if review.drifted:
        problems.append(
            "source: header does not match the declared contract ("
            f"unknown={list(review.unknown)} missing={list(review.missing)}"
            f" reordered={review.reordered})"
        )
    return problems


def _verify_invariants(
    relative: str, *, data_root: Path, digest: str
) -> tuple[tuple[Mapping[str, object], ...], list[str]]:
    """Re-derive the expansion invariants against the persisted corpus.

    Invariants rather than digests, because a digest cannot see a semantic defect: the
    corpus that this check found most recently passed every digest it published.
    """
    invariants = corpus_expansion_invariants(relative, data_root=data_root, archive_sha256=digest)
    problems = [
        f"invariant {item.name} observed {item.observed}, expected {item.expected}:"
        f" {item.statement}"
        for item in invariants
        if not item.holds
    ]
    return tuple(item.to_dict() for item in invariants), problems


@app.callback()
def root() -> None:
    """PSD-COMP: the OpenPowerlifting competition corpus."""
