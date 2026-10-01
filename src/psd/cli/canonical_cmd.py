"""``psd canonical`` -- build, verify, and inspect canonical datasets."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import typer

from psd.paths import DATA_ROOT_ENV_VAR, DataRootError, resolve_data_root
from psd.provenance.manifest import DatasetKind
from psd.serialization.dataset import (
    DatasetLayoutError,
    DatasetWriteSpec,
    build_dataset,
    load_records,
    read_dataset,
    verify_dataset,
    write_dataset,
)

app = typer.Typer(
    name="canonical",
    help="Build, verify, and inspect canonical datasets.",
    no_args_is_help=True,
)

_KINDS: tuple[str, ...] = tuple(kind.value for kind in DatasetKind)


def _data_root(explicit: Path | None) -> Path:
    """Resolve the data root or exit with a clear message."""
    try:
        return resolve_data_root(explicit)
    except DataRootError as error:
        typer.secho(str(error), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from error


@app.command("build")
def build(
    input_dir: Path = typer.Option(..., "--input", help="Directory of <table>.json files."),
    output: Path = typer.Option(
        ..., "--output", help="Dataset directory, relative to the data root."
    ),
    dataset_id: str = typer.Option(..., "--dataset-id", help="Stable dataset identifier."),
    kind: str = typer.Option("training_history", "--kind", help=f"One of: {', '.join(_KINDS)}."),
    name: str | None = typer.Option(None, "--name", help="Human-readable dataset name."),
    data_root: Path | None = typer.Option(
        None, "--data-root", help=f"Override {DATA_ROOT_ENV_VAR}."
    ),
    description: str | None = typer.Option(None, "--description", help="Manifest notes."),
    as_json: bool = typer.Option(False, "--json", help="Emit a JSON summary."),
) -> None:
    """Build a canonical dataset from JSON record files and persist it."""
    root = _data_root(data_root)
    if kind not in _KINDS:
        typer.secho(
            f"Unknown dataset kind {kind!r}; choose one of: {', '.join(_KINDS)}.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=2)
    try:
        records = load_records(input_dir)
    except DatasetLayoutError as error:
        typer.secho(str(error), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from error

    dataset = build_dataset(records, dataset_id=dataset_id, created_at=datetime.now(tz=UTC))
    manifest = write_dataset(
        dataset,
        output,
        DatasetWriteSpec(
            dataset_kind=DatasetKind(kind),
            dataset_name=name,
            description=description,
        ),
        data_root=root,
    )
    if as_json:
        typer.echo(json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True))
        return
    typer.echo(f"dataset   {manifest.dataset_id}")
    typer.echo(f"schema    {manifest.schema_version}")
    typer.echo(f"tables    {len(manifest.artifacts)}")
    typer.echo(f"rows      {dataset.total_rows()}")
    for table, count in dataset.row_counts().items():
        if count:
            typer.echo(f"    {table:<32} {count}")


@app.command("verify")
def verify(
    dataset_dir: Path = typer.Argument(..., help="Dataset directory, relative to the data root."),
    data_root: Path | None = typer.Option(
        None, "--data-root", help=f"Override {DATA_ROOT_ENV_VAR}."
    ),
    as_json: bool = typer.Option(False, "--json", help="Emit a JSON summary."),
) -> None:
    """Verify a persisted dataset against its manifest digests."""
    root = _data_root(data_root)
    try:
        result = verify_dataset(dataset_dir, data_root=root)
    except DatasetLayoutError as error:
        typer.secho(str(error), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from error
    payload = {
        "dataset_dir": str(result.dataset_dir),
        "ok": result.ok,
        "manifest_ok": result.manifest_digest_ok,
        "artifacts_ok": result.artifacts_ok,
        "checked_artifacts": result.checked_artifacts,
        "problems": list(result.problems),
    }
    if as_json:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
    else:
        typer.echo(f"dataset   {result.dataset_dir}")
        typer.echo(f"artifacts {result.checked_artifacts} checked")
        typer.echo(f"status    {'OK' if result.ok else 'PROBLEMS'}")
        for problem in result.problems:
            typer.secho(f"  {problem}", fg=typer.colors.RED, err=True)
    if not result.ok:
        raise typer.Exit(code=1)


@app.command("manifest")
def manifest(
    dataset_dir: Path = typer.Argument(..., help="Dataset directory, relative to the data root."),
    data_root: Path | None = typer.Option(
        None, "--data-root", help=f"Override {DATA_ROOT_ENV_VAR}."
    ),
) -> None:
    """Print a dataset's provenance manifest."""
    root = _data_root(data_root)
    try:
        _dataset, found = read_dataset(dataset_dir, data_root=root)
    except DatasetLayoutError as error:
        typer.secho(str(error), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from error
    typer.echo(json.dumps(found.model_dump(mode="json"), indent=2, sort_keys=True))


@app.callback()
def root() -> None:
    """Canonical dataset commands."""
