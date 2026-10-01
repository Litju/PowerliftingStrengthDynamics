"""``psd paths`` -- inspect the external data-root boundary."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict

import typer

from psd.paths import DATA_ROOT_ENV_VAR, DataRootError, DataRootSubdirectory, resolve_data_root

app = typer.Typer(
    name="paths",
    help="Inspect the external PSD data root. No dataset lives in the Git repository.",
    no_args_is_help=False,
)


class _LayoutEntry(TypedDict):
    """One canonical subdirectory of the data root."""

    name: str
    path: str


@app.callback()
def paths_root() -> None:
    """Declare the group explicitly so Typer does not collapse a single command."""


@app.command("show")
def show(
    data_root: Path | None = typer.Option(
        None,
        "--data-root",
        help="Override PSD_DATA_ROOT instead of reading the environment.",
    ),
    as_json: bool = typer.Option(False, "--json", help="Emit JSON instead of a table."),
) -> None:
    """Print the resolved data root and its canonical layout."""
    try:
        root = resolve_data_root(data_root)
    except DataRootError as error:
        typer.secho(str(error), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from error

    layout: list[_LayoutEntry] = [
        {"name": subdirectory.value, "path": str(root / subdirectory.value)}
        for subdirectory in DataRootSubdirectory
    ]
    if as_json:
        typer.echo(
            json.dumps(
                {
                    "data_root": str(root),
                    "environment_variable": DATA_ROOT_ENV_VAR,
                    "layout": layout,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    typer.echo(f"{DATA_ROOT_ENV_VAR} = {root}")
    typer.echo("")
    for entry in layout:
        exists = "created" if Path(entry["path"]).is_dir() else "not created"
        typer.echo(f"  {entry['name']:<11} {entry['path']}  [{exists}]")
