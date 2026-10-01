"""Typer application root for the ``psd`` CLI.

The CLI is orchestration over importable Python APIs. Canonical schema,
validation, and benchmark logic must never live only inside command functions.
"""

from __future__ import annotations

import typer

from psd import __version__
from psd.cli.canonical_cmd import app as canonical_app
from psd.cli.inspect_cmd import app as inspect_app
from psd.cli.inspect_cmd import register_validate
from psd.cli.paths_cmd import app as paths_app
from psd.cli.schema_cmd import app as schema_app

app = typer.Typer(
    name="psd",
    help="Powerlifting Strength Dynamics (PSD): canonical event-time athlete schema tooling.",
    no_args_is_help=True,
    add_completion=False,
)
app.add_typer(canonical_app, name="canonical")
app.add_typer(inspect_app, name="inspect")
app.add_typer(paths_app, name="paths")
app.add_typer(schema_app, name="schema")
register_validate(app)


@app.callback()
def root() -> None:
    """Root callback.

    Declared explicitly so Typer keeps the application in multi-command group
    mode even while the v0 command surface is still being built out.
    """


@app.command()
def version() -> None:
    """Print the installed PSD version."""
    typer.echo(__version__)


def main() -> None:
    """Console-script entry point."""
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
