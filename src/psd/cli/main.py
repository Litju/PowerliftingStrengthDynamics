"""Typer application root for the ``psd`` CLI."""

from __future__ import annotations

import typer

from psd import __version__

app = typer.Typer(
    name="psd",
    help=("Powerlifting Strength Dynamics (PSD): canonical event-time athlete schema tooling."),
    no_args_is_help=True,
    add_completion=False,
)


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
