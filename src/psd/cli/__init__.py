"""Public ``psd`` command-line interface.

The CLI is orchestration over importable Python APIs. Benchmark and schema
logic must never live only inside command functions.
"""

from __future__ import annotations

from psd.cli.main import app

__all__ = ("app", "main")


def main() -> None:
    """Console-script entry point."""
    app()
