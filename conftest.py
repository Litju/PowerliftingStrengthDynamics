"""Root pytest configuration.

Keeps test scratch space inside the repository checkout so that test runs are
self-contained and reproducible on both Windows and Linux, without depending on
the host's ``TEMP``/``TMPDIR`` location or on the maintainer's external data
root. ``.pytest_cache/`` is git-ignored, so this creates no tracked state.

Repository paths are declared in :mod:`tests.support`, not here, so that this
module stays a plain pytest hook container.
"""

from __future__ import annotations

import pytest

__all__ = ("pytest_configure",)


def pytest_configure(config: pytest.Config) -> None:
    """Pin pytest's temporary base directory inside the checkout."""
    # pytest creates the base directory itself but not its parents, so a fresh
    # checkout with no .pytest_cache/ yet would fail with FileNotFoundError.
    base = config.rootpath / ".pytest_cache" / "tmp"
    base.mkdir(parents=True, exist_ok=True)
    config.option.basetemp = str(base)
