"""Fully typed numeric assertions for the PSD test suite.

``pytest.approx`` carries partially unknown type information upstream, which is
incompatible with project-wide Pyright strict. These helpers give tests precise
float comparison without loosening the type checker.
"""

from __future__ import annotations

import math

__all__ = ("assert_close", "is_close")


def is_close(
    actual: float,
    expected: float,
    *,
    rel_tol: float = 1e-12,
    abs_tol: float = 0.0,
) -> bool:
    """Return whether two floats agree within the given tolerances."""
    return math.isclose(actual, expected, rel_tol=rel_tol, abs_tol=abs_tol)


def assert_close(
    actual: float,
    expected: float,
    *,
    rel_tol: float = 1e-12,
    abs_tol: float = 0.0,
    context: str = "",
) -> None:
    """Assert that two floats agree, reporting both values on failure."""
    if not is_close(actual, expected, rel_tol=rel_tol, abs_tol=abs_tol):
        suffix = f" ({context})" if context else ""
        message = (
            f"floats differ{suffix}: actual={actual!r} expected={expected!r} "
            f"rel_tol={rel_tol!r} abs_tol={abs_tol!r}"
        )
        raise AssertionError(message)
