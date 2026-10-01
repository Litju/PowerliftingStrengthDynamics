"""Unit tests for the ``psd`` package root."""

from __future__ import annotations

from importlib.metadata import version

import psd


def test_package_version_matches_distribution_metadata() -> None:
    assert psd.__version__ == version("powerlifting-strength-dynamics")


def test_package_declares_canonical_semantics_in_docstring() -> None:
    assert psd.__doc__ is not None
    assert "Apache License, Version 2.0" in psd.__doc__
