"""Shared, machine-independent helpers for the PSD test suite."""

from __future__ import annotations

from pathlib import Path

__all__ = ("PROJECT_ROOT", "SRC_ROOT")

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
SRC_ROOT: Path = PROJECT_ROOT / "src"
