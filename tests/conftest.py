"""Pytest configuration and shared path helpers."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
SRC_ROOT: Path = PROJECT_ROOT / "src"
