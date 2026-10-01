"""PSD -- Powerlifting Strength Dynamics.

Powerlifting Strength Dynamics is an open benchmark for longitudinal system
identification and rollout simulation of strength adaptation in powerlifting.

This package implements the canonical, event-time longitudinal athlete schema.
The persisted schema boundary is Apache Arrow / Parquet; Polars is the primary
transformation engine.

Licensed under the Apache License, Version 2.0. The license covers PSD source
code and original project documentation only -- see ``LICENSE``.
"""

from __future__ import annotations

from typing import Final

__all__ = ("__version__",)

__version__: Final[str] = "0.1.0"
