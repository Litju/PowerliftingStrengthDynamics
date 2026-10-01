"""Canonical schema package for PSD.

The persisted schema boundary is Apache Arrow / Parquet. This subpackage owns
the table schemas, deterministic identifiers, controlled vocabularies, and the
canonical ordering rules used for deterministic artifact generation.
"""

from __future__ import annotations

from psd.schema.version import (
    CANONICAL_SCHEMA_NAME,
    ID_SCHEME,
    MANIFEST_VERSION,
    SCHEMA_VERSION,
    SchemaVersion,
    SchemaVersionError,
    assert_schema_readable,
)

__all__ = (
    "CANONICAL_SCHEMA_NAME",
    "ID_SCHEME",
    "MANIFEST_VERSION",
    "SCHEMA_VERSION",
    "SchemaVersion",
    "SchemaVersionError",
    "assert_schema_readable",
)
