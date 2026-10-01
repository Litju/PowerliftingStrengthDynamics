"""Canonical schema package for PSD.

The persisted schema boundary is Apache Arrow / Parquet. This subpackage owns
the entity contracts, the Arrow schemas derived from them, deterministic
identifiers, controlled vocabularies, canonical ordering, and schema versioning.
"""

from __future__ import annotations

from psd.schema.arrow import ArrowTypeMappingError, arrow_type_for, schema_for_model, schema_json
from psd.schema.identifiers import IdPrefix, derive_id, is_valid_id, make_id
from psd.schema.registry import (
    TABLE_SPECS,
    TableSpec,
    arrow_schema_for,
    column_order,
    table_categories,
    table_names,
    table_spec,
)
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
    "TABLE_SPECS",
    "ArrowTypeMappingError",
    "IdPrefix",
    "SchemaVersion",
    "SchemaVersionError",
    "TableSpec",
    "arrow_schema_for",
    "arrow_type_for",
    "assert_schema_readable",
    "column_order",
    "derive_id",
    "is_valid_id",
    "make_id",
    "schema_for_model",
    "schema_json",
    "table_categories",
    "table_names",
    "table_spec",
)
