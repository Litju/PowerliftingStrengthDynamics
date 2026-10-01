"""Schema version access for the schema package.

The version contract itself lives in :mod:`psd.versions`, which has no
intra-project dependencies so that provenance manifests and the schema registry
can both use it without an import cycle. This module re-exports it under
``psd.schema.version`` so schema consumers have one obvious import path.
"""

from __future__ import annotations

from typing import Final

from psd.versions import (
    ALIAS_REGISTRY_VERSION,
    CANONICAL_SCHEMA_NAME,
    MANIFEST_VERSION,
    ONTOLOGY_VERSION,
    SCHEMA_VERSION,
    SchemaVersion,
    SchemaVersionError,
    assert_schema_readable,
)

#: Version of the deterministic identifier scheme.
#:
#: Bump on any change to identifier derivation, because identifiers are content
#: derived and a different derivation changes every identifier in every artifact.
ID_SCHEME: Final[str] = "psd-ids-v1"

__all__ = (
    "ALIAS_REGISTRY_VERSION",
    "CANONICAL_SCHEMA_NAME",
    "ID_SCHEME",
    "MANIFEST_VERSION",
    "ONTOLOGY_VERSION",
    "SCHEMA_VERSION",
    "SchemaVersion",
    "SchemaVersionError",
    "assert_schema_readable",
)
