"""Deterministic Arrow/Parquet serialization.

Everything persisted by PSD must be reproducible. Three mechanisms cooperate:

1. **Explicit ordering.** :mod:`psd.serialization.ordering` sorts every table by
   its declared total ordering before anything is hashed or written. DuckDB and
   SQL give no row-order guarantee without ``ORDER BY``, so canonical artifacts
   never rely on incidental order.
2. **Content digests.** :mod:`psd.serialization.canonical` encodes a table into a
   byte string whose definition depends only on the logical values, never on the
   Parquet writer, row-group size, or PyArrow version.
3. **Reproducible Parquet.** :mod:`psd.serialization.parquet` pins the writer
   options that would otherwise introduce run-to-run variation.

Two digests are therefore recorded per artifact:

``content_sha256``
    Digest of the canonical logical encoding. Equal for the same logical table
    regardless of Parquet options or library version. This is the digest to
    compare when checking reproducibility.
``sha256``
    Digest of the file bytes as written. Sensitive to the PyArrow version and to
    compression settings; recorded for tamper evidence and byte-level audit.
"""

from __future__ import annotations

from psd.serialization.canonical import (
    CanonicalEncodingError,
    canonical_bytes,
    content_digest,
)
from psd.serialization.ordering import OrderingError, canonical_order
from psd.serialization.parquet import (
    PARQUET_PROFILE,
    ParquetWriteResult,
    read_parquet,
    sha256_file,
    write_parquet,
)
from psd.serialization.table import (
    TableSchemaMismatchError,
    empty_table,
    records_to_table,
    table_to_records,
)

__all__ = (
    "PARQUET_PROFILE",
    "CanonicalEncodingError",
    "OrderingError",
    "ParquetWriteResult",
    "TableSchemaMismatchError",
    "canonical_bytes",
    "canonical_order",
    "content_digest",
    "empty_table",
    "read_parquet",
    "records_to_table",
    "sha256_file",
    "table_to_records",
    "write_parquet",
)
