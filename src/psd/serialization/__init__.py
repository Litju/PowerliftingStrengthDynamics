"""Deterministic Arrow/Parquet serialization.

Everything persisted by PSD must be reproducible. Four mechanisms cooperate:

1. **Explicit ordering.** :mod:`psd.serialization.ordering` sorts every table by
   its declared total ordering before anything is hashed or written. DuckDB and
   SQL give no row-order guarantee without ``ORDER BY``, so canonical artifacts
   never rely on incidental order.
2. **Content digests.** :mod:`psd.serialization.canonical` encodes a table into a
   byte string whose definition depends only on the logical values, never on the
   Parquet writer, row-group size, or PyArrow version.
3. **Reproducible Parquet.** :mod:`psd.serialization.parquet` pins the writer
   options that would otherwise introduce run-to-run variation.
4. **Dataset assembly.** :mod:`psd.serialization.dataset` assembles the canonical
   table set, orders it, writes it inside the data root, and records both digests
   per artifact in a deterministic JSON manifest.
5. **Derived artifacts.** :mod:`psd.serialization.derived` persists read-optimised
   projections *of* canonical tables under the same ordering, Parquet profile, and
   content digest, while recording which canonical dataset each one projects so a
   summary can never be mistaken for an independent observation.

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
from psd.serialization.dataset import (
    CanonicalDataset,
    DatasetLayoutError,
    DatasetWriteSpec,
    VerificationResult,
    build_dataset,
    read_dataset,
    verify_dataset,
    write_dataset,
)
from psd.serialization.derived import (
    DerivedArtifactError,
    DerivedTableSpec,
    derived_metadata,
    read_derived_table,
    write_derived_table,
)
from psd.serialization.ordering import OrderingError, canonical_order, canonical_order_polars
from psd.serialization.parquet import (
    PARQUET_PROFILE,
    ParquetProfile,
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
    table_to_rows,
)

__all__ = (
    "PARQUET_PROFILE",
    "CanonicalDataset",
    "CanonicalEncodingError",
    "DatasetLayoutError",
    "DatasetWriteSpec",
    "DerivedArtifactError",
    "DerivedTableSpec",
    "OrderingError",
    "ParquetProfile",
    "ParquetWriteResult",
    "TableSchemaMismatchError",
    "VerificationResult",
    "build_dataset",
    "canonical_bytes",
    "canonical_order",
    "canonical_order_polars",
    "content_digest",
    "derived_metadata",
    "empty_table",
    "read_dataset",
    "read_derived_table",
    "read_parquet",
    "records_to_table",
    "sha256_file",
    "table_to_records",
    "table_to_rows",
    "verify_dataset",
    "write_dataset",
    "write_derived_table",
    "write_parquet",
)
