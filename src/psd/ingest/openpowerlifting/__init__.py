"""OpenPowerlifting adapter for PSD-COMP.

Layering, outermost first:

* :mod:`~psd.ingest.openpowerlifting.source` -- provenance: the public-domain
  dedication, the license of record, and the ``SourceRecord`` for a pinned snapshot.
* :mod:`~psd.ingest.openpowerlifting.snapshot` -- the pinned snapshot: where it came
  from, what the service said, and what the bytes hash to.
* :mod:`~psd.ingest.openpowerlifting.acquire` -- staged download and digest pinning.
* :mod:`~psd.ingest.openpowerlifting.contract` -- the declared CSV schema, with every
  column classified as mapped, preserved, ignored, or unknown.
* :mod:`~psd.ingest.openpowerlifting.transform` -- the canonical conversion.
"""

from __future__ import annotations

from psd.ingest.openpowerlifting.contract import (
    SOURCE_COLUMNS,
    SourceColumnDisposition,
    SourceColumnSpec,
    SourceSchemaError,
    SourceSchemaReview,
    attempt_source_columns,
    column_spec,
    expected_columns,
    read_source_header,
    reported_result_source_columns,
    require_source_schema,
    review_source_schema,
)
from psd.ingest.openpowerlifting.snapshot import (
    OpenPowerliftingSnapshot,
    ServiceSnapshotFacts,
    archive_declared_facts,
    service_snapshot_facts,
)
from psd.ingest.openpowerlifting.source import (
    OPENPOWERLIFTING_BULK_DOCS_URL,
    OPENPOWERLIFTING_BULK_INDEX_URL,
    OPENPOWERLIFTING_DATA_SERVICE_URL,
    OPENPOWERLIFTING_LICENSE_ID,
    OPENPOWERLIFTING_LICENSE_STATEMENT,
    OPENPOWERLIFTING_ORIGIN_SYSTEM,
    OPENPOWERLIFTING_PUBLIC_DOMAIN_NOTE,
    openpowerlifting_source_id,
    openpowerlifting_source_record,
)

__all__ = (
    "OPENPOWERLIFTING_BULK_DOCS_URL",
    "OPENPOWERLIFTING_BULK_INDEX_URL",
    "OPENPOWERLIFTING_DATA_SERVICE_URL",
    "OPENPOWERLIFTING_LICENSE_ID",
    "OPENPOWERLIFTING_LICENSE_STATEMENT",
    "OPENPOWERLIFTING_ORIGIN_SYSTEM",
    "OPENPOWERLIFTING_PUBLIC_DOMAIN_NOTE",
    "SOURCE_COLUMNS",
    "OpenPowerliftingSnapshot",
    "ServiceSnapshotFacts",
    "SourceColumnDisposition",
    "SourceColumnSpec",
    "SourceSchemaError",
    "SourceSchemaReview",
    "archive_declared_facts",
    "attempt_source_columns",
    "column_spec",
    "expected_columns",
    "openpowerlifting_source_id",
    "openpowerlifting_source_record",
    "read_source_header",
    "reported_result_source_columns",
    "require_source_schema",
    "review_source_schema",
    "service_snapshot_facts",
)
