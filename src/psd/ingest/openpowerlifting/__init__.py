"""OpenPowerlifting adapter for PSD-COMP."""

from __future__ import annotations

from psd.ingest.openpowerlifting.snapshot import (
    OpenPowerliftingSnapshot,
    ServiceSnapshotFacts,
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
    "OpenPowerliftingSnapshot",
    "ServiceSnapshotFacts",
    "openpowerlifting_source_id",
    "openpowerlifting_source_record",
    "service_snapshot_facts",
)
