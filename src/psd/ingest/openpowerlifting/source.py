"""OpenPowerlifting source identity and provenance.

OpenPowerlifting's Data Service states, on its own introduction page:

    All competition data available on this website are contributed to the Public
    Domain. [...] To the extent possible under law, all competition data on this
    website are waived of all copyright and related or neighboring rights.
    Although you are under no requirement to do so, if you incorporate
    OpenPowerlifting data into your project, please consider adding a statement of
    attribution.

Three consequences are load-bearing here, and all three were wrong in the RES-235
provisional fixture:

1. **The data carry no SPDX license.** A public-domain dedication is not
   ``CC-BY-SA-4.0``, and it is not ``CC0-1.0`` either -- CC0 is a formal license
   grant that OpenPowerlifting does not make. PSD records the descriptive token
   ``public-domain`` and points at the official statement, and deliberately claims
   no license the source has not granted.
2. **The project is not AGPLv3+ for its data.** OpenPowerlifting's *source code*
   carries its own license. Conflating the two would misstate both.
3. **A public record is not athlete consent.** Competition results are published
   because the competitions happened, not because the lifters agreed to anything.
   The source basis is therefore :attr:`~psd.provenance.sources.ConsentBasis.PUBLIC_RECORD`
   and the publication basis is
   :attr:`~psd.provenance.sources.PublicationBasis.PUBLISHED_COMPETITION_RESULTS`;
   recording ``user_consent`` would assert an agreement that was never sought.
"""

from __future__ import annotations

from datetime import datetime
from typing import Final

from psd.ingest.openpowerlifting.snapshot import OpenPowerliftingSnapshot
from psd.provenance.sources import (
    ConsentBasis,
    DataRegime,
    PublicationBasis,
    RedistributionPolicy,
    SourceNature,
    SourceRecord,
)

__all__ = (
    "OPENPOWERLIFTING_BULK_DOCS_URL",
    "OPENPOWERLIFTING_BULK_INDEX_URL",
    "OPENPOWERLIFTING_DATA_SERVICE_URL",
    "OPENPOWERLIFTING_LICENSE_ID",
    "OPENPOWERLIFTING_LICENSE_STATEMENT",
    "OPENPOWERLIFTING_ORIGIN_SYSTEM",
    "OPENPOWERLIFTING_PUBLIC_DOMAIN_NOTE",
    "openpowerlifting_source_id",
    "openpowerlifting_source_record",
)

#: The official data-service introduction page, which carries the public-domain
#: dedication. This is the licensing/provenance URL of record for the data.
OPENPOWERLIFTING_DATA_SERVICE_URL: Final[str] = "https://openpowerlifting.gitlab.io/opl-csv/"

#: The bulk-download index page. Carries the service-reported snapshot date and
#: revision alongside the download links.
OPENPOWERLIFTING_BULK_INDEX_URL: Final[str] = (
    "https://openpowerlifting.gitlab.io/opl-csv/bulk-csv.html"
)

#: The column-by-column documentation for the bulk CSV. This is the source
#: authority for ingestion semantics.
OPENPOWERLIFTING_BULK_DOCS_URL: Final[str] = (
    "https://openpowerlifting.gitlab.io/opl-csv/bulk-csv-docs.html"
)

#: Descriptive copyright basis. Deliberately *not* an SPDX identifier: a
#: public-domain dedication has none, and inventing one would overstate the grant.
OPENPOWERLIFTING_LICENSE_ID: Final[str] = "public-domain"

#: The dedication, quoted from the official page so the record carries the source's
#: own words rather than a paraphrase that could drift.
OPENPOWERLIFTING_LICENSE_STATEMENT: Final[str] = (
    "All competition data available on this website are contributed to the Public "
    "Domain. To the extent possible under law, all competition data on this website "
    "are waived of all copyright and related or neighboring rights. Attribution is "
    "requested but not required."
)

OPENPOWERLIFTING_ORIGIN_SYSTEM: Final[str] = "openpowerlifting"

_SHA256_LENGTH: Final[int] = 64
_HEX_DIGITS: Final[frozenset[str]] = frozenset("0123456789abcdef")

OPENPOWERLIFTING_PUBLIC_DOMAIN_NOTE: Final[str] = (
    "OpenPowerlifting Data Service competition data, contributed to the Public Domain "
    "and waived of copyright and related rights. The bulk filename is mutable and is "
    "never the artifact identity: a snapshot is identified by its SHA-256. This is the "
    "license of the data, not of the OpenPowerlifting project's own source code, which "
    "is licensed separately. These are public competition records, not athlete-consented "
    "records: no competitor was asked to consent to this archive."
)


def openpowerlifting_source_id(digest: str) -> str:
    """Return the canonical ``source_id`` for one pinned snapshot.

    The identifier is derived from the snapshot digest rather than from the mutable
    ``openpowerlifting-latest.zip`` filename, so two snapshots of the same service are
    two distinct sources in the manifest rather than one silently overwritten.

    Args:
        digest: Lowercase hex SHA-256 of the raw downloaded archive.

    Returns:
        A ``source_id`` matching the registry's identifier pattern.

    Raises:
        ValueError: The digest is not a lowercase hex SHA-256.
    """
    if len(digest) != _SHA256_LENGTH or not _HEX_DIGITS.issuperset(digest):
        msg = f"Expected a lowercase hex SHA-256 snapshot digest; got {digest!r}."
        raise ValueError(msg)
    return f"openpowerlifting_{digest[:16]}"


def openpowerlifting_source_record(
    snapshot: OpenPowerliftingSnapshot,
    *,
    ingested_at: datetime,
    notes: str | None = None,
) -> SourceRecord:
    """Return the ``SourceRecord`` for a pinned OpenPowerlifting snapshot.

    Args:
        snapshot: The digest-pinned snapshot the rows were read from.
        ingested_at: Timezone-aware UTC ingestion instant.
        notes: Additional provenance notes; the public-domain note is always kept.

    Returns:
        A validated ``real`` / ``psd_comp`` source record.
    """
    combined = " ".join(part for part in (OPENPOWERLIFTING_PUBLIC_DOMAIN_NOTE, notes) if part)
    return SourceRecord(
        source_id=openpowerlifting_source_id(snapshot.archive_sha256),
        display_name="OpenPowerlifting bulk competition export",
        nature=SourceNature.REAL,
        regime=DataRegime.COMP,
        origin_system=OPENPOWERLIFTING_ORIGIN_SYSTEM,
        dataset_version=snapshot.dataset_version(),
        license_id=OPENPOWERLIFTING_LICENSE_ID,
        license_url=OPENPOWERLIFTING_DATA_SERVICE_URL,
        publication_basis=PublicationBasis.PUBLISHED_COMPETITION_RESULTS,
        consent_basis=ConsentBasis.PUBLIC_RECORD,
        redistribution=RedistributionPolicy.ALLOWED,
        snapshot_date=snapshot.service_date(),
        snapshot_sha256=snapshot.archive_sha256,
        ingested_at=ingested_at,
        notes=combined[:2048],
    )
