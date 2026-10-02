"""The pinned OpenPowerlifting source snapshot.

The bulk download is published nightly under a mutable name
(``openpowerlifting-latest.zip``), so **the filename can never be the artifact
identity**. A snapshot's identity is its SHA-256, and this model is the record of
everything a later reader needs in order to re-derive it: where it came from, when
it was fetched, what the service said about itself, what the bytes hash to, what the
extracted CSV hashes to, how many rows it holds, and what its header was.

Nothing here is inferred. Every optional field is one the service did not report or
that the transport did not expose; a missing value stays missing rather than being
filled from a filename or a ``Last-Modified`` header, because a guess presented as
provenance is worse than an acknowledged gap.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from psd.timeutil import require_aware

__all__ = (
    "OpenPowerliftingSnapshot",
    "ServiceSnapshotFacts",
    "service_snapshot_facts",
)

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_SERVICE_DATE_PATTERN = re.compile(r"(?P<value>\d{4}-\d{2}-\d{2})")
_SERVICE_REVISION_PATTERN = re.compile(r"(?P<value>[0-9a-f]{7,40})")


class ServiceSnapshotFacts(BaseModel):
    """What the bulk-download page reports about the snapshot it serves.

    The page states an ``Updated:`` date and a ``Revision:`` commit alongside the
    links. Both are worth recording and neither is guaranteed to be present, so both
    are optional and both are parsed rather than typed in.

    Attributes:
        updated_date: Service-reported snapshot date, as ``YYYY-MM-DD``.
        revision: Service-reported revision, the data service's own short commit.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    updated_date: str | None = Field(default=None, pattern=_SERVICE_DATE_PATTERN.pattern)
    revision: str | None = Field(default=None, pattern=_SERVICE_REVISION_PATTERN.pattern)


def service_snapshot_facts(page_text: str) -> ServiceSnapshotFacts:
    """Parse the service-reported snapshot date and revision from the bulk page.

    The parsing is deliberately narrow: it looks for the labels the page actually
    uses and returns ``None`` rather than guessing. A future page revision that
    renames the labels yields no facts, which is an acknowledged gap rather than a
    fabricated one.

    Args:
        page_text: HTML (or plain text) of the bulk-download index page.

    Returns:
        Whatever the page stated, and nothing more.
    """
    updated: str | None = None
    revision: str | None = None
    for line in page_text.splitlines():
        if "Updated:" in line and updated is None:
            match = _SERVICE_DATE_PATTERN.search(line)
            if match is not None:
                updated = match.group("value")
        if "Revision:" in line and revision is None:
            match = _SERVICE_REVISION_PATTERN.search(line)
            if match is not None:
                revision = match.group("value")
    return ServiceSnapshotFacts(updated_date=updated, revision=revision)


class OpenPowerliftingSnapshot(BaseModel):
    """One digest-pinned OpenPowerlifting bulk snapshot.

    Attributes:
        source_url: Exact URL the archive was fetched from. The mutable
            ``latest`` name is recorded for provenance but the digest, not the name,
            identifies the snapshot.
        downloaded_at: Timezone-aware UTC instant the archive finished downloading.
        service: What the bulk-download page reported about itself, when known.
        archive_sha256: SHA-256 of the raw downloaded ZIP bytes.
        archive_byte_size: Size of the downloaded ZIP in bytes.
        csv_member_name: Name of the data file inside the archive.
        csv_sha256: SHA-256 of the extracted CSV bytes.
        csv_byte_size: Size of the extracted CSV in bytes.
        row_count: Data rows in the CSV, excluding the header.
        source_columns: The exact CSV header, in order.
        code_commit: PSD commit that performed the acquisition, when known.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_url: str = Field(min_length=1, max_length=1024)
    downloaded_at: datetime
    service: ServiceSnapshotFacts = ServiceSnapshotFacts()
    archive_sha256: str = Field(pattern=_SHA256_PATTERN.pattern)
    archive_byte_size: int = Field(ge=0)
    csv_member_name: str = Field(min_length=1, max_length=256)
    csv_sha256: str = Field(pattern=_SHA256_PATTERN.pattern)
    csv_byte_size: int = Field(ge=0)
    row_count: int = Field(ge=0)
    source_columns: tuple[str, ...] = ()
    code_commit: str | None = Field(default=None, max_length=64)

    @model_validator(mode="after")
    def _check_consistency(self) -> Self:
        object.__setattr__(
            self,
            "downloaded_at",
            require_aware(self.downloaded_at, field="downloaded_at"),
        )
        if not self.source_columns:
            msg = (
                "source_columns is required: a pinned snapshot without its exact CSV "
                "header cannot be shown to have been read against a known schema."
            )
            raise ValueError(msg)
        if len(set(self.source_columns)) != len(self.source_columns):
            msg = "source_columns must be free of duplicates; the CSV header repeats a name."
            raise ValueError(msg)
        return self

    @property
    def snapshot_id(self) -> str:
        """Return the snapshot's identity: the archive digest, truncated for reading."""
        return self.archive_sha256

    def dataset_version(self) -> str:
        """Return a human-readable version label from whatever the service reported.

        The label is informational. Identity is always :attr:`archive_sha256`, so a
        snapshot whose service page reported nothing still versions unambiguously.
        """
        parts = [
            part for part in (self.service.updated_date, self.service.revision) if part is not None
        ]
        if parts:
            return "opl-" + "-".join(parts)
        return f"opl-sha256-{self.archive_sha256[:16]}"

    def service_date(self) -> datetime | None:
        """Return the service-reported snapshot date as an aware UTC instant.

        The bulk page states a calendar date with no time and no zone. Midnight UTC
        is used rather than the local time of whoever ran the acquisition, because a
        snapshot date is a day, not an instant, and shifting it by the operator's
        timezone would make two identical snapshots differ.
        """
        if self.service.updated_date is None:
            return None
        year, month, day = (int(part) for part in self.service.updated_date.split("-"))
        return datetime(year, month, day, tzinfo=UTC)
