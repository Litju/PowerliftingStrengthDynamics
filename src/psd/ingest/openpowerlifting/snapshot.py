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
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import PurePosixPath
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from psd.timeutil import require_aware

__all__ = (
    "OpenPowerliftingSnapshot",
    "ServiceSnapshotFacts",
    "archive_declared_facts",
    "service_snapshot_facts",
)

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_SERVICE_DATE_PATTERN = re.compile(r"(?P<value>\d{4}-\d{2}-\d{2})")
_SERVICE_REVISION_PATTERN = re.compile(r"(?P<value>[0-9a-f]{7,40})")

#: The member-name form the archive actually publishes, e.g.
#: ``openpowerlifting-2026-09-26-f231b4f6.csv``. Anchored, because a loose pattern
#: would happily match a hex run inside a folder name and claim a revision that was
#: never declared.
_ARCHIVE_MEMBER_PATTERN = re.compile(
    r"^openpowerlifting-(?P<date>\d{4}-\d{2}-\d{2})-(?P<revision>[0-9a-f]{7,40})\.csv$"
)
_BARE_CELL_PATTERN = re.compile(r"[\d,]+")


def archive_declared_facts(member_name: str) -> tuple[str | None, str | None]:
    """Return the ``(date, revision)`` a CSV member name declares, if it declares any.

    Returns:
        The date and revision, either of which is ``None`` when the name does not
        follow the service's dated form. Nothing is inferred when the pattern does not
        match: an unnamed date stays unnamed.
    """
    stem = PurePosixPath(member_name).name
    match = _ARCHIVE_MEMBER_PATTERN.match(stem)
    if match is None:
        return None, None
    return match.group("date"), match.group("revision")


class ServiceSnapshotFacts(BaseModel):
    """What the service says about the snapshot it served.

    Two independent statements exist and both are kept, because a disagreement between
    them is exactly the kind of thing a provenance record exists to surface:

    ``updated_date`` / ``revision``
        Parsed from the bulk-download page's own ``Updated:`` and ``Revision:``
        labels. This is the page's statement about the current snapshot.
    ``archive_declared_date`` / ``archive_declared_revision``
        Parsed from the CSV member name *inside* the archive, which the service names
        in a dated, revision-suffixed form such as
        ``openpowerlifting-2026-09-26-f231b4f6.csv``. This is the archive's own
        statement, observed from the bytes rather than from the page.

    Neither is ever derived from the other or from a filename timestamp: the published
    download name is mutable and says nothing about which snapshot it holds.

    Attributes:
        updated_date: Snapshot date the bulk page reports, as ``YYYY-MM-DD``.
        revision: Data-service revision the bulk page reports.
        archive_declared_date: Snapshot date the archive's member name declares.
        archive_declared_revision: Data-service revision the member name declares.
        advertised_row_count: Row count the bulk page's table states for the complete
            dataset, when it states one.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    updated_date: str | None = Field(default=None, pattern=_SERVICE_DATE_PATTERN.pattern)
    revision: str | None = Field(default=None, pattern=_SERVICE_REVISION_PATTERN.pattern)
    archive_declared_date: str | None = Field(default=None, pattern=_SERVICE_DATE_PATTERN.pattern)
    archive_declared_revision: str | None = Field(
        default=None, pattern=_SERVICE_REVISION_PATTERN.pattern
    )
    advertised_row_count: int | None = Field(default=None, ge=0)

    @property
    def statements_agree(self) -> bool:
        """Whether the page and the archive name the same revision.

        ``False`` is a finding to report, not an error: the page is regenerated when
        the site is rebuilt and the archive is regenerated nightly, so the two can
        legitimately describe different moments.
        """
        if self.archive_declared_revision is None:
            return False
        if self.revision is None:
            return False
        return self.archive_declared_revision.startswith(self.revision) or self.revision.startswith(
            self.archive_declared_revision
        )

    def disagreements(self) -> tuple[str, ...]:
        """Return human-readable descriptions of every disagreement."""
        found: list[str] = []
        if self.archive_declared_revision is not None and not self.statements_agree:
            found.append(
                f"bulk page reports revision {self.revision!r} but the archive's CSV member "
                f"name declares {self.archive_declared_revision!r}"
            )
        if (
            self.updated_date is not None
            and self.archive_declared_date is not None
            and self.updated_date != self.archive_declared_date
        ):
            found.append(
                f"bulk page reports snapshot date {self.updated_date!r} but the archive's "
                f"CSV member name declares {self.archive_declared_date!r}"
            )
        return tuple(found)


def service_snapshot_facts(page_text: str) -> ServiceSnapshotFacts:
    """Parse the service-reported snapshot facts from the bulk-download page.

    The parsing is deliberately narrow: it looks for the labels the page actually
    uses and returns ``None`` rather than guessing. A future page revision that
    renames the labels yields fewer facts, which is an acknowledged gap rather than a
    fabricated one.

    Args:
        page_text: HTML (or plain text) of the bulk-download index page.

    Returns:
        Whatever the page stated, and nothing more.
    """
    updated: str | None = None
    revision: str | None = None
    rows: int | None = None
    lines = page_text.splitlines()
    for index, line in enumerate(lines):
        if "Updated:" in line and updated is None:
            match = _SERVICE_DATE_PATTERN.search(line)
            if match is not None:
                updated = match.group("value")
        if "Revision:" in line and revision is None:
            match = _SERVICE_REVISION_PATTERN.search(line)
            if match is not None:
                revision = match.group("value")
        if rows is None and _BULK_DATASET_NAME in line:
            rows = _rows_after_anchor(lines, index)
    return ServiceSnapshotFacts(
        updated_date=updated,
        revision=revision,
        advertised_row_count=rows,
    )


_TAG_PATTERN = re.compile(r"<[^>]*>")
_BULK_DATASET_NAME = "openpowerlifting-latest.zip"
_ROW_SCAN_WINDOW = 12


def _rows_after_anchor(lines: Sequence[str], anchor_index: int) -> int | None:
    """Return the row count the bulk page's table states after the dataset link.

    The page renders the row count in its own table cell, so the search is for a cell
    whose entire text is a bare integer. The size cell is rendered with a unit suffix
    and therefore never matches, and a cell holding prose never matches either. If the
    page layout changes, this returns ``None``: a missing advertised figure is a gap to
    record, not a figure to guess.
    """
    for line in lines[anchor_index + 1 : anchor_index + 1 + _ROW_SCAN_WINDOW]:
        text = _TAG_PATTERN.sub("", line).strip()
        if not _BARE_CELL_PATTERN.fullmatch(text):
            if text == "" or "<td" in line or "</td>" in line:
                continue
            return None
        return int(text.replace(",", ""))
    return None


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
