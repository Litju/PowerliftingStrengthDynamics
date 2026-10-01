"""Base record contracts shared by every canonical table.

Design rules encoded here
-------------------------

* **Explicit missingness.** Every record carries a ``missingness_reason`` so a
  ``null`` is never reinterpreted. Absence is never encoded as zero.
* **Quality flags.** Trust and provenance markers are part of the record, sorted
  and de-duplicated so equal flag sets serialize identically.
* **Temporal provenance.** Records distinguish ``created_at``,
  ``scheduled_at``, ``performed_at``, ``observed_at``, ``modified_at`` and
  ``ingested_at`` so that "what was known at prediction time" is decidable.
  A field stays ``null`` when the source never supplied it; it is never filled in
  from a related record.
* **Source traceability.** Every row names the source it came from, the raw key
  it had there, and (optionally) the digest of the raw source record.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from psd.schema.vocabulary import MissingnessReason, QualityFlag
from psd.timeutil import require_aware

__all__ = (
    "CANONICAL_RECORD_CONFIG",
    "ID_PATTERN",
    "ContextRecord",
    "EventRecord",
    "ProvenancedRecord",
    "TemporalProvenance",
    "normalize_quality_flags",
    "normalize_timestamp",
)

ID_PATTERN = re.compile(r"^[a-z]{2,8}_[0-9a-f]{32}$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")

CANONICAL_RECORD_CONFIG = ConfigDict(frozen=True, extra="forbid")


def normalize_timestamp(value: datetime | None, *, field: str) -> datetime | None:
    """Normalize an aware timestamp to UTC, rejecting naive values."""
    if value is None:
        return None
    return require_aware(value, field=field)


def normalize_quality_flags(value: tuple[QualityFlag, ...] | list[str]) -> tuple[QualityFlag, ...]:
    """Return flags as a sorted, de-duplicated tuple.

    Sorting makes two equal flag sets produce identical bytes regardless of the
    order a source listed them in.
    """
    return tuple(sorted({QualityFlag(str(flag)) for flag in value}, key=lambda flag: flag.value))


class ProvenancedRecord(BaseModel):
    """Row-level source traceability and missingness.

    Attributes:
        source_id: Source that supplied this record.
        source_record_key: The record's identifier in that source, verbatim.
        source_record_hash: SHA-256 of the raw source record bytes, when known.
        ingested_at: When PSD ingested the record. Always timezone-aware UTC.
        quality_flags: Sorted, de-duplicated trust markers.
        missingness_reason: Why values on this record are absent, when the record
            is known to be incomplete.
    """

    model_config = CANONICAL_RECORD_CONFIG

    source_id: str = Field(min_length=1, max_length=128)
    source_record_key: str | None = Field(default=None, max_length=512)
    source_record_hash: str | None = Field(default=None, pattern=_SHA256_PATTERN.pattern)
    ingested_at: datetime
    quality_flags: tuple[QualityFlag, ...] = ()
    missingness_reason: MissingnessReason | None = None

    @model_validator(mode="after")
    def _finalize(self) -> Self:
        object.__setattr__(
            self, "ingested_at", normalize_timestamp(self.ingested_at, field="ingested_at")
        )
        object.__setattr__(self, "quality_flags", normalize_quality_flags(self.quality_flags))
        return self


class TemporalProvenance(BaseModel):
    """The six canonical temporal-provenance fields.

    ``scheduled_at`` is when the record was intended to happen, ``performed_at``
    when it actually happened, ``observed_at`` when a measurement was taken,
    ``created_at`` when the record was authored, ``modified_at`` when it was last
    changed, and ``ingested_at`` when PSD read it. Confusing any two of these
    destroys longitudinal ordering, so each is stored separately and only filled
    when the source genuinely supplied it.
    """

    model_config = CANONICAL_RECORD_CONFIG

    created_at: datetime | None = None
    scheduled_at: datetime | None = None
    performed_at: datetime | None = None
    observed_at: datetime | None = None
    modified_at: datetime | None = None

    @model_validator(mode="after")
    def _finalize_timestamps(self) -> Self:
        for name in ("created_at", "scheduled_at", "performed_at", "observed_at", "modified_at"):
            value = getattr(self, name)
            object.__setattr__(self, name, normalize_timestamp(value, field=name))
        return self


class EventRecord(ProvenancedRecord, TemporalProvenance):
    """Base class for canonical records on the event timeline."""


class ContextRecord(ProvenancedRecord):
    """Base class for slow-changing context records (no per-event timestamps)."""

    created_at: datetime | None = None

    @field_validator("created_at")
    @classmethod
    def _normalize_created_at(cls, value: datetime | None) -> datetime | None:
        return normalize_timestamp(value, field="created_at")
