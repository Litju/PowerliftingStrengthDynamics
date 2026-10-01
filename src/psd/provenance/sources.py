"""Source registry.

A *source* is where a record came from, under what license and consent, and
whether it is real or synthetic. Source semantics are part of the canonical
model, not a side channel: PSD must keep real and synthetic data explicitly
distinguishable, and third-party datasets must retain their own license,
consent, provenance, and redistribution constraints.
"""

from __future__ import annotations

import re
from datetime import datetime
from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from psd.timeutil import require_aware

__all__ = (
    "ConsentBasis",
    "DataRegime",
    "RedistributionPolicy",
    "SourceNature",
    "SourceRecord",
    "SourceRecordError",
)

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_SOURCE_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,127}$")


class SourceRecordError(ValueError):
    """Raised when source metadata is internally inconsistent."""


class SourceNature(StrEnum):
    """Whether a source describes real or generated athletes.

    This distinction is load-bearing. PSD-Sim ground truth must never be
    projected onto real athletes as if it were validated physiology.
    """

    REAL = "real"
    SYNTHETIC = "synthetic"


class DataRegime(StrEnum):
    """The benchmark data regime a source feeds."""

    COMP = "psd_comp"
    REAL = "psd_real"
    PRO = "psd_pro"
    SIM = "psd_sim"


class RedistributionPolicy(StrEnum):
    """Redistribution rights for the source data itself.

    This is independent of the PSD code license. Apache-2.0 covers PSD source
    code and original documentation; it never extends to third-party data.
    """

    ALLOWED = "allowed"
    RESTRICTED = "restricted"
    NOT_ALLOWED = "not_allowed"
    UNKNOWN = "unknown"


class ConsentBasis(StrEnum):
    """The declared basis for holding athlete data from this source."""

    PUBLIC_LICENSE = "public_license"
    USER_CONSENT = "user_consent"
    DONATION_AGREEMENT = "donation_agreement"
    ETHICS_APPROVED = "ethics_approved"
    NONE_DECLARED = "none_declared"
    UNKNOWN = "unknown"


class SourceRecord(BaseModel):
    """Immutable provenance record for one external data source.

    Attributes:
        source_id: Stable source identifier used as a foreign key everywhere.
        display_name: Human-readable name.
        nature: ``real`` or ``synthetic``.
        regime: Benchmark data regime.
        origin_system: Originating system or publication (for example
            ``openpowerlifting``, ``hevy_export``).
        dataset_version: Version or snapshot label of the source dataset.
        license_id: SPDX-style license identifier for the source data.
        license_url: URL for the source data license or terms.
        consent_basis: Declared basis for holding athlete data.
        redistribution: Redistribution rights for the source data.
        snapshot_date: Date of the source snapshot the records were read from.
        snapshot_sha256: SHA-256 of the source snapshot, when known.
        ingested_at: When PSD ingested the source. Always timezone-aware UTC.
        notes: Free-form provenance notes. Never public athlete free text.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_id: str = Field(
        min_length=1,
        max_length=128,
        pattern=_SOURCE_ID_PATTERN.pattern,
        description="Stable source identifier.",
    )
    display_name: str = Field(min_length=1, max_length=256)
    nature: SourceNature
    regime: DataRegime
    origin_system: str = Field(min_length=1, max_length=128)
    dataset_version: str | None = Field(default=None, max_length=128)
    license_id: str | None = Field(default=None, max_length=128)
    license_url: str | None = Field(default=None, max_length=512)
    consent_basis: ConsentBasis
    redistribution: RedistributionPolicy
    snapshot_date: datetime | None = None
    snapshot_sha256: str | None = None
    ingested_at: datetime
    notes: str | None = Field(default=None, max_length=2048)

    @model_validator(mode="after")
    def _check_consistency(self) -> Self:
        """Enforce the source-semantics invariants."""
        if self.snapshot_sha256 is not None and not _SHA256_PATTERN.match(self.snapshot_sha256):
            msg = f"snapshot_sha256 must be lowercase hex SHA-256; got {self.snapshot_sha256!r}."
            raise ValueError(msg)
        if self.nature is SourceNature.SYNTHETIC and self.regime is not DataRegime.SIM:
            msg = (
                f"Source {self.source_id!r} is synthetic but declares regime "
                f"{self.regime.value!r}; synthetic sources must declare "
                f"{DataRegime.SIM.value!r}."
            )
            raise ValueError(msg)
        if self.nature is SourceNature.REAL and self.regime is DataRegime.SIM:
            msg = (
                f"Source {self.source_id!r} declares regime {self.regime.value!r} but is "
                "real data; simulated regimes must contain synthetic sources."
            )
            raise ValueError(msg)
        if self.nature is SourceNature.REAL and self.consent_basis in {
            ConsentBasis.NONE_DECLARED,
            ConsentBasis.UNKNOWN,
        }:
            msg = (
                f"Source {self.source_id!r} holds real athlete data but declares consent "
                f"basis {self.consent_basis.value!r}. Declare an explicit consent basis "
                "before ingesting real data."
            )
            raise ValueError(msg)
        if self.redistribution is RedistributionPolicy.NOT_ALLOWED and self.license_id is None:
            msg = (
                f"Source {self.source_id!r} forbids redistribution but declares no license; "
                "record the source terms explicitly."
            )
            raise ValueError(msg)
        object.__setattr__(
            self,
            "ingested_at",
            require_aware(self.ingested_at, field="ingested_at"),
        )
        if self.snapshot_date is not None:
            object.__setattr__(
                self,
                "snapshot_date",
                require_aware(self.snapshot_date, field="snapshot_date"),
            )
        return self
