"""Source registry.

A *source* is where a record came from, under what license and consent, and what
kind of thing it is. Source semantics are part of the canonical model, not a side
channel: PSD must keep real athlete data, simulator-generated athlete data, and
authored reference vocabulary explicitly distinguishable, and third-party datasets
must retain their own license, consent, provenance, and redistribution constraints.

The three natures are not degrees of one axis. ``real`` and ``synthetic`` both
describe athletes and are told apart so PSD-Sim ground truth is never projected onto
real athletes as validated physiology. ``reference`` describes neither: it is an
authored vocabulary such as the exercise ontology, which records what an exercise *is*
and holds no observation about any person. Filing reference data as synthetic would
assert that a simulator generated it.

Four independent axes, not one
------------------------------

A source declares four separate things, and conflating any two of them misstates
the record:

``license_id`` / ``license_url``
    The copyright basis for the data as published. OpenPowerlifting dedicates its
    competition data to the public domain and therefore has **no** SPDX license
    identifier; recording one would invent a license the project does not claim.
``publication_basis``
    Why the data was published at all: it is a record of a public competition, a
    simulator's output, an authored vocabulary, or a private export.
``consent_basis``
    The declared basis for holding an individual's data. This is *not* implied by
    either of the two above.
``redistribution``
    The rights attached to redistributing the data, which is a question about the
    copy rather than about the copyright or the athletes.

The distinction between the last two is not pedantic. A public-domain archive of
competition results is freely redistributable and says nothing at all about whether
the lifters in it consented to PSD holding their names. Filing such a source as
``public_license`` under ``consent_basis`` asserted consent that was never given;
:meth:`SourceRecord` now carries an explicit ``PUBLIC_RECORD`` basis so that a
public record can be described without claiming a person agreed to anything.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from datetime import datetime
from enum import StrEnum
from typing import Final, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from psd.timeutil import require_aware

__all__ = (
    "ConsentBasis",
    "DataRegime",
    "PublicationBasis",
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
    """What kind of thing a source contributes.

    Three classifications, and the distinction between them is load-bearing.
    PSD-Sim ground truth must never be projected onto real athletes as if it were
    validated physiology, and *authored reference vocabulary* must never be
    represented as generated athlete observations -- an exercise ontology records
    no athlete at all, and calling it synthetic would imply a simulator produced it.
    """

    #: Observations of a real person, ingested from an external system.
    REAL = "real"
    #: Athlete observations produced by a simulator. This classification means
    #: *generated athlete data* and nothing else.
    SYNTHETIC = "synthetic"
    #: An authored, versioned vocabulary or structure: descriptors, code lists,
    #: schema metadata. It describes concepts and holds no observations.
    REFERENCE = "reference"


class DataRegime(StrEnum):
    """The benchmark data regime a source feeds."""

    COMP = "psd_comp"
    REAL = "psd_real"
    PRO = "psd_pro"
    #: Simulator-generated athlete observations. The only regime
    #: :class:`SourceNature.SYNTHETIC` may declare.
    SIM = "psd_sim"
    #: Authored reference vocabulary. The only regime
    #: :class:`SourceNature.REFERENCE` may declare.
    REFERENCE = "psd_reference"


class RedistributionPolicy(StrEnum):
    """Redistribution rights for the source data itself.

    This is independent of the PSD code license. Apache-2.0 covers PSD source
    code and original documentation; it never extends to third-party data.
    """

    ALLOWED = "allowed"
    RESTRICTED = "restricted"
    NOT_ALLOWED = "not_allowed"
    UNKNOWN = "unknown"


class PublicationBasis(StrEnum):
    """Why a source published its data.

    This is the *publication* axis and it is deliberately separate from both the
    copyright axis (``license_id``/``license_url``) and the athlete-consent axis
    (``consent_basis``). Knowing a competition archive is public tells you nothing
    about its license, and tells you nothing about whether the competitors agreed
    to anything; recording all three separately is what stops one of them from
    standing in for another.
    """

    #: A record of a competition that took place, assembled by the organizing or
    #: sanctioning body or by a community archive of such results. Competitors
    #: published these by competing; nobody asked them to consent to an archive.
    PUBLISHED_COMPETITION_RESULTS = "published_competition_results"
    #: A simulator's output. Describes how the records came to exist, not who
    #: they describe.
    SIMULATOR_GENERATED = "simulator_generated"
    #: A vocabulary or structure PSD authored: it describes concepts and holds no
    #: observation about a person.
    AUTHORED_VOCABULARY = "authored_vocabulary"
    #: A private export made available under some agreement.
    PRIVATE_EXPORT = "private_export"
    UNKNOWN = "unknown"


class ConsentBasis(StrEnum):
    """The declared basis for holding an individual's data from this source.

    ``PUBLIC_RECORD`` and ``PUBLIC_LICENSE`` answer "why may PSD hold this?" with
    something other than "the person said so". They are kept apart from
    :class:`PublicationBasis` because a public record is not necessarily open-licensed,
    and neither says anything about athlete consent:

    ``PUBLIC_RECORD``
        The rows are a record of a public competition. This is the correct basis for
        an archive of published competition results, and asserting ``USER_CONSENT``
        for it would claim an agreement that was never sought.
    ``PUBLIC_LICENSE``
        The rows are published under an open license that permits the use, and no
        individuals are involved at all -- an authored dataset such as PSD-Sim's
        own output.
    """

    PUBLIC_RECORD = "public_record"
    PUBLIC_LICENSE = "public_license"
    USER_CONSENT = "user_consent"
    DONATION_AGREEMENT = "donation_agreement"
    ETHICS_APPROVED = "ethics_approved"
    NONE_DECLARED = "none_declared"
    UNKNOWN = "unknown"


#: The publication basis each nature implies.
#:
#: A nature absent from this mapping makes no claim: real sources may legitimately
#: publish competition results, a private export, or something else.
_IMPLIED_PUBLICATION_BASIS: Final[Mapping[SourceNature, PublicationBasis]] = {
    SourceNature.SYNTHETIC: PublicationBasis.SIMULATOR_GENERATED,
    SourceNature.REFERENCE: PublicationBasis.AUTHORED_VOCABULARY,
}


#: The regime a nature implies, where the nature is more than "real observations".
#:
#: A nature absent from this mapping makes no claim about the regime: real sources
#: may legitimately feed the competition, real, or prospective regimes.
_IMPLIED_REGIME: Final[Mapping[SourceNature, DataRegime]] = {
    SourceNature.SYNTHETIC: DataRegime.SIM,
    SourceNature.REFERENCE: DataRegime.REFERENCE,
}


class SourceRecord(BaseModel):
    """Immutable provenance record for one external data source.

    Attributes:
        source_id: Stable source identifier used as a foreign key everywhere.
        display_name: Human-readable name.
        nature: ``real`` for athlete observations, ``synthetic`` for
            simulator-generated athlete observations, or ``reference`` for authored
            vocabulary that describes concepts rather than people.
        regime: Benchmark data regime.
        origin_system: Originating system or publication (for example
            ``openpowerlifting``, ``hevy_export``).
        dataset_version: Version or snapshot label of the source dataset.
        license_id: Copyright basis for the data as published. An SPDX identifier
            where one applies, and a descriptive token where none does: a source that
            dedicates its data to the public domain has no SPDX license, and
            recording one would assert a grant the source never made.
        license_url: URL for the source data license or terms.
        publication_basis: Why the source published its data at all.
        consent_basis: Declared basis for holding an individual's data. Never
            implied by ``license_id`` or ``publication_basis``.
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
    publication_basis: PublicationBasis = PublicationBasis.UNKNOWN
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
        implied = _IMPLIED_REGIME.get(self.nature)
        if implied is not None and self.regime is not implied:
            msg = (
                f"Source {self.source_id!r} is {self.nature.value} but declares regime "
                f"{self.regime.value!r}; {self.nature.value} sources must declare "
                f"{implied.value!r}."
            )
            raise ValueError(msg)
        # Stated separately from the mapping above so a misfiled regime names the
        # regime it actually violates rather than only the nature that forbids it.
        for nature, regime in _IMPLIED_REGIME.items():
            if self.nature is not nature and self.regime is regime:
                msg = (
                    f"Source {self.source_id!r} declares regime {regime.value!r} but is "
                    f"{self.nature.value}; the {regime.value!r} regime contains only "
                    f"{nature.value} sources."
                )
                raise ValueError(msg)
        implied_publication = _IMPLIED_PUBLICATION_BASIS.get(self.nature)
        if implied_publication is not None and self.publication_basis is not implied_publication:
            msg = (
                f"Source {self.source_id!r} is {self.nature.value} but declares publication "
                f"basis {self.publication_basis.value!r}; {self.nature.value} sources must "
                f"declare {implied_publication.value!r}."
            )
            raise ValueError(msg)
        for nature, basis in _IMPLIED_PUBLICATION_BASIS.items():
            if self.nature is not nature and self.publication_basis is basis:
                msg = (
                    f"Source {self.source_id!r} declares publication basis "
                    f"{basis.value!r} but is {self.nature.value}; that basis describes how "
                    f"only {nature.value} sources come to exist."
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
