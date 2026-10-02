"""Schema and artifact versioning.

This module has no intra-project dependencies on purpose. Provenance manifests
import it, and the schema registry imports provenance manifests; keeping versions
free of both makes that relationship acyclic instead of import-order dependent.

Version policy
--------------

A schema version is ``<series>/<major>.<minor>.<patch>``.

* **patch** -- clarification only. Field meanings are unchanged and no fields
  are added or removed. Existing artifacts remain readable.
* **minor** -- additive change. New nullable fields or new tables may be added.
  Existing artifacts remain readable; readers must tolerate unknown columns.
* **major** -- breaking change. A field is removed or re-interpreted, a column
  type changes, or a controlled-vocabulary member changes meaning. Artifacts
  must be migrated or regenerated.

The same policy governs dataset manifests (``psd-manifest``) and the
deterministic identifier scheme.

Because PSD's canonical validation compares Arrow schemas for exact equality,
:meth:`SchemaVersion.is_readable_by` is the *policy* statement: a reader must
refuse a schema whose semantics it cannot honour. Silent reinterpretation of an
artifact is forbidden.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Self

__all__ = (
    "ALIAS_REGISTRY_VERSION",
    "CANONICAL_SCHEMA_NAME",
    "ID_SCHEME",
    "MANIFEST_VERSION",
    "ONTOLOGY_VERSION",
    "SCHEMA_VERSION",
    "SchemaVersion",
    "SchemaVersionError",
    "assert_schema_readable",
)

_VERSION_PATTERN = re.compile(
    r"^(?P<series>[a-z][a-z0-9-]*)/(?P<major>\d+)\.(?P<minor>\d+)\.(?P<patch>\d+)$"
)

CANONICAL_SCHEMA_NAME = "psd_canonical"


@dataclass(frozen=True, slots=True, order=True)
class SchemaVersion:
    """An immutable, comparable ``series/major.minor.patch`` version."""

    series: str
    major: int
    minor: int
    patch: int

    def __post_init__(self) -> None:
        if not _VERSION_PATTERN.match(self.tag):
            msg = (
                f"Malformed schema version {self.tag!r}; expected "
                "'<series>/<major>.<minor>.<patch>'."
            )
            raise SchemaVersionError(msg)

    @property
    def tag(self) -> str:
        """The canonical serialized form, for example ``psd-canonical/0.1.0``."""
        return f"{self.series}/{self.major}.{self.minor}.{self.patch}"

    @property
    def ordering_key(self) -> tuple[int, int, int]:
        """Numeric ordering key, ignoring the series name."""
        return (self.major, self.minor, self.patch)

    @property
    def compatibility_key(self) -> tuple[int, int]:
        """Compatibility ordering key.

        Patch releases are excluded by design: a patch bump only clarifies
        documentation, so it never changes what an artifact means and never
        blocks a reader.
        """
        return (self.major, self.minor)

    @classmethod
    def parse(cls, value: str) -> Self:
        """Parse a ``series/major.minor.patch`` string."""
        match = _VERSION_PATTERN.match(value.strip())
        if match is None:
            msg = f"Unparseable schema version {value!r}."
            raise SchemaVersionError(msg)
        return cls(
            series=match.group("series"),
            major=int(match.group("major")),
            minor=int(match.group("minor")),
            patch=int(match.group("patch")),
        )

    def is_readable_by(self, reader: Self) -> bool:
        """Return whether a reader at *reader* may read artifacts at ``self``.

        Patch differences are always compatible. An artifact from a newer minor
        release is readable only by a reader at that minor or later, and any
        major difference is unreadable.

        Args:
            reader: The version of the reader's schema knowledge.

        Returns:
            ``True`` when the artifact does not require semantics the reader
            lacks.
        """
        if self.series != reader.series:
            return False
        return self.compatibility_key <= reader.compatibility_key

    def __str__(self) -> str:
        return self.tag


class SchemaVersionError(ValueError):
    """Raised when a schema version is malformed or unreadable."""


#: Version of the canonical Arrow/Parquet athlete-history schema.
#:
#: ``1.1.0`` is a **minor** bump: the only change is an additive one. ``source`` gains
#: ``publication_basis``, so a source can state *why* its data was published instead of
#: borrowing the copyright axis or the athlete-consent axis to say so. The four axes a
#: source must be able to keep apart -- copyright basis, publication basis, athlete
#: consent, and redistribution rights -- were previously collapsed onto three columns,
#: and a public-domain archive of published competition results had to be filed as
#: athlete-consented merely because it was legally redistributable. Adding a nullable
#: column changes no existing field's meaning, so artifacts remain readable by a reader
#: at this minor version; the column is persisted in the declared order so no other
#: table is affected. ``ConsentBasis`` also gains ``public_record``, which asserts a
#: public record rather than a licence or a consent, and ``PublicationBasis`` is a new
#: vocabulary, so neither changes the meaning of an existing member.
#:
#: The steps before it, for the record: ``0.2.0`` added the ``exercise_normalization``
#: table and tightened descriptor columns from free text to controlled vocabularies,
#: ``0.3.0`` added the reference source classification, and ``1.0.0`` was a **major**
#: bump because ``exercise_normalization.confidence`` was *removed*: the column carried
#: a fixed float per resolution method that was never calibrated against held-out
#: labels, so a reader could reasonably have treated ``0.8`` as "correct 80% of the
#: time". The mapping evidence it appeared to summarise is all retained.
SCHEMA_VERSION = SchemaVersion(series="psd-canonical", major=1, minor=1, patch=0)

#: Version of the dataset/provenance manifest contract.
MANIFEST_VERSION = SchemaVersion(series="psd-manifest", major=0, minor=1, patch=0)

#: Version of the exercise ontology: the canonical exercise identities and their
#: observable descriptors.
#:
#: Tracked separately from :data:`SCHEMA_VERSION` because the ontology evolves on
#: a different cadence from the athlete-history tables, and because a consumer
#: must be able to ask "which vocabulary was this label resolved against?"
#: without the canonical schema changing.
#:
#: ``1.0.0`` is a **major** bump: two exercises changed meaning rather than gaining an
#: attribute. ``conventional_deadlift`` was classified ``competition_lift`` and
#: ``sumo_deadlift`` ``sport_specific``, which asserted both that the competition
#: deadlift *is* the conventional stance and that sumo is outside the sport. The rules
#: prescribe neither stance, so both are now stance-qualified ``competition_variation``
#: entries of the stance-unspecified ``deadlift``. A re-interpreted descriptor is a
#: breaking change under this module's policy.
ONTOLOGY_VERSION = SchemaVersion(series="psd-ontology", major=1, minor=0, patch=0)

#: Version of the source-alias registry.
#:
#: Tracked separately from :data:`ONTOLOGY_VERSION` on purpose: a new alias for an
#: existing exercise, or a new source namespace, must be able to ship without
#: claiming that a canonical exercise identity changed.
#:
#: ``1.0.0`` is a **major** bump because two bindings were retargeted:
#: ``Competition Deadlift`` moved from ``conventional_deadlift`` to the
#: stance-unspecified ``deadlift``, matching ``Competition Squat`` and
#: ``Competition Bench``; and ``Deadlift (Conventional)`` moved from ``deadlift`` to
#: ``conventional_deadlift``, because that spelling states a stance and must resolve to
#: the stance-qualified entity. Retargeting an alias changes what a stored source label
#: means, so artifacts built against the old binding must be regenerated.
ALIAS_REGISTRY_VERSION = SchemaVersion(series="psd-ontology-alias", major=1, minor=0, patch=0)

#: Deterministic identifier scheme. Bump on any change to identifier derivation.
ID_SCHEME = "psd-ids-v1"


def assert_schema_readable(
    artifact_version: SchemaVersion,
    reader: SchemaVersion = SCHEMA_VERSION,
) -> None:
    """Raise :class:`SchemaVersionError` when *artifact_version* is unreadable.

    Args:
        artifact_version: Version recorded in the artifact being read.
        reader: Version of the reader, defaulting to the installed schema.

    Raises:
        SchemaVersionError: The artifact cannot be read without silent
            reinterpretation.
    """
    if not artifact_version.is_readable_by(reader):
        msg = (
            f"Artifact schema {artifact_version.tag} is not readable by schema "
            f"{reader.tag}. Refusing to reinterpret it."
        )
        raise SchemaVersionError(msg)
