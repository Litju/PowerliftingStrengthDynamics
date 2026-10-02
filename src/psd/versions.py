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
#: ``1.1.0`` is a **minor** bump: every change is additive, and no existing field
#: changes meaning. The driver was RES-237, which read a real full-scale
#: OpenPowerlifting export and found source semantics the ``1.0.0`` schema could not
#: state without loss:
#:
#: * ``source`` gains ``publication_basis``, so a source can say *why* its data was
#:   published instead of borrowing the copyright or athlete-consent axis to say so.
#:   ``ConsentBasis`` also gains ``public_record``: a public-domain archive of
#:   competition results is freely redistributable and says nothing about whether
#:   the lifters consented, and a public record is not an open license.
#: * ``competition_meet`` is a new table. A meet is context, not an outcome: it has
#:   an identity, a start date, a hosting body, a sanctioning body, and a
#:   sanctioned/unsanctioned status. It had to be first-class because a source
#:   exposes only a meet's *start* date, so meet identity cannot be reconstructed
#:   from the per-athlete results that reference it.
#: * ``competition`` gains the declared event, the participation status and its
#:   numeric placing, the reported age *with its precision*, the explicit age class
#:   and birth-year class, the free-form division, the tested-category flag, the
#:   lifter's own country/state as reported for that result, and the meet reference.
#:   Each exists because omitting it would have meant inferring it: a ``DQ`` code
#:   coerced to a placing, an approximate age rounded to an exact one, an age mined
#:   out of free-text division, or a wrap-free-permitting equipment category read as
#:   proof the athlete wore wraps.
#: * ``competition_attempt`` gains ``source_attempt_raw`` (the signed source value,
#:   where the sign *is* the result) and ``attempt_role``, and widens
#:   ``attempt_number`` to 4. A fourth attempt is a record attempt that contributes
#:   to no total, and without the role a fourth attempt is indistinguishable from a
#:   third.
#: * ``competition_reported_result`` gains ``source_value_raw``,
#:   ``result_source_field``, and ``reported_best_semantics``. A few federations
#:   publish a *negative* reported best meaning "lowest weight attempted and failed";
#:   without the semantics column that value has two readings and neither is a
#:   negative lift.
#: * ``EquipmentClass`` gains ``multi_ply`` and ``straps_allowed``, because folding
#:   them into ``single_ply`` or ``other`` would assert a distinction the source
#:   explicitly draws, or discard a competition category entirely. No existing member
#:   changed meaning, so this is additive rather than a vocabulary redefinition.
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
