"""Deterministic canonical identifiers.

Identifiers are content-derived, so re-ingesting the same source produces the
same identifiers and a canonical artifact can be rebuilt and byte-compared. The
scheme is versioned by :data:`psd.schema.version.ID_SCHEME`; changing any part of
the derivation requires a new scheme tag.

Scheme (``psd-ids-v1``)
-----------------------

``<prefix>_<32 hex characters>``

The digest is ``SHA-256`` over a length-prefixed, type-tagged key so that no
combination of parts can be reinterpreted as a different combination::

    key = for each part: type-tagged and length-prefixed, joined by US (0x1F)
    preimage = ID_SCHEME | 0x1F | prefix | 0x1F | key

Parts are encoded structurally rather than by joining raw strings, so an athlete
key containing a separator cannot collide with a two-part key, and an absent part
cannot collide with the literal text of its marker.

Hierarchical derivation
----------------------

Child identifiers are derived from their parent identifier plus a local ordinal
rather than from source text, so a child is stable given its parent. This makes
identity depend on the ingestion structure: changing an upstream ordinal changes
downstream identifiers. That is intentional and is why the scheme is versioned.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from enum import StrEnum

from psd.schema.version import ID_SCHEME

__all__ = (
    "IDENTIFIER_LENGTH",
    "IdPrefix",
    "derive_id",
    "id_key",
    "is_valid_id",
    "make_id",
)

IDENTIFIER_LENGTH = 32

_UNIT_SEPARATOR = "\x1f"
_HEX_DIGITS = frozenset("0123456789abcdef")


class IdPrefix(StrEnum):
    """Identifier prefix per canonical entity."""

    ATHLETE = "ath"
    ATHLETE_SOURCE_LINK = "aslk"
    BODY_MEASUREMENT = "bms"
    EQUIPMENT_STATE = "eqs"
    SOURCE = "src"
    PROVENANCE = "prv"
    EXERCISE_DEFINITION = "exd"
    EXERCISE_ALIAS = "exa"
    EXERCISE_NORMALIZATION = "exn"
    PROGRAM = "prg"
    PROGRAM_VERSION = "pver"
    PROGRAM_MODIFICATION = "pmod"
    PLANNED_SESSION = "plses"
    PLANNED_EXERCISE = "plex"
    PLANNED_SET = "plset"
    PERFORMED_SESSION = "pses"
    PERFORMED_EXERCISE = "pex"
    PERFORMED_SET = "pset"
    PERFORMED_REP = "prep"
    OBSERVATION = "obs"
    PERFORMANCE_TEST = "ptest"
    VELOCITY_OBSERVATION = "vel"
    COMPETITION = "cmp"
    COMPETITION_ATTEMPT = "catt"
    COMPETITION_REPORTED_RESULT = "cres"


def id_key(parts: Iterable[str | int | None]) -> str:
    """Return the unambiguous, type-tagged, length-prefixed key for *parts*.

    Each part is tagged by type so that no combination of parts can be
    reinterpreted as a different combination:

    * ``None`` becomes ``n`` (absent);
    * ``int`` becomes ``i<value>``;
    * ``str`` becomes ``s<length>:<text>``.
    """
    encoded: list[str] = []
    for part in parts:
        if part is None:
            encoded.append("n")
        elif isinstance(part, int) and not isinstance(part, bool):
            encoded.append(f"i{part}")
        else:
            text = part if isinstance(part, str) else str(part)
            encoded.append(f"s{len(text)}:{text}")
    return _UNIT_SEPARATOR.join(encoded)


def make_id(prefix: IdPrefix, *parts: str | int | None) -> str:
    """Derive a deterministic identifier.

    Args:
        prefix: Entity prefix.
        *parts: Ordered structural parts, most general first. ``None`` marks a
            part that does not apply.

    Returns:
        ``<prefix>_<32 hex characters>``.
    """
    preimage = _UNIT_SEPARATOR.join((ID_SCHEME, prefix.value, id_key(parts)))
    digest = hashlib.sha256(preimage.encode("utf-8")).hexdigest()
    return f"{prefix.value}_{digest[:IDENTIFIER_LENGTH]}"


def derive_id(parent_id: str, prefix: IdPrefix, *parts: str | int | None) -> str:
    """Derive a child identifier from a parent identifier and local ordinals."""
    return make_id(prefix, parent_id, *parts)


def is_valid_id(value: str, prefix: IdPrefix | None = None) -> bool:
    """Return whether *value* is a well-formed canonical identifier."""
    head, separator, digest = value.partition("_")
    if not separator or len(digest) != IDENTIFIER_LENGTH:
        return False
    if not _HEX_DIGITS.issuperset(digest):
        return False
    if prefix is not None:
        return head == prefix.value
    return any(head == candidate.value for candidate in IdPrefix)
