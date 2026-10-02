"""Tests for schema and manifest version policy."""

from __future__ import annotations

import pytest

from psd.schema.registry import table_spec
from psd.schema.version import (
    ID_SCHEME,
    MANIFEST_VERSION,
    SCHEMA_VERSION,
    SchemaVersion,
    SchemaVersionError,
    assert_schema_readable,
)


def test_version_tags_are_canonical() -> None:
    assert SCHEMA_VERSION.tag == "psd-canonical/1.1.0"
    assert MANIFEST_VERSION.tag == "psd-manifest/0.1.0"
    assert ID_SCHEME == "psd-ids-v1"


def test_parse_round_trip() -> None:
    assert SchemaVersion.parse(SCHEMA_VERSION.tag) == SCHEMA_VERSION


@pytest.mark.parametrize(
    "value",
    ["0.1.0", "psd/1", "psd-canonical/0.1", "", "PSD-CANONICAL/0.1.0", "psd-canonical/x.y.z"],
)
def test_malformed_versions_are_rejected(value: str) -> None:
    with pytest.raises(SchemaVersionError):
        SchemaVersion.parse(value)


def test_patch_and_minor_releases_remain_readable() -> None:
    reader = SchemaVersion(series="psd-canonical", major=0, minor=2, patch=0)
    assert SchemaVersion(series="psd-canonical", major=0, minor=1, patch=0).is_readable_by(reader)
    # A patch bump only clarifies documentation, so it never blocks a reader.
    assert SchemaVersion(series="psd-canonical", major=0, minor=2, patch=1).is_readable_by(reader)


def test_newer_minor_is_unreadable_by_older_reader() -> None:
    artifact = SchemaVersion(series="psd-canonical", major=0, minor=3, patch=0)
    reader = SchemaVersion(series="psd-canonical", major=0, minor=2, patch=9)
    assert not artifact.is_readable_by(reader)


def test_major_release_is_not_readable_by_minor_reader() -> None:
    """A removed column is a breaking change, so it moves the major.

    ``1.0.0`` removed ``exercise_normalization.confidence``. The policy above classes a
    removed field as breaking, and this asserts the machinery enforces that reading
    rather than letting a reader silently accept the next incompatible major.
    """
    assert SCHEMA_VERSION.major == 1
    future = SchemaVersion(series=SCHEMA_VERSION.series, major=2, minor=0, patch=0)
    assert not future.is_readable_by(SCHEMA_VERSION)
    with pytest.raises(SchemaVersionError, match="Refusing to reinterpret"):
        assert_schema_readable(future)


def test_the_retired_minor_artifacts_are_still_inside_the_ordering() -> None:
    """``0.2.0`` precedes ``1.0.0``; the schema gate, not the ordering, does the work.

    ``is_readable_by`` is the coarse policy statement. The enforcement is Arrow schema
    equality plus the recorded ``psd_schema_version`` on read, so an artifact written
    before the removal is refused by the column set it carries rather than by an
    ordering comparison that would call a ``0.x`` artifact readable.
    """
    retired = SchemaVersion(series="psd-canonical", major=0, minor=2, patch=0)
    assert retired.is_readable_by(SCHEMA_VERSION)
    columns = table_spec("exercise_normalization").arrow_schema().names
    assert "confidence" not in columns


def test_different_series_is_not_readable() -> None:
    other = SchemaVersion(series="other", major=0, minor=0, patch=1)
    assert not other.is_readable_by(SCHEMA_VERSION)


def test_versions_sort_numerically() -> None:
    ordered = sorted(
        [
            SchemaVersion(series="psd-canonical", major=0, minor=2, patch=0),
            SchemaVersion(series="psd-canonical", major=0, minor=1, patch=0),
            SchemaVersion(series="psd-canonical", major=1, minor=0, patch=0),
        ]
    )
    assert [version.tag for version in ordered] == [
        "psd-canonical/0.1.0",
        "psd-canonical/0.2.0",
        "psd-canonical/1.0.0",
    ]
