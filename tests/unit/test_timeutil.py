"""Tests for timezone-aware timestamp handling."""

from __future__ import annotations

from datetime import UTC, date, datetime, time, timedelta, timezone

import pytest

from psd.timeutil import (
    TimestampAmbiguityError,
    as_utc,
    combine_date_time,
    iso_utc,
    require_aware,
)

pytestmark = pytest.mark.windows_parity


def test_require_aware_rejects_naive_timestamps() -> None:
    with pytest.raises(TimestampAmbiguityError, match="naive timestamp"):
        require_aware(datetime(2026, 1, 1, 8, 0), field="performed_at")


def test_require_aware_normalizes_to_utc() -> None:
    aware = datetime(2026, 1, 1, 8, 0, tzinfo=timezone(timedelta(hours=2)))
    assert require_aware(aware, field="performed_at") == datetime(2026, 1, 1, 6, 0, tzinfo=UTC)


def test_as_utc_rejects_naive() -> None:
    with pytest.raises(TimestampAmbiguityError):
        as_utc(datetime(2026, 1, 1))


def test_iso_utc_is_deterministic_with_microseconds() -> None:
    moment = datetime(2026, 1, 1, 6, 0, 0, 1, tzinfo=UTC)
    assert iso_utc(moment) == "2026-01-01T06:00:00.000001Z"


def test_iso_utc_round_trips_offset_inputs() -> None:
    aware = datetime(2026, 1, 1, 8, 0, tzinfo=timezone(timedelta(hours=2)))
    assert iso_utc(aware) == "2026-01-01T06:00:00.000000Z"


def test_combine_date_time_defaults_to_midnight_utc() -> None:
    assert combine_date_time(date(2026, 3, 14)) == datetime(2026, 3, 14, 0, 0, tzinfo=UTC)


def test_combine_date_time_uses_supplied_time() -> None:
    combined = combine_date_time(date(2026, 3, 14), time(6, 30))
    assert combined == datetime(2026, 3, 14, 6, 30, tzinfo=UTC)
