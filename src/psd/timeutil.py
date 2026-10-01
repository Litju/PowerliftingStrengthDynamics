"""Timezone and timestamp helpers.

PSD timestamps are timezone-aware UTC instants. Naive timestamps are ambiguous:
a source that reports ``2024-03-01 08:00`` without an offset cannot be placed on
the event timeline without guessing a timezone, and guessing would silently
corrupt longitudinal ordering.

Therefore: every canonical timestamp is aware, and ambiguity is represented by
``None`` plus an explicit missingness reason rather than by coercion.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, time

__all__ = (
    "UTC",
    "TimestampAmbiguityError",
    "as_utc",
    "combine_date_time",
    "iso_utc",
    "require_aware",
)


class TimestampAmbiguityError(ValueError):
    """Raised when a timestamp is naive or non-UTC where awareness is required."""


def require_aware(value: datetime, *, field: str) -> datetime:
    """Return *value* as an aware UTC instant, or raise.

    Args:
        value: The timestamp to check.
        field: Field name used in the error message.

    Returns:
        The timestamp normalized to UTC.

    Raises:
        TimestampAmbiguityError: The timestamp has no timezone information.
    """
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        msg = (
            f"{field} is a naive timestamp ({value.isoformat()!r}). PSD requires "
            "timezone-aware timestamps; ambiguous source timestamps must be "
            "recorded as null with a missingness reason instead of being coerced."
        )
        raise TimestampAmbiguityError(msg)
    return value.astimezone(UTC)


def as_utc(value: datetime) -> datetime:
    """Normalize an aware timestamp to UTC."""
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        msg = f"Naive timestamp {value.isoformat()!r} cannot be normalized to UTC."
        raise TimestampAmbiguityError(msg)
    return value.astimezone(UTC)


def combine_date_time(value_date: date, value_time: time | None = None) -> datetime:
    """Combine a date with an optional time into a UTC instant.

    A date without a time is midnight UTC. Callers that must distinguish
    "recorded on this day" from "recorded at this instant" should keep a
    separate granularity field rather than inferring it here.
    """
    moment = value_time if value_time is not None else time(0, 0)
    return datetime.combine(value_date, moment, tzinfo=UTC)


def iso_utc(value: datetime) -> str:
    """Serialize an aware timestamp to a deterministic UTC ISO-8601 string.

    The microsecond field is always emitted so that equal instants always
    produce equal bytes, which the artifact digests depend on.
    """
    normalized = as_utc(value)
    return normalized.isoformat(timespec="microseconds").replace("+00:00", "Z")
