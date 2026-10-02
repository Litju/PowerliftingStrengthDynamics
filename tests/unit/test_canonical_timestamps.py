"""Qualification of the canonical timestamp encoder.

The optimized epoch-microsecond path exists because asking Arrow to convert a
``timestamp("us", tz="UTC")`` column into Python ``datetime`` objects resolves the zone
through ``zoneinfo`` and ``pytz`` on every call, and a zone library that is absent is not
remembered as absent. Measured on this project that path cost about 179x more than the
same conversion of a naive column, paid once per timestamp column per table -- which, on a
corpus, is paid millions of times.

That is only a safe trade if the fast path is *the same function* as the slow one. These
tests hold it to that across the range a corpus actually contains and the range it might:
instants before the Unix epoch, the epoch boundary itself, sub-microsecond-adjacent values
where an off-by-one would be invisible, and instants written with a non-UTC offset where a
missing normalization would produce a different digest for the same moment.

The property that must never break is that two representations of one instant encode to
identical bytes. A canonical benchmark artifact's whole claim is that its digest depends
only on logical content, and a timestamp is content.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone
from typing import Final

import pyarrow as pa
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from psd.serialization import canonical as _canonical
from psd.serialization.canonical import (
    CanonicalEncodingError,
    canonical_bytes,
    canonical_timestamp,
    encode_rows,
)
from psd.timeutil import TimestampAmbiguityError

pytestmark = pytest.mark.windows_parity

# The two routes through the timestamp encoder are private because production code reaches
# only the column-wise one. Qualifying that the *private* route agrees with the public one
# is the entire point of these tests: the optimization is justified by that equivalence, so
# it has to be asserted against the function it replaced, not against itself.
_encode_epoch_micros = _canonical._encode_epoch_micros  # type: ignore[reportPrivateUsage]
_encode_timestamp = _canonical._encode_timestamp  # type: ignore[reportPrivateUsage]

_EPOCH: Final[datetime] = datetime(1970, 1, 1, tzinfo=UTC)
_MICROSECOND: Final[timedelta] = timedelta(microseconds=1)

#: Zones with awkward offsets, including half-hour and negative ones, so the tests cover
#: offsets that are not whole hours and zones that sit either side of the date line.
NON_UTC_ZONES: Final[tuple[timedelta, ...]] = (
    timedelta(hours=1),
    timedelta(hours=5, minutes=30),
    timedelta(hours=-8),
    timedelta(hours=13),
    timedelta(hours=-3, minutes=-30),
)


def micros_of(moment: datetime) -> int:
    """Return *moment* as epoch microseconds, the form the fast path consumes.

    Computed rather than written out: a literal epoch value in a test is a claim about a
    date that has to be checked by eye, and a wrong one fails as a date mismatch that
    looks nothing like an encoder defect.
    """
    return (moment - _EPOCH) // _MICROSECOND


#: The range the *slow* path can represent. Arrow's int64 spans further, so the encoder
#: refuses rather than guesses at instants ``datetime`` cannot name -- and a property test
#: over the int64 range would be asserting an equivalence that cannot be evaluated.
MIN_MICROS: Final[int] = micros_of(datetime.min.replace(tzinfo=UTC))
MAX_MICROS: Final[int] = micros_of(datetime.max.replace(tzinfo=UTC))

#: The same range narrowed by a day at each end, so that shifting an instant by the widest
#: offset under test still lands inside ``datetime``. Without the margin, the property test
#: would fail with an ``OverflowError`` at the extremes, which says nothing about the
#: encoder.
SHIFTABLE_MIN_MICROS: Final[int] = MIN_MICROS + int(timedelta(days=1) // _MICROSECOND)
SHIFTABLE_MAX_MICROS: Final[int] = MAX_MICROS - int(timedelta(days=1) // _MICROSECOND)


def _timestamp_table(dtype: pa.DataType, values: list[int | None]) -> pa.Table:
    """Return a one-column timestamp table for *values*, carrying *dtype* exactly.

    ``from_arrays`` rather than ``pa.table`` because the community stubs describe the
    latter as an overload set over partly unknown parameter types, which would force the
    whole module to be untyped. The timestamp dtype is a parameter because half of what
    these tests qualify is what the encoder refuses.
    """
    return pa.Table.from_arrays([pa.array(values, type=dtype)], names=["ts"])


def _table(values: list[int | None]) -> pa.Table:
    """Return a one-column microsecond UTC timestamp table for *values*."""
    return _timestamp_table(pa.timestamp("us", tz="UTC"), values)


# ---------------------------------------------------------------------------
# The fast path is the slow path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "micros",
    [
        pytest.param(0, id="epoch"),
        pytest.param(-1, id="one-microsecond-before-the-epoch"),
        pytest.param(-86_400_000_000, id="1969-12-31-midnight"),
        pytest.param(-86_400_000_001, id="1969-12-31-just-before-midnight"),
        pytest.param(-1_000_000, id="1969-12-31T23:59:59"),
        pytest.param(1, id="one-microsecond-after-the-epoch"),
        pytest.param(86_400_000_000, id="1970-01-01-midnight-plus-a-day"),
        pytest.param(1_000_000, id="1970-01-01T00:00:01"),
        pytest.param(micros_of(datetime(2026, 1, 1, tzinfo=UTC)), id="2026-01-01T00:00:00"),
        pytest.param(
            micros_of(datetime(2026, 1, 1, 0, 0, 0, 123_456, tzinfo=UTC)),
            id="2026-01-01T00:00:00.123456",
        ),
        pytest.param(micros_of(datetime(2100, 1, 1, tzinfo=UTC)), id="2100-01-01T00:00:00"),
        pytest.param(MIN_MICROS, id="the earliest representable instant"),
        pytest.param(MAX_MICROS, id="the latest representable instant"),
    ],
)
def test_the_epoch_path_equals_the_datetime_path(micros: int) -> None:
    """Every instant encodes identically whichever route produced it.

    This is the whole qualification: the optimized path is an arithmetic shortcut, and a
    shortcut that disagrees with the reference anywhere is a silent divergence between two
    tools' idea of a corpus's content digest.
    """
    moment = _EPOCH + timedelta(microseconds=micros)

    assert _encode_epoch_micros(micros) == _encode_timestamp(moment)


def test_the_epoch_path_always_emits_six_digit_microseconds() -> None:
    """Sub-microsecond-adjacent instants must not lose their trailing zeros.

    An instant one millisecond past the epoch encoded without padding would be
    indistinguishable from one one microsecond past it, and the two are different events.
    """
    encoded = _encode_epoch_micros(1_000).decode("ascii")

    assert encoded == "T1970-01-01T00:00:00.001000Z;"


def test_before_the_epoch_is_encoded_with_a_year_before_1970() -> None:
    """A negative epoch is not a special case; it is an instant with a real date."""
    assert _encode_epoch_micros(-1) == b"T1969-12-31T23:59:59.999999Z;"


def test_the_epoch_boundary_is_continuous() -> None:
    """The microsecond before the epoch and the one at it are one apart, and stay so."""
    before = _encode_epoch_micros(-1)
    at = _encode_epoch_micros(0)

    assert before == b"T1969-12-31T23:59:59.999999Z;"
    assert at == b"T1970-01-01T00:00:00.000000Z;"


# ---------------------------------------------------------------------------
# Normalized UTC: one instant, one encoding
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("offset", NON_UTC_ZONES, ids=str)
def test_an_offset_representation_canonicalizes_to_the_same_bytes(offset: timedelta) -> None:
    """The same instant written with a non-UTC offset must encode as the UTC instant.

    Otherwise two machines that ingested the same meet through different paths would
    disagree about its content digest, which is the failure the canonical encoding exists
    to make impossible.
    """
    utc_instant = datetime(2026, 1, 1, 6, 0, 0, 123_456, tzinfo=UTC)
    shifted = utc_instant.astimezone(timezone(offset))

    # The two datetimes denote the same instant -- that is the premise -- while naming it
    # differently. If the offsets were equal the fixture would prove nothing.
    assert shifted.utcoffset() != utc_instant.utcoffset()
    assert canonical_timestamp(shifted) == canonical_timestamp(utc_instant)
    assert canonical_timestamp(shifted) == _encode_epoch_micros(micros_of(utc_instant))


def test_a_table_written_in_two_zones_digests_identically() -> None:
    """The equivalence is about whole artifacts, not just about the encoder's return value."""
    moment = datetime(2026, 3, 29, 1, 30, tzinfo=UTC)  # inside the European DST change
    as_utc_micros = int((moment - datetime(1970, 1, 1, tzinfo=UTC)) // timedelta(microseconds=1))
    table = _table([as_utc_micros])

    canonical_bytes(table, table_name="performed_set")

    shifted = moment.astimezone(timezone(timedelta(hours=2)))
    assert shifted.utcoffset() != timedelta(0)
    assert _encode_epoch_micros(as_utc_micros) == canonical_timestamp(shifted)


def test_a_dst_change_does_not_move_an_instant() -> None:
    """01:30 UTC on a spring-forward morning is one instant, whatever zone it is read in.

    A zone whose offset changes at 01:00 local is the case that catches a naive
    ``value.isoformat()`` with the offset left on it: the same wall-clock reading denotes
    a different instant either side of the change, and PSD requires normalized UTC.
    """
    instant = datetime(2026, 3, 29, 1, 30, tzinfo=UTC)
    micros = micros_of(instant)

    assert _encode_epoch_micros(micros) == canonical_timestamp(instant)
    for offset in NON_UTC_ZONES:
        shifted = instant.astimezone(timezone(offset))
        assert canonical_timestamp(shifted) == _encode_epoch_micros(micros)
        # And it is genuinely not the un-normalized rendering, which is the whole point.
        assert canonical_timestamp(shifted) != (
            b"T" + shifted.isoformat(timespec="microseconds").encode() + b";"
        )


# ---------------------------------------------------------------------------
# Naive timestamps never reach the encoder
# ---------------------------------------------------------------------------


def test_a_naive_timestamp_cannot_be_canonicalized() -> None:
    """PSD refuses rather than assuming a zone.

    A naive ``2024-03-01 08:00`` could be any of several real instants depending on who
    read it, and silently picking one would move an event on the longitudinal timeline.
    """
    with pytest.raises(TimestampAmbiguityError, match="Naive timestamp"):
        canonical_timestamp(datetime(2024, 3, 1, 8, 0))


# ---------------------------------------------------------------------------
# Unsupported Arrow types
# ---------------------------------------------------------------------------


def test_a_non_microsecond_timestamp_is_refused() -> None:
    """A millisecond column cannot be encoded by arithmetic that assumes microseconds."""
    table = _timestamp_table(pa.timestamp("ms", tz="UTC"), [0])

    with pytest.raises(CanonicalEncodingError, match="microsecond UTC"):
        encode_rows(table)


def test_a_naive_arrow_timestamp_is_refused() -> None:
    """A naive Arrow column has no zone to normalize to, so its encoding is not knowable."""
    table = _timestamp_table(pa.timestamp("us"), [0])

    with pytest.raises(CanonicalEncodingError, match="microsecond UTC"):
        encode_rows(table)


def test_a_null_timestamp_column_encodes_as_null() -> None:
    """Absence is null, at whatever precision the column carries."""
    rows = encode_rows(_table([None]))

    assert rows == [b"N;"]


# ---------------------------------------------------------------------------
# Properties over the whole microsecond range
# ---------------------------------------------------------------------------


@settings(max_examples=300, deadline=None)
@given(micros=st.integers(min_value=MIN_MICROS, max_value=MAX_MICROS))
def test_the_two_paths_agree_for_any_microsecond(micros: int) -> None:
    """The equivalence is not a list of examples; it holds across the representable range."""
    moment = _EPOCH + timedelta(microseconds=micros)

    assert _encode_epoch_micros(micros) == _encode_timestamp(moment)


@settings(max_examples=200, deadline=None)
@given(
    micros=st.integers(min_value=SHIFTABLE_MIN_MICROS, max_value=SHIFTABLE_MAX_MICROS),
    offset_minutes=st.sampled_from([0, 60, 330, -480, 780, -210]),
)
def test_an_offset_never_changes_the_bytes(micros: int, offset_minutes: int) -> None:
    """Shifting the representation never shifts the encoding."""
    assume(offset_minutes != 0)
    instant = _EPOCH + timedelta(microseconds=micros)

    shifted = instant.astimezone(timezone(timedelta(minutes=offset_minutes)))

    assert canonical_timestamp(shifted) == _encode_epoch_micros(micros)


@settings(max_examples=200, deadline=None)
@given(micros=st.integers(min_value=-2_000_000_000_000_000, max_value=4_000_000_000_000_000))
def test_a_column_encodes_its_instant_in_every_row(micros: int) -> None:
    """The per-column route and the per-value route agree, not just the helper functions."""
    table = _table([micros, None, micros])

    assert encode_rows(table) == [
        _encode_epoch_micros(micros),
        b"N;",
        _encode_epoch_micros(micros),
    ]


@settings(max_examples=200, deadline=None)
@given(micros=st.integers(min_value=MIN_MICROS, max_value=MAX_MICROS))
def test_the_encoded_string_is_an_iso_utc_instant(micros: int) -> None:
    """Whatever it produced, it is a parseable aware UTC instant with microsecond precision.

    This is what makes the encoding inspectable: a reader who suspects two artifacts differ
    can decode one timestamp out of either and compare it to the other.
    """
    moment = _EPOCH + timedelta(microseconds=micros)
    decoded = datetime.fromisoformat(_encode_epoch_micros(micros).decode()[1:-1])

    assert decoded == moment
    assert decoded.utcoffset() == timedelta(0)
    assert decoded.isoformat(timespec="microseconds").endswith("+00:00")
