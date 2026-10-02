"""Canonical table encoding and content digests.

Purpose
-------

PSD must be able to prove that two canonical artifacts represent the same
logical table. Comparing Parquet bytes does not prove that: the byte stream
depends on the PyArrow version, row-group size, compression settings, and
dictionary ordering. This module defines an encoding whose bytes depend **only**
on the logical contents.

Encoding specification (``psd-canonical-content/1``)
-----------------------------------------------------

::

    header    := "psd-canonical-content/1\\n"
                 "table:"  <table name> "\\n"
                 "columns:" <comma-joined column names> "\\n"
                 "rows:"   <decimal row count> "\\n"
    row       := <value> for each column, in declared column order
    value     := "N;"                                   (null)
               | "B0;" | "B1;"                          (bool)
               | "I" <decimal> ";"                      (int64)
               | "F" <float.hex()> ";"                  (float64)
               | "T" <ISO-8601 UTC with 6-digit micros> ";"  (timestamp)
               | "S" <byte length> ":" <utf-8 text> ";"  (string)
               | "L" <element count> ";" <value>*        (list)

Design notes
------------

* Every value is type-tagged, so ``"1"`` and ``1`` cannot collide and a string
  can never be confused with a list.
* Strings and lists are length-prefixed, so no value can imitate the encoding of
  a neighbouring value.
* Floats use ``float.hex()``, the exact hexadecimal representation, rather than
  ``repr``. It is lossless and stable across platforms, which ``repr`` is only
  stable for in practice.
* Timestamps are normalized to UTC and always carry microsecond precision, so
  equal instants always produce equal bytes.
* Non-finite floats are rejected: a canonical benchmark artifact must not carry
  ``NaN`` or infinity, whose comparison and digest behaviour is platform
  sensitive.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import Any, Final, cast

import pyarrow as pa

from psd.timeutil import as_utc, iso_utc

__all__ = (
    "CONTENT_ENCODING",
    "CanonicalEncodingError",
    "canonical_bytes",
    "content_digest",
    "content_header",
    "encode_rows",
)

CONTENT_ENCODING: Final[str] = "psd-canonical-content/1"

_LIST_TAG: Final[bytes] = b"L"
_NULL: Final[bytes] = b"N;"
_TERMINATOR: Final[bytes] = b";"


class CanonicalEncodingError(TypeError):
    """Raised when a value cannot be canonically encoded."""


def canonical_bytes(table: pa.Table, *, table_name: str) -> bytes:
    """Return the canonical byte encoding of *table*.

    Args:
        table: Table to encode. Callers must sort it first; this function encodes rows
            in the order given.
        table_name: Logical table name, included in the header so two different
            tables with identical rows never share a digest.

    Returns:
        The canonical encoding.

    Raises:
        CanonicalEncodingError: A value has no canonical encoding, for example a
            non-finite float or an unsupported Arrow type.
    """
    header = content_header(table.schema.names, table_name=table_name, row_count=table.num_rows)
    return b"".join([header, *encode_rows(table)])


def content_header(columns: Sequence[str], *, table_name: str, row_count: int) -> bytes:
    """Return the canonical encoding's header for a described table.

    Separated from :func:`canonical_bytes` so a streaming writer can open the same
    digest over the same header and then feed rows as they are produced, rather than
    materializing the whole table just to hash it. The header declares the row count
    up front, which is why a streaming writer may only open its digest once it knows
    the final count: the declared count must be the count actually written.
    """
    return (
        f"{CONTENT_ENCODING}\ntable:{table_name}\ncolumns:{','.join(columns)}\nrows:{row_count}\n"
    ).encode()


def encode_rows(table: pa.Table) -> list[bytes]:
    """Return the canonical encoding of each row of *table*, in row order.

    Column values are converted a column at a time through Arrow's own converter,
    which is C-speed. A streaming writer calls this per batch so that the Python
    objects exist only for the batch in hand.

    Timestamps take a separate route for a measured reason. Asking Arrow to turn a
    ``timestamp("us", tz="UTC")`` column into Python ``datetime`` objects is not C-speed:
    it resolves the zone through ``zoneinfo`` and ``pytz`` on every call, and a zone
    library that is absent is not remembered as absent, so each call re-searches the
    import path. Measured on this project that path costs about 179x more than the same
    conversion of a naive column, and it is paid once per timestamp column per table --
    which, on a corpus, is paid millions of times. The canonical schema pins every
    timestamp to microsecond UTC, so the epoch-microsecond value already *is* the instant
    and the formatted string follows from arithmetic. :func:`_encode_epoch_micros` produces
    exactly the bytes :func:`_encode_timestamp` does, which a test asserts.
    """
    column_values: list[list[Any]] = [_column_values(column) for column in table.columns]
    width = len(column_values)
    return [
        b"".join(
            _encode_prepared(column_values[column_index][row_index])
            for column_index in range(width)
        )
        for row_index in range(table.num_rows)
    ]


#: The single timestamp type the canonical schema uses. Anything else is refused rather
#: than encoded on the assumption that it is equivalent.
_CANONICAL_TIMESTAMP: Final[pa.DataType] = pa.timestamp("us", tz="UTC")

_EPOCH: Final[datetime] = datetime(1970, 1, 1, tzinfo=UTC)


def _column_values(column: Any) -> list[Any]:
    """Return each value of *column*, pre-encoded where the type allows it.

    *column* is a PyArrow ``ChunkedArray``. The parameter is left untyped for the reason
    given at the top of :mod:`psd.serialization.parquet`: PyArrow ships no inline types, so
    a precise annotation would be a guess dressed as a fact.

    Returns:
        One entry per row. A timestamp column yields bytes directly, since its encoding is
        a pure function of an integer. Everything else yields the Python value, to be
        encoded per value by :func:`_encode_value`.

    Raises:
        CanonicalEncodingError: The column is a timestamp the canonical schema does not
            define, so its encoding is not knowable.
    """
    dtype: pa.DataType = column.type
    if not pa.types.is_timestamp(dtype):
        return column.to_pylist()
    if dtype != _CANONICAL_TIMESTAMP:
        msg = (
            f"Timestamp column of type {dtype!s} cannot be canonically encoded; the "
            "canonical schema pins every timestamp to microsecond UTC."
        )
        raise CanonicalEncodingError(msg)
    micros: list[int | None] = column.cast(pa.int64()).to_pylist()
    return [_NULL if value is None else _encode_epoch_micros(value) for value in micros]


def _encode_prepared(value: Any) -> bytes:
    """Return the encoding of one prepared column value.

    Bytes are already encoded; anything else is a Python value that still needs encoding.
    """
    if isinstance(value, bytes):
        return value
    return _encode_value(value)


def _encode_epoch_micros(micros: int) -> bytes:
    """Return the canonical encoding of an instant given as epoch microseconds UTC.

    Equivalent to :func:`_encode_timestamp` for an aware UTC datetime, including the fixed
    microsecond precision the encoding depends on.
    """
    moment = _EPOCH + timedelta(microseconds=micros)
    return (
        b"T"
        + moment.isoformat(timespec="microseconds").replace("+00:00", "Z").encode("ascii")
        + _TERMINATOR
    )


def content_digest(table: pa.Table, *, table_name: str) -> str:
    """Return the lowercase hex SHA-256 of a table's canonical encoding."""
    return hashlib.sha256(canonical_bytes(table, table_name=table_name)).hexdigest()


def _encode_value(value: object) -> bytes:
    """Return the canonical encoding of a single Python value.

    ``bool`` is checked before ``int`` because it is a subclass, and ``datetime``
    before the sequence cases so a timestamp is never mistaken for a collection.
    """
    if value is None:
        return _NULL
    if isinstance(value, bool):
        return _encode_bool(value)
    if isinstance(value, (int, float)):
        return _encode_number(value)
    if isinstance(value, str):
        return _encode_str(value)
    if isinstance(value, datetime):
        return _encode_timestamp(value)
    if isinstance(value, (list, tuple)):
        return _encode_list(cast("Sequence[object]", value))
    msg = f"No canonical encoding for value of type {type(value).__name__!r}."
    raise CanonicalEncodingError(msg)


def _encode_number(value: int | float) -> bytes:
    if isinstance(value, int):
        return _encode_int(value)
    return _encode_float(value)


def _encode_bool(value: bool) -> bytes:
    return b"B1;" if value else b"B0;"


def _encode_int(value: int) -> bytes:
    return b"I" + str(value).encode("ascii") + _TERMINATOR


def _encode_str(value: str) -> bytes:
    encoded = value.encode()
    return b"S" + str(len(encoded)).encode("ascii") + b":" + encoded + _TERMINATOR


def _encode_timestamp(value: datetime) -> bytes:
    return b"T" + iso_utc(value).encode("ascii") + _TERMINATOR


def _encode_float(value: float) -> bytes:
    if not math.isfinite(value):
        msg = (
            f"Non-finite float {value!r} cannot be canonically encoded; PSD rejects NaN "
            "and infinity in canonical artifacts because their digest behaviour is not "
            "portable."
        )
        raise CanonicalEncodingError(msg)
    return b"F" + value.hex().encode("ascii") + _TERMINATOR


def _encode_list(values: Sequence[object]) -> bytes:
    """Encode a homogeneous sequence with an explicit element count."""
    header = [b"L", str(len(values)).encode("ascii"), _TERMINATOR]
    encoded: list[bytes] = [b"".join(header), *(_encode_value(item) for item in values)]
    return b"".join(encoded)


def canonical_timestamp(value: datetime) -> bytes:
    """Encode a timestamp, exposed for tests of the timestamp rule."""
    return b"T" + iso_utc(as_utc(value)).encode("ascii") + _TERMINATOR
