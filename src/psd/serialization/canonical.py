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
from datetime import datetime
from typing import Any, Final, cast

import pyarrow as pa

from psd.timeutil import as_utc, iso_utc

__all__ = (
    "CONTENT_ENCODING",
    "CanonicalEncodingError",
    "canonical_bytes",
    "content_digest",
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
        table: Table to encode. Callers must sort it first; this function
            encodes rows in the order given.
        table_name: Logical table name, included in the header so two different
            tables with identical rows never share a digest.

    Returns:
        The canonical encoding.

    Raises:
        CanonicalEncodingError: A value has no canonical encoding, for example a
            non-finite float or an unsupported Arrow type.
    """
    header = (
        f"{CONTENT_ENCODING}\n"
        f"table:{table_name}\n"
        f"columns:{','.join(table.schema.names)}\n"
        f"rows:{table.num_rows}\n"
    ).encode()

    column_values: list[list[Any]] = [column.to_pylist() for column in table.columns]
    rows: list[bytes] = [
        _encode_value(column_values[column_index][row_index])
        for row_index in range(table.num_rows)
        for column_index in range(len(column_values))
    ]
    return b"".join([header, *rows])


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
