"""Canonical table encoding and content digests.

Purpose
-------

PSD must be able to prove that two canonical artifacts represent the same
logical table. Comparing Parquet bytes does not prove that: the byte stream
depends on the PyArrow version, row-group size, compression settings, and
dictionary ordering. This module defines an encoding whose bytes depend **only**
on the logical contents.

Encoding specification (``psd-canonical-content/2``)
-----------------------------------------------------

::

    header    := "psd-canonical-content/2\\n"
                 "table:"    <table name> "\\n"
                 "columns:"  <comma-joined encoded column names> "\\n"
                 "excluded:" <comma-joined run-local column names> "\\n"
                 "rows:"     <decimal row count> "\\n"
    row       := <value> for each encoded column, in declared column order
    value     := "N;"                                   (null)
               | "B0;" | "B1;"                          (bool)
               | "T" ... etc, as in version 1

Version 2 adds one line and one rule
------------------------------------

``ingested_at`` is excluded from the digest. It records *when PSD read the row*, not
*what the source said*, and a digest that moves with wall-clock time is not a digest of
anything reproducible. On a corpus the difference is stark: two builds of one pinned
snapshot on different days differ in exactly this column and in nothing else, so with it
included no corpus build could ever be shown to be reproducible -- which is the whole
claim a content digest exists to support. With it excluded, ``content_sha256`` means "the
same logical content", and the header's ``excluded:`` line says what was left out.

Nothing is lost. The column is still persisted on every row, the manifest still records
the run's ``created_at`` and environment, and ``sha256`` still covers the file bytes, so
two runs of the same snapshot differ in every digest they publish *except* the logical one.
That is the honest statement: the rows say the same thing, the runs did not happen at the
same time.

The exclusion is narrow and named rather than implicit. One column, part of the provenance
contract every canonical table shares, excluded by a declared constant that the encoding's
own header reports.

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
    "RUN_LOCAL_COLUMNS",
    "CanonicalEncodingError",
    "canonical_bytes",
    "content_digest",
    "content_header",
    "encode_rows",
    "encoded_columns",
)

CONTENT_ENCODING: Final[str] = "psd-canonical-content/2"

#: Columns that describe the ingestion *run* rather than the source's content.
#:
#: ``ingested_at`` is when PSD read the row. It is genuine provenance and it is persisted;
#: it is simply not content, and a reproducibility check has to be able to say so. See the
#: module docstring for why including it made every corpus build non-reproducible.
RUN_LOCAL_COLUMNS: Final[frozenset[str]] = frozenset({"ingested_at"})


def encoded_columns(columns: Sequence[str]) -> tuple[str, ...]:
    """Return the columns the content digest covers, in declared order.

    Args:
        columns: A table's full declared column list.

    Returns:
        The subset the digest encodes. A table with no run-local column is unchanged.
    """
    return tuple(name for name in columns if name not in RUN_LOCAL_COLUMNS)


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


def content_columns(columns: Sequence[str]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return the encoded and the excluded column lists for *columns*.

    Split out because the digest's header has to name both and the two must be derived from
    the same declaration: a header that listed one set while the rows encoded another would
    be a digest nobody could reproduce.
    """
    excluded = tuple(name for name in columns if name in RUN_LOCAL_COLUMNS)
    return encoded_columns(columns), excluded


def content_header(columns: Sequence[str], *, table_name: str, row_count: int) -> bytes:
    """Return the canonical encoding's header for a described table.

    Separated from :func:`canonical_bytes` so a streaming writer can open the same
    digest over the same header and then feed rows as they are produced, rather than
    materializing the whole table just to hash it. The header declares the row count
    up front, which is why a streaming writer may only open its digest once it knows
    the final count: the declared count must be the count actually written.

    Args:
        columns: The table's **full** declared column list. The header reports both the
            columns the digest encodes and the run-local ones it excludes, so a reader can
            see what the digest covers without reading this module.
        table_name: Logical table name.
        row_count: Rows the encoding that follows describes.

    Returns:
        The header bytes.
    """
    encoded, excluded = content_columns(columns)
    return (
        f"{CONTENT_ENCODING}\ntable:{table_name}\ncolumns:{','.join(encoded)}"
        f"\nexcluded:{','.join(excluded)}\nrows:{row_count}\n"
    ).encode()


def encode_rows(table: pa.Table, *, columns: Sequence[str] | None = None) -> list[bytes]:
    """Return the canonical encoding of each row of *table*, in row order.

    Args:
        table: The rows to encode.
        columns: The columns to encode, in the order they are encoded. Defaults to the
            table's declared columns minus the run-local ones, which is what
            :func:`canonical_bytes` and the streaming writer both use, so the three can
            never disagree about what the digest covers.

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
    selected = encoded_columns(table.schema.names) if columns is None else tuple(columns)
    payload = table if len(selected) == table.num_columns else table.select(list(selected))
    column_values: list[list[Any]] = [_column_values(column) for column in payload.columns]
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
