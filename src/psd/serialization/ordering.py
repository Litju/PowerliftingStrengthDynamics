"""Canonical table ordering.

Persisted artifacts are sorted before hashing or writing. The ordering comes from
the table registry, where every ``order_by`` ends with the table's primary key, so
the order is *total*: no two rows can compare equal and the result cannot depend
on the input row order.

Polars performs the sort. It is the primary tabular transformation engine in the
locked stack, and its multi-column sort with an explicit null placement is
deterministic. Null placement is fixed to ``nulls_last`` so that "not recorded"
rows sort predictably rather than by engine default.

Polars widens Arrow strings to its 64-bit ``Utf8`` representation, so the sorted
frame is cast back to the input schema before it is returned. Canonical ordering
must not change the physical schema; otherwise the same logical table would have
two persisted forms.

If a caller supplies an ordering that is not total, that is an error rather than a
silently non-reproducible artifact.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import polars as pl
import pyarrow as pa

from psd.schema.registry import TableSpec, table_spec

__all__ = ("OrderingError", "canonical_order", "canonical_order_polars", "from_arrow_frame")

#: Polars `from_arrow` is typed as returning `DataFrame | Series` with unknown
#: parameters upstream; the shim pins the concrete frame type PSD always gets.
from_arrow_frame: Callable[[pa.Table], pl.DataFrame] = getattr(pl, "from_arrow")


class OrderingError(ValueError):
    """Raised when a requested ordering cannot produce a total order."""


def canonical_order(table: pa.Table, order_by: Sequence[str], *, table_name: str) -> pa.Table:
    """Return *table* sorted by its declared total ordering.

    Polars is the primary transformation engine and performs the sort. It maps
    Arrow ``string`` to its 64-bit ``Utf8`` representation, so the sorted frame is
    cast back to the input schema. Without that cast the physical Arrow types
    would drift (``string`` to ``large_string``, ``list<string>`` to
    ``large_list<...>``) and a sorted table would no longer match the canonical
    schema it claims to implement.

    Args:
        table: Table to sort.
        order_by: Ordering columns, which must include every primary-key column.
        table_name: Canonical table name, used to look up the primary key.

    Returns:
        A new Arrow table in canonical order, with the input schema preserved.

    Raises:
        OrderingError: The ordering omits a primary-key column, so the sort would
            not be a total order.
    """
    spec = table_spec(table_name)
    _require_total_order(spec, order_by)
    frame = canonical_order_polars(from_arrow_frame(table), order_by)
    sorted_table = frame.to_arrow()
    return sorted_table.cast(table.schema)


def canonical_order_polars(frame: pl.DataFrame, order_by: Sequence[str]) -> pl.DataFrame:
    """Return *frame* sorted by *order_by* with nulls last.

    Args:
        frame: Frame to sort.
        order_by: Ordering columns, ascending.

    Returns:
        A new sorted frame.
    """
    missing = [column for column in order_by if column not in frame.columns]
    if missing:
        msg = f"Cannot order by unknown column(s): {', '.join(missing)}."
        raise OrderingError(msg)
    return frame.sort(by=list(order_by), nulls_last=True, maintain_order=True)


def _require_total_order(spec: TableSpec, order_by: Sequence[str]) -> None:
    """Reject an ordering that leaves ties, which would be order-dependent."""
    missing = [column for column in spec.primary_key if column not in order_by]
    if missing:
        msg = (
            f"Ordering for table {spec.name!r} omits primary-key column(s) "
            f"{', '.join(missing)}; the resulting order would not be total and the "
            "artifact would not be reproducible."
        )
        raise OrderingError(msg)
