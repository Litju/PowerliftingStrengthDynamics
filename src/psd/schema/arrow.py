"""Arrow schema derivation.

PyArrow/Parquet is the persisted schema boundary. Arrow schemas are *derived*
from the Pydantic contracts so the persisted schema and the validated contract
cannot drift: changing a field on a model changes the Arrow type, and a
mismatch between the declared column order and the model is an error rather than
a silent surprise at read time.

Type mapping
------------

============================  ============================
Python                        Arrow
============================  ============================
``str``                       ``string``
``bool``                      ``bool_``
``int``                       ``int64``
``float``                     ``float64``
``datetime``                  ``timestamp("us", tz="UTC")``
``StrEnum``                   ``string``
``tuple[X, ...]``             ``list_<X>``
============================  ============================

Controlled vocabularies are stored as plain strings rather than Arrow
dictionaries: a Parquet file must stay readable without a dictionary decoder,
and membership is enforced by the contracts rather than by the physical type.

``Any`` appears only where Python's typing system cannot express "a resolved
annotation object"; the return types are precise.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from enum import StrEnum
from typing import Any, TypedDict, get_args, get_origin

import pyarrow as pa
from pydantic import BaseModel

__all__ = (
    "ArrowTypeMappingError",
    "ColumnDescription",
    "SchemaDescription",
    "arrow_type_for",
    "schema_for_model",
    "schema_json",
    "undeclared_fields",
)


class ColumnDescription(TypedDict):
    """Machine-readable description of one persisted column."""

    name: str
    type: str
    nullable: bool


class SchemaDescription(TypedDict):
    """Machine-readable description of one persisted table schema."""

    table: str
    schema_version: str
    columns: list[ColumnDescription]


_TIMESTAMP_UNIT = "us"
_TIMESTAMP_TIMEZONE = "UTC"
_EXPECTED_UNION_ARITY = 2

_SCALAR_TYPES: dict[Any, pa.DataType] = {
    str: pa.string(),
    bool: pa.bool_(),
    int: pa.int64(),
    float: pa.float64(),
    datetime: pa.timestamp(_TIMESTAMP_UNIT, tz=_TIMESTAMP_TIMEZONE),
}

_SEQUENCE_ORIGINS: tuple[Any, ...] = (tuple, list, set, frozenset)


class ArrowTypeMappingError(TypeError):
    """Raised when a Python annotation cannot be mapped to an Arrow type."""


def arrow_type_for(annotation: Any) -> pa.DataType:
    """Return the Arrow type for a resolved Python annotation.

    Args:
        annotation: A resolved annotation, as produced by Pydantic.

    Returns:
        The corresponding Arrow type.

    Raises:
        ArrowTypeMappingError: The annotation is not supported by the canonical
            schema boundary.
    """
    scalar = _SCALAR_TYPES.get(annotation)
    if scalar is not None:
        return scalar
    if isinstance(annotation, type) and issubclass(annotation, StrEnum):
        return pa.string()

    origin = get_origin(annotation)
    args: tuple[Any, ...] = get_args(annotation)
    if origin in _SEQUENCE_ORIGINS:
        return pa.list_(_sequence_element_type(annotation, args))
    if origin is not None:
        return _union_type(annotation, args)
    msg = f"No Arrow mapping for annotation {annotation!r}."
    raise ArrowTypeMappingError(msg)


def _sequence_element_type(annotation: Any, args: tuple[Any, ...]) -> pa.DataType:
    """Return the Arrow type of a homogeneous sequence's element type."""
    element_types: list[Any] = [
        arg for arg in args if arg is not Ellipsis and arg is not type(None)
    ]
    if len(element_types) != 1:
        msg = (
            f"Unsupported sequence annotation {annotation!r} in the canonical schema; "
            "only homogeneous sequences are persisted."
        )
        raise ArrowTypeMappingError(msg)
    return arrow_type_for(element_types[0])


def _union_type(annotation: Any, args: tuple[Any, ...]) -> pa.DataType:
    """Return the Arrow type for ``Optional[T]``; reject anything else."""
    payload: list[Any] = [arg for arg in args if arg is not type(None)]
    if len(payload) == 1 and len(args) == _EXPECTED_UNION_ARITY:
        return arrow_type_for(payload[0])
    msg = f"Union annotations other than Optional[T] are unsupported: {annotation!r}."
    raise ArrowTypeMappingError(msg)


def _column_types(model: type[BaseModel], columns: Sequence[str]) -> list[tuple[str, pa.DataType]]:
    """Return ``(name, type)`` pairs for *columns*.

    Declared columns must all exist on the model and must be unique. A *subset*
    of the model's fields is allowed: the registry deliberately withholds
    temporal columns that a record's semantics forbid, and it is the registry's
    job to assert that the withheld set is exactly the permitted one.
    """
    fields = model.model_fields
    declared = tuple(columns)
    unknown = sorted(set(declared) - set(fields))
    if unknown:
        msg = f"Declared columns not on {model.__name__}: {', '.join(unknown)}."
        raise ArrowTypeMappingError(msg)
    if len(set(declared)) != len(declared):
        msg = f"Duplicate columns declared for {model.__name__}."
        raise ArrowTypeMappingError(msg)
    return [(name, arrow_type_for(fields[name].annotation)) for name in declared]


def undeclared_fields(model: type[BaseModel], columns: Sequence[str]) -> tuple[str, ...]:
    """Return model fields deliberately not persisted for *model*."""
    return tuple(name for name in model.model_fields if name not in set(columns))


def schema_for_model[ModelT: BaseModel](model: type[ModelT], columns: Sequence[str]) -> pa.Schema:
    """Build an Arrow schema for *model* using the declared column order.

    Args:
        model: The Pydantic contract.
        columns: Explicit persisted column order. Every name must be a field of
            *model*, and names must be unique.

    Returns:
        The Arrow schema, in declared order.

    Raises:
        ArrowTypeMappingError: A declared column is not a field of the model, or
            a column is declared twice.
    """
    pairs = _column_types(model, columns)
    return pa.schema([pa.field(name, data_type, nullable=True) for name, data_type in pairs])


def _describe_schema(schema: pa.Schema) -> list[ColumnDescription]:
    """Return per-column descriptions for an Arrow schema.

    Reads ``names``/``types`` in lockstep rather than indexing fields, because
    ``pyarrow.Field`` is generic and the community stubs do not parameterize it.
    Every canonical column is nullable by construction, so nullability follows the
    column invariant rather than ``Field.nullable``.
    """
    described: list[ColumnDescription] = []
    for name, data_type in zip(schema.names, schema.types, strict=True):
        described.append({"name": name, "type": str(data_type), "nullable": True})
    return described


def schema_json(schema: pa.Schema, *, table_name: str, schema_version: str) -> SchemaDescription:
    """Return a JSON-serializable description of an Arrow schema.

    Used by ``psd schema show``/``psd schema dump`` so the persisted schema can be
    diffed across versions and reviewed without a Parquet reader.
    """
    return {
        "table": table_name,
        "schema_version": schema_version,
        "columns": _describe_schema(schema),
    }
