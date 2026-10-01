"""Unit tests for Arrow schema derivation from the Pydantic contracts."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

import pyarrow as pa
import pytest
from pydantic import BaseModel, ConfigDict

from psd.schema.arrow import (
    ArrowTypeMappingError,
    arrow_type_for,
    schema_for_model,
    schema_json,
    undeclared_fields,
)
from psd.schema.models import PerformedSetRecord
from psd.schema.registry import arrow_schema_for, column_order, table_spec


class _Flavour(StrEnum):
    """Test-only controlled vocabulary."""

    ONE = "one"


class _Scalars(BaseModel):
    """Model covering every supported scalar mapping."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    text: str
    flag: bool
    count: int
    ratio: float
    moment: datetime
    flavour: _Flavour
    maybe_text: str | None = None
    tags: tuple[str, ...] = ()


_ALL_SCALARS: tuple[str, ...] = tuple(_Scalars.model_fields)


def test_scalar_type_mapping() -> None:
    assert arrow_type_for(str) == pa.string()
    assert arrow_type_for(bool) == pa.bool_()
    assert arrow_type_for(int) == pa.int64()
    assert arrow_type_for(float) == pa.float64()


def test_datetime_maps_to_utc_microsecond_timestamp() -> None:
    data_type = arrow_type_for(datetime)
    assert pa.types.is_timestamp(data_type)
    assert data_type.unit == "us"
    assert data_type.tz == "UTC"


def test_str_enum_maps_to_plain_string() -> None:
    """Vocabularies stay readable without a dictionary decoder."""
    assert arrow_type_for(_Flavour) == pa.string()


def test_optional_and_sequence_mapping() -> None:
    assert arrow_type_for(str | None) == pa.string()
    assert arrow_type_for(tuple[str, ...]) == pa.list_(pa.string())
    assert arrow_type_for(list[str]) == pa.list_(pa.string())


def test_unsupported_annotation_is_rejected() -> None:
    class _Unsupported:
        pass

    with pytest.raises(ArrowTypeMappingError):
        arrow_type_for(_Unsupported)


def test_unsupported_union_is_rejected() -> None:
    with pytest.raises(ArrowTypeMappingError, match="Union annotations"):
        arrow_type_for(str | int)


def test_unsupported_sequence_is_rejected() -> None:
    with pytest.raises(ArrowTypeMappingError, match="homogeneous"):
        arrow_type_for(tuple[str, int])


def test_schema_for_model_preserves_declared_order() -> None:
    reordered = (
        "moment",
        "maybe_text",
        "text",
        "flag",
        "count",
        "ratio",
        "flavour",
        "tags",
    )
    assert set(reordered) == set(_ALL_SCALARS)
    assert tuple(schema_for_model(_Scalars, reordered).names) == reordered


def test_schema_for_model_rejects_unknown_columns() -> None:
    with pytest.raises(ArrowTypeMappingError, match="not on _Scalars"):
        schema_for_model(_Scalars, (*_ALL_SCALARS, "not_a_field"))


def test_schema_for_model_rejects_duplicate_columns() -> None:
    with pytest.raises(ArrowTypeMappingError, match="Duplicate columns"):
        schema_for_model(_Scalars, (*_ALL_SCALARS, "text"))


def test_schema_for_model_allows_withheld_fields() -> None:
    """The registry withholds temporal columns a record's semantics forbid."""
    subset = ("text", "flag")
    assert tuple(schema_for_model(_Scalars, subset).names) == subset
    assert undeclared_fields(_Scalars, subset) == tuple(
        name for name in _ALL_SCALARS if name not in subset
    )


def test_schema_json_describes_columns() -> None:
    schema = schema_for_model(_Scalars, _ALL_SCALARS)
    described = schema_json(schema, table_name="_scalars", schema_version="psd-canonical/0.1.0")
    assert described["table"] == "_scalars"
    assert described["schema_version"] == "psd-canonical/0.1.0"
    assert described["columns"][0] == {"name": "text", "type": "string", "nullable": True}


def test_performed_set_schema_uses_utc_timestamps_and_lists() -> None:
    schema = arrow_schema_for("performed_set")
    fields = dict(zip(schema.names, schema.types, strict=True))
    assert pa.types.is_timestamp(fields["ingested_at"])
    assert fields["ingested_at"].tz == "UTC"
    assert pa.types.is_list(fields["quality_flags"])


def test_registry_column_order_matches_arrow_schema() -> None:
    spec = table_spec("performed_set")
    assert tuple(arrow_schema_for(spec.name).names) == spec.columns


def test_reimporting_the_registry_is_idempotent() -> None:
    """Schema derivation must be repeatable and side-effect free."""
    first = table_spec("athlete").arrow_schema()
    second = table_spec("athlete").arrow_schema()
    assert first.equals(second)


def test_utc_timestamps_round_trip_through_arrow() -> None:
    """Windows-native proof that timezone-aware timestamps survive Arrow."""
    moment = datetime(2026, 3, 14, 6, 30, tzinfo=UTC)
    schema = arrow_schema_for("performed_set")
    row: dict[str, Any] = dict.fromkeys(column_order("performed_set"))
    row["performed_set_id"] = "pset_0000000000000000000000000000000"
    row["performed_exercise_id"] = "pex_0000000000000000000000000000000"
    row["ordinal"] = 1
    row["ingested_at"] = moment
    table = pa.Table.from_pylist([row], schema=schema)
    read_back: dict[str, Any] = table.to_pylist()[0]
    assert read_back["ingested_at"] == moment
    assert read_back["load_kg"] is None


def test_performed_set_model_fields_are_persisted_or_withheld() -> None:
    withheld = undeclared_fields(PerformedSetRecord, column_order("performed_set"))
    assert set(withheld) == {"scheduled_at", "observed_at"}
