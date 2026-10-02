"""Declarative column checks over Polars frames.

The locked stack keeps PyArrow schemas and explicit custom validators
authoritative for canonical persisted tables. This module adds the declarative
*column* layer, which exists because corpus-scale validation checks columns, not
rows: a Python loop over millions of performed sets would not scale, while a
vectorized expression over the same column does.

Three overlays run over each table:

* **schema conformance** -- every canonical column present, no extras, and the
  Polars dtype equal to the canonical Arrow type;
* **numeric ranges and positivity** -- ordinals, RPE/RIR, percentages, link
  confidences, velocities, and a strict positivity rule for every load column,
  because zero is a missing-value sentinel rather than a load;
* **controlled-vocabulary membership** -- every enum column must hold only
  declared members.

Nulls are skipped throughout: a value that is absent is governed by the
missingness rules in :mod:`psd.validation.rules`, not by a range or vocabulary.

Implementation note
-------------------

This layer was previously delegated to ``pandera.polars``. That backend is
incompatible with the locked Polars release: on Polars 1.44 every check raises a
``DeprecationWarning`` from a deprecated ``concat`` path, which Pandera converts
into a schema failure, and its ``isin`` check additionally reports false failures
for values that *are* members of the allowed set. The checks below therefore run
as vectorized Polars expressions against the authoritative Arrow schema, which is
both correct on the locked versions and dependency-free. See the locked stack
document for the recorded amendment.

Vocabulary rules are keyed by ``(table, column)`` rather than by column name,
because a column such as ``method`` means different vocabularies in different
tables.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

import polars as pl
import pyarrow as pa

from psd.schema.registry import table_spec
from psd.schema.vocabulary import VOCABULARIES
from psd.validation.issues import Severity, ValidationIssue

__all__ = (
    "COLUMN_RANGES",
    "POSITIVE_COLUMNS",
    "VOCABULARY_BY_COLUMN",
    "column_issues",
    "vocabulary_issues",
)

#: Vocabulary name per ``(table, column)`` pair.
VOCABULARY_BY_COLUMN: Mapping[tuple[str, str], str] = {
    ("source", "nature"): "source_nature",
    ("source", "regime"): "data_regime",
    ("source", "consent_basis"): "consent_basis",
    ("source", "redistribution"): "redistribution",
    ("athlete", "identity_status"): "identity_status",
    ("athlete", "sex_category"): "sex_category",
    ("athlete_source_link", "link_method"): "identity_link_method",
    ("body_measurement", "measurement_type"): "body_measurement_type",
    ("body_measurement", "method"): "body_measurement_method",
    ("body_measurement", "measurement_context"): "body_mass_context",
    ("body_measurement", "event_time_precision"): "event_time_precision",
    ("body_measurement", "unit_normalized"): "normalized_measurement_unit",
    ("equipment_state", "equipment_item"): "equipment_item",
    ("exercise_definition", "parent_lift"): "parent_lift",
    ("exercise_definition", "specificity_level"): "specificity_level",
    ("exercise_definition", "laterality"): "laterality",
    ("exercise_definition", "implement"): "implement_type",
    ("exercise_definition", "bar_type"): "bar_type",
    ("exercise_definition", "equipment"): "equipment",
    ("exercise_definition", "stance"): "stance",
    ("exercise_definition", "grip"): "grip",
    ("exercise_definition", "range_of_motion"): "range_of_motion",
    ("exercise_definition", "pause_rule"): "pause_rule",
    ("exercise_definition", "tempo"): "tempo",
    ("exercise_definition", "configuration"): "configuration_flag",
    ("exercise_alias", "mapping_status"): "resolution_status",
    ("exercise_normalization", "resolution_status"): "resolution_status",
    ("exercise_normalization", "resolution_method"): "resolution_method",
    ("exercise_normalization", "parent_lift"): "parent_lift",
    ("exercise_normalization", "ambiguity_reason"): "ambiguity_reason",
    ("exercise_normalization", "normalization_rules"): "text_rule",
    ("planned_session", "session_status"): "session_status",
    ("planned_session", "not_performed_reason"): "not_performed_reason",
    ("planned_set", "prescription_basis"): "prescription_basis",
    ("program_modification", "change_kind"): "program_modification_kind",
    ("program_modification", "value_encoding"): "value_encoding",
    ("performed_session", "session_type"): "session_type",
    ("performed_set", "set_status"): "set_status",
    ("performed_set", "incomplete_reason"): "missingness_reason",
    ("performed_set", "load_unit"): "mass_unit",
    ("performed_rep", "rep_status"): "rep_status",
    ("performed_rep", "load_unit"): "mass_unit",
    ("observation", "observation_type"): "observation_type",
    ("observation", "observation_method"): "observation_method",
    ("observation", "observation_scope"): "observation_scope",
    ("observation", "reporter_role"): "reporter_role",
    ("performance_test", "test_type"): "test_type",
    ("velocity_observation", "method"): "velocity_method",
    ("velocity_observation", "load_unit"): "mass_unit",
    ("competition", "event_time_precision"): "event_time_precision",
    ("competition", "equipment_class"): "equipment_class",
    ("competition", "bodyweight_unit"): "mass_unit",
    ("competition_attempt", "lift"): "lift_type",
    ("competition_attempt", "result"): "attempt_result",
    ("competition_attempt", "attempt_order_basis"): "attempt_order_basis",
    ("competition_attempt", "load_unit"): "mass_unit",
    ("competition_reported_result", "result_kind"): "competition_result_kind",
    ("planned_set", "target_load_unit"): "mass_unit",
}

#: Inclusive ``(low, high)`` bounds per numeric column.
COLUMN_RANGES: Mapping[str, tuple[float, float]] = {
    "ordinal": (0.0, 1000.0),
    "rep_ordinal": (1.0, 1000.0),
    "target_reps": (0.0, 1000.0),
    "target_reps_min": (0.0, 1000.0),
    "target_reps_max": (0.0, 1000.0),
    "reps_performed": (0.0, 1000.0),
    "reps_failed": (0.0, 1000.0),
    "target_rpe": (0.0, 10.0),
    "target_rir": (0.0, 10.0),
    "rpe": (0.0, 10.0),
    "rir": (0.0, 10.0),
    "target_percent_one_rm": (0.0, 100.0),
    "velocity_loss_percent": (0.0, 100.0),
    "link_confidence": (0.0, 1.0),
    "attempt_number": (1.0, 3.0),
    "birth_year": (1900.0, 2100.0),
    "mean_velocity_mps": (0.0, 20.0),
    "peak_velocity_mps": (0.0, 30.0),
    "sampling_hz": (0.0, 1000.0),
    "duration_seconds": (0.0, 172800.0),
    "rest_actual_seconds": (0.0, 172800.0),
    "rest_target_seconds": (0.0, 172800.0),
    "target_duration_seconds": (0.0, 172800.0),
    "session_order_index": (0.0, 64.0),
}

#: Columns that must be strictly positive: zero is never a plausible load.
POSITIVE_COLUMNS: frozenset[str] = frozenset(
    {
        "load_raw",
        "load_kg",
        "target_load_raw",
        "target_load_kg",
        "bodyweight_raw",
        "bodyweight_kg",
        "value_normalized",
        "result_normalized",
    }
)

_ROW_INDEX_COLUMN: Final[str] = "__psd_row__"


def column_issues(
    frame: pl.DataFrame, *, table_name: str, limit: int = 50
) -> list[ValidationIssue]:
    """Run the declarative column layer over one table.

    Args:
        frame: Frame whose columns follow the canonical column order.
        table_name: Canonical table name.
        limit: Maximum number of issues to report, so a systematically corrupt
            artifact yields a readable report instead of millions of lines.

    Returns:
        Issues from schema conformance, numeric ranges, and vocabulary membership.
    """
    schema = table_spec(table_name).arrow_schema()
    issues: list[ValidationIssue] = _schema_issues(frame, schema, table_name)
    issues.extend(_numeric_issues(frame, schema, table_name, limit - len(issues)))
    issues.extend(vocabulary_issues(frame, table_name=table_name, limit=limit - len(issues)))
    return issues[:limit]


def _schema_issues(
    frame: pl.DataFrame, schema: pa.Schema, table_name: str
) -> list[ValidationIssue]:
    """Check column presence and dtype conformance."""
    issues: list[ValidationIssue] = []
    missing = [name for name in schema.names if name not in frame.columns]
    if missing:
        issues.append(
            ValidationIssue(
                code="missing_columns",
                severity=Severity.ERROR,
                table=table_name,
                message=f"frame is missing canonical column(s): {', '.join(missing)}",
            )
        )
    extra = [name for name in frame.columns if name not in schema.names]
    if extra:
        issues.append(
            ValidationIssue(
                code="unexpected_columns",
                severity=Severity.ERROR,
                table=table_name,
                message=(
                    f"frame carries column(s) outside the canonical schema: {', '.join(extra)}"
                ),
            )
        )
    for name, declared_type in zip(schema.names, schema.types, strict=True):
        if name not in frame.columns:
            continue
        expected = _expected_polars_dtype(declared_type)
        actual = frame.schema[name]
        if expected is not None and actual != expected:
            issues.append(
                ValidationIssue(
                    code=f"dtype_{name}",
                    severity=Severity.ERROR,
                    table=table_name,
                    message=(
                        f"column {name!r} has dtype {actual!s}; the canonical Arrow schema "
                        f"requires {expected!s}"
                    ),
                )
            )
    return issues


_SIMPLE_DTYPES: Mapping[Any, Any] = {
    pa.string(): pl.String,
    pa.bool_(): pl.Boolean,
    pa.int64(): pl.Int64,
    pa.float64(): pl.Float64,
}


def _expected_polars_dtype(arrow_type: pa.DataType) -> pl.DataType | None:
    """Return the Polars dtype a canonical Arrow type must present as."""
    simple = _SIMPLE_DTYPES.get(arrow_type)
    if simple is not None:
        return simple
    if pa.types.is_timestamp(arrow_type):
        return pl.Datetime(arrow_type.unit, arrow_type.tz)
    if pa.types.is_list(arrow_type):
        inner = _expected_polars_dtype(arrow_type.value_type)
        return None if inner is None else pl.List(inner)
    return None


def _numeric_issues(
    frame: pl.DataFrame, schema: pa.Schema, table_name: str, limit: int
) -> list[ValidationIssue]:
    """Check inclusive ranges and strict positivity.

    Predicates select *violating* rows; nulls evaluate to null and are therefore
    skipped, because absence is governed by the missingness rules rather than by a
    range.
    """
    issues: list[ValidationIssue] = []
    for name in schema.names:
        if limit <= 0:
            break
        if name not in frame.columns:
            continue
        if name in POSITIVE_COLUMNS:
            predicate = pl.col(name) <= 0.0
            description = "must be greater than 0"
        elif name in COLUMN_RANGES:
            low, high = COLUMN_RANGES[name]
            # Compared with explicit comparisons rather than ``is_between`` so that
            # integer columns are checked against the float bounds numerically.
            predicate = ~((pl.col(name) >= low) & (pl.col(name) <= high))
            description = f"must be within [{low}, {high}]"
        else:
            continue
        issues.extend(
            _offenders(
                frame,
                _ColumnProbe(
                    name=name,
                    predicate=predicate,
                    description=description,
                    code_prefix="range",
                ),
                table_name=table_name,
                limit=limit - len(issues),
            )
        )
    return issues


def vocabulary_issues(
    frame: pl.DataFrame, *, table_name: str, limit: int = 50
) -> list[ValidationIssue]:
    """Assert controlled-vocabulary membership, column by column.

    List columns such as ``quality_flags`` are checked element by element, and the
    original row index is preserved so a finding points back at the record.

    Args:
        frame: Frame whose columns follow the canonical column order.
        table_name: Canonical table name.
        limit: Maximum number of issues to report.

    Returns:
        Issues naming the offending column, value, and row index.

    Raises:
        KeyError: A rule references an unknown vocabulary.
    """
    issues: list[ValidationIssue] = []
    for (table, column), vocabulary in VOCABULARY_BY_COLUMN.items():
        if table != table_name or column not in frame.columns:
            continue
        members = VOCABULARIES.get(vocabulary)
        if members is None:
            msg = f"Column {table}.{column} references unknown vocabulary {vocabulary!r}."
            raise KeyError(msg)
        issues.extend(
            _offenders(
                frame,
                _ColumnProbe(
                    name=column,
                    predicate=~pl.col(column).is_in(list(members)),
                    description=f"is not a member of vocabulary {vocabulary!r}",
                    code_prefix="vocabulary",
                    explode=isinstance(frame.schema[column], pl.List),
                ),
                table_name=table_name,
                limit=limit - len(issues),
            )
        )
        if len(issues) >= limit:
            break
    return issues


@dataclass(frozen=True, slots=True)
class _ColumnProbe:
    """One vectorized column check.

    Attributes:
        name: Column being checked.
        predicate: Expression that is true for offending rows.
        description: Human-readable statement of the rule.
        code_prefix: ``range`` or ``vocabulary``.
        explode: Whether the column must be exploded before filtering.
    """

    name: str
    predicate: pl.Expr
    description: str
    code_prefix: str
    explode: bool = False


def _offenders(
    frame: pl.DataFrame,
    probe: _ColumnProbe,
    *,
    table_name: str,
    limit: int,
) -> list[ValidationIssue]:
    """Return issues for rows where the probe predicate holds.

    Row indices are taken before any explode, so a finding points back at the
    original record rather than at an exploded element position.
    """
    if limit <= 0:
        return []
    indexed = frame.with_row_index(_ROW_INDEX_COLUMN)
    if probe.explode:
        # ``empty_as_null`` is stated explicitly because Polars 2.0 will flip its
        # default, and PSD runs with warnings as errors: an ambiguous default would
        # turn a vocabulary check into a hard failure on the next Polars release.
        # An empty list must stay an empty string being checked, not a null that
        # silently skips the check.
        indexed = indexed.explode(probe.name, empty_as_null=True)
    offending = indexed.filter(probe.predicate)
    if offending.height == 0:
        return []
    indices: list[int] = offending.get_column(_ROW_INDEX_COLUMN).to_list()
    values: list[Any] = offending.get_column(probe.name).to_list()
    return [
        ValidationIssue(
            code=f"{probe.code_prefix}_{probe.name}",
            severity=Severity.ERROR,
            table=table_name,
            record_id=f"row:{index}",
            message=(
                f"column {probe.name!r} holds {value!r}, which {probe.description}; the "
                "persisted column contains a value the canonical contract could not have "
                "produced"
            ),
        )
        for index, value in zip(indices, values, strict=True)
    ][:limit]


def vocabulary_names() -> Sequence[str]:
    """Return every vocabulary name referenced by a column rule."""
    return sorted(set(VOCABULARY_BY_COLUMN.values()))
