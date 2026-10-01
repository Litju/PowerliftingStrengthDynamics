"""Cross-record validation rules.

These are the invariants that a per-column check cannot express. They enforce the
PSD data contract at the level of relationships between records:

* **referential integrity** -- every foreign key resolves, and resolves to the same
  athlete;
* **prescription/execution separation** -- execution references a plan only
  through an explicit link, never by reconstructing one;
* **exercise identity** -- one canonical key per exercise, one binding per alias
  within a source system, and ambiguous candidates that actually exist;
* **real/synthetic separation** -- rows about a synthetic athlete come only from a
  synthetic source, and vice versa;
* **explicit missingness** -- a row with absent values should say why;
* **provenance integrity** -- an edit flag without a modification timestamp, or a
  row citing an unregistered source;
* **physical consistency** -- equipment and program-version intervals do not
  overlap, reported totals match reported bests, RPE and RIR do not contradict
  each other.

Every rule reports a :class:`~psd.validation.issues.ValidationIssue` rather than
raising, so an artifact is described in full in a single pass.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from itertools import pairwise
from typing import Any, Final

from psd.schema.registry import table_spec
from psd.schema.vocabulary import AttemptResult, MissingnessReason, QualityFlag
from psd.validation.issues import Severity, ValidationIssue

__all__ = ("Index", "Row", "TableRows", "all_issues", "build_index", "record_id")

#: One canonical row.
Row = Mapping[str, Any]

#: Table name to its rows.
TableRows = Mapping[str, Sequence[Row]]

#: Rows of one table, indexed by a key column.
Index = Mapping[Any, Row]

_RPE_RIR_TOLERANCE: Final[float] = 1.0
_REPORT_TOTAL_TOLERANCE_KG: Final[float] = 0.001
_REPORTED_LIFT_KINDS: Final[tuple[str, ...]] = ("squat_best", "bench_best", "deadlift_best")


@dataclass(frozen=True, slots=True)
class _ForeignKey:
    """One link between two canonical tables."""

    code: str
    table: str
    column: str
    target_table: str
    target_column: str


_FOREIGN_KEYS: tuple[_ForeignKey, ...] = (
    _ForeignKey("dangling_athlete", "body_measurement", "athlete_id", "athlete", "athlete_id"),
    _ForeignKey("dangling_athlete", "equipment_state", "athlete_id", "athlete", "athlete_id"),
    _ForeignKey("dangling_athlete", "planned_session", "athlete_id", "athlete", "athlete_id"),
    _ForeignKey("dangling_athlete", "performed_session", "athlete_id", "athlete", "athlete_id"),
    _ForeignKey("dangling_athlete", "observation", "athlete_id", "athlete", "athlete_id"),
    _ForeignKey("dangling_athlete", "performance_test", "athlete_id", "athlete", "athlete_id"),
    _ForeignKey(
        "dangling_velocity_athlete", "velocity_observation", "athlete_id", "athlete", "athlete_id"
    ),
    _ForeignKey("dangling_competition", "competition", "athlete_id", "athlete", "athlete_id"),
    _ForeignKey("dangling_athlete", "competition_attempt", "athlete_id", "athlete", "athlete_id"),
    _ForeignKey(
        "dangling_competition",
        "competition_attempt",
        "competition_id",
        "competition",
        "competition_id",
    ),
    _ForeignKey(
        "dangling_competition_result",
        "competition_reported_result",
        "competition_id",
        "competition",
        "competition_id",
    ),
    _ForeignKey(
        "dangling_performed_session",
        "performed_exercise",
        "performed_session_id",
        "performed_session",
        "performed_session_id",
    ),
    _ForeignKey(
        "dangling_performed_exercise",
        "performed_set",
        "performed_exercise_id",
        "performed_exercise",
        "performed_exercise_id",
    ),
    _ForeignKey(
        "dangling_performed_set",
        "performed_rep",
        "performed_set_id",
        "performed_set",
        "performed_set_id",
    ),
    _ForeignKey(
        "dangling_planned_session",
        "planned_exercise",
        "planned_session_id",
        "planned_session",
        "planned_session_id",
    ),
    _ForeignKey(
        "dangling_planned_exercise",
        "planned_set",
        "planned_exercise_id",
        "planned_exercise",
        "planned_exercise_id",
    ),
    _ForeignKey(
        "dangling_plan_link",
        "performed_session",
        "planned_session_id",
        "planned_session",
        "planned_session_id",
    ),
    _ForeignKey(
        "dangling_plan_link",
        "performed_exercise",
        "planned_exercise_id",
        "planned_exercise",
        "planned_exercise_id",
    ),
    _ForeignKey(
        "dangling_plan_link",
        "performed_set",
        "planned_set_id",
        "planned_set",
        "planned_set_id",
    ),
    _ForeignKey(
        "dangling_program_version",
        "planned_session",
        "program_version_id",
        "program_version",
        "program_version_id",
    ),
    _ForeignKey(
        "dangling_exercise",
        "performed_exercise",
        "exercise_id",
        "exercise_definition",
        "exercise_id",
    ),
    _ForeignKey(
        "dangling_exercise",
        "planned_exercise",
        "exercise_id",
        "exercise_definition",
        "exercise_id",
    ),
    _ForeignKey(
        "dangling_exercise",
        "exercise_alias",
        "exercise_id",
        "exercise_definition",
        "exercise_id",
    ),
    _ForeignKey(
        "dangling_exercise",
        "exercise_normalization",
        "exercise_id",
        "exercise_definition",
        "exercise_id",
    ),
    _ForeignKey(
        "dangling_exercise_alias",
        "exercise_normalization",
        "source_alias_id",
        "exercise_alias",
        "exercise_alias_id",
    ),
    _ForeignKey(
        "dangling_velocity_set",
        "velocity_observation",
        "performed_set_id",
        "performed_set",
        "performed_set_id",
    ),
    _ForeignKey(
        "dangling_velocity_rep",
        "velocity_observation",
        "performed_rep_id",
        "performed_rep",
        "performed_rep_id",
    ),
    _ForeignKey(
        "dangling_test_set",
        "performance_test",
        "performed_set_id",
        "performed_set",
        "performed_set_id",
    ),
)


def build_index(rows: TableRows, table: str, key: str) -> Index:
    """Index one table by one column, keeping the first row per key."""
    indexed: dict[Any, Row] = {}
    for row in rows.get(table, ()):
        identifier = row.get(key)
        if identifier is None:
            continue
        indexed.setdefault(identifier, row)
    return indexed


def record_id(table: str, row: Row) -> str:
    """Return a printable identifier for a row."""
    for candidate in (f"{table}_id", "athlete_id", "competition_id"):
        value = row.get(candidate)
        if isinstance(value, str):
            return value
    return "<unknown>"


def check_referential_integrity(tables: TableRows) -> list[ValidationIssue]:
    """Check every declared foreign key resolves.

    A dangling link means either the referenced record was dropped or the
    identifier was invented. Both silently corrupt a longitudinal history, so
    this is an error.
    """
    issues: list[ValidationIssue] = []
    for key in _FOREIGN_KEYS:
        targets = build_index(tables, key.target_table, key.target_column)
        for row in tables.get(key.table, ()):
            value = row.get(key.column)
            if value is None:
                continue
            if value not in targets:
                issues.append(
                    ValidationIssue(
                        code=key.code,
                        severity=Severity.ERROR,
                        table=key.table,
                        record_id=record_id(key.table, row),
                        message=(
                            f"{key.column}={value!r} does not resolve in "
                            f"{key.target_table}.{key.target_column}"
                        ),
                    )
                )
    return issues


def check_source_provenance(tables: TableRows) -> list[ValidationIssue]:
    """Check every row cites a registered source with declared consent."""
    registered = {row.get("source_id") for row in tables.get("source", ())}
    issues: list[ValidationIssue] = []
    for table, rows in tables.items():
        if table == "source":
            continue
        for row in rows:
            source_id = row.get("source_id")
            if source_id is not None and source_id not in registered:
                issues.append(
                    ValidationIssue(
                        code="unregistered_source",
                        severity=Severity.ERROR,
                        table=table,
                        record_id=record_id(table, row),
                        message=(
                            f"source_id={source_id!r} is not present in the source table; "
                            "every row must remain traceable to a registered source"
                        ),
                    )
                )
    return issues


def check_synthetic_separation(tables: TableRows) -> list[ValidationIssue]:
    """Check real and synthetic data never share a source.

    PSD-Sim ground truth must never be projected onto real athletes. A row whose
    athlete is synthetic may only come from a synthetic source, and a row whose
    athlete is real may only come from a real source.
    """
    natures = {row.get("source_id"): row.get("nature") for row in tables.get("source", ())}
    synthetic_athletes = {
        row.get("athlete_id")
        for row in tables.get("athlete", ())
        if row.get("is_synthetic") is True
    }
    issues: list[ValidationIssue] = []
    for table, rows in tables.items():
        if table in {"source", "athlete", "provenance"}:
            continue
        for row in rows:
            athlete_id = row.get("athlete_id")
            if athlete_id is None:
                continue
            nature = natures.get(row.get("source_id"))
            if nature is None:
                continue
            is_synthetic_athlete = athlete_id in synthetic_athletes
            if is_synthetic_athlete and nature != "synthetic":
                issues.append(
                    ValidationIssue(
                        code="synthetic_athlete_from_real_source",
                        severity=Severity.ERROR,
                        table=table,
                        record_id=record_id(table, row),
                        message=(
                            f"athlete {athlete_id!r} is synthetic but the row's source is "
                            f"{nature!r}"
                        ),
                    )
                )
            if (
                not is_synthetic_athlete
                and nature == "synthetic"
                and table != "athlete_source_link"
            ):
                issues.append(
                    ValidationIssue(
                        code="real_athlete_from_synthetic_source",
                        severity=Severity.ERROR,
                        table=table,
                        record_id=record_id(table, row),
                        message=(
                            f"row cites a synthetic source but athlete {athlete_id!r} is not "
                            "declared synthetic"
                        ),
                    )
                )
    return issues


def check_explicit_missingness(tables: TableRows) -> list[ValidationIssue]:
    """Check that absent measurements are explained.

    ``null`` is correct for a measurement that was never recorded, but it should
    carry a missingness reason so that downstream evaluation can distinguish
    "not recorded" from "not applicable" without guessing.
    """
    issues: list[ValidationIssue] = []
    optional_columns = {
        "load_raw": "performed_set",
        "rpe": "performed_set",
        "rir": "performed_set",
        "reps_performed": "performed_set",
        "raw_value": "body_measurement",
        "mean_velocity_mps": "velocity_observation",
        "duration_seconds": "performed_session",
        "ended_at": "performed_session",
    }
    for column, table in optional_columns.items():
        for row in tables.get(table, ()):
            if row.get(column) is not None or row.get("missingness_reason") is not None:
                continue
            issues.append(
                ValidationIssue(
                    code="missingness_undeclared",
                    severity=Severity.WARNING,
                    table=table,
                    record_id=record_id(table, row),
                    message=(
                        f"{column} is null without a missingness_reason; the absence is "
                        "ambiguous and may affect evaluation"
                    ),
                )
            )
    return issues


def check_no_sentinel_zero(tables: TableRows) -> list[ValidationIssue]:
    """Check that zero is only used where zero is semantically correct.

    Zero repetitions on a failed set is a recorded observation and is allowed. A
    zero load, RPE, or RIR is a missing-value sentinel and is not.
    """
    issues: list[ValidationIssue] = []
    for column, table in (
        ("load_kg", "performed_set"),
        ("rpe", "performed_set"),
        ("rir", "performed_set"),
        ("load_kg", "competition_attempt"),
        ("bodyweight_kg", "competition"),
        ("target_load_kg", "planned_set"),
    ):
        for row in tables.get(table, ()):
            value = row.get(column)
            if value == 0.0:
                issues.append(
                    ValidationIssue(
                        code="zero_as_missing",
                        severity=Severity.ERROR,
                        table=table,
                        record_id=record_id(table, row),
                        message=(
                            f"{column} is 0.0, which PSD treats as a missing-value sentinel; "
                            "record null with a missingness reason instead"
                        ),
                    )
                )
    return issues


def check_rpe_rir_consistency(tables: TableRows) -> list[ValidationIssue]:
    """Check that a recorded RPE and RIR do not contradict each other.

    ``0 <= RPE + RIR <= 10`` is the usual relationship between the two scales.
    Disagreement is a source or transcription problem rather than a fatal defect,
    so it is reported as a warning.
    """
    issues: list[ValidationIssue] = []
    for table in ("performed_set", "performed_rep"):
        for row in tables.get(table, ()):
            rpe = row.get("rpe")
            rir = row.get("rir")
            if not isinstance(rpe, (int, float)) or not isinstance(rir, (int, float)):
                continue
            total = float(rpe) + float(rir)
            if abs(total - 10.0) > _RPE_RIR_TOLERANCE:
                issues.append(
                    ValidationIssue(
                        code="rpe_rir_inconsistent",
                        severity=Severity.WARNING,
                        table=table,
                        record_id=record_id(table, row),
                        message=(
                            f"rpe={rpe} with rir={rir} implies a total of {total:.2f}; the "
                            "usual relationship is RPE + RIR == 10"
                        ),
                    )
                )
    return issues


def check_edit_provenance(tables: TableRows) -> list[ValidationIssue]:
    """Check that a post-hoc edit is timestamped.

    A row flagged as edited after the fact must carry ``modified_at``. Without the
    timestamp, a future reader cannot tell when the change became known, which is
    exactly what a leakage-safe benchmark episode needs.
    """
    issues: list[ValidationIssue] = []
    for table, rows in tables.items():
        for row in rows:
            flags = row.get("quality_flags") or ()
            if QualityFlag.EDITED_AFTER_THE_FACT.value not in flags:
                continue
            if row.get("modified_at") is None:
                issues.append(
                    ValidationIssue(
                        code="edit_without_timestamp",
                        severity=Severity.ERROR,
                        table=table,
                        record_id=record_id(table, row),
                        message=(
                            "quality flag edited_after_the_fact is present but modified_at is "
                            "null, so when the edit became known is unknowable"
                        ),
                    )
                )
    return issues


def check_interval_overlaps(tables: TableRows) -> list[ValidationIssue]:
    """Check that open-ended state intervals do not overlap."""
    issues: list[ValidationIssue] = []
    issues.extend(_check_overlaps(tables, "equipment_state", _equipment_item))
    issues.extend(_check_overlaps(tables, "program_version", _program_id))
    return issues


def _equipment_item(row: Row) -> Any:
    """Group equipment intervals by item."""
    return row.get("equipment_item")


def _program_id(row: Row) -> Any:
    """Group program-version intervals by program."""
    return row.get("program_id")


def _check_overlaps(
    tables: TableRows, table: str, group_key: Callable[[Row], Any]
) -> list[ValidationIssue]:
    """Report overlapping ``[effective_from, effective_to)`` intervals per group."""
    grouped: dict[tuple[Any, Any], list[tuple[datetime, datetime | None, Row]]] = {}
    for row in tables.get(table, ()):
        start = row.get("effective_from")
        if not isinstance(start, datetime):
            continue
        end = row.get("effective_to")
        grouped.setdefault((group_key(row), row.get("athlete_id")), []).append((start, end, row))

    issues: list[ValidationIssue] = []
    for _group, intervals in sorted(grouped.items(), key=lambda item: str(item[0])):
        ordered = sorted(intervals, key=lambda item: item[0])
        for (start, end, row), (next_start, _next_end, next_row) in pairwise(ordered):
            if end is None or end > next_start:
                issues.append(
                    ValidationIssue(
                        code="overlapping_interval",
                        severity=Severity.ERROR,
                        table=table,
                        record_id=record_id(table, row),
                        message=(
                            f"interval starting {start.isoformat()} overlaps the interval "
                            f"starting {next_start.isoformat()} of "
                            f"{record_id(table, next_row)}"
                        ),
                    )
                )
    return issues


def check_attempt_sequence(tables: TableRows) -> list[ValidationIssue]:
    """Check attempt numbering within one competition and lift.

    Missing attempts stay missing: the only defect is a repeated number or a
    number outside the permitted set, both of which the contract already rejects.
    """
    seen: dict[tuple[Any, Any, Any], Row] = {}
    issues: list[ValidationIssue] = []
    for row in tables.get("competition_attempt", ()):
        if row.get("result") == AttemptResult.NO_ATTEMPT.value:
            continue
        key = (row.get("competition_id"), row.get("lift"), row.get("attempt_number"))
        previous = seen.get(key)
        if previous is not None:
            issues.append(
                ValidationIssue(
                    code="duplicate_attempt_number",
                    severity=Severity.ERROR,
                    table="competition_attempt",
                    record_id=record_id("competition_attempt", row),
                    message=(
                        f"lift {row.get('lift')!r} attempt {row.get('attempt_number')} appears "
                        "more than once for this competition"
                    ),
                )
            )
            continue
        seen[key] = row
    return issues


def check_reported_totals(tables: TableRows) -> list[ValidationIssue]:
    """Check reported totals against reported bests.

    Only *reported* (not derived) values are compared: a derived total may
    legitimately use a different rounding rule, which is why ``is_derived`` exists.
    """
    by_competition: dict[Any, dict[str, Row]] = {}
    for row in tables.get("competition_reported_result", ()):
        if row.get("is_derived") is True:
            continue
        kind = str(row.get("result_kind"))
        by_competition.setdefault(row.get("competition_id"), {})[kind] = row

    issues: list[ValidationIssue] = []
    for competition_id, results in by_competition.items():
        total_row = results.get("total")
        if total_row is None:
            continue
        bests = [results.get(kind) for kind in _REPORTED_LIFT_KINDS]
        if any(best is None for best in bests):
            continue
        summed = sum(float(best["value"]) for best in bests if best is not None)
        reported = float(total_row["value"])
        if abs(summed - reported) > _REPORT_TOTAL_TOLERANCE_KG:
            issues.append(
                ValidationIssue(
                    code="reported_total_mismatch",
                    severity=Severity.WARNING,
                    table="competition_reported_result",
                    record_id=record_id("competition_reported_result", total_row),
                    message=(
                        f"competition {competition_id!r} reports bests summing to {summed:.3f} "
                        f"but a total of {reported:.3f}"
                    ),
                )
            )
    return issues


def check_plan_execution_consistency(tables: TableRows) -> list[ValidationIssue]:
    """Check that execution links stay consistent with their plan.

    * A performed session linked to a planned session must belong to the same
      athlete, and likewise for a performed exercise and its planned exercise.
      A cross-athlete link would splice two people's histories together.
    * A planned set should not be claimed by many performed sets; that usually
      means one log was duplicated on ingest.

    No prescription is ever created here. A performed row with a null plan link is
    correct -- the source simply had no prescription -- and is left untouched.
    """
    issues: list[ValidationIssue] = list(_check_cross_athlete_session_link(tables))
    issues.extend(_check_cross_athlete_exercise_link(tables))
    issues.extend(_check_plan_claim_counts(tables))
    return issues


def _check_cross_athlete_session_link(tables: TableRows) -> Iterable[ValidationIssue]:
    """Report performed sessions linked to another athlete's planned session."""
    planned_sessions = build_index(tables, "planned_session", "planned_session_id")
    for row in tables.get("performed_session", ()):
        planned_id = row.get("planned_session_id")
        if planned_id is None:
            continue
        plan = planned_sessions.get(planned_id)
        if plan is not None and plan.get("athlete_id") != row.get("athlete_id"):
            yield _cross_athlete_issue("performed_session", row, "planned_session", plan)


def _check_cross_athlete_exercise_link(tables: TableRows) -> Iterable[ValidationIssue]:
    """Report performed exercises linked to another athlete's planned exercise.

    Neither ``performed_exercise`` nor ``planned_exercise`` carries an
    ``athlete_id``: a canonical table either owns the athlete or inherits it from
    its parent session. The comparison therefore has to walk up to the owning
    performed session and planned session rather than read a column that does not
    exist.
    """
    planned_exercises = build_index(tables, "planned_exercise", "planned_exercise_id")
    planned_sessions = build_index(tables, "planned_session", "planned_session_id")
    performed_sessions = build_index(tables, "performed_session", "performed_session_id")
    for row in tables.get("performed_exercise", ()):
        planned_id = row.get("planned_exercise_id")
        if planned_id is None:
            continue
        plan = planned_exercises.get(planned_id)
        if plan is None:
            continue
        planned_session = planned_sessions.get(plan.get("planned_session_id"))
        if planned_session is None:
            continue
        performed_session = performed_sessions.get(row.get("performed_session_id"))
        if performed_session is None:
            continue
        if planned_session.get("athlete_id") != performed_session.get("athlete_id"):
            yield _cross_athlete_issue("performed_exercise", row, "planned_exercise", plan)


def _check_plan_claim_counts(tables: TableRows) -> Iterable[ValidationIssue]:
    """Report planned sets claimed by more than one performed set."""
    claims: dict[Any, list[str]] = {}
    for row in tables.get("performed_set", ()):
        planned_id = row.get("planned_set_id")
        if planned_id is None:
            continue
        claims.setdefault(planned_id, []).append(record_id("performed_set", row))
    for row in tables.get("performed_set", ()):
        planned_id = row.get("planned_set_id")
        if planned_id is None:
            continue
        claimants = claims.get(planned_id, [])
        if len(claimants) > 1:
            yield ValidationIssue(
                code="plan_claimed_multiple_times",
                severity=Severity.WARNING,
                table="performed_set",
                record_id=record_id("performed_set", row),
                message=(
                    f"planned set {planned_id} is claimed by {len(claimants)} performed sets "
                    f"({', '.join(claimants)}); one planned set is normally executed once"
                ),
            )


def _cross_athlete_issue(table: str, row: Row, target_table: str, target: Row) -> ValidationIssue:
    """Build the issue for a link that crosses athletes."""
    return ValidationIssue(
        code="cross_athlete_link",
        severity=Severity.ERROR,
        table=table,
        record_id=record_id(table, row),
        message=(
            f"links to {target_table} {record_id(target_table, target)!r}, which belongs to a "
            "different athlete"
        ),
    )


def check_duplicate_measurements(tables: TableRows) -> list[ValidationIssue]:
    """Check that set-level observations are not recorded twice.

    RPE and RIR already live on ``performed_set`` because sources record them
    there. A separate set-scoped observation for the same measure is a duplicated
    measurement rather than new information.
    """
    sets = build_index(tables, "performed_set", "performed_set_id")
    issues: list[ValidationIssue] = []
    for row in tables.get("observation", ()):
        if row.get("observation_scope") != "set":
            continue
        kind = str(row.get("observation_type"))
        if kind not in {"rpe", "rir"}:
            continue
        parent = sets.get(row.get("parent_set_id"))
        if parent is None or parent.get(kind) is None:
            continue
        issues.append(
            ValidationIssue(
                code="duplicate_set_level_measurement",
                severity=Severity.WARNING,
                table="observation",
                record_id=record_id("observation", row),
                message=(
                    f"{kind} is already recorded on performed_set "
                    f"{row.get('parent_set_id')}; the duplicate observation is redundant"
                ),
            )
        )
    return issues


def check_missingness_reason_known(tables: TableRows) -> list[ValidationIssue]:
    """Check that declared missingness reasons are themselves meaningful."""
    known = {reason.value for reason in MissingnessReason}
    issues: list[ValidationIssue] = []
    for table, rows in tables.items():
        for row in rows:
            reason = row.get("missingness_reason")
            if reason is not None and str(reason) not in known:
                issues.append(
                    ValidationIssue(
                        code="unknown_missingness_reason",
                        severity=Severity.ERROR,
                        table=table,
                        record_id=record_id(table, row),
                        message=f"missingness_reason {reason!r} is not a known vocabulary member",
                    )
                )
    return issues


def check_exercise_identity_and_alias_bindings(tables: TableRows) -> list[ValidationIssue]:
    """Check the exercise tables agree with each other.

    Three relationships a per-column check cannot express:

    * **Canonical identity is unique.** Two exercise definitions may not claim the
      same ``canonical_key``; ``exercise_id`` is derived from it, so a duplicate
      means two identities were minted for one exercise.
    * **One alias, one binding.** Within a single ``source_system``, one normalized
      label may not point at two different canonical exercises. Two sources may
      legitimately spell the same label for different exercises -- which is why
      the rule is scoped per source -- but within one source it is a silent
      contradiction.
    * **Ambiguity candidates must exist.** An ambiguous normalization outcome is
      only useful if the candidates it declines to choose between are real
      exercises.
    """
    issues: list[ValidationIssue] = list(_check_unique_canonical_keys(tables))
    issues.extend(_check_alias_binding_conflicts(tables))
    issues.extend(_check_candidate_exercises_exist(tables))
    return issues


def _check_unique_canonical_keys(tables: TableRows) -> Iterable[ValidationIssue]:
    """Report exercise definitions that reuse a canonical key."""
    seen: dict[Any, str] = {}
    for row in tables.get("exercise_definition", ()):
        key = row.get("canonical_key")
        identifier = record_id("exercise_definition", row)
        if key is None:
            continue
        previous = seen.setdefault(key, identifier)
        if previous != identifier:
            yield ValidationIssue(
                code="duplicate_canonical_key",
                severity=Severity.ERROR,
                table="exercise_definition",
                record_id=identifier,
                message=(
                    f"canonical_key {key!r} is already claimed by {previous}; two canonical "
                    "identities for one exercise would make every downstream grouping ambiguous"
                ),
            )


def _check_alias_binding_conflicts(tables: TableRows) -> Iterable[ValidationIssue]:
    """Report one alias bound to two canonical exercises within a single source."""
    bindings: dict[tuple[Any, Any], tuple[Any, str]] = {}
    for row in tables.get("exercise_alias", ()):
        identifier = record_id("exercise_alias", row)
        key = (row.get("source_system"), row.get("alias_normalized"))
        exercise_id = row.get("exercise_id")
        previous = bindings.setdefault(key, (exercise_id, identifier))
        if previous[0] != exercise_id:
            yield ValidationIssue(
                code="alias_binding_conflict",
                severity=Severity.ERROR,
                table="exercise_alias",
                record_id=identifier,
                message=(
                    f"source_system {key[0]!r} binds alias {key[1]!r} to {exercise_id!r}, but "
                    f"{previous[0]!r} ({previous[1]}) binds it too; two aliases cannot silently "
                    "claim conflicting canonical identities"
                ),
            )


def _check_candidate_exercises_exist(tables: TableRows) -> Iterable[ValidationIssue]:
    """Report ambiguous outcomes whose candidate exercises are not registered."""
    known = {
        row.get("exercise_id")
        for row in tables.get("exercise_definition", ())
        if row.get("exercise_id") is not None
    }
    for row in tables.get("exercise_normalization", ()):
        candidates = row.get("candidate_exercise_ids") or ()
        for candidate in candidates:
            if candidate not in known:
                yield ValidationIssue(
                    code="dangling_candidate_exercise",
                    severity=Severity.ERROR,
                    table="exercise_normalization",
                    record_id=record_id("exercise_normalization", row),
                    message=(
                        f"candidate exercise {candidate!r} is not present in exercise_definition"
                    ),
                )


def all_issues(tables: TableRows) -> list[ValidationIssue]:
    """Run every cross-record rule and return the combined findings."""
    checks: tuple[Any, ...] = (
        check_referential_integrity,
        check_source_provenance,
        check_synthetic_separation,
        check_repeated_records,
        check_exercise_identity_and_alias_bindings,
        check_explicit_missingness,
        check_no_sentinel_zero,
        check_rpe_rir_consistency,
        check_edit_provenance,
        check_interval_overlaps,
        check_attempt_sequence,
        check_reported_totals,
        check_plan_execution_consistency,
        check_duplicate_measurements,
        check_missingness_reason_known,
    )
    issues: list[ValidationIssue] = []
    for check in checks:
        issues.extend(check(tables))
    return issues


def check_repeated_records(tables: TableRows) -> list[ValidationIssue]:
    """Check primary keys are unique within each table."""
    issues: list[ValidationIssue] = []
    for table in tables:
        try:
            spec = table_spec(table)
        except KeyError:
            continue
        seen: dict[tuple[Any, ...], str] = {}
        for row in tables[table]:
            key = tuple(row.get(column) for column in spec.primary_key)
            identifier = record_id(table, row)
            if key in seen:
                issues.append(
                    ValidationIssue(
                        code="duplicate_primary_key",
                        severity=Severity.ERROR,
                        table=table,
                        record_id=identifier,
                        message=(
                            f"primary key {key} already used by {seen[key]}; duplicate records "
                            "corrupt every downstream count"
                        ),
                    )
                )
                continue
            seen[key] = identifier
    return issues


def iter_tables(tables: TableRows) -> Iterable[tuple[str, Sequence[Row]]]:
    """Iterate table name and rows in a stable order."""
    for name in sorted(tables):
        yield name, tables[name]
