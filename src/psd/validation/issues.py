"""Structured validation results.

Validation returns issues rather than raising, so a caller can report every
problem in an artifact in one pass. Severity distinguishes a rule violation that
makes an artifact unusable from a signal a human should look at, such as an RPE
and an RIR that disagree.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

__all__ = ("Severity", "ValidationIssue", "ValidationReport")


class Severity(StrEnum):
    """How much a validation issue matters."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    """One finding about a canonical artifact.

    Attributes:
        code: Stable machine-readable code, for example ``dangling_foreign_key``.
        severity: ERROR, WARNING, or INFO.
        table: Table the issue was found in, when applicable.
        record_id: Primary key of the offending row, when applicable.
        message: Human-readable explanation.
    """

    code: str
    severity: Severity
    message: str
    table: str | None = None
    record_id: str | None = None

    def format(self) -> str:
        """Return a single-line human-readable rendering."""
        location = ".".join(part for part in (self.table, self.record_id) if part)
        prefix = f"{location}: " if location else ""
        return f"[{self.severity.value}] {prefix}{self.code}: {self.message}"

    def to_dict(self) -> dict[str, str]:
        """Return a JSON-serializable mapping."""
        payload: dict[str, str] = {
            "code": self.code,
            "severity": self.severity.value,
            "message": self.message,
        }
        if self.table is not None:
            payload["table"] = self.table
        if self.record_id is not None:
            payload["record_id"] = self.record_id
        return payload


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """Outcome of validating a canonical artifact.

    Attributes:
        issues: Every finding, in discovery order.
        tables_checked: Tables that were present and validated.
        row_counts: Rows per table that was validated.
    """

    issues: tuple[ValidationIssue, ...] = ()
    tables_checked: tuple[str, ...] = ()
    row_counts: Mapping[str, int] | None = None

    def by_severity(self, severity: Severity) -> tuple[ValidationIssue, ...]:
        """Return issues with the given severity."""
        return tuple(issue for issue in self.issues if issue.severity is severity)

    @property
    def errors(self) -> tuple[ValidationIssue, ...]:
        """Issues that make the artifact unusable."""
        return self.by_severity(Severity.ERROR)

    @property
    def warnings(self) -> tuple[ValidationIssue, ...]:
        """Issues a human should look at."""
        return self.by_severity(Severity.WARNING)

    @property
    def ok(self) -> bool:
        """Whether validation found no errors."""
        return not self.errors

    def summary(self) -> str:
        """Return a one-line summary suitable for CLI output."""
        return (
            f"{len(self.tables_checked)} tables checked, {len(self.errors)} error(s), "
            f"{len(self.warnings)} warning(s)"
        )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable mapping."""
        return {
            "ok": self.ok,
            "summary": self.summary(),
            "tables_checked": list(self.tables_checked),
            "row_counts": dict(self.row_counts or {}),
            "issues": [issue.to_dict() for issue in self.issues],
        }
