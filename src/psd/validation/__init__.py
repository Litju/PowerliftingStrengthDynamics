"""Canonical artifact validation.

Two layers, in the order the locked stack prescribes:

* PyArrow schemas and explicit custom validators are authoritative for canonical
  persisted tables;
* a declarative column layer over Polars frames adds vectorized whole-column
  checks, which is what corpus-scale validation needs.

Neither layer replaces the other: a column can hold a plausible-looking value
that still breaks a relationship, and a relationship can hold while a column is
outside its vocabulary.
"""

from __future__ import annotations

from psd.validation.declarative import (
    COLUMN_RANGES,
    POSITIVE_COLUMNS,
    VOCABULARY_BY_COLUMN,
    column_issues,
    vocabulary_issues,
)
from psd.validation.issues import Severity, ValidationIssue, ValidationReport
from psd.validation.validate import validate_tables

__all__ = (
    "COLUMN_RANGES",
    "POSITIVE_COLUMNS",
    "VOCABULARY_BY_COLUMN",
    "Severity",
    "ValidationIssue",
    "ValidationReport",
    "column_issues",
    "validate_tables",
    "vocabulary_issues",
)
