"""Fixture package for canonical PSD histories.

These builders are deterministic by construction: every identifier comes from
:func:`psd.schema.identifiers.make_id` and every timestamp is derived from a
fixed base instant, so a fixture produces byte-identical artifacts on every run
and on every platform.

Contents
--------

``normal_history``
    A clean, well-formed training history with a plan, execution, observations,
    tests, and a meet.

``adversarial_history``
    The cases the design documents call out explicitly: missingness, planned
    versus performed, failures, program modification, multiple same-day sessions,
    sparse measurements, weight in pounds, rep-level data for only some sets,
    missing competition attempts, ambiguous identity, and post-hoc edits.

``messy_timeline``
    A multi-year athlete timeline with programming changes, gaps, missing logs,
    deloads, taper weeks, and several meets -- the RES-235 exit gate.

``openpowerlifting``
    A source-shaped OpenPowerlifting CSV covering every edge case RES-237 names: failed
    attempts, a fourth attempt, no-attempt lifters, a negative reported best, a
    component-less total, non-placing participation codes, an approximate age, an
    open-ended weight class, reduced events, a self-described sex category, an unsanctioned
    meet, an unreadable source value, and a source-supplied name disambiguator.
"""

from __future__ import annotations

from tests.fixtures.adversarial_history import adversarial_history_records, synthetic_records
from tests.fixtures.builders import (
    ATHLETE_ID,
    BASE_INSTANT,
    FIXTURE_ALIAS_SOURCE_SYSTEM,
    FIXTURE_EXERCISE_KEYS,
    HISTORY_START,
    HistoryBuilder,
    SourceIds,
    add_athlete,
    add_competition,
    add_exercises,
    add_program,
    add_source_records,
    at,
    counts,
    kg,
    lb,
    program_version_of,
)
from tests.fixtures.messy_timeline import messy_timeline_records
from tests.fixtures.normal_history import normal_history_records
from tests.fixtures.openpowerlifting import (
    SAMPLE_CSV_NAME,
    SAMPLE_ROW_COUNT,
    edge_case_rows,
    sample_csv_text,
    write_sample_csv,
    write_sample_snapshot,
)

__all__ = (
    "ATHLETE_ID",
    "BASE_INSTANT",
    "FIXTURE_ALIAS_SOURCE_SYSTEM",
    "FIXTURE_EXERCISE_KEYS",
    "HISTORY_START",
    "SAMPLE_CSV_NAME",
    "SAMPLE_ROW_COUNT",
    "HistoryBuilder",
    "SourceIds",
    "add_athlete",
    "add_competition",
    "add_exercises",
    "add_program",
    "add_source_records",
    "adversarial_history_records",
    "at",
    "counts",
    "edge_case_rows",
    "kg",
    "lb",
    "messy_timeline_records",
    "normal_history_records",
    "program_version_of",
    "sample_csv_text",
    "synthetic_records",
    "write_sample_csv",
    "write_sample_snapshot",
)
