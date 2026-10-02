"""PSD-COMP corpus audit.

What this reports and why it is not a summary
---------------------------------------------

An audit exists to make a corpus *falsifiable*: to state, in counts a reader can
recompute, what the corpus contains and what the source did to it. It therefore reports
irregularities rather than smoothing them. A corpus whose ambiguities have been quietly
normalised looks identical to a clean one and cannot be told apart from it, which is
precisely the failure this module exists to prevent.

Two disciplines run through everything here:

* **Nothing is repaired.** A source value PSD could not read is counted and left absent
  in the corpus; here it is counted again. A date that disagrees with its meet is
  reported, not reconciled.
* **Nothing is inferred from continuity.** A count of rows is a fact. A count of
  *athletes* is a count of source identities, and the report says so, because the source
  identity is not always one person.

Where a grouping has hundreds of members -- federations, weight classes, divisions -- the
full list would swamp the report, so the model records how many distinct members exist,
lists the largest, and reports the row count of the remainder. A truncated list whose
truncation is not stated would be the same failure as a truncated corpus.

Why DuckDB
----------

The locked stack reserves DuckDB for analytical interrogation and not for storage, and
that is exactly what this is: aggregate questions over tens of millions of persisted
rows, answerable without materialising a table in Python. The corpus is still the
Parquet on disk; nothing here writes to it.

Expansion audit
---------------

:data:`EXPANSION_RULES` states, per expanded table, which source field creates a row,
what a missing source value does, whether zero carries meaning, and what a negative
value means. Those four questions are where a long-to-wide conversion goes wrong: a
melted attempt table looks identical whether an absent cell produced no row or a zero
row, and a signed attempt looks like a negative load to anything that has forgotten the
sign is the result. :func:`expansion_invariants` turns each statement into a count-level
check against the persisted tables, so the documentation and the corpus can be compared.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, Self

import duckdb
from pydantic import BaseModel, ConfigDict, Field, field_validator

from psd.ingest.openpowerlifting.acquire import (
    AcquisitionError,
    read_pinned_snapshot,
    snapshot_directory,
)
from psd.ingest.openpowerlifting.contract import SourceSchemaReview, review_source_schema
from psd.ingest.openpowerlifting.history import CompetitionEventMembership
from psd.ingest.openpowerlifting.snapshot import OpenPowerliftingSnapshot
from psd.ingest.openpowerlifting.transform import corpus_table_names
from psd.paths import resolve_within_data_root
from psd.provenance.manifest import DatasetManifest, manifest_digest
from psd.schema.registry import table_spec
from psd.serialization.dataset import artifact_paths, read_manifest

__all__ = (
    "AUDIT_DIRNAME",
    "AUDIT_VERSION",
    "EXPANSION_RULES",
    "AnomalyAudit",
    "AthleteContextAudit",
    "AttemptAudit",
    "AttemptCoverage",
    "AuditError",
    "AuditRequest",
    "AuditResult",
    "CategoryCoverage",
    "CorpusAudit",
    "CoverageAudit",
    "EntityAudit",
    "ExpansionAudit",
    "ExpansionRule",
    "Invariant",
    "LongitudinalAudit",
    "PerformanceAudit",
    "SourceAudit",
    "audit_corpus",
    "audit_markdown",
    "coverage_table",
    "expansion_invariants",
    "read_audit",
    "write_audit",
)

#: Version of the audit report contract. Bumped when a section's meaning changes, so a
#: stored report can never be read as describing something it did not describe.
AUDIT_VERSION: Final[str] = "psd-comp-audit/1"

#: Where the audit lives inside a dataset directory. Outside ``tables/``, like every other
#: derived artifact: an audit is not a canonical table and must not enter the canonical
#: digest.
AUDIT_DIRNAME: Final[str] = "audit"

#: How many members of a large grouping the report itemises. The full count is always
#: reported alongside, because a truncated list whose truncation is not stated is the same
#: failure as a truncated corpus.
MAX_LISTED_CATEGORIES: Final[int] = 40

#: How many concrete examples of an anomaly the report names. A count without a witness
#: is a number nobody can check.
MAX_LISTED_EXAMPLES: Final[int] = 20

#: DuckDB's own limits for this pass. The pass is aggregate work over persisted rows, so
#: a bounded memory ceiling with spilling is the honest configuration: it either finishes
#: or reports that it could not, and it never silently swaps the machine.
DEFAULT_MEMORY_LIMIT: Final[str] = "4GB"

#: Stands in for a digest the audit could not determine. Never a real digest: an
#: acknowledged gap, so a reader cannot mistake it for an identity that was verified.
_UNKNOWN_DIGEST: Final[str] = "0" * 64
DEFAULT_THREADS: Final[int] = 4

#: How a null member is labelled while grouping. Distinct from any value a source could
#: publish, so an absent category can never be mistaken for a category called "(null)".
_NULL_LABEL: Final[str] = "\x00null"

#: The source's own weighting convention: ``-93`` is an upper bound, ``90+`` is open-ended
#: above its bound, and ``105`` is an absolute class. Only the ``+`` form is open-ended, so
#: that is the form counted. Declared once and handed to DuckDB as a regular expression
#: rather than transcribed into the query.
OPEN_ENDED_WEIGHT_CLASS_PATTERN: Final[str] = r"^\s*-?\d+(\.\d+)?\+\s*$"

#: The source's documented event vocabulary, taken from the one place that declares which
#: lifts each event contests, so the unknown-event counts and the attempt-versus-event
#: invariant cannot drift from the projection that counted the events.
DECLARED_EVENTS: Final[tuple[str, ...]] = tuple(sorted(CompetitionEventMembership))

#: The declared events that contest each lift, derived from the same table.
EVENTS_BY_LIFT: Final[Mapping[str, tuple[str, ...]]] = {
    lift: tuple(
        sorted(event for event, lifts in CompetitionEventMembership.items() if lift in lifts)
    )
    for lift in ("squat", "bench", "deadlift")
}

#: Span buckets for the observed-span distribution. Zero is its own bucket because a
#: one-meet history is the most common case in the corpus and deserves to be visible.
_SPAN_BUCKETS: Final[tuple[tuple[str, int, int | None], ...]] = (
    ("0 days", 0, 0),
    ("1-30 days", 1, 30),
    ("31-180 days", 31, 180),
    ("181-365 days", 181, 365),
    ("1-2 years", 366, 730),
    ("2-5 years", 731, 1825),
    ("5-10 years", 1826, 3650),
    ("over 10 years", 3651, None),
)

_MEET_COUNT_BUCKETS: Final[tuple[tuple[str, int, int], ...]] = (
    ("1 meet", 1, 1),
    ("2 meets", 2, 2),
    ("3 meets", 3, 3),
    ("4 meets", 4, 4),
    ("5-9 meets", 5, 9),
    ("10-19 meets", 10, 19),
    ("20-49 meets", 20, 49),
    ("50 or more meets", 50, 1_000_000),
)


class AuditError(RuntimeError):
    """Raised when the corpus cannot be audited."""


# --------------------------------------------------------------------------
# report models
# --------------------------------------------------------------------------


class _Model(BaseModel):
    """Base for report sections: frozen, extra-forbidding, and timestamp-checked."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable rendering of the section."""
        return self.model_dump(mode="json")


class CategoryCoverage(_Model):
    """How a categorical column distributes over a large member space.

    Attributes:
        distinct_members: How many distinct values the column holds, including null.
        listed_members: How many are itemised in ``counts``.
        rows: Rows per listed member, largest first.
        remainder_rows: Rows belonging to members that are not itemised.
        null_rows: Rows where the value is absent, which is not a member of the grouping.
    """

    distinct_members: int = Field(ge=0)
    listed_members: int = Field(ge=0)
    rows: Mapping[str, int]
    remainder_rows: int = Field(ge=0)
    null_rows: int = Field(ge=0)

    @property
    def truncated(self) -> bool:
        """Whether members exist that the report does not itemise."""
        return self.listed_members < self.distinct_members


class SourceAudit(_Model):
    """Everything needed to identify and re-acquire the pinned snapshot.

    Attributes:
        source_url: Where the archive was fetched from.
        archive_sha256: Digest of the downloaded ZIP; the snapshot's identity.
        archive_bytes: Size of the ZIP in bytes.
        csv_member_name: Name of the data member inside the archive.
        csv_sha256: Digest of the extracted CSV, which is what the build reads.
        csv_bytes: Size of the extracted CSV in bytes.
        source_rows: Data rows in the CSV, excluding the header.
        snapshot_date: Snapshot date the bulk page reported.
        revision: Data-service revision the bulk page reported.
        archive_declared_date: Snapshot date the archive's own member name declares.
        archive_declared_revision: Revision the archive's own member name declares.
        service_reported_row_count: Row count the bulk page advertised, when it said one.
        header: The CSV header exactly as read.
        schema_review: The contract review of that header.
        license_id: Recorded copyright basis of the data.
        license_url: Official statement of that basis.
        consent_basis: Why the records may be ingested.
        publication_basis: Why the source published them.
    """

    source_url: str
    archive_sha256: str
    archive_bytes: int
    csv_member_name: str
    csv_sha256: str
    csv_bytes: int
    source_rows: int
    snapshot_date: str | None
    revision: str | None
    archive_declared_date: str | None
    archive_declared_revision: str | None
    service_reported_row_count: int | None
    header: tuple[str, ...]
    schema_review: Mapping[str, object]
    license_id: str
    license_url: str
    consent_basis: str
    publication_basis: str


class EntityAudit(_Model):
    """Row and cardinality counts for the canonical entities.

    Attributes:
        athletes: Distinct canonical source identities.
        athlete_source_links: Identity-to-source-key links.
        competitions: Source participation rows, one per source entry.
        meets: Distinct shared meets.
        attempts: Expanded attempt rows.
        reported_results: Expanded reported-result rows.
        competitions_with_attempts: Participations with at least one attempt.
        competitions_with_reported_total: Participations carrying a reported ``TotalKg``.
        competitions_without_results: Participations with no reported result at all.
        duplicate_primary_key_rows: Rows sharing a primary key, per table. Zero everywhere
            is the registry's declared uniqueness holding.
    """

    athletes: int
    athlete_source_links: int
    competitions: int
    meets: int
    attempts: int
    reported_results: int
    competitions_with_attempts: int
    competitions_with_reported_total: int
    competitions_without_results: int
    duplicate_primary_key_rows: Mapping[str, int]


class CoverageAudit(_Model):
    """Row distribution across the corpus's declared categorical axes.

    Attributes:
        by_year: Rows per meet year, from the meet start date.
        by_federation: Rows per hosting federation.
        by_parent_federation: Rows per top-level sanctioning body, kept distinct from
            the hosting federation as the source declares them.
        by_event: Rows per declared competition event.
        by_equipment_class: Rows per normalized competition equipment category.
        by_equipment_class_raw: Rows per source equipment word.
        by_sex_category: Rows per normalized source sex category.
        rows_without_a_date: Rows whose meet date was absent or unreadable.
    """

    by_year: CategoryCoverage
    by_federation: CategoryCoverage
    by_parent_federation: CategoryCoverage
    by_event: CategoryCoverage
    by_equipment_class: CategoryCoverage
    by_equipment_class_raw: CategoryCoverage
    by_sex_category: CategoryCoverage
    rows_without_a_date: int = Field(ge=0)


class LongitudinalAudit(_Model):
    """How the corpus distributes over each source identity's competition history.

    Every count here is a count of *source identities*. The source identity is not always
    one biological person, and nothing here attempts to make it one.

    Attributes:
        athletes_with_one_meet: Identities appearing at exactly one meet.
        athletes_with_two_to_four_meets: Identities at two to four meets.
        athletes_with_five_to_nine_meets: Identities at five to nine meets.
        athletes_with_ten_or_more_meets: Identities at ten or more meets.
        multi_meet_athletes: Identities at more than one meet.
        meet_count_distribution: Identities per meet-count bucket.
        observed_span_days_distribution: Identities per observed-span bucket.
        max_meet_count: Largest meet count any identity has.
        max_observed_span_days: Largest observed span, in days.
        athlete_history_rows: Rows in the derived athlete-history artifact, when built.
    """

    athletes_with_one_meet: int
    athletes_with_two_to_four_meets: int
    athletes_with_five_to_nine_meets: int
    athletes_with_ten_or_more_meets: int
    multi_meet_athletes: int
    meet_count_distribution: Mapping[str, int]
    observed_span_days_distribution: Mapping[str, int]
    max_meet_count: int
    max_observed_span_days: int | None
    athlete_history_rows: int | None


class AttemptCoverage(_Model):
    """Attempt-detail availability within a categorical grouping.

    Attributes:
        distinct_members: How many distinct values the grouping holds.
        listed_members: How many are itemised.
        competitions: Participations per listed member.
        competitions_with_attempts: Participations per listed member that carry an
            attempt. Most federations publish only best lifts, so this is far below
            ``competitions`` for most members and that is the corpus's shape, not a defect.
    """

    distinct_members: int = Field(ge=0)
    listed_members: int = Field(ge=0)
    competitions: Mapping[str, int]
    competitions_with_attempts: Mapping[str, int]

    def fraction(self, member: str) -> float:
        """Return the attempt-detail fraction for one member, zero when it has no rows."""
        total = self.competitions.get(member, 0)
        if total == 0:
            return 0.0
        return self.competitions_with_attempts.get(member, 0) / total


class AttemptAudit(_Model):
    """Attempt availability, failures, and record attempts.

    Attributes:
        competitions_with_attempts: Participations carrying at least one attempt.
        competitions_without_attempts: Participations carrying none.
        attempt_detail_fraction: The first over the second.
        attempts_total: Expanded attempt rows.
        failed_attempts: Attempts the source signed negatively.
        failed_attempt_rate: The first over ``attempts_total``.
        fourth_attempts: Record attempts, which contribute to no total.
        fourth_attempt_rate: The first over ``attempts_total``.
        fourth_attempts_by_lift: Record attempts per lift.
        attempts_per_competition: Participations per total-attempt count.
        lifts_without_any_attempt: Participations where a contested lift has no attempt
            at all -- a source that reported some attempts and not all of them.
        coverage_by_federation: Attempt availability per hosting federation.
        coverage_by_year: Attempt availability per meet year.
    """

    competitions_with_attempts: int
    competitions_without_attempts: int
    attempt_detail_fraction: float
    attempts_total: int
    failed_attempts: int
    failed_attempt_rate: float
    fourth_attempts: int
    fourth_attempt_rate: float
    fourth_attempts_by_lift: Mapping[str, int]
    attempts_per_competition: Mapping[str, int]
    lifts_without_any_attempt: int
    coverage_by_federation: AttemptCoverage
    coverage_by_year: AttemptCoverage


class PerformanceAudit(_Model):
    """Reported bests, totals, and the disagreement between them.

    Attributes:
        negative_reported_bests: Reported bests the source signed negatively, meaning
            the lowest weight attempted and failed.
        negative_reported_bests_by_kind: Those, per result kind.
        totals: Participations carrying a reported ``TotalKg``.
        totals_without_all_component_bests: Totals published with at least one component
            lift absent. PSD manufactures nothing to fill the gap, so these are the
            records where a total exists alone.
        totals_with_all_component_bests: Totals whose three component bests are present.
        total_sum_agreements: Totals equal to the sum of their three bests.
        total_sum_disagreements: Totals that are not. A source may total differently --
            best-of-three versus the best single, a rounding rule, a transcription -- so
            this is reported as a count of disagreements to review, never corrected.
        disagreement_examples: Named examples of a disagreeing total.
        fourth_attempt_competitions_with_a_total: Participations that took a record
            attempt *and* published a total.
        fourth_attempt_totals_equal_to_best_sum: Of those, the totals equal to the sum
            of the three bests. This is what "a fourth attempt does not count toward the
            total" means measured against the corpus.
        reported_results_marked_derived: Reported results PSD marked as derived. Zero:
            PSD derives none.
        bests_without_a_total: Reported bests published with no total.
    """

    negative_reported_bests: int
    negative_reported_bests_by_kind: Mapping[str, int]
    totals: int
    totals_without_all_component_bests: int
    totals_with_all_component_bests: int
    total_sum_agreements: int
    total_sum_disagreements: int
    disagreement_examples: tuple[Mapping[str, object], ...]
    fourth_attempt_competitions_with_a_total: int
    fourth_attempt_totals_equal_to_best_sum: int
    reported_results_marked_derived: int
    bests_without_a_total: int


class AthleteContextAudit(_Model):
    """Body mass, age, weight class, and category-coverage fields.

    Attributes:
        competitions: Participations, the denominator for every rate here.
        bodyweight_present: Participations carrying a body mass.
        bodyweight_missing: Participations with none. Missingness is a property of the
            source's reporting, and is never a zero-kilogram reading.
        bodyweight_missing_fraction: The second over ``competitions``.
        ages_exact: Participations whose reported age is a whole number.
        ages_approximate: Participations whose reported age is fractional, which the
            source publishes when it knows only a birth year.
        ages_missing: Participations with no reported age.
        weight_classes_present: Participations carrying a weight class.
        distinct_weight_classes: Distinct source weight-class spellings.
        open_ended_weight_classes: Participations in an open-ended class such as ``90+``.
        tested_category_yes: Participations the source marks as a drug-tested category.
        tested_category_no: Participations explicitly marked untested.
        tested_category_missing: Participations with no ``Tested`` statement.
        meets_sanctioned: Meets the source counts as sanctioned.
        meets_unsanctioned: Meets the source counts as unsanctioned.
        meets_sanction_unknown: Meets with no sanctioned statement.
    """

    competitions: int
    bodyweight_present: int
    bodyweight_missing: int
    bodyweight_missing_fraction: float
    ages_exact: int
    ages_approximate: int
    ages_missing: int
    weight_classes_present: int
    distinct_weight_classes: int
    open_ended_weight_classes: int
    tested_category_yes: int
    tested_category_no: int
    tested_category_missing: int
    meets_sanctioned: int
    meets_unsanctioned: int
    meets_sanction_unknown: int


class AnomalyAudit(_Model):
    """Source irregularities, reported and never repaired.

    Attributes:
        disambiguated_identities: Identities carrying the source's ``#N`` suffix, which
            the source appends to tell same-named lifters apart.
        sex_category_conflicts: Names the source publishes under more than one reported
            sex category. Their identity stays one flagged identity.
        sex_category_conflict_examples: Named examples.
        sex_categories_absent: Identities with no reported sex at all.
        meet_identity_collisions: Places, towns and dates that name more than one meet
            identity, which is where the six-field meet rule splits what may be one meet.
        meet_identity_collision_examples: Named examples.
        competitions_without_a_meet_date: Participations with an absent or unreadable
            meet date.
        competitions_disagreeing_with_their_meet: Participations whose date differs from
            the meet they reference, which would mean the meet row and the entry disagree.
        earliest_meet_date: Earliest observed meet date, as an aware UTC instant. Rendered
            by DuckDB's own formatter under a pinned UTC session, so the same corpus
            reports the same instant on every machine.
        latest_meet_date: Latest observed meet date, on the same basis.
        unexpected_event_values: Declared-event values outside the documented vocabulary.
        unexpected_equipment_values: Equipment words outside the declared vocabulary.
        unexpected_sex_values: Sex values outside the declared vocabulary.
        unknown_place_codes: ``Place`` values that are neither a positive number nor a
            documented status code.
        unknown_place_examples: The offending values.
        placements_outside_range: Participations whose numeric placing is not positive.
        competitions_with_unknown_event: Participations whose event did not resolve.
        schema_drift_unknown_columns: Columns present in the snapshot but not the contract.
        schema_drift_missing_columns: Contract columns absent from the snapshot.
        schema_drift_reordered: Whether the same columns appear in a different order.
        source_disagreements: Statements about the snapshot that contradict each other.
    """

    disambiguated_identities: int
    sex_category_conflicts: int
    sex_category_conflict_examples: tuple[str, ...]
    sex_categories_absent: int
    meet_identity_collisions: int
    meet_identity_collision_examples: tuple[Mapping[str, object], ...]
    competitions_without_a_meet_date: int
    competitions_disagreeing_with_their_meet: int
    earliest_meet_date: str | None
    latest_meet_date: str | None
    unexpected_event_values: tuple[str, ...]
    unexpected_equipment_values: tuple[str, ...]
    unexpected_sex_values: tuple[str, ...]
    unknown_place_codes: int
    unknown_place_examples: tuple[str, ...]
    placements_outside_range: int
    competitions_with_unknown_event: int
    schema_drift_unknown_columns: tuple[str, ...]
    schema_drift_missing_columns: tuple[str, ...]
    schema_drift_reordered: bool
    source_disagreements: tuple[str, ...]


class ExpansionRule(_Model):
    """What creates a row in an expanded table, stated as four questions.

    Attributes:
        table: Canonical table the rule describes.
        source_field: The source field whose presence creates a row.
        absent_source_value: What happens when that field is absent or unreadable.
        zero_meaning: Whether zero carries meaning in this table.
        negative_semantics: What a negative value means, or that it cannot occur.
    """

    table: str
    source_field: str
    absent_source_value: str
    zero_meaning: str
    negative_semantics: str


@dataclass(frozen=True, slots=True)
class Invariant:
    """One count-level check of an expansion claim.

    Attributes:
        name: Stable identifier.
        statement: What must hold, in words.
        expected: The value the claim requires -- almost always zero.
        observed: The value the corpus actually holds.
    """

    name: str
    statement: str
    expected: int
    observed: int

    @property
    def holds(self) -> bool:
        """Whether the corpus satisfies the claim."""
        return self.expected == self.observed

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable rendering."""
        return {
            "name": self.name,
            "statement": self.statement,
            "expected": self.expected,
            "observed": self.observed,
            "holds": self.holds,
        }


class ExpansionAudit(_Model):
    """The declared expansion rules beside the checks that hold them.

    Attributes:
        rules: One rule per expanded table.
        invariants: The count-level checks derived from those rules.
    """

    rules: tuple[ExpansionRule, ...]
    invariants: tuple[Mapping[str, object], ...]

    @property
    def failed_invariants(self) -> tuple[Mapping[str, object], ...]:
        """Invariants the corpus does not satisfy."""
        return tuple(item for item in self.invariants if not bool(item["holds"]))


#: The expansion contract, stated once and checked by
#: :func:`expansion_invariants`. Every row here is a decision the transform makes on
#: purpose; a long-to-wide conversion is exactly where these get lost.
EXPANSION_RULES: Final[tuple[ExpansionRule, ...]] = (
    ExpansionRule(
        table="competition",
        source_field="one OpenPowerlifting source row",
        absent_source_value=(
            "Never: a source row always becomes exactly one competition. An absent or "
            "unreadable meet date leaves competition_date null rather than inventing a "
            "date, and an absent or non-positive body mass leaves bodyweight_kg null "
            "rather than zero."
        ),
        zero_meaning=(
            "Zero cannot appear in any loaded field: the transform nulls a non-positive "
            "body mass and refuses a zero attempt. Zero is never a source value PSD "
            "carries forward."
        ),
        negative_semantics=(
            "None. A negative source value never reaches this table: attempt signs are "
            "resolved into an attempt result, and a reported best keeps its sign on "
            "competition_reported_result instead."
        ),
    ),
    ExpansionRule(
        table="competition_attempt",
        source_field=(
            "one of the twelve attempt columns (Squat1-4Kg, Bench1-4Kg, Deadlift1-4Kg) "
            "holding a present, readable, non-zero number"
        ),
        absent_source_value=(
            "No row at all. The table is long rather than wide precisely so that an absent "
            "cell is an absent observation: a source that publishes only best lifts "
            "produces no attempt rows, and PSD never reconstructs attempts from a best."
        ),
        zero_meaning=(
            "None. A cell reported as 0 is not a load and not a missing value either, so "
            "it is counted and dropped rather than stored as a zero-kilogram attempt."
        ),
        negative_semantics=(
            "The sign is the result: a negative value is a failed attempt at the positive "
            "magnitude, stored as load_kg > 0 with result=bad_lift and source_attempt_raw "
            "keeping the published sign."
        ),
    ),
    ExpansionRule(
        table="competition_reported_result",
        source_field=(
            "one of the eight reported-result columns (Best3SquatKg, Best3BenchKg, "
            "Best3DeadliftKg, TotalKg, Dots, Wilks, Glossbrenner, Goodlift) holding a "
            "present, readable, non-zero number"
        ),
        absent_source_value=(
            "No row at all. A missing best produces no reported-best observation, and a "
            "missing TotalKg produces no total. TotalKg can therefore exist with none of "
            "its component bests present, and PSD never manufactures the components to "
            "balance it."
        ),
        zero_meaning=(
            "None. A cell reported as 0 is counted and dropped. Every stored value is the "
            "absolute magnitude, so a scoring-system score is never negative."
        ),
        negative_semantics=(
            "A negative reported best is the source publishing the lowest weight the "
            "lifter attempted and failed. source_value_raw keeps the sign and "
            "reported_best_semantics is failed_attempt_only, so it can never be read as a "
            "successful negative lift."
        ),
    ),
    ExpansionRule(
        table="competition_meet",
        source_field=(
            "the six meet-identity fields (Date, Federation, MeetCountry, MeetState, "
            "MeetTown, MeetName), normalized to lowercase with whitespace collapsed"
        ),
        absent_source_value=(
            "An absent identity field normalizes to the empty string and takes part in the "
            "identity, so a missing field is part of what distinguishes one meet from "
            "another rather than being filled in."
        ),
        zero_meaning="None. No numeric field on a meet carries source semantics of zero.",
        negative_semantics="None. A meet row cannot carry a negative value.",
    ),
    ExpansionRule(
        table="athlete",
        source_field="one distinct verbatim Name value, including its #N disambiguator",
        absent_source_value=(
            "Not applicable: the source declares Name mandatory. A name is never "
            "normalised, folded, or stripped, so two lifters the source says are distinct "
            "never merge."
        ),
        zero_meaning="None. No athlete field can be zero.",
        negative_semantics="None. An athlete row cannot carry a negative value.",
    ),
)


# --------------------------------------------------------------------------
# request and result
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AuditRequest:
    """One request to audit a built corpus.

    Attributes:
        dataset_dir: The built PSD-COMP dataset, relative to the data root.
        data_root: External PSD data root; resolved from ``PSD_DATA_ROOT`` otherwise.
        archive_sha256: The pinned snapshot to read the source facts from. Defaults to
            the digest the dataset directory is named for, which is the same digest the
            build used.
        generated_at: Timezone-aware UTC instant recorded on the report. It is provenance
            rather than content and never enters a digest.
        output_dir: Where to write the report; defaults to ``<dataset_dir>/audit``.
        memory_limit: DuckDB memory ceiling for this pass.
        threads: DuckDB worker threads.
    """

    dataset_dir: Path
    data_root: Path | None = None
    archive_sha256: str | None = None
    generated_at: datetime | None = None
    output_dir: Path | None = None
    memory_limit: str = DEFAULT_MEMORY_LIMIT
    threads: int = DEFAULT_THREADS


@dataclass(frozen=True, slots=True)
class AuditResult:
    """Outcome of one corpus audit.

    Attributes:
        audit: The report.
        json_path: Where the machine-readable report was written.
        markdown_path: Where the human-readable summary was written.
        audit_seconds: Wall-clock seconds the audit took.
    """

    audit: CorpusAudit
    json_path: Path | None
    markdown_path: Path | None
    audit_seconds: float


class CorpusAudit(_Model):
    """The durable, machine-readable audit of one PSD-COMP corpus.

    Attributes:
        audit_version: Version of the audit contract.
        dataset_id: Canonical dataset identifier.
        dataset_relative_path: Where the corpus lives, relative to the data root.
        schema_version: Canonical schema version of the persisted tables.
        manifest_digest: SHA-256 of the dataset manifest the audit read.
        source: Snapshot identity, licensing, and the source schema review.
        entities: Row and cardinality counts.
        coverage: Distribution across the corpus's categorical axes.
        longitudinal: Structure of each source identity's competition history.
        attempts: Attempt availability, failures, and record attempts.
        performance: Reported bests, totals, and their disagreements.
        athlete_context: Body mass, age, weight class, and category coverage.
        anomalies: Source irregularities, reported and not repaired.
        expansion: The declared expansion contract and its count-level checks.
        generated_at: Timezone-aware UTC instant the report was produced.
    """

    audit_version: str
    dataset_id: str
    dataset_relative_path: str
    schema_version: str
    manifest_digest: str
    source: SourceAudit
    entities: EntityAudit
    coverage: CoverageAudit
    longitudinal: LongitudinalAudit
    attempts: AttemptAudit
    performance: PerformanceAudit
    athlete_context: AthleteContextAudit
    anomalies: AnomalyAudit
    expansion: ExpansionAudit
    generated_at: datetime

    @field_validator("generated_at")
    @classmethod
    def _require_aware_generated_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
            msg = (
                f"generated_at is a naive timestamp ({value.isoformat()!r}); an audit must "
                "state when it ran in a form two machines can compare."
            )
            raise ValueError(msg)
        return value

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> Self:
        """Re-read a persisted report."""
        return cls.model_validate(payload)

    def to_json_bytes(self) -> bytes:
        """Serialize deterministically: sorted keys, fixed indentation, UTF-8."""
        text = json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            separators=(",", ": "),
        )
        return (text + "\n").encode("utf-8")

    @property
    def failed_invariants(self) -> tuple[Mapping[str, object], ...]:
        """Expansion invariants the corpus does not satisfy."""
        return self.expansion.failed_invariants

    @property
    def schema_review(self) -> SourceSchemaReview:
        """Re-derive the contract review recorded in the report."""
        return review_source_schema(self.source.header)

    def summary(self) -> str:
        """Return a concise human-readable summary.

        Deliberately a *summary*: the durable record is the JSON report, and this text is
        the part a reader skims. It leads with what the corpus is, then what is unusual
        about it, because the second is what a reader is looking for.
        """
        source = self.source
        lines: list[str] = [
            f"PSD-COMP corpus audit ({self.audit_version})",
            "",
            f"corpus           {self.dataset_id}",
            f"schema           {self.schema_version}",
            f"manifest sha256  {self.manifest_digest}",
            "",
            "source",
            f"  snapshot       {source.snapshot_date or 'not reported'}"
            f"  revision {source.revision or 'not reported'}",
            f"  archive        {source.archive_bytes:,} bytes  sha256 {source.archive_sha256}",
            f"  csv            {source.csv_bytes:,} bytes  sha256 {source.csv_sha256}",
            f"  source rows    {source.source_rows:,}",
            f"  licence        {source.license_id} ({source.consent_basis})",
            f"  schema         {_schema_line(self)}",
            "",
            "canonical entities",
            f"  athletes       {self.entities.athletes:,}",
            f"  competitions   {self.entities.competitions:,}",
            f"  meets          {self.entities.meets:,}",
            f"  attempts       {self.entities.attempts:,}",
            f"  reported       {self.entities.reported_results:,}",
            "",
            "longitudinal structure",
            f"  1 meet         {self.longitudinal.athletes_with_one_meet:,}",
            f"  2-4 meets      {self.longitudinal.athletes_with_two_to_four_meets:,}",
            f"  5-9 meets      {self.longitudinal.athletes_with_five_to_nine_meets:,}",
            f"  10+ meets      {self.longitudinal.athletes_with_ten_or_more_meets:,}",
            f"  max meets      {self.longitudinal.max_meet_count:,}",
            "",
            "attempts",
            f"  detail         {self.attempts.competitions_with_attempts:,} of"
            f" {self.entities.competitions:,} competitions"
            f" ({_percent(self.attempts.attempt_detail_fraction)})",
            f"  failed         {self.attempts.failed_attempts:,}"
            f" ({_percent(self.attempts.failed_attempt_rate)})",
            f"  fourth         {self.attempts.fourth_attempts:,}"
            f" ({_percent(self.attempts.fourth_attempt_rate)})",
            "",
            "reported performance fields",
            f"  negative bests {self.performance.negative_reported_bests:,}",
            f"  totals         {self.performance.totals:,}, of which"
            f" {self.performance.totals_without_all_component_bests:,}"
            " lack a full set of component bests",
            f"  disagreements  {self.performance.total_sum_disagreements:,}"
            " totals differ from the sum of their bests",
            "",
            "source anomalies (reported, not repaired)",
            f"  sex conflicts  {self.anomalies.sex_category_conflicts:,} names under more"
            " than one reported sex category",
            f"  #N names       {self.anomalies.disambiguated_identities:,}",
            f"  meet splits    {self.anomalies.meet_identity_collisions:,} places sharing a"
            " name across more than one meet identity",
            f"  unknown codes  {self.anomalies.unknown_place_codes:,} Place values outside"
            " the documented vocabulary",
            f"  no meet date   {self.anomalies.competitions_without_a_meet_date:,}",
            "",
            "expansion contract",
            f"  rules          {len(self.expansion.rules)}",
            f"  invariants     {len(self.expansion.invariants)},"
            f" {len(self.failed_invariants)} failing",
        ]
        return "\n".join(lines)


def _schema_line(audit: CorpusAudit) -> str:
    """Return a one-line rendering of the source schema review."""
    rendered = audit.source.schema_review.get("summary")
    return rendered if isinstance(rendered, str) else ""


def _percent(fraction: float) -> str:
    """Return *fraction* as a one-decimal percentage string."""
    return f"{fraction * 100:.1f}%"


def audit_markdown(audit: CorpusAudit) -> str:
    """Return the human-readable Markdown rendering of an audit report.

    The JSON report is the durable record; this is the document a reviewer reads. It
    prints the tables a reviewer would otherwise rebuild by hand, and it says plainly
    which groupings were truncated so a skipped tail is never mistaken for an empty one.
    """
    lines: list[str] = [
        f"# PSD-COMP corpus audit ({audit.audit_version})",
        "",
        f"- corpus: `{audit.dataset_id}`",
        f"- schema: `{audit.schema_version}`",
        f"- manifest SHA-256: `{audit.manifest_digest}`",
        f"- generated: {audit.generated_at.isoformat(timespec='seconds')}",
        "",
        audit.summary(),
        "",
        "## Coverage",
        "",
        coverage_table("By meet year", audit.coverage.by_year),
        coverage_table("By hosting federation", audit.coverage.by_federation),
        coverage_table("By parent federation", audit.coverage.by_parent_federation),
        coverage_table("By declared event", audit.coverage.by_event),
        coverage_table("By equipment category", audit.coverage.by_equipment_class),
        coverage_table("By source equipment word", audit.coverage.by_equipment_class_raw),
        coverage_table("By sex category", audit.coverage.by_sex_category),
        "## Longitudinal structure",
        "",
        _bucket_table("Meet count", audit.longitudinal.meet_count_distribution),
        _bucket_table("Observed span", audit.longitudinal.observed_span_days_distribution),
        "",
        "## Attempts",
        "",
        f"- participations with at least one attempt: "
        f"{audit.attempts.competitions_with_attempts:,}",
        f"- participations with a contested lift but no attempt for it: "
        f"{audit.attempts.lifts_without_any_attempt:,}",
        "",
        _bucket_table("Attempts per participation", audit.attempts.attempts_per_competition),
        _attempt_coverage_table(
            "Attempt detail by federation", audit.attempts.coverage_by_federation
        ),
        _attempt_coverage_table("Attempt detail by year", audit.attempts.coverage_by_year),
        "## Performance fields",
        "",
        "- negative reported bests by kind: "
        f"`{dict(audit.performance.negative_reported_bests_by_kind)}`",
        f"- totals equal to the sum of their bests: {audit.performance.total_sum_agreements:,}",
        f"- totals that disagree: {audit.performance.total_sum_disagreements:,}",
        f"- record-attempt participations also publishing a total: "
        f"{audit.performance.fourth_attempt_competitions_with_a_total:,}, of which"
        f" {audit.performance.fourth_attempt_totals_equal_to_best_sum:,} equal the sum of"
        " their three bests",
        f"- reported results marked derived: {audit.performance.reported_results_marked_derived:,}",
        "",
        "## Source anomalies",
        "",
        f"- earliest meet: `{audit.anomalies.earliest_meet_date or 'none'}`",
        f"- latest meet: `{audit.anomalies.latest_meet_date or 'none'}`",
        f"- unexpected event values: `{list(audit.anomalies.unexpected_event_values)}`",
        f"- unexpected equipment values: `{list(audit.anomalies.unexpected_equipment_values)}`",
        f"- unexpected sex values: `{list(audit.anomalies.unexpected_sex_values)}`",
        f"- unknown place codes: `{list(audit.anomalies.unknown_place_examples)}`",
        "- participations whose date differs from their meet: "
        f"{audit.anomalies.competitions_disagreeing_with_their_meet:,}",
        "- duplicate primary keys: "
        f"`{ {k: v for k, v in audit.entities.duplicate_primary_key_rows.items() if v} }`",
        "",
        "## Expansion contract",
        "",
        "| table | source field | absent source value | zero | negative |",
        "| --- | --- | --- | --- | --- |",
    ]
    lines.extend(_rule_rows(audit))
    lines.extend(
        [
            "",
            "| invariant | statement | expected | observed | holds |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for item in audit.expansion.invariants:
        holds = "yes" if item["holds"] else "**NO**"
        lines.append(
            f"| `{item['name']}` | {item['statement']} | {item['expected']}"
            f" | {item['observed']} | {holds} |"
        )
    lines.append("")
    return "\n".join(lines)


def _rule_rows(audit: CorpusAudit) -> list[str]:
    """Return one Markdown row per declared expansion rule."""
    return [
        f"| `{rule.table}` | {rule.source_field} | {rule.absent_source_value}"
        f" | {rule.zero_meaning} | {rule.negative_semantics} |"
        for rule in audit.expansion.rules
    ]


def _attempt_coverage_table(title: str, coverage: AttemptCoverage) -> str:
    """Return a Markdown table for attempt availability within a grouping.

    Truncation is stated for the same reason it is stated for a plain coverage table: an
    unstated omission reads as an absence of federations rather than an omission.
    """
    note = (
        f" ({coverage.listed_members} of {coverage.distinct_members} members listed)"
        if coverage.listed_members < coverage.distinct_members
        else ""
    )
    rows = [
        f"| {title}{note} | participations | with attempts | attempt detail |",
        "| --- | --- | --- | --- |",
    ]
    rows.extend(
        f"| `{member}` | {total:,} | {coverage.competitions_with_attempts.get(member, 0):,}"
        f" | {_percent(coverage.fraction(member))} |"
        for member, total in coverage.competitions.items()
    )
    return "\n".join(rows)


def coverage_table(title: str, coverage: CategoryCoverage) -> str:
    """Return a Markdown table for one categorical grouping."""
    note = (
        f" ({coverage.listed_members} of {coverage.distinct_members} members listed;"
        f" {coverage.remainder_rows:,} rows in the remainder)"
        if coverage.truncated
        else ""
    )
    rows = [f"| {title}{note} | rows |", "| --- | --- |"]
    rows.extend(f"| `{member}` | {count:,} |" for member, count in coverage.rows.items())
    if coverage.null_rows:
        rows.append(f"| *(not recorded)* | {coverage.null_rows:,} |")
    return "\n".join(rows)


def _bucket_table(title: str, buckets: Mapping[str, int]) -> str:
    """Return a Markdown table for one bucketed distribution."""
    rows = [f"| {title} | identities |", "| --- | --- |"]
    rows.extend(f"| {label} | {count:,} |" for label, count in buckets.items())
    return "\n".join(rows)


# --------------------------------------------------------------------------
# DuckDB helpers
# --------------------------------------------------------------------------

_CONNECTION_FACTORY: Final = duckdb.connect

_LISTED_LIMIT: Final[str] = str(MAX_LISTED_CATEGORIES)
_EXAMPLES_LIMIT: Final[str] = str(MAX_LISTED_EXAMPLES)


def _open(request: AuditRequest) -> duckdb.DuckDBPyConnection:
    """Return a DuckDB connection configured for one audit pass.

    The memory ceiling and spill directory are set explicitly rather than inherited, so
    the pass either finishes within a stated budget or reports that it could not. A
    reader who knows the ceiling can reason about whether the numbers are complete.

    The session timezone is pinned to UTC for the same reason the canonical encoding is:
    DuckDB renders and compares ``TIMESTAMPTZ`` values in the session zone, so an
    operator's local zone would move the reported meet date across a day boundary and
    would make ``date_diff('day', ...)`` count calendar days differently either side of a
    daylight-saving change. The report must be the same numbers on every machine.
    """
    spill = resolve_within_data_root(
        Path("cache") / "duckdb", data_root=request.data_root, create=True
    )
    connection = _CONNECTION_FACTORY(database=":memory:")
    connection.execute(f"SET memory_limit='{request.memory_limit}'")
    connection.execute(f"SET threads={max(1, request.threads)}")
    connection.execute("SET preserve_insertion_order=false")
    connection.execute("SET TimeZone='UTC'")
    connection.execute("SET temp_directory=?", [str(spill)])
    return connection


def _register(connection: duckdb.DuckDBPyConnection, table: str, path: Path) -> None:
    """Create a view over one canonical artifact.

    A view, not a copy: the corpus is the Parquet on disk, and this pass reads it. Column
    projection and predicate pushdown mean each query pays only for the columns it names.

    The path is bound through DuckDB's own Parquet reader rather than interpolated into
    SQL text. A data root on another volume, or a Windows path with a backslash in it,
    would otherwise have to be escaped by hand, and a mis-escaped path is a query that
    silently reads nothing.
    """
    if not path.is_file():
        msg = f"Cannot audit {table}: no artifact at {path}."
        raise AuditError(msg)
    connection.read_parquet(str(path)).create_view(table)


def _scalar(connection: duckdb.DuckDBPyConnection, sql: str) -> int:
    """Return the single integer *sql* produces."""
    row = connection.execute(sql).fetchone()
    if row is None or row[0] is None:
        return 0
    return int(row[0])


def _optional_int(connection: duckdb.DuckDBPyConnection, sql: str) -> int | None:
    """Return the single integer *sql* produces, or ``None`` when it is null."""
    row = connection.execute(sql).fetchone()
    if row is None or row[0] is None:
        return None
    return int(row[0])


def _optional_text(connection: duckdb.DuckDBPyConnection, sql: str) -> str | None:
    """Return the single text value *sql* produces, or ``None`` when it is null."""
    row = connection.execute(sql).fetchone()
    if row is None or row[0] is None:
        return None
    return str(row[0])


def _sql_list(values: Sequence[str]) -> str:
    """Return *values* as a SQL literal list.

    The values are declared constants, never source data, so this is not a place where a
    value could inject SQL. It exists so a vocabulary has one declaration in Python rather
    than one in Python and a transcription in a query.
    """
    return ", ".join(f"'{value}'" for value in values)


def _contested_lifts_sql(alias: str) -> str:
    """Return a SQL row-valued expression listing the lifts *alias*'s event contests.

    Comparing a row-valued expression against a list is what makes "this attempt is not at
    a lift the declared event contests" one comparison rather than three nullable
    ``CASE`` branches per lift, and it is why the same expression can serve both the
    coverage count and the expansion invariant.
    """
    cases = ", ".join(
        f"CASE WHEN {alias}.competition_event IN ({_sql_list(EVENTS_BY_LIFT[lift])})"
        f" THEN '{lift}' END"
        for lift in ("squat", "bench", "deadlift")
    )
    return f"[{cases}]"


def _counts(connection: duckdb.DuckDBPyConnection, sql: str) -> dict[str, int]:
    """Return a two-column result as ``label -> count`` with nulls rendered."""
    result: dict[str, int] = {}
    for label, count in connection.execute(sql).fetchall():
        key = _NULL_LABEL if label is None else str(label)
        result[key] = result.get(key, 0) + int(count)
    return result


def _examples(
    connection: duckdb.DuckDBPyConnection, sql: str, *, limit: str = _EXAMPLES_LIMIT
) -> tuple[Mapping[str, object], ...]:
    """Return the first *limit* rows of *sql* as mappings keyed by column name."""
    cursor = connection.execute(sql)
    names = [description[0] for description in cursor.description]
    return tuple(
        {name: (None if value is None else value) for name, value in zip(names, row, strict=True)}
        for row in cursor.fetchmany(int(limit))
    )


def _category_coverage(
    connection: duckdb.DuckDBPyConnection,
    *,
    table: str,
    column: str,
    listed: str = _LISTED_LIMIT,
) -> CategoryCoverage:
    """Return how *table* distributes over *column*.

    The full member count, the largest members, the rows in the remainder, and the null
    rows are all reported. A truncated list that does not say it was truncated would read
    as a complete one, which is the failure this whole module exists to prevent.
    """
    expression = (
        f'"{column}"' if column in _DERIVED_COLUMNS.get(table, ()) else f'"{table}"."{column}"'
    )
    distinct = _scalar(
        connection,
        f"SELECT count(DISTINCT {expression}) FROM {table}",
    )
    null_rows = _scalar(connection, f'SELECT count(*) FROM "{table}" WHERE {expression} IS NULL')
    counts = _counts(
        connection,
        f"SELECT {expression} AS member, count(*) AS rows FROM {table} GROUP BY member"
        " ORDER BY rows DESC, member ASC",
    )
    # Absence is reported once, as ``null_rows``. Leaving it in the member list as well
    # would print the same rows twice and make a coverage table that does not add up.
    counts.pop(_NULL_LABEL, None)
    listed_counts = dict(list(counts.items())[: int(listed)])
    remainder = sum(count for member, count in counts.items() if member not in listed_counts)
    return CategoryCoverage(
        distinct_members=distinct,
        listed_members=len(listed_counts),
        rows=listed_counts,
        remainder_rows=remainder,
        null_rows=null_rows,
    )


#: Columns that exist only inside a query rather than in the persisted table. Kept as data
#: so :func:`_category_coverage` never builds an identifier it has not declared.
_DERIVED_COLUMNS: Final[Mapping[str, tuple[str, ...]]] = {
    "comp_year": ("year",),
}


# --------------------------------------------------------------------------
# sections
# --------------------------------------------------------------------------


def _source_audit(
    manifest: DatasetManifest, snapshot: OpenPowerliftingSnapshot | None
) -> SourceAudit:
    """Return the source section, from the pinned snapshot and the manifest.

    Both inputs can be absent. The pinned snapshot is provenance rather than a dependency,
    so a corpus whose snapshot has been removed still audits; the fields the snapshot would
    have supplied are then recorded as unknown rather than filled from the filename.
    """
    review = review_source_schema(snapshot.source_columns if snapshot else ())
    service = snapshot.service if snapshot else None
    record = manifest.sources[0] if manifest.sources else None
    return SourceAudit(
        source_url=snapshot.source_url
        if snapshot
        else ((record.notes if record else None) or "unknown"),
        archive_sha256=snapshot.archive_sha256 if snapshot else _UNKNOWN_DIGEST,
        archive_bytes=snapshot.archive_byte_size if snapshot else 0,
        csv_member_name=snapshot.csv_member_name if snapshot else "",
        csv_sha256=snapshot.csv_sha256 if snapshot else _UNKNOWN_DIGEST,
        csv_bytes=snapshot.csv_byte_size if snapshot else 0,
        source_rows=snapshot.row_count if snapshot else 0,
        snapshot_date=service.updated_date if service else None,
        revision=service.revision if service else None,
        archive_declared_date=service.archive_declared_date if service else None,
        archive_declared_revision=service.archive_declared_revision if service else None,
        service_reported_row_count=service.advertised_row_count if service else None,
        header=tuple(snapshot.source_columns) if snapshot else (),
        schema_review=review.to_dict(),
        license_id=(record.license_id or "") if record else "",
        license_url=(record.license_url or "") if record else "",
        consent_basis=(record.consent_basis or "") if record else "",
        publication_basis=(record.publication_basis or "") if record else "",
    )


def _entity_audit(connection: duckdb.DuckDBPyConnection) -> EntityAudit:
    """Return the canonical-entity counts."""
    competitions = _scalar(connection, "SELECT count(*) FROM competition")
    with_attempts = _scalar(
        connection,
        "SELECT count(DISTINCT competition_id) FROM competition_attempt",
    )
    with_total = _scalar(
        connection,
        "SELECT count(DISTINCT competition_id) FROM competition_reported_result"
        " WHERE result_kind = 'total'",
    )
    with_results = _scalar(
        connection,
        "SELECT count(DISTINCT competition_id) FROM competition_reported_result",
    )
    duplicates: dict[str, int] = {}
    for table in corpus_table_names():
        key = ", ".join(f'"{column}"' for column in table_spec(table).primary_key)
        duplicates[table] = _scalar(
            connection,
            f"SELECT coalesce(sum(n - 1), 0) FROM (SELECT count(*) AS n FROM {table}"
            f" GROUP BY {key} HAVING count(*) > 1)",
        )
    return EntityAudit(
        athletes=_scalar(connection, "SELECT count(*) FROM athlete"),
        athlete_source_links=_scalar(connection, "SELECT count(*) FROM athlete_source_link"),
        competitions=competitions,
        meets=_scalar(connection, "SELECT count(*) FROM competition_meet"),
        attempts=_scalar(connection, "SELECT count(*) FROM competition_attempt"),
        reported_results=_scalar(connection, "SELECT count(*) FROM competition_reported_result"),
        competitions_with_attempts=with_attempts,
        competitions_with_reported_total=with_total,
        competitions_without_results=competitions - with_results,
        duplicate_primary_key_rows=duplicates,
    )


def _coverage_audit(connection: duckdb.DuckDBPyConnection) -> CoverageAudit:
    """Return the corpus's distribution across its categorical axes."""
    connection.execute(
        "CREATE OR REPLACE VIEW comp_year AS SELECT *,"
        " CAST(EXTRACT(year FROM competition_date) AS INTEGER) AS year FROM competition"
    )
    without_date = _scalar(
        connection, "SELECT count(*) FROM competition WHERE competition_date IS NULL"
    )
    return CoverageAudit(
        by_year=_category_coverage(connection, table="comp_year", column="year"),
        by_federation=_category_coverage(connection, table="competition", column="federation"),
        by_parent_federation=_category_coverage(
            connection, table="competition", column="sanctioning_body"
        ),
        by_event=_category_coverage(connection, table="competition", column="competition_event"),
        by_equipment_class=_category_coverage(
            connection, table="competition", column="equipment_class"
        ),
        by_equipment_class_raw=_category_coverage(
            connection, table="competition", column="equipment_class_raw"
        ),
        by_sex_category=_category_coverage(connection, table="athlete", column="sex_category"),
        rows_without_a_date=without_date,
    )


def _longitudinal_audit(
    connection: duckdb.DuckDBPyConnection, *, history_rows: int | None
) -> LongitudinalAudit:
    """Return how each source identity's competition history distributes."""
    connection.execute(
        "CREATE OR REPLACE TEMP TABLE meets_per_identity AS"
        " SELECT athlete_id, count(DISTINCT competition_meet_id) AS meet_count,"
        " min(competition_date) AS first_date, max(competition_date) AS last_date,"
        " date_diff('day', min(competition_date), max(competition_date)) AS span_days"
        " FROM competition GROUP BY athlete_id"
    )
    total = _scalar(connection, "SELECT count(*) FROM meets_per_identity")
    meet_buckets = _bucket_counts(
        connection,
        table="meets_per_identity",
        column="meet_count",
        buckets=_MEET_COUNT_BUCKETS,
    )
    span_buckets = _bucket_counts(
        connection,
        table="meets_per_identity",
        column="span_days",
        buckets=_SPAN_BUCKETS,
        default="no dated meets",
    )
    return LongitudinalAudit(
        athletes_with_one_meet=meet_buckets.get("1 meet", 0),
        athletes_with_two_to_four_meets=sum(
            meet_buckets.get(label, 0) for label in ("2 meets", "3 meets", "4 meets")
        ),
        athletes_with_five_to_nine_meets=meet_buckets.get("5-9 meets", 0),
        athletes_with_ten_or_more_meets=sum(
            meet_buckets.get(label, 0)
            for label in ("10-19 meets", "20-49 meets", "50 or more meets")
        ),
        multi_meet_athletes=total - meet_buckets.get("1 meet", 0),
        meet_count_distribution=meet_buckets,
        observed_span_days_distribution=span_buckets,
        max_meet_count=_scalar(
            connection, "SELECT coalesce(max(meet_count), 0) FROM meets_per_identity"
        ),
        max_observed_span_days=_optional_int(
            connection, "SELECT max(span_days) FROM meets_per_identity"
        ),
        athlete_history_rows=history_rows,
    )


def _bucket_counts(
    connection: duckdb.DuckDBPyConnection,
    *,
    table: str,
    column: str,
    buckets: Sequence[tuple[str, int, int | None]],
    default: str | None = None,
) -> dict[str, int]:
    """Return how many rows of *table* fall in each ``(label, low, high)`` bucket.

    The declared ranges are disjoint and cover every non-null value, so the counts sum to
    the table's row count. That is a property of the declared buckets rather than of the
    data, which is why the buckets are data: a reader can check the sum and see that
    nothing fell through.
    """
    quoted = f'"{column}"'
    result = {
        label: _scalar(
            connection,
            f"SELECT count(*) FROM {table} WHERE {quoted} >= {low}"
            if high is None
            else f"SELECT count(*) FROM {table} WHERE {quoted} BETWEEN {low} AND {high}",
        )
        for label, low, high in buckets
    }
    if default is not None:
        result[default] = _scalar(
            connection, f"SELECT count(*) FROM {table} WHERE {quoted} IS NULL"
        )
    return result


def _attempt_coverage(
    connection: duckdb.DuckDBPyConnection, *, group: str, listed: str = _LISTED_LIMIT
) -> AttemptCoverage:
    """Return attempt availability per member of *group*."""
    competitions = _counts(
        connection,
        f"SELECT {group} AS member, count(*) AS rows FROM competition GROUP BY member"
        " ORDER BY rows DESC, member ASC",
    )
    with_attempts = _counts(
        connection,
        "SELECT c.member AS member, count(*) AS rows FROM ("
        f" SELECT {group} AS member, competition_id FROM competition) AS c"
        " SEMI JOIN (SELECT DISTINCT competition_id FROM competition_attempt) AS a"
        " ON c.competition_id = a.competition_id GROUP BY member"
        " ORDER BY rows DESC, member ASC",
    )
    listed_members = dict(list(competitions.items())[: int(listed)])
    return AttemptCoverage(
        distinct_members=len(competitions),
        listed_members=len(listed_members),
        competitions=listed_members,
        competitions_with_attempts={
            member: with_attempts.get(member, 0) for member in listed_members
        },
    )


def _attempt_audit(connection: duckdb.DuckDBPyConnection, competitions: int) -> AttemptAudit:
    """Return the attempt section."""
    connection.execute(
        "CREATE OR REPLACE TEMP TABLE competition_attempts AS"
        " SELECT competition_id, count(*) AS attempts,"
        " count(*) FILTER (WHERE result = 'bad_lift') AS failed,"
        " count(*) FILTER (WHERE attempt_role = 'record_fourth') AS fourths"
        " FROM competition_attempt GROUP BY competition_id"
    )
    total = _scalar(connection, "SELECT count(*) FROM competition_attempt")
    failed = _scalar(
        connection, "SELECT count(*) FROM competition_attempt WHERE result = 'bad_lift'"
    )
    fourths = _scalar(
        connection,
        "SELECT count(*) FROM competition_attempt WHERE attempt_role = 'record_fourth'",
    )
    with_attempts = _scalar(connection, "SELECT count(*) FROM competition_attempts")
    lifts_without_attempts = _scalar(
        connection,
        "SELECT count(*) FROM competition c WHERE c.competition_event IN"
        f" ({_sql_list(DECLARED_EVENTS)}) AND NOT EXISTS ("
        " SELECT 1 FROM competition_attempt a WHERE a.competition_id = c.competition_id"
        f" AND a.lift IN {_contested_lifts_sql('c')})",
    )
    return AttemptAudit(
        competitions_with_attempts=with_attempts,
        competitions_without_attempts=competitions - with_attempts,
        attempt_detail_fraction=(with_attempts / competitions) if competitions else 0.0,
        attempts_total=total,
        failed_attempts=failed,
        failed_attempt_rate=(failed / total) if total else 0.0,
        fourth_attempts=fourths,
        fourth_attempt_rate=(fourths / total) if total else 0.0,
        fourth_attempts_by_lift=_counts(
            connection,
            "SELECT lift, count(*) FROM competition_attempt"
            " WHERE attempt_role = 'record_fourth' GROUP BY lift ORDER BY lift",
        ),
        attempts_per_competition=_counts(
            connection,
            "SELECT attempts, count(*) FROM competition_attempts GROUP BY attempts"
            " ORDER BY attempts",
        ),
        lifts_without_any_attempt=lifts_without_attempts,
        coverage_by_federation=_attempt_coverage(connection, group='"competition"."federation"'),
        coverage_by_year=_attempt_coverage(
            connection,
            group="CAST(EXTRACT(year FROM competition_date) AS INTEGER)",
        ),
    )


def _performance_audit(connection: duckdb.DuckDBPyConnection) -> PerformanceAudit:
    """Return the reported-performance section.

    The total-versus-best-sum comparison is reported, never reconciled. A source may total
    differently from the sum of its published bests, and choosing which reading is right
    is a question about the meet that PSD is not in a position to answer.
    """
    connection.execute(
        "CREATE OR REPLACE TEMP TABLE bests AS SELECT competition_id,"
        " max(CASE WHEN result_kind = 'squat_best' THEN value END) AS squat,"
        " max(CASE WHEN result_kind = 'bench_best' THEN value END) AS bench,"
        " max(CASE WHEN result_kind = 'deadlift_best' THEN value END) AS deadlift,"
        " max(CASE WHEN result_kind = 'total' THEN value END) AS total"
        " FROM competition_reported_result GROUP BY competition_id"
    )
    totals = _scalar(connection, "SELECT count(*) FROM bests WHERE total IS NOT NULL")
    complete = _scalar(
        connection,
        "SELECT count(*) FROM bests WHERE total IS NOT NULL AND squat IS NOT NULL"
        " AND bench IS NOT NULL AND deadlift IS NOT NULL",
    )
    agreements = _scalar(
        connection,
        "SELECT count(*) FROM bests WHERE total IS NOT NULL AND squat IS NOT NULL"
        " AND bench IS NOT NULL AND deadlift IS NOT NULL"
        " AND abs(total - (squat + bench + deadlift)) <= 0.0005",
    )
    connection.execute(
        "CREATE OR REPLACE TEMP TABLE fourth_attempt_competitions AS"
        " SELECT DISTINCT competition_id FROM competition_attempt"
        " WHERE attempt_role = 'record_fourth'"
    )
    fourth_with_total = _scalar(
        connection,
        "SELECT count(*) FROM fourth_attempt_competitions f JOIN bests b"
        " USING (competition_id) WHERE b.total IS NOT NULL",
    )
    fourth_matching = _scalar(
        connection,
        "SELECT count(*) FROM fourth_attempt_competitions f JOIN bests b"
        " USING (competition_id) WHERE b.total IS NOT NULL AND b.squat IS NOT NULL"
        " AND b.bench IS NOT NULL AND b.deadlift IS NOT NULL"
        " AND abs(b.total - (b.squat + b.bench + b.deadlift)) <= 0.0005",
    )
    return PerformanceAudit(
        negative_reported_bests=_scalar(
            connection,
            "SELECT count(*) FROM competition_reported_result"
            " WHERE source_value_raw < 0 AND reported_best_semantics = 'failed_attempt_only'",
        ),
        negative_reported_bests_by_kind=_counts(
            connection,
            "SELECT result_kind, count(*) FROM competition_reported_result"
            " WHERE source_value_raw < 0 GROUP BY result_kind ORDER BY result_kind",
        ),
        totals=totals,
        totals_without_all_component_bests=totals - complete,
        totals_with_all_component_bests=complete,
        total_sum_agreements=agreements,
        total_sum_disagreements=complete - agreements,
        disagreement_examples=_examples(
            connection,
            "SELECT competition_id, total, squat, bench, deadlift,"
            " total - (squat + bench + deadlift) AS difference FROM bests"
            " WHERE total IS NOT NULL AND squat IS NOT NULL AND bench IS NOT NULL"
            " AND deadlift IS NOT NULL AND abs(total - (squat + bench + deadlift)) > 0.0005"
            " ORDER BY abs(difference) DESC, competition_id LIMIT 10",
        ),
        fourth_attempt_competitions_with_a_total=fourth_with_total,
        fourth_attempt_totals_equal_to_best_sum=fourth_matching,
        reported_results_marked_derived=_scalar(
            connection,
            "SELECT count(*) FROM competition_reported_result WHERE is_derived",
        ),
        bests_without_a_total=_scalar(
            connection,
            "SELECT count(*) FROM bests WHERE total IS NULL AND (squat IS NOT NULL"
            " OR bench IS NOT NULL OR deadlift IS NOT NULL)",
        ),
    )


def _athlete_context_audit(
    connection: duckdb.DuckDBPyConnection, competitions: int
) -> AthleteContextAudit:
    """Return the athlete and context-field section."""
    open_ended = _scalar(
        connection,
        "SELECT count(*) FROM competition"
        f" WHERE regexp_matches(weight_class_raw, '{OPEN_ENDED_WEIGHT_CLASS_PATTERN}')",
    )
    return AthleteContextAudit(
        competitions=competitions,
        bodyweight_present=_scalar(
            connection, "SELECT count(*) FROM competition WHERE bodyweight_kg IS NOT NULL"
        ),
        bodyweight_missing=_scalar(
            connection, "SELECT count(*) FROM competition WHERE bodyweight_kg IS NULL"
        ),
        bodyweight_missing_fraction=(
            _scalar(connection, "SELECT count(*) FROM competition WHERE bodyweight_kg IS NULL")
            / competitions
            if competitions
            else 0.0
        ),
        ages_exact=_scalar(
            connection, "SELECT count(*) FROM competition WHERE age_precision = 'exact'"
        ),
        ages_approximate=_scalar(
            connection, "SELECT count(*) FROM competition WHERE age_precision = 'approximate'"
        ),
        ages_missing=_scalar(
            connection, "SELECT count(*) FROM competition WHERE age_reported IS NULL"
        ),
        weight_classes_present=_scalar(
            connection, "SELECT count(*) FROM competition WHERE weight_class_raw IS NOT NULL"
        ),
        distinct_weight_classes=_scalar(
            connection, "SELECT count(DISTINCT weight_class_raw) FROM competition"
        ),
        open_ended_weight_classes=open_ended,
        tested_category_yes=_scalar(
            connection, "SELECT count(*) FROM competition WHERE is_drug_tested_category IS TRUE"
        ),
        tested_category_no=_scalar(
            connection, "SELECT count(*) FROM competition WHERE is_drug_tested_category IS FALSE"
        ),
        tested_category_missing=_scalar(
            connection, "SELECT count(*) FROM competition WHERE is_drug_tested_category IS NULL"
        ),
        meets_sanctioned=_scalar(
            connection, "SELECT count(*) FROM competition_meet WHERE is_sanctioned IS TRUE"
        ),
        meets_unsanctioned=_scalar(
            connection, "SELECT count(*) FROM competition_meet WHERE is_sanctioned IS FALSE"
        ),
        meets_sanction_unknown=_scalar(
            connection, "SELECT count(*) FROM competition_meet WHERE is_sanctioned IS NULL"
        ),
    )


def _anomaly_audit(
    connection: duckdb.DuckDBPyConnection, review: SourceSchemaReview
) -> AnomalyAudit:
    """Return the source-anomaly section."""
    connection.execute(
        "CREATE OR REPLACE TEMP TABLE meet_identity AS SELECT competition_meet_id,"
        " meet_name, meet_town, meet_state, meet_country, meet_date, meet_federation,"
        " lower(trim(coalesce(meet_name, ''))) || '|' || lower(trim(coalesce(meet_town, '')))"
        " || '|' || lower(trim(coalesce(meet_state, ''))) || '|'"
        " || lower(trim(coalesce(meet_country, ''))) AS place_key"
        " FROM competition_meet"
    )
    unknown_codes = _counts(
        connection,
        "SELECT participation_status, count(*) FROM competition"
        " WHERE participation_place IS NULL AND participation_status_kind = 'unknown'"
        " GROUP BY participation_status ORDER BY participation_status",
    )
    return AnomalyAudit(
        disambiguated_identities=_scalar(
            connection,
            "SELECT count(*) FROM athlete_source_link WHERE regexp_matches("
            r" source_athlete_key, '#\d+\s*$')",
        ),
        sex_category_conflicts=_scalar(
            connection, "SELECT count(*) FROM athlete WHERE ambiguity_group_id IS NOT NULL"
        ),
        sex_category_conflict_examples=tuple(
            sorted(
                _counts(
                    connection,
                    "SELECT ambiguity_group_id, count(*) FROM athlete"
                    " WHERE ambiguity_group_id IS NOT NULL GROUP BY ambiguity_group_id"
                    " ORDER BY ambiguity_group_id LIMIT 20",
                )
            )
        ),
        sex_categories_absent=_scalar(
            connection, "SELECT count(*) FROM athlete WHERE sex_category IS NULL"
        ),
        meet_identity_collisions=_scalar(
            connection,
            "SELECT count(*) FROM (SELECT place_key FROM meet_identity"
            " GROUP BY place_key HAVING count(DISTINCT competition_meet_id) > 1)",
        ),
        meet_identity_collision_examples=_examples(
            connection,
            "SELECT place_key, count(DISTINCT competition_meet_id) AS meet_identities,"
            " min(meet_date) AS first_date, max(meet_date) AS last_date,"
            " list(DISTINCT meet_federation) AS federations FROM meet_identity"
            " GROUP BY place_key HAVING count(DISTINCT competition_meet_id) > 1"
            " ORDER BY meet_identities DESC, place_key LIMIT 10",
        ),
        competitions_without_a_meet_date=_scalar(
            connection, "SELECT count(*) FROM competition WHERE competition_date IS NULL"
        ),
        competitions_disagreeing_with_their_meet=_scalar(
            connection,
            "SELECT count(*) FROM competition c JOIN competition_meet m"
            " USING (competition_meet_id) WHERE c.competition_date IS NOT NULL"
            " AND m.meet_date IS NOT NULL AND c.competition_date <> m.meet_date",
        ),
        earliest_meet_date=_optional_text(
            connection,
            "SELECT strftime(min(competition_date), '%Y-%m-%dT%H:%M:%SZ') FROM competition",
        ),
        latest_meet_date=_optional_text(
            connection,
            "SELECT strftime(max(competition_date), '%Y-%m-%dT%H:%M:%SZ') FROM competition",
        ),
        unexpected_event_values=tuple(
            sorted(
                _counts(
                    connection,
                    "SELECT competition_event, count(*) FROM competition"
                    " WHERE competition_event IS NULL OR competition_event NOT IN"
                    f" ({_sql_list(DECLARED_EVENTS)})"
                    " GROUP BY competition_event ORDER BY competition_event",
                )
            )
        ),
        unexpected_equipment_values=tuple(
            sorted(
                _counts(
                    connection,
                    "SELECT equipment_class_raw, count(*) FROM competition"
                    " WHERE equipment_class_raw IS NOT NULL AND equipment_class = 'unknown'"
                    " GROUP BY equipment_class_raw ORDER BY equipment_class_raw",
                )
            )
        ),
        unexpected_sex_values=tuple(
            sorted(
                _counts(
                    connection,
                    "SELECT sex_category_raw, count(*) FROM athlete"
                    " WHERE sex_category_raw IS NOT NULL AND sex_category IS NULL"
                    " GROUP BY sex_category_raw ORDER BY sex_category_raw",
                )
            )
        ),
        unknown_place_codes=sum(unknown_codes.values()),
        unknown_place_examples=tuple(label for label in unknown_codes if label != _NULL_LABEL),
        placements_outside_range=_scalar(
            connection,
            "SELECT count(*) FROM competition WHERE participation_place IS NOT NULL"
            " AND participation_place <= 0",
        ),
        competitions_with_unknown_event=_scalar(
            connection,
            "SELECT count(*) FROM competition WHERE competition_event NOT IN"
            f" ({_sql_list(DECLARED_EVENTS)})",
        ),
        schema_drift_unknown_columns=tuple(review.unknown),
        schema_drift_missing_columns=tuple(review.missing),
        schema_drift_reordered=review.reordered,
        source_disagreements=(),
    )


# --------------------------------------------------------------------------
# expansion invariants
# --------------------------------------------------------------------------


def expansion_invariants(connection: duckdb.DuckDBPyConnection) -> tuple[Invariant, ...]:
    """Return the count-level checks that hold :data:`EXPANSION_RULES` against the corpus.

    Every invariant is a count the rule requires to be zero, phrased so that a reader can
    recompute it. They are the answer to "does this corpus actually do what the
    documentation says", which is the only question an expansion audit can settle.
    """
    checks: tuple[tuple[str, str, str], ...] = (
        (
            "attempts_reference_a_competition",
            "every attempt row references a competition row that exists",
            "SELECT count(*) FROM competition_attempt a WHERE NOT EXISTS"
            " (SELECT 1 FROM competition c WHERE c.competition_id = a.competition_id)",
        ),
        (
            "reported_results_reference_a_competition",
            "every reported-result row references a competition row that exists",
            "SELECT count(*) FROM competition_reported_result r WHERE NOT EXISTS"
            " (SELECT 1 FROM competition c WHERE c.competition_id = r.competition_id)",
        ),
        (
            "competitions_reference_a_meet",
            "every competition row references a meet row that exists",
            "SELECT count(*) FROM competition c WHERE NOT EXISTS"
            " (SELECT 1 FROM competition_meet m"
            " WHERE m.competition_meet_id = c.competition_meet_id)",
        ),
        (
            "attempts_are_never_zero",
            "no stored attempt load is zero or negative: a zero cell is dropped and a"
            " negative source sign is resolved into a result",
            "SELECT count(*) FROM competition_attempt WHERE load_kg IS NULL OR load_kg <= 0",
        ),
        (
            "attempt_result_agrees_with_the_source_sign",
            "an attempt is good exactly when its source value was positive",
            "SELECT count(*) FROM competition_attempt WHERE"
            " (source_attempt_raw > 0) <> (result = 'good_lift')",
        ),
        (
            "reported_result_sign_agrees_with_its_semantics",
            "a reported best is labelled failed_attempt_only exactly when its source"
            " value was negative, and a successful best is never negative",
            "SELECT count(*) FROM competition_reported_result WHERE"
            " (source_value_raw < 0) <> (reported_best_semantics = 'failed_attempt_only')",
        ),
        (
            "reported_values_are_magnitudes",
            "no reported-result value is negative: a negative best is stored as its"
            " magnitude with its source sign kept beside it",
            "SELECT count(*) FROM competition_reported_result WHERE value < 0",
        ),
        (
            "no_reported_result_is_derived",
            "every reported result is source-published: PSD derives none, so none is"
            " marked derived",
            "SELECT count(*) FROM competition_reported_result WHERE is_derived",
        ),
        (
            "fourth_attempts_are_labelled",
            "every attempt numbered four carries the record-attempt role and no other attempt does",
            "SELECT count(*) FROM competition_attempt WHERE"
            " (attempt_number = 4) <> (attempt_role = 'record_fourth')",
        ),
        (
            "attempt_numbers_are_unique_per_lift",
            "a participation records each attempt number at most once per lift",
            "SELECT coalesce(sum(n - 1), 0) FROM (SELECT count(*) AS n"
            " FROM competition_attempt GROUP BY competition_id, lift, attempt_number"
            " HAVING count(*) > 1)",
        ),
        (
            "attempts_stay_inside_the_declared_event",
            "no attempt exists at a lift its participation's declared event does not"
            " contest, so a reduced event never manufactures an absent lift",
            "SELECT count(*) FROM competition_attempt a JOIN competition c"
            " USING (competition_id) WHERE a.lift NOT IN"
            f" {_contested_lifts_sql('c')}",
        ),
        (
            "bodyweight_is_never_zero",
            "no stored body mass is zero or negative: an absent or non-positive weigh-in"
            " stays absent",
            "SELECT count(*) FROM competition WHERE bodyweight_kg IS NOT NULL"
            " AND bodyweight_kg <= 0",
        ),
        (
            "approximate_ages_are_never_rounded",
            "an approximate age keeps a fractional part, and an exact age is a whole number",
            "SELECT count(*) FROM competition WHERE age_precision = 'approximate'"
            " AND abs(age_reported - round(age_reported)) < 1e-9",
        ),
        (
            "participation_places_are_positive",
            "a numeric placing is positive, so no status code was coerced into a rank",
            "SELECT count(*) FROM competition WHERE participation_place IS NOT NULL"
            " AND participation_place <= 0",
        ),
        (
            "every_athlete_has_a_competition",
            "every athlete identity has at least one participation, so a history row is"
            " always backed by events",
            "SELECT count(*) FROM athlete a WHERE NOT EXISTS"
            " (SELECT 1 FROM competition c WHERE c.athlete_id = a.athlete_id)",
        ),
    )
    return tuple(
        Invariant(name=name, statement=statement, expected=0, observed=_scalar(connection, sql))
        for name, statement, sql in checks
    )


# --------------------------------------------------------------------------
# orchestration
# --------------------------------------------------------------------------


def _dataset_paths(request: AuditRequest) -> tuple[DatasetManifest, Path]:
    """Return the dataset manifest and its directory, resolved inside the data root."""
    manifest = read_manifest(request.dataset_dir, data_root=request.data_root)
    directory = resolve_within_data_root(request.dataset_dir, data_root=request.data_root)
    paths = artifact_paths(manifest, request.dataset_dir, data_root=request.data_root)
    missing = sorted(set(corpus_table_names()) - set(paths))
    if missing:
        msg = (
            f"Cannot audit {manifest.dataset_id}: it does not declare the canonical "
            f"table(s) {', '.join(missing)}."
        )
        raise AuditError(msg)
    return manifest, directory


def _snapshot_for(
    request: AuditRequest, manifest: DatasetManifest
) -> OpenPowerliftingSnapshot | None:
    """Return the pinned snapshot this corpus was built from, when it is still readable.

    The snapshot is provenance, not a dependency: an audit of a corpus whose snapshot has
    been removed must still run and still report everything it can derive from the
    corpus. The source section then records that the facts were unavailable, which is an
    acknowledged gap rather than a fabricated one.
    """
    digest = request.archive_sha256
    if digest is None and manifest.sources:
        digest = manifest.sources[0].snapshot_sha256
    if digest is None:
        return None
    try:
        return read_pinned_snapshot(
            snapshot_directory(digest, data_root=request.data_root), data_root=request.data_root
        )
    except (AcquisitionError, ValueError):
        return None


def _history_rows(request: AuditRequest, directory: Path) -> int | None:
    """Return the derived history's row count, when the artifact has been built.

    Reported rather than required: an audit of a corpus whose histories have not been
    derived is still a complete audit of the corpus, and says so by leaving the field
    null instead of guessing a number.
    """
    path = directory / "derived" / "athlete_history.parquet"
    if not path.is_file():
        return None
    connection = _open(request)
    try:
        cursor = connection.execute("SELECT count(*) FROM read_parquet(?)", [str(path)])
        row = cursor.fetchone()
    finally:
        connection.close()
    if row is None or row[0] is None:
        return None
    return int(row[0])


def audit_corpus(request: AuditRequest) -> AuditResult:
    """Audit a built PSD-COMP corpus and, unless told otherwise, persist the report.

    Args:
        request: Which corpus to audit, and where to write the result.

    Returns:
        The report, where it was written, and how long the pass took.

    Raises:
        AuditError: The corpus cannot be audited: its manifest is missing, unreadable, or
            does not declare the canonical tables.
    """
    started = time.perf_counter()
    manifest, directory = _dataset_paths(request)
    snapshot = _snapshot_for(request, manifest)
    generated = request.generated_at or datetime.now(tz=UTC)
    connection = _open(request)
    try:
        for table in corpus_table_names():
            _register(connection, table, directory / "tables" / f"{table}.parquet")
        source = _source_audit(manifest, snapshot)
        entities = _entity_audit(connection)
        coverage = _coverage_audit(connection)
        history_rows = _history_rows(request, directory)
        longitudinal = _longitudinal_audit(connection, history_rows=history_rows)
        attempts = _attempt_audit(connection, entities.competitions)
        performance = _performance_audit(connection)
        context = _athlete_context_audit(connection, entities.competitions)
        review = review_source_schema(source.header)
        anomalies = _anomaly_audit(connection, review)
        invariants = expansion_invariants(connection)
    finally:
        connection.close()

    disagreements = snapshot.service.disagreements() if snapshot else ()
    audit = CorpusAudit(
        audit_version=AUDIT_VERSION,
        dataset_id=manifest.dataset_id,
        dataset_relative_path=str(Path(request.dataset_dir).as_posix()),
        schema_version=manifest.schema_version,
        manifest_digest=manifest_digest(manifest),
        source=source.model_copy(update={"schema_review": _review_dict(source)}),
        entities=entities,
        coverage=coverage,
        longitudinal=longitudinal,
        attempts=attempts,
        performance=performance,
        athlete_context=context,
        anomalies=anomalies.model_copy(update={"source_disagreements": disagreements}),
        expansion=ExpansionAudit(
            rules=EXPANSION_RULES,
            invariants=tuple(item.to_dict() for item in invariants),
        ),
        generated_at=generated,
    )
    json_path, markdown_path = _persist(request, audit)
    return AuditResult(
        audit=audit,
        json_path=json_path,
        markdown_path=markdown_path,
        audit_seconds=time.perf_counter() - started,
    )


def _review_dict(source: SourceAudit) -> Mapping[str, object]:
    """Return the schema review as a mapping, including its one-line summary."""
    payload = dict(source.schema_review)
    review = review_source_schema(source.header)
    payload["summary"] = review.summary()
    return payload


def _persist(request: AuditRequest, audit: CorpusAudit) -> tuple[Path | None, Path | None]:
    """Write the report unless the caller asked for it in memory only."""
    base = (
        request.output_dir
        if request.output_dir is not None
        else Path(request.dataset_dir) / AUDIT_DIRNAME
    )
    directory = resolve_within_data_root(base, data_root=request.data_root, create=True)
    directory.mkdir(parents=True, exist_ok=True)
    json_path = directory / "corpus_audit.json"
    json_path.write_bytes(audit.to_json_bytes())
    markdown_path = directory / "corpus_audit.md"
    markdown_path.write_text(audit_markdown(audit), encoding="utf-8")
    return json_path, markdown_path


def write_audit(
    audit: CorpusAudit,
    directory: Path,
    *,
    data_root: Path | None = None,
) -> tuple[Path, Path]:
    """Write a report's machine-readable and human-readable renderings.

    Args:
        audit: The report.
        directory: Destination directory, relative to the data root.
        data_root: External PSD data root.

    Returns:
        The JSON path and the Markdown path.
    """
    target = resolve_within_data_root(directory, data_root=data_root, create=True)
    target.mkdir(parents=True, exist_ok=True)
    json_path = target / "corpus_audit.json"
    json_path.write_bytes(audit.to_json_bytes())
    markdown_path = target / "corpus_audit.md"
    markdown_path.write_text(audit_markdown(audit), encoding="utf-8")
    return json_path, markdown_path


def read_audit(path: Path) -> CorpusAudit:
    """Read a persisted audit report.

    Raises:
        AuditError: The file is missing or is not an audit report.
    """
    if not path.is_file():
        msg = f"No audit report at {path}."
        raise AuditError(msg)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        msg = f"Unreadable audit report at {path}: {error}"
        raise AuditError(msg) from error
    try:
        return CorpusAudit.model_validate(payload)
    except ValueError as error:
        msg = f"Invalid audit report at {path}: {error}"
        raise AuditError(msg) from error
