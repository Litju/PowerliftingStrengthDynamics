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

The four closure diagnostics
----------------------------

:func:`audit_corpus` answers four questions the corpus itself cannot, and each answer is
kept in its own section rather than folded into the anomaly list:

Identity stability
    What the source identity key can and cannot tell us, including the one thing it
    cannot answer at all. OpenPowerlifting publishes ``Name`` *as* the identity key and
    publishes no independent stable person identifier, so a historical name change is
    **not identifiable** from this source. The audit says so in machine-readable form
    rather than shipping a plausible-looking detector that could only ever be guessing.

Suspicious chronology
    Three source-grounded checks: competitions dated after the pinned snapshot, meet
    identities spanning more than one source date, and per-identity age/date
    inconsistency under the source's own documented age semantics.

Unit consistency
    A raw-to-canonical *mass fidelity* audit. The source states its masses in kilograms
    and carries no per-row unit field, so the honest question is not "does this look like
    a plausible lifter" but "does the canonical value equal the source value, with the
    sign carrying only the attempt result". No plausibility heuristic is applied and no
    pounds are inferred.

Equipment and federation transitions
    Descriptive, computed per athlete-meet rather than per source row so that several
    event entries at one meet are never read as a longitudinal switch.

Each section declares a :class:`~psd.schema.vocabulary.FindingClass`, because a count
whose meaning is unstated cannot be acted on: a source anomaly is reported, an invariant
failure is a defect, a transition is ordinary, and a limitation is a silence.
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
from psd.ingest.openpowerlifting.contract import (
    SOURCE_COLUMNS,
    SourceColumnDisposition,
    SourceSchemaReview,
    review_source_schema,
)
from psd.ingest.openpowerlifting.history import CompetitionEventMembership
from psd.ingest.openpowerlifting.mapping import REPORTED_RESULT_SPECS
from psd.ingest.openpowerlifting.snapshot import OpenPowerliftingSnapshot
from psd.ingest.openpowerlifting.transform import corpus_table_names
from psd.paths import resolve_within_data_root
from psd.provenance.manifest import DatasetManifest, manifest_digest
from psd.schema.registry import table_spec
from psd.schema.vocabulary import EquipmentClass, FindingClass
from psd.serialization.dataset import artifact_paths, read_manifest

__all__ = (
    "AUDIT_DIRNAME",
    "AUDIT_VERSION",
    "EXPANSION_RULES",
    "MASS_SOURCE_FIELDS",
    "NAME_CHANGE_LIMITATION_REASON",
    "AnomalyAudit",
    "AthleteContextAudit",
    "AttemptAudit",
    "AttemptCoverage",
    "AuditError",
    "AuditRequest",
    "AuditResult",
    "CategoryCoverage",
    "ChronologyAudit",
    "CorpusAudit",
    "CoverageAudit",
    "DiagnosticFinding",
    "DiagnosticsAudit",
    "EntityAudit",
    "ExpansionAudit",
    "ExpansionRule",
    "FindingClassMeaning",
    "IdentityStabilityAudit",
    "Invariant",
    "LongitudinalAudit",
    "PerformanceAudit",
    "SourceAudit",
    "TransitionAudit",
    "TransitionFieldAudit",
    "UnitFidelityAudit",
    "audit_corpus",
    "audit_markdown",
    "corpus_expansion_invariants",
    "coverage_table",
    "expansion_invariants",
    "read_audit",
    "write_audit",
)

#: Version of the audit report contract. Bumped when a section's meaning changes, so a
#: stored report can never be read as describing something it did not describe.
AUDIT_VERSION: Final[str] = "psd-comp-audit/2"

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

#: Tolerance for the raw-to-canonical mass comparisons. The transform copies the source
#: float rather than recomputing it, so this exists to absorb decimal representation and
#: not to paper over a real conversion: a genuine pounds-to-kilograms conversion would
#: miss by orders of magnitude, not by the last bit.
UNIT_TOLERANCE: Final[float] = 1e-9

#: The source fields that carry a mass, and therefore have a unit to be wrong about.
#: Declared from the one place that classifies every source column, so this list cannot
#: drift from the contract the build actually enforced.
MASS_SOURCE_FIELDS: Final[tuple[str, ...]] = tuple(
    spec.name
    for spec in SOURCE_COLUMNS
    if spec.disposition is SourceColumnDisposition.MAPPED
    and (spec.name.endswith("Kg") or spec.destination == "competition.bodyweight_kg")
)

#: Why a historical name change is not reported. Stated once and carried into the report
#: verbatim: the reason is a property of the source, not of this audit, and a reader who
#: cannot find the sentence explaining an absent diagnostic will assume it was forgotten.
NAME_CHANGE_LIMITATION_REASON: Final[str] = (
    "OpenPowerlifting Name is itself the source identity key"
)

#: The one question the pinned source cannot answer about identity, and the label a
#: machine-readable consumer is expected to test for.
NAME_CHANGE_LIMITATION_QUESTION: Final[str] = (
    "did any source athlete identity change name over its longitudinal history"
)

#: Which basis supplied the pinned snapshot date, recorded beside every date-dependent
#: chronology count so a reader knows whether the date came from the service or from the
#: archive's own filename.
SNAPSHOT_DATE_BASIS_SERVICE: Final[str] = "service_reported"
SNAPSHOT_DATE_BASIS_ARCHIVE: Final[str] = "archive_declared"
SNAPSHOT_DATE_BASIS_UNAVAILABLE: Final[str] = "unavailable"

#: How many consecutive longitudinal state changes are grouped together when reporting a
#: distribution over athletes. A handful of shape buckets, because a raw per-athlete
#: histogram would be a corpus of numbers nobody reads.
_TRANSITION_BUCKETS: Final[tuple[tuple[str, int, int], ...]] = (
    ("0 transitions", 0, 0),
    ("1 transition", 1, 1),
    ("2-3 transitions", 2, 3),
    ("4-9 transitions", 4, 9),
    ("10-19 transitions", 10, 19),
    ("20-49 transitions", 20, 49),
    ("50 or more transitions", 50, 1_000_000),
)

#: Collision groups are reported by how many source name keys share one base name.
_BASE_GROUP_BUCKETS: Final[tuple[tuple[str, int, int], ...]] = (
    ("2 name keys", 2, 2),
    ("3 name keys", 3, 3),
    ("4-5 name keys", 4, 5),
    ("6-10 name keys", 6, 10),
    ("11 or more name keys", 11, 1_000_000),
)

#: Result kinds that carry a mass, and the subset carrying best-lift semantics. Both are
#: derived from the one declared table of source-column-to-result-kind mappings, so a new
#: reported column cannot be silently excluded from the unit audit.
_MASS_RESULT_KINDS: Final[tuple[str, ...]] = tuple(
    kind for column, kind, _is_best in REPORTED_RESULT_SPECS if column.endswith("Kg")
)
_BEST_RESULT_KINDS: Final[tuple[str, ...]] = tuple(
    kind for _column, kind, is_best in REPORTED_RESULT_SPECS if is_best
)
_TOTAL_RESULT_KIND: Final[str] = "total"

#: The canonical equipment value meaning "the source did not say". It is a declared
#: unknown, not an equipment category, so it is excluded from transition state exactly as
#: a null federation is: a meet where the source stayed silent has no state to order.
_UNKNOWN_EQUIPMENT_CLASS: Final[str] = EquipmentClass.UNKNOWN.value


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


# --------------------------------------------------------------------------
# closure diagnostics
# --------------------------------------------------------------------------


class DiagnosticLimitation(_Model):
    """A question the pinned source cannot answer, stated rather than approximated.

    A limitation is reported as a record rather than as an absent field for one reason:
    "the audit looked and found nothing" and "the source does not carry this at all"
    render identically in JSON, and a consumer that cannot tell them apart will read a
    silence as a clean result.

    Attributes:
        question: What could not be determined.
        identifiable: Always ``False`` for a limitation that exists. Explicit so that a
            machine-readable consumer can test for it rather than infer it from a null.
        reason: Why the source does not support the question.
    """

    question: str
    identifiable: bool = False
    reason: str


class IdentityStabilityAudit(_Model):
    """What the source identity key can and cannot support.

    The finding class is :attr:`FindingClass.SOURCE_LIMITATION` because the governing
    result of this section is a *silence*: OpenPowerlifting publishes ``Name`` as its
    identity key and exposes no independent stable person identifier, so no historical
    name change is discoverable here. Everything else in the section is either a genuine
    conflict in the source or evidence about how the ``#N`` disambiguator is used.

    Country, state, federation, body mass, age class, and division are deliberately
    absent: they legitimately vary over a career, and reading any of them as an identity
    change would manufacture findings out of ordinary longitudinal variation.

    Attributes:
        finding_class: How these counts must be read.
        identities: Source identities in the corpus.
        names_under_multiple_sex_categories: Exact source ``Name`` keys the source
            publishes under more than one reported sex category. One flagged identity
            each; never resolved by picking one.
        name_keys_mapping_to_several_identities: Name keys that produced more than one
            canonical identity. The declared identity is the name alone, so this must be
            zero and a non-zero value is a transform defect.
        identities_mapping_to_several_name_keys: Canonical identities carrying more than
            one source name key. Likewise a transform defect rather than a source fact.
        disambiguated_names: Name keys carrying the source's trailing ``#N`` disambiguator.
        base_name_collision_groups: Distinct base names -- a name key with any trailing
            ``#N`` removed -- shared by more than one name key. This is the observable
            shape of the source's own same-name handling.
        largest_base_name_collision_group: Most name keys sharing one base name.
        base_name_group_size_distribution: Collision groups by how many keys they hold.
        base_names_also_published_unsuffixed: Collision groups where the bare base name is
            itself published as a separate identity. The source can then be describing
            one lifter under two keys, which is precisely why the identities are not
            merged.
        base_name_examples: The largest collision groups, with their member keys.
        limitations: The questions this source cannot answer.
    """

    finding_class: FindingClass = FindingClass.SOURCE_LIMITATION
    identities: int
    names_under_multiple_sex_categories: int
    name_keys_mapping_to_several_identities: int
    identities_mapping_to_several_name_keys: int
    disambiguated_names: int
    base_name_collision_groups: int
    largest_base_name_collision_group: int
    base_name_group_size_distribution: Mapping[str, int]
    base_names_also_published_unsuffixed: int
    base_name_examples: tuple[Mapping[str, object], ...]
    limitations: tuple[DiagnosticLimitation, ...]


class ChronologyAudit(_Model):
    """Suspicious chronology, measured against source facts only.

    Three independent questions, each grounded in something the source states:

    * a competition dated **after** the pinned snapshot cannot have been in it;
    * a meet identity spanning more than one source date would mean the six-field meet
      rule disagreed with itself;
    * an identity whose ages cannot all describe the same person has a date or age the
      source recorded inconsistently.

    The age check uses the source's documented age semantics rather than a tolerance:
    an exact age ``n`` at meet year ``Y`` admits birth year ``Y-n`` or ``Y-n-1``
    depending on whether the birthday had passed, and an approximate age ``n+0.5`` is
    the midpoint of that range and therefore admits exactly ``Y-(n+1)``. Intersecting
    those sets across an identity's history yields birth years compatible with every
    observation, and an empty intersection is a reportable inconsistency.

    This is a diagnostic. No age is rewritten, and no identity is split or merged from it.

    Attributes:
        finding_class: How these counts must be read.
        pinned_snapshot_date: The snapshot date the future-dated check used, as the
            source published it.
        pinned_snapshot_date_basis: Which statement supplied that date.
        future_dated_competitions: Participations dated after the pinned snapshot.
        future_dated_meets: Meet identities dated after the pinned snapshot.
        future_dated_examples: Named examples.
        meets_spanning_multiple_dates: Meet identities with more than one distinct source
            date. Zero by construction, and reported precisely because that construction
            is a claim somebody can check.
        meet_date_examples: Named examples.
        age_observations: Participations carrying both a readable age and a meet date.
        identities_with_age_observations: Source identities among them.
        identities_with_multiple_age_observations: Identities whose history offers more
            than one age observation, the only ones an intersection can speak about.
        identities_without_a_compatible_birth_year: Identities whose observations admit no
            common birth year.
        incompatible_identity_examples: Named examples, worst first.
        age_semantics: The compatibility rule, in words, as applied.
    """

    finding_class: FindingClass = FindingClass.SOURCE_ANOMALY
    pinned_snapshot_date: str | None
    pinned_snapshot_date_basis: str
    future_dated_competitions: int | None
    future_dated_meets: int | None
    future_dated_examples: tuple[Mapping[str, object], ...]
    meets_spanning_multiple_dates: int
    meet_date_examples: tuple[Mapping[str, object], ...]
    age_observations: int
    identities_with_age_observations: int
    identities_with_multiple_age_observations: int
    identities_without_a_compatible_birth_year: int
    incompatible_identity_examples: tuple[Mapping[str, object], ...]
    age_semantics: str


class UnitFidelityAudit(_Model):
    """Raw-to-canonical mass fidelity.

    Not a plausibility check. The bulk source defines ``BodyweightKg``, the twelve
    attempt ``*Kg`` columns, the three ``Best3*Kg`` columns and ``TotalKg`` in kilograms
    and carries no per-row unit field, so the question worth answering is whether the
    canonical value is the source value -- sign included in its meaning -- rather than
    whether a 74 kg lifter looks like a lifter. A physiological range would be a
    different, assumption-heavy diagnostic that this deliberately does not run.

    The scoring-system columns (``Dots``, ``Wilks``, ``Glossbrenner``, ``Goodlift``) are
    dimensionless and are checked to *not* claim kilograms, so a consumer can never read
    a Dots score as a mass.

    Every mismatch counter here is expected to be zero for a correct transform. A
    non-zero value is a defect in PSD.

    Attributes:
        finding_class: How these counts must be read.
        declared_source_unit: What the source documentation states its masses are.
        mass_source_fields: The source columns carrying a mass, taken from the contract.
        bodyweight_rows_checked: Participations carrying a source body mass.
        bodyweight_kg_mismatches: Where the canonical body mass is not the source value.
        bodyweight_unit_mismatches: Where a recorded body mass is not labelled kilograms.
        bodyweight_absent_became_zero: Where an absent body mass became a zero. Absence
            must stay absence.
        attempt_rows_checked: Canonical attempt rows.
        attempt_load_kg_mismatches: Where the canonical load is not ``abs(source)``.
        attempt_sign_result_mismatches: Where the source sign disagrees with the result.
        attempt_unit_mismatches: Where an attempt load is not labelled kilograms.
        attempt_loads_not_positive: Where a stored load is zero or negative.
        reported_rows_checked: Canonical reported-result rows.
        reported_value_mismatches: Where the canonical value is not the magnitude of the
            source value.
        reported_mass_rows_checked: Reported rows carrying a mass.
        reported_mass_unit_mismatches: Where a mass-bearing result is not labelled kilograms.
        reported_best_rows_checked: Reported best-lift rows.
        reported_best_semantics_mismatches: Where the best-lift reading disagrees with the
            sign of the source value.
        reported_totals_checked: Reported totals.
        reported_total_unit_mismatches: Where a total is not labelled kilograms.
        open_ended_weight_classes: ``WeightClassKg`` values in the open-ended ``90+`` form.
        open_ended_weight_classes_coerced: Open-ended classes that reached canonical data
            as a bare measurement. The label is persisted verbatim and no numeric
            weight-class column exists, so this must be zero.
        scoring_rows_checked: Dimensionless reported rows.
        scoring_rows_claiming_a_mass_unit: Scoring rows labelled kilograms.
        mismatch_total: Sum of every mismatch counter above: the single number a correct
            transform must report as zero.
    """

    finding_class: FindingClass = FindingClass.TRANSFORMATION_INVARIANT_FAILURE
    declared_source_unit: str
    mass_source_fields: tuple[str, ...]
    bodyweight_rows_checked: int
    bodyweight_kg_mismatches: int
    bodyweight_unit_mismatches: int
    bodyweight_absent_became_zero: int
    attempt_rows_checked: int
    attempt_load_kg_mismatches: int
    attempt_sign_result_mismatches: int
    attempt_unit_mismatches: int
    attempt_loads_not_positive: int
    reported_rows_checked: int
    reported_value_mismatches: int
    reported_mass_rows_checked: int
    reported_mass_unit_mismatches: int
    reported_best_rows_checked: int
    reported_best_semantics_mismatches: int
    reported_totals_checked: int
    reported_total_unit_mismatches: int
    open_ended_weight_classes: int
    open_ended_weight_classes_coerced: int
    scoring_rows_checked: int
    scoring_rows_claiming_a_mass_unit: int
    mismatch_total: int


class TransitionFieldAudit(_Model):
    """One longitudinal state field's transitions, computed per athlete-meet.

    Attributes:
        field: The canonical column whose value is treated as the state.
        source_column: The source column it came from.
        athlete_meets: Athlete-meets in the corpus.
        athlete_meets_with_a_known_state: Athlete-meets where the source stated exactly
            one value.
        intra_meet_state_conflicts: Athlete-meets where the source stated more than one
            value. No order is invented inside a meet, so these take no part in the
            transition count.
        total_transitions: Changes between consecutive known states.
        athletes_with_transitions: Source identities with at least one change.
        transitions_per_athlete: Athletes bucketed by how many changes they have.
    """

    field: str
    source_column: str
    athlete_meets: int
    athlete_meets_with_a_known_state: int
    intra_meet_state_conflicts: int
    total_transitions: int
    athletes_with_transitions: int
    transitions_per_athlete: Mapping[str, int]


class TransitionAudit(_Model):
    """Descriptive equipment and federation transitions over a competition history.

    Computed per **athlete-meet**, never per source row: a lifter who entered both the
    ``SBD`` and the ``B`` at one meet produces two source rows, and reading those as two
    observations of one meet would turn a single result into an invented switch.

    An athlete-meet where the source stated more than one value for a field is reported as
    an *intra-meet state conflict* and excluded from the ordering, because a meet has no
    internal order to break a tie with. A meet where the source stated nothing contributes
    no state either, so a long gap in reporting cannot be read as a change.

    Equipment here is the competition **category** -- what the rules allowed -- and is
    never a claim about the gear a lifter actually wore.

    Attributes:
        finding_class: How these counts must be read.
        grouping: The unit transitions are computed at.
        ordering: The key athlete-meets are ordered by.
        equipment_reading: What the equipment column means, restated in the report.
        unknown_is_a_state: Always ``False``. A declared unknown is an absence, not a
            state to transition into or out of.
        fields: One entry per tracked field.
    """

    finding_class: FindingClass = FindingClass.DESCRIPTIVE_LONGITUDINAL_TRANSITION
    grouping: str = "athlete_id + competition_meet_id"
    ordering: str = "competition_date, competition_meet_id"
    equipment_reading: str = (
        "equipment_class is the competition category, i.e. the equipment the rules "
        "allowed; it is not evidence of the equipment an athlete actually wore"
    )
    unknown_is_a_state: bool = False
    fields: tuple[TransitionFieldAudit, ...]

    def field(self, name: str) -> TransitionFieldAudit:
        """Return the entry for *name*.

        Raises:
            KeyError: No field by that name is tracked.
        """
        for entry in self.fields:
            if entry.field == name:
                return entry
        raise KeyError(name)


class FindingClassMeaning(_Model):
    """What one finding class means and what a reader should do with it.

    Attributes:
        finding_class: The class being explained.
        meaning: One sentence on what a count in this class is.
        treatment: What the corpus does about it.
    """

    finding_class: FindingClass
    meaning: str
    treatment: str


class DiagnosticFinding(_Model):
    """One classified count from one diagnostic family.

    Attributes:
        family: Which diagnostic section produced the count.
        name: Stable identifier for the count.
        finding_class: What the count means.
        count: The count itself. Negative values never appear; ``None`` is used where a
            count could not be measured, which :class:`UnitFidelityAudit` never needs and
            :class:`ChronologyAudit` does when the snapshot date is unavailable.
        statement: What the count counts, in words.
    """

    family: str
    name: str
    finding_class: FindingClass
    count: int | None
    statement: str


class DiagnosticsAudit(_Model):
    """The four closure diagnostic families and the classification index over them.

    ``findings`` is a flattened index, not a replacement: each count is still in its own
    typed section, and the index exists so a consumer can filter by class without walking
    four schemas. A count appearing in both places is the same count, produced once.

    Attributes:
        identity: Identity and name stability, and what it cannot show.
        chronology: Suspicious chronology.
        units: Raw-to-canonical mass fidelity.
        transitions: Equipment and federation transitions.
        finding_classes: What each class means.
        findings: Every reported count, classified.
    """

    identity: IdentityStabilityAudit
    chronology: ChronologyAudit
    units: UnitFidelityAudit
    transitions: TransitionAudit
    finding_classes: tuple[FindingClassMeaning, ...]
    findings: tuple[DiagnosticFinding, ...]

    def findings_of(self, finding_class: FindingClass) -> tuple[DiagnosticFinding, ...]:
        """Return the findings belonging to *finding_class*."""
        return tuple(item for item in self.findings if item.finding_class is finding_class)

    @property
    def failed_fidelity_findings(self) -> tuple[DiagnosticFinding, ...]:
        """Non-zero transformation-invariant failures: PSD defects, not source facts."""
        return tuple(
            item
            for item in self.findings_of(FindingClass.TRANSFORMATION_INVARIANT_FAILURE)
            if item.count
        )


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
        diagnostics: The four closure diagnostic families, each classified.
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
    diagnostics: DiagnosticsAudit
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
            "",
            "diagnostics by finding class",
        ]
        lines.extend(_finding_class_lines(self.diagnostics))
        return "\n".join(lines)


def _finding_class_lines(diagnostics: DiagnosticsAudit) -> list[str]:
    """Return the classified diagnostic counts, grouped by finding class.

    Grouped rather than flattened: a reader who sees a source anomaly and a transform
    defect in one undifferentiated list learns nothing about either. Within a class the
    counts stay in their declared order so two reports of the same corpus read alike.
    """
    lines: list[str] = []
    for meaning in diagnostics.finding_classes:
        found = diagnostics.findings_of(meaning.finding_class)
        lines.append(f"  {meaning.finding_class.value} ({len(found)} counts)")
        for item in found:
            count = "not measurable" if item.count is None else f"{item.count:,}"
            lines.append(f"    {count:>14}  {item.name}")
    return lines


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
        *_diagnostic_sections(audit.diagnostics),
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


def _diagnostic_sections(diagnostics: DiagnosticsAudit) -> list[str]:
    """Return the Markdown rendering of all four closure diagnostic families."""
    identity = diagnostics.identity
    chronology = diagnostics.chronology
    units = diagnostics.units
    transitions = diagnostics.transitions
    future = (
        f"{chronology.future_dated_competitions:,}"
        if chronology.future_dated_competitions is not None
        else "not measurable (no pinned snapshot date)"
    )
    future_meets = (
        f"{chronology.future_dated_meets:,}"
        if chronology.future_dated_meets is not None
        else "not measurable (no pinned snapshot date)"
    )
    lines: list[str] = [
        "## Diagnostics",
        "",
        "Findings are classified, because the four classes call for four different",
        "responses and a single undifferentiated warning bucket destroys that",
        "distinction.",
        "",
        "| finding class | what a count in this class is | treatment |",
        "| --- | --- | --- |",
    ]
    lines.extend(
        f"| `{meaning.finding_class.value}` | {meaning.meaning} | {meaning.treatment} |"
        for meaning in diagnostics.finding_classes
    )
    lines.extend(
        [
            "",
            "### Identity and name stability",
            "",
            f"- source identities: {identity.identities:,}",
            f"- names under more than one reported sex category:"
            f" {identity.names_under_multiple_sex_categories:,}",
            f"- disambiguated `#N` names: {identity.disambiguated_names:,}",
            f"- base-name collision groups: {identity.base_name_collision_groups:,}"
            f" (largest {identity.largest_base_name_collision_group:,})",
            f"- collision groups where the bare base name is also published:"
            f" {identity.base_names_also_published_unsuffixed:,}",
            f"- name keys mapped to several identities:"
            f" {identity.name_keys_mapping_to_several_identities:,}",
            f"- identities mapped to several name keys:"
            f" {identity.identities_mapping_to_several_name_keys:,}",
            "",
            _bucket_table(
                "Base-name collision group size", identity.base_name_group_size_distribution
            ),
            "",
            "**What this source cannot show:**",
            "",
        ]
    )
    for limitation in identity.limitations:
        marker = "identifiable" if limitation.identifiable else "**not identifiable**"
        lines.append(f"- {limitation.question}: {marker} - {limitation.reason}")
    lines.extend(
        [
            "",
            "### Suspicious chronology",
            "",
            f"- pinned snapshot date: `{chronology.pinned_snapshot_date or 'not reported'}`"
            f" (basis: {chronology.pinned_snapshot_date_basis})",
            f"- future-dated participations: {future}",
            f"- future-dated meets: {future_meets}",
            f"- meet identities spanning more than one source date:"
            f" {chronology.meets_spanning_multiple_dates:,}",
            f"- age observations: {chronology.age_observations:,} across"
            f" {chronology.identities_with_age_observations:,} identities;",
            f"  {chronology.identities_with_multiple_age_observations:,} identities have more"
            " than one",
            f"- identities with no birth year compatible with every age observation:"
            f" {chronology.identities_without_a_compatible_birth_year:,}",
            "",
            f"Age compatibility rule: {chronology.age_semantics}.",
            "",
            "### Unit consistency (raw-to-canonical mass fidelity)",
            "",
            f"The pinned source documents its masses in {units.declared_source_unit} and"
            " carries no per-row unit field, so this section asks whether the canonical"
            " value is the source value rather than whether a lifter looks plausible. No"
            " plausibility heuristic runs and no pounds are inferred.",
            "",
            "| check | rows checked | mismatches |",
            "| --- | --- | --- |",
        ]
    )
    unit_rows = (
        ("body mass (kg)", units.bodyweight_rows_checked, units.bodyweight_kg_mismatches),
        ("body-mass unit label", units.bodyweight_rows_checked, units.bodyweight_unit_mismatches),
        (
            "absent body mass staying absent",
            units.bodyweight_rows_checked,
            units.bodyweight_absent_became_zero,
        ),
        (
            "attempt load = abs(source)",
            units.attempt_rows_checked,
            units.attempt_load_kg_mismatches,
        ),
        (
            "attempt sign carries only the result",
            units.attempt_rows_checked,
            units.attempt_sign_result_mismatches,
        ),
        ("attempt unit label", units.attempt_rows_checked, units.attempt_unit_mismatches),
        (
            "attempt load strictly positive",
            units.attempt_rows_checked,
            units.attempt_loads_not_positive,
        ),
        (
            "reported value = magnitude of source",
            units.reported_rows_checked,
            units.reported_value_mismatches,
        ),
        (
            "reported mass unit label",
            units.reported_mass_rows_checked,
            units.reported_mass_unit_mismatches,
        ),
        (
            "Best3 semantics preserved",
            units.reported_best_rows_checked,
            units.reported_best_semantics_mismatches,
        ),
        (
            "TotalKg unit label",
            units.reported_totals_checked,
            units.reported_total_unit_mismatches,
        ),
        (
            "open-ended weight class stays open-ended",
            units.open_ended_weight_classes,
            units.open_ended_weight_classes_coerced,
        ),
        (
            "dimensionless scores not read as kg",
            units.scoring_rows_checked,
            units.scoring_rows_claiming_a_mass_unit,
        ),
    )
    lines.extend(
        f"| {label} | {checked:,} | {mismatches:,} |" for label, checked, mismatches in unit_rows
    )
    lines.extend(
        [
            f"| **all mass fidelity checks combined** | | **{units.mismatch_total:,}** |",
            "",
            "### Equipment and federation transitions",
            "",
            f"Computed at athlete-meet (`{transitions.grouping}`) and ordered by"
            f" `{transitions.ordering}`, so several event entries at one meet are never"
            " read as a longitudinal switch.",
            "",
            f"Equipment note: {transitions.equipment_reading}.",
            "",
            f"A declared unknown is not a state (`unknown_is_a_state ="
            f" {str(transitions.unknown_is_a_state).lower()}`): a meet the source left"
            " silent, and a meet that contradicts itself, are stepped over rather than"
            " treated as a change.",
            "",
            "| field | source column | athlete-meets | with a known state | intra-meet"
            " conflicts | transitions | identities with transitions |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    lines.extend(
        f"| `{field.field}` | `{field.source_column}` | {field.athlete_meets:,}"
        f" | {field.athlete_meets_with_a_known_state:,} |"
        f" {field.intra_meet_state_conflicts:,} | {field.total_transitions:,}"
        f" | {field.athletes_with_transitions:,} |"
        for field in transitions.fields
    )
    lines.append("")
    for field in transitions.fields:
        lines.extend(
            [
                f"**{field.field} transitions per identity**",
                "",
                _bucket_table(f"{field.field}", field.transitions_per_athlete),
                "",
            ]
        )
    lines.extend(
        [
            "### Classified diagnostic index",
            "",
            "| family | finding | class | count |",
            "| --- | --- | --- | --- |",
        ]
    )
    lines.extend(
        f"| `{item.family}` | `{item.name}` | `{item.finding_class.value}` | {count} |"
        for item in diagnostics.findings
        if (count := "not measurable" if item.count is None else f"{item.count:,}")
    )
    lines.append("")
    return lines


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
    try:
        connection.read_parquet(str(path)).create_view(table)
    except (duckdb.Error, OSError, ValueError) as error:
        # A file that exists but cannot be read is a finding, not a crash: the audit's
        # job is to report the state of the corpus, and an unreadable artifact is a state.
        msg = f"Cannot audit {table}: {path} is unreadable ({error})."
        raise AuditError(msg) from error


def _scalar(
    connection: duckdb.DuckDBPyConnection, sql: str, parameters: Sequence[str] | None = None
) -> int:
    """Return the single integer *sql* produces.

    Args:
        connection: An open pass.
        sql: One statement returning a single integer.
        parameters: Values bound through DuckDB's own parameter list rather than
            interpolated, so a value read out of the corpus can never become SQL.
    """
    row = connection.execute(sql, list(parameters) if parameters else None).fetchone()
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
    connection: duckdb.DuckDBPyConnection,
    sql: str,
    parameters: Sequence[str] | None = None,
    *,
    limit: str = _EXAMPLES_LIMIT,
) -> tuple[Mapping[str, object], ...]:
    """Return the first *limit* rows of *sql* as mappings keyed by column name."""
    cursor = connection.execute(sql, list(parameters) if parameters else None)
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
# closure diagnostics
# --------------------------------------------------------------------------

#: The source's ``#N`` disambiguator. Declared once so the base-name derivation and the
#: disambiguated count cannot disagree about what the suffix looks like.
_DISAMBIGUATOR_PATTERN: Final[str] = r"#\d+\s*$"


def _identity_audit(connection: duckdb.DuckDBPyConnection) -> IdentityStabilityAudit:
    """Return what the source identity key can and cannot support.

    The only honest answer to "did this lifter change name?" from this source is *that
    cannot be told*, and it is returned as a structured limitation rather than left as a
    missing number. Everything else counted here is either a genuine self-contradiction
    in the source or a measurement of how the source's own ``#N`` mechanism is used.
    """
    connection.execute(
        "CREATE OR REPLACE TEMP TABLE name_keys AS SELECT athlete_id, source_athlete_key,"
        f" regexp_matches(source_athlete_key, '{_DISAMBIGUATOR_PATTERN}') AS disambiguated,"
        f" regexp_replace(source_athlete_key, '{_DISAMBIGUATOR_PATTERN}', '') AS base_name"
        " FROM athlete_source_link WHERE source_athlete_key IS NOT NULL"
    )
    connection.execute(
        "CREATE OR REPLACE TEMP TABLE base_name_groups AS SELECT base_name,"
        " count(*) AS members,"
        " count(*) FILTER (WHERE disambiguated) AS disambiguated_members,"
        " count(*) FILTER (WHERE NOT disambiguated) AS unsuffixed_members"
        " FROM name_keys GROUP BY base_name HAVING count(*) > 1"
    )
    return IdentityStabilityAudit(
        identities=_scalar(connection, "SELECT count(*) FROM athlete"),
        names_under_multiple_sex_categories=_scalar(
            connection, "SELECT count(*) FROM athlete WHERE ambiguity_group_id IS NOT NULL"
        ),
        name_keys_mapping_to_several_identities=_scalar(
            connection,
            "SELECT count(*) FROM (SELECT source_athlete_key FROM name_keys"
            " GROUP BY source_athlete_key HAVING count(DISTINCT athlete_id) > 1)",
        ),
        identities_mapping_to_several_name_keys=_scalar(
            connection,
            "SELECT count(*) FROM (SELECT athlete_id FROM athlete_source_link"
            " GROUP BY athlete_id HAVING count(DISTINCT source_athlete_key) > 1)",
        ),
        disambiguated_names=_scalar(
            connection, "SELECT count(*) FROM name_keys WHERE disambiguated"
        ),
        base_name_collision_groups=_scalar(connection, "SELECT count(*) FROM base_name_groups"),
        largest_base_name_collision_group=_optional_int(
            connection, "SELECT max(members) FROM base_name_groups"
        )
        or 0,
        base_name_group_size_distribution=_bucket_counts(
            connection, table="base_name_groups", column="members", buckets=_BASE_GROUP_BUCKETS
        ),
        base_names_also_published_unsuffixed=_scalar(
            connection, "SELECT count(*) FROM base_name_groups WHERE unsuffixed_members > 0"
        ),
        base_name_examples=_examples(
            connection,
            "SELECT base_name, members, disambiguated_members, unsuffixed_members,"
            " list(source_athlete_key ORDER BY source_athlete_key) AS name_keys"
            " FROM base_name_groups JOIN name_keys USING (base_name)"
            " GROUP BY base_name, members, disambiguated_members, unsuffixed_members"
            " ORDER BY members DESC, base_name LIMIT 10",
        ),
        limitations=(
            DiagnosticLimitation(
                question=NAME_CHANGE_LIMITATION_QUESTION,
                identifiable=False,
                reason=NAME_CHANGE_LIMITATION_REASON,
            ),
        ),
    )


def _chronology_audit(
    connection: duckdb.DuckDBPyConnection,
    *,
    snapshot_date: str | None,
    snapshot_date_basis: str,
) -> ChronologyAudit:
    """Return the three source-grounded chronology checks.

    Args:
        connection: An open pass with the canonical tables registered.
        snapshot_date: The pinned snapshot date as ``YYYY-MM-DD``, or ``None`` when the
            source stated none. A future-dated comparison is impossible without it, and
            the report says so rather than comparing against the operator's clock.
        snapshot_date_basis: Which statement supplied the date, recorded beside it.
    """
    future: int | None = None
    future_meets: int | None = None
    examples: tuple[Mapping[str, object], ...] = ()
    if snapshot_date is not None:
        future = _scalar(
            connection,
            "SELECT count(*) FROM competition WHERE competition_date > CAST(? AS TIMESTAMPTZ)",
            [f"{snapshot_date} 00:00:00+00"],
        )
        future_meets = _scalar(
            connection,
            "SELECT count(*) FROM competition_meet WHERE meet_date > CAST(? AS TIMESTAMPTZ)",
            [f"{snapshot_date} 00:00:00+00"],
        )
        examples = _examples(
            connection,
            "SELECT competition_meet_id,"
            " strftime(competition_date, '%Y-%m-%dT%H:%M:%SZ') AS competition_date"
            " FROM competition WHERE competition_date > CAST(? AS TIMESTAMPTZ)"
            " ORDER BY competition_date DESC, competition_meet_id LIMIT 5",
            [f"{snapshot_date} 00:00:00+00"],
        )
    connection.execute(
        "CREATE OR REPLACE TEMP TABLE age_observations AS SELECT competition_id, athlete_id,"
        " competition_date, age_reported, age_precision FROM competition"
        " WHERE age_reported IS NOT NULL AND competition_date IS NOT NULL"
    )
    # ``floor`` rather than a cast: an approximate age is exactly ``n+0.5`` and the rule
    # below turns that into the single birth year it implies.
    connection.execute(
        "CREATE OR REPLACE TEMP TABLE age_pairs AS SELECT athlete_id, competition_id,"
        " year(competition_date) AS meet_year,"
        " year(competition_date) - CAST(floor(age_reported) AS BIGINT) AS earlier_birth_year,"
        " year(competition_date) - CAST(floor(age_reported) AS BIGINT) - 1 AS later_birth_year,"
        " age_precision = 'exact' AS admits_two_years FROM age_observations"
    )
    connection.execute(
        "CREATE OR REPLACE TEMP TABLE age_sets AS SELECT athlete_id, competition_id, meet_year,"
        " earlier_birth_year AS birth_year FROM age_pairs UNION ALL SELECT athlete_id,"
        " competition_id, meet_year, later_birth_year FROM age_pairs WHERE admits_two_years"
    )
    connection.execute(
        "CREATE OR REPLACE TEMP TABLE age_totals AS SELECT athlete_id, count(*) AS observations"
        " FROM age_observations GROUP BY athlete_id"
    )
    # A birth year survives when it is compatible with *every* observation of the identity,
    # so counting how many observations admit it and comparing against that identity's
    # total is the intersection, computed by counting rather than by materialising sets.
    connection.execute(
        "CREATE OR REPLACE TEMP TABLE age_survivors AS SELECT s.athlete_id, s.birth_year"
        " FROM age_sets s JOIN age_totals t USING (athlete_id)"
        " GROUP BY s.athlete_id, s.birth_year, t.observations"
        " HAVING count(*) = t.observations"
    )
    connection.execute(
        "CREATE OR REPLACE TEMP TABLE age_inconsistent AS SELECT t.athlete_id, t.observations"
        " FROM age_totals t"
        " LEFT JOIN (SELECT DISTINCT athlete_id FROM age_survivors) s USING (athlete_id)"
        " WHERE t.observations > 1 AND s.athlete_id IS NULL"
    )
    return ChronologyAudit(
        pinned_snapshot_date=snapshot_date,
        pinned_snapshot_date_basis=snapshot_date_basis,
        future_dated_competitions=future,
        future_dated_meets=future_meets,
        future_dated_examples=examples,
        meets_spanning_multiple_dates=_scalar(
            connection,
            "SELECT count(*) FROM (SELECT competition_meet_id FROM competition"
            " WHERE competition_date IS NOT NULL GROUP BY competition_meet_id"
            " HAVING count(DISTINCT competition_date) > 1)",
        ),
        meet_date_examples=_examples(
            connection,
            "SELECT competition_meet_id, count(DISTINCT competition_date) AS distinct_dates,"
            " min(competition_date) AS first_date, max(competition_date) AS last_date"
            " FROM competition WHERE competition_date IS NOT NULL GROUP BY competition_meet_id"
            " HAVING count(DISTINCT competition_date) > 1"
            " ORDER BY distinct_dates DESC, competition_meet_id LIMIT 5",
        ),
        age_observations=_scalar(connection, "SELECT count(*) FROM age_observations"),
        identities_with_age_observations=_scalar(connection, "SELECT count(*) FROM age_totals"),
        identities_with_multiple_age_observations=_scalar(
            connection, "SELECT count(*) FROM age_totals WHERE observations > 1"
        ),
        identities_without_a_compatible_birth_year=_scalar(
            connection, "SELECT count(*) FROM age_inconsistent"
        ),
        incompatible_identity_examples=_examples(
            connection,
            "SELECT i.athlete_id, a.source_athlete_key, i.observations FROM age_inconsistent i"
            " JOIN athlete_source_link a USING (athlete_id)"
            " ORDER BY i.observations DESC, i.athlete_id LIMIT 10",
        ),
        age_semantics=(
            "exact age n at meet year Y admits birth year Y-n or Y-n-1, depending on "
            "whether the birthday had passed; approximate age n+0.5 is the midpoint of "
            "that range and admits exactly Y-(n+1); an identity's observations are "
            "intersected and an empty intersection is reported"
        ),
    )


def _unit_audit(connection: duckdb.DuckDBPyConnection) -> UnitFidelityAudit:
    """Return the raw-to-canonical mass fidelity counts.

    Every counter named ``*_mismatches`` is expected to be zero. The tolerance exists
    only to absorb decimal representation of a copied float: a real unit error is off by
    a factor, not by the last bit, so nothing legitimate is hidden by it.
    """
    tolerance = repr(UNIT_TOLERANCE)
    mass_kinds = _sql_list(_MASS_RESULT_KINDS)
    best_kinds = _sql_list(_BEST_RESULT_KINDS)
    scoring_kinds = _sql_list(
        tuple(
            kind
            for kind, _column, _best in sorted(REPORTED_RESULT_SPECS)
            if kind not in _MASS_RESULT_KINDS
        )
    )
    bodyweight_kg_mismatches = _scalar(
        connection,
        "SELECT count(*) FROM competition WHERE bodyweight_raw IS NOT NULL"
        f" AND (bodyweight_kg IS NULL OR abs(bodyweight_kg - bodyweight_raw) > {tolerance})",
    )
    bodyweight_unit_mismatches = _scalar(
        connection,
        "SELECT count(*) FROM competition WHERE bodyweight_raw IS NOT NULL"
        " AND bodyweight_unit IS DISTINCT FROM 'kg'",
    )
    bodyweight_absent_became_zero = _scalar(
        connection,
        "SELECT count(*) FROM competition WHERE bodyweight_raw IS NULL AND bodyweight_kg = 0",
    )
    attempt_rows = _scalar(connection, "SELECT count(*) FROM competition_attempt")
    attempt_load_kg_mismatches = _scalar(
        connection,
        "SELECT count(*) FROM competition_attempt WHERE load_kg IS NULL"
        f" OR abs(load_kg - abs(source_attempt_raw)) > {tolerance}"
        f" OR load_raw IS NULL OR abs(load_kg - load_raw) > {tolerance}",
    )
    attempt_sign_result_mismatches = _scalar(
        connection,
        "SELECT count(*) FROM competition_attempt WHERE"
        " (source_attempt_raw > 0) <> (result = 'good_lift')",
    )
    attempt_unit_mismatches = _scalar(
        connection,
        "SELECT count(*) FROM competition_attempt WHERE load_unit IS DISTINCT FROM 'kg'",
    )
    attempt_loads_not_positive = _scalar(
        connection, "SELECT count(*) FROM competition_attempt WHERE load_kg <= 0"
    )
    reported_rows = _scalar(connection, "SELECT count(*) FROM competition_reported_result")
    reported_value_mismatches = _scalar(
        connection,
        "SELECT count(*) FROM competition_reported_result WHERE value IS NULL"
        " OR source_value_raw IS NULL"
        f" OR abs(abs(source_value_raw) - value) > {tolerance}",
    )
    reported_mass_rows = _scalar(
        connection,
        f"SELECT count(*) FROM competition_reported_result WHERE result_kind IN ({mass_kinds})",
    )
    reported_mass_unit_mismatches = _scalar(
        connection,
        f"SELECT count(*) FROM competition_reported_result WHERE result_kind IN ({mass_kinds})"
        " AND unit IS DISTINCT FROM 'kg'",
    )
    reported_best_rows = _scalar(
        connection,
        f"SELECT count(*) FROM competition_reported_result WHERE result_kind IN ({best_kinds})",
    )
    reported_best_semantics_mismatches = _scalar(
        connection,
        f"SELECT count(*) FROM competition_reported_result WHERE result_kind IN ({best_kinds})"
        " AND ((source_value_raw < 0) <> (reported_best_semantics = 'failed_attempt_only'))",
    )
    reported_totals = _scalar(
        connection,
        "SELECT count(*) FROM competition_reported_result WHERE result_kind ="
        f" '{_TOTAL_RESULT_KIND}'",
    )
    reported_total_unit_mismatches = _scalar(
        connection,
        "SELECT count(*) FROM competition_reported_result"
        f" WHERE result_kind = '{_TOTAL_RESULT_KIND}' AND unit IS DISTINCT FROM 'kg'",
    )
    open_ended = _scalar(
        connection,
        "SELECT count(*) FROM competition WHERE regexp_matches(weight_class_raw, "
        f"'{OPEN_ENDED_WEIGHT_CLASS_PATTERN}')",
    )
    # The label is persisted verbatim and no numeric weight-class column exists, so a
    # coercion could only appear as an open-ended label that lost its ``+``.
    open_ended_coerced = _scalar(
        connection,
        "SELECT count(*) FROM competition WHERE regexp_matches(weight_class_raw, "
        f"'{OPEN_ENDED_WEIGHT_CLASS_PATTERN}') AND NOT contains(weight_class_raw, '+')",
    )
    scoring_rows = _scalar(
        connection,
        f"SELECT count(*) FROM competition_reported_result WHERE result_kind IN ({scoring_kinds})",
    )
    scoring_rows_claiming_mass = _scalar(
        connection,
        f"SELECT count(*) FROM competition_reported_result WHERE result_kind IN ({scoring_kinds})"
        " AND unit = 'kg'",
    )
    return UnitFidelityAudit(
        declared_source_unit="kilograms",
        mass_source_fields=MASS_SOURCE_FIELDS,
        bodyweight_rows_checked=_scalar(
            connection, "SELECT count(*) FROM competition WHERE bodyweight_raw IS NOT NULL"
        ),
        bodyweight_kg_mismatches=bodyweight_kg_mismatches,
        bodyweight_unit_mismatches=bodyweight_unit_mismatches,
        bodyweight_absent_became_zero=bodyweight_absent_became_zero,
        attempt_rows_checked=attempt_rows,
        attempt_load_kg_mismatches=attempt_load_kg_mismatches,
        attempt_sign_result_mismatches=attempt_sign_result_mismatches,
        attempt_unit_mismatches=attempt_unit_mismatches,
        attempt_loads_not_positive=attempt_loads_not_positive,
        reported_rows_checked=reported_rows,
        reported_value_mismatches=reported_value_mismatches,
        reported_mass_rows_checked=reported_mass_rows,
        reported_mass_unit_mismatches=reported_mass_unit_mismatches,
        reported_best_rows_checked=reported_best_rows,
        reported_best_semantics_mismatches=reported_best_semantics_mismatches,
        reported_totals_checked=reported_totals,
        reported_total_unit_mismatches=reported_total_unit_mismatches,
        open_ended_weight_classes=open_ended,
        open_ended_weight_classes_coerced=open_ended_coerced,
        scoring_rows_checked=scoring_rows,
        scoring_rows_claiming_a_mass_unit=scoring_rows_claiming_mass,
        mismatch_total=(
            bodyweight_kg_mismatches
            + bodyweight_unit_mismatches
            + bodyweight_absent_became_zero
            + attempt_load_kg_mismatches
            + attempt_sign_result_mismatches
            + attempt_unit_mismatches
            + attempt_loads_not_positive
            + reported_value_mismatches
            + reported_mass_unit_mismatches
            + reported_best_semantics_mismatches
            + reported_total_unit_mismatches
            + open_ended_coerced
            + scoring_rows_claiming_mass
        ),
    )


#: The longitudinal state fields worth tracking, as ``(canonical column, source column)``.
#: Equipment is first because it is the field a reader most often mistakes for a gear
#: observation; federation and parent federation follow because a lifter's sanctioning
#: body is a fact about the meet, not about the person.
_TRANSITION_FIELDS: Final[tuple[tuple[str, str], ...]] = (
    ("equipment_class", "Equipment"),
    ("federation", "Federation"),
    ("sanctioning_body", "ParentFederation"),
)


def _transition_audit(connection: duckdb.DuckDBPyConnection) -> TransitionAudit:
    """Return per-athlete-meet equipment and federation transitions.

    One pass builds the athlete-meet state for every tracked field, and the ordering is
    resolved once. ``lag(... IGNORE NULLS`` is what makes the ordering honest: a meet the
    source left silent, and a meet that contradicts itself, both carry a null state and so
    are stepped over rather than treated as a change of state.
    """
    known: dict[str, str] = {
        column: (
            f"CASE WHEN {column} <> '{_UNKNOWN_EQUIPMENT_CLASS}' THEN {column} END"
            if column == "equipment_class"
            else column
        )
        for column, _source in _TRANSITION_FIELDS
    }
    values = ", ".join(
        f"count(DISTINCT CASE WHEN {known[column]} IS NOT NULL THEN {column} END)"
        f" AS {column}_values"
        for column, _source in _TRANSITION_FIELDS
    )
    states = ", ".join(
        f"min(CASE WHEN {known[column]} IS NOT NULL THEN {column} END) AS {column}_candidate"
        for column, _source in _TRANSITION_FIELDS
    )
    connection.execute(
        "CREATE OR REPLACE TEMP TABLE athlete_meet_state AS SELECT athlete_id,"
        " competition_meet_id, min(competition_date) AS competition_date,"
        f" {values}, {states} FROM competition"
        " GROUP BY athlete_id, competition_meet_id"
    )
    resolved = ", ".join(
        f"CASE WHEN {column}_values = 1 THEN {column}_candidate END AS {column}_state"
        for column, _source in _TRANSITION_FIELDS
    )
    previous = ", ".join(
        f"lag({column}_state IGNORE NULLS) OVER w AS {column}_previous"
        for column, _source in _TRANSITION_FIELDS
    )
    connection.execute(
        f"CREATE OR REPLACE TEMP TABLE athlete_meet_sequence AS SELECT *, {previous}"
        " FROM (SELECT *, " + resolved + " FROM athlete_meet_state)"
        " WINDOW w AS (PARTITION BY athlete_id"
        " ORDER BY competition_date NULLS LAST, competition_meet_id)"
    )
    athlete_meets = _scalar(connection, "SELECT count(*) FROM athlete_meet_state")
    entries: list[TransitionFieldAudit] = []
    for column, source in _TRANSITION_FIELDS:
        connection.execute(
            "CREATE OR REPLACE TEMP TABLE athlete_transitions AS SELECT athlete_id,"
            f" count(*) FILTER (WHERE {column}_state IS NOT NULL"
            f" AND {column}_previous IS NOT NULL"
            f" AND {column}_state <> {column}_previous) AS transitions"
            " FROM athlete_meet_sequence GROUP BY athlete_id"
        )
        entries.append(
            TransitionFieldAudit(
                field=column,
                source_column=source,
                athlete_meets=athlete_meets,
                athlete_meets_with_a_known_state=_scalar(
                    connection,
                    f"SELECT count(*) FROM athlete_meet_sequence WHERE {column}_values = 1",
                ),
                intra_meet_state_conflicts=_scalar(
                    connection,
                    f"SELECT count(*) FROM athlete_meet_state WHERE {column}_values > 1",
                ),
                total_transitions=_scalar(
                    connection, "SELECT coalesce(sum(transitions), 0) FROM athlete_transitions"
                ),
                athletes_with_transitions=_scalar(
                    connection, "SELECT count(*) FROM athlete_transitions WHERE transitions > 0"
                ),
                transitions_per_athlete=_bucket_counts(
                    connection,
                    table="athlete_transitions",
                    column="transitions",
                    buckets=_TRANSITION_BUCKETS,
                ),
            )
        )
    return TransitionAudit(fields=tuple(entries))


_FINDING_CLASS_MEANINGS: Final[tuple[FindingClassMeaning, ...]] = (
    FindingClassMeaning(
        finding_class=FindingClass.SOURCE_ANOMALY,
        meaning=(
            "the pinned source published something unusual, observed in the corpus and "
            "reported verbatim"
        ),
        treatment=(
            "reported, never repaired: repairing it would make the corpus unable to be "
            "distinguished from one built over a source with no anomalies"
        ),
    ),
    FindingClassMeaning(
        finding_class=FindingClass.TRANSFORMATION_INVARIANT_FAILURE,
        meaning="the transform broke a contract it declared",
        treatment=(
            "expected to be zero; a non-zero count is a defect in PSD and not a fact "
            "about the source"
        ),
    ),
    FindingClassMeaning(
        finding_class=FindingClass.DESCRIPTIVE_LONGITUDINAL_TRANSITION,
        meaning=(
            "a value changed between two meets in one lifter's competition history, which "
            "is ordinary rather than wrong"
        ),
        treatment="reported as a count; never acted on, flagged, or resolved",
    ),
    FindingClassMeaning(
        finding_class=FindingClass.SOURCE_LIMITATION,
        meaning=("a question this source cannot answer at all, so no count of it is possible"),
        treatment=(
            "stated as an explicit limitation; the silence must never be read as a clean result"
        ),
    ),
)


def _diagnostic_findings(
    identity: IdentityStabilityAudit,
    chronology: ChronologyAudit,
    units: UnitFidelityAudit,
    transitions: TransitionAudit,
) -> tuple[DiagnosticFinding, ...]:
    """Return the classified index over the four diagnostic sections.

    Built from the sections themselves rather than declared alongside them, so a new
    count cannot be added to a section and then be missing from the index a consumer
    filters on.

    Returns:
        One :class:`DiagnosticFinding` per reported count, families in order.
    """
    findings: list[DiagnosticFinding] = []

    def add(
        family: str,
        finding_class: FindingClass,
        name: str,
        count: int | None,
        statement: str,
    ) -> None:
        findings.append(
            DiagnosticFinding(
                family=family,
                name=name,
                finding_class=finding_class,
                count=count,
                statement=statement,
            )
        )

    identity_class = FindingClass.SOURCE_LIMITATION
    for name, count, statement in (
        (
            "names_under_multiple_sex_categories",
            identity.names_under_multiple_sex_categories,
            "exact source Name keys the source publishes under more than one reported sex category",
        ),
        (
            "name_keys_mapping_to_several_identities",
            identity.name_keys_mapping_to_several_identities,
            "source name keys that produced more than one canonical identity",
        ),
        (
            "identities_mapping_to_several_name_keys",
            identity.identities_mapping_to_several_name_keys,
            "canonical identities carrying more than one source name key",
        ),
        (
            "disambiguated_names",
            identity.disambiguated_names,
            "source name keys carrying the source's trailing #N disambiguator",
        ),
        (
            "base_name_collision_groups",
            identity.base_name_collision_groups,
            "base names shared by more than one source name key",
        ),
        (
            "base_names_also_published_unsuffixed",
            identity.base_names_also_published_unsuffixed,
            "collision groups where the bare base name is itself a separate published identity",
        ),
    ):
        add("identity", identity_class, name, count, statement)
    for limitation in identity.limitations:
        add(
            "identity",
            identity_class,
            f"limitation:{limitation.question}",
            0,
            f"not identifiable: {limitation.reason}",
        )

    for name, count, statement in (
        (
            "future_dated_competitions",
            chronology.future_dated_competitions,
            "participations dated after the pinned snapshot date",
        ),
        (
            "future_dated_meets",
            chronology.future_dated_meets,
            "meet identities dated after the pinned snapshot date",
        ),
        (
            "meets_spanning_multiple_dates",
            chronology.meets_spanning_multiple_dates,
            "meet identities carrying more than one distinct source date",
        ),
        (
            "identities_without_a_compatible_birth_year",
            chronology.identities_without_a_compatible_birth_year,
            "source identities whose age observations admit no common birth year",
        ),
    ):
        add("chronology", FindingClass.SOURCE_ANOMALY, name, count, statement)

    transition_class = FindingClass.DESCRIPTIVE_LONGITUDINAL_TRANSITION
    for field in transitions.fields:
        add(
            "transitions",
            transition_class,
            f"{field.field}_intra_meet_state_conflicts",
            field.intra_meet_state_conflicts,
            f"athlete-meets where the source stated more than one {field.source_column}",
        )
        add(
            "transitions",
            transition_class,
            f"{field.field}_total_transitions",
            field.total_transitions,
            f"changes between consecutive known athlete-meet {field.source_column} values",
        )
        add(
            "transitions",
            transition_class,
            f"{field.field}_athletes_with_transitions",
            field.athletes_with_transitions,
            f"source identities with at least one {field.source_column} change",
        )

    fidelity_class = FindingClass.TRANSFORMATION_INVARIANT_FAILURE
    for name in _UNIT_MISMATCH_COUNTERS:
        add(
            "units",
            fidelity_class,
            name,
            getattr(units, name),
            f"raw-to-canonical mass fidelity: {name}",
        )
    add(
        "units",
        fidelity_class,
        "mass_fidelity_mismatches",
        units.mismatch_total,
        "every raw-to-canonical mass fidelity check combined, which a correct transform"
        " reports as zero",
    )
    return tuple(findings)


#: The unit counters that must be zero for a correct transform, named once so the audit
#: computes them, totals them, and reports them through one list.
_UNIT_MISMATCH_COUNTERS: Final[tuple[str, ...]] = (
    "bodyweight_kg_mismatches",
    "bodyweight_unit_mismatches",
    "bodyweight_absent_became_zero",
    "attempt_load_kg_mismatches",
    "attempt_sign_result_mismatches",
    "attempt_unit_mismatches",
    "attempt_loads_not_positive",
    "reported_value_mismatches",
    "reported_mass_unit_mismatches",
    "reported_best_semantics_mismatches",
    "reported_total_unit_mismatches",
    "open_ended_weight_classes_coerced",
    "scoring_rows_claiming_a_mass_unit",
)


def _diagnostics_audit(
    connection: duckdb.DuckDBPyConnection, *, snapshot: OpenPowerliftingSnapshot | None
) -> DiagnosticsAudit:
    """Return all four closure diagnostic families and the classification index."""
    identity = _identity_audit(connection)
    snapshot_date, snapshot_date_basis = _snapshot_date(snapshot)
    chronology = _chronology_audit(
        connection,
        snapshot_date=snapshot_date,
        snapshot_date_basis=snapshot_date_basis,
    )
    units = _unit_audit(connection)
    transitions = _transition_audit(connection)
    return DiagnosticsAudit(
        identity=identity,
        chronology=chronology,
        units=units,
        transitions=transitions,
        finding_classes=_FINDING_CLASS_MEANINGS,
        findings=_diagnostic_findings(identity, chronology, units, transitions),
    )


def _snapshot_date(snapshot: OpenPowerliftingSnapshot | None) -> tuple[str | None, str]:
    """Return the snapshot date to measure chronology against, and which statement gave it.

    The service page's own statement is preferred over the archive member name because it
    describes the snapshot the service considers current; both are kept by the snapshot
    model and either may be absent, and an absent date means the future-dated check
    cannot run rather than that it found nothing.
    """
    if snapshot is None:
        return None, SNAPSHOT_DATE_BASIS_UNAVAILABLE
    if snapshot.service.updated_date is not None:
        return snapshot.service.updated_date, SNAPSHOT_DATE_BASIS_SERVICE
    if snapshot.service.archive_declared_date is not None:
        return snapshot.service.archive_declared_date, SNAPSHOT_DATE_BASIS_ARCHIVE
    return None, SNAPSHOT_DATE_BASIS_UNAVAILABLE


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


def corpus_expansion_invariants(
    dataset_dir: Path | str,
    *,
    data_root: Path | None = None,
    archive_sha256: str | None = None,
    memory_limit: str = DEFAULT_MEMORY_LIMIT,
    threads: int = DEFAULT_THREADS,
) -> tuple[Invariant, ...]:
    """Re-derive :func:`expansion_invariants` against one persisted corpus.

    A verification step, not a report: it opens the corpus, checks the invariants, and
    writes nothing. The full audit writes a document and answers far more questions, and a
    verification gate that had to produce that document first would be both slower and
    harder to reason about.

    Args:
        dataset_dir: The built corpus, relative to the data root.
        data_root: External PSD data root.
        archive_sha256: Unused here; accepted so a caller can pass one request's identity
            through unchanged.
        memory_limit: DuckDB memory ceiling.
        threads: DuckDB worker threads.

    Returns:
        One invariant per claim, each carrying what the corpus actually holds.

    Raises:
        AuditError: The corpus does not declare the canonical tables the checks need.
    """
    request = AuditRequest(
        dataset_dir=Path(dataset_dir),
        data_root=data_root,
        archive_sha256=archive_sha256,
        memory_limit=memory_limit,
        threads=threads,
    )
    _manifest, directory = _dataset_paths(request)
    connection = _open(request)
    try:
        for table in corpus_table_names():
            _register(connection, table, directory / "tables" / f"{table}.parquet")
        return expansion_invariants(connection)
    finally:
        connection.close()


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
        diagnostics = _diagnostics_audit(connection, snapshot=snapshot)
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
        diagnostics=diagnostics,
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
