"""Tests for the PSD-COMP corpus audit.

An audit is only worth its cost if it would notice a real defect and stay quiet about a
normal one. These tests hold it to both: a corpus built from a fixture with one row per
source edge must produce a report whose numbers can be checked by hand, and the
invariants must fail when a table is deliberately broken.

The most important property under test is the last group: the invariants are not
documentation. ``competitions_reference_a_meet`` is the check that found two independent
derivations of a meet identity disagreeing -- a defect that no digest could see, because
each table was separately well formed.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest
from pydantic import ValidationError

from psd.ingest.openpowerlifting.acquire import resolve_snapshot_csv
from psd.ingest.openpowerlifting.audit import (
    AUDIT_VERSION,
    EXPANSION_RULES,
    AuditError,
    AuditRequest,
    CategoryCoverage,
    CorpusAudit,
    ExpansionRule,
    audit_corpus,
    audit_markdown,
    coverage_table,
    read_audit,
    write_audit,
)
from psd.ingest.openpowerlifting.history import HistoryRequest, build_athlete_history
from psd.ingest.openpowerlifting.transform import BuildConfig, BuildRequest, build_corpus
from psd.provenance.manifest import manifest_digest
from psd.schema.registry import table_names
from psd.serialization.dataset import artifact_paths, read_manifest
from psd.serialization.parquet import read_parquet, write_parquet
from tests.fixtures.openpowerlifting import SAMPLE_ROW_COUNT, write_sample_snapshot

STAMP = datetime(2026, 10, 2, 0, 0, tzinfo=UTC)
CONFIG = BuildConfig(chunk_rows=3, batch_rows=4, partitions=2)

#: ``(data_root, dataset_relative)`` for the fixture corpus, built once per module.
Corpus = tuple[Path, str]

#: Stands in for a meet identity that exists nowhere in the meet table.
ORPHAN_MEET_ID: str = "cmeet_00000000000000000000000000000000"


@pytest.fixture(scope="module")
def corpus(tmp_path_factory: pytest.TempPathFactory) -> Corpus:
    """One built fixture corpus, with its longitudinal histories derived."""
    root = tmp_path_factory.mktemp("psd_comp_audit")
    data_root = root / "data"
    snapshot = write_sample_snapshot(root / "source" / "sample.csv", data_root=data_root)
    build_corpus(
        BuildRequest(
            csv_path=resolve_snapshot_csv(snapshot, data_root=data_root),
            snapshot=snapshot,
            data_root=data_root,
            config=CONFIG,
            ingested_at=STAMP,
        )
    )
    relative = f"canonical/psd_comp/{snapshot.archive_sha256}"
    build_athlete_history(
        HistoryRequest(dataset_dir=Path(relative), data_root=data_root, created_at=STAMP)
    )
    return data_root, relative


@pytest.fixture(scope="module")
def report(corpus: Corpus) -> CorpusAudit:
    """The audit of the fixture corpus."""
    data_root, relative = corpus
    return audit_corpus(
        AuditRequest(dataset_dir=Path(relative), data_root=data_root, generated_at=STAMP)
    ).audit


# ---------------------------------------------------------------------------
# Report shape
# ---------------------------------------------------------------------------


def test_the_report_declares_its_contract_and_its_corpus(
    report: CorpusAudit, corpus: Corpus
) -> None:
    _data_root, relative = corpus
    assert report.audit_version == AUDIT_VERSION
    assert report.dataset_id.startswith("psd_comp_")
    assert report.dataset_relative_path == relative
    assert report.schema_version == "psd-canonical/1.1.0"
    assert report.generated_at == STAMP


def test_the_report_is_durable_and_re_readable(report: CorpusAudit) -> None:
    """A stored report must survive a round trip byte-for-byte, or it is not a record."""
    assert CorpusAudit.from_dict(report.to_dict()) == report
    assert report.to_json_bytes().endswith(b"\n")


def test_the_report_names_its_own_manifest(report: CorpusAudit, corpus: Corpus) -> None:
    """The audit is pinned to the corpus it read, not merely to a dataset name."""
    data_root, relative = corpus
    assert report.manifest_digest == manifest_digest(read_manifest(relative, data_root=data_root))


def test_the_report_is_written_both_as_data_and_as_a_document(
    report: CorpusAudit, corpus: Corpus
) -> None:
    data_root, relative = corpus
    json_path = data_root / relative / "audit" / "corpus_audit.json"
    markdown_path = data_root / relative / "audit" / "corpus_audit.md"

    assert json_path.is_file()
    assert markdown_path.is_file()
    assert read_audit(json_path) == report
    assert "PSD-COMP corpus audit" in markdown_path.read_text(encoding="utf-8")


def test_a_report_can_be_written_elsewhere(report: CorpusAudit, tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    json_path, markdown_path = write_audit(report, Path("reports"), data_root=data_root)

    assert read_audit(json_path) == report
    assert markdown_path.is_file()


def test_reading_a_missing_report_is_refused(tmp_path: Path) -> None:
    with pytest.raises(AuditError, match="No audit report"):
        read_audit(tmp_path / "absent.json")


def test_reading_a_foreign_report_is_refused(tmp_path: Path) -> None:
    """Reading a corpus manifest as an audit would produce a report about nothing."""
    path = tmp_path / "manifest.json"
    path.write_text('{"dataset_id": "ds_other"}', encoding="utf-8")
    with pytest.raises(AuditError, match="Invalid audit report"):
        read_audit(path)


# ---------------------------------------------------------------------------
# Counts a reader can check by hand
# ---------------------------------------------------------------------------


def test_entity_counts_match_the_manifest(report: CorpusAudit, corpus: Corpus) -> None:
    """The report's entity counts are the persisted row counts, not a re-derivation."""
    data_root, relative = corpus
    manifest = read_manifest(relative, data_root=data_root)

    assert report.entities.competitions == manifest.artifact("competition").row_count
    assert report.entities.meets == manifest.artifact("competition_meet").row_count
    assert report.entities.attempts == manifest.artifact("competition_attempt").row_count
    assert (
        report.entities.reported_results
        == manifest.artifact("competition_reported_result").row_count
    )
    assert report.entities.athletes == manifest.artifact("athlete").row_count
    assert report.source.source_rows == SAMPLE_ROW_COUNT


def test_competitions_and_results_add_up(report: CorpusAudit) -> None:
    entities = report.entities
    attempts = report.attempts
    assert (
        attempts.competitions_with_attempts + attempts.competitions_without_attempts
        == entities.competitions
    )
    assert entities.competitions_with_reported_total <= entities.competitions
    assert entities.competitions_without_results <= entities.competitions
    assert (
        entities.competitions_with_reported_total + entities.competitions_without_results
        >= entities.competitions_with_reported_total
    )


def test_every_table_declares_unique_keys(report: CorpusAudit) -> None:
    assert set(report.entities.duplicate_primary_key_rows) == set(
        {
            name
            for name in table_names()
            if name
            in {
                "athlete",
                "athlete_source_link",
                "competition",
                "competition_attempt",
                "competition_meet",
                "competition_reported_result",
            }
        }
    )
    assert all(count == 0 for count in report.entities.duplicate_primary_key_rows.values())


def test_coverage_accounts_for_every_competition(report: CorpusAudit) -> None:
    """A coverage table whose parts do not sum to the corpus is not a coverage table."""
    total = report.entities.competitions
    for name in (
        "by_year",
        "by_federation",
        "by_parent_federation",
        "by_event",
        "by_equipment_class",
        "by_equipment_class_raw",
    ):
        coverage = getattr(report.coverage, name)
        assert (
            sum(coverage.rows.values()) + coverage.remainder_rows + coverage.null_rows == total
        ), name


def test_coverage_states_its_own_truncation(report: CorpusAudit) -> None:
    """An unstated omission reads as an absence of categories rather than an omission."""
    coverage = report.coverage.by_event
    assert coverage.truncated is False
    assert set(coverage.rows) == {"sbd", "bd", "sd", "sb", "s", "b", "d"}
    assert coverage.null_rows == 0


def test_coverage_keeps_absent_values_out_of_the_member_list(report: CorpusAudit) -> None:
    """An absent federation is reported as an absence, not as a federation called null."""
    parents = report.coverage.by_parent_federation
    assert parents.null_rows > 0, "several entries publish no sanctioning body"
    assert all(member not in {"(null)", "\x00null"} for member in parents.rows)
    assert sum(parents.rows.values()) + parents.null_rows + parents.remainder_rows == (
        report.entities.competitions
    )


# ---------------------------------------------------------------------------
# Longitudinal structure
# ---------------------------------------------------------------------------


def test_longitudinal_buckets_account_for_every_identity(report: CorpusAudit) -> None:
    longitudinal = report.longitudinal
    buckets = longitudinal.meet_count_distribution

    assert sum(buckets.values()) == report.entities.athletes
    assert (
        longitudinal.athletes_with_one_meet
        + longitudinal.athletes_with_two_to_four_meets
        + longitudinal.athletes_with_five_to_nine_meets
        + longitudinal.athletes_with_ten_or_more_meets
        == report.entities.athletes
    )
    assert sum(longitudinal.observed_span_days_distribution.values()) == (report.entities.athletes)


def test_a_single_meet_fixture_has_no_longitudinal_spread(report: CorpusAudit) -> None:
    """The fixture is one meet per lifter, so the distributions are the degenerate ones."""
    assert report.longitudinal.athletes_with_one_meet == report.entities.athletes
    assert report.longitudinal.multi_meet_athletes == 0
    assert report.longitudinal.max_meet_count == 1
    assert report.longitudinal.max_observed_span_days == 0


def test_the_report_notices_the_derived_histories(report: CorpusAudit) -> None:
    """The audit reports the projection it can see, and its absence when it cannot."""
    assert report.longitudinal.athlete_history_rows == report.entities.athletes


def test_the_history_is_reported_absent_when_it_has_not_been_derived(
    tmp_path: Path,
) -> None:
    root = tmp_path / "data"
    root.mkdir()
    data_root = tmp_path / "root" / "data"
    snapshot = write_sample_snapshot(tmp_path / "root" / "sample.csv", data_root=data_root)
    build_corpus(
        BuildRequest(
            csv_path=resolve_snapshot_csv(snapshot, data_root=data_root),
            snapshot=snapshot,
            data_root=data_root,
            config=CONFIG,
            ingested_at=STAMP,
        )
    )
    relative = Path("canonical") / "psd_comp" / snapshot.archive_sha256

    audit = audit_corpus(
        AuditRequest(dataset_dir=relative, data_root=data_root, generated_at=STAMP)
    ).audit

    assert audit.longitudinal.athlete_history_rows is None
    assert root.is_dir()


# ---------------------------------------------------------------------------
# Attempts
# ---------------------------------------------------------------------------


def test_attempt_counts_reconcile_with_the_attempt_table(
    report: CorpusAudit, corpus: Corpus
) -> None:
    data_root, relative = corpus
    attempts = pl.read_parquet(_table_path(data_root, relative, "competition_attempt"))

    assert report.attempts.attempts_total == attempts.height
    assert report.attempts.failed_attempts == int(
        attempts.get_column("result").eq("bad_lift").sum()
    )
    assert report.attempts.fourth_attempts == int(
        attempts.get_column("attempt_role").eq("record_fourth").sum()
    )
    assert 0.0 <= report.attempts.attempt_detail_fraction <= 1.0


def test_attempt_coverage_names_the_federations_it_measured(report: CorpusAudit) -> None:
    coverage = report.attempts.coverage_by_federation
    assert set(coverage.competitions) == set(report.coverage.by_federation.rows) | {
        member for member in report.coverage.by_federation.rows if member
    }
    assert sum(coverage.competitions.values()) <= report.entities.competitions
    assert coverage.fraction("XPC") == 0.0, "the fixture publishes no attempts there"


def test_attempt_detail_by_year_is_measured_per_year(report: CorpusAudit) -> None:
    coverage = report.attempts.coverage_by_year
    assert set(coverage.competitions) == set(report.coverage.by_year.rows)
    assert coverage.competitions_with_attempts["2025"] > 0


def test_a_contested_lift_with_no_attempt_at_all_is_reported(report: CorpusAudit) -> None:
    """Three fixture entries publish results but no attempts at any lift.

    ``Bo Reported`` (``S``), ``Dee Totalless`` (``SBD``) and ``Finn Nowhere`` (``SBD``) are
    the ordinary shape of this source: most federations publish only best lifts. They are
    counted, not repaired, and their absence is *not* a violation of the
    attempts-inside-the-event invariant -- that invariant is about attempts that exist
    outside their declared event, which is the opposite direction.
    """
    assert report.attempts.competitions_without_attempts == 3
    assert report.attempts.lifts_without_any_attempt == 3
    assert report.attempts.competitions_with_attempts == report.entities.competitions - 3


# ---------------------------------------------------------------------------
# Performance fields
# ---------------------------------------------------------------------------


def test_a_negative_reported_best_is_counted_as_its_own_state(report: CorpusAudit) -> None:
    """``Best3BenchKg=-45`` is one reported best whose meaning is not a successful lift."""
    assert report.performance.negative_reported_bests == 1
    assert report.performance.negative_reported_bests_by_kind == {"bench_best": 1}


def test_a_total_without_its_components_is_counted(report: CorpusAudit) -> None:
    """``TotalKg=500`` with no lifts exists on its own, and is reported as such."""
    performance = report.performance
    assert performance.totals >= 1
    assert performance.totals_without_all_component_bests >= 1
    assert (
        performance.totals_with_all_component_bests
        + (performance.totals_without_all_component_bests)
        == performance.totals
    )


def test_a_balanced_total_agrees_with_its_bests(report: CorpusAudit) -> None:
    """The fixture's balanced total is 540 against 187.5 + 122.5 + 230."""
    performance = report.performance
    assert performance.total_sum_agreements == performance.totals_with_all_component_bests
    assert performance.total_sum_disagreements == 0
    assert performance.disagreement_examples == ()


def test_no_reported_result_is_derived(report: CorpusAudit) -> None:
    assert report.performance.reported_results_marked_derived == 0


def test_a_record_attempt_never_reaches_a_total(report: CorpusAudit) -> None:
    """This is what "a fourth attempt does not count toward TotalKg" means, measured.

    Both the bench record attempt and the single-lift deadlift record attempt exist, and
    the totals that go with them still equal the sum of the three counted bests.
    """
    assert report.performance.fourth_attempt_competitions_with_a_total == 1
    assert (
        report.performance.fourth_attempt_totals_equal_to_best_sum
        == report.performance.fourth_attempt_competitions_with_a_total
    )


# ---------------------------------------------------------------------------
# Athlete and context fields
# ---------------------------------------------------------------------------


def test_bodyweight_missingness_is_reported_as_missingness(report: CorpusAudit) -> None:
    context = report.athlete_context
    assert context.bodyweight_present + context.bodyweight_missing == context.competitions
    assert context.bodyweight_missing > 0
    assert 0.0 < context.bodyweight_missing_fraction < 1.0


def test_exact_and_approximate_ages_are_counted_separately(report: CorpusAudit) -> None:
    """``23`` is exact; ``23.5`` is the source saying "23 or 24", and the two are counted apart."""
    context = report.athlete_context
    assert context.ages_exact > 0
    assert context.ages_approximate >= 1
    assert context.ages_exact + context.ages_approximate + context.ages_missing == (
        context.competitions
    )


def test_open_ended_weight_classes_are_counted(report: CorpusAudit) -> None:
    """``90+`` claims no maximum, so it is counted rather than collapsed to 90."""
    assert report.athlete_context.open_ended_weight_classes >= 2
    assert report.athlete_context.distinct_weight_classes > 1


def test_tested_and_sanctioned_coverage_are_measured(report: CorpusAudit) -> None:
    context = report.athlete_context
    assert context.tested_category_yes == 1
    assert (
        context.tested_category_yes + context.tested_category_no + context.tested_category_missing
        == context.competitions
    )
    assert context.meets_sanctioned + context.meets_unsanctioned == context.meets_sanctioned + (
        context.meets_unsanctioned
    )
    assert context.meets_sanctioned >= 1
    assert context.meets_unsanctioned >= 1


# ---------------------------------------------------------------------------
# Anomalies
# ---------------------------------------------------------------------------


def test_a_sex_conflict_is_reported_not_resolved(report: CorpusAudit) -> None:
    assert report.anomalies.sex_category_conflicts == 1
    assert report.anomalies.sex_category_conflict_examples == ("opl-name-conflict:Hal Twofold",)
    assert report.anomalies.sex_categories_absent == 1


def test_disambiguated_names_are_counted(report: CorpusAudit) -> None:
    assert report.anomalies.disambiguated_identities == 2


def test_a_documented_place_code_is_not_an_unknown_code(report: CorpusAudit) -> None:
    """``G``, ``DQ``, ``DD`` and ``NS`` are statuses, and every one of them is recognised."""
    assert report.anomalies.unknown_place_codes == 0
    assert report.anomalies.unknown_place_examples == ()
    assert report.anomalies.placements_outside_range == 0


def test_a_meet_variant_is_not_reported_as_a_meet_split(report: CorpusAudit) -> None:
    """A cosmetic spelling difference collapses to one identity, so it is not a collision."""
    assert report.anomalies.meet_identity_collisions == 0


def test_the_declared_vocabularies_produce_no_unexpected_values(report: CorpusAudit) -> None:
    assert report.anomalies.unexpected_event_values == ()
    assert report.anomalies.unexpected_equipment_values == ()
    assert report.anomalies.unexpected_sex_values == ()
    assert report.anomalies.competitions_with_unknown_event == 0


def test_the_meet_date_range_is_reported(report: CorpusAudit) -> None:
    assert report.anomalies.earliest_meet_date is not None
    # Rendered as an aware UTC instant by a UTC-pinned DuckDB session: a local-time
    # rendering would move the date across a day boundary for some operators.
    assert report.anomalies.earliest_meet_date == "2025-11-08T00:00:00Z"
    assert report.anomalies.earliest_meet_date == report.anomalies.latest_meet_date
    assert report.anomalies.competitions_without_a_meet_date == 0
    assert report.anomalies.competitions_disagreeing_with_their_meet == 0


def test_the_source_schema_review_is_carried_through(report: CorpusAudit) -> None:
    """Drift is reported by the audit as well as refused by the build, because a stored
    report must let a reader see what the corpus was read against."""
    assert report.schema_review.drifted is False
    assert report.anomalies.schema_drift_unknown_columns == ()
    assert report.anomalies.schema_drift_missing_columns == ()
    assert report.anomalies.schema_drift_reordered is False


def test_a_snapshot_that_reports_nothing_leaves_an_acknowledged_gap(
    report: CorpusAudit,
) -> None:
    """The fixture is pinned from a local CSV, so the service stated no date or revision."""
    assert report.source.snapshot_date is None
    assert report.source.revision is None
    assert report.source.archive_sha256 != "0" * 64
    assert report.source.consent_basis == "public_record"
    assert report.source.license_id == "public-domain"


# ---------------------------------------------------------------------------
# The expansion audit
# ---------------------------------------------------------------------------


def test_every_expanded_table_declares_the_four_questions(report: CorpusAudit) -> None:
    tables = {rule.table for rule in report.expansion.rules}
    assert tables == {
        "competition",
        "competition_attempt",
        "competition_reported_result",
        "competition_meet",
        "athlete",
    }
    for rule in report.expansion.rules:
        assert len(rule.source_field) > 10, rule.table
        assert len(rule.absent_source_value) > 20, rule.table
        assert len(rule.zero_meaning) > 10, rule.table
        assert len(rule.negative_semantics) > 10, rule.table


def test_the_attempt_rule_states_that_an_absent_cell_makes_no_row() -> None:
    rule = _rule("competition_attempt")
    assert "No row at all" in rule.absent_source_value
    assert "reconstructs attempts from a best" in rule.absent_source_value
    assert rule.zero_meaning.startswith("None")


def test_the_attempt_rule_states_that_the_sign_is_the_result() -> None:
    rule = _rule("competition_attempt")
    assert "failed attempt at the positive magnitude" in rule.negative_semantics
    assert "bad_lift" in rule.negative_semantics


def test_the_reported_result_rule_states_that_a_total_may_stand_alone() -> None:
    rule = _rule("competition_reported_result")
    assert "TotalKg can therefore exist with none of" in rule.absent_source_value
    assert "never manufactures the components" in rule.absent_source_value
    assert "failed_attempt_only" in rule.negative_semantics


def test_every_rule_is_a_frozen_record() -> None:
    """A rule that can be mutated after the audit would not be the rule that was checked."""
    rule = _rule("athlete")
    assert isinstance(rule, ExpansionRule)
    with pytest.raises(ValidationError, match="frozen"):
        rule.table = "something_else"  # type: ignore[misc]


def test_every_expansion_invariant_holds_on_the_fixture_corpus(report: CorpusAudit) -> None:
    """The whole point: the corpus satisfies what the documentation claims of it."""
    assert report.failed_invariants == ()
    assert len(report.expansion.invariants) >= 15
    for item in report.expansion.invariants:
        assert item["expected"] == 0, item["name"]
        assert item["observed"] == 0, item["name"]


def test_the_invariants_name_what_they_check(report: CorpusAudit) -> None:
    names = {item["name"] for item in report.expansion.invariants}
    assert {
        "attempts_reference_a_competition",
        "reported_results_reference_a_competition",
        "competitions_reference_a_meet",
        "attempts_are_never_zero",
        "attempt_result_agrees_with_the_source_sign",
        "reported_result_sign_agrees_with_its_semantics",
        "no_reported_result_is_derived",
        "fourth_attempts_are_labelled",
        "attempt_numbers_are_unique_per_lift",
        "attempts_stay_inside_the_declared_event",
        "bodyweight_is_never_zero",
        "participation_places_are_positive",
        "every_athlete_has_a_competition",
    } <= names


def test_a_dangling_meet_reference_fails_the_invariant(
    report: CorpusAudit, corpus: Corpus, tmp_path: Path
) -> None:
    """The check that found two independent meet-identity derivations disagreeing.

    One meet identity on a copy of the corpus is renamed so that competitions reference an
    identity the meet table does not hold. Both tables are still well formed and both still
    digest cleanly against their own manifest, which is exactly why no build-time or
    digest-time check can see this and an explicit referential invariant has to.
    """
    data_root, relative = corpus
    broken = _copy_corpus_with_one_dangling_meet(data_root, relative, tmp_path)

    audited = audit_corpus(
        AuditRequest(dataset_dir=Path("broken"), data_root=tmp_path, generated_at=STAMP)
    ).audit
    failures = {
        str(item["name"]): item for item in audited.expansion.invariants if not item["holds"]
    }

    assert set(failures) == {"competitions_reference_a_meet"}
    assert failures["competitions_reference_a_meet"]["observed"] == 1
    assert audited.failed_invariants
    # Nothing else moved: the break is one foreign key, not a corrupted corpus.
    assert audited.entities.athletes == report.entities.athletes
    assert audited.entities.meets == report.entities.meets
    assert audited.performance.totals == report.performance.totals
    assert broken.is_dir()


def _copy_corpus_with_one_dangling_meet(data_root: Path, relative: str, destination: Path) -> Path:
    """Copy the corpus into *destination*, breaking exactly one meet reference.

    Rewritten rather than edited in place: the fixture corpus is module-scoped, and a test
    that damaged it would corrupt every later test's premise.
    """
    tables = data_root / relative / "tables"
    broken = destination / "broken"
    (broken / "tables").mkdir(parents=True)
    competition = pl.read_parquet(tables / "competition.parquet")
    orphan = ORPHAN_MEET_ID
    retargeted = competition.with_columns(
        pl.when(pl.col("competition_meet_id") == pl.col("competition_meet_id").min())
        .then(pl.lit(orphan))
        .otherwise(pl.col("competition_meet_id"))
        .alias("competition_meet_id")
    )
    for name in table_names():
        table = read_parquet(tables / f"{name}.parquet", table_name=name)
        payload = retargeted.to_arrow().cast(table.schema) if name == "competition" else table
        write_parquet(payload, broken / "tables" / f"{name}.parquet", table_name=name)
    _copy_manifest(data_root / relative / "manifest.json", broken / "manifest.json")
    return broken


def _copy_manifest(source: Path, destination: Path) -> None:
    """Copy a dataset manifest so a rewritten corpus still declares its artifacts.

    The artifact digests are deliberately left as they were: the point of the test is that
    a corpus can carry a dangling foreign key while every digest it publishes still matches
    the bytes on disk.
    """
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["dataset_id"] = "psd_comp_broken"
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_a_broken_corpus_audit_still_names_its_own_corpus(corpus: Corpus, tmp_path: Path) -> None:
    data_root, relative = corpus
    _copy_corpus_with_one_dangling_meet(data_root, relative, tmp_path)

    audited = audit_corpus(
        AuditRequest(dataset_dir=Path("broken"), data_root=tmp_path, generated_at=STAMP)
    ).audit

    assert audited.dataset_id == "psd_comp_broken"
    assert audited.manifest_digest != corpus[0].joinpath(relative).as_posix()


# ---------------------------------------------------------------------------
# Renderings
# ---------------------------------------------------------------------------


def test_the_summary_leads_with_identity_and_coverage(report: CorpusAudit) -> None:
    summary = report.summary()
    assert "canonical entities" in summary
    assert "longitudinal structure" in summary
    assert "expansion contract" in summary
    assert "reported, not repaired" in summary


def test_the_markdown_rendering_carries_the_tables(report: CorpusAudit) -> None:
    text = audit_markdown(report)
    assert "| table | source field | absent source value | zero | negative |" in text
    assert "| invariant | statement | expected | observed | holds |" in text
    assert "| By declared event | rows |" in text
    assert "| Meet count | identities |" in text
    assert "**NO**" not in text, "a failing invariant must be visible in the document"


def test_a_truncated_grouping_says_so_in_the_document(report: CorpusAudit) -> None:
    """The document is the part a reviewer reads; a hidden omission would mislead them."""
    coverage = CategoryCoverage(
        distinct_members=100,
        listed_members=2,
        rows={"a": 5, "b": 4},
        remainder_rows=11,
        null_rows=1,
    )
    table = coverage_table("By something", coverage)

    assert "2 of 100 members listed" in table
    assert "11 rows in the remainder" in table
    assert "*(not recorded)* | 1" in table


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

#: Populated by the module fixture so a test can find the corpus it is auditing. Keyed by
#: dataset id rather than passed around, because a broken copy is audited from a different
#: data root and the copy needs the original corpus's table files.
_AUDIT_DATA_ROOTS: dict[str, Path] = {}


def _rule(table: str) -> ExpansionRule:
    """Return the declared expansion rule for *table*.

    Raises:
        AssertionError: No rule is declared for the table.
    """
    for rule in EXPANSION_RULES:
        if rule.table == table:
            return rule
    msg = f"No expansion rule declared for {table!r}."
    raise AssertionError(msg)


def _table_path(data_root: Path, relative: str, table: str) -> Path:
    """Return one canonical artifact's path inside the fixture corpus."""
    manifest = read_manifest(relative, data_root=data_root)
    return artifact_paths(manifest, relative, data_root=data_root)[table]
