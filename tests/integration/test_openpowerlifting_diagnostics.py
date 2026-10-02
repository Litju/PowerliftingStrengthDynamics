"""Qualification for the four closure diagnostics.

Each diagnostic is held to the same standard the expansion invariants are: it must report
a positive count on data built to provoke it, report zero on data that is correct, and
*fail* on data that is deliberately almost right. A check that cannot fail proves nothing,
and the negative cases below are the reason to believe the numbers mean what they say.

The zero cases live on a separately built, well-formed corpus rather than being asserted
as "the positive fixture happens to contain none of these", because a fixture can gain a
row and quietly stop proving the zero.

Nothing here acquires or downloads: the fixtures are repository-contained, which is also
what lets the self-hosted Windows parity job run the platform-sensitive subset without
touching the network or the real corpus.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from psd.ingest.openpowerlifting.acquire import resolve_snapshot_csv
from psd.ingest.openpowerlifting.audit import (
    NAME_CHANGE_LIMITATION_REASON,
    AuditRequest,
    CorpusAudit,
    audit_corpus,
    audit_markdown,
)
from psd.ingest.openpowerlifting.snapshot import ServiceSnapshotFacts
from psd.ingest.openpowerlifting.transform import BuildRequest, build_corpus
from psd.schema.vocabulary import FindingClass
from tests.fixtures.openpowerlifting import write_sample_snapshot
from tests.fixtures.openpowerlifting_diagnostics import (
    EARLY_SNAPSHOT_DATE,
    STAMP,
    DiagnosticCorpus,
    age_consistent_rows,
    age_inconsistent_rows,
    audit_rows,
    build_diagnostic_corpus,
    copy_corpus_with_modified_table,
    future_dated_rows,
    identity_rows,
    open_ended_weight_class_rows,
    transition_rows,
)

#: Every finding class the audit is allowed to report. A new class must be added here and
#: explained, which is what stops a fifth category appearing without anyone deciding what
#: it means.
EXPECTED_CLASSES = {
    FindingClass.SOURCE_ANOMALY,
    FindingClass.TRANSFORMATION_INVARIANT_FAILURE,
    FindingClass.DESCRIPTIVE_LONGITUDINAL_TRANSITION,
    FindingClass.SOURCE_LIMITATION,
}

#: The snapshot facts the diagnostic fixtures are pinned with: a date deliberately earlier
#: than the meets they contain, so the future-dated check has something to find.
EARLY_SNAPSHOT = ServiceSnapshotFacts(updated_date=EARLY_SNAPSHOT_DATE, revision="f231b4f6")


@pytest.fixture(scope="module")
def diagnostics_corpus(tmp_path_factory: pytest.TempPathFactory) -> DiagnosticCorpus:
    """One corpus built from every diagnostic row, pinned to an early snapshot date."""
    root = tmp_path_factory.mktemp("psd_comp_diagnostics")
    return build_diagnostic_corpus(audit_rows(), root=root, service=EARLY_SNAPSHOT)


@pytest.fixture(scope="module")
def report(diagnostics_corpus: DiagnosticCorpus) -> CorpusAudit:
    """The audit of the diagnostic corpus."""
    return diagnostics_corpus.audit


@pytest.fixture(scope="module")
def clean_corpus(tmp_path_factory: pytest.TempPathFactory) -> DiagnosticCorpus:
    """The general OpenPowerlifting fixture, which is well formed throughout."""
    root = tmp_path_factory.mktemp("psd_comp_diagnostics_clean")
    data_root = root / "data"
    snapshot = write_sample_snapshot(root / "source" / "sample.csv", data_root=data_root)
    build_corpus(
        BuildRequest(
            csv_path=resolve_snapshot_csv(snapshot, data_root=data_root),
            snapshot=snapshot,
            data_root=data_root,
            ingested_at=STAMP,
        )
    )
    relative = Path("canonical") / "psd_comp" / snapshot.archive_sha256
    report = audit_corpus(
        AuditRequest(dataset_dir=relative, data_root=data_root, generated_at=STAMP)
    ).audit
    return DiagnosticCorpus(
        data_root=data_root, relative=relative, audit=report, service=ServiceSnapshotFacts()
    )


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def test_every_family_declares_how_its_counts_must_be_read(report: CorpusAudit) -> None:
    diagnostics = report.diagnostics
    assert diagnostics.identity.finding_class is FindingClass.SOURCE_LIMITATION
    assert diagnostics.chronology.finding_class is FindingClass.SOURCE_ANOMALY
    assert diagnostics.units.finding_class is FindingClass.TRANSFORMATION_INVARIANT_FAILURE
    assert diagnostics.transitions.finding_class is FindingClass.DESCRIPTIVE_LONGITUDINAL_TRANSITION


def test_the_classification_index_covers_exactly_the_four_classes(report: CorpusAudit) -> None:
    diagnostics = report.diagnostics
    assert {meaning.finding_class for meaning in diagnostics.finding_classes} == EXPECTED_CLASSES
    for meaning in diagnostics.finding_classes:
        assert meaning.meaning
        assert meaning.treatment


def test_the_findings_index_is_not_one_warning_bucket(report: CorpusAudit) -> None:
    """A flat list of unnamed counts is the thing this classification exists to stop."""
    diagnostics = report.diagnostics
    for finding_class in EXPECTED_CLASSES:
        found = diagnostics.findings_of(finding_class)
        assert found, finding_class
        assert all(item.finding_class is finding_class for item in found)
    assert {item.family for item in diagnostics.findings} == {
        "identity",
        "chronology",
        "units",
        "transitions",
    }
    assert len({item.name for item in diagnostics.findings}) == len(diagnostics.findings)


def test_findings_survive_a_json_round_trip_with_their_class(report: CorpusAudit) -> None:
    restored = CorpusAudit.from_dict(report.to_dict())
    assert restored.diagnostics.findings == report.diagnostics.findings
    assert restored.diagnostics.failed_fidelity_findings == ()


def test_every_class_appears_in_both_renderings(report: CorpusAudit) -> None:
    """The classification is in the data *and* in the document, or a reader only gets half."""
    text = audit_markdown(report)
    for meaning in report.diagnostics.finding_classes:
        assert f"`{meaning.finding_class.value}`" in text
        assert meaning.meaning in text
    assert "### Classified diagnostic index" in text
    assert "diagnostics by finding class" in report.summary()


# ---------------------------------------------------------------------------
# A. Identity and name stability
# ---------------------------------------------------------------------------


def test_names_under_two_sex_categories_are_reported_as_one_flagged_identity(
    report: CorpusAudit,
) -> None:
    """``Hal Twofold`` is published as F and as M; the audit reports it, never resolves it."""
    identity = report.diagnostics.identity
    assert identity.names_under_multiple_sex_categories == 1
    assert identity.identities == report.entities.athletes


def test_disambiguated_names_and_their_collision_groups_are_counted(report: CorpusAudit) -> None:
    """Two ``#N`` keys share one base name, and the bare base name is published too."""
    identity = report.diagnostics.identity
    assert identity.disambiguated_names == 2
    assert identity.base_name_collision_groups == 1
    assert identity.largest_base_name_collision_group == 3
    assert identity.base_names_also_published_unsuffixed == 1
    distribution = identity.base_name_group_size_distribution
    assert distribution["3 name keys"] == 1
    assert sum(distribution.values()) == identity.base_name_collision_groups


def test_the_collision_group_names_every_member_key(report: CorpusAudit) -> None:
    example = report.diagnostics.identity.base_name_examples[0]
    assert example["base_name"] == "Ada Liftwell"
    assert example["members"] == 3
    assert example["unsuffixed_members"] == 1
    assert example["name_keys"] == ["Ada Liftwell", "Ada Liftwell#1", "Ada Liftwell#2"]


def test_the_identity_key_maps_one_to_one_in_both_directions(report: CorpusAudit) -> None:
    """The canonical identity is the source name alone, so neither direction may split."""
    identity = report.diagnostics.identity
    assert identity.name_keys_mapping_to_several_identities == 0
    assert identity.identities_mapping_to_several_name_keys == 0


def test_a_historical_name_change_is_reported_as_not_identifiable(report: CorpusAudit) -> None:
    """The one thing this source cannot answer, stated rather than guessed at."""
    limitations = report.diagnostics.identity.limitations
    assert len(limitations) == 1
    limitation = limitations[0]
    assert limitation.identifiable is False
    assert "name" in limitation.question
    assert limitation.reason == NAME_CHANGE_LIMITATION_REASON
    assert limitation.reason == "OpenPowerlifting Name is itself the source identity key"


def test_the_limitation_reaches_the_findings_index_and_the_document(report: CorpusAudit) -> None:
    """A limitation that lives only inside a nested object is a limitation nobody reads."""
    entry = next(
        item
        for item in report.diagnostics.findings
        if item.name.startswith("limitation:")
        and item.finding_class is FindingClass.SOURCE_LIMITATION
    )
    assert entry.count == 0
    assert "not identifiable" in entry.statement
    text = audit_markdown(report)
    assert "What this source cannot show" in text
    assert "**not identifiable**" in text
    assert NAME_CHANGE_LIMITATION_REASON in text


def test_ordinary_longitudinal_variation_is_never_an_identity_change(report: CorpusAudit) -> None:
    """Country, federation and body mass change over a career without being a conflict."""
    names = {item.name for item in report.diagnostics.findings if item.family == "identity"}
    for forbidden in ("country", "region", "state", "federation", "bodyweight", "age_class"):
        assert not any(forbidden in name for name in names), forbidden


def test_identity_reports_nothing_on_a_corpus_without_collisions(tmp_path: Path) -> None:
    """The zero case: lifters with distinct names have no base-name collision group."""
    corpus = build_diagnostic_corpus(transition_rows(), root=tmp_path / "distinct_names")
    identity = corpus.audit.diagnostics.identity
    assert identity.base_name_collision_groups == 0
    assert identity.disambiguated_names == 0
    assert identity.names_under_multiple_sex_categories == 0
    assert sum(identity.base_name_group_size_distribution.values()) == 0


# ---------------------------------------------------------------------------
# B. Suspicious chronology
# ---------------------------------------------------------------------------


def test_future_dated_records_are_counted_against_the_pinned_snapshot(report: CorpusAudit) -> None:
    chronology = report.diagnostics.chronology
    assert chronology.pinned_snapshot_date == EARLY_SNAPSHOT_DATE
    assert chronology.pinned_snapshot_date_basis == "service_reported"
    assert chronology.future_dated_meets == 2
    assert chronology.future_dated_competitions == 8
    assert {example["competition_date"] for example in chronology.future_dated_examples} == {
        "2025-11-08T00:00:00Z",
        "2026-02-14T00:00:00Z",
    }


def test_every_meet_carries_at_most_one_source_date(report: CorpusAudit) -> None:
    """Zero because the meet identity includes the date, and reported because that is a claim."""
    chronology = report.diagnostics.chronology
    assert chronology.meets_spanning_multiple_dates == 0
    assert chronology.meet_date_examples == ()


def test_identities_whose_ages_cannot_share_a_birth_year_are_reported(report: CorpusAudit) -> None:
    """40 in 2020 admits 1980/1979 and 40 in 2024 admits 1984/1983: nothing survives."""
    chronology = report.diagnostics.chronology
    assert chronology.identities_without_a_compatible_birth_year == 2
    reported = {
        example["source_athlete_key"] for example in chronology.incompatible_identity_examples
    }
    assert reported == {"Gil Year", "Hal Fifties"}


def test_mutually_compatible_ages_are_not_reported(tmp_path: Path) -> None:
    """The zero case, built separately so adding a row to the positive fixture cannot erase it.

    ``Ada Aging`` is 40 then 44 across four seasons, and ``Eve Midpoint`` is reported
    approximately as 40.5 then 44.5, which the source's convention pins to exactly one
    birth year each. Both survive the intersection, and neither may be reported.
    """
    corpus = build_diagnostic_corpus(
        age_consistent_rows(), root=tmp_path / "consistent", service=EARLY_SNAPSHOT
    )
    chronology = corpus.audit.diagnostics.chronology
    assert chronology.identities_with_multiple_age_observations == 2
    assert chronology.identities_without_a_compatible_birth_year == 0
    assert chronology.incompatible_identity_examples == ()


def test_the_inconsistent_fixture_is_reported_for_both_lifters_it_names(tmp_path: Path) -> None:
    """``Gil Year`` is the exact case and ``Hal Fifties`` the approximate one."""
    corpus = build_diagnostic_corpus(
        age_inconsistent_rows(), root=tmp_path / "inconsistent", service=EARLY_SNAPSHOT
    )
    chronology = corpus.audit.diagnostics.chronology
    assert chronology.identities_with_multiple_age_observations == 2
    assert chronology.identities_without_a_compatible_birth_year == 2
    assert {
        example["source_athlete_key"] for example in chronology.incompatible_identity_examples
    } == {
        "Gil Year",
        "Hal Fifties",
    }


def test_chronology_says_so_when_no_snapshot_date_exists(tmp_path: Path) -> None:
    """Without a pinned date the future-dated check cannot run, and must not pretend to."""
    corpus = build_diagnostic_corpus(future_dated_rows(), root=tmp_path / "undated")
    chronology = corpus.audit.diagnostics.chronology
    assert chronology.pinned_snapshot_date is None
    assert chronology.pinned_snapshot_date_basis == "unavailable"
    assert chronology.future_dated_competitions is None
    assert chronology.future_dated_meets is None
    assert chronology.future_dated_examples == ()
    # The checks that need no date still ran, and the counts are null rather than zero so
    # a consumer cannot read "not measurable" as "measured, and clean".
    assert chronology.meets_spanning_multiple_dates == 0
    assert chronology.age_observations == 2
    assert chronology.identities_with_age_observations == 2


def test_the_archive_declared_date_is_used_when_the_service_stated_none(tmp_path: Path) -> None:
    corpus = build_diagnostic_corpus(
        future_dated_rows(),
        root=tmp_path / "archive_dated",
        service=ServiceSnapshotFacts(archive_declared_date=EARLY_SNAPSHOT_DATE),
    )
    chronology = corpus.audit.diagnostics.chronology
    assert chronology.pinned_snapshot_date_basis == "archive_declared"
    assert chronology.future_dated_meets == 2


def test_the_age_diagnostic_is_read_only(report: CorpusAudit) -> None:
    """A diagnostic only: the ages the corpus holds are still the ages the source gave."""
    assert report.athlete_context.ages_exact > 0
    assert report.athlete_context.ages_approximate > 0
    assert report.diagnostics.chronology.identities_without_a_compatible_birth_year == 2
    assert report.failed_invariants == ()


def test_the_chronology_rule_is_stated_where_it_is_applied(report: CorpusAudit) -> None:
    semantics = report.diagnostics.chronology.age_semantics
    assert "Y-n" in semantics
    assert "Y-n-1" in semantics
    assert "Y-(n+1)" in semantics
    assert semantics in audit_markdown(report)


# ---------------------------------------------------------------------------
# C. Unit consistency
# ---------------------------------------------------------------------------


def test_a_correct_transform_reports_zero_mass_fidelity_mismatches(report: CorpusAudit) -> None:
    units = report.diagnostics.units
    assert units.declared_source_unit == "kilograms"
    assert units.mismatch_total == 0
    assert report.diagnostics.failed_fidelity_findings == ()
    assert units.bodyweight_rows_checked > 0
    assert units.attempt_rows_checked > 0
    assert units.reported_rows_checked > 0


def test_every_mapped_mass_field_is_named_in_the_audit(report: CorpusAudit) -> None:
    """The list comes from the contract, so a new mass column cannot escape the audit."""
    fields = report.diagnostics.units.mass_source_fields
    assert all(name.endswith("Kg") for name in fields)
    assert {
        "BodyweightKg",
        "Squat1Kg",
        "Bench4Kg",
        "Deadlift1Kg",
        "Best3SquatKg",
        "TotalKg",
    } <= set(fields)
    assert len(fields) == 17


def test_the_zero_case_is_proved_on_a_separate_well_formed_corpus(
    clean_corpus: DiagnosticCorpus,
) -> None:
    units = clean_corpus.audit.diagnostics.units
    assert units.mismatch_total == 0
    assert units.open_ended_weight_classes >= 2
    assert units.open_ended_weight_classes_coerced == 0
    assert units.reported_totals_checked >= 1
    assert units.reported_total_unit_mismatches == 0
    assert units.reported_best_rows_checked > 0
    assert units.reported_best_semantics_mismatches == 0
    assert units.scoring_rows_checked > 0
    assert units.scoring_rows_claiming_a_mass_unit == 0
    assert clean_corpus.audit.diagnostics.failed_fidelity_findings == ()


def test_a_signed_attempt_and_a_negative_best_keep_their_source_meaning(
    report: CorpusAudit,
) -> None:
    """The unit audit reads the sign as a result and the magnitude as a mass, on both tables."""
    assert report.performance.negative_reported_bests == 1
    assert report.diagnostics.units.attempt_sign_result_mismatches == 0
    assert report.diagnostics.units.attempt_load_kg_mismatches == 0
    assert report.diagnostics.units.reported_best_semantics_mismatches == 0
    assert report.diagnostics.units.reported_value_mismatches == 0


def test_an_open_ended_weight_class_stays_open_ended(tmp_path: Path) -> None:
    """``90+`` claims no maximum, so the canonical label must still say so.

    The check is a self-consistency check on canonical data rather than a source
    comparison: the label is persisted verbatim and no numeric weight-class column exists,
    so a canonical ``90`` would be indistinguishable from a source that published ``90``.
    That limit is stated rather than glossed, because the alternative -- inferring the
    source value -- is exactly the reconstruction this corpus refuses to do.
    """
    corpus = build_diagnostic_corpus(
        open_ended_weight_class_rows(), root=tmp_path / "open_ended", service=EARLY_SNAPSHOT
    )
    units = corpus.audit.diagnostics.units
    assert units.open_ended_weight_classes == 1
    assert units.open_ended_weight_classes_coerced == 0


def test_no_missing_source_mass_field_becomes_zero(clean_corpus: DiagnosticCorpus) -> None:
    assert clean_corpus.audit.athlete_context.bodyweight_missing > 0
    assert clean_corpus.audit.diagnostics.units.bodyweight_absent_became_zero == 0


def test_a_body_mass_that_is_not_the_source_value_is_caught(
    diagnostics_corpus: DiagnosticCorpus, tmp_path: Path
) -> None:
    """The negative case: one column rewritten so the canonical mass is not the source mass."""
    _break(
        diagnostics_corpus,
        tmp_path / "bodyweight",
        table="competition",
        modify=pl.col("bodyweight_kg") + 1.0,
    )
    units = _audit_broken_copy(tmp_path / "bodyweight").diagnostics.units
    assert units.bodyweight_kg_mismatches == units.bodyweight_rows_checked
    assert units.mismatch_total == units.bodyweight_rows_checked
    assert units.bodyweight_rows_checked > 0


def test_an_attempt_load_that_is_not_the_source_magnitude_is_caught(
    diagnostics_corpus: DiagnosticCorpus, tmp_path: Path
) -> None:
    _break(
        diagnostics_corpus,
        tmp_path / "attempt",
        table="competition_attempt",
        modify=pl.col("load_kg") * 2.0,
    )
    units = _audit_broken_copy(tmp_path / "attempt").diagnostics.units
    assert units.attempt_load_kg_mismatches == units.attempt_rows_checked
    assert units.mismatch_total == units.attempt_rows_checked


def test_a_signed_attempt_whose_result_disagrees_with_its_sign_is_caught(
    diagnostics_corpus: DiagnosticCorpus, tmp_path: Path
) -> None:
    _break(
        diagnostics_corpus,
        tmp_path / "sign",
        table="competition_attempt",
        modify=pl.when(pl.col("source_attempt_raw") < 0)
        .then(pl.lit("good_lift"))
        .otherwise(pl.col("result"))
        .alias("result"),
    )
    units = _audit_broken_copy(tmp_path / "sign").diagnostics.units
    # ``Zed Signed`` fails a second squat, an opener bench and a second deadlift.
    assert units.attempt_sign_result_mismatches == 3
    assert units.attempt_sign_result_mismatches < units.attempt_rows_checked
    assert units.mismatch_total == 3


def test_a_reported_value_that_is_not_the_source_magnitude_is_caught(
    diagnostics_corpus: DiagnosticCorpus, tmp_path: Path
) -> None:
    _break(
        diagnostics_corpus,
        tmp_path / "reported",
        table="competition_reported_result",
        modify=pl.col("value") + 0.5,
    )
    units = _audit_broken_copy(tmp_path / "reported").diagnostics.units
    assert units.reported_value_mismatches == units.reported_rows_checked
    assert units.mismatch_total == units.reported_rows_checked


def test_a_negative_best_relabelled_as_successful_is_caught(
    diagnostics_corpus: DiagnosticCorpus, tmp_path: Path
) -> None:
    _break(
        diagnostics_corpus,
        tmp_path / "semantics",
        table="competition_reported_result",
        modify=pl.when(pl.col("reported_best_semantics") == "failed_attempt_only")
        .then(pl.lit("successful_best"))
        .otherwise(pl.col("reported_best_semantics"))
        .alias("reported_best_semantics"),
    )
    units = _audit_broken_copy(tmp_path / "semantics").diagnostics.units
    assert units.reported_best_semantics_mismatches == 1
    assert units.mismatch_total == 1


def test_a_scoring_score_relabelled_as_kilograms_is_caught(
    clean_corpus: DiagnosticCorpus, tmp_path: Path
) -> None:
    """Dots and Wilks are dimensionless; a canonical mass label on one is a real defect."""
    _break(
        clean_corpus,
        tmp_path / "scoring",
        table="competition_reported_result",
        modify=pl.when(pl.col("unit").is_null())
        .then(pl.lit("kg"))
        .otherwise(pl.col("unit"))
        .alias("unit"),
    )
    units = _audit_broken_copy(tmp_path / "scoring").diagnostics.units
    assert units.scoring_rows_checked > 0
    assert units.scoring_rows_claiming_a_mass_unit == units.scoring_rows_checked
    assert units.mismatch_total == units.scoring_rows_checked


def test_a_unit_mislabelled_attempt_is_caught(
    clean_corpus: DiagnosticCorpus, tmp_path: Path
) -> None:
    _break(
        clean_corpus,
        tmp_path / "attempt_unit",
        table="competition_attempt",
        modify=pl.lit("lb").alias("load_unit"),
    )
    units = _audit_broken_copy(tmp_path / "attempt_unit").diagnostics.units
    assert units.attempt_unit_mismatches == units.attempt_rows_checked
    assert units.mismatch_total == units.attempt_rows_checked


def test_the_unit_diagnostic_states_that_it_is_not_a_plausibility_check(
    report: CorpusAudit,
) -> None:
    """No physiological range, no inferred pounds: the section has to say so out loud."""
    text = audit_markdown(report)
    assert "no per-row unit field" in text
    assert "No\nplausibility heuristic runs and no pounds are inferred." in text or (
        "plausibility heuristic runs and no pounds are inferred" in text
    )
    assert report.diagnostics.units.declared_source_unit == "kilograms"


def test_every_mass_mismatch_reaches_the_findings_index(report: CorpusAudit) -> None:
    names = {
        item.name
        for item in report.diagnostics.findings_of(FindingClass.TRANSFORMATION_INVARIANT_FAILURE)
    }
    assert {
        "mass_fidelity_mismatches",
        "bodyweight_kg_mismatches",
        "attempt_load_kg_mismatches",
        "attempt_sign_result_mismatches",
        "reported_value_mismatches",
        "reported_best_semantics_mismatches",
        "open_ended_weight_classes_coerced",
        "scoring_rows_claiming_a_mass_unit",
    } <= names


# ---------------------------------------------------------------------------
# D. Equipment and federation transitions
# ---------------------------------------------------------------------------


def test_transitions_are_computed_at_athlete_meet_level(report: CorpusAudit) -> None:
    transitions = report.diagnostics.transitions
    assert transitions.grouping == "athlete_id + competition_meet_id"
    assert transitions.ordering == "competition_date, competition_meet_id"
    equipment = transitions.field("equipment_class")
    # Every athlete-meet is in exactly one of three states: a single known value, more
    # than one contradictory value, or no value at all.
    silent = (
        equipment.athlete_meets
        - equipment.athlete_meets_with_a_known_state
        - equipment.intra_meet_state_conflicts
    )
    assert silent == 1, "only Jan Silent's second meet states no equipment"


def test_a_lifter_who_changes_equipment_twice_is_counted_twice(report: CorpusAudit) -> None:
    """Raw -> Wraps -> Multi-ply across three dated meets."""
    equipment = report.diagnostics.transitions.field("equipment_class")
    assert equipment.source_column == "Equipment"
    assert equipment.total_transitions == 2
    assert equipment.athletes_with_transitions == 1
    assert equipment.transitions_per_athlete["2-3 transitions"] == 1
    assert sum(equipment.transitions_per_athlete.values()) == report.entities.athletes


def test_a_federation_change_is_counted_once(report: CorpusAudit) -> None:
    federation = report.diagnostics.transitions.field("federation")
    assert federation.source_column == "Federation"
    assert federation.total_transitions == 1
    assert federation.athletes_with_transitions == 1
    assert federation.transitions_per_athlete["1 transition"] == 1


def test_parent_federation_transitions_are_reported_too(report: CorpusAudit) -> None:
    sanctioning = report.diagnostics.transitions.field("sanctioning_body")
    assert sanctioning.source_column == "ParentFederation"
    assert sanctioning.total_transitions == 0
    assert sanctioning.athletes_with_transitions == 0
    assert sanctioning.transitions_per_athlete["0 transitions"] == report.entities.athletes


def test_a_meet_that_contradicts_itself_is_a_conflict_not_a_switch(report: CorpusAudit) -> None:
    """``Gil Contradiction`` enters two events and the source states a different one for each."""
    equipment = report.diagnostics.transitions.field("equipment_class")
    assert equipment.intra_meet_state_conflicts == 1
    # The conflicting meet contributes no state, so it cannot also contribute a transition.
    assert equipment.athletes_with_transitions == 1


def test_an_absent_value_is_not_a_state_transition(report: CorpusAudit) -> None:
    """``Jan Silent`` states no equipment at the second meet; that is a silence, not a change."""
    equipment = report.diagnostics.transitions.field("equipment_class")
    assert equipment.athlete_meets_with_a_known_state < equipment.athlete_meets
    assert equipment.transitions_per_athlete["0 transitions"] >= 1


def test_an_unknown_value_is_never_treated_as_a_state(report: CorpusAudit) -> None:
    transitions = report.diagnostics.transitions
    assert transitions.unknown_is_a_state is False
    assert "competition category" in transitions.equipment_reading
    assert "not evidence" in transitions.equipment_reading
    assert transitions.equipment_reading in audit_markdown(report)


def test_transition_counts_reach_the_classified_index(report: CorpusAudit) -> None:
    names = {
        item.name
        for item in report.diagnostics.findings_of(FindingClass.DESCRIPTIVE_LONGITUDINAL_TRANSITION)
    }
    assert {
        "equipment_class_total_transitions",
        "equipment_class_intra_meet_state_conflicts",
        "equipment_class_athletes_with_transitions",
        "federation_total_transitions",
        "sanctioning_body_total_transitions",
    } <= names


def test_a_single_meet_history_has_no_transitions(tmp_path: Path) -> None:
    """The zero case: one meet per lifter cannot produce a longitudinal change."""
    corpus = build_diagnostic_corpus(identity_rows(), root=tmp_path / "single_meet")
    for field in corpus.audit.diagnostics.transitions.fields:
        assert field.total_transitions == 0
        assert field.athletes_with_transitions == 0
        assert field.transitions_per_athlete["0 transitions"] == corpus.audit.entities.athletes
        assert field.intra_meet_state_conflicts == 0


def test_the_transition_fixture_alone_reproduces_its_counts(tmp_path: Path) -> None:
    corpus = build_diagnostic_corpus(
        transition_rows(), root=tmp_path / "transitions", service=EARLY_SNAPSHOT
    )
    transitions = corpus.audit.diagnostics.transitions
    assert transitions.field("equipment_class").total_transitions == 2
    assert transitions.field("equipment_class").intra_meet_state_conflicts == 1
    assert transitions.field("federation").total_transitions == 1
    assert transitions.field("federation").athletes_with_transitions == 1


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _break(
    corpus: DiagnosticCorpus,
    destination: Path,
    *,
    table: str,
    modify: pl.Expr,
) -> Path:
    """Copy *corpus* into *destination* with one column of *table* rewritten."""
    return copy_corpus_with_modified_table(
        data_root=corpus.data_root,
        relative=corpus.relative,
        destination=destination,
        table=table,
        modify=modify,
    )


def _audit_broken_copy(destination: Path) -> CorpusAudit:
    """Audit the deliberately broken corpus written into *destination*."""
    return audit_corpus(
        AuditRequest(
            dataset_dir=Path("broken"),
            data_root=destination / "data",
            generated_at=STAMP,
        )
    ).audit
