"""End-to-end tests for the PSD-COMP corpus build.

The build is where RES-237's real requirements live: source-faithful conversion, the
reported irregularities counted rather than repaired, determinism, and a corpus that the
ordinary dataset reader accepts.

The fixture in :mod:`tests.fixtures.openpowerlifting` carries one row per edge case, so
these tests assert the semantics of a real corpus rather than the shape of a happy path.
The properties under test are the ones a consumer depends on:

* a negative attempt weight becomes a failed attempt at the positive magnitude;
* a fourth attempt is a record attempt and is kept, with its number;
* a negative reported best is published as the lowest weight attempted and failed;
* ``DQ``, ``DD``, ``G`` and ``NS`` are statuses, never placings;
* a value PSD cannot read is counted and dropped, not guessed;
* the same CSV bytes produce the same artifact digests every time.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from psd.ingest.openpowerlifting.acquire import (
    acquire_snapshot_from_local_file,
    resolve_snapshot_csv,
)
from psd.ingest.openpowerlifting.contract import SourceSchemaError, expected_columns
from psd.ingest.openpowerlifting.snapshot import OpenPowerliftingSnapshot
from psd.ingest.openpowerlifting.transform import (
    BuildConfig,
    BuildRequest,
    BuildResult,
    build_corpus,
)
from psd.schema.registry import table_names
from psd.schema.version import SCHEMA_VERSION
from psd.serialization.dataset import read_dataset, verify_dataset
from tests.fixtures.openpowerlifting import SAMPLE_ROW_COUNT, write_sample_snapshot

#: A fixed instant, so the only non-deterministic input to a build is excluded.
STAMP = datetime(2026, 10, 2, 0, 0, tzinfo=UTC)

#: Deliberately tiny. The build partitions and batches identically at any scale; what these
#: values prove is that the partitioning and batching boundaries do not change results.
CONFIG = BuildConfig(chunk_rows=3, batch_rows=4, partitions=2)

DATASET_RELATIVE = "canonical/psd_comp"

#: One built corpus, and the canonical rows of its non-empty tables. Row values are read
#: through `Any` because `to_pylist` returns a heterogeneous column per row and
#: these tests are about which values survived the conversion, not about their types.
Rows = dict[str, list[dict[str, Any]]]
Built = tuple[BuildResult, Rows]


@pytest.fixture(scope="module")
def pinned(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, OpenPowerliftingSnapshot]:
    """The pinned fixture snapshot and its data root, built once for this module."""
    root = tmp_path_factory.mktemp("psd_comp")
    snapshot = write_sample_snapshot(root / "source" / "sample.csv", data_root=root / "data")
    return root, snapshot


@pytest.fixture(scope="module")
def built(pinned: tuple[Path, OpenPowerliftingSnapshot]) -> Built:
    """One corpus build, plus its rows, each annotated with the source name it came from.

    The canonical schema deliberately stores no name -- PSD holds athlete identities, not
    a copy of the source's strings -- so the tests recover the source name the way any
    consumer would: through ``athlete_source_link``, which records the source key the
    identity was built from.
    """
    root, snapshot = pinned
    result = _build(root / "data", snapshot)
    dataset, _manifest = read_dataset(
        f"{DATASET_RELATIVE}/{snapshot.archive_sha256}", data_root=root / "data"
    )
    tables = {name: table.to_pylist() for name, table in dataset.tables.items() if table.num_rows}
    marker = {
        row["athlete_id"]: row["source_athlete_key"]
        for row in tables.get("athlete_source_link", [])
    }
    rows: Rows = {}
    for name, entries in tables.items():
        rows[name] = [
            entry
            if "athlete_id" not in entry
            else {**entry, "athlete_name_marker": marker.get(entry["athlete_id"])}
            for entry in entries
        ]
    return result, rows


def _build(data_root: Path, snapshot: OpenPowerliftingSnapshot) -> BuildResult:
    return build_corpus(
        BuildRequest(
            csv_path=resolve_snapshot_csv(snapshot, data_root=data_root),
            snapshot=snapshot,
            data_root=data_root,
            config=CONFIG,
            ingested_at=STAMP,
        )
    )


def _one(rows: Rows, table: str, **match: object) -> dict[str, object]:
    """Return the single row of *table* whose fields equal *match*.

    Raises:
        AssertionError: There is no such row, or more than one, which is itself the defect:
            the fixture is written so every expected row is unambiguous.
    """
    found = [
        row for row in rows[table] if all(row.get(field) == value for field, value in match.items())
    ]
    assert len(found) == 1, f"expected one {table} row for {match}, found {len(found)}"
    return found[0]


def _by_name(rows: Rows, name: str) -> Rows:
    """Return every canonical row that the source filed under one name marker."""
    return {
        table: [row for row in entries if row.get("athlete_name_marker") == name]
        for table, entries in rows.items()
        if table in {"athlete", "competition", "competition_attempt", "competition_reported_result"}
    }


# ---------------------------------------------------------------------------
# The corpus itself
# ---------------------------------------------------------------------------


def test_build_covers_every_canonical_table(built: Built) -> None:
    """A dataset's table set is uniform, so absent tables are still written."""
    result, _rows = built
    assert set(result.row_counts) == set(table_names())


def test_build_is_declared_at_the_pinned_schema_version(built: Built) -> None:
    result, _rows = built
    assert result.manifest.schema_version == SCHEMA_VERSION.tag


def test_build_records_the_source_and_its_lineage(built: Built) -> None:
    """Provenance is the point of a pinned snapshot; it must reach the manifest."""
    result, _rows = built
    assert len(result.manifest.sources) == 1
    source = result.manifest.sources[0]
    assert source.source_id.startswith("openpowerlifting_")
    assert [entry.transform_name for entry in result.manifest.lineage]


def test_build_verifies_against_its_own_manifest(
    built: Built,
    pinned: tuple[Path, OpenPowerliftingSnapshot],
) -> None:
    """The manifest must describe the bytes actually on disk."""
    root, snapshot = pinned
    result = verify_dataset(
        f"{DATASET_RELATIVE}/{snapshot.archive_sha256}", data_root=root / "data"
    )

    assert result.manifest_digest_ok
    assert result.artifacts_ok
    assert result.problems == ()
    assert result.checked_artifacts == len(table_names())
    assert result.dataset_dir.is_dir()


def test_build_is_reproducible_from_the_same_bytes(
    pinned: tuple[Path, OpenPowerliftingSnapshot],
) -> None:
    """Two builds of one snapshot must agree on every digest, row count and counter."""
    root, snapshot = pinned
    first, second = _build(root / "data", snapshot), _build(root / "data", snapshot)

    assert [a.sha256 for a in first.manifest.artifacts] == [
        a.sha256 for a in second.manifest.artifacts
    ]
    assert [a.content_sha256 for a in first.manifest.artifacts] == [
        a.content_sha256 for a in second.manifest.artifacts
    ]
    assert first.row_counts == second.row_counts
    assert first.counters.to_dict() == second.counters.to_dict()


def test_build_is_independent_of_partition_and_batch_size(
    pinned: tuple[Path, OpenPowerliftingSnapshot],
) -> None:
    """Chunking is a memory strategy, not part of the result."""
    root, snapshot = pinned
    fine = _build(root / "data", snapshot)
    coarse = build_corpus(
        BuildRequest(
            csv_path=resolve_snapshot_csv(snapshot, data_root=root / "data"),
            snapshot=snapshot,
            data_root=root / "data",
            config=BuildConfig(chunk_rows=97, batch_rows=64, partitions=16),
            ingested_at=STAMP,
        )
    )

    assert [a.content_sha256 for a in fine.manifest.artifacts] == [
        a.content_sha256 for a in coarse.manifest.artifacts
    ]
    assert fine.row_counts == coarse.row_counts


# ---------------------------------------------------------------------------
# Irregularities are counted, never repaired
# ---------------------------------------------------------------------------


def test_every_source_row_is_read(
    built: Built,
) -> None:
    result, _rows = built
    assert result.counters.source_rows == SAMPLE_ROW_COUNT


def test_an_unreadable_value_is_counted_and_dropped(built: Built) -> None:
    """``Age=not-a-number`` must be reported, never coerced to zero or guessed."""
    result, rows = built
    assert result.counters.age_values_unparseable == 1
    # The fixture publishes one unreadable age, on the female Hal Twofold entry.
    lifter = _one(rows, "competition", athlete_name_marker="Hal Twofold", participation_place=6)
    assert lifter["age_reported"] is None


def test_nothing_else_was_unreadable(built: Built) -> None:
    """A counter that moves without a reason in the fixture is a regression."""
    result, _rows = built
    assert result.counters.attempt_values_unparseable == 0
    assert result.counters.bodyweight_values_unparseable == 0
    assert result.counters.bodyweight_values_nonpositive == 0
    assert result.counters.dates_unparseable == 0
    assert result.counters.attempt_values_zero == 0
    assert result.counters.unrecognised_values == {}


def test_every_source_value_is_mapped_or_counted(built: Built) -> None:
    """An unmapped enumeration member must never be silently dropped."""
    result, _rows = built
    assert result.counters.unrecognised_values == {}


# ---------------------------------------------------------------------------
# Attempts
# ---------------------------------------------------------------------------


def test_a_failed_attempt_keeps_its_magnitude(built: Built) -> None:
    """``Bench1Kg=-100`` is a 100 kg attempt that was failed, not a negative load."""
    _result, rows = built
    attempt = _one(rows, "competition_attempt", load_kg=100.0, attempt_number=1)

    assert attempt["result"] == "bad_lift"
    assert attempt["source_attempt_raw"] == -100.0


def test_a_successful_attempt_is_marked_good(built: Built) -> None:
    _result, rows = built
    attempt = _one(rows, "competition_attempt", load_kg=110.0, attempt_number=3)

    assert attempt["result"] == "good_lift"
    assert attempt["source_attempt_raw"] == 110.0


def test_a_fourth_attempt_is_a_record_attempt(built: Built) -> None:
    """The source publishes a fourth bench attempt; it is kept, and numbered four."""
    _result, rows = built
    attempt = _one(rows, "competition_attempt", attempt_number=4, lift="bench")

    assert attempt["attempt_role"] == "record_fourth"
    assert attempt["attempt_order_basis"] == "source_explicit"
    assert attempt["load_kg"] == 127.5


def test_a_fourth_attempt_exists_in_a_single_lift_event(built: Built) -> None:
    """A ``D`` entry may take a record attempt; it is still a record attempt, not a third."""
    _result, rows = built
    attempt = _one(
        rows,
        "competition_attempt",
        attempt_number=4,
        athlete_name_marker="Kip Single",
    )
    results = _by_name(rows, "Kip Single")["competition_reported_result"]

    assert attempt["lift"] == "deadlift"
    assert attempt["attempt_role"] == "record_fourth"
    assert attempt["load_kg"] == 272.5
    # The published best and total both exclude it: 265 is the best of the first three.
    assert {result["result_kind"]: result["value"] for result in results} == {
        "deadlift_best": 265.0
    }


def test_every_fourth_attempt_is_labelled_as_one(built: Built) -> None:
    _result, rows = built
    fourths = [row for row in rows["competition_attempt"] if row["attempt_number"] == 4]

    assert len(fourths) == 2
    assert all(row["attempt_role"] == "record_fourth" for row in fourths)


def test_the_first_attempt_is_the_opener(built: Built) -> None:
    _result, rows = built
    attempts = rows["competition_attempt"]
    for attempt in attempts:
        assert attempt["is_opener"] is (attempt["attempt_number"] == 1)


def test_a_lifter_with_no_attempts_gets_none(built: Built) -> None:
    """Reported bests are not attempts; reconstructing them would invent history."""
    _result, rows = built
    reported = [
        r
        for r in rows["competition_reported_result"]
        if r["result_kind"] == "squat_best" and r["value"] == 160.0
    ]
    assert len(reported) == 1
    assert not [
        r
        for r in rows["competition_attempt"]
        if r["competition_id"] == reported[0]["competition_id"] and r["lift"] == "squat"
    ]


# ---------------------------------------------------------------------------
# Reported results
# ---------------------------------------------------------------------------


def test_a_negative_reported_best_means_failed_attempt_only(built: Built) -> None:
    """``Best3BenchKg=-45`` publishes the lowest weight attempted and failed.

    The source publishes it negative to mean "45 kg, and it failed". PSD keeps the sign in
    ``source_value_raw`` so the reader can see what was written, and stores the magnitude in
    ``value`` so it can be compared with real lifts.
    """
    _result, rows = built
    reported = _one(rows, "competition_reported_result", value=45.0)

    assert reported["reported_best_semantics"] == "failed_attempt_only"
    assert reported["source_value_raw"] == -45.0
    assert reported["result_source_field"] == "Best3BenchKg"


def test_a_positive_reported_best_is_the_successful_best(built: Built) -> None:
    _result, rows = built
    reported = _one(rows, "competition_reported_result", value=230.0)

    assert reported["reported_best_semantics"] == "successful_best"
    assert reported["result_kind"] == "deadlift_best"


def test_a_total_is_recorded_without_inventing_components(built: Built) -> None:
    """``TotalKg=500`` with no lifts is kept as published, and stays the only result."""
    _result, rows = built
    results = _by_name(rows, "Dee Totalless")["competition_reported_result"]

    assert [(r["result_kind"], r["value"]) for r in results] == [("total", 500.0)]
    assert results[0]["unit"] == "kg"


def test_scoring_points_are_dimensionless(built: Built) -> None:
    """Dots is points, not kilograms; saying so stops a consumer reading it as a mass."""
    _result, rows = built
    for kind in ("dots", "wilks", "gl_points", "other"):
        reported = [r for r in rows["competition_reported_result"] if r["result_kind"] == kind]
        assert reported, f"the fixture must publish a {kind} score"
        for result in reported:
            assert result["unit"] is None


def test_bests_and_totals_carry_kilograms(built: Built) -> None:
    _result, rows = built
    masses = [r for r in rows["competition_reported_result"] if r["unit"] == "kg"]
    assert {r["result_kind"] for r in masses} == {
        "squat_best",
        "bench_best",
        "deadlift_best",
        "total",
    }


def test_reported_results_are_never_derived(built: Built) -> None:
    """Everything here is published by the source, so nothing is marked derived."""
    _result, rows = built
    assert all(r["is_derived"] is False for r in rows["competition_reported_result"])
    assert all(r["derivation_note"] is None for r in rows["competition_reported_result"])


def test_observed_at_is_the_meet_date(built: Built) -> None:
    _result, rows = built
    for result in rows["competition_reported_result"]:
        assert str(result["observed_at"]).startswith("2025-11-08")


# ---------------------------------------------------------------------------
# Participation, age, equipment
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "expected_code", "expected_kind"),
    [
        pytest.param("Cyd Failed", "DQ", "disqualified", id="DQ"),
        pytest.param("Gale Outly", "DD", "drug_disqualified", id="DD"),
        pytest.param("Eve Guestly", "G", "guest", id="G"),
        pytest.param("Finn Nowhere", "NS", "no_show", id="NS"),
    ],
)
def test_a_non_placing_code_is_never_a_placing(
    built: Built,
    name: str,
    expected_code: str,
    expected_kind: str,
) -> None:
    """``DQ``, ``DD``, ``G`` and ``NS`` are statuses; coercing any to a rank invents one.

    The source code is kept verbatim in ``participation_status`` and the canonical meaning
    in ``participation_status_kind``, so a reader sees both what was published and what it
    means, and neither is a placing.
    """
    _result, rows = built
    competition = _one(rows, "competition", athlete_name_marker=name)

    assert competition["participation_status"] == expected_code
    assert competition["participation_status_kind"] == expected_kind
    assert competition["participation_place"] is None


def test_a_placing_is_kept(built: Built) -> None:
    _result, rows = built
    competition = _one(rows, "competition", athlete_name_marker="Bo Reported")

    assert competition["participation_status"] == "1"
    assert competition["participation_status_kind"] == "placed"
    assert competition["participation_place"] == 1


def test_an_approximate_age_is_marked_approximate(built: Built) -> None:
    """``23.5`` is a whole number plus a half-year; it is not exact."""
    _result, rows = built
    competition = _one(rows, "competition", athlete_name_marker="Ada Liftwell#1")

    assert competition["age_reported"] == 23.5
    assert competition["age_precision"] == "approximate"


def test_a_whole_age_is_exact(built: Built) -> None:
    _result, rows = built
    competition = _one(rows, "competition", athlete_name_marker="Ada Liftwell#2")

    assert competition["age_reported"] == 23.0
    assert competition["age_precision"] == "exact"


def test_a_bounded_weight_class_is_kept_as_published(built: Built) -> None:
    """The source's own signed convention is preserved, not reinterpreted."""
    _result, rows = built
    competition = _one(
        rows, "competition", athlete_name_marker="Ada Liftwell#1", competition_event="sbd"
    )

    assert competition["weight_class_raw"] == "-93"


def test_an_open_ended_weight_class_is_kept_as_published(built: Built) -> None:
    _result, rows = built
    competition = _one(rows, "competition", athlete_name_marker="Finn Nowhere")

    assert competition["weight_class_raw"] == "90+"


def test_bodyweight_is_carried_in_kilograms(built: Built) -> None:
    _result, rows = built
    competition = _one(
        rows, "competition", athlete_name_marker="Ada Liftwell#1", competition_event="sbd"
    )

    assert competition["bodyweight_kg"] == 91.4
    assert competition["bodyweight_unit"] == "kg"


def test_a_missing_bodyweight_stays_missing(built: Built) -> None:
    """An absent weight is not zero, and not the class boundary either."""
    _result, rows = built
    competition = _one(rows, "competition", athlete_name_marker="Dee Totalless")

    assert competition["bodyweight_kg"] is None
    assert competition["bodyweight_raw"] is None


def test_drug_tested_status_is_carried(built: Built) -> None:
    """``Tested=Yes`` describes the category, so it is a flag, not an outcome."""
    _result, rows = built
    tested = _one(
        rows, "competition", athlete_name_marker="Ada Liftwell#1", competition_event="sbd"
    )
    untested = _one(rows, "competition", athlete_name_marker="Ada Liftwell#2")

    assert tested["is_drug_tested_category"] is True
    assert untested["is_drug_tested_category"] is None


def test_equipment_is_mapped_without_losing_the_source_word(built: Built) -> None:
    """The source's vocabulary is richer than the canonical one, so both are kept."""
    _result, rows = built
    pairs = {
        (row["equipment_class_raw"], row["equipment_class"])
        for row in rows["competition"]
        if row["equipment_class_raw"]
    }
    assert pairs == {
        ("Raw", "raw"),
        ("Wraps", "raw_equip"),
        ("Single-ply", "classic_powerlifting"),
        ("Multi-ply", "multi_ply"),
        ("Straps", "straps_allowed"),
        ("Unlimited", "unlimited"),
    }


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------


def test_a_source_supplied_disambiguator_makes_two_lifters(built: Built) -> None:
    """``Name#1`` and ``Name#2`` are the source saying these are two people."""
    _result, rows = built
    first = _by_name(rows, "Ada Liftwell#1")
    second = _by_name(rows, "Ada Liftwell#2")

    assert {r["athlete_id"] for r in first["athlete"]} != {
        r["athlete_id"] for r in second["athlete"]
    }
    assert first["athlete"][0]["identity_status"] == "single_source_unverified"


def test_one_name_under_two_sexes_is_one_flagged_identity(built: Built) -> None:
    """``Hal Twofold`` is published as both F and M, and PSD says so instead of guessing.

    The two readings cannot be separated from this source alone: they may be two people who
    share a name, or one person the source recorded inconsistently. Splitting on the sex
    category would invent two identities; merging silently would hide the conflict. So the
    identity is shared, the sex is left unknown, and the conflict is flagged for review.
    """
    _result, rows = built
    lifters = _by_name(rows, "Hal Twofold")["athlete"]

    assert {row["athlete_id"] for row in lifters} == {lifters[0]["athlete_id"]}
    assert lifters[0]["sex_category"] is None
    assert lifters[0]["sex_category_raw"] is None
    assert lifters[0]["ambiguity_group_id"] == "opl-name-conflict:Hal Twofold"


def test_one_conflicting_name_is_still_one_athlete_row(built: Built) -> None:
    """A name reported twice under two sexes must not become two rows under one identity.

    The identity is derived from the name, so two rows would share one ``athlete_id`` and
    duplicate the primary key the registry declares unique. A corpus that passes every
    digest check while putting two rows under one identity is internally consistent and
    scientifically wrong, and only a key check catches it.
    """
    result, rows = built
    identities = [row["athlete_id"] for row in rows["athlete"]]
    source_names = {
        row["source_athlete_key"]
        for row in rows["athlete_source_link"]
        if row["athlete_id"] == rows["athlete"][0]["athlete_id"]
    }

    assert len(identities) == len(set(identities)), "athlete primary key is not unique"
    assert result.row_counts["athlete"] == len(set(identities))
    assert len(source_names) == 1, "one identity was linked to more than one source name"


def test_the_athlete_key_is_unique_across_the_corpus(built: Built) -> None:
    """Every corpus table's declared key is unique, not only the athlete's."""
    _result, rows = built
    for table, key in (
        ("athlete", ("athlete_id",)),
        ("athlete_source_link", ("athlete_id", "source_id")),
        ("competition", ("competition_id",)),
        ("competition_attempt", ("competition_attempt_id",)),
        ("competition_meet", ("competition_meet_id",)),
        ("competition_reported_result", ("competition_reported_result_id",)),
    ):
        keys = [tuple(row[column] for column in key) for row in rows[table]]
        assert len(keys) == len(set(keys)), f"{table} duplicates its primary key {key}"


def test_one_competition_entry_belongs_to_exactly_one_athlete(built: Built) -> None:
    """``competition_id`` identifies one source participation; it is keyed by athlete."""
    _result, rows = built
    by_competition: dict[str, str] = {}
    for row in rows["competition"]:
        existing = by_competition.setdefault(row["competition_id"], row["athlete_id"])
        assert existing == row["athlete_id"], row["competition_id"]


def test_every_competition_links_to_an_athlete(built: Built) -> None:
    _result, rows = built
    athlete_ids = {row["athlete_id"] for row in rows["athlete"]}

    assert all(row["athlete_id"] in athlete_ids for row in rows["competition"])


def test_every_attempt_links_to_its_competition(built: Built) -> None:
    _result, rows = built
    competition_ids = {row["competition_id"] for row in rows["competition"]}

    assert all(row["competition_id"] in competition_ids for row in rows["competition_attempt"])
    assert all(
        row["competition_id"] in competition_ids for row in rows["competition_reported_result"]
    )


def test_a_self_described_sex_category_is_mapped(built: Built) -> None:
    """``Mx`` is kept, and mapped to a member that says exactly that."""
    _result, rows = built
    lifter = _one(rows, "athlete", athlete_name_marker="Gale Outly")

    assert lifter["sex_category"] == "other_self_described"
    assert lifter["sex_category_raw"] == "Mx"


# ---------------------------------------------------------------------------
# Meets
# ---------------------------------------------------------------------------


def test_a_meet_is_one_place_date_and_federation(built: Built) -> None:
    """Two meets in different towns are two meets, even on one date for one federation."""
    _result, rows = built
    meet_ids = {row["competition_meet_id"] for row in rows["competition"]}

    assert len(meet_ids) == len(rows["competition_meet"])
    assert len(rows["competition_meet"]) == 6


def test_meet_town_is_carried(built: Built) -> None:
    _result, rows = built
    meet = _one(rows, "competition_meet", meet_town="Albany")

    assert meet["meet_federation"] == "CPU"
    assert meet["meet_parent_federation"] == "IPF"
    assert meet["meet_state"] == "NY"
    assert meet["meet_country"] == "USA"


def test_a_variant_spelling_of_a_meet_is_one_meet(built: Built) -> None:
    """``raw   NATIONAL   open`` and ``Raw National Open`` name one meet, not two.

    Meet identity is a normalized six-field key, so a cosmetic difference must not mint a
    second identity: a meet table with two rows for it would leave every foreign key
    pointing at one of them, and a merged pair would hide a real difference. The fixture
    proves the first by asserting one row, and the audit's
    ``competitions_reference_a_meet`` invariant would catch the second.
    """
    _result, rows = built
    dallas = _one(rows, "competition_meet", meet_town="Dallas")
    variant = _one(rows, "competition", athlete_name_marker="Ivy Variant")

    assert variant["competition_meet_id"] == dallas["competition_meet_id"]
    # Which spelling survives is fixed rather than chosen: the lexicographically smallest
    # raw variant tuple, so the outcome depends on neither row order nor chunking.
    assert dallas["meet_name"] == "  raw   NATIONAL   open "
    assert len(rows["competition_meet"]) == len(
        {row["competition_meet_id"] for row in rows["competition"]}
    )


def test_every_competition_references_a_meet_that_exists(built: Built) -> None:
    """A competition pointing at a meet that is not there is a dangling foreign key.

    Nothing else in the build catches this: the two tables are each well formed and each
    digest is computed over its own rows, so a disagreement between them is invisible to
    every integrity check the build performs. It was found by the audit's
    ``competitions_reference_a_meet`` invariant.
    """
    _result, rows = built
    meet_ids = {row["competition_meet_id"] for row in rows["competition_meet"]}

    assert {row["competition_meet_id"] for row in rows["competition"]} <= meet_ids


def test_a_parent_federation_is_not_the_federation(built: Built) -> None:
    """``CPU`` competes under ``IPF``; collapsing the two would merge rival meets."""
    _result, rows = built
    meet = _one(rows, "competition_meet", meet_town="Fresno")

    assert meet["meet_federation"] == "XPC"
    assert meet["meet_parent_federation"] == "WPA"


def test_a_missing_parent_federation_stays_missing(built: Built) -> None:
    _result, rows = built
    meet = _one(rows, "competition_meet", meet_town="Dallas")

    assert meet["meet_parent_federation"] is None
    assert meet["meet_federation"] == "USPA"


def test_sanctioning_is_carried(built: Built) -> None:
    _result, rows = built
    sanctioned = _one(rows, "competition_meet", meet_town="Dallas")
    unsanctioned = _one(rows, "competition_meet", meet_town="ExCeL")

    assert sanctioned["is_sanctioned"] is True
    assert unsanctioned["is_sanctioned"] is False


def test_meet_dates_are_date_only(built: Built) -> None:
    """The source publishes a date. PSD records the precision rather than inventing a time."""
    _result, rows = built
    for meet in rows["competition_meet"]:
        assert meet["event_time_precision"] == "date_only"
        assert str(meet["meet_date"]).startswith("2025-11-08")


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------


def test_reduced_events_are_kept(built: Built) -> None:
    """Every declared event is an event in its own right, not a malformed SBD."""
    _result, rows = built
    events = {row["competition_event"] for row in rows["competition"]}

    assert events == {"sbd", "bd", "sd", "sb", "s", "b", "d"}


def test_a_reduced_event_has_no_attempts_at_the_absent_lift(built: Built) -> None:
    """A ``B`` entry with a squat attempt would be a source error, and PSD must not add one."""
    _result, rows = built
    for attempt in rows["competition_attempt"]:
        competition = _one(rows, "competition", competition_id=attempt["competition_id"])
        lifts = _LIFTS_OF_EVENT[str(competition["competition_event"])]
        assert attempt["lift"] in lifts


_LIFTS_OF_EVENT: dict[str, set[str]] = {
    "sbd": {"squat", "bench", "deadlift"},
    "bd": {"bench", "deadlift"},
    "sd": {"squat", "deadlift"},
    "sb": {"squat", "bench"},
    "b": {"bench"},
    "s": {"squat"},
    "d": {"deadlift"},
}


def test_the_declared_event_covers_every_attempt(built: Built) -> None:
    """The other direction: a lifter entered for ``B`` who only squatted would be a defect."""
    _result, rows = built
    by_competition = {row["competition_id"]: row for row in rows["competition"]}
    seen: dict[str, set[str]] = {}
    for attempt in rows["competition_attempt"]:
        competition_id = str(attempt["competition_id"])
        seen.setdefault(competition_id, set()).add(str(attempt["lift"]))

    for competition_id, lifts in seen.items():
        event = str(by_competition[competition_id]["competition_event"])
        assert lifts <= _LIFTS_OF_EVENT[event]


# ---------------------------------------------------------------------------
# The contract is enforced before a row is read
# ---------------------------------------------------------------------------


def test_a_drifted_header_is_refused_before_any_row_is_read(tmp_path: Path) -> None:
    """An extra column would be read and ignored; a corpus must not be built from that.

    Without the check the transform would infer the unknown column's type, skip it, and
    write a multi-million-row artifact that claims to be the declared corpus while
    describing a different one.
    """
    data_root = tmp_path / "data"
    drifted = tmp_path / "drifted.csv"
    columns = expected_columns()
    drifted.write_text(
        ",".join((*columns, "Squat5Kg")) + "\nJohn,M,SBD,Raw,,,,,,,,,,\n",
        encoding="utf-8",
    )
    snapshot = acquire_snapshot_from_local_file(drifted, data_root=data_root)

    with pytest.raises(SourceSchemaError, match="unknown source column"):
        build_corpus(
            BuildRequest(
                csv_path=resolve_snapshot_csv(snapshot, data_root=data_root),
                snapshot=snapshot,
                data_root=data_root,
                config=CONFIG,
                ingested_at=STAMP,
            )
        )


def test_a_reordered_header_is_refused_too(tmp_path: Path) -> None:
    """Row ordinals are the source-record keys, so a reorder changes what row N means."""
    data_root = tmp_path / "data"
    columns = expected_columns()
    reordered = tmp_path / "reordered.csv"
    reordered.write_text(
        ",".join((columns[1], columns[0], *columns[2:])) + "\nM,John,,,,,,,\n",
        encoding="utf-8",
    )
    snapshot = acquire_snapshot_from_local_file(reordered, data_root=data_root)

    with pytest.raises(SourceSchemaError, match="different order"):
        build_corpus(
            BuildRequest(
                csv_path=resolve_snapshot_csv(snapshot, data_root=data_root),
                snapshot=snapshot,
                data_root=data_root,
                config=CONFIG,
                ingested_at=STAMP,
            )
        )
