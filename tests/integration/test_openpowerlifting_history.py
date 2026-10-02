"""Tests for the longitudinal athlete-history derivation.

The history is a projection of canonical events, and these tests are mostly about that
claim: what it counts, how it orders, and -- most importantly -- what it refuses to do.

The refusal that matters is identity. Longitudinal continuity is the most persuasive way
to resolve an ambiguous source identity: two rows share a name, joining them makes the
record look coherent, and the ambiguity disappears. That is precisely the reasoning PSD
refuses, so there is a test that a history row is one source identity and never a
continuity-derived person.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from psd.ingest.openpowerlifting.acquire import resolve_snapshot_csv
from psd.ingest.openpowerlifting.history import (
    ATHLETE_HISTORY_ARTIFACT,
    ATHLETE_HISTORY_DIRNAME,
    ATHLETE_HISTORY_MANIFEST,
    ATHLETE_HISTORY_SCHEMA,
    ATHLETE_HISTORY_SPEC,
    CompetitionEventMembership,
    HistoryBuildError,
    HistoryRequest,
    build_athlete_history,
    lifts_of_declared_event,
)
from psd.ingest.openpowerlifting.transform import BuildConfig, BuildRequest, build_corpus
from psd.provenance.manifest import DatasetManifest
from psd.schema.registry import table_names
from psd.serialization.dataset import DatasetLayoutError, artifact_paths, read_manifest
from psd.serialization.derived import (
    DerivedArtifactError,
    DerivedParent,
    DerivedTableSpec,
    derived_metadata,
    read_derived_manifest,
    read_derived_table,
    write_derived_table,
)
from tests.fixtures.openpowerlifting import write_sample_snapshot

pytestmark = pytest.mark.windows_parity

STAMP = datetime(2026, 10, 2, 0, 0, tzinfo=UTC)
CONFIG = BuildConfig(chunk_rows=3, batch_rows=4, partitions=2)

#: Row values are read through ``Any`` because ``to_pylist`` returns a heterogeneous
#: column per row, and these tests are about which values the projection computed.
Row = dict[str, Any]

#: ``(data_root, dataset_relative, manifest)`` for the fixture corpus, built once.
Corpus = tuple[Path, str, DatasetManifest]


@pytest.fixture(scope="module")
def corpus(tmp_path_factory: pytest.TempPathFactory) -> Corpus:
    """One built fixture corpus, shared by every test in this module."""
    root = tmp_path_factory.mktemp("psd_comp_history")
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
    return data_root, relative, read_manifest(relative, data_root=data_root)


@pytest.fixture(scope="module")
def histories(corpus: Corpus) -> tuple[Row, ...]:
    """The derived history rows, one per canonical source identity."""
    data_root, relative, _manifest = corpus
    result = build_athlete_history(
        HistoryRequest(dataset_dir=Path(relative), data_root=data_root, created_at=STAMP)
    )
    return tuple(result.table.to_pylist())


def _athlete_id(corpus: Corpus, name: str) -> str:
    """Return the identity the source filed one name under, through its source link."""
    data_root, relative, manifest = corpus
    links = pl.read_parquet(
        artifact_paths(manifest, relative, data_root=data_root)["athlete_source_link"]
    )
    found = links.filter(pl.col("source_athlete_key") == name).get_column("athlete_id")
    assert found.len() == 1, f"expected exactly one identity for {name!r}, found {found.len()}"
    return str(found[0])


def _history(histories: tuple[Row, ...], **match: object) -> Row:
    """Return the single history row whose fields equal *match*.

    Raises:
        AssertionError: There is no such row, or more than one.
    """
    found = [
        row for row in histories if all(row.get(field) == value for field, value in match.items())
    ]
    assert len(found) == 1, f"expected one history row for {match}, found {len(found)}"
    return found[0]


def _derived_path(corpus: Corpus) -> Path:
    """Return where the derived artifact is written."""
    data_root, relative, _manifest = corpus
    return data_root / relative / ATHLETE_HISTORY_DIRNAME / f"{ATHLETE_HISTORY_ARTIFACT}.parquet"


def _manifest_path(corpus: Corpus) -> Path:
    """Return where the derived artifact's manifest is written."""
    data_root, relative, _manifest = corpus
    return data_root / relative / ATHLETE_HISTORY_DIRNAME / ATHLETE_HISTORY_MANIFEST


# ---------------------------------------------------------------------------
# Shape and declaration
# ---------------------------------------------------------------------------


def test_one_history_row_per_canonical_athlete(corpus: Corpus, histories: tuple[Row, ...]) -> None:
    """The history summarises every identity the corpus holds, and invents none."""
    data_root, relative, manifest = corpus
    table = pl.read_parquet(artifact_paths(manifest, relative, data_root=data_root)["athlete"])

    assert len(histories) == table.height
    assert {row["athlete_id"] for row in histories} == set(table.get_column("athlete_id").to_list())


def test_the_artifact_is_ordered_by_its_primary_key(histories: tuple[Row, ...]) -> None:
    """An artifact whose rows could be permuted without changing its digest is not canonical."""
    identities = [row["athlete_id"] for row in histories]

    assert identities == sorted(identities)
    assert ATHLETE_HISTORY_SPEC.order_by == ("athlete_id",)


def test_the_declared_ordering_must_contain_the_primary_key() -> None:
    """A history ordered by anything but its key could not be reproduced from its rows."""
    with pytest.raises(DerivedArtifactError, match="total"):
        DerivedTableSpec.build(
            name="bad",
            arrow_schema=ATHLETE_HISTORY_SCHEMA,
            primary_key=("athlete_id",),
            order_by=("meet_count",),
            summary="",
        )


def test_the_history_is_not_a_canonical_table(corpus: Corpus) -> None:
    """It lives outside ``tables/``, so it cannot enter the canonical table set or digest."""

    data_root, relative, manifest = corpus

    assert ATHLETE_HISTORY_ARTIFACT not in table_names()
    assert ATHLETE_HISTORY_DIRNAME != "tables"
    assert set(artifact_paths(manifest, relative, data_root=data_root)) == set(table_names())


def test_the_artifact_says_which_corpus_it_projects(corpus: Corpus) -> None:
    """A summary that could read as independent evidence is a scientific hazard."""
    _data_root, _relative, manifest = corpus
    metadata = derived_metadata(_derived_path(corpus))

    assert metadata["psd_table"] == ATHLETE_HISTORY_ARTIFACT
    assert metadata["psd_derived_parent_dataset"] == manifest.dataset_id
    assert metadata["psd_content_sha256"]


def test_the_derived_manifest_pins_the_corpus(corpus: Corpus) -> None:
    _data_root, _relative, _manifest = corpus
    persisted = read_derived_manifest(_manifest_path(corpus))

    assert persisted.parent_dataset_id.startswith("psd_comp_")
    assert persisted.ordering == ("athlete_id",)
    assert persisted.artifact_name == ATHLETE_HISTORY_ARTIFACT
    assert persisted.artifact_relative_path == f"{ATHLETE_HISTORY_DIRNAME}/athlete_history.parquet"
    assert persisted.lineage[0].parent_dataset_id == persisted.parent_dataset_id
    assert persisted.lineage[0].transform_name == "openpowerlifting_longitudinal_history"


def test_the_history_re_reads_and_verifies_against_its_own_digest(corpus: Corpus) -> None:
    reread = read_derived_table(_derived_path(corpus), ATHLETE_HISTORY_SPEC)

    assert reread.schema.equals(ATHLETE_HISTORY_SCHEMA, check_metadata=False)
    assert reread.column_names == list(ATHLETE_HISTORY_SCHEMA.names)


def test_a_history_whose_digest_does_not_describe_its_rows_is_refused(
    corpus: Corpus, tmp_path: Path
) -> None:
    """Reading recomputes the digest from the rows, so a stale one is caught.

    The artifact is written under a different declared name, which is what a copied or
    re-declared artifact looks like from the reader's side: the rows are the history's
    rows, and the self-description no longer describes them. The corpus artifact itself is
    never touched -- a test that corrupted it would corrupt every later test's premise.
    """
    original = read_derived_table(_derived_path(corpus), ATHLETE_HISTORY_SPEC)
    renamed = DerivedTableSpec.build(
        name="athlete_history_relabelled",
        arrow_schema=ATHLETE_HISTORY_SCHEMA,
        primary_key=("athlete_id",),
        summary="",
    )
    write_derived_table(
        original,
        tmp_path / "relabelled.parquet",
        renamed,
        parent=DerivedParent(dataset_id="psd_comp_x", manifest_digest="0" * 64),
    )

    with pytest.raises(DerivedArtifactError, match="content digest"):
        read_derived_table(tmp_path / "relabelled.parquet", ATHLETE_HISTORY_SPEC)


def test_the_history_content_digest_tracks_its_content(corpus: Corpus, tmp_path: Path) -> None:
    """Two different histories must not share a digest, or the audit would prove nothing."""
    original = read_derived_table(_derived_path(corpus), ATHLETE_HISTORY_SPEC)
    parent = DerivedParent(dataset_id="psd_comp_x", manifest_digest="0" * 64)
    kept = write_derived_table(
        original, tmp_path / "kept.parquet", ATHLETE_HISTORY_SPEC, parent=parent
    )
    dropped = write_derived_table(
        original.slice(1, original.num_rows - 1),
        tmp_path / "dropped.parquet",
        ATHLETE_HISTORY_SPEC,
        parent=parent,
    )

    assert kept.content_sha256 != dropped.content_sha256
    assert kept.row_count - dropped.row_count == 1


def test_deriving_without_a_corpus_is_refused(tmp_path: Path) -> None:
    """A projection of nothing would be an empty artifact that looks like a finding."""
    with pytest.raises(DatasetLayoutError):
        build_athlete_history(
            HistoryRequest(dataset_dir=Path("canonical/absent"), data_root=tmp_path / "data")
        )


def test_a_corpus_missing_a_needed_table_is_refused_by_name(corpus: Corpus, tmp_path: Path) -> None:
    """The refusal names the table, so an operator knows exactly what to rebuild."""
    _data_root, _relative, manifest = corpus
    thinned = manifest.model_copy(
        update={
            "artifacts": tuple(
                artifact
                for artifact in manifest.artifacts
                if artifact.name != "competition_attempt"
            )
        }
    )
    staged = tmp_path / "thinned"
    (staged / "tables").mkdir(parents=True)
    (staged / "manifest.json").write_bytes(thinned.to_json_bytes())

    with pytest.raises(HistoryBuildError, match="competition_attempt"):
        build_athlete_history(
            HistoryRequest(
                dataset_dir=Path("thinned"),
                data_root=tmp_path,
                output_dir=Path("thinned") / ATHLETE_HISTORY_DIRNAME,
            )
        )
    assert json.loads((staged / "manifest.json").read_text(encoding="utf-8"))["dataset_id"]


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_the_history_is_reproducible(corpus: Corpus) -> None:
    data_root, relative, _manifest = corpus
    request = HistoryRequest(dataset_dir=Path(relative), data_root=data_root, created_at=STAMP)

    first = build_athlete_history(request)
    second = build_athlete_history(request)

    assert first.manifest.artifact_content_sha256 == second.manifest.artifact_content_sha256
    assert first.manifest.artifact_sha256 == second.manifest.artifact_sha256
    assert first.table.equals(second.table)


def test_a_rebuild_from_nothing_agrees_with_a_rebuild_over_the_old_artifact(
    corpus: Corpus,
) -> None:
    """A projection that depended on what was already on disk would not be reproducible."""
    data_root, relative, manifest = corpus
    request = HistoryRequest(dataset_dir=Path(relative), data_root=data_root, created_at=STAMP)
    first = build_athlete_history(request)
    for entry in (data_root / relative / ATHLETE_HISTORY_DIRNAME).iterdir():
        entry.unlink()
    second = build_athlete_history(request)

    assert first.manifest.artifact_content_sha256 == second.manifest.artifact_content_sha256
    assert first.manifest.parent_dataset_id == manifest.dataset_id


def test_derivation_does_not_append_to_the_canonical_tables(corpus: Corpus) -> None:
    """Re-deriving a projection must leave the corpus it projects exactly as it was."""
    data_root, relative, manifest = corpus
    request = HistoryRequest(dataset_dir=Path(relative), data_root=data_root, created_at=STAMP)
    before = build_athlete_history(request)
    corpus_manifest = read_manifest(relative, data_root=data_root)

    assert corpus_manifest == manifest
    assert build_athlete_history(request).table.num_rows == before.table.num_rows


# ---------------------------------------------------------------------------
# Identity is not solved by continuity
# ---------------------------------------------------------------------------


def test_a_conflicting_name_produces_one_history_row(
    corpus: Corpus, histories: tuple[Row, ...]
) -> None:
    """Two source rows, one name, two sex categories -- one identity, not two.

    Continuity would say the two rows are one person competing twice, or two people; the
    source does not say which. The history therefore holds one row, and that row says so.
    """
    identity = _athlete_id(corpus, "Hal Twofold")
    rows = [row for row in histories if row["athlete_id"] == identity]

    assert len(rows) == 1
    assert rows[0]["ambiguity_group_id"] == "opl-name-conflict:Hal Twofold"
    assert rows[0]["sex_category"] is None
    assert rows[0]["identity_status"] == "single_source_unverified"


def test_a_disambiguated_name_stays_two_identities(
    corpus: Corpus, histories: tuple[Row, ...]
) -> None:
    """``#1`` and ``#2`` are the source asserting two people; nothing merges them."""
    first = _athlete_id(corpus, "Ada Liftwell#1")
    second = _athlete_id(corpus, "Ada Liftwell#2")

    assert first != second
    assert len([row for row in histories if row["athlete_id"] == first]) == 1
    assert len([row for row in histories if row["athlete_id"] == second]) == 1


def test_the_identity_caveat_travels_on_every_row(histories: tuple[Row, ...]) -> None:
    """A caveat that lives only in another table can be lost by a summary consumer."""
    for row in histories:
        assert row["identity_status"] is not None
        assert "ambiguity_group_id" in row
        assert row["meet_count"] >= 1


# ---------------------------------------------------------------------------
# Coverage and counts
# ---------------------------------------------------------------------------


def test_meet_and_competition_counts_are_both_reported(
    corpus: Corpus, histories: tuple[Row, ...]
) -> None:
    """Two entries at one meet are two participations and one meet."""
    row = _history(histories, athlete_id=_athlete_id(corpus, "Hal Twofold"))

    assert row["competition_count"] == 2
    assert row["meet_count"] == 1


def test_event_counts_follow_the_declared_event(corpus: Corpus, histories: tuple[Row, ...]) -> None:
    """``BD`` counts a bench and a deadlift and no squat, because no squat was contested."""
    bench_deadlift = _history(histories, athlete_id=_athlete_id(corpus, "Ada Liftwell#2"))
    bench_only = _history(histories, athlete_id=_athlete_id(corpus, "Cyd Failed"))
    full = _history(histories, athlete_id=_athlete_id(corpus, "Gale Outly"))

    assert (bench_deadlift["squat_event_count"], bench_deadlift["bench_event_count"]) == (0, 1)
    assert bench_deadlift["deadlift_event_count"] == 1
    assert bench_deadlift["event_category_count"] == 1
    assert (bench_only["squat_event_count"], bench_only["deadlift_event_count"]) == (0, 0)
    assert (full["squat_event_count"], full["bench_event_count"]) == (1, 0)
    assert full["deadlift_event_count"] == 1


@pytest.mark.parametrize(
    ("event", "expected"),
    [
        pytest.param("sbd", {"squat", "bench", "deadlift"}, id="sbd"),
        pytest.param("bd", {"bench", "deadlift"}, id="bd"),
        pytest.param("sd", {"squat", "deadlift"}, id="sd"),
        pytest.param("sb", {"squat", "bench"}, id="sb"),
        pytest.param("s", {"squat"}, id="squat-only"),
        pytest.param("b", {"bench"}, id="bench-only"),
        pytest.param("d", {"deadlift"}, id="deadlift-only"),
    ],
)
def test_every_declared_event_has_a_declared_membership(event: str, expected: set[str]) -> None:
    assert lifts_of_declared_event(event) == expected


def test_an_unrecognised_event_contests_nothing_known() -> None:
    """A vocabulary member added after the fact must read as a visible zero, not a crash."""
    assert lifts_of_declared_event("triathlon") == frozenset()
    assert lifts_of_declared_event(None) == frozenset()
    assert set(CompetitionEventMembership) == {"sbd", "bd", "sd", "sb", "s", "b", "d"}


def test_attempt_detail_is_measured_not_assumed(corpus: Corpus, histories: tuple[Row, ...]) -> None:
    """Most federations publish only bests, so a fraction near zero is the normal case."""
    with_attempts = _history(histories, athlete_id=_athlete_id(corpus, "Ada Liftwell#1"))
    without = _history(histories, athlete_id=_athlete_id(corpus, "Bo Reported"))

    assert with_attempts["attempt_detail_meet_count"] == 1
    assert with_attempts["attempt_detail_fraction"] == 1.0
    assert without["attempt_detail_meet_count"] == 0
    assert without["attempt_detail_fraction"] == 0.0


def test_absence_is_never_counted_as_an_observation(
    corpus: Corpus, histories: tuple[Row, ...]
) -> None:
    """``Dee Totalless`` publishes no body mass; the absence is not a zero-kilogram reading."""
    row = _history(histories, athlete_id=_athlete_id(corpus, "Dee Totalless"))

    assert row["bodyweight_observation_count"] == 0
    assert row["age_observation_count"] == 1, "the source does publish an age"
    assert row["federation_count"] == 1


def test_a_reported_total_is_counted_and_none_are_derived(
    corpus: Corpus, histories: tuple[Row, ...]
) -> None:
    """``Dee Totalless`` publishes a total with no components; it counts as one total."""
    row = _history(histories, athlete_id=_athlete_id(corpus, "Dee Totalless"))

    assert row["reported_total_count"] == 1


def test_federation_and_equipment_counts_span_the_record(
    corpus: Corpus, histories: tuple[Row, ...]
) -> None:
    """Counts are of distinct non-absent values, so an absent value adds nothing."""
    row = _history(histories, athlete_id=_athlete_id(corpus, "Ada Liftwell#2"))

    assert row["federation_count"] == 1
    assert row["equipment_category_count"] == 1
    assert row["parent_federation_count"] == 0, "an absent sanctioning body is not a federation"


def test_a_single_meet_history_has_no_span(corpus: Corpus, histories: tuple[Row, ...]) -> None:
    """One meet on one day spans zero days; that is a fact, not a defect."""
    row = _history(histories, athlete_id=_athlete_id(corpus, "Ada Liftwell#1"))

    assert row["first_observed_meet_date"] == row["last_observed_meet_date"]
    assert row["observed_span_days"] == 0
    assert row["first_competition_id"] == row["last_competition_id"]
    assert row["meet_count"] == 1


def test_the_boundary_entries_are_chosen_by_the_event_time_order(
    corpus: Corpus, histories: tuple[Row, ...]
) -> None:
    """Two entries on one meet date are ordered by meet then entry, so 'first' is a fact.

    ``Hal Twofold`` is entered twice at one meet on one date, so the meet and entry
    tiebreak is the only thing that decides which of the two the history calls first.
    Without it the answer would depend on row order.
    """
    row = _history(histories, athlete_id=_athlete_id(corpus, "Hal Twofold"))

    assert row["competition_count"] == 2
    assert row["first_competition_id"].startswith("cmp_")
    assert row["first_competition_id"] < row["last_competition_id"]
