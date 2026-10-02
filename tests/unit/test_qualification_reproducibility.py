"""What the qualification harness is allowed to call reproducible.

Two builds of one pinned snapshot must produce the same corpus, and the report has to be
able to say so without also claiming things the design has already given up. A Parquet file
is not a pure function of its rows -- the writer stamps each file with metadata that is not
a fact about the data -- so two builds of identical rows produce different bytes. Identity
is the logical ``content_sha256``.

This module exists because the distinction was initially missed, and the report said
``identical_manifest_excluding_run_metadata: false`` immediately beside
``identical_content_digests: true``. That reads as a reproducibility failure and is not
one: the manifest carries each artifact's byte digest and byte size precisely so that
verification can measure the file it is given, and those are the two fields that must
differ.
"""

from __future__ import annotations

import copy
from typing import Any

from scripts.qualify_psd_comp import artifact_content_digests, corpus_manifest_digest


def _manifest() -> dict[str, Any]:
    """Return a manifest shaped like a real one, with two artifacts."""
    return {
        "manifest_version": "psd-dataset/1",
        "dataset_id": "psd_comp_32a90763bff87e3e",
        "dataset_kind": "canonical",
        "schema_version": "psd-canonical/1.1.0",
        "created_at": "2026-10-02T09:52:48.165287Z",
        "environment": {"python_version": "3.12.13", "code_commit": "abc123"},
        "lineage": [
            {
                "transform_name": "openpowerlifting_to_canonical",
                "created_at": "2026-10-02T09:52:48.165287Z",
                "schema_version": "psd-canonical/1.1.0",
            }
        ],
        "sources": [
            {
                "source_id": "openpowerlifting_32a90763bff87e3e",
                "ingested_at": "2026-10-02T09:52:48.165287Z",
                "snapshot_sha256": "32a9",
            }
        ],
        "artifacts": [
            {
                "name": "athlete",
                "relative_path": "tables/athlete.parquet",
                "content_sha256": "content-athlete",
                "sha256": "bytes-athlete-build-1",
                "byte_size": 18_401_463,
                "row_count": 1_014_126,
            },
            {
                "name": "competition_attempt",
                "relative_path": "tables/competition_attempt.parquet",
                "content_sha256": "content-attempt",
                "sha256": "bytes-attempt-build-1",
                "byte_size": 411_991_524,
                "row_count": 13_948_408,
            },
        ],
    }


def _second_build(manifest: dict[str, Any]) -> dict[str, Any]:
    """Return *manifest* as a second run of the same snapshot would write it.

    Only the fields that are permitted to move move: the wall-clock stamps, the
    environment, and each artifact's physical bytes.
    """
    second = copy.deepcopy(manifest)
    second["created_at"] = "2026-10-02T11:14:02.771904Z"
    second["environment"]["python_version"] = "3.12.14"
    second["lineage"][0]["created_at"] = "2026-10-02T11:14:02.771904Z"
    second["sources"][0]["ingested_at"] = "2026-10-02T11:14:02.771904Z"
    second["artifacts"][0]["sha256"] = "bytes-athlete-build-2"
    second["artifacts"][0]["byte_size"] = 18_402_017
    second["artifacts"][1]["sha256"] = "bytes-attempt-build-2"
    second["artifacts"][1]["byte_size"] = 411_988_902
    return second


def test_a_second_build_of_one_snapshot_is_the_same_corpus() -> None:
    """Differing stamps and differing Parquet bytes must not read as a different corpus."""
    first = _manifest()
    second = _second_build(first)

    assert corpus_manifest_digest(first) == corpus_manifest_digest(second)


def test_the_byte_digests_really_did_differ() -> None:
    """Guard the guard: the comparison above is only meaningful if bytes moved.

    Without this, a change that made the comparison ignore everything would pass.
    """
    first = _manifest()
    second = _second_build(first)

    assert [a["sha256"] for a in first["artifacts"]] != [a["sha256"] for a in second["artifacts"]]
    assert artifact_content_digests(first) == artifact_content_digests(second)


def test_a_changed_content_digest_is_a_different_corpus() -> None:
    """Identity is the content digest, so a changed one must not compare equal."""
    first = _manifest()
    second = _second_build(first)
    second["artifacts"][1]["content_sha256"] = "content-attempt-edited"

    assert corpus_manifest_digest(first) != corpus_manifest_digest(second)


def test_a_changed_row_count_is_a_different_corpus() -> None:
    """A dropped row must not be hidden by an unchanged digest field elsewhere."""
    first = _manifest()
    second = _second_build(first)
    second["artifacts"][1]["row_count"] = 13_948_407

    assert corpus_manifest_digest(first) != corpus_manifest_digest(second)


def test_a_changed_dataset_identity_is_a_different_corpus() -> None:
    first = _manifest()
    second = _second_build(first)
    second["schema_version"] = "psd-canonical/1.2.0"

    assert corpus_manifest_digest(first) != corpus_manifest_digest(second)
