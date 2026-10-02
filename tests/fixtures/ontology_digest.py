"""The pinned exercise-ontology content digest.

The ontology is the one artifact whose *content* is a scientific decision rather than a
refactor: adding a canonical exercise, retargeting an alias, or narrowing a curated
refusal all change what a training log means. A test that only re-derives the digest
from the catalog would pass through every one of those changes silently, so the expected
value is committed here and asserted on every platform CI runs.

This is also the cross-platform determinism gate. ``content_digest`` is defined over
logical values rather than Parquet bytes, so the same three digests must come out on
Windows and on Ubuntu. A platform divergence fails this test on both jobs, naming the
table that diverged.

Updating it
-----------

A change to the intended vocabulary should fail this test. Read the failure, confirm
the change is intended, then replace the value below and say why in the commit message.
Never regenerate it reflexively.

``psd-ontology/1.0.0`` re-pinned this digest for three declared reasons: the
competition-deadlift stance semantics changed, ``Competition Deadlift`` and
``Deadlift (Conventional)`` were retargeted, and ``exercise_normalization`` lost the
uncalibrated ``confidence`` column. All three change what the artifact means, which is
exactly what a committed digest exists to make visible.
"""

from __future__ import annotations

import hashlib

from psd.ontology import default_ontology
from psd.ontology.artifact import build_ontology_dataset
from psd.serialization.canonical import content_digest
from psd.serialization.dataset import CanonicalDataset

__all__ = ("EXPECTED_ONTOLOGY_DIGEST", "EXPECTED_TABLE_DIGESTS", "ontology_digest")

#: Per-table digests, so a divergence names the table that changed.
EXPECTED_TABLE_DIGESTS: dict[str, str] = {
    "exercise_definition": "84a7069051c3b9de2fea287d24aa4974d2d56546383780ee12b9557776bcccc7",
    "exercise_alias": "e7edf50280e1137d7351ca6643a769ee467645d35c4f2bcffaeb86819a960af3",
    "exercise_normalization": "8a24e2f97345073a88328be63c27e36073c28adb1194e207a25ff3381217a0c9",
}

#: A single digest over all three tables, in registry order.
EXPECTED_ONTOLOGY_DIGEST: str = "b7aa0fa75be0860280a0d5c14f3d5e42103d7c73de2eb7da1bc541059ed3f000"

ONTOLOGY_TABLES: tuple[str, ...] = (
    "exercise_definition",
    "exercise_alias",
    "exercise_normalization",
)


def ontology_digest(dataset: CanonicalDataset | None = None) -> str:
    """Return the single committed-value digest for the ontology artifact."""
    resolved = build_ontology_dataset(default_ontology()) if dataset is None else dataset
    digest = hashlib.sha256()
    for table in ONTOLOGY_TABLES:
        digest.update(content_digest(resolved.table(table), table_name=table).encode("ascii"))
    return digest.hexdigest()


def table_digests(dataset: CanonicalDataset | None = None) -> dict[str, str]:
    """Return the per-table digests of the ontology artifact."""
    resolved = build_ontology_dataset(default_ontology()) if dataset is None else dataset
    return {
        table: content_digest(resolved.table(table), table_name=table) for table in ONTOLOGY_TABLES
    }
