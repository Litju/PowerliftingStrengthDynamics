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

``psd-canonical-content/2`` re-pinned it once more for a reason that is *not* a change to
the vocabulary. The content encoding began excluding ``ingested_at`` -- the instant PSD
read a row -- so that two runs of the same source produce the same logical digest. The
ontology's rows are byte-for-byte what they were; only the function that digests them
moved, and the encoding version in the digest's header moved with it.
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
    "exercise_definition": "9be883f7db990b574b143a5f1e66fab75fec045868a85be15b2d1df272900c58",
    "exercise_alias": "f9b2daeba9a668f66b5dc7df1f534d7c84bdf5c28edfb8344119eec614199f60",
    "exercise_normalization": "7b8adef5065f9b97e4f09558b8a117026f791ff46bf3b90968d2c2e3a62c699c",
}

#: A single digest over all three tables, in registry order.
EXPECTED_ONTOLOGY_DIGEST: str = "6ad8f539894bf283862f1f7089387ee2315cd5d2a2eda4dcccf293565c20bd96"

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
