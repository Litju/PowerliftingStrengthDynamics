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
    "exercise_definition": "16574c1f2d8a161078d507f41dcad5167e75b17bb7cc3401c526f662f0778a35",
    "exercise_alias": "3d74dc144ed39bdea7e28d03933a7c89fd0afee3e82c7a29151f6028cc85653e",
    "exercise_normalization": "f29a403dc9af0c55e0e974f9263c62bab5daa9955e07836d76310b64e8aa0681",
}

#: A single digest over all three tables, in registry order.
EXPECTED_ONTOLOGY_DIGEST: str = "d6d381cdd3596747d74cf26407a069bcb5dbd69a7caaa7c6beb0dca6d3268260"

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
