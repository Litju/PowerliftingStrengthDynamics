"""Build and verify a tiny PSD-COMP corpus, with no network and no real data.

The self-hosted Windows parity job must prove that the corpus pipeline still *runs* on
Windows, and it must do so without the 4,036,909-row snapshot. This script is that
middle: a repository-contained CSV of a few rows, put through the real acquisition,
transform, longitudinal-history, audit and verification path, against a temporary data
root, with everything removed afterwards.

It is deliberately not a test. The marked ``windows_parity`` tests already cover the same
code paths with assertions; what this adds is that the whole pipeline runs end to end
outside pytest, on the runner's own Python, writing into the runner's own temp directory
and leaving nothing behind. That is the failure mode a unit test cannot have.

What it must not do, and does not: reach the network, read the real corpus, or leave disk
state. ``--keep`` exists for debugging a failure and is the only way anything survives.

Usage::

    uv run python scripts/tiny_psd_comp.py --data-root "$RUNNER_TEMP/psd-parity"
"""

from __future__ import annotations

import argparse
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from psd.ingest.openpowerlifting.acquire import (
    acquire_snapshot_from_local_file,
    resolve_snapshot_csv,
)
from psd.ingest.openpowerlifting.audit import AuditRequest, audit_corpus
from psd.ingest.openpowerlifting.contract import expected_columns
from psd.ingest.openpowerlifting.history import HistoryRequest, build_athlete_history
from psd.ingest.openpowerlifting.snapshot import ServiceSnapshotFacts
from psd.ingest.openpowerlifting.transform import BuildConfig, BuildRequest, build_corpus
from psd.serialization.dataset import verify_dataset

#: A fixed instant. Nothing in this script reads the clock, so two runs of it produce the
#: same corpus and a difference means something changed rather than that time passed.
STAMP = datetime(2026, 1, 5, 9, 30, tzinfo=UTC)

#: The meet every row shares unless it overrides the field, and the header order the
#: contract declares. Building the CSV from ``expected_columns()`` is what makes a row
#: always have exactly the contract's column count.
MEET: dict[str, str] = {
    "Date": "2025-11-08",
    "Federation": "USPA",
    "MeetCountry": "USA",
    "MeetState": "TX",
    "MeetTown": "Dallas",
    "MeetName": "Tiny Parity Open",
    "Sanctioned": "Yes",
    "Division": "Open",
    "Country": "USA",
    "State": "TX",
    "ParentFederation": "IPF",
}

#: One lifter per shape the transform has to get right: a complete three-attempt result with
#: a record fourth attempt, a failed attempt, a negative reported best, a result with no
#: attempt detail at all, and a total published without its component lifts. Small, and
#: still every expansion invariant has something to bite on.
ROWS: tuple[dict[str, str], ...] = (
    {
        "Name": "Tiny Complete",
        "Sex": "M",
        "Event": "SBD",
        "Equipment": "Raw",
        "Age": "29",
        "AgeClass": "29-39",
        "BodyweightKg": "91.4",
        "WeightClassKg": "-93",
        "Squat1Kg": "180",
        "Squat2Kg": "185",
        "Squat3Kg": "187.5",
        "Squat4Kg": "192.5",
        "Best3SquatKg": "187.5",
        "Bench1Kg": "110",
        "Bench2Kg": "-115",
        "Bench3Kg": "120",
        "Best3BenchKg": "115",
        "Deadlift1Kg": "210",
        "Deadlift2Kg": "220",
        "Deadlift3Kg": "230",
        "Best3DeadliftKg": "230",
        "TotalKg": "532.5",
        "Place": "1",
        "Dots": "500.13",
        "Tested": "Yes",
    },
    {
        "Name": "Tiny Negative",
        "Sex": "F",
        "Event": "B",
        "Equipment": "Wraps",
        "BodyweightKg": "62.7",
        "WeightClassKg": "-63",
        "Bench1Kg": "-45",
        "Bench2Kg": "-50",
        "Best3BenchKg": "-45",
        "Place": "DQ",
    },
    {
        "Name": "Tiny Bestsonly",
        "Sex": "M",
        "Event": "S",
        "Equipment": "Single-ply",
        "Age": "40.5",
        "BodyweightKg": "93.5",
        "WeightClassKg": "105",
        "Best3SquatKg": "160",
        "Place": "1",
    },
    {
        "Name": "Tiny Totalless",
        "Sex": "M",
        "Event": "SBD",
        "Equipment": "Multi-ply",
        "Age": "31",
        "WeightClassKg": "90+",
        "TotalKg": "500",
        "Place": "4",
        "Date": "2025-12-06",
        "MeetName": "Tiny Parity Winter",
        "MeetTown": "Austin",
    },
)


def write_fixture(path: Path) -> Path:
    """Write the tiny source CSV to *path* and return it.

    The rows are written from a mapping against the declared header rather than joined by
    hand, because a hand-counted row that drifts out of alignment fails with "too many
    fields" and says nothing about the semantics being exercised.
    """
    columns = expected_columns()
    lines = [",".join(columns)]
    for row in ROWS:
        values = dict(MEET)
        values.update(row)
        lines.append(",".join(str(values.get(column, "")) for column in columns))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run(data_root: Path, *, scratch: Path) -> dict[str, Any]:
    """Pin, build, derive, audit and verify the tiny corpus. Return what it produced.

    Raises:
        AssertionError: The corpus failed verification, an invariant, or the unit-fidelity
            audit. A parity run that quietly reports a broken pipeline is worse than one
            that fails.
    """
    csv_path = write_fixture(scratch / "source" / "tiny-openpowerlifting.csv")
    snapshot = acquire_snapshot_from_local_file(
        csv_path,
        data_root=data_root,
        service=ServiceSnapshotFacts(updated_date="2025-11-01", revision="f231b4f6"),
    )
    built = build_corpus(
        BuildRequest(
            csv_path=resolve_snapshot_csv(snapshot, data_root=data_root),
            snapshot=snapshot,
            data_root=data_root,
            # Small partitions on purpose: the point is the code path, not throughput.
            config=BuildConfig(chunk_rows=3, batch_rows=4, partitions=2),
            ingested_at=STAMP,
        )
    )
    relative = Path("canonical") / "psd_comp" / snapshot.archive_sha256
    build_athlete_history(
        HistoryRequest(dataset_dir=relative, data_root=data_root, created_at=STAMP)
    )
    verification = verify_dataset(relative, data_root=data_root)
    report = audit_corpus(
        AuditRequest(dataset_dir=relative, data_root=data_root, generated_at=STAMP)
    ).audit
    failed = report.failed_invariants
    assert verification.artifacts_ok, verification.problems
    assert not verification.problems, verification.problems
    assert not failed, [dict(item) for item in failed]
    assert report.diagnostics.units.mismatch_total == 0, report.diagnostics.units
    return {
        "snapshot_sha256": snapshot.archive_sha256,
        "source_rows": snapshot.row_count,
        "dataset_relative_path": str(built.manifest.dataset_id),
        "athletes": report.entities.athletes,
        "meets": report.entities.meets,
        "competitions": report.entities.competitions,
        "attempts": report.entities.attempts,
        "reported_results": report.entities.reported_results,
        "invariants_checked": len(report.expansion.invariants),
        "invariants_failed": len(failed),
        "mass_fidelity_mismatches": report.diagnostics.units.mismatch_total,
        "historical_name_changes_identifiable": not report.diagnostics.identity.limitations[
            0
        ].identifiable,
    }


def main(argv: list[str] | None = None) -> int:
    """Run the tiny pipeline. Return a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path, help="Temporary PSD data root.")
    parser.add_argument(
        "--keep", action="store_true", help="Leave the data root and source behind."
    )
    arguments = parser.parse_args(argv)
    if arguments.data_root.exists():
        shutil.rmtree(arguments.data_root)
    scratch = arguments.data_root.parent / "tiny-psd-comp-source"
    if scratch.exists():
        shutil.rmtree(scratch)
    try:
        summary = run(arguments.data_root, scratch=scratch)
    except AssertionError as error:
        sys.stderr.write(f"tiny_psd_comp: corpus did not qualify: {error}\n")
        return 1
    finally:
        if arguments.keep:
            sys.stdout.write(f"  kept  {arguments.data_root} and {scratch}\n")
        else:
            shutil.rmtree(arguments.data_root, ignore_errors=True)
            shutil.rmtree(scratch, ignore_errors=True)
    for key, value in summary.items():
        sys.stdout.write(f"  {key:<32} {value}\n")
    sys.stdout.write("tiny PSD-COMP build/verify: OK\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
