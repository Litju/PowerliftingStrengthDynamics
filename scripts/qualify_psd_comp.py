"""Full-corpus qualification for PSD-COMP.

Runs the real pipeline against a pinned OpenPowerlifting snapshot under a data root and
records what happened, rather than what an extrapolation predicts. Everything the RES-237
exit gate asks for that is a number comes from here: the actual row counts, the actual
wall-clock per phase, the actual peak process memory, the actual on-disk sizes, and the
actual digest comparison between two builds of the same snapshot.

How peak memory is measured
---------------------------

A kernel high-water mark, not a sample, on both platforms PSD qualifies on:

* **Windows** -- ``GetProcessMemoryInfo(...).PeakWorkingSetSize`` through ``ctypes``. The
  kernel maintains the mark, so it is exact and does not depend on any sampling interval.
* **Linux** -- ``VmHWM`` from ``/proc/self/status``. Also a kernel high-water mark, exact.
* **Anything else** -- a background thread polling ``/proc/self/statm`` every
  ``SAMPLE_INTERVAL_SECONDS``. Reported as sampled, with its resolution, because a sampled
  maximum is only as good as its interval.

The mechanism actually used is recorded in the report. A memory number whose provenance is
unknown is not a measurement, and a report that said "peak RSS" without saying how it was
obtained would be the least useful line in it.

Both exact mechanisms report *this* process. No subprocess is spawned, so no child's memory
is missing from the figure; the one subprocess the harness does start -- the CLI
verification, deliberately, so the qualification exercises the shipped surface -- runs
after the measurement phases and is therefore outside the figure. Its own cost is a
Python interpreter against a Parquet digest pass, and the `verify` phase reports it
separately.

Usage
-----

::

    uv run python scripts/qualify_psd_comp.py \\
        --data-root <absolute data root> --digest <pinned snapshot digest> \\
        --report <path to the JSON report>

Requires the snapshot to be pinned first (``psd openpowerlifting acquire`` or
``--source``). Nothing here reaches the network.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import platform
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from ctypes import wintypes
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from psd.ingest.openpowerlifting.acquire import (
    read_pinned_snapshot,
    resolve_snapshot_csv,
    snapshot_directory,
)
from psd.ingest.openpowerlifting.audit import AuditRequest, audit_corpus
from psd.ingest.openpowerlifting.history import HistoryRequest, build_athlete_history
from psd.ingest.openpowerlifting.transform import BuildConfig, BuildRequest, build_corpus
from psd.serialization.parquet import sha256_file

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Resolution of the sampled fallback, in seconds. Irrelevant on the platforms PSD
#: qualifies on, and reported so a reader knows it is irrelevant rather than guessing.
SAMPLE_INTERVAL_SECONDS = 0.2


# --------------------------------------------------------------------------
# peak memory
# --------------------------------------------------------------------------


def _windows_peak_rss() -> tuple[int, str]:
    """Return ``(peak_rss_bytes, mechanism)`` from the Windows process information API."""

    class _Counters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    counters = _Counters()
    counters.cb = ctypes.sizeof(_Counters)
    # The handle and the pointer need explicit signatures. ctypes would otherwise default
    # every argument to ``c_int``, which truncates a 64-bit pseudo-handle and passes the
    # struct pointer by value -- the call then fails for reasons that look like a bug in
    # this measurement rather than in the measurement's calling convention.
    get_current_process = ctypes.windll.kernel32.GetCurrentProcess
    get_current_process.restype = ctypes.c_void_p
    get_current_process.argtypes = []
    measure = ctypes.windll.psapi.GetProcessMemoryInfo
    measure.argtypes = [ctypes.c_void_p, ctypes.POINTER(_Counters), wintypes.DWORD]
    measure.restype = wintypes.BOOL
    if not measure(get_current_process(), ctypes.byref(counters), counters.cb):
        msg = "GetProcessMemoryInfo failed; peak memory is not measurable here"
        raise OSError(msg)
    return int(counters.PeakWorkingSetSize), "windows:GetProcessMemoryInfo.PeakWorkingSetSize"


def _linux_peak_rss() -> tuple[int, str]:
    """Return ``(peak_rss_bytes, mechanism)`` from ``/proc/self/status``."""
    for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
        if line.startswith("VmHWM:"):
            return int(line.split()[1]) * 1024, "linux:/proc/self/status.VmHWM"
    msg = "/proc/self/status carries no VmHWM; peak memory is not measurable here"
    raise OSError(msg)


def _sampled_rss() -> tuple[int, str]:
    """Return ``(sampled_peak, mechanism)`` by polling ``/proc/self/statm``."""
    page_size = os.sysconf("SC_PAGE_SIZE")
    best = 0
    while True:
        try:
            pages = int(Path("/proc/self/statm").read_text(encoding="utf-8").split()[1])
        except (OSError, IndexError, ValueError):
            return best, "sampled:/proc/self/statm"
        best = max(best, pages * page_size)
        time.sleep(SAMPLE_INTERVAL_SECONDS)


def _select_mechanism() -> tuple[int, str]:
    """Return the best available ``(peak_rss, mechanism)`` for this platform."""
    if os.name == "nt":
        return _windows_peak_rss()
    try:
        return _linux_peak_rss()
    except OSError:
        return _sampled_rss()


class PeakMemory:
    """Track the peak resident set of this process across a run."""

    def __init__(self) -> None:
        self.peak_bytes = 0
        self.mechanism = "unselected"
        self._thread: threading.Thread | None = None
        self._running = False

    def sample(self) -> int:
        """Return the peak resident set observed so far."""
        if self.peak_bytes:
            return self.peak_bytes
        self.peak_bytes, self.mechanism = _select_mechanism()
        if self.mechanism.startswith("sampled"):
            self._running = True
            self._thread = threading.Thread(target=self._poll, daemon=True)
            self._thread.start()
        return self.peak_bytes

    def _poll(self) -> None:
        """Sample repeatedly until the run ends, keeping the maximum."""
        while self._running:
            current, _mechanism = _sampled_rss()
            self.peak_bytes = max(self.peak_bytes, current)
            time.sleep(SAMPLE_INTERVAL_SECONDS)

    def __enter__(self) -> PeakMemory:
        """Resolve the measurement mechanism and start sampling."""
        self.sample()
        return self

    def __exit__(self, *_exc: object) -> None:
        """Stop the sampling thread, if one is running."""
        self._running = False


# --------------------------------------------------------------------------
# phases
# --------------------------------------------------------------------------


def _empty_details() -> dict[str, object]:
    """Return the empty details mapping a phase starts with.

    A named factory rather than a bare ``dict``, so the default carries its type instead of
    being inferred from an unparameterised builtin.
    """
    return {}


@dataclass(slots=True)
class Phase:
    """One measured phase of the qualification run.

    Attributes:
        name: What ran.
        seconds: Wall-clock seconds it took.
        peak_rss_bytes: Peak resident set at the end of the phase.
        details: Anything the phase wants recorded about itself.
    """

    name: str
    seconds: float
    peak_rss_bytes: int
    details: dict[str, object] = field(default_factory=_empty_details)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable rendering."""
        return {
            "name": self.name,
            "seconds": round(self.seconds, 3),
            "peak_rss_bytes": self.peak_rss_bytes,
            "peak_rss_mib": round(self.peak_rss_bytes / (1024 * 1024), 1),
            "details": self.details,
        }


class Recorder:
    """Time each phase and record the peak memory reached during it."""

    def __init__(self, memory: PeakMemory) -> None:
        self.memory = memory
        self.phases: list[Phase] = []

    def run(self, name: str, function: Callable[[], Any], **details: str | int | float) -> Any:
        """Run *function* as a measured phase and return its result."""
        recorded: dict[str, object] = dict(details)
        started = time.perf_counter()
        before = self.memory.sample()
        try:
            return function()
        finally:
            elapsed = time.perf_counter() - started
            self.phases.append(
                Phase(
                    name=name,
                    seconds=elapsed,
                    peak_rss_bytes=max(before, self.memory.sample()),
                    details=recorded,
                )
            )


def directory_bytes(path: Path) -> tuple[int, int]:
    """Return ``(total_bytes, file_count)`` for every file under *path*."""
    total = 0
    count = 0
    if not path.exists():
        return 0, 0
    for entry in path.rglob("*"):
        if entry.is_file():
            total += entry.stat().st_size
            count += 1
    return total, count


def _artifact_digests(manifest: dict[str, Any]) -> dict[str, str]:
    """Return the content digest of every artifact in a persisted manifest."""
    return {artifact["name"]: artifact["content_sha256"] for artifact in manifest["artifacts"]}


#: Manifest fields that describe the *run* rather than the corpus, and therefore differ
#: between two runs of one snapshot. ``ingested_at`` is the same class of field here as it is
#: in the content digest: a wall-clock stamp, not a fact about the corpus.
_RUN_LOCAL_MANIFEST_FIELDS: frozenset[str] = frozenset({"created_at", "ingested_at", "environment"})


def _strip_run_local(value: object) -> object:
    """Return *value* with every run-local manifest field removed, recursively."""
    if isinstance(value, dict):
        stripped: dict[str, object] = {}
        for key, item in cast("dict[str, object]", value).items():
            if key not in _RUN_LOCAL_MANIFEST_FIELDS:
                stripped[key] = _strip_run_local(item)
        return stripped
    if isinstance(value, list):
        return [_strip_run_local(item) for item in cast("list[object]", value)]
    return value


def _manifest_digest_ignoring(manifest: dict[str, Any]) -> str:
    """Return a digest of a manifest describing the corpus rather than the run.

    Two runs of one snapshot legitimately differ in ``created_at``, in the environment
    snapshot, and in every ``ingested_at`` stamp inside the source and lineage records.
    Everything else in the manifest describes the corpus -- its identity, its artifacts and
    their digests, its schema version -- and must be identical.
    """
    return hashlib.sha256(
        json.dumps(_strip_run_local(manifest), sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def _sha256_file(path: Path | None) -> str | None:
    """Return the SHA-256 of *path*, or ``None`` when there is no file."""
    if path is None or not Path(path).is_file():
        return None
    return sha256_file(Path(path))


def _verify_through_the_cli(data_root: Path, digest: str) -> dict[str, Any]:
    """Verify through the shipped CLI, so the qualification exercises the real surface.

    Deliberately a subprocess rather than an in-process call: the point is to measure what
    an operator's command does, and an in-process call would share this process's memory
    figure with the measurement itself.
    """
    # A fixed argv with no shell: the arguments are this script's own, and a shell would
    # only add a way for a data-root path to be reinterpreted.
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "psd",
            "openpowerlifting",
            "verify",
            "--digest",
            digest,
            "--data-root",
            str(data_root),
            "--json",
        ],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        check=False,
    )
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError:
        return {
            "exit_code": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------


def main() -> int:
    """Run the qualification and write its JSON report."""
    parser = argparse.ArgumentParser(description="Qualify a full PSD-COMP corpus build.")
    parser.add_argument("--data-root", required=True, help="External PSD data root.")
    parser.add_argument("--digest", required=True, help="Pinned snapshot digest.")
    parser.add_argument("--report", required=True, help="Where to write the JSON report.")
    parser.add_argument("--drop-staging", action="store_true", help="Delete staged rows after.")
    parser.add_argument(
        "--skip-second-build",
        action="store_true",
        help="Build once only. Reproducibility is then unproven and the report says so.",
    )
    parser.add_argument("--memory-limit", default="4GB", help="DuckDB ceiling for the audit.")
    arguments = parser.parse_args()

    os.environ["PSD_DATA_ROOT"] = str(Path(arguments.data_root))
    data_root = Path(arguments.data_root)
    digest = arguments.digest
    relative = f"canonical/psd_comp/{digest}"
    started_at = datetime.now(tz=UTC)

    with PeakMemory() as memory:
        recorder = Recorder(memory)
        snapshot = recorder.run(
            "read-snapshot",
            lambda: read_pinned_snapshot(
                snapshot_directory(digest, data_root=data_root), data_root=data_root
            ),
        )
        csv_path = recorder.run(
            "resolve-csv", lambda: resolve_snapshot_csv(snapshot, data_root=data_root)
        )

        def _build_once() -> Any:
            return build_corpus(
                BuildRequest(
                    csv_path=csv_path,
                    snapshot=snapshot,
                    data_root=data_root,
                    config=BuildConfig(),
                    keep_staging=not arguments.drop_staging,
                )
            )

        first = recorder.run("build-1", _build_once, source_rows=snapshot.row_count)
        first_manifest = json.loads(
            (data_root / relative / "manifest.json").read_text(encoding="utf-8")
        )

        def _histories() -> Any:
            return build_athlete_history(
                HistoryRequest(dataset_dir=Path(relative), data_root=data_root)
            )

        def _audit() -> Any:
            return audit_corpus(
                AuditRequest(
                    dataset_dir=Path(relative),
                    data_root=data_root,
                    archive_sha256=digest,
                    memory_limit=arguments.memory_limit,
                )
            )

        history = recorder.run("athlete-history", _histories)
        audit = recorder.run("audit", _audit)

        second: Any = None
        second_manifest: dict[str, Any] | None = None
        if not arguments.skip_second_build:
            second = recorder.run("build-2", _build_once, source_rows=snapshot.row_count)
            second_manifest = json.loads(
                (data_root / relative / "manifest.json").read_text(encoding="utf-8")
            )
            recorder.run("athlete-history-2", _histories)
            recorder.run("audit-2", _audit)

        # Outside the measured phases: a subprocess gets its own peak, and folding it in
        # would attribute another interpreter's memory to the build.
        verify_started = time.perf_counter()
        verification = _verify_through_the_cli(data_root, digest)
        verify_seconds = time.perf_counter() - verify_started
        recorder.phases.append(
            Phase(
                name="verify-cli",
                seconds=verify_seconds,
                peak_rss_bytes=0,
                details={
                    "note": (
                        "A subprocess: its memory is its own and is deliberately outside "
                        "this process's peak figure."
                    )
                },
            )
        )

    canonical_bytes, canonical_files = directory_bytes(data_root / relative / "tables")
    derived_bytes, derived_files = directory_bytes(data_root / relative / "derived")
    audit_bytes, audit_files = directory_bytes(data_root / relative / "audit")
    staging_bytes, staging_files = directory_bytes(data_root / "processed" / "psd_comp" / digest)

    report: dict[str, Any] = {
        "generated_at": started_at.isoformat(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "cpu_count": os.cpu_count(),
        },
        "memory_measurement": {
            "mechanism": memory.mechanism,
            "sample_interval_seconds": (
                SAMPLE_INTERVAL_SECONDS if memory.mechanism.startswith("sampled") else None
            ),
            "note": (
                "Kernel high-water mark, exact, this process only."
                if not memory.mechanism.startswith("sampled")
                else "Sampled; the true peak lies within one interval above the figure."
            ),
        },
        "source": {
            "archive_sha256": snapshot.archive_sha256,
            "archive_bytes": snapshot.archive_byte_size,
            "csv_sha256": snapshot.csv_sha256,
            "csv_bytes": snapshot.csv_byte_size,
            "csv_member_name": snapshot.csv_member_name,
            "row_count": snapshot.row_count,
            "service": snapshot.service.model_dump(mode="json"),
            "downloaded_at": snapshot.downloaded_at.isoformat(),
            "code_commit_at_acquisition": snapshot.code_commit,
            "column_count": len(snapshot.source_columns),
        },
        "counts": {
            "source_rows": first.counters.source_rows,
            "canonical": first.row_counts,
            "counters": first.counters.to_dict(),
            "athlete_history_rows": history.manifest.row_count,
        },
        "sizes_bytes": {
            "source_zip": snapshot.archive_byte_size,
            "source_csv": snapshot.csv_byte_size,
            "canonical_tables": canonical_bytes,
            "canonical_files": canonical_files,
            "derived": derived_bytes,
            "derived_files": derived_files,
            "audit": audit_bytes,
            "audit_files": audit_files,
            "staging": staging_bytes,
            "staging_files": staging_files,
        },
        "digests": {
            "build_1": _artifact_digests(first_manifest),
            "athlete_history": {
                "sha256": history.manifest.artifact_sha256,
                "content_sha256": history.manifest.artifact_content_sha256,
                "row_count": history.manifest.row_count,
            },
            "audit_report_sha256": _sha256_file(audit.json_path),
        },
        "verification": verification,
        "audit_summary": audit.audit.summary(),
        "audit_failed_invariants": [dict(item) for item in audit.audit.failed_invariants],
        "audit_entities": {
            "athletes": audit.audit.entities.athletes,
            "competitions": audit.audit.entities.competitions,
            "meets": audit.audit.entities.meets,
            "attempts": audit.audit.entities.attempts,
            "reported_results": audit.audit.entities.reported_results,
            "competitions_with_attempts": audit.audit.entities.competitions_with_attempts,
            "competitions_with_reported_total": (
                audit.audit.entities.competitions_with_reported_total
            ),
            "duplicate_primary_key_rows": dict(audit.audit.entities.duplicate_primary_key_rows),
        },
        "audit_longitudinal": {
            "athletes_with_one_meet": audit.audit.longitudinal.athletes_with_one_meet,
            "athletes_with_two_to_four_meets": (
                audit.audit.longitudinal.athletes_with_two_to_four_meets
            ),
            "athletes_with_five_to_nine_meets": (
                audit.audit.longitudinal.athletes_with_five_to_nine_meets
            ),
            "athletes_with_ten_or_more_meets": (
                audit.audit.longitudinal.athletes_with_ten_or_more_meets
            ),
            "max_meet_count": audit.audit.longitudinal.max_meet_count,
            "meet_count_distribution": dict(audit.audit.longitudinal.meet_count_distribution),
            "observed_span_days_distribution": dict(
                audit.audit.longitudinal.observed_span_days_distribution
            ),
        },
        "audit_attempts": {
            "attempt_detail_fraction": audit.audit.attempts.attempt_detail_fraction,
            "failed_attempts": audit.audit.attempts.failed_attempts,
            "failed_attempt_rate": audit.audit.attempts.failed_attempt_rate,
            "fourth_attempts": audit.audit.attempts.fourth_attempts,
            "fourth_attempt_rate": audit.audit.attempts.fourth_attempt_rate,
            "lifts_without_any_attempt": audit.audit.attempts.lifts_without_any_attempt,
        },
        "audit_performance": {
            "negative_reported_bests": audit.audit.performance.negative_reported_bests,
            "totals": audit.audit.performance.totals,
            "totals_without_all_component_bests": (
                audit.audit.performance.totals_without_all_component_bests
            ),
            "total_sum_agreements": audit.audit.performance.total_sum_agreements,
            "total_sum_disagreements": audit.audit.performance.total_sum_disagreements,
            "reported_results_marked_derived": (
                audit.audit.performance.reported_results_marked_derived
            ),
            "disagreement_examples": [
                dict(item) for item in audit.audit.performance.disagreement_examples[:5]
            ],
        },
        "audit_anomalies": {
            "disambiguated_identities": audit.audit.anomalies.disambiguated_identities,
            "sex_category_conflicts": audit.audit.anomalies.sex_category_conflicts,
            "sex_categories_absent": audit.audit.anomalies.sex_categories_absent,
            "meet_identity_collisions": audit.audit.anomalies.meet_identity_collisions,
            "competitions_without_a_meet_date": (
                audit.audit.anomalies.competitions_without_a_meet_date
            ),
            "competitions_disagreeing_with_their_meet": (
                audit.audit.anomalies.competitions_disagreeing_with_their_meet
            ),
            "earliest_meet_date": audit.audit.anomalies.earliest_meet_date,
            "latest_meet_date": audit.audit.anomalies.latest_meet_date,
            "unknown_place_codes": audit.audit.anomalies.unknown_place_codes,
            "unknown_place_examples": list(audit.audit.anomalies.unknown_place_examples[:20]),
            "unexpected_event_values": list(audit.audit.anomalies.unexpected_event_values[:20]),
            "unexpected_equipment_values": list(
                audit.audit.anomalies.unexpected_equipment_values[:20]
            ),
            "unexpected_sex_values": list(audit.audit.anomalies.unexpected_sex_values[:20]),
            "source_disagreements": list(audit.audit.anomalies.source_disagreements),
        },
        "phases": [phase.to_dict() for phase in recorder.phases],
        "peak_rss_bytes": memory.peak_bytes,
        "peak_rss_mib": round(memory.peak_bytes / (1024 * 1024), 1),
        "total_seconds": round(sum(phase.seconds for phase in recorder.phases), 3),
    }

    if second is not None and second_manifest is not None:
        report["determinism"] = {
            "second_build_run": True,
            "identical_content_digests": _artifact_digests(first_manifest)
            == _artifact_digests(second_manifest),
            "identical_byte_digests": [a["sha256"] for a in first_manifest["artifacts"]]
            == [a["sha256"] for a in second_manifest["artifacts"]],
            "identical_row_counts": first.row_counts == second.row_counts,
            "identical_counters": first.counters.to_dict() == second.counters.to_dict(),
            "identical_manifest_excluding_run_metadata": _manifest_digest_ignoring(first_manifest)
            == _manifest_digest_ignoring(second_manifest),
            "second_build_content_digests": _artifact_digests(second_manifest),
        }
    else:
        report["determinism"] = {
            "second_build_run": False,
            "note": "Only one build ran, so reproducibility is not demonstrated.",
        }

    report_path = Path(arguments.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(report, indent=2, sort_keys=True, default=str) + "\n"
    report_path.write_text(rendered, encoding="utf-8")
    sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
