"""Tests for digest-pinned acquisition of the OpenPowerlifting snapshot.

No test here touches the network. Acquisition is exercised through its local-file
path plus the transfer helper's failure handling, because the properties that matter
are all local: nothing incomplete is ever promoted, identity is the digest, and an
unchanged re-run does not write a second copy.
"""

from __future__ import annotations

import hashlib
import io
import urllib.error
import urllib.request
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import ClassVar

import pytest

from psd.ingest.openpowerlifting import acquire as acquire_module
from psd.ingest.openpowerlifting.acquire import (
    OPENPOWERLIFTING_BULK_URL,
    AcquisitionError,
    acquire_snapshot,
    acquire_snapshot_from_local_file,
    count_data_rows,
    read_pinned_snapshot,
    resolve_snapshot_csv,
    snapshot_directory,
)
from psd.ingest.openpowerlifting.contract import expected_columns
from psd.ingest.openpowerlifting.snapshot import (
    OpenPowerliftingSnapshot,
    ServiceSnapshotFacts,
    service_snapshot_facts,
)
from psd.ingest.openpowerlifting.source import openpowerlifting_source_id

CSV_NAME = "openpowerlifting-latest.csv"
ARCHIVE_NAME = "openpowerlifting-latest.zip"

#: The truncated-transfer fixture advertises this many bytes and delivers far fewer.
_ADVERTISED_LENGTH = 4096
_CHUNK = 16

SAMPLE_ROWS: tuple[str, ...] = (
    "John Doe#1,M,SBD,Raw,23,23-39,,Open,91.4,-93,180,120,200,185,125,210,187.5,127.5,215,"
    "187.5,127.5,215,532.5,1,,,,,,,,USA,TX,USPA,IPF,2025-11-08,USA,TX,Raw Open,Yes",
    "Jane Roe,F,BD,Wraps,31.5,30-39,,Open,62.5,-63,,-100,,110,,,125,100,125,,125,100,225,2,"
    ",,,,GBR,London,USPA,,2025-11-08,USA,TX,Raw Open,Yes",
    "Kim Lee,F,B,Single-ply,,,,-69,,,,,,,,,,,70,70,70,,70,,70,,,,,,,60,30,60,1,,,,,,,,USA,NY,"
    "USPA,,2025-11-08,USA,TX,Raw Open,Yes",
)


def _write_csv(path: Path, rows: tuple[str, ...] = SAMPLE_ROWS) -> Path:
    header = ",".join(expected_columns())
    path.write_text(header + "\n" + "\n".join(rows) + "\n", encoding="utf-8")
    return path


def _write_archive(directory: Path, rows: tuple[str, ...] = SAMPLE_ROWS) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    csv_bytes = _write_csv(directory / "source.csv", rows).read_bytes()
    archive = directory / ARCHIVE_NAME
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr(CSV_NAME, csv_bytes)
    return archive


def _facts() -> ServiceSnapshotFacts:
    return ServiceSnapshotFacts(updated_date="2026-09-25", revision="f231b4f6")


def test_pinning_an_archive_records_every_identification_fact(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    archive = _write_archive(tmp_path)
    expected_digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    expected_size = archive.stat().st_size

    snapshot = acquire_snapshot_from_local_file(archive, data_root=data_root, service=_facts())

    assert snapshot.archive_sha256 == expected_digest
    assert snapshot.csv_member_name == CSV_NAME
    assert len(snapshot.csv_sha256) == 64
    assert snapshot.archive_byte_size == expected_size
    assert snapshot.csv_byte_size > 0
    assert snapshot.row_count == len(SAMPLE_ROWS)
    assert snapshot.source_columns == expected_columns()
    assert snapshot.service.updated_date == "2026-09-25"
    assert snapshot.service.revision == "f231b4f6"
    assert snapshot.downloaded_at.tzinfo is not None


def test_pinning_consumes_the_caller_s_archive(tmp_path: Path) -> None:
    """Acquisition takes ownership: the local file is moved into the pinned snapshot."""
    data_root = tmp_path / "data"
    data_root.mkdir()
    archive = _write_archive(tmp_path)
    acquire_snapshot_from_local_file(archive, data_root=data_root)
    assert not archive.exists()
    pinned = snapshot_directory(hashlib.sha256(b"").hexdigest(), data_root=data_root)
    assert not pinned.exists()


def test_pinning_an_archive_keeps_the_csv_and_the_metadata(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    snapshot = acquire_snapshot_from_local_file(
        _write_archive(tmp_path), data_root=data_root, service=_facts()
    )

    csv_path = resolve_snapshot_csv(snapshot, data_root=data_root)
    assert csv_path.is_file()
    assert csv_path.parent == snapshot_directory(snapshot.archive_sha256, data_root=data_root)
    reread = read_pinned_snapshot(data_root=data_root, archive_sha256=snapshot.archive_sha256)
    assert reread == snapshot


def test_the_csv_digest_describes_the_extracted_bytes(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    archive = _write_archive(tmp_path)
    snapshot = acquire_snapshot_from_local_file(archive, data_root=data_root, service=_facts())
    extracted = resolve_snapshot_csv(snapshot, data_root=data_root)

    assert snapshot.csv_sha256 == hashlib.sha256(extracted.read_bytes()).hexdigest()


def test_identity_is_the_digest_not_the_mutable_filename(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    snapshot = acquire_snapshot_from_local_file(
        _write_archive(tmp_path), data_root=data_root, service=_facts()
    )
    directory = snapshot_directory(snapshot.archive_sha256, data_root=data_root)
    assert directory.name == snapshot.archive_sha256
    assert snapshot.archive_sha256 not in ARCHIVE_NAME
    assert openpowerlifting_source_id(snapshot.archive_sha256).startswith("openpowerlifting_")


def test_identical_bytes_reuse_the_existing_pin(tmp_path: Path) -> None:
    """Re-running against an unchanged source is idempotent, not a second copy."""
    data_root = tmp_path / "data"
    data_root.mkdir()
    snapshot_one = acquire_snapshot_from_local_file(
        _write_archive(tmp_path / "a"), data_root=data_root
    )
    directory = snapshot_directory(snapshot_one.archive_sha256, data_root=data_root)
    marker = directory / "acquired-once"
    marker.write_text("x", encoding="utf-8")

    snapshot_two = acquire_snapshot_from_local_file(
        _write_archive(tmp_path / "b"), data_root=data_root
    )

    assert snapshot_two.archive_sha256 == snapshot_one.archive_sha256
    assert marker.is_file(), "an identical source re-wrote the pinned directory"


def test_a_changed_source_mints_a_new_identity_beside_the_old_one(tmp_path: Path) -> None:
    """A new nightly is a new corpus; the previous snapshot stays readable."""
    data_root = tmp_path / "data"
    data_root.mkdir()
    first_dir = tmp_path / "one"
    first_dir.mkdir()
    first = acquire_snapshot_from_local_file(_write_archive(first_dir), data_root=data_root)

    second_dir = tmp_path / "two"
    second_dir.mkdir()
    second = acquire_snapshot_from_local_file(
        _write_archive(second_dir, SAMPLE_ROWS[:1]), data_root=data_root
    )

    assert second.archive_sha256 != first.archive_sha256
    assert snapshot_directory(first.archive_sha256, data_root=data_root).is_dir()
    assert snapshot_directory(second.archive_sha256, data_root=data_root).is_dir()
    assert resolve_snapshot_csv(first, data_root=data_root).is_file()


def test_pinning_a_bare_csv_is_supported(tmp_path: Path) -> None:
    """A snapshot that was never packaged is still pinnable, digest-identified."""
    data_root = tmp_path / "data"
    data_root.mkdir()
    csv_path = _write_csv(tmp_path / CSV_NAME)
    snapshot = acquire_snapshot_from_local_file(csv_path, data_root=data_root)

    assert snapshot.csv_member_name == CSV_NAME
    assert snapshot.archive_sha256 == snapshot.csv_sha256
    assert snapshot.row_count == len(SAMPLE_ROWS)
    assert resolve_snapshot_csv(snapshot, data_root=data_root).is_file()


def test_a_missing_local_file_is_refused(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    with pytest.raises(AcquisitionError, match="No local OpenPowerlifting file"):
        acquire_snapshot_from_local_file(tmp_path / "absent.zip", data_root=data_root)


def test_an_empty_archive_is_refused(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    archive = tmp_path / ARCHIVE_NAME
    archive.write_bytes(b"")
    with pytest.raises(AcquisitionError, match="is empty"):
        acquire_snapshot_from_local_file(archive, data_root=data_root)


def test_a_corrupt_archive_is_refused(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    archive = tmp_path / ARCHIVE_NAME
    archive.write_bytes(b"this is not a zip file at all")
    with pytest.raises(AcquisitionError, match="not a readable ZIP"):
        acquire_snapshot_from_local_file(archive, data_root=data_root)


def test_an_archive_without_a_csv_is_refused(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    archive = tmp_path / ARCHIVE_NAME
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("README.txt", "no data here")
    with pytest.raises(AcquisitionError, match="contains no CSV member"):
        acquire_snapshot_from_local_file(archive, data_root=data_root)


def test_an_archive_with_two_csvs_is_refused(tmp_path: Path) -> None:
    """Picking one of two CSVs would be a silent substitution of the corpus."""
    data_root = tmp_path / "data"
    data_root.mkdir()
    archive = tmp_path / ARCHIVE_NAME
    header = ",".join(expected_columns())
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("a.csv", header + "\n")
        bundle.writestr("b.csv", header + "\n")
    with pytest.raises(AcquisitionError, match="will not guess which one"):
        acquire_snapshot_from_local_file(archive, data_root=data_root)


def test_an_empty_member_is_refused(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    archive = tmp_path / ARCHIVE_NAME
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr(CSV_NAME, "")
    with pytest.raises(AcquisitionError, match="is empty"):
        acquire_snapshot_from_local_file(archive, data_root=data_root)


def test_an_incomplete_pin_is_never_readable(tmp_path: Path) -> None:
    """A directory without metadata is not a snapshot to fall back on."""
    data_root = tmp_path / "data"
    data_root.mkdir()
    partial = snapshot_directory("a" * 64, data_root=data_root)
    partial.mkdir(parents=True)
    (partial / CSV_NAME).write_text("Name\n", encoding="utf-8")
    with pytest.raises(AcquisitionError, match="No pinned snapshot metadata"):
        read_pinned_snapshot(partial)


def test_resolving_a_missing_csv_refuses_rather_than_rebuilding(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    snapshot = acquire_snapshot_from_local_file(_write_archive(tmp_path), data_root=data_root)
    resolve_snapshot_csv(snapshot, data_root=data_root).unlink()
    with pytest.raises(AcquisitionError, match="Re-acquire the snapshot"):
        resolve_snapshot_csv(snapshot, data_root=data_root)


def test_a_snapshot_directory_must_be_a_digest(tmp_path: Path) -> None:
    with pytest.raises(AcquisitionError, match="Not a lowercase hex SHA-256"):
        snapshot_directory("not-a-digest", data_root=tmp_path)


def test_reading_a_snapshot_needs_a_digest_or_a_directory(tmp_path: Path) -> None:
    with pytest.raises(AcquisitionError, match="directory or an archive digest"):
        read_pinned_snapshot(data_root=tmp_path)


def test_row_counting_handles_a_missing_trailing_newline(tmp_path: Path) -> None:
    csv_path = tmp_path / "no-newline.csv"
    csv_path.write_text("Name,Sex\nJohn,M\nJane,F", encoding="utf-8")
    assert count_data_rows(csv_path) == 2


def test_row_counting_handles_a_trailing_newline(tmp_path: Path) -> None:
    csv_path = tmp_path / "newline.csv"
    csv_path.write_text("Name,Sex\nJohn,M\nJane,F\n", encoding="utf-8")
    assert count_data_rows(csv_path) == 2


def test_row_counting_of_a_header_only_file_is_zero(tmp_path: Path) -> None:
    csv_path = tmp_path / "header.csv"
    csv_path.write_text("Name,Sex\n", encoding="utf-8")
    assert count_data_rows(csv_path) == 0


def test_the_bulk_url_is_the_official_one() -> None:
    assert OPENPOWERLIFTING_BULK_URL == (
        "https://openpowerlifting.gitlab.io/opl-csv/files/openpowerlifting-latest.zip"
    )


def test_a_download_failure_leaves_nothing_promoted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An interrupted transfer must not be able to masquerade as a snapshot."""

    def _boom(*_args: object, **_kwargs: object) -> None:
        message = "connection reset"
        raise urllib.error.URLError(message)

    data_root = tmp_path / "data"
    data_root.mkdir()
    monkeypatch.setattr("urllib.request.urlopen", _boom)
    with pytest.raises(AcquisitionError, match="failed after"):
        acquire_snapshot(data_root=data_root)

    pinned = snapshot_directory("a" * 64, data_root=data_root)
    assert not pinned.exists()


def test_a_short_download_is_refused_rather_than_pinned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A truncated body is the exact failure this staging exists to catch."""

    class _Truncated(io.BytesIO):
        """A response body that stops well short of its advertised length."""

        headers: ClassVar[dict[str, str]] = {"Content-Length": "4096"}

        def read(self, size: int | None = -1) -> bytes:
            return super().read(min(size, _CHUNK) if size and size > 0 else _CHUNK)

    def _fake_urlopen(request: urllib.request.Request, timeout: int = 0) -> _Truncated:
        del request, timeout
        return _Truncated(b"x" * 32)

    def _advertised(url: str, timeout: int = 0) -> int:
        del url, timeout
        return _ADVERTISED_LENGTH

    data_root = tmp_path / "data"
    data_root.mkdir()
    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen)
    monkeypatch.setattr(acquire_module, "response_length", _advertised)

    with pytest.raises(AcquisitionError, match="is incomplete"):
        acquire_snapshot(data_root=data_root)

    staging = data_root / "external" / "openpowerlifting-staging"
    partials = list(staging.rglob("*.partial")) if staging.exists() else []
    assert partials, "the interrupted transfer should leave its partial file as evidence"
    assert not list(data_root.rglob("snapshot.json"))


def test_service_facts_are_parsed_from_the_bulk_page() -> None:
    page = "<html><body><p>Updated: 2026-09-25.</p><p>Revision: f231b4f6.</p></body></html>"
    facts = service_snapshot_facts(page)
    assert facts.updated_date == "2026-09-25"
    assert facts.revision == "f231b4f6"


def test_service_facts_are_absent_rather_than_guessed() -> None:
    """A page that says nothing yields nothing; a filename is never a fallback."""
    assert service_snapshot_facts("<html><body>no facts here</body></html>") == (
        ServiceSnapshotFacts()
    )


def test_snapshot_dataset_version_combines_what_the_service_reported() -> None:
    snapshot = OpenPowerliftingSnapshot(
        source_url=OPENPOWERLIFTING_BULK_URL,
        downloaded_at=datetime(2026, 10, 2, tzinfo=UTC),
        service=_facts(),
        archive_sha256="a" * 64,
        archive_byte_size=1,
        csv_member_name=CSV_NAME,
        csv_sha256="b" * 64,
        csv_byte_size=1,
        row_count=0,
        source_columns=("Name",),
    )
    assert snapshot.dataset_version() == "opl-2026-09-25-f231b4f6"
