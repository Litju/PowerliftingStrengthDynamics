"""Pinned acquisition of the OpenPowerlifting bulk snapshot.

Downloading a nightly file is the least reliable step in the whole pipeline, and its
failure mode is the worst: a truncated archive treated as a complete snapshot yields
a corpus that looks fine and is quietly missing rows. Acquisition is therefore
strictly staged and never guesses::

    staging -> verified transfer -> digest -> extracted CSV -> metadata -> atomic promotion

Nothing is promoted until it is whole. An interrupted transfer leaves a ``.partial``
file at a name that says what it is, and the pinned directory's ``snapshot.json`` is
written last, so the presence of that file is proof the snapshot is complete.

Identity is the digest, never the filename
-----------------------------------------

The published name is ``openpowerlifting-latest.zip`` and it is overwritten nightly.
Snapshots live in a digest-named directory and are identified by their SHA-256. When
the remote ``latest`` changes, acquisition mints a *new* identity beside the old one
rather than replacing it: two snapshots of the same service are two different
corpora, and the earlier one stays readable.

Tests never touch the network. Every entry point accepts a local archive or CSV, so
the network path is only reachable when a caller asks for it explicitly.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

from psd.ingest.openpowerlifting.contract import read_source_header
from psd.ingest.openpowerlifting.snapshot import (
    OpenPowerliftingSnapshot,
    ServiceSnapshotFacts,
    archive_declared_facts,
    service_snapshot_facts,
)
from psd.ingest.openpowerlifting.source import OPENPOWERLIFTING_BULK_INDEX_URL
from psd.paths import resolve_within_data_root
from psd.provenance.environment import detect_git_state
from psd.serialization.parquet import sha256_file

__all__ = (
    "OPENPOWERLIFTING_BULK_URL",
    "OPENPOWERLIFTING_STAGING_DIRNAME",
    "AcquisitionError",
    "DownloadResult",
    "acquire_snapshot",
    "acquire_snapshot_from_local_file",
    "read_pinned_snapshot",
    "resolve_snapshot_csv",
    "snapshot_directory",
)

#: The official bulk download. The filename is mutable; the digest is the identity.
OPENPOWERLIFTING_BULK_URL: Final[str] = (
    "https://openpowerlifting.gitlab.io/opl-csv/files/openpowerlifting-latest.zip"
)

#: Where incomplete downloads and in-progress pins live. The name is the warning.
OPENPOWERLIFTING_STAGING_DIRNAME: Final[str] = "openpowerlifting-staging"

_SOURCE_SUBDIRECTORY: Final[str] = "openpowerlifting"
_SNAPSHOT_METADATA_NAME: Final[str] = "snapshot.json"
_DOWNLOAD_TIMEOUT_SECONDS: Final[int] = 900
_HEAD_TIMEOUT_SECONDS: Final[int] = 60
_BLOCK_SIZE: Final[int] = 1024 * 1024
_SHA256_LENGTH: Final[int] = 64
_HEX_DIGITS: Final[frozenset[str]] = frozenset("0123456789abcdef")
_USER_AGENT: Final[str] = "psd-openpowerlifting-adapter/0.1 (PSD benchmark ingestion)"


class AcquisitionError(RuntimeError):
    """Raised when a snapshot cannot be acquired or pinned safely."""


@dataclass(frozen=True, slots=True)
class DownloadResult:
    """Where a download landed and what it hashed to.

    Attributes:
        path: Absolute path of the completed download.
        sha256: SHA-256 of the downloaded bytes.
        byte_size: Number of bytes downloaded.
    """

    path: Path
    sha256: str
    byte_size: int


def _external_root(data_root: Path | None) -> Path:
    return resolve_within_data_root(Path("external") / _SOURCE_SUBDIRECTORY, data_root=data_root)


def _staging_root(data_root: Path | None) -> Path:
    root = resolve_within_data_root(Path("external"), data_root=data_root)
    return root / OPENPOWERLIFTING_STAGING_DIRNAME


def snapshot_directory(archive_sha256: str, *, data_root: Path | None = None) -> Path:
    """Return the pinned directory for one snapshot digest.

    The digest is the directory name, so two snapshots coexist and neither can
    overwrite the other.

    Raises:
        AcquisitionError: The argument is not a SHA-256 digest.
    """
    if len(archive_sha256) != _SHA256_LENGTH or not _HEX_DIGITS.issuperset(archive_sha256):
        msg = f"Not a lowercase hex SHA-256 digest: {archive_sha256!r}."
        raise AcquisitionError(msg)
    return _external_root(data_root) / archive_sha256


def _utcnow() -> datetime:
    return datetime.now(tz=UTC)


def _code_commit() -> str | None:
    return detect_git_state(Path.cwd()).commit


# --------------------------------------------------------------------------
# transfer
# --------------------------------------------------------------------------


def response_length(url: str, *, timeout: int = _HEAD_TIMEOUT_SECONDS) -> int | None:
    """Return the advertised content length, or ``None`` when unavailable."""
    try:
        request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": _USER_AGENT})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.headers.get("Content-Length")
    except (urllib.error.URLError, OSError, ValueError):
        return None
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def download_archive(
    url: str,
    destination: Path,
    *,
    timeout: int = _DOWNLOAD_TIMEOUT_SECONDS,
) -> DownloadResult:
    """Download *url* to *destination*, verifying it before declaring success.

    The transfer writes to ``<destination>.partial`` and only renames once the whole
    body arrived, the advertised length matched, and the bytes hashed. A caller that
    finds a partial file knows the download was interrupted, and an interrupted
    download can never sit at the destination path looking complete.

    Args:
        url: Source URL.
        destination: Final path for the archive.
        timeout: Per-request timeout in seconds.

    Returns:
        The completed path, its digest, and its size.

    Raises:
        AcquisitionError: The transfer failed, was interrupted, or came up short.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(destination.name + ".partial")
    digest = hashlib.sha256()
    written = 0
    try:
        request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
        with (
            urllib.request.urlopen(request, timeout=timeout) as response,
            staging.open("wb") as handle,
        ):
            while block := response.read(_BLOCK_SIZE):
                handle.write(block)
                digest.update(block)
                written += len(block)
    except (urllib.error.URLError, OSError, ValueError) as error:
        # The partial file is deliberately left behind: it is evidence of an
        # interrupted transfer and it is never at the destination path.
        msg = f"OpenPowerlifting download from {url} failed after {written} bytes: {error}"
        raise AcquisitionError(msg) from error

    if written == 0:
        msg = f"OpenPowerlifting download from {url} produced no bytes."
        raise AcquisitionError(msg)

    expected = response_length(url, timeout=timeout)
    if expected is not None and written != expected:
        msg = (
            f"OpenPowerlifting download from {url} is incomplete: received {written} bytes "
            f"but the server advertises {expected}. The partial file stays at {staging} and "
            "no snapshot was pinned."
        )
        raise AcquisitionError(msg)

    staging.replace(destination)
    return DownloadResult(path=destination, sha256=digest.hexdigest(), byte_size=written)


def _read_service_facts(url: str, *, timeout: int) -> ServiceSnapshotFacts:
    """Fetch and parse the bulk page's snapshot date and revision.

    Best effort by construction: both facts are optional in the snapshot model, and a
    failed fetch must not fail an acquisition whose real identity -- the digest -- is
    already established. Nothing is guessed from the filename or from a
    ``Last-Modified`` header when the page does not say.
    """
    try:
        request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            page = response.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, OSError, ValueError):
        return ServiceSnapshotFacts()
    return service_snapshot_facts(page)


# --------------------------------------------------------------------------
# pinning
# --------------------------------------------------------------------------


def _csv_member_for_archive(archive: Path) -> str:
    """Return the archive member holding the competition data.

    The member name is *not* derivable from the URL: the published archive nests a
    dated, revision-suffixed CSV inside a matching folder, e.g.
    ``openpowerlifting-2026-09-26/openpowerlifting-2026-09-26-f231b4f6.csv``. So the
    archive is asked rather than the URL guessed from, and the name it gives is the
    name that gets recorded.

    Exactly one CSV is expected. More than one means the packaging changed, and
    picking one would be exactly the silent substitution of the corpus this pipeline
    exists to prevent.
    """
    try:
        with zipfile.ZipFile(archive) as bundle:
            members = [name for name in bundle.namelist() if not name.endswith("/")]
    except (OSError, zipfile.BadZipFile) as error:
        msg = f"{archive} is not a readable ZIP archive: {error}"
        raise AcquisitionError(msg) from error
    csv_members = sorted(name for name in members if name.lower().endswith(".csv"))
    if not csv_members:
        msg = f"{archive} contains no CSV member; members were {sorted(members)[:5]}."
        raise AcquisitionError(msg)
    if len(csv_members) > 1:
        msg = (
            f"{archive} contains {len(csv_members)} CSV members ({csv_members}). PSD will "
            "not guess which one is the competition corpus."
        )
        raise AcquisitionError(msg)
    return csv_members[0]


def _extract_member(archive: Path, member: str, destination: Path) -> tuple[str, int]:
    """Extract one member, returning its digest and byte size.

    The digest is computed over the *extracted* bytes rather than copied from archive
    metadata, so the recorded value is a fact about the file PSD will read.
    """
    digest = hashlib.sha256()
    written = 0
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(destination.name + ".partial")
    try:
        with (
            zipfile.ZipFile(archive) as bundle,
            bundle.open(member) as source,
            staging.open("wb") as handle,
        ):
            while block := source.read(_BLOCK_SIZE):
                handle.write(block)
                digest.update(block)
                written += len(block)
    except (OSError, KeyError, zipfile.BadZipFile) as error:
        staging.unlink(missing_ok=True)
        msg = f"Could not extract {member!r} from {archive}: {error}"
        raise AcquisitionError(msg) from error
    if written == 0:
        staging.unlink(missing_ok=True)
        msg = f"{member!r} in {archive} is empty."
        raise AcquisitionError(msg)
    staging.replace(destination)
    return digest.hexdigest(), written


def count_data_rows(csv_path: Path) -> int:
    """Count data rows in a CSV, excluding the header.

    Counted from the same bytes the digest is computed over, so the two describe one
    read. The source forbids quoting and in-field commas, so a newline is exactly a
    record boundary and counting them is exact rather than approximate.
    """
    newlines = 0
    trailing_newline = True
    with csv_path.open("rb") as handle:
        while block := handle.read(_BLOCK_SIZE):
            newlines += block.count(b"\n")
            trailing_newline = block.endswith(b"\n")
    if not trailing_newline:
        newlines += 1
    return max(0, newlines - 1)


@dataclass(frozen=True, slots=True)
class PinRequest:
    """One archive waiting to be verified and pinned.

    Attributes:
        archive: Path of the archive to pin. It is moved into staging, so the caller
            must not still need it afterwards.
        archive_sha256: Digest of the archive bytes.
        archive_byte_size: Size of the archive in bytes.
        source_url: Where the archive came from, recorded for provenance.
        service: What the bulk page reported about itself, when known.
        keep_archive: Whether to keep the raw archive in the pinned directory.
    """

    archive: Path
    archive_sha256: str
    archive_byte_size: int
    source_url: str
    service: ServiceSnapshotFacts
    keep_archive: bool = True


def _with_archive_facts(service: ServiceSnapshotFacts, member: str) -> ServiceSnapshotFacts:
    """Fold the archive's own dated member name into the service facts.

    The page statement and the archive statement are kept side by side so a
    disagreement is visible. Neither overwrites the other, and neither is dropped.
    """
    declared_date, declared_revision = archive_declared_facts(member)
    return service.model_copy(
        update={
            "archive_declared_date": declared_date,
            "archive_declared_revision": declared_revision,
        }
    )


def _pin_archive(request: PinRequest, *, data_root: Path | None) -> OpenPowerliftingSnapshot:
    """Verify, extract, describe, and atomically promote one archive."""
    target = snapshot_directory(request.archive_sha256, data_root=data_root)
    if (target / _SNAPSHOT_METADATA_NAME).is_file():
        # Idempotent: identical bytes are already pinned, so reuse that identity
        # rather than writing a second copy.
        return read_pinned_snapshot(target)

    staging = _staging_root(data_root) / request.archive_sha256
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=True)
    try:
        staged_archive = staging / request.archive.name
        shutil.move(str(request.archive), str(staged_archive))
        member = _csv_member_for_archive(staged_archive)
        csv_sha256, csv_byte_size = _extract_member(staged_archive, member, staging / member)
        csv_path = staging / member
        snapshot = OpenPowerliftingSnapshot(
            source_url=request.source_url,
            downloaded_at=_utcnow(),
            service=_with_archive_facts(request.service, member),
            archive_sha256=request.archive_sha256,
            archive_byte_size=request.archive_byte_size,
            csv_member_name=member,
            csv_sha256=csv_sha256,
            csv_byte_size=csv_byte_size,
            row_count=count_data_rows(csv_path),
            source_columns=read_source_header(csv_path),
            code_commit=_code_commit(),
        )
        if not request.keep_archive:
            staged_archive.unlink(missing_ok=True)
        # Metadata last: until it exists, the directory is not a pinned snapshot.
        (staging / _SNAPSHOT_METADATA_NAME).write_bytes(_snapshot_json_bytes(snapshot))
        _promote(staging, target)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return snapshot


def _pin_bare_csv(
    csv_path: Path,
    *,
    source_url: str,
    service: ServiceSnapshotFacts,
    data_root: Path | None,
) -> OpenPowerliftingSnapshot:
    """Pin a CSV that was never packaged in an archive.

    The CSV is its own source artifact, so its digest is the snapshot identity. The
    snapshot model stays uniform: ``archive_sha256`` names the ingested file and
    ``csv_sha256`` the data read from it, and here they are the same bytes.
    """
    digest = sha256_file(csv_path)
    target = snapshot_directory(digest, data_root=data_root)
    if (target / _SNAPSHOT_METADATA_NAME).is_file():
        return read_pinned_snapshot(target)
    staging = _staging_root(data_root) / digest
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=True)
    try:
        staged_csv = staging / csv_path.name
        shutil.copy2(csv_path, staged_csv)
        snapshot = OpenPowerliftingSnapshot(
            source_url=source_url,
            downloaded_at=_utcnow(),
            service=service,
            archive_sha256=digest,
            archive_byte_size=staged_csv.stat().st_size,
            csv_member_name=csv_path.name,
            csv_sha256=digest,
            csv_byte_size=staged_csv.stat().st_size,
            row_count=count_data_rows(staged_csv),
            source_columns=read_source_header(staged_csv),
            code_commit=_code_commit(),
        )
        (staging / _SNAPSHOT_METADATA_NAME).write_bytes(_snapshot_json_bytes(snapshot))
        _promote(staging, target)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return snapshot


def _promote(staging: Path, target: Path) -> None:
    """Move a fully verified staging directory into its pinned location.

    The rename is the last step, so a failure anywhere before it leaves nothing that
    could be mistaken for a pinned snapshot.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        shutil.rmtree(target)
    try:
        staging.replace(target)
    except OSError:
        # A cross-volume rename is not atomic. Fall back to copying into a temporary
        # sibling and renaming that, which still never exposes a partially populated
        # directory at the digest path.
        temporary = target.with_name(target.name + ".promoting")
        if temporary.exists():
            shutil.rmtree(temporary)
        shutil.copytree(staging, temporary)
        temporary.replace(target)


def _snapshot_json_bytes(snapshot: OpenPowerliftingSnapshot) -> bytes:
    payload = snapshot.model_dump(mode="json")
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


# --------------------------------------------------------------------------
# entry points
# --------------------------------------------------------------------------


def acquire_snapshot(
    *,
    data_root: Path | None = None,
    url: str = OPENPOWERLIFTING_BULK_URL,
    service_index_url: str = OPENPOWERLIFTING_BULK_INDEX_URL,
    timeout: int = _DOWNLOAD_TIMEOUT_SECONDS,
) -> OpenPowerliftingSnapshot:
    """Download and pin the current bulk snapshot.

    Args:
        data_root: External PSD data root; resolved from ``PSD_DATA_ROOT`` otherwise.
        url: Bulk download URL.
        service_index_url: Page consulted for the service-reported snapshot date and
            revision.
        timeout: Per-request timeout in seconds.

    Returns:
        The pinned snapshot. Re-running against unchanged bytes returns the existing
        snapshot without writing a second copy.
    """
    incoming = _staging_root(data_root) / "incoming"
    destination = incoming / url.rsplit("/", maxsplit=1)[-1]
    if not destination.name.endswith(".zip"):
        msg = f"Expected a .zip bulk URL; got {url!r}."
        raise AcquisitionError(msg)
    downloaded = download_archive(url, destination, timeout=timeout)
    service = _read_service_facts(service_index_url, timeout=timeout)
    return _pin_archive(
        PinRequest(
            archive=downloaded.path,
            archive_sha256=downloaded.sha256,
            archive_byte_size=downloaded.byte_size,
            source_url=url,
            service=service,
        ),
        data_root=data_root,
    )


def acquire_snapshot_from_local_file(
    path: Path,
    *,
    data_root: Path | None = None,
    service: ServiceSnapshotFacts | None = None,
) -> OpenPowerliftingSnapshot:
    """Pin a snapshot from a local ``.zip`` or ``.csv``, without touching the network.

    This is the path tests and offline qualification use.

    Args:
        path: Local archive or CSV.
        data_root: External PSD data root.
        service: Service-reported facts, when the caller knows them.

    Returns:
        The pinned snapshot.

    Raises:
        AcquisitionError: The input is missing, empty, or not a readable archive.
    """
    if not path.is_file():
        msg = f"No local OpenPowerlifting file at {path}."
        raise AcquisitionError(msg)
    facts = service if service is not None else ServiceSnapshotFacts()
    if path.suffix.lower() != ".zip":
        return _pin_bare_csv(path, source_url=path.name, service=facts, data_root=data_root)
    digest = sha256_file(path)
    size = path.stat().st_size
    if size == 0:
        msg = f"{path} is empty; no snapshot was pinned."
        raise AcquisitionError(msg)
    return _pin_archive(
        PinRequest(
            archive=path,
            archive_sha256=digest,
            archive_byte_size=size,
            source_url=path.name,
            service=facts,
        ),
        data_root=data_root,
    )


def read_pinned_snapshot(
    directory: Path | None = None,
    *,
    data_root: Path | None = None,
    archive_sha256: str | None = None,
) -> OpenPowerliftingSnapshot:
    """Read a pinned snapshot's metadata.

    Args:
        directory: Snapshot directory; resolved from ``archive_sha256`` otherwise.
        data_root: External PSD data root.
        archive_sha256: Snapshot digest, when the directory is not given.

    Returns:
        The pinned snapshot.

    Raises:
        AcquisitionError: No snapshot metadata is present at the location.
    """
    target = directory
    if target is None:
        if archive_sha256 is None:
            msg = "Either a snapshot directory or an archive digest is required."
            raise AcquisitionError(msg)
        target = snapshot_directory(archive_sha256, data_root=data_root)
    metadata = Path(target) / _SNAPSHOT_METADATA_NAME
    if not metadata.is_file():
        msg = (
            f"No pinned snapshot metadata at {metadata}. A snapshot directory only becomes "
            "a snapshot once its metadata exists, so this is an incomplete or absent "
            "acquisition rather than a snapshot to fall back on."
        )
        raise AcquisitionError(msg)
    try:
        payload = json.loads(metadata.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        msg = f"Unreadable snapshot metadata at {metadata}: {error}"
        raise AcquisitionError(msg) from error
    return OpenPowerliftingSnapshot.model_validate(payload)


def resolve_snapshot_csv(
    snapshot: OpenPowerliftingSnapshot, *, data_root: Path | None = None
) -> Path:
    """Return the path of the extracted CSV for a pinned snapshot.

    Raises:
        AcquisitionError: The pinned snapshot has no extracted CSV, which means the pin
            is incomplete and reading rows from it is refused.
    """
    directory = snapshot_directory(snapshot.archive_sha256, data_root=data_root)
    csv_path = directory / snapshot.csv_member_name
    if not csv_path.is_file():
        msg = (
            f"Pinned snapshot {snapshot.archive_sha256[:16]} has no extracted CSV at "
            f"{csv_path}. Re-acquire the snapshot; do not read rows from a partial pin."
        )
        raise AcquisitionError(msg)
    return csv_path
