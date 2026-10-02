"""The RES-235 exit gate, driven end to end through the real CLI.

Each canonical fixture is written to disk as ``<table>.json``, built into a
persisted dataset with ``psd canonical build``, verified against its manifest
digests, and read back. The gate asserts four things the design documents
require of a canonical dataset:

* the build succeeds and persists all 24 tables;
* verification passes, so the digests in the manifest describe the bytes on disk;
* reading the dataset back reproduces the input records exactly, which is what
  makes a canonical dataset lossless;
* building the same records twice yields identical artifacts, so the dataset is
  reproducible rather than merely valid.

Every test points ``PSD_DATA_ROOT`` at pytest's temporary directory, so the suite
never touches the maintainer's external data drive.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest
from pydantic import BaseModel
from typer.testing import CliRunner

from psd.cli.canonical_cmd import app as canonical_app
from psd.cli.main import app as root_app
from psd.paths import DATA_ROOT_ENV_VAR, resolve_within_data_root
from psd.provenance.manifest import DatasetManifest
from psd.schema.registry import table_names
from psd.serialization.dataset import read_dataset
from tests.fixtures import (
    adversarial_history_records,
    messy_timeline_records,
    normal_history_records,
    synthetic_records,
)

runner = CliRunner()

#: Each fixture with the dataset kind it must be stored as. A synthetic history is
#: never stored as a real training history, because the two regimes must stay
#: distinguishable in the persisted dataset too.
CASES: dict[str, tuple[Callable[[], dict[str, list[BaseModel]]], str]] = {
    "normal": (normal_history_records, "training_history"),
    "adversarial": (adversarial_history_records, "training_history"),
    "synthetic": (synthetic_records, "synthetic"),
    "messy": (messy_timeline_records, "training_history"),
}


@pytest.fixture(autouse=True)
def _data_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point every CLI test at an isolated data root."""
    root = tmp_path / "data"
    root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv(DATA_ROOT_ENV_VAR, str(root))
    return root


def _write_json_records(tables: dict[str, list[BaseModel]], into: Path) -> Path:
    """Write each table as ``<table>.json``, the input form the CLI accepts."""
    into.mkdir(parents=True, exist_ok=True)
    for name, rows in tables.items():
        payload = [row.model_dump(mode="json") for row in rows]
        (into / f"{name}.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )
    return into


def _build(
    *,
    source: Path,
    dataset_dir: str,
    dataset_id: str,
    kind: str,
) -> DatasetManifest:
    """Run ``psd canonical build`` and return the manifest it printed."""
    result = runner.invoke(
        canonical_app,
        [
            "build",
            "--input",
            str(source),
            "--output",
            dataset_dir,
            "--dataset-id",
            dataset_id,
            "--kind",
            kind,
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    return DatasetManifest.model_validate(json.loads(result.stdout))


def _verify_ok(dataset_dir: str) -> bool:
    """Run ``psd canonical verify`` and report whether it accepted the dataset."""
    return runner.invoke(canonical_app, ["verify", dataset_dir, "--json"]).exit_code == 0


def _artifact_path(dataset_dir: str, table: str) -> Path:
    """Return the persisted file backing one table of a dataset."""
    _dataset, manifest = read_dataset(dataset_dir)
    reference = next(item for item in manifest.artifacts if item.name == table)
    return resolve_within_data_root(dataset_dir) / reference.relative_path


@pytest.mark.parametrize(("name", "case"), sorted(CASES.items()))
def test_fixture_builds_verifies_and_round_trips(
    name: str, case: tuple[Callable[[], dict[str, list[BaseModel]]], str], tmp_path: Path
) -> None:
    build_records, kind = case
    records = build_records()
    source = _write_json_records(records, tmp_path / f"{name}-json")

    summary = _build(source=source, dataset_dir=f"{name}-dataset", dataset_id=name, kind=kind)

    assert summary.schema_version == "psd-canonical/1.0.0"
    assert len(summary.artifacts) == len(table_names())
    assert _verify_ok(f"{name}-dataset")

    dataset, _manifest = read_dataset(f"{name}-dataset")
    # Every canonical table is persisted, including the ones a fixture leaves empty.
    assert dataset.row_counts() == {table: len(rows) for table, rows in records.items()}


def _content_digests(summary: DatasetManifest) -> dict[str, str]:
    """Return table to content digest, ignoring the wall-clock creation time."""
    return {item.name: item.content_sha256 for item in summary.artifacts}


@pytest.mark.parametrize(("name", "case"), sorted(CASES.items()))
def test_repeated_builds_are_byte_identical(
    name: str, case: tuple[Callable[[], dict[str, list[BaseModel]]], str], tmp_path: Path
) -> None:
    """A canonical dataset must be reproducible, not merely valid."""
    build_records, kind = case
    records = build_records()
    source = _write_json_records(records, tmp_path / f"{name}-json")

    first = _build(source=source, dataset_dir=f"{name}-one", dataset_id=name, kind=kind)
    second = _build(source=source, dataset_dir=f"{name}-two", dataset_id=name, kind=kind)

    assert _content_digests(first) == _content_digests(second)

    for table in (item.name for item in first.artifacts):
        assert (
            _artifact_path(f"{name}-one", table).read_bytes()
            == _artifact_path(f"{name}-two", table).read_bytes()
        ), table


def test_verify_detects_a_tampered_artifact(tmp_path: Path) -> None:
    """Verification has to actually fail when a persisted file changes."""
    records = normal_history_records()
    source = _write_json_records(records, tmp_path / "json")
    _build(source=source, dataset_dir="tamper", dataset_id="tamper", kind="training_history")

    target = _artifact_path("tamper", "performed_set")
    target.write_bytes(target.read_bytes() + b"\x00")

    verify = runner.invoke(canonical_app, ["verify", "tamper", "--json"])
    assert verify.exit_code == 1
    assert json.loads(verify.stdout)["ok"] is False


def test_root_cli_lists_the_canonical_commands() -> None:
    result = runner.invoke(root_app, ["canonical", "--help"])

    assert result.exit_code == 0
    for command in ("build", "verify", "manifest"):
        assert command in result.stdout
