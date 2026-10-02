"""End-to-end CLI tests for ``psd openpowerlifting``.

The commands are orchestration, so what matters is not that they contain the right logic --
it is that they reach the importable API, that they fail loudly, and that a pipeline can
trust their exit codes. Every test drives the real CLI through ``CliRunner`` against a real
fixture corpus.

No test touches the network. ``acquire`` is exercised through ``--source``, which is the
documented offline path, and the download path is not reachable without an explicit URL.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from psd.cli.main import app as root_app
from psd.ingest.openpowerlifting.acquire import read_pinned_snapshot, snapshot_directory
from psd.ingest.openpowerlifting.audit import AUDIT_VERSION
from psd.ingest.openpowerlifting.contract import expected_columns
from psd.paths import DATA_ROOT_ENV_VAR
from tests.fixtures.openpowerlifting import SAMPLE_ROW_COUNT, write_sample_csv

runner = CliRunner()

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE = 2


@pytest.fixture(autouse=True)
def data_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point every CLI test at an isolated data root."""
    root = tmp_path / "data"
    root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv(DATA_ROOT_ENV_VAR, str(root))
    return root


@pytest.fixture
def source_csv(tmp_path: Path) -> Path:
    """The fixture CSV as a local file, ready to pin."""
    return write_sample_csv(tmp_path / "openpowerlifting-sample.csv")


def _run(*args: str) -> Any:
    """Invoke the root CLI with *args*."""
    return runner.invoke(root_app, list(args))


def _digest(data_root: Path) -> str:
    """Return the digest the fixture snapshot was pinned under."""
    pinned = sorted((data_root / "external" / "openpowerlifting").glob("*"))
    assert pinned, "the fixture snapshot was not pinned"
    return pinned[0].name


def _build(data_root: Path, digest: str, *extra: str) -> Any:
    """Pin and build the fixture corpus through the CLI."""
    return _run(
        "openpowerlifting", "build", "--digest", digest, "--data-root", str(data_root), *extra
    )


# ---------------------------------------------------------------------------
# The surface itself
# ---------------------------------------------------------------------------


def test_the_corpus_workflow_is_reachable_from_the_root() -> None:
    """The five commands exist, and they are one group rather than five root commands."""
    result = _run("openpowerlifting", "--help")

    assert result.exit_code == EXIT_OK
    for command in ("acquire", "inspect", "build", "audit", "verify"):
        assert command in result.output


def test_the_workflow_appears_in_the_root_help() -> None:
    result = _run("--help")

    assert result.exit_code == EXIT_OK
    assert "openpowerlifting" in result.output


def test_every_command_explains_itself() -> None:
    for command in ("acquire", "inspect", "build", "audit", "verify"):
        result = _run("openpowerlifting", command, "--help")

        assert result.exit_code == EXIT_OK, command
        assert len(result.output.strip()) > 200, command


# ---------------------------------------------------------------------------
# acquire / inspect
# ---------------------------------------------------------------------------


def test_acquire_pins_a_local_source_offline(data_root: Path, source_csv: Path) -> None:
    """The offline path is the one tests and qualification use, and it needs no network."""
    result = _run(
        "openpowerlifting", "acquire", "--source", str(source_csv), "--data-root", str(data_root)
    )

    assert result.exit_code == EXIT_OK, result.output
    assert "archive_sha256" in result.output
    assert _digest(data_root) in result.output


def test_acquire_reports_the_snapshot_as_json(data_root: Path, source_csv: Path) -> None:
    result = _run(
        "openpowerlifting",
        "acquire",
        "--source",
        str(source_csv),
        "--data-root",
        str(data_root),
        "--json",
    )
    payload = json.loads(result.output)

    assert result.exit_code == EXIT_OK
    assert payload["row_count"] == SAMPLE_ROW_COUNT
    assert payload["source_columns"] == list(expected_columns())
    assert payload["schema_review"]["drifted"] is False
    assert len(payload["archive_sha256"]) == 64


def test_acquire_exits_non_zero_when_the_source_is_missing(data_root: Path) -> None:
    result = _run(
        "openpowerlifting",
        "acquire",
        "--source",
        str(data_root / "absent.csv"),
        "--data-root",
        str(data_root),
    )

    assert result.exit_code != EXIT_OK
    assert "No local OpenPowerlifting file" in result.output


def test_inspect_exposes_the_pinned_snapshot_identity(data_root: Path, source_csv: Path) -> None:
    """inspect is what a reader runs to learn which snapshot a corpus came from."""
    _run("openpowerlifting", "acquire", "--source", str(source_csv), "--data-root", str(data_root))
    digest = _digest(data_root)

    result = _run("openpowerlifting", "inspect", "--digest", digest, "--data-root", str(data_root))

    assert result.exit_code == EXIT_OK, result.output
    assert digest in result.output
    assert "public-domain" not in result.output  # licensing is in the manifest, not here
    assert "42 declared" in result.output
    assert "rows            13" in result.output


def test_inspect_needs_a_snapshot_to_name(data_root: Path) -> None:
    """A usage error is exit 2, not exit 1: nothing was attempted and nothing failed."""
    result = _run("openpowerlifting", "inspect", "--data-root", str(data_root))

    assert result.exit_code == EXIT_USAGE
    assert "--digest" in result.output


def test_inspect_refuses_an_unpinned_digest(data_root: Path) -> None:
    result = _run(
        "openpowerlifting", "inspect", "--digest", "a" * 64, "--data-root", str(data_root)
    )

    assert result.exit_code == EXIT_USAGE
    assert "No pinned snapshot metadata" in result.output


def test_inspect_exits_non_zero_on_schema_drift(data_root: Path, tmp_path: Path) -> None:
    """Drift is the one thing worth discovering before a multi-hour build.

    The drift is injected through the snapshot's recorded header, because the header is
    what the contract compares against and it is what a snapshot record carries.
    """
    csv_path = write_sample_csv(tmp_path / "drifted.csv")
    _run("openpowerlifting", "acquire", "--source", str(csv_path), "--data-root", str(data_root))
    digest = _digest(data_root)
    snapshot = read_pinned_snapshot(
        snapshot_directory(digest, data_root=data_root), data_root=data_root
    )
    drifted = snapshot.model_copy(update={"source_columns": (*snapshot.source_columns, "Squat5Kg")})
    (snapshot_directory(digest, data_root=data_root) / "snapshot.json").write_bytes(
        (json.dumps(drifted.model_dump(mode="json"), indent=2, sort_keys=True) + "\n").encode()
    )

    result = _run("openpowerlifting", "inspect", "--digest", digest, "--data-root", str(data_root))

    assert result.exit_code == EXIT_FAILED
    assert "DRIFT" in result.output
    assert "Squat5Kg" in result.output


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------


def test_build_produces_the_corpus_and_its_histories(data_root: Path, source_csv: Path) -> None:
    _run("openpowerlifting", "acquire", "--source", str(source_csv), "--data-root", str(data_root))
    digest = _digest(data_root)

    result = _build(data_root, digest, "--json")
    payload = json.loads(result.output)

    assert result.exit_code == EXIT_OK, result.output
    assert payload["archive_sha256"] == digest
    assert payload["csv_sha256"]
    assert payload["row_counts"]["competition"] == SAMPLE_ROW_COUNT
    assert payload["row_counts"]["competition_attempt"] > 0
    assert payload["athlete_history"]["rows"] == payload["row_counts"]["athlete"]
    assert len(payload["athlete_history"]["content_sha256"]) == 64
    assert payload["build_seconds"] > 0


def test_build_reports_where_to_audit_and_verify(data_root: Path, source_csv: Path) -> None:
    _run("openpowerlifting", "acquire", "--source", str(source_csv), "--data-root", str(data_root))
    digest = _digest(data_root)

    result = _build(data_root, digest)

    assert result.exit_code == EXIT_OK, result.output
    assert f"psd openpowerlifting audit --digest {digest}" in result.output
    assert f"psd openpowerlifting verify --digest {digest}" in result.output
    assert "canonical/psd_comp" in result.output


def test_build_can_skip_the_histories(data_root: Path, source_csv: Path) -> None:
    _run("openpowerlifting", "acquire", "--source", str(source_csv), "--data-root", str(data_root))
    digest = _digest(data_root)

    result = _build(data_root, digest, "--skip-histories", "--json")
    payload = json.loads(result.output)

    assert result.exit_code == EXIT_OK, result.output
    assert payload["athlete_history"] is None


def test_build_is_reproducible_through_the_cli(data_root: Path, source_csv: Path) -> None:
    """Two builds of one pinned snapshot through the CLI produce identical digests."""
    _run("openpowerlifting", "acquire", "--source", str(source_csv), "--data-root", str(data_root))
    digest = _digest(data_root)

    first = json.loads(_build(data_root, digest, "--json").output)
    second = json.loads(_build(data_root, digest, "--json").output)

    assert first["content_digests"] == second["content_digests"]
    assert first["row_counts"] == second["row_counts"]


def test_build_exits_non_zero_on_a_drifted_source(data_root: Path, tmp_path: Path) -> None:
    """The build refuses drift before reading a row, so no corpus is written from it."""
    drifted = tmp_path / "drifted.csv"
    drifted.write_text(
        ",".join((*expected_columns(), "Squat5Kg"))
        + "\n"
        + ",".join([""] * (len(expected_columns()) + 1))
        + "\n",
        encoding="utf-8",
    )
    _run("openpowerlifting", "acquire", "--source", str(drifted), "--data-root", str(data_root))
    digest = _digest(data_root)

    result = _build(data_root, digest)

    assert result.exit_code == EXIT_FAILED
    assert "unknown source column" in result.output
    assert not (data_root / "canonical" / "psd_comp" / digest / "manifest.json").exists()


def test_build_refuses_a_partition_count_that_breaks_the_partition_walk(
    data_root: Path, source_csv: Path
) -> None:
    """A partition count that does not divide sixteen is a usage error, not a rounding.

    Buckets are produced by scaling the leading hex digit, which only preserves identity
    order for a divisor of sixteen. Five would silently break the property the whole
    streaming build relies on: that walking the partitions in order *is* the canonical
    order. It is caught in the configuration rather than after the staging pass.
    """
    _run("openpowerlifting", "acquire", "--source", str(source_csv), "--data-root", str(data_root))
    digest = _digest(data_root)

    result = _build(data_root, digest, "--partitions", "5")

    assert result.exit_code == EXIT_USAGE
    assert "partitions must divide the sixteen" in result.output
    assert not (data_root / "canonical" / "psd_comp" / digest / "manifest.json").exists()


def test_build_exits_non_zero_when_no_snapshot_is_pinned(data_root: Path) -> None:
    result = _run("openpowerlifting", "build", "--digest", "b" * 64, "--data-root", str(data_root))

    assert result.exit_code == EXIT_USAGE
    assert "No pinned snapshot metadata" in result.output


# ---------------------------------------------------------------------------
# audit
# ---------------------------------------------------------------------------


def test_audit_writes_both_renderings(data_root: Path, source_csv: Path) -> None:
    _run("openpowerlifting", "acquire", "--source", str(source_csv), "--data-root", str(data_root))
    digest = _digest(data_root)
    _build(data_root, digest)

    result = _run("openpowerlifting", "audit", "--digest", digest, "--data-root", str(data_root))

    assert result.exit_code == EXIT_OK, result.output
    audit_dir = data_root / "canonical" / "psd_comp" / digest / "audit"
    assert (audit_dir / "corpus_audit.json").is_file()
    assert (audit_dir / "corpus_audit.md").is_file()
    assert "canonical entities" in result.output
    assert "expansion contract" in result.output


def test_audit_json_is_the_stored_report(data_root: Path, source_csv: Path) -> None:
    _run("openpowerlifting", "acquire", "--source", str(source_csv), "--data-root", str(data_root))
    digest = _digest(data_root)
    _build(data_root, digest)

    result = _run(
        "openpowerlifting", "audit", "--digest", digest, "--data-root", str(data_root), "--json"
    )
    stored = (
        data_root / "canonical" / "psd_comp" / digest / "audit" / "corpus_audit.json"
    ).read_bytes()

    assert result.exit_code == EXIT_OK
    assert result.output.encode("utf-8") == stored
    assert json.loads(result.output)["audit_version"] == AUDIT_VERSION


def test_audit_reports_irregularities_without_failing(data_root: Path, source_csv: Path) -> None:
    """The fixture publishes a name under two sex categories; the audit reports it and
    exits zero. A failing *invariant* is a verification failure, not an audit failure."""
    _run("openpowerlifting", "acquire", "--source", str(source_csv), "--data-root", str(data_root))
    digest = _digest(data_root)
    _build(data_root, digest)

    result = _run(
        "openpowerlifting", "audit", "--digest", digest, "--data-root", str(data_root), "--json"
    )
    report = json.loads(result.output)

    assert result.exit_code == EXIT_OK
    assert report["anomalies"]["sex_category_conflicts"] == 1
    assert report["expansion"]["invariants"]
    assert all(item["holds"] for item in report["expansion"]["invariants"])


def test_audit_exits_non_zero_when_there_is_no_corpus(data_root: Path, source_csv: Path) -> None:
    _run("openpowerlifting", "acquire", "--source", str(source_csv), "--data-root", str(data_root))
    digest = _digest(data_root)

    result = _run("openpowerlifting", "audit", "--digest", digest, "--data-root", str(data_root))

    assert result.exit_code == EXIT_FAILED
    assert "No dataset manifest" in result.output


def test_audit_needs_a_digest(data_root: Path) -> None:
    result = _run("openpowerlifting", "audit", "--data-root", str(data_root))

    assert result.exit_code == EXIT_USAGE
    assert "--digest" in result.output


# ---------------------------------------------------------------------------
# verify
# ---------------------------------------------------------------------------


def test_verify_passes_on_a_freshly_built_corpus(data_root: Path, source_csv: Path) -> None:
    _run("openpowerlifting", "acquire", "--source", str(source_csv), "--data-root", str(data_root))
    digest = _digest(data_root)
    _build(data_root, digest)

    result = _run(
        "openpowerlifting", "verify", "--digest", digest, "--data-root", str(data_root), "--json"
    )
    payload = json.loads(result.output)

    assert result.exit_code == EXIT_OK, result.output
    assert payload["ok"] is True
    assert payload["source_ok"] is True
    assert payload["artifacts_ok"] is True
    assert payload["checked_artifacts"] > 0
    assert payload["athlete_history_checked"] is True
    assert payload["invariants_failed"] == 0
    assert payload["problems"] == []


def test_verify_exits_non_zero_when_an_artifact_is_tampered_with(
    data_root: Path, source_csv: Path
) -> None:
    """A corpus whose bytes were edited does not verify, whichever digest is checked."""
    _run("openpowerlifting", "acquire", "--source", str(source_csv), "--data-root", str(data_root))
    digest = _digest(data_root)
    _build(data_root, digest)
    table = data_root / "canonical" / "psd_comp" / digest / "tables" / "competition.parquet"
    table.write_bytes(table.read_bytes() + b"tampered")

    result = _run(
        "openpowerlifting", "verify", "--digest", digest, "--data-root", str(data_root), "--json"
    )
    payload = json.loads(result.output)

    assert result.exit_code == EXIT_FAILED
    assert payload["ok"] is False
    assert any("competition" in problem for problem in payload["problems"])


def test_verify_exits_non_zero_when_the_source_has_changed(
    data_root: Path, source_csv: Path
) -> None:
    """The corpus was built from these bytes; if the bytes changed, that is a finding.

    This is the check that separates "the corpus is intact" from "the corpus still means
    what it meant", and no artifact digest can make it.
    """
    _run("openpowerlifting", "acquire", "--source", str(source_csv), "--data-root", str(data_root))
    digest = _digest(data_root)
    _build(data_root, digest)
    csv = next((data_root / "external" / "openpowerlifting" / digest).rglob("*.csv"))
    csv.write_bytes(csv.read_bytes() + b"\n")

    result = _run(
        "openpowerlifting", "verify", "--digest", digest, "--data-root", str(data_root), "--json"
    )
    payload = json.loads(result.output)

    assert result.exit_code == EXIT_FAILED
    assert payload["source_ok"] is False
    assert any("pinned bytes have changed" in problem for problem in payload["problems"])


def test_verify_can_skip_the_invariants(data_root: Path, source_csv: Path) -> None:
    """Skipping is explicit and visible: the output says so rather than omitting it."""
    _run("openpowerlifting", "acquire", "--source", str(source_csv), "--data-root", str(data_root))
    digest = _digest(data_root)
    _build(data_root, digest)

    result = _run(
        "openpowerlifting",
        "verify",
        "--digest",
        digest,
        "--data-root",
        str(data_root),
        "--skip-invariants",
        "--json",
    )
    payload = json.loads(result.output)

    assert result.exit_code == EXIT_OK, result.output
    assert payload["invariants_checked"] == 0
    assert payload["ok"] is True


def test_verify_exits_non_zero_when_there_is_no_corpus(data_root: Path, source_csv: Path) -> None:
    _run("openpowerlifting", "acquire", "--source", str(source_csv), "--data-root", str(data_root))
    digest = _digest(data_root)

    result = _run("openpowerlifting", "verify", "--digest", digest, "--data-root", str(data_root))

    assert result.exit_code == EXIT_USAGE
    assert "No dataset manifest" in result.output


# ---------------------------------------------------------------------------
# Data-root boundary
# ---------------------------------------------------------------------------


def test_a_command_without_a_data_root_explains_how_to_set_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing boundary is a configuration error, and the message says how to fix it."""
    monkeypatch.delenv(DATA_ROOT_ENV_VAR, raising=False)

    result = _run("openpowerlifting", "inspect", "--digest", "a" * 64)

    assert result.exit_code == EXIT_USAGE
    assert DATA_ROOT_ENV_VAR in result.output
