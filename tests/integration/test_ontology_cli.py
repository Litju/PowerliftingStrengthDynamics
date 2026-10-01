"""End-to-end CLI tests for ``psd ontology``.

The commands are driven through the real Typer application so that argument parsing,
exit codes, and the data-root boundary are all exercised, and every test points
``PSD_DATA_ROOT`` at pytest's temporary directory so the suite never touches the
maintainer's external data drive.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from psd.cli.main import app as root_app
from psd.cli.ontology_cmd import app as ontology_app
from psd.ontology import default_ontology
from psd.ontology.artifact import ontology_relative_path
from psd.paths import DATA_ROOT_ENV_VAR
from psd.schema.registry import table_names

runner = CliRunner()
ONTOLOGY = default_ontology()


@pytest.fixture(autouse=True)
def _data_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "data"
    root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv(DATA_ROOT_ENV_VAR, str(root))
    return root


def test_ontology_is_reachable_from_the_root_cli() -> None:
    result = runner.invoke(root_app, ["ontology", "--help"])
    assert result.exit_code == 0, result.output
    for command in ("version", "list", "show", "resolve", "build", "coverage"):
        assert command in result.output


def test_ontology_version_reports_both_versions() -> None:
    result = runner.invoke(ontology_app, ["version"])
    assert result.exit_code == 0, result.output
    assert "psd-ontology/0.1.0" in result.output
    assert "psd-ontology-alias/0.1.0" in result.output
    assert "dataset_id             psd-ontology-0.1.0" in result.output


def test_ontology_list_shows_the_canonical_exercises() -> None:
    result = runner.invoke(ontology_app, ["list"])
    assert result.exit_code == 0, result.output
    for key in ("low_bar_squat", "close_grip_bench", "sumo_deadlift", "leg_press"):
        assert key in result.output


def test_ontology_list_can_filter_by_family() -> None:
    result = runner.invoke(ontology_app, ["list", "--parent", "squat"])
    assert result.exit_code == 0, result.output
    assert "low_bar_squat" in result.output
    assert "close_grip_bench" not in result.output


def test_ontology_list_json_is_machine_readable() -> None:
    result = runner.invoke(ontology_app, ["list", "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    entry = next(item for item in payload if item["canonical_key"] == "close_grip_bench")
    assert entry["grip"] == "close"
    assert entry["parent_lift"] == "bench"
    assert entry["specificity_level"] == "competition_variation"
    assert entry["configuration"] == ["hand_spacing_reduced"]


def test_ontology_list_rejects_an_unknown_family() -> None:
    result = runner.invoke(ontology_app, ["list", "--parent", "not_a_lift"])
    assert result.exit_code == 2
    assert "Unknown parent lift" in result.output


def test_ontology_show_reports_every_descriptor_and_alias() -> None:
    result = runner.invoke(ontology_app, ["show", "close_grip_bench"])
    assert result.exit_code == 0, result.output
    assert "Close-Grip Bench Press" in result.output
    assert "grip" in result.output
    assert "hand_spacing_reduced" in result.output
    assert "CGBP" in result.output


def test_ontology_show_json_carries_the_exercise_identifier() -> None:
    result = runner.invoke(ontology_app, ["show", "low_bar_squat", "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["canonical_key"] == "low_bar_squat"
    assert payload["exercise_id"].startswith("exd_")
    assert payload["stance"] == "low"


def test_ontology_show_rejects_an_unknown_key() -> None:
    result = runner.invoke(ontology_app, ["show", "not_an_exercise"])
    assert result.exit_code == 2
    assert "Unknown canonical exercise key" in result.output


def test_resolve_reports_a_mapping_and_exits_zero() -> None:
    result = runner.invoke(ontology_app, ["resolve", "Low Bar Squat"])
    assert result.exit_code == 0, result.output
    assert "exact_canonical" in result.output
    assert "low_bar_squat" in result.output


def test_resolve_reports_a_refusal_and_exits_nonzero() -> None:
    """An unresolved label is the report, not a failure of the command."""
    result = runner.invoke(ontology_app, ["resolve", "Machine press"])
    assert result.exit_code == 1
    assert "ambiguous" in result.output
    assert "unspecified_machine" in result.output


def test_resolve_json_exposes_the_whole_outcome() -> None:
    result = runner.invoke(ontology_app, ["resolve", "Leg Press?", "--source", "hevy", "--json"])
    assert result.exit_code == 1
    payload = json.loads(result.stdout)[0]
    assert payload["raw_label"] == "Leg Press?"
    assert payload["resolution_status"] == "ambiguous"
    assert payload["ambiguity_reason"] == "question_form_label"
    assert payload["candidate_keys"] == ["leg_press"]
    assert payload["exercise_key"] is None
    assert payload["confidence"] is None


def test_resolve_accepts_several_labels_at_once() -> None:
    result = runner.invoke(ontology_app, ["resolve", "Bench", "Machine press", "--json"])
    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert len(payload) == 2
    assert payload[0]["resolution_status"] == "exact_canonical"
    assert payload[1]["resolution_status"] == "ambiguous"


def test_resolve_reports_the_source_system_it_used() -> None:
    result = runner.invoke(ontology_app, ["resolve", "CGBP", "--source", "hevy", "--json"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)[0]["source_system"] == "hevy"


def test_coverage_summarises_resolutions_and_refusals() -> None:
    result = runner.invoke(ontology_app, ["coverage", "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["aliases"] >= 100
    assert sum(payload["alias_outcomes"].values()) == payload["aliases"]
    assert payload["curated_refusals"] >= 1
    assert "resolved_alias" in payload["alias_outcomes"]
    for status in payload["curated_outcomes"]:
        assert status in {"partial_family", "ambiguous", "unmapped"}


def test_build_persists_the_ontology_as_a_canonical_dataset() -> None:
    result = runner.invoke(ontology_app, ["build"])
    assert result.exit_code == 0, result.output
    assert "exercises" in result.output
    assert "psd-ontology-0.1.0" in result.output


def test_the_built_ontology_verifies_through_the_canonical_command() -> None:
    """The ontology reuses the canonical verifier rather than a private one."""
    assert runner.invoke(ontology_app, ["build"]).exit_code == 0
    relative = Path(ontology_relative_path(ONTOLOGY))
    verify = runner.invoke(root_app, ["canonical", "verify", str(relative)])
    assert verify.exit_code == 0, verify.output
    assert "OK" in verify.output


def test_the_built_ontology_passes_validation() -> None:
    assert runner.invoke(ontology_app, ["build"]).exit_code == 0
    relative = str(Path(ontology_relative_path(ONTOLOGY)))
    validated = runner.invoke(root_app, ["validate", relative, "--strict"])
    assert validated.exit_code == 0, validated.output
    assert "0 error(s)" in validated.output


def test_the_built_ontology_exposes_its_tables_through_inspect() -> None:
    assert runner.invoke(ontology_app, ["build"]).exit_code == 0
    relative = str(Path(ontology_relative_path(ONTOLOGY)))
    tables = runner.invoke(root_app, ["inspect", "tables", relative])
    assert tables.exit_code == 0, tables.output
    for table in ("exercise_definition", "exercise_alias", "exercise_normalization"):
        assert table in tables.output

    rows = runner.invoke(
        root_app, ["inspect", "table", relative, "exercise_definition", "--json", "--limit", "3"]
    )
    assert rows.exit_code == 0, rows.output
    definitions = json.loads(rows.stdout)
    assert {row["canonical_key"] for row in definitions} <= {spec.key for spec in ONTOLOGY.specs}


def test_build_honours_an_explicit_output_directory() -> None:
    result = runner.invoke(ontology_app, ["build", "--output", "canonical/custom-ontology"])
    assert result.exit_code == 0, result.output
    assert "canonical/custom-ontology" in result.output
    verify = runner.invoke(root_app, ["canonical", "verify", "canonical/custom-ontology"])
    assert verify.exit_code == 0, verify.output


def test_build_json_prints_the_manifest() -> None:
    result = runner.invoke(ontology_app, ["build", "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["dataset_kind"] == "reference"
    assert payload["schema_version"] == "psd-canonical/0.2.0"
    assert len(payload["artifacts"]) == len(table_names())
    assert payload["sources"] == []


def test_build_is_reproducible() -> None:
    """Two builds into different directories must agree byte for byte."""
    first = runner.invoke(ontology_app, ["build", "--output", "canonical/one", "--json"])
    second = runner.invoke(ontology_app, ["build", "--output", "canonical/two", "--json"])
    assert first.exit_code == 0, first.output
    assert second.exit_code == 0, second.output
    left = json.loads(first.stdout)
    right = json.loads(second.stdout)
    for digest in ("content_sha256", "sha256", "byte_size", "row_count"):
        assert [item[digest] for item in left["artifacts"]] == [
            item[digest] for item in right["artifacts"]
        ], digest
