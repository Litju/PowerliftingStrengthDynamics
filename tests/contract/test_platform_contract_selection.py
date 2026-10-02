"""The self-hosted Windows parity selection is a contract, and this is where it is held.

PSD's CI has two paths with different jobs. GitHub-hosted Ubuntu owns the broad regression
gate: lint, types, the whole suite, coverage, the build, the wheel smoke. The self-hosted
Windows runner owns one narrow thing -- catching Windows-specific and cross-platform
failures -- and runs ``pytest -m windows_parity`` against a selection defined entirely by
the marker.

A marker-based selection can fail silently in three ways, and each has a check here:

1. **An area stops being covered.** Somebody fixes a Windows bug, marks the test, and later
   somebody refactors the name or drops the marker.
   :func:`test_every_declared_platform_area_is_still_covered` fails when a declared area has
   no marked test left.
2. **The selection quietly becomes the whole suite.** That would put the full cost back on
   the workstation every push, which is exactly what the split exists to prevent.
   :func:`test_the_selection_stays_a_minority_of_the_suite` bounds it.
3. **The self-hosted job grows.** Somebody adds lint, coverage or a service container to the
   workstation job because the hosted one has it.
   :func:`test_the_self_hosted_job_runs_only_what_it_is_allowed_to_run` bounds that.

Every check here reads the repository statically rather than the collected session, for two
reasons: the answer must not depend on which subset pytest happened to be asked for, and a
check that only holds when run a particular way is a check nobody can run while fixing the
thing it guards.

The review rule for the human half of the mechanism is in ``CONTRIBUTING.md``: a new
platform-sensitive regression gets the marker at the moment it is fixed, and
``uv run pytest -m windows_parity`` proves it is in the subset.
"""

from __future__ import annotations

import ast
from pathlib import Path

#: The marker the self-hosted Windows job selects on.
MARKER = "windows_parity"

#: Every platform-contract area the Windows parity gate is required to cover, as
#: ``(area, test module filename, name fragment)``. An area is covered when the module
#: carries a module-level mark, or when a decorated test in it has the fragment in its name
#: or in a marked ``pytest.param`` id.
REQUIRED_AREAS: tuple[tuple[str, str, str], ...] = (
    ("Windows drive, absolute and relative path semantics", "test_paths.py", "path"),
    ("data-root isolation and containment", "test_paths.py", "data_root"),
    ("the data root through the CLI", "test_cli_paths.py", "paths"),
    ("timezone database availability and aware timestamps", "test_timeutil.py", "utc"),
    ("canonical timestamp encoding", "test_canonical_timestamps.py", "timestamp"),
    ("pre-1970 and negative epoch encoding", "test_canonical_timestamps.py", "epoch"),
    ("Arrow and Parquet read-write smoke", "test_serialization.py", "parquet"),
    ("canonical content hashing and digest equivalence", "test_serialization.py", "digest"),
    ("ordering before hashing", "test_serialization.py", "order"),
    (
        "streaming versus single-shot content digest",
        "test_streaming_writer.py",
        "single_shot_digest",
    ),
    ("manifest and dataset digest persistence", "test_dataset.py", "verify"),
    ("dataset layout under the data root", "test_dataset.py", "data_root"),
    ("CLI import, help and version smoke", "test_cli.py", "help"),
    ("CLI schema smoke", "test_cli.py", "version"),
    ("the CLI dataset surface under PSD_DATA_ROOT", "test_cli_dataset.py", "canonical"),
    ("deterministic identifiers", "test_identifiers.py", "identifier"),
    ("locale and path-separator independence", "test_ontology_determinism.py", "determin"),
    ("cross-platform ontology artifact digests", "test_ontology_artifact.py", "digest"),
    ("a small deterministic PSD-COMP build", "test_openpowerlifting_build.py", "build"),
    ("a small deterministic PSD-COMP verify", "test_openpowerlifting_build.py", "verify"),
    (
        "a small deterministic PSD-COMP history build",
        "test_openpowerlifting_history.py",
        "histor",
    ),
    ("a small deterministic PSD-COMP audit", "test_openpowerlifting_audit.py", "audit"),
)

#: Test modules that must never carry the marker. The parity job runs on a workstation
#: without a network and without the real corpus, so anything in here would fail there for
#: the wrong reason and hide a real platform failure behind an environment one.
FORBIDDEN_MODULES: frozenset[str] = frozenset(
    {
        "test_openpowerlifting_acquire.py",
        "test_openpowerlifting_cli.py",
        "test_openpowerlifting_diagnostics.py",
        "test_fixture_exit_gate.py",
        "test_ontology_normalization.py",
        "test_qualification_reproducibility.py",
    }
)

#: The repository checkout, derived from this file rather than imported: the thing under test
#: is repository configuration, so reaching it through a test-support constant that names
#: ``tests/`` would be a small, permanent trap.
REPOSITORY_ROOT: Path = Path(__file__).resolve().parents[2]
TESTS_ROOT: Path = REPOSITORY_ROOT / "tests"
CI_WORKFLOW: Path = REPOSITORY_ROOT / ".github" / "workflows" / "ci.yml"
MANUAL_WORKFLOW: Path = REPOSITORY_ROOT / ".github" / "workflows" / "windows-full-qualification.yml"


def _test_modules() -> list[Path]:
    """Return every test module in the suite, sorted for a stable report."""
    return sorted(TESTS_ROOT.rglob("test_*.py"))


def _mentions_marker(source: str) -> bool:
    """Whether *source* names the parity marker at all."""
    return MARKER in source


def _module_level_marked(path: Path) -> bool:
    """Whether *path* assigns ``pytestmark`` to the parity marker."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(getattr(target, "id", None) == "pytestmark" for target in node.targets):
            continue
        if MARKER in ast.dump(node.value):
            return True
    return False


def _marked_names(path: Path) -> set[str]:
    """Return the names of *path*'s individually marked tests and marked parameter ids."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and any(
            MARKER in ast.dump(decorator) for decorator in node.decorator_list
        ):
            names.add(node.name)
        if isinstance(node, ast.Call) and MARKER in ast.dump(node):
            for keyword in node.keywords:
                value = keyword.value
                if (
                    keyword.arg == "id"
                    and isinstance(value, ast.Constant)
                    and isinstance(value.value, str)
                ):
                    names.add(value.value)
    return names


def _marked_modules() -> dict[str, Path]:
    """Return every test module carrying the marker, keyed by filename."""
    return {
        path.name: path for path in _test_modules() if _mentions_marker(path.read_text("utf-8"))
    }


def _workflow_jobs() -> dict[str, str]:
    """Return each job's body from ``ci.yml``, keyed by job name.

    Parsed from the text rather than schema-loaded on purpose: the checks are about which
    commands a job contains, and reading the lines is the cheapest honest way to see that. A
    YAML round trip would add a dependency to prove something about a handful of strings.
    """
    lines = CI_WORKFLOW.read_text(encoding="utf-8").splitlines()
    start = next(index for index, line in enumerate(lines) if line.rstrip() == "jobs:")
    jobs: dict[str, list[str]] = {}
    current: str | None = None
    for line in lines[start + 1 :]:
        if line.startswith("  ") and not line.startswith("   ") and line.rstrip().endswith(":"):
            current = line.strip().rstrip(":")
            jobs[current] = []
        elif current is not None:
            jobs[current].append(line)
    return {name: "\n".join(body) for name, body in jobs.items()}


def _self_hosted_job() -> str:
    """Return the self-hosted Windows parity job's commands, comments excluded.

    Comments are excluded because they legitimately *name* the things the job must not do --
    "no setup-node, no Docker" is the most useful comment in the file, and a check that
    tripped over it would push the next author to delete the explanation instead of the
    offending step.
    """
    body = next((body for body in _workflow_jobs().values() if "self-hosted" in body), "")
    return "\n".join(line for line in body.splitlines() if not line.lstrip().startswith("#"))


# ---------------------------------------------------------------------------
# the selection
# ---------------------------------------------------------------------------


def test_the_marker_is_declared_so_strict_marker_checking_keeps_working() -> None:
    """``--strict-markers`` is on, so an undeclared marker is an error rather than a typo."""
    text = (REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "markers = [" in text
    assert f'"{MARKER}:' in text


def test_every_declared_platform_area_is_still_covered() -> None:
    """The check that makes the marker maintainable.

    An area is covered when its module carries a module-level mark, or when a test in it is
    individually marked and its name -- or a marked parameter id -- contains the fragment.
    The per-test path matters: the streaming contract is marked test by test precisely
    because the expensive cases in that module are not platform contracts.
    """
    uncovered: list[str] = []
    for area, filename, fragment in REQUIRED_AREAS:
        matches = [path for path in _test_modules() if path.name == filename]
        if not matches:
            uncovered.append(f"{area}: no module named {filename}")
            continue
        path = matches[0]
        if _module_level_marked(path):
            continue
        if any(fragment in name for name in _marked_names(path)):
            continue
        uncovered.append(f"{area}: {filename} has no marked test matching {fragment!r}")
    assert not uncovered, uncovered


def test_the_selection_stays_a_minority_of_the_suite() -> None:
    """The point of the split is that the Windows gate is not the regression gate."""
    modules = _test_modules()
    marked = _marked_modules()
    assert marked, "no test module carries the parity marker at all"
    assert len(marked) < len(modules) / 2, sorted(marked)


def test_the_selection_is_not_the_whole_suite_in_test_count() -> None:
    """A module-count ceiling is a proxy; this is the bound that matters operationally."""
    marked = _marked_modules()
    assert len(marked) <= 20, sorted(marked)


def test_nothing_forbidden_carries_the_marker() -> None:
    """No network, no real-corpus acquisition, no service-dependent or fan-out tests."""
    marked = set(_marked_modules())
    assert not (marked & FORBIDDEN_MODULES), sorted(marked & FORBIDDEN_MODULES)


def test_the_selection_touches_no_real_data_root() -> None:
    """Nothing marked may reach the maintainer's external data root."""
    offenders = sorted(
        path.name
        for path in _marked_modules().values()
        if "qualification" in path.name or "real_data" in path.name
    )
    assert not offenders, offenders


# ---------------------------------------------------------------------------
# the workflows
# ---------------------------------------------------------------------------


def test_the_workflow_runs_the_selection_by_marker_not_by_path() -> None:
    """A file list in workflow YAML rots silently; the marker lives with the tests."""
    text = CI_WORKFLOW.read_text(encoding="utf-8")
    assert f"-m {MARKER}" in text
    assert "tests/unit/" not in text
    assert "tests/integration/" not in text


def test_the_self_hosted_job_runs_only_what_it_is_allowed_to_run() -> None:
    """The workstation job's cost is a policy, and a policy nobody checks decays.

    Checked against the job body rather than the whole file, because the hosted Linux job in
    the same workflow legitimately runs all of this.
    """
    job = _self_hosted_job()
    assert job, "the self-hosted Windows parity job must exist in ci.yml"
    for required in (
        f"-m {MARKER}",
        "uv sync",
        "scripts/tiny_psd_comp.py",
        "scripts/wheel_smoke.py",
        "runs-on: [self-hosted, Windows, X64]",
        "cancel-in-progress: true",
    ):
        assert required in job or required in CI_WORKFLOW.read_text(encoding="utf-8"), required
    for forbidden in (
        "--cov",
        "ruff",
        "pyright",
        "setup-node",
        "docker",
        "postgres",
        "redis",
        "-n auto",
        "xdist",
        "services:",
        "actions/setup-python",
        "qualify_psd_comp",
        "upload-artifact",
    ):
        assert forbidden not in job, forbidden


def test_the_hosted_linux_job_owns_the_full_gate() -> None:
    """If the broad gate ever stops running the broad gate, the split means nothing."""
    text = CI_WORKFLOW.read_text(encoding="utf-8")
    for required in (
        "runs-on: ubuntu-latest",
        "uv run ruff check .",
        "uv run ruff format --check .",
        "uv run pyright",
        "--cov=psd",
        "uv build",
        "scripts/wheel_smoke.py",
    ):
        assert required in text, required


def test_no_hosted_job_runs_on_the_workstation() -> None:
    """A self-hosted label on the broad gate would put the full suite back on the desk."""
    hosted = [body for body in _workflow_jobs().values() if "ubuntu-latest" in body]
    assert hosted, "no hosted Linux job"
    assert all("self-hosted" not in body for body in hosted)


def test_the_documented_command_resolves_the_subset() -> None:
    """The command in the contributor documentation must be the command CI runs."""
    contributing = (REPOSITORY_ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")
    assert f"uv run pytest -m {MARKER}" in contributing


def test_the_workflow_keeps_a_manual_full_windows_path() -> None:
    """Full Windows qualification must remain reachable without being the push path."""
    assert MANUAL_WORKFLOW.is_file(), "the manual full-Windows qualification workflow must exist"
    text = MANUAL_WORKFLOW.read_text(encoding="utf-8")
    assert "workflow_dispatch" in text
    assert "\n  push:" not in text
    assert "\n  pull_request:" not in text
    assert "runs-on: [self-hosted, Windows, X64]" in text
    assert "uv run pytest -q" in text
