# Contributing to Powerlifting Strength Dynamics

Thanks for your interest in PSD. PSD is a **pre-alpha research benchmark**, and
contributions are reviewed for scientific and reproducibility integrity first and
novelty second.

PSD is a benchmark for **longitudinal system identification**. It is not a
training-prescription product, a coaching tool, a clinical tool, or an injury-prediction
system. Contributions that turn it into one are out of scope.

## Development setup

PSD targets **CPython 3.12** and is managed with [uv](https://docs.astral.sh/uv/).
Canonical local development is **Windows-native**; Linux is a first-class CI and
remote-compute target. WSL is not used as the canonical working tree.

```powershell
git clone https://github.com/Litju/PowerliftingStrengthDynamics.git
cd PowerliftingStrengthDynamics
uv sync --locked
uv run psd --help
```

The build backend is `uv_build`. `uv` is pinned in CI (`0.12.4`) to match
`build-system.requires`, so local and CI builds exercise the same backend.

### Data root boundary

The repository never stores datasets. External data resolves through a single typed
environment boundary:

```text
PSD_DATA_ROOT=<absolute path to your external PSD data root>
```

```powershell
uv run psd paths
```

Tests must be self-contained. **No test may require the maintainer's data root.** If a
test needs real external data, mark it explicitly as an opt-in local qualification test
that skips when `PSD_DATA_ROOT` is unset.

## Quality gates

Every change must leave all canonical gates green:

```powershell
uv sync --locked
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv run pytest
uv build
```

CI enforces the same gates, split across two paths with different jobs.

| path | runner | what it owns |
| --- | --- | --- |
| `hosted-linux-quality` | GitHub-hosted Ubuntu | the broad regression gate: lint, format, strict types, the **whole** suite with coverage, the build, a clean wheel smoke |
| `hosted-linux-wheel-smoke` | GitHub-hosted Ubuntu | the packaging gate on its own, so a wheel failure is visibly a packaging failure |
| `self-hosted-windows-parity` | self-hosted Windows workstation | **Windows portability only**: the `windows_parity` test subset, a tiny PSD-COMP fixture build/verify, a wheel and CLI smoke |
| Full Windows qualification | self-hosted Windows workstation | the whole suite on Windows; `workflow_dispatch` only, for Alpha and release cuts |

The self-hosted job is deliberately small. It acquires nothing, builds no real corpus,
runs no coverage, no lint, no type check, no service containers, and no network-dependent
test, and it is configured for `cancel-in-progress` so a new push supersedes the previous
run rather than queueing behind it. Do not add work to it for symmetry: Ubuntu already
proves general correctness, and Windows proves Windows portability.

Type checking is **Pyright strict**; narrow, justified per-scope exceptions are preferable
to weakening project-wide strictness.

Test classes follow the locked stack: unit, property (Hypothesis), integration,
golden/reproducibility, benchmark-contract/leakage, and slow smoke. Expensive model
training is not a pull-request gate.

### The Windows parity subset

The self-hosted job runs exactly one selection:

```powershell
uv run pytest -m windows_parity
```

`windows_parity` is a real pytest marker declared in `pyproject.toml`, and it is the whole
selection mechanism — no file lists in workflow YAML, which rot silently. A test carries it
by declaring `pytestmark = pytest.mark.windows_parity` at module level, or
`@pytest.mark.windows_parity` on a single test or `pytest.param`.

**The review rule: when you fix a Windows-specific or cross-platform failure, mark the test
that proves the fix in the same commit, and run `uv run pytest -m windows_parity` locally
to confirm it is in the subset.** A regression test that only runs on Linux is not evidence
that Windows still works.

Mark a test when it materially exercises one of: Windows drive, UNC, absolute and relative
path semantics; the `PSD_DATA_ROOT` boundary; `pathlib` and data-root containment; pytest
temporary-directory assumptions; timezone-database and `tzdata` availability; canonical
timestamp encoding including pre-1970, negative epoch, UTC normalization and equivalent
aware instants; Arrow/Parquet read-write; canonical content hashing and
streaming-versus-single-shot digest equivalence; deterministic ordering and identifiers
where platform semantics could matter; CLI import, help, version and schema surface; and
the tiny deterministic PSD-COMP build/verify path.

`tests/contract/test_platform_contract_selection.py` enforces the rest: every declared
platform area must still have a marked test, the selection must stay under its size ceiling,
nothing network-dependent or full-corpus may be marked, and the self-hosted job in
`ci.yml` must not have grown a lint, type-check, coverage or service step.

## Change expectations

* **Atomic commits.** Each commit is a coherent, complete, independently reviewable
  piece of work and passes the gates on its own. Do not mix refactors with behavior
  changes, and do not split a single logical change across unrelated commits.
* **Do not squash.** Preserve history.
* **Explain the evidence.** If an implementation reveals a genuine conflict with the
  tentative scientific design documents, amend the relevant design document and state
  the evidence. Do not silently reinterpret the benchmark.
* **Add tests with behavior.** Every new invariant needs a test; every new field needs a
  round-trip test.
* **No notebook-only implementation.** Notebooks may explore, but production behavior
  lives in `src/psd`.
* **No hidden local state.** Everything reproducible must be committed and derived
  from `PSD_DATA_ROOT`.

## Non-negotiable invariants

Changes must not break these. They are the reason the schema exists.

1. `prescription ≠ execution ≠ observation ≠ outcome` is preserved in the data model.
2. Canonical storage is event-time and temporally ordered. Daily/weekly aggregates are
   derived, never authoritative.
3. A missing prescription is never inferred from execution.
4. Absence is never encoded as zero unless zero is semantically correct.
5. Ambiguous timestamps, athlete identity, and source values are flagged, not coerced.
6. Real athlete data, simulator-generated athlete data, and authored reference
   vocabulary remain explicitly distinguishable (`real`/`psd_real`,
   `synthetic`/`psd_sim`, `reference`/`psd_reference`). A reference vocabulary is never
   relabelled synthetic to make a rule pass.
7. No physiological latent-state ontology is imposed on the benchmark.
8. Provenance and missingness survive every transformation.
9. Raw source values are preserved alongside normalized values.
10. Persisted artifacts have explicit canonical ordering before hashing or writing.
11. PyArrow/Parquet is the persisted schema boundary; Polars is the transformation engine.
12. No random row/set-level splits may enter the data layer.
13. No numeric score is published without calibration behind it. A value that would read
    as a probability without being estimated is omitted, not replaced by another
    arbitrary number.

## Issues and pull requests

* Search existing issues first; include the `RES-` issue or the relevant design
  document when a change is design-bound.
* State the scientific or engineering motivation, not only the mechanical change.
* For schema changes, include: the new/changed fields, the missingness semantics, the
  provenance impact, and the migration/`schema_version` bump rationale.
* For scientific changes, state the claim being made, the evidence, and what would
  falsify it.
* Call out anything that changes a locked-stack decision explicitly and early.

## Licensing boundary

Contributions are accepted under the Apache License 2.0. Do **not** contribute datasets
containing athlete data without an explicit consent, privacy, and redistribution basis;
third-party data keeps its own license and provenance.

## Code of Conduct

Participation is governed by [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md). Security issues
go to the process in [SECURITY.md](SECURITY.md), not the public issue tracker.
