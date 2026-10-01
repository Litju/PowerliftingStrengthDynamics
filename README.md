# Powerlifting Strength Dynamics (PSD)

**PSD is a benchmark for longitudinal system identification, not a training-prescription product.**

Powerlifting Strength Dynamics is an open, model-agnostic benchmark for studying the
longitudinal dynamics of strength athletes as *partially observed controlled systems*.

Given an athlete's legitimate history up to a hard temporal cutoff, plus a future
sequence of training interventions that genuinely existed ex ante, the benchmark asks
whether a model has identified useful **rollout** dynamics over progressively longer
horizons — not whether it can solve a disconnected one-rep-max prediction task.

PSD is **pre-alpha**. Nothing in this repository is a validated physiological model,
a coaching recommendation, or a clinical or injury-prediction tool.

## Scientific invariants

These invariants are the reason the schema exists, and every implementation decision
is subordinate to them.

| Invariant | Meaning |
| --- | --- |
| `prescription ≠ execution ≠ observation ≠ outcome` | What was planned, what was performed, what was reported, and what happened are different records and are never collapsed. |
| Event-time canonical storage | The canonical athlete history is an ordered sequence of events. Daily/weekly aggregates are *derived*, never authoritative. |
| No manufactured prescriptions | A missing prescription stays missing. Execution is never back-filled into a plan. |
| No zero-as-missing | Absence is `null` plus an explicit missingness reason. Zero is only used when zero is semantically correct (for example, `0` repetitions completed on a failed set). |
| No silent coercion | Ambiguous timestamps, ambiguous athlete identity, and ambiguous source values are flagged, not guessed. |
| Real and synthetic stay separable | Synthetic ground truth is labelled as synthetic and is never projected onto real athletes as validated physiology. |
| No imposed latent-state ontology | The benchmark does not require any particular decomposition of latent physiological state. |
| Provenance and missingness survive | Raw source values and source identity are preserved alongside normalized representations. |

## Data model

The canonical representation is a set of typed Arrow tables persisted as Parquet.
The central entities are:

* **Context** — `athlete`, `athlete_source_link`, `body_measurement`, `equipment_state`
* **Programming** — `program`, `program_version`, `program_modification`, `planned_session`, `planned_exercise`, `planned_set`
* **Execution** — `performed_session`, `performed_exercise`, `performed_set`, `performed_rep`
* **Observations** — `observation`, `performance_test`, `velocity_observation`
* **Competition** — `competition`, `competition_attempt`, `competition_reported_result`
* **Semantics** — `exercise_definition`, `exercise_alias`, `exercise_normalization`
* **Provenance** — `source`, `provenance`, plus row-level source keys on every event table

Records carry explicit temporal provenance (`created_at`, `scheduled_at`,
`performed_at`, `observed_at`, `modified_at`, `ingested_at`) so that "what was known
at prediction time" is decidable rather than assumed.

## Exercise semantics

Heterogeneous logs call one movement many things, and call different movements many of
the same things. PSD keeps identity, spelling, and refusal separate:

| Layer | Table | What it records |
| --- | --- | --- |
| Identity | `exercise_definition` | What an exercise **is**: parent lift, specificity, implement, bar, apparatus, stance, grip, range of motion, pause, tempo, laterality, setup flags. |
| Spelling | `exercise_alias` | A label some app wrote, bound to exactly one canonical exercise, with the verbatim string preserved. |
| Refusal | `exercise_normalization` | Every mapping decision **including the ones PSD declines to make**, with the reason and, where they exist, the candidate readings. |

Normalizing a label is a six-stage ladder — canonical identity, curated ambiguity,
exact alias, generic qualifier, structured interpretation, family keyword — and the
first defensible answer wins. A stage that cannot name one canonical exercise returns
`partial_family`, `ambiguous`, or `unmapped` with a reason, and no later stage may
upgrade it. Coverage is a property of the vocabulary, not a licence to guess, so
`Machine press`, `Bench variation`, and `Leg press?` stay open on purpose.

The ontology describes what an exercise is and never what it is worth: it records no
transfer coefficient, no specificity score, and no statement that one variation
substitutes for another. Models learn those relationships from the published
descriptors.

```powershell
uv run psd ontology version
uv run psd ontology resolve "Low Bar Squat" "Machine press" --source hevy
uv run psd ontology build
```

`psd ontology build` persists the ontology as an ordinary canonical dataset, so it
inherits canonical ordering, Arrow schema enforcement, writer-independent content
digests, and manifest verification rather than a parallel format.

## Installation

PSD requires **CPython 3.12** and is managed with [uv](https://docs.astral.sh/uv/).

```powershell
git clone https://github.com/Litju/PowerliftingStrengthDynamics.git
cd PowerliftingStrengthDynamics
uv sync --locked
uv run psd --help
```

Canonical local development is **Windows-native**. Linux is a first-class CI and
remote-compute target, not the primary local environment. WSL is not used as the
canonical working tree.

## Development

All canonical gates reduce to:

```powershell
uv sync --locked
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv run pytest
uv build
```

### Data root boundary

PSD never stores datasets in the Git repository. External data is resolved through a
single typed environment boundary:

```text
PSD_DATA_ROOT=E:\Data\Databases\PowerliftingStrengthDynamics
```

The data root holds `raw/`, `external/`, `canonical/`, `synthetic/`, `processed/`,
`manifests/`, `cache/`, and `runs/`. All library paths use `pathlib.Path`. Tests are
self-contained and never depend on the maintainer's data root.

Inspect the resolved layout with:

```powershell
uv run psd paths
```

## Command-line interface

```text
psd schema      # canonical table registry, machine-readable schemas, schema version
psd validate    # validate a canonical dataset: declarative columns plus cross-record rules
psd canonical   # build and verify canonical datasets; print provenance and checksum manifests
psd inspect     # inspect persisted tables and athlete event timelines
psd ontology    # inspect the exercise ontology; normalize raw labels; persist it
psd paths       # resolved external data-root layout
psd version     # installed PSD version
```

There is no separate `psd provenance` command: provenance and checksum manifests are part of
a canonical dataset, so `psd canonical manifest` prints them rather than a second command
re-reading state the manifest already owns.

The CLI is orchestration over importable Python APIs; benchmark and schema logic never
lives only inside command functions.

## Repository status

* **Stage:** pre-alpha (`Development Status :: 2 - Pre-Alpha`).
* **Schema version:** `psd-canonical/0.2.0` — not a frozen public contract.
* **Ontology version:** `psd-ontology/0.1.0`, alias registry
  `psd-ontology-alias/0.1.0` — tracked separately, because adding a source alias must
  not imply that a canonical exercise identity changed.
* **Design authority:** scientific design documents are *tentative*; the technical
  stack and engineering constraints document is *locked* for v0/Alpha.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup, quality gates, atomic-change
expectations, and issue/PR conventions. Participation is governed by the
[Code of Conduct](CODE_OF_CONDUCT.md).

## Licensing boundary

PSD source code and original project documentation are licensed under the
[Apache License 2.0](LICENSE).

**This license does not extend to data.** Third-party datasets, OpenPowerlifting data,
donated athlete histories, and any other externally sourced artifacts retain their own
licenses, consent, provenance, and redistribution constraints, which are recorded in
dataset manifests and documentation. Do not assume that anything found under
`PSD_DATA_ROOT` is Apache-2.0 licensed.

Athlete histories may contain personal data. Donated or crawled data is expected to
have an explicit consent, privacy, and processing basis before ingestion.

## Citation

Citation metadata is provided in [CITATION.cff](CITATION.cff). Cite the project as
pre-alpha software unless a released version or paper states otherwise.

## License

[Apache-2.0](LICENSE)
