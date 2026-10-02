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
| Real, simulated, and authored stay separable | Simulator ground truth is labelled `synthetic`/`psd_sim` and is never projected onto real athletes as validated physiology; authored vocabulary is labelled `reference`/`psd_reference` and is never presented as either. |
| No uncalibrated numbers | A published numeric score must be one PSD actually estimated. Where a number would read as a probability without being calibrated, the field is omitted rather than invented. |
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

**A competition label names a lift, not a stance.** `Competition Squat`, `Competition
Bench`, and `Competition Deadlift` resolve to the bare competition lift, because the
rules define each lift by grip, start, and the completed erect position rather than by
every observable a label could mention. The deadlift makes this concrete: the IPF
Technical Rulebook prescribes neither a conventional nor a sumo stance, and both are
legal performances of the same lift. So `deadlift` is the stance-unspecified
competition lift (`specificity_level = competition_lift`, `stance = not_specified`),
while `conventional_deadlift` and `sumo_deadlift` are two stance-qualified variations
of it — both `competition_variation`, symmetrically, neither privileged. A source that
does state a stance gets the stance-qualified entity, exactly as `Low Bar Squat` gets
`low_bar_squat`.

**Mapping evidence is symbolic.** `exercise_normalization` records the resolution
method that produced the outcome, the alias row a lookup matched, the candidates a
refusal declined to choose between, the ambiguity reason, and the two versions in
force. There is deliberately **no** numeric mapping score: a fixed float per method
would read as a calibrated probability without ever having been calibrated against
held-out labels, so the field is absent rather than retuned.

The ontology describes what an exercise is and never what it is worth: it records no
transfer coefficient, no specificity score, no per-exercise number of any kind, and no
statement that one variation substitutes for another. Models learn those relationships
from the published descriptors.

The ontology is **authored reference data**. Its manifest is `DatasetKind.REFERENCE`,
its one `source` row is `SourceNature.REFERENCE` / `DataRegime.REFERENCE`, and
`SourceNature.SYNTHETIC` / `psd_sim` remains reserved for simulator-generated athlete
observations. A vocabulary that describes exercises describes no athlete at all, so
filing it as PSD-Sim would assert a generator produced it.

```powershell
uv run psd ontology version
uv run psd ontology resolve "Low Bar Squat" "Machine press" --source hevy
uv run psd ontology resolve "Competition Deadlift" "Sumo Deadlift"
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
psd openpowerlifting # acquire, build, audit, and verify the PSD-COMP competition corpus
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
* **Schema version:** `psd-canonical/1.1.0` — not a frozen public contract. The major
  bump removed `exercise_normalization.confidence`, an uncalibrated float that read as a
  probability; artifacts carrying it must be regenerated.
* **Ontology version:** `psd-ontology/1.0.0`, alias registry
  `psd-ontology-alias/1.0.0` — tracked separately, because adding a source alias must
  not imply that a canonical exercise identity changed. Both are major because the
  competition-deadlift stance semantics and two alias bindings changed meaning.
* **Design authority:** scientific design documents are *tentative*; the technical
  stack and engineering constraints document is *locked* for v0/Alpha.

## PSD-COMP competition corpus

`psd openpowerlifting` converts one pinned OpenPowerlifting bulk export into canonical
competition tables plus per-athlete longitudinal histories. The pinned corpus is
**4,036,909 source rows → 1,014,126 athletes, 64,350 meets, 13,948,408 attempts, and
27,832,326 reported results**.

```powershell
uv run psd openpowerlifting acquire  # pin the snapshot by SHA-256, once
uv run psd openpowerlifting inspect  # identity, revision, header, schema agreement
uv run psd openpowerlifting build    # canonical tables + athlete histories
uv run psd openpowerlifting audit    # durable corpus audit, JSON and Markdown
uv run psd openpowerlifting verify   # source, artifacts, invariants
```

### What the corpus asserts, and what it deliberately refuses to

Every source row becomes exactly one `competition`, which is a single **participation**.
Two athletes at the same meet are two competitions; the meet they shared is a separate
`competition_meet`. This is the only defensible reading of the export, and it has
consequences worth stating:

* **Attempt signs are reported, not resolved.** A negative best is a fact about the source,
  and the audit counts them rather than silently correcting them.
* **Missing attempts stay missing.** A blank cell is not a zero. 52.5% of competitions carry
  attempt detail; the rest have none invented for them.
* **Reported totals are kept as reported.** 330,115 competitions have a total that differs
  from the sum of their bests. That is preserved and counted, not reconciled.
* **Ambiguous identities stay ambiguous.** 2,082 names appear under more than one reported
  sex category; the conflict is recorded on the athlete rather than resolved by guesswork.
* **Participation codes are opaque.** `Place` is not modelled as a ranking.

The audit reports all of this and **exits zero**: an unusual source is a fact to publish,
not a build failure. A *contract* failure is different — expansion invariants, referential
integrity, and duplicate keys fail `verify`.

### Identity and meet rules

Athlete identity is derived from the verbatim `Name` including its `#N` disambiguator, so
one name under two sex categories is one athlete with a recorded conflict, not two people
and not one silently chosen sex. A meet is identified by its start date, federation, and
name, so 8,819 place-names that appear across more than one meet identity stay split rather
than merged.

### Reproducibility and memory

Two builds of one snapshot produce **identical content digests for all 26 artifacts**,
identical row counts, and identical counters. Physical Parquet bytes are *expected* to
differ: a Parquet file is not a pure function of its rows. Identity is `content_sha256`;
the byte digest is what `verify` measures.

Staging partitions by the identity's leading **digest** character, which is what makes the
partition walk the canonical order, and it splits into row positions rather than copying
the chunk to hold one more column. A 16-partition staging pass over the full corpus takes
about 35 seconds and produces 16 files.

Full-corpus qualification:

```powershell
uv run python scripts/qualify_psd_comp.py --data-root <data-root> --digest <sha256> `
    --report qualification.json
```

It builds twice, proves the two builds agree, audits, verifies through the shipped CLI, and
writes a JSON report. Expect roughly 85 minutes.

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
