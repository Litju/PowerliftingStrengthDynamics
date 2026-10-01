"""``psd ontology`` -- inspect and persist the exercise ontology.

The command surface is deliberately small and read-mostly. ``resolve`` is the
interesting one: it is how a reviewer checks that a label PSD refused to resolve is
refused for a stated reason rather than by accident, and how they confirm that a
resolved label reached the canonical exercise they expected.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path

import typer

from psd.ontology import default_ontology, exercise_id_for
from psd.ontology.artifact import (
    build_ontology_dataset,
    ontology_dataset_id,
    ontology_relative_path,
    write_ontology,
)
from psd.ontology.registry import DEFAULT_SOURCE_SYSTEM, NormalizationOutcome
from psd.paths import DATA_ROOT_ENV_VAR, DataRootError
from psd.schema.vocabulary import ParentLift

app = typer.Typer(
    name="ontology",
    help="Inspect and persist the PSD exercise ontology.",
    no_args_is_help=True,
)

_PARENTS: tuple[str, ...] = tuple(member.value for member in ParentLift)


def _outcome_payload(outcome: NormalizationOutcome) -> dict[str, object]:
    """Return a JSON-serializable view of one normalization outcome."""
    return {
        "raw_label": outcome.raw_label,
        "normalized_label": outcome.normalized_label,
        "normalization_rules": list(outcome.normalization_rules),
        "source_system": outcome.source_system,
        "resolution_status": outcome.resolution_status.value,
        "resolution_method": outcome.resolution_method.value,
        "exercise_key": outcome.exercise_key,
        "exercise_id": outcome.exercise_id,
        "parent_lift": outcome.parent_lift.value,
        "candidate_keys": list(outcome.candidate_keys),
        "confidence": outcome.confidence,
        "ambiguity_reason": None
        if outcome.ambiguity_reason is None
        else (outcome.ambiguity_reason.value),
    }


@app.command("version")
def version() -> None:
    """Print the ontology and alias-registry versions."""
    ontology = default_ontology()
    typer.echo(f"ontology_version       {ontology.ontology_version.tag}")
    typer.echo(f"alias_registry_version {ontology.alias_registry_version.tag}")
    typer.echo(f"canonical_exercises    {len(ontology.specs)}")
    typer.echo(f"aliases                {len(ontology.aliases)}")
    typer.echo(f"source_systems         {', '.join(ontology.source_systems)}")
    typer.echo(f"unresolved_labels      {len(ontology.curated_ambiguities)}")
    typer.echo(f"dataset_id             {ontology_dataset_id(ontology)}")


@app.command("list")
def list_exercises(
    parent: str | None = typer.Option(
        None, "--parent", help=f"Filter by family. One of: {', '.join(_PARENTS)}."
    ),
    as_json: bool = typer.Option(False, "--json", help="Emit JSON instead of a table."),
) -> None:
    """List canonical exercises."""
    ontology = default_ontology()
    if parent is not None:
        try:
            selected = ParentLift(parent)
        except ValueError:
            typer.secho(
                f"Unknown parent lift {parent!r}; choose one of: {', '.join(_PARENTS)}.",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(code=2) from None
        specs = [ontology.spec(key) for key in ontology.keys_for_parent(selected)]
    else:
        specs = list(ontology.specs)

    if as_json:
        typer.echo(
            json.dumps(
                [
                    {
                        "canonical_key": spec.key,
                        "canonical_name": spec.canonical_name,
                        "parent_lift": spec.parent_lift.value,
                        "specificity_level": spec.specificity_level.value,
                        "implement": spec.implement.value,
                        "bar_type": spec.bar_type.value,
                        "equipment": spec.equipment.value,
                        "stance": spec.stance.value,
                        "grip": spec.grip.value,
                        "range_of_motion": spec.range_of_motion.value,
                        "pause_rule": spec.pause_rule.value,
                        "tempo": spec.tempo.value,
                        "laterality": spec.laterality.value,
                        "configuration": [flag.value for flag in spec.configuration],
                    }
                    for spec in specs
                ],
                indent=2,
                sort_keys=True,
            )
        )
        return

    width = max((len(spec.key) for spec in specs), default=4)
    for spec in specs:
        typer.echo(
            f"{spec.key:<{width}}  {spec.parent_lift.value:<9} "
            f"{spec.specificity_level.value:<20} {spec.canonical_name}"
        )


@app.command("show")
def show(
    key: str = typer.Argument(..., help="Canonical exercise key."),
    as_json: bool = typer.Option(False, "--json", help="Emit JSON instead of a table."),
) -> None:
    """Show one canonical exercise and every alias bound to it."""
    ontology = default_ontology()
    try:
        spec = ontology.spec(key)
    except KeyError as error:
        typer.secho(str(error), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from error
    aliases = [
        {
            "source_system": binding.source_system,
            "alias_raw": binding.raw_label,
            "alias_normalized": binding.normalized_label,
        }
        for binding in ontology.aliases
        if binding.exercise_key == key
    ]
    if as_json:
        typer.echo(
            json.dumps(
                {
                    "canonical_key": spec.key,
                    "canonical_name": spec.canonical_name,
                    "exercise_id": exercise_id_for(spec.key),
                    "parent_lift": spec.parent_lift.value,
                    "specificity_level": spec.specificity_level.value,
                    "implement": spec.implement.value,
                    "bar_type": spec.bar_type.value,
                    "equipment": spec.equipment.value,
                    "laterality": spec.laterality.value,
                    "stance": spec.stance.value,
                    "grip": spec.grip.value,
                    "range_of_motion": spec.range_of_motion.value,
                    "pause_rule": spec.pause_rule.value,
                    "tempo": spec.tempo.value,
                    "configuration": [flag.value for flag in spec.configuration],
                    "note": spec.note,
                    "aliases": aliases,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    typer.echo(f"{spec.key}  ({spec.canonical_name})")
    typer.echo(f"exercise_id       {exercise_id_for(spec.key)}")
    for name, value in (
        ("parent_lift", spec.parent_lift),
        ("specificity", spec.specificity_level),
        ("implement", spec.implement),
        ("bar_type", spec.bar_type),
        ("equipment", spec.equipment),
        ("laterality", spec.laterality),
        ("stance", spec.stance),
        ("grip", spec.grip),
        ("range_of_motion", spec.range_of_motion),
        ("pause_rule", spec.pause_rule),
        ("tempo", spec.tempo),
    ):
        typer.echo(f"{name:<18} {value.value}")
    typer.echo(f"{'configuration':<18} {', '.join(f.value for f in spec.configuration) or '-'}")
    if spec.note:
        typer.echo(f"{'note':<18} {spec.note}")
    typer.echo("")
    typer.echo(f"{len(aliases)} alias(es)")
    for alias in aliases:
        typer.echo(f"  {alias['source_system']:<14} {alias['alias_raw']}")


@app.command("resolve")
def resolve(
    labels: list[str] = typer.Argument(..., help="Raw source labels to normalize."),
    source: str = typer.Option(
        DEFAULT_SOURCE_SYSTEM, "--source", help="Source system to resolve against."
    ),
    as_json: bool = typer.Option(False, "--json", help="Emit JSON instead of a table."),
) -> None:
    """Normalize raw labels and report what, if anything, they identify.

    An unresolved label exits non-zero. That is not an error: it is the report.
    """
    ontology = default_ontology()
    outcomes = [ontology.resolve(label, source_system=source) for label in labels]
    if as_json:
        typer.echo(
            json.dumps(
                [_outcome_payload(outcome) for outcome in outcomes], indent=2, sort_keys=True
            )
        )
    else:
        for outcome in outcomes:
            verdict = outcome.exercise_key or outcome.ambiguity_reason
            typer.echo(
                f"{outcome.raw_label!r} -> {outcome.resolution_status.value} "
                f"[{outcome.resolution_method.value}] {verdict}"
            )
    if not all(outcome.is_resolved for outcome in outcomes):
        raise typer.Exit(code=1)


@app.command("build")
def build(
    output: Path | None = typer.Option(
        None, "--output", help="Dataset directory, relative to the data root."
    ),
    data_root: Path | None = typer.Option(
        None, "--data-root", help=f"Override {DATA_ROOT_ENV_VAR}."
    ),
    as_json: bool = typer.Option(False, "--json", help="Emit a JSON summary."),
) -> None:
    """Persist the ontology as a canonical dataset with a checksum manifest."""
    ontology = default_ontology()
    target = output if output is not None else ontology_relative_path(ontology)
    try:
        manifest = write_ontology(ontology, relative=target, data_root=data_root)
    except DataRootError as error:
        typer.secho(str(error), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from error
    dataset = build_ontology_dataset(ontology)
    if as_json:
        typer.echo(json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True))
        return
    typer.echo(f"dataset    {manifest.dataset_id}")
    typer.echo(f"schema     {manifest.schema_version}")
    typer.echo(f"target     {target}")
    typer.echo(f"exercises  {dataset.row_counts()['exercise_definition']}")
    typer.echo(f"aliases    {dataset.row_counts()['exercise_alias']}")
    typer.echo(f"outcomes   {dataset.row_counts()['exercise_normalization']}")
    typer.echo("verify     psd canonical verify <target>")


@app.command("coverage")
def coverage(as_json: bool = typer.Option(False, "--json", help="Emit JSON.")) -> None:
    """Summarize how much of the registry is resolved versus deliberately open."""
    ontology = default_ontology()
    alias_outcomes = _tally(
        ontology.resolve(binding.raw_label, source_system=binding.source_system)
        for binding in ontology.aliases
    )
    curated_outcomes = _tally(
        ontology.resolve(spec.raw_label) for spec in ontology.curated_ambiguities
    )
    payload: dict[str, object] = {
        "aliases": sum(alias_outcomes.values()),
        "alias_outcomes": alias_outcomes,
        "curated_refusals": sum(curated_outcomes.values()),
        "curated_outcomes": curated_outcomes,
    }
    if as_json:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
        return
    typer.echo(f"aliases            {sum(alias_outcomes.values())}")
    _echo_tally(alias_outcomes)
    typer.echo(f"curated refusals   {sum(curated_outcomes.values())}")
    _echo_tally(curated_outcomes)


def _tally(outcomes: Iterable[NormalizationOutcome]) -> dict[str, int]:
    """Count outcomes per resolution status, sorted by status name."""
    counts: dict[str, int] = {}
    for outcome in outcomes:
        status = outcome.resolution_status.value
        counts[status] = counts.get(status, 0) + 1
    return dict(sorted(counts.items()))


def _echo_tally(counts: Mapping[str, int]) -> None:
    """Print one tally line per resolution status."""
    for name, count in counts.items():
        typer.echo(f"  {name:<20} {count}")


@app.callback()
def root() -> None:
    """Exercise ontology commands."""
