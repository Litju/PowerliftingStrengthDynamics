"""``psd schema`` -- inspect the canonical schema registry."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from psd.schema import (
    ID_SCHEME,
    SCHEMA_VERSION,
    arrow_schema_for,
    schema_json,
    table_categories,
    table_names,
    table_spec,
)
from psd.schema.registry import validate_schema_columns
from psd.schema.vocabulary import VOCABULARIES

app = typer.Typer(
    name="schema",
    help="Inspect the canonical Arrow/Parquet schema.",
    no_args_is_help=True,
)


@app.command("version")
def version() -> None:
    """Print the canonical schema version and identifier scheme."""
    typer.echo(f"schema_version   {SCHEMA_VERSION.tag}")
    typer.echo(f"id_scheme        {ID_SCHEME}")
    typer.echo(f"table_count      {len(table_names())}")
    typer.echo(f"column_count     {sum(len(arrow_schema_for(n)) for n in table_names())}")


@app.command("list")
def list_tables(
    category: str | None = typer.Option(None, "--category", help="Filter by semantic category."),
    as_json: bool = typer.Option(False, "--json", help="Emit JSON instead of a table."),
) -> None:
    """List canonical tables."""
    validate_schema_columns()
    names = list(table_names())
    if category is not None:
        names = [name for name in names if table_spec(name).category == category]
    if as_json:
        typer.echo(
            json.dumps(
                [
                    {
                        "table": name,
                        "category": table_spec(name).category,
                        "columns": len(table_spec(name).columns),
                        "rows_order": list(table_spec(name).order_by),
                        "summary": table_spec(name).summary,
                    }
                    for name in names
                ],
                indent=2,
                sort_keys=True,
            )
        )
        return
    width = max(len(name) for name in names) if names else 4
    for name in names:
        spec = table_spec(name)
        typer.echo(f"{name:<{width}}  {spec.category:<12}  {len(spec.columns):>3} columns")
    typer.echo("")
    for label, members in table_categories().items():
        typer.echo(f"{label}: {', '.join(members)}")


@app.command("show")
def show(
    table: str = typer.Argument(..., help="Canonical table name."),
    as_json: bool = typer.Option(False, "--json", help="Emit JSON instead of a table."),
) -> None:
    """Show one table's schema, primary key, and canonical ordering."""
    try:
        spec = table_spec(table)
    except KeyError as error:
        typer.secho(str(error), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from error
    description = schema_json(
        spec.arrow_schema(), table_name=spec.name, schema_version=SCHEMA_VERSION.tag
    )
    if as_json:
        typer.echo(
            json.dumps(
                {
                    **description,
                    "category": spec.category,
                    "primary_key": list(spec.primary_key),
                    "order_by": list(spec.order_by),
                    "summary": spec.summary,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    typer.echo(f"{spec.name}  ({spec.category})")
    typer.echo(spec.summary)
    typer.echo(f"primary key : {', '.join(spec.primary_key)}")
    typer.echo(f"ordered by  : {', '.join(spec.order_by)}")
    typer.echo("")
    for entry in description["columns"]:
        typer.echo(f"  {entry['name']:<32} {entry['type']}")


@app.command("dump")
def dump(
    output: Path | None = typer.Option(
        None, "--out", help="Write the schema document to this file instead of stdout."
    ),
) -> None:
    """Dump every table's schema as a machine-readable JSON document."""
    document = {
        "schema_version": SCHEMA_VERSION.tag,
        "id_scheme": ID_SCHEME,
        "tables": [
            {
                **schema_json(
                    spec.arrow_schema(), table_name=spec.name, schema_version=SCHEMA_VERSION.tag
                ),
                "category": spec.category,
                "primary_key": list(spec.primary_key),
                "order_by": list(spec.order_by),
                "summary": spec.summary,
            }
            for spec in map(table_spec, table_names())
        ],
    }
    text = json.dumps(document, indent=2, sort_keys=True) + "\n"
    if output is None:
        typer.echo(text, nl=False)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    typer.echo(f"Wrote {len(document['tables'])} table schemas to {output}")


@app.command("vocabularies")
def vocabularies(
    as_json: bool = typer.Option(False, "--json", help="Emit JSON instead of a table."),
) -> None:
    """List the controlled vocabularies and their members."""
    if as_json:
        typer.echo(
            json.dumps({name: list(members) for name, members in VOCABULARIES.items()}, indent=2)
        )
        return
    for name, members in VOCABULARIES.items():
        typer.echo(f"{name} ({len(members)})")
        for member in members:
            typer.echo(f"    {member}")


@app.callback()
def root() -> None:
    """Schema commands."""
