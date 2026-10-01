"""``psd validate`` and ``psd inspect`` -- check and explore canonical datasets."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import typer

from psd.paths import DATA_ROOT_ENV_VAR, DataRootError, resolve_data_root
from psd.schema.registry import column_order, table_names, table_spec
from psd.serialization.dataset import DatasetLayoutError, read_dataset
from psd.serialization.table import table_to_rows
from psd.validation import validate_tables

app = typer.Typer(name="inspect", help="Inspect canonical datasets.", no_args_is_help=True)

TIMELINE_TABLES: tuple[tuple[str, str, str], ...] = (
    ("performed_session", "started_at", "athlete_id"),
    ("planned_session", "scheduled_at", "athlete_id"),
    ("observation", "observed_at", "athlete_id"),
    ("performance_test", "observed_at", "athlete_id"),
    ("velocity_observation", "observed_at", "athlete_id"),
    ("body_measurement", "measured_at", "athlete_id"),
    ("competition", "competition_date", "athlete_id"),
    ("competition_attempt", "attempt_time", "athlete_id"),
)


def _load(dataset_dir: Path, data_root: Path | None) -> Any:
    """Read a dataset, exiting with a clear message on failure."""
    root = resolve_data_root(data_root)
    try:
        dataset, _manifest = read_dataset(dataset_dir, data_root=root)
    except (DatasetLayoutError, DataRootError) as error:
        typer.secho(str(error), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from error
    return dataset


@app.command("tables")
def inspect_tables(
    dataset_dir: Path = typer.Argument(..., help="Dataset directory, relative to the data root."),
    data_root: Path | None = typer.Option(
        None, "--data-root", help=f"Override {DATA_ROOT_ENV_VAR}."
    ),
) -> None:
    """List every canonical table with its row count."""
    dataset = _load(dataset_dir, data_root)
    counts = dataset.row_counts()
    width = max(len(name) for name in counts) if counts else 4
    typer.echo(f"dataset {dataset.dataset_id}  ({dataset.total_rows()} rows)")
    for name, count in counts.items():
        marker = "*" if count else " "
        typer.echo(f" {marker} {name:<{width}}  {count:>8}")
    typer.echo("\n* non-empty tables")


@app.command("table")
def inspect_table(
    dataset_dir: Path = typer.Argument(..., help="Dataset directory, relative to the data root."),
    table: str = typer.Argument(..., help="Canonical table name."),
    limit: int = typer.Option(10, "--limit", min=1, help="Rows to print."),
    as_json: bool = typer.Option(False, "--json", help="Emit JSON rows."),
    data_root: Path | None = typer.Option(
        None, "--data-root", help=f"Override {DATA_ROOT_ENV_VAR}."
    ),
) -> None:
    """Print rows of one canonical table in canonical order."""
    dataset = _load(dataset_dir, data_root)
    if table not in table_names():
        typer.secho(f"Unknown canonical table {table!r}.", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2)
    rows = table_to_rows(dataset.table(table), table_name=table)
    if as_json:
        typer.echo(json.dumps(rows[:limit], indent=2, sort_keys=True, default=str))
        return
    typer.echo(f"{table}  ({len(rows)} rows, ordered by {', '.join(table_spec(table).order_by)})")
    for row in rows[:limit]:
        typer.echo("")
        for column in column_order(table):
            typer.echo(f"  {column:<32} {row.get(column)!r}")


@app.command("timeline")
def inspect_timeline(
    dataset_dir: Path = typer.Argument(..., help="Dataset directory, relative to the data root."),
    athlete_id: str = typer.Option(..., "--athlete", help="Athlete identifier."),
    as_json: bool = typer.Option(False, "--json", help="Emit JSON events."),
    data_root: Path | None = typer.Option(
        None, "--data-root", help=f"Override {DATA_ROOT_ENV_VAR}."
    ),
) -> None:
    """Print an athlete's events in event-time order.

    Canonical storage is event-time, so this view is a projection over the event
    tables, not an aggregation.
    """
    dataset = _load(dataset_dir, data_root)
    events: list[dict[str, Any]] = []
    for table, time_column, athlete_column in TIMELINE_TABLES:
        rows = table_to_rows(dataset.table(table), table_name=table)
        for row in rows:
            if row.get(athlete_column) != athlete_id or row.get(time_column) is None:
                continue
            events.append(
                {
                    "at": row[time_column],
                    "table": table,
                    "record_id": next(
                        (
                            str(value)
                            for key, value in row.items()
                            if key.endswith("_id") and isinstance(value, str)
                        ),
                        "",
                    ),
                    "summary": _summarize(table, row),
                }
            )
    events.sort(key=_event_sort_key)
    if as_json:
        typer.echo(json.dumps(events, indent=2, sort_keys=True, default=str))
        return
    typer.echo(f"{len(events)} events for {athlete_id}")
    for event in events:
        typer.echo(f"  {event['at']}  {event['table']:<22} {event['summary']}")


def _event_sort_key(event: dict[str, Any]) -> tuple[str, str, str]:
    """Sort events by instant, then table, then record for a stable projection."""
    return (str(event["at"]), str(event["table"]), str(event["record_id"]))


_SUMMARIZERS: dict[str, Callable[[dict[str, Any]], str]] = {
    "performed_session": lambda row: f"{row.get('session_type')} ({row.get('duration_seconds')}s)",
    "planned_session": lambda row: f"planned: {row.get('session_status')}",
    "observation": lambda row: f"{row.get('observation_type')} = {row.get('numeric_value')}",
    "velocity_observation": lambda row: f"{row.get('method')} = {row.get('mean_velocity_mps')}",
    "performance_test": lambda row: f"{row.get('test_type')} = {row.get('result_normalized')}",
    "body_measurement": lambda row: (
        f"{row.get('measurement_type')} = {row.get('value_normalized')}"
    ),
    "competition": lambda row: f"{row.get('name')} ({row.get('equipment_class')})",
    "competition_attempt": lambda row: (
        f"{row.get('lift')} #{row.get('attempt_number')} {row.get('load_kg')} {row.get('result')}"
    ),
}


def _summarize(table: str, row: dict[str, Any]) -> str:
    """Return a short human-readable label for one event row."""
    summarizer = _SUMMARIZERS.get(table)
    return "" if summarizer is None else summarizer(row)


def register_validate(target: typer.Typer) -> None:
    """Register the ``psd validate`` command on the root application."""

    @target.command("validate")
    def validate(
        dataset_dir: Path = typer.Argument(
            ..., help="Dataset directory, relative to the data root."
        ),
        strict: bool = typer.Option(False, "--strict", help="Treat warnings as errors."),
        as_json: bool = typer.Option(False, "--json", help="Emit a JSON report."),
        limit: int = typer.Option(10, "--limit", min=1, help="Issues to print."),
        data_root: Path | None = typer.Option(
            None, "--data-root", help=f"Override {DATA_ROOT_ENV_VAR}."
        ),
    ) -> None:
        """Validate a canonical dataset: declarative columns plus cross-record rules."""
        dataset = _load(dataset_dir, data_root)
        report = validate_tables(dataset.tables, strict=strict)
        if as_json:
            typer.echo(json.dumps(report.to_dict(), indent=2, sort_keys=True))
        else:
            typer.echo(f"dataset {dataset.dataset_id}")
            typer.echo(report.summary())
            for issue in report.issues[:limit]:
                typer.echo(f"  {issue.format()}")
            remaining = len(report.issues) - limit
            if remaining > 0:
                typer.echo(f"  ... and {remaining} more")
        if not report.ok:
            raise typer.Exit(code=1)
