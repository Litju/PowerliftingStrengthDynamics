"""Deterministic builders shared by every fixture.

Design rules
------------

* **Deterministic identifiers.** Every id is derived with
  :func:`psd.schema.identifiers.make_id` from the record's structural position, so
  the same fixture always produces the same identifiers and therefore the same
  digests.
* **Deterministic timestamps.** All instants derive from :data:`BASE_INSTANT`
  through :func:`at`; nothing calls ``now()``. A fixture is a fixed input, not a
  snapshot.
* **Contracts, not dicts.** Fixtures build the same Pydantic records a real
  ingestion would, so a fixture cannot encode a state the contracts reject.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, TypeVar, cast

from pydantic import BaseModel

from psd.provenance.sources import (
    ConsentBasis,
    DataRegime,
    RedistributionPolicy,
    SourceNature,
    SourceRecord,
)
from psd.schema.identifiers import IdPrefix, derive_id, make_id
from psd.schema.models import (
    AthleteRecord,
    AthleteSourceLinkRecord,
    CompetitionAttemptRecord,
    CompetitionRecord,
    CompetitionReportedResultRecord,
    ExerciseAliasRecord,
    ExerciseDefinitionRecord,
    ProgramRecord,
    ProgramVersionRecord,
)
from psd.schema.registry import table_names
from psd.schema.vocabulary import (
    AttemptOrderBasis,
    AttemptResult,
    CompetitionResultKind,
    EquipmentClass,
    EventTimePrecision,
    IdentityLinkMethod,
    IdentityStatus,
    ImplementType,
    Laterality,
    LiftType,
    ParentLift,
    SexCategory,
    SpecificityLevel,
)
from psd.units import MassUnit, normalize_mass

__all__ = (
    "ATHLETE_ID",
    "BASE_INSTANT",
    "HISTORY_START",
    "HistoryBuilder",
    "SourceIds",
    "add_athlete",
    "add_competition",
    "add_exercises",
    "add_program",
    "add_source_records",
    "at",
    "counts",
    "kg",
    "lb",
    "program_version_of",
)

BASE_INSTANT = datetime(2021, 1, 4, 7, 0, tzinfo=UTC)
HISTORY_START = datetime(2021, 1, 4, 0, 0, tzinfo=UTC)

ModelT = TypeVar("ModelT", bound=BaseModel)

#: Deterministic athlete identity used by the fixtures.
ATHLETE_ID = make_id(IdPrefix.ATHLETE, "hevy:athlete-001")


def at(day_offset: int, hour: float = 0.0) -> datetime:
    """Return midnight UTC *day_offset* days after :data:`BASE_INSTANT`, shifted.

    Fixtures express every instant as "day *n* at *h* o'clock" so a timestamp can
    never acquire an accidental second day boundary through arithmetic.
    """
    return BASE_INSTANT + timedelta(days=day_offset, hours=hour)


def kg(value: float) -> float:
    """Return *value* as a kilogram mass."""
    return normalize_mass(value, MassUnit.KG)


def lb(value: float) -> tuple[float, float]:
    """Return ``(raw, normalized_kg)`` for a pound source value."""
    return value, normalize_mass(value, MassUnit.LB)


def counts(records: Mapping[str, Sequence[BaseModel]]) -> dict[str, int]:
    """Return row counts per table for tables that have rows."""
    return {name: len(rows) for name, rows in sorted(records.items()) if rows}


@dataclass(frozen=True, slots=True)
class SourceIds:
    """Identifiers of the sources the fixtures draw on.

    Attributes:
        training: A real training-log export.
        competition: A real competition results source.
        synthetic: A synthetic generator used by PSD-Sim-style fixtures.
    """

    training: str = "src_hevy_athlete001_202401"
    competition: str = "src_openpowerlifting_2024"
    synthetic: str = "src_psdsim_generator_v1"


@dataclass(slots=True)
class HistoryBuilder:
    """Accumulates validated records per canonical table.

    Attributes:
        athlete_id: Athlete the history belongs to.
        source_id: Default source for records that omit one.
        ingested_at: Ingestion timestamp stamped on every record.
    """

    athlete_id: str
    source_id: str
    ingested_at: datetime = field(default_factory=lambda: at(0, 9))
    _records: dict[str, list[BaseModel]] = field(
        default_factory=lambda: cast("dict[str, list[BaseModel]]", {})
    )

    def add(self, table: str, record: ModelT) -> ModelT:
        """Add a record and return it, so callers can capture derived ids."""
        self._records.setdefault(table, []).append(record)
        return record

    def extend(self, table: str, records: Iterable[ModelT]) -> None:
        """Add several records to one table."""
        for record in records:
            self.add(table, record)

    def build(self) -> dict[str, list[BaseModel]]:
        """Return every canonical table, filling absent tables as empty lists."""
        return {name: list(self._records.get(name, ())) for name in table_names()}

    def counts(self) -> dict[str, int]:
        """Return row counts per table for tables that have rows."""
        return counts(self._records)

    def provenance(self, **overrides: Any) -> dict[str, Any]:
        """Return the shared provenance columns for a record.

        Args:
            **overrides: Fields to override, for example ``source_id``.

        Returns:
            A mapping suitable for ``**`` expansion into a record constructor.
        """
        payload: dict[str, Any] = {
            "source_id": self.source_id,
            "source_record_key": None,
            "source_record_hash": None,
            "ingested_at": self.ingested_at,
            "quality_flags": (),
            "missingness_reason": None,
        }
        payload.update(overrides)
        return payload


#: Default source identifiers used when a caller does not supply its own.
DEFAULT_SOURCE_IDS: SourceIds = SourceIds()


def add_source_records(
    builder: HistoryBuilder, *, ids: SourceIds = DEFAULT_SOURCE_IDS
) -> dict[str, SourceRecord]:
    """Add the real and synthetic source records to a builder.

    Args:
        builder: Target builder.
        ids: Source identifiers to use for the exported log and the two synthetic
            sources. Defaults to :data:`DEFAULT_SOURCE_IDS`.
    """
    records: dict[str, SourceRecord] = {
        "training": SourceRecord(
            source_id=ids.training,
            display_name="Athlete training log export",
            nature=SourceNature.REAL,
            regime=DataRegime.REAL,
            origin_system="hevy",
            dataset_version="2024-01",
            license_id="proprietary-export",
            consent_basis=ConsentBasis.USER_CONSENT,
            redistribution=RedistributionPolicy.NOT_ALLOWED,
            snapshot_date=at(1),
            ingested_at=at(1),
        ),
        "competition": SourceRecord(
            source_id=ids.competition,
            display_name="OpenPowerlifting export",
            nature=SourceNature.REAL,
            regime=DataRegime.COMP,
            origin_system="openpowerlifting",
            dataset_version="2024.03",
            license_id="cc-by-sa-4.0",
            license_url="https://creativecommons.org/licenses/by-sa/4.0/",
            consent_basis=ConsentBasis.PUBLIC_LICENSE,
            redistribution=RedistributionPolicy.ALLOWED,
            snapshot_date=at(2),
            ingested_at=at(2),
        ),
        "synthetic": SourceRecord(
            source_id=ids.synthetic,
            display_name="PSD-Sim generator",
            nature=SourceNature.SYNTHETIC,
            regime=DataRegime.SIM,
            origin_system="psd-sim",
            dataset_version="0.1.0",
            license_id="apache-2.0",
            consent_basis=ConsentBasis.PUBLIC_LICENSE,
            redistribution=RedistributionPolicy.ALLOWED,
            ingested_at=at(0),
        ),
    }
    for record in records.values():
        builder.add("source", record)
    return records


def add_athlete(
    builder: HistoryBuilder,
    *,
    athlete_id: str = ATHLETE_ID,
    pseudonym: str = "athlete-001",
    is_synthetic: bool = False,
    identity_status: IdentityStatus = IdentityStatus.SINGLE_SOURCE_VERIFIED,
    source_id: str | None = None,
) -> None:
    """Add one athlete and its source link to a builder.

    Args:
        builder: Target builder.
        athlete_id: Athlete identity.
        pseudonym: Source-side key for the athlete.
        is_synthetic: Whether the athlete is simulator-generated, in which case
            demographic fields are left null rather than invented.
        identity_status: How well the identity is corroborated.
        source_id: Source the athlete record came from; defaults to the builder's.
    """

    effective_source = source_id or builder.source_id
    builder.add(
        "athlete",
        AthleteRecord(
            athlete_id=athlete_id,
            pseudonym=pseudonym,
            identity_status=identity_status,
            is_synthetic=is_synthetic,
            synthetic_regime="psd_sim" if is_synthetic else None,
            sex_category=None if is_synthetic else SexCategory.MALE,
            birth_year=None if is_synthetic else 1995,
            country_code=None if is_synthetic else "ES",
            **builder.provenance(source_id=effective_source, source_record_key=pseudonym),
        ),
    )
    builder.add(
        "athlete_source_link",
        AthleteSourceLinkRecord(
            athlete_id=athlete_id,
            source_athlete_key=pseudonym,
            link_method=IdentityLinkMethod.SOURCE_EXPLICIT_ID,
            link_confidence=1.0,
            is_primary=True,
            **builder.provenance(source_id=effective_source),
        ),
    )


def add_exercises(builder: HistoryBuilder) -> dict[str, ExerciseDefinitionRecord]:
    """Add the canonical exercise vocabulary and return it by short name."""

    definitions: tuple[tuple[str, str, ParentLift, SpecificityLevel, bool], ...] = (
        ("squat", "Competition Squat", ParentLift.SQUAT, SpecificityLevel.COMPETITION_LIFT, False),
        (
            "bench",
            "Competition Bench Press",
            ParentLift.BENCH,
            SpecificityLevel.COMPETITION_LIFT,
            False,
        ),
        (
            "deadlift",
            "Competition Deadlift",
            ParentLift.DEADLIFT,
            SpecificityLevel.COMPETITION_LIFT,
            False,
        ),
        (
            "pause_bench",
            "Paused Bench Press",
            ParentLift.BENCH,
            SpecificityLevel.COMPETITION_VARIATION,
            True,
        ),
        (
            "rdl",
            "Romanian Deadlift",
            ParentLift.DEADLIFT,
            SpecificityLevel.SPORT_SPECIFIC,
            False,
        ),
        (
            "lateral_raise",
            "Lateral Raise",
            ParentLift.ACCESSORY,
            SpecificityLevel.ACCESSORY,
            False,
        ),
    )
    exercises: dict[str, ExerciseDefinitionRecord] = {}
    for name, canonical_name, parent_lift, specificity, is_paused in definitions:
        exercise_id = make_id(IdPrefix.EXERCISE_DEFINITION, name)
        exercises[name] = builder.add(
            "exercise_definition",
            ExerciseDefinitionRecord(
                exercise_id=exercise_id,
                canonical_name=canonical_name,
                parent_lift=parent_lift,
                specificity_level=specificity,
                implement=ImplementType.BARBELL,
                laterality=Laterality.BILATERAL,
                range_of_motion="competition"
                if specificity
                in (SpecificityLevel.COMPETITION_LIFT, SpecificityLevel.COMPETITION_VARIATION)
                else None,
                pause=is_paused,
                **builder.provenance(source_record_key=name),
            ),
        )
        builder.add(
            "exercise_alias",
            ExerciseAliasRecord(
                exercise_alias_id=make_id(IdPrefix.EXERCISE_ALIAS, name),
                exercise_id=exercise_id,
                alias_raw=canonical_name,
                alias_normalized=name,
                **builder.provenance(source_record_key=name),
            ),
        )
    return exercises


def add_program(
    builder: HistoryBuilder,
    *,
    versions: Sequence[tuple[str, int]] = (("v1", 0),),
) -> dict[str, ProgramVersionRecord]:
    """Add a program and the requested versions; return the versions by label.

    Args:
        builder: Target builder.
        versions: ``(label, effective_from_day_offset)`` pairs, in order.

    Returns:
        The created version records keyed by label.

    Each version is closed at the next version's ``effective_from``, so the
    canonical ``[effective_from, effective_to)`` intervals tile the timeline
    without overlapping. Only the final version stays open.
    """

    program_id = make_id(IdPrefix.PROGRAM, "block-zero")
    builder.add(
        "program",
        ProgramRecord(
            program_id=program_id,
            name="Block Zero",
            description="Base strength block used by the PSD fixtures.",
            **builder.provenance(source_record_key="block-zero"),
        ),
    )
    result: dict[str, ProgramVersionRecord] = {}
    ordered = sorted(versions, key=lambda item: item[1])
    for index, (label, offset) in enumerate(ordered):
        following = ordered[index + 1][1] if index + 1 < len(ordered) else None
        record = builder.add(
            "program_version",
            ProgramVersionRecord(
                program_version_id=derive_id(program_id, IdPrefix.PROGRAM_VERSION, label),
                program_id=program_id,
                version_label=label,
                effective_from=at(offset),
                effective_to=None if following is None else at(following),
                authored_at=at(offset, -2),
                change_summary=f"{label} of the base block",
                supersedes_program_version_id=(
                    result[ordered[index - 1][0]].program_version_id if index else None
                ),
                **builder.provenance(source_record_key=label),
            ),
        )
        result[label] = record
    return result


def program_version_of(
    program: Mapping[str, ProgramVersionRecord], offset: int
) -> ProgramVersionRecord:
    """Return the program version in force at *offset* days.

    Args:
        program: Versions keyed by label, oldest first.
        offset: Days after the base instant.

    Returns:
        The latest version whose effective instant is at or before *offset*.
    """
    in_force: ProgramVersionRecord | None = None
    for version in program.values():
        elapsed = (version.effective_from - BASE_INSTANT).days
        if offset < elapsed:
            continue
        if in_force is None or version.effective_from >= in_force.effective_from:
            in_force = version
    if in_force is not None:
        return in_force
    return min(program.values(), key=lambda version: version.effective_from)


def add_competition(
    builder: HistoryBuilder,
    *,
    competition_id: str,
    athlete_id: str,
    offset: int,
    attempts: Sequence[tuple[LiftType, int, float | None, AttemptResult]],
    source_id: str,
    reported: Sequence[tuple[CompetitionResultKind, float, bool]] = (),
) -> None:
    """Add one competition with its attempts and reported results.

    Args:
        builder: Target builder.
        competition_id: Deterministic competition identifier.
        athlete_id: Athlete who competed.
        offset: Days after the base instant.
        attempts: ``(lift, attempt_number, load_kg, result)`` tuples. ``load_kg`` is
            null when the source reported an attempt slot that was never taken, such
            as a withdrawal; zero is never used for that. A lift with fewer than
            three attempts simply has no rows for the rest, which is how a missing
            attempt stays missing.
        source_id: Source the competition came from.
        reported: ``(result_kind, value, is_derived)`` tuples.
    """

    builder.add(
        "competition",
        CompetitionRecord(
            competition_id=competition_id,
            athlete_id=athlete_id,
            competition_date=at(offset),
            event_time_precision=EventTimePrecision.DATE_ONLY,
            name="Regional Powerlifting Open",
            federation="fixture-federation",
            sanctioning_body="fixture-sanctioning-body",
            location="Fixture City",
            equipment_class_raw="raw",
            equipment_class=EquipmentClass.RAW,
            weight_class_raw="-93",
            bodyweight_raw=91.4,
            bodyweight_unit="kg",
            bodyweight_kg=kg(91.4),
            participation_status="competed",
            is_championship=False,
            **builder.provenance(source_id=source_id, source_record_key=competition_id),
        ),
    )
    for lift, attempt_number, load, result in attempts:
        builder.add(
            "competition_attempt",
            CompetitionAttemptRecord(
                competition_attempt_id=make_id(
                    IdPrefix.COMPETITION_ATTEMPT, competition_id, lift, attempt_number
                ),
                competition_id=competition_id,
                athlete_id=athlete_id,
                lift=lift,
                attempt_number=attempt_number,
                attempt_order_basis=AttemptOrderBasis.SOURCE_EXPLICIT,
                attempt_time=None,
                load_raw=load,
                load_unit="kg" if load is not None else None,
                load_kg=load,
                result=result,
                is_opener=attempt_number == 1,
                **builder.provenance(
                    source_id=source_id,
                    source_record_key=f"{competition_id}:{lift}:{attempt_number}",
                ),
            ),
        )
    for kind, value, is_derived in reported:
        builder.add(
            "competition_reported_result",
            CompetitionReportedResultRecord(
                competition_reported_result_id=make_id(
                    IdPrefix.COMPETITION_REPORTED_RESULT, competition_id, kind
                ),
                competition_id=competition_id,
                athlete_id=athlete_id,
                result_kind=kind,
                value=value,
                unit="kg" if kind.endswith("best") or kind == "total" else None,
                is_derived=is_derived,
                **builder.provenance(
                    source_id=source_id, source_record_key=f"{competition_id}:{kind}"
                ),
            ),
        )
