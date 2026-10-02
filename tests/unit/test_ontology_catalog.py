"""Unit tests for the exercise catalog: integrity, coverage, and distinctness.

The catalog is data, so the tests that matter most are the ones asserting what the
data does *not* say. An ontology that quietly merged sumo into conventional, or
close-grip bench into competition bench, would still pass every "does it resolve"
test; only the distinctness tests below would catch it. The same holds for claims
*about the sport*: a note asserting that the rules prescribe a deadlift stance passes
every resolution test and is still wrong, so the stance tests assert the prose as well
as the descriptors.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime

import pyarrow as pa
import pytest

from psd.ontology import default_ontology, exercise_id_for
from psd.ontology.catalog import (
    ALIASES,
    CURATED_AMBIGUOUS_LABELS,
    EXERCISES,
    PROBE_LABELS,
    PROVISIONAL_SOURCE_SYSTEMS,
    SOURCE_SYSTEMS,
)
from psd.ontology.records import alias_records, definition_record_for
from psd.ontology.text import normalize_label
from psd.schema.models import (
    COMPETITION_PARENT_LIFTS,
    ExerciseDefinitionRecord,
    pause_flag_for_rule,
)
from psd.schema.registry import arrow_schema_for, column_order
from psd.schema.vocabulary import QualityFlag, SpecificityLevel

ONTOLOGY = default_ontology()
INGESTED_AT = datetime(2021, 1, 4, 7, 0, tzinfo=UTC)


# ---------------------------------------------------------------------------
# catalog integrity
# ---------------------------------------------------------------------------


def test_canonical_keys_are_unique_and_well_formed() -> None:
    keys = [spec.key for spec in EXERCISES]
    assert len(keys) == len(set(keys))
    for key in keys:
        assert key == key.lower()
        assert key.replace("_", "").isalnum()
        assert not key[0].isdigit()


def test_canonical_names_are_unique() -> None:
    """Two exercises may not share a display name; the name is a lookup key too."""
    names = [spec.canonical_name for spec in EXERCISES]
    assert len(names) == len(set(names))


def test_every_alias_names_a_declared_exercise() -> None:
    keys = {spec.key for spec in EXERCISES}
    for alias in ALIASES:
        assert alias.exercise_key in keys, alias.raw_label


def test_every_alias_uses_a_declared_source_system() -> None:
    for alias in ALIASES:
        assert alias.source_system in SOURCE_SYSTEMS, alias.raw_label


def test_source_systems_partition_into_known_and_provisional() -> None:
    """Every namespace is either PSD's own, format-agnostic, or provisional.

    Being provisional is not a defect: those spellings are representative of a
    logging vocabulary rather than verified vendor exports, and the distinction has
    to be recorded in data rather than remembered.
    """
    assert SOURCE_SYSTEMS >= PROVISIONAL_SOURCE_SYSTEMS
    assert PROVISIONAL_SOURCE_SYSTEMS == SOURCE_SYSTEMS & PROVISIONAL_SOURCE_SYSTEMS
    assert "psd_registry" in SOURCE_SYSTEMS
    for alias in ALIASES:
        assert alias.source_system in SOURCE_SYSTEMS, alias.raw_label


def test_provisional_aliases_are_flagged_unverified() -> None:
    """A representative spelling must announce itself as unverified in the artifact.

    The flag is data, not a comment: it survives into Parquet, where an adapter can
    act on it before trusting a vendor vocabulary it has not reconciled.
    """
    for record in alias_records(ONTOLOGY, ingested_at=INGESTED_AT):
        flagged = QualityFlag.UNVERIFIED in record.quality_flags
        assert flagged == (record.source_system in PROVISIONAL_SOURCE_SYSTEMS), record.alias_raw


def test_every_exercise_builds_a_valid_canonical_record() -> None:
    """The catalog and the canonical contract must not be able to disagree.

    A descriptor the contract rejects would mean the ontology could not be persisted
    at all, so this is checked for every exercise rather than sampled.
    """
    stamp = INGESTED_AT
    for spec in EXERCISES:
        record = definition_record_for(ONTOLOGY, spec.key, ingested_at=stamp)
        assert isinstance(record, ExerciseDefinitionRecord)
        assert record.exercise_id == exercise_id_for(spec.key)
        assert record.canonical_key == spec.key


def test_competition_lifts_record_the_rules_range_of_motion() -> None:
    for spec in EXERCISES:
        if spec.specificity_level is not SpecificityLevel.COMPETITION_LIFT:
            continue
        assert spec.range_of_motion.value == "competition", spec.key
        assert spec.parent_lift in COMPETITION_PARENT_LIFTS, spec.key


def test_competition_variations_also_live_inside_a_competition_lift() -> None:
    for spec in EXERCISES:
        if spec.specificity_level is not SpecificityLevel.COMPETITION_VARIATION:
            continue
        assert spec.parent_lift in COMPETITION_PARENT_LIFTS, spec.key


def test_pause_flag_is_derived_from_the_pause_rule_not_stated_separately() -> None:
    """The coarse flag and the precise rule are two views of one declaration.

    Deriving the flag rather than declaring it independently is what makes it
    impossible for the catalog to say ``pause=True`` while naming no pause.
    """
    for spec in EXERCISES:
        record = definition_record_for(ONTOLOGY, spec.key, ingested_at=INGESTED_AT)
        assert record.pause == pause_flag_for_rule(spec.pause_rule), spec.key


def test_competition_lifts_are_not_flagged_as_paused_variations() -> None:
    """A rules-mandated pause is part of the lift, not an added pause.

    This is the mechanism that keeps ``Competition Bench Press`` and ``Paused Bench
    Press`` distinct even though the second exists precisely to add a pause.
    """
    for key in ("squat", "bench", "deadlift", "conventional_deadlift"):
        record = definition_record_for(ONTOLOGY, key, ingested_at=INGESTED_AT)
        assert record.pause is False, key
        assert record.pause_rule.value == "competition", key


def test_an_added_pause_is_flagged_true() -> None:
    for key in ("pause_bench", "pause_squat", "pause_deadlift", "two_count_bench"):
        record = definition_record_for(ONTOLOGY, key, ingested_at=INGESTED_AT)
        assert record.pause is True, key


def test_pause_rule_members_that_the_catalog_never_uses() -> None:
    """Documents which vocabulary members are reserved rather than exercised.

    ``count_3`` is genuinely used; this guards against a later edit that quietly
    leaves a declared pause type unreachable from every exercise.
    """
    used = {spec.pause_rule.value for spec in EXERCISES}
    assert {"competition", "brief", "none", "count_2", "count_3", "long"} <= used


# ---------------------------------------------------------------------------
# the distinctions a model may care about must survive
# ---------------------------------------------------------------------------

#: ``(label, resolved key, the key it must never become)``.
#:
#: Each row is a pair of exercises that share a parent lift and differ only in the
#: descriptor the label names. The second key is the one a "simplification" of the
#: ontology would wrongly collapse them into.
SEMANTIC_DISTINCTIONS: tuple[tuple[str, str, str], ...] = (
    ("Close Grip Bench", "close_grip_bench", "bench"),
    ("Wide Grip Bench", "wide_grip_bench", "close_grip_bench"),
    ("Competition Bench", "bench", "close_grip_bench"),
    ("Low Bar Squat", "low_bar_squat", "high_bar_squat"),
    ("High Bar Squat", "high_bar_squat", "low_bar_squat"),
    ("Squat", "squat", "low_bar_squat"),
    ("Sumo Deadlift", "sumo_deadlift", "conventional_deadlift"),
    ("Conventional Deadlift", "conventional_deadlift", "sumo_deadlift"),
    ("Deadlift", "deadlift", "conventional_deadlift"),
    ("Competition Deadlift", "deadlift", "conventional_deadlift"),
    ("Paused Deadlift", "pause_deadlift", "deadlift"),
    ("Paused Bench", "pause_bench", "bench"),
    ("2ct Bench", "two_count_bench", "pause_bench"),
    ("Long Pause Bench", "long_pause_bench", "two_count_bench"),
    ("3 Count Bench Press", "three_count_bench", "two_count_bench"),
    ("Tempo Bench", "tempo_bench", "bench"),
    ("Incline Bench", "incline_bench", "bench"),
    ("Decline Bench", "decline_bench", "incline_bench"),
    ("Floor Press", "floor_press", "incline_bench"),
    ("Feet-Up Bench", "feet_up_bench", "spoto_press"),
    ("Spoto Press", "spoto_press", "board_press"),
    ("Box Squat", "box_squat", "pin_squat"),
    ("Deficit Deadlift", "deficit_deadlift", "block_pull"),
    ("Block Pull", "block_pull", "rack_pull"),
    ("RDL", "rdl", "stiff_leg_deadlift"),
    ("Trap Bar Deadlift", "trap_bar_deadlift", "conventional_deadlift"),
    ("Hack Squat", "hack_squat", "leg_press"),
    ("Romanian Deadlift", "rdl", "good_morning"),
    ("Dumbbell Curl", "dumbbell_curl", "hammer_curl"),
    ("Hammer Curl", "hammer_curl", "dumbbell_curl"),
    ("Barbell Row", "barbell_row", "pendlay_row"),
    ("Pendlay Row", "pendlay_row", "barbell_row"),
    ("Plank", "plank", "side_plank"),
    ("Side Plank", "side_plank", "plank"),
)


@pytest.mark.parametrize(("label", "resolved", "forbidden"), SEMANTIC_DISTINCTIONS)
def test_distinct_variants_do_not_collapse_together(
    label: str, resolved: str, forbidden: str
) -> None:
    """Two exercises that share a parent lift must never resolve to one exercise.

    This is the test that would fail if someone "simplified" the ontology by merging
    sumo into conventional deadlift or close-grip into competition bench.
    """
    assert resolved != forbidden
    outcome = ONTOLOGY.resolve(label, source_system="hevy")
    assert outcome.exercise_key == resolved, (
        f"{label!r} resolved to {outcome.exercise_key!r}, expected {resolved!r}"
    )
    assert outcome.exercise_key != forbidden


@pytest.mark.parametrize(("label", "resolved", "forbidden"), SEMANTIC_DISTINCTIONS)
def test_distinct_variants_carry_different_identifiers(
    label: str, resolved: str, forbidden: str
) -> None:
    assert exercise_id_for(resolved) != exercise_id_for(forbidden)


@pytest.mark.parametrize(("label", "resolved", "forbidden"), SEMANTIC_DISTINCTIONS)
def test_variants_of_a_competition_lift_stay_inside_its_family(
    label: str, resolved: str, forbidden: str
) -> None:
    """Sharing a parent lift is required for variants and forbidden for a merge.

    Two exercises may be told apart *because* they are the same lift performed
    differently. An accessory is allowed to share a label stem with a lift without
    belonging to it, which is why the accessory family is excluded here.
    """
    left = ONTOLOGY.spec(resolved).parent_lift
    right = ONTOLOGY.spec(forbidden).parent_lift
    if left in COMPETITION_PARENT_LIFTS and right in COMPETITION_PARENT_LIFTS:
        assert left == right, label


# ---------------------------------------------------------------------------
# the ontology encodes what an exercise is, not what it is worth
# ---------------------------------------------------------------------------


FORBIDDEN_FIELDS: tuple[str, ...] = (
    "transfer",
    "coefficient",
    "effectiveness",
    "specificity_score",
    "similarity",
    "rpe_adjustment",
    "loading_recommendation",
    "value",
)


def test_exercise_spec_carries_no_effectiveness_field() -> None:
    fields = set(ExerciseDefinitionRecord.model_fields)
    for name in FORBIDDEN_FIELDS:
        assert not any(name in field for field in fields), name


def test_exercise_definition_persists_no_numeric_effectiveness_column() -> None:
    for name in FORBIDDEN_FIELDS:
        assert name not in column_order("exercise_definition")
        assert name not in column_order("exercise_alias")
        assert name not in column_order("exercise_normalization")


def test_no_exercise_table_holds_a_numeric_column_at_all() -> None:
    """Not one float, so there is no back door for a per-exercise or per-mapping score.

    ``exercise_normalization`` once carried a fixed ``confidence`` per resolution
    method. It was never calibrated, so the field was removed rather than retuned,
    and this asserts the stronger property that makes that stick: adding any numeric
    effectiveness or confidence column to these three tables now fails here first.
    """

    for table in ("exercise_definition", "exercise_alias", "exercise_normalization"):
        schema = arrow_schema_for(table)
        numeric = [
            name
            for name, data_type in zip(schema.names, schema.types, strict=True)
            if pa.types.is_floating(data_type) or pa.types.is_integer(data_type)
        ]
        assert numeric == [], table


# ---------------------------------------------------------------------------
# competition rules prescribe the lift, not every observable of it
# ---------------------------------------------------------------------------


def test_the_catalog_never_claims_the_rules_require_a_deadlift_stance() -> None:
    """No note may assert a stance the rulebook does not prescribe.

    The IPF Technical Rulebook defines the deadlift by bar position, grip, the start
    from the floor, and the completed erect position; it prescribes neither a
    conventional nor a sumo stance. The catalog previously claimed otherwise, and the
    claim is what bound "competition deadlift" to one stance. Checking the declared
    prose keeps the correction from being reintroduced by an edit that looks like
    documentation.
    """
    pattern = re.compile(
        r"(?i)\b(requires?|mandates?|demands?|enforces?)\b[^.]*\b(stance|conventional|sumo)\b"
    )
    offenders = [
        f"{spec.key}: {text}"
        for spec in EXERCISES
        if spec.note is not None
        for text in (pattern.search(spec.note),)
        if text is not None
    ]
    offenders.extend(
        f"alias {alias.raw_label}: {text}"
        for alias in ALIASES
        if alias.note is not None
        for text in (pattern.search(alias.note),)
        if text is not None
    )
    assert offenders == []


def test_the_competition_deadlift_is_the_stance_unspecified_entity() -> None:
    """``deadlift`` is the lift; the two stances are variations of it."""
    competition = ONTOLOGY.spec("deadlift")
    assert competition.specificity_level is SpecificityLevel.COMPETITION_LIFT
    assert competition.stance.value == "not_specified"
    for key in ("conventional_deadlift", "sumo_deadlift"):
        style = ONTOLOGY.spec(key)
        assert style.parent_lift is competition.parent_lift
        assert style.specificity_level is SpecificityLevel.COMPETITION_VARIATION
        assert style.stance is not competition.stance
        assert style.implement == competition.implement
        assert style.bar_type == competition.bar_type
        assert style.range_of_motion == competition.range_of_motion
        assert style.pause_rule == competition.pause_rule


def test_a_competition_label_resolves_to_the_stance_unspecified_lift() -> None:
    """The discipline label names a lift, in every family that has one.

    This is the mapping the reviewer flagged: "Competition Deadlift" used to resolve
    to ``conventional_deadlift``, which silently claimed the rules mandate that stance.
    """
    open_about_stance = {"not_specified", "unknown"}
    for label, key in (
        ("Competition Squat", "squat"),
        ("Competition Bench", "bench"),
        ("Competition Deadlift", "deadlift"),
        ("Comp Deadlift", "deadlift"),
    ):
        outcome = ONTOLOGY.resolve(label, source_system="hevy")
        assert outcome.exercise_key == key, label
        assert ONTOLOGY.spec(key).stance.value in open_about_stance, label


def test_a_stated_stance_still_resolves_to_the_stance_qualified_entity() -> None:
    """The correction must not over-refuse: a source that names a stance is answered."""
    for label, key in (
        ("Conventional Deadlift", "conventional_deadlift"),
        ("Conventional DL", "conventional_deadlift"),
        ("Deadlift (Conventional)", "conventional_deadlift"),
        ("Standard Deadlift", "conventional_deadlift"),
        ("Sumo Deadlift", "sumo_deadlift"),
        ("Sumo DL", "sumo_deadlift"),
        ("Low Bar Squat", "low_bar_squat"),
        ("High Bar Squat", "high_bar_squat"),
    ):
        assert ONTOLOGY.resolve(label, source_system="generic_csv").exercise_key == key, label


# ---------------------------------------------------------------------------
# curated refusals and probes
# ---------------------------------------------------------------------------


def test_curated_labels_are_not_also_registered_as_aliases() -> None:
    """A label may not be both resolvable and deliberately unresolvable."""
    alias_texts = {alias.raw_label for alias in ALIASES}
    for spec in CURATED_AMBIGUOUS_LABELS:
        assert spec.raw_label not in alias_texts, spec.raw_label


def test_curated_labels_normalize_to_something_resolvable_as_a_label() -> None:
    """A curated refusal must still be findable after normalization."""
    for spec in CURATED_AMBIGUOUS_LABELS:
        assert normalize_label(spec.raw_label).text


def test_curated_ambiguities_all_name_declared_candidates() -> None:
    for spec in CURATED_AMBIGUOUS_LABELS:
        for key in spec.candidate_keys:
            assert key in {item.key for item in EXERCISES}, key


def test_probe_labels_include_a_blank_a_question_form_and_an_unknown_name() -> None:
    """The probes must cover the three ways a label fails, including a blank one.

    A whitespace-only label is the interesting case: it is not an error, and it is
    not a resolution either.
    """
    assert any(not label.strip() for label in PROBE_LABELS)
    assert any(label.endswith("?") for label in PROBE_LABELS)
    assert any("Zorblax" in label for label in PROBE_LABELS)


#: The heterogeneity the issue calls out, with the canonical key each label must
#: reach. Every entry is a spelling that genuinely occurs in a training log.
REQUIRED_FORMS: tuple[tuple[str, str], ...] = (
    # squat family
    ("Squat", "squat"),
    ("Competition Squat", "squat"),
    ("Low Bar Squat", "low_bar_squat"),
    ("High Bar Squat", "high_bar_squat"),
    ("Paused Squat", "pause_squat"),
    ("Tempo Squat", "tempo_squat"),
    ("Pin Squat", "pin_squat"),
    ("Box Squat", "box_squat"),
    # bench family
    ("Bench", "bench"),
    ("Bench Press", "bench"),
    ("Competition Bench", "bench"),
    ("Paused Bench", "pause_bench"),
    ("2ct Bench", "two_count_bench"),
    ("Long Pause Bench", "long_pause_bench"),
    ("Close Grip Bench", "close_grip_bench"),
    ("CGBP", "close_grip_bench"),
    ("Spoto Press", "spoto_press"),
    ("Feet-Up Bench", "feet_up_bench"),
    ("Incline Bench", "incline_bench"),
    # deadlift family
    ("Deadlift", "deadlift"),
    ("Competition Deadlift", "deadlift"),
    ("Conventional Deadlift", "conventional_deadlift"),
    ("Sumo Deadlift", "sumo_deadlift"),
    ("Paused Deadlift", "pause_deadlift"),
    ("Deficit Deadlift", "deficit_deadlift"),
    ("Block Pull", "block_pull"),
    ("Rack Pull", "rack_pull"),
    ("Romanian Deadlift", "rdl"),
    ("RDL", "rdl"),
    # machine and accessory work
    ("Leg Press", "leg_press"),
    ("Leg Curl", "leg_curl"),
    ("Leg Extension", "leg_extension"),
    ("Hack Squat", "hack_squat"),
    ("Lat Pulldown", "lat_pulldown"),
    ("Lateral Raise", "lateral_raise"),
    ("Face Pull", "face_pull"),
    ("Barbell Curl", "barbell_curl"),
    ("Dumbbell Curl", "dumbbell_curl"),
    ("Triceps Pushdown", "triceps_pushdown"),
    ("Overhead Press", "overhead_press"),
    ("Plank", "plank"),
    ("Dips", "dips"),
)


@pytest.mark.parametrize(("label", "key"), REQUIRED_FORMS)
@pytest.mark.parametrize("source_system", ["psd_registry", "hevy", "strong", "generic_csv"])
def test_required_forms_resolve_from_every_source_namespace(
    label: str, key: str, source_system: str
) -> None:
    """The required vocabulary must resolve regardless of which app wrote it.

    A cross-namespace resolution is reported as a cross-source hit rather than
    refused, because the alias is unambiguous even when this namespace never spelled
    it that way.
    """
    outcome = ONTOLOGY.resolve(label, source_system=source_system)
    assert outcome.is_resolved, outcome
    assert outcome.exercise_key == key


@pytest.mark.parametrize(("label", "key"), REQUIRED_FORMS)
def test_required_forms_resolve_case_and_punctuation_insensitively(label: str, key: str) -> None:
    assert ONTOLOGY.resolve(label.upper()).exercise_key == key
    assert ONTOLOGY.resolve(f"  {label}  ").exercise_key == key
    assert ONTOLOGY.resolve(label.replace(" ", "-")).exercise_key == key
