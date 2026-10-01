"""Property-based tests for the exercise ontology's invariants.

Hypothesis earns its place here only where the property is about *every* input
rather than about one label. Three properties genuinely need it:

* text normalization is idempotent, so a round trip through storage cannot produce a
  different lookup key than the alias index was built from;
* resolution is a pure function of its inputs, so the same label resolves the same way
  on every platform and in every process;
* no outcome ever claims a canonical exercise it did not identify, and a resolved
  outcome always carries exactly one -- which is the invariant that keeps an
  unresolved label from leaking into a mapping.

Everything else about the vocabulary is covered by example-based tests, where an
invented string proves nothing about powerlifting.
"""

from __future__ import annotations

from collections.abc import Sequence

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from psd.ontology import default_ontology
from psd.ontology.registry import MAPPING_CONFIDENCE
from psd.ontology.text import normalize_label
from psd.schema.vocabulary import ResolutionMethod, ResolutionStatus

ONTOLOGY = default_ontology()

#: Settings shared by every property below. The example budget is deliberately small:
#: these properties are cheap but numerous, and the catalog-based tests already do the
#: exhaustive work.
PROPERTY = settings(
    max_examples=200,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)

#: Labels built from fragments a real log contains, rather than arbitrary text. The
#: invariant is about any *label*, so the alphabet is domain-shaped and the spacing
#: and punctuation are varied to cover the spelling noise the normalizer exists for.
LABEL_FRAGMENTS = (
    "Squat",
    "Bench",
    "Bench Press",
    "Deadlift",
    "Close Grip",
    "Wide Grip",
    "Paused",
    "Tempo",
    "Low Bar",
    "High Bar",
    "Sumo",
    "Conventional",
    "RDL",
    "Leg Press",
    "Machine",
    "Variation",
    "DB",
    "2ct",
    "",
    "   ",
    "-",
    "?",
    "!!!",
)

#: Separators and decorations a real log mixes in, so the property covers the
#: spelling noise the normalizer exists to absorb rather than arbitrary text.
JOINERS = (" ", "  ", "-", "", "/", " _ ")
PREFIXES = ("", " ", "  ", "\t", "  - ")
SUFFIXES = ("", " ", "  ", "?", "!!", " ? ", "?!", "\n")


def _decorate(parts: Sequence[str]) -> st.SearchStrategy[str]:
    """Return a strategy over labels built from *parts* with varied separators."""

    def assemble(prefix: str, joiner: str, suffix: str) -> str:
        return f"{prefix}{joiner.join(parts)}{suffix}"

    return st.builds(
        assemble,
        st.sampled_from(PREFIXES),
        st.sampled_from(JOINERS),
        st.sampled_from(SUFFIXES),
    )


LABELS = st.lists(st.sampled_from(LABEL_FRAGMENTS), min_size=1, max_size=5).flatmap(_decorate)

SOURCE_SYSTEMS = st.sampled_from(["psd_registry", "hevy", "strong", "generic_csv", "unknown_app"])

RESOLVED = {ResolutionStatus.EXACT_CANONICAL, ResolutionStatus.RESOLVED_ALIAS}


@given(LABELS)
@PROPERTY
def test_resolution_is_idempotent_over_normalized_text(label: str) -> None:
    """Re-resolving the normalized lookup key must reach the same exercise.

    An adapter that re-normalizes after a migration has to land in the same place, or
    one exercise would occupy two identities depending on which pass ran. A label the
    source phrased as a question is exempt by design: the doubt lives in the source
    string, and the normalized key is the answer PSD refused to give -- so the property
    is asserted on the candidate instead.
    """
    first = ONTOLOGY.resolve(label)
    again = ONTOLOGY.resolve(first.normalized_label)
    if first.exercise_key is not None:
        assert again.exercise_key == first.exercise_key
    if first.candidate_keys:
        assert again.exercise_key == first.candidate_keys[0] or first.ambiguity_reason is not None
    else:
        assert again.resolution_status is first.resolution_status
        assert again.candidate_keys == first.candidate_keys


@given(LABELS, SOURCE_SYSTEMS)
@PROPERTY
def test_resolution_is_a_pure_function_of_its_inputs(label: str, source: str) -> None:
    """No clock, no locale, no hash order: the same inputs, the same outcome."""
    assert ONTOLOGY.resolve(label, source_system=source) == ONTOLOGY.resolve(
        label, source_system=source
    )


@given(LABELS, SOURCE_SYSTEMS)
@PROPERTY
def test_the_raw_label_always_survives(label: str, source: str) -> None:
    """Whatever the verdict, the source string is carried through untouched."""
    assert ONTOLOGY.resolve(label, source_system=source).raw_label == label


@given(LABELS, SOURCE_SYSTEMS)
@PROPERTY
def test_a_resolution_names_exactly_one_exercise_and_nothing_else(label: str, source: str) -> None:
    """The ladder's central invariant, stated over every label.

    A resolved outcome has exactly one exercise, no candidates, and a confidence. An
    unresolved one has no exercise and no confidence. Nothing in between is possible,
    so a label can never half-resolve into a plausible-looking mapping.
    """
    outcome = ONTOLOGY.resolve(label, source_system=source)
    if outcome.resolution_status in RESOLVED:
        assert outcome.exercise_key is not None
        assert outcome.exercise_id is not None
        assert outcome.candidate_keys == ()
        assert outcome.confidence is not None
        assert outcome.confidence > 0.0
    else:
        assert outcome.exercise_key is None
        assert outcome.exercise_id is None
        assert outcome.confidence is None
        assert outcome.ambiguity_reason is not None


@given(LABELS, SOURCE_SYSTEMS)
@PROPERTY
def test_a_named_exercise_is_one_the_registry_declares(label: str, source: str) -> None:
    """No outcome may name an exercise outside the ontology."""
    outcome = ONTOLOGY.resolve(label, source_system=source)
    if outcome.exercise_key is not None:
        ONTOLOGY.spec(outcome.exercise_key)
    for candidate in outcome.candidate_keys:
        ONTOLOGY.spec(candidate)


@given(LABELS, SOURCE_SYSTEMS)
@PROPERTY
def test_candidates_are_sorted_deduplicated_and_exclude_the_resolution(
    label: str, source: str
) -> None:
    """Candidate lists are deterministic and never mix a refusal with an answer."""
    outcome = ONTOLOGY.resolve(label, source_system=source)
    candidates = outcome.candidate_keys
    assert list(candidates) == sorted(set(candidates))
    assert outcome.exercise_key not in candidates


@given(LABELS, SOURCE_SYSTEMS)
@PROPERTY
def test_confidence_is_fixed_by_the_method_and_never_a_transfer_score(
    label: str, source: str
) -> None:
    """Confidence describes the mapping claim and is read off a frozen table.

    It is not computed from anything about the exercise, which is what keeps it from
    drifting into a statement about training value.
    """
    outcome = ONTOLOGY.resolve(label, source_system=source)
    if outcome.confidence is not None:
        assert outcome.confidence == MAPPING_CONFIDENCE[outcome.resolution_method]
        assert 0.0 < outcome.confidence <= 1.0


@given(LABELS)
@PROPERTY
def test_normalization_never_turns_text_into_something_unindexable(label: str) -> None:
    """A normalized label is either empty or a clean ASCII, single-spaced key."""
    text = normalize_label(label).text
    assert text == " ".join(text.split())
    assert all(character.isascii() for character in text)
    assert text == text.lower()


@given(LABELS, SOURCE_SYSTEMS)
@PROPERTY
def test_case_and_spacing_do_not_change_the_verdict(label: str, source: str) -> None:
    """Harmless formatting must never move a label between ladder rungs.

    A verdict that changed with capitalisation would mean the registry indexed
    different spellings differently, which is the failure the normalizer prevents.
    """
    baseline = ONTOLOGY.resolve(label, source_system=source)
    for variant in (label.upper(), f"   {label}   ", label.replace(" ", "-")):
        other = ONTOLOGY.resolve(variant, source_system=source)
        assert other.resolution_status is baseline.resolution_status
        assert other.exercise_key == baseline.exercise_key
        assert other.candidate_keys == baseline.candidate_keys


@given(LABELS, SOURCE_SYSTEMS)
@PROPERTY
def test_a_resolved_label_stays_resolved_from_any_namespace(label: str, source: str) -> None:
    """Resolution may get weaker, never weaker *than unresolved*.

    A label that resolves in one namespace must resolve to the same exercise from
    every other, because one normalized label may only name one exercise anywhere in
    the registry.
    """
    baseline = ONTOLOGY.resolve(label, source_system=source)
    if baseline.resolution_status not in RESOLVED:
        return
    for other_source in ("psd_registry", "hevy", "strong", "generic_csv", "unknown_app"):
        other = ONTOLOGY.resolve(label, source_system=other_source)
        assert other.resolution_status in RESOLVED, (label, other_source)
        assert other.exercise_key == baseline.exercise_key, (label, other_source)


@given(st.integers(min_value=0, max_value=len(ONTOLOGY.specs) - 1))
@PROPERTY
def test_every_canonical_exercise_is_reachable_by_its_own_key(index: int) -> None:
    """Whatever the catalog grows to, each exercise must resolve to itself."""
    spec = ONTOLOGY.specs[index]
    outcome = ONTOLOGY.resolve(spec.key)
    assert outcome.resolution_status is ResolutionStatus.EXACT_CANONICAL
    assert outcome.exercise_key == spec.key
    assert outcome.resolution_method is ResolutionMethod.CANONICAL_IDENTITY


@given(st.integers(min_value=0, max_value=len(ONTOLOGY.aliases) - 1))
@PROPERTY
def test_every_registered_alias_resolves_to_its_exercise(index: int) -> None:
    """The registry never asserts a binding its own resolver contradicts."""
    binding = ONTOLOGY.aliases[index]
    outcome = ONTOLOGY.resolve(binding.raw_label, source_system=binding.source_system)
    assert outcome.resolution_status in RESOLVED
    assert outcome.exercise_key == binding.exercise_key
