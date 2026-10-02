"""The exercise ontology registry and the deterministic normalization pipeline.

What this module guarantees
---------------------------

Given an ontology version, an alias-registry version, a raw label, and the
requesting source system, :meth:`ExerciseOntology.resolve` returns the same
:class:`NormalizationOutcome` on Windows and on Linux, in any process, forever.
Nothing here reads a clock, a locale, an environment variable, a file, or a random
source, and every table it consults is a frozen tuple.

The resolution ladder
---------------------

Each stage is tried in order and the first one that produces a defensible answer
wins::

    raw source label
        |
        v
    deterministic text normalization            psd.ontology.text
        |
        +-- canonical identity                   the label already *is* an
        |                                        exercise key, name, or id
        v
    curated ambiguity                           labels PSD has seen and is
        |                                        refusing to resolve
        v
    exact known alias                           same source system first, then an
        |                                        unambiguous cross-source hit
        v
    generic qualifier                           "Bench variation": family, no exercise
        |
        v
    structured interpretation                   descriptors read off the label,
        |                                        matched against existing exercises
        v
    family keyword                              "Bench something": bench family only
        |
        v
    canonical exercise OR explicit unresolved / ambiguous / unmapped result

Two rules govern the whole ladder.

**Never force a mapping to improve coverage.** Stages that cannot name one
canonical exercise produce ``PARTIAL_FAMILY``, ``AMBIGUOUS``, or ``UNMAPPED`` with
a reason, and nothing later may upgrade them. Coverage is a property of the
vocabulary, not a licence to guess.

**Raw labels always survive.** Every outcome carries the verbatim string it was
given alongside the normalized form, and every outcome records which text rules
fired, so an unexpected mapping can be traced back to the step that caused it.

Alias collisions
----------------

One normalized label may name exactly one canonical exercise *anywhere* in the
registry, not one per source system. Two apps spelling a label differently must
normalize to different text if they mean different exercises; if they normalize to
the same text and mean different exercises, the registry refuses to be built. That
is stricter than strictly necessary, and deliberately so: it means a lookup can
never depend on which source happened to register the spelling first.

Mapping evidence
----------------

The strength of a mapping claim is recorded *symbolically*, by which stage produced
it: :class:`~psd.schema.vocabulary.ResolutionMethod` says whether the label already
was the canonical identity, matched a registered alias row, matched a row belonging to
another namespace, or was read for descriptors and matched exactly one exercise. A
lookup-backed resolution additionally cites the alias row it looked up, so the claim
can be traced to a concrete artifact.

There is deliberately **no** numeric mapping score. A fixed number per method would
look like a probability without being one: nothing calibrates it against held-out
labels, and a downstream consumer reading ``0.8`` as "correct 80% of the time" would be
reading a number PSD never estimated. Nothing in this module asserts how much any
exercise is worth or how response transfers between them; models are left to learn that
from the descriptors PSD does publish.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from functools import cache
from typing import Final

from psd.ontology.catalog import (
    ALIASES,
    CURATED_AMBIGUOUS_LABELS,
    EXERCISES,
    AliasSpec,
    ExerciseSpec,
    UnresolvedLabelSpec,
)
from psd.ontology.interpret import ParsedFeatures, parse_features
from psd.ontology.text import NormalizedLabel, normalize_label
from psd.schema.identifiers import IdPrefix, make_id
from psd.schema.vocabulary import (
    AmbiguityReason,
    ParentLift,
    ResolutionMethod,
    ResolutionStatus,
)
from psd.versions import ALIAS_REGISTRY_VERSION, ONTOLOGY_VERSION, SchemaVersion

__all__ = (
    "DEFAULT_SOURCE_SYSTEM",
    "AliasBinding",
    "AliasCollisionError",
    "ExerciseOntology",
    "NormalizationOutcome",
    "OntologyError",
    "alias_id_for",
    "default_ontology",
    "exercise_id_for",
)

#: Any descriptor vocabulary PSD matches on.
_Descriptor = StrEnum

#: The member name every descriptor vocabulary uses for "not stated".
_NEUTRAL_MEMBER: Final[str] = "unknown"

#: Source system used when a caller does not name one. Resolving against PSD's own
#: namespace means a bare label is answered by the canonical vocabulary rather than
#: by any particular vendor's spelling habits.
DEFAULT_SOURCE_SYSTEM: Final[str] = "psd_registry"

#: Tokens that say "some variation of" without saying which. Their presence caps a
#: label at family level, however specific the rest of it looks.
GENERIC_QUALIFIER_TOKENS: Final[frozenset[str]] = frozenset(
    {"variation", "variations", "var", "vars"}
)


class OntologyError(ValueError):
    """Raised when an ontology declaration is internally inconsistent."""


class AliasCollisionError(OntologyError):
    """Raised when one alias would bind two canonical identities.

    Alias collisions are the failure mode that quietly corrupts an ontology: a
    second spelling is added for an exercise that already means something else, and
    from then on one string means two movements depending on which row a query
    happens to find. The registry refuses to be built in that state rather than
    picking a winner.
    """


@dataclass(frozen=True, slots=True)
class AliasBinding:
    """One alias row, resolved to a canonical exercise.

    Attributes:
        source_system: Namespace the spelling belongs to.
        raw_label: The spelling verbatim.
        normalized_label: The spelling after deterministic normalization.
        exercise_key: Which canonical exercise it names.
        alias_id: The deterministic canonical identifier of the alias row.
        mapping_status: ``EXACT_CANONICAL`` when a spelling *is* the canonical
            identity, ``RESOLVED_ALIAS`` for a known alias of one.
        note: Why this spelling maps here.
    """

    source_system: str
    raw_label: str
    normalized_label: str
    exercise_key: str
    alias_id: str
    mapping_status: ResolutionStatus = ResolutionStatus.RESOLVED_ALIAS
    note: str | None = None


@dataclass(frozen=True, slots=True)
class NormalizationOutcome:
    """The result of normalizing one raw label.

    Attributes:
        raw_label: The verbatim source string, never rewritten.
        normalized_label: The canonical spelling the pipeline worked from.
        normalization_rules: Which text rules fired, in application order.
        source_system: The namespace the label was resolved against.
        resolution_status: How far the label resolved.
        resolution_method: Which stage produced the result.
        exercise_key: The canonical exercise, when exactly one was identified.
        exercise_id: Its deterministic identifier, when identified.
        parent_lift: The family, when a family was identifiable.
        candidate_keys: The defensible readings PSD declined to choose between,
            sorted so the list is deterministic.
        ambiguity_reason: Why no canonical exercise was forced.
        source_alias_id: The alias row that matched, when one did.

    Every field is either identity, provenance, or an explicit refusal. There is no
    numeric score: the evidence is the resolution method plus, for a lookup, the alias
    row it read.
    """

    raw_label: str
    normalized_label: str
    normalization_rules: tuple[str, ...]
    source_system: str
    resolution_status: ResolutionStatus
    resolution_method: ResolutionMethod
    exercise_key: str | None = None
    exercise_id: str | None = None
    parent_lift: ParentLift = ParentLift.UNKNOWN
    candidate_keys: tuple[str, ...] = ()
    ambiguity_reason: AmbiguityReason | None = None
    source_alias_id: str | None = None

    @property
    def is_resolved(self) -> bool:
        """Whether exactly one canonical exercise was identified."""
        return self.resolution_status in (
            ResolutionStatus.EXACT_CANONICAL,
            ResolutionStatus.RESOLVED_ALIAS,
        )


def exercise_id_for(key: str) -> str:
    """Return the deterministic identifier for a canonical exercise *key*.

    Deriving the identifier from the key rather than from the display name means
    renaming "Low Bar Squat" to "Low-Bar Squat" cannot orphan every artifact that
    cites the exercise.
    """
    return make_id(IdPrefix.EXERCISE_DEFINITION, key)


def alias_id_for(source_system: str, raw_label: str, exercise_key: str) -> str:
    """Return the deterministic identifier for one alias spelling.

    Keyed on the *verbatim* spelling rather than the normalized text, so two
    spellings of one exercise (``Lateral Raise`` and ``Lateral Raises``) survive as
    two rows with distinct identifiers, each pointing at the same exercise.
    """
    return make_id(IdPrefix.EXERCISE_ALIAS, source_system, raw_label, exercise_id_for(exercise_key))


@dataclass(frozen=True, slots=True)
class _CuratedAmbiguity:
    """A curated refusal, keyed by the normalized label it applies to."""

    normalized_label: str
    spec: UnresolvedLabelSpec


@dataclass(frozen=True, slots=True)
class _AliasIndexEntry:
    """Where one normalized label is bound, and by which spellings.

    Attributes:
        exercise_key: The single canonical exercise this label names.
        source_systems: Every namespace that spells it this way.
        alias_ids: One alias identifier per namespace, sorted, so a cross-source
            lookup can cite a concrete row.
    """

    exercise_key: str
    source_systems: tuple[str, ...]
    alias_ids: tuple[str, ...]


class ExerciseOntology:
    """A versioned exercise ontology with a deterministic resolver.

    Construction validates the whole declaration up front, so an inconsistent
    ontology fails before it can answer anything rather than producing a lookup
    that silently answers the wrong question. The registry is immutable after
    construction, which is what makes a resolver safe to share and a test safe to
    run in parallel.
    """

    __slots__ = (
        "_alias_index",
        "_alias_version",
        "_aliases",
        "_curated",
        "_exercise_ids",
        "_id_keys",
        "_identity_texts",
        "_ontology_version",
        "_specs",
    )

    def __init__(
        self,
        specs: Sequence[ExerciseSpec],
        aliases: Sequence[AliasSpec],
        curated: Sequence[UnresolvedLabelSpec],
        *,
        ontology_version: SchemaVersion = ONTOLOGY_VERSION,
        alias_registry_version: SchemaVersion = ALIAS_REGISTRY_VERSION,
    ) -> None:
        """Validate and index an ontology declaration.

        Args:
            specs: Canonical exercise declarations.
            aliases: Source-specific spellings.
            curated: Labels deliberately left unresolved.
            ontology_version: Version of the exercise identities.
            alias_registry_version: Version of the alias bindings.

        Raises:
            AliasCollisionError: A spelling would identify two exercises, or is
                both a registered alias and curated as unresolved.
            OntologyError: Any other declaration is self-contradictory.
        """
        self._ontology_version = ontology_version
        self._alias_version = alias_registry_version
        self._specs = _index_specs(specs)
        self._exercise_ids = {key: exercise_id_for(key) for key in self._specs}
        self._id_keys = {value: key for key, value in self._exercise_ids.items()}
        self._identity_texts = _index_identity_texts(self._specs)
        self._aliases = _build_aliases(aliases, set(self._specs))
        self._alias_index = _index_aliases(self._aliases)
        self._curated = _index_curated(curated, set(self._specs))
        _reject_curated_alias_overlap(self._curated, self._alias_index)

    # -- introspection ------------------------------------------------------

    @property
    def ontology_version(self) -> SchemaVersion:
        """Version of the canonical exercise identities."""
        return self._ontology_version

    @property
    def alias_registry_version(self) -> SchemaVersion:
        """Version of the source-alias bindings."""
        return self._alias_version

    @property
    def specs(self) -> tuple[ExerciseSpec, ...]:
        """Canonical exercises, sorted by key."""
        return tuple(self._specs[key] for key in sorted(self._specs))

    @property
    def aliases(self) -> tuple[AliasBinding, ...]:
        """Every alias binding, in a deterministic order."""
        return tuple(
            sorted(
                self._aliases.values(),
                key=lambda item: (item.source_system, item.normalized_label, item.raw_label),
            )
        )

    @property
    def curated_ambiguities(self) -> tuple[UnresolvedLabelSpec, ...]:
        """Labels this ontology refuses to resolve, sorted by normalized text."""
        return tuple(
            entry.spec
            for entry in sorted(self._curated.values(), key=lambda item: item.normalized_label)
        )

    @property
    def source_systems(self) -> tuple[str, ...]:
        """Namespaces contributing aliases, sorted."""
        return tuple(sorted({binding.source_system for binding in self._aliases.values()}))

    def spec(self, key: str) -> ExerciseSpec:
        """Return the canonical exercise declared for *key*.

        Raises:
            KeyError: No such canonical key.
        """
        found = self._specs.get(key)
        if found is None:
            msg = f"Unknown canonical exercise key {key!r}."
            raise KeyError(msg)
        return found

    def exercise_id(self, key: str) -> str:
        """Return the deterministic identifier for *key*."""
        self.spec(key)
        return self._exercise_ids[key]

    def keys_for_parent(self, parent_lift: ParentLift) -> tuple[str, ...]:
        """Return every canonical key belonging to *parent_lift*, sorted."""
        return tuple(
            sorted(key for key, spec in self._specs.items() if spec.parent_lift is parent_lift)
        )

    # -- resolution ---------------------------------------------------------

    def resolve(
        self, raw_label: str, *, source_system: str = DEFAULT_SOURCE_SYSTEM
    ) -> NormalizationOutcome:
        """Normalize *raw_label* and report what, if anything, it identifies.

        Args:
            raw_label: The verbatim source label.
            source_system: Namespace to resolve against. A binding registered in
                this namespace is reported as a registered alias; a label that only
                another namespace binds still resolves, but is reported as a
                cross-source hit, which names a weaker evidence class.

        Returns:
            The outcome: either exactly one canonical exercise, or a structured
            account of why it does not name one.

        Raises:
            TypeError: *raw_label* is not a string. A possibly-absent source label
                is a missingness decision for the caller, not for the resolver.
        """
        label = normalize_label(raw_label)
        base = _Base(
            raw_label=raw_label,
            normalized_label=label.text,
            normalization_rules=label.rules,
            source_system=source_system,
        )
        identity_key = self._id_keys.get(raw_label.strip())
        if identity_key is not None:
            return _resolved(base, key=identity_key, method=ResolutionMethod.CANONICAL_IDENTITY)
        outcome = self._resolve_text(base, label)
        if label.question_form and outcome.is_resolved:
            return _downgrade_to_question(outcome)
        return outcome

    def _resolve_text(self, base: _Base, label: NormalizedLabel) -> NormalizationOutcome:
        """Run the resolution ladder over one already-normalized label.

        Each stage is a separate method so the ladder reads top to bottom, in the
        order the stages may be tried, with no early-exit logic tangled in.
        """
        if not label.text:
            return _finish(base, _UNMAPPED)
        for stage in (
            self._stage_identity,
            self._stage_curated,
            self._stage_alias,
            self._stage_generic_qualifier,
            self._stage_structured,
            self._stage_family_keyword,
        ):
            outcome = stage(base, label)
            if outcome is not None:
                return outcome
        return _finish(base, _UNMAPPED)

    def _stage_identity(self, base: _Base, label: NormalizedLabel) -> NormalizationOutcome | None:
        """Resolve a label that already *is* a canonical identity."""
        key = self._identity_texts.get(label.text)
        if key is None:
            return None
        return _resolved(base, key=key, method=ResolutionMethod.CANONICAL_IDENTITY)

    def _stage_curated(self, base: _Base, label: NormalizedLabel) -> NormalizationOutcome | None:
        """Apply a curated refusal for a label PSD has decided is under-specified."""
        curated = self._curated.get(label.text)
        return None if curated is None else self._from_curated(base, curated)

    def _stage_alias(self, base: _Base, label: NormalizedLabel) -> NormalizationOutcome | None:
        """Resolve a registered alias, preferring the requesting source system."""
        entry = self._alias_index.get(label.text)
        if entry is None:
            return None
        method = (
            ResolutionMethod.REGISTERED_ALIAS
            if base.source_system in entry.source_systems
            else ResolutionMethod.CROSS_SOURCE_ALIAS
        )
        return _resolved(
            base,
            key=entry.exercise_key,
            method=method,
            source_alias_id=entry.alias_ids[0],
        )

    def _stage_generic_qualifier(
        self, base: _Base, label: NormalizedLabel
    ) -> NormalizationOutcome | None:
        """Cap "a variation of X" at the family X belongs to."""
        if not _has_generic_qualifier(label.text):
            return None
        return _finish(
            base,
            _Verdict(
                status=ResolutionStatus.PARTIAL_FAMILY,
                method=ResolutionMethod.GENERIC_QUALIFIER,
                reason=AmbiguityReason.UNSPECIFIED_VARIATION,
                parent_lift=parse_features(label.text).family,
            ),
        )

    def _stage_structured(self, base: _Base, label: NormalizedLabel) -> NormalizationOutcome | None:
        """Match the label's descriptors against existing canonical exercises."""
        features = parse_features(label.text)
        if not features.has_descriptor or features.family is ParentLift.UNKNOWN:
            return None
        matches = _structural_matches(self._specs, features)
        if len(matches) == 1:
            return _resolved(
                base, key=matches[0], method=ResolutionMethod.STRUCTURED_INTERPRETATION
            )
        if len(matches) > 1:
            return _finish(
                base,
                _Verdict(
                    status=ResolutionStatus.AMBIGUOUS,
                    method=ResolutionMethod.STRUCTURED_INTERPRETATION,
                    reason=AmbiguityReason.MULTIPLE_DEFENSIBLE_MATCHES,
                    parent_lift=features.family,
                    candidates=matches,
                ),
            )
        return None

    def _stage_family_keyword(
        self, base: _Base, label: NormalizedLabel
    ) -> NormalizationOutcome | None:
        """Resolve a bare family mention to the family, and no further."""
        family = parse_features(label.text).family
        if family is ParentLift.UNKNOWN:
            return None
        return _finish(
            base,
            _Verdict(
                status=ResolutionStatus.PARTIAL_FAMILY,
                method=ResolutionMethod.FAMILY_KEYWORD,
                reason=_family_reason(label.text),
                parent_lift=family,
            ),
        )

    def _from_curated(self, base: _Base, curated: _CuratedAmbiguity) -> NormalizationOutcome:
        """Turn a curated refusal into an outcome.

        Candidate existence was checked at construction, so this only has to order
        the candidates deterministically.
        """
        spec = curated.spec
        return _finish(
            base,
            _Verdict(
                status=spec.resolution_status,
                method=ResolutionMethod.CURATED_AMBIGUOUS,
                reason=spec.reason,
                parent_lift=spec.parent_lift,
                candidates=tuple(sorted(spec.candidate_keys)),
            ),
        )


def _family_reason(normalized: str) -> AmbiguityReason:
    """Distinguish "named a machine without naming it" from "name unknown to PSD"."""
    if "machine" in normalized.split():
        return AmbiguityReason.UNSPECIFIED_MACHINE
    return AmbiguityReason.UNKNOWN_SOURCE_TAXONOMY


# ---------------------------------------------------------------------------
# construction-time indexing
# ---------------------------------------------------------------------------


def _index_specs(specs: Sequence[ExerciseSpec]) -> Mapping[str, ExerciseSpec]:
    """Index declarations by key, rejecting duplicates and empty ontologies."""
    indexed: dict[str, ExerciseSpec] = {}
    for spec in specs:
        if spec.key in indexed:
            msg = f"Duplicate canonical exercise key {spec.key!r}."
            raise OntologyError(msg)
        indexed[spec.key] = spec
    if not indexed:
        msg = "An ontology with no canonical exercises cannot resolve anything."
        raise OntologyError(msg)
    return indexed


def _index_identity_texts(specs: Mapping[str, ExerciseSpec]) -> Mapping[str, str]:
    """Map each canonical key text and name text to its canonical key.

    A key and a name can normalize to the same text -- ``squat`` and ``Squat`` --
    which is exactly why they must resolve to the same exercise. Two *different*
    keys producing the same text is a genuine collision and is rejected.
    """
    texts: dict[str, str] = {}
    for key, spec in sorted(specs.items()):
        for candidate in (key.replace("_", " "), spec.canonical_name):
            text = normalize_label(candidate).text
            if not text:
                continue
            owner = texts.get(text)
            if owner is not None and owner != key:
                msg = (
                    f"Canonical key {key!r} and canonical key {owner!r} both normalize to "
                    f"{text!r}; one spelling may identify only one exercise."
                )
                raise AliasCollisionError(msg)
            texts[text] = key
    return texts


def _build_aliases(aliases: Sequence[AliasSpec], keys: set[str]) -> Mapping[str, AliasBinding]:
    """Build every alias binding, keyed by its deterministic alias identifier."""
    bindings: dict[str, AliasBinding] = {}
    for spec in aliases:
        if spec.exercise_key not in keys:
            msg = f"Alias {spec.raw_label!r} names unknown canonical key {spec.exercise_key!r}."
            raise OntologyError(msg)
        normalized = normalize_label(spec.raw_label).text
        if not normalized:
            msg = f"Alias {spec.raw_label!r} normalizes to an empty label."
            raise OntologyError(msg)
        alias_id = alias_id_for(spec.source_system, spec.raw_label, spec.exercise_key)
        bindings[alias_id] = AliasBinding(
            source_system=spec.source_system,
            raw_label=spec.raw_label,
            normalized_label=normalized,
            exercise_key=spec.exercise_key,
            alias_id=alias_id,
            mapping_status=ResolutionStatus.RESOLVED_ALIAS,
            note=spec.note,
        )
    return bindings


def _index_aliases(bindings: Mapping[str, AliasBinding]) -> Mapping[str, _AliasIndexEntry]:
    """Group bindings by normalized label, enforcing one exercise per label.

    One normalized label may name exactly one exercise. Two raw spellings of the
    same exercise in one source are not a conflict -- they become two rows pointing
    at the same exercise -- but two *different* exercises claiming one spelling is,
    and is rejected before any lookup can depend on it.
    """
    grouped: dict[str, list[AliasBinding]] = {}
    for binding in bindings.values():
        grouped.setdefault(binding.normalized_label, []).append(binding)

    index: dict[str, _AliasIndexEntry] = {}
    for normalized, rows in sorted(grouped.items()):
        keys = {item.exercise_key for item in rows}
        if len(keys) > 1:
            msg = (
                f"Alias {normalized!r} binds to conflicting canonical identities: "
                f"{', '.join(sorted(keys))}. One alias may name exactly one exercise."
            )
            raise AliasCollisionError(msg)
        sources = tuple(sorted({item.source_system for item in rows}))
        index[normalized] = _AliasIndexEntry(
            exercise_key=min(keys),
            source_systems=sources,
            alias_ids=tuple(
                min(item.alias_id for item in rows if item.source_system == source_system)
                for source_system in sources
            ),
        )
    return index


def _index_curated(
    curated: Sequence[UnresolvedLabelSpec], keys: set[str]
) -> Mapping[str, _CuratedAmbiguity]:
    """Index curated refusals by the normalized label they apply to.

    Candidate exercises are validated here rather than at query time: a refusal that
    offers a reader a candidate which does not exist is a broken declaration, and
    finding that out on the first query rather than at construction would make the
    defect depend on which label happened to be asked about.
    """
    indexed: dict[str, _CuratedAmbiguity] = {}
    for spec in curated:
        normalized = normalize_label(spec.raw_label).text
        if not normalized:
            msg = f"Curated ambiguity {spec.raw_label!r} normalizes to an empty label."
            raise OntologyError(msg)
        if normalized in indexed:
            msg = f"Duplicate curated ambiguity for normalized label {normalized!r}."
            raise OntologyError(msg)
        missing = [key for key in spec.candidate_keys if key not in keys]
        if missing:
            msg = (
                f"Curated ambiguity {spec.raw_label!r} names unknown canonical key(s): "
                f"{', '.join(sorted(missing))}."
            )
            raise OntologyError(msg)
        indexed[normalized] = _CuratedAmbiguity(normalized_label=normalized, spec=spec)
    return indexed


def _reject_curated_alias_overlap(
    curated: Mapping[str, _CuratedAmbiguity], alias_index: Mapping[str, _AliasIndexEntry]
) -> None:
    """Reject a label that is both a registered alias and curated as unresolved.

    Such a label would make PSD contradict itself about the same string: one table
    says it names an exercise, the other says it does not. Refusing to build is the
    only honest response, because picking either would make the coverage number
    decide the semantics.
    """
    overlaps = sorted(set(curated) & set(alias_index))
    if not overlaps:
        return
    listed = ", ".join(repr(item) for item in overlaps)
    msg = (
        f"Label(s) {listed} are registered as aliases and also curated as unresolved. Remove "
        "one of the two declarations."
    )
    raise AliasCollisionError(msg)


def _structural_matches(
    specs: Mapping[str, ExerciseSpec], features: ParsedFeatures
) -> tuple[str, ...]:
    """Return the canonical keys whose descriptors satisfy every parsed feature.

    A candidate whose descriptor is ``UNKNOWN`` never satisfies a feature the label
    stated: "paused squat" must not match an exercise that simply never said whether
    it pauses. A candidate is only reachable when the label's family also matches,
    and only when the label said at least one thing beyond the family.
    """
    matches = [
        key
        for key, spec in sorted(specs.items())
        if spec.parent_lift is features.family
        and _agrees(spec.implement, features.implement)
        and _agrees(spec.stance, features.stance)
        and _agrees(spec.grip, features.grip)
        and _agrees(spec.range_of_motion, features.range_of_motion)
        and _agrees(spec.pause_rule, features.pause_rule)
        and _agrees(spec.tempo, features.tempo)
    ]
    return tuple(matches)


def _agrees(declared: _Descriptor, parsed: _Descriptor) -> bool:
    """Whether a declared descriptor satisfies a parsed one.

    Every descriptor vocabulary in :mod:`psd.schema.vocabulary` names its neutral
    member ``unknown``, and that member means "not stated" on both sides of this
    comparison. On the parsed side it means the label said nothing, so any declared
    value satisfies it. On the declared side it means the exercise never recorded
    the property, which never satisfies a label that *did* state it -- otherwise
    "paused squat" would match an exercise that merely lacks a pause rule.
    """
    if parsed.value == _NEUTRAL_MEMBER:
        return True
    return declared is parsed


def _has_generic_qualifier(normalized: str) -> bool:
    """Whether the label says "some variation of" without saying which."""
    return bool(set(normalized.split()) & GENERIC_QUALIFIER_TOKENS)


# ---------------------------------------------------------------------------
# outcome construction
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Base:
    """The fields every outcome carries, before a verdict is applied."""

    raw_label: str
    normalized_label: str
    normalization_rules: tuple[str, ...]
    source_system: str


@dataclass(frozen=True, slots=True)
class _Verdict:
    """The decision one resolution stage reached.

    Attributes:
        status: How far the label resolved.
        method: Which stage produced the decision.
        reason: Why no canonical exercise was forced, for a non-resolved outcome.
        parent_lift: The family, when a family was identifiable.
        candidates: The defensible readings the stage declined to choose between.
    """

    status: ResolutionStatus
    method: ResolutionMethod
    reason: AmbiguityReason | None = None
    parent_lift: ParentLift = ParentLift.UNKNOWN
    candidates: tuple[str, ...] = ()


#: The verdict every stage falls through to: nothing defensible was found.
_UNMAPPED: Final[_Verdict] = _Verdict(
    status=ResolutionStatus.UNMAPPED,
    method=ResolutionMethod.NO_MATCH,
    reason=AmbiguityReason.UNKNOWN_SOURCE_TAXONOMY,
)


def _resolved(
    base: _Base,
    *,
    key: str,
    method: ResolutionMethod,
    source_alias_id: str | None = None,
) -> NormalizationOutcome:
    return NormalizationOutcome(
        raw_label=base.raw_label,
        normalized_label=base.normalized_label,
        normalization_rules=base.normalization_rules,
        source_system=base.source_system,
        resolution_status=(
            ResolutionStatus.EXACT_CANONICAL
            if method is ResolutionMethod.CANONICAL_IDENTITY
            else ResolutionStatus.RESOLVED_ALIAS
        ),
        resolution_method=method,
        exercise_key=key,
        exercise_id=exercise_id_for(key),
        source_alias_id=source_alias_id,
    )


def _finish(base: _Base, verdict: _Verdict) -> NormalizationOutcome:
    """Return the outcome for a verdict that named no canonical exercise.

    Every unresolved path goes through here, and none of them carries an alias
    reference: a result that declined to map a label has no mapping claim to trace.
    """
    return NormalizationOutcome(
        raw_label=base.raw_label,
        normalized_label=base.normalized_label,
        normalization_rules=base.normalization_rules,
        source_system=base.source_system,
        resolution_status=verdict.status,
        resolution_method=verdict.method,
        parent_lift=verdict.parent_lift,
        candidate_keys=tuple(sorted(verdict.candidates)),
        ambiguity_reason=verdict.reason,
    )


def _downgrade_to_question(outcome: NormalizationOutcome) -> NormalizationOutcome:
    """Convert a resolution into an ambiguity because the source asked a question.

    A source label ending in a question mark is the source recording doubt: someone
    typed "Leg press?" because they were not sure what the set was. The ontology can
    look that label up, but resolving it would answer a question the source did not
    answer, so the candidate is recorded and the doubt is preserved.
    """
    return replace(
        outcome,
        resolution_status=ResolutionStatus.AMBIGUOUS,
        resolution_method=ResolutionMethod.QUESTION_FORM,
        exercise_key=None,
        exercise_id=None,
        candidate_keys=(outcome.exercise_key,) if outcome.exercise_key is not None else (),
        ambiguity_reason=AmbiguityReason.QUESTION_FORM_LABEL,
        source_alias_id=None,
    )


@cache
def default_ontology() -> ExerciseOntology:
    """Return the ontology PSD ships, built once and shared.

    The catalog is frozen module-level data, so the registry derived from it is
    constant for the life of the process. :func:`functools.cache` gives that
    guarantee without a mutable module global, and building on first use keeps
    ``import psd.ontology`` cheap for callers that only want the constants.
    """
    return ExerciseOntology(EXERCISES, ALIASES, CURATED_AMBIGUOUS_LABELS)
