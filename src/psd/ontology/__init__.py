"""PSD's exercise ontology and deterministic label normalization.

Purpose
-------

Heterogeneous powerlifting logs call the same movement many things, and call
different movements many of the same things. This package makes those logs
interoperable **without pretending two exercises are equivalent**.

The separation it enforces is:

* **Identity.** A canonical exercise is a set of observable descriptors: which
  competition lift it belongs to, what carries the load, foot and hand placement,
  range of motion, pause, tempo, apparatus, laterality, and setup flags.
* **Spelling.** A source alias is a label some app wrote, bound to exactly one
  canonical exercise, with the verbatim string preserved.
* **Refusal.** A label PSD cannot place exactly stays unresolved, partially
  resolved, or ambiguous -- with a reason and, where they exist, the candidate
  readings PSD declined to choose between.

What this package will not do
-----------------------------

It records no transfer coefficient, no specificity score, and no statement that
one exercise is worth more than another. It encodes
``parent_lift = bench``, ``grip = close``, ``pause = false``,
``specificity_level = sport_specific`` -- what an exercise *is* -- and leaves how
an athlete responds to one exercise versus another for a model to learn.

Determinism
-----------

Given the ontology version, the alias-registry version, a raw label, and a source
system, :meth:`ExerciseOntology.resolve` returns the same result on Windows and on
Linux. No clock, locale, environment variable, file, or random source is read, and
the persisted artifacts are ordinary canonical datasets carrying the same manifest
and digest machinery as any other PSD artifact.
"""

from __future__ import annotations

from psd.ontology.catalog import (
    ALIASES,
    CURATED_AMBIGUOUS_LABELS,
    EXERCISES,
    PROBE_LABELS,
    PROVISIONAL_SOURCE_SYSTEMS,
    AliasSpec,
    ExerciseSpec,
    UnresolvedLabelSpec,
)
from psd.ontology.interpret import ParsedFeatures, parse_features
from psd.ontology.registry import (
    DEFAULT_SOURCE_SYSTEM,
    MAPPING_CONFIDENCE,
    AliasBinding,
    AliasCollisionError,
    ExerciseOntology,
    NormalizationOutcome,
    OntologyError,
    alias_id_for,
    default_ontology,
    exercise_id_for,
)
from psd.ontology.text import NormalizedLabel, normalize_label
from psd.versions import ALIAS_REGISTRY_VERSION, ONTOLOGY_VERSION

__all__ = (
    "ALIASES",
    "ALIAS_REGISTRY_VERSION",
    "CURATED_AMBIGUOUS_LABELS",
    "DEFAULT_SOURCE_SYSTEM",
    "EXERCISES",
    "MAPPING_CONFIDENCE",
    "ONTOLOGY_VERSION",
    "PROBE_LABELS",
    "PROVISIONAL_SOURCE_SYSTEMS",
    "AliasBinding",
    "AliasCollisionError",
    "AliasSpec",
    "ExerciseOntology",
    "ExerciseSpec",
    "NormalizationOutcome",
    "NormalizedLabel",
    "OntologyError",
    "ParsedFeatures",
    "UnresolvedLabelSpec",
    "alias_id_for",
    "default_ontology",
    "exercise_id_for",
    "normalize_label",
    "parse_features",
)
