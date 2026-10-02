"""The persisted exercise-ontology artifact.

An ontology artifact is an ordinary canonical dataset. It carries the canonical
``exercise_definition``, ``exercise_alias`` and ``exercise_normalization`` tables,
every other canonical table present and empty, and the same deterministic manifest
with per-artifact digests that any other PSD dataset has. Reusing that machinery
means the ontology inherits canonical ordering, Arrow schema enforcement, content
digests that do not depend on the Parquet writer, and manifest verification --
instead of a parallel format that would need all four rebuilt.

Determinism
-----------

:func:`build_ontology_dataset` is a pure function of the ontology and one timestamp,
and it sorts through the registry's canonical orderings. Two builds from the same
ontology therefore produce byte-identical tables and equal content digests on any
platform, which is what makes the artifact reviewable in a diff and checkable in CI.

Provenance
----------

The dataset is a *reference* artifact: its manifest cites no sources, because the
vocabulary was authored rather than ingested. The one ``source`` row it carries is
the registry itself, classified ``reference`` rather than ``synthetic``, so that every
ontology row's ``source_id`` resolves inside the artifact and no reader can mistake an
authored descriptor set for simulated athlete observations.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

from pydantic import BaseModel

from psd.ontology.records import (
    REGISTRY_SOURCE_ID,
    alias_records,
    definition_records,
    normalization_records,
    probe_outcomes,
)
from psd.ontology.registry import (
    DEFAULT_SOURCE_SYSTEM,
    ExerciseOntology,
    NormalizationOutcome,
    default_ontology,
)
from psd.provenance.environment import EnvironmentSnapshot
from psd.provenance.manifest import DatasetKind, DatasetManifest
from psd.provenance.sources import (
    ConsentBasis,
    DataRegime,
    PublicationBasis,
    RedistributionPolicy,
    SourceNature,
    SourceRecord,
)
from psd.serialization.dataset import (
    CanonicalDataset,
    DatasetWriteSpec,
    build_dataset,
    write_dataset,
)

__all__ = (
    "ONTOLOGY_EPOCH",
    "ONTOLOGY_SOURCE_SYSTEM",
    "build_ontology_dataset",
    "ontology_dataset_id",
    "ontology_relative_path",
    "ontology_source_record",
    "write_ontology",
)

#: Directory the ontology artifact lives in, relative to the data root.
ONTOLOGY_SOURCE_SYSTEM: Final[str] = "psd-ontology"

#: Fixed creation timestamp for a rebuilt artifact, so rebuilding is byte-identical.
#:
#: The manifest's ``created_at`` participates in the manifest digest but not in any
#: table's content digest, so a wall-clock default would still change the manifest
#: bytes on every run. A caller that genuinely wants wall-clock provenance passes its
#: own instant.
ONTOLOGY_EPOCH: Final[datetime] = datetime(2021, 1, 1, tzinfo=UTC)


def ontology_dataset_id(ontology: ExerciseOntology) -> str:
    """Return the stable dataset identifier for an ontology artifact."""
    return f"psd-ontology-{ontology.ontology_version.tag.rsplit('/', 1)[-1]}"


def ontology_relative_path(ontology: ExerciseOntology) -> Path:
    """Return the artifact directory, relative to the data root.

    Built from :class:`pathlib.Path` parts and a filesystem-safe rendering of the
    version tag, so the same relative path works on Windows and on Linux: a literal
    ``/`` would be a separator on one platform and illegal in a Windows filename.
    """
    safe_tag = ontology.ontology_version.tag.replace("/", "-")
    return Path("canonical") / "ontology" / safe_tag


def ontology_source_record(version: str) -> SourceRecord:
    """Return the source record that owns ontology-authored rows.

    The ontology is not an external dataset, but every canonical row must name the
    source it came from, so the registry records itself.

    The record is classified ``reference``, not ``synthetic``: the exercise ontology
    is authored reference data, so it describes no athlete and was not produced by the
    simulator. Filing it as PSD-Sim would assert that a generator produced it, and
    would leave ``SourceNature.SYNTHETIC`` meaning "not real athlete data" rather than
    "generated athlete observations".

    Args:
        version: Ontology version tag, stored as the dataset version.

    Returns:
        A ``reference``/``psd_reference`` source record for the authored vocabulary.
    """
    return SourceRecord(
        source_id=REGISTRY_SOURCE_ID,
        display_name="PSD exercise ontology registry",
        nature=SourceNature.REFERENCE,
        regime=DataRegime.REFERENCE,
        origin_system=ONTOLOGY_SOURCE_SYSTEM,
        dataset_version=version,
        license_id="apache-2.0",
        publication_basis=PublicationBasis.AUTHORED_VOCABULARY,
        consent_basis=ConsentBasis.NONE_DECLARED,
        redistribution=RedistributionPolicy.ALLOWED,
        ingested_at=ONTOLOGY_EPOCH,
        notes=(
            "Authored reference vocabulary, not an external dataset and not simulator "
            "output. Rows describe exercise semantics and carry no observations about "
            "any athlete."
        ),
    )


def build_ontology_dataset(
    ontology: ExerciseOntology,
    *,
    dataset_id: str | None = None,
    created_at: datetime | None = None,
    probe_source_system: str = DEFAULT_SOURCE_SYSTEM,
) -> CanonicalDataset:
    """Build the canonical dataset that persists an ontology.

    Args:
        ontology: Ontology to persist.
        dataset_id: Override the derived dataset identifier.
        created_at: Creation timestamp; defaults to :data:`ONTOLOGY_EPOCH` so a
            rebuild is byte-identical.
        probe_source_system: Namespace the demonstration labels are resolved against.

    Returns:
        A canonically ordered dataset holding every canonical table.

    Notes:
        The artifact deliberately persists **both** resolved and unresolved
        outcomes. An ontology that published only its successes would be
        indistinguishable from one that had silently forced every label into an
        exercise, which is the failure this ontology exists to prevent.
    """
    stamp = created_at if created_at is not None else ONTOLOGY_EPOCH
    records: dict[str, list[BaseModel]] = {
        "source": [ontology_source_record(ontology.ontology_version.tag)],
        "exercise_definition": list(definition_records(ontology, ingested_at=stamp)),
        "exercise_alias": list(alias_records(ontology, ingested_at=stamp)),
        "exercise_normalization": list(
            _audit_outcomes(ontology, probe_source_system=probe_source_system)
        ),
    }
    return build_dataset(
        records,
        dataset_id=dataset_id if dataset_id is not None else ontology_dataset_id(ontology),
        created_at=stamp,
    )


def _audit_outcomes(
    ontology: ExerciseOntology,
    *,
    probe_source_system: str,
    ingested_at: datetime | None = None,
) -> Sequence[BaseModel]:
    """Resolve every registered alias plus the refusal probes, for persistence.

    Resolving the aliases too means the artifact carries evidence that the registry
    is internally consistent: if a registered alias failed to resolve back to the
    exercise it claims, the artifact would say so, instead of the registry's own
    tables merely asserting it.
    """
    stamp = ingested_at if ingested_at is not None else ONTOLOGY_EPOCH
    alias_outcomes = [
        ontology.resolve(binding.raw_label, source_system=binding.source_system)
        for binding in ontology.aliases
    ]
    curated_outcomes = [
        ontology.resolve(spec.raw_label, source_system=probe_source_system)
        for spec in ontology.curated_ambiguities
    ]
    probes = probe_outcomes(ontology, source_system=probe_source_system)
    return normalization_records(
        _deduplicate((*alias_outcomes, *curated_outcomes, *probes)),
        ontology=ontology,
        ingested_at=stamp,
    )


def _deduplicate(outcomes: Sequence[NormalizationOutcome]) -> list[NormalizationOutcome]:
    """Drop outcomes that repeat an identical raw label within one namespace."""
    seen: set[tuple[str, str, str]] = set()
    unique: list[NormalizationOutcome] = []
    for outcome in outcomes:
        key = (outcome.source_system, outcome.raw_label, outcome.normalized_label)
        if key in seen:
            continue
        seen.add(key)
        unique.append(outcome)
    return unique


def write_ontology(
    ontology: ExerciseOntology | None = None,
    *,
    relative: Path | str | None = None,
    environment: EnvironmentSnapshot | None = None,
    data_root: Path | None = None,
) -> DatasetManifest:
    """Persist an ontology artifact and return its manifest.

    Args:
        ontology: Ontology to persist; PSD's shipped ontology when omitted.
        relative: Directory relative to the data root; derived from the ontology
            version when omitted.
        environment: Environment snapshot; captured live when omitted.
        data_root: Explicit data root; resolved from ``PSD_DATA_ROOT`` otherwise.

    Returns:
        The manifest, which is also written to ``manifest.json``.
    """
    chosen = ontology if ontology is not None else default_ontology()
    target = relative if relative is not None else ontology_relative_path(chosen)
    return write_dataset(
        build_ontology_dataset(chosen),
        target,
        DatasetWriteSpec(
            dataset_kind=DatasetKind.REFERENCE,
            dataset_name=f"PSD exercise ontology {chosen.ontology_version.tag}",
            description=_description(chosen),
        ),
        environment=environment,
        data_root=data_root,
    )


def _description(ontology: ExerciseOntology) -> str:
    return (
        f"Exercise ontology {ontology.ontology_version.tag} with alias registry "
        f"{ontology.alias_registry_version.tag}. Descriptors record what each exercise is; no "
        "transfer coefficient or training-value claim is encoded."
    )
