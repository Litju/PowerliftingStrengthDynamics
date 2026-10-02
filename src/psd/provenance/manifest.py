"""Dataset manifests.

Every persisted canonical dataset carries a manifest recording the schema
version, the sources it draws on, the transformation lineage, the environment
that produced it, and the SHA-256 of each artifact. Manifests are JSON per the
locked stack; their serialization is deterministic (sorted keys, fixed
timestamp format) so that a manifest can be hashed and compared.

Determinism note: manifest content intentionally includes ``code_commit`` and
``is_dirty_tree``. A dirty working tree means the artifact is not reproducible
from the commit alone, so that flag is part of the record. Canonical *table*
digests never depend on these values -- see
:mod:`psd.serialization.canonical`.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from datetime import datetime
from enum import StrEnum
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Final, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from psd.provenance.environment import EnvironmentSnapshot
from psd.provenance.sources import DataRegime, SourceNature, SourceRecord
from psd.timeutil import require_aware
from psd.versions import MANIFEST_VERSION, SchemaVersion

__all__ = (
    "ArtifactRef",
    "DatasetKind",
    "DatasetManifest",
    "LineageEntry",
    "manifest_digest",
)

#: The regime a cited source's nature forces, where the nature is more than "real".
#:
#: Mirrors the rule in :class:`~psd.provenance.sources.SourceRecord`. It is repeated
#: here because a manifest may be read from disk without revalidating every source,
#: and the two are the two places a misfiled regime would otherwise hide.
_REGIME_BY_NATURE: Final[Mapping[SourceNature, DataRegime]] = {
    SourceNature.SYNTHETIC: DataRegime.SIM,
    SourceNature.REFERENCE: DataRegime.REFERENCE,
}


class DatasetKind(StrEnum):
    """What a canonical dataset represents.

    ``MIXED`` is deliberately restricted: mixing real and synthetic rows in one
    canonical dataset defeats the purpose of keeping the two distinguishable,
    so it is rejected by validation.
    """

    COMPETITION_HISTORY = "competition_history"
    TRAINING_HISTORY = "training_history"
    PROSPECTIVE_TRAINING = "prospective_training"
    SYNTHETIC = "synthetic"
    MIXED = "mixed"
    #: A versioned reference artifact such as the exercise ontology: vocabulary
    #: and structure, never observations about an athlete. A reference dataset
    #: cites no sources because it was authored, not ingested.
    REFERENCE = "reference"


class ArtifactRef(BaseModel):
    """A persisted artifact with its integrity digest.

    Attributes:
        name: Logical table or artifact name, for example ``performed_set``.
        relative_path: Path relative to the dataset root.
        sha256: Lowercase hex SHA-256 of the file bytes.
        byte_size: File size in bytes.
        row_count: Number of rows, for tabular artifacts.
        content_sha256: Digest of the canonical logical content, independent of
            the Parquet writer. Equal for the same logical table regardless of
            row-group or compression settings.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1, max_length=128)
    relative_path: str = Field(min_length=1, max_length=1024)
    sha256: str = Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")
    content_sha256: str = Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")
    byte_size: int = Field(ge=0)
    row_count: int = Field(ge=0)

    @model_validator(mode="after")
    def _reject_non_relative_path(self) -> Self:
        """Reject any path that is not a portable relative POSIX path.

        A manifest is a persisted artifact: it may be verified on a different
        platform than the one that wrote it. ``Path.is_absolute`` alone is
        platform-dependent, so ``C:/data/x.parquet`` would be rejected on Windows
        and silently accepted on Linux. The check therefore rejects a path that is
        absolute on *any* platform, that carries a drive letter or UNC prefix, or
        that uses a backslash separator, which keeps manifests identical across
        operating systems.
        """
        candidate = self.relative_path
        reasons: list[str] = []
        if Path(candidate).is_absolute() or PurePosixPath(candidate).is_absolute():
            reasons.append("it is absolute")
        if PureWindowsPath(candidate).is_absolute():
            reasons.append("it is absolute on Windows")
        if "\\" in candidate:
            reasons.append("it uses a backslash separator")
        if reasons:
            msg = (
                f"relative_path must be a portable relative POSIX path; got "
                f"{candidate!r}: {'; '.join(reasons)}."
            )
            raise ValueError(msg)
        return self


class LineageEntry(BaseModel):
    """One transformation step in a dataset's derivation.

    Attributes:
        provenance_id: Stable identifier for this lineage row.
        dataset_id: Dataset the step produced.
        parent_dataset_id: Dataset the step consumed, when there is exactly one.
        transform_name: Logical transform name.
        transform_version: Version of that transform.
        code_commit: Commit of the code performing the transform.
        config_sha256: SHA-256 of the human-authored configuration used.
        schema_version: Canonical schema version after the transform.
        random_seed: Seed used, when the transform is stochastic.
        created_at: Timezone-aware UTC creation timestamp.
        description: Human-readable description of the step.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    provenance_id: str = Field(min_length=1, max_length=160)
    dataset_id: str = Field(min_length=1, max_length=160)
    parent_dataset_id: str | None = Field(default=None, max_length=160)
    transform_name: str = Field(min_length=1, max_length=160)
    transform_version: str = Field(min_length=1, max_length=64)
    code_commit: str | None = Field(default=None, max_length=64)
    config_sha256: str | None = Field(default=None, max_length=64, pattern=r"^[0-9a-f]{64}$")
    schema_version: str = Field(min_length=1, max_length=64)
    random_seed: int | None = None
    created_at: datetime
    description: str | None = Field(default=None, max_length=2048)

    @model_validator(mode="after")
    def _require_aware_created_at(self) -> Self:
        object.__setattr__(self, "created_at", require_aware(self.created_at, field="created_at"))
        return self


class DatasetManifest(BaseModel):
    """Machine-readable description of a persisted canonical dataset.

    Attributes:
        dataset_id: Stable dataset identifier.
        dataset_name: Human-readable name.
        dataset_kind: Whether the dataset holds competition, training,
            prospective, or synthetic records.
        schema_version: Canonical schema version of the persisted tables.
        manifest_version: Version of the manifest contract itself.
        created_at: Timezone-aware UTC creation timestamp.
        environment: Interpreter, platform, lockfile, and code state.
        sources: Every source the dataset draws on, with licenses and consent.
        artifacts: Persisted artifacts with integrity digests.
        lineage: Ordered transformation steps.
        notes: Human-readable notes.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    dataset_id: str = Field(min_length=1, max_length=160)
    dataset_name: str = Field(min_length=1, max_length=256)
    dataset_kind: DatasetKind
    schema_version: str = Field(min_length=1, max_length=64)
    manifest_version: str = Field(default=MANIFEST_VERSION.tag, min_length=1, max_length=64)
    created_at: datetime
    environment: EnvironmentSnapshot
    sources: tuple[SourceRecord, ...] = ()
    artifacts: tuple[ArtifactRef, ...] = ()
    lineage: tuple[LineageEntry, ...] = ()
    notes: str | None = Field(default=None, max_length=4096)

    @field_validator("created_at")
    @classmethod
    def _require_aware_created_at(cls, value: datetime) -> datetime:
        return require_aware(value, field="created_at")

    @model_validator(mode="after")
    def _check_consistency(self) -> Self:
        """Enforce dataset-level source semantics and unique keys."""
        source_ids = [source.source_id for source in self.sources]
        if len(source_ids) != len(set(source_ids)):
            msg = "Duplicate source_id entries in manifest sources."
            raise ValueError(msg)

        natures = {source.nature for source in self.sources}
        if self.dataset_kind is DatasetKind.MIXED and len(natures) > 1:
            msg = (
                "Mixed real/synthetic datasets are rejected: keep PSD-Sim data in its own "
                "regime so real/synthetic semantics remain distinguishable."
            )
            raise ValueError(msg)
        if self.dataset_kind is DatasetKind.SYNTHETIC and SourceNature.REAL in natures:
            msg = "A synthetic dataset cannot cite real sources."
            raise ValueError(msg)
        if (
            self.dataset_kind is not DatasetKind.SYNTHETIC
            and self.dataset_kind is not DatasetKind.MIXED
            and SourceNature.SYNTHETIC in natures
        ):
            msg = (
                f"Dataset kind {self.dataset_kind.value!r} cannot cite synthetic sources; "
                "PSD keeps real and synthetic regimes explicitly separate."
            )
            raise ValueError(msg)
        if self.dataset_kind is not DatasetKind.REFERENCE and SourceNature.REFERENCE in natures:
            msg = (
                f"Dataset kind {self.dataset_kind.value!r} cannot cite reference sources; "
                "authored vocabulary describes concepts, not ingested observations, and "
                "belongs in its own reference artifact."
            )
            raise ValueError(msg)
        if self.dataset_kind is DatasetKind.REFERENCE and self.sources:
            msg = (
                "A reference dataset cites no sources: it describes a versioned vocabulary, not "
                "observations ingested from an external system."
            )
            raise ValueError(msg)

        for source in self.sources:
            expected = _REGIME_BY_NATURE.get(source.nature)
            if expected is not None and source.regime is not expected:
                msg = f"Source {source.source_id!r} must declare regime {expected.value!r}."
                raise ValueError(msg)

        artifact_names = [artifact.name for artifact in self.artifacts]
        if len(artifact_names) != len(set(artifact_names)):
            msg = "Duplicate artifact names in manifest."
            raise ValueError(msg)

        SchemaVersion.parse(self.schema_version)
        SchemaVersion.parse(self.manifest_version)
        return self

    def source(self, source_id: str) -> SourceRecord:
        """Return the source with *source_id*.

        Raises:
            KeyError: The source is not registered in this manifest.
        """
        for record in self.sources:
            if record.source_id == source_id:
                return record
        msg = f"Unknown source_id {source_id!r} for dataset {self.dataset_id!r}."
        raise KeyError(msg)

    def artifact(self, name: str) -> ArtifactRef:
        """Return the artifact with *name*.

        Raises:
            KeyError: The artifact is not present in this manifest.
        """
        for record in self.artifacts:
            if record.name == name:
                return record
        msg = f"Unknown artifact {name!r} for dataset {self.dataset_id!r}."
        raise KeyError(msg)

    def to_json_bytes(self) -> bytes:
        """Serialize deterministically: sorted keys, fixed indentation, UTF-8."""
        payload = self.model_dump(mode="json")
        text = json.dumps(
            payload,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            separators=(",", ": "),
        )
        return (text + "\n").encode("utf-8")

    def write_json(self, path: Path) -> bytes:
        """Write the manifest to *path* and return the bytes written."""
        data = self.to_json_bytes()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return data


def manifest_digest(manifest: DatasetManifest) -> str:
    """Return the SHA-256 of a manifest's canonical JSON bytes."""
    return hashlib.sha256(manifest.to_json_bytes()).hexdigest()
