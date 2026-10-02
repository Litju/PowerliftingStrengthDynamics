"""Regression tests for the OpenPowerlifting source contract.

RES-235 established a provisional fixture source record that described
OpenPowerlifting as ``CC-BY-SA-4.0`` with a ``public_license`` consent basis. Both
were wrong, and RES-237 corrected them before any real ingestion happened:

* OpenPowerlifting contributes all competition data to the **Public Domain** and
  waives copyright and related rights. ``CC-BY-SA-4.0`` is a copyleft share-alike
  grant the project does not make, and ``CC0-1.0`` would be an equally invented
  formal license grant.
* A publicly archived competition result is a **public record**, not a record of
  athlete consent. Nobody asked these lifters to consent to an archive, and a
  consent basis is not the place to record that a competition took place.

These tests exist so the incorrect metadata cannot come back. They read the
*production* source record rather than a copy of it, so a future edit that reverts
the license or the basis fails here even if every other test still passes.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from psd.ingest.openpowerlifting import (
    OPENPOWERLIFTING_BULK_DOCS_URL,
    OPENPOWERLIFTING_DATA_SERVICE_URL,
    OPENPOWERLIFTING_LICENSE_ID,
    OPENPOWERLIFTING_LICENSE_STATEMENT,
    OPENPOWERLIFTING_ORIGIN_SYSTEM,
    OpenPowerliftingSnapshot,
    ServiceSnapshotFacts,
    openpowerlifting_source_id,
    openpowerlifting_source_record,
)
from psd.provenance.sources import (
    ConsentBasis,
    DataRegime,
    PublicationBasis,
    RedistributionPolicy,
    SourceNature,
    SourceRecord,
)
from psd.schema.vocabulary import VOCABULARIES
from tests.fixtures.builders import HistoryBuilder, SourceIds, add_source_records
from tests.fixtures.normal_history import normal_history_records


def normal_history_records_source() -> SourceRecord:
    """Return the OpenPowerlifting source row the shared fixtures still carry."""
    wanted = SourceIds().competition
    for record in normal_history_records()["source"]:
        if isinstance(record, SourceRecord) and record.source_id == wanted:
            return record
    pytest.fail(f"shared fixtures no longer carry the {wanted!r} source row")


#: Licenses OpenPowerlifting never claimed for its competition data. ``CC0-1.0`` is
#: listed because it is the tempting substitute once ``CC-BY-SA-4.0`` is rejected: it
#: is a formal public-domain-dedication license *grant*, and OpenPowerlifting does not
#: grant one. Recording it would still invent a license.
FORBIDDEN_LICENSE_IDS = frozenset(
    {
        "cc-by-sa-4.0",
        "cc-by-sa-3.0",
        "cc-by-4.0",
        "cc0-1.0",
        "unlicense",
        "agpl-3.0",
        "agpl-3.0-or-later",
        "gpl-3.0",
        "gpl-3.0-or-later",
        "wtfpl",
        "mit",
        "bsd-3-clause",
        "cc-by-nc-4.0",
    }
)

INGESTED_AT = datetime(2026, 10, 2, 8, 0, tzinfo=UTC)

#: Digest-shaped placeholders. No test here depends on real downloaded bytes.
FAKE_ARCHIVE_DIGEST = "a" * 64
FAKE_CSV_DIGEST = "b" * 64


def _snapshot(**overrides: object) -> OpenPowerliftingSnapshot:
    payload: dict[str, object] = {
        "source_url": (
            "https://openpowerlifting.gitlab.io/opl-csv/files/openpowerlifting-latest.zip"
        ),
        "downloaded_at": INGESTED_AT,
        "service": ServiceSnapshotFacts(updated_date="2026-09-25", revision="f231b4f6"),
        "archive_sha256": FAKE_ARCHIVE_DIGEST,
        "archive_byte_size": 169848348,
        "csv_member_name": "openpowerlifting-latest.csv",
        "csv_sha256": FAKE_CSV_DIGEST,
        "csv_byte_size": 640000000,
        "row_count": 4036910,
        "source_columns": ("Name", "Sex"),
        "code_commit": "c" * 40,
    }
    payload.update(overrides)
    return OpenPowerliftingSnapshot(**payload)  # type: ignore[arg-type]


def test_source_record_describes_a_public_domain_release() -> None:
    record = openpowerlifting_source_record(_snapshot(), ingested_at=INGESTED_AT)
    assert record.nature is SourceNature.REAL
    assert record.regime is DataRegime.COMP
    assert record.origin_system == OPENPOWERLIFTING_ORIGIN_SYSTEM
    assert record.redistribution is RedistributionPolicy.ALLOWED


def test_source_record_never_claims_a_license_openpowerlifting_did_not_grant() -> None:
    record = openpowerlifting_source_record(_snapshot(), ingested_at=INGESTED_AT)
    assert record.license_id == OPENPOWERLIFTING_LICENSE_ID
    assert record.license_id is not None
    assert record.license_id.lower() not in FORBIDDEN_LICENSE_IDS


def test_source_record_points_at_the_official_data_service() -> None:
    record = openpowerlifting_source_record(_snapshot(), ingested_at=INGESTED_AT)
    assert record.license_url == OPENPOWERLIFTING_DATA_SERVICE_URL
    assert record.license_url is not None
    assert record.license_url.startswith("https://openpowerlifting.gitlab.io/opl-csv/")


def test_source_record_uses_a_public_record_basis_not_athlete_consent() -> None:
    """The core correction: legally redistributable is not the same as consented."""
    record = openpowerlifting_source_record(_snapshot(), ingested_at=INGESTED_AT)
    assert record.consent_basis is ConsentBasis.PUBLIC_RECORD
    assert record.consent_basis is not ConsentBasis.USER_CONSENT
    assert record.consent_basis is not ConsentBasis.PUBLIC_LICENSE


def test_source_record_declares_its_publication_basis_separately() -> None:
    record = openpowerlifting_source_record(_snapshot(), ingested_at=INGESTED_AT)
    assert record.publication_basis is PublicationBasis.PUBLISHED_COMPETITION_RESULTS


def test_source_record_quotes_the_dedication() -> None:
    record = openpowerlifting_source_record(_snapshot(), ingested_at=INGESTED_AT)
    assert record.notes is not None
    assert "Public Domain" in record.notes
    assert "not of the OpenPowerlifting project's own source code" in record.notes
    assert "not athlete-consented" in record.notes


def test_license_statement_is_the_services_own_wording() -> None:
    lowered = OPENPOWERLIFTING_LICENSE_STATEMENT.lower()
    assert "public domain" in lowered
    assert "waived of all copyright" in lowered
    assert "attribution is requested but not required" in lowered


def test_snapshot_identity_is_the_digest_not_the_mutable_filename() -> None:
    """Two downloads of the same mutable URL are one snapshot iff their bytes match."""
    source_id = openpowerlifting_source_id(FAKE_ARCHIVE_DIGEST)
    assert source_id == openpowerlifting_source_id(FAKE_ARCHIVE_DIGEST)
    assert source_id != openpowerlifting_source_id("c" * 64)
    assert "latest" not in source_id
    assert source_id.startswith(OPENPOWERLIFTING_ORIGIN_SYSTEM)


def test_source_id_rejects_a_non_digest() -> None:
    with pytest.raises(ValueError, match="lowercase hex SHA-256"):
        openpowerlifting_source_id("NOT-A-DIGEST")


def test_source_id_matches_the_source_record_identifier_pattern() -> None:
    record = openpowerlifting_source_record(_snapshot(), ingested_at=INGESTED_AT)
    assert re.fullmatch(r"[a-z0-9][a-z0-9_.-]{0,127}", record.source_id)
    assert record.source_id == openpowerlifting_source_id(FAKE_ARCHIVE_DIGEST)


def test_source_record_carries_the_service_reported_snapshot_facts() -> None:
    record = openpowerlifting_source_record(_snapshot(), ingested_at=INGESTED_AT)
    assert record.snapshot_sha256 == FAKE_ARCHIVE_DIGEST
    assert record.snapshot_date is not None
    assert record.snapshot_date.date().isoformat() == "2026-09-25"
    dataset_version = record.dataset_version
    assert dataset_version is not None
    assert "2026-09-25" in dataset_version
    assert "f231b4f6" in dataset_version


def test_dataset_version_falls_back_to_the_digest_when_the_service_says_nothing() -> None:
    snapshot = _snapshot(service=ServiceSnapshotFacts())
    expected = f"opl-sha256-{FAKE_ARCHIVE_DIGEST[:16]}"
    assert snapshot.dataset_version() == expected
    assert openpowerlifting_source_record(snapshot, ingested_at=INGESTED_AT).dataset_version == (
        expected
    )


def test_snapshot_rejects_a_missing_header() -> None:
    """A pinned snapshot with no recorded header cannot be shown to match a schema."""
    with pytest.raises(ValidationError, match="source_columns is required"):
        _snapshot(source_columns=())


def test_snapshot_rejects_a_duplicated_header_column() -> None:
    with pytest.raises(ValidationError, match="free of duplicates"):
        _snapshot(source_columns=("Name", "Name"))


def test_snapshot_requires_an_aware_download_timestamp() -> None:
    with pytest.raises(ValidationError, match="naive timestamp"):
        _snapshot(downloaded_at=datetime(2026, 10, 2, 8, 0))


def test_snapshot_rejects_an_uppercase_digest() -> None:
    with pytest.raises(ValidationError):
        _snapshot(archive_sha256="A" * 64)


def test_a_reference_source_cannot_claim_a_public_record_publication_basis() -> None:
    """The publication axis is enforced, not merely documented."""
    with pytest.raises(ValidationError, match="reference sources must declare"):
        SourceRecord(
            source_id="psd_ontology_probe",
            display_name="PSD exercise ontology",
            nature=SourceNature.REFERENCE,
            regime=DataRegime.REFERENCE,
            origin_system="psd-ontology",
            license_id="apache-2.0",
            publication_basis=PublicationBasis.PUBLISHED_COMPETITION_RESULTS,
            consent_basis=ConsentBasis.NONE_DECLARED,
            redistribution=RedistributionPolicy.ALLOWED,
            ingested_at=INGESTED_AT,
        )


def test_a_real_source_cannot_claim_simulator_generated_publication() -> None:
    with pytest.raises(ValidationError, match="that basis describes how only synthetic"):
        SourceRecord(
            source_id="openpowerlifting_probe",
            display_name="Probe",
            nature=SourceNature.REAL,
            regime=DataRegime.COMP,
            origin_system="openpowerlifting",
            license_id=OPENPOWERLIFTING_LICENSE_ID,
            publication_basis=PublicationBasis.SIMULATOR_GENERATED,
            consent_basis=ConsentBasis.PUBLIC_RECORD,
            redistribution=RedistributionPolicy.ALLOWED,
            ingested_at=INGESTED_AT,
        )


def test_public_record_and_public_license_are_distinct_members() -> None:
    """A public record and an open license answer different questions."""
    assert ConsentBasis.PUBLIC_RECORD.value == "public_record"
    assert ConsentBasis.PUBLIC_LICENSE.value == "public_license"
    assert PublicationBasis.PUBLISHED_COMPETITION_RESULTS.value == "published_competition_results"
    assert "public_record" in VOCABULARIES["consent_basis"]
    assert "published_competition_results" in VOCABULARIES["publication_basis"]


def test_shared_fixture_source_record_no_longer_claims_cc_by_sa() -> None:
    """The RES-235 fixture must not drift back to the incorrect license."""
    record = normal_history_records_source()
    assert record.license_id == OPENPOWERLIFTING_LICENSE_ID
    assert record.license_url == OPENPOWERLIFTING_DATA_SERVICE_URL
    assert record.consent_basis is ConsentBasis.PUBLIC_RECORD
    assert record.publication_basis is PublicationBasis.PUBLISHED_COMPETITION_RESULTS


def test_shared_fixture_source_record_names_the_data_service() -> None:
    record = normal_history_records_source()
    assert record.notes is not None
    assert "Public Domain" in record.notes
    assert OPENPOWERLIFTING_BULK_DOCS_URL.startswith("https://openpowerlifting.gitlab.io/")


def test_fixture_keeps_public_record_and_public_license_apart() -> None:
    """A community archive of results is a public record; PSD-Sim's output is a license."""
    builder = HistoryBuilder(athlete_id="ath_probe", source_id=SourceIds().training)
    records = add_source_records(builder)
    assert records["competition"].consent_basis is ConsentBasis.PUBLIC_RECORD
    assert records["competition"].publication_basis is (
        PublicationBasis.PUBLISHED_COMPETITION_RESULTS
    )
    assert records["synthetic"].consent_basis is ConsentBasis.PUBLIC_LICENSE
    assert records["synthetic"].publication_basis is PublicationBasis.SIMULATOR_GENERATED


def test_every_source_fixture_row_declares_a_publication_basis() -> None:
    """No source row may leave the publication axis unstated."""
    for record in normal_history_records()["source"]:
        assert isinstance(record, SourceRecord)
        assert record.publication_basis is not PublicationBasis.UNKNOWN
