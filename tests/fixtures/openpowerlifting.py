"""Deterministic OpenPowerlifting source fixtures.

Every edge case RES-237 names has a row here, so the tests exercise the semantics rather
than a happy path:

* successful and negative (failed) attempts;
* a lifter with no attempt data at all, reporting only a best lift;
* a fourth attempt, which is a record attempt and counts toward no total;
* a negative reported best, which means the lowest weight attempted and failed;
* a total reported with no component lifts at all;
* ``DQ``, ``DD``, ``G`` and ``NS`` participation codes;
* an approximate ``23.5`` age beside an exact ``23``;
* an open-ended ``90+`` weight class beside a bounded ``-93``;
* every declared event: ``SBD``, ``BD``, ``SD``, ``SB``, ``S``, ``B`` and ``D``;
* ``Tested=Yes`` beside an untested result;
* ``ParentFederation`` distinct from ``Federation``, and one absent;
* two lifters sharing a name, disambiguated by the source's ``#N`` suffix;
* one meet whose identity needs the whole six-field rule, and a second entry at that meet
  whose ``MeetName`` the source spells with different case and spacing;
* a name the source reports under two sex categories;
* an unsanctioned meet;
* a source value PSD cannot read.

The fixture is written from a mapping, so a row always has exactly the contract's column
count. That is deliberate: a hand-counted CSV row that drifts out of alignment fails with
"too many fields" and tells you nothing about the semantics under test.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from psd.ingest.openpowerlifting.acquire import acquire_snapshot_from_local_file
from psd.ingest.openpowerlifting.contract import expected_columns

if TYPE_CHECKING:
    from psd.ingest.openpowerlifting.snapshot import OpenPowerliftingSnapshot

__all__ = (
    "SAMPLE_CSV_NAME",
    "SAMPLE_ROW_COUNT",
    "edge_case_rows",
    "sample_csv_text",
    "write_sample_csv",
    "write_sample_snapshot",
)

SAMPLE_CSV_NAME = "openpowerlifting-sample.csv"

BASE_MEET: dict[str, str] = {
    "Date": "2025-11-08",
    "Federation": "USPA",
    "MeetCountry": "USA",
    "MeetState": "TX",
    "MeetTown": "Dallas",
    "MeetName": "Raw National Open",
    "Sanctioned": "Yes",
}


def edge_case_rows() -> tuple[dict[str, Any], ...]:
    """Return the fixture rows, one mapping per source row."""
    return (
        {
            # A complete three-attempt SBD result with a record fourth attempt, an
            # approximate age, and a drug-tested category.
            "Name": "Ada Liftwell#1",
            "Sex": "M",
            "Event": "SBD",
            "Equipment": "Raw",
            "Age": "23.5",
            "AgeClass": "23-39",
            "BirthYearClass": "",
            "Division": "Open",
            "BodyweightKg": "91.4",
            "WeightClassKg": "-93",
            "Squat1Kg": "180",
            "Squat2Kg": "185",
            "Squat3Kg": "187.5",
            "Squat4Kg": "",
            "Best3SquatKg": "187.5",
            "Bench1Kg": "110",
            "Bench2Kg": "115",
            "Bench3Kg": "120",
            "Bench4Kg": "127.5",
            "Best3BenchKg": "122.5",
            "Deadlift1Kg": "210",
            "Deadlift2Kg": "220",
            "Deadlift3Kg": "230",
            "Deadlift4Kg": "",
            "Best3DeadliftKg": "230",
            "TotalKg": "540",
            "Place": "1",
            "Dots": "500.13",
            "Wilks": "499.87",
            "Glossbrenner": "498.75",
            "Goodlift": "37.11",
            "Tested": "Yes",
            "Country": "USA",
            "State": "TX",
            "ParentFederation": "IPF",
        },
        {
            # The same base name without the disambiguator: a different lifter, which
            # the source asserts by not suffixing them the same way.
            "Name": "Ada Liftwell#2",
            "Sex": "M",
            "Event": "BD",
            "Equipment": "Wraps",
            "Age": "23",
            "AgeClass": "23-39",
            "Division": "Open",
            "BodyweightKg": "88.2",
            "WeightClassKg": "-93",
            "Squat1Kg": "",
            "Bench1Kg": "-100",
            "Bench2Kg": "-105",
            "Bench3Kg": "110",
            "Best3BenchKg": "110",
            "Deadlift1Kg": "215",
            "Deadlift2Kg": "225",
            "Deadlift3Kg": "",
            "Best3DeadliftKg": "225",
            "TotalKg": "335",
            "Place": "2",
            "Tested": "",
            "Country": "GBR",
            "State": "London",
            "ParentFederation": "",
        },
        {
            # No attempt data at all, only reported bests. PSD must not reconstruct
            # attempts from these.
            "Name": "Bo Reported",
            "Sex": "M",
            "Event": "S",
            "Equipment": "Single-ply",
            "Age": "40",
            "AgeClass": "40-49",
            "BirthYearClass": "40-49",
            "Division": "Masters",
            "BodyweightKg": "93.5",
            "WeightClassKg": "105",
            "Squat1Kg": "",
            "Squat2Kg": "",
            "Squat3Kg": "",
            "Squat4Kg": "",
            "Best3SquatKg": "160",
            "TotalKg": "",
            "Place": "1",
            "Dots": "300.1",
            "Country": "USA",
            "State": "CA",
            "ParentFederation": "WPA",
            "MeetState": "CA",
            "MeetTown": "Fresno",
            "MeetName": "Masters Open",
            "Federation": "XPC",
        },
        {
            # A negative reported best: the source is publishing the lowest weight the
            # lifter attempted and failed, not a negative lift.
            "Name": "Cyd Failed",
            "Sex": "F",
            "Event": "B",
            "Equipment": "Wraps",
            "BodyweightKg": "62.7",
            "WeightClassKg": "-63",
            "Bench1Kg": "-45",
            "Bench2Kg": "-50",
            "Bench3Kg": "-55",
            "Best3BenchKg": "-45",
            "TotalKg": "",
            "Place": "DQ",
            "Country": "USA",
            "State": "NY",
            "ParentFederation": "IPF",
            "MeetState": "NY",
            "MeetTown": "Albany",
            "MeetName": "Open Classic",
            "Federation": "CPU",
        },
        {
            # A total with no component lifts at all: the source published the sum and
            # nothing else, so PSD records it and manufactures nothing. No bodyweight
            # either, which must stay missing rather than become zero.
            "Name": "Dee Totalless",
            "Sex": "M",
            "Event": "SBD",
            "Equipment": "Multi-ply",
            "Age": "29",
            "Division": "Open",
            "WeightClassKg": "-83",
            "Best3SquatKg": "",
            "Best3BenchKg": "",
            "Best3DeadliftKg": "",
            "TotalKg": "500",
            "Place": "4",
            "Country": "USA",
            "State": "TX",
            "ParentFederation": "IPF",
        },
        {
            # Guest lifter and no-show in the same meet, neither of which is a placing.
            "Name": "Eve Guestly",
            "Sex": "F",
            "Event": "BD",
            "Equipment": "Straps",
            "BodyweightKg": "61",
            "WeightClassKg": "-63",
            "Bench1Kg": "40",
            "Best3BenchKg": "40",
            "Deadlift1Kg": "90",
            "Best3DeadliftKg": "90",
            "Place": "G",
            "Country": "CAN",
            "State": "ON",
        },
        {
            "Name": "Finn Nowhere",
            "Sex": "M",
            "Event": "SBD",
            "Equipment": "Unlimited",
            "Age": "35",
            "Division": "Open",
            "BodyweightKg": "100.5",
            "WeightClassKg": "90+",
            "Place": "NS",
            "Country": "USA",
            "State": "TX",
            "ParentFederation": "IPF",
        },
        {
            # Doping disqualification, a self-described sex category, and a meet the
            # source does not count as sanctioned.
            "Name": "Gale Outly",
            "Sex": "Mx",
            "Event": "SD",
            "Equipment": "Unlimited",
            "Age": "31",
            "Division": "Open",
            "BodyweightKg": "77.5",
            "WeightClassKg": "-83",
            "Squat1Kg": "150",
            "Best3SquatKg": "150",
            "Deadlift1Kg": "200",
            "Deadlift3Kg": "-210",
            "Best3DeadliftKg": "200",
            "Place": "DD",
            "Country": "GBR",
            "State": "London",
            "MeetCountry": "GBR",
            "MeetState": "London",
            "MeetTown": "ExCeL",
            "MeetName": "National Open",
            "Federation": "BPA",
            "ParentFederation": "IPF",
            "Sanctioned": "No",
        },
        {
            # A value PSD cannot read, and a name the source also reports as male.
            "Name": "Hal Twofold",
            "Sex": "F",
            "Event": "B",
            "Equipment": "Raw",
            "Age": "not-a-number",
            "BodyweightKg": "70",
            "WeightClassKg": "-74",
            "Bench1Kg": "55",
            "Best3BenchKg": "55",
            "Place": "6",
            "Country": "USA",
            "State": "TX",
            "ParentFederation": "IPF",
        },
        {
            "Name": "Hal Twofold",
            "Sex": "M",
            "Event": "B",
            "Equipment": "Raw",
            "Age": "44",
            "BodyweightKg": "80",
            "WeightClassKg": "-83",
            "Bench1Kg": "95",
            "Best3BenchKg": "95",
            "Place": "3",
            "Country": "USA",
            "State": "TX",
            "ParentFederation": "IPF",
        },
        {
            # The same meet as the first row, spelled differently: different case and
            # internal spacing in ``MeetName``, everything else identical. The six-field
            # meet rule normalizes both, so this is one meet -- a second meet row would
            # break every foreign key pointing at it, and merging two genuinely different
            # meets would invent a meet that never happened.
            "Name": "Ivy Variant",
            "Sex": "F",
            "Event": "S",
            "Equipment": "Raw",
            "Age": "27",
            "Division": "Open",
            "BodyweightKg": "63.5",
            "WeightClassKg": "-69",
            "Squat1Kg": "120",
            "Squat2Kg": "125",
            "Squat3Kg": "-130",
            "Best3SquatKg": "125",
            "Place": "2",
            "Dots": "350.4",
            "Country": "USA",
            "State": "TX",
            # No ParentFederation, like every other row at this meet: it carries no
            # sanctioning body, and this row must not give it one.
            "MeetName": "  raw   NATIONAL   open ",
        },
        {
            # ``SB`` is a declared event of its own: a squat and a bench, no deadlift. The
            # absent deadlift is not an attempted-and-failed deadlift, so no deadlift row
            # may exist anywhere for this participation.
            "Name": "Jo Subtotal",
            "Sex": "M",
            "Event": "SB",
            "Equipment": "Wraps",
            "Age": "33",
            "Division": "Open",
            "BodyweightKg": "88",
            "WeightClassKg": "-93",
            "Squat1Kg": "170",
            "Squat3Kg": "180",
            "Best3SquatKg": "180",
            "Bench1Kg": "-120",
            "Bench3Kg": "130",
            "Best3BenchKg": "130",
            "Place": "1",
            "Country": "USA",
            "State": "TX",
            "ParentFederation": "IPF",
            "MeetCountry": "USA",
            "MeetState": "TX",
            "MeetTown": "Austin",
            "MeetName": "Open Subtotal Classic",
            "Federation": "USPA",
        },
        {
            # ``D`` alone, with a deadlift record attempt: the one place a fourth attempt is
            # legitimate, and it still contributes to no total.
            "Name": "Kip Single",
            "Sex": "M",
            "Event": "D",
            "Equipment": "Unlimited",
            "Age": "26",
            "Division": "Open",
            "BodyweightKg": "110.5",
            "WeightClassKg": "90+",
            "Deadlift1Kg": "240",
            "Deadlift2Kg": "255",
            "Deadlift3Kg": "265",
            "Deadlift4Kg": "272.5",
            "Best3DeadliftKg": "265",
            "Place": "1",
            "Country": "USA",
            "State": "NV",
            "ParentFederation": "IPF",
            "MeetCountry": "USA",
            "MeetState": "NV",
            "MeetTown": "Reno",
            "MeetName": "Deadlift Open",
            "Federation": "USPA",
        },
    )


def sample_csv_text() -> str:
    """Return the fixture CSV's full text, header included."""
    columns = expected_columns()
    lines = [",".join(columns)]
    for row in edge_case_rows():
        values = dict(BASE_MEET)
        values.update(row)
        lines.append(",".join(str(values.get(column, "")) for column in columns))
    return "\n".join(lines) + "\n"


SAMPLE_ROW_COUNT = len(edge_case_rows())


def write_sample_csv(path: Path) -> Path:
    """Write the fixture CSV to *path* and return it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(sample_csv_text(), encoding="utf-8")
    return path


def write_sample_snapshot(path: Path, *, data_root: Path) -> OpenPowerliftingSnapshot:
    """Write the fixture CSV and pin it as a local snapshot.

    Returns:
        The pinned :class:`~psd.ingest.openpowerlifting.snapshot.OpenPowerliftingSnapshot`.
    """
    csv_path = write_sample_csv(path)
    return acquire_snapshot_from_local_file(csv_path, data_root=data_root)
