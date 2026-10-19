"""Unit tests for the VoteHub race-poll parser (audit tasks 1.1-1.4, 2.2, 2.5)."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from civic_signal.config import ProjectContext
from civic_signal.normalize import CuratedDataBuilder
from civic_signal.normalize.builder import US_STATE_ABBREVIATIONS, US_STATE_LOOKUP
from civic_signal.normalize.votehub import (
    jurisdiction_key,
    load_party_map,
    normalize_name,
    parse_votehub_polls,
    votehub_race_id,
)

ROOT = Path(__file__).resolve().parents[1]
STATES = set(US_STATE_ABBREVIATIONS.values())

MAP_ROWS = """office,jurisdiction,name,party,status,source
senate,Michigan,Abdul El-Sayed,DEM,nominee,test
senate,Michigan,Mike Rogers,REP,nominee,test
senate,Michigan,Haley Stevens,DEM,not_on_ballot,test
senate,Michigan,Lydia Christensen,LIB,minor,test
senate,Alabama,Everett Wess,DEM,nominee,test
senate,Alabama,Barry Moore,REP,nominee,test
senate,Alabama,Evil Twin,REP,nominee,test
senate,Nebraska,Dan Osborn,IND,nominee,test
senate,Nebraska,Pete Ricketts,REP,nominee,test
senate,Ohio,Sherrod Brown,DEM,nominee,test
senate,Ohio,Jon Husted,REP,nominee,test
senate,Georgia,Conflicted Person,DEM,nominee,test
senate,Georgia,Conflicted Person,REP,nominee,test
house,PA-07,Bob Brooks,DEM,nominee,test
house,PA-07,Ryan Mackenzie,REP,nominee,test
house,NY-17,Cait Conley,DEM,nominee,test
house,NY-17,Mike Lawler,REP,nominee,test
house,AK-01,Nick Begich III,REP,nominee,test
house,AK-01,Eric Hafner,DEM,nominee,test
house,CA-40,Ken Calvert,REP,nominee,test
house,CA-40,Young Kim,REP,nominee,test
"""


def _poll(
    poll_id: str,
    subject: str,
    answers: list[tuple[str, float]],
    *,
    poll_type: str = "us-senator",
    pollster: str = "Pollster A",
    start: str = "2026-10-01",
    end: str = "2026-10-03",
    population: str = "lv",
    seat_name: str | None = None,
    partisan: str | None = None,
    internal: bool = False,
) -> dict[str, object]:
    return {
        "id": poll_id,
        "poll_type": poll_type,
        "sample_size": 800,
        "population": population,
        "start_date": start,
        "end_date": end,
        "pollster": pollster,
        "answers": [{"choice": c, "pct": p} for c, p in answers],
        "seat_name": seat_name,
        "internal": internal,
        "partisan": partisan,
        "subject": subject,
    }


@pytest.fixture()
def party_map(tmp_path: Path) -> Path:
    path = tmp_path / "parties.csv"
    path.write_text(MAP_ROWS, encoding="utf-8")
    return path


def _parse(polls, party_map: Path, *, office="senate", version="votehub-race-polls-v2", **extra):
    args = {
        "cycle": 2026,
        "office": office,
        "poll_type": "us-senator" if office == "senate" else "us-representative",
        "parties": ["DEM", "REP"],
        "party_map_path": str(party_map),
        **extra,
    }
    return parse_votehub_polls(
        json.dumps(polls),
        args,
        root=ROOT,
        state_lookup=US_STATE_LOOKUP,
        state_abbreviations=STATES,
        parser_version=version,
    )


def test_same_party_answers_drop_the_poll(party_map: Path) -> None:
    polls = [
        _poll("p1", "2026 Alabama", [("Everett Wess", 40), ("Barry Moore", 45), ("Evil Twin", 5)])
    ]
    frame, acct = _parse(polls, party_map)
    assert frame.is_empty()
    assert acct.dropped_same_party == 1


def test_one_sided_poll_is_dropped_as_missing_side(party_map: Path) -> None:
    polls = [_poll("p1", "2026 Alabama", [("Barry Moore", 55), ("Somebody Else", 30)])]
    frame, acct = _parse(polls, party_map)
    assert frame.is_empty()
    assert acct.dropped_missing_side == 1
    assert acct.unmapped_names["AL: Somebody Else"] == 1


def test_corrected_labels_give_two_sided_polls() -> None:
    real_map = ROOT / "fixtures" / "candidates_2026_parties.csv"
    polls = [
        _poll("al", "2026 Alabama", [("Barry Moore", 50), ("Everett Wess", 38)]),
        _poll("nm", "2026 New Mexico", [("Ben Ray Luján", 52), ("Larry Marker", 38)]),
        _poll("pa", "2026 PA-07", [("Bob Brooks", 47), ("Ryan Mackenzie", 43)],
              poll_type="us-representative", seat_name="PA-07"),
        _poll("vt", "2026 VT-01", [("Becca Balint", 60), ("Gerald Malloy", 25)],
              poll_type="us-representative", seat_name="VT-01"),
    ]  # fmt: skip
    senate, _ = _parse(polls[:2], real_map, require_nominees=True)
    house, _ = _parse(polls[2:], real_map, office="house", require_nominees=True)
    frame = pl.concat([senate, house])
    pct = {row["option_id"]: row["pct"] for row in frame.iter_rows(named=True)}
    assert pct["US-SEN-AL-2026-D"] == 38 and pct["US-SEN-AL-2026-R"] == 50
    assert pct["US-SEN-NM-2026-D"] == 52 and pct["US-SEN-NM-2026-R"] == 38
    assert pct["US-HOUSE-PA-07-2026-D"] == 47 and pct["US-HOUSE-PA-07-2026-R"] == 43
    assert pct["US-HOUSE-VT-01-2026-D"] == 60 and pct["US-HOUSE-VT-01-2026-R"] == 25


def test_unmapped_minor_candidate_becomes_other(party_map: Path) -> None:
    polls = [
        _poll(
            "full",
            "2026 Michigan",
            [("Abdul El-Sayed", 46), ("Mike Rogers", 44), ("Lydia Christensen", 3), ("Nobody", 2)],
        )
    ]
    frame, acct = _parse(polls, party_map)
    assert frame.height == 2
    pct = dict(zip(frame["option_id"], frame["pct"], strict=True))
    assert pct == {"US-SEN-MI-2026-D": 46.0, "US-SEN-MI-2026-R": 44.0}
    assert acct.dropped_unmapped == 0
    assert acct.unmapped_names["MI: Nobody"] == 1


def test_nominee_filter_drops_hypothetical_matchups(party_map: Path) -> None:
    polls = [
        _poll("hyp", "2026 Michigan", [("Haley Stevens", 45), ("Mike Rogers", 44)]),
        _poll("real", "2026 Michigan", [("Abdul El-Sayed", 46), ("Mike Rogers", 44)],
              pollster="Pollster B"),
    ]  # fmt: skip
    frame, acct = _parse(polls, party_map, require_nominees=True)
    assert set(frame["poll_id"].str.slice(0, 13).to_list()) == {"votehub-real-"}
    assert acct.dropped_non_nominee == 1
    assert acct.non_nominee_names["MI: Haley Stevens"] == 1
    unfiltered, _ = _parse(polls, party_map, require_nominees=False)
    assert unfiltered.height == 4


def test_dedupe_prefers_lv_then_full_ballot_then_named(party_map: Path) -> None:
    lv = _poll("lv", "2026 Ohio", [("Sherrod Brown", 48), ("Jon Husted", 45)])
    rv = _poll("rv", "2026 Ohio", [("Sherrod Brown", 47), ("Jon Husted", 44)], population="rv")
    full = _poll(
        "full",
        "2026 Ohio",
        [("Sherrod Brown", 46), ("Jon Husted", 43), ("X Y", 3)],
        pollster="Pollster B",
    )
    h2h = _poll(
        "h2h", "2026 Ohio", [("Sherrod Brown", 49), ("Jon Husted", 46)], pollster="Pollster B"
    )
    named = _poll(
        "named",
        "2026 NY-17",
        [("Cait Conley", 48), ("Mike Lawler", 46)],
        poll_type="us-representative",
        seat_name="NY-17",
        pollster="C",
    )
    generic = _poll(
        "gen",
        "2026 NY-17",
        [("Dem", 49), ("Rep", 45)],
        poll_type="us-representative",
        seat_name="NY-17",
        pollster="C",
    )
    senate, acct = _parse([rv, lv, h2h, full], party_map)
    kept = {pid.split("-")[1] for pid in senate["poll_id"].to_list()}
    assert kept == {"lv", "full"}
    assert acct.dropped_duplicate == 2
    house, hacct = _parse([generic, named], party_map, office="house")
    assert {pid.split("-")[1] for pid in house["poll_id"].to_list()} == {"named"}
    assert hacct.dropped_duplicate == 1
    no_dedupe, _ = _parse([rv, lv], party_map, dedupe_versions=False)
    assert no_dedupe.height == 4


def test_independent_major_option_for_nebraska(party_map: Path) -> None:
    polls = [_poll("ne", "2026 Nebraska", [("Dan Osborn", 47), ("Pete Ricketts", 46)])]
    frame, _ = _parse(polls, party_map, race_parties={"US-SEN-NE-2026": ["IND", "REP"]})
    pct = dict(zip(frame["option_id"], frame["pct"], strict=True))
    assert pct == {"US-SEN-NE-2026-I": 47.0, "US-SEN-NE-2026-R": 46.0}
    without, acct = _parse(polls, party_map)
    assert without.is_empty() and acct.dropped_missing_side == 1


def test_same_party_race_is_skipped(party_map: Path) -> None:
    polls = [_poll("ca", "2026 CA-40", [("Ken Calvert", 40), ("Young Kim", 41)],
                   poll_type="us-representative", seat_name="CA-40")]  # fmt: skip
    frame, acct = _parse(polls, party_map, office="house", same_party_races=["US-HOUSE-CA-40-2026"])
    assert frame.is_empty() and acct.skipped_same_party_race == 1


def test_sponsor_party_and_classes(party_map: Path) -> None:
    polls = [
        _poll("d", "2026 Ohio", [("Sherrod Brown", 50), ("Jon Husted", 44)], partisan="DEM"),
        _poll("i", "2026 Ohio", [("Sherrod Brown", 48), ("Jon Husted", 46)], internal=True,
              pollster="B"),
        _poll("n", "2026 Ohio", [("Sherrod Brown", 47), ("Jon Husted", 46)], pollster="C",
              population="v"),
    ]  # fmt: skip
    frame, _ = _parse(polls, party_map)
    by_poll = {row["poll_id"].split("-")[1]: row for row in frame.iter_rows(named=True)}
    assert by_poll["d"]["sponsor_class"] == "partisan" and by_poll["d"]["sponsor_party"] == "DEM"
    assert by_poll["i"]["sponsor_class"] == "internal"
    assert by_poll["n"]["sponsor_class"] == "nonpartisan" and by_poll["n"]["population"] == "lv"


def test_v1_legacy_mode_drops_whole_poll_on_unmapped(party_map: Path) -> None:
    polls = [
        _poll("full", "2026 Michigan", [("Abdul El-Sayed", 46), ("Mike Rogers", 44), ("Nobody", 2)])
    ]
    frame, acct = _parse(polls, party_map, version="votehub-race-polls-v1")
    assert frame.is_empty() and acct.dropped_unmapped == 1


def test_skips_and_malformed_inputs(party_map: Path) -> None:
    polls = [
        "not-a-dict",
        _poll("gov", "2026 Ohio", [("A", 1)], poll_type="us-governor"),
        _poll("prim", "2026 Michigan Democratic", [("Abdul El-Sayed", 40)]),
        _poll("nowhere", "2026 Atlantis", [("A", 1)]),
        {**_poll("nosubj", "", [("A", 1)])},
        {**_poll("noans", "2026 Ohio", []), "answers": []},
        {**_poll("badans", "2026 Ohio", []), "answers": ["x"]},
        _poll("nopct", "2026 Ohio", [("Sherrod Brown", None), ("Jon Husted", 40)]),
        _poll("conflict", "2026 Georgia", [("Conflicted Person", 40), ("Mike Rogers", 40)]),
    ]
    frame, acct = _parse(polls, party_map)
    assert frame.is_empty()
    assert acct.skipped_other_poll_type == 1
    assert acct.skipped_primary == 1
    assert acct.skipped_unknown_race == 1
    assert acct.dropped_malformed >= 4
    payload = acct.to_dict()
    assert payload["polls_used"] == 0 and isinstance(payload["unmapped_names"], dict)


def test_argument_and_payload_validation(party_map: Path) -> None:
    with pytest.raises(ValueError, match="missing required keys"):
        parse_votehub_polls("[]", {}, root=ROOT, state_lookup={}, state_abbreviations=set())
    with pytest.raises(ValueError, match="office must be"):
        _parse([], party_map, office="governor")
    with pytest.raises(ValueError, match="parties must be non-empty"):
        _parse([], party_map, parties=[])
    with pytest.raises(ValueError, match="not valid JSON"):
        parse_votehub_polls(
            "{oops",
            {"cycle": 2026, "office": "senate", "poll_type": "x", "parties": ["DEM"],
             "party_map_path": str(party_map)},
            root=ROOT, state_lookup={}, state_abbreviations=set(),
        )  # fmt: skip
    with pytest.raises(ValueError, match="JSON array"):
        parse_votehub_polls(
            "{}",
            {"cycle": 2026, "office": "senate", "poll_type": "x", "parties": ["DEM"],
             "party_map_path": str(party_map)},
            root=ROOT, state_lookup={}, state_abbreviations=set(),
        )  # fmt: skip


def test_party_map_validation(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_party_map(tmp_path / "missing.csv", "senate", US_STATE_LOOKUP)
    bad = tmp_path / "bad.csv"
    bad.write_text("who,what\nA,B\n", encoding="utf-8")
    with pytest.raises(ValueError, match="name,party"):
        load_party_map(bad, "senate", US_STATE_LOOKUP)
    loose = tmp_path / "loose.csv"
    loose.write_text("name,party\nGlobal Person,DEM\n,REP\nNo Party,\n", encoding="utf-8")
    pmap = load_party_map(loose, "house", US_STATE_LOOKUP)
    assert pmap.lookup("TX-01", "Global Person").party == "DEM"
    assert pmap.lookup("TX-01", "rep").status == "generic"
    assert pmap.lookup("TX-01", "Unknown") is None


def test_house_race_ids_and_helpers() -> None:
    kwargs = {"office": "house", "cycle": 2026, "state_lookup": US_STATE_LOOKUP,
              "state_abbreviations": STATES}  # fmt: skip
    assert (
        votehub_race_id(subject="2026 AK-01", seat_name=None, house_format="hyphenated", **kwargs)
        == "US-HOUSE-AK-01-2026"
    )
    assert (
        votehub_race_id(subject="x", seat_name="CA-45", house_format="compact", **kwargs)
        == "US-HOUSE-CA45-2026"
    )
    assert (
        votehub_race_id(
            subject="District ZZ-01", seat_name=None, house_format="hyphenated", **kwargs
        )
        is None
    )
    assert (
        votehub_race_id(subject="no seat", seat_name=None, house_format="hyphenated", **kwargs)
        is None
    )
    with pytest.raises(ValueError, match="house_race_id_format"):
        votehub_race_id(subject="x", seat_name="CA-45", house_format="weird", **kwargs)
    assert jurisdiction_key("house", "AK-AL", US_STATE_LOOKUP) == "AK-01"
    assert jurisdiction_key("house", "ny 3", US_STATE_LOOKUP) == "NY-03"
    assert jurisdiction_key("house", "odd", US_STATE_LOOKUP) == "ODD"
    assert jurisdiction_key("senate", "Michigan", US_STATE_LOOKUP) == "MI"
    assert jurisdiction_key("senate", "", US_STATE_LOOKUP) == ""
    assert normalize_name("  Nick   Begich III ") == "nick begich"


def test_house_at_large_and_accent_fallback(party_map: Path) -> None:
    polls = [_poll("ak", "2026 AK-01", [("Nick Begich III", 48), ("Eric Hafner", 40)],
                   poll_type="us-representative")]  # fmt: skip
    frame, _ = _parse(polls, party_map, office="house")
    assert set(frame["race_id"].to_list()) == {"US-HOUSE-AK-01-2026"}
    accented = ROOT / "fixtures" / "candidates_2026_parties.csv"
    nm = [_poll("nm", "2026 New Mexico", [("Ben Ray Lujan", 52), ("Larry Marker", 38)])]
    frame, _ = _parse(nm, accented)
    assert frame.height == 2


def test_builder_writes_votehub_accounting(tmp_path: Path, party_map: Path) -> None:
    ctx = ProjectContext.create(root=ROOT, data_dir=tmp_path / "data", artifacts_dir=tmp_path / "a")
    builder = CuratedDataBuilder(ctx)
    raw = json.dumps([_poll("x", "2026 Ohio", [("Sherrod Brown", 48), ("Jon Husted", 45)])])
    args = {"cycle": 2026, "office": "senate", "poll_type": "us-senator",
            "parties": ["DEM", "REP"], "party_map_path": str(party_map)}  # fmt: skip
    frame = builder._normalize_votehub_race_polls(raw, args, source_id="vh")
    assert frame.schema["end_date"] == pl.Date
    assert builder.parser_accounting["vh"]["polls_used"] == 1
    empty = builder._normalize_votehub_race_polls("[]", args, source_id="vh2")
    assert empty.is_empty() and empty.schema["end_date"] == pl.Date
