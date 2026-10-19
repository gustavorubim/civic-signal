"""2026 House panel built from cited data (audit P0-5)."""

from __future__ import annotations

import csv
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "build_house_2026_panel", ROOT / "scripts" / "build_house_2026_panel.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _panel() -> list[dict[str, str]]:
    with (ROOT / "fixtures" / "house_district_panel.csv").open() as handle:
        return list(csv.DictReader(handle))


def test_choose_sides_cases() -> None:
    script = _load_script()
    sides = script.choose_sides
    assert sides({"DEM": ["Ann Lee"], "REP": ["Bob Roe"]}, ["Bob Roe"]) == (
        ("DEM", "Ann Lee"),
        ("REP", "Bob Roe"),
        "",
    )
    dem, rep, note = sides({"DEM": ["Ann Lee", "Cal Diaz"]}, ["Cal Diaz"])
    assert dem == ("DEM", "Cal Diaz") and rep == ("DEM", "Ann Lee") and "DEM v DEM" in note
    dem, rep, note = sides({"REP": ["Ken Calvert", "Young Kim"]}, ["Ken Calvert", "Young Kim"])
    assert dem[0] == rep[0] == "REP" and "REP v REP" in note
    assert sides({"DEM": ["A B"], "IND": ["C D"]}, [])[1] == ("IND", "C D")
    assert sides({"REP": ["A B"], "IND": ["C D"]}, [])[0] == ("IND", "C D")
    assert sides({"DEM": ["A B"]}, [])[1][0] == "OTH"
    assert sides({"REP": ["A B"]}, [])[0][0] == "OTH"
    _, _, note = sides({"DEM": ["A B"], "REP": ["C D", "E F"]}, [])
    assert "TODO(verify)" in note
    with pytest.raises(ValueError):
        sides({}, [])


def test_downballot_rows_and_national_share(tmp_path: Path) -> None:
    script = _load_script()
    path = tmp_path / "db.csv"
    path.write_text(
        "header,,,,,,,,,,,,\n"
        'AK-AL,Begich,R,x,"150,000","180,000","340,000",a,b,c,d,"153,000",x\n'
        'NC-01,Davis,D,x,"190,000","200,000","395,000",a,b,c,d,,\n'
    )
    rows = script.downballot_rows(path)
    assert set(rows) == {"AK-01", "NC-01"}
    assert rows["AK-01"]["map_changed"] is False
    assert rows["NC-01"]["map_changed"] is True
    president = [
        {
            "year": "2024",
            "party_simplified": "DEMOCRAT",
            "candidatevotes": "49",
            "writein": "False",
        },
        {"year": "2024", "party_simplified": "REPUBLICAN", "candidatevotes": "51"},
        {"year": "2020", "party_simplified": "DEMOCRAT", "candidatevotes": "999"},
    ]
    assert script.national_two_party(president) == pytest.approx(0.49)


def test_build_rows_and_qa_small_fixture() -> None:
    script = _load_script()
    seats = [
        {"office": "house", "jurisdiction": "XX-01", "incumbents": "Bob Roe",
         "incumbent_parties": "REP", "open_seat": "0", "cook_pvi": "-3.0"},
        {"office": "house", "jurisdiction": "XX-02", "incumbents": "",
         "incumbent_parties": "", "open_seat": "1", "cook_pvi": "5.0"},
    ]  # fmt: skip
    ballot = [
        {"office": "house", "jurisdiction": "XX-01", "party": "DEM", "name": "Ann Lee"},
        {"office": "house", "jurisdiction": "XX-01", "party": "REP", "name": "Bob Roe"},
        {"office": "house", "jurisdiction": "XX-02", "party": "DEM", "name": "Cy Fox"},
    ]
    downballot = {
        "XX-01": {"dem": 45.0, "rep": 55.0, "total": 100.0, "map_changed": False},
        "XX-02": {"dem": 60.0, "rep": 40.0, "total": 100.0, "map_changed": True},
    }
    rows = script.build_rows(seats, ballot, downballot, 0.5, [], 8.0)
    first, second = rows
    assert first["partisan_lean"] == "-5.0" and first["rep_incumbent"] == "True"
    assert first["national_swing"] == "4.0"  # (8.0 - 0.0) / 2 share points
    assert first["incumbent_party"] == "REP" and first["open_seat"] == "False"
    assert second["rep_party"] == "OTH" and second["dem_previous_vote_share"] == "1.0"
    assert second["rep_option_id"].endswith("-O") and second["map_changed_2026"] == "True"
    with pytest.raises(ValueError):
        script.qa(rows, seats)  # not 435 rows
    downballot["XX-01"] = {"dem": 90.0, "rep": 1.0, "total": 91.0, "map_changed": False}
    with pytest.raises(ValueError):
        script.build_rows(seats, ballot, downballot, 0.2, [], 8.0)  # |lean| > 60


def test_repo_house_panel_2026_rows() -> None:
    rows = _panel()
    current = [row for row in rows if row["cycle"] == "2026"]
    assert len(current) == 435
    assert len({row["race_id"] for row in current}) == 435
    assert all(row["synthetic"] == "False" for row in current)
    assert all(row["synthetic"] == "True" for row in rows if row["cycle"] != "2026")
    assert max(abs(float(row["partisan_lean"])) for row in current) <= 60
    ca40 = next(row for row in current if row["district"] == "CA-40")
    assert ca40["dem_party"] == ca40["rep_party"] == "REP"
    ca06 = next(row for row in current if row["district"] == "CA-06")
    assert (ca06["dem_party"], ca06["rep_party"]) == ("DEM", "IND")


def test_same_party_race_never_counts_for_dem() -> None:
    """A REP v REP seat (CA-40) gives the DEM seat count nothing in any draw."""
    import polars as pl

    from civic_signal.models import SimulationEngine

    rows = []
    for draw in range(4):
        rows += [
            {"draw_id": draw, "race_id": "CA-40", "party": "REP", "winner": True},
            {"draw_id": draw, "race_id": "CA-40", "party": "REP", "winner": False},
            {"draw_id": draw, "race_id": "X", "party": "DEM", "winner": draw < 2},
            {"draw_id": draw, "race_id": "X", "party": "REP", "winner": draw >= 2},
        ]
    draws = pl.DataFrame(rows).with_columns(pl.lit(0.5).alias("vote_share"))
    catalog = pl.DataFrame(
        {"race_id": ["CA-40", "X"], "office_type": ["house"] * 2,
         "control_body": ["house"] * 2, "seats": [1, 1]}
    )  # fmt: skip
    engine = SimulationEngine({"simulation_count": 4, "control_thresholds": {"house": 2}})
    control = engine._control_forecasts(SimpleNamespace(race_catalog=catalog), draws)
    assert control.filter(pl.col("party") == "DEM")["seat_count_mean"].item() == 0.5
    assert control.filter(pl.col("party") == "DEM")["control_probability"].item() == 0.0
    assert control.filter(pl.col("party") == "REP")["seat_count_mean"].item() == 1.5
