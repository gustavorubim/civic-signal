"""Senate 2026 structure: FL/OH specials, holdovers, independents (audit P0-4, P0-5)."""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl
import yaml

from civic_signal.models import SimulationEngine
from civic_signal.models.fundamentals import FundamentalsModel

ROOT = Path(__file__).resolve().parents[1]


def _panel_2026() -> list[dict[str, str]]:
    with (ROOT / "fixtures" / "senate_state_panel.csv").open() as handle:
        return [row for row in csv.DictReader(handle) if row["cycle"] == "2026"]


def _scenario() -> dict:
    payload = yaml.safe_load((ROOT / "configs" / "scenarios.yaml").read_text())
    return payload["scenarios"]["senate_2026_state"]


def test_total_senate_seats_is_100() -> None:
    rows = _panel_2026()
    holdovers = sum(int(value) for value in _scenario()["holdovers"].values())
    assert len(rows) == 35
    assert holdovers + len(rows) == 100
    assert _scenario()["holdovers"]["REP"] == 31
    states = {row["state"] for row in rows}
    assert {"FL", "OH"} <= states
    specials = {row["state"] for row in rows if row["special"] == "True"}
    assert specials == {"FL", "OH"}


def test_real_incumbency_and_open_seats() -> None:
    rows = {row["state"]: row for row in _panel_2026()}
    assert rows["WV"]["incumbent_party"] == "REP"
    assert rows["WV"]["rep_incumbent"] == "True"
    for state in ["MI", "NH", "MN", "IL", "NC", "KY", "AL", "IA"]:
        assert rows[state]["incumbent_party"] == "none", state
        assert rows[state]["dem_incumbent"] == rows[state]["rep_incumbent"] == "False"
    assert rows["AK"]["rep_incumbent"] == "True"  # Dan S. Sullivan == "Dan Sullivan"
    for row in rows.values():
        assert row["synthetic"] == "False"
        assert row["pres_2024_margin"] != ""
        assert abs(float(row["partisan_lean"])) <= 30


def test_history_rows_flagged_synthetic() -> None:
    with (ROOT / "fixtures" / "senate_state_panel.csv").open() as handle:
        history = [row for row in csv.DictReader(handle) if row["cycle"] != "2026"]
    assert history and all(row["synthetic"] == "True" for row in history)


def test_independent_options_replace_missing_democrats() -> None:
    rows = {row["state"]: row for row in _panel_2026()}
    assert {s for s, r in rows.items() if r["dem_party"] == "IND"} == {"NE", "SD", "ID"}
    assert rows["NE"]["dem_name"] == "Dan Osborn"
    assert rows["NE"]["dem_option_id"] == "US-SEN-NE-2026-I"
    live = yaml.safe_load((ROOT / "configs" / "sources_senate_live.yaml").read_text())
    votehub = next(s for s in live["sources"] if s["id"] == "votehub_senate_polls_2026")
    assert votehub["parser_args"]["race_parties"]["US-SEN-NE-2026"] == ["IND", "REP"]


def test_fundamentals_sign_for_independent_without_democrat() -> None:
    assert FundamentalsModel.party_sign("IND", {"IND", "REP"}) == 1.0
    assert FundamentalsModel.party_sign("IND", {"DEM", "IND", "REP"}) == -1.0
    assert FundamentalsModel.party_sign("DEM", {"DEM", "REP"}) == 1.0
    assert FundamentalsModel.party_sign("REP", {"IND", "REP"}) == -1.0


def _ind_vs_rep_draws() -> tuple[SimpleNamespace, pl.DataFrame]:
    # Race NE: IND wins draws 0,1; race X (DEM vs REP): DEM wins draws 0,2.
    rows = []
    for draw in range(4):
        ind_wins = draw in (0, 1)
        dem_wins = draw in (0, 2)
        rows += [
            {"draw_id": draw, "race_id": "NE", "party": "IND", "winner": ind_wins},
            {"draw_id": draw, "race_id": "NE", "party": "REP", "winner": not ind_wins},
            {"draw_id": draw, "race_id": "X", "party": "DEM", "winner": dem_wins},
            {"draw_id": draw, "race_id": "X", "party": "REP", "winner": not dem_wins},
        ]
    draws = pl.DataFrame(rows).with_columns(pl.lit(0.5).alias("vote_share"))
    catalog = pl.DataFrame(
        {
            "race_id": ["NE", "X"],
            "office_type": ["senate"] * 2,
            "control_body": ["senate"] * 2,
            "seats": [1, 1],
        }
    )
    return SimpleNamespace(race_catalog=catalog), draws


def test_control_with_and_without_independent() -> None:
    bundle, draws = _ind_vs_rep_draws()
    config = {"simulation_count": 4, "control_thresholds": {"senate": 51}}
    holdovers = {"DEM": 50, "REP": 49}
    caucus = {"races": {"NE": "DEM"}, "headline": "exclude"}
    engine = SimulationEngine(config, holdovers=holdovers, independent_caucus=caucus)
    control = engine._control_forecasts(bundle, draws)
    dem = control.filter(pl.col("party") == "DEM").row(0, named=True)
    # Without Osborn: DEM 51 only when X won (draws 0, 2). With: also draw 1 (IND + 50).
    assert dem["control_probability_without_independents"] == 0.5
    assert dem["control_probability_with_independents"] == 0.75
    assert dem["control_probability"] == 0.5
    assert json.loads(dem["independent_caucus_races"]) == ["NE"]
    include = SimulationEngine(
        config, holdovers=holdovers, independent_caucus={**caucus, "headline": "include"}
    )._control_forecasts(bundle, draws)
    assert include.filter(pl.col("party") == "DEM")["control_probability"].item() == 0.75
    plain = SimulationEngine(config, holdovers=holdovers)._control_forecasts(bundle, draws)
    row = plain.filter(pl.col("party") == "DEM").row(0, named=True)
    assert row["independent_caucus_headline"] == "none"
    assert row["control_probability_with_independents"] == row["control_probability"]
    assert np.isclose(plain.filter(pl.col("party") == "IND")["seat_count_mean"].item(), 0.5)


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "build_senate_2026_panel", ROOT / "scripts" / "build_senate_2026_panel.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_script_on_small_fixture() -> None:
    script = _load_script()
    seats = [
        {"office": "senate", "jurisdiction": "NE", "special": "0", "cook_pvi": "-10.0",
         "incumbents": "Pete Ricketts", "incumbent_parties": "REP", "open_seat": "0"},
        {"office": "senate", "jurisdiction": "WY", "special": "0", "cook_pvi": "-23.0",
         "incumbents": "Cynthia Lummis", "incumbent_parties": "REP", "open_seat": "1"},
        {"office": "house", "jurisdiction": "AK-01", "special": "0", "cook_pvi": "-6",
         "incumbents": "", "incumbent_parties": "", "open_seat": "0"},
    ]  # fmt: skip
    ballot = [
        {"office": "senate", "jurisdiction": "Nebraska", "name": "Dan Osborn", "party": "IND"},
        {"office": "senate", "jurisdiction": "Nebraska", "name": "Pete Ricketts", "party": "REP"},
        {"office": "senate", "jurisdiction": "Wyoming", "name": "James W. Byrd", "party": "DEM"},
        {"office": "senate", "jurisdiction": "Wyoming", "name": "Harriet Hageman", "party": "REP"},
    ]  # fmt: skip
    president = [
        {"year": "2024", "state_po": st, "party_simplified": p, "candidatevotes": v,
         "totalvotes": "100", "writein": "False"}
        for st, p, v in [("NE", "DEMOCRAT", "39"), ("NE", "REPUBLICAN", "59"),
                         ("WY", "DEMOCRAT", "26"), ("WY", "REPUBLICAN", "72")]
    ]  # fmt: skip
    senate = [
        {"year": "2024", "state_po": "NE", "stage": "GEN", "special": "True",
         "candidate": "PETE RICKETTS", "party_simplified": "REPUBLICAN", "candidatevotes": "62"},
        {"year": "2024", "state_po": "NE", "stage": "GEN", "special": "True",
         "candidate": "PRESTON LOVE JR.", "party_simplified": "DEMOCRAT", "candidatevotes": "38"},
        {"year": "2020", "state_po": "WY", "stage": "gen", "special": "False",
         "candidate": "CYNTHIA M. LUMMIS", "party_simplified": "OTHER", "candidatevotes": "73"},
        {"year": "2020", "state_po": "WY", "stage": "gen", "special": "False",
         "candidate": "MERAV BEN DAVID", "party_simplified": "OTHER", "candidatevotes": "27"},
        {"year": "2020", "state_po": "WY", "stage": "gen", "special": "False",
         "candidate": "UNDER VOTES", "party_simplified": "OTHER", "candidatevotes": "5"},
    ]  # fmt: skip
    rows = {r["state"]: r for r in script.build_rows(seats, ballot, president, senate, [])}
    assert set(rows) == {"NE", "WY"}
    assert rows["NE"]["dem_party"] == "IND" and rows["NE"]["dem_option_id"].endswith("-I")
    assert rows["NE"]["previous_election"] == "2024 gen special"
    assert rows["NE"]["dem_previous_vote_share"] == "0.38"
    assert rows["NE"]["rep_incumbent"] == "True"
    assert rows["WY"]["incumbent_party"] == "none"
    assert rows["WY"]["dem_previous_vote_share"] == "0.27"
    assert rows["WY"]["pres_2024_margin"] == "-46.0"
    assert script._same_person("Dan Sullivan", "Dan S. Sullivan")
    assert not script._same_person("Dan Sullivan", "Dan Osborn")


def test_environment_precedence(tmp_path: Path) -> None:
    from civic_signal.features.environment import resolve_environment

    assert resolve_environment(tmp_path)["kind"] == "none"
    (tmp_path / "measured_environment.json").write_text(
        json.dumps({"generic_ballot_margin_d": 8.2, "source": "wiki", "measured_at": "d"})
    )
    measured = resolve_environment(tmp_path)
    assert measured["kind"] == "measured" and measured["generic_ballot_margin_d"] == 8.2
    (tmp_path / "environment_override.json").write_text(
        json.dumps({"generic_ballot_margin_d": 2.5, "set_at": "2026-07-09", "reason": "x"})
    )
    override = resolve_environment(tmp_path)
    assert override["kind"] == "override"
    assert override["generic_ballot_margin_d"] == 2.5
    assert override["measured_margin_d"] == 8.2


def test_repo_uses_measured_environment() -> None:
    from civic_signal.features.environment import resolve_environment

    env = resolve_environment(ROOT / "fixtures")
    assert env["kind"] == "measured"
    rows = {row["state"]: row for row in _panel_2026()}
    expected = round(env["generic_ballot_margin_d"] - 2.0, 4)
    assert float(rows["GA"]["national_swing"]) == expected


def test_measure_environment_parses_snapshot_table() -> None:
    spec = importlib.util.spec_from_file_location(
        "measure_environment", ROOT / "scripts" / "measure_environment.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    text = (
        '<section begin="GenericBallotAgg" />\n|-\n|[[Decision Desk HQ]]\n'
        "|{{Party shading/Democratic}} |'''Democrats +7.8%'''\n|-\n|[[FiftyPlusOne]]\n"
        "|'''Republicans +1.0%'''\n|-\n|colspan=2 |'''Average'''\n|'''Democrats +3.4%'''\n"
        '|-\n<section end="GenericBallotAgg" />'
    )
    payload = module.measure(text, "2026-10-19", source="snapshot")
    assert payload["aggregators"] == {"Decision Desk HQ": 7.8, "FiftyPlusOne": -1.0}
    assert payload["generic_ballot_margin_d"] == 3.4
    assert payload["measured_at"] == "2026-10-19"
