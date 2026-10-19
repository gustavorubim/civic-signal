"""Scenario payload helpers: holdovers/caucus, as_of policy, catalog filters."""

from __future__ import annotations

from datetime import date

import polars as pl

from civic_signal.config.scenarios import Scenario


def test_holdovers_fold_independents_into_caucus() -> None:
    scenario = Scenario(
        "s",
        {
            "holdovers": {"dem": 32, "ind": 2, "rep": "31", "bad": "x"},
            "caucus_with": {"ind": "dem"},
            "control_body": "senate",
        },
    )
    assert scenario.holdovers == {"DEM": 32, "IND": 2, "REP": 31}
    assert scenario.caucus_with == {"IND": "DEM"}
    assert scenario.holdover_caucus_seats == {"DEM": 34, "REP": 31}
    assert scenario.control_body == "senate"
    assert Scenario("t", {}).holdovers == {}
    assert Scenario("t", {"holdovers": [1]}).holdovers == {}
    assert Scenario("t", {}).caucus_with == {}
    assert Scenario("t", {}).control_body is None


def test_resolve_as_of_policies() -> None:
    live = Scenario("live", {"as_of_policy": "today", "election_date": "2026-11-03"})
    assert live.resolve_as_of("2026-01-01") == "2026-01-01"
    assert live.resolve_as_of(None, today=date(2026, 10, 19)) == "2026-10-19"
    assert live.resolve_as_of(None, today=date(2026, 12, 1)) == "2026-11-02"
    fixed = Scenario("fixed", {"default_as_of": "2026-05-08"})
    assert fixed.as_of_policy == "default"
    assert fixed.resolve_as_of(None) == "2026-05-08"
    assert Scenario("none", {}).resolve_as_of(None) is None


def test_filter_catalog_by_list_scalar_and_cycle() -> None:
    catalog = pl.DataFrame(
        {
            "race_id": ["a", "b", "c"],
            "office_type": ["senate", "house", "senate"],
            "cycle": [2026, 2026, 2024],
        }
    )
    listed = Scenario("s", {"office_types": ["senate"], "cycle": 2026})
    assert listed.filter_catalog(catalog)["race_id"].to_list() == ["a"]
    assert listed.filter_catalog(catalog, include_cycle=False)["race_id"].to_list() == ["a", "c"]
    scalar = Scenario("s", {"office_type": "house"})
    assert scalar.filter_catalog(catalog)["race_id"].to_list() == ["b"]
    unset = Scenario("s", {"office_type": None})
    assert unset.filter_catalog(catalog).height == 3
    assert Scenario("x_state", {"cycle": 2026, "family": "fam"}).storage_key == "fam"
    assert Scenario("x", {"cycle": 2026}).storage_key == "x"
    assert Scenario("x", {}).metadata() == {"name": "x"}
