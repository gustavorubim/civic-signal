"""True tipping-point metric (audit P1-2)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest

from civic_signal.models import SimulationEngine

ROOT = Path(__file__).resolve().parents[1]
# D margin per draw for three one-seat races; DEM holds 1 seat and needs 3.
MARGINS = {
    "A": [0.10, -0.01, -0.30, 0.30],
    "B": [0.02, 0.20, -0.10, 0.10],
    "C": [-0.05, 0.05, -0.20, 0.20],
}


def _joined() -> pl.DataFrame:
    rows = []
    for race, margins in MARGINS.items():
        for draw, margin in enumerate(margins):
            dem = 0.5 + margin / 2
            rows += [
                {"draw_id": draw, "race_id": race, "party": "DEM", "vote_share": dem,
                 "winner": margin > 0},
                {"draw_id": draw, "race_id": race, "party": "REP", "vote_share": 1 - dem,
                 "winner": margin <= 0},
            ]  # fmt: skip
    return pl.DataFrame(rows).with_columns(
        pl.lit("senate").alias("control_body"), pl.lit(1).alias("seats")
    )


def test_known_tipping_races_and_sum_to_one() -> None:
    engine = SimulationEngine({"simulation_count": 4})
    result = engine._tipping_point_probabilities(_joined(), "senate", "DEM", 1, 3)
    # draw0: A,B,C -> B crosses 3; draw1: B,C,A -> C; draw2 (all lost): B,C,A -> C;
    # draw3: A,C,B -> C.
    rates = {x["race_id"]: x["tipping_point_probability"] for x in result["races"]}
    assert rates == {"B": 0.25, "C": 0.75}
    assert result["none"] == 0.0
    assert sum(rates.values()) + result["none"] == pytest.approx(1.0)
    rep = engine._tipping_point_probabilities(_joined(), "senate", "REP", 1, 3)
    assert sum(x["tipping_point_probability"] for x in rep["races"]) == pytest.approx(1.0)


def test_holdovers_already_at_threshold_means_no_tipping_race() -> None:
    engine = SimulationEngine({"simulation_count": 4})
    result = engine._tipping_point_probabilities(_joined(), "senate", "DEM", 3, 3)
    assert result == {"races": [], "none": 1.0}
    assert engine._tipping_point_probabilities(
        pl.DataFrame(schema={"control_body": pl.Utf8}), "senate", "DEM", 0, 1
    ) == {"races": [], "none": 1.0}


def test_race_without_party_option_sorts_last() -> None:
    joined = _joined().with_columns(
        pl.when(pl.col("race_id") == "A")
        .then(pl.lit("REP"))
        .otherwise(pl.col("party"))
        .alias("party")
    )
    engine = SimulationEngine({"simulation_count": 4})
    result = engine._tipping_point_probabilities(joined, "senate", "DEM", 1, 3)
    # A has no DEM option, so DEM needs both B and C; the weaker of the two tips:
    # C in draws 0-2, B in draw 3 (C +0.20 > B +0.10).
    rates = {x["race_id"]: x["tipping_point_probability"] for x in result["races"]}
    assert rates == {"C": 0.75, "B": 0.25}


def test_control_frame_carries_tipping_and_decisive_columns() -> None:
    joined = _joined()
    catalog = pl.DataFrame(
        {"race_id": list(MARGINS), "office_type": ["senate"] * 3,
         "control_body": ["senate"] * 3, "seats": [1] * 3}
    )  # fmt: skip
    engine = SimulationEngine(
        {"simulation_count": 4, "control_thresholds": {"senate": 3}}, holdovers={"DEM": 1}
    )
    control = engine._control_forecasts(
        SimpleNamespace(race_catalog=catalog), joined.drop(["control_body", "seats"])
    )
    dem = control.filter(pl.col("party") == "DEM").row(0, named=True)
    assert json.loads(dem["tipping_point_races"])[0] == "C"
    assert json.loads(dem["tipping_point_probability"])[0]["race_id"] == "C"
    assert dem["tipping_point_none_probability"] == 0.0
    assert {"race_id", "decisive_probability"} <= set(json.loads(dem["decisive_probability"])[0])


def _load(name: str):
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_tipping_rates_new_and_legacy_rows() -> None:
    package_run = _load("package_run")
    new_row = {
        "tipping_point_probability": json.dumps(
            [{"race_id": "C", "tipping_point_probability": 0.75}]
        ),
        "control_threshold": 3,
    }
    assert package_run.tipping_rates(new_row) == ("tipping_point", [{"race_id": "C", "rate": 0.75}])
    legacy = {"pivotal_rates": json.dumps([{"race_id": "B", "pivotal_rate": 0.4}]),
              "control_threshold": 3, "party": "DEM"}  # fmt: skip
    assert package_run.tipping_rates(legacy) == ("decisive_legacy", [{"race_id": "B", "rate": 0.4}])
