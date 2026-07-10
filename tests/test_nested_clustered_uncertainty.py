from __future__ import annotations

import math

import polars as pl
import pytest

from civic_signal.scoring.backtest import NestedBacktestRunner


def _cluster_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "cycle": [2000, 2004, 2008],
            "race_id": ["R-2000", "R-2004", "R-2008"],
            "option_id": ["D", "D", "D"],
            "actual_winner": [True, False, True],
            "actual_vote_share": [0.52, 0.48, 0.51],
            "lower_90": [0.4, 0.4, 0.4],
            "upper_90": [0.6, 0.6, 0.6],
            "ensemble_probability": [0.8, 0.4, 0.55],
            "prior_only_probability": [0.6, 0.3, 0.5],
        }
    )


def test_cycle_cluster_bootstrap_uses_equal_cycle_weighting() -> None:
    frame = _cluster_frame()
    result = NestedBacktestRunner._paired_cycle_clustered_uncertainty(
        frame,
        config={
            "minimum_cycles_for_uncertainty": 3,
            "cycle_bootstrap_replicates": 500,
            "cycle_bootstrap_seed": 7,
        },
    )
    prior = result["comparisons"]["prior_only"]
    expected_cycle_differences = [0.04 - 0.16, 0.16 - 0.09, 0.2025 - 0.25]

    assert result["status"] == "estimated"
    assert result["cluster_unit"] == "cycle"
    assert result["row_level_resampling"] is False
    assert result["equal_cycle_weighting"] is True
    assert prior["status"] == "estimated"
    assert prior["independent_cycle_count"] == 3
    assert prior["brier_difference"]["estimate"] == pytest.approx(
        sum(expected_cycle_differences) / 3
    )
    assert prior["brier_difference"]["lower_95"] <= prior["brier_difference"]["estimate"]
    assert prior["brier_difference"]["upper_95"] >= prior["brier_difference"]["estimate"]
    assert math.isfinite(prior["brier_difference"]["cluster_standard_error"])


def test_duplicate_race_rows_do_not_create_pseudo_independent_cycles() -> None:
    original = _cluster_frame()
    duplicated_first_cycle = pl.concat(
        [original, *[original.filter(pl.col("cycle") == 2000) for _ in range(99)]],
        how="vertical",
    )
    config = {
        "minimum_cycles_for_uncertainty": 3,
        "cycle_bootstrap_replicates": 500,
        "cycle_bootstrap_seed": 17,
    }

    original_result = NestedBacktestRunner._paired_cycle_clustered_uncertainty(
        original, config=config
    )["comparisons"]["prior_only"]
    duplicated_result = NestedBacktestRunner._paired_cycle_clustered_uncertainty(
        duplicated_first_cycle, config=config
    )["comparisons"]["prior_only"]

    assert duplicated_result["independent_cycle_count"] == 3
    assert original_result["brier_difference"] == duplicated_result["brier_difference"]
    assert original_result["log_score_difference"] == duplicated_result["log_score_difference"]
    duplicated_cycle = next(
        row for row in duplicated_result["cycle_estimates"] if row["cycle"] == 2000
    )
    assert duplicated_cycle["paired_row_count"] == 100


def test_many_rows_from_two_cycles_remain_insufficient() -> None:
    two_cycles = _cluster_frame().filter(pl.col("cycle") < 2008)
    expanded = pl.concat([two_cycles for _ in range(100)], how="vertical")

    result = NestedBacktestRunner._paired_cycle_clustered_uncertainty(
        expanded,
        config={
            "minimum_cycles_for_uncertainty": 3,
            "cycle_bootstrap_replicates": 200,
        },
    )
    prior = result["comparisons"]["prior_only"]

    assert result["status"] == "insufficient_evidence"
    assert prior["status"] == "insufficient_evidence"
    assert prior["independent_cycle_count"] == 2
    assert "race rows are not treated as independent replicates" in prior["reason"]


def test_baseline_scorecard_preserves_market_absence() -> None:
    frame = _cluster_frame().with_columns(
        pl.col("prior_only_probability").alias("previous_cycle_swing_probability"),
        pl.col("prior_only_probability").alias("fundamentals_only_probability"),
        pl.col("prior_only_probability").alias("poll_average_probability"),
        pl.lit(None, dtype=pl.Float64).alias("market_implied_probability"),
    )

    metrics = NestedBacktestRunner._baseline_metrics(frame)

    assert metrics["prior_only"]["status"] == "estimated"
    assert metrics["previous_cycle_swing"]["cycle_count"] == 3
    assert metrics["fundamentals_only"]["row_count"] == 3
    assert metrics["poll_average"]["status"] == "estimated"
    assert metrics["markets_if_present"] == {
        "status": "not_available",
        "row_count": 0,
        "reason": "no eligible outer-fold rows for this comparator",
    }


def test_poll_average_is_normalized_and_requires_multiple_polled_options() -> None:
    polls = pl.DataFrame(
        {
            "race_id": ["R1", "R1", "R1", "R1", "R2"],
            "option_id": ["D", "D", "R", "R", "D"],
            "pct": [52.0, 50.0, 48.0, 47.0, 60.0],
        }
    )

    probabilities = NestedBacktestRunner._poll_average_probabilities(polls)

    assert probabilities[("R1", "D")] == pytest.approx(51.0 / 98.5)
    assert probabilities[("R1", "R")] == pytest.approx(47.5 / 98.5)
    assert ("R2", "D") not in probabilities
