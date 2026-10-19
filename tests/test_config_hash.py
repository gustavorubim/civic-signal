"""model_config_hash covers static config only (audit P1-8)."""

from __future__ import annotations

import polars as pl

from civic_signal.pipeline import ForecastPipeline


def test_same_config_different_data_same_hash() -> None:
    base = {"seed": 1, "simulation_count": 100, "_inference_engine": "bayes"}
    first = {**base, "_fundamentals_prior_rows": [{"race_id": "A", "mean_share": 0.5}]}
    second = {
        **base,
        "_fundamentals_prior_rows": [{"race_id": "A", "mean_share": 0.6}],
        "_learned_horizon_drift_sd_per_sqrt_day": 0.007,
    }
    assert ForecastPipeline._config_hash(first) == ForecastPipeline._config_hash(second)
    assert ForecastPipeline._config_hash(base) != ForecastPipeline._config_hash({**base, "seed": 2})


def test_data_hashes_record_data_separately() -> None:
    manifest = pl.DataFrame(
        {
            "source_id": ["votehub"],
            "content_hash": ["abc"],
            "fetch_status": ["fresh"],
            "fetched_at": ["2026-10-19T12:00:00+00:00"],
        }
    )
    one = ForecastPipeline._data_hashes(
        {"seed": 1, "_fundamentals_prior_rows": [{"a": 1}]}, manifest
    )
    two = ForecastPipeline._data_hashes(
        {"seed": 1, "_fundamentals_prior_rows": [{"a": 2}]}, manifest
    )
    assert one["model_config_hash"] == two["model_config_hash"]
    assert one["fundamentals_prior_sha256"] != two["fundamentals_prior_sha256"]
    assert one["sources"]["votehub"] == {
        "sha256": "abc",
        "fetch_status": "fresh",
        "fetched_at": "2026-10-19T12:00:00+00:00",
    }
    assert ForecastPipeline._data_hashes({}, pl.DataFrame())["fundamentals_prior_sha256"] is None
