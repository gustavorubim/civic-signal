"""End-to-end evidence that nested evaluation uses the publication simulation path."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from civic_signal.config import ProjectContext
from civic_signal.ingest import SyncRunner
from civic_signal.normalize import CuratedDataBuilder
from civic_signal.pipeline import ForecastPipeline
from civic_signal.scoring.backtest import NestedBacktestRunner

ROOT = Path(__file__).resolve().parents[1]


def _fixture_context(tmp_path: Path) -> ProjectContext:
    context = ProjectContext.create(
        root=ROOT,
        data_dir=tmp_path / "data",
        artifacts_dir=tmp_path / "artifacts",
    )
    SyncRunner(context).run()
    CuratedDataBuilder(context).run()
    return context


def test_nested_backtest_scores_real_fixture_through_simulation(tmp_path: Path) -> None:
    """Real NestedBacktestRunner on fixture data; outer fold uses SimulationEngine."""
    context = _fixture_context(tmp_path)

    payload = NestedBacktestRunner(context).run(
        "nested-publication-path",
        scenario="president_state",
        holdout_cycle=2024,
        inference_engine="kalman",
    )
    out_dir = context.artifacts_dir / "backtests" / "nested-publication-path"
    persisted = json.loads((out_dir / "nested_evaluation.json").read_text(encoding="utf-8"))
    manifest = pl.read_parquet(out_dir / "fold_manifest.parquet")
    predictions = pl.read_parquet(out_dir / "nested_predictions.parquet")

    assert persisted["row_count"] == payload["row_count"] == predictions.height
    assert payload["fold_count"] == manifest.height == 1
    assert payload["outer_cycle_excluded"] is True
    assert payload["held_out_permutation_affects_prior_folds"] is False
    assert payload["held_out_permutation_canary"]["passed"] is True
    assert payload["exact_pipeline"] is True
    assert payload["exact_pipeline_scope"] == "components+ensemble+SimulationEngine"
    assert "2024" in payload["training_lineage_sha256"]
    assert len(payload["training_lineage_sha256"]["2024"]) == 64
    assert payload["fold_lineage"][0]["outer_cycle"] == 2024
    assert payload["fold_lineage"][0]["simulation_engine_used"] is True
    assert payload["fold_lineage"][0]["held_out_permutation_canary_passed"] is True
    assert persisted["exact_pipeline"] is True
    assert (
        persisted["training_lineage_sha256"]["2024"] == payload["training_lineage_sha256"]["2024"]
    )
    assert predictions.height > 0
    assert predictions["cycle"].unique().to_list() == [2024]
    assert set(predictions["as_of_offset_days"].unique().to_list()) == {1, 7, 30, 60, 90}
    assert predictions["polling_inference_engine"].unique().to_list() == ["kalman"]
    train_cycles = json.loads(manifest["train_cycles"].item())
    inner_cycles = json.loads(manifest["inner_validation_cycles"].item())
    assert train_cycles[-2:] == [2016, 2020]
    assert all(cycle < 2024 for cycle in train_cycles)
    assert inner_cycles == train_cycles[1:]
    assert manifest["outer_cycle_excluded"].item() is True
    assert manifest["simulation_engine_used"].item() is True
    assert len(manifest["training_lineage_sha256"].item()) == 64
    assert 0.0 <= payload["metrics"]["ensemble"]["brier"] <= 1.0


def test_forecast_pipeline_run_nested_backtest_real_fixture(tmp_path: Path) -> None:
    """Fixture path through ForecastPipeline.run_nested_backtest (real nested runner)."""
    context = _fixture_context(tmp_path)

    payload = ForecastPipeline(context).run_nested_backtest(
        run_id="nested-pipeline-path",
        scenario="president_state",
        holdout_cycle=2024,
        inference_engine="kalman",
    )
    out_dir = context.artifacts_dir / "backtests" / "nested-pipeline-path"
    persisted = json.loads((out_dir / "nested_evaluation.json").read_text(encoding="utf-8"))
    manifest = pl.read_parquet(out_dir / "fold_manifest.parquet")

    assert payload["exact_pipeline"] is True
    assert payload["exact_pipeline_scope"] == "components+ensemble+SimulationEngine"
    assert payload["outer_cycle_excluded"] is True
    assert payload["held_out_permutation_canary"]["passed"] is True
    assert payload["training_lineage_sha256"]["2024"]
    assert persisted["exact_pipeline"] is True
    assert persisted["fold_lineage"][0]["simulation_engine_used"] is True
    assert manifest["simulation_engine_used"].item() is True
    assert payload["row_count"] > 0
