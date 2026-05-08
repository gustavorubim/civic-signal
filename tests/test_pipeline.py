from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from election_outcomes.config import ProjectContext
from election_outcomes.features import FeatureBuilder
from election_outcomes.ingest import SyncRunner
from election_outcomes.models import (
    EnsembleModel,
    FundamentalsModel,
    MarketModel,
    PollingModel,
    PublicSignalModel,
    SimulationEngine,
)
from election_outcomes.normalize import CuratedDataBuilder
from election_outcomes.performance import simulate_binary_draw_arrays
from election_outcomes.pipeline import ForecastPipeline
from election_outcomes.scoring import BacktestRunner, score_predictions

ROOT = Path(__file__).resolve().parents[1]


def context(tmp_path: Path) -> ProjectContext:
    return ProjectContext.create(
        root=ROOT,
        data_dir=tmp_path / "data",
        artifacts_dir=tmp_path / "artifacts",
    )


def build_bundle(tmp_path: Path):
    ctx = context(tmp_path)
    first = SyncRunner(ctx).run()
    CuratedDataBuilder(ctx).run()
    bundle = FeatureBuilder(ctx).run()
    return ctx, first, bundle


def test_sync_is_incremental_and_records_manifest(tmp_path: Path) -> None:
    ctx = context(tmp_path)
    first = SyncRunner(ctx).run()
    second = SyncRunner(ctx).run()

    assert first.fetched_sources == 8
    assert first.failed_sources == 0
    assert second.fetched_sources == 0
    assert second.skipped_sources == 8
    assert (ctx.raw_dir / "source_manifest.parquet").exists()
    assert first.manifest.filter(pl.col("content_hash") == "").is_empty()


def test_feature_builder_assigns_tiers_and_filters_blank_rows(tmp_path: Path) -> None:
    _ctx, _sync, bundle = build_bundle(tmp_path)
    tiers = dict(zip(bundle.race_catalog["race_id"], bundle.race_catalog["tier"], strict=True))

    assert tiers["US-SEN-AZ-2026"] == "A"
    assert tiers["US-HOUSE-CA45-2026"] == "B"
    assert tiers["MAYOR-SPRINGFIELD-2026"] == "C"
    assert bundle.race_catalog["race_id"].null_count() == 0


def test_component_models_and_ensemble_respect_admission(tmp_path: Path) -> None:
    ctx, _sync, bundle = build_bundle(tmp_path)
    active = ForecastPipeline._active_bundle(bundle, "2026-05-08")
    model_config = ctx.read_yaml("model.yaml")

    estimates = [
        PollingModel().run(active),
        FundamentalsModel().run(active),
        MarketModel(model_config).run(active),
        PublicSignalModel(trusted=False).run(active),
    ]
    ensemble = EnsembleModel(model_config).run(active, estimates)

    assert not ensemble.is_empty()
    assert "MAYOR-SPRINGFIELD-2026" not in ensemble["race_id"].to_list()
    public = estimates[-1]
    assert public.filter(pl.col("race_id") == "US-SEN-AZ-2026")["admitted"].sum() == 0


def test_simulation_outputs_forecasts_control_and_ecosystem(tmp_path: Path) -> None:
    ctx, _sync, bundle = build_bundle(tmp_path)
    active = ForecastPipeline._active_bundle(bundle, "2026-05-08")
    model_config = ctx.read_yaml("model.yaml")
    estimates = [
        PollingModel().run(active),
        FundamentalsModel().run(active),
        MarketModel(model_config).run(active),
    ]
    ensemble = EnsembleModel(model_config).run(active, estimates)
    outputs = SimulationEngine(model_config).run(active, ensemble)

    assert outputs.draws.height == 6000
    assert outputs.control_forecasts.height > 0
    assert outputs.ecosystem_forecasts.height == 3
    tier_c = outputs.race_forecasts.filter(pl.col("race_id") == "MAYOR-SPRINGFIELD-2026")
    assert tier_c["winner_probability"].null_count() == tier_c.height


def test_forecast_run_writes_required_artifacts_and_rewards(tmp_path: Path) -> None:
    ctx = context(tmp_path)
    out_dir = ForecastPipeline(ctx).run_forecast(as_of="2026-05-08", run_id="unit")
    required = {
        "race_catalog.parquet",
        "race_forecasts.parquet",
        "forecast_draws.parquet",
        "control_forecasts.parquet",
        "ecosystem_forecasts.parquet",
        "source_manifest.parquet",
        "diagnostics.html",
        "reward_card.json",
        "methodology_snapshot.md",
        "plot_manifest.json",
        "performance.json",
        "plots",
    }

    assert {path.name for path in out_dir.iterdir()} == required
    plot_manifest = json.loads((out_dir / "plot_manifest.json").read_text(encoding="utf-8"))
    plot_paths = [
        out_dir / entry["path"] for entries in plot_manifest.values() for entry in entries
    ]
    assert len(plot_manifest["calibration"]) >= 3
    assert len(plot_manifest["projection"]) >= 4
    assert all(path.exists() and path.stat().st_size > 0 for path in plot_paths)
    forecasts = pl.read_parquet(out_dir / "race_forecasts.parquet")
    assert {"model_config_hash", "source_manifest_hash"}.issubset(forecasts.columns)
    reward_card = json.loads((out_dir / "reward_card.json").read_text(encoding="utf-8"))
    rewards = reward_card["rewards"]
    assert rewards["R0_build"]["passed"] is None
    assert all(value["passed"] is True for key, value in rewards.items() if key != "R0_build")
    performance = json.loads((out_dir / "performance.json").read_text(encoding="utf-8"))
    assert performance["engine"] in {"numba", "python"}
    assert performance["simulation_count"] == 1000


def test_backtest_and_report_rebuild(tmp_path: Path) -> None:
    ctx = context(tmp_path)
    pipeline = ForecastPipeline(ctx)
    pipeline.run_forecast(as_of="2026-05-08", run_id="reportable")
    payload = pipeline.run_backtest(run_id="bt")
    report_dir = pipeline.rebuild_report("reportable")

    assert payload["row_count"] == 4
    assert payload["ablations"]["ensemble"]["beats_or_matches_baseline"]
    assert (ctx.artifacts_dir / "backtests" / "bt" / "scorecard.json").exists()
    assert (
        (report_dir / "diagnostics.html").read_text(encoding="utf-8").startswith("<!doctype html>")
    )


def test_presidential_result_comparison(tmp_path: Path) -> None:
    ctx = context(tmp_path)
    pipeline = ForecastPipeline(ctx)
    pipeline.run_forecast(as_of="2024-10-01", run_id="pres-2024")
    payload = pipeline.compare_results(
        forecast_run_id="pres-2024",
        comparison_id="pres-actuals",
        cycle=2024,
        office_type="president",
    )
    comparison_dir = Path(payload["output_dir"])
    comparison = pl.read_parquet(comparison_dir / "result_comparison.parquet")

    assert payload["race_count"] == 1
    assert payload["row_count"] == 2
    assert payload["winner_accuracy"] in {0.0, 1.0}
    assert (comparison_dir / "result_comparison_summary.json").exists()
    assert (comparison_dir / "result_comparison.html").exists()
    assert (comparison_dir / "plots" / "vote_share_forecast_vs_actual.png").stat().st_size > 0
    assert comparison.filter(pl.col("actual_winner")).height == 1


def test_score_predictions_handles_empty_and_real_rows(tmp_path: Path) -> None:
    _ctx, _sync, _bundle = build_bundle(tmp_path)
    empty_scores = score_predictions(pl.DataFrame())
    real_scores = BacktestRunner(_ctx).evaluate()["metrics"]["ensemble"]

    assert str(empty_scores["brier"]) == "nan"
    assert real_scores["brier"] < 0.25
    assert 0.0 <= real_scores["expected_calibration_error"] <= 1.0


def test_performance_benchmark_and_python_kernel_fallback(tmp_path: Path) -> None:
    ctx = context(tmp_path)
    payload = ForecastPipeline(ctx).run_benchmark(
        as_of="2026-05-08", run_id="perf", draws=100, repeats=1
    )
    assert payload["forecast_draw_rows"] == 600
    assert payload["rows_per_second"] > 0
    assert (ctx.artifacts_dir / "benchmarks" / "perf" / "performance_benchmark.json").exists()

    arrays = simulate_binary_draw_arrays(
        first_shares=pl.Series([0.52]).to_numpy(),
        turnout_bases=pl.Series([1000.0]).to_numpy(),
        national_errors=pl.Series([0.0, 0.01]).to_numpy(),
        local_errors=pl.DataFrame({"a": [0.0, -0.02]}).to_numpy().T,
        use_numba=False,
    )
    assert len(arrays[0]) == 4
    assert arrays[5][0] == 0.52
