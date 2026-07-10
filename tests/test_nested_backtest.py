from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import polars as pl

from civic_signal.config import ProjectContext
from civic_signal.features import FeatureBundle
from civic_signal.scoring.backtest import NestedBacktestRunner

ROOT = Path(__file__).resolve().parents[1]


def _context(tmp_path: Path) -> ProjectContext:
    return ProjectContext.create(
        root=ROOT,
        data_dir=tmp_path / "data",
        artifacts_dir=tmp_path / "artifacts",
    )


def _bundle() -> FeatureBundle:
    catalog = pl.DataFrame(
        {
            "race_id": ["R-2000", "R-2004", "R-2008"],
            "cycle": [2000, 2004, 2008],
            "election_date": ["2000-11-07", "2004-11-02", "2008-11-04"],
            "tier": ["A", "A", "A"],
            "tier_reason": ["test", "test", "test"],
        }
    )
    options = pl.DataFrame(
        {
            "race_id": ["R-2000", "R-2004", "R-2008"],
            "option_id": ["D-2000", "D-2004", "D-2008"],
            "previous_vote_share": [0.49, 0.51, 0.50],
            "party": ["DEM", "DEM", "DEM"],
        }
    )
    results = pl.DataFrame(
        {
            "race_id": ["R-2000", "R-2004", "R-2008"],
            "option_id": ["D-2000", "D-2004", "D-2008"],
            "winner": [False, True, True],
            "vote_share": [0.49, 0.51, 0.52],
            "source_hash": ["a" * 64, "b" * 64, "c" * 64],
        }
    )
    empty = pl.DataFrame(schema={"race_id": pl.Utf8, "source_hash": pl.Utf8})
    return FeatureBundle(
        races=catalog,
        options=options,
        polls=empty,
        markets=empty,
        public_signals=empty,
        fundamentals=empty,
        results=results,
        backtest_predictions=pl.DataFrame(),
        race_catalog=catalog,
    )


def _prediction_rows(cycle: int, count: int = 40) -> pl.DataFrame:
    actual = [index % 2 == 0 for index in range(count)]
    good = [0.7 if value else 0.3 for value in actual]
    return pl.DataFrame(
        {
            "race_id": [f"R-{cycle}-{index}" for index in range(count)],
            "option_id": [f"O-{index}" for index in range(count)],
            "geography": [f"G-{index}" for index in range(count)],
            "cycle": [cycle] * count,
            "as_of": [f"{cycle}-10-01"] * count,
            "as_of_offset_days": [30] * count,
            "polling_inference_engine": ["kalman"] * count,
            "office_type": ["president"] * count,
            "actual_winner": actual,
            "actual_vote_share": [0.52 if value else 0.48 for value in actual],
            "baseline_probability": [0.5] * count,
            "polls_probability": good,
            "fundamentals_probability": good,
            "markets_probability": [0.5] * count,
            "public_signals_probability": [0.5] * count,
            "prior_only_probability": [0.5] * count,
            "previous_cycle_swing_probability": [0.5] * count,
            "fundamentals_only_probability": good,
            "poll_average_probability": good,
            "market_implied_probability": [0.55 if value else 0.45 for value in actual],
            "ensemble_probability": good,
            "predicted_vote_share": good,
            "lower_90": [0.4] * count,
            "upper_90": [0.6] * count,
        }
    )


def test_inner_learning_uses_only_supplied_inner_rows(tmp_path: Path) -> None:
    runner = NestedBacktestRunner(_context(tmp_path))
    result = runner._fit_inner(
        _prediction_rows(2004),
        runner.context.read_yaml("model.yaml"),
        runner.context.read_yaml("backtests.yaml"),
    )

    weights = result["ensemble_learning"]["weight_learning"]
    calibration = result["ensemble_learning"]["probability_calibration"]
    assert weights["row_count"] == 40
    assert calibration["row_count"] == 40
    assert result["hyperparameters"]["row_count"] == 40


def test_nested_fold_manifest_excludes_outer_cycle_and_records_lineage(
    tmp_path: Path, monkeypatch
) -> None:
    bundle = _bundle()
    runner = NestedBacktestRunner(_context(tmp_path))
    monkeypatch.setattr("civic_signal.scoring.backtest.FeatureBuilder.run", lambda _self: bundle)
    monkeypatch.setattr(
        runner.base,
        "_rolling_origin_predictions",
        lambda **kwargs: _prediction_rows(int(kwargs["holdout_cycle"])),
    )
    monkeypatch.setattr(
        runner,
        "_publication_outer_fold",
        lambda **kwargs: (
            _prediction_rows(int(kwargs["target_cycle"]), count=4),
            {"simulation_engine_used": True, "posterior_draw_path_complete": True},
        ),
    )

    payload = runner.run("nested-unit", holdout_cycle=2008, inference_engine="kalman")
    out_dir = runner.context.artifacts_dir / "backtests" / "nested-unit"
    persisted = json.loads((out_dir / "nested_evaluation.json").read_text(encoding="utf-8"))
    manifest = pl.read_parquet(out_dir / "fold_manifest.parquet")
    row = manifest.row(0, named=True)
    expected_lineage = runner._training_lineage_hash(
        bundle, bundle.race_catalog.filter(pl.col("cycle") < 2008)
    )

    assert payload["exact_pipeline"] is True
    assert payload["exact_pipeline_scope"] == "components+ensemble+SimulationEngine"
    assert payload["outer_cycle_excluded"] is True
    assert payload["held_out_permutation_affects_prior_folds"] is False
    assert payload["held_out_permutation_canary"]["passed"] is True
    assert payload["held_out_permutation_canary"]["by_outer_cycle"]["2008"]["passed"] is True
    assert payload["training_lineage_sha256"]["2008"] == expected_lineage
    assert payload["fold_lineage"][0]["training_lineage_sha256"] == expected_lineage
    assert payload["fold_lineage"][0]["held_out_permutation_canary_passed"] is True
    assert persisted["exact_pipeline"] is True
    assert persisted["training_lineage_sha256"]["2008"] == expected_lineage
    assert json.loads(row["train_cycles"]) == [2000, 2004]
    assert json.loads(row["inner_validation_cycles"]) == [2004]
    assert row["fit_cycle_max"] < row["outer_cycle"]
    assert row["training_lineage_sha256"] == expected_lineage
    assert row["held_out_permutation_canary_passed"] is True
    assert set(payload["baseline_metrics"]) == set(runner.BASELINE_COLUMNS)
    assert payload["baseline_metrics"]["poll_average"]["status"] == "estimated"
    assert payload["paired_cycle_clustered_uncertainty"]["status"] == ("insufficient_evidence")
    assert (out_dir / "nested_evaluation.json").exists()
    assert (out_dir / "nested_predictions.parquet").exists()
    assert (out_dir / "baseline_scorecard.json").exists()
    assert (out_dir / "paired_cycle_clustered_uncertainty.json").exists()


def test_exact_pipeline_false_when_publication_simulation_path_not_used(
    tmp_path: Path, monkeypatch
) -> None:
    """Negative: exact_pipeline must not claim pass without SimulationEngine outer scoring."""
    bundle = _bundle()
    runner = NestedBacktestRunner(_context(tmp_path))
    monkeypatch.setattr("civic_signal.scoring.backtest.FeatureBuilder.run", lambda _self: bundle)
    monkeypatch.setattr(
        runner.base,
        "_rolling_origin_predictions",
        lambda **kwargs: _prediction_rows(int(kwargs["holdout_cycle"])),
    )
    monkeypatch.setattr(
        runner,
        "_publication_outer_fold",
        lambda **kwargs: (
            _prediction_rows(int(kwargs["target_cycle"]), count=4),
            {"simulation_engine_used": False, "posterior_draw_path_complete": False},
        ),
    )

    payload = runner.run("nested-no-sim", holdout_cycle=2008, inference_engine="kalman")
    persisted = json.loads(
        (
            runner.context.artifacts_dir / "backtests" / "nested-no-sim" / "nested_evaluation.json"
        ).read_text(encoding="utf-8")
    )

    assert payload["exact_pipeline"] is False
    assert payload["exact_pipeline"] is not True
    assert payload["exact_pipeline_scope"] is None
    assert persisted.get("exact_pipeline") is not True
    assert persisted.get("exact_pipeline_scope") is None
    assert "2008" in payload["training_lineage_sha256"]
    assert payload["fold_lineage"][0]["simulation_engine_used"] is False


def test_exact_pipeline_false_when_flag_missing_from_path_status(
    tmp_path: Path, monkeypatch
) -> None:
    """Negative: incomplete path-status metadata must not claim exact pipeline pass."""
    bundle = _bundle()
    runner = NestedBacktestRunner(_context(tmp_path))
    monkeypatch.setattr("civic_signal.scoring.backtest.FeatureBuilder.run", lambda _self: bundle)
    monkeypatch.setattr(
        runner.base,
        "_rolling_origin_predictions",
        lambda **kwargs: _prediction_rows(int(kwargs["holdout_cycle"])),
    )
    monkeypatch.setattr(
        runner,
        "_publication_outer_fold",
        lambda **kwargs: (_prediction_rows(int(kwargs["target_cycle"]), count=4), {}),
    )

    payload = runner.run("nested-missing-flags", holdout_cycle=2008, inference_engine="kalman")

    assert payload["exact_pipeline"] is False
    assert payload["exact_pipeline_scope"] is None
    assert payload.get("exact_pipeline") is not True


def test_held_out_outcome_permutation_cannot_change_training_lineage(tmp_path: Path) -> None:
    bundle = _bundle()
    runner = NestedBacktestRunner(_context(tmp_path))
    assert runner._held_out_permutation_canary(bundle, bundle.race_catalog, 2008) is True
    training = bundle.race_catalog.filter(pl.col("cycle") < 2008)
    original = runner._training_lineage_hash(bundle, training)
    changed = replace(
        bundle,
        results=bundle.results.with_columns(
            pl.when(pl.col("race_id") == "R-2004")
            .then(~pl.col("winner"))
            .otherwise(pl.col("winner"))
            .alias("winner")
        ),
    )
    assert runner._training_lineage_hash(changed, training) != original


def test_bayes_outer_fold_passes_posterior_draws_and_hides_results(
    tmp_path: Path, monkeypatch
) -> None:
    runner = NestedBacktestRunner(_context(tmp_path))
    bundle = _bundle()
    posterior = pl.DataFrame(
        {"draw_id": [0], "race_id": ["R-2008"], "option_id": ["D-2008"], "vote_share": [0.52]}
    )
    ensemble = pl.DataFrame(
        {
            "race_id": ["R-2008"],
            "option_id": ["D-2008"],
            "vote_share": [0.52],
            "uncertainty": [0.04],
        }
    )
    empty_component = pl.DataFrame()
    monkeypatch.setattr(
        runner.base,
        "_publication_components",
        lambda **_kwargs: (
            [empty_component, empty_component, empty_component, empty_component],
            ensemble,
            runner.context.read_yaml("model.yaml"),
            posterior,
        ),
    )
    captured: dict[str, object] = {}

    class FakeSimulation:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def run(self, model_bundle, _ensemble, posterior_draws=None):
            captured["result_rows"] = model_bundle.results.height
            captured["posterior"] = posterior_draws
            return SimpleNamespace(
                race_forecasts=pl.DataFrame(
                    {
                        "race_id": ["R-2008"],
                        "option_id": ["D-2008"],
                        "winner_probability": [0.6],
                        "vote_share_mean": [0.52],
                        "vote_share_p05": [0.45],
                        "vote_share_p95": [0.59],
                    }
                )
            )

    monkeypatch.setattr("civic_signal.models.simulation.SimulationEngine", FakeSimulation)
    catalog = bundle.race_catalog
    frame, path_status = runner._publication_outer_fold(
        bundle=bundle,
        train_catalog=catalog.filter(pl.col("cycle") < 2008),
        test_catalog=catalog.filter(pl.col("cycle") == 2008),
        target_cycle=2008,
        as_of="2008-10-01",
        as_of_offset_days=30,
        model_config=runner.context.read_yaml("model.yaml"),
        inference_engine="bayes",
        residual_covariance=pl.DataFrame(),
        holdovers=None,
    )

    assert path_status["simulation_engine_used"] is True
    assert path_status["posterior_draw_path_complete"] is True
    assert captured["result_rows"] == 0
    assert captured["posterior"] is posterior
    assert frame["ensemble_probability"].to_list() == [0.6]
    assert frame["prior_only_probability"].to_list() == [1.0]
    assert frame["previous_cycle_swing_probability"].to_list() == [0.5]
    assert frame["fundamentals_only_probability"].to_list() == [None]
    assert frame["poll_average_probability"].to_list() == [None]
    assert frame["market_implied_probability"].to_list() == [None]


def test_empty_ensemble_does_not_claim_exact_publication_path(tmp_path: Path, monkeypatch) -> None:
    runner = NestedBacktestRunner(_context(tmp_path))
    bundle = _bundle()
    empty = pl.DataFrame()
    monkeypatch.setattr(
        runner.base,
        "_publication_components",
        lambda **_kwargs: (
            [empty, empty, empty, empty],
            empty,
            runner.context.read_yaml("model.yaml"),
            empty,
        ),
    )
    catalog = bundle.race_catalog
    frame, path_status = runner._publication_outer_fold(
        bundle=bundle,
        train_catalog=catalog.filter(pl.col("cycle") < 2008),
        test_catalog=catalog.filter(pl.col("cycle") == 2008),
        target_cycle=2008,
        as_of="2008-10-01",
        as_of_offset_days=30,
        model_config=runner.context.read_yaml("model.yaml"),
        inference_engine="kalman",
        residual_covariance=pl.DataFrame(),
        holdovers=None,
    )

    assert frame.is_empty()
    assert path_status["simulation_engine_used"] is False
    assert path_status["posterior_draw_path_complete"] is False


def test_publication_components_never_expose_test_results_to_models(
    tmp_path: Path, monkeypatch
) -> None:
    runner = NestedBacktestRunner(_context(tmp_path))
    bundle = _bundle()
    observed_result_rows: list[int] = []
    estimate = pl.DataFrame(
        {
            "race_id": ["R-2008"],
            "option_id": ["D-2008"],
            "vote_share": [0.52],
            "uncertainty": [0.04],
            "marginal_win_probability": [0.6],
        }
    )

    class FakePolling:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def run(self, model_bundle):
            observed_result_rows.append(model_bundle.results.height)
            return estimate

        def posterior_draws(self, model_bundle):
            observed_result_rows.append(model_bundle.results.height)
            return pl.DataFrame({"draw_id": [0]})

    class FakeFundamentals:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def fit(self, _train_bundle):
            return self

        def run(self, model_bundle):
            observed_result_rows.append(model_bundle.results.height)
            return estimate

    class FakeComponent:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def run(self, model_bundle):
            observed_result_rows.append(model_bundle.results.height)
            return estimate

    class FakeEnsemble(FakeComponent):
        def run(self, model_bundle, _components):
            observed_result_rows.append(model_bundle.results.height)
            return estimate

    monkeypatch.setattr("civic_signal.scoring.backtest.PollingModel", FakePolling)
    monkeypatch.setattr("civic_signal.scoring.backtest.FundamentalsModel", FakeFundamentals)
    monkeypatch.setattr("civic_signal.scoring.backtest.MarketModel", FakeComponent)
    monkeypatch.setattr("civic_signal.scoring.backtest.PublicSignalModel", FakeComponent)
    monkeypatch.setattr("civic_signal.scoring.backtest.EnsembleModel", FakeEnsemble)

    _components, _ensemble, _config, posterior = runner.base._publication_components(
        train_bundle=bundle,
        test_bundle=bundle,
        as_of="2008-10-01",
        model_config=runner.context.read_yaml("model.yaml"),
        inference_engine="kalman",
    )
    assert posterior.height == 1
    assert observed_result_rows and set(observed_result_rows) == {0}
