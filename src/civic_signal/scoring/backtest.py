from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, ClassVar

import numpy as np
import polars as pl

from civic_signal.config import ProjectContext, Scenario, ScenarioRegistry
from civic_signal.features import (
    FeatureBuilder,
    FeatureBundle,
    filter_bundle_by_date,
    filter_results_before_cycle,
    subset_bundle,
)
from civic_signal.inference.hyperpriors import search_hyperpriors
from civic_signal.inference.recalibration import recalibration_map_from_calibration
from civic_signal.models import (
    EnsembleModel,
    FundamentalsModel,
    MarketModel,
    PollingModel,
    PublicSignalModel,
)
from civic_signal.models.common import clamp, normal_cdf
from civic_signal.models.polling import resolve_inference_engine
from civic_signal.scoring.learning import (
    apply_platt_calibration,
    fit_platt_calibration,
    fit_simplex_weights,
    stacked_probability,
)
from civic_signal.scoring.metrics import score_predictions
from civic_signal.storage.io import write_json, write_parquet


@dataclass(frozen=True)
class BacktestArtifacts:
    payload: dict[str, Any]
    rolling_predictions: pl.DataFrame
    component_admission: dict[str, Any]
    residual_covariance: pl.DataFrame
    recalibration_map: pl.DataFrame


class BacktestRunner:
    BASE_COMPONENT_COLUMNS: ClassVar[dict[str, str]] = {
        "baseline": "baseline_probability",
        "polling": "polls_probability",
        "fundamentals": "fundamentals_probability",
        "markets": "markets_probability",
        "public_signals": "public_signals_probability",
        "ensemble": "ensemble_probability",
    }
    STACK_COMPONENT_COLUMNS: ClassVar[dict[str, str]] = {
        "polling": "polls_probability",
        "fundamentals": "fundamentals_probability",
        "markets": "markets_probability",
        "public_signals": "public_signals_probability",
    }
    COMPONENT_COLUMNS: ClassVar[dict[str, str]] = {
        "baseline": "baseline_probability",
        "polling": "polls_probability",
        "fundamentals": "fundamentals_probability",
        "markets": "markets_probability",
        "public_signals": "public_signals_probability",
        "ensemble_configured": "configured_ensemble_probability",
        "ensemble_learned": "learned_ensemble_probability",
        "ensemble": "ensemble_probability",
    }

    def __init__(self, context: ProjectContext) -> None:
        self.context = context

    def evaluate(
        self,
        scenario: str | None = None,
        start_cycle: int | None = None,
        holdout_cycle: int | None = None,
        inference_engine: str | None = None,
        bayesian_backend: str | None = None,
    ) -> dict[str, object]:
        return self._evaluate(
            scenario,
            start_cycle,
            holdout_cycle,
            inference_engine,
            bayesian_backend,
        ).payload

    def _evaluate(
        self,
        scenario: str | None = None,
        start_cycle: int | None = None,
        holdout_cycle: int | None = None,
        inference_engine: str | None = None,
        bayesian_backend: str | None = None,
    ) -> BacktestArtifacts:
        bundle = FeatureBuilder(self.context).run()
        model_config = self.context.read_yaml("model.yaml")
        inference_engine = resolve_inference_engine(model_config, inference_engine)
        if inference_engine != "kalman":
            model_config = json.loads(json.dumps(model_config))
            model_config["_inference_engine"] = inference_engine
            if bayesian_backend:
                model_config["_bayesian_backend"] = bayesian_backend.lower().strip()
        resolved_bayesian_backend = (
            model_config.get("_bayesian_backend") or model_config.get("bayesian", {}).get("backend")
            if inference_engine == "bayes"
            else None
        )
        # Rolling-origin scoring does not need publication-grade posteriors;
        # apply the backtest sampler budget to every fold in this runner.
        sampler_overrides = dict(
            dict(dict(model_config.get("bayesian", {})).get("nuts", {})).get(
                "backtest_overrides", {}
            )
            or {}
        )
        if sampler_overrides and inference_engine == "bayes":
            model_config = json.loads(json.dumps(model_config))
            model_config["bayesian"]["nuts"].update(sampler_overrides)
        backtest_config = self.context.read_yaml("backtests.yaml")
        scenario_obj = ScenarioRegistry.from_context(self.context).get(scenario)
        rolling_predictions = self._rolling_origin_predictions(
            bundle=bundle,
            model_config=model_config,
            backtest_config=backtest_config,
            scenario=scenario_obj,
            start_cycle=start_cycle,
            holdout_cycle=holdout_cycle,
            inference_engine=inference_engine,
        )
        minimum_rows = int(backtest_config.get("minimum_rows_for_trust", 30))
        rolling = self._rolling_origin_summary(rolling_predictions)
        sample_size_too_small = rolling_predictions.height < minimum_rows
        initial_metrics = self._score_columns(rolling_predictions, self.BASE_COMPONENT_COLUMNS)
        initial_ablations = self._ablations(initial_metrics)
        trustworthy = bool(rolling["executed"]) and not sample_size_too_small
        trusted_components = self._trusted_components(
            initial_ablations,
            model_config,
            trustworthy=trustworthy,
        )
        ensemble_learning = self._fit_ensemble_learning(
            rolling_predictions,
            model_config,
            trusted_components,
            trustworthy=trustworthy,
            minimum_rows=minimum_rows,
        )
        rolling_predictions = self._with_learned_ensemble_columns(
            rolling_predictions,
            ensemble_learning,
        )
        metrics = self._score_columns(rolling_predictions, self.COMPONENT_COLUMNS)
        ablations = self._ablations(metrics)
        rolling = self._rolling_origin_summary(rolling_predictions)
        hyperprior_search = search_hyperpriors(rolling_predictions, model_config)
        payload: dict[str, Any] = {
            "generated_at": datetime.now(UTC).isoformat(),
            "method": "rolling_origin_component_refit",
            "inference_engine": inference_engine,
            "bayesian_backend": resolved_bayesian_backend,
            "scenario": scenario,
            "start_cycle": start_cycle,
            "holdout_cycle": holdout_cycle,
            "rolling_origin_executed": rolling["executed"],
            "rolling_origin": rolling,
            "minimum_rows_for_trust": minimum_rows,
            "sample_size_too_small": sample_size_too_small,
            "row_count": rolling_predictions.height,
            "metrics": metrics,
            "ablations": ablations,
            "ensemble_learning": ensemble_learning["weight_learning"],
            "probability_calibration": ensemble_learning["probability_calibration"],
            "bayesian_hyperpriors": hyperprior_search,
            "horizon_calibration": self._horizon_calibration(rolling_predictions),
        }
        recalibration_map = recalibration_map_from_calibration(
            ensemble_learning["probability_calibration"],
            cycles=rolling["cycles"],
            as_of_cuts=sorted(
                int(value)
                for value in rolling_predictions["as_of_offset_days"]
                .drop_nulls()
                .unique()
                .to_list()
            )
            if "as_of_offset_days" in rolling_predictions.columns
            else [],
        ).to_frame()
        component_admission = self._component_admission(
            payload=payload,
            ablations=ablations,
            model_config=model_config,
            scenario=scenario_obj,
            trusted_components=trusted_components,
            ensemble_learning=ensemble_learning,
        )
        covariance = self._residual_covariance(rolling_predictions, model_config)
        return BacktestArtifacts(
            payload,
            rolling_predictions,
            component_admission,
            covariance,
            recalibration_map,
        )

    def _rolling_origin_predictions(
        self,
        bundle: FeatureBundle,
        model_config: dict[str, Any],
        backtest_config: dict[str, Any],
        scenario: Scenario | None,
        start_cycle: int | None,
        holdout_cycle: int | None,
        inference_engine: str,
    ) -> pl.DataFrame:
        base_catalog = (
            scenario.filter_catalog(bundle.race_catalog, include_cycle=False)
            if scenario
            else bundle.race_catalog
        )
        target_catalog = base_catalog
        if start_cycle is not None:
            target_catalog = target_catalog.filter(pl.col("cycle") >= start_cycle)
        if holdout_cycle is not None:
            target_cycles = [holdout_cycle]
        else:
            target_cycles = sorted(
                int(value) for value in target_catalog["cycle"].unique().to_list()
            )
        fold_specs: list[dict[str, Any]] = []
        for target_cycle in target_cycles:
            train_catalog = base_catalog.filter(pl.col("cycle") < target_cycle)
            test_catalog = base_catalog.filter(pl.col("cycle") == target_cycle)
            train_catalog = self._restrict_to_era(train_catalog, test_catalog)
            if train_catalog.is_empty() or test_catalog.is_empty():
                continue
            offsets = self._as_of_offsets(backtest_config)
            for offset_days, as_of in self._cycle_as_of_dates(test_catalog, offsets):
                fold_specs.append(
                    {
                        "target_cycle": target_cycle,
                        "as_of": as_of,
                        "as_of_offset_days": offset_days,
                    }
                )
        if not fold_specs:
            return self._empty_predictions()
        parallel_folds = int(
            dict(backtest_config.get("rolling_origin", {})).get("parallel_folds", 1) or 1
        )
        frames: list[pl.DataFrame] = []
        if parallel_folds > 1 and len(fold_specs) > 1:
            frames = self._run_folds_parallel(
                fold_specs=fold_specs,
                scenario=scenario,
                model_config=model_config,
                inference_engine=inference_engine,
                max_workers=min(parallel_folds, len(fold_specs)),
            )
        else:
            for spec in fold_specs:
                predictions = _execute_backtest_fold(
                    {
                        "context": self.context,
                        "scenario": scenario,
                        "model_config": model_config,
                        "inference_engine": inference_engine,
                        **spec,
                    }
                )
                if not predictions.is_empty():
                    frames.append(predictions)
        return pl.concat(frames, how="diagonal_relaxed") if frames else self._empty_predictions()

    def _run_folds_parallel(
        self,
        *,
        fold_specs: list[dict[str, Any]],
        scenario: Scenario | None,
        model_config: dict[str, Any],
        inference_engine: str,
        max_workers: int,
    ) -> list[pl.DataFrame]:
        """Run rolling-origin folds across processes.

        Folds are independent refits, so this is embarrassingly parallel. The
        spawn context is mandatory: JAX runtimes are not fork-safe. Results are
        reassembled in fold order so artifacts stay byte-stable regardless of
        completion order.
        """
        import concurrent.futures
        import multiprocessing

        payloads = [
            {
                "context": self.context,
                "scenario": scenario,
                "model_config": model_config,
                "inference_engine": inference_engine,
                **spec,
            }
            for spec in fold_specs
        ]
        spawn = multiprocessing.get_context("spawn")
        results: list[pl.DataFrame | None] = [None] * len(payloads)
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=max_workers, mp_context=spawn
        ) as executor:
            futures = {
                executor.submit(_execute_backtest_fold, payload): index
                for index, payload in enumerate(payloads)
            }
            for future in concurrent.futures.as_completed(futures):
                results[futures[future]] = future.result()
        return [frame for frame in results if frame is not None and not frame.is_empty()]

    def _predict_cycle(
        self,
        train_bundle: FeatureBundle,
        test_bundle: FeatureBundle,
        target_cycle: int,
        as_of: str,
        as_of_offset_days: int,
        model_config: dict[str, Any],
        inference_engine: str,
    ) -> pl.DataFrame:
        train_bundle = filter_results_before_cycle(train_bundle, target_cycle)
        fundamentals_model = FundamentalsModel(model_config).fit(train_bundle)
        cycle_model_config = json.loads(json.dumps(model_config))
        if inference_engine == "bayes":
            from civic_signal.inference.fundamentals_prior import build_fundamentals_prior

            fundamentals_prior = build_fundamentals_prior(
                fundamentals_model, test_bundle, cycle_model_config
            )
            cycle_model_config["_fundamentals_prior_rows"] = fundamentals_prior.frame.to_dicts()
        component_estimates = [
            PollingModel(
                cycle_model_config,
                as_of=as_of,
                inference_engine=inference_engine,
            ).run(test_bundle),
            fundamentals_model.run(test_bundle),
            MarketModel(cycle_model_config).run(test_bundle),
            PublicSignalModel(
                trusted=bool(
                    cycle_model_config.get("trusted_components", {}).get("public_signals", False)
                )
            ).run(test_bundle),
        ]
        if all(frame.is_empty() for frame in component_estimates):
            return self._empty_predictions()
        ensemble = EnsembleModel(cycle_model_config).run(test_bundle, component_estimates)
        rows: list[dict[str, Any]] = []
        actuals = {
            (row["race_id"], row["option_id"]): row
            for row in test_bundle.results.iter_rows(named=True)
        }
        component_maps = {
            "polls_probability": self._component_probability(component_estimates[0]),
            "fundamentals_probability": self._component_probability(component_estimates[1]),
            "markets_probability": self._component_probability(component_estimates[2]),
            "public_signals_probability": self._component_probability(component_estimates[3]),
            "ensemble_probability": self._component_probability(ensemble),
        }
        ensemble_share = self._component_share(ensemble)
        ensemble_uncertainty = self._component_uncertainty(ensemble)
        catalog = {row["race_id"]: row for row in test_bundle.race_catalog.iter_rows(named=True)}
        baseline_sigma = self._baseline_sigma(train_bundle, model_config)
        for option in test_bundle.options.iter_rows(named=True):
            key = (option["race_id"], option["option_id"])
            actual = actuals.get(key)
            if actual is None:
                continue
            previous_share = float(option.get("previous_vote_share") or 0.5)
            uncertainty = ensemble_uncertainty.get(key, 0.08)
            predicted_share = ensemble_share.get(key, previous_share)
            race = catalog[str(option["race_id"])]
            row = {
                "race_id": option["race_id"],
                "cycle": target_cycle,
                "as_of": as_of,
                "as_of_offset_days": as_of_offset_days,
                "polling_inference_engine": inference_engine,
                "geography": race.get("geography"),
                "office_type": race.get("office_type"),
                "option_id": option["option_id"],
                "party": option.get("party"),
                "actual_winner": bool(actual["winner"]),
                "actual_vote_share": float(actual["vote_share"]),
                "baseline_probability": normal_cdf((previous_share - 0.5) / baseline_sigma),
                "baseline_sigma": baseline_sigma,
                "predicted_vote_share": predicted_share,
                "lower_90": clamp(predicted_share - 1.645 * uncertainty, 0.0, 1.0),
                "upper_90": clamp(predicted_share + 1.645 * uncertainty, 0.0, 1.0),
            }
            for column, values in component_maps.items():
                row[column] = values.get(key, row["baseline_probability"])
            rows.append(row)
        return pl.DataFrame(rows) if rows else self._empty_predictions()

    @staticmethod
    def _component_probability(frame: pl.DataFrame) -> dict[tuple[str, str], float]:
        if frame.is_empty() or "marginal_win_probability" not in frame.columns:
            return {}
        return {
            (str(row["race_id"]), str(row["option_id"])): float(row["marginal_win_probability"])
            for row in frame.iter_rows(named=True)
        }

    @staticmethod
    def _component_share(frame: pl.DataFrame) -> dict[tuple[str, str], float]:
        if frame.is_empty() or "vote_share" not in frame.columns:
            return {}
        return {
            (str(row["race_id"]), str(row["option_id"])): float(row["vote_share"])
            for row in frame.iter_rows(named=True)
        }

    @staticmethod
    def _component_uncertainty(frame: pl.DataFrame) -> dict[tuple[str, str], float]:
        if frame.is_empty() or "uncertainty" not in frame.columns:
            return {}
        return {
            (str(row["race_id"]), str(row["option_id"])): float(row["uncertainty"])
            for row in frame.iter_rows(named=True)
        }

    @staticmethod
    def _score_columns(
        frame: pl.DataFrame,
        component_columns: dict[str, str],
    ) -> dict[str, dict[str, float]]:
        return {
            component: score_predictions(frame, column)
            for component, column in component_columns.items()
            if column in frame.columns
        }

    @staticmethod
    def _ablations(metrics: dict[str, dict[str, float]]) -> dict[str, dict[str, Any]]:
        baseline_brier = metrics.get("baseline", {}).get("brier")
        if baseline_brier is None or not np.isfinite(baseline_brier):
            return {}
        ablations = {}
        for component, values in metrics.items():
            if component == "baseline":
                continue
            brier = values.get("brier")
            if brier is None or not np.isfinite(brier):
                continue
            ablations[component] = {
                "brier_delta_vs_baseline": brier - baseline_brier,
                "beats_or_matches_baseline": brier <= baseline_brier,
            }
        return ablations

    @staticmethod
    def _trusted_components(
        ablations: dict[str, dict[str, Any]],
        model_config: dict[str, Any],
        *,
        trustworthy: bool,
    ) -> dict[str, bool]:
        configured = {
            str(key): bool(value)
            for key, value in dict(model_config.get("trusted_components", {})).items()
        }
        trusted_components = {}
        for component, configured_trust in configured.items():
            if component == "public_signals":
                trusted_components[component] = False
                continue
            if trustworthy:
                trusted_components[component] = bool(
                    ablations.get(component, {}).get("beats_or_matches_baseline", False)
                )
            else:
                trusted_components[component] = configured_trust
        return trusted_components

    def _fit_ensemble_learning(
        self,
        frame: pl.DataFrame,
        model_config: dict[str, Any],
        trusted_components: dict[str, bool],
        *,
        trustworthy: bool,
        minimum_rows: int,
    ) -> dict[str, Any]:
        configured_weights = {
            component: float(dict(model_config.get("component_weights", {})).get(component, 0.0))
            for component in self.STACK_COMPONENT_COLUMNS
        }
        settings = dict(model_config.get("ensemble_learning", {}))
        if not trustworthy or not bool(settings.get("enabled", True)):
            reason = "disabled" if not bool(settings.get("enabled", True)) else "untrusted_backtest"
            weight_learning = {
                "status": reason,
                "method": "configured_fallback",
                "components": [],
                "component_weights": configured_weights,
                "configured_weights": configured_weights,
                "row_count": frame.height,
                "iterations": 0,
            }
            calibration = {
                "status": reason,
                "method": "platt_logistic_ridge",
                "intercept": 0.0,
                "slope": 1.0,
                "row_count": frame.height,
                "ridge": float(settings.get("calibration_ridge", 1e-3)),
                "min_slope": float(settings.get("calibration_min_slope", 0.25)),
                "max_slope": float(settings.get("calibration_max_slope", 1.0)),
                "max_abs_intercept": float(settings.get("calibration_max_abs_intercept", 2.0)),
            }
            return {
                "weight_learning": weight_learning,
                "probability_calibration": calibration,
            }

        weight_learning = fit_simplex_weights(
            frame,
            self.STACK_COMPONENT_COLUMNS,
            configured_weights,
            trusted_components,
            max_iterations=int(settings.get("max_iterations", 800)),
            learning_rate=float(settings.get("learning_rate", 0.35)),
            l2_prior_strength=float(settings.get("l2_prior_strength", 0.02)),
            min_rows=minimum_rows,
        )
        learned_probability = stacked_probability(
            frame,
            self.STACK_COMPONENT_COLUMNS,
            weight_learning["component_weights"],
        )
        calibration = fit_platt_calibration(
            learned_probability,
            frame["actual_winner"].cast(pl.Float64).to_numpy()
            if "actual_winner" in frame.columns
            else np.array([], dtype=np.float64),
            min_rows=minimum_rows,
            ridge=float(settings.get("calibration_ridge", 1e-3)),
            min_slope=float(settings.get("calibration_min_slope", 0.25)),
            max_slope=float(settings.get("calibration_max_slope", 1.0)),
            max_abs_intercept=float(settings.get("calibration_max_abs_intercept", 2.0)),
        )
        calibration["input_probability"] = "learned_ensemble_probability"
        return {
            "weight_learning": weight_learning,
            "probability_calibration": calibration,
        }

    def _with_learned_ensemble_columns(
        self,
        frame: pl.DataFrame,
        ensemble_learning: dict[str, Any],
    ) -> pl.DataFrame:
        if "ensemble_probability" in frame.columns:
            if "configured_ensemble_probability" in frame.columns:
                frame = frame.drop("configured_ensemble_probability")
            frame = frame.rename({"ensemble_probability": "configured_ensemble_probability"})
        else:
            frame = frame.with_columns(
                pl.lit(None, dtype=pl.Float64).alias("configured_ensemble_probability")
            )
        learned = stacked_probability(
            frame,
            self.STACK_COMPONENT_COLUMNS,
            ensemble_learning["weight_learning"]["component_weights"],
            fallback_column="configured_ensemble_probability",
        )
        calibrated = apply_platt_calibration(
            learned,
            ensemble_learning["probability_calibration"],
        )
        return frame.with_columns(
            pl.Series("learned_ensemble_probability", learned, dtype=pl.Float64),
            pl.Series("ensemble_probability", calibrated, dtype=pl.Float64),
        )

    @staticmethod
    def _restrict_to_era(train_catalog: pl.DataFrame, test_catalog: pl.DataFrame) -> pl.DataFrame:
        """Drop training rows from a different redistricting era than the holdout cycle.

        Non-house scenarios don't carry a `redistricting_era` column or have it null,
        which is treated as compatible with any era.
        """
        if (
            "redistricting_era" not in train_catalog.columns
            or "redistricting_era" not in test_catalog.columns
        ):
            return train_catalog
        eras = test_catalog["redistricting_era"].drop_nulls().unique().to_list()
        if not eras:
            return train_catalog
        return train_catalog.filter(
            pl.col("redistricting_era").is_null() | pl.col("redistricting_era").is_in(eras)
        )

    @staticmethod
    def _as_of_offsets(backtest_config: dict[str, Any]) -> list[int]:
        rolling = dict(backtest_config.get("rolling_origin", {}))
        offsets = rolling.get("as_of_offsets_days", [1])
        parsed = sorted({max(1, int(value)) for value in offsets}, reverse=True)
        return parsed or [1]

    @staticmethod
    def _cycle_as_of_dates(
        test_catalog: pl.DataFrame, offsets_days: list[int]
    ) -> list[tuple[int, str]]:
        election_date = test_catalog.select(pl.col("election_date").min()).item()
        if not hasattr(election_date, "isoformat"):
            election_date = datetime.fromisoformat(str(election_date)).date()
        return [
            (offset_days, (election_date - timedelta(days=offset_days)).isoformat())
            for offset_days in offsets_days
        ]

    @staticmethod
    def _baseline_sigma(train_bundle: FeatureBundle, model_config: dict[str, Any]) -> float:
        baseline = dict(model_config.get("baseline", {}))
        default_sigma = float(baseline.get("previous_share_sigma", 0.08))
        min_rows = int(baseline.get("empirical_min_rows", 20))
        min_sigma = float(baseline.get("empirical_min_sigma", 0.03))
        if train_bundle.results.is_empty() or train_bundle.options.is_empty():
            return max(default_sigma, min_sigma)
        joined = train_bundle.results.join(
            train_bundle.options.select(["race_id", "option_id", "previous_vote_share"]),
            on=["race_id", "option_id"],
            how="inner",
        ).with_columns(
            (pl.col("vote_share") - pl.col("previous_vote_share").fill_null(0.5)).alias(
                "baseline_residual"
            )
        )
        values = joined.select("baseline_residual").drop_nulls()
        if values.height < min_rows:
            return max(default_sigma, min_sigma)
        sigma = float(values["baseline_residual"].std() or default_sigma)
        return max(sigma, min_sigma)

    @staticmethod
    def _rolling_origin_summary(frame: pl.DataFrame) -> dict[str, Any]:
        if frame.is_empty() or "cycle" not in frame.columns:
            return {
                "executed": False,
                "method": "rolling_origin_component_refit",
                "reason": "no scored holdout cycles",
                "cycles": [],
                "per_cycle_metrics": {},
            }
        cycles = sorted(int(value) for value in frame["cycle"].unique().to_list())
        return {
            "executed": True,
            "method": "rolling_origin_component_refit",
            "cycles": cycles,
            "per_cycle_metrics": {
                str(cycle): score_predictions(
                    frame.filter(pl.col("cycle") == cycle), "ensemble_probability"
                )
                for cycle in cycles
            },
        }

    @staticmethod
    def _component_admission(
        payload: dict[str, Any],
        ablations: dict[str, dict[str, Any]],
        model_config: dict[str, Any],
        scenario: Scenario | None,
        trusted_components: dict[str, bool],
        ensemble_learning: dict[str, Any],
    ) -> dict[str, Any]:
        trustworthy = bool(payload["rolling_origin_executed"]) and not bool(
            payload["sample_size_too_small"]
        )
        weight_learning = ensemble_learning["weight_learning"]
        probability_calibration = ensemble_learning["probability_calibration"]
        learned_weights = dict(weight_learning.get("component_weights", {}))
        return {
            "generated_at": payload["generated_at"],
            "scenario": scenario.name if scenario else None,
            "scenario_family": scenario.family if scenario else None,
            "admission_status": "trusted" if trustworthy else "experimental_insufficient_rows",
            "engine_using": "learned_weights_calibrated" if trustworthy else "config_defaults",
            "trusted_components": trusted_components,
            "component_weights": learned_weights
            if trustworthy and learned_weights
            else dict(model_config.get("component_weights", {})),
            "configured_component_weights": dict(model_config.get("component_weights", {})),
            "ensemble_learning": weight_learning,
            "probability_calibration": probability_calibration,
            "bayesian_hyperpriors": payload.get("bayesian_hyperpriors", {}),
            "ablations": ablations,
            "minimum_rows_for_trust": payload["minimum_rows_for_trust"],
            "row_count": payload["row_count"],
        }

    @staticmethod
    def _residual_covariance(
        frame: pl.DataFrame, model_config: dict[str, Any] | None = None
    ) -> pl.DataFrame:
        schema = {
            "row_group": pl.Utf8,
            "column_group": pl.Utf8,
            "covariance": pl.Float64,
            "correlation": pl.Float64,
            "sample_size": pl.Int64,
            "shrinkage": pl.Float64,
            "matrix_rank": pl.Int64,
            "covariance_method": pl.Utf8,
        }
        if frame.is_empty():
            return pl.DataFrame(schema=schema)
        correlation_config = dict((model_config or {}).get("correlation", {}))
        geographic_groups = {
            str(key): str(value)
            for key, value in dict(correlation_config.get("geographic_groups", {})).items()
        }
        min_variance = float(correlation_config.get("residual_min_variance", 0.0004))
        # Cycles are the unit of independent evidence: as-of cuts within a cycle
        # share one election outcome, so they are averaged, never counted as
        # separate samples. With so few cycles a free covariance over ~50 groups
        # is unidentified; instead estimate a three-level factor structure
        # (national + region + state idiosyncratic) with pooled variances.
        residuals = (
            frame.with_columns(
                (pl.col("predicted_vote_share") - pl.col("actual_vote_share")).alias("residual"),
                # Aggregate district geographies (e.g. "AK-01") to their state so
                # the covariance groups match SimulationEngine's state-level
                # covariance lookup keys.
                pl.col("geography")
                .cast(pl.Utf8)
                .str.split("-")
                .list.first()
                .alias("_covariance_group"),
            )
            .group_by(["cycle", "_covariance_group"])
            .agg(pl.col("residual").mean().alias("residual"))
        )
        pivot = residuals.pivot(
            index=["cycle"],
            on="_covariance_group",
            values="residual",
            aggregate_function="mean",
        )
        groups = [column for column in pivot.columns if column != "cycle"]
        if not groups or pivot.height < 2:
            return pl.DataFrame(schema=schema)
        data = pivot.select(groups).fill_null(0.0).to_numpy().astype(np.float64)
        cycle_count = int(data.shape[0])
        region_of = {
            group: geographic_groups.get(str(group).split("-")[0], str(group)) for group in groups
        }
        regions = sorted(set(region_of.values()))
        region_columns = {
            region: [index for index, group in enumerate(groups) if region_of[group] == region]
            for region in regions
        }
        national = data.mean(axis=1)
        # A handful of cycles cannot rule out correlated national polling error,
        # and synthetic panels contain none by construction. Floor the national
        # factor at a historically grounded sd (~2pp national House poll misses
        # in 2016/2020) so seat distributions never claim it away.
        national_floor_sd = float(correlation_config.get("national_error_floor_sd", 0.02))
        national_variance = max(float(np.var(national, ddof=1)), national_floor_sd**2)
        region_deviations: list[float] = []
        state_deviations: list[float] = []
        region_means = np.zeros((cycle_count, len(regions)), dtype=np.float64)
        for region_index, region in enumerate(regions):
            columns = region_columns[region]
            region_means[:, region_index] = data[:, columns].mean(axis=1)
            region_deviations.extend((region_means[:, region_index] - national).tolist())
            for column in columns:
                state_deviations.extend((data[:, column] - region_means[:, region_index]).tolist())
        region_variance = max(
            float(np.var(np.array(region_deviations), ddof=1))
            if len(region_deviations) > 1
            else 0.0,
            0.0,
        )
        state_variance = max(
            float(np.var(np.array(state_deviations), ddof=1)) if len(state_deviations) > 1 else 0.0,
            min_variance * 0.25,
        )
        matrix_rows = []
        total_variance = max(national_variance + region_variance + state_variance, min_variance)
        matrix_rank = min(1 + len(regions) + len(groups), len(groups))
        for row_group in groups:
            for column_group in groups:
                covariance = national_variance
                if region_of[row_group] == region_of[column_group]:
                    covariance += region_variance
                if row_group == column_group:
                    covariance += state_variance
                    covariance = max(covariance, min_variance)
                correlation = covariance / total_variance
                matrix_rows.append(
                    {
                        "row_group": row_group,
                        "column_group": column_group,
                        "covariance": float(covariance),
                        "correlation": float(min(correlation, 1.0)),
                        "sample_size": cycle_count,
                        "shrinkage": 0.0,
                        "matrix_rank": matrix_rank,
                        "covariance_method": "structured_factor_cycle_level",
                    }
                )
        return pl.DataFrame(matrix_rows)

    @staticmethod
    def _horizon_calibration(frame: pl.DataFrame) -> dict[str, Any]:
        """Estimate forecast-error drift per sqrt(day) from rolling-origin cuts.

        Fits var(residual | horizon h) = base + drift^2 * h across the as-of
        offsets, giving an empirical replacement for the hand-set
        forecast_drift_sd_per_sqrt_day constant.
        """
        required = {"as_of_offset_days", "predicted_vote_share", "actual_vote_share"}
        if frame.is_empty() or not required.issubset(frame.columns):
            return {"status": "no_rows"}
        rows = (
            frame.drop_nulls(list(required))
            .with_columns(
                ((pl.col("predicted_vote_share") - pl.col("actual_vote_share")) ** 2).alias(
                    "_squared_residual"
                )
            )
            .group_by("as_of_offset_days")
            .agg(
                pl.col("_squared_residual").mean().alias("variance"),
                pl.len().alias("row_count"),
            )
            .sort("as_of_offset_days")
        )
        if rows.height < 2:
            return {"status": "insufficient_horizons", "horizon_count": rows.height}
        horizons = rows["as_of_offset_days"].cast(pl.Float64).to_numpy()
        variances = rows["variance"].to_numpy()
        design = np.column_stack([np.ones_like(horizons), horizons])
        coefficients, *_ = np.linalg.lstsq(design, variances, rcond=None)
        slope = float(coefficients[1])
        drift_share = math.sqrt(max(slope, 0.0))
        # Share-scale sd converts to logit scale by 1/(p(1-p)) ~= 4 near p=0.5.
        drift_logit = drift_share * 4.0
        return {
            "status": "fitted" if slope > 0 else "no_positive_drift",
            "drift_sd_share_per_sqrt_day": drift_share,
            "drift_sd_logit_per_sqrt_day": drift_logit,
            "base_variance_share": float(max(coefficients[0], 0.0)),
            "horizons_days": [int(value) for value in horizons.tolist()],
            "horizon_variances_share": [float(value) for value in variances.tolist()],
            "row_counts": rows["row_count"].to_list(),
        }

    @staticmethod
    def _structured_covariance_target(
        groups: list[str],
        diagonal_variance: np.ndarray,
        geographic_groups: dict[str, str],
        same_region_corr: float,
        cross_region_corr: float,
    ) -> np.ndarray:
        target = np.zeros((len(groups), len(groups)), dtype=np.float64)
        for row_index, row_group in enumerate(groups):
            row_region = geographic_groups.get(str(row_group).split("-")[0], str(row_group))
            for column_index, column_group in enumerate(groups):
                column_region = geographic_groups.get(
                    str(column_group).split("-")[0], str(column_group)
                )
                if row_index == column_index:
                    corr = 1.0
                elif row_region == column_region:
                    corr = same_region_corr
                else:
                    corr = cross_region_corr
                target[row_index, column_index] = (
                    corr
                    * float(np.sqrt(diagonal_variance[row_index]))
                    * float(np.sqrt(diagonal_variance[column_index]))
                )
        return target

    @staticmethod
    def _nearest_psd(matrix: np.ndarray, min_variance: float) -> np.ndarray:
        symmetric = (matrix + matrix.T) / 2.0
        eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
        clipped = np.maximum(eigenvalues, min_variance)
        psd = (eigenvectors * clipped) @ eigenvectors.T
        return (psd + psd.T) / 2.0

    @staticmethod
    def _empty_predictions() -> pl.DataFrame:
        return pl.DataFrame(
            schema={
                "race_id": pl.Utf8,
                "cycle": pl.Int64,
                "as_of": pl.Utf8,
                "as_of_offset_days": pl.Int64,
                "polling_inference_engine": pl.Utf8,
                "geography": pl.Utf8,
                "office_type": pl.Utf8,
                "option_id": pl.Utf8,
                "party": pl.Utf8,
                "actual_winner": pl.Boolean,
                "actual_vote_share": pl.Float64,
                "baseline_probability": pl.Float64,
                "baseline_sigma": pl.Float64,
                "polls_probability": pl.Float64,
                "fundamentals_probability": pl.Float64,
                "markets_probability": pl.Float64,
                "public_signals_probability": pl.Float64,
                "configured_ensemble_probability": pl.Float64,
                "learned_ensemble_probability": pl.Float64,
                "ensemble_probability": pl.Float64,
                "predicted_vote_share": pl.Float64,
                "lower_90": pl.Float64,
                "upper_90": pl.Float64,
            }
        )

    def run(
        self,
        run_id: str,
        scenario: str | None = None,
        start_cycle: int | None = None,
        holdout_cycle: int | None = None,
        inference_engine: str | None = None,
        bayesian_backend: str | None = None,
    ) -> dict[str, object]:
        artifacts = self._evaluate(
            scenario,
            start_cycle,
            holdout_cycle,
            inference_engine,
            bayesian_backend,
        )
        out_dir = self.context.artifacts_dir / "backtests" / run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        metrics_rows = [
            {"component": component, **values}
            for component, values in artifacts.payload["metrics"].items()
        ]
        write_parquet(pl.DataFrame(metrics_rows), out_dir / "scorecard.parquet")
        write_json(artifacts.payload, out_dir / "scorecard.json")
        write_parquet(artifacts.rolling_predictions, out_dir / "rolling_predictions.parquet")
        write_json(artifacts.component_admission, out_dir / "component_admission.json")
        write_json(artifacts.payload["ensemble_learning"], out_dir / "ensemble_learning.json")
        write_json(
            artifacts.payload["probability_calibration"],
            out_dir / "probability_calibration.json",
        )
        write_json(artifacts.payload["bayesian_hyperpriors"], out_dir / "bayesian_hyperpriors.json")
        write_parquet(artifacts.recalibration_map, out_dir / "recalibration_map.parquet")
        write_parquet(artifacts.residual_covariance, out_dir / "residual_covariance.parquet")
        self._write_latest_artifacts(
            scenario=scenario,
            component_admission=artifacts.component_admission,
            residual_covariance=artifacts.residual_covariance,
            recalibration_map=artifacts.recalibration_map,
        )
        return artifacts.payload

    def _write_latest_artifacts(
        self,
        scenario: str | None,
        component_admission: dict[str, Any],
        residual_covariance: pl.DataFrame,
        recalibration_map: pl.DataFrame,
    ) -> None:
        key = component_admission.get("scenario_family") or scenario or "default"
        latest_dir = self.context.artifacts_dir / "backtests" / "latest"
        write_json(component_admission, latest_dir / f"component_admission_{key}.json")
        write_json(
            component_admission.get("ensemble_learning", {}),
            latest_dir / f"ensemble_learning_{key}.json",
        )
        write_json(
            component_admission.get("probability_calibration", {}),
            latest_dir / f"probability_calibration_{key}.json",
        )
        write_json(
            component_admission.get("bayesian_hyperpriors", {}),
            latest_dir / f"bayesian_hyperpriors_{key}.json",
        )
        write_parquet(recalibration_map, latest_dir / f"recalibration_map_{key}.parquet")
        write_parquet(residual_covariance, latest_dir / f"residual_covariance_{key}.parquet")
        index_path = latest_dir / "index.json"
        index = {}
        if index_path.exists():
            with index_path.open("r", encoding="utf-8") as handle:
                loaded = json.load(handle)
                if isinstance(loaded, dict):
                    index = loaded
        index[str(key)] = {
            "component_admission": f"component_admission_{key}.json",
            "ensemble_learning": f"ensemble_learning_{key}.json",
            "probability_calibration": f"probability_calibration_{key}.json",
            "bayesian_hyperpriors": f"bayesian_hyperpriors_{key}.json",
            "recalibration_map": f"recalibration_map_{key}.parquet",
            "residual_covariance": f"residual_covariance_{key}.parquet",
        }
        write_json(index, index_path)


# --- Fold worker (module level so it is picklable by the spawn context) ---

_WORKER_BUNDLE_CACHE: dict[tuple[str, str, str], FeatureBundle] = {}


def _worker_feature_bundle(context: ProjectContext) -> FeatureBundle:
    """Build (and per-process cache) the feature bundle from curated parquet."""
    key = (str(context.root), str(context.sources_config), str(context.data_dir))
    bundle = _WORKER_BUNDLE_CACHE.get(key)
    if bundle is None:
        bundle = FeatureBuilder(context).run()
        _WORKER_BUNDLE_CACHE[key] = bundle
    return bundle


def _execute_backtest_fold(payload: dict[str, Any]) -> pl.DataFrame:
    """Run one rolling-origin fold; usable serially or in a worker process."""
    context: ProjectContext = payload["context"]
    scenario: Scenario | None = payload["scenario"]
    target_cycle = int(payload["target_cycle"])
    as_of = str(payload["as_of"])
    bundle = _worker_feature_bundle(context)
    base_catalog = (
        scenario.filter_catalog(bundle.race_catalog, include_cycle=False)
        if scenario
        else bundle.race_catalog
    )
    train_catalog = base_catalog.filter(pl.col("cycle") < target_cycle)
    test_catalog = base_catalog.filter(pl.col("cycle") == target_cycle)
    train_catalog = BacktestRunner._restrict_to_era(train_catalog, test_catalog)
    if train_catalog.is_empty() or test_catalog.is_empty():
        return BacktestRunner._empty_predictions()
    train_bundle = filter_bundle_by_date(subset_bundle(bundle, train_catalog), as_of)
    test_bundle = filter_bundle_by_date(subset_bundle(bundle, test_catalog), as_of)
    return BacktestRunner(context)._predict_cycle(
        train_bundle=train_bundle,
        test_bundle=test_bundle,
        target_cycle=target_cycle,
        as_of=as_of,
        as_of_offset_days=int(payload["as_of_offset_days"]),
        model_config=payload["model_config"],
        inference_engine=str(payload["inference_engine"]),
    )
