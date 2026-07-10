from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, replace
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
from civic_signal.models.common import clamp, normal_cdf
from civic_signal.models.ensemble import EnsembleModel
from civic_signal.models.fundamentals import FundamentalsModel
from civic_signal.models.markets import MarketModel
from civic_signal.models.polling import PollingModel, resolve_inference_engine
from civic_signal.models.public_signals import PublicSignalModel
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
        component_estimates, ensemble, _cycle_model_config, _posterior_draws = (
            self._publication_components(
                train_bundle=train_bundle,
                test_bundle=test_bundle,
                as_of=as_of,
                model_config=model_config,
                inference_engine=inference_engine,
            )
        )
        if all(frame.is_empty() for frame in component_estimates):
            return self._empty_predictions()
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
    def _publication_components(
        *,
        train_bundle: FeatureBundle,
        test_bundle: FeatureBundle,
        as_of: str,
        model_config: dict[str, Any],
        inference_engine: str,
    ) -> tuple[list[pl.DataFrame], pl.DataFrame, dict[str, Any], pl.DataFrame]:
        """Fit the same components and ensemble used by the publication pipeline."""
        fundamentals_model = FundamentalsModel(model_config).fit(train_bundle)
        cycle_model_config = json.loads(json.dumps(model_config))
        model_bundle = replace(test_bundle, results=test_bundle.results.head(0))
        if inference_engine == "bayes":
            from civic_signal.inference.fundamentals_prior import build_fundamentals_prior

            fundamentals_prior = build_fundamentals_prior(
                fundamentals_model, model_bundle, cycle_model_config
            )
            cycle_model_config["_fundamentals_prior_rows"] = fundamentals_prior.frame.to_dicts()
        polling_model = PollingModel(
            cycle_model_config,
            as_of=as_of,
            inference_engine=inference_engine,
        )
        polling_estimates = polling_model.run(model_bundle)
        posterior_draws = polling_model.posterior_draws(model_bundle)
        component_estimates = [
            polling_estimates,
            fundamentals_model.run(model_bundle),
            MarketModel(cycle_model_config).run(model_bundle),
            PublicSignalModel(
                trusted=bool(
                    cycle_model_config.get("trusted_components", {}).get("public_signals", False)
                )
            ).run(model_bundle),
        ]
        ensemble = EnsembleModel(cycle_model_config).run(model_bundle, component_estimates)
        return component_estimates, ensemble, cycle_model_config, posterior_draws

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
            "factor_rank": pl.Int64,
            "covariance_method": pl.Utf8,
            "residual_definition": pl.Utf8,
            "representation": pl.Utf8,
            "factor_loadings_json": pl.Utf8,
            "factor_variances_json": pl.Utf8,
            "idiosyncratic_variance": pl.Float64,
            "configured_factor_rank": pl.Int64,
            "psd_constructed": pl.Boolean,
            "minimum_eigenvalue": pl.Float64,
        }
        if frame.is_empty():
            return pl.DataFrame(schema=schema)
        correlation_config = dict((model_config or {}).get("correlation", {}))
        geographic_groups = {
            str(key): str(value)
            for key, value in dict(correlation_config.get("geographic_groups", {})).items()
        }
        min_variance = float(correlation_config.get("residual_min_variance", 0.0004))
        shrinkage = min(
            max(float(correlation_config.get("residual_covariance_shrinkage", 0.0)), 0.0),
            1.0,
        )
        configured_factor_rank = max(int(correlation_config.get("residual_factor_rank", 64)), 1)
        # Cycles are the unit of independent evidence: as-of cuts within a cycle
        # share one election outcome, so they are averaged, never counted as
        # separate samples. With so few cycles a free covariance over ~50 groups
        # is unidentified; instead estimate a three-level factor structure
        # (national + region + state idiosyncratic) with pooled variances.
        residual_rows = frame.with_columns(
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
        residual_definition = "legacy_group_mean_without_race_identity"
        if {"race_id", "option_id"}.issubset(residual_rows.columns):
            # Candidate complements are not independent errors and averaging D/R
            # residuals drives a two-party race toward zero by construction. Use
            # one consistently signed reference option per race and average only
            # repeated forecast horizons for that reference option.
            party_rank = (
                pl.when(pl.col("party").cast(pl.Utf8).str.to_uppercase().is_in(["DEM", "D"]))
                .then(0)
                .otherwise(1)
                if "party" in residual_rows.columns
                else pl.when(
                    pl.col("option_id").cast(pl.Utf8).str.to_uppercase().str.contains(r"-(D|DEM)$")
                )
                .then(0)
                .otherwise(1)
            )
            horizon_keys = ["cycle", "race_id"]
            for horizon in ("as_of_offset_days", "as_of"):
                if horizon in residual_rows.columns:
                    horizon_keys.append(horizon)
            residual_rows = (
                residual_rows.with_columns(party_rank.alias("_reference_rank"))
                .sort([*horizon_keys, "_reference_rank", "option_id"])
                .group_by(horizon_keys, maintain_order=True)
                .agg(
                    pl.col("residual").first(),
                    pl.col("_covariance_group").first(),
                )
                .group_by(["cycle", "race_id", "_covariance_group"])
                .agg(pl.col("residual").mean())
            )
            residual_definition = "one_reference_option_per_race_cycle"
        residuals = residual_rows.group_by(["cycle", "_covariance_group"]).agg(
            pl.col("residual").mean().alias("residual")
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
        empirical_national_variance = max(float(np.var(national, ddof=1)), 0.0)
        national_variance = max(
            (1.0 - shrinkage) * empirical_national_variance + shrinkage * national_floor_sd**2,
            national_floor_sd**2,
        )
        region_deviations: list[float] = []
        state_deviations: list[float] = []
        region_means = np.zeros((cycle_count, len(regions)), dtype=np.float64)
        for region_index, region in enumerate(regions):
            columns = region_columns[region]
            region_means[:, region_index] = data[:, columns].mean(axis=1)
            region_deviations.extend((region_means[:, region_index] - national).tolist())
            for column in columns:
                state_deviations.extend((data[:, column] - region_means[:, region_index]).tolist())
        empirical_region_variance = max(
            float(np.var(np.array(region_deviations), ddof=1))
            if len(region_deviations) > 1
            else 0.0,
            0.0,
        )
        region_target_variance = max(float(correlation_config.get("region_sigma", 0.01)) ** 2, 0.0)
        region_variance = max(
            (1.0 - shrinkage) * empirical_region_variance + shrinkage * region_target_variance,
            0.0,
        )
        empirical_state_variance = max(
            float(np.var(np.array(state_deviations), ddof=1)) if len(state_deviations) > 1 else 0.0,
            0.0,
        )
        state_target_variance = min_variance * 0.25
        state_variance = max(
            (1.0 - shrinkage) * empirical_state_variance + shrinkage * state_target_variance,
            state_target_variance,
        )

        # Explicit PSD representation: Sigma = B diag(v) B' + diag(d).
        # The national factor is always retained. Regional factors are ordered
        # deterministically and truncated to the configured rank; omitted
        # regional variance is folded into the diagonal so marginal variance
        # remains conservative.
        retained_regions = sorted(regions)[: max(configured_factor_rank - 1, 0)]
        factor_variances = {"national": float(national_variance)}
        factor_variances.update(
            {f"region:{region}": float(region_variance) for region in retained_regions}
        )
        loadings_by_group: dict[str, dict[str, float]] = {}
        idiosyncratic_by_group: dict[str, float] = {}
        for group in groups:
            region = region_of[group]
            loadings = {"national": 1.0}
            if region in retained_regions:
                loadings[f"region:{region}"] = 1.0
            loadings_by_group[group] = loadings
            idiosyncratic_by_group[group] = float(
                state_variance + (region_variance if region not in retained_regions else 0.0)
            )

        covariance_matrix = np.zeros((len(groups), len(groups)), dtype=np.float64)
        for row_index, row_group in enumerate(groups):
            for column_index, column_group in enumerate(groups):
                shared = set(loadings_by_group[row_group]) & set(loadings_by_group[column_group])
                covariance_matrix[row_index, column_index] = sum(
                    loadings_by_group[row_group][factor]
                    * loadings_by_group[column_group][factor]
                    * factor_variances[factor]
                    for factor in shared
                )
                if row_index == column_index:
                    covariance_matrix[row_index, column_index] += idiosyncratic_by_group[row_group]
        covariance_matrix = (covariance_matrix + covariance_matrix.T) / 2.0
        minimum_eigenvalue = float(np.linalg.eigvalsh(covariance_matrix).min())
        matrix_rank = int(np.linalg.matrix_rank(covariance_matrix))
        factor_rank = len(factor_variances)
        factor_variances_json = json.dumps(factor_variances, sort_keys=True)
        matrix_rows = []
        for row_group in groups:
            for column_group in groups:
                row_index = groups.index(row_group)
                column_index = groups.index(column_group)
                covariance = float(covariance_matrix[row_index, column_index])
                denominator = math.sqrt(
                    float(covariance_matrix[row_index, row_index])
                    * float(covariance_matrix[column_index, column_index])
                )
                correlation = covariance / max(denominator, 1e-12)
                matrix_rows.append(
                    {
                        "row_group": row_group,
                        "column_group": column_group,
                        "covariance": float(covariance),
                        "correlation": float(min(correlation, 1.0)),
                        "sample_size": cycle_count,
                        "shrinkage": shrinkage,
                        "matrix_rank": matrix_rank,
                        "factor_rank": factor_rank,
                        "covariance_method": "low_rank_factor_plus_diagonal_cycle_level",
                        "residual_definition": residual_definition,
                        "representation": "B_diag_v_Bt_plus_diag_d",
                        "factor_loadings_json": json.dumps(
                            loadings_by_group[row_group], sort_keys=True
                        ),
                        "factor_variances_json": factor_variances_json,
                        "idiosyncratic_variance": idiosyncratic_by_group[row_group],
                        "configured_factor_rank": configured_factor_rank,
                        "psd_constructed": True,
                        "minimum_eigenvalue": minimum_eigenvalue,
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


class NestedBacktestRunner:
    """Cycle-nested evaluation with inner-only tuning and publication-path outer scoring."""

    BASELINE_COLUMNS: ClassVar[dict[str, str]] = {
        "prior_only": "prior_only_probability",
        "previous_cycle_swing": "previous_cycle_swing_probability",
        "fundamentals_only": "fundamentals_only_probability",
        "poll_average": "poll_average_probability",
        "markets_if_present": "market_implied_probability",
    }
    BASELINE_DEFINITIONS: ClassVar[dict[str, str]] = {
        "prior_only": "Uniform probability across the options in each held-out race.",
        "previous_cycle_swing": (
            "Previous option vote share mapped through the training-only baseline share scale."
        ),
        "fundamentals_only": "Training-fitted fundamentals component with no polling/market blend.",
        "poll_average": (
            "Arithmetic mean of eligible held-out poll shares, normalized within race."
        ),
        "markets_if_present": (
            "Eligible public-market component probability; absent rows are not imputed."
        ),
    }

    def __init__(self, context: ProjectContext) -> None:
        self.context = context
        self.base = BacktestRunner(context)

    def run(
        self,
        run_id: str,
        *,
        scenario: str | None = None,
        start_cycle: int | None = None,
        holdout_cycle: int | None = None,
        inference_engine: str | None = None,
        bayesian_backend: str | None = None,
    ) -> dict[str, Any]:
        bundle = FeatureBuilder(self.context).run()
        model_config = self.context.read_yaml("model.yaml")
        engine = resolve_inference_engine(model_config, inference_engine)
        if engine == "bayes":
            model_config = json.loads(json.dumps(model_config))
            model_config["_inference_engine"] = engine
            if bayesian_backend:
                model_config["_bayesian_backend"] = bayesian_backend.lower().strip()
        backtest_config = self.context.read_yaml("backtests.yaml")
        scenario_obj = ScenarioRegistry.from_context(self.context).get(scenario)
        catalog = (
            scenario_obj.filter_catalog(bundle.race_catalog, include_cycle=False)
            if scenario_obj
            else bundle.race_catalog
        )
        cycles = sorted(int(value) for value in catalog["cycle"].unique().to_list())
        outer_cycles = [
            cycle
            for cycle in cycles
            if (start_cycle is None or cycle >= start_cycle)
            and (holdout_cycle is None or cycle == holdout_cycle)
            and len([prior for prior in cycles if prior < cycle]) >= 2
        ]
        fold_rows: list[dict[str, Any]] = []
        outer_frames: list[pl.DataFrame] = []
        for outer_cycle in outer_cycles:
            inner_cycles = [
                cycle
                for cycle in cycles
                if cycle < outer_cycle and any(prior < cycle for prior in cycles)
            ]
            inner_frames = [
                self.base._rolling_origin_predictions(
                    bundle=bundle,
                    model_config=model_config,
                    backtest_config=backtest_config,
                    scenario=scenario_obj,
                    start_cycle=None,
                    holdout_cycle=inner_cycle,
                    inference_engine=engine,
                )
                for inner_cycle in inner_cycles
            ]
            nonempty_inner = [frame for frame in inner_frames if not frame.is_empty()]
            if nonempty_inner:
                inner_predictions = pl.concat(nonempty_inner, how="diagonal_relaxed")
            else:
                inner_predictions = self.base._empty_predictions()
            learning = self._fit_inner(inner_predictions, model_config, backtest_config)
            outer_config = json.loads(json.dumps(model_config))
            outer_config["component_weights"] = dict(
                learning["ensemble_learning"]["weight_learning"]["component_weights"]
            )
            outer_config["probability_calibration"] = dict(
                learning["ensemble_learning"]["probability_calibration"]
            )
            outer_config = self._apply_inner_hyperparameters(
                outer_config, learning["hyperparameters"]
            )
            test_catalog = catalog.filter(pl.col("cycle") == outer_cycle)
            train_catalog = BacktestRunner._restrict_to_era(
                catalog.filter(pl.col("cycle") < outer_cycle), test_catalog
            )
            fold_predictions = []
            path_checks: list[dict[str, bool]] = []
            for offset_days, as_of in self.base._cycle_as_of_dates(
                test_catalog, self.base._as_of_offsets(backtest_config)
            ):
                frame, path_status = self._publication_outer_fold(
                    bundle=bundle,
                    train_catalog=train_catalog,
                    test_catalog=test_catalog,
                    target_cycle=outer_cycle,
                    as_of=as_of,
                    as_of_offset_days=offset_days,
                    model_config=outer_config,
                    inference_engine=engine,
                    residual_covariance=learning["residual_covariance"],
                    holdovers=(scenario_obj.holdover_caucus_seats if scenario_obj else None),
                )
                path_checks.append(path_status)
                if not frame.is_empty():
                    fold_predictions.append(frame)
            outer = (
                pl.concat(fold_predictions, how="diagonal_relaxed")
                if fold_predictions
                else self.base._empty_predictions()
            )
            if not outer.is_empty():
                outer_frames.append(outer)
            train_cycles = sorted(int(value) for value in train_catalog["cycle"].unique().to_list())
            canary = self._held_out_permutation_canary(bundle, catalog, outer_cycle)
            simulation_engine_used = bool(path_checks) and all(
                check.get("simulation_engine_used") is True for check in path_checks
            )
            posterior_draw_path_complete = bool(path_checks) and all(
                check.get("posterior_draw_path_complete") is True for check in path_checks
            )
            fold_rows.append(
                {
                    "outer_cycle": outer_cycle,
                    "train_cycles": json.dumps(train_cycles),
                    "inner_validation_cycles": json.dumps(inner_cycles),
                    "fit_cycle_max": max(inner_cycles) if inner_cycles else None,
                    "outer_cycle_excluded": outer_cycle not in train_cycles
                    and outer_cycle not in inner_cycles,
                    "inner_row_count": inner_predictions.height,
                    "outer_row_count": outer.height,
                    "weights_status": learning["ensemble_learning"]["weight_learning"].get(
                        "status"
                    ),
                    "calibration_status": learning["ensemble_learning"][
                        "probability_calibration"
                    ].get("status"),
                    "hyperparameter_status": learning["hyperparameters"].get("status"),
                    "selected_hyperparameters": json.dumps(
                        learning["hyperparameters"].get("selected", {}), sort_keys=True
                    ),
                    "publication_path": "components+ensemble+SimulationEngine",
                    "simulation_engine_used": simulation_engine_used,
                    "posterior_draw_path_complete": posterior_draw_path_complete,
                    "training_lineage_sha256": self._training_lineage_hash(bundle, train_catalog),
                    "held_out_permutation_affects_training": not canary,
                    "held_out_permutation_canary_passed": canary,
                }
            )
        predictions = (
            pl.concat(outer_frames, how="diagonal_relaxed")
            if outer_frames
            else self.base._empty_predictions()
        )
        manifest = pl.DataFrame(fold_rows) if fold_rows else self._empty_manifest()
        all_excluded = bool(fold_rows) and all(row["outer_cycle_excluded"] for row in fold_rows)
        all_canaries = bool(fold_rows) and all(
            row["held_out_permutation_canary_passed"] for row in fold_rows
        )
        weights_fitted = bool(fold_rows) and all(
            row["weights_status"] == "fitted" for row in fold_rows
        )
        calibration_fitted = bool(fold_rows) and all(
            row["calibration_status"] == "fitted" for row in fold_rows
        )
        hyperparameters_fitted = bool(fold_rows) and all(
            row["hyperparameter_status"] == "fitted" for row in fold_rows
        )
        exact_pipeline = bool(fold_rows) and all(
            row["simulation_engine_used"] is True and row["posterior_draw_path_complete"] is True
            for row in fold_rows
        )
        metrics = self.base._score_columns(predictions, self.base.COMPONENT_COLUMNS)
        baseline_metrics = self._baseline_metrics(predictions)
        clustered_uncertainty = self._paired_cycle_clustered_uncertainty(
            predictions,
            config=dict(backtest_config.get("nested_evaluation") or {}),
        )
        fold_lineage = [
            {
                "outer_cycle": row["outer_cycle"],
                "train_cycles": json.loads(row["train_cycles"]),
                "inner_validation_cycles": json.loads(row["inner_validation_cycles"]),
                "fit_cycle_max": row["fit_cycle_max"],
                "outer_cycle_excluded": row["outer_cycle_excluded"],
                "training_lineage_sha256": row["training_lineage_sha256"],
                "simulation_engine_used": row["simulation_engine_used"],
                "posterior_draw_path_complete": row["posterior_draw_path_complete"],
                "held_out_permutation_canary_passed": row["held_out_permutation_canary_passed"],
                "held_out_permutation_affects_training": row[
                    "held_out_permutation_affects_training"
                ],
            }
            for row in fold_rows
        ]
        training_lineage_sha256 = {
            str(row["outer_cycle"]): row["training_lineage_sha256"] for row in fold_rows
        }
        held_out_permutation_canary = {
            "passed": all_canaries if fold_rows else False,
            "affects_prior_folds": (not all_canaries) if fold_rows else None,
            "by_outer_cycle": {
                str(row["outer_cycle"]): {
                    "passed": row["held_out_permutation_canary_passed"],
                    "training_lineage_sha256": row["training_lineage_sha256"],
                }
                for row in fold_rows
            },
        }
        payload: dict[str, Any] = {
            "generated_at": datetime.now(UTC).isoformat(),
            "method": "outer_cycle_nested_exact_publication_path",
            "scenario": scenario,
            "inference_engine": engine,
            "bayesian_backend": model_config.get("_bayesian_backend")
            if engine == "bayes"
            else None,
            "outer_fold": True,
            "fold_count": len(fold_rows),
            "independent_cycle_count": len(fold_rows),
            "outer_cycle_excluded": all_excluded,
            "held_out_permutation_affects_prior_folds": ((not all_canaries) if fold_rows else None),
            "held_out_permutation_canary": held_out_permutation_canary,
            "inner_folds_fit_hyperparameters": hyperparameters_fitted,
            "inner_folds_fit_weights": weights_fitted,
            "calibration_map_outer_fold": calibration_fitted,
            "inner_tuning_complete": weights_fitted
            and calibration_fitted
            and hyperparameters_fitted,
            # Only true when every outer as-of cut actually ran SimulationEngine
            # (and Bayes posterior draws when required). Never claim pass otherwise.
            "exact_pipeline": exact_pipeline,
            "exact_pipeline_scope": (
                "components+ensemble+SimulationEngine" if exact_pipeline else None
            ),
            "training_lineage_sha256": training_lineage_sha256,
            "fold_lineage": fold_lineage,
            "row_count": predictions.height,
            "metrics": metrics,
            "baseline_definitions": self.BASELINE_DEFINITIONS,
            "baseline_metrics": baseline_metrics,
            "paired_cycle_clustered_uncertainty": clustered_uncertainty,
            "promoted_training_bundle_compatibility": {
                "status": "insufficient_evidence",
                "reason": "no promoted real-data training-bundle registry is available",
            },
            "result_derived_feature_canary": {
                "status": "insufficient_evidence",
                "outcome_lineage_permutation_passed": all_canaries,
                "feature_injection_test_executed": False,
            },
            "fold_manifest": "fold_manifest.parquet",
            "predictions": "nested_predictions.parquet",
            "baseline_scorecard": "baseline_scorecard.json",
            "clustered_uncertainty": "paired_cycle_clustered_uncertainty.json",
        }
        out_dir = self.context.artifacts_dir / "backtests" / run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        write_parquet(manifest, out_dir / "fold_manifest.parquet")
        write_parquet(predictions, out_dir / "nested_predictions.parquet")
        write_json(
            {
                "definitions": self.BASELINE_DEFINITIONS,
                "metrics": baseline_metrics,
            },
            out_dir / "baseline_scorecard.json",
        )
        write_json(clustered_uncertainty, out_dir / "paired_cycle_clustered_uncertainty.json")
        write_json(payload, out_dir / "nested_evaluation.json")
        return payload

    def _fit_inner(
        self,
        predictions: pl.DataFrame,
        model_config: dict[str, Any],
        backtest_config: dict[str, Any],
    ) -> dict[str, Any]:
        minimum = int(backtest_config.get("minimum_rows_for_trust", 30))
        metrics = self.base._score_columns(predictions, self.base.BASE_COMPONENT_COLUMNS)
        ablations = self.base._ablations(metrics)
        trustworthy = not predictions.is_empty() and predictions.height >= minimum
        trusted = self.base._trusted_components(ablations, model_config, trustworthy=trustworthy)
        learning = self.base._fit_ensemble_learning(
            predictions,
            model_config,
            trusted,
            trustworthy=trustworthy,
            minimum_rows=minimum,
        )
        return {
            "ensemble_learning": learning,
            "hyperparameters": search_hyperpriors(predictions, model_config),
            "residual_covariance": self.base._residual_covariance(predictions, model_config),
        }

    @staticmethod
    def _apply_inner_hyperparameters(
        model_config: dict[str, Any], hyperparameters: dict[str, Any]
    ) -> dict[str, Any]:
        selected = dict(hyperparameters.get("selected") or {})
        updated = json.loads(json.dumps(model_config))
        if selected.get("national_sigma") is not None:
            updated.setdefault("correlation", {})["national_sigma"] = float(
                selected["national_sigma"]
            )
        bayesian = updated.setdefault("bayesian", {})
        if selected.get("election_day_extra_sd") is not None:
            bayesian.setdefault("state_space", {})["election_day_extra_sd"] = float(
                selected["election_day_extra_sd"]
            )
        if selected.get("fundamentals_prior_strength") is not None:
            bayesian.setdefault("fundamentals_prior", {})["prior_strength"] = float(
                selected["fundamentals_prior_strength"]
            )
        return updated

    @classmethod
    def _baseline_metrics(cls, predictions: pl.DataFrame) -> dict[str, dict[str, Any]]:
        metrics: dict[str, dict[str, Any]] = {}
        for name, column in cls.BASELINE_COLUMNS.items():
            if predictions.is_empty() or column not in predictions.columns:
                metrics[name] = {
                    "status": "not_available",
                    "row_count": 0,
                    "reason": f"{column} is absent from nested predictions",
                }
                continue
            eligible = predictions.filter(pl.col(column).is_not_null())
            if eligible.is_empty():
                metrics[name] = {
                    "status": "not_available",
                    "row_count": 0,
                    "reason": "no eligible outer-fold rows for this comparator",
                }
                continue
            score = score_predictions(eligible, column)
            metrics[name] = {
                "status": "estimated",
                "row_count": eligible.height,
                "cycle_count": int(eligible["cycle"].n_unique()),
                **score,
            }
        return metrics

    @classmethod
    def _paired_cycle_clustered_uncertainty(
        cls,
        predictions: pl.DataFrame,
        *,
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        settings = config or {}
        minimum_cycles = max(int(settings.get("minimum_cycles_for_uncertainty", 3)), 2)
        replicates = max(int(settings.get("cycle_bootstrap_replicates", 2000)), 100)
        seed = int(settings.get("cycle_bootstrap_seed", 20260508))
        comparisons: dict[str, dict[str, Any]] = {}
        for index, (name, column) in enumerate(cls.BASELINE_COLUMNS.items()):
            cycle_scores = cls._paired_cycle_scores(predictions, column)
            if cycle_scores.is_empty():
                comparisons[name] = {
                    "status": "not_available",
                    "reason": f"no paired ensemble/{name} rows",
                    "independent_cycle_count": 0,
                }
                continue
            cycle_count = cycle_scores.height
            if cycle_count < minimum_cycles:
                comparisons[name] = {
                    "status": "insufficient_evidence",
                    "reason": (
                        f"{cycle_count} independent cycles < required {minimum_cycles}; "
                        "race rows are not treated as independent replicates"
                    ),
                    "independent_cycle_count": cycle_count,
                    "minimum_cycles": minimum_cycles,
                    "cycle_estimates": cycle_scores.to_dicts(),
                }
                continue
            comparison_seed = seed + index * 1009
            comparisons[name] = {
                "status": "estimated",
                "independent_cycle_count": cycle_count,
                "minimum_cycles": minimum_cycles,
                "cycle_estimates": cycle_scores.to_dicts(),
                "brier_difference": cls._cluster_bootstrap_interval(
                    cycle_scores["brier_difference"].to_numpy(),
                    replicates=replicates,
                    seed=comparison_seed,
                ),
                "log_score_difference": cls._cluster_bootstrap_interval(
                    cycle_scores["log_score_difference"].to_numpy(),
                    replicates=replicates,
                    seed=comparison_seed + 1,
                ),
            }
        estimated = [row for row in comparisons.values() if row["status"] == "estimated"]
        return {
            "status": "estimated" if estimated else "insufficient_evidence",
            "method": "paired_nonparametric_cycle_cluster_bootstrap",
            "cluster_unit": "cycle",
            "row_level_resampling": False,
            "equal_cycle_weighting": True,
            "minimum_cycles": minimum_cycles,
            "bootstrap_replicates": replicates,
            "seed": seed,
            "comparisons": comparisons,
        }

    @staticmethod
    def _paired_cycle_scores(predictions: pl.DataFrame, baseline_column: str) -> pl.DataFrame:
        required = {"cycle", "actual_winner", "ensemble_probability", baseline_column}
        if predictions.is_empty() or not required.issubset(predictions.columns):
            return pl.DataFrame()
        eligible = predictions.filter(
            pl.col("actual_winner").is_not_null()
            & pl.col("ensemble_probability").is_not_null()
            & pl.col(baseline_column).is_not_null()
        )
        if eligible.is_empty():
            return pl.DataFrame()
        clipped = eligible.with_columns(
            pl.col("actual_winner").cast(pl.Float64).alias("_actual"),
            pl.col("ensemble_probability").cast(pl.Float64).clip(1e-6, 1 - 1e-6).alias("_ensemble"),
            pl.col(baseline_column).cast(pl.Float64).clip(1e-6, 1 - 1e-6).alias("_baseline"),
        ).with_columns(
            ((pl.col("_ensemble") - pl.col("_actual")) ** 2).alias("_ensemble_brier"),
            ((pl.col("_baseline") - pl.col("_actual")) ** 2).alias("_baseline_brier"),
            (
                -(
                    pl.col("_actual") * pl.col("_ensemble").log()
                    + (1 - pl.col("_actual")) * (1 - pl.col("_ensemble")).log()
                )
            ).alias("_ensemble_log"),
            (
                -(
                    pl.col("_actual") * pl.col("_baseline").log()
                    + (1 - pl.col("_actual")) * (1 - pl.col("_baseline")).log()
                )
            ).alias("_baseline_log"),
        )
        return (
            clipped.group_by("cycle")
            .agg(
                (pl.col("_ensemble_brier") - pl.col("_baseline_brier"))
                .mean()
                .alias("brier_difference"),
                (pl.col("_ensemble_log") - pl.col("_baseline_log"))
                .mean()
                .alias("log_score_difference"),
                pl.len().alias("paired_row_count"),
            )
            .sort("cycle")
        )

    @staticmethod
    def _cluster_bootstrap_interval(
        cycle_differences: np.ndarray,
        *,
        replicates: int,
        seed: int,
    ) -> dict[str, float]:
        values = np.asarray(cycle_differences, dtype=np.float64)
        rng = np.random.default_rng(seed)
        indices = rng.integers(0, len(values), size=(replicates, len(values)))
        bootstrap_means = values[indices].mean(axis=1)
        return {
            "estimate": float(values.mean()),
            "lower_95": float(np.quantile(bootstrap_means, 0.025)),
            "upper_95": float(np.quantile(bootstrap_means, 0.975)),
            "cluster_standard_error": float(values.std(ddof=1) / math.sqrt(len(values))),
            "probability_ensemble_better": float(np.mean(bootstrap_means < 0.0)),
        }

    def _publication_outer_fold(
        self,
        *,
        bundle: FeatureBundle,
        train_catalog: pl.DataFrame,
        test_catalog: pl.DataFrame,
        target_cycle: int,
        as_of: str,
        as_of_offset_days: int,
        model_config: dict[str, Any],
        inference_engine: str,
        residual_covariance: pl.DataFrame,
        holdovers: dict[str, int] | None,
    ) -> tuple[pl.DataFrame, dict[str, bool]]:
        # Lazy import avoids the simulation -> scoring.learning -> scoring package
        # initialization cycle during test collection.
        from civic_signal.models.simulation import SimulationEngine

        incomplete_path = {
            "simulation_engine_used": False,
            "posterior_draw_path_complete": False,
        }
        train_bundle = filter_results_before_cycle(
            filter_bundle_by_date(subset_bundle(bundle, train_catalog), as_of), target_cycle
        )
        test_bundle = filter_bundle_by_date(subset_bundle(bundle, test_catalog), as_of)
        components, ensemble, cycle_config, posterior_draws = self.base._publication_components(
            train_bundle=train_bundle,
            test_bundle=test_bundle,
            as_of=as_of,
            model_config=model_config,
            inference_engine=inference_engine,
        )
        if ensemble.is_empty():
            # No SimulationEngine invocation — never claim exact publication path.
            return self.base._empty_predictions(), incomplete_path
        posterior_path_complete = inference_engine != "bayes" or not posterior_draws.is_empty()
        outputs = SimulationEngine(
            cycle_config,
            residual_covariance=residual_covariance,
            holdovers=holdovers,
        ).run(
            replace(test_bundle, results=test_bundle.results.head(0)),
            ensemble,
            posterior_draws=posterior_draws if not posterior_draws.is_empty() else None,
        )
        forecast_map = {
            (str(row["race_id"]), str(row["option_id"])): row
            for row in outputs.race_forecasts.iter_rows(named=True)
        }
        actuals = {
            (str(row["race_id"]), str(row["option_id"])): row
            for row in test_bundle.results.iter_rows(named=True)
        }
        component_maps = {
            "polls_probability": self.base._component_probability(components[0]),
            "fundamentals_probability": self.base._component_probability(components[1]),
            "markets_probability": self.base._component_probability(components[2]),
            "public_signals_probability": self.base._component_probability(components[3]),
        }
        baseline_sigma = self.base._baseline_sigma(train_bundle, model_config)
        option_counts = {
            str(row["race_id"]): int(row["option_count"])
            for row in test_bundle.options.group_by("race_id")
            .agg(pl.len().alias("option_count"))
            .iter_rows(named=True)
        }
        poll_average = self._poll_average_probabilities(test_bundle.polls)
        rows = []
        for option in test_bundle.options.iter_rows(named=True):
            key = (str(option["race_id"]), str(option["option_id"]))
            actual = actuals.get(key)
            forecast = forecast_map.get(key)
            if actual is None or forecast is None or forecast.get("winner_probability") is None:
                continue
            previous_share = float(option.get("previous_vote_share") or 0.5)
            previous_cycle_swing = normal_cdf((previous_share - 0.5) / baseline_sigma)
            row = {
                "race_id": key[0],
                "option_id": key[1],
                "cycle": target_cycle,
                "as_of": as_of,
                "as_of_offset_days": as_of_offset_days,
                "polling_inference_engine": inference_engine,
                "party": option.get("party"),
                "actual_winner": bool(actual["winner"]),
                "actual_vote_share": float(actual["vote_share"]),
                "baseline_probability": previous_cycle_swing,
                "baseline_sigma": baseline_sigma,
                "prior_only_probability": 1.0 / max(option_counts.get(key[0], 1), 1),
                "previous_cycle_swing_probability": previous_cycle_swing,
                "fundamentals_only_probability": component_maps["fundamentals_probability"].get(
                    key
                ),
                "poll_average_probability": poll_average.get(key),
                "market_implied_probability": component_maps["markets_probability"].get(key),
                "ensemble_probability": float(forecast["winner_probability"]),
                "learned_ensemble_probability": float(forecast["winner_probability"]),
                "configured_ensemble_probability": None,
                "predicted_vote_share": forecast.get("vote_share_mean"),
                "lower_90": forecast.get("vote_share_p05"),
                "upper_90": forecast.get("vote_share_p95"),
            }
            for column, values in component_maps.items():
                row[column] = values.get(key, row["baseline_probability"])
            rows.append(row)
        frame = pl.DataFrame(rows) if rows else self.base._empty_predictions()
        return frame, {
            "simulation_engine_used": True,
            "posterior_draw_path_complete": posterior_path_complete,
        }

    @staticmethod
    def _poll_average_probabilities(
        polls: pl.DataFrame,
    ) -> dict[tuple[str, str], float]:
        required = {"race_id", "option_id", "pct"}
        if polls.is_empty() or not required.issubset(polls.columns):
            return {}
        averages = (
            polls.filter(pl.col("pct").is_not_null())
            .group_by(["race_id", "option_id"])
            .agg(pl.col("pct").cast(pl.Float64).mean().alias("average_pct"))
            .with_columns(
                pl.len().over("race_id").alias("polled_option_count"),
                pl.col("average_pct").sum().over("race_id").alias("race_pct_sum"),
            )
            .filter((pl.col("polled_option_count") > 1) & (pl.col("race_pct_sum") > 0))
            .with_columns((pl.col("average_pct") / pl.col("race_pct_sum")).alias("probability"))
        )
        return {
            (str(row["race_id"]), str(row["option_id"])): float(row["probability"])
            for row in averages.iter_rows(named=True)
        }

    @staticmethod
    def _training_lineage_hash(bundle: FeatureBundle, catalog: pl.DataFrame) -> str:
        race_ids = sorted(str(value) for value in catalog["race_id"].to_list())
        tables: dict[str, list[dict[str, Any]]] = {}
        for name, frame in (
            ("races", bundle.races),
            ("options", bundle.options),
            ("polls", bundle.polls),
            ("markets", bundle.markets),
            ("public_signals", bundle.public_signals),
            ("fundamentals", bundle.fundamentals),
            ("results", bundle.results),
        ):
            if "race_id" not in frame.columns:
                tables[name] = []
                continue
            selected = frame.filter(pl.col("race_id").is_in(race_ids))
            if selected.columns:
                selected = selected.select(sorted(selected.columns)).sort(selected.columns)
            tables[name] = selected.to_dicts()
        return hashlib.sha256(
            json.dumps(
                {"race_ids": race_ids, "tables": tables},
                sort_keys=True,
                default=str,
            ).encode()
        ).hexdigest()

    @classmethod
    def _held_out_permutation_canary(
        cls, bundle: FeatureBundle, catalog: pl.DataFrame, outer_cycle: int
    ) -> bool:
        training = catalog.filter(pl.col("cycle") < outer_cycle)
        before = cls._training_lineage_hash(bundle, training)
        outer_ids = catalog.filter(pl.col("cycle") == outer_cycle)["race_id"].to_list()
        if "winner" not in bundle.results.columns:
            return before == cls._training_lineage_hash(bundle, training)
        permuted_results = bundle.results.with_columns(
            pl.when(pl.col("race_id").is_in(outer_ids))
            .then(~pl.col("winner"))
            .otherwise(pl.col("winner"))
            .alias("winner")
        )
        permuted = FeatureBundle(**{**bundle.__dict__, "results": permuted_results})
        return before == cls._training_lineage_hash(permuted, training)

    @staticmethod
    def _empty_manifest() -> pl.DataFrame:
        return pl.DataFrame(
            schema={
                "outer_cycle": pl.Int64,
                "train_cycles": pl.Utf8,
                "inner_validation_cycles": pl.Utf8,
                "fit_cycle_max": pl.Int64,
                "outer_cycle_excluded": pl.Boolean,
                "inner_row_count": pl.Int64,
                "outer_row_count": pl.Int64,
                "weights_status": pl.Utf8,
                "calibration_status": pl.Utf8,
                "hyperparameter_status": pl.Utf8,
                "selected_hyperparameters": pl.Utf8,
                "publication_path": pl.Utf8,
                "simulation_engine_used": pl.Boolean,
                "posterior_draw_path_complete": pl.Boolean,
                "training_lineage_sha256": pl.Utf8,
                "held_out_permutation_affects_training": pl.Boolean,
                "held_out_permutation_canary_passed": pl.Boolean,
            }
        )


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
