"""Reward-v2 evaluators: recompute R0-R27 from primary artifacts only."""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from civic_signal.scoring.reward_registry import (
    all_reward_ids,
    load_rewards_config,
    make_reward_record,
    profile_required_rewards,
    publication_mode_default,
    threshold_for,
)

SCHEMA_VERSION = "2.0.0"

# Artifact paths relative to a forecast/attempt run directory.
_PRIMARY_ARTIFACTS = {
    "race_forecasts": "race_forecasts.parquet",
    "race_catalog": "race_catalog.parquet",
    "source_manifest": "source_manifest.parquet",
    "forecast_draws": "forecast_draws.parquet",
    "control_forecasts": "control_forecasts.parquet",
    "plot_manifest": "plot_manifest.json",
    "performance": "performance.json",
    "fingerprint": "reproducibility_fingerprint.json",
    "posterior_diagnostics": "posterior_diagnostics.json",
    "daily_update": "latest_daily_update.json",
    "backtest": "backtest_summary.json",
    "run_manifest": "run_manifest.json",
    "publication_decision": "publication_decision.json",
    "promotion_manifest": "promotion_manifest.json",
    "semantic_verification": "semantic_verification.json",
    "as_of_audit": "as_of_audit.json",
    "nested_eval": "nested_evaluation.json",
    "covariance_recovery": "covariance_recovery.json",
    "hierarchy_recovery": "hierarchy_recovery.json",
    "poll_observation_manifest": "poll_observation_manifest.json",
    "feature_lineage": "feature_lineage.json",
    "live_source_canaries": "live_source_canaries.json",
    "benchmark_superiority": "benchmark_superiority.json",
    "contract_parity": "contract_parity.json",
    "ci_manifest": "ci_manifest.json",
    "coverage": "coverage.json",
    "registry_audit": "source_registry_audit.json",
    "recalibration_map": "recalibration_map.parquet",
}


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, dict) else None


def _read_parquet(path: Path) -> pl.DataFrame | None:
    if not path.exists():
        return None
    return pl.read_parquet(path)


def _is_nan(value: Any) -> bool:
    try:
        return value is not None and isinstance(value, float) and math.isnan(value)
    except (TypeError, ValueError):
        return False


def _finite_number(value: Any) -> float | None:
    if value is None or _is_nan(value):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _first_key(mapping: dict[str, Any] | None, *keys: str) -> Any:
    """Return the first present non-None value for keys (0.0 is valid, not missing)."""
    if not mapping:
        return None
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def _finite_from(mapping: dict[str, Any] | None, *keys: str) -> float | None:
    """First finite number among keys; never treats 0.0 as missing via truthiness."""
    return _finite_number(_first_key(mapping, *keys))


def _missing_metric_state(
    reward_id: str,
    *,
    reasons: list[str],
    threshold: dict[str, Any],
    evidence: list[str] | None = None,
    metric: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return make_reward_record(
        reward_id=reward_id,
        state="insufficient_evidence",
        metric=metric or {},
        threshold=threshold,
        evidence=evidence or [],
        failure_reasons=reasons,
        blocks_publication=True,
    )


def _fail(
    reward_id: str,
    *,
    reasons: list[str],
    threshold: dict[str, Any],
    metric: dict[str, Any] | None = None,
    evidence: list[str] | None = None,
) -> dict[str, Any]:
    return make_reward_record(
        reward_id=reward_id,
        state="fail",
        metric=metric or {},
        threshold=threshold,
        evidence=evidence or [],
        failure_reasons=reasons,
        blocks_publication=True,
    )


def _pass(
    reward_id: str,
    *,
    threshold: dict[str, Any],
    metric: dict[str, Any] | None = None,
    evidence: list[str] | None = None,
    negative_tests_passed: list[str] | None = None,
) -> dict[str, Any]:
    return make_reward_record(
        reward_id=reward_id,
        state="pass",
        metric=metric or {},
        threshold=threshold,
        evidence=evidence or [],
        negative_tests_passed=negative_tests_passed or [],
        failure_reasons=[],
        blocks_publication=True,
    )


class RewardV2Evaluator:
    """Recompute reward-v2 states from primary run artifacts."""

    def __init__(
        self,
        rewards_config: dict[str, Any] | None = None,
        rewards_config_path: str | Path | None = None,
        model_config: dict[str, Any] | None = None,
    ) -> None:
        if rewards_config is not None:
            self.config = rewards_config
        else:
            path = str(rewards_config_path) if rewards_config_path else None
            self.config = load_rewards_config(path)
        self.model_config = model_config or {}

    def evaluate_run_dir(
        self,
        run_dir: Path,
        *,
        run_id: str | None = None,
        profile: str = "production",
        publication_mode: str | None = None,
    ) -> dict[str, Any]:
        run_dir = Path(run_dir)
        resolved_run_id = run_id or run_dir.name
        mode = publication_mode or self._infer_publication_mode(run_dir)
        artifacts = self._load_artifacts(run_dir)
        rewards: dict[str, Any] = {}
        for reward_id in all_reward_ids(self.config):
            evaluator = getattr(self, f"_eval_{reward_id}", None)
            if evaluator is None:
                rewards[reward_id] = _missing_metric_state(
                    reward_id,
                    reasons=[f"No evaluator registered for {reward_id}"],
                    threshold=threshold_for(reward_id, self.config),
                )
                continue
            rewards[reward_id] = evaluator(run_dir, artifacts)

        required = profile_required_rewards(profile, self.config)
        # Conditional R9 for production when public signals are admitted.
        if profile == "production" and self._public_signals_admitted(artifacts):
            if "R9_public_signal_discipline" not in required:
                required = [*required, "R9_public_signal_discipline"]

        blocking: list[str] = []
        for reward_id in required:
            record = rewards.get(reward_id)
            if not isinstance(record, dict):
                blocking.append(reward_id)
                continue
            state = record.get("state")
            if state in {"fail", "insufficient_evidence"}:
                blocking.append(reward_id)

        profile_cfg = dict(self.config.get("profiles", {}).get(profile, {}))
        blocks = bool(profile_cfg.get("blocks_publication", True)) and bool(blocking)

        return {
            "schema_version": SCHEMA_VERSION,
            "run_id": resolved_run_id,
            "generated_at": datetime.now(UTC).isoformat(),
            "profile": profile,
            "publication_mode": mode,
            "recomputed": True,
            "blocks_publication": blocks or bool(blocking and profile == "production"),
            "blocking_rewards": sorted(blocking),
            "lineage": {
                "run_dir": str(run_dir),
                "rewards_config_version": self.config.get("version"),
                "primary_artifacts_present": sorted(
                    key for key, path in _PRIMARY_ARTIFACTS.items() if (run_dir / path).exists()
                ),
            },
            "rewards": rewards,
        }

    def _infer_publication_mode(self, run_dir: Path) -> str:
        decision = _read_json(run_dir / "publication_decision.json")
        if decision and decision.get("publication_mode"):
            return str(decision["publication_mode"])
        manifest = _read_json(run_dir / "run_manifest.json")
        if manifest and manifest.get("publication_mode"):
            return str(manifest["publication_mode"])
        return publication_mode_default(self.config)

    def _load_artifacts(self, run_dir: Path) -> dict[str, Any]:
        artifacts: dict[str, Any] = {"run_dir": run_dir}
        for key, rel in _PRIMARY_ARTIFACTS.items():
            path = run_dir / rel
            if rel.endswith(".parquet"):
                artifacts[key] = _read_parquet(path)
            else:
                artifacts[key] = _read_json(path)
        # Prefer nested backtest summary if present under common names.
        if artifacts.get("backtest") is None:
            for name in ("backtest.json", "backtest_payload.json", "metrics.json"):
                payload = _read_json(run_dir / name)
                if payload is not None:
                    artifacts["backtest"] = payload
                    break
        # Also accept posterior diagnostics under the bayesian name written by pipeline.
        if artifacts.get("posterior_diagnostics") is None:
            artifacts["posterior_diagnostics"] = _read_json(run_dir / "posterior_diagnostics.json")
        return artifacts

    def _public_signals_admitted(self, artifacts: dict[str, Any]) -> bool:
        trusted = dict(self.model_config.get("trusted_components", {}))
        if trusted.get("public_signals"):
            return True
        backtest = artifacts.get("backtest") or {}
        ablations = dict(backtest.get("ablations", {})) if isinstance(backtest, dict) else {}
        return bool(ablations.get("public_signals", {}).get("admitted"))

    # --- individual reward evaluators ---------------------------------

    def _eval_R0_build(self, run_dir: Path, artifacts: dict[str, Any]) -> dict[str, Any]:
        thr = threshold_for("R0_build", self.config)
        min_cov = float(thr.get("min_line_coverage_pct", 90.0))
        ci = artifacts.get("ci_manifest") or {}
        coverage = artifacts.get("coverage") or {}
        evidence = []
        if (run_dir / "ci_manifest.json").exists():
            evidence.append("ci_manifest.json")
        if (run_dir / "coverage.json").exists():
            evidence.append("coverage.json")
        coverage_pct = _finite_from(coverage, "line_rate_pct", "percent")
        if coverage_pct is None and ci:
            coverage_pct = _finite_from(ci, "line_coverage_pct")
        commands_ok = ci.get("commands_passed")
        if coverage_pct is None and commands_ok is None:
            return _missing_metric_state(
                "R0_build",
                reasons=["Missing CI/coverage evidence for build gate"],
                threshold=thr,
                evidence=evidence,
            )
        metric = {
            "line_coverage_pct": coverage_pct,
            "commands_passed": commands_ok,
            "tool_versions": ci.get("tool_versions", {}),
        }
        reasons: list[str] = []
        if commands_ok is False:
            reasons.append("Required CI commands failed")
        if coverage_pct is not None and coverage_pct < min_cov:
            reasons.append(f"Line coverage {coverage_pct} < {min_cov}")
        if reasons:
            return _fail(
                "R0_build", reasons=reasons, threshold=thr, metric=metric, evidence=evidence
            )
        if coverage_pct is None:
            return _missing_metric_state(
                "R0_build",
                reasons=["Coverage metric missing or NaN"],
                threshold=thr,
                evidence=evidence,
                metric=metric,
            )
        return _pass("R0_build", threshold=thr, metric=metric, evidence=evidence)

    def _eval_R1_reproducibility(self, run_dir: Path, artifacts: dict[str, Any]) -> dict[str, Any]:
        thr = threshold_for("R1_reproducibility", self.config)
        fp = artifacts.get("fingerprint") or {}
        evidence = ["reproducibility_fingerprint.json"] if fp else []
        if not fp:
            return _missing_metric_state(
                "R1_reproducibility",
                reasons=["Missing reproducibility_fingerprint.json"],
                threshold=thr,
            )
        verified = bool(fp.get("cross_run_verified"))
        metric = {
            "cross_run_verified": verified,
            "compared_to_previous": bool(fp.get("compared_to_previous")),
            "combined_hash": fp.get("combined_hash"),
        }
        if thr.get("require_cross_run_verified", True) and not verified:
            return _fail(
                "R1_reproducibility",
                reasons=["Cross-run fingerprint verification has not succeeded"],
                threshold=thr,
                metric=metric,
                evidence=evidence,
            )
        return _pass("R1_reproducibility", threshold=thr, metric=metric, evidence=evidence)

    def _eval_R2_provenance(self, run_dir: Path, artifacts: dict[str, Any]) -> dict[str, Any]:
        thr = threshold_for("R2_provenance", self.config)
        min_share = float(thr.get("min_provenance_share", 1.0))
        forecasts = artifacts.get("race_forecasts")
        manifest = artifacts.get("source_manifest")
        evidence: list[str] = []
        if forecasts is not None:
            evidence.append("race_forecasts.parquet")
        if manifest is not None:
            evidence.append("source_manifest.parquet")
        if forecasts is None or manifest is None or forecasts.is_empty() or manifest.is_empty():
            return _missing_metric_state(
                "R2_provenance",
                reasons=["Missing race_forecasts or source_manifest for provenance share"],
                threshold=thr,
                evidence=evidence,
            )
        required = {"model_config_hash", "source_manifest_hash"}
        if not required.issubset(set(forecasts.columns)):
            return _fail(
                "R2_provenance",
                reasons=["Forecast rows missing lineage hash columns"],
                threshold=thr,
                metric={"missing_columns": sorted(required - set(forecasts.columns))},
                evidence=evidence,
            )
        forecast_ok = forecasts.filter(
            pl.col("model_config_hash").is_not_null()
            & (pl.col("model_config_hash") != "")
            & pl.col("source_manifest_hash").is_not_null()
            & (pl.col("source_manifest_hash") != "")
        ).height
        manifest_ok = (
            manifest.filter(
                pl.col("content_hash").is_not_null() & (pl.col("content_hash") != "")
            ).height
            if "content_hash" in manifest.columns
            else 0
        )
        share = min(forecast_ok / forecasts.height, manifest_ok / manifest.height)
        metric = {
            "provenance_share": share,
            "forecast_rows": forecasts.height,
            "manifest_rows": manifest.height,
        }
        if share < min_share:
            return _fail(
                "R2_provenance",
                reasons=[f"Provenance share {share} < {min_share}"],
                threshold=thr,
                metric=metric,
                evidence=evidence,
            )
        return _pass("R2_provenance", threshold=thr, metric=metric, evidence=evidence)

    def _eval_R3_sync_integrity(self, run_dir: Path, artifacts: dict[str, Any]) -> dict[str, Any]:
        thr = threshold_for("R3_sync_integrity", self.config)
        manifest = artifacts.get("source_manifest")
        if manifest is None:
            return _missing_metric_state(
                "R3_sync_integrity",
                reasons=["Missing source_manifest.parquet"],
                threshold=thr,
            )
        failed = (
            int(manifest.filter(pl.col("status") == "failed").height)
            if "status" in manifest.columns
            else 0
        )
        failed_auth = 0
        if "status" in manifest.columns and "auth_mode" in manifest.columns:
            failed_auth = int(
                manifest.filter(
                    (pl.col("status") == "failed") & (pl.col("auth_mode") != "public")
                ).height
            )
        stale = 0
        if "freshness_status" in manifest.columns:
            stale = int(manifest.filter(pl.col("freshness_status") == "stale").height)
        metric = {
            "failed": failed,
            "failed_auth": failed_auth,
            "stale": stale,
            "total": int(manifest.height),
        }
        reasons: list[str] = []
        if failed > int(thr.get("max_failed_sources", 0)):
            reasons.append(f"Failed sources: {failed}")
        if failed_auth > int(thr.get("max_failed_auth", 0)):
            reasons.append(f"Failed auth sources: {failed_auth}")
        if stale > 0:
            reasons.append(f"Stale sources exceeding TTL: {stale}")
        evidence = ["source_manifest.parquet"]
        if reasons:
            return _fail(
                "R3_sync_integrity",
                reasons=reasons,
                threshold=thr,
                metric=metric,
                evidence=evidence,
            )
        return _pass("R3_sync_integrity", threshold=thr, metric=metric, evidence=evidence)

    def _eval_R4_calibration(self, run_dir: Path, artifacts: dict[str, Any]) -> dict[str, Any]:
        thr = threshold_for("R4_calibration", self.config)
        backtest = artifacts.get("backtest") or {}
        nested = artifacts.get("nested_eval") or {}
        metrics = {}
        if nested.get("calibration"):
            metrics = dict(nested["calibration"])
        elif isinstance(backtest, dict):
            metrics = dict(backtest.get("metrics", {}).get("ensemble", {}))
        evidence = [
            p for p in ("nested_evaluation.json", "backtest_summary.json") if (run_dir / p).exists()
        ]
        ece = _finite_from(metrics, "expected_calibration_error", "ece")
        intercept = _finite_from(metrics, "calibration_intercept", "intercept")
        slope = _finite_from(metrics, "calibration_slope", "slope")
        ece_upper = _finite_from(metrics, "ece_bootstrap_upper", "ece_upper")
        outer = bool(nested.get("outer_fold") or backtest.get("rolling_origin_executed"))
        if ece is None or intercept is None or slope is None:
            return _missing_metric_state(
                "R4_calibration",
                reasons=["Missing outer-fold calibration metrics (ECE/intercept/slope)"],
                threshold=thr,
                evidence=evidence,
                metric=metrics,
            )
        if not outer and not nested:
            return _missing_metric_state(
                "R4_calibration",
                reasons=["Calibration metrics are not from an outer-fold evaluation"],
                threshold=thr,
                evidence=evidence,
                metric=metrics,
            )
        reasons: list[str] = []
        if ece > float(thr["max_ece"]):
            reasons.append(f"ECE {ece} > {thr['max_ece']}")
        if ece_upper is not None and ece_upper > float(thr["max_ece_bootstrap_upper"]):
            reasons.append(f"ECE bootstrap upper {ece_upper} > {thr['max_ece_bootstrap_upper']}")
        if abs(intercept) > float(thr["max_abs_intercept"]):
            reasons.append(f"|intercept| {abs(intercept)} > {thr['max_abs_intercept']}")
        if slope < float(thr["slope_min"]) or slope > float(thr["slope_max"]):
            reasons.append(f"slope {slope} outside [{thr['slope_min']}, {thr['slope_max']}]")
        # Fold lineage canary: calibration fitted on scored rows.
        if nested.get("calibration_fitted_on_scored_rows") is True:
            reasons.append("Calibration was fit on its scored outer-fold rows")
        metric = {
            "ece": ece,
            "ece_bootstrap_upper": ece_upper,
            "intercept": intercept,
            "slope": slope,
            "outer_fold": outer,
        }
        if reasons:
            return _fail(
                "R4_calibration", reasons=reasons, threshold=thr, metric=metric, evidence=evidence
            )
        return _pass("R4_calibration", threshold=thr, metric=metric, evidence=evidence)

    def _eval_R5_baseline_competition(
        self, run_dir: Path, artifacts: dict[str, Any]
    ) -> dict[str, Any]:
        thr = threshold_for("R5_baseline_competition", self.config)
        nested = artifacts.get("nested_eval") or {}
        bench = nested.get("paired_benchmarks") or artifacts.get("benchmark_superiority") or {}
        backtest = artifacts.get("backtest") or {}
        ablations = dict(backtest.get("ablations", {})) if isinstance(backtest, dict) else {}
        evidence = [
            p
            for p in (
                "nested_evaluation.json",
                "benchmark_superiority.json",
                "backtest_summary.json",
            )
            if (run_dir / p).exists()
        ]
        cycle_count = None
        for source in (bench, nested, backtest if isinstance(backtest, dict) else {}):
            if isinstance(source, dict):
                value = _finite_from(source, "independent_cycle_count")
                if value is not None:
                    cycle_count = value
                    break
        beats = bench.get("beats_all_simple_baselines")
        if beats is None:
            beats = ablations.get("ensemble", {}).get("beats_or_matches_baseline")
        log_ci_upper = None
        for source in (bench, nested):
            if isinstance(source, dict):
                value = _finite_from(source, "log_score_diff_ci_upper")
                if value is not None:
                    log_ci_upper = value
                    break
        if cycle_count is None or beats is None:
            return _missing_metric_state(
                "R5_baseline_competition",
                reasons=["Missing paired baseline competition evidence or cycle count"],
                threshold=thr,
                evidence=evidence,
            )
        reasons: list[str] = []
        min_cycles = float(thr.get("min_independent_cycles", 6))
        if cycle_count < min_cycles:
            reasons.append(f"Independent cycle count {cycle_count} < {min_cycles}")
        if not bool(beats):
            reasons.append("Model does not beat every simple baseline")
        if (
            thr.get("require_log_score_ci_below_zero")
            and log_ci_upper is not None
            and log_ci_upper >= 0
        ):
            reasons.append(f"Log-score paired CI upper {log_ci_upper} is not below zero")
        # Effective-n inflation canary.
        if nested.get("effective_n_inflated") is True:
            reasons.append("Effective sample size was inflated by repeated horizons/options")
        metric = {
            "independent_cycle_count": cycle_count,
            "beats_all_simple_baselines": bool(beats),
            "log_score_diff_ci_upper": log_ci_upper,
        }
        if reasons:
            return _fail(
                "R5_baseline_competition",
                reasons=reasons,
                threshold=thr,
                metric=metric,
                evidence=evidence,
            )
        return _pass("R5_baseline_competition", threshold=thr, metric=metric, evidence=evidence)

    def _eval_R6_component_admission(
        self, run_dir: Path, artifacts: dict[str, Any]
    ) -> dict[str, Any]:
        thr = threshold_for("R6_component_admission", self.config)
        nested = artifacts.get("nested_eval") or {}
        backtest = artifacts.get("backtest") or {}
        ablations = nested.get("component_ablations") or (
            dict(backtest.get("ablations", {})) if isinstance(backtest, dict) else {}
        )
        evidence = [
            p for p in ("nested_evaluation.json", "backtest_summary.json") if (run_dir / p).exists()
        ]
        if not ablations:
            return _missing_metric_state(
                "R6_component_admission",
                reasons=["Missing nested leave-one-component-out ablations"],
                threshold=thr,
                evidence=evidence,
            )
        matched = nested.get("matched_coverage", thr.get("require_matched_coverage", True))
        baseline_filled = bool(
            nested.get("baseline_filled_missing") or ablations.get("_baseline_filled")
        )
        reasons: list[str] = []
        if thr.get("require_matched_coverage") and matched is False:
            reasons.append("Component ablations are not matched-coverage")
        if baseline_filled:
            reasons.append("Missing component estimates were baseline-filled")
        ensemble = dict(ablations.get("ensemble", {}))
        if ensemble and not bool(ensemble.get("beats_or_matches_baseline", True)):
            reasons.append("Ensemble fails baseline comparison under matched LOCO")
        metric = {
            "matched_coverage": matched,
            "baseline_filled_missing": baseline_filled,
            "ablation_keys": sorted(str(k) for k in ablations if not str(k).startswith("_")),
        }
        if reasons:
            return _fail(
                "R6_component_admission",
                reasons=reasons,
                threshold=thr,
                metric=metric,
                evidence=evidence,
            )
        # Without nested exact LOCO, do not auto-pass production.
        if not nested.get("exact_pipeline") and not nested.get("component_ablations"):
            return _missing_metric_state(
                "R6_component_admission",
                reasons=["Nested LOCO admission evidence incomplete"],
                threshold=thr,
                evidence=evidence,
                metric=metric,
            )
        return _pass("R6_component_admission", threshold=thr, metric=metric, evidence=evidence)

    def _eval_R7_sparse_honesty(self, run_dir: Path, artifacts: dict[str, Any]) -> dict[str, Any]:
        thr = threshold_for("R7_sparse_honesty", self.config)
        forecasts = artifacts.get("race_forecasts")
        catalog = artifacts.get("race_catalog")
        evidence: list[str] = []
        if forecasts is not None:
            evidence.append("race_forecasts.parquet")
        if catalog is not None:
            evidence.append("race_catalog.parquet")
        if forecasts is None or catalog is None:
            return _missing_metric_state(
                "R7_sparse_honesty",
                reasons=["Missing race_forecasts or race_catalog"],
                threshold=thr,
                evidence=evidence,
            )
        if "tier" not in catalog.columns or "race_id" not in catalog.columns:
            return _missing_metric_state(
                "R7_sparse_honesty",
                reasons=["race_catalog missing tier/race_id columns"],
                threshold=thr,
                evidence=evidence,
            )
        tier_c = catalog.filter(pl.col("tier") == "C")["race_id"].to_list()
        if not tier_c:
            return _pass(
                "R7_sparse_honesty",
                threshold=thr,
                metric={"tier_c_races": 0, "public_probs_null": True},
                evidence=evidence,
            )
        tier_c_fc = forecasts.filter(pl.col("race_id").is_in(tier_c))
        if "winner_probability" not in tier_c_fc.columns:
            return _fail(
                "R7_sparse_honesty",
                reasons=["Tier C forecasts missing winner_probability column"],
                threshold=thr,
                evidence=evidence,
            )
        public_null = tier_c_fc["winner_probability"].null_count() == tier_c_fc.height
        # Control-bearing internal draws check if column present.
        control_ok = True
        control_policy = artifacts.get("run_manifest") or {}
        withhold_control = bool(control_policy.get("withhold_aggregate_control"))
        if not public_null:
            return _fail(
                "R7_sparse_honesty",
                reasons=["Public Tier C probabilities are not null"],
                threshold=thr,
                metric={"tier_c_races": len(tier_c), "public_probs_null": False},
                evidence=evidence,
            )
        draws = artifacts.get("forecast_draws")
        if draws is not None and "race_id" in draws.columns and not withhold_control:
            missing_draws = set(tier_c) - set(draws["race_id"].unique().to_list())
            if missing_draws and not withhold_control:
                control_ok = False
        metric = {
            "tier_c_races": len(tier_c),
            "public_probs_null": public_null,
            "control_bearing_internal_draws": control_ok,
            "withhold_aggregate_control": withhold_control,
        }
        if not control_ok:
            return _fail(
                "R7_sparse_honesty",
                reasons=[
                    "Sparse control-bearing races lack internal draws and control not withheld"
                ],
                threshold=thr,
                metric=metric,
                evidence=evidence,
            )
        return _pass("R7_sparse_honesty", threshold=thr, metric=metric, evidence=evidence)

    def _eval_R8_uncertainty_quality(
        self, run_dir: Path, artifacts: dict[str, Any]
    ) -> dict[str, Any]:
        thr = threshold_for("R8_uncertainty_quality", self.config)
        tol = float(thr.get("coverage_tolerance", 0.05))
        nested = artifacts.get("nested_eval") or {}
        backtest = artifacts.get("backtest") or {}
        metrics = dict(nested.get("coverage") or backtest.get("metrics", {}).get("ensemble", {}))
        evidence = [
            p for p in ("nested_evaluation.json", "backtest_summary.json") if (run_dir / p).exists()
        ]
        levels = list(thr.get("nominal_levels", [0.5, 0.8, 0.9]))
        observed: dict[str, float | None] = {}
        reasons: list[str] = []
        missing_levels: list[str] = []
        for level in levels:
            key_candidates = [
                f"interval_{int(level * 100)}_coverage",
                f"coverage_{int(level * 100)}",
                f"interval_{level}_coverage",
            ]
            value = None
            for key in key_candidates:
                value = _finite_number(metrics.get(key))
                if value is not None:
                    break
            observed[str(level)] = value
            if value is None:
                missing_levels.append(str(level))
            elif abs(value - float(level)) > tol:
                reasons.append(f"Coverage at {level}: {value} outside ±{tol}")
        if missing_levels:
            return _missing_metric_state(
                "R8_uncertainty_quality",
                reasons=[f"Missing interval coverage for levels: {missing_levels}"],
                threshold=thr,
                evidence=evidence,
                metric={"observed_coverage": observed, "tolerance": tol},
            )
        if nested.get("effective_n_inflated") is True:
            reasons.append("Coverage sample size inflated by candidate complements")
        metric = {"observed_coverage": observed, "tolerance": tol}
        if reasons:
            return _fail(
                "R8_uncertainty_quality",
                reasons=reasons,
                threshold=thr,
                metric=metric,
                evidence=evidence,
            )
        return _pass("R8_uncertainty_quality", threshold=thr, metric=metric, evidence=evidence)

    def _eval_R9_public_signal_discipline(
        self, run_dir: Path, artifacts: dict[str, Any]
    ) -> dict[str, Any]:
        thr = threshold_for("R9_public_signal_discipline", self.config)
        trusted = bool(dict(self.model_config.get("trusted_components", {})).get("public_signals"))
        nested = artifacts.get("nested_eval") or {}
        leakage = nested.get("public_signal_leakage") or artifacts.get("run_manifest") or {}
        evidence = [
            p for p in ("nested_evaluation.json", "run_manifest.json") if (run_dir / p).exists()
        ]
        if not trusted:
            return _pass(
                "R9_public_signal_discipline",
                threshold=thr,
                metric={"public_signals_trusted": False, "status": "experimental"},
                evidence=evidence,
            )
        leakage_ok = leakage.get("leakage_passed")
        nested_value = leakage.get("nested_value_added")
        if leakage_ok is None or nested_value is None:
            return _missing_metric_state(
                "R9_public_signal_discipline",
                reasons=["Public signals admitted without leakage/nested-value evidence"],
                threshold=thr,
                evidence=evidence,
            )
        reasons: list[str] = []
        if leakage.get("result_derived") or leakage.get("post_as_of"):
            reasons.append("Public signal is result-derived or post-as-of")
        if thr.get("require_leakage_audit") and not leakage_ok:
            reasons.append("Leakage audit failed")
        if thr.get("require_nested_value") and not nested_value:
            reasons.append("Nested out-of-sample value not demonstrated")
        metric = {
            "public_signals_trusted": True,
            "leakage_passed": leakage_ok,
            "nested_value_added": nested_value,
        }
        if reasons:
            return _fail(
                "R9_public_signal_discipline",
                reasons=reasons,
                threshold=thr,
                metric=metric,
                evidence=evidence,
            )
        return _pass("R9_public_signal_discipline", threshold=thr, metric=metric, evidence=evidence)

    def _eval_R10_explainability(self, run_dir: Path, artifacts: dict[str, Any]) -> dict[str, Any]:
        thr = threshold_for("R10_explainability", self.config)
        forecasts = artifacts.get("race_forecasts")
        if forecasts is None or forecasts.is_empty():
            return _missing_metric_state(
                "R10_explainability",
                reasons=["Missing race_forecasts for explainability columns"],
                threshold=thr,
            )
        required = {
            "tier_reason",
            "data_quality_flags",
            "top_drivers",
            "component_contributions",
            "uncertainty_explanation",
        }
        missing = sorted(required - set(forecasts.columns))
        evidence = ["race_forecasts.parquet"]
        if missing:
            return _fail(
                "R10_explainability",
                reasons=[f"Missing explanation columns: {missing}"],
                threshold=thr,
                metric={"missing_columns": missing},
                evidence=evidence,
            )
        placeholders = 0
        if thr.get("reject_placeholder_drivers") and "top_drivers" in forecasts.columns:
            for value in forecasts["top_drivers"].to_list():
                text = str(value or "").lower()
                if "placeholder" in text or text in {"", "todo", "tbd", "n/a"}:
                    placeholders += 1
        metric = {"rows": forecasts.height, "placeholder_drivers": placeholders}
        if placeholders:
            return _fail(
                "R10_explainability",
                reasons=[f"Placeholder drivers found in {placeholders} rows"],
                threshold=thr,
                metric=metric,
                evidence=evidence,
            )
        return _pass("R10_explainability", threshold=thr, metric=metric, evidence=evidence)

    def _eval_R11_plot_contract(self, run_dir: Path, artifacts: dict[str, Any]) -> dict[str, Any]:
        thr = threshold_for("R11_plot_contract", self.config)
        manifest = artifacts.get("plot_manifest") or {}
        if not manifest:
            return _missing_metric_state(
                "R11_plot_contract",
                reasons=["Missing plot_manifest.json"],
                threshold=thr,
            )
        calibration = list(manifest.get("calibration") or [])
        projection = list(manifest.get("projection") or [])
        reasons: list[str] = []
        if thr.get("require_calibration_plots") and not calibration:
            reasons.append("No calibration plots in manifest")
        if thr.get("require_projection_plots") and not projection:
            reasons.append("No projection plots in manifest")
        missing_files: list[str] = []
        unlabeled = 0
        for entries in (calibration, projection):
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                rel = entry.get("path")
                if not rel or not (run_dir / str(rel)).exists():
                    missing_files.append(str(rel))
                if (
                    thr.get("require_source_ids")
                    and not entry.get("source_ids")
                    and not entry.get("source_id")
                ):
                    # Soft label check: scope fields also acceptable.
                    if not entry.get("office") and not entry.get("scope"):
                        unlabeled += 1
        if missing_files:
            reasons.append(f"Missing plot files: {missing_files[:5]}")
        metric = {
            "calibration_count": len(calibration),
            "projection_count": len(projection),
            "missing_files": missing_files,
            "unlabeled": unlabeled,
        }
        evidence = ["plot_manifest.json"]
        if reasons:
            return _fail(
                "R11_plot_contract",
                reasons=reasons,
                threshold=thr,
                metric=metric,
                evidence=evidence,
            )
        return _pass("R11_plot_contract", threshold=thr, metric=metric, evidence=evidence)

    def _eval_R12_performance_contract(
        self, run_dir: Path, artifacts: dict[str, Any]
    ) -> dict[str, Any]:
        thr = threshold_for("R12_performance_contract", self.config)
        performance = artifacts.get("performance") or {}
        if not performance:
            return _missing_metric_state(
                "R12_performance_contract",
                reasons=["Missing performance.json"],
                threshold=thr,
            )
        required = {"requested_engine", "engine", "parallel", "numba_available", "simulation_count"}
        missing = sorted(required - set(performance))
        reasons: list[str] = []
        if missing:
            reasons.append(f"Missing performance fields: {missing}")
        else:
            if performance["requested_engine"] == "numba":
                if performance["numba_available"] and performance["engine"] != "numba":
                    reasons.append("Numba requested and available but engine is not numba")
                if not performance["numba_available"] and performance["engine"] not in {
                    "python",
                    "numba",
                }:
                    reasons.append("Unexpected fallback engine")
        mcse = _finite_from(performance, "max_mcse", "mcse_max")
        max_mcse = float(thr.get("max_mcse", 0.0025))
        if mcse is None:
            # Every published probability must meet MCSE; missing metric is not a pass.
            return _missing_metric_state(
                "R12_performance_contract",
                reasons=["Missing max_mcse / mcse_max for published probabilities"],
                threshold=thr,
                evidence=["performance.json"],
                metric={
                    "engine": performance.get("engine"),
                    "requested_engine": performance.get("requested_engine"),
                    "max_mcse": None,
                },
            )
        if mcse > max_mcse:
            reasons.append(f"Max MCSE {mcse} > {max_mcse}")
        regression = _finite_number(performance.get("wall_clock_regression"))
        if regression is not None and regression > float(
            thr.get("max_wall_clock_regression", 0.10)
        ):
            if not performance.get("regression_approved"):
                reasons.append(f"Wall-clock regression {regression} exceeds tolerance")
        metric = {
            "engine": performance.get("engine"),
            "requested_engine": performance.get("requested_engine"),
            "max_mcse": mcse,
            "wall_clock_regression": regression,
        }
        evidence = ["performance.json"]
        if reasons:
            return _fail(
                "R12_performance_contract",
                reasons=reasons,
                threshold=thr,
                metric=metric,
                evidence=evidence,
            )
        return _pass("R12_performance_contract", threshold=thr, metric=metric, evidence=evidence)

    def _eval_R13_posterior_quality(
        self, run_dir: Path, artifacts: dict[str, Any]
    ) -> dict[str, Any]:
        thr = threshold_for("R13_posterior_quality", self.config)
        diagnostics = artifacts.get("posterior_diagnostics")
        if not diagnostics:
            return _missing_metric_state(
                "R13_posterior_quality",
                reasons=["Missing posterior_diagnostics.json"],
                threshold=thr,
            )
        reasons: list[str] = []
        if str(diagnostics.get("fallback_used") or "").strip():
            reasons.append("Fallback sampler path used")
        divergences = int(diagnostics.get("divergences") or 0)
        if divergences > int(thr.get("max_divergences", 0)):
            reasons.append(f"Divergences {divergences} > 0")
        draw_count = int(diagnostics.get("draw_count") or 0)
        if draw_count < int(thr.get("min_draw_count", 100)):
            reasons.append(f"Draw count {draw_count} below minimum")
        r_hat = _finite_number(diagnostics.get("r_hat_max"))
        ess = _finite_from(diagnostics, "ess_min", "bulk_ess_min")
        tail_ess = _finite_from(diagnostics, "tail_ess_min")
        e_bfmi = _finite_from(diagnostics, "e_bfmi_min", "e_bfmi")
        missing_mcmc: list[str] = []
        if r_hat is None:
            missing_mcmc.append("r_hat_max")
        if ess is None:
            missing_mcmc.append("ess_min/bulk_ess_min")
        if tail_ess is None:
            missing_mcmc.append("tail_ess_min")
        if e_bfmi is None:
            missing_mcmc.append("e_bfmi")
        if r_hat is not None and r_hat > float(thr.get("max_r_hat", 1.01)):
            reasons.append(f"R-hat max {r_hat} > {thr.get('max_r_hat')}")
        if ess is not None and ess < float(thr.get("min_bulk_ess", 400)):
            reasons.append(f"ESS min {ess} < {thr.get('min_bulk_ess')}")
        if tail_ess is not None and tail_ess < float(thr.get("min_tail_ess", 400)):
            reasons.append(f"Tail ESS min {tail_ess} < {thr.get('min_tail_ess')}")
        if e_bfmi is not None and e_bfmi <= float(thr.get("min_e_bfmi", 0.3)):
            reasons.append(f"E-BFMI {e_bfmi} <= {thr.get('min_e_bfmi')}")
        if diagnostics.get("chain_count_mislabeled") is True:
            reasons.append("Chain IDs mislabeled")
        metric = {
            "divergences": divergences,
            "draw_count": draw_count,
            "r_hat_max": r_hat,
            "ess_min": ess,
            "tail_ess_min": tail_ess,
            "e_bfmi": e_bfmi,
        }
        evidence = ["posterior_diagnostics.json"]
        if missing_mcmc:
            return _missing_metric_state(
                "R13_posterior_quality",
                reasons=[f"Missing required MCMC diagnostics: {missing_mcmc}"],
                threshold=thr,
                evidence=evidence,
                metric=metric,
            )
        if reasons:
            return _fail(
                "R13_posterior_quality",
                reasons=reasons,
                threshold=thr,
                metric=metric,
                evidence=evidence,
            )
        return _pass("R13_posterior_quality", threshold=thr, metric=metric, evidence=evidence)

    def _eval_R14_calibrated_publication(
        self, run_dir: Path, artifacts: dict[str, Any]
    ) -> dict[str, Any]:
        thr = threshold_for("R14_calibrated_publication", self.config)
        forecasts = artifacts.get("race_forecasts")
        if forecasts is None or forecasts.is_empty():
            return _missing_metric_state(
                "R14_calibrated_publication",
                reasons=["Missing race_forecasts for calibrated publication"],
                threshold=thr,
            )
        if "winner_probability" not in forecasts.columns:
            return _missing_metric_state(
                "R14_calibrated_publication",
                reasons=["winner_probability column missing"],
                threshold=thr,
            )
        published = forecasts.filter(pl.col("winner_probability").is_not_null())
        if published.is_empty():
            return _missing_metric_state(
                "R14_calibrated_publication",
                reasons=["No published probabilities to calibrate"],
                threshold=thr,
            )
        map_frame = artifacts.get("recalibration_map")
        map_present = map_frame is not None
        reasons: list[str] = []
        # Simplex / range coherence.
        bad_range = published.filter(
            (pl.col("winner_probability") < 0.0) | (pl.col("winner_probability") > 1.0)
        ).height
        if bad_range:
            reasons.append(f"{bad_range} probabilities outside [0,1]")
        # Group sum check when multi-option rows are present.
        if {"race_id", "winner_probability"}.issubset(set(published.columns)):
            if "option_id" in published.columns or "party" in published.columns:
                sums = published.group_by("race_id").agg(
                    pl.col("winner_probability").sum().alias("s")
                )
                # Fail multi-row races whose probabilities do not sum near 1.
                multi = (
                    published.group_by("race_id").agg(pl.len().alias("n")).filter(pl.col("n") > 1)
                )
                if multi.height:
                    bad_sums = sums.join(multi, on="race_id", how="inner").filter(
                        (pl.col("s") < 0.99) | (pl.col("s") > 1.01)
                    )
                    if bad_sums.height:
                        reasons.append(f"{bad_sums.height} races fail probability simplex sum")
        lineage = artifacts.get("nested_eval") or {}
        map_lineage_ok = True
        if thr.get("require_outer_fold_map") and map_present:
            map_lineage_ok = bool(lineage.get("calibration_map_outer_fold", True))
            if lineage.get("calibration_map_outer_fold") is False:
                reasons.append("Calibration map is not outer-fold compatible")
            if lineage.get("calibration_map_engine_mismatch") is True:
                reasons.append("Calibration map engine/source mismatch")
                map_lineage_ok = False
        metric = {
            "map_present": map_present,
            "published_rows": published.height,
            "out_of_range": bad_range,
            "map_lineage_ok": map_lineage_ok,
        }
        evidence = ["race_forecasts.parquet"]
        if map_present:
            evidence.append("recalibration_map.parquet")
        if reasons:
            return _fail(
                "R14_calibrated_publication",
                reasons=reasons,
                threshold=thr,
                metric=metric,
                evidence=evidence,
            )
        return _pass("R14_calibrated_publication", threshold=thr, metric=metric, evidence=evidence)

    def _eval_R15_daily_update_quality(
        self, run_dir: Path, artifacts: dict[str, Any]
    ) -> dict[str, Any]:
        thr = threshold_for("R15_daily_update_quality", self.config)
        update = artifacts.get("daily_update") or {}
        if not update:
            return _missing_metric_state(
                "R15_daily_update_quality",
                reasons=["No latest_daily_update.json present"],
                threshold=thr,
            )
        evidence = ["latest_daily_update.json"]
        reasons: list[str] = []
        if update.get("noop") is True or update.get("strategy") == "noop":
            reasons.append("Update is a no-op without likelihood reweighting")
        if bool(update.get("needs_full_refit")) and not bool(update.get("full_refit_executed")):
            reasons.append("Full refit required but not executed")
        if update.get("quality_passed") is False:
            reasons.append("Update quality_passed is false")
        mae = _finite_from(update, "probability_mae_vs_full_refit", "mae_vs_refit")
        max_diff = _finite_from(update, "probability_max_diff_vs_full_refit", "max_diff_vs_refit")
        if mae is not None and mae > float(thr.get("max_probability_mae_vs_refit", 0.005)):
            reasons.append(f"MAE vs full refit {mae} exceeds threshold")
        if max_diff is not None and max_diff > float(
            thr.get("max_probability_diff_vs_refit", 0.02)
        ):
            reasons.append(f"Max diff vs full refit {max_diff} exceeds threshold")
        if update.get("weights_degenerate") is True and not update.get("full_refit_executed"):
            reasons.append("Degenerate weights without full refit")
        metric = {
            "quality_passed": update.get("quality_passed"),
            "needs_full_refit": update.get("needs_full_refit"),
            "mae_vs_refit": mae,
            "max_diff_vs_refit": max_diff,
            "strategy": update.get("strategy"),
        }
        if reasons:
            return _fail(
                "R15_daily_update_quality",
                reasons=reasons,
                threshold=thr,
                metric=metric,
                evidence=evidence,
            )
        # Require both refit-comparison metrics; quality_passed alone is not enough.
        if mae is None or max_diff is None or update.get("quality_passed") is not True:
            return _missing_metric_state(
                "R15_daily_update_quality",
                reasons=[
                    "Update present but quality_passed and/or MAE/max-diff vs full "
                    "refit metrics incomplete"
                ],
                threshold=thr,
                evidence=evidence,
                metric=metric,
            )
        return _pass("R15_daily_update_quality", threshold=thr, metric=metric, evidence=evidence)

    def _eval_R16_real_data_exclusivity(
        self, run_dir: Path, artifacts: dict[str, Any]
    ) -> dict[str, Any]:
        thr = threshold_for("R16_real_data_exclusivity", self.config)
        audit = artifacts.get("registry_audit") or {}
        manifest = artifacts.get("source_manifest")
        evidence: list[str] = []
        if (run_dir / "source_registry_audit.json").exists():
            evidence.append("source_registry_audit.json")
        if manifest is not None:
            evidence.append("source_manifest.parquet")
        synthetic = 0
        fixture_sources = 0
        if audit:
            synthetic = int(audit.get("synthetic_rows") or audit.get("synthetic_row_count") or 0)
            fixture_sources = int(audit.get("fixture_source_count") or 0)
        elif manifest is not None:
            if "source_class" in manifest.columns:
                fixture_sources = int(
                    manifest.filter(
                        pl.col("source_class").is_in(["fixture", "synthetic", "generated"])
                    ).height
                )
            if "is_synthetic" in manifest.columns:
                synthetic = int(manifest.filter(pl.col("is_synthetic") == True).height)  # noqa: E712
            if "source_id" in manifest.columns:
                fixture_like = manifest.filter(
                    pl.col("source_id")
                    .cast(pl.Utf8)
                    .str.contains("(?i)fixture|synthetic|generated")
                ).height
                fixture_sources = max(fixture_sources, int(fixture_like))
        else:
            return _missing_metric_state(
                "R16_real_data_exclusivity",
                reasons=["Missing registry audit and source_manifest for real-data exclusivity"],
                threshold=thr,
                evidence=evidence,
            )
        reasons: list[str] = []
        if synthetic > int(thr.get("max_synthetic_rows", 0)):
            reasons.append(f"Synthetic rows present: {synthetic}")
        if fixture_sources > int(thr.get("max_fixture_sources", 0)):
            reasons.append(f"Fixture/synthetic sources present: {fixture_sources}")
        # Without an explicit production audit, do not claim real-data exclusivity.
        if not audit and fixture_sources == 0 and synthetic == 0:
            # Ambiguous fixture pipeline: insufficient_evidence for production claims.
            return _missing_metric_state(
                "R16_real_data_exclusivity",
                reasons=["No production registry audit proving zero synthetic/fixture inputs"],
                threshold=thr,
                evidence=evidence,
                metric={"synthetic_rows": synthetic, "fixture_sources": fixture_sources},
            )
        metric = {"synthetic_rows": synthetic, "fixture_sources": fixture_sources}
        if reasons:
            return _fail(
                "R16_real_data_exclusivity",
                reasons=reasons,
                threshold=thr,
                metric=metric,
                evidence=evidence,
            )
        return _pass("R16_real_data_exclusivity", threshold=thr, metric=metric, evidence=evidence)

    def _eval_R17_as_of_integrity(self, run_dir: Path, artifacts: dict[str, Any]) -> dict[str, Any]:
        thr = threshold_for("R17_as_of_integrity", self.config)
        audit = artifacts.get("as_of_audit") or {}
        if not audit:
            return _missing_metric_state(
                "R17_as_of_integrity",
                reasons=["Missing as_of_audit.json"],
                threshold=thr,
            )
        future_rows = int(audit.get("future_eligible_rows") or audit.get("violations") or 0)
        metric = {
            "future_eligible_rows": future_rows,
            "time_travel_canaries_passed": audit.get("time_travel_canaries_passed"),
        }
        evidence = ["as_of_audit.json"]
        if future_rows > int(thr.get("max_future_eligible_rows", 0)):
            return _fail(
                "R17_as_of_integrity",
                reasons=[f"Future-eligible rows: {future_rows}"],
                threshold=thr,
                metric=metric,
                evidence=evidence,
            )
        if audit.get("time_travel_canaries_passed") is False:
            return _fail(
                "R17_as_of_integrity",
                reasons=["Time-travel canaries failed"],
                threshold=thr,
                metric=metric,
                evidence=evidence,
            )
        return _pass("R17_as_of_integrity", threshold=thr, metric=metric, evidence=evidence)

    def _eval_R18_nested_evaluation(
        self, run_dir: Path, artifacts: dict[str, Any]
    ) -> dict[str, Any]:
        thr = threshold_for("R18_nested_evaluation", self.config)
        nested = artifacts.get("nested_eval") or {}
        if not nested:
            return _missing_metric_state(
                "R18_nested_evaluation",
                reasons=["Missing nested_evaluation.json"],
                threshold=thr,
            )
        reasons: list[str] = []
        if thr.get("require_exact_pipeline") and not nested.get("exact_pipeline"):
            reasons.append("Exact publication pipeline not used in outer folds")
        if thr.get("require_outer_cycle_exclusion") and not nested.get("outer_cycle_excluded"):
            reasons.append("Outer cycle not excluded from all fitting/promotion")
        if nested.get("held_out_permutation_affects_prior_folds") is True:
            reasons.append("Permuting held-out outcomes changed prior-fold forecasts")
        metric = {
            "exact_pipeline": nested.get("exact_pipeline"),
            "outer_cycle_excluded": nested.get("outer_cycle_excluded"),
            "fold_count": nested.get("fold_count"),
        }
        evidence = ["nested_evaluation.json"]
        if reasons:
            return _fail(
                "R18_nested_evaluation",
                reasons=reasons,
                threshold=thr,
                metric=metric,
                evidence=evidence,
            )
        return _pass("R18_nested_evaluation", threshold=thr, metric=metric, evidence=evidence)

    def _eval_R19_covariance_recovery(
        self, run_dir: Path, artifacts: dict[str, Any]
    ) -> dict[str, Any]:
        thr = threshold_for("R19_covariance_recovery", self.config)
        report = artifacts.get("covariance_recovery") or {}
        if not report:
            return _missing_metric_state(
                "R19_covariance_recovery",
                reasons=["Missing covariance_recovery.json"],
                threshold=thr,
            )
        reasons: list[str] = []
        if thr.get("require_psd") and report.get("is_psd") is False:
            reasons.append("Covariance matrix is not PSD")
        if report.get("complement_averaging") is True:
            reasons.append("Residuals used complement averaging")
        if report.get("one_signed_residual_per_race") is False:
            reasons.append("Not one signed residual per race")
        rel_err = _finite_number(report.get("max_factor_variance_rel_error"))
        corr_rmse = _finite_number(report.get("correlation_rmse"))
        metric = {
            "is_psd": report.get("is_psd"),
            "max_factor_variance_rel_error": rel_err,
            "correlation_rmse": corr_rmse,
        }
        evidence = ["covariance_recovery.json"]
        # Hard scientific failures first.
        if reasons:
            return _fail(
                "R19_covariance_recovery",
                reasons=reasons,
                threshold=thr,
                metric=metric,
                evidence=evidence,
            )
        if rel_err is None or corr_rmse is None or report.get("is_psd") is None:
            return _missing_metric_state(
                "R19_covariance_recovery",
                reasons=[
                    "Missing PSD status and/or factor variance rel error / "
                    "correlation RMSE required for recovery tolerances"
                ],
                threshold=thr,
                evidence=evidence,
                metric=metric,
            )
        if rel_err > float(thr.get("max_factor_variance_rel_error", 0.20)):
            reasons.append(f"Factor variance rel error {rel_err} exceeds threshold")
        if corr_rmse > float(thr.get("max_correlation_rmse", 0.10)):
            reasons.append(f"Correlation RMSE {corr_rmse} exceeds threshold")
        if reasons:
            return _fail(
                "R19_covariance_recovery",
                reasons=reasons,
                threshold=thr,
                metric=metric,
                evidence=evidence,
            )
        return _pass("R19_covariance_recovery", threshold=thr, metric=metric, evidence=evidence)

    def _eval_R20_all_race_hierarchy(
        self, run_dir: Path, artifacts: dict[str, Any]
    ) -> dict[str, Any]:
        thr = threshold_for("R20_all_race_hierarchy", self.config)
        report = artifacts.get("hierarchy_recovery") or {}
        if not report:
            return _missing_metric_state(
                "R20_all_race_hierarchy",
                reasons=["Missing hierarchy_recovery.json"],
                threshold=thr,
            )
        reasons: list[str] = []
        if report.get("all_control_bearing_races_in_model") is False:
            reasons.append("Control-bearing races missing from joint model")
        if thr.get("require_unpolled_propagation") and not report.get(
            "unpolled_propagation_passed"
        ):
            reasons.append("Unpolled race national propagation failed")
        if thr.get("require_label_symmetry") and report.get("label_symmetry_passed") is False:
            reasons.append("Label symmetry failed")
        metric = {
            "all_control_bearing_races_in_model": report.get("all_control_bearing_races_in_model"),
            "unpolled_propagation_passed": report.get("unpolled_propagation_passed"),
            "label_symmetry_passed": report.get("label_symmetry_passed"),
        }
        evidence = ["hierarchy_recovery.json"]
        if reasons:
            return _fail(
                "R20_all_race_hierarchy",
                reasons=reasons,
                threshold=thr,
                metric=metric,
                evidence=evidence,
            )
        return _pass("R20_all_race_hierarchy", threshold=thr, metric=metric, evidence=evidence)

    def _eval_R21_poll_observation_identity(
        self, run_dir: Path, artifacts: dict[str, Any]
    ) -> dict[str, Any]:
        thr = threshold_for("R21_poll_observation_identity", self.config)
        manifest = artifacts.get("poll_observation_manifest") or {}
        if not manifest:
            return _missing_metric_state(
                "R21_poll_observation_identity",
                reasons=["Missing poll_observation_manifest.json"],
                threshold=thr,
            )
        double_count = int(manifest.get("double_count_rows") or 0)
        reasons: list[str] = []
        if double_count > int(thr.get("max_double_count_rows", 0)):
            reasons.append(f"Double-counted poll rows: {double_count}")
        if manifest.get("option_double_count") is True:
            reasons.append("Option double count detected")
        metric = {
            "double_count_rows": double_count,
            "unique_questions": manifest.get("unique_questions"),
            "pollster_lineage_auditable": manifest.get("pollster_lineage_auditable"),
        }
        evidence = ["poll_observation_manifest.json"]
        if reasons:
            return _fail(
                "R21_poll_observation_identity",
                reasons=reasons,
                threshold=thr,
                metric=metric,
                evidence=evidence,
            )
        return _pass(
            "R21_poll_observation_identity", threshold=thr, metric=metric, evidence=evidence
        )

    def _eval_R22_feature_validity(
        self, run_dir: Path, artifacts: dict[str, Any]
    ) -> dict[str, Any]:
        thr = threshold_for("R22_feature_validity", self.config)
        lineage = artifacts.get("feature_lineage") or {}
        if not lineage:
            return _missing_metric_state(
                "R22_feature_validity",
                reasons=["Missing feature_lineage.json"],
                threshold=thr,
            )
        reasons: list[str] = []
        max_snaps = int(lineage.get("max_snapshots_per_feature_key") or 0)
        if max_snaps > int(thr.get("max_snapshots_per_feature_key", 1)):
            reasons.append(f"Multiple snapshots per feature key: {max_snaps}")
        if (
            thr.get("require_incumbent_relative_sign")
            and lineage.get("incumbent_relative_sign") is False
        ):
            reasons.append("Economic features not incumbent-party relative")
        if lineage.get("end_of_cycle_finance_in_early_fold") is True:
            reasons.append("End-of-cycle finance entered early folds")
        if lineage.get("revised_macro_in_early_fold") is True:
            reasons.append("Revised macro values entered early folds")
        metric = {
            "max_snapshots_per_feature_key": max_snaps,
            "incumbent_relative_sign": lineage.get("incumbent_relative_sign"),
        }
        evidence = ["feature_lineage.json"]
        if reasons:
            return _fail(
                "R22_feature_validity",
                reasons=reasons,
                threshold=thr,
                metric=metric,
                evidence=evidence,
            )
        return _pass("R22_feature_validity", threshold=thr, metric=metric, evidence=evidence)

    def _eval_R23_joint_outcome_coherence(
        self, run_dir: Path, artifacts: dict[str, Any]
    ) -> dict[str, Any]:
        thr = threshold_for("R23_joint_outcome_coherence", self.config)
        semantic = artifacts.get("semantic_verification") or {}
        forecasts = artifacts.get("race_forecasts")
        evidence: list[str] = []
        if (run_dir / "semantic_verification.json").exists():
            evidence.append("semantic_verification.json")
        if forecasts is not None:
            evidence.append("race_forecasts.parquet")
        reasons: list[str] = []
        metric: dict[str, Any] = {}
        if semantic:
            if semantic.get("passed") is False:
                reasons.extend(
                    list(semantic.get("failure_reasons") or ["semantic verification failed"])
                )
            metric = {
                "semantic_passed": semantic.get("passed"),
                "reconciliation_ok": semantic.get("reconciliation_ok"),
            }
            if (
                thr.get("require_exact_reconciliation")
                and semantic.get("reconciliation_ok") is False
            ):
                reasons.append("Exact race/control reconciliation failed")
        elif forecasts is not None and "winner_probability" in forecasts.columns:
            published = forecasts.filter(pl.col("winner_probability").is_not_null())
            bad = published.filter(
                (pl.col("winner_probability") < float(thr.get("min_probability", 0.0)))
                | (pl.col("winner_probability") > float(thr.get("max_probability", 1.0)))
            ).height
            metric = {"out_of_range": bad, "published_rows": published.height}
            if bad:
                reasons.append(f"{bad} probabilities outside allowed range")
            # Without full reconciliation artifact, production cannot claim pass.
            return _missing_metric_state(
                "R23_joint_outcome_coherence",
                reasons=[
                    "Probabilities range-checked but full joint reconciliation artifact missing"
                ],
                threshold=thr,
                evidence=evidence,
                metric=metric,
            )
        else:
            return _missing_metric_state(
                "R23_joint_outcome_coherence",
                reasons=["Missing semantic_verification and race_forecasts"],
                threshold=thr,
                evidence=evidence,
            )
        if reasons:
            return _fail(
                "R23_joint_outcome_coherence",
                reasons=reasons,
                threshold=thr,
                metric=metric,
                evidence=evidence,
            )
        return _pass("R23_joint_outcome_coherence", threshold=thr, metric=metric, evidence=evidence)

    def _eval_R24_atomic_publication(
        self, run_dir: Path, artifacts: dict[str, Any]
    ) -> dict[str, Any]:
        thr = threshold_for("R24_atomic_publication", self.config)
        decision = artifacts.get("publication_decision") or {}
        promotion = artifacts.get("promotion_manifest") or {}
        evidence = [
            p
            for p in ("publication_decision.json", "promotion_manifest.json")
            if (run_dir / p).exists()
        ]
        claimed_mode = str(
            decision.get("publication_mode")
            or promotion.get("publication_mode")
            or self._infer_publication_mode(run_dir)
        )
        # Research/fixture/shadow without promotion is fine if not claiming production.
        if claimed_mode in {"research", "fixture", "shadow"} and not promotion.get("verified"):
            return _pass(
                "R24_atomic_publication",
                threshold=thr,
                metric={
                    "publication_mode": claimed_mode,
                    "promotion_verified": False,
                    "policy": "non-production modes need no promotion manifest",
                },
                evidence=evidence,
            )
        if claimed_mode == "production":
            if thr.get("require_verified_promotion_manifest") and not promotion.get("verified"):
                return _fail(
                    "R24_atomic_publication",
                    reasons=["Production publication_mode without verified promotion_manifest"],
                    threshold=thr,
                    metric={"publication_mode": claimed_mode, "promotion_verified": False},
                    evidence=evidence,
                )
            if thr.get("require_immutable_attempt") and not (
                promotion.get("attempt_id") or decision.get("attempt_id")
            ):
                return _fail(
                    "R24_atomic_publication",
                    reasons=["Production promotion missing immutable attempt_id"],
                    threshold=thr,
                    metric={"publication_mode": claimed_mode},
                    evidence=evidence,
                )
        if not decision and not promotion:
            return _missing_metric_state(
                "R24_atomic_publication",
                reasons=["Missing publication_decision and promotion_manifest"],
                threshold=thr,
                evidence=evidence,
            )
        return _pass(
            "R24_atomic_publication",
            threshold=thr,
            metric={
                "publication_mode": claimed_mode,
                "promotion_verified": bool(promotion.get("verified")),
                "attempt_id": promotion.get("attempt_id") or decision.get("attempt_id"),
            },
            evidence=evidence,
        )

    def _eval_R25_live_source_resilience(
        self, run_dir: Path, artifacts: dict[str, Any]
    ) -> dict[str, Any]:
        thr = threshold_for("R25_live_source_resilience", self.config)
        canaries = artifacts.get("live_source_canaries") or {}
        if not canaries:
            return _missing_metric_state(
                "R25_live_source_resilience",
                reasons=["Missing live_source_canaries.json"],
                threshold=thr,
            )
        reasons: list[str] = []
        if thr.get("require_canary_history") and not canaries.get("history"):
            reasons.append("Canary history empty")
        if canaries.get("all_passed") is False:
            reasons.append("One or more live-source canaries failed")
        failures = list(canaries.get("injected_failure_results") or [])
        for item in failures:
            if isinstance(item, dict) and item.get("status") in {None, "success", "ok"}:
                reasons.append(f"Injected failure incorrectly marked success: {item.get('name')}")
        metric = {
            "all_passed": canaries.get("all_passed"),
            "history_count": len(canaries.get("history") or []),
        }
        evidence = ["live_source_canaries.json"]
        if reasons:
            return _fail(
                "R25_live_source_resilience",
                reasons=reasons,
                threshold=thr,
                metric=metric,
                evidence=evidence,
            )
        return _pass("R25_live_source_resilience", threshold=thr, metric=metric, evidence=evidence)

    def _eval_R26_benchmark_superiority(
        self, run_dir: Path, artifacts: dict[str, Any]
    ) -> dict[str, Any]:
        thr = threshold_for("R26_benchmark_superiority", self.config)
        report = artifacts.get("benchmark_superiority") or {}
        nested = artifacts.get("nested_eval") or {}
        if not report and not nested.get("paired_benchmarks"):
            return _missing_metric_state(
                "R26_benchmark_superiority",
                reasons=["Missing benchmark_superiority / preregistered paired scores"],
                threshold=thr,
            )
        payload = report or dict(nested.get("paired_benchmarks") or {})
        reasons: list[str] = []
        if payload.get("preregistered") is False:
            reasons.append("Comparisons were not preregistered")
        if payload.get("metric_changed_after_scoring") is True:
            reasons.append("Metric changed after scoring")
        if payload.get("difficult_cycle_removed") is True:
            reasons.append("Difficult cycle removed from claim")
        if payload.get("scope_mismatch") is True:
            reasons.append("Comparator scope mismatch")
        cycles = _finite_number(payload.get("independent_cycle_count"))
        min_cycles = float(thr.get("min_independent_cycles", 6))
        if cycles is not None and cycles < min_cycles:
            reasons.append(f"Independent cycles {cycles} < {min_cycles}")
        if cycles is None:
            return _missing_metric_state(
                "R26_benchmark_superiority",
                reasons=["Independent cycle count missing for superiority claim"],
                threshold=thr,
                evidence=[
                    p
                    for p in ("benchmark_superiority.json", "nested_evaluation.json")
                    if (run_dir / p).exists()
                ],
                metric=payload if isinstance(payload, dict) else {},
            )
        gap = _finite_number(payload.get("max_comparator_log_score_gap"))
        if gap is None:
            return _missing_metric_state(
                "R26_benchmark_superiority",
                reasons=["Missing max_comparator_log_score_gap for superiority claim"],
                threshold=thr,
                evidence=[
                    p
                    for p in ("benchmark_superiority.json", "nested_evaluation.json")
                    if (run_dir / p).exists()
                ],
                metric={
                    "independent_cycle_count": cycles,
                    "max_comparator_log_score_gap": None,
                    "preregistered": payload.get("preregistered"),
                },
            )
        if gap > float(thr.get("max_comparator_log_score_gap", 0.005)):
            reasons.append(f"Comparator gap {gap} exceeds threshold")
        if payload.get("beats_all_simple_baselines") is False:
            reasons.append("Does not beat all simple baselines")
        if payload.get("best_evidenced_claim") is True and reasons:
            # claim invalidated
            pass
        metric = {
            "independent_cycle_count": cycles,
            "max_comparator_log_score_gap": gap,
            "preregistered": payload.get("preregistered"),
        }
        evidence = [
            p
            for p in ("benchmark_superiority.json", "nested_evaluation.json")
            if (run_dir / p).exists()
        ]
        if reasons:
            return _fail(
                "R26_benchmark_superiority",
                reasons=reasons,
                threshold=thr,
                metric=metric,
                evidence=evidence,
            )
        return _pass("R26_benchmark_superiority", threshold=thr, metric=metric, evidence=evidence)

    def _eval_R27_contract_parity(self, run_dir: Path, artifacts: dict[str, Any]) -> dict[str, Any]:
        thr = threshold_for("R27_contract_parity", self.config)
        report = artifacts.get("contract_parity") or {}
        if not report:
            return _missing_metric_state(
                "R27_contract_parity",
                reasons=["Missing contract_parity.json"],
                threshold=thr,
            )
        stale = int(report.get("stale_claims") or 0)
        reasons: list[str] = []
        if stale > int(thr.get("max_stale_claims", 0)):
            reasons.append(f"Stale documentation claims: {stale}")
        if report.get("passed") is False:
            reasons.extend(list(report.get("failure_reasons") or ["contract parity failed"]))
        metric = {
            "stale_claims": stale,
            "checked_documents": report.get("checked_documents"),
        }
        evidence = ["contract_parity.json"]
        if reasons:
            return _fail(
                "R27_contract_parity",
                reasons=reasons,
                threshold=thr,
                metric=metric,
                evidence=evidence,
            )
        return _pass("R27_contract_parity", threshold=thr, metric=metric, evidence=evidence)


def legacy_passed_from_state(state: str) -> bool | None:
    """Map v2 state onto legacy reward_card passed field."""
    if state == "pass":
        return True
    if state == "fail":
        return False
    if state == "not_applicable":
        return True
    # insufficient_evidence
    return None
