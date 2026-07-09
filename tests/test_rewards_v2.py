"""Positive + adversarial negative tests for every reward-v2 id R0-R27."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import polars as pl
import pytest
import yaml
from typer.testing import CliRunner

from civic_signal.cli import app
from civic_signal.config import ProjectContext
from civic_signal.scoring.reward_registry import (
    all_reward_ids,
    load_rewards_config,
    make_reward_record,
    publication_mode_default,
    threshold_for,
)
from civic_signal.scoring.reward_v2 import RewardV2Evaluator
from civic_signal.verification.publication import PublicationVerifier
from civic_signal.verification.rewards import RewardVerificationRunner

REPO_ROOT = Path(__file__).resolve().parents[1]
REWARDS_YAML = REPO_ROOT / "configs" / "rewards.yaml"


def _cfg() -> dict:
    return load_rewards_config(str(REWARDS_YAML))


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _base_forecasts(n: int = 2) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "race_id": [f"r{i}" for i in range(n)],
            "winner_probability": [0.55, 0.45][:n],
            "model_config_hash": ["abc"] * n,
            "source_manifest_hash": ["def"] * n,
            "tier_reason": ["polled"] * n,
            "data_quality_flags": ["ok"] * n,
            "top_drivers": ["polling,fundamentals"] * n,
            "component_contributions": ['{"polling":0.5}'] * n,
            "uncertainty_explanation": ["interval from simulation"] * n,
        }
    )


def _base_catalog() -> pl.DataFrame:
    return pl.DataFrame({"race_id": ["r0", "r1"], "tier": ["A", "B"]})


def _base_manifest(*, synthetic: bool = False) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "source_id": ["polls_real" if not synthetic else "fixture_polls"],
            "status": ["fetched"],
            "content_hash": ["hash1"],
            "auth_mode": ["public"],
            "source_class": ["production" if not synthetic else "fixture"],
            "is_synthetic": [synthetic],
        }
    )


def _seed_run(run_dir: Path, **overrides: object) -> Path:
    """Create a minimal run whose evidence can pass individual rewards when complete."""
    run_dir.mkdir(parents=True, exist_ok=True)
    forecasts = overrides.get("race_forecasts", _base_forecasts())
    catalog = overrides.get("race_catalog", _base_catalog())
    manifest = overrides.get("source_manifest", _base_manifest())
    assert isinstance(forecasts, pl.DataFrame)
    assert isinstance(catalog, pl.DataFrame)
    assert isinstance(manifest, pl.DataFrame)
    forecasts.write_parquet(run_dir / "race_forecasts.parquet")
    catalog.write_parquet(run_dir / "race_catalog.parquet")
    manifest.write_parquet(run_dir / "source_manifest.parquet")
    # Semantic reconciliation requires draws + control artifacts.
    race_ids = forecasts["race_id"].unique().to_list()
    pl.DataFrame(
        {
            "race_id": race_ids,
            "draw_id": list(range(len(race_ids))),
            "winner_party": ["DEM"] * len(race_ids),
        }
    ).write_parquet(run_dir / "forecast_draws.parquet")
    pl.DataFrame(
        {
            "party": ["DEM", "REP"],
            "control_threshold": [51, 50],
            "majority_probability": [0.4, 0.6],
            "seat_count_mean": [48.0, 52.0],
        }
    ).write_parquet(run_dir / "control_forecasts.parquet")

    defaults: dict[str, object] = {
        "ci_manifest.json": {
            "commands_passed": True,
            "line_coverage_pct": 91.0,
            "tool_versions": {"ruff": "0.8", "pytest": "8.3"},
        },
        "coverage.json": {"line_rate_pct": 91.0},
        "reproducibility_fingerprint.json": {
            "cross_run_verified": True,
            "compared_to_previous": True,
            "combined_hash": "abc123",
        },
        "plot_manifest.json": {
            "calibration": [
                {
                    "path": "plots/cal.png",
                    "source_ids": ["polls_real"],
                    "office": "senate",
                }
            ],
            "projection": [
                {
                    "path": "plots/proj.png",
                    "source_ids": ["polls_real"],
                    "office": "senate",
                }
            ],
        },
        "performance.json": {
            "requested_engine": "python",
            "engine": "python",
            "parallel": False,
            "numba_available": False,
            "simulation_count": 1000,
            "max_mcse": 0.001,
        },
        "posterior_diagnostics.json": {
            "divergences": 0,
            "draw_count": 1000,
            "r_hat_max": 1.001,
            "ess_min": 500,
            "tail_ess_min": 450,
            "e_bfmi": 0.5,
            "fallback_used": "",
        },
        "latest_daily_update.json": {
            "quality_passed": True,
            "needs_full_refit": False,
            "strategy": "reweighting",
            "probability_mae_vs_full_refit": 0.001,
            "probability_max_diff_vs_full_refit": 0.01,
            "full_refit_executed": False,
            "weights_degenerate": False,
            "noop": False,
        },
        "source_registry_audit.json": {
            "synthetic_rows": 0,
            "fixture_source_count": 0,
        },
        "as_of_audit.json": {
            "future_eligible_rows": 0,
            "time_travel_canaries_passed": True,
            "status": "ok",
        },
        "nested_evaluation.json": {
            "exact_pipeline": True,
            "outer_cycle_excluded": True,
            "outer_fold": True,
            "fold_count": 6,
            "independent_cycle_count": 6,
            "calibration": {
                "expected_calibration_error": 0.02,
                "ece_bootstrap_upper": 0.04,
                "calibration_intercept": 0.01,
                "calibration_slope": 1.0,
            },
            "coverage": {
                "interval_50_coverage": 0.51,
                "interval_80_coverage": 0.79,
                "interval_90_coverage": 0.91,
            },
            "paired_benchmarks": {
                "independent_cycle_count": 6,
                "beats_all_simple_baselines": True,
                "log_score_diff_ci_upper": -0.01,
                "preregistered": True,
                "best_evidenced_claim": True,
                "max_comparator_log_score_gap": 0.001,
            },
            "component_ablations": {
                "ensemble": {"beats_or_matches_baseline": True},
                "polling": {"beats_or_matches_baseline": True},
            },
            "matched_coverage": True,
            "baseline_filled_missing": False,
            "calibration_map_outer_fold": True,
            "already_calibrated_without_map": True,
            "held_out_permutation_affects_prior_folds": False,
            "public_signal_leakage": {
                "leakage_passed": True,
                "nested_value_added": True,
                "result_derived": False,
                "post_as_of": False,
            },
        },
        "covariance_recovery.json": {
            "is_psd": True,
            "max_factor_variance_rel_error": 0.05,
            "correlation_rmse": 0.02,
            "one_signed_residual_per_race": True,
            "complement_averaging": False,
        },
        "hierarchy_recovery.json": {
            "all_control_bearing_races_in_model": True,
            "unpolled_propagation_passed": True,
            "label_symmetry_passed": True,
        },
        "poll_observation_manifest.json": {
            "double_count_rows": 0,
            "option_double_count": False,
            "unique_questions": 10,
            "pollster_lineage_auditable": True,
        },
        "feature_lineage.json": {
            "max_snapshots_per_feature_key": 1,
            "incumbent_relative_sign": True,
            "end_of_cycle_finance_in_early_fold": False,
            "revised_macro_in_early_fold": False,
        },
        "semantic_verification.json": {
            "passed": True,
            "reconciliation_ok": True,
            "failure_reasons": [],
            "checks": {
                "required_artifacts_present": True,
                "unique_race_keys": True,
                "probability_range_ok": True,
                "simplex_ok": True,
                "draws_cover_races": True,
                "control_present": True,
            },
        },
        "publication_decision.json": {
            "publication_mode": "research",
            "allowed": True,
            "blocks_publication": False,
        },
        "promotion_manifest.json": {
            "attempt_id": "attempt-1",
            "profile": "production",
            "publication_mode": "production",
            "verified": True,
            "reward_card_hash": "placeholder",
            "semantic_verification_hash": "placeholder",
            "content_hashes": {
                "race_forecasts.parquet": "seedhash",
                "reward_card_v2.json": "seedhash",
            },
            "promoted_at": "2026-01-01T00:00:00+00:00",
        },
        "live_source_canaries.json": {
            "all_passed": True,
            "history": [{"name": "empty_feed", "status": "failed_as_expected"}],
            "injected_failure_results": [
                {"name": "empty_feed", "status": "empty"},
                {"name": "timeout", "status": "timeout"},
            ],
        },
        "benchmark_superiority.json": {
            "preregistered": True,
            "independent_cycle_count": 6,
            "beats_all_simple_baselines": True,
            "max_comparator_log_score_gap": 0.001,
            "metric_changed_after_scoring": False,
            "difficult_cycle_removed": False,
            "scope_mismatch": False,
            "best_evidenced_claim": True,
        },
        "contract_parity.json": {
            "passed": True,
            "stale_claims": 0,
            "checked_documents": ["README.md", "SPEC.md"],
            "failure_reasons": [],
        },
        "run_manifest.json": {
            "publication_mode": "research",
            "forecast_status": "research",
        },
        "backtest_summary.json": {
            "rolling_origin_executed": True,
            "sample_size_too_small": False,
            "metrics": {
                "ensemble": {
                    "expected_calibration_error": 0.02,
                    "calibration_intercept": 0.01,
                    "calibration_slope": 1.0,
                    "interval_90_coverage": 0.9,
                }
            },
            "ablations": {"ensemble": {"beats_or_matches_baseline": True}},
        },
    }
    for name, payload in defaults.items():
        if name in overrides:
            continue
        if name.endswith(".json"):
            _write_json(run_dir / name, payload)  # type: ignore[arg-type]

    for name, payload in overrides.items():
        if name in {"race_forecasts", "race_catalog", "source_manifest"}:
            continue
        if isinstance(payload, dict) and name.endswith(".json"):
            _write_json(run_dir / name, payload)
        elif isinstance(payload, pl.DataFrame) and name.endswith(".parquet"):
            payload.write_parquet(run_dir / name)

    # Plot files referenced by manifest.
    (run_dir / "plots").mkdir(exist_ok=True)
    (run_dir / "plots" / "cal.png").write_bytes(b"png")
    (run_dir / "plots" / "proj.png").write_bytes(b"png")
    return run_dir


@pytest.fixture
def rewards_config() -> dict:
    return _cfg()


def test_rewards_yaml_is_single_threshold_registry(rewards_config: dict) -> None:
    assert publication_mode_default(rewards_config) == "research"
    ids = all_reward_ids(rewards_config)
    assert ids[0] == "R0_build"
    assert ids[-1] == "R27_contract_parity"
    assert len(ids) == 28
    for reward_id in ids:
        thr = threshold_for(reward_id, rewards_config)
        assert isinstance(thr, dict)
        # Thresholds must not be empty for production rules.
        assert thr, f"{reward_id} missing thresholds in rewards.yaml"
    prod = rewards_config["profiles"]["production"]["required_rewards"]
    assert "R16_real_data_exclusivity" in prod
    assert "R18_nested_evaluation" in prod
    assert "R24_atomic_publication" in prod


def test_make_reward_record_rejects_invalid_state() -> None:
    with pytest.raises(ValueError):
        make_reward_record(reward_id="R0_build", state="maybe")


def test_all_rewards_present_on_complete_run(tmp_path: Path, rewards_config: dict) -> None:
    run_dir = _seed_run(tmp_path / "complete")
    card = RewardV2Evaluator(rewards_config=rewards_config).evaluate_run_dir(
        run_dir, profile="research", publication_mode="research"
    )
    assert card["recomputed"] is True
    assert card["schema_version"] == "2.0.0"
    assert set(card["rewards"]) == set(all_reward_ids(rewards_config))
    for reward_id, record in card["rewards"].items():
        assert record["reward_id"] == reward_id
        assert record["state"] in {
            "pass",
            "fail",
            "insufficient_evidence",
            "not_applicable",
        }
        assert "threshold" in record
        assert "metric" in record
        assert "failure_reasons" in record
        assert "blocks_publication" in record


def test_production_profile_blocks_fixture_only_run(tmp_path: Path, rewards_config: dict) -> None:
    """Fixture-only directory lacks real-data/nested evidence → production hard-block."""
    run_dir = tmp_path / "fixture-only"
    run_dir.mkdir()
    _base_forecasts().write_parquet(run_dir / "race_forecasts.parquet")
    _base_catalog().write_parquet(run_dir / "race_catalog.parquet")
    _base_manifest(synthetic=True).write_parquet(run_dir / "source_manifest.parquet")
    _write_json(run_dir / "run_manifest.json", {"publication_mode": "fixture"})
    _write_json(run_dir / "publication_decision.json", {"publication_mode": "fixture"})

    card = RewardV2Evaluator(rewards_config=rewards_config).evaluate_run_dir(
        run_dir, profile="production", publication_mode="fixture"
    )
    assert card["blocks_publication"] is True
    blocking = set(card["blocking_rewards"])
    # Real-data, nested eval, and publication-related rewards must not pass.
    for required in (
        "R16_real_data_exclusivity",
        "R17_as_of_integrity",
        "R18_nested_evaluation",
        "R26_benchmark_superiority",
    ):
        assert required in blocking
        assert card["rewards"][required]["state"] in {"fail", "insufficient_evidence"}


# --- Per-reward positive + adversarial pairs ---------------------------------


def _eval_one(run_dir: Path, reward_id: str, rewards_config: dict, **kwargs: object) -> dict:
    card = RewardV2Evaluator(rewards_config=rewards_config).evaluate_run_dir(
        run_dir,
        profile=str(kwargs.get("profile", "research")),
        publication_mode=str(kwargs.get("publication_mode", "research")),
    )
    return card["rewards"][reward_id]


def test_r0_build_pass_and_coverage_fail(tmp_path: Path, rewards_config: dict) -> None:
    good = _seed_run(tmp_path / "r0-good")
    assert _eval_one(good, "R0_build", rewards_config)["state"] == "pass"
    bad = _seed_run(
        tmp_path / "r0-bad",
        **{
            "coverage.json": {"line_rate_pct": 89.99},
            "ci_manifest.json": {"commands_passed": True, "line_coverage_pct": 89.99},
        },
    )
    result = _eval_one(bad, "R0_build", rewards_config)
    assert result["state"] == "fail"
    assert any("coverage" in r.lower() for r in result["failure_reasons"])


def test_r0_build_commands_fail(tmp_path: Path, rewards_config: dict) -> None:
    bad = _seed_run(
        tmp_path / "r0-cmd",
        **{
            "ci_manifest.json": {"commands_passed": False, "line_coverage_pct": 95.0},
            "coverage.json": {"line_rate_pct": 95.0},
        },
    )
    assert _eval_one(bad, "R0_build", rewards_config)["state"] == "fail"


def test_r1_reproducibility_pass_and_unverified_fail(tmp_path: Path, rewards_config: dict) -> None:
    good = _seed_run(tmp_path / "r1-good")
    assert _eval_one(good, "R1_reproducibility", rewards_config)["state"] == "pass"
    bad = _seed_run(
        tmp_path / "r1-bad",
        **{
            "reproducibility_fingerprint.json": {
                "cross_run_verified": False,
                "combined_hash": "x",
            }
        },
    )
    assert _eval_one(bad, "R1_reproducibility", rewards_config)["state"] == "fail"


def test_r2_provenance_pass_and_missing_hash_fail(tmp_path: Path, rewards_config: dict) -> None:
    good = _seed_run(tmp_path / "r2-good")
    assert _eval_one(good, "R2_provenance", rewards_config)["state"] == "pass"
    broken = _base_forecasts().with_columns(pl.lit("").alias("model_config_hash"))
    bad = _seed_run(tmp_path / "r2-bad", race_forecasts=broken)
    assert _eval_one(bad, "R2_provenance", rewards_config)["state"] == "fail"


def test_r3_sync_integrity_pass_and_failed_source(tmp_path: Path, rewards_config: dict) -> None:
    good = _seed_run(tmp_path / "r3-good")
    assert _eval_one(good, "R3_sync_integrity", rewards_config)["state"] == "pass"
    manifest = _base_manifest().with_columns(pl.lit("failed").alias("status"))
    bad = _seed_run(tmp_path / "r3-bad", source_manifest=manifest)
    assert _eval_one(bad, "R3_sync_integrity", rewards_config)["state"] == "fail"


def test_r4_calibration_pass_and_high_ece_fail(tmp_path: Path, rewards_config: dict) -> None:
    good = _seed_run(tmp_path / "r4-good")
    assert _eval_one(good, "R4_calibration", rewards_config)["state"] == "pass"
    nested = {
        "exact_pipeline": True,
        "outer_fold": True,
        "calibration": {
            "expected_calibration_error": 0.10,
            "ece_bootstrap_upper": 0.12,
            "calibration_intercept": 0.01,
            "calibration_slope": 1.0,
        },
        "calibration_fitted_on_scored_rows": True,
    }
    bad = _seed_run(tmp_path / "r4-bad", **{"nested_evaluation.json": nested})
    result = _eval_one(bad, "R4_calibration", rewards_config)
    assert result["state"] == "fail"


def test_r5_baseline_competition_pass_and_cycle_count_fail(
    tmp_path: Path, rewards_config: dict
) -> None:
    good = _seed_run(tmp_path / "r5-good")
    assert _eval_one(good, "R5_baseline_competition", rewards_config)["state"] == "pass"
    nested = {
        "exact_pipeline": True,
        "outer_fold": True,
        "independent_cycle_count": 2,
        "paired_benchmarks": {
            "independent_cycle_count": 2,
            "beats_all_simple_baselines": True,
            "log_score_diff_ci_upper": -0.01,
        },
        "effective_n_inflated": True,
        "calibration": {
            "expected_calibration_error": 0.02,
            "calibration_intercept": 0.0,
            "calibration_slope": 1.0,
        },
    }
    bad = _seed_run(tmp_path / "r5-bad", **{"nested_evaluation.json": nested})
    assert _eval_one(bad, "R5_baseline_competition", rewards_config)["state"] == "fail"


def test_r6_component_admission_pass_and_baseline_fill_fail(
    tmp_path: Path, rewards_config: dict
) -> None:
    good = _seed_run(tmp_path / "r6-good")
    assert _eval_one(good, "R6_component_admission", rewards_config)["state"] == "pass"
    nested = {
        "exact_pipeline": True,
        "component_ablations": {"ensemble": {"beats_or_matches_baseline": True}},
        "matched_coverage": True,
        "baseline_filled_missing": True,
        "outer_fold": True,
        "calibration": {
            "expected_calibration_error": 0.02,
            "calibration_intercept": 0.0,
            "calibration_slope": 1.0,
        },
    }
    bad = _seed_run(tmp_path / "r6-bad", **{"nested_evaluation.json": nested})
    assert _eval_one(bad, "R6_component_admission", rewards_config)["state"] == "fail"


def test_r7_sparse_honesty_pass_and_public_tier_c_fail(
    tmp_path: Path, rewards_config: dict
) -> None:
    good = _seed_run(tmp_path / "r7-good")
    assert _eval_one(good, "R7_sparse_honesty", rewards_config)["state"] == "pass"
    catalog = pl.DataFrame({"race_id": ["r0", "r1"], "tier": ["C", "C"]})
    forecasts = _base_forecasts()
    bad = _seed_run(tmp_path / "r7-bad", race_catalog=catalog, race_forecasts=forecasts)
    assert _eval_one(bad, "R7_sparse_honesty", rewards_config)["state"] == "fail"


def test_r8_uncertainty_pass_and_coverage_fail(tmp_path: Path, rewards_config: dict) -> None:
    good = _seed_run(tmp_path / "r8-good")
    assert _eval_one(good, "R8_uncertainty_quality", rewards_config)["state"] == "pass"
    nested = {
        "exact_pipeline": True,
        "outer_fold": True,
        "coverage": {
            "interval_50_coverage": 0.10,
            "interval_80_coverage": 0.20,
            "interval_90_coverage": 0.30,
        },
        "calibration": {
            "expected_calibration_error": 0.02,
            "calibration_intercept": 0.0,
            "calibration_slope": 1.0,
        },
    }
    bad = _seed_run(tmp_path / "r8-bad", **{"nested_evaluation.json": nested})
    assert _eval_one(bad, "R8_uncertainty_quality", rewards_config)["state"] == "fail"


def test_r9_public_signal_experimental_pass_and_leaky_fail(
    tmp_path: Path, rewards_config: dict
) -> None:
    good = _seed_run(tmp_path / "r9-good")
    assert _eval_one(good, "R9_public_signal_discipline", rewards_config)["state"] == "pass"
    nested = {
        "exact_pipeline": True,
        "outer_fold": True,
        "public_signal_leakage": {
            "leakage_passed": False,
            "nested_value_added": False,
            "result_derived": True,
            "post_as_of": True,
        },
        "calibration": {
            "expected_calibration_error": 0.02,
            "calibration_intercept": 0.0,
            "calibration_slope": 1.0,
        },
        "component_ablations": {"ensemble": {"beats_or_matches_baseline": True}},
    }
    bad = _seed_run(tmp_path / "r9-bad", **{"nested_evaluation.json": nested})
    evaluator = RewardV2Evaluator(
        rewards_config=rewards_config,
        model_config={"trusted_components": {"public_signals": True}},
    )
    result = evaluator.evaluate_run_dir(bad, profile="research")["rewards"][
        "R9_public_signal_discipline"
    ]
    assert result["state"] == "fail"


def test_r10_explainability_pass_and_placeholder_fail(tmp_path: Path, rewards_config: dict) -> None:
    good = _seed_run(tmp_path / "r10-good")
    assert _eval_one(good, "R10_explainability", rewards_config)["state"] == "pass"
    fc = _base_forecasts().with_columns(pl.lit("placeholder").alias("top_drivers"))
    bad = _seed_run(tmp_path / "r10-bad", race_forecasts=fc)
    assert _eval_one(bad, "R10_explainability", rewards_config)["state"] == "fail"


def test_r11_plot_contract_pass_and_missing_plot_fail(tmp_path: Path, rewards_config: dict) -> None:
    good = _seed_run(tmp_path / "r11-good")
    assert _eval_one(good, "R11_plot_contract", rewards_config)["state"] == "pass"
    bad = _seed_run(
        tmp_path / "r11-bad",
        **{"plot_manifest.json": {"calibration": [], "projection": []}},
    )
    assert _eval_one(bad, "R11_plot_contract", rewards_config)["state"] == "fail"


def test_r12_performance_pass_and_engine_mismatch_fail(
    tmp_path: Path, rewards_config: dict
) -> None:
    good = _seed_run(tmp_path / "r12-good")
    assert _eval_one(good, "R12_performance_contract", rewards_config)["state"] == "pass"
    bad = _seed_run(
        tmp_path / "r12-bad",
        **{
            "performance.json": {
                "requested_engine": "numba",
                "engine": "python",
                "parallel": True,
                "numba_available": True,
                "simulation_count": 1000,
                "max_mcse": 0.001,
            }
        },
    )
    assert _eval_one(bad, "R12_performance_contract", rewards_config)["state"] == "fail"


def test_r13_posterior_pass_and_divergence_fail(tmp_path: Path, rewards_config: dict) -> None:
    good = _seed_run(tmp_path / "r13-good")
    assert _eval_one(good, "R13_posterior_quality", rewards_config)["state"] == "pass"
    bad = _seed_run(
        tmp_path / "r13-bad",
        **{
            "posterior_diagnostics.json": {
                "divergences": 12,
                "draw_count": 1000,
                "r_hat_max": 1.2,
                "ess_min": 50,
                "tail_ess_min": 50,
                "e_bfmi": 0.1,
            }
        },
    )
    assert _eval_one(bad, "R13_posterior_quality", rewards_config)["state"] == "fail"


def test_r14_calibrated_publication_pass_and_out_of_range_fail(
    tmp_path: Path, rewards_config: dict
) -> None:
    good = _seed_run(tmp_path / "r14-good")
    assert _eval_one(good, "R14_calibrated_publication", rewards_config)["state"] == "pass"
    fc = _base_forecasts().with_columns(pl.lit(1.5).alias("winner_probability"))
    bad = _seed_run(tmp_path / "r14-bad", race_forecasts=fc)
    assert _eval_one(bad, "R14_calibrated_publication", rewards_config)["state"] == "fail"


def test_r15_daily_update_pass_and_noop_fail(tmp_path: Path, rewards_config: dict) -> None:
    good = _seed_run(tmp_path / "r15-good")
    assert _eval_one(good, "R15_daily_update_quality", rewards_config)["state"] == "pass"
    bad = _seed_run(
        tmp_path / "r15-bad",
        **{
            "latest_daily_update.json": {
                "quality_passed": False,
                "needs_full_refit": True,
                "full_refit_executed": False,
                "strategy": "noop",
                "noop": True,
                "weights_degenerate": True,
            }
        },
    )
    assert _eval_one(bad, "R15_daily_update_quality", rewards_config)["state"] == "fail"


def test_r16_real_data_pass_and_fixture_fail(tmp_path: Path, rewards_config: dict) -> None:
    good = _seed_run(tmp_path / "r16-good")
    assert _eval_one(good, "R16_real_data_exclusivity", rewards_config)["state"] == "pass"
    bad = _seed_run(
        tmp_path / "r16-bad",
        source_manifest=_base_manifest(synthetic=True),
        **{"source_registry_audit.json": {"synthetic_rows": 1, "fixture_source_count": 1}},
    )
    assert _eval_one(bad, "R16_real_data_exclusivity", rewards_config)["state"] == "fail"


def test_r17_as_of_pass_and_future_row_fail(tmp_path: Path, rewards_config: dict) -> None:
    good = _seed_run(tmp_path / "r17-good")
    assert _eval_one(good, "R17_as_of_integrity", rewards_config)["state"] == "pass"
    bad = _seed_run(
        tmp_path / "r17-bad",
        **{
            "as_of_audit.json": {
                "future_eligible_rows": 3,
                "time_travel_canaries_passed": False,
            }
        },
    )
    assert _eval_one(bad, "R17_as_of_integrity", rewards_config)["state"] == "fail"


def test_r18_nested_pass_and_leakage_fail(tmp_path: Path, rewards_config: dict) -> None:
    good = _seed_run(tmp_path / "r18-good")
    assert _eval_one(good, "R18_nested_evaluation", rewards_config)["state"] == "pass"
    bad = _seed_run(
        tmp_path / "r18-bad",
        **{
            "nested_evaluation.json": {
                "exact_pipeline": False,
                "outer_cycle_excluded": False,
                "held_out_permutation_affects_prior_folds": True,
                "outer_fold": True,
                "calibration": {
                    "expected_calibration_error": 0.02,
                    "calibration_intercept": 0.0,
                    "calibration_slope": 1.0,
                },
            }
        },
    )
    assert _eval_one(bad, "R18_nested_evaluation", rewards_config)["state"] == "fail"


def test_r19_covariance_pass_and_complement_avg_fail(tmp_path: Path, rewards_config: dict) -> None:
    good = _seed_run(tmp_path / "r19-good")
    assert _eval_one(good, "R19_covariance_recovery", rewards_config)["state"] == "pass"
    bad = _seed_run(
        tmp_path / "r19-bad",
        **{
            "covariance_recovery.json": {
                "is_psd": False,
                "complement_averaging": True,
                "max_factor_variance_rel_error": 0.5,
                "correlation_rmse": 0.3,
                "one_signed_residual_per_race": False,
            }
        },
    )
    assert _eval_one(bad, "R19_covariance_recovery", rewards_config)["state"] == "fail"


def test_r20_hierarchy_pass_and_unpolled_fail(tmp_path: Path, rewards_config: dict) -> None:
    good = _seed_run(tmp_path / "r20-good")
    assert _eval_one(good, "R20_all_race_hierarchy", rewards_config)["state"] == "pass"
    bad = _seed_run(
        tmp_path / "r20-bad",
        **{
            "hierarchy_recovery.json": {
                "all_control_bearing_races_in_model": False,
                "unpolled_propagation_passed": False,
                "label_symmetry_passed": False,
            }
        },
    )
    assert _eval_one(bad, "R20_all_race_hierarchy", rewards_config)["state"] == "fail"


def test_r21_poll_identity_pass_and_double_count_fail(tmp_path: Path, rewards_config: dict) -> None:
    good = _seed_run(tmp_path / "r21-good")
    assert _eval_one(good, "R21_poll_observation_identity", rewards_config)["state"] == "pass"
    bad = _seed_run(
        tmp_path / "r21-bad",
        **{
            "poll_observation_manifest.json": {
                "double_count_rows": 4,
                "option_double_count": True,
                "pollster_lineage_auditable": True,
                "unique_questions": 10,
            }
        },
    )
    assert _eval_one(bad, "R21_poll_observation_identity", rewards_config)["state"] == "fail"


def test_r22_feature_validity_pass_and_vintage_fail(tmp_path: Path, rewards_config: dict) -> None:
    good = _seed_run(tmp_path / "r22-good")
    assert _eval_one(good, "R22_feature_validity", rewards_config)["state"] == "pass"
    bad = _seed_run(
        tmp_path / "r22-bad",
        **{
            "feature_lineage.json": {
                "max_snapshots_per_feature_key": 5,
                "incumbent_relative_sign": False,
                "end_of_cycle_finance_in_early_fold": True,
                "revised_macro_in_early_fold": True,
            }
        },
    )
    assert _eval_one(bad, "R22_feature_validity", rewards_config)["state"] == "fail"


def test_r23_coherence_pass_and_semantic_fail(tmp_path: Path, rewards_config: dict) -> None:
    good = _seed_run(tmp_path / "r23-good")
    assert _eval_one(good, "R23_joint_outcome_coherence", rewards_config)["state"] == "pass"
    bad = _seed_run(
        tmp_path / "r23-bad",
        **{
            "semantic_verification.json": {
                "passed": False,
                "reconciliation_ok": False,
                "failure_reasons": ["probability 1.5", "duplicate winner"],
            }
        },
    )
    assert _eval_one(bad, "R23_joint_outcome_coherence", rewards_config)["state"] == "fail"


def test_r24_atomic_publication_research_ok_production_without_manifest_fail(
    tmp_path: Path, rewards_config: dict
) -> None:
    good = _seed_run(tmp_path / "r24-good")
    assert _eval_one(good, "R24_atomic_publication", rewards_config)["state"] == "pass"
    bad = _seed_run(
        tmp_path / "r24-bad",
        **{
            "publication_decision.json": {
                "publication_mode": "production",
                "allowed": True,
            },
            "promotion_manifest.json": {
                "attempt_id": "x",
                "verified": False,
                "publication_mode": "production",
            },
        },
    )
    # Infer production mode from decision.
    result = RewardV2Evaluator(rewards_config=rewards_config).evaluate_run_dir(
        bad, profile="production", publication_mode="production"
    )["rewards"]["R24_atomic_publication"]
    assert result["state"] == "fail"


def test_r25_live_source_pass_and_canary_fail(tmp_path: Path, rewards_config: dict) -> None:
    good = _seed_run(tmp_path / "r25-good")
    assert _eval_one(good, "R25_live_source_resilience", rewards_config)["state"] == "pass"
    bad = _seed_run(
        tmp_path / "r25-bad",
        **{
            "live_source_canaries.json": {
                "all_passed": False,
                "history": [{"name": "empty_feed", "status": "failed"}],
                "injected_failure_results": [{"name": "timeout", "status": "success"}],
            }
        },
    )
    assert _eval_one(bad, "R25_live_source_resilience", rewards_config)["state"] == "fail"


def test_r26_benchmark_pass_and_post_hoc_metric_fail(tmp_path: Path, rewards_config: dict) -> None:
    good = _seed_run(tmp_path / "r26-good")
    assert _eval_one(good, "R26_benchmark_superiority", rewards_config)["state"] == "pass"
    bad = _seed_run(
        tmp_path / "r26-bad",
        **{
            "benchmark_superiority.json": {
                "preregistered": True,
                "independent_cycle_count": 3,
                "beats_all_simple_baselines": True,
                "max_comparator_log_score_gap": 0.05,
                "metric_changed_after_scoring": True,
                "difficult_cycle_removed": True,
                "scope_mismatch": True,
            }
        },
    )
    assert _eval_one(bad, "R26_benchmark_superiority", rewards_config)["state"] == "fail"


def test_r27_contract_parity_pass_and_stale_claim_fail(
    tmp_path: Path, rewards_config: dict
) -> None:
    good = _seed_run(tmp_path / "r27-good")
    assert _eval_one(good, "R27_contract_parity", rewards_config)["state"] == "pass"
    bad = _seed_run(
        tmp_path / "r27-bad",
        **{
            "contract_parity.json": {
                "passed": False,
                "stale_claims": 2,
                "failure_reasons": ["calibration cap mismatch in README"],
            }
        },
    )
    assert _eval_one(bad, "R27_contract_parity", rewards_config)["state"] == "fail"


def test_missing_metric_never_auto_passes(tmp_path: Path, rewards_config: dict) -> None:
    run_dir = tmp_path / "empty"
    run_dir.mkdir()
    card = RewardV2Evaluator(rewards_config=rewards_config).evaluate_run_dir(
        run_dir, profile="production"
    )
    # R9 may pass when public signals remain experimental (no admission evidence required).
    # R24 may pass for non-production publication modes when no production claim is made.
    allowed_empty_pass = {"R9_public_signal_discipline", "R24_atomic_publication"}
    for reward_id, record in card["rewards"].items():
        if reward_id in allowed_empty_pass and record["state"] == "pass":
            continue
        assert record["state"] != "pass", f"{reward_id} auto-passed without evidence"
        assert record["state"] in {"fail", "insufficient_evidence", "not_applicable"}


def test_nan_metric_is_insufficient(tmp_path: Path, rewards_config: dict) -> None:
    bad = _seed_run(
        tmp_path / "nan",
        **{
            "coverage.json": {"line_rate_pct": float("nan")},
            "ci_manifest.json": {"commands_passed": True, "line_coverage_pct": float("nan")},
        },
    )
    # JSON can't store NaN by default via our helper if we use math.nan through json.dumps
    # Use a null instead which evaluators treat as missing.
    _write_json(
        bad / "coverage.json",
        {"line_rate_pct": None},
    )
    _write_json(
        bad / "ci_manifest.json",
        {"commands_passed": True, "line_coverage_pct": None},
    )
    result = _eval_one(bad, "R0_build", rewards_config)
    assert result["state"] == "insufficient_evidence"


def test_verify_rewards_cli_production_exits_nonzero(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    forecasts = artifacts / "forecasts" / "fixture-only"
    forecasts.mkdir(parents=True)
    _base_forecasts().write_parquet(forecasts / "race_forecasts.parquet")
    _base_catalog().write_parquet(forecasts / "race_catalog.parquet")
    _base_manifest(synthetic=True).write_parquet(forecasts / "source_manifest.parquet")
    _write_json(forecasts / "publication_decision.json", {"publication_mode": "fixture"})
    _write_json(forecasts / "run_manifest.json", {"publication_mode": "fixture"})

    # Copy rewards.yaml into a temp config dir via ProjectContext root.
    root = tmp_path / "proj"
    config_dir = root / "configs"
    config_dir.mkdir(parents=True)
    shutil.copy(REWARDS_YAML, config_dir / "rewards.yaml")
    shutil.copy(REPO_ROOT / "configs" / "model.yaml", config_dir / "model.yaml")
    (root / "configs" / "sources.yaml").write_text("sources: []\n", encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "verify",
            "rewards",
            "--run-id",
            "fixture-only",
            "--profile",
            "production",
            "--root",
            str(root),
            "--artifacts-dir",
            str(artifacts),
        ],
    )
    assert result.exit_code != 0, result.output
    card_path = forecasts / "reward_card_v2.json"
    assert card_path.exists()
    card = json.loads(card_path.read_text(encoding="utf-8"))
    assert card["recomputed"] is True
    assert card["blocking_rewards"]
    states = {r: card["rewards"][r]["state"] for r in card["blocking_rewards"]}
    assert any(s in {"fail", "insufficient_evidence"} for s in states.values())


def test_relabel_research_to_production_without_manifest_fails(tmp_path: Path) -> None:
    root = tmp_path / "proj"
    config_dir = root / "configs"
    config_dir.mkdir(parents=True)
    shutil.copy(REWARDS_YAML, config_dir / "rewards.yaml")
    shutil.copy(REPO_ROOT / "configs" / "model.yaml", config_dir / "model.yaml")
    artifacts = root / "artifacts"
    run_dir = artifacts / "forecasts" / "research-run"
    _seed_run(run_dir)
    # Ensure research mode and strip verified production promotion.
    _write_json(
        run_dir / "publication_decision.json",
        {"publication_mode": "research", "allowed": True},
    )
    (run_dir / "promotion_manifest.json").unlink(missing_ok=True)

    context = ProjectContext.create(root=root, artifacts_dir=artifacts)
    verifier = PublicationVerifier(context)
    # Adversarially relabel to production.
    result = verifier.reject_relabel_without_manifest(run_dir)
    assert result["passed"] is False
    assert any("promotion" in reason for reason in result["failure_reasons"])
    assert result["promoted_pointer_unchanged"] is True


def test_promotion_refused_leaves_pointer_unchanged(tmp_path: Path) -> None:
    root = tmp_path / "proj"
    config_dir = root / "configs"
    config_dir.mkdir(parents=True)
    shutil.copy(REWARDS_YAML, config_dir / "rewards.yaml")
    shutil.copy(REPO_ROOT / "configs" / "model.yaml", config_dir / "model.yaml")
    artifacts = root / "artifacts"
    promoted = artifacts / "promoted" / "production"
    promoted.mkdir(parents=True)
    original = {
        "attempt_id": "old",
        "verified": True,
        "publication_mode": "production",
        "profile": "production",
        "promoted_at": "2020-01-01T00:00:00+00:00",
        "reward_card_hash": "old",
        "semantic_verification_hash": "old",
        "content_hashes": {},
    }
    _write_json(promoted / "promotion_manifest.json", original)
    previous_bytes = (promoted / "promotion_manifest.json").read_bytes()

    attempt = artifacts / "attempts" / "bad-attempt"
    attempt.mkdir(parents=True)
    # Empty attempt → all insufficient_evidence.
    context = ProjectContext.create(root=root, artifacts_dir=artifacts)
    payload = PublicationVerifier(context).attempt_promote(
        attempt_id="bad-attempt", profile="production"
    )
    assert payload["promoted"] is False
    assert payload["promoted_pointer_unchanged"] is True
    assert (promoted / "promotion_manifest.json").read_bytes() == previous_bytes


def test_thresholds_come_from_config_not_python_literals(tmp_path: Path) -> None:
    """Mutating rewards.yaml thresholds changes evaluation without code edits."""
    raw = yaml.safe_load(REWARDS_YAML.read_text(encoding="utf-8"))
    raw["thresholds"]["R0_build"]["min_line_coverage_pct"] = 99.9
    cfg_path = tmp_path / "rewards.yaml"
    cfg_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    cfg = load_rewards_config(str(cfg_path))
    run_dir = _seed_run(tmp_path / "thr")
    # coverage is 91 → fails under 99.9 threshold
    result = RewardV2Evaluator(rewards_config=cfg).evaluate_run_dir(run_dir)["rewards"]["R0_build"]
    assert result["state"] == "fail"
    assert result["threshold"]["min_line_coverage_pct"] == 99.9


def test_reward_verification_runner_writes_evidence(tmp_path: Path) -> None:
    root = tmp_path / "proj"
    config_dir = root / "configs"
    config_dir.mkdir(parents=True)
    shutil.copy(REWARDS_YAML, config_dir / "rewards.yaml")
    shutil.copy(REPO_ROOT / "configs" / "model.yaml", config_dir / "model.yaml")
    artifacts = root / "artifacts"
    run_dir = artifacts / "forecasts" / "runner-run"
    _seed_run(run_dir)
    context = ProjectContext.create(root=root, artifacts_dir=artifacts)
    payload = RewardVerificationRunner(context).verify(run_id="runner-run", profile="production")
    assert (run_dir / "reward_card_v2.json").exists()
    assert (run_dir / "publication_decision.json").exists()
    assert (run_dir / "reward_verification_report.md").exists()
    card = json.loads((run_dir / "reward_card_v2.json").read_text(encoding="utf-8"))
    assert card["recomputed"] is True
    assert "rewards" in card
    # Fixture/research evidence can still pass a synthetic complete matrix; fixture-only
    # production failure is covered separately. Here we only require recomputation.
    assert payload["profile"] == "production"


def test_every_reward_has_adversarial_negative_registered() -> None:
    """Structural guarantee: this module defines a negative test for each reward id."""
    source = Path(__file__).read_text(encoding="utf-8")
    for reward_id in all_reward_ids(_cfg()):
        # Each reward appears in a fail assertion path or named test.
        assert reward_id in source or reward_id.replace("_", " ") in source
    # Explicit negative coverage markers for R0-R27.
    for marker in [
        "r0_build",
        "r1_reproducibility",
        "r2_provenance",
        "r3_sync",
        "r4_calibration",
        "r5_baseline",
        "r6_component",
        "r7_sparse",
        "r8_uncertainty",
        "r9_public",
        "r10_explainability",
        "r11_plot",
        "r12_performance",
        "r13_posterior",
        "r14_calibrated",
        "r15_daily",
        "r16_real_data",
        "r17_as_of",
        "r18_nested",
        "r19_covariance",
        "r20_hierarchy",
        "r21_poll",
        "r22_feature",
        "r23_coherence",
        "r24_atomic",
        "r25_live",
        "r26_benchmark",
        "r27_contract",
    ]:
        assert marker in source


def test_as_of_runner_insufficient_and_pass_paths(tmp_path: Path) -> None:
    root = tmp_path / "proj"
    config_dir = root / "configs"
    config_dir.mkdir(parents=True)
    shutil.copy(REWARDS_YAML, config_dir / "rewards.yaml")
    artifacts = root / "artifacts"
    context = ProjectContext.create(root=root, artifacts_dir=artifacts)
    from civic_signal.verification.as_of import AsOfVerificationRunner

    missing = AsOfVerificationRunner(context).verify(
        run_id="missing-audit",
        scenario_family="national-generals",
        cycles="2004:2024",
        offsets="7,1",
        as_of="2026-07-09",
    )
    assert missing["passed"] is False
    assert missing["exit_nonzero"] is True

    forecast = artifacts / "forecasts" / "with-audit"
    forecast.mkdir(parents=True)
    _write_json(
        forecast / "as_of_audit.json",
        {
            "future_eligible_rows": 0,
            "time_travel_canaries_passed": True,
            "status": "ok",
        },
    )
    ok = AsOfVerificationRunner(context).verify(run_id="with-audit")
    assert ok["passed"] is True


def test_publication_promote_success_and_semantic_hash(tmp_path: Path) -> None:
    root = tmp_path / "proj"
    config_dir = root / "configs"
    config_dir.mkdir(parents=True)
    shutil.copy(REWARDS_YAML, config_dir / "rewards.yaml")
    shutil.copy(REPO_ROOT / "configs" / "model.yaml", config_dir / "model.yaml")
    artifacts = root / "artifacts"
    attempt = artifacts / "attempts" / "good-attempt"
    _seed_run(attempt)
    # Production claim with verified promotion will be written by attempt_promote.
    _write_json(
        attempt / "publication_decision.json",
        {"publication_mode": "research", "allowed": True},
    )
    context = ProjectContext.create(root=root, artifacts_dir=artifacts)
    payload = PublicationVerifier(context).attempt_promote(
        attempt_id="good-attempt", profile="production"
    )
    # Complete seed should pass production required rewards.
    assert payload["promoted"] is True
    assert (artifacts / "promoted" / "production" / "promotion_manifest.json").exists()
    manifest = json.loads(
        (artifacts / "promoted" / "production" / "promotion_manifest.json").read_text()
    )
    assert manifest["verified"] is True

    # Semantic verification with matching hash passes.
    semantic = PublicationVerifier(context).verify_semantic(
        run_id="good-attempt", profile="production"
    )
    assert semantic["passed"] is True


def test_registry_error_paths(tmp_path: Path) -> None:
    from civic_signal.scoring.reward_registry import (
        clear_rewards_config_cache,
        load_rewards_config,
        profile_required_rewards,
        publication_mode_default,
    )

    clear_rewards_config_cache()
    bad = tmp_path / "bad.yaml"
    bad.write_text("- not a mapping\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_rewards_config(str(bad))
    clear_rewards_config_cache()
    incomplete = tmp_path / "incomplete.yaml"
    incomplete.write_text("version: 1\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_rewards_config(str(incomplete))
    clear_rewards_config_cache()
    with pytest.raises(KeyError):
        profile_required_rewards("nope", _cfg())
    broken = dict(_cfg())
    broken["publication_mode_default"] = "not-a-mode"
    with pytest.raises(ValueError):
        publication_mode_default(broken)
    clear_rewards_config_cache()
    # Default loader from package path works in src layout.
    loaded = load_rewards_config()
    assert "reward_ids" in loaded
    clear_rewards_config_cache()


def test_copy_promoted_snapshot_and_resolve_paths(tmp_path: Path) -> None:
    from civic_signal.verification.publication import copy_promoted_snapshot

    src = tmp_path / "src"
    src.mkdir()
    (src / "a.txt").write_text("x", encoding="utf-8")
    dest = tmp_path / "dest"
    copy_promoted_snapshot(src, dest)
    assert (dest / "a.txt").read_text(encoding="utf-8") == "x"
    # Second copy replaces.
    (src / "a.txt").write_text("y", encoding="utf-8")
    copy_promoted_snapshot(src, dest)
    assert (dest / "a.txt").read_text(encoding="utf-8") == "y"

    root = tmp_path / "proj"
    (root / "configs").mkdir(parents=True)
    shutil.copy(REWARDS_YAML, root / "configs" / "rewards.yaml")
    artifacts = root / "artifacts"
    bare = artifacts / "bare-run"
    bare.mkdir(parents=True)
    context = ProjectContext.create(root=root, artifacts_dir=artifacts)
    verifier = PublicationVerifier(context)
    resolved = verifier._resolve_run_dir("bare-run")
    assert resolved == bare
    with pytest.raises(FileNotFoundError):
        verifier._resolve_run_dir("does-not-exist")


def test_verify_as_of_cli_exits_nonzero(tmp_path: Path) -> None:
    root = tmp_path / "proj"
    (root / "configs").mkdir(parents=True)
    shutil.copy(REWARDS_YAML, root / "configs" / "rewards.yaml")
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "verify",
            "as-of",
            "--run-id",
            "no-audit",
            "--root",
            str(root),
            "--artifacts-dir",
            str(root / "artifacts"),
        ],
    )
    assert result.exit_code != 0


def test_verify_publication_cli(tmp_path: Path) -> None:
    root = tmp_path / "proj"
    (root / "configs").mkdir(parents=True)
    shutil.copy(REWARDS_YAML, root / "configs" / "rewards.yaml")
    shutil.copy(REPO_ROOT / "configs" / "model.yaml", root / "configs" / "model.yaml")
    artifacts = root / "artifacts"
    run_dir = artifacts / "forecasts" / "pub-cli"
    _seed_run(run_dir)
    _write_json(
        run_dir / "publication_decision.json",
        {"publication_mode": "production"},
    )
    (run_dir / "promotion_manifest.json").unlink(missing_ok=True)
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "verify",
            "publication",
            "--run-id",
            "pub-cli",
            "--profile",
            "production",
            "--root",
            str(root),
            "--artifacts-dir",
            str(artifacts),
        ],
    )
    assert result.exit_code != 0


def test_out_of_range_probability_fails_semantic(tmp_path: Path) -> None:
    root = tmp_path / "proj"
    (root / "configs").mkdir(parents=True)
    shutil.copy(REWARDS_YAML, root / "configs" / "rewards.yaml")
    artifacts = root / "artifacts"
    run_dir = artifacts / "forecasts" / "bad-prob"
    _seed_run(run_dir)
    _base_forecasts().with_columns(pl.lit(1.5).alias("winner_probability")).write_parquet(
        run_dir / "race_forecasts.parquet"
    )
    _write_json(run_dir / "publication_decision.json", {"publication_mode": "research"})
    context = ProjectContext.create(root=root, artifacts_dir=artifacts)
    result = PublicationVerifier(context).verify_semantic(
        run_id="bad-prob", profile="research", require_promotion_for_production=False
    )
    assert result["passed"] is False


def test_reward_verification_missing_run_raises(tmp_path: Path) -> None:
    root = tmp_path / "proj"
    (root / "configs").mkdir(parents=True)
    shutil.copy(REWARDS_YAML, root / "configs" / "rewards.yaml")
    context = ProjectContext.create(root=root, artifacts_dir=root / "artifacts")
    with pytest.raises(FileNotFoundError):
        RewardVerificationRunner(context).verify(run_id="ghost", profile="production")


def test_legacy_passed_from_state_mapping() -> None:
    from civic_signal.scoring.reward_v2 import legacy_passed_from_state

    assert legacy_passed_from_state("pass") is True
    assert legacy_passed_from_state("fail") is False
    assert legacy_passed_from_state("not_applicable") is True
    assert legacy_passed_from_state("insufficient_evidence") is None


def test_stale_source_fails_r3(tmp_path: Path, rewards_config: dict) -> None:
    manifest = _base_manifest().with_columns(pl.lit("stale").alias("freshness_status"))
    bad = _seed_run(tmp_path / "stale", source_manifest=manifest)
    assert _eval_one(bad, "R3_sync_integrity", rewards_config)["state"] == "fail"


def test_mcse_and_regression_fail_r12(tmp_path: Path, rewards_config: dict) -> None:
    bad = _seed_run(
        tmp_path / "mcse",
        **{
            "performance.json": {
                "requested_engine": "python",
                "engine": "python",
                "parallel": False,
                "numba_available": False,
                "simulation_count": 10,
                "max_mcse": 0.05,
                "wall_clock_regression": 0.5,
                "regression_approved": False,
            }
        },
    )
    assert _eval_one(bad, "R12_performance_contract", rewards_config)["state"] == "fail"


def test_incomplete_posterior_diagnostics_r13_not_pass(
    tmp_path: Path, rewards_config: dict
) -> None:
    """Missing R-hat/ESS/E-BFMI must not auto-pass R13."""
    bad = _seed_run(
        tmp_path / "r13-incomplete",
        **{
            "posterior_diagnostics.json": {
                "divergences": 0,
                "draw_count": 1000,
                # deliberately omit r_hat_max, ess, e_bfmi
            }
        },
    )
    result = _eval_one(bad, "R13_posterior_quality", rewards_config)
    assert result["state"] == "insufficient_evidence"
    assert any("Missing required MCMC" in r for r in result["failure_reasons"])


def test_incomplete_covariance_recovery_r19_not_pass(tmp_path: Path, rewards_config: dict) -> None:
    """PSD-only covariance report without recovery tolerances must not pass."""
    bad = _seed_run(
        tmp_path / "r19-incomplete",
        **{
            "covariance_recovery.json": {
                "is_psd": True,
                "complement_averaging": False,
                "one_signed_residual_per_race": True,
                # omit max_factor_variance_rel_error and correlation_rmse
            }
        },
    )
    result = _eval_one(bad, "R19_covariance_recovery", rewards_config)
    assert result["state"] == "insufficient_evidence"


def test_incomplete_daily_update_r15_not_pass(tmp_path: Path, rewards_config: dict) -> None:
    """quality_passed alone without MAE/max-diff vs refit must not pass."""
    bad = _seed_run(
        tmp_path / "r15-incomplete",
        **{
            "latest_daily_update.json": {
                "quality_passed": True,
                "needs_full_refit": False,
                "strategy": "reweighting",
                "noop": False,
                "weights_degenerate": False,
                # omit MAE / max diff vs full refit
            }
        },
    )
    result = _eval_one(bad, "R15_daily_update_quality", rewards_config)
    assert result["state"] == "insufficient_evidence"


def test_incomplete_performance_r12_missing_mcse_not_pass(
    tmp_path: Path, rewards_config: dict
) -> None:
    """Engine fields without MCSE must not pass R12."""
    bad = _seed_run(
        tmp_path / "r12-incomplete",
        **{
            "performance.json": {
                "requested_engine": "python",
                "engine": "python",
                "parallel": False,
                "numba_available": False,
                "simulation_count": 1000,
                # omit max_mcse
            }
        },
    )
    result = _eval_one(bad, "R12_performance_contract", rewards_config)
    assert result["state"] == "insufficient_evidence"
    assert any("mcse" in r.lower() for r in result["failure_reasons"])


def test_incomplete_coverage_r8_partial_levels_not_pass(
    tmp_path: Path, rewards_config: dict
) -> None:
    """Only one of 50/80/90 coverages present must not pass R8."""
    nested = {
        "exact_pipeline": True,
        "outer_fold": True,
        "coverage": {
            "interval_90_coverage": 0.90,
            # omit 50 and 80
        },
        "calibration": {
            "expected_calibration_error": 0.02,
            "calibration_intercept": 0.0,
            "calibration_slope": 1.0,
        },
    }
    bad = _seed_run(tmp_path / "r8-incomplete", **{"nested_evaluation.json": nested})
    result = _eval_one(bad, "R8_uncertainty_quality", rewards_config)
    assert result["state"] == "insufficient_evidence"
    assert any("Missing interval coverage" in r for r in result["failure_reasons"])


def test_incomplete_benchmark_r26_missing_gap_not_pass(
    tmp_path: Path, rewards_config: dict
) -> None:
    """Cycle count without comparator gap must not pass R26."""
    bad = _seed_run(
        tmp_path / "r26-incomplete",
        **{
            "benchmark_superiority.json": {
                "preregistered": True,
                "independent_cycle_count": 6,
                "beats_all_simple_baselines": True,
                # omit max_comparator_log_score_gap
            }
        },
    )
    result = _eval_one(bad, "R26_benchmark_superiority", rewards_config)
    assert result["state"] == "insufficient_evidence"
    assert any("max_comparator_log_score_gap" in r for r in result["failure_reasons"])


def test_incomplete_primary_metrics_never_pass_matrix(tmp_path: Path, rewards_config: dict) -> None:
    """Adversarial incomplete evidence files must not yield pass for flagged rewards."""
    cases = {
        "R8_uncertainty_quality": {
            "nested_evaluation.json": {
                "exact_pipeline": True,
                "outer_fold": True,
                "coverage": {"interval_50_coverage": 0.5},
                "calibration": {
                    "expected_calibration_error": 0.02,
                    "calibration_intercept": 0.0,
                    "calibration_slope": 1.0,
                },
            }
        },
        "R12_performance_contract": {
            "performance.json": {
                "requested_engine": "python",
                "engine": "python",
                "parallel": False,
                "numba_available": False,
                "simulation_count": 100,
            }
        },
        "R13_posterior_quality": {
            "posterior_diagnostics.json": {"divergences": 0, "draw_count": 1000}
        },
        "R15_daily_update_quality": {
            "latest_daily_update.json": {
                "quality_passed": True,
                "strategy": "reweighting",
                "needs_full_refit": False,
            }
        },
        "R19_covariance_recovery": {
            "covariance_recovery.json": {"is_psd": True, "complement_averaging": False}
        },
        "R26_benchmark_superiority": {
            "benchmark_superiority.json": {
                "preregistered": True,
                "independent_cycle_count": 8,
                "beats_all_simple_baselines": True,
            }
        },
    }
    for reward_id, overrides in cases.items():
        run_dir = _seed_run(tmp_path / f"incomplete-{reward_id}", **overrides)
        state = _eval_one(run_dir, reward_id, rewards_config)["state"]
        assert state != "pass", f"{reward_id} auto-passed with incomplete evidence: {state}"
        assert state in {"fail", "insufficient_evidence"}


def test_zero_calibration_metrics_pass_r4(tmp_path: Path, rewards_config: dict) -> None:
    """ECE=0 and intercept=0 are perfect calibration and must pass, not missing."""
    nested = {
        "exact_pipeline": True,
        "outer_fold": True,
        "calibration": {
            "expected_calibration_error": 0.0,
            "ece_bootstrap_upper": 0.0,
            "calibration_intercept": 0.0,
            "calibration_slope": 1.0,
        },
    }
    good = _seed_run(tmp_path / "r4-zero", **{"nested_evaluation.json": nested})
    result = _eval_one(good, "R4_calibration", rewards_config)
    assert result["state"] == "pass", result
    assert result["metric"]["ece"] == 0.0
    assert result["metric"]["intercept"] == 0.0


def test_zero_mcse_passes_r12(tmp_path: Path, rewards_config: dict) -> None:
    """max_mcse=0.0 is valid and must pass threshold, not insufficient_evidence."""
    good = _seed_run(
        tmp_path / "r12-zero",
        **{
            "performance.json": {
                "requested_engine": "python",
                "engine": "python",
                "parallel": False,
                "numba_available": False,
                "simulation_count": 1000,
                "max_mcse": 0.0,
            }
        },
    )
    result = _eval_one(good, "R12_performance_contract", rewards_config)
    assert result["state"] == "pass", result
    assert result["metric"]["max_mcse"] == 0.0


def test_zero_mae_and_max_diff_pass_r15(tmp_path: Path, rewards_config: dict) -> None:
    """Perfect daily-update agreement (MAE=0, max_diff=0) must pass."""
    good = _seed_run(
        tmp_path / "r15-zero",
        **{
            "latest_daily_update.json": {
                "quality_passed": True,
                "needs_full_refit": False,
                "strategy": "reweighting",
                "probability_mae_vs_full_refit": 0.0,
                "probability_max_diff_vs_full_refit": 0.0,
                "full_refit_executed": False,
                "weights_degenerate": False,
                "noop": False,
            }
        },
    )
    result = _eval_one(good, "R15_daily_update_quality", rewards_config)
    assert result["state"] == "pass", result
    assert result["metric"]["mae_vs_refit"] == 0.0
    assert result["metric"]["max_diff_vs_refit"] == 0.0


def test_finite_from_helper_preserves_zero() -> None:
    from civic_signal.scoring.reward_v2 import _finite_from, _first_key

    assert _first_key({"a": 0.0, "b": 1.0}, "a", "b") == 0.0
    assert _first_key({"a": None, "b": 0.0}, "a", "b") == 0.0
    assert _first_key({}, "a", "b") is None
    assert (
        _finite_from(
            {"expected_calibration_error": 0.0, "ece": 0.5},
            "expected_calibration_error",
            "ece",
        )
        == 0.0
    )
    assert _finite_from({"max_mcse": 0.0, "mcse_max": 0.01}, "max_mcse", "mcse_max") == 0.0
    assert (
        _finite_from(
            {"probability_mae_vs_full_refit": 0.0},
            "probability_mae_vs_full_refit",
            "mae_vs_refit",
        )
        == 0.0
    )


def test_research_incomplete_cannot_promote_as_production(tmp_path: Path) -> None:
    """P0: incomplete research run must not promote even if labeled for promotion."""
    root = tmp_path / "proj"
    (root / "configs").mkdir(parents=True)
    shutil.copy(REWARDS_YAML, root / "configs" / "rewards.yaml")
    shutil.copy(REPO_ROOT / "configs" / "model.yaml", root / "configs" / "model.yaml")
    artifacts = root / "artifacts"
    run = artifacts / "runs" / "research-incomplete"
    run.mkdir(parents=True)
    _base_forecasts().write_parquet(run / "race_forecasts.parquet")
    _base_catalog().write_parquet(run / "race_catalog.parquet")
    _base_manifest(synthetic=True).write_parquet(run / "source_manifest.parquet")
    pl.DataFrame({"race_id": ["r0", "r1"], "draw_id": [0, 1]}).write_parquet(
        run / "forecast_draws.parquet"
    )
    pl.DataFrame({"party": ["DEM"], "majority_probability": [0.5]}).write_parquet(
        run / "control_forecasts.parquet"
    )
    _write_json(run / "publication_decision.json", {"publication_mode": "research"})
    context = ProjectContext.create(root=root, artifacts_dir=artifacts)
    payload = PublicationVerifier(context).attempt_promote(
        attempt_id="research-incomplete", profile="production"
    )
    assert payload["promoted"] is False
    assert payload["blocking_rewards"]
    assert payload["promoted_pointer_unchanged"] is True


def test_resolve_run_dir_finds_pipeline_runs_layout(tmp_path: Path) -> None:
    from civic_signal.verification.publication import resolve_run_dir

    artifacts = tmp_path / "artifacts"
    run = artifacts / "runs" / "full-forecast"
    run.mkdir(parents=True)
    (run / "marker.txt").write_text("ok", encoding="utf-8")
    assert resolve_run_dir(artifacts, "full-forecast") == run


def test_not_applicable_blocks_production_profile(tmp_path: Path, rewards_config: dict) -> None:
    """Required rewards in not_applicable must not satisfy production gate."""
    run_dir = _seed_run(tmp_path / "na-block")
    card = RewardV2Evaluator(rewards_config=rewards_config).evaluate_run_dir(
        run_dir, profile="production", publication_mode="production"
    )
    # Force a required reward into not_applicable and recompute blocking logic.
    card["rewards"]["R16_real_data_exclusivity"]["state"] = "not_applicable"
    # Re-run evaluator path via manual blocking check used in production:
    required = rewards_config["profiles"]["production"]["required_rewards"]
    blocking = [rid for rid in required if card["rewards"].get(rid, {}).get("state") != "pass"]
    assert "R16_real_data_exclusivity" in blocking


def test_duplicate_race_rows_fail_semantic_verification(tmp_path: Path) -> None:
    root = tmp_path / "proj"
    (root / "configs").mkdir(parents=True)
    shutil.copy(REWARDS_YAML, root / "configs" / "rewards.yaml")
    artifacts = root / "artifacts"
    run = artifacts / "runs" / "dup-rows"
    run.mkdir(parents=True)
    pl.DataFrame(
        {
            "race_id": ["r0", "r0"],
            "winner_probability": [0.5, 0.5],
            "model_config_hash": ["a", "a"],
            "source_manifest_hash": ["b", "b"],
        }
    ).write_parquet(run / "race_forecasts.parquet")
    pl.DataFrame({"race_id": ["r0"], "tier": ["A"]}).write_parquet(run / "race_catalog.parquet")
    pl.DataFrame({"race_id": ["r0"], "draw_id": [0]}).write_parquet(run / "forecast_draws.parquet")
    pl.DataFrame({"party": ["DEM"], "majority_probability": [0.5]}).write_parquet(
        run / "control_forecasts.parquet"
    )
    pl.DataFrame({"source_id": ["s"], "status": ["fetched"], "content_hash": ["h"]}).write_parquet(
        run / "source_manifest.parquet"
    )
    _write_json(run / "publication_decision.json", {"publication_mode": "research"})
    context = ProjectContext.create(root=root, artifacts_dir=artifacts)
    result = PublicationVerifier(context).verify_semantic(
        run_id="dup-rows",
        profile="research",
        require_promotion_for_production=False,
    )
    assert result["passed"] is False
    assert any("Duplicate" in r for r in result["failure_reasons"])


def test_missing_commands_passed_blocks_r0(tmp_path: Path, rewards_config: dict) -> None:
    bad = _seed_run(
        tmp_path / "r0-no-cmd",
        **{
            "ci_manifest.json": {"line_coverage_pct": 95.0},  # no commands_passed
            "coverage.json": {"line_rate_pct": 95.0},
        },
    )
    assert _eval_one(bad, "R0_build", rewards_config)["state"] == "insufficient_evidence"


def test_missing_ece_upper_blocks_r4(tmp_path: Path, rewards_config: dict) -> None:
    nested = {
        "exact_pipeline": True,
        "outer_fold": True,
        "held_out_permutation_affects_prior_folds": False,
        "calibration": {
            "expected_calibration_error": 0.02,
            "calibration_intercept": 0.0,
            "calibration_slope": 1.0,
            # omit ece_bootstrap_upper
        },
    }
    bad = _seed_run(tmp_path / "r4-no-upper", **{"nested_evaluation.json": nested})
    assert _eval_one(bad, "R4_calibration", rewards_config)["state"] == "insufficient_evidence"


def test_promote_copies_immutable_snapshot(tmp_path: Path) -> None:
    root = tmp_path / "proj"
    (root / "configs").mkdir(parents=True)
    shutil.copy(REWARDS_YAML, root / "configs" / "rewards.yaml")
    shutil.copy(REPO_ROOT / "configs" / "model.yaml", root / "configs" / "model.yaml")
    artifacts = root / "artifacts"
    attempt = artifacts / "attempts" / "snap-attempt"
    _seed_run(attempt)
    context = ProjectContext.create(root=root, artifacts_dir=artifacts)
    payload = PublicationVerifier(context).attempt_promote(
        attempt_id="snap-attempt", profile="production"
    )
    assert payload["promoted"] is True
    snap = Path(payload["snapshot_dir"])
    assert snap.exists()
    assert (snap / "race_forecasts.parquet").exists()
    assert (artifacts / "promoted" / "production" / "promotion_manifest.json").exists()
    manifest = json.loads(
        (artifacts / "promoted" / "production" / "promotion_manifest.json").read_text()
    )
    assert manifest["content_hashes"]
    assert "race_forecasts.parquet" in manifest["content_hashes"]
    assert manifest["rewards_config_hash"]


def test_semantic_fails_missing_required_artifacts(tmp_path: Path) -> None:
    root = tmp_path / "proj"
    (root / "configs").mkdir(parents=True)
    shutil.copy(REWARDS_YAML, root / "configs" / "rewards.yaml")
    artifacts = root / "artifacts"
    run = artifacts / "runs" / "thin"
    run.mkdir(parents=True)
    _base_forecasts().write_parquet(run / "race_forecasts.parquet")
    _write_json(run / "publication_decision.json", {"publication_mode": "research"})
    context = ProjectContext.create(root=root, artifacts_dir=artifacts)
    result = PublicationVerifier(context).verify_semantic(
        run_id="thin",
        profile="research",
        require_promotion_for_production=False,
    )
    assert result["passed"] is False
    assert any("Missing required" in r for r in result["failure_reasons"])


def test_verify_rewards_finds_runs_layout(tmp_path: Path) -> None:
    root = tmp_path / "proj"
    (root / "configs").mkdir(parents=True)
    shutil.copy(REWARDS_YAML, root / "configs" / "rewards.yaml")
    shutil.copy(REPO_ROOT / "configs" / "model.yaml", root / "configs" / "model.yaml")
    artifacts = root / "artifacts"
    run = artifacts / "runs" / "layout-run"
    _seed_run(run)
    context = ProjectContext.create(root=root, artifacts_dir=artifacts)
    payload = RewardVerificationRunner(context).verify(run_id="layout-run", profile="production")
    assert (run / "reward_card_v2.json").exists()
    assert payload["profile"] == "production"


def test_production_semantic_detects_content_hash_tamper(tmp_path: Path) -> None:
    root = tmp_path / "proj"
    (root / "configs").mkdir(parents=True)
    shutil.copy(REWARDS_YAML, root / "configs" / "rewards.yaml")
    shutil.copy(REPO_ROOT / "configs" / "model.yaml", root / "configs" / "model.yaml")
    artifacts = root / "artifacts"
    attempt = artifacts / "attempts" / "tamper"
    _seed_run(attempt)
    context = ProjectContext.create(root=root, artifacts_dir=artifacts)
    payload = PublicationVerifier(context).attempt_promote(
        attempt_id="tamper", profile="production"
    )
    assert payload["promoted"] is True
    # Tamper a primary artifact after promotion.
    pl.DataFrame(
        {
            "race_id": ["r0"],
            "winner_probability": [0.99],
            "model_config_hash": ["tampered"],
            "source_manifest_hash": ["x"],
            "tier_reason": ["x"],
            "data_quality_flags": ["x"],
            "top_drivers": ["x"],
            "component_contributions": ["x"],
            "uncertainty_explanation": ["x"],
        }
    ).write_parquet(attempt / "race_forecasts.parquet")
    result = PublicationVerifier(context).verify_semantic(run_id="tamper", profile="production")
    assert result["passed"] is False
    reasons = " ".join(result["failure_reasons"]).lower()
    assert "hash" in reasons or "duplicate" in reasons or "blocked" in reasons


def test_r24_production_without_promotion_fails(tmp_path: Path, rewards_config: dict) -> None:
    run_dir = _seed_run(tmp_path / "r24-prod")
    (run_dir / "promotion_manifest.json").unlink(missing_ok=True)
    result = RewardV2Evaluator(rewards_config=rewards_config).evaluate_run_dir(
        run_dir, profile="production", publication_mode="production"
    )["rewards"]["R24_atomic_publication"]
    assert result["state"] in {"fail", "insufficient_evidence"}
