from __future__ import annotations

from pathlib import Path

import polars as pl

from civic_signal import cli
from civic_signal.config import ProjectContext
from civic_signal.models.polling_bayes import BayesianPollingModel
from civic_signal.scoring.reward_v2 import RewardV2Evaluator
from civic_signal.storage.io import read_json
from civic_signal.verification.recovery import RecoveryVerificationRunner

ROOT = Path(__file__).resolve().parents[1]


def _context(tmp_path: Path) -> ProjectContext:
    return ProjectContext.create(
        root=ROOT,
        data_dir=tmp_path / "data",
        artifacts_dir=tmp_path / "artifacts",
    )


def test_recovery_runner_exercises_model_but_remains_insufficient(tmp_path: Path) -> None:
    runner = RecoveryVerificationRunner(_context(tmp_path))
    payload = runner.verify(
        run_id="bounded",
        backend="analytic",
        replicates=6,
    )

    assert payload["smoke_checks_passed"] is True
    assert payload["status"] == "insufficient_evidence"
    # Synthetic smoke must never claim production sufficiency for R19/R20.
    assert payload["production_sufficient"] is False
    assert payload["production_sufficient"] is not True
    assert isinstance(payload["remaining_requirements"], list)
    assert len(payload["remaining_requirements"]) >= 1
    assert list(payload["remaining_requirements"]) == list(
        RecoveryVerificationRunner.SYNTHETIC_REMAINING_REQUIREMENTS
    )
    assert all(check["passed"] for check in payload["checks"].values())
    assert payload["checks"]["unpolled_propagation"]["dem_mean"] > 0.5
    assert payload["checks"]["unpolled_propagation"]["rep_mean"] < 0.5
    covariance = payload["checks"]["covariance_recovery"]
    assert covariance["passed"] is True
    assert covariance["production_sufficient"] is False
    assert covariance["is_psd"] is True
    assert (
        covariance["max_factor_variance_rel_error"]
        <= covariance["tolerances"]["max_factor_variance_rel_error"]
    )
    assert covariance["correlation_rmse"] <= covariance["tolerances"]["max_correlation_rmse"]
    assert covariance["label_reversal_max_covariance_delta"] <= 1e-12
    assert covariance["complement_rows_max_covariance_delta"] <= 1e-12
    assert payload["all_control_bearing_races_in_model"] is None
    assert payload["unpolled_propagation_passed"] is None
    assert payload["label_symmetry_passed"] is None
    output = Path(payload["output_dir"])
    persisted = read_json(output / "hierarchy_recovery.json")
    assert persisted["production_sufficient"] is False
    assert persisted["status"] == "insufficient_evidence"
    assert persisted["evidence_scope"] == "synthetic_bounded_recovery_smoke"
    assert len(persisted["remaining_requirements"]) >= 1
    assert (output / "hierarchy_recovery.md").exists()
    assert (output / "covariance_recovery.json").exists()
    covariance_artifact = pl.read_parquet(output / "covariance_recovery.parquet")
    assert covariance_artifact["psd_constructed"].all()
    reward = RewardV2Evaluator(model_config=_context(tmp_path).read_yaml("model.yaml"))
    r20 = reward._eval_R20_all_race_hierarchy(
        output,
        {"hierarchy_recovery": persisted},
    )
    r19 = reward._eval_R19_covariance_recovery(
        output,
        {"covariance_recovery": covariance},
    )
    assert r19["state"] == "insufficient_evidence"
    assert r20["state"] == "insufficient_evidence"


def test_recovery_runner_detects_deliberately_inverted_posterior(
    tmp_path: Path, monkeypatch
) -> None:
    original = BayesianPollingModel._posterior_logit

    def inverted(self, prior_share, observations, prior_sd_logit=None):
        mean, sd = original(
            self,
            prior_share,
            observations,
            prior_sd_logit=prior_sd_logit,
        )
        return -mean, sd

    monkeypatch.setattr(BayesianPollingModel, "_posterior_logit", inverted)
    payload = RecoveryVerificationRunner(_context(tmp_path)).verify(
        run_id="inverted",
        replicates=4,
    )

    assert payload["smoke_checks_passed"] is False
    assert payload["status"] == "failed"
    assert payload["checks"]["parameter_recovery"]["passed"] is False


def test_recovery_runner_detects_missing_unpolled_pooling(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        BayesianPollingModel,
        "_unpolled_hierarchical_shifts",
        lambda self, **kwargs: (
            {},
            {
                "status": "deliberately_disabled",
                "observed_race_count": 0,
                "propagated_race_option_count": 0,
            },
        ),
    )
    payload = RecoveryVerificationRunner(_context(tmp_path)).verify(
        run_id="no-pooling",
        replicates=4,
    )

    assert payload["smoke_checks_passed"] is False
    assert payload["checks"]["unpolled_propagation"]["passed"] is False


def test_recovery_runner_detects_deliberate_label_asymmetry(tmp_path: Path, monkeypatch) -> None:
    original = BayesianPollingModel._posterior_frame

    def asymmetric(cls, rows, options):
        frame = original(rows, options)
        return frame.with_columns(
            pl.when(pl.col("option_id") == "R")
            .then((pl.col("latent_share") + 0.08).clip(1e-6, 1.0 - 1e-6))
            .otherwise(pl.col("latent_share"))
            .alias("latent_share")
        )

    monkeypatch.setattr(BayesianPollingModel, "_posterior_frame", classmethod(asymmetric))
    payload = RecoveryVerificationRunner(_context(tmp_path)).verify(
        run_id="asymmetric",
        replicates=4,
    )

    assert payload["smoke_checks_passed"] is False
    assert payload["checks"]["label_symmetry"]["passed"] is False


def test_verify_recovery_cli_reports_bounded_status(tmp_path: Path, monkeypatch) -> None:
    def fake_verify(self, **kwargs):
        return {
            "smoke_checks_passed": True,
            "status": "insufficient_evidence",
            "output_dir": str(tmp_path / "recovery"),
        }

    monkeypatch.setattr(RecoveryVerificationRunner, "verify", fake_verify)
    cli.verify_recovery(
        run_id="cli",
        backend="analytic",
        replicates=4,
        root=ROOT,
        sources_config="sources.yaml",
        data_dir=tmp_path / "data",
        artifacts_dir=tmp_path / "artifacts",
    )


def test_recovery_input_validation(tmp_path: Path) -> None:
    runner = RecoveryVerificationRunner(_context(tmp_path))
    for kwargs in ({"backend": "bad"}, {"replicates": 1}):
        try:
            runner.verify(**kwargs)
        except ValueError:
            pass
        else:  # pragma: no cover - assertion guard
            raise AssertionError(f"Expected invalid recovery arguments to fail: {kwargs}")
