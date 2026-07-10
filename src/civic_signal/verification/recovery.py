from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any, ClassVar

import numpy as np
import polars as pl

from civic_signal.config import ProjectContext
from civic_signal.features import FeatureBundle
from civic_signal.models.common import logit
from civic_signal.models.polling_bayes import BayesianPollingModel
from civic_signal.scoring.backtest import BacktestRunner
from civic_signal.storage.io import write_json, write_parquet, write_text


@dataclass(frozen=True)
class RecoveryVerificationRunner:
    """Bounded model-recovery checks through the production polling-model facade.

    Synthetic analytic/NUTS smoke may set ``smoke_checks_passed`` but **never**
    sets ``production_sufficient=True``. Persisted ``hierarchy_recovery.json``
    always carries ``production_sufficient=false`` plus
    ``remaining_requirements`` so R19/R20 stay insufficient_evidence.
    """

    context: ProjectContext

    # Explicit remaining work for large-cycle production evidence (R19/R20).
    SYNTHETIC_REMAINING_REQUIREMENTS: ClassVar[tuple[str, ...]] = (
        "Large repeated SBC across offices, cycles, sparse/dense regimes, and priors",
        "Real NumPyro hierarchy recovery with calibrated rank-uniformity thresholds",
        "Control-bearing all-race catalog coverage and redistricting/class structures",
        "Independent promoted-run evidence rather than generated fixture evidence",
        "Large-cycle out-of-sample covariance recovery across missing-state regimes",
    )

    def verify(
        self,
        *,
        run_id: str | None = None,
        backend: str = "analytic",
        replicates: int = 12,
    ) -> dict[str, Any]:
        backend = backend.lower().strip()
        if backend not in {"analytic", "nuts"}:
            raise ValueError("Recovery backend must be analytic or nuts")
        if replicates < 2:
            raise ValueError("Recovery verification requires at least two replicates")
        run_id = run_id or datetime.now(UTC).strftime("recovery-%Y%m%dT%H%M%SZ")
        output_dir = self.context.artifacts_dir / "recovery" / run_id
        checks = self._run_checks(backend=backend, replicates=replicates)
        covariance_check, covariance_artifact = self._covariance_recovery()
        checks["covariance_recovery"] = covariance_check
        smoke_passed = all(bool(check["passed"]) for check in checks.values())
        remaining_requirements = list(self.SYNTHETIC_REMAINING_REQUIREMENTS)
        # Bounded synthetic analytic/NUTS smoke is a regression check only.
        # It must never unlock production R19/R20 promotion evidence.
        payload: dict[str, Any] = {
            "schema_version": "1.0.0",
            "run_id": run_id,
            "generated_at": datetime.now(UTC).isoformat(),
            "backend": backend,
            "replicates": replicates,
            "status": "insufficient_evidence" if smoke_passed else "failed",
            "smoke_checks_passed": smoke_passed,
            "production_sufficient": False,
            "checks": checks,
            # Deliberately null: bounded smoke evidence cannot unlock R20.
            "all_control_bearing_races_in_model": None,
            "unpolled_propagation_passed": None,
            "label_symmetry_passed": None,
            "evidence_scope": "synthetic_bounded_recovery_smoke",
            "remaining_requirements": remaining_requirements,
            "output_dir": str(output_dir),
        }
        # Defense in depth: re-assert synthetic smoke cannot claim production evidence.
        payload["production_sufficient"] = False
        payload["remaining_requirements"] = remaining_requirements
        if not payload["remaining_requirements"]:  # pragma: no cover - invariant guard
            raise RuntimeError("Synthetic recovery smoke must list remaining_requirements")
        write_json(payload, output_dir / "hierarchy_recovery.json")
        write_json(covariance_check, output_dir / "covariance_recovery.json")
        write_parquet(covariance_artifact, output_dir / "covariance_recovery.parquet")
        write_text(self._report(payload), output_dir / "hierarchy_recovery.md")
        return payload

    def _run_checks(self, *, backend: str, replicates: int) -> dict[str, dict[str, Any]]:
        truth = 0.62
        bundle = self._synthetic_bundle(seed=701, truth=truth)
        original = self._fit(bundle, backend=backend)
        polled = self._option_draws(original, "POLLED", "D")
        parameter_bias = abs(float(polled.mean()) - truth) if polled.size else float("inf")
        lower, upper = self._interval(polled)
        parameter = {
            "passed": bool(polled.size and parameter_bias <= 0.05 and lower <= truth <= upper),
            "truth": truth,
            "posterior_mean": float(polled.mean()) if polled.size else None,
            "absolute_bias": parameter_bias if np.isfinite(parameter_bias) else None,
            "interval_90": [lower, upper],
            "tolerance": 0.05,
        }

        swapped = self._fit(self._swap_labels(bundle), backend=backend)
        swapped_equivalent = self._option_draws(swapped, "POLLED", "R")
        symmetry_delta = (
            abs(float(polled.mean()) - float(swapped_equivalent.mean()))
            if polled.size and swapped_equivalent.size
            else float("inf")
        )
        label_symmetry = {
            "passed": bool(symmetry_delta <= 0.025),
            "mean_absolute_delta": symmetry_delta if np.isfinite(symmetry_delta) else None,
            "tolerance": 0.025,
            "comparison": "original D versus relabeled-equivalent R",
        }

        unpolled_d = self._option_draws(original, "UNPOLLED", "D")
        unpolled_r = self._option_draws(original, "UNPOLLED", "R")
        unpolled_diag = (
            original.filter(pl.col("race_id") == "UNPOLLED")["diagnostic_only"].all()
            if not original.filter(pl.col("race_id") == "UNPOLLED").is_empty()
            else False
        )
        d_mean = float(unpolled_d.mean()) if unpolled_d.size else None
        r_mean = float(unpolled_r.mean()) if unpolled_r.size else None
        unpolled = {
            "passed": bool(
                unpolled_d.size
                and unpolled_r.size
                and unpolled_diag
                and d_mean is not None
                and r_mean is not None
                and d_mean > 0.505
                and r_mean < 0.495
                and abs((d_mean + r_mean) - 1.0) <= 1e-10
            ),
            "dem_mean": d_mean,
            "rep_mean": r_mean,
            "prior_mean": 0.5,
            "diagnostic_only": bool(unpolled_diag),
            "expected_direction": "toward same-office/geography observed signed swing",
        }

        ranks: list[float] = []
        covered = 0
        biases: list[float] = []
        for replicate in range(replicates):
            replicate_bundle = self._synthetic_bundle(seed=10_000 + replicate, truth=truth)
            draws = self._option_draws(self._fit(replicate_bundle, backend=backend), "POLLED", "D")
            if not draws.size:
                continue
            ranks.append(float(np.mean(draws < truth)))
            lo, hi = self._interval(draws)
            covered += int(lo <= truth <= hi)
            biases.append(float(draws.mean()) - truth)
        coverage = covered / len(ranks) if ranks else 0.0
        mean_rank = float(np.mean(ranks)) if ranks else None
        mean_bias = float(np.mean(biases)) if biases else None
        sbc = {
            "passed": bool(
                len(ranks) == replicates
                and coverage >= 0.65
                and mean_rank is not None
                and 0.15 <= mean_rank <= 0.85
                and mean_bias is not None
                and abs(mean_bias) <= 0.04
            ),
            "replicates_completed": len(ranks),
            "interval_90_coverage": coverage,
            "mean_rank": mean_rank,
            "mean_bias": mean_bias,
            "production_minimum_replicates": 100,
            "scope": "bounded_smoke_not_rank_uniformity_proof",
        }
        return {
            "parameter_recovery": parameter,
            "label_symmetry": label_symmetry,
            "unpolled_propagation": unpolled,
            "sbc_smoke": sbc,
        }

    @staticmethod
    def _covariance_recovery() -> tuple[dict[str, Any], pl.DataFrame]:
        """Recover a preregistered known low-rank covariance from signed race errors."""
        seed = 47_221
        cycle_count = 600
        factor_rank = 4
        variance_tolerance = 0.20
        correlation_rmse_tolerance = 0.10
        label_tolerance = 1e-12
        complement_tolerance = 1e-12
        national_variance = 0.0004
        region_variance = 0.0001
        idiosyncratic_variance = 0.0001
        region_by_group = {
            "AA": "east",
            "AB": "east",
            "BA": "south",
            "BB": "south",
            "CA": "west",
            "CB": "west",
        }
        rng = np.random.default_rng(seed)
        rows: list[dict[str, object]] = []
        for cycle in range(cycle_count):
            national = rng.normal(0.0, np.sqrt(national_variance))
            region_errors = {
                region: rng.normal(0.0, np.sqrt(region_variance))
                for region in sorted(set(region_by_group.values()))
            }
            for group, region in sorted(region_by_group.items()):
                residual = (
                    national
                    + region_errors[region]
                    + rng.normal(0.0, np.sqrt(idiosyncratic_variance))
                )
                race_id = f"{group}-{cycle}"
                for option_id, party, signed_residual in (
                    (f"{race_id}-D", "DEM", residual),
                    (f"{race_id}-R", "REP", -residual),
                ):
                    rows.append(
                        {
                            "cycle": cycle,
                            "race_id": race_id,
                            "option_id": option_id,
                            "party": party,
                            "geography": group,
                            "predicted_vote_share": 0.5 + signed_residual,
                            "actual_vote_share": 0.5,
                        }
                    )
        frame = pl.DataFrame(rows)
        config = {
            "correlation": {
                "geographic_groups": region_by_group,
                "national_error_floor_sd": np.sqrt(national_variance),
                "region_sigma": np.sqrt(region_variance),
                "residual_min_variance": idiosyncratic_variance * 4.0,
                "residual_covariance_shrinkage": 0.0,
                "residual_factor_rank": factor_rank,
            }
        }
        recovered = BacktestRunner._residual_covariance(frame, config)
        reference_only = BacktestRunner._residual_covariance(
            frame.filter(pl.col("party") == "DEM"), config
        )
        relabeled = BacktestRunner._residual_covariance(
            frame.with_columns(
                pl.when(pl.col("party") == "DEM")
                .then(pl.lit("REP"))
                .otherwise(pl.lit("DEM"))
                .alias("party")
            ),
            config,
        )
        groups = sorted(region_by_group)

        def matrix(artifact: pl.DataFrame) -> np.ndarray:
            lookup = {
                (str(row["row_group"]), str(row["column_group"])): float(row["covariance"])
                for row in artifact.iter_rows(named=True)
            }
            return np.array(
                [[lookup[(row, column)] for column in groups] for row in groups],
                dtype=np.float64,
            )

        recovered_matrix = matrix(recovered)
        truth_matrix = np.zeros_like(recovered_matrix)
        for row_index, row_group in enumerate(groups):
            for column_index, column_group in enumerate(groups):
                value = national_variance
                if region_by_group[row_group] == region_by_group[column_group]:
                    value += region_variance
                if row_group == column_group:
                    value += idiosyncratic_variance
                truth_matrix[row_index, column_index] = value
        truth_sd = np.sqrt(np.diag(truth_matrix))
        truth_correlation = truth_matrix / np.outer(truth_sd, truth_sd)
        recovered_sd = np.sqrt(np.diag(recovered_matrix))
        recovered_correlation = recovered_matrix / np.outer(recovered_sd, recovered_sd)
        factor_variances = json.loads(recovered["factor_variances_json"].item(0))
        truth_factors = {"national": national_variance}
        truth_factors.update(
            {
                f"region:{region}": region_variance
                for region in sorted(set(region_by_group.values()))
            }
        )
        relative_errors = {
            factor: abs(float(factor_variances[factor]) - truth) / truth
            for factor, truth in truth_factors.items()
        }
        max_variance_error = max(relative_errors.values())
        correlation_rmse = float(np.sqrt(np.mean((recovered_correlation - truth_correlation) ** 2)))
        label_delta = float(np.max(np.abs(recovered_matrix - matrix(relabeled))))
        complement_delta = float(np.max(np.abs(recovered_matrix - matrix(reference_only))))
        eigenvalues = np.linalg.eigvalsh(recovered_matrix)
        is_psd = bool(float(eigenvalues.min()) >= -1e-12)
        passed = bool(
            is_psd
            and max_variance_error <= variance_tolerance
            and correlation_rmse <= correlation_rmse_tolerance
            and label_delta <= label_tolerance
            and complement_delta <= complement_tolerance
            and recovered["residual_definition"].item(0) == "one_reference_option_per_race_cycle"
        )
        return (
            {
                "passed": passed,
                "status": "bounded_smoke_passed" if passed else "failed",
                "production_sufficient": False,
                "evidence_scope": "synthetic_known_factor_bounded_recovery",
                "seed": seed,
                "cycle_count": cycle_count,
                "group_count": len(groups),
                "configured_factor_rank": factor_rank,
                "recovered_factor_rank": int(recovered["factor_rank"].item(0)),
                "representation": recovered["representation"].item(0),
                "is_psd": is_psd,
                "minimum_eigenvalue": float(eigenvalues.min()),
                "one_signed_residual_per_race": True,
                "complement_averaging": False,
                "truth_factor_variances": truth_factors,
                "recovered_factor_variances": factor_variances,
                "factor_variance_relative_errors": relative_errors,
                "max_factor_variance_rel_error": max_variance_error,
                "correlation_rmse": correlation_rmse,
                "label_reversal_max_covariance_delta": label_delta,
                "complement_rows_max_covariance_delta": complement_delta,
                "tolerances": {
                    "max_factor_variance_rel_error": variance_tolerance,
                    "max_correlation_rmse": correlation_rmse_tolerance,
                    "max_label_reversal_covariance_delta": label_tolerance,
                    "max_complement_rows_covariance_delta": complement_tolerance,
                },
                "remaining_requirement": (
                    "Large out-of-sample recovery across cycles, offices, missing groups, "
                    "and rank/shrinkage regimes is required for production evidence."
                ),
            },
            recovered,
        )

    def _fit(self, bundle: FeatureBundle, *, backend: str) -> pl.DataFrame:
        config = json.loads(json.dumps(self.context.read_yaml("model.yaml")))
        config["_bayesian_backend"] = backend
        config["bayesian"]["posterior_draw_count"] = 300
        config["bayesian"]["state_space"]["unpolled_pooling_prior_races"] = 1.0
        if backend == "nuts":
            config["bayesian"]["nuts"].update(
                {
                    "num_warmup": 20,
                    "num_samples": 40,
                    "num_chains": 1,
                    "chain_method": "sequential",
                    "wall_clock_timeout_seconds": 120,
                }
            )
        config["_fundamentals_prior_rows"] = [
            {
                "race_id": race_id,
                "option_id": option_id,
                "mean_logit": logit(0.5),
                "sd_logit": 0.25,
            }
            for race_id in ("POLLED", "UNPOLLED")
            for option_id in ("D", "R")
        ]
        model = BayesianPollingModel(config=config, as_of="2026-05-08")
        model.run(bundle)
        return model.posterior_draws(bundle)

    @staticmethod
    def _option_draws(draws: pl.DataFrame, race_id: str, option_id: str) -> np.ndarray:
        if draws.is_empty():
            return np.array([], dtype=np.float64)
        return draws.filter((pl.col("race_id") == race_id) & (pl.col("option_id") == option_id))[
            "latent_share"
        ].to_numpy()

    @staticmethod
    def _interval(values: np.ndarray) -> tuple[float | None, float | None]:
        if not values.size:
            return None, None
        return float(np.quantile(values, 0.05)), float(np.quantile(values, 0.95))

    @staticmethod
    def _synthetic_bundle(*, seed: int, truth: float) -> FeatureBundle:
        rng = np.random.default_rng(seed)
        poll_rows: list[dict[str, object]] = []
        start = date(2026, 4, 20)
        for poll_index in range(8):
            sample_size = 800
            dem = rng.binomial(sample_size, truth) / sample_size
            for option_id, share in (("D", dem), ("R", 1.0 - dem)):
                poll_rows.append(
                    {
                        "poll_id": f"P{poll_index}-{option_id}",
                        "race_id": "POLLED",
                        "option_id": option_id,
                        "pollster": f"HOUSE-{poll_index % 3}",
                        "start_date": start + timedelta(days=poll_index),
                        "end_date": start + timedelta(days=poll_index),
                        "sample_size": sample_size,
                        "population": "LV",
                        "pct": 100.0 * share,
                    }
                )
        options = pl.DataFrame(
            [
                {
                    "race_id": race_id,
                    "option_id": option_id,
                    "party": party,
                    "previous_vote_share": 0.5,
                }
                for race_id in ("POLLED", "UNPOLLED")
                for option_id, party in (("D", "DEM"), ("R", "REP"))
            ]
        )
        races = pl.DataFrame(
            {
                "race_id": ["POLLED", "UNPOLLED"],
                "office_type": ["senate", "senate"],
                "geography": ["SOUTH", "SOUTH"],
                "election_date": [date(2026, 11, 3), date(2026, 11, 3)],
                "cycle": [2026, 2026],
            }
        )
        return FeatureBundle(
            races=races,
            options=options,
            polls=pl.DataFrame(poll_rows),
            markets=pl.DataFrame(),
            public_signals=pl.DataFrame(),
            fundamentals=pl.DataFrame(),
            results=pl.DataFrame(),
            backtest_predictions=pl.DataFrame(),
            race_catalog=races,
        )

    @staticmethod
    def _swap_labels(bundle: FeatureBundle) -> FeatureBundle:
        mapping = pl.DataFrame({"option_id": ["D", "R"], "_swapped": ["R", "D"]})

        def swap(frame: pl.DataFrame) -> pl.DataFrame:
            return (
                frame.join(mapping, on="option_id", how="left")
                .drop("option_id")
                .rename({"_swapped": "option_id"})
            )

        swapped_options = swap(bundle.options).with_columns(
            pl.when(pl.col("party") == "DEM")
            .then(pl.lit("REP"))
            .when(pl.col("party") == "REP")
            .then(pl.lit("DEM"))
            .otherwise(pl.col("party"))
            .alias("party")
        )
        return FeatureBundle(
            races=bundle.races,
            options=swapped_options,
            polls=swap(bundle.polls),
            markets=bundle.markets,
            public_signals=bundle.public_signals,
            fundamentals=bundle.fundamentals,
            results=bundle.results,
            backtest_predictions=bundle.backtest_predictions,
            race_catalog=bundle.race_catalog,
        )

    @staticmethod
    def _report(payload: dict[str, Any]) -> str:
        lines = [
            "# Hierarchy Recovery Verification",
            "",
            f"- Status: `{payload['status']}`",
            f"- Smoke checks passed: `{payload['smoke_checks_passed']}`",
            f"- Production sufficient: `{payload['production_sufficient']}`",
            f"- Backend: `{payload['backend']}`",
            f"- Replicates: `{payload['replicates']}`",
            "",
            "## Checks",
            "",
        ]
        lines.extend(
            f"- `{name}`: `{check['passed']}`" for name, check in payload["checks"].items()
        )
        lines.extend(["", "## Remaining requirements", ""])
        lines.extend(f"- {item}" for item in payload["remaining_requirements"])
        return "\n".join(lines).strip() + "\n"
