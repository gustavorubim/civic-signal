from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from civic_signal.config import ProjectContext
from civic_signal.features import FeatureBundle
from civic_signal.models.simulation import SimulationEngine
from civic_signal.verification.coherence import CoherenceVerificationRunner

ROOT = Path(__file__).resolve().parents[1]


def _context(tmp_path: Path) -> ProjectContext:
    return ProjectContext.create(
        root=ROOT,
        data_dir=tmp_path / "data",
        artifacts_dir=tmp_path / "artifacts",
    )


def _simulation_fixture() -> tuple[FeatureBundle, pl.DataFrame]:
    catalog = pl.DataFrame(
        {
            "race_id": ["R1"],
            "tier": ["A"],
            "tier_reason": ["test"],
            "geography": ["US"],
            "office_type": ["president"],
            "control_body": [None],
            "seats": [1],
            "race_type": ["candidate"],
        }
    )
    options = pl.DataFrame(
        {
            "race_id": ["R1", "R1"],
            "option_id": ["D", "R"],
            "party": ["DEM", "REP"],
        }
    )
    fundamentals = pl.DataFrame(
        {"race_id": ["R1"], "historical_turnout_rate": [0.6], "registered_voters": [1000]}
    )
    empty = pl.DataFrame(schema={"race_id": pl.Utf8})
    bundle = FeatureBundle(
        races=catalog,
        options=options,
        polls=empty,
        markets=empty,
        public_signals=empty,
        fundamentals=fundamentals,
        results=empty,
        backtest_predictions=empty,
        race_catalog=catalog,
    )
    ensemble = pl.DataFrame(
        {
            "race_id": ["R1", "R1"],
            "option_id": ["D", "R"],
            "party": ["DEM", "REP"],
            "vote_share": [0.5, 0.5],
            "uncertainty": [0.04, 0.04],
            "component_disagreement": [0.0, 0.0],
            "admitted": [True, True],
            "explanation": ["test", "test"],
            "component_contributions": ["{}", "{}"],
        }
    )
    return bundle, ensemble


def test_adaptive_simulation_meets_mcse_deterministically() -> None:
    bundle, ensemble = _simulation_fixture()
    config = {
        "seed": 7,
        "simulation_count": 1000,
        "performance": {
            "engine": "python",
            "parallel": False,
            "max_mcse": 0.0025,
            "adaptive_mcse": True,
            "simulation_batch_size": 5000,
            "simulation_max_draws": 50000,
        },
        "uncertainty": {"tier_a_sigma": 0.03, "tier_b_sigma": 0.07},
    }
    first = SimulationEngine(config).run(bundle, ensemble)
    second = SimulationEngine(config).run(bundle, ensemble)

    assert first.performance["mcse_target_met"] is True
    assert first.performance["simulation_count"] == 40000
    assert first.performance["adaptive_attempts"] == [1000, 40000]
    assert first.performance["adaptive_status"] == "target_met"
    assert float(first.performance["max_mcse"]) <= 0.0025
    assert first.draws.equals(second.draws)


def test_adaptive_simulation_reports_honest_cap_failure() -> None:
    bundle, ensemble = _simulation_fixture()
    config = {
        "seed": 7,
        "simulation_count": 1000,
        "performance": {
            "engine": "python",
            "parallel": False,
            "max_mcse": 0.0025,
            "adaptive_mcse": True,
            "simulation_batch_size": 1000,
            "simulation_max_draws": 2000,
        },
        "uncertainty": {"tier_a_sigma": 0.03, "tier_b_sigma": 0.07},
    }
    output = SimulationEngine(config).run(bundle, ensemble)
    assert output.performance["simulation_count"] == 2000
    assert output.performance["mcse_target_met"] is False
    assert output.performance["adaptive_status"] == "cap_reached_target_not_met"


def test_adaptive_simulation_cycles_bayesian_posterior_draw_ids() -> None:
    bundle, ensemble = _simulation_fixture()
    posterior = pl.DataFrame(
        [
            {
                "draw_id": draw_id,
                "race_id": "R1",
                "option_id": option_id,
                "latent_share": 0.51 if option_id == "D" else 0.49,
            }
            for draw_id in range(100)
            for option_id in ("D", "R")
        ]
    )
    config = {
        "seed": 7,
        "simulation_count": 1000,
        "performance": {
            "engine": "python",
            "parallel": False,
            "max_mcse": 0.01,
            "adaptive_mcse": True,
            "simulation_batch_size": 1000,
            "simulation_max_draws": 5000,
        },
        "uncertainty": {"tier_a_sigma": 0.03, "tier_b_sigma": 0.07},
    }
    output = SimulationEngine(config).run(bundle, ensemble, posterior_draws=posterior)
    assert output.performance["simulation_count"] > 100
    assert output.performance["posterior_draws_used"] is True
    assert output.draws["draw_id"].n_unique() == output.performance["simulation_count"]


def test_mcse_uses_raw_draw_frequency_before_platt_calibration() -> None:
    bundle, ensemble = _simulation_fixture()
    ensemble = ensemble.with_columns(
        pl.when(pl.col("option_id") == "D")
        .then(pl.lit(0.9))
        .otherwise(pl.lit(0.1))
        .alias("vote_share")
    )
    config = {
        "seed": 7,
        "simulation_count": 1000,
        "performance": {
            "engine": "python",
            "parallel": False,
            "max_mcse": 0.0025,
            "adaptive_mcse": True,
            "simulation_batch_size": 5000,
            "simulation_max_draws": 50000,
        },
        "uncertainty": {"tier_a_sigma": 0.001, "tier_b_sigma": 0.001},
        "correlation": {"national_sigma": 0.0, "region_sigma": 0.0, "office_sigma": 0.0},
        "probability_calibration": {
            "status": "fitted",
            "method": "platt_logistic_ridge",
            "intercept": 0.0,
            "slope": 0.0,
        },
    }
    output = SimulationEngine(config).run(bundle, ensemble)
    assert output.race_forecasts["winner_probability"].to_list() == [0.5, 0.5]
    assert output.performance["simulation_count"] == 1000
    assert output.performance["adaptive_status"] == "target_met_initial"


def _seed_coherent_run(ctx: ProjectContext, run_id: str = "coherent") -> Path:
    run = ctx.artifacts_dir / "runs" / run_id
    run.mkdir(parents=True)
    catalog = pl.DataFrame(
        {
            "race_id": [
                "SEN",
                "ME-AL",
                "ME-01",
                "ME-02",
                "NE-AL",
                "NE-01",
                "NE-02",
                "NE-03",
                "SPARSE",
            ],
            "tier": ["A"] * 8 + ["C"],
            "office_type": ["senate"] + ["president"] * 7 + ["house"],
            "geography": [
                "GA",
                "ME",
                "ME-01",
                "ME-02",
                "NE",
                "NE-01",
                "NE-02",
                "NE-03",
                "CA-01",
            ],
            "control_body": ["senate"] + [None] * 7 + ["house"],
            "seats": [1, 2, 1, 1, 2, 1, 1, 1, 1],
        }
    )
    forecast_rows = []
    draw_rows = []
    for race_id in ("SEN", "ME-AL", "ME-01", "ME-02", "NE-AL", "NE-01", "NE-02", "NE-03"):
        for option_id, party, probability in (("D", "DEM", 0.5), ("R", "REP", 0.5)):
            forecast_rows.append(
                {"race_id": race_id, "option_id": option_id, "winner_probability": probability}
            )
            draw_rows.extend(
                [
                    {
                        "draw_id": 0,
                        "race_id": race_id,
                        "option_id": option_id,
                        "party": party,
                        "vote_share": 0.6 if option_id == "D" else 0.4,
                        "winner": option_id == "D",
                    },
                    {
                        "draw_id": 1,
                        "race_id": race_id,
                        "option_id": option_id,
                        "party": party,
                        "vote_share": 0.4 if option_id == "D" else 0.6,
                        "winner": option_id == "R",
                    },
                ]
            )
    forecast_rows.extend(
        [
            {"race_id": "SPARSE", "option_id": "D", "winner_probability": None},
            {"race_id": "SPARSE", "option_id": "R", "winner_probability": None},
        ]
    )
    controls = pl.DataFrame(
        {
            "control_body": ["senate", "senate"],
            "party": ["DEM", "REP"],
            "control_threshold": [51, 50],
            "holdover_seats": [50, 49],
            "majority_probability": [0.5, 0.5],
            "control_probability": [0.5, 0.5],
        }
    )
    catalog.write_parquet(run / "race_catalog.parquet")
    pl.DataFrame(forecast_rows).write_parquet(run / "race_forecasts.parquet")
    pl.DataFrame(draw_rows).write_parquet(run / "forecast_draws.parquet")
    controls.write_parquet(run / "control_forecasts.parquet")
    return run


def test_coherence_verifier_reconstructs_controls_tiebreak_and_electors(tmp_path: Path) -> None:
    ctx = _context(tmp_path)
    _seed_coherent_run(ctx)
    # Drive the real runner against parquet artifacts (not mocked checks).
    result = CoherenceVerificationRunner(ctx).verify(run_id="coherent")
    assert result["passed"] is True
    assert result["checks"]["race_probability_simplex"]["passed"] is True
    assert result["checks"]["senate_tie_vp"]["passed"] is True
    recon = result["checks"]["control_reconstruction"]
    assert recon["passed"] is True
    assert recon["method"] == "recompute_from_forecast_draws"
    assert recon["draw_count"] == 2
    assert recon["max_absolute_error"] <= 1e-12
    assert result["checks"]["maine_nebraska_electors"]["passed"] is True
    assert result["checks"]["tier_c_withholding"]["passed"] is True


def test_coherence_rejects_aggregate_only_maine_nebraska_electors(tmp_path: Path) -> None:
    ctx = _context(tmp_path)
    run = _seed_coherent_run(ctx, "aggregate-electors")
    catalog = (
        pl.read_parquet(run / "race_catalog.parquet")
        .filter(~pl.col("race_id").is_in(["ME-01", "ME-02", "NE-01", "NE-02", "NE-03"]))
        .with_columns(
            pl.when(pl.col("race_id") == "ME-AL")
            .then(pl.lit(4))
            .when(pl.col("race_id") == "NE-AL")
            .then(pl.lit(5))
            .otherwise(pl.col("seats"))
            .alias("seats")
        )
    )
    catalog.write_parquet(run / "race_catalog.parquet")
    result = CoherenceVerificationRunner(ctx).verify(run_id="aggregate-electors")
    assert result["passed"] is False
    assert result["checks"]["maine_nebraska_electors"]["passed"] is False


@pytest.mark.parametrize(
    ("corruption", "expected_failed_check"),
    [
        ("range", "race_probability_simplex"),
        ("duplicate", "race_key_uniqueness"),
        ("simplex", "race_probability_simplex"),
        ("control", "control_reconstruction"),
        ("control_probability", "control_reconstruction"),
    ],
)
def test_coherence_verifier_rejects_adversarial_corruption(
    tmp_path: Path, corruption: str, expected_failed_check: str
) -> None:
    """Adversarial corruptions must fail the real recomputed check, not a stub."""
    ctx = _context(tmp_path)
    run = _seed_coherent_run(ctx, corruption)
    forecasts = pl.read_parquet(run / "race_forecasts.parquet")
    controls = pl.read_parquet(run / "control_forecasts.parquet")
    if corruption == "range":
        forecasts = forecasts.with_columns(
            pl.when((pl.col("race_id") == "SEN") & (pl.col("option_id") == "D"))
            .then(pl.lit(1.5))
            .otherwise(pl.col("winner_probability"))
            .alias("winner_probability")
        )
    elif corruption == "duplicate":
        forecasts = pl.concat([forecasts, forecasts.head(1)])
    elif corruption == "simplex":
        # Both options 0.8 → sum 1.6 fails simplex recompute.
        forecasts = forecasts.with_columns(
            pl.when(pl.col("race_id") == "SEN")
            .then(pl.lit(0.8))
            .otherwise(pl.col("winner_probability"))
            .alias("winner_probability")
        )
    elif corruption == "control":
        controls = controls.with_columns(
            pl.when(pl.col("party") == "DEM")
            .then(pl.lit(0.9))
            .otherwise(pl.col("majority_probability"))
            .alias("majority_probability")
        )
    else:
        # Corrupt only control_probability; reconstruction still must fail.
        controls = controls.with_columns(
            pl.when(pl.col("party") == "DEM")
            .then(pl.lit(0.95))
            .otherwise(pl.col("control_probability"))
            .alias("control_probability")
        )
    forecasts.write_parquet(run / "race_forecasts.parquet")
    controls.write_parquet(run / "control_forecasts.parquet")

    result = CoherenceVerificationRunner(ctx).verify(run_id=corruption)
    assert result["passed"] is False
    assert result["checks"][expected_failed_check]["passed"] is False
    if expected_failed_check == "control_reconstruction":
        assert result["checks"]["control_reconstruction"]["max_absolute_error"] > 1e-12
        assert result["checks"]["control_reconstruction"]["method"] == (
            "recompute_from_forecast_draws"
        )
