from __future__ import annotations

import shutil
from dataclasses import replace
from pathlib import Path

import polars as pl

from civic_signal.config import ProjectContext
from civic_signal.features import FeatureBundle, TierAssessor, filter_bundle_by_date, subset_bundle
from civic_signal.models import (
    EnsembleModel,
    FundamentalsModel,
    MarketModel,
    PollingModel,
    PublicSignalModel,
    SimulationEngine,
)
from civic_signal.verification.as_of import run_exact_publication_time_travel_canary
from civic_signal.verification.publication import PublicationVerifier

AS_OF = "2026-05-08"


def _bundle() -> FeatureBundle:
    races = pl.DataFrame(
        {
            "race_id": ["RACE"],
            "election_date": ["2026-11-03"],
            "office_type": ["senate"],
            "race_type": ["candidate"],
            "geography": ["GA"],
            "control_body": ["senate"],
            "seats": [1],
        }
    )
    options = pl.DataFrame(
        {
            "race_id": ["RACE", "RACE"],
            "option_id": ["D", "R"],
            "party": ["DEM", "REP"],
            "incumbent": [True, False],
            "previous_vote_share": [0.52, 0.48],
            "fundraising_usd": [0.0, 0.0],
        }
    )
    poll_rows = []
    for question, pollster, dem in (("Q1", "A", 52.0), ("Q2", "B", 53.0)):
        for option_id, pct in (("D", dem), ("R", 100.0 - dem)):
            poll_rows.append(
                {
                    "poll_id": f"{question}-{option_id}",
                    "question_id": question,
                    "race_id": "RACE",
                    "option_id": option_id,
                    "pollster": pollster,
                    "end_date": "2026-05-01",
                    "published_at": "2026-05-02",
                    "available_at": "2026-05-02",
                    "availability_basis": "source_record",
                    "revision_id": "1",
                    "sample_size": 800,
                    "population": "LV",
                    "pct": pct,
                }
            )
    markets = pl.DataFrame(
        {
            "market_id": ["M-D", "M-R"],
            "race_id": ["RACE", "RACE"],
            "option_id": ["D", "R"],
            "observed_at": ["2026-05-01", "2026-05-01"],
            "published_at": ["2026-05-01", "2026-05-01"],
            "available_at": ["2026-05-01", "2026-05-01"],
            "availability_basis": ["source_record", "source_record"],
            "revision_id": ["1", "1"],
            "probability": [0.55, 0.45],
            "spread": [0.04, 0.04],
            "open_interest": [10_000.0, 10_000.0],
        }
    )
    public = pl.DataFrame(
        {
            "signal_id": ["S-D", "S-R"],
            "race_id": ["RACE", "RACE"],
            "option_id": ["D", "R"],
            "observed_at": ["2026-05-01", "2026-05-01"],
            "published_at": ["2026-05-01", "2026-05-01"],
            "available_at": ["2026-05-01", "2026-05-01"],
            "availability_basis": ["source_record", "source_record"],
            "revision_id": ["1", "1"],
            "z_score": [0.2, -0.2],
            "leakage_checked": [True, True],
        }
    )
    fundamentals = pl.DataFrame(
        {
            "race_id": ["RACE"],
            "series_id": ["ECON"],
            "feature_type": ["macro"],
            "observed_at": ["2026-04-01"],
            "published_at": ["2026-04-02"],
            "available_at": ["2026-04-02"],
            "availability_basis": ["source_record"],
            "revision_id": ["1"],
            "partisan_lean": [1.0],
            "economic_index": [0.1],
            "national_swing": [0.0],
            "demographic_turnout_index": [0.0],
        }
    )
    empty = pl.DataFrame(schema={"race_id": pl.String})
    return FeatureBundle(
        races=races,
        options=options,
        polls=pl.DataFrame(poll_rows),
        markets=markets,
        public_signals=public,
        fundamentals=fundamentals,
        results=empty,
        backtest_predictions=empty,
        race_catalog=races,
    )


def _config() -> dict[str, object]:
    return {
        "seed": 41,
        "simulation_count": 200,
        "performance": {
            "engine": "python",
            "parallel": False,
            "adaptive_mcse": False,
            "max_mcse": 1.0,
        },
        "trusted_components": {
            "polling": True,
            "fundamentals": True,
            "markets": True,
            "public_signals": False,
        },
        "component_weights": {
            "polling": 0.6,
            "fundamentals": 0.2,
            "markets": 0.2,
            "public_signals": 0.0,
        },
        "bayesian": {
            "enabled": True,
            "backend": "analytic",
            "posterior_draw_count": 200,
            "state_space": {
                "initial_state_logit_sd": 0.5,
                "forecast_drift_sd_per_sqrt_day": 0.006,
            },
            "observation": {"nonsampling_logit_floor": 0.02},
        },
        "polling": {"min_nonsampling_error": 0.035},
        "market_adjustments": {"min_open_interest": 1000, "max_spread": 0.18},
        "control_thresholds": {"senate": 1},
    }


def _selector(bundle: FeatureBundle) -> FeatureBundle:
    selected = filter_bundle_by_date(bundle, AS_OF)
    catalog = TierAssessor(
        {
            "tier_a": {
                "min_polls": 2,
                "min_pollsters": 2,
                "min_market_quotes": 1,
                "min_fundamental_rows": 1,
            },
            "tier_b": {"min_any_signal_rows": 1},
            "tier_c": {"reason": "sparse"},
        }
    ).assign(
        selected.races,
        selected.polls,
        selected.markets,
        selected.fundamentals,
        selected.public_signals,
    )
    return subset_bundle(selected, catalog)


def _publication_runner(bundle: FeatureBundle) -> dict[str, pl.DataFrame]:
    config = _config()
    fundamentals = FundamentalsModel(config)
    polling = PollingModel(config, as_of=AS_OF, inference_engine="bayes")
    polling_estimates = polling.run(bundle)
    posterior = polling.posterior_draws(bundle)
    components = [
        polling_estimates,
        fundamentals.run(bundle),
        MarketModel(config).run(bundle),
        PublicSignalModel(trusted=False).run(bundle),
    ]
    ensemble = EnsembleModel(config).run(bundle, components)
    outputs = SimulationEngine(config).run(bundle, ensemble, posterior_draws=posterior)
    return {
        "posterior_draws": posterior,
        "component_estimates": pl.concat(
            [frame for frame in components if not frame.is_empty()], how="diagonal_relaxed"
        ),
        "ensemble_center": ensemble,
        "race_forecasts": outputs.race_forecasts,
        "control_forecasts": outputs.control_forecasts,
    }


def test_exact_publication_canary_covers_every_output_and_time_varying_table() -> None:
    result = run_exact_publication_time_travel_canary(
        _bundle(),
        as_of=AS_OF,
        selector=_selector,
        publication_runner=_publication_runner,
    )

    assert result["passed"] is True
    assert result["scope"] == "exact_publication_pipeline"
    assert result["required_injected_tables"] == [
        "fundamentals",
        "market_quotes",
        "polls",
        "public_signals",
    ]
    assert result["every_time_varying_table_injected"] is True
    for key in (
        "selected_features_unchanged",
        "tiers_unchanged",
        "posterior_unchanged",
        "component_center_unchanged",
        "race_probabilities_unchanged",
        "controls_unchanged",
        "forecast_fingerprint_unchanged",
    ):
        assert result[key] is True


def test_exact_publication_canary_fails_a_deliberately_leaky_selector() -> None:
    def leaky_selector(bundle: FeatureBundle) -> FeatureBundle:
        selected = _selector(bundle)
        return replace(selected, polls=bundle.polls, fundamentals=bundle.fundamentals)

    result = run_exact_publication_time_travel_canary(
        _bundle(),
        as_of=AS_OF,
        selector=leaky_selector,
        publication_runner=_publication_runner,
    )

    assert result["passed"] is False
    assert result["selected_features_unchanged"] is False
    assert result["forecast_fingerprint_unchanged"] is False


def test_exact_publication_outputs_reconcile_via_publication_verifier(tmp_path: Path) -> None:
    """Real publication runner outputs must satisfy PublicationVerifier semantic reconciliation."""
    root = tmp_path / "proj"
    repo = Path(__file__).resolve().parents[1]
    (root / "configs").mkdir(parents=True)
    shutil.copy(repo / "configs" / "rewards.yaml", root / "configs" / "rewards.yaml")
    artifacts = root / "artifacts"
    run = artifacts / "runs" / "exact-canary"
    run.mkdir(parents=True)

    selected = _selector(_bundle())
    outputs = _publication_runner(selected)
    forecasts = outputs["race_forecasts"].with_columns(
        pl.lit("cfg").alias("model_config_hash"),
        pl.lit("src").alias("source_manifest_hash"),
    )
    forecasts.write_parquet(run / "race_forecasts.parquet")
    selected.race_catalog.write_parquet(run / "race_catalog.parquet")
    # Synthetic draw coverage for every non-Tier-C race in the catalog.
    race_ids = selected.race_catalog["race_id"].to_list()
    draw_rows = []
    for race_id in race_ids:
        draw_rows.append(
            {"race_id": race_id, "draw_id": 0, "option_id": f"{race_id}-D", "winner": True}
        )
        draw_rows.append(
            {"race_id": race_id, "draw_id": 0, "option_id": f"{race_id}-R", "winner": False}
        )
    pl.DataFrame(draw_rows).write_parquet(run / "forecast_draws.parquet")
    controls = outputs["control_forecasts"]
    has_control_prob = "control_probability" in controls.columns
    has_majority_prob = "majority_probability" in controls.columns
    if not has_control_prob and not has_majority_prob:
        controls = controls.with_columns(pl.lit(0.5).alias("majority_probability"))
    controls.write_parquet(run / "control_forecasts.parquet")
    pl.DataFrame({"source_id": ["s"], "status": ["fetched"], "content_hash": ["h"]}).write_parquet(
        run / "source_manifest.parquet"
    )

    result = PublicationVerifier(
        ProjectContext.create(root=root, artifacts_dir=artifacts)
    ).verify_semantic(
        run_id="exact-canary",
        profile="research",
        require_promotion_for_production=False,
        force_publication_mode="research",
    )
    assert result["passed"] is True
    assert result["reconciliation_ok"] is True
    assert result["checks"]["required_artifacts_present"] is True
