"""Error structure: explicit national/regional error, no double-counted noise (audit P1-3)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl
import yaml

from civic_signal.inference.fundamentals_prior import cap_prior_share_sd
from civic_signal.models import SimulationEngine
from civic_signal.models.polling_bayes import BayesianPollingModel

ROOT = Path(__file__).resolve().parents[1]
DRAWS = 4000
STATES = ["NC", "GA", "MI", "PA", "WI", "AZ", "ME", "OH", "TX", "IA"]


def _model_config() -> dict:
    config = yaml.safe_load((ROOT / "configs" / "model.yaml").read_text())
    config["simulation_count"] = DRAWS
    return config


def _fixture(dem_means: dict[str, float], race_sd: float):
    """Posterior draws for one D-v-R race per state, race-level SD ``race_sd`` only."""
    rng = np.random.default_rng(5)
    rows, catalog, options, ensemble = [], {}, [], []
    for state, mean in dem_means.items():
        race = f"US-SEN-{state}-2026"
        dem = rng.normal(mean, race_sd, DRAWS)
        for suffix, party, share in (("D", "DEM", dem), ("R", "REP", 1.0 - dem)):
            option = f"{race}-{suffix}"
            rows.append(
                pl.DataFrame(
                    {"draw_id": np.arange(DRAWS), "race_id": race, "option_id": option,
                     "latent_share": share}
                )
            )  # fmt: skip
            options.append({"race_id": race, "option_id": option, "party": party})
            ensemble.append(
                {"race_id": race, "option_id": option, "vote_share": float(share.mean()),
                 "uncertainty": race_sd, "component_disagreement": 0.0}
            )  # fmt: skip
        catalog[race] = {"race_id": race, "tier": "A", "geography": state, "office_type": "senate"}
    bundle = SimpleNamespace(
        race_catalog=pl.DataFrame(list(catalog.values())), options=pl.DataFrame(options)
    )
    return bundle, pl.DataFrame(ensemble), pl.concat(rows), catalog


def _dem_shares(engine: SimulationEngine, dem_means: dict[str, float], race_sd: float):
    bundle, ensemble, posterior, catalog = _fixture(dem_means, race_sd)
    national = (
        np.zeros(DRAWS)
        if engine._uses_residual_covariance_only()
        else engine._stream("national").normal(0, engine.national_sigma, DRAWS)
    )
    frame = engine._posterior_draw_frame(
        bundle,
        ensemble,
        posterior,
        catalog=catalog,
        fundamentals={},
        national_error=national,
        systematic_errors=engine._systematic_errors(catalog),
        turnout_multipliers=np.ones(DRAWS),
    )
    dem = frame.filter(pl.col("party") == "DEM").sort(["race_id", "draw_id"])
    wide = dem.pivot(on="race_id", index="draw_id", values="vote_share").drop("draw_id")
    return wide.to_numpy(), dem


def _mean_pairwise_correlation(matrix: np.ndarray) -> float:
    corr = np.corrcoef(matrix.T)
    return float(corr[np.triu_indices_from(corr, 1)].mean())


def test_configured_error_model_is_national_regional() -> None:
    config = _model_config()
    engine = SimulationEngine(config)
    assert engine.residual_covariance_mode == "national_regional"
    assert 0.025 <= engine.national_sigma <= 0.03
    assert engine.region_sigma == 0.015
    assert engine.posterior_local_sigma_scale == 0.0
    # A learned covariance is ignored in national_regional mode.
    covariance = pl.DataFrame({"row_group": ["NC"], "column_group": ["NC"], "covariance": [0.01]})
    learned = SimulationEngine(config, residual_covariance=covariance)
    assert not learned._uses_learned_covariance()
    assert learned._systematic_error_mode() == "configured_national_region_office"


def test_pairwise_correlation_rises_above_point_three() -> None:
    means = dict.fromkeys(STATES, 0.5)
    new, _ = _dem_shares(SimulationEngine(_model_config()), means, race_sd=0.02)
    old_config = _model_config()
    old_config["correlation"].update(
        {"national_sigma": 0.015, "region_sigma": 0.010, "office_sigma": 0.008,
         "residual_covariance_mode": "residual_covariance_only"}
    )  # fmt: skip
    old_config["uncertainty"]["posterior_local_sigma_scale"] = 1.0
    covariance = pl.DataFrame(
        {"row_group": STATES, "column_group": STATES, "covariance": [0.0004] * len(STATES)}
    )
    old, _ = _dem_shares(
        SimulationEngine(old_config, residual_covariance=covariance), means, race_sd=0.02
    )
    assert _mean_pairwise_correlation(new) > 0.3
    assert _mean_pairwise_correlation(old) < 0.2
    assert new.std(axis=0).mean() < old.std(axis=0).mean()  # no double-counted local noise


def test_plus_ten_race_with_good_polling_is_at_least_ninety_percent() -> None:
    # D+10 margin = 55% two-party share; race-level SD 2 pts from a well-polled race.
    means = {"NC": 0.55, **dict.fromkeys(STATES[1:], 0.5)}
    _, dem = _dem_shares(SimulationEngine(_model_config()), means, race_sd=0.02)
    nc = dem.filter(pl.col("race_id") == "US-SEN-NC-2026")
    assert nc["winner"].mean() >= 0.90


def test_posterior_local_noise_scale() -> None:
    race = {"tier": "A"}
    estimate = {"uncertainty": 0.04, "component_disagreement": 0.0}
    assert (
        SimulationEngine(
            {"uncertainty": {"posterior_local_sigma_scale": 0.0}}
        )._posterior_local_sigma(race, estimate)
        == 0.0
    )
    assert SimulationEngine({})._posterior_local_sigma(race, estimate) == 0.035


def test_prior_share_sd_cap() -> None:
    frame = pl.DataFrame({"mean_share": [0.5, 0.5], "sd_logit": [0.8, 0.1]})
    capped = cap_prior_share_sd(frame, 0.06)
    assert capped["sd_logit"].to_list() == [0.24, 0.1]  # 0.06 / 0.25 = 0.24 logit
    assert capped["sd_capped"].to_list() == [True, False]
    assert cap_prior_share_sd(frame, None)["sd_logit"].to_list() == [0.8, 0.1]


def test_race_forecast_floor_is_configurable() -> None:
    base = {"bayesian": {"backend": "analytic"}}
    default = BayesianPollingModel(config=base)
    assert default.forecast_floor_sd == default.min_nonsampling_error
    tight = BayesianPollingModel(
        config={
            "bayesian": {"backend": "analytic", "state_space": {"race_forecast_floor_sd": 0.02}}
        }
    )
    assert tight._forecast_logit_sd(0.0, 0.5) < default._forecast_logit_sd(0.0, 0.5)
