"""Tiny real NUTS smoke used by scientific CI."""

from __future__ import annotations

from typing import Any

import numpy as np

from civic_signal.inference.nuts import NutsConfig, fit_nuts
from civic_signal.inference.state_space import HyperPriors, StateSpaceData


def minimal_state_space_data() -> StateSpaceData:
    """Two-option race, one pollster, three poll observations (legacy static path)."""
    # poll_s indexes into race_option_keys / prior_logit (option latent index).
    poll_s = np.array([0, 1, 0], dtype=np.int64)
    poll_j = np.array([0, 0, 0], dtype=np.int64)
    poll_t = np.array([0, 0, 1], dtype=np.int64)
    poll_o = np.array([0, 1, 0], dtype=np.int64)
    # Mild logit observations around 0 (50/50).
    poll_logit_y = np.array([0.1, -0.1, 0.05], dtype=np.float64)
    poll_kappa = np.array([0.15, 0.15, 0.2], dtype=np.float64)
    prior_logit = np.array([0.0, 0.0], dtype=np.float64)
    option_office = np.array([0, 0], dtype=np.int64)
    option_geography = np.array([0, 0], dtype=np.int64)
    option_race = np.array([0, 0], dtype=np.int64)
    return StateSpaceData(
        poll_t=poll_t,
        poll_s=poll_s,
        poll_j=poll_j,
        poll_o=poll_o,
        poll_logit_y=poll_logit_y,
        poll_kappa=poll_kappa,
        prior_logit=prior_logit,
        option_office=option_office,
        option_geography=option_geography,
        option_race=option_race,
        race_option_keys=[("race-a", "DEM"), ("race-a", "REP")],
        pollster_ids=["pollster-a"],
        office_ids=["senate"],
        geography_ids=["AA"],
        race_ids=["race-a"],
        dims=(2, 2, 1),
        metadata={"smoke": True, "temporal_model": "static_recency_weighted"},
    )


def run_nuts_smoke(
    *,
    num_warmup: int = 20,
    num_samples: int = 20,
    num_chains: int = 1,
    seed: int = 7,
    wall_clock_timeout_seconds: float = 60.0,
) -> dict[str, Any]:
    """Run a minimal NUTS fit and return a compact health report."""
    data = minimal_state_space_data()
    config = NutsConfig(
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        chain_method="sequential",
        target_accept_prob=0.8,
        wall_clock_timeout_seconds=wall_clock_timeout_seconds,
    )
    result = fit_nuts(data, hyperpriors=HyperPriors(), config=config, seed=seed)
    samples = result.samples
    state = np.asarray(samples.get("state_logit", np.array([])))
    divergences = int(result.diagnostics.get("divergences", 0) or 0)
    return {
        "ok": state.size > 0 and divergences == 0,
        "elapsed_seconds": float(result.elapsed_seconds),
        "sample_keys": sorted(samples.keys()),
        "state_logit_shape": list(state.shape),
        "divergences": divergences,
        "diagnostics": {
            key: result.diagnostics[key]
            for key in (
                "engine",
                "num_warmup",
                "num_samples",
                "num_chains",
                "divergences",
            )
            if key in result.diagnostics
        },
    }
