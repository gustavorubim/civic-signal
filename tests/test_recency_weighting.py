"""Analytic-backend recency weighting (audit task 1.5, P0-1b)."""

from __future__ import annotations

from datetime import date

import pytest

from civic_signal.models.common import inv_logit
from civic_signal.models.polling_bayes import BayesianPollingModel
from civic_signal.models.polling_kalman import PollObservation

AS_OF = date(2026, 10, 19)


def _obs(end: date, share: float, poll_id: str = "p") -> PollObservation:
    return PollObservation(
        poll_id=poll_id,
        pollster="X",
        end_date=end,
        observed_share=share,
        adjusted_share=share,
        observation_variance=0.0004,
        effective_sample_size=800.0,
        house_effect=0.0,
    )


def _model(half_life: float | None = 18, max_age: int | None = None) -> BayesianPollingModel:
    return BayesianPollingModel(
        config={
            "bayesian": {
                "backend": "analytic",
                "polling": {"analytic_half_life_days": half_life, "max_poll_age_days": max_age},
            }
        },
        as_of=str(AS_OF),
    )


def test_weights_are_one_at_age_zero_and_half_at_half_life() -> None:
    model = _model(half_life=18)
    assert model._recency_weight(_obs(AS_OF, 0.5), AS_OF) == pytest.approx(1.0)
    assert model._recency_weight(_obs(date(2026, 10, 1), 0.5), AS_OF) == pytest.approx(0.5)
    assert model._recency_weight(_obs(date(2026, 10, 1), 0.5), None) == 1.0
    assert _model(half_life=None)._recency_weight(_obs(date(2024, 1, 1), 0.5), AS_OF) == 1.0


def test_old_poll_has_negligible_influence_against_fresh_poll() -> None:
    model = _model(half_life=18)
    fresh = _obs(date(2026, 10, 15), 0.55, "fresh")
    stale = _obs(date(2024, 11, 1), 0.40, "stale")
    mean_logit, _ = model._posterior_logit(0.5, [fresh, stale], prior_sd_logit=0.5, as_of=AS_OF)
    fresh_only, _ = model._posterior_logit(0.5, [fresh], prior_sd_logit=0.5, as_of=AS_OF)
    assert inv_logit(mean_logit) == pytest.approx(inv_logit(fresh_only), abs=1e-4)
    unweighted, _ = _model(half_life=None)._posterior_logit(
        0.5, [fresh, stale], prior_sd_logit=0.5, as_of=AS_OF
    )
    assert inv_logit(unweighted) < 0.5  # equal weights average the two polls


def test_half_life_poll_counts_half_as_much() -> None:
    model = _model(half_life=18)
    old = _obs(date(2026, 10, 1), 0.45)
    _, sd_one_old = model._posterior_logit(0.5, [old], prior_sd_logit=10.0, as_of=AS_OF)
    _, sd_one_new = model._posterior_logit(
        0.5, [_obs(AS_OF, 0.45)], prior_sd_logit=10.0, as_of=AS_OF
    )
    # precision halves => variance doubles (prior is effectively flat here)
    assert sd_one_old**2 == pytest.approx(2 * sd_one_new**2, rel=1e-3)


def test_max_poll_age_window() -> None:
    model = _model(half_life=18, max_age=120)
    assert model._within_max_age(_obs(date(2026, 9, 1), 0.5), AS_OF)
    assert not model._within_max_age(_obs(date(2026, 5, 1), 0.5), AS_OF)
    assert model._within_max_age(_obs(date(2026, 5, 1), 0.5), None)
    assert _model(max_age=None)._within_max_age(_obs(date(2020, 1, 1), 0.5), AS_OF)


def test_live_scenario_as_of_defaults_to_today_capped_at_election_eve() -> None:
    from civic_signal.config.scenarios import Scenario

    live = Scenario(
        "senate_2026_state",
        {"as_of_policy": "today", "election_date": "2026-11-03", "default_as_of": "2026-11-02"},
    )
    assert live.resolve_as_of(None, today=date(2026, 10, 19)) == "2026-10-19"
    assert live.resolve_as_of(None, today=date(2026, 11, 20)) == "2026-11-02"
    assert live.resolve_as_of("2026-10-01", today=date(2026, 10, 19)) == "2026-10-01"
    fixture = Scenario("senate_2024_state", {"default_as_of": "2024-11-04"})
    assert fixture.resolve_as_of(None) == "2024-11-04"
    assert fixture.as_of_policy == "default" and fixture.election_date is None
    no_election = Scenario("x", {"as_of_policy": "today"})
    assert no_election.resolve_as_of(None, today=date(2026, 10, 19)) == "2026-10-19"


def test_horizon_equals_election_date_minus_as_of() -> None:
    import polars as pl

    model = _model()
    catalog = pl.DataFrame({"race_id": ["R"], "election_date": [date(2026, 11, 3)]})
    election_days = model._election_day_by_race(catalog)
    meta = model._forecast_horizon_metadata(election_days, date(2026, 10, 19), {("R", "R-D")})
    assert meta["max_horizon_days"] == 15
