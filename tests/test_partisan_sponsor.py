"""Partisan-sponsor house effect (audit task 2.5, P1-4)."""

from __future__ import annotations

from datetime import date

import pytest

from civic_signal.models.polling_kalman import KalmanPollingModel

CONFIG = {"polling": {"partisan_sponsor_shift": 0.02, "partisan_weight": 0.5}}


def _row(option: str, pct: float, sponsor: str | None) -> dict[str, object]:
    return {
        "poll_id": "p",
        "pollster": "X",
        "end_date": date(2026, 10, 1),
        "race_id": "US-SEN-GA-2026",
        "option_id": f"US-SEN-GA-2026-{option}",
        "pct": pct,
        "sample_size": 800,
        "population": "lv",
        "methodology": "mixed",
        "sponsor_class": "partisan" if sponsor else "nonpartisan",
        "sponsor_party": sponsor,
    }


def test_dem_sponsored_d_plus_6_counts_as_d_plus_4_with_half_weight() -> None:
    model = KalmanPollingModel(CONFIG)
    dem = model._observation(_row("D", 53.0, "DEM"), {})
    rep = model._observation(_row("R", 47.0, "DEM"), {})
    assert (dem.adjusted_share - rep.adjusted_share) == pytest.approx(0.04)
    neutral = model._observation(_row("D", 53.0, None), {})
    assert dem.effective_sample_size == pytest.approx(0.5 * neutral.effective_sample_size)


def test_rep_sponsor_moves_toward_dem_and_nonpartisan_unchanged() -> None:
    model = KalmanPollingModel(CONFIG)
    dem = model._observation(_row("D", 47.0, "REP"), {})
    assert dem.adjusted_share == pytest.approx(0.48)
    neutral = model._observation(_row("D", 47.0, None), {})
    assert neutral.adjusted_share == pytest.approx(0.47)
    ind = model._observation({**_row("I", 47.0, "DEM"), "option_id": "US-SEN-NE-2026-I"}, {})
    assert ind.adjusted_share == pytest.approx(0.46)


def test_shift_disabled_by_default() -> None:
    model = KalmanPollingModel({})
    obs = model._observation(_row("D", 53.0, "DEM"), {})
    assert obs.adjusted_share == pytest.approx(0.53)
    internal = {**_row("D", 53.0, None), "sponsor_class": "internal"}
    base = model._observation(_row("D", 53.0, None), {}).effective_sample_size
    assert model._observation(internal, {}).effective_sample_size == pytest.approx(0.85 * base)
    assert KalmanPollingModel._option_side("RACE-X") == 0.0
