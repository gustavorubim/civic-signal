"""Plot scales for the distribution plots (seat histogram, tipping bars, KDE grid)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from matplotlib.ticker import PercentFormatter

from civic_signal.reports.plots import PlotGenerator

RNG = np.random.default_rng(7)
DRAWS = 400


def _fixture() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Senate-like slice: 12 two-party races plus one race an independent sometimes wins."""
    races = [f"US-SEN-S{i:02d}-2026" for i in range(12)] + ["US-SEN-NE-2026"]
    draw_rows, forecast_rows = [], []
    for race in races:
        other = "IND" if race.endswith("NE-2026") else "REP"
        shares = np.clip(RNG.normal(0.5, 0.05, DRAWS), 0.01, 0.99)
        for draw, dem in enumerate(shares):
            draw_rows += [
                {"draw_id": draw, "race_id": race, "option_id": f"{race}-D", "party": "DEM",
                 "vote_share": float(dem), "winner": bool(dem > 0.5)},
                {"draw_id": draw, "race_id": race, "option_id": f"{race}-O", "party": other,
                 "vote_share": float(1 - dem), "winner": bool(dem <= 0.5)},
            ]  # fmt: skip
        p = float((shares > 0.5).mean())
        forecast_rows += [
            {"race_id": race, "option_id": f"{race}-D", "name": "Ann Smith Jr.", "party": "DEM",
             "winner_probability": p, "vote_share_mean": float(shares.mean())},
            {"race_id": race, "option_id": f"{race}-O", "name": "Bob Jones", "party": other,
             "winner_probability": 1 - p, "vote_share_mean": float(1 - shares.mean())},
        ]  # fmt: skip
    catalog = pl.DataFrame(
        {"race_id": races, "control_body": ["senate"] * len(races), "seats": [1] * len(races)}
    )
    tipping = json.dumps(
        [{"race_id": r, "tipping_point_probability": 0.3 / (i + 1)} for i, r in enumerate(races)]
    )
    control = pl.DataFrame(
        [
            {
                "control_body": "senate",
                "party": "DEM",
                "control_threshold": 51,
                "holdover_seats": 44,
                "control_probability": 0.42,
                "seat_count_mean": 50.5,
                "seat_count_p10": 48.0,
                "seat_count_p50": 50.0,
                "seat_count_p90": 53.0,
                "control_probability_with_independents": 0.45,
                "independent_caucus_races": '["US-SEN-NE-2026"]',
                "tipping_point_probability": tipping,
                "tipping_point_none_probability": 0.0,
            },
            {
                "control_body": "senate",
                "party": "REP",
                "control_threshold": 50,
                "holdover_seats": 43,
                "control_probability": 0.55,
                "seat_count_mean": 49.0,
                "seat_count_p10": 46.0,
                "seat_count_p50": 49.0,
                "seat_count_p90": 51.0,
                "control_probability_with_independents": 0.55,
                "independent_caucus_races": "[]",
                "tipping_point_probability": tipping,
                "tipping_point_none_probability": 0.0,
            },
        ]
    )
    return catalog, pl.DataFrame(draw_rows), pl.DataFrame(forecast_rows), control


@pytest.fixture
def captured(monkeypatch: pytest.MonkeyPatch) -> list:
    figures: list = []

    def fake_save(fig, path: Path) -> Path:  # keep the figure open for inspection
        figures.append(fig)
        return path

    monkeypatch.setattr(PlotGenerator, "_save", staticmethod(fake_save))
    return figures


def test_seat_histogram_uses_one_seat_bins_percent_axis_and_tight_window(
    tmp_path: Path, captured: list
) -> None:
    catalog, draws, _forecasts, control = _fixture()
    PlotGenerator()._seat_count_histogram(tmp_path, catalog, draws, control)
    ax = captured[0].axes[0]
    bars = ax.patches
    positions = sorted(bar.get_x() + bar.get_width() / 2 for bar in bars)
    assert all(abs(pos - round(pos)) < 1e-9 for pos in positions)  # centred on integers
    assert all(np.diff(positions) >= 1 - 1e-9)  # one bar per seat count
    heights = [bar.get_height() for bar in bars]
    assert sum(heights) == pytest.approx(1.0)  # share of simulations, not raw draw counts
    assert isinstance(ax.yaxis.get_major_formatter(), PercentFormatter)
    lo, hi = ax.get_xlim()
    assert lo > 40  # the IND 0-1 seat spike no longer drags the axis to zero
    assert min(positions) >= lo and max(positions) <= hi
    dashed = [line for line in ax.get_lines() if line.get_linestyle() == "--"]
    assert dashed and dashed[0].get_xdata()[0] == pytest.approx(50.5)  # majority = 51 seats
    text = " ".join(t.get_text() for t in ax.texts)
    assert "P(DEM control) = 42.0%" in text and "45.0%" in text and "80% range 48-53" in text


def test_party_seat_counts_include_draws_with_zero_wins() -> None:
    catalog, draws, _forecasts, _control = _fixture()
    joined = draws.join(catalog, on="race_id")
    ind = PlotGenerator._party_seat_counts(joined, "senate", "IND")
    assert ind.size == DRAWS and set(np.unique(ind)) <= {0, 1}


def test_seat_window_pads_and_has_min_width() -> None:
    assert PlotGenerator._seat_window(np.array([50] * 100)) == (44, 56)
    lo, hi = PlotGenerator._seat_window(np.arange(180, 281))
    assert lo <= 181 and hi >= 279


def test_tipping_bars_top10_sorted_percent_with_headroom(tmp_path: Path, captured: list) -> None:
    _catalog, _draws, forecasts, control = _fixture()
    PlotGenerator()._tipping_point_bars(tmp_path, control, forecasts)
    ax = captured[0].axes[0]
    widths = [bar.get_width() for bar in ax.patches]
    assert len(widths) == 10 and widths == sorted(widths)  # largest at the top
    assert ax.get_xlim()[1] > max(widths) * 1.1  # value labels stay inside the axes
    assert isinstance(ax.xaxis.get_major_formatter(), PercentFormatter)
    labels = [t.get_text() for t in ax.get_yticklabels()]
    assert labels[-1].startswith("S00") and "D win" in labels[-1]


def test_kde_grid_percent_axis_with_50_line_and_odds(tmp_path: Path, captured: list) -> None:
    _catalog, draws, forecasts, _control = _fixture()
    PlotGenerator()._race_vote_share_kde_grid(tmp_path, forecasts, draws)
    axes = [ax for ax in captured[0].axes if ax.has_data()]
    assert axes
    limits = {ax.get_xlim() for ax in axes}
    assert len(limits) == 1  # one shared window across panels
    lo, hi = limits.pop()
    assert lo < 0.5 < hi and abs((0.5 - lo) - (hi - 0.5)) < 1e-9
    for ax in axes:
        assert isinstance(ax.xaxis.get_major_formatter(), PercentFormatter)
        assert any(
            line.get_linestyle() == "--" and line.get_xdata()[0] == 0.5 for line in ax.get_lines()
        )
        assert any("Win odds" in t.get_text() for t in ax.texts)
    texts = " ".join(t.get_text() for ax in axes for t in ax.texts)
    assert "Smith (D)" in texts and "Jr." not in texts


def test_short_labels() -> None:
    assert PlotGenerator._short_race_label("US-SEN-IA-2026") == "IA"
    assert PlotGenerator._short_race_label("US-HOUSE-AZ-01-2026") == "AZ-01"
    assert PlotGenerator._surname("Juan Garcia Jr.") == "Garcia"
