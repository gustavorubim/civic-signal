from __future__ import annotations

# ruff: noqa: I001

import json
from itertools import pairwise
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import polars as pl

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, PercentFormatter


PARTY_COLORS = {
    "DEM": "#2f78bc",
    "REP": "#c44e52",
    "YES": "#3a8f5d",
    "NO": "#777777",
}
PAPER_COLOR = "#f6f4ef"
AXIS_COLOR = "#ffffff"
GRID_COLOR = "#ded9cf"
TEXT_COLOR = "#242424"
MUTED_COLOR = "#6b6b6b"
GOLD_COLOR = "#c87922"


class PlotGenerator:
    """Static plot generation for calibration and projection diagnostics."""

    def render_all(
        self,
        artifact_dir: Path,
        race_catalog: pl.DataFrame,
        race_forecasts: pl.DataFrame,
        forecast_draws: pl.DataFrame,
        control_forecasts: pl.DataFrame,
        ecosystem_forecasts: pl.DataFrame,
        backtest_predictions: pl.DataFrame,
        backtest_payload: dict[str, Any],
        methodology_benchmark: dict[str, Any] | None = None,
        poll_trajectory: pl.DataFrame | None = None,
        posterior_draws: pl.DataFrame | None = None,
        posterior_diagnostics: dict[str, Any] | None = None,
        fundamentals_prior: pl.DataFrame | None = None,
    ) -> dict[str, list[dict[str, str]]]:
        plot_dir = artifact_dir / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        manifest: dict[str, list[dict[str, str]]] = {
            "distribution": [],
            "calibration": [],
            "projection": [],
            "trajectory": [],
            "drivers": [],
            "stability": [],
            "model_quality": [],
            "posterior": [],
            "fundamentals_prior": [],
            "benchmark": [],
        }
        self._add(
            manifest,
            "distribution",
            self._seat_count_histogram(plot_dir, race_catalog, forecast_draws, control_forecasts),
            "Seat-count distribution per party",
        )
        self._add(
            manifest,
            "distribution",
            self._race_vote_share_kde_grid(plot_dir, race_forecasts, forecast_draws),
            "Vote-share KDEs for the closest races",
        )
        self._add(
            manifest,
            "drivers",
            self._tipping_point_bars(plot_dir, control_forecasts, race_forecasts),
            "Tipping-point race ranking",
        )
        self._add(
            manifest,
            "drivers",
            self._driver_waterfall(plot_dir, race_forecasts),
            "Component drivers for closest races",
        )
        self._add(
            manifest,
            "calibration",
            self._reliability_diagram_with_band(plot_dir, backtest_predictions),
            "Reliability diagram with 95% Wilson band",
        )
        self._add(
            manifest,
            "calibration",
            self._calibration_curve(plot_dir, backtest_predictions),
            "Calibration curve",
        )
        self._add(
            manifest,
            "calibration",
            self._brier_by_component(plot_dir, backtest_payload),
            "Brier score by model component",
        )
        self._add(
            manifest,
            "calibration",
            self._interval_coverage(plot_dir, backtest_payload),
            "Historical interval coverage",
        )
        self._add(
            manifest,
            "projection",
            self._race_probability_bars(plot_dir, race_forecasts),
            "Winner probabilities by race",
        )
        self._add(
            manifest,
            "projection",
            self._vote_share_intervals(plot_dir, race_forecasts),
            "Projected vote-share intervals",
        )
        self._add(
            manifest,
            "projection",
            self._control_projection(plot_dir, control_forecasts),
            "Seat/control projections",
        )
        self._add(
            manifest,
            "projection",
            self._turnout_and_recount(plot_dir, ecosystem_forecasts),
            "Turnout and recount-risk projections",
        )
        self._add(
            manifest,
            "projection",
            self._tier_coverage(plot_dir, race_catalog),
            "Forecast coverage by tier",
        )
        self._add(
            manifest,
            "projection",
            self._electoral_college_distribution(plot_dir, race_catalog, forecast_draws),
            "Modeled Electoral College distribution",
        )
        self._add(
            manifest,
            "projection",
            self._electoral_college_swarm(plot_dir, race_catalog, forecast_draws),
            "Top-line Electoral College simulation swarm",
        )
        self._add(
            manifest,
            "trajectory",
            self._polling_state_trajectory(
                plot_dir, poll_trajectory if poll_trajectory is not None else pl.DataFrame()
            ),
            "Kalman polling trajectories with uncertainty",
        )
        self._add(
            manifest,
            "trajectory",
            self._polling_trajectory(plot_dir, backtest_predictions),
            "Polling trajectory from rolling-origin cuts",
        )
        self._add(
            manifest,
            "stability",
            self._simulation_probability_convergence(plot_dir, forecast_draws),
            "Simulation probability convergence",
        )
        self._add(
            manifest,
            "model_quality",
            self._electoral_college_chain_trace(plot_dir, race_catalog, forecast_draws),
            "Posterior simulation chain traces",
        )
        self._add(
            manifest,
            "model_quality",
            self._kalman_posterior_uncertainty(
                plot_dir, poll_trajectory if poll_trajectory is not None else pl.DataFrame()
            ),
            "Kalman posterior uncertainty",
        )
        self._add(
            manifest,
            "posterior",
            self._posterior_share_intervals(
                plot_dir, posterior_draws if posterior_draws is not None else pl.DataFrame()
            ),
            "Bayesian latent vote-share posterior intervals",
        )
        self._add(
            manifest,
            "posterior",
            self._posterior_diagnostic_bars(plot_dir, posterior_diagnostics or {}),
            "Bayesian posterior diagnostics",
        )
        self._add(
            manifest,
            "fundamentals_prior",
            self._fundamentals_prior_intervals(
                plot_dir, fundamentals_prior if fundamentals_prior is not None else pl.DataFrame()
            ),
            "Fundamentals prior intervals",
        )
        self._add(
            manifest,
            "benchmark",
            self._methodology_benchmark(plot_dir, methodology_benchmark or {}),
            "Silver/FiveThirtyEight methodology benchmark",
        )
        return manifest

    @staticmethod
    def write_manifest(manifest: dict[str, list[dict[str, str]]], artifact_dir: Path) -> None:
        path = artifact_dir / "plot_manifest.json"
        path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    @staticmethod
    def _add(
        manifest: dict[str, list[dict[str, str]]],
        category: str,
        path: Path | None,
        title: str,
    ) -> None:
        if path is None:
            return
        manifest[category].append({"title": title, "path": f"plots/{path.name}"})

    def _calibration_curve(self, plot_dir: Path, frame: pl.DataFrame) -> Path | None:
        if frame.is_empty() or "ensemble_probability" not in frame.columns:
            return None
        df = frame.select(["ensemble_probability", "actual_winner"]).drop_nulls()
        if df.is_empty():
            return None
        probability = df["ensemble_probability"].cast(pl.Float64).to_numpy()
        actual = df["actual_winner"].cast(pl.Float64).to_numpy()
        bins = np.linspace(0, 1, 6)
        xs: list[float] = []
        ys: list[float] = []
        for lower, upper in pairwise(bins):
            mask = (probability >= lower) & (
                probability < upper if upper < 1 else probability <= upper
            )
            if np.any(mask):
                xs.append(float(np.mean(probability[mask])))
                ys.append(float(np.mean(actual[mask])))
        fig, ax = plt.subplots(figsize=(7.5, 5.2), dpi=150)
        ax.plot([0, 1], [0, 1], color=MUTED_COLOR, linestyle="--", linewidth=1)
        ax.scatter(xs, ys, color=PARTY_COLORS["DEM"], s=72, zorder=3)
        for x, y in zip(xs, ys, strict=True):
            ax.text(x + 0.025, y, f"{y:.0%}", va="center", fontsize=9, color=TEXT_COLOR)
        ax.text(
            0.02,
            0.95,
            f"n={len(probability)} held-out rows",
            transform=ax.transAxes,
            color=MUTED_COLOR,
            fontsize=10,
            va="top",
        )
        ax.set_xlabel("Mean forecast probability")
        ax.set_ylabel("Observed win rate")
        ax.set_title("Calibration Curve", loc="left", fontweight="bold")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.xaxis.set_major_formatter(PercentFormatter(1.0))
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        self._style_axis(ax)
        return self._save(fig, plot_dir / "calibration_curve.png")

    def _brier_by_component(self, plot_dir: Path, backtest_payload: dict[str, Any]) -> Path | None:
        metrics = backtest_payload.get("metrics", {})
        rows = [
            (component, values["brier"])
            for component, values in metrics.items()
            if isinstance(values, dict) and "brier" in values
        ]
        if not rows:
            return None
        rows = sorted(rows, key=lambda item: item[1], reverse=True)
        labels, values = zip(*rows, strict=True)
        fig, ax = plt.subplots(figsize=(8.4, 5.2), dpi=150)
        colors = [PARTY_COLORS["DEM"] if label == "ensemble" else "#8d8d8d" for label in labels]
        ax.barh(labels, values, color=colors)
        for idx, value in enumerate(values):
            ax.text(value + 0.002, idx, f"{value:.3f}", va="center", fontsize=9)
        ax.set_ylabel("Brier score")
        ax.set_xlabel("Lower is better")
        ax.set_title("Backtest Brier Score by Component", loc="left", fontweight="bold")
        self._style_axis(ax)
        return self._save(fig, plot_dir / "brier_by_component.png")

    def _interval_coverage(self, plot_dir: Path, backtest_payload: dict[str, Any]) -> Path | None:
        coverage = (
            backtest_payload.get("metrics", {}).get("ensemble", {}).get("interval_90_coverage")
        )
        if coverage is None:
            return None
        observed = float(coverage)
        fig, ax = plt.subplots(figsize=(6.4, 5.1), dpi=150)
        ax.bar(
            ["Nominal 90%", "Observed"],
            [0.9, observed],
            color=["#999999", PARTY_COLORS["DEM"] if observed >= 0.85 else GOLD_COLOR],
        )
        ax.axhline(0.9, color=MUTED_COLOR, linestyle="--", linewidth=1)
        ax.text(1, min(1.0, observed + 0.04), f"{observed:.0%}", ha="center", fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Coverage")
        ax.set_title("Historical Interval Coverage", loc="left", fontweight="bold")
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        self._style_axis(ax)
        return self._save(fig, plot_dir / "interval_coverage.png")

    def _race_probability_bars(self, plot_dir: Path, race_forecasts: pl.DataFrame) -> Path | None:
        if race_forecasts.is_empty() or "winner_probability" not in race_forecasts.columns:
            return None
        frame = race_forecasts.filter(pl.col("winner_probability").is_not_null())
        if frame.is_empty():
            return None
        frame = (
            frame.sort(["race_id", "winner_probability"], descending=[False, True])
            .group_by("race_id", maintain_order=True)
            .head(1)
            .with_columns((pl.col("winner_probability") - 0.5).abs().alias("_distance"))
            .sort("_distance")
            .head(24)
        )
        labels = [self._option_label(row) for row in frame.iter_rows(named=True)]
        values = frame["winner_probability"].to_list()
        colors = [
            PARTY_COLORS.get(str(row["party"]), "#f28e2b") for row in frame.iter_rows(named=True)
        ]
        fig, ax = plt.subplots(figsize=(8.4, 6.0), dpi=150)
        ax.barh(labels, values, color=colors)
        ax.axvline(0.5, color=MUTED_COLOR, linestyle="--", linewidth=1)
        for idx, value in enumerate(values):
            x = min(0.98, float(value) + 0.025)
            ax.text(x, idx, f"{float(value):.1%}", va="center", fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Winner probability")
        ax.set_title("Most Competitive Race Winner Probabilities", loc="left", fontweight="bold")
        ax.xaxis.set_major_formatter(PercentFormatter(1.0))
        ax.invert_yaxis()
        self._style_axis(ax)
        return self._save(fig, plot_dir / "race_probability_bars.png")

    def _vote_share_intervals(self, plot_dir: Path, race_forecasts: pl.DataFrame) -> Path | None:
        required = {"vote_share_mean", "vote_share_p05", "vote_share_p95"}
        if race_forecasts.is_empty() or not required.issubset(set(race_forecasts.columns)):
            return None
        frame = race_forecasts.filter(pl.col("vote_share_mean").is_not_null())
        if frame.is_empty():
            return None
        sorted_frame = (
            frame.with_columns((pl.col("vote_share_mean") - 0.5).abs().alias("_distance"))
            .sort("_distance")
            .head(24)
        )
        labels = [self._option_label(row) for row in sorted_frame.iter_rows(named=True)]
        mean = np.array(sorted_frame["vote_share_mean"].to_list())
        low = np.array(sorted_frame["vote_share_p05"].to_list())
        high = np.array(sorted_frame["vote_share_p95"].to_list())
        fig, ax = plt.subplots(figsize=(8.4, 6.0), dpi=150)
        for idx, row in enumerate(sorted_frame.iter_rows(named=True)):
            color = PARTY_COLORS.get(str(row.get("party")), "#2f4b7c")
            ax.errorbar(
                float(mean[idx]),
                labels[idx],
                xerr=[[float(mean[idx] - low[idx])], [float(high[idx] - mean[idx])]],
                fmt="o",
                color=color,
                ecolor=color,
                elinewidth=2.4,
                capsize=4,
                markersize=6,
            )
            ax.text(
                min(0.98, float(high[idx]) + 0.012),
                labels[idx],
                f"{float(mean[idx]):.1%}",
                va="center",
                fontsize=8,
                color=TEXT_COLOR,
            )
        ax.axvline(0.5, color=MUTED_COLOR, linestyle="--", linewidth=1)
        xmin = max(0, min(float(low.min()) - 0.04, 0.48))
        xmax = min(1, max(float(high.max()) + 0.07, 0.52))
        ax.set_xlim(xmin, xmax)
        ax.set_xlabel("Projected vote share with 90% interval")
        ax.set_title("Most Competitive Vote-Share Intervals", loc="left", fontweight="bold")
        ax.xaxis.set_major_formatter(PercentFormatter(1.0))
        ax.invert_yaxis()
        self._style_axis(ax)
        return self._save(fig, plot_dir / "vote_share_intervals.png")

    def _control_projection(self, plot_dir: Path, control_forecasts: pl.DataFrame) -> Path | None:
        if control_forecasts.is_empty():
            return None
        labels = [
            f"{row['control_body']} | {row['party']}"
            for row in control_forecasts.iter_rows(named=True)
        ]
        values = control_forecasts["seat_count_mean"].to_list()
        colors = [
            PARTY_COLORS.get(str(row["party"]), "#e15759")
            for row in control_forecasts.iter_rows(named=True)
        ]
        fig, ax = plt.subplots(figsize=(8.2, max(4.2, len(labels) * 0.6)), dpi=150)
        ax.barh(labels, values, color=colors)
        for idx, row in enumerate(control_forecasts.iter_rows(named=True)):
            value = float(row["seat_count_mean"])
            ax.text(value + 0.15, idx, f"{value:.1f}", va="center", fontsize=10)
        threshold = (
            int(control_forecasts["control_threshold"].max())
            if "control_threshold" in control_forecasts.columns
            else None
        )
        modeled = (
            int(control_forecasts["modeled_seats"].max())
            if "modeled_seats" in control_forecasts.columns
            else None
        )
        if threshold is not None and modeled is not None:
            if threshold <= max(float(max(values)), float(modeled)) * 1.1:
                ax.axvline(threshold, color=TEXT_COLOR, linewidth=1.2)
            else:
                ax.text(
                    0.02,
                    0.94,
                    f"National threshold {threshold} is outside this {modeled}-seat slice.",
                    transform=ax.transAxes,
                    color=GOLD_COLOR,
                    fontsize=10,
                    va="top",
                )
        ax.set_xlabel("Mean projected seats/wins in modeled races")
        ax.set_title("Control Projection", loc="left", fontweight="bold")
        self._style_axis(ax)
        return self._save(fig, plot_dir / "control_projection.png")

    def _turnout_and_recount(
        self, plot_dir: Path, ecosystem_forecasts: pl.DataFrame
    ) -> Path | None:
        if (
            ecosystem_forecasts.is_empty()
            or "recount_probability" not in ecosystem_forecasts.columns
        ):
            return None
        frame = (
            ecosystem_forecasts.filter(pl.col("recount_probability").is_not_null())
            .sort("recount_probability", descending=True)
            .head(24)
        )
        if frame.is_empty():
            return None
        labels = frame["race_id"].to_list()
        recount = frame["recount_probability"].to_list()
        fig, ax = plt.subplots(figsize=(8.4, 6.0), dpi=150)
        ax.barh(labels, recount, color="#76b7b2")
        for idx, value in enumerate(recount):
            ax.text(
                min(0.98, float(value) + 0.02),
                idx,
                f"{float(value):.1%}",
                va="center",
                fontsize=8,
            )
        ax.set_xlim(0, 1)
        ax.set_xlabel("Recount probability")
        ax.set_title("Highest Recount-Risk Races", loc="left", fontweight="bold")
        ax.xaxis.set_major_formatter(PercentFormatter(1.0))
        ax.text(
            0,
            1.04,
            "This is a close-margin proxy unless a calibrated recount model is available.",
            transform=ax.transAxes,
            color=MUTED_COLOR,
            fontsize=10,
        )
        ax.invert_yaxis()
        self._style_axis(ax)
        return self._save(fig, plot_dir / "turnout_recount_risk.png")

    def _tier_coverage(self, plot_dir: Path, race_catalog: pl.DataFrame) -> Path | None:
        if race_catalog.is_empty():
            return None
        counts = race_catalog.group_by("tier").agg(pl.len().alias("count")).sort("tier")
        fig, ax = plt.subplots(figsize=(6.4, 5), dpi=150)
        bars = ax.bar(counts["tier"].to_list(), counts["count"].to_list(), color="#b07aa1")
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.04,
                f"{int(bar.get_height())}",
                ha="center",
                va="bottom",
                fontsize=10,
            )
        ax.set_ylabel("Race count")
        ax.set_title("Forecast Coverage by Tier", loc="left", fontweight="bold")
        self._style_axis(ax)
        return self._save(fig, plot_dir / "tier_coverage.png")

    def _electoral_college_distribution(
        self, plot_dir: Path, race_catalog: pl.DataFrame, forecast_draws: pl.DataFrame
    ) -> Path | None:
        seat_counts = self._presidential_seat_counts(race_catalog, forecast_draws)
        if seat_counts.is_empty():
            return None
        parties = seat_counts["party"].unique().sort().to_list()
        modeled_seats = self._modeled_presidential_seats(race_catalog)
        fig, ax = plt.subplots(figsize=(8.4, 5.2), dpi=150)
        bins: int | np.ndarray = 20
        if modeled_seats <= 80:
            bins = np.arange(-0.5, modeled_seats + 1.5, 1)
        for party in parties:
            values = seat_counts.filter(pl.col("party") == party)["electoral_votes"].to_numpy()
            ax.hist(
                values,
                bins=bins,
                alpha=0.58,
                label=str(party),
                color=PARTY_COLORS.get(str(party), "#999999"),
            )
        threshold = 270 if modeled_seats >= 270 else modeled_seats / 2
        threshold_label = "270 EV threshold" if modeled_seats >= 270 else "modeled-slice majority"
        ax.axvline(threshold, color=TEXT_COLOR, linestyle="--", linewidth=1.2)
        ax.text(
            threshold,
            ax.get_ylim()[1] * 0.92,
            threshold_label,
            rotation=90,
            va="top",
            ha="right",
            color=TEXT_COLOR,
            fontsize=9,
        )
        if modeled_seats < 270:
            ax.text(
                0.02,
                0.94,
                f"Only {modeled_seats} electoral votes are modeled in this scenario.",
                transform=ax.transAxes,
                color=GOLD_COLOR,
                fontsize=10,
                va="top",
            )
            ax.set_xlim(-0.5, modeled_seats + 0.5)
        ax.set_xlabel("Modeled electoral votes")
        ax.set_ylabel("Draw count")
        ax.set_title("Electoral College Distribution", loc="left", fontweight="bold")
        ax.legend()
        self._style_axis(ax)
        return self._save(fig, plot_dir / "electoral_college_distribution.png")

    def _electoral_college_swarm(
        self, plot_dir: Path, race_catalog: pl.DataFrame, forecast_draws: pl.DataFrame
    ) -> Path | None:
        seat_counts = self._presidential_seat_counts(race_catalog, forecast_draws)
        if seat_counts.is_empty():
            return None
        frame = seat_counts.sort(["party", "draw_id"])
        if frame.height > 100:
            frame = frame.group_by("party", maintain_order=True).head(50)
        modeled_seats = self._modeled_presidential_seats(race_catalog)
        fig, ax = plt.subplots(figsize=(9.2, 4.2), dpi=150)
        parties = frame["party"].unique().sort().to_list()
        for party_index, party in enumerate(parties):
            values = frame.filter(pl.col("party") == party)["electoral_votes"].to_numpy()
            rng = np.random.default_rng(20260508 + len(values))
            y = rng.normal(loc=party_index * 0.25, scale=0.045, size=len(values))
            ax.scatter(
                values,
                y,
                s=46,
                alpha=0.72,
                color=PARTY_COLORS.get(str(party), "#999999"),
                edgecolor="white",
                linewidth=0.5,
                label=str(party),
            )
        threshold = 270 if modeled_seats >= 270 else modeled_seats / 2
        ax.axvline(threshold, color=TEXT_COLOR, linewidth=1.2)
        if modeled_seats < 270:
            ax.text(
                0.02,
                0.91,
                "Representative draw swarm for the modeled state slice only.",
                transform=ax.transAxes,
                color=GOLD_COLOR,
                fontsize=10,
                va="top",
            )
            ax.set_xlim(-0.5, modeled_seats + 0.5)
        ax.set_yticks([])
        ax.set_xlabel("Electoral votes in representative simulation draws")
        ax.set_title("Top-Line Simulation Swarm", loc="left", fontweight="bold")
        ax.legend(loc="upper right", frameon=False)
        self._style_axis(ax)
        return self._save(fig, plot_dir / "topline_electoral_swarm.png")

    def _electoral_college_chain_trace(
        self, plot_dir: Path, race_catalog: pl.DataFrame, forecast_draws: pl.DataFrame
    ) -> Path | None:
        seat_counts = self._presidential_seat_counts(race_catalog, forecast_draws)
        if seat_counts.is_empty():
            return None
        parties = seat_counts["party"].unique().sort().to_list()
        if not parties:
            return None
        chain_count = 4
        fig, axes = plt.subplots(
            len(parties),
            1,
            figsize=(9.2, max(3.6, 2.6 * len(parties))),
            dpi=150,
            sharex=True,
        )
        axes_array = np.atleast_1d(axes)
        for axis, party in zip(axes_array, parties, strict=False):
            party_frame = (
                seat_counts.filter(pl.col("party") == party)
                .with_columns((pl.col("draw_id") % chain_count).alias("chain"))
                .sort("draw_id")
            )
            color = PARTY_COLORS.get(str(party), "#999999")
            for chain in range(chain_count):
                values = party_frame.filter(pl.col("chain") == chain)["electoral_votes"].to_numpy()
                if values.size == 0:
                    continue
                axis.plot(
                    np.arange(values.size),
                    values,
                    linewidth=1.1,
                    alpha=0.78,
                    color=color,
                    label=f"chain {chain + 1}",
                )
            mean_value = float(party_frame["electoral_votes"].mean())
            axis.axhline(mean_value, color=TEXT_COLOR, linewidth=1.0, linestyle="--")
            axis.text(
                0.01,
                0.88,
                f"{party} mean={mean_value:.1f} EV",
                transform=axis.transAxes,
                color=TEXT_COLOR,
                fontsize=9,
                va="top",
            )
            axis.set_ylabel("EV")
            self._style_axis(axis)
        axes_array[0].set_title(
            "Posterior Simulation Chain Traces",
            loc="left",
            fontweight="bold",
        )
        axes_array[-1].set_xlabel("Draw index within split chain")
        axes_array[0].text(
            0.01,
            1.04,
            "MCMC-style split chains from posterior simulations, not a separate MCMC sampler.",
            transform=axes_array[0].transAxes,
            color=MUTED_COLOR,
            fontsize=9,
            va="bottom",
        )
        axes_array[0].legend(loc="upper right", frameon=False, ncol=2, fontsize=8)
        return self._save(fig, plot_dir / "electoral_college_chain_traces.png")

    def _methodology_benchmark(
        self, plot_dir: Path, methodology_benchmark: dict[str, Any]
    ) -> Path | None:
        rows = methodology_benchmark.get("rows", [])
        if not rows:
            return None
        labels = [str(row["dimension"]) for row in rows]
        values = [float(row["score"]) for row in rows]
        fig, ax = plt.subplots(figsize=(9.6, max(4.5, len(labels) * 0.52)), dpi=150)
        colors = [PARTY_COLORS["DEM"] if value >= 0.95 else GOLD_COLOR for value in values]
        ax.barh(labels, values, color=colors)
        for idx, value in enumerate(values):
            ax.text(min(0.98, value + 0.025), idx, f"{value:.2f}", va="center", fontsize=9)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Methodology parity score")
        ax.set_title(
            "Silver/FiveThirtyEight Methodology Benchmark",
            loc="left",
            fontweight="bold",
        )
        self._style_axis(ax)
        return self._save(fig, plot_dir / "silver_methodology_benchmark.png")

    def _polling_trajectory(self, plot_dir: Path, frame: pl.DataFrame) -> Path | None:
        required = {"race_id", "option_id", "polls_probability"}
        if frame.is_empty() or not required.issubset(set(frame.columns)):
            return None
        time_column = (
            "as_of_offset_days"
            if "as_of_offset_days" in frame.columns
            else "as_of"
            if "as_of" in frame.columns
            else None
        )
        if time_column is None:
            return None
        plot_frame = frame.filter(pl.col("polls_probability").is_not_null())
        if "actual_winner" in plot_frame.columns:
            plot_frame = plot_frame.filter(pl.col("actual_winner"))
        if plot_frame.is_empty():
            return None
        sort_columns = ["race_id", time_column]
        plot_frame = plot_frame.sort(sort_columns)
        series_keys = (
            plot_frame.select(["race_id", "option_id"])
            .unique(maintain_order=True)
            .head(8)
            .to_dicts()
        )
        fig, ax = plt.subplots(figsize=(8.8, 5.2), dpi=150)
        for index, key in enumerate(series_keys):
            series = plot_frame.filter(
                (pl.col("race_id") == key["race_id"]) & (pl.col("option_id") == key["option_id"])
            ).sort(time_column)
            if series.height < 1:
                continue
            x_values = series[time_column].to_list()
            y_values = series["polls_probability"].to_list()
            label = f"{key['race_id']} / {str(key['option_id']).split('-')[-1]}"
            color = list(PARTY_COLORS.values())[index % len(PARTY_COLORS)]
            ax.plot(x_values, y_values, marker="o", linewidth=2.0, color=color, label=label)
        ax.axhline(0.5, color=MUTED_COLOR, linestyle="--", linewidth=1)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Polling component win probability")
        if time_column == "as_of_offset_days":
            ax.set_xlabel("Days before election")
            ax.invert_xaxis()
        else:
            ax.set_xlabel("As-of date")
        ax.set_title("Polling Probability Trajectory", loc="left", fontweight="bold")
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.legend(loc="best", frameon=False, fontsize=8)
        self._style_axis(ax)
        return self._save(fig, plot_dir / "polling_probability_trajectory.png")

    def _polling_state_trajectory(self, plot_dir: Path, frame: pl.DataFrame) -> Path | None:
        required = {
            "race_id",
            "option_id",
            "trajectory_date",
            "latent_vote_share",
            "latent_sigma",
        }
        if frame.is_empty() or not required.issubset(set(frame.columns)):
            return None
        final_rows = (
            frame.sort("trajectory_date")
            .group_by(["race_id", "option_id"], maintain_order=True)
            .tail(1)
            .with_columns((pl.col("latent_vote_share") - 0.5).abs().alias("_distance"))
            .sort("_distance")
            .head(8)
        )
        if final_rows.is_empty():
            return None
        selected = frame.join(
            final_rows.select(["race_id", "option_id"]),
            on=["race_id", "option_id"],
            how="inner",
        ).sort(["race_id", "option_id", "trajectory_date"])
        fig, ax = plt.subplots(figsize=(9.2, 5.4), dpi=150)
        colors = list(PARTY_COLORS.values())
        for index, key in enumerate(
            final_rows.select(["race_id", "option_id"]).iter_rows(named=True)
        ):
            series = selected.filter(
                (pl.col("race_id") == key["race_id"]) & (pl.col("option_id") == key["option_id"])
            )
            if series.is_empty():
                continue
            x_values = series["trajectory_date"].to_list()
            mean = np.array(series["latent_vote_share"].to_list(), dtype=float)
            sigma = np.array(series["latent_sigma"].to_list(), dtype=float)
            lower = np.clip(mean - 1.645 * sigma, 0.0, 1.0)
            upper = np.clip(mean + 1.645 * sigma, 0.0, 1.0)
            color = colors[index % len(colors)]
            label = f"{key['race_id']} / {str(key['option_id']).split('-')[-1]}"
            ax.plot(x_values, mean, linewidth=2.0, color=color, label=label)
            ax.fill_between(x_values, lower, upper, color=color, alpha=0.12, linewidth=0)
            if {"poll_count", "mean_observed_share"}.issubset(set(series.columns)):
                poll_rows = series.filter(
                    (pl.col("poll_count") > 0) & pl.col("mean_observed_share").is_not_null()
                )
                if not poll_rows.is_empty():
                    ax.scatter(
                        poll_rows["trajectory_date"].to_list(),
                        poll_rows["mean_observed_share"].to_list(),
                        color=color,
                        edgecolors=AXIS_COLOR,
                        linewidths=0.7,
                        s=32,
                        zorder=4,
                    )
        ax.axhline(0.5, color=MUTED_COLOR, linestyle="--", linewidth=1)
        ax.set_ylim(0.35, 0.65)
        ax.set_xlabel("Date")
        ax.set_ylabel("Latent vote share")
        ax.set_title("Kalman Polling Trajectories", loc="left", fontweight="bold")
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.legend(loc="best", frameon=False, fontsize=7)
        self._style_axis(ax)
        return self._save(fig, plot_dir / "polling_kalman_trajectories.png")

    def _kalman_posterior_uncertainty(self, plot_dir: Path, frame: pl.DataFrame) -> Path | None:
        required = {"race_id", "option_id", "trajectory_date", "latent_sigma"}
        if frame.is_empty() or not required.issubset(set(frame.columns)):
            return None
        final_rows = (
            frame.sort("trajectory_date")
            .group_by(["race_id", "option_id"], maintain_order=True)
            .tail(1)
            .sort("latent_sigma", descending=True)
            .head(8)
        )
        if final_rows.is_empty():
            return None
        selected = frame.join(
            final_rows.select(["race_id", "option_id"]),
            on=["race_id", "option_id"],
            how="inner",
        ).sort(["race_id", "option_id", "trajectory_date"])
        fig, ax = plt.subplots(figsize=(9.2, 5.2), dpi=150)
        colors = list(PARTY_COLORS.values())
        for index, key in enumerate(
            final_rows.select(["race_id", "option_id"]).iter_rows(named=True)
        ):
            series = selected.filter(
                (pl.col("race_id") == key["race_id"]) & (pl.col("option_id") == key["option_id"])
            )
            if series.is_empty():
                continue
            color = colors[index % len(colors)]
            label = f"{key['race_id']} / {str(key['option_id']).split('-')[-1]}"
            ax.plot(
                series["trajectory_date"].to_list(),
                series["latent_sigma"].to_list(),
                linewidth=1.9,
                color=color,
                label=label,
            )
        ax.set_xlabel("Date")
        ax.set_ylabel("Posterior SD")
        ax.set_title("Kalman Posterior Uncertainty", loc="left", fontweight="bold")
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.legend(loc="best", frameon=False, fontsize=7)
        self._style_axis(ax)
        return self._save(fig, plot_dir / "kalman_posterior_uncertainty.png")

    def _posterior_share_intervals(
        self, plot_dir: Path, posterior_draws: pl.DataFrame
    ) -> Path | None:
        required = {"race_id", "option_id", "latent_share"}
        if posterior_draws.is_empty() or not required.issubset(posterior_draws.columns):
            return None
        summary = (
            posterior_draws.group_by(["race_id", "option_id"])
            .agg(
                pl.col("latent_share").mean().alias("mean"),
                pl.col("latent_share").quantile(0.10).alias("p10"),
                pl.col("latent_share").quantile(0.90).alias("p90"),
            )
            .sort("mean", descending=True)
            .head(12)
        )
        if summary.is_empty():
            return None
        labels = [f"{row['race_id']}\n{row['option_id']}" for row in summary.iter_rows(named=True)]
        mean = summary["mean"].to_numpy()
        lower = mean - summary["p10"].to_numpy()
        upper = summary["p90"].to_numpy() - mean
        fig, ax = plt.subplots(figsize=(8.5, 5.8), dpi=150)
        y = np.arange(summary.height)
        ax.errorbar(
            mean,
            y,
            xerr=np.vstack([lower, upper]),
            fmt="o",
            color=PARTY_COLORS["DEM"],
            ecolor=GOLD_COLOR,
            elinewidth=2,
            capsize=3,
        )
        ax.axvline(0.5, color=MUTED_COLOR, linestyle="--", linewidth=1)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.xaxis.set_major_formatter(PercentFormatter(1.0))
        ax.set_xlabel("Latent vote share")
        ax.set_title("Bayesian Polling Posterior Intervals", loc="left", fontweight="bold")
        self._style_axis(ax)
        return self._save(fig, plot_dir / "posterior_latent_share_intervals.png")

    def _posterior_diagnostic_bars(
        self, plot_dir: Path, diagnostics: dict[str, Any]
    ) -> Path | None:
        if not diagnostics or str(diagnostics.get("engine", "kalman")) == "kalman":
            return None
        metrics = {
            "draws": float(diagnostics.get("draw_count") or 0),
            "race-options": float(diagnostics.get("race_option_count") or 0),
            "polls": float(diagnostics.get("poll_count") or 0),
            "divergences": float(diagnostics.get("divergences") or 0),
        }
        if not any(value > 0 for value in metrics.values()):
            return None
        fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=150)
        labels = list(metrics)
        values = np.array(list(metrics.values()))
        colors = [PARTY_COLORS["DEM"], PARTY_COLORS["REP"], GOLD_COLOR, MUTED_COLOR]
        ax.bar(labels, values, color=colors[: len(labels)])
        for index, value in enumerate(values):
            ax.text(index, value, f"{value:.0f}", ha="center", va="bottom", fontsize=9)
        ax.set_ylabel("Count")
        ax.set_title("Bayesian Posterior Diagnostics", loc="left", fontweight="bold")
        subtitle = (
            f"engine={diagnostics.get('engine')} | "
            f"parameterization={diagnostics.get('parameterization', 'n/a')}"
        )
        ax.text(0.01, 0.94, subtitle, transform=ax.transAxes, color=MUTED_COLOR, fontsize=9)
        self._style_axis(ax)
        return self._save(fig, plot_dir / "posterior_diagnostics.png")

    def _fundamentals_prior_intervals(
        self, plot_dir: Path, fundamentals_prior: pl.DataFrame
    ) -> Path | None:
        required = {"race_id", "option_id", "mean_share", "mean_logit", "sd_logit"}
        if fundamentals_prior.is_empty() or not required.issubset(fundamentals_prior.columns):
            return None
        frame = (
            fundamentals_prior.with_columns(
                (1.0 / (1.0 + (-pl.col("mean_logit") + pl.col("sd_logit")).exp())).alias("p10"),
                (1.0 / (1.0 + (-pl.col("mean_logit") - pl.col("sd_logit")).exp())).alias("p90"),
            )
            .sort("mean_share", descending=True)
            .head(12)
        )
        if frame.is_empty():
            return None
        labels = [f"{row['race_id']}\n{row['option_id']}" for row in frame.iter_rows(named=True)]
        mean = frame["mean_share"].to_numpy()
        lower = mean - frame["p10"].to_numpy()
        upper = frame["p90"].to_numpy() - mean
        fig, ax = plt.subplots(figsize=(8.5, 5.8), dpi=150)
        y = np.arange(frame.height)
        ax.errorbar(
            mean,
            y,
            xerr=np.vstack([lower, upper]),
            fmt="o",
            color=GOLD_COLOR,
            ecolor=PARTY_COLORS["DEM"],
            elinewidth=2,
            capsize=3,
        )
        ax.axvline(0.5, color=MUTED_COLOR, linestyle="--", linewidth=1)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.xaxis.set_major_formatter(PercentFormatter(1.0))
        ax.set_xlabel("Election-day prior vote share")
        ax.set_title("Fundamentals Prior Intervals", loc="left", fontweight="bold")
        self._style_axis(ax)
        return self._save(fig, plot_dir / "fundamentals_prior_intervals.png")

    def _simulation_probability_convergence(
        self, plot_dir: Path, forecast_draws: pl.DataFrame
    ) -> Path | None:
        required = {"draw_id", "race_id", "option_id", "winner"}
        if forecast_draws.is_empty() or not required.issubset(set(forecast_draws.columns)):
            return None
        probabilities = (
            forecast_draws.group_by(["race_id", "option_id"])
            .agg(pl.col("winner").mean().alias("final_probability"))
            .filter(pl.col("final_probability").is_not_null())
            .sort("final_probability", descending=True)
            .head(8)
        )
        if probabilities.is_empty():
            return None
        selected = forecast_draws.join(
            probabilities.select(["race_id", "option_id"]),
            on=["race_id", "option_id"],
            how="inner",
        )
        fig, ax = plt.subplots(figsize=(8.8, 5.2), dpi=150)
        for index, row in enumerate(probabilities.iter_rows(named=True)):
            series = selected.filter(
                (pl.col("race_id") == row["race_id"]) & (pl.col("option_id") == row["option_id"])
            ).sort("draw_id")
            wins = series["winner"].cast(pl.Int8).to_numpy()
            if wins.size == 0:
                continue
            cumulative = np.cumsum(wins) / np.arange(1, wins.size + 1)
            draw_ids = np.arange(1, wins.size + 1)
            label = f"{row['race_id']} / {str(row['option_id']).split('-')[-1]}"
            color = list(PARTY_COLORS.values())[index % len(PARTY_COLORS)]
            ax.plot(draw_ids, cumulative, linewidth=1.8, color=color, label=label)
            ax.axhline(float(row["final_probability"]), color=color, alpha=0.22, linewidth=1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Simulation draws consumed")
        ax.set_ylabel("Cumulative winner probability")
        ax.set_title("Simulation Probability Convergence", loc="left", fontweight="bold")
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.legend(loc="best", frameon=False, fontsize=8)
        self._style_axis(ax)
        return self._save(fig, plot_dir / "simulation_probability_convergence.png")

    @staticmethod
    def _option_label(row: dict[str, Any]) -> str:
        party = str(row.get("party") or "")
        name = str(row.get("name") or row.get("option_id"))
        race_id = str(row.get("race_id") or "")
        return f"{name} ({party})\n{race_id}" if party else f"{name}\n{race_id}"

    @staticmethod
    def _modeled_presidential_seats(race_catalog: pl.DataFrame) -> int:
        if race_catalog.is_empty() or "seats" not in race_catalog.columns:
            return 0
        president = race_catalog.filter(
            (pl.col("office_type") == "president") & (pl.col("control_body") == "president")
        )
        return 0 if president.is_empty() else int(president["seats"].sum())

    @staticmethod
    def _presidential_seat_counts(
        race_catalog: pl.DataFrame, forecast_draws: pl.DataFrame
    ) -> pl.DataFrame:
        if race_catalog.is_empty() or forecast_draws.is_empty():
            return pl.DataFrame()
        president = race_catalog.filter(
            (pl.col("office_type") == "president") & (pl.col("control_body") == "president")
        )
        if president.is_empty():
            return pl.DataFrame()
        winner_counts = (
            forecast_draws.join(
                president.select(["race_id", "seats"]),
                on="race_id",
                how="inner",
            )
            .filter(pl.col("winner"))
            .group_by(["draw_id", "party"])
            .agg(pl.col("seats").sum().alias("electoral_votes"))
        )
        draws = forecast_draws.select("draw_id").unique()
        parties = forecast_draws.select("party").unique()
        return (
            draws.join(parties, how="cross")
            .join(winner_counts, on=["draw_id", "party"], how="left")
            .with_columns(pl.col("electoral_votes").fill_null(0))
            .sort(["party", "draw_id"])
        )

    @staticmethod
    def _save(fig: plt.Figure, path: Path) -> Path:
        fig.tight_layout()
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        return path

    @staticmethod
    def _style_axis(ax: plt.Axes) -> None:
        ax.set_facecolor(AXIS_COLOR)
        ax.figure.set_facecolor(PAPER_COLOR)
        ax.grid(axis="x", color=GRID_COLOR, linewidth=0.8, alpha=0.9)
        ax.set_axisbelow(True)
        ax.tick_params(colors=TEXT_COLOR)
        ax.title.set_color(TEXT_COLOR)
        ax.xaxis.label.set_color(TEXT_COLOR)
        ax.yaxis.label.set_color(TEXT_COLOR)
        ax.title.set_fontsize(14)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color(GRID_COLOR)

    # -----------------------------------------------------------------------
    # Distribution-aware additions (Phase 2 of the dashboard redesign).
    # -----------------------------------------------------------------------

    def _seat_count_histogram(
        self,
        plot_dir: Path,
        race_catalog: pl.DataFrame,
        forecast_draws: pl.DataFrame,
        control_forecasts: pl.DataFrame,
    ) -> Path | None:
        """Distribution of DEM seats (modeled wins + holdovers) with the majority line.

        One-seat bins, y-axis in percent of simulations, x-range clipped to the
        p0.5-p99.5 window around the distribution. Bars at or above the DEM control
        threshold are blue, the rest red. Minor parties (IND/OTH) are not drawn as
        their own histogram: their 0-2 seat spikes stretched the x-axis to zero.
        """
        if race_catalog.is_empty() or forecast_draws.is_empty() or control_forecasts.is_empty():
            return None
        catalog = race_catalog.select(["race_id", "control_body", "seats"])
        joined = forecast_draws.join(catalog, on="race_id", how="inner").filter(
            pl.col("control_body").is_not_null()
        )
        if joined.is_empty():
            return None
        body = str(control_forecasts.row(0, named=True)["control_body"])
        control = {
            str(row["party"]).upper(): row
            for row in control_forecasts.iter_rows(named=True)
            if str(row["control_body"]) == body
        }
        focus = "DEM" if "DEM" in control else sorted(control)[0]
        focus_row = control[focus]
        threshold = int(focus_row["control_threshold"])
        holdover = int(focus_row.get("holdover_seats") or 0)
        counts = self._party_seat_counts(joined, body, focus) + holdover
        if counts.size == 0:
            return None
        values, freq = np.unique(counts, return_counts=True)
        share = freq / counts.size
        lo, hi = self._seat_window(counts)
        fig, ax = plt.subplots(figsize=(9.6, 4.6), dpi=150)
        other = "REP" if focus == "DEM" else "DEM"
        colors = [
            PARTY_COLORS.get(focus, ACCENT_FALLBACK)
            if value >= threshold
            else PARTY_COLORS.get(other, MUTED_COLOR)
            for value in values
        ]
        ax.bar(values, share, width=0.9, color=colors, alpha=0.85, linewidth=0)
        ax.axvline(threshold - 0.5, color=TEXT_COLOR, linestyle="--", linewidth=1.3)
        ymax = float(share.max()) * 1.45
        ax.set_ylim(0, ymax)
        mean = float(focus_row.get("seat_count_mean") or np.mean(counts))
        p10 = float(focus_row.get("seat_count_p10") or np.quantile(counts, 0.10))
        p50 = float(focus_row.get("seat_count_p50") or np.quantile(counts, 0.50))
        p90 = float(focus_row.get("seat_count_p90") or np.quantile(counts, 0.90))
        band_y = ymax * 0.74
        ax.plot([p10, p90], [band_y, band_y], color=MUTED_COLOR, linewidth=2.2)
        ax.plot([p10, p90], [band_y, band_y], "|", color=MUTED_COLOR, markersize=12)
        ax.plot([mean], [band_y], "o", color=TEXT_COLOR, markersize=6)
        ax.text(
            mean,
            band_y + ymax * 0.035,
            f"mean {mean:.1f} | median {p50:.0f} | 80% range {p10:.0f}-{p90:.0f}",
            ha="center",
            va="bottom",
            fontsize=9,
            color=TEXT_COLOR,
        )
        control_probability = float(focus_row.get("control_probability") or 0.0)
        other_probability = float((control.get(other) or {}).get("control_probability") or 0.0)
        summary = f"P({focus} control) = {control_probability:.1%}"
        if other in control:
            summary += f"   P({other} control) = {other_probability:.1%}"
        with_ind = focus_row.get("control_probability_with_independents")
        caucus = str(focus_row.get("independent_caucus_races") or "[]")
        if with_ind is not None and caucus not in ("", "[]"):
            summary += f"\nIf independents caucus with {focus}: {float(with_ind):.1%}"
        summary += f"\nDashed line = majority ({threshold}+ {focus} seats)"
        ax.text(
            0.01,
            0.97,
            summary,
            transform=ax.transAxes,
            fontsize=10,
            va="top",
            ha="left",
            color=TEXT_COLOR,
            fontweight="bold",
        )
        ax.set_xlim(lo - 0.5, hi + 0.5)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=12))
        ax.set_xlabel(
            f"{focus} {body} seats (incl. {holdover} not up)"
            if holdover
            else f"{focus} {body} seats"
        )
        ax.set_ylabel("Share of simulations")
        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=1 if ymax < 0.05 else 0))
        ax.set_title(
            f"{body.title()} Seat Distribution ({counts.size:,} simulations)",
            loc="left",
            fontweight="bold",
        )
        self._style_axis(ax)
        ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8, alpha=0.9)
        return self._save(fig, plot_dir / "seat_count_histogram.png")

    @staticmethod
    def _party_seat_counts(joined: pl.DataFrame, body: str, party: str) -> np.ndarray:
        """Seats won by ``party`` in every draw, including draws where it won none."""
        body_draws = joined.filter(pl.col("control_body") == body)
        draws = body_draws.select("draw_id").unique()
        wins = (
            body_draws.filter(pl.col("winner") & (pl.col("party").str.to_uppercase() == party))
            .group_by("draw_id")
            .agg(pl.col("seats").sum().alias("seat_count"))
        )
        return (
            draws.join(wins, on="draw_id", how="left")
            .with_columns(pl.col("seat_count").fill_null(0))
            .sort("draw_id")["seat_count"]
            .to_numpy()
            .astype(int)
        )

    @staticmethod
    def _seat_window(counts: np.ndarray, pad: int = 2, min_width: int = 12) -> tuple[int, int]:
        """Integer x-window covering p0.5-p99.5 of the draws, padded, at least min_width."""
        lo = int(np.floor(np.quantile(counts, 0.005))) - pad
        hi = int(np.ceil(np.quantile(counts, 0.995))) + pad
        if hi - lo < min_width:
            mid = (hi + lo) / 2
            lo, hi = int(np.floor(mid - min_width / 2)), int(np.ceil(mid + min_width / 2))
        return max(lo, 0), hi

    @staticmethod
    def _surname(name: str) -> str:
        """Last name for compact labels, skipping suffixes (``Jr.``, ``III``)."""
        suffixes = {"jr", "jr.", "sr", "sr.", "ii", "iii", "iv"}
        parts = [part for part in name.replace(",", " ").split() if part.lower() not in suffixes]
        return parts[-1] if parts else name

    @staticmethod
    def _short_race_label(race_id: str) -> str:
        """US-SEN-IA-2026 -> IA, US-HOUSE-AZ-01-2026 -> AZ-01."""
        parts = race_id.split("-")
        if len(parts) >= 3 and parts[0] == "US":
            parts = parts[2:]
        if parts and parts[-1].isdigit() and len(parts[-1]) == 4:
            parts = parts[:-1]
        return "-".join(parts) or race_id

    def _race_vote_share_kde_grid(
        self,
        plot_dir: Path,
        race_forecasts: pl.DataFrame,
        forecast_draws: pl.DataFrame,
        max_panels: int = 12,
    ) -> Path | None:
        """Small multiples of vote-share KDEs for the most competitive races.

        All panels share one x-window (vote share in %, p0.5-p99.5 of the plotted
        draws) so widths are comparable across races; each panel shows win odds."""
        if race_forecasts.is_empty() or forecast_draws.is_empty():
            return None
        candidates = race_forecasts.filter(pl.col("winner_probability").is_not_null())
        if candidates.is_empty():
            return None
        ranked = (
            candidates.with_columns((pl.col("winner_probability") - 0.5).abs().alias("_dist"))
            .group_by("race_id", maintain_order=True)
            .agg(pl.col("_dist").min().alias("_dist"))
            .sort("_dist")
            .head(max_panels)
        )
        race_ids = ranked["race_id"].to_list()
        if not race_ids:
            return None
        shares = (
            forecast_draws.filter(pl.col("race_id").is_in(race_ids))["vote_share"]
            .drop_nulls()
            .to_numpy()
            .astype(float)
        )
        if shares.size:
            lo = float(np.quantile(shares, 0.005)) - 0.02
            hi = float(np.quantile(shares, 0.995)) + 0.02
            half = max(0.5 - lo, hi - 0.5, 0.1)
            window = (max(0.0, 0.5 - half), min(1.0, 0.5 + half))
        else:
            window = (0.25, 0.75)
        n = len(race_ids)
        cols = min(3, n)
        rows = int(np.ceil(n / cols))
        fig, axes = plt.subplots(
            rows, cols, figsize=(cols * 4.6, rows * 3.0), dpi=150, squeeze=False
        )
        for index, race_id in enumerate(race_ids):
            ax = axes[index // cols][index % cols]
            self._render_race_kde_panel(ax, race_id, race_forecasts, forecast_draws, window)
        for index in range(n, rows * cols):
            axes[index // cols][index % cols].axis("off")
        fig.suptitle(
            "Vote-Share Distribution — Most Competitive Races (dashed line = 50%)",
            x=0.02,
            ha="left",
            fontsize=14,
            fontweight="bold",
            color=TEXT_COLOR,
        )
        return self._save(fig, plot_dir / "race_vote_share_kde.png")

    def _render_race_kde_panel(
        self,
        ax: plt.Axes,
        race_id: str,
        race_forecasts: pl.DataFrame,
        forecast_draws: pl.DataFrame,
        window: tuple[float, float] = (0.25, 0.75),
    ) -> None:
        race_draws = forecast_draws.filter(pl.col("race_id") == race_id)
        race_meta = race_forecasts.filter(pl.col("race_id") == race_id)
        if race_draws.is_empty() or race_meta.is_empty():
            ax.text(0.5, 0.5, "no draws", ha="center", va="center", color=MUTED_COLOR)
            ax.set_xticks([])
            ax.set_yticks([])
            return
        xs = np.linspace(window[0], window[1], 240)
        odds: list[str] = []
        for option in race_meta.sort("winner_probability", descending=True).iter_rows(named=True):
            samples = (
                race_draws.filter(pl.col("option_id") == option["option_id"])["vote_share"]
                .drop_nulls()
                .to_numpy()
                .astype(float)
            )
            if samples.size < 5:
                continue
            party = str(option.get("party") or "").upper()
            color = PARTY_COLORS.get(party, ACCENT_FALLBACK)
            name = self._surname(str(option.get("name") or party))
            label = f"{name} ({party[:1] or '?'})"
            density = self._gaussian_kde(samples, xs)
            ax.fill_between(xs, density, alpha=0.18, color=color)
            ax.plot(xs, density, color=color, linewidth=1.6, label=label)
            ax.axvline(
                float(option.get("vote_share_mean") or np.mean(samples)),
                color=color,
                linestyle="-",
                linewidth=0.9,
                alpha=0.7,
            )
            probability = option.get("winner_probability")
            if probability is not None:
                odds.append(f"{label} {float(probability):.0%}")
        ax.axvline(0.5, color=MUTED_COLOR, linestyle="--", linewidth=0.9)
        ax.set_xlim(*window)
        ax.set_ylim(bottom=0)
        ax.set_yticks([])
        ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        ax.tick_params(axis="x", labelsize=8, colors=TEXT_COLOR)
        ax.set_title(self._short_race_label(race_id), fontsize=11, color=TEXT_COLOR, loc="left")
        if odds:
            ax.text(
                0.99,
                0.97,
                "Win odds\n" + "\n".join(odds[:3]),
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=7.5,
                color=TEXT_COLOR,
            )
        ax.set_xlabel("Projected vote share", fontsize=9)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color(GRID_COLOR)

    def _tipping_point_bars(
        self,
        plot_dir: Path,
        control_forecasts: pl.DataFrame,
        race_forecasts: pl.DataFrame | None = None,
        top_n: int = 10,
    ) -> Path | None:
        """Horizontal bars of true tipping-point probabilities (audit P1-2).

        Top ``top_n`` races, sorted, x-axis in percent with headroom for labels; each
        label carries the race's DEM win probability when race forecasts are given.
        Falls back to the legacy pivotal-flip rate for control frames that predate it."""
        if control_forecasts.is_empty():
            return None
        if "tipping_point_probability" in control_forecasts.columns:
            column, key, xlabel = (
                "tipping_point_probability",
                "tipping_point_probability",
                "Chance the race is the tipping point for control (% of simulations)",
            )
        elif "pivotal_rates" in control_forecasts.columns:
            column, key, xlabel = (
                "pivotal_rates",
                "pivotal_rate",
                "Legacy pivotal-flip rate (P this race decides control)",
            )
        else:
            return None
        rows: list[dict[str, object]] = []
        none_probability: float | None = None
        body = ""
        for record in control_forecasts.iter_rows(named=True):
            if column == "tipping_point_probability" and str(record.get("party")).upper() != "DEM":
                continue  # one chamber-wide ordering; the DEM view is the headline
            raw = record.get(column)
            if not raw:
                continue
            try:
                payload = json.loads(str(raw))
            except json.JSONDecodeError:
                continue
            body = str(record.get("control_body") or body)
            if record.get("tipping_point_none_probability") is not None:
                none_probability = float(record["tipping_point_none_probability"])
            for item in payload:
                rows.append(
                    {
                        "race_id": str(item.get("race_id")),
                        "party": str(record.get("party") or "").upper(),
                        "pivotal_rate": float(item.get(key) or 0.0),
                    }
                )
        if not rows:
            return None
        frame = pl.DataFrame(rows).sort("pivotal_rate", descending=True).head(top_n).reverse()
        dem_odds: dict[str, float] = {}
        if race_forecasts is not None and not race_forecasts.is_empty():
            needed = {"race_id", "party", "winner_probability"}
            if needed.issubset(race_forecasts.columns):
                for row in race_forecasts.filter(
                    (pl.col("party").str.to_uppercase() == "DEM")
                    & pl.col("winner_probability").is_not_null()
                ).iter_rows(named=True):
                    dem_odds[str(row["race_id"])] = float(row["winner_probability"])
        labels = []
        for row in frame.iter_rows(named=True):
            label = self._short_race_label(str(row["race_id"]))
            if column != "tipping_point_probability":
                label += f" ({row['party']})"
            if str(row["race_id"]) in dem_odds:
                label += f"  · D win {dem_odds[str(row['race_id'])]:.0%}"
            labels.append(label)
        values = frame["pivotal_rate"].to_list()
        colors = [
            PARTY_COLORS.get(str(row["party"]).upper(), ACCENT_FALLBACK)
            for row in frame.iter_rows(named=True)
        ]
        fig, ax = plt.subplots(figsize=(9.6, max(3.6, len(labels) * 0.42 + 1.0)), dpi=150)
        ax.barh(labels, values, color=colors)
        vmax = max(values) if values else 0.0
        ax.set_xlim(0, max(vmax * 1.18, 0.01))
        decimals = 1 if vmax < 0.05 else 0
        ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=decimals))
        for index, value in enumerate(values):
            ax.text(value + vmax * 0.012, index, f"{value:.1%}", va="center", fontsize=9)
        ax.set_xlabel(xlabel)
        title = (
            f"{body.title()} Tipping-Point Races (top {len(labels)})"
            if body
            else "Tipping-Point Races"
        )
        ax.set_title(title, loc="left", fontweight="bold", pad=20)
        subtitle = (
            "Race whose win first carries DEM to a majority when races are ordered by DEM margin"
        )
        if none_probability is not None and none_probability > 0.005:
            subtitle += f"; majority out of reach in {none_probability:.1%} of simulations"
        ax.text(
            0.0,
            1.01,
            subtitle,
            transform=ax.transAxes,
            fontsize=8.5,
            color=MUTED_COLOR,
            va="bottom",
        )
        self._style_axis(ax)
        return self._save(fig, plot_dir / "tipping_point_bars.png")

    def _driver_waterfall(
        self,
        plot_dir: Path,
        race_forecasts: pl.DataFrame,
        max_panels: int = 6,
    ) -> Path | None:
        """Per-race component contribution waterfalls for the closest races."""
        if race_forecasts.is_empty() or "component_contributions" not in race_forecasts.columns:
            return None
        candidates = race_forecasts.filter(
            pl.col("winner_probability").is_not_null()
            & pl.col("component_contributions").is_not_null()
        )
        if candidates.is_empty():
            return None
        candidates = candidates.with_columns(
            (pl.col("winner_probability") - 0.5).abs().alias("_dist")
        ).sort("_dist")
        chosen: list[dict[str, object]] = []
        seen_races: set[str] = set()
        for row in candidates.iter_rows(named=True):
            race_id = str(row["race_id"])
            if race_id in seen_races:
                continue
            seen_races.add(race_id)
            chosen.append(row)
            if len(chosen) >= max_panels:
                break
        if not chosen:
            return None
        cols = min(3, len(chosen))
        rows_n = int(np.ceil(len(chosen) / cols))
        fig, axes = plt.subplots(
            rows_n, cols, figsize=(cols * 4.5, rows_n * 3.2), dpi=150, squeeze=False
        )
        for index, row in enumerate(chosen):
            ax = axes[index // cols][index % cols]
            self._render_driver_waterfall(ax, row)
        for index in range(len(chosen), rows_n * cols):
            axes[index // cols][index % cols].axis("off")
        fig.suptitle(
            "Component Drivers — Closest Races",
            x=0.02,
            ha="left",
            fontsize=14,
            fontweight="bold",
            color=TEXT_COLOR,
        )
        return self._save(fig, plot_dir / "driver_waterfall.png")

    def _render_driver_waterfall(self, ax: plt.Axes, row: dict[str, object]) -> None:
        try:
            payload = json.loads(str(row.get("component_contributions") or "{}"))
        except json.JSONDecodeError:
            payload = {}
        if not isinstance(payload, dict) or not payload:
            ax.text(0.5, 0.5, "no decomposition", ha="center", va="center", color=MUTED_COLOR)
            ax.set_xticks([])
            ax.set_yticks([])
            return
        components = []
        contributions = []
        for component, info in payload.items():
            if not isinstance(info, dict):
                continue
            try:
                contribution = (
                    float(info.get("weighted_marginal_win_probability", 0.0))
                    - float(info.get("weight", 0.0)) * 0.5
                )
            except (TypeError, ValueError):
                continue
            components.append(component)
            contributions.append(contribution)
        if not components:
            ax.text(0.5, 0.5, "empty contributions", ha="center", va="center", color=MUTED_COLOR)
            ax.set_xticks([])
            ax.set_yticks([])
            return
        order = sorted(zip(components, contributions, strict=True), key=lambda item: item[1])
        components_sorted, contributions_sorted = zip(*order, strict=True)
        colors = [
            PARTY_COLORS["DEM"] if value > 0 else PARTY_COLORS["REP"]
            for value in contributions_sorted
        ]
        ax.barh(list(components_sorted), list(contributions_sorted), color=colors)
        ax.axvline(0, color=MUTED_COLOR, linewidth=0.8)
        ax.set_title(
            f"{row.get('race_id')} → {str(row.get('option_id') or '').split('-')[-1]}",
            fontsize=11,
            color=TEXT_COLOR,
            loc="left",
        )
        ax.set_xlabel("Contribution to D-leaning win probability", fontsize=9)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color(GRID_COLOR)

    def _reliability_diagram_with_band(self, plot_dir: Path, frame: pl.DataFrame) -> Path | None:
        """Calibration curve with Wilson confidence band and forecast histogram."""
        if frame.is_empty() or "ensemble_probability" not in frame.columns:
            return None
        df = frame.select(["ensemble_probability", "actual_winner"]).drop_nulls()
        if df.is_empty():
            return None
        probability = df["ensemble_probability"].cast(pl.Float64).to_numpy()
        actual = df["actual_winner"].cast(pl.Float64).to_numpy()
        bins = np.linspace(0, 1, 11)
        xs: list[float] = []
        ys: list[float] = []
        ns: list[int] = []
        for lower, upper in pairwise(bins):
            mask = (probability >= lower) & (
                probability < upper if upper < 1 else probability <= upper
            )
            if not np.any(mask):
                continue
            xs.append(float(np.mean(probability[mask])))
            ys.append(float(np.mean(actual[mask])))
            ns.append(int(mask.sum()))
        if not xs:
            return None
        z = 1.96
        lower_band = []
        upper_band = []
        for y, n in zip(ys, ns, strict=True):
            if n == 0:
                lower_band.append(y)
                upper_band.append(y)
                continue
            denom = 1 + z * z / n
            centre = (y + z * z / (2 * n)) / denom
            half = (z / denom) * np.sqrt(y * (1 - y) / n + z * z / (4 * n * n))
            lower_band.append(max(0.0, centre - half))
            upper_band.append(min(1.0, centre + half))
        fig, (ax_main, ax_hist) = plt.subplots(
            2, 1, figsize=(7.5, 6.0), dpi=150, gridspec_kw={"height_ratios": [3, 1]}
        )
        ax_main.plot([0, 1], [0, 1], color=MUTED_COLOR, linestyle="--", linewidth=1)
        ax_main.fill_between(xs, lower_band, upper_band, alpha=0.18, color=PARTY_COLORS["DEM"])
        ax_main.scatter(xs, ys, s=72, color=PARTY_COLORS["DEM"], zorder=3)
        for x, y in zip(xs, ys, strict=True):
            ax_main.text(x + 0.02, y, f"{y:.0%}", va="center", fontsize=9, color=TEXT_COLOR)
        ax_main.set_xlim(0, 1)
        ax_main.set_ylim(0, 1)
        ax_main.set_xlabel("Mean forecast probability")
        ax_main.set_ylabel("Observed win rate")
        ax_main.set_title(
            "Reliability Diagram (95% Wilson band)",
            loc="left",
            fontweight="bold",
        )
        ax_main.xaxis.set_major_formatter(PercentFormatter(1.0))
        ax_main.yaxis.set_major_formatter(PercentFormatter(1.0))
        self._style_axis(ax_main)
        ax_hist.hist(probability, bins=bins, color=PARTY_COLORS["DEM"], alpha=0.55)
        ax_hist.set_xlim(0, 1)
        ax_hist.set_ylabel("n", fontsize=9)
        ax_hist.set_xlabel("Forecast probability distribution")
        ax_hist.xaxis.set_major_formatter(PercentFormatter(1.0))
        for spine in ("top", "right", "left"):
            ax_hist.spines[spine].set_visible(False)
        ax_hist.spines["bottom"].set_color(GRID_COLOR)
        return self._save(fig, plot_dir / "reliability_diagram.png")

    @staticmethod
    def _gaussian_kde(samples: np.ndarray, xs: np.ndarray) -> np.ndarray:
        """Plain Gaussian KDE without depending on scipy."""
        if samples.size == 0:
            return np.zeros_like(xs)
        std = float(np.std(samples))
        if std == 0:
            std = 1e-3
        bw = std * (4 / (3 * samples.size)) ** (1 / 5)
        bw = max(bw, 1e-3)
        diffs = (xs[:, None] - samples[None, :]) / bw
        kernel = np.exp(-0.5 * diffs * diffs) / np.sqrt(2 * np.pi)
        return kernel.sum(axis=1) / (samples.size * bw)


ACCENT_FALLBACK = "#547c70"
