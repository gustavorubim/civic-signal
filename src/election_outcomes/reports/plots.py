from __future__ import annotations

import json
from itertools import pairwise
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import polars as pl

matplotlib.use("Agg")
import matplotlib.pyplot as plt


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
    ) -> dict[str, list[dict[str, str]]]:
        plot_dir = artifact_dir / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        manifest: dict[str, list[dict[str, str]]] = {
            "calibration": [],
            "projection": [],
        }
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
        fig, ax = plt.subplots(figsize=(7, 5), dpi=140)
        ax.plot([0, 1], [0, 1], color="#777777", linestyle="--", linewidth=1)
        ax.scatter(xs, ys, color="#1f77b4", s=60)
        ax.set_xlabel("Mean forecast probability")
        ax.set_ylabel("Observed win rate")
        ax.set_title("Calibration Curve")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
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
        labels, values = zip(*rows, strict=True)
        fig, ax = plt.subplots(figsize=(8, 5), dpi=140)
        ax.bar(labels, values, color="#4c78a8")
        ax.set_ylabel("Brier score")
        ax.set_title("Backtest Brier Score by Component")
        ax.tick_params(axis="x", rotation=25)
        return self._save(fig, plot_dir / "brier_by_component.png")

    def _interval_coverage(self, plot_dir: Path, backtest_payload: dict[str, Any]) -> Path | None:
        coverage = (
            backtest_payload.get("metrics", {}).get("ensemble", {}).get("interval_90_coverage")
        )
        if coverage is None:
            return None
        fig, ax = plt.subplots(figsize=(6, 5), dpi=140)
        ax.bar(["Nominal 90%", "Observed"], [0.9, float(coverage)], color=["#999999", "#59a14f"])
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Coverage")
        ax.set_title("Historical Interval Coverage")
        return self._save(fig, plot_dir / "interval_coverage.png")

    def _race_probability_bars(self, plot_dir: Path, race_forecasts: pl.DataFrame) -> Path | None:
        frame = race_forecasts.filter(pl.col("winner_probability").is_not_null())
        if frame.is_empty():
            return None
        labels = [
            f"{row['race_id']} | {row['name']}"
            for row in frame.sort(["race_id", "winner_probability"]).iter_rows(named=True)
        ]
        values = frame.sort(["race_id", "winner_probability"])["winner_probability"].to_list()
        fig, ax = plt.subplots(figsize=(10, max(5, len(labels) * 0.45)), dpi=140)
        ax.barh(labels, values, color="#f28e2b")
        ax.set_xlim(0, 1)
        ax.set_xlabel("Winner probability")
        ax.set_title("Race-Level Winner Probabilities")
        return self._save(fig, plot_dir / "race_probability_bars.png")

    def _vote_share_intervals(self, plot_dir: Path, race_forecasts: pl.DataFrame) -> Path | None:
        frame = race_forecasts.filter(pl.col("vote_share_mean").is_not_null())
        required = {"vote_share_mean", "vote_share_p05", "vote_share_p95"}
        if frame.is_empty() or not required.issubset(set(frame.columns)):
            return None
        sorted_frame = frame.sort(["race_id", "option_id"])
        labels = [f"{row['race_id']} | {row['name']}" for row in sorted_frame.iter_rows(named=True)]
        mean = np.array(sorted_frame["vote_share_mean"].to_list())
        low = np.array(sorted_frame["vote_share_p05"].to_list())
        high = np.array(sorted_frame["vote_share_p95"].to_list())
        fig, ax = plt.subplots(figsize=(10, max(5, len(labels) * 0.45)), dpi=140)
        ax.errorbar(mean, labels, xerr=[mean - low, high - mean], fmt="o", color="#2f4b7c")
        ax.axvline(0.5, color="#777777", linestyle="--", linewidth=1)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Projected vote share with 90% interval")
        ax.set_title("Vote-Share Projection Intervals")
        return self._save(fig, plot_dir / "vote_share_intervals.png")

    def _control_projection(self, plot_dir: Path, control_forecasts: pl.DataFrame) -> Path | None:
        if control_forecasts.is_empty():
            return None
        labels = [
            f"{row['control_body']} | {row['party']}"
            for row in control_forecasts.iter_rows(named=True)
        ]
        values = control_forecasts["seat_count_mean"].to_list()
        fig, ax = plt.subplots(figsize=(8, max(4, len(labels) * 0.5)), dpi=140)
        ax.barh(labels, values, color="#e15759")
        ax.set_xlabel("Mean projected seats/wins in modeled races")
        ax.set_title("Control Projection")
        return self._save(fig, plot_dir / "control_projection.png")

    def _turnout_and_recount(
        self, plot_dir: Path, ecosystem_forecasts: pl.DataFrame
    ) -> Path | None:
        if ecosystem_forecasts.is_empty():
            return None
        labels = ecosystem_forecasts["race_id"].to_list()
        recount = ecosystem_forecasts["recount_probability"].to_list()
        fig, ax = plt.subplots(figsize=(10, max(5, len(labels) * 0.45)), dpi=140)
        ax.barh(labels, recount, color="#76b7b2")
        ax.set_xlim(0, 1)
        ax.set_xlabel("Recount probability")
        ax.set_title("Recount Risk by Race")
        return self._save(fig, plot_dir / "turnout_recount_risk.png")

    def _tier_coverage(self, plot_dir: Path, race_catalog: pl.DataFrame) -> Path | None:
        if race_catalog.is_empty():
            return None
        counts = race_catalog.group_by("tier").agg(pl.len().alias("count")).sort("tier")
        fig, ax = plt.subplots(figsize=(6, 5), dpi=140)
        ax.bar(counts["tier"].to_list(), counts["count"].to_list(), color="#b07aa1")
        ax.set_ylabel("Race count")
        ax.set_title("Forecast Coverage by Tier")
        return self._save(fig, plot_dir / "tier_coverage.png")

    @staticmethod
    def _save(fig: plt.Figure, path: Path) -> Path:
        fig.tight_layout()
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        return path
