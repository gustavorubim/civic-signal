from __future__ import annotations

import html
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import polars as pl

from election_outcomes.storage.io import write_json, write_parquet, write_text

matplotlib.use("Agg")
import matplotlib.pyplot as plt


class ResultComparator:
    """Compare a forecast run against actual results."""

    def compare(
        self,
        forecast_run_dir: Path,
        curated_results: pl.DataFrame,
        comparison_id: str,
        cycle: int | None = None,
        office_type: str | None = None,
        race_id: str | None = None,
    ) -> dict[str, Any]:
        race_catalog = pl.read_parquet(forecast_run_dir / "race_catalog.parquet")
        race_forecasts = pl.read_parquet(forecast_run_dir / "race_forecasts.parquet")
        comparison = self._comparison_frame(
            race_catalog=race_catalog,
            race_forecasts=race_forecasts,
            curated_results=curated_results,
            cycle=cycle,
            office_type=office_type,
            race_id=race_id,
        )
        summary = self._summary(
            comparison=comparison,
            comparison_id=comparison_id,
            cycle=cycle,
            office_type=office_type,
            race_id=race_id,
        )
        output_dir = forecast_run_dir / "comparisons" / comparison_id
        output_dir.mkdir(parents=True, exist_ok=True)
        write_parquet(comparison, output_dir / "result_comparison.parquet")
        insight_artifacts = self._write_insight_tables(comparison, output_dir)
        plot_manifest = self._write_plots(comparison, output_dir)
        summary["insight_artifacts"] = insight_artifacts
        summary["plot_manifest"] = plot_manifest
        write_json(summary, output_dir / "result_comparison_summary.json")
        write_text(
            self._html_report(summary=summary, comparison=comparison),
            output_dir / "result_comparison.html",
        )
        write_text(
            self._narrative(summary=summary, comparison=comparison),
            output_dir / "narrative.md",
        )
        return {**summary, "output_dir": str(output_dir)}

    def _comparison_frame(
        self,
        race_catalog: pl.DataFrame,
        race_forecasts: pl.DataFrame,
        curated_results: pl.DataFrame,
        cycle: int | None,
        office_type: str | None,
        race_id: str | None,
    ) -> pl.DataFrame:
        race_meta = race_catalog.select(
            [
                "race_id",
                "cycle",
                "election_date",
                "geography_type",
                "geography",
                "office_type",
                "race_type",
                "seats",
                "control_body",
                "tier",
                "tier_reason",
            ]
        ).unique()
        forecasts = race_forecasts.drop(["tier", "tier_reason"], strict=False).join(
            race_meta, on="race_id", how="left"
        )
        forecasts = self._apply_filters(forecasts, cycle, office_type, race_id)
        actuals = curated_results.select(
            [
                "race_id",
                "option_id",
                pl.col("vote_share").alias("actual_vote_share"),
                pl.col("turnout").alias("actual_turnout"),
                pl.col("winner").alias("actual_winner"),
            ]
        )
        comparison = forecasts.join(actuals, on=["race_id", "option_id"], how="inner")
        if comparison.is_empty():
            return comparison
        comparison = comparison.with_columns(
            (pl.col("vote_share_mean") - pl.col("actual_vote_share")).alias("vote_share_error"),
            (pl.col("vote_share_mean") - pl.col("actual_vote_share"))
            .abs()
            .alias("absolute_vote_share_error"),
            (pl.col("winner_probability") - pl.col("actual_winner").cast(pl.Float64))
            .pow(2)
            .alias("brier_contribution"),
        )
        race_outcomes = self._race_outcome_frame(comparison)
        outcome_columns = [
            column
            for column in race_outcomes.columns
            if column
            in {
                "race_id",
                "predicted_winner_option_id",
                "predicted_winner_name",
                "predicted_winner_party",
                "predicted_winner_probability",
                "actual_winner_option_id",
                "actual_winner_name",
                "actual_winner_party",
                "actual_winner_probability",
                "race_winner_correct",
            }
        ]
        comparison = comparison.join(
            race_outcomes.select(outcome_columns), on="race_id", how="left"
        )
        return comparison.with_columns(
            (pl.col("option_id") == pl.col("predicted_winner_option_id")).alias("predicted_winner")
        )

    @staticmethod
    def _apply_filters(
        frame: pl.DataFrame, cycle: int | None, office_type: str | None, race_id: str | None
    ) -> pl.DataFrame:
        filtered = frame
        if cycle is not None:
            filtered = filtered.filter(pl.col("cycle") == cycle)
        if office_type is not None:
            filtered = filtered.filter(pl.col("office_type") == office_type)
        if race_id is not None:
            filtered = filtered.filter(pl.col("race_id") == race_id)
        return filtered

    def _summary(
        self,
        comparison: pl.DataFrame,
        comparison_id: str,
        cycle: int | None,
        office_type: str | None,
        race_id: str | None,
    ) -> dict[str, Any]:
        base: dict[str, Any] = {
            "comparison_id": comparison_id,
            "generated_at": datetime.now(UTC).isoformat(),
            "filters": {
                "cycle": cycle,
                "office_type": office_type,
                "race_id": race_id,
            },
            "row_count": comparison.height,
            "race_count": comparison["race_id"].n_unique()
            if "race_id" in comparison.columns
            else 0,
        }
        if comparison.is_empty():
            return {
                **base,
                "winner_accuracy": None,
                "state_accuracy": None,
                "state_accuracy_n": 0,
                "ec_winner_accuracy": None,
                "electoral_college": self._empty_electoral_college_summary(),
                "mean_absolute_vote_share_error": None,
                "brier_score": None,
                "upset_count": 0,
                "actual_winner_probabilities": [],
                "largest_misses": [],
                "race_outcomes": [],
            }
        race_outcomes = self._race_outcome_frame(comparison)
        winner_accuracy = self._mean_bool_or_none(race_outcomes, "race_winner_correct")
        presidential_states = race_outcomes.filter(
            (pl.col("office_type") == "president") & (pl.col("geography_type") == "state")
        )
        state_accuracy = self._mean_bool_or_none(presidential_states, "race_winner_correct")
        electoral_college = self._electoral_college_summary(presidential_states)
        actual_winner_rows = comparison.filter(pl.col("actual_winner"))
        upset_count = actual_winner_rows.filter(pl.col("winner_probability") < 0.5).height
        return {
            **base,
            "winner_accuracy": None if winner_accuracy is None else float(winner_accuracy),
            "state_accuracy": None if state_accuracy is None else float(state_accuracy),
            "state_accuracy_n": presidential_states.height,
            "ec_winner_accuracy": electoral_college["winner_accuracy"],
            "electoral_college": electoral_college,
            "mean_absolute_vote_share_error": self._mean_or_none(
                comparison, "absolute_vote_share_error"
            ),
            "brier_score": self._mean_or_none(comparison, "brier_contribution"),
            "upset_count": upset_count,
            "actual_winner_probabilities": self._actual_winner_probabilities(race_outcomes),
            "largest_misses": self._largest_misses(comparison),
            "race_outcomes": self._json_records(race_outcomes),
        }

    def _write_insight_tables(self, comparison: pl.DataFrame, output_dir: Path) -> dict[str, str]:
        artifacts: dict[str, str] = {}
        race_outcomes = self._race_outcome_frame(comparison)
        if not race_outcomes.is_empty():
            write_parquet(race_outcomes, output_dir / "race_outcomes.parquet")
            artifacts["race_outcomes"] = "race_outcomes.parquet"
        largest_misses = self._largest_miss_frame(comparison, limit=25)
        if not largest_misses.is_empty():
            write_parquet(largest_misses, output_dir / "largest_misses.parquet")
            artifacts["largest_misses"] = "largest_misses.parquet"
        return artifacts

    @staticmethod
    def _race_outcome_frame(comparison: pl.DataFrame) -> pl.DataFrame:
        schema = {
            "race_id": pl.Utf8,
            "cycle": pl.Int64,
            "geography_type": pl.Utf8,
            "geography": pl.Utf8,
            "office_type": pl.Utf8,
            "race_type": pl.Utf8,
            "seats": pl.Int64,
            "control_body": pl.Utf8,
            "predicted_winner_option_id": pl.Utf8,
            "predicted_winner_name": pl.Utf8,
            "predicted_winner_party": pl.Utf8,
            "predicted_winner_probability": pl.Float64,
            "actual_winner_option_id": pl.Utf8,
            "actual_winner_name": pl.Utf8,
            "actual_winner_party": pl.Utf8,
            "actual_winner_probability": pl.Float64,
            "race_winner_correct": pl.Boolean,
        }
        if comparison.is_empty() or "race_id" not in comparison.columns:
            return pl.DataFrame(schema=schema)

        name_expr = pl.col("name") if "name" in comparison.columns else pl.lit(None)
        party_expr = pl.col("party") if "party" in comparison.columns else pl.lit(None)
        base_columns = [
            column
            for column in [
                "race_id",
                "cycle",
                "geography_type",
                "geography",
                "office_type",
                "race_type",
                "seats",
                "control_body",
            ]
            if column in comparison.columns
        ]
        sorted_forecast = comparison.with_columns(
            pl.col("winner_probability").fill_null(-1.0).alias("_winner_probability_sort")
        ).sort(
            ["race_id", "_winner_probability_sort", "option_id"],
            descending=[False, True, False],
        )
        predicted = (
            sorted_forecast.group_by("race_id", maintain_order=True)
            .head(1)
            .select(
                [
                    *base_columns,
                    pl.col("option_id").alias("predicted_winner_option_id"),
                    name_expr.alias("predicted_winner_name"),
                    party_expr.alias("predicted_winner_party"),
                    pl.col("winner_probability").alias("predicted_winner_probability"),
                ]
            )
        )
        actual = (
            comparison.filter(pl.col("actual_winner"))
            .sort(["race_id", "option_id"])
            .group_by("race_id", maintain_order=True)
            .head(1)
            .select(
                [
                    "race_id",
                    pl.col("option_id").alias("actual_winner_option_id"),
                    name_expr.alias("actual_winner_name"),
                    party_expr.alias("actual_winner_party"),
                    pl.col("winner_probability").alias("actual_winner_probability"),
                ]
            )
        )
        outcome = predicted.join(actual, on="race_id", how="left").with_columns(
            (pl.col("predicted_winner_option_id") == pl.col("actual_winner_option_id")).alias(
                "race_winner_correct"
            )
        )
        for column, dtype in schema.items():
            if column not in outcome.columns:
                outcome = outcome.with_columns(pl.lit(None, dtype=dtype).alias(column))
        return outcome.select(list(schema))

    @staticmethod
    def _largest_miss_frame(comparison: pl.DataFrame, limit: int = 10) -> pl.DataFrame:
        columns = [
            "race_id",
            "geography",
            "office_type",
            "option_id",
            "name",
            "party",
            "winner_probability",
            "actual_winner_probability",
            "vote_share_mean",
            "actual_vote_share",
            "vote_share_error",
            "absolute_vote_share_error",
            "actual_winner",
            "predicted_winner",
            "race_winner_correct",
        ]
        present = [column for column in columns if column in comparison.columns]
        if comparison.is_empty() or "absolute_vote_share_error" not in comparison.columns:
            return pl.DataFrame()
        frame = comparison.filter(pl.col("absolute_vote_share_error").is_not_null())
        if frame.is_empty():
            return pl.DataFrame()
        return frame.sort("absolute_vote_share_error", descending=True).head(limit).select(present)

    @classmethod
    def _largest_misses(cls, comparison: pl.DataFrame, limit: int = 10) -> list[dict[str, Any]]:
        return cls._json_records(cls._largest_miss_frame(comparison, limit=limit))

    @classmethod
    def _actual_winner_probabilities(cls, race_outcomes: pl.DataFrame) -> list[dict[str, Any]]:
        columns = [
            "race_id",
            "geography",
            "office_type",
            "seats",
            "actual_winner_option_id",
            "actual_winner_name",
            "actual_winner_party",
            "actual_winner_probability",
            "race_winner_correct",
        ]
        present = [column for column in columns if column in race_outcomes.columns]
        if race_outcomes.is_empty():
            return []
        return cls._json_records(race_outcomes.select(present).sort("race_id"))

    @classmethod
    def _electoral_college_summary(cls, race_outcomes: pl.DataFrame) -> dict[str, Any]:
        if race_outcomes.is_empty() or "seats" not in race_outcomes.columns:
            return cls._empty_electoral_college_summary()
        frame = race_outcomes.filter(
            (pl.col("seats").is_not_null())
            & (pl.col("seats") > 0)
            & pl.col("actual_winner_party").is_not_null()
            & pl.col("predicted_winner_party").is_not_null()
        )
        if frame.is_empty():
            return cls._empty_electoral_college_summary()
        modeled_votes = int(frame["seats"].sum())
        full_ec = modeled_votes >= 270
        threshold = 270.0 if full_ec else modeled_votes / 2.0
        predicted_counts = cls._party_vote_counts(frame, "predicted_winner_party")
        actual_counts = cls._party_vote_counts(frame, "actual_winner_party")
        predicted_winner = cls._party_with_most_votes(predicted_counts, threshold, full_ec)
        actual_winner = cls._party_with_most_votes(actual_counts, threshold, full_ec)
        winner_correct = (
            None
            if predicted_winner is None or actual_winner is None
            else predicted_winner == actual_winner
        )
        return {
            "available": True,
            "scope": "full_electoral_college" if full_ec else "modeled_state_slice",
            "modeled_electoral_votes": modeled_votes,
            "state_count": frame.height,
            "threshold": threshold,
            "winner_accuracy": None if winner_correct is None else float(winner_correct),
            "winner_correct": winner_correct,
            "predicted_winner_party": predicted_winner,
            "actual_winner_party": actual_winner,
            "predicted_party_electoral_votes": predicted_counts,
            "actual_party_electoral_votes": actual_counts,
        }

    @staticmethod
    def _empty_electoral_college_summary() -> dict[str, Any]:
        return {
            "available": False,
            "scope": "not_applicable",
            "modeled_electoral_votes": 0,
            "state_count": 0,
            "threshold": None,
            "winner_accuracy": None,
            "winner_correct": None,
            "predicted_winner_party": None,
            "actual_winner_party": None,
            "predicted_party_electoral_votes": [],
            "actual_party_electoral_votes": [],
        }

    @classmethod
    def _party_vote_counts(cls, frame: pl.DataFrame, party_column: str) -> list[dict[str, Any]]:
        counts = (
            frame.group_by(party_column)
            .agg(pl.col("seats").sum().alias("electoral_votes"))
            .rename({party_column: "party"})
            .sort("electoral_votes", descending=True)
        )
        return cls._json_records(counts)

    @staticmethod
    def _party_with_most_votes(
        counts: list[dict[str, Any]], threshold: float, require_threshold: bool
    ) -> str | None:
        if not counts:
            return None
        top = counts[0]
        votes = float(top.get("electoral_votes") or 0)
        if len(counts) > 1 and votes == float(counts[1].get("electoral_votes") or 0):
            return None
        if require_threshold and votes < threshold:
            return None
        return str(top.get("party"))

    @staticmethod
    def _mean_bool_or_none(frame: pl.DataFrame, column: str) -> float | None:
        if frame.is_empty() or column not in frame.columns:
            return None
        value = frame.select(pl.col(column).cast(pl.Float64).mean()).item()
        return None if value is None else float(value)

    @classmethod
    def _json_records(cls, frame: pl.DataFrame) -> list[dict[str, Any]]:
        return [
            {key: cls._json_value(value) for key, value in row.items()} for row in frame.to_dicts()
        ]

    @staticmethod
    def _json_value(value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, float) and np.isnan(value):
            return None
        if hasattr(value, "item"):
            return value.item()
        return value

    @staticmethod
    def _mean_or_none(frame: pl.DataFrame, column: str) -> float | None:
        if frame.is_empty() or column not in frame.columns:
            return None
        value = frame.select(pl.col(column).mean()).item()
        return None if value is None else float(value)

    def _write_plots(
        self, comparison: pl.DataFrame, output_dir: Path
    ) -> dict[str, list[dict[str, str]]]:
        plot_dir = output_dir / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        manifest: dict[str, list[dict[str, str]]] = {"comparison": []}
        self._add_plot(
            manifest,
            self._vote_share_plot(comparison, plot_dir),
            "Forecast vote share versus actual vote share",
        )
        self._add_plot(
            manifest,
            self._winner_probability_plot(comparison, plot_dir),
            "Winner probability versus actual outcome",
        )
        self._add_plot(
            manifest,
            self._actual_winner_probability_plot(comparison, plot_dir),
            "Actual-winner probabilities",
        )
        self._add_plot(
            manifest,
            self._largest_misses_plot(comparison, plot_dir),
            "Largest vote-share misses",
        )
        return manifest

    @staticmethod
    def _add_plot(manifest: dict[str, list[dict[str, str]]], path: Path | None, title: str) -> None:
        if path is None:
            return
        manifest["comparison"].append({"title": title, "path": f"plots/{path.name}"})

    def _vote_share_plot(self, comparison: pl.DataFrame, plot_dir: Path) -> Path | None:
        frame = comparison.filter(pl.col("vote_share_mean").is_not_null())
        if frame.is_empty():
            return None
        actual = frame["actual_vote_share"].to_numpy()
        predicted = frame["vote_share_mean"].to_numpy()
        labels = frame["option_id"].to_list()
        fig, ax = plt.subplots(figsize=(7, 6), dpi=140)
        ax.scatter(actual, predicted, color="#4c78a8", s=70)
        ax.plot([0, 1], [0, 1], linestyle="--", color="#777777", linewidth=1)
        for x_value, y_value, label in zip(actual, predicted, labels, strict=True):
            ax.annotate(str(label).split("-")[-1], (x_value, y_value), fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Actual vote share")
        ax.set_ylabel("Forecast mean vote share")
        ax.set_title("Forecast vs Actual Vote Share")
        return self._save(fig, plot_dir / "vote_share_forecast_vs_actual.png")

    def _winner_probability_plot(self, comparison: pl.DataFrame, plot_dir: Path) -> Path | None:
        frame = comparison.filter(pl.col("winner_probability").is_not_null())
        if frame.is_empty():
            return None
        x_values = frame["winner_probability"].to_numpy()
        y_values = frame["actual_winner"].cast(pl.Int8).to_numpy()
        colors = np.where(y_values == 1, "#59a14f", "#e15759")
        fig, ax = plt.subplots(figsize=(7, 4), dpi=140)
        ax.scatter(x_values, y_values, color=colors, s=80)
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.15, 1.15)
        ax.set_xlabel("Forecast winner probability")
        ax.set_ylabel("Actual winner")
        ax.set_yticks([0, 1])
        ax.set_title("Winner Probability vs Actual Outcome")
        return self._save(fig, plot_dir / "winner_probability_vs_actual.png")

    def _actual_winner_probability_plot(
        self, comparison: pl.DataFrame, plot_dir: Path
    ) -> Path | None:
        race_outcomes = self._race_outcome_frame(comparison)
        if race_outcomes.is_empty():
            return None
        frame = race_outcomes.filter(pl.col("actual_winner_probability").is_not_null()).sort(
            "actual_winner_probability"
        )
        if frame.is_empty():
            return None
        labels = [
            f"{row['geography'] or row['race_id']}\n{row['actual_winner_party']}"
            for row in frame.iter_rows(named=True)
        ]
        values = frame["actual_winner_probability"].to_list()
        colors = [
            "#59a14f" if bool(row["race_winner_correct"]) else "#e15759"
            for row in frame.iter_rows(named=True)
        ]
        fig, ax = plt.subplots(figsize=(8, max(3.8, len(labels) * 0.46)), dpi=140)
        ax.barh(labels, values, color=colors)
        ax.axvline(0.5, color="#777777", linestyle="--", linewidth=1)
        for idx, value in enumerate(values):
            ax.text(min(0.98, float(value) + 0.02), idx, f"{float(value):.1%}", va="center")
        ax.set_xlim(0, 1)
        ax.set_xlabel("Forecast probability assigned to actual winner")
        ax.set_title("Actual-Winner Probability by Race")
        return self._save(fig, plot_dir / "actual_winner_probabilities.png")

    def _largest_misses_plot(self, comparison: pl.DataFrame, plot_dir: Path) -> Path | None:
        frame = self._largest_miss_frame(comparison, limit=12)
        if frame.is_empty():
            return None
        frame = frame.sort("absolute_vote_share_error")
        labels = [
            f"{row.get('name') or row['option_id']}\n{row['race_id']}"
            for row in frame.iter_rows(named=True)
        ]
        values = frame["absolute_vote_share_error"].to_list()
        colors = [
            "#e15759" if bool(row.get("actual_winner")) else "#4c78a8"
            for row in frame.iter_rows(named=True)
        ]
        fig, ax = plt.subplots(figsize=(8.4, max(4.0, len(labels) * 0.5)), dpi=140)
        ax.barh(labels, values, color=colors)
        for idx, value in enumerate(values):
            ax.text(float(value) + 0.002, idx, f"{float(value):.1%}", va="center", fontsize=9)
        ax.set_xlabel("Absolute vote-share error")
        ax.set_title("Largest Vote-Share Misses")
        return self._save(fig, plot_dir / "largest_vote_share_misses.png")

    @staticmethod
    def _save(fig: plt.Figure, path: Path) -> Path:
        fig.tight_layout()
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        return path

    @staticmethod
    def _html_report(summary: dict[str, Any], comparison: pl.DataFrame) -> str:
        rows = ""
        row_columns = [
            "race_id",
            "option_id",
            "winner_probability",
            "actual_winner_probability",
            "vote_share_mean",
            "actual_vote_share",
            "actual_winner",
            "predicted_winner",
            "race_winner_correct",
            "absolute_vote_share_error",
        ]
        if not comparison.is_empty():
            present = [column for column in row_columns if column in comparison.columns]
            for row in comparison.select(present).iter_rows(named=True):
                rows += (
                    "<tr>"
                    + "".join(
                        f"<td>{html.escape(ResultComparator._format_cell(value))}</td>"
                        for value in row.values()
                    )
                    + "</tr>"
                )
        plot_figures = "".join(
            '<figure><img src="'
            f'{html.escape(entry["path"])}" width="800" alt="{html.escape(entry["title"])}">'
            f"<figcaption>{html.escape(entry['title'])}</figcaption></figure>"
            for entry in summary.get("plot_manifest", {}).get("comparison", [])
        )
        header_cells = "".join(
            f"<th>{html.escape(column.replace('_', ' ').title())}</th>" for column in row_columns
        )
        return f"""<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><title>Result Comparison</title></head>
<body>
<h1>Result Comparison: {html.escape(str(summary["comparison_id"]))}</h1>
<h2>Summary</h2>
<pre>{html.escape(json.dumps(summary, indent=2, sort_keys=True))}</pre>
<h2>Plots</h2>
{plot_figures}
<h2>Rows</h2>
<table>
<thead><tr>{header_cells}</tr></thead>
<tbody>{rows}</tbody>
</table>
</body>
</html>
"""

    @staticmethod
    def _narrative(summary: dict[str, Any], comparison: pl.DataFrame) -> str:
        if comparison.is_empty():
            return (
                "# Forecast Comparison Narrative\n\n"
                "No matching forecast and result rows were found.\n"
            )
        largest_error = (
            comparison.sort("absolute_vote_share_error", descending=True)
            .select(["race_id", "option_id", "absolute_vote_share_error"])
            .row(0, named=True)
        )
        misses = comparison.filter(pl.col("actual_winner") & ~pl.col("predicted_winner"))
        miss_text = (
            "No winner misses among matched races."
            if misses.is_empty()
            else "Missed winners: "
            + ", ".join(
                f"{row['race_id']} ({row['option_id']})" for row in misses.iter_rows(named=True)
            )
            + "."
        )
        actual_probability_rows = ResultComparator._markdown_records(
            summary["actual_winner_probabilities"],
            [
                "race_id",
                "actual_winner_party",
                "actual_winner_probability",
                "race_winner_correct",
            ],
        )
        largest_miss_rows = ResultComparator._markdown_records(
            summary["largest_misses"],
            [
                "race_id",
                "option_id",
                "absolute_vote_share_error",
                "actual_winner",
                "predicted_winner",
            ],
        )
        state_accuracy = summary["state_accuracy"]
        state_count = summary["state_accuracy_n"]
        ec_accuracy = summary["ec_winner_accuracy"]
        ec_scope = summary["electoral_college"]["scope"]
        return f"""# Forecast Comparison Narrative

- Compared races: `{summary["race_count"]}`
- Matched rows: `{summary["row_count"]}`
- Winner accuracy: `{summary["winner_accuracy"]}`
- Presidential state accuracy: `{state_accuracy}` over `{state_count}` state races
- Electoral College winner accuracy: `{ec_accuracy}` ({ec_scope})
- Mean absolute vote-share error: `{summary["mean_absolute_vote_share_error"]}`
- Brier score: `{summary["brier_score"]}`
- Upset count: `{summary["upset_count"]}`

{miss_text}

Largest vote-share error: `{largest_error["race_id"]}` / `{largest_error["option_id"]}`.

Absolute error: `{largest_error["absolute_vote_share_error"]}`.

Actual-winner probabilities:

{actual_probability_rows}

Largest misses:

{largest_miss_rows}
"""

    @staticmethod
    def _format_cell(value: Any) -> str:
        if value is None:
            return "n/a"
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value)

    @staticmethod
    def _markdown_records(records: list[dict[str, Any]], columns: list[str]) -> str:
        if not records:
            return "- n/a"
        lines = []
        for record in records:
            parts = [f"{column}={record.get(column)}" for column in columns]
            lines.append("- " + "; ".join(parts))
        return "\n".join(lines)
