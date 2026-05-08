from __future__ import annotations

import polars as pl

from election_outcomes.features import FeatureBundle
from election_outcomes.models.common import clamp, normalize_rows


class EnsembleModel:
    component = "ensemble"

    def __init__(self, config: dict[str, object]) -> None:
        self.config = config
        self.weights = {
            str(key): float(value)
            for key, value in dict(config.get("component_weights", {})).items()
        }
        self.trusted = {
            str(key): bool(value)
            for key, value in dict(config.get("trusted_components", {})).items()
        }

    def run(self, bundle: FeatureBundle, component_estimates: list[pl.DataFrame]) -> pl.DataFrame:
        estimates = pl.concat(
            [df for df in component_estimates if not df.is_empty()], how="diagonal"
        )
        rows: list[dict[str, object]] = []
        if estimates.is_empty():
            return normalize_rows(rows)
        catalog = {row["race_id"]: row for row in bundle.race_catalog.iter_rows(named=True)}
        for key, group in estimates.group_by(["race_id", "option_id"], maintain_order=True):
            race_id, option_id = key
            race = catalog[str(race_id)]
            if race["tier"] == "C":
                continue
            weighted_probability = weighted_share = weight_total = uncertainty_total = 0.0
            drivers: list[str] = []
            for row in group.iter_rows(named=True):
                component = str(row["component"])
                admitted = bool(row["admitted"]) and self.trusted.get(component, False)
                if not admitted:
                    continue
                weight = self.weights.get(component, 0.0)
                weighted_probability += weight * float(row["win_probability"])
                weighted_share += weight * float(row["vote_share"])
                uncertainty_total += weight * float(row["uncertainty"])
                weight_total += weight
                drivers.append(component)
            if weight_total <= 0:
                continue
            rows.append(
                {
                    "race_id": race_id,
                    "option_id": option_id,
                    "component": self.component,
                    "win_probability": clamp(weighted_probability / weight_total),
                    "vote_share": clamp(weighted_share / weight_total),
                    "uncertainty": uncertainty_total / weight_total,
                    "admitted": True,
                    "explanation": " + ".join(drivers),
                }
            )
        return self._normalize_by_race(normalize_rows(rows))

    @staticmethod
    def _normalize_by_race(frame: pl.DataFrame) -> pl.DataFrame:
        if frame.is_empty():
            return frame
        totals = frame.group_by("race_id").agg(
            pl.col("win_probability").sum().alias("prob_total"),
            pl.col("vote_share").sum().alias("share_total"),
        )
        return (
            frame.join(totals, on="race_id", how="left")
            .with_columns(
                (pl.col("win_probability") / pl.col("prob_total")).alias("win_probability"),
                (pl.col("vote_share") / pl.col("share_total")).alias("vote_share"),
            )
            .drop(["prob_total", "share_total"])
        )
