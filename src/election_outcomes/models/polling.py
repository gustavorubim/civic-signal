from __future__ import annotations

import math
from typing import ClassVar

import polars as pl

from election_outcomes.features import FeatureBundle
from election_outcomes.models.common import clamp, logistic, normalize_rows


class PollingModel:
    component = "polling"

    POPULATION_WEIGHTS: ClassVar[dict[str, float]] = {"lv": 1.1, "rv": 1.0, "a": 0.85}
    METHODOLOGY_WEIGHTS: ClassVar[dict[str, float]] = {
        "live_phone": 1.1,
        "mixed": 1.05,
        "online": 0.95,
    }

    def run(self, bundle: FeatureBundle) -> pl.DataFrame:
        if bundle.polls.is_empty():
            return normalize_rows([])
        rows: list[dict[str, object]] = []
        for key, group in bundle.polls.group_by(["race_id", "option_id"], maintain_order=True):
            race_id, option_id = key
            weighted = 0.0
            total_weight = 0.0
            for row in group.iter_rows(named=True):
                sample = max(float(row.get("sample_size") or 600), 1.0)
                pop_weight = self.POPULATION_WEIGHTS.get(str(row.get("population")), 1.0)
                method_weight = self.METHODOLOGY_WEIGHTS.get(str(row.get("methodology")), 1.0)
                sponsor_weight = 0.85 if str(row.get("sponsor_class")) != "nonpartisan" else 1.0
                weight = math.sqrt(sample) * pop_weight * method_weight * sponsor_weight
                weighted += weight * (float(row["pct"]) / 100.0)
                total_weight += weight
            share = clamp(weighted / total_weight)
            rows.append(
                {
                    "race_id": race_id,
                    "option_id": option_id,
                    "component": self.component,
                    "win_probability": logistic((share - 0.5) / 0.04),
                    "vote_share": share,
                    "uncertainty": 0.055,
                    "admitted": True,
                    "explanation": "Sample-size and methodology weighted polling average.",
                }
            )
        return normalize_rows(rows)
