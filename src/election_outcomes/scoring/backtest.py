from __future__ import annotations

from datetime import UTC, datetime
from typing import ClassVar

import polars as pl

from election_outcomes.config import ProjectContext
from election_outcomes.scoring.metrics import score_predictions
from election_outcomes.storage.io import write_json, write_parquet


class BacktestRunner:
    COMPONENT_COLUMNS: ClassVar[dict[str, str]] = {
        "baseline": "baseline_probability",
        "polling": "polls_probability",
        "fundamentals": "fundamentals_probability",
        "markets": "markets_probability",
        "public_signals": "public_signals_probability",
        "ensemble": "ensemble_probability",
    }

    def __init__(self, context: ProjectContext) -> None:
        self.context = context

    def evaluate(self) -> dict[str, object]:
        path = self.context.curated_dir / "backtest_predictions.parquet"
        frame = pl.read_parquet(path) if path.exists() else pl.DataFrame()
        metrics = {
            component: score_predictions(frame, column)
            for component, column in self.COMPONENT_COLUMNS.items()
            if column in frame.columns
        }
        baseline_brier = metrics.get("baseline", {}).get("brier")
        ablations = {}
        for component, values in metrics.items():
            if component == "baseline" or baseline_brier is None:
                continue
            ablations[component] = {
                "brier_delta_vs_baseline": values["brier"] - baseline_brier,
                "beats_or_matches_baseline": values["brier"] <= baseline_brier,
            }
        return {
            "generated_at": datetime.now(UTC).isoformat(),
            "row_count": frame.height,
            "metrics": metrics,
            "ablations": ablations,
        }

    def run(self, run_id: str) -> dict[str, object]:
        payload = self.evaluate()
        out_dir = self.context.artifacts_dir / "backtests" / run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        metrics_rows = [
            {"component": component, **values} for component, values in payload["metrics"].items()
        ]
        write_parquet(pl.DataFrame(metrics_rows), out_dir / "scorecard.parquet")
        write_json(payload, out_dir / "scorecard.json")
        return payload
