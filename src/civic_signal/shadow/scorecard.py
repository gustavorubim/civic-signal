"""Office/horizon shadow scorecards with cycle-clustered uncertainty placeholders."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from civic_signal.storage.io import write_json


class ShadowScorecardBuilder:
    """Assemble scorecards from completed shadow runs and optional preregistration."""

    def __init__(self, profile_dir: Path) -> None:
        self.profile_dir = Path(profile_dir)

    def build(
        self,
        *,
        history: pl.DataFrame,
        preregistration: dict[str, Any] | None = None,
        run_artifacts_root: Path | None = None,
    ) -> dict[str, Any]:
        completed = (
            history.filter(pl.col("status") == "completed")
            if not history.is_empty() and "status" in history.columns
            else history
        )
        offices = list((preregistration or {}).get("offices") or [])
        horizons = list((preregistration or {}).get("horizon_buckets") or [])
        rows: list[dict[str, Any]] = []
        for office in offices or ["all"]:
            for horizon in horizons or ["all"]:
                rows.append(
                    {
                        "office": office,
                        "horizon_bucket": horizon,
                        "run_count": int(completed.height) if not completed.is_empty() else 0,
                        "mean_log_score": None,
                        "brier_score": None,
                        "interval_90_coverage": None,
                        "cycle_clustered_ci": None,
                        "status": "insufficient_evidence"
                        if completed.is_empty()
                        else "observational",
                        "note": (
                            "Scores remain observational until nested outer-fold and "
                            "preregistered comparator snapshots are available."
                        ),
                    }
                )
        comparator_defs = list((preregistration or {}).get("comparators") or [])
        completed_count = int(completed.height) if not completed.is_empty() else 0
        payload = {
            "schema_version": "1.0.0",
            "generated_at": datetime.now(UTC).isoformat(),
            "profile_dir": str(self.profile_dir),
            "preregistration_present": preregistration is not None,
            "preregistration_frozen": bool((preregistration or {}).get("frozen")),
            "completed_shadow_runs": completed_count,
            "office_horizon_rows": rows,
            "comparators": comparator_defs,
            # "Best evidenced" / R26-style claims remain blocked in shadow mode.
            "claim_status": "insufficient_evidence",
            "best_claim_status": "insufficient_evidence",
            "public_production_promotion": False,
            "claim_reasons": [
                "Shadow scorecards are recorded for monitoring only until multi-cycle "
                "nested evaluation and preregistered public comparators are complete.",
                "No public production promotion from shadow scorecards.",
            ],
        }
        # Optional: attach control summaries from latest completed run.
        if not completed.is_empty() and "forecast_run_dir" in completed.columns:
            latest = completed.sort("scheduled_date").tail(1)
            run_dir = Path(str(latest["forecast_run_dir"][0] or ""))
            if run_dir.exists() and (run_dir / "control_forecasts.parquet").exists():
                controls = pl.read_parquet(run_dir / "control_forecasts.parquet")
                payload["latest_control_snapshot"] = controls.to_dicts()
        write_json(payload, self.profile_dir / "shadow_scorecard.json")
        return payload
