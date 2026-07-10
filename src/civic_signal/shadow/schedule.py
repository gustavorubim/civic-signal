"""Shadow schedule generation and history persistence."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl

from civic_signal.storage.io import ensure_parent, write_json


class ShadowSchedule:
    """Immutable daily schedule for a shadow profile."""

    def __init__(self, profile_dir: Path, profile_id: str) -> None:
        self.profile_dir = Path(profile_dir)
        self.profile_id = profile_id
        self.history_path = self.profile_dir / "schedule_history.parquet"
        self.profile_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def date_range(window_start: str, window_end: str) -> list[date]:
        start = date.fromisoformat(window_start)
        end = date.fromisoformat(window_end)
        if end < start:
            raise ValueError("window_end must be on or after window_start")
        days = (end - start).days + 1
        return [start + timedelta(days=offset) for offset in range(days)]

    def planned_run_id(self, as_of: date | str) -> str:
        day = as_of if isinstance(as_of, date) else date.fromisoformat(str(as_of)[:10])
        return f"shadow-{self.profile_id}-{day.isoformat()}"

    def ensure_schedule(
        self,
        *,
        window_start: str,
        window_end: str,
        scenario: str | None = None,
    ) -> pl.DataFrame:
        rows = []
        for day in self.date_range(window_start, window_end):
            rows.append(
                {
                    "profile_id": self.profile_id,
                    "scheduled_date": day.isoformat(),
                    "run_id": self.planned_run_id(day),
                    "as_of": day.isoformat(),
                    "scenario": scenario,
                    "status": "scheduled",
                    "forecast_run_dir": "",
                    "recorded_at": "",
                    "fallback_used": False,
                    "source_age_breach": False,
                    "silent_publication_failure": False,
                    "full_refit": False,
                    "promotion_refused": False,
                    "public_probabilities_published": False,
                    "health_passed": False,
                }
            )
        planned = pl.DataFrame(rows)
        existing = self.load_history()
        if existing.is_empty():
            self._write_history(planned)
            return planned
        # Preserve completed history rows; add only missing scheduled dates.
        existing_dates = set(existing["scheduled_date"].to_list())
        additions = planned.filter(~pl.col("scheduled_date").is_in(sorted(existing_dates)))
        if additions.is_empty():
            return existing
        merged = pl.concat([existing, additions], how="vertical_relaxed").sort("scheduled_date")
        self._write_history(merged)
        return merged

    def load_history(self) -> pl.DataFrame:
        if not self.history_path.exists():
            return pl.DataFrame(
                schema={
                    "profile_id": pl.String,
                    "scheduled_date": pl.String,
                    "run_id": pl.String,
                    "as_of": pl.String,
                    "scenario": pl.String,
                    "status": pl.String,
                    "forecast_run_dir": pl.String,
                    "recorded_at": pl.String,
                    "fallback_used": pl.Boolean,
                    "source_age_breach": pl.Boolean,
                    "silent_publication_failure": pl.Boolean,
                    "full_refit": pl.Boolean,
                    "promotion_refused": pl.Boolean,
                    "public_probabilities_published": pl.Boolean,
                    "health_passed": pl.Boolean,
                }
            )
        return pl.read_parquet(self.history_path)

    def record_run(
        self,
        *,
        scheduled_date: str,
        status: str,
        forecast_run_dir: str,
        fallback_used: bool = False,
        source_age_breach: bool = False,
        silent_publication_failure: bool = False,
        full_refit: bool = False,
        promotion_refused: bool = False,
        public_probabilities_published: bool = False,
        health_passed: bool = True,
        extra: dict[str, Any] | None = None,
    ) -> pl.DataFrame:
        history = self.load_history()
        if history.is_empty() or scheduled_date not in set(history["scheduled_date"].to_list()):
            raise KeyError(f"scheduled_date {scheduled_date} not in shadow schedule")
        now = datetime.now().astimezone().isoformat()
        updates = {
            "status": status,
            "forecast_run_dir": forecast_run_dir,
            "recorded_at": now,
            "fallback_used": bool(fallback_used),
            "source_age_breach": bool(source_age_breach),
            "silent_publication_failure": bool(silent_publication_failure),
            "full_refit": bool(full_refit),
            "promotion_refused": bool(promotion_refused),
            "public_probabilities_published": bool(public_probabilities_published),
            "health_passed": bool(health_passed),
        }
        if extra:
            updates.update(extra)
        rows = history.to_dicts()
        found = False
        for row in rows:
            if str(row.get("scheduled_date")) == scheduled_date:
                row.update(updates)
                found = True
        if not found:
            raise KeyError(f"scheduled_date {scheduled_date} not in shadow schedule")
        updated = pl.DataFrame(rows)
        self._write_history(updated)
        return updated

    def _write_history(self, frame: pl.DataFrame) -> None:
        ensure_parent(self.history_path)
        frame.write_parquet(self.history_path)

    def write_manifest(self, payload: dict[str, Any]) -> Path:
        path = self.profile_dir / "schedule_manifest.json"
        write_json(payload, path)
        return path
