"""Execute scheduled shadow forecasts without publishing public probabilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from civic_signal.config import ProjectContext
from civic_signal.pipeline import ForecastPipeline
from civic_signal.shadow.health import ShadowHealthMonitor
from civic_signal.shadow.preregistration import ShadowPreregistration
from civic_signal.shadow.schedule import ShadowSchedule
from civic_signal.shadow.scorecard import ShadowScorecardBuilder
from civic_signal.storage.io import read_json, write_json


class ShadowForecastRunner:
    """Run one or more shadow-dated forecasts for a profile."""

    def __init__(self, context: ProjectContext, profile_id: str | None = None) -> None:
        self.context = context
        self.shadow_config = self._load_shadow_config()
        self.profile_id = profile_id or str(
            self.shadow_config.get("default_profile") or "2026-general-shadow"
        )
        profiles = dict(self.shadow_config.get("profiles") or {})
        if self.profile_id not in profiles:
            raise KeyError(f"Unknown shadow profile: {self.profile_id}")
        self.profile = dict(profiles[self.profile_id])
        self.profile_dir = self.context.artifacts_dir / "shadow" / self.profile_id
        self.schedule = ShadowSchedule(self.profile_dir, self.profile_id)
        self.health = ShadowHealthMonitor(
            max_source_age_hours=float(self.profile.get("max_source_age_hours", 72))
        )
        self.prereg = ShadowPreregistration(self.context, self.profile_id)

    def _load_shadow_config(self) -> dict[str, Any]:
        path = self.context.config_dir / "shadow.yaml"
        if not path.exists():
            return {"default_profile": "2026-general-shadow", "profiles": {}}
        return self.context.read_yaml("shadow.yaml")

    def ensure_preregistration(self, *, cycle: int = 2026, force: bool = False) -> dict[str, Any]:
        if self.prereg.is_frozen() and not force:
            loaded = self.prereg.load()
            assert loaded is not None
            return loaded
        return self.prereg.freeze(
            cycle=cycle,
            offices=list(self.profile.get("offices") or []),
            horizon_buckets=list(self.profile.get("horizon_buckets") or []),
            force=force,
        )

    def run_window(
        self,
        *,
        window_start: str,
        window_end: str,
        execute: bool = True,
        scenario: str | None = None,
        inference_engine: str | None = None,
        bayesian_backend: str | None = None,
        quiet: bool = True,
    ) -> dict[str, Any]:
        scenario = scenario or self.profile.get("scenario")
        history = self.schedule.ensure_schedule(
            window_start=window_start,
            window_end=window_end,
            scenario=scenario,
        )
        self.schedule.write_manifest(
            {
                "profile_id": self.profile_id,
                "window_start": window_start,
                "window_end": window_end,
                "scenario": scenario,
                "scheduled_days": int(history.height),
                "execute": execute,
            }
        )
        prereg = None
        if self.profile.get("require_preregistration", True):
            prereg = self.ensure_preregistration()

        executed: list[dict[str, Any]] = []
        if execute:
            for row in history.iter_rows(named=True):
                if row.get("status") == "completed" and row.get("forecast_run_dir"):
                    continue
                result = self.run_day(
                    as_of=str(row["as_of"]),
                    scenario=scenario,
                    inference_engine=inference_engine,
                    bayesian_backend=bayesian_backend,
                    quiet=quiet,
                    scheduled_date=str(row["scheduled_date"]),
                )
                executed.append(result)

        history = self.schedule.load_history()
        ShadowScorecardBuilder(self.profile_dir).build(
            history=history,
            preregistration=prereg or self.prereg.load(),
        )
        return {
            "profile_id": self.profile_id,
            "window_start": window_start,
            "window_end": window_end,
            "scheduled_days": int(history.height),
            "executed": executed,
            "history_path": str(self.schedule.history_path),
            "scorecard_path": str(self.profile_dir / "shadow_scorecard.json"),
            "preregistration_path": str(self.prereg.path),
            "profile_dir": str(self.profile_dir),
        }

    def run_day(
        self,
        *,
        as_of: str,
        scenario: str | None = None,
        inference_engine: str | None = None,
        bayesian_backend: str | None = None,
        quiet: bool = True,
        scheduled_date: str | None = None,
        run_id: str | None = None,
    ) -> dict[str, Any]:
        day = scheduled_date or str(as_of)[:10]
        run_id = run_id or self.schedule.planned_run_id(day)
        scenario = scenario or self.profile.get("scenario")
        inference_engine = inference_engine or self.profile.get("inference_engine")
        bayesian_backend = bayesian_backend or self.profile.get("bayesian_backend")

        # Ensure schedule row exists for single-day invocations.
        history = self.schedule.load_history()
        if history.is_empty() or day not in set(history["scheduled_date"].to_list()):
            self.schedule.ensure_schedule(window_start=day, window_end=day, scenario=scenario)

        pipeline = ForecastPipeline(self.context)
        out_dir = pipeline.run_forecast(
            as_of=as_of,
            run_id=run_id,
            scenario=scenario,
            inference_engine=inference_engine,
            bayesian_backend=bayesian_backend,
            quiet=quiet,
        )
        self._stamp_shadow_mode(out_dir, as_of=as_of, run_id=run_id)
        health = self.health.evaluate_run(out_dir, as_of=as_of)
        flags = dict(health.get("flags") or {})

        # Shadow must never present production public probabilities.
        public_probs = bool(flags.get("public_probabilities_published"))
        silent_failure = public_probs
        health_passed = bool(health.get("passed"))
        run_status = "completed" if health_passed and not silent_failure else "failed"

        # Shadow never promotes to public production; record the refusal explicitly.
        self.schedule.record_run(
            scheduled_date=day,
            status=run_status,
            forecast_run_dir=str(out_dir),
            fallback_used=bool(flags.get("fallback_used")),
            source_age_breach=bool(flags.get("source_age_breach")),
            silent_publication_failure=silent_failure,
            full_refit=bool(flags.get("full_refit")),
            promotion_refused=True,
            public_probabilities_published=public_probs,
            health_passed=health_passed,
        )
        return {
            "scheduled_date": day,
            "run_id": run_id,
            "output_dir": str(out_dir),
            "status": run_status,
            "health": health,
        }

    def _stamp_shadow_mode(self, out_dir: Path, *, as_of: str, run_id: str) -> None:
        """Force shadow labeling: no public production probabilities."""
        run_manifest = {}
        if (out_dir / "run_manifest.json").exists():
            run_manifest = read_json(out_dir / "run_manifest.json")
        run_manifest.update(
            {
                "run_id": run_id,
                "as_of": as_of,
                "publication_mode": "shadow",
                "forecast_status": "shadow",
                "evidence_scope": "shadow",
                "promotion_profile_id": self.profile_id,
                "public_probabilities_published": False,
            }
        )
        write_json(run_manifest, out_dir / "run_manifest.json")
        write_json(
            {
                "run_id": run_id,
                "profile": "shadow",
                "publication_mode": "shadow",
                "allowed": False,
                "blocks_publication": True,
                "blocking_rewards": [],
                "reason": "Shadow mode stores forecasts without public probability publication",
                "shadow_profile_id": self.profile_id,
            },
            out_dir / "publication_decision.json",
        )
