"""Source health and operational monitors for shadow runs."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from civic_signal.storage.io import read_json, write_json


class ShadowHealthMonitor:
    """Build machine-readable health reports for a shadow forecast attempt."""

    def __init__(self, max_source_age_hours: float = 72.0) -> None:
        self.max_source_age_hours = float(max_source_age_hours)

    def evaluate_run(self, run_dir: Path, *, as_of: str | None = None) -> dict[str, Any]:
        run_dir = Path(run_dir)
        manifest_path = run_dir / "source_manifest.parquet"
        performance = self._read_json(run_dir / "performance.json")
        posterior = self._read_json(run_dir / "posterior_diagnostics.json")
        daily = self._read_json(run_dir / "latest_daily_update.json")
        decision = self._read_json(run_dir / "publication_decision.json")
        run_manifest = self._read_json(run_dir / "run_manifest.json")

        source_health = self._source_health(manifest_path, as_of=as_of or run_manifest.get("as_of"))
        fallback_used = bool(
            str(posterior.get("fallback_used") or "").strip()
            or performance.get("fallback_used")
            or daily.get("fallback_used")
        )
        full_refit = bool(daily.get("full_refit_executed") or daily.get("needs_full_refit"))
        mcse = performance.get("max_mcse")
        if mcse is None:
            mcse = performance.get("mcse_max")
        sampler_ok = True
        if posterior:
            if int(posterior.get("divergences") or 0) > 0:
                sampler_ok = False
            r_hat = posterior.get("r_hat_max")
            if r_hat is not None and float(r_hat) > 1.01:
                sampler_ok = False
        sampler_required = str(run_manifest.get("inference_engine") or "").lower() == "bayes"

        public_probs = False
        if decision.get("publication_mode") == "production" and decision.get("allowed") is True:
            public_probs = True
        if run_manifest.get("forecast_status") == "production":
            public_probs = True

        poll_movement = self._poll_movement_proxy(run_dir)
        posterior_drift = float(daily.get("posterior_drift") or daily.get("drift") or 0.0)

        report = {
            "schema_version": "1.0.0",
            "run_dir": str(run_dir),
            "generated_at": datetime.now(UTC).isoformat(),
            "monitors": {
                "source_freshness": {
                    "passed": not source_health["source_age_breach"],
                    "max_age_hours": source_health["max_age_hours"],
                    "threshold_hours": self.max_source_age_hours,
                    "stale_sources": source_health["stale_sources"],
                },
                "source_coverage": {
                    "passed": source_health["coverage_ok"],
                    "source_count": source_health["source_count"],
                    "failed_sources": source_health["failed_sources"],
                },
                "poll_movement": {
                    "passed": True,
                    "metric": poll_movement,
                },
                "posterior_drift": {
                    "passed": posterior_drift <= 0.05,
                    "metric": posterior_drift,
                },
                "sampler_health": {
                    "passed": (not sampler_required) or (bool(posterior) and sampler_ok),
                    "required": sampler_required,
                    "posterior_diagnostics_present": bool(posterior),
                },
                "update_vs_refit": {
                    "passed": not (
                        daily.get("needs_full_refit") and not daily.get("full_refit_executed")
                    ),
                    "needs_full_refit": daily.get("needs_full_refit"),
                    "full_refit_executed": daily.get("full_refit_executed"),
                    "mae_vs_refit": daily.get("probability_mae_vs_full_refit")
                    or daily.get("mae_vs_refit"),
                },
                "calibration": {
                    "passed": True,
                    "note": "Shadow records calibration when present; observational gate.",
                    "map_present": (run_dir / "recalibration_map.parquet").exists(),
                },
                "mcse": {
                    "passed": mcse is not None and float(mcse) <= 0.0025,
                    "max_mcse": mcse,
                },
                "fallback_frequency": {
                    "passed": not fallback_used,
                    "fallback_used": fallback_used,
                },
            },
            "flags": {
                "fallback_used": fallback_used,
                "source_age_breach": source_health["source_age_breach"],
                "full_refit": full_refit,
                "public_probabilities_published": public_probs,
                "silent_publication_failure": public_probs,
            },
            "source_health": source_health,
        }
        required_monitors = {
            "source_freshness",
            "source_coverage",
            "posterior_drift",
            "sampler_health",
            "update_vs_refit",
            "mcse",
            "fallback_frequency",
        }
        report["passed"] = all(
            bool(report["monitors"][name]["passed"]) for name in required_monitors
        )
        write_json(report, run_dir / "source_health.json")
        write_json(report, run_dir / "shadow_monitors.json")
        return report

    def _source_health(self, manifest_path: Path, *, as_of: str | None) -> dict[str, Any]:
        if not manifest_path.exists():
            return {
                "source_count": 0,
                "failed_sources": [],
                "stale_sources": [],
                "max_age_hours": None,
                "source_age_breach": True,
                "coverage_ok": False,
            }
        frame = pl.read_parquet(manifest_path)
        source_count = int(frame.height)
        failed = []
        if "status" in frame.columns:
            failed = [
                str(value)
                for value in frame.filter(pl.col("status") == "failed")["source_id"].to_list()
            ]
        stale: list[str] = []
        ages: list[float] = []
        reference = datetime.now(UTC)
        if as_of:
            try:
                reference = datetime.fromisoformat(str(as_of)[:10]).replace(tzinfo=UTC)
            except ValueError:
                pass
        if "retrieved_at" in frame.columns:
            for row in frame.iter_rows(named=True):
                retrieved = row.get("retrieved_at")
                if not retrieved:
                    continue
                try:
                    stamp = datetime.fromisoformat(str(retrieved).replace("Z", "+00:00"))
                except ValueError:
                    continue
                if stamp.tzinfo is None:
                    stamp = stamp.replace(tzinfo=UTC)
                age_hours = (reference - stamp.astimezone(UTC)).total_seconds() / 3600.0
                ages.append(age_hours)
                if age_hours > self.max_source_age_hours:
                    stale.append(str(row.get("source_id")))
        if "freshness_status" in frame.columns:
            stale.extend(
                str(value)
                for value in frame.filter(pl.col("freshness_status") == "stale")[
                    "source_id"
                ].to_list()
            )
        max_age = max(ages) if ages else None
        return {
            "source_count": source_count,
            "failed_sources": sorted(set(failed)),
            "stale_sources": sorted(set(stale)),
            "max_age_hours": max_age,
            "source_age_breach": bool(stale)
            or (max_age is not None and max_age > self.max_source_age_hours),
            "coverage_ok": source_count > 0 and not failed,
        }

    @staticmethod
    def _poll_movement_proxy(run_dir: Path) -> dict[str, Any]:
        path = run_dir / "poll_trajectory.parquet"
        if not path.exists():
            return {"rows": 0, "movement_sd": None}
        frame = pl.read_parquet(path)
        movement = None
        for column in ("mean_share", "posterior_mean", "estimate"):
            if column in frame.columns:
                try:
                    movement = float(frame.select(pl.col(column).std()).item() or 0.0)
                except Exception:  # pragma: no cover - defensive
                    movement = None
                break
        return {"rows": int(frame.height), "movement_sd": movement}

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        try:
            payload = read_json(path)
        except Exception:  # pragma: no cover
            return {}
        return payload if isinstance(payload, dict) else {}
