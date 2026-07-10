"""Verify a shadow window against M8 readiness rules."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import polars as pl

from civic_signal.config import ProjectContext
from civic_signal.shadow.preregistration import ShadowPreregistration
from civic_signal.shadow.schedule import ShadowSchedule
from civic_signal.shadow.scorecard import ShadowScorecardBuilder
from civic_signal.storage.io import read_json, write_json, write_text


class ShadowVerificationRunner:
    """Recompute shadow readiness; incomplete history cannot pass."""

    def __init__(self, context: ProjectContext) -> None:
        self.context = context
        self.shadow_config = (
            context.read_yaml("shadow.yaml")
            if (context.config_dir / "shadow.yaml").exists()
            else {}
        )

    def verify(
        self,
        *,
        profile: str,
        window_start: str,
        window_end: str,
    ) -> dict[str, Any]:
        profiles = dict(self.shadow_config.get("profiles") or {})
        profile_cfg = dict(profiles.get(profile) or {})
        profile_declared = profile in profiles
        if not profile_declared:
            # Unknown profiles may be inspected, but can never satisfy a
            # production-readiness window without a checked-in contract.
            profile_cfg = {
                "min_consecutive_days": 60,
                "allow_predeclared_window": False,
                "require_preregistration": True,
                "max_fallback_rate": 0.0,
                "max_silent_publication_failures": 0,
                "require_no_unquarantined_fallback": True,
            }

        profile_dir = self.context.artifacts_dir / "shadow" / profile
        schedule = ShadowSchedule(profile_dir, profile)
        history = schedule.load_history()
        if history.is_empty():
            history = schedule.ensure_schedule(
                window_start=window_start,
                window_end=window_end,
                scenario=profile_cfg.get("scenario"),
            )

        planned_dates = [d.isoformat() for d in ShadowSchedule.date_range(window_start, window_end)]
        history_window = history.filter(pl.col("scheduled_date").is_in(planned_dates))
        by_date = {str(row["scheduled_date"]): row for row in history_window.iter_rows(named=True)}

        missing_runs: list[str] = []
        completed_dates: list[str] = []
        violations: list[str] = []
        if not profile_declared:
            violations.append(f"unknown shadow profile: {profile}")
        fallback_days = 0
        silent_failures = 0
        source_breaches = 0
        full_refits = 0
        promotion_refusals = 0
        corrected_inputs = 0
        health_failures = 0

        for day in planned_dates:
            row = by_date.get(day)
            if row is None or row.get("status") in {None, "scheduled", "missing"}:
                missing_runs.append(day)
                continue
            if row.get("status") != "completed":
                missing_runs.append(day)
                if row.get("health_passed") is False:
                    health_failures += 1
                if row.get("silent_publication_failure"):
                    silent_failures += 1
                continue
            completed_dates.append(day)
            if row.get("health_passed") is False:
                health_failures += 1
                violations.append(f"{day}: required shadow health monitors failed")
            if row.get("fallback_used"):
                fallback_days += 1
            if row.get("source_age_breach"):
                source_breaches += 1
            if row.get("full_refit"):
                full_refits += 1
            if row.get("promotion_refused"):
                promotion_refusals += 1
            if row.get("public_probabilities_published"):
                silent_failures += 1
                violations.append(f"{day}: public probabilities published during shadow")
            # Optional correction flag if present.
            if row.get("source_correction_reconciled") is False:
                corrected_inputs += 1
                violations.append(f"{day}: source correction not reconciled to snapshot")

        min_days = int(profile_cfg.get("min_consecutive_days", 60))
        consecutive = self._max_consecutive_days(completed_dates)
        window_complete = not missing_runs
        allow_window = bool(profile_cfg.get("allow_predeclared_window", True))
        declared_window = dict(profile_cfg.get("predeclared_window") or {})
        exact_declared_window = bool(
            allow_window
            and declared_window.get("window_start") == window_start
            and declared_window.get("window_end") == window_end
        )
        length_ok = consecutive >= min_days or (
            exact_declared_window and window_complete and consecutive > 0
        )

        fallback_rate = fallback_days / max(len(completed_dates), 1) if completed_dates else 1.0
        max_fallback = float(profile_cfg.get("max_fallback_rate", 0.0))
        if fallback_rate > max_fallback:
            violations.append(
                f"fallback rate {fallback_rate:.3f} exceeds max_fallback_rate {max_fallback}"
            )
        if source_breaches:
            violations.append(f"unresolved source-age breaches on {source_breaches} day(s)")
        if silent_failures > int(profile_cfg.get("max_silent_publication_failures", 0)):
            violations.append(f"silent publication failures: {silent_failures}")
        if profile_cfg.get("require_no_unquarantined_fallback", True) and fallback_days:
            violations.append("unquarantined fallback used during shadow window")
        if missing_runs:
            violations.append(f"missing scheduled runs: {len(missing_runs)}")
        if not length_ok:
            violations.append(
                f"consecutive completed days {consecutive} < required {min_days} "
                "and no exact configured predeclared window is complete"
            )

        prereg = ShadowPreregistration(self.context, profile).load()
        prereg_present = bool(prereg and prereg.get("frozen"))
        if profile_cfg.get("require_preregistration", True) and not prereg_present:
            violations.append("missing frozen preregistration.json")

        scorecard_path = profile_dir / "shadow_scorecard.json"
        scorecard_present = scorecard_path.exists()
        scorecard_payload: dict[str, Any] | None = None
        if not scorecard_present and not history_window.is_empty():
            scorecard_payload = ShadowScorecardBuilder(profile_dir).build(
                history=history_window, preregistration=prereg
            )
            scorecard_present = scorecard_path.exists()
        elif scorecard_present:
            try:
                scorecard_payload = read_json(scorecard_path)
            except Exception:  # pragma: no cover - defensive
                scorecard_payload = None

        if scorecard_payload is not None:
            if scorecard_payload.get("claim_status") not in {
                None,
                "insufficient_evidence",
            }:
                violations.append(
                    "shadow scorecard claim_status must remain insufficient_evidence "
                    f"(got {scorecard_payload.get('claim_status')!r})"
                )
            if scorecard_payload.get("public_production_promotion") is True:
                violations.append("shadow scorecard must not allow public production promotion")
            if scorecard_payload.get("best_claim_status") not in {
                None,
                "insufficient_evidence",
            }:
                violations.append(
                    "shadow scorecard best_claim_status must remain insufficient_evidence"
                )

        passed = not violations and length_ok and (not missing_runs)
        # Plan: missing schedules or incomplete history cannot be converted into a pass.
        if missing_runs:
            passed = False

        report = {
            "schema_version": "1.0.0",
            "profile": profile,
            "window_start": window_start,
            "window_end": window_end,
            "passed": passed,
            "exit_nonzero": not passed,
            "consecutive_days": consecutive,
            "min_consecutive_days": min_days,
            "profile_declared": profile_declared,
            "exact_predeclared_window": exact_declared_window,
            "scheduled_runs": len(planned_dates),
            "completed_runs": len(completed_dates),
            "missing_runs": missing_runs,
            "fallback_days": fallback_days,
            "fallback_rate": fallback_rate,
            "source_age_breaches": source_breaches,
            "full_refits": full_refits,
            "promotion_refusals": promotion_refusals,
            "silent_publication_failures": silent_failures,
            "corrected_inputs_unreconciled": corrected_inputs,
            "health_failures": health_failures,
            "preregistration_present": prereg_present,
            "scorecard_present": scorecard_present,
            "public_production_promotion": False,
            "violations": violations,
            "history_path": str(schedule.history_path),
            "report_path": str(profile_dir / "shadow_readiness.json"),
        }
        write_json(report, profile_dir / "shadow_readiness.json")
        write_text(self._human_report(report), profile_dir / "shadow_readiness.md")
        return report

    @staticmethod
    def _max_consecutive_days(completed_dates: list[str]) -> int:
        if not completed_dates:
            return 0
        days = sorted(date.fromisoformat(value) for value in completed_dates)
        best = 1
        current = 1
        for index in range(1, len(days)):
            if days[index] == days[index - 1] + timedelta(days=1):
                current += 1
                best = max(best, current)
            else:
                current = 1
        return best

    @staticmethod
    def _human_report(report: dict[str, Any]) -> str:
        lines = [
            f"# Shadow readiness — {report['profile']}",
            "",
            f"- Window: `{report['window_start']}` → `{report['window_end']}`",
            f"- Passed: `{report['passed']}`",
            f"- Consecutive completed days: `{report['consecutive_days']}` "
            f"(required ≥ `{report['min_consecutive_days']}`)",
            f"- Scheduled runs: `{report['scheduled_runs']}`",
            f"- Completed runs: `{report['completed_runs']}`",
            f"- Missing runs: `{len(report['missing_runs'])}`",
            f"- Preregistration frozen: `{report['preregistration_present']}`",
            f"- Scorecard present: `{report['scorecard_present']}`",
            "",
            "## Violations",
            "",
        ]
        if not report["violations"]:
            lines.append("_None_")
        else:
            for item in report["violations"]:
                lines.append(f"- {item}")
        if report["missing_runs"]:
            lines.extend(["", "## Missing scheduled dates", ""])
            for day in report["missing_runs"][:30]:
                lines.append(f"- `{day}`")
            if len(report["missing_runs"]) > 30:
                lines.append(f"- … {len(report['missing_runs']) - 30} more")
        lines.append("")
        return "\n".join(lines)
