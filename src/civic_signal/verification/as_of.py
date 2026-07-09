"""As-of integrity verification surface."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from civic_signal.config import ProjectContext
from civic_signal.storage.io import write_json, write_text


class AsOfVerificationRunner:
    """Write as-of audit evidence and report violations."""

    def __init__(self, context: ProjectContext) -> None:
        self.context = context

    def verify(
        self,
        *,
        run_id: str,
        scenario_family: str | None = None,
        cycles: str | None = None,
        offsets: str | None = None,
        as_of: str | None = None,
    ) -> dict[str, Any]:
        out_dir = self.context.artifacts_dir / "as_of_audits" / run_id
        out_dir.mkdir(parents=True, exist_ok=True)

        # Prefer an existing audit attached to a forecast run.
        forecast_audit = self.context.artifacts_dir / "forecasts" / run_id / "as_of_audit.json"
        if forecast_audit.exists():
            from civic_signal.storage.io import read_json

            audit = read_json(forecast_audit)
        else:
            audit = {
                "run_id": run_id,
                "scenario_family": scenario_family,
                "cycles": cycles,
                "offsets": offsets,
                "as_of": as_of,
                "future_eligible_rows": None,
                "violations": None,
                "time_travel_canaries_passed": None,
                "status": "insufficient_evidence",
                "detail": (
                    "No as-of audit primary artifact available; real historical "
                    "archive required for production pass."
                ),
            }

        write_json(audit, out_dir / "as_of_audit.json")
        future = audit.get("future_eligible_rows")
        passed = (
            future == 0
            and audit.get("time_travel_canaries_passed") is not False
            and audit.get("status") != "insufficient_evidence"
            and future is not None
        )
        payload = {
            "run_id": run_id,
            "passed": bool(passed),
            "exit_nonzero": not bool(passed),
            "audit_path": str(out_dir / "as_of_audit.json"),
            "generated_at": datetime.now(UTC).isoformat(),
            "audit": audit,
        }
        write_json(payload, out_dir / "as_of_verification.json")
        write_text(
            f"# As-of verification\n\npassed={payload['passed']}\n\n```json\n{payload}\n```\n",
            out_dir / "as_of_verification_report.md",
        )
        return payload
