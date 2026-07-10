"""Freeze model/baseline definitions before a shadow evaluation window."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from civic_signal.config import ProjectContext
from civic_signal.storage.io import read_json, write_json


class ShadowPreregistration:
    """Write and load immutable preregistration artifacts for a shadow profile."""

    def __init__(self, context: ProjectContext, profile_id: str) -> None:
        self.context = context
        self.profile_id = profile_id
        self.profile_dir = context.artifacts_dir / "shadow" / profile_id
        self.path = self.profile_dir / "preregistration.json"

    def freeze(
        self,
        *,
        cycle: int,
        offices: list[str],
        horizon_buckets: list[int] | list[str],
        primary_model: dict[str, Any] | None = None,
        baselines: list[str] | None = None,
        comparators: list[dict[str, Any]] | None = None,
        model_config: dict[str, Any] | None = None,
        frozen_at: str | None = None,
        force: bool = False,
    ) -> dict[str, Any]:
        if self.path.exists() and not force:
            existing = read_json(self.path)
            if existing.get("frozen") is True:
                raise FileExistsError(
                    f"Preregistration already frozen at {self.path}; "
                    "post-freeze repairs require force=True and a new version line."
                )
        config = model_config or (
            self.context.read_yaml("model.yaml")
            if (self.context.config_dir / "model.yaml").exists()
            else {}
        )
        payload = {
            "schema_version": "1.0.0",
            "profile_id": self.profile_id,
            "cycle": int(cycle),
            "offices": list(offices),
            "horizon_buckets": list(horizon_buckets),
            "primary_model": primary_model
            or {
                "model_version": config.get("model_version"),
                "inference_engine": "bayes",
                "bayesian_backend": dict(config.get("bayesian", {})).get("backend", "nuts"),
                "trusted_components": dict(config.get("trusted_components", {})),
            },
            "baselines": baselines
            or [
                "prior_only",
                "previous_cycle_swing",
                "fundamentals_only",
                "polling_average",
                "market_implied",
            ],
            "comparators": comparators or [],
            "metrics": {
                "primary": "mean_log_score",
                "secondary": ["brier_score", "interval_90_coverage"],
                "uncertainty": "cycle_clustered_bootstrap",
            },
            "frozen": True,
            "frozen_at": frozen_at or datetime.now(UTC).isoformat(),
            "version_line": "primary",
            "post_freeze_repairs_require_new_version": True,
        }
        write_json(payload, self.path)
        return payload

    def load(self) -> dict[str, Any] | None:
        if not self.path.exists():
            return None
        return read_json(self.path)

    def is_frozen(self) -> bool:
        payload = self.load()
        return bool(payload and payload.get("frozen") is True)
