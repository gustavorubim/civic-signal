"""Atomic publication checks and production-label semantic verification."""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from civic_signal.config import ProjectContext
from civic_signal.scoring.reward_v2 import RewardV2Evaluator
from civic_signal.storage.io import read_json, write_json


class PublicationVerifier:
    """Ensure production labels require a verified promotion manifest."""

    def __init__(self, context: ProjectContext) -> None:
        self.context = context
        self.rewards_config = context.read_yaml("rewards.yaml")

    def verify_semantic(
        self,
        *,
        run_id: str,
        profile: str = "production",
        require_promotion_for_production: bool = True,
    ) -> dict[str, Any]:
        run_dir = self._resolve_run_dir(run_id)
        failures: list[str] = []

        decision = self._read_optional(run_dir / "publication_decision.json")
        promotion = self._read_optional(run_dir / "promotion_manifest.json")
        run_manifest = self._read_optional(run_dir / "run_manifest.json")
        mode = str(
            (decision or {}).get("publication_mode")
            or (run_manifest or {}).get("publication_mode")
            or (promotion or {}).get("publication_mode")
            or "research"
        )

        if mode == "production" and require_promotion_for_production:
            if not promotion:
                failures.append("production label without promotion_manifest.json")
            elif not promotion.get("verified"):
                failures.append("production label without verified promotion_manifest")
            else:
                # Recompute rewards; do not trust stored booleans.
                model_config = (
                    self.context.read_yaml("model.yaml")
                    if (self.context.config_dir / "model.yaml").exists()
                    else {}
                )
                card = RewardV2Evaluator(
                    rewards_config=self.rewards_config,
                    model_config=model_config,
                ).evaluate_run_dir(run_dir, run_id=run_id, profile=profile, publication_mode=mode)
                if card.get("blocking_rewards"):
                    failures.append(
                        "production promotion blocked by rewards: "
                        + ", ".join(card["blocking_rewards"])
                    )
                stored_hash = promotion.get("reward_card_hash")
                recomputed_hash = _reward_states_hash(card.get("rewards", {}))
                if stored_hash and stored_hash != recomputed_hash:
                    failures.append("promotion reward_card_hash does not match recomputed rewards")

        # Basic semantic range checks on probabilities when present.
        forecasts_path = run_dir / "race_forecasts.parquet"
        if forecasts_path.exists():
            import polars as pl

            frame = pl.read_parquet(forecasts_path)
            if "winner_probability" in frame.columns:
                bad = frame.filter(
                    pl.col("winner_probability").is_not_null()
                    & ((pl.col("winner_probability") < 0.0) | (pl.col("winner_probability") > 1.0))
                ).height
                if bad:
                    failures.append(f"{bad} probabilities outside [0, 1]")

        passed = not failures
        payload = {
            "run_id": run_id,
            "profile": profile,
            "publication_mode": mode,
            "passed": passed,
            "failure_reasons": failures,
            "reconciliation_ok": passed,
            "generated_at": datetime.now(UTC).isoformat(),
            "promoted_pointer_unchanged": True,
        }
        write_json(payload, run_dir / "semantic_verification.json")
        return payload

    def attempt_promote(
        self,
        *,
        attempt_id: str,
        profile: str = "production",
        profile_id: str | None = None,
    ) -> dict[str, Any]:
        """Atomically promote only when every required reward recomputes to pass."""
        attempt_dir = self._resolve_run_dir(attempt_id)
        model_config = (
            self.context.read_yaml("model.yaml")
            if (self.context.config_dir / "model.yaml").exists()
            else {}
        )
        card = RewardV2Evaluator(
            rewards_config=self.rewards_config,
            model_config=model_config,
        ).evaluate_run_dir(
            attempt_dir,
            run_id=attempt_id,
            profile=profile,
            publication_mode="production",
        )
        write_json(card, attempt_dir / "reward_card_v2.json")

        blocking = list(card.get("blocking_rewards") or [])
        promoted_root = self.context.artifacts_dir / "promoted" / (profile_id or profile)
        previous = None
        promoted_path = promoted_root / "promotion_manifest.json"
        if promoted_path.exists():
            previous = promoted_path.read_bytes()

        if blocking:
            decision = {
                "attempt_id": attempt_id,
                "profile": profile,
                "publication_mode": "research",
                "allowed": False,
                "blocks_publication": True,
                "blocking_rewards": blocking,
                "reason": f"Promotion refused: {', '.join(blocking)}",
                "generated_at": datetime.now(UTC).isoformat(),
            }
            write_json(decision, attempt_dir / "publication_decision.json")
            # Ensure promoted pointer unchanged.
            if previous is not None and promoted_path.exists():
                assert promoted_path.read_bytes() == previous
            return {
                "promoted": False,
                "attempt_id": attempt_id,
                "blocking_rewards": blocking,
                "promoted_pointer_unchanged": True,
                "decision": decision,
            }

        reward_hash = _reward_states_hash(card.get("rewards", {}))
        semantic = self.verify_semantic(
            run_id=attempt_id,
            profile=profile,
            require_promotion_for_production=False,
        )
        if not semantic.get("passed"):
            return {
                "promoted": False,
                "attempt_id": attempt_id,
                "blocking_rewards": ["semantic_verification"],
                "promoted_pointer_unchanged": True,
                "semantic": semantic,
            }

        manifest = {
            "attempt_id": attempt_id,
            "profile": profile,
            "publication_mode": "production",
            "promoted_at": datetime.now(UTC).isoformat(),
            "reward_card_hash": reward_hash,
            "semantic_verification_hash": hashlib.sha256(
                json.dumps(semantic, sort_keys=True, default=str).encode()
            ).hexdigest(),
            "content_hashes": {
                "reward_card_v2": reward_hash,
            },
            "verified": True,
            "blocking_rewards": [],
        }
        # Atomic write: temp then rename.
        promoted_root.mkdir(parents=True, exist_ok=True)
        tmp = promoted_root / f".promotion_manifest.{attempt_id}.tmp"
        write_json(manifest, tmp)
        tmp.replace(promoted_path)
        write_json(manifest, attempt_dir / "promotion_manifest.json")
        write_json(
            {
                "attempt_id": attempt_id,
                "profile": profile,
                "publication_mode": "production",
                "allowed": True,
                "blocks_publication": False,
                "blocking_rewards": [],
                "reason": "All profile-required rewards pass",
                "generated_at": datetime.now(UTC).isoformat(),
            },
            attempt_dir / "publication_decision.json",
        )
        return {
            "promoted": True,
            "attempt_id": attempt_id,
            "promotion_manifest": str(promoted_path),
            "manifest": manifest,
        }

    def reject_relabel_without_manifest(self, run_dir: Path) -> dict[str, Any]:
        """If a research/fixture run is relabeled production without promotion, fail."""
        run_dir = Path(run_dir)
        decision_path = run_dir / "publication_decision.json"
        if decision_path.exists():
            decision = read_json(decision_path)
        else:
            decision = {"publication_mode": "research"}
        # Mutate label only (adversarial path under test).
        decision = dict(decision)
        decision["publication_mode"] = "production"
        write_json(decision, decision_path)
        result = self.verify_semantic(run_id=run_dir.name, profile="production")
        return result

    def _resolve_run_dir(self, run_id: str) -> Path:
        for candidate in (
            self.context.artifacts_dir / "forecasts" / run_id,
            self.context.artifacts_dir / "attempts" / run_id,
            self.context.artifacts_dir / run_id,
        ):
            if candidate.exists():
                return candidate
        # Create under attempts for pure unit tests that pass absolute paths via run_id.
        path = Path(run_id)
        if path.exists():
            return path
        raise FileNotFoundError(f"Run directory not found for run_id={run_id}")

    @staticmethod
    def _read_optional(path: Path) -> dict[str, Any] | None:
        if not path.exists():
            return None
        return read_json(path)


def _reward_states_hash(rewards: dict[str, Any]) -> str:
    """Hash reward states only so volatile metrics do not invalidate promotion."""
    states = {
        key: (value.get("state") if isinstance(value, dict) else value)
        for key, value in sorted(rewards.items())
    }
    return hashlib.sha256(json.dumps(states, sort_keys=True, default=str).encode()).hexdigest()


def copy_promoted_snapshot(source: Path, dest: Path) -> None:
    """Helper for tests: deep-copy a promoted tree without following specials."""
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(source, dest)
