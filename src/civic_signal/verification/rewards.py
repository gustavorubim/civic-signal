"""Recompute reward-v2 profiles and hard-block publication on fail/insufficient."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from civic_signal.config import ProjectContext
from civic_signal.scoring.reward_registry import load_rewards_config, publication_mode_default
from civic_signal.scoring.reward_v2 import RewardV2Evaluator
from civic_signal.storage.io import write_json, write_text


class RewardVerificationRunner:
    """Run `verify rewards --profile` against primary artifacts only."""

    def __init__(self, context: ProjectContext) -> None:
        self.context = context
        self.rewards_config = context.read_yaml("rewards.yaml")

    def verify(
        self,
        *,
        run_id: str,
        profile: str = "production",
        publication_mode: str | None = None,
    ) -> dict[str, Any]:
        run_dir = self.context.artifacts_dir / "forecasts" / run_id
        if not run_dir.exists():
            # Also accept attempts and bare artifact roots used in tests.
            for candidate in (
                self.context.artifacts_dir / "attempts" / run_id,
                self.context.artifacts_dir / run_id,
                self.context.artifacts_dir / "forecasts" / run_id,
            ):
                if candidate.exists():
                    run_dir = candidate
                    break
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory not found for run_id={run_id}")

        model_config: dict[str, Any] = {}
        model_path = self.context.config_dir / "model.yaml"
        if model_path.exists():
            model_config = self.context.read_yaml("model.yaml")

        evaluator = RewardV2Evaluator(
            rewards_config=self.rewards_config,
            model_config=model_config,
        )
        card = evaluator.evaluate_run_dir(
            run_dir,
            run_id=run_id,
            profile=profile,
            publication_mode=publication_mode,
        )

        # Never trust a previously written boolean: always recompute.
        write_json(card, run_dir / "reward_card_v2.json")

        decision = self._publication_decision(card, run_id=run_id, profile=profile)
        write_json(decision, run_dir / "publication_decision.json")

        report = self._human_report(card, decision)
        write_text(report, run_dir / "reward_verification_report.md")

        input_hashes = {
            "rewards_config": self._file_hash(self.context.config_dir / "rewards.yaml"),
            "run_dir": str(run_dir),
        }
        payload = {
            "run_id": run_id,
            "profile": profile,
            "passed": not card["blocks_publication"] and not card["blocking_rewards"],
            "blocks_publication": card["blocks_publication"],
            "blocking_rewards": card["blocking_rewards"],
            "publication_mode": card["publication_mode"],
            "reward_card_path": str(run_dir / "reward_card_v2.json"),
            "publication_decision_path": str(run_dir / "publication_decision.json"),
            "report_path": str(run_dir / "reward_verification_report.md"),
            "input_hashes": input_hashes,
            "generated_at": datetime.now(UTC).isoformat(),
            "exit_nonzero": bool(card["blocking_rewards"]),
        }
        write_json(payload, run_dir / "reward_verification.json")
        return payload

    def _publication_decision(
        self, card: dict[str, Any], *, run_id: str, profile: str
    ) -> dict[str, Any]:
        mode = card.get("publication_mode") or publication_mode_default(self.rewards_config)
        blocking = list(card.get("blocking_rewards") or [])
        allowed = mode != "production" or not blocking
        if profile == "production" and blocking:
            allowed = False
            mode = mode if mode != "production" else "research"
        return {
            "run_id": run_id,
            "profile": profile,
            "publication_mode": card.get("publication_mode") or mode,
            "allowed": allowed and not (profile == "production" and blocking),
            "blocks_publication": bool(blocking) if profile == "production" else False,
            "blocking_rewards": blocking,
            "reason": (
                "All profile-required rewards pass"
                if not blocking
                else f"Blocking rewards: {', '.join(blocking)}"
            ),
            "generated_at": datetime.now(UTC).isoformat(),
            "reward_card_hash": hashlib.sha256(
                json.dumps(
                    {
                        key: (value.get("state") if isinstance(value, dict) else value)
                        for key, value in sorted((card.get("rewards") or {}).items())
                    },
                    sort_keys=True,
                    default=str,
                ).encode()
            ).hexdigest(),
        }

    @staticmethod
    def _human_report(card: dict[str, Any], decision: dict[str, Any]) -> str:
        lines = [
            f"# Reward verification — {card.get('run_id')}",
            "",
            f"- Profile: `{card.get('profile')}`",
            f"- Publication mode: `{card.get('publication_mode')}`",
            f"- Recomputed: `{card.get('recomputed')}`",
            f"- Blocks publication: `{card.get('blocks_publication')}`",
            f"- Decision allowed: `{decision.get('allowed')}`",
            "",
            "## Blocking rewards",
            "",
        ]
        blocking = card.get("blocking_rewards") or []
        if not blocking:
            lines.append("_None_")
        else:
            for reward_id in blocking:
                record = card.get("rewards", {}).get(reward_id, {})
                lines.append(
                    f"- `{reward_id}`: **{record.get('state')}** — "
                    f"{'; '.join(record.get('failure_reasons') or [])}"
                )
        lines.extend(["", "## All rewards", ""])
        for reward_id, record in sorted((card.get("rewards") or {}).items()):
            if not isinstance(record, dict):
                continue
            lines.append(
                f"- `{reward_id}`: `{record.get('state')}` "
                f"({'; '.join(record.get('failure_reasons') or ['ok'])})"
            )
        lines.append("")
        return "\n".join(lines)

    @staticmethod
    def _file_hash(path: Path) -> str | None:
        if not path.exists():
            return None
        digest = hashlib.sha256()
        digest.update(path.read_bytes())
        return digest.hexdigest()


def default_publication_mode(context: ProjectContext | None = None) -> str:
    if context is not None and (context.config_dir / "rewards.yaml").exists():
        return publication_mode_default(context.read_yaml("rewards.yaml"))
    return publication_mode_default(load_rewards_config())
