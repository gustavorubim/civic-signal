"""Generate R27 contract-parity evidence from config/docs/CLI claims."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from civic_signal.storage.io import write_json

REQUIRED_CLI_SURFACES = (
    "backtest nested",
    "data audit",
    "shadow run",
    "verify as-of",
    "verify coherence",
    "verify publication",
    "verify recovery",
    "verify rewards",
    "verify scientific",
    "verify shadow",
)

CANONICAL_REWARD_IDS = (
    "R0_build",
    "R1_reproducibility",
    "R2_provenance",
    "R3_sync_integrity",
    "R4_calibration",
    "R5_baseline_competition",
    "R6_component_admission",
    "R7_sparse_honesty",
    "R8_uncertainty_quality",
    "R9_public_signal_discipline",
    "R10_explainability",
    "R11_plot_contract",
    "R12_performance_contract",
    "R13_posterior_quality",
    "R14_calibrated_publication",
    "R15_daily_update_quality",
    "R16_real_data_exclusivity",
    "R17_as_of_integrity",
    "R18_nested_evaluation",
    "R19_covariance_recovery",
    "R20_all_race_hierarchy",
    "R21_poll_observation_identity",
    "R22_feature_validity",
    "R23_joint_outcome_coherence",
    "R24_atomic_publication",
    "R25_live_source_resilience",
    "R26_benchmark_superiority",
    "R27_contract_parity",
)


class ContractParityChecker:
    """Find stale claims between rewards config, docs, and active code contracts."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def run(self, *, output_path: Path | None = None) -> dict[str, Any]:
        rewards_path = self.root / "configs" / "rewards.yaml"
        readme = self.root / "README.md"
        spec = self.root / "SPEC.md"
        checked = []
        stale_claims: list[str] = []
        failure_reasons: list[str] = []
        reward_ids: list[str] = []
        thresholds: dict[str, Any] = {}
        profile_reward_ids: set[str] = set()
        generated_assertions: dict[str, Any] = {}

        if rewards_path.exists():
            checked.append("configs/rewards.yaml")
            text = rewards_path.read_text(encoding="utf-8")
            rewards = yaml.safe_load(text) or {}
            if rewards.get("publication_mode_default") != "research":
                stale_claims.append("rewards.yaml missing publication_mode_default: research")
            reward_ids = [str(value) for value in rewards.get("reward_ids") or []]
            thresholds = dict(rewards.get("thresholds") or {})
            missing_canonical_rewards = sorted(set(CANONICAL_REWARD_IDS) - set(reward_ids))
            unexpected_reward_ids = sorted(set(reward_ids) - set(CANONICAL_REWARD_IDS))
            for reward_id in missing_canonical_rewards:
                stale_claims.append(f"rewards.yaml missing {reward_id}")
            if unexpected_reward_ids:
                stale_claims.append(
                    f"rewards.yaml contains noncanonical reward IDs: {unexpected_reward_ids}"
                )
            for profile in dict(rewards.get("profiles") or {}).values():
                profile_reward_ids.update(
                    str(value) for value in profile.get("required_rewards") or []
                )
                profile_reward_ids.update(
                    str(value) for value in dict(profile.get("conditional_rewards") or {})
                )
            if len(reward_ids) != len(set(reward_ids)):
                stale_claims.append("rewards.yaml reward_ids contains duplicates")
            missing_thresholds = sorted(set(reward_ids) - set(thresholds))
            extra_thresholds = sorted(set(thresholds) - set(reward_ids))
            unknown_profile_rewards = sorted(profile_reward_ids - set(reward_ids))
            empty_thresholds = sorted(
                reward_id
                for reward_id, threshold in thresholds.items()
                if not isinstance(threshold, dict) or not threshold
            )
            if missing_thresholds:
                stale_claims.append(f"reward IDs missing thresholds: {missing_thresholds}")
            if extra_thresholds:
                stale_claims.append(f"thresholds reference unknown reward IDs: {extra_thresholds}")
            if unknown_profile_rewards:
                stale_claims.append(
                    f"profiles reference unknown reward IDs: {unknown_profile_rewards}"
                )
            if empty_thresholds:
                stale_claims.append(f"reward thresholds are empty: {empty_thresholds}")
            generated_assertions["reward_registry"] = {
                "reward_ids": reward_ids,
                "threshold_ids": sorted(thresholds),
                "profile_reward_ids": sorted(profile_reward_ids),
                "missing_thresholds": missing_thresholds,
                "extra_thresholds": extra_thresholds,
                "unknown_profile_rewards": unknown_profile_rewards,
                "empty_thresholds": empty_thresholds,
                "canonical_reward_ids": list(CANONICAL_REWARD_IDS),
                "missing_canonical_rewards": missing_canonical_rewards,
                "unexpected_reward_ids": unexpected_reward_ids,
            }
        else:
            failure_reasons.append("configs/rewards.yaml missing")

        documented_reward_ids: set[str] = set()
        documented_cli_text = ""
        for label, path in (("README.md", readme), ("SPEC.md", spec)):
            if not path.exists():
                failure_reasons.append(f"{label} missing")
                continue
            checked.append(label)
            body = path.read_text(encoding="utf-8")
            body_lower = body.lower()
            documented_cli_text += f"\n{body_lower}"
            documented_reward_ids.update(
                re.findall(r"R(?:[0-9]|1[0-9]|2[0-7])_[A-Za-z0-9_]+", body)
            )
            if "reward-v2" not in body_lower and "reward_v2" not in body_lower:
                stale_claims.append(f"{label} does not mention reward-v2")
            if "publication_mode" not in body_lower and "research" not in body_lower:
                stale_claims.append(f"{label} missing research/publication_mode language")
            # Production default language must remain qualified if present.
            if re.search(r"bayesian path is the production default", body_lower):
                if "research" not in body_lower and "reward-v2" not in body_lower:
                    stale_claims.append(
                        f"{label} claims production default without research/reward-v2 qualifier"
                    )
        missing_documented_rewards = sorted(set(reward_ids) - documented_reward_ids)
        if missing_documented_rewards:
            stale_claims.append(
                f"SPEC/README missing configured reward IDs: {missing_documented_rewards}"
            )

        # Generated implementation and CLI surface assertions.
        checked.append("src/civic_signal/cli.py")
        cli_path = self.root / "src" / "civic_signal" / "cli.py"
        evaluator_path = self.root / "src" / "civic_signal" / "scoring" / "reward_v2.py"
        evaluator_reward_ids: set[str] = set()
        if evaluator_path.exists():
            checked.append("src/civic_signal/scoring/reward_v2.py")
            evaluator_reward_ids = set(
                re.findall(
                    r"def _eval_(R(?:[0-9]|1[0-9]|2[0-7])_[A-Za-z0-9_]+)",
                    evaluator_path.read_text(encoding="utf-8"),
                )
            )
            missing_evaluators = sorted(set(reward_ids) - evaluator_reward_ids)
            extra_evaluators = sorted(evaluator_reward_ids - set(reward_ids))
            if missing_evaluators:
                stale_claims.append(f"reward IDs missing evaluator methods: {missing_evaluators}")
            if extra_evaluators:
                stale_claims.append(f"evaluator methods missing reward IDs: {extra_evaluators}")
        else:
            failure_reasons.append("src/civic_signal/scoring/reward_v2.py missing")

        actual_cli_surfaces: set[str] = set()
        if cli_path.exists():
            cli_text = cli_path.read_text(encoding="utf-8")
            prefixes = {
                "backtest_app": "backtest",
                "data_app": "data",
                "shadow_app": "shadow",
                "verify_app": "verify",
            }
            for group, command in re.findall(
                r'@(backtest_app|data_app|shadow_app|verify_app)\.command\("([^"]+)"\)',
                cli_text,
            ):
                actual_cli_surfaces.add(f"{prefixes[group]} {command}")
            missing_cli = sorted(set(REQUIRED_CLI_SURFACES) - actual_cli_surfaces)
            undocumented_cli = sorted(
                surface
                for surface in REQUIRED_CLI_SURFACES
                if surface.lower() not in documented_cli_text
            )
            if missing_cli:
                stale_claims.append(f"cli.py missing expected surfaces: {missing_cli}")
            if undocumented_cli:
                stale_claims.append(f"SPEC/README missing CLI surfaces: {undocumented_cli}")
        else:
            failure_reasons.append("cli.py missing")

        generated_assertions.update(
            {
                "documented_reward_ids": sorted(documented_reward_ids),
                "evaluator_reward_ids": sorted(evaluator_reward_ids),
                "required_cli_surfaces": list(REQUIRED_CLI_SURFACES),
                "actual_cli_surfaces": sorted(actual_cli_surfaces),
            }
        )

        passed = not stale_claims and not failure_reasons
        payload = {
            "schema_version": "1.0.0",
            "generated_at": datetime.now(UTC).isoformat(),
            "passed": passed,
            "stale_claims": len(stale_claims),
            "stale_claim_details": stale_claims,
            "failure_reasons": failure_reasons or ([] if passed else stale_claims),
            "checked_documents": checked,
            "generated_assertions": generated_assertions,
        }
        if output_path is not None:
            write_json(payload, output_path)
        return payload
