"""Load the single reward-v2 threshold and profile registry from config."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

REWARD_STATES = frozenset({"pass", "fail", "insufficient_evidence", "not_applicable"})
PUBLICATION_MODES = frozenset({"research", "shadow", "production", "fixture"})

DEFAULT_REWARDS_CONFIG = "rewards.yaml"


def _default_config_path() -> Path:
    # src/civic_signal/scoring/reward_registry.py -> repo root / configs
    return Path(__file__).resolve().parents[3] / "configs" / DEFAULT_REWARDS_CONFIG


@lru_cache(maxsize=8)
def load_rewards_config(path: str | None = None) -> dict[str, Any]:
    """Load rewards.yaml. Cached by path string for deterministic unit tests."""
    config_path = Path(path) if path else _default_config_path()
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{config_path} must contain a mapping")
    if "thresholds" not in data or "profiles" not in data or "reward_ids" not in data:
        raise ValueError(f"{config_path} missing required keys thresholds/profiles/reward_ids")
    return data


def clear_rewards_config_cache() -> None:
    load_rewards_config.cache_clear()


def threshold_for(reward_id: str, config: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = config if config is not None else load_rewards_config()
    return dict(cfg.get("thresholds", {}).get(reward_id, {}))


def profile_required_rewards(profile: str, config: dict[str, Any] | None = None) -> list[str]:
    cfg = config if config is not None else load_rewards_config()
    profiles = cfg.get("profiles", {})
    if profile not in profiles:
        raise KeyError(f"Unknown reward profile: {profile}")
    return list(profiles[profile].get("required_rewards", []))


def publication_mode_default(config: dict[str, Any] | None = None) -> str:
    cfg = config if config is not None else load_rewards_config()
    mode = str(cfg.get("publication_mode_default", "research"))
    if mode not in PUBLICATION_MODES:
        raise ValueError(f"Invalid publication_mode_default: {mode}")
    return mode


def all_reward_ids(config: dict[str, Any] | None = None) -> list[str]:
    cfg = config if config is not None else load_rewards_config()
    return list(cfg.get("reward_ids", []))


def make_reward_record(
    *,
    reward_id: str,
    state: str,
    metric: dict[str, Any] | None = None,
    threshold: dict[str, Any] | None = None,
    evidence: list[str] | None = None,
    lineage: dict[str, Any] | None = None,
    negative_tests_passed: list[str] | None = None,
    failure_reasons: list[str] | None = None,
    blocks_publication: bool = True,
    scope: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if state not in REWARD_STATES:
        raise ValueError(f"Invalid reward state {state!r} for {reward_id}")
    return {
        "reward_id": reward_id,
        "state": state,
        "scope": scope or {"offices": [], "election_types": [], "horizon_buckets": []},
        "metric": metric or {},
        "threshold": threshold or {},
        "evidence": list(evidence or []),
        "lineage": lineage or {},
        "negative_tests_passed": list(negative_tests_passed or []),
        "failure_reasons": list(failure_reasons or []),
        "blocks_publication": bool(blocks_publication),
    }
