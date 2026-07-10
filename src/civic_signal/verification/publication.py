"""Atomic publication checks and production-label semantic verification."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from civic_signal.config import ProjectContext
from civic_signal.scoring.reward_v2 import RewardV2Evaluator
from civic_signal.storage.io import read_json, write_json
from civic_signal.verification.schema import artifact_schema_errors

# Primary scientific artifacts that must be content-hashed on promotion.
_PROMOTED_ARTIFACTS = (
    "race_forecasts.parquet",
    "race_catalog.parquet",
    "forecast_draws.parquet",
    "control_forecasts.parquet",
    "source_manifest.parquet",
    "reward_card_v2.json",
    "run_manifest.json",
    "performance.json",
    "reproducibility_fingerprint.json",
    "semantic_verification.json",
    "ci_manifest.json",
    "coverage.json",
    "backtest_summary.json",
    "nested_evaluation.json",
    "covariance_recovery.json",
    "hierarchy_recovery.json",
    "poll_observation_manifest.json",
    "feature_lineage.json",
    "live_source_canaries.json",
    "benchmark_superiority.json",
    "contract_parity.json",
    "as_of_audit.json",
    "selected_feature_lineage.parquet",
    "latest_daily_update.json",
    "posterior_diagnostics.json",
    "plot_manifest.json",
)


def resolve_run_dir(artifacts_dir: Path, run_id: str) -> Path:
    """Locate a run directory under the normal pipeline and attempt layouts."""
    for candidate in (
        artifacts_dir / "runs" / run_id,
        artifacts_dir / "forecasts" / run_id,
        artifacts_dir / "attempts" / run_id,
        artifacts_dir / run_id,
    ):
        if candidate.exists():
            return candidate
    path = Path(run_id)
    if path.exists():
        return path
    raise FileNotFoundError(f"Run directory not found for run_id={run_id}")  # pragma: no cover


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
        force_publication_mode: str | None = None,
        write_artifact: bool = False,
    ) -> dict[str, Any]:
        run_dir = self._resolve_run_dir(run_id)
        failures: list[str] = []
        checks: dict[str, Any] = {}

        decision = self._read_optional(run_dir / "publication_decision.json")
        promotion = self._read_optional(run_dir / "promotion_manifest.json")
        run_manifest = self._read_optional(run_dir / "run_manifest.json")
        mode = str(
            force_publication_mode
            or (decision or {}).get("publication_mode")
            or (run_manifest or {}).get("publication_mode")
            or (promotion or {}).get("publication_mode")
            or "research"
        )

        if mode == "production" and require_promotion_for_production:
            if not promotion:
                failures.append("production label without promotion_manifest.json")
            elif artifact_schema_errors(
                self.context.root, promotion, "promotion_manifest.schema.json"
            ):
                failures.append("production promotion_manifest fails runtime JSON Schema")
            elif not _promotion_manifest_is_complete(promotion):
                failures.append("production label without verified promotion_manifest")
            else:
                model_config = (
                    self.context.read_yaml("model.yaml")
                    if (self.context.config_dir / "model.yaml").exists()
                    else {}
                )
                card = RewardV2Evaluator(
                    rewards_config=self.rewards_config,
                    model_config=model_config,
                ).evaluate_run_dir(
                    run_dir, run_id=run_id, profile=profile, publication_mode="production"
                )
                reward_schema_errors = artifact_schema_errors(
                    self.context.root, card, "reward_card_v2.schema.json"
                )
                if reward_schema_errors:
                    failures.append("recomputed reward_card_v2 fails runtime JSON Schema")
                if card.get("blocking_rewards"):
                    failures.append(
                        "production promotion blocked by rewards: "
                        + ", ".join(card["blocking_rewards"])
                    )
                stored_hash = promotion.get("reward_card_hash")
                recomputed_hash = _reward_card_integrity_hash(card)
                if stored_hash != recomputed_hash:
                    failures.append(
                        "promotion reward_card_hash does not match recomputed reward integrity"
                    )
                semantic_artifact = self._read_optional(run_dir / "semantic_verification.json")
                if semantic_artifact is None or promotion.get(
                    "semantic_verification_hash"
                ) != _semantic_integrity_hash(semantic_artifact):
                    failures.append("promotion semantic_verification_hash does not match artifact")
                current_config_hash = _file_hash(self.context.config_dir / "rewards.yaml")
                if promotion.get("rewards_config_hash") != current_config_hash:
                    failures.append("promotion rewards_config_hash does not match current config")
                stored_content = dict(promotion.get("content_hashes") or {})
                current_content = _content_hashes(run_dir)
                if sorted(stored_content) != sorted(current_content):
                    failures.append(
                        "promotion content_hashes do not cover the complete artifact set"
                    )
                for key, digest in stored_content.items():
                    if current_content.get(key) != digest:
                        failures.append(f"promoted content hash mismatch for {key}")

        # Full semantic reconciliation against primary forecast artifacts.
        recon = self._reconcile_forecast(run_dir)
        checks.update(recon["checks"])
        failures.extend(recon["failures"])

        passed = not failures
        payload = {
            "run_id": run_id,
            "profile": profile,
            "publication_mode": mode,
            "passed": passed,
            "failure_reasons": failures,
            "reconciliation_ok": passed and bool(checks.get("required_artifacts_present")),
            "checks": checks,
            "generated_at": datetime.now(UTC).isoformat(),
            "promoted_pointer_unchanged": True,
        }
        if write_artifact:
            write_json(payload, run_dir / "semantic_verification.json")
        return payload

    def _reconcile_forecast(self, run_dir: Path) -> dict[str, Any]:
        import polars as pl

        failures: list[str] = []
        checks: dict[str, Any] = {
            "required_artifacts_present": False,
            "unique_race_keys": None,
            "probability_range_ok": None,
            "simplex_ok": None,
            "draws_cover_races": None,
            "control_present": None,
            "source_manifest_present": None,
            "catalog_forecast_coverage_ok": None,
            "non_sparse_probabilities_present": None,
            "estimand_support_withholding_ok": None,
            "control_probability_range_ok": None,
        }

        required = {
            "race_forecasts.parquet",
            "race_catalog.parquet",
            "forecast_draws.parquet",
            "control_forecasts.parquet",
            "source_manifest.parquet",
        }
        missing = sorted(name for name in required if not (run_dir / name).exists())
        checks["missing_required_artifacts"] = missing
        if missing:
            failures.append(f"Missing required forecast artifacts: {missing}")
            return {"failures": failures, "checks": checks}
        checks["required_artifacts_present"] = True

        forecasts = pl.read_parquet(run_dir / "race_forecasts.parquet")
        catalog = pl.read_parquet(run_dir / "race_catalog.parquet")
        draws = pl.read_parquet(run_dir / "forecast_draws.parquet")
        control = pl.read_parquet(run_dir / "control_forecasts.parquet")
        manifest = pl.read_parquet(run_dir / "source_manifest.parquet")
        checks["source_manifest_present"] = not manifest.is_empty()
        if manifest.is_empty():
            failures.append("source_manifest is empty")
        else:
            required_manifest = {"source_id", "status", "content_hash"}
            missing_manifest = sorted(required_manifest - set(manifest.columns))
            if missing_manifest:
                failures.append(f"source_manifest missing columns: {missing_manifest}")
            else:
                for column in sorted(required_manifest):
                    if manifest.filter(
                        pl.col(column).is_null() | (pl.col(column).cast(pl.String) == "")
                    ).height:
                        failures.append(f"source_manifest contains blank {column}")

        if "race_id" not in forecasts.columns:
            failures.append("race_forecasts missing race_id")
            return {"failures": failures, "checks": checks}
        if "race_id" not in catalog.columns:
            failures.append("race_catalog missing race_id")
            return {"failures": failures, "checks": checks}

        catalog_ids = set(catalog["race_id"].drop_nulls().unique().to_list())
        forecast_ids = set(forecasts["race_id"].drop_nulls().unique().to_list())
        catalog_duplicate_count = catalog.height - catalog.select("race_id").n_unique()
        if catalog_duplicate_count:
            failures.append(f"race_catalog contains {catalog_duplicate_count} duplicate race IDs")
        missing_forecasts = sorted(catalog_ids - forecast_ids)
        extra_forecasts = sorted(forecast_ids - catalog_ids)
        checks["catalog_forecast_coverage_ok"] = not missing_forecasts and not extra_forecasts
        if missing_forecasts:
            failures.append(f"Forecasts missing catalog races: {missing_forecasts[:10]}")
        if extra_forecasts:
            failures.append(f"Forecasts contain races absent from catalog: {extra_forecasts[:10]}")

        # Unique race/option identities: no duplicate option rows per race.
        if "option_id" in forecasts.columns:
            dup = (
                forecasts.group_by(["race_id", "option_id"])
                .agg(pl.len().alias("n"))
                .filter(pl.col("n") > 1)
            )
            checks["unique_race_keys"] = dup.is_empty()
            if not dup.is_empty():
                failures.append(f"Duplicate race/option rows: {dup.height}")
        else:
            # Without option_id, race_id rows should still be unique if one probability each.
            dup = forecasts.group_by("race_id").agg(pl.len().alias("n")).filter(pl.col("n") > 1)
            # Multi-row per race is allowed only with distinct parties if present.
            if "party" in forecasts.columns:
                dup = (
                    forecasts.group_by(["race_id", "party"])
                    .agg(pl.len().alias("n"))
                    .filter(pl.col("n") > 1)
                )
            checks["unique_race_keys"] = dup.is_empty()
            if not dup.is_empty():
                failures.append(f"Duplicate race identity rows: {dup.height}")

        if "winner_probability" in forecasts.columns:
            published = forecasts.filter(pl.col("winner_probability").is_not_null())
            bad = published.filter(
                pl.col("winner_probability").is_nan()
                | ~pl.col("winner_probability").is_finite()
                | (pl.col("winner_probability") < 0.0)
                | (pl.col("winner_probability") > 1.0)
            ).height
            checks["probability_range_ok"] = bad == 0
            if bad:
                failures.append(f"{bad} probabilities outside [0, 1]")
            tier_c = (
                set(catalog.filter(pl.col("tier") == "C")["race_id"].to_list())
                if "tier" in catalog.columns
                else set()
            )
            non_sparse = catalog_ids - tier_c
            non_sparse_with_probability = set(published["race_id"].unique().to_list())
            missing_probability = sorted(non_sparse - non_sparse_with_probability)
            checks["non_sparse_probabilities_present"] = not missing_probability
            if missing_probability:
                failures.append(
                    "Non-sparse catalog races missing winner_probability: "
                    f"{missing_probability[:10]}"
                )
            if tier_c:
                sparse_values = forecasts.filter(
                    pl.col("race_id").is_in(sorted(tier_c))
                    & pl.col("winner_probability").is_not_null()
                ).height
                if sparse_values:
                    failures.append("Tier C forecasts must withhold winner_probability")
            if "estimand_support_blocked" in catalog.columns:
                blocked_catalog = catalog.filter(
                    pl.col("estimand_support_blocked").fill_null(False)
                )
                blocked_ids = set(blocked_catalog["race_id"].to_list())
                blocked_published = forecasts.filter(
                    pl.col("race_id").is_in(sorted(blocked_ids))
                    & pl.col("winner_probability").is_not_null()
                ).height
                blocked_not_tier_c = (
                    blocked_catalog.filter(pl.col("tier") != "C").height
                    if "tier" in blocked_catalog.columns
                    else blocked_catalog.height
                )
                blank_status = (
                    blocked_catalog.filter(
                        pl.col("estimand_support_status").is_null()
                        | (pl.col("estimand_support_status") == "")
                    ).height
                    if "estimand_support_status" in blocked_catalog.columns
                    else blocked_catalog.height
                )
                checks["estimand_support_withholding_ok"] = not (
                    blocked_published or blocked_not_tier_c or blank_status
                )
                if blocked_published:
                    failures.append(
                        f"{blocked_published} unsupported-estimand forecast rows publish "
                        "winner_probability"
                    )
                if blocked_not_tier_c:
                    failures.append(
                        f"{blocked_not_tier_c} unsupported-estimand races are not Tier C"
                    )
                if blank_status:
                    failures.append(
                        f"{blank_status} unsupported-estimand races lack a blocking status"
                    )
            else:
                checks["estimand_support_withholding_ok"] = True
            multi = published.group_by("race_id").agg(pl.len().alias("n")).filter(pl.col("n") > 1)
            if multi.height:
                sums = published.group_by("race_id").agg(
                    pl.col("winner_probability").sum().alias("s")
                )
                bad_sums = sums.join(multi, on="race_id", how="inner").filter(
                    (pl.col("s") < 0.99) | (pl.col("s") > 1.01)
                )
                checks["simplex_ok"] = bad_sums.is_empty()
                if not bad_sums.is_empty():
                    failures.append(f"{bad_sums.height} races fail probability simplex sum")
            else:
                checks["simplex_ok"] = True
        else:
            failures.append("race_forecasts missing winner_probability")
            checks["probability_range_ok"] = False

        # Interval ordering when present.
        lo = "share_p10" if "share_p10" in forecasts.columns else None
        mid = "share_p50" if "share_p50" in forecasts.columns else None
        hi = "share_p90" if "share_p90" in forecasts.columns else None
        if lo and mid and hi:
            bad_iv = forecasts.filter(
                (pl.col(lo) > pl.col(mid)) | (pl.col(mid) > pl.col(hi))
            ).height
            checks["interval_ordering_ok"] = bad_iv == 0
            if bad_iv:
                failures.append(f"{bad_iv} rows violate interval ordering")

        # Draw completeness for non-Tier-C races (or all if no tier).
        race_ids = catalog_ids
        if "tier" in catalog.columns and "race_id" in catalog.columns:
            tier_c = set(catalog.filter(pl.col("tier") == "C")["race_id"].to_list())
            required_draws = race_ids - tier_c
        else:
            required_draws = race_ids
        if "race_id" in draws.columns:
            draw_races = set(draws["race_id"].unique().to_list())
            missing_draws = sorted(required_draws - draw_races)
            checks["draws_cover_races"] = not missing_draws
            if missing_draws:
                failures.append(f"Draws missing for races: {missing_draws[:10]}")
            if "draw_id" not in draws.columns:
                failures.append("forecast_draws missing draw_id")
            elif "option_id" in draws.columns:
                duplicate_options = (
                    draws.group_by(["race_id", "draw_id", "option_id"])
                    .agg(pl.len().alias("n"))
                    .filter(pl.col("n") > 1)
                )
                if not duplicate_options.is_empty():
                    failures.append(f"Duplicate draw race/option keys: {duplicate_options.height}")
            if "winner" in draws.columns:
                winner_counts = draws.group_by(["race_id", "draw_id"]).agg(
                    pl.col("winner").cast(pl.Int8).sum().alias("winners")
                )
                bad_winner_counts = winner_counts.filter(pl.col("winners") != 1)
                if not bad_winner_counts.is_empty():
                    failures.append(
                        f"Draws must have exactly one winner: {bad_winner_counts.height}"
                    )
        else:
            failures.append("forecast_draws missing race_id")
            checks["draws_cover_races"] = False

        checks["control_present"] = not control.is_empty()
        if control.is_empty():
            failures.append("control_forecasts is empty")
        else:
            # Seat totals / majority probability sanity when columns exist.
            probability_columns = [
                column
                for column in ("control_probability", "majority_probability")
                if column in control.columns
            ]
            if not probability_columns:
                failures.append(
                    "control_forecasts missing control_probability or majority_probability"
                )
                checks["control_probability_range_ok"] = False
            else:
                bad_ctrl = control.filter(
                    pl.any_horizontal(
                        *[
                            pl.col(column).is_null()
                            | ~pl.col(column).is_finite()
                            | (pl.col(column) < 0.0)
                            | (pl.col(column) > 1.0)
                            for column in probability_columns
                        ]
                    )
                ).height
                checks["control_probability_range_ok"] = bad_ctrl == 0
                if bad_ctrl:
                    failures.append(f"{bad_ctrl} control probabilities out of range")

        # Lineage presence on forecasts.
        for col in ("model_config_hash", "source_manifest_hash"):
            if col not in forecasts.columns:
                failures.append(f"Forecast lineage column missing: {col}")
            elif forecasts.filter(pl.col(col).is_null() | (pl.col(col) == "")).height:
                failures.append(f"Forecast rows missing {col}")

        return {"failures": failures, "checks": checks}

    def attempt_promote(
        self,
        *,
        attempt_id: str,
        profile: str = "production",
        profile_id: str | None = None,
    ) -> dict[str, Any]:
        """Atomically promote only when every required reward recomputes to pass."""
        attempt_dir = self._resolve_run_dir(attempt_id)
        promoted_root = self.context.artifacts_dir / "promoted" / (profile_id or profile)
        promoted_path = promoted_root / "promotion_manifest.json"
        snapshot_dir = promoted_root / "attempts" / attempt_id
        previous = promoted_path.read_bytes() if promoted_path.exists() else None

        # Generate fresh semantic evidence before evaluating R23. This avoids
        # allowing a stale or fixture-seeded semantic_verification.json to
        # decide whether an attempt is promotable.
        semantic = self.verify_semantic(
            run_id=attempt_id,
            profile=profile,
            require_promotion_for_production=False,
            force_publication_mode="production",
            write_artifact=True,
        )
        if semantic.get("passed") is not True or semantic.get("reconciliation_ok") is not True:
            return {
                "promoted": False,
                "attempt_id": attempt_id,
                "blocking_rewards": ["semantic_verification"],
                "promoted_pointer_unchanged": True,
                "semantic": semantic,
            }

        model_config = (
            self.context.read_yaml("model.yaml")
            if (self.context.config_dir / "model.yaml").exists()
            else {}
        )
        # Always evaluate as production for promotion attempts.
        evaluator = RewardV2Evaluator(
            rewards_config=self.rewards_config,
            model_config=model_config,
        )
        evaluator._promotion_candidate = True
        card = evaluator.evaluate_run_dir(
            attempt_dir,
            run_id=attempt_id,
            profile=profile,
            publication_mode="production",
        )
        reward_schema_errors = artifact_schema_errors(
            self.context.root, card, "reward_card_v2.schema.json"
        )
        if reward_schema_errors:
            return {
                "promoted": False,
                "attempt_id": attempt_id,
                "blocking_rewards": ["reward_card_schema"],
                "promoted_pointer_unchanged": True,
                "schema_errors": reward_schema_errors,
            }
        write_json(card, attempt_dir / "reward_card_v2.json")

        blocking = list(card.get("blocking_rewards") or [])
        # Require every required reward to be pass (already encoded in blocking).
        non_pass = [
            rid
            for rid, rec in (card.get("rewards") or {}).items()
            if rid in set(profile_required(self.rewards_config, profile))
            and not (isinstance(rec, dict) and rec.get("state") == "pass")
        ]
        blocking = sorted(set(blocking) | set(non_pass))

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
            if previous is not None and promoted_path.exists():
                assert promoted_path.read_bytes() == previous
            return {
                "promoted": False,
                "attempt_id": attempt_id,
                "blocking_rewards": blocking,
                "promoted_pointer_unchanged": True,
                "decision": decision,
            }

        if snapshot_dir.exists():
            return {
                "promoted": False,
                "attempt_id": attempt_id,
                "blocking_rewards": ["immutable_attempt_collision"],
                "promoted_pointer_unchanged": True,
                "reason": f"Immutable snapshot already exists: {snapshot_dir}",
            }

        content_hashes = _content_hashes(attempt_dir)
        reward_hash = _reward_card_integrity_hash(card)
        semantic_hash = _semantic_integrity_hash(semantic)
        config_hash = _file_hash(self.context.config_dir / "rewards.yaml")
        if config_hash is None:
            return {
                "promoted": False,
                "attempt_id": attempt_id,
                "blocking_rewards": ["missing_rewards_config"],
                "promoted_pointer_unchanged": True,
            }

        manifest = {
            "attempt_id": attempt_id,
            "profile": profile,
            "publication_mode": "production",
            "promoted_at": datetime.now(UTC).isoformat(),
            "reward_card_hash": reward_hash,
            "semantic_verification_hash": semantic_hash,
            "rewards_config_hash": config_hash,
            "content_hashes": content_hashes,
            "verified": True,
            "blocking_rewards": [],
        }
        promotion_schema_errors = artifact_schema_errors(
            self.context.root, manifest, "promotion_manifest.schema.json"
        )
        if promotion_schema_errors:
            return {
                "promoted": False,
                "attempt_id": attempt_id,
                "blocking_rewards": ["promotion_manifest_schema"],
                "promoted_pointer_unchanged": True,
                "schema_errors": promotion_schema_errors,
            }
        # Build the immutable snapshot in a temporary sibling directory, then
        # rename it into place. Existing snapshots are never replaced.
        promoted_root.mkdir(parents=True, exist_ok=True)
        snapshot_dir.parent.mkdir(parents=True, exist_ok=True)
        snapshot_tmp = Path(
            tempfile.mkdtemp(prefix=f".{attempt_id}.", dir=str(snapshot_dir.parent))
        )
        try:
            shutil.rmtree(snapshot_tmp)
            shutil.copytree(attempt_dir, snapshot_tmp)
            write_json(manifest, snapshot_tmp / "promotion_manifest.json")
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
                snapshot_tmp / "publication_decision.json",
            )
            snapshot_tmp.replace(snapshot_dir)
        except Exception:
            if snapshot_tmp.exists():
                shutil.rmtree(snapshot_tmp)
            raise

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
            "snapshot_dir": str(snapshot_dir),
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
        decision = dict(decision)
        decision["publication_mode"] = "production"
        write_json(decision, decision_path)
        return self.verify_semantic(run_id=run_dir.name, profile="production")

    def _resolve_run_dir(self, run_id: str) -> Path:
        return resolve_run_dir(self.context.artifacts_dir, run_id)

    @staticmethod
    def _read_optional(path: Path) -> dict[str, Any] | None:
        if not path.exists():
            return None
        return read_json(path)


def profile_required(config: dict[str, Any], profile: str) -> list[str]:
    return list(config.get("profiles", {}).get(profile, {}).get("required_rewards", []))


def _file_hash(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _content_hashes(run_dir: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for name in _PROMOTED_ARTIFACTS:
        path = run_dir / name
        digest = _file_hash(path)
        if digest:
            hashes[name] = digest
    return hashes


def _promotion_manifest_is_complete(manifest: dict[str, Any]) -> bool:
    """Validate the structural contract before trusting a production pointer."""
    required = {
        "attempt_id",
        "profile",
        "publication_mode",
        "promoted_at",
        "reward_card_hash",
        "semantic_verification_hash",
        "rewards_config_hash",
        "content_hashes",
        "verified",
        "blocking_rewards",
    }
    if not required.issubset(manifest):
        return False
    if manifest.get("verified") is not True:
        return False
    if manifest.get("publication_mode") != "production":
        return False
    if not isinstance(manifest.get("attempt_id"), str) or not manifest["attempt_id"]:
        return False
    if not isinstance(manifest.get("content_hashes"), dict) or not manifest["content_hashes"]:
        return False
    if manifest.get("blocking_rewards") != []:
        return False
    for key in ("reward_card_hash", "semantic_verification_hash", "rewards_config_hash"):
        value = manifest.get(key)
        if not isinstance(value, str) or len(value) != 64:
            return False
    return all(
        isinstance(key, str) and isinstance(value, str) and len(value) == 64
        for key, value in manifest["content_hashes"].items()
    )


def _semantic_integrity_hash(payload: dict[str, Any]) -> str:
    """Hash semantic results without the volatile generation timestamp."""
    stable = {
        "run_id": payload.get("run_id"),
        "profile": payload.get("profile"),
        "publication_mode": payload.get("publication_mode"),
        "passed": payload.get("passed"),
        "failure_reasons": payload.get("failure_reasons"),
        "reconciliation_ok": payload.get("reconciliation_ok"),
        "checks": payload.get("checks"),
    }
    return hashlib.sha256(json.dumps(stable, sort_keys=True, default=str).encode()).hexdigest()


def _reward_card_integrity_hash(card: dict[str, Any]) -> str:
    """Hash all stable reward content, excluding only volatile card metadata."""
    payload = {}
    for key, value in sorted((card.get("rewards") or {}).items()):
        if not isinstance(value, dict):
            payload[key] = value
            continue
        payload[key] = {
            "state": value.get("state"),
            "metric": value.get("metric"),
            "threshold": value.get("threshold"),
            "evidence": value.get("evidence"),
            "negative_tests_passed": value.get("negative_tests_passed"),
            "failure_reasons": value.get("failure_reasons"),
        }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def copy_promoted_snapshot(source: Path, dest: Path) -> None:
    """Helper for tests: deep-copy a promoted tree without following specials."""
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(source, dest)
