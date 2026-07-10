"""Extra coverage for publication hard-gate paths."""

from __future__ import annotations

import shutil
from pathlib import Path

import polars as pl
from test_rewards_v2 import REPO_ROOT, REWARDS_YAML, _seed_run, _write_json

from civic_signal.config import ProjectContext
from civic_signal.verification.publication import (
    PublicationVerifier,
    _content_hashes,
    _file_hash,
    _promotion_manifest_is_complete,
    _reward_card_integrity_hash,
    copy_promoted_snapshot,
    profile_required,
    resolve_run_dir,
)


def test_helpers_and_copy(tmp_path: Path) -> None:
    assert _file_hash(tmp_path / "missing") is None
    f = tmp_path / "x.txt"
    f.write_text("hi", encoding="utf-8")
    assert _file_hash(f)
    required = profile_required(
        {"profiles": {"production": {"required_rewards": ["R0_build"]}}},
        "production",
    )
    assert required == ["R0_build"]
    assert _reward_card_integrity_hash(
        {
            "rewards": {
                "R0_build": {
                    "state": "pass",
                    "threshold": {},
                    "evidence": [],
                    "failure_reasons": [],
                }
            }
        }
    )
    src = tmp_path / "src"
    src.mkdir()
    (src / "a").write_text("1", encoding="utf-8")
    dest = tmp_path / "dest"
    copy_promoted_snapshot(src, dest)
    assert (dest / "a").read_text(encoding="utf-8") == "1"
    (src / "a").write_text("2", encoding="utf-8")
    copy_promoted_snapshot(src, dest)
    assert (dest / "a").read_text(encoding="utf-8") == "2"


def test_semantic_interval_and_control_range(tmp_path: Path) -> None:
    root = tmp_path / "proj"
    (root / "configs").mkdir(parents=True)
    shutil.copy(REWARDS_YAML, root / "configs" / "rewards.yaml")
    artifacts = root / "artifacts"
    run = artifacts / "runs" / "iv"
    run.mkdir(parents=True)
    pl.DataFrame(
        {
            "race_id": ["r0"],
            "winner_probability": [0.5],
            "share_p10": [0.6],
            "share_p50": [0.5],
            "share_p90": [0.4],
            "model_config_hash": ["a"],
            "source_manifest_hash": ["b"],
        }
    ).write_parquet(run / "race_forecasts.parquet")
    pl.DataFrame({"race_id": ["r0"], "tier": ["A"]}).write_parquet(run / "race_catalog.parquet")
    pl.DataFrame({"race_id": ["r0"], "draw_id": [0]}).write_parquet(run / "forecast_draws.parquet")
    pl.DataFrame({"party": ["DEM"], "majority_probability": [1.5]}).write_parquet(
        run / "control_forecasts.parquet"
    )
    pl.DataFrame({"source_id": ["s"], "status": ["fetched"], "content_hash": ["h"]}).write_parquet(
        run / "source_manifest.parquet"
    )
    _write_json(run / "publication_decision.json", {"publication_mode": "research"})
    ctx = ProjectContext.create(root=root, artifacts_dir=artifacts)
    result = PublicationVerifier(ctx).verify_semantic(
        run_id="iv", profile="research", require_promotion_for_production=False
    )
    assert result["passed"] is False
    assert any("interval" in r.lower() or "control" in r.lower() for r in result["failure_reasons"])


def test_content_hashes_on_seed(tmp_path: Path) -> None:
    run = _seed_run(tmp_path / "h")
    hashes = _content_hashes(run)
    assert "race_forecasts.parquet" in hashes
    assert resolve_run_dir(tmp_path, "h") == run


def test_semantic_out_of_range_and_missing_lineage(tmp_path: Path) -> None:
    root = tmp_path / "proj"
    (root / "configs").mkdir(parents=True)
    shutil.copy(REWARDS_YAML, root / "configs" / "rewards.yaml")
    artifacts = root / "artifacts"
    run = artifacts / "runs" / "oor"
    run.mkdir(parents=True)
    pl.DataFrame(
        {
            "race_id": ["r0", "r1"],
            "option_id": ["a", "b"],
            "winner_probability": [1.5, -0.1],
        }
    ).write_parquet(run / "race_forecasts.parquet")
    pl.DataFrame({"race_id": ["r0", "r1"], "tier": ["A", "A"]}).write_parquet(
        run / "race_catalog.parquet"
    )
    pl.DataFrame({"race_id": ["r0"], "draw_id": [0]}).write_parquet(run / "forecast_draws.parquet")
    pl.DataFrame({"party": ["DEM"], "majority_probability": [0.5]}).write_parquet(
        run / "control_forecasts.parquet"
    )
    pl.DataFrame({"source_id": ["s"], "status": ["fetched"], "content_hash": ["h"]}).write_parquet(
        run / "source_manifest.parquet"
    )
    _write_json(run / "publication_decision.json", {"publication_mode": "research"})
    ctx = ProjectContext.create(root=root, artifacts_dir=artifacts)
    result = PublicationVerifier(ctx).verify_semantic(
        run_id="oor", profile="research", require_promotion_for_production=False
    )
    assert result["passed"] is False
    joined = " ".join(result["failure_reasons"])
    assert "outside" in joined or "lineage" in joined or "Draws missing" in joined


def test_reject_relabel_without_decision_file(tmp_path: Path) -> None:
    root = tmp_path / "proj"
    (root / "configs").mkdir(parents=True)
    shutil.copy(REWARDS_YAML, root / "configs" / "rewards.yaml")
    artifacts = root / "artifacts"
    run = artifacts / "runs" / "nolabel"
    _seed_run(run)
    (run / "publication_decision.json").unlink(missing_ok=True)
    (run / "promotion_manifest.json").unlink(missing_ok=True)
    ctx = ProjectContext.create(root=root, artifacts_dir=artifacts)
    result = PublicationVerifier(ctx).reject_relabel_without_manifest(run)
    assert result["passed"] is False


def test_semantic_empty_control_and_manifest(tmp_path: Path) -> None:
    root = tmp_path / "proj"
    (root / "configs").mkdir(parents=True)
    shutil.copy(REWARDS_YAML, root / "configs" / "rewards.yaml")
    artifacts = root / "artifacts"
    run = artifacts / "runs" / "emptyctrl"
    run.mkdir(parents=True)
    pl.DataFrame(
        {
            "race_id": ["r0"],
            "winner_probability": [0.5],
            "model_config_hash": ["a"],
            "source_manifest_hash": ["b"],
        }
    ).write_parquet(run / "race_forecasts.parquet")
    pl.DataFrame({"race_id": ["r0"], "tier": ["A"]}).write_parquet(run / "race_catalog.parquet")
    pl.DataFrame({"race_id": ["r0"], "draw_id": [0]}).write_parquet(run / "forecast_draws.parquet")
    pl.DataFrame(
        {"party": [], "majority_probability": []},
        schema={"party": pl.Utf8, "majority_probability": pl.Float64},
    ).write_parquet(run / "control_forecasts.parquet")
    pl.DataFrame(
        {"source_id": [], "status": [], "content_hash": []},
        schema={"source_id": pl.Utf8, "status": pl.Utf8, "content_hash": pl.Utf8},
    ).write_parquet(run / "source_manifest.parquet")
    _write_json(run / "publication_decision.json", {"publication_mode": "research"})
    ctx = ProjectContext.create(root=root, artifacts_dir=artifacts)
    result = PublicationVerifier(ctx).verify_semantic(
        run_id="emptyctrl", profile="research", require_promotion_for_production=False
    )
    assert result["passed"] is False


def test_semantic_simplex_failure_multi_option(tmp_path: Path) -> None:
    root = tmp_path / "proj"
    (root / "configs").mkdir(parents=True)
    shutil.copy(REWARDS_YAML, root / "configs" / "rewards.yaml")
    artifacts = root / "artifacts"
    run = artifacts / "runs" / "simplex"
    run.mkdir(parents=True)
    pl.DataFrame(
        {
            "race_id": ["r0", "r0"],
            "option_id": ["dem", "rep"],
            "winner_probability": [0.7, 0.7],
            "model_config_hash": ["a", "a"],
            "source_manifest_hash": ["b", "b"],
        }
    ).write_parquet(run / "race_forecasts.parquet")
    pl.DataFrame({"race_id": ["r0"], "tier": ["A"]}).write_parquet(run / "race_catalog.parquet")
    pl.DataFrame({"race_id": ["r0"], "draw_id": [0]}).write_parquet(run / "forecast_draws.parquet")
    pl.DataFrame({"party": ["DEM"], "majority_probability": [0.5]}).write_parquet(
        run / "control_forecasts.parquet"
    )
    pl.DataFrame({"source_id": ["s"], "status": ["fetched"], "content_hash": ["h"]}).write_parquet(
        run / "source_manifest.parquet"
    )
    _write_json(run / "publication_decision.json", {"publication_mode": "research"})
    ctx = ProjectContext.create(root=root, artifacts_dir=artifacts)
    result = PublicationVerifier(ctx).verify_semantic(
        run_id="simplex", profile="research", require_promotion_for_production=False
    )
    assert result["passed"] is False
    assert any("simplex" in r.lower() for r in result["failure_reasons"])


def test_promote_semantic_failure_path(tmp_path: Path) -> None:
    """Promotion refuses when semantic reconciliation cannot pass."""
    root = tmp_path / "proj"
    (root / "configs").mkdir(parents=True)
    shutil.copy(REWARDS_YAML, root / "configs" / "rewards.yaml")
    shutil.copy(REPO_ROOT / "configs" / "model.yaml", root / "configs" / "model.yaml")
    artifacts = root / "artifacts"
    attempt = artifacts / "attempts" / "sem-fail"
    _seed_run(attempt)
    # Break required artifact set after seed (delete control).
    (attempt / "control_forecasts.parquet").unlink()
    ctx = ProjectContext.create(root=root, artifacts_dir=artifacts)
    # Even if rewards somehow pass, semantic should block; with deleted control,
    # rewards may also block. Either way promotion is false.
    payload = PublicationVerifier(ctx).attempt_promote(attempt_id="sem-fail", profile="production")
    assert payload["promoted"] is False


def test_resolve_absolute_run_path(tmp_path: Path) -> None:
    run = tmp_path / "abs-run"
    run.mkdir()
    assert resolve_run_dir(tmp_path / "artifacts", str(run)) == run


def test_legacy_artifacts_exist_helper(tmp_path: Path) -> None:
    from civic_signal.scoring.rewards import RewardEvaluator

    assert RewardEvaluator._artifacts_exist(tmp_path) is False


def test_semantic_verification_is_pure_by_default(tmp_path: Path) -> None:
    root = tmp_path / "proj"
    (root / "configs").mkdir(parents=True)
    shutil.copy(REWARDS_YAML, root / "configs" / "rewards.yaml")
    artifacts = root / "artifacts"
    run = artifacts / "runs" / "pure"
    _seed_run(run)
    before = (run / "semantic_verification.json").read_bytes()
    ctx = ProjectContext.create(root=root, artifacts_dir=artifacts)
    result = PublicationVerifier(ctx).verify_semantic(
        run_id="pure", profile="research", require_promotion_for_production=False
    )
    assert result["passed"] is True
    assert (run / "semantic_verification.json").read_bytes() == before


def test_semantic_rejects_null_probability_and_catalog_gap(tmp_path: Path) -> None:
    root = tmp_path / "proj"
    (root / "configs").mkdir(parents=True)
    shutil.copy(REWARDS_YAML, root / "configs" / "rewards.yaml")
    artifacts = root / "artifacts"
    run = artifacts / "runs" / "invalid"
    _seed_run(run)
    pl.DataFrame(
        {
            "race_id": ["r0"],
            "winner_probability": [None],
            "model_config_hash": ["a"],
            "source_manifest_hash": ["b"],
        },
        schema={
            "race_id": pl.String,
            "winner_probability": pl.Float64,
            "model_config_hash": pl.String,
            "source_manifest_hash": pl.String,
        },
    ).write_parquet(run / "race_forecasts.parquet")
    pl.DataFrame({"race_id": ["r0", "r_missing"], "tier": ["A", "A"]}).write_parquet(
        run / "race_catalog.parquet"
    )
    pl.DataFrame(
        {"control_body": ["senate"], "party": ["DEM"], "control_probability": [1.5]}
    ).write_parquet(run / "control_forecasts.parquet")
    ctx = ProjectContext.create(root=root, artifacts_dir=artifacts)
    result = PublicationVerifier(ctx).verify_semantic(
        run_id="invalid", profile="research", require_promotion_for_production=False
    )
    assert result["passed"] is False
    reasons = " ".join(result["failure_reasons"]).lower()
    assert "missing winner_probability" in reasons or "missing catalog" in reasons
    assert "control" in reasons


def test_partial_promotion_manifest_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "proj"
    (root / "configs").mkdir(parents=True)
    shutil.copy(REWARDS_YAML, root / "configs" / "rewards.yaml")
    artifacts = root / "artifacts"
    run = artifacts / "runs" / "partial"
    _seed_run(run)
    _write_json(run / "publication_decision.json", {"publication_mode": "production"})
    _write_json(
        run / "promotion_manifest.json",
        {
            "verified": True,
            "attempt_id": "partial",
            "publication_mode": "production",
            "content_hashes": {"race_forecasts.parquet": "0" * 64},
        },
    )
    ctx = ProjectContext.create(root=root, artifacts_dir=artifacts)
    result = PublicationVerifier(ctx).verify_semantic(run_id="partial", profile="production")
    assert result["passed"] is False
    assert any("manifest" in reason or "hash" in reason for reason in result["failure_reasons"])


def test_reusing_attempt_id_does_not_replace_snapshot(tmp_path: Path) -> None:
    root = tmp_path / "proj"
    (root / "configs").mkdir(parents=True)
    shutil.copy(REWARDS_YAML, root / "configs" / "rewards.yaml")
    shutil.copy(REPO_ROOT / "configs" / "model.yaml", root / "configs" / "model.yaml")
    artifacts = root / "artifacts"
    attempt = artifacts / "attempts" / "collision"
    _seed_run(attempt)
    ctx = ProjectContext.create(root=root, artifacts_dir=artifacts)
    verifier = PublicationVerifier(ctx)
    first = verifier.attempt_promote(attempt_id="collision", profile="production")
    assert first["promoted"] is True
    snapshot = Path(first["snapshot_dir"])
    marker = snapshot / "immutable-marker.txt"
    marker.write_text("keep", encoding="utf-8")

    second = verifier.attempt_promote(attempt_id="collision", profile="production")
    assert second["promoted"] is False
    assert "immutable_attempt_collision" in second["blocking_rewards"]
    assert marker.read_text(encoding="utf-8") == "keep"


def test_promotion_manifest_structural_contract() -> None:
    valid = {
        "attempt_id": "a",
        "profile": "production",
        "publication_mode": "production",
        "promoted_at": "2026-01-01T00:00:00Z",
        "reward_card_hash": "a" * 64,
        "semantic_verification_hash": "b" * 64,
        "rewards_config_hash": "c" * 64,
        "content_hashes": {"race_forecasts.parquet": "d" * 64},
        "verified": True,
        "blocking_rewards": [],
    }
    assert _promotion_manifest_is_complete(valid)
    for key in ("attempt_id", "reward_card_hash", "semantic_verification_hash"):
        bad = dict(valid)
        bad.pop(key)
        assert not _promotion_manifest_is_complete(bad)
    for key, value in (
        ("verified", False),
        ("publication_mode", "research"),
        ("attempt_id", ""),
        ("content_hashes", {}),
        ("blocking_rewards", ["R0"]),
        ("reward_card_hash", "short"),
    ):
        bad = dict(valid)
        bad[key] = value
        assert not _promotion_manifest_is_complete(bad)
    bad_content = dict(valid)
    bad_content["content_hashes"] = {"race_forecasts.parquet": "short"}
    assert not _promotion_manifest_is_complete(bad_content)


def test_semantic_rejects_missing_manifest_and_catalog_columns(tmp_path: Path) -> None:
    root = tmp_path / "proj"
    (root / "configs").mkdir(parents=True)
    shutil.copy(REWARDS_YAML, root / "configs" / "rewards.yaml")
    artifacts = root / "artifacts"
    run = artifacts / "runs" / "schema"
    _seed_run(run)
    pl.DataFrame({"source_id": ["s"]}).write_parquet(run / "source_manifest.parquet")
    pl.DataFrame({"winner_probability": [0.5]}).write_parquet(run / "race_forecasts.parquet")
    pl.DataFrame({"tier": ["A"]}).write_parquet(run / "race_catalog.parquet")
    ctx = ProjectContext.create(root=root, artifacts_dir=artifacts)
    result = PublicationVerifier(ctx).verify_semantic(
        run_id="schema", profile="research", require_promotion_for_production=False
    )
    assert result["passed"] is False
    reasons = " ".join(result["failure_reasons"])
    assert "source_manifest" in reasons
    assert "race_forecasts" in reasons or "race_catalog" in reasons


def test_semantic_rejects_duplicate_catalog_draw_and_control_keys(tmp_path: Path) -> None:
    root = tmp_path / "proj"
    (root / "configs").mkdir(parents=True)
    shutil.copy(REWARDS_YAML, root / "configs" / "rewards.yaml")
    artifacts = root / "artifacts"
    run = artifacts / "runs" / "duplicate"
    _seed_run(run)
    pl.DataFrame({"race_id": ["r0", "r0"], "tier": ["A", "A"]}).write_parquet(
        run / "race_catalog.parquet"
    )
    pl.DataFrame(
        {"race_id": ["r0"], "draw_id": [0], "option_id": ["r0-D"], "winner": [True]}
    ).write_parquet(run / "forecast_draws.parquet")
    pl.DataFrame({"party": ["DEM"]}).write_parquet(run / "control_forecasts.parquet")
    ctx = ProjectContext.create(root=root, artifacts_dir=artifacts)
    result = PublicationVerifier(ctx).verify_semantic(
        run_id="duplicate", profile="research", require_promotion_for_production=False
    )
    assert result["passed"] is False
    reasons = " ".join(result["failure_reasons"]).lower()
    assert "duplicate" in reasons or "control_forecasts" in reasons


def test_reward_hash_handles_non_mapping_records() -> None:
    assert _reward_card_integrity_hash({"rewards": {"unexpected": "value"}})


def test_production_promote_without_complete_rewards_hard_blocks(tmp_path: Path) -> None:
    """PublicationVerifier.attempt_promote must refuse incomplete production evidence."""
    root = tmp_path / "proj"
    (root / "configs").mkdir(parents=True)
    shutil.copy(REWARDS_YAML, root / "configs" / "rewards.yaml")
    shutil.copy(REPO_ROOT / "configs" / "model.yaml", root / "configs" / "model.yaml")
    artifacts = root / "artifacts"
    attempt = artifacts / "attempts" / "incomplete-rewards"
    _seed_run(attempt)
    # Strip evidence that production rewards require so recomputation hard-blocks.
    (attempt / "ci_manifest.json").unlink(missing_ok=True)
    (attempt / "coverage.json").unlink(missing_ok=True)
    (attempt / "nested_evaluation.json").unlink(missing_ok=True)
    (attempt / "as_of_audit.json").unlink(missing_ok=True)
    (attempt / "latest_daily_update.json").unlink(missing_ok=True)
    (attempt / "latest_update_vs_full_refit_audit.json").unlink(missing_ok=True)
    (attempt / "posterior_diagnostics.json").unlink(missing_ok=True)

    verifier = PublicationVerifier(ProjectContext.create(root=root, artifacts_dir=artifacts))
    payload = verifier.attempt_promote(attempt_id="incomplete-rewards", profile="production")

    assert payload["promoted"] is False
    assert payload.get("promoted_pointer_unchanged") is True
    assert payload.get("blocking_rewards")
    assert not (artifacts / "promoted" / "production" / "promotion_manifest.json").exists()
    decision = attempt / "publication_decision.json"
    # Decision artifact is create-once evidence of the refused promotion.
    if decision.exists():
        from civic_signal.storage.io import read_json

        body = read_json(decision)
        assert body.get("allowed") is False or body.get("blocks_publication") is True
        assert body.get("publication_mode") in {"research", "production"}


def test_research_semantic_verify_may_proceed_without_promotion(tmp_path: Path) -> None:
    """Research mode uses PublicationVerifier without requiring complete production rewards."""
    root = tmp_path / "proj"
    (root / "configs").mkdir(parents=True)
    shutil.copy(REWARDS_YAML, root / "configs" / "rewards.yaml")
    artifacts = root / "artifacts"
    run = artifacts / "runs" / "research-ok"
    _seed_run(run)
    # Remove production-only reward evidence; research semantic path still reconciles forecasts.
    (run / "ci_manifest.json").unlink(missing_ok=True)
    (run / "coverage.json").unlink(missing_ok=True)
    (run / "nested_evaluation.json").unlink(missing_ok=True)
    (run / "promotion_manifest.json").unlink(missing_ok=True)
    _write_json(run / "publication_decision.json", {"publication_mode": "research"})

    result = PublicationVerifier(
        ProjectContext.create(root=root, artifacts_dir=artifacts)
    ).verify_semantic(
        run_id="research-ok",
        profile="research",
        require_promotion_for_production=False,
        force_publication_mode="research",
    )
    assert result["publication_mode"] == "research"
    assert result["passed"] is True
    assert result["reconciliation_ok"] is True


def test_create_once_atomic_promotion_evidence(tmp_path: Path) -> None:
    """Successful promote writes immutable snapshot; second promote is create-once blocked."""
    root = tmp_path / "proj"
    (root / "configs").mkdir(parents=True)
    shutil.copy(REWARDS_YAML, root / "configs" / "rewards.yaml")
    shutil.copy(REPO_ROOT / "configs" / "model.yaml", root / "configs" / "model.yaml")
    artifacts = root / "artifacts"
    attempt = artifacts / "attempts" / "atomic-once"
    _seed_run(attempt)
    verifier = PublicationVerifier(ProjectContext.create(root=root, artifacts_dir=artifacts))

    first = verifier.attempt_promote(attempt_id="atomic-once", profile="production")
    if not first.get("promoted"):
        # Fixture seed may still fail some production rewards; pointer stays untouched.
        assert first.get("promoted_pointer_unchanged") is True
        snapshot_path = artifacts / "promoted" / "production" / "attempts" / "atomic-once"
        # Hard-block path must not invent a create-once snapshot either.
        assert not snapshot_path.exists()
        return

    snapshot = Path(first["snapshot_dir"])
    marker = snapshot / "create-once-marker.txt"
    marker.write_text("immutable", encoding="utf-8")
    promoted_manifest = Path(first["promotion_manifest"]).read_bytes()

    second = verifier.attempt_promote(attempt_id="atomic-once", profile="production")
    assert second["promoted"] is False
    assert "immutable_attempt_collision" in second["blocking_rewards"]
    assert marker.read_text(encoding="utf-8") == "immutable"
    assert Path(first["promotion_manifest"]).read_bytes() == promoted_manifest
