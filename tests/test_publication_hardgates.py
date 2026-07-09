"""Extra coverage for publication hard-gate paths."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import polars as pl

from civic_signal.config import ProjectContext
from civic_signal.verification.publication import (
    PublicationVerifier,
    _content_hashes,
    _file_hash,
    _reward_card_integrity_hash,
    copy_promoted_snapshot,
    profile_required,
    resolve_run_dir,
)
from test_rewards_v2 import REPO_ROOT, REWARDS_YAML, _seed_run, _write_json


def test_helpers_and_copy(tmp_path: Path) -> None:
    assert _file_hash(tmp_path / "missing") is None
    f = tmp_path / "x.txt"
    f.write_text("hi", encoding="utf-8")
    assert _file_hash(f)
    assert profile_required({"profiles": {"production": {"required_rewards": ["R0_build"]}}}, "production") == [
        "R0_build"
    ]
    assert _reward_card_integrity_hash(
        {"rewards": {"R0_build": {"state": "pass", "threshold": {}, "evidence": [], "failure_reasons": []}}}
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
    payload = PublicationVerifier(ctx).attempt_promote(
        attempt_id="sem-fail", profile="production"
    )
    assert payload["promoted"] is False


def test_resolve_absolute_run_path(tmp_path: Path) -> None:
    run = tmp_path / "abs-run"
    run.mkdir()
    assert resolve_run_dir(tmp_path / "artifacts", str(run)) == run


def test_legacy_artifacts_exist_helper(tmp_path: Path) -> None:
    from civic_signal.scoring.rewards import RewardEvaluator

    assert RewardEvaluator._artifacts_exist(tmp_path) is False
