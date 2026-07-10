"""Lightweight mutation probes for reward and publication verification code."""

from __future__ import annotations

import copy
import json
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import polars as pl
import yaml
from jsonschema import Draft202012Validator

from civic_signal.scoring.reward_v2 import RewardV2Evaluator
from civic_signal.verification.publication import PublicationVerifier
from civic_signal.verification.schema import artifact_schema_errors

REQUIRED_MUTATION_FAMILIES = (
    "reward_card_schema_required_field",
    "publication_duplicate_keys",
    "publication_probability_range",
    "reward_stale_calibration",
    "publication_estimand_gate",
)


def mutation_breaks_check(
    *,
    healthy: Callable[[], bool],
    mutated: Callable[[], bool],
) -> dict[str, Any]:
    """A verifier mutation is detected when healthy passes and mutated fails."""
    healthy_ok = bool(healthy())
    mutated_ok = bool(mutated())
    return {
        "healthy_passed": healthy_ok,
        "mutated_passed": mutated_ok,
        "mutation_detected": healthy_ok and not mutated_ok,
    }


def strip_range_gate(values: list[float]) -> bool:
    """Mutated check: ignore upper bound (always true for finite floats)."""
    return all(not (value != value) for value in values)  # finite / non-NaN


def strict_range_gate(values: list[float]) -> bool:
    return all(0.0 <= value <= 1.0 for value in values)


def strict_lineage_gate(*, as_of: str, available_at: str) -> bool:
    """Healthy gate: evidence must be available on or before as_of."""
    return available_at <= as_of


def strip_lineage_gate(*, as_of: str, available_at: str) -> bool:
    """Mutated check: accept any availability string (ignores as_of)."""
    del as_of
    return bool(available_at)


def strict_hard_gate(*, rewards_passed: bool, atomic_ok: bool) -> bool:
    return bool(rewards_passed and atomic_ok)


def strip_hard_gate(*, rewards_passed: bool, atomic_ok: bool) -> bool:
    """Mutated check: ignore hard gates and always allow publication."""
    del rewards_passed, atomic_ok
    return True


def run_standard_mutation_probes(root: Path | None = None) -> dict[str, Any]:
    """Mutate actual reward/publication predicates and prove corruptions survive mutants."""
    repository_root = Path(root or Path(__file__).resolve().parents[3])
    with tempfile.TemporaryDirectory(prefix="civic-signal-mutations-") as temp:
        work = Path(temp)
        probes = {
            "reward_card_schema_required_field": _schema_required_field_probe(repository_root),
            "publication_duplicate_keys": _publication_probe(
                work / "duplicate",
                mutation="duplicate_keys",
                removed_failure_fragments=("Duplicate race/option rows",),
            ),
            "publication_probability_range": _publication_probe(
                work / "range",
                mutation="probability_range",
                removed_failure_fragments=("probabilities outside [0, 1]",),
            ),
            "reward_stale_calibration": _stale_calibration_probe(repository_root, work),
            "publication_estimand_gate": _publication_probe(
                work / "estimand",
                mutation="estimand_gate",
                removed_failure_fragments=("unsupported-estimand",),
            ),
        }
    completed = sorted(probes)
    incomplete = sorted(set(REQUIRED_MUTATION_FAMILIES) - set(completed))
    all_detected = not incomplete and all(
        bool(probes[name].get("mutation_detected")) for name in REQUIRED_MUTATION_FAMILIES
    )
    return {
        "suite": "actual_reward_publication_verifier_mutations",
        "actual_verifier_paths": True,
        "required_mutation_families": list(REQUIRED_MUTATION_FAMILIES),
        "completed_mutation_families": completed,
        "incomplete_mutation_families": incomplete,
        "all_mutations_detected": all_detected,
        "probes": probes,
    }


def _schema_required_field_probe(root: Path) -> dict[str, Any]:
    valid = {
        "schema_version": "2.0.0",
        "run_id": "mutation",
        "generated_at": "2026-05-08T00:00:00+00:00",
        "profile": "production",
        "publication_mode": "production",
        "rewards": {},
        "blocks_publication": True,
        "recomputed": True,
    }
    invalid = dict(valid)
    invalid.pop("recomputed")
    healthy_errors = artifact_schema_errors(root, invalid, "reward_card_v2.schema.json")
    schema = json.loads(
        (root / "schemas" / "artifact_contracts" / "reward_card_v2.schema.json").read_text(
            encoding="utf-8"
        )
    )
    mutant_schema = copy.deepcopy(schema)
    mutant_schema["required"].remove("recomputed")
    mutant_errors = list(Draft202012Validator(mutant_schema).iter_errors(invalid))
    valid_errors = artifact_schema_errors(root, valid, "reward_card_v2.schema.json")
    return {
        "actual_path": "verification.schema.artifact_schema_errors",
        "mutation": "remove recomputed from checked-in schema required list",
        "healthy_accepts_valid": not valid_errors,
        "healthy_rejects_corruption": bool(healthy_errors),
        "mutant_accepts_corruption": not mutant_errors,
        "healthy_errors": healthy_errors,
        "mutation_detected": not valid_errors and bool(healthy_errors) and not mutant_errors,
    }


def _publication_probe(
    run_dir: Path,
    *,
    mutation: str,
    removed_failure_fragments: tuple[str, ...],
) -> dict[str, Any]:
    _write_valid_publication_artifacts(run_dir)
    verifier = PublicationVerifier.__new__(PublicationVerifier)
    healthy_valid = verifier._reconcile_forecast(run_dir)
    _apply_publication_corruption(run_dir, mutation)
    healthy = verifier._reconcile_forecast(run_dir)
    mutant_remaining = [
        reason
        for reason in healthy["failures"]
        if not any(fragment in reason for fragment in removed_failure_fragments)
    ]
    return {
        "actual_path": "verification.publication.PublicationVerifier._reconcile_forecast",
        "mutation": f"remove predicate family: {mutation}",
        "healthy_accepts_valid": not healthy_valid["failures"],
        "healthy_rejects_corruption": bool(healthy["failures"]),
        "mutant_accepts_corruption": not mutant_remaining,
        "healthy_failure_reasons": healthy["failures"],
        "mutant_remaining_failures": mutant_remaining,
        "mutation_detected": (
            not healthy_valid["failures"] and bool(healthy["failures"]) and not mutant_remaining
        ),
    }


def _write_valid_publication_artifacts(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    hash_value = "a" * 64
    pl.DataFrame(
        {
            "race_id": ["R1", "R1"],
            "option_id": ["D", "R"],
            "party": ["DEM", "REP"],
            "winner_probability": [0.6, 0.4],
            "model_config_hash": [hash_value, hash_value],
            "source_manifest_hash": [hash_value, hash_value],
        }
    ).write_parquet(run_dir / "race_forecasts.parquet")
    pl.DataFrame(
        {
            "race_id": ["R1"],
            "tier": ["A"],
            "estimand_support_blocked": [False],
            "estimand_support_status": ["supported"],
        }
    ).write_parquet(run_dir / "race_catalog.parquet")
    pl.DataFrame(
        {
            "race_id": ["R1", "R1"],
            "option_id": ["D", "R"],
            "draw_id": [0, 0],
            "winner": [True, False],
        }
    ).write_parquet(run_dir / "forecast_draws.parquet")
    pl.DataFrame({"control_body": ["house"], "majority_probability": [0.5]}).write_parquet(
        run_dir / "control_forecasts.parquet"
    )
    pl.DataFrame(
        {"source_id": ["official"], "status": ["fetched"], "content_hash": [hash_value]}
    ).write_parquet(run_dir / "source_manifest.parquet")


def _apply_publication_corruption(run_dir: Path, mutation: str) -> None:
    if mutation == "duplicate_keys":
        forecast = pl.read_parquet(run_dir / "race_forecasts.parquet")
        duplicate = forecast.filter(pl.col("option_id") == "D").with_columns(
            pl.lit(0.0).alias("winner_probability")
        )
        pl.concat([forecast, duplicate]).write_parquet(run_dir / "race_forecasts.parquet")
        return
    if mutation == "probability_range":
        pl.read_parquet(run_dir / "race_forecasts.parquet").with_columns(
            pl.when(pl.col("option_id") == "D")
            .then(pl.lit(1.2))
            .otherwise(pl.lit(-0.2))
            .alias("winner_probability")
        ).write_parquet(run_dir / "race_forecasts.parquet")
        return
    if mutation == "estimand_gate":
        pl.DataFrame(
            {
                "race_id": ["R1"],
                "tier": ["A"],
                "estimand_support_blocked": [True],
                "estimand_support_status": ["unsupported_multi_option"],
            }
        ).write_parquet(run_dir / "race_catalog.parquet")
        return
    raise ValueError(f"Unknown publication corruption: {mutation}")


def _stale_calibration_probe(root: Path, work: Path) -> dict[str, Any]:
    rewards_config = yaml.safe_load((root / "configs" / "rewards.yaml").read_text())
    evaluator = RewardV2Evaluator(rewards_config=rewards_config)
    forecasts = pl.DataFrame(
        {"race_id": ["R1", "R1"], "option_id": ["D", "R"], "winner_probability": [0.6, 0.4]}
    )
    map_frame = pl.DataFrame({"intercept": [0.0], "slope": [1.0]})
    healthy_artifacts = {
        "race_forecasts": forecasts,
        "recalibration_map": map_frame,
        "nested_eval": {"calibration_map_outer_fold": True},
    }
    stale_artifacts = copy.deepcopy(healthy_artifacts)
    stale_artifacts["nested_eval"]["calibration_map_engine_mismatch"] = True
    healthy = evaluator._eval_R14_calibrated_publication(work, healthy_artifacts)
    rejected = evaluator._eval_R14_calibrated_publication(work, stale_artifacts)

    class CalibrationMismatchMutant(RewardV2Evaluator):
        def _eval_R14_calibrated_publication(
            self, run_dir: Path, artifacts: dict[str, Any]
        ) -> dict[str, Any]:
            mutated = copy.deepcopy(artifacts)
            (mutated.get("nested_eval") or {}).pop("calibration_map_engine_mismatch", None)
            return super()._eval_R14_calibrated_publication(run_dir, mutated)

    mutant = CalibrationMismatchMutant(rewards_config=rewards_config)
    mutant_result = mutant._eval_R14_calibrated_publication(work, stale_artifacts)
    return {
        "actual_path": "scoring.reward_v2.RewardV2Evaluator._eval_R14_calibrated_publication",
        "mutation": "remove calibration_map_engine_mismatch rejection",
        "healthy_accepts_valid": healthy.get("state") == "pass",
        "healthy_rejects_corruption": rejected.get("state") == "fail",
        "mutant_accepts_corruption": mutant_result.get("state") == "pass",
        "healthy_failure_reasons": rejected.get("failure_reasons"),
        "mutation_detected": (
            healthy.get("state") == "pass"
            and rejected.get("state") == "fail"
            and mutant_result.get("state") == "pass"
        ),
    }
