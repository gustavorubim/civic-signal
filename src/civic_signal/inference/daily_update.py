from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from civic_signal.features.slicing import select_latest_eligible_snapshots
from civic_signal.storage.io import read_json, write_json, write_parquet

_POLL_LINEAGE_COLUMNS = (
    "poll_id",
    "survey_id",
    "question_id",
    "revision_id",
    "race_id",
    "option_id",
    "pct",
    "sample_size",
    "pollster",
    "population",
    "sponsor_class",
    "methodology",
    "end_date",
    "published_at",
    "available_at",
    "availability_basis",
    "source_id",
    "source_hash",
    "parser_version",
    "likelihood_question_key",
    "likelihood_contrast_role",
)


@dataclass(frozen=True)
class DailyUpdateResult:
    strategy: str
    posterior_summary: pl.DataFrame
    diagnostics: dict[str, Any]
    fallback_used: str | None
    needs_full_refit: bool
    output_dir: Path


def select_new_eligible_polls(
    polls: pl.DataFrame,
    *,
    anchor_as_of: str,
    update_as_of: str,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Select poll revisions first available strictly after an anchor forecast.

    A poll is new to the update when its recorded ``available_at`` is later than
    the anchor cutoff and no later than the update cutoff. The shared snapshot
    selector first applies event/availability cutoffs and deterministic revision
    selection, so a superseded row cannot enter by file order.
    """
    anchor_cutoff = date.fromisoformat(anchor_as_of[:10])
    update_cutoff = date.fromisoformat(update_as_of[:10])
    if update_cutoff < anchor_cutoff:
        raise ValueError("Daily update as_of cannot be earlier than the anchor as_of")
    audit: dict[str, Any] = {
        "anchor_as_of": anchor_cutoff.isoformat(),
        "update_as_of": update_cutoff.isoformat(),
        "input_poll_rows": polls.height,
        "missing_available_at_rows": 0,
        "eligible_at_update_rows": 0,
        "selected_new_poll_rows": 0,
        "selection_status": "no_new_eligible_polls",
    }
    if polls.is_empty():
        return polls, audit
    if "available_at" not in polls.columns:
        audit["missing_available_at_rows"] = polls.height
        audit["selection_status"] = "insufficient_availability_lineage"
        return polls.head(0), audit

    available_date = (
        pl.col("available_at").cast(pl.String).str.slice(0, 10).str.strptime(pl.Date, strict=False)
    )
    audit["missing_available_at_rows"] = polls.filter(available_date.is_null()).height
    eligible = select_latest_eligible_snapshots(polls, "polls", update_cutoff.isoformat())
    audit["eligible_at_update_rows"] = eligible.height
    if eligible.is_empty():
        return eligible, audit
    new_rows = eligible.filter(available_date > anchor_cutoff)
    audit["selected_new_poll_rows"] = new_rows.height
    if not new_rows.is_empty():
        audit["selection_status"] = "selected"
    return new_rows, audit


def build_new_poll_lineage(
    polls: pl.DataFrame,
    *,
    anchor_as_of: str,
    update_as_of: str,
) -> pl.DataFrame:
    """Return exact source and revision lineage for rows used by an update."""
    selected = [column for column in _POLL_LINEAGE_COLUMNS if column in polls.columns]
    if polls.is_empty():
        schema = {column: pl.String for column in _POLL_LINEAGE_COLUMNS}
        schema.update({"anchor_as_of": pl.String, "update_as_of": pl.String})
        return pl.DataFrame(schema=schema)
    lineage = polls.select(selected).with_columns(
        pl.lit(anchor_as_of[:10]).alias("anchor_as_of"),
        pl.lit(update_as_of[:10]).alias("update_as_of"),
    )
    sort_columns = [
        column
        for column in ("race_id", "question_id", "poll_id", "option_id", "revision_id")
        if column in lineage.columns
    ]
    return lineage.sort(sort_columns) if sort_columns else lineage


def select_independent_poll_contrasts(
    polls: pl.DataFrame,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Reduce poll option rows to independent likelihood contrasts.

    Binary questions contribute one deterministic positive/reference-party row;
    the complementary row contains the same information. Multi-option questions
    contribute K-1 share rows under an explicitly diagonal approximation until
    a full multinomial likelihood is implemented.
    """
    if polls.is_empty():
        return polls, _empty_contrast_diagnostics()
    groups: dict[str, list[dict[str, Any]]] = {}
    for index, row in enumerate(polls.iter_rows(named=True)):
        key = _poll_question_key(row, index)
        groups.setdefault(key, []).append(row)

    selected: list[dict[str, Any]] = []
    binary = 0
    multi = 0
    single = 0
    duplicate_option_rows = 0
    for key in sorted(groups):
        by_option: dict[str, dict[str, Any]] = {}
        for row in groups[key]:
            option_id = str(row.get("option_id") or "")
            existing = by_option.get(option_id)
            if existing is not None:
                duplicate_option_rows += 1
            if existing is None or _stable_poll_row_key(row) > _stable_poll_row_key(existing):
                by_option[option_id] = row
        rows = [by_option[option_id] for option_id in sorted(by_option)]
        if len(rows) == 1:
            single += 1
            chosen = [(rows[0], "single_option")]
        elif len(rows) == 2:
            binary += 1
            chosen = [(_positive_reference_row(rows), "binary_reference_contrast")]
        else:
            multi += 1
            # K-1 simplex dimensions relative to a deterministic final category.
            chosen = [(row, "multi_option_k_minus_one_diagonal_share") for row in rows[:-1]]
        for row, role in chosen:
            selected.append(
                {
                    **row,
                    "likelihood_question_key": key,
                    "likelihood_contrast_role": role,
                }
            )
    frame = pl.DataFrame(selected) if selected else polls.head(0)
    return frame, {
        "question_count": len(groups),
        "input_option_rows": polls.height,
        "independent_contrast_rows": frame.height,
        "binary_question_count": binary,
        "multi_option_question_count": multi,
        "single_option_question_count": single,
        "duplicate_option_rows_removed": duplicate_option_rows,
        "binary_method": "one_positive_reference_contrast_per_question",
        "multi_option_method": "k_minus_one_diagonal_share_approximation",
        "multi_option_covariance_modeled": False,
    }


def _poll_question_key(row: dict[str, Any], index: int) -> str:
    race_id = str(row.get("race_id") or "")
    for column in ("question_id", "survey_id"):
        value = row.get(column)
        if value not in {None, ""}:
            return f"{column}:{race_id}:{value}"
    poll_id = row.get("poll_id")
    if poll_id not in {None, ""}:
        normalized_poll_id = re.sub(
            r"[-_:](?:DEM|REP|IND|OTHER|D|R|I|O|YES|NO)$",
            "",
            str(poll_id),
            flags=re.IGNORECASE,
        )
        return f"poll_id:{race_id}:{normalized_poll_id}"
    fallback = tuple(
        str(row.get(column) or "")
        for column in ("pollster", "start_date", "end_date", "sample_size")
    )
    if any(fallback):
        return f"fallback:{race_id}:{'|'.join(fallback)}"
    return f"row:{race_id}:{index}"


def _positive_reference_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    preferred = {"DEM": 0, "D": 0, "YES": 0, "REP": 1, "R": 1, "NO": 1}
    return min(
        rows,
        key=lambda row: (
            preferred.get(str(row.get("party") or row.get("option_id") or "").upper(), 2),
            str(row.get("option_id") or ""),
        ),
    )


def _stable_poll_row_key(row: dict[str, Any]) -> tuple[str, ...]:
    return tuple(
        str(row.get(column) or "")
        for column in ("available_at", "published_at", "revision_id", "source_hash", "poll_id")
    )


def _empty_contrast_diagnostics() -> dict[str, Any]:
    return {
        "question_count": 0,
        "input_option_rows": 0,
        "independent_contrast_rows": 0,
        "binary_question_count": 0,
        "multi_option_question_count": 0,
        "single_option_question_count": 0,
        "duplicate_option_rows_removed": 0,
        "binary_method": "one_positive_reference_contrast_per_question",
        "multi_option_method": "k_minus_one_diagonal_share_approximation",
        "multi_option_covariance_modeled": False,
    }


def compare_update_vs_full_refit(
    updated_posterior: pl.DataFrame,
    full_refit_posterior: pl.DataFrame,
    *,
    max_probability_mae: float = 0.005,
    max_probability_diff: float = 0.02,
    full_refit_run_id: str | None = None,
) -> dict[str, Any]:
    """Measure option-level probability differences versus an exact full-refit posterior.

    Both frames must carry ``race_id``, ``option_id``, and either draw-level
    ``latent_share`` rows or a pre-aggregated ``latent_share_mean`` column. Metrics are
    null only when inputs cannot form a matched comparison; a successful comparison
    always records finite MAE and max absolute difference.
    """
    update_means = _option_probability_means(updated_posterior, label="update")
    refit_means = _option_probability_means(full_refit_posterior, label="full_refit")
    if update_means.is_empty() or refit_means.is_empty():
        return {
            "status": "failed",
            "comparison_executed": True,
            "reason": "Update or full-refit posterior produced no comparable option means",
            "probability_mae_vs_full_refit": None,
            "probability_max_diff_vs_full_refit": None,
            "matched_option_count": 0,
            "full_refit_run_id": full_refit_run_id,
        }
    joined = update_means.join(refit_means, on=["race_id", "option_id"], how="inner")
    if joined.is_empty():
        return {
            "status": "failed",
            "comparison_executed": True,
            "reason": "No overlapping race/option keys between update and full-refit posteriors",
            "probability_mae_vs_full_refit": None,
            "probability_max_diff_vs_full_refit": None,
            "matched_option_count": 0,
            "full_refit_run_id": full_refit_run_id,
        }
    diffs = (joined["update_mean"] - joined["full_refit_mean"]).abs()
    mae = float(diffs.mean())
    max_diff = float(diffs.max())
    within = mae <= float(max_probability_mae) and max_diff <= float(max_probability_diff)
    return {
        "status": "passed" if within else "failed",
        "comparison_executed": True,
        "reason": (
            "Exact full-refit publication-path comparison executed against supplied "
            "full-refit posterior draws/summary."
        ),
        "probability_mae_vs_full_refit": mae,
        "probability_max_diff_vs_full_refit": max_diff,
        "matched_option_count": joined.height,
        "max_probability_mae_threshold": float(max_probability_mae),
        "max_probability_diff_threshold": float(max_probability_diff),
        "full_refit_run_id": full_refit_run_id,
    }


def run_daily_update(
    anchor_run_dir: Path,
    as_of: str,
    config: dict[str, Any],
    new_polls: pl.DataFrame | None = None,
    *,
    anchor_as_of: str | None = None,
    selection_audit: dict[str, Any] | None = None,
    full_refit_posterior: pl.DataFrame | None = None,
    full_refit_run_id: str | None = None,
) -> DailyUpdateResult:
    daily_config = dict(config.get("daily_update", {}))
    strategy = str(daily_config.get("strategy", "reweighting"))
    if strategy != "reweighting":
        raise ValueError(
            f"Daily update strategy {strategy!r} is not implemented; only 'reweighting' "
            "may be selected until another literal algorithm exists"
        )
    resampling_method = str(daily_config.get("resampling_method", "systematic"))
    if resampling_method != "systematic":
        raise ValueError(
            f"Daily update resampling method {resampling_method!r} is not implemented; "
            "only 'systematic' matches the executed algorithm"
        )
    posterior_path = anchor_run_dir / "posterior_draws.parquet"
    if not posterior_path.exists():
        raise FileNotFoundError(f"Anchor run has no posterior_draws.parquet: {anchor_run_dir}")
    posterior = pl.read_parquet(posterior_path)
    if posterior.is_empty():
        raise ValueError("Anchor posterior_draws.parquet is empty")
    new_polls = new_polls if new_polls is not None else pl.DataFrame()
    anchor_as_of = anchor_as_of or as_of
    if date.fromisoformat(as_of[:10]) < date.fromisoformat(anchor_as_of[:10]):
        raise ValueError("Daily update as_of cannot be earlier than the anchor as_of")
    likelihood_polls, contrast_diagnostics = select_independent_poll_contrasts(new_polls)
    poll_lineage = build_new_poll_lineage(
        likelihood_polls,
        anchor_as_of=anchor_as_of,
        update_as_of=as_of,
    )
    updated_posterior, weight_diagnostics = _reweight_posterior(
        posterior,
        likelihood_polls,
        config=daily_config,
    )
    summary = _posterior_summary(updated_posterior, as_of=as_of)
    diagnostics = _diagnostics(
        strategy=strategy,
        posterior=posterior,
        updated_posterior=updated_posterior,
        summary=summary,
        new_polls=new_polls,
        config=daily_config,
        weight_diagnostics=weight_diagnostics,
        poll_lineage=poll_lineage,
        selection_audit=selection_audit or {},
        contrast_diagnostics=contrast_diagnostics,
        anchor_as_of=anchor_as_of,
        update_as_of=as_of,
        full_refit_posterior=full_refit_posterior,
        full_refit_run_id=full_refit_run_id,
    )
    history = _append_history(anchor_run_dir, summary, diagnostics)
    output_dir = anchor_run_dir / "updates" / as_of
    write_parquet(summary, output_dir / "posterior_summary.parquet")
    write_parquet(updated_posterior, output_dir / "posterior_draws_reweighted.parquet")
    write_parquet(poll_lineage, output_dir / "new_poll_lineage.parquet")
    write_parquet(history, anchor_run_dir / "posterior_history.parquet")
    write_json(diagnostics, output_dir / "daily_update_diagnostics.json")
    write_json(
        diagnostics["update_vs_full_refit"],
        output_dir / "update_vs_full_refit_audit.json",
    )
    write_json(diagnostics, anchor_run_dir / "latest_daily_update.json")
    write_json(
        diagnostics["update_vs_full_refit"],
        anchor_run_dir / "latest_update_vs_full_refit_audit.json",
    )
    return DailyUpdateResult(
        strategy=strategy,
        posterior_summary=summary,
        diagnostics=diagnostics,
        fallback_used=diagnostics["fallback_used"],
        needs_full_refit=bool(diagnostics["needs_full_refit"]),
        output_dir=output_dir,
    )


def _posterior_summary(posterior: pl.DataFrame, as_of: str) -> pl.DataFrame:
    return (
        posterior.group_by(["race_id", "option_id"])
        .agg(
            pl.col("latent_share").mean().alias("latent_share_mean"),
            pl.col("latent_share").quantile(0.10).alias("latent_share_p10"),
            pl.col("latent_share").quantile(0.90).alias("latent_share_p90"),
            pl.col("latent_logit").mean().alias("latent_logit_mean"),
            pl.col("draw_id").n_unique().alias("draw_count"),
        )
        .with_columns(
            pl.lit(as_of).alias("as_of"),
            pl.lit(datetime.now(UTC).isoformat()).alias("updated_at"),
        )
        .sort(["race_id", "option_id"])
    )


def _diagnostics(
    strategy: str,
    posterior: pl.DataFrame,
    updated_posterior: pl.DataFrame,
    summary: pl.DataFrame,
    new_polls: pl.DataFrame,
    config: dict[str, Any],
    weight_diagnostics: dict[str, Any],
    poll_lineage: pl.DataFrame,
    selection_audit: dict[str, Any],
    contrast_diagnostics: dict[str, Any],
    anchor_as_of: str,
    update_as_of: str,
    full_refit_posterior: pl.DataFrame | None = None,
    full_refit_run_id: str | None = None,
) -> dict[str, Any]:
    particle_count = int(posterior["draw_id"].n_unique())
    ess_ratio = float(weight_diagnostics["effective_sample_size_ratio"])
    before = posterior.group_by(["race_id", "option_id"]).agg(
        pl.col("latent_share").mean().alias("before")
    )
    after = updated_posterior.group_by(["race_id", "option_id"]).agg(
        pl.col("latent_share").mean().alias("after")
    )
    joined = before.join(after, on=["race_id", "option_id"], how="inner")
    drift = float((joined["after"] - joined["before"]).abs().max() or 0.0)
    drift_threshold = float(config.get("posterior_drift_threshold", 0.05))
    max_age = int(config.get("full_refit_days_since_anchor", 7))
    anchor_age_days = (
        date.fromisoformat(update_as_of[:10]) - date.fromisoformat(anchor_as_of[:10])
    ).days
    anchor_age_exceeded = anchor_age_days > max_age
    min_ess_ratio = float(config.get("minimum_ess_ratio", 0.5))
    noop = not bool(weight_diagnostics["likelihood_reweighted"])
    needs_full_refit = ess_ratio < min_ess_ratio or drift > drift_threshold or anchor_age_exceeded
    # Honest default: comparison metrics stay null until an exact full-refit path runs.
    # Fixture callers force the comparison by supplying full_refit_posterior draws.
    if full_refit_posterior is not None:
        update_vs_full_refit = compare_update_vs_full_refit(
            updated_posterior,
            full_refit_posterior,
            max_probability_mae=float(config.get("max_probability_mae_vs_refit", 0.005)),
            max_probability_diff=float(config.get("max_probability_diff_vs_refit", 0.02)),
            full_refit_run_id=full_refit_run_id or "fixture-full-refit",
        )
        mae = update_vs_full_refit.get("probability_mae_vs_full_refit")
        max_diff = update_vs_full_refit.get("probability_max_diff_vs_full_refit")
    else:
        update_vs_full_refit = {
            "status": "unavailable",
            "comparison_executed": False,
            "reason": (
                "An exact full-refit publication-path comparison was not executed for this "
                "update; probability differences are intentionally null."
            ),
            "probability_mae_vs_full_refit": None,
            "probability_max_diff_vs_full_refit": None,
            "matched_option_count": 0,
            "full_refit_run_id": None,
        }
        mae = None
        max_diff = None
    quality_passed = not noop and not needs_full_refit
    exact_refit_evidence = (
        update_vs_full_refit.get("comparison_executed") is True
        and update_vs_full_refit.get("status") == "passed"
        and mae is not None
        and max_diff is not None
    )
    return {
        "strategy": strategy,
        "requested_strategy": strategy,
        "executed_strategy": (
            "no_op_previous_posterior" if noop else "likelihood_reweighting_systematic_resampling"
        ),
        "status": "updated" if not noop else "no_new_likelihood_data",
        "new_poll_count": new_polls.height,
        "new_poll_lineage_rows": poll_lineage.height,
        "new_poll_lineage_sha256": _frame_sha256(poll_lineage),
        "new_poll_lineage_path": "new_poll_lineage.parquet",
        "poll_selection_audit": selection_audit,
        "poll_contrast_audit": contrast_diagnostics,
        "matched_new_poll_count": weight_diagnostics["matched_new_poll_count"],
        "matched_independent_contrast_count": weight_diagnostics["matched_new_poll_count"],
        "posterior_row_count": posterior.height,
        "posterior_summary_rows": summary.height,
        "particle_count": particle_count,
        "effective_sample_size_ratio": ess_ratio,
        "minimum_ess_ratio": min_ess_ratio,
        "max_importance_weight": weight_diagnostics["max_importance_weight"],
        "weight_entropy": weight_diagnostics["weight_entropy"],
        "weights_degenerate": ess_ratio < min_ess_ratio,
        "importance_weight_sum": weight_diagnostics["importance_weight_sum"],
        "importance_weight_nonfinite_count": weight_diagnostics[
            "importance_weight_nonfinite_count"
        ],
        "pareto_k": None,
        "pareto_diagnostic_status": "unavailable",
        "pareto_diagnostic_reason": (
            "Raw likelihood importance weights are used; PSIS/Pareto-k smoothing is not "
            "implemented, so no Pareto-k value is claimed."
        ),
        "likelihood_reweighted": not noop,
        "noop": noop,
        "posterior_drift": drift,
        "posterior_drift_threshold": drift_threshold,
        "full_refit_days_since_anchor": max_age,
        "anchor_as_of": anchor_as_of[:10],
        "anchor_age_days": anchor_age_days,
        "anchor_age_exceeds_refit_threshold": anchor_age_exceeded,
        "resampling_method": weight_diagnostics["resampling_method"],
        "systematic_resampling_offset": weight_diagnostics["systematic_resampling_offset"],
        "fallback_used": None,
        "needs_full_refit": needs_full_refit,
        # Comparison against a supplied full-refit posterior is not itself a full refit.
        "full_refit_executed": False,
        "quality_passed": quality_passed,
        "r15_evidence_complete": bool(quality_passed and exact_refit_evidence),
        "probability_mae_vs_full_refit": mae,
        "probability_max_diff_vs_full_refit": max_diff,
        "update_vs_full_refit": update_vs_full_refit,
        "updated_at": datetime.now(UTC).isoformat(),
    }


def _option_probability_means(posterior: pl.DataFrame, *, label: str) -> pl.DataFrame:
    """Collapse draw-level or summary posteriors to option-level probability means."""
    required = {"race_id", "option_id"}
    if posterior.is_empty() or not required.issubset(posterior.columns):
        return pl.DataFrame(
            schema={"race_id": pl.String, "option_id": pl.String, f"{label}_mean": pl.Float64}
        )
    if "latent_share_mean" in posterior.columns:
        means = posterior.group_by(["race_id", "option_id"]).agg(
            pl.col("latent_share_mean").cast(pl.Float64).mean().alias(f"{label}_mean")
        )
    elif "latent_share" in posterior.columns:
        means = posterior.group_by(["race_id", "option_id"]).agg(
            pl.col("latent_share").cast(pl.Float64).mean().alias(f"{label}_mean")
        )
    else:
        return pl.DataFrame(
            schema={"race_id": pl.String, "option_id": pl.String, f"{label}_mean": pl.Float64}
        )
    return means.filter(pl.col(f"{label}_mean").is_not_null() & pl.col(f"{label}_mean").is_finite())


def _reweight_posterior(
    posterior: pl.DataFrame,
    new_polls: pl.DataFrame,
    *,
    config: dict[str, Any],
) -> tuple[pl.DataFrame, dict[str, Any]]:
    required = {"race_id", "option_id", "pct"}
    draw_ids = sorted(int(value) for value in posterior["draw_id"].unique().to_list())
    particle_count = len(draw_ids)
    if new_polls.is_empty() or not required.issubset(new_polls.columns):
        return posterior, _uniform_weight_diagnostics(particle_count)
    polls = new_polls.select(
        "race_id",
        "option_id",
        pl.when(pl.col("pct").cast(pl.Float64) > 1.0)
        .then(pl.col("pct").cast(pl.Float64) / 100.0)
        .otherwise(pl.col("pct").cast(pl.Float64))
        .alias("_observed_share"),
        (
            pl.col("sample_size").cast(pl.Float64).fill_null(600.0)
            if "sample_size" in new_polls.columns
            else pl.lit(600.0)
        ).alias("_sample_size"),
    ).filter(
        pl.col("_observed_share").is_between(0.0, 1.0, closed="both")
        & (pl.col("_sample_size") > 0.0)
    )
    matched = posterior.select(["draw_id", "race_id", "option_id", "latent_share"]).join(
        polls,
        on=["race_id", "option_id"],
        how="inner",
    )
    if matched.is_empty():
        return posterior, _uniform_weight_diagnostics(particle_count)
    nonsampling_sd = float(config.get("poll_nonsampling_sd", 0.03))
    likelihood = (
        matched.with_columns(
            (
                pl.col("_observed_share")
                * (1.0 - pl.col("_observed_share"))
                / pl.col("_sample_size")
                + nonsampling_sd**2
            )
            .sqrt()
            .alias("_poll_sd")
        )
        .with_columns(
            (
                -0.5
                * ((pl.col("_observed_share") - pl.col("latent_share")) / pl.col("_poll_sd")) ** 2
                - pl.col("_poll_sd").log()
            ).alias("_log_likelihood")
        )
        .group_by("draw_id")
        .agg(pl.col("_log_likelihood").sum())
    )
    log_likelihood = {
        int(row["draw_id"]): float(row["_log_likelihood"])
        for row in likelihood.iter_rows(named=True)
    }
    log_weights = np.array([log_likelihood.get(draw_id, 0.0) for draw_id in draw_ids])
    log_weights -= float(log_weights.max())
    weights = np.exp(log_weights)
    weights /= float(weights.sum())
    ess = 1.0 / float(np.square(weights).sum())
    rng = np.random.default_rng(int(config.get("seed", 20260508)))
    sampled_indices, systematic_offset = _systematic_resample(weights, rng)
    sampled = np.asarray(draw_ids, dtype=np.int64)[sampled_indices]
    mapping = pl.DataFrame(
        {
            "draw_id": sampled.tolist(),
            "_updated_draw_id": list(range(particle_count)),
        }
    )
    updated = (
        mapping.join(posterior, on="draw_id", how="left")
        .drop("draw_id")
        .rename({"_updated_draw_id": "draw_id"})
        .select(posterior.columns)
    )
    entropy = float(-(weights * np.log(np.clip(weights, 1e-300, None))).sum())
    return updated, {
        "likelihood_reweighted": True,
        "matched_new_poll_count": polls.join(
            posterior.select(["race_id", "option_id"]).unique(),
            on=["race_id", "option_id"],
            how="inner",
        ).height,
        "effective_sample_size_ratio": ess / max(particle_count, 1),
        "max_importance_weight": float(weights.max()),
        "weight_entropy": entropy,
        "importance_weight_sum": float(weights.sum()),
        "importance_weight_nonfinite_count": int((~np.isfinite(weights)).sum()),
        "resampling_method": "systematic",
        "systematic_resampling_offset": systematic_offset,
    }


def _uniform_weight_diagnostics(particle_count: int) -> dict[str, Any]:
    return {
        "likelihood_reweighted": False,
        "matched_new_poll_count": 0,
        "effective_sample_size_ratio": 1.0 if particle_count else 0.0,
        "max_importance_weight": 1.0 / particle_count if particle_count else None,
        "weight_entropy": float(np.log(particle_count)) if particle_count else None,
        "importance_weight_sum": 1.0 if particle_count else 0.0,
        "importance_weight_nonfinite_count": 0,
        "resampling_method": "none",
        "systematic_resampling_offset": None,
    }


def _systematic_resample(
    weights: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float]:
    """Return low-variance systematic resampling indices for normalized weights."""
    count = int(weights.size)
    if count == 0:
        return np.array([], dtype=np.int64), 0.0
    offset = float(rng.random()) / count
    positions = offset + np.arange(count, dtype=np.float64) / count
    cumulative = np.cumsum(weights, dtype=np.float64)
    cumulative[-1] = 1.0
    return np.searchsorted(cumulative, positions, side="left").astype(np.int64), offset


def _frame_sha256(frame: pl.DataFrame) -> str:
    rows = frame.to_dicts()
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _append_history(
    anchor_run_dir: Path,
    summary: pl.DataFrame,
    diagnostics: dict[str, Any],
) -> pl.DataFrame:
    snapshot = summary.with_columns(
        pl.lit(str(diagnostics["strategy"])).alias("strategy"),
        pl.lit(float(diagnostics["effective_sample_size_ratio"])).alias(
            "effective_sample_size_ratio"
        ),
        pl.lit(float(diagnostics["posterior_drift"])).alias("posterior_drift"),
        pl.lit(bool(diagnostics["needs_full_refit"])).alias("needs_full_refit"),
    )
    path = anchor_run_dir / "posterior_history.parquet"
    if not path.exists():
        return snapshot
    previous = pl.read_parquet(path)
    return pl.concat([previous, snapshot], how="diagonal_relaxed").unique(
        subset=["as_of", "race_id", "option_id"], keep="last"
    )


def read_latest_daily_update(anchor_run_dir: Path) -> dict[str, Any] | None:
    path = anchor_run_dir / "latest_daily_update.json"
    return read_json(path) if path.exists() else None
