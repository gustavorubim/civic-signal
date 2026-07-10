from __future__ import annotations

from pathlib import Path

import polars as pl

from civic_signal.inference.daily_update import (
    compare_update_vs_full_refit,
    run_daily_update,
    select_independent_poll_contrasts,
    select_new_eligible_polls,
)
from civic_signal.storage.io import read_json


def _anchor(tmp_path: Path) -> Path:
    anchor = tmp_path / "anchor"
    rows = []
    for draw_id in range(100):
        dem_share = 0.40 if draw_id < 50 else 0.65
        for option_id, share in (("D", dem_share), ("R", 1.0 - dem_share)):
            rows.append(
                {
                    "draw_id": draw_id,
                    "race_id": "RACE",
                    "option_id": option_id,
                    "latent_share": share,
                    "latent_logit": 0.0,
                }
            )
    anchor.mkdir(parents=True)
    pl.DataFrame(rows).write_parquet(anchor / "posterior_draws.parquet")
    return anchor


def test_daily_update_applies_poll_likelihood_and_records_ess(tmp_path: Path) -> None:
    anchor = _anchor(tmp_path)
    poll = pl.DataFrame(
        {
            "race_id": ["RACE"],
            "option_id": ["D"],
            "pct": [65.0],
            "sample_size": [1000],
        }
    )

    result = run_daily_update(
        anchor,
        "2026-05-09",
        {
            "daily_update": {
                "strategy": "reweighting",
                "minimum_ess_ratio": 0.05,
                "poll_nonsampling_sd": 0.03,
            }
        },
        new_polls=poll,
    )

    dem = result.posterior_summary.filter(pl.col("option_id") == "D").row(0, named=True)
    assert dem["latent_share_mean"] > 0.60
    assert result.diagnostics["likelihood_reweighted"] is True
    assert result.diagnostics["noop"] is False
    assert 0.0 < result.diagnostics["effective_sample_size_ratio"] < 1.0
    assert result.diagnostics["matched_new_poll_count"] == 1
    assert result.diagnostics["executed_strategy"] == (
        "likelihood_reweighting_systematic_resampling"
    )
    assert result.diagnostics["resampling_method"] == "systematic"
    assert result.diagnostics["new_poll_lineage_rows"] == 1
    assert len(result.diagnostics["new_poll_lineage_sha256"]) == 64
    assert result.diagnostics["pareto_k"] is None
    assert result.diagnostics["pareto_diagnostic_status"] == "unavailable"
    assert result.diagnostics["update_vs_full_refit"]["status"] == "unavailable"
    assert result.diagnostics["probability_mae_vs_full_refit"] is None
    assert (result.output_dir / "posterior_draws_reweighted.parquet").exists()
    assert (result.output_dir / "new_poll_lineage.parquet").exists()
    lineage = pl.read_parquet(result.output_dir / "new_poll_lineage.parquet")
    assert lineage["pct"].to_list() == [65.0]
    audit = read_json(result.output_dir / "update_vs_full_refit_audit.json")
    assert audit["comparison_executed"] is False
    assert audit["probability_max_diff_vs_full_refit"] is None


def test_daily_update_labels_no_new_likelihood_data_as_noop(tmp_path: Path) -> None:
    result = run_daily_update(
        _anchor(tmp_path),
        "2026-05-09",
        {"daily_update": {"strategy": "reweighting"}},
        new_polls=pl.DataFrame(),
    )

    assert result.diagnostics["status"] == "no_new_likelihood_data"
    assert result.diagnostics["noop"] is True
    assert result.diagnostics["likelihood_reweighted"] is False
    assert result.diagnostics["effective_sample_size_ratio"] == 1.0
    assert result.diagnostics["quality_passed"] is False
    assert result.diagnostics["r15_evidence_complete"] is False
    assert result.diagnostics["executed_strategy"] == "no_op_previous_posterior"
    assert result.diagnostics["resampling_method"] == "none"


def test_select_new_eligible_polls_uses_availability_window_and_latest_revision() -> None:
    polls = pl.DataFrame(
        {
            "poll_id": ["old", "new", "future", "late-event"],
            "question_id": ["Q1", "Q1", "Q2", "Q3"],
            "revision_id": ["1", "2", "1", "1"],
            "race_id": ["RACE"] * 4,
            "option_id": ["D"] * 4,
            "end_date": ["2026-05-08", "2026-05-09", "2026-05-09", "2026-05-11"],
            "available_at": [
                "2026-05-08T12:00:00Z",
                "2026-05-09T12:00:00Z",
                "2026-05-10T00:00:00Z",
                "2026-05-09T12:00:00Z",
            ],
            "pct": [50.0, 52.0, 53.0, 54.0],
        }
    )

    selected, audit = select_new_eligible_polls(
        polls,
        anchor_as_of="2026-05-08",
        update_as_of="2026-05-09",
    )

    assert selected["poll_id"].to_list() == ["new"]
    assert audit["eligible_at_update_rows"] == 1
    assert audit["selected_new_poll_rows"] == 1
    assert audit["selection_status"] == "selected"


def test_select_new_eligible_polls_requires_availability_lineage() -> None:
    selected, audit = select_new_eligible_polls(
        pl.DataFrame(
            {
                "poll_id": ["P1"],
                "race_id": ["RACE"],
                "option_id": ["D"],
                "end_date": ["2026-05-09"],
                "pct": [51.0],
            }
        ),
        anchor_as_of="2026-05-08",
        update_as_of="2026-05-09",
    )

    assert selected.is_empty()
    assert audit["missing_available_at_rows"] == 1
    assert audit["selection_status"] == "insufficient_availability_lineage"


def test_select_new_eligible_polls_rejects_backdated_update() -> None:
    try:
        select_new_eligible_polls(
            pl.DataFrame(),
            anchor_as_of="2026-05-09",
            update_as_of="2026-05-08",
        )
    except ValueError as error:
        assert "earlier" in str(error)
    else:  # pragma: no cover - assertion guard
        raise AssertionError("Expected a backdated daily update to be rejected")


def test_binary_poll_question_uses_one_reference_contrast() -> None:
    polls = pl.DataFrame(
        {
            "question_id": ["Q1", "Q1"],
            "race_id": ["RACE", "RACE"],
            "option_id": ["D", "R"],
            "party": ["DEM", "REP"],
            "pct": [54.0, 46.0],
            "sample_size": [900, 900],
        }
    )

    contrasts, audit = select_independent_poll_contrasts(polls)

    assert contrasts["option_id"].to_list() == ["D"]
    assert contrasts["likelihood_contrast_role"].to_list() == ["binary_reference_contrast"]
    assert audit["input_option_rows"] == 2
    assert audit["independent_contrast_rows"] == 1
    assert audit["binary_question_count"] == 1


def test_multi_option_poll_uses_explicit_k_minus_one_approximation() -> None:
    polls = pl.DataFrame(
        {
            "question_id": ["Q3"] * 3,
            "race_id": ["RACE"] * 3,
            "option_id": ["A", "B", "C"],
            "pct": [40.0, 35.0, 25.0],
        }
    )

    contrasts, audit = select_independent_poll_contrasts(polls)

    assert contrasts["option_id"].to_list() == ["A", "B"]
    assert audit["independent_contrast_rows"] == 2
    assert audit["multi_option_question_count"] == 1
    assert audit["multi_option_method"] == "k_minus_one_diagonal_share_approximation"
    assert audit["multi_option_covariance_modeled"] is False


def test_option_suffixed_poll_ids_share_one_binary_question() -> None:
    contrasts, audit = select_independent_poll_contrasts(
        pl.DataFrame(
            {
                "poll_id": ["SURVEY-7-DEM", "SURVEY-7-REP"],
                "race_id": ["RACE", "RACE"],
                "option_id": ["D", "R"],
                "pct": [53.0, 47.0],
            }
        )
    )

    assert contrasts.height == 1
    assert audit["binary_question_count"] == 1


def test_daily_update_does_not_double_count_complementary_binary_rows(tmp_path: Path) -> None:
    result = run_daily_update(
        _anchor(tmp_path),
        "2026-05-09",
        {"daily_update": {"strategy": "reweighting", "minimum_ess_ratio": 0.01}},
        new_polls=pl.DataFrame(
            {
                "question_id": ["Q1", "Q1"],
                "race_id": ["RACE", "RACE"],
                "option_id": ["D", "R"],
                "pct": [65.0, 35.0],
                "sample_size": [1000, 1000],
            }
        ),
        anchor_as_of="2026-05-08",
    )

    audit = result.diagnostics["poll_contrast_audit"]
    assert audit["input_option_rows"] == 2
    assert audit["independent_contrast_rows"] == 1
    lineage = pl.read_parquet(result.output_dir / "new_poll_lineage.parquet")
    assert lineage["option_id"].to_list() == ["D"]


def test_systematic_resampling_is_deterministic(tmp_path: Path) -> None:
    poll = pl.DataFrame(
        {
            "race_id": ["RACE"],
            "option_id": ["D"],
            "pct": [65.0],
            "sample_size": [1000],
        }
    )
    first = run_daily_update(
        _anchor(tmp_path / "first"),
        "2026-05-09",
        {"daily_update": {"strategy": "reweighting", "seed": 19}},
        new_polls=poll,
    )
    second = run_daily_update(
        _anchor(tmp_path / "second"),
        "2026-05-09",
        {"daily_update": {"strategy": "reweighting", "seed": 19}},
        new_polls=poll,
    )

    first_draws = pl.read_parquet(first.output_dir / "posterior_draws_reweighted.parquet")
    second_draws = pl.read_parquet(second.output_dir / "posterior_draws_reweighted.parquet")
    assert first_draws.to_dicts() == second_draws.to_dicts()
    assert (
        first.diagnostics["systematic_resampling_offset"]
        == second.diagnostics["systematic_resampling_offset"]
    )


def test_anchor_age_forces_full_refit_without_fabricating_execution(tmp_path: Path) -> None:
    result = run_daily_update(
        _anchor(tmp_path),
        "2026-05-17",
        {
            "daily_update": {
                "strategy": "reweighting",
                "full_refit_days_since_anchor": 7,
                "minimum_ess_ratio": 0.01,
            }
        },
        new_polls=pl.DataFrame(
            {
                "race_id": ["RACE"],
                "option_id": ["D"],
                "pct": [65.0],
                "sample_size": [1000],
            }
        ),
        anchor_as_of="2026-05-08",
    )

    assert result.diagnostics["anchor_age_days"] == 9
    assert result.diagnostics["anchor_age_exceeds_refit_threshold"] is True
    assert result.needs_full_refit is True
    assert result.diagnostics["full_refit_executed"] is False
    assert result.diagnostics["quality_passed"] is False


def test_unimplemented_update_strategy_is_rejected(tmp_path: Path) -> None:
    try:
        run_daily_update(
            _anchor(tmp_path),
            "2026-05-09",
            {"daily_update": {"strategy": "svi_warm_start"}},
        )
    except ValueError as error:
        assert "not implemented" in str(error)
    else:  # pragma: no cover - assertion guard
        raise AssertionError("Expected an unimplemented strategy to be rejected")


def test_mislabeled_resampling_method_is_rejected(tmp_path: Path) -> None:
    try:
        run_daily_update(
            _anchor(tmp_path),
            "2026-05-09",
            {
                "daily_update": {
                    "strategy": "reweighting",
                    "resampling_method": "multinomial",
                }
            },
        )
    except ValueError as error:
        assert "systematic" in str(error)
    else:  # pragma: no cover - assertion guard
        raise AssertionError("Expected a mismatched resampling label to be rejected")


def test_forced_full_refit_comparison_records_mae_and_max_diff(tmp_path: Path) -> None:
    """Fixture-forced comparison must record measured MAE/max_diff, not nulls or fake zeros."""
    anchor = _anchor(tmp_path)
    poll = pl.DataFrame(
        {
            "race_id": ["RACE"],
            "option_id": ["D"],
            "pct": [65.0],
            "sample_size": [1000],
        }
    )
    # Near-identical full-refit draws so the comparison is expected to pass thresholds.
    full_refit_rows = []
    for draw_id in range(100):
        dem_share = 0.62 if draw_id < 40 else 0.63
        for option_id, share in (("D", dem_share), ("R", 1.0 - dem_share)):
            full_refit_rows.append(
                {
                    "draw_id": draw_id,
                    "race_id": "RACE",
                    "option_id": option_id,
                    "latent_share": share,
                    "latent_logit": 0.0,
                }
            )
    full_refit = pl.DataFrame(full_refit_rows)

    result = run_daily_update(
        anchor,
        "2026-05-09",
        {
            "daily_update": {
                "strategy": "reweighting",
                "minimum_ess_ratio": 0.01,
                "max_probability_mae_vs_refit": 0.05,
                "max_probability_diff_vs_refit": 0.10,
            }
        },
        new_polls=poll,
        full_refit_posterior=full_refit,
        full_refit_run_id="fixture-full-refit",
    )

    audit = result.diagnostics["update_vs_full_refit"]
    assert audit["comparison_executed"] is True
    assert audit["status"] in {"passed", "failed"}
    assert isinstance(audit["probability_mae_vs_full_refit"], float)
    assert isinstance(audit["probability_max_diff_vs_full_refit"], float)
    assert (
        result.diagnostics["probability_mae_vs_full_refit"]
        == audit["probability_mae_vs_full_refit"]
    )
    assert (
        result.diagnostics["probability_max_diff_vs_full_refit"]
        == audit["probability_max_diff_vs_full_refit"]
    )
    assert audit["full_refit_run_id"] == "fixture-full-refit"
    # Reweighting without an executed full-refit strategy remains honest.
    assert result.diagnostics["full_refit_executed"] is False
    disk = read_json(result.output_dir / "update_vs_full_refit_audit.json")
    assert disk["comparison_executed"] is True
    assert disk["probability_mae_vs_full_refit"] is not None


def test_forced_full_refit_comparison_fails_when_disagreement_is_large(tmp_path: Path) -> None:
    full_refit_rows = []
    for draw_id in range(100):
        for option_id, share in (("D", 0.10), ("R", 0.90)):
            full_refit_rows.append(
                {
                    "draw_id": draw_id,
                    "race_id": "RACE",
                    "option_id": option_id,
                    "latent_share": share,
                    "latent_logit": 0.0,
                }
            )
    result = run_daily_update(
        _anchor(tmp_path),
        "2026-05-09",
        {
            "daily_update": {
                "strategy": "reweighting",
                "minimum_ess_ratio": 0.01,
                "max_probability_mae_vs_refit": 0.005,
                "max_probability_diff_vs_refit": 0.02,
            }
        },
        new_polls=pl.DataFrame(
            {
                "race_id": ["RACE"],
                "option_id": ["D"],
                "pct": [65.0],
                "sample_size": [1000],
            }
        ),
        full_refit_posterior=pl.DataFrame(full_refit_rows),
        full_refit_run_id="fixture-divergent-refit",
    )

    audit = result.diagnostics["update_vs_full_refit"]
    assert audit["comparison_executed"] is True
    assert audit["status"] == "failed"
    assert audit["probability_mae_vs_full_refit"] > 0.02
    assert result.diagnostics["r15_evidence_complete"] is False


def test_compare_update_vs_full_refit_is_callable_directly() -> None:
    update = pl.DataFrame(
        {
            "race_id": ["RACE", "RACE"],
            "option_id": ["D", "R"],
            "latent_share": [0.55, 0.45],
        }
    )
    refit = pl.DataFrame(
        {
            "race_id": ["RACE", "RACE"],
            "option_id": ["D", "R"],
            "latent_share": [0.56, 0.44],
        }
    )
    audit = compare_update_vs_full_refit(
        update,
        refit,
        max_probability_mae=0.02,
        max_probability_diff=0.05,
        full_refit_run_id="direct",
    )
    assert audit["comparison_executed"] is True
    assert audit["status"] == "passed"
    assert abs(float(audit["probability_mae_vs_full_refit"]) - 0.01) < 1e-12
    assert abs(float(audit["probability_max_diff_vs_full_refit"]) - 0.01) < 1e-12
