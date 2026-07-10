from __future__ import annotations

import json
from dataclasses import replace
from datetime import date
from pathlib import Path

import polars as pl
import pytest

from civic_signal.config import ProjectContext
from civic_signal.features import FeatureBundle
from civic_signal.inference.state_space import build_state_space_data
from civic_signal.models.polling_bayes import BayesianPollingModel

ROOT = Path(__file__).resolve().parents[1]


def _context(tmp_path: Path) -> ProjectContext:
    return ProjectContext.create(
        root=ROOT,
        data_dir=tmp_path / "data",
        artifacts_dir=tmp_path / "artifacts",
    )


def _bundle(
    poll_rows: list[dict[str, object]],
    option_rows: list[dict[str, object]] | None = None,
) -> FeatureBundle:
    options = pl.DataFrame(
        option_rows
        or [
            {"race_id": "R1", "option_id": "D", "party": "DEM", "previous_vote_share": 0.5},
            {"race_id": "R1", "option_id": "R", "party": "REP", "previous_vote_share": 0.5},
        ]
    )
    catalog = pl.DataFrame(
        {
            "race_id": ["R1"],
            "cycle": [2026],
            "office_type": ["senate"],
            "geography": ["GA"],
            "election_date": ["2026-11-03"],
            "tier": ["A"],
        }
    )
    empty = pl.DataFrame(schema={"race_id": pl.Utf8})
    return FeatureBundle(
        races=catalog,
        options=options,
        polls=pl.DataFrame(poll_rows),
        markets=empty,
        public_signals=empty,
        fundamentals=empty,
        results=empty,
        backtest_predictions=empty,
        race_catalog=catalog,
    )


def _question(
    dem: float,
    rep: float,
    *,
    methodology: str = "live_phone",
    sponsor: str = "nonpartisan",
    duplicate_dem: bool = False,
) -> list[dict[str, object]]:
    rows = []
    for option_id, pct in (("D", dem), ("R", rep)):
        rows.append(
            {
                "poll_id": f"P1-{option_id}",
                "survey_id": "S1",
                "question_id": "Q1",
                "race_id": "R1",
                "option_id": option_id,
                "pollster": "Example",
                "start_date": "2026-04-01",
                "end_date": "2026-04-03",
                "population": "lv",
                "sample_size": 1000,
                "sponsor_class": sponsor,
                "methodology": methodology,
                "pct": pct,
                "source_hash": "a" * 64,
            }
        )
    if duplicate_dem:
        rows.append({**rows[0], "poll_id": "P1-D-duplicate"})
    return rows


def _model(ctx: ProjectContext, **observation_overrides: object) -> BayesianPollingModel:
    config = json.loads(json.dumps(ctx.read_yaml("model.yaml")))
    config["_bayesian_backend"] = "analytic"
    config["bayesian"]["posterior_draw_count"] = 500
    config["bayesian"]["observation"].update(observation_overrides)
    return BayesianPollingModel(config, as_of="2026-05-08")


def test_duplicate_option_rows_become_one_shared_question_contrast(tmp_path: Path) -> None:
    ctx = _context(tmp_path)
    bundle = _bundle(_question(52, 48, duplicate_dem=True))
    model = _model(ctx)
    estimates = model.run(bundle)
    diagnostics = model.diagnostics(bundle)["poll_likelihood"]
    draws = model.posterior_draws(bundle)

    assert diagnostics["question_count"] == 1
    assert diagnostics["duplicate_question_option_rows_removed"] == 1
    assert diagnostics["analytic_binary_joint_race_count"] == 1
    assert diagnostics["analytic_binary_option_rows_independent"] is False
    assert estimates.height == 2
    sums = draws.group_by("draw_id").agg(pl.col("latent_share").sum().alias("total"))
    assert sums["total"].to_list() == pytest.approx([1.0] * sums.height)


def test_nuts_state_space_consumes_binary_rows_once_as_one_margin(tmp_path: Path) -> None:
    ctx = _context(tmp_path)
    bundle = _bundle(_question(52, 48, duplicate_dem=True))
    model = _model(ctx)
    prepared, audit = model._prepare_likelihood_polls(
        bundle, bundle.polls, split_overlap_precision=False
    )
    prepared = prepared.with_columns(
        pl.col("start_date").str.to_date(),
        pl.col("end_date").str.to_date(),
    )
    data = build_state_space_data(
        replace(bundle, polls=prepared),
        as_of="2026-05-08",
        office_type=None,
        election_day_by_race={"R1": date(2026, 11, 3)},
    )

    assert audit["paired_binary_question_count"] == 1
    assert data.poll_logit_y.size == 2  # raw option rows retained for lineage
    assert data.margin_poll_y.size == 1  # one independent contrast reaches NUTS
    assert data.margin_poll_kappa.size == 1
    assert data.legacy_poll_indices.size == 0  # neither option row is fitted again


def test_same_question_id_in_different_surveys_remains_distinct(tmp_path: Path) -> None:
    ctx = _context(tmp_path)
    first = _question(52, 48)
    second = [
        {
            **row,
            "survey_id": "S2",
            "poll_id": str(row["poll_id"]).replace("P1", "P2"),
            "source_hash": "b" * 64,
        }
        for row in _question(51, 49)
    ]
    bundle = _bundle([*first, *second])
    model = _model(ctx)
    prepared, audit = model._prepare_likelihood_polls(
        bundle, bundle.polls, split_overlap_precision=True
    )

    assert audit["question_count"] == 2
    assert audit["paired_binary_question_count"] == 2
    assert prepared["_likelihood_question_id"].n_unique() == 2


def test_incomplete_binary_question_is_not_fitted_as_independent_option(
    tmp_path: Path,
) -> None:
    ctx = _context(tmp_path)
    bundle = _bundle(_question(52, 48)[:1])
    model = _model(ctx)
    prepared, audit = model._prepare_likelihood_polls(
        bundle, bundle.polls, split_overlap_precision=True
    )

    assert prepared.is_empty()
    assert audit["incomplete_binary_question_count"] == 1
    assert model.run(bundle).is_empty()


def test_mode_and_sponsor_effects_shift_center_and_inflate_uncertainty(tmp_path: Path) -> None:
    ctx = _context(tmp_path)
    bundle = _bundle(_question(56, 44, methodology="online", sponsor="partisan"))
    neutral = _model(
        ctx,
        mode_bias_share={"online": 0.0},
        sponsor_bias_share={"partisan": 0.0},
        mode_nonsampling_sd={"online": 0.0},
        sponsor_nonsampling_sd={"partisan": 0.0},
    ).run(bundle)
    adjusted = _model(
        ctx,
        mode_bias_share={"online": 0.02},
        sponsor_bias_share={"partisan": 0.01},
        mode_nonsampling_sd={"online": 0.03},
        sponsor_nonsampling_sd={"partisan": 0.03},
    ).run(bundle)
    neutral_d = neutral.filter(pl.col("option_id") == "D").row(0, named=True)
    adjusted_d = adjusted.filter(pl.col("option_id") == "D").row(0, named=True)
    assert adjusted_d["vote_share"] < neutral_d["vote_share"]
    assert adjusted_d["uncertainty"] >= neutral_d["uncertainty"]


def test_undecided_other_is_proportionally_allocated_with_uncertainty(tmp_path: Path) -> None:
    ctx = _context(tmp_path)
    bundle = _bundle(_question(45, 45))
    model = _model(ctx)
    prepared, audit = model._prepare_likelihood_polls(
        bundle, bundle.polls, split_overlap_precision=True
    )
    assert prepared.sort("option_id")["pct"].to_list() == pytest.approx([50.0, 50.0])
    assert audit["mean_unallocated_share"] == pytest.approx(0.10)
    assert prepared["_likelihood_nonsampling_sd"].min() > model.min_nonsampling_error


def test_multi_option_polling_estimand_is_withheld(tmp_path: Path) -> None:
    ctx = _context(tmp_path)
    options = [
        {"race_id": "R1", "option_id": option, "party": party, "previous_vote_share": 1 / 3}
        for option, party in (("D", "DEM"), ("R", "REP"), ("I", "IND"))
    ]
    polls = [
        *_question(45, 40),
        {
            **_question(45, 40)[0],
            "poll_id": "P1-I",
            "option_id": "I",
            "pct": 15,
        },
    ]
    bundle = _bundle(polls, options)
    model = _model(ctx)
    assert model.run(bundle).is_empty()
    diagnostics = model.diagnostics(bundle)["poll_likelihood"]
    assert diagnostics["unsupported_multi_option_race_count"] == 1
    assert diagnostics["multi_option_estimand_status"].startswith("withheld")
    assert model.posterior_draws(bundle).is_empty()


def test_binary_label_symmetry_under_party_and_option_swap(tmp_path: Path) -> None:
    ctx = _context(tmp_path)
    original = _bundle(_question(54, 46))
    swapped_options = original.options.with_columns(
        pl.when(pl.col("option_id") == "D")
        .then(pl.lit("REP"))
        .otherwise(pl.lit("DEM"))
        .alias("party")
    )
    swapped_polls = original.polls.with_columns(
        pl.when(pl.col("option_id") == "D").then(pl.lit(46.0)).otherwise(pl.lit(54.0)).alias("pct")
    )
    swapped = replace(original, options=swapped_options, polls=swapped_polls)
    first = _model(ctx).run(original)
    second = _model(ctx).run(swapped)
    original_dem = first.filter(pl.col("option_id") == "D")["vote_share"].item()
    swapped_dem = second.filter(pl.col("option_id") == "R")["vote_share"].item()
    assert original_dem == pytest.approx(swapped_dem, abs=0.01)
