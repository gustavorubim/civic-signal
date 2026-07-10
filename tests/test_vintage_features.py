from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import polars as pl
import pytest

from civic_signal.config import ProjectContext
from civic_signal.features import FeatureBundle, filter_bundle_by_date
from civic_signal.features.slicing import feature_vintage_lineage_summary
from civic_signal.models.fundamentals import FundamentalsModel
from civic_signal.storage.io import read_json
from civic_signal.verification.as_of import (
    AsOfVerificationRunner,
    build_selected_feature_lineage,
    run_adversarial_time_travel_canary,
)

ROOT = Path(__file__).resolve().parents[1]


def _bundle() -> FeatureBundle:
    races = pl.DataFrame(
        {
            "race_id": ["RACE"],
            "election_date": ["2026-11-03"],
            "office_type": ["senate"],
        }
    )
    options = pl.DataFrame(
        {
            "race_id": ["RACE", "RACE"],
            "option_id": ["D", "R"],
            "party": ["DEM", "REP"],
            "incumbent": [False, True],
            "previous_vote_share": [0.5, 0.5],
            "fundraising_usd": [9_999.0, 9_999.0],
        }
    )
    rows = [
        {
            "race_id": "RACE",
            "feature_id": None,
            "series_id": "ECON",
            "feature_type": "macro",
            "observed_at": "2026-04-01",
            "published_at": "2026-04-02",
            "available_at": "2026-04-02",
            "availability_basis": "source_record",
            "revision_id": "1",
            "economic_index": 0.1,
            "source_id": "macro",
            "source_hash": "macro-1",
        },
        {
            "race_id": "RACE",
            "feature_id": None,
            "series_id": "ECON",
            "feature_type": "macro",
            "observed_at": "2026-04-01",
            "published_at": "2026-05-05",
            "available_at": "2026-05-05",
            "availability_basis": "source_record",
            "revision_id": "2",
            "economic_index": 0.2,
            "source_id": "macro",
            "source_hash": "macro-2",
        },
        {
            "race_id": "RACE",
            "feature_id": None,
            "series_id": "ECON",
            "feature_type": "macro",
            "observed_at": "2026-04-01",
            "published_at": "2026-05-09",
            "available_at": "2026-05-09",
            "availability_basis": "source_record",
            "revision_id": "3",
            "economic_index": 9.0,
            "source_id": "macro",
            "source_hash": "macro-future",
        },
        {
            "race_id": "RACE",
            "feature_id": "FIN-D",
            "series_id": None,
            "option_id": "D",
            "feature_type": "finance",
            "observed_at": "2026-05-01",
            "published_at": "2026-05-05",
            "available_at": "2026-05-05",
            "availability_basis": "source_record",
            "revision_id": "1",
            "fundraising_usd": 100.0,
            "source_id": "fec",
            "source_hash": "fin-d-1",
        },
        {
            "race_id": "RACE",
            "feature_id": "FIN-D",
            "series_id": None,
            "option_id": "D",
            "feature_type": "finance",
            "observed_at": "2026-05-01",
            "published_at": "2026-05-09",
            "available_at": "2026-05-09",
            "availability_basis": "source_record",
            "revision_id": "2",
            "fundraising_usd": 99_999.0,
            "source_id": "fec",
            "source_hash": "fin-d-future",
        },
        {
            "race_id": "RACE",
            "feature_id": "FIN-R",
            "series_id": None,
            "option_id": "R",
            "feature_type": "finance",
            "observed_at": "2026-05-01",
            "published_at": "2026-05-05",
            "available_at": "2026-05-05",
            "availability_basis": "source_record",
            "revision_id": "1",
            "fundraising_usd": 200.0,
            "source_id": "fec",
            "source_hash": "fin-r-1",
        },
        {
            "race_id": "RACE",
            "feature_id": None,
            "series_id": None,
            "rating_id": "RATE-D",
            "option_id": "D",
            "feature_type": "rating",
            "observed_at": "2026-05-01",
            "published_at": "2026-05-06",
            "available_at": "2026-05-06",
            "availability_basis": "source_record",
            "revision_id": "1",
            "rating_score": 4.0,
            "source_id": "ratings",
            "source_hash": "rating-d-1",
        },
    ]
    empty = pl.DataFrame(schema={"race_id": pl.String})
    return FeatureBundle(
        races=races,
        options=options,
        polls=empty,
        markets=empty,
        public_signals=empty,
        fundamentals=pl.DataFrame(rows),
        results=empty,
        backtest_predictions=empty,
        race_catalog=races,
    )


def test_vintage_selection_filters_before_revision_and_overlays_option_features() -> None:
    selected = filter_bundle_by_date(_bundle(), "2026-05-08")

    macro = selected.fundamentals.filter(pl.col("series_id") == "ECON")
    assert macro["revision_id"].to_list() == ["2"]
    assert macro["economic_index"].to_list() == [0.2]
    assert macro["economic_index_incumbent_relative"].to_list() == [-0.2]
    assert macro["incumbent_relative_sign_applied"].to_list() == [True]
    option_values = selected.options.sort("option_id").select(
        "option_id",
        "fundraising_usd",
        "fundraising_vintage_applied",
        "rating_score",
        "rating_vintage_applied",
    )
    assert option_values["fundraising_usd"].to_list() == [100.0, 200.0]
    assert option_values["fundraising_vintage_applied"].to_list() == [True, True]
    assert option_values["rating_score"].to_list() == [4.0, None]
    assert option_values["rating_vintage_applied"].to_list() == [True, False]


def test_hostile_future_macro_finance_and_rating_vintages_leave_features_unchanged() -> None:
    bundle = _bundle()
    baseline = filter_bundle_by_date(bundle, "2026-05-08")
    hostile = bundle.fundamentals.head(3).with_columns(
        pl.lit("2026-05-09").alias("published_at"),
        pl.lit("2026-05-09").alias("available_at"),
        pl.lit("999").alias("revision_id"),
        pl.lit(999_999.0).alias("economic_index"),
        pl.lit(999_999.0).alias("fundraising_usd"),
        pl.lit(999_999.0).alias("rating_score"),
    )
    augmented = replace(
        bundle,
        fundamentals=pl.concat([bundle.fundamentals, hostile], how="diagonal_relaxed"),
    )
    counterfactual = filter_bundle_by_date(augmented, "2026-05-08")

    assert baseline.fundamentals.to_dicts() == counterfactual.fundamentals.to_dicts()
    assert baseline.options.to_dicts() == counterfactual.options.to_dicts()


def test_coalesced_snapshot_identity_keeps_null_primary_series_distinct() -> None:
    bundle = _bundle()
    second_series = bundle.fundamentals.head(1).with_columns(
        pl.lit("LABOR").alias("series_id"),
        pl.lit("labor-1").alias("source_hash"),
    )
    selected = filter_bundle_by_date(
        replace(
            bundle,
            fundamentals=pl.concat([bundle.fundamentals, second_series], how="diagonal_relaxed"),
        ),
        "2026-05-08",
    )

    macro = selected.fundamentals.filter(pl.col("feature_type") == "macro")
    assert set(macro["snapshot_identity"].to_list()) == {
        "series_id:ECON",
        "series_id:LABOR",
    }


def test_fundamentals_model_consumes_incumbent_relative_economy_once() -> None:
    selected = filter_bundle_by_date(_bundle(), "2026-05-08")
    model = FundamentalsModel()
    model.coefficients = {key: 0.0 for key in model.DEFAULT_COEFFICIENTS}
    model.coefficients["economic_index"] = 0.25

    macro = selected.fundamentals.filter(pl.col("series_id") == "ECON").row(0, named=True)
    shares = model._raw_shares(selected.options, macro)

    assert shares["R"] > 0.5
    assert shares["D"] < 0.5
    assert shares["R"] - 0.5 == pytest.approx(0.5 - shares["D"])


def test_model_coalesces_separate_selected_macro_features_without_future_revision() -> None:
    bundle = _bundle()
    separate = pl.DataFrame(
        [
            {
                "race_id": "RACE",
                "feature_id": "PARTISAN",
                "feature_type": "macro",
                "observed_at": "2026-04-01",
                "published_at": "2026-04-02",
                "available_at": "2026-04-02",
                "revision_id": "1",
                "partisan_lean": 3.0,
                "source_hash": "partisan-1",
            },
            {
                "race_id": "RACE",
                "feature_id": "DEMOGRAPHIC",
                "feature_type": "macro",
                "observed_at": "2026-04-01",
                "published_at": "2026-04-03",
                "available_at": "2026-04-03",
                "revision_id": "1",
                "demographic_turnout_index": 1.5,
                "source_hash": "demographic-1",
            },
            {
                "race_id": "RACE",
                "feature_id": "PARTISAN",
                "feature_type": "macro",
                "observed_at": "2026-04-01",
                "published_at": "2026-05-09",
                "available_at": "2026-05-09",
                "revision_id": "2",
                "partisan_lean": 999.0,
                "source_hash": "partisan-future",
            },
        ]
    )
    selected = filter_bundle_by_date(
        replace(
            bundle,
            fundamentals=pl.concat([bundle.fundamentals, separate], how="diagonal_relaxed"),
        ),
        "2026-05-08",
    )

    combined = FundamentalsModel._fundamentals_by_race(selected.fundamentals)["RACE"]
    assert combined["economic_index"] == 0.2
    assert combined["partisan_lean"] == 3.0
    assert combined["demographic_turnout_index"] == 1.5


def test_open_seat_requires_explicit_incumbent_party_or_neutralizes_economy() -> None:
    bundle = _bundle()
    open_options = bundle.options.with_columns(pl.lit(False).alias("incumbent"))
    selected = filter_bundle_by_date(replace(bundle, options=open_options), "2026-05-08")
    macro = selected.fundamentals.filter(pl.col("series_id") == "ECON")
    assert macro["incumbent_relative_sign_applied"].to_list() == [False]
    assert macro["economic_index_incumbent_relative"].to_list() == [0.0]

    explicit = bundle.fundamentals.with_columns(pl.lit("REP").alias("incumbent_party"))
    selected_explicit = filter_bundle_by_date(
        replace(bundle, options=open_options, fundamentals=explicit), "2026-05-08"
    )
    explicit_macro = selected_explicit.fundamentals.filter(pl.col("series_id") == "ECON")
    assert explicit_macro["incumbent_relative_sign_applied"].to_list() == [True]
    assert explicit_macro["economic_index_incumbent_relative"].to_list() == [-0.2]


def test_vintage_lineage_and_selected_lineage_record_actual_observation_time() -> None:
    selected = filter_bundle_by_date(_bundle(), "2026-05-08")
    summary = feature_vintage_lineage_summary(
        selected.fundamentals,
        {"passed": True, "injected_tables": ["fundamentals"]},
    )
    lineage = build_selected_feature_lineage(selected, "2026-05-08")

    assert summary["incumbent_relative_sign"] is True
    assert summary["revised_macro_in_early_fold"] is False
    assert summary["end_of_cycle_finance_in_early_fold"] is False
    assert summary["revised_rating_in_early_fold"] is False
    macro = lineage.filter(pl.col("snapshot_id").str.contains("series_id:ECON"))
    assert macro["observed_at"].to_list() == ["2026-04-01"]
    assert "observed_at<=as_of" in macro["selection_predicate"].item()


def test_vintage_selected_lineage_survives_as_of_runner_and_rejects_future_injection(
    tmp_path: Path,
) -> None:
    """filter_bundle_by_date lineage is durable; future vintage injection fails as-of verify."""
    selected = filter_bundle_by_date(_bundle(), "2026-05-08")
    lineage = build_selected_feature_lineage(selected, "2026-05-08")

    def _vintage_fingerprint(bundle: FeatureBundle) -> str:
        frame = bundle.fundamentals
        columns = [
            column
            for column in (
                "feature_id",
                "series_id",
                "option_id",
                "revision_id",
                "economic_index",
                "fundraising_usd",
                "rating_score",
            )
            if column in frame.columns
        ]
        rows = frame.select(columns).sort(columns).to_dicts() if columns else []
        return json.dumps(rows, sort_keys=True, default=str)

    canary = run_adversarial_time_travel_canary(
        _bundle(),
        as_of="2026-05-08",
        selector=lambda bundle: filter_bundle_by_date(bundle, "2026-05-08"),
        forecast_fingerprint=_vintage_fingerprint,
        forecast_scope="vintage_fixture_forecast",
    )
    assert canary["passed"] is True

    ctx = ProjectContext.create(
        root=ROOT,
        data_dir=tmp_path / "data",
        artifacts_dir=tmp_path / "artifacts",
    )
    run_dir = ctx.artifacts_dir / "runs" / "vintage-asof"
    run_dir.mkdir(parents=True)
    (run_dir / "run_manifest.json").write_text(
        json.dumps({"as_of": "2026-05-08"}), encoding="utf-8"
    )
    lineage.write_parquet(run_dir / "selected_feature_lineage.parquet")

    ok = AsOfVerificationRunner(ctx).verify(
        run_id="vintage-asof",
        write_run_artifact=True,
        canary_evidence=canary,
    )
    assert ok["passed"] is True
    lineage_path = ctx.artifacts_dir / "as_of_audits/vintage-asof/selected_feature_lineage.parquet"
    canary_path = ctx.artifacts_dir / "as_of_audits/vintage-asof/time_travel_canaries.json"
    assert lineage_path.exists()
    assert canary_path.exists()
    canary_artifact = read_json(
        ctx.artifacts_dir / "as_of_audits/vintage-asof/time_travel_canaries.json"
    )
    assert canary_artifact["time_travel_canaries_passed"] is True

    future = lineage.head(1).with_columns(
        pl.lit("2026-05-09").alias("available_at"),
        pl.lit("future-vintage").alias("row_key"),
        pl.lit("future-vintage-key").alias("selection_key"),
    )
    poisoned_dir = ctx.artifacts_dir / "runs" / "vintage-future"
    poisoned_dir.mkdir(parents=True)
    (poisoned_dir / "run_manifest.json").write_text(
        json.dumps({"as_of": "2026-05-08"}), encoding="utf-8"
    )
    pl.concat([lineage, future], how="diagonal_relaxed").write_parquet(
        poisoned_dir / "selected_feature_lineage.parquet"
    )
    failed = AsOfVerificationRunner(ctx).verify(run_id="vintage-future")
    assert failed["passed"] is False
    assert failed["audit"]["future_eligible_rows"] == 1
    assert any("future-eligible" in reason for reason in failed["audit"]["reasons"])
