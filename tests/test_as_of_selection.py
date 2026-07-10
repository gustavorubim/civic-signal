from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import polars as pl

from civic_signal.config import ProjectContext
from civic_signal.features import (
    FeatureBundle,
    TierAssessor,
    filter_bundle_by_date,
    select_latest_eligible_snapshots,
    subset_bundle,
)
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
            "race_id": ["r1"],
            "election_date": ["2026-11-03"],
        }
    )
    polls = pl.DataFrame(
        {
            "race_id": ["r1", "r1", "r1"],
            "poll_id": ["p1", "p1", "p1"],
            "option_id": ["dem", "dem", "dem"],
            "pollster": ["A", "A", "A"],
            "end_date": ["2026-05-01", "2026-05-01", "2026-05-01"],
            "published_at": ["2026-05-02", "2026-05-03", "2026-05-09"],
            "available_at": ["2026-05-02", "2026-05-03", "2026-05-09"],
            "availability_basis": ["source_record"] * 3,
            "revision_id": ["1", "2", "3"],
            "pct": [48.0, 49.0, 99.0],
            "source_id": ["polls"] * 3,
            "source_hash": ["h1", "h2", "h3"],
        }
    )
    fundamentals = pl.DataFrame(
        {
            "race_id": ["r1"],
            "as_of": ["2026-05-01"],
            "available_at": ["2026-05-01"],
            "availability_basis": ["source_record"],
            "economic_index": [0.2],
            "source_id": ["macro"],
            "source_hash": ["fh1"],
        }
    )
    race_only_schema = {"race_id": pl.String}
    return FeatureBundle(
        races=races,
        options=pl.DataFrame({"race_id": ["r1"], "option_id": ["dem"]}),
        polls=polls,
        markets=pl.DataFrame(schema=race_only_schema),
        public_signals=pl.DataFrame(schema=race_only_schema),
        fundamentals=fundamentals,
        results=pl.DataFrame(schema=race_only_schema),
        backtest_predictions=pl.DataFrame(schema=race_only_schema),
        race_catalog=races,
    )


def _selector(bundle: FeatureBundle) -> FeatureBundle:
    selected = filter_bundle_by_date(bundle, "2026-05-08")
    tier_config = {
        "tier_a": {
            "min_polls": 1,
            "min_pollsters": 1,
            "min_market_quotes": 1,
            "min_fundamental_rows": 1,
        },
        "tier_b": {"min_any_signal_rows": 1},
        "tier_c": {"reason": "sparse"},
    }
    catalog = TierAssessor(tier_config).assign(
        selected.races,
        selected.polls,
        selected.markets,
        selected.fundamentals,
        selected.public_signals,
    )
    return subset_bundle(selected, catalog)


def _literal_forecast_fingerprint(bundle: FeatureBundle) -> str:
    payload = {
        "poll_mean": bundle.polls["pct"].mean() if not bundle.polls.is_empty() else None,
        "fundamental_mean": (
            bundle.fundamentals["economic_index"].mean()
            if not bundle.fundamentals.is_empty()
            else None
        ),
        "tiers": bundle.race_catalog.select(["race_id", "tier"]).to_dicts(),
    }
    value = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(value.encode()).hexdigest()


def test_snapshot_selection_filters_before_latest_revision_and_is_order_invariant() -> None:
    polls = _bundle().polls

    selected = select_latest_eligible_snapshots(polls, "polls", "2026-05-08")
    reversed_selected = select_latest_eligible_snapshots(polls.reverse(), "polls", "2026-05-08")

    assert selected.height == 1
    assert selected["revision_id"].to_list() == ["2"]
    assert selected["pct"].to_list() == [49.0]
    assert selected.to_dicts() == reversed_selected.to_dicts()


def test_null_primary_question_id_coalesces_to_distinct_poll_ids() -> None:
    polls = pl.DataFrame(
        {
            "race_id": ["r1", "r1"],
            "question_id": [None, None],
            "poll_id": ["p1", "p2"],
            "option_id": ["dem", "dem"],
            "end_date": ["2026-05-01", "2026-05-01"],
            "available_at": ["2026-05-02", "2026-05-02"],
            "pct": [48.0, 52.0],
        }
    )

    selected = select_latest_eligible_snapshots(polls, "polls", "2026-05-08")

    assert selected.height == 2
    assert set(selected["snapshot_identity"].to_list()) == {"poll_id:p1", "poll_id:p2"}


def test_selected_lineage_records_selection_key_revision_and_predicate() -> None:
    selected = _selector(_bundle())

    lineage = build_selected_feature_lineage(selected, "2026-05-08")

    assert {
        "selection_key",
        "snapshot_id",
        "revision_id",
        "observed_at",
        "published_at",
        "selection_predicate",
    }.issubset(lineage.columns)
    poll = lineage.filter(pl.col("table") == "polls").row(0, named=True)
    assert poll["revision_id"] == "2"
    assert poll["selection_key"] == "r1|poll_id:p1|dem"
    assert "available_at<=as_of" in poll["selection_predicate"]


def test_adversarial_canary_proves_features_tiers_and_literal_forecast_unchanged() -> None:
    result = run_adversarial_time_travel_canary(
        _bundle(),
        as_of="2026-05-08",
        selector=_selector,
        forecast_fingerprint=_literal_forecast_fingerprint,
        forecast_scope="literal_fixture_forecast",
    )

    assert result["passed"] is True
    assert result["injected_tables"] == ["polls", "fundamentals"]
    assert result["selected_features_unchanged"] is True
    assert result["tiers_unchanged"] is True
    assert result["forecast_fingerprint_unchanged"] is True


def test_adversarial_canary_fails_selector_that_leaks_future_revisions() -> None:
    def broken_selector(bundle: FeatureBundle) -> FeatureBundle:
        catalog = TierAssessor(
            {
                "tier_a": {"min_polls": 1, "min_pollsters": 1},
                "tier_b": {"min_any_signal_rows": 1},
                "tier_c": {"reason": "sparse"},
            }
        ).assign(
            bundle.races,
            bundle.polls,
            bundle.markets,
            bundle.fundamentals,
            bundle.public_signals,
        )
        return replace(bundle, race_catalog=catalog)

    result = run_adversarial_time_travel_canary(
        _bundle(),
        as_of="2026-05-08",
        selector=broken_selector,
        forecast_fingerprint=_literal_forecast_fingerprint,
        forecast_scope="literal_fixture_forecast",
    )

    assert result["passed"] is False
    assert result["selected_features_unchanged"] is False
    assert result["forecast_fingerprint_unchanged"] is False


def test_as_of_runner_writes_lineage_and_canary_artifacts(tmp_path: Path) -> None:
    """AsOfVerificationRunner persists selected lineage + time-travel canary evidence."""
    ctx = ProjectContext.create(
        root=ROOT,
        data_dir=tmp_path / "data",
        artifacts_dir=tmp_path / "artifacts",
    )
    selected = _selector(_bundle())
    lineage = build_selected_feature_lineage(selected, "2026-05-08")
    canary = run_adversarial_time_travel_canary(
        _bundle(),
        as_of="2026-05-08",
        selector=_selector,
        forecast_fingerprint=_literal_forecast_fingerprint,
        forecast_scope="literal_fixture_forecast",
    )
    run_dir = ctx.artifacts_dir / "runs" / "asof-ok"
    run_dir.mkdir(parents=True)
    (run_dir / "run_manifest.json").write_text(
        json.dumps({"as_of": "2026-05-08"}), encoding="utf-8"
    )
    lineage.write_parquet(run_dir / "selected_feature_lineage.parquet")

    payload = AsOfVerificationRunner(ctx).verify(
        run_id="asof-ok",
        as_of="2026-05-08",
        write_run_artifact=True,
        canary_evidence=canary,
    )

    audit_dir = ctx.artifacts_dir / "as_of_audits" / "asof-ok"
    assert payload["passed"] is True
    assert (audit_dir / "selected_feature_lineage.parquet").exists()
    assert (audit_dir / "as_of_audit.json").exists()
    assert (audit_dir / "time_travel_canaries.json").exists()
    assert (run_dir / "as_of_audit.json").exists()
    assert (run_dir / "time_travel_canaries.json").exists()

    canary_artifact = read_json(audit_dir / "time_travel_canaries.json")
    audit = read_json(audit_dir / "as_of_audit.json")
    assert canary_artifact["time_travel_canaries_passed"] is True
    assert canary_artifact["full_path_canary_provided"] is True
    assert canary_artifact["full_path_canary"]["passed"] is True
    assert audit["status"] == "ok"
    assert audit["reasons"] == []
    assert audit["future_eligible_rows"] == 0
    assert pl.read_parquet(audit_dir / "selected_feature_lineage.parquet").height == lineage.height


def test_as_of_runner_fails_when_future_rows_present_in_lineage(tmp_path: Path) -> None:
    """Future-row injection into selected lineage must fail verification."""
    ctx = ProjectContext.create(
        root=ROOT,
        data_dir=tmp_path / "data",
        artifacts_dir=tmp_path / "artifacts",
    )
    selected = _selector(_bundle())
    lineage = build_selected_feature_lineage(selected, "2026-05-08")
    future = lineage.head(1).with_columns(
        pl.lit("2026-05-09").alias("available_at"),
        pl.lit("future|leaked").alias("row_key"),
        pl.lit("future|selection").alias("selection_key"),
    )
    poisoned = pl.concat([lineage, future], how="diagonal_relaxed")
    run_dir = ctx.artifacts_dir / "runs" / "asof-future"
    run_dir.mkdir(parents=True)
    (run_dir / "run_manifest.json").write_text(
        json.dumps({"as_of": "2026-05-08"}), encoding="utf-8"
    )
    poisoned.write_parquet(run_dir / "selected_feature_lineage.parquet")

    payload = AsOfVerificationRunner(ctx).verify(run_id="asof-future", as_of="2026-05-08")

    assert payload["passed"] is False
    assert payload["exit_nonzero"] is True
    assert payload["audit"]["status"] == "failed"
    assert payload["audit"]["future_eligible_rows"] == 1
    assert any("future-eligible" in reason for reason in payload["audit"]["reasons"])
    canary_artifact = read_json(
        ctx.artifacts_dir / "as_of_audits/asof-future/time_travel_canaries.json"
    )
    assert canary_artifact["future_eligible_rows"] == 1
    assert canary_artifact["status"] == "failed"


def test_filter_bundle_by_date_rejects_future_available_revisions() -> None:
    """Shipped selector entry point must drop future revisions before latest selection."""
    selected = filter_bundle_by_date(_bundle(), "2026-05-08")
    assert selected.polls.height == 1
    assert selected.polls["revision_id"].to_list() == ["2"]
    assert selected.polls["pct"].to_list() == [49.0]
    assert "99.0" not in {str(value) for value in selected.polls["pct"].to_list()}
