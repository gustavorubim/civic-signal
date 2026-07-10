from __future__ import annotations

import json
from pathlib import Path

import polars as pl
from jsonschema import Draft202012Validator

from civic_signal.config import ProjectContext
from civic_signal.ingest import SyncRunner
from civic_signal.ingest.sources import SourceRegistry
from civic_signal.normalize import CuratedDataBuilder
from civic_signal.storage.io import read_json
from civic_signal.verification.data_audit import DataAuditRunner

ROOT = Path(__file__).resolve().parents[1]


def _context(tmp_path: Path) -> ProjectContext:
    return ProjectContext.create(
        root=ROOT,
        data_dir=tmp_path / "data",
        artifacts_dir=tmp_path / "artifacts",
    )


def test_source_priority_defaults_and_manifest_provenance(tmp_path: Path) -> None:
    ctx = _context(tmp_path)
    registry = SourceRegistry.from_context(ctx)
    assert {source.source_class for source in registry.sources} == {"fixture"}
    assert {source.priority for source in registry.sources} == {50}

    manifest = SyncRunner(ctx, registry=registry).run().manifest
    assert "source_priority" in manifest.columns
    assert manifest["source_priority"].unique().to_list() == [50]
    assert registry.by_table()["polls"].table == "polls"
    assert SourceRegistry._default_priority("official_public") == 400
    assert SourceRegistry._default_priority("production_web") == 300
    assert SourceRegistry._default_priority("production") == 250
    assert SourceRegistry._default_priority("synthetic") == 0
    assert SourceRegistry._default_priority("unclassified") == 100


def test_dedupe_is_class_ranked_and_independent_of_file_order() -> None:
    rows = [
        {
            "race_id": "R1",
            "option_id": "D",
            "name": "synthetic",
            "source_class": "synthetic",
            "source_priority": 9999,
            "_source_retrieved_at": "2026-05-10T00:00:00Z",
            "source_id": "z-synthetic",
            "source_hash": "3" * 64,
        },
        {
            "race_id": "R1",
            "option_id": "D",
            "name": "fixture",
            "source_class": "fixture",
            "source_priority": 9999,
            "_source_retrieved_at": "2026-05-10T00:00:00Z",
            "source_id": "y-fixture",
            "source_hash": "2" * 64,
        },
        {
            "race_id": "R1",
            "option_id": "D",
            "name": "lower-production",
            "source_class": "production_web",
            "source_priority": 100,
            "_source_retrieved_at": "2026-05-10T00:00:00Z",
            "source_id": "b-production",
            "source_hash": "1" * 64,
        },
        {
            "race_id": "R1",
            "option_id": "D",
            "name": "winner",
            "source_class": "official_public",
            "source_priority": 400,
            "_source_retrieved_at": "2026-05-01T00:00:00Z",
            "source_id": "a-official",
            "source_hash": "0" * 64,
        },
    ]
    forward = CuratedDataBuilder._dedupe(pl.DataFrame(rows), "options")
    reverse = CuratedDataBuilder._dedupe(pl.DataFrame(list(reversed(rows))), "options")

    assert forward["name"].to_list() == ["winner"]
    assert reverse.to_dicts() == forward.to_dicts()


def test_poll_entities_preserve_revisions_but_model_table_selects_one() -> None:
    common = {
        "poll_id": "provider-10-DEM",
        "survey_id": "provider-10",
        "question_id": "provider-10-q1",
        "race_id": "R1",
        "option_id": "D",
        "pollster": "Example Polling",
        "start_date": "2026-04-01",
        "end_date": "2026-04-03",
        "population": "lv",
        "sample_size": 800,
        "source_class": "production_web",
        "source_priority": 300,
        "source_id": "public-polls",
        "parser_version": "polls-v1",
        "available_at": "2026-04-04T00:00:00Z",
        "availability_basis": "source_record",
    }
    polls = CuratedDataBuilder._ensure_poll_identity(
        pl.DataFrame(
            [
                {
                    **common,
                    "pct": 48.0,
                    "source_hash": "a" * 64,
                    "_source_retrieved_at": "2026-04-04T00:00:00Z",
                },
                {
                    **common,
                    "pct": 49.0,
                    "source_hash": "b" * 64,
                    "_source_retrieved_at": "2026-04-05T00:00:00Z",
                },
            ]
        )
    )
    selected = CuratedDataBuilder._dedupe(polls, "polls")
    entities = CuratedDataBuilder._poll_entity_tables(polls)

    assert selected.height == 1
    assert selected["pct"].to_list() == [49.0]
    assert entities["poll_surveys"].height == 1
    assert entities["poll_questions"].height == 1
    assert entities["poll_revisions"].height == 2
    assert entities["poll_revisions"]["revision_id"].n_unique() == 2


def test_fixture_build_writes_canonical_poll_entities(tmp_path: Path) -> None:
    ctx = _context(tmp_path)
    SyncRunner(ctx).run()
    result = CuratedDataBuilder(ctx).run()

    assert {"poll_surveys", "poll_questions", "poll_revisions"}.issubset(result.tables)
    assert result.tables["poll_surveys"]["survey_id"].null_count() == 0
    assert result.tables["poll_questions"]["question_id"].null_count() == 0
    assert result.tables["poll_revisions"]["revision_id"].null_count() == 0
    for name in ("poll_surveys", "poll_questions", "poll_revisions"):
        assert (ctx.curated_dir / f"{name}.parquet").exists()


def test_canonical_entity_json_schemas_are_valid() -> None:
    schema_paths = [
        ROOT / "schemas/raw_contracts/source_snapshot.schema.json",
        *sorted((ROOT / "schemas/curated_tables").glob("*.schema.json")),
    ]
    expected = {
        "source_snapshot.schema.json",
        "poll_survey.schema.json",
        "poll_question.schema.json",
        "poll_revision.schema.json",
        "race.schema.json",
        "option.schema.json",
        "official_result.schema.json",
        "fundamental_snapshot.schema.json",
        "market_quote.schema.json",
    }
    assert expected.issubset({path.name for path in schema_paths})
    for path in schema_paths:
        Draft202012Validator.check_schema(json.loads(path.read_text(encoding="utf-8")))


def test_production_data_audit_fails_fixture_registry_with_reasons(tmp_path: Path) -> None:
    """Default fixture registry must never auto-pass production free-web exclusivity."""
    ctx = _context(tmp_path)
    SyncRunner(ctx).run()

    production = DataAuditRunner(ctx).verify(run_id="fixture-prod", profile="production")
    research = DataAuditRunner(ctx).verify(run_id="fixture-research", profile="research")

    assert production["passed"] is False
    assert production["exit_nonzero"] is True
    assert production["audit"]["status"] == "failed"
    assert production["audit"]["free_public_web_only"] is False
    assert production["audit"]["production_ready"] is False
    assert production["audit"]["fixture_source_count"] > 0
    assert production["audit"]["fixture_sources"]
    assert production["audit"]["nonfree_sources"]
    reasons = production["audit"]["reasons"]
    assert reasons
    assert any("fixture" in reason for reason in reasons)
    assert any("free_public_web_only is false" in reason for reason in reasons)

    registry_artifact = read_json(
        ctx.artifacts_dir / "data_audits/fixture-prod/source_registry_audit.json"
    )
    terms_artifact = read_json(ctx.artifacts_dir / "data_audits/fixture-prod/terms_audit.json")
    assert registry_artifact["free_public_web_only"] is False
    assert registry_artifact["reasons"] == reasons
    assert terms_artifact["free_public_web_only"] is False
    assert terms_artifact["reasons"] == reasons
    assert terms_artifact["fixture_sources"] == production["audit"]["fixture_sources"]

    # Research remains usable for local fixture development; production never auto-passes.
    assert research["passed"] is True
    assert research["audit"]["status"] == "ok"
    assert research["audit"]["free_public_web_only"] is False
    assert research["audit"]["reasons"]
    assert research["audit"]["production_ready"] is False


def test_production_data_audit_fails_nonfree_source_class_even_with_snapshots(
    tmp_path: Path,
) -> None:
    """Synthetic/non-free_public_web sources fail production even when snapshots look healthy."""
    ctx = ProjectContext.create(
        root=ROOT,
        sources_config="sources.yaml",
        data_dir=tmp_path / "data",
        artifacts_dir=tmp_path / "artifacts",
    )
    SyncRunner(ctx).run()
    # Force healthy fetch statuses so the free-web policy is the sole failure mode.
    manifest = pl.read_parquet(ctx.raw_dir / "source_manifest.parquet").with_columns(
        pl.lit("fetched").alias("status"),
        pl.lit("synthetic").alias("source_class"),
        pl.lit("fixture_local").alias("access_policy"),
    )
    manifest.write_parquet(ctx.raw_dir / "source_manifest.parquet")

    result = DataAuditRunner(ctx).verify(run_id="synthetic-prod", profile="production")

    assert result["passed"] is False
    assert result["audit"]["status"] == "failed"
    assert result["audit"]["free_public_web_only"] is False
    assert result["audit"]["fixture_sources"]
    assert any(
        "synthetic" in reason or "fixture" in reason for reason in result["audit"]["reasons"]
    )
    assert any("free_public_web" in reason for reason in result["audit"]["reasons"])
