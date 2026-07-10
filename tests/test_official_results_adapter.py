from __future__ import annotations

import json
import urllib.error
from pathlib import Path

import polars as pl
import pytest
import yaml
from jsonschema import Draft202012Validator

from civic_signal.config import ProjectContext
from civic_signal.ingest import SourceDefinition, SourceRegistry, SourceSyncError, SyncRunner
from civic_signal.normalize import CuratedDataBuilder
from civic_signal.verification.data_audit import DataAuditRunner

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_CSV = ROOT / "tests" / "fixtures" / "medsl_house_2018_contract.csv"
PINNED_URL = (
    "https://raw.githubusercontent.com/MEDSL/constituency-returns/"
    "fe67c056502fc09ddb1ace2ff8f87c53233a744e/1976-2018-house.csv"
)
REQUIRED_COLUMNS = [
    "year",
    "state_po",
    "office",
    "district",
    "stage",
    "candidate",
    "party",
    "writein",
    "mode",
    "candidatevotes",
    "totalvotes",
    "unofficial",
    "version",
]


def _source(source_id: str, table: str, parser_version: str) -> dict[str, object]:
    return {
        "id": source_id,
        "table": table,
        "type": "http_csv",
        "path": "medsl_clerk_house_1976_2018.csv",
        "parser_version": parser_version,
        "license": (
            "Underlying returns are U.S. House Office of the Clerk federal public records; "
            "MEDSL requests citation and publishes no explicit data-license file."
        ),
        "terms_url": "https://github.com/MEDSL/constituency-returns#readme",
        "citation": "MIT Election Data and Science Lab, Constituency-Level Election Returns.",
        "auth_mode": "public",
        "source_class": "official_public",
        "access_policy": "free_public_web",
        "terms_status": "reviewed_for_use",
        "priority": 400,
        "url": PINNED_URL,
        "parser_args": {"cycles": [2018], "required_columns": REQUIRED_COLUMNS},
    }


def _context(tmp_path: Path, sources: list[dict[str, object]]) -> ProjectContext:
    config_dir = tmp_path / "configs"
    config_dir.mkdir(parents=True)
    (config_dir / "official.yaml").write_text(
        yaml.safe_dump({"sources": sources}, sort_keys=False), encoding="utf-8"
    )
    return ProjectContext.create(
        root=ROOT,
        config_dir=config_dir,
        sources_config="official.yaml",
        data_dir=tmp_path / "data",
        artifacts_dir=tmp_path / "artifacts",
    )


def _results_definition() -> SourceDefinition:
    item = _source(
        "medsl_clerk_house_results_1976_2018",
        "results",
        "medsl-house-constituency-results-v1",
    )
    return SourceDefinition(
        id=str(item["id"]),
        table=str(item["table"]),
        type=str(item["type"]),
        path=Path(str(item["path"])),
        parser_version=str(item["parser_version"]),
        license=str(item["license"]),
        url=str(item["url"]),
        auth_mode=str(item["auth_mode"]),
        parser_args=dict(item["parser_args"]),
        source_class=str(item["source_class"]),
        access_policy=str(item["access_policy"]),
        terms_status=str(item["terms_status"]),
        terms_url=str(item["terms_url"]),
        citation=str(item["citation"]),
        priority=int(item["priority"]),
    )


def _parse_contract_results(frame: pl.DataFrame) -> pl.DataFrame:
    return CuratedDataBuilder(ProjectContext.create(root=ROOT))._normalize_medsl_house(
        frame,
        "medsl-house-constituency-results-v1",
        {"cycles": [2018]},
    )


def test_checked_in_production_registries_never_inherit_fixtures() -> None:
    official_context = ProjectContext.create(
        root=ROOT, sources_config="sources_official_results.yaml"
    )
    public_context = ProjectContext.create(root=ROOT, sources_config="sources_public_web.yaml")
    official = SourceRegistry.from_context(official_context)
    public = SourceRegistry.from_context(public_context)
    public_yaml = yaml.safe_load((ROOT / "configs" / "sources_public_web.yaml").read_text())

    assert public_yaml["extends"] == "sources_official_results.yaml"
    assert {source.id for source in official.sources} == {
        "medsl_clerk_house_races_1976_2018",
        "medsl_clerk_house_results_1976_2018",
    }
    assert {source.source_class for source in official.sources} == {"official_public"}
    assert not {"fixture", "synthetic", "generated"}.intersection(
        source.source_class for source in public.sources
    )
    assert {source.id for source in official.sources}.issubset(
        source.id for source in public.sources
    )


def test_official_house_adapter_builds_canonical_races_results_and_audit(
    tmp_path: Path, monkeypatch
) -> None:
    sources = [
        _source(
            "medsl_clerk_house_races_1976_2018",
            "races",
            "medsl-house-constituency-races-v1",
        ),
        _source(
            "medsl_clerk_house_results_1976_2018",
            "results",
            "medsl-house-constituency-results-v1",
        ),
    ]
    context = _context(tmp_path, sources)
    payload = CONTRACT_CSV.read_bytes()
    monkeypatch.setattr(SyncRunner, "_http_get_with_retry", staticmethod(lambda _url: payload))

    registry = SourceRegistry.from_context(context)
    sync = SyncRunner(context, registry=registry).run()
    curated = CuratedDataBuilder(context).run().tables
    audit = DataAuditRunner(context).verify(run_id="official-house", profile="production")

    assert {source.source_class for source in registry.sources} == {"official_public"}
    assert {source.priority for source in registry.sources} == {400}
    assert sync.failed_sources == 0
    assert sync.manifest["status"].to_list() == ["fetched", "fetched"]
    assert sync.manifest["terms_url"].str.starts_with("https://").all()
    assert sync.manifest["citation"].str.len_chars().min() > 0
    assert pl.read_parquet(context.raw_dir / "snapshot_index.parquet").height == 2

    races = curated["races"]
    results = curated["results"]
    assert races["race_id"].to_list() == ["US-HOUSE-WI-01-2018", "US-HOUSE-WI-02-2018"]
    assert races["election_date"].cast(pl.Utf8).unique().to_list() == ["2018-11-06"]
    assert races["source_id"].unique().to_list() == ["medsl_clerk_house_races_1976_2018"]
    assert results.height == 5
    assert results.group_by("race_id").agg(pl.col("vote_share").sum())["vote_share"].to_list() == [
        pytest.approx(1.0),
        pytest.approx(1.0),
    ]
    assert results.group_by("race_id").agg(pl.col("winner").sum())["winner"].to_list() == [1, 1]
    assert results["source_id"].unique().to_list() == ["medsl_clerk_house_results_1976_2018"]
    assert results["source_hash"].str.len_chars().unique().to_list() == [64]
    assert results["official"].all()

    assert audit["passed"] is True
    assert audit["audit"]["free_public_web_only"] is True
    assert audit["audit"]["fixture_source_count"] == 0
    assert audit["audit"]["synthetic_row_count"] == 0
    assert audit["audit"]["fixture_sources"] == []
    assert audit["audit"]["failed_sources"] == []

    race_schema = json.loads(
        (ROOT / "schemas" / "curated_tables" / "race.schema.json").read_text(encoding="utf-8")
    )
    result_schema = json.loads(
        (ROOT / "schemas" / "curated_tables" / "official_result.schema.json").read_text(
            encoding="utf-8"
        )
    )
    race_row = races.row(0, named=True)
    race_row["election_date"] = race_row["election_date"].isoformat()
    Draft202012Validator(race_schema).validate(race_row)
    Draft202012Validator(result_schema).validate(results.row(0, named=True))


def test_exact_top_tie_is_explicitly_unresolved_without_multiple_winners() -> None:
    frame = pl.read_csv(CONTRACT_CSV).filter(
        (pl.col("district") == 2) & (pl.col("mode") == "total")
    )
    tied = frame.with_columns(
        pl.lit(155000).alias("candidatevotes"),
        pl.lit(310000).alias("totalvotes"),
    )

    results = _parse_contract_results(tied)

    assert results["winner"].sum() == 0
    assert results["winner_status"].unique().to_list() == ["tie_unresolved"]
    assert results["vote_share"].sum() == pytest.approx(1.0)


def test_fusion_party_precedence_and_output_are_file_order_invariant() -> None:
    frame = pl.read_csv(CONTRACT_CSV).filter(
        (pl.col("district") == 1) & (pl.col("mode") == "total")
    )
    frame = frame.with_columns(
        pl.when(pl.col("candidate") == "Dan Candidate")
        .then(pl.lit(145000))
        .when(pl.col("candidate") == "Rae Candidate")
        .then(pl.lit(149000))
        .otherwise(pl.lit(1000))
        .alias("candidatevotes")
    )
    fusion_row = frame.filter(pl.col("candidate") == "Dan Candidate").row(0, named=True)
    fusion_row.update({"party": "working families", "candidatevotes": 5000})
    fusion = pl.concat([frame, pl.DataFrame([fusion_row])], how="diagonal_relaxed")

    forward = _parse_contract_results(fusion)
    reverse = _parse_contract_results(fusion.reverse())

    assert forward.to_dicts() == reverse.to_dicts()
    dan = forward.filter(pl.col("candidate") == "Dan Candidate")
    assert dan["party"].to_list() == ["DEM"]
    assert dan["vote_count"].to_list() == [150000.0]
    assert dan["winner"].to_list() == [True]
    assert forward["vote_share"].sum() == pytest.approx(1.0)
    assert forward["winner"].sum() == 1


def test_result_reconciliation_rejects_vote_share_mismatch() -> None:
    frame = pl.read_csv(CONTRACT_CSV).filter(
        (pl.col("district") == 2) & (pl.col("mode") == "total")
    )
    malformed = frame.with_columns(pl.lit(400000).alias("totalvotes"))

    with pytest.raises(ValueError, match="vote-share/winner reconciliation"):
        _parse_contract_results(malformed)


@pytest.mark.parametrize(
    ("payload", "expected_status"),
    [
        (b"", "empty"),
        (b"year,state_po\n2018,WI\n", "schema_change"),
    ],
)
def test_http_source_records_empty_and_schema_change_statuses(
    tmp_path: Path, monkeypatch, payload: bytes, expected_status: str
) -> None:
    source = _results_definition()
    context = _context(tmp_path, [_source(source.id, source.table, source.parser_version)])
    monkeypatch.setattr(SyncRunner, "_http_get_with_retry", staticmethod(lambda _url: payload))

    result = SyncRunner(context, registry=SourceRegistry([source])).run()

    assert result.failed_sources == 1
    assert result.manifest["status"].to_list() == [expected_status]
    assert result.manifest["refresh_status"].to_list() == [expected_status]
    assert result.manifest["content_hash"].to_list() == [""]


def test_rate_limit_reuses_only_immutable_cached_snapshot(tmp_path: Path, monkeypatch) -> None:
    source = _results_definition()
    context = _context(tmp_path, [_source(source.id, source.table, source.parser_version)])
    payload = CONTRACT_CSV.read_bytes()
    monkeypatch.setattr(SyncRunner, "_http_get_with_retry", staticmethod(lambda _url: payload))
    first = SyncRunner(context, registry=SourceRegistry([source])).run()
    first_row = first.manifest.row(0, named=True)

    def rate_limited(_url: str) -> bytes:
        raise SourceSyncError("rate_limited", "HTTP 429 test contract")

    monkeypatch.setattr(SyncRunner, "_http_get_with_retry", staticmethod(rate_limited))
    second = SyncRunner(context, registry=SourceRegistry([source])).run()
    second_row = second.manifest.row(0, named=True)

    assert second_row["status"] == "stale_reused"
    assert second_row["refresh_status"] == "rate_limited"
    assert second_row["content_hash"] == first_row["content_hash"]
    assert second_row["raw_path"] == first_row["raw_path"]
    assert second_row["retrieved_at"] == first_row["retrieved_at"]
    snapshot_index = pl.read_parquet(context.raw_dir / "snapshot_index.parquet")
    assert snapshot_index.height == 1
    assert DataAuditRunner(context).verify(run_id="stale", profile="production")["passed"] is False


def test_http_429_is_classified_as_rate_limited(monkeypatch) -> None:
    def raise_429(*_args, **_kwargs):
        raise urllib.error.HTTPError(PINNED_URL, 429, "too many requests", {}, None)

    monkeypatch.setattr("civic_signal.ingest.sync.urllib.request.urlopen", raise_429)
    monkeypatch.setattr("civic_signal.ingest.sync.time.sleep", lambda _seconds: None)

    with pytest.raises(SourceSyncError, match="HTTP 429") as exc_info:
        SyncRunner._http_get_with_retry(PINNED_URL)
    assert exc_info.value.status == "rate_limited"
