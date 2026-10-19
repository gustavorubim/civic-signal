"""Fail loudly on live fetch failure unless cached data is allowed (audit P1-7)."""

from __future__ import annotations

from pathlib import Path

import pytest

from civic_signal.config import ProjectContext
from civic_signal.ingest import SourceDefinition, SourceFetchError, SourceRegistry, SyncRunner

ROOT = Path(__file__).resolve().parents[1]


def _context(tmp_path: Path, sources_config: str, allow_cached: bool | None = None):
    return ProjectContext.create(
        root=ROOT,
        sources_config=sources_config,
        data_dir=tmp_path / "data",
        artifacts_dir=tmp_path / "artifacts",
        allow_cached=allow_cached,
    )


def _registry() -> SourceRegistry:
    return SourceRegistry(
        [
            SourceDefinition(
                id="votehub_test",
                table="polls",
                type="http_json",
                path=None,
                parser_version="v1",
                license="CC BY 4.0",
                url="https://api.votehub.com/polls?poll_type=us-senator",
            )
        ]
    )


def _ok(monkeypatch) -> None:
    monkeypatch.setattr(SyncRunner, "_http_get_with_retry", staticmethod(lambda url: b"[]"))


def _fail(monkeypatch) -> None:
    def boom(url: str) -> bytes:
        raise RuntimeError("HTTP fetch failed after 3 attempts")

    monkeypatch.setattr(SyncRunner, "_http_get_with_retry", staticmethod(boom))


def test_fresh_fetch_records_status(tmp_path: Path, monkeypatch) -> None:
    _ok(monkeypatch)
    result = SyncRunner(_context(tmp_path, "sources_senate_live.yaml"), _registry()).run()
    row = result.manifest.row(0, named=True)
    assert row["fetch_status"] == "fresh" and row["fetched_at"]


def test_live_config_raises_on_fetch_error(tmp_path: Path, monkeypatch) -> None:
    _ok(monkeypatch)
    SyncRunner(_context(tmp_path, "sources_senate_live.yaml"), _registry()).run()
    _fail(monkeypatch)
    with pytest.raises(SourceFetchError, match="--allow-cached"):
        SyncRunner(_context(tmp_path, "sources_senate_live.yaml"), _registry()).run()


def test_allow_cached_reuses_snapshot(tmp_path: Path, monkeypatch) -> None:
    _ok(monkeypatch)
    first = SyncRunner(_context(tmp_path, "sources_senate_live.yaml"), _registry()).run()
    fetched_at = first.manifest["fetched_at"][0]
    _fail(monkeypatch)
    context = _context(tmp_path, "sources_senate_live.yaml", allow_cached=True)
    row = SyncRunner(context, _registry()).run().manifest.row(0, named=True)
    assert row["fetch_status"] == "cached"
    assert row["status"] == "stale_reused"
    assert row["fetched_at"] == fetched_at  # time of the snapshot actually used


def test_non_live_config_defaults_to_cached_and_explicit_override(
    tmp_path: Path, monkeypatch
) -> None:
    _ok(monkeypatch)
    SyncRunner(_context(tmp_path, "sources.yaml"), _registry()).run()
    _fail(monkeypatch)
    row = (
        SyncRunner(_context(tmp_path, "sources.yaml"), _registry())
        .run()
        .manifest.row(0, named=True)
    )
    assert row["fetch_status"] == "cached"
    with pytest.raises(SourceFetchError):
        SyncRunner(_context(tmp_path, "sources.yaml"), _registry(), allow_cached=False).run()


def test_cli_exposes_allow_cached() -> None:
    from typer.testing import CliRunner

    from civic_signal.cli import app

    result = CliRunner().invoke(app, ["forecast", "run", "--help"])
    assert "--allow-cached" in result.output
