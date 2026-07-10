from __future__ import annotations

import csv
import hashlib
import io
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import ClassVar

import polars as pl

from civic_signal.config import ProjectContext
from civic_signal.ingest.sources import SourceDefinition, SourceRegistry
from civic_signal.storage.io import read_json, write_json, write_parquet

HTTP_RETRY_ATTEMPTS = 3
HTTP_BACKOFF_SECONDS = (1.0, 2.0, 4.0)
# The archived FiveThirtyEight Datasette mirror routinely takes ~30s to respond.
HTTP_TIMEOUT_SECONDS = 60
# Wikipedia returns 403 for the default Python urllib User-Agent; identify per
# https://meta.wikimedia.org/wiki/User-Agent_policy.
HTTP_USER_AGENT = "civic-signal/0.1 (election forecasting research; +https://github.com/gustavorubim/civic-signal)"
HTTP_MAX_BYTES = 500 * 1024 * 1024
HTTP_ALLOWED_CONTENT_TYPES = (
    "text/csv",
    "text/x-wiki",
    "text/plain",
    "application/csv",
    "application/octet-stream",
)


@dataclass(frozen=True)
class SyncResult:
    manifest: pl.DataFrame
    fetched_sources: int
    skipped_sources: int
    failed_sources: int


class SourceSyncError(RuntimeError):
    """A source failure with a stable machine-readable manifest status."""

    VALID_STATUSES: ClassVar[set[str]] = {"empty", "rate_limited", "schema_change", "failed"}

    def __init__(self, status: str, message: str) -> None:
        if status not in self.VALID_STATUSES:
            raise ValueError(f"Unsupported source sync status: {status}")
        self.status = status
        super().__init__(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class SyncRunner:
    """Incremental local sync for source definitions."""

    def __init__(self, context: ProjectContext, registry: SourceRegistry | None = None) -> None:
        self.context = context
        self.registry = registry or SourceRegistry.from_context(context)

    def run(self) -> SyncResult:
        self.context.raw_dir.mkdir(parents=True, exist_ok=True)
        self.context.state_dir.mkdir(parents=True, exist_ok=True)
        state_path = self.context.state_dir / "sync_state.json"
        previous = read_json(state_path) if state_path.exists() else {}
        previous_manifest_path = self.context.raw_dir / "source_manifest.parquet"
        previous_rows: dict[str, dict[str, object]] = {}
        if previous_manifest_path.exists():
            previous_manifest = pl.read_parquet(previous_manifest_path)
            previous_rows = {
                str(row["source_id"]): row
                for row in previous_manifest.iter_rows(named=True)
                if row.get("source_id") is not None
            }

        rows: list[dict[str, object]] = []
        active_source_ids = {source.id for source in self.registry.sources}
        state: dict[str, str] = {
            str(source_id): str(content_hash)
            for source_id, content_hash in previous.items()
            if source_id in active_source_ids
        }
        fetched = skipped = failed = 0
        retrieved_at = datetime.now(UTC).isoformat()

        for source in self.registry.sources:
            try:
                row, did_fetch = self._sync_one(source, previous, retrieved_at)
                if not did_fetch:
                    previous_row = previous_rows.get(source.id, {})
                    if str(previous_row.get("content_hash") or "") == str(
                        row.get("content_hash") or ""
                    ):
                        original = str(
                            previous_row.get("original_snapshot_at")
                            or previous_row.get("retrieved_at")
                            or retrieved_at
                        )
                        row["retrieved_at"] = original
                        row["original_snapshot_at"] = original
                row["checked_at"] = retrieved_at
                fetched += int(did_fetch)
                skipped += int(not did_fetch)
                state[source.id] = str(row["content_hash"])
            except Exception as exc:  # pragma: no cover - defensive manifest path
                failed += 1
                row = self._failure_row(source, retrieved_at, exc)
            rows.append(row)

        manifest = pl.DataFrame(rows)
        write_parquet(manifest, self.context.raw_dir / "source_manifest.parquet")
        self._update_snapshot_index(manifest)
        write_json(state, state_path)
        return SyncResult(manifest, fetched, skipped, failed)

    def _update_snapshot_index(self, manifest: pl.DataFrame) -> None:
        """Retain one immutable index row for every observed source-content pair."""
        index_path = self.context.raw_dir / "snapshot_index.parquet"
        snapshots = manifest.filter(pl.col("content_hash").cast(pl.Utf8) != "")
        if index_path.exists():
            snapshots = pl.concat(
                [pl.read_parquet(index_path), snapshots],
                how="diagonal_relaxed",
            )
        snapshots = snapshots.unique(
            subset=["source_id", "content_hash"],
            keep="first",
            maintain_order=True,
        )
        if {"source_id", "original_snapshot_at"}.issubset(snapshots.columns):
            snapshots = snapshots.sort(["source_id", "original_snapshot_at"])
        write_parquet(snapshots, index_path)

    def _sync_one(
        self,
        source: SourceDefinition,
        previous: dict[str, str],
        retrieved_at: str,
    ) -> tuple[dict[str, object], bool]:
        if source.type in {"http_csv", "http_text"}:
            return self._sync_http(source, previous, retrieved_at)
        if source.type != "fixture":
            raise ValueError(f"Unsupported source type: {source.type}")
        if source.path is None:
            raise ValueError(f"Fixture source {source.id} requires a local path")
        content_hash = _sha256(source.path)
        raw_path = self.context.raw_dir / source.id / f"{content_hash}{source.path.suffix}"
        did_fetch = previous.get(source.id) != content_hash or not raw_path.exists()
        if did_fetch:
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source.path, raw_path)
        return (
            {
                "source_id": source.id,
                "table": source.table,
                "url": source.url,
                "raw_path": str(raw_path),
                "retrieved_at": retrieved_at,
                "original_snapshot_at": retrieved_at,
                "checked_at": retrieved_at,
                "content_hash": content_hash,
                "license": source.license,
                "terms_url": source.terms_url,
                "citation": source.citation,
                "parser_version": source.parser_version,
                "parser_args": source.parser_args_json(),
                "auth_mode": source.auth_mode,
                "source_class": source.source_class,
                "access_policy": source.access_policy,
                "terms_status": source.terms_status,
                "source_priority": source.priority,
                "status": "fetched" if did_fetch else "unchanged",
                "refresh_status": "",
                "error": "",
                "downstream_usage": "",
            },
            did_fetch,
        )

    def _sync_http(
        self,
        source: SourceDefinition,
        previous: dict[str, str],
        retrieved_at: str,
    ) -> tuple[dict[str, object], bool]:
        try:
            payload = self._http_get_with_retry(source.url)
        except Exception as exc:
            cached = self._cached_http_row(source, previous, retrieved_at, exc)
            if cached is not None:
                return cached, False
            raise
        self._validate_http_payload(source, payload)
        content_hash = hashlib.sha256(payload).hexdigest()
        suffix = self._http_suffix(source)
        raw_path = self.context.raw_dir / source.id / f"{content_hash}{suffix}"
        did_fetch = previous.get(source.id) != content_hash or not raw_path.exists()
        if did_fetch:
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            raw_path.write_bytes(payload)
        return (
            {
                "source_id": source.id,
                "table": source.table,
                "url": source.url,
                "raw_path": str(raw_path),
                "retrieved_at": retrieved_at,
                "original_snapshot_at": retrieved_at,
                "checked_at": retrieved_at,
                "content_hash": content_hash,
                "license": source.license,
                "terms_url": source.terms_url,
                "citation": source.citation,
                "parser_version": source.parser_version,
                "parser_args": source.parser_args_json(),
                "auth_mode": source.auth_mode,
                "source_class": source.source_class,
                "access_policy": source.access_policy,
                "terms_status": source.terms_status,
                "source_priority": source.priority,
                "status": "fetched" if did_fetch else "unchanged",
                "refresh_status": "",
                "error": "",
                "downstream_usage": "",
            },
            did_fetch,
        )

    def _cached_http_row(
        self,
        source: SourceDefinition,
        previous: dict[str, str],
        retrieved_at: str,
        exc: Exception,
    ) -> dict[str, object] | None:
        previous_hash = str(previous.get(source.id) or "")
        if not previous_hash:
            return None
        raw_path = self.context.raw_dir / source.id / f"{previous_hash}{self._http_suffix(source)}"
        if not raw_path.exists():
            return None
        manifest_path = self.context.raw_dir / "source_manifest.parquet"
        if manifest_path.exists():
            manifest = pl.read_parquet(manifest_path)
            previous_rows = manifest.filter(pl.col("source_id") == source.id)
            if previous_rows.is_empty():
                return None
            previous_row = previous_rows.tail(1).row(0, named=True)
            if (
                str(previous_row.get("url") or "") != source.url
                or str(previous_row.get("parser_version") or "") != source.parser_version
            ):
                return None
        return {
            "source_id": source.id,
            "table": source.table,
            "url": source.url,
            "raw_path": str(raw_path),
            "retrieved_at": retrieved_at,
            "original_snapshot_at": retrieved_at,
            "checked_at": retrieved_at,
            "content_hash": previous_hash,
            "license": source.license,
            "terms_url": source.terms_url,
            "citation": source.citation,
            "parser_version": source.parser_version,
            "parser_args": source.parser_args_json(),
            "auth_mode": source.auth_mode,
            "source_class": source.source_class,
            "access_policy": source.access_policy,
            "terms_status": source.terms_status,
            "source_priority": source.priority,
            "status": "stale_reused",
            "refresh_status": self._error_status(exc),
            "error": f"refresh failed; reused previous raw snapshot: {exc}",
            "downstream_usage": "",
        }

    @staticmethod
    def _http_get_with_retry(url: str) -> bytes:
        request = urllib.request.Request(url, headers={"User-Agent": HTTP_USER_AGENT})
        last_exc: Exception | None = None
        for attempt in range(HTTP_RETRY_ATTEMPTS):
            try:
                with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT_SECONDS) as response:
                    raw_type = response.headers.get("Content-Type") or ""
                    content_type = raw_type.split(";")[0].strip().lower()
                    if content_type and not any(
                        content_type.startswith(allowed) for allowed in HTTP_ALLOWED_CONTENT_TYPES
                    ):
                        raise ValueError(f"Unexpected content-type {content_type!r} from {url}")
                    declared = response.headers.get("Content-Length")
                    if declared is not None and int(declared) > HTTP_MAX_BYTES:
                        raise ValueError(
                            f"Payload {int(declared)} bytes exceeds max {HTTP_MAX_BYTES}: {url}"
                        )
                    payload = response.read(HTTP_MAX_BYTES + 1)
                    if len(payload) > HTTP_MAX_BYTES:
                        raise ValueError(
                            f"Payload exceeds max {HTTP_MAX_BYTES} bytes while reading: {url}"
                        )
                    return payload
            except urllib.error.HTTPError as exc:
                if exc.code == 429:
                    last_exc = SourceSyncError(
                        "rate_limited", f"HTTP 429 rate limit response from {url}"
                    )
                elif 400 <= exc.code < 500:
                    raise SourceSyncError(
                        "failed", f"HTTP {exc.code} client error from {url}"
                    ) from exc
                else:
                    last_exc = exc
                if attempt < HTTP_RETRY_ATTEMPTS - 1:
                    time.sleep(HTTP_BACKOFF_SECONDS[attempt])
            except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
                last_exc = exc
                if attempt < HTTP_RETRY_ATTEMPTS - 1:
                    time.sleep(HTTP_BACKOFF_SECONDS[attempt])
        if isinstance(last_exc, SourceSyncError):
            raise last_exc
        raise RuntimeError(
            f"HTTP fetch failed after {HTTP_RETRY_ATTEMPTS} attempts: {url}"
        ) from last_exc

    @staticmethod
    def _validate_http_payload(source: SourceDefinition, payload: bytes) -> None:
        if not payload.strip():
            raise SourceSyncError("empty", f"Source {source.id} returned an empty payload")
        if source.type != "http_csv":
            return
        required = source.parser_args.get("required_columns")
        if required is None:
            return
        if not isinstance(required, list) or not required:
            raise SourceSyncError(
                "schema_change",
                f"Source {source.id} required_columns contract must be a non-empty list",
            )
        try:
            header = next(csv.reader(io.StringIO(payload.decode("utf-8-sig"))))
        except (UnicodeDecodeError, StopIteration, csv.Error) as exc:
            raise SourceSyncError(
                "schema_change", f"Source {source.id} CSV header could not be parsed"
            ) from exc
        missing = sorted(str(column) for column in required if str(column) not in header)
        if missing:
            raise SourceSyncError(
                "schema_change",
                f"Source {source.id} missing required CSV columns: {missing}",
            )

    @staticmethod
    def _http_suffix(source: SourceDefinition) -> str:
        if source.path and source.path.suffix:
            return source.path.suffix
        parsed = urllib.parse.urlparse(source.url)
        suffix = Path(parsed.path).suffix
        return suffix or ".dat"

    @staticmethod
    def _failure_row(
        source: SourceDefinition,
        retrieved_at: str,
        exc: Exception,
    ) -> dict[str, object]:
        return {
            "source_id": source.id,
            "table": source.table,
            "url": source.url,
            "raw_path": "",
            "retrieved_at": retrieved_at,
            "original_snapshot_at": retrieved_at,
            "checked_at": retrieved_at,
            "content_hash": "",
            "license": source.license,
            "terms_url": source.terms_url,
            "citation": source.citation,
            "parser_version": source.parser_version,
            "parser_args": source.parser_args_json(),
            "auth_mode": source.auth_mode,
            "source_class": source.source_class,
            "access_policy": source.access_policy,
            "terms_status": source.terms_status,
            "source_priority": source.priority,
            "status": SyncRunner._error_status(exc),
            "refresh_status": SyncRunner._error_status(exc),
            "error": str(exc),
            "downstream_usage": "",
        }

    @staticmethod
    def _error_status(exc: Exception) -> str:
        status = getattr(exc, "status", "failed")
        return str(status) if str(status) in SourceSyncError.VALID_STATUSES else "failed"
