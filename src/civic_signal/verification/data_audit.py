"""Production source-registry and snapshot audit."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import polars as pl

from civic_signal.config import ProjectContext
from civic_signal.ingest.sources import SourceDefinition, SourceRegistry
from civic_signal.storage.io import write_json, write_parquet

_FIXTURE_SOURCE_CLASSES = frozenset({"fixture", "synthetic", "generated"})
_HEALTHY_STATUSES = frozenset({"fetched", "unchanged"})


class DataAuditRunner:
    """Audit configured inputs before they are eligible for production use."""

    def __init__(self, context: ProjectContext) -> None:
        self.context = context

    def verify(
        self,
        *,
        run_id: str,
        profile: str = "production",
        as_of: str | None = None,
    ) -> dict[str, Any]:
        out_dir = self.context.artifacts_dir / "data_audits" / run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        registry = SourceRegistry.from_context(self.context)
        manifest_path = self.context.raw_dir / "source_manifest.parquet"
        manifest = pl.read_parquet(manifest_path) if manifest_path.exists() else pl.DataFrame()
        snapshot_index_path = self.context.raw_dir / "snapshot_index.parquet"
        source_snapshot_index = (
            pl.read_parquet(snapshot_index_path) if snapshot_index_path.exists() else pl.DataFrame()
        )

        rows = [self._audit_source(source, manifest) for source in registry.sources]
        configured_ids = {source.id for source in registry.sources}
        observed_ids = (
            set(manifest["source_id"].cast(pl.String).to_list())
            if not manifest.is_empty() and "source_id" in manifest.columns
            else set()
        )
        missing_snapshots = sorted(configured_ids - observed_ids)
        required_snapshot_columns = {
            "source_id",
            "content_hash",
            "parser_version",
            "retrieved_at",
            "original_snapshot_at",
        }
        missing_snapshot_columns = sorted(
            required_snapshot_columns - set(source_snapshot_index.columns)
        )
        indexed_pairs = (
            {
                (str(row["source_id"]), str(row["content_hash"]))
                for row in source_snapshot_index.iter_rows(named=True)
            }
            if not missing_snapshot_columns
            else set()
        )
        current_pairs = {
            (str(row["source_id"]), str(row["content_hash"]))
            for row in manifest.iter_rows(named=True)
            if str(row.get("content_hash") or "")
        }
        missing_snapshot_versions = sorted(
            f"{source_id}:{content_hash}"
            for source_id, content_hash in current_pairs - indexed_pairs
        )
        invalid_content_hashes = sorted(
            source_id
            for source_id, content_hash in current_pairs
            if len(content_hash) != 64
            or any(character not in "0123456789abcdef" for character in content_hash.lower())
        )
        failed = sorted(row["source_id"] for row in rows if row["status"] not in _HEALTHY_STATUSES)
        nonfree = sorted(row["source_id"] for row in rows if not row["free_public_web"])
        fixture_sources = sorted(
            row["source_id"] for row in rows if row["source_class"] in _FIXTURE_SOURCE_CLASSES
        )
        synthetic_sources = sorted(
            row["source_id"] for row in rows if row["source_class"] in {"synthetic", "generated"}
        )
        free_public_web_only = not nonfree and not fixture_sources
        production_ready = bool(
            manifest_path.exists()
            and not missing_snapshots
            and not failed
            and not nonfree
            and not fixture_sources
            and free_public_web_only
            and snapshot_index_path.exists()
            and not missing_snapshot_columns
            and not missing_snapshot_versions
            and not invalid_content_hashes
        )
        reasons = self._production_reasons(
            manifest_present=manifest_path.exists(),
            snapshot_index_present=snapshot_index_path.exists(),
            missing_snapshots=missing_snapshots,
            failed=failed,
            nonfree=nonfree,
            fixture_sources=fixture_sources,
            synthetic_sources=synthetic_sources,
            missing_snapshot_columns=missing_snapshot_columns,
            missing_snapshot_versions=missing_snapshot_versions,
            invalid_content_hashes=invalid_content_hashes,
            free_public_web_only=free_public_web_only,
        )
        # Production never auto-passes: fixtures, synthetics, and non-free_public_web
        # sources are hard failures. Research/fixture profiles remain operationally
        # usable when a manifest is present so local development is not blocked.
        if profile == "production":
            passed = production_ready
            if production_ready:
                status = "ok"
            elif self._has_policy_violations(
                rows=rows,
                nonfree=nonfree,
                fixture_sources=fixture_sources,
                invalid_content_hashes=invalid_content_hashes,
            ):
                # Explicit free-web / fixture / degraded-status breaches fail hard.
                # Pure missing snapshot evidence remains insufficient_evidence.
                status = "failed"
            else:
                status = "insufficient_evidence"
        else:
            passed = bool(manifest_path.exists())
            status = "ok" if passed else "insufficient_evidence"

        audit = {
            "schema_version": "1.0.0",
            "run_id": run_id,
            "profile": profile,
            "as_of": as_of,
            "generated_at": datetime.now(UTC).isoformat(),
            "registry_file": self.context.sources_config,
            "source_count": len(rows),
            "observed_snapshot_count": len(observed_ids),
            "missing_snapshots": missing_snapshots,
            "failed_sources": failed,
            "nonfree_sources": nonfree,
            "fixture_sources": fixture_sources,
            "fixture_source_count": len(fixture_sources),
            "synthetic_sources": synthetic_sources,
            "synthetic_row_count": 0,
            "free_public_web_only": free_public_web_only,
            "production_ready": production_ready,
            "reasons": reasons,
            "append_only_snapshot_index_present": snapshot_index_path.exists(),
            "snapshot_version_count": source_snapshot_index.height,
            "missing_snapshot_index_columns": missing_snapshot_columns,
            "missing_snapshot_versions": missing_snapshot_versions,
            "invalid_content_hashes": invalid_content_hashes,
            "status": status,
            "sources": rows,
        }
        write_json(audit, out_dir / "source_registry_audit.json")
        write_json(
            {
                "run_id": run_id,
                "profile": profile,
                "status": status,
                "free_public_web_only": free_public_web_only,
                "production_ready": production_ready,
                "reasons": reasons,
                "fixture_sources": fixture_sources,
                "nonfree_sources": nonfree,
                "sources": [
                    {
                        "source_id": row["source_id"],
                        "license": row["license"],
                        "terms_status": row["terms_status"],
                        "access_policy": row["access_policy"],
                        "terms_url": row["terms_url"],
                        "citation": row["citation"],
                        "source_class": row["source_class"],
                        "free_public_web": row["free_public_web"],
                    }
                    for row in rows
                ],
            },
            out_dir / "terms_audit.json",
        )
        if source_snapshot_index.is_empty():
            snapshot_index = pl.DataFrame(
                schema={
                    "source_id": pl.String,
                    "content_hash": pl.String,
                    "status": pl.String,
                }
            )
        else:
            columns = [
                column
                for column in (
                    "source_id",
                    "content_hash",
                    "status",
                    "retrieved_at",
                    "original_snapshot_at",
                    "checked_at",
                    "parser_version",
                    "url",
                )
                if column in source_snapshot_index.columns
            ]
            snapshot_index = source_snapshot_index.select(columns)
        write_parquet(snapshot_index, out_dir / "snapshot_index_audit.parquet")
        return {
            "run_id": run_id,
            "profile": profile,
            "passed": passed,
            "exit_nonzero": not passed,
            "audit_path": str(out_dir / "source_registry_audit.json"),
            "terms_audit_path": str(out_dir / "terms_audit.json"),
            "snapshot_index_path": str(out_dir / "snapshot_index_audit.parquet"),
            "audit": audit,
        }

    @staticmethod
    def _has_policy_violations(
        *,
        rows: list[dict[str, Any]],
        nonfree: list[str],
        fixture_sources: list[str],
        invalid_content_hashes: list[str],
    ) -> bool:
        degraded = any(
            row["status"] not in _HEALTHY_STATUSES | {"missing_snapshot"} for row in rows
        )
        return bool(nonfree or fixture_sources or invalid_content_hashes or degraded)

    @staticmethod
    def _production_reasons(
        *,
        manifest_present: bool,
        snapshot_index_present: bool,
        missing_snapshots: list[str],
        failed: list[str],
        nonfree: list[str],
        fixture_sources: list[str],
        synthetic_sources: list[str],
        missing_snapshot_columns: list[str],
        missing_snapshot_versions: list[str],
        invalid_content_hashes: list[str],
        free_public_web_only: bool,
    ) -> list[str]:
        reasons: list[str] = []
        if not manifest_present:
            reasons.append("missing source_manifest.parquet")
        if not snapshot_index_present:
            reasons.append("missing append-only snapshot_index.parquet")
        if missing_snapshots:
            reasons.append(f"missing snapshots for sources: {', '.join(missing_snapshots)}")
        if failed:
            reasons.append(f"degraded or missing source statuses: {', '.join(failed)}")
        if fixture_sources:
            reasons.append(
                "fixture/synthetic/generated sources present: " + ", ".join(fixture_sources)
            )
        if synthetic_sources:
            reasons.append("synthetic/generated sources present: " + ", ".join(synthetic_sources))
        if nonfree:
            reasons.append("sources that are not free_public_web eligible: " + ", ".join(nonfree))
        if not free_public_web_only:
            reasons.append("free_public_web_only is false")
        if missing_snapshot_columns:
            reasons.append("snapshot index missing columns: " + ", ".join(missing_snapshot_columns))
        if missing_snapshot_versions:
            reasons.append(
                "current source/hash versions absent from snapshot index: "
                + ", ".join(missing_snapshot_versions[:10])
                + ("..." if len(missing_snapshot_versions) > 10 else "")
            )
        if invalid_content_hashes:
            reasons.append(
                "invalid content hashes for sources: " + ", ".join(invalid_content_hashes)
            )
        return reasons

    @staticmethod
    def _audit_source(source: SourceDefinition, manifest: pl.DataFrame) -> dict[str, Any]:
        row: dict[str, Any] = {}
        if not manifest.is_empty() and "source_id" in manifest.columns:
            matches = manifest.filter(pl.col("source_id").cast(pl.String) == source.id)
            if not matches.is_empty():
                row = matches.tail(1).row(0, named=True)
        url = str(row.get("url") or source.url)
        auth_mode = str(row.get("auth_mode") or source.auth_mode).lower()
        source_class = str(row.get("source_class") or source.source_class).lower()
        access_policy = str(row.get("access_policy") or source.access_policy).lower()
        terms_status = str(row.get("terms_status") or source.terms_status).lower()
        license_text = str(row.get("license") or source.license).strip()
        terms_url = str(row.get("terms_url") or source.terms_url).strip()
        citation = str(row.get("citation") or source.citation).strip()
        terms_metadata_complete = bool(license_text) and (
            source_class != "official_public" or (terms_url.startswith("https://") and citation)
        )
        free_public_web = bool(
            source.type.startswith("http_")
            and url.startswith("https://")
            and auth_mode in {"public", "none", "api_key", "free_api_key"}
            and access_policy == "free_public_web"
            and terms_status in {"reviewed_for_use", "documented_public"}
            and source_class in {"official_public", "production", "production_web"}
            and terms_metadata_complete
        )
        return {
            "source_id": source.id,
            "table": source.table,
            "type": source.type,
            "url": url,
            "auth_mode": auth_mode,
            "source_class": source_class,
            "access_policy": access_policy,
            "terms_status": terms_status,
            "license": license_text,
            "terms_url": terms_url,
            "citation": citation,
            "terms_metadata_complete": terms_metadata_complete,
            "status": str(row.get("status") or "missing_snapshot"),
            "refresh_status": str(row.get("refresh_status") or ""),
            "content_hash": str(row.get("content_hash") or ""),
            "retrieved_at": row.get("retrieved_at"),
            "original_snapshot_at": row.get("original_snapshot_at"),
            "parser_version": str(row.get("parser_version") or source.parser_version),
            "free_public_web": free_public_web,
        }
