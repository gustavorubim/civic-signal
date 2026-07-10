"""Deterministic as-of integrity verification for selected model inputs."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl

from civic_signal.config import ProjectContext
from civic_signal.features import (
    FeatureBundle,
    snapshot_event_column,
    snapshot_selection_key_columns,
    snapshot_selection_predicate,
)
from civic_signal.storage.io import read_json, write_json, write_parquet, write_text

_LINEAGE_TABLES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("polls", "end_date", ("poll_id", "race_id", "option_id")),
    (
        "market_quotes",
        "observed_at",
        ("market_id", "race_id", "option_id", "observed_at"),
    ),
    (
        "public_signals",
        "observed_at",
        ("signal_id", "race_id", "option_id", "observed_at"),
    ),
    (
        "fundamentals",
        "as_of",
        (
            "race_id",
            "feature_id",
            "series_id",
            "option_id",
            "observed_at",
            "as_of",
            "revision_id",
        ),
    ),
)

_LINEAGE_SCHEMA: dict[str, pl.DataType] = {
    "table": pl.String,
    "row_key": pl.String,
    "selection_key": pl.String,
    "snapshot_id": pl.String,
    "revision_id": pl.String,
    "observed_at": pl.String,
    "published_at": pl.String,
    "selection_predicate": pl.String,
    "source_id": pl.String,
    "source_hash": pl.String,
    "available_at": pl.String,
    "availability_basis": pl.String,
    "availability_is_explicit": pl.Boolean,
    "selected_as_of": pl.String,
}

_CORE_LINEAGE_COLUMNS = {
    "table",
    "row_key",
    "source_id",
    "source_hash",
    "available_at",
    "availability_basis",
    "availability_is_explicit",
    "selected_as_of",
}

_SELECTION_CONTRACT_COLUMNS = {
    "selection_key",
    "snapshot_id",
    "revision_id",
    "observed_at",
    "published_at",
    "selection_predicate",
}


def build_selected_feature_lineage(bundle: FeatureBundle, as_of: str) -> pl.DataFrame:
    """Materialize auditable availability lineage for every selected time-varying row."""
    frames: list[pl.DataFrame] = []
    bundle_tables = {
        "polls": bundle.polls,
        "market_quotes": bundle.markets,
        "public_signals": bundle.public_signals,
        "fundamentals": bundle.fundamentals,
    }
    for table, default_proxy_column, key_columns in _LINEAGE_TABLES:
        frame = bundle_tables[table]
        if frame.is_empty():
            continue
        proxy_column = snapshot_event_column(frame, table)
        if proxy_column not in frame.columns:
            proxy_column = default_proxy_column
        present_keys = [column for column in key_columns if column in frame.columns]
        if not present_keys:
            present_keys = [proxy_column] if proxy_column in frame.columns else []
        row_key = _row_key_expression(present_keys)
        selection_columns = snapshot_selection_key_columns(frame, table)
        selection_key = _row_key_expression(selection_columns)
        has_available_at = "available_at" in frame.columns
        if has_available_at:
            available_at = pl.col("available_at").cast(pl.String)
            basis = (
                pl.col("availability_basis").cast(pl.String)
                if "availability_basis" in frame.columns
                else pl.lit("source_record")
            )
            availability_is_explicit = ~basis.is_in(["event_date_proxy", "missing"])
        else:
            available_at = (
                pl.col(proxy_column).cast(pl.String)
                if proxy_column in frame.columns
                else pl.lit(None, dtype=pl.String)
            )
            basis = pl.lit("event_date_proxy")
            availability_is_explicit = pl.lit(False)
        snapshot_id = pl.concat_str(
            [
                pl.lit(table),
                selection_key,
                _string_column(frame, "revision_id").fill_null("<no-revision>"),
                available_at.fill_null("<no-availability>"),
                _string_column(frame, "source_hash").fill_null("<no-source-hash>"),
            ],
            separator="|",
        )
        frames.append(
            frame.select(
                pl.lit(table).alias("table"),
                row_key.alias("row_key"),
                selection_key.alias("selection_key"),
                snapshot_id.alias("snapshot_id"),
                _string_column(frame, "revision_id").alias("revision_id"),
                _string_column(frame, proxy_column).alias("observed_at"),
                _string_column(frame, "published_at").alias("published_at"),
                pl.lit(snapshot_selection_predicate(table, frame)).alias("selection_predicate"),
                _string_column(frame, "source_id").alias("source_id"),
                _string_column(frame, "source_hash").alias("source_hash"),
                available_at.alias("available_at"),
                basis.alias("availability_basis"),
                availability_is_explicit.alias("availability_is_explicit"),
                pl.lit(as_of).alias("selected_as_of"),
            )
        )
    if not frames:
        return _empty_lineage()
    return pl.concat(frames, how="vertical_relaxed").sort(["table", "row_key"])


class AsOfVerificationRunner:
    """Recompute row-level temporal integrity instead of trusting declared evidence."""

    def __init__(self, context: ProjectContext) -> None:
        self.context = context

    def verify(
        self,
        *,
        run_id: str,
        scenario_family: str | None = None,
        cycles: str | None = None,
        offsets: str | None = None,
        as_of: str | None = None,
        write_run_artifact: bool = False,
        canary_evidence: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        out_dir = self.context.artifacts_dir / "as_of_audits" / run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        run_dir = self._resolve_run_dir(run_id)
        cutoff = self._resolve_as_of(as_of, run_dir)
        lineage = self._load_lineage(run_dir)

        canary_artifact: dict[str, Any]
        if cutoff is not None and lineage is not None:
            audit = self._compute_audit(
                run_id=run_id,
                cutoff=cutoff,
                lineage=lineage,
                scenario_family=scenario_family,
                cycles=cycles,
                offsets=offsets,
                canary_evidence=canary_evidence,
            )
            # Durable selected-feature lineage for independent recompute/audit.
            write_parquet(lineage, out_dir / "selected_feature_lineage.parquet")
            if write_run_artifact and run_dir is not None:
                write_parquet(lineage, run_dir / "selected_feature_lineage.parquet")
            canary_artifact = {
                "run_id": run_id,
                "as_of": cutoff.isoformat(),
                "generated_at": datetime.now(UTC).isoformat(),
                "lineage_filter_canary_passed": audit.get("lineage_filter_canary_passed"),
                "time_travel_canaries_passed": audit.get("time_travel_canaries_passed"),
                "full_path_canary_provided": audit.get("full_path_canary_provided"),
                "full_path_canary": audit.get("full_path_canary"),
                "future_eligible_rows": audit.get("future_eligible_rows"),
                "status": audit.get("status"),
            }
        else:
            audit = self._insufficient_audit(
                run_id=run_id,
                as_of=cutoff.isoformat() if cutoff else as_of,
                scenario_family=scenario_family,
                cycles=cycles,
                offsets=offsets,
                missing_cutoff=cutoff is None,
                missing_lineage=lineage is None,
            )
            canary_artifact = {
                "run_id": run_id,
                "as_of": cutoff.isoformat() if cutoff else as_of,
                "generated_at": datetime.now(UTC).isoformat(),
                "lineage_filter_canary_passed": None,
                "time_travel_canaries_passed": None,
                "full_path_canary_provided": canary_evidence is not None,
                "full_path_canary": canary_evidence,
                "future_eligible_rows": None,
                "status": "insufficient_evidence",
                "detail": audit.get("detail"),
            }

        write_json(audit, out_dir / "as_of_audit.json")
        write_json(canary_artifact, out_dir / "time_travel_canaries.json")
        if write_run_artifact and run_dir is not None:
            write_json(audit, run_dir / "as_of_audit.json")
            write_json(canary_artifact, run_dir / "time_travel_canaries.json")
        passed = self._passed(audit)
        payload = {
            "run_id": run_id,
            "passed": passed,
            "exit_nonzero": not passed,
            "audit_path": str(out_dir / "as_of_audit.json"),
            "lineage_path": str(out_dir / "selected_feature_lineage.parquet"),
            "canary_path": str(out_dir / "time_travel_canaries.json"),
            "generated_at": datetime.now(UTC).isoformat(),
            "audit": audit,
            "canary": canary_artifact,
        }
        write_json(payload, out_dir / "as_of_verification.json")
        write_text(self._report(payload), out_dir / "as_of_verification_report.md")
        return payload

    def _resolve_run_dir(self, run_id: str) -> Path | None:
        from civic_signal.verification.publication import resolve_run_dir

        try:
            return resolve_run_dir(self.context.artifacts_dir, run_id)
        except FileNotFoundError:
            return None

    @staticmethod
    def _resolve_as_of(explicit: str | None, run_dir: Path | None) -> date | None:
        value = explicit
        manifest = run_dir / "run_manifest.json" if run_dir is not None else None
        if value is None and manifest is not None and manifest.exists():
            value = str(read_json(manifest).get("as_of") or "") or None
        if value is None:
            return None
        try:
            return date.fromisoformat(value[:10])
        except ValueError:
            return None

    @staticmethod
    def _load_lineage(run_dir: Path | None) -> pl.DataFrame | None:
        if run_dir is None:
            return None
        path = run_dir / "selected_feature_lineage.parquet"
        if not path.exists():
            return None
        return pl.read_parquet(path)

    def _compute_audit(
        self,
        *,
        run_id: str,
        cutoff: date,
        lineage: pl.DataFrame,
        scenario_family: str | None,
        cycles: str | None,
        offsets: str | None,
        canary_evidence: dict[str, Any] | None,
    ) -> dict[str, Any]:
        missing_columns = sorted(_CORE_LINEAGE_COLUMNS - set(lineage.columns))
        if missing_columns:
            return self._insufficient_audit(
                run_id=run_id,
                as_of=cutoff.isoformat(),
                scenario_family=scenario_family,
                cycles=cycles,
                offsets=offsets,
                missing_columns=missing_columns,
            )

        dated = lineage.with_columns(
            pl.col("available_at")
            .cast(pl.String)
            .str.slice(0, 10)
            .str.strptime(pl.Date, strict=False)
            .alias("_available_date")
        )
        future_rows = dated.filter(pl.col("_available_date") > cutoff).height
        missing_availability = dated.filter(pl.col("_available_date").is_null()).height
        implicit_availability = dated.filter(
            ~pl.col("availability_is_explicit").fill_null(False)
        ).height
        duplicate_column = "selection_key" if "selection_key" in dated.columns else "row_key"
        duplicate_keys = (
            dated.group_by(["table", duplicate_column]).len().filter(pl.col("len") > 1).height
        )
        lineage_canary_passed = self._time_travel_canary(dated, cutoff)
        full_canary_passed = (
            canary_evidence.get("passed") is True if canary_evidence is not None else None
        )
        canary_passed = lineage_canary_passed and full_canary_passed is not False
        selection_contract_complete = _SELECTION_CONTRACT_COLUMNS.issubset(lineage.columns)
        reasons: list[str] = []
        if future_rows:
            reasons.append(f"future-eligible rows after as_of: {future_rows}")
        if missing_availability:
            reasons.append(f"rows missing availability timestamps: {missing_availability}")
        if implicit_availability:
            reasons.append(f"rows with implicit event-date availability: {implicit_availability}")
        if duplicate_keys:
            reasons.append(f"duplicate selected snapshot keys: {duplicate_keys}")
        if not lineage_canary_passed:
            reasons.append("lineage filter canary failed after future-row injection")
        if full_canary_passed is False:
            reasons.append("full-path adversarial canary failed")
        if not selection_contract_complete:
            reasons.append("selection contract columns incomplete")
        status = "ok" if not reasons else "failed"
        return {
            "run_id": run_id,
            "scenario_family": scenario_family,
            "cycles": cycles,
            "offsets": offsets,
            "as_of": cutoff.isoformat(),
            "status": status,
            "recomputed": True,
            "lineage_row_count": lineage.height,
            "lineage_sha256": self._lineage_hash(lineage),
            "future_eligible_rows": future_rows,
            "violations": future_rows,
            "missing_availability_rows": missing_availability,
            "implicit_availability_rows": implicit_availability,
            "duplicate_snapshot_keys": duplicate_keys,
            "selection_contract_complete": selection_contract_complete,
            "time_travel_canaries_passed": canary_passed,
            "lineage_filter_canary_passed": lineage_canary_passed,
            "full_path_canary_provided": canary_evidence is not None,
            "full_path_canary": canary_evidence,
            "reasons": reasons,
            "generated_at": datetime.now(UTC).isoformat(),
        }

    @staticmethod
    def _insufficient_audit(
        *,
        run_id: str,
        as_of: str | None,
        scenario_family: str | None,
        cycles: str | None,
        offsets: str | None,
        missing_cutoff: bool = False,
        missing_lineage: bool = False,
        missing_columns: list[str] | None = None,
    ) -> dict[str, Any]:
        reasons = []
        if missing_cutoff:
            reasons.append("missing as_of cutoff")
        if missing_lineage:
            reasons.append("missing selected_feature_lineage.parquet")
        if missing_columns:
            reasons.append(f"missing lineage columns: {', '.join(missing_columns)}")
        return {
            "run_id": run_id,
            "scenario_family": scenario_family,
            "cycles": cycles,
            "offsets": offsets,
            "as_of": as_of,
            "status": "insufficient_evidence",
            "recomputed": False,
            "future_eligible_rows": None,
            "violations": None,
            "missing_availability_rows": None,
            "implicit_availability_rows": None,
            "duplicate_snapshot_keys": None,
            "selection_contract_complete": False,
            "time_travel_canaries_passed": None,
            "reasons": reasons,
            "detail": "; ".join(reasons) or "as-of evidence is incomplete",
        }

    @staticmethod
    def _time_travel_canary(lineage: pl.DataFrame, cutoff: date) -> bool:
        selected = lineage.filter(pl.col("_available_date") <= cutoff).drop("_available_date")
        before = AsOfVerificationRunner._lineage_hash(selected)
        future = {column: None for column in lineage.drop("_available_date").columns}
        future.update(
            {
                "table": "__canary__",
                "row_key": "future-row",
                "available_at": (cutoff + timedelta(days=1)).isoformat(),
                "availability_basis": "canary",
                "availability_is_explicit": True,
                "selected_as_of": cutoff.isoformat(),
            }
        )
        augmented = pl.concat(
            [lineage.drop("_available_date"), pl.DataFrame([future])],
            how="diagonal_relaxed",
        ).with_columns(
            pl.col("available_at")
            .cast(pl.String)
            .str.slice(0, 10)
            .str.strptime(pl.Date, strict=False)
            .alias("_available_date")
        )
        after = AsOfVerificationRunner._lineage_hash(
            augmented.filter(pl.col("_available_date") <= cutoff).drop("_available_date")
        )
        return before == after

    @staticmethod
    def _lineage_hash(lineage: pl.DataFrame) -> str:
        columns = sorted(column for column in lineage.columns if not column.startswith("_"))
        rows = lineage.select(columns).sort(columns).to_dicts() if columns else []
        payload = json.dumps(rows, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _passed(audit: dict[str, Any]) -> bool:
        return bool(
            audit.get("status") == "ok"
            and audit.get("recomputed") is True
            and audit.get("future_eligible_rows") == 0
            and audit.get("missing_availability_rows") == 0
            and audit.get("implicit_availability_rows") == 0
            and audit.get("duplicate_snapshot_keys") == 0
            and audit.get("time_travel_canaries_passed") is True
        )

    @staticmethod
    def _report(payload: dict[str, Any]) -> str:
        audit = payload["audit"]
        return (
            "# As-of verification\n\n"
            f"- passed: {payload['passed']}\n"
            f"- status: {audit.get('status')}\n"
            f"- as_of: {audit.get('as_of')}\n"
            f"- lineage rows: {audit.get('lineage_row_count')}\n"
            f"- future rows: {audit.get('future_eligible_rows')}\n"
            f"- missing availability: {audit.get('missing_availability_rows')}\n"
            f"- duplicate keys: {audit.get('duplicate_snapshot_keys')}\n"
            f"- selection contract complete: {audit.get('selection_contract_complete')}\n"
            f"- canary passed: {audit.get('time_travel_canaries_passed')}\n"
        )


def run_adversarial_time_travel_canary(
    bundle: FeatureBundle,
    *,
    as_of: str,
    selector: Callable[[FeatureBundle], FeatureBundle],
    forecast_fingerprint: Callable[[FeatureBundle], str],
    forecast_scope: str,
) -> dict[str, Any]:
    """Insert future revisions and prove the complete selector is invariant.

    The caller supplies the same selector used by the publication path and a
    deterministic forecast fingerprint function.  This lets fixture tests run
    a literal forecast while production runs can declare a narrower dependency
    fingerprint without overstating what was exercised.
    """
    cutoff = date.fromisoformat(as_of[:10])
    try:
        baseline = selector(bundle)
        augmented, injected_tables = _inject_future_revisions(bundle, cutoff)
        counterfactual = selector(augmented)
        baseline_features = _bundle_feature_hash(baseline)
        counterfactual_features = _bundle_feature_hash(counterfactual)
        baseline_tiers = _tier_hash(baseline.race_catalog)
        counterfactual_tiers = _tier_hash(counterfactual.race_catalog)
        baseline_forecast = forecast_fingerprint(baseline)
        counterfactual_forecast = forecast_fingerprint(counterfactual)
        feature_passed = baseline_features == counterfactual_features
        tier_passed = baseline_tiers == counterfactual_tiers
        forecast_passed = baseline_forecast == counterfactual_forecast
        return {
            "passed": bool(feature_passed and tier_passed and forecast_passed),
            "scope": forecast_scope,
            "cutoff": cutoff.isoformat(),
            "injected_tables": injected_tables,
            "selected_features_unchanged": feature_passed,
            "tiers_unchanged": tier_passed,
            "forecast_fingerprint_unchanged": forecast_passed,
            "baseline_feature_sha256": baseline_features,
            "counterfactual_feature_sha256": counterfactual_features,
            "baseline_tier_sha256": baseline_tiers,
            "counterfactual_tier_sha256": counterfactual_tiers,
            "baseline_forecast_sha256": baseline_forecast,
            "counterfactual_forecast_sha256": counterfactual_forecast,
        }
    except (ValueError, TypeError, pl.exceptions.PolarsError) as exc:
        return {
            "passed": False,
            "scope": forecast_scope,
            "cutoff": cutoff.isoformat(),
            "detail": f"canary execution failed: {exc}",
        }


def run_exact_publication_time_travel_canary(
    bundle: FeatureBundle,
    *,
    as_of: str,
    selector: Callable[[FeatureBundle], FeatureBundle],
    publication_runner: Callable[[FeatureBundle], dict[str, pl.DataFrame]],
) -> dict[str, Any]:
    """Inject future rows and compare the exact deterministic publication outputs."""
    cutoff = date.fromisoformat(as_of[:10])
    try:
        baseline = selector(bundle)
        augmented, injected_tables = _inject_future_revisions(bundle, cutoff)
        counterfactual = selector(augmented)
        baseline_outputs = publication_runner(baseline)
        counterfactual_outputs = publication_runner(counterfactual)
        baseline_features = _bundle_feature_hash(baseline)
        counterfactual_features = _bundle_feature_hash(counterfactual)
        baseline_tiers = _tier_hash(baseline.race_catalog)
        counterfactual_tiers = _tier_hash(counterfactual.race_catalog)
        output_names = (
            "posterior_draws",
            "component_estimates",
            "ensemble_center",
            "race_forecasts",
            "control_forecasts",
        )
        hashes: dict[str, dict[str, str]] = {}
        unchanged: dict[str, bool] = {}
        for name in output_names:
            before = baseline_outputs.get(name, pl.DataFrame())
            after = counterfactual_outputs.get(name, pl.DataFrame())
            before_hash = _json_hash(_frame_records(before))
            after_hash = _json_hash(_frame_records(after))
            hashes[name] = {"baseline_sha256": before_hash, "counterfactual_sha256": after_hash}
            unchanged[name] = before_hash == after_hash
        feature_passed = baseline_features == counterfactual_features
        tier_passed = baseline_tiers == counterfactual_tiers
        posterior_passed = unchanged["posterior_draws"]
        component_passed = unchanged["component_estimates"] and unchanged["ensemble_center"]
        race_passed = unchanged["race_forecasts"]
        control_passed = unchanged["control_forecasts"]
        forecast_passed = posterior_passed and component_passed and race_passed and control_passed
        required_tables = {
            name
            for name, frame in (
                ("polls", bundle.polls),
                ("market_quotes", bundle.markets),
                ("public_signals", bundle.public_signals),
                ("fundamentals", bundle.fundamentals),
            )
            if not frame.is_empty() and "available_at" in frame.columns
        }
        every_table_injected = required_tables.issubset(injected_tables)
        return {
            "passed": bool(
                feature_passed and tier_passed and forecast_passed and every_table_injected
            ),
            "scope": "exact_publication_pipeline",
            "cutoff": cutoff.isoformat(),
            "injected_tables": injected_tables,
            "required_injected_tables": sorted(required_tables),
            "every_time_varying_table_injected": every_table_injected,
            "selected_features_unchanged": feature_passed,
            "tiers_unchanged": tier_passed,
            "posterior_unchanged": posterior_passed,
            "component_center_unchanged": component_passed,
            "race_probabilities_unchanged": race_passed,
            "controls_unchanged": control_passed,
            "forecast_fingerprint_unchanged": forecast_passed,
            "baseline_feature_sha256": baseline_features,
            "counterfactual_feature_sha256": counterfactual_features,
            "baseline_tier_sha256": baseline_tiers,
            "counterfactual_tier_sha256": counterfactual_tiers,
            "output_hashes": hashes,
        }
    except (ValueError, TypeError, KeyError, pl.exceptions.PolarsError) as exc:
        return {
            "passed": False,
            "scope": "exact_publication_pipeline",
            "cutoff": cutoff.isoformat(),
            "detail": f"exact publication canary execution failed: {exc}",
        }


def _inject_future_revisions(
    bundle: FeatureBundle, cutoff: date
) -> tuple[FeatureBundle, list[str]]:
    from dataclasses import replace

    future = cutoff + timedelta(days=1)
    tables = {
        "polls": bundle.polls,
        "market_quotes": bundle.markets,
        "public_signals": bundle.public_signals,
        "fundamentals": bundle.fundamentals,
    }
    augmented: dict[str, pl.DataFrame] = {}
    injected: list[str] = []
    for table, frame in tables.items():
        if frame.is_empty() or "available_at" not in frame.columns:
            augmented[table] = frame
            continue
        row = frame.head(1)
        expressions: list[pl.Expr] = [
            _future_value_expression(frame, "available_at", future),
        ]
        if "published_at" in frame.columns:
            expressions.append(_future_value_expression(frame, "published_at", future))
        if "revision_id" in frame.columns:
            expressions.append(
                pl.lit("999999").cast(frame.schema["revision_id"]).alias("revision_id")
            )
        for value_column in ("pct", "probability", "value", "z_score", "economic_index"):
            if value_column in frame.columns:
                expressions.append(
                    pl.lit(999.0).cast(frame.schema[value_column]).alias(value_column)
                )
                break
        future_row = row.with_columns(expressions)
        augmented[table] = pl.concat([frame, future_row], how="vertical_relaxed")
        injected.append(table)
    return (
        replace(
            bundle,
            polls=augmented["polls"],
            markets=augmented["market_quotes"],
            public_signals=augmented["public_signals"],
            fundamentals=augmented["fundamentals"],
        ),
        injected,
    )


def _future_value_expression(frame: pl.DataFrame, column: str, future: date) -> pl.Expr:
    dtype = frame.schema[column]
    if dtype == pl.Date:
        value: object = future
    elif isinstance(dtype, pl.Datetime):
        value = datetime.combine(future, datetime.min.time(), tzinfo=UTC)
    else:
        value = future.isoformat()
    return pl.lit(value).cast(dtype).alias(column)


def _bundle_feature_hash(bundle: FeatureBundle) -> str:
    payload = {
        "polls": _frame_records(bundle.polls),
        "market_quotes": _frame_records(bundle.markets),
        "public_signals": _frame_records(bundle.public_signals),
        "fundamentals": _frame_records(bundle.fundamentals),
    }
    return _json_hash(payload)


def _tier_hash(catalog: pl.DataFrame) -> str:
    columns = [
        column
        for column in ("race_id", "tier", "tier_reason", "poll_count", "market_count")
        if column in catalog.columns
    ]
    return _json_hash(_frame_records(catalog.select(columns)))


def _frame_records(frame: pl.DataFrame) -> list[dict[str, Any]]:
    columns = sorted(frame.columns)
    return frame.select(columns).sort(columns).to_dicts() if columns else []


def _json_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _row_key_expression(columns: list[str]) -> pl.Expr:
    if not columns:
        return pl.lit("__unkeyed__")
    return pl.concat_str(
        [pl.col(column).cast(pl.String).fill_null("<null>") for column in columns],
        separator="|",
    )


def _string_column(frame: pl.DataFrame, column: str) -> pl.Expr:
    if column in frame.columns:
        return pl.col(column).cast(pl.String)
    return pl.lit(None, dtype=pl.String)


def _empty_lineage() -> pl.DataFrame:
    return pl.DataFrame(schema=_LINEAGE_SCHEMA)
