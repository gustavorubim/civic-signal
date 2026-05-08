from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import polars as pl

from election_outcomes.config import ProjectContext
from election_outcomes.storage.io import write_parquet


@dataclass(frozen=True)
class CuratedBuildResult:
    tables: dict[str, pl.DataFrame]


class CuratedDataBuilder:
    """Normalize raw snapshots into typed curated Parquet tables."""

    NUMERIC_COLUMNS: ClassVar[set[str]] = {
        "cycle",
        "seats",
        "measure_threshold",
        "incumbent",
        "previous_vote_share",
        "fundraising_usd",
        "sample_size",
        "pct",
        "probability",
        "spread",
        "volume",
        "open_interest",
        "value",
        "z_score",
        "partisan_lean",
        "incumbency_advantage",
        "economic_index",
        "demographic_turnout_index",
        "historical_turnout_rate",
        "registered_voters",
        "vote_share",
        "turnout",
        "winner",
        "actual_winner",
        "actual_vote_share",
        "baseline_probability",
        "polls_probability",
        "fundamentals_probability",
        "markets_probability",
        "public_signals_probability",
        "ensemble_probability",
        "predicted_vote_share",
        "lower_90",
        "upper_90",
    }
    BOOL_COLUMNS: ClassVar[set[str]] = {"incumbent", "winner", "actual_winner", "leakage_checked"}
    INT_COLUMNS: ClassVar[set[str]] = {"cycle", "seats", "sample_size"}

    def __init__(self, context: ProjectContext) -> None:
        self.context = context

    def run(self) -> CuratedBuildResult:
        manifest_path = self.context.raw_dir / "source_manifest.parquet"
        if not manifest_path.exists():
            raise FileNotFoundError("Run sync before build-features")
        manifest = pl.read_parquet(manifest_path).filter(pl.col("status") != "failed")
        tables: dict[str, pl.DataFrame] = {}
        self.context.curated_dir.mkdir(parents=True, exist_ok=True)

        for row in manifest.iter_rows(named=True):
            table = str(row["table"])
            frame = self._read_source(row)
            tables[table] = frame
            write_parquet(frame, self.context.curated_dir / f"{table}.parquet")
        usage_manifest = manifest.with_columns(
            pl.concat_str([pl.lit("curated:"), pl.col("table")]).alias("downstream_usage")
        )
        write_parquet(usage_manifest, self.context.curated_dir / "source_manifest.parquet")
        return CuratedBuildResult(tables)

    def _read_source(self, row: dict[str, object]) -> pl.DataFrame:
        frame = pl.read_csv(
            str(row["raw_path"]),
            infer_schema_length=1000,
            null_values=["", "null", "None"],
            try_parse_dates=True,
        )
        if frame.columns:
            frame = frame.filter(pl.col(frame.columns[0]).is_not_null())
        frame = self._coerce(frame)
        return frame.with_columns(
            pl.lit(row["source_id"]).alias("source_id"),
            pl.lit(row["content_hash"]).alias("source_hash"),
            pl.lit(row["parser_version"]).alias("parser_version"),
        )

    def _coerce(self, frame: pl.DataFrame) -> pl.DataFrame:
        expressions = []
        for column in frame.columns:
            if column in self.BOOL_COLUMNS:
                expressions.append(pl.col(column).cast(pl.Boolean, strict=False))
            elif column in self.INT_COLUMNS:
                expressions.append(pl.col(column).cast(pl.Int64, strict=False))
            elif column in self.NUMERIC_COLUMNS:
                expressions.append(pl.col(column).cast(pl.Float64, strict=False))
        return frame.with_columns(expressions) if expressions else frame
