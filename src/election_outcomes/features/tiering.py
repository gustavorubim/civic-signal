from __future__ import annotations

import polars as pl


class TierAssessor:
    def __init__(self, tier_config: dict[str, object]) -> None:
        self.config = tier_config

    def assign(
        self,
        races: pl.DataFrame,
        polls: pl.DataFrame,
        markets: pl.DataFrame,
        fundamentals: pl.DataFrame,
        public_signals: pl.DataFrame,
    ) -> pl.DataFrame:
        counts = self._counts(races, polls, markets, fundamentals, public_signals)
        return (
            races.join(counts, on="race_id", how="left")
            .with_columns(
                [
                    pl.col("poll_count").fill_null(0),
                    pl.col("pollster_count").fill_null(0),
                    pl.col("market_count").fill_null(0),
                    pl.col("fundamental_count").fill_null(0),
                    pl.col("public_signal_count").fill_null(0),
                ]
            )
            .with_columns(
                pl.struct(
                    [
                        "poll_count",
                        "pollster_count",
                        "market_count",
                        "fundamental_count",
                        "public_signal_count",
                        "race_type",
                    ]
                )
                .map_elements(
                    self._tier_for_row,
                    return_dtype=pl.Struct({"tier": pl.String, "tier_reason": pl.String}),
                )
                .alias("tier_struct")
            )
            .unnest("tier_struct")
        )

    def _counts(
        self,
        races: pl.DataFrame,
        polls: pl.DataFrame,
        markets: pl.DataFrame,
        fundamentals: pl.DataFrame,
        public_signals: pl.DataFrame,
    ) -> pl.DataFrame:
        frames = [races.select("race_id").unique()]
        frames.append(
            polls.group_by("race_id").agg(
                pl.len().alias("poll_count"),
                pl.col("pollster").n_unique().alias("pollster_count"),
            )
        )
        frames.append(markets.group_by("race_id").agg(pl.len().alias("market_count")))
        frames.append(fundamentals.group_by("race_id").agg(pl.len().alias("fundamental_count")))
        frames.append(public_signals.group_by("race_id").agg(pl.len().alias("public_signal_count")))
        result = frames[0]
        for frame in frames[1:]:
            result = result.join(frame, on="race_id", how="left")
        return result

    def _tier_for_row(self, row: dict[str, object]) -> dict[str, str]:
        tier_a = self.config.get("tier_a", {})
        tier_b = self.config.get("tier_b", {})
        poll_count = int(row["poll_count"] or 0)
        pollster_count = int(row["pollster_count"] or 0)
        market_count = int(row["market_count"] or 0)
        fundamental_count = int(row["fundamental_count"] or 0)
        public_count = int(row["public_signal_count"] or 0)
        has_a_polling = poll_count >= int(tier_a.get("min_polls", 2)) and pollster_count >= int(
            tier_a.get("min_pollsters", 2)
        )
        has_a_market = market_count >= int(tier_a.get("min_market_quotes", 1))
        has_fundamentals = fundamental_count >= int(tier_a.get("min_fundamental_rows", 1))
        if has_fundamentals and (has_a_polling or has_a_market):
            return {"tier": "A", "tier_reason": "Validated polls/markets plus fundamentals."}
        any_signal = poll_count + market_count + fundamental_count + public_count
        if any_signal >= int(tier_b.get("min_any_signal_rows", 1)) and has_fundamentals:
            return {
                "tier": "B",
                "tier_reason": "Sparse forecast with fundamentals and wide uncertainty.",
            }
        reason = str(self.config.get("tier_c", {}).get("reason", "Insufficient validated data."))
        return {"tier": "C", "tier_reason": reason}
