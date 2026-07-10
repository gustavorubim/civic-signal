from __future__ import annotations

from dataclasses import replace
from datetime import date

import polars as pl

from civic_signal.features.builder import FeatureBundle

_SNAPSHOT_IDENTITY_CANDIDATES: dict[str, tuple[tuple[str, ...], ...]] = {
    "polls": (("question_id",), ("poll_id",), ("survey_id",)),
    "market_quotes": (("market_id",), ("provider",)),
    "public_signals": (("signal_type",), ("signal_id",)),
    "fundamentals": (
        ("feature_id",),
        ("series_id",),
        ("rating_id",),
        ("finance_id",),
        ("feature_type",),
    ),
}

_EVENT_COLUMN_CANDIDATES = {
    "polls": ("end_date",),
    "market_quotes": ("observed_at",),
    "public_signals": ("observed_at",),
    "fundamentals": ("observed_at", "as_of"),
}


def subset_bundle(bundle: FeatureBundle, race_catalog: pl.DataFrame) -> FeatureBundle:
    active_ids = race_catalog["race_id"].to_list() if "race_id" in race_catalog.columns else []

    def by_race(frame: pl.DataFrame) -> pl.DataFrame:
        if "race_id" not in frame.columns:
            return frame
        return frame.filter(pl.col("race_id").is_in(active_ids))

    return replace(
        bundle,
        races=by_race(bundle.races),
        options=by_race(bundle.options),
        polls=by_race(bundle.polls),
        markets=by_race(bundle.markets),
        public_signals=by_race(bundle.public_signals),
        fundamentals=by_race(bundle.fundamentals),
        results=by_race(bundle.results),
        backtest_predictions=by_race(bundle.backtest_predictions),
        race_catalog=race_catalog,
    )


def filter_bundle_by_date(bundle: FeatureBundle, as_of: str) -> FeatureBundle:
    fundamentals = select_latest_eligible_snapshots(bundle.fundamentals, "fundamentals", as_of)
    fundamentals = add_incumbent_relative_economic_features(fundamentals, bundle.options)
    options = apply_vintage_option_features(bundle.options, fundamentals)
    return replace(
        bundle,
        options=options,
        polls=select_latest_eligible_snapshots(bundle.polls, "polls", as_of),
        markets=select_latest_eligible_snapshots(bundle.markets, "market_quotes", as_of),
        public_signals=select_latest_eligible_snapshots(
            bundle.public_signals, "public_signals", as_of
        ),
        fundamentals=fundamentals,
    )


def select_latest_eligible_snapshots(frame: pl.DataFrame, table: str, as_of: str) -> pl.DataFrame:
    """Select one deterministic eligible revision per race/feature/horizon key.

    Rows are filtered on both their event time and recorded availability before
    revision selection.  This ordering is important: a later revision cannot
    displace an older, genuinely available value in a historical forecast.
    """
    frame = with_snapshot_identity(frame, table)
    event_column = snapshot_event_column(frame, table)
    if frame.is_empty() or event_column not in frame.columns:
        return frame
    cutoff = date.fromisoformat(as_of[:10])
    event_date = _date_expression(event_column)
    eligible = event_date.is_not_null() & (event_date <= cutoff)
    if "available_at" in frame.columns:
        available_date = _date_expression("available_at")
        eligible = eligible & available_date.is_not_null() & (available_date <= cutoff)
    if "published_at" in frame.columns:
        published_date = _date_expression("published_at")
        eligible = eligible & (published_date.is_null() | (published_date <= cutoff))
    filtered = frame.filter(eligible)
    if filtered.is_empty():
        return filtered

    key_columns = snapshot_selection_key_columns(filtered, table)
    if not key_columns:
        return filtered
    original_columns = filtered.columns
    order_columns = [
        *key_columns,
        "_selection_observed_at",
        "_selection_published_at",
        "_selection_available_at",
    ]
    selected = (
        filtered.with_columns(
            _timestamp_expression(filtered, event_column, event_column).alias(
                "_selection_observed_at"
            ),
            _timestamp_expression(filtered, "available_at", event_column).alias(
                "_selection_available_at"
            ),
            _timestamp_expression(filtered, "published_at", event_column).alias(
                "_selection_published_at"
            ),
            _revision_numeric_expression(filtered).alias("_selection_revision_numeric"),
            _string_expression(filtered, "revision_id", "").alias("_selection_revision_text"),
            pl.struct(sorted(original_columns)).hash(seed=0).alias("_selection_row_hash"),
        )
        .sort(
            [
                *order_columns,
                "_selection_revision_numeric",
                "_selection_revision_text",
                "_selection_row_hash",
            ],
            nulls_last=False,
        )
        .unique(subset=key_columns, keep="last", maintain_order=True)
        .select(original_columns)
    )
    canonical = [column for column in [*key_columns, event_column] if column in selected.columns]
    return selected.sort(canonical) if canonical else selected


def snapshot_selection_key_columns(frame: pl.DataFrame, table: str) -> list[str]:
    """Return the canonical race/feature/horizon identity present in a table."""
    columns: list[str] = []
    if "race_id" in frame.columns:
        columns.append("race_id")
    if "snapshot_identity" in frame.columns:
        columns.append("snapshot_identity")
    else:
        for candidate in _SNAPSHOT_IDENTITY_CANDIDATES[table]:
            present = [column for column in candidate if column in frame.columns]
            if present:
                columns.extend(present)
                break
    if table in {"polls", "market_quotes", "public_signals", "fundamentals"} and (
        "option_id" in frame.columns
    ):
        columns.append("option_id")
    for horizon in ("horizon", "horizon_days", "as_of_offset_days"):
        if horizon in frame.columns:
            columns.append(horizon)
            break
    return list(dict.fromkeys(columns))


def snapshot_selection_predicate(table: str, frame: pl.DataFrame | None = None) -> str:
    """Machine-readable description of the shared snapshot selection rule."""
    event_column = snapshot_event_column(
        frame if frame is not None else pl.DataFrame(schema={}), table
    )
    return (
        f"{event_column}<=as_of AND published_at<=as_of AND available_at<=as_of; group by "
        "race/feature/option/horizon; choose "
        "max(observed_at,published_at,available_at,revision_id,row_hash)"
    )


def snapshot_event_column(frame: pl.DataFrame, table: str) -> str:
    candidates = _EVENT_COLUMN_CANDIDATES[table]
    return next((column for column in candidates if column in frame.columns), candidates[-1])


def with_snapshot_identity(frame: pl.DataFrame, table: str) -> pl.DataFrame:
    """Attach a coalesced canonical identity across heterogeneous source schemas."""
    if frame.is_empty() or "snapshot_identity" in frame.columns:
        return frame
    expressions: list[pl.Expr] = []
    for candidate in _SNAPSHOT_IDENTITY_CANDIDATES[table]:
        column = candidate[0]
        if column in frame.columns:
            expressions.append(
                pl.when(pl.col(column).is_not_null())
                .then(pl.concat_str([pl.lit(f"{column}:"), pl.col(column).cast(pl.String)]))
                .otherwise(None)
            )
    default_identity = "wide_fundamental_snapshot" if table == "fundamentals" else "missing"
    identity = (
        pl.coalesce(expressions).fill_null(default_identity)
        if expressions
        else pl.lit(default_identity)
    )
    return frame.with_columns(identity.alias("snapshot_identity"))


def add_incumbent_relative_economic_features(
    fundamentals: pl.DataFrame,
    options: pl.DataFrame,
) -> pl.DataFrame:
    """Express economic conditions on the incumbent-party axis after vintage selection."""
    if fundamentals.is_empty() or "race_id" not in fundamentals.columns:
        return fundamentals
    incumbent_party: dict[str, str] = {}
    major_party_races: set[str] = set()
    if {"race_id", "party", "incumbent"}.issubset(options.columns):
        for race_key, group in options.group_by("race_id"):
            race_id = str(race_key[0] if isinstance(race_key, tuple) else race_key)
            parties = {str(value or "").upper() for value in group["party"].to_list()}
            if parties & {"DEM", "REP"}:
                major_party_races.add(race_id)
        for row in options.filter(pl.col("incumbent").fill_null(False)).iter_rows(named=True):
            incumbent_party[str(row["race_id"])] = str(row.get("party") or "")
    signs: list[float] = []
    sources: list[str] = []
    applied: list[bool] = []
    relative_values: list[float] = []
    for row in fundamentals.iter_rows(named=True):
        explicit = row.get("incumbent_party")
        party = str(explicit or incumbent_party.get(str(row["race_id"]), ""))
        sign = _party_sign(party)
        economic_value = float(row.get("economic_index") or 0.0)
        signs.append(sign)
        sources.append("fundamental_record" if explicit else "candidate_incumbency")
        applied.append(
            sign != 0.0 or economic_value == 0.0 or str(row["race_id"]) not in major_party_races
        )
        relative_values.append(economic_value * sign if sign != 0.0 else 0.0)
    return fundamentals.with_columns(
        pl.Series("incumbent_party_sign", signs, dtype=pl.Float64),
        pl.Series("incumbent_party_sign_source", sources, dtype=pl.String),
        pl.Series("incumbent_relative_sign_applied", applied, dtype=pl.Boolean),
        pl.Series("economic_index_incumbent_relative", relative_values, dtype=pl.Float64),
    )


def apply_vintage_option_features(
    options: pl.DataFrame,
    fundamentals: pl.DataFrame,
) -> pl.DataFrame:
    """Overlay selected option-level finance and rating vintages onto model options."""
    if options.is_empty() or fundamentals.is_empty() or "option_id" not in fundamentals.columns:
        return _ensure_option_model_columns(options)
    updates: dict[tuple[str, str], dict[str, float]] = {}
    for row in fundamentals.iter_rows(named=True):
        race_id = str(row.get("race_id") or "")
        option_id = str(row.get("option_id") or "")
        if not race_id or not option_id:
            continue
        feature_type = str(row.get("feature_type") or "").lower()
        values = updates.setdefault((race_id, option_id), {})
        finance = _first_numeric(row, "fundraising_usd", "finance_value")
        rating = _first_numeric(row, "rating_score", "rating_value")
        generic = _first_numeric(row, "value")
        if finance is not None or feature_type in {"finance", "fundraising"}:
            values["fundraising_usd"] = finance if finance is not None else float(generic or 0.0)
        if rating is not None or feature_type in {"rating", "ratings"}:
            values["rating_score"] = rating if rating is not None else float(generic or 0.0)
    if not updates:
        return options
    rows = []
    for (race_id, option_id), values in sorted(updates.items()):
        rows.append(
            {
                "race_id": race_id,
                "option_id": option_id,
                "_vintage_fundraising_usd": values.get("fundraising_usd"),
                "_vintage_rating_score": values.get("rating_score"),
            }
        )
    overlay = pl.DataFrame(rows)
    joined = options.join(overlay, on=["race_id", "option_id"], how="left")
    existing_finance = (
        pl.col("fundraising_usd").cast(pl.Float64)
        if "fundraising_usd" in joined.columns
        else pl.lit(None, dtype=pl.Float64)
    )
    existing_rating = (
        pl.col("rating_score").cast(pl.Float64)
        if "rating_score" in joined.columns
        else pl.lit(None, dtype=pl.Float64)
    )
    updated = joined.with_columns(
        pl.coalesce([pl.col("_vintage_fundraising_usd"), existing_finance]).alias(
            "fundraising_usd"
        ),
        pl.coalesce([pl.col("_vintage_rating_score"), existing_rating]).alias("rating_score"),
        pl.col("_vintage_fundraising_usd").is_not_null().alias("fundraising_vintage_applied"),
        pl.col("_vintage_rating_score").is_not_null().alias("rating_vintage_applied"),
    ).drop("_vintage_fundraising_usd", "_vintage_rating_score")
    return _ensure_option_model_columns(updated)


def _ensure_option_model_columns(options: pl.DataFrame) -> pl.DataFrame:
    """Guarantee option columns expected by the fundamentals ridge feature set."""
    if options.is_empty():
        return options
    extras: list[pl.Expr] = []
    if "fundraising_usd" not in options.columns:
        extras.append(pl.lit(None, dtype=pl.Float64).alias("fundraising_usd"))
    if "rating_score" not in options.columns:
        extras.append(pl.lit(None, dtype=pl.Float64).alias("rating_score"))
    if "fundraising_vintage_applied" not in options.columns:
        extras.append(pl.lit(False).alias("fundraising_vintage_applied"))
    if "rating_vintage_applied" not in options.columns:
        extras.append(pl.lit(False).alias("rating_vintage_applied"))
    return options.with_columns(extras) if extras else options


def _first_numeric(row: dict[str, object], *columns: str) -> float | None:
    for column in columns:
        value = row.get(column)
        if value is not None:
            return float(value)
    return None


def _party_sign(party: object) -> float:
    value = str(party or "").upper()
    if value in {"DEM", "YES"}:
        return 1.0
    if value in {"REP", "NO"}:
        return -1.0
    return 0.0


def feature_vintage_lineage_summary(
    fundamentals: pl.DataFrame,
    canary: dict[str, object],
) -> dict[str, object]:
    """Summarize which selected vintage contracts have enough evidence to claim."""
    if fundamentals.is_empty():
        return {
            "incumbent_relative_sign": None,
            "end_of_cycle_finance_in_early_fold": None,
            "revised_macro_in_early_fold": None,
            "revised_rating_in_early_fold": None,
            "vintage_rows": {"macro": 0, "finance": 0, "ratings": 0},
        }
    feature_type = (
        pl.col("feature_type").cast(pl.String).str.to_lowercase()
        if "feature_type" in fundamentals.columns
        else pl.lit("")
    )
    macro_mask = feature_type.is_in(["macro", "economy", "economic"])
    finance_mask = feature_type.is_in(["finance", "fundraising"])
    rating_mask = feature_type.is_in(["rating", "ratings"])
    if "economic_index" in fundamentals.columns:
        macro_mask = macro_mask | pl.col("economic_index").is_not_null()
    if "fundraising_usd" in fundamentals.columns:
        finance_mask = finance_mask | pl.col("fundraising_usd").is_not_null()
    if "finance_value" in fundamentals.columns:
        finance_mask = finance_mask | pl.col("finance_value").is_not_null()
    if "rating_score" in fundamentals.columns:
        rating_mask = rating_mask | pl.col("rating_score").is_not_null()
    if "rating_value" in fundamentals.columns:
        rating_mask = rating_mask | pl.col("rating_value").is_not_null()
    counts = {
        "macro": fundamentals.filter(macro_mask).height,
        "finance": fundamentals.filter(finance_mask).height,
        "ratings": fundamentals.filter(rating_mask).height,
    }
    required = {"available_at", "published_at", "revision_id", "snapshot_identity"}
    event_column = snapshot_event_column(fundamentals, "fundamentals")

    def complete(mask: pl.Expr) -> bool:
        subset = fundamentals.filter(mask)
        columns = {*required, event_column}
        return bool(
            not subset.is_empty()
            and columns.issubset(subset.columns)
            and all(subset[column].null_count() == 0 for column in columns)
        )

    completeness = {
        "macro": complete(macro_mask),
        "finance": complete(finance_mask),
        "ratings": complete(rating_mask),
    }
    canary_passed = canary.get("passed") is True and "fundamentals" in list(
        canary.get("injected_tables") or []
    )
    sign = None
    if "incumbent_relative_sign_applied" in fundamentals.columns:
        sign = bool(fundamentals["incumbent_relative_sign_applied"].all())
    return {
        "incumbent_relative_sign": sign,
        "end_of_cycle_finance_in_early_fold": (
            False if counts["finance"] > 0 and completeness["finance"] and canary_passed else None
        ),
        "revised_macro_in_early_fold": (
            False if counts["macro"] > 0 and completeness["macro"] and canary_passed else None
        ),
        "revised_rating_in_early_fold": (
            False if counts["ratings"] > 0 and completeness["ratings"] and canary_passed else None
        ),
        "vintage_lineage_complete": completeness,
        "vintage_rows": counts,
    }


def _date_expression(column: str) -> pl.Expr:
    return pl.col(column).cast(pl.String).str.slice(0, 10).str.strptime(pl.Date, strict=False)


def _timestamp_expression(frame: pl.DataFrame, column: str, fallback: str) -> pl.Expr:
    selected = column if column in frame.columns else fallback
    return pl.col(selected).cast(pl.String).str.to_datetime(strict=False, time_zone="UTC")


def _revision_numeric_expression(frame: pl.DataFrame) -> pl.Expr:
    if "revision_id" not in frame.columns:
        return pl.lit(-1, dtype=pl.Int64)
    return pl.col("revision_id").cast(pl.String).cast(pl.Int64, strict=False).fill_null(-1)


def _string_expression(frame: pl.DataFrame, column: str, default: str) -> pl.Expr:
    if column not in frame.columns:
        return pl.lit(default, dtype=pl.String)
    return pl.col(column).cast(pl.String).fill_null(default)


def filter_results_before_cycle(bundle: FeatureBundle, target_cycle: int) -> FeatureBundle:
    historical_ids = (
        bundle.race_catalog.filter(pl.col("cycle") < target_cycle)["race_id"].to_list()
        if "cycle" in bundle.race_catalog.columns
        else []
    )

    def historical(frame: pl.DataFrame) -> pl.DataFrame:
        if "race_id" not in frame.columns:
            return frame
        return frame.filter(pl.col("race_id").is_in(historical_ids))

    return replace(
        bundle,
        races=historical(bundle.races),
        options=historical(bundle.options),
        polls=historical(bundle.polls),
        markets=historical(bundle.markets),
        public_signals=historical(bundle.public_signals),
        fundamentals=historical(bundle.fundamentals),
        results=historical(bundle.results),
        backtest_predictions=historical(bundle.backtest_predictions),
        race_catalog=bundle.race_catalog.filter(pl.col("race_id").is_in(historical_ids)),
    )
