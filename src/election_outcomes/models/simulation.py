from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import polars as pl

from election_outcomes.features import FeatureBundle
from election_outcomes.performance.kernels import (
    NUMBA_AVAILABLE,
    configure_numba_threads,
    simulate_binary_draw_arrays,
)


@dataclass(frozen=True)
class SimulationOutputs:
    draws: pl.DataFrame
    race_forecasts: pl.DataFrame
    control_forecasts: pl.DataFrame
    ecosystem_forecasts: pl.DataFrame
    performance: dict[str, object]


class SimulationEngine:
    def __init__(self, config: dict[str, object]) -> None:
        self.config = config
        self.seed = int(config.get("seed", 20260508))
        self.draw_count = int(config.get("simulation_count", 1000))
        uncertainty = dict(config.get("uncertainty", {}))
        self.tier_sigma = {
            "A": float(uncertainty.get("tier_a_sigma", 0.035)),
            "B": float(uncertainty.get("tier_b_sigma", 0.075)),
        }
        performance = dict(config.get("performance", {}))
        requested_engine = str(performance.get("engine", "numba"))
        self.parallel = bool(performance.get("parallel", True))
        self.numba_threads = configure_numba_threads(int(performance.get("numba_threads", 0) or 0))
        self.use_numba = requested_engine == "numba" and self.parallel and NUMBA_AVAILABLE
        self.engine = "numba" if self.use_numba else "python"
        self.requested_engine = requested_engine

    def run(self, bundle: FeatureBundle, ensemble: pl.DataFrame) -> SimulationOutputs:
        draws = self._draws(bundle, ensemble)
        forecasts = self._race_forecasts(bundle, draws)
        control = self._control_forecasts(bundle, draws)
        ecosystem = self._ecosystem_forecasts(bundle, draws)
        return SimulationOutputs(draws, forecasts, control, ecosystem, self.performance_metadata())

    def performance_metadata(self) -> dict[str, object]:
        return {
            "requested_engine": self.requested_engine,
            "engine": self.engine,
            "parallel": self.parallel,
            "numba_available": NUMBA_AVAILABLE,
            "numba_threads": self.numba_threads,
            "simulation_count": self.draw_count,
        }

    def _draws(self, bundle: FeatureBundle, ensemble: pl.DataFrame) -> pl.DataFrame:
        if ensemble.is_empty():
            return pl.DataFrame()
        rng = np.random.default_rng(self.seed)
        catalog = {row["race_id"]: row for row in bundle.race_catalog.iter_rows(named=True)}
        options_by_race = {}
        for key, group in bundle.options.group_by("race_id", maintain_order=True):
            race_key = key[0] if isinstance(key, tuple) else key
            options_by_race[str(race_key)] = group.sort("option_id")
        estimates = {
            row["race_id"]: row
            for row in ensemble.sort("option_id")
            .group_by("race_id", maintain_order=True)
            .map_groups(lambda group: group.head(1))
            .iter_rows(named=True)
        }
        fundamentals = {row["race_id"]: row for row in bundle.fundamentals.iter_rows(named=True)}
        binary_specs: list[dict[str, object]] = []
        multi_option_specs: list[dict[str, object]] = []
        national_error = rng.normal(0, 0.015, self.draw_count)
        for race_id, options in options_by_race.items():
            race = catalog[race_id]
            if race["tier"] == "C" or race_id not in estimates:
                continue
            estimate_rows = ensemble.filter(pl.col("race_id") == race_id).sort("option_id")
            first = estimate_rows.row(0, named=True)
            sigma = max(
                self.tier_sigma.get(str(race["tier"]), 0.08), float(first["uncertainty"]) * 0.5
            )
            if len(options) == 2:
                binary_specs.append(
                    {
                        "race_id": race_id,
                        "options": options,
                        "first_share": float(first["vote_share"]),
                        "turnout_base": self._turnout_base(str(race_id), fundamentals),
                        "local_error": rng.standard_t(df=5, size=self.draw_count)
                        * sigma
                        / np.sqrt(5 / 3),
                    }
                )
                continue
            multi_option_specs.append(
                {
                    "race_id": race_id,
                    "options": options,
                    "estimate_rows": estimate_rows,
                    "turnout_base": self._turnout_base(str(race_id), fundamentals),
                    "local_error": rng.standard_t(df=5, size=self.draw_count)
                    * sigma
                    / np.sqrt(5 / 3),
                }
            )

        frames: list[pl.DataFrame] = []
        binary_frame = self._binary_draw_frame(binary_specs, national_error)
        if not binary_frame.is_empty():
            frames.append(binary_frame)
        multi_frame = self._multi_option_draw_frame(multi_option_specs, national_error, rng)
        if not multi_frame.is_empty():
            frames.append(multi_frame)
        return pl.concat(frames, how="vertical") if frames else pl.DataFrame()

    def _binary_draw_frame(
        self, specs: list[dict[str, object]], national_error: np.ndarray
    ) -> pl.DataFrame:
        if not specs:
            return pl.DataFrame()
        first_shares = np.array([spec["first_share"] for spec in specs], dtype=np.float64)
        turnout_bases = np.array([spec["turnout_base"] for spec in specs], dtype=np.float64)
        local_errors = np.vstack([spec["local_error"] for spec in specs]).astype(np.float64)
        (
            draw_ids,
            correlated_error_draw_ids,
            race_indices,
            option_indices,
            turnouts,
            vote_shares,
            winners,
        ) = simulate_binary_draw_arrays(
            first_shares,
            turnout_bases,
            national_error.astype(np.float64),
            local_errors,
            self.use_numba,
        )
        draw_frame = pl.DataFrame(
            {
                "draw_id": draw_ids,
                "correlated_error_draw_id": correlated_error_draw_ids,
                "race_index": race_indices,
                "option_index": option_indices,
                "turnout": turnouts,
                "vote_share": vote_shares,
                "winner": winners,
            }
        )
        race_map = pl.DataFrame(
            {
                "race_index": list(range(len(specs))),
                "race_id": [str(spec["race_id"]) for spec in specs],
            }
        )
        option_rows = []
        for race_index, spec in enumerate(specs):
            options = spec["options"]
            for option_index, option in enumerate(options.iter_rows(named=True)):
                option_rows.append(
                    {
                        "race_index": race_index,
                        "option_index": option_index,
                        "option_id": option["option_id"],
                        "party": option["party"],
                    }
                )
        option_map = pl.DataFrame(option_rows)
        return (
            draw_frame.join(race_map, on="race_index", how="left")
            .join(option_map, on=["race_index", "option_index"], how="left")
            .select(
                [
                    "draw_id",
                    "correlated_error_draw_id",
                    "race_id",
                    "option_id",
                    "party",
                    "turnout",
                    "vote_share",
                    "winner",
                ]
            )
        )

    def _multi_option_draw_frame(
        self,
        specs: list[dict[str, object]],
        national_error: np.ndarray,
        rng: np.random.Generator,
    ) -> pl.DataFrame:
        rows: list[dict[str, object]] = []
        for spec in specs:
            race_id = str(spec["race_id"])
            options = spec["options"]
            option_shares = self._multi_option_shares(spec["estimate_rows"], rng)
            turnout_base = float(spec["turnout_base"])
            for draw_id in range(self.draw_count):
                shares = [float(series[draw_id]) for series in option_shares]
                winner_index = int(np.argmax(shares))
                turnout = round(turnout_base * max(0.6, 1 + national_error[draw_id]))
                for index, option in enumerate(options.iter_rows(named=True)):
                    rows.append(
                        {
                            "draw_id": draw_id,
                            "correlated_error_draw_id": draw_id,
                            "race_id": race_id,
                            "option_id": option["option_id"],
                            "party": option["party"],
                            "turnout": turnout,
                            "vote_share": shares[index],
                            "winner": index == winner_index,
                        }
                    )
        return pl.DataFrame(rows)

    def _multi_option_shares(
        self,
        estimate_rows: pl.DataFrame,
        rng: np.random.Generator,
    ) -> list[np.ndarray]:
        shares = estimate_rows.sort("option_id")["vote_share"].to_numpy()
        alpha = np.maximum(shares * 70, 1.0)
        sampled = rng.dirichlet(alpha, size=self.draw_count)
        return [sampled[:, index] for index in range(sampled.shape[1])]

    @staticmethod
    def _turnout_base(race_id: str, fundamentals: dict[str, dict[str, object]]) -> float:
        row = fundamentals.get(race_id, {})
        voters = float(row.get("registered_voters") or 100_000)
        turnout_rate = float(row.get("historical_turnout_rate") or 0.5)
        return voters * turnout_rate

    def _race_forecasts(self, bundle: FeatureBundle, draws: pl.DataFrame) -> pl.DataFrame:
        catalog = bundle.race_catalog
        options = bundle.options
        if draws.is_empty():
            base = options.join(
                catalog.select(["race_id", "tier", "tier_reason"]), on="race_id", how="left"
            )
            return base.select(
                "race_id",
                "option_id",
                "tier",
                "tier_reason",
                pl.lit(None, dtype=pl.Float64).alias("winner_probability"),
            )
        intervals = draws.group_by(["race_id", "option_id"]).agg(
            pl.col("winner").mean().alias("winner_probability"),
            pl.col("vote_share").mean().alias("vote_share_mean"),
            pl.col("vote_share").median().alias("vote_share_median"),
            pl.col("vote_share").quantile(0.25).alias("vote_share_p25"),
            pl.col("vote_share").quantile(0.75).alias("vote_share_p75"),
            pl.col("vote_share").quantile(0.10).alias("vote_share_p10"),
            pl.col("vote_share").quantile(0.90).alias("vote_share_p90"),
            pl.col("vote_share").quantile(0.05).alias("vote_share_p05"),
            pl.col("vote_share").quantile(0.95).alias("vote_share_p95"),
            pl.col("vote_share").quantile(0.025).alias("vote_share_p025"),
            pl.col("vote_share").quantile(0.975).alias("vote_share_p975"),
        )
        base = options.join(
            catalog.select(["race_id", "tier", "tier_reason"]), on="race_id", how="left"
        )
        joined = base.join(intervals, on=["race_id", "option_id"], how="left")
        return joined.with_columns(
            pl.when(pl.col("tier") == "C")
            .then(None)
            .otherwise(pl.col("winner_probability"))
            .alias("winner_probability"),
            pl.when(pl.col("tier") == "C")
            .then(pl.lit("probability_withheld"))
            .otherwise(pl.lit("trusted_probability"))
            .alias("data_quality_flags"),
        )

    def _control_forecasts(self, bundle: FeatureBundle, draws: pl.DataFrame) -> pl.DataFrame:
        if draws.is_empty():
            return pl.DataFrame()
        winner_draws = draws.filter(pl.col("winner")).join(
            bundle.race_catalog.select(["race_id", "office_type", "control_body"]),
            on="race_id",
            how="left",
        )
        rows: list[dict[str, object]] = []
        for key, group in winner_draws.group_by(["control_body", "party"], maintain_order=True):
            control_body, party = key
            if not control_body:
                continue
            counts = (
                group.group_by("draw_id").agg(pl.len().alias("seat_count"))["seat_count"].to_numpy()
            )
            tipping = (
                group.group_by("race_id")
                .agg(pl.len().alias("wins"))
                .sort("wins")
                .head(3)["race_id"]
                .to_list()
            )
            rows.append(
                {
                    "control_body": control_body,
                    "party": party,
                    "seat_count_mean": float(np.mean(counts)),
                    "seat_count_p10": float(np.quantile(counts, 0.10)),
                    "seat_count_p50": float(np.quantile(counts, 0.50)),
                    "seat_count_p90": float(np.quantile(counts, 0.90)),
                    "control_probability": float(np.mean(counts >= max(np.median(counts), 1))),
                    "tipping_point_races": json.dumps(tipping),
                }
            )
        return pl.DataFrame(rows)

    def _ecosystem_forecasts(self, bundle: FeatureBundle, draws: pl.DataFrame) -> pl.DataFrame:
        catalog = {row["race_id"]: row for row in bundle.race_catalog.iter_rows(named=True)}
        rows: list[dict[str, object]] = []
        for race_id, group in draws.group_by("race_id", maintain_order=True):
            race_key = str(race_id[0] if isinstance(race_id, tuple) else race_id)
            pivot = group.group_by("draw_id").agg(
                pl.col("vote_share").sort(descending=True).head(2).alias("top_two"),
                pl.col("turnout").max().alias("turnout"),
            )
            margins = np.array([values[0] - values[1] for values in pivot["top_two"].to_list()])
            turnout = pivot["turnout"].to_numpy()
            rows.append(
                {
                    "race_id": race_key,
                    "tier": catalog[race_key]["tier"],
                    "turnout_mean": float(np.mean(turnout)),
                    "turnout_p10": float(np.quantile(turnout, 0.10)),
                    "turnout_p90": float(np.quantile(turnout, 0.90)),
                    "demographic_composition": json.dumps(
                        {"supported": bool(catalog[race_key]["tier"] != "C")}
                    ),
                    "recount_probability": float(np.mean(margins <= 0.01)),
                    "certification_risk_probability": float(np.mean(margins <= 0.005) * 0.6),
                    "ballot_measure_supported": bool(
                        catalog[race_key]["race_type"] == "ballot_measure"
                    ),
                }
            )
        return pl.DataFrame(rows)
