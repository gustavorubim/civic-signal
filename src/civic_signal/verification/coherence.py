"""Recompute race, draw, elector, and chamber-control coherence from run artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import polars as pl

from civic_signal.config import ProjectContext
from civic_signal.scientific.properties import probability_simplex_ok
from civic_signal.storage.io import write_json, write_text


@dataclass(frozen=True)
class CoherenceVerificationRunner:
    context: ProjectContext

    def verify(self, *, run_id: str, audit_id: str | None = None) -> dict[str, Any]:
        run_dir = self.context.artifacts_dir / "runs" / run_id
        if not run_dir.exists():
            raise FileNotFoundError(f"Forecast run not found: {run_dir}")
        required = {
            "catalog": run_dir / "race_catalog.parquet",
            "forecasts": run_dir / "race_forecasts.parquet",
            "draws": run_dir / "forecast_draws.parquet",
            "controls": run_dir / "control_forecasts.parquet",
        }
        missing = [name for name, path in required.items() if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Missing coherence artifacts: {', '.join(missing)}")
        catalog = pl.read_parquet(required["catalog"])
        forecasts = pl.read_parquet(required["forecasts"])
        draws = pl.read_parquet(required["draws"])
        controls = pl.read_parquet(required["controls"])
        checks = {
            "race_probability_simplex": self._race_simplex(forecasts),
            "race_key_uniqueness": self._unique(forecasts, ["race_id", "option_id"]),
            "tier_c_withholding": self._tier_c_withholding(catalog, forecasts),
            "draw_key_uniqueness": self._unique(draws, ["draw_id", "race_id", "option_id"]),
            "draw_simplex_and_winner": self._draw_coherence(draws),
            "control_key_uniqueness": self._unique(controls, ["control_body", "party"]),
            "control_probability_range": self._probability_range(
                controls, ["majority_probability", "control_probability"]
            ),
            "control_reconstruction": self._control_reconstruction(catalog, draws, controls),
            "senate_tie_vp": self._senate_tie_vp(controls),
            "maine_nebraska_electors": self._maine_nebraska(catalog),
        }
        passed = all(bool(check["passed"]) for check in checks.values())
        audit_id = audit_id or run_id
        output_dir = self.context.artifacts_dir / "coherence" / audit_id
        payload = {
            "schema_version": "1.0.0",
            "run_id": run_id,
            "audit_id": audit_id,
            "generated_at": datetime.now(UTC).isoformat(),
            "status": "passed" if passed else "failed",
            "passed": passed,
            "checks": checks,
            "output_dir": str(output_dir),
        }
        write_json(payload, output_dir / "coherence_verification.json")
        write_text(self._report(payload), output_dir / "coherence_verification.md")
        return payload

    @staticmethod
    def _race_simplex(frame: pl.DataFrame) -> dict[str, Any]:
        result = probability_simplex_ok(frame)
        return {"passed": bool(result.pop("ok")), **result}

    @staticmethod
    def _unique(frame: pl.DataFrame, keys: list[str]) -> dict[str, Any]:
        missing = [key for key in keys if key not in frame.columns]
        if missing:
            return {"passed": False, "missing_columns": missing}
        duplicates = frame.group_by(keys).len().filter(pl.col("len") > 1).height
        return {"passed": duplicates == 0, "duplicate_keys": duplicates, "keys": keys}

    @staticmethod
    def _probability_range(frame: pl.DataFrame, columns: list[str]) -> dict[str, Any]:
        present = [column for column in columns if column in frame.columns]
        if not present:
            return {"passed": False, "reason": "no probability columns"}
        bad = 0
        for column in present:
            bad += frame.filter(
                pl.col(column).is_not_null()
                & (~pl.col(column).is_finite() | (pl.col(column) < 0) | (pl.col(column) > 1))
            ).height
        return {"passed": bad == 0, "out_of_range_values": bad, "columns": present}

    @staticmethod
    def _tier_c_withholding(catalog: pl.DataFrame, forecasts: pl.DataFrame) -> dict[str, Any]:
        if "tier" not in catalog.columns or "winner_probability" not in forecasts.columns:
            return {"passed": False, "reason": "missing tier/probability columns"}
        tier_c = catalog.filter(pl.col("tier") == "C").select("race_id")
        published = forecasts.join(tier_c, on="race_id", how="inner").filter(
            pl.col("winner_probability").is_not_null()
        )
        return {
            "passed": published.is_empty(),
            "tier_c_races": tier_c.height,
            "published_tier_c_rows": published.height,
        }

    @staticmethod
    def _draw_coherence(draws: pl.DataFrame, atol: float = 1e-9) -> dict[str, Any]:
        required = {"draw_id", "race_id", "vote_share", "winner"}
        if not required.issubset(draws.columns):
            return {"passed": False, "missing_columns": sorted(required - set(draws.columns))}
        grouped = draws.group_by(["draw_id", "race_id"]).agg(
            pl.col("vote_share").sum().alias("share_sum"),
            pl.col("winner").cast(pl.Int64).sum().alias("winner_count"),
        )
        bad = grouped.filter(
            ((pl.col("share_sum") - 1.0).abs() > atol) | (pl.col("winner_count") != 1)
        )
        return {
            "passed": bad.is_empty(),
            "bad_draw_races": bad.height,
            "draw_races": grouped.height,
        }

    @staticmethod
    def _control_reconstruction(
        catalog: pl.DataFrame,
        draws: pl.DataFrame,
        controls: pl.DataFrame,
        atol: float = 1e-12,
    ) -> dict[str, Any]:
        """Recompute chamber majority rates from draws and compare to controls.

        Does not trust stored majority/control probabilities: each control row is
        rebuilt as mean_d(I[seats_won(d) + holdovers >= threshold]).
        """
        if controls.is_empty():
            return {"passed": True, "status": "not_applicable", "rows": 0}
        required = {"draw_id", "race_id", "party", "winner"}
        if not required.issubset(draws.columns):
            return {"passed": False, "missing_draw_columns": sorted(required - set(draws.columns))}
        catalog_cols = ["race_id"]
        for column in ("control_body", "seats"):
            if column not in catalog.columns:
                return {"passed": False, "missing_catalog_columns": [column]}
            catalog_cols.append(column)
        joined = draws.join(catalog.select(catalog_cols), on="race_id", how="left")
        draw_ids = sorted(joined["draw_id"].unique().to_list())
        if not draw_ids:
            return {"passed": False, "reason": "no draws available for reconstruction", "rows": 0}
        failures = 0
        max_error = 0.0
        for row in controls.iter_rows(named=True):
            body = row.get("control_body")
            party = row.get("party")
            threshold = int(row.get("control_threshold") or 0)
            holdovers = int(row.get("holdover_seats") or 0)
            wins = joined.filter(
                (pl.col("control_body") == body) & (pl.col("party") == party) & pl.col("winner")
            )
            counts = {
                item["draw_id"]: float(item["seats"] or 0.0)
                for item in wins.group_by("draw_id")
                .agg(pl.col("seats").sum())
                .iter_rows(named=True)
            }
            reconstructed = sum(
                counts.get(draw_id, 0.0) + holdovers >= threshold for draw_id in draw_ids
            ) / float(len(draw_ids))
            # Compare every published control probability field that is present.
            for field in ("majority_probability", "control_probability"):
                if field not in row or row.get(field) is None:
                    continue
                reported = float(row[field])
                error = abs(reconstructed - reported)
                max_error = max(max_error, error)
                failures += int(error > atol)
        return {
            "passed": failures == 0,
            "mismatched_control_rows": failures,
            "max_absolute_error": max_error,
            "rows": controls.height,
            "draw_count": len(draw_ids),
            "method": "recompute_from_forecast_draws",
        }

    def _senate_tie_vp(self, controls: pl.DataFrame) -> dict[str, Any]:
        config = self.context.read_yaml("model.yaml")
        party = str(config.get("control_tiebreak_party") or "").upper()
        bodies = {str(value).lower() for value in config.get("control_tiebreak_bodies", ["senate"])}
        senate = controls.filter(
            pl.col("control_body").cast(pl.Utf8).str.to_lowercase() == "senate"
        )
        if senate.is_empty():
            return {"passed": True, "status": "not_applicable", "rows": 0}
        configured = int(dict(config.get("control_thresholds", {})).get("senate", 51))
        bad = 0
        for row in senate.iter_rows(named=True):
            expected = (
                configured - 1
                if "senate" in bodies and str(row["party"]).upper() == party
                else configured
            )
            bad += int(int(row["control_threshold"]) != expected)
        return {
            "passed": bad == 0,
            "tiebreak_party": party,
            "configured_threshold": configured,
            "bad_threshold_rows": bad,
        }

    @staticmethod
    def _maine_nebraska(catalog: pl.DataFrame) -> dict[str, Any]:
        if not {"office_type", "geography", "seats"}.issubset(catalog.columns):
            return {"passed": False, "reason": "missing elector columns"}
        president = catalog.filter(pl.col("office_type") == "president")
        failures = 0
        states: dict[str, Any] = {}
        for state, expected_total in (("ME", 4), ("NE", 5)):
            rows = president.filter(pl.col("geography").cast(pl.Utf8).str.starts_with(state))
            if rows.is_empty():
                states[state] = {"status": "not_in_scope"}
                continue
            seats = sorted(int(value) for value in rows["seats"].to_list())
            if rows.height == 1:
                ok = False
                status = "aggregate_state_allocation_cannot_prove_split_electors"
            else:
                ok = sum(seats) == expected_total and seats == [1] * (expected_total - 2) + [2]
                status = "district_split_allocation"
            failures += int(not ok)
            states[state] = {"status": status, "seats": seats, "passed": ok}
        return {"passed": failures == 0, "bad_states": failures, "states": states}

    @staticmethod
    def _report(payload: dict[str, Any]) -> str:
        lines = ["# Coherence Verification", "", f"Passed: **{payload['passed']}**", ""]
        for name, check in payload["checks"].items():
            lines.append(f"- {name}: {'pass' if check['passed'] else 'fail'}")
        return "\n".join(lines) + "\n"
