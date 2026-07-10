"""Load and validate golden scientific fixtures for M7 CI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl

from civic_signal.scientific.properties import (
    control_reconciliation_ok,
    interval_ordering_ok,
    option_order_invariance_ok,
    probability_simplex_ok,
)


def default_fixture_root() -> Path:
    packaged = Path(__file__).resolve().parent / "golden_fixtures"
    if packaged.exists():
        return packaged
    # Prefer checkout layout: <repo>/src/civic_signal/scientific/this.py
    candidate = Path(__file__).resolve().parents[3] / "tests" / "golden_fixtures"
    if candidate.exists():
        return candidate
    # Fallback for unusual layouts during packaging smoke.
    return Path.cwd() / "tests" / "golden_fixtures"


def load_json_fixture(name: str, *, root: Path | None = None) -> dict[str, Any]:
    path = (root or default_fixture_root()) / name
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def frame_from_records(records: list[dict[str, Any]]) -> pl.DataFrame:
    return pl.DataFrame(records) if records else pl.DataFrame()


def validate_golden_bundle(root: Path | None = None) -> dict[str, Any]:
    """Validate the four required golden cases against property checks."""
    fixture_root = root or default_fixture_root()
    cases: dict[str, Any] = {}

    small = load_json_fixture("small_election.json", root=fixture_root)
    race = frame_from_records(list(small.get("race_forecasts") or []))
    cases["small_election"] = {
        "simplex": probability_simplex_ok(race),
        "intervals": interval_ordering_ok(race),
        "option_order": option_order_invariance_ok(race),
        "expected_race_count": small.get("expected_race_count"),
        "actual_race_count": int(race["race_id"].n_unique()) if "race_id" in race.columns else 0,
    }

    chamber = load_json_fixture("chamber_control.json", root=fixture_root)
    race_c = frame_from_records(list(chamber.get("race_forecasts") or []))
    control = frame_from_records(list(chamber.get("control_forecasts") or []))
    cases["chamber_control"] = {
        "reconciliation": control_reconciliation_ok(race_c, control),
        "expected_control_rows": chamber.get("expected_control_rows"),
        "actual_control_rows": int(control.height),
    }

    multi = load_json_fixture("multi_option_race.json", root=fixture_root)
    race_m = frame_from_records(list(multi.get("race_forecasts") or []))
    cases["multi_option_race"] = {
        "simplex": probability_simplex_ok(race_m),
        "option_order": option_order_invariance_ok(race_m),
        "expected_options": multi.get("expected_options"),
        "actual_options": int(race_m.height),
    }

    quarantine = load_json_fixture("quarantine_failure.json", root=fixture_root)
    race_q = frame_from_records(list(quarantine.get("race_forecasts") or []))
    simplex_q = probability_simplex_ok(race_q)
    cases["quarantine_failure"] = {
        "simplex": simplex_q,
        "expect_fail": bool(quarantine.get("expect_property_failure", True)),
        "failed_as_expected": (not simplex_q["ok"])
        if quarantine.get("expect_property_failure", True)
        else simplex_q["ok"],
    }

    parity = load_json_fixture("parity_fingerprint.json", root=fixture_root)
    cases["parity_fingerprint"] = parity

    failures: list[str] = []
    if not cases["small_election"]["simplex"]["ok"]:
        failures.append("small_election simplex")
    if (
        cases["small_election"]["actual_race_count"]
        != cases["small_election"]["expected_race_count"]
    ):
        failures.append("small_election race count")
    if not cases["chamber_control"]["reconciliation"]["ok"]:
        failures.append("chamber_control reconciliation")
    if (
        cases["chamber_control"]["actual_control_rows"]
        != cases["chamber_control"]["expected_control_rows"]
    ):
        failures.append("chamber_control row count")
    if not cases["multi_option_race"]["simplex"]["ok"]:
        failures.append("multi_option simplex")
    if not cases["quarantine_failure"]["failed_as_expected"]:
        failures.append("quarantine did not fail as expected")

    return {
        "passed": not failures,
        "failures": failures,
        "cases": cases,
    }
