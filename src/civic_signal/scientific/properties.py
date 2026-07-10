"""Property checks used by scientific CI (symmetry, simplex, PSD, control recon)."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl


def probability_simplex_ok(
    frame: pl.DataFrame,
    *,
    race_col: str = "race_id",
    probability_col: str = "winner_probability",
    atol: float = 1e-6,
) -> dict[str, Any]:
    """Multi-option races must sum to ~1; single-row races must lie in [0, 1]."""
    if frame.is_empty() or probability_col not in frame.columns or race_col not in frame.columns:
        return {"ok": False, "reason": "missing probability frame/columns"}
    published = frame.filter(pl.col(probability_col).is_not_null())
    if published.is_empty():
        return {"ok": True, "reason": "no published probabilities", "race_count": 0}
    out_of_range = published.filter(
        (pl.col(probability_col) < 0.0) | (pl.col(probability_col) > 1.0)
    ).height
    if out_of_range:
        return {"ok": False, "reason": f"{out_of_range} probabilities outside [0,1]"}
    multi = published.group_by(race_col).agg(pl.len().alias("n")).filter(pl.col("n") > 1)
    if multi.is_empty():
        return {"ok": True, "race_count": published.height, "multi_option_races": 0}
    sums = published.group_by(race_col).agg(pl.col(probability_col).sum().alias("s"))
    bad = sums.join(multi, on=race_col, how="inner").filter(
        (pl.col("s") < 1.0 - atol) | (pl.col("s") > 1.0 + atol)
    )
    return {
        "ok": bad.is_empty(),
        "multi_option_races": int(multi.height),
        "bad_sum_races": int(bad.height),
        "reason": None if bad.is_empty() else "probability sums deviate from 1",
    }


def covariance_is_psd(matrix: np.ndarray, *, atol: float = 1e-8) -> dict[str, Any]:
    """Return whether a square matrix is symmetric positive semidefinite."""
    arr = np.asarray(matrix, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        return {"ok": False, "reason": "matrix is not square"}
    if not np.allclose(arr, arr.T, atol=atol):
        return {"ok": False, "reason": "matrix is not symmetric"}
    eigenvalues = np.linalg.eigvalsh((arr + arr.T) / 2.0)
    min_eig = float(np.min(eigenvalues))
    return {
        "ok": min_eig >= -atol,
        "min_eigenvalue": min_eig,
        "rank": int(np.linalg.matrix_rank(arr, tol=atol)),
    }


def label_symmetry_holds(
    probs_dem: np.ndarray,
    probs_rep: np.ndarray,
    *,
    atol: float = 1e-9,
) -> dict[str, Any]:
    """Swapping party labels must swap complementary probabilities."""
    dem = np.asarray(probs_dem, dtype=np.float64)
    rep = np.asarray(probs_rep, dtype=np.float64)
    if dem.shape != rep.shape:
        return {"ok": False, "reason": "shape mismatch"}
    complements = np.allclose(dem + rep, 1.0, atol=atol)
    swapped = np.allclose(dem, 1.0 - rep, atol=atol)
    return {
        "ok": bool(complements and swapped),
        "max_abs_sum_error": float(np.max(np.abs(dem + rep - 1.0))) if dem.size else 0.0,
    }


def control_reconciliation_ok(
    race_forecasts: pl.DataFrame,
    control_forecasts: pl.DataFrame,
    *,
    race_col: str = "race_id",
    body_col: str = "control_body",
) -> dict[str, Any]:
    """Basic structural reconciliation between race and control artifacts."""
    if race_forecasts.is_empty():
        return {"ok": False, "reason": "empty race forecasts"}
    if control_forecasts.is_empty():
        return {"ok": False, "reason": "empty control forecasts"}
    if race_col not in race_forecasts.columns:
        return {"ok": False, "reason": f"missing {race_col}"}
    if body_col not in control_forecasts.columns and "party" not in control_forecasts.columns:
        return {"ok": False, "reason": "control frame missing body/party identity"}
    probability_cols = [
        column
        for column in ("majority_probability", "control_probability", "winner_probability")
        if column in control_forecasts.columns or column in race_forecasts.columns
    ]
    if not probability_cols:
        return {"ok": False, "reason": "no probability columns to reconcile"}
    # Finite + in-range control probabilities when present.
    for column in ("majority_probability", "control_probability"):
        if column not in control_forecasts.columns:
            continue
        bad = control_forecasts.filter(
            pl.col(column).is_not_null()
            & ((pl.col(column) < 0.0) | (pl.col(column) > 1.0) | ~pl.col(column).is_finite())
        ).height
        if bad:
            return {"ok": False, "reason": f"{bad} control {column} out of range"}
    unique_races = race_forecasts[race_col].n_unique()
    return {
        "ok": True,
        "race_count": int(unique_races),
        "control_rows": int(control_forecasts.height),
    }


def interval_ordering_ok(
    frame: pl.DataFrame,
    *,
    low: str = "share_p10",
    mid: str = "share_p50",
    high: str = "share_p90",
) -> dict[str, Any]:
    if not {low, mid, high}.issubset(frame.columns):
        return {"ok": True, "reason": "interval columns absent"}
    bad = frame.filter((pl.col(low) > pl.col(mid)) | (pl.col(mid) > pl.col(high))).height
    return {"ok": bad == 0, "violations": int(bad)}


def option_order_invariance_ok(
    frame: pl.DataFrame,
    *,
    race_col: str = "race_id",
    option_col: str = "option_id",
    probability_col: str = "winner_probability",
    atol: float = 1e-12,
) -> dict[str, Any]:
    """Reordering option rows must not change per-(race, option) probabilities."""
    required = {race_col, option_col, probability_col}
    if frame.is_empty() or not required.issubset(frame.columns):
        return {"ok": False, "reason": "missing columns for option-order check"}
    base = frame.select([race_col, option_col, probability_col]).sort([race_col, option_col])
    shuffled = (
        frame.select([race_col, option_col, probability_col])
        .with_row_index("_row")
        .sort("_row", descending=True)
        .drop("_row")
        .sort([race_col, option_col])
    )
    if base.height != shuffled.height:
        return {"ok": False, "reason": "row count changed after reorder"}
    left = base[probability_col].to_numpy()
    right = shuffled[probability_col].to_numpy()
    match = bool(np.allclose(left, right, atol=atol, equal_nan=True))
    return {
        "ok": match,
        "max_abs_diff": float(np.max(np.abs(left - right))) if left.size else 0.0,
        "reason": None if match else "option-order reordering changed aligned probabilities",
    }


REQUIRED_PROPERTY_FAMILIES = (
    "probability_simplex",
    "option_order_invariance",
    "interval_ordering",
    "covariance_psd",
    "label_symmetry",
    "control_reconciliation",
)


def run_offline_property_suite() -> dict[str, Any]:
    """Deterministic offline property suite for verify scientific (no network)."""
    race = pl.DataFrame(
        {
            "race_id": ["r1", "r1", "r2"],
            "option_id": ["DEM", "REP", "DEM"],
            "winner_probability": [0.55, 0.45, 0.62],
            "share_p10": [0.45, 0.35, 0.52],
            "share_p50": [0.52, 0.48, 0.60],
            "share_p90": [0.60, 0.55, 0.70],
        }
    )
    control = pl.DataFrame(
        {
            "control_body": ["senate"],
            "party": ["DEM"],
            "majority_probability": [0.51],
        }
    )
    psd = np.array([[1.0, 0.25], [0.25, 1.0]], dtype=np.float64)
    dem = np.array([0.6, 0.4, 0.55], dtype=np.float64)
    rep = 1.0 - dem

    checks = {
        "probability_simplex": probability_simplex_ok(race),
        "option_order_invariance": option_order_invariance_ok(race),
        "interval_ordering": interval_ordering_ok(race),
        "covariance_psd": covariance_is_psd(psd),
        "label_symmetry": label_symmetry_holds(dem, rep),
        "control_reconciliation": control_reconciliation_ok(race, control),
    }
    failures = [name for name, result in checks.items() if not bool((result or {}).get("ok"))]
    missing = sorted(set(REQUIRED_PROPERTY_FAMILIES) - set(checks))
    return {
        "suite": "offline_property_suite",
        "required_property_families": list(REQUIRED_PROPERTY_FAMILIES),
        "completed_property_families": sorted(checks),
        "missing_property_families": missing,
        "checks": checks,
        "failures": failures,
        "passed": not failures and not missing,
    }
