"""Honest quality labels (audit P1-9)."""

from __future__ import annotations

from types import SimpleNamespace

import polars as pl

from civic_signal.models import SimulationEngine
from civic_signal.normalize import CuratedDataBuilder
from civic_signal.reports.silver_benchmark import effective_bayesian_backend


def _engine(calibration: dict, **extra) -> SimulationEngine:
    return SimulationEngine(
        {"simulation_count": 4, "probability_calibration": calibration, **extra}
    )


def test_calibration_labels() -> None:
    assert _engine({})._calibration_label() == "not_configured"
    identity = _engine({"status": "fitted", "slope": 1.01, "intercept": 0.0})
    assert identity._calibration_label() == "identity_on_synthetic_backtest"
    fitted = _engine({"status": "fitted", "slope": 0.7, "intercept": 0.1})
    assert fitted._calibration_label() == "fitted_on_synthetic_backtest"
    real = _engine({"status": "fitted", "slope": 0.7, "training_basis": "real_results"})
    assert real._calibration_label() == "fitted_on_real_results"
    frame = pl.DataFrame({"winner_probability": [0.6, None]})
    out = identity._apply_probability_calibration(frame)
    assert out["probability_calibration_status"].unique().to_list() == [
        "identity_on_synthetic_backtest"
    ]


def test_backend_used_is_reported() -> None:
    engine = _engine(
        {}, _inference_engine="bayes", _bayesian_backend="analytic", bayesian={"backend": "nuts"}
    )
    assert engine.performance_metadata()["bayesian_backend_used"] == "analytic"
    assert _engine({}).performance_metadata()["bayesian_backend_used"] is None
    assert effective_bayesian_backend({"bayesian": {"backend": "nuts"}}) == "nuts"
    assert (
        effective_bayesian_backend(
            {"_bayesian_backend": "analytic", "bayesian": {"backend": "nuts"}}
        )
        == "analytic"
    )


def test_synthetic_fundamentals_flag() -> None:
    engine = _engine({})
    bundle = SimpleNamespace(
        fundamentals=pl.DataFrame(
            {"race_id": ["A", "B", "C"], "synthetic_inputs": ["economic_index", "", None]}
        )
    )
    assert engine._synthetic_fundamentals_races(bundle) == {"A"}
    assert (
        engine._synthetic_fundamentals_races(SimpleNamespace(fundamentals=pl.DataFrame())) == set()
    )


def test_builder_synthetic_inputs_expr() -> None:
    base = pl.DataFrame(
        {"synthetic": ["True", "False", "False"], "synthetic_fields": ["", "economic_index", ""]}
    )
    out = base.select(CuratedDataBuilder._synthetic_inputs_expr(base))["synthetic_inputs"]
    assert out.to_list() == ["all", "economic_index", ""]
    bare = pl.DataFrame({"x": [1]})
    assert bare.select(CuratedDataBuilder._synthetic_inputs_expr(bare)).item() is None
    only_fields = pl.DataFrame({"synthetic_fields": ["a"]})
    assert only_fields.select(CuratedDataBuilder._synthetic_inputs_expr(only_fields)).item() == "a"
