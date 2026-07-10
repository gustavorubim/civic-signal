"""Scoring and reward evaluation."""

from __future__ import annotations

from typing import Any

__all__ = [
    "BacktestRunner",
    "CycleEvaluationReport",
    "NestedBacktestRunner",
    "ResultComparator",
    "RewardEvaluator",
    "RewardV2Evaluator",
    "score_predictions",
]


def __getattr__(name: str) -> Any:
    # Lazy exports avoid circular imports with civic_signal.models.
    if name == "BacktestRunner":
        from civic_signal.scoring.backtest import BacktestRunner

        return BacktestRunner
    if name == "NestedBacktestRunner":
        from civic_signal.scoring.backtest import NestedBacktestRunner

        return NestedBacktestRunner
    if name == "CycleEvaluationReport":
        from civic_signal.scoring.cycle_eval import CycleEvaluationReport

        return CycleEvaluationReport
    if name == "ResultComparator":
        from civic_signal.scoring.results import ResultComparator

        return ResultComparator
    if name == "RewardEvaluator":
        from civic_signal.scoring.rewards import RewardEvaluator

        return RewardEvaluator
    if name == "RewardV2Evaluator":
        from civic_signal.scoring.reward_v2 import RewardV2Evaluator

        return RewardV2Evaluator
    if name == "score_predictions":
        from civic_signal.scoring.metrics import score_predictions

        return score_predictions
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
