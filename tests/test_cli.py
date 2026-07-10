from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import polars as pl
import pytest
import typer

from civic_signal import cli


class _FakePipeline:
    calls: ClassVar[list[tuple[str, dict[str, object]]]] = []

    def __init__(self, context: object) -> None:
        self.context = context

    def _record(self, name: str, **kwargs: object) -> None:
        self.calls.append((name, kwargs))

    def sync(self) -> pl.DataFrame:
        self._record("sync")
        return pl.DataFrame({"source_id": ["public"]})

    def build_features(self) -> SimpleNamespace:
        self._record("build_features")
        return SimpleNamespace(race_catalog=pl.DataFrame({"race_id": ["race"]}))

    def run_forecast(self, **kwargs: object) -> Path:
        self._record("run_forecast", **kwargs)
        return Path("/tmp/forecast")

    def run_daily_update(self, **kwargs: object) -> dict[str, object]:
        self._record("run_daily_update", **kwargs)
        return {"strategy": "full_refit", "needs_full_refit": True, "output_dir": "/tmp/update"}

    def run_backtest(self, **kwargs: object) -> dict[str, object]:
        self._record("run_backtest", **kwargs)
        return {"row_count": 3}

    def run_nested_backtest(self, **kwargs: object) -> dict[str, object]:
        self._record("run_nested_backtest", **kwargs)
        return {"fold_count": 2, "row_count": 6}

    def refresh_hyperpriors(self, **kwargs: object) -> dict[str, object]:
        self._record("refresh_hyperpriors", **kwargs)
        return {"scenarios": ["a"], "promoted": False, "output_dir": "/tmp/hyper"}

    def run_phase0_spike(self, **kwargs: object) -> dict[str, object]:
        self._record("run_phase0_spike", **kwargs)
        return {
            "go_no_go": {"status": "insufficient_evidence", "bayes_minus_kalman": None},
            "output_dir": "/tmp/phase0",
        }

    def run_phase0b_spike(self, **kwargs: object) -> dict[str, object]:
        self._record("run_phase0b_spike", **kwargs)
        return {
            "selected_strategy": "full_refit",
            "global_smc_rejected": True,
            "output_dir": "/tmp/phase0b",
        }

    def rebuild_report(self, **kwargs: object) -> Path:
        self._record("rebuild_report", **kwargs)
        return Path("/tmp/report")

    def verify_run(self, **kwargs: object) -> dict[str, object]:
        self._record("verify_run", **kwargs)
        return {"passed": False, "output_dir": "/tmp/verify"}

    def assess_methodology_readiness(self, **kwargs: object) -> dict[str, object]:
        self._record("assess_methodology_readiness", **kwargs)
        return {
            "eligible_for_default_switch": False,
            "status": "insufficient_evidence",
            "output_dir": "/tmp/readiness",
        }

    def verify_historical_calibration(self, **kwargs: object) -> dict[str, object]:
        self._record("verify_historical_calibration", **kwargs)
        return {"passed": False, "output_dir": "/tmp/calibration"}

    def run_benchmark(self, **kwargs: object) -> dict[str, object]:
        self._record("run_benchmark", **kwargs)
        return {"rows_per_second": 1000.0, "performance": {"engine": "python"}}

    def compare_results(self, **kwargs: object) -> dict[str, object]:
        self._record("compare_results", **kwargs)
        return {
            "race_count": 2,
            "winner_accuracy": 0.5,
            "output_dir": "/tmp/comparison",
        }

    def run_cycle_eval(self, **kwargs: object) -> dict[str, object]:
        self._record("run_cycle_eval", **kwargs)
        return {
            "cycle_count": 2,
            "aggregate": {
                "mean_state_accuracy": 0.75,
                "mean_brier_score": None,
                "majority_winner_accuracy": 1.0,
            },
            "output_dir": "/tmp/cycle-eval",
        }


@pytest.fixture
def fake_pipeline(monkeypatch: pytest.MonkeyPatch) -> type[_FakePipeline]:
    _FakePipeline.calls = []
    monkeypatch.setattr(cli, "ForecastPipeline", _FakePipeline)
    return _FakePipeline


def _paths(tmp_path: Path) -> dict[str, object]:
    return {
        "root": tmp_path,
        "sources_config": "sources.yaml",
        "data_dir": tmp_path / "data",
        "artifacts_dir": tmp_path / "artifacts",
    }


def test_cli_helpers_and_pipeline_commands(
    tmp_path: Path, fake_pipeline: type[_FakePipeline]
) -> None:
    paths = _paths(tmp_path)
    assert cli._parse_cycles("2008, 2012,,") == [2008, 2012]
    with pytest.raises(typer.BadParameter):
        cli._parse_cycles(" , ")

    cli.sync(root=tmp_path, sources_config="sources.yaml", data_dir=tmp_path / "data")
    cli.build_features(root=tmp_path, sources_config="sources.yaml", data_dir=tmp_path / "data")
    cli.forecast_run(
        as_of="2026-05-08",
        run_id="forecast",
        scenario=None,
        inference_engine="bayes",
        bayesian_backend="analytic",
        quiet=False,
        **paths,
    )
    cli.forecast_run(
        as_of=None,
        run_id=None,
        scenario="scenario",
        inference_engine=None,
        bayesian_backend=None,
        quiet=True,
        **paths,
    )
    cli.forecast_update(from_anchor="anchor", as_of="2026-05-09", **paths)
    cli.backtest_run(
        run_id="backtest",
        scenario="scenario",
        start_cycle=2008,
        holdout_cycle=2024,
        inference_engine="kalman",
        bayesian_backend=None,
        **paths,
    )
    cli.backtest_nested(
        run_id="nested",
        scenario="scenario",
        start_cycle=2008,
        holdout_cycle=2024,
        inference_engine="kalman",
        bayesian_backend=None,
        **paths,
    )
    cli.backtest_refresh_hyperpriors(
        run_id="refresh",
        scenarios=" a, b, ",
        holdout_cycle=2024,
        inference_engine="bayes",
        bayesian_backend="analytic",
        **paths,
    )
    cli.backtest_refresh_hyperpriors(
        run_id=None,
        scenarios=None,
        holdout_cycle=None,
        inference_engine=None,
        bayesian_backend=None,
        **paths,
    )
    cli.spike_phase0(
        run_id="phase0",
        scenario="president_state",
        holdout_cycle=2024,
        bayesian_backend="analytic",
        **paths,
    )
    cli.spike_phase0b(run_id="phase0b", **paths)
    cli.report_build(run_id="forecast", **paths)
    cli.verify_readiness(
        run_id="ready",
        forecast_run_id="forecast",
        bayes_backtest_run_id="bayes",
        legacy_backtest_run_id="legacy",
        scenario="president_state",
        **paths,
    )
    cli.verify_historical_calibration(
        run_id="calibration",
        scenario="historical",
        as_of="2022-11-01",
        inference_engine="bayes",
        bayesian_backend="nuts",
        quiet=True,
        **paths,
    )
    cli.benchmark_run(as_of="2026-05-08", run_id="perf", draws=100, repeats=1, **paths)
    cli.results_compare(
        forecast_run_id="forecast",
        comparison_id="actuals",
        cycle=2024,
        office_type="president",
        race_id=None,
        **paths,
    )
    cli.results_cycle_eval(
        cycles="2020,2024",
        as_of_mm_dd="10-05",
        run_id="cycles",
        scenario_template="president_{cycle}_state",
        forecast_run_prefix="eval",
        comparison_id="actuals",
        office_type="president",
        reuse_existing=True,
        inference_engine="bayes",
        bayesian_backend="analytic",
        **paths,
    )

    names = {name for name, _ in fake_pipeline.calls}
    assert {
        "sync",
        "build_features",
        "run_forecast",
        "run_daily_update",
        "run_backtest",
        "run_nested_backtest",
        "refresh_hyperpriors",
        "run_phase0_spike",
        "run_phase0b_spike",
        "rebuild_report",
        "assess_methodology_readiness",
        "verify_historical_calibration",
        "run_benchmark",
        "compare_results",
        "run_cycle_eval",
    } <= names


def test_verify_run_routes_real_runner_branches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_pipeline: type[_FakePipeline]
) -> None:
    paths = _paths(tmp_path)

    class FakePhase8:
        def __init__(self, context: object) -> None:
            self.context = context

        def run(self, **kwargs: object) -> dict[str, object]:
            return {"passed": True, "output_dir": "/tmp/phase8"}

    import civic_signal.verification as verification

    monkeypatch.setattr(verification, "Phase8VerificationRunner", FakePhase8)
    cli.verify_run(
        run_id="phase8",
        scenario="scenario",
        as_of="2026-05-08",
        inference_engine="bayes",
        bayesian_backend="analytic",
        quiet=True,
        reproducibility_check=False,
        daily_update=False,
        **paths,
    )
    cli.verify_run(
        run_id="existing",
        scenario=None,
        as_of=None,
        inference_engine="bayes",
        bayesian_backend=None,
        quiet=False,
        reproducibility_check=True,
        daily_update=True,
        **paths,
    )
    with pytest.raises(typer.BadParameter):
        cli.verify_run(
            run_id=None,
            scenario=None,
            as_of=None,
            inference_engine="bayes",
            bayesian_backend=None,
            quiet=False,
            reproducibility_check=True,
            daily_update=True,
            **paths,
        )
    assert any(name == "verify_run" for name, _ in fake_pipeline.calls)


@pytest.mark.parametrize(
    ("function_name", "module_name", "class_name", "payload"),
    [
        (
            "verify_rewards",
            "civic_signal.verification.rewards",
            "RewardVerificationRunner",
            {
                "passed": False,
                "blocking_rewards": ["R18_nested_evaluation"],
                "reward_card_path": "/tmp/rewards.json",
                "exit_nonzero": True,
            },
        ),
        (
            "verify_publication",
            "civic_signal.verification.publication",
            "PublicationVerifier",
            {
                "passed": False,
                "publication_mode": "research",
                "failure_reasons": ["not promoted"],
            },
        ),
        (
            "verify_as_of",
            "civic_signal.verification.as_of",
            "AsOfVerificationRunner",
            {"passed": False, "audit_path": "/tmp/as-of.json", "exit_nonzero": True},
        ),
        (
            "data_audit",
            "civic_signal.verification.data_audit",
            "DataAuditRunner",
            {
                "passed": False,
                "audit": {"status": "insufficient_evidence"},
                "audit_path": "/tmp/data-audit.json",
                "exit_nonzero": True,
            },
        ),
        (
            "verify_coherence",
            "civic_signal.verification.coherence",
            "CoherenceVerificationRunner",
            {"passed": False, "output_dir": "/tmp/coherence"},
        ),
    ],
)
def test_audit_cli_commands_propagate_nonzero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    function_name: str,
    module_name: str,
    class_name: str,
    payload: dict[str, object],
) -> None:
    module = __import__(module_name, fromlist=[class_name])

    class FakeRunner:
        def __init__(self, context: object) -> None:
            self.context = context

        def verify(self, **kwargs: object) -> dict[str, object]:
            return payload

        def verify_semantic(self, **kwargs: object) -> dict[str, object]:
            return payload

    monkeypatch.setattr(module, class_name, FakeRunner)
    common = _paths(tmp_path)
    function = getattr(cli, function_name)
    with pytest.raises(typer.Exit):
        if function_name == "verify_rewards":
            function(
                run_id="run",
                profile="production",
                publication_mode=None,
                **common,
            )
        elif function_name == "verify_publication":
            function(run_id="run", profile="production", **common)
        elif function_name == "verify_as_of":
            function(
                run_id="run",
                scenario_family=None,
                cycles=None,
                offsets=None,
                as_of=None,
                **common,
            )
        elif function_name == "verify_coherence":
            function(run_id="run", audit_id=None, **common)
        else:
            function(run_id="audit", profile="production", as_of=None, **common)


def test_shadow_and_scientific_cli_routes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _paths(tmp_path)
    import civic_signal.shadow as shadow_module
    import civic_signal.verification.scientific as scientific_module
    import civic_signal.verification.shadow as shadow_verification_module

    class FakeShadowVerificationRunner:
        def __init__(self, context: object) -> None:
            self.context = context

        def verify(self, **kwargs: object) -> dict[str, object]:
            return {
                "passed": False,
                "consecutive_days": 0,
                "missing_runs": ["2026-01-01"],
                "violations": ["window incomplete"],
                "report_path": "/tmp/shadow-readiness.json",
                "exit_nonzero": True,
            }

    class FakeScientificVerificationRunner:
        def __init__(self, context: object) -> None:
            self.context = context

        def verify(self, **kwargs: object) -> dict[str, object]:
            return {
                "passed": False,
                "failures": ["live canary failed"],
                "report_path": "/tmp/scientific.json",
                "exit_nonzero": True,
            }

    class FakeShadowForecastRunner:
        def __init__(self, context: object, profile_id: str) -> None:
            self.context = context
            self.profile_id = profile_id

        def ensure_preregistration(self, **kwargs: object) -> dict[str, object]:
            return {"profile_id": self.profile_id, "frozen_at": "2026-01-01T00:00:00Z"}

        def run_window(self, **kwargs: object) -> dict[str, object]:
            return {
                "profile_id": self.profile_id,
                "scheduled_days": 2,
                "executed": ["one"],
                "profile_dir": "/tmp/shadow",
            }

    monkeypatch.setattr(
        shadow_verification_module, "ShadowVerificationRunner", FakeShadowVerificationRunner
    )
    monkeypatch.setattr(
        scientific_module, "ScientificVerificationRunner", FakeScientificVerificationRunner
    )
    monkeypatch.setattr(shadow_module, "ShadowForecastRunner", FakeShadowForecastRunner)

    with pytest.raises(typer.Exit):
        cli.verify_shadow(
            profile="shadow",
            window_start="2026-01-01",
            window_end="2026-01-02",
            **paths,
        )
    with pytest.raises(typer.Exit):
        cli.verify_scientific(live_canaries=False, nuts_smoke=False, **paths)
    cli.shadow_preregister(profile="shadow", cycle=2026, force=False, **paths)
    cli.shadow_run(
        profile="shadow",
        window_start="2026-01-01",
        window_end="2026-01-02",
        execute=False,
        scenario=None,
        inference_engine=None,
        bayesian_backend=None,
        quiet=True,
        **paths,
    )
