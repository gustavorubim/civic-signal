"""Integration tests for shadow and scientific verification orchestration."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from civic_signal.config import ProjectContext
from civic_signal.shadow.health import ShadowHealthMonitor
from civic_signal.shadow.runner import ShadowForecastRunner
from civic_signal.shadow.schedule import ShadowSchedule
from civic_signal.storage.io import read_json, write_json
from civic_signal.verification.scientific import ScientificVerificationRunner
from civic_signal.verification.shadow import ShadowVerificationRunner

ROOT = Path(__file__).resolve().parents[1]


def _context(tmp_path: Path) -> ProjectContext:
    return ProjectContext.create(
        root=ROOT,
        data_dir=tmp_path / "data",
        artifacts_dir=tmp_path / "artifacts",
    )


def _healthy_forecast_attempt(context: ProjectContext, run_id: str, as_of: str) -> Path:
    out_dir = context.artifacts_dir / "forecasts" / as_of / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "source_id": ["free-source"],
            "status": ["fetched"],
            "retrieved_at": [f"{as_of}T00:00:00+00:00"],
            "freshness_status": ["fresh"],
        }
    ).write_parquet(out_dir / "source_manifest.parquet")
    pl.DataFrame({"posterior_mean": [0.49, 0.51]}).write_parquet(
        out_dir / "poll_trajectory.parquet"
    )
    write_json({"max_mcse": 0.001}, out_dir / "performance.json")
    write_json({"divergences": 0}, out_dir / "posterior_diagnostics.json")
    write_json(
        {"run_id": run_id, "as_of": as_of, "inference_engine": "kalman"},
        out_dir / "run_manifest.json",
    )
    return out_dir


def test_shadow_window_executes_stamps_and_does_not_repeat_completed_days(
    tmp_path: Path, monkeypatch
) -> None:
    context = _context(tmp_path)

    def fake_forecast(self, *, as_of: str, run_id: str, **kwargs) -> Path:
        del self, kwargs
        return _healthy_forecast_attempt(context, run_id, str(as_of)[:10])

    monkeypatch.setattr("civic_signal.shadow.runner.ForecastPipeline.run_forecast", fake_forecast)
    runner = ShadowForecastRunner(context, profile_id="2026-general-shadow")
    preregistration = runner.ensure_preregistration(cycle=2026)
    assert runner.ensure_preregistration(cycle=2026) == preregistration

    first = runner.run_window(
        window_start="2026-06-01",
        window_end="2026-06-02",
        execute=True,
        inference_engine="kalman",
    )
    second = runner.run_window(
        window_start="2026-06-01",
        window_end="2026-06-02",
        execute=True,
        inference_engine="kalman",
    )

    assert len(first["executed"]) == 2
    assert second["executed"] == []
    history = runner.schedule.load_history()
    assert history["status"].to_list() == ["completed", "completed"]
    assert history["promotion_refused"].to_list() == [True, True]
    for result in first["executed"]:
        run_dir = Path(result["output_dir"])
        manifest = read_json(run_dir / "run_manifest.json")
        decision = read_json(run_dir / "publication_decision.json")
        health = read_json(run_dir / "source_health.json")
        assert result["status"] == "completed"
        assert manifest["publication_mode"] == "shadow"
        assert manifest["public_probabilities_published"] is False
        assert decision["allowed"] is False
        assert decision["blocks_publication"] is True
        assert health["passed"] is True
        assert health["monitors"]["poll_movement"]["metric"]["rows"] == 2

    readiness = ShadowVerificationRunner(context).verify(
        profile="2026-general-shadow",
        window_start="2026-06-01",
        window_end="2026-06-02",
    )
    # Real runner: short undeclared window cannot pass readiness even when complete.
    assert readiness["passed"] is False
    assert readiness["public_production_promotion"] is False
    assert readiness["promotion_refusals"] == 2


def test_shadow_single_day_and_health_failure_boundaries(tmp_path: Path, monkeypatch) -> None:
    context = _context(tmp_path)

    def unhealthy_forecast(self, *, as_of: str, run_id: str, **kwargs) -> Path:
        del self, kwargs
        out_dir = _healthy_forecast_attempt(context, run_id, str(as_of)[:10])
        write_json({"max_mcse": 0.02, "fallback_used": True}, out_dir / "performance.json")
        write_json(
            {"divergences": 2, "r_hat_max": 1.2},
            out_dir / "posterior_diagnostics.json",
        )
        write_json(
            {
                "needs_full_refit": True,
                "full_refit_executed": False,
                "posterior_drift": 0.1,
            },
            out_dir / "latest_daily_update.json",
        )
        write_json(
            {"run_id": run_id, "as_of": as_of, "inference_engine": "bayes"},
            out_dir / "run_manifest.json",
        )
        return out_dir

    monkeypatch.setattr(
        "civic_signal.shadow.runner.ForecastPipeline.run_forecast", unhealthy_forecast
    )
    runner = ShadowForecastRunner(context, profile_id="2026-general-shadow")
    result = runner.run_day(as_of="2026-06-03", run_id="unhealthy-shadow")

    assert result["status"] == "failed"
    assert result["health"]["flags"]["fallback_used"] is True
    assert result["health"]["flags"]["full_refit"] is True
    assert result["health"]["monitors"]["sampler_health"]["passed"] is False
    assert result["health"]["monitors"]["update_vs_refit"]["passed"] is False
    assert result["health"]["monitors"]["posterior_drift"]["passed"] is False

    missing = ShadowHealthMonitor().evaluate_run(tmp_path / "missing-run", as_of="not-a-date")
    assert missing["passed"] is False
    assert missing["source_health"]["source_count"] == 0


def test_shadow_health_parses_status_and_malformed_optional_evidence(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    pl.DataFrame(
        {
            "source_id": ["bad", "stale"],
            "status": ["failed", "fetched"],
            "retrieved_at": ["not-a-time", None],
            "freshness_status": ["fresh", "stale"],
        }
    ).write_parquet(run_dir / "source_manifest.parquet")
    (run_dir / "performance.json").write_text("[]", encoding="utf-8")
    pl.DataFrame({"unrelated": [1]}).write_parquet(run_dir / "poll_trajectory.parquet")

    report = ShadowHealthMonitor(max_source_age_hours=1).evaluate_run(run_dir, as_of="bad-date")

    assert report["passed"] is False
    assert report["source_health"]["failed_sources"] == ["bad"]
    assert report["source_health"]["stale_sources"] == ["stale"]
    assert report["monitors"]["poll_movement"]["metric"] == {
        "rows": 1,
        "movement_sd": None,
    }


def test_scientific_verifier_fails_closed_when_required_suites_report_failures(
    tmp_path: Path, monkeypatch
) -> None:
    context = _context(tmp_path)
    parity = {
        "python_numba_match": False,
        "serial_parallel_match": False,
        "python_fingerprint": "actual-python",
        "numba_fingerprint": "actual-numba",
        "numba_available": True,
    }
    golden = {
        "passed": False,
        "failures": ["broken golden"],
        "cases": {
            "parity_fingerprint": {
                "python_fingerprint": "expected-python",
                "numba_fingerprint": "expected-numba",
            }
        },
    }
    monkeypatch.setattr(
        "civic_signal.verification.scientific.numerical_parity_report", lambda: parity
    )
    monkeypatch.setattr(
        "civic_signal.verification.scientific.validate_golden_bundle", lambda root: golden
    )
    monkeypatch.setattr(
        "civic_signal.verification.scientific.run_standard_mutation_probes",
        lambda _root=None: {
            "all_mutations_detected": False,
            "actual_verifier_paths": False,
            "incomplete_mutation_families": ["publication_probability_range"],
        },
    )
    monkeypatch.setattr(
        "civic_signal.verification.scientific.ContractParityChecker.run",
        lambda self: {
            "passed": False,
            "stale_claim_details": ["stale documentation"],
            "stale_claims": 1,
            "checked_documents": [],
        },
    )
    monkeypatch.setattr(
        "civic_signal.verification.scientific.LiveSourceCanaryRunner.run",
        lambda self: {"all_passed": False, "history": []},
    )
    monkeypatch.setattr("civic_signal.scientific.nuts_smoke.run_nuts_smoke", lambda: {"ok": False})

    payload = ScientificVerificationRunner(context).verify(
        include_live_canaries=True,
        include_nuts_smoke=True,
        fixture_root=tmp_path / "fixtures",
    )

    assert payload["passed"] is False
    assert payload["checks_passed"] is False
    assert payload["evidence_state"] == "fail"
    assert payload["missing_required_suites"] == ["actual reward/publication source mutation suite"]
    assert payload["canaries"]["live"] is True
    assert payload["nuts_smoke"] == {"ok": False}
    assert len(payload["failures"]) >= 8
    persisted = json.loads(
        (context.artifacts_dir / "scientific" / "scientific_report.json").read_text(
            encoding="utf-8"
        )
    )
    assert persisted["failures"] == payload["failures"]


def test_shadow_verifier_counts_failed_and_unreconciled_attempts(tmp_path: Path) -> None:
    context = _context(tmp_path)
    profile = "ad-hoc-audit"
    schedule = ShadowSchedule(context.artifacts_dir / "shadow" / profile, profile)
    schedule.ensure_schedule(window_start="2026-07-01", window_end="2026-07-02")
    schedule.record_run(
        scheduled_date="2026-07-01",
        status="failed",
        forecast_run_dir="/failed",
        source_age_breach=True,
        silent_publication_failure=True,
        full_refit=True,
        promotion_refused=True,
        health_passed=False,
    )
    schedule.record_run(
        scheduled_date="2026-07-02",
        status="completed",
        forecast_run_dir="/completed-but-unhealthy",
        source_age_breach=True,
        full_refit=True,
        promotion_refused=True,
        public_probabilities_published=True,
        health_passed=False,
        extra={"source_correction_reconciled": False},
    )

    report = ShadowVerificationRunner(context).verify(
        profile=profile,
        window_start="2026-07-01",
        window_end="2026-07-02",
    )

    assert report["passed"] is False
    assert report["health_failures"] == 2
    assert report["silent_publication_failures"] == 2
    assert report["source_age_breaches"] == 1
    assert report["full_refits"] == 1
    assert report["promotion_refusals"] == 1
    assert report["corrected_inputs_unreconciled"] == 1
    assert report["completed_runs"] == 1
    assert report["missing_runs"] == ["2026-07-01"]
    assert report["profile_declared"] is False

    assert ShadowVerificationRunner._max_consecutive_days([]) == 0
    assert (
        ShadowVerificationRunner._max_consecutive_days(["2026-07-01", "2026-07-02", "2026-07-04"])
        == 2
    )
    human = ShadowVerificationRunner._human_report(
        {**report, "missing_runs": [f"2026-07-{day:02d}" for day in range(1, 32)]}
    )
    assert "1 more" in human


def test_scoring_lazy_exports_and_contract_parity_failure_details(tmp_path: Path) -> None:
    import civic_signal.scoring as scoring
    from civic_signal.scientific.contract_parity import ContractParityChecker

    for name in scoring.__all__:
        assert getattr(scoring, name) is not None
    try:
        unknown = scoring.not_a_scoring_export
    except AttributeError as exc:
        assert "not_a_scoring_export" in str(exc)
    else:
        raise AssertionError(f"unknown scoring export unexpectedly resolved: {unknown}")

    root = tmp_path / "contract"
    (root / "configs").mkdir(parents=True)
    (root / "src" / "civic_signal").mkdir(parents=True)
    (root / "configs" / "rewards.yaml").write_text("publication_mode_default: other\n")
    (root / "README.md").write_text("documentation without required contract words\n")
    (root / "SPEC.md").write_text("documentation without required contract words\n")
    (root / "src" / "civic_signal" / "cli.py").write_text("app = object()\n")

    result = ContractParityChecker(root).run()

    assert result["passed"] is False
    assert result["stale_claims"] >= 7
    assert (
        "rewards.yaml missing publication_mode_default: research" in result["stale_claim_details"]
    )
    assert "rewards.yaml missing R26_benchmark_superiority" in result["stale_claim_details"]
