"""M8 shadow forecasting: schedule, monitors, preregistration, readiness."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import polars as pl
import pytest
from typer.testing import CliRunner

from civic_signal.cli import app
from civic_signal.config import ProjectContext
from civic_signal.shadow.health import ShadowHealthMonitor
from civic_signal.shadow.preregistration import ShadowPreregistration
from civic_signal.shadow.runner import ShadowForecastRunner
from civic_signal.shadow.schedule import ShadowSchedule
from civic_signal.shadow.scorecard import ShadowScorecardBuilder
from civic_signal.storage.io import write_json
from civic_signal.verification.shadow import ShadowVerificationRunner

REPO_ROOT = Path(__file__).resolve().parents[1]


def _ctx(tmp_path: Path) -> ProjectContext:
    return ProjectContext.create(
        root=REPO_ROOT,
        data_dir=tmp_path / "data",
        artifacts_dir=tmp_path / "artifacts",
    )


def _complete_history(
    schedule: ShadowSchedule,
    *,
    window_start: str,
    window_end: str,
    fallback: bool = False,
    public_probs: bool = False,
    source_breach: bool = False,
) -> pl.DataFrame:
    history = schedule.ensure_schedule(window_start=window_start, window_end=window_end)
    for row in history.iter_rows(named=True):
        schedule.record_run(
            scheduled_date=str(row["scheduled_date"]),
            status="completed",
            forecast_run_dir=f"/tmp/shadow-run-{row['scheduled_date']}",
            fallback_used=fallback,
            source_age_breach=source_breach,
            silent_publication_failure=public_probs,
            public_probabilities_published=public_probs,
        )
    return schedule.load_history()


def test_schedule_date_range_and_history(tmp_path: Path) -> None:
    schedule = ShadowSchedule(tmp_path / "shadow" / "p", "p")
    days = schedule.date_range("2026-01-01", "2026-01-03")
    assert [d.isoformat() for d in days] == ["2026-01-01", "2026-01-02", "2026-01-03"]
    history = schedule.ensure_schedule(
        window_start="2026-01-01", window_end="2026-01-03", scenario="demo"
    )
    assert history.height == 3
    assert set(history["status"].to_list()) == {"scheduled"}
    schedule.record_run(
        scheduled_date="2026-01-01",
        status="completed",
        forecast_run_dir="/x",
    )
    again = schedule.ensure_schedule(window_start="2026-01-01", window_end="2026-01-04")
    assert again.height == 4
    jan1 = again.filter(pl.col("scheduled_date") == "2026-01-01")
    assert jan1["status"][0] == "completed"


def test_preregistration_freeze_is_create_once(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    pre = ShadowPreregistration(ctx, "demo-profile")
    first = pre.freeze(cycle=2026, offices=["senate"], horizon_buckets=[7, 1])
    assert first["frozen"] is True
    assert pre.is_frozen()
    with pytest.raises(FileExistsError):
        pre.freeze(cycle=2026, offices=["senate"], horizon_buckets=[7])
    second = pre.freeze(cycle=2026, offices=["house"], horizon_buckets=[14], force=True)
    assert second["offices"] == ["house"]


def test_health_monitor_flags_stale_and_fallback(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    old = (date.today() - timedelta(days=10)).isoformat() + "T00:00:00+00:00"
    pl.DataFrame(
        {
            "source_id": ["a", "b"],
            "status": ["fetched", "failed"],
            "retrieved_at": [old, old],
        }
    ).write_parquet(run / "source_manifest.parquet")
    write_json({"fallback_used": "analytic", "divergences": 0}, run / "posterior_diagnostics.json")
    write_json({"max_mcse": 0.01}, run / "performance.json")
    write_json(
        {"publication_mode": "production", "allowed": True},
        run / "publication_decision.json",
    )
    report = ShadowHealthMonitor(max_source_age_hours=24).evaluate_run(run)
    assert report["flags"]["fallback_used"] is True
    assert report["flags"]["source_age_breach"] is True
    assert report["flags"]["public_probabilities_published"] is True
    assert report["monitors"]["mcse"]["passed"] is False
    assert (run / "source_health.json").exists()


def test_scorecard_is_observational_without_nested_scores(tmp_path: Path) -> None:
    schedule = ShadowSchedule(tmp_path / "shadow" / "p", "p")
    history = _complete_history(schedule, window_start="2026-01-01", window_end="2026-01-05")
    prereg = {
        "frozen": True,
        "offices": ["senate", "house"],
        "horizon_buckets": [7, 1],
        "comparators": [],
    }
    card = ShadowScorecardBuilder(tmp_path / "shadow" / "p").build(
        history=history, preregistration=prereg
    )
    assert card["completed_shadow_runs"] == 5
    assert card["claim_status"] == "insufficient_evidence"
    assert card["best_claim_status"] == "insufficient_evidence"
    assert card["public_production_promotion"] is False
    assert len(card["office_horizon_rows"]) == 4


def test_verify_shadow_requires_complete_window_or_60_days(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    profile = "unit-shadow"
    # Tiny profile config override via writing shadow.yaml is heavy; use defaults.
    schedule = ShadowSchedule(ctx.artifacts_dir / "shadow" / profile, profile)
    # Incomplete 3-day window
    schedule.ensure_schedule(window_start="2026-02-01", window_end="2026-02-03")
    schedule.record_run(
        scheduled_date="2026-02-01",
        status="completed",
        forecast_run_dir="/r1",
    )
    # missing 02 and 03
    ShadowPreregistration(ctx, profile).freeze(cycle=2026, offices=["senate"], horizon_buckets=[7])
    result = ShadowVerificationRunner(ctx).verify(
        profile=profile,
        window_start="2026-02-01",
        window_end="2026-02-03",
    )
    assert result["passed"] is False
    assert result["missing_runs"]
    assert result["exit_nonzero"] is True

    # Completing an undeclared short window is still insufficient evidence.
    for day in ["2026-02-02", "2026-02-03"]:
        schedule.record_run(scheduled_date=day, status="completed", forecast_run_dir=f"/r-{day}")
    ShadowScorecardBuilder(ctx.artifacts_dir / "shadow" / profile).build(
        history=schedule.load_history(),
        preregistration=ShadowPreregistration(ctx, profile).load(),
    )
    short = ShadowVerificationRunner(ctx).verify(
        profile=profile,
        window_start="2026-02-01",
        window_end="2026-02-03",
    )
    assert short["passed"] is False
    assert short["consecutive_days"] == 3
    assert short["profile_declared"] is False
    assert (ctx.artifacts_dir / "shadow" / profile / "shadow_readiness.json").exists()


def test_verify_shadow_fails_on_fallback_and_public_probs(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    profile = "bad-shadow"
    schedule = ShadowSchedule(ctx.artifacts_dir / "shadow" / profile, profile)
    _complete_history(
        schedule,
        window_start="2026-03-01",
        window_end="2026-03-02",
        fallback=True,
        public_probs=True,
    )
    ShadowPreregistration(ctx, profile).freeze(cycle=2026, offices=["senate"], horizon_buckets=[1])
    result = ShadowVerificationRunner(ctx).verify(
        profile=profile,
        window_start="2026-03-01",
        window_end="2026-03-02",
    )
    assert result["passed"] is False
    assert result["silent_publication_failures"] >= 1
    assert result["fallback_days"] >= 1


def test_verify_shadow_60_day_consecutive_gate(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    profile = "2026-general-shadow"
    schedule = ShadowSchedule(ctx.artifacts_dir / "shadow" / profile, profile)
    start = date(2026, 4, 1)
    end = start + timedelta(days=59)
    history = schedule.ensure_schedule(window_start=start.isoformat(), window_end=end.isoformat())
    assert history.height == 60
    for row in history.iter_rows(named=True):
        schedule.record_run(
            scheduled_date=str(row["scheduled_date"]),
            status="completed",
            forecast_run_dir="/ok",
        )
    ShadowPreregistration(ctx, profile).freeze(
        cycle=2026, offices=["senate", "house"], horizon_buckets=[30, 7]
    )
    result = ShadowVerificationRunner(ctx).verify(
        profile=profile,
        window_start=start.isoformat(),
        window_end=end.isoformat(),
    )
    assert result["passed"] is True
    assert result["consecutive_days"] == 60
    assert result["scheduled_runs"] == 60


def test_shadow_cli_preregister_and_verify(tmp_path: Path) -> None:
    runner = CliRunner()
    artifacts = tmp_path / "artifacts"
    # Preregister via CLI
    result = runner.invoke(
        app,
        [
            "shadow",
            "preregister",
            "--profile",
            "cli-shadow",
            "--cycle",
            "2026",
            "--root",
            str(REPO_ROOT),
            "--artifacts-dir",
            str(artifacts),
        ],
    )
    # Profile cli-shadow not in shadow.yaml - runner raises KeyError
    # Use configured profile instead.
    result = runner.invoke(
        app,
        [
            "shadow",
            "preregister",
            "--profile",
            "2026-general-shadow",
            "--cycle",
            "2026",
            "--root",
            str(REPO_ROOT),
            "--artifacts-dir",
            str(artifacts),
        ],
    )
    assert result.exit_code == 0, result.output
    assert (artifacts / "shadow" / "2026-general-shadow" / "preregistration.json").exists()

    schedule = ShadowSchedule(artifacts / "shadow" / "2026-general-shadow", "2026-general-shadow")
    _complete_history(schedule, window_start="2026-05-01", window_end="2026-05-03")
    verify = runner.invoke(
        app,
        [
            "verify",
            "shadow",
            "--profile",
            "2026-general-shadow",
            "--window-start",
            "2026-05-01",
            "--window-end",
            "2026-05-03",
            "--root",
            str(REPO_ROOT),
            "--artifacts-dir",
            str(artifacts),
        ],
    )
    assert verify.exit_code != 0
    assert "passed=False" in verify.output


def test_shadow_runner_schedule_only(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    payload = ShadowForecastRunner(ctx, profile_id="2026-general-shadow").run_window(
        window_start="2026-06-01",
        window_end="2026-06-02",
        execute=False,
    )
    assert payload["scheduled_days"] == 2
    assert payload["executed"] == []
    history = pl.read_parquet(payload["history_path"])
    assert history.height == 2
    assert set(history["status"].to_list()) == {"scheduled"}
    # Scorecard still refuses best/public-production claims for schedule-only windows.
    from civic_signal.storage.io import read_json

    scorecard = read_json(Path(payload["scorecard_path"]))
    assert scorecard["claim_status"] == "insufficient_evidence"
    assert scorecard["public_production_promotion"] is False
