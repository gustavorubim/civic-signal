from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from civic_signal.config import ProjectContext
from civic_signal.pipeline import ForecastPipeline

app = typer.Typer(help="U.S. election forecasting engine.")
forecast_app = typer.Typer(help="Forecast commands.")
backtest_app = typer.Typer(help="Backtest commands.")
report_app = typer.Typer(help="Report commands.")
benchmark_app = typer.Typer(help="Performance benchmark commands.")
results_app = typer.Typer(help="Forecast-vs-actual result comparison commands.")
verify_app = typer.Typer(help="Run artifact verification commands.")
data_app = typer.Typer(help="Audit production data and source inputs.")
shadow_app = typer.Typer(help="Shadow forecasting (non-public pre-production runs).")
spike_app = typer.Typer(help="Methodology spike commands.")
app.add_typer(forecast_app, name="forecast")
app.add_typer(backtest_app, name="backtest")
app.add_typer(report_app, name="report")
app.add_typer(benchmark_app, name="benchmark")
app.add_typer(results_app, name="results")
app.add_typer(verify_app, name="verify")
app.add_typer(data_app, name="data")
app.add_typer(shadow_app, name="shadow")
app.add_typer(spike_app, name="spike")
console = Console()


def _context(
    root: Path | None = None,
    sources_config: str = "sources.yaml",
    data_dir: Path | None = None,
    artifacts_dir: Path | None = None,
) -> ProjectContext:
    return ProjectContext.create(
        root=root,
        sources_config=sources_config,
        data_dir=data_dir,
        artifacts_dir=artifacts_dir,
    )


def _parse_cycles(cycles: str) -> list[int]:
    values = [part.strip() for part in cycles.split(",") if part.strip()]
    if not values:
        raise typer.BadParameter("Provide at least one cycle")
    return [int(value) for value in values]


@app.command()
def sync(
    root: Path | None = typer.Option(None, help="Project root."),
    sources_config: str = typer.Option("sources.yaml", help="Source registry config file."),
    data_dir: Path | None = typer.Option(None, help="Data directory override."),
) -> None:
    """Snapshot configured sources into the raw lake."""
    context = _context(root=root, sources_config=sources_config, data_dir=data_dir)
    manifest = ForecastPipeline(context).sync()
    console.print(f"[green]Synced[/green] {manifest.height} sources into {context.raw_dir}")


@app.command("build-features")
def build_features(
    root: Path | None = typer.Option(None, help="Project root."),
    sources_config: str = typer.Option("sources.yaml", help="Source registry config file."),
    data_dir: Path | None = typer.Option(None, help="Data directory override."),
) -> None:
    """Build curated tables and race tiering."""
    context = _context(root=root, sources_config=sources_config, data_dir=data_dir)
    bundle = ForecastPipeline(context).build_features()
    console.print(f"[green]Built[/green] {bundle.race_catalog.height} race catalog rows")


@forecast_app.command("run")
def forecast_run(
    as_of: str | None = typer.Option(None, help="Forecast date in YYYY-MM-DD form."),
    run_id: str | None = typer.Option(None, help="Stable run id."),
    scenario: str | None = typer.Option(None, help="Scenario key from configs/scenarios.yaml."),
    inference_engine: str | None = typer.Option(
        None, help="Polling inference engine: kalman or bayes. Defaults to configs/model.yaml."
    ),
    bayesian_backend: str | None = typer.Option(
        None, help="Bayesian backend when --inference-engine bayes: analytic or nuts."
    ),
    quiet: bool = typer.Option(False, help="Suppress the completion message."),
    root: Path | None = typer.Option(None, help="Project root."),
    sources_config: str = typer.Option("sources.yaml", help="Source registry config file."),
    data_dir: Path | None = typer.Option(None, help="Data directory override."),
    artifacts_dir: Path | None = typer.Option(None, help="Artifacts directory override."),
) -> None:
    """Refresh data and emit the full forecast artifact contract."""
    context = _context(
        root=root,
        sources_config=sources_config,
        data_dir=data_dir,
        artifacts_dir=artifacts_dir,
    )
    out_dir = ForecastPipeline(context).run_forecast(
        as_of=as_of,
        run_id=run_id,
        scenario=scenario,
        inference_engine=inference_engine,
        bayesian_backend=bayesian_backend,
        quiet=quiet,
    )
    if not quiet:
        console.print(f"[green]Forecast complete[/green]: {out_dir}")


@forecast_app.command("update")
def forecast_update(
    from_anchor: str = typer.Option(..., help="Bayesian anchor forecast run id."),
    as_of: str = typer.Option(..., help="Daily update date in YYYY-MM-DD form."),
    root: Path | None = typer.Option(None, help="Project root."),
    sources_config: str = typer.Option("sources.yaml", help="Source registry config file."),
    data_dir: Path | None = typer.Option(None, help="Data directory override."),
    artifacts_dir: Path | None = typer.Option(None, help="Artifacts directory override."),
) -> None:
    """Run a cached-posterior daily update from a Bayesian anchor run."""
    context = _context(
        root=root,
        sources_config=sources_config,
        data_dir=data_dir,
        artifacts_dir=artifacts_dir,
    )
    payload = ForecastPipeline(context).run_daily_update(anchor_run_id=from_anchor, as_of=as_of)
    console.print(
        "[green]Daily update complete[/green]: "
        f"{payload['strategy']}, refit={payload['needs_full_refit']}"
    )
    console.print(payload["output_dir"])


@backtest_app.command("run")
def backtest_run(
    run_id: str | None = typer.Option(None, help="Stable backtest run id."),
    scenario: str | None = typer.Option(None, help="Scenario key from configs/scenarios.yaml."),
    start_cycle: int | None = typer.Option(None, help="First holdout cycle to score."),
    holdout_cycle: int | None = typer.Option(None, help="Single holdout cycle to score."),
    inference_engine: str | None = typer.Option(
        None,
        help=(
            "Polling inference engine for rolling-origin folds: kalman or bayes. "
            "Defaults to configs/model.yaml."
        ),
    ),
    bayesian_backend: str | None = typer.Option(
        None, help="Bayesian backend for rolling-origin folds: analytic or nuts."
    ),
    root: Path | None = typer.Option(None, help="Project root."),
    sources_config: str = typer.Option("sources.yaml", help="Source registry config file."),
    data_dir: Path | None = typer.Option(None, help="Data directory override."),
    artifacts_dir: Path | None = typer.Option(None, help="Artifacts directory override."),
) -> None:
    """Run historical backtest scorecards and ablations."""
    context = _context(
        root=root,
        sources_config=sources_config,
        data_dir=data_dir,
        artifacts_dir=artifacts_dir,
    )
    payload = ForecastPipeline(context).run_backtest(
        run_id=run_id,
        scenario=scenario,
        start_cycle=start_cycle,
        holdout_cycle=holdout_cycle,
        inference_engine=inference_engine,
        bayesian_backend=bayesian_backend,
    )
    console.print(f"[green]Backtest complete[/green]: {payload['row_count']} rows")


@backtest_app.command("nested")
def backtest_nested(
    run_id: str | None = typer.Option(None, help="Stable nested-backtest run id."),
    scenario: str | None = typer.Option(None, help="Scenario key from configs/scenarios.yaml."),
    start_cycle: int | None = typer.Option(None, help="First outer holdout cycle to score."),
    holdout_cycle: int | None = typer.Option(None, help="Single outer holdout cycle to score."),
    inference_engine: str | None = typer.Option(
        None, help="Polling engine for inner and outer folds: kalman or bayes."
    ),
    bayesian_backend: str | None = typer.Option(
        None, help="Bayesian backend for nested folds: analytic or nuts."
    ),
    root: Path | None = typer.Option(None, help="Project root."),
    sources_config: str = typer.Option("sources.yaml", help="Source registry config file."),
    data_dir: Path | None = typer.Option(None, help="Data directory override."),
    artifacts_dir: Path | None = typer.Option(None, help="Artifacts directory override."),
) -> None:
    """Run outer-cycle nested evaluation through the publication simulation path."""
    context = _context(
        root=root,
        sources_config=sources_config,
        data_dir=data_dir,
        artifacts_dir=artifacts_dir,
    )
    payload = ForecastPipeline(context).run_nested_backtest(
        run_id=run_id,
        scenario=scenario,
        start_cycle=start_cycle,
        holdout_cycle=holdout_cycle,
        inference_engine=inference_engine,
        bayesian_backend=bayesian_backend,
    )
    console.print(
        f"[green]Nested backtest complete[/green]: {payload['fold_count']} outer folds, "
        f"{payload['row_count']} rows"
    )


@backtest_app.command("refresh-hyperpriors")
def backtest_refresh_hyperpriors(
    run_id: str | None = typer.Option(None, help="Stable refresh run id."),
    scenarios: str | None = typer.Option(
        None, help="Comma-separated scenario keys. Defaults to model.yaml hyperprior_refresh."
    ),
    holdout_cycle: int | None = typer.Option(None, help="Optional single holdout cycle."),
    inference_engine: str | None = typer.Option(
        None, help="Polling inference engine. Defaults to model.yaml hyperprior_refresh."
    ),
    bayesian_backend: str | None = typer.Option(
        None, help="Bayesian backend for refresh candidates: analytic or nuts."
    ),
    root: Path | None = typer.Option(None, help="Project root."),
    sources_config: str = typer.Option("sources.yaml", help="Source registry config file."),
    data_dir: Path | None = typer.Option(None, help="Data directory override."),
    artifacts_dir: Path | None = typer.Option(None, help="Artifacts directory override."),
) -> None:
    """Write scheduled hyperprior refresh candidates without promoting latest artifacts."""
    context = _context(
        root=root,
        sources_config=sources_config,
        data_dir=data_dir,
        artifacts_dir=artifacts_dir,
    )
    scenario_list = (
        [part.strip() for part in scenarios.split(",") if part.strip()] if scenarios else None
    )
    payload = ForecastPipeline(context).refresh_hyperpriors(
        run_id=run_id,
        scenarios=scenario_list,
        inference_engine=inference_engine,
        holdout_cycle=holdout_cycle,
        bayesian_backend=bayesian_backend,
    )
    console.print(
        "[green]Hyperprior refresh candidates written[/green]: "
        f"{len(payload['scenarios'])} scenarios, promoted={payload['promoted']}"
    )
    console.print(payload["output_dir"])


@spike_app.command("phase-0")
def spike_phase0(
    run_id: str | None = typer.Option(None, help="Stable spike run id."),
    scenario: str = typer.Option("president_state", help="Backtest scenario family."),
    holdout_cycle: int = typer.Option(2024, help="Holdout cycle to compare."),
    bayesian_backend: str | None = typer.Option(
        None, help="Bayesian backend for the Bayes comparison leg: analytic or nuts."
    ),
    root: Path | None = typer.Option(None, help="Project root."),
    sources_config: str = typer.Option("sources.yaml", help="Source registry config file."),
    data_dir: Path | None = typer.Option(None, help="Data directory override."),
    artifacts_dir: Path | None = typer.Option(None, help="Artifacts directory override."),
) -> None:
    """Run the Phase 0 Kalman-vs-Bayes rolling-origin comparison harness."""
    context = _context(
        root=root,
        sources_config=sources_config,
        data_dir=data_dir,
        artifacts_dir=artifacts_dir,
    )
    payload = ForecastPipeline(context).run_phase0_spike(
        run_id=run_id,
        scenario=scenario,
        holdout_cycle=holdout_cycle,
        bayesian_backend=bayesian_backend,
    )
    console.print(
        "[green]Phase 0 spike complete[/green]: "
        f"{payload['go_no_go']['status']}, "
        f"bayes_minus_kalman={payload['go_no_go']['bayes_minus_kalman']}"
    )
    console.print(payload["output_dir"])


@spike_app.command("phase-0b")
def spike_phase0b(
    run_id: str | None = typer.Option(None, help="Stable spike run id."),
    root: Path | None = typer.Option(None, help="Project root."),
    sources_config: str = typer.Option("sources.yaml", help="Source registry config file."),
    data_dir: Path | None = typer.Option(None, help="Data directory override."),
    artifacts_dir: Path | None = typer.Option(None, help="Artifacts directory override."),
) -> None:
    """Run the Phase 0b geometry and daily-update acceleration bakeoff."""
    context = _context(
        root=root,
        sources_config=sources_config,
        data_dir=data_dir,
        artifacts_dir=artifacts_dir,
    )
    payload = ForecastPipeline(context).run_phase0b_spike(run_id=run_id)
    console.print(
        "[green]Phase 0b spike complete[/green]: "
        f"selected={payload['selected_strategy']}, "
        f"global_smc_rejected={payload['global_smc_rejected']}"
    )
    console.print(payload["output_dir"])


@report_app.command("build")
def report_build(
    run_id: str = typer.Option(..., help="Forecast run id to rebuild."),
    root: Path | None = typer.Option(None, help="Project root."),
    sources_config: str = typer.Option("sources.yaml", help="Source registry config file."),
    data_dir: Path | None = typer.Option(None, help="Data directory override."),
    artifacts_dir: Path | None = typer.Option(None, help="Artifacts directory override."),
) -> None:
    """Rebuild diagnostics and methodology files for an existing forecast run."""
    context = _context(
        root=root,
        sources_config=sources_config,
        data_dir=data_dir,
        artifacts_dir=artifacts_dir,
    )
    out_dir = ForecastPipeline(context).rebuild_report(run_id=run_id)
    console.print(f"[green]Report rebuilt[/green]: {out_dir}")


@verify_app.command("run")
def verify_run(
    run_id: str | None = typer.Option(None, help="Forecast run id to verify."),
    scenario: str | None = typer.Option(
        None,
        help="Scenario key to run before verification, e.g. 2026-multioffice-verification.",
    ),
    as_of: str | None = typer.Option(None, help="Forecast date in YYYY-MM-DD form."),
    inference_engine: str = typer.Option(
        "bayes", help="Polling inference engine for scenario verification: kalman or bayes."
    ),
    bayesian_backend: str | None = typer.Option(
        None,
        help="Bayesian backend for scenario verification: analytic or nuts.",
    ),
    quiet: bool = typer.Option(
        False, help="Suppress forecast progress during scenario verification."
    ),
    reproducibility_check: bool = typer.Option(
        True,
        "--reproducibility-check/--no-reproducibility-check",
        help="Run the same scenario/run id twice before verification.",
    ),
    daily_update: bool = typer.Option(
        True,
        "--daily-update/--no-daily-update",
        help="Run the Bayesian daily-update gate after the scenario forecast.",
    ),
    root: Path | None = typer.Option(None, help="Project root."),
    sources_config: str = typer.Option("sources.yaml", help="Source registry config file."),
    data_dir: Path | None = typer.Option(None, help="Data directory override."),
    artifacts_dir: Path | None = typer.Option(None, help="Artifacts directory override."),
) -> None:
    """Verify an existing run, or orchestrate a Phase 8 scenario verification."""
    context = _context(
        root=root,
        sources_config=sources_config,
        data_dir=data_dir,
        artifacts_dir=artifacts_dir,
    )
    if scenario:
        from civic_signal.verification import Phase8VerificationRunner

        payload = Phase8VerificationRunner(context).run(
            run_id=run_id,
            scenario=scenario,
            as_of=as_of,
            inference_engine=inference_engine,
            bayesian_backend=bayesian_backend,
            quiet=quiet,
            reproducibility_check=reproducibility_check,
            daily_update=daily_update,
        )
        color = "green" if payload["passed"] else "yellow"
        console.print(f"[{color}]Phase 8 verification complete[/{color}]: {payload['passed']}")
        console.print(payload["output_dir"])
        return
    if run_id is None:
        raise typer.BadParameter("run_id is required unless --scenario is provided")
    payload = ForecastPipeline(context).verify_run(run_id=run_id)
    color = "green" if payload["passed"] else "yellow"
    console.print(f"[{color}]Verification complete[/{color}]: {payload['passed']}")
    if payload.get("output_dir"):
        console.print(payload["output_dir"])


@verify_app.command("readiness")
def verify_readiness(
    run_id: str | None = typer.Option(None, help="Stable readiness audit run id."),
    forecast_run_id: str | None = typer.Option(
        None, help="Forecast run id containing Phase 8 and reward artifacts."
    ),
    bayes_backtest_run_id: str | None = typer.Option(
        None, help="Bayesian backtest run id for rolling-origin evidence."
    ),
    legacy_backtest_run_id: str | None = typer.Option(
        None, help="Legacy Kalman backtest run id for rolling-origin comparison."
    ),
    scenario: str = typer.Option("president_state", help="Scenario family being assessed."),
    root: Path | None = typer.Option(None, help="Project root."),
    sources_config: str = typer.Option("sources.yaml", help="Source registry config file."),
    data_dir: Path | None = typer.Option(None, help="Data directory override."),
    artifacts_dir: Path | None = typer.Option(None, help="Artifacts directory override."),
) -> None:
    """Audit whether Bayes is eligible to become the production default."""
    context = _context(
        root=root,
        sources_config=sources_config,
        data_dir=data_dir,
        artifacts_dir=artifacts_dir,
    )
    payload = ForecastPipeline(context).assess_methodology_readiness(
        run_id=run_id,
        forecast_run_id=forecast_run_id,
        bayes_backtest_run_id=bayes_backtest_run_id,
        legacy_backtest_run_id=legacy_backtest_run_id,
        scenario=scenario,
    )
    color = "green" if payload["eligible_for_default_switch"] else "yellow"
    console.print(f"[{color}]Methodology readiness[/{color}]: {payload['status']}")
    console.print(payload["output_dir"])


@verify_app.command("rewards")
def verify_rewards(
    run_id: str = typer.Option(..., help="Forecast or attempt run id to recompute rewards for."),
    profile: str = typer.Option(
        "production",
        help="Reward profile: fixture, research, shadow, or production.",
    ),
    publication_mode: str | None = typer.Option(
        None, help="Override publication_mode label for recomputation."
    ),
    root: Path | None = typer.Option(None, help="Project root."),
    sources_config: str = typer.Option("sources.yaml", help="Source registry config file."),
    data_dir: Path | None = typer.Option(None, help="Data directory override."),
    artifacts_dir: Path | None = typer.Option(None, help="Artifacts directory override."),
) -> None:
    """Recompute reward-v2 from primary artifacts for a profile (never trusts stored booleans)."""
    from civic_signal.verification.rewards import RewardVerificationRunner

    context = _context(
        root=root,
        sources_config=sources_config,
        data_dir=data_dir,
        artifacts_dir=artifacts_dir,
    )
    payload = RewardVerificationRunner(context).verify(
        run_id=run_id,
        profile=profile,
        publication_mode=publication_mode,
    )
    color = "green" if payload["passed"] else "red"
    console.print(
        f"[{color}]Reward verification[/{color}]: profile={profile} "
        f"passed={payload['passed']} blocking={payload['blocking_rewards']}"
    )
    console.print(payload["reward_card_path"])
    if payload.get("exit_nonzero"):
        raise typer.Exit(code=1)


@verify_app.command("publication")
def verify_publication(
    run_id: str = typer.Option(..., help="Attempt or forecast run id."),
    profile: str = typer.Option("production", help="Promotion profile name."),
    root: Path | None = typer.Option(None, help="Project root."),
    sources_config: str = typer.Option("sources.yaml", help="Source registry config file."),
    data_dir: Path | None = typer.Option(None, help="Data directory override."),
    artifacts_dir: Path | None = typer.Option(None, help="Artifacts directory override."),
) -> None:
    """Semantic publication verification; production requires verified promotion manifest."""
    from civic_signal.verification.publication import PublicationVerifier

    context = _context(
        root=root,
        sources_config=sources_config,
        data_dir=data_dir,
        artifacts_dir=artifacts_dir,
    )
    payload = PublicationVerifier(context).verify_semantic(run_id=run_id, profile=profile)
    color = "green" if payload["passed"] else "red"
    console.print(
        f"[{color}]Publication verification[/{color}]: passed={payload['passed']} "
        f"mode={payload['publication_mode']}"
    )
    if payload.get("failure_reasons"):
        console.print(payload["failure_reasons"])
    if not payload["passed"]:
        raise typer.Exit(code=1)


@verify_app.command("as-of")
def verify_as_of(
    run_id: str = typer.Option(..., help="As-of audit run id."),
    scenario_family: str | None = typer.Option(None, help="Scenario family under audit."),
    cycles: str | None = typer.Option(None, help="Cycle range, e.g. 2004:2024."),
    offsets: str | None = typer.Option(None, help="Comma-separated horizon offsets."),
    as_of: str | None = typer.Option(None, help="As-of date YYYY-MM-DD."),
    root: Path | None = typer.Option(None, help="Project root."),
    sources_config: str = typer.Option("sources.yaml", help="Source registry config file."),
    data_dir: Path | None = typer.Option(None, help="Data directory override."),
    artifacts_dir: Path | None = typer.Option(None, help="Artifacts directory override."),
) -> None:
    """Verify as-of integrity; missing historical archive yields insufficient evidence."""
    from civic_signal.verification.as_of import AsOfVerificationRunner

    context = _context(
        root=root,
        sources_config=sources_config,
        data_dir=data_dir,
        artifacts_dir=artifacts_dir,
    )
    payload = AsOfVerificationRunner(context).verify(
        run_id=run_id,
        scenario_family=scenario_family,
        cycles=cycles,
        offsets=offsets,
        as_of=as_of,
    )
    color = "green" if payload["passed"] else "yellow"
    console.print(f"[{color}]As-of verification[/{color}]: passed={payload['passed']}")
    console.print(payload["audit_path"])
    if payload.get("exit_nonzero"):
        raise typer.Exit(code=1)


@verify_app.command("coherence")
def verify_coherence(
    run_id: str = typer.Option(..., help="Forecast run id to audit."),
    audit_id: str | None = typer.Option(None, help="Optional coherence audit id."),
    root: Path | None = typer.Option(None, help="Project root."),
    sources_config: str = typer.Option("sources.yaml", help="Source registry config file."),
    data_dir: Path | None = typer.Option(None, help="Data directory override."),
    artifacts_dir: Path | None = typer.Option(None, help="Artifacts directory override."),
) -> None:
    """Reconstruct race, draw, elector, and chamber-control coherence."""
    from civic_signal.verification.coherence import CoherenceVerificationRunner

    context = _context(
        root=root,
        sources_config=sources_config,
        data_dir=data_dir,
        artifacts_dir=artifacts_dir,
    )
    payload = CoherenceVerificationRunner(context).verify(run_id=run_id, audit_id=audit_id)
    color = "green" if payload["passed"] else "red"
    console.print(f"[{color}]Coherence verification[/{color}]: passed={payload['passed']}")
    console.print(payload["output_dir"])
    if not payload["passed"]:
        raise typer.Exit(code=1)


@verify_app.command("recovery")
def verify_recovery(
    run_id: str | None = typer.Option(None, help="Stable recovery audit id."),
    backend: str = typer.Option(
        "analytic", help="Bayesian polling backend for the bounded smoke: analytic or nuts."
    ),
    replicates: int = typer.Option(12, min=2, help="Synthetic recovery replicates."),
    root: Path | None = typer.Option(None, help="Project root."),
    sources_config: str = typer.Option("sources.yaml", help="Source registry config file."),
    data_dir: Path | None = typer.Option(None, help="Data directory override."),
    artifacts_dir: Path | None = typer.Option(None, help="Artifacts directory override."),
) -> None:
    """Run bounded Bayesian recovery checks without claiming production sufficiency."""
    from civic_signal.verification.recovery import RecoveryVerificationRunner

    context = _context(
        root=root,
        sources_config=sources_config,
        data_dir=data_dir,
        artifacts_dir=artifacts_dir,
    )
    payload = RecoveryVerificationRunner(context).verify(
        run_id=run_id,
        backend=backend,
        replicates=replicates,
    )
    color = "green" if payload["smoke_checks_passed"] else "red"
    console.print(
        f"[{color}]Recovery smoke[/{color}]: checks={payload['smoke_checks_passed']} "
        f"evidence={payload['status']}"
    )
    console.print(payload["output_dir"])
    if not payload["smoke_checks_passed"]:
        raise typer.Exit(code=1)


@verify_app.command("shadow")
def verify_shadow(
    profile: str = typer.Option(
        "2026-general-shadow",
        help="Shadow profile id from configs/shadow.yaml.",
    ),
    window_start: str = typer.Option(..., help="Inclusive window start YYYY-MM-DD."),
    window_end: str = typer.Option(..., help="Inclusive window end YYYY-MM-DD."),
    root: Path | None = typer.Option(None, help="Project root."),
    sources_config: str = typer.Option("sources.yaml", help="Source registry config file."),
    data_dir: Path | None = typer.Option(None, help="Data directory override."),
    artifacts_dir: Path | None = typer.Option(None, help="Artifacts directory override."),
) -> None:
    """Verify a shadow window; incomplete history cannot pass."""
    from civic_signal.verification.shadow import ShadowVerificationRunner

    context = _context(
        root=root,
        sources_config=sources_config,
        data_dir=data_dir,
        artifacts_dir=artifacts_dir,
    )
    payload = ShadowVerificationRunner(context).verify(
        profile=profile,
        window_start=window_start,
        window_end=window_end,
    )
    color = "green" if payload["passed"] else "red"
    console.print(
        f"[{color}]Shadow verification[/{color}]: passed={payload['passed']} "
        f"consecutive={payload['consecutive_days']} missing={len(payload['missing_runs'])}"
    )
    console.print(payload["report_path"])
    if payload.get("violations"):
        console.print(payload["violations"])
    if payload.get("exit_nonzero"):
        raise typer.Exit(code=1)


@verify_app.command("scientific")
def verify_scientific(
    live_canaries: bool = typer.Option(
        False,
        "--live-canaries",
        help="Hit real free-public HTTP canaries (offline mocks used by default).",
    ),
    nuts_smoke: bool = typer.Option(
        False,
        "--nuts-smoke",
        help="Run a tiny real NumPyro/NUTS fit (slow; optional CI job).",
    ),
    root: Path | None = typer.Option(None, help="Project root."),
    sources_config: str = typer.Option("sources.yaml", help="Source registry config file."),
    data_dir: Path | None = typer.Option(None, help="Data directory override."),
    artifacts_dir: Path | None = typer.Option(None, help="Artifacts directory override."),
) -> None:
    """Run M7 scientific checks; optional NUTS/live suites do not hard-fail offline CI."""
    from civic_signal.verification.scientific import ScientificVerificationRunner

    context = _context(
        root=root,
        sources_config=sources_config,
        data_dir=data_dir,
        artifacts_dir=artifacts_dir,
    )
    payload = ScientificVerificationRunner(context).verify(
        include_live_canaries=live_canaries,
        include_nuts_smoke=nuts_smoke,
    )
    color = "green" if payload["passed"] else "red"
    console.print(
        f"[{color}]Scientific verification[/{color}]: passed={payload['passed']} "
        f"state={payload.get('evidence_state')} "
        f"failures={len(payload.get('failures') or [])}"
    )
    console.print(payload["report_path"])
    if payload.get("failures"):
        console.print(payload["failures"])
    if payload.get("missing_optional_suites"):
        console.print(f"optional pending: {payload['missing_optional_suites']}")
    if payload.get("exit_nonzero"):
        raise typer.Exit(code=1)


@shadow_app.command("preregister")
def shadow_preregister(
    profile: str = typer.Option("2026-general-shadow", help="Shadow profile id."),
    cycle: int = typer.Option(2026, help="Election cycle being preregistered."),
    force: bool = typer.Option(False, help="Allow rewriting a frozen preregistration."),
    root: Path | None = typer.Option(None, help="Project root."),
    sources_config: str = typer.Option("sources.yaml", help="Source registry config file."),
    data_dir: Path | None = typer.Option(None, help="Data directory override."),
    artifacts_dir: Path | None = typer.Option(None, help="Artifacts directory override."),
) -> None:
    """Freeze primary model and baseline definitions before a shadow window."""
    from civic_signal.shadow import ShadowForecastRunner

    context = _context(
        root=root,
        sources_config=sources_config,
        data_dir=data_dir,
        artifacts_dir=artifacts_dir,
    )
    payload = ShadowForecastRunner(context, profile_id=profile).ensure_preregistration(
        cycle=cycle, force=force
    )
    console.print(f"[green]Preregistration frozen[/green]: {payload['profile_id']}")
    console.print(payload.get("frozen_at"))


@shadow_app.command("run")
def shadow_run(
    profile: str = typer.Option("2026-general-shadow", help="Shadow profile id."),
    window_start: str = typer.Option(..., help="Inclusive window start YYYY-MM-DD."),
    window_end: str = typer.Option(..., help="Inclusive window end YYYY-MM-DD."),
    execute: bool = typer.Option(
        True,
        "--execute/--schedule-only",
        help="Execute forecasts or only materialize the schedule.",
    ),
    scenario: str | None = typer.Option(None, help="Override scenario key."),
    inference_engine: str | None = typer.Option(None, help="kalman or bayes."),
    bayesian_backend: str | None = typer.Option(None, help="analytic or nuts."),
    quiet: bool = typer.Option(True, help="Suppress forecast progress."),
    root: Path | None = typer.Option(None, help="Project root."),
    sources_config: str = typer.Option("sources.yaml", help="Source registry config file."),
    data_dir: Path | None = typer.Option(None, help="Data directory override."),
    artifacts_dir: Path | None = typer.Option(None, help="Artifacts directory override."),
) -> None:
    """Run scheduled shadow forecasts (no public production probabilities)."""
    from civic_signal.shadow import ShadowForecastRunner

    context = _context(
        root=root,
        sources_config=sources_config,
        data_dir=data_dir,
        artifacts_dir=artifacts_dir,
    )
    payload = ShadowForecastRunner(context, profile_id=profile).run_window(
        window_start=window_start,
        window_end=window_end,
        execute=execute,
        scenario=scenario,
        inference_engine=inference_engine,
        bayesian_backend=bayesian_backend,
        quiet=quiet,
    )
    console.print(
        f"[green]Shadow window ready[/green]: profile={payload['profile_id']} "
        f"days={payload['scheduled_days']} executed={len(payload['executed'])}"
    )
    console.print(payload["profile_dir"])


@data_app.command("audit")
def data_audit(
    run_id: str = typer.Option(..., help="Stable data-audit run id."),
    profile: str = typer.Option("production", help="Audit profile."),
    as_of: str | None = typer.Option(None, help="Input cutoff date YYYY-MM-DD."),
    root: Path | None = typer.Option(None, help="Project root."),
    sources_config: str = typer.Option(
        "sources_public_web.yaml", help="Source registry config file."
    ),
    data_dir: Path | None = typer.Option(None, help="Data directory override."),
    artifacts_dir: Path | None = typer.Option(None, help="Artifacts directory override."),
) -> None:
    """Audit source access, terms, snapshots, and fixture exclusion."""
    from civic_signal.verification.data_audit import DataAuditRunner

    context = _context(
        root=root,
        sources_config=sources_config,
        data_dir=data_dir,
        artifacts_dir=artifacts_dir,
    )
    payload = DataAuditRunner(context).verify(run_id=run_id, profile=profile, as_of=as_of)
    color = "green" if payload["passed"] else "yellow"
    console.print(
        f"[{color}]Data audit[/{color}]: passed={payload['passed']} "
        f"status={payload['audit']['status']}"
    )
    console.print(payload["audit_path"])
    if payload["exit_nonzero"]:
        raise typer.Exit(code=1)


@verify_app.command("historical-calibration")
def verify_historical_calibration(
    run_id: str | None = typer.Option(None, help="Stable historical calibration audit id."),
    scenario: str = typer.Option(
        "2022-midterm-historical-calibration",
        help="Scenario key for the historical calibration audit.",
    ),
    as_of: str | None = typer.Option(None, help="Forecast date in YYYY-MM-DD form."),
    inference_engine: str = typer.Option(
        "bayes", help="Polling inference engine for the calibration forecast: kalman or bayes."
    ),
    bayesian_backend: str | None = typer.Option(
        "nuts", help="Bayesian backend for the calibration forecast: analytic or nuts."
    ),
    quiet: bool = typer.Option(False, help="Suppress forecast progress during audit forecast."),
    root: Path | None = typer.Option(None, help="Project root."),
    sources_config: str = typer.Option("sources.yaml", help="Source registry config file."),
    data_dir: Path | None = typer.Option(None, help="Data directory override."),
    artifacts_dir: Path | None = typer.Option(None, help="Artifacts directory override."),
) -> None:
    """Run the compact 2022 Phase 4/5/7 historical calibration gate."""
    context = _context(
        root=root,
        sources_config=sources_config,
        data_dir=data_dir,
        artifacts_dir=artifacts_dir,
    )
    payload = ForecastPipeline(context).verify_historical_calibration(
        run_id=run_id,
        scenario=scenario,
        as_of=as_of,
        inference_engine=inference_engine,
        bayesian_backend=bayesian_backend,
        quiet=quiet,
    )
    color = "green" if payload["passed"] else "yellow"
    console.print(f"[{color}]Historical calibration[/{color}]: {payload['passed']}")
    console.print(payload["output_dir"])


@benchmark_app.command("run")
def benchmark_run(
    as_of: str = typer.Option(..., help="Benchmark date in YYYY-MM-DD form."),
    run_id: str | None = typer.Option(None, help="Stable benchmark run id."),
    draws: int | None = typer.Option(None, help="Override benchmark draw count."),
    repeats: int | None = typer.Option(None, help="Override repeat count."),
    root: Path | None = typer.Option(None, help="Project root."),
    sources_config: str = typer.Option("sources.yaml", help="Source registry config file."),
    data_dir: Path | None = typer.Option(None, help="Data directory override."),
    artifacts_dir: Path | None = typer.Option(None, help="Artifacts directory override."),
) -> None:
    """Benchmark simulation throughput using the configured performance engine."""
    context = _context(
        root=root,
        sources_config=sources_config,
        data_dir=data_dir,
        artifacts_dir=artifacts_dir,
    )
    payload = ForecastPipeline(context).run_benchmark(
        as_of=as_of,
        run_id=run_id,
        draws=draws,
        repeats=repeats,
    )
    console.print(
        "[green]Benchmark complete[/green]: "
        f"{payload['rows_per_second']:.0f} rows/sec via {payload['performance']['engine']}"
    )


@results_app.command("compare")
def results_compare(
    forecast_run_id: str = typer.Option(..., help="Existing forecast run id."),
    comparison_id: str | None = typer.Option(None, help="Stable comparison id."),
    cycle: int | None = typer.Option(None, help="Restrict comparison to a cycle."),
    office_type: str | None = typer.Option(None, help="Restrict comparison to an office type."),
    race_id: str | None = typer.Option(None, help="Restrict comparison to a race id."),
    root: Path | None = typer.Option(None, help="Project root."),
    sources_config: str = typer.Option("sources.yaml", help="Source registry config file."),
    data_dir: Path | None = typer.Option(None, help="Data directory override."),
    artifacts_dir: Path | None = typer.Option(None, help="Artifacts directory override."),
) -> None:
    """Compare a forecast run with known actual results."""
    context = _context(
        root=root,
        sources_config=sources_config,
        data_dir=data_dir,
        artifacts_dir=artifacts_dir,
    )
    payload = ForecastPipeline(context).compare_results(
        forecast_run_id=forecast_run_id,
        comparison_id=comparison_id,
        cycle=cycle,
        office_type=office_type,
        race_id=race_id,
    )
    console.print(
        "[green]Comparison complete[/green]: "
        f"{payload['race_count']} races, winner accuracy={payload['winner_accuracy']}"
    )
    console.print(payload["output_dir"])


@results_app.command("cycle-eval")
def results_cycle_eval(
    cycles: str = typer.Option(
        "2008,2012,2016,2020,2024",
        help="Comma-separated presidential cycles to evaluate.",
    ),
    as_of_mm_dd: str = typer.Option(
        "10-05", help="Same-date forecast cut in MM-DD form, e.g. 10-05."
    ),
    run_id: str | None = typer.Option(None, help="Stable cycle-eval run id."),
    scenario_template: str = typer.Option(
        "president_{cycle}_state", help="Scenario key template containing {cycle}."
    ),
    forecast_run_prefix: str = typer.Option("eval", help="Prefix for generated forecast run ids."),
    comparison_id: str = typer.Option("actuals", help="Comparison id inside each forecast run."),
    office_type: str = typer.Option("president", help="Office type passed to results compare."),
    reuse_existing: bool = typer.Option(
        False,
        help="Reuse complete forecast/comparison artifacts for matching generated run ids.",
    ),
    inference_engine: str | None = typer.Option(
        None, help="Polling inference engine for generated forecast runs."
    ),
    bayesian_backend: str | None = typer.Option(
        None, help="Bayesian backend for generated forecast runs: analytic or nuts."
    ),
    root: Path | None = typer.Option(None, help="Project root."),
    sources_config: str = typer.Option("sources.yaml", help="Source registry config file."),
    data_dir: Path | None = typer.Option(None, help="Data directory override."),
    artifacts_dir: Path | None = typer.Option(None, help="Artifacts directory override."),
) -> None:
    """Run same-date forecast-vs-actual comparisons across historical cycles."""
    context = _context(
        root=root,
        sources_config=sources_config,
        data_dir=data_dir,
        artifacts_dir=artifacts_dir,
    )
    payload = ForecastPipeline(context).run_cycle_eval(
        cycles=_parse_cycles(cycles),
        as_of_mm_dd=as_of_mm_dd,
        run_id=run_id,
        scenario_template=scenario_template,
        forecast_run_prefix=forecast_run_prefix,
        comparison_id=comparison_id,
        office_type=office_type,
        reuse_existing=reuse_existing,
        inference_engine=inference_engine,
        bayesian_backend=bayesian_backend,
    )
    aggregate = payload["aggregate"]

    def _fmt(value: object, digits: int) -> str:
        if value is None:
            return "n/a"
        return f"{float(value):.{digits}f}"

    console.print(
        "[green]Cycle evaluation complete[/green]: "
        f"{payload['cycle_count']} cycles, "
        f"mean state accuracy={_fmt(aggregate['mean_state_accuracy'], 3)}, "
        f"mean Brier={_fmt(aggregate['mean_brier_score'], 4)}, "
        f"majority winner accuracy={_fmt(aggregate.get('majority_winner_accuracy'), 3)}"
    )
    console.print(payload["output_dir"])


if __name__ == "__main__":
    app()
