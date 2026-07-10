"""M7 scientific CI: properties, parity, canaries, mutations, golden, NUTS smoke."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from typer.testing import CliRunner

from civic_signal.cli import app
from civic_signal.config import ProjectContext
from civic_signal.scientific.contract_parity import ContractParityChecker
from civic_signal.scientific.golden import validate_golden_bundle
from civic_signal.scientific.live_canaries import CanaryCase, LiveSourceCanaryRunner
from civic_signal.scientific.mutation import (
    mutation_breaks_check,
    run_standard_mutation_probes,
    strict_range_gate,
    strip_range_gate,
)
from civic_signal.scientific.parity import numerical_parity_report
from civic_signal.scientific.properties import (
    control_reconciliation_ok,
    covariance_is_psd,
    interval_ordering_ok,
    label_symmetry_holds,
    option_order_invariance_ok,
    probability_simplex_ok,
    run_offline_property_suite,
)
from civic_signal.verification.scientific import ScientificVerificationRunner

REPO_ROOT = Path(__file__).resolve().parents[1]
GOLDEN = REPO_ROOT / "tests" / "golden_fixtures"


def _ctx(tmp_path: Path) -> ProjectContext:
    return ProjectContext.create(
        root=REPO_ROOT,
        data_dir=tmp_path / "data",
        artifacts_dir=tmp_path / "artifacts",
    )


def test_probability_simplex_and_interval_ordering() -> None:
    good = pl.DataFrame(
        {
            "race_id": ["a", "a", "b"],
            "option_id": ["DEM", "REP", "DEM"],
            "winner_probability": [0.6, 0.4, 0.7],
            "share_p10": [0.4, 0.3, 0.5],
            "share_p50": [0.55, 0.45, 0.65],
            "share_p90": [0.7, 0.6, 0.8],
        }
    )
    assert probability_simplex_ok(good)["ok"] is True
    assert interval_ordering_ok(good)["ok"] is True

    bad_sum = good.with_columns(pl.lit(0.9).alias("winner_probability"))
    assert probability_simplex_ok(bad_sum)["ok"] is False

    bad_interval = good.with_columns(pl.lit(0.9).alias("share_p10"))
    assert interval_ordering_ok(bad_interval)["ok"] is False


def test_label_symmetry_and_option_order() -> None:
    dem = np.array([0.55, 0.4, 0.5])
    rep = 1.0 - dem
    assert label_symmetry_holds(dem, rep)["ok"] is True
    assert label_symmetry_holds(dem, rep + 0.1)["ok"] is False

    frame = pl.DataFrame(
        {
            "race_id": ["r1", "r1", "r1"],
            "option_id": ["C", "A", "B"],
            "winner_probability": [0.2, 0.5, 0.3],
        }
    )
    assert option_order_invariance_ok(frame)["ok"] is True


def test_covariance_psd_and_control_recon() -> None:
    psd = np.array([[1.0, 0.2], [0.2, 1.0]])
    assert covariance_is_psd(psd)["ok"] is True
    not_sym = np.array([[1.0, 0.5], [0.1, 1.0]])
    assert covariance_is_psd(not_sym)["ok"] is False
    not_psd = np.array([[1.0, 2.0], [2.0, 1.0]])
    assert covariance_is_psd(not_psd)["ok"] is False

    races = pl.DataFrame({"race_id": ["a", "b"], "winner_probability": [0.6, 0.4]})
    control = pl.DataFrame(
        {"control_body": ["senate"], "party": ["DEM"], "majority_probability": [0.55]}
    )
    assert control_reconciliation_ok(races, control)["ok"] is True
    assert control_reconciliation_ok(races, pl.DataFrame())["ok"] is False


def test_numerical_parity_matches_golden_fingerprint() -> None:
    report = numerical_parity_report(seed=20260508, race_count=4, draw_count=64)
    assert report["python_numba_match"] is True
    assert report["serial_parallel_match"] is True
    expected = (GOLDEN / "parity_fingerprint.json").read_text(encoding="utf-8")
    import json

    golden = json.loads(expected)
    assert report["python_fingerprint"] == golden["python_fingerprint"]
    if report["numba_available"]:
        assert report["numba_fingerprint"] == golden["numba_fingerprint"]
        assert report["python_fingerprint"] == report["numba_fingerprint"]


def test_mutation_probes_detect_weakened_verifiers() -> None:
    classic = mutation_breaks_check(
        healthy=lambda: strict_range_gate([0.1, 0.9]),
        mutated=lambda: strict_range_gate([1.5]),
    )
    assert classic["mutation_detected"] is True
    assert strip_range_gate([1.5]) is True
    probes = run_standard_mutation_probes()
    assert probes["all_mutations_detected"] is True
    assert probes["actual_verifier_paths"] is True
    assert probes["incomplete_mutation_families"] == []
    assert set(probes["completed_mutation_families"]) == set(probes["required_mutation_families"])
    for probe in probes["probes"].values():
        assert probe["healthy_accepts_valid"] is True
        assert probe["healthy_rejects_corruption"] is True
        assert probe["mutant_accepts_corruption"] is True
        assert probe["mutation_detected"] is True


def test_contract_parity_against_repo() -> None:
    result = ContractParityChecker(REPO_ROOT).run()
    assert result["passed"] is True, result
    generated = result["generated_assertions"]
    registry = generated["reward_registry"]
    assert registry["missing_thresholds"] == []
    assert registry["extra_thresholds"] == []
    assert registry["unknown_profile_rewards"] == []
    assert set(registry["reward_ids"]) == set(generated["evaluator_reward_ids"])
    assert set(registry["reward_ids"]).issubset(generated["documented_reward_ids"])
    assert set(generated["required_cli_surfaces"]).issubset(generated["actual_cli_surfaces"])


def test_golden_bundle_validates() -> None:
    result = validate_golden_bundle(GOLDEN)
    assert result["passed"] is True, result["failures"]
    assert result["cases"]["quarantine_failure"]["failed_as_expected"] is True


def test_offline_canaries_success_and_injected_failure(tmp_path: Path) -> None:
    def fetcher(url: str, timeout: float) -> tuple[int, bytes]:
        del timeout
        if "missing" in url:
            raise OSError("not found")
        if "empty" in url:
            return 200, b""
        return 200, b"ok,1\n"

    runner = LiveSourceCanaryRunner(fetcher=fetcher)
    payload = runner.run(
        [
            CanaryCase(name="ok", url="https://example.test/ok"),
            CanaryCase(name="missing", url="https://example.invalid/missing", expect_success=False),
            CanaryCase(name="empty", url="https://example.test/empty", expect_success=False),
        ],
        output_path=str(tmp_path / "canaries.json"),
    )
    assert payload["all_passed"] is True
    assert (tmp_path / "canaries.json").exists()
    assert len(payload["injected_failure_results"]) == 2


def test_offline_property_suite_covers_required_families() -> None:
    suite = run_offline_property_suite()
    assert suite["passed"] is True, suite
    assert suite["missing_property_families"] == []
    assert set(suite["completed_property_families"]) == set(suite["required_property_families"])


def test_scientific_verification_runner(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    payload = ScientificVerificationRunner(ctx).verify(
        include_live_canaries=False,
        include_nuts_smoke=False,
        fixture_root=GOLDEN,
    )
    assert payload["passed"] is True, payload["failures"]
    assert payload["checks_passed"] is True, payload["failures"]
    assert payload["exit_nonzero"] is False
    assert payload["evidence_state"] == "checks_passed_optional_suites_pending"
    assert payload["missing_optional_suites"]
    assert (tmp_path / "artifacts" / "scientific" / "scientific_report.json").exists()
    assert payload["properties"]["passed"] is True
    assert payload["parity"]["python_numba_match"] is True
    assert payload["mutations"]["all_mutations_detected"] is True
    assert payload["canaries"]["all_passed"] is True
    assert payload["contract_parity"]["passed"] is True
    # Default offline path never hits the network for canaries.
    assert payload["canaries"]["live"] is False
    assert payload["nuts_smoke"] is None


def test_scientific_verification_fails_closed_on_incomplete_mutation_family(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "civic_signal.verification.scientific.run_standard_mutation_probes",
        lambda _root=None: {
            "actual_verifier_paths": True,
            "all_mutations_detected": True,
            "incomplete_mutation_families": ["publication_estimand_gate"],
        },
    )
    payload = ScientificVerificationRunner(_ctx(tmp_path)).verify(
        include_live_canaries=False,
        include_nuts_smoke=False,
        fixture_root=GOLDEN,
    )
    assert payload["passed"] is False
    assert payload["exit_nonzero"] is True
    assert payload["missing_required_suites"] == ["actual reward/publication source mutation suite"]
    assert (
        "actual reward/publication mutation suite is incomplete or survived" in payload["failures"]
    )


def test_verify_scientific_cli(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "verify",
            "scientific",
            "--root",
            str(REPO_ROOT),
            "--artifacts-dir",
            str(tmp_path / "artifacts"),
            "--data-dir",
            str(tmp_path / "data"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Scientific verification" in result.output


@pytest.mark.nuts
def test_nuts_smoke_tiny() -> None:
    from civic_signal.scientific.nuts_smoke import run_nuts_smoke

    report = run_nuts_smoke(
        num_warmup=10,
        num_samples=10,
        num_chains=1,
        wall_clock_timeout_seconds=120.0,
    )
    assert report["ok"] is True, report
    assert report["divergences"] == 0
    assert report["state_logit_shape"]


def test_property_edge_cases() -> None:
    assert probability_simplex_ok(pl.DataFrame())["ok"] is False
    null_only = pl.DataFrame(
        {"race_id": ["a"], "winner_probability": [None]},
    )
    assert probability_simplex_ok(null_only)["ok"] is True
    single = pl.DataFrame({"race_id": ["a"], "winner_probability": [0.7]})
    assert probability_simplex_ok(single)["ok"] is True
    oob = pl.DataFrame({"race_id": ["a"], "winner_probability": [1.2]})
    assert probability_simplex_ok(oob)["ok"] is False

    assert covariance_is_psd(np.array([1.0, 2.0]))["ok"] is False
    assert label_symmetry_holds(np.array([0.5]), np.array([0.5, 0.5]))["ok"] is False
    assert label_symmetry_holds(np.array([]), np.array([]))["ok"] is True

    races = pl.DataFrame({"race_id": ["x"]})
    assert control_reconciliation_ok(races, pl.DataFrame({"party": ["DEM"]}))["ok"] is False
    assert (
        control_reconciliation_ok(pl.DataFrame(), pl.DataFrame({"party": ["DEM"]}))["ok"] is False
    )
    assert (
        control_reconciliation_ok(
            pl.DataFrame({"other": [1]}),
            pl.DataFrame({"control_body": ["senate"], "majority_probability": [0.5]}),
        )["ok"]
        is False
    )
    bad_ctrl = pl.DataFrame(
        {"control_body": ["senate"], "majority_probability": [1.5]},
    )
    assert (
        control_reconciliation_ok(
            pl.DataFrame({"race_id": ["a"]}),
            bad_ctrl,
        )["ok"]
        is False
    )
    assert interval_ordering_ok(pl.DataFrame({"x": [1]}))["ok"] is True
    assert option_order_invariance_ok(pl.DataFrame({"race_id": ["a"]}))["ok"] is False


def test_contract_parity_failure_paths(tmp_path: Path) -> None:
    # Empty root: missing docs/config.
    empty = ContractParityChecker(tmp_path).run(output_path=tmp_path / "parity.json")
    assert empty["passed"] is False
    assert (tmp_path / "parity.json").exists()

    # Stale claim: production default language without research qualifier.
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "rewards.yaml").write_text(
        "publication_mode_default: production\n", encoding="utf-8"
    )
    (tmp_path / "README.md").write_text(
        "The Bayesian path is the production default without caveats.\n",
        encoding="utf-8",
    )
    (tmp_path / "SPEC.md").write_text("no reward content\n", encoding="utf-8")
    (tmp_path / "src" / "civic_signal").mkdir(parents=True)
    (tmp_path / "src" / "civic_signal" / "cli.py").write_text("app = None\n", encoding="utf-8")
    stale = ContractParityChecker(tmp_path).run()
    assert stale["passed"] is False
    assert stale["stale_claims"] >= 1


def test_canary_expected_success_failure_and_defaults(tmp_path: Path) -> None:
    def bad_success(url: str, timeout: float) -> tuple[int, bytes]:
        del url, timeout
        return 500, b"nope"

    failed = LiveSourceCanaryRunner(fetcher=bad_success).run(
        [CanaryCase(name="need_ok", url="https://example.test/x", expect_success=True)]
    )
    assert failed["all_passed"] is False

    def unexpected_ok(url: str, timeout: float) -> tuple[int, bytes]:
        del url, timeout
        return 200, b"data"

    unexpected = LiveSourceCanaryRunner(fetcher=unexpected_ok).run(
        [CanaryCase(name="should_fail", url="https://example.test/x", expect_success=False)]
    )
    assert unexpected["all_passed"] is False
    assert unexpected["injected_failure_results"][0]["status"] == "success"

    cases = LiveSourceCanaryRunner.default_cases()
    assert len(cases) >= 2
    assert any(not case.expect_success for case in cases)


def test_scientific_runner_failure_branches(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    # Corrupt golden quarantine so failed_as_expected is false.
    bad_root = tmp_path / "bad_fixtures"
    bad_root.mkdir()
    for name in (
        "small_election.json",
        "chamber_control.json",
        "multi_option_race.json",
        "quarantine_failure.json",
        "parity_fingerprint.json",
    ):
        (bad_root / name).write_text((GOLDEN / name).read_text(encoding="utf-8"), encoding="utf-8")
    quarantine = json.loads((bad_root / "quarantine_failure.json").read_text(encoding="utf-8"))
    quarantine["race_forecasts"] = [
        {"race_id": "ok", "option_id": "DEM", "winner_probability": 0.5},
        {"race_id": "ok", "option_id": "REP", "winner_probability": 0.5},
    ]
    (bad_root / "quarantine_failure.json").write_text(json.dumps(quarantine), encoding="utf-8")
    # Force fingerprint drift.
    parity = json.loads((bad_root / "parity_fingerprint.json").read_text(encoding="utf-8"))
    parity["python_fingerprint"] = "0" * 64
    parity["numba_fingerprint"] = "1" * 64
    (bad_root / "parity_fingerprint.json").write_text(json.dumps(parity), encoding="utf-8")

    payload = ScientificVerificationRunner(ctx).verify(
        include_live_canaries=False,
        include_nuts_smoke=False,
        fixture_root=bad_root,
    )
    assert payload["passed"] is False
    assert any("fingerprint" in item or "golden" in item for item in payload["failures"])


def test_golden_default_root_and_load() -> None:
    from civic_signal.scientific.golden import default_fixture_root, load_json_fixture

    root = default_fixture_root()
    assert root.exists()
    payload = load_json_fixture("small_election.json", root=root)
    assert payload["expected_race_count"] == 1


def test_python_kernel_parity_path() -> None:
    from civic_signal.performance.kernels import (
        NUMBA_AVAILABLE,
        configure_numba_threads,
        python_binary_draw_kernel,
        simulate_binary_draw_arrays,
    )

    first = np.array([0.5, 0.6], dtype=np.float64)
    turnout = np.array([1000.0, 2000.0], dtype=np.float64)
    national = np.array([0.0, 0.01], dtype=np.float64)
    local = np.zeros((2, 2), dtype=np.float64)
    party = np.ones(2, dtype=np.float64)
    mult = np.ones(2, dtype=np.float64)
    py = python_binary_draw_kernel(first, turnout, national, local, party, mult)
    via = simulate_binary_draw_arrays(
        first,
        turnout,
        national,
        local,
        use_numba=False,
        party_signs=party,
        turnout_multipliers=mult,
    )
    assert all(np.array_equal(a, b) for a, b in zip(py, via, strict=True))
    if NUMBA_AVAILABLE:
        numba = simulate_binary_draw_arrays(
            first,
            turnout,
            national,
            local,
            use_numba=True,
            party_signs=party,
            turnout_multipliers=mult,
        )
        assert all(np.allclose(a, b) for a, b in zip(py, numba, strict=True))
        previous = configure_numba_threads(1)
        assert previous is not None
        configure_numba_threads(previous)


def test_scoring_lazy_exports() -> None:
    import civic_signal.scoring as scoring

    assert scoring.score_predictions is not None
    assert scoring.BacktestRunner is not None
    assert scoring.RewardV2Evaluator is not None
    assert scoring.RewardEvaluator is not None
    assert scoring.ResultComparator is not None
    assert scoring.CycleEvaluationReport is not None
    with pytest.raises(AttributeError):
        _ = scoring.does_not_exist  # type: ignore[attr-defined]


def test_recovery_analytic_smoke(tmp_path: Path) -> None:
    from civic_signal.verification.recovery import RecoveryVerificationRunner

    ctx = _ctx(tmp_path)
    runner = RecoveryVerificationRunner(ctx)
    with pytest.raises(ValueError):
        runner.verify(backend="nope", replicates=2)
    with pytest.raises(ValueError):
        runner.verify(backend="analytic", replicates=1)
    payload = runner.verify(run_id="recovery-unit", backend="analytic", replicates=2)
    assert payload["production_sufficient"] is False
    assert payload["evidence_scope"] == "synthetic_bounded_recovery_smoke"
    assert "parameter_recovery" in payload["checks"]
    assert "label_symmetry" in payload["checks"]
    out = Path(payload["output_dir"])
    assert (out / "hierarchy_recovery.json").exists()
    assert (out / "hierarchy_recovery.md").exists()


def test_verify_recovery_cli(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "verify",
            "recovery",
            "--backend",
            "analytic",
            "--replicates",
            "2",
            "--run-id",
            "cli-recovery",
            "--root",
            str(REPO_ROOT),
            "--artifacts-dir",
            str(tmp_path / "artifacts"),
            "--data-dir",
            str(tmp_path / "data"),
        ],
    )
    assert result.exit_code in {0, 1}, result.output
    assert "Recovery" in result.output


def test_shadow_runner_execute_day_mocked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from civic_signal.shadow.runner import ShadowForecastRunner

    ctx = _ctx(tmp_path)
    runner = ShadowForecastRunner(ctx, profile_id="2026-general-shadow")
    runner.ensure_preregistration(cycle=2026)

    def fake_forecast(**kwargs):  # type: ignore[no-untyped-def]
        run_id = kwargs.get("run_id") or "shadow-fake"
        out = ctx.artifacts_dir / "runs" / str(run_id)
        out.mkdir(parents=True, exist_ok=True)
        pl.DataFrame(
            {
                "source_id": ["a"],
                "status": ["fetched"],
                "retrieved_at": ["2026-05-08T00:00:00+00:00"],
            }
        ).write_parquet(out / "source_manifest.parquet")
        from civic_signal.storage.io import write_json

        write_json({"fallback_used": None, "divergences": 0}, out / "posterior_diagnostics.json")
        write_json({"max_mcse": 0.001}, out / "performance.json")
        write_json(
            {"publication_mode": "shadow", "allowed": False},
            out / "publication_decision.json",
        )
        return out

    monkeypatch.setattr(
        "civic_signal.shadow.runner.ForecastPipeline",
        lambda context: type(
            "P",
            (),
            {"run_forecast": staticmethod(lambda **kw: fake_forecast(**kw))},
        )(),
    )
    payload = runner.run_window(
        window_start="2026-06-01",
        window_end="2026-06-01",
        execute=True,
        quiet=True,
    )
    assert payload["scheduled_days"] == 1
    assert len(payload["executed"]) == 1


def test_coherence_verification_runner(tmp_path: Path) -> None:
    from civic_signal.verification.coherence import CoherenceVerificationRunner

    ctx = _ctx(tmp_path)
    run = ctx.artifacts_dir / "runs" / "coh-unit"
    run.mkdir(parents=True)
    pl.DataFrame(
        {
            "race_id": ["r1", "r1", "r2"],
            "option_id": ["DEM", "REP", "DEM"],
            "tier": ["A", "A", "C"],
            "control_body": ["senate", "senate", None],
            "seats": [1, 1, 0],
            "office_type": ["senate", "senate", "senate"],
            "geography": ["AA", "AA", "BB"],
        }
    ).write_parquet(run / "race_catalog.parquet")
    pl.DataFrame(
        {
            "race_id": ["r1", "r1"],
            "option_id": ["DEM", "REP"],
            "winner_probability": [0.55, 0.45],
            "party": ["DEM", "REP"],
        }
    ).write_parquet(run / "race_forecasts.parquet")
    pl.DataFrame(
        {
            "draw_id": [0, 0, 1, 1],
            "race_id": ["r1", "r1", "r1", "r1"],
            "option_id": ["DEM", "REP", "DEM", "REP"],
            "party": ["DEM", "REP", "DEM", "REP"],
            "vote_share": [0.6, 0.4, 0.4, 0.6],
            "winner": [True, False, False, True],
        }
    ).write_parquet(run / "forecast_draws.parquet")
    pl.DataFrame(
        {
            "control_body": ["senate"],
            "party": ["DEM"],
            "majority_probability": [0.0],
            "control_probability": [0.0],
            "control_threshold": [51],
            "holdover_seats": [0],
        }
    ).write_parquet(run / "control_forecasts.parquet")

    payload = CoherenceVerificationRunner(ctx).verify(run_id="coh-unit", audit_id="coh-audit")
    assert payload["passed"] is True, payload["checks"]
    assert (Path(payload["output_dir"]) / "coherence_verification.json").exists()
    with pytest.raises(FileNotFoundError):
        CoherenceVerificationRunner(ctx).verify(run_id="missing-run")
