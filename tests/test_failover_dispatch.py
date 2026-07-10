from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from civic_signal.config import ProjectContext
from civic_signal.features import FeatureBundle
from civic_signal.inference.failover import (
    FailoverPolicy,
    FailoverRefusedError,
    dispatch_failover,
    execute_with_failover,
    exercise_timeout_failover,
    load_previous_posterior_artifact,
)
from civic_signal.models.polling_bayes import BayesianPollingModel

ROOT = Path(__file__).resolve().parents[1]


def _bundle() -> FeatureBundle:
    catalog = pl.DataFrame(
        {
            "race_id": ["R1"],
            "cycle": [2026],
            "office_type": ["senate"],
            "geography": ["GA"],
            "election_date": ["2026-11-03"],
            "tier": ["A"],
        }
    )
    options = pl.DataFrame(
        [
            {"race_id": "R1", "option_id": "D", "party": "DEM", "previous_vote_share": 0.5},
            {"race_id": "R1", "option_id": "R", "party": "REP", "previous_vote_share": 0.5},
        ]
    )
    polls = pl.DataFrame(
        [
            {
                "poll_id": f"P1-{option_id}",
                "survey_id": "S1",
                "question_id": "Q1",
                "race_id": "R1",
                "option_id": option_id,
                "pollster": "Example",
                "start_date": "2026-04-01",
                "end_date": "2026-04-03",
                "population": "lv",
                "sample_size": 1000,
                "sponsor_class": "nonpartisan",
                "methodology": "live_phone",
                "pct": pct,
                "source_hash": "a" * 64,
            }
            for option_id, pct in (("D", 52.0), ("R", 48.0))
        ]
    )
    empty = pl.DataFrame(schema={"race_id": pl.String})
    return FeatureBundle(
        races=catalog,
        options=options,
        polls=polls,
        markets=empty,
        public_signals=empty,
        fundamentals=empty,
        results=empty,
        backtest_predictions=empty,
        race_catalog=catalog,
    )


def _config(tmp_path: Path) -> dict[str, object]:
    context = ProjectContext.create(
        root=ROOT,
        data_dir=tmp_path / "data",
        artifacts_dir=tmp_path / "artifacts",
    )
    config = json.loads(json.dumps(context.read_yaml("model.yaml")))
    config["_bayesian_backend"] = "nuts"
    config["bayesian"]["posterior_draw_count"] = 100
    return config


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        (FailoverPolicy.ANALYTIC, "analytic-result"),
        (FailoverPolicy.KALMAN, "kalman-result"),
    ],
)
def test_dispatcher_executes_literal_handler_label(label: str, expected: str) -> None:
    result = dispatch_failover(
        FailoverPolicy(fallback_order=(label, FailoverPolicy.REFUSE)),
        primary_engine="numpyro-nuts",
        reason="forced failure",
        handlers={label: lambda: expected},
    )

    assert result.result == expected
    assert result.audit["fallback_used"] == label
    assert result.audit["executed_path"] == label
    assert result.audit["attempts"][-1]["status"] == "executed"
    assert result.audit["publication_blocked"] is True
    assert result.audit["quarantine_required"] is True


def test_previous_posterior_requires_loaded_compatible_artifact(tmp_path: Path) -> None:
    artifact = tmp_path / "previous.json"
    artifact.write_text('{"payload": "previous-result"}', encoding="utf-8")
    loaded = load_previous_posterior_artifact(
        artifact,
        loader=lambda path: json.loads(path.read_text(encoding="utf-8"))["payload"],
        validator=lambda payload: (payload == "previous-result", "compatible", {"id": "p1"}),
    )

    result = dispatch_failover(
        FailoverPolicy(fallback_order=(FailoverPolicy.PREVIOUS_POSTERIOR, FailoverPolicy.REFUSE)),
        primary_engine="numpyro-nuts",
        reason="forced failure",
        previous_posterior=loaded,
    )

    assert result.result == "previous-result"
    assert result.audit["fallback_used"] == FailoverPolicy.PREVIOUS_POSTERIOR
    assert result.audit["attempts"][0]["artifact_path"] == str(artifact)


@pytest.mark.parametrize("failure", ["corrupt", "incompatible", "not_loaded"])
def test_bad_previous_posterior_reaches_literal_refuse(tmp_path: Path, failure: str) -> None:
    if failure == "not_loaded":
        loaded = None
    else:
        artifact = tmp_path / "previous.json"
        artifact.write_text("not-json" if failure == "corrupt" else "{}", encoding="utf-8")
        loaded = load_previous_posterior_artifact(
            artifact,
            loader=lambda path: json.loads(path.read_text(encoding="utf-8")),
            validator=lambda _payload: (False, "model hash mismatch", {}),
        )

    with pytest.raises(FailoverRefusedError) as raised:
        dispatch_failover(
            FailoverPolicy(
                fallback_order=(FailoverPolicy.PREVIOUS_POSTERIOR, FailoverPolicy.REFUSE)
            ),
            primary_engine="numpyro-nuts",
            reason="forced failure",
            previous_posterior=loaded,
        )

    audit = raised.value.audit
    assert audit["status"] == "refused"
    assert audit["fallback_used"] is None
    assert audit["executed_path"] == FailoverPolicy.REFUSE
    assert audit["publication_blocked"] is True
    assert audit["attempts"][0]["status"] in {
        "unavailable",
        "incompatible",
        "corrupt_or_unreadable",
    }


def test_svi_and_misordered_refuse_are_rejected() -> None:
    with pytest.raises(ValueError, match="Unimplemented Bayesian fallback"):
        FailoverPolicy(fallback_order=("svi_fallback",))
    with pytest.raises(ValueError, match="final ordered path"):
        FailoverPolicy(fallback_order=(FailoverPolicy.REFUSE, FailoverPolicy.ANALYTIC))


def test_bayesian_model_executes_kalman_fallback(tmp_path: Path, monkeypatch) -> None:
    config = _config(tmp_path)
    config["bayesian"]["nuts"]["failover"]["fallback_order"] = [
        FailoverPolicy.KALMAN,
        FailoverPolicy.REFUSE,
    ]
    monkeypatch.setattr(
        BayesianPollingModel,
        "_fit_nuts_backend",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("forced nuts failure")),
    )
    model = BayesianPollingModel(config, as_of="2026-05-08")

    estimates = model.run(_bundle())
    diagnostics = model.diagnostics()

    assert not estimates.is_empty()
    assert diagnostics["engine"] == "legacy-kalman-fallback"
    assert diagnostics["fallback_used"] == FailoverPolicy.KALMAN
    assert diagnostics["failover_audit"]["executed_path"] == FailoverPolicy.KALMAN
    assert model.posterior_draws(_bundle()).is_empty()


def test_bayesian_model_reuses_validated_previous_posterior(tmp_path: Path, monkeypatch) -> None:
    bundle = _bundle()
    base_config = _config(tmp_path)
    base_config["_bayesian_backend"] = "analytic"
    previous_model = BayesianPollingModel(base_config, as_of="2026-05-07")
    previous_model.run(bundle)
    artifact = tmp_path / "posterior.parquet"
    previous_model.posterior_draws(bundle).with_columns(
        pl.lit("compatible-hash").alias("model_config_hash"),
        pl.lit("source-hash").alias("source_manifest_hash"),
    ).write_parquet(artifact)

    config = _config(tmp_path)
    config["bayesian"]["nuts"]["failover"] = {
        "fallback_order": [FailoverPolicy.PREVIOUS_POSTERIOR, FailoverPolicy.REFUSE],
        "block_publication_on_fallback": True,
        "previous_posterior": {
            "path": str(artifact),
            "model_config_hash": "compatible-hash",
            "source_manifest_hash": "source-hash",
            "as_of": "2026-05-07",
            "max_age_days": 2,
        },
    }
    monkeypatch.setattr(
        BayesianPollingModel,
        "_fit_nuts_backend",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("forced nuts failure")),
    )
    model = BayesianPollingModel(config, as_of="2026-05-08")

    estimates = model.run(bundle)
    diagnostics = model.diagnostics()

    assert not estimates.is_empty()
    assert diagnostics["engine"] == "previous-posterior-reuse"
    assert diagnostics["fallback_used"] == FailoverPolicy.PREVIOUS_POSTERIOR
    assert diagnostics["previous_posterior_artifact"]["age_days"] == 1
    assert diagnostics["failover_audit"]["publication_blocked"] is True


def test_bayesian_model_refuses_incompatible_previous_posterior(
    tmp_path: Path, monkeypatch
) -> None:
    bundle = _bundle()
    artifact = tmp_path / "posterior.parquet"
    pl.DataFrame({"broken": [True]}).write_parquet(artifact)
    config = _config(tmp_path)
    config["bayesian"]["nuts"]["failover"] = {
        "fallback_order": [FailoverPolicy.PREVIOUS_POSTERIOR, FailoverPolicy.REFUSE],
        "previous_posterior": {
            "path": str(artifact),
            "model_config_hash": "compatible-hash",
            "source_manifest_hash": "source-hash",
            "as_of": "2026-05-07",
        },
    }
    monkeypatch.setattr(
        BayesianPollingModel,
        "_fit_nuts_backend",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("forced nuts failure")),
    )

    with pytest.raises(FailoverRefusedError) as raised:
        BayesianPollingModel(config, as_of="2026-05-08").run(bundle)

    assert raised.value.audit["attempts"][0]["status"] == "corrupt_or_unreadable"
    assert raised.value.audit["executed_path"] == FailoverPolicy.REFUSE


def test_bayesian_model_refuses_previous_posterior_source_lineage_mismatch(
    tmp_path: Path, monkeypatch
) -> None:
    bundle = _bundle()
    base_config = _config(tmp_path)
    base_config["_bayesian_backend"] = "analytic"
    previous_model = BayesianPollingModel(base_config, as_of="2026-05-07")
    previous_model.run(bundle)
    artifact = tmp_path / "posterior.parquet"
    previous_model.posterior_draws(bundle).with_columns(
        pl.lit("compatible-hash").alias("model_config_hash"),
        pl.lit("artifact-source-hash").alias("source_manifest_hash"),
    ).write_parquet(artifact)
    config = _config(tmp_path)
    config["bayesian"]["nuts"]["failover"] = {
        "fallback_order": [FailoverPolicy.PREVIOUS_POSTERIOR, FailoverPolicy.REFUSE],
        "previous_posterior": {
            "path": str(artifact),
            "model_config_hash": "compatible-hash",
            "source_manifest_hash": "different-source-hash",
            "as_of": "2026-05-07",
        },
    }
    monkeypatch.setattr(
        BayesianPollingModel,
        "_fit_nuts_backend",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("forced nuts failure")),
    )

    with pytest.raises(FailoverRefusedError) as raised:
        BayesianPollingModel(config, as_of="2026-05-08").run(bundle)

    first_attempt = raised.value.audit["attempts"][0]
    assert first_attempt["status"] == "incompatible"
    assert "source_manifest_hash mismatch" in first_attempt["reason"]
    assert raised.value.audit["executed_path"] == FailoverPolicy.REFUSE


def test_timeout_negative_matrix_leaves_audit_and_blocks_publication() -> None:
    """Timeout → analytic fallback records attempt status and publication block."""
    import time

    def _slow() -> str:
        time.sleep(0.05)
        return "never"

    policy = FailoverPolicy(
        timeout_seconds=0.01,
        fallback_order=(FailoverPolicy.ANALYTIC, FailoverPolicy.REFUSE),
        block_publication_on_fallback=True,
    )
    result = execute_with_failover(
        primary=_slow,
        fallback=lambda: "analytic-timeout-result",
        policy=policy,
        primary_engine="numpyro-nuts",
    )
    assert result.result == "analytic-timeout-result"
    assert result.audit["status"] == "fallback_used"
    assert result.audit["fallback_used"] == FailoverPolicy.ANALYTIC
    assert result.audit["publication_blocked"] is True
    assert result.audit["quarantine_required"] is True
    assert result.audit["attempts"][0]["status"] == "executed"
    assert (
        "timeout" in str(result.audit["reason"]).lower()
        or result.audit["elapsed_seconds"] is not None
    )


def test_timeout_refuse_path_leaves_refused_audit() -> None:
    exercised = exercise_timeout_failover(
        FailoverPolicy(
            timeout_seconds=0.01,
            fallback_order=(FailoverPolicy.REFUSE,),
            block_publication_on_fallback=True,
        )
    )
    assert exercised["passed"] is True
    audit = exercised["audit"]
    assert audit["status"] == "refused"
    assert audit["executed_path"] == FailoverPolicy.REFUSE
    assert audit["publication_blocked"] is True
    assert audit["attempts"][-1]["path"] == FailoverPolicy.REFUSE
    assert audit["attempts"][-1]["status"] == "executed"


def test_timeout_previous_posterior_reuse_and_unavailable_matrix(tmp_path: Path) -> None:
    artifact = tmp_path / "previous.json"
    artifact.write_text('{"payload": "prev"}', encoding="utf-8")
    loaded = load_previous_posterior_artifact(
        artifact,
        loader=lambda path: json.loads(path.read_text(encoding="utf-8"))["payload"],
        validator=lambda payload: (True, "compatible", {"id": "prev"}),
    )
    reused = exercise_timeout_failover(
        FailoverPolicy(
            timeout_seconds=0.01,
            fallback_order=(FailoverPolicy.PREVIOUS_POSTERIOR, FailoverPolicy.REFUSE),
            block_publication_on_fallback=True,
        ),
        previous_posterior=loaded,
    )
    assert reused["passed"] is True
    assert reused["result"] == "prev"
    assert reused["audit"]["fallback_used"] == FailoverPolicy.PREVIOUS_POSTERIOR
    assert reused["audit"]["publication_blocked"] is True
    assert reused["audit"]["attempts"][0]["status"] == "executed"

    refused = exercise_timeout_failover(
        FailoverPolicy(
            timeout_seconds=0.01,
            fallback_order=(FailoverPolicy.PREVIOUS_POSTERIOR, FailoverPolicy.REFUSE),
        ),
        previous_posterior=None,
    )
    assert refused["passed"] is True
    assert refused["audit"]["status"] == "refused"
    assert refused["audit"]["attempts"][0]["status"] == "unavailable"
    assert refused["audit"]["publication_blocked"] is True


def test_publication_block_respects_policy_flag() -> None:
    result = dispatch_failover(
        FailoverPolicy(
            fallback_order=(FailoverPolicy.ANALYTIC, FailoverPolicy.REFUSE),
            block_publication_on_fallback=False,
        ),
        primary_engine="numpyro-nuts",
        reason="forced",
        handlers={FailoverPolicy.ANALYTIC: lambda: "ok"},
    )
    assert result.audit["publication_blocked"] is False
    assert result.audit["quarantine_required"] is False
    assert result.audit["fallback_used"] == FailoverPolicy.ANALYTIC
