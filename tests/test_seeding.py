"""Per-race RNG streams and simulation-count overrides (audit task 1.7, P1-1)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl
from polars.testing import assert_frame_equal

from civic_signal.config import ProjectContext
from civic_signal.features import FeatureBuilder
from civic_signal.ingest import SyncRunner
from civic_signal.models import EnsembleModel, FundamentalsModel, MarketModel, SimulationEngine
from civic_signal.models.common import stable_hash
from civic_signal.models.polling import PollingModel
from civic_signal.models.polling_bayes import BayesianPollingModel
from civic_signal.normalize import CuratedDataBuilder
from civic_signal.pipeline import ForecastPipeline

ROOT = Path(__file__).resolve().parents[1]
AS_OF = "2026-05-08"


def _bundle(tmp_path: Path):
    ctx = ProjectContext.create(
        root=ROOT, data_dir=tmp_path / "data", artifacts_dir=tmp_path / "artifacts"
    )
    SyncRunner(ctx).run()
    CuratedDataBuilder(ctx).run()
    bundle = FeatureBuilder(ctx).run()
    return ctx, ForecastPipeline._active_bundle(bundle, AS_OF), bundle


def _ensemble(ctx, active, bundle, config):
    estimates = [
        PollingModel(config, as_of=AS_OF).run(active),
        FundamentalsModel(config).fit(bundle).run(active),
        MarketModel(config).run(active),
    ]
    return EnsembleModel(config).run(active, estimates)


def test_stable_hash_is_process_independent() -> None:
    assert stable_hash("US-SEN-NE-2026") == stable_hash("US-SEN-NE-2026")
    assert stable_hash("a") != stable_hash("b")
    assert 0 <= stable_hash("x") < 2**32


def test_simulation_is_deterministic_and_race_isolated(tmp_path: Path) -> None:
    ctx, active, bundle = _bundle(tmp_path)
    config = ctx.read_yaml("model.yaml")
    ensemble = _ensemble(ctx, active, bundle, config)
    first = SimulationEngine(config).run(active, ensemble).draws
    second = SimulationEngine(config).run(active, ensemble).draws
    assert_frame_equal(first, second)

    races = (
        ensemble.filter(pl.col("admitted")).select("race_id").unique().sort("race_id")["race_id"]
    ).to_list()
    assert len(races) >= 2
    changed, untouched = races[0], races[1:]
    # A new poll in race A moves only race A's ensemble estimate.
    shifted = ensemble.with_columns(
        pl.when(pl.col("race_id") == changed)
        .then(pl.col("vote_share") * 0.9)
        .otherwise(pl.col("vote_share"))
        .alias("vote_share")
    )
    third = SimulationEngine(config).run(active, shifted).draws

    def subset(frame: pl.DataFrame) -> pl.DataFrame:
        return frame.filter(pl.col("race_id").is_in(untouched)).sort(
            ["race_id", "option_id", "draw_id"]
        )

    assert_frame_equal(subset(first), subset(third))
    assert not first.filter(pl.col("race_id") == changed).equals(
        third.filter(pl.col("race_id") == changed)
    )


def test_bayes_posterior_draws_are_race_isolated() -> None:
    model = BayesianPollingModel(config={"seed": 7, "bayesian": {"backend": "analytic"}})
    a1 = model._option_rng("RACE-A", "RACE-A-D").normal(size=5)
    a2 = model._option_rng("RACE-A", "RACE-A-D").normal(size=5)
    b = model._option_rng("RACE-B", "RACE-B-D").normal(size=5)
    np.testing.assert_array_equal(a1, a2)
    assert not np.allclose(a1, b)
    other_seed = BayesianPollingModel(config={"seed": 8, "bayesian": {"backend": "analytic"}})
    assert not np.allclose(a1, other_seed._option_rng("RACE-A", "RACE-A-D").normal(size=5))


def test_named_streams_differ_by_key() -> None:
    engine = SimulationEngine({"simulation_count": 50, "seed": 3})
    np.testing.assert_array_equal(
        engine._stream("local", "X").normal(size=4), engine._stream("local", "X").normal(size=4)
    )
    assert not np.allclose(
        engine._stream("local", "X").normal(size=4), engine._stream("local", "Y").normal(size=4)
    )
    assert np.all(engine._local_noise("X", 0.0) == 0.0)
    assert engine._local_noise("X", 0.05).shape == (50,)


def test_scenario_simulation_count_override() -> None:
    base = {"simulation_count": 1000, "bayesian": {"posterior_draw_count": 1000}}
    scenario = SimpleNamespace(payload={"simulation_count": 20000})
    out = ForecastPipeline._apply_scenario_simulation_count(base, scenario)
    assert out["simulation_count"] == 20000
    assert out["bayesian"]["posterior_draw_count"] == 20000
    assert base["simulation_count"] == 1000  # input not mutated
    assert ForecastPipeline._apply_scenario_simulation_count(base, None) is base
    assert (
        ForecastPipeline._apply_scenario_simulation_count(base, SimpleNamespace(payload={})) is base
    )


def test_live_scenarios_raise_simulation_counts() -> None:
    import yaml

    payload = yaml.safe_load((ROOT / "configs" / "scenarios.yaml").read_text())
    scenarios = payload.get("scenarios", payload)
    assert scenarios["senate_2026_state"]["simulation_count"] == 20000
    assert scenarios["house_2026_district"]["simulation_count"] == 10000


def test_pivotal_races_vectorised_matches_definition() -> None:
    engine = SimulationEngine({"simulation_count": 4, "seed": 1})
    # Two one-seat races, threshold 2 seats: both must be won for control.
    body = pl.DataFrame(
        {
            "race_id": ["A"] * 4 + ["B"] * 4,
            "draw_id": [0, 1, 2, 3] * 2,
            "winner": [True, True, False, False, True, False, True, False],
            "seats": [1] * 8,
            "control_body": ["senate"] * 8,
            "party": ["DEM"] * 8,
        }
    )
    counts = np.array([2, 1, 1, 0])
    rows = {
        row["race_id"]: row["pivotal_rate"]
        for row in engine._pivotal_races(body, counts, "senate", "DEM", 2)
    }
    # draw0: both won (count 2) -> each pivotal; draw1: A won, B lost (count 1) -> B pivotal;
    # draw2: A lost, count 1 -> A pivotal; draw3: count 0 -> nobody pivotal.
    assert rows == {"A": 0.5, "B": 0.5}


def test_multi_option_error_matrix_is_party_signed() -> None:
    shares = np.array([[0.4, 0.4, 0.2], [0.4, 0.4, 0.2]])
    error = np.array([0.5, -0.5])
    out = SimulationEngine._apply_multi_option_error_matrix(shares, error, [1.0, -1.0, 0.0])
    np.testing.assert_allclose(out.sum(axis=1), 1.0)
    assert out[0, 0] > out[0, 1]  # positive error helps the DEM-signed option
    assert out[1, 0] < out[1, 1]
    single = SimulationEngine._apply_multi_option_error_matrix(np.ones((3, 1)), np.zeros(3), [1.0])
    assert single.shape == (3, 1)


def test_multi_option_draw_frame_is_deterministic() -> None:
    engine = SimulationEngine({"simulation_count": 200, "seed": 11})
    options = pl.DataFrame(
        {
            "race_id": ["R"] * 3,
            "option_id": ["R-D", "R-I", "R-R"],
            "party": ["DEM", "IND", "REP"],
        }
    )
    estimate_rows = pl.DataFrame(
        {
            "race_id": ["R"] * 3,
            "option_id": ["R-D", "R-I", "R-R"],
            "vote_share": [0.45, 0.10, 0.45],
            "uncertainty": [0.05, 0.05, 0.05],
        }
    )
    spec = {
        "race_id": "R",
        "options": options,
        "estimate_rows": estimate_rows,
        "turnout_base": 1000.0,
        "systematic_error": np.zeros(200),
        "local_error": engine._local_noise("R", 0.03),
    }
    zeros, ones = np.zeros(200), np.ones(200)
    first = engine._multi_option_draw_frame([spec], zeros, ones)
    second = engine._multi_option_draw_frame([spec], zeros, ones)
    assert_frame_equal(first, second)
    assert first.height == 600
    assert first.group_by("draw_id").agg(pl.col("winner").sum())["winner"].to_list() == [1] * 200
    assert engine._multi_option_draw_frame([], zeros, ones).is_empty()
