"""Unit tests for scripts/package_run.py and scripts/compare_runs.py (audit P2-4, P2-5)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import polars as pl
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


package_run = _load("package_run")
compare_runs = _load("compare_runs")


def _chamber_summary(p: float, mean: float) -> dict:
    return {
        "run_date": "2026-10-26",
        "as_of": "2026-10-26",
        "inference": {"engine": "bayes", "bayesian_backend": "analytic", "simulation_count": 100},
        "control": {
            "DEM": {
                "control_probability": p,
                "control_probability_with_independents": p + 0.02,
                "seat_count_mean": mean,
                "seat_count_p10": mean - 4,
                "seat_count_p50": mean,
                "seat_count_p90": mean + 4,
            },
            "REP": {"control_probability": 1 - p, "seat_count_mean": 100 - mean},
        },
        "source_fetch_status": [{"source_id": "votehub", "fetch_status": "fresh"}],
    }


def test_write_top_summary_with_week_over_week(tmp_path: Path) -> None:
    prior, new = tmp_path / "weekly" / "2026-10-19", tmp_path / "weekly" / "2026-10-26"
    for run, p, mean in ((prior, 0.40, 49.0), (new, 0.45, 49.5)):
        (run / "senate").mkdir(parents=True)
        (run / "senate" / "summary.json").write_text(json.dumps(_chamber_summary(p, mean)))
    path = package_run.write_top_summary(new, prior=prior)
    payload = json.loads(path.read_text())
    assert payload["run_kind"] == "weekly"
    assert payload["as_of"] == "2026-10-26"  # no hard-coded election-eve date
    assert payload["senate"]["dem_control_prob"] == 0.45
    assert payload["senate"]["dem_control_prob_with_independents"] == pytest.approx(0.47)
    wow = payload["week_over_week"]["senate"]
    assert wow["vs"] == "2026-10-19"
    assert wow["dem_control_prob"]["delta"] == pytest.approx(0.05)
    assert wow["dem_seats"]["mean"]["delta"] == pytest.approx(0.5)
    assert payload["source_fetch_status"]["senate"][0]["fetch_status"] == "fresh"
    assert "house" not in payload
    with pytest.raises(FileNotFoundError):
        package_run.write_top_summary(tmp_path / "empty")


def test_source_fetch_status_handles_old_manifests() -> None:
    old = pl.DataFrame(
        {"source_id": ["a"], "status": ["unchanged"], "retrieved_at": ["t0"], "content_hash": ["h"]}
    )
    assert package_run.source_fetch_status(old) == [
        {"source_id": "a", "fetch_status": "unchanged", "fetched_at": "t0", "sha256": "h"}
    ]


def test_poll_stem_and_seat_distribution() -> None:
    assert package_run.poll_stem("abc123::US-SEN-NC-2026") == package_run.poll_stem(
        "abc123::US-SEN-NC-2026"
    )
    seats = pl.DataFrame(
        {"draw_id": list(range(4)), "party": ["DEM"] * 4, "seat_count_total": [48, 50, 51, 53]}
    )
    dist = package_run.seat_distribution(seats, "DEM")
    assert dist["mean"] == pytest.approx(50.5)


def test_mc_se_helpers() -> None:
    assert compare_runs.mc_se(0.5, 10000) == pytest.approx(0.005)
    assert compare_runs.se_diff(0.5, 10000, 0.5, 10000) == pytest.approx(0.005 * 2**0.5)
    assert compare_runs.bonferroni_z(1) == pytest.approx(1.96, abs=0.01)
    assert compare_runs.bonferroni_z(35) > 3.0


def _draws(rows: list[tuple[str, str, float, bool]]) -> pl.DataFrame:
    out = []
    for race, option, logit, diagnostic in rows:
        for draw in range(3):
            out.append(
                {"race_id": race, "option_id": option, "latent_logit": logit + 0.01 * draw,
                 "diagnostic_only": diagnostic, "draw_id": draw}
            )  # fmt: skip
    return pl.DataFrame(out)


def test_posterior_shift_polled_status_uses_every_option() -> None:
    rows = [
        # AK-01 style: the DEM option has no poll-bearing draws, the REP option does.
        ("AK", "AK-D", 0.0, True),
        ("AK", "AK-R", 0.0, False),
        # NE style: IND v REP with no "-D" option at all.
        ("NE", "NE-I", -0.2, False),
        ("NE", "NE-R", 0.2, False),
        # Unpolled race.
        ("AL", "AL-D", -1.0, True),
        ("AL", "AL-R", 1.0, True),
    ]
    shift = compare_runs.posterior_shift(_draws(rows), _draws(rows))
    status = {r["race_id"]: r["prior_only"] for r in shift.iter_rows(named=True)}
    assert status == {"AK": False, "NE": False, "AL": True}
    assert shift.filter(pl.col("race_id") == "NE")["m"].item() == pytest.approx(-0.19)
    assert compare_runs.posterior_shift(None, _draws(rows)) is None


def test_markdown_reports_no_significant_movers() -> None:
    lines = compare_runs.to_markdown.__code__.co_consts
    assert any(isinstance(c, str) and "No significant movers" in c for c in lines), (
        "to_markdown must say so when no race moved more than 2 MC SE"
    )


def test_dumbbell_with_no_movers(tmp_path: Path) -> None:
    res = {
        "old_run_label": "old",
        "new_run_label": "new",
        "senate": {
            "top_movers": [],
            "control": {"dem_control_prob": {"old": 0.4, "new": 0.4, "delta": 0.0}},
        },
    }
    compare_runs.dumbbell(res, tmp_path / "d.png")
    assert (tmp_path / "d.png").exists()


def test_scripts_have_no_hard_coded_election_eve_or_fl_oh_caveat() -> None:
    for name in ("package_run.py", "compare_runs.py"):
        text = (ROOT / "scripts" / name).read_text()
        assert "2026-11-02" not in text
        assert "FL/OH" not in text and "FL and OH" not in text


def test_weekly_run_script_uses_fixed_pipeline() -> None:
    text = (ROOT / "scripts" / "weekly_run.sh").read_text()
    commands = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))
    assert "--as-of" not in commands  # forecast date defaults to today
    assert "--prior" in commands and "--run-kind" in commands
    assert "sources_${CH}_live.yaml" in commands
    assert "--bayesian-backend analytic" in commands
