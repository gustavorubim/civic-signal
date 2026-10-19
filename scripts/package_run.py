"""Package a completed civic-signal chamber forecast into a portable run folder.

Copies the key artifacts/plots from an artifacts run directory into
``<out>/<chamber>/`` and writes ``<out>/<chamber>/summary.json`` using the same
chamber-block schema as the weekly digest (``runs/weekly/<date>/summary.json``),
extended with full per-race tables, seat-distribution stats, full tipping-point
frequencies, per-race poll counts, VoteHub raw-poll accounting, and the config used.

Example:
    PYTHONPATH=src uv run python scripts/package_run.py \
        --chamber senate --run-date 2026-09-27 \
        --artifact-root artifacts/senate-2026-live-adhoc-2026-09-27/runs/senate-2026-midterm-live \
        --data-dir data/senate-live-adhoc-2026-09-27 \
        --out runs/adhoc/2026-09-27 \
        --command-file runs/adhoc/2026-09-27/logs/commands.sh
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]

COPY_FILES = [
    "control_forecasts.parquet",
    "methodology_snapshot.md",
    "performance.json",
    "plot_manifest.json",
    "posterior_diagnostics.json",
    "race_forecasts.parquet",
    "reproducibility_fingerprint.json",
    "reward_card.json",
    "source_manifest.parquet",
    "stability_metrics.json",
    # extra (not in the weekly folder) so compare_runs.py is self-contained:
    "seat_posterior.parquet",
    "forecast_draws.parquet",
    "fundamentals_prior.parquet",
    "race_catalog.parquet",
]
PLOTS = [
    "seat_count_histogram.png",
    "race_vote_share_kde.png",
    "tipping_point_bars.png",
    "race_probability_bars.png",
    "control_projection.png",
]
CLOSEST_N = {"senate": 12, "house": 15}
CLOSEST_POLL_N = {"senate": 8, "house": 10}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _r(x: float | None, nd: int = 4) -> float | None:
    return None if x is None else round(float(x), nd)


def poll_stem(poll_id: str) -> str:
    for suffix in ("-DEM", "-REP", "-IND"):
        if poll_id.endswith(suffix):
            return poll_id[: -len(suffix)]
    return poll_id


def seat_distribution(seat_posterior: pl.DataFrame, party: str) -> dict[str, Any]:
    s = (
        seat_posterior.filter(pl.col("party") == party)
        .sort("draw_id")["seat_count_total"]
        .to_numpy()
        .astype(float)
    )
    q = {
        f"p{int(p * 100):02d}": float(np.quantile(s, p))
        for p in (0.05, 0.10, 0.25, 0.5, 0.75, 0.9, 0.95)
    }
    return {
        "draws": int(s.size),
        "mean": _r(s.mean(), 3),
        "sd": _r(s.std(ddof=1), 3),
        "min": float(s.min()),
        "max": float(s.max()),
        **q,
        "width_p10_p90": q["p90"] - q["p10"],
        "width_p05_p95": q["p95"] - q["p05"],
    }


def anti_rep_options(rf: pl.DataFrame) -> pl.DataFrame:
    """One row per race: the DEM option, else IND (races where the Democrat withdrew,
    e.g. NE Osborn), else another non-REP option. Races with no such option (CA-40,
    REP v REP) are left out."""
    rank = (
        pl.when(pl.col("party") == "DEM")
        .then(0)
        .when(pl.col("party") == "IND")
        .then(1)
        .when(pl.col("party") != "REP")
        .then(2)
        .otherwise(None)
    )
    return (
        rf.with_columns(rank.alias("_rank"))
        .filter(pl.col("_rank").is_not_null() & pl.col("winner_probability").is_not_null())
        .sort(["race_id", "_rank", "winner_probability"], descending=[False, False, True])
        .unique("race_id", keep="first", maintain_order=True)
        .drop("_rank")
    )


def tipping_rates(
    control_row: dict[str, Any],
    forecast_draws: pl.DataFrame | None = None,
    seat_posterior: pl.DataFrame | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """(metric, [{"race_id", "rate"}]) for one control row, best first (audit P1-2).

    Runs since P1-2 carry the true tipping point (``tipping_point_probability``). Older runs
    only have the legacy "pivotal"/decisive metric, recomputed for every race when draws are
    available. The metric name says which one it is.
    """
    raw = control_row.get("tipping_point_probability")
    if raw:
        items = json.loads(raw)
        return "tipping_point", [
            {"race_id": x["race_id"], "rate": float(x["tipping_point_probability"])} for x in items
        ]
    threshold = int(control_row["control_threshold"])
    party = str(control_row.get("party") or "DEM")
    if forecast_draws is not None and seat_posterior is not None:
        items = full_pivotal_rates(forecast_draws, seat_posterior, party, threshold)
    else:
        items = json.loads(control_row.get("pivotal_rates") or "[]")
    return "decisive_legacy", [
        {"race_id": x["race_id"], "rate": float(x["pivotal_rate"])} for x in items
    ]


def full_pivotal_rates(
    forecast_draws: pl.DataFrame, seat_posterior: pl.DataFrame, party: str, threshold: int
) -> list[dict[str, Any]]:
    """Legacy decisive ("pivotal") rate for every race: P(this race's flip changes control).

    Not the tipping point; see ``tipping_rates``. The pipeline only persists the top 3."""
    counts = (
        seat_posterior.filter(pl.col("party") == party)
        .sort("draw_id")
        .select(["draw_id", "seat_count_total"])
    )
    d = forecast_draws.filter(pl.col("party") == party).select(["draw_id", "race_id", "winner"])
    d = d.join(counts, on="draw_id", how="left")
    n = counts.height
    piv = (
        d.with_columns(
            (
                (
                    pl.col("winner")
                    & (pl.col("seat_count_total") >= threshold)
                    & (pl.col("seat_count_total") - 1 < threshold)
                )
                | (
                    ~pl.col("winner")
                    & (pl.col("seat_count_total") < threshold)
                    & (pl.col("seat_count_total") + 1 >= threshold)
                )
            ).alias("pivotal")
        )
        .group_by("race_id")
        .agg((pl.col("pivotal").sum() / n).alias("pivotal_rate"))
        .sort(["pivotal_rate", "race_id"], descending=[True, False])
    )
    return [
        {"race_id": r["race_id"], "pivotal_rate": round(float(r["pivotal_rate"]), 4)}
        for r in piv.iter_rows(named=True)
    ]


OUTCOME_CLASSES = {
    "polls_used": "used",
    "skipped_primary": "primary",
    "skipped_unknown_race": "race_unresolved",
    "dropped_unmapped": "unmapped_candidate",
    "dropped_same_party": "same_party",
    "dropped_missing_side": "missing_side",
    "dropped_non_nominee": "non_nominee",
    "dropped_duplicate": "duplicate",
    "skipped_same_party_race": "same_party_race",
}


def classify_raw_votehub(
    raw_path: Path,
    parser_args: dict[str, Any],
    curated_stems: set[str],
    modeled: set[str],
    parser_version: str = "votehub-race-polls-v2",
) -> dict[str, Any]:
    """Account for every raw VoteHub poll by re-running the production parser on it.

    Classes: used, primary, race_unresolved, unmapped_candidate (v1 only), same_party,
    missing_side, non_nominee, duplicate, same_party_race, parsed_not_modeled, other.
    """
    from civic_signal.normalize.builder import US_STATE_ABBREVIATIONS, US_STATE_LOOKUP
    from civic_signal.normalize.votehub import parse_votehub_polls

    payload = json.loads(raw_path.read_text(encoding="utf-8"))
    frame, accounting = parse_votehub_polls(
        json.dumps(payload),
        parser_args,
        root=ROOT,
        state_lookup=US_STATE_LOOKUP,
        state_abbreviations=set(US_STATE_ABBREVIATIONS.values()),
        parser_version=parser_version,
    )
    race_by_poll = {
        poll_stem(str(pid)): str(race)
        for pid, race in zip(frame["poll_id"], frame["race_id"], strict=True)
    }
    classes = ["used", *sorted(set(OUTCOME_CLASSES.values()) - {"used"})]
    out: dict[str, list] = {k: [] for k in [*classes, "parsed_not_modeled", "other"]}
    for pid, outcome in accounting.outcomes.items():
        cls = OUTCOME_CLASSES.get(outcome, "other")
        if cls == "used":
            race = race_by_poll.get(pid)
            if race is not None and race not in modeled:
                cls = "parsed_not_modeled"
            elif pid not in curated_stems:
                cls = "other"
        out[cls].append(pid)
    unmapped = dict(accounting.unmapped_names.most_common())
    return {
        "raw_poll_count": len(payload),
        "parser_version": parser_version,
        "counts": {k: len(v) for k, v in out.items()},
        "parser_accounting": accounting.to_dict(),
        "ids": out,
        "unmapped_candidate_name_count": len(unmapped),
        "unmapped_candidate_names": unmapped,
    }


def source_fetch_status(manifest: pl.DataFrame) -> list[dict[str, Any]]:
    """fetch_status / fetched_at per source (audit P1-7); older manifests lack them."""
    rows = []
    for r in manifest.iter_rows(named=True):
        rows.append(
            {
                "source_id": r.get("source_id"),
                "fetch_status": r.get("fetch_status") or r.get("status"),
                "fetched_at": r.get("fetched_at") or r.get("retrieved_at"),
                "sha256": r.get("content_hash"),
            }
        )
    return rows


def _chamber_headline(block: dict[str, Any]) -> dict[str, Any]:
    dem = block["control"]["DEM"]
    rep = block["control"].get("REP", {})
    return {
        "dem_control_prob": dem["control_probability"],
        "dem_control_prob_with_independents": dem.get("control_probability_with_independents"),
        "dem_seats_mean": dem["seat_count_mean"],
        "p10": dem["seat_count_p10"],
        "p50": dem["seat_count_p50"],
        "p90": dem["seat_count_p90"],
        "rep_control_prob": rep.get("control_probability"),
        "rep_seats_mean": rep.get("seat_count_mean"),
    }


def _delta(old: float | None, new: float | None) -> dict[str, Any]:
    d = None if old is None or new is None else round(new - old, 4)
    return {"old": old, "new": new, "delta": d}


def write_top_summary(
    run_dir: Path, run_kind: str | None = None, prior: Path | None = None
) -> Path:
    """Top-level ``<run>/summary.json`` combining the chamber summaries (audit P2-5).

    Same headline schema as the weekly digest; adds a week-over-week block when ``prior``
    (an earlier packaged run folder) is given.
    """
    run_dir = Path(run_dir)
    chambers = {}
    for ch in ("senate", "house"):
        path = run_dir / ch / "summary.json"
        if path.exists():
            chambers[ch] = json.loads(path.read_text())
    if not chambers:
        raise FileNotFoundError(f"no chamber summaries under {run_dir}")
    any_block = next(iter(chambers.values()))
    payload: dict[str, Any] = {
        "digest_date": any_block.get("run_date"),
        "as_of": any_block.get("as_of"),
        "run_kind": run_kind or run_dir.parent.name,
        "generated_at": datetime.now(UTC).isoformat(),
        "inference": {
            "engine": any_block["inference"].get("engine"),
            "backend": any_block["inference"].get("bayesian_backend"),
            "sims": {ch: b["inference"].get("simulation_count") for ch, b in chambers.items()},
            "as_of": any_block.get("as_of"),
        },
        "source_fetch_status": {ch: b.get("source_fetch_status", []) for ch, b in chambers.items()},
    }
    for ch, block in chambers.items():
        payload[ch] = _chamber_headline(block)
    if prior is not None:
        wow: dict[str, Any] = {}
        for ch in chambers:
            prior_path = Path(prior) / ch / "summary.json"
            if not prior_path.exists():
                continue
            old = _chamber_headline(json.loads(prior_path.read_text()))
            new = payload[ch]
            wow[ch] = {
                "vs": Path(prior).name,
                "dem_control_prob": _delta(old["dem_control_prob"], new["dem_control_prob"]),
                "dem_seats": {
                    k: _delta(old[key], new[key])
                    for k, key in (
                        ("mean", "dem_seats_mean"),
                        ("p10", "p10"),
                        ("p50", "p50"),
                        ("p90", "p90"),
                    )
                },
            }
        payload["compared_against"] = str(prior)
        payload["week_over_week"] = wow
    out = run_dir / "summary.json"
    out.write_text(json.dumps(payload, indent=2, default=str))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--chamber", required=True, choices=["senate", "house"])
    ap.add_argument("--run-date", required=True)
    ap.add_argument(
        "--as-of",
        default=None,
        help="Forecast as-of date; defaults to the run's posterior_diagnostics as_of, else today.",
    )
    ap.add_argument("--artifact-root", required=True, type=Path)
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--command-file", type=Path, default=None)
    ap.add_argument("--sources-config", default=None)
    ap.add_argument("--scenario", default=None)
    ap.add_argument(
        "--run-kind", default=None, help="weekly/adhoc; defaults to the parent folder name."
    )
    ap.add_argument(
        "--prior",
        type=Path,
        default=None,
        help="Previous packaged run folder for the week-over-week block of summary.json.",
    )
    args = ap.parse_args()

    ch = args.chamber
    art = args.artifact_root.resolve()
    out = (args.out / ch).resolve()
    (out / "plots").mkdir(parents=True, exist_ok=True)
    for f in COPY_FILES:
        if (art / f).exists():
            shutil.copy2(art / f, out / f)
    for p in PLOTS:
        if (art / "plots" / p).exists():
            shutil.copy2(art / "plots" / p, out / "plots" / p)
    curated = args.data_dir.resolve() / "curated"
    shutil.copy2(curated / "polls.parquet", out / "polls.parquet")

    rf = pl.read_parquet(art / "race_forecasts.parquet")
    cf = pl.read_parquet(art / "control_forecasts.parquet")
    sp = pl.read_parquet(art / "seat_posterior.parquet")
    fd = pl.read_parquet(art / "forecast_draws.parquet")
    catalog = pl.read_parquet(art / "race_catalog.parquet")
    manifest = pl.read_parquet(art / "source_manifest.parquet")
    diag = json.loads((art / "posterior_diagnostics.json").read_text())
    if args.as_of is None:
        args.as_of = str(diag.get("as_of") or datetime.now(UTC).date().isoformat())
    perf = json.loads((art / "performance.json").read_text())
    reward_text = (art / "reward_card.json").read_text()
    pdraws = pl.read_parquet(
        art / "posterior_draws.parquet", columns=["race_id", "diagnostic_only"]
    )
    prior_only = pdraws.group_by("race_id").agg(
        pl.col("diagnostic_only").fill_null(False).all().alias("po")
    )
    model_polled = set(prior_only.filter(~pl.col("po"))["race_id"].to_list())
    polls = pl.read_parquet(curated / "polls.parquet")
    modeled = set(catalog["race_id"].to_list())

    # control_summary.csv / dem_race_probabilities.csv (same as weekly folder)
    csv_cols = [
        "control_body",
        "party",
        "control_threshold",
        "modeled_seats",
        "holdover_seats",
        "seat_count_mean",
        "seat_count_p10",
        "seat_count_p50",
        "seat_count_p90",
        "majority_probability",
        "control_probability",
    ]
    cf.select(csv_cols).write_csv(out / "control_summary.csv")
    dem = anti_rep_options(rf).sort("winner_probability", descending=True)
    dem.select(
        [
            "race_id",
            "state",
            "party",
            "winner_probability",
            "vote_share_mean",
            "tier",
            "top_drivers",
        ]
    ).write_csv(out / "dem_race_probabilities.csv")

    # control block
    control: dict[str, Any] = {}
    thresholds: dict[str, int] = {}
    for row in cf.iter_rows(named=True):
        thresholds[row["party"]] = int(row["control_threshold"])
        control[row["party"]] = {
            "control_probability": row["control_probability"],
            "majority_probability": row["majority_probability"],
            "seat_count_mean": round(row["seat_count_mean"], 3),
            "seat_count_p10": row["seat_count_p10"],
            "seat_count_p50": row["seat_count_p50"],
            "seat_count_p90": row["seat_count_p90"],
            "control_threshold": row["control_threshold"],
            "modeled_seats": row["modeled_seats"],
            "holdover_seats": row["holdover_seats"],
            "control_probability_with_independents": row.get(
                "control_probability_with_independents"
            ),
            "control_probability_without_independents": row.get(
                "control_probability_without_independents"
            ),
            "tipping_point_races": json.loads(row["tipping_point_races"]),
            "tipping_point_probability": json.loads(row.get("tipping_point_probability") or "[]"),
            "decisive_probability": json.loads(
                row.get("decisive_probability") or row.get("pivotal_rates") or "[]"
            ),
        }

    # polls per race (2026 VoteHub rows in curated polls, modeled races only)
    p26 = polls.filter((pl.col("cycle") == 2026) & pl.col("source_id").str.starts_with("votehub"))
    p26 = p26.with_columns(
        pl.col("poll_id").map_elements(poll_stem, return_dtype=pl.Utf8).alias("poll_stem")
    )
    per_race = p26.group_by("race_id").agg(
        pl.len().alias("poll_rows"),
        pl.col("poll_stem").n_unique().alias("poll_count_distinct"),
        pl.col("pollster").n_unique().alias("pollster_count"),
        pl.col("end_date").max().alias("latest_poll_end_date"),
    )
    pr_map = {r["race_id"]: r for r in per_race.iter_rows(named=True)}

    rep = rf.filter(pl.col("party") == "REP").select(
        ["race_id", pl.col("winner_probability").alias("rep_p")]
    )
    rep_map = dict(zip(rep["race_id"], rep["rep_p"], strict=False))
    piv_dem = full_pivotal_rates(fd, sp, "DEM", thresholds["DEM"])
    piv_map = {r["race_id"]: r["pivotal_rate"] for r in piv_dem}
    dem_control_row = cf.filter(pl.col("party") == "DEM").row(0, named=True)
    tip_metric, tip_dem = tipping_rates(dem_control_row, fd, sp)
    tip_map = {r["race_id"]: r["rate"] for r in tip_dem}
    races = []
    for r in dem.sort("race_id").iter_rows(named=True):
        info = pr_map.get(r["race_id"], {})
        races.append(
            {
                "race_id": r["race_id"],
                "state": r["state"],
                # anti-REP option: DEM, else IND (NE/ID/SD), else another non-REP party
                "dem_option_party": r["party"],
                "dem_win_prob": _r(r["winner_probability"], 4),
                "rep_win_prob": _r(rep_map.get(r["race_id"]), 4),
                "dem_win_prob_raw": _r(r.get("raw_winner_probability"), 4),
                "dem_vote_share_mean": _r(r["vote_share_mean"], 4),
                "dem_vote_share_p10": _r(r["vote_share_p10"], 4),
                "dem_vote_share_p90": _r(r["vote_share_p90"], 4),
                "tier": r["tier"],
                "top_drivers": r["top_drivers"],
                "poll_count": int(info.get("poll_rows", 0)),
                "poll_count_distinct": int(info.get("poll_count_distinct", 0)),
                "pollster_count": int(info.get("pollster_count", 0)),
                "latest_poll_end_date": str(info["latest_poll_end_date"])
                if info.get("latest_poll_end_date")
                else None,
                "polled": bool(info.get("poll_rows", 0)),
                "model_polled": r["race_id"] in model_polled,
                "dem_tipping_point_probability": tip_map.get(r["race_id"], 0.0),
                "dem_decisive_probability": piv_map.get(r["race_id"], 0.0),
            }
        )
    closest = sorted(races, key=lambda x: (abs(x["dem_win_prob"] - 0.5), x["race_id"]))[
        : CLOSEST_N[ch]
    ]
    closest_block = [
        {
            k: x[k]
            for k in (
                "race_id",
                "state",
                "dem_win_prob",
                "rep_win_prob",
                "dem_vote_share_mean",
                "tier",
                "top_drivers",
            )
        }
        for x in closest
    ]
    closest_polls = sorted(
        [
            {
                "race_id": x["race_id"],
                "state": x["state"],
                "poll_count": x["poll_count"],
                "pollster_count": x["pollster_count"],
                "tier": x["tier"],
            }
            for x in closest[: CLOSEST_POLL_N[ch]]
        ],
        key=lambda x: x["race_id"],
    )

    vh = manifest.filter(pl.col("source_id").str.starts_with("votehub"))
    vh_sources = [
        {
            k: row[k]
            for k in (
                "source_id",
                "table",
                "url",
                "status",
                "retrieved_at",
                "raw_path",
                "parser_version",
                "license",
                "content_hash",
            )
        }
        for row in vh.iter_rows(named=True)
    ]
    raw_accounting = {}
    for row in vh.iter_rows(named=True):
        stems = set(p26["poll_stem"].to_list())
        raw_accounting[row["source_id"]] = classify_raw_votehub(
            Path(row["raw_path"]),
            json.loads(row["parser_args"]),
            stems,
            modeled,
            parser_version=str(row["parser_version"]),
        )
    poll_diag = {
        "poll_count": diag.get("poll_count"),
        "polling_observed_race_option_count": diag.get("polling_observed_race_option_count"),
        "prior_only_race_option_count": diag.get("prior_only_race_option_count"),
        "race_option_count": diag.get("race_option_count"),
    }
    if ch == "senate":
        poll_diag["senate_classes"] = (
            diag.get("office_methodology", {}).get("senate_joint", {}).get("senate_classes")
        )
    else:
        polled_d = sum(1 for x in races if x["polled"])
        poll_diag["unpolled_district_count"] = len(races) - polled_d
        poll_diag["district_count"] = len(races)
    poll_diag.update(
        {
            "votehub_source_count": len(vh_sources),
            "votehub_sources": vh_sources,
            "curated_files": sorted(p.name for p in curated.glob("*.parquet")),
            "source_ids": manifest["source_id"].to_list(),
            "unmapped_mentions_in_reward_card": [
                ln.strip() for ln in reward_text.splitlines() if "unmapped" in ln.lower()
            ],
            "curated_votehub_rows_all": p26.height,
            "curated_votehub_rows_modeled_races": int(
                p26.filter(pl.col("race_id").is_in(list(modeled))).height
            ),
            "curated_votehub_distinct_polls_modeled_races": int(
                p26.filter(pl.col("race_id").is_in(list(modeled)))["poll_stem"].n_unique()
            ),
            "polled_race_count": sum(1 for x in races if x["polled"]),
            "model_polled_race_count": len(model_polled),
            "votehub_raw_accounting": {
                k: {kk: vv for kk, vv in v.items() if kk != "ids"}
                for k, v in raw_accounting.items()
            },
        }
    )
    (out / "votehub_raw_accounting.json").write_text(json.dumps(raw_accounting, indent=2))

    plots = {
        p.removesuffix(".png"): str(out / "plots" / p)
        for p in PLOTS[:4]
        if (out / "plots" / p).exists()
    }
    git_head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True
    ).stdout.strip()
    git_dirty = subprocess.run(
        ["git", "status", "--short"], cwd=ROOT, capture_output=True, text=True
    ).stdout.splitlines()
    summary: dict[str, Any] = {
        "run_date": args.run_date,
        "chamber": ch,
        "as_of": args.as_of,
        "generated_at": datetime.now(UTC).isoformat(),
        "inference": {
            "engine": "bayes",
            "bayesian_backend": "analytic"
            if "analytic" in str(diag.get("engine"))
            else diag.get("engine"),
            "posterior_engine": diag.get("engine"),
            "run_id": art.name,
            "artifact_root": str(art),
            "simulation_count": perf.get("simulation_count"),
            "posterior_draw_count": diag.get("draw_count"),
            "model_config_hash": diag.get("model_config_hash"),
            "source_manifest_hash": diag.get("source_manifest_hash"),
        },
        "config_used": {
            "command_file": str(args.command_file.resolve()) if args.command_file else None,
            "command": args.command_file.read_text() if args.command_file else None,
            "sources_config": args.sources_config,
            "scenario": args.scenario,
            "data_dir": str(args.data_dir.resolve()),
            "artifacts_root": str(art),
            "model_yaml_sha256": _sha(ROOT / "configs" / "model.yaml"),
            "scenarios_yaml_sha256": _sha(ROOT / "configs" / "scenarios.yaml"),
            "sources_config_sha256": _sha(ROOT / "configs" / args.sources_config)
            if args.sources_config
            else None,
            "simulation_count": perf.get("simulation_count"),
            "git_head": git_head,
            "git_dirty_files": git_dirty,
        },
        "control": control,
        "seat_distribution": {p: seat_distribution(sp, p) for p in ("DEM", "REP")},
        "closest_races": closest_block,
        "tipping_point_frequencies": {
            "party": "DEM",
            "threshold": thresholds["DEM"],
            "metric": tip_metric,
            "races": [{"race_id": x["race_id"], "rate": round(x["rate"], 4)} for x in tip_dem],
            "decisive_probability": piv_dem,
        },
        "poll_diagnostics": poll_diag,
        "plots": plots,
        "closest_race_poll_counts": closest_polls,
        "races": races,
    }
    if ch == "senate":
        states = set(catalog["state"].to_list())
        summary["senate_panel"] = {
            "senate_state_count": len(states),
            "special_elections_modeled": sorted(
                str(r) for r in catalog["race_id"].to_list() if "SPECIAL" in str(r).upper()
            ),
        }
    summary["source_fetch_status"] = source_fetch_status(manifest)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"wrote {out / 'summary.json'}")
    top = write_top_summary(args.out, run_kind=args.run_kind, prior=args.prior)
    print(f"wrote {top}")


if __name__ == "__main__":
    main()
