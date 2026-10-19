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
from types import SimpleNamespace
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
    for suffix in ("-DEM", "-REP"):
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


def full_pivotal_rates(
    forecast_draws: pl.DataFrame, seat_posterior: pl.DataFrame, party: str, threshold: int
) -> list[dict[str, Any]]:
    """Recompute pivotal rates for every race (the pipeline only persists the top 3)."""
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


def classify_raw_votehub(
    raw_path: Path, parser_args: dict[str, Any], curated_stems: set[str], modeled: set[str]
) -> dict[str, Any]:
    """Account for every raw VoteHub poll (used, primary, unresolved, unmapped, not modeled)."""
    from civic_signal.normalize.builder import CuratedDataBuilder as B

    office = str(parser_args.get("office", "")).lower()
    fake = SimpleNamespace(
        context=SimpleNamespace(root=ROOT), _votehub_normalize_name=B._votehub_normalize_name
    )
    party_map = B._load_votehub_party_map(fake, str(parser_args["party_map_path"]), office)
    parties = [str(p).upper() for p in parser_args.get("parties", ["DEM", "REP"])]
    payload = json.loads(raw_path.read_text(encoding="utf-8"))
    out: dict[str, list] = {
        k: []
        for k in (
            "used",
            "primary",
            "race_unresolved",
            "unmapped_candidate",
            "parsed_not_modeled",
            "other",
        )
    }
    unmapped_names: dict[str, int] = {}
    for poll in payload:
        pid = f"votehub-{poll.get('id')}"
        subject = str(poll.get("subject") or "")
        if str(poll.get("poll_type") or "") != parser_args.get("poll_type"):
            out["other"].append(pid)
            continue
        if B._votehub_is_primary_subject(subject):
            out["primary"].append(pid)
            continue
        race_id = B._votehub_race_id(
            subject=subject,
            seat_name=poll.get("seat_name"),
            office=office,
            cycle=int(parser_args.get("cycle", 2026)),
            house_format=str(parser_args.get("house_race_id_format", "hyphenated")),
        )
        if race_id is None:
            out["race_unresolved"].append(pid)
            continue
        missing = []
        for a in poll.get("answers") or []:
            ch = str((a or {}).get("choice") or "")
            party = B._votehub_lookup_party(party_map, ch, subject)
            if party is None or party not in parties:
                missing.append(ch)
        if missing:
            out["unmapped_candidate"].append(pid)
            for m in missing:
                unmapped_names[m] = unmapped_names.get(m, 0) + 1
            continue
        if pid in curated_stems and race_id in modeled:
            out["used"].append(pid)
        elif race_id not in modeled:
            out["parsed_not_modeled"].append(pid)
        else:
            out["other"].append(pid)
    return {
        "raw_poll_count": len(payload),
        "counts": {k: len(v) for k, v in out.items()},
        "ids": out,
        "unmapped_candidate_name_count": len(unmapped_names),
        "unmapped_candidate_names": dict(
            sorted(unmapped_names.items(), key=lambda kv: (-kv[1], kv[0]))
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--chamber", required=True, choices=["senate", "house"])
    ap.add_argument("--run-date", required=True)
    ap.add_argument("--as-of", default="2026-11-02")
    ap.add_argument("--artifact-root", required=True, type=Path)
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--command-file", type=Path, default=None)
    ap.add_argument("--sources-config", default=None)
    ap.add_argument("--scenario", default=None)
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
    dem = rf.filter(pl.col("party") == "DEM").sort("winner_probability", descending=True)
    dem.select(
        ["race_id", "state", "winner_probability", "vote_share_mean", "tier", "top_drivers"]
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
            "tipping_point_races": json.loads(row["tipping_point_races"]),
            "pivotal_rates": json.loads(row["pivotal_rates"]),
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
    races = []
    for r in dem.sort("race_id").iter_rows(named=True):
        info = pr_map.get(r["race_id"], {})
        races.append(
            {
                "race_id": r["race_id"],
                "state": r["state"],
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
                "dem_pivotal_rate": piv_map.get(r["race_id"], 0.0),
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
            Path(row["raw_path"]), json.loads(row["parser_args"]), stems, modeled
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
            "races": piv_dem,
        },
        "poll_diagnostics": poll_diag,
        "plots": plots,
        "closest_race_poll_counts": closest_polls,
        "races": races,
    }
    if ch == "senate":
        states = set(catalog["state"].to_list())
        summary["class_ii_panel"] = {
            "fl_oh_present": {"FL": "FL" in states, "OH": "OH" in states},
            "missing_fl_oh": [s for s in ("FL", "OH") if s not in states],
            "senate_state_count": len(states),
            "note": (
                "FL and OH special/Class III 2026 seats are absent from the senate_2026_state "
                "race_catalog; their VoteHub polls parse but are not modeled."
            ),
        }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"wrote {out / 'summary.json'}")


if __name__ == "__main__":
    main()
