"""Compare two civic-signal Senate/House run folders and emit JSON + Markdown (+ optional dumbbell plot).

A "run folder" is either a weekly digest folder (runs/weekly/<date>/) or an ad hoc folder
(runs/adhoc/<date>/) containing ``senate/`` and/or ``house/`` subfolders with at least
``race_forecasts.parquet``, ``control_forecasts.parquet``, ``performance.json``,
``posterior_diagnostics.json`` and ``source_manifest.parquet``. Artifacts not copied into
the folder (``seat_posterior.parquet``, ``forecast_draws.parquet``,
``fundamentals_prior.parquet``) are looked up in the run's artifact root as recorded in
``<run>/summary.json`` (``inference.artifact_roots``) or ``<run>/<chamber>/summary.json``
(``inference.artifact_root``). Curated poll rows are read from ``<run>/<chamber>/polls.parquet``
or from ``--old-inputs/--new-inputs <dir>/<chamber>/polls.parquet``.

Usage:
    PYTHONPATH=src uv run python scripts/compare_runs.py \
        runs/weekly/2026-09-21 runs/adhoc/2026-09-27 \
        --old-inputs runs/adhoc/2026-09-27/baseline_inputs \
        --out runs/adhoc/2026-09-27/comparison_vs_2026-09-21 \
        --plot runs/adhoc/2026-09-27/movers_dumbbell.png \
        --noise-runs runs/adhoc/2026-09-27/noise_check/seed1 runs/adhoc/2026-09-27/noise_check/seed2

Outputs <out>.json and <out>.md. Read-only with respect to both run folders.
"""

# ruff: noqa: E501, RUF001
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from package_run import classify_raw_votehub, poll_stem, tipping_rates

TOP_N = {"senate": 8, "house": 12}
BAND = (0.35, 0.65)
CURRENT_MAJORITY = {"senate": "REP", "house": "REP"}  # 119th Congress majority party


# --------------------------------------------------------------------------- loading
def _artifact_root(run: Path, chamber: str) -> Path | None:
    top = run / "summary.json"
    if top.exists():
        roots = json.loads(top.read_text()).get("inference", {}).get("artifact_roots", {})
        if roots.get(chamber):
            return Path(roots[chamber])
    sub = run / chamber / "summary.json"
    if sub.exists():
        root = json.loads(sub.read_text()).get("inference", {}).get("artifact_root")
        if root:
            return Path(root)
    return None


def _find(run: Path, chamber: str, name: str) -> Path | None:
    p = run / chamber / name
    if p.exists():
        return p
    root = _artifact_root(run, chamber)
    if root and (root / name).exists():
        return root / name
    return None


def load_run(run: Path, chamber: str, inputs: Path | None) -> dict[str, Any]:
    d = run / chamber
    rf = pl.read_parquet(d / "race_forecasts.parquet")
    cf = pl.read_parquet(d / "control_forecasts.parquet")
    perf = json.loads((d / "performance.json").read_text())
    diag = json.loads((d / "posterior_diagnostics.json").read_text())
    manifest = pl.read_parquet(d / "source_manifest.parquet")
    out: dict[str, Any] = {
        "folder": str(run),
        "rf": rf,
        "cf": cf,
        "perf": perf,
        "diag": diag,
        "manifest": manifest,
    }
    out["n_sims"] = int(perf.get("simulation_count") or diag.get("draw_count") or 1000)
    for key, name in [
        ("sp", "seat_posterior.parquet"),
        ("fd", "forecast_draws.parquet"),
        ("fp", "fundamentals_prior.parquet"),
        ("cat", "race_catalog.parquet"),
        ("pd", "posterior_draws.parquet"),
    ]:
        p = _find(run, chamber, name)
        out[key] = pl.read_parquet(p) if p else None
        out[key + "_path"] = str(p) if p else None
    poll_path = d / "polls.parquet"
    if not poll_path.exists() and inputs is not None:
        poll_path = inputs / chamber / "polls.parquet"
    out["polls_path"] = str(poll_path) if poll_path.exists() else None
    if poll_path.exists():
        p = pl.read_parquet(poll_path)
        p = p.filter((pl.col("cycle") == 2026) & pl.col("source_id").str.starts_with("votehub"))
        p = p.with_columns(
            pl.col("poll_id").map_elements(poll_stem, return_dtype=pl.Utf8).alias("poll_stem")
        )
        modeled = set(rf["race_id"].unique().to_list())
        out["polls_all"] = p
        out["polls"] = p.filter(pl.col("race_id").is_in(list(modeled)))
    else:
        out["polls_all"] = out["polls"] = None
    return out


# --------------------------------------------------------------------------- helpers
def mc_se(p: float, n: int) -> float:
    return math.sqrt(max(p * (1 - p), 0.0) / n)


def se_diff(p1: float, n1: int, p2: float, n2: int) -> float:
    # Conservative: assumes independent Monte Carlo draws. Floor at p=0.5-equivalent/ n for p in {0,1}.
    return math.sqrt(max(p1 * (1 - p1), 1e-12) / n1 + max(p2 * (1 - p2), 1e-12) / n2)


def seat_stats(sp: pl.DataFrame | None, party: str) -> dict[str, Any] | None:
    if sp is None:
        return None
    s = (
        sp.filter(pl.col("party") == party)
        .sort("draw_id")["seat_count_total"]
        .to_numpy()
        .astype(float)
    )
    q = {
        f"p{int(k * 100):02d}": float(np.quantile(s, k))
        for k in (0.05, 0.10, 0.25, 0.5, 0.75, 0.9, 0.95)
    }
    return {
        "mean": float(s.mean()),
        "sd": float(s.std(ddof=1)),
        **q,
        "width_p10_p90": q["p90"] - q["p10"],
        "width_p05_p95": q["p95"] - q["p05"],
        "n": int(s.size),
        "se_mean": float(s.std(ddof=1) / math.sqrt(s.size)),
    }


def race_table(r: dict[str, Any]) -> pl.DataFrame:
    rf = r["rf"]
    dem = rf.filter(pl.col("party") == "DEM").select(
        "race_id",
        "state",
        "tier",
        pl.col("winner_probability").alias("p"),
        pl.col("vote_share_mean").alias("vs"),
        pl.col("vote_share_p10").alias("vs_p10"),
        pl.col("vote_share_p90").alias("vs_p90"),
    )
    inc = (
        rf.filter(pl.col("incumbent").fill_null(False))
        .select("race_id", pl.col("party").alias("incumbent_party"))
        .unique("race_id")
    )
    dem = dem.join(inc, on="race_id", how="left")
    if r["polls"] is not None:
        pc = (
            r["polls"]
            .group_by("race_id")
            .agg(
                pl.len().alias("poll_rows"), pl.col("poll_stem").n_unique().alias("polls_distinct")
            )
        )
        dem = dem.join(pc, on="race_id", how="left").with_columns(
            pl.col("poll_rows").fill_null(0), pl.col("polls_distinct").fill_null(0)
        )
    return dem


def flip_stats(r: dict[str, Any]) -> dict[str, Any] | None:
    """Seats changing party vs the incumbent-flagged option in race_forecasts, per simulation."""
    fd, rf = r["fd"], r["rf"]
    if fd is None:
        return None
    inc = (
        rf.filter(pl.col("incumbent").fill_null(False))
        .select("race_id", pl.col("party").alias("inc_party"))
        .unique("race_id")
    )
    w = (
        fd.filter(pl.col("winner"))
        .select("draw_id", "race_id", "party")
        .join(inc, on="race_id", how="inner")
    )
    per = w.group_by("draw_id").agg(
        (pl.col("party") != pl.col("inc_party")).sum().alias("flips"),
        ((pl.col("party") == "DEM") & (pl.col("inc_party") == "REP")).sum().alias("r_to_d"),
        ((pl.col("party") == "REP") & (pl.col("inc_party") == "DEM")).sum().alias("d_to_r"),
    )
    f = per["flips"].to_numpy().astype(float)
    return {
        "races_with_incumbent_flag": inc.height,
        "mean_seat_flips_per_sim": float(f.mean()),
        "p10": float(np.quantile(f, 0.1)),
        "p50": float(np.quantile(f, 0.5)),
        "p90": float(np.quantile(f, 0.9)),
        "mean_R_to_D": float(per["r_to_d"].mean()),
        "mean_D_to_R": float(per["d_to_r"].mean()),
    }


def poll_record(
    row: dict[str, Any], dem_pct: float | None, rep_pct: float | None
) -> dict[str, Any]:
    margin = None if dem_pct is None or rep_pct is None else round(dem_pct - rep_pct, 1)
    return {
        "poll_id": row["poll_stem"],
        "pollster": row["pollster"],
        "field_dates": f"{row['start_date']} to {row['end_date']}",
        "sample_size": None if row["sample_size"] is None else int(row["sample_size"]),
        "population": row["population"],
        "sponsor_class": row["sponsor_class"],
        "dem_pct": dem_pct,
        "rep_pct": rep_pct,
        "dem_minus_rep": margin,
    }


def polls_by_stem(p: pl.DataFrame) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in p.sort(["race_id", "end_date", "poll_stem"]).iter_rows(named=True):
        rec = out.setdefault(
            row["poll_stem"], {"row": row, "DEM": None, "REP": None, "race_id": row["race_id"]}
        )
        party = (
            "DEM"
            if row["option_id"].endswith("-D")
            else "REP"
            if row["option_id"].endswith("-R")
            else None
        )
        if party:
            rec[party] = row["pct"]
    return out


def posterior_shift(
    old_pd: pl.DataFrame | None, new_pd: pl.DataFrame | None
) -> pl.DataFrame | None:
    """Per-race z-score of the change in posterior mean DEM latent logit (resampling SE)."""
    if old_pd is None or new_pd is None:
        return None

    def agg(d: pl.DataFrame) -> pl.DataFrame:
        # Audit P2-5: a race is polled if ANY option has poll-bearing draws, not only the
        # "-D" option (which mislabelled AL and AK-01 and dropped IND-v-REP races like NE).
        status = d.group_by("race_id").agg(
            pl.col("diagnostic_only").fill_null(False).all().alias("prior_only")
        )
        side = d.with_columns(
            pl.when(pl.col("option_id").str.ends_with("-D"))
            .then(0)
            .when(pl.col("option_id").str.ends_with("-I"))
            .then(1)
            .otherwise(2)
            .alias("_rank")
        )
        best = side.group_by("race_id").agg(pl.col("_rank").min().alias("_best"))
        side = side.join(best, on="race_id").filter(pl.col("_rank") == pl.col("_best"))
        first = side.group_by("race_id").agg(pl.col("option_id").min().alias("_opt"))
        side = side.join(first, on="race_id").filter(pl.col("option_id") == pl.col("_opt"))
        return (
            side.group_by("race_id")
            .agg(
                pl.col("latent_logit").mean().alias("m"),
                pl.col("latent_logit").std().alias("s"),
                pl.len().alias("n"),
            )
            .join(status, on="race_id")
        )

    o, n = agg(old_pd), agg(new_pd)
    return n.join(o, on="race_id", suffix="_old").with_columns(
        (pl.col("m") - pl.col("m_old")).alias("d_logit"),
        (
            (pl.col("m") - pl.col("m_old"))
            / (pl.col("s") ** 2 / pl.col("n") + pl.col("s_old") ** 2 / pl.col("n_old")).sqrt()
        ).alias("post_z"),
    )


def bonferroni_z(n_tests: int, alpha: float = 0.05) -> float:
    from statistics import NormalDist

    return NormalDist().inv_cdf(1 - alpha / (2 * max(n_tests, 1)))


def noise_run_stats(ch: str, new: dict[str, Any], noise_runs: list[Path]) -> dict[str, Any] | None:
    """Empirical simulation noise from re-runs on identical inputs with different simulator seeds."""
    tabs, ctrl = [], []
    base = (
        new["rf"]
        .filter(pl.col("party") == "DEM")
        .select("race_id", pl.col("winner_probability").alias("p0"))
    )
    c0 = new["cf"].filter(pl.col("party") == "DEM").row(0, named=True)
    ctrl.append(
        {
            "run": "primary",
            "dem_control_prob": c0["control_probability"],
            "dem_seat_mean": round(c0["seat_count_mean"], 3),
            "tipping_top3": json.loads(c0["tipping_point_races"]),
        }
    )
    for nr in noise_runs:
        d = nr / ch
        if not (d / "race_forecasts.parquet").exists():
            continue
        rf = (
            pl.read_parquet(d / "race_forecasts.parquet")
            .filter(pl.col("party") == "DEM")
            .select("race_id", pl.col("winner_probability").alias("p1"))
        )
        tabs.append(
            base.join(rf, on="race_id").with_columns((pl.col("p1") - pl.col("p0")).alias("dp"))
        )
        c = (
            pl.read_parquet(d / "control_forecasts.parquet")
            .filter(pl.col("party") == "DEM")
            .row(0, named=True)
        )
        ctrl.append(
            {
                "run": str(nr),
                "dem_control_prob": c["control_probability"],
                "dem_seat_mean": round(c["seat_count_mean"], 3),
                "tipping_top3": json.loads(c["tipping_point_races"]),
            }
        )
    if not tabs:
        return None
    allt = pl.concat(tabs)
    comp = allt.filter(pl.col("p0").is_between(0.2, 0.8))
    cps = [c["dem_control_prob"] for c in ctrl]
    sms = [c["dem_seat_mean"] for c in ctrl]
    return {
        "runs": ctrl,
        "dem_control_prob_range": [min(cps), max(cps)],
        "dem_seat_mean_range": [min(sms), max(sms)],
        "sd_dp_all_races": round(float(allt["dp"].std()), 4),
        "mean_abs_dp_all_races": round(float(allt["dp"].abs().mean()), 4),
        "max_abs_dp": round(float(allt["dp"].abs().max()), 4),
        "sd_dp_races_0.2_0.8": round(float(comp["dp"].std()), 4) if comp.height > 1 else None,
        "note": "Seed-only reruns capture simulator noise but not posterior-draw resampling noise (posterior seeds are data-derived).",
    }


# --------------------------------------------------------------------------- compare
def compare_chamber(
    ch: str, old: dict[str, Any], new: dict[str, Any], noise_runs: list[Path] | None = None
) -> dict[str, Any]:
    n_old, n_new = old["n_sims"], new["n_sims"]
    res: dict[str, Any] = {"chamber": ch, "simulation_count": {"old": n_old, "new": n_new}}
    # ---- control
    co = old["cf"].filter(pl.col("party") == "DEM").row(0, named=True)
    cn = new["cf"].filter(pl.col("party") == "DEM").row(0, named=True)
    po, pn = co["control_probability"], cn["control_probability"]
    sed = se_diff(po, n_old, pn, n_new)
    res["control"] = {
        "dem_control_prob": {"old": po, "new": pn, "delta": round(pn - po, 4)},
        "mc_se_old": round(mc_se(po, n_old), 4),
        "mc_se_new": round(mc_se(pn, n_new), 4),
        "mc_se_delta": round(sed, 4),
        "z": round((pn - po) / sed, 2) if sed else None,
        "flip_prob_from_current_majority": {
            "current_majority": CURRENT_MAJORITY[ch],
            "old": po if CURRENT_MAJORITY[ch] == "REP" else 1 - po,
            "new": pn if CURRENT_MAJORITY[ch] == "REP" else 1 - pn,
        },
        "control_threshold_dem": co["control_threshold"],
    }
    # ---- seats
    so, sn = seat_stats(old["sp"], "DEM"), seat_stats(new["sp"], "DEM")
    seats: dict[str, Any] = {}
    for k in ["mean", "p10", "p50", "p90"]:
        ov = so[k] if so else co[f"seat_count_{k}"]
        nv = sn[k] if sn else cn[f"seat_count_{k}"]
        seats[k] = {"old": round(ov, 3), "new": round(nv, 3), "delta": round(nv - ov, 3)}
    if so and sn:
        for k in ["sd", "width_p10_p90", "width_p05_p95", "p05", "p95"]:
            seats[k] = {
                "old": round(so[k], 3),
                "new": round(sn[k], 3),
                "delta": round(sn[k] - so[k], 3),
            }
        seats["mean"]["mc_se_delta"] = round(math.sqrt(so["se_mean"] ** 2 + sn["se_mean"] ** 2), 3)
    res["dem_seats"] = seats
    fo, fn = flip_stats(old), flip_stats(new)
    res["seat_flips_vs_incumbent_party"] = {"old": fo, "new": fn}
    # ---- tipping (audit P1-2: true tipping point; legacy runs carry the decisive metric)
    metric_old, fo_p = tipping_rates(co, old["fd"], old["sp"])
    metric_new, fn_p = tipping_rates(cn, new["fd"], new["sp"])
    to = [x["race_id"] for x in fo_p[:3]]
    tn = [x["race_id"] for x in fn_p[:3]]
    om = {x["race_id"]: x["rate"] for x in fo_p}
    o10 = [x["race_id"] for x in fo_p[:10]]
    n10 = [x["race_id"] for x in fn_p[:10]]
    tip: dict[str, Any] = {
        "metric_old": metric_old,
        "metric_new": metric_new,
        "top3_old": fo_p[:3],
        "top3_new": fn_p[:3],
        "entered_top3": [x for x in tn if x not in to],
        "left_top3": [x for x in to if x not in tn],
        "top10_old": fo_p[:10],
        "top10_new": fn_p[:10],
        "entered_top10": [
            {"race_id": x["race_id"], "old": om.get(x["race_id"]), "new": x["rate"]}
            for x in fn_p[:10]
            if x["race_id"] not in o10
        ],
        "left_top10": [x for x in o10 if x not in n10],
        "rate_mc_se_at_top": round(mc_se(fn_p[0]["rate"], n_new), 4) if fn_p else None,
    }
    res["tipping"] = tip
    # ---- races
    ro, rn = race_table(old), race_table(new)
    j = rn.join(
        ro.select(
            "race_id",
            pl.col("p").alias("p_old"),
            pl.col("vs").alias("vs_old"),
            *(
                [
                    pl.col("poll_rows").alias("poll_rows_old"),
                    pl.col("polls_distinct").alias("polls_distinct_old"),
                ]
                if "poll_rows" in ro.columns
                else []
            ),
        ),
        on="race_id",
        how="full",
        coalesce=True,
    )
    j = j.with_columns(
        (pl.col("p") - pl.col("p_old")).alias("dp"), (pl.col("vs") - pl.col("vs_old")).alias("dvs")
    )
    # ---- inputs
    inp: dict[str, Any] = {
        "model_poll_count": {
            "old": old["diag"].get("poll_count"),
            "new": new["diag"].get("poll_count"),
        },
        "polling_observed_race_option_count": {
            "old": old["diag"].get("polling_observed_race_option_count"),
            "new": new["diag"].get("polling_observed_race_option_count"),
        },
        "prior_only_race_option_count": {
            "old": old["diag"].get("prior_only_race_option_count"),
            "new": new["diag"].get("prior_only_race_option_count"),
        },
    }
    new_by_race: dict[str, list] = {}
    removed_by_race: dict[str, list] = {}
    revised_by_race: dict[str, list] = {}
    if old["polls"] is not None and new["polls"] is not None:
        bo, bn = polls_by_stem(old["polls"]), polls_by_stem(new["polls"])
        inp["curated_votehub_rows_modeled"] = {
            "old": old["polls"].height,
            "new": new["polls"].height,
            "delta": new["polls"].height - old["polls"].height,
        }
        inp["curated_votehub_rows_all_incl_unmodeled"] = {
            "old": old["polls_all"].height,
            "new": new["polls_all"].height,
        }
        inp["distinct_polls_modeled"] = {"old": len(bo), "new": len(bn), "delta": len(bn) - len(bo)}
        for s, rec in bn.items():
            if s not in bo:
                new_by_race.setdefault(rec["race_id"], []).append(
                    poll_record(rec["row"], rec["DEM"], rec["REP"])
                )
            elif (rec["DEM"], rec["REP"]) != (bo[s]["DEM"], bo[s]["REP"]):
                revised_by_race.setdefault(rec["race_id"], []).append(
                    {
                        "poll_id": s,
                        "old": [bo[s]["DEM"], bo[s]["REP"]],
                        "new": [rec["DEM"], rec["REP"]],
                    }
                )
        for s, rec in bo.items():
            if s not in bn:
                removed_by_race.setdefault(rec["race_id"], []).append(
                    poll_record(rec["row"], rec["DEM"], rec["REP"])
                )
        polled_old = set(old["polls"]["race_id"].unique().to_list())
        polled_new = set(new["polls"]["race_id"].unique().to_list())
        inp["polled_race_count"] = {"old": len(polled_old), "new": len(polled_new)}
        inp["races_with_new_polls"] = {k: len(v) for k, v in sorted(new_by_race.items())}
        inp["new_poll_total"] = sum(len(v) for v in new_by_race.values())
        inp["races_with_removed_polls"] = {k: len(v) for k, v in sorted(removed_by_race.items())}
        inp["races_with_revised_polls"] = {k: len(v) for k, v in sorted(revised_by_race.items())}
        inp["unpolled_to_polled"] = sorted(polled_new - polled_old)
        inp["polled_to_unpolled"] = sorted(polled_old - polled_new)
        # new polls parsed but not modeled (races outside the scenario's race catalog)
        so_all, sn_all = set(old["polls_all"]["poll_stem"]), set(new["polls_all"]["poll_stem"])
        unmod = new["polls_all"].filter(
            ~pl.col("race_id").is_in(list(set(rn["race_id"])))
            & pl.col("poll_stem").is_in(list(sn_all - so_all))
        )
        inp["new_polls_in_unmodeled_races"] = {
            k: v
            for k, v in unmod.group_by("race_id").agg(pl.col("poll_stem").n_unique()).iter_rows()
        }
    # source manifest content-hash changes
    mo = {r["source_id"]: r for r in old["manifest"].iter_rows(named=True)}
    mn = {r["source_id"]: r for r in new["manifest"].iter_rows(named=True)}
    inp["source_changes"] = [
        {
            "source_id": s,
            "old_hash": (mo.get(s) or {}).get("content_hash", "")[:12],
            "new_hash": (mn.get(s) or {}).get("content_hash", "")[:12],
            "old_retrieved_at": (mo.get(s) or {}).get("retrieved_at"),
            "new_retrieved_at": (mn.get(s) or {}).get("retrieved_at"),
            "new_status": (mn.get(s) or {}).get("status"),
        }
        for s in sorted(set(mo) | set(mn))
        if (mo.get(s) or {}).get("content_hash") != (mn.get(s) or {}).get("content_hash")
    ]
    # raw VoteHub accounting (new raw polls and why they were or weren't used)
    raw_new: dict[str, Any] = {}
    for s, rec in mn.items():
        if not s.startswith("votehub") or s not in mo:
            continue
        ro_path, rn_path = Path(mo[s]["raw_path"]), Path(rec["raw_path"])
        if not (ro_path.exists() and rn_path.exists()):
            raw_new[s] = {"note": "raw file missing", "old": str(ro_path), "new": str(rn_path)}
            continue
        try:
            stems_new = (
                set(new["polls_all"]["poll_stem"]) if new["polls_all"] is not None else set()
            )
            modeled = set(rn["race_id"].to_list())
            cn_acc = classify_raw_votehub(
                rn_path, json.loads(rec["parser_args"]), stems_new, modeled
            )
            old_ids = {f"votehub-{p.get('id')}" for p in json.loads(ro_path.read_text())}
            new_raw = json.loads(rn_path.read_text())
            new_ids = {f"votehub-{p.get('id')}" for p in new_raw}
            added = new_ids - old_ids
            by_cls = {k: sorted(set(v) & added) for k, v in cn_acc["ids"].items()}
            detail = []
            for p in new_raw:
                pid = f"votehub-{p.get('id')}"
                if pid in added and pid not in by_cls["used"]:
                    cls = next((k for k, v in by_cls.items() if pid in v), "?")
                    detail.append(
                        {
                            "poll_id": pid,
                            "class": cls,
                            "subject": p.get("subject"),
                            "seat_name": p.get("seat_name"),
                            "pollster": p.get("pollster"),
                            "end_date": p.get("end_date"),
                            "choices": [a.get("choice") for a in p.get("answers") or []],
                        }
                    )
            raw_new[s] = {
                "raw_poll_count": {"old": len(old_ids), "new": len(new_ids)},
                "added_raw_polls": len(added),
                "removed_raw_polls": len(old_ids - new_ids),
                "added_by_class": {k: len(v) for k, v in by_cls.items()},
                "added_not_used_detail": detail,
                "new_run_accounting": cn_acc["counts"],
                "new_run_unmapped_candidate_name_count": cn_acc["unmapped_candidate_name_count"],
            }
        except Exception as exc:  # pragma: no cover - diagnostic only
            raw_new[s] = {"error": repr(exc)}
    inp["votehub_raw"] = raw_new
    inp["new_polls_by_race"] = new_by_race
    res["inputs"] = inp
    # ---- fundamentals change
    fund_delta: dict[str, float] = {}
    if old["fp"] is not None and new["fp"] is not None:
        f = (
            new["fp"]
            .filter(pl.col("option_id").str.ends_with("-D"))
            .select("race_id", pl.col("mean_share").alias("m_new"))
            .join(
                old["fp"]
                .filter(pl.col("option_id").str.ends_with("-D"))
                .select("race_id", pl.col("mean_share").alias("m_old")),
                on="race_id",
            )
        )
        fund_delta = {r["race_id"]: r["m_new"] - r["m_old"] for r in f.iter_rows(named=True)}
        res["fundamentals_prior_change"] = {
            "races_changed_gt_1e-6": sum(1 for v in fund_delta.values() if abs(v) > 1e-6),
            "max_abs_change": round(max((abs(v) for v in fund_delta.values()), default=0.0), 6),
        }
    # common (shared) shift: median change in DEM vote-share mean among races with no new polls
    no_new = j.filter(~pl.col("race_id").is_in(list(new_by_race)))
    common_dvs = float(no_new["dvs"].median()) if no_new.height else 0.0
    res["common_shift"] = {
        "median_dvs_races_without_new_polls": round(common_dvs, 5),
        "mean_dvs_all_races": round(float(j["dvs"].mean()), 5),
        "mean_dp_all_races": round(float(j["dp"].mean()), 5),
        "mean_abs_dp_all_races": round(float(j["dp"].abs().mean()), 5),
        "share_races_dvs_positive": round(float((j["dvs"] > 0).mean()), 3),
    }
    ps = posterior_shift(old["pd"], new["pd"])
    post_z: dict[str, float] = {}
    shared_shift_significant = False
    model_polled: set[str] | None = None
    if ps is not None:
        post_z = {r["race_id"]: r["post_z"] for r in ps.iter_rows(named=True)}
        pchk: dict[str, Any] = {}
        for lab, sub in (
            ("unpolled", ps.filter(pl.col("prior_only") & pl.col("prior_only_old"))),
            ("polled", ps.filter(~pl.col("prior_only"))),
        ):
            if sub.height:
                z = sub["post_z"].to_numpy()
                pchk[lab] = {
                    "races": int(z.size),
                    "mean_z": round(float(z.mean()), 3),
                    "sd_z": round(float(z.std(ddof=1)) if z.size > 1 else 0.0, 3),
                    "frac_abs_z_gt_2": round(float(np.mean(np.abs(z) > 2)), 3),
                    "mean_z_times_sqrt_n": round(float(z.mean() * math.sqrt(z.size)), 2),
                }
        shared_shift_significant = abs(pchk.get("unpolled", {}).get("mean_z_times_sqrt_n", 0.0)) > 3
        model_polled = {r["race_id"] for r in ps.iter_rows(named=True) if not r["prior_only"]}
        pchk["interpretation"] = (
            "Per-race z of the change in posterior mean DEM latent logit, using draw-resampling SE. If unpolled races show "
            "mean_z ~ 0 and sd_z ~ 1, their movement is posterior-draw resampling noise (no shared shift)."
        )
        pchk["shared_shift_in_unpolled_races"] = shared_shift_significant
        res["posterior_resampling_check"] = pchk
    bz = bonferroni_z(j.height)
    # ---- movers
    j = j.with_columns(pl.col("dp").abs().alias("adp")).sort(
        ["adp", "race_id"], descending=[True, False]
    )
    movers = []
    # Audit P2-5: only races that actually moved (|dP| > 0) can be movers.
    for r in j.filter(pl.col("adp") > 0).head(TOP_N[ch]).iter_rows(named=True):
        sed_r = se_diff(r["p_old"], n_old, r["p"], n_new)
        vs_sd = (r["vs_p90"] - r["vs_p10"]) / 2.5631 if r.get("vs_p90") is not None else None
        vs_se_diff = math.sqrt(2) * vs_sd / math.sqrt(n_new) if vs_sd else None
        new_p = new_by_race.get(r["race_id"], [])
        fd_r = fund_delta.get(r["race_id"], 0.0)
        polled = (
            (r["race_id"] in model_polled) if model_polled is not None else bool(r.get("poll_rows"))
        )
        z_dp = r["dp"] / sed_r if sed_r else 0.0
        pz = post_z.get(r["race_id"])
        if new_p:
            margins = [p["dem_minus_rep"] for p in new_p if p["dem_minus_rep"] is not None]
            driver = (
                f"new polls ({len(new_p)}; mean D-R {np.mean(margins):+.1f})"
                if margins
                else f"new polls ({len(new_p)})"
            )
            if abs(z_dp) <= 2:
                driver += "; change still within 2 MC SE"
        elif abs(fd_r) > 1e-6:
            driver = "fundamentals prior changed"
        elif shared_shift_significant and r["dp"] * common_dvs > 0:
            driver = "correlated national/shared-error shift (no new polls)"
        elif abs(z_dp) <= 2:
            driver = "Monte Carlo noise (no new polls; |dP| within 2 MC SE)"
        elif abs(z_dp) <= bz:
            driver = (
                f"Monte Carlo noise (no new polls; exceeds 2 SE but below Bonferroni z={bz:.2f} for {j.height} races; posterior-draw z={pz:+.2f})"
                if pz is not None
                else f"Monte Carlo noise (below Bonferroni z={bz:.2f})"
            )
        else:
            driver = "unexplained (no new polls; exceeds Bonferroni-adjusted MC noise)"
        movers.append(
            {
                "race_id": r["race_id"],
                "state": r["state"],
                "tier": r["tier"],
                "old_p": round(r["p_old"], 4),
                "new_p": round(r["p"], 4),
                "delta": round(r["dp"], 4),
                "mc_se_delta": round(sed_r, 4),
                "z": round(r["dp"] / sed_r, 2) if sed_r else None,
                "old_vs": round(r["vs_old"], 4),
                "new_vs": round(r["vs"], 4),
                "delta_vs": round(r["dvs"], 4),
                "vs_mc_se_delta": round(vs_se_diff, 4) if vs_se_diff else None,
                "polled": polled,
                "status": "polled"
                if polled
                else (
                    "fundamentals-only (curated poll rows not usable by model)"
                    if r.get("poll_rows")
                    else "fundamentals-only"
                ),
                "poll_rows_old": r.get("poll_rows_old"),
                "poll_rows_new": r.get("poll_rows"),
                "new_polls": new_p,
                "removed_polls": removed_by_race.get(r["race_id"], []),
                "revised_polls": revised_by_race.get(r["race_id"], []),
                "fundamentals_prior_delta": round(fd_r, 6),
                "posterior_logit_shift_z": None if pz is None else round(pz, 2),
                "driver": driver,
            }
        )
    res["top_movers"] = movers
    res["significant_mover_count"] = sum(
        1 for m in movers if m["z"] is not None and abs(m["z"]) > 2
    )
    # ---- band / favourite
    lo, hi = BAND
    inb = lambda p: p is not None and lo <= p <= hi  # noqa: E731
    res["competitive_band"] = {
        "band": list(BAND),
        "count_old": int(ro.filter(pl.col("p").is_between(lo, hi)).height),
        "count_new": int(rn.filter(pl.col("p").is_between(lo, hi)).height),
        "entered": [
            {"race_id": r["race_id"], "old_p": r["p_old"], "new_p": r["p"]}
            for r in j.sort("race_id").iter_rows(named=True)
            if inb(r["p"]) and not inb(r["p_old"])
        ],
        "left": [
            {"race_id": r["race_id"], "old_p": r["p_old"], "new_p": r["p"]}
            for r in j.sort("race_id").iter_rows(named=True)
            if inb(r["p_old"]) and not inb(r["p"])
        ],
    }
    fav = lambda p: "DEM" if p > 0.5 else "REP" if p < 0.5 else "TOSSUP"  # noqa: E731
    res["favorite_flips"] = [
        {
            "race_id": r["race_id"],
            "old_p": r["p_old"],
            "new_p": r["p"],
            "old_fav": fav(r["p_old"]),
            "new_fav": fav(r["p"]),
            "within_2_mc_se": abs(r["dp"]) <= 2 * se_diff(r["p_old"], n_old, r["p"], n_new),
        }
        for r in j.sort("race_id").iter_rows(named=True)
        if fav(r["p_old"]) != fav(r["p"])
    ]
    # ---- noise yardstick
    res["noise"] = {
        "mc_se_prob_at_0.5_single_run": round(mc_se(0.5, n_new), 4),
        "mc_se_delta_at_0.5_two_independent_runs": round(se_diff(0.5, n_old, 0.5, n_new), 4),
        "two_se_threshold_at_0.5": round(2 * se_diff(0.5, n_old, 0.5, n_new), 4),
        "races_exceeding_2se": int(
            sum(
                1
                for r in j.iter_rows(named=True)
                if abs(r["dp"]) > 2 * se_diff(r["p_old"], n_old, r["p"], n_new)
            )
        ),
        "race_count": j.height,
        "expected_exceedances_if_pure_noise": round(0.0455 * j.height, 1),
        "bonferroni_z": round(bz, 2),
        "races_exceeding_bonferroni": int(
            sum(
                1
                for r in j.iter_rows(named=True)
                if abs(r["dp"]) > bz * se_diff(r["p_old"], n_old, r["p"], n_new)
            )
        ),
        "note": "Binomial MC SE sqrt(p(1-p)/n). The simulator seed is fixed in configs/model.yaml, but posterior-draw "
        "seeds are derived from the data fingerprint, so draws differ whenever inputs change; treat SEs as approximate.",
    }
    if noise_runs:
        res["seed_noise_check"] = noise_run_stats(ch, new, noise_runs)
    res["_all_races"] = [
        {
            k: (round(v, 4) if isinstance(v, float) else v)
            for k, v in r.items()
            if k
            in ("race_id", "p_old", "p", "dp", "vs_old", "vs", "dvs", "poll_rows_old", "poll_rows")
        }
        for r in j.sort("race_id").iter_rows(named=True)
    ]
    return res


# --------------------------------------------------------------------------- output
def fmt_p(x: float | None) -> str:
    return "—" if x is None else f"{x:.3f}"


def fmt_d(x: float | None, nd: int = 3) -> str:
    return "—" if x is None else f"{x:+.{nd}f}"


def _piv_list(items: list[dict[str, Any]]) -> str:
    return ", ".join("{} ({:.3f})".format(x["race_id"], x["rate"]) for x in items)


def _move_list(items: list[dict[str, Any]]) -> str:
    out = ", ".join(
        "{} ({:.3f}→{:.3f})".format(x["race_id"], x["old_p"], x["new_p"]) for x in items
    )
    return out or "none"


def to_markdown(res: dict[str, Any]) -> str:
    L = [
        f"# Run comparison: {res['new_run']} vs {res['old_run']}",
        "",
        f"Generated {res['generated_at']}.",
        "",
    ]
    for ch in ("senate", "house"):
        c = res.get(ch)
        if not c:
            continue
        L += [f"## {ch.title()}", ""]
        ct = c["control"]
        L += [
            "| Metric | Old | New | Change |",
            "|---|---:|---:|---:|",
            f"| DEM P(control) | {fmt_p(ct['dem_control_prob']['old'])} | {fmt_p(ct['dem_control_prob']['new'])} | {fmt_d(ct['dem_control_prob']['delta'])} (MC SE of change {ct['mc_se_delta']:.3f}, z={ct['z']}) |",
        ]
        for k, v in c["dem_seats"].items():
            extra = f" (MC SE {v['mc_se_delta']:.2f})" if "mc_se_delta" in v else ""
            L.append(
                f"| DEM seats {k} | {v['old']:.2f} | {v['new']:.2f} | {fmt_d(v['delta'], 2)}{extra} |"
            )
        fo, fn = (
            c["seat_flips_vs_incumbent_party"]["old"],
            c["seat_flips_vs_incumbent_party"]["new"],
        )
        if fo and fn:
            L.append(
                f"| Seat flips vs incumbent party per sim (mean) | {fo['mean_seat_flips_per_sim']:.2f} | {fn['mean_seat_flips_per_sim']:.2f} | {fn['mean_seat_flips_per_sim'] - fo['mean_seat_flips_per_sim']:+.2f} |"
            )
            L.append(
                f"| — R→D / D→R (mean) | {fo['mean_R_to_D']:.2f} / {fo['mean_D_to_R']:.2f} | {fn['mean_R_to_D']:.2f} / {fn['mean_D_to_R']:.2f} | |"
            )
        L.append(
            f"| Chamber flip prob (from {ct['flip_prob_from_current_majority']['current_majority']} majority) | {fmt_p(ct['flip_prob_from_current_majority']['old'])} | {fmt_p(ct['flip_prob_from_current_majority']['new'])} | |"
        )
        L.append("")
        inp = c["inputs"]
        L += ["### Inputs", ""]
        L.append(
            f"- Model poll rows (posterior_diagnostics.poll_count): {inp['model_poll_count']['old']} → {inp['model_poll_count']['new']}"
        )
        if "distinct_polls_modeled" in inp:
            L.append(
                f"- Distinct VoteHub polls in modeled races: {inp['distinct_polls_modeled']['old']} → {inp['distinct_polls_modeled']['new']} ({inp['new_poll_total']} new)"
            )
            L.append(
                f"- Polled races: {inp['polled_race_count']['old']} → {inp['polled_race_count']['new']}; unpolled→polled: {', '.join(inp['unpolled_to_polled']) or 'none'}; polled→unpolled: {', '.join(inp['polled_to_unpolled']) or 'none'}"
            )
            L.append(
                f"- Races with new polls: {', '.join(f'{k} (+{v})' for k, v in inp['races_with_new_polls'].items()) or 'none'}"
            )
            L.append(
                f"- Removed polls: {inp['races_with_removed_polls'] or 'none'}; revised polls: {inp['races_with_revised_polls'] or 'none'}"
            )
            if inp.get("new_polls_in_unmodeled_races"):
                L.append(
                    f"- New polls in races not modeled (dropped): {inp['new_polls_in_unmodeled_races']}"
                )
        for s, v in inp.get("votehub_raw", {}).items():
            if "raw_poll_count" in v:
                L.append(
                    f"- {s}: raw polls {v['raw_poll_count']['old']} → {v['raw_poll_count']['new']} (+{v['added_raw_polls']}/−{v['removed_raw_polls']}); new raw polls by outcome: {v['added_by_class']}; unmapped candidate names in new run: {v['new_run_unmapped_candidate_name_count']}"
                )
        L.append(
            f"- Source content-hash changes: {', '.join(x['source_id'] for x in inp['source_changes']) or 'none'}"
        )
        if "fundamentals_prior_change" in c:
            L.append(
                f"- Fundamentals prior: {c['fundamentals_prior_change']['races_changed_gt_1e-6']} races changed (max |Δ mean share| {c['fundamentals_prior_change']['max_abs_change']})"
            )
        cs = c["common_shift"]
        L.append(
            f"- Chamber-wide shift: median Δ DEM vote share (races w/o new polls) {cs['median_dvs_races_without_new_polls']:+.4f}; mean Δ P(DEM) {cs['mean_dp_all_races']:+.4f}; mean |Δ P| {cs['mean_abs_dp_all_races']:.4f}"
        )
        L.append("")
        L += [
            f"### Top {len(c['top_movers'])} movers by |Δ DEM win prob|",
            "",
        ]
        if not c.get("significant_mover_count"):
            L += [
                "No significant movers: no race moved by more than 2 MC SE"
                + (" (no race moved at all)." if not c["top_movers"] else "."),
                "",
            ]
        L += [
            "| Race | Old P | New P | Δ | z (vs MC SE) | Δ vote share | Posterior z | Status | Polls old→new | Driver |",
            "|---|---:|---:|---:|---:|---:|---:|---|---|---|",
        ]
        for m in c["top_movers"]:
            L.append(
                f"| {m['race_id']} | {m['old_p']:.3f} | {m['new_p']:.3f} | {m['delta']:+.3f} | {m['z']} | {m['delta_vs']:+.4f} | {m['posterior_logit_shift_z']} | {m['status']} | {m['poll_rows_old'] or 0}→{m['poll_rows_new'] or 0} rows | {m['driver']} |"
            )
        L.append("")
        for m in c["top_movers"]:
            if m["new_polls"] or m["removed_polls"] or m["revised_polls"]:
                L.append(f"**{m['race_id']}** new polls since baseline:")
                for p in m["new_polls"]:
                    L.append(
                        f"- {p['pollster']} ({p['field_dates']}, n={p['sample_size']} {p['population']}, {p['sponsor_class']}): D {p['dem_pct']} / R {p['rep_pct']} → D{p['dem_minus_rep']:+.1f}"
                        if p["dem_minus_rep"] is not None
                        else f"- {p['pollster']} ({p['field_dates']}): D {p['dem_pct']} / R {p['rep_pct']}"
                    )
                for p in m["removed_polls"]:
                    L.append(f"- removed: {p['pollster']} ({p['field_dates']})")
                for p in m["revised_polls"]:
                    L.append(f"- revised: {p['poll_id']} {p['old']} → {p['new']}")
                L.append("")
        allnew = inp.get("new_polls_by_race") or {}
        if allnew:
            L += [
                "### All new polls since baseline (modeled races)",
                "",
                "| Race | Pollster | Field dates | n / pop | D | R | D−R |",
                "|---|---|---|---|---:|---:|---:|",
            ]
            for race, plist in sorted(allnew.items()):
                for p in plist:
                    L.append(
                        f"| {race} | {p['pollster']} | {p['field_dates']} | {p['sample_size']} {p['population']} | {p['dem_pct'] if p['dem_pct'] is not None else '—'} | {p['rep_pct'] if p['rep_pct'] is not None else '—'} | {fmt_d(p['dem_minus_rep'], 1) if p['dem_minus_rep'] is not None else '— (one-sided)'} |"
                    )
            L.append("")
        for s_id, v in inp.get("votehub_raw", {}).items():
            det = v.get("added_not_used_detail") or []
            if det:
                L += [f"New raw VoteHub polls NOT used ({s_id}):", ""]
                for d in det:
                    L.append(
                        f"- [{d['class']}] {d['subject']}{' / ' + str(d['seat_name']) if d['seat_name'] else ''} — {d['pollster']} (end {d['end_date']}): {', '.join(str(x) for x in d['choices'])}"
                    )
                L.append("")
        tp = c["tipping"]
        L += [
            "### Tipping points",
            "",
            f"- Metric: old run `{tp['metric_old']}`, new run `{tp['metric_new']}`"
            + (
                " (not comparable: the old run predates the true tipping-point metric)"
                if tp["metric_old"] != tp["metric_new"]
                else ""
            ),
            f"- Top-3 old: {_piv_list(tp['top3_old'])}",
            f"- Top-3 new: {_piv_list(tp['top3_new'])}",
            f"- Entered top-3: {', '.join(tp['entered_top3']) or 'none'}; left: {', '.join(tp['left_top3']) or 'none'}",
        ]
        if tp.get("rate_mc_se_at_top") is not None:
            L.append(
                f"- Top-10 entered: {', '.join(x['race_id'] for x in tp['entered_top10']) or 'none'}; left: {', '.join(tp['left_top10']) or 'none'} (MC SE at top ≈ {tp['rate_mc_se_at_top']:.3f})"
            )
        L.append("")
        cb = c["competitive_band"]
        L += [
            f"### Competitive band {cb['band'][0]:.0%}–{cb['band'][1]:.0%}",
            "",
            f"- Races in band: {cb['count_old']} → {cb['count_new']}",
            f"- Entered: {_move_list(cb['entered'])}",
            f"- Left: {_move_list(cb['left'])}",
            "",
        ]
        ff = c["favorite_flips"]
        L += ["### Favorite flips", ""]
        L += [
            f"- {x['race_id']}: {x['old_fav']} ({x['old_p']:.3f}) → {x['new_fav']} ({x['new_p']:.3f}){' — within 2 MC SE' if x['within_2_mc_se'] else ''}"
            for x in ff
        ] or ["- none"]
        nz = c["noise"]
        L += [
            "",
            "### Noise yardstick",
            "",
            f"- n={c['simulation_count']['new']} sims: MC SE of a probability at 0.5 = {nz['mc_se_prob_at_0.5_single_run']:.4f}; SE of a run-to-run change = {nz['mc_se_delta_at_0.5_two_independent_runs']:.4f} (2 SE = {nz['two_se_threshold_at_0.5']:.3f}).",
            f"- Races whose |Δ P| exceeds 2 MC SE: {nz['races_exceeding_2se']} of {nz['race_count']} (≈{nz['expected_exceedances_if_pure_noise']} expected from noise alone); exceeding Bonferroni z={nz['bonferroni_z']}: {nz['races_exceeding_bonferroni']}.",
            f"- {nz['note']}",
        ]
        pr = c.get("posterior_resampling_check")
        if pr:
            for lab in ("unpolled", "polled"):
                if lab in pr:
                    v = pr[lab]
                    L.append(
                        f"- Posterior mean-logit shift z, {lab} races (n={v['races']}): mean {v['mean_z']:+.3f}, sd {v['sd_z']:.3f}, share |z|>2 {v['frac_abs_z_gt_2']:.3f} (mean·√n = {v['mean_z_times_sqrt_n']:+.2f})"
                    )
            L.append(
                f"- Shared shift detected in unpolled races: {pr['shared_shift_in_unpolled_races']}"
            )
        sn = c.get("seed_noise_check")
        if sn:
            L.append(
                f"- Seed-only reruns on identical inputs: DEM P(control) range {sn['dem_control_prob_range'][0]:.3f}–{sn['dem_control_prob_range'][1]:.3f}; DEM seat mean range {sn['dem_seat_mean_range'][0]:.2f}–{sn['dem_seat_mean_range'][1]:.2f}; per-race sd(ΔP) {sn['sd_dp_all_races']:.4f} (races 0.2–0.8: {sn['sd_dp_races_0.2_0.8']}); max |ΔP| {sn['max_abs_dp']:.3f}"
            )
            L.append(
                "- Seed-only tipping top-3: "
                + " | ".join(
                    ", ".join(
                        x.replace("US-SEN-", "").replace("US-HOUSE-", "").replace("-2026", "")
                        for x in r["tipping_top3"]
                    )
                    for r in sn["runs"]
                )
            )
        L.append("")
    return "\n".join(L)


def dumbbell(res: dict[str, Any], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    chambers = [ch for ch in ("senate", "house") if res.get(ch)]
    heights = [max(len(res[ch]["top_movers"]), 1) for ch in chambers]
    fig, axes = plt.subplots(
        len(chambers),
        1,
        figsize=(11, 0.55 * sum(heights) + 2.6),
        gridspec_kw={"height_ratios": heights},
        squeeze=False,
    )
    for ax, ch in zip(axes[:, 0], chambers, strict=True):
        ms = list(reversed(res[ch]["top_movers"]))
        y = np.arange(len(ms))
        ax.axvspan(BAND[0], BAND[1], color="#f2f2f2", zorder=0, label="35–65% competitive band")
        ax.axvline(0.5, color="#888", lw=0.8, ls="--", zorder=1)
        for i, m in enumerate(ms):
            col = "#1f5fbf" if m["delta"] > 0 else "#c0392b"
            se2 = 2 * m["mc_se_delta"]
            ax.plot(
                [m["old_p"] - se2, m["old_p"] + se2],
                [i, i],
                color="#bbbbbb",
                lw=7,
                alpha=0.55,
                solid_capstyle="butt",
                zorder=1,
            )
            ax.annotate(
                "",
                xy=(m["new_p"], i),
                xytext=(m["old_p"], i),
                arrowprops={"arrowstyle": "-|>", "color": col, "lw": 1.6},
                zorder=2,
            )
            ax.scatter([m["old_p"]], [i], s=60, color="#7f7f7f", zorder=3)
            ax.scatter(
                [m["new_p"]],
                [i],
                s=70,
                color=col,
                marker="o" if m["polled"] else "D",
                zorder=4,
                edgecolor="black",
                linewidth=0.5,
            )
            tag = f"{m['delta']:+.3f}" + (
                f"  (+{len(m['new_polls'])} poll{'s' if len(m['new_polls']) != 1 else ''})"
                if m["new_polls"]
                else ""
            )
            ax.text(
                max(m["old_p"], m["new_p"], m["old_p"] + se2) + 0.02,
                i,
                tag,
                va="center",
                fontsize=9.5,
            )
        labels = [
            f"{m['race_id'].replace('US-SEN-', '').replace('US-HOUSE-', '').replace('-2026', '')}{'' if m['polled'] else ' *'}"
            for m in ms
        ]
        ax.set_yticks(y, labels, fontsize=10)
        ax.set_xlim(0, 1.08)
        ax.set_ylim(-0.7, len(ms) - 0.3)
        ax.set_xlabel("DEM win probability", fontsize=10)
        ct = res[ch]["control"]["dem_control_prob"]
        ax.set_title(
            f"{ch.title()}: top {len(ms)} movers  |  DEM P(control) {ct['old']:.3f} → {ct['new']:.3f} ({ct['delta']:+.3f})",
            fontsize=11,
            loc="left",
        )
        ax.grid(axis="x", alpha=0.3)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    handles = [
        Line2D([], [], marker="o", color="#7f7f7f", ls="", label=f"{res['old_run_label']} (old)"),
        Line2D(
            [],
            [],
            marker="o",
            color="#1f5fbf",
            ls="",
            markeredgecolor="black",
            label=f"{res['new_run_label']} (new), DEM up",
        ),
        Line2D(
            [],
            [],
            marker="o",
            color="#c0392b",
            ls="",
            markeredgecolor="black",
            label="new, DEM down",
        ),
        Line2D(
            [],
            [],
            marker="D",
            color="white",
            markeredgecolor="black",
            ls="",
            label="diamond / * = no 2026 polls (fundamentals-only)",
        ),
        Line2D(
            [],
            [],
            color="#bbbbbb",
            lw=6,
            alpha=0.5,
            label="±2 MC SE noise band for the change (centered on old)",
        ),
        Patch(color="#f2f2f2", label="35–65% band"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        fontsize=8.5,
        frameon=False,
        bbox_to_anchor=(0.5, 0.0),
    )
    fig.suptitle("Biggest movers in DEM win probability", fontsize=13, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("old_run", type=Path)
    ap.add_argument("new_run", type=Path)
    ap.add_argument(
        "--out", type=Path, required=True, help="Output path stem; writes <out>.json and <out>.md"
    )
    ap.add_argument(
        "--old-inputs",
        type=Path,
        default=None,
        help="Dir with <chamber>/polls.parquet for the old run",
    )
    ap.add_argument(
        "--new-inputs",
        type=Path,
        default=None,
        help="Dir with <chamber>/polls.parquet for the new run",
    )
    ap.add_argument("--plot", type=Path, default=None, help="Write a movers dumbbell PNG here")
    ap.add_argument("--chambers", nargs="+", default=["senate", "house"])
    ap.add_argument(
        "--noise-runs",
        nargs="*",
        type=Path,
        default=[],
        help="Run folders with the NEW run's inputs but different simulator seeds (<dir>/<chamber>/race_forecasts.parquet, control_forecasts.parquet)",
    )
    args = ap.parse_args()
    res: dict[str, Any] = {
        "old_run": str(args.old_run.resolve()),
        "new_run": str(args.new_run.resolve()),
        "old_run_label": args.old_run.resolve().name,
        "new_run_label": args.new_run.resolve().name,
        "generated_at": datetime.now(UTC).isoformat(),
        "competitive_band": list(BAND),
        "top_n": TOP_N,
    }
    for ch in args.chambers:
        if not (args.old_run / ch).exists() or not (args.new_run / ch).exists():
            continue
        old = load_run(args.old_run, ch, args.old_inputs)
        new = load_run(args.new_run, ch, args.new_inputs)
        res[ch] = compare_chamber(ch, old, new, args.noise_runs)
        res[ch]["paths"] = {
            "old": {
                k: old.get(k) for k in ("sp_path", "fd_path", "fp_path", "pd_path", "polls_path")
            },
            "new": {
                k: new.get(k) for k in ("sp_path", "fd_path", "fp_path", "pd_path", "polls_path")
            },
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    Path(f"{args.out}.json").write_text(json.dumps(res, indent=2, default=str))
    Path(f"{args.out}.md").write_text(to_markdown(res))
    if args.plot:
        dumbbell(res, args.plot)
    print(f"wrote {args.out}.json, {args.out}.md" + (f", {args.plot}" if args.plot else ""))


if __name__ == "__main__":
    main()
