# Impact of the 2026-10 audit fixes

Branch `audit-fixes-2026-10`. All numbers come from packaged runs on the box:

- `runs/adhoc/before-fixes`: pre-fix code. Frozen 2026-10-19 VoteHub and Wikipedia payloads, as_of 2026-11-02, 1,000 sims. It matches `runs/weekly/2026-10-19` exactly.
- `runs/adhoc/after-fixes`: fixed code on the **same** frozen payloads (sha256 `89a4e76d…` Senate, unchanged House). as_of 2026-10-19, which is the new default on that date. 20,000 Senate and 10,000 House sims.
- `runs/adhoc/2026-10-19-postfix`: fixed code with a **fresh** fetch (21:44 UTC = 17:44 EDT) through `scripts/weekly_run.sh`.
  - VoteHub returned byte-identical payloads, so it reproduces after-fixes exactly: 0 races moved, "No significant movers". All of the change below is **method**, not poll movement.
  - The Wikipedia race-presence page hash changed, but that page is audit-only and had no effect on the forecast.

Comparison reports:
- `runs/adhoc/after-fixes/comparison_vs_before-fixes.{md,json}` and `movers_dumbbell.png`;
- `runs/adhoc/2026-10-19-postfix/comparison_vs_2026-10-19.{md,json}`;
- `runs/adhoc/2026-10-19-postfix/comparison_vs_after-fixes.{md,json}`.

## Headline

| | Before | After | Change |
|---|---:|---:|---:|
| **Senate** P(D control), independents excluded (headline) | 0.314 | **0.501** | +0.187 |
| Senate P(D control), counting Osborn (NE) as D-caucusing | n/a (no IND option) | 0.503 | |
| Senate D seats mean | 48.95 | 50.15 | +1.21 |
| Senate D seats p10 / p50 / p90 | 45 / 49 / 53 | 45 / 51 / 55 | |
| **House** P(D control) | 0.831 | **0.717** | -0.114 |
| House D seats mean | 230.5 | 227.9 | -2.6 |
| House D seats p10 / p50 / p90 | 213 / 231.5 / 247 | 209 / 226 / 250 | |

Monte Carlo SE at the new sim counts:
- Senate control P: 0.0035 (20k). Senate seat mean: 0.03.
- House control P: 0.0045 (10k). House seat mean: 0.17.

Each per-race seed comes from the config seed (P1-1), so a rerun on the same inputs is bit-identical; the postfix run confirms this. Before the fix, at 1k sims, the SE was about 0.015.

Including or excluding Osborn barely matters (0.501 vs 0.503) because the model gives him only 8.1% in NE (see open issues).

## Attribution by phase (same frozen payloads)

| Step | Senate P(D) | House P(D) |
|---|---:|---:|
| Baseline (pre-fix, 1k sims, as_of 11-02) | 0.314 | 0.831 |
| Phase 1: parser v2 (P0-2/3/7), recency-weighted posterior (P0-1b), 20k/10k sims (P1-1) | 0.195 | 0.895 |
| Phase 2: real 2026 Senate rows + FL/OH specials + independents (P0-4/6) | 0.333 | |
| + measured environment D+8.2 instead of the D+2.5 override (P0-6) | 0.473 | |
| + partisan-sponsor adjustment (P1-4) | 0.471 | 0.889 |
| Phase 3: real 2026 House panel (P0-5), swing fixed to share units | 0.373 | 0.887 |
| + explicit national/regional error, no double local noise (P1-3) | 0.455 | 0.711 |
| P1-2 tipping point (metric only) | 0.455 | 0.711 |
| as_of 2026-10-19 instead of the hard-coded 2026-11-02 (P1-5 default) | **0.501** | **0.717** |

Phase 4 (fetch strictness, hashes, labels, CI, scripts) does not change any probability.

Notes on the attribution:
- The as_of row is measured: the same code on the same data with as_of 11-02 gives Senate 0.455, mean 49.75, 45/50/54. The old hard-coded date pretended it was election eve, which shrank the remaining drift and lowered recency weights.
- The 0.992 House value from the margin-unit swing bug (an intermediate p3a run) was fixed before commit and is not part of the attribution.

## Biggest movers (before → after, same payload)

All Senate moves exceed 2 MC SE (z ≥ 9.5). Race P is P(D), or P(IND) where no Democrat is on the ballot.

| Race | Old P | New P | Δ | Cause |
|---|---:|---:|---:|---|
| MT | 0.247 | 0.011 | -0.236 | P0-2/P0-3 parser v2: 10 polls newly used (rows 10→20). Example: Tavern Research 2026-07-23→27, D 24 / R 49; Hart Research 2026-08-01→04, D 15 / R 45. Modeled D vs R; Bodnar (I) counted as other (TODO(verify)). |
| CO | 0.756 | 0.958 | +0.202 | P0-4 real 2026 rows + P0-6 measured environment (fundamentals prior +2.4 pts share; unpolled) |
| MI | 0.497 | 0.693 | +0.196 | P0-3 dedupe/nominee filter (rows 122→72) + P0-1b recency. Example: Emerson 2026-09-12→14, D 48.2 / R 45.7; Suffolk 2026-09-16→20, D 47 / R 40 |
| AK | 0.495 | 0.305 | -0.190 | P0-4/P0-6 fundamentals prior and recency on unchanged polls |
| VA | 0.791 | 0.934 | +0.143 | Parser v2 (first usable poll: Public Sentiment Institute 2026-06-12→16, D 50 / R 33) + fundamentals |
| NH | 0.700 | 0.841 | +0.141 | Recency + fundamentals; UNH 2026-09-17→21, D 50 / R 42 now used |
| SC | 0.284 | 0.147 | -0.137 | Parser v2 (InsiderAdvantage 2026-09-28→29, D 45 / R 46) + error model |
| KS | 0.147 | 0.274 | +0.127 | Parser v2 (Emerson 2026-09-15→16, D 45 / R 43; Wedgewood 2026-09-22→24, D 50 / R 48) + fundamentals |

- New races: FL (0.145) and OH (0.669) specials (P0-4). OH is the #4 tipping point (10.1%).
- Tipping point (new metric, P1-2): IA 15.1%, TX 12.6%, ME 11.3%, OH 10.1%, MI 9.3%.

**House:** the top 12 movers (|Δ| ≈ 0.97) are all from P0-5. The pre-fix 2026 House rows were synthetic, with fictional leans. Examples:
- IL-12: 0.978 → 0.000;
- TX-30: 0.023 → 1.000;
- VA-09: 0.975 → 0.000;
- UT-01: 0.028 → 1.000 (new 2026 map, plus Lighthouse Research 2026-08-31→09-10, D 62.7 / R 21.2).

About 130 favorite flips have the same cause. The competitive band shrank from 33 to 14 races. The new tipping points are AZ-01, KY-06 and OH-09 (2.4%, 2.4%, 2.1%).

## Poll accounting, before → after (same raw payload)

| | Senate before | Senate after | House before | House after |
|---|---:|---:|---:|---:|
| Raw polls | 479 | 479 | 134 | 134 |
| Used | 288 | 284 | 60 | 103 |
| Dropped: unmapped candidate | 96 | 0 | 74 | 0 |
| Dropped: non-nominee (hypothetical/primary loser) | (not checked) | 91 | (not checked) | 4 |
| Dropped: missing D or R side (one-sided) | (not checked) | 26 | (not checked) | 13 |
| Dropped: same-party answers | (not checked) | 2 | (not checked) | 4 |
| Skipped: same-party race (CA top-two) | — | 0 | — | 3 |
| Dropped: duplicate | (not checked) | 16 | (not checked) | 7 |
| Primary polls skipped | 60 | 60 | 0 | 0 |
| Parsed but not modeled | 35 | 0 | 0 | 0 |

How the counts change:
- **Senate:** the v1 parser used 288 polls, but some were hypothetical matchups and duplicates. The v2 parser uses 284 general-election nominee polls.
- **Model poll rows:** 572 → 568 (Senate) and 112 → 206 (House).
- **Polled races:** 18 → 25 Senate and 33 → 63 House.

## Open TODO(verify) and known limitations

- **NE (Osborn, I):** the two VoteHub polls are near-ties (GBAO 2026-09-03→08, Osborn 47 / Ricketts 46, DEM-sponsored; National Public Affairs 2026-10-02→05, 46 / 49). The fundamentals prior is still 39.1% for Osborn, so the model gives him only 8.1%. That prior uses the 2024 presidential two-party share (0.374), not Osborn's own 2024 Senate result (46.5% of the vote, 46.65% two-party; Nebraska Board of State Canvassers via uselectionatlas.org and Wikipedia). This is the biggest remaining modeling question. TODO(verify) and consider an independent-candidate prior.
- **Independents' caucus intentions** (Osborn, Bengs, Achilles): the headline excludes them. TODO(verify) after the election.
- **Montana** three-way race (Alme R, Bankhead D, Bodnar I) is modeled as D vs R.
- **AK-01:** Begich vs Hill (I) polls omit Hafner (D) and are dropped. Schultz's REP label is from the prior map.
- **House previous shares** are 2024 presidential, not House results (the MIT House file is guestbook-gated). The 6-pt race-level prior cap needs checking on real House data.
- **Third-party mappings** to spot-check: MA Tache OTH / Deaton REP; RI Bahry IND / McKay REP; MI Tim Long OTH.
- **Louisiana** multi-candidate seats and **California** top-two outcomes need re-checking after certification.
- **National error term** (2.5 pts share) is to be re-estimated from raw_polls; see `docs/error_model.md`.
- **Synthetic inputs remain:** fundamentals coefficients, calibration (labelled `identity/fitted_on_synthetic_backtest`), fundraising/economic fields. Treat probabilities as model output under that caveat.
- **P1-6** (one latent two-party share per race) is deferred.
