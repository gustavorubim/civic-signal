# CODEX_TASKS.md: civic-signal audit fixes (Oct 2026)

This is the work queue for an autonomous coding agent (OpenAI Codex) on branch
`audit-fixes-2026-10`. It's self-contained. The full evidence is in
[`docs/audit-2026-10-19.md`](docs/audit-2026-10-19.md); task IDs below (P0-x, P1-x, P2-x) refer to it.
Read `AGENTS.md` and `SPEC.md` before starting. Their rules still apply.

---------------------------------------------------------------------------------------------------
## 0. Context

- **What this is:** a forecast of the US 2026 midterms (Senate and House). **Election day is Tue Nov 3, 2026.**
  The goal is to make the forecast *correct* before then, not just green.
- **Stack:** Python 3.13, `uv`, Polars, NumPyro/JAX. Pipeline: ingest → normalize → features/tiering →
  ensemble (polls, fundamentals, markets) → Monte Carlo simulation → artifacts.
- **Live poll source:** VoteHub API (`https://api.votehub.com/polls`, no key, **CC BY 4.0**, attribution required).
  Parser `votehub-race-polls-v1` lives in `src/civic_signal/normalize/builder.py` (around lines 1500–1800).
  The candidate→party map is `fixtures/candidates_2026_parties.csv` (columns `office,jurisdiction,name,party`;
  derived from Wikipedia, **CC BY-SA**; see `fixtures/candidates_2026_parties.SOURCES.txt`).
  The old FiveThirtyEight Datasette mirror (`configs/sources_live.yaml`) has **no 2026 rows**. Don't use it.
- **Production inference:** the analytic Bayesian backend (`--inference-engine bayes --bayesian-backend analytic`,
  `src/civic_signal/models/polling_bayes.py`). NUTS (`inference/state_space.py`) is kept for calibration only.
- **Seed:** `configs/model.yaml` `seed: 20260508`. Sims: `simulation_count: 1000` (to be raised, Task 1.7).
- **Canonical production commands** (bash; `PYTHONPATH=src` is set in run scripts):

```bash
uv run civic-signal forecast run --as-of 2026-11-02 --run-id senate-2026-midterm-live \
  --scenario senate_2026_state --sources-config sources_senate_live.yaml \
  --data-dir data/senate-live-<tag> --artifacts-dir artifacts/senate-2026-live-<tag> \
  --inference-engine bayes --bayesian-backend analytic

uv run civic-signal forecast run --as-of 2026-11-02 --run-id house-2026-midterm-live \
  --scenario house_2026_district --sources-config sources_house_live.yaml \
  --data-dir data/house-live-<tag> --artifacts-dir artifacts/house-2026-live-<tag> \
  --inference-engine bayes --bayesian-backend analytic

# Package and compare (run folders live under runs/, which is gitignored)
uv run python scripts/package_run.py --chamber senate --run-date <YYYY-MM-DD> --as-of 2026-11-02 \
  --artifact-root artifacts/senate-2026-live-<tag>/runs/senate-2026-midterm-live \
  --data-dir data/senate-live-<tag> --out runs/<kind>/<date> \
  --sources-config sources_senate_live.yaml --scenario senate_2026_state
uv run python scripts/compare_runs.py runs/<old> runs/<new> --out runs/<new>/comparison_vs_<old> \
  --plot runs/<new>/movers_dumbbell.png
```

- **Use a fresh `--data-dir`** for every run. Otherwise sync can silently reuse cached raw files (see P1-7).

---------------------------------------------------------------------------------------------------
## 1. Ground rules (non-negotiable)

1. **Branch:** work on `audit-fixes-2026-10` (or short-lived branches off it). **Never push to `main`.** Don't open
   or merge PRs unless the human asks.
2. **Small commits, one per task** (or sub-task), with message prefix `fix(P0-2): ...`, `feat(P0-4): ...` and so on.
3. **Every commit must pass:**
   ```bash
   uv sync
   uv run ruff check
   uv run ruff format --check
   uv run pytest -q
   uv run pytest --cov=src/civic_signal --cov-fail-under=90   # coverage gate from AGENTS.md; don't lower it
   ```
   Lint debt that already existed (5 ruff errors, 8 unformatted files) is cleaned up in Task 4.4. Until then,
   don't add new violations, and fix any you touch.
4. **Add or adjust tests for every fix.** Prefer small, deterministic, fixture-backed tests
   (extend `tests/golden_fixtures/votehub_race_polls_sample.json` or add new fixtures). No network in tests.
5. **Never fabricate data.** Candidate parties, nominees, primary dates, incumbents, district leans and election results
   must come from a cited source, recorded in a `*.SOURCES.txt` or docs. If you can't verify a fact, leave the
   existing value or mark it with `# TODO(verify): ...` (or a `verify` column / `status=unverified`) and list it in your
   final report. Known items that **must be verified, not assumed**:
   - **Matt Schultz** (AK-01, opponent of Begich): party? (the map says REP; probably DEM)
   - **CA-40** general election: is it really Calvert (R) vs Young Kim (R) after top-two? If so, P(D) = 0.
   - **Dan Osborn** (NE, independent): stated caucus intention, so control can be reported both with and without him.
   - **Troy Jackson** as the ME Democratic nominee; the MI/NC/GA/TX/IA nominees used by the nominee filter.
   - Third-party names mapped to major parties: MA Joe Tache / John Deaton, RI Michael Bahry / Raymond McKay, MI Tim Long.
   - Senate retirements/open seats (MI, NH, MN, IL, NC, KY, AL, IA) and which 2025–26 redistricted House maps are final.
6. **Don't change the meaning of the audit.** If you find the audit was wrong about something, say so in your report.
   Don't silently "fix" it away.
7. **Don't commit** `data/`, `artifacts/`, `runs/`, secrets, or large binaries.
8. Update `README.md` / `docs/` when user-facing behavior changes (AGENTS.md rule).
9. Keep a running log in `docs/codex_progress.md`: task ID, commit SHA, what changed, tests added, open `TODO(verify)` items.

---------------------------------------------------------------------------------------------------
## 2. Before you start: capture the "before" baseline

Run both production commands above with `--data-dir data/{senate,house}-live-before` and
`--artifacts-dir artifacts/{senate,house}-2026-live-before`, then package to `runs/adhoc/before-fixes`.
Save the raw VoteHub payload paths. **The "after" run (Task 5.1) must reuse the same raw poll payloads**
(copy the `data/*-before/raw` dir, or run sync once and then disable network), so deltas are method-driven only.

---------------------------------------------------------------------------------------------------
## Phase 1: data ingest correctness (highest impact)

### Task 1.0: Commit hygiene ✅ DONE
The VoteHub work, scripts, party map and audit are committed on this branch; `runs/` is gitignored. Nothing to do.

### Task 1.1 (P0-2): Fix party labels, and guard against same-party / missing-party polls
- **Problem:** the parser sums pct for all answers that map to the same party, so one wrong label turns a head-to-head
  into a one-sided poll at about 85–90%.
- **Evidence:**
  - `src/civic_signal/normalize/builder.py:1593`: `party_pct[party] = party_pct.get(party, 0.0) + pct`
  - `fixtures/candidates_2026_parties.csv` (file line numbers, header = line 1):
    `:96` Everett Wess (AL, Democrat) labeled REP → both AL polls REP≈87%.
    `:140` Larry Marker (NM, Republican) labeled DEM → NM ≈99.5% D.
    `:65` Bob Brooks (PA-07, Democrat) labeled REP → PA-07 P(D)=2.3% despite Brooks +4/+5 polls.
    `:90` Gerald Malloy (VT-01, Republican) labeled DEM.
    `:3` Matt Schultz (AK-01) labeled REP → AK-01 P(D)=0.4% **[verify]**.
    `:16-17` CA-40 Calvert and Young Kim both REP **[verify general]**.
    `:119-120` MA Tache/Deaton both REP, `:148-149` RI Bahry/McKay both REP, `:126` MI Tim Long DEM **[verify]**.
- **Change:**
  1. Correct Wess→DEM, Marker→REP, Brooks→DEM, Malloy→REP. Change Schultz only if verified; otherwise add a TODO(verify).
     Re-label third-party/independent names to `OTH`/`IND` (or the correct minor party) when verified.
  2. In the VoteHub parser, for a race that models DEM and REP: **reject the poll (count it as `dropped_same_party`)**
     when ≥2 answers map to the same major party, **and reject it (`dropped_missing_side`)** when either
     DEM or REP is absent. Exception: a race explicitly configured as same-party (CA-40 R-vs-R if verified), which needs
     its own handling (P(D)=0 or a non-D/R race type). Record drop reasons in the raw accounting output
     (`votehub_raw_accounting.json`).
  3. Fix `scripts/compare_runs.py` labeling AL as "fundamentals-only (curated poll rows not usable)" when AL polls are in
     (it's keyed per DEM option); see also Task 4.6.
- **Acceptance:** new tests: (a) two answers with the same party → poll dropped with reason; (b) only one side present →
  dropped; (c) corrected rows yield two-sided polls for AL/NM/PA-07/VT-01 fixtures. After the fix the live run shows
  0 one-sided polls in the accounting file (previously 4 Senate + 8 House).

### Task 1.2 (P0-3): Unmapped names become OTHER instead of dropping the whole poll
- **Problem:** one unmapped answer `break`s the loop and throws away the poll. That's 96/479 Senate and 74/134 (55%)
  House polls, mostly the newest full-ballot polls (MI lost 8 Sep–Oct polls, TX 12, OH BGSU Brown+6, NC, MT, MN, NE, NE-02, NM-03...).
- **Evidence:** `builder.py:1585-1587` (`if party is None or party not in parties: unmapped = True; break`).
- **Change:** an unmapped or out-of-scope name maps to `OTHER` (don't break). Keep the poll if both the DEM and REP
  nominees are present (after Task 1.4's nominee filter). Carry `OTHER` as a separate share or ignore it for D/R;
  don't add it to D or R. Log the distinct unmapped names with counts to the accounting JSON so they can be backfilled.
  Backfill **verified** missing major-party names (e.g., NE Osborn as IND and Ricketts as REP, OH nominees,
  IL/SC opponents, House nominees for polled districts) with sources in `candidates_2026_parties.SOURCES.txt`.
- **Acceptance:** a test with a full-ballot poll (D, R, Libertarian unmapped) keeps the poll with correct D/R pct.
  Live accounting: "dropped for unmapped" goes from 96→~0 Senate and 74→small House, and unmapped names are listed.

### Task 1.3 (P0-7): Deduplicate poll versions
- **Problem:** LV+RV (or two LV) versions of the same poll are counted twice: 15 Senate and 2 House cases
  (HPU/YouGov NC, TIPP MI, Alaska Survey Research). Once Task 1.2 lands, full-ballot and H2H versions of the same poll will
  both count (co/efficient OH). Generic "Dem/Rep" district ballots are pooled with candidate polls (MI-04, TX-23, NY-17, VA-01, AZ-06).
- **Change:** dedupe by (race, pollster, start_date, end_date). Preference: LV > RV > A, then full-ballot over
  head-to-head, then candidate-named over generic party ballot. Record dropped duplicates in accounting.
- **Acceptance:** a fixture with LV+RV twins → 1 row kept (LV). Full-ballot + H2H → full-ballot kept. Generic + candidate →
  candidate kept.

### Task 1.4 (P0-1a): Nominee column and filter for hypothetical / pre-primary matchups
- **Problem:** the map includes primary losers and never-candidates (Stevens, McMorrow MI; Mills, Platner ME; Cornyn TX;
  Wahls IA; Scott Brown NH; Tillis, Lara Trump NC; Buddy Carter GA; Craig MN; Moulton MA; Brink MI-07; Bains CA-22, ...),
  so stale hypothetical head-to-heads are used as if they were the real race. Raw D–R means: MI −0.1 (used) vs +3.5
  (nominee-only since Aug 15); TX +1.1 vs +2.6; GA +6.8 vs +9.1; NC +8.2 vs +10.1.
- **Change:** add a `status` column to `fixtures/candidates_2026_parties.csv` (`nominee` | `primary_loser` | `withdrawn` |
  `hypothetical` | `minor` | `unverified`), populated from cited sources. In the parser, keep only polls whose D and R
  choices are both `status=nominee` (for NE: IND Osborn vs REP Ricketts). Optionally also drop polls ending before the
  state's primary date (add `primary_date` per race if sourced). Keep a config flag `require_nominees: true` in the live
  source configs so the old behavior stays testable.
- **Acceptance:** tests showing a Stevens-vs-Rogers poll excluded and an El-Sayed-vs-Rogers poll kept **[verify MI nominee]**.
  Any nominee you couldn't verify stays `unverified` and is listed in the progress log.

### Task 1.5 (P0-1b): Recency weighting in the analytic backend
- **Problem:** `_posterior_logit` (`src/civic_signal/models/polling_bayes.py:620`) pools all polls back to Nov 2024 with
  equal weight. Recency exists only in the NUTS path (`inference/state_space.py:181`, `_recency_weight`;
  `configs/model.yaml:63` `poll_half_life_days: 7`, while state_space defaults to 21).
- **Change:** multiply each poll's observation precision by `0.5 ** (age_days / half_life_days)`, where `age_days = as_of − end_date`.
  Read `half_life_days` from config (new key, e.g. `polling.analytic_half_life_days`, default **18**, valid range 14–21).
  Optionally add a hard window `max_poll_age_days: 120`. Reuse `_recency_weight` rather than duplicating it.
- **Acceptance:** unit tests: identical polls at age 0 and age = half-life → weights 1.0 and 0.5. A 2024 poll has
  negligible influence relative to a fresh one. Existing tests updated deliberately (explain any golden changes).

### Task 1.6 (P1-5): `as_of` defaults to today
- **Problem:** `--as-of 2026-11-02` is hard-coded (`configs/scenarios.yaml:400` and ~`:485` `default_as_of`,
  `scripts/package_run.py --as-of` default, `runs/rerun_midterms.sh`), so `max_horizon_days: 1`. With recency weighting
  this becomes wrong in a material way.
- **Change:** the CLI `--as-of` defaults to today's date (`date.today()`) when omitted for live sources. The scenario
  `default_as_of` stays only as a fallback for fixture/backtest runs. `package_run.py --as-of` defaults to today. Keep
  `election_date: 2026-11-03` separate from `as_of`.
- **Acceptance:** test that the live CLI without `--as-of` uses today (monkeypatch the date), and that the horizon in
  diagnostics = election_date − as_of.

### Task 1.7 (P1-1): Fixed per-race seeds and more simulations
- **Problem:** `configs/model.yaml:3` `simulation_count: 1000` gives MC SE ≈ 0.015. The draw seed is
  `sha256(bundle fingerprint incl. polls)` (`polling_bayes.py:732` `_draw_seed`; `polling_kalman.py:~544`), and races
  draw sequentially from one RNG, so any new poll reshuffles every race (unpolled races "move" 3–5 pts).
- **Change:** derive the seed from the config `seed` (20260508) only, and give each race its own stream:
  `np.random.SeedSequence([seed, stable_hash(race_id)])` (and the same for the simulation's national/regional
  draws). Raise to **20,000** sims for Senate and **10,000** for House (scenario-level override or config). Check runtime
  (it was ~15s Senate / ~35s House at 1k with analytic); vectorize if needed and run
  `uv run civic-signal benchmark run --as-of 2026-05-08 --run-id perf` (AGENTS.md).
- **Acceptance:** test: adding a poll to race A leaves race B's draws byte-identical. Two runs on the same inputs
  are identical. Report wall-clock before and after.

---------------------------------------------------------------------------------------------------
## Phase 2: Senate structure and inputs

### Task 2.1 (P0-4): Add the FL and OH specials; fix holdovers
- **Evidence:** `fixtures/senate_state_panel.csv` 2026 has 33 Class II states and no FL/OH.
  `configs/scenarios.yaml:404-407` holdovers DEM 32 / REP 33 / IND 2 silently count FL and OH as safe R.
  VoteHub has 26 FL and 25 OH polls (recent OH: CNN/SSRS Brown+6, BGSU Brown+6, co/efficient Husted+1; FL: Moody
  +2…+12 over Angie Nixon **[verify nominee]**).
- **Change:** add `US-SEN-FL-2026` and `US-SEN-OH-2026` rows (special elections; real 2024 presidential state margin, appointed
  incumbent flags, sourced). Set holdovers to **REP 31**. Map the OH/FL nominees in the party map. Make sure the live
  configs include them.
- **Acceptance:** total seats = 100 in simulation (assert in a test). FL/OH appear in race outputs with polls used.

### Task 2.2 (P0-4): Nebraska independent (Osborn) and dual control reporting
- **Problem:** NE is modeled as a generic "DEM nominee" at 3.4%. There's no Democratic nominee, and Osborn (I) vs Ricketts
  polls are dropped (both names unmapped). Recent polls: Osborn +1, Ricketts +3.
- **Change:** support an IND major option for NE (Osborn vs Ricketts) using his polls. Add control outputs for both
  "D control counting Osborn with D" and "D control without Osborn". Which one is the headline is a config choice,
  documented with a **TODO(verify)** on his stated caucus intention.
- **Acceptance:** tests for an IND-vs-REP race and for both control variants. The NE probability is driven by Osborn polls.

### Task 2.3 (P0-5 Senate): Real 2026 Senate panel rows
- **Problem:** the Senate 2026 rows are synthetic. Open seats are flagged as incumbents (MI, NH, MN, IL as D; NC, KY, AL, IA as R
  **[verify retirements]**), WV is marked a DEM incumbent (Capito is R), and the leans are approximate.
- **Change:** rebuild only the 2026 rows from cited data: 2024 presidential margin by state, 2020 Class II result,
  incumbent/open status (e.g. `unitedstates/congress-legislators`, CC0, plus sourced retirements). Keep the synthetic
  history for backtests and mark it `synthetic=true`.
- **Acceptance:** a test that WV's incumbent party is REP and open seats have `incumbent=none`. A SOURCES file is added.

### Task 2.4 (P0-6): Re-measure the national environment
- **Evidence:** `fixtures/environment_override.json` pins D+2.5 (Jul 9), while measured was D+5.8 (`fixtures/measured_environment.json`,
  never refreshed). The panels are baked with it (House `national_swing` 5.0, Senate 0.5).
- **Change:** re-run `scripts/measure_environment.py` with current generic-ballot data (sourced, never invented), regenerate
  the panels, and make an explicit, documented decision to keep or remove the override (default: remove it and use measured).
  Record the value and source in run metadata.
- **Acceptance:** run metadata shows the environment value, source, and date. A test covers override precedence (already exists; keep it passing).

### Task 2.5 (P1-4): Partisan-sponsor house effect
- **Problem:** VoteHub `partisan: DEM|REP` (100/479 Senate polls) collapses to `"partisan"` (`builder.py:1606`). The model
  only applies ×0.85 effective sample size (`polling_kalman.py:497`), which is effectively no adjustment.
- **Change:** keep the sponsor party (`sponsor_party` column). In the analytic backend, shift sponsored polls ~2 pts of share
  against the sponsor (configurable `partisan_sponsor_shift: 0.02`) and/or halve their weight (`partisan_weight: 0.5`).
- **Acceptance:** tests that a DEM-sponsored D+6 poll counts as about D+4 with half the weight, and that nonpartisan polls are unchanged.

---------------------------------------------------------------------------------------------------
## Phase 3: House baseline and error model

### Task 3.1 (P0-5 House): Real 2024 presidential-by-CD baseline and incumbents
- **Problem:** `scripts/generate_house_panel.py` is "synthetic-but-plausible ... NOT actual historical House election results."
  The 2026 rows have 223 D / 212 R incumbents (real ≈ 215 / 220) and implausible PVIs (TX-15 D+15.3, WI-01 D+2.6, CA-40 D+48,
  CA-48 D+51, NE-02 R+38.8, MI-06 R+7.4), with no 2025–26 redistricting.
- **Change:** replace **only the 2026 House rows** with: 2024 presidential margin by CD **on the 2026 maps** (e.g., The Downballot
  or Daily Kos Elections CD data; record license), incumbents and open seats (`unitedstates/congress-legislators` + sourced
  retirements), and 2024 House results (MIT Election Data + Science Lab). Write a mapping/QA script
  (`scripts/build_house_2026_panel.py`) with sanity checks: 435 rows, incumbent counts match the source, no |lean| > 60.
  Backfill House nominee names (replace "DEM/REP nominee XX-NN" placeholders) where sourced. Mark redistricted states whose
  map status is unverified with TODO(verify). If the data can't be obtained offline, implement the loader and QA script,
  leave the data file for a human to supply, and say so clearly.
- **Acceptance:** QA script tests on a small fixture. Panel row counts are correct. Synthetic history remains for backtests,
  flagged `synthetic=true`.

### Task 3.2 (P1-3): Error structure (national/regional error, no double-counted noise)
- **Evidence:** common-factor SD 2.4 pts Senate / 2.0 House, mean per-race SD 7.0 / 13.3 pts, mean pairwise corr 0.10 / 0.02.
  `configs/model.yaml:162` `residual_covariance_mode: residual_covariance_only` zeroes `national_sigma` and uses a covariance learned
  from synthetic panels. The simulation adds heavy-tailed local noise at tier floors (A 0.035 / B 0.075, `model.yaml:146-147`;
  `src/civic_signal/models/simulation.py:~310`) on top of posterior draws that already include `min_nonsampling_error` 0.035 +
  election-day SD. Result: leaners get pulled to 50% (NC Cooper +10 → 84%, GA Ossoff +9 → 77%).
- **Change:** an explicit national error (SD ~2.5–3 pts of share) plus regional error (~1.5 pts). Remove or shrink the extra local
  noise when posterior draws are used. Make it configurable and documented. Sanity-check against historical polling error
  (document the source).
- **Acceptance:** tests that the implied mean pairwise correlation rises (e.g., >0.3 Senate on a fixture) and that a +10 race with
  good polling isn't below ~90%. Document the before/after per-race SD.

### Task 3.3 (P1-2): True tipping-point metric
- **Problem:** `simulation.py:862-890` `_pivotal_races` counts "decisiveness", so mostly-R races dominate (KS at 14.7% P(D) ranks 3rd).
- **Change:** implement the 538-style tipping point: within each draw, sort races by D margin, accumulate seats (plus holdovers) to the
  control threshold, and record the race that crosses it. Output `tipping_point_probability`. Keep the old metric renamed
  `decisive_probability`. Update the plots (`reports/plots.py` tipping bars) and compare_runs keys.
- **Acceptance:** a hand-built 3-race fixture with a known tipping race. Probabilities sum to 1 across races per chamber.

---------------------------------------------------------------------------------------------------
## Phase 4: robustness, hygiene, docs

### Task 4.1 (P1-7): Fail loudly on fetch failure
`ingest/sync.py:129` `_sync_http` silently falls back to cached raw files on any HTTP error. On live configs, fail (non-zero exit)
unless `--allow-cached` is passed. Always record `status=cached|fresh` and `fetched_at` in the manifest and summary.json. Check
whether VoteHub `/polls` paginates **[verify]**. Tests: a mocked HTTP error → raises; with `--allow-cached` → status=cached.

### Task 4.2 (P1-8): Stop config-hash churn
`pipeline.py:139` injects `_fundamentals_prior_rows` (data) into `model_config` before hashing (`:245`, `_config_hash`), so
`model_config_hash` changes every run. Hash only the static config. Record data hashes (raw VoteHub sha256, fundamentals hash) in
separate fields. Test: same config + different data → same `model_config_hash`.

### Task 4.3 (P1-9, P1-6): Honest quality labels; two-party share
Flag races that use synthetic fundamentals. Label calibration as `identity_on_synthetic_backtest` instead of `fitted`, and the
backend as the one actually used (`model.yaml:40` says `nuts`; production uses analytic). Optional (M effort): fit one latent D
two-party share per race instead of independent D and R options (P1-6).

### Task 4.4 (P2-2/3): Lint, format, CI
Fix the 5 ruff errors (`scripts/generate_senate_panel.py:474` F841, `models/fundamentals.py:265` E501,
`normalize/builder.py:1659` E501 and `:1777` RUF010, `scoring/backtest.py:793` E501) and run `ruff format`. Add
`.github/workflows/ci.yml` (uv sync, ruff check, ruff format --check, pytest with the coverage gate). Optional: pyright on `src/`.

### Task 4.5 (P2-4): Edge-case tests
Cover one-sided/same-party polls, minor-party candidates, nominee filtering, duplicates, specials (FL/OH), independents (NE),
House at-large ids (AK-01, VT-01, ...), and unit tests for `scripts/compare_runs.py` and `scripts/package_run.py`.

### Task 4.6 (P2-5/6): Script bugs
- `scripts/package_run.py` doesn't write the top-level `runs/<kind>/<date>/summary.json`. Add it (combined Senate + House +
  week-over-week block).
- `scripts/compare_runs.py`: when nothing moved, the top movers list zero-delta races alphabetically. Require |Δ| > 0 and
  report "no significant movers" when none has |z| > 2 (MC SE-based). Fix the per-DEM-option "polled" status (mislabels AL, AK-01).
- Remove hard-coded "2026-11-02" and the FL/OH caveat text (`package_run.py` ~216, ~502; compare_runs.py).

### Task 4.7 (P2-7/9): Stale docs and dead paths
README (~:632, ~:1050ff) and `configs/sources_live.yaml` still point to the FiveThirtyEight mirror. Mark them deprecated or update
them to VoteHub. Document the live run procedure (the commands in §0), digest fields, and the synthetic-baseline caveat. Remove or
document the `models/ecosystem.py` stub.

### Task 4.8 (P2-8): Licensing and attribution
Add a header/README note that `fixtures/candidates_2026_parties.csv` is derived from Wikipedia (**CC BY-SA 4.0**, ShareAlike) and
add a VoteHub CC BY 4.0 attribution next to `tests/golden_fixtures/votehub_race_polls_sample.json` (e.g., a `.SOURCES.txt`).
Repo code stays MIT.

---------------------------------------------------------------------------------------------------
## Phase 5: before/after report (required final step)

### Task 5.1: Method-driven deltas, reported separately from poll movement
1. Run both chambers with the fixed code on **the same raw VoteHub payloads** as the §2 baseline → `runs/adhoc/after-fixes`.
2. `uv run python scripts/compare_runs.py runs/adhoc/before-fixes runs/adhoc/after-fixes --out runs/adhoc/after-fixes/comparison_vs_before-fixes --plot runs/adhoc/after-fixes/movers_dumbbell.png`
3. Optionally attribute deltas per phase (run after each phase on the same payloads).
4. Then, separately, run on **fresh** VoteHub data and compare after-fixes(old data) → after-fixes(new data): that's poll movement.
5. Write `docs/codex_fix_impact.md` covering: Senate D control (with/without Osborn) and House D control before → after; seats
   p10/p50/p90; the biggest race movers (old P, new P, Δ, which task caused it); poll accounting before → after (used,
   dropped-unmapped, dropped-same-party, duplicates, non-nominee); MC SE at the new sim counts; and all open `TODO(verify)` items.
   Never invent poll numbers. Quote pollster, dates and margin from the payload.

---------------------------------------------------------------------------------------------------
## Definition of done
- All tasks either committed or explicitly deferred with a reason in `docs/codex_progress.md`.
- `uv run ruff check`, `uv run ruff format --check`, and `uv run pytest --cov=src/civic_signal --cov-fail-under=90` pass.
- `docs/codex_fix_impact.md` exists. The branch is pushed (not main). No PR unless asked.
