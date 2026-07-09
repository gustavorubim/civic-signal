# Methodology Review — July 2026

Full-stack audit of the statistical methodology: state-space/NUTS polling model,
analytic bridge, fundamentals prior, ensemble stacking, calibration, markets,
simulation engine, and residual covariance estimation. Findings are ordered by
priority. P0 items are structural defects; P1 items materially distort
uncertainty; P2 items are component-level science upgrades.

---

## P0 — Structural defects

### P0.1 Hierarchical effects are not party-signed, so they cannot move races
`inference/state_space.py::state_space_model` adds office, geography, and race
effects to **every option's logit identically** (they are indexed by race /
geography / office, and both candidates in a race share all three). A common
shift applied to both candidates' logits does not shift the race — after
within-race share normalization it only **compresses or stretches the margin
symmetrically** (verified numerically: a D=60/40 race becomes 57.6/42.4 under a
+0.5 common effect and 62.3/37.7 under −0.5, regardless of party).

Consequences:
- The hierarchy cannot express "the Midwest swung toward Democrats" — the
  single most important correlation structure in an election model.
- It injects direction-less margin noise instead, a spurious mean-reversion
  channel.
- Cross-race correlated polling error via the hierarchy is impossible as
  specified; all cross-race correlation currently comes from the simulation
  layer's separate systematic-error draws.

**Fix**: reparameterize the latent state as a **party-signed margin** — one
latent per race (logit of the D share of the two-party vote), with office /
geography / race effects entering that margin directly. Multi-way races get a
log-ratio parameterization against a reference option. This also fixes P0.2.

### P0.2 D and R are modeled as independent latents observed independently
Each (race, option) is an independent latent with its own poll observations.
The two shares of the same poll are treated as independent Gaussians — the
poll's information is double-counted and the near-perfect negative correlation
between D and R readings is ignored. Sum-to-one is imposed post-hoc in
`_posterior_frame`. The margin parameterization from P0.1 resolves this: one
observation per poll (the margin), one latent per race.

### P0.3 Simulation correlated errors are keyed to option sort order, not party
`models/simulation.py` + `performance/kernels.py` add national/regional/office
errors to whichever option sorts **first by `option_id`**. This behaves as a
pro-D national swing only because "-D" sorts before "-R" alphabetically. A race
whose first-sorted option is an independent receives the "national partisan
swing" on the independent. **Fix**: explicit party sign convention (+1 DEM
reference, −1 REP, 0 or separate factor for others).

### P0.4 The national vote-share shock doubles as the turnout multiplier
`turnout = base × max(0.6, 1 + national_error)`: a pro-D vote-share shock
mechanically raises turnout everywhere (and an R-wave lowers it), which has no
substantive justification and contaminates turnout intervals. **Fix**: separate
turnout shock (its own national + local components).

---

## P1 — Uncertainty structure

### P1.1 No time dynamics — `poll_t` is built and never used
The "state-space" model has a static latent per race; time enters only through
exponential recency down-weighting (half-life 7d) and noise inflation
∝ √age. A race trending steadily toward one candidate is estimated at its
recency-weighted mean — systematically behind the trend, with no way to
extrapolate momentum or express "the race is moving." **Fix**: reverse random
walk anchored on election day (Linzer 2013 / 538 structure). The tensors
(`poll_t`, drift constants) already exist; only the NumPyro model needs the
walk.

### P1.2 Forecast-horizon uncertainty is a hand-set constant, not calibrated
`forecast_drift_sd_per_sqrt_day = 0.006` (logit) gives ≈1.6pp of share drift at
120 days out, well below empirical poll-error-vs-horizon curves (~4-5pp at 4
months for Senate races). Tier floors in the simulation partially compensate —
but that conflates two different uncertainties. **Fix**: estimate drift-by-
horizon empirically from the rolling-origin backtests (the T-90/T-60/…/T-1
harness already produces exactly the data needed) and feed it back as a
promoted hyperprior, like the covariance promotion flow.

### P1.3 Residual covariance is nearly unidentified and the sample is inflated
`scoring/backtest.py::_residual_covariance` treats each (cycle, as_of) cut as an
independent observation. As-of cuts within a cycle share the same election
outcome, so their residuals are strongly dependent — effective n is the number
of cycles (6), not cycles × cuts (~36). The empirical covariance over ~50 state
groups from 6 observations is rank-deficient noise; the 0.60 shrinkage toward
the two-parameter regional target is doing all the work. **Fix**: be structural
and honest — estimate a factor model (national + 4 regional factors + state
loadings) with only the factor variances free, fit on cycle-level residuals,
and document the effective sample size. Same dependence issue applies to
ensemble weight learning and Platt fitting on stacked as-of cuts — cluster or
subsample by (cycle, race).

### P1.4 Platt calibration slope is capped at 1.0 — can only flatten
`calibration_max_slope: 1.0` means calibration can shrink confidence but never
sharpen it. Combined with tier sigma floors this bakes in permanent
underconfidence that no amount of backtest evidence can correct. **Fix**: allow
slope > 1 (e.g. cap 1.5) gated on leave-one-cycle-out validation.

### P1.5 National error sigma is a guess where it matters most
`national_sigma: 0.015` (1.5pp of share) governs the fattest tail in seat-count
space when residual covariance is absent, and empirical uniform-swing misses
run 2.0–2.5pp. Validate against historical generic-ballot-vs-outcome misses;
promote from backtests instead of config.

---

## P2 — Component-level upgrades

### P2.1 Fundamentals regression (for real data)
Current: ridge on 5 features with the previous-share coefficient pinned at 1.0
(target = actual − previous). Adequate for the synthetic panel; for real data:
- free mean-reversion coefficient on previous share (historically ~0.7–0.9);
- incumbency split by open seat vs running incumbent (retirement effects);
- interaction of `national_swing` with district elasticity;
- midterm presidential-party penalty as an explicit term;
- keep the leave-one-cycle-out predictive variance — that part is right.

### P2.2 Multi-option error model is ad hoc
`_apply_multi_option_error` perturbs centered log-shares scaled by their spread,
then clips to [0.02, 0.98] and renormalizes — magnitude depends on the share
configuration in an unprincipled way. Also `_multi_option_shares` uses
Dirichlet(α = 70·share) with a hard-coded 70. **Fix**: logistic-normal errors on
log-ratios vs a reference option, concentration tied to the posterior sd.

### P2.3 House effects: identification and double handling
Empirical-Bayes house effects are subtracted from observations at data build
AND the model refits `pollster_effect` on the adjusted data. Sum-to-zero across
pollsters assumes the average pollster in the window is unbiased — the classic
identification gap (2016/2020-style correlated industry error is invisible).
Keep one mechanism (the model's), and anchor pollster quality on historical
accuracy vs certified results once real data lands.

### P2.4 Poll quality weights are hand-set; pollster track record unused
Population/methodology multipliers (0.85–1.1) are folklore constants. With live
data, weight pollsters by historical accuracy and herding-adjust (538-style)
rather than by mode alone.

### P2.5 Markets: favorite-longshot bias set to zero
`favorite_longshot_bias: 0.0` — prediction markets have a documented longshot
bias; calibrate the debiasing curve on resolved markets before trusting the
component at weight 0.20.

### P2.6 Senate control threshold ignores the VP tiebreak
`control_thresholds.senate: 51` for both parties; 50 seats suffice for the
Vice President's party. Threshold should be party-conditional.

### P2.7 NUTS `target_accept_prob: 0.99`
Unusually high — inflates step-count and wall-clock (which matters because the
timeout triggers fallbacks that block publication). 0.9 is standard with a
noncentered parameterization; reserve 0.99 for diagnosed divergences.

### P2.8 Ensemble mixture uncertainty
Ensemble uncertainty = weighted mean of component sds + separate disagreement
term. The proper mixture variance is Σw(σ²+μ²) − μ̄²; the current heuristic
undercounts when components disagree strongly. Minor, but easy to fix when the
disagreement term is revisited.

---

## Prerequisite: data before science

Every item above is second-order to the input problem: the panels are
synthetic. The single biggest upgrade to forecast quality is real inputs —
MIT Election Lab / Daily Kos results for fundamentals and district baselines,
plus a living 2026 poll feed (the 538 Datasette mirror is a frozen archive
with zero 2026 rows). The model card should continue to say so until then.

## Suggested sequencing

1. **P0.3 + P0.4** (simulation sign convention, turnout decoupling): small,
   isolated, immediately correct.
2. **P0.1 + P0.2** (party-signed margin latent): one reparameterization fixes
   both; touches `state_space.py`, `_posterior_frame`, and the prior plumbing.
3. **P1.1** (reverse random walk) on top of the new latent.
4. **P1.3 + P1.2 + P1.5** (honest covariance + backtest-calibrated horizon
   drift and national sigma): mostly `scoring/backtest.py`.
5. **P1.4 + P2.x** opportunistically.
