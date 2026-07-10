# Civic Signal Specification

## Summary

Civic Signal is a U.S.-only, research-grade election forecasting engine that can be run
manually from time to time. Each run refreshes public data incrementally, snapshots
source provenance, builds a race catalog, runs a hybrid statistical ensemble, backtests
trusted components, and emits auditable artifacts with measurable rewards.

The default implementation is fixture-backed so the full modeling, artifact, reward,
plotting, and validation contract is deterministic. The opt-in live registry
(`configs/sources_live.yaml`) adds keyless HTTP CSV polling ingestion for the 2020
Wisconsin presidential archive, 2026 Senate/Governor/House poll streams when upstream
FiveThirtyEight/Datasette rows exist, keyless FRED UNRATE macro fundamentals for the
compact 2026 multi-office smoke races, and neutral Wikipedia race-presence metadata.
The separate `configs/sources_public_web.yaml` registry is the production candidate
surface: it contains free HTTPS sources explicitly classified as `free_public_web`.
It extends only `configs/sources_official_results.yaml`, a production-only registry with
no fixture/generated inheritance. The first implemented official-history contract uses a
commit-pinned MEDSL House constituency CSV for 1976-2018 whose documented underlying source
is the Office of the Clerk. It materializes canonical `races` and `results` tables with
content-hash, parser, authority, citation, and retrieval lineage. MEDSL requests citation
and publishes no explicit data-license file in that repository; raw snapshots therefore
remain local and unredistributed while the manifest records that limitation and terms URL.
Freely issued API keys are permitted when auth mode, rate limits, and terms are recorded;
authentication is not treated as evidence of paid access. Local fixture registries are
never valid production inputs. The R16 gate also requires source-class, access-policy, and
terms-status fields in the source manifest.

## Required Run Outcomes

Every `forecast run` must create `artifacts/runs/<run_id>/` with:

- `race_catalog.parquet`: discovered races with Tier A/B/C status and tier reasons.
- `race_forecasts.parquet`: winner probabilities, vote-share means/medians, interval
  columns, model tier, data-quality flags, driver attribution, uncertainty explanation,
  source lineage, and model-config lineage.
- `forecast_draws.parquet`: simulation draws by race, option, turnout, vote share,
  winner flag, and correlated-error draw id.
- `control_forecasts.parquet`: seat/control probabilities, seat distributions, and
  tipping-point races.
- `ecosystem_forecasts.parquet`: turnout, demographic-composition support, ballot-measure
  support, and close-margin administrative-risk fields that are withheld unless explicitly
  enabled as experimental output.
- `source_manifest.parquet`: source ids, URLs/paths, retrieval timestamps, content hashes,
  parser versions, license/terms notes, status, and downstream usage.
- `selected_feature_lineage.parquet`: the exact selected poll, market, public-signal,
  and fundamentals row keys with canonical selection keys, snapshot/revision ids,
  selection predicates, source hashes, availability timestamps/basis, and forecast cutoff.
- `feature_lineage.json`: proof that filtering precedes revision selection and that at
  most one eligible snapshot exists per race/feature/option/horizon. It reports selected
  macro/finance/rating vintage counts and incumbent-relative economic-sign proof. Any
  feature type without complete observation/publication/availability/revision lineage
  keeps its leakage result null rather than claiming a pass.
- `as_of_audit.json`: a freshly recomputed temporal-integrity audit over selected feature
  lineage. It records future-eligible, missing-availability, implicit-proxy, and duplicate
  rows plus lineage-filter and adversarial feature/tier/forecast-fingerprint canaries.
  Production requires the adversarial canary scope to be `exact_publication_pipeline`;
  narrower deterministic component-center evidence remains research-only.
- `diagnostics.html`: top-line summary, paired Electoral College distribution and
  simulation swarm, scorecards, reward status, source coverage, model-quality section,
  and embedded plots.
- `reward_card.json`: machine-readable reward checks.
- `semantic_verification.json`: read-only semantic reconciliation evidence for the
  catalog, published probabilities, draws, controls, and source manifest.
- `methodology_snapshot.md`: model version, config, source coverage, and limitations.
- `model_card.md`: learned/configured/placeholder parameter status, component admission,
  backtest sample status, covariance status, and source coverage for the run.
- `silver_benchmark.json` and `silver_benchmark.html`: methodology-readiness comparison
  against public Silver/FiveThirtyEight forecast traits and source anchors, scored on
  the explicit four-tier `absent`/`scaffold`/`functional`/`production` scale for the
  configured run scope, not for unmodeled nationwide live coverage.
- `reproducibility_fingerprint.json`: stable artifact hashes excluding volatile
  retrieval/status fields, with same-run-id comparison status when available.
- `plot_manifest.json`: projection, calibration, trajectory, stability, model-quality,
  and benchmark plot index.
- `plots/`: static PNG diagnostics.
- `performance.json`: requested acceleration engine, actual engine, parallel mode,
  Numba availability, thread count, and simulation count.
- `recalibration_map.parquet`: persisted Platt/logit probability calibration map when
  a latest rolling-origin backtest map is available and applied to the run.
- `posterior_draws.parquet`: race-constrained Bayesian election-day latent-share
  posterior draws unless `forecast run --inference-engine kalman` is used.
- `state_space_trajectory.parquet`: Bayesian trajectory summaries by race,
  option, and date with model/source lineage hashes.
- `pollster_house_effects.parquet`: Bayesian empirical-Bayes pollster house-effect
  estimates used by the Bayesian polling bridge.
- `posterior_diagnostics.json`: Bayesian posterior diagnostics, draw count,
  parameterization, fallback status, and lineage hashes.
- `fundamentals_prior.parquet`: Election-Day fundamentals prior used by the
  Bayesian polling bridge.
- `seat_posterior.parquet`: draw-level seat/control posterior summaries. Senate,
  House, and Governor body-specific posterior files are emitted when those rows exist.
- `posterior_history.parquet`, `latest_daily_update.json`, and `updates/<as-of>/`:
  created by `forecast update` from a Bayesian anchor run.
- `timeout_failover_audit.json`: Phase 8 forced-timeout audit showing the configured
  Bayesian NUTS fallback order without marking the forecast itself as a fallback.
- `phase8_verification.json` and `visual_qa_checklist.json`: created by
  `verify run --scenario ...` after orchestrating the fixture-backed multi-office
  verification path.
- `methodology_readiness.json`: created by `verify readiness` under
  `artifacts/readiness/<run_id>/` to audit the Bayesian default-switch contract against
  dependencies, docs/config defaults, Phase 8 artifacts, reward gates, live-source scope,
  and rolling-origin legacy comparison evidence.
- `verification.json`: created by `verify run` after required artifact, schema, plot,
  and reward checks.
- `senate_joint_posterior.parquet`: Phase 4 office-methodology artifact when Senate
  posterior rows exist, with shared Senate environment, class effect, state deviation,
  and holdover-aware seat posterior summaries. NUTS runs label this as a decomposition
  of the fitted state-space draw stream; analytic runs label it as a bridge.
- `house_hierarchical_posterior.parquet`: Phase 5 office-methodology artifact when
  House posterior rows exist, with redistricting-era partition, state effects, district
  idiosyncrasy, sparse-district flags, and non-dense covariance method.
- `cross_office_posterior.parquet`: Phase 7 office-methodology artifact when at least
  two midterm offices share posterior draws, with national environment and per-office
  offsets on the common draw stream.
- `comparisons/<comparison_id>/`: optional forecast-vs-actual comparison artifacts
  created by `results compare`.

Forecasts must distinguish:

- Candidate races: president, Senate, House, governors, state offices, local offices,
  primaries, runoffs, and ranked-choice races where data permits.
- Ballot measures: yes/no vote-share and pass probability.
- Control outcomes: seat-count and governing-control distributions.
- Ecosystem outcomes: turnout and demographic turnout support. Recount and
  certification-delay proxies stay withheld unless the experimental close-margin proxy is
  explicitly enabled or replaced by a calibrated administrative-risk model.

## Verifiable Rewards

Use vector rewards so the system cannot hide weak behavior behind one aggregate score.
Reward-v2 (`configs/rewards.yaml`, `reward_card_v2.json`) is the promotion contract:
each reward is `pass`, `fail`, `insufficient_evidence`, or `not_applicable`; thresholds
live only in config; production verification **recomputes** rewards from primary
artifacts; `fail` and `insufficient_evidence` hard-block the `production` profile.
Default `publication_mode` is `research` until the production profile can pass.
`data audit --profile production` is the source/snapshot eligibility entry point.
`verify rewards --profile <name>` is the recompute entry point. Rewards `R0`–`R15` are
strengthened; `R16`–`R27` cover real-data exclusivity, as-of integrity, nested evaluation,
covariance recovery, hierarchy, poll identity, feature validity, joint coherence, atomic
publication, live-source resilience, benchmark superiority, and contract parity.

Legacy `reward_card.json` continues to emit a simplified `passed` field for existing
dashboards. Strengthened pass rules (summary):

- `R0_build`: `uv sync`, `ruff check`, `ruff format --check`, and
  `pytest --cov=src/civic_signal --cov-fail-under=90` pass.
- `R1_reproducibility`: fixed inputs and run config produce the same stable artifact
  fingerprint when the same `run_id` is rerun, excluding wall-clock retrieval metadata
  and incremental-sync status fields.
- `R2_provenance`: forecast rows trace to source hashes and model-config hashes.
- `R3_sync_integrity`: incremental sync fetches new/changed sources only, dedupes records,
  and records failures explicitly. Raw snapshots are keyed by source and SHA-256 content
  hash and retained in an append-only index. Curated deduplication first enforces the hard
  source-class order official/production > fixture > synthetic, then applies explicit
  source priority, retrieval time, source ID, and content hash; registry order cannot alter
  the winner.
- `R4_calibration`: backtests report Brier score, log score, calibration line, expected
  calibration error, interval coverage, learned ensemble weights, and the probability
  calibration transform applied to published marginal race probabilities.
  `verify historical-calibration` also writes a compact 2022 Senate/House/Governor
  audit under `artifacts/historical_calibration/<run_id>/` with per-office ECE gates for
  the Phase 4, Phase 5, and Phase 7 office-methodology plan. The optional
  `sources_historical_panels.yaml` registry expands that audit to production-dimension
  synthetic Senate and House panels without changing default fixture runtime.
- `R5_baseline_competition`: trusted ensemble beats or matches declared baselines on a
  rolling-origin holdout with enough historical rows; otherwise it is labeled
  experimental.
- `R6_component_admission`: polls, fundamentals, markets, and public signals enter trusted
  output only when ablation evidence supports them. Forecast runs apply the current
  rolling-origin admission artifact before ensemble weighting; components rejected by
  admission can still appear as diagnostics or priors, but are not counted as trusted
  ensemble inputs.
- `R7_sparse_honesty`: Tier C races are tracked but do not receive trusted probabilities.
- `R8_uncertainty_quality`: reported forecast intervals have empirical coverage within
  configured tolerance on rolling-origin historical samples large enough to trust.
- `R9_public_signal_discipline`: news/pageview/social-like signals remain experimental
  until leakage and ablation checks pass.
- `R10_explainability`: forecast rows include tier reason, data-quality flags, top
  drivers, component contributions, and uncertainty explanation.
- `R11_plot_contract`: calibration and projection plots are generated and referenced by
  `plot_manifest.json`.
- `R12_performance_contract`: forecast runs record the configured acceleration path and
  use the Numba engine when it is requested and available.
- `R13_posterior_quality`: Bayesian runs emit posterior diagnostics with sufficient
  draws, no divergences, and valid R-hat/ESS checks when MCMC diagnostics are available.
- `R14_calibrated_publication`: published probabilities either use a persisted
  recalibration map or demonstrate acceptable rolling-origin calibration without a map.
- `R15_daily_update_quality`: daily Bayesian updates pass strategy-specific quality
  gates and do not require a full refit.
- `R16_real_data_exclusivity`: production inputs contain zero synthetic/fixture rows.
- `R17_as_of_integrity`: selected records have explicit availability evidence satisfying
  `available_at ≤ as_of`; missing timestamps, event-date proxies, duplicate snapshot keys,
  a failed future-row canary, a stale audit, or a lineage-hash mismatch block production.
  Production additionally requires `scope=exact_publication_pipeline`: hostile future rows
  are injected into every nonempty time-varying table and the same selected-bundle,
  component, ensemble, posterior, simulation, race-probability, and control path must remain
  bit-for-bit stable. A narrower component-center fingerprint is research evidence only.
- `R18_nested_evaluation`: outer-cycle exact-pipeline evaluation with fold lineage.
- `R19_covariance_recovery`: one signed residual per race; PSD covariance; recovery tolerances.
- `R20_all_race_hierarchy`: control-bearing races in the joint model; unpolled propagation.
- `R21_poll_observation_identity`: canonical survey/question identity without double count.
- `R22_feature_validity`: one eligible snapshot per feature key; vintage-correct inputs.
- `R23_joint_outcome_coherence`: probabilities and draws reconcile to control outcomes.
- `R24_atomic_publication`: only a verified immutable attempt can replace the promoted
  pointer; the snapshot is create-once and its complete present-artifact hash set is
  validated on every production verification.
- `R25_live_source_resilience`: adapter canaries for empty/stale/malformed live feeds.
- `R26_benchmark_superiority`: preregistered “best evidenced” criteria for a named scope.
- `R27_contract_parity`: README/SPEC/config/schema/CLI claims agree.

Scientific CI (M7) recomputes property tests (simplex, option-order invariance,
interval ordering, covariance PSD, label symmetry, control reconciliation),
Numba/Python and serial/parallel numerical parity fingerprints, and mutation probes against
the actual reward-card schema, publication reconciliation, and calibrated-publication
reward evaluator. Required corruptions cover a removed schema-required field, duplicate
forecast keys, out-of-range probabilities, stale calibration lineage, and blocked-estimand
publication. Each probe must show that the checked-in verifier accepts a valid fixture,
rejects the corruption, and that a controlled predicate-removal mutant accepts that same
corruption. A missing family or surviving mutant is a required-suite failure.

R27 contract parity is generated from `configs/rewards.yaml`, evaluator method names,
README/SPEC text, and actual CLI decorators. Reward IDs and threshold IDs must match
exactly; profile and conditional rewards must be registered; every reward must have a
nonempty threshold and evaluator; and required verification, audit, nested-backtest, and
shadow CLI surfaces must both exist and be documented. Golden fixtures under
`tests/golden_fixtures/` and offline canaries also run by default. Entry point:
`verify scientific` → `artifacts/scientific/scientific_report.json` (`passed` fails closed
on the complete offline suite; `missing_required_suites` identifies incomplete mutation
evidence; `missing_optional_suites` records live free-web canaries and tiny NUTS smoke when
not requested). Bounded recovery smoke is separate: `verify recovery`
writes `artifacts/recovery/<run_id>/hierarchy_recovery.json`,
`covariance_recovery.json`, and `covariance_recovery.parquet`. Both recovery branches always
mark bounded synthetic evidence `production_sufficient=false` (insufficient for R19/R20
promotion without large-cycle evidence).

Shadow (M8) is a separate publication mode: scheduled forecasts under
`artifacts/shadow/<profile>/` with frozen preregistration, source-health monitors,
office/horizon scorecards, and `verify shadow` readiness. Shadow runs set
`publication_mode=shadow` and must not publish public production probabilities.
Production promotion still requires the production reward profile; a green shadow
window is a necessary pre-P4 operational gate, not a substitute for nested science.
Shadow readiness requires 60 clean consecutive days unless the checked-in profile names
the exact alternative predeclared window; an arbitrary short or unknown-profile window
cannot pass.

Primary baselines:

- Historical partisan lean/fundamentals only.
- Polling average where polls exist.
- Market-implied where liquid markets exist.
- Incumbent/party prior for sparse races.
- Previous-cycle swing baseline.

Canonical warehouse entities are source snapshot, poll survey, poll question, poll
revision, race, option, official result, fundamental snapshot, and market quote. Their
checked-in JSON Schemas live under `schemas/raw_contracts/` and
`schemas/curated_tables/`. The curated poll projection preserves all immutable revisions in
`poll_revisions.parquet` but admits only one deterministic highest-ranked revision per
question/race/option to model-facing `polls.parquet`.

Every HTTP source may declare a deterministic `required_columns` contract. A successful
non-empty payload is content-addressed and appended to the immutable snapshot index;
unchanged payloads preserve their original retrieval timestamp. Empty payloads, HTTP 429
responses, and required-column drift are recorded as `empty`, `rate_limited`, and
`schema_change`. When an immutable prior snapshot exists, refresh failure may produce only
`stale_reused` with the triggering condition in `refresh_status`. Production data audit
fails all degraded states and reports explicit zero fixture/synthetic counts; it accepts
`official_public` sources only when HTTPS, free-web access, reviewed terms, citation/license
metadata, and snapshot-index lineage are present.

## Repository Design

```text
civic-signal/
  pyproject.toml
  README.md
  AGENTS.md
  SPEC.md
  configs/
    sources.yaml
    model.yaml
    backtests.yaml
    tiers.yaml
    rewards.yaml
  fixtures/
    *.csv
  schemas/
    raw_contracts/
    curated_tables/
    artifact_contracts/
  src/civic_signal/
    cli.py
    config/
    ingest/
    normalize/
    storage/
    features/
    inference/
    models/
    observability/
    scoring/
    reports/
  tests/
    unit/
    integration/
    golden_fixtures/
  data/       # gitignored raw/cache/curated local lake
  artifacts/  # gitignored run outputs
  docs/
```

Important config contracts:

- `configs/sources.yaml`: default fixture source registry and parser metadata.
- `configs/sources_live.yaml`: opt-in live source overlay for HTTP CSV/text/API adapters.
- `configs/scenarios.yaml`: scenario filters and defaults such as 2024 presidential
  state-level runs.
- `configs/model.yaml`: model version, seed, simulation count, component weights,
  trusted-component flags, uncertainty settings, performance settings, and reward
  thresholds.
- `configs/tiers.yaml`: Tier A/B/C thresholds and sparse-race policy.
- `configs/backtests.yaml`: rolling-origin settings, as-of date sweep, metrics, and
  baselines.

Sync writes content-addressed raw responses plus `raw/snapshot_index.parquet`. The index
retains each distinct `(source_id, content_hash)` version with parser, URL, original
snapshot time, and latest check metadata. Unchanged content preserves its original
`retrieved_at`; `checked_at` records the refresh attempt and is excluded from scientific
fingerprints. Production data audit fails when a current source/hash is absent from the
index or its hash/parser/timestamp lineage is incomplete.

Current implementation note: the repo runs a rolling-origin component refit harness and
writes `rolling_predictions.parquet`, `component_admission.json`, `ensemble_learning.json`,
`probability_calibration.json`, `recalibration_map.parquet`,
`bayesian_hyperpriors.json`, and `residual_covariance.parquet`. The rolling-origin
harness evaluates multiple pre-election as-of cuts when data exists
(`T-90/T-60/T-30/T-7/T-1` by default). It must not certify `R5`, `R6`, or `R8` until the
historical race store reaches the configured sample threshold. Latest trusted backtest
artifacts under `artifacts/backtests/latest/` are consumed by later forecast runs. When
no promoted residual covariance exists for a scenario, `forecast run` uses the same-run
rolling-origin covariance it already evaluated as a provisional simulation input rather
than reverting to only configured national/region/office shock terms. The
component-admission runtime fallback is only for trusted components that are expected
but unavailable in the current forecast slice; it must not override learned admission
when rolling-origin evidence rejects every available component. In that case races stay
tracked without trusted probabilities until better evidence or sources are added. The
same harness supports `backtest run --inference-engine bayes` plus
`--bayesian-backend analytic|nuts` so the Bayesian bridge and production NUTS backend
can both be scored against the legacy Kalman path without changing global model
configuration.

`backtest nested` is the non-promoting R18 evidence surface. Each fold manifest must show
`max(inner_validation_cycles) < outer_cycle`, exclude the outer cycle from every fit set,
and bind training race IDs plus source hashes into a lineage digest. Inner folds alone fit
hyperparameters, simplex weights, and Platt calibration. Outer scoring refits the same
component models on pre-outer data and executes `EnsembleModel` and `SimulationEngine`;
Bayesian outer folds must pass polling posterior draws into simulation. The model-facing
outer bundle contains no held-out results; results are joined only after prediction.
Held-out result permutation must leave the training-lineage digest unchanged. The bounded
output must separately score uniform prior-only, previous-cycle swing, fundamentals-only,
eligible poll-average, and markets-if-present comparators. Missing market evidence remains
null/not-available rather than being filled by another baseline. Paired Brier and log-score
differences are collapsed to one mean per outer election cycle before a deterministic
nonparametric bootstrap resamples whole cycles with equal weight. Row-level resampling is
forbidden. `configs/backtests.yaml` sets the explicit minimum independent-cycle count; fewer
cycles are `insufficient_evidence` regardless of race-row count. The estimator writes
`baseline_scorecard.json` and `paired_cycle_clustered_uncertainty.json`. Nested execution still
cannot support a best-evidenced promotion claim while promoted real-data training-bundle
compatibility and the result-derived feature-injection canary remain insufficient.

`backtest refresh-hyperpriors` is the scheduled refresh surface for hyperprior drift.
It writes candidate artifacts under `artifacts/hyperprior_refreshes/<run_id>/`,
including `hyperprior_refresh_manifest.json`, scenario-local candidate hyperpriors, and
a comparison report. It must not update `artifacts/backtests/latest/`; candidate
promotion requires a separate explicit review. The refresh command accepts the same
Bayesian backend override as `backtest run`.

Phase 0 methodology spikes write `artifacts/spikes/<run_id>/comparison.json`,
`phase0_comparison.parquet`, per-engine rolling predictions, and per-engine scorecards.
The current spike compares the legacy Kalman and opt-in Bayes polling engines on the
configured presidential holdout cycle and records the Bayes-minus-Kalman log-loss gate.
Phase 0b methodology spikes write `phase0b_summary.json`,
`geometry_comparison.parquet`, and `acceleration_bakeoff.parquet`. That artifact is the
gate for centered-vs-non-centered posterior geometry and for accepting any global SMC
daily-update path. The configured production update strategy remains cached posterior
reweighting unless Phase 0b proves another non-global strategy dominates and records
fallback semantics.

## Modeling Specification

For the detailed statistical rationale, see
[`docs/technical_appendix.md`](docs/technical_appendix.md).

Canonical latent targets:

- Candidate races: latent election-day vote-share simplex and major-party margin.
- Ballot measures: latent yes-share and pass probability.
- Control: derived from correlated race-level simulations.
- Turnout: separate turnout-rate and vote-count projections by geography and election type.

Component models:

- Polling model: the Bayesian path is the production default polling engine in config for
  research/operational engineering runs; public-production publication remains gated by
  reward-v2. The legacy deterministic Kalman/state-space polling estimates remain available
  through `--inference-engine kalman`; they are initialized from previous vote share when
  available, with sample-size observation variance, methodology/population/sponsor
  effective-sample adjustments, iterative empirical-Bayes pollster house-effect
  shrinkage, and posterior uncertainty proxy. Forecast and backtest commands resolve
  their default inference engine from `configs/model.yaml`. The Bayesian bridge exports
  logit-normal posterior draws, state-space trajectory summaries, posterior diagnostics,
  and pollster house-effect artifacts behind the same component schema. Candidate
  offices without eligible polls may receive fundamentals-prior-only posterior draws so
  sparse House/Senate races still produce auditable uncertainty artifacts and sparse
  forecast rows. When same-office/geography polled races exist, one party-signed residual
  per observed race is partially pooled through global, office, and geography levels and
  applied as a shrunk logit shift to those fundamentals-only options. Complementary party
  rows are averaged within race before pooling, preventing cancellation and double count;
  `unpolled_pooling_prior_races` controls shrinkage toward zero. `verify recovery` exercises
  parameter recovery, label symmetry, this unpolled propagation, and a bounded SBC smoke
  through the real polling model. Its default artifact remains `insufficient_evidence` for
  R20 because a small synthetic analytic check is not a large real-NUTS hierarchy recovery
  study or proof that every control-bearing production race is present. The Bayesian
  backend defaults to compact hierarchical NumPyro/NUTS with two vectorized chains, 500
  warmup iterations, 2,000 sampling iterations per chain, and a `0.99` target
  acceptance probability; `--bayesian-backend analytic` selects the deterministic bridge
  for fast smoke runs. The JAX/NumPyro/ArviZ dependencies are base dependencies so the
  NUTS path is available after plain `uv sync`. The NUTS backend pools options through
  non-centered office, geography, and race-level effects plus pollster effects. Poll
  observations use one canonical survey/question identity, explicit methodology and sponsor
  bias terms, mode/sponsor/nonsampling variance, a 7-day recency half-life, population
  screen, and poll-age process variance. Two-party option rows from one question form one
  shared contrast rather than independent evidence. Undecided/other mass is proportionally
  excluded from the two-party estimand and increases nonsampling uncertainty. Polling
  estimands with more than two modeled options are withheld until a coherent K-category
  likelihood replaces the diagonal approximation; simplex normalization alone is not a
  likelihood. Incomplete binary questions are discarded rather than treated as independent
  option evidence. Either unsupported status is a race-level trust gate: the forecast catalog
  preserves `original_tier`, sets `estimand_support_blocked`, downgrades the race to Tier C,
  and withholds winner probabilities even if fundamentals or markets exist. Semantic
  publication verification recomputes this withholding invariant from the catalog and
  forecast artifacts. The
  exported posterior draw artifact inflates that state from `as_of` to election day with
  `bayesian.state_space.forecast_drift_sd_per_sqrt_day` and constrains all options
  within each race to sum to one before ensemble calibration.
- Fundamentals model: historical vote share, partisan lean, incumbency, finance, economy,
  demographics, turnout history, and election type through a standardized ridge fit when
  enough prior-cycle rows exist, otherwise explicit defaults. Macro, candidate-finance,
  and rating snapshots are filtered on observation, publication, and availability before
  deterministic revision selection and before any feature is constructed. Option-level
  finance/rating vintages then overlay static option records; undated values are not used.
  Economic conditions are first signed to the incumbent party and then multiplied once by
  the modeled option's party sign. For an open major-party race, an explicit
  `incumbent_party` is required; otherwise the economic contribution is neutral and the
  lineage proof is incomplete. Ballot/nonpartisan races treat the sign as not applicable.
  Bayesian runs convert the fitted fundamentals model into an Election-Day prior artifact.
- Market model: public read-only market probabilities adjusted for liquidity and spread,
  then mapped to vote-share proxy through a configurable normal inverse-CDF scale.
- Public-signal model: news/pageview/official-release features, experimental by default.
- Ensemble: weighted blend of trusted components with vote-share normalization by race,
  component-disagreement tracking, rolling-origin simplex weight learning when the
  backtest is trustworthy, calibrated marginal winner probabilities, and a persisted
  recalibration map when a latest rolling-origin calibration artifact is available.
  If a learned trusted component has no current estimates for the forecast scope, the
  run records a runtime admission fallback and uses the first available component in
  polling/fundamentals/markets/public-signals order rather than publishing an all-null
  forecast.
- Simulation: structured-factor election-error draws. When a residual covariance artifact
  is available, that covariance replaces the configured national/region/office layers;
  covariance fitting uses one consistently signed reference-party residual per
  race/cycle and averages repeated horizons, never complementary candidate-option errors;
  the persisted covariance is an explicit configurable-rank, shrinkage-regularized
  `B diag(v) B' + diag(d)` representation with nonnegative variances, so it is PSD by
  construction. Simulation consumes those exact factor loadings, variances, and diagonal
  terms rather than a separately approximated matrix;
  otherwise the engine falls back to national, region, and office factors plus
  heavy-tailed local error. Bayesian posterior draws seed the race-level simulation
  distribution for Bayesian races; their log-ratio deviations are recentered on the
  admitted ensemble vote-share target so the ensemble controls the forecast center while
  the posterior retains its uncertainty. The simulator then applies national, region, office,
  and heavy-tailed local forecast-error layers. Race-level winners, vote shares, and
  turnout are always emitted; thresholded control outcomes are emitted only for races with a
  configured `control_body`, so non-control tracker rows can participate in posterior
  and cross-office artifacts without changing seat-count math.
  Simulation must adapt deterministically from the configured initial draw count in fixed
  batches until the maximum raw empirical race/control binomial MCSE, computed before
  probability calibration, is at most `0.0025`, or
  stop at the configured cap and record a blocking non-convergence status. Bayesian
  posterior draw IDs are deterministically cycled when more simulation draws are required;
  adaptive stopping must never disable posterior use.
- Daily update: `forecast update --from-anchor <run_id> --as-of <date>` appends
  posterior summaries from a Bayesian anchor run, writes update diagnostics, and refreshes
  `R15_daily_update_quality`. Eligibility is computed from immutable poll lineage:
  `anchor_as_of < available_at <= update_as_of` and poll event time must be no later than
  the update. The exact selected revision/source rows are persisted with the update. The
  current implemented strategy is the Phase 0b-selected cached-posterior likelihood
  reweighting path with deterministic-seeded, lower-variance systematic resampling.
  A binary poll question contributes one positive/reference-party likelihood contrast;
  its complementary D/R row is counted in the contrast audit but is not persisted as a
  used likelihood row or treated as a second independent observation.
  Multi-option questions use K-1 diagonal share likelihood terms relative to a dropped
  category as an explicit approximation; cross-option simplex covariance remains pending.
  It records ESS, degeneracy, and anchor-age diagnostics; anchor age beyond
  `full_refit_days_since_anchor` forces `needs_full_refit`. Pareto-k remains explicitly
  unavailable until PSIS is implemented. Strategy/resampling config values that do not
  name literal implemented algorithms are rejected. No-new-poll updates are no-ops, and R15 remains
  insufficient until an exact update-vs-full-refit publication-path audit supplies measured
  probability differences. When that comparison is not executed, MAE and max-diff remain
  null with `status=unavailable` (never a fabricated pass). When a fixture/audit supplies
  a full-refit posterior, the comparison records measured MAE/max-diff and
  `status=passed|failed`. Full SVI or SMC strategies remain gated until Phase 0b accepts
  them for the target scope.
- NUTS failover: `bayesian.nuts.wall_clock_timeout_seconds` and
  `bayesian.nuts.failover.fallback_order` define production timeout semantics. The ordered
  dispatcher may literally execute `previous_posterior_reuse`,
  `analytic_logit_normal_fallback`, `kalman_fallback`, or terminal `refuse`; its audit must
  distinguish skipped/unavailable/incompatible/failed attempts from the one executed path.
  Previous-posterior reuse requires a readable posterior artifact whose schema, unique
  draw/race/option keys, finite shares, model-config hash, source-manifest hash, artifact
  as-of age, and exact race-option lineage all match the declared current contract. SVI is
  not implemented and any SVI fallback label is rejected. Executed fallbacks retain
  `fallback_used` and are publication-quarantined by posterior-quality rewards; `refuse`
  raises and emits no forecast. Phase 8 exercises the timeout policy on a fixture and
  records that audit separately from the forecast-level `fallback_used` field.
- Performance: two-option race draw generation uses a Numba parallel kernel with a Python
  fallback, while table transforms should stay vectorized through Polars/DuckDB.
- Live-source readiness: Phase 8 records live 2026 scope from the actual curated source
  manifest and curated tables. It may only report `claimed` when successful non-file
  sources contribute model-bearing target-year rows for every expected verification
  office. Neutral race-presence or other metadata-only rows must be reported separately
  as `metadata_only` and must not unlock a production-default switch.
- Production promotion: the Bayesian path is the production default polling engine in
  config for operational research forecasts, but artifact `publication_mode` defaults to
  `research`. A run may be labeled public production only after reward-v2 production
  profile recomputation passes and a verified `promotion_manifest.json` is written. Fixture
  Phase 8 success proves orchestration only.

Statistical upgrade path:

- The hierarchical NumPyro/NUTS state-space contract is documented in
  [`docs/technical_appendix.md`](docs/technical_appendix.md) §4.6 and is the
  production polling component. The analytic backend remains available as a fast
  smoke-run bridge through `--bayesian-backend analytic`.
- The production NUTS backend emits office-specific decomposition artifacts from
  the fitted shared state-space draw stream. The NUTS backend receives the same
  fitted fundamentals-prior logit means as the analytic bridge, so prior
  construction is shared across Bayesian backends.
- Future work: a richer hierarchical calibration model that can replace the
  rolling-origin simplex/Platt layer once the historical panel is deep enough.
- Until that replacement exists, keep Platt/logit recalibration slope-bounded at
  `ensemble_learning.calibration_max_slope: 1.0` for publication so the calibration
  layer cannot sharpen probabilities from sparse historical panels.
- Extend live source adapters while preserving raw-source hash and curated-table
  contracts.
- Extend Numba kernels to multi-option/ranked-choice simulation and score aggregation when
  those paths become measurable bottlenecks.

## Plotting Specification

Every forecast run must emit calibration and projection visuals:

- Calibration curve.
- Brier score by component.
- Historical interval coverage.
- Winner-probability bars.
- Vote-share interval projections.
- Seat/control projections.
- Turnout/recount-risk projections when calibrated or explicitly enabled proxy fields are
  available.
- Forecast coverage by tier.
- Electoral College distribution and representative simulation swarm for presidential
  scenarios. These two top-line presidential views should render together in the lead
  diagnostics summary, not only in the lower projection grid.
- Polling probability trajectories when rolling-origin polling probability and as-of
  cut columns are available.
- Simulation probability convergence when draw-level winner rows are available.
- `verify coherence` recomputes probability ranges and simplices, key uniqueness, Tier C
  withholding, draw-level winner/share coherence, Maine/Nebraska elector allocation,
  Senate tie-to-VP thresholds, and chamber-control probabilities reconstructed from draws.
  Maine and Nebraska require explicit statewide two-elector plus district one-elector rows;
  aggregate-only state rows cannot satisfy this check.
- MCMC-style posterior simulation chain traces for Electoral College totals.
- Kalman posterior uncertainty traces for state-space polling fits.
- Bayesian posterior latent-share intervals and posterior diagnostics when
  `--inference-engine bayes` is used.
- Fundamentals-prior interval plots when Bayesian runs emit `fundamentals_prior.parquet`.
- Daily posterior history panels once `forecast update` has produced
  `posterior_history.parquet`.

Plots are generated from local artifacts and do not require API credentials.

## Performance Specification

Performance settings live under `performance` in `configs/model.yaml`:

- `engine`: requested acceleration engine, currently `numba` or `python`.
- `parallel`: enables parallel Numba execution.
- `numba_threads`: optional positive thread cap; `0` uses Numba's default.
- `benchmark_draws` and `benchmark_repeats`: benchmark CLI defaults.

Benchmark command:

```bash
uv run civic-signal benchmark run --as-of 2026-05-08 --run-id perf
```

Benchmark output:

```text
artifacts/benchmarks/<run_id>/performance_benchmark.json
```

This benchmark is a simulation-throughput contract, not a sampler benchmark. The setup
ensemble uses deterministic Kalman polling so the reported throughput isolates
`SimulationEngine` and the configured Numba/Python draw backend even when the production
forecast default is Bayesian/NUTS.

## Historical Result Comparison

An existing forecast run can be compared against curated actual results:

```bash
uv run civic-signal results compare \
  --forecast-run-id 2024-presidential \
  --comparison-id 2024-presidential-actuals \
  --cycle 2024 \
  --office-type president
```

Comparison output:

```text
artifacts/runs/<forecast_run_id>/comparisons/<comparison_id>/
  result_comparison.parquet
  race_outcomes.parquet
  largest_misses.parquet
  result_comparison_summary.json
  result_comparison.html
  narrative.md
  plots/
```

The comparison reports winner accuracy, mean absolute vote-share error, Brier score, and
upset count over the filtered races/options. Presidential comparisons also report
state-level winner accuracy, modeled Electoral College winner accuracy with an explicit
`full_electoral_college` or `modeled_state_slice` scope, actual-winner probabilities,
and the largest option-level vote-share misses.

A same-date presidential cycle evaluation should be available as a first-class command:

```bash
uv run civic-signal results cycle-eval \
  --run-id oct5-presidential-cycle-eval \
  --cycles 2008,2012,2016,2020,2024 \
  --as-of-mm-dd 10-05
```

Cycle-eval output:

```text
artifacts/cycle_evals/<run_id>/
  cycle_summary.parquet
  cycle_summary.json
  cycle_eval.html
  narrative.md
  plots/
```

The summary must report simulated/control Electoral College winner probability, EV
p10/p50/p90, simulated/control EC winner accuracy, deterministic state-topline EC winner
as an audit field, state accuracy, Brier score, vote-share MAE, upset count, missed
states, and links to each cycle's diagnostics and comparison report.

Cycle evaluation must validate all requested scenario keys and cycle-specific dates
before starting any forecast. It may support explicit artifact reuse, but reuse must be
operator-selected rather than silent.

## API Credentials

No external credentials are required for fixture-backed runs, backtests, or plots.

Likely live-ingestion credentials:

- `GOOGLE_CIVIC_API_KEY` for Google Civic Information API.
- `CENSUS_API_KEY` for higher-volume Census API usage.
- `GDELT_API_KEY` for GDELT Cloud endpoints that require bearer auth.

Usually public/read-only first:

- Polymarket market/event data.
- Kalshi public market data.
- Wikimedia pageviews.
- FEC and official election-office downloads where available.

Every live adapter must re-check current terms/rate limits and write auth mode, URL,
retrieval time, content hash, parser version, and failures to the source manifest.

## Acceptance Criteria

The repo is healthy only when all of these pass:

```bash
uv sync
uv run ruff check
uv run ruff format --check
uv run pytest --cov=src/civic_signal --cov-fail-under=90
```

A working forecast smoke test is:

```bash
uv run civic-signal forecast run --as-of 2026-05-08 --run-id smoke
uv run civic-signal verify run --run-id smoke
uv run civic-signal benchmark run --as-of 2026-05-08 --run-id perf
```

An explicit Bayesian smoke test is:

```bash
uv run civic-signal forecast run \
  --as-of 2026-05-08 \
  --run-id bayes-smoke \
  --inference-engine bayes \
  --bayesian-backend nuts \
  --quiet
uv run civic-signal forecast update --from-anchor bayes-smoke --as-of 2026-05-09
uv run civic-signal verify run --run-id bayes-smoke
```

A Phase 0 methodology spike smoke test is:

```bash
uv run civic-signal spike phase-0 \
  --scenario president_state \
  --holdout-cycle 2024 \
  --run-id phase0-smoke \
  --bayesian-backend nuts
```

A Phase 0b acceleration spike smoke test is:

```bash
uv run civic-signal spike phase-0b --run-id phase0b-smoke
```

A fixture-backed Phase 8 multi-office smoke test is:

```bash
uv run civic-signal verify run \
  --scenario 2026-multioffice-verification \
  --run-id phase8-smoke \
  --as-of 2026-05-08 \
  --inference-engine bayes \
  --quiet
```

The same Phase 8 harness can exercise the compact hierarchical NumPyro/NUTS backend:

```bash
uv run civic-signal verify run \
  --scenario 2026-multioffice-verification \
  --run-id phase8-nuts-smoke \
  --as-of 2026-05-08 \
  --inference-engine bayes \
  --bayesian-backend nuts \
  --quiet
```

The Phase 4/5/7 historical calibration gate is:

```bash
uv run civic-signal verify historical-calibration \
  --run-id midterm-2022-calibration \
  --bayesian-backend nuts \
  --quiet
```

It writes per-office Senate, House, and Governor calibration metrics plus explicit
Phase 4, Phase 5, and Phase 7 gate results. This is a compact fixture gate; production
claims still require a broader historical panel.

For production-dimension synthetic Senate and House coverage:

```bash
uv run civic-signal verify historical-calibration \
  --run-id historical-panels-2022-nuts \
  --sources-config sources_historical_panels.yaml \
  --data-dir data/historical-panels-nuts \
  --artifacts-dir artifacts/historical-panels-nuts \
  --bayesian-backend nuts \
  --quiet
```

A production-default readiness audit is:

```bash
uv run civic-signal verify readiness \
  --run-id bayes-default-readiness \
  --forecast-run-id phase8-smoke \
  --bayes-backtest-run-id president-state-bayes-backtest \
  --legacy-backtest-run-id president-state-backtest
```

The smoke run must create all required Parquet/JSON/HTML/Markdown/PNG artifacts, and
`verify run` must pass artifact/schema/plot checks. The reward card must record every
implemented reward state. `R0_build` is validated by the external commands above, `R1`
passes only after a same-run-id reproducibility rerun, and experimental component gates
may fail only when the failure is explicit in `reward_card.json` and `diagnostics.html`.
