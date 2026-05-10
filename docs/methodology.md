# Methodology

This engine estimates a joint distribution over election outcomes:

$$
\Pr(\boldsymbol{\theta}, \mathbf{W}, \mathbf{S} \mid \mathcal{D}_{\le a})
$$

where `theta` is race-level vote share, `W` is the winner indicator, `S` is
seat/control or Electoral College count, and `a` is the forecast `as_of` date.

The current implementation is an auditable approximation to that posterior:

1. **Polling**: The Bayesian path is the production default. Forecast and backtest
   commands resolve their default inference engine from `configs/model.yaml`; use
   `--inference-engine kalman` to force the legacy Gaussian state-space/Kalman polling
   model. The default Bayesian backend is compact hierarchical NumPyro/NUTS with
   non-centered office, geography, race-option, and pollster effects. The analytic
   logit-normal bridge remains available through `--bayesian-backend analytic` for fast
   smoke runs. JAX/NumPyro/ArviZ are base dependencies, so both Bayesian backends are
   available after plain `uv sync` and consume the fitted fundamentals prior as their
   initial Election-Day logit state.
2. **Fundamentals**: standardized ridge model over prior share, partisan lean, economic,
   demographic, incumbency, and finance features when enough historical rows exist;
   otherwise an explicit fallback prior.
3. **Markets**: public read-only market probabilities gated by liquidity and spread,
   then mapped to vote-share proxy through an inverse-normal transform.
4. **Public Signals**: news/pageview/official-release signals, experimental by default
   and admitted only after leakage and rolling-origin ablation checks.
5. **Ensemble**: weighted component blend over admitted components, with contribution
   attribution retained by race and option. Trusted rolling-origin backtests learn
   non-negative simplex weights and a bounded Platt/logit calibration transform that is
   applied to published marginal race probabilities. The default slope cap is `2.0`,
   deliberately lower than the earlier experimental cap so calibration can correct
   residual bias without turning close races into overconfident publications.
6. **Simulation**: correlated draw engine with either learned geography residual covariance
   or configured national/region/office factors, plus heavy-tailed local errors. Local
   uncertainty includes a component-disagreement term so divergent model inputs raise
   simulation variance.

The rigorous mathematical contract lives in
[`technical_appendix.md`](technical_appendix.md). That document also identifies which
parts are implemented approximations and which remain frontier targets. Under the
production NUTS backend, Senate, House, governor, and cross-office artifacts are
office-specific decompositions of the shared fitted state-space posterior. Under the
analytic backend, those same artifacts remain explicitly labeled as bridge summaries.

Live-source readiness follows the same distinction. Keyless Wikipedia race-presence
rows are accepted as neutral public-signal metadata for provenance and scope auditing,
but they are not model-bearing observations while public signals remain untrusted. The
FRED UNRATE adapter is model-bearing because it contributes a live national macro
fundamentals signal; the remaining race-specific priors in that adapter are configured
assumptions. The Bayesian path is the production default in config for operational
forecasts, and the current broad live-scope rolling-origin comparison is eligible:
Bayes/NUTS beats the legacy Kalman ensemble log score without coverage degradation.
NUTS observations use the same empirical-Bayes pollster house-effect adjustment as the
analytic bridge, and Bayesian component probabilities are normalized within each race
before ensemble calibration.

The office-methodology calibration check is executable through
`verify historical-calibration`. It runs a compact 2022 Senate/House/Governor fixture,
writes per-office ECE/Brier/log-score metrics, and gates the Phase 4 Senate, Phase 5
House, and Phase 7 cross-office acceptance criteria. The fixture is intentionally small:
it validates the audit path and should be treated as smoke evidence until a
production-sized historical panel is available.
For production-dimension synthetic evidence, run the same command with
`--sources-config sources_historical_panels.yaml`; that registry loads full Senate and
House panels for 2014-2026 while keeping default runs compact.
