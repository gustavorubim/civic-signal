# Simulation error model (audit P1-3)

Each simulated election-day D two-party share is

    share = posterior draw (race-level error)  +  sign x (national + regional + office error)

* **Race-level error** comes from the polling/fundamentals posterior draws: poll sampling and
  nonsampling error, `bayesian.state_space.election_day_extra_sd`, and a floor of
  `race_forecast_floor_sd` (0.02 share) per option. With posterior draws in use,
  `uncertainty.posterior_local_sigma_scale: 0.0` adds no second layer of local noise. Before
  this change a heavy-tailed tier-floor noise (A 3.5 / B 7.5 pts) went on top.
* **National error**: `correlation.national_sigma: 0.025` share (5 pts of margin), one draw
  shared by every race (signed toward DEM).
* **Regional error**: `region_sigma: 0.015` share per Census region
  (`geographic_groups`). `office_sigma` is 0.
* `residual_covariance_mode: national_regional` uses only these configured terms. The old
  default, `residual_covariance_only`, zeroed the national error and used a covariance learned
  from the synthetic backtest panels. That mode is still available.
* **Unpolled races** draw from the fundamentals prior. Its cross-validated SD is fit on the
  synthetic panel (about 12-14 pts of share per House seat), so it is capped at
  `bayesian.fundamentals_prior.max_share_sd: 0.06` (6 pts of share).

## Sources and sanity checks
* FiveThirtyEight, "Why Early Senate And Governor Polls Have Plenty To Tell Us About November"
  (2022): Senate and governor polls in the final three weeks of the 1998-2020 campaigns had a
  weighted average absolute error of 5.4 pts of margin, about 3.4 pts of share as an SD for a
  single poll, sampling error included.
  https://fivethirtyeight.com/features/why-early-senate-and-governor-polls-have-plenty-to-tell-us-about-november/
* Cycle-level (correlated) bias in FiveThirtyEight's raw_polls.csv (CC BY 4.0), tabulated by
  politigoat.com/pollsters: 1998-2024 cycle biases have an RMS of about 2.7 pts of margin
  (1.3 pts of share); the largest were D+4.7 (2020) and R+4.8 (1998). Our 2.5-pt-share national
  term is roughly twice that. On purpose, it also covers the remaining two weeks of opinion
  movement and the uncertainty in the measured generic-ballot environment. TODO(verify):
  re-estimate it from raw_polls.csv directly.
* Race-level cap: in the MIT EDSL Senate file (doi:10.7910/DVN/PEJ5QU) and President file
  (doi:10.7910/DVN/42MVDX), 2012-2020 contested Senate races minus the same state's
  presidential two-party share, demeaned by year, give SD 6.7 pts (robust MAD-based 3.9 pts,
  n=163). That is a fundamentals-only error with no incumbency term, so 6 pts is conservative.
  TODO(verify) with real House results (the MIT House file is guestbook-gated).

## Before / after (frozen 2026-10-19 payloads, analytic backend)
| | per-race SD of D share (mean) | mean pairwise corr | common-factor SD |
|---|---|---|---|
| Senate before (p3b) | 6.60 pts | 0.107 | 2.32 pts |
| Senate after (p3c) | 4.21 pts | 0.400 | 2.67 pts |
| House before (p3b) | 12.13 pts | 0.025 | 2.00 pts |
| House after (p3c) | 5.24 pts | 0.228 | 2.52 pts |

Example leaners: NC Cooper (mean D share 54.1%) 81.8% → 88.4%; GA Ossoff (54.2%) 81.8% → 86.3%.
Test fixture: a D+10 race with 2-pt race-level SD wins 92.6% of draws, and the mean pairwise
correlation across 10 tied races is 0.53 (tests/test_error_structure.py).
