# Backtest & Inference Speed-Up Plan

Target: the NUTS backtest pair (Senate + House, 10 rolling-origin folds each)
currently runs ~45 minutes sequentially on an Apple M4 (10 cores, JAX CPU,
single device). Combined goal: **~4–6 minutes** with no methodology change.

## Phase 1 — Free wins (no code)

- **Run chambers concurrently.** Senate and House backtests/forecasts are
  independent processes with separate `--data-dir`/`--artifacts-dir`; launch
  them in parallel. ~2× wall clock immediately.
- Already landed on `methodology-v2`: `target_accept_prob 0.99 → 0.9` cuts
  NUTS step counts materially.

## Phase 2 — Backtest-grade sampler budget (config + ~10 lines)

Rolling-origin *scoring* does not need publication-grade posteriors.

- `configs/model.yaml`: add
  ```yaml
  bayesian:
    nuts:
      backtest_overrides:
        num_warmup: 300
        num_samples: 800
  ```
- `scoring/backtest.py::_predict_cycle`: merge the overrides into the fold's
  model-config copy before constructing `PollingModel`.
- Expected: ~2.5× per fold. Publication forecasts keep the full budget.
- Validation gate: |ΔBrier| < 0.001 and |ΔECE| < 0.005 vs the full-budget
  scorecard on one chamber before adopting.

## Phase 3 — Fold-level multiprocessing (the big one)

The 10 folds per chamber are embarrassingly parallel.

- `scoring/backtest.py::_rolling_origin_predictions`: dispatch (cycle, offset)
  folds to a `ProcessPoolExecutor` (spawn context — JAX is not fork-safe).
- Worker contract: receives context paths + scenario + fold params + a JSON
  model config; rebuilds its bundle from curated parquet (cheap) and returns
  the fold's predictions frame (Arrow IPC bytes).
- `configs/backtests.yaml`: `parallel_folds: 5` (default `min(5, cores − 2)`;
  `1` restores serial behavior for debugging).
- Determinism: per-fold seeds are already derived from fold identity; collect
  results and concatenate in sorted fold order so artifacts stay byte-stable.
- Memory: each worker holds a JAX runtime (~0.5 GB); 5 workers ≈ 2.5 GB — fine.
- Expected: ~4× on the fold loop; combined with Phase 2, backtest pair drops
  to roughly 4–6 minutes.

## Phase 4 — Chain parallelism + compilation cache (modest, cheap)

- **Parallel chains**: set `XLA_FLAGS=--xla_force_host_platform_device_count=N`
  (or `numpyro.set_host_device_count`) before JAX initializes — in the CLI
  entrypoint and the Phase 3 worker initializer — and use
  `chain_method="parallel"` when devices ≥ chains. Up to 2× on the sampling
  phase with 2 chains.
- **Persistent JAX compilation cache**: `jax.config.update(
  "jax_compilation_cache_dir", <cache dir>)` in `inference/nuts.py`. Kills the
  ~10–20 s re-JIT per fold; the on-disk cache is shared across workers and
  sessions.

## Phase 5 — GPU (deferred; hardware-gated)

- **Not on this Mac**: `jax-metal` is experimental and lacks float64, which
  `numpyro.enable_x64()` requires for HMC numerical stability on logit-scale
  posteriors. Do not trade correctness for an unproven speedup.
- **On CUDA hardware**: viable via a `performance.jax_platform` config gate,
  but today's models are small (33–870 latents); kernel-launch overhead can
  make small models slower on GPU. Revisit only if the House model moves to a
  dense joint covariance over 435 races × weekly walk states.

## Validation & rollout

1. Implement Phases 2–4 on `methodology-v2` after the current comparison suite
   completes (editing mid-run would desync the in-flight forecast stages).
2. Benchmark: record per-phase wall clock in `performance.json`; compare
   serial-vs-parallel `rolling_predictions.parquet` fingerprints for
   determinism.
3. Only then consider raising `parallel_folds` beyond 5 (M4 has 4 performance
   cores; efficiency cores give diminishing returns).
