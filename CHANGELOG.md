# Changelog

## 2026-09-24 - Search-signal fixes and full-scale readiness

- Replaced the garbled-token heuristic, which rejected 840 of 869 seed prompts,
  with character-shape rules. Deliberate letter-transposition seeds are now valid.
- Near-duplicate marks are recomputed on every ranking instead of permanently
  zeroing parents (55–100% of parent slots were affected across the 19 real-model historical runs).
- Added the search-only `attack_progress` tie-breaker below the true objective,
  bounded retries for no-op mutations, model-free skipping of exact-duplicate
  offspring, a per-seed survivor cap, parent resampling, categorical tone
  adaptation for any number of tones, filter warmup, and a minimum number of
  valid positive candidates before a filter update.
- Added concurrent model evaluation, streamed run artifacts, atomic checkpoints,
  `--resume`, `--preset full|smoke`, `scripts/estimate_budget.py`,
  `analysis/convergence.py`, and a simulated target for A/B validation.
- Each finished run now writes `fitness.png` and `dashboard.png` to
  `<run-dir>/plots/` (disable with `--no-plots`); `analysis/plot_run.py` plots
  existing or in-progress runs. Added `matplotlib` to `requirements.txt`.

- Runs can stop on a fixed budget of attacker model calls (`--max-evaluations`)
  with the filter schedule on the same clock, `--population small|medium|large`,
  nested initial populations (`--initial-population-nest`), a campaign launcher
  (`experiments/run_campaign.py`), and a server benchmark
  (`scripts/benchmark_server.py`).

Several new controls are on by default (`progress_tiebreak`, `mutation_retry_limit=3`,
`skip_duplicate_offspring`, `tone_adaptation="categorical"`), so the same seed
no longer reproduces pre-change trajectories. Use `--no-progress-tiebreak
--mutation-retry-limit 0 --evaluate-duplicate-offspring --tone-adaptation uniform`
to approximate the previous search controls. The quality-gate fixes cannot be
switched off. See [`docs/scaling.md`](docs/scaling.md).

## 2026-07-27 - Measurement and coevolution correction series

- Reset evaluation state after every mutation and require a prompt-specific
  direct baseline.
- Made `behavioral_deviation` and `semantic_recovery` explicit MR modes.
- Added multi-sample evaluation, retries, variance reporting, and invalid-state
  handling.
- Replaced similarity-only attack scoring with a compliance-primary defensive
  evaluator and sanitized auxiliary references.
- Bounded structural mutation growth and activated validity, quality, duplicate,
  and diversity constraints.
- Corrected CMA-style plus/comma survival and preserved control-vector state.
- Re-evaluated the active population after accepted filter changes.
- Added complete lineage, provenance, sanitized manifests, and aggregate
  summaries.
- Added controlled smoke/full dry-run ablations with confidence intervals,
  effect sizes, explicit exclusions, and convergence plots.

Historical pilot results produced before this series are not directly comparable
to corrected MR or behavioral-deviation results. See
[`docs/migration.md`](docs/migration.md).
