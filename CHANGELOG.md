# Changelog

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
