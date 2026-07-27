# Final implementation report

Date: 2026-07-27

Branch: `emre`

Scope: Codex implementation tracker T0-T11

## Outcome

The P0/P1 correctness gates and the P2 experimental gate are implemented. The
repository now enforces prompt-specific baselines, explicit MR semantics,
stochastic-evaluation accounting, compliance-primary attack measurement,
bounded mutation, quality/diversity constraints, correct CMA-style survival,
same-filter-version comparisons, full provenance, and controlled ablations.

## Implementation commits

| Task | Commit | Outcome |
| --- | --- | --- |
| T0 | `db027a3` | Controlled baseline |
| T1 | `29f4955` | Child evaluation-state reset |
| T2 | `da66886` | Explicit MR modes |
| T3 | `66d81f0` | Stochastic evaluation |
| T4 | `2b237a2` | Defensive attack evaluator |
| T5 | `6108c96` | Mutation growth controls |
| T6 | `cda81e1` | Quality/diversity constraints |
| T7 | `e34b434` | CMA-style survival |
| T8 | `1244ded` | Filter-version re-evaluation |
| T9 | `77dc4b7` | Logging and provenance |
| T10 | `315f059` | Controlled ablations |

T11 documentation and evidence are delivered by the commit containing this
report.

## Validation evidence

- T10 smoke matrix: 7 conditions x 2 seeds x 10 generations; 14/14 complete.
- T10 full dry-run matrix: 7 conditions x 5 seeds x 10 generations; 35/35
  complete.
- Both matrices used one initial-pool hash and one documented deterministic
  model setting; no run was excluded.
- Aggregate condition statistics include 95% confidence intervals and Cohen's d.
- Six convergence plots cover fitness, attack objective, MR, diversity, sigma,
  and benign filter refusal.
- Full automated-suite and fresh-environment results are recorded in
  `IMPLEMENTATION_LOG.md`.

Evidence locations:

- `outputs/runs/t10_smoke/`
- `outputs/runs/t10_full/`
- `outputs/runs/t10_full/analysis/condition_statistics.csv`
- `outputs/runs/t10_full/analysis/analysis_manifest.json`

## Interpretation

The committed T10 experiments are deterministic dry-run validation. They show
that the matrix, provenance, exclusions, statistics, and plots are reproducible;
they do not establish real-model attack or filter effectiveness. Compliance,
auxiliary similarity, prompt growth, MR, and diversity are retained separately
so metric artifacts remain visible. Fixed-filter and coevolution results are
reported separately.

## Limitations and unresolved decisions

- A model-backed multi-seed study remains necessary for scientific claims.
- The defensive evaluator is replaceable and should be independently calibrated
  against a representative labeled dataset.
- A representative benign dataset is required for meaningful coevolution
  acceptance; the built-in set is only a sanity check.
- The CMA implementation adapts controls over discrete text operations and must
  be described as CMA-inspired, not full continuous CMA-ES over text.
- Legacy GA code remains for comparison and is not the primary validated path.

## Audit map

- Architecture: `docs/architecture.md`
- Metric formulas/directions: `docs/metrics.md`
- Safety and sanitization: `docs/safety.md`
- Historical migration: `docs/migration.md`
- Task-by-task validation: `IMPLEMENTATION_LOG.md`
- Release summary: `CHANGELOG.md`
