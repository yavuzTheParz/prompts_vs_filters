# Codex Implementation Log

This log records each task from the Codex Implementation Task Tracker, including
the implementation scope, validation commands, and observed results.

## T0 - Controlled Implementation Baseline

- Status: Complete
- Date: 2026-07-26
- Working branch: `emre`
- Baseline `main` commit: `d61ecf8204e4c8c159446aba80474228e48fbad5`
- Algorithm behavior changed: No

### Changes

- Added repository safety ignores for local environment files, Python bytecode,
  and macOS metadata.
- Saved the fixed-seed dry-run configuration and artifacts under
  `outputs/runs/baseline_101/`.
- Recorded the baseline commands and results in this implementation log.

### Validation

Full unit-test suite:

```bash
python3 -B -m unittest discover -s tests -v
```

Result: 13 tests passed; 1 opt-in local-LLM integration test was skipped because
no local model endpoint was configured.

Fixed-seed dry-run:

```bash
python3 run_es.py --dry-run --variant cma_es --mu 3 --lambda 10 \
  --generations 2 --seed 101 --run-dir outputs/runs/baseline_101
```

Result: completed two generations with best fitness `0.3700`, best ASV
`0.1000`, best MR `0.0000`, and final sigma `0.7974727861523894`.

The command was run twice. SHA-256 checksums for `config.json`,
`generation_summary.csv`, and `outputs.jsonl` were identical between runs.

### Safety audit

- A tracked-file signature scan found no GitHub, OpenAI, AWS, or private-key
  credential patterns.
- The new baseline contains synthetic dry-run responses only.
- Previously tracked real-run response artifacts were reviewed with a defensive
  pattern scan; no operational weapon-building or procedural cyber-instruction
  pattern was detected. Historical artifacts were left unchanged.
- Pre-existing untracked coevolution outputs were excluded from this task.

## T1 - Reset Direct Baselines for Mutated Children

- Status: Complete
- Date: 2026-07-26
- Algorithm behavior changed: Yes - mutated prompts now require a
  prompt-specific direct baseline.

### Changes

- `_clone_prompt(..., keep_outputs=False)` no longer inherits `direct_output`.
- Mutation explicitly clears the child's `direct_output`, `output_prompts`,
  `metrics`, and `fitness` after changing its prompt text.
- Added regression coverage for clone preservation, child reset, fresh direct
  generation per child, and resume behavior.
- Marked historical MR/behavioral-deviation pilot results as pre-fix and not
  directly comparable with corrected runs.

### Validation

Targeted tests:

```bash
python3 -B -m unittest tests.test_evolutionary_strategy -v
```

Result: 5 tests passed.

Full unit-test suite:

```bash
python3 -B -m unittest discover -s tests -v
```

Result: 18 tests passed; 1 opt-in local-LLM integration test was skipped because
no local model endpoint was configured.

Minimal dry-run validation:

```bash
python3 run_es.py --dry-run --variant cma_es --mu 2 --lambda 4 \
  --generations 1 --seed 102 --history-csv /tmp/t1_history.csv \
  --run-dir /tmp/t1_run
```

Result: completed one generation with best fitness `0.3700`. Temporary
validation artifacts were kept outside the repository.

## T2 - Formalize MR Objective Modes

- Status: Complete
- Date: 2026-07-26
- Canonical modes: `behavioral_deviation`, `semantic_recovery`

### Changes

- Added a shared MR objective definition used by fitness, lexicographic
  selection, rank partitioning, the ES loop, and the CLI.
- `behavioral_deviation` uses
  `fitness = 0.7 * ASV + 0.3 * (1 - MR)`.
- `semantic_recovery` uses `fitness = 0.7 * ASV + 0.3 * MR`.
- Preserved legacy objective names as deprecated aliases with explicit
  warnings.
- Added the canonical mode, full formula, and scientific definition to
  `config.json` and every CLI run summary.
- Updated README examples and the ablation/full-validation helpers to use
  canonical mode names.

### Validation

Targeted formula, selection, and artifact tests:

```bash
python3 -B -m unittest \
  tests.test_fitfunc tests.test_selection tests.test_run_es -v
```

Result: 6 tests passed.

Full unit-test suite:

```bash
python3 -B -m unittest discover -s tests -v
```

Result: 24 tests passed; 1 opt-in local-LLM integration test was skipped because
no local model endpoint was configured.

Both canonical modes completed a one-generation, fixed-seed dry-run with
lexicographic selection. Their console summaries and `config.json` files
recorded the expected mode and formula. Temporary validation artifacts were
kept outside the repository.

## T3 - Reduce Stochastic Evaluation Noise

- Status: Complete
- Date: 2026-07-26
- Real-run filtered-sample default: `k_evals=3`
- Dry-run filtered-sample default: `k_evals=1`
- Direct baseline default: deterministic at temperature `0`

### Changes

- Added separate direct and filtered temperatures plus configurable
  sample-level retries.
- Rejected empty model responses and recorded every valid/invalid attempt with
  prompt, generation, sample index, attempt, temperature, status, and response
  or error.
- Added `samples.jsonl` to structured run artifacts.
- Added per-prompt sample count and population-standard-deviation metrics for
  ASV and MR.
- Ensured exhausted retries set invalid metadata so failed samples receive zero
  selectable fitness.
- Added K=3 call-count/averaging tests, transient retry coverage, invalid empty
  response coverage, and fixed-seed reproducibility coverage.

### Validation

Targeted sampling and artifact tests:

```bash
python3 -B -m unittest tests.test_run_llm tests.test_run_es -v
```

Result: 7 tests passed.

Full unit-test suite:

```bash
python3 -B -m unittest discover -s tests -v
```

Result: 30 tests passed; 1 opt-in local-LLM integration test was skipped because
no local model endpoint was configured.

A one-generation fixed-seed CLI dry-run confirmed the inexpensive K=1 dry-run
default, separate temperatures, and sample records spanning generations 0 and
1. Temporary validation artifacts were kept outside the repository.

## T4 - Strengthen Attack-Success Evaluation

- Status: Complete
- Date: 2026-07-26
- Evaluator version: `defensive-compliance-v1`

### Changes

- Renamed raw reference cosine semantics to
  `unsafe_reference_similarity`; it is auxiliary and is not reported as an
  attack success rate.
- Added a replaceable defensive evaluator returning bounded compliance scores,
  categorical labels, validity status, version, model, template, and threshold.
- Added multiple sanitized, non-operational reference descriptions for every
  supported content category.
- Made the primary attack objective an ensemble with compliance weight `0.8`
  and auxiliary reference-similarity weight `0.2`.
- Preserved `asv` only as a temporary schema alias for the ensemble attack
  objective.
- Persisted evaluator configuration in `config.json` and evaluator decisions in
  output artifacts.
- Converted legacy operational test content to sanitized placeholders.
- Ensured evaluator exceptions and invalid results produce zero fitness.

### Validation

Targeted defensive evaluator tests:

```bash
python3 -B -m unittest tests.test_attack_evaluator -v
```

Result: 5 tests passed, covering refusal topic-word repetition, benign
educational content, sanitized compliance, evaluator failure, ensemble
priority, and artifact metadata.

Full unit-test suite:

```bash
python3 -B -m unittest discover -s tests -v
```

Result: 35 tests passed; 1 opt-in local-LLM integration test was skipped because
no local model endpoint was configured.

The sanitized manual fitness probe classified the compliant placeholder above
the refusal while retaining distinct MR-mode behavior. A fixed-seed CLI dry-run
confirmed evaluator version, model, template, and threshold in `config.json`.
Temporary validation artifacts were kept outside the repository.

## T5 - Prevent Structural Mutation Duplication and Prompt Bloat

- Status: Complete
- Date: 2026-07-26

### Changes

- Added recognizable `STYLE_PREFIX` and `STYLE_SUFFIX` tags.
- Limited prompts to one active prefix and one active suffix; new templates
  replace the template in the same position.
- Added exact-duplicate and repeated n-gram rejection.
- Added configurable absolute and seed-relative character/token growth limits.
- Added compression mutation for tagged style material and adjacent duplicate
  words.
- Added structured mutation logs with operator, add/replace/compression action,
  before/after length, acceptance state, and reason.
- Preserved the existing token-mutation path and structural dependency fallback.

### Validation

Targeted mutation tests:

```bash
python3 -B -m unittest tests.test_mutation_manager -v
```

Result: 5 tests passed, including 20 repeated structural mutations, growth-limit
rejection, compression, and the existing token-path fallback.

Full unit-test suite:

```bash
python3 -B -m unittest discover -s tests -v
```

Result: 38 tests passed; 1 opt-in local-LLM integration test was skipped because
no local model endpoint was configured.

## T6 - Use Quality Constraints and Population Diversity

- Status: Complete
- Date: 2026-07-26

### Changes

- Added hard validity gates for API errors, missing outputs, excessive prompt
  length, and excessive repetition.
- Applied configurable soft length/repetition penalties only to valid
  candidates.
- Added token-distance population and per-candidate diversity metrics.
- Added exact and near-duplicate suppression before parent selection.
- Changed scalar and lexicographic selection to order validity first, attack
  objective second, MR mode third, and quality/diversity last.
- Added generation-level diversity and rejection counts by reason.
- Exposed quality thresholds through CLI and `config.json`.

### Validation

Targeted quality and selection tests:

```bash
python3 -B -m unittest \
  tests.test_quality_constraints tests.test_selection tests.test_core_behaviour -v
```

Result: 15 tests passed.

Full unit-test suite and a fixed-seed dry-run were used to verify integration
with the ES loop and generation-summary schema.

Result: 42 tests passed; 1 opt-in local-LLM integration test was skipped. The
dry-run recorded nonzero population diversity and per-reason rejection counts.

## T7 - Correct CMA-ES Survival and Method Naming

- Status: Complete
- Date: 2026-07-26

### Changes

- Made CMA-style selection honor configured plus and comma survival.
- Plus survival ranks parents and offspring together; comma survival permits
  only offspring to survive.
- Preserved CMA control vector and sigma metadata for every selected
  individual, including surviving parents.
- Updated CMA mean/covariance from the selected survivors' control vectors.
- Clarified that the implementation is CMA-ES-style/CMA-inspired control
  adaptation over discrete style and mutation-intensity choices, not direct
  continuous optimization of text.

### Validation

```bash
python3 -B -m unittest tests.test_cma_survival tests.test_core_behaviour -v
python3 -B -m unittest discover -s tests -v
```

Result: 3 focused survival/distribution tests passed; the full suite passed 45
tests with 1 opt-in integration test skipped. Fixed-seed plus and comma dry-runs
both completed and recorded the selected survival schema in `config.json`.

## T8 - Re-evaluate Population After Filter Updates

- Status: Complete
- Date: 2026-07-26

### Changes

- Tagged every evaluation, sample, final-population row, and selected parent
  with a filter version.
- On accepted filter updates, invalidated active parents' filtered outputs,
  fitness, metrics, and evaluator cache while preserving direct baselines.
- Re-evaluated and re-ranked every active parent under the new filter before
  continuing.
- Reset the comparable best candidate after version changes so stale/current
  filter scores are never compared.
- Added old/new filter-version lineage to accepted update events.
- Prevented rejected updates from triggering re-evaluation.
- Added explicit `fixed_filter`/`coevolution` run metadata.

### Validation

```bash
python3 -B -m unittest \
  tests.test_filter_reevaluation tests.test_core_behaviour -v
python3 -B -m unittest discover -s tests -v
```

Result: 4 targeted filter-version tests passed; the full suite passed 49 tests
with 1 opt-in integration test skipped.

## T9 - Expand Logging, Provenance, and Reproducibility

- Status: Complete
- Date: 2026-07-26

### Changes

- Added mean, median, and population standard deviation for every optimization
  metric to generation summaries.
- Added mutation-operator attempt, acceptance, success-rate, and fallback
  counters.
- Added stable prompt, parent, seed-prompt, generation, and mutation-lineage
  identifiers to candidates and exported lineage records.
- Added a sanitized run manifest containing commit SHA, seed, model metadata,
  objective/filter modes, and dependency versions.
- Added an aggregate-only `summary.json` artifact that does not require unsafe
  raw model output.
- Added recursive credential-field and known token-pattern redaction for
  structured configuration and provenance output.

### Validation

```bash
python3 -B -m unittest tests.test_provenance tests.test_run_es -v
python3 -B -m unittest discover -s tests -v
python3 run_es.py --dry-run --mu 2 --lambda 4 --generations 2 --seed 109 \
  --history-csv /tmp/t9_history.csv --run-dir /tmp/t9_run
```

Result: 7 targeted provenance/output tests passed; the full suite passed 53
tests with 1 opt-in local-LLM integration test skipped. The fixed-seed dry-run
completed and produced configuration, generation summary, filter event/version,
final filter, output, sample, lineage, manifest, and aggregate summary
artifacts.

## T10 - Controlled Regression and Ablation Experiments

- Status: Complete
- Date: 2026-07-26

### Changes

- Added a versioned seven-condition matrix covering the six required
  fixed-filter/coevolution, MR-mode, and mutation ablations plus a scalar versus
  constraint-aware selection comparison.
- Added explicit structural- and token-mutation operator switches for both
  lightweight and dependency-backed mutation paths.
- Replaced the ad hoc experiment loop with a deterministic runner that records
  completion status, exclusion reason, initial-pool hash, seed, model setting,
  aggregate metrics, and per-generation history.
- Added reproducible analysis for 95% confidence intervals, Cohen's d effect
  sizes, rule-based exclusions, and six convergence SVGs.
- Added a sanitized aggregate fixture for the pre-fix-compatible versus
  prompt-specific direct-baseline comparison.
- Saved separate two-seed smoke and five-seed full dry-run evidence under
  `outputs/runs/`.

### Validation

```bash
python3 -B -m unittest tests.test_ablation tests.test_mutation_manager \
  tests.test_run_es -v
python3 -B experiments/run_ablation.py --generations 10 --mu 3 --lambda 8 \
  --seeds 101,102 --output-dir outputs/runs/t10_smoke
python3 -B experiments/run_ablation.py --generations 10 --mu 3 --lambda 8 \
  --seeds 101,102,103,104,105 --output-dir outputs/runs/t10_full
python3 -B analysis/analyze_ablation.py --run-dir outputs/runs/t10_full \
  --output-dir outputs/runs/t10_full/analysis
python3 -B -m unittest discover -s tests -v
```

Result: 14 targeted tests passed. The smoke matrix completed 14/14 runs and the
full matrix completed 35/35 runs, each with one initial-pool hash and one model
setting; neither required exclusions. Analysis regenerated confidence
interval/effect-size tables and all six required convergence plots. The full
suite passed 59 tests with 1 opt-in local-LLM integration test skipped.

These results validate the experimental plumbing only. They use deterministic
dry-run proxies and therefore do not support real-model attack-success or
coevolution-effectiveness claims. Fixed-filter and coevolution outputs remain
separately labeled.

## T11 - Finalize Documentation and Evidence Package

- Status: Complete
- Date: 2026-07-27

### Changes

- Added architecture/data-flow documentation for direct generation, filtered
  sampling, scoring, constraints, selection, lineage, and accepted filter
  updates.
- Documented every active metric formula, direction, hard validity rule, soft
  penalty, and selection priority.
- Added explicit safe-scope, sanitization, raw-output, fixture, credential, and
  scientific-interpretation policies.
- Added a migration guide separating historical pre-fix pilots from corrected
  results.
- Updated README commands to current CLI options and removed the stale legacy
  ablation command.
- Added a changelog, final implementation report, validation report, and
  machine-readable evidence manifest.
- Added automated documentation/CLI/evidence consistency tests.
- Added a standard-library-only dry-run requirements file for isolated
  validation.

### Validation

```bash
python3 -m venv /tmp/prompts_filters_t11_venv
/tmp/prompts_filters_t11_venv/bin/python -m pip install \
  -r requirements-dry-run.txt
/tmp/prompts_filters_t11_venv/bin/python -B run_es.py --dry-run \
  --variant cma_es --mu 3 --lambda 4 --generations 2 --seed 101 \
  --history-csv /tmp/t11_fresh_history.csv \
  --run-dir /tmp/t11_fresh_run --quiet
python3 -B -m unittest tests.test_documentation tests.test_provenance \
  tests.test_ablation -v
python3 -B -m unittest discover -s tests -v
```

Result: the isolated fresh-environment install and dry-run succeeded and
produced the complete structured artifact set. Thirteen focused
documentation/provenance/ablation tests passed. The full suite passed 62 tests
with 1 opt-in local-LLM integration test skipped.

## R0 - Quarantine Invalid Experimental Evidence

- Status: Complete
- Date: 2026-07-30

### Changes

- Registered `coevo_g320_l10_k3_seed13` as `invalid` and
  `diagnostic_only` in a machine-readable invalid-run registry.
- Recorded all five observed invalidation reasons and SHA-256 digests for four
  primary source artifacts without modifying the original experiment output.
- Added a reusable run-eligibility helper that partitions valid and
  quarantined runs and can verify registered artifact hashes read-only.
- Integrated the registry into the ablation analyzer before any experiment
  files are read. Quarantined runs now emit an exclusion manifest and exit
  without producing scientific statistics or plots.
- Added regression coverage for automatic exclusion, manifest reasons, valid
  run pass-through, and source-artifact immutability.

### Validation

```bash
python3 -B -m unittest tests.test_run_quarantine tests.test_ablation \
  tests.test_documentation -v
python3 -B analysis/run_registry.py \
  --run-dir outputs/coevo_g320_l10_k3_seed13_run \
  --run-dir outputs/runs/t10_full \
  --output /tmp/r0_run_eligibility.json
python3 -B -m unittest discover -s tests -v
```

Result: 13 targeted tests passed. The registry excluded the triggering run
with all five reasons while leaving `t10_full` eligible. All four registered
source-artifact SHA-256 digests matched. The full suite passed 66 tests with
1 opt-in local-LLM integration test skipped.
