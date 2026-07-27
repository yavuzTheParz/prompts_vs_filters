# Migration from historical pilot results

Runs made before commit `29f4955` are pre-fix pilot results. A mutated child
could inherit its parent's `direct_output`, causing MR and behavioral deviation
to compare the child's filtered response with the wrong direct baseline.
Historical `asv` could also be interpreted as attack success even when it was
primarily embedding similarity.

Do not combine or directly compare those values with post-fix results.

## Required labels

- Label old tables and plots as `historical_pre_fix_pilot`.
- Do not rename old `asv` values to compliance or attack success.
- State that historical MR/BD may use inherited direct baselines.
- Use only runs at or after the full correction series for new comparisons.
- Keep `behavioral_deviation` and `semantic_recovery` modes in separate groups.
- Keep fixed-filter and coevolution conditions in separate groups.

## Re-running

Recreate important pilots with a fixed seed, explicit MR mode, prompt-specific
direct baseline, model/endpoint configuration, filter mode, commit SHA, and
dependency versions. Use the manifest and aggregate summary produced by
`run_es.py`; use the controlled T10 matrix for cross-condition analysis.
