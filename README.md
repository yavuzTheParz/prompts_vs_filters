# Prompts vs Filters

This repository implements a prompt-population evolution framework for studying prompt/filter coevolution in LLM safety research. The current main runner uses a **CMA-ES-style Evolution Strategy** over prompt mutation controls rather than crossover-based genetic algorithms, because the project focuses on mutation operators over prompt style, tone, and structure.

## What is included

- `evolutionary_strategy.py`: main ES implementation.
  - `variant="cma_es"`: default CMA-ES-style adaptation over mutation style and mutation intensity.
  - `variant="one_fifth"`: global mutation intensity adapted with the 1/5 success rule.
  - `variant="self_adaptive"`: per-individual mutation intensity adapted with log-normal self-adaptation.
- `run_es.py`: command-line entrypoint for ES runs.
- `mutation_manager.py`: structural and style-aware prompt mutation operators.
- `run_llm.py`: LLM output assignment for filtered and direct prompt responses.
- `fitfunc.py`: SBERT-based ASV/MR fitness evaluation, with optional BERTScore support.
- `selection.py`: scalar and lexicographical population ordering.
- `filter_evolution.py`: optional filter-prompt update loop.
- `genetic_algorithm.py`: legacy GA-style runner kept for comparison.
- `experiments/run_ablation.py`: lightweight ablation runner for ES variants and selection modes.
- `prompts/initial_population.csv`: initial labelled prompt population.

## Installation

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Optional, but recommended for better part-of-speech-aware mutation:

```bash
python -m spacy download en_core_web_sm
```

If the SpaCy English model is not installed, the project still runs with a fallback token-selection strategy.

## Quick dry run

This runs without an LLM server and uses placeholder model outputs. It is useful for checking imports, mutation flow, and ES bookkeeping:

```bash
python run_es.py --dry-run --variant cma_es --mu 3 --lambda 10 --generations 2
```

Legacy/self-adaptive ES dry run:

```bash
python run_es.py --dry-run --variant self_adaptive --mu 3 --lambda 10 --generations 2
```

## Running with a local LLM server

The local client expects a `/generate` endpoint that accepts JSON like:

```json
{
  "prompt": "...",
  "max_new_tokens": 256,
  "temperature": 0.7,
  "top_p": 0.9,
  "do_sample": true
}
```

and returns one of the following fields:

```json
{"text": "..."}
```

or

```json
{"generated_text": "..."}
```

Run ES with explicit server URL:

```bash
python run_es.py \
  --base-url http://127.0.0.1:8000 \
  --model local-qwen \
  --variant cma_es \
  --mu 5 \
  --lambda 20 \
  --generations 10
```

Alternatively, use environment variables:

```bash
export LOCAL_LLM_BASE_URL="http://127.0.0.1:8000"
export LOCAL_LLM_API_KEY="YOUR_SECRET_KEY"  # optional
python run_es.py --variant cma_es --mu 5 --lambda 20 --generations 10
```

Use `.env.example` as the template for local secrets. Do not commit real API keys.

## Lexicographical selection

Scalar selection remains the default for backward compatibility. Lexicographical selection ranks candidates by ASV first, then minimizes MR as the tie-breaker:

```bash
python run_es.py --dry-run --selection-mode lexicographic --mr-objective minimize --variant cma_es --mu 3 --lambda 10 --generations 2
```

Use `--mr-objective maximize` for the semantic-preservation variant, where ASV is still primary but higher MR wins ties and scalar fitness rewards MR directly. This lets experiments compare both interpretations:

```bash
python run_es.py --dry-run --selection-mode lexicographic --mr-objective maximize --variant cma_es --mu 3 --lambda 10 --generations 2
```

## Optional filter coevolution

Filter updates are disabled by default. To try a candidate filter rule every generation using the current top prompts:

```bash
python run_es.py \
  --base-url http://127.0.0.1:8000 \
  --selection-mode lexicographic \
  --mr-objective minimize \
  --filter-update-every 1 \
  --top-k-filter 5 \
  --benign-csv prompts/benign.csv \
  --run-dir outputs/runs/pilot_001
```

If `--benign-csv` is omitted, a small built-in benign sanity set is used. For real experiments, provide a representative benign set so the filter cannot improve by refusing everything.

When `--run-dir` is set, the run directory stores `config.json`, `generation_summary.csv`, `filter_events.jsonl`, `filter_versions.jsonl`, `final_filter_prompt.txt`, `individuals.jsonl`, `outputs.jsonl`, and `final_population.csv`.

## Output

`run_es.py` prints the final best prompt and writes convergence/history data to:

```text
outputs/es_history.csv
```

The CSV contains:

- generation
- best_fitness
- mean_parent_fitness
- success_rate
- sigma
- cma_mean_style
- cma_mean_log_sigma
- cma_cov_00
- cma_cov_01
- cma_cov_11
- best_asv
- best_mr
- best_behavioral_deviation
- best_fluency
- best_diversity
- best_length_penalty
- best_repetition_penalty
- filter_attempted
- filter_changed
- filter_length
- filter_old_attack_refusal_rate
- filter_new_attack_refusal_rate
- filter_old_benign_refusal_rate
- filter_new_benign_refusal_rate

## Ablation runner

The lightweight ablation runner compares ES variants and selection modes across seeds:

```bash
python experiments/run_ablation.py --generations 2 --seeds 1,2,3
```

Add `--with-filter-coevolution` to include the filter-update condition in dry-run bookkeeping.

## Notes on CMA-ES and sigma

In the prompt project, the search space is discrete text. The CMA-ES variant therefore does not optimize text directly. It samples a small continuous control vector and maps it to mutation style and mutation intensity. `sigma` remains the base mutation intensity; `cma_step_size` and the learned covariance control exploration around style and intensity choices.

## Legacy GA runner

The legacy GA-style runner is still available:

```bash
python genetic_algorithm.py
```

For the proposal and main experiments, prefer `run_es.py` and `evolutionary_strategy.py`.

## Safety scope

This repository is for controlled defensive evaluation, filter robustness measurement, and benchmark construction. Do not use generated prompts or outputs to obtain or distribute harmful instructions.
