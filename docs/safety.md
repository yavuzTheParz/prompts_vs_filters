# Safety and sanitization policy

This repository is for defensive measurement of prompt/filter coevolution. It
does not treat generated text as instructions and does not require operational
harmful content for fixtures, references, aggregate analysis, or committed
evidence.

## Data policy

- Committed attack references are broad, sanitized, non-operational category
  descriptions.
- Test fixtures use refusals, benign educational responses, or abstract
  placeholders; they must not contain actionable harmful procedures.
- T10 evidence contains aggregate metrics and plots only. It contains no raw
  prompt/response corpus.
- Raw `outputs.jsonl` and `samples.jsonl` from real experiments may contain
  sensitive model text. Review, sanitize, and store them outside version control
  unless an approved research-data process explicitly governs them.

## Secret policy

- Supply credentials through environment variables, never tracked files or CLI
  examples containing real values.
- Structured run payloads recursively redact keys containing `api_key`, `token`,
  `password`, or `secret`, plus recognized GitHub/OpenAI token patterns.
- `.env` files are ignored; `.env.example` contains placeholders only.
- Before publishing, scan staged artifacts and confirm no credentials or local
  temporary files are included.

## Interpretation boundary

Dry-run outputs validate control flow, provenance, statistics, and analysis
reproducibility. They are not measurements of real-model safety, attack success,
or filter effectiveness. Scientific claims require post-fix, model-backed runs
with documented endpoints, dependencies, datasets, sampling settings, and
separately reported fixed-filter/coevolution conditions.
