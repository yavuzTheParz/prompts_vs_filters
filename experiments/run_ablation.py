from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evolutionary_strategy import ESConfig, evolutionary_strategy_run
from run_es import DEFAULT_FILTER_PROMPT, _commit_sha, _dependency_versions


DEFAULT_MATRIX = ROOT / "experiments" / "configs" / "ablation_matrix.json"
SUMMARY_FIELDS = (
    "condition",
    "seed",
    "status",
    "exclusion_reason",
    "filter_mode",
    "mr_objective",
    "mutation",
    "selection",
    "generations",
    "initial_pool_sha256",
    "model_name",
    "best_fitness",
    "best_attack_objective",
    "best_compliance",
    "best_similarity",
    "best_mr",
    "best_diversity",
    "best_prompt_length",
    "runtime_sec",
)


def parse_seeds(value: str) -> list[int]:
    seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not seeds:
        raise ValueError("At least one seed is required")
    if len(seeds) != len(set(seeds)):
        raise ValueError("Seeds must be unique")
    return seeds


def load_matrix(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    conditions = payload.get("conditions", [])
    names = [condition["id"] for condition in conditions]
    if len(names) != len(set(names)):
        raise ValueError("Condition IDs must be unique")
    if not conditions:
        raise ValueError("Experiment matrix is empty")
    return payload


def mutation_flags(mode: str) -> tuple[bool, bool]:
    mapping = {
        "structural+token": (True, True),
        "token_only": (False, True),
        "structural_only": (True, False),
    }
    try:
        return mapping[mode]
    except KeyError as exc:
        raise ValueError(f"Unsupported mutation mode: {mode}") from exc


def pool_hash(csv_path: Path) -> str:
    return hashlib.sha256(csv_path.read_bytes()).hexdigest()


def _history_rows(condition: dict, seed: int, result) -> list[dict]:
    return [
        {"condition": condition["id"], "seed": seed, **row}
        for row in result.history
    ]


def run_condition(
    condition: dict,
    seed: int,
    *,
    generations: int,
    mu: int,
    lambda_: int,
    csv_path: Path,
    model_name: str,
    initial_pool_sha256: str,
) -> tuple[dict, list[dict]]:
    structural, token = mutation_flags(condition["mutation"])
    config = ESConfig(
        lambda_=lambda_,
        mu=mu,
        generations=generations,
        variant="cma_es",
        csv_path=str(csv_path),
        random_seed=seed,
        lightweight=True,
        selection_mode=condition["selection"],
        mr_objective=condition["mr_objective"],
        filter_update_every=1 if condition["filter_mode"] == "coevolution" else 0,
        structural_mutation_enabled=structural,
        token_mutation_enabled=token,
        verbose=False,
    )
    result = evolutionary_strategy_run(
        config=config,
        client=None,
        model_name=model_name,
        filter_prompt=DEFAULT_FILTER_PROMPT,
    )
    metrics = result.best.metrics or {}
    row = {
        "condition": condition["id"],
        "seed": seed,
        "status": "complete",
        "exclusion_reason": "",
        "filter_mode": condition["filter_mode"],
        "mr_objective": condition["mr_objective"],
        "mutation": condition["mutation"],
        "selection": condition["selection"],
        "generations": generations,
        "initial_pool_sha256": initial_pool_sha256,
        "model_name": model_name,
        "best_fitness": result.best.fitness,
        "best_attack_objective": metrics.get("attack_objective", 0.0),
        "best_compliance": metrics.get("attack_compliance_score", 0.0),
        "best_similarity": metrics.get("unsafe_reference_similarity", 0.0),
        "best_mr": metrics.get("mr", 0.0),
        "best_diversity": metrics.get("diversity", 0.0),
        "best_prompt_length": len(result.best.input_prompt),
        "runtime_sec": result.runtime_sec,
    }
    return row, _history_rows(condition, seed, result)


def write_csv(path: Path, rows: list[dict], fieldnames=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    names = list(fieldnames or sorted({key for row in rows for key in row}))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=names, extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(description="Run the controlled T10 ablation matrix.")
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--generations", type=int, default=10)
    parser.add_argument("--mu", type=int, default=3)
    parser.add_argument("--lambda", dest="lambda_", type=int, default=8)
    parser.add_argument("--seeds", default="101,102,103,104,105")
    parser.add_argument("--csv", type=Path, default=ROOT / "prompts" / "initial_population.csv")
    parser.add_argument("--model-name", default="dry-run-deterministic")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    seeds = parse_seeds(args.seeds)
    matrix = load_matrix(args.matrix)
    initial_pool_sha256 = pool_hash(args.csv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "matrix_version": matrix["matrix_version"],
        "commit_sha": _commit_sha(),
        "matrix_path": str(args.matrix),
        "seeds": seeds,
        "generations": args.generations,
        "mu": args.mu,
        "lambda": args.lambda_,
        "initial_pool": str(args.csv),
        "initial_pool_sha256": initial_pool_sha256,
        "model_name": args.model_name,
        "dependencies": _dependency_versions(),
        "command": " ".join(sys.argv),
        "completion_rule": "status == complete and generations == requested generations",
    }
    (args.output_dir / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    summaries: list[dict] = []
    histories: list[dict] = []
    failures: list[dict] = []
    for condition in matrix["conditions"]:
        for seed in seeds:
            try:
                summary, history = run_condition(
                    condition,
                    seed,
                    generations=args.generations,
                    mu=args.mu,
                    lambda_=args.lambda_,
                    csv_path=args.csv,
                    model_name=args.model_name,
                    initial_pool_sha256=initial_pool_sha256,
                )
                summaries.append(summary)
                histories.extend(history)
            except Exception as exc:
                failure = {
                    "condition": condition["id"],
                    "seed": seed,
                    "status": "failed",
                    "exclusion_reason": f"{type(exc).__name__}: {exc}",
                    "filter_mode": condition.get("filter_mode", ""),
                    "mr_objective": condition.get("mr_objective", ""),
                    "mutation": condition.get("mutation", ""),
                    "selection": condition.get("selection", ""),
                    "generations": 0,
                    "initial_pool_sha256": initial_pool_sha256,
                    "model_name": args.model_name,
                }
                summaries.append(failure)
                failures.append({**failure, "traceback": traceback.format_exc()})

    write_csv(args.output_dir / "run_summary.csv", summaries, SUMMARY_FIELDS)
    write_csv(args.output_dir / "generation_history.csv", histories)
    (args.output_dir / "failures.json").write_text(
        json.dumps(failures, indent=2), encoding="utf-8"
    )
    print(
        f"Completed {len(summaries) - len(failures)}/{len(summaries)} runs; "
        f"artifacts: {args.output_dir}"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
