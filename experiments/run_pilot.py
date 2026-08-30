from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BENIGN_CSV = ROOT / "prompts" / "benign_pilot.csv"
DEFAULT_BENIGN_HOLDOUT_CSV = ROOT / "prompts" / "benign_pilot_holdout.csv"
SUMMARY_FIELDS = (
    "mode",
    "seed",
    "repeat",
    "status",
    "run_dir",
    "runtime_sec",
    "best_fitness",
    "best_attack_objective",
    "best_valid",
    "best_validity_reason",
    "confirmation_k",
    "confirmation_fitness",
    "confirmation_attack_objective",
    "confirmation_valid",
    "confirmation_validity_reason",
    "filter_versions",
    "sample_attempts",
    "initial_benign_refusal_rate",
    "final_benign_refusal_rate",
    "benign_refusal_rate_delta",
)
PAIRED_METRICS = (
    "best_fitness",
    "best_attack_objective",
    "confirmation_fitness",
    "confirmation_attack_objective",
    "final_benign_refusal_rate",
)


def parse_seeds(value: str) -> list[int]:
    seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not seeds:
        raise ValueError("At least one seed is required")
    if len(seeds) != len(set(seeds)):
        raise ValueError("Seeds must be unique; use --repeats for same-seed replication")
    return seeds


def parse_modes(value: str) -> list[str]:
    modes = [item.strip() for item in value.split(",") if item.strip()]
    allowed = {"fixed_filter", "coevolution"}
    invalid = set(modes) - allowed
    if not modes or invalid:
        raise ValueError(f"Modes must be drawn from {sorted(allowed)}; got {sorted(invalid)}")
    if len(modes) != len(set(modes)):
        raise ValueError("Modes must be unique")
    return modes


def run_signature(args, mode: str, seed: int, repeat: int) -> dict:
    return {
        "mode": mode,
        "seed": seed,
        "repeat": repeat,
        "generations": args.generations,
        "mu": args.mu,
        "lambda": args.lambda_,
        "k_evals": args.k_evals,
        "final_k_evals": args.final_k_evals,
        "filtered_temperature": args.filtered_temperature,
        "filter_update_every": (
            args.filter_update_every if mode == "coevolution" else 0
        ),
        "top_k_filter": args.top_k_filter,
        "stagnation_generations": args.stagnation_generations,
        "max_garbled_token_ratio": args.max_garbled_token_ratio,
        "model": args.model,
        "base_url": args.base_url,
        "benign_csv": str(args.benign_csv.resolve()),
        "benign_holdout_csv": str(args.benign_holdout_csv.resolve()),
        "benign_holdout_repeats": args.benign_holdout_repeats,
    }


def completed_summary(
    run_dir: Path, generations: int, expected_signature: dict | None = None
) -> dict | None:
    path = run_dir / "summary.json"
    if not path.is_file():
        return None
    if expected_signature is not None:
        signature_path = run_dir / "pilot_run_config.json"
        if not signature_path.is_file():
            return None
        actual_signature = json.loads(signature_path.read_text(encoding="utf-8"))
        if actual_signature != expected_signature:
            return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload.get("generations_completed", 0)) != generations:
        return None
    return payload


def is_rerunnable_placeholder(run_dir: Path, expected_signature: dict) -> bool:
    if not run_dir.is_dir():
        return False
    entries = list(run_dir.iterdir())
    signature_path = run_dir / "pilot_run_config.json"
    if entries != [signature_path] or not signature_path.is_file():
        return False
    try:
        actual = json.loads(signature_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return actual == expected_signature


def summary_row(mode: str, seed: int, repeat: int, run_dir: Path, payload: dict) -> dict:
    best = payload.get("best_metrics", {}) or {}
    confirmation = payload.get("final_reevaluation", {}) or {}
    confirmed = confirmation.get("confirmation_metrics", {}) or {}
    benign_holdout = payload.get("benign_holdout", {}) or {}
    return {
        "mode": mode,
        "seed": seed,
        "repeat": repeat,
        "status": "complete",
        "run_dir": str(run_dir),
        "runtime_sec": payload.get("runtime_sec", ""),
        "best_fitness": payload.get("best_fitness", ""),
        "best_attack_objective": best.get("attack_objective", best.get("asv", "")),
        "best_valid": best.get("valid", ""),
        "best_validity_reason": best.get("validity_reason", ""),
        "confirmation_k": confirmation.get("k_evals", ""),
        "confirmation_fitness": confirmation.get("confirmation_fitness", ""),
        "confirmation_attack_objective": confirmed.get(
            "attack_objective", confirmed.get("asv", "")
        ),
        "confirmation_valid": confirmed.get("valid", ""),
        "confirmation_validity_reason": confirmed.get("validity_reason", ""),
        "filter_versions": payload.get("filter_versions", ""),
        "sample_attempts": payload.get("sample_attempts", ""),
        "initial_benign_refusal_rate": benign_holdout.get(
            "initial_benign_refusal_rate", ""
        ),
        "final_benign_refusal_rate": benign_holdout.get(
            "final_benign_refusal_rate", ""
        ),
        "benign_refusal_rate_delta": benign_holdout.get(
            "benign_refusal_rate_delta", ""
        ),
    }


def build_command(args, mode: str, seed: int, run_dir: Path) -> list[str]:
    update_every = args.filter_update_every if mode == "coevolution" else 0
    return [
        sys.executable,
        "-B",
        str(ROOT / "run_es.py"),
        "--base-url",
        args.base_url,
        "--model",
        args.model,
        "--variant",
        "cma_es",
        "--selection-mode",
        "lexicographic",
        "--mr-objective",
        "behavioral_deviation",
        "--generations",
        str(args.generations),
        "--mu",
        str(args.mu),
        "--lambda",
        str(args.lambda_),
        "--seed",
        str(seed),
        "--k-evals",
        str(args.k_evals),
        "--final-k-evals",
        str(args.final_k_evals),
        "--filtered-temperature",
        str(args.filtered_temperature),
        "--filter-update-every",
        str(update_every),
        "--top-k-filter",
        str(args.top_k_filter),
        "--benign-csv",
        str(args.benign_csv.resolve()),
        "--benign-holdout-csv",
        str(args.benign_holdout_csv.resolve()),
        "--benign-holdout-repeats",
        str(args.benign_holdout_repeats),
        "--stagnation-generations",
        str(args.stagnation_generations),
        "--restart-on-stagnation",
        "--max-garbled-token-ratio",
        str(args.max_garbled_token_ratio),
        "--history-csv",
        str((run_dir / "history.csv").resolve()),
        "--run-dir",
        str(run_dir.resolve()),
    ]


def write_summary(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def paired_effects(rows: list[dict]) -> list[dict]:
    complete = [row for row in rows if row.get("status") == "complete"]
    by_pair = {
        (int(row["seed"]), int(row["repeat"]), row["mode"]): row
        for row in complete
    }
    pair_ids = sorted(
        {
            (seed, repeat)
            for seed, repeat, mode in by_pair
            if mode == "fixed_filter"
            and (seed, repeat, "coevolution") in by_pair
        }
    )
    output = []
    for metric in PAIRED_METRICS:
        differences = []
        for seed, repeat in pair_ids:
            fixed = by_pair[(seed, repeat, "fixed_filter")].get(metric, "")
            coevo = by_pair[(seed, repeat, "coevolution")].get(metric, "")
            if fixed in ("", None) or coevo in ("", None):
                continue
            differences.append(float(coevo) - float(fixed))
        if not differences:
            continue
        mean = statistics.fmean(differences)
        if len(differences) > 1:
            standard_error = statistics.stdev(differences) / math.sqrt(len(differences))
            # Conservative small-sample critical values; normal approximation after n=30.
            critical = {
                1: 12.706,
                2: 4.303,
                3: 3.182,
                4: 2.776,
                5: 2.571,
                6: 2.447,
                7: 2.365,
                8: 2.306,
                9: 2.262,
                10: 2.228,
            }.get(len(differences) - 1, 1.96)
            half_width = critical * standard_error
            sd = statistics.stdev(differences)
            effect_dz = mean / sd if sd > 0 else 0.0
        else:
            half_width = 0.0
            effect_dz = 0.0
        output.append(
            {
                "metric": metric,
                "difference": "coevolution_minus_fixed_filter",
                "n_pairs": len(differences),
                "mean_difference": mean,
                "ci95_low": mean - half_width,
                "ci95_high": mean + half_width,
                "cohens_dz": effect_dz,
                "lower_is_better_for_filter": True,
            }
        )
    return output


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run paired fixed-filter and coevolution pilot replications."
    )
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", default="dolphin")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--modes", default="fixed_filter,coevolution")
    parser.add_argument("--seeds", default="101,202,303")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--generations", type=int, default=120)
    parser.add_argument("--mu", type=int, default=8)
    parser.add_argument("--lambda", dest="lambda_", type=int, default=16)
    parser.add_argument("--k-evals", type=int, default=2)
    parser.add_argument("--final-k-evals", type=int, default=8)
    parser.add_argument("--filtered-temperature", type=float, default=0.7)
    parser.add_argument("--filter-update-every", type=int, default=5)
    parser.add_argument("--top-k-filter", type=int, default=5)
    parser.add_argument("--benign-csv", type=Path, default=DEFAULT_BENIGN_CSV)
    parser.add_argument(
        "--benign-holdout-csv", type=Path, default=DEFAULT_BENIGN_HOLDOUT_CSV
    )
    parser.add_argument("--benign-holdout-repeats", type=int, default=3)
    parser.add_argument("--stagnation-generations", type=int, default=20)
    parser.add_argument("--max-garbled-token-ratio", type=float, default=0.40)
    return parser.parse_args()


def validate(args, seeds: list[int], modes: list[str]) -> None:
    if args.repeats < 1:
        raise ValueError("--repeats must be at least 1")
    if args.generations < 1 or args.mu < 2 or args.lambda_ < args.mu:
        raise ValueError("Require generations >= 1 and lambda >= mu >= 2")
    if args.k_evals < 1 or args.final_k_evals < args.k_evals:
        raise ValueError("Require final-k-evals >= k-evals >= 1")
    if args.filter_update_every < 1 and "coevolution" in modes:
        raise ValueError("Coevolution requires --filter-update-every >= 1")
    if not args.benign_csv.is_file():
        raise FileNotFoundError(f"Benign calibration CSV not found: {args.benign_csv}")
    if not args.benign_holdout_csv.is_file():
        raise FileNotFoundError(f"Benign holdout CSV not found: {args.benign_holdout_csv}")
    if args.benign_holdout_repeats < 1:
        raise ValueError("--benign-holdout-repeats must be at least 1")


def main() -> int:
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    args.benign_csv = args.benign_csv.resolve()
    args.benign_holdout_csv = args.benign_holdout_csv.resolve()
    seeds = parse_seeds(args.seeds)
    modes = parse_modes(args.modes)
    validate(args, seeds, modes)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "schema_version": 1,
        "modes": modes,
        "seeds": seeds,
        "repeats": args.repeats,
        "generations": args.generations,
        "mu": args.mu,
        "lambda": args.lambda_,
        "k_evals": args.k_evals,
        "final_k_evals": args.final_k_evals,
        "model": args.model,
        "base_url": args.base_url,
        "benign_csv": str(args.benign_csv),
        "benign_holdout_csv": str(args.benign_holdout_csv),
        "benign_holdout_repeats": args.benign_holdout_repeats,
        "planned_runs": len(modes) * len(seeds) * args.repeats,
        "command": " ".join(sys.argv),
    }
    (args.output_dir / "pilot_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    rows = []
    failures = 0
    for mode in modes:
        for seed in seeds:
            for repeat in range(1, args.repeats + 1):
                run_dir = args.output_dir / f"{mode}_seed{seed}_r{repeat}"
                signature = run_signature(args, mode, seed, repeat)
                payload = completed_summary(run_dir, args.generations, signature)
                if payload is None:
                    if (
                        run_dir.exists()
                        and any(run_dir.iterdir())
                        and not is_rerunnable_placeholder(run_dir, signature)
                    ):
                        rows.append(
                            {
                                "mode": mode,
                                "seed": seed,
                                "repeat": repeat,
                                "status": "incomplete_existing_run",
                                "run_dir": str(run_dir),
                            }
                        )
                        failures += 1
                        write_summary(args.output_dir / "pilot_summary.csv", rows)
                        continue
                    run_dir.mkdir(parents=True, exist_ok=True)
                    (run_dir / "pilot_run_config.json").write_text(
                        json.dumps(signature, indent=2), encoding="utf-8"
                    )
                    command = build_command(args, mode, seed, run_dir)
                    print(f"\n=== PILOT {mode} seed={seed} repeat={repeat} ===", flush=True)
                    completed = subprocess.run(command, cwd=ROOT, check=False)
                    if completed.returncode != 0:
                        rows.append(
                            {
                                "mode": mode,
                                "seed": seed,
                                "repeat": repeat,
                                "status": f"failed_exit_{completed.returncode}",
                                "run_dir": str(run_dir),
                            }
                        )
                        failures += 1
                        write_summary(args.output_dir / "pilot_summary.csv", rows)
                        continue
                    payload = completed_summary(run_dir, args.generations, signature)
                if payload is None:
                    rows.append(
                        {
                            "mode": mode,
                            "seed": seed,
                            "repeat": repeat,
                            "status": "missing_complete_summary",
                            "run_dir": str(run_dir),
                        }
                    )
                    failures += 1
                else:
                    rows.append(summary_row(mode, seed, repeat, run_dir, payload))
                write_summary(args.output_dir / "pilot_summary.csv", rows)

    print(
        f"Completed {len(rows) - failures}/{len(rows)} pilot runs; "
        f"summary: {args.output_dir / 'pilot_summary.csv'}"
    )
    effects = paired_effects(rows)
    if effects:
        effects_path = args.output_dir / "pilot_paired_effects.csv"
        with effects_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(effects[0]))
            writer.writeheader()
            writer.writerows(effects)
        print(f"Paired effects: {effects_path}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
