from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evolutionary_strategy import ESConfig, evolutionary_strategy_run
from run_es import DEFAULT_FILTER_PROMPT


def parse_args():
    parser = argparse.ArgumentParser(description="Run a small ES ablation grid.")
    parser.add_argument("--generations", type=int, default=2)
    parser.add_argument("--mu", type=int, default=3)
    parser.add_argument("--lambda", dest="lambda_", type=int, default=8)
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--csv", default="prompts/initial_population.csv")
    parser.add_argument("--output-root", default="outputs/experiments")
    parser.add_argument("--with-filter-coevolution", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    seeds = [int(seed.strip()) for seed in args.seeds.split(",") if seed.strip()]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for variant in ("one_fifth", "self_adaptive"):
        for selection in ("scalar", "lexicographic"):
            for coevolved in ([False, True] if args.with_filter_coevolution else [False]):
                for seed in seeds:
                    config = ESConfig(
                        lambda_=args.lambda_,
                        mu=args.mu,
                        generations=args.generations,
                        variant=variant,
                        csv_path=args.csv,
                        random_seed=seed,
                        lightweight=True,
                        selection_mode=selection,
                        filter_update_every=1 if coevolved else 0,
                        verbose=False,
                    )
                    result = evolutionary_strategy_run(
                        config=config,
                        client=None,
                        filter_prompt=DEFAULT_FILTER_PROMPT,
                    )
                    rows.append(
                        {
                            "variant": variant,
                            "selection": selection,
                            "coevolved_filter": coevolved,
                            "seed": seed,
                            "best_fitness": result.best.fitness,
                            "best_asv": result.best.metrics.get("asv", 0.0),
                            "best_mr": result.best.metrics.get("mr", 0.0),
                            "runtime_sec": result.runtime_sec,
                        }
                    )

    output_path = output_dir / "ablation_summary.csv"
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} ablation rows to {output_path}")


if __name__ == "__main__":
    main()
