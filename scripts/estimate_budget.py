"""Estimate model calls and wall time for an ES run before launching it.

Per generation the search spends, for each evaluated child, one deterministic
direct sample plus K filtered samples; surviving parents add parent_resample_k
samples each until max_samples_per_prompt. Each filter update evaluates the
benign set and the top-k attack prompts once for the current filter and once
per candidate rule (up to 1 LLM proposal + ~9 fallback rules), then
re-evaluates all parents under an accepted filter.

    python3 -B scripts/estimate_budget.py --mu 16 --lambda 64 --generations 300 \
        --k-evals 3 --parent-resample-k 1 --filter-update-every 15 \
        --seconds-per-call 2.5 --concurrency 8
"""

from __future__ import annotations

import argparse


def estimate_fixed_budget(args) -> dict:
    """Budget mode: attacker calls are fixed; generations and defender calls follow."""
    duplicate_share = max(0.0, min(1.0, args.duplicate_share))
    per_generation = args.lambda_ * (1.0 - duplicate_share) * (1 + args.k_evals)
    per_generation += args.mu * args.parent_resample_k
    generations = args.max_evaluations / per_generation
    updates = 0
    if args.filter_update_every_evaluations > 0:
        updates = max(
            0,
            (args.max_evaluations - args.filter_warmup_evaluations)
            // args.filter_update_every_evaluations,
        )
    defender = updates * ((args.benign_prompts + args.top_k_filter) * (1 + args.candidate_rules) + 1)
    total = args.max_evaluations + defender + args.final_k_evals + 1
    hours = total * args.seconds_per_call / max(1, args.concurrency) / 3600.0
    return {
        "attacker_calls": args.max_evaluations,
        "approx_generations": round(generations),
        "filter_update_attempts": updates,
        "defender_calls": round(defender),
        "final_confirmation_calls": args.final_k_evals + 1,
        "total_calls": round(total),
        "estimated_hours": round(hours, 1),
        "estimated_hours_sequential": round(total * args.seconds_per_call / 3600.0, 1),
    }


def estimate(args) -> dict:
    if getattr(args, "max_evaluations", 0):
        return estimate_fixed_budget(args)
    duplicate_share = max(0.0, min(1.0, args.duplicate_share))
    children = args.lambda_ * (1.0 - duplicate_share)
    search_calls = args.generations * children * (1 + args.k_evals)
    # Upper bound: every parent slot gains samples each generation. Long-lived
    # elites stop at max_samples_per_prompt, so the real figure is lower.
    resample_calls = args.generations * args.mu * args.parent_resample_k
    updates = 0
    if args.filter_update_every > 0:
        updates = max(0, (args.generations - args.filter_warmup_generations) // args.filter_update_every)
    per_update = (args.benign_prompts + args.top_k_filter) * (1 + args.candidate_rules) + 1
    per_update += args.mu * (1 + args.k_evals)
    filter_calls = updates * per_update
    confirmation = args.final_k_evals + 1
    total = search_calls + resample_calls + filter_calls + confirmation
    hours = total * args.seconds_per_call / max(1, args.concurrency) / 3600.0
    return {
        "search_calls": round(search_calls),
        "parent_resample_calls": round(resample_calls),
        "filter_update_calls": round(filter_calls),
        "filter_updates": updates,
        "final_confirmation_calls": confirmation,
        "total_calls": round(total),
        "estimated_hours": round(hours, 1),
        "estimated_hours_sequential": round(total * args.seconds_per_call / 3600.0, 1),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mu", type=int, default=16)
    parser.add_argument("--lambda", dest="lambda_", type=int, default=64)
    parser.add_argument("--generations", type=int, default=300)
    parser.add_argument("--k-evals", type=int, default=3)
    parser.add_argument("--parent-resample-k", type=int, default=1)
    parser.add_argument("--max-samples-per-prompt", type=int, default=12)
    parser.add_argument("--filter-update-every", type=int, default=15)
    parser.add_argument("--filter-warmup-generations", type=int, default=30)
    parser.add_argument("--top-k-filter", type=int, default=8)
    parser.add_argument("--benign-prompts", type=int, default=40)
    parser.add_argument("--candidate-rules", type=int, default=4,
                        help="Average candidate rules evaluated per update (1 proposal + fallbacks until one is accepted).")
    parser.add_argument("--final-k-evals", type=int, default=16)
    parser.add_argument("--duplicate-share", type=float, default=0.05,
                        help="Share of offspring skipped as exact duplicates.")
    parser.add_argument("--max-evaluations", type=int, default=0,
                        help="Fixed attacker-call budget; switches to budget mode.")
    parser.add_argument("--filter-update-every-evaluations", type=int, default=0)
    parser.add_argument("--filter-warmup-evaluations", type=int, default=0)
    parser.add_argument("--seconds-per-call", type=float, default=2.5)
    parser.add_argument("--concurrency", type=int, default=8)
    args = parser.parse_args(argv)
    for key, value in estimate(args).items():
        print(f"{key:28s} {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
