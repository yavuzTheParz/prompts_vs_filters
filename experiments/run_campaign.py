"""Launch the fixed-budget parameter-analysis campaign, one run after another.

The campaign compares six search configurations (three population sizes times
two survival schemes) under a fixed budget of attacker model calls, with the
same seeds for every configuration so the seed is a block for the Friedman
test. It has two phases:

  fixed   every configuration with the filter held constant (parameter analysis)
  coevo   selected configurations with filter coevolution, on the same seeds

Finished runs (summary.json present) are skipped, interrupted runs
(checkpoint.pkl present) are resumed, and everything else starts fresh, so the
same command can be re-issued after any interruption. `--shard i/n` splits the
run list across machines or GPUs without overlap.

    # show what would run, without running anything
    python3 -B experiments/run_campaign.py --phase fixed --base-url http://HOST:8000 --plan

    # phase 1: 6 configurations x 20 seeds, fixed filter
    python3 -B experiments/run_campaign.py --phase fixed --base-url http://HOST:8000

    # phase 2: coevolution for the default and the best configuration
    python3 -B experiments/run_campaign.py --phase coevo \\
        --configs medium_plus,large_comma --base-url http://HOST:8000

    # two machines: the first takes shard 1/2, the second 2/2
    python3 -B experiments/run_campaign.py --phase fixed --shard 1/2 --base-url http://HOST_A:8000
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]

POPULATIONS = ("small", "medium", "large")
SURVIVAL = {"plus": "(mu+lambda)", "comma": "(mu,lambda)"}
ALL_CONFIGS = tuple(f"{population}_{survival}" for population in POPULATIONS for survival in SURVIVAL)
# Filter schedule as a share of the budget, so every budget size gets the same
# number of update attempts (13) after the same relative warm-up.
FILTER_INTERVAL_SHARE = 1 / 15
FILTER_WARMUP_SHARE = 2 / 15


def parse_seeds(text: str) -> List[int]:
    seeds: List[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            low, high = part.split("-", 1)
            seeds.extend(range(int(low), int(high) + 1))
        else:
            seeds.append(int(part))
    if len(set(seeds)) != len(seeds):
        raise ValueError("Duplicate seeds")
    return seeds


def parse_configs(text: str) -> List[str]:
    if text.strip().lower() == "all":
        return list(ALL_CONFIGS)
    configs = [item.strip() for item in text.split(",") if item.strip()]
    unknown = [item for item in configs if item not in ALL_CONFIGS]
    if unknown:
        raise ValueError(f"Unknown configurations {unknown}; choose from {', '.join(ALL_CONFIGS)}")
    return configs


def run_name(phase: str, config: str, seed: int) -> str:
    return f"{phase}_{config}_s{seed:02d}"


def run_state(run_dir: Path) -> str:
    """done: finished; resume: interrupted with a checkpoint; fresh: start over."""
    summary = run_dir / "summary.json"
    if summary.is_file():
        try:
            json.loads(summary.read_text(encoding="utf-8"))
            return "done"
        except json.JSONDecodeError:
            pass
    if (run_dir / "checkpoint.pkl").is_file():
        return "resume"
    return "fresh"


def build_command(args, phase: str, config: str, seed: int, run_dir: Path, state: str) -> List[str]:
    population, survival = config.split("_")
    command = [
        sys.executable, "-B", str(ROOT / "run_es.py"),
        "--preset", "full",
        "--population", population,
        "--survival", SURVIVAL[survival],
        "--seed", str(seed),
        "--max-evaluations", str(args.max_evaluations),
        "--llm-concurrency", str(args.concurrency),
        "--model", args.model,
        "--run-dir", str(run_dir),
        "--history-csv", str(run_dir / "history.csv"),
    ]
    if phase == "fixed":
        command += ["--filter-update-every-evaluations", "0"]
    else:
        command += [
            "--filter-update-every-evaluations",
            str(max(1, round(args.max_evaluations * FILTER_INTERVAL_SHARE))),
            "--filter-warmup-evaluations",
            str(round(args.max_evaluations * FILTER_WARMUP_SHARE)),
        ]
    if args.benign_holdout_csv:
        command += ["--benign-holdout-csv", args.benign_holdout_csv]
    if args.dry_run:
        command += ["--dry-run"]
    else:
        command += ["--base-url", args.base_url]
    if state == "resume":
        command += ["--resume"]
    return command


def plan(args) -> List[Dict[str, object]]:
    seeds = parse_seeds(args.seeds)
    configs = parse_configs(args.configs)
    root = Path(args.output_root)
    # Seeds vary slowest, so an interrupted campaign still holds complete
    # blocks (every configuration for the seeds finished so far).
    jobs = []
    for seed in seeds:
        for config in configs:
            name = run_name(args.phase, config, seed)
            jobs.append({"name": name, "config": config, "seed": seed, "run_dir": root / name})
    if args.shard:
        index, total = (int(part) for part in args.shard.split("/"))
        if not 1 <= index <= total:
            raise ValueError("--shard must be i/n with 1 <= i <= n")
        jobs = [job for position, job in enumerate(jobs) if position % total == index - 1]
    for job in jobs:
        job["state"] = run_state(job["run_dir"])
        job["command"] = build_command(args, args.phase, job["config"], job["seed"], job["run_dir"], job["state"])
    return jobs


def write_status(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["name", "config", "seed", "state", "outcome", "seconds", "attacker_calls", "stop_reason"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_summary(run_dir: Path) -> Dict[str, object]:
    try:
        return json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def execute(args, jobs: List[Dict[str, object]]) -> int:
    root = Path(args.output_root)
    suffix = f"_{args.shard.replace('/', 'of')}" if args.shard else ""
    status_path = root / f"campaign_status_{args.phase}{suffix}.csv"
    failures = 0
    for position, job in enumerate(jobs, start=1):
        run_dir: Path = job["run_dir"]
        if job["state"] == "done":
            summary = read_summary(run_dir)
            job.update(outcome="skipped_done", seconds=0.0,
                       attacker_calls=summary.get("attacker_calls"), stop_reason=summary.get("stop_reason"))
            write_status(status_path, jobs)
            continue
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"[{position}/{len(jobs)}] {job['name']} ({job['state']})", flush=True)
        started = time.time()
        with (run_dir / "stdout.log").open("a", encoding="utf-8") as log:
            log.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(job['command'])}\n")
            log.flush()
            code = subprocess.call(job["command"], cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT)
        summary = read_summary(run_dir)
        ok = code == 0 and bool(summary)
        failures += 0 if ok else 1
        job.update(
            outcome="completed" if ok else f"failed_exit_{code}",
            seconds=round(time.time() - started, 1),
            attacker_calls=summary.get("attacker_calls"),
            stop_reason=summary.get("stop_reason"),
        )
        write_status(status_path, jobs)
        print(f"    -> {job['outcome']} in {job['seconds'] / 3600:.2f} h", flush=True)
        if not ok and args.stop_on_failure:
            break
    print(f"Status: {status_path} ({failures} failed)")
    return 1 if failures else 0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--phase", choices=["fixed", "coevo"], required=True)
    parser.add_argument("--configs", default="all",
                        help=f"'all' or a comma list from: {', '.join(ALL_CONFIGS)}")
    parser.add_argument("--seeds", default="1-20", help="Comma list and/or ranges, e.g. 1-20 or 1,2,5-8.")
    parser.add_argument("--max-evaluations", type=int, default=15000,
                        help="Attacker model calls per run.")
    parser.add_argument("--concurrency", type=int, default=8, help="Parallel model requests per run.")
    parser.add_argument("--base-url", default=None, help="Model server URL (required unless --dry-run or --plan).")
    parser.add_argument("--model", default="dphn/Dolphin3.0-Llama3.1-8B",
                        help="Model label recorded in the manifest.")
    parser.add_argument("--benign-holdout-csv", default="prompts/benign_pilot_holdout.csv")
    parser.add_argument("--output-root", default="outputs/runs/campaign_v1")
    parser.add_argument("--shard", default=None, help="i/n: run only every n-th job starting at i.")
    parser.add_argument("--plan", action="store_true", help="Print the run list and commands, then exit.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run without a model server (pipeline check only; not evidence).")
    parser.add_argument("--stop-on-failure", action="store_true")
    args = parser.parse_args(argv)
    if not args.plan and not args.dry_run and not args.base_url:
        parser.error("--base-url is required for real runs")
    if args.plan and not args.base_url:
        args.base_url = "http://SERVER:8000"
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    jobs = plan(args)
    counts: Dict[str, int] = {}
    for job in jobs:
        counts[job["state"]] = counts.get(job["state"], 0) + 1
    print(f"{len(jobs)} runs in phase '{args.phase}': {counts}")
    if args.plan:
        for job in jobs:
            print(f"{job['state']:6s} {job['name']}")
            print("       " + " ".join(job["command"]))
        return 0
    return execute(args, jobs)


if __name__ == "__main__":
    raise SystemExit(main())
