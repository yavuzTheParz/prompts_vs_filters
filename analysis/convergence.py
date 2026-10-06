"""Convergence-speed and critical-point analysis for ES run directories.

Historical runs were scored by different evaluator versions, so their recorded
fitness columns are not comparable. This script re-scores every filtered sample
in `samples.jsonl` with the current evaluator and aggregates by generation,
giving one consistent search signal per run:

- offspring compliance rate: share of filtered samples labelled compliant
- mean attack progress: the graded search signal (see evaluators.attack_progress)

On top of that it reports, per run:

- first generation with any compliant sample, and t50/t90 convergence speed
  (first generation the smoothed signal reaches 50%/90% of its peak)
- critical improvement points: the largest rises of the best-so-far signal
- filter-update shocks: signal before/after each accepted update and the
  generations needed to recover half of the pre-update level
- zero-fitness plateau, quality-gate losses, and diversity collapse from the
  recorded generation summary

Usage:
    python3 -B analysis/convergence.py --run-dir outputs/main_v16_final \
        --run-dir outputs/main_v18_grammar_exfil --output-dir analysis/convergence
    python3 -B analysis/convergence.py --all-runs --output-dir analysis/convergence
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluators import (  # noqa: E402
    ATTACK_EVALUATOR_VERSION,
    ATTACK_PROGRESS_VERSION,
    DefensiveComplianceEvaluator,
    attack_progress,
)

csv.field_size_limit(sys.maxsize)


def _float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def smooth(values: Sequence[float], window: int) -> List[float]:
    """Trailing moving average; robust to the heavy per-generation noise."""
    out = []
    for index in range(len(values)):
        chunk = values[max(0, index - window + 1) : index + 1]
        out.append(sum(chunk) / len(chunk) if chunk else 0.0)
    return out


def first_reaching(series: Sequence[float], generations: Sequence[int], level: float) -> Optional[int]:
    for generation, value in zip(generations, series):
        if value >= level and value > 0.0:
            return generation
    return None


def longest_flat_run(series: Sequence[float], generations: Sequence[int]) -> Dict[str, Optional[int]]:
    """Longest stretch without a new best-so-far value."""
    best = float("-inf")
    run_start = generations[0] if generations else None
    longest = {"length": 0, "start": None, "end": None}
    for generation, value in zip(generations, series):
        if value > best + 1e-12:
            best = value
            run_start = generation
        length = generation - run_start
        if length > longest["length"]:
            longest = {"length": length, "start": run_start, "end": generation}
    return longest


def _correlation(left: Sequence[float], right: Sequence[float]) -> Optional[float]:
    if len(left) < 5 or statistics.pstdev(left) == 0 or statistics.pstdev(right) == 0:
        return None
    return round(statistics.correlation(left, right), 3)


def progress_lead(progress: Sequence[float], compliance: Sequence[float], lag: int = 3) -> Dict[str, Optional[float]]:
    """Does progress among non-compliant responses precede compliance?

    `progress - compliance` removes the part of progress that is compliance
    itself; a positive correlation with compliance `lag` generations later is
    evidence that the tie-breaker points the search toward compliance.
    """
    residual = [p - c for p, c in zip(progress, compliance)]
    return {
        "same_generation": _correlation(progress, compliance),
        f"noncompliant_progress_to_compliance_lag{lag}": (
            _correlation(residual[:-lag], compliance[lag:]) if len(residual) > lag else None
        ),
    }


def rescore_samples(run_dir: Path, evaluator: DefensiveComplianceEvaluator) -> Dict[int, Dict[str, float]]:
    per_generation: Dict[int, Dict[str, float]] = defaultdict(
        lambda: {"n": 0, "compliant": 0, "progress": 0.0, "filter_version": 0, "calls": 0}
    )
    path = run_dir / "samples.jsonl"
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            generation = int(_float(row.get("generation")))
            # Every recorded attempt is one attacker-side model call.
            per_generation[generation]["calls"] += 1
            if row.get("kind") != "filtered" or row.get("status") != "valid":
                continue
            result = evaluator.evaluate(row.get("text", ""))
            bucket = per_generation[generation]
            bucket["n"] += 1
            bucket["compliant"] += int(result.label == "compliant")
            bucket["progress"] += attack_progress(result)
            bucket["filter_version"] = max(
                bucket["filter_version"], int(_float(row.get("filter_version")))
            )
    return dict(per_generation)


def read_summary(run_dir: Path) -> List[Dict[str, str]]:
    """Final CSV when the run finished; streamed history for live or crashed runs."""
    path = run_dir / "generation_summary.csv"
    if path.is_file():
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    streamed = run_dir / "history.jsonl"
    if streamed.is_file():
        with streamed.open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    return []


def quarantine_status(run_dir: Path) -> Optional[str]:
    registry = ROOT / "experiments" / "invalid_runs.json"
    if not registry.is_file():
        return None
    data = json.loads(registry.read_text(encoding="utf-8"))
    relative = str(run_dir.resolve().relative_to(ROOT)) if run_dir.resolve().is_relative_to(ROOT) else str(run_dir)
    for entry in data.get("runs", []):
        if entry.get("path") == relative:
            return f"{entry.get('status')}:{entry.get('use')}"
    return None


def analyze_run(run_dir: Path, evaluator: DefensiveComplianceEvaluator, window: int) -> Dict[str, object]:
    config = {}
    if (run_dir / "config.json").is_file():
        config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    args = config.get("args", {})
    if not args and (run_dir / "checkpoint.pkl").is_file():
        import pickle

        with (run_dir / "checkpoint.pkl").open("rb") as handle:
            saved = pickle.load(handle).get("config", {})
        args = {**saved, "lambda_": saved.get("lambda_")}
    rescored = rescore_samples(run_dir, evaluator)
    summary_rows = read_summary(run_dir)

    generations = sorted(g for g in rescored if g >= 1 and rescored[g]["n"] > 0)
    cumulative, running_calls = {}, 0
    for g in sorted(rescored):
        running_calls += rescored[g]["calls"]
        cumulative[g] = running_calls
    calls_at = lambda g: cumulative.get(g) if g is not None else None  # noqa: E731
    compliance = [rescored[g]["compliant"] / rescored[g]["n"] for g in generations]
    progress = [rescored[g]["progress"] / rescored[g]["n"] for g in generations]
    smooth_compliance = smooth(compliance, window)
    smooth_progress = smooth(progress, window)

    peak = max(smooth_compliance, default=0.0)
    best_so_far = []
    running = 0.0
    for value in smooth_compliance:
        running = max(running, value)
        best_so_far.append(running)

    # Critical improvement points: largest single-generation rises of the
    # best-so-far smoothed compliance curve.
    rises = []
    for index in range(1, len(best_so_far)):
        delta = best_so_far[index] - best_so_far[index - 1]
        if delta > 0:
            rises.append(
                {
                    "generation": generations[index],
                    "from": round(best_so_far[index - 1], 4),
                    "to": round(best_so_far[index], 4),
                    "delta": round(delta, 4),
                    "share_of_peak": round(delta / peak, 3) if peak else 0.0,
                }
            )
    rises.sort(key=lambda row: row["delta"], reverse=True)
    critical_points = sorted(
        [row for row in rises if row["share_of_peak"] >= 0.10][:6],
        key=lambda row: row["generation"],
    )

    # Filter-update shocks.
    by_generation = {int(_float(row.get("generation"))): row for row in summary_rows}
    update_generations = [
        g for g, row in sorted(by_generation.items()) if _float(row.get("filter_changed")) > 0
    ]
    shocks = []
    index_of = {g: i for i, g in enumerate(generations)}
    for update in update_generations:
        before = [compliance[index_of[g]] for g in range(update - window + 1, update + 1) if g in index_of]
        after = [compliance[index_of[g]] for g in range(update + 1, update + window + 1) if g in index_of]
        pre = statistics.mean(before) if before else 0.0
        post = statistics.mean(after) if after else 0.0
        recovery = None
        if pre > 0:
            for g in generations:
                if g > update and smooth_compliance[index_of[g]] >= 0.5 * pre:
                    recovery = g - update
                    break
        shocks.append(
            {
                "generation": update,
                "pre_compliance_rate": round(pre, 4),
                "post_compliance_rate": round(post, 4),
                "drop": round(pre - post, 4),
                "generations_to_half_recovery": recovery,
            }
        )

    best_fitness = [_float(row.get("best_fitness")) for row in summary_rows]
    mean_fitness = [_float(row.get("mean_parent_fitness", row.get("mean_fitness"))) for row in summary_rows]
    mu = int(_float(args.get("mu"), 1)) or 1
    rejected = defaultdict(float)
    for row in summary_rows:
        for key, value in row.items():
            if key.startswith("rejected_"):
                rejected[key[len("rejected_"):]] += _float(value)
    parent_slots = mu * max(1, len(summary_rows))
    diversity = [_float(row.get("population_diversity")) for row in summary_rows]
    diversity_collapse = next(
        (int(_float(row.get("generation"))) for row in summary_rows if 0 < _float(row.get("population_diversity")) < 0.30),
        None,
    )
    mean_zero_from = None
    for index in range(len(mean_fitness)):
        if all(value == 0.0 for value in mean_fitness[index:]):
            mean_zero_from = int(_float(summary_rows[index].get("generation")))
            break

    first_attack = next((g for g, value in zip(generations, compliance) if value > 0), None)
    total_samples = sum(rescored[g]["n"] for g in generations)
    total_compliant = sum(rescored[g]["compliant"] for g in generations)
    early = [value for g, value in zip(generations, compliance) if g <= max(1, len(generations) // 4)]
    late = [value for g, value in zip(generations, compliance) if g > 3 * len(generations) // 4]

    return {
        "run": run_dir.name,
        "path": str(run_dir),
        "quarantine": quarantine_status(run_dir),
        "config": {
            "mu": args.get("mu"),
            "lambda": args.get("lambda_"),
            "generations": args.get("generations"),
            "k_evals": args.get("k_evals"),
            "filter_update_every": args.get("filter_update_every"),
            "selection_mode": args.get("selection_mode"),
            "filtered_temperature": args.get("filtered_temperature"),
        },
        "generations_observed": len(generations),
        "filtered_samples": total_samples,
        "overall_compliance_rate": round(total_compliant / total_samples, 4) if total_samples else 0.0,
        "early_quarter_compliance_rate": round(statistics.mean(early), 4) if early else 0.0,
        "late_quarter_compliance_rate": round(statistics.mean(late), 4) if late else 0.0,
        "mean_attack_progress": round(statistics.mean(progress), 4) if progress else 0.0,
        "progress_compliance_correlation": progress_lead(progress, compliance),
        "first_compliant_generation": first_attack,
        "peak_smoothed_compliance_rate": round(peak, 4),
        "t50_generation": first_reaching(smooth_compliance, generations, 0.5 * peak),
        "t90_generation": first_reaching(smooth_compliance, generations, 0.9 * peak),
        "first_compliant_evaluations": calls_at(first_attack),
        "t50_evaluations": calls_at(first_reaching(smooth_compliance, generations, 0.5 * peak)),
        "t90_evaluations": calls_at(first_reaching(smooth_compliance, generations, 0.9 * peak)),
        "total_attacker_calls": running_calls,
        "peak_generation": generations[smooth_compliance.index(peak)] if generations and peak > 0 else None,
        "longest_no_improvement": longest_flat_run(smooth_compliance, generations),
        "critical_improvement_points": critical_points,
        "filter_updates": len(update_generations),
        "filter_shocks": shocks,
        "recorded_best_fitness_zero_fraction": (
            round(sum(value == 0.0 for value in best_fitness) / len(best_fitness), 3) if best_fitness else None
        ),
        "recorded_mean_fitness_zero_from_generation": mean_zero_from,
        "quality_gate_rejections_per_parent_slot": {
            key: round(value / parent_slots, 3) for key, value in sorted(rejected.items()) if value
        },
        "diversity_start": round(diversity[0], 3) if diversity else None,
        "diversity_end": round(diversity[-1], 3) if diversity else None,
        "diversity_collapse_generation": diversity_collapse,
        "series": {
            "generation": generations,
            "attacker_calls": [cumulative[g] for g in generations],
            "compliance_rate": [round(v, 4) for v in compliance],
            "smoothed_compliance_rate": [round(v, 4) for v in smooth_compliance],
            "mean_attack_progress": [round(v, 4) for v in progress],
            "smoothed_attack_progress": [round(v, 4) for v in smooth_progress],
            "filter_update_generations": update_generations,
        },
    }


def plot(results: List[Dict[str, object]], path: Path) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    columns = 2
    rows = (len(results) + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(12, 3.2 * rows), squeeze=False)
    for axis, result in zip(axes.flat, results):
        series = result["series"]
        axis.plot(series["generation"], series["smoothed_compliance_rate"], color="#1f6feb", label="compliance (smoothed)")
        axis.plot(series["generation"], series["smoothed_attack_progress"], color="#8a8a8a", label="progress (smoothed)")
        for generation in series["filter_update_generations"]:
            axis.axvline(generation, color="#d1242f", alpha=0.25, linewidth=1)
        for point in result["critical_improvement_points"]:
            axis.axvline(point["generation"], color="#1a7f37", alpha=0.6, linestyle="--", linewidth=1)
        axis.set_title(result["run"], fontsize=9)
        axis.set_ylim(0, 1)
        axis.tick_params(labelsize=7)
    for axis in list(axes.flat)[len(results):]:
        axis.axis("off")
    axes.flat[0].legend(fontsize=7, loc="upper right")
    figure.suptitle("Re-scored offspring signal (red: filter update, green dashed: critical improvement)", fontsize=10)
    figure.tight_layout(rect=(0, 0, 1, 0.985))
    figure.savefig(path, dpi=120)
    plt.close(figure)
    return True


def write_markdown(results: List[Dict[str, object]], path: Path) -> None:
    lines = [
        "# Convergence analysis",
        "",
        f"All samples re-scored with `{ATTACK_EVALUATOR_VERSION}` and `{ATTACK_PROGRESS_VERSION}`.",
        "Compliance rate = share of filtered samples labelled compliant per generation.",
        "",
        "| run | μ/λ/G/K | filter every | overall | early¼ | late¼ | first hit | t50 | t90 | t90 (calls) | calls | peak | longest flat | filter updates | mean fit = 0 from | diversity start→end |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        c = r["config"]
        flat = r["longest_no_improvement"]
        lines.append(
            f"| {r['run']}{' ⚠' if r['quarantine'] else ''} | {c['mu']}/{c['lambda']}/{c['generations']}/{c['k_evals']} "
            f"| {c['filter_update_every']} | {r['overall_compliance_rate']:.3f} | {r['early_quarter_compliance_rate']:.3f} "
            f"| {r['late_quarter_compliance_rate']:.3f} | {r['first_compliant_generation']} | {r['t50_generation']} "
            f"| {r['t90_generation']} | {r['t90_evaluations']} | {r['total_attacker_calls']} "
            f"| {r['peak_smoothed_compliance_rate']:.3f}@{r['peak_generation']} "
            f"| {flat['length']} ({flat['start']}–{flat['end']}) | {r['filter_updates']} "
            f"| {r['recorded_mean_fitness_zero_from_generation']} | {r['diversity_start']}→{r['diversity_end']} |"
        )
    lines += ["", "⚠ = listed in experiments/invalid_runs.json (diagnostic use only).", ""]
    leads = [
        r["progress_compliance_correlation"]["noncompliant_progress_to_compliance_lag3"]
        for r in results
        if r["progress_compliance_correlation"]["noncompliant_progress_to_compliance_lag3"] is not None
    ]
    if leads:
        lines += [
            "## Does attack progress lead compliance?",
            "",
            f"Correlation of non-compliant progress with compliance 3 generations later: "
            f"median {statistics.median(leads):.2f}, positive in {sum(v > 0 for v in leads)}/{len(leads)} runs. "
            "Weak or mixed values mean the tie-breaker is unproven on this model; "
            "compare against `--no-progress-tiebreak`.",
            "",
        ]
    for r in results:
        lines.append(f"## {r['run']}")
        if r["critical_improvement_points"]:
            lines.append("Critical improvement points (best-so-far smoothed compliance):")
            for point in r["critical_improvement_points"]:
                lines.append(
                    f"- gen {point['generation']}: {point['from']:.3f} → {point['to']:.3f} "
                    f"(+{point['delta']:.3f}, {point['share_of_peak']:.0%} of peak)"
                )
        if r["filter_shocks"]:
            lines.append("Filter-update shocks:")
            for shock in r["filter_shocks"]:
                lines.append(
                    f"- gen {shock['generation']}: {shock['pre_compliance_rate']:.3f} → "
                    f"{shock['post_compliance_rate']:.3f}, half-recovery after "
                    f"{shock['generations_to_half_recovery']} generations"
                )
        if r["quality_gate_rejections_per_parent_slot"]:
            lines.append(f"Quality-gate rejections per parent slot: {r['quality_gate_rejections_per_parent_slot']}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", action="append", default=[], help="Run directory with samples.jsonl.")
    parser.add_argument("--all-runs", action="store_true", help="Analyze every outputs/*/ directory with samples.jsonl.")
    parser.add_argument("--window", type=int, default=5, help="Smoothing window in generations.")
    parser.add_argument("--output-dir", default="analysis/convergence")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    run_dirs = [Path(path) for path in args.run_dir]
    if args.all_runs:
        run_dirs += sorted(path.parent for path in (ROOT / "outputs").glob("*/samples.jsonl"))
    if not run_dirs:
        raise SystemExit("Pass --run-dir or --all-runs")
    evaluator = DefensiveComplianceEvaluator()
    results = [analyze_run(run_dir, evaluator, max(1, args.window)) for run_dir in run_dirs]
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "convergence.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    write_markdown(results, output / "convergence.md")
    plotted = plot(results, output / "convergence.png")
    print(f"Wrote {output / 'convergence.json'}, {output / 'convergence.md'}" + (f", {output / 'convergence.png'}" if plotted else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
