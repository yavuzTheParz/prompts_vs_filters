from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.run_registry import DEFAULT_REGISTRY, assess_run


SUMMARY_METRICS = (
    "best_fitness",
    "best_attack_objective",
    "best_compliance",
    "best_similarity",
    "best_mr",
    "best_diversity",
    "best_prompt_length",
)
CONVERGENCE_METRICS = (
    "best_fitness",
    "best_attack_objective",
    "mean_mr",
    "population_diversity",
    "sigma",
    "filter_new_benign_refusal_rate",
)


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def complete_rows(rows: list[dict], requested_generations: int) -> list[dict]:
    return [
        row
        for row in rows
        if row.get("status") == "complete"
        and int(float(row.get("generations", 0))) == requested_generations
    ]


def mean_ci(values: list[float]) -> tuple[float, float, float]:
    mean = statistics.fmean(values)
    if len(values) < 2:
        return mean, mean, mean
    half_width = 1.96 * statistics.stdev(values) / math.sqrt(len(values))
    return mean, mean - half_width, mean + half_width


def cohens_d(values: list[float], reference: list[float]) -> float:
    if len(values) < 2 or len(reference) < 2:
        return 0.0
    pooled_variance = (
        (len(values) - 1) * statistics.variance(values)
        + (len(reference) - 1) * statistics.variance(reference)
    ) / (len(values) + len(reference) - 2)
    if pooled_variance <= 0:
        return 0.0
    return (statistics.fmean(values) - statistics.fmean(reference)) / math.sqrt(
        pooled_variance
    )


def summarize(rows: list[dict]) -> list[dict]:
    by_condition: dict[str, list[dict]] = {}
    for row in rows:
        by_condition.setdefault(row["condition"], []).append(row)
    reference = by_condition.get("A", [])
    output = []
    for condition, group in sorted(by_condition.items()):
        for metric in SUMMARY_METRICS:
            values = [float(row[metric]) for row in group]
            ref_values = [float(row[metric]) for row in reference]
            mean, low, high = mean_ci(values)
            output.append(
                {
                    "condition": condition,
                    "filter_mode": group[0]["filter_mode"],
                    "metric": metric,
                    "n": len(values),
                    "mean": mean,
                    "ci95_low": low,
                    "ci95_high": high,
                    "cohens_d_vs_A": cohens_d(values, ref_values),
                }
            )
    return output


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def convergence_series(rows: list[dict], metric: str) -> dict[str, list[tuple[int, float]]]:
    grouped: dict[tuple[str, int], list[float]] = {}
    for row in rows:
        value = row.get(metric, "")
        if value in ("", None):
            continue
        key = (row["condition"], int(float(row["generation"])))
        grouped.setdefault(key, []).append(float(value))
    series: dict[str, list[tuple[int, float]]] = {}
    for (condition, generation), values in grouped.items():
        series.setdefault(condition, []).append((generation, statistics.fmean(values)))
    return {condition: sorted(points) for condition, points in series.items()}


def write_svg(path: Path, title: str, series: dict[str, list[tuple[int, float]]]) -> None:
    width, height = 900, 480
    left, top, right, bottom = 70, 45, 25, 55
    points = [point for values in series.values() for point in values]
    x_values = [point[0] for point in points] or [0, 1]
    y_values = [point[1] for point in points] or [0.0, 1.0]
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)
    if x_min == x_max:
        x_max += 1
    if y_min == y_max:
        y_min -= 0.5
        y_max += 0.5

    def sx(value):
        return left + (value - x_min) / (x_max - x_min) * (width - left - right)

    def sy(value):
        return height - bottom - (value - y_min) / (y_max - y_min) * (
            height - top - bottom
        )

    colors = ("#2457C5", "#D0473E", "#198754", "#8B5CF6", "#E18A00", "#008C95", "#555555")
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{left}" y="25" font-family="sans-serif" font-size="18">{title}</text>',
        f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#333"/>',
        f'<text x="{width/2}" y="{height-12}" text-anchor="middle" font-family="sans-serif" font-size="12">generation</text>',
        f'<text x="18" y="{height/2}" transform="rotate(-90 18 {height/2})" text-anchor="middle" font-family="sans-serif" font-size="12">{title}</text>',
    ]
    for index, (condition, values) in enumerate(sorted(series.items())):
        color = colors[index % len(colors)]
        coordinates = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in values)
        lines.append(f'<polyline points="{coordinates}" fill="none" stroke="{color}" stroke-width="2"/>')
        lines.append(
            f'<text x="{width-right-80}" y="{top+16*index}" font-family="sans-serif" '
            f'font-size="11" fill="{color}">{condition}</text>'
        )
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def baseline_comparison(path: Path) -> list[dict]:
    rows = read_csv(path)
    grouped: dict[str, list[float]] = {}
    for row in rows:
        grouped.setdefault(row["handling"], []).append(float(row["mr"]))
    pre = grouped["pre_fix_compatible"]
    post = grouped["post_fix_prompt_specific"]
    return [
        {
            "comparison": "post_fix_prompt_specific_minus_pre_fix_compatible",
            "metric": "mr",
            "pre_fix_mean": statistics.fmean(pre),
            "post_fix_mean": statistics.fmean(post),
            "mean_difference": statistics.fmean(post) - statistics.fmean(pre),
            "cohens_d": cohens_d(post, pre),
            "source": "sanitized_aggregate_fixture",
        }
    ]


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze T10 ablation artifacts.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--invalid-run-registry",
        type=Path,
        default=DEFAULT_REGISTRY,
        help="Machine-readable registry of runs excluded from analysis.",
    )
    parser.add_argument(
        "--baseline-fixture",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "experiments"
        / "configs"
        / "baseline_handling_fixture.csv",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    eligibility = assess_run(
        args.run_dir,
        registry_path=args.invalid_run_registry,
    )
    if not eligibility["eligible"]:
        report = {
            "status": "excluded_by_registry",
            "included_runs": 0,
            "excluded_runs": 1,
            "run_id": eligibility["run_id"],
            "run_path": eligibility["path"],
            "exclusion_reasons": eligibility["exclusion_reasons"],
            "registry": eligibility["registry"],
            "plots": [],
        }
        (args.output_dir / "exclusions.json").write_text(
            json.dumps([eligibility], indent=2) + "\n",
            encoding="utf-8",
        )
        (args.output_dir / "analysis_manifest.json").write_text(
            json.dumps(report, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            f"Excluded quarantined run {eligibility['run_id']}: "
            + ", ".join(eligibility["exclusion_reasons"])
        )
        return 2

    manifest = json.loads((args.run_dir / "experiment_manifest.json").read_text())
    requested_generations = int(manifest["generations"])
    all_rows = read_csv(args.run_dir / "run_summary.csv")
    included = complete_rows(all_rows, requested_generations)
    excluded = [row for row in all_rows if row not in included]
    if not included:
        raise ValueError("No complete runs available for analysis")

    stats = summarize(included)
    write_csv(args.output_dir / "condition_statistics.csv", stats)
    write_csv(args.output_dir / "baseline_handling_comparison.csv", baseline_comparison(args.baseline_fixture))
    (args.output_dir / "exclusions.json").write_text(
        json.dumps(excluded, indent=2), encoding="utf-8"
    )

    history = read_csv(args.run_dir / "generation_history.csv")
    plot_files = []
    for metric in CONVERGENCE_METRICS:
        path = args.output_dir / f"convergence_{metric}.svg"
        write_svg(path, metric, convergence_series(history, metric))
        plot_files.append(path.name)

    report = {
        "status": "complete",
        "registry_status": eligibility["status"],
        "included_runs": len(included),
        "excluded_runs": len(excluded),
        "conditions": sorted({row["condition"] for row in included}),
        "filter_modes_reported_separately": sorted(
            {row["filter_mode"] for row in included}
        ),
        "plots": plot_files,
        "interpretation_fields": [
            "best_compliance",
            "best_similarity",
            "best_prompt_length",
            "best_diversity",
        ],
    }
    (args.output_dir / "analysis_manifest.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(f"Analyzed {len(included)} runs; excluded {len(excluded)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
