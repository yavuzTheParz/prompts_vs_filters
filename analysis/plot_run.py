"""Per-run result plots for ES experiments.

Reads the per-generation history of one run and writes two PNGs:

- ``fitness.png``: best, best-ever, parent-mean and offspring-mean fitness
- ``dashboard.png``: fitness plus attack objective, attack-success rates,
  population response labels, MR, prompt quality/diversity, mutation step
  size, tone probabilities, quality-gate rejections, and filter length

Accepted inputs: a run directory (``generation_summary.csv``, or
``history.jsonl`` for live/crashed runs) or a history CSV such as
``outputs/es_history.csv``. Panels whose columns are missing (older runs) are
skipped. Vertical dashed lines mark accepted filter updates; dotted lines mark
stagnation restarts.

``run_es.py`` calls :func:`plot_run` automatically when a run finishes.

Usage:
    python3 -B analysis/plot_run.py outputs/main_v17_garbled_filter_fallback
    python3 -B analysis/plot_run.py outputs/es_history.csv --output-dir outputs/es_plots
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

csv.field_size_limit(sys.maxsize)

# Validated categorical order (fixed; colors follow the series, never its rank).
# Across panels: best = blue, parents = orange, offspring = aqua.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
MUTED = "#8a8985"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"
FILTER_LINE = "#52514e"

RESPONSE_LABELS = [
    ("population_compliant_count", "compliant"),
    ("population_ambiguous_count", "ambiguous"),
    ("population_benign_educational_count", "benign-educational"),
    ("population_refusal_count", "refusal"),
    ("population_invalid_count", "invalid"),
]
MAX_REJECTION_SERIES = 5


def _float(value, default: Optional[float] = None) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def load_history(source: Path) -> List[Dict[str, object]]:
    """Rows of one run, from a run directory or a history CSV/JSONL file."""
    if source.is_dir():
        for name in ("generation_summary.csv", "history.jsonl"):
            if (source / name).is_file():
                return load_history(source / name)
        return []
    if source.suffix == ".jsonl":
        with source.open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    with source.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _column(rows: Sequence[Dict[str, object]], *names: str) -> Optional[List[float]]:
    """First of ``names`` present with at least one numeric value."""
    for name in names:
        if rows and name in rows[0]:
            values = [_float(row.get(name)) for row in rows]
            if any(value is not None for value in values):
                return [float("nan") if value is None else value for value in values]
    return None


def _running_max(values: Sequence[float]) -> List[float]:
    out, best = [], float("-inf")
    for value in values:
        if value == value:  # skip NaN
            best = max(best, value)
        out.append(best if best != float("-inf") else float("nan"))
    return out


def _events(rows: Sequence[Dict[str, object]], generations: Sequence[float], name: str) -> List[float]:
    return [g for g, row in zip(generations, rows) if (_float(row.get(name), 0.0) or 0.0) > 0]


def _style_axis(axis, title: str, ylabel: str = "", unit_range: bool = False) -> None:
    axis.set_title(title, loc="left", fontsize=10, color=TEXT, fontweight="semibold")
    axis.set_facecolor(SURFACE)
    axis.grid(axis="y", color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(GRID)
    axis.tick_params(colors=TEXT_SECONDARY, labelsize=8, length=0)
    axis.set_xlabel("generation", fontsize=8, color=TEXT_SECONDARY)
    if ylabel:
        axis.set_ylabel(ylabel, fontsize=8, color=TEXT_SECONDARY)
    if unit_range:
        axis.set_ylim(-0.02, 1.02)


def _legend(axis) -> None:
    """One row above the plot area, so it never covers data."""
    handles, labels = axis.get_legend_handles_labels()
    if len(handles) > 1:
        axis.legend(handles, labels, fontsize=7, frameon=False, loc="lower right",
                    bbox_to_anchor=(1.0, 1.0), ncol=min(len(handles), 3),
                    labelcolor=TEXT_SECONDARY, handlelength=1.6, columnspacing=1.0,
                    borderaxespad=0.2)


def _mark_events(axis, filter_updates: Sequence[float], restarts: Sequence[float]) -> None:
    for generation in filter_updates:
        axis.axvline(generation, color=FILTER_LINE, alpha=0.35, linestyle="--", linewidth=1, zorder=0)
    for generation in restarts:
        axis.axvline(generation, color=SERIES[7], alpha=0.5, linestyle=":", linewidth=1.2, zorder=0)


def _lines(axis, generations, series: Sequence[tuple]) -> None:
    for label, values, color, style in series:
        axis.plot(generations, values, color=color, linewidth=2 if style == "-" else 1.5,
                  linestyle=style, label=label)
    _legend(axis)


def _fitness_series(rows, generations) -> List[tuple]:
    best = _column(rows, "best_fitness")
    if best is None:
        return []
    best_ever = _column(rows, "best_ever_fitness") or _running_max(best)
    series = [
        ("best-ever", best_ever, SERIES[0], "-"),
        ("generation best", best, SERIES[0], "--"),
    ]
    parent_mean = _column(rows, "mean_parent_fitness", "mean_fitness")
    if parent_mean is not None:
        series.append(("parent mean", parent_mean, SERIES[1], "-"))
    offspring_mean = _column(rows, "offspring_mean_fitness")
    if offspring_mean is not None:
        series.append(("offspring mean", offspring_mean, SERIES[2], "-"))
    return series


def _panels(rows, generations) -> List[Dict[str, object]]:
    """Every panel with data; each has one y-scale."""
    panels = []

    def add(title, series, ylabel="", unit_range=False, kind="lines"):
        series = [item for item in series if item[1] is not None]
        if series:
            panels.append({"title": title, "series": series, "ylabel": ylabel,
                           "unit_range": unit_range, "kind": kind})

    add("Fitness", _fitness_series(rows, generations), "fitness")
    add("Attack objective", [
        ("best", _column(rows, "best_attack_objective", "best_asv"), SERIES[0], "-"),
        ("parent mean", _column(rows, "mean_attack_objective", "mean_asv"), SERIES[1], "-"),
        ("offspring mean progress", _column(rows, "offspring_mean_attack_progress"), SERIES[2], "-"),
    ], "score", unit_range=True)
    add("Attack success rate", [
        ("parents", _column(rows, "mean_attack_success"), SERIES[1], "-"),
        ("offspring", _column(rows, "offspring_attack_success_rate"), SERIES[2], "-"),
        ("ES success rate", _column(rows, "success_rate"), SERIES[3], "-"),
    ], "rate", unit_range=True)
    counts = [(label, _column(rows, column)) for column, label in RESPONSE_LABELS]
    label_totals = [sum(values[i] for _, values in counts if values is not None and values[i] == values[i])
                    for i in range(len(rows))]
    add("Parent response labels", [] if not any(label_totals) else [
        (label, None if values is None else [v / t if t and v == v else 0.0 for v, t in zip(values, label_totals)],
         SERIES[index], "-")
        for index, (label, values) in enumerate(counts)
    ], "share of samples", unit_range=True, kind="stack")
    add("Metamorphic relation (MR)", [
        ("best MR", _column(rows, "best_mr"), SERIES[0], "-"),
        ("parent mean MR", _column(rows, "mean_mr"), SERIES[1], "-"),
    ], "MR", unit_range=True)
    add("Prompt quality & diversity", [
        ("population diversity", _column(rows, "population_diversity"), SERIES[0], "-"),
        ("mean fluency", _column(rows, "mean_fluency"), SERIES[1], "-"),
        ("mean garbled-token ratio", _column(rows, "mean_garbled_token_ratio"), SERIES[2], "-"),
    ], "score", unit_range=True)
    add("Mutation step size (σ)", [("σ", _column(rows, "sigma"), SERIES[0], "-")], "σ")

    tones = sorted(key[len("tone_prob_"):] for key in (rows[0] if rows else {}) if key.startswith("tone_prob_"))
    add("Tone probabilities", [
        (tone, _column(rows, f"tone_prob_{tone}"), SERIES[index % len(SERIES)], "-")
        for index, tone in enumerate(tones[: len(SERIES)])
    ], "probability", unit_range=True)

    rejection_columns = [key for key in (rows[0] if rows else {}) if key.startswith("rejected_")]
    totals = {key: sum(v for v in (_column(rows, key) or []) if v == v) for key in rejection_columns}
    ranked = [key for key in sorted(totals, key=totals.get, reverse=True) if totals[key] > 0]
    if ranked:
        shown, rest = ranked[:MAX_REJECTION_SERIES], ranked[MAX_REJECTION_SERIES:]
        series = [(key[len("rejected_"):].replace("_", " "), _column(rows, key), SERIES[i], "-")
                  for i, key in enumerate(shown)]
        if rest:
            other = [sum((_column(rows, key) or [0.0])[i] for key in rest) for i in range(len(rows))]
            series.append(("other", other, MUTED, "-"))
        title = "Quality-gate rejections" + (f" ({series[0][0]})" if len(series) == 1 else "")
        add(title, series, "offspring", kind="bars")

    add("Filter prompt length", [("chars", _column(rows, "filter_length"), SERIES[0], "-")], "characters", kind="step")
    return panels


def _draw(axis, panel, generations) -> None:
    kind, series = panel["kind"], panel["series"]
    if kind == "stack":
        values = [[0.0 if v != v else v for v in item[1]] for item in series]
        axis.stackplot(generations, *values, colors=[item[2] for item in series],
                       labels=[item[0] for item in series], edgecolor=SURFACE, linewidth=0.6)
        _legend(axis)
    elif kind == "bars":
        bottom = [0.0] * len(generations)
        steps = [b - a for a, b in zip(generations, generations[1:]) if b > a]
        width = 0.8 * min(steps) if steps else 0.8
        for label, values, color, _ in series:
            values = [0.0 if v != v else v for v in values]
            axis.bar(generations, values, bottom=bottom, color=color, width=width, label=label,
                     edgecolor=SURFACE, linewidth=0.4)
            bottom = [b + v for b, v in zip(bottom, values)]
        _legend(axis)
    elif kind == "step":
        label, values, color, _ = series[0]
        axis.step(generations, values, where="post", color=color, linewidth=2)
    else:
        _lines(axis, generations, series)


def _headline(rows, generations, run_name: str, summary: Dict[str, object]) -> str:
    best = _column(rows, "best_fitness") or []
    best_ever = _column(rows, "best_ever_fitness") or _running_max(best)
    peak = max((v for v in best_ever if v == v), default=float("nan"))
    peak_gen = next((g for g, v in zip(generations, best_ever) if v == peak), None)
    updates = len(_events(rows, generations, "filter_changed"))
    parts = [
        run_name,
        f"{len(rows)} generations",
        f"best-ever fitness {peak:.4f}" + (f" @ gen {peak_gen:g}" if peak_gen is not None else ""),
        f"final best {best[-1]:.4f}" if best else "",
        f"{updates} filter updates",
    ]
    if summary.get("runtime_sec") is not None:
        seconds = float(summary["runtime_sec"])
        parts.append(f"runtime {seconds / 60:.1f} min" if seconds >= 60 else f"runtime {seconds:.1f} s")
    return "  ·  ".join(part for part in parts if part)


def plot_run(source, output_dir=None, title: Optional[str] = None) -> List[Path]:
    """Write ``fitness.png`` and ``dashboard.png``; returns the written paths.

    Returns an empty list when there is no history or matplotlib is missing.
    """
    source = Path(source)
    rows = load_history(source)
    if not rows:
        return []
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []

    if output_dir is None:
        output_dir = source / "plots" if source.is_dir() else source.with_name(f"{source.stem}_plots")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    generations = _column(rows, "generation") or [float(i + 1) for i in range(len(rows))]
    filter_updates = _events(rows, generations, "filter_changed")
    restarts = _events(rows, generations, "restart_triggered")
    summary = {}
    summary_path = (source if source.is_dir() else source.parent) / "summary.json"
    if source.is_dir() and summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    run_name = title or (source.name if source.is_dir() else source.stem)
    headline = _headline(rows, generations, run_name, summary)
    legend_note = "dashed: accepted filter update" + ("  ·  dotted red: stagnation restart" if restarts else "")
    written = []

    fitness = _fitness_series(rows, generations)
    if fitness:
        figure, axis = plt.subplots(figsize=(10, 4.8), facecolor=SURFACE)
        _style_axis(axis, "Fitness by generation", "fitness")
        _mark_events(axis, filter_updates, restarts)
        _lines(axis, generations, fitness)
        best_ever = fitness[0][1]
        peak = max((v for v in best_ever if v == v), default=None)
        if peak is not None:
            peak_gen = next(g for g, v in zip(generations, best_ever) if v == peak)
            axis.plot([peak_gen], [peak], marker="o", markersize=8, color=SERIES[0],
                      markeredgecolor=SURFACE, markeredgewidth=2, zorder=5)
            axis.annotate(f"{peak:.4f} @ gen {peak_gen:g}", (peak_gen, peak), xytext=(6, 6),
                          textcoords="offset points", fontsize=8, color=TEXT)
        figure.suptitle(headline, fontsize=9, color=TEXT_SECONDARY, x=0.01, ha="left")
        figure.text(0.99, 0.01, legend_note, fontsize=7, color=TEXT_SECONDARY, ha="right")
        figure.tight_layout(rect=(0, 0.03, 1, 0.95))
        path = output_dir / "fitness.png"
        figure.savefig(path, dpi=150, facecolor=SURFACE)
        plt.close(figure)
        written.append(path)

    panels = _panels(rows, generations)
    if panels:
        columns = 2
        grid_rows = (len(panels) + columns - 1) // columns
        figure, axes = plt.subplots(grid_rows, columns, figsize=(14, 3.1 * grid_rows),
                                    squeeze=False, facecolor=SURFACE)
        for axis, panel in zip(axes.flat, panels):
            _style_axis(axis, panel["title"], panel["ylabel"], panel["unit_range"])
            _mark_events(axis, filter_updates, restarts)
            _draw(axis, panel, generations)
        for axis in list(axes.flat)[len(panels):]:
            axis.axis("off")
        figure.suptitle(headline, fontsize=10, color=TEXT, x=0.01, ha="left")
        figure.text(0.99, 0.005, legend_note, fontsize=8, color=TEXT_SECONDARY, ha="right")
        figure.tight_layout(rect=(0, 0.015, 1, 0.975))
        path = output_dir / "dashboard.png"
        figure.savefig(path, dpi=130, facecolor=SURFACE)
        plt.close(figure)
        written.append(path)
    return written


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sources", nargs="+", help="Run directories or history CSV/JSONL files.")
    parser.add_argument("--output-dir", default=None,
                        help="Where to write the PNGs (default: <run-dir>/plots or <csv-stem>_plots). "
                             "Only valid with a single source.")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.output_dir and len(args.sources) > 1:
        raise SystemExit("--output-dir needs exactly one source")
    status = 0
    for source in args.sources:
        written = plot_run(source, args.output_dir)
        if written:
            print(f"{source}: wrote " + ", ".join(str(path) for path in written))
        else:
            print(f"{source}: nothing plotted (no history found or matplotlib missing)")
            status = 1
    return status


if __name__ == "__main__":
    raise SystemExit(main())
