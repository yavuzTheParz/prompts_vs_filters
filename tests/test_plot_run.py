from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from analysis.plot_run import load_history, plot_run
from run_es import write_plots

try:
    import matplotlib  # noqa: F401

    HAVE_MATPLOTLIB = True
except Exception:
    HAVE_MATPLOTLIB = False


def _history(generations: int = 6):
    rows = []
    for generation in range(1, generations + 1):
        rows.append(
            {
                "generation": generation,
                "best_fitness": 0.1 * generation if generation != 4 else 0.0,
                "mean_parent_fitness": 0.05 * generation,
                "best_attack_objective": 0.5,
                "mean_attack_objective": 0.3,
                "mean_attack_success": 0.5,
                "success_rate": 0.2,
                "population_compliant_count": 2,
                "population_refusal_count": 1,
                "best_mr": 0.8,
                "mean_mr": 0.85,
                "population_diversity": 0.9,
                "sigma": 1.0,
                "rejected_near_duplicate": 1,
                "filter_length": 144 + 10 * (generation > 3),
                "filter_changed": 1.0 if generation == 3 else 0.0,
            }
        )
    return rows


def _write_csv(path: Path, rows) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class LoadHistoryTests(unittest.TestCase):
    def test_run_dir_prefers_final_csv_and_falls_back_to_stream(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "history.jsonl").write_text(
                "\n".join(json.dumps(row) for row in _history(2)), encoding="utf-8"
            )
            self.assertEqual(len(load_history(root)), 2)
            _write_csv(root / "generation_summary.csv", _history(5))
            self.assertEqual(len(load_history(root)), 5)

    def test_missing_history_plots_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(plot_run(tmp), [])


@unittest.skipUnless(HAVE_MATPLOTLIB, "matplotlib not installed")
class PlotRunTests(unittest.TestCase):
    def test_run_dir_writes_fitness_and_dashboard(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_csv(root / "generation_summary.csv", _history())
            written = plot_run(root)
            self.assertEqual([path.name for path in written], ["fitness.png", "dashboard.png"])
            for path in written:
                self.assertEqual(path.parent, root / "plots")
                self.assertGreater(path.stat().st_size, 0)

    def test_history_csv_defaults_to_sibling_plot_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "es_history.csv"
            _write_csv(csv_path, _history())
            written = plot_run(csv_path)
            self.assertTrue(written)
            self.assertEqual(written[0].parent, Path(tmp) / "es_history_plots")

    def test_legacy_history_without_new_columns_still_plots(self):
        rows = [{"generation": g, "best_fitness": 0.1 * g} for g in range(1, 4)]
        with tempfile.TemporaryDirectory() as tmp:
            _write_csv(Path(tmp) / "generation_summary.csv", rows)
            self.assertEqual(len(plot_run(tmp)), 2)

    def test_write_plots_uses_run_dir_and_respects_no_plots(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_csv(root / "generation_summary.csv", _history())
            args = SimpleNamespace(run_dir=str(root), history_csv=None, plots=False)
            self.assertEqual(write_plots(args), [])
            self.assertFalse((root / "plots").exists())
            args.plots = True
            self.assertEqual(len(write_plots(args)), 2)


if __name__ == "__main__":
    unittest.main()
