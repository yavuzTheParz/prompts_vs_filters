from __future__ import annotations

import unittest
from pathlib import Path

from analysis.analyze_ablation import complete_rows, mean_ci, summarize
from experiments.run_ablation import (
    DEFAULT_MATRIX,
    load_matrix,
    mutation_flags,
    parse_seeds,
    pool_hash,
    run_condition,
)


ROOT = Path(__file__).resolve().parents[1]


class AblationTests(unittest.TestCase):
    def test_matrix_covers_required_conditions_and_comparisons(self):
        conditions = load_matrix(DEFAULT_MATRIX)["conditions"]
        self.assertEqual({row["id"] for row in conditions[:6]}, set("ABCDEF"))
        self.assertEqual(
            {row["mutation"] for row in conditions},
            {"structural+token", "token_only", "structural_only"},
        )
        self.assertEqual({row["selection"] for row in conditions}, {"scalar", "lexicographic"})
        self.assertEqual(
            {row["filter_mode"] for row in conditions},
            {"fixed_filter", "coevolution"},
        )

    def test_seed_and_initial_pool_controls(self):
        self.assertEqual(parse_seeds("101, 102"), [101, 102])
        with self.assertRaises(ValueError):
            parse_seeds("101,101")
        digest = pool_hash(ROOT / "prompts" / "initial_population.csv")
        self.assertEqual(len(digest), 64)

    def test_mutation_modes_are_explicit(self):
        self.assertEqual(mutation_flags("structural+token"), (True, True))
        self.assertEqual(mutation_flags("token_only"), (False, True))
        self.assertEqual(mutation_flags("structural_only"), (True, False))

    def test_incomplete_runs_are_excluded_by_rule(self):
        rows = [
            {"status": "complete", "generations": "10", "condition": "A"},
            {"status": "failed", "generations": "0", "condition": "B"},
            {"status": "complete", "generations": "9", "condition": "C"},
        ]
        self.assertEqual(complete_rows(rows, 10), [rows[0]])

    def test_statistics_include_ci_and_effect_size(self):
        rows = []
        for condition, values in (("A", [1.0, 2.0]), ("B", [2.0, 3.0])):
            for value in values:
                row = {
                    "condition": condition,
                    "filter_mode": "fixed_filter",
                    **{metric: str(value) for metric in (
                        "best_fitness", "best_attack_objective", "best_compliance",
                        "best_similarity", "best_mr", "best_diversity",
                        "best_prompt_length",
                    )},
                }
                rows.append(row)
        stats = summarize(rows)
        self.assertIn("ci95_low", stats[0])
        self.assertTrue(any(float(row["cohens_d_vs_A"]) != 0 for row in stats if row["condition"] == "B"))
        self.assertEqual(mean_ci([2.0]), (2.0, 2.0, 2.0))

    def test_one_condition_smoke_records_control_metadata(self):
        condition = load_matrix(DEFAULT_MATRIX)["conditions"][0]
        csv_path = ROOT / "prompts" / "initial_population.csv"
        digest = pool_hash(csv_path)
        row, history = run_condition(
            condition,
            101,
            generations=1,
            mu=2,
            lambda_=2,
            csv_path=csv_path,
            model_name="test-dry-run",
            initial_pool_sha256=digest,
        )
        self.assertEqual(row["status"], "complete")
        self.assertEqual(row["initial_pool_sha256"], digest)
        self.assertEqual(len(history), 1)


if __name__ == "__main__":
    unittest.main()
