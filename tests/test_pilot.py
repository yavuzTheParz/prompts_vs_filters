from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from experiments.run_pilot import (
    build_command,
    completed_summary,
    is_rerunnable_placeholder,
    paired_effects,
    parse_modes,
    parse_seeds,
)


class PilotRunnerTests(unittest.TestCase):
    def test_seed_repetitions_are_explicit(self):
        self.assertEqual(parse_seeds("101, 202"), [101, 202])
        with self.assertRaises(ValueError):
            parse_seeds("101,101")

    def test_modes_reject_unknown_values(self):
        self.assertEqual(parse_modes("fixed_filter,coevolution"), ["fixed_filter", "coevolution"])
        with self.assertRaises(ValueError):
            parse_modes("coevolution,unknown")

    def test_fixed_filter_command_disables_updates(self):
        args = SimpleNamespace(
            base_url="http://127.0.0.1:8000",
            model="dolphin",
            generations=120,
            mu=8,
            lambda_=16,
            k_evals=2,
            final_k_evals=8,
            filtered_temperature=0.7,
            filter_update_every=5,
            top_k_filter=5,
            benign_csv=Path("benign.csv"),
            benign_holdout_csv=Path("benign_holdout.csv"),
            benign_holdout_repeats=3,
            stagnation_generations=20,
            max_garbled_token_ratio=0.4,
        )
        command = build_command(args, "fixed_filter", 101, Path("run"))
        index = command.index("--filter-update-every")
        self.assertEqual(command[index + 1], "0")
        self.assertIn("--final-k-evals", command)

    def test_completed_summary_requires_exact_generation_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "summary.json").write_text(
                '{"generations_completed": 120}', encoding="utf-8"
            )
            self.assertIsNotNone(completed_summary(run_dir, 120))
            self.assertIsNone(completed_summary(run_dir, 121))

    def test_completed_summary_rejects_mismatched_run_signature(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "summary.json").write_text(
                '{"generations_completed": 120}', encoding="utf-8"
            )
            (run_dir / "pilot_run_config.json").write_text(
                '{"seed": 101}', encoding="utf-8"
            )
            self.assertIsNotNone(
                completed_summary(run_dir, 120, {"seed": 101})
            )
            self.assertIsNone(completed_summary(run_dir, 120, {"seed": 202}))

    def test_matching_signature_only_directory_can_be_retried(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "pilot_run_config.json").write_text(
                '{"seed": 101}', encoding="utf-8"
            )
            self.assertTrue(is_rerunnable_placeholder(run_dir, {"seed": 101}))
            (run_dir / "partial.json").write_text("{}", encoding="utf-8")
            self.assertFalse(is_rerunnable_placeholder(run_dir, {"seed": 101}))

    def test_paired_effects_compare_matching_seed_and_repeat(self):
        rows = []
        for seed, fixed, coevo in ((101, 0.5, 0.2), (202, 0.4, 0.1)):
            for mode, value in (("fixed_filter", fixed), ("coevolution", coevo)):
                rows.append(
                    {
                        "status": "complete",
                        "seed": seed,
                        "repeat": 1,
                        "mode": mode,
                        "best_fitness": value,
                        "best_attack_objective": value,
                        "confirmation_fitness": value,
                        "confirmation_attack_objective": value,
                    }
                )
        effects = paired_effects(rows)
        fitness = next(row for row in effects if row["metric"] == "confirmation_fitness")
        self.assertEqual(fitness["n_pairs"], 2)
        self.assertAlmostEqual(fitness["mean_difference"], -0.3)


if __name__ == "__main__":
    unittest.main()
