from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from evolutionary_strategy import ESConfig, evolutionary_strategy_run
from run_es import DEFAULT_FILTER_PROMPT, _resolve_cli_k_evals, _resolve_initial_filter_prompt, write_run_dir


class RunArtifactMRObjectiveTests(unittest.TestCase):
    def test_k_eval_defaults_keep_dry_runs_cheap(self):
        self.assertEqual(_resolve_cli_k_evals(None, dry_run=True), 1)
        self.assertEqual(_resolve_cli_k_evals(None, dry_run=False), 3)

    def test_initial_filter_prompt_can_be_overridden_inline_or_by_file(self):
        self.assertEqual(
            _resolve_initial_filter_prompt(
                SimpleNamespace(initial_filter_prompt=None, filter_prompt_file=None)
            ),
            DEFAULT_FILTER_PROMPT,
        )
        self.assertEqual(
            _resolve_initial_filter_prompt(
                SimpleNamespace(
                    initial_filter_prompt="Answer directly.",
                    filter_prompt_file=None,
                )
            ),
            "Answer directly.",
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "filter.txt"
            path.write_text("Weak starting filter.\n", encoding="utf-8")
            self.assertEqual(
                _resolve_initial_filter_prompt(
                    SimpleNamespace(
                        initial_filter_prompt=None,
                        filter_prompt_file=str(path),
                    )
                ),
                "Weak starting filter.",
            )

    def test_initial_filter_prompt_rejects_two_sources(self):
        with self.assertRaises(ValueError):
            _resolve_initial_filter_prompt(
                SimpleNamespace(
                    initial_filter_prompt="one",
                    filter_prompt_file="two.txt",
                )
            )

    def test_repeated_fixed_seed_dry_runs_are_identical(self):
        first = evolutionary_strategy_run(
            ESConfig(
                lambda_=3,
                mu=2,
                generations=2,
                lightweight=True,
                random_seed=29,
                verbose=False,
            ),
            client=None,
        )
        second = evolutionary_strategy_run(
            ESConfig(
                lambda_=3,
                mu=2,
                generations=2,
                lightweight=True,
                random_seed=29,
                verbose=False,
            ),
            client=None,
        )

        self.assertEqual(first.history, second.history)
        self.assertEqual(first.sample_records, second.sample_records)
        self.assertEqual(first.best.input_prompt, second.best.input_prompt)

    def test_config_records_canonical_mode_and_full_formula(self):
        config = ESConfig(
            lambda_=2,
            mu=2,
            generations=1,
            lightweight=True,
            random_seed=23,
            verbose=False,
            mr_objective="semantic_recovery",
        )
        result = evolutionary_strategy_run(config=config, client=None)
        args = SimpleNamespace(
            api_key="must-not-be-written",
            model="fake-model",
            mr_objective="semantic_recovery",
        )

        with tempfile.TemporaryDirectory() as tmp:
            write_run_dir(tmp, args, config, result)
            payload = json.loads((Path(tmp) / "config.json").read_text())
            samples = [
                json.loads(line)
                for line in (Path(tmp) / "samples.jsonl").read_text().splitlines()
            ]

        self.assertEqual(payload["mr_objective"]["mode"], "semantic_recovery")
        self.assertEqual(
            payload["mr_objective"]["formula"],
            "fitness = 0.7 * attack_compliance_score + 0.3 * gated(MR)",
        )
        self.assertEqual(
            payload["attack_evaluator"]["version"],
            "defensive-compliance-v5",
        )
        self.assertIn("threshold", payload["attack_evaluator"])
        self.assertIn("prompt_template", payload["attack_evaluator"])
        self.assertNotIn("api_key", payload["args"])
        self.assertTrue(samples)
        self.assertTrue(
            all("generation" in sample and "sample_index" in sample for sample in samples)
        )


if __name__ == "__main__":
    unittest.main()
