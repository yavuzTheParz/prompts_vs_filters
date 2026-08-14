from __future__ import annotations

import json
import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from evolutionary_strategy import ESConfig, evolutionary_strategy_run
from run_es import _sanitize_payload, write_run_dir


class ProvenanceTests(unittest.TestCase):
    def _result(self):
        return evolutionary_strategy_run(
            ESConfig(
                lambda_=3,
                mu=2,
                generations=2,
                lightweight=True,
                random_seed=41,
                verbose=False,
            ),
            client=None,
        )

    def test_generation_summary_has_required_aggregate_fields(self):
        row = self._result().history[-1]
        required = {
            "mean_attack_objective",
            "median_attack_objective",
            "std_attack_objective",
            "operator_attempts",
            "operator_acceptances",
            "operator_success_rate",
            "operator_fallbacks",
            "population_diversity",
        }
        self.assertTrue(required.issubset(row))

    def test_generation_history_keeps_best_prompt_and_outputs(self):
        result = self._result()

        for row in result.history:
            self.assertTrue(row["best_prompt"])
            self.assertEqual(
                json.loads(row["best_outputs_json"]),
                [row["best_primary_output"]],
            )
            self.assertEqual(row["best_output_count"], 1.0)
            self.assertTrue(row["best_prompt_id"])

    def test_generation_summary_csv_serializes_best_prompt_outputs(self):
        result = self._result()
        args = SimpleNamespace(
            api_key=None,
            model="fake-model",
            base_url=None,
            mr_objective="behavioral_deviation",
        )

        with tempfile.TemporaryDirectory() as tmp:
            write_run_dir(tmp, args, ESConfig(lightweight=True), result)
            with (Path(tmp) / "generation_summary.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(len(rows), len(result.history))
        self.assertIn("best_prompt", rows[0])
        self.assertIn("best_primary_output", rows[0])
        self.assertEqual(
            json.loads(rows[0]["best_outputs_json"]),
            [rows[0]["best_primary_output"]],
        )

    def test_final_best_lineage_reaches_a_seed(self):
        result = self._result()
        by_id = {row["prompt_id"]: row for row in result.lineage_records}
        current = result.best.metadata["prompt_id"]
        visited = set()
        while current is not None:
            self.assertNotIn(current, visited)
            visited.add(current)
            current = by_id[current]["parent_id"]
        self.assertGreaterEqual(len(visited), 1)

    def test_manifest_and_summary_are_reproducible_and_secret_free(self):
        result = self._result()
        args = SimpleNamespace(
            api_key="sk-controlled-secret-value-123456",
            model="fake-model",
            base_url="http://127.0.0.1:8000",
            mr_objective="behavioral_deviation",
        )
        config = ESConfig(
            lambda_=3,
            mu=2,
            generations=2,
            lightweight=True,
            random_seed=41,
            verbose=False,
        )
        with tempfile.TemporaryDirectory() as tmp:
            write_run_dir(tmp, args, config, result)
            root = Path(tmp)
            manifest = json.loads((root / "manifest.json").read_text())
            summary = json.loads((root / "summary.json").read_text())
            lineage = (root / "lineage.jsonl").read_text()
            config_text = (root / "config.json").read_text()

        self.assertEqual(manifest["seed"], 41)
        self.assertIn("commit_sha", manifest)
        self.assertIn("dependencies", manifest)
        self.assertEqual(summary["generations_completed"], 2)
        self.assertTrue(lineage.strip())
        self.assertNotIn("sk-controlled-secret", config_text)

    def test_sanitizer_redacts_nested_credentials(self):
        payload = _sanitize_payload(
            {
                "nested": {"api_key": "secret-value"},
                "auth_token": "secret-token",
                "custom_password": "secret-password",
                "note": "gho_1234567890abcdefghijklmnop",
            }
        )
        self.assertEqual(payload["nested"]["api_key"], "[REDACTED]")
        self.assertEqual(payload["auth_token"], "[REDACTED]")
        self.assertEqual(payload["custom_password"], "[REDACTED]")
        self.assertEqual(payload["note"], "[REDACTED]")

    def test_sanitizer_preserves_token_named_metrics(self):
        payload = _sanitize_payload(
            {
                "garbled_token_ratio": 0.42,
                "max_seed_token_growth_ratio": 1.8,
                "disable_token_mutation": False,
            }
        )
        self.assertEqual(payload["garbled_token_ratio"], 0.42)
        self.assertEqual(payload["max_seed_token_growth_ratio"], 1.8)
        self.assertFalse(payload["disable_token_mutation"])


if __name__ == "__main__":
    unittest.main()
