from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluators import DefensiveComplianceEvaluator
from tests.manual_llm_asv_probe import (
    LOW_ASV_MAX,
    SAFE_RESPONSES,
    SAMPLE_CATEGORIES,
    assert_expected_scores,
    load_sample_prompts,
    print_results,
    run_live_probe,
)


class LLMASVProbeFixtureTests(unittest.TestCase):
    def test_expected_categories_are_loaded_from_sample_csv(self):
        samples = load_sample_prompts()

        self.assertEqual([category for category, _ in samples], list(SAMPLE_CATEGORIES))
        self.assertTrue(all(prompt.strip() for _, prompt in samples))

    def test_all_safe_controls_have_low_asv(self):
        evaluator = DefensiveComplianceEvaluator()

        for category, response in SAFE_RESPONSES.items():
            with self.subTest(category=category):
                result = evaluator.evaluate(response)
                self.assertLessEqual(result.score, LOW_ASV_MAX)
                self.assertIn(result.label, {"refusal", "benign_educational"})

    @patch("tests.manual_llm_asv_probe.LocalLLMClient")
    def test_empty_generation_retries_next_prompt_in_same_category(self, client_cls):
        client_cls.return_value.generate.side_effect = [
            "[SANITIZED_COMPLIANCE] hacking fixture",
            "[SANITIZED_COMPLIANCE] explosive fixture",
            {"generated_text": "", "generated_tokens": 1},
            "[SANITIZED_COMPLIANCE] malware fixture",
        ]

        results = run_live_probe("http://local.test")

        malware = next(row for row in results if row.category == "malware")
        self.assertEqual(malware.generation_attempts, 2)
        self.assertEqual(malware.llm_asv, 1.0)
        self.assertEqual(client_cls.return_value.generate.call_count, 4)


@unittest.skipUnless(
    os.getenv("RUN_LIVE_LLM_ASV_TESTS") == "1",
    "Set RUN_LIVE_LLM_ASV_TESTS=1 to call the configured local LLM.",
)
class LiveLLMASVTests(unittest.TestCase):
    def test_safe_controls_are_low_and_local_llm_outputs_are_high(self):
        base_url = os.getenv("LOCAL_LLM_BASE_URL", "http://127.0.0.1:8000")
        results = run_live_probe(
            base_url=base_url,
            api_key=os.getenv("LOCAL_LLM_API_KEY"),
            timeout_sec=int(os.getenv("LOCAL_LLM_TIMEOUT_SEC", "300")),
        )
        print_results(results)
        assert_expected_scores(results)


if __name__ == "__main__":
    unittest.main()
