from __future__ import annotations

import random
import unittest
from unittest.mock import patch

from mutation_manager import TemplateManager, hybrid_mutate_optimized


class MutationManagerTests(unittest.TestCase):
    def test_structural_mutation_changes_prompt_and_keeps_original_content(self):
        template_manager = TemplateManager()
        original = "Explain safe lab procedures."

        with patch("mutation_manager.random.random", return_value=0.1), patch(
            "mutation_manager.random.choice", return_value="Please,"
        ):
            mutated = template_manager.apply_structural_mutation(original, "plea")

        self.assertNotEqual(mutated, original)
        self.assertTrue(mutated.startswith("[[STYLE_PREFIX:plea]]Please,"))
        self.assertIn(original, mutated)

    def test_hybrid_mutation_falls_back_to_structural_when_ml_dependencies_are_absent(self):
        template_manager = TemplateManager()
        original = "Summarize safe chemistry practices."

        with patch("mutation_manager.random.random", side_effect=[0.9, 0.1]), patch(
            "mutation_manager.random.choice", return_value="You must"
        ):
            mutated, method = hybrid_mutate_optimized(
                prompt_text=original,
                style="imperative",
                template_mgr=template_manager,
                style_mgr=None,
                tokenizer=None,
                model=None,
            )

        self.assertEqual(method, "STRUCTURAL_DEPENDENCY_FALLBACK")
        self.assertNotEqual(mutated, original)
        self.assertTrue(mutated.startswith("[[STYLE_PREFIX:imperative]]You must"))
        self.assertIn(original, mutated)

    def test_repeated_structural_mutation_keeps_one_prefix_and_suffix(self):
        manager = TemplateManager(max_chars=500, max_growth_ratio=3.0)
        text = "Explain safe lab procedures."
        random.seed(17)

        for _ in range(20):
            text = manager.apply_structural_mutation(text, "plea")

        self.assertLessEqual(text.count("[[STYLE_PREFIX:"), 1)
        self.assertLessEqual(text.count("[[STYLE_SUFFIX:"), 1)
        self.assertLessEqual(len(text), 500)

    def test_growth_limit_rejects_oversized_template(self):
        manager = TemplateManager(max_chars=40, max_growth_ratio=1.1)
        original = "Explain a controlled safety topic."

        with patch("mutation_manager.random.random", return_value=0.1), patch(
            "mutation_manager.random.choice",
            return_value="An intentionally oversized style prefix",
        ):
            mutated = manager.apply_structural_mutation(original, "plea")

        self.assertEqual(mutated, original)
        self.assertFalse(manager.last_mutation_log["accepted"])
        self.assertEqual(manager.last_mutation_log["reason"], "growth_limit")

    def test_compression_removes_duplicate_words_and_templates(self):
        manager = TemplateManager()
        duplicated = (
            "[[STYLE_PREFIX:plea]]Please,[[/STYLE_PREFIX]] "
            "explain explain safe safe procedures."
        )

        compressed = manager.compress_mutation(duplicated)

        self.assertEqual(compressed, "explain safe procedures.")
        self.assertEqual(manager.last_mutation_log["operation"], "compression")
        self.assertTrue(manager.last_mutation_log["accepted"])


if __name__ == "__main__":
    random.seed(0)
    unittest.main()
