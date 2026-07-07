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
        self.assertTrue(mutated.startswith("Please,"))
        self.assertIn("explain safe lab procedures.", mutated)

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
        self.assertTrue(mutated.startswith("You must"))
        self.assertIn("summarize safe chemistry practices.", mutated)


if __name__ == "__main__":
    random.seed(0)
    unittest.main()
