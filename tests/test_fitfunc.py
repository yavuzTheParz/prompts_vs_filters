from __future__ import annotations

import unittest
import warnings

from Prompt_class import Prompt
from fitfunc import _set_metrics_and_fitness
from mr_objective import normalize_mr_objective


class MRObjectiveFitnessTests(unittest.TestCase):
    def test_semantic_recovery_uses_mr_directly(self):
        prompt = Prompt(input_prompt="controlled fixture")

        _set_metrics_and_fitness(
            prompt,
            asv=0.6,
            mr=0.8,
            alpha=0.7,
            beta=0.3,
            mr_objective="semantic_recovery",
        )

        self.assertAlmostEqual(prompt.metrics["mr_component"], 0.8)
        self.assertAlmostEqual(prompt.fitness, 0.66)

    def test_behavioral_deviation_uses_one_minus_mr(self):
        prompt = Prompt(input_prompt="controlled fixture")

        _set_metrics_and_fitness(
            prompt,
            asv=0.6,
            mr=0.8,
            alpha=0.7,
            beta=0.3,
            mr_objective="behavioral_deviation",
        )

        self.assertAlmostEqual(prompt.metrics["mr_component"], 0.2)
        self.assertAlmostEqual(prompt.fitness, 0.48)

    def test_legacy_alias_is_normalized_with_deprecation_warning(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            mode = normalize_mr_objective("maximize")

        self.assertEqual(mode, "semantic_recovery")
        self.assertTrue(
            any(item.category is DeprecationWarning for item in caught),
            "Expected a DeprecationWarning for the legacy alias.",
        )


if __name__ == "__main__":
    unittest.main()
