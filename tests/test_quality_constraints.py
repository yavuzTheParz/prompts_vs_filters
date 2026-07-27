from __future__ import annotations

import unittest

from Prompt_class import Prompt
from quality_constraints import (
    apply_quality_constraints,
    mark_near_duplicates,
    population_diversity,
)
from selection import sort_population


def candidate(text: str, attack: float = 0.5, repetition: float = 0.0) -> Prompt:
    return Prompt(
        input_prompt=text,
        direct_output="direct",
        output_prompts=["filtered"],
        fitness=attack,
        metrics={
            "attack_objective": attack,
            "mr": 0.5,
            "repetition_penalty": repetition,
            "length_penalty": 0.0,
        },
    )


class QualityConstraintTests(unittest.TestCase):
    def test_repetitive_prompt_ranks_below_valid_equivalent(self):
        repetitive = candidate("repeat repeat repeat repeat", repetition=0.8)
        valid = candidate("explain a varied controlled topic", repetition=0.0)
        apply_quality_constraints([repetitive, valid])

        ranked = sort_population([repetitive, valid], mode="scalar")

        self.assertIs(ranked[0], valid)
        self.assertEqual(repetitive.metrics["valid"], 0.0)

    def test_exact_duplicates_cannot_remain_valid_together(self):
        first = candidate("same controlled prompt")
        second = candidate("same controlled prompt")
        apply_quality_constraints([first, second])
        reasons = mark_near_duplicates([first, second], threshold=0.0)

        self.assertEqual(reasons["near_duplicate"], 1)
        self.assertEqual(sum(p.metrics["valid"] for p in (first, second)), 1.0)

    def test_population_diversity_mixed_and_identical(self):
        mixed = [candidate("alpha beta"), candidate("gamma delta")]
        identical = [candidate("alpha beta"), candidate("alpha beta")]

        self.assertGreater(population_diversity(mixed), 0.0)
        self.assertEqual(population_diversity(identical), 0.0)

    def test_api_error_is_invalid_and_zero_fitness(self):
        prompt = candidate("controlled prompt", attack=0.9)
        prompt.metadata["api_error"] = "controlled failure"
        reasons = apply_quality_constraints([prompt])

        self.assertEqual(reasons["api_error"], 1)
        self.assertEqual(prompt.metrics["valid"], 0.0)
        self.assertEqual(prompt.fitness, 0.0)


if __name__ == "__main__":
    unittest.main()
