from __future__ import annotations

import unittest

from Prompt_class import Prompt
from selection import sort_population


class MRObjectiveSelectionTests(unittest.TestCase):
    def setUp(self):
        self.lower_mr = Prompt(
            input_prompt="lower MR",
            metrics={"asv": 0.6, "mr": 0.2},
        )
        self.higher_mr = Prompt(
            input_prompt="higher MR",
            metrics={"asv": 0.6, "mr": 0.8},
        )

    def test_lexicographic_tie_break_reverses_between_modes(self):
        deviation_order = sort_population(
            [self.higher_mr, self.lower_mr],
            mode="lexicographic",
            mr_objective="behavioral_deviation",
        )
        recovery_order = sort_population(
            [self.lower_mr, self.higher_mr],
            mode="lexicographic",
            mr_objective="semantic_recovery",
        )

        self.assertIs(deviation_order[0], self.lower_mr)
        self.assertIs(recovery_order[0], self.higher_mr)

    def test_rank_partitioning_uses_the_selected_mode(self):
        deviation_order = sort_population(
            [self.higher_mr, self.lower_mr],
            mode="rank_partitioning",
            mr_objective="behavioral_deviation",
        )
        recovery_order = sort_population(
            [self.lower_mr, self.higher_mr],
            mode="rank_partitioning",
            mr_objective="semantic_recovery",
        )

        self.assertIs(deviation_order[0], self.lower_mr)
        self.assertIs(recovery_order[0], self.higher_mr)

    def test_compliance_ranks_above_high_similarity_ambiguous(self):
        ambiguous = Prompt(
            input_prompt="ambiguous",
            fitness=0.9,
            metrics={
                "valid": 1.0,
                "attack_compliance_score": 0.25,
                "attack_objective": 0.25,
                "unsafe_reference_similarity": 1.0,
                "mr": 0.1,
            },
        )
        compliant = Prompt(
            input_prompt="compliant",
            fitness=0.7,
            metrics={
                "valid": 1.0,
                "attack_compliance_score": 1.0,
                "attack_objective": 1.0,
                "unsafe_reference_similarity": 0.0,
                "mr": 0.9,
            },
        )

        for mode in ("scalar", "lexicographic", "rank_partitioning"):
            ranked = sort_population([ambiguous, compliant], mode=mode)
            self.assertIs(ranked[0], compliant, mode)

    def test_zero_attack_score_does_not_rank_by_mr(self):
        first = Prompt(
            input_prompt="first",
            metrics={
                "valid": 1.0,
                "attack_compliance_score": 0.0,
                "attack_objective": 0.0,
                "unsafe_reference_similarity": 1.0,
                "mr": 0.1,
            },
        )
        second = Prompt(
            input_prompt="second",
            metrics={
                "valid": 1.0,
                "attack_compliance_score": 0.0,
                "attack_objective": 0.0,
                "unsafe_reference_similarity": 0.0,
                "mr": 0.9,
            },
        )

        for mode in ("scalar", "lexicographic", "rank_partitioning"):
            ranked = sort_population([first, second], mode=mode)
            self.assertEqual([p.input_prompt for p in ranked], ["first", "second"], mode)


if __name__ == "__main__":
    unittest.main()
