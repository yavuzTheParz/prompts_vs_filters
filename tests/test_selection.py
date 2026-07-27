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


if __name__ == "__main__":
    unittest.main()
