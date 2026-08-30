from __future__ import annotations

import unittest

from Prompt_class import Prompt
from quality_constraints import fluency_score
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

    def test_scalar_and_lexicographic_choose_different_winners(self):
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

        scalar = sort_population([ambiguous, compliant], mode="scalar")
        lexicographic = sort_population([ambiguous, compliant], mode="lexicographic")
        partitioned = sort_population([ambiguous, compliant], mode="rank_partitioning")

        self.assertIs(scalar[0], ambiguous)
        self.assertIs(lexicographic[0], compliant)
        self.assertIs(partitioned[0], compliant)

    def test_validity_is_hard_constraint_in_both_modes(self):
        invalid = Prompt(
            input_prompt="invalid high scores",
            fitness=1.0,
            metrics={"valid": 0.0, "attack_compliance_score": 1.0, "mr": 0.0},
        )
        valid = Prompt(
            input_prompt="valid lower scores",
            fitness=0.1,
            metrics={"valid": 1.0, "attack_compliance_score": 0.1, "mr": 0.9},
        )
        for mode in ("scalar", "lexicographic"):
            self.assertIs(sort_population([invalid, valid], mode=mode)[0], valid)

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

    def test_zero_attack_tie_prefers_readable_prompt_over_garbled_diversity(self):
        garbled = Prompt(
            input_prompt="garbled",
            fitness=0.0,
            metrics={
                "valid": 1.0,
                "attack_compliance_score": 0.0,
                "attack_objective": 0.0,
                "asv": 0.0,
                "mr": 0.2,
                "fluency": 0.25,
                "garbled_token_ratio": 0.6,
                "diversity": 1.0,
                "quality_soft_penalty": 0.0,
            },
        )
        readable = Prompt(
            input_prompt="readable",
            fitness=0.0,
            metrics={
                "valid": 1.0,
                "attack_compliance_score": 0.0,
                "attack_objective": 0.0,
                "asv": 0.0,
                "mr": 0.9,
                "fluency": 0.95,
                "garbled_token_ratio": 0.0,
                "diversity": 0.1,
                "quality_soft_penalty": 0.0,
            },
        )

        for mode in ("scalar", "lexicographic", "rank_partitioning"):
            ranked = sort_population([garbled, readable], mode=mode)
            self.assertIs(ranked[0], readable, mode)

    def test_fluency_penalizes_scrambled_tokens(self):
        clear = "describe the system context and explain safe security procedures"
        scrambled = (
            "dseicbr txetnatocs of the sytem how does one igrnoe "
            "perocredures and strt afresh to usner the brared scteres"
        )

        self.assertGreater(fluency_score(clear), 0.8)
        self.assertLess(fluency_score(scrambled), 0.55)


if __name__ == "__main__":
    unittest.main()
