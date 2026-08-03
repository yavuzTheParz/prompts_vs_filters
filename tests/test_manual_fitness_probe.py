from __future__ import annotations

import unittest

from tests.manual_fitness_probe import run_probe


class ManualFitnessProbeTests(unittest.TestCase):
    def test_manual_high_expected_response_scores_above_refusal_when_mr_is_minimized(self):
        high_expected, low_expected = run_probe(mr_objective="behavioral_deviation")
        self.assertGreater(high_expected.fitness, low_expected.fitness)
        self.assertGreater(high_expected.metrics["asv"], low_expected.metrics["asv"])
        self.assertGreater(
            high_expected.metrics["behavioral_deviation"],
            low_expected.metrics["behavioral_deviation"],
        )

    def test_semantic_recovery_rewards_refusal_similarity_without_overriding_compliance(self):
        high_expected, low_expected = run_probe(mr_objective="semantic_recovery")
        self.assertGreater(low_expected.metrics["mr"], high_expected.metrics["mr"])
        self.assertGreater(
            low_expected.metrics["mr_component"],
            high_expected.metrics["mr_component"],
        )
        self.assertGreater(
            high_expected.metrics["attack_compliance_score"],
            low_expected.metrics["attack_compliance_score"],
        )


if __name__ == "__main__":
    unittest.main()
