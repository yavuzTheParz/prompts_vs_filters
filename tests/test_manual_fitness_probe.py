from __future__ import annotations

import unittest

from manual_fitness_probe import run_probe


class ManualFitnessProbeTests(unittest.TestCase):
    def test_manual_high_expected_response_scores_above_refusal_when_mr_is_minimized(self):
        high_expected, low_expected = run_probe(mr_objective="minimize")
        self.assertGreater(high_expected.fitness, low_expected.fitness)
        self.assertGreater(high_expected.metrics["asv"], low_expected.metrics["asv"])
        self.assertGreater(
            high_expected.metrics["behavioral_deviation"],
            low_expected.metrics["behavioral_deviation"],
        )

    def test_refusal_scores_above_manual_high_when_mr_is_maximized(self):
        high_expected, low_expected = run_probe(mr_objective="maximize")
        self.assertGreater(low_expected.fitness, high_expected.fitness)
        self.assertGreater(low_expected.metrics["mr"], high_expected.metrics["mr"])


if __name__ == "__main__":
    unittest.main()
