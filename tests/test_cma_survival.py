from __future__ import annotations

import unittest

from Prompt_class import Prompt
from evolutionary_strategy import (
    ESConfig,
    _select_cma_survivors,
    _update_cma_distribution,
)


def individual(name: str, fitness: float, vector, sigma: float) -> Prompt:
    return Prompt(
        input_prompt=name,
        fitness=fitness,
        metrics={"attack_objective": fitness, "mr": 0.5, "valid": 1.0},
        metadata={"cma_vector": list(vector), "cma_sigma": sigma},
    )


class CMASurvivalTests(unittest.TestCase):
    def setUp(self):
        self.config = ESConfig(mu=1, lambda_=2, variant="cma_es")
        self.parent = individual("superior parent", 0.9, [0.0, 0.0], 1.0)
        self.offspring = [
            individual("child one", 0.4, [1.0, 0.2], 1.2),
            individual("child two", 0.3, [-1.0, -0.2], 0.8),
        ]

    def test_superior_parent_survives_plus_selection(self):
        selected, vectors, sigmas = _select_cma_survivors(
            [self.parent], self.offspring, 1, self.config, "plus"
        )

        self.assertIs(selected[0], self.parent)
        self.assertEqual(vectors, [[0.0, 0.0]])
        self.assertEqual(sigmas, [1.0])

    def test_superior_parent_cannot_survive_comma_selection(self):
        selected, vectors, sigmas = _select_cma_survivors(
            [self.parent], self.offspring, 1, self.config, "comma"
        )

        self.assertIs(selected[0], self.offspring[0])
        self.assertEqual(vectors, [[1.0, 0.2]])
        self.assertEqual(sigmas, [1.2])

    def test_distribution_update_uses_selected_control_vectors(self):
        mean, covariance = _update_cma_distribution(
            [[1.0, 0.5], [-1.0, -0.5]],
            [[1.0, 0.0], [0.0, 1.0]],
            1e-6,
        )

        self.assertEqual(mean, [0.0, 0.0])
        self.assertAlmostEqual(covariance[0][0], 2.0)
        self.assertAlmostEqual(covariance[0][1], 1.0)
        self.assertAlmostEqual(covariance[1][1], 0.5)


if __name__ == "__main__":
    unittest.main()
