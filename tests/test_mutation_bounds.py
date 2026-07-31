from __future__ import annotations

import math
import unittest
from unittest.mock import patch

from Prompt_class import Prompt
from evolutionary_strategy import (
    ESConfig,
    _mutate_prompt,
    _mutation_repetitions,
    evolutionary_strategy_run,
)


def controlled_population() -> list[Prompt]:
    return [
        Prompt(input_prompt="Explain controlled access procedures clearly."),
        Prompt(input_prompt="Summarize benign laboratory safety practices."),
    ]


class MutationBoundTests(unittest.TestCase):
    def test_sigma_max_respects_max_mutations_per_child(self):
        config = ESConfig(
            lightweight=True,
            sigma=6.0,
            sigma_min=0.25,
            sigma_max=6.0,
            max_mutations_per_child=2,
        )
        parent = controlled_population()[0]

        child, logs = _mutate_prompt(
            parent,
            sigma=config.sigma_max,
            config=config,
            forced_style="imperative",
        )

        self.assertEqual(
            _mutation_repetitions(6.0, 0.25, 6.0, 2),
            2,
        )
        self.assertLessEqual(len(logs), 2)
        self.assertEqual(child.metadata["last_mutation_attempts"], len(logs))

    def test_noop_mutation_does_not_count_as_progress(self):
        config = ESConfig(
            lambda_=2,
            mu=2,
            generations=2,
            lightweight=True,
            random_seed=301,
            structural_mutation_enabled=False,
            token_mutation_enabled=True,
            max_mutations_per_child=1,
            verbose=False,
        )

        with patch(
            "evolutionary_strategy._lightweight_token_mutate_text",
            side_effect=lambda text: (text, "CONTROLLED_NOOP"),
        ):
            result = evolutionary_strategy_run(
                config,
                client=None,
                initial_population=controlled_population(),
            )

        self.assertTrue(
            all(row["operator_acceptances"] == 0.0 for row in result.history)
        )
        self.assertTrue(
            all(row["operator_noops"] == 2.0 for row in result.history)
        )
        self.assertTrue(
            all(row["success_rate"] == 0.0 for row in result.history)
        )
        self.assertTrue(
            all(
                "NOOP_REJECT" in log
                for row in result.lineage_records
                if isinstance(row["mutation_operator"], list)
                for log in row["mutation_operator"]
            )
        )

    def test_320_generation_run_respects_seed_relative_limits(self):
        char_ratio = 1.25
        token_ratio = 1.25
        result = evolutionary_strategy_run(
            ESConfig(
                lambda_=2,
                mu=2,
                generations=320,
                lightweight=True,
                random_seed=302,
                survival_schema="(mu,lambda)",
                structural_mutation_enabled=False,
                token_mutation_enabled=True,
                max_mutations_per_child=2,
                max_seed_body_growth_ratio=char_ratio,
                max_seed_token_growth_ratio=token_ratio,
                verbose=False,
            ),
            client=None,
            initial_population=controlled_population(),
        )

        self.assertEqual(len(result.history), 320)
        self.assertTrue(
            all(
                len(row["mutation_operator"]) <= 2
                for row in result.lineage_records
                if isinstance(row["mutation_operator"], list)
            )
        )
        for prompt in result.population:
            body = prompt.internal_prompt.body
            seed_chars = int(prompt.metadata["seed_body_chars"])
            seed_tokens = int(prompt.metadata["seed_body_tokens"])
            self.assertLessEqual(
                len(body),
                math.ceil(seed_chars * char_ratio),
            )
            self.assertLessEqual(
                len(body.split()),
                math.ceil(seed_tokens * token_ratio),
            )
            self.assertLessEqual(
                prompt.metadata["mutation_count_since_seed"],
                prompt.metadata["mutation_attempts_since_seed"],
            )

    def test_stagnation_restart_is_deterministic_for_fixed_seed(self):
        def run_once():
            config = ESConfig(
                lambda_=2,
                mu=2,
                generations=8,
                lightweight=True,
                random_seed=303,
                structural_mutation_enabled=False,
                token_mutation_enabled=True,
                max_mutations_per_child=1,
                stagnation_generations=2,
                restart_on_stagnation=True,
                verbose=False,
            )
            with patch(
                "evolutionary_strategy._lightweight_token_mutate_text",
                side_effect=lambda text: (text, "CONTROLLED_NOOP"),
            ):
                return evolutionary_strategy_run(
                    config,
                    client=None,
                    initial_population=controlled_population(),
                )

        first = run_once()
        second = run_once()
        first_restarts = [
            int(row["restart_triggered"]) for row in first.history
        ]
        second_restarts = [
            int(row["restart_triggered"]) for row in second.history
        ]

        self.assertEqual(first_restarts, second_restarts)
        self.assertEqual(first_restarts, [0, 1, 0, 1, 0, 1, 0, 1])
        self.assertEqual(first.history[-1]["restart_count"], 4.0)


if __name__ == "__main__":
    unittest.main()
