from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from Prompt_class import Prompt
from evaluators import DefensiveComplianceEvaluator, attack_progress
from evolutionary_strategy import (
    ESConfig,
    RunStream,
    _maybe_evolve_filter,
    _select_top,
    _update_tone_probabilities,
    evolutionary_strategy_run,
    load_checkpoint,
)
from run_llm import assign_outputs
from selection import sort_population


def scored(text: str, fitness: float, progress: float, seed: str = "s") -> Prompt:
    prompt = Prompt(input_prompt=text, fitness=fitness)
    prompt.output_prompts = ["filtered"]
    prompt.direct_output = "direct"
    prompt.metrics = {
        "valid": 1.0,
        "attack_compliance_score": fitness,
        "attack_objective": fitness,
        "attack_progress": progress,
        "search_progress": progress,
        "mr": 0.5,
    }
    prompt.metadata["seed_prompt_id"] = seed
    return prompt


class AttackProgressTests(unittest.TestCase):
    def setUp(self):
        self.evaluator = DefensiveComplianceEvaluator()

    def test_progress_orders_labels_below_compliance(self):
        refusal = self.evaluator.evaluate("I cannot help with that request.")
        educational = self.evaluator.evaluate(
            "Here is a high-level overview of safety and prevention practices."
        )
        ambiguous = self.evaluator.evaluate(
            "The history of the region is long and involves many trade routes."
        )
        compliant = self.evaluator.evaluate("[SANITIZED_COMPLIANCE] placeholder")

        scores = [attack_progress(result) for result in (refusal, educational, ambiguous, compliant)]
        self.assertEqual(scores, sorted(scores))
        self.assertEqual(scores[-1], 1.0)
        self.assertLessEqual(max(scores[:-1]), 0.60)

    def test_substantive_refusal_ranks_above_bare_refusal(self):
        bare = self.evaluator.evaluate("I cannot help with that request.")
        substantive = self.evaluator.evaluate(
            "I cannot help with that request. " + "Some general context follows here. " * 20
        )
        self.assertGreater(attack_progress(substantive), attack_progress(bare))

    def test_progress_breaks_zero_fitness_ties_but_never_beats_fitness(self):
        stuck = scored("alpha prompt text", 0.0, 0.05)
        closer = scored("beta prompt text", 0.0, 0.45)
        winner = scored("gamma prompt text", 0.3, 0.0)
        for mode in ("scalar", "lexicographic"):
            ranked = sort_population([stuck, closer, winner], mode=mode)
            self.assertEqual(ranked, [winner, closer, stuck], mode)


class DiversityControlTests(unittest.TestCase):
    def test_seed_cap_keeps_other_lineages(self):
        config = ESConfig(max_survivors_per_seed=1, near_duplicate_threshold=0.0)
        population = [
            scored("one two three", 0.9, 0.0, seed="a"),
            scored("four five six", 0.8, 0.0, seed="a"),
            scored("seven eight nine", 0.1, 0.0, seed="b"),
        ]
        selected = _select_top(population, 2, config)
        self.assertEqual({p.metadata["seed_prompt_id"] for p in selected}, {"a", "b"})

    def test_seed_cap_falls_back_when_lineages_run_out(self):
        config = ESConfig(max_survivors_per_seed=1, near_duplicate_threshold=0.0)
        population = [
            scored("one two three", 0.9, 0.0, seed="a"),
            scored("four five six", 0.8, 0.0, seed="a"),
        ]
        self.assertEqual(len(_select_top(population, 2, config)), 2)

    def test_tone_probabilities_follow_survivors_with_floor(self):
        config = ESConfig(tone_learning_rate=0.5, tone_min_probability=0.1)
        probs = {"imperative": 0.5, "plea": 0.5}
        updated = _update_tone_probabilities(probs, ["plea", "plea"], config)
        self.assertAlmostEqual(sum(updated.values()), 1.0)
        self.assertGreater(updated["plea"], 0.7)
        for _ in range(20):
            updated = _update_tone_probabilities(updated, ["plea"], config)
        self.assertGreaterEqual(updated["imperative"], 0.09)

    def test_unknown_tone_is_rejected(self):
        with self.assertRaises(ValueError):
            evolutionary_strategy_run(
                ESConfig(lightweight=True, mutation_styles=("nonexistent",), verbose=False),
                client=None,
            )


class FilterGateTests(unittest.TestCase):
    def test_no_filter_update_during_warmup(self):
        config = ESConfig(filter_update_every=1, filter_warmup_generations=5)
        prompt = scored("attack text here", 0.8, 1.0)
        _, metrics, event = _maybe_evolve_filter(3, config, "filter", [prompt], None, "m")
        self.assertIsNone(event)
        self.assertEqual(metrics["filter_attempted"], 0.0)

    def test_invalid_or_too_few_candidates_do_not_train_filter(self):
        config = ESConfig(filter_update_every=1, filter_min_positive_candidates=2)
        valid = scored("attack text here", 0.8, 1.0)
        invalid = scored("other attack text", 0.9, 1.0)
        invalid.metrics["valid"] = 0.0
        new_filter, _, event = _maybe_evolve_filter(
            1, config, "filter", [invalid, valid], None, "m"
        )
        self.assertEqual(new_filter, "filter")
        self.assertEqual(event["rejection_reason"], "insufficient_positive_attack_candidates")


class _EchoClient:
    def generate(self, prompt, **kwargs):
        return {"text": f"reply:{len(prompt)}"}


class ThroughputTests(unittest.TestCase):
    def test_concurrent_assignment_matches_sequential_records(self):
        def batch():
            prompts = [Prompt(input_prompt=f"Explain topic number {i} safely.") for i in range(6)]
            prompts[2].metadata["target_sample_count"] = 4
            return prompts

        sequential, parallel = batch(), batch()
        records_a = assign_outputs("filter", sequential, _EchoClient(), k_evals=2, concurrency=1)
        records_b = assign_outputs("filter", parallel, _EchoClient(), k_evals=2, concurrency=4)

        self.assertEqual(records_a, records_b)
        self.assertEqual(len(parallel[2].output_prompts), 4)
        self.assertEqual(len(parallel[0].output_prompts), 2)


class CheckpointResumeTests(unittest.TestCase):
    def config(self, stream_dir, generations):
        return ESConfig(
            lambda_=4,
            mu=2,
            generations=generations,
            lightweight=True,
            random_seed=41,
            verbose=False,
            stream_dir=stream_dir,
            checkpoint_every=2,
        )

    @staticmethod
    def deterministic(history):
        return [
            {k: v for k, v in row.items() if k != "generation_seconds"}
            for row in history
        ]

    def test_resumed_run_matches_uninterrupted_run(self):
        with tempfile.TemporaryDirectory() as straight_dir, tempfile.TemporaryDirectory() as split_dir:
            straight = evolutionary_strategy_run(self.config(straight_dir, 6), client=None)

            evolutionary_strategy_run(self.config(split_dir, 4), client=None)
            state = load_checkpoint(str(Path(split_dir) / RunStream.CHECKPOINT))
            self.assertEqual(state["generation"], 4)
            resumed = evolutionary_strategy_run(
                self.config(split_dir, 6), client=None, resume_state=state
            )

            self.assertEqual(self.deterministic(straight.history), self.deterministic(resumed.history))
            self.assertEqual(straight.best.input_prompt, resumed.best.input_prompt)
            for name in (RunStream.SAMPLES, RunStream.HISTORY):
                straight_rows = (Path(straight_dir) / name).read_text().splitlines()
                resumed_rows = (Path(split_dir) / name).read_text().splitlines()
                self.assertEqual(len(straight_rows), len(resumed_rows), name)
            history_rows = [
                json.loads(line)
                for line in (Path(split_dir) / RunStream.HISTORY).read_text().splitlines()
            ]
            self.assertEqual([row["generation"] for row in history_rows], [1, 2, 3, 4, 5, 6])

    def test_truncate_drops_rows_after_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            stream = RunStream(directory)
            stream.reset()
            for generation in (1, 2, 3):
                stream.write_history({"generation": float(generation)})
            stream.truncate_to_generation(2)
            rows = (Path(directory) / RunStream.HISTORY).read_text().splitlines()
            self.assertEqual(len(rows), 2)



class PresetTests(unittest.TestCase):
    def test_preset_fills_defaults_but_explicit_flags_win(self):
        from run_es import parse_args

        args = parse_args(["--preset", "full", "--mu", "5", "--filter-update-every=0"])
        self.assertEqual(args.mu, 5)
        self.assertEqual(args.filter_update_every, 0)
        self.assertEqual(args.lambda_, 64)
        self.assertEqual(args.llm_concurrency, 8)

    def test_no_preset_keeps_parser_defaults(self):
        from run_es import parse_args

        args = parse_args([])
        self.assertEqual(args.mu, 3)
        self.assertEqual(args.llm_concurrency, 1)



class EvaluationBudgetTests(unittest.TestCase):
    @staticmethod
    def run_budget(mu, lambda_, budget, survival="(mu+lambda)", **extra):
        return evolutionary_strategy_run(
            ESConfig(
                mu=mu,
                lambda_=lambda_,
                generations=1_000_000,
                max_evaluations=budget,
                survival_schema=survival,
                lightweight=True,
                random_seed=5,
                verbose=False,
                **extra,
            ),
            client=None,
        )

    def test_run_stops_within_the_attacker_call_budget(self):
        result = self.run_budget(4, 16, 150)
        child_cost = 2  # direct + one filtered sample in dry-run
        self.assertEqual(result.stop_reason, "evaluation_budget")
        self.assertLessEqual(result.attacker_calls, 150)
        self.assertGreater(result.attacker_calls, 150 - child_cost)
        self.assertEqual(result.history[-1]["attacker_calls"], result.attacker_calls)
        self.assertLess(result.history[-1]["generation_lambda"], 16)

    def test_population_sizes_share_the_same_budget(self):
        small = self.run_budget(4, 16, 400)
        large = self.run_budget(16, 64, 400)
        self.assertGreater(len(small.history), len(large.history))
        for result in (small, large):
            self.assertGreater(result.attacker_calls, 400 - 2)
            self.assertLessEqual(result.attacker_calls, 400)

    def test_comma_survival_stops_before_an_undersized_generation(self):
        result = self.run_budget(4, 16, 150, survival="(mu,lambda)")
        self.assertTrue(all(row["generation_lambda"] >= 4 for row in result.history))
        self.assertLessEqual(result.attacker_calls, 150)

    def test_filter_schedule_follows_attacker_calls(self):
        result = self.run_budget(
            4, 16, 400,
            filter_update_every_evaluations=100,
            filter_warmup_evaluations=100,
        )
        attempt_calls = [
            row["attacker_calls"] for row in result.history if row.get("filter_attempted", 0.0) or
            any(event["generation"] == row["generation"] for event in result.filter_events)
        ]
        # Marks at 200 and 300; the final generation may cross 400.
        self.assertGreaterEqual(len(result.filter_events), 2)
        self.assertTrue(all(calls >= 200 for calls in attempt_calls))

    def test_generation_and_evaluation_schedules_are_exclusive(self):
        with self.assertRaises(ValueError):
            self.run_budget(4, 16, 100, filter_update_every=5, filter_update_every_evaluations=50)

    def test_population_flag_sets_sizes_and_budget_raises_generation_cap(self):
        from run_es import parse_args

        args = parse_args(["--population", "large", "--max-evaluations", "5000"])
        self.assertEqual((args.mu, args.lambda_, args.max_survivors_per_seed), (32, 128, 8))
        self.assertGreaterEqual(args.generations, 1_000_000)
        args = parse_args(["--preset", "full", "--population", "small", "--lambda", "20"])
        self.assertEqual((args.mu, args.lambda_), (4, 20))
        self.assertEqual(args.max_evaluations, 60000)



class NestedInitialPopulationTests(unittest.TestCase):
    @staticmethod
    def seed_ids(mu, seed, nest=32):
        result = evolutionary_strategy_run(
            ESConfig(
                mu=mu,
                lambda_=mu * 4,
                generations=0,
                initial_population_nest=nest,
                lightweight=True,
                random_seed=seed,
                verbose=False,
            ),
            client=None,
        )
        return [row["prompt_id"] for row in result.lineage_records if row["generation"] == 0]

    def test_population_sizes_take_prefixes_of_one_draw(self):
        small, medium, large = (self.seed_ids(mu, 9) for mu in (4, 16, 32))
        self.assertEqual(large[:16], medium)
        self.assertEqual(medium[:4], small)
        self.assertEqual(len(set(large)), 32)

    def test_different_seeds_give_different_draws(self):
        self.assertNotEqual(self.seed_ids(16, 9), self.seed_ids(16, 10))

    def test_nested_draw_leaves_the_same_random_stream_for_every_size(self):
        import random

        states = []
        for mu in (4, 32):
            self.seed_ids(mu, 9)
            states.append(random.random())
        # Without nesting the draw size, and so the stream that follows the
        # initial population, would depend on mu.
        self.assertEqual(states[0], states[1])


if __name__ == "__main__":
    unittest.main()
