from __future__ import annotations

import unittest

from Prompt_class import Prompt
from fitfunc import evaluate_fitness
from run_llm import assign_outputs


class RecordingClient:
    def __init__(self):
        self.calls = []

    def generate(self, prompt: str, **kwargs) -> str:
        self.calls.append((prompt, kwargs))
        return f"response {len(self.calls)}"


class TransientFailureClient:
    def __init__(self):
        self.calls = 0
        self.filtered_failures = 0

    def generate(self, prompt: str, **kwargs) -> str:
        self.calls += 1
        if prompt.startswith("System:") and self.filtered_failures == 0:
            self.filtered_failures += 1
            raise ConnectionError("temporary test failure")
        return "valid response"


class EmptyResponseClient:
    def generate(self, prompt: str, **kwargs) -> str:
        return ""


class RunLLMSamplingTests(unittest.TestCase):
    def test_extracts_openai_compatible_local_response(self):
        from run_llm import _extract_response_text

        response = {"choices": [{"message": {"content": "local response"}}]}

        self.assertEqual(_extract_response_text(response), "local response")

    def test_k_three_uses_exact_call_counts_temperatures_and_indices(self):
        prompt = Prompt(input_prompt="controlled test", metadata={"generation": 4})
        client = RecordingClient()

        records = assign_outputs(
            "Safety filter.",
            [prompt],
            client,
            model_name="fake-model",
            k_evals=3,
            direct_temperature=0.0,
            filtered_temperature=0.6,
            generation=4,
        )

        self.assertEqual(len(client.calls), 4)
        self.assertEqual(client.calls[0][1]["temperature"], 0.0)
        self.assertTrue(all(call[1]["temperature"] == 0.6 for call in client.calls[1:]))
        filtered = [record for record in records if record["kind"] == "filtered"]
        self.assertEqual([record["sample_index"] for record in filtered], [0, 1, 2])
        self.assertTrue(all(record["generation"] == 4 for record in records))

    def test_transient_failure_is_retried_and_recovers(self):
        prompt = Prompt(input_prompt="controlled test")
        client = TransientFailureClient()

        records = assign_outputs(
            "Safety filter.",
            [prompt],
            client,
            k_evals=1,
            max_sample_retries=1,
        )

        self.assertEqual(client.calls, 3)
        self.assertEqual([record["status"] for record in records], ["valid", "invalid", "valid"])
        self.assertNotIn("api_error", prompt.metadata)
        self.assertTrue(prompt.metadata["valid_llm_response"])

    def test_exhausted_empty_samples_mark_prompt_invalid(self):
        prompt = Prompt(input_prompt="controlled test")

        records = assign_outputs(
            "Safety filter.",
            [prompt],
            EmptyResponseClient(),
            k_evals=1,
            max_sample_retries=1,
        )

        self.assertTrue(all(record["status"] == "invalid" for record in records))
        self.assertIn("api_error", prompt.metadata)
        self.assertFalse(prompt.metadata["valid_llm_response"])
        self.assertEqual(prompt.direct_output, "")
        self.assertEqual(prompt.output_prompts, [])

    def test_k_three_metrics_are_averaged_and_report_standard_deviation(self):
        prompt = Prompt(
            input_prompt="controlled test",
            direct_output="direct",
            output_prompts=["sample-1", "sample-2", "sample-3"],
        )
        similarities = {
            ("sample-1", "reference"): 0.3,
            ("sample-2", "reference"): 0.6,
            ("sample-3", "reference"): 0.9,
            ("sample-1", "direct"): 0.2,
            ("sample-2", "direct"): 0.5,
            ("sample-3", "direct"): 0.8,
        }

        evaluate_fitness(
            [prompt],
            ref_outputs=["reference"],
            direct_outputs=["direct"],
            embed_func=lambda value: value,
            sim_func=lambda left, right: similarities[(left, right)],
            mr_objective="semantic_recovery",
        )

        self.assertAlmostEqual(prompt.metrics["asv"], 0.25)
        self.assertAlmostEqual(prompt.metrics["unsafe_reference_similarity"], 0.6)
        self.assertAlmostEqual(prompt.metrics["mr"], 0.5)
        self.assertAlmostEqual(prompt.metrics["asv_std"], 0.0)
        self.assertAlmostEqual(
            prompt.metrics["unsafe_reference_similarity_std"],
            0.2449489743,
        )
        self.assertAlmostEqual(prompt.metrics["mr_std"], 0.2449489743)
        self.assertEqual(prompt.metrics["sample_count"], 3.0)
        self.assertEqual(prompt.metrics["ambiguous_count"], 3.0)
        self.assertEqual(prompt.metrics["attack_success"], 0.0)


if __name__ == "__main__":
    unittest.main()
