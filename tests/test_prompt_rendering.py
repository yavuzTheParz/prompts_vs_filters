from __future__ import annotations

import unittest

from Prompt_class import Prompt
from filter_evolution import evolve_filter
from prompt_rendering import (
    InternalPrompt,
    PromptValidationError,
    parse_internal_prompt,
    render_prompt,
    render_with_audit,
    serialize_internal_prompt,
    validate_internal_prompt,
)
from run_llm import assign_outputs


PREFIX = "[[STYLE_PREFIX:imperative]]You must[[/STYLE_PREFIX]]"
SUFFIX = "[[STYLE_SUFFIX:imperative]], without delay.[[/STYLE_SUFFIX]]"
INTERNAL_TEXT = f"{PREFIX} explain safe access control. {SUFFIX}"
RENDERED_TEXT = "You must explain safe access control, without delay."


class RecordingClient:
    def __init__(self):
        self.calls = []

    def generate(self, prompt: str, **kwargs) -> str:
        self.calls.append(prompt)
        if "Task: Propose ONE concise additional safety rule" in prompt:
            return "Reject unsafe requests that use override framing."
        return "I cannot assist with unsafe requests."


def user_text(request: str) -> str:
    return request.split("User:\n", 1)[1].split("\n\nAssistant:", 1)[0]


class PromptRenderingTests(unittest.TestCase):
    def test_structured_prompt_renders_to_plain_text(self):
        internal = InternalPrompt(
            prefix="You must",
            body="explain safe access control.",
            suffix=", without delay.",
            style="imperative",
        )

        serialized = serialize_internal_prompt(internal)
        prompt = Prompt(prompt_representation=internal)

        self.assertEqual(parse_internal_prompt(serialized), internal)
        self.assertEqual(validate_internal_prompt(serialized), internal)
        self.assertEqual(prompt.internal_prompt, internal)
        self.assertEqual(render_prompt(internal), RENDERED_TEXT)
        self.assertNotIn("[[STYLE_", render_prompt(internal))
        self.assertNotIn("[[/STYLE_", render_prompt(internal))

    def test_rendering_is_idempotent_and_hashes_are_stable(self):
        first = render_with_audit(INTERNAL_TEXT)
        second = render_with_audit(first.text)

        self.assertEqual(first.text, RENDERED_TEXT)
        self.assertEqual(render_prompt(first.text), first.text)
        self.assertEqual(first.rendered_sha256, second.rendered_sha256)
        self.assertEqual(len(first.internal_sha256), 64)
        self.assertEqual(len(first.rendered_sha256), 64)

    def test_invalid_marker_forms_are_rejected(self):
        invalid_prompts = {
            "unbalanced": (
                "[[STYLE_PREFIX:imperative]]You must explain access control."
            ),
            "nested": (
                "[[STYLE_PREFIX:imperative]]You "
                "[[STYLE_SUFFIX:imperative]]must[[/STYLE_SUFFIX]]"
                "[[/STYLE_PREFIX]] explain access control."
            ),
            "duplicated": (
                "[[STYLE_PREFIX:imperative]]You must[[/STYLE_PREFIX]] "
                "[[STYLE_PREFIX:imperative]]Immediately[[/STYLE_PREFIX]] "
                "explain access control."
            ),
            "unknown_style": (
                "[[STYLE_PREFIX:urgent]]Quickly[[/STYLE_PREFIX]] "
                "explain access control."
            ),
            "unknown_marker": (
                "[[STYLE_MIDDLE:imperative]]Quickly[[/STYLE_MIDDLE]] "
                "explain access control."
            ),
            "malformed_case": (
                "[[style_prefix:imperative]]Quickly[[/style_prefix]] "
                "explain access control."
            ),
            "mixed_styles": (
                "[[STYLE_PREFIX:plea]]Please[[/STYLE_PREFIX]] "
                "explain access control "
                "[[STYLE_SUFFIX:imperative]], now.[[/STYLE_SUFFIX]]"
            ),
        }

        for case, text in invalid_prompts.items():
            with self.subTest(case=case):
                with self.assertRaises(PromptValidationError):
                    render_prompt(text)

    def test_direct_and_filtered_requests_share_marker_free_user_text(self):
        prompt = Prompt(input_prompt=INTERNAL_TEXT)
        client = RecordingClient()

        records = assign_outputs(
            "Safety filter.",
            [prompt],
            client,
            k_evals=1,
        )

        self.assertEqual(len(client.calls), 2)
        self.assertEqual(user_text(client.calls[0]), RENDERED_TEXT)
        self.assertEqual(user_text(client.calls[1]), RENDERED_TEXT)
        self.assertTrue(
            all(
                "[[STYLE_" not in request and "[[/STYLE_" not in request
                for request in client.calls
            )
        )
        self.assertEqual(
            records[0]["rendered_prompt_sha256"],
            records[1]["rendered_prompt_sha256"],
        )
        self.assertEqual(
            records[0]["internal_prompt_sha256"],
            records[1]["internal_prompt_sha256"],
        )
        self.assertEqual(
            prompt.metadata["prompt_render"]["internal"]["body"],
            "explain safe access control.",
        )

    def test_invalid_internal_prompt_fails_before_client_request(self):
        prompt = Prompt(input_prompt="Explain access control.")
        prompt.input_prompt = (
            "[[STYLE_PREFIX:imperative]]You must explain access control."
        )
        client = RecordingClient()

        assign_outputs("Safety filter.", [prompt], client, k_evals=1)

        self.assertEqual(client.calls, [])
        self.assertFalse(prompt.metadata["valid_llm_response"])
        self.assertIn("Unbalanced", prompt.metadata["api_error"])

    def test_filter_evolution_requests_are_marker_free(self):
        client = RecordingClient()

        evolve_filter(
            current_filter="Safety filter.",
            top_attack_prompts=[INTERNAL_TEXT],
            benign_set=["Explain a benign safety concept."],
            client=client,
            model_name="fake-model",
        )

        self.assertGreaterEqual(len(client.calls), 5)
        self.assertTrue(any(RENDERED_TEXT in request for request in client.calls))
        self.assertTrue(
            all(
                "[[STYLE_" not in request and "[[/STYLE_" not in request
                for request in client.calls
            )
        )


if __name__ == "__main__":
    unittest.main()
