from __future__ import annotations

from typing import List

from llm_client import LocalLLMClient


def _clean_assistant_text(text: str) -> str:
    text = (text or "").strip()
    marker = "Assistant:"
    if marker in text:
        text = text.split(marker, 1)[-1].strip()
    return text


def _extract_response_text(response) -> str:
    if response is None:
        return ""
    if isinstance(response, str):
        return _clean_assistant_text(response)
    if isinstance(response, dict):
        for key in ("text", "generated_text", "content", "response"):
            if key in response and response[key] is not None:
                return _clean_assistant_text(str(response[key]))
    return _clean_assistant_text(str(response))


def _generate(client, prompt: str, model_name: str, temperature: float = 0.7) -> str:
    if client is None:
        return "Test Mode: No API Client provided."

    if isinstance(client, LocalLLMClient) or hasattr(client, "generate"):
        resp = client.generate(
            prompt=prompt,
            max_new_tokens=256,
            temperature=temperature,
            top_p=0.9,
            do_sample=temperature > 0,
        )
        return _extract_response_text(resp)

    # OpenAI/Groq-style fallback.
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return response.choices[0].message.content.strip()


def assign_outputs(
    filter_prompt: str,
    prompts: List["Prompt"],
    client,
    model_name: str = "local-qwen",
    generate_direct: bool = True,
    k_evals: int = 1,
) -> None:
    """
    Populate Prompt.output_prompts using the current filter prompt.

    The fitness function averages over Prompt.output_prompts, so k_evals controls
    how many stochastic filtered responses are collected per prompt. Direct output
    is generated once per prompt without the defensive filter context and reused as
    the MR baseline.
    """
    k_evals = max(1, int(k_evals or 1))

    for prompt_obj in prompts:
        try:
            if generate_direct and not getattr(prompt_obj, "direct_output", ""):
                direct_prompt = f"User:\n{prompt_obj.input_prompt}\n\nAssistant:"
                prompt_obj.direct_output = _generate(
                    client, direct_prompt, model_name=model_name, temperature=0.7
                )

            combined_prompt = (
                f"System:\n{filter_prompt}\n\n"
                f"User:\n{prompt_obj.input_prompt}\n\n"
                f"Assistant:"
            )
            while len(getattr(prompt_obj, "output_prompts", []) or []) < k_evals:
                assistant_message = _generate(
                    client, combined_prompt, model_name=model_name, temperature=0.7
                )
                prompt_obj.output_prompts.append(assistant_message)

        except Exception as e:
            print(f"   [API ERROR]: {e}")
            if not getattr(prompt_obj, "direct_output", ""):
                prompt_obj.direct_output = "Error: Could not fetch direct response."
            if not getattr(prompt_obj, "output_prompts", None):
                prompt_obj.output_prompts = []
            while len(prompt_obj.output_prompts) < k_evals:
                prompt_obj.output_prompts.append("Error: Could not fetch response.")
