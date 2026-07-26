from __future__ import annotations

import os
from typing import Dict, List, Optional


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

    if hasattr(client, "generate"):
        resp = client.generate(
            prompt=prompt,
            max_new_tokens=256,
            temperature=temperature,
            top_p=0.9,
            do_sample=temperature > 0,
        )
        return _extract_response_text(resp)

    # OpenAI/Groq-style fallback
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return response.choices[0].message.content.strip()


def _resolve_k_evals(k_evals: Optional[int]) -> int:
    """
    Resolve the number of stochastic evaluations per prompt.

    Priority: explicit argument > environment variable > default of 1.
    The proposal requires K-times evaluation to average out LLM stochasticity.
    """
    if k_evals is not None:
        try:
            return max(1, int(k_evals))
        except (TypeError, ValueError):
            pass
    raw = os.getenv("PROMPTS_VS_FILTERS_K_EVALS", "1")
    try:
        return max(1, int(raw))
    except (TypeError, ValueError):
        return 1


def assign_outputs(
    filter_prompt: str,
    prompts: List["Prompt"],
    client,
    model_name: str = "local-qwen",
    generate_direct: bool = True,
    k_evals: Optional[int] = None,
    direct_temperature: float = 0.0,
    filtered_temperature: float = 0.7,
    max_sample_retries: int = 2,
    generation: int = 0,
) -> List[Dict[str, object]]:
    """
    Populate Prompt.output_prompts and Prompt.direct_output.

    K-times evaluation (k_evals > 1) collects multiple stochastic attacked responses per
    prompt. The fitness function averages ASV and MR over all K outputs, which is the
    correct proposal-aligned evaluation approach.

    - direct_output: generated once without the defensive filter (MR baseline)
    - output_prompts: generated K times with the defensive filter (ASV/MR evaluation)

    If a prompt already has k_evals outputs, it is skipped. This allows resuming partial
    evaluation runs without redundant API calls.
    """
    k = _resolve_k_evals(k_evals)
    new_records: List[Dict[str, object]] = []

    def generate_sample(prompt_obj, prompt, kind, sample_index, temperature):
        attempts = max(1, int(max_sample_retries) + 1)
        last_error = ""
        for attempt in range(1, attempts + 1):
            try:
                text = _generate(
                    client,
                    prompt,
                    model_name=model_name,
                    temperature=temperature,
                )
                if not text:
                    raise ValueError("empty model response")
                record = {
                    "generation": int(generation),
                    "input_prompt": prompt_obj.input_prompt,
                    "kind": kind,
                    "sample_index": int(sample_index),
                    "attempt": attempt,
                    "status": "valid",
                    "temperature": float(temperature),
                    "text": text,
                }
                new_records.append(record)
                prompt_obj.metadata.setdefault("sample_records", []).append(record)
                return text
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                record = {
                    "generation": int(generation),
                    "input_prompt": prompt_obj.input_prompt,
                    "kind": kind,
                    "sample_index": int(sample_index),
                    "attempt": attempt,
                    "status": "invalid",
                    "temperature": float(temperature),
                    "error": last_error,
                }
                new_records.append(record)
                prompt_obj.metadata.setdefault("sample_records", []).append(record)
        raise RuntimeError(
            f"{kind} sample {sample_index} failed after {attempts} attempts: {last_error}"
        )

    for prompt_obj in prompts:
        try:
            # Direct output: one sample without the filter, used as the MR reference baseline
            if generate_direct and not getattr(prompt_obj, "direct_output", ""):
                direct_prompt = f"User:\n{prompt_obj.input_prompt}\n\nAssistant:"
                prompt_obj.direct_output = generate_sample(
                    prompt_obj,
                    direct_prompt,
                    "direct",
                    0,
                    direct_temperature,
                )

            # Attacked output: K samples through the defensive filter
            combined_prompt = (
                f"System:\n{filter_prompt}\n\n"
                f"User:\n{prompt_obj.input_prompt}\n\n"
                f"Assistant:"
            )

            current_outputs = getattr(prompt_obj, "output_prompts", None) or []
            prompt_obj.output_prompts = list(current_outputs)

            while len(prompt_obj.output_prompts) < k:
                sample_index = len(prompt_obj.output_prompts)
                assistant_message = generate_sample(
                    prompt_obj,
                    combined_prompt,
                    "filtered",
                    sample_index,
                    filtered_temperature,
                )
                prompt_obj.output_prompts.append(assistant_message)

        except Exception as e:
            print(f"   [API ERROR]: {e}")
            prompt_obj.metadata["api_error"] = str(e)
            prompt_obj.metadata["valid_llm_response"] = False
            if not getattr(prompt_obj, "direct_output", ""):
                prompt_obj.direct_output = ""
            if not getattr(prompt_obj, "output_prompts", None):
                prompt_obj.output_prompts = []
        else:
            prompt_obj.metadata.pop("api_error", None)
            prompt_obj.metadata["valid_llm_response"] = True

    return new_records
