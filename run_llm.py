from dataclasses import dataclass, field
from enum import Enum
from typing import List



#client: API aslında, filter_prompt: mainden gelir, model_name: hangi LLM'i kullanıyoruz, "gpt-4.1-mini" etc., None: ne return edecek.
def run_llm(filter_prompt: str, prompts: List[Prompt], client, model_name: str) -> None:
    """
    Filter prompt'u system rolüne, offspring inputlarını user rolüne koyup
    LLM'e gönderir. Cevapları her Prompt objesinin output_prompts listesine ekler.
    """

    for prompt_obj in prompts:
        messages = [
            {"role": "system", "content": filter_prompt},
            {"role": "user", "content": prompt_obj.input_prompt},
        ]

        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
        )

        assistant_message = response.choices[0].message.content.strip()

        # Son cevabı logla
        prompt_obj.output_prompts.append(assistant_message)

