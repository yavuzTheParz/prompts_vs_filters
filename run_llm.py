from typing import List

# from Prompt_class import Prompt   # sende zaten var
from llm_client import LocalLLMClient


def assign_outputs(filter_prompt: str, prompts: List["Prompt"], client, model_name: str) -> None:
    """
    client:
      - None ise test mode
      - LocalLLMClient ise kendi FastAPI server'ına istek atar
      - (İstersen eski Groq/OpenAI client desteğini koruyacak şekilde de genişletebiliriz)
    """

    for prompt_obj in prompts:
        if client is None:
            prompt_obj.output_prompts.append("Test Mode: No API Client provided.")
            continue

        try:
            # --- Server endpoint'in şu an tek string prompt bekliyor ---
            # Bu yüzden system+user'ı tek metinde birleştiriyoruz.
            combined_prompt = (
                f"System:\n{filter_prompt}\n\n"
                f"User:\n{prompt_obj.input_prompt}\n\n"
                f"Assistant:"
            )

            if isinstance(client, LocalLLMClient):
                resp = client.generate(
                    prompt=combined_prompt,
                    max_new_tokens=256,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                )
                assistant_message = resp["text"].strip()

                # Model bazen promptu da geri echo edebilir.
                # Basit temizlik: "Assistant:" sonrası kısmı almaya çalış.
                marker = "Assistant:"
                if marker in assistant_message:
                    assistant_message = assistant_message.split(marker, 1)[-1].strip()

                prompt_obj.output_prompts.append(assistant_message)
                continue

            # --- (Opsiyonel) Eski Groq/OpenAI yolu kalsın istersen ---
            # Burayı kaldırabilirsin.
            messages = [
                {"role": "system", "content": filter_prompt},
                {"role": "user", "content": prompt_obj.input_prompt},
            ]
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.7
            )
            assistant_message = response.choices[0].message.content.strip()
            prompt_obj.output_prompts.append(assistant_message)

        except Exception as e:
            print(f"   [API ERROR]: {e}")
            prompt_obj.output_prompts.append("Error: Could not fetch response.")