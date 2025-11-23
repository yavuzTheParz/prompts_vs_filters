from dataclasses import dataclass, field
from enum import Enum
from typing import List
from Prompt_class import Prompt, Structure, Content  # dosya adına göre değiştir

def assign_outputs(prompts: List[Prompt]):
    #run_llm()
     for prompt in prompts:
        # Her bir prompt için geçici test outputs yaz
        if prompt.content ==  Content.bomb_weapons:
            prompt.output_prompts = ["I'm sorry, I can't assist with that request.Creating or discussing harmful actions is illegal and dangerous. Please make sure to act responsibly and safely."]
        elif prompt.content == Content.violence:
            prompt.output_prompts = ["I'm sorry, I can't assist with that request. Discrimination is harmful and unethical. It is important to treat all individuals with respect, kindness, and fairness."]
        elif prompt.content == Content.hacking_cybercrime:
            prompt.output_prompts = ["I'm sorry, I can't assist with that request. Hacking is illegal and unethical. It is important to respect privacy and security, and to follow legal and responsible practices online."]
        else:
            prompt.output_prompts = ["Default simulated output"]  # Diğer içerikler için varsayılan çıktı

    

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

