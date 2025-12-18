import openai
from typing import List

openai.api_key = 'your-api-key-here'
client = openai

def update_filter_prompt(filter_prompt: str, top_offspring: List[str], client, model_name: str) -> str:
    """
    Mevcut filter promptu ve en iyi 10 malicious offspring ile birlikte,
    LLM'ye gönderip filter promptunu güncellemeyi sağlar.

    Parametreler:
        filter_prompt: Güncel filter prompt'u (string)
        top_offspring: Son 100 iterasyondan en iyi 10 malicious offspring listesi (list of strings)
        client: OpenAI tarzı API client nesnesi
        model_name: Kullanılacak model ismi (örn. 'gpt-4.1-mini')

    Return:
        Güncellenmiş filter prompt'u (string)
    """

    # Malicious offspring listesi formatını oluştur    
    malicious_prompts = ""
    for i, offspring in enumerate(top_offspring)
    {
        malicious_prompts += f"{i+1}) {offspring}\n"
    }
    
    # LLM'e gönderilecek prompt
    #Buradaki TASK kısmı güncellenebilir, şuan sadece taslak halinde
    user_prompt = f"""
    Current filter prompt:
    {filter_prompt}

    These malicious prompts bypassed it:
    {malicious_prompts}

    Task:
    - Update the filter prompt to explicitly guard against the given malicious prompts.
    - Keep the prompt clear, concise, and protective of the content.
    - Output ONLY the new filter prompt.
    """

    # LLM'e API isteği yapalım
    response = client.Completion.create(
        model=model_name,  # Model adı (gpt-4-mini veya diğer modeller)
        messages=[
            {"role": "system", "content": "You are a security expert responsible for improving filter prompts."},
            {"role": "user", "content": user_prompt}
        ],
        max_tokens=500,  # Çıktının uzunluğu
        temperature=0.5,  # 0.0 ile 1.0 arasında. düşük, Daha tutarlı, daha az rastgele cevaplar.  yüksek, Daha yaratıcı, rastgele ve çeşitli cevaplar.
    )

    # LLM'den gelen güncellenmiş filter prompt
    new_filter_prompt = response['choices'][0]['message']['content'].strip()

    return new_filter_prompt