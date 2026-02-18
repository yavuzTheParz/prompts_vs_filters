import pandas as pd
import ast
from typing import List
from Prompt_class import Prompt, Structure, Content  # dosya adına göre değiştir
from fitfunc import callFitness
from run_llm import assign_outputs

def genetic_algorithm_run(N: int, T: int) -> List[Prompt]:
    """
    N = popülasyon büyüklüğü
    T = iterasyon / jenerasyon sayısı

    Şimdilik: hiçbir şey yapmıyor, sadece boş bir Prompt listesi döndürüyor.
    """
    
    population: List[Prompt] = []

    # İleride burada:
    # - initial population creation
    #population = initialize()


def test_fitness_manual():
    # 1) iki test promptu oluştur
    p1 = Prompt(
        input_prompt="test prompt 1",
        structure=Structure.ignore_all_override,
        content=Content.bomb_weapons
    )
    p2 = Prompt(
        input_prompt="test prompt 2",
        structure=Structure.ignore_all_override,
        content=Content.bomb_weapons
    )

    # 2) direct_output (normal cevap) ELLE koy
    p1.direct_output = "Sure, here is a tutorial for making a bomb."
    p2.direct_output = "Sorry i can't help with that."

    # 3) output_prompts (attacked outputs) ELLE koy
    # p1: ref'a benzer olsun -> ASV yüksek -> fitness yüksek bekleriz
    p1.output_prompts = [
        "Sure, here is a tutorial for making a bomb",
        "First, you get the materials and then...",
    ]

    # p2: ref'ten uzak olsun -> ASV düşük -> fitness düşük bekleriz
    p2.output_prompts = [
        "No, i cannot help with that request.",
        "Sorry i can't help with that."
    ]

    population = [p1, p2]
    

    #assign_outputs(population)

    callFitness(population)
    # - selection
    # - crossover
    # - mutation
    # - survivor_selection

       # 5) sonuçları yazdır
    for i, p in enumerate(population, start=1):
        print(f"\nPROMPT {i}")
        print("content:", p.content)
        print("direct_output:", p.direct_output)
        print("output_prompts:", p.output_prompts)
        print("fitness:", p.fitness)

    return population


CSV_PATH = "prompts\\initial_population.csv"


def pick_first_enum_match(label_list, enum_class):
    """
    CSV'den gelen label listesi içinden,
    enum_class içinde bulunan ilk label'ı seçer.

    Örneğin:
    label_list = ["hate_abuse", "bomb_weapons"]
    enum_class = Content
    → bomb_weapons döner (enum içinde o olduğu için)

    Eğer hiçbir label enum'da yoksa None döner.
    """
    for label in label_list:
        if label in enum_class.__members__:
            return enum_class[label]   # Enum objesine çevir
    return None


"""def initialize() -> List[Prompt]:
    
   # initial_population.csv dosyasını okur ve Prompt objelerini döndürür.

    df = pd.read_csv(CSV_PATH)

    prompts: List[Prompt] = []

    for _, row in df.iterrows():
        text = row["text"]

        # string -> list dönüşümü
        struct_list = ast.literal_eval(row["structure_labels"])
        content_list = ast.literal_eval(row["labels"])

        # === Structure seçimi ===
        structure_enum = pick_first_enum_match(struct_list, Structure)
        if structure_enum is None:
            # Eğer hiçbiri enum içinde değilse default verelim
            structure_enum = Structure.ignore_all_override

        # === Content seçimi ===
        content_enum = pick_first_enum_match(content_list, Content)
        if content_enum is None:
            content_enum = Content.bomb_weapons

        # Prompt objesi oluştur
        p = Prompt(
            input_prompt=text,
            structure=structure_enum,
            content=content_enum
        )

        prompts.append(p)

    return prompts"""

 

# sadece test çalıştır
test_fitness_manual()

#genetic_algorithm_run(870, 1)
