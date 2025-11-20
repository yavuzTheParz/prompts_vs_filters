import pandas as pd
import ast
from typing import List
from Prompt_class import Prompt, Structure, Content  # dosya adına göre değiştir

def genetic_algorithm_run(N: int, T: int) -> List[Prompt]:
    """
    N = popülasyon büyüklüğü
    T = iterasyon / jenerasyon sayısı

    Şimdilik: hiçbir şey yapmıyor, sadece boş bir Prompt listesi döndürüyor.
    """
    
    population: List[Prompt] = []

    # İleride burada:
    # - initial population creation
    population = initialize()
    print(population[0])

    # - fitness hesaplama
    # - selection
    # - crossover
    # - mutation
    # - survivor selection
    # vs. adımları olacak.

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


def initialize() -> List[Prompt]:
    """
    initial_population.csv dosyasını okur ve Prompt objelerini döndürür.
    """
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

    return prompts


#genetic_algorithm_run(870, 1)
