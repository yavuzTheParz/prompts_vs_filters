import pandas as pd
import ast
from collections import Counter

# === DOSYA ADLARI ===
INPUT_CSV = "filtered-prompts-with-label-1_labeled.csv"
OUTPUT_STRUCT = "subset_structure.csv"
OUTPUT_CONTENT = "subset_content.csv"
OUTPUT_COMBINED = "initial_population.csv"
OUTPUT_STATS = "label_stats.txt"   # YENİ: label sayıları buraya yazılacak

# === KOLON ADLARI ===
TEXT_COL = "text"
LABEL_COL = "labels"
STRUCT_COL = "structure_labels"

# === STRUCTURE LABEL SETİ (OR MANTIĞI) ===
STRUCT_SET = {
    "ignore_all_override",
    "question_request",
    "imperative_instruction",
}

# === CONTENT LABEL SETİ (OR MANTIĞI) ===
CONTENT_SET = {
    "bomb_weapons",
    "hacking_cybercrime",
    "violence",
}


def to_list(val):
    """
    '["a", "b"]' gibi stringleri listeye çevirir.
    Zaten listeyse aynen döner.
    """
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        try:
            return ast.literal_eval(val)
        except Exception:
            return [v.strip() for v in val.split(",") if v.strip()]
    return []


def has_any_from_set(lst, label_set):
    """
    lst: örn. ['bomb_weapons', 'exam_cheating']
    label_set: örn. {'bomb_weapons', 'hate_abuse'}
    -> Kesişim varsa True
    """
    return bool(set(lst) & label_set)


def write_label_stats(df: pd.DataFrame, stats_path: str):
    """
    Tüm structure_labels ve labels kolonları için:
    - Hangi labeldan kaç tane prompt var?
    - Sonuçları txt dosyasına yazar (en çoktan en aza).
    """
    struct_counter = Counter()
    content_counter = Counter()

    # structure_labels sayımı
    for lst in df[STRUCT_COL]:
        for lab in lst:
            struct_counter[lab] += 1

    # content labels sayımı
    for lst in df[LABEL_COL]:
        for lab in lst:
            content_counter[lab] += 1

    # büyükten küçüğe sırala
    struct_sorted = sorted(struct_counter.items(), key=lambda x: x[1], reverse=True)
    content_sorted = sorted(content_counter.items(), key=lambda x: x[1], reverse=True)

    with open(stats_path, "w", encoding="utf-8") as f:
        f.write("=== STRUCTURE LABEL COUNTS ===\n")
        for label, count in struct_sorted:
            f.write(f"{label}: {count}\n")

        f.write("\n\n=== CONTENT (BEHAVIOR) LABEL COUNTS ===\n")
        for label, count in content_sorted:
            f.write(f"{label}: {count}\n")

    print(f"Label istatistikleri yazıldı: {stats_path}")


def main():
    print(f"Dosya okunuyor: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)

    # labels ve structure_labels'ı listeye çevir
    df[LABEL_COL] = df[LABEL_COL].apply(to_list)
    df[STRUCT_COL] = df[STRUCT_COL].apply(to_list)

    print(f"Toplam satır: {len(df)}")

    # === TÜM DATA İÇİN LABEL İSTATİSTİKLERİ ===
    write_label_stats(df, OUTPUT_STATS)

    # === STRUCTURE FİLTRESİ (yalnızca structure'a bakan OR filtresi) ===
    struct_mask = df[STRUCT_COL].apply(lambda lst: has_any_from_set(lst, STRUCT_SET))
    df_struct = df[struct_mask].copy()
    print(f"Seçilen structure'lardan en az birini içeren satır sayısı: {len(df_struct)}")
    df_struct.to_csv(OUTPUT_STRUCT, index=False)
    print(f"Kaydedildi: {OUTPUT_STRUCT}")

    # === CONTENT FİLTRESİ (yalnızca content'e bakan OR filtresi) ===
    content_mask = df[LABEL_COL].apply(lambda lst: has_any_from_set(lst, CONTENT_SET))
    df_content = df[content_mask].copy()
    print(f"Seçilen content label'lardan en az birini içeren satır sayısı: {len(df_content)}")
    df_content.to_csv(OUTPUT_CONTENT, index=False)
    print(f"Kaydedildi: {OUTPUT_CONTENT}")

    # === ORTAK FİLTRE: HEM STRUCTURE HEM CONTENT UYUMLU OLANLAR (AND) ===
    joint_mask = struct_mask & content_mask
    df_joint = df[joint_mask].copy()
    print(f"Hem structure hem content şartını sağlayan satır sayısı: {len(df_joint)}")

    # Aynı text birden fazla kez varsa unique olsun:
    df_joint = df_joint.drop_duplicates(subset=[TEXT_COL]).reset_index(drop=True)

    df_joint.to_csv(OUTPUT_COMBINED, index=False)
    print(f"Initial Population kaydedildi: {OUTPUT_COMBINED}")


if __name__ == "__main__":
    main()
