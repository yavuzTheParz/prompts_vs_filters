import pandas as pd
import ast

# === DOSYA ADLARI ===
INPUT_CSV = "filtered-prompts-with-label-1_labeled.csv"
OUTPUT_STRUCT = "subset_structure.csv"
OUTPUT_CONTENT = "subset_content.csv"
OUTPUT_COMBINED = "initial_population.csv"

# === KOLON ADLARI ===
TEXT_COL = "text"
LABEL_COL = "labels"
STRUCT_COL = "structure_labels"

# === SEÇİLEN STRUCTURE LABEL'LAR ===
STRUCT_BASE = "ignore_all_override"
OTHER_STRUCTS = [
    "role_reprogramming",
    "poem_request",
]

# === SEÇİLEN CONTENT LABEL'LAR ===
BEHAV_BASE = "bomb_weapons"
OTHER_BEHAVS = [
    "hacking_cybercrime",
    "misinformation",
    "hate_abuse",
]


# ======================
# HELPER FONKSİYONLAR
# ======================

def to_list(val):
    """String içindeki listeyi gerçek listeye çevirir."""
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        try:
            return ast.literal_eval(val)
        except:
            return [v.strip() for v in val.split(",") if v.strip()]
    return []


def has_base_and_any_of(lst, base_label, other_labels):
    """base label + diğer label'lardan en az birini içeriyorsa True."""
    s = set(lst)
    return (base_label in s) and bool(s & set(other_labels))


# ======================
# ANA PROGRAM
# ======================

def main():
    print(f"Dosya okunuyor: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)

    # liste dönüşümleri
    df[LABEL_COL] = df[LABEL_COL].apply(to_list)
    df[STRUCT_COL] = df[STRUCT_COL].apply(to_list)

    print(f"Toplam satır: {len(df)}")

    # === STRUCTURE FİLTRESİ ===
    struct_mask = df[STRUCT_COL].apply(
        lambda lst: has_base_and_any_of(lst, STRUCT_BASE, OTHER_STRUCTS)
    )
    df_struct = df[struct_mask].copy()

    print(f"Structure filtresi sonucu satır: {len(df_struct)}")
    df_struct.to_csv(OUTPUT_STRUCT, index=False)
    print(f"Kaydedildi: {OUTPUT_STRUCT}")

    # === CONTENT FİLTRESİ ===
    content_mask = df[LABEL_COL].apply(
        lambda lst: has_base_and_any_of(lst, BEHAV_BASE, OTHER_BEHAVS)
    )
    df_content = df[content_mask].copy()

    print(f"Content filtresi sonucu satır: {len(df_content)}")
    df_content.to_csv(OUTPUT_CONTENT, index=False)
    print(f"Kaydedildi: {OUTPUT_CONTENT}")

   # === BİRLEŞİK INITIAL POPULATION ===
df_combined = pd.concat([df_struct, df_content], ignore_index=True)

# IMPORTANT FIX: only drop duplicates by TEXT column (lists cannot be hashed)
df_combined = df_combined.drop_duplicates(subset=[TEXT_COL]).reset_index(drop=True)

print(f"Initial Population (Birleşik, unique) satır: {len(df_combined)}")
df_combined.to_csv(OUTPUT_COMBINED, index=False)
print(f"Kaydedildi: {OUTPUT_COMBINED}")


if __name__ == "__main__":
    main()
