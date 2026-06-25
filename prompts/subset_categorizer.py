import pandas as pd
import ast

# === FILE NAMES ===
INPUT_CSV = "filtered-prompts-with-label-1_labeled.csv"
OUTPUT_STRUCT = "subset_structure.csv"
OUTPUT_CONTENT = "subset_content.csv"
OUTPUT_COMBINED = "initial_population.csv"

# === COLUMN NAMES ===
TEXT_COL = "text"
LABEL_COL = "labels"
STRUCT_COL = "structure_labels"

# === SELECTED STRUCTURE LABELS ===
STRUCT_BASE = "ignore_all_override"
OTHER_STRUCTS = [
    "role_reprogramming",
    "poem_request",
]

# === SELECTED CONTENT LABELS ===
BEHAV_BASE = "bomb_weapons"
OTHER_BEHAVS = [
    "hacking_cybercrime",
    "misinformation",
    "hate_abuse",
]


def to_list(val):
    """Convert string representation of a list into a real list."""
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        try:
            return ast.literal_eval(val)
        except Exception:
            return [v.strip() for v in val.split(",") if v.strip()]
    return []


def has_base_and_any_of(lst, base_label, other_labels):
    """Return True if base label + at least one of other_labels is present."""
    s = set(lst)
    return (base_label in s) and bool(s & set(other_labels))


def main():
    print(f"Reading: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)

    df[LABEL_COL] = df[LABEL_COL].apply(to_list)
    df[STRUCT_COL] = df[STRUCT_COL].apply(to_list)

    print(f"Total rows: {len(df)}")

    # === STRUCTURE FILTER ===
    struct_mask = df[STRUCT_COL].apply(
        lambda lst: has_base_and_any_of(lst, STRUCT_BASE, OTHER_STRUCTS)
    )
    df_struct = df[struct_mask].copy()
    print(f"Structure filter rows: {len(df_struct)}")
    df_struct.to_csv(OUTPUT_STRUCT, index=False)
    print(f"Saved: {OUTPUT_STRUCT}")

    # === CONTENT FILTER ===
    content_mask = df[LABEL_COL].apply(
        lambda lst: has_base_and_any_of(lst, BEHAV_BASE, OTHER_BEHAVS)
    )
    df_content = df[content_mask].copy()
    print(f"Content filter rows: {len(df_content)}")
    df_content.to_csv(OUTPUT_CONTENT, index=False)
    print(f"Saved: {OUTPUT_CONTENT}")

    # FIX: this block was previously mis-indented outside the function body,
    # causing a SyntaxError / NameError at module level.
    # === COMBINED INITIAL POPULATION ===
    df_combined = pd.concat([df_struct, df_content], ignore_index=True)
    df_combined = df_combined.drop_duplicates(subset=[TEXT_COL]).reset_index(drop=True)
    print(f"Initial Population (combined, unique) rows: {len(df_combined)}")
    df_combined.to_csv(OUTPUT_COMBINED, index=False)
    print(f"Saved: {OUTPUT_COMBINED}")


if __name__ == "__main__":
    main()