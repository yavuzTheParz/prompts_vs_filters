import pandas as pd
import re

# === AYARLAR ===
CSV_PATH = "filtered-prompts-with-label-1.csv"
OUTPUT_PATH = "filtered-prompts-with-label-1_labeled.csv"

TEXT_COL = "text"
SENTENCE_LIMIT = 2

# === BEHAVIOR LABEL KELİMELERİ ===
KEYWORDS = {
    "bomb_weapons": ["bomb", "explosive", "tnt", "detonator", "molotov",
                      "gun", "pistol", "rifle", "ak-47", "knife", "grenade"],

    "chemical_threat": ["mustard gas", "chlorine gas", "sarin", "vx nerve",
                        "chemical weapon", "toxic gas", "poison gas"],

    "biological_threat": ["bioweapon", "anthrax", "smallpox", "virus spread",
                          "engineer a virus", "biological attack"],

    "hacking_cybercrime": ["hack", "hacking", "exploit", "sql injection", "xss",
                           "buffer overflow", "ddos", "bruteforce", "brute force",
                           "backdoor", "rootkit", "shellcode", "0-day", "zero-day",
                           "pentest", "penetration test bypass"],

    "malware": ["malware", "ransomware", "trojan", "keylogger",
                "virus", "worm", "botnet"],

    "fraud_scams": ["phishing", "scam", "advance fee", "romance scam",
                    "fake invoice", "steal credit card", "carding"],

    "financial_crime": ["money laundering", "launder money", "tax evasion",
                        "offshore account", "shell company"],

    "drugs": ["cook meth", "make meth", "lsd", "cocaine", "heroin",
              "mdma", "ecstasy", "grow weed", "marijuana", "cannabis",
              "drug trafficking"],

    "violence": ["kill", "murder", "stab", "beat up", "assassinate",
                 "torture", "kidnap", "abduct"],

    "extremism": ["terrorist", "terrorism", "jihad", "extremist group",
                  "join isis", "join daesh", "white supremacist"],

    "hate_abuse": ["hate speech", "racial slur", "nazi", "kkk",
                   "degrade women", "degrade men", "homophobic", "transphobic"],

    "harassment_bullying": ["harass", "bully", "stalk", "doxx", "doxxing",
                            "humiliate", "intimidate"],

    "sexual_content": ["explicit sex", "porn", "sexual position",
                       "erotic story", "fetish"],

    "child_safety": ["underage", "minor", "child porn", "cp", "loli",
                     "kid nudes", "teen nudes"],

    "self_harm": ["kill myself", "commit suicide", "end my life",
                  "self-harm", "self harm", "cut myself"],

    "medical_harm": ["dangerous dose", "overdose on", "ignore doctor",
                     "fake vaccine", "unproven cure"],

    "privacy_invasion": ["doxx", "doxxing", "leak address", "leak personal info",
                         "steal personal data", "track location"],

    "impersonation": ["impersonate", "pretend to be", "fake identity",
                      "spoof caller id", "spoof email"],

    "misinformation": ["fake news", "spread conspiracy", "disinformation",
                       "make people believe a lie"],

    "political_manipulation": ["targeted political ads", "manipulate voters",
                               "swing the election", "microtarget voters",
                               "propaganda campaign"],

    "exam_cheating": ["cheat on exam", "exam cheat", "test answers",
                      "leak exam questions", "proxy test taker"],

    "academic_misconduct": ["write my essay", "write my homework", "plagiarize",
                            "bypass plagiarism", "ghostwrite thesis"],

    "policy_bypass_jailbreak": ["ignore previous instructions", "jailbreak",
                                "unfiltered", "act as dan", "do anything now",
                                "bypass safety", "break the rules",
                                "pretend to not be an ai"],

    "other_illegal": ["illegal", "black market", "dark web", "buy fake id",
                      "smuggle", "counterfeit", "forged passport"]
}


def label_structure(text: str):
    if not isinstance(text, str):
        return ["no_structure"]

    t = text.strip().lower()
    labels = set()

    # 1) Ignore / disregard / do not follow → override
    if t.startswith("ignore all") or t.startswith("disregard all") or "do not follow" in t:
        labels.add("ignore_all_override")
    # 2) CONTEXT WIPE / RESET → "start with a clean sheet", "clean slate"
    if (
        "start with a clean sheet" in t
        or "start from a clean sheet" in t
        or "start with a clean slate" in t
        or "start from a clean slate" in t
        or "begin with a clean sheet" in t
        or "begin with a clean slate" in t
        or "forget all previous context" in t
        or "forget the previous context" in t
        or "reset the conversation" in t
    ):
        labels.add("context_wipe_reset")

        # FORCED START CONSTRAINT — "start with", "begin with", "must start with"
    if (
        t.startswith("start with")
        or t.startswith("begin with")
        or " must start with " in t
        or " should start with " in t
        or " response must start with " in t
        or " response should start with " in t
        or " starting with " in t.split()[:6]  # early position constraint
        or " begins with " in t.split()[:6]
    ):
        labels.add("forced_start_constraint")



    # 2) Poem / story / song istekleri → creative writing request
    if (
        t.startswith("write me a poem")
        or t.startswith("write a poem")
        or t.startswith("write me a story")
        or t.startswith("write a story")
        or t.startswith("compose a song")        # YENİ
        or t.startswith("compose me a song")     # YENİ
        or t.startswith("compose a poem")        # ek mantıklı varyant
        or t.startswith("compose") and "song" in t.split()[:4]  # daha geniş yakalama
    ):
        labels.add("poem_request")

    # 3) Role reprogramming / persona tanımlama
    if (
        t.startswith("from now on")
        or t.startswith("you are now")
        or t.startswith("from this moment on")
        or t.startswith("act as ")
        or t.startswith("pretend you are")
        or t.startswith("you are")
    ):
        labels.add("role_reprogramming")

    # 4) Stepwise injection
    if "first," in t and "second," in t:
        labels.add("stepwise_instruction")

    # 5) Şantaj / tehdit yapısı
    if (
        " or else i will" in t
        or " or else i'll" in t
        or (" if you don't" in t and " i will" in t)
        or (" if you do not" in t and " i will" in t)
    ):
        labels.add("threat_coercion")

    # 6) Soru yapısı (fallback)
    starts_like_question = t.startswith(
        (
            "how ", "what ", "why ", "when ", "where ", "who ",
            "can you", "could you", "would you",
            "is it", "are you", "do you", "does ", "did "
        )
    )
    if not labels and (t.endswith("?") or starts_like_question):
        labels.add("question_request")

    # 7) Genel emir cümlesi (fallback)
    if not labels:
        if re.match(r"^(write|explain|generate|give|list|provide|create|summarize|tell|show|compose)\b", t):
            labels.add("imperative_instruction")

    if not labels:
        labels.add("no_structure")

    return sorted(labels)


# === BEHAVIOR LABELS ===
def label_prompt(text: str):
    if not isinstance(text, str):
        return ["benign"]

    t = text.lower()
    labels = set()

    for label, words in KEYWORDS.items():
        for kw in words:
            if kw in t:
                labels.add(label)
                break

    if not labels:
        labels.add("benign")

    return sorted(labels)


# === CÜMLE SAYACI ===
def count_sentences(text: str):
    if not isinstance(text, str):
        return 0

    parts = re.split(r"[\.!?]+", text)
    return len([p.strip() for p in parts if p.strip()])


# === ANA PROGRAM ===
def main():
    print("CSV okunuyor...")
    df = pd.read_csv(CSV_PATH)

    if TEXT_COL not in df.columns:
        raise ValueError(
            f'"{TEXT_COL}" adında bir kolon bulunamadı. Kolonlar: {list(df.columns)}'
        )

    before_len = len(df)
    print(f"Toplam satır: {before_len}")

    print("Cümle filtresi uygulanıyor...")
    df["sentence_count"] = df[TEXT_COL].apply(count_sentences)
    df = df[df["sentence_count"] <= SENTENCE_LIMIT].copy()
    print(f"Filtre sonrası kalan satır: {len(df)} (silinen: {before_len - len(df)})")

    print("Behavior label ataması...")
    df["labels"] = df[TEXT_COL].apply(label_prompt)

    print("Structure label ataması...")
    df["structure_labels"] = df[TEXT_COL].apply(label_structure)

    print("Kaydediliyor:", OUTPUT_PATH)
    df.to_csv(OUTPUT_PATH, index=False)
    print("Tamamdır Devasa Frontal Lobelu Lordum.")


if __name__ == "__main__":
    main()
