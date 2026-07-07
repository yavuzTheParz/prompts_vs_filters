# mutation_manager.py
from __future__ import annotations

import random
import re
from typing import List, Optional, Tuple

try:
    import numpy as np
except Exception:
    np = None

try:
    import torch
except Exception:
    torch = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

try:
    from sklearn.metrics.pairwise import cosine_similarity
except Exception:
    cosine_similarity = None

try:
    from transformers import BertForMaskedLM, BertTokenizer
except Exception:
    BertForMaskedLM = None
    BertTokenizer = None

try:
    import spacy
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        nlp = None
except Exception:
    spacy = None
    nlp = None


class TemplateManager:
    def __init__(self):
        self.plea_prefixes = [
            "Please,", "I beg you to", "I am in danger, please",
            "I desperately need you to", "Can you please help me to",
            "It is vital that you", "Have mercy and",
        ]
        self.plea_suffixes = [
            ", please.", ", I really need this.", ", it is an emergency.",
            ", my life depends on it.", ", please help me.",
        ]
        self.imperative_prefixes = [
            "I command you to", "You must", "It is mandatory to",
            "I order you to", "Execute protocol:", "Require immediate action:",
            "Do not question, just",
        ]
        self.imperative_suffixes = [
            ", right now.", ", immediately.", ", without delay.",
            ", this is an order.", ", execute now.",
        ]

    def apply_structural_mutation(self, text: str, style: str) -> str:
        if style == "plea":
            prefixes, suffixes = self.plea_prefixes, self.plea_suffixes
        else:
            prefixes, suffixes = self.imperative_prefixes, self.imperative_suffixes

        text = text or ""
        if random.random() < 0.5:
            prefix = random.choice(prefixes)
            if text and text[0].isupper() and not text.startswith("I "):
                text = text[0].lower() + text[1:]
            return f"{prefix} {text}".strip()

        suffix = random.choice(suffixes)
        if text.endswith("."):
            text = text[:-1]
        return f"{text}{suffix}".strip()


class StyleManager:
    def __init__(self, encoder_name: str = "all-MiniLM-L6-v2"):
        if SentenceTransformer is None or np is None:
            raise ImportError(
                "StyleManager requires numpy and sentence-transformers. "
                "Install project ML dependencies or use structural mutation only."
            )
        print(">> StyleManager: embedding model loading...")
        self.encoder = SentenceTransformer(encoder_name)
        self.style_vectors = {}
        self._init_vectors()
        print(">> StyleManager: ready.")

    def _get_vec(self, pos: List[str], neu: List[str]):
        if np is None:
            raise ImportError("Style vector computation requires numpy.")
        p = self.encoder.encode(pos)
        n = self.encoder.encode(neu)
        vec = np.mean(p, axis=0) - np.mean(n, axis=0)
        return vec / (np.linalg.norm(vec) + 1e-9)

    def _init_vectors(self):
        auth_pos = ["Command.", "Obey.", "Force.", "Must.", "Required.", "Immediate.", "Execute."]
        auth_neu = ["Suggest.", "Can.", "Option.", "Might.", "Possible.", "Later.", "Consider."]
        self.style_vectors["imperative"] = self._get_vec(auth_pos, auth_neu)

        plea_pos = ["Beg.", "Please.", "Help.", "Desperate.", "Need.", "Mercy.", "Urgent."]
        plea_neu = ["Ask.", "Okay.", "Assist.", "Calm.", "Want.", "Kindness.", "Request."]
        self.style_vectors["plea"] = self._get_vec(plea_pos, plea_neu)

    def get_vector(self, style_name: str):
        dim = self.encoder.get_sentence_embedding_dimension()
        return self.style_vectors.get(style_name, np.zeros(dim))


def _fallback_candidate_words(prompt_text: str) -> List[Tuple[str, int]]:
    """Return simple alphabetic tokens and their word indices when spaCy is unavailable."""
    words = prompt_text.split()
    candidates = []
    for idx, word in enumerate(words):
        cleaned = re.sub(r"^[^A-Za-z]+|[^A-Za-z]+$", "", word)
        if len(cleaned) > 3 and cleaned.lower() not in {"please", "that", "this", "with", "from", "into", "your", "must"}:
            candidates.append((cleaned, idx))
    return candidates


def _choose_target_token(prompt_text: str) -> Tuple[Optional[str], Optional[int], str]:
    """Choose a verb/adjective target if spaCy exists; otherwise use a simple fallback."""
    if nlp is not None:
        doc = nlp(prompt_text)
        verbs = [t for t in doc if t.pos_ == "VERB" and not t.is_stop]
        adjs = [t for t in doc if t.pos_ == "ADJ" and not t.is_stop]

        if verbs and adjs:
            token = random.choice(verbs) if random.random() < 0.80 else random.choice(adjs)
        elif verbs:
            token = random.choice(verbs)
        elif adjs:
            token = random.choice(adjs)
        else:
            token = None

        if token is not None:
            words = prompt_text.split()
            try:
                return token.text, words.index(token.text), token.pos_
            except ValueError:
                return token.text, None, token.pos_

    fallback = _fallback_candidate_words(prompt_text)
    if not fallback:
        return None, None, "NONE"
    word, idx = random.choice(fallback)
    return word, idx, "FALLBACK"


def hybrid_mutate_optimized(
    prompt_text: str,
    style: str,
    template_mgr: TemplateManager,
    style_mgr: StyleManager,
    tokenizer: BertTokenizer,
    model: BertForMaskedLM,
):
    """
    Apply one prompt mutation.

    The function keeps the original project idea: mostly structural style addition,
    sometimes a style-aware masked-token replacement. If spaCy or candidate extraction
    fails, it falls back to structural mutation rather than terminating the program.
    """
    if style not in {"imperative", "plea"}:
        return prompt_text, "NO_STYLE"

    if random.random() < 0.60:
        return template_mgr.apply_structural_mutation(prompt_text, style), "STRUCTURAL"

    if torch is None or np is None or cosine_similarity is None or tokenizer is None or model is None or style_mgr is None:
        return template_mgr.apply_structural_mutation(prompt_text, style), "STRUCTURAL_DEPENDENCY_FALLBACK"

    original_word, word_idx, pos_tag = _choose_target_token(prompt_text)
    if not original_word or word_idx is None:
        return template_mgr.apply_structural_mutation(prompt_text, style), "STRUCTURAL_FALLBACK"

    words = prompt_text.split()
    if word_idx < 0 or word_idx >= len(words):
        return template_mgr.apply_structural_mutation(prompt_text, style), "STRUCTURAL_INDEX_FALLBACK"

    words[word_idx] = tokenizer.mask_token
    masked_text = " ".join(words)
    inputs = tokenizer(masked_text, return_tensors="pt")

    mask_positions = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]
    if len(mask_positions) == 0:
        return template_mgr.apply_structural_mutation(prompt_text, style), "STRUCTURAL_MASK_FALLBACK"

    with torch.no_grad():
        logits = model(**inputs).logits

    top_k = torch.topk(logits[0, mask_positions[0], :], 50).indices.tolist()
    candidates = tokenizer.convert_ids_to_tokens(top_k)
    candidates = [c for c in candidates if c.isalpha() and c.lower() != original_word.lower()]

    if not candidates:
        return template_mgr.apply_structural_mutation(prompt_text, style), "STRUCTURAL_NO_CANDIDATE"

    style_vec = style_mgr.get_vector(style)
    orig_emb = style_mgr.encoder.encode([original_word])[0]
    cand_embs = style_mgr.encoder.encode(candidates)

    best_word = original_word
    best_score = -float("inf")
    found = False

    for i, cand_emb in enumerate(cand_embs):
        sem_sim = cosine_similarity([cand_emb], [orig_emb])[0][0]
        if sem_sim < 0.50:
            continue
        style_score = float(np.dot(cand_emb, style_vec))
        final_score = (0.6 * style_score) + (0.4 * sem_sim)
        if final_score > best_score:
            best_score = final_score
            best_word = candidates[i]
            found = True

    if not found:
        return prompt_text, f"NO_BETTER_CANDIDATE({original_word})"

    words = prompt_text.split()
    words[word_idx] = best_word
    return " ".join(words), f"STYLE_INJECT[{pos_tag}]({original_word}->{best_word})"


if __name__ == "__main__":
    if BertTokenizer is None or BertForMaskedLM is None:
        raise ImportError("The mutation demo requires transformers.")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForMaskedLM.from_pretrained("bert-base-uncased")
    model.eval()
    tm = TemplateManager()
    sm = StyleManager()
    for sent, style in [
        ("Summarize why access control matters.", "imperative"),
        ("Can you explain the concept safely?", "plea"),
    ]:
        print("Original:", sent)
        print("Mutated:", hybrid_mutate_optimized(sent, style, tm, sm, tokenizer, model))
