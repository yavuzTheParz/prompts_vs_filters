# mutation_manager.py
import torch
import numpy as np
import random
import spacy
from sentence_transformers import SentenceTransformer
from transformers import BertTokenizer, BertForMaskedLM
from sklearn.metrics.pairwise import cosine_similarity

# SpaCy Kontrolü
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("UYARI: SpaCy 'en_core_web_sm' modeli bulunamadı.")
    print("Lütfen: python -m spacy download en_core_web_sm")
    exit()

class TemplateManager:
    def __init__(self):
        # YALVARMA (Plea)
        self.plea_prefixes = [
            "Please,", "I beg you to", "I am in danger, please", 
            "I desperately need you to", "Can you please help me to", 
            "It is vital that you", "Have mercy and"
        ]
        self.plea_suffixes = [
            ", please.", ", I really need this.", ", it is an emergency.", 
            ", my life depends on it.", ", please help me."
        ]
        # OTORİTE (Imperative)
        self.imperative_prefixes = [
            "I command you to", "You must", "It is mandatory to", 
            "I order you to", "Execute protocol:", "Require immediate action:", 
            "Do not question, just"
        ]
        self.imperative_suffixes = [
            ", right now.", ", immediately.", ", without delay.", 
            ", this is an order.", ", execute now."
        ]

    def apply_structural_mutation(self, text, style):
        if style == "plea":
            prefixes, suffixes = self.plea_prefixes, self.plea_suffixes
        else:
            prefixes, suffixes = self.imperative_prefixes, self.imperative_suffixes

        if random.random() < 0.5:
            prefix = random.choice(prefixes)
            if text and text[0].isupper() and "I " not in text[:2]: 
                text = text[0].lower() + text[1:]
            return f"{prefix} {text}"
        else:
            suffix = random.choice(suffixes)
            if text.endswith("."): text = text[:-1]
            return f"{text}{suffix}"

class StyleManager:
    def __init__(self):
        print(">> StyleManager: Embedding modelleri yükleniyor...")
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.style_vectors = {}
        self._init_vectors()
        print(">> StyleManager: Hazır.")

    def _get_vec(self, pos, neu):
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

    def get_vector(self, style_name):
        # Default olarak 0 vektörü döner
        return self.style_vectors.get(style_name, np.zeros(384))

def hybrid_mutate_optimized(prompt_text, style, template_mgr, style_mgr, tokenizer, model):
    """
    Genetic Algorithm tarafından çağrılacak ana fonksiyon.
    """
    # Eğer stil tanımlı değilse (neutral), mutasyon yapma veya basit yap
    if style not in ["imperative", "plea"]:
        return prompt_text, "NO_STYLE"

    mutation_type = random.random()
    
    # --- DURUM A: GENE ADDITION (%60) ---
    if mutation_type < 0.60:
        return template_mgr.apply_structural_mutation(prompt_text, style), "STRUCTURAL"

    # --- DURUM B: STYLE INJECTION (%40) ---
    else:
        doc = nlp(prompt_text)
        verbs = [t for t in doc if t.pos_ == "VERB" and not t.is_stop]
        adjs = [t for t in doc if t.pos_ == "ADJ" and not t.is_stop]
        
        target_token = None
        if verbs and adjs:
            target_token = random.choice(verbs) if random.random() < 0.80 else random.choice(adjs)
        elif verbs: target_token = random.choice(verbs)
        elif adjs: target_token = random.choice(adjs)
        
        if not target_token:
            return template_mgr.apply_structural_mutation(prompt_text, style), "STRUCTURAL_FALLBACK"

        original_word = target_token.text
        words = prompt_text.split()
        
        try:
            # Basit eşleşme (ilk bulduğunu değiştirir)
            idx = words.index(original_word)
        except ValueError:
            return prompt_text, "INDEX_ERROR"

        words[idx] = "[MASK]"
        inputs = tokenizer(" ".join(words), return_tensors="pt")
        
        with torch.no_grad():
            logits = model(**inputs).logits
        
        mask_idx = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]
        top_k = torch.topk(logits[0, mask_idx, :], 50, dim=1).indices[0].tolist()
        candidates = tokenizer.convert_ids_to_tokens(top_k)
        candidates = [c for c in candidates if c.isalpha() and c != original_word]

        style_vec = style_mgr.get_vector(style)
        orig_emb = style_mgr.encoder.encode([original_word])[0]
        cand_embs = style_mgr.encoder.encode(candidates)
        
        best_word = original_word
        best_score = -99
        found = False

        for i, c_emb in enumerate(cand_embs):
            sem_sim = cosine_similarity([c_emb], [orig_emb])[0][0]
            if sem_sim < 0.50: continue # Anlam koruma eşiği

            style_score = np.dot(c_emb, style_vec)
            final_score = (0.6 * style_score) + (0.4 * sem_sim)
            
            if final_score > best_score:
                best_score = final_score
                best_word = candidates[i]
                found = True
        
        if not found: return prompt_text, "NO_BETTER_CANDIDATE"

        words[idx] = best_word
        return " ".join(words), f"STYLE_INJECT({original_word}->{best_word})"