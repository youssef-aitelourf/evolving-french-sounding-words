"""
problem.py — Modèle de langage Markovien d'ordre 2, dictionnaire Lexique383 et
             fonction objectif pour l'évolution de mots pseudo-français.

Adapté du code de démarrage fourni (yaceben/8INF852_TP02_Version_Gen_Starter).

Fournit :
  - Chargement/cache du dictionnaire Lexique383 (~150 000 mots)
  - Construction/cache du modèle trigramme caractère (ordre 2)
  - Calcul de perplexité (fidèle à gen_lm.perplexité du code de démarrage)
  - Fonction objectif (fitness) à MINIMISER
  - Évaluation parallèle de populations via ThreadPoolExecutor
"""
from __future__ import annotations

import pickle
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from zipfile import ZipFile

import nltk
import numpy as np
from nltk import ConditionalFreqDist
from nltk.lm.preprocessing import flatten, pad_both_ends

# ── Alphabet français autorisé ─────────────────────────────────────────────────
# Lettres minuscules + caractères accentués courants en français
ALPHABET: list[str] = list("abcdefghijklmnopqrstuvwxyzéèêëàâäîïôöùûüçæœ")
ALPHABET_SET: frozenset[str] = frozenset(ALPHABET)
ALPHABET_CHAR_TO_IDX: dict[str, int] = {c: i for i, c in enumerate(ALPHABET)}

MIN_LEN: int = 4
MAX_LEN: int = 16

# ── Chemins de cache ───────────────────────────────────────────────────────────
_HERE = Path(__file__).parent
CACHE_DIR = _HERE / "lexique"
DICT_CACHE = CACHE_DIR / "dictionary.pkl"
MODEL_CACHE = CACHE_DIR / "trigram_model.pkl"
LEXIQUE_URL = "http://www.lexique.org/databases/Lexique383/Lexique383.zip"

# Probabilité minimale (évite log2(0))
_LOG2_EPS: float = np.log2(1e-10)


# ─────────────────────────────────────────────────────────────────────────────
# Dictionnaire
# ─────────────────────────────────────────────────────────────────────────────

def _download_lexique(verbose: bool) -> list[str]:
    """Télécharge Lexique383 et retourne la liste triée de mots uniques."""
    import requests  # import local pour éviter la dépendance si déjà en cache

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tsv_path = CACHE_DIR / "Lexique383.tsv"
    zip_path = CACHE_DIR / "Lexique383.zip"

    if not tsv_path.exists():
        if verbose:
            print("Téléchargement de Lexique383 (peut prendre quelques secondes)...")
        resp = requests.get(LEXIQUE_URL, timeout=180)
        resp.raise_for_status()
        zip_path.write_bytes(resp.content)
        with ZipFile(zip_path) as zf:
            zf.extract("Lexique383.tsv", path=CACHE_DIR)
        if verbose:
            print("Lexique383 téléchargé et extrait.")

    words: set[str] = set()
    with open(tsv_path, encoding="utf-8") as fh:
        fh.readline()  # ignorer l'en-tête
        for line in fh:
            parts = line.split()
            if parts:
                w = parts[0].lower().strip()
                if w:
                    words.add(w)
    return sorted(words)


def load_dictionary(
    force_rebuild: bool = False,
    verbose: bool = True,
) -> tuple[list[str], set[str]]:
    """Charge le dictionnaire français depuis le cache ou Lexique383.

    Returns:
        (liste triée, ensemble pour lookup O(1))
    """
    if not force_rebuild and DICT_CACHE.exists():
        with open(DICT_CACHE, "rb") as f:
            words: list[str] = pickle.load(f)
        if verbose:
            print(f"Dictionnaire chargé depuis cache : {len(words):,} mots.")
        return words, set(words)

    words = _download_lexique(verbose)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(DICT_CACHE, "wb") as f:
        pickle.dump(words, f)
    if verbose:
        print(f"Dictionnaire sauvegardé : {len(words):,} mots.")
    return words, set(words)


# ─────────────────────────────────────────────────────────────────────────────
# Modèle de langage — trigramme caractère (ordre 2)
# ─────────────────────────────────────────────────────────────────────────────

def build_trigram_model(
    words: list[str],
    force_rebuild: bool = False,
    verbose: bool = True,
) -> ConditionalFreqDist:
    """Construit un modèle Markovien d'ordre 2 sur les caractères.

    Réplique fidèlement `build_trigram_model` du code de démarrage fourni :
      - chaque mot est converti en liste de chars
      - padding avec <s><s>..mot..</s> (pad_both_ends n=3)
      - tous les mots sont aplatis en une séquence unique (flatten)
      - les trigrammes sont extraits et stockés dans un ConditionalFreqDist

    P(m_i | m_{i-1}, m_{i-2}) estimé empiriquement sur Lexique383.
    """
    if not force_rebuild and MODEL_CACHE.exists():
        with open(MODEL_CACHE, "rb") as f:
            model = pickle.load(f)
        if verbose:
            print("Modèle de langage chargé depuis cache.")
        return model

    if verbose:
        print(f"Construction du modèle trigramme sur {len(words):,} mots...")

    corpus = [list(w) for w in words if w]
    # Aplatir toutes les séquences paddées (identique au code de démarrage)
    text = list(flatten(pad_both_ends(sent, n=3) for sent in corpus))

    model = ConditionalFreqDist()
    for ctx1, ctx2, char in nltk.trigrams(text):
        model[(ctx1, ctx2)][char] += 1

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODEL_CACHE, "wb") as f:
        pickle.dump(model, f)
    if verbose:
        print("Modèle construit et mis en cache.")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Perplexité
# ─────────────────────────────────────────────────────────────────────────────

def perplexity(word: str, model: ConditionalFreqDist) -> float:
    """Calcule PPL(M) selon le modèle Markovien d'ordre 2.

    Équation (3) de l'énoncé :
        PPL(M) = 2^( -1/n * Σ log2 P(m_i | m_{i-1}, m_{i-2}) )

    Réplique fidèlement `perplexité` du code de démarrage fourni.
    Plus la valeur est basse, plus le mot est « français ».
    """
    chars = list(pad_both_ends(list(word), n=3))
    log_sum = 0.0
    n = 0
    for ctx1, ctx2, char in nltk.trigrams(chars):
        freq = model[(ctx1, ctx2)].freq(char)
        log_sum += np.log2(freq) if freq > 0.0 else _LOG2_EPS
        n += 1
    if n == 0:
        return np.inf
    return 2.0 ** (-log_sum / n)


# ─────────────────────────────────────────────────────────────────────────────
# Pénalités
# ─────────────────────────────────────────────────────────────────────────────

def _count_invalid_chars(word: str) -> int:
    return sum(1 for c in word if c not in ALPHABET_SET)


def _max_consecutive(word: str) -> int:
    """Longueur maximale d'une répétition de caractère consécutif."""
    if not word:
        return 0
    best = count = 1
    for i in range(1, len(word)):
        count = count + 1 if word[i] == word[i - 1] else 1
        if count > best:
            best = count
    return best


# ─────────────────────────────────────────────────────────────────────────────
# Fonction objectif
# ─────────────────────────────────────────────────────────────────────────────

def fitness(
    word: str,
    model: ConditionalFreqDist,
    dictionary_set: set[str],
    *,
    dict_penalty: float = 50.0,
    repeat_penalty: float = 200.0,
    invalid_char_penalty: float = 500.0,
    length_penalty: float = 1000.0,
) -> float:
    """Fonction objectif à MINIMISER.

    Composition du score :
        perplexité(mot)                           — base, ↓ = plus français
        + dict_penalty     si le mot est dans le dictionnaire (pénalité douce)
        + repeat_penalty   si > 3 caractères consécutifs identiques
        + invalid_char_penalty * nb_chars_invalides
        + length_penalty   si longueur hors [MIN_LEN, MAX_LEN]
    """
    if not word:
        return length_penalty * 10.0

    # Caractères invalides → rejet immédiat (pas de perplexité calculée)
    n_inv = _count_invalid_chars(word)
    if n_inv > 0:
        return invalid_char_penalty * n_inv + length_penalty * 5.0

    score = 0.0

    # Pénalité longueur (douce : on calcule quand même la perplexité)
    if len(word) < MIN_LEN or len(word) > MAX_LEN:
        score += length_penalty

    # Perplexité — cœur de la fonction objective
    score += perplexity(word, model)

    # Pénalité dictionnaire (douce)
    if word.lower() in dictionary_set:
        score += dict_penalty

    # Pénalité répétitions excessives (> 3 consécutifs identiques)
    if _max_consecutive(word) > 3:
        score += repeat_penalty

    return score


# ─────────────────────────────────────────────────────────────────────────────
# Évaluation parallèle de population
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_population(
    population: list[str],
    model: ConditionalFreqDist,
    dictionary_set: set[str],
    dict_penalty: float = 50.0,
    repeat_penalty: float = 200.0,
    invalid_char_penalty: float = 500.0,
    length_penalty: float = 1000.0,
    n_jobs: int = 1,
) -> list[float]:
    """Évalue la fitness de toute une population.

    Args:
        population       : liste de mots candidats.
        model            : modèle trigramme (ConditionalFreqDist).
        dictionary_set   : ensemble des mots du dictionnaire (lookup O(1)).
        dict_penalty     : pénalité si mot existant.
        repeat_penalty   : pénalité pour répétitions excessives.
        invalid_char_penalty : pénalité par caractère invalide.
        length_penalty   : pénalité si longueur hors bornes.
        n_jobs           : 1=séquentiel, -1=tous cores, N>1=N threads.

    Returns:
        Liste de scores dans le même ordre que `population`.
    """
    fn = partial(
        fitness,
        model=model,
        dictionary_set=dictionary_set,
        dict_penalty=dict_penalty,
        repeat_penalty=repeat_penalty,
        invalid_char_penalty=invalid_char_penalty,
        length_penalty=length_penalty,
    )
    if n_jobs == 1 or len(population) < 20:
        return [fn(w) for w in population]
    workers = None if n_jobs <= 0 else n_jobs
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(fn, population))
