"""
algorithms.py — Algorithme Génétique (GA) et EDA/UMDA pour l'évolution de
                mots pseudo-français.

GA  : recombinaison explicite (crossover + mutation) avec mécanismes de
      diversité activables/désactivables (élitisme, étalon, losers, reseeds).
UMDA: Univariate Marginal Distribution Algorithm — modélisation probabiliste
      des distributions marginales par position de caractère.

Les deux algorithmes partagent la même interface (eval_fn, max_evals, rng) et
retournent un AlgorithmResult, ce qui garantit la comparabilité des expériences.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from problem import ALPHABET, ALPHABET_CHAR_TO_IDX, MIN_LEN, MAX_LEN

# Type alias pour la fonction d'évaluation
EvalFn = Callable[[list[str]], list[float]]

_N_ALPHA = len(ALPHABET)
_LENGTHS = list(range(MIN_LEN, MAX_LEN + 1))
_N_LENGTHS = len(_LENGTHS)


# ─────────────────────────────────────────────────────────────────────────────
# Structures de données
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GenerationRecord:
    """Snapshot d'une génération pour l'historique Monte-Carlo."""
    generation: int
    eval_count: int
    best_fitness: float
    best_word: str
    pop_best_fitness: float  # meilleur de la population courante
    pop_best_word: str       # mot ayant la meilleure fitness dans la génération courante
    diversity: float         # fraction de mots uniques dans la population


@dataclass
class AlgorithmResult:
    """Résultat complet d'une exécution algorithmique."""
    name: str
    best_word: str
    best_fitness: float
    history: list[GenerationRecord]
    # Suivi du meilleur mot (et sa fitness) à chaque génération
    best_word_history: list[tuple[str, float]] = field(default_factory=list)
    # Pool de tous les mots uniques rencontrés (toute la population par génération)
    # word → meilleure fitness observée ; sert à construire l'annexe des 500 mots
    word_pool: dict[str, float] = field(default_factory=dict)


def _diversity(population: list[str]) -> float:
    """Fraction de mots uniques dans la population (diversité génotypique)."""
    return len(set(population)) / max(len(population), 1)


# ─────────────────────────────────────────────────────────────────────────────
# Algorithme Génétique
# ─────────────────────────────────────────────────────────────────────────────

class GeneticAlgorithm:
    """Algorithme génétique pour l'évolution de mots pseudo-français.

    Paramètres configurables (tous en kwargs):
        population_size : int   — taille de la population
        use_elitism     : bool  — élitisme activé/désactivé
        n_elites        : int   — nombre d'élites conservées
        use_etalon      : bool  — logique de l'étalon activée/désactivée
        etalon_prob     : float — probabilité qu'un parent soit l'étalon
        use_losers      : bool  — conservation des losers activée/désactivée
        n_losers        : int   — nombre de losers conservés
        use_reseeds     : bool  — reseeds activés/désactivés
        n_reseeds       : int   — nombre d'individus réinitialisés par génération
        pm              : float — probabilité de mutation par caractère
        pc              : float — probabilité de croisement
        crossover_type  : str   — "single_point" ou "uniform"
        tournament_k    : int   — taille du tournoi de sélection
        max_evals       : int   — budget total d'évaluations de la fonction objectif
        verbosity       : int   — 0=silencieux, 1=résumé, 2=détaillé
        rng             : np.random.Generator | None

    Opérateurs de croisement :
        single_point : coupe en un point aléatoire, swap des suffixes.
                       Préserve les préfixes → exploite la structure phonotactique locale.
        uniform      : pour chaque position, caractère pris aléatoirement de l'un
                       ou l'autre parent. Plus disruptif → favorise l'exploration.
    """

    def __init__(
        self,
        eval_fn: EvalFn,
        *,
        population_size: int = 200,
        use_elitism: bool = True,
        n_elites: int = 5,
        use_etalon: bool = True,
        etalon_prob: float = 0.3,
        use_losers: bool = True,
        n_losers: int = 5,
        use_reseeds: bool = True,
        n_reseeds: int = 5,
        pm: float = 0.15,
        pc: float = 0.80,
        crossover_type: str = "single_point",
        tournament_k: int = 3,
        max_evals: int = 20_000,
        verbosity: int = 1,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.eval_fn = eval_fn
        self.population_size = population_size
        self.use_elitism = use_elitism
        self.n_elites = n_elites
        self.use_etalon = use_etalon
        self.etalon_prob = etalon_prob
        self.use_losers = use_losers
        self.n_losers = n_losers
        self.use_reseeds = use_reseeds
        self.n_reseeds = n_reseeds
        self.pm = pm
        self.pc = pc
        self.crossover_type = crossover_type
        self.tournament_k = tournament_k
        self.max_evals = max_evals
        self.verbosity = verbosity
        self.rng = rng if rng is not None else np.random.default_rng()

    # ── Initialisation ──────────────────────────────────────────────────────

    def _random_word(self) -> str:
        """Génère un mot aléatoire de longueur aléatoire dans [MIN_LEN, MAX_LEN].

        Chaque caractère est tiré uniformément de l'alphabet autorisé.
        La génération est indépendante du modèle de langage (conformément à l'énoncé).
        """
        length = int(self.rng.integers(MIN_LEN, MAX_LEN + 1))
        indices = self.rng.integers(0, _N_ALPHA, size=length)
        return "".join(ALPHABET[i] for i in indices)

    def _init_population(self) -> list[str]:
        return [self._random_word() for _ in range(self.population_size)]

    # ── Sélection ───────────────────────────────────────────────────────────

    def _tournament_select(
        self,
        population: list[str],
        fitnesses: list[float],
    ) -> str:
        """Sélection par tournoi (minimisation). Retourne l'individu gagnant."""
        n = len(population)
        k = min(self.tournament_k, n)
        indices = self.rng.choice(n, size=k, replace=False)
        best_idx = int(min(indices, key=lambda i: fitnesses[i]))
        return population[best_idx]

    # ── Croisements ─────────────────────────────────────────────────────────

    def _crossover_single_point(self, p1: str, p2: str) -> tuple[str, str]:
        """Croisement monopont adapté aux chaînes de longueur variable.

        Un point de coupure k ∈ [1, min(|p1|,|p2|)] est choisi uniformément.
        Enfant 1 = p1[:k] + p2[k:]   →  longueur = |p2|
        Enfant 2 = p2[:k] + p1[k:]   →  longueur = |p1|

        Propriété : les longueurs des parents sont préservées (échangées).
        Justification : préserve les structures phonotactiques de début de mot
        (préfixe), ce qui favorise l'exploitation locale. Un bon début de mot
        peut se propager efficacement à travers les générations.
        """
        n1, n2 = len(p1), len(p2)
        min_n = min(n1, n2)
        if min_n <= 1:
            return p1, p2
        k = int(self.rng.integers(1, min_n))
        c1 = self._clip(p1[:k] + p2[k:])
        c2 = self._clip(p2[:k] + p1[k:])
        return c1, c2

    def _crossover_uniform(self, p1: str, p2: str) -> tuple[str, str]:
        """Croisement uniforme adapté aux chaînes de longueur variable.

        La longueur de chaque enfant est interpolée entre |p1| et |p2|.
        À chaque position, le caractère est sélectionné aléatoirement (50/50)
        chez l'un ou l'autre parent ; si la position dépasse la longueur d'un
        parent, on prend le caractère de l'autre.

        Justification : brassage total de l'information génétique, aucune
        localité n'est favorisée. Plus exploratoire que le monopont, mais
        potentiellement destructeur pour les blocs fonctionnels. Utile en
        début d'optimisation pour diversifier rapidement la population.
        """
        n1, n2 = len(p1), len(p2)
        lo, hi = min(n1, n2), max(n1, n2)

        def _build(target_len: int) -> str:
            chars = []
            for i in range(target_len):
                has1, has2 = i < n1, i < n2
                if has1 and has2:
                    chars.append(p1[i] if self.rng.random() < 0.5 else p2[i])
                elif has1:
                    chars.append(p1[i])
                else:
                    chars.append(p2[i])
            return "".join(chars)

        len_c1 = int(self.rng.integers(lo, hi + 1))
        len_c2 = int(self.rng.integers(lo, hi + 1))
        return self._clip(_build(len_c1)), self._clip(_build(len_c2))

    def _crossover(self, p1: str, p2: str) -> tuple[str, str]:
        if self.crossover_type == "single_point":
            return self._crossover_single_point(p1, p2)
        return self._crossover_uniform(p1, p2)

    # ── Mutation ────────────────────────────────────────────────────────────

    def _mutate(self, word: str) -> str:
        """Mutation en deux phases :

        Phase 1 — substitution par position :
            Chaque caractère est remplacé par un caractère aléatoire de
            l'alphabet avec probabilité pm. Conserve la longueur.

        Phase 2 — mutation structurelle (longueur) :
            Avec probabilité pm/2 : insertion d'un caractère aléatoire.
            Avec probabilité pm/2 : suppression d'un caractère aléatoire.
            Les contraintes [MIN_LEN, MAX_LEN] sont respectées.
        """
        chars = [
            ALPHABET[int(self.rng.integers(_N_ALPHA))] if self.rng.random() < self.pm else c
            for c in word
        ]

        # Deux tirages indépendants : insertion et suppression sont indépendantes
        # (probabilité pm/2 chacune, non mutuellement exclusives)
        if self.rng.random() < self.pm / 2 and len(chars) < MAX_LEN:
            pos = int(self.rng.integers(0, len(chars) + 1))
            chars.insert(pos, ALPHABET[int(self.rng.integers(_N_ALPHA))])
        if self.rng.random() < self.pm / 2 and len(chars) > MIN_LEN:
            pos = int(self.rng.integers(0, len(chars)))
            chars.pop(pos)

        return "".join(chars)  # toujours dans [MIN_LEN, MAX_LEN] après mutation

    # ── Utilitaires ─────────────────────────────────────────────────────────

    @staticmethod
    def _clip(word: str) -> str:
        """Force la longueur dans [MIN_LEN, MAX_LEN] (troncature ou padding)."""
        if len(word) > MAX_LEN:
            return word[:MAX_LEN]
        while len(word) < MIN_LEN:
            word += ALPHABET[0]
        return word

    # ── Boucle principale ───────────────────────────────────────────────────

    def run(self) -> AlgorithmResult:
        """Exécute le GA jusqu'à épuisement du budget d'évaluations.

        Returns:
            AlgorithmResult avec historique complet par génération.
        """
        eval_count = 0
        history: list[GenerationRecord] = []
        best_word_history: list[tuple[str, float]] = []
        word_pool: dict[str, float] = {}  # {word: best_fitness_seen}

        # ── Population initiale ──────────────────────────────────────────
        population = self._init_population()
        fitnesses = self.eval_fn(population)
        eval_count += len(population)

        # Initialisation du pool avec tous les mots de la population initiale
        for w, f in zip(population, fitnesses):
            if w not in word_pool or f < word_pool[w]:
                word_pool[w] = f

        best_idx = int(np.argmin(fitnesses))
        best_fitness: float = fitnesses[best_idx]
        best_word: str = population[best_idx]
        etalon: str = best_word  # champion courant (logique de l'étalon)

        generation = 0
        while eval_count < self.max_evals:
            generation += 1

            # Tri par fitness croissante
            sorted_idx = sorted(range(len(population)), key=lambda i: fitnesses[i])

            # ── Élitisme ─────────────────────────────────────────────────
            n_el = self.n_elites if self.use_elitism else 0
            n_el = min(n_el, len(population))
            elites = [population[sorted_idx[i]] for i in range(n_el)]
            elite_fits = [fitnesses[sorted_idx[i]] for i in range(n_el)]

            # ── Losers ───────────────────────────────────────────────────
            n_lo = self.n_losers if self.use_losers else 0
            n_lo = min(n_lo, max(0, len(population) - n_el))
            losers = [population[sorted_idx[-(i + 1)]] for i in range(n_lo)]
            loser_fits = [fitnesses[sorted_idx[-(i + 1)]] for i in range(n_lo)]

            # ── Reseeds ──────────────────────────────────────────────────
            n_rs = self.n_reseeds if self.use_reseeds else 0
            reseeds = [self._random_word() for _ in range(n_rs)]

            # ── Enfants ──────────────────────────────────────────────────
            n_ch = self.population_size - n_el - n_lo - n_rs
            n_ch = max(1, n_ch)
            children: list[str] = []
            for _ in range(n_ch):
                p1 = self._tournament_select(population, fitnesses)

                # Logique de l'étalon : un parent peut être le champion
                if self.use_etalon and self.rng.random() < self.etalon_prob:
                    p2 = etalon
                else:
                    p2 = self._tournament_select(population, fitnesses)

                # Croisement (avec probabilité pc)
                if self.rng.random() < self.pc:
                    c1, c2 = self._crossover(p1, p2)
                    child = c1 if self.rng.random() < 0.5 else c2
                else:
                    child = p1 if self.rng.random() < 0.5 else p2

                children.append(self._mutate(child))

            # ── Évaluation des nouveaux individus uniquement ──────────────
            new_candidates = children + reseeds
            new_fits = self.eval_fn(new_candidates)
            eval_count += len(new_candidates)

            # ── Assemblage de la nouvelle population ──────────────────────
            population = elites + children + losers + reseeds
            fitnesses = (
                elite_fits
                + new_fits[:n_ch]
                + loser_fits
                + new_fits[n_ch:]
            )

            # ── Mise à jour du meilleur global ────────────────────────────
            gen_best_idx = int(np.argmin(fitnesses))
            gen_best_fit = fitnesses[gen_best_idx]
            if gen_best_fit < best_fitness:
                best_fitness = gen_best_fit
                best_word = population[gen_best_idx]
                if self.use_etalon:
                    etalon = best_word

            best_word_history.append((best_word, best_fitness))
            gen_best_word_str = population[gen_best_idx]

            # ── Mise à jour du pool (tous les mots uniques de la génération) ──
            for w, f in zip(population, fitnesses):
                if w not in word_pool or f < word_pool[w]:
                    word_pool[w] = f

            rec = GenerationRecord(
                generation=generation,
                eval_count=eval_count,
                best_fitness=best_fitness,
                best_word=best_word,
                pop_best_fitness=gen_best_fit,
                pop_best_word=gen_best_word_str,
                diversity=_diversity(population),
            )
            history.append(rec)

            if self.verbosity >= 2 and generation % 10 == 0:
                print(
                    f"    Gen {generation:4d} | evals {eval_count:6d} | "
                    f"best={best_fitness:.2f} ({best_word!r}) | "
                    f"div={rec.diversity:.2f}"
                )

        if self.verbosity >= 1:
            print(
                f"  GA ({self.crossover_type}) terminé : {eval_count} évaluations | "
                f"meilleur={best_word!r} (fit={best_fitness:.4f})"
            )

        return AlgorithmResult(
            name=f"GA_{self.crossover_type}",
            best_word=best_word,
            best_fitness=best_fitness,
            history=history,
            best_word_history=best_word_history,
            word_pool=word_pool,
        )


# ─────────────────────────────────────────────────────────────────────────────
# EDA — UMDA (Univariate Marginal Distribution Algorithm)
# ─────────────────────────────────────────────────────────────────────────────

class UMDA:
    """Estimation of Distribution Algorithm : UMDA discret pour chaînes de
    longueur variable.

    À chaque génération :
      1. Sélection des `n_selection` meilleurs individus (par fitness).
      2. Estimation des distributions marginales :
           - P(longueur)      : distribution empirique + lissage de Laplace
           - P(char | pos=i)  : distribution empirique + lissage de Laplace
             (calculée sur les individus de longueur > i)
      3. Échantillonnage d'une nouvelle population complète.
      4. Évaluation et recommencement.

    Aucun opérateur de croisement ou de mutation explicite n'est utilisé.
    La diversité est maintenue via le lissage de Laplace et la variabilité
    naturelle de l'échantillonnage.

    Paramètres configurables :
        population_size  : int   — taille de la population
        selection_ratio  : float — fraction de la population utilisée pour
                                   l'estimation (ex. 0.5 = top 50%)
        smoothing        : float — paramètre α du lissage de Laplace pour
                                   les distributions de caractères
        length_smoothing : float — paramètre α pour la distribution de longueur
        n_elites         : int   — nombre de meilleurs individus préservés entre
                                   générations (mémoire inter-générations). Réduit
                                   la variance et prévient la perte de la meilleure
                                   solution. 0 = comportement UMDA classique.
        max_evals        : int   — budget total d'évaluations
        verbosity        : int   — 0, 1 ou 2
        rng              : np.random.Generator | None
    """

    def __init__(
        self,
        eval_fn: EvalFn,
        *,
        population_size: int = 200,
        selection_ratio: float = 0.5,
        smoothing: float = 0.01,
        length_smoothing: float = 0.01,
        n_elites: int = 1,
        max_evals: int = 20_000,
        verbosity: int = 1,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.eval_fn = eval_fn
        self.population_size = population_size
        self.selection_ratio = selection_ratio
        self.smoothing = smoothing
        self.length_smoothing = length_smoothing
        self.n_elites = n_elites
        self.max_evals = max_evals
        self.verbosity = verbosity
        self.rng = rng if rng is not None else np.random.default_rng()

    def _random_word(self) -> str:
        length = int(self.rng.integers(MIN_LEN, MAX_LEN + 1))
        indices = self.rng.integers(0, _N_ALPHA, size=length)
        return "".join(ALPHABET[i] for i in indices)

    def _init_population(self) -> list[str]:
        return [self._random_word() for _ in range(self.population_size)]

    def _estimate_distributions(
        self,
        selected: list[str],
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        """Estime les distributions marginales à partir des individus sélectionnés.

        Returns:
            length_dist : ndarray shape (N_LENGTHS,) — P(longueur) avec lissage
            pos_dists   : liste de MAX_LEN ndarrays shape (N_ALPHA,) — P(char|pos)
        """
        n = len(selected)
        alpha_l = self.length_smoothing
        alpha_c = self.smoothing

        # Distribution de longueur
        len_counts = np.full(_N_LENGTHS, alpha_l)
        for w in selected:
            lw = len(w)
            if MIN_LEN <= lw <= MAX_LEN:
                len_counts[lw - MIN_LEN] += 1.0
        length_dist = len_counts / len_counts.sum()

        # Distribution de caractère par position
        pos_dists: list[np.ndarray] = []
        for pos in range(MAX_LEN):
            char_counts = np.full(_N_ALPHA, alpha_c)
            for w in selected:
                if pos < len(w):
                    c = w[pos]
                    if c in ALPHABET_CHAR_TO_IDX:
                        char_counts[ALPHABET_CHAR_TO_IDX[c]] += 1.0
            pos_dists.append(char_counts / char_counts.sum())

        return length_dist, pos_dists

    def _sample(
        self,
        length_dist: np.ndarray,
        pos_dists: list[np.ndarray],
    ) -> str:
        """Échantillonne un nouveau mot selon les distributions estimées."""
        l = int(self.rng.choice(_N_LENGTHS, p=length_dist)) + MIN_LEN
        chars = [
            ALPHABET[int(self.rng.choice(_N_ALPHA, p=pos_dists[pos]))]
            for pos in range(l)
        ]
        return "".join(chars)

    def run(self) -> AlgorithmResult:
        """Exécute l'UMDA jusqu'à épuisement du budget d'évaluations.

        Returns:
            AlgorithmResult avec historique complet par génération.
        """
        eval_count = 0
        history: list[GenerationRecord] = []
        best_word_history: list[tuple[str, float]] = []
        word_pool: dict[str, float] = {}

        # ── Population initiale ──────────────────────────────────────────
        population = self._init_population()
        fitnesses = self.eval_fn(population)
        eval_count += len(population)

        for w, f in zip(population, fitnesses):
            if w not in word_pool or f < word_pool[w]:
                word_pool[w] = f

        best_idx = int(np.argmin(fitnesses))
        best_fitness: float = fitnesses[best_idx]
        best_word: str = population[best_idx]

        generation = 0
        while eval_count < self.max_evals:
            generation += 1

            # ── Sélection des meilleurs ───────────────────────────────────
            n_sel = max(2, int(self.population_size * self.selection_ratio))
            sorted_idx = sorted(range(len(population)), key=lambda i: fitnesses[i])
            selected = [population[i] for i in sorted_idx[:n_sel]]

            # ── Estimation des distributions marginales ───────────────────
            length_dist, pos_dists = self._estimate_distributions(selected)

            # ── Élitisme : préservation des meilleurs individus ───────────
            n_el = min(self.n_elites, len(population))
            elites = [population[i] for i in sorted_idx[:n_el]]
            elite_fits = [fitnesses[i] for i in sorted_idx[:n_el]]

            # ── Échantillonnage du reste de la population ─────────────────
            n_sampled = self.population_size - n_el
            new_population = [
                self._sample(length_dist, pos_dists)
                for _ in range(n_sampled)
            ]

            # ── Évaluation (seuls les individus échantillonnés sont nouveaux)
            new_fitnesses = self.eval_fn(new_population)
            eval_count += len(new_population)

            population = elites + new_population
            fitnesses  = elite_fits + new_fitnesses

            # ── Mise à jour du meilleur global ────────────────────────────
            gen_best_idx = int(np.argmin(fitnesses))
            gen_best_fit = fitnesses[gen_best_idx]
            if gen_best_fit < best_fitness:
                best_fitness = gen_best_fit
                best_word = population[gen_best_idx]

            best_word_history.append((best_word, best_fitness))
            gen_best_word_str = population[gen_best_idx]

            # Mise à jour du pool (tous les mots uniques de la génération)
            for w, f in zip(population, fitnesses):
                if w not in word_pool or f < word_pool[w]:
                    word_pool[w] = f

            rec = GenerationRecord(
                generation=generation,
                eval_count=eval_count,
                best_fitness=best_fitness,
                best_word=best_word,
                pop_best_fitness=gen_best_fit,
                pop_best_word=gen_best_word_str,
                diversity=_diversity(population),
            )
            history.append(rec)

            if self.verbosity >= 2 and generation % 10 == 0:
                print(
                    f"    Gen {generation:4d} | evals {eval_count:6d} | "
                    f"best={best_fitness:.2f} ({best_word!r}) | "
                    f"div={rec.diversity:.2f}"
                )

        if self.verbosity >= 1:
            print(
                f"  UMDA terminé : {eval_count} évaluations | "
                f"meilleur={best_word!r} (fit={best_fitness:.4f})"
            )

        return AlgorithmResult(
            name="UMDA",
            best_word=best_word,
            best_fitness=best_fitness,
            history=history,
            best_word_history=best_word_history,
            word_pool=word_pool,
        )
