"""
experiments.py — Protocole expérimental Monte-Carlo.

Fournit :
  - run_monte_carlo() : exécution multi-seeds d'un algorithme
  - Quatre expériences du barème :
      exp_crossover_comparison()  — i.   Comparaison de deux croisements
      exp_etalon_comparison()     — ii.  Effet de la logique de l'étalon
      exp_losers_comparison()     — iii. Effet de la conservation des losers
      exp_ga_vs_eda()             — iv.  Meilleur GA vs UMDA
  - save_experiment_outputs()     : persistance CSV + pickle

Garanties de reproductibilité :
  - Générateur PCG64 (plus fort que le MT19937 par défaut)
  - Seed sauvegardée dans chaque ligne du DataFrame résultat
  - Budget d'évaluations contrôlé par max_evals (±10% de tolérance)
"""
from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from algorithms import AlgorithmResult, EvalFn, GeneticAlgorithm, UMDA


# ─────────────────────────────────────────────────────────────────────────────
# Noyau Monte-Carlo
# ─────────────────────────────────────────────────────────────────────────────

def run_monte_carlo(
    make_algo: Callable[[np.random.Generator], AlgorithmResult],
    seeds: list[int],
    label: str,
    verbosity: int = 1,
    shared_pool: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Lance `len(seeds)` exécutions indépendantes d'un algorithme.

    Chaque exécution utilise un générateur PCG64 initialisé avec son seed
    propre, garantissant la reproductibilité et l'indépendance des runs.

    Args:
        make_algo    : callable(rng) → AlgorithmResult — fabrique et exécute l'algo
        seeds        : liste de seeds aléatoires (un par run)
        label        : étiquette de l'algorithme dans le DataFrame
        verbosity    : niveau de verbosité
        shared_pool  : si fourni, accumule les word_pool de tous les runs (muté en place)

    Returns:
        DataFrame avec colonnes :
            run, seed, algo, generation, eval_count, best_fitness, best_word,
            pop_best_fitness, pop_best_word, diversity
    """
    rows: list[dict] = []

    for run_no, seed in enumerate(seeds):
        # PCG64 : générateur cryptographiquement plus fort que MT19937
        rng = np.random.Generator(np.random.PCG64(seed))
        if verbosity >= 1:
            print(f"  Run {run_no + 1:2d}/{len(seeds)} (seed={seed:10d})...")

        result = make_algo(rng)

        # Accumulation du word_pool inter-runs
        if shared_pool is not None:
            for w, f in result.word_pool.items():
                if w not in shared_pool or f < shared_pool[w]:
                    shared_pool[w] = f

        for rec in result.history:
            rows.append(
                {
                    "run": run_no,
                    "seed": seed,
                    "algo": label,
                    "generation": rec.generation,
                    "eval_count": rec.eval_count,
                    "best_fitness": rec.best_fitness,
                    "best_word": rec.best_word,
                    "pop_best_fitness": rec.pop_best_fitness,
                    "pop_best_word": rec.pop_best_word,
                    "diversity": rec.diversity,
                }
            )

    return pd.DataFrame(rows)


def collect_best_words(
    dfs: list[pd.DataFrame],
    dictionary_set: set[str],
    top_n: int = 500,
) -> pd.DataFrame:
    """Collecte les meilleurs mots uniques produits sur tous les runs.

    Retourne un DataFrame trié par fitness (ascendant) avec les top_n mots
    qui ne sont pas dans le dictionnaire.
    """
    combined = pd.concat(dfs, ignore_index=True)
    # La première occurrence de chaque (algo, best_word) correspond au moment
    # où ce mot est devenu le global best → best_fitness = sa vraie fitness
    word_df = (
        combined.sort_values("eval_count")
        .drop_duplicates(subset=["algo", "best_word"])
        [["algo", "best_word", "best_fitness"]]
        .copy()
    )
    word_df = word_df[~word_df["best_word"].isin(dictionary_set)]
    word_df = word_df.sort_values("best_fitness").reset_index(drop=True)
    return word_df.head(top_n)


# ─────────────────────────────────────────────────────────────────────────────
# Fabrique d'algorithmes
# ─────────────────────────────────────────────────────────────────────────────

def _make_ga_factory(eval_fn: EvalFn, ga_cfg: dict, **overrides) -> Callable:
    """Retourne une factory callable(rng) → AlgorithmResult pour le GA."""
    params = {
        "population_size": ga_cfg["population_size"],
        "use_elitism": ga_cfg.get("use_elitism", True),
        "n_elites": ga_cfg["n_elites"],
        "use_etalon": ga_cfg.get("use_etalon", True),
        "etalon_prob": ga_cfg.get("etalon_prob", 0.3),
        "use_losers": ga_cfg.get("use_losers", True),
        "n_losers": ga_cfg["n_losers"],
        "use_reseeds": ga_cfg.get("use_reseeds", True),
        "n_reseeds": ga_cfg["n_reseeds"],
        "pm": ga_cfg["pm"],
        "pc": ga_cfg["pc"],
        "crossover_type": ga_cfg.get("crossover", "single_point"),
        "tournament_k": ga_cfg.get("tournament_k", 3),
        "max_evals": ga_cfg["max_evals"],
        "verbosity": 0,  # verbosité interne muette (gérée par run_monte_carlo)
    }
    params.update(overrides)

    def factory(rng: np.random.Generator) -> AlgorithmResult:
        ga = GeneticAlgorithm(eval_fn, **params, rng=rng)
        return ga.run()

    return factory


def _make_eda_factory(eval_fn: EvalFn, eda_cfg: dict, **overrides) -> Callable:
    """Retourne une factory callable(rng) → AlgorithmResult pour l'UMDA."""
    params = {
        "population_size": eda_cfg["population_size"],
        "selection_ratio": eda_cfg["selection_ratio"],
        "smoothing": eda_cfg["smoothing"],
        "length_smoothing": eda_cfg.get("length_smoothing", eda_cfg["smoothing"]),
        "n_elites": eda_cfg.get("n_elites", 1),
        "max_evals": eda_cfg["max_evals"],
        "verbosity": 0,
    }
    params.update(overrides)

    def factory(rng: np.random.Generator) -> AlgorithmResult:
        eda = UMDA(eval_fn, **params, rng=rng)
        return eda.run()

    return factory


# ─────────────────────────────────────────────────────────────────────────────
# Expériences
# ─────────────────────────────────────────────────────────────────────────────

def experiment_crossover_comparison(
    eval_fn: EvalFn,
    config: dict,
    verbosity: int = 1,
) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, float]]]:
    """Expérience i — Comparaison de deux opérateurs de croisement.

    GA avec croisement monopont (single_point) vs croisement uniforme (uniform).
    Tous les autres paramètres sont identiques. Mêmes seeds.

    Budget comparable : même max_evals pour les deux configurations.
    """
    ga_cfg = config["ga"]
    seeds = config["general"]["random_seeds"]
    n_runs = config["experiments"]["n_runs"]
    run_seeds = seeds[:n_runs]

    if verbosity >= 1:
        print("\n[Expérience i] Comparaison des croisements : single_point vs uniform")

    results: dict[str, pd.DataFrame] = {}
    pools: dict[str, dict[str, float]] = {}

    for cx_type in ("single_point", "uniform"):
        label = f"GA_{cx_type}"
        if verbosity >= 1:
            print(f"\n  Croisement : {cx_type}")
        pool: dict[str, float] = {}
        factory = _make_ga_factory(eval_fn, ga_cfg, crossover_type=cx_type)
        results[label] = run_monte_carlo(factory, run_seeds, label, verbosity, shared_pool=pool)
        pools[label] = pool

    return results, pools


def experiment_etalon_comparison(
    eval_fn: EvalFn,
    config: dict,
    verbosity: int = 1,
) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, float]]]:
    """Expérience ii — Effet de la logique de l'étalon.

    GA avec étalon activé vs désactivé. Croisement fixé à single_point.
    """
    ga_cfg = config["ga"]
    seeds = config["general"]["random_seeds"]
    n_runs = config["experiments"]["n_runs"]
    run_seeds = seeds[:n_runs]

    if verbosity >= 1:
        print("\n[Expérience ii] Effet de l'étalon : avec vs sans")

    results: dict[str, pd.DataFrame] = {}
    pools: dict[str, dict[str, float]] = {}
    for use_etalon in (True, False):
        label = "GA_avec_etalon" if use_etalon else "GA_sans_etalon"
        if verbosity >= 1:
            print(f"\n  Étalon : {use_etalon}")
        pool: dict[str, float] = {}
        factory = _make_ga_factory(
            eval_fn, ga_cfg,
            crossover_type="single_point",
            use_etalon=use_etalon,
        )
        results[label] = run_monte_carlo(factory, run_seeds, label, verbosity, shared_pool=pool)
        pools[label] = pool

    return results, pools


def experiment_losers_comparison(
    eval_fn: EvalFn,
    config: dict,
    verbosity: int = 1,
) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, float]]]:
    """Expérience iii — Effet de la conservation des losers.

    GA avec losers vs sans losers. Croisement single_point, étalon activé.
    """
    ga_cfg = config["ga"]
    seeds = config["general"]["random_seeds"]
    n_runs = config["experiments"]["n_runs"]
    run_seeds = seeds[:n_runs]

    if verbosity >= 1:
        print("\n[Expérience iii] Effet des losers : avec vs sans")

    results: dict[str, pd.DataFrame] = {}
    pools: dict[str, dict[str, float]] = {}
    for use_losers in (True, False):
        label = "GA_avec_losers" if use_losers else "GA_sans_losers"
        if verbosity >= 1:
            print(f"\n  Losers : {use_losers}")
        pool: dict[str, float] = {}
        factory = _make_ga_factory(
            eval_fn, ga_cfg,
            crossover_type="single_point",
            use_etalon=True,
            use_losers=use_losers,
            n_losers=ga_cfg["n_losers"],
        )
        results[label] = run_monte_carlo(factory, run_seeds, label, verbosity, shared_pool=pool)
        pools[label] = pool

    return results, pools


def experiment_ga_vs_eda(
    eval_fn: EvalFn,
    config: dict,
    verbosity: int = 1,
) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, float]]]:
    """Expérience iv — Meilleur GA (single_point + étalon + losers) vs UMDA.

    Budget comparable : même max_evals. Les différences de budget réelles sont
    documentées dans le rapport (tolérance ±10% garantie par construction).
    """
    ga_cfg = config["ga"]
    eda_cfg = config["eda"]
    seeds = config["general"]["random_seeds"]
    n_runs = config["experiments"]["n_runs"]
    run_seeds = seeds[:n_runs]

    if verbosity >= 1:
        print("\n[Expérience iv] GA (meilleur config) vs UMDA")

    results: dict[str, pd.DataFrame] = {}
    pools: dict[str, dict[str, float]] = {}

    # Meilleur GA : single_point + étalon + losers
    if verbosity >= 1:
        print("\n  Algorithme : GA (single_point + étalon + losers)")
    pool_ga: dict[str, float] = {}
    ga_factory = _make_ga_factory(
        eval_fn, ga_cfg,
        crossover_type="single_point",
        use_etalon=True,
        use_losers=True,
    )
    results["GA_best"] = run_monte_carlo(ga_factory, run_seeds, "GA_best", verbosity, shared_pool=pool_ga)
    pools["GA_best"] = pool_ga

    # UMDA
    if verbosity >= 1:
        print("\n  Algorithme : UMDA")
    pool_eda: dict[str, float] = {}
    eda_factory = _make_eda_factory(eval_fn, eda_cfg)
    results["UMDA"] = run_monte_carlo(eda_factory, run_seeds, "UMDA", verbosity, shared_pool=pool_eda)
    pools["UMDA"] = pool_eda

    return results, pools


# ─────────────────────────────────────────────────────────────────────────────
# Persistance
# ─────────────────────────────────────────────────────────────────────────────

def save_experiment_outputs(
    dfs: dict[str, pd.DataFrame],
    output_dir: Path,
    experiment_name: str,
) -> dict[str, Path]:
    """Sauvegarde un ensemble de DataFrames en CSV et pickle.

    Returns:
        Dictionnaire {format: chemin} des fichiers sauvegardés.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    combined = pd.concat(list(dfs.values()), ignore_index=True)
    paths: dict[str, Path] = {}

    csv_path = output_dir / f"{experiment_name}.csv"
    pkl_path = output_dir / f"{experiment_name}.pkl"
    combined.to_csv(csv_path, index=False, encoding="utf-8")
    combined.to_pickle(pkl_path)

    paths["csv"] = csv_path
    paths["pkl"] = pkl_path
    return paths
