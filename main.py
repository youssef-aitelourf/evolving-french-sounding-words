"""
main.py — Point d'entrée CLI du TP2 : évolution discrète de mots pseudo-français.

Usage :
    python main.py                              # utilise config.yaml, mode=all
    python main.py --mode exp_crossover         # expérience i uniquement
    python main.py --mode exp_etalon            # expérience ii uniquement
    python main.py --mode exp_losers            # expérience iii uniquement
    python main.py --mode exp_ga_vs_eda         # expérience iv uniquement
    python main.py --mode run_ga                # une exécution GA (démo)
    python main.py --mode run_eda               # une exécution UMDA (démo)
    python main.py --config my_config.yaml      # config personnalisée
    python main.py --force-rebuild              # reconstruire dict + modèle

Tous les paramètres sont configurables dans config.yaml sans modifier le code.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np
import yaml

from analysis import run_full_analysis, save_best_words_annex
from algorithms import GeneticAlgorithm, UMDA
from experiments import (
    experiment_crossover_comparison,
    experiment_etalon_comparison,
    experiment_ga_vs_eda,
    experiment_losers_comparison,
    save_experiment_outputs,
)
from problem import build_trigram_model, evaluate_population, load_dictionary


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULT_CONFIG = Path(__file__).parent / "config.yaml"


def load_config(path: Path) -> dict:
    """Charge la configuration YAML."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _build_eval_fn(model, dictionary_set: set[str], config: dict):
    """Construit la fonction d'évaluation à partir de la config."""
    prob_cfg = config["problem"]
    return partial(
        evaluate_population,
        model=model,
        dictionary_set=dictionary_set,
        dict_penalty=prob_cfg.get("dict_penalty", 50.0),
        repeat_penalty=prob_cfg.get("repeat_penalty", 200.0),
        invalid_char_penalty=prob_cfg.get("invalid_char_penalty", 500.0),
        length_penalty=prob_cfg.get("length_penalty", 1000.0),
        n_jobs=prob_cfg.get("n_jobs", 1),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Modes
# ─────────────────────────────────────────────────────────────────────────────

def mode_run_ga(config: dict, eval_fn, dictionary_set: set[str], verbosity: int) -> None:
    """Exécution unique du GA (mode démonstration)."""
    print("\n── Exécution GA (démonstration) ─────────────────────────────────────")
    ga_cfg = config["ga"]
    seed = config["general"]["random_seeds"][0]
    rng = np.random.Generator(np.random.PCG64(seed))
    ga = GeneticAlgorithm(
        eval_fn,
        population_size=ga_cfg["population_size"],
        use_elitism=ga_cfg.get("use_elitism", True),
        n_elites=ga_cfg["n_elites"],
        use_etalon=ga_cfg.get("use_etalon", True),
        etalon_prob=ga_cfg.get("etalon_prob", 0.3),
        use_losers=ga_cfg.get("use_losers", True),
        n_losers=ga_cfg["n_losers"],
        use_reseeds=ga_cfg.get("use_reseeds", True),
        n_reseeds=ga_cfg["n_reseeds"],
        pm=ga_cfg["pm"],
        pc=ga_cfg["pc"],
        crossover_type=ga_cfg.get("crossover", "single_point"),
        tournament_k=ga_cfg.get("tournament_k", 3),
        max_evals=ga_cfg["max_evals"],
        verbosity=verbosity,
        rng=rng,
    )
    result = ga.run()
    print(f"\nMeilleur mot  : {result.best_word!r}")
    print(f"Fitness       : {result.best_fitness:.4f}")
    print(f"Générations   : {len(result.history)}")
    print(f"Dans dict     : {result.best_word in dictionary_set}")


def mode_run_eda(config: dict, eval_fn, dictionary_set: set[str], verbosity: int) -> None:
    """Exécution unique de l'UMDA (mode démonstration)."""
    print("\n── Exécution UMDA (démonstration) ───────────────────────────────────")
    eda_cfg = config["eda"]
    seed = config["general"]["random_seeds"][0]
    rng = np.random.Generator(np.random.PCG64(seed))
    eda = UMDA(
        eval_fn,
        population_size=eda_cfg["population_size"],
        selection_ratio=eda_cfg["selection_ratio"],
        smoothing=eda_cfg["smoothing"],
        length_smoothing=eda_cfg.get("length_smoothing", eda_cfg["smoothing"]),
        max_evals=eda_cfg["max_evals"],
        verbosity=verbosity,
        rng=rng,
    )
    result = eda.run()
    print(f"\nMeilleur mot  : {result.best_word!r}")
    print(f"Fitness       : {result.best_fitness:.4f}")
    print(f"Générations   : {len(result.history)}")
    print(f"Dans dict     : {result.best_word in dictionary_set}")


def _run_and_save_experiment(
    name: str,
    dfs: dict,
    pools: dict,
    config: dict,
    dictionary_set: set[str],
    output_dir: Path,
    verbosity: int,
) -> None:
    """Sauvegarde les données et génère les analyses d'une expérience."""
    exp_dir = output_dir / name

    # Persistance
    paths = save_experiment_outputs(dfs, exp_dir, name)
    if verbosity >= 1:
        print(f"  Données sauvegardées : {paths['csv']}")

    # Analyses
    plot_paths = run_full_analysis(dfs, name, exp_dir)
    if verbosity >= 1:
        for k, p in plot_paths.items():
            print(f"  [{k}] → {p}")

    # Annexe (mots générés via word_pool)
    annex_paths = save_best_words_annex(pools, dictionary_set, exp_dir)
    if verbosity >= 1:
        for algo, p in annex_paths.items():
            print(f"  [annexe {algo}] → {p}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TP2 — 8INF852 : Évolution discrète de mots pseudo-français",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=_DEFAULT_CONFIG,
        help="Chemin vers le fichier de configuration YAML",
    )
    parser.add_argument(
        "--mode",
        choices=[
            "run_ga",
            "run_eda",
            "exp_crossover",
            "exp_etalon",
            "exp_losers",
            "exp_ga_vs_eda",
            "all",
        ],
        default="all",
        help="Mode d'exécution",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Répertoire de sortie (écrase config.yaml si spécifié)",
    )
    parser.add_argument(
        "--verbosity",
        type=int,
        default=None,
        choices=[0, 1, 2],
        help="Niveau de verbosité (écrase config.yaml si spécifié)",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Reconstruire le dictionnaire et le modèle même si le cache existe",
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    # Surcharges CLI
    if args.output_dir is not None:
        config["general"]["output_dir"] = str(args.output_dir)
    if args.verbosity is not None:
        config["general"]["verbosity"] = args.verbosity

    verbosity: int = config["general"].get("verbosity", 1)
    output_dir = Path(config["general"].get("output_dir", "results"))
    output_dir.mkdir(parents=True, exist_ok=True)

    if verbosity >= 1:
        print("=" * 60)
        print("  TP2 — 8INF852 : Évolution discrète de mots pseudo-français")
        print("=" * 60)
        print(f"  Mode        : {args.mode}")
        print(f"  Config      : {args.config}")
        print(f"  Sortie      : {output_dir}")
        print()

    # ── Chargement du dictionnaire et du modèle ──────────────────────────
    t0 = time.time()
    force = args.force_rebuild

    if verbosity >= 1:
        print("[1/3] Chargement du dictionnaire...")
    words, dictionary_set = load_dictionary(force_rebuild=force, verbose=verbosity >= 1)

    if verbosity >= 1:
        print("[2/3] Chargement/construction du modèle de langage...")
    model = build_trigram_model(words, force_rebuild=force, verbose=verbosity >= 1)

    if verbosity >= 1:
        print(f"[3/3] Préparation terminée en {time.time() - t0:.1f}s.")
        print()

    # ── Fonction d'évaluation (closure sur model + dictionary) ───────────
    eval_fn = _build_eval_fn(model, dictionary_set, config)

    # ── Dispatch selon le mode ───────────────────────────────────────────
    if args.mode == "run_ga":
        mode_run_ga(config, eval_fn, dictionary_set, verbosity)

    elif args.mode == "run_eda":
        mode_run_eda(config, eval_fn, dictionary_set, verbosity)

    elif args.mode == "exp_crossover":
        dfs, pools = experiment_crossover_comparison(eval_fn, config, verbosity)
        _run_and_save_experiment(
            "exp_crossover", dfs, pools, config, dictionary_set, output_dir, verbosity
        )

    elif args.mode == "exp_etalon":
        dfs, pools = experiment_etalon_comparison(eval_fn, config, verbosity)
        _run_and_save_experiment(
            "exp_etalon", dfs, pools, config, dictionary_set, output_dir, verbosity
        )

    elif args.mode == "exp_losers":
        dfs, pools = experiment_losers_comparison(eval_fn, config, verbosity)
        _run_and_save_experiment(
            "exp_losers", dfs, pools, config, dictionary_set, output_dir, verbosity
        )

    elif args.mode == "exp_ga_vs_eda":
        dfs, pools = experiment_ga_vs_eda(eval_fn, config, verbosity)
        _run_and_save_experiment(
            "exp_ga_vs_eda", dfs, pools, config, dictionary_set, output_dir, verbosity
        )

    elif args.mode == "all":
        all_pools_for_annex: dict = {}

        for exp_name, exp_fn in [
            ("exp_crossover", experiment_crossover_comparison),
            ("exp_etalon", experiment_etalon_comparison),
            ("exp_losers", experiment_losers_comparison),
            ("exp_ga_vs_eda", experiment_ga_vs_eda),
        ]:
            if verbosity >= 1:
                print(f"\n{'─' * 60}")
            dfs, pools = exp_fn(eval_fn, config, verbosity)
            _run_and_save_experiment(
                exp_name, dfs, pools, config, dictionary_set, output_dir, verbosity
            )
            # Merge pools: keep best fitness per word across all experiments
            for algo, pool in pools.items():
                if algo not in all_pools_for_annex:
                    all_pools_for_annex[algo] = {}
                for w, f in pool.items():
                    if w not in all_pools_for_annex[algo] or f < all_pools_for_annex[algo][w]:
                        all_pools_for_annex[algo][w] = f

        # Annexe globale (tous algorithmes confondus)
        if verbosity >= 1:
            print("\n── Génération de l'annexe globale ──────────────────────────")
        global_annex = save_best_words_annex(
            all_pools_for_annex, dictionary_set, output_dir / "annexe"
        )
        if verbosity >= 1:
            for algo, p in global_annex.items():
                print(f"  [{algo}] → {p} ({sum(1 for _ in open(p)) - 1} mots)")

    if verbosity >= 1:
        print(f"\nTerminé. Résultats dans : {output_dir.resolve()}")


if __name__ == "__main__":
    main()
